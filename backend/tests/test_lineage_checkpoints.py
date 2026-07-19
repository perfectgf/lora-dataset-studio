"""◉ Graph checkpoints: the lineage payload carries each run's saves as nodes so
the graph can draw them under the run and anchor a continuation's edge on the
exact checkpoint it resumed. Covers a cloud run's harvested staging saves surfaced
as node.checkpoints (step/final/download_url), a lone run with saves still yielding
a checkpoint-bearing node, the dataset-wide forest the LoRA manager opens, and the
per-checkpoint download endpoints (cloud staging + path-traversal guard)."""
import os

import pytest


def _rec(dataset_id=1, family='zimage', source='local', base_model='',
         variant='turbo', steps=1000, version=1, parent=None, resumed_from=None,
         cloud_run_id=None):
    from app.models import TrainingRunRecord
    from app.extensions import db
    r = TrainingRunRecord(
        dataset_id=dataset_id, family=family, source=source, base_model=base_model,
        variant=variant, steps=steps, version=version, fingerprint='fp', manifest='[]',
        parent_record_id=parent, resumed_from=resumed_from, cloud_run_id=cloud_run_id)
    db.session.add(r)
    db.session.commit()
    return r


def _cloud_run(dataset_id, staging, status='done', steps=1500, final=True):
    """A finished cloud run whose staging holds a few harvested epochs + the
    unsuffixed final save — the shape _run_staging_checkpoints reads."""
    from app.models import CloudTrainingRun
    from app.extensions import db
    import json
    staging.mkdir(parents=True, exist_ok=True)
    for s in (500, 1000, 1500):
        (staging / f'lora_x_{s:09d}.safetensors').write_bytes(b'W')
    if final:
        (staging / 'lora_x.safetensors').write_bytes(b'F')   # unsuffixed final
    run = CloudTrainingRun(dataset_id=dataset_id, status=status, job_name='j',
                           vast_label='lds-1', staging_dir=str(staging),
                           train_params=json.dumps({'steps': steps}),
                           checkpoint_local_path=str(staging / 'lora_x.safetensors'))
    db.session.add(run)
    db.session.commit()
    return run


# --- node.checkpoints (cloud harvested saves) --------------------------------

def test_cloud_node_carries_its_checkpoints(app, tmp_path):
    from app.services import cloud_training as ct
    with app.app_context():
        crun = _cloud_run(1, tmp_path / 'stg')
        rec = _rec(source='cloud', steps=1500, cloud_run_id=crun.id)
        tree = ct.run_lineage(rec.id)
        node = tree['nodes'][0]
        cks = node['checkpoints']
        assert [c['step'] for c in cks] == [500, 1000, 1500, 1500]
        # the unsuffixed final of a done run is flagged final; step nodes are not
        finals = [c for c in cks if c['final']]
        assert len(finals) == 1 and finals[0]['filename'] == 'lora_x.safetensors'
        # every pill is present and downloadable through the cloud endpoint
        assert all(c['present'] for c in cks)
        one = cks[0]
        assert one['download_url'] == (
            f'/api/dataset/1/train/cloud/checkpoint?run_id={crun.id}'
            f"&filename={one['filename']}")


def test_lone_run_with_checkpoints_still_has_a_node(app, tmp_path):
    """A single run (no parent, no children) is `single` — but it still carries
    its checkpoints so the graph draws it (the button now opens on any run that
    saved at least one checkpoint, not only 2+ run lineages)."""
    from app.services import cloud_training as ct
    with app.app_context():
        crun = _cloud_run(1, tmp_path / 'lone')
        rec = _rec(source='cloud', steps=1500, cloud_run_id=crun.id)
        tree = ct.run_lineage(rec.id)
        assert tree['single'] is True
        assert len(tree['nodes'][0]['checkpoints']) == 4


def test_edge_resumed_from_matches_a_parent_checkpoint_step(app, tmp_path):
    """The continuation's resumed_from equals one of the parent's checkpoint
    steps — the frontend anchors the run→run edge on THAT pill. Here the child
    resumed from 1000, a real parent save."""
    from app.services import cloud_training as ct
    with app.app_context():
        pcrun = _cloud_run(1, tmp_path / 'p')
        parent = _rec(source='cloud', steps=1500, cloud_run_id=pcrun.id)
        child = _rec(steps=2000, parent=parent.id, resumed_from=1000)
        tree = ct.run_lineage(child.id)
        pnode = next(n for n in tree['nodes'] if n['record_id'] == parent.id)
        edge = tree['edges'][0]
        assert edge['resumed_from'] == 1000
        assert 1000 in {c['step'] for c in pnode['checkpoints']}


# --- dataset-wide forest (LoRA manager ◉ Graph) ------------------------------

def test_dataset_lineage_gathers_all_runs(app, tmp_path):
    from app.services import cloud_training as ct
    with app.app_context():
        a = _rec(dataset_id=7, steps=1000)
        b = _rec(dataset_id=7, steps=1500, parent=a.id, resumed_from=1000)
        c = _rec(dataset_id=7, steps=800)          # a second, independent tree
        _rec(dataset_id=8, steps=1000)             # another dataset — excluded
        tree = ct.dataset_lineage(7)
        assert tree['root_id'] is None and tree['current_id'] is None
        assert {n['record_id'] for n in tree['nodes']} == {a.id, b.id, c.id}
        assert tree['edges'] == [
            {'parent': a.id, 'child': b.id, 'resumed_from': 1000, 'superseded': False}]


def test_dataset_lineage_scoped_by_family(app):
    from app.services import cloud_training as ct
    with app.app_context():
        z = _rec(dataset_id=9, family='zimage')
        _rec(dataset_id=9, family='sdxl')
        tree = ct.dataset_lineage(9, train_type='zimage')
        assert {n['record_id'] for n in tree['nodes']} == {z.id}


def test_dataset_lineage_empty_is_safe(app):
    from app.services import cloud_training as ct
    with app.app_context():
        tree = ct.dataset_lineage(4242)
        assert tree == {'nodes': [], 'edges': [], 'root_id': None,
                        'current_id': None, 'single': True}


# --- per-checkpoint download endpoint (cloud staging) ------------------------

def test_cloud_checkpoint_download_by_filename(app, client, tmp_path):
    from app.models import CloudTrainingRun
    from app.extensions import db
    with app.app_context():
        d = tmp_path / 'stg2'
        d.mkdir()
        (d / 'lora_x_000000500.safetensors').write_bytes(b'HELLO')
        (d / 'lora_x.safetensors').write_bytes(b'FINAL')
        run = CloudTrainingRun(dataset_id=1, status='done', job_name='j',
                               vast_label='lds-2', staging_dir=str(d),
                               checkpoint_local_path=str(d / 'lora_x.safetensors'))
        db.session.add(run)
        db.session.commit()
        rid = run.id
    # a specific harvested epoch streams from staging
    r = client.get(f'/api/dataset/1/train/cloud/checkpoint?run_id={rid}'
                   '&filename=lora_x_000000500.safetensors')
    assert r.status_code == 200 and r.data == b'HELLO'
    # no filename → the run's final LoRA (historical behaviour)
    r = client.get(f'/api/dataset/1/train/cloud/checkpoint?run_id={rid}')
    assert r.status_code == 200 and r.data == b'FINAL'
    # path traversal / unknown file is refused
    for bad in ('..%2f..%2fsecret', 'nope.safetensors', 'lora_x_000000500.txt'):
        assert client.get(
            f'/api/dataset/1/train/cloud/checkpoint?run_id={rid}&filename={bad}'
        ).status_code == 404
