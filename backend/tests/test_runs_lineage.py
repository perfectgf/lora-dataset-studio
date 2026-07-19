"""Runs-hub genealogy tree: lineage resolution over the persisted parent_record_id
edges. Covers the graph shapes the Runs tree must render — a simple continuation
chain, a superseded branch (resumed below where the parent ended), a legacy root
with no persisted parent — plus the two write paths that STAMP the edge
(local continue_training and cloud continue_cloud_run)."""
import json

import pytest


def _rec(reg, dataset_id=1, family='zimage', source='local', base_model='',
         variant='turbo', steps=1000, version=1, parent=None, resumed_from=None,
         cloud_run_id=None):
    """Insert a TrainingRunRecord directly — the graph logic is independent of
    how a launch produced the row, so tests build lineages explicitly."""
    from app.models import TrainingRunRecord
    from app.extensions import db
    r = TrainingRunRecord(
        dataset_id=dataset_id, family=family, source=source, base_model=base_model,
        variant=variant, steps=steps, version=version, fingerprint='fp', manifest='[]',
        parent_record_id=parent, resumed_from=resumed_from, cloud_run_id=cloud_run_id)
    db.session.add(r)
    db.session.commit()
    return r


# --- resolve_lineage graph shapes ---------------------------------------------

def test_simple_chain_root_first_bfs(app):
    from app.services import checkpoint_registry as reg
    with app.app_context():
        a = _rec(reg, steps=1000)
        b = _rec(reg, steps=1500, parent=a.id, resumed_from=1000)
        c = _rec(reg, steps=2000, parent=b.id, resumed_from=1500)
        # resolving from ANY node returns the full chain, root-first
        for start in (a, b, c):
            order = reg.resolve_lineage(start.id)
            assert [r.id for r in order] == [a.id, b.id, c.id]


def test_branch_includes_siblings(app):
    """Two continuations off the SAME parent — resolving one must surface both
    branches (the tree shows the fork)."""
    from app.services import checkpoint_registry as reg
    with app.app_context():
        root = _rec(reg, steps=1000)
        left = _rec(reg, steps=1500, parent=root.id, resumed_from=1000)
        right = _rec(reg, steps=1200, parent=root.id, resumed_from=500)
        ids = {r.id for r in reg.resolve_lineage(left.id)}
        assert ids == {root.id, left.id, right.id}


def test_cycle_is_survived(app):
    """A malformed self-parent can't loop the climb forever."""
    from app.services import checkpoint_registry as reg
    from app.extensions import db
    with app.app_context():
        a = _rec(reg)
        a.parent_record_id = a.id           # pathological
        db.session.commit()
        assert [r.id for r in reg.resolve_lineage(a.id)] == [a.id]


def test_newest_record_for_matches_the_lane(app):
    from app.services import checkpoint_registry as reg
    with app.app_context():
        _rec(reg, base_model='', variant='turbo', steps=1000)
        newer = _rec(reg, base_model='', variant='turbo', steps=1500)
        _rec(reg, base_model='custom.safetensors', variant='turbo')   # other lane
        _rec(reg, variant='base')                                     # other variant
        got = reg.newest_record_for(1, 'zimage', '', 'turbo')
        assert got.id == newer.id


def test_records_with_children(app):
    from app.services import checkpoint_registry as reg
    with app.app_context():
        a = _rec(reg)
        _rec(reg, parent=a.id, resumed_from=1000)
        lone = _rec(reg)
        assert reg.records_with_children([a.id, lone.id]) == {a.id}
        assert reg.records_with_children([]) == set()


# --- run_lineage payload (nodes + edges + superseded) -------------------------

def test_run_lineage_chain_payload(app):
    from app.services import checkpoint_registry as reg
    from app.services import cloud_training as ct
    with app.app_context():
        a = _rec(reg, steps=1000)
        b = _rec(reg, steps=1500, parent=a.id, resumed_from=1000)
        tree = ct.run_lineage(b.id)
        assert tree['single'] is False
        assert tree['root_id'] == a.id and tree['current_id'] == b.id
        assert {n['record_id'] for n in tree['nodes']} == {a.id, b.id}
        cur = next(n for n in tree['nodes'] if n['record_id'] == b.id)
        assert cur['is_current'] is True and cur['resumed_from'] == 1000
        assert tree['edges'] == [
            {'parent': a.id, 'child': b.id, 'resumed_from': 1000, 'superseded': False}]
        # continued from the parent's LAST step -> no saves set aside
        assert all(n['has_superseded_tail'] is False for n in tree['nodes'])


def test_run_lineage_superseded_branch(app):
    """Resuming from step 500 of a run that reached 1000 sets its 500→1000 saves
    aside — the edge and the parent node are flagged so the UI greys the branch."""
    from app.services import checkpoint_registry as reg
    from app.services import cloud_training as ct
    with app.app_context():
        parent = _rec(reg, steps=1000)
        child = _rec(reg, steps=700, parent=parent.id, resumed_from=500)
        tree = ct.run_lineage(child.id)
        edge = tree['edges'][0]
        assert edge['superseded'] is True and edge['resumed_from'] == 500
        pnode = next(n for n in tree['nodes'] if n['record_id'] == parent.id)
        assert pnode['has_superseded_tail'] is True


def test_run_lineage_legacy_root_origin_unknown(app):
    """A record carrying a resume step but no persisted parent (pre-feature run)
    is an honest root flagged origin_unknown — never an invented edge."""
    from app.services import checkpoint_registry as reg
    from app.services import cloud_training as ct
    with app.app_context():
        orphan = _rec(reg, steps=1500, parent=None, resumed_from=1000)
        tree = ct.run_lineage(orphan.id)
        assert tree['single'] is True and tree['edges'] == []
        assert tree['nodes'][0]['origin_unknown'] is True


def test_run_lineage_unknown_id(app):
    from app.services import cloud_training as ct
    with app.app_context():
        assert ct.run_lineage(999999)['nodes'] == []


# --- all_runs lineage flag ----------------------------------------------------

def test_all_runs_flags_lineage_rows(app):
    from app.services import checkpoint_registry as reg
    from app.services import cloud_training as ct
    with app.app_context():
        a = _rec(reg, steps=1000)
        _rec(reg, steps=1500, parent=a.id, resumed_from=1000)
        lone = _rec(reg, dataset_id=2)
        rows = {r['record_id']: r for r in ct.all_runs(limit=50)['recent']}
        assert rows[a.id]['lineage'] is True          # is a parent
        child_id = next(i for i in rows if rows[i].get('parent_record_id') == a.id)
        assert rows[child_id]['lineage'] is True       # has a parent
        assert rows[lone.id]['lineage'] is False       # neither


# --- write paths stamp the edge ----------------------------------------------

class _FakeProc:
    pid = 4242

    def wait(self):
        return None


def _stub_launch(monkeypatch, tmp_path, app):
    """Reach launch_training (real config + real register_launch) without
    spawning ai-toolkit — the proven seam from test_continue_flexible."""
    import os
    from app import config as cfg
    from app.services import lora_training as lt
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda a, **k: _FakeProc())
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
    (tmp_path / 'exported').mkdir(exist_ok=True)
    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit',
                        lambda u, d, masked=True: str(tmp_path / 'exported'))


def test_local_continue_stamps_parent_and_resume(app, tmp_path, monkeypatch):
    import os
    from app.services import lora_training as lt
    from app.services import checkpoint_registry as reg
    from app.services import face_dataset_service as svc
    from app.models import TrainingRunRecord
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Chain', 'chaintrig')
        ds.train_type = 'zimage'
        ds.train_variant = 'turbo'
        svc.db.session.commit()
        trig = lt._safe_trigger(ds)
        run_dir = lt._run_dir(LOCAL_USER, ds.id, None, 'zimage', 'turbo')
        os.makedirs(run_dir, exist_ok=True)
        for s in (500, 1000):
            with open(os.path.join(run_dir, f'lora_{trig}_{s:09d}.safetensors'), 'wb') as fh:
                fh.write(b'W')
        # the PARENT record of this lane (as the initial launch would have left it)
        parent = reg.register_launch(LOCAL_USER, ds.id, 'zimage', 'local',
                                     base_model='', variant='turbo', steps=1000)
        assert parent.parent_record_id is None       # fresh launch = root

        # a real continuation launch (default resume = latest 1000) stamps the edge
        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        child = TrainingRunRecord.query.order_by(TrainingRunRecord.id.desc()).first()
        assert child.id != parent.id
        assert child.parent_record_id == parent.id
        assert child.resumed_from == 1000
