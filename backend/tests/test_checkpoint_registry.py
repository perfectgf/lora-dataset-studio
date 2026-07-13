"""Provenance registry: dataset fingerprint -> human version (v1/v2/...),
manifest diffs, and the version suffix on deployed checkpoint names."""
import json
import os

import pytest

from app.config import LOCAL_USER


@pytest.fixture()
def ds_with_images(app, client):
    """A dataset with 2 kept images (files on disk so the content proxy works)."""
    from app import config as cfg
    from app.extensions import db
    from app.models import FaceDatasetImage
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Prov', 'trigger_word': 'prov'}).get_json()['id']
    with app.app_context():
        img_dir = cfg.dataset_images_root() / str(ds_id)
        img_dir.mkdir(parents=True, exist_ok=True)
        ids = []
        for i in range(2):
            (img_dir / f'im{i}.png').write_bytes(b'PNG' + bytes([i]))
            row = FaceDatasetImage(dataset_id=ds_id, filename=f'im{i}.png',
                                   status='keep', caption=f'a photo {i}')
            db.session.add(row)
            db.session.commit()
            ids.append(row.id)
    return ds_id, ids


def test_register_allocates_versions_by_fingerprint(app, ds_with_images):
    from app.services import checkpoint_registry as reg
    ds_id, ids = ds_with_images
    with app.app_context():
        r1 = reg.register_launch(LOCAL_USER, ds_id, 'zimage', 'local', steps=1000)
        assert r1.version == 1
        # unchanged dataset -> same version on a re-launch
        r2 = reg.register_launch(LOCAL_USER, ds_id, 'zimage', 'local', steps=1200)
        assert r2.version == 1
        # edit a caption -> new fingerprint -> v2
        from app.models import FaceDatasetImage
        img = FaceDatasetImage.query.get(ids[0])
        img.caption = 'edited caption'
        from app.extensions import db
        db.session.commit()
        r3 = reg.register_launch(LOCAL_USER, ds_id, 'zimage', 'local', steps=1000)
        assert r3.version == 2
        # families version independently
        rk = reg.register_launch(LOCAL_USER, ds_id, 'krea', 'cloud', cloud_run_id=7)
        assert rk.version == 1 and rk.source == 'cloud' and rk.cloud_run_id == 7


def test_manifest_diff_counts_changes():
    from app.services import checkpoint_registry as reg
    old = [[1, 'aaaa', 'f1'], [2, 'bbbb', 'f2'], [3, 'cccc', 'f3']]
    new = [[2, 'bbbb', 'f2X'], [3, 'CHANGED', 'f3'], [4, 'dddd', 'f4']]
    d = reg.manifest_diff(old, new)
    assert d == {'images_added': 1, 'images_removed': 1,
                 'captions_changed': 1, 'images_edited': 1}


def test_dataset_state_flags_drift(app, ds_with_images):
    from app.services import checkpoint_registry as reg
    from app.extensions import db
    from app.models import FaceDatasetImage
    ds_id, ids = ds_with_images
    with app.app_context():
        assert reg.dataset_state(LOCAL_USER, ds_id, 'zimage')['registered'] is False
        reg.register_launch(LOCAL_USER, ds_id, 'zimage', 'local')
        st = reg.dataset_state(LOCAL_USER, ds_id, 'zimage')
        assert st['registered'] is True and st['version'] == 1
        assert st['changed'] is False and st['diff'] is None
        # remove an image -> drift with a readable diff
        FaceDatasetImage.query.get(ids[1]).status = 'reject'
        db.session.commit()
        st = reg.dataset_state(LOCAL_USER, ds_id, 'zimage')
        assert st['changed'] is True
        assert st['diff']['images_removed'] == 1


def test_import_suffixes_deployed_name_with_version(app, ds_with_images, tmp_path, monkeypatch):
    """A local import resolves the file's run via the registry and suffixes
    _v<N>; without any registry row the name stays EXACTLY as before."""
    from app import config as cfg
    from app.services import lora_training as lt
    from app.services import checkpoint_registry as reg
    ds_id, _ = ds_with_images
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(tmp_path / 'comfy')},
                         'aitoolkit': {'dir': str(tmp_path / 'aitk')}})
        run_dir = tmp_path / 'run'
        run_dir.mkdir()
        ck = run_dir / 'lora_prov_000001000.safetensors'
        ck.write_bytes(b'W')
        monkeypatch.setattr(lt, '_run_dir', lambda *a, **k: str(run_dir))
        # no registry row yet -> unchanged historical name
        dest = lt.import_checkpoint(LOCAL_USER, ds_id, ck.name)
        assert os.path.basename(dest) == ck.name
        # registered BEFORE the file was written -> _v1 suffix
        reg.register_launch(LOCAL_USER, ds_id, 'zimage', 'local')
        os.utime(ck)                        # file newer than the record
        dest = lt.import_checkpoint(LOCAL_USER, ds_id, ck.name)
        assert os.path.basename(dest) == 'lora_prov_000001000_v1.safetensors'
        # both deployed files are listed (the _v suffix passes the boundary)
        names = [c['filename'] for c in lt.list_imported_checkpoints(LOCAL_USER, ds_id)]
        assert any(n.endswith('_v1.safetensors') for n in names)


def test_ensure_baseline_retrofits_pretrained_datasets(app, ds_with_images):
    """Deployed-project rule: a dataset trained BEFORE the registry existed
    (evidence: checkpoints/cloud runs, zero records) gets a retroactive v1
    baseline — versioning must cover the past, not only future runs."""
    from app.services import checkpoint_registry as reg
    ds_id, _ = ds_with_images
    with app.app_context():
        # no training evidence -> nothing registered
        reg.ensure_baseline(LOCAL_USER, ds_id, 'zimage', had_training=False)
        assert reg.latest_record(ds_id, 'zimage') is None
        # evidence -> v1 baseline, source 'legacy'; idempotent
        reg.ensure_baseline(LOCAL_USER, ds_id, 'zimage', had_training=True)
        rec = reg.latest_record(ds_id, 'zimage')
        assert rec.version == 1 and rec.source == 'legacy'
        reg.ensure_baseline(LOCAL_USER, ds_id, 'zimage', had_training=True)
        assert reg.latest_record(ds_id, 'zimage').id == rec.id   # no duplicate


def test_cloud_launch_registers_and_stamps_version(app, client, monkeypatch, ds_with_images):
    from app.services import cloud_training as ct
    ds_id, _ = ds_with_images
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    monkeypatch.setattr(ct, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_reconcile_before_launch', lambda a: None)
    monkeypatch.setattr(ct.lt, 'export_dataset_to_aitoolkit',
                        lambda uid, did, masked=True, dest_dir=None: dest_dir)
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds: 1000)
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **kw: None)
    with app.app_context():
        ct.launch_cloud_training(LOCAL_USER, ds_id)
        run = ct.get_active_run()
        assert json.loads(run.train_params)['version'] == 1
        assert ct._run_payload(run)['version'] == 1
