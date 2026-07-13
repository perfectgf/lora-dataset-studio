"""Checkpoint management: app-wide trash (nothing destroyed directly),
selective delete, run cleanup, cloud staging purge, source-side save cap."""
import json
import os

import pytest

from app.config import LOCAL_USER


@pytest.fixture()
def ds(app, client):
    return client.post('/api/dataset/create',
                       json={'name': 'Mgmt', 'trigger_word': 'mgmt'}).get_json()['id']


def test_trash_roundtrip(app, tmp_path):
    from app.services import trash
    with app.app_context():
        f = tmp_path / 'x.safetensors'
        f.write_bytes(b'12345')
        moved = trash.send_to_trash(str(f), context='test ctx/©')
        assert not f.exists() and os.path.isfile(moved)
        assert trash.trash_size() >= 5
        res = trash.empty_trash()
        assert res['removed'] >= 1 and res['freed_bytes'] >= 5
        assert trash.trash_size() == 0


def test_delete_checkpoint_goes_to_trash_and_is_whitelisted(app, ds, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import trash
    with app.app_context():
        run_dir = tmp_path / 'lora_mgmt'
        run_dir.mkdir()
        ck = run_dir / 'lora_mgmt_000001000.safetensors'
        ck.write_bytes(b'W')
        monkeypatch.setattr(lt, '_run_dir', lambda *a, **k: str(run_dir))
        with pytest.raises(ValueError, match='unknown'):
            lt.delete_checkpoint(LOCAL_USER, ds, '../../evil.safetensors')
        assert lt.delete_checkpoint(LOCAL_USER, ds, ck.name) == ck.name
        assert not ck.exists()                       # moved, not destroyed
        assert trash.trash_size() >= 1


def test_delete_checkpoint_refused_while_training(app, ds, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    with app.app_context():
        monkeypatch.setattr(lt, '_local_training_active_for', lambda d: True)
        with pytest.raises(ValueError, match='training right now'):
            lt.delete_checkpoint(LOCAL_USER, ds, 'x.safetensors')


def test_cleanup_keeps_only_the_keep_set(app, ds, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    with app.app_context():
        run_dir = tmp_path / 'lora_mgmt'
        run_dir.mkdir()
        names = [f'lora_mgmt_{i:09d}.safetensors' for i in (1000, 2000, 3000)]
        for n in names:
            (run_dir / n).write_bytes(b'W')
        (run_dir / 'lora_mgmt.safetensors').write_bytes(b'F')   # final
        monkeypatch.setattr(lt, '_run_dir', lambda *a, **k: str(run_dir))
        res = lt.cleanup_checkpoints(LOCAL_USER, ds,
                                     keep=['lora_mgmt.safetensors', names[2]])
        assert res['removed'] == 2
        left = sorted(os.listdir(run_dir))
        assert left == sorted(['lora_mgmt.safetensors', names[2]])


def test_purge_finished_runs_spares_active_and_pod_kept(app, ds, tmp_path):
    from app.extensions import db
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    with app.app_context():
        def mk(status, sub):
            d = tmp_path / sub
            d.mkdir()
            (d / 'ck.safetensors').write_bytes(b'W' * 10)
            r = CloudTrainingRun(dataset_id=ds, status=status, job_name='j',
                                 vast_label=f'lds-{sub}', staging_dir=str(d),
                                 checkpoint_local_path=str(d / 'ck.safetensors'))
            db.session.add(r)
            db.session.commit()
            return d
        done_dir = mk('done', 'r_done')
        active_dir = mk('training', 'r_active')
        kept_dir = mk('error_pod_kept', 'r_kept')
        res = ct.purge_finished_runs()
        assert res['purged_runs'] == 1 and res['freed_bytes'] >= 10
        assert not done_dir.exists()                 # trashed
        assert active_dir.exists() and kept_dir.exists()   # spared


def test_delete_cloud_checkpoint_terminal_only(app, ds, tmp_path):
    from app.extensions import db
    from app.models import CloudTrainingRun
    from app.services import cloud_training as ct
    with app.app_context():
        d = tmp_path / 'r10'
        d.mkdir()
        ck = d / 'lds10_x_000001000.safetensors'
        ck.write_bytes(b'W')
        run = CloudTrainingRun(dataset_id=ds, status='done', job_name='j',
                               vast_label='lds-10', staging_dir=str(d),
                               checkpoint_local_path=str(ck))
        active = CloudTrainingRun(dataset_id=ds, status='training', job_name='j',
                                  vast_label='lds-11', staging_dir=str(d))
        db.session.add_all([run, active])
        db.session.commit()
        with pytest.raises(ValueError, match='active'):
            ct.delete_cloud_checkpoint(ds, active.id, ck.name)
        assert ct.delete_cloud_checkpoint(ds, run.id, ck.name) == ck.name
        assert not ck.exists()
        assert run.checkpoint_local_path is None      # ready flag cleared


def test_max_step_saves_setting_reaches_job_config(app, ds):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as fds
    with app.app_context():
        dso = fds.get_dataset(LOCAL_USER, ds)
        cfg_job = lt.build_job_config(dso, '/tmp/x', steps=1000,
                                      training_folder='/pod/out')
        save = cfg_job['config']['process'][0]['save']
        assert save['max_step_saves_to_keep'] == 4            # new default
        lt.update_train_settings(LOCAL_USER, ds, {'max_step_saves': 2})
        dso = fds.get_dataset(LOCAL_USER, ds)
        cfg_job = lt.build_job_config(dso, '/tmp/x', steps=1000,
                                      training_folder='/pod/out')
        assert cfg_job['config']['process'][0]['save']['max_step_saves_to_keep'] == 2
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, ds, {'max_step_saves': 99})


def test_trash_routes(app, client, tmp_path):
    from app.services import trash
    with app.app_context():
        f = tmp_path / 'y.bin'
        f.write_bytes(b'123')
        trash.send_to_trash(str(f), context='route-test')
    assert client.get('/api/trash').get_json()['size_bytes'] >= 3
    res = client.post('/api/trash/empty').get_json()
    assert res['ok'] is True and res['freed_bytes'] >= 3
    assert client.get('/api/trash').get_json()['size_bytes'] == 0
