import json
import os
import time

import pytest


def _configure_aitoolkit(tmp_path, monkeypatch, app):
    """Fake ai-toolkit install: venv python + run.py present, dir configured."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    venv_py = root / 'venv' / 'Scripts' / 'python.exe'
    venv_py.write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid

    def wait(self):
        return None


def test_recommended_steps_clamps(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'T', 't')
        for _ in range(5):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='x.webp'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 1500        # 5*120=600 -> clamp
        for _ in range(35):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='y.webp'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 3500        # 40*120=4800 -> clamp


def test_job_config_masked_fields_only_when_masks_exist(app, tmp_path):
    from app.services import lora_training as lt
    folder = tmp_path / 'ds'
    folder.mkdir()
    assert lt._mask_fields(str(folder)) == {}
    masks = tmp_path / 'ds_masks'
    masks.mkdir()
    (masks / 'a.png').touch()
    assert lt._mask_fields(str(folder)) == {'mask_path': str(masks), 'mask_min_value': 0.1}


def test_enqueue_snapshots_steps_and_not_before(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'q')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        lt.enqueue_training(LOCAL_USER, ds.id, steps=2000, not_before='2999-01-01T00:00')
        q = lt.get_train_queue()
        assert q[0]['steps'] == 2000 and q[0]['not_before'].startswith('2999')
        assert lt._due_index(q) is None                   # scheduled in the future -> not due


def test_step_cap_floor_500(app, tmp_path, monkeypatch):
    """launch_training(steps=200) floors to 500; the written job config reflects it."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    _configure_aitoolkit(tmp_path, monkeypatch, app)

    captured = {}

    def fake_popen(args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return _FakeProc()

    monkeypatch.setattr(lt.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Floor', 'floortrig')
        for i in range(12):
            svc.db.session.add(lt.FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                   filename=f'x{i}.webp', caption='a caption here'))
        svc.db.session.commit()

        def fake_export(user_id, dataset_id, masked=True):
            folder = tmp_path / 'exported'
            folder.mkdir(exist_ok=True)
            return str(folder)

        monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit', fake_export)

        result = lt.launch_training(LOCAL_USER, ds.id, steps=200, masked=False)
        assert result['steps'] == 500

        with open(result['config_path'], encoding='utf-8') as fh:
            config = json.load(fh)
        train_cfg = config['config']['process'][0]['train']
        assert train_cfg['steps'] == 500
        ds0 = config['config']['process'][0]['datasets'][0]
        assert 'mask_path' not in ds0


def test_masked_training_zero_written_disables_masks(app, tmp_path, monkeypatch):
    """generate_person_masks returns a dict with written=0 -> masks dir removed,
    masked training disabled (adaptation for the dict-shaped contract)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        from app import config as cfg
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Mask0', 'masktrig0')
        img_dir = svc._dataset_dir(ds.id)
        for i in range(3):
            fn = f'k{i}.png'
            from PIL import Image
            Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
            svc.db.session.add(lt.FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        svc.db.session.commit()

        monkeypatch.setattr(lt, 'generate_person_masks',
                            lambda paths, out_dir: {'ok': True, 'written': 0, 'results': {}})

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        masks_dir = lt._masks_dir(out)
        assert not os.path.isdir(masks_dir)


def test_masked_training_written_nonzero_keeps_masks(app, tmp_path, monkeypatch):
    """generate_person_masks returns written>0 -> masks dir survives, masked training stays on."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        from app import config as cfg
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Mask1', 'masktrig1')
        img_dir = svc._dataset_dir(ds.id)
        for i in range(3):
            fn = f'k{i}.png'
            from PIL import Image
            Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
            svc.db.session.add(lt.FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        svc.db.session.commit()

        def fake_masks(paths, out_dir):
            os.makedirs(out_dir, exist_ok=True)
            for p in paths:
                base = os.path.splitext(os.path.basename(p))[0]
                open(os.path.join(out_dir, base + '.png'), 'wb').close()
            return {'ok': True, 'written': len(paths), 'results': {p: 'ok' for p in paths}}

        monkeypatch.setattr(lt, 'generate_person_masks', fake_masks)

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        masks_dir = lt._masks_dir(out)
        assert os.path.isdir(masks_dir)
        assert lt._mask_fields(out) == {'mask_path': masks_dir, 'mask_min_value': 0.1}


def test_launch_training_unconfigured_aitoolkit_raises_runtime_error(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoAI', 'noaitrig')
        with pytest.raises(RuntimeError):
            lt.launch_training(LOCAL_USER, ds.id)


def test_build_job_config_masked_false_no_mask_path(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'NoMask', 'nomasktrig')
        folder = tmp_path / 'plain'
        folder.mkdir()
        config = lt.build_job_config(ds, str(folder), steps=1500)
        ds0 = config['config']['process'][0]['datasets'][0]
        assert 'mask_path' not in ds0


def test_continue_training_refuses_while_in_progress(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Busy', 'busytrig')
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=3600)
        with pytest.raises(ValueError):
            lt.continue_training(LOCAL_USER, ds.id)


def test_process_training_queue_rearms_ttl_while_pid_alive(app, monkeypatch):
    """A training run longer than the 4h TTL must not have its flags expire
    mid-run: each poll while the pid is still alive re-arms them."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=1)
        queue_manager._set_system_state('training_pid', 4242, ttl_seconds=1)
        queue_manager._set_system_state('training_dataset_id', 7, ttl_seconds=1)
        queue_manager._set_system_state('training_target_step', 1500, ttl_seconds=1)
        monkeypatch.setattr(lt, '_pid_alive', lambda pid: True)

        assert lt.process_training_queue() is None  # still running -> no action taken

        time.sleep(1.2)  # past the ORIGINAL (pre-rearm) TTL
        assert queue_manager._get_system_state('training_in_progress', False) is True
        assert queue_manager._get_system_state('training_pid', None) == 4242
        assert queue_manager._get_system_state('training_dataset_id', None) == 7
        assert queue_manager._get_system_state('training_target_step', None) == 1500


def test_import_list_delete_checkpoint_roundtrip_filesystem_scan(app, tmp_path):
    """import_checkpoint copies into the ComfyUI loras dir (ownership stripped:
    no lora_ownership call); list_imported_checkpoints finds it via a plain
    filesystem scan filtered by trigger boundary (not the old ownership-filtered
    list_test_checkpoints call); delete_imported_checkpoint removes it, confined
    to the family's deploy folder."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg

    with app.app_context():
        aitoolkit_dir = tmp_path / 'aitoolkit'
        comfy_dir = tmp_path / 'comfy'
        cfg.save_config({'aitoolkit': {'dir': str(aitoolkit_dir)},
                         'comfyui': {'base_dir': str(comfy_dir)}})
        ds = svc.create_dataset(LOCAL_USER, 'Import', 'ImportTrig')

        # Fake ai-toolkit run dir with one numbered checkpoint (list_checkpoints reads this).
        run_dir = lt._output_dir() / f'u{ds.user_id}_ImportTrig' / 'lora_ImportTrig'
        run_dir.mkdir(parents=True)
        ck_name = 'lora_ImportTrig_000001500.safetensors'
        (run_dir / ck_name).write_bytes(b'fake-weights')

        assert [c['filename'] for c in lt.list_checkpoints(LOCAL_USER, ds.id)] == [ck_name]
        assert lt.list_imported_checkpoints(LOCAL_USER, ds.id) == []  # not deployed yet

        dest = lt.import_checkpoint(LOCAL_USER, ds.id, ck_name)
        assert os.path.isfile(dest)
        assert 'z image' in dest  # zimage is the default family

        imported = lt.list_imported_checkpoints(LOCAL_USER, ds.id)
        assert len(imported) == 1
        assert imported[0]['filename'] == os.path.join('z image', ck_name)
        assert imported[0]['label']  # format_trained_lora_label produced something non-empty

        removed_name = lt.delete_imported_checkpoint(LOCAL_USER, ds.id, imported[0]['filename'])
        assert removed_name == ck_name
        assert not os.path.isfile(dest)
        assert lt.list_imported_checkpoints(LOCAL_USER, ds.id) == []
