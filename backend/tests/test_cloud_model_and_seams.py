"""CloudTrainingRun model + the three cloud seams in lora_training
(export dest_dir / build training_folder / import src_dir). aitoolkit is NOT
configured in these tests: the seams must work without it."""
import io
import os

import pytest
from PIL import Image

CONCEPTLESS = {'name': 'Lola', 'trigger_word': 'lola'}


def _mkds(client):
    return client.post('/api/dataset/create', json=CONCEPTLESS).get_json()['id']


def _png(color=(255, 0, 0)):
    # The brief's literal 1x1 PNG byte string is malformed (PIL raises "broken
    # data stream" on decode) -- use the same generator pattern as the existing
    # suite (test_dataset_service.py._png) instead.
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def test_cloud_run_table_exists(app):
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        run = CloudTrainingRun(dataset_id=1, run_name='r', job_name='j',
                               status='preparing', vast_label='lds-1')
        db.session.add(run)
        db.session.commit()
        got = CloudTrainingRun.query.first()
        assert got.status == 'preparing'
        assert got.created_at is not None


def test_export_with_dest_dir_needs_no_aitoolkit(app, client, tmp_path, monkeypatch):
    ds_id = _mkds(client)
    # seed one kept image + caption via the service layer (real helper names:
    # import_images() is a module-level fn, not a dataset_dir/list_images/set_caption
    # trio -- see face_dataset_service.py: _dataset_dir, import_images,
    # set_image_caption, and FaceDatasetImage.query for listing).
    from app.services import face_dataset_service as fds
    from app.services import lora_training as lt
    from app.models import FaceDatasetImage
    with app.app_context():
        # register through the normal import path
        ids, failed = fds.import_images('local', ds_id, [_png()], crop=False)
        assert failed == 0 and len(ids) == 1
        for im in FaceDatasetImage.query.filter_by(dataset_id=ds_id).all():
            fds.set_image_caption('local', im.id, 'a test caption')
        dest = tmp_path / 'staging' / 'dataset'
        out = lt.export_dataset_to_aitoolkit('local', ds_id, masked=False,
                                             dest_dir=str(dest))
        assert out == str(dest)
        files = os.listdir(out)
        assert any(f.endswith('.png') for f in files)
        assert any(f.endswith('.txt') for f in files)


def test_build_job_config_with_training_folder_override(app, client):
    ds_id = _mkds(client)
    from app.services import face_dataset_service as fds
    from app.services import lora_training as lt
    with app.app_context():
        ds = fds.get_dataset('local', ds_id)
        cfg_ = lt.build_job_config(ds, '/staging/ds', steps=500,
                                   training_folder='__POD__')
        proc = cfg_['config']['process'][0]
        assert proc['training_folder'] == '__POD__'
        assert proc['datasets'][0]['folder_path'] == '/staging/ds'
        assert cfg_['config']['name'].startswith('lora_')
        # zimage default: official HF base — never a local path
        assert proc['model']['name_or_path'] == 'Tongyi-MAI/Z-Image-Turbo'


def test_build_job_config_local_path_unchanged_without_override(app, client, monkeypatch):
    # without training_folder AND without aitoolkit configured, it must still
    # raise the historic RuntimeError (local behavior untouched)
    ds_id = _mkds(client)
    from app.services import face_dataset_service as fds
    from app.services import lora_training as lt
    with app.app_context():
        ds = fds.get_dataset('local', ds_id)
        with pytest.raises(RuntimeError):
            lt.build_job_config(ds, '/staging/ds', steps=500)


def test_import_checkpoint_from_src_dir(app, client, tmp_path, monkeypatch):
    ds_id = _mkds(client)
    from app.services import lora_training as lt
    src = tmp_path / 'staging'
    src.mkdir()
    (src / 'job_000000500.safetensors').write_bytes(b'ckpt')
    loras = tmp_path / 'loras'
    monkeypatch.setattr(lt, '_lora_dest_dir', lambda ds, family=None: str(loras))
    with app.app_context():
        dest = lt.import_checkpoint('local', ds_id, 'job_000000500.safetensors',
                                    src_dir=str(src))
        assert os.path.exists(dest)
        assert dest.startswith(str(loras))


def test_import_checkpoint_final_name_from_src_dir(app, client, tmp_path, monkeypatch):
    """The run's FINAL artifact (lora_<trigger>.safetensors, no step suffix)
    must be importable through src_dir — ai-toolkit produces it on completion."""
    ds_id = _mkds(client)
    from app.services import lora_training as lt
    src = tmp_path / 'staging'
    src.mkdir()
    (src / 'lora_lola.safetensors').write_bytes(b'ckpt')
    loras = tmp_path / 'loras'
    monkeypatch.setattr(lt, '_lora_dest_dir', lambda ds, family=None: str(loras))
    with app.app_context():
        dest = lt.import_checkpoint('local', ds_id, 'lora_lola.safetensors',
                                    src_dir=str(src))
        assert os.path.exists(dest)
        assert dest.startswith(str(loras))


def test_import_checkpoint_normalizes_cloud_job_names(app, client, tmp_path, monkeypatch):
    """Pod checkpoints arrive named `lds<run>_u<user>_<trigger>_<base>[_step]` —
    deployed as-is they are invisible to every `lora_<trigger>_…` matcher
    (Test Studio dropdown + whitelist rejected them: "unusable cloud
    checkpoints", user-reported). Import must rename to the local convention
    and never double the base tag."""
    ds_id = _mkds(client)
    from app.services import lora_training as lt
    src = tmp_path / 'staging'
    src.mkdir()
    (src / 'lds12_ulocal_lola_Krea-2-Raw_000000250.safetensors').write_bytes(b'ckpt')
    (src / 'lds12_ulocal_lola_Krea-2-Raw.safetensors').write_bytes(b'ckpt')
    loras = tmp_path / 'loras'
    monkeypatch.setattr(lt, '_lora_dest_dir', lambda ds, family=None: str(loras))
    with app.app_context():
        dest = lt.import_checkpoint(
            'local', ds_id, 'lds12_ulocal_lola_Krea-2-Raw_000000250.safetensors',
            src_dir=str(src), version=1)
        name = os.path.basename(dest)
        assert name.startswith('lora_lola_000000250')
        assert 'lds12' not in name
        assert name.count('Krea-2-Raw') <= 1          # embedded tag never doubled
        final = lt.import_checkpoint(
            'local', ds_id, 'lds12_ulocal_lola_Krea-2-Raw.safetensors',
            src_dir=str(src), version=1)
        assert os.path.basename(final).startswith('lora_lola')


def test_default_steps_matches_launch_training_logic(app, client):
    from app.services import face_dataset_service as fds
    from app.services import lora_training as lt
    ds_id = _mkds(client)
    with app.app_context():
        ds = fds.get_dataset('local', ds_id)
        n = lt.default_steps(ds)
        assert isinstance(n, int) and n > 0


def test_run_config_dataset_overrides_without_mutating(app, client):
    """The monitor's config view forces a run's stamped family/variant onto the
    dataset it hands build_job_config, WITHOUT touching the real row — the seam
    that keeps two concurrent multi-family cloud runs isolated (incident
    2026-07-14)."""
    from app.services import cloud_training as ct
    from app.services import face_dataset_service as fds
    ds_id = _mkds(client)
    with app.app_context():
        ds = fds.get_dataset('local', ds_id)
        ds.train_type = 'zimage'
        ds.train_variant = 'turbo'
        view = ct._run_config_dataset(ds, {'train_type': 'krea', 'variant': 'base'})
        assert view.train_type == 'krea'                # overridden
        assert view.train_variant == 'base'
        assert view.trigger_word == ds.trigger_word     # every other attr delegates
        assert view.id == ds.id
        assert ds.train_type == 'zimage'                # real row untouched
        assert ds.train_variant == 'turbo'
        # A legacy run that stamped neither falls back to the real dataset as-is.
        assert ct._run_config_dataset(ds, {}) is ds
        # A partial override (family only) still delegates the missing one.
        v2 = ct._run_config_dataset(ds, {'train_type': 'krea'})
        assert v2.train_type == 'krea'
        assert v2.train_variant == 'turbo'
