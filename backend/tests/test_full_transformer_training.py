"""Focused contract tests for the Krea 2 dense-training MVP."""
import json
from app.extensions import db

import pytest
from sqlalchemy import text


def _dataset(app, *, train_type='krea'):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset(
            LOCAL_USER, f'Dense {train_type}', f'dense_{train_type}',
            train_type=train_type)
        return ds.id


def _valid_training_capabilities(monkeypatch):
    monkeypatch.setattr(
        'app.routes.training.capabilities.probe',
        lambda: {'aitoolkit': {'valid': True}, 'cloud_training': True})


def test_training_mode_defaults_to_lora_on_new_rows(app):
    from app.models import FaceDataset
    from app.extensions import db

    with app.app_context():
        ds = FaceDataset(user_id='local', name='Legacy default',
                         trigger_word='legacy_default')
        db.session.add(ds)
        db.session.commit()
        assert ds.training_mode == 'lora'
        assert db.session.execute(text(
            'SELECT training_mode FROM face_dataset WHERE id=:id'),
            {'id': ds.id}).scalar_one() == 'lora'


def test_training_mode_additive_migration_backfills_legacy_rows(app):
    """A populated pre-feature table gains a non-null LoRA default idempotently."""
    from app import _apply_additive_migrations
    from app.extensions import db

    with app.app_context():
        db.session.execute(text('DROP TABLE face_dataset'))
        db.session.execute(text(
            'CREATE TABLE face_dataset ('
            'id INTEGER PRIMARY KEY, user_id TEXT, name TEXT, trigger_word TEXT)'))
        db.session.execute(text(
            "INSERT INTO face_dataset (id, user_id, name, trigger_word) "
            "VALUES (1, 'local', 'Legacy', 'legacy')"))
        db.session.commit()

        _apply_additive_migrations()
        _apply_additive_migrations()

        columns = {row[1]: row for row in db.session.execute(
            text('PRAGMA table_info(face_dataset)'))}
        assert columns['training_mode'][3] == 1  # NOT NULL
        assert columns['training_mode'][4] == "'lora'"
        assert db.session.execute(text(
            'SELECT training_mode FROM face_dataset WHERE id=1')).scalar_one() == 'lora'
        db.session.execute(text(
            "INSERT INTO face_dataset (id, user_id, name, trigger_word) "
            "VALUES (2, 'local', 'Also legacy', 'also_legacy')"))
        db.session.commit()
        assert db.session.execute(text(
            'SELECT training_mode FROM face_dataset WHERE id=2')).scalar_one() == 'lora'


def test_settings_persists_and_base_info_serializes_training_mode(
        app, client, monkeypatch):
    from app.models import FaceDataset

    _valid_training_capabilities(monkeypatch)
    dataset_id = _dataset(app)
    response = client.post(
        f'/api/dataset/{dataset_id}/train/settings',
        json={'training_mode': 'full_transformer'})
    assert response.status_code == 200
    assert response.get_json()['training_mode'] == 'full_transformer'

    with app.app_context():
        assert db.session.get(FaceDataset, dataset_id).training_mode == 'full_transformer'

    info = client.get(f'/api/dataset/{dataset_id}/train/base-info')
    assert info.status_code == 200
    assert info.get_json()['training_mode'] == 'full_transformer'

    invalid = client.post(
        f'/api/dataset/{dataset_id}/train/settings',
        json={'training_mode': 'full_model'})
    assert invalid.status_code == 400
    assert "'lora' or 'full_transformer'" in invalid.get_json()['error']


def test_settings_atomically_validates_and_persists_dense_selection(
        app, client, monkeypatch):
    from app.extensions import db
    from app.models import FaceDataset

    _valid_training_capabilities(monkeypatch)
    dataset_id = _dataset(app, train_type='zimage')
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.train_base_model = 'persisted-custom.safetensors'
        ds.train_variant = 'turbo'
        ds.train_slider = json.dumps({
            'enabled': True, 'positive': 'strong', 'negative': 'weak'})
        db.session.commit()
    response = client.post(
        f'/api/dataset/{dataset_id}/train/settings',
        json={'training_mode': 'full_transformer', 'train_type': 'krea',
              'base_model': '', 'variant': 'base',
              'disable_slider_for_full_transformer': True})
    assert response.status_code == 200
    assert response.get_json()['slider']['enabled'] is False
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        assert (ds.training_mode, ds.train_type, ds.train_base_model,
                ds.train_variant) == ('full_transformer', 'krea', None, 'base')
        assert json.loads(ds.train_slider) == {
            'positive': 'strong', 'negative': 'weak'}

        before = (ds.training_mode, ds.train_type, ds.train_base_model,
                  ds.train_variant, ds.train_settings, ds.train_slider)

    # The complete candidate is rejected before any column changes. Turbo is no
    # longer one of those candidates (see the Turbo/custom-base tests below);
    # a base that is not an absolute path still is, because the krea2 loader
    # would ignore it and train something other than what the panel shows.
    response = client.post(
        f'/api/dataset/{dataset_id}/train/settings',
        json={'training_mode': 'full_transformer', 'train_type': 'krea',
              'base_model': 'relative-name.safetensors', 'variant': 'base'})
    assert response.status_code == 400
    assert 'full path' in response.get_json()['error']
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        assert (ds.training_mode, ds.train_type, ds.train_base_model,
                ds.train_variant, ds.train_settings, ds.train_slider) == before


def test_dense_slider_disable_validation_failure_changes_nothing(
        app, client, monkeypatch):
    from app.extensions import db
    from app.models import FaceDataset

    _valid_training_capabilities(monkeypatch)
    dataset_id = _dataset(app)
    original_slider = json.dumps({'enabled': True, 'positive': 'up'})
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.train_variant = 'turbo'
        ds.train_slider = original_slider
        db.session.commit()
        before = (ds.training_mode, ds.train_type, ds.train_base_model,
                  ds.train_variant, ds.train_slider)

    response = client.post(
        f'/api/dataset/{dataset_id}/train/settings',
        json={'training_mode': 'full_transformer', 'train_type': 'zimage',
              'base_model': '', 'variant': 'turbo',
              'disable_slider_for_full_transformer': True})
    assert response.status_code == 400
    assert 'only for Krea 2' in response.get_json()['error']
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        assert (ds.training_mode, ds.train_type, ds.train_base_model,
                ds.train_variant, ds.train_slider) == before


def test_dense_slider_disable_rolls_back_failed_commit(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import FaceDataset
    from app.services import lora_training as lt

    dataset_id = _dataset(app, train_type='zimage')
    original_slider = json.dumps({'enabled': True, 'positive': 'up'})
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.train_base_model = 'old-zimage.safetensors'
        ds.train_variant = 'turbo'
        ds.train_slider = original_slider
        db.session.commit()

        monkeypatch.setattr(
            db.session, 'commit',
            lambda: (_ for _ in ()).throw(RuntimeError('forced commit failure')))
        with pytest.raises(RuntimeError, match='forced commit failure'):
            lt.update_train_settings(LOCAL_USER, dataset_id, {
                'training_mode': 'full_transformer', 'train_type': 'krea',
                'base_model': '', 'variant': 'base',
                'disable_slider_for_full_transformer': True,
            })

        db.session.expire_all()
        restored = db.session.get(FaceDataset, dataset_id)
        assert restored.training_mode == 'lora'
        assert restored.train_type == 'zimage'
        assert restored.train_base_model == 'old-zimage.safetensors'
        assert restored.train_variant == 'turbo'
        assert restored.train_slider == original_slider


def test_local_launch_blocks_persisted_full_and_forwards_explicit_lora_override(
        app, client, monkeypatch):
    from app.extensions import db
    from app.models import FaceDataset

    _valid_training_capabilities(monkeypatch)
    dataset_id = _dataset(app)
    called = []

    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.training_mode = 'full_transformer'
        db.session.commit()

    monkeypatch.setattr(
        'app.services.lora_training.launch_training',
        lambda *args, **kwargs: called.append((args, kwargs)) or {'pid': 1})

    # No field in this legacy-shaped payload: the persisted dense mode is still
    # authoritative and must be rejected before the launch seam is reached.
    response = client.post(
        f'/api/dataset/{dataset_id}/train',
        json={'train_type': 'krea', 'variant': 'base'})
    assert response.status_code == 400
    assert 'cloud-only' in response.get_json()['error']
    assert called == []

    # An explicit switch back to LoRA is forwarded so the service can persist it
    # before build_job_config reads the dataset row.
    response = client.post(
        f'/api/dataset/{dataset_id}/train',
        json={'training_mode': 'lora', 'train_type': 'krea',
              'variant': 'base'})
    assert response.status_code == 200
    assert called[0][1]['training_mode'] == 'lora'


def test_full_transformer_krea_config_is_dense_and_conservative(app, tmp_path):
    from app.models import FaceDataset
    from app.services import lora_training as lt

    dataset_id = _dataset(app)
    images = tmp_path / 'images'
    images.mkdir()
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.training_mode = 'full_transformer'
        ds.train_variant = 'base'
        # Poison every hidden LoRA cadence/resolution lever.  Dense training has
        # its own explicit 26-GB-checkpoint recipe and must ignore these values.
        ds.train_settings = json.dumps({
            'rank': 64, 'optimizer': 'prodigy', 'learning_rate': 1,
            'resolution': '768', 'save_every': 1000, 'sample_every': 1000,
            'quantize': True, 'quantize_te': True, 'low_vram': True,
        })
        cfg = lt.build_job_config(
            ds, str(images), steps=123,
            training_folder=str(tmp_path / 'cloud-run'))

    assert cfg['config']['name'].startswith('Krea')
    process = cfg['config']['process'][0]
    assert 'network' not in process
    assert process['model'] == {
        'arch': 'krea2',
        'name_or_path': 'krea/Krea-2-Raw',
        'quantize': False,
        'low_vram': False,
        'quantize_te': False,
        'model_kwargs': {'vae_path': 'Qwen/Qwen-Image-2512'},
    }
    assert process['datasets'][0]['cache_latents_to_disk'] is True
    assert process['datasets'][0]['cache_text_embeddings'] is True
    assert process['datasets'][0]['resolution'] == [1024]
    assert process['train'] == {
        'batch_size': 1,
        'steps': 123,
        'gradient_accumulation': 1,
        'train_unet': True,
        'train_text_encoder': False,
        'unload_text_encoder': True,
        'gradient_checkpointing': True,
        'noise_scheduler': 'flowmatch',
        'timestep_type': 'linear',
        'optimizer': 'adafactor',
        'lr': 1e-6,
        'dtype': 'bf16',
    }
    assert process['save']['dtype'] == 'bf16'
    assert process['save']['save_every'] == 250
    assert process['save']['max_step_saves_to_keep'] == 1
    assert process['sample']['sample_every'] == 250


def test_full_transformer_snapshot_matches_emitted_dense_recipe(app):
    from app.models import FaceDataset
    from app.services import lora_training as lt

    dataset_id = _dataset(app)
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.training_mode = 'full_transformer'
        ds.train_variant = 'base'
        ds.train_settings = json.dumps({
            'rank': 64, 'alpha': 32, 'network_type': 'lokr',
            'optimizer': 'prodigy', 'learning_rate': 1,
            'resolution': '768', 'save_every': 1000, 'sample_every': 1000,
            'quantize': True, 'quantize_te': True, 'low_vram': True,
        })
        snapshot = lt.launch_settings_snapshot(ds, masked=False)

    assert snapshot['training_mode'] == 'full_transformer'
    assert snapshot['artifact_kind'] == 'full_transformer'
    assert snapshot['effective_base'] == 'krea/Krea-2-Raw'
    assert snapshot['vae_path'] == 'Qwen/Qwen-Image-2512'
    assert snapshot['resolution'] == [1024]
    assert snapshot['save_every'] == 250
    assert snapshot['max_step_saves'] == 1
    assert snapshot['sample_every'] == 250
    assert snapshot['optimizer'] == 'adafactor'
    assert snapshot['lr'] == 1e-6
    assert snapshot['batch_size'] == snapshot['grad_accum'] == 1
    assert snapshot['dtype'] == snapshot['save_dtype'] == 'bf16'
    assert snapshot['timestep_type'] == 'linear'
    assert snapshot['quantize'] is False
    assert snapshot['quantize_te'] is False
    assert snapshot['low_vram'] is False
    assert snapshot['masked'] is False
    for lora_only in ('rank', 'alpha', 'network_type', 'conv', 'conv_alpha',
                      'ema', 'lokr_factor', 'lokr_full_rank'):
        assert lora_only not in snapshot


def test_full_transformer_rejects_non_krea_and_slider(app, tmp_path):
    """The two refusals that SURVIVED. Turbo and a custom base were scope
    decisions and are gone — what they were protecting (a run trained on a base
    it does not name) is now prevented by resolving the base from the selection;
    see test_dense_recipe_and_quantized_base.py, which asserts the emitted
    `model.name_or_path` per recipe rather than the absence of an exception."""
    from app.models import FaceDataset
    from app.services import lora_training as lt

    dataset_id = _dataset(app)
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.training_mode = 'full_transformer'
        ds.train_type = 'zimage'
        with pytest.raises(ValueError, match='only for Krea 2'):
            lt.build_job_config(ds, '/dataset', training_folder='/cloud')

        ds.train_type = 'krea'
        ds.train_variant = 'turbo'
        assert lt.build_job_config(
            ds, '/dataset', training_folder='/cloud',
        )['config']['process'][0]['model']['name_or_path'] == 'krea/Krea-2-Turbo'

        ds.train_variant = 'base'
        ds.train_slider = json.dumps({'enabled': True})
        with pytest.raises(ValueError, match='Slider LoRA'):
            lt.build_job_config(ds, '/dataset', training_folder='/cloud')


def test_legacy_and_explicit_lora_configs_are_identical(app, tmp_path):
    """The new discriminator must not alter one byte of the established Krea LoRA."""
    from app.models import FaceDataset
    from app.services import lora_training as lt

    dataset_id = _dataset(app)
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.train_variant = 'base'
        ds.training_mode = 'lora'
        explicit = lt.build_job_config(
            ds, '/dataset', steps=1500, training_folder='/cloud')
        ds.training_mode = None  # shape of a pre-migration/test-double dataset
        legacy = lt.build_job_config(
            ds, '/dataset', steps=1500, training_folder='/cloud')

    assert legacy == explicit
    process = explicit['config']['process'][0]
    assert explicit['config']['name'].startswith('lora_')
    assert process['network']['type'] == 'lora'
    assert process['model']['name_or_path'] == 'krea/Krea-2-Raw'
    assert process['model']['quantize'] is True


def test_preflight_route_forwards_and_echoes_training_mode(
        app, client, monkeypatch):
    _valid_training_capabilities(monkeypatch)
    dataset_id = _dataset(app)
    seen = {}

    def fake_preflight(user_id, requested_dataset_id, **kwargs):
        seen.update(kwargs)
        return {'blockers': [], 'warnings': [], 'checks': [], 'verdict': 'ready',
                'training_mode': kwargs['training_mode']}

    monkeypatch.setattr(
        'app.services.lora_training.training_preflight', fake_preflight)
    response = client.get(
        f'/api/dataset/{dataset_id}/train/preflight'
        '?lane=cloud&train_type=krea&variant=base&base_model='
        '&training_mode=full_transformer')
    assert response.status_code == 200
    assert seen['training_mode'] == 'full_transformer'
    assert seen['base_model'] == ''
    assert response.get_json()['training_mode'] == 'full_transformer'


def test_preflight_uses_exact_selected_base_and_variant(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.models import FaceDataset
    from app.services import lora_training as lt

    dataset_id = _dataset(app)
    monkeypatch.setattr(
        'app.services.cloud_training.full_transformer_token_preflight',
        lambda **_kw: {'ok': True, 'configured': True, 'namespace': 'tester'})
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.training_mode = 'full_transformer'
        ds.train_variant = 'turbo'
        ds.train_base_model = 'persisted-custom.safetensors'
        db.session.commit()

        report = lt.training_preflight(
            LOCAL_USER, dataset_id, train_type='krea', variant='base',
            base_model='', training_mode='full_transformer', lane='cloud')
        dense = next(c for c in report['checks'] if c['id'] == 'training_mode')
        assert dense['status'] == 'ok'
        assert not any('custom base models are not supported' in blocker
                       for blocker in report['blockers'])


@pytest.mark.parametrize(
    ('token_status', 'expected_status'),
    [
        ({'ok': False, 'configured': False,
          'error': 'HF_CLOUD_TOKEN is required in Settings'}, 'fail'),
        ({'ok': False, 'configured': True,
          'error': ('HF_CLOUD_TOKEN requires repository write access; '
                    'read-only tokens cannot be used')}, 'fail'),
        ({'ok': True, 'configured': True, 'namespace': 'tester'}, 'ok'),
        ({'ok': True, 'configured': True, 'namespace': 'tester',
          'severity': 'warning',
          'warning': 'Global write access for tester is accepted with a warning.'}, 'warn'),
    ],
)
def test_dense_preflight_route_reports_dedicated_cloud_token(
        app, client, monkeypatch, token_status, expected_status):
    _valid_training_capabilities(monkeypatch)
    dataset_id = _dataset(app)
    monkeypatch.setattr(
        'app.services.cloud_training.full_transformer_token_preflight',
        lambda **_kw: dict(token_status))

    response = client.get(
        f'/api/dataset/{dataset_id}/train/preflight'
        '?lane=cloud&train_type=krea&variant=base&base_model='
        '&training_mode=full_transformer')
    assert response.status_code == 200
    payload = response.get_json()
    token_check = next(
        check for check in payload['checks']
        if check['id'] == 'hf_cloud_token')
    assert token_check['status'] == expected_status
    assert token_check['scope'] == 'cloud'
    assert payload['hf_cloud_token_status'] == token_status
    if expected_status == 'fail':
        assert token_check['bypassable'] is False
        assert token_check['target'] == 'gf-training'
        assert token_status['error'] in payload['blockers']
        assert payload['verdict'] == 'blocked'
    else:
        expected_warning = token_status.get('warning')
        if expected_warning:
            assert token_check['detail'] == expected_warning
        else:
            assert 'tester' in token_check['detail']
        assert not any('HF_CLOUD_TOKEN' in blocker
                       for blocker in payload['blockers'])


def test_local_continuation_routes_force_source_lora_mode(
        app, client, monkeypatch):
    from app.extensions import db
    from app.models import FaceDataset

    _valid_training_capabilities(monkeypatch)
    dataset_id = _dataset(app)
    with app.app_context():
        ds = db.session.get(FaceDataset, dataset_id)
        ds.training_mode = 'full_transformer'
        db.session.commit()

    local_seen = {}
    cloud_seen = {}
    monkeypatch.setattr(
        'app.services.lora_training.continue_training',
        lambda *args, **kwargs: local_seen.update(kwargs) or {'started': True})
    monkeypatch.setattr(
        'app.services.cloud_training.continue_local_run_in_cloud',
        lambda *args, **kwargs: cloud_seen.update(kwargs) or {'run_id': 9})

    local = client.post(
        f'/api/dataset/{dataset_id}/train/continue',
        json={'training_mode': 'full_transformer', 'extra_steps': 500})
    cloud = client.post(
        f'/api/dataset/{dataset_id}/train/cloud/continue-local',
        json={'training_mode': 'full_transformer', 'extra_steps': 500})
    assert local.status_code == cloud.status_code == 200
    assert local_seen['training_mode'] == 'lora'
    assert cloud_seen['training_mode'] == 'lora'
