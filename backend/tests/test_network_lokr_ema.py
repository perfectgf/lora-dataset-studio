"""LoKr network variant + EMA of weights — two community-recipe levers (Krea-2:
LoKr + low rank + EMA 0.99 → likeness by ~step 500). Both are arch-generic in
ai-toolkit (network.type='lokr' → LokrModule on every arch; train.ema_config is a
TrainConfig knob), so they are offered on EVERY family with no whitelist — and the
knob is a real config change, never a silent no-op."""
import json

import pytest

from app.config import LOCAL_USER

FAMILIES = ('zimage', 'krea', 'flux', 'flux2klein', 'sdxl')


def _mk(svc, tt, tmp_path):
    """A dataset of family `tt`. SDXL needs a base checkpoint to build a config;
    an absolute path is treated as opt-in custom weights (bypasses the whitelist)."""
    from app.extensions import db
    ds = svc.create_dataset(LOCAL_USER, tt.upper(), f'trg_{tt}', train_type=tt)
    if tt == 'sdxl':
        ds.train_base_model = str(tmp_path / 'base.safetensors')
        db.session.commit()
    return ds


def _process(lt, ds, folder):
    return lt.build_job_config(ds, str(folder), 1500)['config']['process'][0]


def test_default_no_lokr_no_ema_every_family(app, tmp_path):
    """Untouched dataset → plain LoRA, no ema_config, on all five families: the
    default config is byte-for-byte what it was before these levers existed."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        for tt in FAMILIES:
            p = _process(lt, _mk(svc, tt, tmp_path), folder)
            assert p['network']['type'] == 'lora', tt
            assert 'ema_config' not in p['train'], tt


def test_lokr_emitted_for_every_family(app, tmp_path):
    """network.type flips to 'lokr' for EVERY family (proof there is no family
    whitelist — ai-toolkit builds LokrModule regardless of arch). rank/alpha still
    ride in linear/linear_alpha; lokr_factor is left at the ai-toolkit auto default
    (-1) so it is deliberately NOT emitted."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        for tt in FAMILIES:
            ds = _mk(svc, tt, tmp_path)
            lt.update_train_settings(LOCAL_USER, ds.id, {'network_type': 'lokr'})
            p = _process(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
            assert p['network']['type'] == 'lokr', tt
            assert 'linear' in p['network'] and 'linear_alpha' in p['network'], tt
            assert 'lokr_factor' not in p['network'], tt


def test_ema_emitted_for_every_family(app, tmp_path):
    """train.ema_config carries the exact ai-toolkit keys {use_ema, ema_decay} on
    every family when EMA is turned on."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        for tt in FAMILIES:
            ds = _mk(svc, tt, tmp_path)
            lt.update_train_settings(LOCAL_USER, ds.id, {'ema': 0.99})
            p = _process(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
            assert p['train']['ema_config'] == {'use_ema': True, 'ema_decay': 0.99}, tt


def test_recipe_combo_lokr_lowrank_ema999(app, tmp_path):
    """The full recipe on one dataset: LoKr + a low rank + EMA 0.999 all land
    together in the same config."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        ds = _mk(svc, 'zimage', tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id,
                                 {'network_type': 'lokr', 'rank': 8, 'ema': 0.999})
        p = _process(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
        assert p['network']['type'] == 'lokr'
        assert p['network']['linear'] == 8 and p['network']['linear_alpha'] == 8
        assert p['train']['ema_config'] == {'use_ema': True, 'ema_decay': 0.999}


def test_validation_rejects_and_clears(app, tmp_path):
    """Invalid values are rejected with a reason that names the key; 'lora' / 'off'
    clear each lever back to its default (key removed from the stored settings)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'kt', train_type='krea')
        with pytest.raises(ValueError) as e1:
            lt.update_train_settings(LOCAL_USER, ds.id, {'network_type': 'dora'})
        assert 'network_type' in str(e1.value)
        with pytest.raises(ValueError) as e2:
            lt.update_train_settings(LOCAL_USER, ds.id, {'ema': 0.95})
        assert 'ema' in str(e2.value)
        lt.update_train_settings(LOCAL_USER, ds.id, {'network_type': 'lokr', 'ema': 0.99})
        stored = lt.snapshot_train_settings(LOCAL_USER, ds.id)
        assert stored['network_type'] == 'lokr' and stored['ema'] == 0.99
        lt.update_train_settings(LOCAL_USER, ds.id, {'network_type': 'lora', 'ema': 'off'})
        stored2 = lt.snapshot_train_settings(LOCAL_USER, ds.id)
        assert 'network_type' not in stored2 and 'ema' not in stored2


def test_effective_settings_exposes_choices_and_supported(app):
    """effective_train_settings (what the Advanced panel reads) surfaces the two
    levers, their choice lists, and network_type_supported=True for every family."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'kt', train_type='krea')
        eff = lt.effective_train_settings(ds)
        assert eff['network_type'] is None and eff['ema'] is None          # defaults
        assert eff['network_type_choices'] == ['lora', 'lokr']
        assert eff['ema_choices'] == [0.99, 0.999]
        assert eff['network_type_supported'] is True
        lt.update_train_settings(LOCAL_USER, ds.id, {'network_type': 'lokr', 'ema': 0.99})
        eff2 = lt.effective_train_settings(svc.get_dataset(LOCAL_USER, ds.id))
        assert eff2['network_type'] == 'lokr' and eff2['ema'] == 0.99


def test_launch_snapshot_and_share_carry_lokr_ema(app):
    """The launch snapshot (stamped into the run's provenance and rendered by the
    ⎘ Share config) omits the levers by default and carries them once set — and the
    share renderer knows both keys as first-class rows."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services.run_share import _KNOWN_SETTING_KEYS
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'kt', train_type='krea')
        snap = lt.launch_settings_snapshot(ds, 'krea')
        assert 'network_type' not in snap and 'ema' not in snap
        lt.update_train_settings(LOCAL_USER, ds.id, {'network_type': 'lokr', 'ema': 0.99})
        snap2 = lt.launch_settings_snapshot(svc.get_dataset(LOCAL_USER, ds.id), 'krea')
        assert snap2['network_type'] == 'lokr' and snap2['ema'] == 0.99
        assert 'network_type' in _KNOWN_SETTING_KEYS and 'ema' in _KNOWN_SETTING_KEYS


def test_preset_apply_schema_tolerant_and_version_tolerant(client, app):
    """A preset carrying the new keys applies through the validated path (valid
    lands, invalid reported, unknown ignored); an OLD preset without the keys leaves
    them at their defaults (forward/backward version tolerance)."""
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'P', 'trigger_word': 'pt', 'train_type': 'krea'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'settings': {
        'network_type': 'lokr',      # valid → applied
        'ema': 0.42,                 # invalid value → rejected with reason
        'lokr_alpha_beta': 3,        # unknown key → ignored
    }})
    body = r.get_json()
    assert body['ok'] is True
    assert body['ignored'] == ['lokr_alpha_beta']
    assert [x['key'] for x in body['rejected']] == ['ema']
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'network_type': 'lokr'}
    # an older preset (no network/ema keys) REPLACES → the levers fall back to default
    client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'settings': {'rank': 16}})
    with app.app_context():
        from app.services import lora_training as lt
        from app.services import face_dataset_service as svc
        eff = lt.effective_train_settings(svc.get_dataset('local', ds_id))
        assert eff['network_type'] is None and eff['ema'] is None


def test_cloud_rebuild_carries_lokr_ema(app, tmp_path):
    """The cloud pod rebuilds the job at boot via _run_config_dataset + build_job_config
    (the run's stamped family/variant over a view of the live dataset). The dataset's
    train_settings pass through that view, so LoKr + EMA reach the rented GPU."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services.cloud_training import _run_config_dataset
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        ds = svc.create_dataset(LOCAL_USER, 'K', 'kt', train_type='krea')
        lt.update_train_settings(LOCAL_USER, ds.id, {'network_type': 'lokr', 'ema': 0.99})
        view = _run_config_dataset(svc.get_dataset(LOCAL_USER, ds.id),
                                   {'train_type': 'krea', 'variant': 'base'})
        p = lt.build_job_config(view, str(folder), 1500)['config']['process'][0]
        assert p['network']['type'] == 'lokr'
        assert p['train']['ema_config'] == {'use_ema': True, 'ema_decay': 0.99}
