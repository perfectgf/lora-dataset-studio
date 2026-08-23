"""LoKr network controls + EMA and the Krea Raw community-recipe fields.

LoKr and EMA are arch-generic in ai-toolkit; the Krea-only fields below are
deliberately scoped so a source-labelled Krea recipe is never a silent no-op.
"""

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
    ride in linear/linear_alpha; the auto factor is omitted, while full-rank is
    explicitly false so a future ai-toolkit default cannot erase those dimensions."""
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
            assert p['network']['lokr_full_rank'] is False, tt
            assert 'lokr_factor' not in p['network'], tt


def test_explicit_lokr_factor_is_emitted_for_every_family(app, tmp_path):
    """A selected factor reaches ai-toolkit rather than merely living in a preset."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        for tt in FAMILIES:
            ds = _mk(svc, tt, tmp_path)
            lt.update_train_settings(LOCAL_USER, ds.id,
                                     {'network_type': 'lokr', 'lokr_factor': 16})
            p = _process(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
            assert p['network']['lokr_factor'] == 16, tt
            assert p['network']['lokr_full_rank'] is False, tt


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


def test_krea_community_recipe_fields_reach_the_real_train_config(app, tmp_path):
    """Balanced, Automagic v2 and Differential Guidance are emitted as the exact
    ai-toolkit fields the Krea Raw recipe announces — no decorative preset keys."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        ds = _mk(svc, 'krea', tmp_path)
        lt.update_train_settings(LOCAL_USER, ds.id, {
            'network_type': 'lokr', 'lokr_factor': 16,
            'rank': 32, 'alpha': 32, 'timestep_type': 'sigmoid',
            'optimizer': 'automagic2', 'learning_rate': 1e-4,
            'content_or_style': 'balanced',
            'do_differential_guidance': True,
            'differential_guidance_scale': 3,
        })
        p = _process(lt, svc.get_dataset(LOCAL_USER, ds.id), folder)
        assert p['network'] == {
            'type': 'lokr', 'linear': 32, 'linear_alpha': 32,
            'lokr_full_rank': False, 'lokr_factor': 16,
        }
        assert p['train']['optimizer'] == 'automagic2'
        assert p['train']['lr'] == 1e-4
        assert p['train']['timestep_type'] == 'sigmoid'
        assert p['train']['content_or_style'] == 'balanced'
        assert p['train']['do_differential_guidance'] is True
        assert p['train']['differential_guidance_scale'] == 3.0


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
        with pytest.raises(ValueError) as e3:
            lt.update_train_settings(LOCAL_USER, ds.id, {'lokr_factor': 12})
        assert 'lokr_factor' in str(e3.value)
        with pytest.raises(ValueError) as e4:
            lt.update_train_settings(LOCAL_USER, ds.id, {'differential_guidance_scale': 11})
        assert 'differential_guidance_scale' in str(e4.value)
        lt.update_train_settings(LOCAL_USER, ds.id, {
            'network_type': 'lokr', 'lokr_factor': 16, 'ema': 0.99,
            'content_or_style': 'balanced', 'do_differential_guidance': True,
            'differential_guidance_scale': 3,
        })
        stored = lt.snapshot_train_settings(LOCAL_USER, ds.id)
        assert stored['network_type'] == 'lokr' and stored['lokr_factor'] == 16
        assert stored['ema'] == 0.99 and stored['content_or_style'] == 'balanced'
        assert stored['do_differential_guidance'] is True
        lt.update_train_settings(LOCAL_USER, ds.id, {
            'network_type': 'lora', 'lokr_factor': 'auto', 'ema': 'off',
            'content_or_style': 'auto', 'do_differential_guidance': False,
            'differential_guidance_scale': 'auto',
        })
        stored2 = lt.snapshot_train_settings(LOCAL_USER, ds.id)
        assert {'network_type', 'lokr_factor', 'ema', 'content_or_style',
                'do_differential_guidance', 'differential_guidance_scale'}.isdisjoint(stored2)


@pytest.mark.parametrize('bad_value', [1, 'true', {}, []])
def test_differential_guidance_requires_a_real_boolean(app, bad_value):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Strict bool', 'strictbool', train_type='krea')
        with pytest.raises(ValueError, match='do_differential_guidance'):
            lt.update_train_settings(LOCAL_USER, ds.id,
                                     {'do_differential_guidance': bad_value})
        assert 'do_differential_guidance' not in lt.snapshot_train_settings(
            LOCAL_USER, ds.id)


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
        assert eff['lokr_factor'] is None and eff['lokr_factor_choices'] == [4, 8, 16, 32]
        assert eff['ema_choices'] == [0.99, 0.999]
        assert eff['network_type_supported'] is True
        assert eff['krea_recipe_supported'] is True
        assert eff['content_or_style'] is None and eff['content_or_style_default'] == 'balanced'
        assert eff['do_differential_guidance'] is False and eff['differential_guidance_scale'] == 3.0
        lt.update_train_settings(LOCAL_USER, ds.id, {
            'network_type': 'lokr', 'lokr_factor': 16, 'ema': 0.99,
            'content_or_style': 'balanced', 'do_differential_guidance': True,
            'differential_guidance_scale': 3,
        })
        eff2 = lt.effective_train_settings(svc.get_dataset(LOCAL_USER, ds.id))
        assert eff2['network_type'] == 'lokr' and eff2['ema'] == 0.99
        assert eff2['lokr_factor'] == 16
        assert eff2['content_or_style'] == 'balanced'
        assert eff2['do_differential_guidance'] is True


def test_launch_snapshot_and_share_carry_lokr_ema_and_krea_recipe(app):
    """The launch snapshot (stamped into the run's provenance and rendered by the
    ⎘ Share config) stamps network/EMA and Krea recipe fields with their effective
    value — including defaults — and the share renderer knows every key as a first-class row.

    These two used to be omitted while they matched the default. That reads as
    "compact" until two runs are compared: an absent `ema` was then
    indistinguishable from a run recorded before the key existed, so the compare
    panel could not say which of the two had EMA on — the one question the EMA
    experiment exists to answer. An explicit 'off' cannot be misread."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.services.run_share import _KNOWN_SETTING_KEYS
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'kt', train_type='krea')
        snap = lt.launch_settings_snapshot(ds, 'krea')
        assert snap['network_type'] == 'lora' and snap['ema'] == 'off'
        assert snap['content_or_style'] == 'balanced'
        assert snap['do_differential_guidance'] is False
        assert snap['differential_guidance_scale'] == 'off'
        lt.update_train_settings(LOCAL_USER, ds.id, {
            'network_type': 'lokr', 'lokr_factor': 16, 'ema': 0.99,
            'content_or_style': 'balanced', 'do_differential_guidance': True,
            'differential_guidance_scale': 3,
        })
        snap2 = lt.launch_settings_snapshot(svc.get_dataset(LOCAL_USER, ds.id), 'krea')
        assert snap2['network_type'] == 'lokr' and snap2['ema'] == 0.99
        assert snap2['lokr_full_rank'] is False and snap2['lokr_factor'] == 16
        assert snap2['content_or_style'] == 'balanced'
        assert snap2['do_differential_guidance'] is True
        assert snap2['differential_guidance_scale'] == 3.0
        assert {'network_type', 'lokr_factor', 'lokr_full_rank', 'ema',
                'content_or_style', 'do_differential_guidance',
                'differential_guidance_scale'} <= _KNOWN_SETTING_KEYS


def test_preset_apply_schema_tolerant_and_version_tolerant(client, app):
    """A preset carrying the new keys applies through the validated path (valid
    lands, invalid reported, unknown ignored); an OLD preset without the keys leaves
    them at their defaults (forward/backward version tolerance)."""
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'P', 'trigger_word': 'pt', 'train_type': 'krea'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'settings': {
        'network_type': 'lokr',      # valid → applied
        'lokr_factor': 16,            # valid → applied
        'ema': 0.42,                 # invalid value → rejected with reason
        'content_or_style': 'balanced',
        'do_differential_guidance': True,
        'differential_guidance_scale': 3,
        'lokr_alpha_beta': 3,        # unknown key → ignored
    }})
    body = r.get_json()
    assert body['ok'] is True
    assert body['ignored'] == ['lokr_alpha_beta']
    assert [x['key'] for x in body['rejected']] == ['ema']
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {
            'network_type': 'lokr', 'lokr_factor': 16,
            'content_or_style': 'balanced', 'do_differential_guidance': True,
            'differential_guidance_scale': 3.0,
        }
    # an older preset (no network/ema keys) REPLACES → the levers fall back to default
    client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'settings': {'rank': 16}})
    with app.app_context():
        from app.services import lora_training as lt
        from app.services import face_dataset_service as svc
        eff = lt.effective_train_settings(svc.get_dataset('local', ds_id))
        assert eff['network_type'] is None and eff['lokr_factor'] is None and eff['ema'] is None


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
        lt.update_train_settings(LOCAL_USER, ds.id, {
            'network_type': 'lokr', 'lokr_factor': 16, 'ema': 0.99,
            'content_or_style': 'balanced', 'do_differential_guidance': True,
            'differential_guidance_scale': 3,
        })
        view = _run_config_dataset(svc.get_dataset(LOCAL_USER, ds.id),
                                   {'train_type': 'krea', 'variant': 'base'})
        p = lt.build_job_config(view, str(folder), 1500)['config']['process'][0]
        assert p['network']['type'] == 'lokr'
        assert p['network']['lokr_factor'] == 16 and p['network']['lokr_full_rank'] is False
        assert p['train']['ema_config'] == {'use_ema': True, 'ema_decay': 0.99}
        assert p['train']['content_or_style'] == 'balanced'
        assert p['train']['do_differential_guidance'] is True
        assert p['train']['differential_guidance_scale'] == 3.0
