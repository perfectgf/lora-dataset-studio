"""Flexible continuation: custom step counts, resuming from an EARLIER checkpoint
without destroying the run's later saves, and the safe-subset settings overrides a
continue is allowed to change. Covers both the local (ai-toolkit seed) and the
override-validation path shared with cloud."""
import glob
import json
import os

import pytest


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid

    def wait(self):
        return None


def _configure_aitoolkit(tmp_path, monkeypatch, app):
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


def _stub_launch(monkeypatch, tmp_path, app):
    """Neutralize everything past the seed decision so a continue reaches
    launch_training and writes a real job config without spawning ai-toolkit."""
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda a, **k: _FakeProc())
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
    (tmp_path / 'exported').mkdir(exist_ok=True)
    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit',
                        lambda u, d, masked=True: str(tmp_path / 'exported'))


def _seed_run(lt, svc, LOCAL_USER, name, trig, steps):
    """A zimage/turbo dataset with a real ai-toolkit run dir holding one numbered
    save per step. Returns (dataset, run_dir, trigger)."""
    ds = svc.create_dataset(LOCAL_USER, name, trig)
    ds.train_type = 'zimage'
    ds.train_variant = 'turbo'
    svc.db.session.commit()
    trigger = lt._safe_trigger(ds)
    run_dir = lt._run_dir(LOCAL_USER, ds.id, None, 'zimage', 'turbo')
    os.makedirs(run_dir, exist_ok=True)
    for s in steps:
        p = os.path.join(run_dir, f'lora_{trigger}_{s:09d}.safetensors')
        with open(p, 'wb') as fh:
            fh.write(f'WEIGHTS-{s}'.encode())
    return ds, run_dir, trigger


# --- Custom step count --------------------------------------------------------

def test_continue_custom_extra_steps_targets_last_plus_extra(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Custom steps', 'customsteps')
        launched = {}
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(
            lt, 'list_checkpoints',
            lambda *a, **k: [{'step': 250, 'filename': 'a.safetensors'},
                             {'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda *a, **k: launched.update(k) or {'started': True})

        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=333)

    assert res['resumed_from'] == 1000               # default = latest
    assert res['target_steps'] == 1333
    assert launched['steps'] == 1333                 # last + custom extra


def test_continue_extra_steps_floor_100(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Floor', 'floorextra')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 500, 'filename': 'a.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})

        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=5)   # below the floor

    assert res['target_steps'] == 600                # 500 + max(100, 5)


# --- Resume from an EARLIER checkpoint, non-destructively ----------------------

def test_continue_from_lower_checkpoint_seeds_clean_and_keeps_originals(
        app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'Lower', 'lowertrig',
                                      [500, 1000])
        # baseline: both saves visible to the hub
        assert [c['step'] for c in lt.list_checkpoints(
            LOCAL_USER, ds.id, None, 'zimage', 'turbo')] == [500, 1000]

        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=200, from_step=500)

        assert res['resumed_from'] == 500
        assert res['target_steps'] == 700
        with open(res['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        assert cfg['config']['process'][0]['train']['steps'] == 700

        # Fresh save_root holds ONLY the seeded 500 — ai-toolkit resumes from it,
        # never the over-cooked 1000.
        assert sorted(os.listdir(run_dir)) == [f'lora_{trig}_000000500.safetensors']
        with open(os.path.join(run_dir, f'lora_{trig}_000000500.safetensors'), 'rb') as fh:
            assert fh.read() == b'WEIGHTS-500'         # the exact earlier weights

        # The original run is set aside intact — BOTH saves recoverable, nothing deleted.
        training_folder = os.path.dirname(run_dir)
        superseded = glob.glob(training_folder + '_superseded_*')
        assert len(superseded) == 1
        assert sorted(os.listdir(os.path.join(superseded[0], f'lora_{trig}'))) == [
            f'lora_{trig}_000000500.safetensors',
            f'lora_{trig}_000001000.safetensors']


def test_continue_from_latest_does_not_archive_or_seed(app, tmp_path, monkeypatch):
    """Explicitly picking the latest step is the historical in-place resume — no
    folder is set aside, both saves stay in place."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'Latest', 'latesttrig',
                                      [500, 1000])
        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=200, from_step=1000)

        assert res['resumed_from'] == 1000 and res['target_steps'] == 1200
        assert sorted(os.listdir(run_dir)) == [
            f'lora_{trig}_000000500.safetensors',
            f'lora_{trig}_000001000.safetensors']
        assert glob.glob(os.path.dirname(run_dir) + '_superseded_*') == []


def test_continue_from_missing_step_is_rejected(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'Miss', 'misstrig',
                                      [500, 1000])
        with pytest.raises(ValueError, match='no checkpoint at step 750'):
            lt.continue_training(LOCAL_USER, ds.id, from_step=750)
        # nothing archived on the rejected request
        assert glob.glob(os.path.dirname(run_dir) + '_superseded_*') == []


# --- Safe-subset settings overrides -------------------------------------------

def test_continue_applies_safe_overrides(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Overrides', 'ovrtrig')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})

        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500,
                             overrides={'save_every': 250, 'sample_every': 500})

        eff = lt.effective_train_settings(svc.get_dataset(LOCAL_USER, ds.id))
    assert eff['save_every'] == 250
    assert eff['sample_every'] == 500


def test_continue_rejects_forbidden_override(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Forbidden', 'forbidtrig')
        launched = []
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: launched.append(k) or {'started': True})

        # rank changes the LoRA weight shape → the checkpoint could not load.
        with pytest.raises(ValueError, match='cannot change when continuing.*rank'):
            lt.continue_training(LOCAL_USER, ds.id, overrides={'rank': 32})
        # a forbidden key fails BEFORE any launch and without persisting settings.
        assert launched == []
        assert svc.get_dataset(LOCAL_USER, ds.id).train_settings in (None, '{}')


def test_validate_resume_overrides_value_and_key_rules():
    from app.services import lora_training as lt
    assert lt.validate_resume_overrides(None) == {}
    assert lt.validate_resume_overrides({'save_every': 500}) == {'save_every': 500}
    # multi-line prompt string is normalized to a trimmed list
    assert lt.validate_resume_overrides(
        {'sample_prompts': 'a\n \nb'}) == {'sample_prompts': ['a', 'b']}
    with pytest.raises(ValueError, match='save_every must be one of'):
        lt.validate_resume_overrides({'save_every': 7})
    with pytest.raises(ValueError, match='cannot change when continuing.*optimizer'):
        lt.validate_resume_overrides({'optimizer': 'prodigy'})
    # timestep_type is the deliberate safe exception (two-phase texture recipe):
    # honored when it names a real weighting, refused on anything else.
    assert lt.validate_resume_overrides(
        {'timestep_type': 'shift'}) == {'timestep_type': 'shift'}
    with pytest.raises(ValueError, match='timestep_type must be one of'):
        lt.validate_resume_overrides({'timestep_type': 'lowest-noise'})


# --- Cloud continue: same knobs, seeding an arbitrary checkpoint onto a pod ----

@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(cloud_training, '_reconcile_before_launch', lambda a: None)
    return cloud_training


@pytest.fixture()
def seeded_dataset(app, client):
    return client.post('/api/dataset/create',
                       json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']


def _seed_done_run(ct, dataset_id, staging, steps=1000, **params):
    """A 'done' cloud run whose staging holds two harvested checkpoints (500, 1000)."""
    p = {'steps': steps, 'variant': 'turbo', 'train_type': 'zimage', 'masked': True}
    p.update(params)
    run = ct.CloudTrainingRun(
        dataset_id=dataset_id, status='done', job_name='lds1_x',
        vast_label='lds-1', staging_dir=str(staging), train_params=json.dumps(p))
    ct.db.session.add(run)
    ct.db.session.commit()
    (staging / 'lds1_x_000000500.safetensors').write_bytes(b'w500')
    (staging / 'lds1_x_000001000.safetensors').write_bytes(b'w1000')
    return run


def test_cloud_continue_from_lower_checkpoint_selects_it_and_keeps_staging(
        ct, app, seeded_dataset, monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(dataset_id=dataset_id, **kw), {'ok': True})[1])
        res = ct.continue_cloud_run('local', src.id, extra_steps=300, from_step=500)

    assert res['resumed_from'] == 500 and res['target_steps'] == 800
    assert captured['steps'] == 800
    assert captured['resume_step'] == 500
    assert captured['resume_ckpt_path'] == str(staging / 'lds1_x_000000500.safetensors')
    # the source run's staging is read-only here — nothing moved or deleted
    assert (staging / 'lds1_x_000000500.safetensors').exists()
    assert (staging / 'lds1_x_000001000.safetensors').exists()


def test_cloud_continue_default_is_latest_checkpoint(ct, app, seeded_dataset,
                                                     monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        res = ct.continue_cloud_run('local', src.id, extra_steps=500)
    assert res['resumed_from'] == 1000 and captured['resume_step'] == 1000


def test_cloud_continue_merges_safe_overrides_into_snapshot(ct, app, seeded_dataset,
                                                            monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging,
                             train_settings_snapshot=json.dumps({'rank': 32}))
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        ct.continue_cloud_run('local', src.id, extra_steps=500,
                              overrides={'sample_every': 250})
    merged = json.loads(captured['train_settings_snapshot'])
    assert merged['sample_every'] == 250          # override folded in
    assert merged['rank'] == 32                    # run's own settings preserved


def test_cloud_continue_rejects_forbidden_override(ct, app, seeded_dataset,
                                                   monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='cannot change when continuing.*alpha'):
            ct.continue_cloud_run('local', src.id, overrides={'alpha': 8})
        assert launched == []


def test_cloud_continue_from_missing_step_is_rejected(ct, app, seeded_dataset,
                                                      monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        with pytest.raises(ValueError, match='no harvested checkpoint at step 700'):
            ct.continue_cloud_run('local', src.id, from_step=700)
