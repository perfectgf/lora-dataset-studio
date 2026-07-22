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


# --- LR factor: scale the continuation's learning rate (½ polish / ⅒ finish) -----

def test_resolve_resume_lr_rules():
    from app.services import lora_training as lt
    assert lt.resolve_resume_lr({}, None) is None            # keep current
    assert lt.resolve_resume_lr({}, 1) is None
    # factors scale the family-fixed 1e-4 default of a non-adaptive run
    assert lt.resolve_resume_lr({}, 0.5) == pytest.approx(5e-5)
    assert lt.resolve_resume_lr({}, 0.1) == pytest.approx(1e-5)
    # an explicit stored learning_rate is the base a further continue scales
    assert lt.resolve_resume_lr({'learning_rate': 5e-5}, 0.5) == pytest.approx(2.5e-5)
    with pytest.raises(ValueError, match='lr_factor must be one of'):
        lt.resolve_resume_lr({}, 0.25)
    # Prodigy adapts the LR itself → the factor is refused, not silently swallowed
    with pytest.raises(ValueError, match='Prodigy'):
        lt.resolve_resume_lr({'optimizer': 'prodigy'}, 0.5)


def test_validate_resume_overrides_lr_factor():
    from app.services import lora_training as lt
    assert lt.validate_resume_overrides({'lr_factor': 0.5}) == {'lr_factor': 0.5}
    assert lt.validate_resume_overrides({'lr_factor': 0.1}) == {'lr_factor': 0.1}
    # keep-current (1) is a no-op: dropped so nothing redundant persists
    assert lt.validate_resume_overrides({'lr_factor': 1}) == {}
    with pytest.raises(ValueError, match='lr_factor must be one of'):
        lt.validate_resume_overrides({'lr_factor': 0.75})


def test_continue_lr_factor_half_reduces_effective_lr(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'LR half', 'lrhalf')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})

        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500,
                             overrides={'lr_factor': 0.5})

        eff = lt.effective_train_settings(svc.get_dataset(LOCAL_USER, ds.id))
    # 1e-4 run → 5e-5, exposed as the effective LR the Continue dialog reads back.
    assert eff['learning_rate'] == pytest.approx(5e-5)


def test_continue_keep_current_lr_leaves_default(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'LR keep', 'lrkeep')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training', lambda *a, **k: {'started': True})

        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)   # no lr_factor

        after = svc.get_dataset(LOCAL_USER, ds.id)
        assert lt.effective_train_settings(after)['learning_rate'] == pytest.approx(1e-4)
        # no learning_rate persisted → the dataset stays byte-identical to a default
        assert 'learning_rate' not in (lt._train_settings(after))


def test_continue_lr_factor_emitted_in_job_config(app, tmp_path, monkeypatch):
    """The reduced LR reaches the ACTUAL ai-toolkit job config (and thus the run's
    provenance snapshot), not just the stored settings."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _stub_launch(monkeypatch, tmp_path, app)
    with app.app_context():
        ds, run_dir, trig = _seed_run(lt, svc, LOCAL_USER, 'LR cfg', 'lrcfg', [1000])
        res = lt.continue_training(LOCAL_USER, ds.id, extra_steps=200,
                                   overrides={'lr_factor': 0.1})
        with open(res['config_path'], encoding='utf-8') as fh:
            cfg = json.load(fh)
        assert cfg['config']['process'][0]['train']['lr'] == pytest.approx(1e-5)
        # provenance/⎘ Share-config snapshot records the effective LR too
        assert lt.launch_settings_snapshot(
            svc.get_dataset(LOCAL_USER, ds.id))['lr'] == pytest.approx(1e-5)


def test_continue_lr_factor_refused_on_prodigy_no_side_effect(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'LR prodigy', 'lrprodigy')
        ds.train_settings = json.dumps({'optimizer': 'prodigy'})
        svc.db.session.commit()
        launched = []
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        monkeypatch.setattr(lt, 'list_checkpoints',
                            lambda *a, **k: [{'step': 1000, 'filename': 'b.safetensors'}])
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **k: launched.append(k) or {'started': True})

        with pytest.raises(ValueError, match='Prodigy'):
            lt.continue_training(LOCAL_USER, ds.id, overrides={'lr_factor': 0.5})
        # refused BEFORE launch, and no learning_rate leaked into the settings
        assert launched == []
        assert 'learning_rate' not in lt._train_settings(
            svc.get_dataset(LOCAL_USER, ds.id))


def test_cloud_continue_lr_factor_folds_learning_rate_into_snapshot(
        ct, app, seeded_dataset, monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        ct.continue_cloud_run('local', src.id, extra_steps=500,
                              overrides={'lr_factor': 0.5})
    merged = json.loads(captured['train_settings_snapshot'])
    # a default (1e-4) cloud run continues at 5e-5, carried in the per-run snapshot
    assert merged['learning_rate'] == pytest.approx(5e-5)


def test_cloud_continue_lr_factor_refused_on_prodigy(ct, app, seeded_dataset,
                                                     monkeypatch, tmp_path):
    staging = tmp_path / 'run_src'
    staging.mkdir()
    with app.app_context():
        src = _seed_done_run(ct, seeded_dataset, staging,
                             train_settings_snapshot=json.dumps({'optimizer': 'prodigy'}))
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='Prodigy'):
            ct.continue_cloud_run('local', src.id, overrides={'lr_factor': 0.5})
        assert launched == []


# --- Lane choice: continue a LOCAL run's checkpoint IN THE CLOUD ---------------
# The mirror of continue_cloud_run. Same pod-side seam (resume_ckpt_path on a
# fresh pod); the file comes from the ai-toolkit run dir instead of a cloud run's
# staging. launch_cloud_training is stubbed — no pod is ever rented in tests.

def _local_run_for_cloud(app, tmp_path, monkeypatch, steps=(500, 1000)):
    """A real local run dir with saves — the source a cloud continuation seeds from."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    return _seed_run(lt, svc, LOCAL_USER, 'CloudFromLocal', 'cloudfromlocal', steps)


def test_local_checkpoint_can_be_continued_in_the_cloud(ct, app, monkeypatch, tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, run_dir, trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(dataset_id=dataset_id, **kw), {'ok': True})[1])
        res = ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=200,
                                             from_step=500)
    assert res['resumed_from'] == 500 and res['target_steps'] == 700
    assert captured['steps'] == 700 and captured['resume_step'] == 500
    # the LOCAL file is what gets seeded onto the fresh pod
    assert captured['resume_ckpt_path'] == os.path.join(
        run_dir, f'lora_{trig}_000000500.safetensors')
    # and unlike the local lane, nothing on disk is archived or re-seeded
    assert sorted(os.listdir(run_dir)) == [f'lora_{trig}_000000500.safetensors',
                                           f'lora_{trig}_000001000.safetensors']


def test_local_to_cloud_continue_defaults_to_the_latest_save(ct, app, monkeypatch,
                                                             tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        res = ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=500)
    assert res['resumed_from'] == 1000 and captured['resume_step'] == 1000


def test_local_to_cloud_continue_rejects_a_step_that_is_not_a_save(ct, app,
                                                                  monkeypatch, tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training', lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='no local checkpoint at step 777'):
            ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, from_step=777)
        assert launched == []


def test_local_to_cloud_continue_without_any_local_save_is_refused(ct, app,
                                                                   monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Empty lane', 'emptylane')
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training', lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='no local checkpoint to continue from'):
            ct.continue_local_run_in_cloud(LOCAL_USER, ds.id)
        assert launched == []


def test_local_to_cloud_continue_refuses_a_forbidden_override(ct, app, monkeypatch,
                                                              tmp_path):
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        launched = []
        monkeypatch.setattr(ct, 'launch_cloud_training', lambda *a, **k: launched.append(k))
        with pytest.raises(ValueError, match='cannot change when continuing.*alpha'):
            ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, overrides={'alpha': 8})
        assert launched == []


def test_local_to_cloud_continue_keeps_overrides_out_of_the_dataset(ct, app,
                                                                    monkeypatch, tmp_path):
    """A cloud launch freezes its settings in a per-run snapshot — continuing a
    local run in the cloud must NOT persist the tweak on the dataset (that is a
    local-lane behaviour, and it would silently change the next local run)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _run_dir, _trig = _local_run_for_cloud(app, tmp_path, monkeypatch)
        captured = {}
        monkeypatch.setattr(ct, 'launch_cloud_training',
                            lambda user_id, dataset_id, **kw:
                            (captured.update(**kw), {'ok': True})[1])
        ct.continue_local_run_in_cloud(LOCAL_USER, ds.id, extra_steps=500,
                                       overrides={'sample_every': 250,
                                                  'lr_factor': 0.5})
        persisted = lt._train_settings(svc.get_dataset(LOCAL_USER, ds.id))
    merged = json.loads(captured['train_settings_snapshot'])
    assert merged['sample_every'] == 250
    # a default (1e-4) run continues at 5e-5, carried in the run snapshot only
    assert merged['learning_rate'] == pytest.approx(5e-5)
    assert 'sample_every' not in persisted and 'learning_rate' not in persisted
