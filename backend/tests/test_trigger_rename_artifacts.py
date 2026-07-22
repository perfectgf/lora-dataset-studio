"""Renaming a dataset's trigger word must carry its ALREADY PRODUCED artefacts
along. The trigger is the on-disk naming key (`u{user}_{trigger}` run folders,
`lora_{trigger}` deployed files), so before rename_training_artifacts a rename
left every past LoRA, run folder, export and job config under the old name —
orphaned from the dataset that made them, while the next run started from an
empty folder. This is the mirror of purge_training_artifacts (delete), and it
inherits the same guards: exact trigger boundary (never a sibling), and a
no-op on an empty trigger."""
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _ds(train_type='zimage', trigger='Lola69382'):
    return SimpleNamespace(id=1, user_id='local', trigger_word=trigger,
                           train_type=train_type, train_base_model=None,
                           train_variant=None)


@pytest.fixture()
def training(app, tmp_path):
    """ai-toolkit AND ComfyUI both configured, so all four backends the rename
    sweeps (deployed LoRAs, output/, datasets/, config/generated) are live."""
    with app.app_context():
        from app import config as cfg
        aitk, comfy = tmp_path / 'aitoolkit', tmp_path / 'comfyui'
        (aitk / 'config' / 'generated').mkdir(parents=True)
        (aitk / 'output').mkdir(parents=True)
        (aitk / 'datasets').mkdir(parents=True)
        (comfy / 'models' / 'loras' / 'z image').mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitk)},
                         'comfyui': {'base_dir': str(comfy)}})
        from app.services import lora_training as lt
        yield lt, aitk, comfy / 'models' / 'loras' / 'z image'


def _seed(lt, aitk, loras, trigger='Lola69382'):
    """One artefact in each of the four backends, named as the app names them."""
    run = f'ulocal_{trigger}_Z-Image-Turbo'
    (aitk / 'output' / run).mkdir()
    (aitk / 'output' / run / 'ckpt_000002500.safetensors').write_text('w')
    (aitk / 'datasets' / run).mkdir()
    (aitk / 'config' / 'generated' / f'{run}.json').write_text('{}')
    lora = loras / f'lora_{trigger}_Z-Image-Turbo_000002500.safetensors'
    lora.write_text('w')
    return run, lora


def test_rename_moves_every_backend(training):
    lt, aitk, loras = training
    old_run, old_lora = _seed(lt, aitk, loras)
    out = lt.rename_training_artifacts('local', 'Lola69382', 'LolaV2')
    assert out['ok'] and not out['conflicts']
    new_run = 'ulocal_LolaV2_Z-Image-Turbo'
    # run folder, export folder and job config all followed the trigger...
    assert (aitk / 'output' / new_run).is_dir()
    assert (aitk / 'datasets' / new_run).is_dir()
    assert (aitk / 'config' / 'generated' / f'{new_run}.json').is_file()
    # ...with their CONTENTS intact (a rename, never a re-create).
    assert (aitk / 'output' / new_run / 'ckpt_000002500.safetensors').read_text() == 'w'
    # the deployed LoRA keeps its family/step suffix, only the trigger changes
    assert (loras / 'lora_LolaV2_Z-Image-Turbo_000002500.safetensors').is_file()
    assert not (aitk / 'output' / old_run).exists() and not old_lora.exists()
    assert len(out['renamed']) == 4


def test_rename_respects_the_trigger_boundary(training):
    """The guard that matters: a sibling trigger sharing a prefix (Lola vs Lola2)
    must not be dragged along — the same boundary rule the purge uses."""
    lt, aitk, loras = training
    _seed(lt, aitk, loras, trigger='Lola')
    sibling_dir = aitk / 'output' / 'ulocal_Lola2_Z-Image-Turbo'
    sibling_dir.mkdir()
    sibling_lora = loras / 'lora_Lola2_Z-Image-Turbo_000002500.safetensors'
    sibling_lora.write_text('w')

    out = lt.rename_training_artifacts('local', 'Lola', 'Renamed')
    assert out['ok']
    assert sibling_dir.is_dir() and sibling_lora.is_file()   # untouched
    assert (aitk / 'output' / 'ulocal_Renamed_Z-Image-Turbo').is_dir()


def test_rename_aborts_whole_plan_on_a_collision(training):
    """A half-renamed set is worse than none: artefacts split across two triggers
    with no record of the split. One existing destination cancels everything."""
    lt, aitk, loras = training
    _seed(lt, aitk, loras)
    # the target trigger already has a run folder (another dataset uses it)
    (aitk / 'output' / 'ulocal_Taken_Z-Image-Turbo').mkdir()

    out = lt.rename_training_artifacts('local', 'Lola69382', 'Taken')
    assert not out['ok'] and out['conflicts'] and out['renamed'] == []
    # NOTHING moved — not even the artefacts that had no conflict of their own
    assert (aitk / 'output' / 'ulocal_Lola69382_Z-Image-Turbo').is_dir()
    assert (loras / 'lora_Lola69382_Z-Image-Turbo_000002500.safetensors').is_file()
    assert not (aitk / 'config' / 'generated' / 'ulocal_Taken_Z-Image-Turbo.json').exists()


@pytest.mark.parametrize('old,new', [('', 'New'), ('Old', ''), ('Same', 'Same')])
def test_rename_is_a_no_op_without_a_real_move(training, old, new):
    """Empty either side would sweep on `ulocal_` alone (every dataset); an
    unchanged trigger has nothing to move. Both are successful no-ops."""
    lt, aitk, loras = training
    _seed(lt, aitk, loras)
    out = lt.rename_training_artifacts('local', old, new)
    assert out == {'renamed': [], 'conflicts': [], 'ok': True}
    assert (aitk / 'output' / 'ulocal_Lola69382_Z-Image-Turbo').is_dir()


def test_rename_is_idempotent(training):
    lt, aitk, loras = training
    _seed(lt, aitk, loras)
    assert len(lt.rename_training_artifacts('local', 'Lola69382', 'LolaV2')['renamed']) == 4
    again = lt.rename_training_artifacts('local', 'Lola69382', 'LolaV2')
    assert again['ok'] and again['renamed'] == []      # nothing left under the old name


# --- wired into the dataset settings edit -------------------------------------


def test_updating_the_trigger_renames_artifacts_and_rewrites_rows(app, tmp_path):
    """End to end: the edit that changes the trigger moves the files AND repoints
    the rows that named them, so the Test Studio history and the cloud run keep
    pointing at real files instead of dangling on the old name."""
    with app.app_context():
        from app import config as cfg
        from app.extensions import db
        from app.models import FaceDataset, LoraTestImage, CloudTrainingRun
        from app.services import face_dataset_service as fds

        aitk, comfy = tmp_path / 'aitoolkit', tmp_path / 'comfyui'
        for sub in (aitk / 'config' / 'generated', aitk / 'output', aitk / 'datasets'):
            sub.mkdir(parents=True)
        loras = comfy / 'models' / 'loras' / 'z image'
        loras.mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitk)},
                         'comfyui': {'base_dir': str(comfy)}})

        ds = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(ds)
        db.session.commit()
        run = f'u{ds.user_id}_Lola69382_Z-Image-Turbo'
        (aitk / 'output' / run).mkdir()
        old_lora = 'lora_Lola69382_Z-Image-Turbo_000002500.safetensors'
        (loras / old_lora).write_text('w')
        # a Studio image + the pinned best settings both name that LoRA, and the
        # stored value carries the ComfyUI subfolder prefix
        shot = LoraTestImage(dataset_id=ds.id, strength=0.9,
                             checkpoint=f'z image\\{old_lora}')
        db.session.add(shot)
        ds.best_settings = json.dumps({'lora_filename': old_lora, 'strength': 0.9})
        # terminal: an ACTIVE run would (correctly) refuse the rename outright
        cloud = CloudTrainingRun(dataset_id=ds.id, run_name=run, status='done',
                                 staging_dir=str(aitk / 'output' / run / 'stage'))
        db.session.add(cloud)
        db.session.commit()

        res = fds.update_dataset_settings('local', ds.id, trigger_word='LolaV2')
        assert res['ok'] and res['trigger_rename']['ok']

        new_lora = 'lora_LolaV2_Z-Image-Turbo_000002500.safetensors'
        assert (loras / new_lora).is_file() and not (loras / old_lora).exists()
        assert (aitk / 'output' / 'ulocal_LolaV2_Z-Image-Turbo').is_dir()
        # the subfolder prefix survives; only the basename was remapped
        assert shot.checkpoint == f'z image\\{new_lora}'
        assert json.loads(ds.best_settings)['lora_filename'] == new_lora
        assert cloud.run_name == 'ulocal_LolaV2_Z-Image-Turbo'
        assert cloud.staging_dir.endswith(os.path.join('ulocal_LolaV2_Z-Image-Turbo', 'stage'))


def test_trigger_rename_is_refused_while_a_run_is_active(app, tmp_path):
    """The run folder IS ai-toolkit's auto-resume key — moving it under a live job
    would strand the run, so the edit is refused (409) instead."""
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDataset
        from app.services import face_dataset_service as fds

        ds = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(ds)
        db.session.commit()
        with patch('app.services.lora_training.is_local_run_active', return_value=True):
            with pytest.raises(RuntimeError):
                fds.update_dataset_settings('local', ds.id, trigger_word='LolaV2')
        db.session.refresh(ds)
        assert ds.trigger_word == 'Lola69382'        # nothing changed


def test_a_style_renames_through_its_name(app, tmp_path):
    """A style is always-on, so its trigger field is hidden and the token naming its
    files is retained internally — unreachable. Its NAME is the only identity it can
    edit, so for a style the name drives the naming token too; otherwise a style could
    never rename the LoRAs it already produced."""
    with app.app_context():
        from app import config as cfg
        from app.extensions import db
        from app.models import FaceDataset
        from app.services import face_dataset_service as fds

        aitk, comfy = tmp_path / 'aitoolkit', tmp_path / 'comfyui'
        (aitk / 'output').mkdir(parents=True)
        loras = comfy / 'models' / 'loras' / 'z image'
        loras.mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitk)},
                         'comfyui': {'base_dir': str(comfy)}})

        ds = FaceDataset(user_id='local', name='Test', trigger_word='Test',
                         kind='style')
        db.session.add(ds)
        db.session.commit()
        (aitk / 'output' / 'ulocal_Test_Z-Image-Turbo').mkdir()
        (loras / 'lora_Test_Z-Image-Turbo_000002500.safetensors').write_text('w')

        res = fds.update_dataset_settings('local', ds.id, name='Analog film')
        assert res['trigger_rename']['ok'] and res['trigger_rename']['files'] == 2
        # the space is sanitized the same way the training code names files
        assert (aitk / 'output' / 'ulocal_Analog_film_Z-Image-Turbo').is_dir()
        assert (loras / 'lora_Analog_film_Z-Image-Turbo_000002500.safetensors').is_file()
        assert ds.trigger_word == 'Analog_film'   # internal token followed the name


def test_a_character_name_change_leaves_the_trigger_alone(app, tmp_path):
    """The style rule must NOT leak to character/concept datasets: there the trigger
    is a real summon word the user types in prompts, so a display-name edit must
    never move it (and never touch disk)."""
    with app.app_context():
        from app import config as cfg
        from app.extensions import db
        from app.models import FaceDataset
        from app.services import face_dataset_service as fds

        aitk = tmp_path / 'aitoolkit'
        (aitk / 'output').mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitk)}})
        ds = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(ds)
        db.session.commit()
        run = aitk / 'output' / 'ulocal_Lola69382_Z-Image-Turbo'
        run.mkdir()

        res = fds.update_dataset_settings('local', ds.id, name='Analog film')
        assert 'trigger_rename' not in res
        assert ds.trigger_word == 'Lola69382' and run.is_dir()


def test_renaming_only_the_name_touches_nothing_on_disk(app, tmp_path):
    """The dataset NAME is display-only — it never appears in an artefact name, so
    editing it must not move a single file (and must not need the active-run guard)."""
    with app.app_context():
        from app import config as cfg
        from app.extensions import db
        from app.models import FaceDataset
        from app.services import face_dataset_service as fds

        aitk = tmp_path / 'aitoolkit'
        (aitk / 'output').mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitk)}})
        ds = FaceDataset(user_id='local', name='Lola', trigger_word='Lola69382')
        db.session.add(ds)
        db.session.commit()
        run = aitk / 'output' / 'ulocal_Lola69382_Z-Image-Turbo'
        run.mkdir()

        res = fds.update_dataset_settings('local', ds.id, name='Analog film')
        assert 'trigger_rename' not in res
        assert run.is_dir()


def test_renaming_a_style_survives_the_modal_echoing_the_old_trigger(app, tmp_path):
    """The reported bug. A style has no trigger FIELD, so the settings modal sends the
    STORED token back verbatim. Honouring that echo overwrote the token the name had
    just derived, and renaming a style changed its label and nothing else — on disk
    everything kept the old name, so an imported checkpoint still carried it."""
    with app.app_context():
        from app import config as cfg
        from app.extensions import db
        from app.models import FaceDataset
        from app.services import face_dataset_service as fds

        aitk = tmp_path / 'aitoolkit'
        (aitk / 'output').mkdir(parents=True)
        cfg.save_config({'aitoolkit': {'dir': str(aitk)}})
        ds = FaceDataset(user_id='local', name='Test', trigger_word='zsty_29', kind='style')
        db.session.add(ds)
        db.session.commit()
        (aitk / 'output' / 'ulocal_zsty_29_Krea-2-Raw').mkdir()

        # exactly what DatasetSettingsModal sends for a style: the OLD trigger echoed
        res = fds.update_dataset_settings('local', ds.id, name='Telegram test',
                                          trigger_word='zsty_29', kind='style')
        assert ds.trigger_word == 'Telegram_test'      # the name won, not the echo
        assert res['trigger_rename']['ok']
        assert (aitk / 'output' / 'ulocal_Telegram_test_Krea-2-Raw').is_dir()


def test_rename_reaches_the_files_inside_the_run_folder(app, tmp_path):
    """ai-toolkit stamps the trigger at THREE levels. Renaming only the outer folder
    fixed nothing visible: import_checkpoint deploys under the SOURCE FILE's stem, so
    the LoRA still landed in ComfyUI under the old trigger."""
    with app.app_context():
        from app import config as cfg
        from app.services import lora_training as lt

        aitk = tmp_path / 'aitoolkit'
        run = aitk / 'output' / 'ulocal_zsty_29_Krea-2-Raw' / 'lora_zsty_29'
        run.mkdir(parents=True)
        (run / 'lora_zsty_29.safetensors').write_text('w')
        (run / 'lora_zsty_29_000000250.safetensors').write_text('w')
        (run.parent / 'training.log').write_text('log')     # not trigger-named: untouched
        cfg.save_config({'aitoolkit': {'dir': str(aitk)}})

        out = lt.rename_training_artifacts('local', 'zsty_29', 'Telegram_test')
        assert out['ok']
        new_run = aitk / 'output' / 'ulocal_Telegram_test_Krea-2-Raw'
        inner = new_run / 'lora_Telegram_test'
        assert inner.is_dir(), 'the lora_<trigger> subfolder must follow'
        assert (inner / 'lora_Telegram_test.safetensors').is_file()
        assert (inner / 'lora_Telegram_test_000000250.safetensors').is_file()
        assert (new_run / 'training.log').is_file()        # untouched, as it should be
