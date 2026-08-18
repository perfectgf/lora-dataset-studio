"""Staged ComfyUI input copies are given back when the job is over.

Every local engine copies its source image into ComfyUI's `input/` folder under a
unique name (`utils/comfy_fs.stage_input_copy`). Nothing ever deleted one, so a
✨ Improve batch of 50 left 50 full-resolution duplicates behind, every run,
forever. Measured on a three-month-old install: 3 896 orphans, 0.67 GB — and the
same count is what ComfyUI enumerates for LoadImage on every prompt validation.

The contract these tests hold:
  * a job that ENDS drops what it staged — on success, on failure, on cancel;
  * the boot sweep clears what precise deletion could never reach (rows queued
    before this shipped, a process killed mid-job) and NEVER touches a copy a
    live job may still be waiting on, nor a file the user put there themselves.
"""
import json
import os
import time

import pytest
from PIL import Image

from app.utils import comfy_fs


def _touch(path, age_seconds=0):
    path.write_bytes(b'x')
    if age_seconds:
        past = time.time() - age_seconds
        os.utime(path, (past, past))
    return path


def _image(path):
    Image.new('RGB', (32, 24), (30, 120, 220)).save(path, 'PNG')
    return path


# --- the per-job half -------------------------------------------------------

@pytest.mark.parametrize('failed', [False, True])
def test_finished_job_drops_its_staged_inputs(app, tmp_path, monkeypatch, failed):
    """Success and failure are the SAME case here: the copy is dead either way.
    Special-casing success is exactly how a failing batch would still fill the
    disk while looking clean."""
    from app import job_queue
    from app.models import ImageGenerationQueue

    comfy_input = tmp_path / 'comfy' / 'input'
    comfy_input.mkdir(parents=True)
    monkeypatch.setattr('app.config.comfyui_dir',
                        lambda kind: str(comfy_input) if kind == 'input' else None)

    src = _touch(comfy_input / 'edit_source_abc12345_face.png')
    ref = _touch(comfy_input / 'edit_ref1_abc12345_ref.png')
    kept = _touch(comfy_input / 'my_own_upload.png')

    with app.app_context():
        job = ImageGenerationQueue(
            job_id='job-1', user_id='local', status='completed',
            workflow_data='{}',
            job_metadata=json.dumps({'model_name': 'klein_edit_dataset',
                                     'staged_inputs': [src.name, ref.name]}))
        job_queue._dispatch_completion(job, None if failed else 'out.png', failed)

    assert not src.exists(), 'the staged source copy outlived its job'
    assert not ref.exists(), 'the staged extra-reference copy outlived its job'
    assert kept.exists(), 'cleanup must only remove what the job itself staged'


def test_job_without_staged_inputs_is_untouched(app, tmp_path, monkeypatch):
    """Rows queued before this shipped carry no `staged_inputs`. They must
    complete exactly as before, not raise on a missing key."""
    from app import job_queue
    from app.models import ImageGenerationQueue

    comfy_input = tmp_path / 'comfy' / 'input'
    comfy_input.mkdir(parents=True)
    monkeypatch.setattr('app.config.comfyui_dir', lambda kind: str(comfy_input))
    legacy = _touch(comfy_input / 'edit_source_old00000_face.png')

    with app.app_context():
        job = ImageGenerationQueue(job_id='job-2', user_id='local', status='completed',
                                   workflow_data='{}',
                                   job_metadata=json.dumps({'model_name': 'klein_edit_dataset'}))
        job_queue._dispatch_completion(job, 'out.png', False)

    assert legacy.exists()


def test_a_missing_staged_file_never_breaks_completion(app, tmp_path, monkeypatch):
    """A duplicate completion, or a user who emptied the folder, must not turn a
    finished generation into a crash on the way out."""
    from app import job_queue
    from app.models import ImageGenerationQueue

    comfy_input = tmp_path / 'comfy' / 'input'
    comfy_input.mkdir(parents=True)
    monkeypatch.setattr('app.config.comfyui_dir', lambda kind: str(comfy_input))

    with app.app_context():
        job = ImageGenerationQueue(job_id='job-3', user_id='local', status='completed',
                                   workflow_data='{}',
                                   job_metadata=json.dumps(
                                       {'staged_inputs': ['edit_source_gone_face.png']}))
        job_queue._dispatch_completion(job, 'out.png', False)   # must not raise


# --- the enqueue side records what it staged --------------------------------

def test_klein_enqueue_records_every_name_it_staged(app, tmp_path, monkeypatch):
    """The list is the whole mechanism: a name that is staged but not recorded is
    a file nothing will ever delete."""
    from app.services import klein_edit_helper as keh

    comfy_input = tmp_path / 'comfy' / 'input'
    comfy_input.mkdir(parents=True)
    source = _image(tmp_path / 'face.png')
    extra = _image(tmp_path / 'angle.png')

    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'resolve_klein_unet', lambda selected=None: 'u.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_vae', lambda: 'v.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_text_encoder', lambda: 't.safetensors')
    monkeypatch.setattr(keh, '_consistency_lora', lambda: ('c.safetensors', None))
    monkeypatch.setattr(keh, '_lora_abs', lambda name: None)
    monkeypatch.setattr(keh.comfy_fs, 'ensure_input_usable', lambda d: str(comfy_input))
    monkeypatch.setattr(keh, '_comfy_input_dir', lambda: str(comfy_input))

    captured = {}
    monkeypatch.setattr(keh.queue_manager, 'add_job',
                        lambda **kw: captured.update(kw) or kw.get('job_id'))

    with app.app_context():
        keh.enqueue_klein_edit(user_id='local', source_filename='face.png',
                               edit_prompt='improve', source_path=str(source),
                               extra_ref_paths=[str(extra)])

    staged = captured['metadata']['staged_inputs']
    on_disk = sorted(p.name for p in comfy_input.iterdir())
    assert sorted(staged) == on_disk, (
        f'staged {on_disk} but only recorded {sorted(staged)} for deletion')


# --- the boot-sweep half ----------------------------------------------------

def test_boot_sweep_clears_old_orphans_but_spares_recent_and_foreign_files(tmp_path):
    """Fence 1 (name) and fence 2 (age). The input folder belongs to ComfyUI and
    holds images the USER dropped there: erasing one to reclaim disk would be far
    worse than the disk."""
    d = tmp_path / 'input'
    d.mkdir()
    old = _touch(d / 'edit_source_0a1b2c3d_a.png', age_seconds=72 * 3600)
    old_ref = _touch(d / 'edit_ref1_0a1b2c3d_a.png', age_seconds=72 * 3600)
    old_krea = _touch(d / 'krea_source_4e5f6a7b_a.png', age_seconds=72 * 3600)
    recent = _touch(d / 'edit_source_8c9d0e1f_a.png', age_seconds=60)
    old_user_file = _touch(d / 'holiday.png', age_seconds=365 * 24 * 3600)

    removed = comfy_fs.prune_staged_inputs(str(d))

    assert removed == 3
    assert not old.exists() and not old_ref.exists() and not old_krea.exists()
    assert recent.exists(), 'a fresh copy may still belong to a queued job'
    assert old_user_file.exists(), "the user's own input images are not ours to delete"


@pytest.mark.parametrize('name', [
    'wmklein_crop_0a1b2c3d.png',     # crop lane
    'wmklein_frame_0a1b2c3d.png',    # masked lane — the full photo…
    'wmklein_mask_0a1b2c3d.png',     # …and the mask beside it
])
def test_every_klein_staged_input_is_collectable(tmp_path, name):
    """Both lanes unlink their own staged copies in a `finally`. This is the
    backstop for the case that skips it — the app or ComfyUI dying between the
    stage and the job — and a name the sweeper does not recognise leaks into the
    user's ComfyUI input folder for good. The masked lane shipped with names
    outside the family (`wmkleinmask_img_…`), which is exactly that leak."""
    d = tmp_path / 'input'
    d.mkdir()
    orphan = _touch(d / name, age_seconds=365 * 24 * 3600)

    assert comfy_fs.is_staged_input_name(name), f'{name} must be recognised as ours'
    assert comfy_fs.prune_staged_inputs(str(d)) == 1
    assert not orphan.exists()


@pytest.mark.parametrize('name', [
    'edit_reference.png',            # a user file that merely STARTS like ours
    'edit_references_backup.png',
    'krea_sources.png',
    'edit_source_notahexuid_a.png',  # right lane, no uid -> not minted by us
    'edit_source.png',
    'wmklein_crop_notes.txt',
    'my_edit_source_0a1b2c3d_a.png',  # our shape, but not at the start
])
def test_the_sweep_never_matches_a_file_it_did_not_mint(tmp_path, name):
    """A loose `startswith` would have eaten the first three of these. The match
    is on the full minted shape `<lane>_<8 hex uid>_…`, anchored at the start."""
    d = tmp_path / 'input'
    d.mkdir()
    victim = _touch(d / name, age_seconds=365 * 24 * 3600)

    assert comfy_fs.prune_staged_inputs(str(d)) == 0
    assert victim.exists(), f'{name} is not ours to delete'
    assert not comfy_fs.is_staged_input_name(name)


def test_an_input_a_live_job_still_points_at_is_never_swept(tmp_path):
    """Fence 3, and the one that has to hold when the other two are stretched: a
    queue that took longer than anyone planned must not cost a user the image of
    a generation still in flight."""
    d = tmp_path / 'input'
    d.mkdir()
    in_flight = _touch(d / 'edit_source_0a1b2c3d_a.png', age_seconds=90 * 24 * 3600)
    orphan = _touch(d / 'edit_source_4e5f6a7b_b.png', age_seconds=90 * 24 * 3600)

    removed = comfy_fs.prune_staged_inputs(str(d), keep={in_flight.name})

    assert removed == 1
    assert in_flight.exists(), 'swept an input a non-terminal job still references'
    assert not orphan.exists()


def test_boot_sweep_collects_the_keep_set_from_unfinished_queue_rows(app, tmp_path,
                                                                    monkeypatch):
    """The fence is only real if the boot path actually builds it: a pending row's
    staged input survives a sweep that deletes its finished neighbour's."""
    from app import job_queue
    from app.extensions import db
    from app.models import ImageGenerationQueue

    d = tmp_path / 'input'
    d.mkdir()
    monkeypatch.setattr('app.config.comfyui_dir',
                        lambda kind: str(d) if kind == 'input' else None)
    pending_file = _touch(d / 'edit_source_0a1b2c3d_a.png', age_seconds=90 * 24 * 3600)
    done_file = _touch(d / 'edit_source_4e5f6a7b_b.png', age_seconds=90 * 24 * 3600)

    with app.app_context():
        db.session.add(ImageGenerationQueue(
            job_id='still-queued', user_id='local', status='pending',
            workflow_data='{}',
            job_metadata=json.dumps({'staged_inputs': [pending_file.name]})))
        db.session.add(ImageGenerationQueue(
            job_id='all-done', user_id='local', status='completed',
            workflow_data='{}',
            job_metadata=json.dumps({'staged_inputs': [done_file.name]})))
        db.session.commit()

        qm = job_queue.JobQueueManager()
        qm.init_app(app)
        qm._prune_staged_inputs()

    assert pending_file.exists(), "a pending job's staged input was swept"
    assert not done_file.exists()


def test_prune_max_age_outlives_the_worst_case_queue_drain():
    """The number is not a taste: a full fan-out queued at once, each job burning
    the whole poll timeout, is the longest a staged copy can legitimately wait."""
    from app import job_queue
    from app.services.face_dataset_service import MAX_FANOUT

    worst_case = MAX_FANOUT * job_queue.POLL_TIMEOUT_SECONDS
    assert comfy_fs.STAGED_INPUT_MAX_AGE_SECONDS > worst_case


def test_prune_survives_a_missing_or_unset_input_folder(tmp_path):
    assert comfy_fs.prune_staged_inputs(None) == 0
    assert comfy_fs.prune_staged_inputs(str(tmp_path / 'nope')) == 0
