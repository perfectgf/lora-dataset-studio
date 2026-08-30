"""Which table does a cloud run's `dataset_id` point into — and every reader
that would otherwise answer "the face one" for a video run.

THE BUG THIS FILE EXISTS TO PREVENT
-----------------------------------
`cloud_training_run.dataset_id` has always meant a `face_dataset.id`. Letting a
run point at a `video_dataset.id` instead puts two tables in ONE integer space:
video dataset #3 and face dataset #3 both exist, both are plausible, and every
consumer that resolves the id without asking which table would quietly serve the
wrong one. Nothing raises. The hub would show a video run under a face dataset's
name, a face dataset's checkpoint route would serve a video run's weights, and
`import_checkpoint` would deploy a Wan LoRA into a face dataset's ComfyUI folder.

So each test below pins ONE reader, and each one is written the same way: create a
face dataset and a video dataset that COLLIDE on id, then assert the reader picks
the right one. A test that only exercises a video run in isolation would pass
against the bug.

THE LEGACY HALF MATTERS AS MUCH
-------------------------------
The column is additive and nullable. Every row in every existing user database
predates it and reads NULL, and NULL must mean `face_dataset` — the meaning those
rows have always had. A default that only applies to newly-inserted rows would
leave every historical run unroutable.
"""
import json
from app.extensions import db
import os

import pytest

from app.services import cloud_run_dataset as crd


def _face_dataset(name='a face set'):
    from app.models import FaceDataset
    from app.extensions import db
    ds = FaceDataset(user_id='local', name=name, trigger_word='trg')
    db.session.add(ds)
    db.session.commit()
    return ds


def _video_dataset(tmp_path=None, name='a video set', out_dir=None, frames=81,
                   profile='wan22_14b', width=384, height=384, clips=1):
    """A built video dataset: the row PLUS the flat mp4 + .txt folder on disk.

    The folder is not optional garnish — the launcher counts clips before it
    reserves anything, because a folder with none uploads captions alone and
    trains on nothing. A fixture that skipped it would only ever exercise that
    refusal."""
    from app.models import VideoDataset
    from app.extensions import db
    if out_dir is None:
        out_dir = str(tmp_path / 'vds')
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, clips + 1):
        with open(os.path.join(out_dir, f'clip_{i:04d}.mp4'), 'wb') as fh:
            fh.write(b'\x00')
        with open(os.path.join(out_dir, f'clip_{i:04d}.txt'), 'w') as fh:
            fh.write('a person walking')
    vd = VideoDataset(user_id='local', name=name, target_profile=profile,
                      fps=16, frames=frames, width=width, height=height,
                      output_dir=out_dir)
    db.session.add(vd)
    db.session.commit()
    return vd


def _run(dataset_id, dataset_table=None, status='done', steps=100, **kw):
    from app.models import CloudTrainingRun
    from app.extensions import db
    run = CloudTrainingRun(dataset_id=dataset_id, status=status, job_name='j',
                           vast_label='lds-x',
                           train_params=json.dumps({'steps': steps}), **kw)
    if dataset_table is not None:
        run.dataset_table = dataset_table
    db.session.add(run)
    db.session.commit()
    return run


# --- the column itself --------------------------------------------------------

def test_a_run_that_predates_the_column_reads_as_a_face_run(app):
    """NULL is not "unknown", it is "face_dataset" — the only meaning the column
    could have had before it existed. Every row in every shipped database is in
    this state, so getting it wrong strands all of them at once."""
    with app.app_context():
        run = _run(_face_dataset().id)
        assert run.dataset_table is None          # nothing was stamped
        assert crd.table_of(run) == crd.FACE
        assert crd.is_video(run) is False


def test_the_column_is_declared_as_an_additive_migration(app):
    """A new column on an EXISTING SQLite database only appears if it is listed
    in _SCHEMA_ADDITIONS — `db.create_all()` never alters a table that is already
    there. Without this entry the feature works on a fresh install and raises
    OperationalError on every real one."""
    from app import _SCHEMA_ADDITIONS
    entries = [e for e in _SCHEMA_ADDITIONS
               if e[0] == 'cloud_training_run' and e[1] == 'dataset_table']
    assert len(entries) == 1
    # Nullable, no NOT NULL: the historical rows must be allowed to stay NULL and
    # be READ as face, rather than be rewritten by a migration.
    assert 'NOT NULL' not in entries[0][2].upper()


def test_an_unknown_table_value_is_refused_rather_than_guessed(app):
    """A value this build does not know (a downgrade, a hand-edited row) must not
    fall back to face — that is exactly the silent mis-attribution the column
    exists to stop."""
    with app.app_context():
        run = _run(1, dataset_table='something_else')
        with pytest.raises(ValueError):
            crd.table_of(run)


# --- one test per reader ------------------------------------------------------

def test_the_run_name_shown_in_the_hub_comes_from_the_right_table(app, tmp_path):
    """`_dataset_name` fed the run hub and the shared run config. On a colliding
    id it named the FACE dataset for a video run — a label that is not merely
    unhelpful but wrong about what was trained."""
    from app.services import cloud_training as ct
    with app.app_context():
        face = _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        assert face.id == vid.id            # the collision this file is about
        assert ct._dataset_name(face.id) == 'portraits'
        assert ct.run_dataset_name(_run(face.id)) == 'portraits'
        assert ct.run_dataset_name(_run(vid.id, crd.VIDEO)) == 'surf clips'


def test_a_face_datasets_route_cannot_claim_a_video_run(app, tmp_path):
    """Two routes gate on `run.dataset_id != dataset_id` and then serve that
    run's checkpoints. With one integer space that comparison is TRUE for a video
    run of the same id, so a face dataset's checkpoint endpoint would hand out a
    video run's weights."""
    with app.app_context():
        face = _face_dataset()
        vid = _video_dataset(tmp_path)
        face_run = _run(face.id)
        video_run = _run(vid.id, crd.VIDEO)
        assert crd.owns(face_run, face.id) is True
        assert crd.owns(video_run, face.id) is False
        assert crd.owns(video_run, vid.id, crd.VIDEO) is True
        assert crd.owns(face_run, vid.id, crd.VIDEO) is False


def test_the_launch_guardrail_counts_only_runs_of_the_same_table(app, tmp_path, monkeypatch):
    """The single-active-run guard refuses a second launch on the same
    (dataset, family). A video run of the colliding id must not block a face
    dataset's launch, and vice versa — one user's video training would otherwise
    lock another dataset's button with no explanation."""
    from app.services import cloud_training as ct
    with app.app_context():
        # The FLEET limit is deliberately not scoped by table — it is about the
        # account's pods and its money, which one lane cannot claim — so it has
        # to be lifted here or it, not the per-dataset key, is what refuses.
        ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
        face = _face_dataset()
        vid = _video_dataset(tmp_path)
        _run(vid.id, crd.VIDEO, status='training')
        # An ACTIVE video run of the same id leaves the face lane free.
        ct._assert_launch_guardrails(face.id, 'zimage')
        # ...and the video lane is the one that is now taken.
        with pytest.raises(RuntimeError):
            ct._assert_launch_guardrails(vid.id, 'video', crd.VIDEO)


def test_an_active_run_with_an_unreadable_table_blocks_rather_than_allows(app):
    """`crd.owns` answers False for a row whose table it cannot read, which is
    the right answer for a route deciding whether to serve a file and the WRONG
    one for a guard deciding whether to rent a second GPU. Fail-open here means
    paying twice for one answer, so the guard asks the question itself and blocks
    on the ambiguity — the same caution it already applies to an unknown family."""
    from app.services import cloud_training as ct
    with app.app_context():
        ct.cfg.save_config({'cloud': {'max_concurrent_runs': 5}})
        run = _run(7, status='training')
        run.dataset_table = 'from_a_newer_build'
        from app.extensions import db
        db.session.commit()
        with pytest.raises(RuntimeError):
            ct._assert_launch_guardrails(7, 'zimage')


def test_a_video_run_never_registers_a_face_provenance_record(app, tmp_path):
    """`checkpoint_registry.register_launch` freezes a manifest of face-dataset
    IMAGES and their caption hashes. There is nothing to freeze for a video
    dataset, and registering anyway would file the run under face dataset #N —
    it would then appear in that dataset's lineage graph forever."""
    from app.models import TrainingRunRecord
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path)
        cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                        _provision=lambda run: None)
        assert TrainingRunRecord.query.filter_by(dataset_id=vid.id).count() == 0


def test_a_video_run_is_never_imported_into_a_face_datasets_lora_folder(
        app, tmp_path, monkeypatch):
    """`_import_result` calls `lt.import_checkpoint('local', run.dataset_id, …)`,
    which deploys into the ComfyUI folder of THAT face dataset's family. For a
    video run the id names a different table entirely, so the deploy would land a
    Wan LoRA in a face dataset's Z-Image folder. Out of scope to do properly —
    so it must be SKIPPED, loudly enough to find later, never guessed."""
    from app.services import cloud_training as ct
    with app.app_context():
        vid = _video_dataset(tmp_path)
        run = _run(vid.id, crd.VIDEO, checkpoint_local_path='/nowhere/x.safetensors')
        monkeypatch.setattr(ct.lt, 'import_checkpoint', lambda *a, **k: pytest.fail(
            'a video run reached the face import path'))
        assert ct._import_result(run) is None


def test_a_video_run_has_no_local_run_directory_to_mirror_into(
        app, tmp_path, monkeypatch):
    """The mirror copies harvested checkpoints into `lt._run_dir(user, dataset_id,
    …)` — a path built from a FACE dataset's folder. There is no local video
    training lane, so there is no directory to mirror into; the mirror must stand
    down rather than write into a face dataset's run folder."""
    from app.services import cloud_training as ct
    with app.app_context():
        vid = _video_dataset(tmp_path)
        run = _run(vid.id, crd.VIDEO)
        monkeypatch.setattr(ct.lt, '_run_dir', lambda *a, **k: pytest.fail(
            'a video run asked for a face run directory'))
        assert ct._mirror_into_local_run(run) is None


def test_the_shared_run_config_does_not_invent_a_face_dataset(app, tmp_path):
    """run_share resolves `db.session.get(FaceDataset, dataset_id)` to describe the run
    it exports. On a video run that returns the colliding FACE row, and the
    exported config would describe someone else's dataset."""
    from app.services import run_share
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO)
        text = run_share.build_run_config_text(f'cloud-{run.id}')['text']
        assert 'portraits' not in text


# --- the launcher -------------------------------------------------------------

def test_the_video_launcher_stamps_the_table_on_the_run(app, tmp_path):
    """Without the stamp every reader above falls back to face, and the run is
    indistinguishable from a face run the moment the launch call returns."""
    from app.models import CloudTrainingRun
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path)
        res = cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                              _provision=lambda run: None)
        run = db.session.get(CloudTrainingRun, res['run_id'])
        assert run.dataset_table == crd.VIDEO
        assert crd.is_video(run) is True
        assert run.dataset_id == vid.id


def test_the_video_launcher_skips_the_image_preflight_entirely(
        app, tmp_path, monkeypatch):
    """`assert_trainable` counts IMAGES and their captions, and
    `export_dataset_to_aitoolkit` re-exports them with rembg masks. A video
    dataset has neither: its folder is already the flat mp4 + .txt shape
    ai-toolkit wants. Calling either would fail on an empty image set — and
    "fixing" that by relaxing the preflight would relax it for face runs too."""
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path)
        monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **k: pytest.fail(
            'the image preflight ran on a video dataset'))
        monkeypatch.setattr(ct.lt, 'export_dataset_to_aitoolkit', lambda *a, **k: pytest.fail(
            'the image export ran on a video dataset'))
        cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                        _provision=lambda run: None)


def test_the_clips_are_uploaded_from_the_dataset_folder_itself(app, tmp_path):
    """No staging copy. The video dataset's `output_dir` is ALREADY the flat
    mp4 + homonym .txt folder, and a dataset of 81-frame clips is gigabytes — a
    copy would double the disk and the wait for nothing. The upload seam must
    point straight at it."""
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    out = tmp_path / 'vds'
    out.mkdir()
    (out / 'clip_0001.mp4').write_bytes(b'\x00')
    (out / 'clip_0001.txt').write_text('a person walking')
    with app.app_context():
        vid = _video_dataset(out_dir=str(out))
        res = cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                              _provision=lambda run: None)
        from app.models import CloudTrainingRun
        run = db.session.get(CloudTrainingRun, res['run_id'])
        assert ct._staging_dataset_dir(run) == str(out)


def test_the_pod_job_is_built_by_the_video_branch(app, tmp_path):
    """The config the pod actually receives. Proves the launcher reaches
    `video_training.build_job_config` (num_frames, the MoE boundary, the arch)
    and not `lt.build_job_config`, which would read the face columns this row
    does not have."""
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path, frames=81)
        res = cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                              _provision=lambda run: None)
        from app.models import CloudTrainingRun
        run = db.session.get(CloudTrainingRun, res['run_id'])
        cfg = ct._build_pod_job_config(run, '/staged/vid',
                                       {'DATASETS_FOLDER': '/workspace/datasets',
                                        'TRAINING_FOLDER': '/workspace/out'})
        proc = cfg['config']['process'][0]
        assert proc['type'] == 'diffusion_trainer'
        assert proc['datasets'][0]['num_frames'] == 81
        assert proc['train']['switch_boundary_every'] == 10
        assert proc['model']['arch'] == 'wan22_14b'
        assert proc['train']['steps'] == 500


# --- refusals, before any money is spent --------------------------------------

def test_a_target_with_no_verified_base_is_refused_before_renting_anything(app, tmp_path):
    """The whole point of raising in the builder was to fail before the pod. That
    only holds if the launcher builds the config BEFORE it reserves or rents —
    otherwise the refusal arrives from the monitor thread, minutes later, with a
    GPU already on the clock."""
    from app.models import CloudTrainingRun
    from app.services import cloud_video_training as cvt
    from app.services import video_training as vt
    with app.app_context():
        vid = _video_dataset(tmp_path, profile='ltx23', frames=81)
        with pytest.raises(vt.VideoTrainingUnsupported) as e:
            cvt.launch_cloud_video_training(
                'local', vid.id, steps=500,
                _provision=lambda run: pytest.fail('a pod was rented anyway'))
        assert 'base' in str(e.value).lower()
        # and no half-created run row is left behind to block the next attempt
        assert CloudTrainingRun.query.filter_by(dataset_id=vid.id).count() == 0


def test_the_generic_profile_is_refused_at_launch(app, tmp_path):
    """Same gate, the other refusal: `generic` has no `aitk_arch` at all."""
    from app.models import CloudTrainingRun
    from app.services import cloud_video_training as cvt
    from app.services import video_training as vt
    with app.app_context():
        vid = _video_dataset(tmp_path, profile='generic', frames=40)
        with pytest.raises(vt.VideoTrainingUnsupported):
            cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                            _provision=lambda run: None)
        assert CloudTrainingRun.query.filter_by(dataset_id=vid.id).count() == 0


def test_an_empty_dataset_folder_is_refused_before_renting_anything(app, tmp_path):
    """A folder with no .mp4 uploads nothing (the extension filter is the only
    thing that ships clips) and the pod trains on an empty set. Cheap to check
    here; expensive to discover on a rented GPU."""
    from app.services import cloud_video_training as cvt
    empty = tmp_path / 'empty'
    empty.mkdir()
    with app.app_context():
        vid = _video_dataset(out_dir=str(empty), clips=0)
        with pytest.raises(ValueError) as e:
            cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                            _provision=lambda run: None)
        assert 'clip' in str(e.value).lower()


def test_two_video_runs_on_one_dataset_are_refused(app, tmp_path):
    """The same single-active-run guard the face lane has, for the same reason: a
    second pod on the same dataset is money spent twice on one answer. It raises
    RuntimeError, exactly as the face lane's does — the two launches share that
    guard rather than each having their own idea of the refusal."""
    from app.services import cloud_video_training as cvt
    out = tmp_path / 'vds2'
    out.mkdir()
    (out / 'clip_0001.mp4').write_bytes(b'\x00')
    with app.app_context():
        vid = _video_dataset(out_dir=str(out))
        cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                        _provision=lambda run: None)
        with pytest.raises(RuntimeError):
            cvt.launch_cloud_video_training('local', vid.id, steps=500,
                                            _provision=lambda run: None)


# --- the HTTP surface ---------------------------------------------------------

def test_the_launch_route_starts_a_video_run(app, client, tmp_path, monkeypatch):
    """Without a route the lane is reachable only from a Python shell. This is the
    smallest surface that makes it real, and it is the one place a user-supplied
    step count arrives from outside."""
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path)
        vid_id = vid.id
    monkeypatch.setattr(cvt, '_start_pod', lambda run: None)
    r = client.post(f'/api/video-dataset/{vid_id}/train/cloud',
                    json={'steps': 700})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['steps'] == 700 and body['clips'] == 1
    with app.app_context():
        from app.models import CloudTrainingRun
        run = db.session.get(CloudTrainingRun, body['run_id'])
        assert crd.is_video(run) is True


def test_the_launch_route_reports_an_unsupported_target_as_a_refusal(
        app, client, tmp_path, monkeypatch):
    """A target with no verified base must come back as a 4xx the UI can render,
    not a 500 — it is a choice the user can correct, not a server fault."""
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path, profile='ltx23', frames=81)
        vid_id = vid.id
    monkeypatch.setattr(cvt, '_start_pod',
                        lambda run: pytest.fail('a pod was started anyway'))
    r = client.post(f'/api/video-dataset/{vid_id}/train/cloud', json={})
    assert r.status_code == 400
    assert 'base' in r.get_json()['error'].lower()


def test_an_unknown_video_dataset_is_a_404(app, client):
    r = client.post('/api/video-dataset/98765/train/cloud', json={})
    assert r.status_code == 404


def test_the_progress_route_reads_the_video_lane_not_the_face_one(
        app, client, tmp_path):
    """`latest_run_for` resolves the newest run BY INTEGER. Polled from a video
    dataset's page it must find the video run, not the face run of the same id —
    the exact mis-attribution its table scope exists to stop."""
    from app.services import cloud_training as ct
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        _run(vid.id, status='done')                    # a FACE run, same id
        video_run = _run(vid.id, crd.VIDEO, status='training')
        vid_id, video_run_id = vid.id, video_run.id
        assert ct.latest_run_for(vid_id, dataset_table=crd.VIDEO).id == video_run_id
        # ...and the face lane still sees only its own newest run
        assert ct.latest_run_for(vid_id).id != video_run_id
    r = client.get(f'/api/video-dataset/{vid_id}/train/cloud/progress')
    assert r.status_code == 200
    assert r.get_json()['run_id'] == video_run_id


def test_a_face_datasets_checkpoint_panel_lists_none_of_a_video_runs_saves(
        app, tmp_path):
    """`cloud_checkpoint_groups` feeds the Checkpoints panel of ONE dataset, and
    every save it lists is offered for deployment into that dataset's ComfyUI
    folder. Queried by id alone it would list a colliding video run's Wan saves
    there — a two-file MoE checkpoint presented as this face dataset's LoRA."""
    from app.services import cloud_training as ct
    with app.app_context():
        face = _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        staging = tmp_path / 'vrun'
        staging.mkdir()
        for stage in ('high_noise', 'low_noise'):
            (staging / f'lora_v_000000050_{stage}.safetensors').write_bytes(b'W')
        _run(vid.id, crd.VIDEO, staging_dir=str(staging))
        assert ct.cloud_checkpoint_groups(face.id) == []
        groups = ct.cloud_checkpoint_groups(vid.id, dataset_table=crd.VIDEO)
        assert len(groups) == 1
        assert sorted(c['filename'] for c in groups[0]['checkpoints']) == [
            'lora_v_000000050_high_noise.safetensors',
            'lora_v_000000050_low_noise.safetensors']
        # and both halves carry the step the pair actually saved at
        assert {c['step'] for c in groups[0]['checkpoints']} == {50}


def test_retry_and_continue_never_relaunch_a_video_run_as_a_face_one(
        app, tmp_path, monkeypatch):
    """Both rebuild their arguments from a run's stamped params and call
    `launch_cloud_training`, which resolves `dataset_id` as a FACE dataset. On a
    colliding id that is not an error — it is a face training launched on someone
    else's data, and billed.

    They used to REFUSE a video run for that reason. They now dispatch to the
    video lane's own relaunchers instead, and this test keeps the half that was
    never about the refusal: whatever the two entry points do with a video run,
    the face launcher must not see it. `lt.assert_trainable` is the first thing
    `launch_cloud_training` calls on a dataset, so a video run reaching it fails
    here rather than on someone's bill."""
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    called = []
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='error')
        monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **k: pytest.fail(
            'a video run re-entered the face launcher'))
        monkeypatch.setattr(cvt, 'launch_cloud_video_training',
                            lambda *a, **k: called.append(k) or {'run_id': 1})
        ct.retry_cloud_run('local', run.id)
        run.status = 'done'
        from app.extensions import db
        db.session.commit()
        # Nothing harvested: the video lane's own refusal, in its own words —
        # which is the proof the call landed there and not in the face lane.
        with pytest.raises(ValueError) as e2:
            ct.continue_cloud_run('local', run.id)
        assert 'checkpoint' in str(e2.value).lower()
    assert len(called) == 1


def test_a_video_pod_boots_the_video_image_and_a_face_pod_keeps_the_pin(app, tmp_path):
    """The face lane's image tag is pinned to the ai-toolkit commit its dense
    recipe was validated against (2026-07-12). The video lane cannot use it: the
    `minimax_h3` architecture landed in ai-toolkit on 2026-08-03, so a video pod
    booted on the face pin refuses the job only AFTER the rental. Same colliding
    setup as everything in this file — the resolver must answer per RUN, not per
    config, or the fresher tag would leak into face runs (whose verdicts were
    read against the old one) the day it was added."""
    from app.services.cloud_training import _pod_image_for
    c = {'image': 'toolkit:face-pin', 'video_image': 'toolkit:video-fresh'}
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        video_run = _run(vid.id, crd.VIDEO)
        face_run = _run(vid.id)                      # NULL table = face, always
        assert _pod_image_for(video_run, c) == 'toolkit:video-fresh'
        assert _pod_image_for(face_run, c) == 'toolkit:face-pin'
        # No video_image configured: an older trainer beats no trainer — Wan
        # runs still work on the shared pin, so the lane falls back rather than
        # refusing to provision at all.
        assert _pod_image_for(video_run, {'image': 'toolkit:face-pin'}) \
            == 'toolkit:face-pin'


def test_the_video_family_rents_at_least_48_gb_of_vram():
    """`_provision` resolves min VRAM as `min_vram_gb.get(family, 24)`. Before
    the 'video' entry existed that fallback rented 24 GB pods for a lane that
    runs with low_vram OFF — resident weights alone (H3: ~21 GB transformer +
    ~16 GB text encoder; Wan 2.2: two experts) exceed that, and the OOM arrives
    after the money is spent. The default the resolution reads must say so."""
    from app.config import DEFAULTS
    floors = DEFAULTS['cloud']['min_vram_gb']
    assert floors.get('video', 24) >= 48


def test_a_video_pod_is_never_rented_with_the_face_lane_s_disk():
    """A video pod holds its base, its transfer's working copy and a latent
    cache on ONE vast allocation, and MiniMax H3's base alone is 42.5 GB — the
    60 GB the face lane rents cannot hold that, and the shortfall surfaces
    mid-download, after the pod is paid for. Third of the same family as the image pin and the VRAM
    floor: everything that decides whether the money buys a training must be
    decided BEFORE the rental.

    The floor lives in code and not only in DEFAULTS because `config.json`
    freezes the whole `cloud` block as it was saved: an install that opened
    Settings before this key existed carries `disk_gb: 60` and no
    `video_disk_gb` at all, and a resolver reading the config alone would rent
    60 GB from it forever."""
    from app.config import DEFAULTS
    from app.services.cloud_training import _VIDEO_DISK_FLOOR_GB, _disk_gb_for
    from app.services.video_training_local import WEIGHT_FOOTPRINTS
    stale = {'disk_gb': 60}                      # a cloud block frozen in July
    assert _disk_gb_for(stale, {'train_type': 'video'}) >= _VIDEO_DISK_FLOOR_GB
    # The face lane is untouched by all of this — its pin is its own verdict.
    assert _disk_gb_for(stale, {'train_type': 'zimage'}) == 60
    # Not a round number picked for comfort: the floor has to clear the largest
    # base this lane declares AND still leave the face lane's entire allocation
    # over for the image, the caches and the saves. Declaring a bigger base
    # later fails HERE rather than on a rented pod.
    biggest = max(f['gigabytes'] for f in WEIGHT_FOOTPRINTS.values())
    assert _VIDEO_DISK_FLOOR_GB > biggest + 60
    assert DEFAULTS['cloud']['video_disk_gb'] >= _VIDEO_DISK_FLOOR_GB


def test_only_the_video_family_carries_a_compute_capability_floor():
    """Fourth of the "decide before the rental" family, and the one that only a
    price list makes visible: every video job this app writes sets dtype bf16,
    Turing has no bf16 — and Turing is where the cheapest offer lives. Measured
    on the live market (2026-08-29): the cheapest board clearing this lane's
    48 GB VRAM and 120 GB disk floors was a Quadro RTX 8000, compute_cap 750, at
    $0.261/h against $0.802 for the next one up. Cheapest-above-the-floors
    therefore reaches for the single card in the list that cannot do the work.

    Scoped to 'video' on purpose: the face families' offer pools are what their
    recipes were measured against, and widening or narrowing them here would be
    a change nobody asked for, made invisibly."""
    from app.config import DEFAULTS
    floors = DEFAULTS['cloud']['min_compute_cap']
    assert floors['video'] >= 800          # Ampere is the first bf16 generation
    assert set(floors) == {'video'}


def test_video_gpu_tiers_prices_every_class_and_refuses_to_invent_estimates(
        app, tmp_path, monkeypatch):
    """The face lane's launch shows tiers; the video lane picked the cheapest
    suitable offer silently. Tiers now exist here too, with two honesty rules:
    the search applies the SAME floors the launch does (VRAM, disk, compute
    generation — a picker listing cards the launch refuses is a menu of dead
    ends), and the estimate comes from one measured run scaled by latent rows,
    so a frame count off the 17n+5 grid gets None, never a made-up number."""
    from app.services import cloud_video_training as cvt
    from app.services import cloud_training as ct
    from app.services import vast_client
    seen = {}

    def fake_search(**kw):
        seen.update(kw)
        return [
            {'gpu_name': 'A100 SXM4', 'offer_id': 1, 'dph_total': 1.0,
             'gpu_ram_gb': 80, 'reliability': 0.99, 'machine_id': 1},
            {'gpu_name': 'A100 SXM4', 'offer_id': 2, 'dph_total': 0.9,
             'gpu_ram_gb': 80, 'reliability': 0.99, 'machine_id': 2},
            {'gpu_name': 'H100 PCIE', 'offer_id': 3, 'dph_total': 2.3,
             'gpu_ram_gb': 80, 'reliability': 0.99, 'machine_id': 3},
        ]

    monkeypatch.setattr(vast_client, 'search_offers', fake_search)
    monkeypatch.setattr(ct.cfg, 'secret', lambda k: 'key' if k == 'VAST_API_KEY' else None)
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')      # frames=81: off H3 grid
        data = cvt.video_gpu_tiers('local', vid.id, steps=100)
    assert seen['min_vram_gb'] == 48
    assert seen['min_compute_cap'] == 800
    assert seen['min_disk_gb'] >= 120
    by_name = {t['gpu_name']: t for t in data['tiers']}
    assert by_name['A100 SXM4']['dph_total'] == 0.9       # cheapest of the class
    # 81 frames is legal for Wan and OFF the measured H3 grid: no estimate.
    assert by_name['A100 SXM4']['est_minutes'] is None
    assert by_name['A100 SXM4']['estimate_status'] == 'unavailable'


def test_the_video_estimate_reproduces_the_measured_run():
    """21 s/step at 107 frames on an A100 SXM4 is the one number this model was
    built from; 100 steps of it took ~35 minutes on run #166. The estimate must
    give that back exactly - it is a citation, not a curve fit."""
    from app.services import gpu_speed
    assert round(gpu_speed.video_estimate_minutes('A100 SXM4', 107, 100)) == 35
    assert gpu_speed.video_estimate_minutes('A100 SXM4', 40, 100) is None
    assert gpu_speed.video_latent_rows(39) == 12


def test_a_replayed_run_keeps_every_stamped_training_flag():
    """_relaunch_args exists so a retry replays the ORIGINAL training, not
    today's dataset row. That promise is only as good as the list of flags it
    copies - do_i2v was missed the day it shipped, and a retried i2v run would
    have silently trained t2v. Pinned here so the next flag cannot repeat it."""
    from app.services.cloud_video_training import _relaunch_args
    args = _relaunch_args({'base_model': '', 'low_vram': True, 'do_i2v': True,
                           'sample_prompts': ['a wave'], 'distillation': 'off',
                           'requested_gpu': 'A100 SXM4'})
    assert args == {'base_model': None, 'low_vram': True, 'do_i2v': True,
                    'sample_prompts': ['a wave'], 'distillation': 'off',
                    'gpu_name': 'A100 SXM4'}


def test_previews_and_the_distillation_override_ride_the_stamp(
        app, tmp_path, monkeypatch):
    """Two launch-time levers, both stamped so the pod rebuild minutes later
    replays the launch and not the present: `sample_prompts` (capped at 4 -
    each preview is a full video generation on the paid GPU) and
    `distillation: off`, which exists for MEASUREMENT - it is the only way to
    run one dataset with and without upstream's de-distillation recipe and
    compare the previews. 'auto' stamps nothing and keeps the gated default."""
    from app.services import cloud_video_training as cvt
    calls = []
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        monkeypatch.setattr(cvt, '_start_pod', lambda run: calls.append(run))
        out = cvt.launch_cloud_video_training(
            'local', vid.id, steps=100, sample_prompts=['a wave', '  ', 'a dog'],
            distillation='off', _provision=lambda run: calls.append(run))
        from app.models import CloudTrainingRun
        run = db.session.get(CloudTrainingRun, out['run_id'])
        p = json.loads(run.train_params)
        assert p['sample_prompts'] == ['a wave', 'a dog']    # blanks dropped
        assert p['distillation'] == 'off'
        with pytest.raises(ValueError):
            cvt.launch_cloud_video_training(
                'local', vid.id, steps=100,
                sample_prompts=['1', '2', '3', '4', '5'],
                _provision=lambda run: None)
        with pytest.raises(ValueError):
            cvt.launch_cloud_video_training(
                'local', vid.id, steps=100, distillation='sideways',
                _provision=lambda run: None)


def test_the_off_stamp_beats_a_capable_image_and_prompts_reach_the_config(
        app, tmp_path, monkeypatch):
    """A capable image normally arms the recipe; the experiment stamp must win
    or the A/B has no control arm. And the stamped prompts come out as the
    sample block, sized to the dataset's own frames and fps."""
    from app.services import cloud_training as ct
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO)
        run.train_params = json.dumps({
            'train_type': 'video', 'steps': 100,
            'target_profile': vid.target_profile, 'frames': vid.frames,
            'distillation': 'off', 'sample_prompts': ['a wave at dusk']})
        db.session.commit()
        monkeypatch.setattr(ct.cfg, 'get', lambda k=None: {
            'video_image': 'vastai/ostris-ai-toolkit:x-2026-08-27-cuda-12.9'}
            if k == 'cloud' else {})
        cfg = ct._build_pod_job_config(run, str(tmp_path / 'stage'),
                                       {'DATASETS_FOLDER': '/workspace/datasets',
                                        'TRAINING_FOLDER': '/workspace/output'})
        proc = cfg['config']['process'][0]
        assert 'assistant_lora_path' not in proc['model']       # off won
        assert 'do_guidance_loss' not in proc['train']
        assert proc['sample']['prompts'] == ['a wave at dusk']
        assert proc['sample']['num_frames'] == vid.frames
        assert proc['sample']['fps'] == vid.fps


def test_the_automatic_retry_of_a_video_run_goes_down_the_video_lane(
        app, tmp_path, monkeypatch):
    """Found live on run #169: a video run's boot-timeout retry went down the
    FACE path, whose dataset lookup reads the face table, and died on "dataset
    not found" — the safety net that exists to survive a bad host was the one
    thing that could not. The retry now branches on the run's own table and
    replays the video stamps, with the same bookkeeping the face path writes
    (auto_retry_of is how a crash finds its child; the count bounds the
    ladder)."""
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    seen = {}

    def fake_launch(user_id, dataset_id, **kw):
        seen.update(kw, dataset_id=dataset_id)
        return {'run_id': 999, 'status': 'preparing'}

    monkeypatch.setattr(cvt, 'launch_cloud_video_training', fake_launch)
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='error')
        run.vast_instance_id = '123'
        run.gpu_name = 'A100 SXM4'
        run.train_params = json.dumps({
            'train_type': 'video', 'steps': 100, 'distillation': 'off',
            'sample_prompts': ['a wave'], 'do_i2v': False})
        db.session.commit()
        out = ct._maybe_auto_retry(run, 'pod did not become ready in time')
        assert out == {'run_id': 999, 'status': 'preparing'}
        assert seen['dataset_id'] == vid.id
        assert seen['distillation'] == 'off'          # the experiment survives
        assert seen['sample_prompts'] == ['a wave']
        assert seen['auto_retry_of'] == run.id
        assert seen['auto_retry_count'] == 1
        assert seen['gpu_name'] == 'A100 SXM4'

