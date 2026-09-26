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







# --- one test per reader ------------------------------------------------------

















# --- the launcher -------------------------------------------------------------

def test_the_video_launcher_stamps_the_table_on_the_run(app, tmp_path):
    """Without the stamp every reader above falls back to face, and the run is
    indistinguishable from a face run the moment the launch call returns."""
    from app.models import CloudTrainingRun
    from lds_cloud_training import cloud_video_training as cvt
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
    from lds_cloud_training import cloud_training as ct
    from lds_cloud_training import cloud_video_training as cvt
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
    from lds_cloud_training import cloud_training as ct
    from lds_cloud_training import cloud_video_training as cvt
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




# --- refusals, before any money is spent --------------------------------------

def test_a_target_with_no_verified_base_is_refused_before_renting_anything(app, tmp_path):
    """The whole point of raising in the builder was to fail before the pod. That
    only holds if the launcher builds the config BEFORE it reserves or rents —
    otherwise the refusal arrives from the monitor thread, minutes later, with a
    GPU already on the clock."""
    from app.models import CloudTrainingRun
    from lds_cloud_training import cloud_video_training as cvt
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
    from lds_cloud_training import cloud_video_training as cvt
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
    from lds_cloud_training import cloud_video_training as cvt
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
    from lds_cloud_training import cloud_video_training as cvt
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

























def test_a_replayed_run_keeps_every_stamped_training_flag():
    """_relaunch_args exists so a retry replays the ORIGINAL training, not
    today's dataset row. That promise is only as good as the list of flags it
    copies - do_i2v was missed the day it shipped, and a retried i2v run would
    have silently trained t2v. Pinned here so the next flag cannot repeat it."""
    from lds_cloud_training.cloud_video_training import _relaunch_args
    args = _relaunch_args({'base_model': '', 'low_vram': True, 'do_i2v': True, 'rank': 32,
                           'sample_prompts': ['a wave'], 'distillation': 'off',
                           'requested_gpu': 'A100 SXM4'})
    assert args == {'base_model': None, 'low_vram': True, 'do_i2v': True, 'rank': 32,
                    'sample_prompts': ['a wave'], 'distillation': 'off',
                    'gpu_name': 'A100 SXM4'}


def test_previews_and_the_distillation_override_ride_the_stamp(
        app, tmp_path, monkeypatch):
    """Two launch-time levers, both stamped so the pod rebuild minutes later
    replays the launch and not the present: every requested `sample_prompts`
    entry (including selections beyond the former four-prompt limit) and
    `distillation: off`, which exists for MEASUREMENT - it is the only way to
    run one dataset with and without upstream's de-distillation recipe and
    compare the previews. 'auto' stamps nothing and keeps the gated default."""
    from lds_cloud_training import cloud_video_training as cvt
    calls = []
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        monkeypatch.setattr(cvt, '_start_pod', lambda run: calls.append(run))
        out = cvt.launch_cloud_video_training(
            'local', vid.id, steps=100, sample_prompts=['a wave', '  ', 'a dog', 'a cat', 'a bird', 'a boat', 'a train'],
            distillation='off', _provision=lambda run: calls.append(run))
        from app.models import CloudTrainingRun
        run = db.session.get(CloudTrainingRun, out['run_id'])
        p = json.loads(run.train_params)
        assert p['sample_prompts'] == ['a wave', 'a dog', 'a cat', 'a bird', 'a boat', 'a train']
        assert p['distillation'] == 'off'
        with pytest.raises(ValueError):
            cvt.launch_cloud_video_training(
                'local', vid.id, steps=100, distillation='sideways',
                _provision=lambda run: None)


def test_the_off_stamp_beats_a_capable_image_and_prompts_reach_the_config(
        app, tmp_path, monkeypatch):
    """A capable image normally arms the recipe; the experiment stamp must win
    or the A/B has no control arm. And the stamped prompts come out as the
    sample block, sized to the dataset's own frames and fps."""
    from lds_cloud_training import cloud_training as ct
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
