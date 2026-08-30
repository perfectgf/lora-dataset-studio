"""What happens to a video cloud run AFTER it is launched.

`test_cloud_video_launch.py` pins the launch and every reader that could confuse
a video run with the face run of the same id. This file pins the rest of the
round trip, which until now simply did not exist: getting the weights back,
relaunching a run that failed, continuing one that finished, saying afterwards
what a run was made of — and, before any of that, checking that the rented pod
can actually DECODE the clips it was sent.

THE ORDER OF THE CHECKS IS THE POINT
------------------------------------
A pod is billed by the hour from the moment it boots. Run #138 spent one on an
upload nobody verified. So the decoder probe runs on the pod's own clips, with
the pod's own decoder, immediately after the upload and BEFORE `start_job` — the
last moment at which the money spent is minutes rather than hours.

WHY THE PAIR KEEPS COMING BACK
------------------------------
A Wan 2.2 MoE checkpoint is TWO files (`_high_noise` / `_low_noise`), and every
operation here has a way of getting that wrong that raises nothing: a download
that serves one half, a continue that seeds one half onto the pod and trains the
other expert from scratch. Each of those has a test below, and each of them is
written against the pair rather than against a single-file arch, because a test
on MiniMax H3 — which saves ONE file per step, verified against ai-toolkit's own
`MinimaxH3Model` (no `save_lora` override, so `network_mixins.save_weights`
writes a single safetensors) — would pass against every one of those bugs.
"""
import json
from app.extensions import db
import os

import pytest

from app.services import cloud_run_dataset as crd
from test_cloud_video_launch import _face_dataset, _run, _video_dataset


def _saves(run, tmp_path, names):
    """Give a run harvested checkpoints on disk, through its staging dir.

    The store would do as well (`run_checkpoint_files` reads both); staging is
    the one a test can point anywhere without touching global config."""
    from app.extensions import db
    d = tmp_path / f'run_{run.id}_saves'
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_bytes(b'W' * 16)
    run.staging_dir = str(d)
    db.session.commit()
    return d


# ── 1. Getting the weights back ───────────────────────────────────────────────

def test_a_video_runs_checkpoint_is_downloadable_from_its_own_dataset(
        app, client, tmp_path):
    """The face route resolves `dataset_id` in the face table, so a video run's
    weights had no way home at all. This is that way, and it is scoped to the
    video table for the same reason the face one is scoped to its own."""
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done')
        _saves(run, tmp_path, ['video_surf_000000050_high_noise.safetensors'])
        vid_id, run_id = vid.id, run.id
    r = client.get(f'/api/video-dataset/{vid_id}/train/cloud/checkpoint'
                   f'?run_id={run_id}'
                   '&filename=video_surf_000000050_high_noise.safetensors')
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.data == b'W' * 16


def test_the_video_download_refuses_a_run_of_the_face_table(app, client, tmp_path):
    """The whole collision, on the endpoint that hands over BYTES: a face run
    carrying the same integer must not be served here. `crd.owns` is the test,
    and this asserts the route actually asks it."""
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        face_run = _run(vid.id, status='done')          # NULL table = face
        _saves(face_run, tmp_path, ['lora_trg_000000050.safetensors'])
        vid_id, face_run_id = vid.id, face_run.id
    r = client.get(f'/api/video-dataset/{vid_id}/train/cloud/checkpoint'
                   f'?run_id={face_run_id}&filename=lora_trg_000000050.safetensors')
    assert r.status_code == 404


def test_the_video_download_takes_a_basename_and_nothing_else(app, client, tmp_path):
    """`run_checkpoint_path` refuses anything that is not a bare name, and this
    endpoint reads its filename straight off the query string."""
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done')
        _saves(run, tmp_path, ['video_surf_000000050.safetensors'])
        vid_id, run_id = vid.id, run.id
    r = client.get(f'/api/video-dataset/{vid_id}/train/cloud/checkpoint'
                   f'?run_id={run_id}&filename=../../../etc/passwd')
    assert r.status_code == 404


def test_the_listing_keeps_both_halves_of_a_wan_pair_at_one_step(
        app, client, tmp_path):
    """A MoE checkpoint is a PAIR, and a panel that offers one half offers a LoRA
    that cannot be loaded. Both files are listed, both under the same step, and
    the step is the one the pair actually saved at — not the run's total."""
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips', profile='wan22_14b')
        run = _run(vid.id, crd.VIDEO, status='done', steps=500)
        _saves(run, tmp_path, [
            'video_surf_000000050_high_noise.safetensors',
            'video_surf_000000050_low_noise.safetensors'])
        vid_id = vid.id
    r = client.get(f'/api/video-dataset/{vid_id}/train/cloud/checkpoints')
    assert r.status_code == 200, r.get_data(as_text=True)
    groups = r.get_json()['groups']
    assert len(groups) == 1
    steps = groups[0]['steps']
    assert len(steps) == 1 and steps[0]['step'] == 50
    assert sorted(steps[0]['files']) == [
        'video_surf_000000050_high_noise.safetensors',
        'video_surf_000000050_low_noise.safetensors']


def test_a_minimax_h3_run_lists_one_file_per_step(app, client, tmp_path):
    """H3 has no `save_lora` override in ai-toolkit, so its LoRA is written by
    `network_mixins.save_weights` as ONE safetensors. Grouping must not invent a
    missing half for it — a UI that waited for a pair would never enable the
    button."""
    with app.app_context():
        vid = _video_dataset(tmp_path, 'h3 clips', profile='minimax_h3',
                             frames=107)
        run = _run(vid.id, crd.VIDEO, status='done', steps=500)
        _saves(run, tmp_path, ['video_h3_000000250.safetensors',
                               'video_h3.safetensors'])
        vid_id = vid.id
    r = client.get(f'/api/video-dataset/{vid_id}/train/cloud/checkpoints')
    steps = r.get_json()['groups'][0]['steps']
    assert [s['step'] for s in steps] == [250, 500]      # final = the run's total
    assert all(len(s['files']) == 1 for s in steps)
    assert steps[-1]['final'] is True


def test_the_listing_shows_nothing_of_the_face_run_of_the_same_id(
        app, client, tmp_path):
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        face_run = _run(vid.id, status='done')
        _saves(face_run, tmp_path, ['lora_trg_000000050.safetensors'])
        vid_id = vid.id
    r = client.get(f'/api/video-dataset/{vid_id}/train/cloud/checkpoints')
    assert r.get_json()['groups'] == []


# ── 2. Relaunching what failed ────────────────────────────────────────────────

def test_retrying_a_video_run_goes_through_the_video_launcher(
        app, tmp_path, monkeypatch):
    """Retry used to refuse, because rebuilding the arguments and calling
    `launch_cloud_training` resolves `dataset_id` as a FACE dataset — on a
    colliding id that is a face training on someone else's data, billed. It now
    rebuilds them for the VIDEO launcher instead, with the params stamped at the
    original launch."""
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    seen = {}
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='error')
        run.train_params = json.dumps({
            'train_type': 'video', 'steps': 700, 'base_model': 'org/base',
            'low_vram': True, 'requested_gpu': 'RTX 5090'})
        from app.extensions import db
        db.session.commit()
        monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **k: pytest.fail(
            'a video run re-entered the face launcher'))
        monkeypatch.setattr(cvt, 'launch_cloud_video_training',
                            lambda *a, **k: seen.update(args=a, kw=k) or {'run_id': 99})
        out = ct.retry_cloud_run('local', run.id)
        vid_id = vid.id
    assert out['run_id'] == 99
    assert seen['args'][1] == vid_id
    assert seen['kw']['steps'] == 700
    assert seen['kw']['base_model'] == 'org/base'
    assert seen['kw']['low_vram'] is True
    assert seen['kw']['gpu_name'] == 'RTX 5090'


def test_only_a_failed_video_run_can_be_retried(app, tmp_path, monkeypatch):
    """Same rule as the face lane. A finished run has a Continue, not a Retry;
    retrying it would rent a pod to redo work that is already on disk."""
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done')
        monkeypatch.setattr(cvt, '_start_pod',
                            lambda r: pytest.fail('a pod was rented anyway'))
        with pytest.raises(ValueError) as e:
            cvt.retry_cloud_video_run('local', run.id)
        assert 'failed' in str(e.value).lower()


def test_a_run_naming_an_unknown_table_is_still_refused(app, tmp_path):
    """The named refusal stays for the case it was written for. A hand-edited or
    downgraded row cannot say which lane it belongs to, and guessing is the
    silent mis-attribution this whole column exists to prevent."""
    from app.services import cloud_training as ct
    with app.app_context():
        run = _run(1, dataset_table='something_else', status='error')
        with pytest.raises(ValueError) as e:
            ct.retry_cloud_run('local', run.id)
        assert 'something_else' in str(e.value)


# ── 3. Continuing what finished ───────────────────────────────────────────────

def test_continuing_a_video_run_targets_the_resumed_step_plus_the_extra(
        app, tmp_path, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import cloud_video_training as cvt
    seen = {}
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done', steps=500)
        _saves(run, tmp_path, ['video_surf_000000200.safetensors',
                               'video_surf_000000400.safetensors'])
        monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **k: pytest.fail(
            'a video run re-entered the face launcher'))
        monkeypatch.setattr(cvt, 'launch_cloud_video_training',
                            lambda *a, **k: seen.update(kw=k) or {'run_id': 7})
        ct.continue_cloud_run('local', run.id, extra_steps=300)
    assert seen['kw']['steps'] == 700                     # 400 harvested + 300
    assert seen['kw']['resume_step'] == 400
    assert [os.path.basename(p) for p in seen['kw']['resume_ckpt_paths']] == [
        'video_surf_000000400.safetensors']


def test_continuing_a_wan_run_seeds_BOTH_experts_of_the_chosen_step(
        app, tmp_path, monkeypatch):
    """The bug this exists to stop raises nothing: seed one half onto the fresh
    pod and ai-toolkit resumes one expert while the other restarts from zero.
    The result is a LoRA that loads, trains, and is quietly half as trained as
    the user believes."""
    from app.services import cloud_video_training as cvt
    seen = {}
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips', profile='wan22_14b')
        run = _run(vid.id, crd.VIDEO, status='done', steps=500)
        _saves(run, tmp_path, [
            'video_surf_000000400_high_noise.safetensors',
            'video_surf_000000400_low_noise.safetensors'])
        monkeypatch.setattr(cvt, 'launch_cloud_video_training',
                            lambda *a, **k: seen.update(kw=k) or {'run_id': 7})
        cvt.continue_cloud_video_run('local', run.id, extra_steps=100)
    assert sorted(os.path.basename(p)
                  for p in seen['kw']['resume_ckpt_paths']) == [
        'video_surf_000000400_high_noise.safetensors',
        'video_surf_000000400_low_noise.safetensors']


def test_the_seed_puts_every_half_of_the_pair_in_the_pods_save_root(
        app, tmp_path, monkeypatch):
    """What `_seed_resume_checkpoint` actually ships. ai-toolkit's auto-resume
    globs `<save_root>/<job_name>*.safetensors` and `Wan2214bModel.load_lora`
    then reads its sibling by rewriting `_high_noise` into `_low_noise` — so the
    two files must arrive under THIS job's prefix WITH their stage suffixes
    intact. Renaming them both to the same stepped name would leave one file."""
    from app.services import cloud_training as ct
    pushed = []
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips', profile='wan22_14b')
        run = _run(vid.id, crd.VIDEO, status='provisioning')
        d = _saves(run, tmp_path, [
            'old_000000400_high_noise.safetensors',
            'old_000000400_low_noise.safetensors'])
        run.job_name = 'lds9_video_surf'
        run.train_params = json.dumps({
            'steps': 500, 'resume_step': 400,
            'resume_ckpt_paths': [str(d / 'old_000000400_high_noise.safetensors'),
                                  str(d / 'old_000000400_low_noise.safetensors')]})
        from app.extensions import db
        db.session.commit()
        monkeypatch.setattr(ct, '_push_resume_checkpoint',
                            lambda run, remote, ps, src, dest, name:
                            pushed.append((os.path.basename(src), dest, name)))
        ct._seed_resume_checkpoint(run, object(),
                                   {'TRAINING_FOLDER': '/workspace/out',
                                    'DATASETS_FOLDER': '/workspace/datasets'})
    assert len(pushed) == 2
    assert {p[1] for p in pushed} == {'/workspace/out/lds9_video_surf'}
    assert sorted(p[2] for p in pushed) == [
        'lds9_video_surf_000000400_high_noise.safetensors',
        'lds9_video_surf_000000400_low_noise.safetensors']


def test_continuing_a_video_run_with_nothing_harvested_is_refused(
        app, tmp_path, monkeypatch):
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done')
        monkeypatch.setattr(cvt, '_start_pod',
                            lambda r: pytest.fail('a pod was rented anyway'))
        with pytest.raises(ValueError) as e:
            cvt.continue_cloud_video_run('local', run.id)
        assert 'checkpoint' in str(e.value).lower()


def test_a_still_running_video_run_cannot_be_continued(app, tmp_path, monkeypatch):
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='training')
        _saves(run, tmp_path, ['video_surf_000000400.safetensors'])
        monkeypatch.setattr(cvt, '_start_pod',
                            lambda r: pytest.fail('a pod was rented anyway'))
        with pytest.raises(ValueError):
            cvt.continue_cloud_video_run('local', run.id)


def test_the_relaunch_routes_refuse_a_run_of_the_other_table(app, client, tmp_path):
    """These two POST endpoints RENT A GPU. Reached from a video dataset's page
    with a face run's id — the collision, again — they must not relaunch it."""
    with app.app_context():
        _face_dataset('portraits')
        vid = _video_dataset(tmp_path, 'surf clips')
        face_run = _run(vid.id, status='error')             # NULL table = face
        vid_id, face_run_id = vid.id, face_run.id
    for verb in ('retry', 'continue'):
        r = client.post(f'/api/video-dataset/{vid_id}/train/cloud/{verb}',
                        json={'run_id': face_run_id})
        assert r.status_code == 404, (verb, r.get_data(as_text=True))


def test_the_continue_route_relaunches_through_the_video_lane(
        app, client, tmp_path, monkeypatch):
    from app.services import cloud_video_training as cvt
    seen = {}
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done', steps=500)
        _saves(run, tmp_path, ['video_surf_000000400.safetensors'])
        vid_id, run_id = vid.id, run.id
    monkeypatch.setattr(cvt, 'launch_cloud_video_training',
                        lambda *a, **k: seen.update(kw=k) or {'run_id': 8})
    r = client.post(f'/api/video-dataset/{vid_id}/train/cloud/continue',
                    json={'run_id': run_id, 'extra_steps': 250})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()['run_id'] == 8
    assert seen['kw']['steps'] == 650                     # 400 harvested + 250


def test_a_continue_with_nothing_to_resume_from_is_a_400_not_a_500(
        app, client, tmp_path):
    """It is a state the user can act on ("launch a fresh run"), not a server
    fault — and a 500 would render as an unexplained failure."""
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done')
        vid_id, run_id = vid.id, run.id
    r = client.post(f'/api/video-dataset/{vid_id}/train/cloud/continue',
                    json={'run_id': run_id})
    assert r.status_code == 400
    assert 'checkpoint' in r.get_json()['error'].lower()


# ── 4. Saying what a run was made of ──────────────────────────────────────────

def test_a_harvested_video_run_writes_a_manifest_beside_its_weights(
        app, tmp_path, monkeypatch):
    """The face lane files a `TrainingRunRecord`, whose manifest is a list of
    face-dataset IMAGES and whose `dataset_id` is a face id — a video run filed
    there would sit in another dataset's lineage graph for good. So the video
    lane records its own provenance where its weights are: which dataset, which
    target profile, which run, which files. A folder of `.safetensors` found on
    a disk a year later can then still say what it is."""
    from app.services import video_run_lineage as vrl
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips', profile='minimax_h3',
                             frames=107)
        run = _run(vid.id, crd.VIDEO, status='done', steps=500)
        d = _saves(run, tmp_path, ['video_surf_000000250.safetensors'])
        run.train_params = json.dumps({'steps': 500, 'target_profile': 'minimax_h3',
                                       'base_model': 'org/h3'})
        from app.extensions import db
        db.session.commit()
        monkeypatch.setattr(vrl, 'lineage_dir', lambda r: str(d))
        vrl.record(run)
        written = json.loads((d / vrl.MANIFEST_NAME).read_text(encoding='utf-8'))
    assert written['run_id'] == run.id
    assert written['dataset_table'] == crd.VIDEO
    assert written['dataset_id'] == vid.id
    assert written['dataset_name'] == 'surf clips'
    assert written['target_profile'] == 'minimax_h3'
    assert written['base_model'] == 'org/h3'
    assert written['files'] == ['video_surf_000000250.safetensors']


def test_the_manifest_never_carries_a_machine_path(app, tmp_path, monkeypatch):
    """It sits next to weights people share. Filenames yes, absolute paths no —
    the same rule every diagnostic surface in this app already follows."""
    from app.services import video_run_lineage as vrl
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done')
        d = _saves(run, tmp_path, ['video_surf_000000250.safetensors'])
        monkeypatch.setattr(vrl, 'lineage_dir', lambda r: str(d))
        vrl.record(run)
        raw = (d / vrl.MANIFEST_NAME).read_text(encoding='utf-8')
    assert str(tmp_path) not in raw
    assert ':\\' not in raw and '/home/' not in raw


def test_a_continued_video_run_names_the_run_it_grew_from(
        app, tmp_path, monkeypatch):
    """Genealogy for the video lane, kept inside the video lane: the child
    stamps its parent's RUN id, not a face `TrainingRunRecord` id. Without it a
    3000-step LoRA made of three continuations looks like three unrelated runs."""
    from app.services import cloud_video_training as cvt
    seen = {}
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='done', steps=500)
        _saves(run, tmp_path, ['video_surf_000000400.safetensors'])
        monkeypatch.setattr(cvt, 'launch_cloud_video_training',
                            lambda *a, **k: seen.update(kw=k) or {'run_id': 7})
        cvt.continue_cloud_video_run('local', run.id, extra_steps=100)
        assert seen['kw']['parent_run_id'] == run.id


def test_the_parent_run_id_is_stamped_on_the_child_row(app, tmp_path, monkeypatch):
    from app.services import cloud_video_training as cvt
    with app.app_context():
        vid = _video_dataset(tmp_path)
        out = cvt.launch_cloud_video_training(
            'local', vid.id, steps=300, parent_run_id=41,
            _provision=lambda run: None)
        from app.models import CloudTrainingRun
        child = db.session.get(CloudTrainingRun, out['run_id'])
        assert json.loads(child.train_params)['parent_run_id'] == 41


# ── 5. Does the rented pod know how to read an mp4 at all ─────────────────────

def test_the_probe_asks_the_pod_about_a_clip_that_actually_reached_it(app):
    """The probe is only worth its command if it reads a file from the UPLOADED
    folder with the decoder the TRAINER uses. ai-toolkit's video loader opens
    clips with `cv2.VideoCapture` and falls back to PyAV for what OpenCV cannot
    decode (toolkit/dataloader_mixins.py) — so the command names both, in that
    order, and points at the pod-side dataset directory."""
    from app.services import pod_video_probe as pvp
    cmd = pvp.build_probe_command('/workspace/datasets/lds9_video_surf',
                                  want_audio=False)
    assert '/workspace/datasets/lds9_video_surf' in cmd
    assert 'cv2' in cmd and 'av' in cmd
    assert '.mp4' in cmd


def test_a_pod_that_cannot_decode_the_clips_refuses_before_the_job_starts(
        app, tmp_path, monkeypatch):
    """Run #138 paid for a pod whose upload phase nobody verified. This is the
    same lesson one step later: the refusal must land BEFORE `start_job`, when
    the bill is minutes of boot rather than hours of a job producing nothing."""
    from app.services import cloud_training as ct
    from app.services import pod_video_probe as pvp
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='uploading')
        run.job_name = 'lds9_video_surf'
        from app.extensions import db
        db.session.commit()
        monkeypatch.setattr(
            pvp, 'probe_decoder',
            lambda *a, **k: (_ for _ in ()).throw(
                pvp.PodDecoderUnusable('neither OpenCV nor PyAV decoded clip_0001.mp4')))
        with pytest.raises(RuntimeError) as e:
            ct._assert_pod_can_decode(run, object(),
                                      {'DATASETS_FOLDER': '/workspace/datasets'})
        assert 'decode' in str(e.value).lower()


def test_a_check_that_cannot_run_does_not_ground_the_run(app, tmp_path, monkeypatch):
    """A probe that FAILED and a probe that could not be RUN are opposite facts,
    and only the first is about this pod's clips.

    vast's remote-exec endpoint accepts `ls`, `rm` and `du` and nothing else, so
    the probe program never reaches a pod rented there — measured on run #165,
    where it ended the run a minute after boot with the money spent and the
    dataset already uploaded. A guard that turns a missing capability of the
    provider into a failed launch takes the whole lane down for as long as the
    restriction lasts, which is the exact opposite of what the guard is for. The
    launch carries on, no blinder than it was before the probe existed, and the
    phase says so rather than claiming a check that never happened."""
    from app.services import cloud_training as ct
    from app.services import pod_video_probe as pvp
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='uploading')
        run.job_name = 'lds9_video_surf'
        from app.extensions import db
        db.session.commit()
        monkeypatch.setattr(
            pvp, 'probe_decoder',
            lambda *a, **k: (_ for _ in ()).throw(
                pvp.PodProbeUnavailable('vast will not run this command')))
        assert ct._assert_pod_can_decode(
            run, object(), {'DATASETS_FOLDER': '/workspace/datasets'}) is None
        assert 'unavailable' in (run.phase_detail or '').lower()


def test_a_pod_that_decodes_fine_lets_the_run_through(app, tmp_path, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import pod_video_probe as pvp
    asked = {}
    with app.app_context():
        vid = _video_dataset(tmp_path, 'surf clips')
        run = _run(vid.id, crd.VIDEO, status='uploading')
        run.job_name = 'lds9_video_surf'
        from app.extensions import db
        db.session.commit()
        monkeypatch.setattr(pvp, 'probe_decoder',
                            lambda *a, **k: asked.update(k) or
                            {'ok': True, 'decoder': 'cv2', 'frames': 81,
                             'clip': 'clip_0001.mp4'})
        ct._assert_pod_can_decode(run, object(),
                                  {'DATASETS_FOLDER': '/workspace/datasets'})
    assert asked['pod_dataset_dir'] == '/workspace/datasets/lds9_video_surf'


def test_an_audio_target_makes_the_probe_demand_an_audio_track(
        app, tmp_path, monkeypatch):
    """MiniMax H3 trains on the clip's audio (`audio.muxed` in its target
    profile). A pod that decodes the video and silently finds no audio stream
    trains a video-only LoRA under an audio target's name — so the probe asks
    for the track when, and only when, the profile says it is trained on."""
    from app.services import cloud_training as ct
    from app.services import pod_video_probe as pvp
    asked = {}
    with app.app_context():
        vid = _video_dataset(tmp_path, 'h3 clips', profile='minimax_h3',
                             frames=107)
        run = _run(vid.id, crd.VIDEO, status='uploading')
        run.job_name = 'lds9_video_h3'
        from app.extensions import db
        db.session.commit()
        monkeypatch.setattr(pvp, 'probe_decoder',
                            lambda *a, **k: asked.update(k) or {'ok': True})
        ct._assert_pod_can_decode(run, object(),
                                  {'DATASETS_FOLDER': '/workspace/datasets'})
    assert asked['want_audio'] is True


def test_a_face_run_is_never_probed_for_a_video_decoder(app, tmp_path, monkeypatch):
    """It uploads jpegs. Spending a pod command — and a refusal — on a decoder it
    will never call would be a new way for an image run to fail."""
    from app.services import cloud_training as ct
    from app.services import pod_video_probe as pvp
    with app.app_context():
        face = _face_dataset('portraits')
        run = _run(face.id, status='uploading')
        monkeypatch.setattr(pvp, 'probe_decoder', lambda *a, **k: pytest.fail(
            'a face run was probed for a video decoder'))
        ct._assert_pod_can_decode(run, object(),
                                  {'DATASETS_FOLDER': '/workspace/datasets'})


def test_the_probe_reports_which_decoder_answered(app, monkeypatch):
    """The verdict is read from the pod's ONE result line, exactly like the Hub
    transfers — `dense_pod_hub.run_program` is the shared executor, so there is
    one idea in this app of what a failure on a rented pod looks like."""
    from app.services import dense_pod_hub as dph
    from app.services import pod_video_probe as pvp
    monkeypatch.setattr(pvp, '_run_program',
                        lambda *a, **k: {'ok': True, 'decoder': 'pyav',
                                         'frames': 81, 'clip': 'clip_0001.mp4'})
    out = pvp.probe_decoder(object(), instance_id='i-1',
                            pod_dataset_dir='/workspace/datasets/j',
                            tmp_dir='.', want_audio=False)
    assert out['decoder'] == 'pyav'
    assert pvp.RESULT_PREFIX == dph.RESULT_PREFIX


def test_a_pod_program_failure_becomes_a_named_decoder_refusal(app, monkeypatch):
    """`run_program` raises `PodHubError` for everything from a missing result
    line to a non-zero verdict. Letting that name reach the run's error field
    would tell a user with a video dataset that Hugging Face went wrong."""
    from app.services import dense_pod_hub as dph
    from app.services import pod_video_probe as pvp
    monkeypatch.setattr(pvp, '_run_program', lambda *a, **k: (_ for _ in ()).throw(
        dph.PodHubError('no .mp4 reached the pod')))
    with pytest.raises(pvp.PodDecoderUnusable) as e:
        pvp.probe_decoder(object(), instance_id='i-1',
                          pod_dataset_dir='/workspace/datasets/j', tmp_dir='.')
    assert 'mp4' in str(e.value)


def test_a_deleted_dataset_s_runs_never_attach_to_the_id_s_next_owner(
        app, tmp_path, monkeypatch):
    """SQLite reuses rowids on tables without AUTOINCREMENT, so a fresh video
    dataset can legitimately wear a deleted one's id — and ownership by
    (id, table) alone then hands it the dead dataset's cloud history, weights
    download included (seen live: a new stills set displayed a deleted smoke
    set's run #166). Deletion now stamps the runs, and `owns()` refuses stamped
    runs forever; the column stays NOT NULL as shipped, so the stamp rides the
    params."""
    from app.services import video_bank_service as svc
    with app.app_context():
        vid = _video_dataset(tmp_path, 'doomed')
        run = _run(vid.id, crd.VIDEO, status='done')
        old_id = vid.id
        assert crd.owns(run, old_id, crd.VIDEO)
        assert svc.delete_video_dataset('local', old_id)
        db.session.refresh(run)
        assert json.loads(run.train_params)['dataset_deleted'] is True
        # The successor wearing the same integer gets NOTHING of the ghost's.
        assert not crd.owns(run, old_id, crd.VIDEO)

