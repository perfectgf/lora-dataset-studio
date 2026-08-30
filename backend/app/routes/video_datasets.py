"""🎬 Video datasets API — the flat folders the trainers actually read.

A video dataset is a directory of `clip_0001.mp4` files with homonym `clip_0001.txt`
captions, and nothing else. No subfolder is ever created under it: ai-toolkit's
dataset scan is `os.walk` — recursive — and excludes only dotfiles and a directory
literally named `_controls`, so anything we wrote there for our own convenience
would be trained on without a word.

This blueprint also serves the TARGET CATALOGUE, because the frontend must not
hard-code it. Two of its fields are the difference between a good week and a
wasted one: `training_verified` (we know the model's geometry, but no trainer for
it is known to exist) and `licence_note` (MiniMax H3's licence grants no rights at
all in the EU, the UK, South Korea or the USA, and the restriction reaches the
OUTPUTS — a user must not discover that in a forum thread after building a set).
"""
import logging
import mimetypes
from ..extensions import db

from flask import Blueprint, jsonify, request, send_file

from ..config import LOCAL_USER
from ..services import video_bank_service as svc
from ..services import video_targets

logger = logging.getLogger(__name__)

bp = Blueprint('video_datasets', __name__, url_prefix='/api')


def _missing(dataset_id):
    return jsonify({'error': f'video dataset {dataset_id} not found'}), 404


@bp.get('/video/targets')
def video_targets_list():
    """The target catalogue, rendered for a picker. GET {'targets': [...]}.

    `default_seconds` is computed here rather than left to the client: "81 frames"
    means nothing to someone choosing clips out of a rush, and the intervals
    arithmetic ((frames-1)/fps, because N frames span N-1 intervals) is exactly the
    off-by-one that decides how much source a cut needs."""
    out = []
    for key in video_targets.PROFILE_KEYS:
        profile = video_targets.get(key)
        default_frames = profile['frame_default']
        out.append({
            'key': key,
            'label': profile['label'],
            'fps': profile['fps'],
            'frame_choices': list(profile['frame_choices']),
            'frame_default': default_frames,
            'default_seconds': (video_targets.clip_seconds(key, default_frames)
                                if default_frames else None),
            'size_multiple': profile['size_multiple'],
            'recommended_sizes': [list(s) for s in profile['recommended_sizes']],
            # Sizes WE verified survive the trainer's re-bucketing unchanged —
            # a separate field on purpose; recommended_sizes stays the model's
            # own claim (resolution_note quotes it back to the user as such).
            'exact_sizes': [list(s) for s in profile.get('exact_sizes', ())],
            'picker_hint': profile.get('picker_hint'),
            # Kept as a plain boolean because the picker only ever asks
            # "does this target want sound?"; `audio` carries the format
            # the exporter has to impose (32 kHz stereo for MiniMax H3).
            'keep_audio': profile['audio'] is not None,
            'audio': profile['audio'],
            'aitk_arch': profile['aitk_arch'],
            'max_pixels': profile['max_pixels'],
            'caption_style': profile['caption_style'],
            # Two vocabularies that must not be conflated: the app can know a
            # model's geometry perfectly and still have no way to train it.
            'training_verified': profile['training_verified'],
            'licence_note': profile['licence_note'],
        })
    return jsonify({'targets': out})


@bp.get('/video-datasets')
def video_datasets_list():
    """Every built video training set. GET {'datasets': [...]}"""
    return jsonify({'datasets': svc.list_video_datasets(LOCAL_USER)})


@bp.post('/video-datasets/from-dataset')
def video_dataset_from_face_dataset():
    """Build an H3 STILLS set from an existing image dataset — body
    {dataset_id, name?}. Reuses the image lane's own exporter (curated images,
    edited captions, trigger — all already there), so the two lanes cannot
    disagree about what a caption or a trigger means."""
    data = request.get_json(silent=True) or {}
    try:
        out = svc.create_stills_dataset_from_face_dataset(
            LOCAL_USER, int(data.get('dataset_id') or 0), name=data.get('name'))
    except (TypeError, ValueError) as e:
        msg = str(e) or 'dataset_id must be a number'
        return jsonify({'error': msg}), 404 if 'not found' in msg else 400
    return jsonify({'ok': True, **out}), 201


@bp.get('/video-dataset/<int:dataset_id>')
def video_dataset_get(dataset_id):
    """The dataset and its clips, each carrying the source file and the bounds it
    was cut at — so a later re-export to another target is a re-encode from the
    original rather than a re-scan from scratch."""
    payload = svc.video_dataset_payload(LOCAL_USER, dataset_id)
    if payload is None:
        return _missing(dataset_id)
    return jsonify(payload)


@bp.get('/video-dataset/<int:dataset_id>/clip/<int:clip_id>/media')
def video_dataset_clip_media(dataset_id, clip_id):
    """One promoted clip's bytes. A dataset you cannot re-watch is a list of
    filenames, and watching a cut IS how you find out the length was wrong before
    paying for a training run.

    ``conditional=True`` for the same reason as the bank's source route (Range),
    though these files are a few megabytes rather than a rush.

    Cached for a DAY, not a year, and the difference is not caution: SQLite reuses
    rowids after a delete unless the column is AUTOINCREMENT, so
    /video-dataset/7/clip/12/media can legitimately become a different clip. A day
    covers a working session without pinning a stale clip to that URL forever."""
    path = svc.dataset_clip_media_path(LOCAL_USER, dataset_id, clip_id)
    if path is None:
        return jsonify({'error': 'clip file not available'}), 404
    # A stills set serves images through the same route; the extension decides.
    guessed = mimetypes.guess_type(path)[0] or 'video/mp4'
    return send_file(path, mimetype=guessed, conditional=True, max_age=86400)


@bp.post('/video-dataset/<int:dataset_id>/clip/<int:clip_id>/caption')
def video_dataset_caption(dataset_id, clip_id):
    """Body {caption}. Writes the row AND rewrites the .txt sidecar.

    The disk write is the feature, not the bookkeeping: the trainer never reads
    our database, it reads the file next to the .mp4. A caption saved to one and
    not the other trains the dataset on the previous text while the interface
    shows the new one, with nothing anywhere to reveal it.

    An empty caption empties the file; it never deletes it. A MISSING sidecar
    crashes musubi-tuner (FileNotFoundError out of a worker future, no handler on
    the path) and makes diffusion-pipe drop the clip in silence."""
    data = request.get_json(silent=True) or {}
    out = svc.set_dataset_clip_caption(LOCAL_USER, dataset_id, clip_id,
                                       data.get('caption'))
    if out is None:
        return _missing(dataset_id)
    return jsonify(out)


@bp.post('/video-dataset/<int:dataset_id>/references')
def video_dataset_references(dataset_id):
    """Attach 1-4 identity reference images (multipart field `files`). Replaces
    the previous set whole. 400 names every refusal; the target that needs
    them is the only one that accepts them."""
    files = request.files.getlist('files')
    images = [(f.filename, f.read()) for f in files if f and f.filename]
    try:
        out = svc.set_dataset_references(LOCAL_USER, dataset_id, images)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if out is None:
        return _missing(dataset_id)
    return jsonify({'ok': True, **out})


@bp.delete('/video-dataset/<int:dataset_id>')
def video_dataset_delete(dataset_id):
    """Throw away a badly cut dataset — the ENCODE, never the triage.

    The bank's clips survive untouched; they only stop claiming to have been
    promoted, so the user can re-cut at a different length without re-triaging."""
    if not svc.delete_video_dataset(LOCAL_USER, dataset_id):
        return _missing(dataset_id)
    return jsonify({'ok': True})


@bp.post('/video-dataset/<int:dataset_id>/train')
def video_dataset_train_local(dataset_id):
    """Train a LoRA on this video dataset with the ai-toolkit installed here.

    Its own endpoint, and not the face lane's `/dataset/<id>/train`, for the same
    reason the cloud one is separate: that route's id means a `face_dataset`, and
    the two tables share one integer space.

    Three refusals get their own status because the UI has to say three different
    things: an uncatalogued or unsupported target is a 400 (a choice the user can
    change), no ai-toolkit and a card already taken are 409s (the request is fine,
    the machine is not), and absent weights are a 409 carrying the repository and
    the size so the panel can ask for a yes instead of just saying no."""
    from ..services import video_training
    from ..services import video_training_local as vtl
    from ..gpu_window import GpuBusyError
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(vtl.start_video_training(
            LOCAL_USER, dataset_id,
            steps=body.get('steps') or 1000,
            base_model=(body.get('base_model') or '').strip() or None,
            low_vram=bool(body.get('low_vram', True)),
            do_i2v=bool(body.get('do_i2v', False)),
            accept_download=bool(body.get('accept_download', False))))
    except vtl.VideoWeightsMissing as e:
        return jsonify({'error': str(e), 'needs_download': True,
                        'repo': e.repo, 'gigabytes': e.gigabytes,
                        # None when the drive could not be measured. The panel
                        # must render that as silence, not as zero and not as
                        # room — the two read as opposite answers.
                        'free_gigabytes': e.free_gigabytes}), 409
    except video_training.VideoTrainingUnsupported as e:
        return jsonify({'error': str(e)}), 400
    except GpuBusyError as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        if 'not found' in str(e):
            return _missing(dataset_id)
        # 'a training is already in progress' is a state refusal, not a malformed
        # request — the same 409 the face lane's launch route returns for it.
        if 'in progress' in str(e):
            return jsonify({'error': str(e)}), 409
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 409


@bp.get('/video-dataset/<int:dataset_id>/train/progress')
def video_dataset_train_progress(dataset_id):
    """The local run's live progress, for the card to poll.

    Reports `active` only for a run whose fence names THIS dataset AND the video
    table — a face training of the colliding id must not drive this bar."""
    from ..services import video_training_local as vtl
    try:
        progress = vtl.video_training_progress(dataset_id, LOCAL_USER)
    except ValueError:
        return _missing(dataset_id)
    progress['checkpoints'] = vtl.list_run_checkpoints(dataset_id, LOCAL_USER)
    return jsonify(progress)


@bp.post('/video-dataset/<int:dataset_id>/train/stop')
def video_dataset_train_stop(dataset_id):
    """Stop the local run of THIS video dataset.

    Names the table alongside the id, which is what stops this button from
    killing the face dataset of the same number. `ok: false` is the honest answer
    when the fence names another run — the click was refused, not silently
    ignored."""
    from ..services import cloud_run_dataset as crd
    from ..services import lora_training as lt
    stopped = lt.stop_training(expected_dataset_id=dataset_id,
                               expected_dataset_table=crd.VIDEO)
    return jsonify({'ok': bool(stopped)})


@bp.post('/video-dataset/<int:dataset_id>/train/cloud')
def video_dataset_train_cloud(dataset_id):
    """Rent a pod and train a LoRA on this video dataset.

    Deliberately its own endpoint rather than the face lane's
    `/dataset/<id>/train/cloud`: that route's id means a `face_dataset`, and the
    two tables share one integer space — the same URL shape for both would make
    the id alone ambiguous at the outermost layer, which is precisely the
    confusion the run's `dataset_table` column exists to end.

    A target we have no verified base for is a 400, not a 500: the user picked a
    model this build cannot train unattended, and that is a choice they can
    correct — the message names what to do."""
    from ..services import cloud_video_training as cvt
    from ..services import video_training
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(cvt.launch_cloud_video_training(
            LOCAL_USER, dataset_id,
            steps=body.get('steps') or 1000,
            base_model=(body.get('base_model') or '').strip() or None,
            low_vram=bool(body.get('low_vram', False)),
            do_i2v=bool(body.get('do_i2v', False)),
            sample_prompts=body.get('sample_prompts'),
            distillation=body.get('distillation') or 'auto',
            gpu_name=body.get('gpu_name')))
    except video_training.VideoTrainingUnsupported as e:
        return jsonify({'error': str(e)}), 400
    except ValueError as e:
        # 'video dataset not found' is the only ValueError that is a 404; the
        # rest ('no clips on disk') are things about THIS dataset the user can
        # act on, and a 404 would tell them the dataset does not exist.
        if 'not found' in str(e):
            return _missing(dataset_id)
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        # The launch guard: already running, fleet limit, budget. 409 — the
        # request was well-formed, the state refuses it.
        return jsonify({'error': str(e)}), 409


@bp.get('/video-dataset/<int:dataset_id>/train/cloud/offers')
def video_dataset_cloud_offers(dataset_id):
    """Live GPU tiers for the launch — price/h, VRAM, and a rough time+cost per
    class. Read-only: rents nothing. Estimates are one-measured-run rough and
    the payload labels them so."""
    from ..services import cloud_video_training as cvt
    try:
        data = cvt.video_gpu_tiers(LOCAL_USER, dataset_id,
                                   steps=request.args.get('steps', type=int))
    except ValueError as e:
        if 'not found' in str(e):
            return _missing(dataset_id)
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 409
    return jsonify({'ok': True, **data})


@bp.get('/video-dataset/<int:dataset_id>/train/cloud/progress')
def video_dataset_train_cloud_progress(dataset_id):
    """The newest cloud run OF THIS VIDEO DATASET, for the page to poll.

    Scoped to the video table explicitly. Resolved by integer alone it would
    return the face dataset of the same id's run — the same phase, cost and
    progress bar, for a training the user is not watching."""
    from ..services import cloud_run_dataset as crd
    from ..services import cloud_training as ct
    run = ct.latest_run_for(dataset_id, dataset_table=crd.VIDEO)
    if run is None:
        return jsonify({'run_id': None, 'status': None})
    return jsonify({
        'run_id': run.id, 'status': run.status,
        'phase_detail': run.phase_detail or '',
        'gpu': run.gpu_name, 'price_per_hour': run.price_per_hour,
        'error': run.error,
        'steps': ct._run_param(run, 'steps'),
        'saves': len(ct.run_checkpoint_files(run)),
        'created_at': run.created_at.isoformat() if run.created_at else None,
        'finished_at': run.finished_at.isoformat() if run.finished_at else None,
    })


def _video_run(dataset_id, run_id):
    """One cloud run OF THIS VIDEO DATASET, or None.

    The ownership test is the PAIR (id, table), never the id alone: a face run
    carrying the same integer is a different training on someone else's data,
    and these three routes serve its weights and relaunch it."""
    from ..models import CloudTrainingRun
    from ..services import cloud_run_dataset as crd
    try:
        run = db.session.get(CloudTrainingRun, int(run_id))
    except (TypeError, ValueError):
        return None
    return run if run and crd.owns(run, dataset_id, crd.VIDEO) else None


@bp.get('/video-dataset/<int:dataset_id>/train/cloud/checkpoints')
def video_dataset_cloud_checkpoints(dataset_id):
    """Every LoRA this dataset's cloud runs brought back, grouped by run then by
    STEP.

    Grouped by step and not by file, because a Wan 2.2 checkpoint IS two files —
    `_high_noise` and `_low_noise` — and a list of individual files invites a UI
    to offer half a LoRA. MiniMax H3 has one file per step (ai-toolkit's
    `MinimaxH3Model` defines no `save_lora`, so the generic single-file save
    applies), and the same shape carries it without a special case."""
    from ..models import CloudTrainingRun
    from ..services import cloud_run_dataset as crd
    from ..services import cloud_training as ct
    from ..services import cloud_video_training as cvt
    if not svc.get_video_dataset(LOCAL_USER, dataset_id):
        return _missing(dataset_id)
    groups = []
    for run in (CloudTrainingRun.query.filter_by(dataset_id=dataset_id)
                .order_by(CloudTrainingRun.id.desc()).all()):
        if not crd.owns(run, dataset_id, crd.VIDEO):
            continue
        steps = cvt.harvested_steps(run)
        if not steps:
            continue
        groups.append({
            'run_id': run.id, 'status': run.status,
            'active': run.status in ct.ACTIVE_STATES,
            'gpu': run.gpu_name, 'price_per_hour': run.price_per_hour,
            'target_profile': ct._run_param(run, 'target_profile'),
            'parent_run_id': ct._run_param(run, 'parent_run_id'),
            'created_at': run.created_at.isoformat() if run.created_at else None,
            'finished_at': run.finished_at.isoformat() if run.finished_at else None,
            # Paths stay server-side: the client asks for a file by NAME and the
            # server resolves it against this run's own saves.
            'steps': [{'step': s['step'], 'final': s['final'],
                       'files': s['files']} for s in steps],
        })
    return jsonify({'groups': groups})


@bp.get('/video-dataset/<int:dataset_id>/train/cloud/checkpoint')
def video_dataset_cloud_checkpoint(dataset_id):
    """Download ONE harvested save of one of this dataset's cloud runs.

    Both halves of a Wan pair are fetched as two calls to this route — a single
    archive would be friendlier and would also be a second format to explain to
    every loader downstream; two files is what ai-toolkit wrote and what the
    loaders expect side by side."""
    from flask import abort
    from ..services import cloud_training as ct
    import os
    run = _video_run(dataset_id, request.args.get('run_id'))
    if not run:
        abort(404)
    # Resolved through the run's own save list, which is basename-only by
    # construction — the client can never point this at a path of its choosing.
    path = ct.run_checkpoint_path(run, request.args.get('filename'))
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True)


@bp.post('/video-dataset/<int:dataset_id>/train/cloud/retry')
def video_dataset_cloud_retry(dataset_id):
    """↻ Relaunch a failed run of this dataset on a fresh pod."""
    from ..services import cloud_video_training as cvt
    run = _video_run(dataset_id, (request.get_json(silent=True) or {}).get('run_id'))
    if not run:
        return _missing(dataset_id)
    return _relaunch(lambda: cvt.retry_cloud_video_run(LOCAL_USER, run.id))


@bp.post('/video-dataset/<int:dataset_id>/train/cloud/continue')
def video_dataset_cloud_continue(dataset_id):
    """▶ Train an existing LoRA of this dataset further, from one of its
    harvested steps."""
    from ..services import cloud_video_training as cvt
    body = request.get_json(silent=True) or {}
    run = _video_run(dataset_id, body.get('run_id'))
    if not run:
        return _missing(dataset_id)
    return _relaunch(lambda: cvt.continue_cloud_video_run(
        LOCAL_USER, run.id, extra_steps=body.get('extra_steps', 1000),
        from_step=body.get('from_step')))


def _relaunch(call):
    """The two relaunch routes answer identically, and the mapping is the same
    one the launch route uses: anything the user can act on is a 400, and the
    launch guard (already running, fleet limit, budget) is a 409 — the request
    was well-formed, the state refuses it.

    No branch for `VideoTrainingUnsupported`: it IS a ValueError, and the launch
    route only names it separately because there ValueError also carries the
    "dataset not found" 404. Here it would be a line that never runs."""
    try:
        return jsonify({'ok': True, **call()})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 409
