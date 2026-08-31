"""🎬 Video Test Studio API — one clip at a time, from a video LoRA.

The image studio runs a GRID: a dozen cells cost a dozen seconds and the answer
is in the contact sheet. Video does not work that way. A clip is minutes, so the
same shape here would be half an hour of waiting before the first thing anyone
could look at. This lane therefore queues ONE clip per launch and keeps a
history, and "compare" means playing two rows of that history side by side.

Everything else is deliberately the image studio's: the enqueue goes through the
same queue manager, the missing-asset refusal is the same structured 409, and
the completion callback rides the same worker thread. A user should not be able
to tell there are two services behind the app.

No login — single local user (`cfg.LOCAL_USER`), like every other blueprint here.
"""
import logging
import os
import uuid

from flask import Blueprint, jsonify, request, send_file

from ..config import LOCAL_USER
from ..services import lora_test_studio as lts
from ..services import video_test_studio as vts
from ._common import (_map_error, _require_comfyui, _require_no_stalled_comfyui,
                      _studio_missing_response)

logger = logging.getLogger(__name__)

bp = Blueprint('video_studio', __name__, url_prefix='/api/video-studio')


def _clip_dict(clip):
    """One clip row as the panel reads it.

    Every setting that shaped the clip travels with it, because the whole point
    of a history here is answering "what was different about that one" without
    reading a ComfyUI graph.
    """
    return {
        'id': clip.id, 'status': clip.status, 'error': clip.error,
        'filename': clip.filename, 'prompt': clip.prompt, 'mode': clip.mode,
        'seed': clip.seed, 'steps': clip.steps, 'frames': clip.frames,
        'megapixels': clip.megapixels, 'fps': clip.fps,
        'base_model': clip.base_model, 'lora': clip.lora,
        'lora_strength': clip.lora_strength, 'turbo': bool(clip.turbo),
        'sparse': clip.sparse or '', 'latent_upscale': bool(clip.latent_upscale),
        'eros': (clip.base_model == vts.BASE_EROS),
        'rating': clip.rating, 'run_id': clip.run_id,
        'dataset_id': clip.dataset_id,
        'created_at': clip.created_at.isoformat() if clip.created_at else None,
        'seconds': (round((clip.frames - 1) / clip.fps, 2)
                    if clip.frames and clip.fps else None),
    }


@bp.get('/options')
def video_studio_options():
    """Everything the panel would otherwise hard-code.

    Clip lengths, the sparse levels, the megapixel bounds and the step defaults
    all live in the service (and, for lengths and fps, in the shared target
    catalogue behind it). Publishing them keeps the two halves from drifting:
    a front-end that restates `22, 39, 56…` is a front-end that will still offer
    them the day the catalogue changes.

    Also says whether the third-party 10Eros weight is actually on this disk, so
    the checkbox can be offered as unavailable rather than silently ignored.
    """
    profile = vts._profile()
    # ONE probe for the whole payload: which node classes this ComfyUI
    # registers decides both the per-option availability and whether Sage will
    # be in the graph. Asking twice could answer differently in the same reply.
    classes = vts.registered_classes()
    missing = vts.missing_weights()
    return jsonify({
        # What this machine is still missing, and what Setup can do about it.
        # `action` is a setup_installer action name, so the Setup screen turns
        # each row into its own button; None means the app will not fetch that
        # file and `place_in` says where to put it by hand.
        'missing_weights': missing,
        'ready': vts.studio_ready(missing),
        'options_available': vts.option_availability(classes),
        'sage': vts.sage_available(classes),
        'frame_choices': list(profile.get('frame_choices') or ()),
        # The catalogue's own default is a TRAINING clip length (39 frames,
        # 1.6 s). Publishing it here would open the studio on a clip too short
        # to judge motion in, so the generation default is the studio's own and
        # the training one is published beside it rather than in its place.
        'frame_default': vts.FRAMES_DEFAULT,
        'training_frame_default': profile.get('frame_default'),
        'fps': profile.get('fps'),
        'frames_min': vts.FRAMES_MIN, 'frames_max': vts.FRAMES_MAX,
        'megapixels': {'min': vts.MP_MIN, 'max': vts.MP_MAX,
                       'default': vts.MP_DEFAULT},
        'sparse_modes': list(vts.SPARSE_MODES),
        'turbo_steps': vts.TURBO_STEPS, 'default_steps': vts.DEFAULT_STEPS,
        'base_official': vts.BASE_OFFICIAL, 'base_eros': vts.BASE_EROS,
        'eros_available': vts.eros_on_disk(),
    })


@bp.get('/loras')
def video_studio_loras():
    """What can be tested: LoRAs already visible to ComfyUI, and trained runs.

    Two lists rather than one, because they are two different actions. A
    deployed LoRA is one click from a clip; a trained checkpoint has to be
    copied into ComfyUI first, and pretending otherwise would hide a 300 MB
    file operation behind a Generate button.
    """
    return jsonify({'deployed': vts.deployed_loras(),
                    'trained': vts.trained_loras()})


@bp.post('/deploy')
def video_studio_deploy():
    """Copy one trained checkpoint where ComfyUI can load it."""
    data = request.get_json(silent=True) or {}
    try:
        name = vts.deploy_checkpoint(data.get('run_id'), data.get('filename'))
    except (ValueError, TypeError) as exc:
        return _map_error(exc)
    except OSError as exc:
        logger.exception('video studio: deploy failed')
        return jsonify({'ok': False,
                        'error': f'Could not copy the checkpoint into ComfyUI: {exc}'}), 500
    return jsonify({'ok': True, 'filename': name})


@bp.post('/source')
def video_studio_source():
    """Stage the i2v start image into ComfyUI's input folder.

    Three ways in, because a video LoRA is tested against three different kinds
    of picture and making the user export to disk first would be busywork:

      * an UPLOAD (multipart `image`) — the general case;
      * a BANK image (`bank_id` + `image_id`) — animating the very portrait the
        LoRA was trained from;
      * the FIRST FRAME of a dataset clip (`dataset_id` + `filename`) — the
        honest baseline, since that frame is material the LoRA actually saw.

    Whatever the route in, the file lands through `stage_input_image`, which
    strips EXIF/GPS and bounds the decode: ComfyUI's input folder may be a
    different machine, so it is a disclosure boundary rather than a copy.

    Returns the staged NAME (what the graph's LoadImage will reference) and the
    aspect ratio, which the latent upscale needs to size its target.
    """
    try:
        src_path, cleanup = _resolve_source(request)
    except ValueError as exc:
        return _map_error(exc)
    try:
        from ..utils import comfy_fs
        from .. import config as cfg
        input_dir = comfy_fs.ensure_input_usable(cfg.comfyui_dir('input'))
        dest = f'lds_vstudio_{uuid.uuid4().hex[:10]}.png'
        staged = comfy_fs.stage_input_image(src_path, dest, input_dir)
        ratio = _image_ratio(staged)
    except ValueError as exc:
        return _map_error(exc)
    except Exception as exc:                      # noqa: BLE001
        logger.exception('video studio: staging the source image failed')
        return jsonify({'ok': False, 'error': str(exc)}), 409
    finally:
        if cleanup:
            try:
                os.unlink(src_path)
            except OSError:
                pass
    return jsonify({'ok': True, 'image': dest, 'ratio': ratio})


def _resolve_source(req):
    """(path to read the picture from, whether the caller must delete it).

    Kept apart from the route so each source is one readable branch, and so the
    temp-file lifetime is explicit: only the upload branch creates one.
    """
    upload = req.files.get('image')
    if upload is not None:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix='.upload')
        with os.fdopen(fd, 'wb') as fh:
            fh.write(upload.read())
        return tmp, True

    data = req.get_json(silent=True) or {}
    if data.get('bank_id') and data.get('image_id'):
        # `resolved_image_path` and nothing else: it is THE reader-side resolver
        # (watermark-cleaned blob, manual rotation, bank-side crop), and a reader
        # calling `abs_image_path` directly would quietly animate the version the
        # user already cleaned.
        from ..models import BankImage
        from ..services import image_bank_service as banks
        bank = banks.get_bank(LOCAL_USER, int(data['bank_id']))
        row = (BankImage.query
               .filter_by(id=int(data['image_id']), bank_id=int(data['bank_id']))
               .first() if bank else None)
        path = banks.resolved_image_path(bank, row) if row else None
        if not path or not os.path.isfile(path):
            raise ValueError('that bank image is not on disk')
        return path, False

    if data.get('dataset_id') and data.get('filename'):
        return _dataset_clip_frame(int(data['dataset_id']), data['filename']), True

    raise ValueError('attach an image, or name a bank image or a dataset clip')


def _dataset_clip_frame(dataset_id, filename):
    """Decode the FIRST frame of a dataset clip to a temp PNG, full size.

    Not the bank's thumbnail: that one is capped at 480 px for a gallery, and
    feeding a 480 px still to a 1 MP generation would blame the LoRA for a
    softness the source never had.
    """
    from ..services import video_bank_service as vbs
    name = os.path.basename(str(filename or ''))
    if not name:
        raise ValueError('clip not found')
    path = os.path.join(str(vbs.dataset_dir(dataset_id)), name)
    if not os.path.isfile(path):
        raise ValueError('clip not found')
    try:
        import av
        import tempfile
    except ImportError:
        raise ValueError('reading a frame needs the video extras — install them '
                         'from the Setup screen')
    fd, tmp = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frame.to_image().convert('RGB').save(tmp, 'PNG')
                return tmp
    except Exception as exc:                      # noqa: BLE001 — any decode error
        raise ValueError(f'could not read a frame from that clip: {exc}')
    raise ValueError('that clip has no decodable frame')


def _image_ratio(path):
    """width / height of the staged picture, or None when it cannot be read.

    None is a fine answer: the only consumer is the latent upscale's target
    size, which falls back to the node's own defaults.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            return round(im.width / max(1, im.height), 4)
    except Exception:                             # noqa: BLE001
        return None


@bp.post('/generate')
def video_studio_generate():
    """Queue one clip.

    Gated like the image studio's `/run`: ComfyUI has to be reachable and must
    not be sitting on a stalled prompt, because a job queued behind a wedged one
    looks exactly like a job that is simply slow — and here "simply slow" is a
    plausible five minutes.
    """
    blocked = _require_comfyui() or _require_no_stalled_comfyui()
    if blocked:
        return blocked
    data = request.get_json(silent=True) or {}
    prompt = str(data.get('prompt') or '').strip()
    mode = 't2v' if str(data.get('mode') or 'i2v').lower() == 't2v' else 'i2v'
    image = data.get('image')
    if mode == 'i2v' and not image:
        return jsonify({'ok': False,
                        'error': 'Pick a start image, or switch to text-to-video.'}), 400
    if not prompt:
        return jsonify({'ok': False,
                        'error': 'Describe the motion you want to see.'}), 400
    lora = data.get('lora') or None
    if lora and lts._is_unsafe_external_lora_name(lora):
        # The same guard the image studio applies to a LoRA name: this string
        # reaches a loader that resolves it under the loras roots, and a rooted
        # or `..`-bearing name is the one shape that walks out of them.
        return jsonify({'ok': False, 'error': 'invalid LoRA name'}), 400
    try:
        out = vts.enqueue_clip(
            LOCAL_USER, prompt=prompt, mode=mode, image=image, lora=lora,
            lora_strength=data.get('lora_strength', 1.0),
            run_id=data.get('run_id'), dataset_id=data.get('dataset_id'),
            seed=data.get('seed'), steps=data.get('steps'),
            frames=data.get('frames'), megapixels=data.get('megapixels',
                                                           vts.MP_DEFAULT),
            aspect=data.get('aspect', 'auto'), turbo=bool(data.get('turbo')),
            eros=bool(data.get('eros')), sparse=data.get('sparse', ''),
            latent_upscale=bool(data.get('latent_upscale')),
            source_ratio=data.get('ratio'))
    except lts.StudioAssetsMissing as exc:
        return _studio_missing_response(exc)
    except (ValueError, TypeError) as exc:
        return _map_error(exc)
    return jsonify({'ok': True, **out})


@bp.get('/clips')
def video_studio_clips():
    """The history, newest first. `limit` caps it (default 24, hard max 200)."""
    from ..models import VideoTestClip
    try:
        limit = max(1, min(200, int(request.args.get('limit', 24))))
    except (TypeError, ValueError):
        limit = 24
    rows = (VideoTestClip.query.order_by(VideoTestClip.id.desc())
            .limit(limit).all())
    return jsonify({'clips': [_clip_dict(c) for c in rows]})


@bp.get('/clip/<int:clip_id>')
def video_studio_clip(clip_id):
    """One clip — what the panel polls while a job is in flight."""
    from ..models import VideoTestClip
    clip = VideoTestClip.query.filter_by(id=clip_id).first()
    if clip is None:
        return jsonify({'error': 'clip not found'}), 404
    return jsonify(_clip_dict(clip))


@bp.get('/clip/<int:clip_id>/video')
def video_studio_clip_media(clip_id):
    """The mp4 itself.

    `conditional=True` for the same reason as the bank's rushes: without a
    206-capable response the player downloads the whole clip before it can seek
    a single second of it.
    """
    from ..models import VideoTestClip
    clip = VideoTestClip.query.filter_by(id=clip_id).first()
    if clip is None or not clip.filename:
        return jsonify({'error': 'clip not available'}), 404
    path = os.path.join(str(vts.clips_dir()), os.path.basename(clip.filename))
    if not os.path.isfile(path):
        return jsonify({'error': 'clip file not found'}), 404
    return send_file(path, mimetype='video/mp4', conditional=True, max_age=0)


@bp.post('/clip/<int:clip_id>/rate')
def video_studio_rate(clip_id):
    """👍 / 👎 / clear — the image studio's scale, so one habit covers both."""
    from ..extensions import db
    from ..models import VideoTestClip
    clip = VideoTestClip.query.filter_by(id=clip_id).first()
    if clip is None:
        return jsonify({'error': 'clip not found'}), 404
    data = request.get_json(silent=True) or {}
    try:
        rating = int(data.get('rating', 0))
    except (TypeError, ValueError):
        rating = 0
    clip.rating = 1 if rating > 0 else (-1 if rating < 0 else 0)
    db.session.commit()
    return jsonify({'ok': True, 'rating': clip.rating})


@bp.delete('/clip/<int:clip_id>')
def video_studio_delete(clip_id):
    """Drop a clip and its file.

    The row goes even when the file cannot: a history entry pointing at nothing
    is worse than a stray mp4, and the file was in the app's own folder.
    """
    from ..extensions import db
    from ..models import VideoTestClip
    clip = VideoTestClip.query.filter_by(id=clip_id).first()
    if clip is None:
        return jsonify({'error': 'clip not found'}), 404
    if clip.filename:
        try:
            os.unlink(os.path.join(str(vts.clips_dir()),
                                   os.path.basename(clip.filename)))
        except OSError:
            pass
    db.session.delete(clip)
    db.session.commit()
    return jsonify({'ok': True})
