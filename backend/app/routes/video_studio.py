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
import io
import logging
import os
import uuid

from flask import Blueprint, jsonify, request, send_file

from ..config import LOCAL_USER
from ..gpu_window import gpu_exclusive_vision_window
from ..services import lora_test_studio as lts
from ..services import neural_render as _nr
from ..services import video_test_studio as vts
from ._common import (_map_error, _require_comfyui, _require_no_stalled_comfyui,
                      _studio_missing_response)

logger = logging.getLogger(__name__)

# ✨ How many start frames ONE vision window will write for in a row.
# Not a performance guard: it is the honest ceiling of a single held window.
# Each frame is model work during which no clip can render, so past this the
# right answer is two batches, not a longer freeze nobody can interrupt.
MAX_WRITE_BATCH = 12

# The window's TTL has to cover the WHOLE batch, not one click: a TTL cut for a
# single ✨ press would expire halfway through twelve frames and let a queued
# clip take the card out from under the writer. Generous per frame on purpose —
# a cold model on a slow machine is the case that must not trip it — and the
# window is released as soon as the loop ends.
_WRITE_TTL_PER_FRAME = 180
_WRITE_TTL_FLOOR = 600


def _write_batch_ttl(count):
    return max(_WRITE_TTL_FLOOR, _WRITE_TTL_PER_FRAME * max(1, int(count)))

bp = Blueprint('video_studio', __name__, url_prefix='/api/video-studio')


def _joined_state(clip):
    """⏭ True (the file is the joint), False (left as the part — every failure
    path writes why in `error`), None (no verdict: still rendering, or the
    join is running right now)."""
    if clip.status != 'done':
        return None
    if clip.filename and str(clip.filename).endswith('_joined.mp4'):
        return True
    if clip.error and 'continuation not joined' in str(clip.error):
        return False
    return None


def _json_or_none(text):
    if not text:
        return None
    try:
        import json
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _clip_dict(clip):
    """One clip row as the panel reads it.

    Every setting that shaped the clip travels with it, because the whole point
    of a history here is answering "what was different about that one" without
    reading a ComfyUI graph.
    """
    return {
        'id': clip.id, 'status': clip.status, 'error': clip.error,
        'filename': clip.filename, 'prompt': clip.prompt, 'mode': clip.mode,
        'aspect': getattr(clip, 'aspect', None) or 'auto',
        # The staged start frame, so ↻ Reuse can hand it back: without it a
        # reused image-to-video clip lands in i2v mode with nothing to animate
        # and Generate stays blocked — every dial restored except the one that
        # decides whether the button works at all.
        'source_image': clip.source_image,
        # ↗ The clip this one was smoothed from, so the card can say so.
        'vfi_of': getattr(clip, 'vfi_of', None),
        # ⏭ The clip this one continues (joined behind it), and whether the join happened.
        'continues_of': getattr(clip, 'continues_of', None),
        # Three states, not two: joined; left as the part, which every failure
        # path says in `error`; or no verdict yet — the part still rendering,
        # or landed seconds ago with the join under way (the completion is
        # committed before the join runs). The card's "(not joined)" is for
        # the middle one only.
        'joined': _joined_state(clip),
        # ✨ The clip this one was neural-rendered from, same reading.
        'nr_of': getattr(clip, 'nr_of', None),
        # ✨ The dials that made a neural render, or null — the pills read them.
        'nr_params': _json_or_none(getattr(clip, 'nr_params', None)),
        'seed': clip.seed, 'steps': clip.steps, 'frames': clip.frames,
        'megapixels': clip.megapixels, 'fps': clip.fps,
        'base_model': clip.base_model, 'lora': clip.lora,
        'lora_strength': clip.lora_strength, 'turbo': bool(clip.turbo),
        # ⚡ Which acceleration ran; rows older than the choice say `turbo` from the flag.
        'accel': (getattr(clip, 'accel', None) or ('turbo' if clip.turbo else '')),
        'sparse': clip.sparse or '', 'latent_upscale': bool(clip.latent_upscale),
        'eros': (clip.base_model == vts.BASE_EROS),
        'rating': clip.rating, 'run_id': clip.run_id,
        'dataset_id': clip.dataset_id,
        'created_at': clip.created_at.isoformat() if clip.created_at else None,
        'seconds': (round((clip.frames - 1) / clip.fps, 2)
                    if clip.frames and clip.fps else None),
        # ⏱ How long the queue spent on it, or null when the queue could not say.
        'render_seconds': clip.render_seconds,
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
        # ⚡ The three arena accelerations, each with whether THIS machine has it.
        'accelerations': vts.accelerations_status(),
        'base_official': vts.BASE_OFFICIAL, 'base_eros': vts.BASE_EROS,
        'eros_available': vts.eros_on_disk(),
        # ✨ DLSS 5 neural rendering — ready + the sentences naming what is
        # missing, so the clip history's button can refuse in words.
        'neural_render': _nr.status(),
        # How the running ComfyUI was started, judged against this machine's
        # RAM: None, or the flag that turns minutes per clip into seconds.
        'launch_advice': vts.launch_advice(*vts.comfyui_launch_facts()),
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


@bp.post('/clip/<int:clip_id>/vfi')
def video_studio_clip_vfi(clip_id):
    """↗ Smooth a finished clip — RIFE frame interpolation, as a new clip.

    The same recipe the maintainer's image generator uses (rife49, x2, ensemble)
    so a clip smoothed here is the clip smoothed there. A new row, never an edit
    of the original: the studio exists to compare, and overwriting the thing
    being compared would end that.
    """
    data = request.get_json(silent=True) or {}
    try:
        out = vts.interpolate_clip(LOCAL_USER, clip_id,
                                   multiplier=data.get('multiplier'))
    except (ValueError, TypeError) as exc:
        return _map_error(exc)
    return jsonify({'ok': True, **out})


@bp.get('/clip/<int:clip_id>/last-frame.png')
def video_studio_clip_last_frame_png(clip_id):
    """⏭ The clip's last frame, as the strip's preview."""
    try:
        path = vts.last_frame_png(clip_id)
    except LookupError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 404
    except (ValueError, TypeError) as exc:
        return _map_error(exc)
    resp = send_file(path, mimetype='image/png', conditional=True, max_age=0)
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@bp.post('/clip/<int:clip_id>/last-frame')
def video_studio_clip_last_frame(clip_id):
    """⏭ Stage the clip's last frame as the next start frame — the same staging
    as the Bank, Gallery and upload routes, so the graph loads it the same way.
    The reply names the clip so the launch can say it continues it."""
    try:
        png = vts.last_frame_png(clip_id)
        from ..utils import comfy_fs
        from .. import config as cfg
        input_dir = comfy_fs.ensure_input_usable(cfg.comfyui_dir('input'))
        dest = f'lds_vstudio_{uuid.uuid4().hex[:10]}.png'
        staged = comfy_fs.stage_input_image(png, dest, input_dir)
        ratio = _image_ratio(staged)
    except LookupError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 404
    except (ValueError, TypeError) as exc:
        return _map_error(exc)
    except Exception as exc:                      # noqa: BLE001
        logger.exception('video studio: staging the last frame of clip %s failed', clip_id)
        return jsonify({'ok': False, 'error': str(exc)}), 409
    return jsonify({'ok': True, 'image': dest, 'ratio': ratio, 'continues': int(clip_id),
                    'preview': f'/api/video-studio/clip/{int(clip_id)}/last-frame.png'})


def _clip_seconds(data: dict):
    """How long the clip the panel is set to will be, for the motion writer.

    The panel sends `seconds` — its own readback of the Length dial. A body
    that carries only `frames` (the launch itself) is converted here with the
    SAME arithmetic the readback uses: the snapped count, N-1 intervals at the
    target's fps. Either way the writer paces the action to the clip that
    will actually render, which is the whole point of passing it. None when
    neither is known: the writer then paces nothing rather than guessing.
    """
    if data.get('seconds') is not None:
        try:
            return float(data.get('seconds'))
        except (TypeError, ValueError):
            return None
    if data.get('frames') is None:
        return None
    fps = float(vts._profile().get('fps') or 24.0)
    return (vts.snap_frames(data.get('frames')) - 1) / fps


@bp.post('/motion/suggest')
def video_studio_motion_suggest():
    """✨ Propose the movement, by looking at the staged start frame.

    A PROPOSAL and the wording says so: the model sees a still, so it can read
    who is there and how they are posed, never what happens next. The user
    edits it like any other text. `seconds` (or `frames`) is the clip length
    the dials are set to — the proposal is paced to fill exactly that.

    Runs inside the GPU-exclusive vision window, like the image studio's
    twin (`/api/studio/describe`): the writer is a vision pass, and a vision
    pass that fights a queued clip for VRAM loses — H3 alone fills most of
    the card. The window refuses while ComfyUI has work queued or rendering
    (503, the reason in `detail`), and on entry asks ComfyUI to let go of its
    models, so the NEXT clip pays H3's load again. That cost is the app's
    standing GPU arbitration, paid once per ✨ click, not a choice of this
    route; the two routes below share it.
    """
    from ..services import video_motion_prompt as vmp
    data = request.get_json(silent=True) or {}
    try:
        with gpu_exclusive_vision_window(flag_ttl=600):
            out = vmp.suggest_from_frame(
                data.get('image'),
                instruction=data.get('instruction'),
                model=data.get('model'),
                seconds=_clip_seconds(data),
                shots=data.get('shots', 1))
    except Exception as exc:
        # Like the image studio's twin: a bad ask is 400, the Ollama/LM Studio
        # fence 409 with its code (the panel offers the unload), the window's
        # refusal 503 with its reason, any other transport failure a 409
        # sentence. The narrow clause this replaced let the fence through as
        # a bare 500 with no message to show.
        return _map_error(exc)
    return jsonify({'ok': True, 'prompt': out})


@bp.post('/motion/write-batch')
def video_studio_motion_write_batch():
    """✨ One prompt PER start frame, written inside a SINGLE vision window.

    WHY THE LOOP IS HERE AND NOT IN THE PANEL. Entering the vision window asks
    ComfyUI to let go of its models, so the next clip pays the video model's
    load again — tens of gigabytes for H3 (the cost is stated in
    `/motion/suggest`). A panel that called the single-frame writers once per
    picture would pay that reload once per picture. Holding one window for the
    whole strip pays it once, which is the entire point of writing every prompt
    BEFORE the first clip is queued.

    Body `{images: [...], prompt?, model?, seconds?|frames?, shots?}` — the same
    pieces the two ✨ buttons send. A `prompt` present means ENRICH each frame
    from it (what `/motion/enhance` does for one); absent means PROPOSE from the
    frame alone (`/motion/suggest`). The panel's own rule, applied N times.

    PARTIAL RESULTS ARE THE CONTRACT. Every entry answers `{index, image,
    prompt}` or `{index, image, error}`, and the call is 200 as long as the
    window opened: one unreadable frame must not cost the eleven others their
    prompt. Only a failure to OPEN the window is an error status — there is
    nothing partial to keep then.
    """
    from ..services import video_motion_prompt as vmp
    data = request.get_json(silent=True) or {}
    images = [str(n) for n in (data.get('images') or []) if str(n or '').strip()]
    if not images:
        return jsonify({'ok': False, 'error': 'No start frame to write for.'}), 400
    if len(images) > MAX_WRITE_BATCH:
        return jsonify({'ok': False,
                        'error': (f'{len(images)} pictures at once is more than one '
                                  f'window writes for (max {MAX_WRITE_BATCH}).')}), 400
    typed = str(data.get('prompt') or '').strip()
    seconds = _clip_seconds(data)
    shots = data.get('shots', 1)
    model = data.get('model')
    out = []
    try:
        # ONE window for the whole strip — the reason this route exists.
        with gpu_exclusive_vision_window(flag_ttl=_write_batch_ttl(len(images))):
            for i, name in enumerate(images):
                try:
                    if typed:
                        written = vmp.enhance(typed, image=name, model=model,
                                              seconds=seconds, shots=shots)
                    else:
                        written = vmp.suggest_from_frame(name, instruction='',
                                                         model=model, seconds=seconds,
                                                         shots=shots)
                    out.append({'index': i, 'image': name, 'prompt': written})
                except Exception as exc:          # noqa: BLE001
                    logger.warning('video studio: batch write failed on %s: %s', name, exc)
                    out.append({'index': i, 'image': name,
                                'error': str(exc) or exc.__class__.__name__})
    except Exception as exc:
        # The window itself: fence, refusal, transport. Nothing was written.
        return _map_error(exc)
    return jsonify({'ok': True, 'results': out,
                    'written': sum(1 for r in out if str(r.get('prompt') or '').strip()),
                    'failed': sum(1 for r in out if r.get('error'))})


@bp.post('/motion/enhance')
def video_studio_motion_enhance():
    """✨ The same intent, with more of the detail a sampler can use.

    Never destructive: the field is only written on success, and a model
    that answers nothing usable is a 409 with the sentence to show — never
    an empty prompt handed back as if it had worked.

    `image` — the staged start frame, when the panel has one — anchors the
    rewrite on the picture that will actually be animated, so an instruction
    cannot enrich the prompt with scenery the frame does not contain. Without
    one the clip is text-to-video and the writer is told so: no picture is
    referenced. `seconds` (or `frames`) paces the rewrite to the clip length.
    Same GPU-exclusive vision window as `/motion/suggest` (see there).
    """
    from ..services import video_motion_prompt as vmp
    data = request.get_json(silent=True) or {}
    original = str(data.get('prompt') or '').strip()
    try:
        with gpu_exclusive_vision_window(flag_ttl=600):
            out = vmp.enhance(data.get('prompt'), image=data.get('image'),
                              model=data.get('model'), seconds=_clip_seconds(data),
                              shots=data.get('shots', 1))
    except Exception as exc:
        return _map_error(exc)
    # `unchanged` is how the panel tells "the model had nothing to add" from
    # "the request worked": the two look identical in the field.
    return jsonify({'ok': True, 'prompt': out,
                    'unchanged': str(out or '').strip() == original})


@bp.get('/motion/models')
def video_studio_motion_models():
    """⚙ Which local models can write the motion, and which one does today."""
    from ..services import video_motion_prompt as vmp
    return jsonify(vmp.model_choices())


@bp.put('/motion/model')
def video_studio_motion_model_set():
    """⚙ Remember the model that writes the motion. Empty returns to the
    provider's own vision model."""
    from ..services import video_motion_prompt as vmp
    data = request.get_json(silent=True) or {}
    return jsonify({'ok': True, 'model': vmp.set_model(data.get('model'))})


@bp.post('/lora/import')
def video_studio_lora_import():
    """Bring a LoRA the user already has into the picker.

    Multipart `file`, or JSON `{path}` for a file on this machine — the second
    is the one that matters for a 300 MB weight, since nothing crosses HTTP.
    The picker listed only what this app trained and what was already in
    ComfyUI's folder, so anything downloaded had to be moved there by hand with
    the app open beside a file explorer.

    400 for every refusal, because all of them are things the user can fix: the
    wrong extension, an unusable name, a file that is not there, or a DIFFERENT
    weight already under that name (never overwritten — that would silently
    change what every clip made with that name meant).
    """
    upload = request.files.get('file')
    data = request.get_json(silent=True) or {}
    try:
        out = vts.import_external_lora(
            src_path=(str(data.get('path')).strip() if data.get('path') else None),
            upload=upload,
            filename=(upload.filename if upload is not None else None))
    except (ValueError, TypeError) as exc:
        return _map_error(exc)
    except OSError as exc:
        logger.exception('video studio: lora import failed')
        return jsonify({'ok': False,
                        'error': f'Could not copy that LoRA into ComfyUI: {exc}'}), 500
    return jsonify({'ok': True, **out})


@bp.post('/source')
def video_studio_source():
    """Stage the i2v start image into ComfyUI's input folder.

    Three ways in, because a video LoRA is tested against three different kinds
    of picture and making the user export to disk first would be busywork:

      * an UPLOAD (multipart `image`) — the general case;
      * a BANK image (`bank_id` + `image_id`) — animating the very portrait the
        LoRA was trained from;
      * an image from the app's own GALLERY (`gallery_image_id`) — the picture
        someone just generated, animated without a round trip through disk;
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

    if data.get('gallery_image_id'):
        # An image the app itself generated — the Gallery feed's own row id.
        # Served at full size from the dataset folder it lives in, exactly as
        # /api/dataset/<id>/img/<name> serves it, so what gets animated is the
        # picture the user is looking at rather than a thumbnail of it.
        from ..extensions import db
        from ..models import LoraTestImage
        from ..services.dataset_storage import dataset_path
        row = db.session.get(LoraTestImage, int(data['gallery_image_id']))
        if row is None or not row.filename or not row.dataset_id:
            raise ValueError('that generated image is not in the gallery any more')
        path = os.path.join(str(dataset_path(int(row.dataset_id))),
                            os.path.basename(str(row.filename)))
        if not os.path.isfile(path):
            raise ValueError('that generated image is no longer on disk')
        return path, False

    raise ValueError('attach an image, or name a bank image, a dataset clip or '
                     'a gallery image')


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


def _staged_image_ratio(name):
    """width / height of an ALREADY staged start frame, by its staged name.

    The ratio travels with the pick; a caller replaying a past clip has only
    the name. Reading it back costs one PIL open of a file this app wrote
    itself, and returns None on anything unreadable — the same fine answer
    _image_ratio gives, with the same single consumer.
    """
    safe = os.path.basename(str(name or ''))
    if not safe:
        return None
    try:
        from .. import config as cfg
        folder = cfg.comfyui_dir('input')
    except Exception:  # noqa: BLE001 — no input folder is "no ratio", not a 500
        return None
    if not folder:
        return None
    return _image_ratio(os.path.join(str(folder), safe))


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

    The answer names the `prompt` that ran (the rewrite when ✨ Enrich at
    launch worked, the text as typed when it did not) next to the `seed` the
    graph got: a batch of start frames launches its first clip, then sends
    the rest with THAT prompt and THAT seed, since the vision window is shut
    to it as soon as this clip sits in ComfyUI's queue.
    """
    blocked = _require_comfyui() or _require_no_stalled_comfyui()
    if blocked:
        return blocked
    data = request.get_json(silent=True) or {}
    prompt = str(data.get('prompt') or '').strip()
    mode = 't2v' if str(data.get('mode') or 'i2v').lower() == 't2v' else 'i2v'
    image = data.get('image')
    # ✨ Enrich at launch. Done HERE, before the graph is built, so the clip row
    # records the prompt that actually ran — a card naming a prompt the sampler
    # never read would be the one lie this screen cannot afford. A failed
    # enrichment keeps the original rather than refusing the launch: the user
    # asked for a clip, not for an essay. The writer gets what the ✨ Enrich
    # button gets: the start frame (only when one will be animated) and the
    # clip length, so the launch and the button write the same prompt.
    from ..services import video_motion_prompt as vmp
    enrich_skipped = None
    if data.get('enhance') and prompt:
        try:
            # The same GPU-exclusive vision window as the ✨ buttons (see
            # `/motion/suggest`): a clip already queued or rendering refuses
            # it, and that refusal is one more reason to launch un-enriched.
            with gpu_exclusive_vision_window(flag_ttl=600):
                prompt = vmp.enhance(prompt, image=(image if mode == 'i2v' else None),
                                     seconds=_clip_seconds(data),
                                     shots=data.get('shots', 1))
        except Exception as exc:
            # Every failure, the fence and the window included: the clip still
            # launches, and the answer carries the reason so the panel can say it.
            enrich_skipped = str(exc) or exc.__class__.__name__
            logger.warning('video studio: launch enrichment skipped: %s', exc)
    if mode == 'i2v' and not image:
        return jsonify({'ok': False,
                        'error': 'Pick a start image, or switch to text-to-video.'}), 400
    if not prompt:
        return jsonify({'ok': False,
                        'error': 'Describe the motion you want to see.'}), 400
    # The official I2V header, in code, at generation — the reference writer's
    # own rule: a prompt typed by hand or pasted from elsewhere gets the line
    # that tells the encoder the picture IS the first frame, and one the ✨
    # writers wrote, or a clip reused, is never headed twice. Text-to-video is
    # the mirror: a prompt written for a start frame and then launched without
    # one names a picture the encoder is not given — the header, the identity
    # sentence and the tag go. Done before the row is written, so the card
    # shows the prompt that ran.
    prompt = (vmp.inject_alignment_header(prompt) if mode == 'i2v'
              else vmp.strip_picture_references(prompt))
    if not vmp.has_motion(prompt):
        # Judged AFTER the rewrite, on the description alone: a prompt that
        # was nothing but the header, or labels around nothing — a clip's
        # prompt pasted back with its motion deleted — passed the check above
        # and reached the sampler empty.
        return jsonify({'ok': False,
                        'error': 'The prompt carries no motion once its header and '
                                 'labels are set aside — describe what you want to '
                                 'see move.'}), 400
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
            accel=data.get('accel'), continues=data.get('continues'),
            eros=bool(data.get('eros')), sparse=data.get('sparse', ''),
            latent_upscale=bool(data.get('latent_upscale')),
            # The ratio only sizes the latent upscale, and a client that has
            # the staged NAME but not the shape (↻ Reuse) would otherwise fall
            # back to the node's landscape defaults — turning a reused portrait
            # clip into a wide one on the upscale pass. Re-read here from the
            # file itself, which is still where the staging put it.
            source_ratio=(data.get('ratio')
                          or (_staged_image_ratio(image) if mode == 'i2v' else None)))
    except lts.StudioAssetsMissing as exc:
        return _studio_missing_response(exc)
    except (ValueError, TypeError) as exc:
        return _map_error(exc)
    if enrich_skipped:
        out = {**out, 'enrich_skipped': enrich_skipped}
    return jsonify({'ok': True, 'prompt': prompt, **out})


@bp.get('/clips')
def video_studio_clips():
    """The history, newest first. `limit` caps a page (default 24, hard max
    200); `before=<id>` pages further back.

    THE SOURCE OF A RENDER RIDES ALONG. A smoothed or neural-rendered clip
    points at the clip it was made from (`vfi_of`, `nr_of`), and that clip is
    older by construction — after a few renders it falls off the newest page,
    and the pair the studio exists to compare reads as "the original was
    deleted" (reported on the first evening). So every source of a listed
    render is appended to the page it belongs with, whatever its age, and the
    list stays newest first. `has_more` says whether a further page exists —
    judged on the page proper, not on the sources it carried along.
    """
    from ..models import VideoTestClip
    try:
        limit = max(1, min(200, int(request.args.get('limit', 24))))
    except (TypeError, ValueError):
        limit = 24
    query = VideoTestClip.query.order_by(VideoTestClip.id.desc())
    try:
        before = int(request.args.get('before', 0))
    except (TypeError, ValueError):
        before = 0
    if before > 0:
        query = query.filter(VideoTestClip.id < before)
    page = query.limit(limit).all()
    listed = {c.id for c in page}
    wanted = ({getattr(c, 'nr_of', None) for c in page} | {getattr(c, 'vfi_of', None) for c in page}
              | {getattr(c, 'continues_of', None) for c in page})
    wanted = {i for i in wanted if i and i not in listed}
    sources = (VideoTestClip.query.filter(VideoTestClip.id.in_(wanted)).all()
               if wanted else [])
    rows = sorted(page + sources, key=lambda c: c.id, reverse=True)
    # `oldest_id` is the boundary of the page PROPER. The panel keeps what it
    # loaded below that boundary and pages further back from it; a source
    # that rode along is older than the page by construction and must not
    # move the boundary — it did, and every clip loaded between the two
    # vanished at the next poll (read as "Smooth deleted my clips").
    return jsonify({'clips': [_clip_dict(c) for c in rows],
                    'has_more': len(page) == limit, 'page_size': limit,
                    'oldest_id': page[-1].id if page else 0})


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


@bp.get('/clip/<int:clip_id>/comparison')
def video_studio_clip_comparison(clip_id):
    """⬇ A finished render and the clip it came from, as ONE mp4 side by side.

    The dataset surface offers the same verb on its own rendered clips; the two
    routes differ only in where the pair comes from (a backup beside the dataset
    there, another row of the history here)."""
    from ..models import VideoTestClip
    clip = VideoTestClip.query.filter_by(id=clip_id).first()
    if clip is None or not clip.filename or not clip.nr_of:
        return jsonify({'error': 'this clip is not a neural render — nothing to compare'}), 404
    source = VideoTestClip.query.filter_by(id=clip.nr_of).first()
    if source is None or not source.filename:
        return jsonify({'error': 'the clip this render came from is gone'}), 404
    clips = str(vts.clips_dir())
    render_path = os.path.join(clips, os.path.basename(clip.filename))
    source_path = os.path.join(clips, os.path.basename(source.filename))
    try:
        data = _nr.build_comparison(source_path, render_path, left_label='Original',
                                    right_label='Neural render (DLSS 5)')
    except _nr.ComparisonBusyError as exc:
        res = jsonify({'error': str(exc)})
        res.headers['Retry-After'] = '5'
        return res, 429
    except _nr.ComparisonTooLargeError as exc:
        return jsonify({'error': str(exc)}), 413
    except _nr.NeuralRenderError as exc:
        return jsonify({'error': str(exc)}), 400
    return send_file(io.BytesIO(data), mimetype='video/mp4', as_attachment=True,
                     download_name=f'clip-{source.id}-vs-neural-{clip.id}.mp4', max_age=0)


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
    # ⏭ The last-frame cache the clip may have grown: derived from the mp4,
    # written by the app, and SQLite reuses a deleted id — a stale sidecar
    # under the next clip's id is one mtime check away from being served.
    try:
        os.unlink(os.path.join(str(vts.clips_dir()), f'clip_{int(clip_id)}_last.png'))
    except OSError:
        pass
    db.session.delete(clip)
    db.session.commit()
    return jsonify({'ok': True})


@bp.post('/clip/<int:clip_id>/neural-render')
def video_studio_clip_neural_render(clip_id):
    """✨ Re-render a finished clip through DLSS 5 Neural Rendering — as a new
    clip, never an edit (the studio exists to compare). Body: the dials
    (tone, structure, automask, temporal). The row appears at once in
    ``pending`` and the list's own poll shows it land; a refusal (the model
    is not set up, the clip is gone, a render of it is already running) is a
    400 with the sentence to show."""
    from flask import current_app
    from ..services import neural_render as nr
    data = request.get_json(silent=True) or {}
    try:
        out = nr.start_studio_render(current_app._get_current_object(), LOCAL_USER, clip_id, data)
    except nr.NeuralRenderError as exc:
        return jsonify({'error': str(exc)}), 400
    return jsonify({'ok': True, **out})
