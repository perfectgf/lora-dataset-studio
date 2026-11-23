"""Studio blueprint: dataset-agnostic checkpoint x strength comparison runs
(run_id-driven) across every trained LoRA — the cross-dataset selector +
comparison-run lifecycle. Per-dataset /dataset/<id>/lora-test/* routes live in
datasets.py (single-dataset sweep, same service).

No login — single local user (`cfg.LOCAL_USER`). `/run` and `/run/<id>/resume`
actually enqueue ComfyUI jobs, so they're gated on `capabilities.probe()`
(409 with a UI hint) — everything else (checkpoints/prompts listings, run
status, cancel) stays reachable even when ComfyUI is offline so run history
never goes dark.
"""
import base64

from flask import Blueprint, jsonify, request

from ..config import LOCAL_USER
from ..gpu_window import gpu_exclusive_vision_window
from ..services import face_dataset_service as fds
from ..services import lora_test_studio as lts
# The engine-preflight misses answer the SAME actionable 409 here as in the
# dataset lane — the body is what itemizes the missing assets and starts their
# download. Imported rather than re-derived (routes/bank.py does the same).
from .datasets import _improve_engine_error
from ._common import (_map_error, _require_comfyui, _require_no_stalled_comfyui,
                      _studio_arch_mismatch_response, _studio_missing_response)

bp = Blueprint('studio', __name__, url_prefix='/api/studio')

# SQLite Integer primary keys are signed 64-bit values.  Python ints are
# arbitrary precision, so validate before an oversized JSON value reaches a DB bind.
_MAX_DATABASE_ID = (1 << 63) - 1


def _read_uploaded_image():
    """Pull the image bytes from either a multipart `image` file or a JSON
    `image_base64` (a data: URL prefix is tolerated). Raises ValueError with an
    actionable message when nothing usable is attached."""
    f = request.files.get('image')
    if f is not None:
        return f.read()
    d = request.get_json(silent=True) or {}
    b64 = d.get('image_base64') or d.get('image')
    if isinstance(b64, str) and b64:
        if b64.startswith('data:'):
            b64 = b64.split(',', 1)[-1]
        try:
            return base64.b64decode(b64)
        except Exception:
            raise ValueError('invalid base64 image')
    raise ValueError('no image provided')


def _family_axes(kind):
    """The CFG / steps ladders of a family, published WITHOUT a dataset.

    They are the very constants `studio_payload` already sends per dataset — the
    multi-LoRA comparison has no dataset to read them from, which is exactly why
    its panel could not offer a steps axis while every other generation surface
    could (user report: "I cannot set the number of steps when two LoRAs are
    selected, in blend or in comparison"). Same source, so the screens cannot
    drift into two ladders. `steps2` stays SDXL-only: the second pass is a
    property of that workflow, not a setting the others hide."""
    defaults = lts.studio_family_defaults(kind)
    return {
        'cfg_choices': [1.0] if kind == 'flux2klein' else lts.CFG_CHOICES,
        'default_cfg': defaults['cfg'],
        'steps_choices': lts.STEPS_CHOICES, 'default_steps': defaults['steps'],
        'steps2_choices': lts.STEPS_CHOICES if kind == 'sdxl' else None,
        'default_steps2': lts.DEFAULT_STEPS if kind == 'sdxl' else None,
        # Measured machine speed (observed median), with the same key/source
        # as the dataset payload so both Studio modes estimate the same run
        # consistently. NULL means insufficient history; UI keeps its estimate.
        'seconds_per_image': lts.measured_seconds_per_image(kind),
    }


def _base_defaults(kind, models):
    """Per-BASE cfg/steps, keyed by the same `filename` this route publishes.

    The per-dataset payload has carried these for a while; this route did not,
    and it is the one feeding the comparison / blend screen. So an undistilled
    base picked THERE silently kept the family's Turbo numbers (cfg 1 / 8 steps)
    and rendered the same blurry sketch the solo screen had already been fixed
    for. Same helper, so the two branches cannot answer differently.

    The '' entry (the wired workflow UNET) is deliberately absent: it is not a
    file, `studio_model_defaults` has no name to read, and `default_cfg` /
    `default_steps` are already its answer."""
    return lts.studio_model_defaults(
        kind, [m['filename'] for m in models if m.get('filename')])


@bp.get('/base-models')
def studio_base_models():
    """Bases of a family + (additive) its `axes` ladders and `model_defaults`.
    `models` keeps its exact shape — an older frontend reads it unchanged and
    simply ignores the two extras."""
    kind = (request.args.get('type') or 'zimage').lower()
    if not lts.can_generate_with(kind):
        return jsonify({'error': str(lts._no_generation_lane(kind))}), 400
    axes = _family_axes(kind)
    capabilities = {'generation_capabilities': lts.generation_capabilities(kind)}
    if kind == 'sdxl':
        models = lts.list_sdxl_base_models()
        return jsonify({**capabilities, 'models': models, 'axes': axes,
                        'model_defaults': _base_defaults(kind, models)})
    if kind == 'krea':
        # Local Krea alternatives exclude the elected default (krea_default_base).
        # The first empty-filename entry still selects that default. base_note
        # identifies a default different from Setup's file, even when there are
        # no alternatives: those users particularly need the explanation.
        entry = lts.krea_default_base_entry()
        # Defaults come from the actually elected file. Otherwise a non-distilled
        # default inherited CFG 1/eight steps and blurry results, repeating #18.
        # Publish the empty key for surfaces without a selector.
        base_defaults = None
        if entry['source']:
            base_defaults = lts.krea_model_defaults(entry['source'])
            axes = {**axes, 'default_cfg': base_defaults['cfg'],
                    'default_steps': base_defaults['steps']}
        alts = lts.krea_alt_base_models()
        if not alts:
            return jsonify({**capabilities, 'models': [], 'axes': axes, 'base_note': entry['note'],
                            'model_defaults': {'': base_defaults} if base_defaults else {}})
        out = [{'filename': '', 'label': entry['label']}]
        out += [{'filename': m, 'label': m.split('\\')[-1].rsplit('.', 1)[0]} for m in alts]
        defaults = _base_defaults(kind, out)
        if base_defaults:
            defaults[''] = base_defaults
        return jsonify({**capabilities, 'models': out, 'axes': axes, 'base_note': entry['note'],
                        'model_defaults': defaults})
    out = [{'filename': m or '', 'label': lts._basename(m) if m else 'Default model'}
           for m in lts.family_base_models(kind)]
    if kind in lts.TRAINED_IMAGE_FAMILIES:
        from ..services.trained_image_models import generation_readiness
        readiness = generation_readiness(kind)
        capabilities['generation_readiness'] = readiness
        capabilities['default_model'] = (readiness.get('assets') or {}).get('diffusion_model', '')
        if readiness.get('config_error'):
            capabilities['base_note'] = readiness['config_error']
    elif kind == 'zimage':
        from ..services.trained_image_models import zimage_studio_readiness
        capabilities['generation_readiness'] = zimage_studio_readiness()
    return jsonify({**capabilities, 'models': out, 'axes': axes,
                    'model_defaults': _base_defaults(kind, out)})


@bp.get('/checkpoints')
def studio_checkpoints():
    return jsonify({'loras': lts.list_all_testable_checkpoints(LOCAL_USER)})


@bp.get('/recent-prompts')
def studio_recent_prompts():
    """Global recent test prompts across datasets, feeding Recent prompts
    in comparison mode and the full Studio."""
    return jsonify({'ok': True, 'prompts': lts.user_recent_prompts(LOCAL_USER)})


@bp.post('/recent-prompts/delete')
def studio_recent_prompts_delete():
    """Delete a recent prompt and associated cells/images across all datasets."""
    d = request.get_json(silent=True) or {}
    return jsonify({'ok': True,
                    'deleted': lts.delete_prompt_everywhere(LOCAL_USER, d.get('prompt'))})


@bp.get('/civitai/images')
def studio_civitai_images():
    """🌐 Civitai prompt browser: top images paired with their generation
    prompt, for the Studio/Canvas prompt field. Listing is public; prompts
    need the (free) Civitai API key — `has_key` tells the UI which story to
    show. Continuation is (`next_cursor`, `next_skip`) echoed back verbatim.

    400 = bad filter value · 409 = Civitai unreachable / key refused (the
    sentence carries the remedy)."""
    from ..services import civitai_browser
    a = request.args
    try:
        res = civitai_browser.browse(
            period=a.get('period', 'week'), sort=a.get('sort', 'reactions'),
            level=a.get('level', 'none'), cursor=a.get('cursor') or None,
            skip=a.get('skip', 0), want=a.get('want', 12),
            require_prompt=a.get('require_prompt', '1') != '0')
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/random-caption')
def studio_random_caption():
    """Pick one usable training caption from the selected local dataset OR bank.

    ``{dataset_id}`` and ``{bank_id}`` are alternatives, never both. The dataset
    form is byte-identical to the request that shipped before banks were a
    source, so an older tab keeps working unchanged.

    WHY BANKS TOO (asked for by the maintainer): a bank is captioned by the 🏷️
    Caption pass long before anything is promoted, and it is usually the biggest
    pile of real captions on the machine. Offering only datasets put the richest
    source out of reach of the one shortcut that exists to reach it.
    """
    data = request.get_json(silent=True)
    # NEITHER key given (or no object at all) names BOTH in the refusal: the old
    # wording said "dataset_id" only, which stopped being the whole truth the day
    # a bank became a legal source.
    if not isinstance(data, dict) or (data.get('bank_id') is None
                                      and data.get('dataset_id') is None):
        return jsonify({'error': 'dataset_id or bank_id must be a positive integer'}), 400
    has_bank = data.get('bank_id') is not None
    has_dataset = data.get('dataset_id') is not None
    if has_bank and has_dataset:
        # Two sources in one request is a caller bug, and guessing which one it
        # meant would draw from the wrong pile in silence.
        return jsonify({'error': 'send either dataset_id or bank_id, not both'}), 400
    key = 'bank_id' if has_bank else 'dataset_id'
    source_id = data.get(key)
    if (isinstance(source_id, bool) or not isinstance(source_id, int)
            or source_id <= 0 or source_id > _MAX_DATABASE_ID):
        return jsonify({'error': f'{key} must be a positive integer'}), 400
    noun = 'bank' if has_bank else 'dataset'
    try:
        if has_bank:
            from ..services import image_bank_service as banks
            caption = banks.random_kept_caption(LOCAL_USER, source_id)
        else:
            caption = fds.random_kept_caption(LOCAL_USER, source_id)
    except LookupError:
        # Keep an inaccessible source indistinguishable from one that does not exist.
        return jsonify({
            'error': (f'The selected {noun} was not found or is inaccessible. '
                      f'Choose a {noun} from your library and try again.')
        }), 404
    if caption is None:
        return jsonify({
            'error': (f'This {noun} has no usable kept captions. Caption at least one '
                      'kept image and try again.')
        }), 422
    return jsonify({'ok': True, 'caption': caption})


@bp.post('/image/<int:image_id>/repair')
def studio_image_repair(image_id):
    """Repaint a drawn zone of a GENERATED image from a free prompt.

    Asked for by .samexit on Discord: fix one detail instead of regenerating the
    whole picture. The image is addressed by its id and the file is resolved
    server-side — a client-supplied path is how an in-place overwrite becomes an
    arbitrary write.

    {boxes: [[x,y,w,h] normalised...], prompt: "..."}. A missing prompt is a 400
    rather than a silent fall back to the watermark reconstruction sentence.
    """
    data = request.get_json(silent=True) or {}
    boxes = data.get('boxes') or data.get('regions')
    mask = data.get('mask')
    if not boxes and not mask:
        return jsonify({'error': 'draw or paint the area to repair first'}), 400
    try:
        result = lts.repair_generated_image(LOCAL_USER, image_id, boxes,
                                            data.get('prompt'), mask=mask)
    except Exception as e:
        engine_error = _improve_engine_error(e)
        if engine_error:
            return engine_error
        return _map_error(e)
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(result)


@bp.post('/image/<int:image_id>/repair/undo')
def studio_image_repair_undo(image_id):
    """↩ Put back the pixels from just before the last ✦ Repair of a generated
    image. One step deep. {'undone': false} = there was nothing to undo."""
    try:
        result = lts.undo_generated_repair(LOCAL_USER, image_id)
    except (ValueError, RuntimeError) as e:
        return _map_error(e)
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(result)


@bp.post('/describe-image')
def studio_describe_image():
    """Describe an uploaded image into a ready-to-paste Studio TEST PROMPT (Ollama
    vision). Accepts multipart (`image`) or JSON (`image_base64`). Runs inside the
    GPU-exclusive vision window (frees ComfyUI VRAM first, blocks during training /
    another vision pass); the service unloads the model after (keep_alive=0).

    400 = bad/oversized/non-image upload · 409 = Ollama unavailable/rejected (its own
    reason carried through) · 503 = GPU busy."""
    try:
        image_bytes = _read_uploaded_image()
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    try:
        with gpu_exclusive_vision_window(flag_ttl=600):
            prompt = lts.describe_test_prompt(image_bytes)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'prompt': prompt})


@bp.post('/enhance-prompt')
def studio_enhance_prompt():
    """Enrich the typed test prompt with a local Ollama text model — the captioning
    model by default, or the one the ⚙️ Enhance options picked (`ollama_model`; ''
    or absent = the default, same spread-if-set contract as the Bank's caption
    dials). Runs inside the GPU-exclusive vision window so it never fights a queued
    generation for VRAM.

    400 = empty/oversized prompt or invalid ollama_model · 409 = Ollama unavailable
    or answered nothing (its own reason carried through) · 503 = GPU busy."""
    d = request.get_json(silent=True) or {}
    from ..services.ollama_control import normalize_ollama_model_ref
    raw_model = d.get('ollama_model')
    try:
        model = normalize_ollama_model_ref(
            '' if raw_model is None else raw_model, allow_empty=True) or None
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    try:
        with gpu_exclusive_vision_window(flag_ttl=600):
            enhanced = lts.enhance_test_prompt(d.get('prompt'), model=model)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'prompt': enhanced})


@bp.post('/run')
def studio_run():
    gate = _require_comfyui()
    if gate:
        return gate
    gate = _require_no_stalled_comfyui()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = lts.create_comparison_run(
            LOCAL_USER, d.get('selections') or [], d.get('strengths') or [],
            # Shared Generate settings: one object, identical wire keys.
            lts.StudioGenSettings.from_payload(d),
            # Prompt batch: one pass per checked prompt. create_comparison_run
            # already accepted this argument, but this route failed to forward it,
            # unlike dataset and canvas launch routes. Absent body field remains
            # None, preserving previous single-prompt behavior.
            prompts=d.get('prompts'),
            external_loras=d.get('external_loras'), combine=d.get('combine'))
    except Exception as e:
        from ..services.lora_test_studio import StudioArchMismatch, StudioAssetsMissing
        if isinstance(e, StudioArchMismatch):   # wrong-arch checkpoint → actionable 409
            return _studio_arch_mismatch_response(e)
        if isinstance(e, StudioAssetsMissing):  # models/nodes absent → actionable 409
            return _studio_missing_response(e)
        return _map_error(e)
    return jsonify({'ok': True, **{k: res[k] for k in ('created', 'seed', 'count', 'run_id')}})


@bp.get('/run/<run_id>/status')
def studio_run_status(run_id):
    payload = lts.studio_payload_run(LOCAL_USER, run_id)
    return (jsonify(payload), 200) if payload else (jsonify({'error': 'not found'}), 404)


@bp.post('/run/<run_id>/cancel')
def studio_run_cancel(run_id):
    return jsonify({'ok': True, 'cancelled': lts.cancel_run(LOCAL_USER, run_id=run_id)})


@bp.post('/run/<run_id>/confirm-comfyui-restart')
def studio_run_confirm_comfyui_restart(run_id):
    data = request.get_json(silent=True) or {}
    if data.get('confirmed_comfyui_restart') is not True:
        return jsonify({'error': 'Confirm that you restarted ComfyUI before clearing this paused job.'}), 400
    # This one action must observe a freshly responsive replacement process; a
    # cached green capability result cannot act as a restart gate.
    gate = _require_comfyui(force=True)
    if gate:
        return gate
    try:
        cancelled = lts.confirm_unknown_comfyui_restart(
            LOCAL_USER, run_id=run_id, restart_confirmed=True)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'cancelled': cancelled, 'resumable': True})


@bp.post('/run/<run_id>/resume')
def studio_run_resume(run_id):
    gate = _require_comfyui()
    if gate:
        return gate
    gate = _require_no_stalled_comfyui()
    if gate:
        return gate
    try:
        res = lts.resume_run(LOCAL_USER, run_id=run_id)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})
