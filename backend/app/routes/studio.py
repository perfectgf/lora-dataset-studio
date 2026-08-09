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
from ..utils.comfyui import get_zimage_models
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
    return {
        'cfg_choices': lts.CFG_CHOICES, 'default_cfg': lts.DEFAULT_CFG,
        'steps_choices': lts.STEPS_CHOICES, 'default_steps': lts.DEFAULT_STEPS,
        'steps2_choices': lts.STEPS_CHOICES if kind == 'sdxl' else None,
        'default_steps2': lts.DEFAULT_STEPS if kind == 'sdxl' else None,
        # Rythme mesuré de la machine (médiane observée) — même clé et même
        # source que le payload par dataset, pour que les deux branches du Studio
        # n'annoncent jamais deux durées pour un seul lancement. null = pas assez
        # d'historique, l'UI garde son « ~ » et son défaut.
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
    axes = _family_axes(kind)
    if kind == 'sdxl':
        models = lts.list_sdxl_base_models()
        return jsonify({'models': models, 'axes': axes,
                        'model_defaults': _base_defaults(kind, models)})
    if kind == 'krea':
        # Bases Krea locales ALTERNATIVES au défaut ÉLU (cf. lts.krea_default_base).
        # L'entrée de tête (filename vide → base_model absent → défaut élu) reste le
        # défaut ; `base_note` dit quel fichier c'est quand ce n'est pas celui que
        # Setup installe. La note est publiée MÊME sans alternative : c'est
        # précisément l'install où l'utilisateur n'a rien d'autre qui doit le lire.
        entry = lts.krea_default_base_entry()
        # Les chiffres du défaut sont ceux du fichier RÉELLEMENT élu : sans ça une
        # base non distillée élue par défaut repartait sur cfg 1 / 8 steps — la
        # même esquisse floue que le correctif #18 avait déjà réglée ailleurs. La
        # clé '' est publiée pour l'écran qui la lit sans sélecteur.
        base_defaults = None
        if entry['source']:
            base_defaults = lts.krea_model_defaults(entry['source'])
            axes = {**axes, 'default_cfg': base_defaults['cfg'],
                    'default_steps': base_defaults['steps']}
        alts = lts.krea_alt_base_models()
        if not alts:
            return jsonify({'models': [], 'axes': axes, 'base_note': entry['note'],
                            'model_defaults': {'': base_defaults} if base_defaults else {}})
        out = [{'filename': '', 'label': entry['label']}]
        out += [{'filename': m, 'label': m.split('\\')[-1].rsplit('.', 1)[0]} for m in alts]
        defaults = _base_defaults(kind, out)
        if base_defaults:
            defaults[''] = base_defaults
        return jsonify({'models': out, 'axes': axes, 'base_note': entry['note'],
                        'model_defaults': defaults})
    out = [{'filename': m, 'label': m.split('\\')[-1]} for m in get_zimage_models()]
    return jsonify({'models': out, 'axes': axes,
                    'model_defaults': _base_defaults(kind, out)})


@bp.get('/checkpoints')
def studio_checkpoints():
    return jsonify({'loras': lts.list_all_testable_checkpoints(LOCAL_USER)})


@bp.get('/recent-prompts')
def studio_recent_prompts():
    """Prompts de test récents GLOBAUX (tous datasets) — alimente le menu
    « Recent prompts » du mode comparaison ET du studio riche."""
    return jsonify({'ok': True, 'prompts': lts.user_recent_prompts(LOCAL_USER)})


@bp.post('/recent-prompts/delete')
def studio_recent_prompts_delete():
    """Supprime un prompt récent (+ cellules/images) sur TOUS les datasets."""
    d = request.get_json(silent=True) or {}
    return jsonify({'ok': True,
                    'deleted': lts.delete_prompt_everywhere(LOCAL_USER, d.get('prompt'))})


@bp.post('/random-caption')
def studio_random_caption():
    """Pick one usable training caption from the selected local dataset."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'dataset_id must be a positive integer'}), 400
    dataset_id = data.get('dataset_id')
    if (isinstance(dataset_id, bool) or not isinstance(dataset_id, int)
            or dataset_id <= 0 or dataset_id > _MAX_DATABASE_ID):
        return jsonify({'error': 'dataset_id must be a positive integer'}), 400
    try:
        caption = fds.random_kept_caption(LOCAL_USER, dataset_id)
    except LookupError:
        # Keep an inaccessible dataset indistinguishable from one that does not exist.
        return jsonify({
            'error': ('The selected dataset was not found or is inaccessible. '
                      'Choose a dataset from your library and try again.')
        }), 404
    if caption is None:
        return jsonify({
            'error': ('This dataset has no usable kept captions. Caption at least one '
                      'kept image and try again.')
        }), 422
    return jsonify({'ok': True, 'caption': caption})


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
    """Enrich the typed test prompt with the local Ollama text model (same client and
    same model as captioning). Runs inside the GPU-exclusive vision window so it never
    fights a queued generation for VRAM.

    400 = empty/oversized prompt · 409 = Ollama unavailable or answered nothing (its own
    reason carried through) · 503 = GPU busy."""
    d = request.get_json(silent=True) or {}
    try:
        with gpu_exclusive_vision_window(flag_ttl=600):
            enhanced = lts.enhance_test_prompt(d.get('prompt'))
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
            seed=d.get('seed'), prompt=d.get('prompt'), z_model=d.get('z_model'),
            z_models=d.get('z_models'),
            aspects=d.get('aspects'), cfgs=d.get('cfgs'), steps_list=d.get('steps'),
            steps2_list=d.get('steps2'), count=d.get('count'),
            permanent_loras=d.get('permanent_loras'), batch_loras=d.get('batch_loras'),
            external_loras=d.get('external_loras'),
            rebalance=d.get('rebalance'),
            rebalance_strength=d.get('rebalance_strength'),
            # Parité Generate — réglages globaux du run.
            negative=d.get('negative'), sampler=d.get('sampler'), scheduler=d.get('scheduler'),
            weight_dtype=d.get('weight_dtype'), enhancer=d.get('enhancer'),
            enhancer_strength=d.get('enhancer_strength'), detail_amount=d.get('detail_amount'),
            resolution_tier=d.get('resolution_tier'),
            resolution_multiplier=d.get('resolution_multiplier'),
            init_image=d.get('init_image'), denoise=d.get('denoise'),
            combine=d.get('combine'))
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
