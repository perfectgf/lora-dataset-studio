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
from ..services import lora_test_studio as lts
from ..utils.comfyui import get_zimage_models
from ._common import (_map_error, _require_comfyui, _studio_arch_mismatch_response,
                      _studio_missing_response)

bp = Blueprint('studio', __name__, url_prefix='/api/studio')


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


@bp.get('/base-models')
def studio_base_models():
    kind = (request.args.get('type') or 'zimage').lower()
    if kind == 'sdxl':
        return jsonify({'models': lts.list_sdxl_base_models()})
    if kind == 'krea':
        # Bases Krea locales ALTERNATIVES au UNET câblé de krea2_turbo.json (node 20).
        # « Official » (filename vide → z_model absent → node intact) en tête = défaut.
        # Aucune alternative sur disque → liste vide, le front masque le sélecteur.
        alts = lts.krea_alt_base_models()
        if not alts:
            return jsonify({'models': []})
        out = [{'filename': '', 'label': 'Official – Krea 2 Turbo'}]
        out += [{'filename': m, 'label': m.split('\\')[-1].rsplit('.', 1)[0]} for m in alts]
        return jsonify({'models': out})
    out = [{'filename': m, 'label': m.split('\\')[-1]} for m in get_zimage_models()]
    return jsonify({'models': out})


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


@bp.post('/run')
def studio_run():
    gate = _require_comfyui()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = lts.create_comparison_run(
            LOCAL_USER, d.get('selections') or [], d.get('strengths') or [],
            seed=d.get('seed'), prompt=d.get('prompt'), z_model=d.get('z_model'),
            aspects=d.get('aspects'), cfgs=d.get('cfgs'), steps_list=d.get('steps'),
            steps2_list=d.get('steps2'), count=d.get('count'),
            permanent_loras=d.get('permanent_loras'), batch_loras=d.get('batch_loras'),
            rebalance=d.get('rebalance'),
            rebalance_strength=d.get('rebalance_strength'),
            # Parité Generate — réglages globaux du run.
            negative=d.get('negative'), sampler=d.get('sampler'), scheduler=d.get('scheduler'),
            weight_dtype=d.get('weight_dtype'), enhancer=d.get('enhancer'),
            enhancer_strength=d.get('enhancer_strength'), detail_amount=d.get('detail_amount'),
            resolution_tier=d.get('resolution_tier'),
            resolution_multiplier=d.get('resolution_multiplier'),
            init_image=d.get('init_image'), denoise=d.get('denoise'))
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


@bp.post('/run/<run_id>/resume')
def studio_run_resume(run_id):
    gate = _require_comfyui()
    if gate:
        return gate
    try:
        res = lts.resume_run(LOCAL_USER, run_id=run_id)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})
