"""Settings API: config/secrets CRUD + capability probes."""
from flask import Blueprint, jsonify, request

from .. import capabilities
from .. import config as cfg

bp = Blueprint('settings', __name__, url_prefix='/api')

_TEST_TARGETS = {
    'gemini': capabilities.probe_gemini,
    'openai': capabilities.probe_openai,
    'comfyui': capabilities.probe_comfyui,
    'ollama': capabilities.probe_ollama,
    'aitoolkit': capabilities.probe_aitoolkit,
    'face_scoring': capabilities.probe_face_scoring,
    'masks': capabilities.probe_masks,
}


def _secret_presence() -> dict:
    return {name: bool(cfg.secret(name)) for name in cfg.SECRET_KEYS}


def _settings_payload() -> dict:
    return {'config': cfg.load_config(), 'secrets': _secret_presence()}


@bp.get('/settings')
def get_settings():
    return jsonify(_settings_payload())


@bp.put('/settings')
def put_settings():
    body = request.get_json(force=True, silent=True) or {}
    if 'config' in body and not isinstance(body['config'], dict):
        return jsonify({'error': "'config' must be an object"}), 400
    if 'secrets' in body and not isinstance(body['secrets'], dict):
        return jsonify({'error': "'secrets' must be an object"}), 400
    config_partial = body.get('config') or {}
    unknown = set(config_partial) - set(cfg.DEFAULTS)
    if unknown:
        return jsonify({'error': f"unknown config section '{sorted(unknown)[0]}'"}), 400
    # Each section must stay an object -- _deep_merge only recurses when both
    # sides are dicts, so a non-dict value here would REPLACE the whole section
    # (e.g. {"ollama": "x"} silently overwriting ollama.url + ollama.vision_model).
    for k, v in config_partial.items():
        if not isinstance(v, dict):
            return jsonify({'error': f"config section '{k}' must be an object"}), 400
    cfg.save_config(config_partial)
    cfg.set_secrets(body.get('secrets') or {})
    # A changed ComfyUI location must take effect NOW: the base/model listers cache
    # their scans for 5 min, so without this the training-base dropdowns keep showing
    # the pre-save (often empty) list right after the user points the app at ComfyUI.
    # (The wizard's _scan_models view refreshes via the frontend's forced
    # /api/capabilities?force=1 call, so no probe(force) is needed here.)
    if 'comfyui' in config_partial:
        from ..utils import comfyui
        comfyui.clear_model_caches()
    return jsonify(_settings_payload())


@bp.get('/capabilities')
def get_capabilities():
    force = bool(request.args.get('force'))
    return jsonify(capabilities.probe(force=force))


@bp.post('/settings/test/<target>')
def test_connection(target):
    probe_fn = _TEST_TARGETS.get(target)
    if probe_fn is None:
        return jsonify({'error': f"unknown test target '{target}'"}), 404
    return jsonify(probe_fn())
