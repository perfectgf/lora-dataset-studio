"""Capability probes: what's actually reachable/configured right now.

Consumed by the Settings UI (engine cards, "Test connection" buttons) and by
feature gating elsewhere in the app. `_http_ok` is the single network seam —
every reachability probe goes through it so tests can patch one symbol.
`_import_ok` is the equivalent seam for the slow subprocess import-probes.
"""
import copy
import re
import subprocess
import sys
import time

import requests

from . import config as cfg

_CACHE_TTL = 30
_cache = None
_cache_ts = 0.0

_IMPORT_TTL = 600
_import_cache = {}  # key -> (ts, ok)

_ZIMAGE_RE = re.compile(r'z[ -]?image', re.IGNORECASE)
_MODEL_SUFFIXES = ('.safetensors', '.gguf')


def _http_ok(url, timeout=3) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


def _import_ok(python: str, module_expr: str, timeout=20) -> bool:
    try:
        result = subprocess.run([python, '-c', module_expr], capture_output=True, timeout=timeout)
        return result.returncode == 0
    except Exception:
        return False


def _cached_import(key: str, python: str, module_expr: str) -> bool:
    now = time.time()
    cache_key = f'{key}:{python}:{module_expr}'
    cached = _import_cache.get(cache_key)
    if cached is not None and now - cached[0] < _IMPORT_TTL:
        return cached[1]
    ok = _import_ok(python, module_expr)
    _import_cache[cache_key] = (now, ok)
    return ok


def probe_gemini() -> dict:
    ok = bool(cfg.secret('GEMINI_API_KEY'))
    return {'ok': ok, 'detail': 'key set' if ok else 'key missing'}


def probe_openai() -> dict:
    ok = bool(cfg.secret('OPENAI_API_KEY'))
    return {'ok': ok, 'detail': 'key set' if ok else 'key missing'}


def probe_comfyui() -> dict:
    api_url = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api_url:
        return {'ok': False, 'detail': 'comfyui.api_url not configured'}
    ok = _http_ok(f'{api_url}/history')
    return {'ok': ok, 'detail': api_url if ok else f'unreachable: {api_url}'}


def probe_ollama() -> dict:
    url = (cfg.get('ollama.url') or '').rstrip('/')
    if not url:
        return {'ok': False, 'detail': 'ollama.url not configured'}
    ok = _http_ok(f'{url}/api/tags')
    return {'ok': ok, 'detail': url if ok else f'unreachable: {url}'}


def _ollama_tags(url, timeout=3) -> list:
    """Model names Ollama reports at /api/tags. Network seam (patched in tests)."""
    try:
        resp = requests.get(f'{url}/api/tags', timeout=timeout)
        if resp.status_code >= 400:
            return []
        return [m.get('name', '') for m in (resp.json().get('models') or [])]
    except Exception:
        return []


def _model_present(configured: str, names: list) -> bool:
    if not configured:
        return False
    if configured in names:
        return True
    base = configured.split(':')[0]                    # config w/o :tag matches any tag
    return any((n or '').split(':')[0] == base for n in names)


def probe_ollama_model() -> dict:
    url = (cfg.get('ollama.url') or '').rstrip('/')
    model = cfg.get('ollama.vision_model') or ''
    if not url:
        return {'ok': False, 'detail': 'ollama.url not configured'}
    if not model:
        return {'ok': False, 'detail': 'ollama.vision_model not configured'}
    if not _http_ok(f'{url}/api/tags'):                # gate on the stubbed seam first:
        return {'ok': False, 'detail': f'ollama unreachable: {url}'}
    ok = _model_present(model, _ollama_tags(url))
    return {'ok': ok, 'detail': f'{model} ready' if ok else f'{model} not pulled'}


def clear_import_cache() -> None:
    """Drop cached import-probe results and the main probe cache so the next
    probe re-checks freshly installed packages instead of a stale 600s 'False'."""
    global _cache, _cache_ts
    _import_cache.clear()
    _cache = None
    _cache_ts = 0.0


def probe_aitoolkit() -> dict:
    d = cfg.aitoolkit_path('dir')
    if not d:
        return {'ok': False, 'detail': 'aitoolkit.dir not configured'}
    venv_python = cfg.aitoolkit_path('venv_python')
    ok = (d / 'run.py').exists() and bool(venv_python) and venv_python.exists()
    return {'ok': ok, 'detail': str(d) if ok else f'invalid aitoolkit dir: {d}'}


def probe_face_scoring() -> dict:
    python = cfg.get('face_scoring.python') or sys.executable
    ok = _cached_import('face_scoring', python, 'import insightface, onnxruntime')
    return {'ok': ok, 'detail': 'insightface + onnxruntime import OK' if ok else 'import failed'}


def probe_masks() -> dict:
    python = cfg.get('masks.python') or sys.executable
    ok = _cached_import('masks', python, 'import rembg')
    return {'ok': ok, 'detail': 'rembg import OK' if ok else 'import failed'}


def _model_files(folder) -> list:
    try:
        if not folder.is_dir():
            return []
        return sorted(
            p.name for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _MODEL_SUFFIXES
        )
    except OSError:
        return []


def _scan_models() -> dict:
    result = {'zimage': [], 'sdxl': [], 'krea': [], 'klein': []}
    try:
        models_dir = cfg.comfyui_dir('models')
    except Exception:
        return result
    if not models_dir:
        return result
    try:
        if not models_dir.is_dir():
            return result
    except OSError:
        return result

    for base_name in ('unet', 'diffusion_models'):
        base = models_dir / base_name
        try:
            if not base.is_dir():
                continue
            subfolders = [p for p in base.iterdir() if p.is_dir()]
        except OSError:
            continue
        for sub in subfolders:
            name = sub.name
            if _ZIMAGE_RE.search(name):
                result['zimage'].extend(_model_files(sub))
            elif base_name == 'unet':
                if name.lower() == 'klein':
                    result['klein'].extend(_model_files(sub))
                elif name.lower().startswith('krea'):
                    result['krea'].extend(_model_files(sub))

    result['sdxl'] = _model_files(models_dir / 'checkpoints')
    return result


def probe(force=False) -> dict:
    global _cache, _cache_ts
    now = time.time()
    if _cache is not None and not force and (now - _cache_ts) < _CACHE_TTL:
        return copy.deepcopy(_cache)

    comfy = probe_comfyui()
    ollama = probe_ollama()
    aitoolkit = probe_aitoolkit()
    gemini = probe_gemini()
    openai_ = probe_openai()
    face_scoring = probe_face_scoring()
    masks = probe_masks()
    models = _scan_models()

    caps = {
        'configured': cfg.is_configured(),
        'engines': {
            'nanobanana': gemini['ok'],
            'chatgpt': openai_['ok'],
            'klein': comfy['ok'] and bool(models['klein']),
        },
        'comfyui': {
            'reachable': comfy['ok'],
            'api_url': cfg.get('comfyui.api_url') or '',
            'models': models,
        },
        'ollama': {
            'reachable': ollama['ok'],
            'url': cfg.get('ollama.url') or '',
            'vision_model': cfg.get('ollama.vision_model') or '',
            'vision_model_ready': probe_ollama_model()['ok'],
        },
        'aitoolkit': {
            'configured': bool(cfg.get('aitoolkit.dir')),
            'valid': aitoolkit['ok'],
        },
        'captioners': {
            'joycaption': aitoolkit['ok'] and (cfg.BACKEND_DIR / 'infer' / 'joycaption_infer.py').exists(),
            'ollama': ollama['ok'],
        },
        'face_scoring': face_scoring['ok'],
        'masks': masks['ok'],
        'training_visible': aitoolkit['ok'],
        'studio_visible': comfy['ok'],
    }

    _cache, _cache_ts = caps, now
    return copy.deepcopy(caps)
