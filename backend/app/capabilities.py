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
from pathlib import Path

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


def _import_ok(python: str, module_expr: str, timeout=60):
    """True/False = the import deterministically succeeded/failed. None = TIMEOUT —
    unknown, NOT a proven absence. The very first `import rembg` after an install
    compiles numba/scikit-image caches while the antivirus scans 40 MB of fresh
    DLLs: measured ~20 s cold vs ~1 s warm — a 20 s timeout read as False showed
    'Person masks ✗' for 10 min right after a SUCCESSFUL install."""
    try:
        result = subprocess.run([python, '-c', module_expr], capture_output=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return False


def _cached_import(key: str, python: str, module_expr: str) -> bool:
    now = time.time()
    cache_key = f'{key}:{python}:{module_expr}'
    cached = _import_cache.get(cache_key)
    if cached is not None and now - cached[0] < _IMPORT_TTL:
        return cached[1]
    ok = _import_ok(python, module_expr)
    if ok is None:
        # Timeout → report not-ready NOW but don't poison the cache: the next
        # probe re-tries against a warm import instead of a 600 s false ✗.
        return False
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


def probe_ollama_model(reachable=None) -> dict:
    # `reachable` lets probe() pass the reachability it already computed, so we
    # don't re-hit /api/tags a second time (and don't pay a second blocking
    # timeout when Ollama is configured-but-down). Called standalone -> we probe.
    url = (cfg.get('ollama.url') or '').rstrip('/')
    model = cfg.get('ollama.vision_model') or ''
    if not url:
        return {'ok': False, 'detail': 'ollama.url not configured'}
    if not model:
        return {'ok': False, 'detail': 'ollama.vision_model not configured'}
    if reachable is None:
        reachable = _http_ok(f'{url}/api/tags')        # gate on the stubbed seam first
    if not reachable:
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


# Prebuilt wheels for the ML extras (insightface 0.7.3, numpy<2, onnxruntime,
# rembg, opencv) exist for CPython 3.10–3.12 only. On a newer interpreter (3.13+)
# there is no numpy<2 / insightface wheel, so `pip install -r requirements-ml.txt`
# falls back to source builds that can't resolve (numpy build-dep clash) — the
# cryptic failure a fresh-clone user hits. Surface the version so the setup can
# warn UP FRONT instead of after a 200-line pip traceback.
_ML_PY_MIN = (3, 10)
_ML_PY_MAX = (3, 12)


def python_ml_status() -> dict:
    """Version of THIS interpreter (the one `ml_extras` installs into via
    sys.executable) and whether it is inside the wheel-supported ML range."""
    v = sys.version_info
    return {
        'version': f'{v.major}.{v.minor}.{v.micro}',
        'ml_supported': _ML_PY_MIN <= (v.major, v.minor) <= _ML_PY_MAX,
        'ml_range': f'{_ML_PY_MIN[0]}.{_ML_PY_MIN[1]}–{_ML_PY_MAX[0]}.{_ML_PY_MAX[1]}',
    }


def probe_scrape_deps() -> dict:
    """The scraper's optional Python deps (requirements-scrape.txt). find_spec
    only (no import cost): the scrape stack runs IN-PROCESS, so the app's own
    interpreter is the one that must see the packages. curl_cffi + gallery_dl
    are the two hard requirements (picazor/civitai fetch, gallery enumeration);
    bs4/cloudscraper ride along in the same install."""
    import importlib.util
    missing = [m for m in ('curl_cffi', 'gallery_dl', 'bs4', 'cloudscraper')
               if importlib.util.find_spec(m) is None]
    return {'ok': not missing,
            'detail': 'scrape deps OK' if not missing else f"missing: {', '.join(missing)}"}


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
            # Any 'klein'-named subfolder under unet/ OR diffusion_models/ counts:
            # shared installs keep e.g. diffusion_models/'Flux2 klein'/ (the KV
            # variant) next to our canonical unet/klein/ download — hiding it made
            # the picker blind to models the user already owns.
            elif 'klein' in name.lower():
                result['klein'].extend(_model_files(sub))
            elif base_name == 'unet' and name.lower().startswith('krea'):
                result['krea'].extend(_model_files(sub))

    result['klein'] = sorted(set(result['klein']))
    result['sdxl'] = _model_files(models_dir / 'checkpoints')
    return result


# --- Auto-detection (Setup wizard) -----------------------------------------
# Discover already-installed tools so the wizard can fill config itself. Two
# signals: a REACHABLE default port (safe to auto-apply — it answered) and a
# folder found on disk (a guess → the UI confirms before writing it).
_OLLAMA_DEFAULT_URL = 'http://127.0.0.1:11434'
_COMFYUI_DEFAULT_URL = 'http://127.0.0.1:8188'


def _common_roots() -> list:
    home = Path.home()
    candidates = [Path('C:/'), Path('D:/'), home, home / 'Downloads', home / 'Desktop',
                  home / 'projects', home / 'source' / 'repos', Path('C:/tools')]
    out, seen = [], set()
    for r in candidates:
        try:
            if r not in seen and r.is_dir():
                seen.add(r)
                out.append(r)
        except OSError:
            continue
    return out


def _find_install_dir(names, marker) -> str:
    """Shallow scan of common roots for a folder named in `names` satisfying
    `marker(path)`. First hit as a string, else ''. Shallow (root/name only) to
    stay fast — a deep recursive walk of C:\\ would be far too slow for a probe."""
    for root in _common_roots():
        for name in names:
            cand = root / name
            try:
                if cand.is_dir() and marker(cand):
                    return str(cand)
            except OSError:
                continue
    return ''


def _detect_ollama() -> dict:
    if not _http_ok(f'{_OLLAMA_DEFAULT_URL}/api/tags'):
        return {}
    out = {'url': _OLLAMA_DEFAULT_URL}
    names = _ollama_tags(_OLLAMA_DEFAULT_URL)
    vls = [n for n in names if 'vl' in (n or '').lower() or 'vision' in (n or '').lower()]
    # Prefer an -instruct tag over the Thinking variant: on a caption/omission task the
    # Thinking model reasons out loud instead of captioning (see get_vision_model), so if
    # both are installed, pick instruct; failing that, anything that isn't 'thinking'.
    vl = (next((n for n in vls if 'instruct' in (n or '').lower()), '')
          or next((n for n in vls if 'thinking' not in (n or '').lower()), '')
          or (vls[0] if vls else ''))
    if vl:
        out['vision_model'] = vl
    return out


# --- GPU VRAM probe (nvidia-smi, cached) ---------------------------------------
_gpu_cache = {'ts': 0.0, 'gb': None}
_GPU_TTL = 600


def gpu_vram_gb():
    """Total VRAM of GPU 0 in GB via nvidia-smi, cached 10 min. None when it can't
    be determined (no NVIDIA GPU / nvidia-smi absent) — callers must treat None
    as 'unknown', never as 0 (an unknown GPU must not trigger OOM warnings)."""
    import subprocess
    now = time.time()
    if _gpu_cache['ts'] and (now - _gpu_cache['ts']) < _GPU_TTL:
        return _gpu_cache['gb']
    gb = None
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if proc.returncode == 0:
            first = (proc.stdout or '').strip().splitlines()
            if first:
                gb = round(float(first[0].strip()) / 1024, 1)   # MiB -> GB
    except (OSError, ValueError, subprocess.TimeoutExpired):
        gb = None
    _gpu_cache.update(ts=now, gb=gb)
    return gb


def _is_comfyui_dir(d) -> bool:
    try:
        return (d / 'main.py').exists() and (d / 'models').is_dir()
    except OSError:
        return False


def resolve_comfyui_base(path: str) -> dict:
    """Resolve a user-entered ComfyUI folder to the one that actually holds main.py +
    models/ (which is what every base/model lister scans). Handles the common
    portable-bundle mistake: users point at ...\\ComfyUI_windows_portable, but main.py
    and models/ live one level down in ...\\ComfyUI_windows_portable\\ComfyUI. Without
    this, comfyui.base_dir\\models never exists -> "No SDXL checkpoint found" even though
    ComfyUI is running.

    Returns {valid, resolved, nested}: `nested` is True when we descended into a child
    ComfyUI/ (the caller can then auto-correct base_dir to `resolved`)."""
    if not path:
        return {'valid': False, 'resolved': '', 'nested': False}
    p = Path(path)
    if _is_comfyui_dir(p):
        return {'valid': True, 'resolved': str(p), 'nested': False}
    child = p / 'ComfyUI'
    if _is_comfyui_dir(child):
        return {'valid': True, 'resolved': str(child), 'nested': True}
    return {'valid': False, 'resolved': str(p), 'nested': False}


def _detect_comfyui() -> dict:
    out = {}
    if _http_ok(f'{_COMFYUI_DEFAULT_URL}/history'):
        out['api_url'] = _COMFYUI_DEFAULT_URL
    base = _find_install_dir(('ComfyUI', 'comfyui'), _is_comfyui_dir)
    if not base:
        # portable bundle nests the app: <root>/ComfyUI_windows_portable/ComfyUI/
        portable = _find_install_dir(('ComfyUI_windows_portable',),
                                     lambda d: _is_comfyui_dir(d / 'ComfyUI'))
        if portable:
            base = str(Path(portable) / 'ComfyUI')
    if base:
        out['base_dir'] = base
    return out


def _detect_aitoolkit() -> dict:
    d = _find_install_dir(('ai-toolkit', 'ai_toolkit', 'aitoolkit'),
                          lambda p: (p / 'run.py').exists())
    return {'dir': d} if d else {}


def autodetect() -> dict:
    """Best-effort discovery of installed tools for the Setup wizard. A value under
    a reachable default port (url/api_url) is safe to auto-apply; a disk-scanned
    path (base_dir/dir) is a suggestion the UI should confirm. Never raises."""
    return {
        'ollama': _detect_ollama(),
        'comfyui': _detect_comfyui(),
        'aitoolkit': _detect_aitoolkit(),
    }


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
    base_dir = cfg.get('comfyui.base_dir') or ''
    comfy_dir = resolve_comfyui_base(base_dir)

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
            'base_dir': base_dir,
            'dir_configured': bool(base_dir),
            'dir_valid': comfy_dir['valid'],       # base_dir really is a ComfyUI install
            'resolved_dir': comfy_dir['resolved'],
            'models': models,
        },
        'ollama': {
            'reachable': ollama['ok'],
            'url': cfg.get('ollama.url') or '',
            'vision_model': cfg.get('ollama.vision_model') or '',
            'vision_model_ready': probe_ollama_model(reachable=ollama['ok'])['ok'],
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
        'python': python_ml_status(),
        'scrape_deps': probe_scrape_deps()['ok'],
        'training_visible': aitoolkit['ok'],
        'studio_visible': comfy['ok'],
    }

    _cache, _cache_ts = caps, now
    return copy.deepcopy(caps)
