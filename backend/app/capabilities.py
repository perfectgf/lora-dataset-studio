"""Capability probes: what's actually reachable/configured right now.

Consumed by the Settings UI (engine cards, "Test connection" buttons) and by
feature gating elsewhere in the app. `_http_ok` is the single network seam —
every reachability probe goes through it so tests can patch one symbol.
`_import_ok` is the equivalent seam for the slow subprocess import-probes.
"""
import copy
import os
import re
import shutil
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
# Aligned with klein_edit_helper / utils.comfyui (was missing '.sft', so the
# picker under-listed vs the resolvers). '.gguf' kept deliberately: the app
# supports ComfyUI-GGUF quantised diffusion models, and dropping it would hide
# models existing installs already see (see comfy_model_paths module docstring).
_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')


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
    """ChatGPT engine readiness: a pay-per-use API key OR a connected ChatGPT
    subscription (Codex OAuth) both light the engine up."""
    from .services import chatgpt_oauth
    key = bool(cfg.secret('OPENAI_API_KEY'))
    sub = chatgpt_oauth.status()['connected']
    parts = (['key set'] if key else []) + (['subscription connected'] if sub else [])
    return {'ok': key or sub, 'detail': ' + '.join(parts) if parts else 'key missing'}


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


# Ollama install detection, EXECUTION-INDEPENDENT: it must answer "installed"
# even when the server is stopped. probe_ollama() above only sees a RUNNING
# server (HTTP probe), which made an installed-but-stopped Ollama read as absent
# — misleading. This pair lets the UI tell 'not installed' from 'installed but
# stopped' (→ offer a Start button) from 'running'.
_OLLAMA_WIN_BINARY = ('Programs', 'Ollama', 'ollama.exe')   # under %LOCALAPPDATA%


def _ollama_binary() -> str:
    """Absolute path to the Ollama CLI binary if installed, else ''. Two signals,
    neither of which needs the server running:
      1. ``shutil.which('ollama')`` — the official installer adds Ollama to PATH
         (Windows per-user, macOS/Linux /usr/local/bin), so this is the primary hit.
      2. Windows fallback: the per-user location the official installer writes to,
         ``%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe`` (verified against
         docs.ollama.com/windows) — covers a shell whose PATH was not refreshed
         since the install. First hit wins; never raises."""
    exe = shutil.which('ollama')
    if exe:
        return exe
    if os.name == 'nt':
        local = os.environ.get('LOCALAPPDATA')
        if local:
            cand = Path(local).joinpath(*_OLLAMA_WIN_BINARY)
            try:
                if cand.is_file():
                    return str(cand)
            except OSError:
                pass
    return ''


def probe_ollama_installed() -> dict:
    """Is the Ollama binary present on disk (independent of the server running)?
    The `installed` capability the UI reads alongside `reachable` to pick between
    an install guide (not installed) and a Start button (installed but stopped)."""
    path = _ollama_binary()
    return {'ok': bool(path), 'binary_path': path,
            'detail': path or 'ollama binary not found (PATH or default install location)'}


def _ollama_tags(url, timeout=3) -> list:
    """Model identifiers Ollama reports at /api/tags. Each entry is read from BOTH
    the `name` and `model` fields: across Ollama versions either one can be the field
    that carries the (namespaced) identifier, and builds exist that populate `model`
    while leaving `name` blank — reading a single field made a genuinely-pulled model
    read as absent (issue #7: a namespaced model present in the list, yet
    vision_model=no). Blanks are dropped and order is preserved. Network seam
    (patched in tests)."""
    try:
        resp = requests.get(f'{url}/api/tags', timeout=timeout)
        if resp.status_code >= 400:
            return []
        out, seen = [], set()
        for m in (resp.json().get('models') or []):
            if not isinstance(m, dict):
                continue
            for key in ('name', 'model'):
                v = (m.get(key) or '').strip()
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
        return out
    except Exception:
        return []


def _normalize_model_ref(ref: str) -> tuple:
    """Split an Ollama model reference into (repo, tag) for comparison, folding away
    only the two purely-transport differences that make the SAME model look different
    across `/api/tags` shapes:

      * a leading registry-host segment — `registry.ollama.ai/huihui_ai/x:t` or
        `localhost:5000/org/x:t` — is stripped (first path segment containing a '.'
        or a ':' is the host);
      * an implicit tag defaults to 'latest'.

    The publisher namespace and model name are PRESERVED ('huihui_ai/qwen3-vl-
    abliterated'), so an abliterated build never matches the vanilla one and two
    different publishers of the same name never collide. ('', '') for an empty ref.
    """
    ref = (ref or '').strip()
    if not ref:
        return ('', '')
    # The tag is the text after the LAST ':' — unless that ':' belongs to a host:port
    # (there's a '/' after it) or there is no ':' at all -> implicit 'latest'.
    repo, sep, tag = ref.rpartition(':')
    if not sep or '/' in tag:
        repo, tag = ref, 'latest'
    tag = tag or 'latest'
    segs = repo.split('/')
    if len(segs) > 1 and ('.' in segs[0] or ':' in segs[0]):   # leading registry host
        segs = segs[1:]
    return ('/'.join(segs), tag)


def _model_present(configured: str, names: list) -> bool:
    if not configured:
        return False
    if configured in names:                            # fast path: byte-exact match
        return True
    cfg_repo, cfg_tag = _normalize_model_ref(configured)
    if not cfg_repo:
        return False
    # A config value WITHOUT an explicit tag matches ANY tag of that repo (unchanged
    # semantics) — detected on the model segment so 'localhost:5000/x' isn't misread.
    cfg_has_tag = ':' in configured.rsplit('/', 1)[-1]
    for n in names:
        n_repo, n_tag = _normalize_model_ref(n)
        if n_repo and n_repo == cfg_repo and (not cfg_has_tag or n_tag == cfg_tag):
            return True
    return False


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


def probe_ollama_connection() -> dict:
    """The Settings 'Test' button for the Ollama card: an HONEST end-to-end check —
    the server is reachable AND the configured vision model is actually pulled.

    The old test target was probe_ollama (reachability only), so the green check
    disagreed with the Setup step / diagnostic model probe on the very same machine
    (issue #7: Test ✓ green while vision_model=no). This delegates to the SAME
    probe_ollama_model the Setup and diagnostic use, so all three now resolve through
    one seam and can never contradict — the same 'is_available() defers to the probe'
    unification as JoyCaption."""
    reach = probe_ollama()
    if not reach['ok']:
        return reach                              # not configured / unreachable — as-is
    return probe_ollama_model(reachable=True)


def ollama_diagnostic() -> dict:
    """Paste-safe Ollama snapshot for /api/diagnostic: the configured vision-model
    string alongside the model tags the probe actually sees at /api/tags. This is the
    pair a bug report needs to tell a genuine 'not pulled' from a name/shape mismatch
    (issue #7) — without it, a report can only say vision_model=no with no way to see
    that the model IS listed under a slightly different identifier. Model names are
    not secrets; the list is de-duplicated and capped (count + per-entry length) so a
    large local library can't bloat the pasted report."""
    url = (cfg.get('ollama.url') or '').rstrip('/')
    configured = cfg.get('ollama.vision_model') or ''
    tags = _ollama_tags(url) if url else []
    return {'vision_model': configured, 'tags_seen': [t[:80] for t in tags[:20]]}


def comfyui_runtime(timeout=3) -> dict:
    """Live ComfyUI runtime snapshot for /api/diagnostic: version, GPU + VRAM and
    the current queue depth. NETWORK — deliberately kept OUT of probe() (which must
    stay network-free, it runs on every /api/capabilities call); the diagnostic is a
    manual one-click action, so a couple of short GETs are fine here.

    Returns {} when ComfyUI isn't configured / doesn't answer. Paste-safe: only the
    ComfyUI version string, the GPU name + VRAM totals (GPU model is not identity)
    and the queue counts — never a path or a secret. Never raises."""
    api = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api:
        return {}
    out = {}
    try:
        r = requests.get(f'{api}/system_stats', timeout=timeout)
        if r.status_code == 200:
            j = r.json() or {}
            sysinfo = j.get('system') or {}
            out['version'] = ((sysinfo.get('comfyui_version') or '').strip() or None)
            devs = j.get('devices') or []
            if devs and isinstance(devs[0], dict):
                d0 = devs[0]
                name = (d0.get('name') or '').strip()
                if name:
                    out['gpu'] = name[:60]
                total = d0.get('vram_total')
                free = d0.get('vram_free')
                if isinstance(total, (int, float)) and total > 0:
                    out['vram_total_gb'] = round(total / 1024 ** 3, 1)
                if isinstance(free, (int, float)) and free >= 0:
                    out['vram_free_gb'] = round(free / 1024 ** 3, 1)
    except Exception:
        pass
    try:
        r = requests.get(f'{api}/queue', timeout=timeout)
        if r.status_code == 200:
            j = r.json() or {}
            out['queue_running'] = len(j.get('queue_running') or [])
            out['queue_pending'] = len(j.get('queue_pending') or [])
    except Exception:
        pass
    return out


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
    has_run = (d / 'run.py').exists()
    # is_file(), NOT exists(): the training launch gate (lora_training.is_installed)
    # checks is_file(), so a venv_python that resolves to a directory or a broken
    # link would make the diagnostic report ai-toolkit=yes while training still
    # says "not installed". Keep the two in lockstep so the diagnostic never lies.
    ok = has_run and bool(venv_python) and venv_python.is_file()
    if ok:
        return {'ok': True, 'detail': str(d)}
    if has_run:
        # The folder IS an ai-toolkit checkout — only the interpreter is
        # missing (no venv/.venv: conda/uv/system installs, user-reported).
        # Name the actionable fix instead of a blanket "invalid dir".
        return {'ok': False,
                'detail': (f'ai-toolkit found at {d} but no venv/.venv inside — '
                           'set its Python interpreter in Settings → Local tools')}
    return {'ok': False, 'detail': f'invalid aitoolkit dir: {d}'}


# JoyCaption's runtime deps that ai-toolkit does NOT ship: the training venv has
# torch/torchvision, but joycaption_infer.py also needs transformers (AutoTokenizer
# / LlavaForConditionalGeneration), bitsandbytes (the NF4 4-bit load) and accelerate
# (required by from_pretrained with a quantization_config). Missing any of these is
# the ModuleNotFoundError users hit (issue #6). One import expr = one cached probe.
_JOYCAPTION_IMPORTS = 'import transformers, bitsandbytes, accelerate'
_JOYCAPTION_INSTALL = 'transformers bitsandbytes accelerate'


def probe_joycaption(aitoolkit: dict | None = None) -> dict:
    """Honest JoyCaption readiness. The old probe declared it available on the mere
    existence of the script + a configured ai-toolkit, so the app offered JoyCaption
    and then crashed with `ModuleNotFoundError: No module named 'transformers'` when
    the batch actually ran (issue #6). This checks the ai-toolkit venv can really
    import the captioning deps, through the cached subprocess seam so probe() stays
    fast and network-free (a per-probe subprocess would be unacceptable).

    `detail` names what to do: the exact `<venv_python> -m pip install …` command so
    the user can fix it without reading a stack trace. NEVER installs anything."""
    if aitoolkit is None:
        aitoolkit = probe_aitoolkit()
    if not aitoolkit['ok']:
        return {'ok': False, 'detail': aitoolkit['detail']}
    script = cfg.BACKEND_DIR / 'infer' / 'joycaption_infer.py'
    if not script.exists():
        return {'ok': False, 'detail': f'{script.name} not found'}
    venv_python = cfg.aitoolkit_path('venv_python')
    ok = _cached_import('joycaption', str(venv_python), _JOYCAPTION_IMPORTS)
    if ok:
        return {'ok': True, 'detail': 'JoyCaption deps import OK'}
    return {'ok': False,
            'detail': (f'JoyCaption deps ({_JOYCAPTION_INSTALL.replace(" ", ", ")}) '
                       f'are not importable in the ai-toolkit venv — run: '
                       f'"{venv_python}" -m pip install {_JOYCAPTION_INSTALL}')}


VAST_API_BASE = 'https://console.vast.ai/api/v0'


def probe_vast() -> dict:
    """Live check of the vast.ai API key (used by the Settings 'Test' button).
    The capability gate itself is key-presence only — probe() must stay
    network-free for this entry (it runs on every /api/capabilities call)."""
    key = cfg.secret('VAST_API_KEY')
    if not key:
        return {'ok': False, 'detail': 'API key missing'}
    try:
        r = requests.get(f'{VAST_API_BASE}/users/current/',
                         headers={'Authorization': f'Bearer {key}'}, timeout=8)
        if r.status_code == 200:
            email = (r.json() or {}).get('email') or 'account'
            return {'ok': True, 'detail': f'connected as {email}'}
        return {'ok': False, 'detail': f'vast.ai returned HTTP {r.status_code}'}
    except Exception as e:
        return {'ok': False, 'detail': f'unreachable: {e}'}


def probe_face_scoring() -> dict:
    python = cfg.get('face_scoring.python') or sys.executable
    ok = _cached_import('face_scoring', python, 'import insightface, onnxruntime')
    return {'ok': ok, 'detail': 'insightface + onnxruntime import OK' if ok else 'import failed'}


def face_gpu_available() -> bool:
    """True only when the face interpreter can run InsightFace on CUDA — i.e.
    onnxruntime exposes CUDAExecutionProvider (needs onnxruntime-gpu + a working
    CUDA/cuDNN runtime). The stock face_scoring extra ships CPU onnxruntime, so
    this is False until the user installs onnxruntime-gpu into that interpreter.
    Same cached subprocess probe as the import checks (exit 0 == available)."""
    python = cfg.get('face_scoring.python') or sys.executable
    return _cached_import(
        'face_gpu', python,
        "import onnxruntime,sys; "
        "sys.exit(0 if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else 1)")


def probe_masks() -> dict:
    python = cfg.get('masks.python') or sys.executable
    ok = _cached_import('masks', python, 'import rembg')
    return {'ok': ok, 'detail': 'rembg import OK' if ok else 'import failed'}


def probe_bank_scoring() -> dict:
    """Bank scoring extra (CLIP aesthetic + NSFW + style). Dedicated interpreter
    key (bank_scoring.python), else the app's own. Same subprocess-import probe as
    the other ML extras — torch/open_clip/transformers must all import. When False,
    the bank's Score button is disabled with an install hint (never a mute ✗)."""
    python = cfg.get('bank_scoring.python') or sys.executable
    ok = _cached_import('bank_scoring', python, 'import torch, open_clip, transformers')
    return {'ok': ok,
            'detail': 'torch + open_clip + transformers import OK' if ok else 'import failed'}


def probe_watermark_inpaint() -> dict:
    """LaMa inpainting availability (simple-lama-inpainting, ML extra). Dedicated
    interpreter key, else reuse the ML python (masks.python) then sys.executable —
    same subprocess-probe pattern/timeout handling as probe_masks. When False the
    Clean pass still runs crop-only (LaMa-routed images are skipped, not failed)."""
    python = cfg.get('watermark.python') or cfg.get('masks.python') or sys.executable
    ok = _cached_import('watermark', python, 'import simple_lama_inpainting')
    return {'ok': ok, 'detail': 'simple-lama-inpainting import OK' if ok else 'import failed'}


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
    bs4/cloudscraper/instaloader ride along in the same install. Every module the
    scrape stack imports belongs here: an omission reads as "installed" while the
    source that needs it still raises at runtime (instaloader did, until 2026-07)."""
    import importlib.util
    missing = [m for m in ('curl_cffi', 'gallery_dl', 'bs4', 'cloudscraper', 'instaloader')
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
    # Roots come from the SAME resolver ComfyUI uses (base <models> folders + any
    # extra_model_paths.yaml roots), so the picker/probe list exactly what a running
    # ComfyUI would load. With no yaml the roots are the historical [unet,
    # diffusion_models] / [checkpoints], so the output is byte-for-byte unchanged.
    from .services import comfy_model_paths
    result = {'zimage': [], 'sdxl': [], 'krea': [], 'klein': []}
    try:
        models_dir = cfg.comfyui_dir('models')
    except Exception:
        models_dir = None
    # krea is historically scanned ONLY from the base <models>/unet folder; track it
    # so an extra root (treated like diffusion_models) doesn't change that bucket.
    unet_default = os.path.normpath(str(models_dir / 'unet')) if models_dir else None

    for root in comfy_model_paths.search_roots('diffusion_models'):
        root_path = Path(root)
        try:
            subfolders = [p for p in root_path.iterdir() if p.is_dir()]
        except OSError:
            continue
        krea_eligible = (os.path.normpath(root) == unet_default)
        for sub in subfolders:
            name = sub.name
            if _ZIMAGE_RE.search(name):
                result['zimage'].extend(_model_files(sub))
            # Any 'klein'-named subfolder counts: shared installs keep e.g.
            # diffusion_models/'Flux2 klein'/ (the KV variant) next to our canonical
            # unet/klein/ download — hiding it made the picker blind to models the
            # user already owns.
            elif 'klein' in name.lower():
                result['klein'].extend(_model_files(sub))
            elif krea_eligible and name.lower().startswith('krea'):
                result['krea'].extend(_model_files(sub))
        # Flat / Stability-Matrix layouts drop the model straight INTO
        # diffusion_models/ with no klein/ subfolder — scan the root's own files
        # too and bucket the 'klein'-named ones. These are bare names (no prefix),
        # which is exactly what UNETLoader loads for a file at the root of a
        # registered folder. Mirrors klein_edit_helper._klein_unet_folders so the
        # picker lists only what the resolver can build.
        for name in _model_files(root_path):
            if 'klein' in name.lower():
                result['klein'].append(name)

    result['klein'] = sorted(set(result['klein']))
    sdxl = []
    for root in comfy_model_paths.search_roots('checkpoints'):
        sdxl.extend(_model_files(Path(root)))
    result['sdxl'] = sdxl
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
    # Preference among installed vision models. The uncensored *abliterated* build wins
    # first: the app's describe/caption work is NSFW-heavy and the vanilla qwen3-vl
    # refuses it outright, so an abliterated model must beat a censored one even when the
    # censored one is an -instruct tag. WITHIN a tier we still prefer -instruct over the
    # Thinking variant (Thinking reasons out loud instead of captioning; see
    # get_vision_model), then anything non-thinking. First match wins (order preserved).
    lo = [(n, (n or '').lower()) for n in vls]
    vl = (next((n for n, low in lo if 'abliterated' in low and 'instruct' in low), '')
          or next((n for n, low in lo if 'abliterated' in low), '')
          or next((n for n, low in lo if 'instruct' in low), '')
          or next((n for n, low in lo if 'thinking' not in low), '')
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
    """A real ComfyUI install: classic (main.py at the root) OR the Desktop
    app's basedir (models/ + custom_nodes/, no main.py — a user had to
    symlink main.py to pass the old check). Everything the app does with this
    folder is SCAN models/, so that is the hard requirement."""
    try:
        if not (d / 'models').is_dir():
            return False
        return (d / 'main.py').exists() or (d / 'custom_nodes').is_dir()
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


def classify_comfyui_dir(path: str) -> dict:
    """Rich verdict on a user-entered ComfyUI folder, so the Setup wizard can say
    something ACTIONABLE the moment the field is edited (before any save) instead of
    a blanket "invalid". `resolve_comfyui_base` only splits valid/nested/other; this
    keeps that split and additionally names WHY a folder isn't ComfyUI:

      status ∈
        'empty'       — nothing typed yet (the caller drives the skip flow).
        'valid'       — the folder itself is a ComfyUI install.
        'nested'      — the folder isn't, but ``<folder>/ComfyUI`` is (the launcher/
                        portable-wrapper mistake). `suggestion` = that child, to adopt.
        'missing'     — the path doesn't exist on disk.
        'empty_dir'   — the directory exists but is empty.
        'not_comfyui' — the directory (or a file at that path) exists and has content,
                        but holds no main.py/models/ and no child ComfyUI.

    `resolved` is the path a valid/nested verdict would adopt (child for nested,
    the folder itself otherwise). Pure + never raises — a filesystem hiccup degrades
    to 'not_comfyui' rather than throwing into the request."""
    raw = (path or '').strip()
    if not raw:
        return {'status': 'empty', 'resolved': '', 'suggestion': ''}
    p = Path(raw)
    if _is_comfyui_dir(p):
        return {'status': 'valid', 'resolved': str(p), 'suggestion': ''}
    child = p / 'ComfyUI'
    if _is_comfyui_dir(child):
        return {'status': 'nested', 'resolved': str(child), 'suggestion': str(child)}
    try:
        exists, is_dir = p.exists(), p.is_dir()
    except OSError:
        exists, is_dir = False, False
    if not exists:
        return {'status': 'missing', 'resolved': str(p), 'suggestion': ''}
    if is_dir:
        try:
            is_empty = not any(p.iterdir())
        except OSError:
            is_empty = False
        if is_empty:
            return {'status': 'empty_dir', 'resolved': str(p), 'suggestion': ''}
    # A file at that path, or a non-empty folder that simply isn't a ComfyUI checkout.
    return {'status': 'not_comfyui', 'resolved': str(p), 'suggestion': ''}


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
    ollama_installed = probe_ollama_installed()
    aitoolkit = probe_aitoolkit()
    gemini = probe_gemini()
    openai_ = probe_openai()
    face_scoring = probe_face_scoring()
    masks = probe_masks()
    bank_scoring = probe_bank_scoring()
    watermark_inpaint = probe_watermark_inpaint()
    joycaption = probe_joycaption(aitoolkit)
    models = _scan_models()
    # Klein engine readiness is now honest tri-component: the graph needs the UNET
    # AND the VAE AND the text-encoder. All three gate on the RESOLVER (the exact
    # value the generate would feed each loader node), not the raw scan — so the
    # UNET check matches the vae/te ones and a model the resolver can build but the
    # old bool(models['klein']) scan structure differed on can never disagree with
    # it (picker == probe == resolver). The old unet-only check also lit the engine
    # while a generate would 409 for the missing vae/te; that 409 already names +
    # auto-downloads the gap, so a badge that flips to "not ready" here is
    # actionable. Resolvers are cheap listdir, network-free, and
    # extra_model_paths-aware. Lazy import avoids an import cycle.
    from .services import klein_edit_helper as _keh
    # Per-asset gaps (setup_installer action names still absent on disk), so the
    # Setup UI can name exactly what's missing and keep only the relevant download
    # buttons visible — judging the whole engine on the UNET alone let the step go
    # green and hid the TE/VAE buttons the moment the model landed. Disk-only
    # (network-free) and reachability-independent, so the front can separate
    # "ComfyUI unreachable" from "an asset is missing".
    klein_missing = _keh.klein_missing_assets()
    # Present-but-INVALID assets: the file EXISTS under the resolved name but is not
    # real weights — a licence-gate HTML page saved as .safetensors (the #help crash:
    # Setup green, then UNETLoader dies on "Expecting value: line 1 column 1"), a
    # truncated download, or a suspiciously tiny stub. Header-only + cached, so cheap
    # here. A REQUIRED asset that is *blocking*-invalid must NOT light the engine
    # green (advisory too_small does not gate) — an honest badge points the user at
    # the broken file instead of letting a doomed generate crash ComfyUI.
    klein_invalid = _keh.klein_invalid_assets()
    klein_blocking_invalid = any(
        i['blocking'] and i['asset'] in _keh.KLEIN_REQUIRED for i in klein_invalid)
    klein_ready = (comfy['ok'] and bool(_keh.resolve_klein_unet())
                   and bool(_keh.resolve_klein_vae())
                   and bool(_keh.resolve_klein_text_encoder())
                   and not klein_blocking_invalid)
    base_dir = cfg.get('comfyui.base_dir') or ''
    comfy_dir = resolve_comfyui_base(base_dir)
    # Conscious "continue without ComfyUI" skip (Setup wizard). DERIVED, not just the
    # stored flag: a directory being configured ANNULS the skip on the spot, so the
    # flag can never mask a real error of a set-up ComfyUI — a configured install
    # always has base_dir, so `skipped` is false whenever there's something to error
    # on. The engine/studio gates below are computed independently of this and stay
    # the source of truth; `skipped` only lets the Setup step render neutral.
    comfy_skipped = bool(cfg.get('comfyui.setup_skipped')) and not base_dir

    from .services import chatgpt_oauth
    sub_status = chatgpt_oauth.status()

    caps = {
        'configured': cfg.is_configured(),
        'engines': {
            'nanobanana': gemini['ok'],
            'chatgpt': openai_['ok'],
            'klein': klein_ready,
        },
        'chatgpt_subscription': {
            'connected': sub_status['connected'],
            'email': sub_status['email'],
            'plan': sub_status['plan'],
            'codex_cli_detected': chatgpt_oauth.codex_auth_path().is_file(),
        },
        'comfyui': {
            'reachable': comfy['ok'],
            'api_url': cfg.get('comfyui.api_url') or '',
            'base_dir': base_dir,
            'dir_configured': bool(base_dir),
            'dir_valid': comfy_dir['valid'],       # base_dir really is a ComfyUI install
            'resolved_dir': comfy_dir['resolved'],
            # Effective "continue without ComfyUI" state: the user chose to skip AND no
            # directory is configured. Only drives the Setup step's neutral "skipped"
            # display — never the engine/studio gates below, so it cannot hide a real
            # error of a configured ComfyUI (which always has a base_dir → skipped=False).
            'skipped': comfy_skipped,
            'models': models,
            # setup_installer action names for the Klein assets NOT yet on disk
            # (subset of klein_model / klein_text_encoder / klein_vae / klein_lora).
            # Empty required-trio => the Klein engine is asset-ready.
            'klein_missing': klein_missing,
            # Klein assets PRESENT on disk but not real, loadable weights:
            # [{asset, filename, verdict, blocking, reason}]. Distinct from
            # klein_missing (the file exists, it just can't load) — drives the Setup
            # "present but INVALID: <asset> (<reason>)" line and the diagnostic, and
            # a blocking-invalid required asset also keeps engines.klein dark above.
            'klein_invalid': klein_invalid,
        },
        'ollama': {
            'reachable': ollama['ok'],
            # Installed = binary on disk, even when the server is stopped. The UI
            # reads (installed, reachable) as three states: not installed /
            # installed-but-stopped (→ "Start Ollama" button) / running.
            'installed': ollama_installed['ok'],
            'binary_path': ollama_installed['binary_path'],   # local-only; drives the Start route
            'url': cfg.get('ollama.url') or '',
            'vision_model': cfg.get('ollama.vision_model') or '',
            'vision_model_ready': probe_ollama_model(reachable=ollama['ok'])['ok'],
        },
        'aitoolkit': {
            'configured': bool(cfg.get('aitoolkit.dir')),
            'valid': aitoolkit['ok'],
        },
        'cloud_training': bool(cfg.secret('VAST_API_KEY')),
        # Publish-to-HF is gated purely on the HF_TOKEN secret being present (the
        # write-scope check is a live preflight at publish time, not here — probe()
        # must stay network-free). The ⋯ More menu entry keys off this.
        'hf_publish': bool(cfg.secret('HF_TOKEN')),
        'captioners': {
            # Honest: the ai-toolkit venv must actually import the JoyCaption deps,
            # not merely have the script on disk (issue #6). `detail` carries the
            # exact pip command when it can't, so the UI/error can name the fix.
            'joycaption': joycaption['ok'],
            'joycaption_detail': joycaption['detail'],
            'ollama': ollama['ok'],
        },
        'face_scoring': face_scoring['ok'],
        'masks': masks['ok'],
        # Bank scoring extra (CLIP aesthetic + NSFW + style clustering). Gates the
        # bank's "Score (aesthetic · NSFW · style)" button; False → install hint.
        'bank_scoring': bank_scoring['ok'],
        # Lets the front adapt the watermark Clean tooltip: when False, Clean is
        # crop-only (LaMa-routed watermarks are skipped with an install hint).
        'watermark_inpaint': watermark_inpaint['ok'],
        # Klein-inpaint (V2, quality) readiness = same as the Klein engine (ComfyUI
        # reachable + Klein models on disk). The custom-node preflight is a clean-time
        # 409. Greys the batch's "Klein (quality)" option when False.
        'watermark_klein': klein_ready,
        # Persisted "allow automatic crop" preference (Settings ▸ Watermark inpainting).
        # The batch Clean bar reads it here to seed/reflect its inline toggle and the
        # review lightbox uses it as the per-image crop-vs-inpaint default; when False,
        # auto-routing repaints border marks instead of cropping them.
        'watermark_allow_crop': bool(cfg.get('watermark.allow_crop')),
        'python': python_ml_status(),
        'scrape_deps': probe_scrape_deps()['ok'],
        'training_visible': aitoolkit['ok'] or bool(cfg.secret('VAST_API_KEY')),
        'studio_visible': comfy['ok'],
    }

    _cache, _cache_ts = caps, now
    return copy.deepcopy(caps)
