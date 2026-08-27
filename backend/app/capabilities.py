"""Capability probes: what's actually reachable/configured right now.

Consumed by the Settings UI (engine cards, "Test connection" buttons) and by
feature gating elsewhere in the app. `_http_ok` is the single network seam —
every reachability probe goes through it so tests can patch one symbol.
`_import_ok` is the equivalent seam for the slow subprocess import-probes.
"""
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from . import config as cfg
from .services import ffmpeg_tools
from .services import infer_env
from .utils import comfy_fs

_CACHE_TTL = 30
_cache = None
_cache_ts = 0.0

_IMPORT_TTL = 600
# How long an UNKNOWN verdict (the probe never answered) is remembered. Short,
# because it must re-try soon against a warm import — but not zero: the Bank
# panel polls its readiness every ~2 s, and an uncached unknown meant a fresh
# 90 s `import torch` subprocess on EVERY poll.
_UNKNOWN_TTL = 60
# Budget for one cold import probe. Aligned with services.scoring_python
# .PROBE_TIMEOUT (90 s), which the repo already documents as the honest floor
# for a cold `import torch` behind an antivirus: at 60 s the SAME interpreter
# answered 'CUDA' to one probe and 'no answer' to the other.
_IMPORT_TIMEOUT = 90
_import_cache = {}  # key -> (ts, ok|None)  — None = unknown, kept briefly
# EVERY probe that vouches for a ``backend/infer/*`` worker runs in the exact
# environment that worker runs in — isolated from the process owner's user
# site-packages (``services.infer_env``). The two must move together: a probe
# isolated while its worker is not reports ✓ against an environment the worker
# never sees, and the failure then lands per ITEM, hours later, as if the data
# were at fault. That asymmetry is what turned an unrelated package left in a
# user site-packages into a bank of 861 shots marked permanently unreadable.
#
# `infer_env.is_borrowed` draws the first line and states why: OUR interpreter
# keeps its user site, because on a system-Python install that directory is
# where pip actually put torch. These keys draw the second one — probes whose
# real launch we do not shape, and which must therefore not be sanitised either:
#   * ai-toolkit's TRAINING venv ('aitoolkit_torch' / 'aitoolkit_alive'). Note
#     that JoyCaption runs in that SAME venv and IS isolated: the rule is MATCH
#     THE LAUNCH, not "whose environment is it" — JoyCaption's script is ours and
#     we build its argv, while the training bridge hands the venv over untouched.
#   * the interpreter probed for PyAV/cv2 ('video_decode'): that decode happens
#     in-process, inside Flask, which has no ``-s`` of its own — so even a
#     configured `video.python` must be asked the unisolated question.
_USER_SITE_IMPORT_KEYS = frozenset((
    'aitoolkit_torch', 'aitoolkit_alive', 'video_decode'))

_ZIMAGE_RE = re.compile(r'z[ -]?image', re.IGNORECASE)
# Aligned with klein_edit_helper / utils.comfyui (was missing '.sft', so the
# picker under-listed vs the resolvers). '.gguf' is listed but NOT loadable by the
# shipped graphs — they emit core `UNETLoader`, which has no '.gguf' support (core
# ComfyUI's supported_pt_extensions does not include it) and would need the
# ComfyUI-GGUF pack's separate `UnetLoaderGGUF` node, which nothing here emits.
# It stays listed so a .gguf a user already has is not silently invisible; what
# makes that honest is `utils.comfyui.unavailable_model_files`, which names the
# extension as the cause before anything is queued (naniii2352, Discord).
_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')


def _http_ok(url, timeout=3, reason=None, *, readiness=False) -> bool:
    """True when `url` answers with anything below 500.

    `reason`, when a dict is passed, is FILLED with why a False was returned:
    'timeout' (the server accepted the connection and then didn't answer in time —
    it is up, it is slow) or 'unreachable' (nothing answered). It is an out-param
    rather than a richer return type on purpose: this function is the single
    network seam the whole test suite patches (`lambda *a, **k: False`), and a
    stub that ignores the dict simply leaves the reason unknown instead of
    breaking. Callers must treat an unfilled dict as "don't know"."""
    effective_timeout = timeout
    request_options = {}
    if readiness:
        # Setup polls this endpoint continuously during managed-container boot.
        # Keep both halves of the request bounded even if a future caller passes
        # a larger value, do not follow a redirect to an arbitrary/body-heavy
        # destination, and never download the response body.
        try:
            effective_timeout = max(0.05, min(float(timeout), _SETUP_READINESS_TIMEOUT))
        except (TypeError, ValueError):
            effective_timeout = _SETUP_READINESS_TIMEOUT
        request_options = {'allow_redirects': False, 'stream': True}
    resp = None
    try:
        resp = requests.get(url, timeout=effective_timeout, **request_options)
        return resp.status_code < 500
    except Exception as e:
        if isinstance(reason, dict):
            reason['why'] = ('timeout' if isinstance(e, requests.exceptions.Timeout)
                             else 'unreachable')
            reason['waited'] = effective_timeout
        return False
    finally:
        if readiness and resp is not None:
            close = getattr(resp, 'close', None)
            if callable(close):
                close()


def _import_ok(python, module_expr: str, timeout=_IMPORT_TIMEOUT):
    """True/False = the import deterministically succeeded/failed. None = TIMEOUT —
    unknown, NOT a proven absence. The very first `import rembg` after an install
    compiles numba/scikit-image caches while the antivirus scans 40 MB of fresh
    DLLs: measured ~20 s cold vs ~1 s warm — a 20 s timeout read as False showed
    'Person masks ✗' for 10 min right after a SUCCESSFUL install.

    ``python`` is normally one executable path.  The cache layer passes an argv
    prefix such as ``(python, '-s')`` for every feature whose real worker uses
    the isolated contract — every BORROWED interpreter but the ones named in
    ``_USER_SITE_IMPORT_KEYS``.  ``env`` then carries the inherited half of the
    same instruction, so the probe and the worker see one environment, not two.
    """
    try:
        prefix = (list(python) if isinstance(python, (tuple, list))
                  else [python])
        isolated = infer_env.NO_USER_SITE_FLAG in prefix[1:]
        result = subprocess.run(
            [*prefix, '-c', module_expr], capture_output=True, timeout=timeout,
            env=infer_env.worker_env(prefix[0]) if isolated else None)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return False


def _cached_import_state(key: str, python: str, module_expr: str):
    """Three-valued, cached import probe: True / False / None.

    None means the probe DID NOT ANSWER — a cold-import timeout, not a proven
    absence. Callers that only gate a feature can flatten it to False
    (_cached_import); callers whose wrong answer costs something real — the
    GPU-exclusive window — must be able to tell the two apart.

    Both verdicts are cached, with very different lifetimes: a decided answer
    for _IMPORT_TTL (a venv does not change between two probes), an unknown for
    _UNKNOWN_TTL so it re-tries soon against a now-warm import WITHOUT spawning
    a fresh 90 s subprocess on every 2 s poll of the Bank panel."""
    now = time.time()
    cache_key = f'{key}:{python}:{module_expr}'
    cached = _import_cache.get(cache_key)
    if cached is not None:
        ttl = _IMPORT_TTL if cached[1] is not None else _UNKNOWN_TTL
        if now - cached[0] < ttl:
            return cached[1]
    probe_python = (python if (key in _USER_SITE_IMPORT_KEYS
                               or not infer_env.is_borrowed(python))
                    else (python, infer_env.NO_USER_SITE_FLAG))
    ok = _import_ok(probe_python, module_expr)
    _import_cache[cache_key] = (now, ok)
    return ok


def _cached_import(key: str, python: str, module_expr: str) -> bool:
    """The boolean gate: an unknown reads as not-ready (nothing is offered on a
    capability we could not prove), but it is no longer re-probed on every call."""
    return _cached_import_state(key, python, module_expr) is True


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


def probe_openrouter() -> dict:
    """OpenRouter engine readiness. Presence of the key only — same contract as
    probe_gemini. Deliberately NOT a live call: verifying the key would mean an
    authenticated request on every capability poll, and OpenRouter bills per
    request; a key that is set but out of credits is reported at generation time,
    with the provider's own words."""
    ok = bool(cfg.secret('OPENROUTER_API_KEY'))
    return {'ok': ok, 'detail': 'key set' if ok else 'key missing'}


def _object_info_timeout() -> int:
    """The /object_info read budget, published so the UI can quote it. Lazy import:
    utils.comfyui imports config, not capabilities, and this keeps it that way."""
    from .utils import comfyui as _cu
    return _cu.object_info_timeout()


def _dataset_import_policy() -> dict:
    """The effective import resolution/encoding, published for the UI. Lazy
    import for the same reason as _object_info_timeout: the service imports
    config, not capabilities."""
    from .services import face_dataset_service as _fds
    p = _fds.import_encode_policy()
    return {'max_side': p['max_side'], 'encoding': p['encoding'],
            'capped': p['capped'], 'ceiling': p['ceiling'],
            'input_max_side': p['input_max_side'],
            'input_max_pixels': p['input_max_pixels'],
            'preserve_max_side': p['preserve_max_side'],
            'preserve_max_pixels': p['preserve_max_pixels']}


def comfyui_down_message(status, waited) -> str:
    """THE sentence for a ComfyUI that isn't answering. Two causes, two remedies —
    never one fits-all line.

    'ComfyUI isn't running' used to be said for BOTH, and it is a lie in one of
    them: a heavily-customised ComfyUI is up and simply takes longer than the
    budget to enumerate its nodes and models. Someone who has just started ComfyUI
    and reads "it isn't running" goes and checks the one thing they already know is
    true, and finds nothing (j_o_e_l., Discord — he then measured the real number
    himself). So the slow case names the delay, says the server IS up, and points
    at the knob that fixes it."""
    if status == 'timeout':
        return (f'ComfyUI took more than {waited}s to answer. It is running — it is '
                f'slow to enumerate its nodes and model files, which is normal on an '
                f'install with many custom nodes. Raise "ComfyUI response timeout" in '
                f'Settings ▸ Local tools ▸ ComfyUI.')
    return ('No answer from ComfyUI — nothing is listening at that address. Start '
            'ComfyUI, or correct the ComfyUI API URL in Settings ▸ Local tools.')


def probe_comfyui() -> dict:
    """{ok, detail, status, hint}. `status` is 'ok' / 'slow' / 'unreachable' /
    'unconfigured' — the two failure modes are published SEPARATELY so every
    surface (Test button, engine cards, the 409 on a blocked generation) can say
    the true one instead of the convenient one.

    The verdict itself still rides on the cheap `/history` probe. When that one
    can't tell us why (it is the patched seam in tests, and a 3 s budget is short
    enough to trip on a busy server), the LAST /object_info attempt is consulted:
    that probe knows the difference first-hand, because it is the one that spends
    the long budget."""
    api_url = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api_url:
        return {'ok': False, 'detail': 'comfyui.api_url not configured',
                'status': 'unconfigured',
                'hint': 'Set the ComfyUI API URL in Settings ▸ Local tools.'}
    reason = {}
    ok = _http_ok(f'{api_url}/history', reason=reason)
    if ok:
        return {'ok': True, 'detail': api_url, 'status': 'ok', 'hint': ''}
    from .utils import comfyui as _cu
    health = _cu.object_info_health()
    why, waited = reason.get('why'), reason.get('waited', 3)
    if why != 'timeout' and health['status'] == 'timeout':
        # /history gave up after 3 s while the heavy probe proved the server is
        # THERE and merely slow. Believe the probe that waited longer.
        why, waited = 'timeout', health['waited']
    status = 'slow' if why == 'timeout' else 'unreachable'
    return {'ok': False, 'status': status,
            'detail': (f'slow: {api_url} (>{waited}s)' if status == 'slow'
                       else f'unreachable: {api_url}'),
            'hint': comfyui_down_message('timeout' if status == 'slow' else 'down', waited)}


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


def probe_ollama_model(reachable=None, model=None) -> dict:
    # `reachable` lets probe() pass the reachability it already computed, so we
    # don't re-hit /api/tags a second time (and don't pay a second blocking
    # timeout when Ollama is configured-but-down). `model` lets a per-dataset
    # caption action verify its effective override; omitted -> the configured
    # global vision model exactly as before. Called standalone -> we probe.
    url = (cfg.get('ollama.url') or '').rstrip('/')
    model = (cfg.get('ollama.vision_model') or '') if model is None else model
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
    # The encoder verdict is a probe too (it RUNS ffmpeg), cached the same way —
    # so it has to be dropped here or the video row keeps its pre-install ✗ for
    # ten minutes after the install that fixed it.
    ffmpeg_tools.clear_cache()
    _cache = None
    _cache_ts = 0.0


# Where an ai-toolkit checkout keeps the Python that runs it. There is NO single
# answer: the README's `venv/`, a `.venv/`, a conda env, or a portable bundle that
# ships `python_embeded/python.exe` (a community easy-install script does exactly
# that — reported on Reddit by Psyko_2000). So we do not guess a layout: we knock
# on the folder and on each of its immediate sub-folders with every interpreter
# shape the app already knows (`scoring_python._INTERPRETER_SPOTS`) and keep the
# ones that answer as a real file. An empty list means "we found nothing here",
# which is a fact the UI can state honestly — never a claim that a venv is the
# only way. Nothing is executed and nothing is written; it is a stat() sweep one
# level deep, computed only for a checkout whose interpreter we could not resolve.
def aitoolkit_python_candidates(root, limit: int = 4) -> list:
    from .services.scoring_python import interpreters_in
    try:
        root = Path(root)
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []
    out, seen = [], set()
    for folder in (root, *children[:64]):
        for cand in interpreters_in(folder):
            key = os.path.normcase(str(cand))
            if key in seen:
                continue
            seen.add(key)
            out.append(str(cand))
            if len(out) >= limit:
                return out
    return out


def probe_aitoolkit() -> dict:
    d = cfg.aitoolkit_path('dir')
    if not d:
        return {'ok': False, 'detail': 'aitoolkit.dir not configured',
                'has_run': False, 'python_candidates': []}
    venv_python = cfg.aitoolkit_path('venv_python')
    has_run = (d / 'run.py').exists()
    # is_file(), NOT exists(): the training launch gate (lora_training.is_installed)
    # checks is_file(), so a venv_python that resolves to a directory or a broken
    # link would make the diagnostic report ai-toolkit=yes while training still
    # says "not installed". Keep the two in lockstep so the diagnostic never lies.
    ok = has_run and bool(venv_python) and venv_python.is_file()
    if ok:
        return {'ok': True, 'detail': str(d), 'has_run': True,
                'python_candidates': []}
    if has_run:
        # The folder IS an ai-toolkit checkout — we just could not see which
        # Python runs it (conda/uv/system/portable installs all land here).
        # State the finding and hand over both ways out.
        found = aitoolkit_python_candidates(d)
        detail = (f'ai-toolkit found at {d} but no Python interpreter found '
                  'inside — create a venv there, or set its Python interpreter '
                  'in Settings → Local tools')
        if found:
            detail += f' (candidate: {found[0]})'
        return {'ok': False, 'detail': detail, 'has_run': True,
                'python_candidates': found}
    return {'ok': False, 'detail': f'invalid aitoolkit dir: {d}',
            'has_run': False, 'python_candidates': []}


def _torch_import_state(python: str):
    """`import torch` on this interpreter: True / False / None (unknown).

    Stricter than `_cached_import_state` about what counts as a FAILED IMPORT.
    That helper's seam reports False for anything that goes wrong, including a
    path that cannot be executed at all — and "this file is not a working Python"
    is a different problem from "this Python has no torch", with a different
    sentence and, above all, one we must not state as a fact when a training
    launch hangs on it. So a failed import is confirmed by a second, trivial
    `pass` probe: if the interpreter cannot even run that, the answer is UNKNOWN.
    The extra subprocess only ever runs in the already-failing case, and both
    answers go through the same cache."""
    state = _cached_import_state('aitoolkit_torch', python, 'import torch')
    if state is not False:
        return state
    if _cached_import_state('aitoolkit_alive', python, 'pass') is not True:
        return None
    return False


def aitoolkit_interpreter_report() -> dict:
    """Can the interpreter ai-toolkit is configured with actually `import torch`?

    {'python': str, 'torch': True|False|None, 'alternative': str}.

    `torch` is THREE-valued on purpose and reuses the same cached subprocess seam
    every other ML capability goes through: True = proven importable, False =
    proven not, None = we did not find out (no interpreter on disk, or a cold
    import that timed out). None must never be treated as a refusal — a first
    `import torch` on a cold venv behind an antivirus takes tens of seconds.

    `alternative` is the venv the checkout carries (`venv/`, `.venv/`) when an
    EXPLICIT `aitoolkit.python` is set, is broken, and that venv works — the
    swap we can then offer instead of leaving the user to guess. Empty
    otherwise; it costs a second probe only in the already-broken case.

    Deliberately NOT part of probe(): probe() is polled and must stay cheap and
    never spawn a torch import. This is called from a launch attempt, from the
    Test button, and from a crash report — moments where the answer is worth
    paying for."""
    out = {'python': '', 'torch': None, 'alternative': ''}
    python = cfg.aitoolkit_path('venv_python')
    if not python or not Path(python).is_file():
        return out
    out['python'] = str(python)
    out['torch'] = _torch_import_state(str(python))
    if out['torch'] is not False:
        return out
    # Broken, and only because of an explicit override? Then the checkout's own
    # venv is the obvious way out — but only claim it after PROVING it works.
    if not (cfg.get('aitoolkit.python') or '').strip():
        return out
    derived = cfg.aitoolkit_path('venv_python_derived')
    if derived and Path(derived).is_file() and os.path.normcase(str(derived)) \
            != os.path.normcase(str(python)):
        if _torch_import_state(str(derived)) is True:
            out['alternative'] = str(derived)
    return out


def probe_aitoolkit_test() -> dict:
    """The Settings ▸ Local tools "Test" button for ai-toolkit. Same folder checks
    as probe_aitoolkit, plus the one the folder checks cannot see: does the chosen
    interpreter have torch? A Test that goes green on a Python without torch is
    the exact trap of GitHub #19 (strouder) — everything configured, every run
    dead on `No module named 'torch'`. An UNKNOWN probe keeps the green: we refuse
    to fail a test on an answer we do not have."""
    result = probe_aitoolkit()
    if not result.get('ok'):
        return result
    from .services.training_diagnostics import interpreter_verdict
    report = aitoolkit_interpreter_report()
    verdict = interpreter_verdict(report['python'], report['torch'],
                                  alternative=report['alternative'])
    if verdict:
        return {**result, 'ok': False, 'detail': verdict['message']}
    return result


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


# The EXACT import expression each ML capability's probe runs, keyed by the
# setup_installer action name that installs it. It lives here, once, because the
# installer's post-install verification must re-run the SAME import the probe
# will: when the two drift, an install can report success while the capability
# stays ✗ with no reason shown anywhere (issue #24, 1Tomber — the masks install
# said "already satisfied" for every package it knew about while `import rembg`
# died on a dependency that was not in its list).
CAPABILITY_IMPORTS = {
    'face_scoring': 'import insightface, onnxruntime',
    'masks': 'import rembg',
    # numpy and PIL are named even though nothing pip-installs them directly:
    # they arrive transitively (torchvision, which open_clip_torch and timm both
    # require, hard-requires both) and BOTH workers that run in this interpreter
    # import them on any real call — infer/bank_score_infer.py reaches PIL at load
    # time through bank_image_guard and numpy inside its scoring path, and
    # infer/video_ai_check_infer.py imports both in _preprocess. Not at module
    # scope in every case, which changes WHEN it fails and not whether: a probe
    # that names only the headline packages reports ✓ while the feature dies on
    # the first call, which is issue #24's exact shape; this entry under-imported
    # for four waves and the AI check is what surfaced it.
    # `from PIL import Image` rather than `import PIL`, because the bare package
    # imports without its submodules and would pass on a broken install.
    # setup_installer._verify_bank_scoring_import runs THIS string, so the
    # install's honesty gate cannot fall behind the probe again.
    'bank_scoring': ('import torch, open_clip, transformers, numpy; '
                     'from PIL import Image'),
    'bank_siglip2': ('import torch, transformers, numpy; from PIL import Image; '
                     'from transformers import Siglip2Model, AutoProcessor'),
    'watermark_inpaint': 'import simple_lama_inpainting',
    # The detector extra runs backend/infer/watermark_detect_infer.py, which needs
    # torch (both models) and transformers (BOTH heads are transformers-native —
    # that is precisely why Grounding DINO was chosen over Florence-2, whose
    # trust_remote_code file no longer loads). Nothing else: no einops, no
    # flash-attn, no vendored modelling code.
    'watermark_detect': 'import torch, transformers',
    # The video lane is TWO extras, because its two halves belong in two different
    # environments. PyAV is imported IN-PROCESS by Flask (probing a file, pulling a
    # thumbnail frame), so it has to live in the app's own interpreter and stays
    # small. TransNetV2 drags torch, so it rides the environment bank scoring
    # already manages — the same call the watermark detector made, for the same
    # ~2.5 GB reason. ffmpeg is not an import at all and is resolved separately
    # (services/ffmpeg_tools).
    # cv2 and numpy are named because the camera-motion pass tracks corners and
    # fits a similarity transform IN THIS PROCESS, on the frames PyAV hands it —
    # so a probe that names only `av` reports the video extra ready and then the
    # pass dies on its first clip. Issue #24's exact shape, and the second time
    # this list has under-imported: the AI check surfaced the same omission in
    # `bank_scoring`. setup_installer's 'video' package tuple installs both.
    'video': 'import av, cv2, numpy',
    # av: the worker decodes with PyAV in this same environment — a probe that
    # skips it answers "ready" about a detector that cannot open a single file.
    'shot_detect': 'import torch, transnetv2_pytorch, av',
    # 🔳 Burned-in text — the safe-zone pass AND the image lanes' 🔤 Find text.
    # ALL FOUR names, because infer/video_text_infer.py imports all four:
    # rapidocr_onnxruntime is the engine; cv2 reads a frame and learns its size
    # (the boxes come back normalised, which needs one); numpy + PIL back the
    # reader's fallbacks for paths cv2.imread cannot open on Windows (any
    # non-ASCII path — measured on a real bank) while keeping the EXIF-upright
    # geometry. cv2 arrives as a DEPENDENCY of the engine rather than in its
    # own right, which is exactly the shape that made issue #24 — a probe
    # importing only the headline module reports ✓ while the feature dies on
    # the first call; test_video_safe_zone pins this list to the worker's real
    # imports so a fifth dependency cannot ship unprobed. There is nothing
    # else: no torch, no transformers, and no model download (the 1.4.x wheels
    # carry the PP-OCRv4 ONNX files, ~16 MB, so this works offline the day it
    # is installed).
    'video_text': 'import rapidocr_onnxruntime, cv2, numpy, PIL',
}


def probe_face_scoring() -> dict:
    python = cfg.get('face_scoring.python') or sys.executable
    ok = _cached_import('face_scoring', python, CAPABILITY_IMPORTS['face_scoring'])
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


def resolve_face_device():
    """(device, use_gpu) for a face pass — the SINGLE answer both surfaces use.

    The Bank pass and the dataset scorer ask the identical question ("may I put
    InsightFace on the GPU for this run?"), so they must not answer it apart:
    a divergence here is invisible until someone notices one surface is ten
    times slower than the other. See the Bank/Dataset parity rule in CLAUDE.md.

    `face_scoring.device`: 'auto' (default - GPU when truly available) | 'cpu'
    | 'cuda'. A 'cuda' request on an interpreter without CUDA still degrades to
    CPU here, so the caller never opens the GPU-exclusive window for a pass that
    is going to run on CPU anyway.
    """
    pref = str(cfg.get('face_scoring.device') or 'auto').lower()
    use_gpu = pref in ('auto', 'cuda') and face_gpu_available()
    return ('cuda' if use_gpu else 'cpu'), use_gpu


def bank_scoring_gpu_available() -> bool:
    """True only when the bank-scoring interpreter can actually run torch on
    CUDA. The scoring child picks its own device (``cuda if
    torch.cuda.is_available()``), and the PARENT has to know the same answer:
    it decides whether to take the GPU-exclusive window, which unloads ComfyUI
    and blocks a training start for the whole pass. The stock extra installs
    CPU-only torch, so this is False until the user puts a CUDA build in that
    interpreter — and a pass that never touches the GPU must never hold it.

    UNKNOWN is NOT False here. 'the probe did not answer' and 'torch has no
    CUDA' used to collapse into the same answer, and they have opposite costs:
    the child decides on its own with `torch.cuda.is_available()`, so on a
    machine that HAS a card an unanswered probe means we may be leaving the GPU
    unprotected while the pass takes it — ComfyUI still loaded, a training start
    still allowed. So an unknown resolves to 'is there a card at all' (cached
    nvidia-smi read): a card-less machine still never holds a window it cannot
    use, and a machine with one is protected until the probe answers."""
    python = cfg.get('bank_scoring.python') or sys.executable
    state = _cached_import_state(
        'bank_scoring_gpu', python,
        'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)')
    if state is None:
        return gpu_vram_gb() is not None
    return state


def bank_siglip2_gpu_available() -> bool:
    """True only when the resolved SigLIP2 interpreter proves CUDA works.

    Unlike Score, the parent sends an explicit device to the SigLIP2 child. An
    unanswered probe must therefore resolve to CPU: guessing CUDA from the host
    card (or from Score's borrowed runtime) would tell a CPU-only managed torch
    build to use a device it cannot open.
    """
    from .services import bank_semantic_models as assets
    state = _cached_import_state(
        'bank_siglip2_gpu', assets.semantic_python(),
        'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)')
    return state is True


def probe_masks() -> dict:
    python = cfg.get('masks.python') or sys.executable
    ok = _cached_import('masks', python, CAPABILITY_IMPORTS['masks'])
    return {'ok': ok, 'detail': 'rembg import OK' if ok else 'import failed'}


def probe_video_text() -> dict:
    """Can this install read burned-in text? Used by 🔳 Safe zone.

    Deliberately NOT folded into `probe_video`'s three pieces. Those three are
    what the LANE needs to exist at all — without decoding there is no bank —
    while this one gates HALF of one optional pass: with no OCR engine the safe
    zone still measures letterbox and pillarbox bands and says so on every shot.
    Folding it in would grey out buttons that work.
    """
    python = cfg.get('video_text.python') or sys.executable
    ok = _cached_import('video_text', python, CAPABILITY_IMPORTS['video_text'])
    return {'ok': ok,
            'detail': 'RapidOCR import OK' if ok
                      else 'install the burned-in text extra from Setup'}


def probe_video() -> dict:
    """The video lane, reported as THREE pieces rather than one verdict.

    Decoding, shot detection and encoding fail independently and are fixed
    differently — one is a pip package, one is a pip package that drags torch, one
    is a binary. A single "video unavailable" is how a user ends up reinstalling
    the wrong thing, and it is the exact shape of the defect this lane is meant to
    avoid: a .mp4 in an image bank is skipped today with no message at all.

    'ok' means a bank can be taken all the way to a dataset. The parts are
    reported separately so a caller can still offer what does work — with no
    encoder you can scan, detect and triage, you just cannot export yet.
    """
    decode = _cached_import('video_decode', cfg.get('video.python') or sys.executable,
                            CAPABILITY_IMPORTS['video'])
    detect = _cached_import(
        'video_detect',
        (cfg.get('shot_detect.python') or cfg.get('bank_scoring.python')
         or sys.executable),
        CAPABILITY_IMPORTS['shot_detect'])
    # RUNS the binary rather than trusting that a file sits at the resolved path:
    # a truncated download and a quarantined stub both pass os.path.isfile and
    # then crash from inside an export. Cached at the same TTL as the import
    # probes, so this poll-path costs one subprocess per 10 min, not per call.
    encoder = ffmpeg_tools.ffmpeg_ready()
    encode = bool(encoder['ok'])
    missing = []
    if not decode:
        missing.append('av (video decoding)')
    if not detect:
        missing.append('shot detection (transnetv2-pytorch)')
    if not encode:
        missing.append(f"ffmpeg (clip encoding) — {encoder['reason']}")
    return {
        'ok': bool(decode and detect and encode),
        'detail': 'video extra ready' if not missing else 'missing: ' + ', '.join(missing),
        'decode': bool(decode),
        'detect': bool(detect),
        'encode': bool(encode),
    }


def probe_bank_scoring() -> dict:
    """Bank scoring extra (CLIP aesthetic + NSFW + style). Dedicated interpreter
    key (bank_scoring.python), else the app's own. Same subprocess-import probe as
    the other ML extras — torch/open_clip/transformers must all import. When False,
    the bank's Score button is disabled with an install hint (never a mute ✗)."""
    python = cfg.get('bank_scoring.python') or sys.executable
    ok = _cached_import('bank_scoring', python, CAPABILITY_IMPORTS['bank_scoring'])
    return {'ok': ok,
            'detail': 'torch + open_clip + transformers import OK' if ok else 'import failed'}


def probe_bank_siglip2() -> dict:
    """Optional Bank semantic engine: packages AND the pinned local checkpoint.

    Files are checked before importing torch so an install that never requested
    SigLIP2 does not pay a heavy subprocess probe on every capability poll.
    """
    from .services import bank_semantic_models as assets
    if not assets.weights_present():
        return {
            'ok': False,
            'detail': ('SigLIP2 weights are not downloaded yet '
                       '(Setup ▸ Quality tools ▸ SigLIP2 semantic engine)'),
            'model': assets.MODEL_ID,
        }
    python = assets.semantic_python()
    ok = _cached_import('bank_siglip2', python, CAPABILITY_IMPORTS['bank_siglip2'])
    return {
        'ok': ok,
        'detail': ('torch + transformers + Pillow + pinned SigLIP2 weights ready' if ok
                   else 'weights are present but this transformers build cannot load SigLIP2'),
        'model': assets.MODEL_ID,
    }


def probe_watermark_inpaint() -> dict:
    """LaMa inpainting availability (simple-lama-inpainting, ML extra). Dedicated
    interpreter key, else reuse the ML python (masks.python) then sys.executable —
    same subprocess-probe pattern/timeout handling as probe_masks. When False the
    Clean pass still runs crop-only (LaMa-routed images are skipped, not failed)."""
    python = cfg.get('watermark.python') or cfg.get('masks.python') or sys.executable
    ok = _cached_import('watermark', python, CAPABILITY_IMPORTS['watermark_inpaint'])
    return {'ok': ok, 'detail': 'simple-lama-inpainting import OK' if ok else 'import failed'}


def probe_watermark_detect() -> dict:
    """The dedicated watermark DETECTOR extra (SigLIP2 ranker + Grounding DINO
    locator). Dedicated interpreter key, else the bank-scoring one — which is not
    a fallback but the intended shared home: it already holds torch and
    transformers, and a second copy would cost the user another ~2.5 GB.

    Importing is necessary but NOT sufficient: the weights (~0.9 GB) must also be
    on disk, and an environment that imports torch while the models were never
    downloaded would light this capability green and then fail a whole pass with a
    network error. So the presence of the model cache is part of the verdict, and
    the two failures are reported apart — 'the packages are missing' and 'the
    weights are missing' send the user to different buttons.

    False here is never a refusal: the Find pass keeps using the vision model,
    exactly as it does today (fail-open).

    ORDER IS DELIBERATE — the filesystem check runs FIRST. The import check is a
    subprocess that runs `import torch`, and on the overwhelmingly common machine
    (this extra not installed) that subprocess can never change the answer, since
    no weights means not-ready whatever imports. Probing the other way round made
    every capability poll — and every test that drops the probe cache — pay a
    torch import for a capability nobody has. That cost is paid by every agent and
    every CI run, forever, which is precisely how a suite drifts from minutes to
    an hour."""
    if not watermark_detect_weights_present():
        return {'ok': False,
                'detail': 'the detector weights are not downloaded yet '
                          '(Setup ▸ Quality tools ▸ Watermark detector)'}
    python = cfg.get('watermark_detect.python') or cfg.get('bank_scoring.python') or sys.executable
    ok = _cached_import('watermark_detect', python,
                        CAPABILITY_IMPORTS['watermark_detect'])
    if not ok:
        return {'ok': False,
                'detail': 'the weights are there but torch + transformers do not '
                          'import in the detector environment'}
    return {'ok': True, 'detail': 'torch + transformers OK, weights on disk'}


def watermark_detect_gpu_available() -> bool:
    """True only when the detector's interpreter can actually run torch on CUDA.

    Same reasoning as bank_scoring_gpu_available, and the same UNKNOWN handling:
    the parent uses this to decide whether to take the GPU-exclusive window (which
    unloads ComfyUI and blocks a training start for the whole pass), so an
    unanswered probe on a machine that HAS a card resolves to "assume the card is
    in play" — leaving the GPU unprotected is the expensive mistake, not holding a
    window one extra time."""
    python = (cfg.get('watermark_detect.python') or cfg.get('bank_scoring.python')
              or sys.executable)
    if (cfg.get('watermark_detect.device') or 'auto').lower() == 'cpu':
        return False
    state = _cached_import_state(
        'watermark_detect_gpu', python,
        'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)')
    if state is None:
        return gpu_vram_gb() is not None
    return state


def _watermark_detect_threshold() -> float:
    from .services import watermark_detector
    return watermark_detector.threshold()


def watermark_detect_weights_present() -> bool:
    """True when BOTH model repos are cached under the detector's models_root.
    A cheap directory check — huggingface_hub names its cache folders
    ``models--<owner>--<name>``, and a snapshot folder that exists but is empty is
    a half-finished download, which counts as absent."""
    from .services import watermark_detector
    root = watermark_detector.models_root()
    if not root:
        return False
    for repo in watermark_detector.MODEL_REPOS:
        folder = os.path.join(root, 'models--' + repo.replace('/', '--'), 'snapshots')
        try:
            snaps = [d for d in os.listdir(folder)
                     if os.listdir(os.path.join(folder, d))]
        except OSError:
            return False
        if not snaps:
            return False
    return True


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
    interpreter is the one that must see the packages (or, for gallery_dl /
    yt_dlp, the one `python -m` re-launches as a subprocess — same interpreter,
    same site-packages). curl_cffi + gallery_dl are the two hard requirements
    (picazor/civitai fetch, gallery enumeration); bs4/cloudscraper/instaloader/
    ddgs/yt_dlp ride along in the same install. Every module the scrape stack
    imports (directly or via `python -m`) belongs here: an omission reads as
    "installed" while the source that needs it still raises at runtime
    (instaloader did, until 2026-07; ddgs and yt_dlp did too, until this fix)."""
    import importlib.util
    missing = [m for m in ('curl_cffi', 'gallery_dl', 'bs4', 'cloudscraper', 'instaloader',
                            'ddgs', 'yt_dlp')
               if importlib.util.find_spec(m) is None]
    return {'ok': not missing,
            'detail': 'scrape deps OK' if not missing else f"missing: {', '.join(missing)}"}


def _scan_models() -> dict:
    """Model files per family — by the SAME listers the generate path resolves
    against, so probe == picker == resolver by construction.

    This function used to be the FIFTH scanner: its own one-level `iterdir`
    walk with its own family rules, while every resolver had moved to the
    recursive, extra_model_paths-aware scans in `comfy_model_paths`. The
    divergence was measurable three ways: a model filed two folders deep was
    loadable and generated with, but ✗ here; a Krea file at the root of a
    search folder was accepted by `get_krea_models`' root rule and invisible
    here; and the SDXL bucket read one flat level of `checkpoints/` when
    ComfyUI's CheckpointLoader lists that tree recursively. Names are now the
    RELATIVE paths the loaders actually take, exactly as the resolvers return
    them.

    Kept as a function (rather than inlining the four calls at the probe site)
    because tests and the probe payload both address the four-bucket dict.
    """
    from .services import comfy_model_paths, klein_edit_helper
    from .utils import comfyui as comfy
    klein = [os.path.join(prefix, f) if prefix else f
             for prefix, files in klein_edit_helper._klein_unet_folders()
             for f in files]
    return {
        'zimage': comfy.get_zimage_models(),
        # `('',)` matches every relative directory: an SDXL checkpoint has no
        # family token — sitting anywhere under a `checkpoints` root IS the
        # claim, and ComfyUI lists that tree recursively.
        'sdxl': comfy_model_paths.scan_family_tree(
            comfy_model_paths.search_roots('checkpoints'), ('',),
            root_file_accept=lambda _f: True),
        'krea': comfy.get_krea_models(),
        'klein': sorted(set(klein)),
    }


# --- Auto-detection (Setup wizard) -----------------------------------------
# Discover already-installed tools so the wizard can fill config itself. Two
# signals: a REACHABLE default port (safe to auto-apply — it answered) and a
# folder found on disk (a guess → the UI confirms before writing it).
_OLLAMA_DEFAULT_URL = 'http://127.0.0.1:11434'
_COMFYUI_DEFAULT_URL = 'http://127.0.0.1:8188'
_SETUP_READINESS_TIMEOUT = 1.0
_DOCKER_OLLAMA_URLS = {
    'host': 'http://host.docker.internal:11434',
    'docker': 'http://ollama:11434',
}


def _validated_setup_http_base(value) -> str:
    """Return a strict HTTP(S) origin for a lightweight readiness probe.

    Launcher-provided endpoints are trusted configuration, but still validate
    their shape before handing them to requests: no credentials, path, query or
    fragment, and no malformed port. An invalid value is a configuration error,
    never a reason to fall back to a stale app setting.
    """
    if not isinstance(value, str) or not value.strip():
        return ''
    raw = value.strip().rstrip('/')
    try:
        parsed = urlsplit(raw)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            return ''
        _ = parsed.port  # validate a numeric/in-range port without resolving the host
    except (TypeError, ValueError):
        return ''
    return raw


def setup_comfyui_mode() -> str:
    """Classify who owns ComfyUI for Setup UI/actions (path-free enum only)."""
    runtime = (os.environ.get('LDS_RUNTIME') or '').strip().lower()
    docker_mode = (os.environ.get('LDS_DOCKER_COMFY_MODE') or '').strip().lower()
    if runtime == 'docker-gpu':
        return 'integrated'
    if runtime == 'docker-external-comfy' and docker_mode == 'external':
        return 'external-host'
    return 'external'


def setup_is_docker_runtime() -> bool:
    """Whether Setup is running inside one of LDS's Docker deployments."""
    return (os.environ.get('LDS_RUNTIME') or '').strip().lower().startswith('docker')


def setup_ollama_deployment_url(mode: str) -> str:
    """The fixed in-container endpoint for a persisted Docker deployment mode."""
    return _DOCKER_OLLAMA_URLS.get(mode, '')


def _setup_ollama_base_url(docker_runtime: bool, mode: str) -> str:
    """Resolve the authoritative probe endpoint without exposing it in readiness."""
    if mode == 'none':
        return ''
    if docker_runtime:
        # The in-app deployment choice is authoritative. Never fall back to an
        # environment variable or an old native 127.0.0.1 setting: either could
        # silently probe a different service than the one Setup says is selected.
        raw = setup_ollama_deployment_url(mode)
    else:
        raw = cfg.get('ollama.url') or _OLLAMA_DEFAULT_URL
    return _validated_setup_http_base(raw)


def setup_runtime_readiness() -> dict:
    """Return the small, paste-safe boot snapshot used by Setup polling.

    Only services owned by the selected deployment are ever reported as
    ``starting``: integrated ComfyUI in the GPU stack, and the companion
    Ollama container when explicitly requested.  An external/lightweight
    Docker ComfyUI remains manual, while ``host`` Ollama is unreachable rather
    than indefinitely starting and ``none`` is explicitly disabled.

    This deliberately avoids the full capability probe (filesystem walks,
    imports and model inspection).  Probes are bounded to one second and the
    response contains enums/booleans only -- never configured URLs or paths.
    Integrated ComfyUI uses its fixed internal loopback endpoint instead of a
    user-configured URL.
    """
    runtime = (os.environ.get('LDS_RUNTIME') or '').strip().lower()
    docker_runtime = runtime.startswith('docker')
    comfy_mode = setup_comfyui_mode()
    integrated_comfyui = comfy_mode == 'integrated'
    comfy_ready = False
    if integrated_comfyui:
        comfy_ready = _http_ok(
            f'{_COMFYUI_DEFAULT_URL}/history',
            timeout=_SETUP_READINESS_TIMEOUT,
            readiness=True,
        )
    comfy = {
        'mode': comfy_mode,
        'state': (
            'ready' if comfy_ready else 'starting'
        ) if integrated_comfyui else 'manual',
        'ready': comfy_ready,
        'poll': integrated_comfyui and not comfy_ready,
    }

    if docker_runtime:
        configured_mode = cfg.get('ollama.deployment_mode', '')
        configured_mode = (configured_mode.strip().lower()
                           if isinstance(configured_mode, str) else '')
        ollama_mode = (configured_mode if configured_mode in ('none', 'host', 'docker')
                       else 'unconfigured')
    else:
        # Native installs keep their historical URL-based behavior. A deployment
        # choice saved while using Docker must not disable or redirect native Ollama.
        ollama_mode = 'local'

    ollama_ready = False
    ollama_url = _setup_ollama_base_url(docker_runtime, ollama_mode)
    if ollama_mode not in ('none', 'unconfigured') and ollama_url:
        ollama_ready = _http_ok(
            f'{ollama_url}/api/tags',
            timeout=_SETUP_READINESS_TIMEOUT,
            readiness=True,
        )
    ollama_state = (
        'unconfigured' if ollama_mode == 'unconfigured'
        else 'disabled' if ollama_mode == 'none'
        else 'misconfigured' if not ollama_url
        else 'ready' if ollama_ready
        else 'starting' if ollama_mode == 'docker'
        else 'unreachable'
    )
    ollama = {
        'mode': ollama_mode,
        'state': ollama_state,
        'ready': ollama_ready,
        'poll': ollama_mode == 'docker' and bool(ollama_url) and not ollama_ready,
    }
    return {'comfyui': comfy, 'ollama': ollama}


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


# --- ai-toolkit torch probe (what actually trains) -----------------------------
# ai-toolkit runs in ITS OWN venv, which LDS never installs — it only reads the
# interpreter the user pointed at. Whether that venv's torch carries kernels for
# the local GPU is invisible from here, and getting it wrong is silent: RTX 50
# (Blackwell, sm_120) + a stable wheel = `is_available()` True, then a hard
# "no kernel image is available for execution on the device" at the first real
# computation. So: probe the venv, but only when it can matter.
#
# COST DISCIPLINE. `import torch` in a cold venv costs seconds, so the expensive
# probe is gated behind a ~100 ms nvidia-smi capability read: a GPU below
# compute 10.0 (everything up to Ada / RTX 40) can never hit the trap and pays
# nothing at all. What we do run is cached 10 min — a venv does not change
# between two runs.
_cc_cache = {'ts': 0.0, 'cc': None}
_TORCH_PROBE_TTL = 600
_torch_probe_cache = {}   # interpreter path -> (ts, info)

# First capability major that stable wheels may not cover (Blackwell = 12).
# 10 is deliberately lower than 12: it keeps the gate honest if a future
# generation lands before the wheels do.
_RISKY_CC_MAJOR = 10

_TORCH_PROBE_CODE = (
    'import json, torch\n'
    'cap = name = None\n'
    'try:\n'
    '    if torch.cuda.is_available():\n'
    '        cap = list(torch.cuda.get_device_capability(0))\n'
    '        name = torch.cuda.get_device_name(0)\n'
    'except Exception:\n'
    '    pass\n'
    'print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,\n'
    '                  "capability": cap, "device_name": name,\n'
    '                  "arch_list": list(torch.cuda.get_arch_list())}))\n'
)


def gpu_compute_capability():
    """(major, minor) of GPU 0 via nvidia-smi, cached 10 min. None = unknown."""
    now = time.time()
    if _cc_cache['ts'] and (now - _cc_cache['ts']) < _GPU_TTL:
        return _cc_cache['cc']
    cc = None
    try:
        proc = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if proc.returncode == 0:
            lines = (proc.stdout or '').strip().splitlines()
            if lines:
                major, _, minor = lines[0].strip().partition('.')
                cc = (int(major), int(minor or 0))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        cc = None
    _cc_cache.update(ts=now, cc=cc)
    return cc


def _torch_probe(python: str, timeout=90):
    """Raw torch facts from `python`, as a dict, or None. None is UNKNOWN — torch
    not importable, interpreter broken, cold-import timeout — never a claim."""
    try:
        proc = subprocess.run([python, '-c', _TORCH_PROBE_CODE],
                              capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        info = json.loads(((proc.stdout or '').strip().splitlines() or [''])[-1])
    except Exception:
        return None
    return info if isinstance(info, dict) else None


def aitoolkit_torch_info():
    """torch/GPU facts from the ai-toolkit venv — the interpreter that TRAINS —
    or None when we cannot know (no ai-toolkit, no NVIDIA GPU, GPU old enough to
    be covered by every wheel, torch not importable, probe timeout). Callers must
    treat None as 'no information', never as a verdict."""
    cc = gpu_compute_capability()
    if cc is None or cc[0] < _RISKY_CC_MAJOR:
        return None                       # cheap exit: no torch import at all
    python = cfg.aitoolkit_path('venv_python')
    if not python or not Path(python).is_file():
        return None
    key = str(python)
    now = time.time()
    hit = _torch_probe_cache.get(key)
    if hit and (now - hit[0]) < _TORCH_PROBE_TTL:
        return hit[1]
    info = _torch_probe(key)
    if info is None:
        return None      # a cold-import timeout must not be cached as a fact
    _torch_probe_cache[key] = (now, info)
    return info


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
    the folder itself otherwise). Never raises — a filesystem hiccup degrades
    to 'not_comfyui' rather than throwing into the request.

    Every verdict also carries `input_check` = {path, ok, problem}: "this IS a
    ComfyUI install" was only ever half the question, and the wizard used to
    certify the half it could see. The other half is whether the app can actually
    HAND FILES to that install — every local engine copies its source into
    `input/`. When ComfyUI runs in another container that folder is not shared by
    default, so the wizard went green and the first generation died on a bare 500
    (reported on Discord by nofaceman). `ok` is None when there is nothing to probe
    (no valid folder yet); a False NEVER blocks the wizard — someone may well
    configure the app before mounting their volumes."""
    raw = (path or '').strip()
    if not raw:
        return {'status': 'empty', 'resolved': '', 'suggestion': '',
                'input_check': _input_check('')}
    p = Path(raw)
    if _is_comfyui_dir(p):
        return {'status': 'valid', 'resolved': str(p), 'suggestion': '',
                'input_check': _input_check(str(p))}
    child = p / 'ComfyUI'
    if _is_comfyui_dir(child):
        return {'status': 'nested', 'resolved': str(child), 'suggestion': str(child),
                'input_check': _input_check(str(child))}
    try:
        exists, is_dir = p.exists(), p.is_dir()
    except OSError:
        exists, is_dir = False, False
    if not exists:
        return {'status': 'missing', 'resolved': str(p), 'suggestion': '',
                'input_check': _input_check('')}
    if is_dir:
        try:
            is_empty = not any(p.iterdir())
        except OSError:
            is_empty = False
        if is_empty:
            return {'status': 'empty_dir', 'resolved': str(p), 'suggestion': '',
                    'input_check': _input_check('')}
    # A file at that path, or a non-empty folder that simply isn't a ComfyUI checkout.
    return {'status': 'not_comfyui', 'resolved': str(p), 'suggestion': '',
            'input_check': _input_check('')}


def _input_check(base_dir: str) -> dict:
    """Can this process actually put a file in the input/ folder of the ComfyUI at
    `base_dir`? Honours a SAVED `comfyui.input_dir` override, so the wizard judges
    the folder the app would really use rather than a layout assumption.
    {'path','ok','problem'}; ok=None = nothing probed. Never raises."""
    if not base_dir:
        return {'path': '', 'ok': None, 'problem': ''}
    try:
        override = cfg.get('comfyui.input_dir') or ''
    except Exception:
        override = ''
    try:
        target = cfg.resolve_comfyui_dir('input', base_dir, override)
        path = str(target) if target else ''
        verdict = comfy_fs.probe_folder('input', path)
    except Exception:
        return {'path': '', 'ok': None, 'problem': ''}
    return {'path': comfy_fs.safe_path(path), 'ok': verdict['ok'],
            'problem': verdict['problem']}


def classify_comfyui_folders(base_dir: str, overrides: dict | None = None) -> dict:
    """Resolve the four ComfyUI working folders for a candidate (possibly unsaved)
    ComfyUI section, and say for each one where the path came from and whether it is
    actually there. Feeds the Settings preview so an override is never a leap of
    faith: an empty field SHOWS the derived path it falls back to, and a typed path
    that does not exist says so instead of failing silently at generation time.

    Returns {<config key>: {kind, source, resolved, exists, usable, problem}} where
      source ∈ 'override' (the field is filled) | 'derived' (from the install dir)
               | 'unset'  (no install dir and no override — nothing to resolve).

    `exists` alone certifies half the contract: a folder can be there and still be
    unusable from THIS process — the case that costs the most (ComfyUI in another
    container, an input/ mounted read-only) because the URL test goes green and the
    first generation dies on a copy. `usable` is the other half: for the folders the
    app WRITES into it is a real write-then-delete probe, for the others a read
    check. True/False, or None when there is nothing to probe (no path, or the path
    is not on disk — `exists` already says that). `problem` is the sentence to show.
    Read-only apart from the probe file it removes again, never raises."""
    overrides = overrides or {}
    out = {}
    for kind in cfg.COMFY_DIR_KINDS:
        key, _sub = cfg._COMFY_DERIVED[kind]
        explicit = str(overrides.get(key) or '').strip()
        p = cfg.resolve_comfyui_dir(kind, base_dir, explicit)
        resolved = str(p) if p else ''
        source = 'override' if explicit else ('derived' if resolved else 'unset')
        if kind == 'loras' and not explicit:
            # Deploys follow extra_model_paths.yaml (GitHub #25), so this preview
            # has to as well — showing <base>/models/loras while LoRAs land in a
            # yaml root would rebuild the very divergence that bug WAS. Only for
            # the SAVED install: the yaml is read next to the live base_dir, so a
            # not-yet-saved base_dir would be described with another tree's yaml.
            yaml_root = _yaml_loras_root(base_dir)
            if yaml_root and yaml_root != resolved:
                resolved, source = yaml_root, 'extra_paths'
        try:
            exists = bool(resolved) and Path(resolved).is_dir()
        except OSError:
            exists = False
        usable, problem = None, ''
        if exists:
            verdict = comfy_fs.probe_folder(kind, resolved)
            usable, problem = verdict['ok'], verdict['problem']
        out[key] = {'kind': kind, 'source': source,
                    'resolved': resolved, 'exists': exists,
                    'usable': usable, 'problem': problem}
    return out


def _yaml_loras_root(base_dir: str) -> str:
    """The loras root ``extra_model_paths.yaml`` puts first, or '' when there is no
    yaml, no saved install, or the previewed base_dir is not the saved one (the
    yaml lives next to the SAVED base_dir — describing another tree with it would
    be a lie). Never raises."""
    try:
        saved = (cfg.get('comfyui.base_dir') or '').strip()
        if not saved or os.path.normcase(os.path.normpath(saved)) != \
                os.path.normcase(os.path.normpath((base_dir or '').strip() or '.')):
            return ''
        from .services import comfy_model_paths
        return comfy_model_paths.write_root('loras') or ''
    except Exception:
        return ''


# ComfyUI takes its custom folders on the COMMAND LINE only (--input-directory,
# --output-directory, --models-directory); there is no config file to read. It does,
# however, echo its own argv back in /system_stats.system.argv, so the running
# instance can be ASKED what it was started with instead of guessing a layout.
_COMFY_ARGV_FLAGS = {'--output-directory': 'output_dir', '--input-directory': 'input_dir',
                     '--models-directory': 'models_dir'}


def parse_comfy_argv_dirs(argv) -> dict:
    """Extract the folder overrides ComfyUI was launched with from its own argv.

    Both argparse spellings are accepted (`--input-directory X` and
    `--input-directory=X`). RELATIVE paths are deliberately DROPPED: they resolve
    against ComfyUI's working directory, which we do not know, and this app never
    guesses a path by convention. `--base-directory` is likewise not turned into
    input/output suggestions — the install-directory field already derives those, and
    inventing them here would be a layout assumption, not an answer. Never raises."""
    out = {}
    if not isinstance(argv, (list, tuple)):
        return out
    items = [str(a) for a in argv]
    for i, tok in enumerate(items):
        flag, _, inline = tok.partition('=')
        key = _COMFY_ARGV_FLAGS.get(flag)
        if not key:
            continue
        value = inline if inline else (items[i + 1] if i + 1 < len(items) else '')
        value = value.strip().strip('"')
        # A following token that is itself a flag means the value was missing.
        if not value or (not inline and value.startswith('-')):
            continue
        try:
            if not os.path.isabs(value):
                continue
        except (OSError, ValueError):
            continue
        out[key] = os.path.normpath(value)
    return out


def detect_comfyui_folders(timeout=3) -> dict:
    """Ask the RUNNING ComfyUI which custom folders it was started with.
    NETWORK — one short GET, kept out of probe() like comfyui_runtime_info.

    Returns {} when ComfyUI is not configured, not reachable, too old to echo its
    argv (the field landed in 2025 releases), or simply started with no custom
    folder flags. An empty dict means "nothing to offer", never "use the defaults" —
    the caller leaves the manual field alone. Never raises."""
    api = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api:
        return {}
    try:
        r = requests.get(f'{api}/system_stats', timeout=timeout)
        if r.status_code != 200:
            return {}
        return parse_comfy_argv_dirs(((r.json() or {}).get('system') or {}).get('argv'))
    except Exception:
        return {}


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


def _probe_klein(comfy):
    """The Klein engine's readiness, moved verbatim (2026-08-23): missing
    assets, present-but-invalid weights, unsupported widget enums, and the
    single ready verdict klein_edit_helper owns. Returns the helper module
    too — the payload section calls its override/pin-gap reporters
    directly."""
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
    # Capability gap on the graph's WIDGET VALUES, not its nodes. The shipped Klein
    # workflow used to pin `scheduler: "beta57"`, a value the third-party RES4LYF
    # pack injects into ComfyUI's CORE list — so a stock install passed every asset
    # AND node check here, went green, and then refused the very first generation
    # with a raw ComfyUI 400 (reported by IndependentProcess0 on Reddit). That
    # value is gone, but the blind spot it exposed is not: this is the net for the
    # next one, and for a workflow file a user has edited themselves. Nothing is
    # substituted (a scheduler changes the render), so the gap has to be visible
    # BEFORE a batch is launched, which means here. /object_info is cached, and
    # this fails OPEN — an unreachable ComfyUI reports no gap, `reachable` already
    # says that.
    klein_unsupported_enums = _keh.klein_unsupported_enums() if comfy['ok'] else []
    # The verdict itself lives in klein_edit_helper so the watermark cleaner reads
    # the SAME four conditions instead of its own laxer copy; the ingredients are
    # already computed here for the payload, so they are handed over rather than
    # re-probed.
    klein_ready = _keh.klein_engine_ready(
        comfy['ok'], missing=klein_missing, invalid=klein_invalid,
        unsupported_enums=klein_unsupported_enums)
    return (_keh, klein_missing, klein_invalid, klein_unsupported_enums,
            klein_ready)


def _probe_krea(comfy):
    """The Krea 2 Identity Edit engine's readiness, moved verbatim: asset
    and node-pack gaps kept apart, pack-on-disk detection, invalid-weights
    verdicts, pinned-but-absent files, the resolved base name, and the
    four-part ready verdict."""
    # Krea 2 Identity Edit — the second LOCAL engine. Readiness is honest and
    # four-part (base model + identity LoRA + text encoder + VAE) AND depends on
    # a custom-node pack, unlike Klein whose graph is core-nodes-only. Both gaps
    # are published separately so the engine card can name the RIGHT one: "install
    # the node pack" and "place the LoRA here" are different actions.
    # Disk scan = cheap listdir, network-free. The node probe is /object_info,
    # cached and fail-OPEN (unreachable ComfyUI reports no missing node here —
    # `reachable` already says that, and two red flags for one cause is noise).
    from .services import krea_edit_helper as _krh
    krea_missing = _krh.krea_missing_assets()
    krea_nodes_missing = _krh.krea_missing_nodes() if comfy['ok'] else []
    # Pack ON DISK but not yet exposed by /object_info = "restart ComfyUI", NOT
    # "install the pack". Now that the app installs the pack itself, telling
    # someone to install what they just watched install would be the whole
    # feature failing at the last inch.
    krea_nodes_installed = _krh.krea_node_pack_installed()
    # Present-but-INVALID, exactly like Klein's: the Krea base sits behind a HF
    # licence gate and the identity LoRA behind a Civitai login, so a browser
    # download without the licence/login saves the HTML gate PAGE as
    # .safetensors — present, not loadable, and today the only symptom was a raw
    # ComfyUI "Expecting value: line 1 column 1". Every Krea asset is required
    # (KREA_REQUIRED == all of them), so any blocking-invalid one keeps the
    # engine dark; the advisory too_small does not gate.
    krea_invalid = _krh.krea_invalid_assets()
    krea_blocking_invalid = any(i['blocking'] for i in krea_invalid)
    # A model file PINNED in Settings that is not on disk. Kept apart from
    # krea_missing because it is a different sentence and a different fix: the
    # file the user chose is absent, so nothing should be elected in its place.
    # See krea_edit_helper.KreaPinnedModelMissing for the run that made this a
    # gate instead of a log line.
    krea_pin_gaps = _krh.krea_pin_gaps()
    # WHICH Krea base this install actually loads, named. `krea.base_model` blank
    # means "elect one", and until now nothing on screen said what got elected —
    # the only way to find out was to read a finished PNG's metadata. That silence
    # is expensive: a folder holding both the canonical Turbo build and a
    # community finetune whose name also reads as "turbo" puts BOTH in the top
    # regime tier, and the tie-break can hand the run to the finetune. Every
    # quality judgement made after that is about a model the user never chose.
    # It is the SAME resolve_krea_unet() the generation path calls, not a second
    # ranking. Cost: this is its THIRD call in this probe (krea_missing_assets and
    # krea_invalid_assets each made one above), and the expensive part — the
    # header tie-break between several same-tier candidates — is cached in
    # model_integrity on (abspath, mtime_ns, size). Measured on a worst-case tree
    # of six candidates that ALL read as "turbo" (so the tie-break really runs):
    # 24 ms for the cold call, 1.4 ms for every later one. This one is always a
    # later one, and probe() is itself cached for 30 s; a single-candidate install
    # reads no header at all (step 3 of resolve_krea_unet only runs when more than
    # one survives step 2). Empty string = nothing loadable on disk.
    krea_base_resolved = _krh.resolve_krea_unet() or ''
    krea_ready = (comfy['ok'] and not krea_missing and not krea_nodes_missing
                  and not krea_blocking_invalid and not krea_pin_gaps)
    return (krea_missing, krea_nodes_missing, krea_nodes_installed,
            krea_invalid, krea_pin_gaps, krea_base_resolved, krea_ready)


def _probe_seedvr2(comfy):
    """The SeedVR2 upscaler's readiness, moved verbatim: weights / node
    pack / loadability gaps, the single ready verdict, and the optional
    tiling lane with its full-frame megapixel ceiling."""
    # SeedVR2 — the fidelity upscaler (issue #32). Same three-part shape as Krea
    # (weights on disk / node pack present / weights actually loadable), because
    # it has the same three ways to be half-installed. It is NOT a generation
    # engine: nothing in the dataset catalog can be produced by it, so it is
    # published as an upscaler capability and the engine picker that offers it is
    # the ✨ improve pass's, not the variation catalog's.
    from .services import seedvr2_helper as _svr
    seedvr2_missing = _svr.seedvr2_missing_assets()
    seedvr2_nodes_missing = _svr.seedvr2_missing_nodes() if comfy['ok'] else []
    seedvr2_nodes_installed = _svr.seedvr2_node_pack_installed()
    seedvr2_invalid = _svr.seedvr2_invalid_assets()
    seedvr2_ready = _svr.engine_ready(comfy['ok'], missing=seedvr2_missing,
                                      invalid=seedvr2_invalid,
                                      nodes_missing=seedvr2_nodes_missing)
    # The OPTIONAL high-resolution lane (tiling), contributed by SurpassHR
    # (GitHub #32). Its absence is not a fault: without it the default lane
    # still upscales, it is only capped by what this card can hold in one pass.
    # The ceiling is published so the UI can say that BEFORE a run dies — the
    # report behind this feature is someone meeting the limit as a CUDA OOM.
    seedvr2_tiling_nodes_missing = _svr.ttp_missing_nodes() if comfy['ok'] else []
    seedvr2_tiling_ready = _svr.tiling_available(comfy['ok'])
    seedvr2_ceiling_mp = _svr.full_frame_ceiling_mp()
    return (seedvr2_missing, seedvr2_nodes_missing, seedvr2_nodes_installed,
            seedvr2_invalid, seedvr2_ready, seedvr2_tiling_ready,
            seedvr2_tiling_nodes_missing, seedvr2_ceiling_mp)


def _comfyui_caps_section(comfy, base_dir, comfy_dir, comfy_launcher,
                          comfy_skipped, models, _keh, klein_missing,
                          klein_unsupported_enums, klein_invalid,
                          krea_missing, krea_base_resolved,
                          krea_nodes_missing, krea_nodes_installed,
                          krea_invalid, krea_pin_gaps, seedvr2_missing,
                          seedvr2_nodes_missing, seedvr2_nodes_installed,
                          seedvr2_invalid, seedvr2_ready,
                          seedvr2_tiling_ready,
                          seedvr2_tiling_nodes_missing, seedvr2_ceiling_mp):
    """The probe payload's ComfyUI card — the largest section of the caps
    dict, moved verbatim: reachability with its honest hint, directory
    validity, launcher support, the model scan, and every per-engine gap
    list the Setup screen turns into a button."""
    from .services import lanpaint_helper as _lph
    from .services import qwen_camera_helper as _qch
    # 📷 Camera angles. Computed here rather than in a probe of its own for the
    # same reason LanPaint is: the lane has no pins, no custom-node pack and no
    # invalid-file class — it is four filenames on disk. `camera_missing` names
    # setup_installer actions, so the Setup screen turns each one into the
    # button that installs it; `camera_ready` is the single verdict every
    # surface reads, so the picker and the Setup card cannot disagree about
    # whether a view can be rendered.
    camera_missing = _qch.camera_missing_assets()
    return {
        'reachable': comfy['ok'],
        # WHY it isn't reachable, when it isn't: 'ok' | 'slow' | 'unreachable'
        # | 'unconfigured'. `reachable` alone made every screen say "ComfyUI
        # isn't running" at a ComfyUI that was running and busy; `hint` is the
        # matching sentence, so the wording lives in ONE place instead of being
        # re-invented per card. See probe_comfyui / comfyui_down_message.
        'status': comfy.get('status', 'ok' if comfy['ok'] else 'unreachable'),
        'hint': comfy.get('hint', ''),
        # Read budget currently granted to the heavy /object_info enumeration,
        # so a screen can quote the number the user would raise.
        'object_info_timeout_s': _object_info_timeout(),
        'api_url': cfg.get('comfyui.api_url') or '',
        'base_dir': base_dir,
        'dir_configured': bool(base_dir),
        'dir_valid': comfy_dir['valid'],       # base_dir really is a ComfyUI install
        'resolved_dir': comfy_dir['resolved'],
        # The start button is stricter than normal ComfyUI model discovery:
        # only the saved NVIDIA portable layout and local standard URL qualify.
        'portable_launcher_supported': comfy_launcher['portable_supported'],
        'portable_launcher_local_api': comfy_launcher['local_api_safe'],
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
        # Widget values the shipped Klein graph needs that THIS ComfyUI doesn't
        # offer: [{node_id, class_type, input, value, pack, url}]. Empty on a
        # capable install AND on an unreachable one (fail-open).
        'klein_unsupported_enums': klein_unsupported_enums,
        # Krea 2 Edit gaps, kept apart from Klein's: asset KEYS not on disk
        # (krea_edit_helper.KREA_ASSETS) and the custom-node class_types this
        # ComfyUI doesn't expose. Empty + empty => the engine is ready.
        'krea_missing': krea_missing,
        # The ComfyUI-relative name of the Krea base the next run WILL load
        # (pin honoured, else the election). '' = none on disk. Published so
        # the Settings field can name it instead of promising "auto".
        'krea_base_resolved': krea_base_resolved,
        'krea_nodes_missing': krea_nodes_missing,
        'krea_nodes_installed': krea_nodes_installed,
        # Krea assets PRESENT on disk but not real, loadable weights — same
        # [{asset, filename, verdict, blocking, reason}] shape as
        # klein_invalid, so one banner covers both engines.
        'krea_invalid': krea_invalid,
        # SeedVR2 gaps, kept apart from the generation engines' for the same
        # reason theirs are kept apart from each other: "download the
        # weights" and "install the node pack in ComfyUI" are different
        # actions with different buttons.
        'seedvr2_missing': seedvr2_missing,
        'seedvr2_nodes_missing': seedvr2_nodes_missing,
        'seedvr2_nodes_installed': seedvr2_nodes_installed,
        'seedvr2_invalid': seedvr2_invalid,
        # The single verdict every SeedVR2 surface reads (Settings card,
        # Setup step, the improve engine picker) so none of them re-derives
        # readiness from a different subset of the four gaps above.
        'seedvr2_ready': seedvr2_ready,
        # Optional tiled lane: ready / which TTP classes are absent / the
        # full-frame megapixel ceiling this GPU is good for (None = unknown
        # card, and then the UI says nothing rather than inventing a number).
        'seedvr2_tiling_ready': seedvr2_tiling_ready,
        'seedvr2_tiling_nodes_missing': seedvr2_tiling_nodes_missing,
        'seedvr2_ceiling_mp': seedvr2_ceiling_mp,
        # LanPaint — the masked Repair lane's sampler (services/lanpaint_helper).
        # Same two-part shape as the other node packs, computed here rather
        # than in a probe of its own because it has no assets, pins or invalid
        # files: which classes /object_info lacks (fail-open when ComfyUI is
        # down — `reachable` already says that), and whether the pack is on
        # disk, which turns "install the pack" into "restart ComfyUI".
        'lanpaint_nodes_missing': (_lph.lanpaint_missing_nodes()
                                   if comfy['ok'] else []),
        'lanpaint_nodes_installed': _lph.lanpaint_node_pack_installed(),
        # setup_installer action names for the camera-angle assets NOT on disk.
        # One of them is `krea_vae` on purpose — the Qwen VAE ships with the
        # Krea 2 lane and this lane points at that button rather than declaring
        # a second download of the same file.
        'camera_missing': camera_missing,
        # The speed LoRA missing does NOT make the lane un-ready: it renders at
        # 20 steps instead of 4. Only the four REQUIRED assets gate it.
        'camera_ready': not any(a in camera_missing
                                for a in _qch.CAMERA_REQUIRED),
        # Klein assets PRESENT on disk but not real, loadable weights:
        # [{asset, filename, verdict, blocking, reason}]. Distinct from
        # klein_missing (the file exists, it just can't load) — drives the Setup
        # "present but INVALID: <asset> (<reason>)" line and the diagnostic, and
        # a blocking-invalid required asset also keeps engines.klein dark above.
        'klein_invalid': klein_invalid,
        # User-pinned Klein model files (Settings ▸ Image engine), only the
        # slots that are SET: {slot: {configured, found}}. `found` False means
        # the pin fell back to auto-detection — drives the honest "not found"
        # badge next to the Settings field, so a typo is never silent.
        # Ported from socrasteeze's branch (GitHub #20).
        'klein_overrides': _keh.klein_override_status(),
        # Pinned-but-absent model files, per engine:
        # [{slot, key, configured[, status]}]. Non-empty keeps that engine
        # dark above, and the engine card reads THIS to say which file the
        # user chose is missing — never "download the base model", which
        # would send them to fix something that is already there.
        'klein_pin_gaps': _keh.klein_pin_gaps(),
        'krea_pin_gaps': krea_pin_gaps,
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
    openrouter_ = probe_openrouter()
    face_scoring = probe_face_scoring()
    masks = probe_masks()
    bank_scoring = probe_bank_scoring()
    bank_siglip2 = probe_bank_siglip2()
    watermark_inpaint = probe_watermark_inpaint()
    watermark_detect = probe_watermark_detect()
    video = probe_video()
    video_text = probe_video_text()
    scrape_deps = probe_scrape_deps()
    joycaption = probe_joycaption(aitoolkit)
    models = _scan_models()
    (_keh, klein_missing, klein_invalid, klein_unsupported_enums,
     klein_ready) = _probe_klein(comfy)
    (krea_missing, krea_nodes_missing, krea_nodes_installed, krea_invalid,
     krea_pin_gaps, krea_base_resolved, krea_ready) = _probe_krea(comfy)
    (seedvr2_missing, seedvr2_nodes_missing, seedvr2_nodes_installed,
     seedvr2_invalid, seedvr2_ready, seedvr2_tiling_ready,
     seedvr2_tiling_nodes_missing,
     seedvr2_ceiling_mp) = _probe_seedvr2(comfy)
    base_dir = cfg.get('comfyui.base_dir') or ''
    from .services import comfyui_control
    comfy_launcher = comfyui_control.launcher_status()
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

    from .services.face_dataset_service import MAX_FANOUT as _max_fanout

    caps = {
        'configured': cfg.is_configured(),
        # Images one generation batch may queue at once. Published so the
        # workspace can say "75 is over the limit" BEFORE the click instead of
        # letting a multi-engine run be refused after the fact — the server
        # stays the authority, the UI just mirrors the number it is told.
        'max_fanout': _max_fanout,
        'engines': {
            'nanobanana': gemini['ok'],
            'chatgpt': openai_['ok'],
            'openrouter': openrouter_['ok'],
            'klein': klein_ready,
            'krea': krea_ready,
        },
        'chatgpt_subscription': {
            'connected': sub_status['connected'],
            'email': sub_status['email'],
            'plan': sub_status['plan'],
            'codex_cli_detected': chatgpt_oauth.codex_auth_path().is_file(),
        },
        'comfyui': _comfyui_caps_section(
            comfy, base_dir, comfy_dir, comfy_launcher, comfy_skipped,
            models, _keh, klein_missing, klein_unsupported_enums,
            klein_invalid, krea_missing, krea_base_resolved,
            krea_nodes_missing, krea_nodes_installed, krea_invalid,
            krea_pin_gaps, seedvr2_missing, seedvr2_nodes_missing,
            seedvr2_nodes_installed, seedvr2_invalid, seedvr2_ready,
            seedvr2_tiling_ready, seedvr2_tiling_nodes_missing,
            seedvr2_ceiling_mp),
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
            # Kept apart so Setup can report what it ACTUALLY found: "this isn't
            # an ai-toolkit checkout" and "this checkout has no interpreter we
            # can see" are different problems with different fixes.
            'dir_valid': bool(aitoolkit.get('has_run')),
            'python_candidates': list(aitoolkit.get('python_candidates') or []),
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
        # Optional, user-selected semantic alternative. It is deliberately not
        # folded into bank_scoring: CLIP aesthetic scoring remains usable without
        # the additional 1.5 GB checkpoint.
        'bank_siglip2': bank_siglip2['ok'],
        'bank_siglip2_detail': bank_siglip2['detail'],
        'bank_siglip2_model': bank_siglip2['model'],
        # Lets the front adapt the watermark Clean tooltip: when False, Clean is
        # crop-only (LaMa-routed watermarks are skipped with an install hint).
        'watermark_inpaint': watermark_inpaint['ok'],
        # The dedicated detector extra. True → 🚩 Find runs the classifier instead
        # of the vision model (roughly ten times faster, and it does not need
        # Ollama at all). False changes NOTHING: the vision model still does the
        # work, so this only ever unlocks a faster route, never blocks the old one.
        'watermark_detect': watermark_detect['ok'],
        'watermark_detect_detail': watermark_detect['detail'],
        # Will the detector actually reach CUDA in ITS interpreter? The Setup
        # install pins the app-managed CPU-torch venv, so on a machine with a
        # card this is routinely False until the user picks a GPU Python — and
        # nothing used to say so. Drives the 🚩 panel's CPU note + picker
        # opener; only asked once the extra is installed (the probe imports
        # torch in a subprocess, and a missing extra already answers it).
        'watermark_detect_gpu': bool(watermark_detect['ok']
                                     and watermark_detect_gpu_available()),
        # The measured flag threshold, published so the panel and the Settings
        # field quote the SAME number the pass will actually use.
        'watermark_detect_threshold': _watermark_detect_threshold(),
        # The video lane, reported as its three independent pieces. A single
        # boolean would be a lie here: decoding, shot detection and encoding come
        # from three different installs and fail apart. The front uses the parts to
        # say WHICH one to fix — never "video unavailable", which is how a user
        # reinstalls the wrong thing.
        'video': video['ok'],
        'video_detail': video['detail'],
        'video_decode': video['decode'],
        'video_detect': video['detect'],
        'video_encode': video['encode'],
        # 🔳 The safe-zone pass's OCR half, published on its own because it is
        # the only capability in this app whose absence downgrades a pass instead
        # of blocking it: the bands are still measured. The workspace uses it to
        # say "bands only" on the button rather than greying it out.
        'video_text': video_text['ok'],
        'video_text_detail': video_text['detail'],
        # Klein-inpaint (V2, quality) readiness = same as the Klein engine (ComfyUI
        # reachable + Klein models on disk). The custom-node preflight is a clean-time
        # 409. Greys the batch's "Klein (quality)" option when False.
        'watermark_klein': klein_ready,
        # Persisted "allow automatic crop" preference (Settings ▸ Watermark inpainting).
        # The batch Clean bar reads it here to seed/reflect its inline toggle and the
        # review lightbox uses it as the per-image crop-vs-inpaint default; when False,
        # auto-routing repaints border marks instead of cropping them.
        'watermark_allow_crop': bool(cfg.get('watermark.allow_crop')),
        # What an imported photo will actually be STORED as (Settings ▸ Captioning
        # & quality ▸ Dataset import). Published so the import screens can quote
        # the number instead of keeping their own copy of the default — the whole
        # point of the setting is that the rule stops being invisible.
        'dataset_import': _dataset_import_policy(),
        'python': python_ml_status(),
        'scrape_deps': scrape_deps['ok'],
        # WHICH modules are absent, same convention as joycaption/video/siglip2
        # above. The install banner used to recite a hand-written list of three
        # package names; the probe watches seven, so a machine flagged because
        # `ddgs` or `yt_dlp` is missing read a warning that named neither and
        # could not explain why it was being asked to reinstall. The banner now
        # quotes this string's list instead of keeping its own copy.
        'scrape_deps_detail': scrape_deps['detail'],
        'training_visible': aitoolkit['ok'] or bool(cfg.secret('VAST_API_KEY')),
        'studio_visible': comfy['ok'],
    }

    _cache, _cache_ts = caps, now
    return copy.deepcopy(caps)
