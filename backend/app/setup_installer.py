"""Setup installer: run whitelisted, self-contained installs in a background
thread and expose their live state for polling. Actions:

  ml_extras          -> pip install -r backend/requirements-ml.txt (the app's own venv):
                        installs ALL the ML extras at once — kept for a first-time setup
  face_scoring       -> pip install JUST the face-scoring packages (insightface + onnx-
                        runtime, versions read from requirements-ml.txt) into the inter-
                        preter probe_face_scoring resolves — install/repair ONE feature
  masks              -> pip install JUST the person-mask package (rembg) into the inter-
                        preter probe_masks resolves — install/repair ONE feature
  watermark_inpaint  -> pip install JUST the watermark-inpainting package (simple-lama-
                        inpainting, version floor read from requirements-ml.txt) into the
                        interpreter the LaMa wrapper resolves — the scoped install shown
                        next to the Curate 🧽 tools, so a user who already has rembg/
                        insightface doesn't redo the whole ML extras step
  (face_scoring/masks/watermark_inpaint all follow the same shape: ML interpreter resolved
   per capability, requirements-ml.txt pinned as a -c constraint, probe cache invalidated
   on success so the capability flips without a restart.)
  ollama_model       -> stream Ollama's /api/pull for the configured vision model
  klein_model        -> download the Klein fp8 diffusion model into <ComfyUI>/models/unet/klein/
                        (BFL repo is LICENSE-GATED: needs the agreement accepted on HF +
                        an HF_TOKEN secret; a 401 logs the exact recovery steps)
  klein_lora         -> download the consistency LoRA into <ComfyUI>/models/loras/klein/
  klein_text_encoder -> qwen_3_8b_fp8mixed into <ComfyUI>/models/text_encoders/
  klein_vae          -> flux2-vae into <ComfyUI>/models/vae/

No shell, no client-supplied arguments: each action's command/URL/destination is fixed.
"""
import logging
import os
import re
import shutil
import subprocess
import sys
import threading

import requests

from . import capabilities
from . import config as cfg
# Import resolution helpers from the Klein helper – they now respect extra_model_paths.yaml
from .services.klein_edit_helper import (
    resolve_klein_unet,
    resolve_klein_vae,
    resolve_klein_text_encoder,
    _consistency_lora,  # internal but needed for status
    KleinModelsMissing,
)

logger = logging.getLogger(__name__)

# Fixed catalog of the Klein downloads (checked 2026-07-10): the three Comfy-Org/
# dx8152 files are public; the BFL diffusion model is gated (401 without a token).
_KLEIN_DOWNLOADS = {
    'klein_model': {
        'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors',
        'dest': ('unet', 'klein', 'flux-2-klein-9b-fp8.safetensors'),
        'min_free_gb': 15, 'gated': True,
        'license_url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8',
    },
    'klein_lora': {
        'url': 'https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency/resolve/main/Flux2-Klein-9B-consistency-V2.safetensors',
        'dest': ('loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors'),
        'min_free_gb': 1, 'gated': False,
    },
    'klein_text_encoder': {
        'url': 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors',
        'dest': ('text_encoders', 'qwen_3_8b_fp8mixed.safetensors'),
        'min_free_gb': 12, 'gated': False,
    },
    'klein_vae': {
        'url': 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors',
        'dest': ('vae', 'flux2-vae.safetensors'),
        'min_free_gb': 2, 'gated': False,
    },
}

INSTALL_ACTIONS = ('ml_extras', 'scrape_extras', 'ollama_model',
                   'face_scoring', 'masks', 'watermark_inpaint') + tuple(_KLEIN_DOWNLOADS)

_ML_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-ml.txt'
_SCRAPE_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-scrape.txt'
# pip -r installers share one worker; both target THIS interpreter (the scrape
# stack runs in-process, so any other environment would be invisible to the app).
_PIP_REQUIREMENTS = {'ml_extras': _ML_REQUIREMENTS, 'scrape_extras': _SCRAPE_REQUIREMENTS}
# The single package the watermark-inpaint scoped install adds. The NAME lives
# here (an identifier), but the VERSION SPEC is parsed from requirements-ml.txt
# so there's exactly one place a version floor is ever written.
_WATERMARK_PKG = 'simple-lama-inpainting'

# --- ML extras, split per capability -------------------------------------------
# requirements-ml.txt is a FLAT pip file (not grouped by feature), so the
# package->capability grouping lives HERE. The VERSIONS are never duplicated: each
# package's exact requirement line is read from requirements-ml.txt via
# _requirement_spec(), and that same file rides along as a `-c` constraint so a
# scoped install can't bump numpy past insightface's <2 ABI ceiling. A dedicated
# test (test_no_orphan_ml_package) asserts EVERY line in requirements-ml.txt is
# owned by at least one capability below — a package added to the file but
# forgotten here would silently never be installed by any scoped action.
#
#   face_scoring  insightface (face embeddings) + onnxruntime (its runtime). numpy
#                 is pinned <2 *for insightface's* ABI; opencv-python-headless is
#                 the server-safe cv2 that insightface & rembg both pull — listed so
#                 the scoped install prefers the headless variant, matching the
#                 monolithic `-r` install.
#   masks         rembg (u2net background removal), + the same shared numpy /
#                 headless-opencv floor.
#   watermark_inpaint  simple-lama-inpainting (has its own dedicated worker below;
#                 listed here only so the anti-orphan test sees its package covered).
_CAPABILITY_PACKAGES = {
    'face_scoring': ('insightface', 'onnxruntime', 'numpy', 'opencv-python-headless'),
    'masks': ('rembg', 'numpy', 'opencv-python-headless'),
    'watermark_inpaint': (_WATERMARK_PKG,),
}
# The capabilities served by the GENERIC per-capability pip worker
# (_run_ml_capability). watermark_inpaint keeps its own worker, so it's excluded.
_CAPABILITY_ML_ACTIONS = ('face_scoring', 'masks')

# Actions whose success makes a NEW importable package appear -> the probe
# import-cache must be dropped so the capability flips without waiting out the
# 600 s TTL (ml_extras/scrape_extras via -r, the scoped per-capability installs).
_IMPORT_CACHE_ACTIONS = (frozenset(_PIP_REQUIREMENTS)
                         | set(_CAPABILITY_ML_ACTIONS) | {'watermark_inpaint'})
_LOG_MAX = 400  # ring-buffer the log so a chatty pip can't grow unbounded

_lock = threading.Lock()
_runs = {}  # action -> {'state', 'returncode', 'log'}


class AlreadyRunning(Exception):
    pass


class Precondition(Exception):
    pass


def _new_run():
    return {'state': 'running', 'returncode': None, 'log': [], 'progress': None}


def _append(action, line):
    log = _runs[action]['log']
    log.append(line.rstrip('\n'))
    if len(log) > _LOG_MAX:
        del log[:-_LOG_MAX]


def _set_progress(action, done, total):
    """Publish a live byte-progress snapshot for a streaming download, separate
    from the text log (so a smooth % bar never spams the log). `total` may be 0
    when the server sends no content-length -> pct is None (indeterminate)."""
    run = _runs.get(action)
    if run is None:
        return
    run['progress'] = {
        'done': done,
        'total': total,
        'pct': (done * 100 // total) if total else None,
    }


def _quote(p: str) -> str:
    # Quote paths with spaces so the manual command is copy-paste-safe: the
    # portable bundle can be extracted under e.g. C:\Users\...\LoRA Dataset Studio\.
    return f'"{p}"' if ' ' in p else p


def _canon(name: str) -> str:
    """PEP 503 canonical form: -_. all fold to a single dash, case-insensitive."""
    return re.sub(r'[-_.]+', '-', name).lower()


def _requirement_spec(name: str, requirements=_ML_REQUIREMENTS) -> str:
    """The full requirement line for `name` as written in a requirements file
    (e.g. 'simple-lama-inpainting>=0.1.2') — the version floor lives in ONE place
    (requirements-ml.txt), never duplicated in this module. Package-name match is
    canonicalised (PEP 503: -_. all fold together, case-insensitive) and tolerant
    of version/marker/extras suffixes. Falls back to the bare name if the file or
    line is missing (an unpinned `pip install <name>` still works)."""
    canon = _canon(name)
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()   # drop comments / blank lines
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]   # name before any spec/marker
            if _canon(token) == canon:
                return line
    except OSError:
        pass
    return name


def _ml_requirement_names(requirements=_ML_REQUIREMENTS) -> set:
    """Canonical names of every package declared in a requirements file (comments
    and blank lines dropped). Used by the anti-orphan test to prove each ML package
    is mapped to a capability in _CAPABILITY_PACKAGES."""
    names = set()
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]
            names.add(_canon(token))
    except OSError:
        pass
    return names


def _watermark_python() -> str:
    """Interpreter the watermark LaMa wrapper resolves. Reuse the wrapper's OWN
    resolver (watermark.python > masks.python > sys.executable) so the install
    target and the later import can never drift apart."""
    from .services import watermark_lama
    return watermark_lama.lama_python()


def _capability_python(action) -> str:
    """Interpreter a scoped ML install targets — MUST match the resolution its
    matching probe uses, so the install target and the later import can't drift:
      face_scoring -> face_scoring.python  (see capabilities.probe_face_scoring)
      masks        -> masks.python         (see capabilities.probe_masks)
      watermark_inpaint -> the wrapper chain (watermark.python > masks.python)."""
    if action == 'watermark_inpaint':
        return _watermark_python()
    return cfg.get(f'{action}.python') or sys.executable


def manual_command(action) -> str:
    """The exact command that reproduces an install BY HAND, scoped to THIS app's
    own interpreter (sys.executable). A copy-paste then targets the SAME
    environment the app imports from -- the portable bundle's python\\python.exe or
    the dev venv -- instead of whatever bare `pip` happens to be first on PATH
    (which is the whole point of the user's question: a plain `pip install` would
    land in the wrong environment and the extras would never be importable)."""
    if action in _PIP_REQUIREMENTS:
        return f'{_quote(sys.executable)} -m pip install -r {_quote(str(_PIP_REQUIREMENTS[action]))}'
    if action in _CAPABILITY_ML_ACTIONS:
        # One scoped capability (face_scoring | masks): the exact version-pinned
        # lines from requirements-ml.txt, quoted (the '>=' / '<' are shell
        # redirection unquoted), plus that file as a -c constraint. Interpreter =
        # the same one the capability's probe resolves.
        specs = ' '.join(f'"{_requirement_spec(p)}"' for p in _CAPABILITY_PACKAGES[action])
        return (f'{_quote(_capability_python(action))} -m pip install {specs} '
                f'-c {_quote(str(_ML_REQUIREMENTS))}')
    if action == 'watermark_inpaint':
        # Quote the spec: the '>=' in 'simple-lama-inpainting>=0.1.2' is shell
        # redirection unquoted. Interpreter = the wrapper's resolved python.
        return f'{_quote(_watermark_python())} -m pip install "{_requirement_spec(_WATERMARK_PKG)}"'
    if action == 'ollama_model':
        model = (cfg.get('ollama.vision_model') or '').strip() or '<vision-model>'
        return f'ollama pull {model}'
    if action in _KLEIN_DOWNLOADS:
        spec = _KLEIN_DOWNLOADS[action]
        try:
            dest = _klein_dest_path(action)
        except Precondition:
            dest = os.path.join('<ComfyUI>', 'models', *spec['dest'])
        return f'curl -L -o "{dest}" "{spec["url"]}"'
    return ''


def status(action) -> dict:
    run = _runs.get(action)
    cmd = manual_command(action)
    if run is None:
        return {'state': 'idle', 'returncode': None, 'log': [], 'progress': None,
                'manual_command': cmd}
    return {'state': run['state'], 'returncode': run['returncode'],
            'log': list(run['log']), 'progress': run.get('progress'),
            'manual_command': cmd}


def start(action) -> dict:
    if action not in INSTALL_ACTIONS:
        raise ValueError(f'unknown action: {action}')
    with _lock:
        run = _runs.get(action)
        if run and run['state'] == 'running':
            raise AlreadyRunning(action)
        if action == 'ollama_model':
            _check_ollama_precondition()
        if action in _KLEIN_DOWNLOADS:
            _check_klein_precondition(action)
        _runs[action] = _new_run()
    threading.Thread(target=_execute, args=(action,), daemon=True).start()
    return status(action)


def _check_ollama_precondition():
    if not (cfg.get('ollama.url') or '').strip():
        raise Precondition('ollama.url not configured')
    if not (cfg.get('ollama.vision_model') or '').strip():
        raise Precondition('ollama.vision_model not configured')


def _klein_dest_path(action) -> str:
    """Absolute destination for a Klein download, under the VALIDATED ComfyUI
    models root. Raises Precondition when base_dir isn't a real install (we must
    never scatter multi-GB files under a wrong folder)."""
    r = capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')
    if not r['valid']:
        raise Precondition('point the app at a valid ComfyUI folder first (Setup, ComfyUI step)')
    spec = _KLEIN_DOWNLOADS[action]
    return os.path.join(r['resolved'], 'models', *spec['dest'])


def _check_klein_precondition(action):
    """Don't block the download if the file already exists in the main location
    OR in any extra_model_paths.yaml location (via resolution helpers)."""
    # First check if it's already present in the main dest
    try:
        dest = _klein_dest_path(action)
        if os.path.isfile(dest):
            # Already there – no need to download
            return
    except Precondition:
        # ComfyUI not configured – let it raise
        raise

    # If not in main, check extras via resolution
    installed = _klein_is_installed(action)
    if installed:
        # The file exists somewhere in extra paths – we can skip the download
        return

    # If not installed anywhere, check disk space for the download
    spec = _KLEIN_DOWNLOADS[action]
    try:
        # Use the parent of the ComfyUI models root for disk space check
        r = capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')
        if r['valid']:
            base_dir = r['resolved']
            free_gb = shutil.disk_usage(os.path.dirname(base_dir)).free / 1e9
            if free_gb < spec['min_free_gb']:
                raise Precondition(f'not enough disk space: {free_gb:.1f} GB free, '
                                   f"~{spec['min_free_gb']} GB needed for this file")
    except OSError:
        pass   # unknown -> never block on a stat failure


def _klein_is_installed(action) -> bool:
    """Return True if the Klein asset is found in the main location OR
    in any extra_model_paths.yaml location (via the resolution functions)."""
    if action == 'klein_model':
        return resolve_klein_unet() is not None
    if action == 'klein_vae':
        return resolve_klein_vae() is not None
    if action == 'klein_text_encoder':
        return resolve_klein_text_encoder() is not None
    if action == 'klein_lora':
        _, path = _consistency_lora()
        return path is not None and os.path.exists(path)
    return False


def get_klein_installed_status():
    """Return a dict with keys for each Klein asset and a bool indicating if
    it's installed (either in the main ComfyUI models folder or in any extra
    path defined in extra_model_paths.yaml)."""
    return {
        'unet': resolve_klein_unet() is not None,
        'vae': resolve_klein_vae() is not None,
        'text_encoder': resolve_klein_text_encoder() is not None,
        'consistency_lora': _consistency_lora()[1] is not None,
    }


def _execute(action):
    try:
        rc = _WORKERS[action](action)
        _runs[action]['returncode'] = rc
        _runs[action]['state'] = 'success' if rc == 0 else 'error'
        if action in _IMPORT_CACHE_ACTIONS and rc == 0:
            try:
                capabilities.clear_import_cache()
            except Exception:
                # never downgrade a successful install; surface at debug only
                logger.debug('clear_import_cache failed after %s', action, exc_info=True)
        if action in _KLEIN_DOWNLOADS and rc == 0:
            # The training-base/model listers cache their scans 5 min — a freshly
            # downloaded model must show up on the next probe, not in 5 minutes.
            try:
                from .utils import comfyui
                comfyui.clear_model_caches()
            except Exception:
                logger.debug('clear_model_caches failed after %s', action, exc_info=True)
    except Exception as e:  # never let a worker thread die silently
        _append(action, f'error: {e}')
        _runs[action]['returncode'] = -1
        _runs[action]['state'] = 'error'


def _run_ml_extras(action) -> int:
    """Generic `pip install -r` worker (name kept for existing callers/tests):
    serves ml_extras AND scrape_extras via _PIP_REQUIREMENTS."""
    # ml_extras (insightface/numpy<2/onnx) has no wheels outside Python 3.10–3.12;
    # on a newer interpreter pip source-builds and fails with a cryptic numpy
    # conflict. Lead the log with a plain-English explanation + the fix so the
    # traceback that follows is already contextualized. (scrape_extras is pure
    # Python — no such ceiling — so it's exempt.)
    if action == 'ml_extras':
        ps = capabilities.python_ml_status()
        if not ps['ml_supported']:
            for line in (
                '=' * 64,
                f"NOTE: this app runs on Python {ps['version']}, but the ML extras",
                f"need Python {ps['ml_range']} (insightface / numpy<2 / onnxruntime",
                "publish no wheels for newer versions → pip will try to BUILD them",
                "from source and the install will likely fail below.",
                "",
                "These extras are OPTIONAL — they only add face-resemblance scoring",
                "and background masking. You can:",
                "  1. Skip them (the app works without them), or",
                "  2. Install them into a separate Python 3.11/3.12 venv and set",
                "     face_scoring.python + masks.python to it in Settings.",
                '=' * 64,
            ):
                _append(action, line)
    proc = subprocess.Popen(
        [sys.executable, '-m', 'pip', 'install', '-r',
         str(_PIP_REQUIREMENTS.get(action, _ML_REQUIREMENTS))],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        _append(action, line)
    proc.wait()
    return proc.returncode


def _run_watermark_inpaint(action) -> int:
    """Install JUST the watermark-inpainting package (simple-lama-inpainting, plus
    its torch/opencv deps) into the interpreter the LaMa wrapper resolves — NOT
    necessarily this app's venv (the ML extras can live in a separate 3.10–3.12
    env pointed to by watermark.python/masks.python). A user who already ran the
    ML extras step keeps rembg/insightface: pip skips the already-satisfied ones.
    The version floor is READ from requirements-ml.txt (single source of truth),
    and that file rides along as a CONSTRAINT (-c) so pulling torch can never bump
    numpy past insightface's <2 ceiling and silently break face scoring."""
    python = _watermark_python()
    spec = _requirement_spec(_WATERMARK_PKG)
    _append(action, f'target interpreter: {python}')
    _append(action, f'installing {spec}  (constraints: requirements-ml.txt)')
    proc = subprocess.Popen(
        [python, '-m', 'pip', 'install', spec, '-c', str(_ML_REQUIREMENTS)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        _append(action, line)
    proc.wait()
    return proc.returncode


def _run_ml_capability(action) -> int:
    """Install JUST the packages ONE ML capability needs (face_scoring | masks)
    into the interpreter that capability's probe resolves — so a user can install
    or REPAIR a single feature without the monolithic `-r requirements-ml.txt`.
    Versions come solely from requirements-ml.txt (via _requirement_spec) and that
    file rides along as a `-c` constraint, so pulling insightface/rembg deps can
    never bump numpy past the <2 ABI ceiling and break the other ML capabilities.
    Same shape as _run_watermark_inpaint (resolved ML python, -c constraint)."""
    python = _capability_python(action)
    specs = [_requirement_spec(p) for p in _CAPABILITY_PACKAGES[action]]
    # face_scoring pulls insightface, which only has wheels for Python 3.10–3.12.
    # When targeting THIS interpreter (no dedicated env) and it's out of range,
    # lead with the plain-English reason so the pip source-build failure below is
    # already contextualised — same courtesy the monolithic ml_extras worker gives.
    if action == 'face_scoring' and python == sys.executable:
        ps = capabilities.python_ml_status()
        if not ps['ml_supported']:
            _append(action, f"NOTE: Python {ps['version']} is outside the ML wheel "
                            f"range {ps['ml_range']} — insightface has no wheel here, "
                            "so pip will try to build it and likely fail. Install into a "
                            "separate 3.11/3.12 env and set face_scoring.python instead.")
    _append(action, f'target interpreter: {python}')
    _append(action, f"installing {', '.join(specs)}  (constraints: requirements-ml.txt)")
    proc = subprocess.Popen(
        [python, '-m', 'pip', 'install', *specs, '-c', str(_ML_REQUIREMENTS)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        _append(action, line)
    proc.wait()
    return proc.returncode


def _run_klein_download(action) -> int:
    """Stream one Klein asset into the validated ComfyUI tree. Writes to a .part
    file then renames (a killed download never leaves a half file the model
    scanners would pick up). Progress lines land in the ring log (~every 512 MB).
    Gated repo without credentials -> actionable recovery steps, rc 1."""
    spec = _KLEIN_DOWNLOADS[action]
    dest = _klein_dest_path(action)
    # If the file already exists (main or extra), skip download
    if os.path.isfile(dest):
        _append(action, f'already present: {dest}')
        return 0
    # If it exists in an extra path, we could copy or symlink, but for simplicity
    # we just note it and skip download (the user can symlink).
    if _klein_is_installed(action):
        _append(action, f'file found in extra_model_paths.yaml – skipping download (symlink or copy to main if needed)')
        return 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {}
    token = cfg.secret('HF_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    _append(action, f"downloading {spec['url']}")
    _append(action, f'-> {dest}')
    part = dest + '.part'
    try:
        with requests.get(spec['url'], stream=True, timeout=(10, 120),
                          headers=headers, allow_redirects=True) as resp:
            if resp.status_code in (401, 403):
                if spec.get('gated'):
                    _append(action, f'HTTP {resp.status_code} - this repository is license-gated.')
                    _append(action, f"1. Open {spec['license_url']} and accept the agreement (free)")
                    _append(action, '2. Create a read token at https://huggingface.co/settings/tokens')
                    _append(action, '3. Paste it as HF_TOKEN in Settings -> API keys, then retry')
                    _append(action, '   (or download the file manually into the folder above)')
                else:
                    _append(action, f'HTTP {resp.status_code}')
                return 1
            if resp.status_code >= 400:
                _append(action, f'HTTP {resp.status_code}')
                return 1
            total = int(resp.headers.get('content-length') or 0)
            done = 0
            next_mark = 0
            _set_progress(action, 0, total)   # show the bar from the first byte
            with open(part, 'wb') as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    _set_progress(action, done, total)   # live % for the UI bar (every chunk)
                    if done >= next_mark:                 # coarse milestone in the text log
                        pct = f' ({done * 100 // total}%)' if total else ''
                        _append(action, f'{done / 1e9:.2f} / {total / 1e9:.2f} GB{pct}')
                        next_mark = done + 512 * 1024 * 1024
        if total and done < total:
            _append(action, f'incomplete download ({done}/{total} bytes) - retry')
            os.remove(part)
            return 1
        os.replace(part, dest)
        _append(action, f'done -> {dest}')
        return 0
    except requests.RequestException as e:
        _append(action, f'network error: {e}')
        try:
            os.remove(part)
        except OSError:
            pass
        return 1


def _run_ollama_model(action) -> int:
    url = (cfg.get('ollama.url') or '').rstrip('/')
    model = cfg.get('ollama.vision_model') or ''
    resp = requests.post(f'{url}/api/pull', json={'name': model, 'stream': True},
                         stream=True, timeout=None)
    if resp.status_code >= 400:
        _append(action, f'HTTP {resp.status_code}')
        return 1
    for line in resp.iter_lines():
        if line:
            _append(action, line.decode('utf-8', 'replace') if isinstance(line, bytes) else str(line))
    return 0


_WORKERS = {**{a: _run_ml_extras for a in _PIP_REQUIREMENTS},   # ml_extras + scrape_extras
            'ollama_model': _run_ollama_model,
            **{a: _run_ml_capability for a in _CAPABILITY_ML_ACTIONS},  # face_scoring + masks
            'watermark_inpaint': _run_watermark_inpaint,
            **{a: _run_klein_download for a in _KLEIN_DOWNLOADS}}
# Structural invariant: every whitelisted action MUST have a worker — a missing
# entry surfaces as a cryptic "error: '<action>'" KeyError at runtime (live
# repro: scrape_extras was added to INSTALL_ACTIONS but not here).
assert set(INSTALL_ACTIONS) == set(_WORKERS), \
    f'INSTALL_ACTIONS/_WORKERS mismatch: {set(INSTALL_ACTIONS) ^ set(_WORKERS)}'