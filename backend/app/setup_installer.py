"""Setup installer: run whitelisted, self-contained installs in a background
thread and expose their live state for polling. Actions:

  ml_extras          -> pip install -r backend/requirements-ml.txt (the app's own venv)
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
import shutil
import subprocess
import sys
import threading

import requests

from . import capabilities
from . import config as cfg

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

INSTALL_ACTIONS = ('ml_extras', 'scrape_extras', 'ollama_model') + tuple(_KLEIN_DOWNLOADS)

_ML_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-ml.txt'
_SCRAPE_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-scrape.txt'
# pip -r installers share one worker; both target THIS interpreter (the scrape
# stack runs in-process, so any other environment would be invisible to the app).
_PIP_REQUIREMENTS = {'ml_extras': _ML_REQUIREMENTS, 'scrape_extras': _SCRAPE_REQUIREMENTS}
_LOG_MAX = 400  # ring-buffer the log so a chatty pip can't grow unbounded

_lock = threading.Lock()
_runs = {}  # action -> {'state', 'returncode', 'log'}


class AlreadyRunning(Exception):
    pass


class Precondition(Exception):
    pass


def _new_run():
    return {'state': 'running', 'returncode': None, 'log': []}


def _append(action, line):
    log = _runs[action]['log']
    log.append(line.rstrip('\n'))
    if len(log) > _LOG_MAX:
        del log[:-_LOG_MAX]


def _quote(p: str) -> str:
    # Quote paths with spaces so the manual command is copy-paste-safe: the
    # portable bundle can be extracted under e.g. C:\Users\...\LoRA Dataset Studio\.
    return f'"{p}"' if ' ' in p else p


def manual_command(action) -> str:
    """The exact command that reproduces an install BY HAND, scoped to THIS app's
    own interpreter (sys.executable). A copy-paste then targets the SAME
    environment the app imports from -- the portable bundle's python\\python.exe or
    the dev venv -- instead of whatever bare `pip` happens to be first on PATH
    (which is the whole point of the user's question: a plain `pip install` would
    land in the wrong environment and the extras would never be importable)."""
    if action in _PIP_REQUIREMENTS:
        return f'{_quote(sys.executable)} -m pip install -r {_quote(str(_PIP_REQUIREMENTS[action]))}'
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
        return {'state': 'idle', 'returncode': None, 'log': [], 'manual_command': cmd}
    return {'state': run['state'], 'returncode': run['returncode'],
            'log': list(run['log']), 'manual_command': cmd}


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
    dest = _klein_dest_path(action)
    spec = _KLEIN_DOWNLOADS[action]
    try:
        free_gb = shutil.disk_usage(os.path.dirname(os.path.dirname(dest))).free / 1e9
        if free_gb < spec['min_free_gb']:
            raise Precondition(f'not enough disk space: {free_gb:.1f} GB free, '
                               f"~{spec['min_free_gb']} GB needed for this file")
    except OSError:
        pass   # unknown -> never block on a stat failure


def _execute(action):
    try:
        rc = _WORKERS[action](action)
        _runs[action]['returncode'] = rc
        _runs[action]['state'] = 'success' if rc == 0 else 'error'
        if action in _PIP_REQUIREMENTS and rc == 0:
            try:
                capabilities.clear_import_cache()
            except Exception:
                # never downgrade a successful install; surface at debug only
                logger.debug('clear_import_cache failed after ml_extras', exc_info=True)
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
    proc = subprocess.Popen(
        [sys.executable, '-m', 'pip', 'install', '-r',
         str(_PIP_REQUIREMENTS.get(action, _ML_REQUIREMENTS))],
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
    if os.path.isfile(dest):
        _append(action, f'already present: {dest}')
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
            with open(part, 'wb') as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if done >= next_mark:
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


_WORKERS = {'ml_extras': _run_ml_extras, 'ollama_model': _run_ollama_model,
            **{a: _run_klein_download for a in _KLEIN_DOWNLOADS}}
