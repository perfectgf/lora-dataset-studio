"""Start the local Ollama server on demand — powers the Settings / Setup
"Start Ollama" button.

Whether Ollama is INSTALLED (binary on disk, server up or not) is detected
passively in capabilities (`_ollama_binary` / `probe_ollama_installed`). This
module owns the one thing that mutates the machine: LAUNCHING the server. It
spawns a DETACHED ``ollama serve`` that outlives this app (and an app restart),
then polls the configured URL until it answers (or a timeout), and reports
``{ok, reachable, ...}``.

GUARD-RAIL: nothing here may run from a passive probe. It is only ever reached
through an explicit user click on the Start button (POST /api/ollama/start), so
the app can never silently spawn a server behind the user's back.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse

import requests

from .. import capabilities
from .. import config as cfg

logger = logging.getLogger(__name__)

_DEFAULT_URL = 'http://127.0.0.1:11434'
# How long to wait for a freshly-spawned server to answer before giving up. A
# cold ``ollama serve`` binds its port in well under this; the ceiling only bites
# a genuinely stuck launch (→ structured error + the stderr tail).
_READY_TIMEOUT = 15.0
_POLL_INTERVAL = 0.5
_STDERR_TAIL = 2000     # chars of the launch log surfaced on failure


def _url() -> str:
    return (cfg.get('ollama.url') or _DEFAULT_URL).rstrip('/')


def _reachable(url) -> bool:
    """Does a server answer at `url`? Goes through capabilities._http_ok so the
    suite patches one seam and the real network is never hit in tests."""
    return capabilities._http_ok(f'{url}/api/tags')


def _spawn_detached(binary: str):
    """Spawn ``ollama serve`` fully DETACHED so it survives this app. stdout+stderr
    go to a temp FILE, never a PIPE: a chatty server would eventually fill an
    undrained pipe buffer and stall — the file avoids that AND lets us read the
    tail if the launch dies early. Returns the Popen; its ``stderr_path``
    attribute points at the log file."""
    log = tempfile.NamedTemporaryFile(prefix='lds_ollama_', suffix='.log', delete=False)
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                  close_fds=True)
    if os.name == 'nt':
        # DETACHED_PROCESS (0x8): no console, not tied to ours.
        # CREATE_NEW_PROCESS_GROUP (0x200): our termination / Ctrl-C can't cascade.
        # CREATE_NO_WINDOW (0x8000000): belt-and-braces, no console window flashes.
        kwargs['creationflags'] = 0x00000008 | 0x00000200 | 0x08000000
    else:
        kwargs['start_new_session'] = True     # POSIX: own session, survives us
    proc = subprocess.Popen([binary, 'serve'], **kwargs)
    log.close()          # parent lets go; the child keeps the fd it inherited
    proc.stderr_path = log.name
    return proc


def _stderr_tail(proc) -> str:
    path = getattr(proc, 'stderr_path', None)
    if not path:
        return ''
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            return fh.read()[-_STDERR_TAIL:].strip()
    except OSError:
        return ''


def start_ollama(*, wait_timeout: float = _READY_TIMEOUT,
                 poll_interval: float = _POLL_INTERVAL) -> dict:
    """Idempotently bring the local Ollama server up. Returns:
      already up           -> {ok:True,  reachable:True, already_running:True}
      started & answered   -> {ok:True,  reachable:True}
      not installed        -> {ok:False, reachable:False, error:...}
      launch failed        -> {ok:False, reachable:False, error:...}
      spawned, never ready -> {ok:False, reachable:False, error:..., stderr:...}
    Never raises."""
    url = _url()
    # Idempotent no-op: a running server must never be double-spawned.
    if _reachable(url):
        return {'ok': True, 'reachable': True, 'already_running': True}

    binary = capabilities._ollama_binary()
    if not binary:
        return {'ok': False, 'reachable': False,
                'error': 'Ollama is not installed (no binary on PATH or in the default '
                         'install location). Install it from https://ollama.com/download.'}
    try:
        proc = _spawn_detached(binary)
    except OSError as e:
        logger.warning('ollama start: launch failed: %s', e)
        return {'ok': False, 'reachable': False, 'error': f'could not launch Ollama: {e}'}

    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if _reachable(url):
            return {'ok': True, 'reachable': True}
        if proc.poll() is not None:
            # Exited before answering (port already bound by a broken instance,
            # missing runner, etc.) -> stop waiting and surface why.
            break
        time.sleep(poll_interval)

    # Final check closes the race between the last sleep and the deadline.
    if _reachable(url):
        return {'ok': True, 'reachable': True}
    out = {'ok': False, 'reachable': False,
           'error': f'Ollama did not become reachable within {int(wait_timeout)}s.'}
    tail = _stderr_tail(proc)
    if tail:
        out['stderr'] = tail
    return out


def _is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or '').lower()
    return host == 'localhost' or host == '::1' or host.startswith('127.')


def ensure_captioning_ready() -> dict:
    """Start a stopped local Ollama on demand and verify the configured model.

    A remote configured URL is never replaced by a local process. The caller is an
    explicit caption action, not a passive capability probe.
    """
    url = _url()
    if not _reachable(url):
        if not _is_loopback_url(url):
            return {'ok': False, 'reachable': False,
                    'error': f'Remote Ollama server is unreachable: {url}' }
        started = start_ollama()
        if not started.get('ok') or not started.get('reachable'):
            return {'ok': False, 'reachable': False,
                    'error': started.get('error') or 'Ollama could not start'}
    model = capabilities.probe_ollama_model(reachable=True)
    if not model.get('ok'):
        return {'ok': False, 'reachable': True,
                'error': model.get('detail') or 'Configured Ollama vision model is not available'}
    return {'ok': True, 'reachable': True, 'model_ready': True}


# --- Installed-model listing + parametrized pull -----------------------------
# The Captions ⚙️ Options popover lets a user pick which pulled Ollama model captions
# a dataset, and pull a new one they name. A pull takes an ARBITRARY model name, so it
# lives here (an Ollama action) rather than in setup_installer, whose fixed catalog
# deliberately takes no client-supplied arguments. The name is validated to Ollama's
# own reference charset (never shelled out — it's a JSON field to the local server).
_MODEL_REF_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$')

# One pull at a time, tracked so the popover can poll {state, model, progress, log, error}.
_pull_lock = threading.Lock()
_pull = None            # None = never run this process
_PULL_LOG_TAIL = 40     # status lines kept for the UI


def list_models() -> dict:
    """Installed Ollama models for the model picker. {ok, reachable, models:[...]}.
    Never raises; an unreachable server returns reachable=False and an empty list."""
    url = _url()
    if not _reachable(url):
        return {'ok': False, 'reachable': False, 'models': []}
    return {'ok': True, 'reachable': True, 'models': capabilities._ollama_tags(url)}


def _pull_snapshot() -> dict:
    if _pull is None:
        return {'state': 'idle', 'model': '', 'progress': None, 'log': [], 'error': None}
    return {'state': _pull['state'], 'model': _pull['model'],
            'progress': _pull['progress'], 'log': list(_pull['log']),
            'error': _pull['error']}


def pull_status() -> dict:
    with _pull_lock:
        return _pull_snapshot()


def _run_pull(model: str):
    url = _url()
    try:
        resp = requests.post(f'{url}/api/pull', json={'name': model, 'stream': True},
                             stream=True, timeout=(10, None))
        if resp.status_code >= 400:
            _finish_pull('error', error=f'Ollama rejected the pull (HTTP {resp.status_code})')
            return
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode('utf-8', 'replace'))
            except (ValueError, TypeError):
                continue
            if obj.get('error'):
                _finish_pull('error', error=str(obj['error']))
                return
            status = str(obj.get('status') or '').strip()
            total, done = obj.get('total'), obj.get('completed')
            pct = int(done * 100 / total) if total and done is not None else None
            with _pull_lock:
                if _pull is not None:
                    _pull['progress'] = pct
                    if status and (not _pull['log'] or _pull['log'][-1] != status):
                        _pull['log'].append(status)
                        del _pull['log'][:-_PULL_LOG_TAIL]
        _finish_pull('success')
    except requests.RequestException as e:
        _finish_pull('error', error=f'network error: {e}')


def _finish_pull(state: str, error: str | None = None):
    with _pull_lock:
        if _pull is not None:
            _pull['state'] = state
            _pull['error'] = error
            if state == 'success':
                _pull['progress'] = 100
    # A finished pull adds/updates a model → the cached capability probe must re-check.
    if state == 'success':
        try:
            capabilities.clear_import_cache()
        except Exception:  # noqa: BLE001 - never fail a successful pull on a cache miss
            logger.debug('clear_import_cache after ollama pull failed', exc_info=True)


def start_pull(model: str) -> dict:
    """Start pulling an Ollama model by name (background). Returns the pull status.
    A blank/invalid name is rejected; a pull already running is left untouched and its
    status returned (the UI polls it). Requires a reachable server."""
    global _pull
    model = (model or '').strip()
    if not _MODEL_REF_RE.match(model):
        return {**_pull_snapshot(), 'ok': False,
                'error': 'Enter a valid Ollama model name (e.g. '
                         'huihui_ai/qwen3-vl-abliterated:8b-instruct).'}
    if not _reachable(_url()):
        return {**_pull_snapshot(), 'ok': False,
                'error': 'Ollama is not reachable — start it first.'}
    with _pull_lock:
        if _pull is not None and _pull['state'] == 'running':
            return {'ok': True, 'already_running': True, **_pull_snapshot()}
        _pull = {'state': 'running', 'model': model, 'progress': None,
                 'log': [], 'error': None}
    threading.Thread(target=_run_pull, args=(model,), daemon=True).start()
    with _pull_lock:
        return {'ok': True, **_pull_snapshot()}
