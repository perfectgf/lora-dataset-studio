"""Setup installer: run whitelisted, self-contained installs in a background
thread and expose their live state for polling. Two actions only:

  ml_extras    -> pip install -r backend/requirements-ml.txt (the app's own venv)
  ollama_model -> stream Ollama's /api/pull for the configured vision model

No shell, no client-supplied arguments: each action's command is fixed.
"""
import subprocess
import sys
import threading

import requests

from . import capabilities
from . import config as cfg

INSTALL_ACTIONS = ('ml_extras', 'ollama_model')

_ML_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-ml.txt'
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


def status(action) -> dict:
    run = _runs.get(action)
    if run is None:
        return {'state': 'idle', 'returncode': None, 'log': []}
    return {'state': run['state'], 'returncode': run['returncode'], 'log': list(run['log'])}


def start(action) -> dict:
    if action not in INSTALL_ACTIONS:
        raise ValueError(f'unknown action: {action}')
    with _lock:
        run = _runs.get(action)
        if run and run['state'] == 'running':
            raise AlreadyRunning(action)
        if action == 'ollama_model':
            _check_ollama_precondition()
        _runs[action] = _new_run()
    threading.Thread(target=_execute, args=(action,), daemon=True).start()
    return status(action)


def _check_ollama_precondition():
    if not (cfg.get('ollama.url') or '').strip():
        raise Precondition('ollama.url not configured')
    if not (cfg.get('ollama.vision_model') or '').strip():
        raise Precondition('ollama.vision_model not configured')


def _execute(action):
    try:
        rc = _WORKERS[action](action)
        _runs[action]['returncode'] = rc
        _runs[action]['state'] = 'success' if rc == 0 else 'error'
        if action == 'ml_extras' and rc == 0:
            try:
                capabilities.clear_import_cache()
            except Exception:
                pass  # don't downgrade a successful install
    except Exception as e:  # never let a worker thread die silently
        _append(action, f'error: {e}')
        _runs[action]['returncode'] = -1
        _runs[action]['state'] = 'error'


def _run_ml_extras(action) -> int:
    proc = subprocess.Popen(
        [sys.executable, '-m', 'pip', 'install', '-r', str(_ML_REQUIREMENTS)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        _append(action, line)
    proc.wait()
    return proc.returncode


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


_WORKERS = {'ml_extras': _run_ml_extras, 'ollama_model': _run_ollama_model}
