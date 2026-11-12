"""Refuse to start on a Python older than this version supports, before anything runs.

Pillow 13 and filelock 4.2 (backend/requirements.txt) need Python 3.11 or newer,
so an environment built on 3.10 cannot hold this version's dependencies. An
in-app update reaches here with the new files and the old environment: the
restart reuses the interpreter it had, and pip refuses the new pins before it
changes anything. This check runs before every import that could touch the
environment or the data folder (ensure_pillow_consistent may run pip,
create_app opens the database), so that environment stays exactly as it was
and the message says how this kind of install gets a supported Python:

* start.bat (Windows ZIP or Git install): run it again. It rebuilds .venv on
  Python 3.11-3.12, downloading 3.12 when none is installed, and keeps data/,
  config.json and the Image Bank.
* Pinokio: Stop, Update (it rebuilds env/ on Python 3.12), Start.
* Anything else: recreate the virtual environment on Python 3.11 or newer.

Under the start.bat supervisor the process then waits instead of exiting. An
exit would be relaunched five times, then hand control back to the start.bat
that was running before the update, whose file the update replaced: cmd would
go on reading the new file from the old position. Closing the window ends it.
"""
import os
import sys
import time

MINIMUM = (3, 11)


def supported(version_info=None) -> bool:
    return tuple((version_info or sys.version_info)[:2]) >= MINIMUM


def _install_kind(repo_root, runtime) -> str:
    if (runtime or '').strip().lower() == 'pinokio':
        return 'pinokio'
    if os.name == 'nt' and os.path.isfile(os.path.join(repo_root, 'start.bat')):
        return 'start.bat'
    return 'other'


def refusal(*, repo_root, executable, version, runtime=None) -> str:
    """What to say, for this kind of install. Plain text, one paragraph."""
    floor = '.'.join(map(str, MINIMUM))
    kind = _install_kind(repo_root, runtime)
    if kind == 'start.bat':
        return (f'[LDS] This version of LoRA Dataset Studio needs Python {floor} or newer,\n'
                f'      and its environment (.venv) still runs Python {version}. Nothing was changed.\n'
                f'      Close this window, then double-click start.bat: it rebuilds .venv on\n'
                f'      Python 3.12 (downloading it if needed) and starts the app. Datasets,\n'
                f'      settings and the Image Bank are kept; optional ML extras that were\n'
                f'      installed into .venv are installed again from Setup.')
    if kind == 'pinokio':
        return (f'[LDS] This version of LoRA Dataset Studio needs Python {floor} or newer,\n'
                f'      and the Pinokio environment (env) still runs Python {version}.\n'
                f'      Nothing was changed. In Pinokio 3.8.27 or newer: Stop, click Update (it\n'
                f'      rebuilds env on Python 3.12; the first Update after this version may only\n'
                f'      fetch the new installer, then click it again), then Start.')
    return (f'[LDS] This version of LoRA Dataset Studio needs Python {floor} or newer;\n'
            f'      {executable} is Python {version}. Nothing was changed.\n'
            f'      Recreate the virtual environment on Python 3.11 or 3.12, install\n'
            f'      backend/requirements.txt into it, then start the app again.')


def enforce(repo_root, *, sleep=time.sleep, stdin=None, version_info=None):
    """Return when this interpreter is supported; otherwise say why and stop."""
    version_info = version_info or sys.version_info
    if getattr(sys, 'frozen', False) or supported(version_info):
        return
    runtime = os.environ.get('LDS_RUNTIME')
    print(refusal(repo_root=repo_root, executable=sys.executable,
                  version='.'.join(map(str, version_info[:3])), runtime=runtime), flush=True)
    if os.environ.get('LDS_RESTART_MODE', '').strip().lower() == 'supervisor':
        while True:            # the window stays readable; closing it ends the process tree
            sleep(3600)
    stdin = sys.stdin if stdin is None else stdin
    if (runtime or '').strip().lower() != 'pinokio' and stdin is not None and stdin.isatty():
        try:
            input('Press Enter to close this window.')
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(1)
