"""Install the requirements an update changed, before the app imports them.

start.bat runs the backend under supervise.py, and an in-app update restarts it
through that supervisor (exit 75). The relaunch reuses the environment as it is,
so a version that moved a pin (Pillow 12.3.0 -> 13.0.0) kept running on the old
package until start.bat was launched again: only start.bat, the detached restart
helper (LDS_SUPERVISE=0) and Pinokio's Update ran pip. run.py calls this first,
before anything imports the app or a package it pins: when an installed version
does not satisfy backend/requirements.txt, pip installs the file once and the boot
goes on. A failure is reported and the boot goes on with what is installed, as it
did before.

Only the environment start.bat owns is touched: the repository's own .venv.
Any other interpreter -- a system Python, a hand-made venv, Pinokio's env (its
Update tab installs with uv, into an env without pip), Docker's immutable image,
a frozen build -- is left to whoever installs it.
"""
import subprocess
import sys
from pathlib import Path

def owned_venv(repo_root, prefix=None) -> bool:
    """Whether this interpreter runs from the repository's own .venv (start.bat's)."""
    try:
        return Path(prefix or sys.prefix).resolve() == (Path(repo_root) / '.venv').resolve()
    except OSError:
        return False


def unsatisfied(requirements_path) -> list:
    """`name installed -> wanted` for every applicable requirement this
    environment does not meet; a missing distribution is listed as itself."""
    from importlib.metadata import PackageNotFoundError, version
    from packaging.requirements import Requirement

    missing = []
    for raw in Path(requirements_path).read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line or line.startswith('-'):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            missing.append(f'{requirement.name} (missing) -> {requirement.specifier or "any"}')
            continue
        if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
            missing.append(f'{requirement.name} {installed} -> {requirement.specifier}')
    return missing


def ensure(repo_root, *, run=subprocess.run, out=print, prefix=None):
    """Bring start.bat's .venv up to backend/requirements.txt; never stops the boot."""
    if getattr(sys, 'frozen', False) or not owned_venv(repo_root, prefix):
        return None
    requirements = Path(repo_root) / 'backend' / 'requirements.txt'
    try:
        missing = unsatisfied(requirements)
    except Exception as exc:                 # noqa: BLE001 -- an unreadable file or no packaging
        out(f'[LDS] could not compare the installed packages with requirements.txt ({exc}); starting as is.',
            flush=True)
        return None
    if not missing:
        return []
    shown = ', '.join(missing[:5]) + (f' and {len(missing) - 5} more' if len(missing) > 5 else '')
    out(f'[LDS] installing the requirements this version changed: {shown}', flush=True)
    try:
        code = run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements)],
                   cwd=str(repo_root), timeout=900).returncode
    except Exception as exc:                 # noqa: BLE001 -- pip missing, timeout
        code, missing = None, missing + [str(exc)]
    if code != 0:
        out('[LDS] installing them failed -- see pip above. The app starts with the packages already '
            'installed; close it and run start.bat to try again.', flush=True)
    return missing
