"""In-app self-update for GIT checkouts: report how many commits behind origin the
working tree is, `git pull --ff-only`, reinstall deps only if requirements changed,
then relaunch the server.

Only meaningful for a git checkout. A packaged build (the portable bundle) has no
`.git`, so `is_git_checkout()` is False and the caller falls back to the releases
page — a running bundle can't safely overwrite its own locked exe/dlls anyway.
`git` must be on PATH; if it isn't we say so rather than fail cryptically (a clone
user has git by definition, so this only bites an unusual setup).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

from ..config import REPO_ROOT

_GIT_TIMEOUT = 120


def is_git_checkout(root=None) -> bool:
    return (root or REPO_ROOT).joinpath('.git').exists()


def _git(root, *args, timeout=_GIT_TIMEOUT):
    """Run a git subcommand in `root`. Returns the CompletedProcess (never raises on
    non-zero — callers inspect returncode)."""
    git = shutil.which('git')
    if not git:
        raise FileNotFoundError('git')
    return subprocess.run([git, '-C', str(root), *args],
                          capture_output=True, text=True, timeout=timeout)


def current_sha(root=None):
    """Short SHA of the local checkout — local-only (no fetch), None outside a
    git checkout or when git is unavailable. Lets the passive update check show
    the current build without touching the network."""
    root = root or REPO_ROOT
    if not is_git_checkout(root):
        return None
    try:
        return (_git(root, 'rev-parse', '--short', 'HEAD').stdout or '').strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def git_update_status(root=None) -> dict | None:
    """`git fetch` + how many commits behind the upstream branch we are. None when this
    isn't a git checkout (caller then uses the release-tag check). Network/git failures
    degrade to a reason string, never an exception."""
    root = root or REPO_ROOT
    from ..version import APP_VERSION
    if not is_git_checkout(root):
        return None
    base = {'ok': True, 'is_git': True, 'current': APP_VERSION, 'update_available': False}
    try:
        branch = (_git(root, 'rev-parse', '--abbrev-ref', 'HEAD').stdout or '').strip() or 'main'
        fetch = _git(root, 'fetch', '--quiet', 'origin', branch)
        if fetch.returncode != 0:
            base['reason'] = 'git fetch failed (offline, or no access to the remote).'
            return base
        behind = (_git(root, 'rev-list', '--count', f'HEAD..origin/{branch}').stdout or '0').strip()
        base['branch'] = branch
        base['current_sha'] = (_git(root, 'rev-parse', '--short', 'HEAD').stdout or '').strip()
        base['remote_sha'] = (_git(root, 'rev-parse', '--short', f'origin/{branch}').stdout or '').strip()
        try:
            n = int(behind)
        except ValueError:
            n = 0
        base['behind'] = n
        base['update_available'] = n > 0
    except FileNotFoundError:
        base['git_missing'] = True
        base['reason'] = 'git is not installed / not on PATH — install Git to enable in-app updates.'
    except subprocess.SubprocessError:
        base['reason'] = 'git command timed out.'
    return base


def apply_update(root=None) -> dict:
    """`git pull --ff-only` the current branch; if requirements changed in the pulled
    range, reinstall them. Returns {'ok', 'changed', 'from', 'to', 'deps_changed',
    'log', ...}. Does NOT restart — the route schedules that when `changed` is True, so
    this stays pure/testable."""
    root = root or REPO_ROOT
    if not is_git_checkout(root):
        from ..config import get as cfg_get
        repo = cfg_get('updates.repo') or 'perfectgf/lora-dataset-studio'
        return {'ok': False, 'manual': True,
                'reason': 'This is a packaged build (no git checkout) — download the latest '
                          'release and replace the folder.',
                'url': f'https://github.com/{repo}/releases'}
    try:
        branch = (_git(root, 'rev-parse', '--abbrev-ref', 'HEAD').stdout or '').strip() or 'main'
        before = (_git(root, 'rev-parse', 'HEAD').stdout or '').strip()
        pull = _git(root, 'pull', '--ff-only', 'origin', branch)
    except FileNotFoundError:
        return {'ok': False, 'reason': 'git is not installed / not on PATH.'}
    except subprocess.SubprocessError:
        return {'ok': False, 'reason': 'git pull timed out.'}
    log = ((pull.stdout or '') + (pull.stderr or '')).strip()
    if pull.returncode != 0:
        return {'ok': False, 'reason': 'git pull --ff-only failed — local edits or a diverged '
                                       'branch. Resolve them, or re-clone.', 'log': log[-1500:]}
    after = (_git(root, 'rev-parse', 'HEAD').stdout or '').strip()
    changed = bool(before) and before != after
    deps_changed = False
    if changed:
        names = (_git(root, 'diff', '--name-only', before, after).stdout or '')
        deps_changed = any('requirements' in n for n in names.splitlines())
        if deps_changed:
            req = root / 'backend' / 'requirements.txt'
            if req.exists():
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(req)],
                                   capture_output=True, text=True, timeout=900)
                except subprocess.SubprocessError:
                    pass   # non-fatal: the restart still loads new code; deps can be redone
    return {'ok': True, 'changed': changed, 'from': before[:8], 'to': after[:8],
            'deps_changed': deps_changed, 'log': log[-1500:]}


def schedule_restart(delay: float = 1.2) -> None:
    """Relaunch the server, then hard-exit this process. A DETACHED helper waits for our
    port to free (this process fully gone) before binding, so we never hit the Windows
    'address already in use' rebind race a bare os.execv can trigger. The helper inherits
    our env, so LDS_HOST/LDS_PORT and the LDS_ACCESS_TOKEN (hence the bind + the token)
    stay identical across the restart. `delay` lets the HTTP response flush first."""
    py = sys.executable
    run_py = os.path.abspath(sys.argv[0])
    workdir = os.path.dirname(run_py) or None
    port = int(os.environ.get('LDS_PORT') or 5000)
    helper = (
        'import socket,time,subprocess\n'
        f'port={port!r}\n'
        'for _ in range(120):\n'
        '    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n'
        '    try:\n'
        '        s.bind(("127.0.0.1",int(port))); s.close(); break\n'
        '    except OSError:\n'
        '        s.close(); time.sleep(0.5)\n'
        f'subprocess.Popen([{py!r},{run_py!r}], cwd={workdir!r})\n'
    )

    def _spawn_then_exit():
        import time
        time.sleep(delay)
        flags = 0
        if os.name == 'nt':
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
        try:
            subprocess.Popen([py, '-c', helper], cwd=workdir, env=dict(os.environ),
                             creationflags=flags, close_fds=True)
        finally:
            os._exit(0)

    threading.Thread(target=_spawn_then_exit, daemon=True).start()
