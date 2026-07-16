"""In-app self-update for GIT checkouts: report how many commits behind origin the
working tree is, `git pull --ff-only`, then relaunch the server.

Dependency installation is deliberately deferred to the detached restart helper.
Running pip inside the live Flask process can corrupt locked packages on Windows.

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

from ..config import REPO_ROOT, get as _cfg_get

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
        # Links so the user can read WHAT the pending update contains before
        # pulling: a compare view of exactly the incoming commits when behind,
        # else the branch history. Short SHAs work fine in GitHub URLs.
        repo = _cfg_get('updates.repo') or 'perfectgf/lora-dataset-studio'
        base['repo'] = repo
        base['commits_url'] = f'https://github.com/{repo}/commits/{branch}'
        if n > 0 and base['current_sha'] and base['remote_sha']:
            base['compare_url'] = (f'https://github.com/{repo}/compare/'
                                   f"{base['current_sha']}...{base['remote_sha']}")
    except FileNotFoundError:
        base['git_missing'] = True
        base['reason'] = 'git is not installed / not on PATH — install Git to enable in-app updates.'
    except subprocess.SubprocessError:
        base['reason'] = 'git command timed out.'
    return base


def apply_update(root=None) -> dict:
    """`git pull --ff-only` and report whether requirements changed.

    This function never invokes pip: the route passes ``deps_changed`` to
    :func:`schedule_restart`, whose detached helper installs requirements only
    after this process has exited and released imported package files.
    """
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
    return {'ok': True, 'changed': changed, 'from': before[:8], 'to': after[:8],
            'deps_changed': deps_changed, 'log': log[-1500:]}


def _dependency_install_command(root=None, executable=None) -> list[str]:
    """Command embedded in the detached helper (kept separate for unit tests)."""
    root = root or REPO_ROOT
    executable = executable or sys.executable
    return [executable, '-m', 'pip', 'install', '-q', '-r',
            str(root / 'backend' / 'requirements.txt')]


def _restart_helper_code(py, run_py, workdir, port, *, install_requirements=False,
                         root=None):
    root = root or REPO_ROOT
    dependency_command = (_dependency_install_command(root, py)
                          if install_requirements else None)
    return (
        'import os,socket,time,subprocess\n'
        f'port={port!r}\n'
        f'dependency_command={dependency_command!r}\n'
        f'repo_root={str(root)!r}\n'
        'for _ in range(120):\n'
        '    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n'
        '    try:\n'
        '        s.bind(("127.0.0.1",int(port))); s.close(); break\n'
        '    except OSError:\n'
        '        s.close(); time.sleep(0.5)\n'
        # Package files are now unlocked because the old Flask process is gone.
        'if dependency_command:\n'
        '    try:\n'
        '        subprocess.run(dependency_command, cwd=repo_root, check=False, timeout=900)\n'
        '    except Exception as exc:\n'
        '        print(f"[LDS] dependency update failed: {exc}", flush=True)\n'
        # New visible console for the relaunched server: the helper itself is
        # DETACHED, so a default spawn would leave the server console-less and
        # the old launcher window frozen on stale output.
        'flags=0x00000010 if os.name=="nt" else 0\n'
        f'subprocess.Popen([{py!r},{run_py!r}], cwd={workdir!r}, creationflags=flags)\n'
    )


def schedule_restart(delay: float = 1.2, *, install_requirements: bool = False) -> None:
    """Relaunch the server, then hard-exit this process. A DETACHED helper waits for our
    port to free (this process fully gone) before binding, so we never hit the Windows
    'address already in use' rebind race a bare os.execv can trigger. The helper inherits
    our env, so LDS_HOST/LDS_PORT and the LDS_ACCESS_TOKEN (hence the bind + the token)
    stay identical across the restart. When ``install_requirements`` is true,
    pip runs in that helper only after the old port is free (and therefore after
    this process released imported package files). ``delay`` lets the HTTP
    response flush first."""
    py = sys.executable
    run_py = os.path.abspath(sys.argv[0])
    workdir = os.path.dirname(run_py) or None
    # Wait on the port we actually serve on: LDS_PORT wins, else the configured
    # server.port — NOT a hardcoded 5000. On a machine where 5000 belongs to
    # another app, the helper stalled its whole 60 s retry budget against a
    # port that would never free before relaunching (observed live 2026-07-12).
    port = int(os.environ.get('LDS_PORT') or _cfg_get('server.port') or 5000)
    helper = _restart_helper_code(py, run_py, workdir, port,
                                  install_requirements=install_requirements)

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
