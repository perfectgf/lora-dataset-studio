"""In-app self-update, for both install shapes:

* A GIT checkout: report how many commits behind origin the tree is,
  `git pull --ff-only`, then relaunch the server.
* A packaged install (the Windows release ZIP — no `.git`): compare the latest
  GitHub release tag to the local version, then, on apply, download that
  release's ZIP asset and swap the app's files in place (backing up the old
  ones, rolling back on any mid-way failure), then relaunch. `apply_zip_update`
  preserves everything a ZIP archive never carries anyway — `data/`,
  `config.json`, `.env`, `.venv`, `.python` — so user state and the live runtime
  survive the swap. See `_PROTECTED_TOP_LEVEL`.

Dependency installation is deliberately deferred to the detached restart helper.
Running pip inside the live Flask process can corrupt locked packages on Windows.

`git` must be on PATH for the checkout path; if it isn't we say so rather than
fail cryptically (a clone user has git by definition, so this only bites an
unusual setup).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from ..config import REPO_ROOT, get as _cfg_get

_GIT_TIMEOUT = 120

# Top-level entries the ZIP update must NEVER overwrite or delete: user state and
# the live runtime. A real release ZIP contains none of these (it ships only the
# app source + built frontend + launcher), so this guard is defense in depth — it
# also protects a user who unpacked the archive on top of an existing install.
_PROTECTED_TOP_LEVEL = frozenset({
    'data', 'config.json', '.env', '.venv', 'venv', '.python', '.git',
})


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


# --- Release-ZIP update path (packaged install, no .git) ----------------------

def latest_release(repo=None, timeout=6) -> dict:
    """The repo's latest GitHub release as {version, tag, zip_url, html_url}, or
    {'reason': ...} when the public feed can't be read (offline, rate-limited, no
    release yet). No auth — the release feed of a public repo is world-readable;
    a 403 (rate limit) or network error degrades to a reason, never an exception.
    `version` is the tag with a leading v/V stripped so it compares to APP_VERSION.
    `zip_url` is the .zip asset's download URL (the '*-windows.zip' one wins if
    several exist), or None when the release published no ZIP asset."""
    import requests
    repo = repo or _cfg_get('updates.repo') or 'perfectgf/lora-dataset-studio'
    try:
        r = requests.get(f'https://api.github.com/repos/{repo}/releases/latest',
                         timeout=timeout, headers={'Accept': 'application/vnd.github+json'})
    except requests.RequestException:
        return {'reason': 'offline or GitHub unreachable'}
    if r.status_code != 200:
        return {'reason': f'release feed answered {r.status_code} (no public release yet?)'}
    j = r.json()
    tag = (j.get('tag_name') or '').strip()
    zip_url = None
    for asset in (j.get('assets') or []):
        name = (asset.get('name') or '').lower()
        url = asset.get('browser_download_url')
        if name.endswith('.zip') and url:
            zip_url = url
            if 'windows' in name:      # the supported bundle — prefer it, keep looking otherwise
                break
    return {'version': tag.lstrip('vV').strip() or None, 'tag': tag or None,
            'zip_url': zip_url, 'html_url': j.get('html_url')}


def _staged_app_root(extract_dir) -> Path:
    """Locate the app root inside an extracted release. Compress-Archive wraps
    everything in a single top-level folder (`LoRA-Dataset-Studio.../`), so the
    real tree is usually one level down; tolerate an unwrapped archive too. The
    app root is whichever directory directly contains `backend/`."""
    extract_dir = Path(extract_dir)
    if (extract_dir / 'backend').is_dir():
        return extract_dir
    subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    for d in subdirs:
        if (d / 'backend').is_dir():
            return d
    return subdirs[0] if len(subdirs) == 1 else extract_dir


def plan_update_items(staged_root) -> list[str]:
    """Top-level paths (relative to the app root) the release will replace,
    derived from what the ZIP actually contains — so it adapts if a release adds
    or drops a file. `frontend` is narrowed to `frontend/dist` (never clobber a
    dev-style frontend/src, even though a ZIP install has none). Protected
    user-state / runtime entries are skipped (see `_PROTECTED_TOP_LEVEL`)."""
    staged_root = Path(staged_root)
    items = []
    for name in sorted(p.name for p in staged_root.iterdir()):
        if name in _PROTECTED_TOP_LEVEL:
            continue
        if name == 'frontend':
            if (staged_root / 'frontend' / 'dist').exists():
                items.append('frontend/dist')
            continue
        items.append(name)
    return items


def _requirements_changed(staged_root, root) -> bool:
    """Whether the staged backend/requirements.txt differs from the live one, so
    the caller can ask the restart helper to re-run pip. Any read error errs
    towards True (re-install rather than skip a needed dependency)."""
    staged = Path(staged_root) / 'backend' / 'requirements.txt'
    live = Path(root) / 'backend' / 'requirements.txt'
    try:
        a = staged.read_bytes() if staged.exists() else b''
        b = live.read_bytes() if live.exists() else b''
        return a != b
    except OSError:
        return True


def apply_release_files(staged_root, repo_root, backup_root) -> list[str]:
    """Replace the app's files with the staged release, transactionally.

    For each top-level item (see :func:`plan_update_items`): move the current
    version into `backup_root`, then move the staged version into place. On any
    failure the completed moves are rolled back in reverse and the exception is
    re-raised, leaving the install byte-for-byte as it was. Moves are atomic
    renames on the same volume, so each switch is near-instant and there is no
    partially-copied tree to reason about. Never touches user state / the runtime
    (`_PROTECTED_TOP_LEVEL`). Returns the list of replaced relative paths."""
    staged_root, repo_root, backup_root = Path(staged_root), Path(repo_root), Path(backup_root)
    if backup_root.exists():
        shutil.rmtree(backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)
    done = []                                   # [(rel, had_original)] already switched
    try:
        for rel in plan_update_items(staged_root):
            src = staged_root / rel
            if not src.exists():
                continue
            dst = repo_root / rel
            bak = backup_root / rel
            bak.parent.mkdir(parents=True, exist_ok=True)
            had = dst.exists()
            if had:
                shutil.move(str(dst), str(bak))     # current -> backup
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))     # new -> live
            except Exception:
                if had and bak.exists():            # restore just-backed-up item, then unwind the rest
                    shutil.move(str(bak), str(dst))
                raise
            done.append((rel, had))
        return [rel for rel, _ in done]
    except Exception:
        for rel, had in reversed(done):
            dst = repo_root / rel
            bak = backup_root / rel
            try:
                if dst.exists():
                    shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            except OSError:
                pass
            if had and bak.exists():
                try:
                    shutil.move(str(bak), str(dst))
                except OSError:
                    pass
        raise


def apply_zip_update(root=None) -> dict:
    """Release-ZIP update for a NON-git install: resolve the latest release,
    download its ZIP asset, extract it, and swap the app's files in place
    (backing up the old ones; rolling back on failure). Restart is scheduled by
    the caller. Returns a dict shaped like :func:`apply_update`
    ({ok, changed, from, to, deps_changed, ...}) so the route treats both paths
    the same. `deps_changed` lets the caller defer pip to the restart helper."""
    root = Path(root or REPO_ROOT)
    from ..version import APP_VERSION
    from ..config import data_dir
    rel = latest_release()
    if rel.get('reason'):
        return {'ok': False, 'reason': rel['reason']}
    latest = rel.get('version')
    if not latest:
        return {'ok': False, 'reason': 'the latest release has no version tag to update to.'}
    if not (latest > APP_VERSION):          # date-based versions -> plain string comparison
        return {'ok': True, 'changed': False, 'from': APP_VERSION, 'to': latest}
    if not rel.get('zip_url'):
        repo = _cfg_get('updates.repo') or 'perfectgf/lora-dataset-studio'
        return {'ok': False, 'manual': True,
                'reason': 'the latest release published no downloadable ZIP asset.',
                'url': rel.get('html_url') or f'https://github.com/{repo}/releases'}

    # Everything transient lives under data/ (writable, protected from the swap,
    # same volume as the app -> atomic renames). Wiped and recreated each run.
    work = data_dir() / '_update'
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    zip_path, staging, backup = work / 'release.zip', work / 'staging', work / 'backup'

    try:
        _download_file(rel['zip_url'], zip_path)
    except Exception as exc:
        return {'ok': False, 'reason': f'download failed: {exc}'}
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging)
    except (zipfile.BadZipFile, OSError) as exc:
        return {'ok': False, 'reason': f'the downloaded release is not a readable ZIP: {exc}'}

    app_root = _staged_app_root(staging)
    if not (app_root / 'backend' / 'app').is_dir() or not (app_root / 'backend' / 'run.py').exists():
        return {'ok': False,
                'reason': 'the release archive is missing backend/ — refusing to swap.'}
    deps_changed = _requirements_changed(app_root, root)

    # Move out of any directory being replaced before the swap: after an in-app
    # restart the process CWD is backend/, and Windows refuses to move the CWD or
    # a directory in use as CWD. Pinning CWD to the app root sidesteps that.
    try:
        os.chdir(str(root))
    except OSError:
        pass
    try:
        apply_release_files(app_root, root, backup)
    except Exception as exc:
        return {'ok': False,
                'reason': f'installing the new files failed and was rolled back: {exc}',
                'backup': str(backup)}
    return {'ok': True, 'changed': True, 'from': APP_VERSION, 'to': latest,
            'deps_changed': deps_changed, 'backup': str(backup)}


def _download_file(url, dest, timeout=300) -> None:
    """Stream a URL to `dest` (chunked so a large ZIP never loads fully into RAM)."""
    import requests
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)


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
