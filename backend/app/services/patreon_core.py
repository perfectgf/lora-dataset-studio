"""LDS core updates through the Patreon channel (PATREON-PROTOCOL v1, bridge).

The Patreon catalog is a TUF repository (plugins/store/patreon.py). Its signed
``core-releases.json`` lists core releases. Every artifact's length and SHA-256
come from the TUF target metadata alone, never from that list or a grant; the
bytes come through a membership-checked download grant (product ``lds_core``).

* Git checkout, including Pinokio: fetch the verified bundle into a private
  ref, then fast-forward only. Local commits or modified tracked files refuse,
  like ``git pull --ff-only``. Pinokio restarts the app from its own launcher.
* ZIP install: the verified archive goes through the existing staged swap with
  its backup and rollback (``updater.apply_zip_update``). The data folder
  records the channel (``updater.patreon_channel``) so that the public ZIP
  update refuses to replace this version unless explicitly forced.
* Docker: never applied from inside the container. The host updater
  (``update-docker.bat patreon``, Windows ZIP installations only) runs
  ``fetch-source`` below in a disposable container of the current image,
  checks the result, then applies the release source archive with its own
  transaction and image rebuild.

In-app installs (Git, Pinokio, ZIP) apply nothing while LDS has work running:
the idle gate of plugin Apply (plugins/restart.py) is frozen before the first
file changes and held until the process exits. The Docker host updater cannot
see that work, so the Updates card and the guide ask members to stop it first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flask import current_app, jsonify

from .. import config as cfg
from ..config import REPO_ROOT
from ..plugins import restart
from ..plugins.storage import managed_path
from ..plugins.store.client import MAX_CATALOG_BYTES, StoreError, StoreNotConfigured, StoreSession
from ..plugins.store.patreon import (VERIFICATION_UNAVAILABLE, MembershipRequired, PatreonClient,
                                     VerificationUnavailable, access_summary, patreon_core_source)
from . import updater

CORE_PRODUCT = 'lds_core'
CORE_RELEASES_TARGET = 'core-releases.json'
CORE_RELEASES_FORMAT = 'lds-core-releases/1'
CORE_MAX_BYTES = 512 * 1024 * 1024
BUNDLE_REF = 'refs/heads/patreon'
INCOMING_REF = 'refs/lds/patreon/incoming'
PUBLIC_BRANCH = 'v2'
STATUS_TTL_SECONDS = 600
IDLE_ACTION = 'installing the LDS update'
# APP_VERSION's own scheme. Compared as numbers (updater.version_newer): a
# string comparison would put 2026.10.11.9 after 2026.10.11.10.
_VERSION = re.compile(r'\d{4}\.\d{2}\.\d{2}(?:\.\d+)?\Z')
_SHA = re.compile(r'[0-9a-f]{40}\Z')

DOCKER_LIMIT = ('This Patreon release has no Docker package: a container cannot replace its own image, '
                'and the host update script needs the release source archive. Your installed version '
                'keeps working.')
# The host updater exists for Windows installations from the ZIP only (it
# refuses a Git checkout), and it cannot see LDS's work: unlike the in-app
# updates, nothing waits for running work before the image rebuild.
DOCKER_HOST_COMMAND = 'update-docker.bat patreon'
DOCKER_HOST_UPDATE = ('LDS cannot update itself inside Docker. For an LDS installation from the Windows ZIP, '
                      f'stop running work first, then run {DOCKER_HOST_COMMAND} in the LDS folder on the '
                      'computer that runs Docker.')
PATREON_REQUIRED = ('This update is available to Patreon members first. Connect Patreon or join, '
                    'then check again.')
AHEAD_OF_RELEASE = ('This installation is ahead of the members release: it already has public changes that '
                    'release does not include. Wait for the next Patreon release; your installed version keeps '
                    'working.')
LOCAL_HISTORY = ('This checkout has local commits or follows another history, so it cannot fast-forward to the '
                 'Patreon release. Your files were not changed.')
MISSING_BASE = ('This update builds on public LDS history that this checkout does not have yet. Check the '
                'connection to the public repository and retry.')
PINOKIO_HOLD = ('An LDS update was installed. Stop the app in Pinokio, click Update, then Start it '
                'to finish.')

# fetch-source output, read by scripts/update-docker-gpu.ps1: the manifest is
# written last, so its presence means the archive beside it is complete.
SOURCE_FORMAT = 'lds-core-source/1'
SOURCE_ARCHIVE = 'lds-core-source.zip'
SOURCE_MANIFEST = 'manifest.json'
# Exit codes of `python -m app.services.patreon_core fetch-source` (argparse
# usage errors exit with 2).
EXIT_FETCHED, EXIT_FAILED, EXIT_CURRENT, EXIT_MEMBERSHIP = 0, 1, 3, 4
_CREDENTIAL_MAX_BYTES = 16384


class CoreChannelError(StoreError):
    """A user-facing reason the Patreon core update stops; no file changed."""


@dataclass(frozen=True)
class CoreRelease:
    version: str
    commit: str
    base: str
    published_at: str
    notes: tuple
    bundle: str
    archive: str
    # Full source tree for the Docker host updater; absent from older releases.
    source: str | None = None

    def payload(self):
        return {'version': self.version, 'published_at': self.published_at,
                'notes': list(self.notes), 'commit': self.commit[:12]}


@dataclass(frozen=True)
class Channel:
    release: CoreRelease
    bundle_info: object
    archive_info: object
    source_info: object = None


def parse_core_releases(value):
    """Strict ``lds-core-releases/1``; newest first. Never carries lengths or hashes.

    ``source`` is optional; when present it must name the release's own
    ``core/lds-core-<version>.source.zip``."""
    try:
        if not isinstance(value, dict) or value.get('format') != CORE_RELEASES_FORMAT:
            raise ValueError
        releases = value.get('releases')
        if not isinstance(releases, list) or len(releases) > 500:
            raise ValueError
        result, versions = [], set()
        for item in releases:
            if not isinstance(item, dict):
                raise ValueError
            version, commit, base = item.get('version'), item.get('commit'), item.get('base')
            if (not isinstance(version, str) or not _VERSION.fullmatch(version) or version in versions
                    or not isinstance(commit, str) or not _SHA.fullmatch(commit)
                    or not isinstance(base, str) or not _SHA.fullmatch(base) or base == commit):
                raise ValueError
            published = item.get('published_at')
            parsed = datetime.fromisoformat(published.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                raise ValueError
            notes = item.get('notes', [])
            if (not isinstance(notes, list) or len(notes) > 50
                    or any(not isinstance(note, str) or len(note) > 1000 for note in notes)):
                raise ValueError
            if (item.get('bundle') != f'core/lds-core-{version}.bundle'
                    or item.get('archive') != f'core/lds-core-{version}.zip'
                    or ('source' in item and item['source'] != f'core/lds-core-{version}.source.zip')):
                raise ValueError
            versions.add(version)
            result.append(CoreRelease(version, commit, base, published, tuple(notes),
                                      item['bundle'], item['archive'], item.get('source')))
    except (ValueError, TypeError, AttributeError, OverflowError):
        raise StoreError('The Patreon core release list is invalid.') from None
    return sorted(result, key=lambda release: updater.version_key(release.version), reverse=True)


def _read_releases(session):
    path, _ = session.target(CORE_RELEASES_TARGET, maximum=MAX_CATALOG_BYTES)
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise StoreError('The Patreon core release list is invalid.') from None
    return parse_core_releases(value)


def read_channel():
    """Latest Patreon core release and its TUF target infos, from one fresh session."""
    with StoreSession(patreon_core_source()) as session:
        releases = _read_releases(session)
        if not releases:
            return None
        latest = releases[0]
        source = session.target_info(latest.source, maximum=CORE_MAX_BYTES) if latest.source else None
        return Channel(latest, session.target_info(latest.bundle, maximum=CORE_MAX_BYTES),
                       session.target_info(latest.archive, maximum=CORE_MAX_BYTES), source)


def install_mode(root=None):
    root = Path(root or REPO_ROOT)
    if updater.is_docker_runtime():
        return 'docker'
    if updater.is_git_checkout(root):
        return 'pinokio' if updater.is_pinokio_runtime() else 'git'
    return 'zip'


def _git(root, *args):
    try:
        return updater._git(root, *args)
    except FileNotFoundError:
        raise CoreChannelError('git is not installed / not on PATH; install Git to update this checkout.') from None
    except subprocess.SubprocessError:
        raise CoreChannelError('A git command timed out. Your files were not changed.') from None


def _head(root):
    return (_git(root, 'rev-parse', 'HEAD').stdout or '').strip()


def _has_commit(root, sha):
    return _git(root, 'cat-file', '-e', sha + '^{commit}').returncode == 0


def _dirty(root):
    return bool((_git(root, 'status', '--porcelain', '--untracked-files=no').stdout or '').strip())


def _git_has_release(root, release):
    """True when HEAD already contains the release commit."""
    head = _head(root)
    return head == release.commit or (_has_commit(root, release.commit)
                                      and updater._is_ancestor(root, release.commit, 'HEAD'))


def _fast_forward_blocker(root, release, *, fetch=False):
    """Why HEAD cannot fast-forward to the release, decided before any
    download; None when it can. It can when HEAD is the release's public base
    or behind it, or exactly the Patreon release installed here, which the next
    one builds on. A base this checkout lacks is fetched with ``fetch``;
    without it (status) the apply decides, after fetching."""
    if _head(root) == updater.patreon_marker(root):
        return None
    target = release.commit if _has_commit(root, release.commit) else release.base
    if target == release.base and not _has_commit(root, release.base):
        if not fetch:
            return None
        _git(root, 'fetch', '--quiet', '--no-tags', 'origin', PUBLIC_BRANCH)
        if not _has_commit(root, release.base):
            raise CoreChannelError(MISSING_BASE)
    if updater._is_ancestor(root, 'HEAD', target):
        return None
    # Public commits only: the next release will carry them. Otherwise the
    # checkout holds work of its own.
    public = f'refs/remotes/origin/{PUBLIC_BRANCH}'
    return AHEAD_OF_RELEASE if updater._is_ancestor(root, 'HEAD', public) else LOCAL_HISTORY


_status_cache = {'ts': 0.0, 'data': None}
_members_cache = {'ts': 0.0, 'data': None}
_installed = {}
_apply_lock = threading.Lock()


def forget_status():
    _status_cache.update(ts=0.0, data=None)


def members_versions():
    """Published members core versions, newest first, for the Plugins page's
    membership features. Empty when the channel cannot be read, so nothing is
    then presented as reserved; like the status, only a complete answer is
    reused."""
    now = time.time()
    if _members_cache['data'] is not None and now - _members_cache['ts'] < STATUS_TTL_SECONDS:
        return _members_cache['data']
    try:
        with StoreSession(patreon_core_source()) as session:
            versions = tuple(release.version for release in _read_releases(session))
    except (StoreError, OSError, ValueError):
        return ()
    _members_cache.update(ts=now, data=versions)
    return versions


def status(root=None, *, force=False):
    """What the Updates card shows for the Patreon channel. Never raises."""
    now = time.time()
    if not force and _status_cache['data'] is not None and now - _status_cache['ts'] < STATUS_TTL_SECONDS:
        return _with_access(dict(_status_cache['data']))
    root = Path(root or REPO_ROOT)
    from ..version import APP_VERSION
    mode = install_mode(root)
    result = {'status': 'ready', 'message': '', 'install_mode': mode, 'current': APP_VERSION,
              'current_sha': updater.current_sha(root), 'on_channel': False, 'release': None,
              'update_available': False, 'access': None, 'entitled': False, 'can_apply': False,
              'blocked_reason': None, 'host_command': None, 'restart_required': bool(_installed)}
    try:
        if mode in ('git', 'pinokio'):
            marker = updater.patreon_marker(root)
            result['on_channel'] = bool(marker and updater._is_ancestor(root, marker, 'HEAD'))
        else:
            # ZIP: written by apply_zip; Docker: by the host updater, in the
            # lane's data folder that this container mounts as its own.
            result['on_channel'] = updater.patreon_channel() is not None
        channel = read_channel()
        if channel is not None:
            release = channel.release
            result['release'] = release.payload()
            if mode in ('git', 'pinokio'):
                result['update_available'] = not _git_has_release(root, release)
                if result['update_available'] and _dirty(root):
                    result['blocked_reason'] = ('Modified tracked files would be overwritten. Commit or '
                                                'discard them, then check again.')
                elif result['update_available']:
                    result['blocked_reason'] = _fast_forward_blocker(root, release)
            else:
                result['update_available'] = updater.version_newer(release.version, APP_VERSION)
            if mode == 'docker' and result['update_available']:
                # The container never replaces its own image; the host updater
                # applies the release source archive (fetch_source below).
                if release.source:
                    result['host_command'] = DOCKER_HOST_COMMAND
                else:
                    result['blocked_reason'] = DOCKER_LIMIT
        _with_access(result)
    except StoreNotConfigured as exc:
        result.update(status='not_configured', message=str(exc))
    except (StoreError, OSError, ValueError) as exc:
        result.update(status='unavailable', message=str(exc) if isinstance(exc, StoreError)
                      else 'The Patreon update channel could not be checked.')
    # Only a complete answer is reused: a failure is checked again next time.
    if result['status'] == 'ready':
        _status_cache.update(ts=now, data=result)
    return result


def _with_access(result):
    """Membership changes on Patreon, outside LDS (connecting, renewal, cancelling):
    only the signed channel and Git checks are cached, never the membership."""
    try:
        access = access_summary()
    except StoreNotConfigured:
        access = {'status': 'not_configured', 'patreon': None}
    except (StoreError, OSError):
        access = {'status': 'unavailable', 'patreon': None}
    result['access'] = access
    result['entitled'] = bool(access['patreon'] and access['patreon']['status'] == 'active')
    result['restart_required'] = bool(_installed)
    result['can_apply'] = bool(result['update_available'] and result['entitled'] and not result['blocked_reason']
                               and result['install_mode'] != 'docker' and not _installed)
    return result


def _download(release, target, info, *, destination, on_progress=None):
    return PatreonClient().download_target(CORE_PRODUCT, release.version, target, info, destination,
                                           maximum=CORE_MAX_BYTES, on_progress=on_progress)


def _fetch_bundle(root, bundle, release):
    """Verified bundle bytes into a private ref; nothing visible changes yet."""
    heads = _git(root, 'bundle', 'list-heads', str(bundle))
    lines = [line.split() for line in (heads.stdout or '').splitlines() if line.strip()]
    if heads.returncode != 0 or lines != [[release.commit, BUNDLE_REF]]:
        raise CoreChannelError('The Patreon update bundle does not contain the published release. '
                               'Your files were not changed.')
    if not _has_commit(root, release.base):
        # The bundle starts at a public commit. Fetching the public branch is
        # the same network access as an ordinary update check.
        _git(root, 'fetch', '--quiet', '--no-tags', 'origin', PUBLIC_BRANCH)
        if not _has_commit(root, release.base):
            raise CoreChannelError(MISSING_BASE)
    fetch = _git(root, 'fetch', '--quiet', '--no-tags', str(bundle), f'+{BUNDLE_REF}:{INCOMING_REF}')
    fetched = (_git(root, 'rev-parse', '--verify', '--quiet', INCOMING_REF + '^{commit}').stdout or '').strip()
    if fetch.returncode != 0 or fetched != release.commit:
        raise CoreChannelError('git could not read the verified Patreon update. Your files were not changed.')
    if not updater._is_ancestor(root, release.base, release.commit):
        raise CoreChannelError('The Patreon release does not build on its published public base. '
                               'Your files were not changed.')


def _update_ref(root, value):
    args = ('update-ref', updater.PATREON_MARKER_REF, value) if value else ('update-ref', '-d', updater.PATREON_MARKER_REF)
    if _git(root, *args).returncode != 0:
        raise CoreChannelError('The Patreon channel marker could not be written. Your files were not changed.')


def _fast_forward(root, release):
    """Fast-forward only, under the idle gate. Marker before HEAD moves."""
    if _dirty(root):
        raise CoreChannelError('Modified tracked files would be overwritten. Commit or discard them, then retry.')
    if not updater._is_ancestor(root, 'HEAD', release.commit):
        raise CoreChannelError(LOCAL_HISTORY)
    before = _head(root)
    old = updater.patreon_marker(root)
    # A marker already in HEAD's history keeps protecting the new HEAD, which
    # descends from it. Otherwise set the marker first, so that no moment
    # exists where HEAD is ahead on the channel without one.
    preset = old is None or not updater._is_ancestor(root, old, 'HEAD')
    if preset:
        _update_ref(root, release.commit)
    merge = _git(root, 'merge', '--ff-only', release.commit)
    if merge.returncode != 0:
        if preset:
            _update_ref(root, old)
        raise CoreChannelError('git could not fast-forward to the Patreon release (an untracked file may be '
                               'in the way). Your files were not changed.')
    _update_ref(root, release.commit)
    after = _head(root)
    names = (_git(root, 'diff', '--name-only', before, after).stdout or '')
    return {'ok': True, 'changed': before != after, 'from': before[:8], 'to': after[:8],
            'version': release.version,
            'deps_changed': any('requirements' in name for name in names.splitlines())}


def apply_git(root, channel, *, pinokio=False):
    """Download, verify and fetch; then freeze the idle gate and fast-forward.

    Returns ``(payload, gate)``: a non-None gate is frozen and owned by the
    calling thread, which must keep it until the process exits."""
    release = channel.release
    if _dirty(root):
        raise CoreChannelError('Modified tracked files would be overwritten. Commit or discard them, then retry.')
    if _git_has_release(root, release):
        return {'ok': True, 'changed': False, 'version': release.version}, None
    # Decided before the download: a checkout ahead of the release, or with
    # work of its own, is told so instead of after fetching the bundle.
    blocker = _fast_forward_blocker(root, release, fetch=True)
    if blocker:
        raise CoreChannelError(blocker)
    folder = managed_path(cfg.data_dir(), 'plugin-store', 'core')
    bundle = _download(release, release.bundle, channel.bundle_info,
                       destination=folder / (channel.bundle_info.hashes['sha256'] + '.bundle'))
    try:
        _fetch_bundle(root, bundle, release)
    finally:
        Path(bundle).unlink(missing_ok=True)
    if not updater._is_ancestor(root, 'HEAD', release.commit):
        raise CoreChannelError(LOCAL_HISTORY)
    gate = restart.freeze_when_idle(IDLE_ACTION)
    try:
        result = _fast_forward(root, release)
    except BaseException:
        gate.release()
        raise
    if not result['changed']:
        gate.release()
        return result, None
    _installed.update(version=release.version, pinokio=pinokio)
    forget_status()
    return result, gate


def apply_zip(root, channel):
    """Start the async staged swap of the verified archive (packaged install)."""
    from ..version import APP_VERSION
    release, info = channel.release, channel.archive_info
    if not updater.version_newer(release.version, APP_VERSION):
        return {'ok': True, 'changed': False, 'version': release.version}
    # Fail before a long download when work is already running; the worker
    # checks again, holding the gate, right before the first file moves.
    restart.freeze_when_idle(IDLE_ACTION).release()
    app = current_app._get_current_object()

    def download(destination, on_bytes):
        _download(release, release.archive, info, destination=destination, on_progress=on_bytes)

    def reserve():
        return restart.freeze_when_idle(IDLE_ACTION, app=app)

    forget_status()
    # The data folder records the channel before the first file moves, so a
    # public release never replaces this one silently (updater.patreon_channel).
    return updater.start_verified_zip_update(root, {'version': release.version, 'zip_size': info.length},
                                             download=download, reserve=reserve,
                                             channel={'version': release.version, 'commit': release.commit},
                                             expected_version=release.version)


def apply_response(root=None):
    """Body of ``POST /api/update/patreon/apply`` (admin and CSRF checked by the route)."""
    root = Path(root or REPO_ROOT)

    def refuse(reason, code='update_refused'):
        # `error` is what the shared fetch client shows; `reason` matches /update/apply.
        return jsonify(ok=False, code=code, reason=reason, error=reason), 409

    if not _apply_lock.acquire(blocking=False):
        return refuse('An LDS update is already being prepared.')
    try:
        if updater.zip_update_in_progress():
            return refuse('Another update is already in progress.')
        if _installed:
            return refuse('An LDS update is already installed. Restart LDS to finish it.')
        mode = install_mode(root)
        if mode == 'docker':
            return refuse(DOCKER_HOST_UPDATE, 'docker_limit')
        channel = read_channel()
        if channel is None:
            return jsonify(ok=True, changed=False, reason='No Patreon release is published yet.')
        access = access_summary()
        membership = (access['patreon'] or {}).get('status')
        if membership == 'verification_required':
            return refuse(VERIFICATION_UNAVAILABLE, 'verification_required')
        if membership != 'active':
            return refuse(PATREON_REQUIRED, 'patreon_required')
        if mode == 'zip':
            result = apply_zip(root, channel)
            if not result.get('ok'):
                return refuse(result.get('reason') or 'The update could not start.')
            return jsonify(result)
        result, gate = apply_git(root, channel, pinokio=mode == 'pinokio')
        if gate is None:
            return jsonify(result)
        if mode == 'pinokio':
            # Pinokio owns the process: keep admissions closed until the user
            # stops the app there, so no new work runs on mixed code.
            gate.message = PINOKIO_HOLD
            response = jsonify({**result, 'restarting': False, 'restart_required': True,
                                'install_mode': 'pinokio',
                                'instructions': list(updater.PINOKIO_UPDATE_INSTRUCTIONS)})
        else:
            try:
                updater.schedule_restart(install_requirements=bool(result.get('deps_changed')))
                response = jsonify({**result, 'restarting': True})
            except Exception:
                response = jsonify({**result, 'restarting': False, 'restart_required': True,
                                    'reason': 'The update was installed, but the automatic restart could not '
                                              'be scheduled. Restart LDS manually.'})
        # Werkzeug closes the response after flushing it. The gate's locks
        # belong to this thread, which must stay alive until the process exits.
        response.call_on_close(gate.finished.wait)
        return response
    except restart.RestartBlocked as exc:
        return refuse(str(exc), 'update_blocked')
    except MembershipRequired as exc:
        forget_status()
        return refuse(str(exc), 'patreon_required')
    except VerificationUnavailable as exc:
        return refuse(str(exc), 'verification_required')
    except updater.UpdateInProgressError:
        return refuse('Another update is already in progress.')
    except StoreNotConfigured as exc:
        return refuse(str(exc), 'not_configured')
    except (StoreError, OSError, ValueError) as exc:
        return refuse(str(exc) if isinstance(exc, StoreError)
                      else 'The Patreon update could not be prepared. Your files were not changed.')
    finally:
        _apply_lock.release()


# --- Docker: the host updater's disposable fetch -----------------------------

@contextmanager
def _private_data_dir():
    """A writable data folder for one fetch; the installation's own stays as it is."""
    previous = os.environ.get('LDS_DATA_DIR')
    with tempfile.TemporaryDirectory(prefix='lds-patreon-fetch-', ignore_cleanup_errors=True) as work:
        folder = Path(work).resolve()
        os.environ['LDS_DATA_DIR'] = str(folder)
        try:
            yield folder
        finally:
            if previous is None:
                os.environ.pop('LDS_DATA_DIR', None)
            else:
                os.environ['LDS_DATA_DIR'] = previous


def _copy_credential(data, work):
    """Copy only the installation credential out of the read-only data folder.

    Reading it in place would create a lock file and adjust permissions beside
    it, which a read-only mount refuses. No credential means not connected,
    which the membership check reports."""
    folder = PatreonClient().folder
    source = Path(data).joinpath(folder.relative_to(work), 'installation.json')
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    except FileNotFoundError:
        return
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _CREDENTIAL_MAX_BYTES:
            raise CoreChannelError('The saved Patreon connection could not be read. Nothing was changed.')
        value = stream.read(_CREDENTIAL_MAX_BYTES + 1)
    folder.mkdir(parents=True, exist_ok=True)
    copy = os.open(folder / 'installation.json',
                   os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
    with os.fdopen(copy, 'wb') as stream:
        stream.write(value)


def _write_file(path, data):
    """Complete file or none: temporary file, fsync, rename in the same folder."""
    descriptor, temporary = tempfile.mkstemp(prefix='.manifest-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch_source(out, data, *, newer_than=None):
    """Write the latest release's verified source archive, then its manifest.

    Run by the Docker host updater (scripts/update-docker-gpu.ps1, channel
    ``patreon``) in a disposable container of the current image: ``data`` is
    the installation's data folder, mounted read-only, and ``out`` an empty
    host folder, the only writable mount. TUF starts from the root shipped in
    the image; the grant and the length/SHA-256 checks are those of every core
    download. Returns ``{'status': 'current', 'version': latest or None}``
    when nothing newer than ``newer_than`` is published, else the manifest."""
    out = Path(out)
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):
        raise CoreChannelError('The output folder must be an existing, empty folder.')
    if newer_than is not None and not _VERSION.fullmatch(newer_than):
        raise CoreChannelError('The installed version is not a valid LDS version.')
    with _private_data_dir() as work:
        _copy_credential(data, work)
        channel = read_channel()
        if channel is None or (newer_than is not None
                               and not updater.version_newer(channel.release.version, newer_than)):
            return {'status': 'current', 'version': channel.release.version if channel else None}
        release, info = channel.release, channel.source_info
        if info is None:
            raise CoreChannelError(DOCKER_LIMIT)
        membership = (access_summary()['patreon'] or {}).get('status')
        if membership == 'verification_required':
            raise VerificationUnavailable(VERIFICATION_UNAVAILABLE)
        if membership != 'active':
            raise MembershipRequired(PATREON_REQUIRED)
        _download(release, release.source, info, destination=out / SOURCE_ARCHIVE)
    manifest = {'format': SOURCE_FORMAT, 'version': release.version, 'commit': release.commit,
                'archive': SOURCE_ARCHIVE, 'length': info.length, 'sha256': info.hashes['sha256']}
    _write_file(out / SOURCE_MANIFEST, json.dumps(manifest, indent=2).encode('utf-8') + b'\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m app.services.patreon_core',
                                     description='Patreon core channel commands for the Docker host updater.')
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('fetch-source', help='download the verified source archive of the latest release')
    fetch.add_argument('--out', required=True, help='empty folder that receives the archive and manifest.json')
    fetch.add_argument('--data', required=True, help="the installation's data folder, only read")
    fetch.add_argument('--newer-than', help='installed version: a release that is not newer downloads nothing')
    args = parser.parse_args(argv)
    try:
        result = fetch_source(args.out, args.data, newer_than=args.newer_than)
    except (MembershipRequired, VerificationUnavailable) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return EXIT_MEMBERSHIP
    except (StoreError, OSError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, StoreError) else 'The Patreon release could not be fetched.'
        print(f'[ERROR] {reason}', file=sys.stderr)
        return EXIT_FAILED
    if result.get('status') == 'current':
        latest = f' (latest: v{result["version"]})' if result['version'] else ''
        print(f'No newer LDS release on the Patreon channel{latest}.')
        return EXIT_CURRENT
    print(f'Fetched LDS v{result["version"]} ({result["commit"][:12]}), verified against the signed '
          'Patreon catalog.')
    return EXIT_FETCHED


if __name__ == '__main__':
    sys.exit(main())
