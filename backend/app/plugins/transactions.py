"""Durable package cohorts. Publication and boot health are separate phases.

Only a single nonterminal cohort may exist for a plugin root. The journal is
written before changing live state; its SQLite commit marker is authoritative
after registration. Recovery never restores a shared database file.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import time
import uuid

from .. import config as cfg
from . import official
from .install import ArchiveError, extract_plugin_zip
from .manifest import load_manifest

TERMINAL = {'committed', 'rolled_back'}
PHASES = {'preparing', 'prepared', 'applying', 'awaiting_health', 'rolling_back'} | TERMINAL
_LOCK = threading.RLock()


class PluginBootRollback(RuntimeError):
    """Boot must exit before serving after rollback of newly imported code."""

    exit_code = 75


def _storage():
    from . import storage
    return storage


def _folder(root, txid):
    if not isinstance(txid, str) or len(txid) != 32 or any(c not in '0123456789abcdef' for c in txid):
        raise _storage().StorageError('Invalid plugin transaction identifier.')
    return _storage().managed_path(root, '.transactions', txid)


def _save(root, transaction):
    _storage()._write_json(_folder(root, transaction['id']) / 'transaction.json', transaction)


@contextmanager
def admission(root):
    """Serialize prepare/apply across local threads and OS processes, fail fast."""
    root = Path(root)
    with _LOCK:
        lock = _storage().managed_path(root, '.transactions', 'admission.lock')
        lock.parent.mkdir(parents=True, exist_ok=True)
        if lock.exists() and (not lock.is_file() or lock.stat().st_nlink != 1):
            raise _storage().StorageError('Linked plugin transaction metadata cannot be managed.')
        with lock.open('a+b') as stream:
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise _storage().StorageError('Another process is preparing or applying a plugin transaction.') from exc
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def journals(root):
    folder = _storage().managed_path(root, '.transactions')
    if not folder.exists():
        return []
    result = []
    for child in sorted(folder.iterdir()):
        if not child.is_dir():
            continue
        path = _folder(root, child.name) / 'transaction.json'
        if not path.exists():
            continue
        try:
            if path.stat().st_nlink != 1:
                raise ValueError('linked journal')
            value = json.loads(path.read_text(encoding='utf-8'))
            if (not isinstance(value, dict) or value.get('id') != child.name
                    or value.get('phase') not in PHASES or not isinstance(value.get('items'), list)):
                raise ValueError('invalid journal')
            ids = set()
            for item in value['items']:
                _storage()._id(item['id'])
                if (item['id'] in ids or item.get('action') not in {'install', 'update', 'remove', 'enable'}
                        or type(item.get('desired_enabled')) is not bool):
                    raise ValueError('invalid member')
                ids.add(item['id'])
            result.append(value)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise _storage().StorageError('A plugin transaction journal is invalid; recovery requires attention.') from exc
    if sum(tx['phase'] not in TERMINAL for tx in result) > 1:
        raise _storage().StorageError('Multiple unfinished plugin transactions require recovery; no group was applied.')
    return result


def active(root):
    return next((tx for tx in journals(root) if tx['phase'] not in TERMINAL), None)


def pending(root):
    try:
        transaction = active(root)
        if transaction is None:
            return {}, []
        return {item['id']: {**item, 'token': transaction['id'], 'transaction_id': transaction['id'],
                             'transaction_phase': transaction['phase']}
                for item in transaction['items']}, []
    except (OSError, ValueError) as exc:
        return {}, [{'id': 'plugins', 'error': str(exc)}]


def _digest(path):
    return hashlib.sha256((path / 'plugin.json').read_bytes()).hexdigest()


def _sync_tree(path, *, package=False):
    """Flush copied bytes before a journal claims recovery material is ready."""
    entries = []
    for directory, dirs, files in os.walk(path, followlinks=False):
        directory = Path(directory)
        for name in [*dirs, *files]:
            child = directory / name
            if child.is_symlink() or child.resolve() != child:
                if package:
                    raise _storage().StorageError('A prepared package contains a linked path.')
                continue
            if child.is_file():
                if package and child.stat().st_nlink != 1:
                    raise _storage().StorageError('A prepared package contains a linked file.')
                digest = hashlib.sha256()
                mode = child.stat().st_mode
                readonly = os.name == 'nt' and not (mode & stat.S_IWRITE)
                if readonly:
                    child.chmod(mode | stat.S_IWRITE)
                try:
                    # Windows FlushFileBuffers requires a writable handle even
                    # though this pass changes no bytes. Restore copied modes.
                    with child.open('r+b' if os.name == 'nt' else 'rb') as stream:
                        if package:
                            while chunk := stream.read(1024 * 1024):
                                digest.update(chunk)
                        os.fsync(stream.fileno())
                finally:
                    if readonly:
                        child.chmod(mode)
                if package:
                    entries.append((child.relative_to(path).as_posix(), digest.hexdigest()))
        if os.name != 'nt':
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()


def _installed(root, plugin_id):
    path = _storage().managed_path(root, plugin_id)
    if path.is_dir():
        return load_manifest(path, official=bool(official.installed_provenance(plugin_id, path)))
    return None


def _check_schema(previous, candidate, enabled):
    old = (previous.contract if previous else {}).get('data_schema', 1)
    new = candidate.contract.get('data_schema', 1)
    if previous and new < old:
        raise _storage().StorageError('A package cannot downgrade its installed data_schema. Reinstall a compatible release.')
    if previous and new != old and not enabled:
        raise _storage().StorageError('Turn on the plugin before applying an update that migrates its data schema.')


def _snapshot_metadata(plugin_id):
    result = {}
    if official.is_official_id(plugin_id):
        for key, path in (('receipt', official.receipt_path(plugin_id)), ('retired', official.retired_path(plugin_id))):
            if path.exists():
                if not path.is_file() or path.stat().st_nlink != 1:
                    raise _storage().StorageError('Linked plugin provenance cannot be managed.')
                result[key] = base64.b64encode(path.read_bytes()).decode('ascii')
            else:
                result[key] = None
    path = cfg._config_path()
    raw = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    enabled = (raw.get('plugins') or {}).get('enabled') or {}
    result['enabled_present'] = plugin_id in enabled
    result['enabled_value'] = enabled.get(plugin_id)
    return result


def prepare_batch(root, items, *, preflight=None):
    """Extract every package before publishing one plan; never change live code."""
    root = Path(root)
    items = list(items)
    if not items or len(items) > 64:
        raise _storage().StorageError('A plugin transaction must contain between 1 and 64 members.')
    with admission(root):
        legacy, errors = _storage()._legacy_pending(root)
        if active(root) is not None or legacy or errors:
            raise _storage().StorageError('Apply or recover the pending plugin change before preparing another group.')
        if preflight is not None:
            # Trusted host-only validation, before any journal or extraction.
            # Keep it inside the same cross-process admission as preparation.
            preflight()
        tx = {'id': uuid.uuid4().hex, 'format': 1, 'phase': 'preparing', 'created_at': time.time(), 'items': []}
        seen = set()
        # Construct and validate the whole accepted plan before extraction.
        for accepted in items:
            manifest = accepted.get('manifest')
            plugin_id = manifest.id if manifest is not None else accepted.get('id')
            _storage()._id(plugin_id)
            if plugin_id in seen:
                raise _storage().StorageError('A plugin transaction contains duplicate members.')
            seen.add(plugin_id)
            enabled = accepted.get('desired_enabled', False)
            if type(enabled) is not bool:
                raise _storage().StorageError('Transaction enablement must be a boolean.')
            removing, enabling = accepted.get('action') == 'remove', accepted.get('enable_only') is True
            previous = _installed(root, plugin_id)
            if removing:
                if previous is None:
                    raise _storage().StorageError('The plugin to remove is not installed.')
                action = 'remove'
            elif enabling:
                if manifest is None or not enabled:
                    raise _storage().StorageError('An enable-only member requires its installed manifest.')
                action = 'enable'
            else:
                if manifest is None or manifest.bundled:
                    raise _storage().StorageError('A transaction requires a distributable plugin manifest.')
                if bool(accepted.get('replacing')) != bool(previous):
                    raise _storage().StorageError('The installed plugin changed while its plan was being prepared.')
                _check_schema(previous, manifest, enabled)
                action = 'update' if previous else 'install'
            item = {'id': plugin_id, 'action': action, 'desired_enabled': enabled,
                    'manifest': manifest.summary() if manifest else previous.summary(),
                    'before': _snapshot_metadata(plugin_id), 'had_code': previous is not None}
            if previous is not None:
                item['before_code_sha256'] = _digest(Path(previous.dir))
                item['before_schema'] = previous.contract.get('data_schema', 1)
            if removing:
                keep = accepted.get('keep_data', True)
                if type(keep) is not bool:
                    raise _storage().StorageError('keep_data must be a boolean.')
                item['keep_data'] = keep
            if enabling:
                # The location is used only for validation; live code is never
                # moved for an enable-only dependency (including bundled code).
                item['enable_digest'] = _digest(Path(manifest.dir))
                item['enable_bundled'] = manifest.bundled
            if manifest is not None and manifest.official:
                item['provenance'] = accepted.get('provenance')
            tx['items'].append(item)
        _save(root, tx)
        try:
            for accepted, item in zip(items, tx['items']):
                if item['action'] in {'enable', 'remove'}:
                    continue
                package = _storage().managed_path(_folder(root, tx['id']), 'items', item['id'], 'package')
                extract_plugin_zip(accepted['archive'], package, accepted.get('prefix', ''))
                if (package / 'data').exists():
                    raise ArchiveError('A plugin archive must not include a data folder; use ctx.data_dir.')
                trusted = official.valid_provenance(item.get('provenance'), item['id'], package)
                actual = load_manifest(package, official=trusted)
                expected = accepted['manifest']
                if (actual.summary() != expected.summary() or actual.python_package != expected.python_package
                        or actual.owns != expected.owns or actual.bundled):
                    raise _storage().StorageError('The extracted plugin does not match its accepted manifest.')
                if expected.official and not trusted:
                    raise _storage().StorageError('An official plugin needs verified installation provenance.')
                item['manifest_sha256'] = _digest(package)
                item['files_sha256'] = _sync_tree(package, package=True)
            tx['phase'] = 'prepared'
            _save(root, tx)
        except BaseException:
            # A process death leaves preparing, which recovery can abort. Ordinary
            # failures publish an abort record; staged material is never live.
            tx['phase'] = 'rolled_back'
            tx['reason'] = 'The packages could not be prepared. Installed code and data were not changed.'
            _save(root, tx)
            raise
        return {'id': tx['id']}


def set_enabled(root, transaction_id, plugin_id, enabled):
    if type(enabled) is not bool:
        raise _storage().StorageError('Transaction enablement must be a boolean.')
    with admission(root):
        tx = active(root)
        if tx is None or tx['id'] != transaction_id or tx['phase'] != 'prepared':
            raise _storage().StorageError('This transaction cannot be changed while it is applying.')
        for item in tx['items']:
            if item['id'] == plugin_id:
                if item['action'] == 'remove':
                    raise _storage().StorageError('Apply the pending removal before reinstalling this plugin.')
                item['desired_enabled'] = enabled
                _save(root, tx)
                return
        raise _storage().StorageError('The plugin is not part of this transaction.')


def _member(root, tx, item):
    return _storage().managed_path(_folder(root, tx['id']), 'items', item['id'])


def _rename(source, destination):
    """Publish a prepared directory and flush its directory entry too."""
    if os.name == 'nt':
        import ctypes
        move = ctypes.WinDLL('kernel32', use_last_error=True).MoveFileExW
        move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong]
        move.restype = ctypes.c_int
        # MOVEFILE_WRITE_THROUGH; no replace-existing or cross-volume fallback.
        if not move(str(source), str(destination), 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        source.rename(destination)
        for directory in {source.parent, destination.parent}:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _remove_tree(path):
    # Every caller obtained this exact path through managed_path. rmtree does
    # not traverse interior links; a link at the managed root was rejected.
    if path.exists():
        def clear_readonly(function, name, error):
            child = Path(name)
            if (os.name != 'nt' or not isinstance(error[1], PermissionError)
                    or child.is_symlink() or child.resolve() != child
                    or (child != path and path not in child.parents)
                    or child.stat().st_nlink != 1):
                raise error[1]
            child.chmod(child.stat().st_mode | stat.S_IWRITE)
            function(name)
        shutil.rmtree(path, onerror=clear_readonly)


def snapshot(root, tx):
    """Freeze the last pre-boot data, after the old process has stopped."""
    if tx.get('snapshot_ready'):
        return
    for item in tx['items']:
        live = _storage().managed_path(root, item['id'])
        if item['had_code'] and (not live.exists() or _digest(live) != item['before_code_sha256']):
            raise _storage().StorageError('The installed package changed after preparation; no plan was applied.')
        data = _storage().migrate_data(root, item['id'])
        item['had_data'] = data.exists()
        backup = _storage().managed_path(_member(root, tx, item), 'data-before')
        staging = _storage().managed_path(_member(root, tx, item), 'data-copying')
        if item['had_data'] and not backup.exists():
            _remove_tree(staging)
            shutil.copytree(data, staging, symlinks=True)
            _sync_tree(staging)
            _rename(staging, backup)
    tx['snapshot_ready'] = True
    _save(root, tx)


def publish(root, tx):
    for item in tx['items']:
        if item['action'] not in {'enable', 'remove'}:
            package = _storage().managed_path(_member(root, tx, item), 'package')
            if _digest(package) != item['manifest_sha256']:
                raise _storage().StorageError('The prepared manifest changed before publication.')
            if item.get('files_sha256') and _sync_tree(package, package=True) != item['files_sha256']:
                raise _storage().StorageError('The prepared package bytes changed before publication.')
    for item in tx['items']:
        pid = item['id']
        live = _storage().managed_path(root, pid)
        folder = _member(root, tx, item)
        previous = _storage().managed_path(folder, 'previous')
        package = _storage().managed_path(folder, 'package')
        folder.mkdir(parents=True, exist_ok=True)
        if item['action'] == 'enable':
            if item.get('enable_bundled'):
                from .loader import bundled_dir
                candidate = _storage().managed_path(bundled_dir(), pid)
            else:
                candidate = live
            if _digest(candidate) != item['enable_digest']:
                raise _storage().StorageError('An installed dependency changed after preparation.')
        else:
            if live.exists():
                if previous.exists():
                    raise _storage().StorageError('Both current and recovery packages exist; nothing was overwritten.')
                _rename(live, previous)
            if item['action'] != 'remove':
                if _digest(package) != item['manifest_sha256']:
                    raise _storage().StorageError('The prepared manifest changed before publication.')
                _rename(package, live)
                if item.get('provenance'):
                    official.record_install(pid, live, item['provenance'])
        if item['action'] == 'remove':
            official.forget_install(pid)
            if not item.get('keep_data', True):
                _remove_tree(_storage().data_dir(pid))
        cfg.save_config({'plugins': {'enabled': {pid: None if item['action'] == 'remove' else item['desired_enabled']}}})


def _restore_metadata(item):
    pid, before = item['id'], item['before']
    if official.is_official_id(pid):
        for key, path in (('receipt', official.receipt_path(pid)), ('retired', official.retired_path(pid))):
            value = before.get(key)
            if value is None:
                path.unlink(missing_ok=True)
            else:
                value = base64.b64decode(value, validate=True)
                json.loads(value)  # Reject damaged recovery evidence before publication.
                _storage()._write_bytes(path, value)
    with cfg._lock:
        path = cfg._config_path()
        value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        flags = value.setdefault('plugins', {}).setdefault('enabled', {})
        if before['enabled_present']:
            flags[pid] = before['enabled_value']
        else:
            flags.pop(pid, None)
        _storage()._write_json(path, value)
        cfg._cache = None


def _restore_data(root, tx, item):
    data = _storage().data_dir(item['id'])
    if item.get('had_data'):
        backup = _storage().managed_path(_member(root, tx, item), 'data-before')
        if not backup.is_dir():
            raise _storage().StorageError('Plugin data recovery material is missing; no data was overwritten.')
        local = _storage().managed_path(cfg.data_dir(), 'plugin-data', '.recovery', tx['id'], item['id'])
        restored = _storage().managed_path(local, 'restored')
        failed = _storage().managed_path(local, 'failed')
        # Replaying after a crash during restoration starts from the immutable
        # pre-boot backup. The copy is published on the data volume atomically.
        _remove_tree(restored)
        shutil.copytree(backup, restored, symlinks=True)
        _sync_tree(restored)
        if data.exists():
            _remove_tree(failed)
            _rename(data, failed)
        _rename(restored, data)
    else:
        _remove_tree(data)


def rollback(root, tx):
    if not tx.get('snapshot_ready'):
        tx['phase'] = 'rolled_back'
        _save(root, tx)
        return
    tx['phase'] = 'rolling_back'
    _save(root, tx)
    for item in reversed(tx['items']):
        live = _storage().managed_path(root, item['id'])
        folder = _member(root, tx, item)
        previous = _storage().managed_path(folder, 'previous')
        failed = _storage().managed_path(folder, 'failed')
        if previous.exists():
            if live.exists():
                if failed.exists():
                    raise _storage().StorageError('Plugin recovery found two replacement packages; no code was overwritten.')
                _rename(live, failed)
            _rename(previous, live)
        elif not item['had_code'] and item['action'] != 'enable' and live.exists():
            if _digest(live) != item.get('manifest_sha256') or failed.exists():
                raise _storage().StorageError('Plugin recovery found unexpected installed code; no code was overwritten.')
            _rename(live, failed)
        _restore_data(root, tx, item)
        _restore_metadata(item)
    tx['phase'] = 'rolled_back'
    _save(root, tx)


def purge_committed(root):
    """Delete the recovery material of committed cohorts; their journals stay for history.

    A committed cohort never rolls back, so its previous package, pre-boot data
    snapshot and staging copies only cost disk space: a whole plugin data folder
    per update. Unfinished and rolled-back cohorts keep everything."""
    removed = 0
    with admission(root):
        for tx in journals(root):
            if tx['phase'] != 'committed':
                continue
            items = _storage().managed_path(_folder(root, tx['id']), 'items')
            if items.exists():
                _remove_tree(items)
                removed += 1
    return removed


def history(root, limit=10):
    """A bounded public projection, never expose provenance or machine paths."""
    from .loader import _paste_safe
    return [{'id': tx['id'], 'phase': tx['phase'], 'created_at': tx.get('created_at'),
             'plugins': [{'id': item['id'], 'action': item['action'],
                          'version': item.get('manifest', {}).get('version')} for item in tx['items']],
             'reason': _paste_safe(tx.get('reason', ''))}
            for tx in sorted(journals(root), key=lambda value: value.get('created_at', 0), reverse=True)[:max(1, min(limit, 10))]]
