"""One-click "Back up everything": a single master archive that bundles every
dataset's portable backup plus a secrets-free copy of the app config — and the
restore that rebuilds them on a fresh install.

The master archive is a plain ZIP:

    manifest.json               # this format's manifest (FULL_BACKUP_FORMAT/VERSION)
    config.json                 # secrets-free app config (see export_safe_config)
    datasets/<id>-<slug>.zip    # one per-dataset backup, the face_dataset_service
                                #   ``lds-dataset-backup`` format, STORED (already
                                #   compressed — never re-deflated)

WHY IT IS SAFE ON SECRETS
-------------------------
API keys, the HF token and the scraping credentials live in ``.env``
(``cfg.SECRET_KEYS``), a *different file* from ``config.json`` — so
``cfg.load_config()`` structurally cannot carry them. The only credential that
lives in config.json is the LAN ``server.access_token``; ``export_safe_config``
blanks it. ``test_full_backup`` proves the produced archive bytes contain no
secret value.

WHY IT IS A LOGICAL BACKUP, NOT A DUMP
--------------------------------------
It archives datasets (through their portable per-dataset format) and the config,
nothing else — never ``data/envs``, the trash, or the raw ``studio.db``. A
dataset that can't be read cleanly (files mid-write during generation) is
skipped and reported, never aborting the whole run.

Both ``build_full_backup`` and ``restore_full_backup`` are synchronous and
directly testable; the thread wrappers (start_backup/start_restore/status) add
only the in-process progress bookkeeping the routes poll — the same shape as
``setup_installer``.
"""
import copy
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from typing import Callable, Optional

from .. import config as cfg
from ..extensions import db
from . import face_dataset_service as svc

logger = logging.getLogger(__name__)

FULL_BACKUP_FORMAT = 'lds-full-backup'
FULL_BACKUP_VERSION = 1

_MANIFEST_NAME = 'manifest.json'
_CONFIG_NAME = 'config.json'
_DATASETS_PREFIX = 'datasets/'

# Config-level credentials to strip. SECRET_KEYS never reach config.json (they
# live in .env), so this is only the LAN access token, which does.
_SENSITIVE_CONFIG_PATHS = (('server', 'access_token'),)

# Free-space margin: the master archive is ~the sum of the datasets' image bytes
# (STORED, no recompression) and each per-dataset backup is staged to one temp
# file at a time, so we need room for the archive plus a headroom cushion.
_DISK_SAFETY_BYTES = 256 * 1024 * 1024
_DISK_HEADROOM_FACTOR = 1.10

# A restore reuses the per-dataset import cap; a master carrying more entries than
# this is almost certainly not one of ours (or is corrupt) — reject before looping.
_MAX_DATASET_ENTRIES = 5000


class AlreadyRunning(Exception):
    """A backup/restore of the same kind is already in flight."""


class DiskSpaceError(Exception):
    """Not enough free space on the data volume to write the archive."""


# ---------------------------------------------------------------------------
# Config export / import (secrets-free)
# ---------------------------------------------------------------------------

def _blank_path(conf: dict, path) -> None:
    node = conf
    for key in path[:-1]:
        if not isinstance(node, dict):
            return
        node = node.get(key)
    if isinstance(node, dict) and node.get(path[-1]):
        node[path[-1]] = ''


def _drop_path(conf: dict, path) -> None:
    node = conf
    for key in path[:-1]:
        if not isinstance(node, dict):
            return
        node = node.get(key)
    if isinstance(node, dict):
        node.pop(path[-1], None)


def export_safe_config() -> dict:
    """The full merged config with every config-borne credential blanked. Secrets
    proper (SECRET_KEYS) are in .env and never appear here in the first place."""
    conf = cfg.load_config()          # already a deep copy
    for path in _SENSITIVE_CONFIG_PATHS:
        _blank_path(conf, path)
    return conf


def _sanitize_incoming_config(conf) -> dict:
    """Make an archive's config safe to merge into the live one on restore:
    keep only known DEFAULTS sections (drop anything unexpected/secret-named) and
    DROP the sensitive paths entirely — so a non-destructive merge can never
    overwrite an access token the user already set on the new machine."""
    if not isinstance(conf, dict):
        return {}
    conf = copy.deepcopy(conf)
    for path in _SENSITIVE_CONFIG_PATHS:
        _drop_path(conf, path)
    return {k: v for k, v in conf.items()
            if k in cfg.DEFAULTS and isinstance(v, dict)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(value: str) -> str:
    """A short filesystem-safe token from a dataset name, for the entry path."""
    out = re.sub(r'[^A-Za-z0-9._-]+', '-', (value or '').strip()).strip('-')
    return (out or 'dataset')[:60]


def _tree_bytes(root) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def check_disk_space() -> dict:
    """Raise DiskSpaceError when the data volume can't hold the archive.
    Returns {'needed_bytes', 'free_bytes'} on success."""
    root = cfg.dataset_images_root()
    needed = int(_tree_bytes(root) * _DISK_HEADROOM_FACTOR) + _DISK_SAFETY_BYTES
    try:
        free = shutil.disk_usage(str(cfg.backups_dir())).free
    except OSError:
        return {'needed_bytes': needed, 'free_bytes': None}
    if free < needed:
        raise DiskSpaceError(
            f'not enough free disk space: about {needed // (1024 * 1024)} MiB '
            f'needed, {free // (1024 * 1024)} MiB free')
    return {'needed_bytes': needed, 'free_bytes': free}


def _new_temp(suffix: str = '.zip') -> str:
    """A temp path on the backups volume (same volume as the final archive, so
    staging costs no cross-volume copy)."""
    fd, path = tempfile.mkstemp(dir=str(cfg.backups_dir()), suffix=suffix)
    os.close(fd)
    return path


ProgressCb = Optional[Callable[[int, int, Optional[str]], None]]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_full_backup(user_id, out_path: str, *, progress: ProgressCb = None,
                      check_disk: bool = True) -> dict:
    """Write the master archive to ``out_path``. Best-effort per dataset: one that
    can't be read is skipped and reported, never aborting the run. Returns a
    result dict {ok, name, path, size_bytes, datasets_total, datasets_backed_up,
    skipped:[{id,name,reason}], config_included}."""
    datasets = svc.list_datasets(user_id)
    total = len(datasets)
    if check_disk:
        check_disk_space()
    try:
        stats = svc.dataset_list_stats(user_id)
    except Exception:
        stats = {}

    manifest_datasets = []
    skipped = []
    backed_up = 0
    part = out_path + '.part'
    try:
        with zipfile.ZipFile(part, 'w', zipfile.ZIP_DEFLATED) as master:
            master.writestr(_CONFIG_NAME,
                            json.dumps(export_safe_config(), ensure_ascii=False, indent=1))
            for i, ds in enumerate(datasets):
                if progress:
                    progress(i, total, ds.name)
                tmp = None
                try:
                    tmp = _new_temp()
                    with open(tmp, 'wb') as fh:
                        svc.write_backup_zip(user_id, ds.id, fh)
                    entry = f'{_DATASETS_PREFIX}{ds.id}-{_slug(ds.name)}.zip'
                    # STORED: the per-dataset zip is already DEFLATE'd; re-deflating
                    # spends CPU for no gain.
                    master.write(tmp, entry, compress_type=zipfile.ZIP_STORED)
                    manifest_datasets.append({
                        'source_id': ds.id,
                        'name': ds.name,
                        'kind': ds.kind,
                        'trigger_word': ds.trigger_word,
                        'entry': entry,
                        'images': int((stats.get(ds.id) or {}).get('images_total', 0)),
                        'bytes': os.path.getsize(tmp),
                    })
                    backed_up += 1
                except Exception as exc:
                    logger.exception('full backup: dataset #%s (%r) skipped',
                                     ds.id, ds.name)
                    skipped.append({'id': ds.id, 'name': ds.name,
                                    'reason': _friendly(exc)})
                finally:
                    if tmp:
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                if progress:
                    progress(i + 1, total, ds.name)
            manifest = {
                'format': FULL_BACKUP_FORMAT,
                'version': FULL_BACKUP_VERSION,
                'app_version': _app_version(),
                'created_at': int(time.time()),
                'config_included': True,
                'datasets': manifest_datasets,
                'skipped': skipped,
            }
            master.writestr(_MANIFEST_NAME,
                            json.dumps(manifest, ensure_ascii=False, indent=1))
        os.replace(part, out_path)
    except Exception:
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    size = os.path.getsize(out_path)
    logger.info("full backup written: %s (%d dataset(s), %d skipped, %d bytes)",
                os.path.basename(out_path), backed_up, len(skipped), size)
    return {
        'ok': True,
        'name': os.path.basename(out_path),
        'path': out_path,
        'size_bytes': size,
        'datasets_total': total,
        'datasets_backed_up': backed_up,
        'skipped': skipped,
        'config_included': True,
    }


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def detect_backup_format(stream) -> Optional[str]:
    """The 'format' field of a backup zip's manifest, or None when the stream is
    not a readable backup. Leaves the stream rewound for the caller."""
    try:
        stream.seek(0)
    except (OSError, ValueError, AttributeError):
        return None
    fmt = None
    try:
        with zipfile.ZipFile(stream) as z:
            if _MANIFEST_NAME in z.namelist():
                manifest = json.loads(z.read(_MANIFEST_NAME).decode('utf-8'))
                if isinstance(manifest, dict):
                    fmt = manifest.get('format')
    except (zipfile.BadZipFile, ValueError, KeyError, UnicodeError):
        fmt = None
    finally:
        try:
            stream.seek(0)
        except (OSError, ValueError, AttributeError):
            pass
    return fmt


def _dedupe_name(base: str, taken: set) -> str:
    """A library-readable name that doesn't collide with an existing one. Import
    never overwrites (each restore is a new id + new folder); this only keeps the
    list legible — the DB is safe regardless."""
    base = (base or 'Restored dataset').strip() or 'Restored dataset'
    candidate = f'{base} (restored)'
    if candidate.casefold() not in taken:
        return candidate[:100]
    n = 2
    while f'{base} (restored {n})'.casefold() in taken:
        n += 1
    return f'{base} (restored {n})'[:100]


def restore_full_backup(user_id, master_path: str, *,
                        progress: ProgressCb = None) -> dict:
    """Rebuild config + datasets from a master archive. Config merges
    non-destructively (secrets are never present, so nothing entered on the new
    machine is overwritten). Each dataset is imported as a NEW dataset (import
    never merges/overwrites); a name that collides with an existing one gets a
    suffix. Returns {ok, datasets_total, restored, skipped:[{entry,reason}],
    renamed:[{from,to}], config_restored}."""
    with zipfile.ZipFile(master_path) as z:
        names = z.namelist()
        if _MANIFEST_NAME not in names:
            raise ValueError('not a full backup (manifest.json missing)')
        try:
            manifest = json.loads(z.read(_MANIFEST_NAME).decode('utf-8'))
        except (ValueError, UnicodeError):
            raise ValueError('not a full backup (unreadable manifest)')
        if not isinstance(manifest, dict) or manifest.get('format') != FULL_BACKUP_FORMAT:
            raise ValueError('not a full backup')
        version = manifest.get('version')
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError('invalid full-backup version')
        if version > FULL_BACKUP_VERSION:
            raise ValueError('backup made by a newer version of the app - update first')

        config_restored = False
        if _CONFIG_NAME in names:
            try:
                incoming = json.loads(z.read(_CONFIG_NAME).decode('utf-8'))
            except (ValueError, UnicodeError):
                incoming = None
            sanitized = _sanitize_incoming_config(incoming)
            if sanitized:
                cfg.save_config(sanitized)
                config_restored = True

        entries = sorted(n for n in names
                         if n.startswith(_DATASETS_PREFIX) and n.endswith('.zip')
                         and '/' not in n[len(_DATASETS_PREFIX):])
        if len(entries) > _MAX_DATASET_ENTRIES:
            raise ValueError('too many datasets in backup')
        total = len(entries)
        taken = {(d.name or '').strip().casefold()
                 for d in svc.list_datasets(user_id)}
        restored = 0
        skipped = []
        renamed = []
        for i, entry in enumerate(entries):
            if progress:
                progress(i, total, entry)
            tmp = None
            try:
                tmp = _new_temp()
                with z.open(entry) as src, open(tmp, 'wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                with open(tmp, 'rb') as fh:
                    ds = svc.import_backup_zip(user_id, fh)
                name = (ds.name or '').strip()
                if name.casefold() in taken:
                    new_name = _dedupe_name(name, taken)
                    ds.name = new_name
                    db.session.commit()
                    renamed.append({'from': name, 'to': new_name})
                    name = new_name
                taken.add(name.casefold())
                restored += 1
            except Exception as exc:
                logger.exception('full restore: entry %r skipped', entry)
                skipped.append({'entry': entry, 'reason': _friendly(exc)})
            finally:
                if tmp:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            if progress:
                progress(i + 1, total, entry)
    logger.info('full backup restored: %d dataset(s), %d skipped, config=%s',
                restored, len(skipped), config_restored)
    return {
        'ok': True,
        'datasets_total': total,
        'restored': restored,
        'skipped': skipped,
        'renamed': renamed,
        'config_restored': config_restored,
    }


# ---------------------------------------------------------------------------
# Open the backups folder in the host file explorer
# ---------------------------------------------------------------------------

def open_backups_folder() -> str:
    path = str(cfg.backups_dir())
    if os.name == 'nt':
        os.startfile(path)                                   # noqa: S606 (local app)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])
    logger.info('opened backups folder: %s', path)
    return path


def resolve_backup_file(name: str) -> Optional[str]:
    """Absolute path of a produced archive, or None. ``name`` must be a plain
    basename inside the backups dir (no separators/traversal) that actually exists."""
    if not name or os.path.basename(name) != name or not name.endswith('.zip'):
        return None
    path = cfg.backups_dir() / name
    return str(path) if path.is_file() else None


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _app_version() -> str:
    try:
        from ..version import APP_VERSION
        return APP_VERSION
    except Exception:
        return ''


def _friendly(exc: Exception) -> str:
    msg = str(exc).strip()
    return cfg_redact(msg) if msg else exc.__class__.__name__


def cfg_redact(msg: str) -> str:
    """Keep any surfaced reason paste-safe (a dataset error could cite a path)."""
    try:
        from ..utils.redact import redact_user_paths
        return redact_user_paths(msg)[:300]
    except Exception:
        return msg[:300]


# ---------------------------------------------------------------------------
# Background-job wrappers (poll shape mirrors setup_installer)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_runs = {'backup': None, 'restore': None}


def _new_state() -> dict:
    return {'state': 'running', 'done': 0, 'total': 0, 'current': None,
            'result': None, 'error': None, 'started_at': int(time.time())}


def _set_progress(kind: str, done: int, total: int, current) -> None:
    st = _runs.get(kind)
    if st is None:
        return
    st['done'], st['total'], st['current'] = done, total, current


def status(kind: str) -> dict:
    with _lock:
        st = _runs.get(kind)
        if st is None:
            return {'state': 'idle'}
        return copy.deepcopy(st)


def is_running(kind: str) -> bool:
    with _lock:
        st = _runs.get(kind)
        return bool(st and st['state'] == 'running')


def start_backup(app, user_id) -> None:
    with _lock:
        st = _runs.get('backup')
        if st and st['state'] == 'running':
            raise AlreadyRunning('a backup is already running')
        _runs['backup'] = _new_state()
    threading.Thread(target=_run_backup, args=(app, user_id),
                     daemon=True, name='full-backup').start()


def _run_backup(app, user_id) -> None:
    with app.app_context():
        try:
            name = f'lds-full-backup-{time.strftime("%Y%m%d-%H%M%S")}.zip'
            out = str(cfg.backups_dir() / name)
            result = build_full_backup(
                user_id, out,
                progress=lambda d, t, c: _set_progress('backup', d, t, c))
            with _lock:
                _runs['backup'].update(state='done', result=result,
                                       done=result['datasets_total'],
                                       total=result['datasets_total'])
        except Exception as exc:
            logger.exception('full backup job failed')
            with _lock:
                if _runs.get('backup'):
                    _runs['backup'].update(state='error', error=_friendly(exc))


def start_restore(app, user_id, master_path: str) -> None:
    with _lock:
        st = _runs.get('restore')
        if st and st['state'] == 'running':
            raise AlreadyRunning('a restore is already running')
        _runs['restore'] = _new_state()
    threading.Thread(target=_run_restore, args=(app, user_id, master_path),
                     daemon=True, name='full-restore').start()


def _run_restore(app, user_id, master_path: str) -> None:
    with app.app_context():
        try:
            result = restore_full_backup(
                user_id, master_path,
                progress=lambda d, t, c: _set_progress('restore', d, t, c))
            with _lock:
                _runs['restore'].update(state='done', result=result,
                                        done=result['datasets_total'],
                                        total=result['datasets_total'])
        except Exception as exc:
            logger.exception('full restore job failed')
            with _lock:
                if _runs.get('restore'):
                    _runs['restore'].update(state='error', error=_friendly(exc))
        finally:
            try:
                os.remove(master_path)
            except OSError:
                pass
