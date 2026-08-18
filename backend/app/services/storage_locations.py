"""What lives where on disk — the map behind Settings › Storage.

Two jobs, deliberately kept apart:

  * **Describe.** ``locations()`` answers "which folder holds which category",
    with no disk walk at all. Sizes are a SEPARATE call (``sizes()``), because
    walking a dataset root or a 100 GB run folder takes seconds and must never
    ride a page mount — the same policy the cloud hub's staging sizes already
    follow.
  * **Relocate.** ``validate_target()`` proves a candidate folder is really
    writable BEFORE anything is saved, and ``start_move()`` carries what is
    already there, with progress, as an explicit job. A location change never
    moves files behind the user's back: the caller chooses `move` or `adopt`.

Why this exists at all: the app writes into eight or nine roots, only three of
which were ever visible in Settings. When C: fills up, "put it all on another
drive" was a config.json edit and a manual copy, and the copy silently competed
with a running app.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from .. import config as cfg

logger = logging.getLogger(__name__)


def _frontend_dist() -> Path:
    return cfg.REPO_ROOT / 'frontend' / 'dist'


def _trash_root() -> Path:
    return cfg._data_dir() / 'trash'


def _run_archive_root() -> Path:
    return cfg._data_dir() / 'run_images'


def _aitoolkit_root() -> Path | None:
    root = cfg.get('aitoolkit.dir') or ''
    return Path(root) if root else None


def _hf_home() -> Path | None:
    try:
        return Path(cfg.aitoolkit_path('hf_home'))
    except Exception:
        return None


# key            stable id — stored nowhere, but the frontend and the help
#                registry both name it, so treat it as an alias-bound label.
# section/field  the config key that relocates it (None = not relocatable here).
# holds          one sentence, in the user's terms, about what is inside.
_DEFS = (
    {'key': 'datasets', 'label': 'Dataset images',
     'section': 'paths', 'field': 'dataset_images_root',
     'resolve': cfg.dataset_images_root,
     'holds': 'Every image of every dataset, plus its captions and masks.'},
    {'key': 'banks', 'label': 'Image banks (working data)',
     'section': None, 'field': None,
     'resolve': cfg.banks_root,
     'holds': 'Thumbnails and face embeddings of your banks — never the source images.'},
    {'key': 'bank_sources', 'label': 'Bank source images',
     'section': None, 'field': None,
     'resolve': cfg.bank_sources_root,
     'holds': 'Images copied into a bank by “Import to bank”.'},
    {'key': 'video_banks', 'label': 'Video banks (working data)',
     'section': None, 'field': None,
     'resolve': cfg.video_banks_root,
     'holds': 'Thumbnails of the shots detected in your video banks — never the '
              'source videos, and never the clips themselves.'},
    {'key': 'video_bank_sources', 'label': 'Video bank source files',
     'section': None, 'field': None,
     'resolve': cfg.video_bank_sources_root,
     'holds': 'Videos downloaded by “Scrape the web into a bank”, for the banks '
              'that scrape created. A scrape sent to a bank of your own adds its '
              'clips to that bank’s folder instead, so nothing here is a copy of '
              'your footage.'},
    {'key': 'video_datasets', 'label': 'Video datasets',
     'section': 'paths', 'field': 'video_datasets_dir',
     'resolve': cfg.video_datasets_root,
     'holds': 'The encoded training clips and their captions. This is where the '
              'video lane actually uses disk — a bank stores only timestamps.'},
    {'key': 'cloud_runs', 'label': 'Cloud run staging',
     'section': 'paths', 'field': 'cloud_runs_dir',
     'resolve': cfg.cloud_runs_root,
     'holds': 'Working files of cloud training runs: the exported dataset copy, '
              'sample images and logs. Safe to clean once a run has ended.'},
    {'key': 'checkpoints', 'label': 'Checkpoint store',
     'section': 'paths', 'field': 'checkpoints_dir',
     'resolve': cfg.checkpoints_root,
     'holds': 'The trained .safetensors of your cloud runs. Nothing here is '
              'ever removed by a cleanup.'},
    {'key': 'trash', 'label': 'Trash',
     'section': None, 'field': None,
     'resolve': _trash_root,
     'holds': 'Everything the app “deletes”, until you empty it.'},
    {'key': 'run_archive', 'label': 'Run image archive',
     'section': None, 'field': None,
     'resolve': _run_archive_root,
     'holds': 'Deduplicated copies of the images each run trained on.'},
    {'key': 'backups', 'label': 'Backups',
     'section': None, 'field': None,
     'resolve': cfg.backups_dir,
     'holds': 'Archives written by “Back up everything”.'},
    {'key': 'aitoolkit', 'label': 'ai-toolkit install',
     'section': 'aitoolkit', 'field': 'dir',
     'resolve': _aitoolkit_root,
     'holds': 'The local trainer and the runs it writes.'},
    {'key': 'hf_home', 'label': 'Hugging Face cache',
     'section': 'aitoolkit', 'field': 'hf_home',
     'resolve': _hf_home,
     'holds': 'Base models downloaded for local training — the biggest cache of all.'},
    {'key': 'dist', 'label': 'App build (frontend/dist)',
     'section': None, 'field': None,
     'resolve': _frontend_dist,
     'holds': 'The compiled interface served to your browser. Part of the app, '
              'not your data.'},
)

LOCATION_KEYS = tuple(d['key'] for d in _DEFS)


def _definition(key):
    for d in _DEFS:
        if d['key'] == key:
            return d
    return None


def locations() -> list:
    """Every category, its effective path and whether it can be relocated.

    No `stat`, no walk: this is what the tab renders on arrival. `configured`
    distinguishes "you set this path" from "this is the default under the data
    folder", so a Reset button knows there is something to reset."""
    out = []
    for d in _DEFS:
        try:
            path = d['resolve']()
        except Exception as e:                   # a bad custom path must not 500
            logger.warning('storage: could not resolve %s: %s', d['key'], e)
            path = None
        configured = ''
        if d['section'] and d['field']:
            configured = str(cfg.get(f"{d['section']}.{d['field']}") or '')
        out.append({
            'key': d['key'], 'label': d['label'], 'holds': d['holds'],
            'path': str(path) if path else '',
            'exists': bool(path and os.path.isdir(str(path))),
            'section': d['section'], 'field': d['field'],
            'configured': configured,
            'relocatable': bool(d['section'] and d['field']),
        })
    return out


def sizes(keys=None) -> dict:
    """`{key: bytes}` for the requested categories (all of them when `keys` is
    None). Walks the disk — only ever called from an explicit “Measure” click.
    A category whose folder is missing or unreadable answers 0 rather than
    failing the whole request."""
    from .lora_training import _dir_size
    wanted = LOCATION_KEYS if keys is None else [str(k) for k in keys]
    out = {}
    for key in wanted:
        d = _definition(key)
        if not d:
            continue
        try:
            path = d['resolve']()
            out[key] = _dir_size(str(path)) if path and os.path.isdir(str(path)) else 0
        except Exception as e:
            logger.warning('storage: could not size %s: %s', key, e)
            out[key] = 0
    return out


def free_space(path) -> dict | None:
    """`{free_bytes, total_bytes}` of the volume holding `path` (or its nearest
    existing parent), or None when it cannot be read. This is the number that
    answers "will the move fit"."""
    p = Path(path)
    for candidate in (p, *p.parents):
        if candidate.exists():
            try:
                usage = shutil.disk_usage(str(candidate))
            except OSError:
                return None
            return {'free_bytes': usage.free, 'total_bytes': usage.total}
    return None


def validate_target(key, path) -> dict:
    """Can this category be pointed at `path`? Never writes config.

    Answers `{ok, reason, path, exists, empty, free_bytes, total_bytes}`. It DOES
    touch the filesystem: it creates the folder if missing and writes a probe
    file into it (removed again), so a "check" can leave an empty directory
    behind. That is the price of an honest answer — see below.

    The checks are the ones that actually bite:
      * a relative path (config is read from several working directories);
      * a path that cannot be created, or is not writable — proved by writing a
        probe file, not by reading permission bits, which lie on Windows;
      * the target being INSIDE the folder it would replace, which would make a
        move recurse into itself."""
    d = _definition(key)
    if not d or not (d['section'] and d['field']):
        return {'ok': False, 'reason': 'This location cannot be changed here.'}
    raw = str(path or '').strip()
    if not raw:
        # Blank is legitimate: it means "back to the default".
        return {'ok': True, 'reason': '', 'path': '', 'default': True,
                'exists': True, 'empty': False}
    target = Path(os.path.expanduser(raw))
    if not target.is_absolute():
        return {'ok': False, 'reason': 'Use a full path (for example D:\\lds-data '
                                       'or /mnt/data/lds).', 'path': raw}
    try:
        current = d['resolve']()
    except Exception:
        current = None
    if current:
        try:
            if target.resolve() != Path(current).resolve() \
                    and Path(current).resolve() in target.resolve().parents:
                return {'ok': False, 'path': str(target),
                        'reason': 'That folder is inside the one it would replace.'}
        except OSError:
            pass
    existed = target.is_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {'ok': False, 'path': str(target),
                'reason': f'The app cannot create that folder ({e.strerror or e}).'}
    probe = target / f'.lds-write-test-{uuid.uuid4().hex[:8]}'
    try:
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
    except OSError as e:
        return {'ok': False, 'path': str(target),
                'reason': f'That folder is not writable ({e.strerror or e}).'}
    try:
        entries = [n for n in os.listdir(target) if not n.startswith('.')]
    except OSError:
        entries = []
    info = free_space(str(target)) or {}
    return {'ok': True, 'reason': '', 'path': str(target),
            'exists': existed, 'empty': not entries,
            'entries': len(entries), **info}


# ── Move jobs ────────────────────────────────────────────────────────────────
# One at a time, in a thread, with a progress the tab can poll. Copy-then-delete
# rather than os.replace: the whole point of this feature is moving ACROSS
# volumes, where a rename is not available.

_jobs = {}
_jobs_lock = threading.Lock()


def move_progress(job_id=None) -> dict | None:
    """Snapshot of a move job (the latest one when `job_id` is None)."""
    with _jobs_lock:
        if job_id:
            job = _jobs.get(str(job_id))
            return dict(job) if job else None
        if not _jobs:
            return None
        latest = max(_jobs.values(), key=lambda j: j['started_at'])
        return dict(latest)


def _active_job():
    for job in _jobs.values():
        if job['phase'] in ('scanning', 'copying'):
            return job
    return None


def start_move(key, dest) -> dict:
    """Move a category's current content to `dest` in the background.

    Returns the job id. Raises ValueError when the target is unusable or another
    move is already running — two concurrent moves over the same volume would
    only fight for the same disk."""
    check = validate_target(key, dest)
    if not check.get('ok'):
        raise ValueError(check.get('reason') or 'unusable target folder')
    if check.get('default'):
        raise ValueError('Pick a folder to move into.')
    d = _definition(key)
    src = d['resolve']()
    if not src:
        raise ValueError('This location has no folder to move yet.')
    src = Path(src)
    with _jobs_lock:
        if _active_job():
            raise ValueError('A move is already running — wait for it to finish.')
        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {'id': job_id, 'key': key, 'phase': 'scanning',
                         'files': 0, 'files_total': 0, 'bytes': 0,
                         'bytes_total': 0, 'error': '', 'current': '',
                         'dest': check['path'], 'started_at': time.time()}
    thread = threading.Thread(target=_run_move, name=f'lds-move-{key}',
                              args=(job_id, str(src), check['path']), daemon=True)
    thread.start()
    return {'job_id': job_id, 'dest': check['path']}


def _update(job_id, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)


def _run_move(job_id, src, dest):
    try:
        pairs = []
        total = 0
        for root, _dirs, files in os.walk(src):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, src)
                try:
                    total += os.path.getsize(full)
                except OSError:
                    pass
                pairs.append((full, os.path.join(dest, rel)))
        _update(job_id, phase='copying', files_total=len(pairs), bytes_total=total)
        done_bytes = 0
        for i, (full, target) in enumerate(pairs, 1):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not os.path.exists(target):
                shutil.copy2(full, target)
            try:
                done_bytes += os.path.getsize(target)
            except OSError:
                pass
            _update(job_id, files=i, bytes=done_bytes,
                    current=os.path.basename(full))
        # Only once every byte is at the destination do we remove the source.
        # A half-copied move that loses the original is exactly the failure this
        # whole feature exists to stop happening.
        shutil.rmtree(src, ignore_errors=True)
        _update(job_id, phase='done', current='')
        logger.info('storage: moved %s file(s) to a new location', len(pairs))
    except Exception as e:
        logger.exception('storage: move failed')
        _update(job_id, phase='error', error=str(e))
