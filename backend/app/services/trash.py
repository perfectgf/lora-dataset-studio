"""App-wide trash: NOTHING the app deletes is destroyed directly — files and
folders are MOVED into data/trash/<timestamp>_<context>/ so a wrong click on a
1 GB checkpoint is recoverable. Settings shows the trash size and an
'Empty trash' button (the only place bytes actually die).

Cross-drive moves (ComfyUI on another drive) degrade to copy+delete via
shutil.move — slower for GB files but deletes are rare."""
from __future__ import annotations

import errno
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .. import config as cfg

logger = logging.getLogger(__name__)

# A just-written file can stay held open for a beat by an antivirus scan
# (Bitdefender ATD here) or a lingering preview handle. On Windows that turns a
# move into WinError 32/5; a short retry rides over the scan window before we
# give up. Module-level so tests can shrink the delay.
_LOCK_RETRIES = 4
_LOCK_RETRY_DELAY = 0.4  # seconds; ~1.2s of added latency only on a locked path


class TrashLockError(OSError):
    """A file under ``path`` is still open in another process (an antivirus scan
    of a just-written image, an open preview, a lingering handle), so it can't be
    moved to Trash yet. Subclasses OSError so existing ``except OSError`` callers
    still catch it; delete_dataset/delete_image translate it into an actionable
    message instead of a bare 500."""

    def __init__(self, path, cause: OSError | None = None):
        self.path = str(path)
        self.cause = cause
        super().__init__(f'{self.path} is locked by another process')


def _is_sharing_violation(err: OSError | None) -> bool:
    """The file is held open elsewhere: Windows sharing violation (32) / access
    denied (5), or a POSIX permission error."""
    if err is None:
        return False
    if os.name == 'nt' and getattr(err, 'winerror', None) in (5, 32):
        return True
    return isinstance(err, PermissionError)


def _is_cross_device(err: OSError) -> bool:
    """Source and destination are on different volumes (ComfyUI on another
    drive): a rename can't span them, so a copy+delete is required. EXDEV on
    POSIX, ERROR_NOT_SAME_DEVICE (17) on Windows."""
    return err.errno == errno.EXDEV or getattr(err, 'winerror', None) == 17


def trash_root() -> Path:
    root = cfg._data_dir() / 'trash'
    root.mkdir(parents=True, exist_ok=True)
    return root


def send_to_trash(path, context='') -> str:
    """Move a file or folder into the trash; returns its new location.

    An atomic rename is tried first: it either moves the whole tree or fails
    without touching a byte, so a locked file aborts cleanly instead of leaving a
    half-copied folder in Trash next to a half-deleted source (which
    ``shutil.move``'s copytree+rmtree fallback does on Windows). Only a genuine
    cross-drive move degrades to copy+delete. A file still held open by another
    process is retried briefly, then raised as :class:`TrashLockError` so the
    caller can surface an actionable message rather than a bare 500.

    Raises FileNotFoundError on a missing source (callers whitelist first)."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(str(src))
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    safe_ctx = ''.join(ch if ch.isalnum() or ch in '-_' else '_'
                       for ch in str(context))[:60]
    base = f'{stamp}_{safe_ctx}' if safe_ctx else stamp
    dest_dir = trash_root() / base
    n = 1
    while dest_dir.exists():                     # same-second collision
        n += 1
        dest_dir = trash_root() / f'{base}_{n}'
    dest_dir.mkdir(parents=True)
    dest = dest_dir / src.name
    last_err: OSError | None = None
    for attempt in range(_LOCK_RETRIES):
        try:
            os.rename(src, dest)
            logger.info('trashed %s -> %s', src, dest)
            return str(dest)
        except OSError as e:
            last_err = e
            if _is_cross_device(e):
                # Different drive: copy then delete (rare, and never mid-write).
                shutil.move(str(src), str(dest))
                logger.info('trashed (cross-device) %s -> %s', src, dest)
                return str(dest)
            if _is_sharing_violation(e) and attempt < _LOCK_RETRIES - 1:
                time.sleep(_LOCK_RETRY_DELAY)
                continue
            break
    # Gave up: never leave the empty staging dir (or a partial copy) behind.
    try:
        dest_dir.rmdir()
    except OSError:
        pass
    if _is_sharing_violation(last_err):
        raise TrashLockError(src, last_err) from last_err
    raise last_err if last_err is not None else OSError(f'could not trash {src}')


def open_trash_folder() -> str:
    """Open the fixed app trash directory in the host file explorer."""
    path = str(trash_root())
    if os.name == 'nt':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])
    logger.info('opened trash folder: %s', path)
    return path


def trash_size() -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(trash_root()):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def empty_trash() -> dict:
    """The one place bytes actually die. Returns {'removed', 'freed_bytes'}."""
    root = trash_root()
    freed = trash_size()
    removed = 0
    for entry in list(root.iterdir()):
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except OSError as e:
            logger.warning('empty_trash: could not remove %s: %s', entry, e)
    return {'removed': removed, 'freed_bytes': freed}
