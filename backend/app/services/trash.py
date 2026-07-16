"""App-wide trash: NOTHING the app deletes is destroyed directly — files and
folders are MOVED into data/trash/<timestamp>_<context>/ so a wrong click on a
1 GB checkpoint is recoverable. Settings shows the trash size and an
'Empty trash' button (the only place bytes actually die).

Cross-drive moves (ComfyUI on another drive) degrade to copy+delete via
shutil.move — slower for GB files but deletes are rare."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .. import config as cfg

logger = logging.getLogger(__name__)


def trash_root() -> Path:
    root = cfg._data_dir() / 'trash'
    root.mkdir(parents=True, exist_ok=True)
    return root


def send_to_trash(path, context='') -> str:
    """Move a file or folder into the trash; returns its new location.
    Raises on a missing source (callers whitelist first)."""
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
    shutil.move(str(src), str(dest))
    logger.info('trashed %s -> %s', src, dest)
    return str(dest)


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
