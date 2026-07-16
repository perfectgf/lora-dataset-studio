"""Filesystem paths for dataset-owned images.

Callers must choose explicitly between resolving a path and creating its
directory.  Read-only routes should never mutate the filesystem as a side
effect of looking up a missing file.
"""
from __future__ import annotations

from .. import config as cfg


def dataset_path(dataset_id: int) -> str:
    """Return the dataset directory path without creating it."""
    return str(cfg.dataset_images_root() / str(dataset_id))


def ensure_dataset_dir(dataset_id: int) -> str:
    """Create the dataset directory if needed and return its path."""
    path = cfg.dataset_images_root() / str(dataset_id)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
