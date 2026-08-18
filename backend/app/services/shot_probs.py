"""Where the detector's per-frame probabilities live after the pass ends.

WHY THEY ARE KEPT AT ALL. The expensive part of shot detection is decoding —
the network runs on 48x27 frames and is never the bottleneck. So the ONE thing
worth never paying twice is the decode, and the entire output of that decode,
for thresholding purposes, is two float vectors per file. Keeping them turns
"I disagree with this cut list" from a GPU pass into a file read: a sweep across
six thresholds, a per-file override, a re-detection of one source, all instant
and all offline. Everything in `shot_boundaries` is downstream of this file.

ONE .NPZ PER SOURCE, NOT ONE PER BANK. The search lane keeps a single vector
store for the whole bank because every search reads the whole matrix at once.
Here the opposite is true: the gesture is "re-cut THIS file", and a bank-wide
store would mean rewriting tens of megabytes inside an interactive click. The
files sit next to the bank's thumbnails, under the bank's own folder, so
deleting the bank takes them along with everything else derived from it.

SIZE, AND WHY NOT float16. A one-hour 30 fps rush is ~108 000 frames: 2 x 432 KB
of float32, compressed hard because these vectors are near-zero almost
everywhere. float16 would halve that and was tried first — its step near 0.8 is
~4e-4, which is larger than the rounding the worker already applied and is
enough to move a single frame across a threshold set exactly there. Saving
200 KB by making a re-threshold disagree with the pass that filled the cache is
a bad trade: the entire promise of this file is that the two agree.

A MISSING OR UNREADABLE CACHE IS "NOT CACHED", NEVER AN ERROR. The caller's
answer to both is the same — run detection on this file — and a half-written
.npz from a killed process must not be able to fail a page.
"""
from __future__ import annotations

import logging

from . import atomic_npz

logger = logging.getLogger(__name__)

# The two arrays a cache file holds. 'all' may be absent: the second head is
# what the labels are made of, and a file detected without it still
# re-thresholds perfectly — only the cut/dissolve chip is unavailable.
_SINGLE = 'single'
_ALL = 'all'


def _bank_dir(bank_id):
    """Indirected through a function so tests can point the store somewhere
    else without standing up a bank. Imported lazily: video_bank_service
    imports this module's callers, and a top-level import would be a cycle."""
    from .video_bank_service import _bank_dir as bank_dir
    return bank_dir(bank_id)


def probs_dir(bank_id):
    return _bank_dir(bank_id) / 'shot_probs'


def probs_path(bank_id, source_id):
    """One file per SOURCE — see the module docstring for why not per bank."""
    return probs_dir(bank_id) / f'source_{int(source_id)}.npz'


def save_probs(bank_id, source_id, single, every=None):
    """Persist one source's two probability vectors. Atomic: a crash mid-write
    cannot leave a source with half a cache (which would read back as a
    plausible, shorter video)."""
    import numpy as np
    path = probs_path(bank_id, source_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {_SINGLE: np.asarray(list(single or []), dtype='float32')}
    if every is not None:
        arrays[_ALL] = np.asarray(list(every), dtype='float32')
    atomic_npz.save_npz_atomic(path, arrays)


def load_probs(bank_id, source_id):
    """{'single': [float], 'all': [float]|None} — or None when there is no
    usable cache for this source."""
    import numpy as np
    path = probs_path(bank_id, source_id)
    try:
        with np.load(str(path), allow_pickle=False) as store:
            if _SINGLE not in store.files:
                return None
            single = [float(v) for v in store[_SINGLE]]
            every = ([float(v) for v in store[_ALL]]
                     if _ALL in store.files else None)
    except FileNotFoundError:
        return None
    except Exception as error:      # noqa: BLE001 — corrupt reads as "not cached"
        logger.info('video bank %s: shot probabilities for source %s are '
                    'unreadable (%s) — this file will need a real pass',
                    bank_id, source_id, type(error).__name__)
        return None
    return {'single': single, 'all': every}


def has_probs(bank_id, source_id) -> bool:
    """Cheap existence check for the UI's "instant" affordance. Deliberately
    NOT a full read: the panel asks this for every source in a bank."""
    try:
        return probs_path(bank_id, source_id).exists()
    except OSError:
        return False


def forget(bank_id, source_id):
    """Drop one source's cache. Never raises — a locked file is not worth a
    500, and the worst case is a stale cache the next pass overwrites."""
    try:
        probs_path(bank_id, source_id).unlink(missing_ok=True)
    except OSError:
        logger.info('video bank %s: could not remove the shot probabilities of '
                    'source %s', bank_id, source_id)
