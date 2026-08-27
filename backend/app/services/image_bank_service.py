"""🗃️ Image bank service — triage a big unsorted folder into dataset-ready
selections.

The founding use case: "I exported 9 000 images from Telegram — now what?".
A bank references that folder IN PLACE (no copy, the source files are never
written to) and layers a funnel on top:

  1. inventory      — instant: walk the folder, register every image file;
  2. quality pass   — background thread, CPU/pure-PIL: sharpness, noise,
                      uniformity, dimensions, dHash + duplicate groups;
                      raw scores persist, FLAGS are computed at read time
                      against the config thresholds ('bank' section) so
                      recalibrating never needs a rescan;
  3. duplicates     — near-duplicate groups (same 64-bit dHash family as the
                      dataset import dedup) with keep-best / keep-first /
                      manual resolution — losers are REJECTED (a status),
                      never deleted from disk;
  4. subject pass   — optional, needs the face-scoring extra: InsightFace
                      embeddings (cached in an .npz next to the thumbs) +
                      person clustering, to sort a mixed dump by WHO is in
                      the frame without any reference photo;
  5. promotion      — the kept selection enters a dataset through the configured
                      import policy (raw-preserve by default, optional WebP
                      normalisation) + perceptual dedup, inheriting every downstream tool
                      (captions, watermarks, face scoring, training).

Long passes run through bank_jobs (one background thread per bank, polled via
the bank payload) — a 9 000-image folder must scan in minutes without ever
holding an HTTP request open or freezing the UI.
"""
from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from PIL import Image, ImageOps
from sqlalchemy import and_, case, func, or_, text

from .. import config as cfg
from ..extensions import db
from ..models import BankImage, FaceDataset, FaceDatasetImage, ImageBank
from . import (bank_jobs, bank_semantic_engine, bank_transfer_metadata, bank_undo, caption_origin,
               dataset_activity, image_encoding, path_guard, trash)
# The scope vocabulary is a leaf (pass_scopes.py) so face_dataset_service never
# imports this module; the three names stay readable as banks.* for every caller.
from .pass_scopes import PASS_SCOPES, CAPTION_SCOPES, normalize_pass_statuses  # noqa: F401
# Re-exported on purpose: the promote tests patch `banks._existing_dhash_rows`.
from .face_dataset_service import _existing_dhash_rows  # noqa: F401
from .face_dataset_service import (SCRAPE_IMPORT_MAX, _dhash, _download_scrape_item,
                                   _dataset_ingest_lock,
                                   _hamming, _SCRAPE_DL_WORKERS,
                                   _watermark_regions_payload,
                                   _source_metadata_storage, bank_deterministic_analysis,
                                   import_images, _preserved_import_extension,
                                   rollback_imported_images,
                                   normalize_watermark_regions)
from .image_quality import ANALYSIS_MAX_SIDE, quality_metrics
from .image_provenance import ORIGINS, provenance_metrics
from . import infer_env

logger = logging.getLogger(__name__)

# At most one SHA-proven path map per engine. Generic analysis writers may bind
# thousands of rows in one pass; alternating CLIP/SigLIP cache loads per row would
# defeat the semantic engine's intentionally one-cache array memo.
_SEMANTIC_GROUP_PROOF_LOCK = threading.RLock()
_semantic_group_proof_memo = {}

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
# Sanity cap — a bank is a triage layer, not a filesystem indexer: this is what
# catches "I pointed it at a whole drive", nothing more. Measured on a synthetic
# tree of 200 000 files (50 subfolders, warm cache): the walk refresh_bank runs
# costs 2.5 s, the per-file getsize 1.8 s, the chunked INSERT of every row 0.2 s,
# and reading the known-relpath set back 0.14 s. The walk is the only real cost and
# it is already cooldown-limited (FOLDER_SYNC_COOLDOWN), so 50 000 was far below
# what the code can carry; the per-image passes that ARE expensive have always
# been background jobs with their own progress bar.
BANK_MAX_FILES = 200_000
THUMB_MAX_SIDE = 320
_COMMIT_EVERY = 25          # scan DB flush cadence
_PROMOTE_CHUNK = 20         # files per import_images call (bounded memory)
_SQL_IN_CHUNK = 500         # SQLite bound-variable ceiling is 999
# --- duplicate regrouping budgets (see rebuild_dup_groups) -------------------
# Rows written between two commits. The whole regrouping used to be ONE
# transaction: a global clear plus one UPDATE per group, ~5 000 of them on a
# 50 000-image bank, measured holding the single SQLite write lock 6 to 9.5 s —
# past the 5 s busy_timeout, so every other writer in the app died on
# `database is locked` (3 attempts out of 3) while the progress bar sat at 100 %.
_DUP_WRITE_ROWS = 2000
# And the pause AFTER each batch. Without it the next batch re-takes the write
# lock in the microsecond after the commit, inside the sleep of another writer's
# busy handler: batching alone still left a concurrent writer waiting 620 ms and
# refused (measured, test_bank_scan_no_db_lock.py). Five batches on a 50 000-image
# bank make this 100 ms of wall time in total.
_DUP_WRITE_YIELD = 0.02
# Bucket size from which the pairwise comparison goes to numpy. Below it the
# plain Python loop is cheaper than allocating arrays, and it is bounded by
# construction (at most 21 pairs).
_DUP_NUMPY_FROM = 8
# Ceiling on the cells of ONE comparison block, so peak memory follows this
# constant and not the bucket size: 2e6 cells ≈ 34 MB of uint64 XOR plus its
# popcount lookup, freed at every block.
_DUP_BLOCK_CELLS = 2_000_000
_POPCOUNT8 = None           # lazily built uint8 popcount table (see _popcount_lut)
# Semantic cosine grouping is exact, but one request must not turn the 200k Bank
# ceiling into 20 billion pair checks.  750M still covers a single 36k-image
# SigLIP2 Bank (648M pairs); larger global partitions need an ANN implementation
# rather than silently allocating/working without a bound. CLIP normally stays
# well below this because its historical style clusters partition the work.
_SEMANTIC_EXACT_PAIR_LIMIT = 750_000_000
_SEMANTIC_TILE = 512       # at most 512x512 float/bool cells at once
# A quality pass that keeps finding NOTHING on disk is not looking at a bank of
# broken images — it is looking at the wrong folder. Bail out after this many
# absent files (when they are at least half of what has been walked) rather than
# grind through 30 000 of them; a handful of genuinely deleted files stays a
# non-event and only shows up in the folder-sync note.
_MISSING_ABORT_AT = 20
MOVED_FOLDER_MSG = (
    "this folder no longer holds the bank's images — it may have been moved, "
    'renamed, or sit on a drive that is disconnected. Nothing was changed: '
    'point the bank at its new folder, then run the pass again.')

# A Bank points at a live user folder: files can arrive or be replaced long after
# import.  Keep the same byte/pixel envelope as Dataset ingress before any Bank
# consumer decodes, copies to an infer child, or reads the whole file into RAM.
# 96 MiB covers a valid 16 MP RGBA/BMP master plus container overhead.
BANK_SOURCE_MAX_BYTES = 96 * 1024 * 1024


def _bank_source_size_guard(path, *, label: str) -> int:
    try:
        size = os.path.getsize(path)
    except (OSError, TypeError, MemoryError) as exc:
        raise ValueError(f'{label} is unavailable') from exc
    if size > BANK_SOURCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large (max {BANK_SOURCE_MAX_BYTES // (1024 * 1024)} MiB)')
    return size


@contextmanager
def safe_bank_source(path, *, label: str = 'bank image'):
    """Open one live Bank source after static/header safety validation.

    The yielded Pillow object remains open so callers can safely perform their
    one intended decode.  Header validation happens first, with Pillow's bomb
    warning promoted locally to an exception; the global warning policy is never
    mutated by a background worker.
    """
    _bank_source_size_guard(path, label=label)
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as image:
            fmt = (image.format or '').upper()
            if fmt not in image_encoding.EDITABLE_FORMATS:
                raise ValueError(f'{label} has unsupported format: {fmt or "unknown"}')
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError(f'{label} must be a static image')
            image_encoding.validate_input_header_dimensions(image, label=label)
            yield image


def _read_bounded_bank_source_bytes(path, *, label: str = 'bank image') -> bytes:
    """Take one bounded raw snapshot without claiming it is a valid image.

    The quality scanner needs this lower-level seam so it can bind an
    ``unreadable`` verdict to the exact malformed bytes it inspected.  Every
    consumer that needs a usable image must still go through
    :func:`_read_safe_bank_source_bytes` below.
    """
    _bank_source_size_guard(path, label=label)
    try:
        with open(path, 'rb') as source:
            raw = source.read(BANK_SOURCE_MAX_BYTES + 1)
    except (OSError, TypeError, MemoryError) as exc:
        raise ValueError(f'{label} is unavailable') from exc
    if len(raw) > BANK_SOURCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large (max {BANK_SOURCE_MAX_BYTES // (1024 * 1024)} MiB)')
    return raw


def _read_safe_bank_source_bytes(path, *, label: str = 'bank image') -> bytes:
    """Read one live source with both a raw-byte cap and content validation.

    This closes the time-of-check/time-of-use gap for paths handed as bytes to
    Dataset import or Ollama: the exact bounded bytes read are revalidated before
    any downstream service sees them.
    """
    raw = _read_bounded_bank_source_bytes(path, label=label)
    try:
        _preserved_import_extension(raw, label=label)
    except MemoryError as exc:
        raise ValueError(f'{label} is invalid') from exc
    return raw


def _is_safe_bank_source(path, *, label: str = 'bank image') -> bool:
    """Header-check a live source for an infer/caption path without decoding it."""
    if not path or not os.path.isfile(path):
        return False
    try:
        with safe_bank_source(path, label=label):
            return True
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        return False


# --- thresholds -------------------------------------------------------------
# The blur default before the per-tile scoring rework. On the current scale
# (p90 of tile variances, image_quality.py) it flags nothing: measured on a
# real 36 921-image bank, the LOWEST stored blur_score was 103.9. Any config
# still carrying exactly this value carries the stale default, not a choice —
# a full-config Save writes every default back into config.json, so most
# installs hold it without anyone ever having picked it.
_STALE_SHARPNESS_DEFAULT = 100.0


def thresholds() -> dict:
    """The 'bank' config section, sanitized (a corrupt config.json value falls
    back to the default instead of poisoning every flag computation)."""
    out = {}
    for key, default in cfg.DEFAULTS['bank'].items():
        try:
            out[key] = float(cfg.get(f'bank.{key}', default))
        except (TypeError, ValueError):
            out[key] = float(default)
    # Migrate the dead pre-rework default to the recalibrated one. Only the
    # EXACT old default: 90 or 120 in a config is a hand-tuned value and stays,
    # even though it is almost certainly dead too — a deliberate setting is
    # never rewritten, a stale default is not a setting.
    if out['sharpness_min'] == _STALE_SHARPNESS_DEFAULT:
        out['sharpness_min'] = float(cfg.DEFAULTS['bank']['sharpness_min'])
    out['dup_distance'] = int(out['dup_distance'])
    out['min_side'] = int(out['min_side'])
    return out


# --- storage helpers --------------------------------------------------------
def _bank_dir(bank_id) -> Path:
    return cfg.banks_root() / str(bank_id)


def _thumbs_dir(bank_id) -> Path:
    return _bank_dir(bank_id) / 'thumbs'


def _face_cache_path(bank_id) -> Path:
    return _bank_dir(bank_id) / 'face_cache.npz'


def _score_cache_path(bank_id) -> Path:
    return _bank_dir(bank_id) / 'score_cache.npz'


def _semantic_cache_path(bank_id) -> Path:
    """The SigLIP2 cache path, fixed for transfer/runtime-cache plumbing."""
    return bank_semantic_engine.semantic_cache_path(bank_id, 'siglip2')


def engine_model_key(engine) -> str:
    """Public pure wrapper used by routes/tests without importing cache internals."""
    return bank_semantic_engine.engine_model_key(engine)


def semantic_cache_path(bank_or_id, engine=None) -> Path:
    selected = (engine if engine is not None else
                getattr(bank_or_id, 'semantic_engine', None))
    return bank_semantic_engine.semantic_cache_path(bank_or_id, selected)


def semantic_counts(bank_or_id, engine=None, total=None, *,
                    eligible_paths=None) -> dict:
    return bank_semantic_engine.semantic_counts(
        bank_or_id, engine=engine, total=total,
        eligible_paths=eligible_paths)


def _abs_under(base: str, relpath: str) -> str | None:
    """The containment-checked realpath of ``relpath`` under an ALREADY resolved
    ``base``. Split out of ``abs_image_path`` so a loop over thousands of rows can
    resolve the bank folder ONCE instead of per row: ``os.path.realpath`` is a
    filesystem call, and re-resolving the same unchanging bank folder for every
    image cost 424 ms of the 6 353-row curation pool alone (measured). Same
    strings out, one syscall in."""
    full = os.path.realpath(os.path.join(base, relpath))
    if os.path.normcase(full).startswith(os.path.normcase(base + os.sep)):
        return full
    return None


def abs_image_path(bank: ImageBank, row: BankImage) -> str | None:
    """Absolute SOURCE path of a bank image, or None when it escapes the
    bank's folder (belt & braces — relpaths only ever come from our own walk).

    ⚠️ This is the user's own file. It is READ-ONLY for us, and it is NOT what
    the app should display or copy once a watermark has been cleaned — every
    reader must go through resolved_image_path() instead (see its docstring)."""
    return _abs_under(os.path.realpath(bank.source_path), row.relpath)


_POLL_PATH_LOCK = threading.RLock()
_POLL_PATH_CACHE: 'OrderedDict[int, tuple[str, dict[str, str | None]]]' = OrderedDict()
# ~250 000 resolved relpaths across all banks: five banks the size of the one that
# produced this fix, a few tens of MB of strings. Whole banks are evicted, oldest
# first, so a cache miss costs exactly what the uncached code always cost.
_POLL_PATH_BUDGET = 250_000


def _poll_path_memo(bank: ImageBank) -> tuple[str, dict[str, str | None]]:
    """``(resolved bank folder, relpath -> containment-checked absolute path)``
    for paths a POLL only ever uses as comparison keys.

    ⚠️ Read this before reusing it anywhere else. ``_abs_under`` is a directory
    escape guard, and this memoizes its verdict. That is sound HERE and nowhere
    obvious else, because the caller (:func:`_semantic_eligible_paths`) never
    opens, serves, copies or writes these paths — it hands them to the semantic
    cache inspector as a membership set deciding which cached rows count toward
    ``ok / total``. A path is a key here, not a capability. Every path that DOES
    become a capability — serving bytes, cleaning, promoting, deleting — still
    goes through :func:`abs_image_path` and pays the live guard on every call.

    The verdict is memoized including its REJECTIONS: a relpath that escapes the
    bank folder is remembered as ``None`` and can never be promoted to allowed by
    a later hit. The whole map is dropped when the bank's resolved folder changes.

    Why it exists: the workspace polls ``bank_payload`` every 2 s, and this list
    was resolved from scratch each time — ``os.path.realpath`` twice per image,
    and on Windows each one OPENS the file through ``nt._getfinalpathname``.
    Measured on a 50 397-image bank (36 870 eligible): 147 502 syscalls and 12.5 s
    per poll, 28.9 s while a pass ran. Requests issued every 2 s that take 12 s
    pile up, and this payload carries the job banner — so the bank became
    unreadable AND its Stop button unreachable, which is how it was reported.
    """
    base = os.path.realpath(bank.source_path)
    with _POLL_PATH_LOCK:
        entry = _POLL_PATH_CACHE.get(bank.id)
        if entry is None or entry[0] != base:
            entry = (base, {})
        _POLL_PATH_CACHE[bank.id] = entry
        _POLL_PATH_CACHE.move_to_end(bank.id)
        while (len(_POLL_PATH_CACHE) > 1
               and sum(len(m) for _b, m in _POLL_PATH_CACHE.values())
               > _POLL_PATH_BUDGET):
            _POLL_PATH_CACHE.popitem(last=False)
    return entry


def reset_poll_path_memo() -> None:
    """Forget every memoized poll-path verdict (tests, bank deletion)."""
    with _POLL_PATH_LOCK:
        _POLL_PATH_CACHE.clear()


def _clean_dir(bank_id) -> Path:
    """Where watermark-cleaned versions live — the bank's OWN working directory,
    next to thumbs/. Never inside the user's folder."""
    return _bank_dir(bank_id) / 'clean'


def clean_image_path(bank_id, image_id) -> Path:
    """The cleaned blob of one image (may not exist)."""
    return _clean_dir(bank_id) / f'{image_id}.webp'


def _rotated_dir(bank_id) -> Path:
    """Where manually TURNED copies live — the bank's own working directory,
    next to clean/. Never inside the user's folder."""
    return _bank_dir(bank_id) / 'rotated'


def rotated_image_path(bank_id, image_id, rotation, source: str,
                       source_fingerprint=None) -> Path:
    """Where the turned copy lives, content-addressed by its exact source.

    A Bank watches a live user folder. Keying only on image id + angle made a
    same-path replacement reuse the previous file's rotated pixels forever.
    The full source SHA makes a cached derivative valid for exactly one raw or
    cleaned generation while preserving the original image format.
    """
    source_fingerprint = (source_fingerprint
                          or bank_transfer_metadata.content_fingerprint_path(source))
    if not _valid_analysis_fingerprint(source_fingerprint):
        raise ValueError('rotation source could not be fingerprinted')
    ext = os.path.splitext(source)[1].lower() or '.png'
    return (_rotated_dir(bank_id)
            / f'{image_id}.{source_fingerprint}.r{int(rotation)}{ext}')


def _prune_rotated_generations(destination: Path, image_id) -> None:
    """Best-effort removal of obsolete content-addressed/legacy siblings."""
    try:
        candidates = list(destination.parent.glob(f'{image_id}.*.r*'))
        candidates.extend(destination.parent.glob(f'{image_id}.r*'))
        for stale in candidates:
            if stale != destination:
                try:
                    stale.unlink()
                except OSError:
                    pass   # already gone or locked: a stale sibling owes nothing
    except OSError:
        pass   # already gone or locked: a stale thumb owes nothing


def _edited_dir(bank_id) -> Path:
    """Where ✂ Crop / ✨ Upscale & improve results live — the bank's own working
    directory, next to clean/ and rotated/. Never inside the user's folder."""
    return _bank_dir(bank_id) / 'edited'


def edited_image_path(bank_id, image_id, generation) -> Path:
    """The edited blob of one image, at one generation (may not exist).

    The generation is in the NAME rather than being overwritten in place for two
    reasons that both bit the rotation lane first: on Windows the file may still
    be held open by the response that just served it, and a thumbnail cached for
    an hour must not be able to outlive the edit that replaced it.
    """
    return _edited_dir(bank_id) / f'{image_id}.e{int(generation or 0)}.webp'


def _prune_edited_generations(bank_id, image_id, keep: Path | None) -> None:
    """Best-effort removal of every edited blob of one image except ``keep``.
    A leftover is harmless — nothing points at it once the row moved on — so a
    locked file is not an error."""
    try:
        for stale in _edited_dir(bank_id).glob(f'{image_id}.e*.webp'):
            if keep is not None and stale == keep:
                continue
            try:
                stale.unlink()
            except OSError:
                pass   # already gone or locked: a stale thumb owes nothing
    except OSError:
        pass   # already gone or locked: a stale thumb owes nothing


def _drop_edited_blob(bank_id, row) -> None:
    """Forget a bank-side edit: delete every generation of its blob, give the
    baked turn back to the row and clear the markers so the readers fall back to
    the clean/source chain. No commit (the caller owns the transaction)."""
    _prune_edited_generations(bank_id, row.id, None)
    if row.edit_method:
        row.rotation = row.edit_baked_rotation or None
    row.edit_method = None
    row.edit_generation = None
    row.edit_baked_rotation = None


def _ensure_rotated(bank_id, row: BankImage, source: str) -> str:
    """Materialise (once) the turned copy of ``source`` and return its path.

    ALWAYS built from the pristine source — never from a previously rotated copy
    — so the angle is applied exactly once no matter how many times the user
    clicked. Fails OPEN: an encoder problem serves the un-turned image rather
    than a 404, because a bank must stay browsable."""
    from .face_dataset_service import (normalize_rotation,
                                       transformed_image_bytes, rotate_transform)
    try:
        turn = normalize_rotation(row.rotation)
    except ValueError:
        return source
    if not turn:
        return source
    source_fingerprint = bank_transfer_metadata.content_fingerprint_path(source)
    if not _valid_analysis_fingerprint(source_fingerprint):
        return source
    dst = rotated_image_path(
        bank_id, row.id, turn, source,
        source_fingerprint=source_fingerprint)
    if dst.is_file():
        _prune_rotated_generations(dst, row.id)
        return str(dst)
    try:
        # The generic transform below also checks the pixel budget. This Bank
        # gate additionally constrains raw bytes/static content from a live
        # user folder before any edit helper gets a chance to decode it.
        with safe_bank_source(source, label='bank rotation'):
            pass
        payload = transformed_image_bytes(
            source, rotate_transform(turn), max_source_bytes=BANK_SOURCE_MAX_BYTES)
        if (bank_transfer_metadata.content_fingerprint_path(source)
                != source_fingerprint):
            raise OSError('rotation source changed while it was being read')
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Two simultaneous GETs of the same turned image (grid + lightbox) each
        # build it. A SHARED temp name would make one of them fail on Windows
        # (the other still holds it) and degrade to serving the un-turned source;
        # a unique one lets both win and the last atomic replace decide.
        tmp = dst.with_name(f'{dst.name}.{os.getpid()}-{threading.get_ident()}.tmp')
        try:
            tmp.write_bytes(payload)
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass   # already gone: the temp owed nothing
        if (bank_transfer_metadata.content_fingerprint_path(source)
                == source_fingerprint):
            _prune_rotated_generations(dst, row.id)
        return str(dst)
    except (ValueError, OSError) as e:
        logger.warning('bank image %s could not be rotated: %s', row.id, e)
        return source


def resolved_image_path(bank: ImageBank, row: BankImage) -> str | None:
    """THE path every reader must use: the watermark-cleaned version when one
    exists, the untouched source otherwise — turned by the row's manual rotation
    when it carries one.

    A bank is a read-only view over a folder we must never write to, so neither
    cleaning a watermark nor turning an image can rewrite the source: both write
    a separate blob under the bank's working directory. That only pays off if the
    READERS prefer it, so there is exactly ONE resolver and all three known
    readers call it: promotion (the blob handed to import_images), the grid
    thumbnail, and the /bank/<id>/file route. A new reader calling
    abs_image_path() directly would silently serve the watermarked original —
    test_bank_watermark_clean.py asserts these three go through here.

    A bank-side ✂ crop / ✨ improve wins over both: it was made FROM whatever
    this resolver returned at the time (cleaned and turned included), so its blob
    already contains those transforms and re-applying them would apply them
    twice. Fails open like the rest — a missing edited blob serves the chain
    below rather than a 404."""
    if row.edit_method:
        edited = edited_image_path(bank.id, row.id, row.edit_generation)
        if edited.is_file():
            return str(edited)
    path = None
    if row.watermark_clean_method:
        cleaned = clean_image_path(bank.id, row.id)
        if cleaned.is_file():
            path = str(cleaned)
    if path is None:
        path = abs_image_path(bank, row)
    if path is None or not getattr(row, 'rotation', None):
        return path
    return _ensure_rotated(bank.id, row, path)


def analysis_image_path(bank: ImageBank, row: BankImage, *,
                        refresh_rotation=False) -> str | None:
    """Strict effective path used by every analysis/caption/cache lane.

    Display remains fail-open so a broken derivative never produces a 404.
    Analysis must fail closed: a requested clean without raw watermark authority,
    or a rotation that cannot be materialised, may never silently analyse the
    raw fallback and bless it as though it were the displayed transform.
    """
    raw = abs_image_path(bank, row)
    if not raw or not os.path.isfile(raw):
        return None
    # A bank-side edit is the effective image, and it is fail-CLOSED here for the
    # same reason as everything else in this function: analysing the source and
    # blessing the result as though it described the crop the user is looking at
    # is exactly the silent mismatch this variant exists to prevent.
    if row.edit_method:
        edited = edited_image_path(bank.id, row.id, row.edit_generation)
        return str(edited) if edited.is_file() else None
    path = raw
    if row.watermark_clean_method:
        if (not _valid_analysis_fingerprint(row.watermark_fingerprint)
                or bank_transfer_metadata.content_fingerprint_path(raw)
                != row.watermark_fingerprint):
            return None
        cleaned = clean_image_path(bank.id, row.id)
        if not cleaned.is_file():
            return None
        path = str(cleaned)
    if getattr(row, 'rotation', None):
        if refresh_rotation:
            try:
                candidate = rotated_image_path(
                    bank.id, row.id, row.rotation, path)
                # A completed analysis fingerprint authorises this exact
                # derivative as the current effective generation. Rebuilding it
                # would change its stat signature and make the exact Score/Face
                # caches stale on every pass/promotion, so refresh only an
                # unbound or mismatched copy.
                trusted = (candidate.is_file()
                           and _valid_analysis_fingerprint(
                               row.analysis_fingerprint)
                           and bank_transfer_metadata.content_fingerprint_path(
                               candidate) == row.analysis_fingerprint)
                if not trusted:
                    candidate.unlink(missing_ok=True)
            except (OSError, TypeError, ValueError):
                return None
        rotated = _ensure_rotated(bank.id, row, path)
        if _same_resolved_path(rotated, path):
            return None
        path = rotated
    return path if path and os.path.isfile(path) else None


# --- CRUD -------------------------------------------------------------------
class BankSourceFolderUnavailable(RuntimeError):
    """The owned Bank exists, but its recorded source is not an existing folder."""


def get_bank(user_id, bank_id) -> ImageBank | None:
    return ImageBank.query.filter_by(id=bank_id, user_id=user_id).first()


def _open_host_folder(path: str) -> None:
    """Reveal/explore a validated directory without executing the target."""
    if os.name == 'nt':
        os.startfile(path, 'explore')  # noqa: S606 - validated local directory
    elif sys.platform == 'darwin':
        # ``open <bundle>.app`` launches it. Finder's reveal mode selects the
        # target instead, which is the folder-button contract even for bundles.
        subprocess.Popen(['/usr/bin/open', '-R', path], shell=False)
    else:
        subprocess.Popen(['xdg-open', path], shell=False)


def open_bank_source_folder(user_id, bank_id) -> str | None:
    """Open one owned Bank's server-stored source folder in the host explorer.

    The caller supplies only the Bank id: no client-provided filesystem path
    reaches this boundary.  Unlike training-folder helpers this never creates a
    directory; a moved, disconnected or file-valued source is an explicit error.
    ``None`` deliberately conflates an unknown Bank with another user's Bank.
    """
    if (isinstance(bank_id, bool) or not isinstance(bank_id, int)
            or not 1 <= bank_id <= (1 << 63) - 1):
        return None

    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    stored_path = bank.source_path
    if (not isinstance(stored_path, str) or not stored_path
            or not os.path.isabs(stored_path)):
        raise BankSourceFolderUnavailable(
            'bank source folder is unavailable or is not a directory')
    try:
        # strict=True proves every component still exists before opening it and
        # resolves junctions/symlinks. Do not impose a drive-letter whitelist:
        # reachable UNC shares are absolute, legitimate Bank sources on Windows.
        path = os.path.realpath(stored_path, strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise BankSourceFolderUnavailable(
            'bank source folder is unavailable or is not a directory') from exc
    if not os.path.isdir(path):
        raise BankSourceFolderUnavailable(
            'bank source folder is unavailable or is not a directory')
    _open_host_folder(path)
    logger.info('opened Bank source folder: bank=%s path=%s', bank.id, path)
    return path


def _serialized_bank_mutation(kind):
    """Guard a synchronous ``(user_id, bank_id, ...)`` mutation atomically.

    The HTTP blueprint's busy check remains useful for a fast, friendly refusal,
    but it can never be the write fence: another thread may reserve the Bank
    after ``before_request`` returns.  Every decorated service therefore owns a
    short Bank lease for the whole mutation.  Background jobs/pipelines pass
    their exact reservation in ``_bank_lease`` and are validated rather than
    trying to acquire their own non-reentrant slot.
    """
    def decorate(fn):
        @wraps(fn)
        def guarded(user_id, bank_id, *args, _bank_lease=None, **kwargs):
            with bank_jobs.mutation_lease(
                    bank_id, kind, capability=_bank_lease) as lease:
                return fn(user_id, bank_id, *args, _bank_lease=lease, **kwargs)
        return guarded
    return decorate


def _job_bank_capability(job):
    """Return a modern registry capability, not a legacy test/job mapping."""
    return job if isinstance(job, dict) and '_keys' in job else None


def _walk_image_relpaths(folder, *, onerror=None):
    """Yield every image file under ``folder``, as a path RELATIVE to it.

    Same strings ``os.path.relpath(os.path.join(root, f), folder)`` returned,
    obtained by SLICING the walk root instead. ``os.walk`` builds each root by
    joining onto the folder we handed it, so the folder is always a literal
    prefix of the root — the relative part is already there and does not have to
    be recomputed from two absolute paths.

    That recomputation was the cost of the bank list: ``ntpath.relpath`` runs
    ``abspath`` on both sides and compares them component by component through
    ``os.path.normcase``, ~10 normcase calls per file. Measured on a real
    library of 8 banks / 86 493 images, one ``GET /api/banks``:
    85 821 relpath + 833 762 normcase calls, 2.0 s of the 3.5 s profile. Slicing
    makes it 0 relpath and one normcase per file (the dedup key, still needed).

    ``folder`` is normalised first, exactly as relpath would: a source_path
    spelled with a trailing separator or forward slashes must yield the SAME
    relpaths as before, or a refresh would re-insert the whole bank under
    differently spelled keys."""
    folder = os.path.normpath(folder)
    cut = len(folder) + (0 if folder.endswith(os.sep) else 1)
    for root, _dirs, files in os.walk(folder, onerror=onerror):
        head = root[cut:]
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                yield (head + os.sep + f) if head else f


def create_bank(user_id, name, folder):
    """Register a folder as a bank: walk it recursively and create one row per
    image file. Instant (no decode) — scoring is the separate scan pass.
    Returns (bank, added). ValueError on a missing folder / too many files."""
    name = (name or '').strip()
    # Windows «Copier en tant que chemin» pastes the path quoted — unquote so
    # the direct paste works first try (same nicety as the dataset folder import).
    folder = (folder or '').strip().strip('"\'')
    if not name:
        raise ValueError('name is required')
    if not folder or not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder or "(empty)"}')
    # A bank and a dataset only ever TRANSIT images (by copy) — they never share
    # them. This folder is the one place that door was open: a dataset's storage
    # folder pasted here made a bank over the dataset's LIVE files, and this
    # bank's 🗑 Delete rejected then deleted images out of the dataset.
    conflict = path_guard.dataset_folder_conflict(folder)
    if conflict:
        raise ValueError(conflict['message'])
    folder = os.path.realpath(folder)
    rels = []
    for rel in _walk_image_relpaths(folder):
        rels.append(rel)
        if len(rels) > BANK_MAX_FILES:
            # Bail on the walk itself: a bank pointed at a whole drive
            # must not be counted to the end before being refused.
            raise ValueError(
                f'this folder holds more than {BANK_MAX_FILES:,} images '
                '— point the bank at a subfolder, or split it in two')
    bank = ImageBank(user_id=user_id, name=name, source_path=folder)
    db.session.add(bank)
    db.session.flush()          # need bank.id for the child rows
    _insert_bank_images(bank.id, folder, rels)
    db.session.commit()
    return bank, len(rels)


_INSERT_CHUNK = 2000


def _insert_bank_images(bank_id, folder, rels, source_metadata_by_relpath=None) -> int:
    """Insert one BankImage per relpath, in chunks, through a CORE insert.

    Row-by-row ``db.session.add`` costs ~141 us/file (measured: 7.1 s for a
    50 000-image folder), which at the folder sizes this app now accepts would
    hold the HTTP request open for half a minute. The same rows through a
    chunked core insert cost a fraction of that, and Python-side column defaults
    (status, timestamps) are applied exactly as the ORM would.

    ``source_metadata_by_relpath`` is an optional {relpath: stored JSON string}
    map (already validated + serialized — see ``_source_metadata_storage``) for
    the scrape → bank intake, so a scraped image's provenance is not lost the
    moment it lands in a Bank folder and can later ride along on promotion to a
    Dataset (``_dataset_row_bank_values`` reads this same column back). Every
    row gets the key, missing entries as None, so all dicts in one chunked
    insert share the same columns."""
    lookup = source_metadata_by_relpath or {}
    rows = []
    for rel in rels:
        try:
            size = os.path.getsize(os.path.join(folder, rel))
        except OSError:
            size = None
        rows.append({'bank_id': bank_id, 'relpath': rel, 'file_size': size,
                     'source_metadata': lookup.get(rel)})
    # A "this subfolder is one person" assertion is a RULE, not a stamp: a file
    # that lands in an asserted folder joins its person group here, on insert,
    # with no pass and no click (see services/folder_person.py).
    from . import folder_person
    folder_person.stamp_new_rows(bank_id, rows)
    for i0 in range(0, len(rows), _INSERT_CHUNK):
        db.session.execute(BankImage.__table__.insert(),
                           rows[i0:i0 + _INSERT_CHUNK])
    return len(rows)


# --- folder sync (incremental re-inventory) ---------------------------------
# A bank points at a LIVE folder: the user keeps scraping/exporting into it long
# after the bank was created. Re-walking it is cheap (~5 ms for 3 000 files), so
# the app does it for them instead of making them rebuild a bank they have
# already triaged. The cooldown is not about CPU — it keeps the workspace's 2 s
# poll from hitting the disk (possibly a spun-down external drive) constantly.
FOLDER_SYNC_COOLDOWN = 60.0
_folder_sync = {}       # bank_id -> {'at': monotonic, 'result': {...}}
_EMPTY_SYNC = {'added': 0, 'missing': 0, 'unavailable': False, 'error': None,
               'not_added': 0, 'limit': BANK_MAX_FILES}


def reset_folder_sync():
    """Drop the per-bank walk cooldowns (tests: bank ids restart at 1 with an
    in-memory DB, so a stale entry would silently skip the next test's walk)."""
    _folder_sync.clear()


def _sync_cached(bank_id) -> dict:
    """The last known folder state, with the per-walk EVENTS zeroed — nothing was
    added, and nothing was left out, by the call being answered from the cache.
    Reporting them again would re-toast one walk's outcome on every 2 s poll."""
    last = _folder_sync.get(bank_id)
    return {**(last['result'] if last else _EMPTY_SYNC), 'added': 0, 'not_added': 0}


def refresh_bank(user_id, bank_id, force=False, *, _bank_lease=None,
                 source_metadata_by_relpath=None) -> dict | None:
    """Re-inventory a bank's source folder: register the images that appeared in
    it since the last walk.

    STRICTLY ADDITIVE — the only write is an INSERT of relpaths we don't know
    yet. No row is ever deleted and no decision is ever reset (status, scores,
    quality_state, duplicate/semantic groups, captions, face verdicts), so a
    bank triaged over hours survives any number of refreshes. New rows land
    exactly like freshly inventoried ones (pending, unscanned), which is all the
    downstream passes need: the quality scan already only picks up rows with no
    quality_state, and it rebuilds the duplicate groups when it lands.

    Files that VANISHED from the folder are counted, never removed: an unplugged
    drive or a renamed folder would otherwise wipe a whole triage in one silent
    pass. The count is surfaced so the user can decide.

    The BANK_MAX_FILES ceiling is counted against what the walk found ON DISK,
    never against the bank's row history, and it no longer refuses the batch: as
    many new files as fit are added and the remainder is reported in
    ``not_added``. See the comment at the check for the bug that motivated both.

    Returns {'added', 'missing', 'unavailable', 'error', 'not_added', 'limit'},
    or None when the bank is unknown. ``force`` bypasses the cooldown (bank
    opened by hand). ``source_metadata_by_relpath`` — see ``_insert_bank_images``
    — is threaded through only for the scrape → bank intake; every other caller
    omits it and new rows get no provenance, exactly as before."""
    if _bank_lease is None:
        try:
            with bank_jobs.mutation_lease(bank_id, 'folder_sync') as lease:
                return refresh_bank(
                    user_id, bank_id, force=force, _bank_lease=lease,
                    source_metadata_by_relpath=source_metadata_by_relpath)
        except bank_jobs.BankJobBusy:
            # Polling remains readable while a pass owns the Bank; inventory is
            # additive and can safely wait for the next refresh.
            return _sync_cached(bank_id)
    bank_jobs.require_reservation(_bank_lease, bank_id)
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    now = time.monotonic()
    last = _folder_sync.get(bank_id)
    if not force and last and (now - last['at']) < FOLDER_SYNC_COOLDOWN:
        return _sync_cached(bank_id)
    # A live pass owns this bank's rows (the scan job works off a snapshot of
    # them and reports progress against a fixed total). Adding rows underneath
    # it is harmless for the data but would silently fall outside that total —
    # the next refresh, a second later, picks them up.
    folder = bank.source_path
    if not folder or not os.path.isdir(folder):
        return _remember_sync(bank_id, now, {**_EMPTY_SYNC, 'unavailable': True})

    known = {os.path.normcase(rel) for (rel,) in
             db.session.query(BankImage.relpath).filter_by(bank_id=bank_id)}
    seen, new_rels = set(), []
    try:
        for rel in _walk_image_relpaths(folder, onerror=lambda _e: None):
            key = os.path.normcase(rel)
            seen.add(key)
            if key not in known:
                new_rels.append(rel)
    except OSError:
        # The folder went away mid-walk (drive unplugged) — report it and keep
        # every row: a partial walk must never be read as "these files are gone".
        return _remember_sync(bank_id, now, {**_EMPTY_SYNC, 'unavailable': True})

    new_rels.sort()             # deterministic order — a partial add takes a PREFIX
    # The cap is a property of the FOLDER, not of the bank's history. ``seen`` is
    # what the walk just found on disk (surviving rows + new files); rows whose
    # file the user deleted are NOT in it and must not consume the budget.
    #
    # THE BUG THIS REPLACES: the test was `len(known) + len(new_rels) > cap`,
    # where `known` is every relpath ever inventoried — refresh is deliberately
    # additive and never drops a row for a vanished file. So a folder that once
    # held 50 000 images and now holds 10 000 still counted as 50 000 for ever,
    # and the refusal told the user "the folder now holds more than 50000
    # images" about a folder holding a fifth of that. Measured before the fix:
    # 10 001 files on disk, 50 000 rows, one new file → refused.
    not_added = 0
    over = len(seen) - BANK_MAX_FILES
    if over > 0:
        # Non-blocking: add what fits (oldest-first by name) instead of refusing
        # the whole batch. Everything already in the bank keeps working, and the
        # count that was left out is reported so the user can act on it.
        keep = max(len(new_rels) - over, 0)
        not_added = len(new_rels) - keep
        new_rels = new_rels[:keep]
    if new_rels:
        _insert_bank_images(bank_id, folder, new_rels, source_metadata_by_relpath)
        db.session.commit()
    return _remember_sync(bank_id, now, {
        'added': len(new_rels),
        'missing': sum(1 for k in known if k not in seen),
        'unavailable': False, 'error': None,
        # Reported every time so the UI can phrase a partial add honestly.
        'not_added': not_added, 'limit': BANK_MAX_FILES})


def _remember_sync(bank_id, at, result) -> dict:
    _folder_sync[bank_id] = {'at': at, 'result': result}
    return dict(result)


def folder_sync_state(user_id) -> dict:
    """What the bank LIST can say about every source folder WITHOUT walking it —
    {bank_id: folder_sync}. The cheap counterpart of ``refresh_banks``.

    The list used to force a full re-walk of every bank's folder on every load.
    That is a side effect a navigation should not pay for: measured on a real
    library of 8 banks / 86 493 images it cost 690-1 190 ms per load (1 341 to
    1 777 ms on the reporter's own instance), just to open a page the user may
    only be passing through.

    What is kept here is what costs one syscall per BANK instead of one per
    image: whether the folder is still there at all. That is the warning that
    actually matters on this page (an unplugged drive, a renamed folder), and it
    is now reported even on a cold process, where the walk-based version had
    nothing cached to report. The rest of the last known walk (``missing``) rides
    along, with the per-walk EVENTS zeroed — nothing was added by a call that
    did not look.

    ``walked`` says whether these numbers come from a walk at all, and ``age``
    how old it is (seconds, None when never), so the page can SAY that its
    counts may lag instead of quietly showing stale ones. The walk itself is one
    click away (``?rescan=1``) and still automatic when a bank is opened."""
    now = time.monotonic()
    out = {}
    for bank_id, source_path in (
            ImageBank.query
            .with_entities(ImageBank.id, ImageBank.source_path)
            .filter_by(user_id=user_id).all()):
        last = _folder_sync.get(bank_id)
        state = {**(last['result'] if last else _EMPTY_SYNC),
                 'added': 0, 'not_added': 0}
        # The probe beats the cache in BOTH directions: a folder that went away
        # since the last walk is reported now, and one that came back stops
        # being reported as gone.
        state['unavailable'] = not (source_path and os.path.isdir(source_path))
        state['walked'] = last is not None
        state['age'] = round(now - last['at'], 1) if last else None
        out[bank_id] = state
    return out


def refresh_banks(user_id, force=False) -> dict:
    """refresh_bank() over every bank of the user — {bank_id: result}. The
    EXPLICIT rescan behind the bank list's 🔄 button (and nothing else): it
    forces the walk, because a user who just clicked it dropped files in a
    folder a second ago and the cooldown would swallow exactly that.

    Not what a plain page load runs any more — see ``folder_sync_state`` for why
    and for what the list says instead. A bank whose folder is unavailable
    simply reports it; it never fails the list."""
    out = {}
    ids = [row.id for row in ImageBank.query.with_entities(ImageBank.id)
           .filter_by(user_id=user_id).all()]
    for bank_id in ids:
        res = refresh_bank(user_id, bank_id, force=force)
        if res is not None:
            out[bank_id] = res
    # SAME SHAPE as folder_sync_state, or the page that renders both would call
    # the answer to a rescan stale — it did, until a screenshot showed the note
    # still reading "what the app knew last time" right under the toast saying
    # the folders had just been checked. A bank a running pass owns is not
    # walked even under force, and reports its real (older) age.
    now = time.monotonic()
    for bank_id, res in out.items():
        last = _folder_sync.get(bank_id)
        res['walked'] = last is not None
        res['age'] = round(now - last['at'], 1) if last else None
    return out


# --- relocate ---------------------------------------------------------------
_MISSING_SAMPLE = 8         # relpaths quoted back so the user can recognise them


class BankRelocateMismatch(ValueError):
    """The candidate folder holds none of the bank's files. Carries the counts
    so the route can report them instead of a bare sentence."""
    def __init__(self, message, preview):
        super().__init__(message)
        self.preview = preview


def _relocate_target(folder) -> str:
    """Normalise a pasted folder into an absolute path we can walk. Same
    unquoting nicety as create_bank (Windows «Copy as path» pastes quoted)."""
    folder = (folder or '').strip().strip('"\'')
    if not folder:
        raise ValueError('a folder is required')
    if not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder}')
    # Relocation is the SAME door as creation, just later: repointing a bank at a
    # dataset's storage folder shares the files exactly as creating it there
    # would. Refused in the dry run too, so the dialog never offers the move.
    conflict = path_guard.dataset_folder_conflict(folder)
    if conflict:
        raise ValueError(conflict['message'])
    return os.path.realpath(folder)


def relocate_preview(user_id, bank_id, folder) -> dict:
    """Dry-run a relocation: how much of THIS bank is in THAT folder?

    Nothing is written. Every known relpath is looked up under the candidate
    folder (one os.walk, matched case-insensitively through normcase — same
    keying as refresh_bank, so a drive letter or a folder re-created in another
    case still counts as the same tree). Returns {folder, total, found, missing,
    missing_sample, extra, same_folder}; ValueError on an unknown bank/folder.

    The point of the two numbers is that the user decides: a bank moved whole
    reads 29 759 found / 0 missing, and a mistyped folder reads 0 / 29 759 — a
    distinction the caller must never make silently on their behalf."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        raise ValueError('bank not found')
    target = _relocate_target(folder)
    known = {}
    for (rel,) in db.session.query(BankImage.relpath).filter_by(bank_id=bank_id):
        known.setdefault(os.path.normcase(rel), rel)
    seen = {os.path.normcase(rel) for rel in
            _walk_image_relpaths(target, onerror=lambda _e: None)}
    hit = seen & set(known)
    gone = sorted(known[k] for k in set(known) - hit)
    return {
        'folder': target,
        'total': len(known),
        'found': len(hit),
        'missing': len(gone),
        'missing_sample': gone[:_MISSING_SAMPLE],
        'extra': len(seen - hit),
        'same_folder': os.path.normcase(target) == os.path.normcase(
            os.path.realpath(bank.source_path or '')),
    }


@_serialized_bank_mutation('relocate')
def relocate_bank(user_id, bank_id, folder, confirm=False, *,
                  _bank_lease=None) -> dict:
    """Point a bank at a NEW folder, keeping every row and every analysis.

    Moving a bank costs nothing by construction: BankImage.relpath is relative
    to source_path, and scores / dhash / duplicate groups / face verdicts /
    captions / keep-reject decisions all hang off the row id. So this only
    rewrites ONE string — the danger is never the write, it is aiming it wrong.

    Hence: no call applies anything without ``confirm``; a folder that holds
    NONE of the bank's files is refused outright (that is a different folder,
    not a moved one); and a partial match goes through but deletes nothing —
    rows whose file did not come along keep their analysis and simply read as
    missing in the folder-sync note. Returns the preview dict plus
    {'applied', 'needs_confirm', 'overlaps'}."""
    out = relocate_preview(user_id, bank_id, folder)
    out['needs_confirm'] = out['missing'] > 0
    out['applied'] = False
    if out['total'] and not out['found']:
        raise BankRelocateMismatch(
            'none of this bank\'s '
            f"{out['total']} image(s) are in that folder — it does not look "
            'like this bank. Pick the folder that CONTAINS the images '
            '(the one you moved), not its parent.', out)
    if not confirm:
        return out
    bank = get_bank(user_id, bank_id)
    bank.source_path = out['folder']
    db.session.commit()
    _folder_sync.pop(bank_id, None)     # next walk must see the new folder
    out['applied'] = True
    out['overlaps'] = overlapping_banks(user_id, bank_id)
    return out


class BankFolderUnavailable(RuntimeError):
    """The bank's source folder cannot be walked at all right now — so nothing
    may be forgotten: an unplugged drive reads as "every file is missing"."""


def _rows_missing_on_disk(bank):
    """A FRESH walk's verdict: ([(id, relpath)] whose file is not in the folder,
    how many rows' files are). Sorted by relpath so samples read naturally.

    Fail closed, unlike refresh_bank's walk: refresh is additive and can afford
    to shrug at an unreadable subfolder, but the caller HERE is about to delete
    rows for every file the walk did not find — a directory it could not read
    must abort the whole verdict, never read as "those files are gone"."""
    folder = bank.source_path
    if not folder or not os.path.isdir(folder):
        raise BankFolderUnavailable(
            'the source folder is unavailable (moved, renamed or on a '
            'disconnected drive) — nothing was forgotten. Reconnect it, or use '
            'Move folder… if it lives somewhere else now.')

    def _abort(err):
        raise err
    try:
        seen = {os.path.normcase(rel)
                for rel in _walk_image_relpaths(folder, onerror=_abort)}
    except OSError as e:
        raise BankFolderUnavailable(
            'part of the source folder could not be read '
            f'({e}) — nothing was forgotten.') from e
    rows = (db.session.query(BankImage.id, BankImage.relpath)
            .filter_by(bank_id=bank.id).all())
    gone = sorted(((rid, rel) for rid, rel in rows
                   if os.path.normcase(rel) not in seen), key=lambda t: t[1])
    return gone, len(rows) - len(gone)


def forget_missing_preview(user_id, bank_id) -> dict:
    """What 🧹 Forget missing would drop, counted by a walk done NOW — the
    folder-sync banner's number can be a cooldown old, and the confirmation
    dialog must show the count the delete would actually use. Read-only.
    Raises BankFolderUnavailable when the folder cannot be walked, ValueError
    on an unknown bank. Returns {'missing', 'present', 'missing_sample'}."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        raise ValueError('bank not found')
    gone, present = _rows_missing_on_disk(bank)
    return {'missing': len(gone), 'present': present,
            'missing_sample': [rel for _rid, rel in gone[:_MISSING_SAMPLE]]}


@_serialized_bank_mutation('forget_missing')
def forget_missing(user_id, bank_id, *, _bank_lease=None) -> dict:
    """Drop the rows whose source file is no longer in the folder.

    The folder-sync warning's OTHER remedy. 📦 Move folder… answers "the folder
    moved"; this answers "the files are really gone" — a downloader that cleans
    up its own intermediates, a sync client, a by-hand tidy of the folder. Those
    rows fail to load for ever and keep counting against the bank's ceiling.

    Only database rows are touched — the whole premise is that the files are
    already gone, so there is nothing on disk to delete. The dropped rows take
    their decisions and analyses with them, which is what the confirmation
    dialog says out loud. Rows whose file IS on disk are never touched.

    The verdict comes from a fresh fail-closed walk (see _rows_missing_on_disk):
    an unavailable folder or a read error refuses the whole operation, because
    an unplugged drive must never be able to erase a triage. Deletion commits
    chunk by chunk like delete_rejected, so an interruption leaves a consistent
    bank that has simply forgotten fewer rows. Returns {'removed', 'remaining'}."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        raise ValueError('bank not found')
    gone, present = _rows_missing_on_disk(bank)
    ids = [rid for rid, _rel in gone]
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        BankImage.query.filter(
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])
        ).delete(synchronize_session=False)
        db.session.commit()
    if ids:
        # The pending ↩ offer may point at rows this run just dropped —
        # withdraw it rather than advertise a restore that would find nothing.
        bank_undo.clear(bank_id)
        _folder_sync.pop(bank_id, None)   # next poll re-walks; the banner clears
    return {'removed': len(ids), 'remaining': present}


def _is_imported_source(path) -> bool:
    """True when the bank's folder is one WE made ("Import to bank"), i.e. it sits
    under bank_sources_root — as opposed to a folder of the user's own that a bank
    merely points at, which we must never touch."""
    try:
        root = os.path.realpath(cfg.bank_sources_root())
        p = os.path.realpath(str(path or ''))
        # commonpath RAISES on Windows when the two paths sit on different drives
        # ("Paths don't have the same drive") — and a bank pointing at another disk
        # is precisely the common case. Raising here turned every such delete into
        # a 500. Different drive == certainly not under our root, so: False.
        return bool(p) and os.path.commonpath([root, p]) == root and p != root
    except (OSError, ValueError):
        return False


def _remove_partial_import_folder(path, *, context) -> bool:
    """Permanently remove an app-owned *partial* import folder, fail closed.

    This is deliberately narrower than ordinary Bank deletion, which prefers a
    recoverable move to Trash.  Failed/cancelled transfers have no user-owned
    result to recover, though, and leaving their full copy behind is both an
    orphan and a false "discarded" claim.  Resolve the target immediately before
    removal and accept only a strict child of ``bank_sources_root``; an arbitrary
    user source (or the import root itself) is never handed to ``rmtree``.
    """
    raw = str(path or '')
    if not raw:
        return False
    if not os.path.lexists(raw):
        return True
    try:
        resolved = os.path.realpath(raw)
    except (OSError, TypeError, ValueError):
        return False
    if not _is_imported_source(resolved):
        logger.error('%s: refusing to remove an unbounded folder %r', context, raw)
        return False
    try:
        shutil.rmtree(resolved)
    except Exception:  # noqa: BLE001 — cleanup must report every removal failure
        logger.error('%s: could not remove partial import folder %s',
                     context, resolved, exc_info=True)
        return False
    return not os.path.lexists(resolved)


@_serialized_bank_mutation('delete')
def delete_bank(user_id, bank_id, *, _allow_busy=False,
                _bank_lease=None) -> bool:
    """Drop the bank's ROWS and working data (thumbs + face cache). A folder of the
    user's OWN and its images are never touched.

    The one exception is a bank built by "Import to bank": its folder is a copy WE
    made under bank_sources_root, so deleting the bank must take it too — otherwise
    a full duplicate of the dataset stays on disk forever with nothing in the UI
    pointing at it. It goes to Trash, not unlink, so it stays recoverable."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return False
    imported_source = bank.source_path if _is_imported_source(bank.source_path) else None
    from . import folder_person
    folder_person.drop_for_bank(bank_id)   # children first — no relationship()
    folder_person.drop_probes_for_bank(bank_id)
    BankImage.query.filter_by(bank_id=bank_id).delete(synchronize_session=False)
    db.session.delete(bank)
    db.session.commit()
    bank_undo.clear(bank_id)     # its rows are gone; a stale offer would outlive them
    reset_score_memo()           # ~45 MB of embeddings for a bank that no longer is
    shutil.rmtree(_bank_dir(bank_id), ignore_errors=True)
    if imported_source and os.path.isdir(imported_source):
        try:
            trash.send_to_trash(imported_source, context=f'bank-{bank_id}')
        except OSError:
            logger.warning('delete_bank: could not trash the imported copy %s',
                           imported_source, exc_info=True)
    return True


# --- flags & payloads -------------------------------------------------------
def _watermark_history_inactive(row: BankImage) -> bool:
    """Whether copied watermark history is known not to describe this payload.

    A transformed source Bank legitimately has two identities: watermark data
    belongs to its pristine raw file while effective analysis belongs to its
    clean/rotated derivative.  After promotion that derivative is baked into a
    new Bank's raw file and the transform markers are intentionally removed; an
    unequal pair of exact fingerprints is then historical evidence only.
    """
    if row.watermark_clean_method or row.rotation:
        return False
    if not _valid_analysis_fingerprint(row.watermark_fingerprint):
        return True
    return bool(
        _valid_analysis_fingerprint(row.analysis_fingerprint)
        and row.analysis_fingerprint != row.watermark_fingerprint)


def _watermark_history_inactive_clause():
    """SQL mirror of :func:`_watermark_history_inactive`."""
    return and_(
        BankImage.watermark_clean_method.is_(None),
        BankImage.rotation.is_(None),
        or_(
            BankImage.watermark_fingerprint.is_(None),
            func.length(BankImage.watermark_fingerprint) != 64,
            and_(
                BankImage.analysis_fingerprint.isnot(None),
                func.length(BankImage.analysis_fingerprint) == 64,
                BankImage.watermark_fingerprint.isnot(None),
                func.length(BankImage.watermark_fingerprint) == 64,
                BankImage.analysis_fingerprint
                != BankImage.watermark_fingerprint,
            ),
        ),
    )


def image_flags(row: BankImage, th: dict) -> list:
    """Threshold verdicts for one image, recomputed from the raw scores."""
    if row.quality_state == 'unreadable':
        return ['unreadable']
    flags = []
    if row.quality_state == 'ok':
        if row.blur_score is not None and row.blur_score < th['sharpness_min']:
            flags.append('blur')
        if row.noise_score is not None and row.noise_score > th['noise_max']:
            flags.append('noise')
        if row.uniformity_score is not None and row.uniformity_score < th['uniformity_min']:
            flags.append('uniform')
        if row.width and row.height and min(row.width, row.height) < th['min_side']:
            flags.append('small')
        # Effective resolution — the picture stops before the pixels do. Same
        # read-time-verdict philosophy as blur: raw score in, threshold applied
        # here, so retuning detail_min re-sorts the bank with no rescan.
        if row.detail_ratio is not None and row.detail_ratio < th['detail_min']:
            flags.append('soft_detail')
        if row.bars_ratio is not None and row.bars_ratio > th['bars_max']:
            flags.append('bars')
    # V2 scoring flags — derived from the persisted scores against the live
    # thresholds too, but NOT gated on the quality state (a watermarked or NSFW
    # image can be perfectly sharp). Only present once the relevant pass has run.
    if row.aesthetic_score is not None and row.aesthetic_score < th['aesthetic_min']:
        flags.append('low_aesthetic')
    if row.nsfw_score is not None and row.nsfw_score > th['nsfw_max']:
        flags.append('nsfw')
    if (row.watermark_state == 'detected'
            and not _watermark_history_inactive(row)):
        flags.append('watermark')
    return flags


def _promoted_dataset_by_image(image_ids) -> dict:
    """{bank_image_id: dataset_id} for the images a dataset REALLY holds right
    now, read off the back-links. Only the ids of the page being rendered, so a
    30 000-image bank still costs one small query. An image promoted into
    several datasets reports the lowest id — the ⬆ badge only says THAT it
    landed somewhere, and a stable pick keeps the grid from flickering."""
    out: dict = {}
    ids = [int(i) for i in image_ids]
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = (db.session.query(FaceDatasetImage.bank_image_id,
                                 func.min(FaceDatasetImage.dataset_id))
                .filter(FaceDatasetImage.bank_image_id.in_(ids[i0:i0 + _SQL_IN_CHUNK]))
                .group_by(FaceDatasetImage.bank_image_id).all())
        out.update({bid: ds for bid, ds in rows})
    return out


def _page_images(rows, th: dict) -> list:
    """One page of grid payloads, with the ⬆ promoted state resolved in a single
    extra query for the whole page (never one per row)."""
    promoted_by = _promoted_dataset_by_image([r.id for r in rows])
    return [_image_dict(r, th, promoted_by) for r in rows]


def _image_dict(row: BankImage, th: dict, promoted_by: dict | None = None) -> dict:
    # ⬆ promoted = the dataset that holds this image TODAY (back-link), falling
    # back to the legacy one-way flag for promotions that predate it. Deriving it
    # means the badge disappears when the user deletes the image in the dataset,
    # instead of advertising a copy that is gone.
    promoted = (promoted_by or {}).get(row.id, row.promoted_dataset_id)
    # Current scans bind width/height to the same EFFECTIVE bytes as every other
    # analysis lane, so a SHA-bound row is already in display orientation.  Only
    # a legacy, unbound rotated row may still carry raw inventory dimensions and
    # needs the historical read-time transpose.
    rotation = int(row.rotation or 0) % 360
    legacy_raw_dimensions = (
        rotation in (90, 270)
        and not _valid_analysis_fingerprint(row.analysis_fingerprint))
    width, height = ((row.height, row.width) if legacy_raw_dimensions
                     else (row.width, row.height))
    # The mask editor's seed, carried ONLY on the rows that can open it: a bank
    # page is thousands of images and the other 99% would pay for three null keys.
    watermark_active = not _watermark_history_inactive(row)
    mask = {}
    if watermark_active and (
            row.watermark_state == 'detected'
            or row.watermark_regions is not None):
        import json as _json
        try:
            bbox = _json.loads(row.watermark_bbox or '')
        except (ValueError, TypeError):
            bbox = None
        mask = {'watermark_bbox': bbox if isinstance(bbox, list) and len(bbox) == 4
                else None,
                **_watermark_regions_payload(row)}
    return {
        **mask,
        'id': row.id,
        'name': os.path.basename(row.relpath),
        'relpath': row.relpath,
        'rotation': rotation,
        # ✂/✨ made in the Bank. The GENERATION travels with the marker because
        # the thumbnail route is cached for an hour under an URL that does not
        # change on its own — it is what the grid busts that cache with.
        'edit_method': row.edit_method,
        'edit_generation': int(row.edit_generation or 0),
        'width': width, 'height': height, 'file_size': row.file_size,
        'quality_state': row.quality_state,
        'blur_score': row.blur_score, 'noise_score': row.noise_score,
        'uniformity_score': row.uniformity_score,
        'aesthetic_score': row.aesthetic_score, 'nsfw_score': row.nsfw_score,
        'style_cluster': row.style_cluster,
        'watermark_state': row.watermark_state if watermark_active else None,
        'watermark_clean_method': row.watermark_clean_method,
        # Who ruled on THIS image, and with what score. Per image rather than per
        # bank because a bank scanned over weeks holds both routes' verdicts, and
        # "why is this one flagged?" is asked about one tile, not about a bank.
        # NULL source = scanned before the app recorded it; NULL score = the
        # vision route, which writes a sentence and not a number.
        'watermark_source': row.watermark_source if watermark_active else None,
        'watermark_score': row.watermark_score if watermark_active else None,
        'detail_ratio': row.detail_ratio, 'bars_ratio': row.bars_ratio,
        'jpeg_quality': row.jpeg_quality,
        'origin': row.origin, 'origin_evidence': row.origin_evidence,
        'subfolder': _subfolder_of(row.relpath),
        'flags': image_flags(row, th),
        'dup_group': row.dup_group,
        'semantic_dup_group': row.semantic_dup_group,
        'face_state': row.face_state, 'face_cluster': row.face_cluster,
        # 'asserted' = the person id came from the user's "this subfolder is one
        # person", not from an embedding. The grid says so rather than passing a
        # declaration off as a measurement.
        'face_cluster_origin': row.face_cluster_origin,
        'framing': row.framing,
        # 🎨 what the picture is made of, and how sure the classifier was. The
        # margin travels with the verdict on purpose: a tile badge that cannot
        # show its own confidence is how a guess becomes a fact.
        'medium': row.medium, 'medium_margin': row.medium_margin,
        # ⤢ the raw head yaw in degrees, signed as the detector reports it. The
        # BUCKET is not stored here — it is derived from this number by the same
        # two cuts the SQL uses, so a re-tune can never leave the badge and the
        # chip disagreeing.
        'face_yaw': row.face_yaw,
        'status': row.status, 'reject_reason': row.reject_reason,
        'promoted_dataset_id': promoted,
        # The OTHER destination. Kept as its own key rather than overloading the
        # dataset one, which is stored in user databases and read as a dataset id.
        'promoted_bank_id': row.promoted_bank_id,
        'caption': row.caption,
        # WHO wrote that caption ('asserted' | 'joycaption' | 'ollama' | NULL =
        # never recorded — services/caption_origin.py). Sent with the text and
        # not derived from the current backend setting: the 'auto' backend CHAINS
        # two engines inside one run, so a bank holds both, and the settings value
        # names a policy rather than the writer of any one sentence.
        'caption_origin': row.caption_origin,
    }


def _flag_filter(flag: str, th: dict):
    """SQLAlchemy criterion for one flag name (mirrors image_flags)."""
    if flag == 'unreadable':
        return BankImage.quality_state == 'unreadable'
    # V2 scoring flags — not gated on quality_state (see image_flags) and only
    # true where the score actually exists (a NULL score is "not scored", never
    # "below threshold"). watermark is a discrete state, no threshold column.
    if flag == 'low_aesthetic':
        return and_(BankImage.aesthetic_score.isnot(None),
                    BankImage.aesthetic_score < th['aesthetic_min'])
    if flag == 'nsfw':
        return and_(BankImage.nsfw_score.isnot(None),
                    BankImage.nsfw_score > th['nsfw_max'])
    if flag == 'watermark':
        return and_(BankImage.watermark_state == 'detected',
                    ~_watermark_history_inactive_clause())
    ok = BankImage.quality_state == 'ok'
    crit = {
        'blur': BankImage.blur_score < th['sharpness_min'],
        'noise': BankImage.noise_score > th['noise_max'],
        'uniform': BankImage.uniformity_score < th['uniformity_min'],
        'small': or_(BankImage.width < th['min_side'],
                     BankImage.height < th['min_side']),
        # NULL-safe: a row scanned before the provenance pass existed carries no
        # score, and "not measured" must never read as "below threshold".
        'soft_detail': and_(BankImage.detail_ratio.isnot(None),
                            BankImage.detail_ratio < th['detail_min']),
        'bars': and_(BankImage.bars_ratio.isnot(None),
                     BankImage.bars_ratio > th['bars_max']),
    }.get(flag)
    return (ok & crit) if crit is not None else None


def _clean_criterion(th):
    """quality_state 'ok' AND none of the six metric flags — THE definition of
    the ✨ Clean chip, shared by its filter and its counter so the number on
    the chip and the grid it opens can never disagree. 'unreadable' needs no
    clause: it IS a quality_state, excluded by the 'ok' pin. Each negated
    criterion is NULL-safe where the score can be absent (soft_detail, bars
    carry their own isnot(None)); blur/noise/uniformity are written together
    with quality_state='ok' by the scan, so they are never NULL on 'ok' rows.
    """
    return and_(BankImage.quality_state == 'ok',
                *[~_flag_filter(f, th)
                  for f in ('blur', 'noise', 'uniform', 'small',
                            'soft_detail', 'bars')])


_QUALITY_FLAGS = ('blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars',
                  'unreadable')
# V2 score-derived flags. Kept separate from _QUALITY_FLAGS so the "flagged" /
# "clean" quality aggregate stays about the CPU quality pass, while these count
# and filter independently (each only meaningful once its pass has run).
_SCORE_FLAGS = ('low_aesthetic', 'nsfw', 'watermark')

# Resolution tiers for the Bank grid — bucketed on MEGAPIXELS (width×height, the
# same rank as the resolution sort) so a mixed dump can be skimmed and mass-acted
# one tier at a time. Each entry is (stable_id, lo, hi) in raw pixels, a HALF-OPEN
# [lo, hi) range (lower-inclusive, upper-exclusive); hi=None means "no ceiling".
# So a 1000×1000 (1.00 MP) and a 1024×1024 (1.05 MP) both land in 'res_1_2', and a
# 2000×2000 (4.00 MP) lands in 'res_gt_4'. The 0.25 MP floor (not 0.30) is chosen
# so a 512×512 (0.26 MP) — a legit small training crop — sits in '0.25–1 MP', while
# only true junk (Telegram thumbnails ~0.1–0.2 MP, ≤448²) falls in '< 0.25 MP'.
# Ids are user-facing filter keys — never rename without an alias.
_RES_BUCKETS = (
    ('res_lt_025', 0, 250_000),
    ('res_025_1', 250_000, 1_000_000),
    ('res_1_2', 1_000_000, 2_000_000),
    ('res_2_4', 2_000_000, 4_000_000),
    ('res_gt_4', 4_000_000, None),
)
_RES_BOUNDS = {bid: (lo, hi) for bid, lo, hi in _RES_BUCKETS}

# Framing buckets — the SAME four shot types the datasets classify (face close-up,
# bust, full body, back view). 'unknown' is a parseable-but-not-one-of-four answer;
# it counts and filters but is never a training target. Ids are user-facing filter
# keys — never rename without an alias. The built-in character composition aims for
# a 12/6/6/1 face/bust/body/back mix; the coverage advice phrases against that
# proportion, never as a hard rule.
_FRAMINGS = ('face', 'bust', 'body', 'back')
_FRAMING_KEYS = _FRAMINGS + ('unknown',)
_FRAMING_TARGET = {'face': 12, 'bust': 6, 'body': 6, 'back': 1}


# --- 🎨 Medium: what the picture is MADE OF ---------------------------------
# A DIFFERENT question from `origin`, which reads the file's metadata. origin
# answers "who made this file" (a generator, a camera, or — usually — nobody can
# tell); medium answers "what does this picture look like it is". An AI-generated
# photorealistic portrait is origin='ai' AND medium='photo'; a scanned manga page
# is origin='unknown' AND medium='anime'. Neither implies the other and the UI
# must never present one as evidence for the other.
#
# Ids are user-facing filter keys AND stored column values — never rename one
# without an alias.
MEDIUMS = ('photo', 'anime', 'render3d', 'illustration')
MEDIUM_KEYS = MEDIUMS + ('unsure',)

# Zero-shot CLIP prototypes. Each bucket is the MEAN of its phrases, re-normed
# (standard prompt ensembling) — one vector per bucket, encoded ONCE per install
# and then cached on disk forever by clip_text_encoder.
#
# The two keys starting with '_' are DISTRACTORS and are deliberately NOT
# buckets: a forced four-way choice has no way to say "none of these", so every
# banner, website screenshot and thumbnail collage in a scrape dump had to land
# in one of the four — measured, that was the single biggest source of wrong
# verdicts (a '3D render' pile that was mostly text banners). Giving the junk
# somewhere honest to go and mapping it to 'unsure' removed 769 wrong verdicts
# out of 23 532 images on the reference bank without costing a single right one.
MEDIUM_PROTOTYPES = {
    'photo': ('a photograph', 'a photo of a real person',
              'a photograph taken with a camera', 'a real-life photo',
              'a candid snapshot of real people'),
    'anime': ('an anime drawing', 'an anime style illustration', 'a manga panel',
              'a cartoon character drawn in anime style', 'anime artwork'),
    'render3d': ('a 3D render', 'a 3D computer graphics render', 'a CGI render',
                 'a rendered 3D character', 'a videogame screenshot'),
    'illustration': ('an illustration', 'a digital painting', 'a pencil drawing',
                     'a painted artwork', 'a comic book drawing'),
    '_text': ('a text banner', 'a poster with large text', 'a logo',
              'an advertisement banner with writing', 'a page of text'),
    '_screen': ('a screenshot of a website', 'a screenshot of an app interface',
                'a computer desktop screenshot', 'a collage of thumbnails',
                'a photo grid montage'),
}

# The two cuts that turn a ranking into a VERDICT, and the reason they are not
# one number. Both are a cosine MARGIN — the winning prototype's similarity minus
# the runner-up's — never an absolute similarity, because this project has
# already measured (see search_by_text) that no absolute CLIP cut separates
# "relevant" from "unrelated" on a real corpus.
#
# MEASURED, on the reference machine's 23 532-image bank, against 167 images
# labelled BY EYE from contact sheets (100 uniformly random + the 25 strongest
# candidates of each non-photo bucket):
#   * at no cut at all, the four-way argmax is 99/100 right on the random sample
#     but only 2/25 right on its own top 'anime' picks and 4/25 on 'render3d' —
#     because CLIP reads a picture's SUBJECT as much as its medium, so a
#     photograph of somebody cosplaying an anime character scores as 'anime'.
#     That confusion is the whole reason the non-photo bar is where it is.
#   * photo verdicts survive a low bar: 0.005 keeps 90 of the 159 photographs
#     and got none of them wrong.
#   * non-photo verdicts need a bar six times higher: at 0.030 the pass named the
#     2 real anime drawings and nothing else, with zero false positives; at 0.020
#     it also named 3 cosplay photographs and a text banner.
# The result is a classifier that is almost never wrong and often silent — which
# is the trade this app takes everywhere else too. On that bank it answers
# photo for 21 138 images, anime for 2, and 'unsure' for 2 392. The 'unsure'
# pile is a REAL answer, and the UI says how big it is rather than hiding it.
MEDIUM_MARGIN_PHOTO = 0.005
MEDIUM_MARGIN_OTHER = 0.030


def medium_verdict(sims: dict) -> tuple:
    """(medium, margin) from {bucket: cosine} — the ONE place the cuts are
    applied, so the pass, the tests and any future re-tune read the same rule.

    A distractor winning, or a margin under this bucket's cut, is 'unsure'. The
    margin is returned either way: it is what makes the verdict re-tunable
    without recomputing a single embedding."""
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    if len(ranked) < 2:
        return 'unsure', None
    name, best = ranked[0]
    margin = float(best - ranked[1][1])
    if name.startswith('_'):
        return 'unsure', margin
    cut = MEDIUM_MARGIN_PHOTO if name == 'photo' else MEDIUM_MARGIN_OTHER
    return (name if margin >= cut else 'unsure'), margin


def _medium_counts(bank_id) -> dict:
    """Per-bucket image counts for the 🎨 Medium chips in ONE GROUP BY. Rows with
    a NULL medium (never classified — no ✨ Score embedding to read) are excluded:
    "not classified yet" is a different statement from "unsure", and merging them
    would let an unrun pass look like an undecided one."""
    q = (db.session.query(BankImage.medium, func.count(BankImage.id))
         .filter(BankImage.bank_id == bank_id, BankImage.medium.isnot(None)))
    got = {k: n for k, n in q.group_by(BankImage.medium).all()}
    return {k: int(got.get(k, 0)) for k in MEDIUM_KEYS}


# --- ⤢ Angle: where the head is pointing ------------------------------------
# Measured IN THE PIXELS by the 🎭 Faces pass (InsightFace/antelopev2 estimates a
# head pose from its five landmarks), never guessed from a caption. Only the
# ABSOLUTE yaw is ever read: "turned left" and "turned right" are the same shot
# type for a training set, and treating them as two would halve every count for
# no gain.
#
# Ids are user-facing filter keys — never rename without an alias.
ANGLES = ('frontal', 'three_quarter', 'profile', 'behind')

# MEASURED on 144 randomly sampled face-scanned images of the reference bank,
# laid out in |yaw| order and read off contact sheets:
#   * 0-15°  nothing reads as turned at all;
#   * 16-25° the turn becomes visible around 20°, which is where the eye puts the
#     boundary — so 20 is kept (it was also the starting hypothesis, and here the
#     measurement agrees with it instead of the usual other way round);
#   * 26-50° unmistakably three-quarter, both eyes still visible at 47°;
#   * the sample contains NO face between 57.0° and 73.7°, and the four faces
#     above 73° are true profiles. Any cut inside that empty band is equally
#     supported by this data; 60 sits in the middle of it.
# Distribution obtained with these two numbers: 60% frontal, 38% three-quarter,
# 3% profile (n=144, |yaw| median 16.1°, p95 40.8°, max 78.8°).
#
# KNOWN LIMIT, and it is not small: a head turned far enough that one eye is
# hidden often defeats the DETECTOR, so the hardest profiles never reach this
# column at all — they come back as 'no_face' and count as "not measured". The
# 'profile' bucket therefore under-counts, and the UI says so instead of
# presenting 3% as the truth about a bank.
ANGLE_FRONTAL_MAX = 20.0
ANGLE_PROFILE_MIN = 60.0

# What one image costs the ⤢ backfill, in seconds — antelopev2 on the CPU path,
# measured at ~2 s/image over 144 images on the reference machine. Used ONLY to
# price the offer before the click; a slow machine takes longer and the wording
# says "about".
ANGLE_BACKFILL_S_PER_IMAGE = 2.0


def _angle_case():
    """A single SQL CASE mapping each row to its angle bucket id, or NULL for
    "not measured". Used to COUNT (one GROUP BY) and to FILTER, so the chips and
    the grid can never disagree about what a bucket contains.

    'behind' is the one bucket that is not a yaw: a back view has no face to
    measure, so it is the crossing of two facts the app ALREADY holds — the 🎭
    Faces pass found no face AND the 📐 Framing pass called the shot a back view.
    Requiring both is what keeps a landscape with nobody in it out of a bucket
    that claims a person is present; the cost is that 'behind' stays empty until
    BOTH passes have run, which the UI states rather than hides."""
    yaw = func.abs(BankImage.face_yaw)
    return case(
        (BankImage.face_yaw.isnot(None), case(
            (yaw < ANGLE_FRONTAL_MAX, 'frontal'),
            (yaw < ANGLE_PROFILE_MIN, 'three_quarter'),
            else_='profile')),
        ((BankImage.face_state == 'no_face') & (BankImage.framing == 'back'),
         'behind'),
        else_=None)


def _angle_counts(bank_id) -> dict:
    """Per-bucket image counts for the ⤢ Angle chips in ONE GROUP BY. Rows the
    CASE maps to NULL (no yaw, and not a proven back view) are "not measured" and
    are counted separately by bank_payload, never folded into 'frontal'."""
    bucket = _angle_case()
    rows = (db.session.query(bucket, func.count(BankImage.id))
            .filter(BankImage.bank_id == bank_id)
            .group_by(bucket).all())
    got = {k: n for k, n in rows if k is not None}
    return {k: int(got.get(k, 0)) for k in ANGLES}


def _framing_counts(bank_id, extra_crit=None) -> dict:
    """Per-bucket image counts for the 📐 Framing chips in ONE GROUP BY. Rows with
    a NULL framing (not classified) are excluded, so every key is present with a
    real count. ``extra_crit`` narrows the pool (e.g. status='keep' for coverage)."""
    q = (db.session.query(BankImage.framing, func.count(BankImage.id))
         .filter(BankImage.bank_id == bank_id, BankImage.framing.isnot(None)))
    if extra_crit is not None:
        q = q.filter(extra_crit)
    got = {k: n for k, n in q.group_by(BankImage.framing).all()}
    return {k: int(got.get(k, 0)) for k in _FRAMING_KEYS}


def _origin_counts(bank_id) -> dict:
    """Per-state image counts for the 🔎 Origin chips in ONE GROUP BY.

    Every state of ORIGINS is always present with a real count, INCLUDING
    'unknown' — which is the honest majority answer on any scraped or chat-sourced
    bank (measured: 3000/3000 on a real Telegram export) and has to be visible as
    such. Rows with a NULL origin (scanned before this pass existed, or
    unreadable) are excluded: they are "not measured", a different thing again."""
    q = (db.session.query(BankImage.origin, func.count(BankImage.id))
         .filter(BankImage.bank_id == bank_id, BankImage.origin.isnot(None)))
    got = {k: n for k, n in q.group_by(BankImage.origin).all()}
    return {k: int(got.get(k, 0)) for k in ORIGINS}


def _subfolder_of(relpath: str) -> str:
    """Top-level subfolder of a bank-relative path ('' for a root-level file) —
    the natural scoping axis for a Telegram export (one folder per chat/date)."""
    parts = (relpath or '').replace('\\', '/').split('/', 1)
    return parts[0] if len(parts) > 1 else ''


def _unresolved_dup_groups_q(bank_id, col=BankImage.dup_group):
    """Groups still holding ≥2 NON-rejected members — i.e. still to resolve. ``col``
    selects the stage: dup_group (exact/resized) or semantic_dup_group (crops)."""
    return (db.session.query(col)
            .filter(BankImage.bank_id == bank_id,
                    col.isnot(None),
                    BankImage.status != 'reject')
            .group_by(col)
            .having(func.count(BankImage.id) >= 2))


def _res_bucket_case():
    """A single SQL CASE mapping each scanned row to its resolution-tier id, used
    both to COUNT per tier (one GROUP BY) and — via _RES_BOUNDS — to FILTER a page
    to one tier. Rows with a NULL dimension never reach this (callers pre-filter
    width/height NOT NULL), so no NULL-misfile into the top tier."""
    area = BankImage.width * BankImage.height
    whens = [(area < hi, bid) for bid, _lo, hi in _RES_BUCKETS if hi is not None]
    return case(*whens, else_=_RES_BUCKETS[-1][0])


def _res_bucket_counts(bank_id) -> dict:
    """Per-tier image counts for the resolution chips (bank-wide, like the flag
    totals) in ONE GROUP BY. Unscanned rows (width/height NULL) are excluded, so a
    tier that no image falls into simply reports 0. Every tier id is present."""
    bucket = _res_bucket_case()
    rows = (db.session.query(bucket, func.count(BankImage.id))
            .filter(BankImage.bank_id == bank_id,
                    BankImage.width.isnot(None),
                    BankImage.height.isnot(None))
            .group_by(bucket).all())
    got = {bid: n for bid, n in rows}
    return {bid: int(got.get(bid, 0)) for bid, _lo, _hi in _RES_BUCKETS}


# --- explicit grid sorts -----------------------------------------------------
# The grid can ORDER on what the passes already MEASURED, instead of only
# filtering on it (asked for by nofaceman on Discord). One entry per sortable
# quantity; the UI offers each in both directions as '<key>_desc' / '<key>_asc'.
# Ids are user-facing query values — treat them like catalog labels and never
# rename one without an alias.
#   res       megapixels (width×height) — the original sort, kept as-is.
#   aesthetic the ✨ Score pass's 1–10 rating: ↓ surfaces the keepers, ↑ the duds.
#   sharp     the 🔎 Scan pass's Laplacian variance: ↑ surfaces the blurry misses.
# The first version stopped there, reasoning that noise / uniformity / bars /
# detail_ratio / NSFW each already have a CHIP that filters and orders worst-first.
# That reasoning does not survive contact with a real dump (asked again on
# Discord): a chip only ranks the rows that CROSS its threshold, so "the noisiest
# images I am keeping" — every one of them under the threshold — was unreachable,
# and no chip ranks the other way at all ("the cleanest first", "the safest
# first"). Sorting is not filtering, so the two do not duplicate each other; the
# precedent was already in this table, since 'sharp' coexists with the blur chip
# and 'res' with the small chip. So: EVERY quantity a pass persists on bank_image
# is sortable, both ways. The menu is grouped by pass in the UI to stay readable.
#   noise/flat/detail/bars/jpeg  the 🔎 Scan + provenance passes' raw figures.
#   nsfw                         the ✨ Score pass's 0–1 probability.
#   face                         the 🎭 Face pass's detection confidence.
#   size                         bytes on disk — the one figure NO chip exposes.
# Deliberately still NOT here: anything that is a LABEL rather than a measure
# (framing, origin, cluster and group ids, watermark/quality state). Those are
# facets — ordering by an id number means nothing to a reviewer.
_SORT_KEYS = {
    'aesthetic': lambda: BankImage.aesthetic_score,
    'sharp': lambda: BankImage.blur_score,
    'res': lambda: BankImage.width * BankImage.height,
    'noise': lambda: BankImage.noise_score,
    'flat': lambda: BankImage.uniformity_score,
    'detail': lambda: BankImage.detail_ratio,
    'bars': lambda: BankImage.bars_ratio,
    'jpeg': lambda: BankImage.jpeg_quality,
    'nsfw': lambda: BankImage.nsfw_score,
    'face': lambda: BankImage.face_det,
    'size': lambda: BankImage.file_size,
    #   yaw          the 🎭 Faces pass's head yaw, read as an ABSOLUTE angle so
    #                the two directions mean "most turned away" / "most
    #                face-on" rather than "turned left" / "turned right", which
    #                is not a distinction a training set cares about.
    'yaw': lambda: func.abs(BankImage.face_yaw),
    #   medium_conf  the 🎨 Medium pass's confidence gap. ↑ is the useful one: it
    #                opens on the images the classifier nearly could not call,
    #                which is exactly the pile a human should check by hand.
    'medium_conf': lambda: BankImage.medium_margin,
}
# Menu order (the UI renders it in this order); ids are stored query values, so a
# key may be added here but never renamed without an alias.
_SORT_ORDER = ('res', 'size', 'aesthetic', 'nsfw', 'sharp', 'noise', 'flat',
               'detail', 'bars', 'jpeg', 'face', 'yaw', 'medium_conf')
GRID_SORTS = tuple(f'{k}_{d}' for k in _SORT_ORDER for d in ('desc', 'asc'))


def _sort_order(sort):
    """The ORDER BY tuple for an explicit grid sort, or None for 'default' and
    for anything unknown (an unrecognised value must degrade to the server's own
    order, never 500). Rows the relevant pass never reached carry NULL and sink
    to the END in BOTH directions — ordering by "is NULL" first (0 before 1)
    — because a sort that opens on the un-measured pile is worse than no sort.
    Tie-break on id so a page boundary is stable."""
    key, _, direction = (sort or '').rpartition('_')
    if direction not in ('asc', 'desc') or key not in _SORT_KEYS:
        return None
    col = _SORT_KEYS[key]()
    ranked = col.desc() if direction == 'desc' else col.asc()
    return (col.is_(None).asc(), ranked, BankImage.id.asc())


def _flag_counts(bank_id, th) -> tuple[dict, dict]:
    """Two per-flag maps, because the UI asks two different questions of them.

    ``flags[f]``      — every image in the bank carrying the flag, whatever its
                        status. This is the FACET number: clicking the chip shows
                        exactly these rows, rejected ones included.
    ``actionable[f]`` — what a 🧹 Auto-reject on that flag would really flip.
                        ``apply_flags`` only ever touches ``status='pending'``
                        (its contract: a manual — or an earlier automatic — ✓/✕
                        is never overridden), so it is the same criterion
                        narrowed to the undecided pile.

    The two coincide on a bank nothing has been decided on yet, and diverge the
    moment one pass has run — which is exactly when the user starts trusting the
    number. Measured on a real 99 000-image bank at its SECOND pass: the button
    offered "5 930 flagged" for blur and rejected 0, because all 5 930 had been
    rejected by the first pass. Reading that as "the feature is broken" is the
    only reasonable conclusion, and it was the counter's fault, not the pass's.

    One query per flag, not two: the pending half rides along as a conditional
    SUM, so telling the truth costs nothing extra on a 100 000-image bank."""
    flags, actionable = {}, {}
    for flag in _QUALITY_FLAGS + _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is None:
            flags[flag] = actionable[flag] = 0
            continue
        total, pending = (
            db.session.query(
                func.count(BankImage.id),
                func.coalesce(
                    func.sum(case((BankImage.status == 'pending', 1), else_=0)), 0))
            .filter(BankImage.bank_id == bank_id).filter(crit).one())
        flags[flag] = int(total or 0)
        actionable[flag] = int(pending or 0)
    # ✨ Clean rides along: the one chip of the Quality row that had no number,
    # which read as "not a filter" next to six counted neighbours. Facet count
    # only — "auto-reject the clean ones" is not an action anything offers.
    flags['clean'] = int(
        db.session.query(func.count(BankImage.id))
        .filter(BankImage.bank_id == bank_id, _clean_criterion(th)).scalar() or 0)
    return flags, actionable


def _bp_caption_scope_sums(bank_id):
    """The caption pass's per-pile truth, moved verbatim (2026-08-23): for
    each status pile, how many rows a caption run would actually write
    (no caption yet), how many a HUMAN wrote (a forced run skips them),
    and how many carry a caption nobody recorded. One query, nine
    conditional sums. Returns the nine figures in that order."""
    # 🏷️ What each caption SCOPE would really caption: in that status AND still
    # without a caption. NOT counts.keep / counts.pending — the pass skips rows
    # that already have one, so quoting the status total would advertise a number
    # the run does not act on. That exact mistake was paid for once already by the
    # 🧹 Auto-reject counter (see _flag_counts): a button that offers "5 930" and
    # moves 0 reads as a broken feature, and it is the counter's fault. One query,
    # two conditional sums, so telling the truth costs nothing on a 100 000-image
    # bank.
    #
    # THREE figures per pile, not one, because 🔄 Re-caption has three different
    # things to say and folding any two of them would be a lie of a familiar kind:
    #   caption_todo_*     : no caption at all — what a normal pass writes;
    #   caption_asserted_* : a caption a HUMAN wrote — what a forced pass now SKIPS
    #                        (services/caption_origin.py), so "keeping the 3 you
    #                        wrote" is a measured number and not a hope;
    #   caption_unrecorded_*: a caption whose author was never recorded — rewritten
    #                        like any other, but it is NOT "machine-written", and
    #                        the screen must not claim it is.
    # Whatever is left (pile − the three) is machine-written and rewritten in silence.
    def _cap_sum(status, condition):
        return func.coalesce(func.sum(
            case((and_(BankImage.status == status, condition), 1), else_=0)), 0)

    no_caption = or_(BankImage.caption.is_(None), BankImage.caption == '')
    asserted = caption_origin.protected_clause(BankImage)
    unrecorded = caption_origin.unrecorded_clause(BankImage)
    (todo_keep, todo_pending, todo_reject, asserted_keep, asserted_pending,
     asserted_reject, unrecorded_keep, unrecorded_pending, unrecorded_reject) = (
        db.session.query(
            _cap_sum('keep', no_caption), _cap_sum('pending', no_caption),
            _cap_sum('reject', no_caption),
            _cap_sum('keep', asserted), _cap_sum('pending', asserted),
            _cap_sum('reject', asserted),
            _cap_sum('keep', unrecorded), _cap_sum('pending', unrecorded),
            _cap_sum('reject', unrecorded))
        .filter(BankImage.bank_id == bank_id).one())
    return (todo_keep, todo_pending, todo_reject, asserted_keep,
            asserted_pending, asserted_reject, unrecorded_keep,
            unrecorded_pending, unrecorded_reject)


def _bp_counts(base, total, todo_keep, todo_pending, todo_reject,
               asserted_keep, asserted_pending, asserted_reject,
               unrecorded_keep, unrecorded_pending, unrecorded_reject):
    """The workspace's headline counters, moved verbatim: piles, scan
    coverage with its named blind spot, promotion (back-link OR legacy
    flag OR dataset rows that point here), per-pass coverage, edits split
    by which gesture produced them, and the angle backfill offer."""
    counts = {
        'total': total,
        'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
        # The blind spot, named. An image no quality pass ever measured carries
        # no verdict, and EVERY quality flag is gated on `quality_state == 'ok'`
        # (see _flag_filter) — so these rows are structurally unreachable by
        # 🧹 Auto-reject. That is not "they are clean": it is "we know nothing
        # about them", and the two must never render as the same 0.
        'unscanned': base.filter(BankImage.quality_state.is_(None)).count(),
        # ...and how many of those a 🔎 Scan would actually pick up. Rejected
        # rows are out of the scan pool (_scan_pool), so the two numbers differ
        # by exactly the images that were thrown away before being measured. The
        # UI offers the gesture, so it quotes the number the gesture moves.
        'unscanned_scannable': base.filter(BankImage.quality_state.is_(None),
                                           BankImage.status != 'reject').count(),
        'pending': base.filter_by(status='pending').count(),
        'keep': base.filter_by(status='keep').count(),
        'reject': base.filter_by(status='reject').count(),
        # …and the caption-pass sizes of the two scopes it can be aimed at.
        'caption_todo_keep': int(todo_keep or 0),
        'caption_todo_pending': int(todo_pending or 0),
        # …and the third pile, now that a caption run can be aimed at the bin.
        # The keys keep their historical names — several of them are read by
        # shipped clients and by localStorage-backed screens, so the set GROWS,
        # it is never renamed.
        'caption_todo_reject': int(todo_reject or 0),
        # …and the two figures 🔄 Re-caption needs to stop lumping "what you wrote"
        # together with "what nobody recorded".
        'caption_asserted_keep': int(asserted_keep or 0),
        'caption_asserted_pending': int(asserted_pending or 0),
        'caption_asserted_reject': int(asserted_reject or 0),
        'caption_unrecorded_keep': int(unrecorded_keep or 0),
        'caption_unrecorded_pending': int(unrecorded_pending or 0),
        'caption_unrecorded_reject': int(unrecorded_reject or 0),
        # Images a dataset REALLY holds today (back-link), plus the ones promoted
        # before that link existed (legacy flag). Counting the flag alone kept
        # advertising copies the user had since deleted.
        'promoted': base.filter(or_(
            BankImage.promoted_dataset_id.isnot(None),
            # ...or into another BANK, the second destination. Counted here so
            # the "promoted" stat and the ⬆ badge on the tiles never disagree.
            BankImage.promoted_bank_id.isnot(None),
            BankImage.id.in_(db.session.query(FaceDatasetImage.bank_image_id)
                             .filter(FaceDatasetImage.bank_image_id.isnot(None))),
        )).count(),
        # V2 pass progress — how many images the scoring / watermark passes reached
        # (so the UI can show "scored 0/9000" and enable the threshold facets).
        'scored': base.filter(or_(BankImage.aesthetic_score.isnot(None),
                                  BankImage.nsfw_score.isnot(None))).count(),
        'watermark_scanned': base.filter(
            BankImage.watermark_state.isnot(None),
            ~_watermark_history_inactive_clause()).count(),
        'framing_classified': base.filter(BankImage.framing.isnot(None)).count(),
        # ✂/✨ edited HERE, split by which edit produced the blob. Two numbers
        # rather than one total: ↩ Revert undoes both, but "12 cropped" and "12
        # upscaled" are answers to different questions, and a single figure would
        # let a GPU pass hide behind a hand crop.
        'cropped': base.filter(BankImage.edit_method == 'crop').count(),
        'improved': base.filter(BankImage.edit_method == 'improve').count(),
        # 🎨 Medium — how many rows the pass has a verdict for ('unsure'
        # included: it IS a verdict). Drives the chip row's appearance and the
        # Sort menu's "run 🎨 Medium first" state.
        'medium_classified': base.filter(BankImage.medium.isnot(None)).count(),
        # ⤢ Angle — how many rows carry a measured yaw…
        'angle_measured': base.filter(BankImage.face_yaw.isnot(None)).count(),
        # …and how many were face-scanned by a build that computed the yaw and
        # threw it away. This is the ONLY number that can offer the backfill
        # honestly: it is the exact size of the re-measure job, so the UI can
        # price the click before the user makes it instead of after.
        'angle_backfillable': base.filter(BankImage.face_state.isnot(None),
                                          BankImage.face_yaw.is_(None)).count(),
    }
    return counts


def _bp_pass_scopes(bank_id, counts, th, todo_keep, todo_pending,
                    todo_reject):
    """What each pass would REALLY walk, per pile, moved verbatim: every
    entry is computed from the same clause its pass's pool filters on,
    never from a second copy of the predicate — the numbers on the launch
    dialogs have to be the numbers the runs move."""
    # 🎛 WHAT EACH PASS WOULD REALLY DO, PER PILE. The launch dialogs put a
    # number on every scope line, and that number has to be the number the run
    # walks — the whole reason these exist. So each entry is computed from the
    # SAME clause its pass's pool filters on (_scan_todo_clause,
    # _watermark_todo_clause, …), never from a second copy of the predicate.
    #
    # Two figures per pass, because every one of these dialogs offers two runs:
    #   todo — what is left to do in that pile (the plain button);
    #   all  — the pile itself (the "do it again anyway" line: rescan / rescore /
    #          force), which is counts.keep/pending/reject and is NOT repeated
    #          here. The client adds it from `counts`, so the two can never drift.
    #
    # Cost: three conditional sums per pass in one query each, the shape _cap_sum
    # already proved free on a 100 000-image bank.
    all_by_status = {'keep': counts['keep'], 'pending': counts['pending'],
                     'reject': counts['reject']}
    pass_scopes = {
        'scan': {'todo': _todo_by_status(bank_id, _scan_todo_clause()),
                 'all': dict(all_by_status)},
        # ⚠️ WATERMARK'S "all" IS NOT THE PILE. Even a rescan leaves 'dismissed'
        # rows alone (the user already ruled on them), so quoting counts.keep on
        # its rescan line would offer images the run provably skips. This is the
        # exact shape of the defect these numbers exist to prevent, and it is why
        # every 'all' here is a measured query rather than a reused total.
        'watermark': {'todo': _todo_by_status(bank_id, _watermark_todo_clause()),
                      'all': _todo_by_status(bank_id, _watermark_not_dismissed())},
        # 🔤 Find text — same two figures, same dismissed exclusion as the
        # watermark scan: even its rescan line never re-examines a row the user
        # ruled on, so quoting the pile size would offer images the run skips.
        'text_scan': {'todo': _todo_by_status(bank_id, _text_todo_clause()),
                      'all': _todo_by_status(bank_id, _watermark_not_dismissed())},
        # ✂ AUTO-CROP AND 🧽 REPAINT — the two levels that produce a new IMAGE.
        # Their pool is not "the images in that pile": it is the flagged rows
        # that carry an authorised geometry (_clean_todo_clause), which is why
        # these two entries exist at all rather than their windows reusing
        # counts.keep. Neither has a "do it again" lane — a cleaned image leaves
        # the pool and comes back only through ↩ Undo cleaning — so 'all'
        # repeats 'todo' rather than being left absent, which would render as a
        # permanent "counting…".
        #
        # ⚠️ WHAT THIS NUMBER IS. It is the pool each level WALKS (the figure its
        # progress bar counts to), not the number of images it will change: ✂
        # crops only the marks the router puts in a border band, and that
        # decision needs each image's real dimensions. The crop window says so in
        # as many words rather than quoting a routed figure that would cost
        # thousands of file headers on every payload.
        'watermark_crop': {'todo': _todo_by_status(bank_id, _crop_todo_clause()),
                           'all': _todo_by_status(bank_id, _crop_todo_clause())},
        'watermark_inpaint': {'todo': _todo_by_status(bank_id, _clean_todo_clause()),
                              'all': _todo_by_status(bank_id, _clean_todo_clause())},
        'framing': {'todo': _todo_by_status(bank_id, BankImage.framing.is_(None)),
                    'all': dict(all_by_status)},
        # 🎨 Medium is the one pass whose pool is not its work: it computes NO
        # image inference and reads the CLIP embedding ✨ Score cached, so a row
        # with no score CANNOT be classified however wide the scope is. Counting
        # only the pool made the window promise "Classify 2 images" over a run
        # that could only answer "0 classified, 2 skipped (not scored yet)" —
        # the same class of defect as a count that does not match the run, one
        # step earlier. 'blocked' is that subset, measured, per pile.
        'medium': {'todo': _todo_by_status(bank_id, BankImage.medium.is_(None)),
                   'all': dict(all_by_status),
                   'blocked': _todo_by_status(bank_id, and_(
                       BankImage.medium.is_(None), _unscored_clause())),
                   'blocked_all': _todo_by_status(bank_id, _unscored_clause())},
        # ⤢ has no "do it again" lane at all: a measured yaw is a measured yaw.
        # Both figures are the same pool, said twice rather than left absent.
        'angles': {'todo': _todo_by_status(bank_id, _angle_todo_clause()),
                   'all': _todo_by_status(bank_id, _angle_todo_clause())},
        # ✨ Upscale & improve — the pool is everything not already improved, and
        # there is no "do it again" lane on purpose: an improved image leaves the
        # pool and comes back only through ↩ Revert. Like the two cleaning levels
        # above, 'all' repeats 'todo' rather than being absent (which the dialogs
        # render as a permanent "counting…").
        'improve': {'todo': _todo_by_status(bank_id, _improve_todo_clause()),
                    'all': _todo_by_status(bank_id, _improve_todo_clause())},
        # 👥 takes no scope (its clusters are one numbering of the whole bank),
        # but its dialog still has to quote the ONE number it will run on — and
        # that number is not the pile: rows in a folder the user declared to be a
        # single person are skipped entirely.
        'faces': {'todo': _todo_by_status(bank_id, or_(
            BankImage.face_cluster_origin.is_(None),
            BankImage.face_cluster_origin != 'asserted')),
            'all': dict(all_by_status)},
        # Caption's three are already computed above, by the same rule and with
        # the same filter the job uses. Mirrored here so one client-side reader
        # serves every dialog; the flat caption_todo_* keys stay for the callers
        # that already read them. Its 'all' is the pile — what a FORCED run walks
        # before the "keep what you wrote" protection, which bankCaptionScope.js
        # subtracts from the caption_asserted_* figures.
        'caption': {'todo': {'keep': int(todo_keep or 0),
                             'pending': int(todo_pending or 0),
                             'reject': int(todo_reject or 0)},
                    'all': dict(all_by_status),
                    # …and WHAT KIND of images this pile holds, measured, because
                    # one thing the captioners are known to do badly is tied to
                    # that: see captionNsfwNotice.js. Two figures per pile and
                    # never one, for the same reason `unscanned` exists beside the
                    # flag totals — an image ✨ Score never reached is not a SFW
                    # image, and a share computed over the whole pile would quietly
                    # treat "unknown" as "clean" and understate itself.
                    'nsfw': _todo_by_status(bank_id, _flag_filter('nsfw', th)),
                    'nsfw_measured': _todo_by_status(
                        bank_id, BankImage.nsfw_score.isnot(None))},
    }
    return pass_scopes


def _bp_similarity_summaries(bank_id, base):
    """The duplicate and cluster summaries, moved verbatim: exact dup
    groups, stage-2 semantic near-dups (same shape), the forty biggest
    person clusters (cover = surest face) and style clusters (cover =
    lowest id), plus how many rows the face pass reached. Returns
    (dup, semantic_dup, clusters, faces_scanned, style_clusters)."""
    dup_rows = (db.session.query(BankImage.dup_group, func.count(BankImage.id))
                .filter(BankImage.bank_id == bank_id,
                        BankImage.dup_group.isnot(None))
                .group_by(BankImage.dup_group).all())
    dup = {'groups': len(dup_rows),
           'images': sum(n for _g, n in dup_rows),
           'unresolved': _unresolved_dup_groups_q(bank_id).count()}
    # Stage-2 semantic near-duplicate groups (crops/variants), same summary shape.
    sem_rows = (db.session.query(BankImage.semantic_dup_group, func.count(BankImage.id))
                .filter(BankImage.bank_id == bank_id,
                        BankImage.semantic_dup_group.isnot(None))
                .group_by(BankImage.semantic_dup_group).all())
    semantic_dup = {
        'groups': len(sem_rows),
        'images': sum(n for _g, n in sem_rows),
        'unresolved': _unresolved_dup_groups_q(
            bank_id, BankImage.semantic_dup_group).count()}
    # Person clusters, biggest first; cover = the member with the surest face.
    cl_rows = (db.session.query(BankImage.face_cluster, func.count(BankImage.id))
               .filter(BankImage.bank_id == bank_id,
                       BankImage.face_cluster.isnot(None))
               .group_by(BankImage.face_cluster)
               .order_by(func.count(BankImage.id).desc(),
                         BankImage.face_cluster.asc())
               .limit(40).all())
    clusters = []
    for cid, size in cl_rows:
        cover = (BankImage.query
                 .filter_by(bank_id=bank_id, face_cluster=cid)
                 .order_by(BankImage.face_det.desc().nullslast(),
                           BankImage.id.asc())
                 .first())
        clusters.append({'id': cid, 'size': size,
                         'cover_image_id': cover.id if cover else None})
    faces_scanned = base.filter(BankImage.face_state.isnot(None)).count()
    # Style clusters (group by visual style), biggest first — the "group by
    # style" counterpart to the person clusters above. Cover = the lowest id of
    # the cluster (stable, no per-image quality signal to rank by here).
    st_rows = (db.session.query(BankImage.style_cluster, func.count(BankImage.id))
               .filter(BankImage.bank_id == bank_id,
                       BankImage.style_cluster.isnot(None))
               .group_by(BankImage.style_cluster)
               .order_by(func.count(BankImage.id).desc(),
                         BankImage.style_cluster.asc())
               .limit(40).all())
    style_clusters = []
    for cid, size in st_rows:
        cover = (BankImage.query
                 .filter_by(bank_id=bank_id, style_cluster=cid)
                 .order_by(BankImage.id.asc()).first())
        style_clusters.append({'id': cid, 'size': size,
                               'cover_image_id': cover.id if cover else None})
    return dup, semantic_dup, clusters, faces_scanned, style_clusters


def bank_payload(user_id, bank_id) -> dict | None:
    """Everything the bank workspace needs on one poll: counts, flag totals,
    duplicate/cluster summaries, live job, thresholds."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    base = BankImage.query.filter_by(bank_id=bank_id)
    total = base.count()
    (todo_keep, todo_pending, todo_reject, asserted_keep,
     asserted_pending, asserted_reject, unrecorded_keep,
     unrecorded_pending, unrecorded_reject) = _bp_caption_scope_sums(
        bank_id)
    counts = _bp_counts(
        base, total, todo_keep, todo_pending, todo_reject, asserted_keep,
        asserted_pending, asserted_reject, unrecorded_keep,
        unrecorded_pending, unrecorded_reject)
    pass_scopes = _bp_pass_scopes(bank_id, counts, th, todo_keep,
                                  todo_pending, todo_reject)
    framing = _framing_counts(bank_id)
    flags, flags_actionable = _flag_counts(bank_id, th)
    res_buckets = _res_bucket_counts(bank_id)
    origins = _origin_counts(bank_id)
    (dup, semantic_dup, clusters, faces_scanned,
     style_clusters) = _bp_similarity_summaries(bank_id, base)
    semantic = semantic_engine_info(user_id, bank_id)
    counts['semantic_ready'] = bool(semantic and semantic['ready'])
    counts['semantic_indexed'] = int(
        semantic['counts']['ok'] if semantic else 0)
    return {
        'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
        'semantic_engine': (semantic['engine'] if semantic else
                            _selected_semantic_engine(bank)),
        'created_at': bank.created_at.isoformat() if bank.created_at else None,
        'counts': counts, 'flags': flags,
        # Per-pass × per-pile run sizes — see the comment where they are built.
        'pass_scopes': pass_scopes,
        # A SECOND map, not a replacement: 'flags' answers the facet's question
        # ("show me these") and 'flags_actionable' answers the button's ("how
        # many would this reject"). Overwriting the first would have broken the
        # chip row, whose count legitimately includes already-rejected images.
        'flags_actionable': flags_actionable,
        'res_buckets': res_buckets,
        'framing': framing, 'origins': origins,
        'mediums': _medium_counts(bank_id), 'angles': _angle_counts(bank_id),
        # What the ⤢ backfill would cost, in the app's own words. ~2 s/image is
        # the MEASURED cost of antelopev2 on this project's CPU path (144 images,
        # ~2 s each); it is a ballpark shown before the click, never a promise.
        'angle_backfill_minutes': max(1, round(
            counts['angle_backfillable'] * ANGLE_BACKFILL_S_PER_IMAGE / 60)
        ) if counts['angle_backfillable'] else None,
        'dup': dup,
        'semantic_dup': semantic_dup,
        'semantic': semantic,
        'clusters': clusters, 'faces_scanned': faces_scanned,
        'style_clusters': style_clusters,
        'activity': bank_jobs.get(bank_id),
        # ↩ the one-step-back offer, so the bar survives a reload (the decision
        # it takes back is in the database, not in a tab).
        'undo': bank_undo.peek(bank_id),
        'pipeline_report': _load_pipeline_report(bank),
        # When each pass last completed here — what lets a screen answer "already
        # done, on this many images" instead of offering an identical re-run.
        'last_passes': last_passes(bank),
        'score_device': score_device_info(bank_id),
        # Non-null only on a bank created before the create-time guard, whose
        # folder IS a dataset's storage. The workspace turns it into a standing
        # banner and disables 🗑 Delete rejected, which the server refuses anyway.
        'dataset_conflict': bank_dataset_conflict(user_id, bank_id),
        'thresholds': th,
    }


def bank_activity(user_id, bank_id) -> dict | None:
    """JUST the live job — the handful of bytes the progress banner and its Stop
    button need. ``None`` when the bank is gone.

    ⏱ WHY THIS EXISTS AS ITS OWN CALL. The banner used to ride on
    :func:`bank_payload`, the heaviest read on the page: ~60 full-table aggregates
    over the bank, ~1.4 s on a 50 397-image bank at rest and worse under the write
    load of the very pass the banner is reporting. That payload is polled every
    2 s WHILE A JOB RUNS, so the one moment the user needs the Stop button is the
    one moment the request carrying it cannot land. It was reported as "I can't
    see my banks and I can't even stop the scan because I have no progress bar" —
    followed by seven Stop clicks in 20 ms against a UI that answered nothing.

    Nothing here touches the filesystem and only one indexed row is read, so the
    banner's cost no longer depends on the size of the bank at all. The full
    payload keeps its own, slower refresh: counts advancing a few seconds late is
    a cosmetic delay, a Stop button that never arrives is not.
    """
    if not get_bank(user_id, bank_id):
        return None
    return {'activity': bank_jobs.get(bank_id)}


def flag_preview(user_id, bank_id, overrides=None) -> dict | None:
    """Per-flag image counts for a CANDIDATE threshold set — what the bank WOULD
    look like at those numbers, without saving anything.

    This is what turns tuning a threshold from a guess into a decision: the
    quality scan persists RAW scores and every verdict is recomputed at read
    time (see BankImage's docstring), so answering "how many images would a
    sharpness floor of 140 flag?" is the same COUNT the payload already runs,
    with a different dict. No decode, no pass, no write.

    Only the read-time thresholds are meaningful here. The four grouping ones
    (dup_distance, face/style/semantic similarity) are baked into stored group
    ids by their pass, so a count against a candidate value would be a number
    about the OLD grouping — the UI says "applies at the next pass" instead.

    Unknown keys and junk values are ignored rather than 400: this is a live
    preview firing on every keystroke, and a half-typed "0." must degrade to
    "no change yet", never to an error toast."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    for key, val in (overrides or {}).items():
        if key not in cfg.DEFAULTS['bank']:
            continue
        try:
            th[key] = float(val)
        except (TypeError, ValueError):
            continue   # a malformed override falls back to the shipped threshold
    th['dup_distance'] = int(th['dup_distance'])
    th['min_side'] = int(th['min_side'])
    base = BankImage.query.filter_by(bank_id=bank_id)
    # Same producer as the workspace payload, so the two can never answer the
    # same question differently — and so the panel's "before/after" line can
    # tell "would be flagged" from "would be rejected" if it ever needs to.
    flags, flags_actionable = _flag_counts(bank_id, th)
    return {'flags': flags, 'flags_actionable': flags_actionable,
            'thresholds': th, 'total': base.count()}


def _load_json_column(bank, attr):
    """Parsed JSON column, or None. A corrupt blob is swallowed — one broken
    field must never 500 the whole bank payload."""
    import json as _json
    raw = getattr(bank, attr, None)
    if not raw:
        return None
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        return None


def last_passes(bank: ImageBank) -> dict:
    """{step: {at, detail, counts, ...}} — when each pass last completed here."""
    return _load_json_column(bank, 'last_passes') or {}


def note_pass_run(bank_id, step, *, detail=None, counts=None, **extra):
    """Record that ``step`` just COMPLETED on this bank.

    One write serves two questions the Bank could not answer before. "Have I
    already run this?" — the pass reads its own row back and can recognise an
    unchanged bank instead of recomputing it. And "does the Launch-all report
    still speak for this step?" — a report is a photograph of one run, so a step
    somebody has re-run since must not keep announcing that run's verdict (a
    Launch-all stopped before ✂ ran went on saying "cancelled before it ran"
    over a standalone run that had just grouped 2358 shots).

    Failure here never sinks the pass that produced the result: the work is
    done and committed, and a lost bookkeeping row costs a redundant re-run at
    worst — so it is logged, not raised."""
    import json as _json
    import time as _time
    bank = db.session.get(ImageBank, bank_id)
    if bank is None:
        return
    row = {'at': _time.time(), 'detail': detail, 'counts': counts or {}}
    row.update({k: v for k, v in extra.items() if v is not None})
    journal = last_passes(bank)
    journal[step] = row
    try:
        bank.last_passes = _json.dumps(journal)
        db.session.commit()
    except Exception as e:  # noqa: BLE001 — bookkeeping, never the user's result
        db.session.rollback()
        logger.warning('bank %s: could not record the %s pass run: %s',
                       bank_id, step, e)


# Which report step a completed standalone job journals as. ONE row per step
# key the Launch-all report uses, so _load_pipeline_report can annotate its
# rows "re-run since" for EVERY pass — until this map existed only
# semantic_dedup wrote its journal and a report's red "cancelled before it ran"
# outlived successful 👥/🚩 re-runs indefinitely. Deliberately absent:
#   semantic_dedup — journals itself, with engine/threshold/signature the
#                    launch window reads back (a plain overwrite would erase them);
#   pipeline       — the Launch-all run WRITES the report, it never supersedes it;
#   angles/medium/promote/delete_rejected/dataset_import — not report steps.
_JOURNALED_JOB_KINDS = {
    'scan': 'scan', 'score': 'score', 'faces': 'faces',
    'watermark': 'watermark', 'framing': 'framing', 'caption': 'caption',
    'semantic_index': 'semantic_index',
}


def _journal_completed_job(bank_id, kind, detail):
    step = _JOURNALED_JOB_KINDS.get(kind)
    if step and isinstance(bank_id, int):     # video-lane keys are 'video:<id>'
        note_pass_run(bank_id, step, detail=detail)


bank_jobs.on_complete = _journal_completed_job


def _load_pipeline_report(bank: ImageBank):
    """The persisted 'Launch all' summary, with every step that has been re-run
    since annotated (``superseded_at``/``superseded_detail``). The report itself
    is never rewritten — it stays the honest record of that run; the annotation
    is what stops it speaking for a step whose story has moved on."""
    report = _load_json_column(bank, 'pipeline_report')
    if not report or not isinstance(report.get('steps'), list):
        return report
    journal = last_passes(bank)
    written_at = report.get('finished_at') or 0
    for entry in report['steps']:
        run = journal.get(entry.get('step'))
        if run and (run.get('at') or 0) > written_at:
            entry['superseded_at'] = run['at']
            entry['superseded_detail'] = run.get('detail')
    return report


def list_banks(user_id, dataset_id=None) -> list:
    """Every bank of the user, newest first, with its triage counters, the ids
    of its card preview images — and, when ``dataset_id`` is given, how many
    kept images each bank would promote into THAT dataset (``promotable``).

    The promotable counts ride along on purpose: the dataset-side bank chooser
    used to ask /bank/<id>/promotable once per bank, so a library of 12 banks
    cost 13 requests to open one panel. One grouped query answers them all."""
    promotable = _promotable_counts(user_id, dataset_id) if dataset_id is not None else None
    out = []
    for bank in (ImageBank.query.filter_by(user_id=user_id)
                 .order_by(ImageBank.created_at.desc()).all()):
        base = BankImage.query.filter_by(bank_id=bank.id)
        row = {
            'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
            'created_at': bank.created_at.isoformat() if bank.created_at else None,
            'total': base.count(),
            'keep': base.filter_by(status='keep').count(),
            'reject': base.filter_by(status='reject').count(),
            'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
            'preview_ids': _preview_ids(bank.id),
            'activity': bank_jobs.get(bank.id),
        }
        if promotable is not None:
            row['promotable'] = promotable.get(bank.id, 0)
        out.append(row)
    return out


def random_kept_caption(user_id, bank_id) -> str | None:
    """Return one cleaned caption from an owned BANK's kept images.

    The Bank's half of ``face_dataset_service.random_kept_caption``, and
    deliberately the same contract down to the return values: ``None`` means the
    bank exists but has no non-blank kept caption, and an inaccessible bank
    raises ``LookupError`` so the route answers 404 without leaking whether it
    exists. Same shape of query too — the count and the random offset stay in
    SQL, so a 50 000-image bank never materialises its captions to pick one.

    WHY THE BANK IS A LEGITIMATE SOURCE: a bank is captioned by the 🏷️ Caption
    pass long before anything is promoted, and it is usually the biggest pile of
    real captions on the machine. Restricting this draw to datasets meant the
    prompt shortcut could not reach them (asked for by the maintainer).

    ``_PYTHON_STRIP_CHARS`` is imported rather than copied: it exists to make
    SQL's trim() agree with Python's ``str.strip()``, and two copies of that
    character set would be two answers to "is this caption blank".
    """
    if not get_bank(user_id, bank_id):
        raise LookupError('bank not found')
    from .face_dataset_service import _PYTHON_STRIP_CHARS

    cleaned = func.trim(BankImage.caption, _PYTHON_STRIP_CHARS)
    eligible = (db.session.query(BankImage.caption)
                .join(ImageBank, BankImage.bank_id == ImageBank.id)
                .filter(BankImage.bank_id == bank_id,
                        ImageBank.user_id == str(user_id),
                        BankImage.status == 'keep',
                        BankImage.caption.isnot(None),
                        cleaned != ''))
    count = eligible.count()
    if not count:
        return None
    caption = (eligible.order_by(BankImage.id.asc())
               .offset(random.randrange(count)).limit(1).scalar())
    # Same belt as the dataset lane: an unusual Unicode whitespace-only value
    # that slipped through SQLite's trim character set must not come back as a
    # "caption" made of spaces.
    return (caption or '').strip() or None


def _promotable_counts(user_id, dataset_id) -> dict | None:
    """{bank_id: promotable count} for EVERY bank at once — the batched form of
    promotable_count(), same eligibility rule (see _promotable_query). None when
    the dataset is gone or the id is junk, so the caller omits the field rather
    than publishing zeros it can't stand behind. Banks with nothing eligible are
    absent from the mapping; list_banks reads them as 0."""
    try:
        dataset_id = int(dataset_id)
    except (TypeError, ValueError):
        return None
    if not FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first():
        return None
    rows = (db.session.query(BankImage.bank_id, func.count(BankImage.id))
            .join(ImageBank, ImageBank.id == BankImage.bank_id)
            .filter(ImageBank.user_id == user_id,
                    BankImage.status == 'keep',
                    _not_already_on(dataset_id))
            .group_by(BankImage.bank_id).all())
    return {bank_id: n for bank_id, n in rows}


PREVIEW_COUNT = 5


def _preview_ids(bank_id, limit=PREVIEW_COUNT) -> list:
    """The first few image ids of a bank, for the card's thumbnail strip.
    Ordered by id (= inventory order), so the strip is STABLE across reloads —
    and rejected shots are skipped so a triaged bank doesn't advertise its
    discards. Kept images are deliberately NOT promoted to the front: most banks
    sit at zero keeps for their whole life, and re-ordering the strip as the user
    triages would make the card flicker under them. One query per bank, so the
    whole page still costs a single HTTP request."""
    rows = (BankImage.query.with_entities(BankImage.id)
            .filter(BankImage.bank_id == bank_id, BankImage.status != 'reject')
            .order_by(BankImage.id.asc()).limit(limit).all())
    return [r[0] for r in rows]


# --- free-text matching (the search bar, and its inverse) --------------------
# ONE definition of "this image's text mentions <term>", used by the search
# filter, by the exclude filter and by the curation pool, so the three can never
# drift into disagreeing about what a term matches.
_MAX_TEXT_TERMS = 12          # a checklist has a handful of tags, not a corpus


def _text_match(term):
    """Rows whose CAPTION or RELPATH contains `term`, case-insensitively. LIKE
    metacharacters in the term are escaped so a literal '%'/'_' matches itself.
    The caption is COALESCED to '' because this criterion is also used NEGATED:
    in SQL, NULL LIKE x is NULL, and NOT NULL is still NULL — so an uncaptioned
    row would be dropped by an exclude filter instead of kept, which is the exact
    opposite of "show me what does NOT have this tag yet"."""
    esc = term.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    like = f'%{esc}%'
    return or_(func.coalesce(BankImage.caption, '').ilike(like, escape='\\'),
               BankImage.relpath.ilike(like, escape='\\'))


# Separators a caption puts around a word. The word-boundary match below turns
# each of them into a space, then looks for ' <tag> ' in the padded result — the
# closest a plain LIKE gets to \b without a REGEXP function, and it needs no
# extension, no engine hook and no index.
_WORD_SEPARATORS = (',', '.', ';', ':', '!', '?', '(', ')', '[', ']', '"', "'",
                    '/', '\\', '-', '_', '\n', '\r', '\t')


def _spaced_text():
    """`' ' || lower(caption) || ' ' || lower(relpath) || ' '` with every
    separator replaced by a space — the haystack a whole-word match looks in."""
    haystack = func.lower(
        func.coalesce(BankImage.caption, '') + ' ' + BankImage.relpath)
    for sep in _WORD_SEPARATORS:
        haystack = func.replace(haystack, sep, ' ')
    return ' ' + haystack + ' '


def _tag_match(tag):
    """Rows whose caption or path mentions `tag` AS A WORD.

    Deliberately stricter than the 🚫 exclude field's substring match, and the
    difference is not an inconsistency: an exclude term is typed by hand, where
    a partial match is often what someone means, while a tag chip comes FROM a
    caption's own tokens — matching 'car' inside 'scarf' would make the feature
    lie about what it found. The substring/word-boundary split between the two
    is the known parity debt; this is the side that pays it first."""
    esc = tag.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    return _spaced_text().ilike(f'% {esc} %', escape='\\')


def _text_terms(value) -> list:
    """Split an exclude field into terms on commas, trimmed, de-duplicated
    case-insensitively and capped. One field, several tags ('nsfw, logo') — a
    checklist pass usually hides more than one thing at a time."""
    out, seen = [], set()
    for tok in str(value or '').split(','):
        tok = tok.strip()
        if tok and tok.lower() not in seen:
            seen.add(tok.lower())
            out.append(tok)
    return out[:_MAX_TEXT_TERMS]


def _apply_text_filters(q, search=None, exclude=None, tags=None):
    """search ∩ ALL(tags) ∩ NOT(exclude₁) ∩ NOT(exclude₂)… — the positive term
    narrows to what mentions it, each exclude term hides what mentions it. They
    compose: searching 'dress' while excluding 'red' is a legitimate (and useful)
    question. An exclude term that is ALSO the search term simply yields nothing,
    honestly.

    ``tags`` is the 🏷️ chip filter, and it is an AND on purpose: the gesture is
    "images that have THIS and THIS", i.e. narrowing from one image's attributes.
    An OR would grow the result set with every chip ticked, which reads as the
    filter going backwards. Each tag matches as a WORD (see _tag_match) — a
    dedicated parameter with its own criterion, never folded into `search` or
    `exclude`, so two features can never fight over one key."""
    term = (search or '').strip()
    if term:
        q = q.filter(_text_match(term))
    for tag in _text_terms(tags):
        q = q.filter(_tag_match(tag))
    for bad in _text_terms(exclude):
        q = q.filter(~_text_match(bad))
    return q


# The facets the grid composes, in the order _apply_facets applies them. These
# names are BOTH the keyword arguments below and the ids ``skip`` accepts, which
# is what lets facet_counts measure a facet with every OTHER filter in force and
# its own left out. Stored query keys — never rename one without an alias.
FACETS = ('status', 'flag', 'cluster', 'group', 'semantic_group', 'style',
          'framing', 'origin', 'medium', 'angle', 'subfolder', 'search',
          'exclude', 'tags', 'res_bucket')


def _apply_facets(q, th, skip=None, *, status=None, flag=None, cluster=None,
                  group=None, semantic_group=None, style=None, subfolder=None,
                  search=None, exclude=None, tags=None, res_bucket=None,
                  framing=None, origin=None, medium=None, angle=None):
    """Narrow ``q`` by the composing facets and return ``(q, order)``.

    ONE place, because the grid and the chip counters ask the same question and
    a second copy of these predicates is exactly how the two drift apart — the
    chip that says 4 043 and opens on 12 rows is that drift, not a rounding
    error. ``order`` is the flag's worst-first ordering (the caller may override
    it with an explicit sort); counters ignore it.

    ``skip`` names ONE facet (a FACETS id) to leave OUT. That is the whole
    counting rule: a facet's own value must never narrow its own counts, or
    picking "blur" would show its neighbours at 0 and the user could no longer
    change their mind without clearing everything first."""
    if status in ('pending', 'keep', 'reject') and skip != 'status':
        q = q.filter(BankImage.status == status)
    order = BankImage.id.asc()
    if skip == 'flag':
        flag = None
    if flag == 'flagged':
        crits = [c for c in (_flag_filter(f, th) for f in _QUALITY_FLAGS)
                 if c is not None]
        q = q.filter(or_(*crits))
    elif flag == 'clean':
        q = q.filter(_clean_criterion(th))
    elif flag == 'dups':
        q = q.filter(BankImage.dup_group.isnot(None))
        order = (BankImage.dup_group.asc(), BankImage.id.asc())
    elif flag == 'semantic_dups':
        q = q.filter(BankImage.semantic_dup_group.isnot(None))
        order = (BankImage.semantic_dup_group.asc(), BankImage.id.asc())
    elif flag == 'no_face':
        # Literally "no face was found" — ONLY face_state == 'no_face'. The other
        # non-scorable states (low_det / too_small / extreme_pose) DID detect a
        # face; lumping them in here surfaced photos with visible faces under a
        # "No face" chip. 'unreadable'/'error' are read failures, not "no face".
        q = q.filter(BankImage.face_state == 'no_face')
    elif flag in _QUALITY_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
        order = {'blur': BankImage.blur_score.asc(),
                 'noise': BankImage.noise_score.desc(),
                 'uniform': BankImage.uniformity_score.asc(),
                 'small': BankImage.width.asc(),
                 'soft_detail': BankImage.detail_ratio.asc(),
                 'bars': BankImage.bars_ratio.desc(),
                 'unreadable': BankImage.id.asc()}[flag]
    elif flag in _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
        # Worst first: least aesthetic / most-confident NSFW at the top.
        order = {'low_aesthetic': BankImage.aesthetic_score.asc(),
                 'nsfw': BankImage.nsfw_score.desc(),
                 'watermark': BankImage.id.asc()}[flag]
    if cluster is not None and skip != 'cluster':
        q = q.filter(BankImage.face_cluster == int(cluster))
    if group is not None and skip != 'group':
        q = q.filter(BankImage.dup_group == int(group))
    if semantic_group is not None and skip != 'semantic_group':
        q = q.filter(BankImage.semantic_dup_group == int(semantic_group))
    if style is not None and skip != 'style':
        q = q.filter(BankImage.style_cluster == int(style))
    if framing in _FRAMING_KEYS and skip != 'framing':
        # One framing bucket (face/bust/body/back/unknown) — composes with every
        # other facet. An unknown/absent value simply doesn't filter.
        q = q.filter(BankImage.framing == framing)
    if origin in ORIGINS and skip != 'origin':
        # One provenance state. 'unknown' is a real, selectable answer — it is
        # what a stripped file honestly is, and the user must be able to see that
        # pile rather than have it silently merged into "not AI".
        q = q.filter(BankImage.origin == origin)
    if medium in MEDIUM_KEYS and skip != 'medium':
        # One medium bucket. 'unsure' is selectable on purpose — it is the pile
        # the classifier honestly could not call, and the only way to work
        # through it is to be able to look at it.
        q = q.filter(BankImage.medium == medium)
    if angle in ANGLES and skip != 'angle':
        # One head-angle bucket, recomputed from the stored yaw at read time (so
        # re-tuning the two cuts re-slices the bank with no rescan) — see
        # _angle_case for what 'behind' costs and requires.
        q = q.filter(_angle_case() == angle)
    if subfolder is not None and skip != 'subfolder':
        # '' scopes to root-level files; any other value to that top-level folder
        # and everything nested under it. startswith() escapes LIKE metachars.
        if subfolder == '':
            q = q.filter(~BankImage.relpath.contains(os.sep))
        else:
            q = q.filter(BankImage.relpath.startswith(subfolder + os.sep))
    # Full-text over caption + relpath, positive (search) and negative (exclude).
    q = _apply_text_filters(q, None if skip == 'search' else search,
                            None if skip == 'exclude' else exclude,
                            None if skip == 'tags' else tags)
    if res_bucket in _RES_BOUNDS and skip != 'res_bucket':
        # One resolution tier: [lo, hi) on megapixels (width×height). The NOT-NULL
        # guards drop unscanned rows (a NULL product would satisfy neither bound
        # cleanly), so a tier never leaks unscanned images. Composes with the sort.
        lo, hi = _RES_BOUNDS[res_bucket]
        area = BankImage.width * BankImage.height
        q = q.filter(BankImage.width.isnot(None), BankImage.height.isnot(None))
        if lo:
            q = q.filter(area >= lo)
        if hi is not None:
            q = q.filter(area < hi)
    return q, order


def list_images(user_id, bank_id, status=None, flag=None, cluster=None,
                group=None, style=None, subfolder=None, search=None,
                semantic_group=None, sort=None, res_bucket=None, framing=None,
                origin=None, medium=None, angle=None, ids=None, exclude=None,
                tags=None, ids_only=False, offset=0, limit=200) -> dict | None:
    """One PAGE of the bank grid (a 9 000-image bank must never ship whole).
    Filters compose: status ∩ flag ∩ cluster ∩ dup-group ∩ style ∩ subfolder ∩ search.
    ``search`` is a plain full-text term matched (case-insensitive LIKE) against the
    caption AND the relpath — so captions double as searchable tags for a big dump
    ("red dress"), combinable with every other filter.
    ``exclude`` is the INVERSE of that search and the reason both live here: a
    comma-separated list of terms, each HIDING the images whose caption or path
    mentions it. Searching answers "where is X"; excluding answers "what have I
    not done yet", which is how a captioned bank gets worked through as a
    checklist. Uncaptioned rows are never hidden by it (see _text_match).
    Flag filters sort by the relevant score (worst first) so the review reads
    top-down. ``sort`` (a GRID_SORTS id) overrides that order and covers EVERY
    quantity the passes persist — resolution (megapixels, so 900×900 outranks
    1200×300), file size, aesthetic rating, NSFW probability, sharpness, noise,
    flatness, detail ratio, letterbox bars, JPEG quality, face confidence — each
    way. Rows the matching pass never reached (NULL) always sink to the end, in
    BOTH directions (see _sort_order). It composes with every filter,
    and — since "Select all in filter" / ▶ Review page this SAME endpoint — the
    selection walks the order the user is looking at.
    ``res_bucket`` (a _RES_BUCKETS id) narrows to one resolution tier — a
    half-open [lo, hi) megapixel band — and composes with every filter AND the
    sort (the tier + Resolution↑/↓ combo is the mixed-dump cleanup flow).
    ``medium`` (a MEDIUM_KEYS id) narrows to one medium — what the picture is
    MADE of, from the 🎨 Medium pass — including 'unsure', which is a real
    verdict and has to be reachable. ``angle`` (an ANGLES id) narrows to one head
    angle, recomputed from the stored yaw at read time. Both compose with
    everything, like every other facet.
    ``ids`` is the "show selected" VIEW: an explicit ordered list of image ids
    that OVERRIDES every facet/sort (the selection IS the scope) and renders the
    page in the SAME order the caller passed — so a similarity ranking from
    ``select_similar`` shows reference-first, closest-to-farthest, instead of the
    default id order. Unknown/foreign ids are dropped silently."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    if ids is not None:
        # Explicit id view — the selection is the scope, order is preserved.
        # Dedupe keeping first occurrence so the requested order is authoritative.
        seen = set()
        ordered = [i for i in ids if not (i in seen or seen.add(i))]
        by_id = {}
        # Chunk the IN() so a big selection can't blow past SQLite's bound-
        # variable limit (default 999); the curation cap is 2 000 ids.
        for start in range(0, len(ordered), 500):
            chunk = ordered[start:start + 500]
            for r in (BankImage.query
                      .filter(BankImage.bank_id == bank_id,
                              BankImage.id.in_(chunk)).all()):
                by_id[r.id] = r
        ordered_rows = [by_id[i] for i in ordered if i in by_id]
        if ids_only:
            # Same scope, same order, ids only — the caller passed a selection and
            # wants it back minus the ids that no longer exist.
            return {'ids': [r.id for r in ordered_rows]}
        total = len(ordered_rows)
        off = max(0, int(offset))
        page = ordered_rows[off:off + max(1, min(500, int(limit)))]
        return {'images': _page_images(page, th), 'total': total, 'offset': off}
    q, order = _apply_facets(
        BankImage.query.filter_by(bank_id=bank_id), th,
        status=status, flag=flag, cluster=cluster, group=group,
        semantic_group=semantic_group, style=style, subfolder=subfolder,
        search=search, exclude=exclude, tags=tags, res_bucket=res_bucket,
        framing=framing, origin=origin, medium=medium, angle=angle)
    explicit = _sort_order(sort)
    if explicit is not None:
        # An explicit sort (resolution / aesthetic / sharpness) wins over the flag
        # worst-first order; see _sort_order for the NULL-sinks-last contract.
        order = explicit
    order_by = order if isinstance(order, tuple) else (order,)
    if ids_only:
        # The LEAN answer: the ids of the WHOLE filter, in the order above, in one
        # query and one response. ▶ Review and "Select all in filter" want a
        # snapshot of ids and nothing else; walking the paginated grid for it made
        # the browser ask 46 times for 16 MB of image payloads — thumbnails, flags,
        # captions, promotion state — and throw all but the integer away. Measured
        # on a 22 940-image bank: 3.8 s with an active measure sort, because every
        # one of those 46 pages re-ran the COUNT and re-applied the ORDER BY over
        # the full table with a growing OFFSET.
        # No pagination here on purpose: 23 000 ids are ~180 kB of JSON, where the
        # same 23 000 rows are 16 MB. The cap that matters is the bank's size, and
        # a bank that cannot fit its own ids in a response cannot fit its grid either.
        return {'ids': [r[0] for r in
                        q.with_entities(BankImage.id).order_by(*order_by).all()]}
    total = q.count()
    rows = q.order_by(*order_by).offset(max(0, int(offset))) \
            .limit(max(1, min(500, int(limit)))).all()
    return {'images': _page_images(rows, th), 'total': total,
            'offset': max(0, int(offset))}


def facet_counts(user_id, bank_id, **f) -> dict | None:
    """Every chip counter, measured under the filters ACTUALLY in force.

    Same shape (and the same keys) as the matching slices of ``bank_payload`` —
    'counts', 'flags', 'res_buckets', 'framing', 'origins', 'mediums', 'angles'
    — so the workspace can swap one for the other without a second code path.

    Why this exists. The payload's numbers are bank-wide, which was RIGHT for the
    question they were built to answer ("clicking this chip shows exactly these
    rows") and became wrong the moment any other facet was on. Measured on a
    50 397-image bank with ✕ Rejected picked: the chips offered "📺 Noisy 1 136"
    and "🔞 NSFW 20 540" over grids of 527 and 3 364 rows. Those numbers were
    never lies about the bank; they had simply stopped describing anything the
    user could see, which from where they sit is the same thing.

    Each facet is counted with EVERY OTHER filter applied and its own left out
    (``_apply_facets(skip=…)``). Counting a facet with itself applied would show
    the picked value's neighbours at 0 — a filter you cannot change your mind
    about without clearing everything first, which is a worse bug than the one
    being fixed.

    Cost: SEVEN queries whatever the bank's size — one per facet family, each a
    GROUP BY or a row of conditional SUMs. The bank-wide flag map it replaces
    spends one query PER FLAG (ten), so the filtered answer is cheaper than the
    unfiltered one. Nothing here writes."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    facets = {k: f.get(k) for k in FACETS}

    def pool(skip, extra=None):
        q, _order = _apply_facets(BankImage.query.filter_by(bank_id=bank_id), th,
                                  skip, **facets)
        return q if extra is None else q.filter(extra)

    def by_bucket(skip, col, keys, extra=None):
        """One GROUP BY over the pool. Rows the column maps to NULL are dropped:
        "never classified" is not a bucket, and folding it into one would let an
        unrun pass look like a measured verdict."""
        rows = (pool(skip, extra).with_entities(col, func.count(BankImage.id))
                .group_by(col).all())
        got = {k: n for k, n in rows if k is not None}
        return {k: int(got.get(k, 0)) for k in keys}

    def summed(q, conditions):
        """N conditional SUMs in ONE pass over the pool."""
        if not conditions:
            return []
        row = q.with_entities(*[
            func.coalesce(func.sum(case((c, 1), else_=0)), 0)
            for c in conditions]).one()
        return [int(v or 0) for v in row]

    total, pending, keep, reject = pool('status').with_entities(
        func.count(BankImage.id),
        *[func.coalesce(func.sum(case((BankImage.status == s, 1), else_=0)), 0)
          for s in ('pending', 'keep', 'reject')]).one()
    names = _QUALITY_FLAGS + _SCORE_FLAGS
    crits = {n: _flag_filter(n, th) for n in names}
    live = [n for n in names if crits[n] is not None]
    flags = dict.fromkeys(names, 0)
    flags.update(zip(live, summed(pool('flag'), [crits[n] for n in live])))
    # Same key as _flag_counts adds bank-wide, measured under the filters in
    # force here — the ✨ Clean chip prints this one like its six neighbours.
    flags['clean'] = summed(pool('flag'), [_clean_criterion(th)])[0]
    return {
        # 'total' is the filtered pool itself — what "All" would show. The other
        # three are its status split, each counted with the status facet lifted.
        'counts': {'total': int(total or 0), 'pending': int(pending or 0),
                   'keep': int(keep or 0), 'reject': int(reject or 0)},
        'flags': flags,
        'res_buckets': by_bucket(
            'res_bucket', _res_bucket_case(), [b for b, _lo, _hi in _RES_BUCKETS],
            extra=and_(BankImage.width.isnot(None), BankImage.height.isnot(None))),
        'framing': by_bucket('framing', BankImage.framing, _FRAMING_KEYS),
        'origins': by_bucket('origin', BankImage.origin, ORIGINS),
        'mediums': by_bucket('medium', BankImage.medium, MEDIUM_KEYS),
        'angles': by_bucket('angle', _angle_case(), ANGLES),
    }


# --- thumbnails -------------------------------------------------------------
def _thumb_path(bank_id, row: BankImage) -> Path:
    """Where this image's thumbnail lives. A watermark-cleaned or manually
    TURNED image gets its OWN thumbnail file, named after the cleaning method and
    the angle: the cached source thumbnail is never overwritten (so an undo — or
    a fourth quarter turn — instantly shows the original again) and the grid can
    never serve a stale pre-clean crop or a sideways tile. Deleting a cached
    thumbnail in place would be the fragile version of this — on Windows the file
    may still be held open by the response that just served it."""
    suffix = f'.{row.watermark_clean_method}' if row.watermark_clean_method else ''
    if getattr(row, 'rotation', None):
        suffix += f'.r{int(row.rotation)}'
    # A bank-side edit replaces the whole chain above, and each RE-edit must get
    # its own name too: re-cropping produces different pixels under an unchanged
    # method, so keying on the method alone would serve the first crop forever.
    if getattr(row, 'edit_method', None):
        suffix += f'.e{int(row.edit_generation or 0)}'
    return _thumbs_dir(bank_id) / f'{row.id}{suffix}.webp'


def drop_derived(bank_id, image_id) -> None:
    """Best-effort removal of every DERIVED blob of one image — the cleaned and
    turned thumbnails plus the turned full-size copies (an undo, a re-scan or a
    new clean just invalidated them). The pristine `<id>.webp` thumbnail of the
    untouched source is deliberately kept. A leftover is harmless — nothing
    points at it once the row's state moved on — so a locked file is not an
    error."""
    for pattern, folder in ((f'{image_id}.*.webp', _thumbs_dir(bank_id)),
                            (f'{image_id}.r*', _rotated_dir(bank_id)),
                            (f'{image_id}.*.r*', _rotated_dir(bank_id))):
        try:
            for stale in folder.glob(pattern):
                try:
                    stale.unlink()
                except OSError:
                    pass   # already gone or locked: a stale thumb owes nothing
        except OSError:
            pass   # already gone or locked: a stale thumb owes nothing


def _drop_analysis_thumbnails(bank_id, image_id) -> None:
    """Remove every thumbnail without touching the current full-size transform."""
    try:
        for stale in _thumbs_dir(bank_id).glob(f'{image_id}*.webp'):
            try:
                stale.unlink()
            except OSError:
                pass   # already gone or locked: a stale thumb owes nothing
    except OSError:
        pass   # already gone or locked: a stale thumb owes nothing


#: Historical alias — external callers/tests may still name the narrow version.
drop_clean_thumbs = drop_derived


def _clear_bank_pixel_analysis(
        row: BankImage, *, preserve_current_derivative=False) -> None:
    """Clear effective-byte lanes while preserving raw/user-owned history."""
    asserted_cluster = (row.face_cluster, row.face_cluster_origin) \
        if row.face_cluster_origin == 'asserted' else (None, None)
    for name in bank_transfer_metadata.BANK_PIXEL_DERIVED_FIELDS:
        setattr(row, name, None)
    if asserted_cluster[1] == 'asserted':
        row.face_cluster, row.face_cluster_origin = asserted_cluster
    row.analysis_fingerprint = None
    # A writer that has already validated the current rotated path must not
    # delete that very path while binding its fingerprint: doing so invalidates
    # the hash-bearing runtime cache the child just wrote. Explicit mutations
    # still take the default and discard every old turned copy.
    if preserve_current_derivative:
        _drop_analysis_thumbnails(row.bank_id, row.id)
    else:
        drop_derived(row.bank_id, row.id)
        try:
            (_thumbs_dir(row.bank_id) / f'{row.id}.webp').unlink(missing_ok=True)
        except OSError:
            pass   # already gone or locked: dropping a thumb is best-effort


def _clear_bank_watermark_analysis(
        row: BankImage, *, preserve_effective_derivative=False) -> None:
    """Clear raw-source watermark authority and any clean output it owns.

    A raw watermark verdict is independent from Score/Face/quality lanes.  When
    the row was never cleaned, rebinding that verdict must not delete a valid
    rotated derivative: it contains the same effective pixels those lanes
    measured.  A clean marker is different because removing it changes the
    effective image and therefore necessarily drops all derived copies.
    """
    had_clean = bool(row.watermark_clean_method)
    for name in bank_transfer_metadata.BANK_WATERMARK_ANALYSIS_FIELDS:
        setattr(row, name, None)
    row.watermark_fingerprint = None
    row.watermark_clean_method = None
    if had_clean:
        # Same reasoning as undo_watermark_clean: a bank-side edit made on top of
        # a cleaned image is made OF those pixels, so it cannot outlive them.
        _drop_edited_blob(row.bank_id, row)
    if had_clean or not preserve_effective_derivative:
        try:
            clean_image_path(row.bank_id, row.id).unlink(missing_ok=True)
        except OSError:
            pass   # already gone: the derived copy owed nothing
        drop_derived(row.bank_id, row.id)


def _invalidate_effective_analysis(
        row: BankImage, *, preserve_current_derivative=False) -> None:
    _clear_bank_pixel_analysis(
        row, preserve_current_derivative=preserve_current_derivative)
    row.width = None
    row.height = None


def _valid_analysis_fingerprint(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in '0123456789abcdef' for ch in value))


def _strict_semantic_group_for_generation(row: BankImage, path, fingerprint,
                                          engine) -> int | None:
    """Return one engine lane only under its own exact cache authority."""
    selected = bank_semantic_engine.normalize_engine(engine)
    group = getattr(row, f'{selected}_semantic_dup_group', None)
    if (group is None or not path
            or not _valid_analysis_fingerprint(fingerprint)):
        return None
    bank = db.session.get(ImageBank, row.bank_id)
    if bank is None:
        return None
    key = str(path)
    generation = bank_semantic_engine.cache_generation(bank, selected)
    memo_key = (bank.id, selected, generation)
    with _SEMANTIC_GROUP_PROOF_LOCK:
        memo = _semantic_group_proof_memo.get(selected)
        if memo is not None and memo[0] == memo_key:
            proofs = memo[1]
        else:
            embeddings = bank_semantic_engine.load_semantic_embeddings(
                bank, engine=selected)
            proofs = {
                candidate: bank_semantic_engine.embedding_fingerprint(
                    candidate, engine=selected)
                for candidate in embeddings
            }
            _semantic_group_proof_memo[selected] = (memo_key, proofs)
    return int(group) if proofs.get(key) == fingerprint else None


def _invalidate_write_generation(row: BankImage, path, fingerprint, *,
                                 preserve_current_derivative=True) -> None:
    """Invalidate shared lanes but retain independently proven engine history."""
    groups = {
        engine: _strict_semantic_group_for_generation(
            row, path, fingerprint, engine)
        for engine in ('clip', 'siglip2')
    }
    bank = db.session.get(ImageBank, row.bank_id)
    selected_engine = _selected_semantic_engine(bank) if bank else 'clip'
    _invalidate_effective_analysis(
        row, preserve_current_derivative=preserve_current_derivative)
    for engine, group in groups.items():
        if group is not None:
            setattr(row, f'{engine}_semantic_dup_group', group)
    row.semantic_dup_group = groups.get(selected_engine)


def _prepare_analysis_write(row: BankImage, path, measured_fingerprint) -> bool:
    """Authorise one lane write for the exact bytes a worker measured.

    The row is always reloaded by callers after inference.  This second digest
    closes external-file TOCTOU: a result for bytes A cannot be attached after
    the path became B.  On a real identity change every other derived lane is
    explicitly invalidated before this job writes its own result and the shared
    fingerprint in the same transaction.
    """
    if not _valid_analysis_fingerprint(measured_fingerprint):
        return False
    # ``path`` is what the worker was handed.  It is not necessarily what the
    # row resolves to NOW: a clean/rotation marker can change while inference is
    # in flight and the old derivative may remain readable at its old path.  A
    # digest of that submitted path alone would then authorise the wrong visual
    # generation.  Re-resolve the live row before considering the result.
    bank = db.session.get(ImageBank, row.bank_id)
    current_path = analysis_image_path(bank, row) if bank is not None else None
    if not _same_resolved_path(current_path, path):
        current_live = bank_transfer_metadata.content_fingerprint_path(current_path)
        # Preserve a newer already-authorised generation; otherwise make every
        # effective lane visibly empty until one of its workers measures current
        # bytes again.
        if row.analysis_fingerprint != current_live:
            _invalidate_write_generation(row, current_path, current_live)
            row.analysis_fingerprint = current_live
        return False
    live = bank_transfer_metadata.content_fingerprint_path(path)
    if live != measured_fingerprint:
        # Do not erase a newer, already-authorised row because an old worker
        # returned late.  When the row itself is stale, leave it as an empty
        # row tied to the bytes that are now live.
        if row.analysis_fingerprint != live:
            _invalidate_write_generation(row, path, live)
            row.analysis_fingerprint = live
        return False
    if row.analysis_fingerprint != measured_fingerprint:
        _invalidate_write_generation(row, path, measured_fingerprint)
    row.analysis_fingerprint = measured_fingerprint
    return True


def _restore_proven_siglip2_group(row: BankImage, submitted_path,
                                  preserved, *, selected_engine,
                                  accepted_fingerprint=None) -> bool:
    """Restore an independently SHA-proven SigLIP2 lane after CLIP invalidation."""
    if preserved is None:
        return False
    group, fingerprint = preserved
    current = accepted_fingerprint == fingerprint
    if not current:
        bank = db.session.get(ImageBank, row.bank_id)
        current_path = analysis_image_path(bank, row) if bank else None
        current = (
            _same_resolved_path(current_path, submitted_path)
            and bank_transfer_metadata.content_fingerprint_path(current_path)
            == fingerprint)
    if not current:
        return False
    row.siglip2_semantic_dup_group = group
    if selected_engine == 'siglip2':
        row.semantic_dup_group = group
    return True


def _prepare_watermark_write(row: BankImage, raw_path,
                             measured_fingerprint) -> bool:
    """Authorise watermark geometry for one exact raw, pre-rotation payload."""
    if not _valid_analysis_fingerprint(measured_fingerprint):
        return False
    live = bank_transfer_metadata.content_fingerprint_path(raw_path)

    def rebind(new_fingerprint):
        """Forget only generations that are actually stale.

        Score normally runs before Watermark in the full pipeline.  The first
        watermark attestation therefore must not erase the freshly measured
        effective lanes.  Even after a raw source replacement, a newer Score
        worker may already have rebound those lanes to the live effective bytes;
        comparing their shared fingerprint keeps that newer work intact.
        """
        bank = db.session.get(ImageBank, row.bank_id)
        current_path = analysis_image_path(bank, row) if bank is not None else None
        current_effective = bank_transfer_metadata.content_fingerprint_path(
            current_path)
        effective_is_current = (
            _valid_analysis_fingerprint(row.analysis_fingerprint)
            and row.analysis_fingerprint == current_effective)
        had_clean = bool(row.watermark_clean_method)
        _clear_bank_watermark_analysis(
            row, preserve_effective_derivative=not had_clean)
        if not effective_is_current or had_clean:
            _invalidate_effective_analysis(
                row, preserve_current_derivative=not had_clean)
        row.watermark_fingerprint = new_fingerprint

    if live != measured_fingerprint:
        if row.watermark_fingerprint != live:
            rebind(live)
        return False
    if row.watermark_fingerprint != measured_fingerprint:
        # NULL means this is the first raw-source attestation, not a generation
        # change.  No clean transform can be trusted without an attestation, but
        # an ordinary raw/rotated row keeps every already-proven effective lane.
        if row.watermark_fingerprint is None and not row.watermark_clean_method:
            bank = db.session.get(ImageBank, row.bank_id)
            current_path = (analysis_image_path(bank, row)
                            if bank is not None else None)
            current_effective = (
                bank_transfer_metadata.content_fingerprint_path(current_path))
            if (_valid_analysis_fingerprint(row.analysis_fingerprint)
                    and row.analysis_fingerprint != current_effective):
                _invalidate_effective_analysis(
                    row, preserve_current_derivative=True)
            row.watermark_fingerprint = measured_fingerprint
            return True
        rebind(measured_fingerprint)
    return True


def ensure_thumb(bank: ImageBank, row: BankImage) -> Path | None:
    """The image's grid thumbnail, generated lazily when the scan hasn't made
    it yet (so the grid is browsable straight after inventory). Built from the
    RESOLVED path, so a cleaned image shows clean in the grid."""
    tpath = _thumb_path(bank.id, row)
    if tpath.is_file():
        return tpath
    src = resolved_image_path(bank, row)
    if not src or not os.path.isfile(src):
        return None
    try:
        tpath.parent.mkdir(parents=True, exist_ok=True)
        with safe_bank_source(src, label='bank thumbnail') as im:
            im.draft(None, (THUMB_MAX_SIDE * 2, THUMB_MAX_SIDE * 2))
            im = im.convert('RGB')
            im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
            im.save(tpath, 'WEBP', quality=72)
        return tpath
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        return None


# --- quality scan (background) ----------------------------------------------
def _scan_one(src_root: str, thumbs: Path, item: tuple) -> dict:
    """Worker: decode ONE file, compute metrics + dHash + thumbnail. Pure
    filesystem/PIL — no DB access (the job thread owns the session)."""
    if len(item) == 2:  # compatibility for the pure worker unit test
        image_id, relpath = item
        path = os.path.join(src_root, relpath)
        tpath = thumbs / f'{image_id}.webp'
    else:
        image_id, path, thumb_path = item
        tpath = Path(thumb_path)
    out = {'id': image_id, 'quality_state': 'unreadable', 'width': None,
           'height': None, 'file_size': None, 'dhash': None, 'metrics': None,
           'provenance': None, 'fingerprint': None, 'path': path}
    if not path:
        out['quality_state'] = 'missing'
        return out
    try:
        out['file_size'] = os.path.getsize(path)
    except OSError:
        # ABSENT ≠ CORRUPT. A file that is simply not there says nothing about
        # the image — the folder moved, or its drive is unplugged. 'missing' is
        # an in-memory signal for the job loop only (it is never written to
        # quality_state, so the row stays unscanned and a later pass retries it).
        if not os.path.exists(path):
            out['quality_state'] = 'missing'
            return out
    try:
        # Fingerprint the bounded raw snapshot BEFORE Pillow validates it.  A
        # malformed image is still a stable generation we can safely label
        # ``unreadable`` and reject; without this digest the strict write fence
        # would leave it pending and rescan the same broken file forever.
        payload = _read_bounded_bank_source_bytes(path, label='bank scan')
        out['file_size'] = len(payload)
        out['fingerprint'] = bank_transfer_metadata.content_fingerprint_bytes(payload)
        _preserved_import_extension(payload, label='bank scan')
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as im:
                out['width'], out['height'] = im.size
                # JPEG fast path: decode at reduced scale — the metrics run on a
                # ≤1024 working copy anyway, and dHash (9×8) is resize-invariant.
                im.draft(None, (ANALYSIS_MAX_SIDE * 2, ANALYSIS_MAX_SIDE * 2))
                im.load()
                out['metrics'] = quality_metrics(im)
                # Provenance rides along on the SAME decode — re-opening the file for
                # it would double the I/O of a 36 000-image pass for nothing.
                out['provenance'] = provenance_metrics(im)
                out['dhash'] = f'{_dhash(im):016x}'
                if not tpath.is_file():
                    tpath.parent.mkdir(parents=True, exist_ok=True)
                    t = im.convert('RGB')
                    t.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
                    t.save(tpath, 'WEBP', quality=72)
        out['quality_state'] = 'ok'
    except (OSError, ValueError, SyntaxError, MemoryError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        # When bounded validation succeeded before Pillow rejected the image,
        # ``fingerprint`` still authorises the unreadable verdict.  A failure
        # before a complete raw snapshot leaves it NULL and parent writes none.
        pass
    return out


# --- THE SCOPE OF A PASS, in one place ---------------------------------------
#
# Every per-image pass used to hard-code `status != 'reject'` and offer nothing.
# The launch dialogs ask the user WHERE to run instead, so the filter became a
# parameter — and a parameter needs exactly one definition, because two copies of
# a pool filter is precisely how a button comes to announce a number it does not
# act on (the 🧹 Auto-reject "5 930 flagged / 0 moved" defect, and the reason
# _caption_scope_q was written this way in the first place).
#
# THE DEFAULT IS `None`, AND IT IS NOT "all". None keeps the historical
# `status != 'reject'` filter, so a caller that sends no scope posts the
# byte-identical request it posted before these options existed.
#
# 'reject' IS reachable now, and that is a change of principle held deliberately
# behind an explicit choice: the bin is work you already threw away, and every
# pass that reaches it spends real time (GPU, in most cases) on images you
# decided against. It is never a default, never part of the default, and the
# dialog that offers it says what it costs.
# PASS_SCOPES itself lives in pass_scopes.py — a leaf both surfaces import, so the
# dataset side never has to reach into this module for three words.


def _unscored_clause():
    """Rows the ✨ Score pass never reached — no aesthetic AND no NSFW value, so
    no cached CLIP embedding either. The 🎨 Medium pass can do nothing with them,
    and `medium_prereq` reads the same two columns to decide the bank has been
    scored at all: one definition, so the window's warning and the pass's
    "skipped (not scored yet)" can never disagree."""
    return and_(BankImage.aesthetic_score.is_(None),
                BankImage.nsfw_score.is_(None))


def _scope_clause(statuses):
    """The WHERE fragment for a scope. None → the historical non-rejected set."""
    if statuses is None:
        return BankImage.status != 'reject'
    return BankImage.status.in_(statuses)


def _scoped_pool(bank_id, statuses=None, ids=None):
    """The rows a pass may touch: this bank, this scope, and — when the user made
    a selection — only those ids. The selection is INTERSECTED with the scope,
    never widened, the same contract the caption pass has always had."""
    q = BankImage.query.filter_by(bank_id=bank_id).filter(_scope_clause(statuses))
    if ids:
        q = q.filter(BankImage.id.in_([int(i) for i in ids][:_SQL_IN_CHUNK]))
    return q


def _todo_by_status(bank_id, todo_clause):
    """{'keep': n, 'pending': n, 'reject': n} — how much work THIS pass has left
    in each pile, in ONE query.

    ``todo_clause`` must be the SAME expression the pass's pool filters on. That
    is the whole contract of this helper: a counter written from a second copy of
    the predicate is a counter that will disagree with the run, which is the one
    defect these dialogs exist to prevent. Shape and cost copied from _cap_sum —
    three conditional sums cost nothing on a 100 000-image bank."""
    def _sum(status):
        cond = BankImage.status == status
        if todo_clause is not None:
            cond = and_(cond, todo_clause)
        return func.coalesce(func.sum(case((cond, 1), else_=0)), 0)
    keep, pending, reject = (
        db.session.query(_sum('keep'), _sum('pending'), _sum('reject'))
        .filter(BankImage.bank_id == bank_id).one())
    return {'keep': int(keep or 0), 'pending': int(pending or 0),
            'reject': int(reject or 0)}


# The user-facing word for each pile. Same split every scope surface draws: the
# column stores 'keep'/'pending'/'reject', the reader sees these.
_PILE_WORDS = {'keep': 'kept', 'pending': 'undecided', 'reject': 'rejected'}


def scope_left_out(todo, statuses):
    """(n, words) — the work this pass HAS TO DO that its scope will not reach.

    THE DEFECT THIS ANSWERS. 🎨 Medium on a 50 397-image bank reported "0
    classified, 2 skipped (not scored yet)" in a few seconds and read as "nothing
    happened". Every figure in it was exact; the one that explained the screen —
    25 464 rejected images the default scope drops before the pool is even built
    (_scope_clause: status != 'reject') — was in no sentence anywhere.

    ``todo`` is a _todo_by_status mapping computed from the pass's OWN clause, so
    what is counted here is work, never mere population. Returns (0, '') when the
    scope reaches everything there is to do — a pass with nothing to report must
    stay quiet, or the note becomes noise on every run."""
    included = set(statuses) if statuses is not None else {'keep', 'pending'}
    out = [(p, int(todo.get(p) or 0)) for p in PASS_SCOPES if p not in included]
    out = [(p, n) for p, n in out if n > 0]
    if not out:
        return 0, ''
    return (sum(n for _p, n in out),
            ' + '.join(f'{n} {_PILE_WORDS[p]}' for p, n in out) if len(out) > 1
            else _PILE_WORDS[out[0][0]])


def _scope_note(bank_id, todo_clause, statuses, ids=None):
    """The sentence naming what the SCOPE left out, or '' when it left out
    nothing. Empty for a run aimed at a selection: there the user named the
    images one by one, and 'left out by the scope' would describe their own
    click back at them."""
    if ids:
        return ''
    n, words = scope_left_out(_todo_by_status(bank_id, todo_clause), statuses)
    if not n:
        return ''
    return f'; {n} image(s) left out by the scope ({words})'


def _skipped_note(*, vanished=0, missing=0, unanswered=0, fenced=0, stale=0,
                  unreadable=0) -> str:
    """The clauses a pass owes about the images it did NOT write, in ONE
    definition — same reasoning as `_framing_pool`, and the same failure when it
    was four copies.

    Every pass ends twice: once when it runs out of rows, once when the user
    stops it. Those two lines were written separately, so they drifted: the
    finished line named the failures and the CANCELLED one named only the
    successes. A 📐 Framing run stopped after 12 images, of which 8 could not
    reach the vision model (the GPU window kept expiring behind a
    `database is locked`), therefore reported "cancelled — 4 classified so far"
    and nothing else. Read literally, that says the other 8 were never processed;
    read as the user did, it says framing does not persist. Both endings now
    render these clauses from here, so a number can no longer exist in one and
    not the other.

    The wording is deliberately about the IMAGE, not about the internals: an
    image that changed under an analysis is "changed while the pass ran", never
    a fingerprint mismatch.
    """
    note = ''
    if vanished:
        note += f', {vanished} skipped (deleted while the pass ran)'
    if missing:
        note += f', {missing} skipped (the file was no longer on disk)'
    if unanswered:
        note += (f', {unanswered} not analysed (the vision model returned '
                 'nothing — check Ollama in Settings, then run it again)')
    if fenced:
        # NOT "unreadable": the file was fine and the model never saw it. The
        # row was left empty on purpose so a re-run finishes it, and that is the
        # half the user cannot guess.
        note += (f', {fenced} not analysed (the vision GPU window expired before '
                 'the model could start — run the pass again to finish them)')
    if stale:
        note += f', {stale} skipped (the image changed while the pass ran)'
    if unreadable:
        note += f', {unreadable} unreadable'
    return note


def start_scan(app, user_id, bank_id, rescan=False, regroup=False,
               statuses=None, ids=None):
    """Launch the quality pass. Raises BankJobBusy when a job is already live,
    ValueError when the bank is unknown.

    ``statuses`` / ``ids`` narrow WHAT IS MEASURED (see _scoped_pool). They do
    NOT narrow the duplicate re-grouping in the tail of the pass: that step
    re-reads every stored hash in the bank and renumbers all of it, because a
    grouping derived from a subset would renumber groups the user has already
    worked with. The dialog says so rather than letting the two read as one.

    ``regroup`` asks for the duplicate grouping EXPLICITLY, whatever the scan
    itself finds. That is what the 🎚 threshold panel's "↻ Re-group duplicates"
    sends: on an already-scanned bank its pool is empty, and the tail of the pass
    now only regroups when the stored hashes moved — without this flag that
    button would have quietly become a no-op."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    want = normalize_pass_statuses(statuses)
    total = _scan_pool(bank_id, rescan, want, ids).count()
    return bank_jobs.start(app, bank_id, 'scan',
                           _scan_job(bank_id, rescan, regroup=regroup,
                                     statuses=want, ids=ids),
                           total=total)


def _scan_todo_clause():
    """The "not measured yet" half of the scan pool, on its own — so the counter
    and the pool are one expression, never two that can drift."""
    return or_(BankImage.quality_state.is_(None),
               and_(BankImage.quality_state == 'ok', BankImage.origin.is_(None)))


def _scan_pool(bank_id, rescan, statuses=None, ids=None):
    """What the quality pass has to look at. With no scope given, rejected images
    are OUT like they always were: on a 30 000-image bank two thirds of a rescan
    went to shots the user had already thrown away.

    Skipping them cannot swallow a FIRST scan: an image is only rejected by hand
    (after it was scanned) or by this very pass when it turns out unreadable, so
    a never-scanned image is still pending here. Un-reject one and it comes back
    into the pool on its own — that is why the DEFAULT filter is `!= reject`
    rather than an explicit pending/keep list. A scope given explicitly replaces
    it, bin included, because at that point the user has said so."""
    q = _scoped_pool(bank_id, statuses, ids)
    if not rescan:
        # Never-scanned rows, PLUS rows a previous build scanned before the
        # provenance signals existed. Retrofitting the bank the user already has
        # is the point: telling them "only images scanned from now on get an
        # effective resolution" would leave a 36 000-image bank permanently half
        # measured, with no way to fix it short of a full rescan of everything.
        # `origin` is the sentinel because it is the one signal that always lands
        # on a readable file (one of ai/camera/unknown, never NULL) — keying off
        # detail_ratio would re-pick flat images forever, since those legitimately
        # measure nothing.
        q = q.filter(_scan_todo_clause())
    return q


def _scan_job(bank_id, rescan, regroup=False, statuses=None, ids=None):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = (_scan_pool(bank_id, rescan, statuses, ids)
                .order_by(BankImage.id.asc()).all())
        items = [
            (row.id, analysis_image_path(bank, row, refresh_rotation=True),
             str(_thumb_path(bank_id, row)))
            for row in rows
        ]
        bank_jobs.progress(job, done=0, total=len(items), detail='quality scan')
        thumbs = _thumbs_dir(bank_id)
        thumbs.mkdir(parents=True, exist_ok=True)
        src_root = bank.source_path
        if items and not os.path.isdir(src_root or ''):
            bank_jobs.fail(job, MOVED_FOLDER_MSG)
            return
        workers = min(8, os.cpu_count() or 4)
        done = 0
        missing = 0
        vanished = 0
        stale = 0       # measured, then the file changed before the write-back
        hashed = 0      # rows whose STORED hash actually changed (see the tail)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            it = iter(items)
            futures = deque()

            def submit_next():
                nxt = next(it, None)
                if nxt is not None:
                    futures.append(ex.submit(_scan_one, src_root, thumbs, nxt))

            for _ in range(workers * 2):
                submit_next()
            while futures:
                res = futures.popleft().result()
                if res['quality_state'] == 'missing':
                    # Leave the row EXACTLY as it was (unscanned, undecided) and
                    # count it. Grading an absent file would auto-reject it, and
                    # a folder that moved makes every file absent at once — that
                    # path silently rejects a whole bank, so it must not exist.
                    missing += 1
                    done += 1
                    bank_jobs.bump(job)
                    if missing >= _MISSING_ABORT_AT and missing * 2 >= done:
                        db.session.commit()
                        bank_jobs.fail(job, MOVED_FOLDER_MSG)
                        return
                    if not bank_jobs.cancelled(job):
                        submit_next()
                    continue
                # no_autoflush: this re-read is a SELECT, and SQLAlchemy flushes
                # the session before one by default. That flush takes the single
                # SQLite write lock at the FIRST image after a mutation and holds
                # it until the commit 25 images later, so every other writer in
                # the app queues behind a scan it has nothing to do with —
                # measured on a live bank: a sort click went from 5.0 ms median
                # to 327.6 ms (p90 2 183 ms, worst 5 564 ms, one HTTP 500 out of
                # 62 clicks). Suppressing the flush here took the worst wait to
                # 16.7 ms with zero failures and no change in scan throughput
                # (70-81 img/s): the rows are still written, by the commit that
                # already paces this loop.
                with db.session.no_autoflush:
                    row = _live_image(res['id'])
                if row is None:
                    # Deleted while the pass ran (see _live_image): skip it and
                    # keep going rather than sinking a scan of thousands.
                    vanished += 1
                else:
                    source_path = res.get('path')
                    if not _prepare_analysis_write(
                            row, source_path, res.get('fingerprint')):
                        stale += 1
                        done += 1
                        if done % _COMMIT_EVERY == 0:
                            db.session.commit()
                        bank_jobs.bump(job)
                        if not bank_jobs.cancelled(job):
                            submit_next()
                        continue
                    if row.dhash != res['dhash']:
                        hashed += 1
                    row.quality_state = res['quality_state']
                    row.width, row.height = res['width'], res['height']
                    row.dhash = res['dhash']
                    if res['metrics']:
                        row.blur_score = res['metrics']['blur_score']
                        row.noise_score = res['metrics']['noise_score']
                        row.uniformity_score = res['metrics']['uniformity_score']
                    if res['provenance']:
                        p = res['provenance']
                        row.detail_ratio = p['detail_ratio']
                        row.bars_ratio = p['bars_ratio']
                        row.jpeg_quality = p['jpeg_quality']
                        row.origin = p['origin']
                        row.origin_evidence = p['origin_evidence']
                    # An unreadable file can never be promoted — auto-reject it
                    # (only over 'pending': a manual decision is never flipped).
                    if res['quality_state'] == 'unreadable' and row.status == 'pending':
                        row.status, row.reject_reason = 'reject', 'unreadable'
                done += 1
                if done % _COMMIT_EVERY == 0:
                    db.session.commit()
                bank_jobs.bump(job)
                if not bank_jobs.cancelled(job):
                    submit_next()
        db.session.commit()
        if bank_jobs.cancelled(job):
            return
        tail = (f' — {missing} file(s) were not on disk and were left '
                'untouched') if missing else ''
        tail += _skipped_note(vanished=vanished, stale=stale)
        # Regroup only when the input to the grouping CHANGED, or when the
        # caller asked for it on purpose (the 🎚 panel's "↻ Re-group duplicates",
        # which is how a new dup_distance is applied without decoding anything).
        #
        # This is the whole freeze the owner reported. A scan with rescan off has
        # an empty-to-tiny pool on an already-scanned bank — measured 2 rows out
        # of 50 397 — and the tail then re-grouped the 50 389 hashed rows anyway:
        # the bar reached 100 %, said "grouping duplicates", and the app went
        # away for 96 to 124 s. When no stored hash moved and no row disappeared,
        # the grouping's input is byte-for-byte what produced the groups already
        # on screen, so running it can only reproduce them.
        if regroup or hashed or vanished:
            bank_jobs.progress(job, detail='grouping duplicates')
            groups = rebuild_dup_groups(
                bank_id, job=job, _bank_lease=_job_bank_capability(job))
            if bank_jobs.cancelled(job):
                return
            head = f'done — {groups} duplicate group(s)'
        elif done:
            head = f'done — {done} image(s) checked, no new hash to group'
        else:
            head = 'done — every image was already scanned'
        bank_jobs.progress(job, detail=f'{head}{tail}' + _scope_note(
            bank_id, None if rescan else _scan_todo_clause(), statuses, ids))
    return run


# --- duplicate groups -------------------------------------------------------
def _job_progress(job, **values):
    """bank_jobs.progress for a phase that also runs OUTSIDE a job (tests, and
    any future caller): no job, no reporting, no branch at every call site."""
    if job is not None:
        bank_jobs.progress(job, **values)


def _job_cancelled(job) -> bool:
    return job is not None and bank_jobs.cancelled(job)


def _popcount_lut():
    """256-entry uint8 popcount table, built once.

    numpy only grew `bitwise_count` in 2.0 and the app still ships on 1.26, so
    the table is the portable way to popcount a whole XOR block in one C pass —
    which is also the point: the GIL is released for the duration instead of
    being held by a Python loop over pairs."""
    global _POPCOUNT8
    if _POPCOUNT8 is None:
        import numpy as np
        _POPCOUNT8 = np.array([bin(i).count('1') for i in range(256)],
                              dtype='uint8')
    return _POPCOUNT8


def _dup_groups_from_hashes(hashes, d, job=None):
    """The grouping itself: 64-bit hashes in, groups of ≥2 POSITIONS out, biggest
    group first and ties broken by the smallest position. None when the job was
    stopped mid-way (nothing has been written at that point).

    Banded prefilter (pigeonhole: two hashes within Hamming d share at least one
    of d+1 equal bands) keeps this out of the full O(n²); candidate pairs are
    then verified exactly and grouped by union-find.

    WHY IT LOOKS LIKE `rebuild_semantic_dup_groups`. The pairs used to be walked
    in Python, with a `set` of every pair already looked at. That set is not
    bounded by anything: 13 MB at 2 000 images, 829 MB at 16 000, ~4.3 GB
    extrapolated at 36 000 — and the garbage collector sweeping it was ~79 % of
    the wall time (disabling the GC took one measured phase from 68 s to 14 s and
    a typical handler's p95 from 351 ms back to 22 ms). Comparing a whole block
    at once in numpy has no such set, releases the GIL while it works, and caps
    its own memory at `_DUP_BLOCK_CELLS`. The pair set it unions is EXACTLY the
    one the Python loops produced — same buckets, same `x < y`, same threshold —
    so the groups are identical, which `test_bank_dup_groups_identity.py` pins
    against a verbatim copy of the old code.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bands = max(1, min(16, d + 1))
    band_bits = 64 // bands
    buckets: dict = {}
    for i, h in enumerate(hashes):
        for b in range(bands):
            key = (b, (h >> (b * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(i)
    work = [m for m in buckets.values() if len(m) >= 2]
    total = sum(len(m) for m in work)
    _job_progress(job, done=0, total=total,
                  detail=f'grouping duplicates — comparing {n} hash(es)')
    # NumPy is an optional acceleration dependency. Most buckets are tiny and
    # need no array at all; importing it eagerly made an otherwise pure-Python
    # quality scan crash at the final grouping step in the lean Flask venv.
    np = None
    lut = None
    seen = 0
    for members in work:
        if _job_cancelled(job):
            return None
        m = len(members)
        if m >= _DUP_NUMPY_FROM and np is None:
            try:
                import numpy as np_module
                np = np_module
            except ImportError:
                np = False
        if m < _DUP_NUMPY_FROM or np is False:
            # Tiny buckets are the majority and an array costs more than the
            # handful of comparisons it would save.
            for x in range(m):
                for y in range(x + 1, m):
                    a, b = members[x], members[y]
                    if find(a) != find(b) and _hamming(hashes[a], hashes[b]) <= d:
                        union(a, b)
            seen += m
            _job_progress(job, done=seen)
            continue
        if lut is None:
            lut = _popcount_lut()
        arr = np.array([hashes[i] for i in members], dtype='uint64')
        cols = np.arange(m, dtype='int64')[None, :]
        step = max(1, min(m, _DUP_BLOCK_CELLS // m))
        for i0 in range(0, m, step):
            if _job_cancelled(job):
                return None
            block = arr[i0:i0 + step, None] ^ arr[None, :]
            rows_n = block.shape[0]
            dist = lut[block.view('uint8').reshape(rows_n, m, 8)].sum(
                axis=2, dtype='uint8')
            # `> row index` is the vectorised form of the old `for y in
            # range(x + 1, ...)`: the diagonal and the mirror pair are skipped,
            # so each unordered pair is offered to union-find exactly once.
            keep = (dist <= d) & (cols > np.arange(
                i0, i0 + rows_n, dtype='int64')[:, None])
            for a_rel, b in np.argwhere(keep):
                union(members[i0 + int(a_rel)], members[int(b)])
            seen += rows_n
            _job_progress(job, done=min(seen, total))
    comps: dict = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted((m for m in comps.values() if len(m) >= 2),
                  key=lambda m: (-len(m), m[0]))


def rebuild_dup_groups(bank_id, max_distance=None, job=None, *,
                       _bank_lease=None) -> int:
    """Recompute near-duplicate groups over every hashed image of the bank.
    Groups of ≥2 get a 1-based id ordered by size (biggest first). Returns the
    group count (what was written, when the job was stopped part-way).

    ``job`` makes the phase VISIBLE and STOPPABLE. It used to be neither: no
    progress, no cancel check anywhere in its body, so a bank of 50 000 images
    left the progress bar at 100 % under the words "grouping duplicates" for 96
    to 124 s with no way out — which reads as a dead application, and was
    reported as one."""
    if _bank_lease is None:
        with bank_jobs.mutation_lease(bank_id, 'duplicate_regroup') as lease:
            return rebuild_dup_groups(
                bank_id, max_distance=max_distance, job=job,
                _bank_lease=lease)
    bank_jobs.require_reservation(_bank_lease, bank_id)
    th = thresholds()
    d = int(th['dup_distance'] if max_distance is None else max_distance)
    bank = db.session.get(ImageBank, bank_id)
    if not bank:
        return 0
    rows = (BankImage.query
            .filter(BankImage.bank_id == bank_id, BankImage.dhash.isnot(None))
            .order_by(BankImage.id.asc()).all())
    # Capture the exact effective identity beside the dHash.  The comparison is
    # deliberately CPU-only and may take minutes on a large bank; a source file
    # can be replaced underneath us during that window even though Bank writes
    # are job-guarded.
    proven = []
    invalidated = False
    for row in rows:
        path = analysis_image_path(bank, row)
        live = bank_transfer_metadata.content_fingerprint_path(path)
        if live is not None and row.analysis_fingerprint == live:
            try:
                numeric_hash = int(row.dhash, 16)
            except (TypeError, ValueError):
                _invalidate_effective_analysis(row)
                invalidated = True
                continue
            proven.append((row.id, path, live, row.dhash, numeric_hash))
        elif row.analysis_fingerprint is not None and live is not None:
            _invalidate_effective_analysis(row)
            row.analysis_fingerprint = live
            invalidated = True
        elif row.analysis_fingerprint is not None:
            _invalidate_effective_analysis(row)
            invalidated = True
    if invalidated:
        # One changed member makes every old group id suspect.  Clearing the
        # partition in one statement is safer than leaving old peers pointing at
        # a group whose changed member has just been removed.
        BankImage.query.filter_by(bank_id=bank_id).update(
            {'dup_group': None}, synchronize_session=False)
    db.session.commit()
    ids = [item[0] for item in proven]
    hashes = [item[4] for item in proven]
    # Everything below the comparison is pure CPU over data already in memory,
    # so the session must not sit on a transaction across it — same guard, same
    # reason, as every inference pass (see _release_db_before_inference).
    _release_db_before_inference()
    groups = _dup_groups_from_hashes(hashes, d, job=job)
    if groups is None:
        return 0            # stopped before any write: the bank is untouched
    # Revalidate the COMPLETE input immediately before publishing the global
    # partition.  A per-row check while writing is too late: group numbers have
    # meaning only as one coherent result over this captured set.
    live_rows = {}
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        live_rows.update({
            row.id: row for row in BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        })
    partition_valid = True
    for image_id, expected_path, expected_fp, expected_dhash, _numeric in proven:
        row = live_rows.get(image_id)
        current_path = analysis_image_path(bank, row) if row is not None else None
        current_fp = bank_transfer_metadata.content_fingerprint_path(current_path)
        if (row is None or row.dhash != expected_dhash
                or row.analysis_fingerprint != expected_fp
                or current_fp != expected_fp
                or not _same_resolved_path(current_path, expected_path)):
            partition_valid = False
            # Do not erase a newer, correctly rebound row.  Only invalidate a
            # row that still advertises the stale identity we captured.
            if (row is not None and row.analysis_fingerprint == expected_fp
                    and current_fp != expected_fp):
                _invalidate_effective_analysis(row)
                row.analysis_fingerprint = current_fp
    if not partition_valid:
        BankImage.query.filter_by(bank_id=bank_id).update(
            {'dup_group': None}, synchronize_session=False)
        db.session.commit()
        return 0
    _job_progress(job, done=0, total=len(groups),
                  detail=f'grouping duplicates — writing {len(groups)} group(s)')
    BankImage.query.filter_by(bank_id=bank_id).update(
        {'dup_group': None}, synchronize_session=False)
    db.session.commit()
    # ONE prepared statement per batch, and a real pause between batches.
    #
    # The clear plus one `UPDATE ... WHERE id IN (…)` per group — about 5 000 of
    # them on a 50 000-image bank — used to be a single transaction holding the
    # write lock 6 to 9.5 s, past the 5 s busy_timeout: `database is locked` for
    # everybody else, 3 attempts out of 3. Splitting it into batches is only half
    # the fix, and the half that measures nothing on its own:
    #   • 4 000 separate UPDATE statements cost 0.67 s of pure statement overhead
    #     whatever the transaction boundaries are; one executemany over the same
    #     8 000 rows is ~30 ms, so the lock is barely taken at all;
    #   • and without the pause, the next batch re-takes the lock in the
    #     microsecond after the commit, inside the sleep of the other writer's
    #     busy handler. Batching alone left a concurrent writer waiting 620 ms
    #     and still refused — measured, and the reason this loop yields.
    # The cost of the trade is stated where the user reads it: groups land
    # biggest-first, and a stop part-way keeps the ones already written (the
    # panel already says "the groups are whatever it had reached").
    stmt = text('UPDATE bank_image SET dup_group = :gid WHERE id = :iid')
    pending: list = []

    def flush():
        if not pending:
            return
        db.session.execute(stmt, pending)
        db.session.commit()
        pending.clear()
        time.sleep(_DUP_WRITE_YIELD)

    for gid, members in enumerate(groups, start=1):
        pending.extend({'gid': gid, 'iid': ids[i]} for i in members)
        if len(pending) >= _DUP_WRITE_ROWS:
            flush()
            _job_progress(job, done=gid)
            if _job_cancelled(job):
                return gid
    flush()
    _job_progress(job, done=len(groups))
    return len(groups)


# --- semantic near-duplicate groups (stage 2 — crops / re-compressed variants) --
# One-entry memo for the parsed score cache. Reading it is 350 ms on a 14 700-row
# bank (40 MB .npz + a stat per row), and a user tuning a curation slider clicks
# three or four times on the SAME unchanged cache — that was 350 ms paid over and
# over for a file nobody touched. Bounded on purpose:
#   • ONE bank at a time (~45 MB of float32 at 14 700 × 768 — switching banks frees
#     the previous one rather than accumulating);
#   • keyed on the .npz's own (size, mtime_ns), so a finished ✨ Score pass — which
#     rewrites the file — invalidates it without anyone having to remember to;
#   • and it expires anyway after _SCORE_MEMO_TTL, because the per-row staleness
#     stats it skips are how a since-edited IMAGE gets dropped. A short window
#     covers the double-click; a session-long one would hide a real edit.
_SCORE_MEMO_TTL = 60.0
_score_memo = None            # (key, at, {path: emb}) — see reset_score_memo()
_score_hashes = {}            # {path: exact SHA-256 hex} for that one memo


def reset_score_memo() -> None:
    """Drop the parsed-score-cache memo (tests; bank deletion)."""
    global _score_memo, _score_hashes
    _score_memo = None
    _score_hashes = {}
    with _SEMANTIC_GROUP_PROOF_LOCK:
        _semantic_group_proof_memo.clear()
    bank_semantic_engine.reset_memo()


def _load_score_embeddings(bank: ImageBank) -> dict:
    """{abs_path: emb (np.float32, L2-normed)} from the ✨ Score pass cache, for the
    scored 'ok' images whose file still matches what was scored. Empty when the pass
    never ran (no cache) — the caller then surfaces the "run Score first" hint. A
    STALE entry (a same-path edit since scoring, detected via the cached size+mtime
    signature) is dropped, so a semantic group is never built on an outdated
    embedding. Reads the .npz directly (numpy is in the Flask venv); torch/open_clip
    are NOT needed here — stage 2 costs no new GPU work, it reuses Score's output."""
    global _score_memo, _score_hashes
    import numpy as np
    path = _score_cache_path(bank.id)
    if not path.is_file():
        return {}
    try:
        st = path.stat()
        key = (bank.id, str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and _score_memo is not None:
        mkey, at, cached = _score_memo
        if mkey == key and (time.time() - at) < _SCORE_MEMO_TTL:
            return cached
    try:
        with np.load(str(path), allow_pickle=False) as z:
            paths = [str(p) for p in z['paths']]
            states = [str(s) for s in z['states']]
            embs = z['embs']
            sigs = ([str(s) for s in z['sigs']] if 'sigs' in z.files
                    else [''] * len(paths))
            hashes = z['hashes'] if 'hashes' in z.files else None
            if (hashes is None or hashes.shape != (len(paths), 32)
                    or hashes.dtype != np.dtype('uint8')):
                return {}
    except Exception as e:  # noqa: BLE001 — a corrupt cache = "no embeddings", never fatal
        logger.warning('bank %s score cache unreadable: %s', bank.id, e)
        return {}
    out = {}
    exact_hashes = {}
    for i, p in enumerate(paths):
        if states[i] != 'ok':
            continue
        emb = embs[i]
        if float(np.abs(emb).sum()) <= 0:       # zero sentinel = errored image
            continue
        sig = sigs[i]
        if sig:                                 # drop a since-edited file
            try:
                st = os.stat(p)
                if f'{st.st_size}:{st.st_mtime_ns}' != sig:
                    continue
            except OSError:
                continue   # vanished mid-check: treated like a signature mismatch
        digest = hashes[i].tobytes()
        if (digest == b'\0' * 32
                or bank_transfer_metadata.content_fingerprint_path(p)
                != digest.hex()):
            continue
        out[p] = np.asarray(emb, dtype='float32')
        exact_hashes[p] = digest.hex()
    if key is not None:
        _score_memo = (key, time.time(), out)
    _score_hashes = exact_hashes
    return out


def _score_embedding_fingerprint(path) -> str | None:
    return _score_hashes.get(str(path))


def _selected_semantic_engine(bank) -> str:
    return bank_semantic_engine.normalize_engine(
        getattr(bank, 'semantic_engine', None))


def _load_semantic_embeddings(bank: ImageBank) -> dict:
    """Selected Bank embedding space, preserving the exact CLIP legacy seam."""
    engine = _selected_semantic_engine(bank)
    if engine == 'clip':
        # Exact call is load-bearing for historical tests/mocks and for the
        # byte-compatible Score cache loader above.
        return _load_score_embeddings(bank)
    return bank_semantic_engine.load_semantic_embeddings(bank, engine)


def _semantic_embedding_fingerprint(bank: ImageBank, path) -> str | None:
    engine = _selected_semantic_engine(bank)
    if engine == 'clip':
        return _score_embedding_fingerprint(path)
    return bank_semantic_engine.embedding_fingerprint(path, engine)


def _semantic_eligible_paths(bank: ImageBank) -> tuple[int, tuple[str, ...]]:
    """Return non-reject cache-path candidates without touching image bytes.

    SigLIP2 deliberately indexes the whole Bank so a later restore does not need
    fresh GPU work.  Readiness is narrower: rejected/deleted rows are not current
    semantic consumers and must not inflate the workspace's ``ok / total``.

    This runs in the workspace poll.  It must not call ``analysis_image_path``:
    that resolver hashes clean sources and can materialise/prune rotations.  Raw
    and clean paths are deterministic; already-materialised rotation filenames
    are enumerated once from the app-owned directory, with no image stat/SHA.

    ⏱ It also must not walk the disk. Two things keep it off it, and both are
    pinned by tests:
      * the bank folder is resolved ONCE, not once per row — the principle
        ``test_curation_selection_cost.py`` already pins for the curation pool;
      * each row's containment verdict is memoized per bank (:func:`_poll_path_memo`,
        read its warning), so the 2 s poll re-resolves only relpaths it has never
        seen. Steady state: zero filesystem calls for a 36 870-row pool, where it
        used to be 147 502.
    Only the four columns this needs are selected: hydrating whole ORM rows to
    read a relpath is the other half of the cost on a 50 000-image bank.
    """
    rows = (db.session.query(
        BankImage.id, BankImage.relpath, BankImage.rotation,
        BankImage.watermark_clean_method)
        .filter(BankImage.bank_id == bank.id, BankImage.status != 'reject')
        .order_by(BankImage.id.asc()).all())
    rotated = {}
    if any(row.rotation for row in rows):
        try:
            for candidate in _rotated_dir(bank.id).iterdir():
                if candidate.is_file():
                    prefix = candidate.name.split('.', 1)[0]
                    if prefix.isdigit():
                        rotated.setdefault(int(prefix), []).append(candidate)
        except OSError:
            pass   # unreadable entry: the rotation scan skips it
    base, resolved = _poll_path_memo(bank)
    paths = []
    for image_id, relpath, rotation, clean_method in rows:
        turn = int(rotation or 0)
        if turn:
            marker = f'.r{turn}.'
            paths.extend(str(candidate) for candidate in rotated.get(image_id, ())
                         if marker in candidate.name)
        elif clean_method:
            paths.append(str(clean_image_path(bank.id, image_id)))
        else:
            try:
                raw = resolved[relpath]
            except KeyError:
                raw = _abs_under(base, relpath)
                resolved[relpath] = raw
            if raw is not None:
                paths.append(raw)
    return len(rows), tuple(paths)


def _resolve_semantic_device() -> tuple[str, bool]:
    """Configured SigLIP2 device resolved against this exact ML Python."""
    from ..capabilities import bank_siglip2_gpu_available
    preference = str(cfg.get('bank_semantic.device') or 'auto').strip().lower()
    if preference not in ('auto', 'cpu', 'cuda'):
        preference = 'auto'
    cuda_available = bank_siglip2_gpu_available()
    use_gpu = preference != 'cpu' and cuda_available
    return ('cuda' if use_gpu else 'cpu'), use_gpu


def semantic_device_info() -> dict:
    preference = str(cfg.get('bank_semantic.device') or 'auto').strip().lower()
    if preference not in ('auto', 'cpu', 'cuda'):
        preference = 'auto'
    device, use_gpu = _resolve_semantic_device()
    return {'requested': preference, 'device': device, 'gpu': use_gpu}


def _semantic_text_status(engine) -> dict:
    from . import clip_text_encoder
    selected = bank_semantic_engine.normalize_engine(engine)
    try:
        status = (clip_text_encoder.status() if selected == 'clip'
                  else clip_text_encoder.status(engine=selected))
    except TypeError:
        # Rolling-upgrade compatibility until the paired encoder service lands.
        status = clip_text_encoder.status()
    out = dict(status or {})
    out.setdefault('available', False)
    out.setdefault('reason', None)
    out.setdefault('weights_warning', None)
    out.setdefault('warm', False)
    out['engine'] = selected
    out['model_label'] = bank_semantic_engine.engine_model_label(selected)
    if selected == 'siglip2':
        from ..capabilities import probe_bank_siglip2
        probe = probe_bank_siglip2()
        out['weights_warning'] = None
        if not probe.get('ok'):
            out['available'] = False
            out['reason'] = probe.get('detail') or 'SigLIP2 is not installed'
    return out


def semantic_engine_info(user_id, bank_id) -> dict | None:
    """Selected semantic space and safe cache coverage, with no local paths."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    engine = _selected_semantic_engine(bank)
    total, eligible_paths = _semantic_eligible_paths(bank)
    measured = semantic_counts(
        bank, engine=engine, total=total, eligible_paths=eligible_paths)
    counts = {key: measured[key] for key in ('total', 'cached', 'ok', 'stale')}
    return {
        'engine': engine,
        'model_key': measured['model_key'],
        'model_label': measured['model_label'],
        'dimension': measured['dimension'],
        'source': measured['source'],
        'ready': measured['ready'],
        'complete': measured['complete'],
        'needs_index': measured['needs_index'],
        'error': measured['error'],
        'counts': counts,
        'text': _semantic_text_status(engine),
        'device': (score_device_info(bank_id) if engine == 'clip'
                   else semantic_device_info()),
        # A selected SigLIP2 space never repurposes these CLIP-only products.
        'clip_owned': {'style': True, 'medium': True},
    }


def semantic_index_readiness(user_id, bank_id) -> dict | None:
    return semantic_engine_info(user_id, bank_id)


# Backward/rolling-upgrade aliases used by service and route tests.
semantic_readiness = semantic_index_readiness
semantic_engine_readiness = semantic_index_readiness


def _semantic_dup_lane(engine):
    """ORM column holding the durable partition for one semantic engine."""
    selected = bank_semantic_engine.normalize_engine(engine)
    return (BankImage.clip_semantic_dup_group if selected == 'clip'
            else BankImage.siglip2_semantic_dup_group)


def set_semantic_engine(user_id, bank_id, engine) -> dict | None:
    """Atomically swap the active duplicate projection under the mutation fence.

    Embedding caches are independent of these database lanes and are deliberately
    untouched.  The outgoing active partition is saved before the target lane is
    restored, so repeated CLIP/SigLIP2 switches are lossless.
    """
    if not isinstance(engine, str) or engine not in ('clip', 'siglip2'):
        raise ValueError('semantic engine must be clip or siglip2')
    selected = engine
    with bank_jobs.mutation_lease(bank_id, 'semantic_engine'):
        bank = get_bank(user_id, bank_id)
        if not bank:
            return None
        previous = _selected_semantic_engine(bank)
        if previous != selected:
            outgoing_lane = _semantic_dup_lane(previous)
            incoming_lane = _semantic_dup_lane(selected)
            bank.semantic_engine = selected
            BankImage.query.filter_by(bank_id=bank_id).update(
                {
                    outgoing_lane: BankImage.semantic_dup_group,
                    BankImage.semantic_dup_group: incoming_lane,
                },
                synchronize_session=False)
            db.session.commit()
            bank_semantic_engine.reset_memo()
    return semantic_engine_info(user_id, bank_id)


def _semantic_dup_threshold(engine, threshold=None) -> float:
    """One finite cosine threshold for explicit runs and persisted config."""
    if threshold in (None, ''):
        if engine == 'clip':
            raw = thresholds()['semantic_dup_threshold']
        else:
            raw = cfg.get('bank_semantic.siglip2_semantic_dup_threshold')
            if raw in (None, ''):
                raw = cfg.DEFAULTS['bank_semantic'][
                    'siglip2_semantic_dup_threshold']
    else:
        raw = threshold
    if isinstance(raw, bool):
        raise ValueError(
            'semantic duplicate threshold must be a number between 0 and 1')
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            'semantic duplicate threshold must be a number between 0 and 1') from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            'semantic duplicate threshold must be finite and between 0 and 1')
    return value


def _sdg_collect_items(bank, rows, path_by_id, emb_by_path, engine,
                       block_by_style, preserved_siglip2_groups):
    """Pair every live row with its cached embedding and block key. Moved
    verbatim from rebuild_semantic_dup_groups; skips rows whose analysis
    write cannot be prepared and restores proven SigLIP2 groups en route."""
    items = []
    for r in rows:
        p = path_by_id.get(r.id)
        emb = emb_by_path.get(p) if p else None
        if emb is None:
            continue
        fingerprint = _semantic_embedding_fingerprint(bank, p)
        prepared = _prepare_analysis_write(r, p, fingerprint)
        if engine == 'clip':
            _restore_proven_siglip2_group(
                r, p, preserved_siglip2_groups.get(r.id),
                selected_engine=engine,
                accepted_fingerprint=fingerprint if prepared else None)
        if not prepared:
            continue
        block = (r.style_cluster if r.style_cluster is not None else -1) \
            if block_by_style else 0
        items.append((r.id, block, emb, p, fingerprint, r.style_cluster))
    return items


def _sdg_group_by_similarity(items, t, _phase):
    """Union-find same-shot grouping over the style blocks (tiled exact
    cosine, bounded memory). Moved verbatim from rebuild_semantic_dup_groups;
    raises ValueError when the exact-pair budget is exceeded."""
    import numpy as np
    blocks: dict = {}
    for idx, (_id, block, _emb, _path, _fp, _style) in enumerate(items):
        blocks.setdefault(block, []).append(idx)
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_count = sum(
        len(members) * (len(members) - 1) // 2
        for members in blocks.values())
    if pair_count > _SEMANTIC_EXACT_PAIR_LIMIT:
        raise ValueError(
            'this semantic partition is too large for exact same-shot grouping '
            f'({pair_count:,} candidate pairs; safe limit '
            f'{_SEMANTIC_EXACT_PAIR_LIMIT:,}). Both semantic caches are unchanged. '
            'Use CLIP style-blocked grouping or split the Bank; a bounded ANN '
            'path is required for a larger global SigLIP2 partition.')

    compared = 0
    comparable = sum(len(m) for m in blocks.values() if len(m) >= 2)
    _phase(0, comparable, f'comparing {comparable:,} image(s) for same-shot groups')
    for members in blocks.values():
        if len(members) < 2:
            continue
        E = np.stack([items[i][2] for i in members]).astype('float32')
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        m = len(members)
        # Two-dimensional tiles keep similarity and match-index memory bounded
        # independently of Bank size.  ``flatnonzero`` sees at most 512 values;
        # unlike the former 512xN ``argwhere`` it can never materialise millions
        # of pairs in one allocation when a low threshold creates a dense graph.
        for i0 in range(0, m, _SEMANTIC_TILE):
            i1 = min(m, i0 + _SEMANTIC_TILE)
            for j0 in range(i0, m, _SEMANTIC_TILE):
                j1 = min(m, j0 + _SEMANTIC_TILE)
                tile_members = members[i0:i1] + members[j0:j1]
                first_root = find(tile_members[0])
                if all(find(member) == first_root for member in tile_members[1:]):
                    continue  # every possible edge is already represented
                sims = E[i0:i1] @ E[j0:j1].T
                same_tile = i0 == j0
                for local_a in range(i1 - i0):
                    start = local_a + 1 if same_tile else 0
                    hits = np.flatnonzero(sims[local_a, start:] >= t) + start
                    left = members[i0 + local_a]
                    for local_b in hits:
                        right = members[j0 + int(local_b)]
                        if find(left) != find(right):
                            union(left, right)
        compared += len(members)
        _phase(compared, comparable,
               f'comparing {comparable:,} image(s) for same-shot groups')
    comps: dict = {}
    for i in range(len(items)):
        comps.setdefault(find(i), []).append(i)
    groups = sorted((m for m in comps.values() if len(m) >= 2),
                    key=lambda m: (-len(m), items[m[0]][0]))
    return groups


def _sdg_partition_still_valid(bank, bank_id, items, engine, block_by_style,
                               cache_path, cache_generation, _phase):
    """Re-hash every member to prove nothing moved during the CPU phase.
    Moved verbatim from rebuild_semantic_dup_groups; repairs a stale CLIP
    analysis fingerprint en route. Returns (partition_valid,
    cache_still_current) - the whole partition is refused if any member
    changed or the cache rolled a generation."""
    try:
        cache_stat = cache_path.stat()
        cache_still_current = (
            cache_stat.st_size, cache_stat.st_mtime_ns) == cache_generation
    except OSError:
        cache_still_current = False
    item_ids = [item[0] for item in items]
    live_rows = {}
    for i0 in range(0, len(item_ids), _SQL_IN_CHUNK):
        live_rows.update({
            row.id: row for row in BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(item_ids[i0:i0 + _SQL_IN_CHUNK])).all()
        })
    partition_valid = cache_still_current
    # THE slow phase, and the one that used to run unannounced: it re-reads and
    # SHA-256s every file to prove none moved while the CPU comparison ran.
    _phase(0, len(items), f'verifying {len(items):,} file(s) are unchanged')
    for checked, (image_id, _block, _emb, expected_path, expected_fp,
                  expected_style) in enumerate(items, start=1):
        if checked % 200 == 0 or checked == len(items):
            _phase(checked, len(items),
                   f'verifying {len(items):,} file(s) are unchanged')
        row = live_rows.get(image_id)
        current_path = analysis_image_path(bank, row) if row is not None else None
        current_fp = bank_transfer_metadata.content_fingerprint_path(current_path)
        clip_invalid = (engine == 'clip' and row is not None
                        and row.analysis_fingerprint != expected_fp)
        if (row is None or clip_invalid or current_fp != expected_fp
                or not _same_resolved_path(current_path, expected_path)
                or (block_by_style and row.style_cluster != expected_style)):
            partition_valid = False
            if (engine == 'clip' and row is not None
                    and row.analysis_fingerprint == expected_fp
                    and current_fp != expected_fp):
                _invalidate_effective_analysis(row)
                row.analysis_fingerprint = current_fp
    return partition_valid, cache_still_current


def rebuild_semantic_dup_groups(bank_id, threshold=None, *,
                                _bank_lease=None, on_phase=None) -> int | None:
    """Stage-2 near-duplicate grouping in the Bank's selected semantic space.

    CLIP preserves the historical Score-cache/style-blocking path byte for byte;
    SigLIP2 reads its independent semantic cache and always compares globally,
    because ``style_cluster`` remains an explicitly CLIP-owned product.

    Cost: a semantic near-dup (cosine ≥ threshold) is necessarily inside one style
    union-find component (that clustering uses style_threshold ≤ threshold), so we
    BLOCK by style_cluster and only compare within a block — Σ block² dot-products,
    not the full n². A config with style_threshold > threshold would break that
    guarantee, so we fall back to a single global block then. Re-running at another
    threshold is CPU-only and near-instant: it re-reads the cached embeddings — no
    GPU, no re-scan."""
    if _bank_lease is None:
        with bank_jobs.mutation_lease(bank_id, 'semantic_dedup') as lease:
            return rebuild_semantic_dup_groups(
                bank_id, threshold=threshold, _bank_lease=lease, on_phase=on_phase)
    # The pass used to announce itself once and then work in silence — on a big
    # bank that is minutes of an empty bar, and the quietest phase is the
    # slowest one (proving nothing moved re-hashes every file). Each phase now
    # says what it is doing and fills the bar as it goes.
    def _phase(done, total, detail):
        if on_phase:
            on_phase(done, total, detail)
    bank_jobs.require_reservation(_bank_lease, bank_id)
    bank = db.session.get(ImageBank, bank_id)
    if not bank:
        return None
    engine = _selected_semantic_engine(bank)
    semantic_lane = _semantic_dup_lane(engine)
    cache_path = (_score_cache_path(bank_id) if engine == 'clip'
                  else _semantic_cache_path(bank_id))
    try:
        cache_stat = cache_path.stat()
        cache_generation = (cache_stat.st_size, cache_stat.st_mtime_ns)
    except OSError:
        return None
    emb_by_path = _load_semantic_embeddings(bank)
    if not emb_by_path:
        return None
    th = thresholds()
    t = _semantic_dup_threshold(engine, threshold)
    block_by_style = engine == 'clip' and th['style_threshold'] <= t
    rows = (BankImage.query.filter_by(bank_id=bank_id)
            .order_by(BankImage.id.asc()).all())
    path_by_id = {row.id: analysis_image_path(bank, row) for row in rows}
    preserved_siglip2_groups = (
        _preserved_siglip2_groups(
            bank_id, {path: row.id for row in rows
                      if (path := path_by_id.get(row.id)) is not None})
        if engine == 'clip' else {})
    # (image_id, block_key, embedding, path, fingerprint, style_cluster)
    items = _sdg_collect_items(bank, rows, path_by_id, emb_by_path, engine,
                               block_by_style, preserved_siglip2_groups)
    if not items:
        BankImage.query.filter_by(bank_id=bank_id).update(
            {BankImage.semantic_dup_group: None, semantic_lane: None},
            synchronize_session=False)
        db.session.commit()
        return 0
    # Do not retain a SQLite read/write transaction across the O(n²) CPU phase.
    _release_db_before_inference()
    groups = _sdg_group_by_similarity(items, t, _phase)
    partition_valid, cache_still_current = _sdg_partition_still_valid(
        bank, bank_id, items, engine, block_by_style, cache_path,
        cache_generation, _phase)
    if not partition_valid:
        BankImage.query.filter_by(bank_id=bank_id).update(
            {BankImage.semantic_dup_group: None, semantic_lane: None},
            synchronize_session=False)
        db.session.commit()
        if not cache_still_current:
            if engine == 'clip':
                reset_score_memo()
            else:
                bank_semantic_engine.reset_memo()
        return 0
    # Publish a complete replacement in one transaction only after validation.
    BankImage.query.filter_by(bank_id=bank_id).update(
        {BankImage.semantic_dup_group: None, semantic_lane: None},
        synchronize_session=False)
    for gid, members in enumerate(groups, start=1):
        member_ids = [items[i][0] for i in members]
        for i0 in range(0, len(member_ids), _SQL_IN_CHUNK):
            BankImage.query.filter(
                BankImage.id.in_(member_ids[i0:i0 + _SQL_IN_CHUNK])).update(
                {BankImage.semantic_dup_group: gid, semantic_lane: gid},
                synchronize_session=False)
    db.session.commit()
    return len(groups)


def _semantic_partition_signature(bank_id, threshold) -> str | None:
    """A cheap fingerprint of everything the grouping reads, so a pass can tell
    an untouched bank from a changed one WITHOUT redoing the work.

    Deliberately made of facts a query already knows — the embedding cache's
    size and mtime, the eligible row ids, their style blocks, the engine and the
    threshold. No file is opened: the whole point is to skip a phase that reads
    and SHA-256s every image in the bank.

    What it does NOT capture: an image REPLACED on disk at the same path while
    its cached embedding stayed behind. That cache is stale by then, and the
    full pass answers it the same way it always did — it refuses the partition
    and clears it. Users who suspect that hold ⇧ / pass ``force`` to skip the
    shortcut."""
    bank = db.session.get(ImageBank, bank_id)
    if bank is None:
        return None
    engine = _selected_semantic_engine(bank)
    cache_path = (_score_cache_path(bank_id) if engine == 'clip'
                  else _semantic_cache_path(bank_id))
    try:
        stat = cache_path.stat()
    except OSError:
        return None
    rows = (db.session.query(BankImage.id, BankImage.style_cluster)
            .filter(BankImage.bank_id == bank_id)
            .order_by(BankImage.id.asc()).all())
    h = hashlib.sha256()
    for image_id, style in rows:
        h.update(f'{image_id}:{"" if style is None else style};'.encode())
    return (f'{engine}/{float(threshold):.6f}/{stat.st_size}/{stat.st_mtime_ns}'
            f'/{len(rows)}/{h.hexdigest()[:32]}')


def start_semantic_dedup(app, user_id, bank_id, threshold=None, force=False):
    """Launch the CPU semantic near-duplicate pass in the selected space."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    engine = _selected_semantic_engine(bank)
    threshold = _semantic_dup_threshold(engine, threshold)
    if not _load_semantic_embeddings(bank):
        if engine == 'clip':
            raise ValueError('run ✨ Score first — semantic near-duplicates reuse '
                             'its embeddings')
        raise ValueError('run Semantic index first — no SigLIP2 embeddings are '
                         'available')
    return bank_jobs.start(app, bank_id, 'semantic_dedup',
                           _semantic_dedup_job(bank_id, threshold, force=force),
                           total=0)


def _semantic_dedup_job(bank_id, threshold, force=False):
    def run(job):
        bank_jobs.progress(job, done=0, total=0, detail='finding crops & variants')
        signature = _semantic_partition_signature(bank_id, threshold)
        previous = last_passes(db.session.get(ImageBank, bank_id)).get('semantic_dedup')
        if (not force and signature and previous
                and previous.get('signature') == signature):
            # Nothing the grouping reads has moved since that run. Saying so
            # costs a query; proving it the long way costs re-reading every
            # embedding and SHA-256-ing every file in the bank for an answer
            # already on screen.
            n = (previous.get('counts') or {}).get('semantic_groups', 0)
            bank_jobs.progress(
                job, done=0, total=0,
                detail=f'already up to date — {n} group(s), nothing changed')
            return
        n = rebuild_semantic_dup_groups(
            bank_id, threshold, _bank_lease=_job_bank_capability(job),
            on_phase=lambda done, total, detail: bank_jobs.progress(
                job, done=done, total=total, detail=detail))
        if n is None:
            bank_jobs.progress(job, detail='no embeddings — run ✨ Score first')
            return
        detail = f'done — {n} semantic near-duplicate group(s)'
        bank_jobs.progress(job, detail=detail)
        note_pass_run(bank_id, 'semantic_dedup', detail=detail,
                      counts={'semantic_groups': n},
                      engine=_selected_semantic_engine(db.session.get(ImageBank, bank_id)),
                      threshold=float(threshold),
                      signature=_semantic_partition_signature(bank_id, threshold))
    return run


def dup_groups_payload(user_id, bank_id, offset=0, limit=50,
                       col=BankImage.dup_group) -> dict | None:
    """Unresolved groups (≥2 non-rejected members) with their full membership,
    for the resolution panel. ``col`` picks the stage: dup_group (exact/resized) or
    semantic_dup_group (crops/variants)."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    gids = [g for (g,) in _unresolved_dup_groups_q(bank_id, col)
            .order_by(col.asc()).all()]
    total = len(gids)
    page = gids[max(0, int(offset)):max(0, int(offset)) + max(1, min(200, int(limit)))]
    groups = []
    for gid in page:
        rows = (BankImage.query.filter(BankImage.bank_id == bank_id, col == gid)
                .order_by(BankImage.id.asc()).all())
        groups.append({'group': gid,
                       'best_id': _best_of(rows).id if rows else None,
                       'images': _page_images(rows, th)})
    return {'groups': groups, 'total': total, 'offset': max(0, int(offset))}


def semantic_dup_groups_payload(user_id, bank_id, offset=0, limit=50) -> dict | None:
    """dup_groups_payload for stage 2 (semantic_dup_group)."""
    return dup_groups_payload(user_id, bank_id, offset=offset, limit=limit,
                              col=BankImage.semantic_dup_group)


def _best_of(rows):
    """'Keep best' heuristic for a duplicate group. When the aesthetic pass has
    run it leads (the ask: keep the NICE copy, not merely the biggest); a
    scored image always outranks an unscored one (sentinel -1 < the ~1..10 range).
    Then most pixels, sharpest, heaviest file — a Telegram dump's duplicates are
    mostly re-compressed or downscaled copies, so surface area is the honest
    fallback key."""
    def key(r):
        return (r.aesthetic_score if r.aesthetic_score is not None else -1.0,
                (r.width or 0) * (r.height or 0), r.blur_score or 0.0,
                r.file_size or 0, -r.id)
    return max(rows, key=key)


@_serialized_bank_mutation('resolve_dups')
def resolve_dups(user_id, bank_id, strategy='best', group=None, keep_ids=None,
                 col=BankImage.dup_group, attr='dup_group', reason='duplicate',
                 respect_existing_keep=True, snapshot=None, *,
                 _bank_lease=None):
    """Resolve duplicate groups: keep one member, REJECT the others (a status,
    never a file deletion, so it's reversible). strategy 'best'|'first' applies to
    one group or, when ``group`` is None, to every unresolved group at once;
    explicit ``keep_ids`` (manual pick) applies to their own groups. Only
    non-rejected members are touched. ``col``/``attr``/``reason`` pick the stage:
    dup_group (exact/'duplicate') or semantic_dup_group (crops/'semantic_dup').

    ``respect_existing_keep`` (default True) protects members the user already
    KEPT from a bulk resolve — right for the AUTOMATIC pipeline auto-reject, so a
    mass resolve never un-keeps a manual pick. An EXPLICIT resolve the user fired
    from a dup/same-shot group passes False: the whole point is to collapse the
    group to ONE, and the members of a same-shot group are typically ALL 'keep',
    so respecting keep would reject nobody. The elected keeper is always safe
    (``r.id in keep``); with False every OTHER member falls to reject, keep
    included. Returns {'resolved': groups, 'rejected': images}.

    ``snapshot``: pass a live :class:`bank_undo.Snapshot` to fold this resolve
    into a WIDER undo step (the pipeline's auto-reject is a flag pass plus this
    one); omit it and the call publishes its own one-step undo offer."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    own_snapshot = snapshot is None
    if own_snapshot:
        snapshot = bank_undo.Snapshot(
            'Resolve same-shot groups' if reason == 'semantic_dup'
            else 'Resolve duplicate groups')
    keep_by_group = {}
    if keep_ids:
        rows = BankImage.query.filter(BankImage.bank_id == bank_id,
                                      BankImage.id.in_(list(keep_ids)[:_SQL_IN_CHUNK])).all()
        for r in rows:
            g = getattr(r, attr)
            if g:
                keep_by_group.setdefault(g, set()).add(r.id)
        gids = list(keep_by_group)
    elif group is not None:
        gids = [int(group)]
    else:
        gids = [g for (g,) in _unresolved_dup_groups_q(bank_id, col).all()]
    resolved = rejected = 0
    for gid in gids:
        rows = (BankImage.query.filter(BankImage.bank_id == bank_id, col == gid)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        if len(rows) < 2 and gid not in keep_by_group:
            continue
        if gid in keep_by_group:
            keep = keep_by_group[gid]
        elif strategy == 'first':
            keep = {rows[0].id}
        else:
            keep = {_best_of(rows).id}
        changed = False
        for r in rows:
            if r.id in keep or (respect_existing_keep and r.status == 'keep'):
                continue
            snapshot.note(r, 'reject', reason)
            r.status, r.reject_reason = 'reject', reason
            rejected += 1
            changed = True
        if changed or len(rows) >= 2:
            resolved += 1
    db.session.commit()
    if own_snapshot:
        snapshot.commit(bank_id)
    return {'resolved': resolved, 'rejected': rejected}


def resolve_semantic_dups(user_id, bank_id, strategy='best', group=None,
                          keep_ids=None, respect_existing_keep=True,
                          snapshot=None, *, _bank_lease=None):
    """resolve_dups for stage 2 (semantic_dup_group, reject reason
    'semantic_dup')."""
    return resolve_dups(user_id, bank_id, strategy=strategy, group=group,
                        keep_ids=keep_ids, col=BankImage.semantic_dup_group,
                        attr='semantic_dup_group', reason='semantic_dup',
                        respect_existing_keep=respect_existing_keep,
                        snapshot=snapshot, _bank_lease=_bank_lease)


# --- statuses & flag application --------------------------------------------
_STATUS_UNDO_LABEL = {'keep': 'Keep images', 'reject': 'Reject images',
                      'pending': 'Set images back to undecided'}


@_serialized_bank_mutation('status')
def set_status(user_id, bank_id, ids, status, *, _bank_lease=None) -> int:
    """Manual keep/reject/pending on a selection. Returns rows changed.

    Snapshots the prior (status, reason) of every row it actually flips, so the
    workspace can offer ONE step back — this is the gesture that puts hundreds of
    decisions in flight at once (select the whole filter, then ✕)."""
    if status not in ('pending', 'keep', 'reject'):
        raise ValueError('bad status')
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ids = [int(i) for i in (ids or [])]
    reason = 'manual' if status == 'reject' else None
    snapshot = bank_undo.Snapshot(_STATUS_UNDO_LABEL[status])
    n = 0
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        for r in rows:
            snapshot.note(r, status, reason)
            r.status = status
            r.reject_reason = reason
            n += 1
    db.session.commit()
    snapshot.commit(bank_id)
    return n


@_serialized_bank_mutation('rotate')
def rotate_images(user_id, bank_id, ids, delta, *, _bank_lease=None) -> dict:
    """Turn a selection by ``delta`` degrees CLOCKWISE (idea by 1Tomber, #17).

    The user's files are NEVER written to: the new angle is stored on the row and
    the derived blobs (turned copy + thumbnail) are dropped so the ONE resolver
    rebuilds them from the pristine source on the next read. That is what makes
    the turn free of loss where it counts — a fourth quarter turn puts the row
    back at 0 and every reader is served the original bytes again, byte for byte.

    ``delta`` may be negative (-90 = turn left). Returns {'rotated': n,
    'rotations': {image_id: angle}} so the grid can re-render without a refetch.
    """
    from .face_dataset_service import normalize_rotation
    step = normalize_rotation(delta)
    if step == 0:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ids = [int(i) for i in (ids or [])]
    if not ids:
        raise ValueError('select at least one image')
    rotations = {}
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        for r in rows:
            r.rotation = (int(r.rotation or 0) + step) % 360 or None
            _invalidate_effective_analysis(r)
            rotations[r.id] = int(r.rotation or 0)
    db.session.commit()
    reset_score_memo()
    return {'rotated': len(rotations), 'rotations': rotations}


def _edit_state(row: BankImage) -> dict:
    """What the grid needs to re-render one edited row without a refetch: the
    marker, the generation (its cache-busting key) and the new geometry."""
    return {'id': row.id, 'edit_method': row.edit_method,
            'edit_generation': int(row.edit_generation or 0),
            'rotation': int(row.rotation or 0),
            'width': row.width, 'height': row.height}


def _measure_edited_blob(row: BankImage, blob: Path) -> None:
    """Bind the row's displayed dimensions to the blob we just wrote.

    ``_invalidate_effective_analysis`` empties width/height along with every
    other measured lane, which is right — they described the pre-edit pixels. But
    the grid lays its tiles out from them, so leaving them NULL until some later
    pass happens to run makes every freshly cropped tile collapse. These two are
    the one pair we can restore honestly and for free: we wrote the file, so its
    size is a fact, not an inference."""
    try:
        with safe_bank_source(str(blob), label='bank image edit') as im:
            row.width, row.height = image_encoding.visual_size_from_header(im)
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        pass   # an unreadable edit keeps the stored dimensions: size is a fact, not a guess


@_serialized_bank_mutation('crop')
def crop_image(user_id, bank_id, image_id, x, y, w, h, *,
               _bank_lease=None) -> dict | None:
    """Cut ONE bank image down to the box (x, y, w, h), in the pixels of the
    image as it is DISPLAYED. Returns the row's new edit state, or None when the
    image is unknown (404). Raises ValueError on an unusable box.

    Requested by nofaceman on Discord: the Bank has the filtering and the
    curation, so being sent to a Dataset (and back out through an export) just to
    reframe a shot is a detour. Cropping here means curate → crop → re-analyse →
    promote, in one place.

    NOTHING IS RESAMPLED. Unlike the dataset crop — which normalises the long
    side down to the training resolution, because a dataset image is training
    material — a bank is UPSTREAM of that choice. Shrinking here would decide the
    training resolution before the user has picked a dataset, and would do it
    invisibly. So this is a pure cut, published losslessly, and the dataset's own
    import stays the one place that normalises.

    The user's file is never written to (the crop is a blob in the bank's own
    ``edited/``), the previous generation is kept until the new one is published,
    and every measured lane is invalidated so the next pass re-reads THIS image —
    which is what makes "crop, then re-analyse, then curate" work at all."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    row = BankImage.query.filter_by(bank_id=bank_id, id=int(image_id)).first()
    if row is None:
        return None
    try:
        box = (int(x), int(y), int(x) + int(w), int(y) + int(h))
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid crop box') from exc
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        raise ValueError('invalid crop box')
    # The box was drawn on what the ONE resolver serves — cleaned and turned
    # included — so that is what has to be cut, and the turn is baked in by the
    # same move (see BankImage.edit_baked_rotation).
    src = resolved_image_path(bank, row)
    if not src or not os.path.isfile(src):
        raise ValueError('the image could not be read from its folder')
    from .face_dataset_service import _apply_watermark_crop
    generation = int(row.edit_generation or 0) + 1
    baked = (row.edit_baked_rotation if row.edit_method
             else (int(row.rotation or 0) % 360 or None))
    try:
        dst = _stage_edited_copy(bank_id, row, src, generation)
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError(f'the image could not be prepared for cropping: {exc}') from exc
    if not _apply_watermark_crop(str(dst), box):
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass   # rollback is best-effort: the crop target may never have landed
        raise ValueError('invalid crop box')
    row.edit_method = 'crop'
    row.edit_generation = generation
    row.edit_baked_rotation = baked
    row.rotation = None
    _invalidate_effective_analysis(row)
    _measure_edited_blob(row, dst)
    db.session.commit()
    # Only once the row POINTS at the new generation: pruning first would leave a
    # crash between the two with a marker and no blob.
    _prune_edited_generations(bank_id, row.id, dst)
    reset_score_memo()
    return _edit_state(row)


@_serialized_bank_mutation('edit_revert')
def revert_edits(user_id, bank_id, image_ids=None, *, _bank_lease=None) -> dict:
    """Throw away bank-side ✂ crop / ✨ improve results and go back to the
    image the bank started from. ``image_ids`` empty = every edited image.

    Free of consequence for the same reason the watermark undo is: the only
    thing deleted is a blob we made ourselves. The turn that the edit baked in
    is given back to the row rather than dropped with it."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    q = BankImage.query.filter_by(bank_id=bank_id).filter(
        BankImage.edit_method.isnot(None))
    if image_ids:
        ids = [int(i) for i in image_ids]
        q = q.filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK]))
    rows = q.all()
    for row in rows:
        _drop_edited_blob(bank_id, row)
        _invalidate_effective_analysis(row)
    if rows:
        db.session.commit()
        reset_score_memo()
    return {'reverted': len(rows),
            'images': [_edit_state(r) for r in rows]}


@_serialized_bank_mutation('apply_flags')
def apply_flags(user_id, bank_id, flags, snapshot=None, *,
                _bank_lease=None) -> dict:
    """Bulk-reject the PENDING images carrying the given flags. Manual ✓/✕
    decisions are never flipped (only status='pending' is touched) — same
    contract as the dataset auto-triage. Returns per-flag reject counts.

    This is the mis-set-threshold accident in one click, so it snapshots what it
    flipped. ``snapshot``: pass a live :class:`bank_undo.Snapshot` to fold the
    pass into a wider undo step (the pipeline does); omit it and the call
    publishes its own offer."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    own_snapshot = snapshot is None
    if own_snapshot:
        snapshot = bank_undo.Snapshot('Auto-reject by flag')
    th = thresholds()
    out = {}
    for flag in flags or []:
        if flag not in _QUALITY_FLAGS + _SCORE_FLAGS:
            continue
        crit = _flag_filter(flag, th)
        if crit is None:
            continue
        rows = (BankImage.query.filter_by(bank_id=bank_id, status='pending')
                .filter(crit).all())
        for r in rows:
            snapshot.note(r, 'reject', flag)
            r.status, r.reject_reason = 'reject', flag
        out[flag] = len(rows)
    db.session.commit()
    if own_snapshot:
        snapshot.commit(bank_id)
    return out


# --- ↩ undo the last bulk decision ------------------------------------------
_UNDO_NAME_SAMPLE = 8        # conflicting files quoted back so the user can find them


@_serialized_bank_mutation('undo')
def undo_last(user_id, bank_id, *, _bank_lease=None) -> dict:
    """Put every row the last bulk decision changed back to what it was.

    Three outcomes per row, all counted, because a restore that quietly missed
    half of its rows would be worse than no undo at all:

    * **restored** — the row is still there and still carries what the action
      set, so it goes back to its recorded prior value (status AND reason: the
      flag counters read the reason column);
    * **missing** — the row left the bank since (a re-scan dropped a file that
      disappeared from the folder);
    * **conflict** — someone changed it since (another tab, ▶ Review, a later
      pass). It is LEFT ALONE and named: overwriting a newer decision with an
      older one is not "undo", it is a second accident.

    Rows the action never touched are untouched here by construction — only the
    snapshot's ids are read. Synchronous even on a big lot: the work is one
    indexed SELECT plus in-place writes over the ids we already know, so a
    5 000-image restore lands in well under a second — a progress bar would take
    longer to render than the job it reports on.
    """
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    snap = bank_undo.take(bank_id)
    if not snap or not snap['rows']:
        raise ValueError('nothing to undo')

    entries = snap['rows']
    ids = list(entries)
    seen = set()
    restored = conflicts = 0
    conflict_names = []
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        for r in rows:
            seen.add(r.id)
            entry = entries[r.id]
            if (r.status, r.reject_reason) != tuple(entry['after']):
                conflicts += 1
                if len(conflict_names) < _UNDO_NAME_SAMPLE:
                    conflict_names.append(os.path.basename(r.relpath))
                continue
            r.status, r.reject_reason = entry['before']
            restored += 1
    db.session.commit()
    return {'label': snap['label'], 'total': len(ids), 'restored': restored,
            'missing': len(ids) - len(seen), 'conflicts': conflicts,
            'conflict_names': conflict_names}


# --- curation selectors (diversity · reference similarity) ------------------
# Both reuse the CLIP embeddings the ✨ Score pass already cached — no GPU, no
# re-scan (same contract as the semantic-dedup stage). They only ever build a
# SELECTION (a set of image ids the UI checks); the user reviews it before any
# Keep / Reject / Promote — nothing is mutated or deleted here.
_CURATION_MAX_N = 2000       # a curated LoRA set is 20–200 images; this is generous


def _pool_query(bank_id, th, *, status=None, flag=None, cluster=None,
                style=None, subfolder=None, search=None, exclude=None, tags=None,
                medium=None, angle=None):
    """The candidate-pool query for the curation selectors — the SAME filter
    composition as list_images (status ∩ flag ∩ cluster ∩ style ∩ subfolder ∩
    search ∩ NOT exclude), minus the ordering/pagination, so "give me 60 diverse
    images" is
    composable with whatever the grid is currently showing.

    Kept as its own function (a small, deliberate mirror of the list_images WHERE
    clauses) rather than a shared refactor: three curation-related branches touch
    this file in parallel, so an additive helper rebases clean where an edit to
    the list_images hot path would collide. When NO status is chosen the reject
    pile is excluded — you curate from what you might keep, never from the bin."""
    q = BankImage.query.filter_by(bank_id=bank_id)
    if status in ('pending', 'keep', 'reject'):
        q = q.filter(BankImage.status == status)
    else:
        q = q.filter(BankImage.status != 'reject')
    if flag == 'flagged':
        crits = [c for c in (_flag_filter(f, th) for f in _QUALITY_FLAGS)
                 if c is not None]
        q = q.filter(or_(*crits))
    elif flag == 'clean':
        q = q.filter(_clean_criterion(th))
    elif flag == 'dups':
        q = q.filter(BankImage.dup_group.isnot(None))
    elif flag == 'semantic_dups':
        q = q.filter(BankImage.semantic_dup_group.isnot(None))
    elif flag == 'no_face':
        q = q.filter(BankImage.face_state == 'no_face')
    elif flag in _QUALITY_FLAGS + _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
    if cluster is not None:
        q = q.filter(BankImage.face_cluster == int(cluster))
    if style is not None:
        q = q.filter(BankImage.style_cluster == int(style))
    # The two newest facets narrow a curation pick exactly as they narrow the
    # grid — "60 diverse images, among the photographs, three-quarter only" has
    # to mean the same thing in both places or the selection stops matching what
    # the user is looking at.
    if medium in MEDIUM_KEYS:
        q = q.filter(BankImage.medium == medium)
    if angle in ANGLES:
        q = q.filter(_angle_case() == angle)
    if subfolder is not None:
        if subfolder == '':
            q = q.filter(~BankImage.relpath.contains(os.sep))
        else:
            q = q.filter(BankImage.relpath.startswith(subfolder + os.sep))
    return _apply_text_filters(q, search, exclude, tags)


def _pool_embeddings(bank, emb_by_path, filters):
    """(ids, E) for the filtered pool rows that HAVE a cached embedding: ids is a
    list ordered by image id (so every tie-break below is deterministic), E is the
    matching (m×d) float32 matrix, L2-normalised. Empty ids ⇒ E is None."""
    import numpy as np
    rows = (_pool_query(bank.id, thresholds(), **filters)
            .order_by(BankImage.id.asc()).all())
    base = os.path.realpath(bank.source_path)   # once, not once per row
    prefix = os.path.normcase(base + os.sep)
    ids, vecs = [], []
    for r in rows:
        if r.watermark_clean_method or r.rotation:
            p = analysis_image_path(bank, r)
            emb = emb_by_path.get(p) if p else None
            if emb is not None:
                ids.append(r.id)
                vecs.append(emb)
            continue
        # Fast path: the keys of emb_by_path were THEMSELVES produced by
        # _abs_under (the ✨ Score pass walks the same rows), so a lexical
        # normpath that HITS the dict is provably the very string realpath would
        # have returned — no filesystem call needed to know that. A miss is the
        # only case that can be a symlink/junction, and it falls through to the
        # real resolution, so the result set is identical either way. That matters
        # because realpath is a syscall and this loop runs once per pool image:
        # 756 ms of a 6 353-row pool, against 6 ms for the lexical form.
        p = os.path.normpath(os.path.join(base, r.relpath))
        emb = emb_by_path.get(p) if os.path.normcase(p).startswith(prefix) else None
        if emb is None:
            p = _abs_under(base, r.relpath)
            emb = emb_by_path.get(p) if p else None
        if emb is not None:
            ids.append(r.id)
            vecs.append(emb)
    if not ids:
        return ids, None
    E = np.stack(vecs).astype('float32')
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return ids, E


_TYPICALITY_DEFAULT = 0.5    # see select_diverse — 0 restores pure farthest-point
_TYPICALITY_K = 10           # neighbours whose mean similarity IS the local density
_TYPICALITY_BLOCK = 512      # rows per similarity block — NEVER a full (m×m) matrix
_TYPICALITY_Z = 3.0          # robust deviations below the median density = full penalty
_TYPICALITY_DECADES = 3.0    # novelty discount at full guard + full penalty: 10⁻³
_TYPICALITY_MIN_POOL = 32    # under this a median/MAD "tail" is noise — guard off

_BLAS_GEMM = ...             # unprobed sentinel; None once probed and unavailable


def _fast_gemm_nt():
    """scipy's single-precision GEMM (``A @ B.T``), or None when unreachable.

    WHY this exists, measured rather than assumed: the numpy wheels this app runs
    on ship WITHOUT an optimised BLAS (``threadpoolctl.threadpool_info()`` returns
    an empty list), so ``E @ E.T`` runs single-threaded at ~5 GFLOP/s. On a real
    6 353-image pool that is 12.6 s for ONE similarity pass — 89 % of a curation
    click. scipy's wheels bundle OpenBLAS (24 threads here): the identical product
    takes 0.14 s, ~90× less, for the same 62 GFLOP.

    scipy is not in ``requirements.txt``, and it does not need to be: this lane
    only runs when numpy is present, numpy only arrives with
    ``requirements-ml.txt``, and insightface (the first line of that file) depends
    on scipy. So every install that CAN reach this code has the fast path, and the
    numpy fallback below is the belt-and-braces branch, not the normal one.

    Probed once per process and remembered — importing scipy.linalg is ~200 ms."""
    global _BLAS_GEMM
    if _BLAS_GEMM is ...:
        try:
            from scipy.linalg.blas import sgemm
            _BLAS_GEMM = sgemm
        except Exception as e:  # noqa: BLE001 — no scipy / broken wheel = slow path
            logger.info('no scipy BLAS for curation sampling (%s); '
                        'falling back to numpy matmul', e)
            _BLAS_GEMM = None
    return _BLAS_GEMM


def _sim_block(A, E):
    """``A @ E.T`` for L2-normed float32 rows — the similarity block of the
    typicality pass, routed through an optimised BLAS when one is reachable.

    Returns a C-contiguous (len(A) × len(E)) float32 array: scipy hands back a
    Fortran-ordered result, and the ``np.partition(..., axis=1)`` that follows
    walks rows, so the copy pays for itself several times over.

    Float caveat, stated because it is the one thing this change can affect: a
    different BLAS sums the same products in a different order, so a similarity
    can differ from numpy's by ~1e-6 (measured max absolute deviation over the
    real bank). That is far below any threshold this module compares against, and
    the selections were verified id-for-id identical on the production bank across
    n ∈ {20, 60, 200} and typicality ∈ {0.25, 0.5, 1.0} — but it is an equality
    of results, not of bits. ``typicality=0`` never reaches here at all, so the
    golden "historical behaviour" path is untouched by construction."""
    import numpy as np
    gemm = _fast_gemm_nt()
    if gemm is not None and A.dtype == np.float32 and E.dtype == np.float32:
        try:
            return np.ascontiguousarray(gemm(1.0, A, E, trans_b=True))
        except Exception as e:  # noqa: BLE001 — never fail a click over an optimisation
            logger.warning('scipy BLAS gemm refused the pool (%s); using numpy', e)
    return A @ E.T


def _isolation_penalty(E, *, k=_TYPICALITY_K, block=_TYPICALITY_BLOCK):
    """How ALONE each row is, as (m,) floats in [0, 1] — 0 for anything at or above
    the pool's median local density, 1 for the genuinely isolated tail.

    Local density = mean cosine similarity to the ``k`` nearest OTHER rows (the
    same cached embeddings the whole curation lane uses). Turning that density
    into a penalty is done on a ROBUST scale — median and MAD, not min/max — for
    two reasons that matter here:

      • a single meme in 24 000 photos would own the whole min/max range and
        squash every real difference to nothing;
      • everything at or above the median density gets penalty exactly 0, so the
        normal population of the bank is left strictly untouched. This selector
        DISCOUNTS the isolated tail, it never REWARDS the centre — which is what
        keeps a typicality guard from quietly turning "the 60 most varied" into
        60 look-alikes from the middle of the cloud.

    The ramp is quadratic, so being *slightly* below the median density barely
    costs anything and only the real tail is hit hard: at 1 robust deviation the
    penalty is 0.11, at 2 it is 0.44, at 3 and beyond it saturates at 1.

    Memory: the similarity pass runs in row blocks, so peak allocation is
    (block × m) float32 (~49 MB at m=24 000, measured 148 MB peak including
    temporaries) instead of the ~2.3 GB a full ``E @ E.T`` would need.
    Deterministic — median/MAD/partition, no sampling, no RNG.

    Time: this is an exact all-pairs pass, Θ(m²·d) — the same shape of work the
    semantic-dedup stage already does, and the reason the guard is computed ONLY
    when it is on. It is also, by a wide margin, the most expensive thing a
    curation click does; ``_sim_block`` explains why the product goes through
    scipy's BLAS rather than numpy's (12.6 s → 0.14 s on a 6 353-image pool,
    measured — the numpy this app ships on has no optimised BLAS, so the "~50×
    slower" case that paragraph used to describe as a hazard WAS the normal one).
    An approximation (subsampling the reference set) was rejected on purpose: it
    would make a small but legitimate group — eight shots of one rare outfit —
    look isolated and get penalised, which is exactly the variety this selector
    exists to preserve."""
    import numpy as np
    m = int(E.shape[0])
    k = min(int(k), m - 1)
    if k < 1 or m < _TYPICALITY_MIN_POOL:
        # Too few rows for a median and a MAD to mean anything: a 12-image pool
        # has no "isolated tail", only 12 images. Staying out is the honest
        # answer AND keeps small banks on the historical behaviour.
        return np.zeros(m, dtype='float32')
    dens = np.empty(m, dtype='float32')
    for a in range(0, m, block):
        S = _sim_block(E[a:a + block], E)    # (b, m) block — never (m, m)
        rows = np.arange(S.shape[0])
        S[rows, rows + a] = -np.inf          # a row is not its own neighbour
        top = np.partition(S, m - k, axis=1)[:, m - k:]
        dens[a:a + block] = top.mean(axis=1)
    med = float(np.median(dens))
    mad = float(np.median(np.abs(dens - med)))
    scale = 1.4826 * mad                     # MAD → σ-comparable, robust
    if not (scale > 1e-6):                   # degenerate pool (all alike) ⇒ no tail
        return np.zeros(m, dtype='float32')
    z = (med - dens) / scale                 # >0 only BELOW the median density
    ramp = np.clip(z / _TYPICALITY_Z, 0.0, 1.0)
    return (ramp * ramp).astype('float32')


def _farthest_point(E, factor, n):
    """Greedy farthest-point sampling over the L2-normed rows of ``E`` (m×d),
    returning ``n`` ROW POSITIONS in pick order. Seeded on position 0 — callers
    pass rows in ascending id order, so the seed and every tie-break resolve to
    the lowest id and the result is deterministic.

    ``factor`` is the per-row novelty multiplier in (0, 1] (the typicality guard,
    see ``_isolation_penalty``) or None to sample on pure max-min distance. Shared
    by ``select_diverse`` (whole pool) and ``select_balanced`` (one call per
    bucket, on a slice of the same E and the same pool-wide factor) so the guard
    behaves identically in both — there is exactly one copy of this loop."""
    import numpy as np
    m = int(E.shape[0])
    if n <= 0 or m == 0:
        return []
    if n >= m:
        return list(range(m))
    # min_dist[i] = cosine distance from row i to the NEAREST chosen row so far.
    chosen = [0]                                 # seed = lowest id (E[0])
    min_dist = 1.0 - E @ E[0]
    min_dist[0] = -np.inf                        # never re-pick a chosen row
    for _ in range(n - 1):
        score = min_dist if factor is None else min_dist * factor
        nxt = int(np.argmax(score))              # ties → lowest index = lowest id
        if not np.isfinite(score[nxt]):          # pool exhausted (all chosen)
            break
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - E @ E[nxt])
        min_dist[nxt] = -np.inf
    return chosen


def _verified_semantic_result_provenance(bank: ImageBank, captured_engine: str,
                                         captured_model_key: str) -> dict:
    """Refresh the selector before publishing an engine-bound result.

    A second browser tab can PATCH the Bank while NumPy ranks a large pool.
    Returning those old-space ids under the new selector would make the response
    internally plausible but semantically false, so fail closed and let it retry.
    """
    try:
        db.session.refresh(bank, attribute_names=['semantic_engine'])
    except Exception as exc:  # deleted/detached/concurrently unavailable
        raise ValueError(
            'the Bank changed during semantic selection — retry') from exc
    current_engine = _selected_semantic_engine(bank)
    current_model_key = engine_model_key(current_engine)
    if ((current_engine, current_model_key)
            != (captured_engine, captured_model_key)):
        raise ValueError(
            'the semantic engine changed during selection — retry in the '
            'current space')
    return {'engine': captured_engine, 'model_key': captured_model_key}


def select_diverse(user_id, bank_id, n=60, *, typicality=_TYPICALITY_DEFAULT,
                   filters=None):
    """Farthest-point sampling over the ✨ Score CLIP embeddings, tempered by a
    TYPICALITY guard: the ``n`` images of the (filtered) pool that best COVER the
    visual space — the antidote to a dump of 4 000 near-identical shots. Greedy
    FPS: seed with the lowest-id row (deterministic), then repeatedly add the
    point whose nearest already-chosen neighbour is FARTHEST (max-min cosine
    distance). The SAMPLING itself is O(n·m·d) — one (m×d)·(d,) product per pick,
    110 ms at m=6 353 / n=60 (measured).

    That figure used to be quoted as the cost of the WHOLE call ("~sub-second
    even at m=24 000"), and on real data it was wrong by a factor of thirty: the
    click took 32 s, of which the loop below was 0.1 s. The rest was the guard's
    all-pairs pass (see ``_isolation_penalty`` / ``_sim_block``). Whatever else
    changes here, keep this docstring measured — a comment promising a second
    where the user waits half a minute is a debt, not documentation.

    ``typicality`` (0–1) exists because pure max-min distance is, mathematically,
    the criterion that prefers ISOLATED points: on a collected bank the first
    picks are therefore structurally biased towards the aberrations (a meme, a
    photo of someone else, a botched frame) rather than towards variety of the
    subject. Each candidate's novelty is multiplied by
    ``10 ** (-3 × typicality × isolation)`` (see ``_isolation_penalty``), so an
    image still gets picked for the variety it adds, but being alone stops being
    a quality in itself. The discount is GEOMETRIC on purpose: an aberration's
    distance advantage over a normal image is a ratio (2–4× in practice, and it
    grows as the easy variety gets used up), so a merely linear penalty would need
    a near-maximal setting to ever bite.

      • 0    → EXACTLY the historical behaviour, pick for pick (the guard is not
               even computed);
      • 0.5  → the default: a saturated outlier keeps ~3% of its novelty (÷32) —
               decisively beaten by any genuinely varied shot — while a merely
               below-average image (penalty 0.11) keeps 68%;
      • 1    → the isolated tail is all but excluded (÷1000).

    The guard is bounded on BOTH sides: rows at or above the median density are
    never penalised at all (factor exactly 1.0), so it cannot collapse the
    selection into look-alikes from the middle of the cloud.

    Returns {'image_ids': [...] (sorted), 'pool': m, 'requested': n,
    'typicality': w}. Raises ValueError (→400, "run ✨ Score first") when no
    embedding exists yet, so the UI shows the clear hint instead of an empty,
    unexplained selection."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    engine = _selected_semantic_engine(bank)
    model_key = engine_model_key(engine)
    emb_by_path = _load_semantic_embeddings(bank)
    if not emb_by_path:
        if engine == 'clip':
            raise ValueError('run ✨ Score first — diversity sampling reuses its '
                             'embeddings')
        raise ValueError('run Semantic index first — diversity sampling needs '
                         'SigLIP2 embeddings')
    n = max(1, min(int(n), _CURATION_MAX_N))
    try:
        w = 0.0 if typicality is None else float(typicality)
    except (TypeError, ValueError):
        w = _TYPICALITY_DEFAULT
    w = max(0.0, min(1.0, w))
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    m = len(ids)
    if m <= n:                                   # whole pool already fits
        provenance = _verified_semantic_result_provenance(
            bank, engine, model_key)
        return {'image_ids': sorted(ids), 'pool': m, 'requested': n,
                'typicality': w, **provenance}
    # Novelty multiplier, in (0, 1] — never 0, so the -inf "already chosen"
    # sentinel below stays -inf (0 × -inf would be a NaN) and so even a fully
    # penalised row stays a last resort rather than an unpickable one.
    factor = None
    if w > 0.0:
        factor = 10.0 ** (-_TYPICALITY_DECADES * w * _isolation_penalty(E))
    chosen = _farthest_point(E, factor, n)
    provenance = _verified_semantic_result_provenance(
        bank, engine, model_key)
    return {'image_ids': sorted(ids[i] for i in chosen),
            'pool': m, 'requested': n, 'typicality': w, **provenance}


# --- balanced selection (coverage of the LABELS, not of the embedding space) --
# `select_diverse` answers "is my set VARIED?"; this one answers a different
# question no per-image score can ask: "does my set COVER what I want to be able
# to generate?". Asking for the 60 best/most varied of a bank that is 49% full
# body and 3.6% face shots returns those proportions — the LoRA then renders one
# framing well and the rest badly, with nothing having said so.
#
# WHICH AXIS. Measured on a real 43 000-image bank rather than assumed: `framing`
# had 13 000 rows classified across all four buckets (body 49%, bust 37%, back
# 11%, face 3.6%) — a discrete label with a real, actionable imbalance. Over the
# same bank `face_cluster` covered 4.7% of the rows and shattered them into 561
# clusters whose biggest held 34% — on a mono-subject bank a semantic/identity
# split is sparse and arbitrary, so balancing on it would spread a selection over
# noise. Hence: framing is the DEFAULT axis, and person is an explicit opt-in for
# the genuinely multi-subject dump.
_BALANCE_AXES = ('framing', 'framing+person')
_BALANCE_DEFAULT_AXIS = 'framing'   # stored in localStorage — never rename


def _balanced_quotas(sizes: dict, n: int) -> dict:
    """Split ``n`` picks as evenly as possible over the buckets, capped by what
    each one actually HAS (largest-remainder water-filling).

    A bucket that cannot serve its equal share is filled to the brim and its
    unused share is redistributed over the buckets that still have room — the
    ARBITRATION being: asking for 60 when a perfect split only yields 42 returns
    60, not 42, because throwing 18 usable images away buys a purity nobody asked
    for. What is forbidden is doing it SILENTLY: every caller gets ``fair_share``
    next to ``selected`` per bucket, so the top-up is visible as the deficit it
    is. Deterministic: buckets are walked in sorted key order, and a leftover
    single pick goes to the bucket with the most room left (key ascending on a
    tie)."""
    keys = sorted(sizes)
    quota = {k: 0 for k in keys}
    open_keys = [k for k in keys if sizes[k] > 0]
    remaining = min(int(n), sum(sizes.values()))
    while open_keys and remaining > 0:
        base = remaining // len(open_keys)
        if base == 0:                       # fewer picks left than buckets
            order = sorted(open_keys, key=lambda k: (-(sizes[k] - quota[k]), k))
            for k in order[:remaining]:
                quota[k] += 1
            break
        capped = [k for k in open_keys if sizes[k] - quota[k] <= base]
        if capped:
            for k in capped:
                take = sizes[k] - quota[k]
                quota[k] += take
                remaining -= take
            capped_set = set(capped)
            open_keys = [k for k in open_keys if k not in capped_set]
        else:
            for k in open_keys:
                quota[k] += base
            remaining -= base * len(open_keys)
    return quota


def _pool_labels(bank, filters) -> dict:
    """{image_id: (framing, face_cluster)} for the same pool ``_pool_embeddings``
    walks — one extra column read, no GPU."""
    rows = _pool_query(bank.id, thresholds(), **(filters or {})).all()
    return {r.id: (r.framing, r.face_cluster) for r in rows}


def _balance_axis_hint(axis, m, unlabelled, unknown) -> str:
    """The honest message for a bank that simply has not been labelled yet — the
    DEFAULT state of a fresh bank, not an error. Names the pass that is missing
    and the numbers, instead of returning an empty or misleading selection."""
    what = ('the shot type of each image' if axis == 'framing'
            else 'the shot type AND the person of each image')
    passes = ('run the 📐 Framing pass first' if axis == 'framing'
              else 'run the 📐 Framing and 👥 Group by person passes first')
    tail = ''
    if unknown and not unlabelled:
        tail = (f' — {unknown} of {m} came back as "unknown" framing, which is a '
                f'classification the balance cannot use')
    else:
        tail = f' — {unlabelled} of {m} images here have no label yet'
    return (f'{passes}: balanced selection needs {what}{tail}. '
            f'🎨 Pick diverse works without it.')


def select_balanced(user_id, bank_id, n=60, *, axis=_BALANCE_DEFAULT_AXIS,
                    typicality=_TYPICALITY_DEFAULT, filters=None):
    """Select ``n`` images SPREAD OVER the labels of ``axis`` instead of taking
    the top of one ranking: an even split across framings (and optionally across
    people), each bucket filled with the same farthest-point + typicality
    sampling ``select_diverse`` uses. So it is "the most varied 15 face shots,
    the most varied 15 busts, …" rather than "the most varied 60", which on a
    lopsided bank is 30 bodies and 2 faces.

    It ACCOMPANIES ``select_diverse``, it does not replace it: variety inside a
    space and coverage of a label are different questions, and the diverse
    selector still works on a bank with no labels at all (which is most banks
    until the 📐 Framing pass has run).

    Composition with the typicality guard: the isolation penalty is computed ONCE
    over the WHOLE filtered pool and then sliced per bucket — deliberately, since
    "alone in the bank" is a property of the bank. Computing it per bucket would
    make every member of a small bucket look isolated and penalise exactly the
    images the balance exists to bring in.

    Returns {'image_ids', 'pool', 'requested', 'selected', 'typicality', 'axis',
    'buckets': [{key, framing, cluster, available, fair_share, selected, short}],
    'unlabelled', 'unknown', 'shortfall'}. Raises ValueError (→400) when Score
    has not run, or when nothing in the filter carries the axis label."""
    import numpy as np
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if axis not in _BALANCE_AXES:
        axis = _BALANCE_DEFAULT_AXIS
    engine = _selected_semantic_engine(bank)
    model_key = engine_model_key(engine)
    emb_by_path = _load_semantic_embeddings(bank)
    if not emb_by_path:
        if engine == 'clip':
            raise ValueError('run ✨ Score first — balanced selection reuses its '
                             'embeddings')
        raise ValueError('run Semantic index first — balanced selection needs '
                         'SigLIP2 embeddings')
    n = max(1, min(int(n), _CURATION_MAX_N))
    try:
        w = 0.0 if typicality is None else float(typicality)
    except (TypeError, ValueError):
        w = _TYPICALITY_DEFAULT
    w = max(0.0, min(1.0, w))
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    m = len(ids)
    if not m:
        raise ValueError('nothing to select from in the current filter')

    labels = _pool_labels(bank, filters or {})
    buckets, meta = {}, {}
    unlabelled = unknown = 0
    for pos, iid in enumerate(ids):
        fr, cl = labels.get(iid, (None, None))
        if fr is None:
            unlabelled += 1
            continue
        if fr not in _FRAMINGS:              # 'unknown' — a real classification,
            unknown += 1                     # but not one you can balance on
            continue
        if axis == 'framing+person':
            if cl is None:
                unlabelled += 1
                continue
            key = f'{fr}#{int(cl)}'
            meta[key] = (fr, int(cl))
        else:
            key = fr
            meta[key] = (fr, None)
        buckets.setdefault(key, []).append(pos)
    if not buckets:
        raise ValueError(_balance_axis_hint(axis, m, unlabelled, unknown))

    sizes = {k: len(v) for k, v in buckets.items()}
    quota = _balanced_quotas(sizes, n)
    # What a split with no ceiling WOULD have given each bucket — the yardstick
    # a shortfall is reported against ("back: 3 of an even 15").
    fair = _balanced_quotas({k: n for k in sizes}, n)

    factor = None
    if w > 0.0:
        factor = 10.0 ** (-_TYPICALITY_DECADES * w * _isolation_penalty(E))
    chosen, report = [], []
    for key in sorted(sizes):
        pos = buckets[key]                   # ascending id order (pool order)
        take = quota[key]
        if take >= len(pos):
            picked = list(pos)
        elif take <= 0:
            picked = []
        else:
            idx = np.asarray(pos)
            sub = np.ascontiguousarray(E[idx])
            subf = None if factor is None else np.ascontiguousarray(factor[idx])
            picked = [pos[i] for i in _farthest_point(sub, subf, take)]
        chosen.extend(picked)
        fr, cl = meta[key]
        report.append({'key': key, 'framing': fr, 'cluster': cl,
                       'available': len(pos), 'fair_share': fair[key],
                       'selected': len(picked),
                       'short': len(pos) < fair[key]})
    order = {k: i for i, k in enumerate(_FRAMINGS)}
    report.sort(key=lambda b: (order.get(b['framing'], 99),
                               b['cluster'] if b['cluster'] is not None else -1))
    provenance = _verified_semantic_result_provenance(
        bank, engine, model_key)
    return {'image_ids': sorted(ids[i] for i in chosen),
            'pool': m, 'requested': n, 'selected': len(chosen),
            'typicality': w, 'axis': axis, 'buckets': report,
            'unlabelled': unlabelled, 'unknown': unknown,
            'shortfall': max(0, n - len(chosen)), **provenance}


def select_similar(user_id, bank_id, ref_id, n=60, min_score=None, *, filters=None):
    """Rank the (filtered) pool by CLIP cosine similarity to a REFERENCE bank image
    (its own cached ✨ Score embedding) — "keep what looks like THIS", to pull one
    person / look out of a mixed dump. Returns the top-``n`` most similar ids, OR
    everything with cosine ≥ ``min_score`` when that is given; the reference itself
    (cosine 1.0) is always included. Reuses the cached embeddings — no GPU.

    Returns {'results': [{id, score}], 'image_ids': [...], 'pool': m, 'ref_id'}.
    Raises ValueError (→400) when Score hasn't run or the reference has no cached
    embedding (e.g. it was rejected before Score, or edited since)."""
    import numpy as np
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    engine = _selected_semantic_engine(bank)
    model_key = engine_model_key(engine)
    emb_by_path = _load_semantic_embeddings(bank)
    if not emb_by_path:
        if engine == 'clip':
            raise ValueError('run ✨ Score first — reference similarity reuses '
                             'its embeddings')
        raise ValueError('run Semantic index first — reference similarity needs '
                         'SigLIP2 embeddings')
    ref = db.session.get(BankImage, int(ref_id))
    if ref is None or ref.bank_id != bank_id:
        raise ValueError('reference image not found in this bank')
    ref_path = analysis_image_path(bank, ref)
    ref_emb = emb_by_path.get(ref_path) if ref_path else None
    if ref_emb is None:
        if engine == 'clip':
            raise ValueError('the reference image has no ✨ Score embedding — '
                             'score it first (it may have been rejected before '
                             'Score ran)')
        raise ValueError('the reference image has no SigLIP2 embedding — run '
                         'Semantic index first')
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    if not ids:
        provenance = _verified_semantic_result_provenance(
            bank, engine, model_key)
        return {'results': [], 'image_ids': [], 'pool': 0,
                'ref_id': int(ref_id), **provenance}
    rv = np.asarray(ref_emb, dtype='float32')
    rv /= (np.linalg.norm(rv) + 1e-8)
    sims = E @ rv                                 # cosine similarity, (m,)
    order = np.argsort(-sims, kind='stable')     # desc; stable ⇒ id tie-break
    if min_score is not None:
        keep = [int(k) for k in order if sims[k] >= float(min_score)]
    else:
        n = max(1, min(int(n), _CURATION_MAX_N))
        keep = [int(k) for k in order[:n]]
    results = [{'id': ids[k], 'score': round(float(sims[k]), 4)} for k in keep]
    provenance = _verified_semantic_result_provenance(
        bank, engine, model_key)
    return {'results': results, 'image_ids': [ids[k] for k in keep],
            'pool': len(ids), 'ref_id': int(ref_id), **provenance}


# ── 🔤 pushing an attribute DOWN a text ranking ──────────────────────────────
# CLIP has no negation: "a woman without a bikini" ranks bikinis HIGHER, because
# the word is ignored rather than applied (measured: 60% of the top 60 carried
# a bikini against a 10.1% base rate — an inversion, not a miss). What the
# embedding space DOES support is arithmetic, so the exclusion is subtracted
# instead of spoken: score = sim(positive) − weight · sim(excluded).
#
# The weight below was calibrated on 2026-08-01 over 7,316 real bank images that
# carry BOTH a cached CLIP embedding and a vision caption — the caption supplies
# the ground truth (a different model from CLIP, so the labels are not circular).
# 19 positive/excluded pairs, top-60 measured against the app's own default n:
#
#   weight   top-60 carrying the excluded trait   top-60 still on-topic
#     0.00              23.0% (mean)                    89.7% (mean)
#     0.30              11.9%                           89.5%
#     0.50               8.9%                           88.5%
#     0.60               7.6%                           87.7%   ← the knee
#     0.75               6.3%                           85.8%
#     1.00               3.8%                           79.8%
#     1.50               1.4%                           64.4%
#
# Exclusion improves all the way up; RELEVANCE is what buys it, and it holds
# essentially flat to 0.6 (−2.0 points) then falls off a cliff (−9.9 at 1.0,
# −25.3 at 1.5). 0.6 is where two thirds of the unwanted trait is gone for a
# cost the ranking does not feel. The extremes are offered as Gentle/Strong
# because one measured case — excluding "a bikini" from "a woman at the beach"
# — is INSEPARABLE (the trait is most of what the positive query means), and no
# weight fixes that: at 0.6 it still returned 66.7% bikinis, and by the time the
# weight bites the beach is gone too. That case is why the UI promises a
# push-down and never an absence.
PUSH_DOWN_WEIGHT_DEFAULT = 0.6
PUSH_DOWN_WEIGHT_MAX = 2.0


def _push_down_weight(value):
    """The requested push-down strength, clamped. Anything unreadable falls back
    to the calibrated default rather than to 0 — a silently ignored exclusion
    would look exactly like an exclusion that found nothing."""
    try:
        w = float(value)
    except (TypeError, ValueError):
        return PUSH_DOWN_WEIGHT_DEFAULT
    if w != w:                                   # NaN
        return PUSH_DOWN_WEIGHT_DEFAULT
    return max(0.0, min(w, PUSH_DOWN_WEIGHT_MAX))


def search_by_text(user_id, bank_id, query, n=60, *, push_down=None,
                   push_down_weight=None, filters=None):
    """Rank the (filtered) pool by CLIP similarity to a written QUERY — "brunette
    outdoors, wide shot" instead of a reference picture.

    Mechanically this is ``select_similar`` with the reference vector produced by
    CLIP's TEXT tower rather than read from the image cache: same embeddings, same
    cosine, same deterministic stable-argsort ordering. Ranking costs no GPU and no
    re-scan; only encoding the phrase leaves the Flask process (see
    clip_text_encoder, which caches every phrase on disk).

    It REFINES the current filter rather than replacing it — the candidate pool is
    exactly what the grid is showing, so "wide shot, inside this subfolder, among
    the undecided" composes without a second search grammar.

    TOP-N ONLY — there is deliberately no similarity threshold, and adding one
    would be a mistake rather than a feature. Measured on a real bank (48 images,
    8 unrelated datasets, this exact model): verified-correct top-1 hits scored
    0.177–0.233, while guaranteed-unrelated pairs reached up to 0.197 (median
    0.112). The two distributions OVERLAP — the unrelated ceiling outranks two
    genuinely correct answers. No cut exists that separates "relevant" from
    "unrelated": below ~0.20 it admits false positives, above ~0.18 it discards
    true ones. A threshold control would therefore offer the illusion of a
    boundary that does not exist, so the ranking is the whole product.

    ``push_down`` (or ``-term`` inside the query) names what to push DOWN. It is
    NOT a filter and the UI must never call it one: the excluded phrase is
    encoded like the positive one and SUBTRACTED, weighted, from the score, so
    an image that matches it falls in the ranking instead of disappearing. See
    PUSH_DOWN_WEIGHT_DEFAULT above for the calibration and for the measured case
    where it cannot work at all.

    Returns {'results': [{id, score}], 'image_ids', 'pool', 'filtered', 'unscored',
    'query', 'cached', 'score_range', 'pool_median', 'push_down', 'push_down_weight',
    'push_down_moved', 'push_down_median'}.
      * ``unscored`` — an image with no ✨ Score embedding CANNOT be found by text;
        saying "0 results" without saying that would let the user conclude the
        image is gone.
      * ``pool_median`` — the median cosine over the WHOLE candidate pool for this
        query. It is the empirical "what a typical image here scores" baseline,
        measured per bank and per query, and it is what lets the UI judge whether
        a ranking discriminates at all without hard-coding any constant. On a
        single-subject bank (image-to-image cosine 0.60–0.89) the discriminating
        gap compresses by 30–70%, and that is the app's MAIN use case, not an
        edge case — so the baseline has to be measured, never assumed.
      * ``push_down_median`` — {pool, results}: how strongly a TYPICAL image of
        this bank matches the excluded phrase, against how strongly the returned
        set does. Same reasoning as ``pool_median``, applied to the push-down:
        results well below pool means it worked here, results level with pool
        means it did not, and neither claim needs a universal constant. This is
        the only honest way to report an exclusion whose strength depends
        entirely on how entangled the two phrases are in this corpus.
      * ``push_down_moved`` — how many places in the returned ranking hold a
        different image than they would have without the exclusion. 0 is the
        signal that the push-down changed nothing at all, which the UI has to
        say out loud.

    Raises ValueError (→400) for an empty query or an unscored bank, and
    clip_text_encoder.TextEncodeError (→503) when no interpreter can run CLIP."""
    import numpy as np
    from . import clip_text_encoder
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    for value, label in ((query, 'query'), (push_down, 'push_down')):
        limit_error = clip_text_encoder.query_limit_error(value, label)
        if limit_error:
            raise ValueError(limit_error)
    # `-term` inside the query means the same thing as the panel's second field,
    # so both are folded into one excluded phrase before anything is encoded.
    positive, from_query = clip_text_encoder.split_query(query)
    text = clip_text_encoder.normalize_query(positive)
    excl = ', '.join(t for t in (clip_text_encoder.normalize_query(push_down),
                                 clip_text_encoder.normalize_query(from_query)) if t)
    # The inline ``-term`` shorthand and the dedicated field are individually
    # bounded above, but joining them can exceed that bound. Validate the exact
    # strings that will become cache keys so this stays a request error (400),
    # never a misleading encoder-unavailable response (503).
    for value, label in ((text, 'query'), (excl, 'combined push_down')):
        limit_error = clip_text_encoder.query_limit_error(value, label)
        if limit_error:
            raise ValueError(limit_error)
    if not text:
        # A bare "-hat" is not a search. Ranking by "least like a hat" would
        # return whatever is least like ANYTHING, which is noise wearing the
        # costume of an answer — so it is refused rather than served.
        raise ValueError('a search query is required — an excluded term alone '
                         'cannot rank anything')
    weight = _push_down_weight(push_down_weight if push_down_weight is not None
                               else PUSH_DOWN_WEIGHT_DEFAULT)
    engine = _selected_semantic_engine(bank)
    model_key = engine_model_key(engine)
    emb_by_path = _load_semantic_embeddings(bank)
    if not emb_by_path:
        if engine == 'clip':
            raise ValueError('run ✨ Score first — text search ranks the '
                             'embeddings it computes')
        raise ValueError('run Semantic index first — text search ranks the '
                         'SigLIP2 embeddings it computes')
    filters = filters or {}
    # How many rows the current filter holds AT ALL — the denominator that makes
    # "searched 120 of 400" (and therefore the unscored warning) truthful.
    filtered = _pool_query(bank_id, thresholds(), **filters).count()
    ids, E = _pool_embeddings(bank, emb_by_path, filters)
    # Encode AFTER the cheap refusals: never make someone wait on a CLIP load to
    # then be told their bank was never scored.
    qv, cached = (clip_text_encoder.encode_query(text) if engine == 'clip' else
                  clip_text_encoder.encode_query(text, engine=engine))
    nv = None
    if excl:
        # A second encode, and a second phrase in the same persistent cache — so
        # a repeated exclusion is as free as a repeated query. `cached` stays
        # true only when BOTH halves were already known, because it is shown to
        # promise instant, and half a cache hit is not instant.
        nv, ncached = (clip_text_encoder.encode_query(excl) if engine == 'clip'
                       else clip_text_encoder.encode_query(excl, engine=engine))
        cached = bool(cached) and bool(ncached)
    missing = max(0, int(filtered) - len(ids))
    base = {'query': text, 'cached': bool(cached), 'filtered': int(filtered),
            'pool': len(ids), 'unscored': missing, 'unindexed': missing,
            'engine': engine, 'model_key': model_key,
            'push_down': excl or None,
            'push_down_weight': round(weight, 3) if excl else None}
    if not ids:
        _verified_semantic_result_provenance(bank, engine, model_key)
        return {**base, 'results': [], 'image_ids': [], 'score_range': None,
                'pool_median': None, 'push_down_moved': None,
                'push_down_median': None}
    qv = np.asarray(qv, dtype='float32')
    qv /= (float(np.linalg.norm(qv)) + 1e-8)
    sims = E @ qv                                # cosine similarity, (m,)
    n = max(1, min(int(n), _CURATION_MAX_N))
    excl_sims = None
    scores = sims
    if nv is not None:
        nv = np.asarray(nv, dtype='float32')
        nv /= (float(np.linalg.norm(nv)) + 1e-8)
        excl_sims = E @ nv
        scores = sims - weight * excl_sims
    order = np.argsort(-scores, kind='stable')   # desc; stable ⇒ id tie-break
    keep = [int(k) for k in order[:n]]
    # The RANKING score is what ordered the list, so it is what the list reports:
    # showing the raw positive cosine here would make the order look arbitrary
    # (a lower-cosine image legitimately outranks a higher one once its excluded
    # match is paid for). Both halves are kept alongside so nothing is hidden.
    results = [{'id': ids[k], 'score': round(float(scores[k]), 4)} for k in keep]
    if excl_sims is not None:
        for r, k in zip(results, keep):
            r['match'] = round(float(sims[k]), 4)
            r['excluded_match'] = round(float(excl_sims[k]), 4)
    # The span of what came back, plus the pool's own median — together they let
    # the UI say whether this ranking discriminates, using only numbers measured
    # on THIS bank for THIS query. An absolute band would be wrong everywhere:
    # the same model's "good" ceiling barely moves between corpora while its
    # floor climbs sharply on real photographs of people.
    # The range is always in POSITIVE-match units, never in composite ones: it is
    # read against ``pool_median`` to decide whether the ranking discriminates,
    # and a composite score (which can even go negative) is not on that scale.
    # Without an exclusion this is exactly the old first/last pair; with one it
    # is the best and worst MATCH among what came back, which is the thing the
    # sentence above the grid actually claims.
    match_of = [float(sims[k]) for k in keep]
    score_range = ({'top': round(max(match_of), 4),
                    'bottom': round(min(match_of), 4)} if results else None)
    moved = excl_median = None
    if excl_sims is not None:
        # What the SAME query would have returned unexcluded — the only way to
        # tell the user whether the push-down did anything, rather than leaving
        # them to compare two screens from memory.
        #
        # POSITIONS changed, not membership. Membership alone is silently wrong
        # whenever n reaches the whole pool: asking for 60 out of 22 images
        # returns the same 22 however hard the push, so a membership count reads
        # 0 and the UI announces "changed nothing" over a grid the user can SEE
        # was reordered. Comparing the two orderings slot by slot is right in
        # both regimes — it counts newcomers when the list is a cut of a larger
        # pool, and re-ranking when it is not.
        plain = [int(k) for k in np.argsort(-sims, kind='stable')[:n]]
        moved = int(sum(1 for a, b in zip(keep, plain) if a != b))
        if len(keep) < len(ids):
            excl_median = {'pool': round(float(np.median(excl_sims)), 4),
                           'results': round(float(np.median(excl_sims[keep])), 4)}
        # else: the returned set IS the pool, so its median and the pool's are
        # the same number BY CONSTRUCTION. Reporting them would let the UI read
        # "level with a typical image — too tangled to separate" off an identity,
        # which is how it once declared a visibly-reordered grid a failure. When
        # a comparison cannot carry information, the honest output is none: the
        # places-changed count already says what happened.
    _verified_semantic_result_provenance(bank, engine, model_key)
    return {**base, 'results': results,
            'image_ids': [ids[k] for k in keep], 'score_range': score_range,
            'pool_median': round(float(np.median(sims)), 4),
            'push_down_moved': moved, 'push_down_median': excl_median}


def _trash_or_remove(path: str) -> str:
    """Get a rejected source file out of the user's folder, keeping it
    recoverable (OS trash → app trash → permanent unlink). The policy itself is
    app-wide and lives in ``services.trash``; this is the bank's entry point into
    it, kept as a name because the tests and the delete sweep both address it."""
    return trash.dispose(path, context='bank-rejected')


def _bank_folders(user_id, exclude_id=None) -> list:
    """[(bank, normalised realpath)] for the user's banks with a usable folder."""
    out = []
    for b in ImageBank.query.filter_by(user_id=user_id).all():
        if exclude_id is not None and b.id == exclude_id:
            continue
        if not b.source_path:
            continue
        try:
            out.append((b, os.path.normcase(os.path.realpath(b.source_path))))
        except (OSError, ValueError):
            continue   # an unresolvable source path cannot collide with anything
    return out


def overlapping_banks(user_id, bank_id) -> list:
    """The user's OTHER banks whose folder contains, or sits inside, this one's.

    Two banks over nested folders see the same files, and a bank never owns its
    source folder — so a delete run from one silently amputates the other. The
    UI has to be able to say so BEFORE the click. Returns
    [{'id', 'name', 'source_path', 'relation'}] with relation 'parent' (it
    contains us) or 'child' (it sits inside us)."""
    bank = get_bank(user_id, bank_id)
    if not bank or not bank.source_path:
        return []
    try:
        mine = os.path.normcase(os.path.realpath(bank.source_path))
    except (OSError, ValueError):
        return []
    out = []
    for other, theirs in _bank_folders(user_id, exclude_id=bank_id):
        if theirs == mine:
            relation = 'same'
        elif mine.startswith(theirs + os.sep):
            relation = 'parent'
        elif theirs.startswith(mine + os.sep):
            relation = 'child'
        else:
            continue
        out.append({'id': other.id, 'name': other.name,
                    'source_path': other.source_path, 'relation': relation})
    return sorted(out, key=lambda o: o['id'])


class BankSharesDataset(ValueError):
    """A destructive action was asked of a bank whose folder IS a dataset's.

    A subclass of ValueError so nothing that already catches ValueError starts
    500-ing, but a distinct type so the one route that answers 404 to "bank not
    found" can tell this apart and answer 400 with the explanation instead."""


def bank_dataset_conflict(user_id, bank_id) -> dict | None:
    """Does THIS bank already sit on a dataset's storage folder? None when it
    does not — which is every bank the guard in create_bank/_relocate_target has
    ever seen.

    The guard alone protects nobody who already has such a bank: it was created
    before the guard existed, it is in their database right now, and the click
    that hurts is 🗑 Delete rejected. So the conflict is DETECTED at open time
    (it rides in the workspace payload, which is also the 2 s poll — one
    realpath, no walk) and the destructive action refuses. Deliberately nothing
    else: the bank stays fully readable and fully triageable, and NOTHING is
    ever removed on the app's own initiative. Only the user can decide whether
    that bank should be relocated or dropped."""
    bank = get_bank(user_id, bank_id)
    if not bank or not bank.source_path:
        return None
    return path_guard.dataset_folder_conflict(bank.source_path)


def rejected_delete_preview(user_id, bank_id) -> dict | None:
    """What a 🗑 Delete rejected would actually destroy — the honest warning the
    confirmation needs. Counts the rejected files of this bank that ANOTHER bank
    also lists, per bank, by matching absolute paths against that bank's own
    inventory. None when the bank is gone.

    Returns {'rejected', 'mode', 'shared': [{'id','name','relation','files'}]}.
    ``mode`` is where the files would go, resolved the same way the deletion
    resolves it, so the dialog never promises the wrong thing."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    rows = BankImage.query.filter_by(bank_id=bank_id, status='reject').all()
    mine = {}
    for r in rows:
        p = abs_image_path(bank, r)
        if p:
            mine[os.path.normcase(p)] = True
    shared = []
    for other in overlapping_banks(user_id, bank_id):
        ob = db.session.get(ImageBank, other['id'])
        if ob is None:
            continue
        n = 0
        for (rel,) in db.session.query(BankImage.relpath).filter_by(bank_id=ob.id):
            try:
                full = os.path.normcase(os.path.realpath(
                    os.path.join(os.path.realpath(ob.source_path), rel)))
            except (OSError, ValueError):
                continue   # an unresolvable path cannot be the duplicate we are looking for
            if full in mine:
                n += 1
        if n:
            shared.append({'id': other['id'], 'name': other['name'],
                           'relation': other['relation'], 'files': n})
    return {'rejected': len(rows), 'mode': _delete_mode(), 'shared': shared,
            # The hard stop, next to the soft warnings: another BANK losing files
            # is a warning the user may accept, a DATASET losing them is not.
            'dataset_conflict': bank_dataset_conflict(user_id, bank_id)}


def _delete_mode() -> str:
    """Where a deleted source file WOULD go, without deleting anything, so the
    confirmation can say it. Same probe as ``services.trash.disposal_mode``."""
    return trash.disposal_mode()


def _delete_rejected_guard(user_id, bank_id, check_busy=True):
    """The three refusals 🗑 Delete rejected owes the caller BEFORE anything is
    touched — unknown bank, occupied bank, folder shared with a dataset. Split
    out so the route can answer 404/409/400 synchronously while the deletion
    itself runs in the background. ``check_busy`` is off for the job's own body:
    by then the registry holds OUR job and would refuse us."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if check_busy and bank_jobs.running(bank_id):
        raise RuntimeError('a job is running on this bank — stop it first')
    # An install that predates the create-time guard can still hold a bank whose
    # folder IS a dataset's. Here, and only here, that stops being cosmetic: these
    # "rejects" are the dataset's images. Refuse — never silently delete, never
    # silently clean up on the user's behalf.
    conflict = bank_dataset_conflict(user_id, bank_id)
    if conflict:
        raise BankSharesDataset(
            'This bank points at a dataset\'s own image folder, so deleting its '
            'rejected files would delete images out of the dataset. Nothing was '
            f'deleted. {conflict["message"]}')
    return bank


_DELETE_DETAIL = {
    'trash': 'moving files to the Recycle Bin',
    'app_trash': 'moving files to the app Trash',
    'delete': 'deleting files permanently',
}


def delete_rejected_summary(out: dict) -> str:
    """One plain sentence for the progress bar / completion toast. The workspace
    already toasts a finished job's ``detail``, so this IS the outcome the user
    reads — it must carry the numbers, not the word "done"."""
    gone = out['deleted'] + out['trashed'] + out['already_absent']
    where = ('permanently deleted' if out['mode'] == 'delete'
             else 'moved to the app Trash' if out['mode'] == 'app_trash'
             else 'moved to the Recycle Bin')
    text = f'{gone} rejected file(s) {where}'
    if out['skipped']:
        text += f' · {len(out["skipped"])} could not be removed and kept their row'
    return text


def start_delete_rejected(app, user_id, bank_id) -> dict:
    """Start 🗑 Delete rejected as a background bank job and return immediately.

    THE WAIT THIS REPLACES: the deletion ran inside the POST. On a bank with
    thousands of rejects — every file individually handed to the OS Recycle Bin,
    which is not fast — the dialog sat on "Deleting…" for minutes with no count,
    no way to stop, and no way to tell a slow run from a crashed one. It is now
    an ordinary bank job: the same one-per-bank registry, the same progress bar,
    the same Stop button, the same completion toast as every other pass.

    Cancelling is safe by construction — each file is removed then its row
    dropped, so a stopped run leaves a consistent bank and simply has rejects
    left over. Returns {'total', 'job'}; ``job['result']`` holds the full
    outcome once it has finished (immediately so under TESTING, where bank_jobs
    runs inline)."""
    _delete_rejected_guard(user_id, bank_id)
    total = BankImage.query.filter_by(bank_id=bank_id, status='reject').count()

    def _run(job):
        out = delete_rejected(
            user_id, bank_id, job=job,
            _bank_lease=_job_bank_capability(job))
        job['result'] = out
        bank_jobs.progress(job, detail=delete_rejected_summary(out))

    job = bank_jobs.start(app, bank_id, 'delete_rejected', _run, total=total)
    return {'total': total, 'job': job}


@_serialized_bank_mutation('delete_rejected')
def delete_rejected(user_id, bank_id, job=None, *, _bank_lease=None) -> dict:
    """Delete the SOURCE files of every status='reject' image from disk, then
    drop their bank_image rows.

    This is the ONLY bank action that writes to the user's source folder. Where
    the files land is _trash_or_remove's decision (OS trash, else the app's own
    trash, else a permanent unlink) and rides back in 'mode' — the confirmation
    dialog says it BEFORE the click, via rejected_delete_preview().

    ⚠️ A bank does not own its folder. When another bank sits over the same tree,
    these files are ITS files too and it will find them gone; the preview names
    those banks so the warning can.

    Non-rejected images are never touched. Per-file failures (permission, a path
    that escapes the bank folder) are collected and reported; they never abort
    the batch. A row is dropped only when its file is gone afterwards (deleted,
    trashed, or already absent) — a file we failed to remove keeps its row so the
    user can see and retry it. Returns
    {'mode', 'deleted', 'trashed', 'already_absent', 'rows_removed', 'skipped',
    'cancelled'} where 'trashed' counts everything that stayed recoverable.

    ``job`` is the bank_jobs snapshot when this runs as a background pass (the
    normal path — see start_delete_rejected): progress is reported per file and
    Stop is honoured between files.
    """
    bank = _delete_rejected_guard(user_id, bank_id, check_busy=job is None)

    # (id, relpath) tuples, not ORM rows, and ONE realpath for the whole batch:
    # the loop commits as it goes, which expires live ORM objects and would make
    # every later row pay a re-SELECT (and abs_image_path a realpath) per file.
    rows = (db.session.query(BankImage.id, BankImage.relpath)
            .filter_by(bank_id=bank_id, status='reject').all())
    root = os.path.realpath(bank.source_path)
    out = {'mode': 'trash', 'deleted': 0, 'trashed': 0, 'already_absent': 0,
           'rows_removed': 0, 'skipped': [], 'cancelled': False}
    if job is not None:
        # The bar already prints "done / total"; a detail repeating it would just
        # say the same numbers twice. Spend the width on where the files GO.
        bank_jobs.progress(job, done=0, total=len(rows),
                           detail=_DELETE_DETAIL.get(_delete_mode(),
                                                     'removing files'))
    pending_ids, removed = [], 0
    modes_used = set()

    def _drop(ids):
        """Commit one chunk of row removals. Done as we go, not at the end: a
        Stop (or a crash) must not leave files gone from disk with their rows
        still claiming they are there."""
        for i0 in range(0, len(ids), _SQL_IN_CHUNK):
            BankImage.query.filter(
                BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])
            ).delete(synchronize_session=False)
        db.session.commit()

    for i, (row_id, relpath) in enumerate(rows, 1):
        if job is not None and bank_jobs.cancelled(job):
            out['cancelled'] = True
            break
        path = _abs_under(root, relpath)
        if path is None:
            # relpath escapes the bank folder — refuse to touch it, keep the row.
            out['skipped'].append({'relpath': relpath, 'reason': 'unsafe_path'})
        elif not os.path.exists(path):
            out['already_absent'] += 1
            pending_ids.append(row_id)
        else:
            try:
                mode = _trash_or_remove(path)
            except OSError as e:
                out['skipped'].append({'relpath': relpath, 'reason': str(e)})
            else:
                modes_used.add(mode)
                if mode == 'delete':
                    out['deleted'] += 1
                else:
                    out['trashed'] += 1   # OS trash or app trash — recoverable
                pending_ids.append(row_id)
        if job is not None:
            bank_jobs.progress(job, done=i)
        if len(pending_ids) >= _SQL_IN_CHUNK:
            _drop(pending_ids)
            removed += len(pending_ids)
            pending_ids = []

    _drop(pending_ids)
    out['rows_removed'] = removed + len(pending_ids)
    # The pending ↩ offer points at rows this run just dropped — restoring them
    # would find nothing. Withdraw it rather than advertise a restore we cannot
    # perform (the files themselves went to a trash only the user can reach).
    bank_undo.clear(bank_id)
    # Report the WORST outcome that happened: one permanently removed file makes
    # the run 'delete', whatever the rest did. The UI wording follows this.
    for mode in ('delete', 'app_trash', 'trash'):
        if mode in modes_used:
            out['mode'] = mode
            break
    return out


# --- cooperative-cancel subprocess driver (shared by the face + score passes) --
# A bank inference pass runs for minutes over thousands of images. Its embeddings
# are cached incrementally, but a brutal proc.kill on Stop still throws away the
# in-flight slice AND leaves the UI mute. Instead we ask the child to stop CLEANLY:
# drop a sentinel file it polls between images so it flushes its cache and reports
# how much it kept; a watchdog timer hard-kills it only if it doesn't stop within
# the grace period. The child also writes a plain-text ``<cache>.count`` sidecar we
# read back here (the Flask venv has no numpy to open the .npz), so a stopped pass
# always shows an honest count — even in the rare hard-kill case.
_INFER_CANCEL_GRACE = 15.0   # seconds a cleanly-cancelled child gets before a kill
_CACHED_RE = re.compile(r'(\d+) image\(s\), (\d+) cached')
# A step the child cannot count per image (writing a 70 MB cache, the n² style
# grouping). The sentence is written child-side in user-facing English and shown
# VERBATIM — a phase nobody translates cannot drift away from what runs. Any
# infer child may emit these; one that emits none behaves exactly as before.
_PHASE_RE = re.compile(r'\[phase\] (.+)$')


def _safe_kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001 — already gone is fine
        pass


def _read_cache_count(cache_path):
    """The count the child last flushed to ``<cache>.count``, or None."""
    try:
        with open(str(cache_path) + '.count', encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _stopped_detail(noun, data, cache_path, total, suffix='', *,
                    final='relaunch to finish and cluster'):
    """The honest end-of-pass line when the user Stopped it. Prefers the child's
    own cancel counts, falls back to the flushed sidecar count, and never invents
    a number it can't back up. ``suffix`` states what reached the DATABASE, which
    is a different claim from what reached the cache and must not be implied."""
    n = data.get('ready') if 'ready' in data else data.get('cached')
    if n is None:
        n = _read_cache_count(cache_path)
    if n is None:
        return ('Stopped — progress saved to cache'
                f'{suffix}; {final}')
    n = int(n)
    remaining = data.get('remaining')
    remaining = int(remaining) if remaining is not None else max(0, int(total) - n)
    return (f'Stopped — {n} {noun} ({remaining} remaining)'
            f'{suffix}; {final}')


# --- rows that can vanish under a long pass ---------------------------------
# A pass reads its rows, then spends minutes to hours walking them, committing as
# it goes. `expire_on_commit` is on, so every row it has not reached yet becomes
# a lazy re-SELECT — and that re-SELECT can come back empty, because a bank's
# rows CAN disappear mid-pass: `delete_bank` cancels the live job cooperatively
# and then drops every row and the bank itself immediately, so the still-running
# thread keeps iterating over rows that are already gone (`delete_rejected` can
# do the same if the job registry has aged the pass out as stale).
#
# What SQLAlchemy does then is the trap: plain attribute access on an expired
# row whose database row is gone raises ObjectDeletedError, and so does a commit
# carrying a write staged on one. Either killed the WHOLE pass — one deleted
# image and thousands of analysed ones never got written. `Session.get(...,
# populate_existing=True)` is the one access that answers None instead of
# raising, so every long pass re-reads through the helper below immediately
# before it touches a row, and skips (and counts) what is no longer there.
def _live_image(image_id):
    """The bank row as the database has it RIGHT NOW, or None when it is gone.

    Always re-reads (``populate_existing``): a row the session still holds
    unexpired would otherwise be returned from the identity map, and the whole
    point here is to ask the database whether the image still exists."""
    if image_id is None:
        return None
    return db.session.get(BankImage, image_id, populate_existing=True)


def _detach_bank(bank):
    """Take the ImageBank row OUT of the session so a pass can keep reading it.

    Passes read the bank (its source folder) once per image, for hours. Left in
    the session, it is expired by each commit exactly like the images are — and
    it is deleted by the very same `delete_bank` that removes them, so it turns
    into an ObjectDeletedError of its own. Detaching a fully loaded instance
    keeps every column readable and immune to expiry; the passes doing this only
    ever READ the bank, so nothing is lost by leaving the session's copy behind.
    """
    if bank is not None:
        db.session.expunge(bank)
    return bank


def _release_db_before_inference():
    """End the session's transaction before an inference subprocess we may sit
    in for an HOUR (bank scoring on CPU is measured near that on a big bank).

    The pass reads its rows, hands a path list to a child process, waits, then
    writes the results back. Reading first is fine; keeping the SAME session
    transaction open across the wait is not. WAL (app/__init__.py) buys concurrent
    READERS, never concurrent writers: one stray write joining the transaction
    ahead of the child — a status stamp, a counter, a `flush()` inherited from the
    caller — takes the single write lock and holds it for the whole inference, so
    every other writer in the app dies on `database is locked` after the 5 s
    busy_timeout. That is the exact failure that abandoned two paid cloud runs on
    2026-07-26 (see cloud_training._COMMIT_RETRIES), for a five-second holder;
    this one would hold it for an hour.

    Committing here costs nothing on the nominal path (nothing is pending) and
    removes the trap for good. Callers must not hold ORM objects across it:
    `expire_on_commit` is on, so rows are re-fetched after the child returns."""
    db.session.commit()


def _infer_subprocess_argv(python, script) -> list:
    """Use the borrowed interpreter without unrelated user-site packages.

    Every ``backend/infer/*`` script, not only Score's: they share the borrowed
    interpreter and therefore share its user site-packages, and the readiness
    probe that vouches for each of them runs isolated (``services.infer_env``).
    Launching any of them differently is how a green capability card turns into
    a per-item failure once the real pass starts."""
    return infer_env.worker_argv(python, script)


def _drive_infer_subprocess(job, python, script, payload, cache_path,
                            progress_re, window):
    """Run an infer subprocess, streaming its stderr progress into ``job`` and
    honouring Stop cooperatively. Returns (data, stderr_tail, returncode) where
    ``data`` is the child's last JSON line (``cancelled: true`` when it stopped
    cleanly). On the first "N cached" line it sets a "resuming" hint, so relaunching
    over a partly-cached bank doesn't look like a full recompute."""
    import json
    import threading
    cancel_file = str(cache_path) + '.cancel'
    try:
        os.remove(cancel_file)   # never inherit a stale sentinel from a past run
    except OSError:
        pass
    hint = {'shown': False}
    with window:
        # Borrowed ML interpreters (notably ComfyUI portable) must not inherit
        # unrelated per-user site-packages. The readiness probe uses the same
        # ``-s`` contract; launching Score differently would turn a green GPU
        # choice into an open_clip crash once the real pass starts.
        proc = subprocess.Popen(
            _infer_subprocess_argv(python, script), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            env=infer_env.worker_env(python),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        killer = {'timer': None}

        def _cancel():
            # Ask for a clean stop (the child flushes + exits), and arm a watchdog
            # that hard-kills ONLY if it doesn't stop within the grace period.
            try:
                with open(cancel_file, 'w', encoding='utf-8') as f:
                    f.write('1')
            except OSError:
                pass   # if the sentinel cannot be written the grace-kill timer still fires
            t = threading.Timer(_INFER_CANCEL_GRACE, _safe_kill, args=(proc,))
            t.daemon = True
            t.start()
            killer['timer'] = t

        bank_jobs.set_cancel_hook(job, _cancel)
        stderr_tail = deque(maxlen=5)

        def _drain_stderr():
            for line in proc.stderr:
                line = line.strip()
                if line:
                    stderr_tail.append(line)
                m = progress_re.search(line)
                if m:
                    bank_jobs.progress(job, done=int(m.group(1)), total=int(m.group(2)))
                mp = _PHASE_RE.search(line)
                if mp:
                    # A step with no per-image counter. The count is CLEARED with
                    # the sentence, on purpose: leaving "373 / 373" up next to
                    # "grouping styles…" is what made a working pass look frozen
                    # behind a full bar. done=total=0 renders as no figure and no
                    # bar (see ProgressBar), which is the honest shape for a step
                    # nobody can count.
                    bank_jobs.progress(job, done=0, total=0, detail=mp.group(1))
                if not hint['shown']:
                    mc = _CACHED_RE.search(line)
                    if mc:
                        hint['shown'] = True
                        total, cached = int(mc.group(1)), int(mc.group(2))
                        if 0 < cached < total:
                            bank_jobs.progress(
                                job,
                                detail=f'resuming — {cached} of {total} already cached')

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except OSError:
            pass  # process died early — surfaced through the exit path below
        stdout = proc.stdout.read()
        proc.wait()
        t.join(timeout=5)
        if killer['timer']:
            killer['timer'].cancel()
    try:
        os.remove(cancel_file)
    except OSError:
        pass   # the sentinel may be gone already: the pass is over either way
    line = next((ln for ln in reversed(stdout.splitlines())
                 if ln.strip().startswith('{')), '')
    try:
        data = json.loads(line) if line else {}
    except json.JSONDecodeError:
        data = {}
    return data, stderr_tail, proc.returncode


# --- subject (face) pass ----------------------------------------------------
_EMBED_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'face_embed_infer.py')
_PROGRESS_RE = re.compile(r'\[embed\] (\d+)/(\d+)')


def _resolve_face_device():
    """(device, use_gpu) for the face pass — see capabilities.resolve_face_device.

    Kept as a thin local name because this module's tests patch it, but the RULE
    itself now lives in one place, shared with the dataset scorer: both surfaces
    run the same face models and must reach the same verdict about the GPU."""
    from .. import capabilities
    return capabilities.resolve_face_device()


def _angle_todo_clause():
    """Face-scanned rows with no measured yaw — the ⤢ backfill's whole pool."""
    return and_(BankImage.face_state.isnot(None), BankImage.face_yaw.is_(None))


def _angle_pool(bank_id, statuses=None, ids=None):
    """The ⤢ angle backfill's rows.

    ⚠️ ITS HISTORICAL DEFAULT IS NOT THE OTHER PASSES' DEFAULT. This lane never
    filtered on status at all — rejected rows were measured too, and nobody
    decided that: the `if angles_only` branch simply carried no status clause
    while the `else` branch did. Passing statuses=None keeps that behaviour
    exactly (the whole bank, bin included), because changing it silently would
    move work the user can see counted in `angle_backfillable`. A scope given
    explicitly narrows it — which is the only thing this lane gains, since it
    writes nothing but `face_yaw` and leaves every cluster alone, so a partial
    run cannot renumber anything."""
    q = BankImage.query.filter_by(bank_id=bank_id).filter(_angle_todo_clause())
    if statuses is not None:
        q = q.filter(BankImage.status.in_(statuses))
    if ids:
        q = q.filter(BankImage.id.in_([int(i) for i in ids][:_SQL_IN_CHUNK]))
    return q


def start_faces(app, user_id, bank_id, angles_only=False, statuses=None, ids=None):
    """Launch the face embedding + person clustering pass over the bank's
    non-rejected images. Needs the face-scoring extra (Setup ▸ Quality tools).

    ``angles_only`` is the ⤢ BACKFILL, and it is a different job on purpose. The
    face pass has always computed a head yaw and thrown it away (it used it once
    to decide 'extreme_pose'), so every bank scanned before this release has
    faces with no angle and its cache cannot answer for them — the number is not
    in the .npz to be read back. Re-measuring them means re-running the detector,
    which on a big bank is hours: far too much to slip into a pass somebody
    started for something else. So it is its OWN action, offered on the ⤢ row
    with its own count and its own estimate, never automatic and never at boot.
    It touches ONLY `face_yaw`: person clusters are computed over the whole bank
    at once, and re-deriving them from a partial re-run would renumber a
    clustering the user has already worked with."""
    from .face_similarity import is_available
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    # The identity lane takes NO scope, and refusing it is the design (see the
    # docstring): person clusters are one numbering of the whole bank.
    want = normalize_pass_statuses(statuses) if angles_only else None
    scope_ids = ids if angles_only else None
    if not angles_only and (statuses or ids):
        raise ValueError(
            '👥 Group by person always covers the whole bank: the person clusters '
            'are one numbering computed from every face at once, so a partial run '
            'would renumber groups you have already worked with')
    q = BankImage.query.filter_by(bank_id=bank_id)
    if angles_only:
        q = _angle_pool(bank_id, want, scope_ids)
        # BEFORE the install probe, and the order is the point: sending someone
        # to install a 300 MB extra so they can do work that does not exist is a
        # true sentence that answers the wrong question. (Same reasoning as the
        # framing pass, which checks occupancy before it probes Ollama.)
        if not q.count():
            raise ValueError('every face-scanned image already has an angle')
    else:
        q = q.filter(BankImage.status != 'reject')
    if not is_available():
        raise RuntimeError(
            'face scoring is not installed (Quality tools step in Setup)')
    # Images the user has DECLARED to be one person (their subfolder is asserted)
    # are not embedded at all — that skip is the whole point of the assertion, and
    # counting them in the total would promise work this pass will not do. The ⤢
    # angle lane is NOT affected: measuring where a head points is not identifying
    # who it is, and its pool is already restricted to rows a face pass measured.
    total = q.count() if angles_only \
        else max(q.count() - _asserted_image_count(bank_id), 0)
    return bank_jobs.start(app, bank_id, 'angles' if angles_only else 'faces',
                           _faces_job(bank_id, angles_only, want, scope_ids),
                           total=total)


def _asserted_image_count(bank_id) -> int:
    return (BankImage.query.filter_by(bank_id=bank_id)
            .filter(BankImage.status != 'reject',
                    BankImage.face_cluster_origin == 'asserted').count())


def _faces_job(bank_id, angles_only=False, statuses=None, ids=None):
    def run(job):
        import json as _json
        import sys
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        if angles_only:
            q = _angle_pool(bank_id, statuses, ids)
        else:
            q = BankImage.query.filter_by(bank_id=bank_id)
            # Asserted rows are EXCLUDED from the identity lane, not merely
            # preserved: the user already told us who is in them, so paying an
            # embedding to re-derive it is exactly the cost "Single person here"
            # exists to avoid (services/folder_person.py).
            q = q.filter(BankImage.status != 'reject',
                         or_(BankImage.face_cluster_origin.is_(None),
                             BankImage.face_cluster_origin != 'asserted'))
        def asserted_membership():
            if angles_only:
                return ()
            return tuple((int(image_id), cluster_id)
                         for image_id, cluster_id in (
                db.session.query(BankImage.id, BankImage.face_cluster)
                .filter(BankImage.bank_id == bank_id,
                        BankImage.face_cluster_origin == 'asserted')
                .order_by(BankImage.id.asc()).all()))

        asserted_generation = asserted_membership()
        rows = q.order_by(BankImage.id.asc()).all()
        eligible_ids = [r.id for r in rows]
        by_path = {}
        unresolved_ids = []
        for r in rows:
            p = analysis_image_path(bank, r, refresh_rotation=True)
            if (_is_safe_bank_source(p, label='bank face pass')
                    and p not in by_path):
                by_path[p] = r.id
            else:
                unresolved_ids.append(r.id)
                # A strict resolver failure means the row cannot continue to
                # advertise measurements for an older effective generation.
                _invalidate_effective_analysis(r)
        paths = list(by_path)
        skipped_asserted = 0 if angles_only else _asserted_image_count(bank_id)
        bank_jobs.progress(job, done=0, total=len(paths),
                           detail='measuring angles' if angles_only else 'face pass')
        if not paths:
            if not angles_only and eligible_ids:
                for i0 in range(0, len(eligible_ids), _SQL_IN_CHUNK):
                    BankImage.query.filter(
                        BankImage.id.in_(eligible_ids[i0:i0 + _SQL_IN_CHUNK]),
                        or_(BankImage.face_cluster_origin.is_(None),
                            BankImage.face_cluster_origin != 'asserted')).update(
                        {'face_cluster': None}, synchronize_session=False)
            db.session.commit()
            return
        _bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        th = thresholds()
        # Device: 'cpu' (default, never touches the GPU/ComfyUI) or 'cuda'.
        # 'auto' = GPU when the face interpreter actually exposes CUDA
        # (onnxruntime-gpu installed), else CPU. The GPU path is used ONLY when
        # CUDA is truly available AND must run inside the GPU-exclusive window so
        # it never competes with a training / scoring pass; a CPU pass stays out
        # of the window (it can run alongside GPU work).
        from ..gpu_window import gpu_exclusive_vision_window
        from contextlib import nullcontext
        device, use_gpu = _resolve_face_device()
        cache_path = _face_cache_path(bank_id)
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('face_scoring.models_root') or None,
            'cache': str(cache_path),
            'cancel_file': str(cache_path) + '.cancel',
            'threshold': th['face_threshold'],
            'device': device,
            # Only the ⤢ backfill overrides the cache. A normal pass must stay
            # exactly as cheap as it was: it writes the yaw for every image it
            # actually looks at, and leaves the already-cached ones alone rather
            # than silently turning a resume into hours of re-detection.
            'require_yaw': bool(angles_only),
        })
        python = cfg.get('face_scoring.python') or sys.executable
        window = gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu else nullcontext()
        _release_db_before_inference()
        data, stderr_tail, returncode = _drive_infer_subprocess(
            job, python, _EMBED_SCRIPT, payload, cache_path, _PROGRESS_RE, window)
        # Stopped by the user — say exactly what's kept, never a mute ✗ (the cached
        # embeddings are safe; relaunching skips them and only finishes the rest).
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            bank_jobs.progress(job, detail=_stopped_detail(
                'face embeddings cached', data, cache_path, len(paths)))
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'face pass produced no output '
                                       f'(rc={returncode})')
        results = data.get('results') or {}
        clusters = data.get('clusters') or {}
        # The child numbers its clusters 1..n, unaware of the asserted groups that
        # already own ids in this bank. Push them above the highest asserted id so
        # a computed cluster can never land ON a folder the user declared — the two
        # kinds share one id space (that is what lets them be merged later), and
        # sharing it means allocating in it.
        from . import folder_person
        offset = folder_person.asserted_offset(bank_id)
        done = vanished = stale = 0
        cluster_valid = not unresolved_ids
        valid_rows = {}
        for p, image_id in by_path.items():
            row = _live_image(image_id)
            if row is None:      # deleted while the pass ran — see _live_image
                vanished += 1
                continue
            if not angles_only and row.face_cluster_origin == 'asserted':
                # The user may have asserted this folder while inference was in
                # flight.  Their newer decision wins and this row can no longer
                # be a member of the computed partition.
                cluster_valid = False
                continue
            res = results.get(p) or {}
            if not _prepare_analysis_write(row, p, res.get('fingerprint')):
                # Counted, not only used to void the partition: in ⤢ angles-only
                # mode there IS no partition, so this row would otherwise leave
                # no trace anywhere.
                stale += 1
                cluster_valid = False
                continue
            # A yaw is written whenever the child measured one. It is never
            # written back as NULL over a value we already have: the ⤢ backfill
            # re-runs on rows that HAVE no angle, and a face that fails detection
            # this time must leave the row "not measured", not blank a number
            # that was right.
            yaw = res.get('yaw')
            if yaw is not None:
                row.face_yaw = float(yaw)
            if not angles_only:
                row.face_state = res.get('state')
                row.face_det = res.get('det')
                valid_rows[p] = row
            done += 1
            if done % 200 == 0:
                db.session.commit()
        if not angles_only:
            if asserted_membership() != asserted_generation:
                cluster_valid = False
            if cluster_valid and len(valid_rows) == len(by_path):
                # Revalidate the complete partition after all scalar write-back.
                # Otherwise an early member can change while later rows are
                # being committed and still receive a cluster for its old bytes.
                for p, image_id in by_path.items():
                    row = _live_image(image_id)
                    fingerprint = (results.get(p) or {}).get('fingerprint')
                    live = bank_transfer_metadata.content_fingerprint_path(p)
                    if (row is None or row.face_cluster_origin == 'asserted'
                            or row.analysis_fingerprint != fingerprint
                            or live != fingerprint):
                        cluster_valid = False
                        if (row is not None
                                and row.analysis_fingerprint == fingerprint
                                and live != fingerprint):
                            _invalidate_effective_analysis(row)
                            row.analysis_fingerprint = live
                        break
            if cluster_valid and len(valid_rows) == len(by_path):
                for p, row in valid_rows.items():
                    cid = clusters.get(p)
                    row.face_cluster = None if cid is None else int(cid) + offset
                    row.face_cluster_origin = None
            else:
                # Person ids are one partition. If any member changed after
                # inference, a mixture of old/new numbering is not meaningful.
                (BankImage.query.filter(
                    BankImage.bank_id == bank_id,
                    BankImage.status != 'reject',
                    or_(BankImage.face_cluster_origin.is_(None),
                        BankImage.face_cluster_origin != 'asserted')).update(
                    {'face_cluster': None}, synchronize_session=False))
        db.session.commit()
        if vanished:
            logger.info('bank face pass: %s image(s) were deleted while it ran', vanished)
        sizes = {}
        for cid in clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        # Same sentence-with-its-shape as the style grouping, for the same
        # reason: "1 person cluster of 2+" over a bank where ONE cluster holds
        # everybody reads as "nobody grouped", the exact opposite of the truth.
        detail = (f'done — {sum(1 for r in results.values() if r.get("yaw") is not None)}'
                  ' angle(s) measured' if angles_only
                  else 'done — ' + group_summary(
                      sizes.values(), 'person cluster', th['face_threshold'],
                      '🎚 Filter thresholds ▸ face_threshold'))
        if not angles_only and not cluster_valid:
            detail += ' (person grouping discarded: an image changed during write-back)'
        if skipped_asserted:
            # Never mute: an image the pass did not look at must be reported as
            # such, or "0 clusters" would read as "no one in this bank".
            detail += (f', {skipped_asserted} image(s) skipped '
                       f'(subfolder asserted as one person)')
        # The embeddings are cached NOW, so asking "which folders look like one
        # person?" costs nothing here and would cost a pass of its own later.
        # It only ever produces a suggestion the user confirms (folder_person).
        if not angles_only and not bank_jobs.cancelled(job):
            detail += folder_person.probe_after_faces(job, bank_id)
        detail += _skipped_note(vanished=vanished, stale=stale)
        bank_jobs.progress(job, detail=detail)
    return run


# --- scoring pass (aesthetic · NSFW · style) --------------------------------
_SCORE_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'bank_score_infer.py')
_SCORE_PROGRESS_RE = re.compile(r'\[score\] (\d+)/(\d+)')


def _gpu_busy_reason() -> str | None:
    """A human reason the GPU is unavailable right now, or None. Same system
    flags training and the vision window use, so a bank GPU pass never races a
    training run or a captioning pass (the 'never concurrent with a training'
    guarantee). Checked up-front for an immediate 503 rather than a doomed 202."""
    from ..job_queue import queue_manager
    if queue_manager._get_system_state('training_in_progress'):
        return 'training is running on the GPU — try again once it finishes'
    if queue_manager._get_system_state('vision_in_progress'):
        return 'a vision/GPU pass is already running — try again in a moment'
    return None


def _resolve_score_device() -> tuple:
    """(device, use_gpu) for the scoring pass — what the CHILD will actually do.

    bank_score_infer.py picks `cuda if torch.cuda.is_available() else cpu` on its
    own, so the parent asks the same interpreter the same question. It matters
    beyond a label: use_gpu decides whether we take the GPU-exclusive window,
    which unloads ComfyUI and blocks any training start for the whole pass. The
    extra ships CPU-only torch, so the honest answer is usually 'cpu' — and an
    hour of CPU work must not hold a GPU it never touches."""
    from ..capabilities import bank_scoring_gpu_available
    use_gpu = bank_scoring_gpu_available()
    return ('cuda' if use_gpu else 'cpu'), use_gpu


_SEMANTIC_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'bank_semantic_infer.py')
_SEMANTIC_PROGRESS_RE = re.compile(r'\[semantic\] (\d+)/(\d+)')


def start_semantic_index(app, user_id, bank_id, rescan=False):
    """Build/resume the selected image space; CLIP delegates to Score exactly."""
    from ..capabilities import probe_bank_siglip2
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    engine = _selected_semantic_engine(bank)
    if engine == 'clip':
        return start_score(app, user_id, bank_id, rescore=bool(rescan))
    capability = probe_bank_siglip2()
    if not capability.get('ok'):
        raise RuntimeError(capability.get('detail') or (
            'SigLIP2 is not installed (Quality tools step in Setup)'))
    _device, use_gpu = _resolve_semantic_device()
    reason = _gpu_busy_reason() if use_gpu else None
    if reason:
        raise RuntimeError(reason)
    # SigLIP2 is a semantic Bank index, not the aesthetic Score pool: the bin is
    # searchable and participates in near-duplicate grouping too.
    total = BankImage.query.filter_by(bank_id=bank_id).count()
    return bank_jobs.start(
        app, bank_id, 'semantic_index',
        _semantic_index_job(bank_id, bool(rescan)), total=total)


def _semantic_index_job(bank_id, rescan=False):
    def run(job):
        import json as _json
        from contextlib import nullcontext

        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        paths = []
        for row in rows:
            path = analysis_image_path(bank, row, refresh_rotation=True)
            if _is_safe_bank_source(path, label='bank semantic index'):
                paths.append(path)
        device, use_gpu = _resolve_semantic_device()
        bank_jobs.progress(job, done=0, total=len(paths), detail=(
            f'SigLIP2 semantic index ({device.upper()})'))
        if not paths:
            bank_jobs.progress(job, detail=(
                'done — nothing to index: this Bank has no readable image'))
            return
        _bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        cache_path = _semantic_cache_path(bank_id)
        payload = _json.dumps(bank_semantic_engine.image_worker_payload(
            paths, bank_id, engine='siglip2', device=device, rescan=bool(rescan)))
        from . import bank_semantic_models
        python = bank_semantic_models.semantic_python()
        window = (gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu
                  else nullcontext())
        _release_db_before_inference()
        data, stderr_tail, returncode = _drive_infer_subprocess(
            job, python, _SEMANTIC_SCRIPT, payload, cache_path,
            _SEMANTIC_PROGRESS_RE, window)
        bank_semantic_engine.reset_memo()
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            bank_jobs.progress(job, detail=_stopped_detail(
                'images indexed', data, cache_path, len(paths),
                suffix='; progress is resumable from this cache',
                final='relaunch to finish'))
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or (
                f'SigLIP2 semantic index produced no output (rc={returncode})'))
        ready = int(data.get('ready') or 0)
        failed = int(data.get('failed') or 0)
        computed = int(data.get('computed') or 0)
        reused = int(data.get('reused') or 0)
        unreadable = max(0, len(rows) - len(paths))
        detail = (f'done — {ready} semantic embedding(s) ready '
                  f'({computed} computed, {reused} reused)')
        if failed:
            detail += f', {failed} failed (will retry)'
        if unreadable:
            detail += f', {unreadable} unreadable image(s) still need indexing'
        bank_jobs.progress(job, done=ready, total=len(rows), detail=detail)
    return run


# Public aliases kept explicit for tests/callers that construct the job directly.
semantic_index_job = _semantic_index_job
start_semantic = start_semantic_index


# CLIP ViT-L/14 measured at ~336 ms/image on CPU against ~15 ms on a recent
# card. Used only to warn, never to refuse — a slow pass is still a pass.
SCORE_CPU_MS_PER_IMAGE = 336


def score_device_info(bank_id=None) -> dict:
    """What ✨ Score will run on, and — when that is the CPU — how long the bank
    would take and whether this machine even has a card to switch to.

    The pass is not slow by accident: the scoring extra installs CPU-only torch
    on purpose (Setup builds it a small venv rather than pushing a ~2.5 GB CUDA
    download on people with no GPU). That is a defensible default, but it has to
    be VISIBLE — an unexplained hour looks like a hang, and the user cannot fix
    what nobody told them about."""
    from ..capabilities import gpu_vram_gb
    device, use_gpu = _resolve_score_device()
    out = {'device': device, 'gpu': use_gpu, 'gpu_present': False,
           'eta_minutes': None}
    if use_gpu:
        return out
    out['gpu_present'] = gpu_vram_gb() is not None
    if bank_id is not None:
        pending = (BankImage.query.filter_by(bank_id=bank_id)
                   .filter(BankImage.status != 'reject')
                   .filter(BankImage.aesthetic_score.is_(None)).count())
        # Rounded UP to a whole minute while there is anything left: "0 minutes"
        # for work that is about to start reads as "instant" and is a lie.
        out['eta_minutes'] = (max(1, round(pending * SCORE_CPU_MS_PER_IMAGE / 60000))
                              if pending else None)
    return out


def start_score(app, user_id, bank_id, rescore=False):
    """Launch the scoring pass (LAION aesthetic + NSFW + style clustering) over
    the bank's non-rejected images. Needs the bank-scoring extra (Setup ▸ Quality
    tools). Serialized against training/vision ONLY when it will really run on
    the GPU: refusing a CPU pass because 'the GPU is busy' would block an hour of
    work that never wanted the card.

    The pool is ALWAYS the whole bank, and deliberately so — "already scored" is
    not a reason to leave an image out. The style ids are a partition computed
    from every embedding at once, so a pass handed only the unscored rows would
    number a sub-population from 1 and land those ids on top of unrelated groups
    already in the database; the semantic-dedup pass, which blocks by
    style_cluster, would then stop comparing across the seam and miss crops
    without saying anything. Skipping work is the CHILD's job and it already does
    it: a cached, unchanged image is never embedded again, and a fully cached
    bank does not even load CLIP.

    ``rescore=True`` is the explicit "throw the cache away and recompute" lane
    (same shape as the quality pass's ``rescan``) — for a new model, or scores
    you no longer trust. The plain ✨ Score button keeps meaning exactly what it
    always meant: a complete pass that resumes."""
    from ..capabilities import probe_bank_scoring
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not probe_bank_scoring().get('ok'):
        raise RuntimeError('bank scoring is not installed '
                           '(Quality tools step in Setup)')
    _device, use_gpu = _resolve_score_device()
    reason = _gpu_busy_reason() if use_gpu else None
    if reason:
        raise RuntimeError(reason)
    total = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.status != 'reject').count())
    return bank_jobs.start(app, bank_id, 'score', _score_job(bank_id, rescore),
                           total=total)


# Rows written between two commits in the score write-back. Small enough that a
# Stop (or a crash) never costs more than a few seconds of re-run, large enough
# that we are not committing per row — same trade-off as the duplicate regrouping
# budget above, and for the same reason: long single transactions hold SQLite's
# one write lock and everything else in the app dies on "database is locked".
_SCORE_COMMIT_EVERY = 200


def _preserved_siglip2_groups(bank_id, by_path) -> dict:
    """Exact inactive SigLIP2 groups that a CLIP write may carry through.

    The first Score run can be the first writer of ``analysis_fingerprint`` on a
    migrated row.  Its generic generation binder clears every pixel-derived
    field before writing CLIP results.  A SigLIP2 group is independent and may be
    restored only when that engine's pinned runtime cache proves the same SHA.
    """
    wanted_ids = sorted(set(by_path.values()))
    if not wanted_ids:
        return {}
    # Chunked like every other id lookup in this file. `by_path` is the WHOLE
    # pool the score pass was handed — 33 932 ids on a real bank — and a single
    # IN() of that size raises "too many SQL variables" against SQLite's 999
    # ceiling. It failed where it hurts most: this runs on the Stop/salvage path,
    # so the exception replaced the write-back and threw away 1 225 images of
    # finished GPU work, which is precisely what that path exists to save.
    rows = []
    for i0 in range(0, len(wanted_ids), _SQL_IN_CHUNK):
        rows.extend(BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(wanted_ids[i0:i0 + _SQL_IN_CHUNK]),
            BankImage.siglip2_semantic_dup_group.isnot(None)).all())
    if not rows:
        return {}
    bank = db.session.get(ImageBank, bank_id)
    if bank is None:
        return {}
    embeddings = bank_semantic_engine.load_semantic_embeddings(
        bank, engine='siglip2')
    row_by_id = {row.id: row for row in rows}
    preserved = {}
    for path, image_id in by_path.items():
        row = row_by_id.get(image_id)
        fingerprint = bank_semantic_engine.embedding_fingerprint(
            path, engine='siglip2')
        if row is not None and path in embeddings and fingerprint is not None:
            preserved[image_id] = (
                row.siglip2_semantic_dup_group, fingerprint)
    return preserved


def _apply_score_results(job, by_path, results, interruptible, *,
                         preserved_siglip2_groups=None,
                         selected_engine='clip'):
    """Write the PER-IMAGE scores (aesthetic, nsfw) back. Returns
    (written, scored, vanished, stale, stopped).

    Two rules earn their place here:

    * a head's value is written only when the child produced one. It used to be
      assigned unconditionally from ``res.get(...)``, so a run where the
      aesthetic weights failed to download did not merely skip that column — it
      wrote NULL over every aesthetic score the bank already had. Same rule as
      the face pass and its yaw: never blank a good value because this run had
      nothing to say.
    * ``interruptible`` is False on the salvage path (the child was stopped and
      handed us what it had computed). There, the job flag is ALREADY set, so
      honouring it would mean discarding the exact GPU work we came to save.

    Whatever it wrote is committed either way: the job registry is in-memory, so
    a pass that only wrote at the very end lost everything to a restart."""
    written = scored = vanished = stale = 0
    stopped = False
    # The write-back is the parent's own mute step, and on a large bank it is
    # minutes long: 21 000 rows one at a time, committed in batches. It used to
    # run under the child's last progress line ("373 / 373"), so the pass looked
    # finished and hung at the same time. Its own counter starts here.
    total = len(by_path)
    bank_jobs.progress(job, done=0, total=total, detail=(
        f'writing {total} score(s) to the database…'
        + (' — a few minutes on a bank this size' if total >= 5000 else '')))
    # What Stop costs HERE, in the phase the user is actually looking at. The
    # commit batch below is why the promise has to name a wait at all: the flag
    # is read once every `_SCORE_COMMIT_EVERY` rows, so a click can sit for a
    # batch, and a button that says nothing about that is indistinguishable from
    # a button that did not register the click.
    #
    # The SALVAGE call (interruptible=False) reaches this same code with the
    # cancel flag already set, and it deliberately ignores it — this write IS
    # the rescue of the stopped run's GPU work. Promising "the current batch"
    # there would be a lie the user watches fail: the counter runs to the end.
    bank_jobs.set_stop_notice(
        job,
        cost='Scores already written stay. The style grouping is not written '
             'yet and only re-runs whole, so it needs another full pass.',
        wait=(f'Stopping — finishing the current batch of {_SCORE_COMMIT_EVERY} '
              'rows, then saving.') if interruptible else
             ('Stopping — saving the scores this run already computed; this '
              'last write runs to the end so none of them is thrown away.'))
    for p, image_id in by_path.items():
        res = results.get(p)
        if res is None:
            continue          # never reached by this run — leave the row alone
        row = _live_image(image_id)
        if row is None:      # deleted while the pass ran — see _live_image
            vanished += 1
            continue
        prepared = _prepare_analysis_write(row, p, res.get('fingerprint'))
        preserved = (preserved_siglip2_groups or {}).get(image_id)
        if not prepared:
            # The child measured a different incarnation of this live path.
            # Its hash-bearing cache entry is discarded/recomputed next run;
            # no scalar from it may be attached to the replacement bytes.
            _restore_proven_siglip2_group(
                row, p, preserved, selected_engine=selected_engine)
            stale += 1
            continue
        _restore_proven_siglip2_group(
            row, p, preserved, selected_engine=selected_engine,
            accepted_fingerprint=res.get('fingerprint'))
        if 'aesthetic' in res:
            row.aesthetic_score = res['aesthetic']
        if 'nsfw' in res:
            row.nsfw_score = res['nsfw']
        # Counted HERE, on the row we actually wrote — not from the child's
        # report. The child scores a PATH; this loop is the only place that
        # knows whether the image behind it still exists. Counting the report
        # made the pass announce "scored 3 image(s), 1 skipped" over a bank of
        # three, which is two claims that cannot both be true.
        if res.get('state') == 'ok':
            scored += 1
        written += 1
        if written % _SCORE_COMMIT_EVERY == 0:
            db.session.commit()
            bank_jobs.progress(job, done=written)
            # Stop is honoured HERE and not per row, on purpose: the rows in
            # hand are already computed, so abandoning them buys nothing except
            # a relaunch. A bank smaller than one commit therefore always
            # finishes rather than reporting "stopped after 1 image".
            if interruptible and bank_jobs.cancelled(job):
                stopped = True
                break
    db.session.commit()
    return written, scored, vanished, stale, stopped


# A style grouping that swallowed almost the whole bank and one that grouped
# nothing at all are OPPOSITE failures — and until this function existed they
# printed the same sentence. "1 style group(s) of 2+" was what a bank of 24 931
# images reported when ONE group held 24 928 of them: read as "almost nothing
# grouped", when the truth was "everything did". Below these two floors the
# diagnosis is suppressed entirely — on a handful of images a single group is an
# ordinary answer, not a symptom, and a warning there would be noise.
_STYLE_DIAGNOSIS_MIN = 20      # images, under which no verdict is offered
_STYLE_DOMINANT_SHARE = 0.9    # of the grouped images, in ONE group


def group_summary(sizes, noun='style group', threshold=None, setting=''):
    """The end-of-pass sentence about a CLUSTERING, from the group SIZES.

    Never quotes a count of groups without the shape behind it: the reader has to
    be able to tell "the threshold is too permissive to separate anything" from
    "too strict to join anything" without opening the database. Shared by the
    style grouping and the person clustering — both are one union-find over
    embeddings, so both fail in exactly these two opposite ways."""
    sizes = sorted((int(n) for n in sizes), reverse=True)
    total = sum(sizes)
    if not total:
        return f'no {noun} — no image carried a usable embedding'
    multi = sum(1 for n in sizes if n >= 2)
    biggest = sizes[0]
    th = '' if threshold is None else f' ({threshold:g})'
    where = f' — retune it in {setting} and re-run' if setting else ''
    if total >= _STYLE_DIAGNOSIS_MIN and biggest >= _STYLE_DOMINANT_SHARE * total:
        return (f'1 {noun} holds {biggest} of {total} images — the threshold{th} '
                'is too permissive to separate anything' + where)
    if total >= _STYLE_DIAGNOSIS_MIN and not multi:
        return (f'no two images grouped — {len(sizes)} {noun}s of one; the '
                f'threshold{th} is too strict to join anything' + where)
    return (f'{multi} {noun}(s) of 2+ '
            f'(the biggest holds {biggest} of {total} images)')


def _write_style_clusters(by_path, clusters, results):
    """Write the style partition — all of it or none of it.

    ``style_cluster`` is not a per-image measurement, it is one numbering of the
    whole bank, recomputed (and renumbered) every pass. Half of a new partition
    next to half of the old one is not "partial progress", it is two different
    meanings sharing an id space: the 🎨 chip would mix unrelated groups and the
    semantic-dedup pass, which only compares inside a cluster, would silently
    stop looking across the seam. So this is deliberately NOT interruptible, and
    the child hands us ``None`` (not a half-clustering) whenever it was stopped.

    Grouped bulk UPDATEs rather than a write per row: same number of rows, a few
    hundred statements instead of tens of thousands."""
    # Revalidate the entire partition immediately before its bulk write.  One
    # changed member changes the meaning of the global clustering, so unlike
    # independent scalar scores this lane is all-or-none.
    for p, image_id in by_path.items():
        result = results.get(p) or {}
        row = _live_image(image_id)
        fingerprint = result.get('fingerprint')
        if (row is None or row.analysis_fingerprint != fingerprint
                or bank_transfer_metadata.content_fingerprint_path(p) != fingerprint):
            all_ids = list(by_path.values())
            for i0 in range(0, len(all_ids), _SQL_IN_CHUNK):
                BankImage.query.filter(
                    BankImage.id.in_(all_ids[i0:i0 + _SQL_IN_CHUNK])).update(
                    {'style_cluster': None}, synchronize_session=False)
            db.session.commit()
            return False
    by_cid: dict = {}
    for p, image_id in by_path.items():
        by_cid.setdefault(clusters.get(p), []).append(image_id)
    since_commit = 0
    for cid, ids in by_cid.items():
        for i0 in range(0, len(ids), _SQL_IN_CHUNK):
            chunk = ids[i0:i0 + _SQL_IN_CHUNK]
            BankImage.query.filter(BankImage.id.in_(chunk)).update(
                {'style_cluster': cid}, synchronize_session=False)
            since_commit += len(chunk)
            if since_commit >= _DUP_WRITE_ROWS:
                db.session.commit()
                # Same yield the duplicate regrouping needs: without it the next
                # batch re-takes SQLite's single write lock in the microsecond
                # after the commit, inside another writer's busy-handler sleep,
                # and that writer dies on "database is locked".
                time.sleep(_DUP_WRITE_YIELD)
                since_commit = 0
    db.session.commit()
    return True


def _score_job(bank_id, rescore=False):
    def run(job):
        import json as _json
        import sys
        from contextlib import nullcontext

        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        by_path = {}
        for r in rows:
            p = analysis_image_path(bank, r, refresh_rotation=True)
            if _is_safe_bank_source(p, label='bank scoring pass'):
                by_path[p] = r.id
        paths = list(by_path)
        device, use_gpu = _resolve_score_device()
        # Say WHICH device, every time. On CPU this pass is ~20× slower (CLIP
        # ViT-L is the whole cost), and a progress bar crawling for an hour with
        # no explanation reads as a hang.
        bank_jobs.progress(job, done=0, total=len(paths),
                           detail=f'scoring pass ({device.upper()})')
        # Phase 1 of three, each with a different price — see the cancelled
        # branch below: whatever the child computed before it was told to stop
        # is written to the database, and only the global style partition is
        # lost (it is computed over the whole bank at once and cannot be halved).
        bank_jobs.set_stop_notice(
            job,
            cost='Every image already scored is saved when you stop. The style '
                 'grouping is computed over the whole bank at once, so it is '
                 'the one thing a stop loses.',
            wait='Stopping — waiting for the image being analyzed, then saving '
                 'what was scored.')
        if not paths:
            # Never leave the label of a pass that did not run: the one-click
            # tunnel falls back to a count read from the DATABASE when a step
            # says nothing, so a mute return here reported "scored N image(s)"
            # over a run that scored none of them.
            bank_jobs.progress(job, detail=(
                'done — nothing to score: every image in this bank is either '
                'rejected or unreadable'))
            return
        _bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        th = thresholds()
        cache_path = _score_cache_path(bank_id)
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('bank_scoring.models_root') or None,
            'cache': str(cache_path),
            'cancel_file': str(cache_path) + '.cancel',
            'style_threshold': th['style_threshold'],
            'rescore': bool(rescore),
        })
        python = cfg.get('bank_scoring.python') or sys.executable
        # The GPU-exclusive window frees ComfyUI's VRAM and blocks any training
        # start for the whole pass — so it is taken ONLY when the child really
        # runs on the card, exactly like the face pass. Holding it through an
        # hour of CPU inference was the worst of both worlds: the GPU idle and
        # unusable, the work slow anyway.
        window = (gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu
                  else nullcontext())
        _release_db_before_inference()
        data, stderr_tail, returncode = _drive_infer_subprocess(
            job, python, _SCORE_SCRIPT, payload, cache_path, _SCORE_PROGRESS_RE,
            window)
        score_results = data.get('results') or {}
        preserved_siglip2_groups = (
            _preserved_siglip2_groups(bank_id, by_path)
            if score_results else {})
        current_bank = db.session.get(ImageBank, bank_id)
        selected_engine = (
            _selected_semantic_engine(current_bank) if current_bank else 'clip')
        # Stopped by the user. The scores the child DID compute are paid GPU work
        # and land in the database here — the pass used to return empty-handed, so
        # an hour of inference reached the disk cache and never reached a single
        # row. The style partition is the one thing that cannot come back with it:
        # it is global by construction and takes minutes (measured 181 s over
        # 23 000 images), which is why the child hands us None for it instead of
        # half of one. Relaunching finishes from the cache and clusters then.
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            saved = 0
            if score_results:
                _, saved, _v, _stale, _s = _apply_score_results(
                    job, by_path, score_results, interruptible=False,
                    preserved_siglip2_groups=preserved_siglip2_groups,
                    selected_engine=selected_engine)
            suffix = (f'; {saved} score(s) saved, style groups need a full pass'
                      if saved else '')
            bank_jobs.progress(job, detail=_stopped_detail(
                'images scored', data, cache_path, len(paths), suffix=suffix))
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'scoring pass produced no output '
                                       f'(rc={returncode})')
        results = score_results
        clusters = data.get('clusters') or {}
        computed = data.get('computed')
        reused = data.get('reused')
        written, scored, vanished, stale, stopped = _apply_score_results(
            job, by_path, results, interruptible=True,
            preserved_siglip2_groups=preserved_siglip2_groups,
            selected_engine=selected_engine)
        if vanished:
            logger.info('bank scoring pass: %s image(s) were deleted while it ran',
                        vanished)
        # Owed by BOTH endings of this pass — the stopped one below and the
        # finished one at the tail (see _skipped_note).
        skipped = _skipped_note(vanished=vanished, stale=stale)
        if stopped:
            # Stopped while saving. What landed is committed; the partition is
            # left alone precisely because only part of the bank got its scores.
            bank_jobs.progress(job, detail=(
                f'Stopped while saving — {written} image(s) written '
                # Rows that vanished under the pass are neither written nor
                # left: counting them here would inflate the only number the
                # user has to judge how much is still pending.
                f'({len(paths) - written - vanished} left); nothing was '
                'recomputed, relaunch finishes from the cache' + skipped))
            return
        # The last mute step, and the one whose Stop semantics differ from every
        # other: the partition is written whole or not at all (see
        # _write_style_clusters), so a Stop landing here does NOT undo it.
        # The sentence about Stop used to live INSIDE this label, where it was
        # read by everyone at all times and answered a question only the person
        # reaching for the button is asking — and at 400 px it pushed the
        # counter and the pass name off their line. It now sits next to the
        # button, as this phase's promise.
        bank_jobs.progress(job, done=0, total=0, detail=(
            f'writing the style grouping over {len(clusters)} image(s)'))
        bank_jobs.set_stop_notice(
            job,
            cost='This step finishes even if you Stop — a half-written grouping '
                 'would mix two numberings. Nothing already written is lost.',
            wait='Stopping — the style grouping is written whole first.')
        style_written = _write_style_clusters(by_path, clusters, results)
        # 🎨 Medium runs next inside this same job and stops on its own terms;
        # leaving the grouping's promise up would describe a step that is over.
        bank_jobs.set_stop_notice(job)
        sizes = {}
        for cid in clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        # The child's own report — used ONLY to name a head that produced nothing
        # (below). It counts PATHS it was handed, so it is the wrong thing to
        # report as "scored": see the counter in the write-back loop.
        ok = [r for r in results.values() if r.get('state') == 'ok']
        # Name any head that produced nothing, so a degraded pass says so out loud
        # (graceful degradation must be visible, never a silent gap).
        missing = []
        if ok and not any('aesthetic' in r for r in ok):
            missing.append('aesthetic')
        if ok and not any('nsfw' in r for r in ok):
            missing.append('NSFW')
        detail = (f'done — scored {scored} image(s), '
                  + group_summary(sizes.values(), 'style group',
                                  th['style_threshold'],
                                  '🎚 Filter thresholds ▸ style_threshold'))
        if not style_written:
            detail += ' (style grouping discarded: an image changed during write-back)'
        # Where the work went. Both numbers count images HANDED to the pass, so
        # they add up to the pool and never stand in for `scored`, which counts
        # rows actually written. Saying "scored N" over a bank that recomputed
        # nothing would be the auto-reject mistake again: a true-looking total
        # covering an action that did not happen.
        if reused:
            detail += (f' · {computed} newly computed, '
                       f'{reused} reused from cache')
        detail += skipped
        if missing:
            detail += f' ({" + ".join(missing)} head unavailable'
            # WHY, when the child said so. Both heads fetch their weights over the
            # network on first use, so a host with no egress loses BOTH at once and
            # the pass reports "done" over a bank whose every score is empty — the
            # sort stays greyed out and the sentence explained nothing. The child
            # always knew the exception; it just never travelled. Older children
            # (a stopped update) send no `head_errors`, so the sentence degrades to
            # exactly what it said before rather than growing an empty bracket.
            why = data.get('head_errors') or {}
            causes = list(dict.fromkeys(
                str(why[k]) for k in ('aesthetic', 'nsfw') if why.get(k)))
            if causes:
                detail += ' — ' + ' · '.join(causes)
            detail += ')'
        if not bank_jobs.cancelled(job):
            detail += _chain_medium_after_score(job, bank_id)
        bank_jobs.progress(job, detail=detail)
    return run


def _chain_medium_after_score(job, bank_id) -> str:
    """Run 🎨 Medium immediately after ✨ Score, inside the SAME job.

    Why automatically: Medium computes NO image inference of its own — it reads
    the embeddings Score just cached and multiplies them by a handful of text
    vectors (0.16 s for 23 000 images, no GPU). Leaving it behind a second button
    meant the bank sat there with the data for the answer and not the answer,
    and most users never learned the pass existed. Chaining it here is the only
    moment it is genuinely free: the embeddings are on disk and warm.

    The manual 🎨 Medium button stays exactly as it was — it is how you re-run
    the pass alone, and how ``rescan`` re-classifies rows that already have a
    verdict, which this chain deliberately never does.

    NEVER raises: Score has already succeeded by the time this runs, and a
    classification that could not run must not turn a finished pass red. It
    returns a suffix for the pass's detail line, so what happened is visible
    either way — including "skipped", with the reason."""
    reason = medium_prereq(bank_id)
    if reason:
        # The common one is "the text encoder is not installed" — a Setup step,
        # not a failure of this bank. Said once, in passing, never as an alarm.
        return f' · 🎨 Medium skipped ({reason})'
    before = (BankImage.query.filter_by(bank_id=bank_id)
              .filter(BankImage.medium.isnot(None)).count())
    try:
        # NO SCOPE, explicitly. The scope of a 🎨 Medium run is a per-run choice
        # made in its own dialog and stored nowhere, so there is nothing here to
        # read — inventing one would make this chained pass and the button
        # disagree about the same word. The chain therefore always runs the
        # DEFAULT pool (every non-rejected row with no verdict yet), which is
        # byte-identical to what it did before the scopes existed, and the
        # dialog says so under "not decided here".
        _medium_job(bank_id, False, None, None)(job)
    except Exception as e:  # noqa: BLE001 — see the docstring
        logger.warning('bank %s: chained medium pass failed', bank_id,
                       exc_info=True)
        return (f' · 🎨 Medium could not run ({type(e).__name__}) — '
                'the 🎨 Medium button re-runs it')
    after = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.medium.isnot(None)).count())
    return f' · 🎨 Medium: {after - before} classified'


# --- watermark pass (reuses the dataset Qwen3-VL overlaid-mark detector) -----
def _watermark_not_dismissed():
    return or_(BankImage.watermark_state.is_(None),
               BankImage.watermark_state != 'dismissed',
               _watermark_history_inactive_clause())


def _watermark_todo_clause():
    """The "not answered yet" half, on its own — one expression for the pool and
    for the counter that prices it."""
    return and_(_watermark_not_dismissed(),
                or_(_watermark_history_inactive_clause(),
                    BankImage.watermark_fingerprint.is_(None),
                    func.length(BankImage.watermark_fingerprint) != 64,
                    BankImage.watermark_state.is_(None),
                    and_(BankImage.watermark_state == 'detected',
                         BankImage.watermark_bbox.is_(None))))


def _watermark_scan_query(bank_id, rescan, statuses=None, ids=None):
    """The rows the detection pass should look at.

    Not a rescan = "finish the job": rows never scanned, PLUS rows flagged
    'detected' with no bbox. The latter exist because the pass used to parse the
    box and keep only the boolean — they would be invisible to both cleaning
    levels forever, so a plain re-run adopts them instead of asking the user to
    guess. 'dismissed' rows are never re-examined (the user already ruled), even
    on a rescan — same anti-frustration rule as the dataset detector."""
    q = _scoped_pool(bank_id, statuses, ids).filter(_watermark_not_dismissed())
    if not rescan:
        q = q.filter(_watermark_todo_clause())
    return q


def start_watermark(app, user_id, bank_id, rescan=False, statuses=None, ids=None):
    """Launch the overlaid-watermark scan over the bank's non-rejected images.

    TWO routes, and which one runs is decided here, once:

      * the dedicated DETECTOR extra when it is installed — a SigLIP2 classifier
        that answers the binary question in ~0.14 s per image against the vision
        model's ~1.7 s, plus a second model that locates the mark. It does not
        need Ollama at all, so a machine with no vision model can still scan.
      * otherwise the vision model, exactly as before. That is the fail-open
        contract: the extra can only ever ADD a faster route, never remove the
        one that has always worked.

    Serialized against training/vision (503 when the GPU is held)."""
    from ..capabilities import probe_ollama_model
    from . import watermark_detector
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    # Occupancy BEFORE the model probe, and the order is the whole point: a bank
    # that is already busy is busy whether or not Ollama answers. Probing first
    # made a busy bank report "the vision model is not available" whenever the
    # model happened to be unreachable -- the wrong reason, sending the user to
    # fix an unrelated thing while the real answer was "wait, a pass is running".
    # It also made the refusal depend on an EXTERNAL SERVICE, which is why CI
    # (no Ollama) failed a release on it while two agents read it as a flake.
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy((bank_jobs.get(bank_id) or {}).get('kind') or 'background')
    # WHICH route, resolved once, from the setting both surfaces read. 'auto' —
    # the default — resolves exactly the way this line used to read the probe
    # directly, so an untouched install behaves identically. A pinned 'detector'
    # with no extra installed does not refuse: it runs the vision route and
    # carries the reason into the job's own progress line (`note`), because a
    # silent fallback is how a user changes a detector and sees nothing change.
    resolution = watermark_detector.resolve_backend()
    use_detector = resolution['backend'] == 'detector'
    if not use_detector and not probe_ollama_model().get('ok'):
        raise RuntimeError('the vision model is not available '
                           '(Settings ▸ Local tools)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    want = normalize_pass_statuses(statuses)
    return bank_jobs.start(app, bank_id, 'watermark',
                           _watermark_job(bank_id, rescan, use_detector=use_detector,
                                          statuses=want, ids=ids,
                                          note=resolution['detail'] if resolution['fell_back'] else ''),
                           total=_watermark_scan_query(bank_id, rescan, want, ids).count())


def _watermark_job(bank_id, rescan, use_detector=False, statuses=None, ids=None, note=''):
    if use_detector:
        return _watermark_detector_job(bank_id, rescan, statuses, ids)

    def run(job):
        import json as _json
        from .face_dataset_service import WATERMARK_BBOX_PROMPT, _parse_watermark_bbox
        from .vision_ollama import describe_image_ollama, unload_vision_model
        from .vision_pool import map_vision
        from ..gpu_window import gpu_exclusive_vision_window
        bank = _detach_bank(db.session.get(ImageBank, bank_id))
        if not bank:
            return
        rows = _watermark_scan_query(bank_id, rescan, statuses, ids).order_by(BankImage.id.asc()).all()
        # `note` is set only when the user PINNED the detector and it could not
        # run. Said at the start (so it is visible while the pass runs) and again
        # in the final sentence (so it survives the pass).
        bank_jobs.progress(job, done=0, total=len(rows),
                           detail=(f'watermark scan — ℹ {note}' if note else 'watermark scan'))
        if not rows:
            return
        row_ids = [r.id for r in rows]
        detected = clean = errors = unanswered = vanished = 0
        missing = stale = 0

        def prepared():
            """Yielded on the JOB's thread, one image per free slot in the pool.
            Everything that reads the database or has a side effect lives here,
            so it stays off the workers, keeps its original order, and is only
            paid for by images the pass actually reaches (which matters: the
            discard below is destructive)."""
            nonlocal vanished
            for rid in row_ids:
                # no_autoflush, same reason as the quality scan and 📐 Framing:
                # this SELECT would otherwise flush, and this generator is pulled
                # AHEAD of the calls in flight (vision_pool), so what it flushed
                # would be locked in for their whole duration. The per-image
                # commit below already keeps that window to one call — this keeps
                # it to none, which matters because the writer queueing behind it
                # is the vision GPU window's own renewal.
                with db.session.no_autoflush:
                    row = _live_image(rid)
                if row is None:      # deleted since the pass started — see _live_image
                    logger.info('bank watermark scan: image %s was deleted mid-pass, '
                                'skipping it', rid)
                    vanished += 1
                    bank_jobs.bump(job)
                    continue
                # Always detect on the SOURCE pixels: a re-scan of an already
                # cleaned image drops its cleaned version first (otherwise we
                # would be asking "is there a watermark?" about our own edit).
                if row.watermark_clean_method:
                    _discard_clean_blob(bank_id, row)
                    _invalidate_effective_analysis(row)
                # Watermark boxes are stored in raw, EXIF-oriented source
                # coordinates because cleaning consumes ``abs_image_path``.
                # Rotation/clean derivatives belong to the effective-analysis
                # lanes and would make the saved geometry unusable here.
                yield rid, abs_image_path(bank, row)

        def ask(item):
            """WORKER thread: read the file, ask Ollama. Touches no session — the
            path was resolved above, on the owning thread. Returns None (not '')
            for a file that is gone, so the caller can tell "nothing to analyse"
            from "the model answered nothing"."""
            _rid, path = item
            if not path or not os.path.isfile(path):
                return None
            payload = _read_safe_bank_source_bytes(
                path, label='bank watermark scan')
            fingerprint = bank_transfer_metadata.content_fingerprint_bytes(payload)
            try:
                raw = describe_image_ollama(
                    payload, WATERMARK_BBOX_PROMPT, num_predict=400,
                    prefer_json=True, fmt='json', keep_alive='5m')
                return {'raw': raw, 'fingerprint': fingerprint, 'error': None}
            except Exception as exc:  # noqa: BLE001 — preserve per-image semantics
                return {'raw': None, 'fingerprint': fingerprint,
                        'error': f'{type(exc).__name__}: {exc}'}

        with gpu_exclusive_vision_window(flag_ttl=1800):
            try:
                # The calls overlap (see vision_pool); the loop body — every
                # database write below — still runs here, on this one thread.
                for (rid, _path), answer, error in map_vision(
                        prepared(), ask,
                        should_cancel=lambda: bank_jobs.cancelled(job)):
                    # Re-read: the answer we are about to store took ~1.7 s to
                    # arrive, and the image can have been deleted in that time.
                    # no_autoflush for the same reason as in prepared() above.
                    with db.session.no_autoflush:
                        row = _live_image(rid)
                    if row is None:
                        logger.info('bank watermark scan: image %s was deleted while '
                                    'it was being analysed, skipping it', rid)
                        vanished += 1
                        bank_jobs.bump(job)
                        continue
                    answer = answer if isinstance(answer, dict) else {}
                    fingerprint = answer.get('fingerprint')
                    raw = answer.get('raw')
                    call_error = error or answer.get('error')
                    if call_error is not None and not _prepare_watermark_write(
                            row, _path, fingerprint):
                        # No verdict is safer than attaching a worker error to
                        # bytes the worker did not actually read. Counted, for
                        # the same reason the empty answer below is: an image
                        # this pass did not write has to appear in its report.
                        stale += 1
                        bank_jobs.bump(job)
                        db.session.commit()
                        continue
                    if call_error is None and raw is not None and not _prepare_watermark_write(
                            row, _path, fingerprint):
                        stale += 1
                        bank_jobs.bump(job)
                        db.session.commit()
                        continue
                    if call_error is not None:  # one bad file never sinks the pass
                        row.watermark_state = 'error'
                        errors += 1
                    elif raw is None:      # file gone: leave the row as it was
                        missing += 1
                    # Empty output = Ollama unreachable, NOT "clean": leave the
                    # state untouched so a retry can finish it (same reasoning as
                    # the dataset detector), never falsely mark everything clean.
                    elif not raw.strip():
                        # COUNTED, not merely skipped: a pass where every image
                        # came back empty reported "done — 0 with a watermark,
                        # 0 clean", which reads as "looked at them all, found
                        # nothing" when in truth nothing could be looked at. The
                        # rows stay unscanned on purpose — the report must say so.
                        unanswered += 1
                    else:
                        bbox = _parse_watermark_bbox(raw)
                        # Stamp WHICH detector ruled, on every row this pass
                        # touches. A bank scanned over weeks can hold verdicts
                        # from both routes, and they disagree at the margins —
                        # so "why is this one flagged?" has to stay answerable
                        # per image rather than per bank.
                        row.watermark_source = 'vision'
                        row.watermark_score = None      # this route has no score
                        # 🔤 guard: a row whose zones the text pass owns keeps
                        # them — a 'none' verdict here is about WATERMARKS and
                        # must not unflag text still waiting for a repaint, and
                        # a found box folds into the regions (they win over the
                        # bbox at cleaning time, so a box left outside them
                        # would never be repainted).
                        text_zones = row.text_state == 'detected'
                        if bbox:
                            row.watermark_state = 'detected'
                            # Keep the box — the crop/inpaint levels route on it.
                            row.watermark_bbox = _json.dumps([round(v, 4) for v in bbox])
                            if text_zones:
                                from .text_regions import text_mask_regions
                                existing, _m, _p = _clean_regions(row)
                                merged, _ = text_mask_regions(
                                    [], [list(b) for b in existing]
                                    + [[round(v, 4) for v in bbox]])
                                row.watermark_regions = _json.dumps(merged)
                            detected += 1
                        else:
                            if not text_zones:
                                row.watermark_state = 'none'
                            row.watermark_bbox = None
                            clean += 1
                    bank_jobs.bump(job)
                    # Never sit in an Ollama call with a transaction open. This
                    # loop WRITES — the 'error' stamp above, the states, and the
                    # destructive discard in prepared() — and the next iteration
                    # waits on a call measured at ~1.7 s, tens of thousands of
                    # times over a big bank. The rhythm used to be "commit every
                    # 25 PARSED answers", which is not a rhythm at all: a pass
                    # whose files erroured, or whose model answered nothing,
                    # never incremented that counter, so the single write lock
                    # was held across an UNBOUNDED number of calls and every
                    # other writer in the app died on `database is locked` past
                    # the 5 s busy_timeout (the failure that abandoned two paid
                    # cloud runs on 2026-07-26 — see
                    # _release_db_before_inference). Committing once per image
                    # bounds the hold by the work traversed whatever the answers
                    # look like, and costs a millisecond against a 1.7 s call.
                    db.session.commit()
            finally:
                db.session.commit()
                unload_vision_model()  # hand the VRAM back to ComfyUI
        skipped = _skipped_note(vanished=vanished, missing=missing,
                                unanswered=unanswered, stale=stale,
                                unreadable=errors)
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {detected} with a watermark '
                                           f'so far' + (f' · ℹ {note}' if note else '')
                                           + skipped)
            return
        detail = f'done — {detected} with a watermark, {clean} clean'
        if note:
            detail += f' · ℹ {note}'
        detail += skipped
        detail += _scope_note(bank_id, _watermark_not_dismissed() if rescan
                              else _watermark_todo_clause(), statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


def _watermark_detector_job(bank_id, rescan, statuses=None, ids=None):
    """The same pass, run by the dedicated detector extra instead of the vision
    model. Deliberately the same SHAPE as the vision job above, because the two
    have to be interchangeable: same resume semantics, same per-image commit,
    same survives-a-deletion discipline, same honest final sentence.

    The one structural difference is where the work happens. The vision route
    overlaps network calls through a thread pool; here a single child process
    holds both models (loading them costs ~10 s, so it must be paid once) and
    streams one verdict per image back. Cancellation therefore travels as a
    sentinel FILE the child polls between images — killing a process that holds
    two loaded models mid-forward is how a stop turns into a corrupted half-write.
    """
    def run(job):
        import contextlib
        import json as _json
        import tempfile
        from . import watermark_detector
        from ..gpu_window import gpu_exclusive_vision_window
        bank = _detach_bank(db.session.get(ImageBank, bank_id))
        if not bank:
            return
        rows = _watermark_scan_query(bank_id, rescan, statuses, ids).order_by(BankImage.id.asc()).all()
        bank_jobs.progress(job, done=0, total=len(rows), detail='watermark scan')
        if not rows:
            return
        detected = clean = errors = vanished = stale = 0
        threshold = watermark_detector.threshold()

        # Paths are resolved HERE, on the owning thread, exactly like the vision
        # route: everything that reads the database or has a side effect (the
        # destructive discard below included) stays off the worker.
        planned = []
        for row_id in [r.id for r in rows]:
            row = _live_image(row_id)
            if row is None:      # deleted since the pass started — see _live_image
                logger.info('bank watermark scan: image %s was deleted mid-pass, '
                            'skipping it', row_id)
                vanished += 1
                bank_jobs.bump(job)
                continue
            # Always detect on the SOURCE pixels: a re-scan of an already cleaned
            # image drops its cleaned version first, or we would be asking "is
            # there a watermark?" about our own edit.
            if row.watermark_clean_method:
                _discard_clean_blob(bank_id, row)
                _invalidate_effective_analysis(row)
            path = abs_image_path(bank, row)
            if not path:
                # No resolvable file. Counted and bumped rather than quietly
                # dropped: a pass that says "done — 12 with a watermark" over a
                # bank of 20 while 8 were never looked at is the kind of silent
                # arithmetic this pass has already been fixed for once.
                vanished += 1
                bank_jobs.bump(job)
                continue
            planned.append((row_id, path))
        db.session.commit()
        if not planned:
            bank_jobs.progress(
                job, detail='done — nothing left to scan'
                + (f' ({vanished} skipped: gone while the pass ran)' if vanished else ''))
            return
        by_path = {}
        for row_id, path in planned:
            by_path.setdefault(path, []).append(row_id)

        cancel_dir = tempfile.mkdtemp(prefix='lds-wmdet-')
        cancel_file = os.path.join(cancel_dir, 'cancel')

        def should_cancel():
            if not bank_jobs.cancelled(job):
                return False
            try:                            # the child polls for this file
                open(cancel_file, 'wb').close()
            except OSError:
                pass
            return True

        # The GPU window is taken only when this extra would actually USE the
        # card. The stock install is CPU-only torch, and a pass that never
        # touches the GPU must never unload ComfyUI or block a training start —
        # the exact rule the scoring pass already follows.
        from ..capabilities import watermark_detect_gpu_available
        on_gpu = watermark_detect_gpu_available()
        window = (gpu_exclusive_vision_window(flag_ttl=1800) if on_gpu
                  else contextlib.nullcontext())
        located = 0
        # Filled by scan() from the child's summary: which device the ranker
        # actually ran on. The pass detail repeats it because "the scan is slow"
        # has exactly one usual cause — a CPU torch in the detector env — and the
        # user cannot fix what nothing tells them.
        run_info = {}
        try:
            with window:
                for path, state, score, regions, fingerprint, error in watermark_detector.scan(
                        [p for _rid, p in planned],
                        should_cancel=should_cancel, cancel_file=cancel_file,
                        info=run_info):
                    # Match on the path the child echoed, popping it so a bank
                    # that holds the same file twice gets one verdict each
                    # rather than both landing on the first row.
                    waiting = by_path.get(path) or []
                    row_id = waiting.pop(0) if waiting else None
                    row = _live_image(row_id) if row_id is not None else None
                    if row is None:
                        logger.info('bank watermark scan: image %s was deleted while '
                                    'it was being analysed, skipping it', row_id)
                        vanished += 1
                        bank_jobs.bump(job)
                        continue
                    if not _prepare_watermark_write(row, path, fingerprint):
                        stale += 1      # same silent skip as the vision route
                        bank_jobs.bump(job)
                        db.session.commit()
                        continue
                    row.watermark_source = 'detector'
                    row.watermark_score = (round(float(score), 4)
                                           if score is not None else None)
                    # 🔤 guard, same as the vision route: text zones survive
                    # this scan, a 'none' verdict never unflags them, and a
                    # found box folds into the regions they live in.
                    text_zones = row.text_state == 'detected'
                    if state == 'error':
                        # One bad file never sinks the pass, same as the vision route.
                        row.watermark_state = 'error'
                        errors += 1
                    elif state == 'detected':
                        row.watermark_state = 'detected'
                        # Only ONE box is persisted: the child's FIRST, which it
                        # orders most-peripheral-first precisely because this
                        # line only takes one (see _merge_boxes — "biggest" put
                        # a crop on the subject). watermark_bbox holds
                        # one rectangle (it is what both cleaning levels route
                        # on), and the multi-zone column next to it means
                        # something else entirely — it is the HAND-DRAWN (or 🔤
                        # text-pass) override, and writing machine output there
                        # would make every flagged image look hand-corrected and
                        # silently exclude it from ✂ Auto-crop. Losing the
                        # smaller boxes is the honest cost; the mask editor
                        # still lets the user add them back.
                        if regions:
                            row.watermark_bbox = _json.dumps(
                                [round(float(v), 4) for v in regions[0][:4]])
                            if text_zones:
                                from .text_regions import text_mask_regions
                                existing, _m, _p = _clean_regions(row)
                                merged, _ = text_mask_regions(
                                    [], [list(b) for b in existing]
                                    + [[round(float(v), 4) for v in regions[0][:4]]])
                                row.watermark_regions = _json.dumps(merged)
                            located += 1
                        else:
                            # Flagged with no box: known to be marked, position
                            # unknown. Exactly the state the pre-box builds
                            # produced, and _watermark_scan_query already adopts
                            # those rows on a plain re-run.
                            row.watermark_bbox = None
                        detected += 1
                    else:
                        if not text_zones:
                            row.watermark_state = 'none'
                        row.watermark_bbox = None
                        clean += 1
                    bank_jobs.bump(job)
                    # Per image, for the same reason the vision route commits per
                    # image: never hold the single SQLite write lock across an
                    # unbounded number of inferences.
                    db.session.commit()
        except watermark_detector.DetectorUnavailable as e:
            # The extra probed OK but could not actually run (weights half
            # downloaded, a torch that no longer imports in that env). Say so and
            # leave every unscanned row untouched — a retry, or an uninstall back
            # to the vision model, both finish the job.
            db.session.commit()
            logger.warning('bank watermark scan: detector unavailable (%s)', e)
            bank_jobs.progress(
                job, detail=f'stopped — the watermark detector could not run ({e}). '
                            'Nothing was mis-flagged; the images it had not reached '
                            'are still unscanned.')
            return
        finally:
            db.session.commit()
            shutil.rmtree(cancel_dir, ignore_errors=True)
        skipped = _skipped_note(vanished=vanished, stale=stale, unreadable=errors)
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {detected} with a watermark '
                                           f'so far' + skipped)
            return
        # 'on GPU' / 'on CPU' comes from the child's own summary, not from the
        # capability probe: the probe says what SHOULD happen, the summary says
        # what DID. A CPU verdict on a machine with a card is the cue to pick a
        # GPU Python in Settings ▸ Watermarks.
        ran_on = {'cuda': ' on GPU', 'cpu': ' on CPU'}.get(run_info.get('device'), '')
        detail = (f'done — {detected} with a watermark, {clean} clean '
                  f'(detector{ran_on}, score ≥ {threshold:g})')
        if detected and located < detected:
            detail += (f', {detected - located} flagged without a position '
                       '(they cannot be cropped or repainted until you draw a zone '
                       'in ▶ Review)')
        detail += skipped
        detail += _scope_note(bank_id, _watermark_not_dismissed() if rescan
                              else _watermark_todo_clause(), statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


# --- 🔤 Find text: the OCR pass that feeds the same funnel -------------------
# Burned-in text — speech bubbles, subtitles, captions, sound effects — is the
# same problem as a watermark once found: pixels stamped over the picture that a
# LoRA would learn. So this pass does NOT grow a second funnel. It reads the
# text with the SAME RapidOCR engine the Video bank's 🔳 Safe zone pass ships
# (capability `video_text`: Apache-2.0, CPU-only, weights inside the wheel),
# folds the per-line boxes into zones (services/text_regions.py), and writes
# those zones into the one channel the cleaning levels already consume —
# watermark_regions + watermark_state='detected'. From there 🧽 Inpaint, the
# mask editor, ↩ Undo, dismiss and promote all behave as if the user had drawn
# the zones by hand, which is the intent: "repaint exactly these" (and ✂
# Auto-crop skips them BY CONSTRUCTION — _crop_todo_clause excludes region-
# carrying rows — because cropping a bubble out of the middle of a page is not
# a thing).
#
# The machine-into-the-hand-channel worry (see _watermark_detector_job, which
# refuses to do this for the WATERMARK box) does not apply here: for text the
# two effects that refusal protects against are exactly the wanted behaviour —
# no crop routing, every zone repainted.
#
# `text_state` is the pass's own memory (resume + report), never a state the
# cleaning levels read. 'dismissed' rows are excluded like every other machine
# pass: dismiss means the MACHINE must stop asking (set_watermark_regions says
# so in as many words), and the hand editor remains the way back in.
TEXT_SCAN_CHUNK = 40   # images per OCR child — the video lane's measured chunk


def _text_todo_clause():
    """Rows the plain 🔤 button still owes an answer: never scanned, or errored
    (a retry adopts those, same as every other pass's todo)."""
    return and_(_watermark_not_dismissed(),
                or_(BankImage.text_state.is_(None),
                    BankImage.text_state == 'error'))


def _text_scan_query(bank_id, rescan, statuses=None, ids=None):
    """The rows the text pass should look at. Not a rescan = finish the job;
    rescan = re-read everything except 'dismissed' (the user already ruled on
    those, and this pass would re-flag them — the exact frustration dismiss
    exists to end)."""
    q = _scoped_pool(bank_id, statuses, ids).filter(_watermark_not_dismissed())
    if not rescan:
        q = q.filter(_text_todo_clause())
    return q


def start_text_scan(app, user_id, bank_id, rescan=False, statuses=None, ids=None):
    """Launch the 🔤 burned-in text scan over the bank's non-rejected images.

    CPU only, by construction: RapidOCR runs on the onnxruntime this app
    installs, so this pass NEVER takes the GPU window — it can measure a bank
    while a training run owns the card, exactly like the video lane's safe-zone
    pass. That is also why there is no _gpu_busy_reason() gate here."""
    from ..capabilities import probe_video_text
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy((bank_jobs.get(bank_id) or {}).get('kind') or 'background')
    probe = probe_video_text()
    if not probe.get('ok'):
        detail = probe.get('detail')
        raise RuntimeError('the text reader is not installed'
                           + (f' — {detail}' if detail else '')
                           + '. Install "Burned-in text" from Setup, then run it again.')
    want = normalize_pass_statuses(statuses)
    return bank_jobs.start(app, bank_id, 'text_scan',
                           _text_scan_job(bank_id, rescan, want, ids),
                           total=_text_scan_query(bank_id, rescan, want, ids).count())


def _text_scan_job(bank_id, rescan, statuses=None, ids=None):
    """Same discipline as the two watermark scan jobs, deliberately: paths are
    resolved on the owning thread, verdicts commit per image, a deletion
    mid-pass is counted and skipped, and a stop keeps everything already
    learned. The one structural difference is the engine seam — the video
    lane's `read_text_boxes` (one child per chunk of 40, the measured shape),
    borrowed rather than re-derived so the two surfaces cannot drift."""
    def run(job):
        import json as _json
        from .text_regions import text_mask_regions
        from .video_safe_zone import read_text_boxes
        bank = _detach_bank(db.session.get(ImageBank, bank_id))
        if not bank:
            return
        rows = _text_scan_query(bank_id, rescan, statuses, ids).order_by(BankImage.id.asc()).all()
        bank_jobs.progress(job, done=0, total=len(rows), detail='text scan')
        if not rows:
            return
        found = clean = errors = vanished = stale = missing = 0
        uncovered = 0     # zones past the 32-zone mask cap, summed — never silent
        planned = []
        for row_id in [r.id for r in rows]:
            row = _live_image(row_id)
            if row is None:      # deleted since the pass started — see _live_image
                logger.info('bank text scan: image %s was deleted mid-pass, '
                            'skipping it', row_id)
                vanished += 1
                bank_jobs.bump(job)
                continue
            path = abs_image_path(bank, row)
            if not path:
                vanished += 1
                bank_jobs.bump(job)
                continue
            if not os.path.isfile(path):
                # The whole source folder can be gone (moved, renamed, a
                # disconnected drive — the bank header says so): counted as
                # "no longer on disk" and NOT sent to the OCR child, which
                # would burn a whole chunk discovering the same absence one
                # file at a time. The row keeps its state — reconnect the
                # folder and a plain run picks it up.
                missing += 1
                bank_jobs.bump(job)
                continue
            planned.append((row_id, path))
        db.session.commit()
        if not planned:
            # The same clauses as the full ending — this early exit is exactly
            # where a whole vanished source folder lands (every file counted
            # 'no longer on disk'), and it must say so, not just "nothing left".
            bank_jobs.progress(
                job, detail='done — nothing left to scan'
                + _skipped_note(vanished=vanished, missing=missing))
            return
        for chunk_start in range(0, len(planned), TEXT_SCAN_CHUNK):
            if bank_jobs.cancelled(job):
                break
            chunk = planned[chunk_start:chunk_start + TEXT_SCAN_CHUNK]
            frames = [{'key': str(rid), 'path': path} for rid, path in chunk]
            try:
                boxes_by_key = read_text_boxes(
                    frames, should_stop=lambda: bank_jobs.cancelled(job))
            except RuntimeError as e:
                # The engine could not run (uninstalled mid-pass, a broken
                # onnxruntime DLL). Say so and leave every unscanned row
                # untouched — a retry finishes the job. Same shape as the
                # watermark scan's DetectorUnavailable exit.
                db.session.commit()
                logger.warning('bank text scan: reader unavailable (%s)', e)
                bank_jobs.progress(
                    job, detail=f'stopped — the text reader could not run ({e}). '
                                'Nothing was mis-flagged; the images it had not '
                                'reached are still unscanned.')
                return
            for rid, path in chunk:
                key = str(rid)
                if key not in boxes_by_key:
                    # Absent ≠ empty is the child's contract, and absence has
                    # exactly two causes. During a Stop it means "never
                    # reached": leave the row unscanned for the next run. With
                    # NO stop asked it means the child could not READ the file
                    # (its reader keys everything it opens, readable-but-empty
                    # included) — that is a per-image error, counted and
                    # retryable, never a reason to call the whole pass
                    # cancelled. A real bank hit this on all 81 images at once
                    # (an unreadable path) and the pass reported "cancelled —
                    # 0 with text" over a stop nobody had asked for.
                    if bank_jobs.cancelled(job):
                        continue
                    row = _live_image(rid)
                    if row is None:
                        vanished += 1
                        bank_jobs.bump(job)
                        continue
                    row.text_state = 'error'
                    errors += 1
                    bank_jobs.bump(job)
                    db.session.commit()
                    continue
                row = _live_image(rid)
                if row is None:
                    logger.info('bank text scan: image %s was deleted while it '
                                'was being read, skipping it', rid)
                    vanished += 1
                    bank_jobs.bump(job)
                    continue
                fingerprint = bank_transfer_metadata.content_fingerprint_path(path)
                if not _prepare_watermark_write(row, path, fingerprint):
                    stale += 1      # same silent skip as both watermark routes
                    bank_jobs.bump(job)
                    db.session.commit()
                    continue
                existing, _manual, problem = _clean_regions(row)
                if problem:
                    # An unreadable stored mask is a broken state, not a value —
                    # this pass replaces it with valid geometry rather than
                    # merging into garbage.
                    existing = []
                regions, dropped = text_mask_regions(
                    boxes_by_key[key], [list(b) for b in existing])
                if not regions:
                    row.text_state = 'none'
                    clean += 1
                else:
                    if row.watermark_clean_method:
                        # Re-flagging a cleaned image: the next repaint restarts
                        # from the SOURCE pixels, so the already-cleaned blob is
                        # dropped now — keeping it would show a "cleaned" badge
                        # over an image the funnel says needs work. The old
                        # zones were folded into `regions` above, so nothing the
                        # previous clean covered is lost on the way back.
                        _discard_clean_blob(bank_id, row)
                        _invalidate_effective_analysis(row)
                    row.watermark_regions = _json.dumps(regions)
                    row.watermark_state = 'detected'
                    row.text_state = 'detected'
                    found += 1
                    uncovered += dropped
                bank_jobs.bump(job)
                # Per image, same reason as every scan here: never hold the one
                # SQLite write lock across an unbounded run.
                db.session.commit()
        db.session.commit()
        skipped = _skipped_note(vanished=vanished, missing=missing, stale=stale,
                                unreadable=errors)
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {found} with text so far'
                                           + skipped)
            return
        detail = f'done — {found} with text, {clean} without'
        if uncovered:
            # The mask channel holds 32 zones per image; a text-heavy page can
            # produce more. The kept zones are the biggest blocks — say what the
            # cap left out rather than reporting "done" over a partial mask.
            detail += (f', {uncovered} zone(s) beyond the 32-zone mask cap '
                       '(draw them in ▶ Review if they matter)')
        detail += skipped
        detail += _scope_note(bank_id, _text_todo_clause(), statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


# --- watermark cleaning: two MANUAL levels ----------------------------------
# The bank's folder belongs to the user and is never written to, so "cleaning a
# watermark" here means writing a SEPARATE cleaned blob under the bank's own
# working directory (clean/<image_id>.webp) and pointing the readers at it
# (resolved_image_path). The original therefore stays untouched by construction,
# which is what makes undo trivial and both levels risk-free.
#
# The escalation is deliberate and each level is launched BY HAND:
#   level 1 — ✂ auto-crop: CPU/PIL, invents no pixel. Only touches marks the
#             dataset router calls croppable (inside a border band, and the crop
#             still leaves a usable image). Everything else is left flagged.
#   level 2 — 🧽 inpaint: LaMa (fast, non-generative) or Klein (ComfyUI, handles
#             on-subject marks) over what is STILL flagged. allow_crop=False, so
#             a border mark that level 1 skipped (or that the user never cropped)
#             is repainted rather than cropped — level 2 is the "repaint what's
#             left" lane, not a second router.
# Both reuse the dataset routing/engines verbatim; nothing about the decision
# logic is re-implemented here.
def _clean_todo_clause():
    """"Still flagged, geometry authorised for the raw on disk, and SOMETHING to
    act on" — the cleaning levels' work, with no status in it.

    Split out of _clean_pool_query so the pool a level WALKS and the per-pile
    counter its window quotes are ONE expression (the contract of
    _todo_by_status). "Something to act on" is the stored bbox OR a hand-drawn
    mask: a row the detector left without a box (an older build) becomes
    cleanable the moment the user draws the zones themselves — that drawing IS
    the missing box. The fingerprint conditions are not decoration: a geometry
    that was not attested against the current raw bytes must never drive a
    write (see _needs_rescan_count, which counts exactly their complement)."""
    return and_(BankImage.watermark_state == 'detected',
                BankImage.watermark_fingerprint.isnot(None),
                func.length(BankImage.watermark_fingerprint) == 64,
                ~_watermark_history_inactive_clause(),
                or_(BankImage.watermark_bbox.isnot(None),
                    BankImage.watermark_regions.isnot(None)))


def _crop_todo_clause():
    """✂ Auto-crop's own work: the same flagged rows MINUS the hand-masked ones.

    A hand mask is 🧽 Inpaint's material (see _watermark_crop_job) — cropping
    cannot express several zones, nor a zone on the subject. Counting them here
    would price a run that can only skip them."""
    return and_(_clean_todo_clause(), BankImage.watermark_regions.is_(None))


def _clean_pool_query(bank_id, statuses=None, ids=None):
    """Images a cleaning level can act on, inside the run's scope.
    'cleaned'/'dismissed'/'none' rows are out by construction.

    ``statuses``/``ids`` are the two dials every other pass takes, and they
    INTERSECT the flagged set — they never widen it. Left alone
    (``statuses=None``) _scope_clause yields ``status != 'reject'``, which is
    character for character the filter this pool has always carried, so an
    untouched run walks exactly the rows it walked before."""
    return _scoped_pool(bank_id, statuses, ids).filter(_clean_todo_clause())


def _needs_rescan_count(bank_id) -> int:
    """Flagged rows whose geometry is not authorised for their current raw."""
    return (BankImage.query.filter_by(bank_id=bank_id, watermark_state='detected')
            .filter(BankImage.status != 'reject')
            .filter(or_(
                and_(BankImage.watermark_bbox.is_(None),
                     BankImage.watermark_regions.is_(None)),
                BankImage.watermark_fingerprint.is_(None),
                func.length(BankImage.watermark_fingerprint) != 64,
                _watermark_history_inactive_clause(),
            )).count())


def _drop_clean_blob_by_id(bank_id, image_id) -> None:
    """_discard_clean_blob for an image whose ROW is gone (deleted mid-pass).

    Only the blob and the derived thumbnails are left to clean up — there is no
    `watermark_clean_method` to clear, because there is no row. Without this a
    cancelled or interrupted cleaning level would leave the staged copy behind
    with nothing pointing at it."""
    try:
        clean_image_path(bank_id, image_id).unlink()
    except OSError:
        pass   # already gone: the staged copy owed nothing
    drop_derived(bank_id, image_id)


def _discard_clean_blob(bank_id, row) -> None:
    """Forget a cleaned version: delete the blob, drop the stale thumbnail and
    clear the method so the readers fall back to the source. No commit (the
    caller owns the transaction)."""
    _drop_clean_blob_by_id(bank_id, row.id)
    row.watermark_clean_method = None


def _clean_bbox(row):
    """The stored bbox as a 4-float tuple, or None when it's unusable."""
    try:
        import json as _json
        box = _json.loads(row.watermark_bbox or '')
    except (ValueError, TypeError):
        return None
    if not (isinstance(box, list) and len(box) == 4):
        return None
    try:
        return tuple(float(v) for v in box)
    except (TypeError, ValueError):
        return None


def _clean_regions(row):
    """THE mask both cleaning levels must act on — ``(boxes, manual, problem)``.

    A hand-drawn mask WINS over the detector's bbox. That is the entire point of
    letting the user edit it: a correction the cleaning pass then ignores is
    worse than no editor at all, because the user believes the fix landed.

    ``manual`` says the boxes came from the user (drives the routing: a hand mask
    is a REPAINT instruction — it can hold several zones and zones on the subject,
    neither of which a border crop can express, exactly like the dataset lane).
    An EMPTY manual mask returns ``([], True, None)``: an explicit "nothing to
    repaint here", never a silent fall back to the box. ``problem`` is set only
    when the stored JSON cannot be read as a mask (a genuine failure)."""
    if row.watermark_regions is not None:
        import json as _json
        try:
            stored = _json.loads(row.watermark_regions or '')
        except (ValueError, TypeError):
            return [], True, 'unreadable watermark regions'
        try:
            regions = normalize_watermark_regions(stored, allow_null=False)
        except ValueError as e:
            return [], True, f'invalid watermark regions: {e}'
        return [tuple(box) for box in regions], True, None
    bbox = _clean_bbox(row)
    return ([bbox] if bbox else []), False, None


def _current_watermark_source(bank, row) -> tuple[str | None, str | None]:
    path = abs_image_path(bank, row)
    if not path or not os.path.isfile(path):
        return None, None
    fingerprint = bank_transfer_metadata.content_fingerprint_path(path)
    if (not _valid_analysis_fingerprint(row.watermark_fingerprint)
            or fingerprint != row.watermark_fingerprint):
        return None, fingerprint
    return path, fingerprint


def _source_size(bank, row):
    """(path, W, H) in the browser/VLM's visual orientation.

    Watermark boxes come from the EXIF-oriented vision payload. Returning raw
    camera dimensions here would route/crop a 90° JPEG in a different coordinate
    system than both the user and the model saw.
    """
    path, _fingerprint = _current_watermark_source(bank, row)
    if not path:
        return None, 0, 0
    try:
        with safe_bank_source(path, label='bank watermark') as im:
            # Header-only: level badges poll this for potentially thousands
            # of files, so do not materialise an EXIF transpose just to know
            # whether browser/VLM geometry swaps axes.
            return path, *image_encoding.visual_size_from_header(im)
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        return None, 0, 0


def _stage_upright_webp(dst: Path, src_path, *, label: str) -> Path:
    """Create an upright, metadata-free WebP copy of ``src_path`` at ``dst``.

    Crop, LaMa and Klein all consume visual/VLM boxes. They therefore edit a
    freshly rebuilt WebP in one of the Bank's own working directories, never a
    byte copy of a raw EXIF-tagged source. The source remains read-only and
    recoverable; the blob is atomically published only once its staging write
    succeeded.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f'.{dst.name}.part-{uuid.uuid4().hex[:8]}')
    try:
        with safe_bank_source(src_path, label=label) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            mode = ('RGBA' if ('A' in oriented.getbands()
                               or 'transparency' in getattr(oriented, 'info', {}))
                    else 'RGB')
            clean = Image.new(mode, oriented.size)
            clean.paste(oriented.convert(mode))
        image_encoding.save_edit(clean, str(tmp), 'WEBP', image_encoding.LOSSLESS)
        os.replace(tmp, dst)
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        try:
            tmp.unlink()
        except OSError:
            pass   # rollback is best-effort: the temp may never have landed
        raise
    return dst


def _stage_clean_copy(bank_id, row, src_path) -> Path:
    """The watermark cleaners' working image, in the Bank's ``clean/``."""
    return _stage_upright_webp(clean_image_path(bank_id, row.id), src_path,
                               label='bank watermark clean')


def _stage_edited_copy(bank_id, row, src_path, generation) -> Path:
    """The ✂ crop / ✨ improve working image, in the Bank's ``edited/``."""
    return _stage_upright_webp(
        edited_image_path(bank_id, row.id, generation), src_path,
        label='bank image edit')


def start_watermark_crop(app, user_id, bank_id, statuses=None, ids=None):
    """Level 1 — crop away every watermark that sits in a border band. Pure
    CPU/PIL, no model, no GPU: this level is always available. ValueError when
    there is nothing to crop (the UI disables the button, this is the race).

    ``statuses``/``ids`` narrow WHERE the crop applies, exactly like every other
    pass, and they intersect the flagged pool rather than widening it. This level
    WRITES AN IMAGE, so the narrowing matters more here than on a pass that only
    computes a verdict — but nothing of the user's is overwritten: the crop lands
    in the bank's own ``clean/`` copy and ↩ Undo cleaning deletes it. Left alone,
    the pool is the one this level has always walked."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    want = normalize_pass_statuses(statuses)
    scoped = _clean_pool_query(bank_id, want, ids)
    # Hand-masked rows are level 2's, so they don't count as work for this level:
    # launching a crop that can only skip them would report "0 cropped" and read
    # as a broken button.
    total = scoped.filter(BankImage.watermark_regions.is_(None)).count()
    if not total:
        if scoped.count():
            raise ValueError('the flagged images all carry a hand-edited mask — '
                             'use 🧽 Inpaint, which repaints the zones you drew')
        # The scope is NAMED in the refusal when there is work outside it. A bare
        # "run the watermark scan first" on a bank holding thousands of flagged
        # images in another pile sends the user to re-run a pass that already did
        # its job.
        if _clean_pool_query(bank_id, list(PASS_SCOPES)).count():
            raise ValueError('nothing to crop in this scope — the flagged images '
                             'are in another pile (kept / undecided / unkept)')
        raise ValueError('no flagged image to clean — run the watermark scan first')
    return bank_jobs.start(app, bank_id, 'watermark_crop',
                           _watermark_crop_job(bank_id, want, ids), total=total)


def _watermark_crop_job(bank_id, statuses=None, ids=None):
    def run(job):
        from .face_dataset_service import _apply_watermark_crop, _route_watermark
        bank = _detach_bank(db.session.get(ImageBank, bank_id))
        if not bank:
            return
        rows = (_clean_pool_query(bank_id, statuses, ids)
                .order_by(BankImage.id.asc()).all())
        bank_jobs.progress(job, done=0, total=len(rows), detail='auto-crop')
        row_ids = [r.id for r in rows]
        cropped = left = failed = vanished = 0
        try:
            for rid in row_ids:
                if bank_jobs.cancelled(job):
                    break
                row = _live_image(rid)
                if row is None:      # deleted since the pass started — see _live_image
                    logger.info('bank auto-crop: image %s was deleted mid-pass, '
                                'skipping it', rid)
                    vanished += 1
                    bank_jobs.bump(job)
                    continue
                boxes, manual, _problem = _clean_regions(row)
                if manual:
                    # A hand mask is level 2's material, mask emptied or not. It
                    # can carry several zones and zones on the subject; cropping
                    # cannot express either, and quietly cropping the detector's
                    # old box would clean pixels the user did NOT point at.
                    left += 1
                    bank_jobs.bump(job)
                    continue
                bbox = boxes[0] if boxes else None
                src, width, height = _source_size(bank, row)
                if not bbox or not src:
                    failed += 1
                    bank_jobs.bump(job)
                    continue
                expected_raw_fingerprint = row.watermark_fingerprint
                route, box = _route_watermark(bbox, width, height, allow_crop=True)
                if route != 'crop':
                    left += 1              # level 2's job — stays 'detected'
                    bank_jobs.bump(job)
                    continue
                try:
                    dst = _stage_clean_copy(bank_id, row, src)
                except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
                        Image.DecompressionBombWarning):
                    _discard_clean_blob(bank_id, row)
                    failed += 1
                    bank_jobs.bump(job)
                    continue
                cleaned_ok = _apply_watermark_crop(str(dst), box)
                generation_ok = _prepare_watermark_write(
                    row, src, expected_raw_fingerprint)
                if cleaned_ok and generation_ok:
                    row.watermark_state = 'cleaned'
                    row.watermark_clean_method = 'crop'
                    _invalidate_effective_analysis(row)
                    cropped += 1
                else:
                    _discard_clean_blob(bank_id, row)
                    failed += 1
                if (cropped + failed) % 25 == 0:
                    db.session.commit()
                bank_jobs.bump(job)
        finally:
            db.session.commit()
            if cropped:
                reset_score_memo()
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {cropped} cropped so far')
            return
        detail = f'done — {cropped} cropped, {left} left for inpainting'
        if vanished:
            detail += f', {vanished} skipped (deleted while the pass ran)'
        if failed:
            detail += f', {failed} unreadable'
        # What the SCOPE never reached, named — otherwise "3 cropped" on a bank
        # with thousands of flagged images in another pile reads as a broken
        # level rather than as the narrow run the user asked for.
        detail += _scope_note(bank_id, _crop_todo_clause(), statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


def _watermark_inpaint_prereq(method) -> str | None:
    """Why level 2 can't run right now, or None. Actionable text — an unavailable
    engine must say what to install, never fail silently mid-pass."""
    from . import watermark_klein, watermark_lama
    if method == 'klein':
        if not watermark_klein.is_available():
            return ('Klein inpainting needs ComfyUI running and the Klein weights '
                    '(Setup ▸ Generation models)')
        return None
    if not watermark_lama.is_available():
        return 'LaMa inpainting is not installed (Setup ▸ Quality tools)'
    return None


# --- ✨ Upscale & improve, on the Bank ---------------------------------------
# The Bank's version of the dataset pass of the same name, and deliberately the
# SAME two engines through the SAME enqueue helpers (face_dataset_service owns
# that dispatch; a parallel copy here would drift the day a third engine lands).
# What differs is everything around them: a bank image has no candidate row to
# review — a bank IS the review — so the result replaces the displayed image and
# ↩ Revert takes it back, exactly like the watermark clean it sits next to.
_IMPROVE_POLL_SECONDS = 2.0
#: One image, one ComfyUI round-trip. Generous because a 4-8 MP Klein pass on a
#: modest card genuinely takes minutes, and a pass that gives up early would
#: report "failed" for a job that is still rendering.
_IMPROVE_TIMEOUT_SECONDS = 1800


def _improve_todo_clause():
    """The rows ✨ Upscale & improve still has work on: everything that is not
    already an improved image. Re-running the pass over a scope must not spend
    GPU-minutes re-rendering what it rendered last time — ↩ Revert is how a user
    asks for a second attempt, and it is one click."""
    return or_(BankImage.edit_method.is_(None),
               BankImage.edit_method != 'improve')


def _improve_pool_query(bank_id, statuses=None, ids=None):
    return _scoped_pool(bank_id, statuses, ids).filter(_improve_todo_clause())


def _await_queue_job(job_id, timeout, *, should_cancel=None):
    """Block until the queue worker finishes ``job_id`` → (status, filename,
    error). ``db.session.rollback()`` each poll drops this thread's stale read
    snapshot so the worker thread's commits become visible — the same
    cross-thread read pattern as watermark_klein, which is the other pass that
    owns its own ComfyUI result instead of waiting for a completion callback."""
    from ..models import ImageGenerationQueue
    deadline = time.monotonic() + timeout
    while True:
        db.session.rollback()
        row = ImageGenerationQueue.query.filter_by(job_id=job_id).first()
        if row is not None and row.status in ('completed', 'failed', 'cancelled'):
            return row.status, row.result_filename, row.error_message
        if should_cancel is not None and should_cancel():
            return 'cancelled', None, None
        if time.monotonic() >= deadline:
            return 'timeout', None, None
        time.sleep(_IMPROVE_POLL_SECONDS)


def _fetch_comfy_result(filename) -> bytes | None:
    """The finished render's bytes: off disk when ComfyUI shares its output
    folder with us, over its /view API when it does not (a container or a remote
    install). Same two-step every other lane uses — a path-only read is how a
    remote ComfyUI silently produces "nothing came back"."""
    if not filename:
        return None
    from .face_dataset_service import _comfy_output_dir
    out_dir = _comfy_output_dir()
    src = os.path.join(out_dir, filename) if out_dir else None
    if src and os.path.isfile(src):
        try:
            with open(src, 'rb') as fh:
                data = fh.read()
        except OSError:
            data = None
        if data:
            try:
                os.remove(src)      # a temporary render, never user data
            except OSError:
                pass
            return data
    from ..utils.comfyui import fetch_output_image_bytes
    return fetch_output_image_bytes(filename)


def _publish_edited_bytes(bank_id, row, generation, data: bytes) -> Path:
    """Write a finished render into the row's next edited generation.

    Re-encoded rather than dropped in as-is: the blob's extension is what tells
    every later reader (and PIL) what it holds, and the whole edited/ lane is
    lossless WebP. Atomic, like every other blob the bank publishes."""
    dst = edited_image_path(bank_id, row.id, generation)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f'.{dst.name}.part-{uuid.uuid4().hex[:8]}')
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.load()
            mode = ('RGBA' if ('A' in opened.getbands()
                               or 'transparency' in getattr(opened, 'info', {}))
                    else 'RGB')
            image_encoding.save_edit(opened.convert(mode), str(tmp), 'WEBP',
                                     image_encoding.LOSSLESS)
        os.replace(tmp, dst)
    except (OSError, ValueError, MemoryError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        try:
            tmp.unlink()
        except OSError:
            pass   # rollback is best-effort: the temp may never have landed
        raise
    return dst


def start_improve(app, user_id, bank_id, statuses=None, ids=None, engine=None):
    """✨ Upscale & improve every image in scope, in the Bank itself.

    Requested by nofaceman on Discord, in the same breath as the crop: the Bank
    is where the filtering and the curation are, so having to promote into a
    Dataset just to upscale — and export back out to curate the result — is the
    detour this removes.

    Refuses BEFORE the 202 rather than failing image by image: a missing engine
    raises its own structured exception (the route turns each into its own
    actionable 409, with the auto-download offer intact) and a busy GPU raises
    RuntimeError → 503. That is the whole reason the preflight is here and not in
    the job thread — one refusal per pass, not one per image.
    """
    from . import face_dataset_service as fds
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    engine = fds.resolve_improve_engine(engine)
    want = normalize_pass_statuses(statuses)
    total = _improve_pool_query(bank_id, want, ids).count()
    if not total:
        if _improve_pool_query(bank_id, list(PASS_SCOPES)).count():
            raise ValueError('nothing to improve in this scope — the images left '
                             'to improve are in another pile (kept / undecided / '
                             'unkept)')
        raise ValueError('every image in this bank has already been improved — '
                         '↩ Revert one to run it again')
    # Raises KleinNodesMissing / KleinModelsMissing / SeedVR2ModelsMissing, which
    # the route maps to its own actionable 409 — do not flatten them here.
    fds._improve_preflight(engine)
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    return bank_jobs.start(app, bank_id, 'improve',
                           _improve_job(bank_id, engine, want, ids), total=total)


def _improve_job(bank_id, engine, statuses=None, ids=None):
    def run(job):
        from . import face_dataset_service as fds
        bank = _detach_bank(db.session.get(ImageBank, bank_id))
        if not bank:
            return
        rows = (_improve_pool_query(bank_id, statuses, ids)
                .order_by(BankImage.id.asc()).all())
        bank_jobs.progress(job, done=0, total=len(rows),
                           detail=f'{engine} upscale')
        # WHAT STOP COSTS HERE, said before the click — and this pass owes that
        # sentence more than any other: it is the only one whose unit of work is
        # a multi-minute GPU render, so "it stops at its next safe point" would
        # be read as instant. The render in flight is not interrupted (it is
        # ComfyUI's, not ours) — we stop waiting for it, and its result is the
        # one thing this pass drops on the way out.
        bank_jobs.set_stop_notice(
            job,
            cost='Every image already improved stays. The one being rendered '
                 'right now is abandoned — its GPU time is lost, and the image '
                 'keeps its current version.',
            wait='Stopping — no new image is sent to ComfyUI; the pass ends as '
                 'soon as it stops waiting for the render in flight.')
        row_ids = [r.id for r in rows]
        improved = failed = vanished = unreadable = 0
        # A bank has no dataset to inherit a Klein model pick from, so this runs
        # on the auto-resolved one — the same value every bank-side Klein pass
        # has always used (see routes/datasets.py, GET /api/klein-model).
        prompt = fds._improve_prompt()
        try:
            for rid in row_ids:
                if bank_jobs.cancelled(job):
                    break
                row = _live_image(rid)
                if row is None:     # deleted since the pass started
                    logger.info('bank improve: image %s was deleted mid-pass, '
                                'skipping it', rid)
                    vanished += 1
                    bank_jobs.bump(job)
                    continue
                src = resolved_image_path(bank, row)
                if not src or not os.path.isfile(src):
                    unreadable += 1
                    bank_jobs.bump(job)
                    continue
                generation = int(row.edit_generation or 0) + 1
                baked = (row.edit_baked_rotation if row.edit_method
                         else (int(row.rotation or 0) % 360 or None))
                # Nothing of ours may hold a write transaction while a multi-minute
                # GPU round-trip runs: the poll below reads across threads, and a
                # held SQLite lock is how a long pass starves everything else.
                db.session.commit()
                try:
                    job_id = fds._enqueue_improve(
                        engine, user_id=bank.user_id, source=row,
                        source_path=src, prompt=prompt,
                        label=None, dataset=None,
                        extra_metadata={'is_bank_improve': True,
                                        'bank_id': bank_id,
                                        'bank_image_id': row.id})
                except Exception as exc:
                    logger.warning('bank improve: image %s could not be queued: %s',
                                   rid, exc)
                    failed += 1
                    bank_jobs.bump(job)
                    continue
                status, filename, err = _await_queue_job(
                    job_id, _IMPROVE_TIMEOUT_SECONDS,
                    should_cancel=lambda: bank_jobs.cancelled(job))
                if status != 'completed':
                    if status != 'cancelled':
                        logger.warning('bank improve: image %s %s (%s)',
                                       rid, status, err or 'no detail')
                        failed += 1
                    bank_jobs.bump(job)
                    if bank_jobs.cancelled(job):
                        break
                    continue
                data = _fetch_comfy_result(filename)
                row = _live_image(rid)
                if row is None:
                    vanished += 1
                    bank_jobs.bump(job)
                    continue
                if not data:
                    logger.warning('bank improve: the finished image for %s could '
                                   'not be retrieved from ComfyUI', rid)
                    failed += 1
                    bank_jobs.bump(job)
                    continue
                try:
                    dst = _publish_edited_bytes(bank_id, row, generation, data)
                except (OSError, ValueError, MemoryError,
                        Image.DecompressionBombError,
                        Image.DecompressionBombWarning) as exc:
                    logger.warning('bank improve: image %s could not be written: %s',
                                   rid, exc)
                    failed += 1
                    bank_jobs.bump(job)
                    continue
                row.edit_method = 'improve'
                row.edit_generation = generation
                row.edit_baked_rotation = baked
                row.rotation = None
                _invalidate_effective_analysis(row)
                _measure_edited_blob(row, dst)
                improved += 1
                db.session.commit()
                _prune_edited_generations(bank_id, row.id, dst)
                bank_jobs.bump(job)
        finally:
            db.session.commit()
            if improved:
                reset_score_memo()
        if bank_jobs.cancelled(job):
            bank_jobs.progress(
                job, detail=f'cancelled — {improved} improved so far')
            return
        detail = f'done — {improved} improved with {engine}'
        detail += _skipped_note(vanished=vanished, unreadable=unreadable)
        if failed:
            detail += f', {failed} failed'
        detail += _scope_note(bank_id, _improve_todo_clause(), statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


def start_watermark_inpaint(app, user_id, bank_id, method='auto',
                            statuses=None, ids=None):
    """Level 2 — repaint what is STILL flagged after the crop level.
    ``method``: 'auto'/'lama' (LaMa, non-generative, small off-centre marks; marks
    on the subject stay flagged for manual review) or 'klein' (masked Flux.2 Klein
    through ComfyUI, which also handles on-subject marks). RuntimeError (→ 503) on
    a missing engine or a busy GPU, ValueError (→ 400) on a bad method / empty pool.

    ``statuses``/``ids`` narrow WHERE the repaint applies — the same two dials as
    every other pass, and the same intersection rule. This level REPAINTS PIXELS,
    which is why it needed them: 16 000 images repainted from one click was a run
    nobody could aim. What it writes is the bank's own ``clean/`` copy and never
    the user's file, so ↩ Undo cleaning takes it back."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    method = (method or 'auto').lower()
    if method not in ('auto', 'lama', 'klein'):
        raise ValueError("method must be 'auto', 'lama' or 'klein'")
    want = normalize_pass_statuses(statuses)
    total = _clean_pool_query(bank_id, want, ids).count()
    if not total:
        # "Nothing HERE" and "nothing anywhere" are two situations with two
        # different next moves. Saying "every flagged image is handled" while
        # thousands sit in another pile is the refusal reading as a lie.
        if _clean_pool_query(bank_id, list(PASS_SCOPES)).count():
            raise ValueError('nothing to repaint in this scope — the flagged '
                             'images are in another pile (kept / undecided / unkept)')
        raise ValueError('nothing left to inpaint — every flagged image is handled')
    problem = _watermark_inpaint_prereq(method)
    if problem:
        raise RuntimeError(problem)
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    return bank_jobs.start(app, bank_id, 'watermark_inpaint',
                           _watermark_inpaint_job(bank_id, method, want, ids),
                           total=total)


def _watermark_inpaint_job(bank_id, method, statuses=None, ids=None):
    def run(job):
        from contextlib import nullcontext
        from . import watermark_klein, watermark_lama
        from .face_dataset_service import _clean_inpaint_engine, _route_watermark
        from ..gpu_window import gpu_exclusive_vision_window
        bank = _detach_bank(db.session.get(ImageBank, bank_id))
        if not bank:
            return
        rows = (_clean_pool_query(bank_id, statuses, ids)
                .order_by(BankImage.id.asc()).all())
        bank_jobs.progress(job, done=0, total=len(rows), detail='inpainting')
        row_ids = [r.id for r in rows]
        counts = {'inpainted': 0, 'klein': 0, 'review': 0, 'failed': 0,
                  'skipped': 0, 'empty': 0, 'vanished': 0}
        error = None
        lama_ok = watermark_lama.is_available()
        klein_ok = method == 'klein' and watermark_klein.is_available()
        # LaMa on the GPU pauses ComfyUI through the exclusive vision window;
        # Klein must NOT take that window — ComfyUI owns the GPU there and
        # holding it would deadlock its worker (same split as the dataset route).
        device = 'cpu' if method == 'klein' else watermark_lama.resolve_device()
        # (image_id, dst_path, [bbox], raw_path, raw_fingerprint) for the single
        # LaMa batch — ids, not ORM rows: this list is held across a batch that
        # can run for minutes, and a row deleted in that window must be skippable
        # rather than fatal. The raw identity closes that same minutes-long
        # window for external source replacements.
        pending = []
        window = (gpu_exclusive_vision_window(flag_ttl=1800)
                  if device == 'cuda' else nullcontext())
        try:
            with window:
                for rid in row_ids:
                    if bank_jobs.cancelled(job):
                        break
                    # no_autoflush: this SELECT flushes by default, and what it
                    # flushed would be locked in until the write-back below —
                    # i.e. across the LaMa batch, which is minutes long. See the
                    # quality scan for the original diagnosis.
                    with db.session.no_autoflush:
                        row = _live_image(rid)
                    if row is None:  # deleted since the pass started — _live_image
                        logger.info('bank inpaint: image %s was deleted mid-pass, '
                                    'skipping it', rid)
                        counts['vanished'] += 1
                        bank_jobs.bump(job)
                        continue
                    boxes, manual, problem = _clean_regions(row)
                    src, width, height = _source_size(bank, row)
                    if problem or not src:
                        counts['failed'] += 1
                        error = error or (
                            {'kind': 'failed', 'detail': problem} if problem else None)
                        bank_jobs.bump(job)
                        continue
                    expected_raw_fingerprint = row.watermark_fingerprint
                    if manual and not boxes:
                        # The user deleted every zone. That is an ANSWER, not a
                        # missing value: repaint nothing, and never fall back to
                        # the detector's box. The row stays flagged (and visible)
                        # so it can be masked again or dismissed.
                        counts['empty'] += 1
                        bank_jobs.bump(job)
                        continue
                    if not boxes:
                        counts['failed'] += 1
                        bank_jobs.bump(job)
                        continue
                    if manual:
                        # Hand-drawn zones bypass the router entirely — same rule
                        # as the dataset: what the user drew IS the decision, and
                        # every zone is repainted with the selected engine.
                        engine = 'klein' if method == 'klein' else 'lama'
                    else:
                        # allow_crop=False: level 2 REPAINTS what is left, including a
                        # border mark the user chose not to crop.
                        route, _box = _route_watermark(boxes[0], width, height,
                                                       allow_crop=False)
                        engine = _clean_inpaint_engine(route, method)
                    if engine == 'review':
                        counts['review'] += 1       # stays flagged, needs Klein or a human
                        bank_jobs.bump(job)
                        continue
                    if (engine == 'klein' and not klein_ok) or \
                       (engine == 'lama' and not lama_ok):
                        counts['skipped'] += 1      # engine gone since launch
                        bank_jobs.bump(job)
                        continue
                    try:
                        dst = _stage_clean_copy(bank_id, row, src)
                    except (OSError, ValueError, MemoryError,
                            Image.DecompressionBombError, Image.DecompressionBombWarning):
                        # Committed here, not left pending until the write-back:
                        # the blob this row pointed at is already deleted from
                        # disk, so an interrupted pass must not survive with a
                        # row still naming it — and a pending write on the way
                        # into a minutes-long batch is exactly the holder this
                        # pass is tested against (test_bank_infer_no_db_lock).
                        _discard_clean_blob(bank_id, row)
                        db.session.commit()
                        counts['failed'] += 1
                        bank_jobs.bump(job)
                        continue
                    if engine == 'klein':
                        # No klein_model on purpose. The Klein model choice lives on
                        # the DATASET (it describes what a dataset is made of) and a
                        # bank has none to inherit — so this pass keeps the auto
                        # resolution it has always used. Deliberately NOT a global
                        # setting and NOT a third picker on the bank: that would be a
                        # second authority for the same UNETLoader. What the bank DOES
                        # owe the user is the name of the model that will run, which
                        # the panel now states (BankWatermarkPanel → /api/klein-model).
                        ok, err = watermark_klein.inpaint_watermark_klein(
                            bank.user_id, str(dst), [list(b) for b in boxes])
                        generation_ok = _prepare_watermark_write(
                            row, src, expected_raw_fingerprint)
                        if ok and generation_ok:
                            row.watermark_state = 'cleaned'
                            row.watermark_clean_method = 'klein'
                            _invalidate_effective_analysis(row)
                            counts['klein'] += 1
                        else:
                            _discard_clean_blob(bank_id, row)
                            counts['skipped' if (err or {}).get('kind') == 'unavailable'
                                   else 'failed'] += 1
                            error = err or error
                        db.session.commit()
                        bank_jobs.bump(job)
                        continue
                    pending.append((rid, dst, [list(b) for b in boxes], src,
                                    expected_raw_fingerprint))
                    bank_jobs.bump(job)
                if pending and bank_jobs.cancelled(job):
                    # Stop means stop: the staged copies of rows we never got to
                    # repaint are thrown away rather than running a long batch
                    # after the user asked out (they stay 'detected', retryable).
                    for pid, _dst, _boxes, _src, _fingerprint in pending:
                        _drop_clean_blob_by_id(bank_id, pid)
                    pending = []
                if pending:
                    results = watermark_lama.inpaint_batch(
                        [{'image_path': str(dst), 'bboxes': boxes}
                         for _rid, dst, boxes, _src, _fingerprint in pending],
                        device=device)
                    for pid, dst, _boxes, src, expected_raw_fingerprint in pending:
                        row = _live_image(pid)
                        if row is None:
                            # Deleted while the batch ran: no row is left to point
                            # at the repainted copy, so throw the copy away too.
                            logger.info('bank inpaint: image %s was deleted while the '
                                        'batch ran, skipping it', pid)
                            _drop_clean_blob_by_id(bank_id, pid)
                            counts['vanished'] += 1
                            continue
                        ok, err = results.get(str(dst), (
                            False, {'kind': 'failed', 'detail': 'missing inpaint result'}))
                        generation_ok = _prepare_watermark_write(
                            row, src, expected_raw_fingerprint)
                        if ok and generation_ok:
                            row.watermark_state = 'cleaned'
                            row.watermark_clean_method = 'lama'
                            _invalidate_effective_analysis(row)
                            counts['inpainted'] += 1
                        else:
                            _discard_clean_blob(bank_id, row)
                            counts['skipped' if (err or {}).get('kind') == 'unavailable'
                                   else 'failed'] += 1
                            error = err or error
        finally:
            db.session.commit()
            if counts['inpainted'] or counts['klein']:
                reset_score_memo()
        done = counts['inpainted'] + counts['klein']
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {done} inpainted so far')
            return
        detail = f'done — {done} inpainted'
        if counts['review']:
            detail += (f", {counts['review']} on the subject "
                       '(switch the engine to Klein to repaint those)')
        if counts['empty']:
            # NOT lumped in with 'review': "on the subject" would send the user to
            # Klein for images where the honest answer is "your mask is empty".
            detail += (f", {counts['empty']} with an empty mask (draw a zone in "
                       '▶ Review, or dismiss them)')
        if counts['vanished']:
            detail += (f", {counts['vanished']} skipped "
                       '(deleted while the pass ran)')
        if counts['skipped']:
            detail += f", {counts['skipped']} skipped (engine unavailable)"
        if counts['failed']:
            detail += f", {counts['failed']} failed"
            if error and error.get('detail'):
                detail += f" — {error['detail']}"
        # What the SCOPE never reached, named. Silent when it reached everything,
        # and silent on a selection — there the user pointed at the images.
        detail += _scope_note(bank_id, _clean_todo_clause(), statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


@_serialized_bank_mutation('watermark_regions')
def set_watermark_regions(user_id, bank_id, image_id, regions, *,
                          _bank_lease=None) -> dict | None:
    """Replace one image's hand-drawn watermark mask — and, when the row was not
    flagged, make the drawn zone the flag itself (reported missing in the Bank by
    Qeeyana on Reddit; the miss case by vvilams on Discord).

    ``regions`` is None (drop the override, go back to the detected box) or a list
    of normalized boxes — validated by the DATASET's validator, deliberately: one
    definition of a legal mask means the two lanes cannot drift apart. Returns the
    same payload shape the dataset route returns, None when the bank/image is
    unknown, ValueError on an illegal mask.

    Only 'cleaned' still refuses: those pixels were already replaced, so geometry
    drawn now describes an image that no longer exists (↩ Undo is the way back).
    'dismissed' does NOT refuse — it means the machine must stop re-flagging the
    image, never that the user cannot draw the mark after a second look. The
    detector is a classifier, and a mark it scores under the threshold used to
    have no manual recourse at all, because the one tool for it was reachable
    only from images the detector had already found.

    The mask is NOT cleared when a clean succeeds, unlike the dataset: the Bank's
    ↩ Undo is a first-class action (it only deletes our own blob), and handing an
    image back with its hand-drawn zones erased would mean redrawing them."""
    if not get_bank(user_id, bank_id):
        return None
    owned = BankImage.query.filter_by(id=image_id, bank_id=bank_id)
    row = owned.one_or_none()
    if not row:
        return None
    if row.watermark_state == 'cleaned':
        raise RuntimeError('this image was already cleaned — undo the cleaning '
                           'before masking it again')
    was_detected = row.watermark_state == 'detected'
    normalized = normalize_watermark_regions(regions)
    import json as _json
    stored = _json.dumps(normalized) if normalized is not None else None
    bank = db.session.get(ImageBank, bank_id)
    raw_path = abs_image_path(bank, row) if bank else None
    # A row no scan ever reached carries no attestation. That is the FIRST one,
    # not a changed file: measure it now and let _prepare_watermark_write bind it
    # (its `watermark_fingerprint is None` branch). Passing the stored NULL would
    # fail the validity gate and read as "the source image changed" — a refusal
    # that would keep this dead end shut for anyone who never ran the pass.
    expected_raw_fingerprint = (
        row.watermark_fingerprint
        or bank_transfer_metadata.content_fingerprint_path(raw_path))
    if not _prepare_watermark_write(
            row, raw_path, expected_raw_fingerprint):
        # Keep the fail-closed invalidation performed by the authority check.
        # Rolling it back would leave stale geometry active after this request
        # has proved that the source bytes changed.
        db.session.commit()
        raise RuntimeError('the source image changed — scan it again before masking')
    row.watermark_state = 'detected'
    row.watermark_regions = stored
    if not was_detected:
        # Say WHO flagged it. Editing the box of a row the detector did find
        # leaves its verdict — and its provenance — alone.
        row.watermark_source = 'manual'
    db.session.commit()
    return _watermark_regions_payload(row)


@_serialized_bank_mutation('watermark_undo')
def undo_watermark_clean(user_id, bank_id, image_ids=None, *,
                         _bank_lease=None) -> int:
    """Throw away cleaned versions and re-flag the images. The source was never
    modified, so undoing is just deleting our own blob — which is exactly what
    makes running both levels risk-free. ``image_ids`` empty = every cleaned
    image of the bank. Returns how many rows were restored.

    A bank-side ✂ crop / ✨ improve made on top of a cleaned image goes with it.
    It has to: that blob was produced FROM the cleaned pixels, so keeping it
    would leave an image the user is told is no longer cleaned while the mark it
    is being restored to is still gone."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    q = BankImage.query.filter_by(bank_id=bank_id, watermark_state='cleaned')
    if image_ids:
        ids = [int(i) for i in image_ids]
        q = q.filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK]))
    rows = q.all()
    restored = 0
    for row in rows:
        raw_path = abs_image_path(bank, row)
        expected_raw_fingerprint = row.watermark_fingerprint
        if not _prepare_watermark_write(
                row, raw_path, expected_raw_fingerprint):
            continue
        _drop_edited_blob(bank_id, row)
        _discard_clean_blob(bank_id, row)
        _invalidate_effective_analysis(row)
        # Back to 'detected' with its bbox intact, so it re-enters both levels
        # (e.g. to retry with the other engine).
        row.watermark_state = 'detected'
        restored += 1
    if rows:
        db.session.commit()
        if restored:
            reset_score_memo()
    return restored


@_serialized_bank_mutation('watermark_dismiss')
def dismiss_watermarks(user_id, bank_id, image_ids, *,
                       _bank_lease=None) -> int:
    """Rule a flag a FALSE positive: 'detected' → 'dismissed'. Those images leave
    both cleaning levels and are never re-flagged by a later scan — without this,
    level 2 would happily repaint a legitimate logo on a T-shirt. Mirrors the
    dataset's dismiss_watermarks. Returns how many rows changed."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ids = [int(i) for i in (image_ids or [])]
    if not ids:
        return 0
    rows = (BankImage.query
            .filter_by(bank_id=bank_id, watermark_state='detected')
            .filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK])).all())
    dismissed = 0
    for row in rows:
        raw_path = abs_image_path(bank, row)
        expected_raw_fingerprint = row.watermark_fingerprint
        if not _prepare_watermark_write(
                row, raw_path, expected_raw_fingerprint):
            continue
        row.watermark_state = 'dismissed'
        dismissed += 1
    if rows:
        db.session.commit()
    return dismissed


def _watermark_source_counts(bank_id) -> dict:
    """How many SCANNED rows each detection route produced. A row with a state but
    no source predates the column — counted as 'unknown', never attributed."""
    scanned = BankImage.query.filter_by(bank_id=bank_id).filter(
        BankImage.watermark_state.isnot(None),
        ~_watermark_history_inactive_clause())
    detector = scanned.filter(BankImage.watermark_source == 'detector').count()
    vision = scanned.filter(BankImage.watermark_source == 'vision').count()
    return {'detector': detector, 'vision': vision,
            'unknown': scanned.count() - detector - vision}


def _detector_ready() -> bool:
    """Whether the NEXT run will use the detector — which is now the resolved
    SETTING, not merely "is the extra installed". The panel's sentence says "a new
    run uses …", so reading availability alone would make it lie the moment a user
    pinned the vision model on a machine that has the extra.

    Never raises: a probe that explodes must not 500 the whole panel — it just
    means the next run is the vision model, which is the shipped behaviour."""
    try:
        from . import watermark_detector
        return watermark_detector.resolve_backend()['backend'] == 'detector'
    except Exception:      # noqa: BLE001
        return False


def _detector_threshold() -> float:
    try:
        from . import watermark_detector
        return watermark_detector.threshold()
    except Exception:      # noqa: BLE001
        return 0.94


def watermark_levels(user_id, bank_id) -> dict | None:
    """Where each cleaning level stands — the numbers the UI shows per level.
    None when the bank is gone."""
    if not get_bank(user_id, bank_id):
        return None
    from .face_dataset_service import _route_watermark
    bank = db.session.get(ImageBank, bank_id)
    base = BankImage.query.filter_by(bank_id=bank_id)
    flagged = croppable = hand_masked = empty_masks = 0
    for row in _clean_pool_query(bank_id).all():
        flagged += 1
        boxes, manual, _problem = _clean_regions(row)
        if manual:
            # A hand mask never routes to the crop level (see _watermark_crop_job),
            # so it must not be counted as croppable — the ✂ button would offer
            # work it will then skip.
            hand_masked += 1
            if not boxes:
                empty_masks += 1
            continue
        # Stored scan dimensions are raw camera-raster dimensions. Watermark
        # boxes instead come from the upright vision/browser frame, so this
        # tally intentionally re-reads the inexpensive header through
        # `_source_size` rather than incorrectly trusting row.width/height.
        _path, width, height = _source_size(bank, row)
        if boxes and width and _route_watermark(boxes[0], width, height,
                                                allow_crop=True)[0] == 'crop':
            croppable += 1
    return {
        'scanned': base.filter(
            BankImage.watermark_state.isnot(None),
            ~_watermark_history_inactive_clause()).count(),
        # What a plain re-run would still look at. Detection resumes where it
        # stopped (the pass commits every 25 rows), but nothing said so: a
        # progress bar that restarts at 0 each run reads as "it started over
        # and is re-analysing what I already did". This is the number that
        # answers that, so the panel can say "N left to scan" out loud.
        'unscanned': _watermark_scan_query(bank_id, rescan=False).count(),
        'flagged': flagged,
        'croppable': croppable,
        'inpaintable': flagged - croppable,
        # Flagged images whose mask the user drew by hand, and how many of those
        # were deliberately emptied. Both are surfaced: an empty mask repaints
        # nothing, and a level that silently skips images is how a user ends up
        # believing a watermark was removed when it was not.
        'hand_masked': hand_masked,
        'empty_masks': empty_masks,
        'cropped': base.filter_by(watermark_clean_method='crop').count(),
        'inpainted': base.filter(
            BankImage.watermark_clean_method.in_(('lama', 'klein'))).count(),
        'dismissed': base.filter(
            BankImage.watermark_state == 'dismissed',
            ~_watermark_history_inactive_clause()).count(),
        'needs_rescan': _needs_rescan_count(bank_id),
        # WHO ruled on the images already scanned. Surfaced because the two
        # routes are not the same instrument and a bank scanned over weeks can
        # hold both: 'detector' rows carry a score and a threshold you can
        # change, 'vision' rows carry a sentence a model wrote. 'unknown' is
        # every row scanned before this was recorded — named rather than
        # attributed, because guessing which one produced them would be a lie.
        'sources': _watermark_source_counts(bank_id),
        # Which route the NEXT run would take, and (when it is the detector) the
        # score it will flag at. The panel quotes these instead of keeping its own
        # copy of the default.
        'next_source': ('detector' if _detector_ready() else 'vision'),
        'threshold': _detector_threshold(),
        # A few already-cleaned ids so the panel can offer a before/after strip
        # (each image is served cleaned, or original with ?original=1) without a
        # second endpoint just to list them.
        'cleaned_sample': [r.id for r in
                           base.filter(BankImage.watermark_clean_method.isnot(None))
                           .order_by(BankImage.id.asc()).limit(8).all()],
        # 🔤 Find text — the OCR pass's own tallies, next to the levels it
        # feeds. `found` rows are also counted in `flagged` above (their zones
        # went into the same funnel); this block is what lets the panel say the
        # pass's own progress and label its button, nothing routes on it.
        'text': {
            'scanned': base.filter(BankImage.text_state.isnot(None)).count(),
            'found': base.filter(BankImage.text_state == 'detected').count(),
            'unscanned': _text_scan_query(bank_id, rescan=False).count(),
        },
    }


# --- framing pass (reuses the dataset face/bust/body/back classifier) -------
def _framing_pool(bank_id, rescan, statuses=None, ids=None):
    """The rows the framing pass will classify.

    ONE definition, called by the launch (to price the run) and by the job (to do
    it). It used to be written out twice, identically — which is exactly the
    structure that lets a button announce a number it does not act on the moment
    somebody edits one copy."""
    q = _scoped_pool(bank_id, statuses, ids)
    if not rescan:
        q = q.filter(BankImage.framing.is_(None))
    return q


def start_framing(app, user_id, bank_id, rescan=False, statuses=None, ids=None):
    """Classify every non-rejected image by SHOT TYPE (face / bust / body / back),
    reusing the SAME Qwen3-VL classifier the datasets use (CLASSIFY_PROMPT). Feeds
    the 📐 Framing filter chips and the coverage advice. Needs the vision model
    pulled; serialized against training/vision like the watermark pass (503 when
    the GPU is held). ``rescan`` re-classifies rows that already have a framing."""
    from ..capabilities import probe_ollama_model
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    # Occupancy BEFORE the model probe, and the order is the whole point: a bank
    # that is already busy is busy whether or not Ollama answers. Probing first
    # made a busy bank report "the vision model is not available" whenever the
    # model happened to be unreachable -- the wrong reason, sending the user to
    # fix an unrelated thing while the real answer was "wait, a pass is running".
    # It also made the refusal depend on an EXTERNAL SERVICE, which is why CI
    # (no Ollama) failed a release on it while two agents read it as a flake.
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy((bank_jobs.get(bank_id) or {}).get('kind') or 'background')
    if not probe_ollama_model().get('ok'):
        raise RuntimeError('the vision model is not available '
                           '(Settings ▸ Local tools)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    want = normalize_pass_statuses(statuses)
    return bank_jobs.start(app, bank_id, 'framing',
                           _framing_job(bank_id, rescan, want, ids),
                           total=_framing_pool(bank_id, rescan, want, ids).count())


# How many images in a row may fail to reach the vision model before the pass
# says so ON SCREEN instead of at the end. Five is short enough to appear within
# a few seconds of a real outage and long enough that a couple of unlucky images
# never trip it.
_FENCE_STREAK_WARN = 5


def _framing_job(bank_id, rescan, statuses=None, ids=None):
    def run(job):
        from .face_dataset_service import CLASSIFY_PROMPT, _parse_classify
        from .vision_ollama import (LocalOllamaFenceError, describe_image_ollama,
                                    unload_vision_model)
        from .vision_pool import map_vision
        from ..gpu_window import gpu_exclusive_vision_window
        bank = _detach_bank(db.session.get(ImageBank, bank_id))
        if not bank:
            return
        rows = (_framing_pool(bank_id, rescan, statuses, ids)
                .order_by(BankImage.id.asc()).all())
        bank_jobs.progress(job, done=0, total=len(rows), detail='framing')
        todo_clause = None if rescan else BankImage.framing.is_(None)
        if not rows:
            # A pass that ran over nothing used to return MUTE, so the screen
            # showed the previous pass's line and the click read as ignored.
            bank_jobs.progress(job, detail=(
                'done — nothing to classify in this scope'
                + _scope_note(bank_id, todo_clause, statuses, ids)))
            return
        row_ids = [r.id for r in rows]
        classified = errors = vanished = stale = fenced = 0
        missing = unanswered = 0
        fence_streak = 0

        def prepared():
            """Path resolution reads the row, so it belongs on the job's own
            thread — pulled one image per free slot in the pool."""
            nonlocal vanished
            for rid in row_ids:
                # no_autoflush, for the reason spelled out in the quality scan
                # (_scan_job) and paid for here in lost work rather than in
                # latency: this re-read is a SELECT, SQLAlchemy flushes before
                # one, and that flush takes the single SQLite write lock at the
                # first image after a mutation. This generator is pulled BEFORE
                # each model call, so the lock would then be held THROUGH the
                # call — see the loop below for what starves behind it.
                with db.session.no_autoflush:
                    row = _live_image(rid)
                    if row is None:  # deleted since the pass started — _live_image
                        logger.info('bank framing pass: image %s was deleted mid-pass, '
                                    'skipping it', rid)
                        vanished += 1
                        bank_jobs.bump(job)
                        continue
                    path = analysis_image_path(bank, row, refresh_rotation=True)
                yield rid, path

        def ask(item):
            """WORKER thread: file + network only, no session. None means the
            file is gone, as opposed to '' meaning the model said nothing."""
            _rid, path = item
            if not path or not os.path.isfile(path):
                return None
            payload = _read_safe_bank_source_bytes(path, label='bank framing scan')
            fingerprint = bank_transfer_metadata.content_fingerprint_bytes(payload)
            try:
                raw = describe_image_ollama(
                    payload, CLASSIFY_PROMPT, num_predict=400,
                    prefer_json=True, fmt='json', keep_alive='5m')
                return {'raw': raw, 'fingerprint': fingerprint, 'error': None,
                        'fenced': False}
            except Exception as exc:  # noqa: BLE001 — one image, not the pass
                # A fence refusal is kept apart from a bad file: the image was
                # never shown to the model, so it is neither unreadable nor
                # classified, and only a re-run fixes it (see _skipped_note).
                return {'raw': None, 'fingerprint': fingerprint,
                        'error': f'{type(exc).__name__}: {exc}',
                        'fenced': isinstance(exc, LocalOllamaFenceError)}

        with gpu_exclusive_vision_window(flag_ttl=1800):
            try:
                # The calls overlap (see vision_pool); every write below still
                # happens here, on this one thread.
                for (rid, _path), answer, error in map_vision(
                        prepared(), ask,
                        should_cancel=lambda: bank_jobs.cancelled(job)):
                    # Re-read: the classification we are about to store took a
                    # model call to arrive, and the image can be gone by now.
                    # no_autoflush for the same reason as in prepared() above.
                    with db.session.no_autoflush:
                        row = _live_image(rid)
                    if row is None:
                        logger.info('bank framing pass: image %s was deleted while it '
                                    'was being classified, skipping it', rid)
                        vanished += 1
                        bank_jobs.bump(job)
                        continue
                    answer = answer if isinstance(answer, dict) else {}
                    fingerprint = answer.get('fingerprint')
                    raw = answer.get('raw')
                    call_error = error or answer.get('error')
                    if answer.get('fenced'):
                        fence_streak += 1
                        if fence_streak >= _FENCE_STREAK_WARN:
                            # WHILE it happens, not only in the last line: a
                            # storm of these means the vision GPU window cannot
                            # be renewed at all (it is what a `database is
                            # locked` on system_state looks like from here), and
                            # every remaining image will fail the same way. The
                            # user watching the bar advance has to be able to
                            # stop and fix Ollama instead of paying for a pass
                            # that classifies nothing.
                            bank_jobs.progress(job, detail=(
                                f'framing — {fence_streak} images in a row could not '
                                'reach the vision model (the GPU window keeps '
                                'expiring). They stay unclassified and a re-run '
                                'finishes them — stop the pass if you want to look '
                                'at Ollama first.'))
                    elif fence_streak:
                        fence_streak = 0
                        bank_jobs.progress(job, detail='framing')
                    if ((call_error is not None or raw is not None)
                            and not _prepare_analysis_write(
                                row, _path, fingerprint)):
                        # The bytes moved between the model call and this write,
                        # so the answer describes an image that no longer exists
                        # here. Nothing is stored — and it is COUNTED, because a
                        # pass that reached 204 images and reported "12
                        # classified" explained nothing about the other 192.
                        stale += 1
                        bank_jobs.bump(job)
                        # This branch mutates too — _prepare_analysis_write
                        # invalidates the lanes it refused — so it owes the same
                        # per-image commit as the bottom of the loop. Skipping it
                        # would carry a pending write into the next model call.
                        db.session.commit()
                        continue
                    if call_error is not None:  # one bad file never sinks the pass
                        if answer.get('fenced'):
                            fenced += 1
                        else:
                            errors += 1
                    elif raw is None:      # file gone: leave the row as it was
                        missing += 1
                    # Empty output = Ollama unreachable, NOT "unknown": leave the
                    # framing NULL so a retry can finish it (same reasoning as the
                    # watermark/dataset classifier), never mislabel everything.
                    elif not raw.strip():
                        # COUNTED, exactly like the watermark pass counts it: a
                        # run where every answer came back empty said "done — 0
                        # classified", which reads as "looked at them all", when
                        # in truth not one image was looked at.
                        unanswered += 1
                    else:
                        framing, _label = _parse_classify(raw)
                        row.framing = framing        # face|bust|body|back|unknown
                        classified += 1
                    bank_jobs.bump(job)
                    # Per image, exactly like the watermark pass — and here the
                    # old rhythm ("commit every 25 CLASSIFIED images") was worse
                    # than slow, it was self-defeating. That counter only moves
                    # on a SUCCESS, so a pass whose answers erroured never
                    # committed at all; meanwhile every image costs an Ollama
                    # call (~1.7 s, measured at 5 s on the install that reported
                    # this), so the single SQLite write lock was parked for up
                    # to two minutes at a time. The writer starved behind it is
                    # not a sort click: `_set_system_state('vision_in_progress')`
                    # renews the vision GPU window before EVERY call, dies on
                    # `database is locked` past the 5 s busy_timeout, and
                    # _admit_local_ollama is fail-closed — so the pass locked
                    # out its own permission to work and fenced its remaining
                    # images out of the model (4 classified out of 12 on a
                    # 211-image bank, reported 2026-08-07).
                    # Cost of committing per image instead, measured on a WAL
                    # database over 500 rows: 0.17 ms median, 0.22 ms p90, 0.10 s
                    # in total — against model calls counted in seconds. The
                    # hold is now bounded by ONE image whatever the answers look
                    # like, which is the property, not the batch size.
                    db.session.commit()
            finally:
                db.session.commit()
                unload_vision_model()  # hand the VRAM back to ComfyUI
        skipped = _skipped_note(vanished=vanished, missing=missing,
                                unanswered=unanswered, fenced=fenced,
                                stale=stale, unreadable=errors)
        if bank_jobs.cancelled(job):
            bank_jobs.progress(
                job, detail=f'cancelled — {classified} classified so far' + skipped)
            return
        detail = f'done — {classified} classified' + skipped
        detail += _scope_note(bank_id, todo_clause, statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


# --- 🎨 medium pass (zero-shot CLIP over the ✨ Score embeddings) ------------
def medium_prereq(bank_id=None) -> str | None:
    """Why 🎨 Medium cannot run here, or None. Two distinct refusals, and telling
    them apart is the point: a bank with no ✨ Score embeddings needs a pass run,
    while a machine that cannot encode text needs a Setup step. Answering either
    with the other sends the user to fix the wrong thing."""
    from . import clip_text_encoder
    if bank_id is not None:
        scored = (BankImage.query.filter_by(bank_id=bank_id)
                  .filter(or_(BankImage.aesthetic_score.isnot(None),
                              BankImage.nsfw_score.isnot(None))).count())
        if not scored:
            return ('run ✨ Score first — 🎨 Medium reads the embeddings it '
                    'computes, and computes none of its own')
    return clip_text_encoder.unavailable_reason()


def _medium_pool(bank_id, rescan, statuses=None, ids=None):
    """The rows 🎨 Medium will classify — ONE definition for the launch, the job
    and the counter, for the same reason as _framing_pool."""
    q = _scoped_pool(bank_id, statuses, ids)
    if not rescan:
        q = q.filter(BankImage.medium.is_(None))
    return q


def start_medium(app, user_id, bank_id, rescan=False, statuses=None, ids=None):
    """Classify every SCORED image by MEDIUM (photograph / anime / 3D render /
    illustration, or an honest 'unsure') from the CLIP embedding the ✨ Score pass
    already cached.

    NO NEW IMAGE INFERENCE, by construction: the only model call this pass makes
    is encoding ~30 short phrases through CLIP's text tower, once per install
    (clip_text_encoder caches every phrase on disk, forever, across restarts), on
    the CPU. Everything after that is a (n × 768) @ (768 × 6) matrix product in
    the Flask process. An image the ✨ Score pass never reached has no embedding
    and stays NULL — "not scored yet", which is neither a medium nor an 'unsure'.

    ``rescan`` re-classifies rows that already carry a verdict; without it the
    pass finishes the job (rows with medium IS NULL), so it can be re-run after a
    ✨ Score that added images without redoing the whole bank.

    Never takes the GPU window and is not refused when the GPU is busy: it does
    not touch the card, and blocking it behind a training run would be a refusal
    with no cause."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if bank_jobs.running(bank_id):
        raise bank_jobs.BankJobBusy((bank_jobs.get(bank_id) or {}).get('kind') or 'background')
    reason = medium_prereq(bank_id)
    if reason:
        raise RuntimeError(reason)
    want = normalize_pass_statuses(statuses)
    return bank_jobs.start(app, bank_id, 'medium',
                           _medium_job(bank_id, rescan, want, ids),
                           total=_medium_pool(bank_id, rescan, want, ids).count())


def _medium_prototype_matrix():
    """(bucket_names, P) — one L2-normed row per bucket, each the MEAN of its
    phrases' text vectors (prompt ensembling). Raises clip_text_encoder's
    TextEncodeError, which the route turns into an announced 503."""
    import numpy as np
    from . import clip_text_encoder
    names, rows = [], []
    for name, phrases in MEDIUM_PROTOTYPES.items():
        vecs = [clip_text_encoder.encode_query(p)[0] for p in phrases]
        m = np.mean(np.stack(vecs).astype('float32'), axis=0)
        rows.append(m / (float(np.linalg.norm(m)) + 1e-8))
        names.append(name)
    return names, np.stack(rows).astype('float32')


def _medium_job(bank_id, rescan, statuses=None, ids=None):
    def run(job):
        import numpy as np
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        bank_jobs.progress(job, done=0, detail='medium pass — encoding prototypes')
        names, P = _medium_prototype_matrix()
        emb_by_path = _load_score_embeddings(bank)
        rows = (_medium_pool(bank_id, rescan, statuses, ids)
                .order_by(BankImage.id.asc()).all())
        bank_jobs.progress(job, done=0, total=len(rows), detail='medium pass')
        base = os.path.realpath(bank.source_path)
        prefix = os.path.normcase(base + os.sep)
        done = classified = unscored = stale = 0
        for r in rows:
            if bank_jobs.cancelled(job):
                break
            # Same two-step path resolution as _pool_embeddings: a lexical
            # normpath that HITS the cache is provably what realpath would have
            # returned, and realpath is a syscall this loop cannot afford once
            # per row on a 20 000-image bank.
            if r.watermark_clean_method or r.rotation:
                p = analysis_image_path(bank, r)
                emb = emb_by_path.get(p) if p else None
            else:
                p = os.path.normpath(os.path.join(base, r.relpath))
                emb = (emb_by_path.get(p)
                       if os.path.normcase(p).startswith(prefix) else None)
                if emb is None:
                    p2 = _abs_under(base, r.relpath)
                    emb = emb_by_path.get(p2) if p2 else None
                    if emb is not None:
                        p = p2
            if emb is None:
                unscored += 1          # no embedding → stays NULL, honestly
            else:
                fingerprint = _score_embedding_fingerprint(p)
                if not _prepare_analysis_write(r, p, fingerprint):
                    # An embedding EXISTS here — it just describes bytes this
                    # file no longer has. Counting it as "not scored yet" sent
                    # the user to re-run ✨ Score, which is not what fixes it.
                    stale += 1
                    done += 1
                    bank_jobs.bump(job)
                    continue
                e = np.asarray(emb, dtype='float32')
                e = e / (float(np.linalg.norm(e)) + 1e-8)
                sims = P @ e
                verdict, margin = medium_verdict(dict(zip(names, sims)))
                r.medium = verdict
                r.medium_margin = round(float(margin), 5) if margin is not None else None
                classified += 1
            done += 1
            bank_jobs.bump(job)
            if done % 500 == 0:
                db.session.commit()
        db.session.commit()
        skipped = ''
        if unscored:
            # Never silent: an image with no ✨ Score embedding CANNOT get a
            # medium, and "0 results" without that sentence reads as a bug.
            skipped += f', {unscored} skipped (not scored yet)'
        skipped += _skipped_note(stale=stale)
        detail = f'done — {classified} classified' + skipped
        detail += _scope_note(bank_id,
                              None if rescan else BankImage.medium.is_(None),
                              statuses, ids)
        if bank_jobs.cancelled(job):
            detail = (f'stopped — {classified} classified, '
                      f'{len(rows) - done} left (re-run to finish)' + skipped)
        bank_jobs.progress(job, detail=detail)
    return run


# --- caption pass (reuses the dataset caption engines) ----------------------
# The statuses a caption pass may be AIMED at. 'reject' is absent on purpose and
# it is not an oversight: you curate from what you might keep, never from the bin
# — the same rule every other pass on this page follows. Values are the ones
# stored in the column ('keep' / 'pending'); the UI says "Kept" and "Undecided".
# Renaming either would break stored queries, so the wire keeps the column's
# vocabulary and the translation happens in the label.
# 'reject' JOINED THIS TUPLE, and it is a change of principle, held on purpose.
# The bin used to be unreachable ("you curate from what you might keep, never
# from what you threw away") and that is still the right DEFAULT — it is not in
# the default scope, and a request without `statuses` never touches it. But a
# rejected image is not a deleted one: people un-reject, promote out of the bin,
# and search by caption across the whole bank. Refusing to caption it was a rule
# the app enforced on the user rather than a fact about the data, and the launch
# dialog is where the cost of aiming a GPU pass at the bin can finally be stated
# instead of assumed.
# CAPTION_SCOPES comes from pass_scopes.py with PASS_SCOPES (same tuple, two names).


def _normalize_caption_statuses(statuses):
    """Validate a per-run caption scope, or None for the historical set.

    None / [] → None, which means the pass keeps its original
    ``status != 'reject'`` filter: a client that sends no scope is
    byte-identical to before this option existed. Anything outside
    CAPTION_SCOPES raises ValueError → 400, exactly like a bad vocabulary."""
    return normalize_pass_statuses(statuses, CAPTION_SCOPES)


def _caption_name_option(name, value):
    """Normalize one free-text caption option, or raise ValueError → 400.

    Every one of these options is a NAME out of a closed list. A non-string reached
    ``.strip()`` and answered 500 — a bad request rendered as a broken server, while
    ``statuses`` next door already answered 400 for exactly the same mistake. The
    validation now matches the promise the route's docstring makes ("invalid → 400")."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'invalid caption {name}: expected a name, not '
                         f'{type(value).__name__}')
    return value.strip().lower() or None


def _caption_scope_q(bank_id, statuses):
    """The rows a caption pass may touch, for this bank and this scope.

    ONE definition, called by the launch (to price the run) and by the job (to do
    it). Two copies of this filter is precisely how a button comes to announce a
    number it does not act on."""
    return _scoped_pool(bank_id, statuses)


def start_caption(app, user_id, bank_id, ids=None, force=False, vocabulary=None,
                  length=None, backend=None, ollama_model=None, statuses=None,
                  include_asserted=False):
    """Launch the caption pass over a selection (``ids``) or, when empty, every
    non-rejected readable image. Reuses the dataset caption engines (JoyCaption /
    Ollama per Settings) through a dataset-free descriptive brick; the captions
    double as the bank's search text and ride along on promotion. Serialized
    against training/vision like the score/watermark passes (503 when the GPU is
    held). BankJobBusy when a job is already live, ValueError on a bad bank/config.

    ``vocabulary`` picks a caption REGISTER (one of face_dataset_service's
    CAPTION_VOCABULARIES: 'explicit' | 'clinical' | 'safe') — the SAME lane the
    dataset caption uses, appended as an instruction. Explicit only spells sexual
    content out when the backend runs an abliterated Ollama model; the choice rides
    per-call (the UI passes it), so a call WITHOUT it is byte-identical to before
    (no instruction appended). Richer captions also mean richer 🔍 search text.

    ``length`` picks the SIZE preset (one of CAPTION_LENGTHS: 'concise' | 'detailed';
    None/'' = standard, nothing appended) — an axis orthogonal to the register, again
    the same lane and the same text as the dataset caption.

    ``backend`` overrides ``captioning.backend`` for THIS run only (one of
    CAPTION_BACKENDS) and ``ollama_model`` overrides ``ollama.vision_model`` for THIS
    run only — the global settings stay the default and are never written. Which model
    writes a caption is not a matter of taste: a captioner that describes what it sees
    in evasive terms produces captions that are about something slightly other than the
    images, and a LoRA trained on them learns to look away too, with nothing in the
    output to reveal it (measured on the video lane, commit "the model that writes the
    captions stops being a decision we made"). Both empty → the global settings, so a
    call without them is byte-identical to before. 'auto' is left alone deliberately:
    it CHAINS JoyCaption then Ollama on what JoyCaption missed, so forcing 'joycaption'
    removes the Ollama half rather than "picking one of two".

    ``statuses`` picks the SCOPE of the run: any combination of CAPTION_SCOPES
    ('keep' | 'pending' | 'reject'). None = the historical non-rejected set,
    byte-identical to before. ['reject'] aims the pass AT THE BIN, which is never a
    default and never part of one — it is reachable only because the caller asked for
    it by name. When ``ids`` are also given the two INTERSECT (the selection is
    narrowed by the scope, never widened).

    ``force`` rewrites captions that already exist — and, since captions carry an
    origin (services/caption_origin.py), it now SPARES the ones a human wrote or
    corrected, the way the embeddings pass spares an asserted face cluster. Rows
    whose origin was never recorded (every row that predates the column) are
    rewritten: their authorship cannot be recovered, and sparing them would make
    this button inert on every bank that exists today. ``include_asserted`` is the
    explicit way OUT of that protection — a separate opt-in, never a default,
    for the person who does want their own captions redone by a better model.
    It means nothing without ``force`` and is ignored there (an unforced pass
    only ever touches rows with no caption at all)."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    from .face_dataset_service import (CAPTION_BACKENDS, CAPTION_LENGTHS,
                                       CAPTION_VOCABULARIES)
    vocab = _caption_name_option('vocabulary', vocabulary)
    if vocab and vocab not in CAPTION_VOCABULARIES:
        raise ValueError(f'invalid caption vocabulary: {vocab}')
    size = _caption_name_option('length', length)
    if size and size not in CAPTION_LENGTHS:
        raise ValueError(f'invalid caption length: {size}')
    want = _normalize_caption_statuses(statuses)
    engine = _caption_name_option('backend', backend)
    if engine and engine not in CAPTION_BACKENDS:
        raise ValueError(f'invalid captioning backend: {engine}')
    # Same charset check the Caption Lab uses (never shelled out — it is a JSON
    # field to the local Ollama server), and the same allow_empty contract.
    from .ollama_control import normalize_ollama_model_ref
    model = normalize_ollama_model_ref(ollama_model or '', allow_empty=True) or None
    # The RESOLVED engine decides whether there is anything to run at all — a
    # per-run 'none' is refused exactly like the global one, and a per-run engine
    # rescues a run on an install whose global setting is 'none'.
    resolved = engine or (cfg.get('captioning.backend') or 'auto').lower()
    if resolved == 'none':
        raise ValueError('no captioning backend configured (Settings ▸ Captioning & quality)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    ids = [int(i) for i in ids] if ids else None
    q = _caption_scope_q(bank_id, want)
    if ids is not None:
        q = q.filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK]))
    keep_asserted = bool(force) and not include_asserted
    if not force:
        q = q.filter(or_(BankImage.caption.is_(None), BankImage.caption == ''))
    elif keep_asserted:
        # The protection, in the SAME query that prices the run — so the number
        # the button quotes is the number the job walks. Two definitions of this
        # filter is exactly how a button comes to announce work it does not do.
        q = q.filter(caption_origin.unprotected_clause(BankImage))
    total = q.count()
    return bank_jobs.start(app, bank_id, 'caption',
                           _caption_job(bank_id, ids, force, vocab, size,
                                        backend=engine, ollama_model=model,
                                        statuses=want,
                                        keep_asserted=keep_asserted),
                           total=total)


# --- Scene captions -------------------------------------------------------
# A bank of reference images read as a source of MISE EN SCENE: every row that
# already carries a caption becomes one ordered scene, and a generation panel
# can run those captions IN ORDER with the user's own LoRA supplying the
# character. A READ, never a pass — no GPU, no writes, alive while a job runs.

# The shape of a scene card — ceiling, cut, framing fallback, label — lives in
# services/scene_captions.py, because a DATASET offers the very same cards from
# its own captions. One definition, so the two surfaces cannot answer differently
# for the same caption (test_scene_caption_parity.py reads both against it).
# Re-exported under the historical names: callers and tests already say
# `banks.SCENE_MAX_PROMPT`.
from .scene_captions import SCENE_MAX_PROMPT  # noqa: E402,F401  re-exported as banks.SCENE_MAX_PROMPT
from .scene_captions import (            # noqa: E402  (module-level, grouped with its section)
    scene_framing as _scene_framing,
    scene_label as _scene_label,
    scene_prompt as _scene_prompt,
)


def export_scene_captions(user_id, bank_id, statuses=None):
    """The bank's captions as ORDERED scene cards.

    One card per captioned image, in bank order (row id ascending = import
    order). Order is the point: each card is one beat of a sequence, so a
    missing framing is NOT a gate — the card rides the row's classified framing
    when there is one and 'body' otherwise, because refusing a page would
    silently drop a beat from the middle. Only a missing caption skips a row,
    and it is COUNTED, never guessed.

    Labels carry the sequence number ("Scene 3 — page_003.jpg") so the reading
    order stays visible wherever the card lands. ``image_id`` lets a UI show the
    page the scene came from (bank thumb route); it is display-only and never
    rides a generation payload.
    """
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    want = _normalize_caption_statuses(statuses)
    th = thresholds()
    rows = (_caption_scope_q(bank_id, want)
            .order_by(BankImage.id.asc()).all())
    scenes = []
    skipped = {'no_caption': 0}
    for row in rows:
        caption = (row.caption or '').strip()
        if not caption:
            skipped['no_caption'] += 1
            continue
        stem = os.path.basename(row.relpath or '') or f'image {row.id}'
        label = _scene_label(len(scenes), stem)
        nsfw = (row.nsfw_score is not None
                and row.nsfw_score > th['nsfw_max'])
        framing = _scene_framing(row.framing)
        scenes.append({'label': label, 'framing': framing,
                       'prompt': _scene_prompt(caption),
                       'image_id': row.id,
                       **({'nsfw': True} if nsfw else {})})
    return {'bank_id': bank_id, 'bank_name': bank.name,
            'scenes': scenes, 'skipped': skipped}


# The name each stored origin gets in a sentence. 'asserted' is absent ON PURPOSE:
# no pass ever writes it, and a pass that reported it would be reporting something
# it did not do. Keys are the stored values (services/caption_origin.py) — frozen.
_CAPTION_WRITER_NAMES = {
    caption_origin.JOYCAPTION: 'JoyCaption',
    caption_origin.OLLAMA: 'the Ollama vision model',
}


def _caption_writers_note(wrote, unrecorded=0):
    """" (340 by JoyCaption, 87 by the Ollama vision model)", or ''.

    PARENTHESES, not a dash: the detail line continues with comma-separated skip
    counts ("3 kept (written by you)", "1 skipped (image changed)"), and a
    dash-introduced list would let those read as more writers.

    WHY THE RUN HAS TO SAY THIS. The default backend is 'auto', which is a CHAIN:
    JoyCaption writes what it can, the Ollama vision model covers the rest. The two
    write differently and, on some material, one of them writes something that is
    not in the picture at all — so "427 captioned" describes two halves the user has
    no way to tell apart afterwards. These numbers come from the rows that were
    STAMPED, so they are what the run wrote and not what it was asked to write.

    Silence when a single engine wrote everything would be a smaller lie but still
    one — a one-engine run is exactly the thing worth confirming — so the note is
    emitted for one writer as well.  '' only when nothing was written at all.

    An engine name this build does not know is printed AS ITSELF rather than
    dropped: swallowing it would rebuild the blind spot this note exists to close.
    """
    parts = []
    for key in (caption_origin.JOYCAPTION, caption_origin.OLLAMA):
        n = int(wrote.get(key) or 0)
        if n:
            parts.append(f'{n} by {_CAPTION_WRITER_NAMES[key]}')
    for key in sorted(k for k in (wrote or {}) if k not in _CAPTION_WRITER_NAMES):
        n = int(wrote.get(key) or 0)
        if n:
            parts.append(f'{n} by {key}')
    if unrecorded:
        # NOT "by nobody": the engine did not report a name, so the row stores
        # NULL and the sentence says exactly that much and no more.
        parts.append(f'{int(unrecorded)} whose engine did not report a name')
    return f' ({", ".join(parts)})' if parts else ''


def _caption_job(bank_id, ids, force, vocabulary=None, length=None, *,
                 backend=None, ollama_model=None, statuses=None,
                 keep_asserted=False):
    def run(job):
        from .face_dataset_service import caption_paths, caption_preset_instructions
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        q = _caption_scope_q(bank_id, statuses)
        if ids is not None:
            rows = []
            for i0 in range(0, len(ids), _SQL_IN_CHUNK):
                rows.extend(q.filter(BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all())
            rows.sort(key=lambda r: r.id)
        else:
            rows = q.order_by(BankImage.id.asc()).all()
        skipped_asserted = 0
        if not force:
            rows = [r for r in rows if not (r.caption or '').strip()]
        elif keep_asserted:
            # Same rule as the launch filter, re-applied on the rows the job
            # actually loaded: a caption can be typed in the seconds between
            # pricing the run and starting it, and the newer word wins.
            spared = [r for r in rows if caption_origin.is_protected(r)]
            skipped_asserted = len(spared)
            rows = [r for r in rows if not caption_origin.is_protected(r)]
        by_path = {}
        for r in rows:
            p = analysis_image_path(bank, r, refresh_rotation=True)
            if _is_safe_bank_source(p, label='bank caption pass'):
                fingerprint = bank_transfer_metadata.content_fingerprint_path(p)
                if fingerprint is not None:
                    by_path[p] = (r.id, fingerprint)
        paths = list(by_path)
        bank_jobs.progress(job, done=0, total=len(paths), detail='captioning')
        if not paths:
            return
        captioned = vanished = stale = spared_mid_pass = 0
        # WHO ended up writing, counted from the rows that were actually STAMPED —
        # not from the pool, and not from the backend that was asked for. 'auto'
        # is a chain (JoyCaption, then Ollama on what it missed), so the requested
        # value would mislabel roughly half a bank, and the pool would count rows
        # the pass then skipped as stale/deleted/newer-caption-won. The same
        # mistake the flag counters were built to stop making.
        wrote = {}
        # None is a real outcome (an engine that reports no name) and it is kept
        # apart rather than folded into either engine — the column stores NULL for
        # it and reads as "never recorded", not as "machine".
        unrecorded_writes = 0

        def _on_caption(path, caption, engine=None):
            nonlocal captioned, vanished, stale, spared_mid_pass, unrecorded_writes
            planned = by_path.get(path)
            if planned is None:
                return           # a path this pass never asked about — not ours
            image_id, expected_fingerprint = planned
            row = _live_image(image_id)
            if row is None:      # deleted while it was being captioned — _live_image
                logger.info('bank caption pass: image %s was deleted mid-pass, '
                            'skipping its caption', image_id)
                vanished += 1
                return
            # Inference may take seconds. A tracked rotate/clean or an external
            # same-path replacement during that window must not attach the
            # caption computed from A to the now-current pixels B.
            current_path = analysis_image_path(bank, row, refresh_rotation=True)
            if (not _same_resolved_path(path, current_path)
                    or bank_transfer_metadata.content_fingerprint_path(current_path)
                    != expected_fingerprint):
                logger.info('bank caption pass: image %s changed mid-pass, '
                            'skipping its stale caption', image_id)
                stale += 1
                return
            # A caption written while this image was in inference wins unless
            # the user explicitly allowed overwriting asserted captions.
            if ((not force and (row.caption or '').strip())
                    or (keep_asserted and caption_origin.is_protected(row))):
                spared_mid_pass += 1
                return
            # WHICH engine wrote this row, reported by the engine that wrote it —
            # not the backend that was asked for. 'auto' chains both, so the
            # requested value would mislabel roughly half the bank.
            caption_origin.stamp(row, caption,
                                 caption_origin.engine_origin(engine))
            # Read off the ROW, and read BEFORE the commit — the commit expires the
            # instance, so reading after it would cost one SELECT per image on a
            # 100 000-image bank. `stamp` clears the origin on a blank caption, so
            # the stored value is what the row really carries, not what was passed.
            stored = row.caption_origin
            db.session.commit()
            captioned += 1
            if stored:
                wrote[stored] = wrote.get(stored, 0) + 1
            else:
                unrecorded_writes += 1

        # GPU-exclusive for the whole pass, exactly like the score/watermark passes:
        # frees ComfyUI VRAM and blocks a training start for the duration.
        # The vocabulary register and the length preset ride in as the SAME appended
        # instructions the dataset pass uses, in the same order (None when neither is
        # set → byte-identical to the plain pass).
        # The engine and the Ollama model ride the SAME per-call seam the Caption
        # Lab uses (caption_paths already takes both); None on either means "the
        # global setting", so a run that picks nothing is byte-identical.
        extra = caption_preset_instructions(vocabulary, length)
        with gpu_exclusive_vision_window(flag_ttl=1800):
            caption_paths(
                paths,
                backend=backend,
                ollama_model=ollama_model,
                extra_instructions=extra,
                should_cancel=lambda: bank_jobs.cancelled(job),
                on_caption=_on_caption,
                progress=lambda d, t: bank_jobs.progress(job, done=d, total=t))
        skipped = ''
        if skipped_asserted:
            # Named in the RESULT, not only in the warning before the click: the
            # user has to be able to see afterwards that the protection did
            # something, otherwise it is a promise with no evidence.
            skipped += f', {skipped_asserted} kept (written by you)'
        skipped += _skipped_note(vanished=vanished, stale=stale)
        if spared_mid_pass:
            skipped += f', {spared_mid_pass} kept (newer caption won)'
        if bank_jobs.cancelled(job):
            bank_jobs.progress(
                job, detail=f'cancelled — {captioned} captioned so far' + skipped)
            return
        detail = f'done — {captioned} captioned'
        detail += _caption_writers_note(wrote, unrecorded_writes)
        detail += skipped
        detail += _scope_note(
            bank_id,
            None if force else or_(BankImage.caption.is_(None),
                                   BankImage.caption == ''),
            statuses, ids)
        bank_jobs.progress(job, detail=detail)
    return run


# --- "Launch all" pipeline --------------------------------------------------
# The overnight funnel: the user configures it once, hits Launch all, and comes
# back to a triaged, optionally pre-captioned bank. It chains the EXISTING passes
# in the validated order. Each pass already filters status != 'reject', so
# running auto-reject BEFORE the heavy passes means score/watermark/person only
# ever touch the SURVIVORS — the costly work never pays for images we just
# dropped (the deliberate cost/quality trade-off: duplicate "keep best" therefore
# ranks on sharpness/size, not the aesthetic score that isn't computed yet).
PIPELINE_STEPS = ('scan', 'auto_reject', 'score', 'semantic_dedup', 'watermark',
                  'faces', 'framing', 'caption')
_SIGLIP2_PIPELINE_STEPS = (
    'scan', 'auto_reject', 'score', 'semantic_index', 'semantic_dedup',
    'watermark', 'faces', 'framing', 'caption')
# Auto-reject inside the pipeline runs right after the quality scan, so it can
# only act on the CPU-scan flags (and duplicates). The score-derived flags
# (low_aesthetic/nsfw/watermark) have no data yet at that point.
#
# NOT every quality flag, though. `soft_detail` and `bars` are excluded ON
# PURPOSE — they are provenance HINTS, not verdicts, and their own documentation
# says so: a crisp watermark rescues an enlargement's detail ratio while a
# motion-blurred native shot sinks it, and `bars` fires on any dark-themed
# screenshot. The standalone 🧹 Auto-reject button still offers them, because
# there a human is looking at the flagged count, can undo on the spot, and the
# hint under the checkbox says "check before mass-rejecting". The pipeline is the
# opposite situation: unattended, and auto-reject runs FIRST, so anything it
# drops never reaches the score / watermark / caption passes at all — the mistake
# becomes invisible instead of reviewable. Offering a non-verdict as an overnight
# bulk rejection contradicts the measurement it is built on.
# The pipeline UI never offered these two; this makes the API agree with it.
_PIPELINE_EXCLUDED_REJECT_FLAGS = ('soft_detail', 'bars')
PIPELINE_REJECT_FLAGS = tuple(f for f in _QUALITY_FLAGS
                              if f not in _PIPELINE_EXCLUDED_REJECT_FLAGS)


def _sanitize_pipeline_steps(steps, engine='clip') -> list:
    """Keep only known steps, in the canonical pipeline order (the client can't
    reorder or invent a pass)."""
    want = set(steps or [])
    canonical = (_SIGLIP2_PIPELINE_STEPS
                 if bank_semantic_engine.normalize_engine(engine) == 'siglip2'
                 else PIPELINE_STEPS)
    return [s for s in canonical if s in want]


def _score_prereq() -> str | None:
    from ..capabilities import probe_bank_scoring
    if not probe_bank_scoring().get('ok'):
        return 'bank scoring extra not installed (Setup ▸ Quality tools)'
    return None


def _watermark_prereq() -> str | None:
    from ..capabilities import probe_ollama_model
    if not probe_ollama_model().get('ok'):
        return 'vision model not available (Settings ▸ Local tools)'
    return None


def _faces_prereq() -> str | None:
    from .face_similarity import is_available
    if not is_available():
        return 'face scoring extra not installed (Setup ▸ Quality tools)'
    return None


def _framing_prereq() -> str | None:
    from ..capabilities import probe_ollama_model
    if not probe_ollama_model().get('ok'):
        return 'vision model not available (Settings ▸ Local tools)'
    return None


def _caption_prereq() -> str | None:
    if (cfg.get('captioning.backend') or 'auto').lower() == 'none':
        return 'no captioning backend configured (Settings ▸ Captioning & quality)'
    return None


def start_pipeline(app, user_id, bank_id, steps=None, reject_flags=None,
                   resolve_dups=False):
    """Launch the chained triage pipeline. ``steps`` selects which passes run
    (canonical order enforced); ``reject_flags`` + ``resolve_dups`` configure the
    auto-reject step. One background job like every other pass — BankJobBusy when
    one is already live, ValueError on a bad bank / empty step list."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    steps = _sanitize_pipeline_steps(steps, _selected_semantic_engine(bank))
    if not steps:
        raise ValueError('no pipeline steps selected')
    reject_flags = [f for f in (reject_flags or []) if f in PIPELINE_REJECT_FLAGS]
    return bank_jobs.start(
        app, bank_id, 'pipeline',
        _pipeline_job(user_id, bank_id, steps, reject_flags, bool(resolve_dups)),
        total=0)


def _bank_counts(bank_id) -> dict:
    """Live headline counts used for the per-step tallies and the final report."""
    base = BankImage.query.filter_by(bank_id=bank_id)
    dup_groups = (db.session.query(BankImage.dup_group)
                  .filter(BankImage.bank_id == bank_id,
                          BankImage.dup_group.isnot(None))
                  .distinct().count())
    style_groups = (db.session.query(BankImage.style_cluster)
                    .filter(BankImage.bank_id == bank_id,
                            BankImage.style_cluster.isnot(None))
                    .distinct().count())
    semantic_groups = (db.session.query(BankImage.semantic_dup_group)
                       .filter(BankImage.bank_id == bank_id,
                               BankImage.semantic_dup_group.isnot(None))
                       .distinct().count())
    person_groups = (db.session.query(BankImage.face_cluster)
                     .filter(BankImage.bank_id == bank_id,
                             BankImage.face_cluster.isnot(None))
                     .distinct().count())
    return {
        'total': base.count(),
        'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
        'reject': base.filter_by(status='reject').count(),
        'scored': base.filter(or_(BankImage.aesthetic_score.isnot(None),
                                  BankImage.nsfw_score.isnot(None))).count(),
        'watermark_detected': base.filter(
            BankImage.watermark_state == 'detected',
            ~_watermark_history_inactive_clause()).count(),
        'framing_classified': base.filter(BankImage.framing.isnot(None)).count(),
        'captioned': base.filter(and_(BankImage.caption.isnot(None),
                                       BankImage.caption != '')).count(),
        'dup_groups': dup_groups,
        'style_groups': style_groups,
        'semantic_groups': semantic_groups,
        'person_groups': person_groups,
    }


def _pipeline_job(user_id, bank_id, steps, reject_flags, resolve_dups):
    def run(job):
        import json as _json
        import time as _time
        from ..gpu_window import GpuBusyError
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        results = []
        pipe = {'steps': list(steps), 'total_steps': len(steps),
                'index': 0, 'current': steps[0], 'results': results}

        def _sync(current=None, index=None):
            if index is not None:
                pipe['index'] = index
            if current is not None:
                pipe['current'] = current
            pipe['results'] = list(results)
            bank_jobs.set_pipeline(job, pipe)

        _sync()
        # Each entry closes with a status: 'done' | 'skipped' (reason) | 'error'.
        for i, step in enumerate(steps):
            if bank_jobs.cancelled(job):
                break
            _sync(current=step, index=i)
            bank_jobs.progress(job, done=0, total=0,
                               detail=f'step {i + 1}/{len(steps)}: {step}')
            entry = {'step': step, 'status': 'done', 'reason': None,
                     'detail': None, 'counts': {}}
            try:
                _run_pipeline_step(job, user_id, bank_id, step,
                                   reject_flags, resolve_dups, entry)
            except GpuBusyError as e:
                # A vision/training job grabbed the GPU mid-pipeline — skip this
                # pass and keep going (never wake the user for a transient clash).
                entry['status'] = 'skipped'
                entry['reason'] = f'GPU busy — {e}'
            except Exception as e:  # noqa: BLE001 — one bad pass never sinks the rest
                entry['status'] = 'error'
                entry['reason'] = f'{type(e).__name__}: {e}'
                db.session.rollback()
            # A step that executed stays 'done' even if a cancel landed at its
            # tail (its inner run already returned early); only steps we never
            # reach are recorded as cancelled, below.
            results.append(entry)
            _sync()

        cancelled = bank_jobs.cancelled(job)
        # Any step never reached (cancel, or a hard earlier break) is recorded so
        # the morning-after report has a row for every requested pass.
        reached = {e['step'] for e in results}
        for step in steps:
            if step not in reached:
                results.append({'step': step, 'status': 'cancelled' if cancelled
                                else 'skipped',
                                'reason': 'cancelled before it ran' if cancelled
                                else 'not reached', 'detail': None, 'counts': {}})
        _sync()

        report = {
            'started_at': job.get('started_at'),
            'finished_at': _time.time(),
            'cancelled': cancelled,
            'requested_steps': list(steps),
            'reject_flags': list(reject_flags),
            'resolve_dups': resolve_dups,
            'steps': results,
            'counts': _bank_counts(bank_id),
        }
        bank = db.session.get(ImageBank, bank_id)
        if bank is not None:
            bank.pipeline_report = _json.dumps(report)
            db.session.commit()
        done_n = sum(1 for e in results if e['status'] == 'done')
        skipped_n = sum(1 for e in results if e['status'] in ('skipped', 'cancelled'))
        err_n = sum(1 for e in results if e['status'] == 'error')
        tail = f'done — {done_n}/{len(steps)} steps ran'
        if skipped_n:
            tail += f', {skipped_n} skipped'
        if err_n:
            tail += f', {err_n} errored'
        if cancelled:
            tail = f'cancelled — {done_n}/{len(steps)} steps ran'
        bank_jobs.progress(job, detail=tail)
    return run


def _run_pipeline_step(job, user_id, bank_id, step, reject_flags, resolve_dups, entry):
    """Run ONE pipeline pass into ``entry``, reusing the standalone pass work.
    Prerequisite missing → entry marked 'skipped' with a reason, pipeline
    continues. Reuses each pass's inner ``run(job)`` so progress, cancellation
    and the GPU-exclusive window behave exactly as the standalone buttons."""
    if step == 'scan':
        _scan_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'scanned': c['scanned'], 'dup_groups': c['dup_groups']}
        entry['detail'] = (job.get('detail')
                           or f"scanned {c['scanned']}, {c['dup_groups']} duplicate group(s)")
        return
    if step == 'auto_reject':
        # ONE undo offer for the whole step, not one per sub-pass: the user fired
        # "Launch all", so the unit they would take back is the auto-reject, not
        # its second half. The later steps only ADD analysis columns, so undoing
        # this one leaves them consistent.
        snap = bank_undo.Snapshot('Launch all — auto-reject')
        capability = _job_bank_capability(job)
        rejected = (apply_flags(user_id, bank_id, reject_flags, snapshot=snap,
                                _bank_lease=capability)
                    if reject_flags else {})
        dup_rejected = 0
        if resolve_dups:
            dup_rejected = resolve_dups_keep_best(
                user_id, bank_id, snapshot=snap, _bank_lease=capability)
        snap.commit(bank_id)
        n = sum(rejected.values()) + dup_rejected
        entry['counts'] = {'rejected': n, 'by_flag': rejected,
                           'duplicates': dup_rejected}
        parts = [f'{v} {k}' for k, v in rejected.items() if v]
        if dup_rejected:
            parts.append(f'{dup_rejected} duplicate')
        entry['detail'] = (f"rejected {n} image(s)"
                           + (f" ({', '.join(parts)})" if parts else '')
                           + ' — manual ✓/✕ untouched')
        return
    if step == 'score':
        reason = _score_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _score_job(bank_id)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'scored': c['scored'], 'style_groups': c['style_groups']}
        entry['detail'] = job.get('detail') or f"scored {c['scored']} image(s)"
        return
    if step == 'semantic_index':
        bank = db.session.get(ImageBank, bank_id)
        if bank is None or _selected_semantic_engine(bank) != 'siglip2':
            entry['status'], entry['reason'] = (
                'skipped', 'Semantic index is available only with SigLIP2')
            return
        from ..capabilities import probe_bank_siglip2
        probe = probe_bank_siglip2()
        if not probe.get('ok'):
            entry['status'], entry['reason'] = (
                'skipped', probe.get('detail') or 'SigLIP2 is not installed')
            return
        _device, use_gpu = _resolve_semantic_device()
        reason = _gpu_busy_reason() if use_gpu else None
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _semantic_index_job(bank_id)(job)
        measured = semantic_counts(
            bank, engine='siglip2',
            total=BankImage.query.filter_by(bank_id=bank_id).count())
        entry['counts'] = {'semantic_indexed': measured['ok'],
                           'semantic_total': measured['total']}
        entry['detail'] = (job.get('detail') or
                           f"indexed {measured['ok']} image(s) with SigLIP2")
        return
    if step == 'semantic_dedup':
        # Runs right after Score, reusing its cached embeddings (no GPU). Groups
        # crops/variants for review; resolution stays a UI action (keep best/first
        # /manual) — near-dups are fuzzier than exact dHash, so the overnight run
        # surfaces them rather than auto-rejecting. Skipped-with-reason (never a
        # mute ✗) when Score produced no embeddings.
        # Same phase reporting as the standalone button — an overnight run is
        # exactly when a silent hour is hardest to tell from a hung one.
        n = rebuild_semantic_dup_groups(
            bank_id, _bank_lease=_job_bank_capability(job),
            on_phase=lambda done, total, detail: bank_jobs.progress(
                job, done=done, total=total, detail=detail))
        if n is None:
            bank = db.session.get(ImageBank, bank_id)
            if bank is not None and _selected_semantic_engine(bank) == 'siglip2':
                reason = 'run Semantic index first — no SigLIP2 embeddings'
            else:
                reason = 'run ✨ Score first — no embeddings'
            entry['status'], entry['reason'] = 'skipped', reason
            return
        entry['counts'] = {'semantic_groups': n}
        entry['detail'] = f'{n} semantic near-duplicate group(s) to review'
        # The journal records the pass, not the button that fired it. Leaving
        # Launch-all out would let the launch window quote a week-old standalone
        # run over one that finished last night.
        bank = db.session.get(ImageBank, bank_id)
        engine = _selected_semantic_engine(bank) if bank is not None else None
        threshold = _semantic_dup_threshold(engine)
        note_pass_run(bank_id, 'semantic_dedup', detail=entry['detail'],
                      counts={'semantic_groups': n}, engine=engine,
                      threshold=float(threshold),
                      signature=_semantic_partition_signature(bank_id, threshold))
        return
    if step == 'watermark':
        reason = _watermark_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _watermark_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'watermarks': c['watermark_detected']}
        entry['detail'] = job.get('detail') or f"{c['watermark_detected']} with a watermark"
        return
    if step == 'faces':
        reason = _faces_prereq()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _faces_job(bank_id)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'person_groups': c['person_groups']}
        entry['detail'] = job.get('detail') or f"{c['person_groups']} person cluster(s)"
        return
    if step == 'framing':
        reason = _framing_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _framing_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'framing_classified': c['framing_classified']}
        entry['detail'] = job.get('detail') or f"{c['framing_classified']} classified by framing"
        return
    if step == 'caption':
        reason = _caption_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        before = _bank_counts(bank_id)['captioned']
        _caption_job(bank_id, None, False)(job)
        after = _bank_counts(bank_id)['captioned']
        entry['counts'] = {'captioned': max(0, after - before), 'total_captioned': after}
        entry['detail'] = job.get('detail') or f"{after} captioned"
        return
    entry['status'], entry['reason'] = 'skipped', 'unknown step'


def resolve_dups_keep_best(user_id, bank_id, snapshot=None, *,
                           _bank_lease=None) -> int:
    """Auto-resolve every unresolved duplicate group keeping the best member,
    for the pipeline's auto-reject step. Returns the number REJECTED."""
    out = resolve_dups(user_id, bank_id, strategy='best', snapshot=snapshot,
                       _bank_lease=_bank_lease)
    return out.get('rejected', 0)


# --- subfolders (scoping facet) ---------------------------------------------
def subfolders_payload(user_id, bank_id) -> dict | None:
    """Top-level subfolders of the bank's source folder with image counts, for
    the scoping picker. Computed once on open (not polled) — a Telegram export
    nests one folder per chat/date. '' = files at the bank root."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    from collections import Counter
    counts: Counter = Counter()
    for (rel,) in (db.session.query(BankImage.relpath)
                   .filter(BankImage.bank_id == bank_id).all()):
        counts[_subfolder_of(rel)] += 1
    items = [{'name': name, 'count': n}
             for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {'subfolders': items, 'total': sum(counts.values())}


# --- coverage advice (idea by @antonp) --------------------------------------
# A read-only ADVICE panel: from what you'd actually train on (the kept set, or
# every non-rejected image before anything is kept), it says what leans and what
# is thin for a good LoRA — purely from data the passes already computed (framing,
# person clusters, style clusters, resolution). It NEVER selects or rejects; it
# only phrases honest, non-alarmist sentences. Everything here is pure DB math,
# zero GPU — the framing part just needs the 📐 Framing pass to have run.
def _pct(n, total) -> int:
    return int(round(100 * n / total)) if total else 0


# --- visual spread, from the embeddings the ✨ Score pass already cached -------
# The framing/cluster numbers above are labels; they cannot see that two hundred
# images labelled "body" are the same body in the same pose. The CLIP embeddings
# can, and they are already on disk — so this costs one pass over vectors that
# were computed for other reasons, and nothing new runs.
#
# The statistic is the MEAN PAIRWISE COSINE SIMILARITY of the pool. For L2-normed
# vectors it has a closed form that avoids ever building the n×n matrix:
#
#     sum_{i,j} e_i . e_j  ==  || sum_i e_i ||^2
#
# so the mean over off-diagonal pairs is (||s||^2 - n) / (n^2 - n) — one pass,
# O(n·d), exact, and it does not care how big the bank is. That matters: the
# panel is read-only and opened casually, so it must not become the reason a
# 47k-image bank stalls.
#
# THRESHOLDS ARE MEASURED, NOT GUESSED. Taken on the two real banks available
# (117 and 46 775 usable embeddings — deliberately different in size and
# content):
#     whole pool                    0.645   and  0.650
#     random subsets n=20…1000      0.643 … 0.692   (i.e. NOT a function of n)
#     an image + its nearest nbrs   0.788   and  0.900   <- redundant extreme
#     a farthest-point sample       0.538   and  0.220   <- varied extreme
# So ~0.65 is what an ordinary bank looks like, a genuinely repetitive selection
# lands around 0.79-0.90, and the statistic is stable across pool size — which is
# why no n-correction is applied. The bands below sit clear of the ordinary
# reading so a normal bank is never nagged.
_SPREAD_REDUNDANT = 0.80     # at/above: measured territory of nearest-neighbour sets
_SPREAD_LEANING = 0.72       # at/above: tighter than either real bank's whole pool
_SPREAD_MIN_POOL = 10        # under this the mean is noise, so we decline to judge


def _coverage_embeddings(bank, crit):
    """The (m×d) L2-normed embedding matrix for the coverage pool.

    A sibling of ``_pool_embeddings`` rather than a call into it: that one takes
    the curation FILTER dict and goes through ``_pool_query``, which has no way to
    express the coverage pool's "everything not rejected" fallback. Same path
    resolution, same fast-path reasoning — see ``_pool_embeddings``.
    """
    import numpy as np
    emb_by_path = _load_semantic_embeddings(bank)
    if not emb_by_path:
        return None
    rows = (BankImage.query.filter(BankImage.bank_id == bank.id, crit)
            .order_by(BankImage.id.asc()).all())
    base = os.path.realpath(bank.source_path)
    prefix = os.path.normcase(base + os.sep)
    vecs = []
    for r in rows:
        if r.watermark_clean_method or r.rotation:
            p = analysis_image_path(bank, r)
            emb = emb_by_path.get(p) if p else None
            if emb is not None:
                vecs.append(emb)
            continue
        p = os.path.normpath(os.path.join(base, r.relpath))
        emb = emb_by_path.get(p) if os.path.normcase(p).startswith(prefix) else None
        if emb is None:
            p = _abs_under(base, r.relpath)
            emb = emb_by_path.get(p) if p else None
        if emb is not None:
            vecs.append(emb)
    if not vecs:
        return None
    E = np.stack(vecs).astype('float32')
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return E


def _visual_spread(bank, crit, *, engine=None) -> dict:
    """How alike the pool LOOKS, from the selected semantic embeddings.

    Returns ``{'scored': m, 'similarity': float|None, 'band': str}`` where band is
    'redundant' | 'leaning' | 'varied' | 'unknown'. 'unknown' with scored 0 means
    the ✨ Score pass has not run (or its cache went stale) — the panel says so
    rather than drawing a bar it did not measure.
    """
    engine = (_selected_semantic_engine(bank) if engine is None
              else bank_semantic_engine.normalize_engine(engine))
    calibrated = engine == 'clip'
    out = {'scored': 0, 'semantic_indexed': 0, 'similarity': None,
           'band': 'unknown', 'calibrated': calibrated,
           'engine': engine, 'model_key': engine_model_key(engine)}
    try:
        E = _coverage_embeddings(bank, crit)
    except Exception:
        # A read-only advisory must never be the thing that breaks the panel:
        # a corrupt or half-written cache degrades to "not measured".
        return out
    if E is None or len(E) < 2:
        out['scored'] = 0 if E is None else len(E)
        out['semantic_indexed'] = out['scored']
        return out
    n = len(E)
    out['scored'] = n
    out['semantic_indexed'] = n
    if n < _SPREAD_MIN_POOL:
        return out
    s = E.sum(axis=0, dtype='float64')
    sim = (float(s @ s) - n) / (n * n - n)
    sim = max(-1.0, min(1.0, sim))
    out['similarity'] = round(sim, 4)
    if calibrated:
        out['band'] = ('redundant' if sim >= _SPREAD_REDUNDANT
                       else 'leaning' if sim >= _SPREAD_LEANING else 'varied')
    return out


def _coverage_pool_crit(bank_id):
    """The pool the panel describes: the KEPT images, or — before anything is
    kept — every non-rejected one (so the panel is useful from the first look).

    Extracted so the caption and embedding reads share ONE definition with the
    label counts instead of each re-deriving it and drifting. It is returned by a
    function rather than smuggled out inside the stats dict: that dict gets
    jsonify()'d, and an SQLAlchemy criterion in it is a 500 waiting for whoever
    forgets to strip it.
    """
    kept_n = BankImage.query.filter_by(bank_id=bank_id, status='keep').count()
    return ((BankImage.status == 'keep') if kept_n > 0
            else (BankImage.status != 'reject'))


def _coverage_stats(bank_id) -> dict:
    """Everything the coverage panel needs, from the pool the user would train on.
    Pure aggregate SQL, no GPU. Every value here is JSON-serialisable."""
    base = BankImage.query.filter_by(bank_id=bank_id)
    crit = _coverage_pool_crit(bank_id)
    pool_is_kept = base.filter_by(status='keep').count() > 0
    pool = base.filter(crit)
    total = pool.count()

    framing = _framing_counts(bank_id, extra_crit=crit)
    framing_known = sum(framing[k] for k in _FRAMINGS)
    framing_available = framing_known + framing['unknown'] > 0

    # Person clusters within the pool, biggest first (list of sizes).
    person_rows = (db.session.query(BankImage.face_cluster, func.count(BankImage.id))
                   .filter(BankImage.bank_id == bank_id, crit,
                           BankImage.face_cluster.isnot(None))
                   .group_by(BankImage.face_cluster)
                   .order_by(func.count(BankImage.id).desc()).all())
    person_sizes = [int(n) for _c, n in person_rows]
    style_rows = (db.session.query(BankImage.style_cluster, func.count(BankImage.id))
                  .filter(BankImage.bank_id == bank_id, crit,
                          BankImage.style_cluster.isnot(None))
                  .group_by(BankImage.style_cluster)
                  .order_by(func.count(BankImage.id).desc()).all())
    style_sizes = [int(n) for _c, n in style_rows]
    top_person_id = int(person_rows[0][0]) if person_rows else None

    # Resolution: how much of the pool is small (< 1 MP), where low-res caps detail.
    res_scanned = pool.filter(BankImage.width.isnot(None),
                              BankImage.height.isnot(None)).count()
    under_1mp = pool.filter(BankImage.width.isnot(None), BankImage.height.isnot(None),
                            BankImage.width * BankImage.height < 1_000_000).count()

    return {
        'pool': 'kept' if pool_is_kept else 'candidates',
        'total': total,
        'framing': framing, 'framing_known': framing_known,
        'framing_available': framing_available,
        'person': {'clusters': person_sizes, 'top_id': top_person_id,
                   'total': sum(person_sizes),
                   'singletons': sum(1 for n in person_sizes if n == 1)},
        'style': {'clusters': style_sizes, 'total': sum(style_sizes)},
        'resolution': {'scanned': res_scanned, 'under_1mp': under_1mp},
    }


def _coverage_advice(stats: dict) -> list:
    """Turn the coverage stats into a short list of honest, actionable sentences.
    Each is {'tone': 'warn'|'info', 'text': ...}. Pure function of ``stats`` (unit
    of the logic — deterministic on a known distribution). Never alarmist: a
    dominance reads as a QUESTION, a thin axis as a gentle 'add a few'."""
    total = stats['total']
    pool = stats['pool']
    noun = 'kept' if pool == 'kept' else 'candidate'
    out = []
    if total == 0:
        return [{'tone': 'info',
                 'text': 'Nothing to advise on yet — keep some images (or run a '
                         'pass) and the coverage read appears here.'}]

    # Size — most families want a couple dozen.
    if total < 20:
        out.append({'tone': 'warn',
                    'text': f'Only {total} {noun} — most LoRA families train more '
                            f'reliably with 20+ images.'})

    # Framing balance (needs the 📐 Framing pass).
    fr, known = stats['framing'], stats['framing_known']
    if not stats['framing_available']:
        out.append({'tone': 'info',
                    'text': 'Run the 📐 Framing pass to see how your face / bust / '
                            'body / back shots balance.'})
    elif known > 0:
        dom = max(_FRAMINGS, key=lambda k: fr[k])
        dom_share = fr[dom] / known
        thin = [k for k in _FRAMINGS if _pct(fr[k], known) < 10]
        if dom_share >= 0.55 and total >= 8:
            add = ' / '.join(k for k in ('body', 'back', 'bust', 'face')
                             if k in thin) or 'other angles'
            out.append({'tone': 'warn',
                        'text': f'{_pct(fr[dom], known)}% {dom} shots — add '
                                f'{add} for a fuller character.'})
        elif fr['back'] == 0 and known >= 10:
            out.append({'tone': 'info',
                        'text': 'No back views — a few help a character hold up '
                                'from behind (optional).'})

    # Person mix — a dominance is a question, not a verdict.
    ppl = stats['person']
    if ppl['total'] > 0 and ppl['clusters']:
        top = ppl['clusters'][0]
        if len(ppl['clusters']) >= 2 and top / ppl['total'] >= 0.5:
            out.append({'tone': 'info',
                        'text': f'Person #{ppl["top_id"]} is {_pct(top, ppl["total"])}% '
                                f'of the set — is this one subject, or a mix?'})
        if ppl['singletons'] >= 3 and total >= 10:
            out.append({'tone': 'info',
                        'text': f'{ppl["singletons"]} people appear only once — a '
                                f'character LoRA wants one consistent subject.'})

    # Style spread — only when it's genuinely mixed (no single dominant style).
    st = stats['style']
    if len(st['clusters']) >= 2 and st['total'] > 0 \
            and st['clusters'][0] / st['total'] < 0.7 and total >= 8:
        out.append({'tone': 'info',
                    'text': f'{len(st["clusters"])} visual styles in the set — '
                            f'mixing photoreal and illustration can dilute a LoRA.'})

    # Resolution — low-res caps the training resolution.
    res = stats['resolution']
    if res['scanned'] > 0:
        share = res['under_1mp'] / res['scanned']
        if share >= 0.3:
            out.append({'tone': 'info',
                        'text': f'{_pct(res["under_1mp"], res["scanned"])}% are under '
                                f'1 MP — low-res images cap the detail a LoRA can learn.'})

    if not out:
        out.append({'tone': 'info',
                    'text': 'Nothing stands out — your set looks reasonably balanced.'})
    # Warnings first so the panel reads worst-to-mildest.
    out.sort(key=lambda a: 0 if a['tone'] == 'warn' else 1)
    return out


def _spread_advice(spread, total) -> list:
    """The visual-spread sentence, or nothing. Pure function of the measurement.

    Deliberately quiet: a bank in the ordinary band says NOTHING, because a panel
    that comments on every axis every time trains people to skim it.
    """
    if total < _SPREAD_MIN_POOL:
        return []
    if not spread['scored']:
        if spread.get('engine') == 'siglip2':
            return [{'tone': 'info',
                     'text': 'Run Semantic index to also see how varied the set '
                             'LOOKS — the labels above cannot tell two hundred '
                             'near-identical shots from two hundred different ones.'}]
        return [{'tone': 'info',
                 'text': 'Run ✨ Score to also see how varied the set LOOKS — the '
                         'labels above cannot tell two hundred near-identical shots '
                         'from two hundred different ones.'}]
    if spread['similarity'] is None:
        return []
    pct = int(round(100 * spread['similarity']))
    if spread['band'] == 'redundant':
        return [{'tone': 'warn',
                 'text': f'The images look very alike ({pct}% average similarity) — '
                         f'a set this repetitive teaches one look. Use ⚖️ or the '
                         f'diverse pick to spread it out.'}]
    if spread['band'] == 'leaning':
        return [{'tone': 'info',
                 'text': f'The images lean alike ({pct}% average similarity) — '
                         f'workable, but more variety would generalise better.'}]
    return []


# The bank has no `kind` (character/concept/style) — that lives on datasets only.
# It is judged as a CHARACTER source rather than offered as a choice, because the
# rest of this panel already is: the framing target it phrases against is the
# character 12/6/6/1 mix, the person-cluster advice says "a character LoRA wants
# one consistent subject", and the style-cluster advice treats a style mix as
# dilution. Adding a selector here would make one axis kind-aware inside a panel
# whose every other axis assumes a character, and would introduce a stored
# preference (with the alias obligations that carry) to fix an inconsistency this
# panel does not otherwise have. The UI says which lens it is using instead of
# leaving the user to infer it.
_COVERAGE_KIND = 'character'


def coverage(user_id, bank_id) -> dict | None:
    """The read-only coverage advice for the bank (idea by @antonp). Returns the
    distributions the panel renders plus the generated advice, or None if the bank
    is gone. Never mutates.

    Three sources, all already computed by earlier passes: the labels (framing,
    person/style clusters, resolution), the CAPTIONS from the 🏷️ pass — scanned by
    the shared `caption_coverage` lexicon, the same one the dataset panel uses, so
    the two never drift — and the CLIP embeddings from ✨ Score, for whether the
    pool actually LOOKS varied. No model runs.
    """
    from . import caption_coverage

    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    engine = _selected_semantic_engine(bank)
    model_key = engine_model_key(engine)
    stats = _coverage_stats(bank_id)
    crit = _coverage_pool_crit(bank_id)

    captions = [c for (c,) in db.session.query(BankImage.caption)
                .filter(BankImage.bank_id == bank_id, crit).all()]
    variety = caption_coverage.analyse(captions, kind=_COVERAGE_KIND)
    spread = _visual_spread(bank, crit, engine=engine)
    # Nested, because `analyse` returns its own `total`/`axes` and the bank payload
    # already owns `total` — flattening would silently overwrite the pool size.
    stats['variety'] = variety
    stats['visual'] = spread
    stats['engine'] = engine
    stats['model_key'] = model_key
    stats['semantic_indexed'] = int(spread['scored'])

    advice = _coverage_advice(stats)
    if stats['total']:
        # On an empty pool _coverage_advice already says the one true thing
        # ("nothing to advise on yet"); the other two sources would each add their
        # own phrasing of the same emptiness.
        advice += _spread_advice(spread, stats['total'])
        advice += caption_coverage.advice(variety)
    # One list, warnings first — the panel reads worst-to-mildest across all three
    # sources rather than as three separate verdicts the user has to merge.
    advice.sort(key=lambda a: 0 if a['tone'] == 'warn' else 1)
    stats['advice'] = advice
    _verified_semantic_result_provenance(bank, engine, model_key)
    return stats


# --- promotion --------------------------------------------------------------
def _promoted_here(dataset_id):
    """The bank_image ids the dataset STILL holds — one row per promoted image,
    written by the promotion itself (FaceDatasetImage.bank_image_id). Deleting
    the image in the dataset deletes the row, so this shrinks on its own."""
    return (db.session.query(FaceDatasetImage.bank_image_id)
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.bank_image_id.isnot(None)))


def _not_already_on(dataset_id):
    """The criterion 'this kept image is not already sitting on that dataset'.

    Measured, not remembered: an image counts as already there only while the
    dataset really holds a row pointing back at it. Delete that image in the
    dataset and the bank offers it again — the old one-way promoted_dataset_id
    flag never came back, so a bank could end up advertising nothing promotable
    into a dataset it had no image left in, which reads as "the bank lost my
    images".

    promoted_dataset_id survives as the LEGACY answer, for images promoted
    before the back-link existed: nothing writes it any more (a promotion clears
    it as it records the link), so it can only ever describe a pre-upgrade
    promotion, and it is dropped for good the next time that image is promoted.
    """
    return and_(BankImage.id.notin_(_promoted_here(dataset_id)),
                or_(BankImage.promoted_dataset_id.is_(None),
                    BankImage.promoted_dataset_id != dataset_id))


def _promotable_query(bank_id, dataset_id):
    """The KEPT images eligible to promote into ``dataset_id``. Per-target, not a
    global 'promoted anywhere' lock — an image promoted to dataset A stays
    promotable to B. (The dataset-side perceptual dedup on import is the real
    guard against genuine duplicates.)"""
    return (BankImage.query.filter_by(bank_id=bank_id, status='keep')
            .filter(_not_already_on(dataset_id)))


def promotable_count(user_id, bank_id, dataset_id) -> int | None:
    """How many kept images the 'promote all' path would send to ``dataset_id``
    right now — the honest number behind the modal's copy line. None = bank or
    dataset gone."""
    if not get_bank(user_id, bank_id):
        return None
    if not FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first():
        return None
    return _promotable_query(bank_id, dataset_id).count()


_IMPORT_FOLDER_SAFE = re.compile(r'[^A-Za-z0-9 _-]')


def _normalize_import_bank_name(name) -> str:
    if not isinstance(name, str):
        raise ValueError('name must be text')
    name = name.strip()
    if not name:
        raise ValueError('name is required')
    if len(name) > 100:
        raise ValueError('name is too long (max 100 characters)')
    return name


def _import_folder_for(name: str) -> str:
    """A fresh, unused folder under bank_sources_root for an imported bank.
    Suffixes -2, -3… rather than reusing a folder: two imports of the same name
    must never end up sharing (and silently merging) one set of files."""
    stem = _IMPORT_FOLDER_SAFE.sub('_', name).strip() or 'bank'
    root = cfg.bank_sources_root()
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / stem
    i = 2
    while True:
        try:
            # Reservation and creation are the same atomic filesystem action.
            # ``exists()`` followed by ``makedirs(exist_ok=True)`` let two
            # concurrent imports silently share one source folder.
            os.mkdir(candidate)
            return str(candidate)
        except FileExistsError:
            candidate = root / f'{stem}-{i}'
            i += 1


def _stage_import_bank(user_id, name) -> ImageBank:
    """Reserve a private folder and FLUSH its still-uncommitted Bank row.

    The generated id can then be reserved in ``bank_jobs`` before the row becomes
    visible to another request.  Callers either commit it or remove the private
    folder after rolling the transaction back.
    """
    folder = _import_folder_for(name)
    try:
        bank = ImageBank(user_id=user_id, name=name, source_path=folder)
        db.session.add(bank)
        db.session.flush()
        return bank
    except Exception:
        db.session.rollback()
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _discard_unlaunched_import_bank(user_id, bank_id, folder, *,
                                    _bank_lease=None):
    """Remove a staged/committed destination whose worker never took ownership."""
    try:
        db.session.rollback()
    except Exception:  # noqa: BLE001 — cleanup must not mask the launch failure
        logger.warning('bank import: rollback failed', exc_info=True)
    if bank_id is not None and get_bank(user_id, bank_id) is not None:
        return _discard_promoted_bank(
            user_id, bank_id, _bank_lease=_bank_lease)
    # A flush rolled back the row, so delete_bank has no path from which to find
    # the folder.  ``folder`` came directly from _import_folder_for; retain the
    # root check as a final defence against a future caller passing another path.
    return _remove_partial_import_folder(
        folder, context='bank import launch cleanup')


def _scrape_blob_name(raw: bytes) -> str | None:
    """The filename a downloaded blob gets in a bank folder: its own content
    hash. Two consequences, both wanted:

    * a resume that re-downloads the SAME bytes writes the same name, so it
      overwrites itself instead of piling up `photo (2).jpg` — idempotent without
      anyone having to decide what a duplicate is;
    * that is file IDENTITY, not curation. Near-duplicates (a re-encode, a crop,
      the same shot at another size) keep separate names and reach the bank, where
      the duplicate-group pass and the semantic pass are the ONE place that rules
      on them. The dataset outlet's dHash gate deliberately does not run here —
      two different definitions of "duplicate" over one pile is the failure mode
      this whole path exists to avoid.

    None when the bytes are not a raster image we store."""
    if (not isinstance(raw, (bytes, bytearray)) or not raw
            or len(raw) > BANK_SOURCE_MAX_BYTES):
        return None
    try:
        # Same static-content + frame + header + full-decode validator as Dataset
        # preserve import. A scraped blob is later scanned from the live Bank
        # folder, so accepting it here would otherwise create a new bomb ingress.
        ext = _preserved_import_extension(raw, label='bank scrape')
    except (ValueError, MemoryError):
        return None
    return f'{hashlib.sha256(raw).hexdigest()[:24]}{ext}'


def scrape_import_to_bank(user_id, items, bank_id=None, name=None, *,
                          _bank_lease=None, _created=False) -> dict:
    """🕸 Scrape → BANK: the scraper's second destination.

    Downloads the SELECTED scanned images ({'url','title'}) into a bank's source
    FOLDER, then lets the ordinary folder walk inventory them. Two modes:
    ``bank_id`` appends to an existing bank (resume — a bank points at a live
    folder, so a second scrape simply adds to the pile it already holds), while
    ``name`` creates a new bank under ``bank_sources_root()`` exactly like
    "Import to bank" does.

    Deliberately does NOT reuse `scrape_import_urls`: that path is a DATASET
    intake and rightly refuses what cannot be trained on (side < 768 px, ratio
    > 3:1) and what it judges a perceptual duplicate. A bank is the step BEFORE
    that judgement — "too small" and "near-duplicate" are verdicts its own passes
    produce, with thresholds the user moves. Filtering at download time would
    delete the evidence before the triage tool ever sees it. What IS kept from
    that path is the download itself (`_download_scrape_item`: SSRF guard,
    content-type allow-list, image-magic check, size cap) and the per-request cap
    — AND, like the dataset intake, each item's provenance (validated the same
    way, via `normalize_source_metadata`/`_source_metadata_storage`), so a Bank
    image scraped in does not lose its origin the way it used to; promotion to
    a Dataset already forwards `BankImage.source_metadata` unchanged.

    Returns {'bank_id', 'name', 'created', 'saved', 'already_there', 'added',
    'skipped': {...}}. ``added`` is what the folder walk actually inventoried.
    Raises ValueError (bad input) or BankJobBusy (a pass owns the bank)."""
    items = [it for it in (items or []) if isinstance(it, dict) and it.get('url')]
    if not items:
        raise ValueError('no items')
    if len(items) > SCRAPE_IMPORT_MAX:
        raise ValueError(f'max {SCRAPE_IMPORT_MAX} images per import')

    if bank_id is not None:
        if _bank_lease is None:
            # Preserve the established quick 409 (and its test seam); the
            # atomic lease immediately below is still authoritative if this
            # advisory read races a new owner.
            if bank_jobs.running(bank_id):
                snap = bank_jobs.get(bank_id) or {}
                raise bank_jobs.BankJobBusy(
                    snap.get('kind') or 'background')
            with bank_jobs.mutation_lease(bank_id, 'scrape_import') as lease:
                return scrape_import_to_bank(
                    user_id, items, bank_id=bank_id, name=name,
                    _bank_lease=lease, _created=_created)
        bank_jobs.require_reservation(_bank_lease, bank_id)
        bank = get_bank(user_id, bank_id)
        if bank is None:
            raise ValueError('bank not found')
        folder = bank.source_path
        if not folder or not os.path.isdir(folder):
            raise ValueError('this bank\'s folder is unavailable — relocate it first')
        # Appending here WRITES into the bank's folder. On a legacy bank sitting
        # on a dataset that means downloading scraped files straight into the
        # dataset's training images — not destructive, but exactly the sharing
        # this guard exists to end.
        conflict = path_guard.dataset_folder_conflict(folder)
        if conflict:
            raise ValueError(
                'This bank points at a dataset\'s own image folder, so scraping '
                f'into it would drop files inside the dataset. {conflict["message"]}')
    else:
        name = (name or '').strip()
        if not name:
            raise ValueError('name is required')
        bank = reservation = None
        folder = None
        try:
            # Make the new row visible only after its mutation reservation is
            # installed, exactly like the background Bank import paths.
            bank = _stage_import_bank(user_id, name)
            folder = bank.source_path
            reservation = bank_jobs.reserve(bank.id, 'scrape_import')
            db.session.commit()
            return scrape_import_to_bank(
                user_id, items, bank_id=bank.id, name=name,
                _bank_lease=reservation, _created=True)
        except Exception:
            if bank is not None and not bank_jobs.launched(reservation):
                # Before commit this rolls the staged row back; after commit it
                # removes the otherwise empty/partial app-owned Bank.
                _discard_unlaunched_import_bank(
                    user_id, bank.id, folder, _bank_lease=reservation)
            raise
        finally:
            bank_jobs.abort(reservation)

    with ThreadPoolExecutor(max_workers=_SCRAPE_DL_WORKERS) as pool:
        # Kept paired with its item (not a separate byte list): the blob name is
        # only known once the bytes are in hand, and that is also the only way to
        # find back which item's provenance a given blob owns.
        downloaded = list(zip(items, pool.map(_download_scrape_item, items)))
    # Downloads can be slow. Refresh the capability before the first write so a
    # stale/purged lease can never publish beside a newer Bank owner.
    bank_jobs.require_reservation(_bank_lease, bank.id)

    skipped: dict[str, int] = {}
    saved = already_there = 0
    # relpath (== content-hash blob name, files land flat in the bank folder) ->
    # stored provenance JSON, handed to the folder walk below so a freshly
    # inventoried row is born WITH its source instead of the walk having no way
    # to attach one to a bare file it just found on disk. Validated the same way
    # as the dataset intake (normalize_source_metadata via _source_metadata_storage)
    # — never trusted raw from the client.
    source_metadata_by_blob: dict[str, str] = {}
    for item, (reason, raw) in downloaded:
        if reason != 'ok' or not raw:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        blob_name = _scrape_blob_name(raw)
        if blob_name is None:
            skipped['not_image'] = skipped.get('not_image', 0) + 1
            continue
        stored_metadata = _source_metadata_storage(item, image_url=item.get('url'))
        if stored_metadata:
            source_metadata_by_blob[blob_name] = stored_metadata
        dest = os.path.join(folder, blob_name)
        if os.path.exists(dest):
            already_there += 1
            continue
        try:
            with open(dest, 'wb') as fh:
                fh.write(raw)
        except OSError:
            logger.warning('bank scrape: could not write %s', blob_name, exc_info=True)
            skipped['errors'] = skipped.get('errors', 0) + 1
            continue
        saved += 1

    # ONE inventory path for every bank: the same walk that picks up files the
    # user drops in the folder by hand picks these up too. No third insert path.
    sync = refresh_bank(
        user_id, bank.id, force=True, _bank_lease=_bank_lease,
        source_metadata_by_relpath=source_metadata_by_blob) or {}
    return {'bank_id': bank.id, 'name': bank.name, 'created': _created,
            'saved': saved, 'already_there': already_there,
            'added': sync.get('added', 0), 'skipped': skipped}


_BANK_TRANSFER_WATERMARK_STATES = frozenset(
    ('none', 'detected', 'dismissed', 'cleaned', 'failed', 'error'))
_BANK_TRANSFER_FRAMINGS = frozenset(('face', 'bust', 'body', 'back', 'unknown'))
_BANK_TRANSFER_STATUSES = frozenset(('pending', 'keep', 'reject'))


def _copied_image_dimensions(path) -> tuple[int | None, int | None]:
    """Read destination dimensions without letting a Pillow bomb sink promotion."""
    try:
        with safe_bank_source(path, label='bank import') as im:
            return im.size
    except (Image.DecompressionBombError, Image.DecompressionBombWarning,
            OSError, TypeError, ValueError, MemoryError):
        return None, None


def _cache_bundle_matches_analysis(bundle, analysis, *, semantic_engine=None) -> dict:
    """Return only cache lanes that exactly corroborate snapshot scalars."""
    def same_optional(left, right, tolerance):
        if left is None or right is None:
            return left is None and right is None
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance)

    def value(name):
        return analysis.get(name) if isinstance(analysis, dict) else getattr(
            analysis, name)

    out = {}
    score = (bundle or {}).get('score')
    if score is not None:
        clip_group_needs_embedding = (
            value('clip_semantic_dup_group') is not None
            or (semantic_engine == 'clip'
                and value('semantic_dup_group') is not None))
        def score_scalar_matches(cache_value, field, tolerance):
            stored = value(field)
            # A group-only historical snapshot may not carry optional score
            # scalars, but any scalar it *does* advertise must still agree.
            return ((stored is None
                     and (clip_group_needs_embedding or cache_value is None))
                    or (stored is not None and cache_value is not None
                        and same_optional(cache_value, stored, tolerance)))

        if (score['state'] == 'ok'
                and score_scalar_matches(
                    score['aesthetic'], 'aesthetic_score', 1e-4)
                and score_scalar_matches(score['nsfw'], 'nsfw_score', 1e-4)) \
                or (not clip_group_needs_embedding
                    and score['state'] == 'error'
                    and value('aesthetic_score') is None
                    and value('nsfw_score') is None):
            out['score'] = score
    face = (bundle or {}).get('face')
    if face is not None and face['state'] == value('face_state'):
        if (same_optional(face['det'], value('face_det'), 1e-3)
                and same_optional(face['yaw'], value('face_yaw'), 1e-2)):
            out['face'] = face
    # The independent semantic lane has no scalar DB columns to corroborate.
    # Its own exact provenance/model contract was already checked while loading
    # the sidecar/runtime cache, so retaining it here is the only honest test.
    semantic = (bundle or {}).get('semantic')
    if semantic is not None:
        out['semantic'] = semantic
    return out


def _dataset_row_bank_values(row: FaceDatasetImage, copied_path, preserve_analysis: bool,
                             *, analysis_cache_dir=None):
    """The current user-facing Dataset data, plus a compatible old Bank analysis.

    The snapshot never gets to overwrite caption/framing/watermark/provenance or
    curation: those fields are the user's present Dataset choices.  It only adds
    final-WebP measurements when the just-copied bytes still equal the promotion
    fingerprint.  This is deliberately checked against ``copied_path`` after the
    copy, closing the tiny edit-during-copy race as well.
    """
    copied_fingerprint = bank_transfer_metadata.content_fingerprint_path(
        copied_path)
    transfer_metadata = bank_transfer_metadata.capture_transfer_metadata(
        row.transfer_metadata, dataset=row,
        dataset_fingerprint=copied_fingerprint)
    if transfer_metadata is None:
        raise RuntimeError('Dataset transfer metadata is malformed or too large')
    current_values = {
        'caption': row.caption,
        # WHO wrote that caption travels with it. This is THE path that used to
        # destroy hand-written work: a caption typed in the Dataset editor landed
        # in the Bank as an anonymous string, and the Bank's forced pass had no
        # way left to tell it from one of its own.
        'caption_origin': row.caption_origin,
        'framing': row.framing if row.framing in _BANK_TRANSFER_FRAMINGS else None,
        'watermark_state': (row.watermark_state
                            if row.watermark_state in _BANK_TRANSFER_WATERMARK_STATES
                            else None),
        'watermark_bbox': row.watermark_bbox if isinstance(row.watermark_bbox, str) else None,
        'watermark_regions': (row.watermark_regions
                              if isinstance(row.watermark_regions, str) else None),
        'watermark_source': (row.watermark_source
                             if row.watermark_source in ('detector', 'vision') else None),
        'watermark_score': (float(row.watermark_score)
                            if isinstance(row.watermark_score, (int, float))
                            and not isinstance(row.watermark_score, bool)
                            and 0.0 <= float(row.watermark_score) <= 1.0 else None),
        'source_metadata': _source_metadata_storage(row.source_metadata),
        # Dataset-to-Bank only starts from kept rows.  Keep the current explicit
        # decision should it change while the background copy is waiting, but do
        # not invent a Bank-only 'failed' status for a Dataset failure row.
        'status': (row.status if row.status in _BANK_TRANSFER_STATUSES else 'keep'),
        'reject_reason': None,
        'transfer_metadata': transfer_metadata,
        'watermark_clean_method': None,
        # A Bank rotation has already been materialised into the Dataset pixels
        # during promotion.  Restoring it here would turn the copied file twice.
        'rotation': None,
    }
    values = {}
    compatible = None
    cache_bundle = {}
    source_engine = None
    if preserve_analysis:
        compatible = bank_transfer_metadata.compatible_snapshot(
            row.bank_analysis_snapshot, copied_path)
        if compatible:
            values.update(compatible['analysis'])
            source_engine = bank_transfer_metadata.bank_semantic_engine_for_fingerprint(
                row.transfer_metadata, copied_fingerprint)
            cache_ref = compatible.get('cache_ref')
            if cache_ref:
                if not analysis_cache_dir:
                    raise RuntimeError('Bank analysis cache directory is unavailable')
                loaded = bank_transfer_metadata.read_cache_sidecar(
                    analysis_cache_dir, cache_ref)
                if loaded is None:
                    raise RuntimeError(
                        'Bank analysis snapshot references a missing or invalid cache')
                if (source_engine is None and 'score' in loaded
                        and 'semantic' not in loaded):
                    # Pre-selector v3 sidecars carried only the historical CLIP
                    # Score space, so this is the one unambiguous legacy claim.
                    source_engine = 'clip'
                cache_bundle = _cache_bundle_matches_analysis(
                    loaded, compatible['analysis'],
                    semantic_engine=source_engine)
                if set(cache_bundle) != set(loaded):
                    raise RuntimeError(
                        'Bank analysis cache does not match its snapshot')
            score_active = any(compatible['analysis'].get(name) is not None for name in (
                'aesthetic_score', 'nsfw_score', 'medium', 'medium_margin',
                'style_cluster'))
            active_group = values.get('semantic_dup_group')
            if active_group is not None and source_engine in ('clip', 'siglip2'):
                lane = f'{source_engine}_semantic_dup_group'
                # Old v3 snapshots only had the active column.  Promote that
                # value into its SHA/provenance-bound engine lane on restore.
                if values.get(lane) is None:
                    values[lane] = active_group
            if source_engine in ('clip', 'siglip2'):
                values['semantic_dup_group'] = values.get(
                    f'{source_engine}_semantic_dup_group')
            else:
                # Keep explicit inactive lanes as sealed history, but never put
                # an ambiguous legacy group into an arbitrary active space.
                values['semantic_dup_group'] = None
            clip_group_active = values.get('clip_semantic_dup_group') is not None
            siglip2_group_active = (
                values.get('siglip2_semantic_dup_group') is not None)
            # Face state/detection/yaw are portable scalar facts already sealed
            # by the snapshot's exact bytes (or its explicit legacy_tofu
            # assurance).  Only a computed cluster needs the embedding cache to
            # preserve the relation.  Requiring a cache for scalar-only legacy
            # rows made Bank -> Dataset succeed but discarded the entire Bank on
            # the return trip.
            face_computed_cluster = (
                compatible['analysis'].get('face_cluster') is not None
                and compatible['analysis'].get('face_cluster_origin') != 'asserted')
            if ((score_active and 'score' not in cache_bundle)
                    or (clip_group_active and 'score' not in cache_bundle)
                    or (siglip2_group_active and 'semantic' not in cache_bundle)
                    or (face_computed_cluster and 'face' not in cache_bundle)):
                raise RuntimeError(
                    'Bank analysis snapshot is missing a required '
                    'Score/Semantic/Face cache')
        # An incompatible snapshot remains a historical vault. Its SHA prevents
        # activation, but a later restore of the Dataset bytes may make it useful
        # again; importing must never destroy that evidence.
    watermark_names = bank_transfer_metadata.BANK_WATERMARK_ANALYSIS_FIELDS
    current_has_watermark = any(
        current_values.get(name) is not None for name in watermark_names)
    # Dataset fields are the current user-owned truth. When every visible
    # watermark column is empty, retain the snapshot's inactive raw-source
    # history instead of erasing it merely because it was not actionable on the
    # promoted effective bytes.
    values.update({
        name: value for name, value in current_values.items()
        if current_has_watermark or name not in watermark_names
    })
    values['width'], values['height'] = _copied_image_dimensions(copied_path)
    if compatible:
        values['analysis_fingerprint'] = (
            compatible['fingerprint']
            if compatible.get('assurance') == 'exact' else None)
    else:
        # Dataset-owned current framing/watermark values are copied from a
        # synchronously reserved master and can therefore be bound to the exact
        # destination bytes even when there is no historical Bank snapshot.
        has_current_pixel_lane = current_values.get('framing') is not None
        values['analysis_fingerprint'] = (
            bank_transfer_metadata.content_fingerprint_path(copied_path)
            if has_current_pixel_lane else None)
    values['watermark_fingerprint'] = (
        copied_fingerprint if current_has_watermark
        else compatible.get('watermark_fingerprint') if compatible
        else None)
    return values, compatible, cache_bundle


_DATASET_BANK_GENERATION_FIELDS = tuple(dict.fromkeys((
    *bank_transfer_metadata.DATASET_PORTABLE_FIELDS,
    'bank_analysis_snapshot', 'transfer_metadata',
)))


def _dataset_bank_generation(row: FaceDatasetImage) -> tuple:
    return tuple(getattr(row, name) for name in _DATASET_BANK_GENERATION_FIELDS)


def _file_generation(path) -> tuple | None:
    try:
        stat = os.stat(path)
        return (stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)
    except (OSError, TypeError, ValueError):
        return None


def start_dataset_import(app, user_id, dataset_id, name, preserve_analysis=True):
    """The REVERSE of promote: turn a dataset back into a bank. Copies the
    dataset's KEPT images into a folder of their own and registers it as a bank
    under `name`, so the dataset's material can be re-triaged with the bank tools
    (perceptual + semantic dedup, framing, scores) without disturbing it.  By
    default the compatible Bank analysis saved on each Dataset image is restored;
    ``preserve_analysis=False`` instead starts a fresh unanalysed Bank while
    still carrying the Dataset's current captions, framing, watermark metadata,
    source attribution and curation decision.

    COPIES rather than pointing the bank at the dataset's live folder: the two
    would otherwise share files, and curating one would mutate the other. That
    mirrors promote, which copies in the other direction — each side owns its
    images. Kept images only, again mirroring promote (which only ever carries
    kept ones across).

    Background job: hundreds of files is a slow copy, and the bank page already
    renders bank_jobs progress. The destination id and folder are staged first,
    but its row is committed only AFTER the Bank id is reserved; it can therefore
    never become a visible, writable half-built Bank.
    Raises ValueError (-> 400) on a missing dataset, a blank name, invalid
    preservation mode, or nothing kept."""
    from .dataset_storage import dataset_path
    dataset_id = dataset_activity.normalize_dataset_id(dataset_id)
    if not isinstance(preserve_analysis, bool):
        raise ValueError('preserve_analysis must be a boolean')
    name = _normalize_import_bank_name(name)
    token = None
    bank = None
    reservation = None
    bank_id = None
    bank_folder = None
    with _dataset_ingest_lock(user_id, dataset_id):
        ds = FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first()
        if not ds:
            raise ValueError('dataset not found')
        # Reserve BEFORE reading the selected rows.  A query-then-reserve pair
        # leaves a window in which delete/edit can change the generation that the
        # worker believes it captured.
        token = dataset_activity.begin_exclusive(
            dataset_id, 'bank_export', detail='copying to Bank')
        if token is None:
            raise dataset_activity.DatasetActivityBusy(
                'This dataset already has work in progress. Wait for it to '
                'finish before copying it to a Bank.')
        try:
            rows = (FaceDatasetImage.query
                    .filter_by(dataset_id=dataset_id, status='keep')
                    .filter(FaceDatasetImage.filename.isnot(None))
                    .order_by(FaceDatasetImage.id.asc()).all())
            if not rows:
                raise ValueError('nothing to import — keep some images first')
            if len(rows) > BANK_MAX_FILES:
                raise ValueError(f'too many images (max {BANK_MAX_FILES})')
            dataset_activity.progress(token, total=len(rows))
            src_dir = str(dataset_path(dataset_id))
            image_rows = []
            seen_names = set()
            for row in rows:
                filename = row.filename
                if (not isinstance(filename, str) or not filename
                        or os.path.basename(filename) != filename):
                    raise RuntimeError('Dataset contains an invalid image filename')
                key = os.path.normcase(filename)
                if key in seen_names:
                    raise RuntimeError('Dataset contains duplicate image filenames')
                seen_names.add(key)
                source = os.path.join(src_dir, filename)
                source_generation = _file_generation(source)
                if (source_generation is None
                        or source_generation[2] > BANK_SOURCE_MAX_BYTES):
                    raise RuntimeError('A kept Dataset image is unavailable or too large')
                sidecar_generation = None
                if preserve_analysis:
                    snapshot = bank_transfer_metadata.parse_snapshot(
                        row.bank_analysis_snapshot)
                    cache_ref = snapshot.get('cache_ref') if snapshot else None
                    if cache_ref:
                        sidecar_path = os.path.join(
                            src_dir, '.bank-analysis-cache', f'{cache_ref}.npz')
                        sidecar_generation = _file_generation(sidecar_path)
                        if (sidecar_generation is None
                                or sidecar_generation[2]
                                > bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES):
                            raise RuntimeError(
                                'Bank analysis snapshot references a missing or '
                                'oversized cache')
                image_rows.append((
                    row.id, filename, source_generation,
                    _dataset_bank_generation(row), sidecar_generation))
            bank = _stage_import_bank(user_id, name)
            bank_id, bank_folder = bank.id, bank.source_path
            reservation = bank_jobs.reserve(
                bank_id, 'dataset_import', total=len(rows))
            # The Bank first becomes externally visible with its reservation
            # already installed.  The worker is launched only after durability.
            db.session.commit()
            bank_jobs.start(
                app, bank_id, 'dataset_import',
                _dataset_import_job(
                    bank_id, dataset_id, src_dir, image_rows, token,
                    preserve_analysis),
                total=len(rows), reservation=reservation)
            # Compatibility with tests/integrators that temporarily replace
            # bank_jobs.start with an inline runner: their runner cannot adopt
            # our registry entry, so release just that unlaunched reservation.
            if not bank_jobs.launched(reservation):
                bank_jobs.abort(reservation)
        except Exception:
            bank_jobs.abort(reservation)
            if bank is not None:
                _discard_unlaunched_import_bank(
                    user_id, bank_id, bank_folder)
            dataset_activity.end(token)
            raise
    return bank_id


def _dataset_import_job(bank_id, dataset_id, src_dir, image_rows, activity_token,
                        preserve_analysis=True):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        user_id = bank.user_id if bank else None

        def abort(message):
            if user_id is not None:
                _fail_discarding_promoted_bank(
                    job, user_id, bank_id, message)
            else:
                bank_jobs.fail(job, message)

        try:
            if not bank:
                return
            copied = 0
            cache_entries = {}
            cache_fingerprints = {}
            compatible_semantic_engines = []
            group_maps = {
                name: {} for name in bank_transfer_metadata.BANK_LOCAL_GROUP_FIELDS}
            group_next = {
                name: 1 for name in bank_transfer_metadata.BANK_LOCAL_GROUP_FIELDS}
            analysis_cache_dir = os.path.join(src_dir, '.bank-analysis-cache')
            for i, (image_id, filename, expected_file_generation,
                    expected_generation, expected_sidecar_generation) in enumerate(
                        image_rows, 1):
                if bank_jobs.cancelled(job):
                    abort('Dataset copy cancelled — the partial Bank was discarded.')
                    return
                row = (FaceDatasetImage.query
                       .filter_by(id=image_id, dataset_id=dataset_id)
                       .populate_existing().one_or_none())
                if (row is None or row.filename != filename
                        or _dataset_bank_generation(row) != expected_generation):
                    abort('The Dataset changed during copy — the new Bank was '
                          'discarded and the Dataset was left unchanged.')
                    return
                src = os.path.join(src_dir, filename)
                try:
                    if _file_generation(src) != expected_file_generation:
                        raise ValueError('source generation changed')
                    payload = _read_safe_bank_source_bytes(
                        src, label='dataset-to-bank import')
                except (OSError, TypeError, ValueError, MemoryError,
                        Image.DecompressionBombError,
                        Image.DecompressionBombWarning):
                    abort('A Dataset image disappeared or became unreadable during '
                          'copy — the new Bank was discarded.')
                    return
                if _file_generation(src) != expected_file_generation:
                    abort('A Dataset image changed during copy — the new Bank was '
                          'discarded and the Dataset was left unchanged.')
                    return
                expected_fingerprint = (
                    bank_transfer_metadata.content_fingerprint_bytes(payload))
                if expected_fingerprint is None:
                    abort('Could not fingerprint a Dataset image — the new Bank '
                          'was discarded.')
                    return
                compatible_now = (bank_transfer_metadata.compatible_snapshot(
                    row.bank_analysis_snapshot, src) if preserve_analysis else None)
                cache_ref = (compatible_now.get('cache_ref')
                             if compatible_now else None)
                if expected_sidecar_generation is not None and cache_ref:
                    sidecar_path = os.path.join(
                        analysis_cache_dir, f'{cache_ref}.npz')
                    try:
                        if _file_generation(sidecar_path) != expected_sidecar_generation:
                            raise OSError('sidecar generation changed')
                        with open(sidecar_path, 'rb') as sidecar:
                            sidecar_raw = sidecar.read(
                                bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES + 1)
                    except OSError:
                        sidecar_raw = b''
                    if (_file_generation(sidecar_path) != expected_sidecar_generation
                            or len(sidecar_raw) != expected_sidecar_generation[2]
                            or bank_transfer_metadata.read_cache_sidecar_bytes(
                                sidecar_raw) is None):
                        abort('Bank analysis cache changed during copy — the new '
                              'Bank was discarded.')
                        return
                dest = os.path.join(bank.source_path, filename)
                try:
                    with open(dest, 'wb') as destination:
                        destination.write(payload)
                    size = os.path.getsize(dest)
                    if (size != len(payload)
                            or bank_transfer_metadata.content_fingerprint_path(dest)
                            != expected_fingerprint):
                        raise OSError('destination bytes did not verify')
                    values, compatible, cache_bundle = _dataset_row_bank_values(
                        row, dest, preserve_analysis,
                        analysis_cache_dir=analysis_cache_dir)
                except Exception:  # noqa: BLE001 — any partial transfer aborts
                    logger.warning('dataset import: copy %s failed', filename,
                                   exc_info=True)
                    abort('Could not preserve the complete Dataset image and its '
                          'analysis — the new Bank was discarded.')
                    return
                engine_claim = None
                if compatible:
                    engine_claim = (
                        bank_transfer_metadata.bank_semantic_engine_for_fingerprint(
                            row.transfer_metadata, compatible['fingerprint']))
                    if (engine_claim is None and 'score' in cache_bundle
                            and 'semantic' not in cache_bundle):
                        engine_claim = 'clip'  # unambiguous pre-selector sidecar
                    compatible_semantic_engines.append(engine_claim)
                scope = compatible.get('group_scope') if compatible else None
                for field in bank_transfer_metadata.BANK_LOCAL_GROUP_FIELDS:
                    old = values.get(field)
                    if old is None:
                        continue
                    if not scope:
                        values[field] = None
                        continue
                    key = (scope, int(old))
                    mapped = group_maps[field].get(key)
                    if mapped is None:
                        mapped = group_next[field]
                        group_next[field] += 1
                        group_maps[field][key] = mapped
                    values[field] = mapped
                if engine_claim in ('clip', 'siglip2'):
                    # Group id remapping is lane-local.  Mirror the selected
                    # remapped lane last so active and persisted state cannot
                    # diverge merely because another lane used different ids.
                    values['semantic_dup_group'] = values.get(
                        f'{engine_claim}_semantic_dup_group')
                else:
                    values['semantic_dup_group'] = None
                if values.get('face_cluster') is None:
                    values['face_cluster_origin'] = None
                db.session.add(BankImage(
                    bank_id=bank_id, relpath=filename, file_size=size, **values))
                if cache_bundle and compatible:
                    cache_entries[dest] = cache_bundle
                    cache_fingerprints[dest] = compatible['fingerprint']
                copied += 1
                if i % 200 == 0:
                    db.session.commit()
                bank_jobs.bump(job)
                dataset_activity.bump(activity_token)
            # A synchronous Dataset mutation may have passed its HTTP busy
            # check immediately before the export reservation was installed.
            # Revalidate the complete source generation after the last copy so
            # an early row cannot change while later rows are still being read.
            for (image_id, filename, expected_file_generation,
                 expected_generation, expected_sidecar_generation) in image_rows:
                current = (FaceDatasetImage.query
                           .filter_by(id=image_id, dataset_id=dataset_id)
                           .populate_existing().one_or_none())
                source = os.path.join(src_dir, filename)
                if (current is None or current.filename != filename
                        or _dataset_bank_generation(current) != expected_generation
                        or _file_generation(source) != expected_file_generation):
                    abort('The Dataset changed during copy — the new Bank was '
                          'discarded and the Dataset was left unchanged.')
                    return
                if expected_sidecar_generation is not None:
                    current_snapshot = bank_transfer_metadata.parse_snapshot(
                        current.bank_analysis_snapshot)
                    current_ref = (current_snapshot.get('cache_ref')
                                   if current_snapshot else None)
                    sidecar = (os.path.join(
                        analysis_cache_dir, f'{current_ref}.npz')
                               if current_ref else None)
                    if _file_generation(sidecar) != expected_sidecar_generation:
                        abort('The Dataset analysis cache changed during copy — '
                              'the new Bank was discarded and the Dataset was '
                              'left unchanged.')
                        return
            if (compatible_semantic_engines
                    and compatible_semantic_engines[0] in ('clip', 'siglip2')
                    and all(engine == compatible_semantic_engines[0]
                            for engine in compatible_semantic_engines)):
                bank.semantic_engine = compatible_semantic_engines[0]
            # Uniform history restores that space; mixed/ambiguous history keeps
            # the Bank's durable default (CLIP).  In both cases materialise the
            # active compatibility column from the selected persisted lane.  A
            # NULL active value beside a non-NULL selected lane would let the
            # first engine switch save NULL over valid imported history.
            selected_lane = getattr(
                BankImage,
                f'{_selected_semantic_engine(bank)}_semantic_dup_group')
            BankImage.query.filter_by(bank_id=bank_id).update(
                {BankImage.semantic_dup_group: selected_lane},
                synchronize_session=False)
            db.session.flush()
            _clear_singleton_copy_duplicate_groups(bank_id)
            db.session.commit()
            if not _write_required_transfer_caches(
                    bank_id, cache_entries, cache_fingerprints):
                abort('Could not preserve every Score/Face cache and Semantic cache — the new Bank '
                      'was discarded and the Dataset was left unchanged.')
                return
            reset_score_memo()
            bank_jobs.progress(job, detail=f'{copied} image(s) imported')
        except Exception:  # noqa: BLE001 — make the advertised operation atomic
            logger.exception('dataset import crashed; discarding destination Bank')
            abort('Dataset copy failed — the partial Bank was discarded and the '
                  'Dataset was left unchanged.')
        finally:
            dataset_activity.end(activity_token)
    return run


# --- ⬆ Promote, second destination: a NEW BANK -------------------------------
# Promotion used to lead exactly one place: a dataset. A dataset is the strict,
# training-bound container — isolating 200 candidates out of a 9 000-image dump
# to keep working on them is a different intent, and forcing it through a dataset
# commits material the user has not decided on yet.
#
# Built on the SAME machinery as "Import to bank" (start_dataset_import): a name,
# a folder of its own under bank_sources_root, a background job, and the new
# bank's id back so the UI can jump to it. What is deliberately NOT reused is
# hardlinking: run_archive.py already settled that question for this app — the
# app rewrites images IN PLACE (re-crop, "Reset to auto", watermark cleaning) and
# an in-place rewrite reuses the inode, so two "independent" banks would become
# one at the first edit. Banks never share their files. It costs the bytes.
_SQLITE_ID_MAX = (1 << 63) - 1


def _normalize_promotion_ids(ids) -> list[int]:
    """Validate and stable-deduplicate an explicit promotion selection."""
    if ids is None:
        return []
    if not isinstance(ids, (list, tuple)):
        raise ValueError('image_ids must be a list')
    if len(ids) > BANK_MAX_FILES:
        raise ValueError(f'too many image ids (max {BANK_MAX_FILES})')
    out = []
    seen = set()
    for value in ids:
        if (isinstance(value, bool) or not isinstance(value, int)
                or value <= 0 or value > _SQLITE_ID_MAX):
            raise ValueError('image_ids must contain positive integers')
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _promote_source_rows(bank_id, ids) -> list:
    """The rows a promotion would carry: the explicit selection, or every KEPT
    image when the selection is empty (same rule as promoting to a dataset).

    Ordered by relpath so the copy, the count and the size preview all describe
    the same set in the same order."""
    wanted = _normalize_promotion_ids(ids)
    if wanted:
        rows = []
        for i0 in range(0, len(wanted), _SQL_IN_CHUNK):
            rows.extend(BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(wanted[i0:i0 + _SQL_IN_CHUNK])).all())
    else:
        rows = BankImage.query.filter_by(bank_id=bank_id, status='keep').all()
    rows.sort(key=lambda r: r.relpath)
    return rows


def selection_size(user_id, bank_id, ids) -> dict | None:
    """{'count', 'bytes'} for what a promotion would COPY — the honest weight the
    confirmation shows BEFORE the click.

    Real bytes, not an order of magnitude: today's images average ~300 KB, so
    200 of them are ~60 MB and nobody needs warning; a video bank is three orders
    of magnitude above that and the same dialog must not lie about it. Reads the
    size the scan already recorded (one column, no disk hit), and only stats the
    watermark-CLEANED blobs, which are what a promotion actually copies for those
    rows and whose size the column does not describe. None = bank gone."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    rows = _promote_source_rows(bank_id, ids)
    total = 0
    for r in rows:
        if r.watermark_clean_method:
            try:
                total += os.path.getsize(resolved_image_path(bank, r))
                continue
            except (OSError, TypeError):
                pass            # fall back to the recorded source size
        total += int(r.file_size or 0)
    return {'count': len(rows), 'bytes': total}


def start_bank_promote(app, user_id, bank_id, ids, name):
    """Copy a selection into a BRAND NEW bank named ``name``. 202 + background
    job, like every other pass; returns the new bank's id so the UI can jump to
    the bank being filled.

    The job is registered against the SOURCE bank — that is the bank the user is
    looking at, the one whose rows get marked, and the one a concurrent scan
    would race. So an already-busy source bank is the established 409, and the
    progress bar appears where the user clicked.

    Raises ValueError (-> 400) on a missing bank, a blank name or an empty
    selection, BankJobBusy (-> 409) while another pass runs on the source."""
    name = _normalize_import_bank_name(name)
    ids = _normalize_promotion_ids(ids)
    # Advisory fast-path for the established error shape. The later atomic
    # multi-Bank reservation remains the real fence if this read races.
    if bank_jobs.running(bank_id):
        snap = bank_jobs.get(bank_id) or {}
        raise bank_jobs.BankJobBusy(snap.get('kind') or 'background')
    dest = None
    reservation = None
    dest_id = None
    dest_folder = None
    try:
        # Flush gives the destination an id without making it visible. Reserve
        # source + destination atomically, then commit, then launch: no request
        # can ever observe a half-built destination without its write guard.
        dest = _stage_import_bank(user_id, name)
        dest_id, dest_folder = dest.id, dest.source_path
        reservation = bank_jobs.reserve(
            bank_id, 'bank_promote', reserve_ids=(dest_id,))
        # Selection is frozen only after source + destination are reserved.
        # Otherwise a rotate/status request admitted just before this launch can
        # resume after our query and make the copied generation disagree with
        # the selection the user confirmed.
        bank = get_bank(user_id, bank_id)
        if not bank:
            raise ValueError('bank not found')
        dest.semantic_engine = _selected_semantic_engine(bank)
        rows = _promote_source_rows(bank_id, ids)
        if not rows:
            raise ValueError('nothing to promote — keep or select some images first')
        if len(rows) > BANK_MAX_FILES:
            raise ValueError(f'too many images (max {BANK_MAX_FILES})')
        bank_jobs.progress(reservation, total=len(rows))
        db.session.commit()
        bank_jobs.start(app, bank_id, 'bank_promote',
                        _bank_promote_job(user_id, bank_id, dest_id,
                                          [r.id for r in rows]),
                        total=len(rows), reserve_ids=(dest_id,),
                        reservation=reservation)
        if not bank_jobs.launched(reservation):
            bank_jobs.abort(reservation)
    except Exception:
        bank_jobs.abort(reservation)
        if dest is not None:
            _discard_unlaunched_import_bank(user_id, dest_id, dest_folder)
        raise
    return dest_id


def _discard_promoted_bank(user_id, dest_bank_id, *, _bank_lease=None):
    """Unmake a destination bank that never became one. Uncommitted rows are
    rolled back first, then delete_bank takes the row, the working data and the
    copy folder (it is under bank_sources_root, so it is OURS to remove)."""
    try:
        db.session.rollback()
    except Exception:  # noqa: BLE001 — teardown must not mask the real failure
        logger.warning('bank promote: rollback failed', exc_info=True)
    bank = get_bank(user_id, dest_bank_id)
    if bank is None:
        # Idempotent retry after a successful discard.
        return True
    folder = bank.source_path
    if not _is_imported_source(folder):
        logger.error('bank promote: refusing to discard destination %s with an '
                     'unbounded source folder %r', dest_bank_id, folder)
        return False
    deleted = False
    try:
        deleted = delete_bank(
            user_id, dest_bank_id, _allow_busy=True,
            _bank_lease=_bank_lease)
    except Exception:  # noqa: BLE001
        # delete_bank commits the row removal before it asks Trash to move the
        # folder.  A Trash failure can therefore raise after the DB is already
        # clean; the bounded fallback below must still run.
        logger.warning('bank promote: normal discard failed; trying bounded '
                       'partial-folder cleanup', exc_info=True)
    folder_removed = _remove_partial_import_folder(
        folder, context=f'bank promote {dest_bank_id} cleanup')
    row_removed = get_bank(user_id, dest_bank_id) is None
    if not (row_removed and folder_removed):
        logger.error('bank promote: incomplete discard for destination %s '
                     '(delete=%s row_removed=%s folder_removed=%s)',
                     dest_bank_id, deleted, row_removed, folder_removed)
        return False
    return True


def _fail_discarding_promoted_bank(job, user_id, dest_bank_id, message):
    """Discard a transfer destination and make cleanup failure user-visible."""
    if not _discard_promoted_bank(
            user_id, dest_bank_id,
            _bank_lease=_job_bank_capability(job)):
        message = (f'{message} Cleanup also failed: the partial Bank folder '
                   'could not be removed. Check the application logs and the '
                   'Bank imports folder.')
    bank_jobs.fail(job, message)


def _same_resolved_path(left, right) -> bool:
    """Whether two resolver results identify the same canonical path."""
    if not left or not right:
        return False
    try:
        return os.path.normcase(os.path.realpath(left)) == os.path.normcase(
            os.path.realpath(right))
    except (OSError, TypeError, ValueError):
        return False


def _bank_row_analysis(row: BankImage) -> dict:
    values = {
        name: getattr(row, name)
        for name in bank_transfer_metadata.BANK_DIRECT_COPY_ANALYSIS_FIELDS
    }
    # Rolling-upgrade/manual rows can still have only the historical active
    # value.  Seal a self-describing v3 snapshot without mutating the source DB.
    bank = db.session.get(ImageBank, row.bank_id)
    engine = _selected_semantic_engine(bank) if bank else 'clip'
    lane = f'{engine}_semantic_dup_group'
    if values.get('semantic_dup_group') is not None:
        values[lane] = values['semantic_dup_group']
    else:
        values['semantic_dup_group'] = values.get(lane)
    return values


def _raw_source_fingerprint(bank: ImageBank | None,
                            row: BankImage | None) -> str | None:
    """Fingerprint the immutable/raw Bank source for lineage checks only.

    Transfer payload readers stay pinned to :func:`analysis_image_path`; this
    helper deliberately isolates the one raw-path lookup needed to prove that a
    tracked clean/rotation still descends from the Dataset bytes in its capsule.
    """
    if bank is None or row is None:
        return None
    return bank_transfer_metadata.content_fingerprint_path(
        abs_image_path(bank, row))


_BANK_TRANSFER_GENERATION_FIELDS = (
    'relpath', 'file_size', 'width', 'height',
    'caption', 'caption_origin', 'source_metadata',
    'status', 'reject_reason', 'transfer_metadata',
    'rotation', 'watermark_clean_method', 'analysis_fingerprint',
    'watermark_fingerprint',
    *bank_transfer_metadata.BANK_DIRECT_COPY_ANALYSIS_FIELDS,
)


def _bank_transfer_generation(row: BankImage) -> tuple:
    return tuple(
        getattr(row, name) for name in _BANK_TRANSFER_GENERATION_FIELDS)


_SCORE_TRANSFER_FIELDS = (
    'aesthetic_score', 'nsfw_score', 'medium', 'medium_margin',
    'style_cluster')
_FACE_MEASURED_TRANSFER_FIELDS = ('face_state', 'face_det', 'face_yaw')
_LEGACY_UNPROVED_TRANSFER_FIELDS = (
    'framing', 'dup_group')
_ASSERTED_FACE_TRANSFER_FIELDS = ('face_cluster', 'face_cluster_origin')


def _asserted_face_transfer_values(row: BankImage) -> dict:
    """Return only a complete user-owned face membership assertion.

    The numeric cluster is Bank-local, but the assertion itself is not derived
    from pixels.  It therefore survives a rotation/clean while every measured
    face field is invalidated.  An orphan ``origin`` marker stays out of this
    lane and fails closed as ordinary analysis metadata.
    """
    if (row.face_cluster is None
            or row.face_cluster_origin != 'asserted'):
        return {}
    return {
        'face_cluster': row.face_cluster,
        'face_cluster_origin': 'asserted',
    }


def _has_bank_pixel_analysis(row: BankImage) -> bool:
    asserted = _asserted_face_transfer_values(row)
    non_pixel = _ASSERTED_FACE_TRANSFER_FIELDS if asserted else ()
    return any(getattr(row, name) is not None
               for name in bank_transfer_metadata.BANK_PIXEL_DERIVED_FIELDS
               if name not in non_pixel)


def _has_bank_watermark_analysis(row: BankImage) -> bool:
    return any(getattr(row, name) is not None
               for name in bank_transfer_metadata.BANK_WATERMARK_ANALYSIS_FIELDS)


def _complete_legacy_quality_matches(row: BankImage, payload) -> bool:
    required = ('quality_state', 'blur_score', 'noise_score',
                'uniformity_score', 'dhash', 'origin')
    if row.quality_state != 'ok' or any(getattr(row, name) is None
                                        for name in required):
        return False
    fresh = bank_deterministic_analysis(payload)
    return fresh is not None and all(
        fresh.get(name) == getattr(row, name)
        for name in bank_transfer_metadata.DETERMINISTIC_ANALYSIS_FIELDS)


def _analysis_transfer_assurance(row: BankImage, path, payload, *,
                                 cache_bundle=None) -> str | None:
    """Return ``exact`` / ``legacy_tofu`` or refuse full analysis transport.

    ``legacy_tofu`` is deliberately carried as such in the Dataset snapshot and
    never becomes ``analysis_fingerprint``.  It is the narrow local-compat path
    for a pre-migration row whose complete Quality lane can be freshly reproduced
    from these bytes; framing/watermark still have no historical hash.
    """
    if not isinstance(payload, bytes):
        return None
    fingerprint = bank_transfer_metadata.content_fingerprint_bytes(payload)
    if fingerprint is None:
        return None
    stored = getattr(row, 'analysis_fingerprint', None)
    if stored == fingerprint:
        return 'exact'
    if stored is not None:
        return None
    available = set(cache_bundle or {})
    score_active = (any(getattr(row, name) is not None
                        for name in _SCORE_TRANSFER_FIELDS)
                    or 'score' in available)
    semantic_active = 'semantic' in available
    source_bank = db.session.get(ImageBank, row.bank_id)
    source_engine = _selected_semantic_engine(source_bank) if source_bank else 'clip'
    active_group = getattr(row, 'semantic_dup_group', None)
    clip_group_active = (
        getattr(row, 'clip_semantic_dup_group', None) is not None
        or (source_engine == 'clip' and active_group is not None))
    siglip2_group_active = (
        getattr(row, 'siglip2_semantic_dup_group', None) is not None
        or (source_engine == 'siglip2' and active_group is not None))
    face_measured = any(getattr(row, name) is not None
                        for name in _FACE_MEASURED_TRANSFER_FIELDS)
    face_computed_cluster = (row.face_cluster is not None
                             and row.face_cluster_origin != 'asserted')
    face_active = face_measured or face_computed_cluster or 'face' in available
    if score_active and 'score' not in available:
        return None
    if (clip_group_active and 'score' not in available):
        return None
    if siglip2_group_active and 'semantic' not in available:
        return None
    # A computed cluster is a relation backed by the face embeddings and cannot
    # travel without that runtime lane. Historical scalar measurements
    # (state/detection/yaw) are different: early face caches carried neither a
    # SHA nor a stat signature. They may still cross the one-time legacy TOFU
    # bridge below when the complete deterministic Quality lane reproduces the
    # current bytes. Requiring the obsolete cache here made one such scalar on
    # one selected image discard an otherwise fully proven promotion.
    face_cache_missing = face_active and 'face' not in available
    if face_computed_cluster and face_cache_missing:
        return None

    # For an unbound legacy row these values are only a narrow TOFU signal. New
    # scans store effective dimensions; old rotated rows may still store raw
    # dimensions and therefore safely fail this gate until re-scanned.
    if row.file_size is not None and int(row.file_size) != len(payload):
        return None
    dimensions = _copied_image_dimensions(path)
    if (row.width is not None and row.height is not None
            and dimensions != (row.width, row.height)):
        return None

    deterministic_active = any(
        getattr(row, name) is not None
        for name in bank_transfer_metadata.DETERMINISTIC_ANALYSIS_FIELDS)
    unproved_active = any(getattr(row, name) is not None
                          for name in _LEGACY_UNPROVED_TRANSFER_FIELDS)
    if deterministic_active or unproved_active:
        return ('legacy_tofu' if _complete_legacy_quality_matches(row, payload)
                else None)
    # With no complete deterministic lane there is no TOFU evidence for an
    # unhashed historical face measurement. Keep failing closed in that case.
    if face_cache_missing:
        return None
    # Only SHA-bound Score/Semantic/Face cache lanes remain. Every carried pixel fact is
    # individually proven, so this is exact despite the legacy NULL row marker.
    return ('exact' if (score_active or semantic_active or face_active
                        or _has_bank_watermark_analysis(row)) else None)


def _captured_asserted_face_analysis(row: BankImage, payload: bytes, *,
                                     group_scope: str) -> dict | None:
    """Seal a manual face assertion without carrying any unproved pixel fact."""
    asserted = _asserted_face_transfer_values(row)
    if not asserted:
        return None
    analysis = {
        name: None
        for name in bank_transfer_metadata.BANK_DIRECT_COPY_ANALYSIS_FIELDS
    }
    analysis.update(asserted)
    return bank_transfer_metadata.captured_bank_analysis(
        analysis, payload, assurance='exact', group_scope=group_scope)


def _cache_bundle_matches_row(bundle, row: BankImage) -> dict:
    """Keep only cache lanes whose scalar results agree with the DB row."""
    bank = db.session.get(ImageBank, row.bank_id)
    return _cache_bundle_matches_analysis(
        bundle, row,
        semantic_engine=_selected_semantic_engine(bank) if bank else 'clip')


def _write_required_transfer_caches(bank_id, entries, fingerprints) -> bool:
    """Write every selected cache lane or report an incomplete destination."""
    expected = {
        kind: sum(kind in bundle for bundle in entries.values())
        for kind in ('score', 'semantic', 'face')
    }
    try:
        actual = bank_transfer_metadata.write_runtime_caches(
            _score_cache_path(bank_id), _face_cache_path(bank_id), entries,
            semantic_path=_semantic_cache_path(bank_id),
            expected_fingerprints=fingerprints)
    except Exception:  # noqa: BLE001 — destination cleanup owns the failure
        logger.warning('bank transfer: cache write failed', exc_info=True)
        return False
    return actual == expected


def _bank_portable_capture(row: BankImage, bank: ImageBank | None) -> dict:
    """Portable Bank row plus the container-owned semantic-space choice."""
    values = {
        name: getattr(row, name, None)
        for name in bank_transfer_metadata.BANK_PORTABLE_FIELDS
    }
    values['semantic_engine'] = (
        _selected_semantic_engine(bank) if bank is not None else 'clip')
    return values


def _bank_copy_values(row: BankImage, copied_path, *,
                      preserve_analysis_candidate: bool,
                      source_fingerprint: str | None = None,
                      source_payload: bytes | None = None,
                      cache_bundle=None) -> tuple[dict, bool, str | None]:
    """Metadata for an independent Bank -> Bank file copy.

    The SHA-256 compares the exact validated source bytes with the completed
    destination.  Quality Scan is deliberately not a prerequisite: a manually
    kept row may already have paid Score/Semantic/Face work while every quality field is
    still NULL.
    """
    fresh_analysis = bank_deterministic_analysis(copied_path)
    dimensions = _copied_image_dimensions(copied_path)
    assurance = (_analysis_transfer_assurance(
        row, copied_path, source_payload, cache_bundle=cache_bundle)
                 if preserve_analysis_candidate and isinstance(source_payload, bytes)
                 else None)
    preserve_analysis = (
        preserve_analysis_candidate
        and isinstance(source_fingerprint, str)
        and bank_transfer_metadata.content_fingerprint_path(copied_path)
        == source_fingerprint
        and assurance is not None
    )
    values = {
        name: None
        for name in bank_transfer_metadata.BANK_TRANSFORM_STALE_ANALYSIS_FIELDS
    }
    values.update({
        name: None
        for name in bank_transfer_metadata.DETERMINISTIC_ANALYSIS_FIELDS
    })
    values.update(fresh_analysis or {})
    if preserve_analysis:
        values.update(_bank_row_analysis(row))
    # A folder/person assertion is user-owned membership, not a measurement of
    # the pre-transform pixels.  Reapply only that pair after the stale-analysis
    # reset even when no full analysis generation can be transported.
    values.update(_asserted_face_transfer_values(row))
    values.update({
        'caption': row.caption,
        # Authorship is a fact about the caption, independent of whether the
        # copied pixels required fresh analysis.
        'caption_origin': row.caption_origin,
        'source_metadata': _source_metadata_storage(row.source_metadata),
        # The target file IS already cleaned/rotated when the source was resolved.
        # Replaying either pointer would look for a non-existent derived blob or
        # rotate the image a second time.
        'watermark_clean_method': None,
        'rotation': None,
        'status': (row.status if row.status in ('pending', 'keep', 'reject')
                   else 'pending'),
        'reject_reason': row.reject_reason,
    })
    source_bank = db.session.get(ImageBank, row.bank_id)
    raw_fingerprint = _raw_source_fingerprint(source_bank, row)
    transfer_metadata = bank_transfer_metadata.capture_transfer_metadata(
        row.transfer_metadata, bank=_bank_portable_capture(row, source_bank),
        bank_fingerprint=source_fingerprint,
        rebind_dataset_from=raw_fingerprint)
    if transfer_metadata is None:
        raise RuntimeError('Bank transfer metadata is malformed or too large')
    values['transfer_metadata'] = transfer_metadata
    values['width'], values['height'] = dimensions
    values['analysis_fingerprint'] = (
        source_fingerprint if assurance == 'exact'
        else None if preserve_analysis
        else (source_fingerprint if fresh_analysis is not None else None))
    values['watermark_fingerprint'] = (
        row.watermark_fingerprint if preserve_analysis else None)
    return values, preserve_analysis, assurance


def _clear_singleton_copy_duplicate_groups(bank_id) -> None:
    """Drop duplicate labels that no longer describe a destination relation."""
    for field in bank_transfer_metadata.BANK_COPY_DUPLICATE_GROUP_FIELDS:
        column = getattr(BankImage, field)
        singleton_groups = [
            group for group, in (
                db.session.query(column)
                .filter(BankImage.bank_id == bank_id, column.isnot(None))
                .group_by(column)
                .having(func.count(BankImage.id) == 1)
                .all()
            )
        ]
        for i0 in range(0, len(singleton_groups), _SQL_IN_CHUNK):
            (BankImage.query
             .filter(BankImage.bank_id == bank_id,
                     column.in_(singleton_groups[i0:i0 + _SQL_IN_CHUNK]))
             .update({field: None}, synchronize_session=False))


def _bank_promote_job(user_id, src_bank_id, dest_bank_id, ids):
    def transfer(job):
        src = db.session.get(ImageBank, src_bank_id)
        dest = db.session.get(ImageBank, dest_bank_id)
        if not src or not dest:
            if dest is not None:
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'The source Bank disappeared — the new bank was discarded.')
            return
        rows = _promote_source_rows(src_bank_id, ids)
        bank_jobs.progress(job, done=0, total=len(rows), detail='copying')
        # A corrupted/legacy DB can contain duplicate relpaths, and a Bank from
        # a case-sensitive volume can contain A.jpg beside a.jpg. Both collapse
        # onto one destination on Windows. Validate a one-row/one-file mapping
        # before the first byte is written, including file-vs-directory clashes.
        target_by_id = {}
        target_keys = set()
        destination_root = os.path.abspath(dest.source_path)
        for row in rows:
            relpath = row.relpath
            if (not isinstance(relpath, str) or not relpath
                    or os.path.isabs(relpath)):
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'A selected source has an invalid path — the new Bank was '
                    'discarded.')
                return
            normalized = os.path.normpath(relpath)
            target = os.path.abspath(os.path.join(destination_root, normalized))
            try:
                contained = (os.path.commonpath([destination_root, target])
                             == destination_root and target != destination_root)
            except (OSError, ValueError):
                contained = False
            key = normalized.replace('\\', '/').casefold()
            if (not contained or key in target_keys
                    or key in ('.', '..') or key.startswith('../')):
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'Selected source paths collide in the new Bank — the new '
                    'Bank was discarded before copying.')
                return
            target_keys.add(key)
            target_by_id[row.id] = target
        for key in target_keys:
            parts = key.split('/')
            if any('/'.join(parts[:i]) in target_keys
                   for i in range(1, len(parts))):
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'A selected source path collides with another image folder '
                    'in the new Bank — the new Bank was discarded before copying.')
                return
        copied, unreadable, source_plans = [], 0, []
        cache_index = bank_transfer_metadata.load_runtime_cache_index(
            _score_cache_path(src_bank_id), _face_cache_path(src_bank_id),
            semantic_path=_semantic_cache_path(src_bank_id),
            wanted_paths=list(dict.fromkeys(
                path for row in rows
                for path in (abs_image_path(src, row),
                             analysis_image_path(src, row)) if path)))
        cache_entries, cache_fingerprints = {}, {}
        for r in rows:
            if bank_jobs.cancelled(job):
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'Copy cancelled — the partial Bank was discarded.')
                return
            expected_generation = _bank_transfer_generation(r)
            # RESOLVED path: a watermark-cleaned image must land cleaned, same
            # rule as promoting to a dataset.  Analysis is bound to these
            # effective bytes, so a baked clean/rotation remains transferable
            # only when its shared fingerprint proves this exact payload.
            p = analysis_image_path(src, r, refresh_rotation=True)
            preserve_analysis_candidate = True
            try:
                # Read + validate the exact bounded bytes before writing. A
                # `copy2` after a header-only check could race a live folder
                # replacement and carry a different (unsafe) file into the new
                # Bank.
                payload = _read_safe_bank_source_bytes(
                    p, label='bank-to-bank promotion')
            except (OSError, TypeError, ValueError, MemoryError,
                    Image.DecompressionBombError, Image.DecompressionBombWarning):
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'A selected source image became unreadable — the new Bank '
                    'was discarded.')
                return
            target = target_by_id[r.id]
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, 'wb') as target_file:
                    target_file.write(payload)
                size = os.path.getsize(target)
            except OSError as e:
                # The source opened, so this is the DESTINATION refusing: disk
                # full, read-only, drive unplugged. Carrying on would leave a
                # bank holding half the selection and presenting as finished —
                # the one outcome worse than failing. Unmake it and say so.
                logger.warning('bank promote: writing the copy failed',
                               exc_info=True)
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'Could not write the copies — the new bank was discarded '
                    'and nothing was changed. Check the free space on the drive '
                    "holding the app's data, then try again. "
                    f'({e.strerror or "write failed"})')
                return
            source_fingerprint = bank_transfer_metadata.content_fingerprint_bytes(payload)
            bundle = bank_transfer_metadata.cache_bundle_for_transfer(
                cache_index, p, payload)
            bundle = _cache_bundle_matches_row(bundle, r)
            try:
                values, preserved, assurance = _bank_copy_values(
                    r, target,
                    preserve_analysis_candidate=preserve_analysis_candidate,
                    source_fingerprint=source_fingerprint,
                    source_payload=payload, cache_bundle=bundle)
            except (RuntimeError, TypeError, ValueError):
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'Could not preserve the complete Bank metadata — the new '
                    'bank was discarded and the source Bank was left unchanged.')
                return
            if (_has_bank_pixel_analysis(r) or bundle) and not preserved:
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'Could not prove every analysis lane for the exact source '
                    'bytes — the new bank was discarded and the source Bank was '
                    'left unchanged. Re-run the stale analysis passes and try '
                    'again.')
                return
            db.session.add(BankImage(
                bank_id=dest_bank_id, relpath=r.relpath, file_size=size,
                **values))
            if preserved:
                if bundle:
                    cache_entries[target] = bundle
                    cache_fingerprints[target] = source_fingerprint
            copied.append(r.id)
            source_plans.append((
                r.id, source_fingerprint, expected_generation))
            if len(copied) % 200 == 0:
                db.session.commit()
            bank_jobs.bump(job)
        if not copied:
            _fail_discarding_promoted_bank(
                job, user_id, dest_bank_id,
                'Nothing could be copied — the new bank was discarded. The '
                'selected files could not be read.')
            return
        # The route-level busy check is advisory; a synchronous mutation could
        # have passed it just before this transfer reserved the Bank. Revalidate
        # every source generation after all copies and before publishing caches
        # or provenance. Any drift discards the complete destination.
        for source_id, expected_fingerprint, expected_generation in source_plans:
            current = (BankImage.query
                       .filter_by(id=source_id, bank_id=src_bank_id)
                       .populate_existing().one_or_none())
            current_path = (analysis_image_path(src, current)
                            if current is not None else None)
            if (current is None
                    or _bank_transfer_generation(current) != expected_generation
                    or bank_transfer_metadata.content_fingerprint_path(
                        current_path) != expected_fingerprint):
                _fail_discarding_promoted_bank(
                    job, user_id, dest_bank_id,
                    'The source Bank changed during the copy — the new bank was '
                    'discarded and the source was left unchanged.')
                return
        # The selected set is not necessarily the copied set: unreadable files
        # and cancellation can remove group peers. Flush the rows actually
        # written, then clear only duplicate/semantic labels left with one member.
        # Face/style clusters remain useful classifications even as singletons.
        db.session.flush()
        _clear_singleton_copy_duplicate_groups(dest_bank_id)
        db.session.commit()
        if not _write_required_transfer_caches(
                dest_bank_id, cache_entries, cache_fingerprints):
            _fail_discarding_promoted_bank(
                job, user_id, dest_bank_id,
                'Could not preserve every Score/Face cache and Semantic cache — the new bank was '
                'discarded and the source Bank was left unchanged.')
            return
        reset_score_memo()
        # Marked LAST, and only for what really landed: the source keeps its rows
        # (a promotion never removes anything from the bank it came from) and now
        # says where they went, exactly like a promotion to a dataset.
        for i0 in range(0, len(copied), _SQL_IN_CHUNK):
            (BankImage.query
             .filter(BankImage.bank_id == src_bank_id,
                     BankImage.id.in_(copied[i0:i0 + _SQL_IN_CHUNK]))
             .update({'promoted_bank_id': dest_bank_id},
                     synchronize_session=False))
        db.session.commit()
        detail = f'{len(copied)} image(s) copied into "{dest.name}"'
        if unreadable:
            detail += f', {unreadable} unreadable'
        bank_jobs.progress(job, detail=detail)

    def run(job):
        try:
            return transfer(job)
        except Exception:  # noqa: BLE001 — destination creation is all-or-nothing
            db.session.rollback()
            logger.exception('bank-to-bank promotion crashed')
            _fail_discarding_promoted_bank(
                job, user_id, dest_bank_id,
                'Bank copy failed — the partial Bank was discarded and the '
                'source Bank was left unchanged.')
            return None
    return run


def start_promote(app, user_id, bank_id, ids, dataset_id):
    """Copy a selection into a dataset through the normal import path
    (normalize + perceptual dedup vs the dataset). ``ids`` empty = every KEPT
    image not already on THIS dataset. Background job (a big promotion decodes
    hundreds of files)."""
    dataset_id = dataset_activity.normalize_dataset_id(dataset_id)
    ids = _normalize_promotion_ids(ids)
    activity_token = None
    reservation = bank_jobs.reserve(bank_id, 'promote')
    try:
        # Freeze BOTH ends before selecting rows. Bank first is safe: the only
        # reverse transfer creates a brand-new Bank id while holding the Dataset
        # lock, so there is no opposing Dataset->existing-Bank lock order.
        with _dataset_ingest_lock(user_id, dataset_id):
            bank = get_bank(user_id, bank_id)
            if not bank:
                raise ValueError('bank not found')
            ds = FaceDataset.query.filter_by(
                id=dataset_id, user_id=user_id).first()
            if not ds:
                raise ValueError('dataset not found')
            if not ids:
                ids = [r.id for r in
                       _promotable_query(bank_id, dataset_id)
                       .order_by(BankImage.id.asc()).all()]
            if not ids:
                raise ValueError('nothing to promote — keep some images first')
            bank_jobs.progress(reservation, total=len(ids))
            activity_token = dataset_activity.begin_exclusive(
                dataset_id, 'bank_import', total=len(ids),
                detail='copying images from a Bank')
            if activity_token is None:
                raise dataset_activity.DatasetActivityBusy(
                    'This dataset already has work in progress. Wait for it to '
                    'finish before copying images from a Bank.')
            result = bank_jobs.start(
                app, bank_id, 'promote',
                _promote_job(
                    user_id, bank_id, ids, dataset_id, activity_token),
                total=len(ids), reservation=reservation)
            if not bank_jobs.launched(reservation):
                # Compatibility for inline test/integration runners that do not
                # understand explicit reservation adoption.
                bank_jobs.abort(reservation)
                dataset_activity.end(activity_token)
            return result
    except Exception:
        bank_jobs.abort(reservation)
        if activity_token is not None:
            dataset_activity.end(activity_token)
        raise


def _promote_job(user_id, bank_id, ids, dataset_id, activity_token=None):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = []
        for i0 in range(0, len(ids), _SQL_IN_CHUNK):
            rows.extend(BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all())
        # Byte-identical rows from DIFFERENT Bank sources remain independent,
        # but retrying the SAME source id is idempotent. The Dataset lock wraps
        # this query through the final commit, closing double-click/concurrent
        # job races without merging unrelated provenance.
        existing_source_ids = set()
        row_ids = [row.id for row in rows]
        for i0 in range(0, len(row_ids), _SQL_IN_CHUNK):
            existing_source_ids.update(
                source_id for source_id, in db.session.query(
                    FaceDatasetImage.bank_image_id).filter(
                        FaceDatasetImage.dataset_id == dataset_id,
                        FaceDatasetImage.bank_image_id.in_(
                            row_ids[i0:i0 + _SQL_IN_CHUNK])).all())
        rows = [row for row in rows if row.id not in existing_source_ids]
        rows.sort(key=lambda r: r.id)
        already_present = len(existing_source_ids)
        bank_jobs.progress(
            job, done=already_present, total=already_present + len(rows),
            detail='promoting')
        dataset_activity.progress(
            activity_token, done=already_present,
            total=already_present + len(rows), detail='copying images from a Bank')
        if not rows:
            bank_jobs.progress(
                job, detail=f'done — {already_present} already present')
            dataset_activity.progress(
                activity_token,
                detail=f'done — {already_present} already present')
            return
        stats: dict = {}
        imported_ids = []
        provenance_changes = []
        cache_index = bank_transfer_metadata.load_runtime_cache_index(
            _score_cache_path(bank_id), _face_cache_path(bank_id),
            semantic_path=_semantic_cache_path(bank_id),
            wanted_paths=[path for row in rows
                          if (path := analysis_image_path(bank, row))])
        # One promotion scope, deliberately not the numeric Bank id: backups can
        # move across installations where ids collide, and separate promotions
        # of the same Bank must remain separate provenance events.
        group_scope = uuid.uuid4().hex
        def abort(message):
            db.session.rollback()
            if ((imported_ids or provenance_changes)
                    and not rollback_imported_images(
                        user_id, dataset_id, imported_ids,
                        provenance_changes=provenance_changes)):
                logger.error('bank promotion rollback did not remove every row')
                message += ' Automatic rollback was incomplete; inspect the Dataset.'
            bank_jobs.fail(job, message)

        # Admission pass: no Dataset mutation occurs until every selected row
        # has an exact readable payload and every advertised analysis lane is
        # proven by its shared fingerprint plus required cache.
        plans = []
        try:
            for row in rows:
                if bank_jobs.cancelled(job):
                    abort('Promotion cancelled before import; nothing was changed.')
                    return
                path = analysis_image_path(
                    bank, row, refresh_rotation=True)
                payload = _read_safe_bank_source_bytes(
                    path, label='bank dataset promotion preflight')
                fingerprint = bank_transfer_metadata.content_fingerprint_bytes(payload)
                bundle = bank_transfer_metadata.cache_bundle_for_transfer(
                    cache_index, path, payload)
                bundle = _cache_bundle_matches_row(bundle, row)
                assurance = _analysis_transfer_assurance(
                    row, path, payload, cache_bundle=bundle)
                advertised = _has_bank_pixel_analysis(row) or bool(bundle)
                if advertised and assurance is None:
                    abort('Could not prove every analysis lane for the exact '
                          'promoted bytes. Nothing was imported; re-run the stale '
                          'analysis passes and try again.')
                    return
                plans.append((
                    row.id, fingerprint, _bank_transfer_generation(row)))
        except (OSError, TypeError, ValueError, MemoryError,
                Image.DecompressionBombError, Image.DecompressionBombWarning):
            abort('A selected Bank image is unreadable; nothing was imported.')
            return

        try:
            for c0 in range(0, len(plans), _PROMOTE_CHUNK):
                if bank_jobs.cancelled(job):
                    abort('Promotion cancelled — imported rows were rolled back.')
                    return
                chunk_plans = plans[c0:c0 + _PROMOTE_CHUNK]
                blobs, chunk_rows = [], []
                caps, cap_origins, frms, source_meta, snapshots = [], [], [], [], []
                watermark_states, watermark_bboxes, watermark_regions = [], [], []
                watermark_sources, watermark_scores = [], []
                statuses, transfer_metadatas = [], []
                for row_id, expected_fingerprint, expected_generation in chunk_plans:
                    row = (BankImage.query
                           .filter_by(id=row_id, bank_id=bank_id)
                           .populate_existing().one_or_none())
                    if (row is None
                            or _bank_transfer_generation(row)
                            != expected_generation):
                        abort('The source Bank changed during promotion; imported '
                              'rows were rolled back.')
                        return
                    path = analysis_image_path(bank, row)
                    payload = _read_safe_bank_source_bytes(
                        path, label='bank dataset promotion')
                    if (bank_transfer_metadata.content_fingerprint_bytes(payload)
                            != expected_fingerprint):
                        abort('A source Bank image changed during promotion; '
                              'imported rows were rolled back.')
                        return
                    cache_bundle = bank_transfer_metadata.cache_bundle_for_transfer(
                        cache_index, path, payload)
                    cache_bundle = _cache_bundle_matches_row(cache_bundle, row)
                    assurance = _analysis_transfer_assurance(
                        row, path, payload, cache_bundle=cache_bundle)
                    advertised = _has_bank_pixel_analysis(row) or bool(cache_bundle)
                    if advertised and assurance is None:
                        abort('A source analysis/cache changed during promotion; '
                              'imported rows were rolled back.')
                        return
                    captured = None
                    if assurance is not None:
                        captured = bank_transfer_metadata.captured_bank_analysis(
                            _bank_row_analysis(row), payload,
                            assurance=assurance, group_scope=group_scope,
                            cache_bundle=cache_bundle,
                            watermark_fingerprint=row.watermark_fingerprint)
                        if captured is None:
                            abort('Could not seal complete Bank analysis; imported '
                                  'rows were rolled back.')
                            return
                    else:
                        # Folder/person membership is user-owned metadata.  Seal
                        # only that assertion for the Dataset round-trip; every
                        # pixel-derived field remains NULL until the Dataset
                        # importer measures its final bytes itself.
                        captured = _captured_asserted_face_analysis(
                            row, payload, group_scope=group_scope)
                        if (_asserted_face_transfer_values(row)
                                and captured is None):
                            abort('Could not seal the asserted face membership; '
                                  'imported rows were rolled back.')
                            return
                    blobs.append(payload)
                    chunk_rows.append(row)
                    caps.append(row.caption)
                    cap_origins.append(row.caption_origin)
                    source_meta.append(row.source_metadata)
                    snapshots.append(captured if captured is not None else True)
                    raw_fingerprint = _raw_source_fingerprint(bank, row)
                    transfer_metadata = (
                        bank_transfer_metadata.capture_transfer_metadata(
                            row.transfer_metadata,
                            bank=_bank_portable_capture(row, bank),
                            bank_fingerprint=expected_fingerprint,
                            rebind_dataset_from=raw_fingerprint))
                    if transfer_metadata is None:
                        abort('Could not preserve complete Bank metadata; imported '
                              'rows were rolled back.')
                        return
                    transfer_metadatas.append(transfer_metadata)
                    statuses.append(row.status)
                    # Pixel-derived Dataset columns travel only under the same
                    # exact/legacy claim as the opaque Bank snapshot.
                    frms.append(row.framing if assurance is not None else None)
                    watermark_actionable = (
                        assurance is not None
                        and row.watermark_fingerprint == expected_fingerprint)
                    watermark_states.append(
                        row.watermark_state if watermark_actionable else None)
                    watermark_bboxes.append(
                        row.watermark_bbox if watermark_actionable else None)
                    watermark_regions.append(
                        row.watermark_regions if watermark_actionable else None)
                    watermark_sources.append(
                        row.watermark_source if watermark_actionable else None)
                    watermark_scores.append(
                        row.watermark_score if watermark_actionable else None)
                new_ids, bad = import_images(
                    user_id, dataset_id, blobs, dedupe=False, stats=stats,
                    captions=caps, caption_origins=cap_origins,
                    bank_image_ids=[row.id for row in chunk_rows],
                    framings=frms, source_metadata=source_meta,
                    bank_analysis_snapshots=snapshots,
                    watermark_states=watermark_states,
                    watermark_bboxes=watermark_bboxes,
                    watermark_regions=watermark_regions,
                    watermark_sources=watermark_sources,
                    watermark_scores=watermark_scores,
                    statuses=statuses,
                    transfer_metadatas=transfer_metadatas,
                    preserve_exact_bytes=True,
                    created_ids_sink=imported_ids,
                    provenance_changes_sink=provenance_changes,
                    _dataset_activity_token=activity_token)
                if bad or len(new_ids) != len(chunk_rows):
                    abort('Could not import every selected image; imported rows '
                          'were rolled back.')
                    return
                bank_jobs.bump(job, len(chunk_rows))
                dataset_activity.bump(activity_token, len(chunk_rows))
            # Final source-generation fence. It catches any synchronous Bank
            # write that raced the HTTP busy check after an earlier chunk had
            # already committed into the Dataset.
            for row_id, expected_fingerprint, expected_generation in plans:
                current = (BankImage.query
                           .filter_by(id=row_id, bank_id=bank_id)
                           .populate_existing().one_or_none())
                current_path = (analysis_image_path(bank, current)
                                if current is not None else None)
                if (current is None
                        or _bank_transfer_generation(current)
                        != expected_generation
                        or bank_transfer_metadata.content_fingerprint_path(
                            current_path) != expected_fingerprint):
                    abort('The source Bank changed during promotion; imported '
                          'rows were rolled back.')
                    return
            # Link the source only after the complete destination is durable.
            source_ids = [row_id for row_id, _fingerprint, _gen in plans]
            unlinked = set(stats.get('bank_unlinked') or ())
            for i0 in range(0, len(source_ids), _SQL_IN_CHUNK):
                for source_row in (BankImage.query
                                   .filter(BankImage.bank_id == bank_id,
                                           BankImage.id.in_(
                                               source_ids[i0:i0 + _SQL_IN_CHUNK]))):
                    source_row.promoted_dataset_id = (
                        dataset_id if source_row.id in unlinked else None)
            db.session.commit()
        except Exception:  # noqa: BLE001 — promotion is all-or-nothing
            logger.exception('bank dataset promotion failed')
            abort('Promotion failed — imported rows were rolled back.')
            return
        small = stats.get('small', 0)
        detail = f'done — {len(imported_ids)} imported'
        if already_present:
            detail += f', {already_present} already present'
        if small:
            detail += f', {small} under the recommended size'
        bank_jobs.progress(job, detail=detail)

    def run_locked(job):
        lock = _dataset_ingest_lock(user_id, dataset_id)
        acquired = False
        # Waiting behind another long Dataset operation must keep the Bank
        # reservation alive; otherwise its one-hour stale TTL could detach this
        # worker and let a second promotion start beside it.
        try:
            while not acquired:
                acquired = lock.acquire(timeout=1.0)
                if acquired:
                    break
                dataset_activity.progress(
                    activity_token, detail='waiting for the Dataset write lock')
                if bank_jobs.cancelled(job):
                    bank_jobs.progress(job, detail='cancelled before promotion')
                    return None
            if bank_jobs.cancelled(job):
                bank_jobs.progress(job, detail='cancelled before promotion')
                return None
            return run(job)
        finally:
            dataset_activity.end(activity_token)
            if acquired:
                lock.release()

    return run_locked
