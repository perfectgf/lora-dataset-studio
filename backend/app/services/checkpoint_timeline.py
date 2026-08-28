"""Checkpoint render timelines and bounded animated GIF exports.

A timeline is not merely "all images from a run in step order".  It is a
comparison of the *same render* across training checkpoints: changing a seed,
prompt, sampler, img2img source, or any other image-shaping setting starts a new
series.  ``LoraTestImage.run_id`` is part of that identity as well, so two Test
Studio launches with otherwise identical knobs can never be spliced together.

Only database-owned filenames are resolved, and only when their real path is a
regular file below that row's real dataset directory.  The GIF route therefore
accepts an opaque SHA-256 series id, never a client-provided filesystem path.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from hashlib import sha256
import json
import math
import ntpath
import os
import re
import stat
from tempfile import SpooledTemporaryFile
from threading import BoundedSemaphore
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from ..models import LoraTestImage
from .dataset_storage import dataset_path


CANDIDATE_CAP = 2000
SERIES_CAP = 20
FRAME_CAP = 60

GIF_KEYFRAME_CAP = 16
GIF_FADE_FRAME_CAP = 3
GIF_MAX_EDGE_CAP = 768
GIF_SOURCE_MAX_EDGE = 4096
GIF_SOURCE_PIXEL_CAP = 8_000_000
GIF_SOURCE_FILE_BYTE_CAP = 64 * 1024 * 1024
GIF_OUTPUT_BYTE_CAP = 64 * 1024 * 1024
GIF_SPOOL_MEMORY_BYTES = 4 * 1024 * 1024
TIMELINE_PREVIEW_MAX_EDGE = 1280
TIMELINE_PREVIEW_OUTPUT_BYTE_CAP = 8 * 1024 * 1024
TIMELINE_PREVIEW_SPOOL_MEMORY_BYTES = 1024 * 1024
GIF_DEFAULT_MAX_EDGE = 512
GIF_DEFAULT_FADE_FRAMES = 2
GIF_DEFAULT_DURATION_MS = 700

SQLITE_INT64_MAX = (1 << 63) - 1
SERIES_ID_RE = re.compile(r'^[0-9a-f]{64}$')


class GifRenderBusyError(RuntimeError):
    """Only one bounded GIF render may occupy the local worker at a time."""


class GifRenderTooLargeError(RuntimeError):
    """The encoded GIF exceeded the server-side output budget."""


_GIF_RENDER_GATE = BoundedSemaphore(1)
_GIF_RESPONSE_GATE = BoundedSemaphore(2)
_TIMELINE_PREVIEW_GATE = BoundedSemaphore(4)
_TIMELINE_PREVIEW_RESPONSE_GATE = BoundedSemaphore(16)

# ``step`` below means the inference sampler's step count.  The checkpoint axis
# is the singular ``LoraTestImage.step`` and is deliberately absent here.
CONDITION_FIELDS = (
    'dataset_id',
    'prompt',
    'seed',
    'strength',
    'z_model',
    'aspect',
    'cfg',
    'steps',
    'steps2',
    'negative',
    'sampler',
    'scheduler',
    'extra_loras',
    'krea_rebalance',
    'weight_dtype',
    'enhancer_strength',
    'detail_amount',
    'resolution_tier',
    'resolution_multiplier',
    'init_image',
    'denoise',
    # False = the prompt was sent without the dataset's trigger word — a cell
    # rendered with the trigger and one without are never the same series.
    'inject_trigger',
)


def _json_value(value: Any) -> Any:
    """Return a stable JSON value, including for corrupt legacy float values."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _conditions(row: LoraTestImage) -> dict[str, Any]:
    return {name: _json_value(getattr(row, name)) for name in CONDITION_FIELDS}


def _series_identity(run_id: str, conditions: dict[str, Any]) -> tuple[str, str]:
    signature = {'run_id': run_id, **conditions}
    canonical = json.dumps(signature, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False)
    return sha256(canonical.encode('utf-8')).hexdigest(), canonical


def _created_at(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _safe_existing_file(row: LoraTestImage) -> str | None:
    """Resolve one stored relative filename without permitting path escape.

    The lexical checks make Windows-style traversal hostile on POSIX too (and
    vice versa), while the final ``realpath`` containment check rejects symlink
    escapes.  This is intentionally read-only: a timeline request never creates
    a missing dataset directory.
    """
    raw = str(row.filename or '')
    if not raw or '\x00' in raw or os.path.isabs(raw) or ntpath.isabs(raw):
        return None
    drive, _ = ntpath.splitdrive(raw)
    if drive:
        return None
    parts = [part for part in re.split(r'[\\/]+', raw) if part not in ('', '.')]
    if not parts or any(part == '..' for part in parts):
        return None

    base = os.path.realpath(os.path.abspath(dataset_path(row.dataset_id)))
    candidate = os.path.realpath(os.path.abspath(os.path.join(base, *parts)))
    try:
        if os.path.normcase(os.path.commonpath((base, candidate))) != os.path.normcase(base):
            return None
    except ValueError:  # Different Windows drives.
        return None
    return candidate if os.path.isfile(candidate) else None


def _even_sample(items: list[Any], cap: int) -> list[Any]:
    """Keep both ends and an even training progression within ``cap`` slots."""
    if len(items) <= cap:
        return list(items)
    if cap <= 1:
        return [items[-1]]
    last = len(items) - 1
    indexes = [(i * last) // (cap - 1) for i in range(cap)]
    return [items[i] for i in indexes]


def _frame(row: LoraTestImage, series_id: str) -> dict[str, Any]:
    return {
        'id': row.id,
        'dataset_id': row.dataset_id,
        'record_id': row.record_id,
        'step': row.step,
        # Never hand the original to Chromium here.  This endpoint re-encodes a
        # metadata-free, dimension-bounded WebP used by playback and WebM.
        'url': (f'/api/train/run/{row.record_id}/timeline/'
                f'{series_id}/frame/{row.id}'),
        'created_at': _created_at(row.created_at),
    }


def _checked_record_id(record_id: Any) -> int:
    if isinstance(record_id, bool):
        raise LookupError('timeline run not found')
    try:
        value = int(record_id)
    except (TypeError, ValueError, OverflowError):
        raise LookupError('timeline run not found') from None
    if value < 1 or value > SQLITE_INT64_MAX:
        raise LookupError('timeline run not found')
    return value


def _build_timeline(record_id: int) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Build the public payload plus a private series-id -> paths lookup."""
    record_id = _checked_record_id(record_id)
    query = LoraTestImage.query.filter(
        LoraTestImage.record_id == record_id,
        LoraTestImage.status == 'done',
        LoraTestImage.filename.isnot(None),
        LoraTestImage.step.isnot(None),
        LoraTestImage.run_id.isnot(None),
    )
    candidate_count = query.count()
    # Newest first makes the first usable row for a duplicate checkpoint the
    # winner.  ``id`` is the durable creation sequence used by the galleries.
    rows = query.order_by(LoraTestImage.id.desc()).limit(CANDIDATE_CAP).all()

    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    missing_or_unsafe = 0
    duplicate_steps = 0
    for row in rows:
        path = _safe_existing_file(row)
        if path is None:
            missing_or_unsafe += 1
            continue
        conditions = _conditions(row)
        series_id, canonical = _series_identity(str(row.run_id), conditions)
        group = groups.setdefault(series_id, {
            'id': series_id,
            'canonical': canonical,
            'run_id': str(row.run_id),
            'conditions': conditions,
            'by_step': {},
            'latest_id': row.id,
            'latest_created_at': row.created_at,
            'duplicate_steps': 0,
        })
        step = int(row.step)
        if step in group['by_step']:
            duplicate_steps += 1
            group['duplicate_steps'] += 1
            continue
        group['by_step'][step] = (row, path)

    complete, insufficient_series = [], []
    for group in groups.values():
        if len(group['by_step']) >= 2:
            complete.append(group)
        else:
            insufficient_series.append(group)

    # Most recently rendered series first.  The digest tie-breaker makes the
    # answer deterministic even for rows sharing a database timestamp/id seam.
    complete.sort(key=lambda g: (g['latest_id'], g['id']), reverse=True)
    series_count = len(complete)
    selected_groups = complete[:SERIES_CAP]
    omitted_groups = complete[SERIES_CAP:]

    series_payload, private_paths = [], {}
    frame_cap_excluded = 0
    returned_frames = 0
    complete_distinct_frames = sum(len(g['by_step']) for g in complete)
    for group in selected_groups:
        ordered = [group['by_step'][step]
                   for step in sorted(group['by_step'])]
        shown_pairs = _even_sample(ordered, FRAME_CAP)
        frame_count = len(ordered)
        shown = len(shown_pairs)
        frame_cap_excluded += frame_count - shown
        returned_frames += shown
        public_frames = [_frame(row, group['id']) for row, _path in shown_pairs]
        series_payload.append({
            'id': group['id'],
            'run_id': group['run_id'],
            'conditions': group['conditions'],
            'created_at': _created_at(group['latest_created_at']),
            'frame_count': frame_count,
            'shown': shown,
            'excluded': group['duplicate_steps'] + (frame_count - shown),
            'truncated': shown < frame_count,
            'steps': [frame['step'] for frame in public_frames],
            'frames': public_frames,
        })
        private_paths[group['id']] = [path for _row, path in shown_pairs]

    beyond_candidate_cap = max(0, candidate_count - len(rows))
    insufficient_frames = sum(len(g['by_step']) for g in insufficient_series)
    series_cap_frames = sum(len(g['by_step']) for g in omitted_groups)
    excluded_counts = {
        'beyond_candidate_cap': beyond_candidate_cap,
        'missing_or_unsafe_file': missing_or_unsafe,
        'duplicate_steps': duplicate_steps,
        'insufficient_series': len(insufficient_series),
        'insufficient_series_frames': insufficient_frames,
        'beyond_series_cap': len(omitted_groups),
        'beyond_series_cap_frames': series_cap_frames,
        'beyond_frame_cap': frame_cap_excluded,
    }
    excluded = max(0, candidate_count - returned_frames)
    truncated = bool(beyond_candidate_cap or omitted_groups or frame_cap_excluded)
    payload = {
        'record_id': record_id,
        'count': series_count,
        'shown': len(series_payload),
        'frame_count': complete_distinct_frames,
        'frames_shown': returned_frames,
        'excluded': excluded,
        'excluded_counts': excluded_counts,
        'truncated': truncated,
        'candidate_count': candidate_count,
        'candidates_scanned': len(rows),
        'candidate_cap': CANDIDATE_CAP,
        'series_cap': SERIES_CAP,
        'frame_cap': FRAME_CAP,
        'series': series_payload,
    }
    return payload, private_paths


def checkpoint_timeline(record_id: int) -> dict[str, Any]:
    """Return every usable render series for one training run."""
    return _build_timeline(record_id)[0]


def _clamped_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(low, min(parsed, high))


def _open_regular_source(path: str):
    """Open once, no-follow where supported, then prove the handle is regular."""
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OSError('timeline source is not a regular file')
        lexical = os.lstat(path)
        if (stat.S_ISLNK(lexical.st_mode) or
                (getattr(lexical, 'st_file_attributes', 0) & 0x400)):
            raise OSError('timeline source is a link or reparse point')
        return os.fdopen(descriptor, 'rb', closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _load_keyframe(path: str, max_edge: int) -> Image.Image | None:
    """Decode one source at a time and return an already bounded RGBA image.

    Checking dimensions before ``load`` rejects decompression bombs and keeps a
    malicious/corrupt database row from turning a GIF request into an
    unbounded allocation.  ``draft`` gives JPEG decoders a chance to downsample
    during decode; ``thumbnail`` is the format-independent hard bound.
    """
    try:
        with _open_regular_source(path) as handle:
            if os.fstat(handle.fileno()).st_size > GIF_SOURCE_FILE_BYTE_CAP:
                return None
            with Image.open(handle) as source:
                # Pillow reads dimensions from the header before pixel decode;
                # this app-owned limit is far below Pillow's warning threshold
                # and avoids mutating process-global warning filters in Flask.
                if (getattr(source, 'n_frames', 1) != 1 or
                        source.width < 1 or source.height < 1 or
                        max(source.width, source.height) > GIF_SOURCE_MAX_EDGE or
                        source.width * source.height > GIF_SOURCE_PIXEL_CAP):
                    return None
                try:
                    source.draft('RGB', (max_edge, max_edge))
                except (AttributeError, OSError, ValueError):
                    pass
                source.seek(0)
                transposed = ImageOps.exif_transpose(source)
                try:
                    transposed.thumbnail((max_edge, max_edge),
                                         Image.Resampling.LANCZOS)
                    if transposed.width < 1 or transposed.height < 1:
                        return None
                    return transposed.convert('RGBA')
                finally:
                    if transposed is not source:
                        transposed.close()
    except (Image.DecompressionBombError, OSError, ValueError,
            UnidentifiedImageError):
        return None


def _letterbox(images: list[Image.Image], max_edge: int) -> list[Image.Image]:
    # Inputs already arrive bounded from ``_load_keyframe``.  Keep this second
    # thumbnail as a defensive invariant without retaining full-size copies.
    for image in images:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvases = []
    for image in images:
        canvas = Image.new('RGB', (width, height), (0, 0, 0))
        left = (width - image.width) // 2
        top = (height - image.height) // 2
        if image.mode == 'RGBA':
            canvas.paste(image, (left, top), image)
        else:
            canvas.paste(image, (left, top))
        canvases.append(canvas)
    return canvases


def render_timeline_preview(record_id: int, series_id: str,
                            image_id: int) -> tuple[Any, str]:
    """Return one safe WebP preview that browsers may decode directly."""
    series_id = str(series_id)
    if SERIES_ID_RE.fullmatch(series_id) is None:
        raise LookupError('timeline frame not found')
    record_id = _checked_record_id(record_id)
    image_id = _checked_record_id(image_id)
    if not _TIMELINE_PREVIEW_GATE.acquire(timeout=10):
        raise GifRenderBusyError('timeline previews are busy; retry shortly')
    try:
        row = LoraTestImage.query.filter(
            LoraTestImage.id == image_id,
            LoraTestImage.record_id == record_id,
            LoraTestImage.status == 'done',
            LoraTestImage.filename.isnot(None),
            LoraTestImage.step.isnot(None),
            LoraTestImage.run_id.isnot(None),
        ).first()
        if row is None:
            raise LookupError('timeline frame not found')
        actual_series_id, _canonical = _series_identity(
            str(row.run_id), _conditions(row))
        if actual_series_id != series_id:
            raise LookupError('timeline frame not found')
        path = _safe_existing_file(row)
        if path is None:
            raise LookupError('timeline frame not found')

        image = _load_keyframe(path, TIMELINE_PREVIEW_MAX_EDGE)
        if image is None:
            raise LookupError('timeline frame is not a safe readable image')
        try:
            preview = Image.new('RGB', image.size, (8, 11, 18))
            preview.paste(image, (0, 0), image)
        finally:
            image.close()

        output = SpooledTemporaryFile(
            max_size=TIMELINE_PREVIEW_SPOOL_MEMORY_BYTES, mode='w+b')
        try:
            try:
                preview.save(output, format='WEBP', quality=88, method=4)
            finally:
                preview.close()
            if output.tell() > TIMELINE_PREVIEW_OUTPUT_BYTE_CAP:
                raise GifRenderTooLargeError(
                    'timeline preview exceeds the export limit')
            output.seek(0)
            return output, f'run-{record_id}-frame-{image_id}.webp'
        except Exception:
            output.close()
            raise
    finally:
        _TIMELINE_PREVIEW_GATE.release()


def _render_timeline_gif_unlocked(record_id: int, series_id: str, *,
                                  duration_ms: Any = None,
                                  fade_frames: Any = None,
                                  max_edge: Any = None) -> tuple[Any, str]:
    """Render one published series as a bounded, genuinely blended GIF.

    Up to 16 evenly sampled keyframes are decoded.  Between each pair, one to
    three real ``Image.blend`` frames are emitted; these are visual transitions,
    not repeated holds carrying different timing metadata.
    """
    series_id = str(series_id)
    if SERIES_ID_RE.fullmatch(series_id) is None:
        raise LookupError('timeline series not found')
    payload, paths_by_series = _build_timeline(record_id)
    paths = paths_by_series.get(series_id)
    if paths is None:
        raise LookupError('timeline series not found')

    edge = _clamped_int(max_edge, GIF_DEFAULT_MAX_EDGE, 32, GIF_MAX_EDGE_CAP)
    fades = _clamped_int(fade_frames, GIF_DEFAULT_FADE_FRAMES,
                         1, GIF_FADE_FRAME_CAP)
    hold_ms = _clamped_int(duration_ms, GIF_DEFAULT_DURATION_MS, 40, 5000)
    key_paths = _even_sample(paths, GIF_KEYFRAME_CAP)
    images = [image for image in
              (_load_keyframe(path, edge) for path in key_paths)
              if image is not None]
    if len(images) < 2:
        for image in images:
            image.close()
        raise LookupError('timeline series has fewer than two readable frames')

    try:
        keyframes = _letterbox(images, edge)
    finally:
        for image in images:
            image.close()

    animation: list[Image.Image] = []
    try:
        # Palette frames use one byte/pixel instead of retaining every RGB
        # transition.  The RGB keyframes remain small and bounded for blending.
        animation.append(keyframes[0].quantize(colors=256))
        durations = [hold_ms]
        transition_ms = max(20, min(500, hold_ms // (fades + 1)))
        previous = keyframes[0]
        for current in keyframes[1:]:
            for index in range(1, fades + 1):
                blended = Image.blend(previous, current,
                                      index / (fades + 1))
                try:
                    animation.append(blended.quantize(colors=256))
                finally:
                    blended.close()
                durations.append(transition_ms)
            animation.append(current.quantize(colors=256))
            durations.append(hold_ms)
            previous = current

        output = SpooledTemporaryFile(max_size=GIF_SPOOL_MEMORY_BYTES,
                                      mode='w+b')
        try:
            animation[0].save(
                output,
                format='GIF',
                save_all=True,
                append_images=animation[1:],
                duration=durations,
                loop=0,
                disposal=2,
                optimize=False,
            )
            if output.tell() > GIF_OUTPUT_BYTE_CAP:
                raise GifRenderTooLargeError('timeline GIF exceeds the export limit')
        except Exception:
            output.close()
            raise
    finally:
        for image in animation:
            image.close()
        for image in keyframes:
            image.close()
    output.seek(0)
    name = f'run-{record_id}-timeline-{series_id[:12]}.gif'
    return output, name


def render_timeline_gif(record_id: int, series_id: str, *,
                        duration_ms: Any = None, fade_frames: Any = None,
                        max_edge: Any = None) -> tuple[Any, str]:
    """Serialize GIF work so threaded requests cannot multiply its RAM budget."""
    if not _GIF_RENDER_GATE.acquire(blocking=False):
        raise GifRenderBusyError('another timeline GIF is already rendering')
    try:
        return _render_timeline_gif_unlocked(
            record_id, series_id, duration_ms=duration_ms,
            fade_frames=fade_frames, max_edge=max_edge)
    finally:
        _GIF_RENDER_GATE.release()


def acquire_gif_response_slot() -> bool:
    """Bound completed GIF spools that are still held by slow responses."""
    return _GIF_RESPONSE_GATE.acquire(blocking=False)


def release_gif_response_slot() -> None:
    _GIF_RESPONSE_GATE.release()


def acquire_preview_response_slot() -> bool:
    """Bound encoded WebP spools retained by slow browser responses."""
    return _TIMELINE_PREVIEW_RESPONSE_GATE.acquire(blocking=False)


def release_preview_response_slot() -> None:
    _TIMELINE_PREVIEW_RESPONSE_GATE.release()


__all__ = [
    'CANDIDATE_CAP', 'SERIES_CAP', 'FRAME_CAP',
    'GIF_KEYFRAME_CAP', 'GIF_FADE_FRAME_CAP', 'GIF_MAX_EDGE_CAP',
    'GIF_SOURCE_MAX_EDGE', 'GIF_SOURCE_PIXEL_CAP',
    'GIF_SOURCE_FILE_BYTE_CAP', 'GIF_OUTPUT_BYTE_CAP',
    'TIMELINE_PREVIEW_MAX_EDGE', 'TIMELINE_PREVIEW_OUTPUT_BYTE_CAP',
    'GifRenderBusyError', 'GifRenderTooLargeError',
    'checkpoint_timeline', 'render_timeline_preview', 'render_timeline_gif',
    'acquire_gif_response_slot', 'release_gif_response_slot',
    'acquire_preview_response_slot', 'release_preview_response_slot',
]
