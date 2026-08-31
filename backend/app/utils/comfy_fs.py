"""The FILESYSTEM half of the ComfyUI contract — and what to say when it breaks.

The app talks to ComfyUI over TWO channels, and only one of them is the network:

1. the HTTP API (`comfyui.api_url`) — what the Setup wizard tests;
2. the FILESYSTEM — every local engine copies its source image into ComfyUI's
   `input/` folder and reads the render back from `output/` (the read side already
   degrades to `GET /view`; the write side has no fallback).

A ComfyUI that answers on its URL therefore proves nothing about channel 2. When
ComfyUI runs in another container, in WSL, or on another machine, `input/` is not
shared by default: the copy lands somewhere ComfyUI cannot see, or fails outright
with an OSError that no route maps — a bare 500 with no detail (reported on
Discord by nofaceman).

This module makes that failure SPEAK. Every staged write goes through
`stage_input_copy` / `stage_input_write`, which name the operation, the folder and
the plausible cause, and raise `ComfyFolderUnavailable` (a RuntimeError, so the
existing route mapping answers 409 + message instead of 500 + nothing).

Every message is PASTE-SAFE: paths run through `redact_user_paths`, because these
strings are written to be dropped into a public help thread.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import time
import uuid
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .redact import redact_user_paths
from ..services import image_encoding, input_budget  # noqa: F401 - installs the shared budget

logger = logging.getLogger(__name__)

# Which of the four ComfyUI folders the APP writes into (the rest it only reads).
# Drives both the staging guards and the Settings/Setup writability probe.
WRITTEN_KINDS = ('input', 'loras')

_KIND_LABELS = {'input': "ComfyUI's input folder", 'output': "ComfyUI's output folder",
                'models': "ComfyUI's models folder", 'loras': "ComfyUI's LoRA folder"}

# The one sentence that turns "it failed" into "here is what to do". Said the same
# way everywhere (staging error, Settings warning, Setup note) so it is recognisable.
SHARED_FOLDER_HINT = (
    'If ComfyUI runs in another container, in WSL or on another machine, this folder '
    'must be a shared volume visible to LoRA Dataset Studio at that exact path — '
    'pointing the app at ComfyUI\'s URL is not enough.'
)

_SETTINGS_HINT = ('Settings > Local tools > ComfyUI > Advanced: ComfyUI folder '
                  'overrides lets you point the app at the right folder.')


class ComfyFolderUnavailable(RuntimeError):
    """A ComfyUI working folder cannot be used from this process.

    RuntimeError on purpose: the routes' `_map_error` already turns a RuntimeError
    into a 409 carrying the message, which is exactly the contract here — an
    actionable refusal, never an opaque 500."""


def safe_path(path) -> str:
    """A path as it may be SHOWN to the user / pasted in public: home-dir prefix
    stripped. Everything in this module goes through it."""
    return redact_user_paths(str(path or ''))


def folder_label(kind: str) -> str:
    return _KIND_LABELS.get(kind, f"ComfyUI's {kind} folder")


def folder_problem(kind: str, path) -> str:
    """Why `path` cannot serve as the `kind` folder, or '' when it looks usable.

    Checks only what is CHEAP and certain — configured, exists, is a directory,
    and (for the folders the app writes into) actually writable by this process,
    verified by writing and deleting a probe file rather than trusting a
    permission bit (a bind-mounted read-only volume lies to `os.access`).
    Never raises."""
    if not path:
        return (f'{folder_label(kind)} is not configured. Set the ComfyUI install '
                'directory (or an explicit folder override) in Settings.')
    shown = safe_path(path)
    try:
        if not os.path.exists(path):
            return (f'{folder_label(kind)} does not exist here: {shown}. '
                    + SHARED_FOLDER_HINT)
        if not os.path.isdir(path):
            return f'{folder_label(kind)} is not a directory: {shown}.'
    except OSError as exc:
        return f'{folder_label(kind)} could not be checked: {shown} ({_cause(exc)}).'
    if kind in WRITTEN_KINDS:
        err = _write_probe(path)
        if err:
            return (f'{folder_label(kind)} is not writable from LoRA Dataset Studio: '
                    f'{shown} ({err}). ' + SHARED_FOLDER_HINT)
    else:
        if not os.access(path, os.R_OK):
            return (f'{folder_label(kind)} is not readable from LoRA Dataset Studio: '
                    f'{shown}. ' + SHARED_FOLDER_HINT)
    return ''


def _write_probe(path) -> str:
    """'' when a file can be created AND removed in `path`, else the cause.
    A real write: read-only bind mounts, full disks and ACL surprises all pass
    `os.access(W_OK)` on at least one platform."""
    probe = os.path.join(str(path), f'.lds-write-test-{uuid.uuid4().hex[:8]}')
    try:
        with open(probe, 'wb') as fh:
            fh.write(b'lds')
    except OSError as exc:
        # `strerror` only ("Permission denied"), not the full repr: that one appends
        # the probe file's path, which the sentence around it already gives — the
        # message is read on a 400px-wide screen, every repeated path costs a line.
        return f'{exc.__class__.__name__}: {exc.strerror}' if exc.strerror else _cause(exc)
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass
    return ''


def probe_folder(kind: str, path) -> dict:
    """{'ok', 'problem'} for one folder — the Settings preview and the Setup wizard
    both render this, so a broken mount is a WARNING at configuration time instead
    of a failed generation an hour later. Never raises, never blocks."""
    problem = folder_problem(kind, path)
    return {'ok': not problem, 'problem': problem}


def _cause(exc: BaseException) -> str:
    """Exception rendered short and paste-safe: class + message, no traceback, no
    home path. Empty messages still say WHICH error it was."""
    msg = safe_path(str(exc)).strip()
    return f'{exc.__class__.__name__}: {msg}' if msg else exc.__class__.__name__


def _staging_error(action: str, kind: str, path, cause: str) -> ComfyFolderUnavailable:
    return ComfyFolderUnavailable(
        f'{action} failed. {folder_label(kind)}: {safe_path(path)} '
        f'({cause}). {SHARED_FOLDER_HINT} {_SETTINGS_HINT}')


def ensure_input_usable(path) -> str:
    """Check ComfyUI's input folder BEFORE a job is built, and raise a named,
    paste-safe ComfyFolderUnavailable (-> 409 + message) when it can't be used.

    Takes the path rather than resolving it, so each service keeps its own
    `_comfy_input_dir()` seam: that one stays a PURE resolver (the folder-override
    tests point it at paths that don't exist), and the checking lives here, at the
    one moment where a broken mount actually matters."""
    problem = folder_problem('input', path)
    if problem:
        raise ComfyFolderUnavailable(f'{problem} {_SETTINGS_HINT}')
    return str(path)


def stage_input_copy(src_path, dest_name, input_dir) -> str:
    """Copy `src_path` into ComfyUI's input folder as `dest_name`; return the full
    destination path. Any filesystem failure becomes a named, paste-safe 409."""
    base = str(input_dir)
    dest = os.path.join(base, dest_name)
    try:
        shutil.copy2(src_path, dest)
    except OSError as exc:
        raise _staging_error('Copying the source image into ComfyUI', 'input',
                             base, _cause(exc)) from exc
    return dest


def stage_input_image(src_path, dest_name, input_dir) -> str:
    """Stage a bounded, visual-orientation PNG for a ComfyUI workflow.

    ComfyUI may run in a different process, container, or machine. Its input
    directory is therefore a disclosure boundary, not a harmless ``copy2``:
    source EXIF/XMP/GPS must stay local and no camera master above the shared
    8192 px / 16 Mi-pixel budget may be decoded there.  The caller supplies the
    minted ``.png`` basename used by its workflow; this function returns its full
    path after an atomic write.  Local masters are read only.

    ``stage_input_copy`` remains for non-image filesystem tests/backward
    compatibility. Image-generation lanes must call this function instead.
    """
    base = str(input_dir)
    name = os.path.basename(str(dest_name or ''))
    if not name or not name.lower().endswith('.png'):
        raise ValueError('ComfyUI image staging requires a .png destination name')
    dest = os.path.join(base, name)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(src_path) as source:
                # Same shared budget as Dataset import, resolved live so a
                # raised setting reaches this staging lane too.
                image_encoding.validate_input_header_dimensions(
                    source, label='ComfyUI image staging')
                source.load()
                oriented = ImageOps.exif_transpose(source)
                has_alpha = ('A' in oriented.getbands()
                             or 'transparency' in getattr(oriented, 'info', {}))
                mode = 'RGBA' if has_alpha else 'RGB'
                # A freshly allocated canvas intentionally has no inherited
                # `info`: converting or copying alone can retain PNG/XMP fields.
                clean = Image.new(mode, oriented.size)
                clean.paste(oriented.convert(mode))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('ComfyUI image staging rejected an unsafe image header; '
                         'reduce the image before use') from exc
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f'ComfyUI image staging could not read the source image: {exc}') from exc

    tmp = f'{dest}.part-{uuid.uuid4().hex[:8]}'
    try:
        clean.save(tmp, 'PNG', compress_level=6)
        os.replace(tmp, dest)
    except OSError as exc:
        raise _staging_error('Writing the sanitized source image into ComfyUI', 'input',
                             base, _cause(exc)) from exc
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return dest


def stage_input_write(dest_name, writer, input_dir) -> str:
    """Same guarantee for callers that GENERATE the file (a PIL crop, say) instead
    of copying one: `writer(path)` does the write, everything it can raise on the
    way out is turned into the same named failure."""
    base = str(input_dir)
    dest = os.path.join(base, dest_name)
    try:
        writer(dest)
    except OSError as exc:
        raise _staging_error('Writing the source image into ComfyUI', 'input',
                             base, _cause(exc)) from exc
    return dest


# --- Un-staging: the other half of the contract ----------------------------
# Every staged copy above is a FULL-RESOLUTION duplicate of a user image, and
# nothing ever deleted one. Measured on a three-month-old install: 3 896 orphans,
# 0.67 GB, from 17/04 to the day of writing — and the count is also what ComfyUI
# enumerates for LoadImage on every prompt validation.
#
# The precise deletion happens per job (job_queue drops what the job's metadata
# records the moment the job reaches a terminal state). `prune_staged_inputs` is
# the safety net for what precise deletion cannot reach: the files already
# orphaned before this existed, and any job whose process died before completing.
#
# The sweep deletes inside a folder that belongs to ComfyUI, not to us, so it is
# fenced THREE times over and each fence is independently sufficient to spare a
# file we have no business touching:
#
#  1. NAME. Not a loose prefix — a full match on the exact shape the staging code
#     mints, `<lane>_<8 hex uid>_<original name>`. A user's own `edit_reference.png`
#     or `krea_sources.png` does not match; nothing without one of our uids does.
#  2. AGE. Nothing younger than STAGED_INPUT_MAX_AGE_SECONDS, which is set above
#     the longest a staged copy can legitimately still be waiting for its job.
#  3. LIVE JOBS. The caller passes the names every non-terminal queue row still
#     references; those are skipped whatever their age or name.
_STAGED_INPUT_RE = re.compile(
    r'^(?:edit_source|edit_ref\d+|krea_source)_[0-9a-f]{8}_'
    r'|^wmklein_(?:crop|frame|mask)_[0-9a-f]{8}\.png$')

# A staged input is dead once its job can no longer run. The worst case is a full
# fan-out queued at once (MAX_FANOUT jobs) each burning the whole poll timeout
# (15 min) before the last one starts — about 15 h. 48 h clears that with a wide
# margin, so age ALONE would already spare an input a live job still needs.
STAGED_INPUT_MAX_AGE_SECONDS = 48 * 3600


def is_staged_input_name(name) -> bool:
    """Whether `name` is one this app minted in ComfyUI's input folder.

    Deliberately strict: the input folder is shared with ComfyUI and with the
    user, and the cost of a false positive (deleting someone's image) is not
    comparable to the cost of a false negative (one stale copy survives until a
    later sweep).
    """
    return bool(_STAGED_INPUT_RE.match(os.path.basename(str(name or ''))))


def drop_staged_inputs(names, input_dir) -> int:
    """Delete staged input copies by BASENAME. Returns how many were removed.

    Basenames only, never paths: these names travel through job metadata, which
    is stored in the database and shown in diagnostics — a machine path there
    would not be paste-safe. Best-effort by design: a file already gone (a
    duplicate completion, a user who cleaned the folder) is not an error, and a
    failure to delete must never break job completion.
    """
    if not names or not input_dir:
        return 0
    removed = 0
    for name in names:
        base = os.path.basename(str(name or ''))
        if not base:
            continue
        try:
            os.remove(os.path.join(str(input_dir), base))
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning('comfy_fs: could not drop staged input %s (%s)', base, exc)
    return removed


def prune_staged_inputs(input_dir, max_age_seconds=STAGED_INPUT_MAX_AGE_SECONDS,
                        keep=None, now=None) -> int:
    """Delete staged inputs older than `max_age_seconds`. Returns the count.

    `keep` is the set of basenames still referenced by a job that has not
    reached a terminal state. They are spared unconditionally — the age fence
    alone already covers them, but a queue that took longer than anyone planned
    must not cost a user an in-flight generation.
    """
    if not input_dir:
        return 0
    now = time.time() if now is None else now
    cutoff = now - max_age_seconds
    spared = {os.path.basename(str(n)) for n in (keep or ())}
    removed = 0
    try:
        entries = os.scandir(str(input_dir))
    except OSError:
        return 0
    with entries:
        for entry in entries:
            if not is_staged_input_name(entry.name) or entry.name in spared:
                continue
            try:
                if not entry.is_file() or entry.stat().st_mtime >= cutoff:
                    continue
                os.remove(entry.path)
                removed += 1
            except OSError:
                continue
    if removed:
        logger.info('comfy_fs: pruned %d staged input copies older than %d h',
                    removed, max_age_seconds // 3600)
    return removed


# --- Claiming a just-written output ---------------------------------------
# Completion linking used to `shutil.move` ComfyUI's PNG into the dataset dir.
# Two Windows facts made that raise after the useful work was already done:
#
# 1. A rename cannot span volumes, so shutil falls back to copy + unlink.
# 2. A file ComfyUI just flushed is often still held open (ComfyUI, AV, a
#    preview). Unlink then raises WinError 32, the callback aborted, and the
#    tile stayed pending even though dest already had the bytes.
#
# Dest present is success. Source unlink is best-effort with a short retry.
# Module-level so tests can shrink the delay.
_OUTPUT_LOCK_RETRIES = 4
_OUTPUT_LOCK_RETRY_DELAY = 0.4  # seconds; ~1.2s extra only on a locked path


def _is_sharing_violation(err: OSError | None) -> bool:
    """The file is held open elsewhere: Windows sharing violation (32) / access
    denied (5), or a POSIX permission error. Same rule as trash._is_sharing_violation."""
    if err is None:
        return False
    if os.name == 'nt' and getattr(err, 'winerror', None) in (5, 32):
        return True
    return isinstance(err, PermissionError)


def claim_output_file(src, dst) -> bool:
    """Copy a just-written ComfyUI output into `dst`; delete `src` if we can.

    Returns True when ``dst`` exists (including a leftover from a previous
    attempt). Returns False when the file could not be copied — the caller
    may then fetch over ComfyUI's ``/view`` API.
    """
    src = str(src or '')
    dst = str(dst or '')
    if not dst:
        return False
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if os.path.isfile(dst) and not os.path.exists(src):
        return True
    if not src or not os.path.exists(src):
        return os.path.isfile(dst)

    copied = os.path.isfile(dst)
    last_err: OSError | None = None
    tmp = None
    for attempt in range(_OUTPUT_LOCK_RETRIES):
        try:
            tmp = f'{dst}.part-{uuid.uuid4().hex[:8]}'
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            copied = True
            tmp = None
            break
        except OSError as exc:
            last_err = exc
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                tmp = None
            if _is_sharing_violation(exc) and attempt < _OUTPUT_LOCK_RETRIES - 1:
                time.sleep(_OUTPUT_LOCK_RETRY_DELAY)
                continue
            break

    if not copied:
        if last_err is not None:
            logger.warning('comfy_fs: could not copy output %s -> %s: %s',
                           safe_path(src), safe_path(dst), last_err)
        return False

    for attempt in range(_OUTPUT_LOCK_RETRIES):
        try:
            os.unlink(src)
            break
        except FileNotFoundError:
            break
        except OSError as exc:
            if _is_sharing_violation(exc) and attempt < _OUTPUT_LOCK_RETRIES - 1:
                time.sleep(_OUTPUT_LOCK_RETRY_DELAY)
                continue
            logger.info('comfy_fs: left ComfyUI output in place (locked): %s',
                        safe_path(src))
            break
    return True
