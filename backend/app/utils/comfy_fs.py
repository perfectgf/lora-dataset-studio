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

There is a THIRD failure, and it is the quiet one: the folder exists, this process
writes into it happily, and it is simply not the folder ComfyUI reads. Nothing
local can tell — every write succeeds, every probe is green — so the job is queued
and dies inside ComfyUI on `LoadImage -> "Invalid image file: <the file we just
wrote>"`. `comfyui_sees_input` closes that hole by ASKING the running ComfyUI,
whose answer is the same predicate the validation uses.

Every message is PASTE-SAFE: paths run through `redact_user_paths`, because these
strings are written to be dropped into a public help thread.
"""
from __future__ import annotations

import logging
import ntpath
import os
import posixpath
import re
import shutil
import time
import uuid
import warnings
from urllib.parse import urljoin

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from .redact import redact_url_secrets, redact_user_paths
from .. import config as cfg
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


# --- Is this ComfyUI's input folder, or just A folder? ----------------------
# `folder_problem` answers "can THIS process write here", which was the whole
# question while the only failures were a folder that was missing or read-only.
# It is not the whole question. A folder can exist, be writable, and belong to a
# DIFFERENT ComfyUI — a second install, a portable copy — or be shadowed by the
# one thing no amount of looking at disk can reveal: ComfyUI takes
# `--input-directory` / `--base-directory` on the COMMAND LINE and records them
# nowhere. When that happens:
#
#   * every staged write succeeds, so no OSError names anything;
#   * `probe_folder` is green, so Setup and Settings stay silent;
#   * ComfyUI validates the graph, does not find the file, and answers 400
#     `custom_validation_failed: image - Invalid image file: <name>`.
#
# That 400 is deterministic — `job_queue` never retries a WORKFLOW_INVALIDE — so
# the tile dies at once and the only trace is a JSON body in the log. Reported as
# GitHub #64 by mikemil828: Krea 2, "it stops instantly without an error message",
# and the whole answer was two folders apart inside the diagnostic report.
#
# The running ComfyUI can simply be ASKED, and its answer is the SAME predicate
# the validation uses: `/view?filename=<name>&type=input` resolves through
# `folder_paths.get_directory_by_type('input')`, which is the very
# `get_input_directory()` that `LoadImage.VALIDATE_INPUTS` checks with
# `exists_annotated_filepath`. A 404 there IS the refusal the generation would
# collect minutes later — in words, before anything is queued.
#
# HEAD, not GET: the question is existence, and a GET streams the whole PNG back
# over the loopback to answer it.

# (connect, read), both short. The work behind the answer is one `os.path.isfile`;
# a ComfyUI too busy to reply must never hold a job up, because an unanswered
# probe means "could not ask", never "not there".
VISIBILITY_TIMEOUT = (3, 5)


def comfyui_sees_input(name) -> bool | None:
    """Can the ComfyUI at `comfyui.api_url` serve `name` from ITS input folder?

    True and False are answers. None means the question could not be put — ComfyUI
    down, a build too old, a proxy that refuses HEAD, anything unexpected — and
    every caller reads None as "carry on". This check may cost a user a generation
    only when ComfyUI has SAID, in the same terms the validator uses, that it
    cannot see the file. Never raises.
    """
    base = os.path.basename(str(name or ''))
    api = (cfg.get('comfyui.api_url') or '').strip()
    if not base or not api:
        return None
    try:
        r = requests.head(urljoin(api.rstrip('/') + '/', 'view'),
                          params={'filename': base, 'type': 'input'},
                          timeout=VISIBILITY_TIMEOUT, allow_redirects=False)
    except Exception:                    # unreachable, DNS, proxy, TLS — anything
        return None
    status = getattr(r, 'status_code', None)
    if status == 200:
        return True
    if status == 404:
        return False
    # 400/403 (a name this route rejects), 405 (no HEAD), 5xx, a login page: the
    # answer is untrustworthy, which is not the same as "the file is not there".
    return None


def _comfy_folder_note() -> str:
    """One clause naming where the RUNNING ComfyUI says it keeps its input folder,
    or '' when it says nothing useful.

    REPORTED, never inferred. ComfyUI echoes its own `sys.argv` in `/system_stats`,
    so every clause below is its own words: the flags are quoted as flags and the
    script path as "runs from", because turning either into "<X>/input" would be a
    guess about somebody else's install — the same guess `parse_comfy_argv_dirs`
    refuses to make for the Setup fields. Never raises.
    """
    api = (cfg.get('comfyui.api_url') or '').strip()
    if not api:
        return ''
    try:
        r = requests.get(f'{api.rstrip("/")}/system_stats', timeout=VISIBILITY_TIMEOUT)
        if r.status_code != 200:
            return ''
        argv = ((r.json() or {}).get('system') or {}).get('argv')
    except Exception:
        return ''
    if not isinstance(argv, (list, tuple)):
        return ''
    items = [str(a) for a in argv]
    for flag in ('--input-directory', '--base-directory'):
        value = _argv_value(items, flag)
        if value:
            return f'it was started with `{flag} {safe_path(value)}`'
    # No flag: argv[0] is the main.py that is actually running, which is the
    # answer whenever the cause is a SECOND install rather than a flag.
    script = items[0] if items else ''
    if script and _is_absolute_anywhere(script) and \
            _path_module(script).basename(script).lower() == 'main.py':
        return ('the ComfyUI answering there runs from '
                f'{safe_path(_path_module(script).dirname(script))}')
    return ''


def _path_module(path: str):
    """The path flavour the STRING uses, not the one this OS runs. Everything
    parsed here was written by another process that may be on another kernel."""
    return ntpath if '\\' in path else posixpath


def _is_absolute_anywhere(value: str) -> bool:
    """Absolute under EITHER convention.

    Not `os.path.isabs`: ComfyUI may be answering from WSL or a container, so a
    Windows app routinely reads `/workspace/ComfyUI/input` — and on Windows
    `ntpath.isabs` calls that RELATIVE (a leading slash is drive-relative there;
    Python 3.13 made the rule explicit). Judging another machine's path by this
    machine's rules is how the container case — the one this module exists for —
    goes silent."""
    return ntpath.isabs(value) or posixpath.isabs(value)


def _argv_value(items, flag) -> str:
    """The absolute path given to `flag` in an argv list, both argparse spellings
    (`--flag X` and `--flag=X`), or ''.

    Relative values are dropped: they resolve against a working directory this
    process does not know. Absolute ones are returned VERBATIM — normalising a
    path that came from another machine would rewrite `/mnt/shared/input` as
    `\\mnt\\shared\\input` on a Windows reader, and this string is quoted back to
    the user as ComfyUI's own words."""
    for i, tok in enumerate(items):
        name, _, inline = tok.partition('=')
        if name != flag:
            continue
        value = (inline if inline else (items[i + 1] if i + 1 < len(items) else '')).strip().strip('"')
        if not value or (not inline and value.startswith('-')):
            continue
        if _is_absolute_anywhere(value):
            return value
    return ''


def _mismatch_message(what: str, input_dir) -> str:
    """The sentence shown when ComfyUI proves it does not read `input_dir`.

    Names BOTH sides — the folder the app used and where ComfyUI says it looks —
    because naming only one leaves the reader with the half they already knew."""
    note = _comfy_folder_note()
    api = safe_path(redact_url_secrets((cfg.get('comfyui.api_url') or '').strip()))
    return (f'ComfyUI cannot see {what}. The app used {safe_path(input_dir)}, but '
            f'the ComfyUI answering at {api} reads its input folder somewhere else'
            + (f' — {note}' if note else '') + '. '
            + SHARED_FOLDER_HINT + ' ' + _SETTINGS_HINT)


def _refuse_invisible_stage(dest, input_dir):
    """Drop the file that was just staged and raise the named refusal.

    Deleted on the way out because the job it was staged for is not going to run:
    leaving a full-resolution copy in a folder that belongs to ComfyUI is exactly
    the litter the sweep at the bottom of this module exists to clear, and there
    is no reason to create the work."""
    try:
        os.remove(dest)
    except OSError:
        pass
    raise ComfyFolderUnavailable(
        _mismatch_message('the source image the app just staged', input_dir))


def input_visibility_problem(path) -> str:
    """'' unless the running ComfyUI PROVES it does not read `path`.

    The staging guard's question, asked without a job to lose: a probe file is
    written into the folder, ComfyUI is asked whether it can serve that file, and
    the probe is removed either way. The Settings preview and the Setup wizard
    both render this, so a folder that is present, writable and simply not
    ComfyUI's is amber at configuration time instead of a dead tile an hour later.

    The probe carries a leading dot and NO image extension on purpose: ComfyUI's
    own listings skip both (`LoadImage` filters its combo by content type,
    `/internal/files` skips dotfiles), so it cannot surface in a node's dropdown
    even during the moment it exists. Never raises.
    """
    if not path:
        return ''
    probe = f'.lds-comfy-visibility-{uuid.uuid4().hex[:8]}'
    target = os.path.join(str(path), probe)
    try:
        with open(target, 'wb') as fh:
            fh.write(b'lds')
    except OSError:
        return ''            # unwritable: `folder_problem` already says so, louder
    try:
        seen = comfyui_sees_input(probe)
    finally:
        try:
            os.remove(target)
        except OSError:
            pass
    if seen is not False:
        return ''
    return _mismatch_message('a file written into this folder', path)


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
    # The file is on disk — the remaining question is whether it is on disk where
    # ComfyUI looks. Asked here rather than at each lane: this is the one funnel
    # every image lane goes through, so a lane added later inherits the guard.
    if comfyui_sees_input(dest) is False:
        _refuse_invisible_stage(dest, base)
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
    if comfyui_sees_input(dest) is False:
        _refuse_invisible_stage(dest, base)
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
