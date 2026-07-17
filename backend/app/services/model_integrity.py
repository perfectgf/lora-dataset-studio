"""Model-file integrity: is a weights file on disk a REAL, loadable model, or just
a file that happens to have the right name and extension?

Two real incidents motivate this — both passed a plain "the file exists" check,
went green in Setup, then failed cryptically far downstream:

  1. A licence-gated model downloaded from a browser WITHOUT accepting the licence
     (or without an HF token) saves the HTML gate PAGE to ``<name>.safetensors``.
     ComfyUI's UNETLoader then dies at generation time on
     ``Expecting value: line 1 column 1 (char 0)`` — ``json.loads`` choking on
     ``<!doctype html>…`` — long after the Setup step reported the model present.
  2. A partial download or a broken Stability-Matrix symlink leaves a truncated /
     empty ``<name>.safetensors``. ComfyUI loads garbage and renders SILENTLY
     distorted images with no error anywhere (the kostas212 report).

This validates only the file HEADER, never the multi-GB weight body, so it is
cheap enough (~ms) to run from the probes / preflights that already fire at the
right moments. Layout of the containers we accept:

  * ``.safetensors`` / ``.sft``: an 8-byte little-endian header length ``N``,
    then ``N`` bytes of a JSON object (the tensor index). We read the 8 bytes,
    sanity-check ``N``, confirm the file is at least ``8 + N`` long, and parse the
    JSON object (bounded — a real header is well under a megabyte; anything past
    the parse budget is accepted structurally without reading it whole).
  * ``.gguf``: the 4-byte magic ``GGUF``. ComfyUI-GGUF quantised models are common
    in this community and the app already lists ``.gguf`` alongside safetensors.

Verdicts (see the constants): ``valid`` / ``html_or_text`` (a saved HTML gate
page, a Git-LFS pointer, a JSON error body — the gate case) / ``truncated_or_garbage``
(implausible header length, unparsable header, or a file shorter than its declared
header) / ``too_small`` (structurally valid but far below a plausible floor for its
type — advisory only, never blocking) / ``missing`` (not on disk — the missing-asset
preflights own that case, this validator does not block on it).

The structural verdict is a pure function of the file bytes; a deployed model never
mutates in place. It is cached by ``(abspath, mtime_ns, size)`` so repeated probe /
preflight passes read each header at most once. The ``too_small`` floor is applied
AFTER the cache (a cheap size compare) so the same file can be judged against
different floors without a re-read."""
from __future__ import annotations

import json
import logging
import os
import struct
import threading

logger = logging.getLogger(__name__)

# Verdict codes -------------------------------------------------------------
VALID = 'valid'
HTML_OR_TEXT = 'html_or_text'
TRUNCATED_OR_GARBAGE = 'truncated_or_garbage'
TOO_SMALL = 'too_small'
MISSING = 'missing'

# A real safetensors header is well under ~10 MB; past this the 8 leading bytes
# are not a header length at all (an HTML/text/garbage file read as a uint64).
_HEADER_LEN_MAX = 100 * 1024 * 1024
# Parse the JSON header in full up to here; a larger (but structurally plausible)
# header is accepted without reading it whole — the point is never to read the GB
# of weight tensors that follow, not to fully parse a pathologically large index.
_HEADER_PARSE_BUDGET = 8 * 1024 * 1024
# One cheap upfront read that covers the GGUF magic (4 B), the safetensors length
# (8 B) and enough of the header start to sniff a text/HTML/LFS signature.
_INITIAL_READ = 512

_MAGIC_GGUF = b'GGUF'

# Test seam: the module opens files through this name so a test can wrap it and
# assert the whole (multi-GB) file is never read — only the header.
_open = open

_lock = threading.Lock()
_cache: dict = {}   # (abspath, mtime_ns, size) -> (verdict_code, reason_or_None)


def clear_cache() -> None:
    """Drop the structural-verdict cache (test hygiene; production self-invalidates
    on the (path, mtime, size) key)."""
    with _lock:
        _cache.clear()


def _human_size(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n / 1024:.1f} {unit}'.replace('.0 ', ' ')
        n /= 1024
    return f'{n:.0f} B'


def _looks_textual(head: bytes):
    """A short flavour string if ``head`` looks like text / markup rather than a
    binary model header, else None. Covers the licence-gate HTML page, a Git-LFS
    pointer (repo cloned without LFS), and a JSON error body."""
    s = head.lstrip()
    if not s:
        return None
    low = s[:64].lower()
    if low.startswith(b'version https://git-lfs'):
        return 'lfs'
    if s[:1] == b'<':                    # <!doctype html>, <html>, <?xml, <!--
        return 'html'
    if s[:1] == b'{' and b'error' in s[:400].lower():
        return 'json_error'
    return None


# Reason sentences — the actionable message a user actually needs. `name` is the
# bare filename (no path — these bubble up to a paste-safe diagnostic and to the UI).
def _reason_text(name: str, flavour: str) -> str:
    if flavour == 'lfs':
        return (f"{name} is a Git LFS pointer, not the real weights — the repo was "
                f"cloned without Git LFS. Delete it and download the actual file "
                f"(or run 'git lfs pull').")
    if flavour == 'json_error':
        return (f"{name} is not a real model file — it looks like a JSON error "
                f"response, not model weights. Delete it and re-download.")
    return (f"{name} is not a real model (looks like an HTML page — a licence-gated "
            f"download saved without accepting the licence / without a token?). "
            f"Delete it and re-download the real weights.")


def _reason_garbage(name: str) -> str:
    return (f"{name} is not a valid model file — its header is unreadable "
            f"(a truncated or corrupted download?). Delete it and re-download.")


def _reason_truncated(name: str) -> str:
    return (f"{name} is incomplete — the file is shorter than its own header says "
            f"(a partial / interrupted download?). Delete it and re-download.")


def _reason_too_small(name: str, size: int) -> str:
    return (f"{name} is only {_human_size(size)} — far smaller than a real model file "
            f"(a partial download or broken symlink?). It may be incomplete; "
            f"re-download it if generation looks wrong.")


def _structural(path: str, size: int):
    """(verdict_code, reason_or_None) from the file HEADER only. One of VALID /
    HTML_OR_TEXT / TRUNCATED_OR_GARBAGE. Never raises; never reads the weight body."""
    name = os.path.basename(path)
    try:
        with _open(path, 'rb') as fh:
            head = fh.read(_INITIAL_READ)
            if len(head) < 8:
                # Not even room for a length prefix — an empty stub or a broken symlink.
                flavour = _looks_textual(head)
                return ((HTML_OR_TEXT, _reason_text(name, flavour)) if flavour
                        else (TRUNCATED_OR_GARBAGE, _reason_garbage(name)))
            if head[:4] == _MAGIC_GGUF:
                return VALID, None
            n = struct.unpack('<Q', head[:8])[0]
            if n <= 0 or n > _HEADER_LEN_MAX:
                # The 8 leading bytes aren't a plausible header length → this is not
                # a safetensors. Is it a recognisable text/HTML/LFS file (the gate case)?
                flavour = _looks_textual(head)
                return ((HTML_OR_TEXT, _reason_text(name, flavour)) if flavour
                        else (TRUNCATED_OR_GARBAGE, _reason_garbage(name)))
            if size < 8 + n:
                # Declared header doesn't fit in the file → truncated download.
                return TRUNCATED_OR_GARBAGE, _reason_truncated(name)
            want = min(n, _HEADER_PARSE_BUDGET)
            body = head[8:8 + want]
            if len(body) < want:
                body += fh.read(want - len(body))
            if body.lstrip()[:1] != b'{':
                # Plausible-looking length but the header isn't a JSON object.
                flavour = _looks_textual(head)
                return ((HTML_OR_TEXT, _reason_text(name, flavour)) if flavour
                        else (TRUNCATED_OR_GARBAGE, _reason_garbage(name)))
            if n > _HEADER_PARSE_BUDGET:
                # Structurally a JSON object of a plausible declared size, too large to
                # parse whole cheaply — accept it rather than read megabytes of index.
                return VALID, None
            try:
                obj = json.loads(body[:n].decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                return TRUNCATED_OR_GARBAGE, _reason_garbage(name)
            if not isinstance(obj, dict):
                return TRUNCATED_OR_GARBAGE, _reason_garbage(name)
            return VALID, None
    except OSError:
        return TRUNCATED_OR_GARBAGE, _reason_garbage(name)


def _cached_structural(path: str, st: os.stat_result):
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    res = _structural(path, st.st_size)
    with _lock:
        _cache[key] = res
    return res


def _result(verdict: str, path: str, size: int, reason) -> dict:
    return {
        'verdict': verdict,
        'ok': verdict == VALID,
        # Only html_or_text and truncated_or_garbage BLOCK — they mean the file
        # can't be loaded at all. too_small is advisory; missing is the other
        # preflight's job.
        'blocking': verdict in (HTML_OR_TEXT, TRUNCATED_OR_GARBAGE),
        'reason': reason,
        'filename': os.path.basename(path),
        'size': size,
    }


def validate_model_file(path, min_bytes=None) -> dict:
    """Header-only integrity verdict for a single model file. Returns a dict:
    ``{verdict, ok, blocking, reason, filename, size}`` (JSON-safe, so it drops
    straight into the capabilities payload / diagnostic).

    ``min_bytes`` — an optional lower bound for this file's TYPE. A structurally
    valid file below it is downgraded to ``too_small`` (advisory, ``blocking``
    False): a real model is orders of magnitude larger, so this catches a partial
    download / broken symlink that still happens to carry a complete small header.
    Absent ``min_bytes`` never yields ``too_small``.

    Never raises. ``valid``/``too_small`` mean "load will work" / "load will work
    but the file is suspiciously small"; ``blocking`` verdicts must stop the run
    with ``reason`` shown to the user."""
    try:
        st = os.stat(path)
    except OSError:
        name = os.path.basename(str(path))
        return _result(MISSING, str(path), 0, f'{name} is not on disk.')
    size = st.st_size
    verdict, reason = _cached_structural(str(path), st)
    if verdict == VALID and min_bytes and size < min_bytes:
        return _result(TOO_SMALL, str(path), size, _reason_too_small(os.path.basename(str(path)), size))
    return _result(verdict, str(path), size, reason)
