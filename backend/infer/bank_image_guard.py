"""Self-contained input guard for live Image Bank inference workers.

The Bank source directory is deliberately live: a file accepted by the Flask
parent can be replaced before its path reaches a dedicated ML interpreter.
Workers therefore must consume *only* the bounded bytes returned here, never
re-open the original path for decoding.  This module intentionally depends on
Pillow plus the standard library only; the infer environments do not import the
Flask app or its services.
"""
from __future__ import annotations

import contextlib
import io
import os
import stat
import warnings

from PIL import Image, UnidentifiedImageError


MAX_FILE_BYTES = 96 * 1024 * 1024
MAX_SIDE = 8192
MAX_PIXELS = 16 * 1024 * 1024
STATIC_FORMATS = frozenset(("JPEG", "PNG", "WEBP", "BMP"))

#: The parent app's configured image input budget (`image_input.max_side` /
#: `image_input.max_pixels`), handed down as environment because this module runs
#: in a SEPARATE interpreter: the provider-injection trick `input_budget` uses for
#: `image_encoding` (same process, loaded by path) cannot reach across a
#: subprocess boundary. Absent or unusable -> the constants above, which is the
#: conservative side of the split.
#:
#: This existed as a real defect, not a tidiness gap: the app's budget was made
#: configurable (shipped default 64 Mi-pixels / 16384 px) precisely so a 24 MP
#: camera master could be imported and looked at, while these constants stayed at
#: the OLD fixed 16 Mi-pixels / 8192 px. A dataset of DSLR photos therefore
#: imported, displayed and trained fine but could not be CAPTIONED: JoyCaption
#: refused 52 of 89 images with "rejects images above 8192 px per side or
#: 16777216 pixels", one refusal per image, and the run ended reporting only the
#: 37 it wrote.
ENV_MAX_SIDE = "LDS_INFER_MAX_SIDE"
ENV_MAX_PIXELS = "LDS_INFER_MAX_PIXELS"


class BankImageGuardError(ValueError):
    """The supplied live Bank path cannot safely enter an infer worker."""


def _budget_from_env(name: str, default: int) -> int:
    """One non-negative budget number from the environment; 0 means NO limit.

    Anything unusable falls back to the shipped constant rather than to "no
    limit": a typo in an environment variable must never silently disarm a memory
    guard.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def effective_limits() -> tuple[int, int]:
    """The `(max_side, max_pixels)` this worker enforces right now; 0 = no limit."""
    return (_budget_from_env(ENV_MAX_SIDE, MAX_SIDE),
            _budget_from_env(ENV_MAX_PIXELS, MAX_PIXELS))


def _too_large_message(width: int, height: int,
                       max_side: int, max_pixels: int) -> str:
    """Name only the limits that are actually armed — a disabled one (0) must not
    appear in the sentence as "above 0 px"."""
    limits = []
    if max_side:
        limits.append(f"{max_side} px per side")
    if max_pixels:
        limits.append(f"{max_pixels} pixels")
    return (f"bank image rejects images above {' or '.join(limits)} "
            f"(got {width}x{height})")


def _read_bounded_regular_file(path: str) -> bytes:
    """Return exactly one regular file's bytes without retaining over 96 MiB.

    The descriptor is checked rather than the path alone, so an atomic
    replacement between a parent's preflight and this worker is harmless: the
    bytes inspected below are the only bytes later given to cv2/Pillow.  A
    one-byte post-read probe catches a file that grew after ``fstat`` without
    appending that byte to the retained payload.
    """
    try:
        with open(path, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise BankImageGuardError("bank image is not a regular file")
            if info.st_size > MAX_FILE_BYTES:
                raise BankImageGuardError(
                    f"bank image is too large (max {MAX_FILE_BYTES // (1024 * 1024)} MiB)")
            payload = source.read(MAX_FILE_BYTES)
            # A concurrent append can invalidate the fstat result.  Do not add
            # this sentinel byte to ``payload``: peak retained image data stays
            # at the published 96 MiB ceiling.
            if source.read(1):
                raise BankImageGuardError(
                    f"bank image is too large (max {MAX_FILE_BYTES // (1024 * 1024)} MiB)")
    except BankImageGuardError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise BankImageGuardError("bank image is unavailable") from exc
    return payload


@contextlib.contextmanager
def _pillow_bomb_threshold(max_pixels: int):
    """Raise (never lower) Pillow's bomb threshold for one header check.

    ``max_pixels == 0`` is the app's "no limit" and removes the threshold for the
    scope, which is exactly what the Settings card warns a 0 does.
    """
    previous = Image.MAX_IMAGE_PIXELS
    # ``None`` already means "no threshold": that is the highest setting there is,
    # so a finite budget must not replace it.
    target = None if (max_pixels == 0 or previous is None) else max(previous, max_pixels)
    changed = target != previous
    if changed:
        Image.MAX_IMAGE_PIXELS = target
    try:
        yield
    finally:
        if changed:
            Image.MAX_IMAGE_PIXELS = previous


def _validate_image_bytes(payload: bytes) -> None:
    """Validate a raster header/content without materialising its pixels."""
    max_side, max_pixels = effective_limits()
    try:
        # Pillow's default decompression-bomb policy is process-global.  Infer
        # workers must not alter it for another task in the same interpreter, so
        # promote the warning only inside this short header-validation scope.
        #
        # Pillow's own threshold (~89 Mi-pixels) would otherwise overrule a larger
        # configured budget with a DIFFERENT message ("bank image is invalid"),
        # exactly the trap `input_budget._align_pillow_bomb_threshold` exists to
        # avoid in the app process.  Raise it for this scope only, never lower it,
        # and always put it back.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with _pillow_bomb_threshold(max_pixels), Image.open(io.BytesIO(payload)) as image:
                fmt = (image.format or "").upper()
                if fmt not in STATIC_FORMATS:
                    raise BankImageGuardError(
                        f"bank image has unsupported format: {fmt or 'unknown'}")
                if getattr(image, "n_frames", 1) != 1:
                    raise BankImageGuardError("bank image must be a static image")
                width, height = image.size
                if (not isinstance(width, int) or not isinstance(height, int)
                        or width <= 0 or height <= 0):
                    raise BankImageGuardError("bank image is invalid")
                if ((max_side and (width > max_side or height > max_side))
                        or (max_pixels and width * height > max_pixels)):
                    raise BankImageGuardError(
                        _too_large_message(width, height, max_side, max_pixels))
                # ``verify`` validates container integrity without asking a
                # caller to decode a second, path-raced file.
                image.verify()
    except BankImageGuardError:
        raise
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise BankImageGuardError("bank image is invalid") from exc


def read_validated_bank_image(path: str) -> bytes:
    """Read and validate one live Bank path, returning its safe decode snapshot.

    Callers must pass the returned bytes to their decoder (``cv2.imdecode`` or
    ``Image.open(BytesIO(...))``).  Re-opening ``path`` afterwards would restore
    the time-of-check/time-of-use race this guard exists to remove.
    """
    payload = _read_bounded_regular_file(path)
    _validate_image_bytes(payload)
    return payload
