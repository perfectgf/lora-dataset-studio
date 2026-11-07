"""ONE encoding rule for every operation that rewrites an image the user is EDITING.

The distinction this module exists to hold
------------------------------------------
The app writes image bytes for two very different reasons:

* **Edits of the working image** — mirror, rotate, crop, watermark crop, watermark
  inpainting. The result *replaces* the file the dataset row points at, and that file
  is what `shutil.copy2` hands to the trainer. Whatever is thrown away here is thrown
  away for good, and it lands in the LoRA weights.
* **Throwaway derivatives** — grid thumbnails, the ≤2048 px copy uploaded to a
  generation API. They are regenerated from the working image at will, so lossy is
  free there and staying small is the whole point.

Only the first family belongs here. Thumbnails (`quality=72`) and
`normalize_to_webp` (transport to the engines) are derivatives ON PURPOSE and must
NOT be routed through this module.

The rule
--------
Keep the format the file already has, and encode it under the policy THE CALLING
OPERATION asks for. The policy is a required argument, never a module-wide constant:
mirror and rotate carry a published byte-identity promise, and tuning the crop's
encoder must not be able to break it without the change being visible at the call
site.

Preserving the format is also what keeps the file NAME honest: dataset files are
written with their true extension at import, so rewriting a `.png` as WEBP bytes
would leave a file that lies to everything guessing by extension.

`LOSSLESS` — the only policy in use today; every caller measured its way to it.
* **PNG** — DEFLATE only, native mode kept. Lossless by construction.
* **WEBP** — `lossless=True`. Repeated edits therefore never accumulate damage.
* **JPEG** — has NO lossless mode. Re-encoding always costs something, so the honest
  move is to lose as little as possible rather than pretend: quality 95 with no
  chroma subsampling roughly halves the error of the old q92 WEBP path (measured on
  a real photograph: PSNR 47.6 dB vs 43.0 dB) at ~40% of the size a lossless WEBP of
  the same pixels would take. Converting a user's JPEG to a 2.4x heavier lossless
  file to preserve pixels that were already lossy is a bad trade.
* **BMP** — uncompressed RGB is its native, lossless path. BMP has no dependable
  alpha/ICC round trip in Pillow, so an edit deliberately writes RGB pixels only.

`HIGH_QUALITY` — the documented escape hatch for an operation that decides the size
of `LOSSLESS` is not worth it. It is deliberately NOT the crop's policy, and the
measurement is why (real photograph, 624x624 box rescaled to a 1024 long side; the
error is against the resampled image, so it isolates the ENCODING):

    WEBP q92 (the old crop) 200,756 B  1.00x  max 16   mean 1.367   PSNR 42.93 dB
    WEBP q95             264,644 B   1.32x   max 14   mean 1.084   PSNR 44.84 dB
    WEBP q98             327,570 B   1.63x   max 15   mean 0.924   PSNR 46.06 dB
    WEBP q100 (lossy!)   349,738 B   1.74x   max 16   mean 0.878   PSNR 46.41 dB
    WEBP lossless        921,912 B   4.59x   max  0   mean 0.000   exact

Lossy WEBP has an ERROR FLOOR it cannot pass: libwebp subsamples chroma to 4:2:0 at
every quality (per-channel mean error at q92 — R 1.367, G 1.200, B 1.535: blue is
worst, which is the signature). Paying 1.74x to reach q100 buys 3.5 dB and still
leaves visible drift. And the drift COMPOUNDS across a curation loop — five
successive crops of the same file:

    q92 -> PSNR 44.77 dB (max 21)      q100 -> PSNR 45.09 dB (max 19)
    lossless -> byte-identical to the first crop

More quality does not stop accumulation; only lossless does. On a dataset whose whole
purpose is to become model weights, 4.59x of disk is the cheaper side of that trade.

Why `method=4` and not 6
------------------------
Measured on a 1024x768 photograph, lossless WEBP: `method=6` takes 4.6 s and produces
777,602 bytes; `method=4` takes 0.35 s for 785,268 bytes. 13x the wall time for 1% of
the size, on an interactive click. Not worth it. (`method` is a search-effort knob, so
which one wins on size is content-dependent — on a grainier fixture method=4 came out
2% SMALLER. The time ratio is what decides, not the size.)

What this rule COSTS (measured, same photograph, 624x624 box → 1024 long side)
-----------------------------------------------------------------------------
A cropped WEBP file goes from 201,400 B to 945,746 B (4.70x); the same crop taken from
a PNG source lands at 921,912 B. Roughly 4.6-4.7x either way. That is the price of not
throwing pixels away, and it is stated in the What's-new entry rather than hidden.
"""
from __future__ import annotations

# PIL is the ONLY dependency on purpose: `backend/infer/lama_infer.py` runs under the
# dedicated ML interpreter, where the Flask app package is not importable. It loads
# this file directly by path, so nothing here may reach for the app, the DB or config.
from PIL import Image

# Shared ingress/staging safety budget. Dataset import and every ComfyUI hand-off
# use the same header limits: no route may quietly decode or disclose a larger
# camera master just because it happens after the initial upload.
#
# These are the SHIPPED DEFAULTS, not the law. The effective budget is a user
# setting (`image_input.max_side` / `image_input.max_pixels`), and this module
# cannot read it: it must keep working under the ML interpreter that loads this
# file by path. So the app INJECTS a provider (see
# `app/services/input_budget.py`) instead of this module reaching for config.
# With no provider installed — the ML interpreter, a bare unit test — the
# defaults below apply, which is the conservative side of the choice.
#
# Sized in DECODED MEMORY, which is what the guard actually protects: a decoded
# RGB pixel costs 3 bytes, RGBA 4, and an edit or analysis pass can hold a
# second copy at the same time. 64 Mi-pixels is ~192 MiB for one RGB decode
# (~256 MiB RGBA), ~384-512 MiB while a working copy exists. That admits every
# 35 mm / phone master shipping today (a 61 MP 9504x6336 is 57 Mi-pixels, a
# 48 MP phone frame 46 Mi) and ordinary stitched panoramas, without putting a
# single import near a gigabyte. 128 Mi-pixels would have doubled both numbers
# (~768 MiB for one edit) and that is not a default, it is a choice — which is
# exactly what the setting is for.
DEFAULT_INPUT_MAX_SIDE = 16384
DEFAULT_INPUT_MAX_PIXELS = 64 * 1024 * 1024

#: Back-compatible names for the shipped defaults. Read `input_budget()` instead:
#: these two do NOT move when the user changes the setting.
INPUT_MAX_SIDE = DEFAULT_INPUT_MAX_SIDE
INPUT_MAX_PIXELS = DEFAULT_INPUT_MAX_PIXELS

#: Where a user changes the budget. Quoted by the rejection message, because
#: "reduce the image before import" was the only honest advice while the number
#: was hardcoded, and is a dead end now that it is not.
INPUT_BUDGET_SETTING_PATH = 'Settings ▸ Captioning & quality ▸ Image size budget'


class ImageTooLargeError(ValueError):
    """An image refused for its size (the budget below, or a bomb-sized header).

    A ValueError like every other refusal here, so no caller changes. The type
    exists so an import can COUNT why a file was refused without reading the
    message text.
    """

_budget_provider = None


def set_input_budget_provider(provider) -> None:
    """Install the callable that resolves the effective budget, or None to reset.

    `provider()` returns `(max_side, max_pixels)`, each a non-negative int where
    **0 means no limit**. The app installs one that reads its config live, so a
    changed setting takes effect without a restart and every consumer of this
    guard sees the same number. Kept as an injection point rather than an import
    because this module may not touch the app, the DB or config (see the note
    above the `PIL` import).
    """
    global _budget_provider
    _budget_provider = provider


def input_budget() -> tuple[int, int]:
    """The effective `(max_side, max_pixels)`; 0 on either means no limit.

    Total by construction: a provider that raises or answers nonsense degrades
    to the shipped defaults rather than letting every image path fail.
    """
    provider = _budget_provider
    if provider is None:
        return DEFAULT_INPUT_MAX_SIDE, DEFAULT_INPUT_MAX_PIXELS
    try:
        max_side, max_pixels = provider()
        side = int(max_side)
        pixels = int(max_pixels)
        if side < 0 or pixels < 0:
            raise ValueError((max_side, max_pixels))
    except Exception:                              # noqa: BLE001 - never fail closed here
        return DEFAULT_INPUT_MAX_SIDE, DEFAULT_INPUT_MAX_PIXELS
    return side, pixels


def input_budget_sentence() -> str:
    """One human phrase for the effective budget, including "no limit"."""
    side, pixels = input_budget()
    parts = []
    if pixels:
        mebi = pixels / (1024 * 1024)
        shown = f'{mebi:g}'
        parts.append(f'{shown} Mi-pixels')
    if side:
        parts.append(f'{side} px per side')
    return ' and '.join(parts) if parts else 'any size (no limit)'


def validate_input_header_dimensions(im: Image.Image, *, label: str,
                                     max_side: int | None = None,
                                     max_pixels: int | None = None) -> tuple[int, int]:
    """Validate a raster header before any caller decodes pixels.

    This is the shared ingress/live-file guard.  The caller supplies a concise
    label for its user-facing error, but the byte/pixel budget is deliberately
    global so a later Bank scan, thumbnail or edit cannot materialise a source
    that Dataset import would have rejected.  `max_side`/`max_pixels` exist for
    a caller that must pin a budget explicitly (tests, a narrower staging lane);
    omitted, the shared configured budget applies. 0 on either means no limit.
    """
    try:
        width, height = im.size
        valid_size = (isinstance(width, int) and isinstance(height, int)
                      and width > 0 and height > 0)
    except (AttributeError, TypeError, ValueError, OverflowError, MemoryError) as exc:
        raise ValueError(f'{label} received an unreadable image') from exc
    if not valid_size:
        raise ValueError(f'{label} received an unreadable image')
    budget_side, budget_pixels = input_budget()
    side = budget_side if max_side is None else int(max_side)
    pixels = budget_pixels if max_pixels is None else int(max_pixels)
    if ((side and (width > side or height > side))
            or (pixels and width * height > pixels)):
        limits = ' or '.join(
            part for part in ((f'{side} px per side' if side else ''),
                              (f'{pixels} pixels' if pixels else '')) if part)
        raise ImageTooLargeError(
            f'{label} rejects images above {limits} (got {width}x{height}); '
            f'raise or remove this budget in {INPUT_BUDGET_SETTING_PATH}, '
            'or reduce the image before import')
    return width, height


def visual_size_from_header(im: Image.Image) -> tuple[int, int]:
    """Return displayed dimensions without decoding pixel data.

    ``ImageOps.exif_transpose(im).size`` looks innocent but Pillow may call
    ``load()`` while materialising the transposed image. Route badges and polling
    paths only need width/height, so inspect EXIF orientation from the header and
    swap axes for the four 90° orientations instead. Pixel-edit paths must still
    use ``ImageOps.exif_transpose`` before operating on coordinates.
    """
    width, height = im.size
    try:
        orientation = im.getexif().get(274, 1)
    except (AttributeError, OSError, ValueError):
        orientation = 1
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)

#: Formats an edit is allowed to write back as-is. Anything else is a format the
#: dataset pipeline never produces; see `source_format`.
EDITABLE_FORMATS = ('PNG', 'WEBP', 'JPEG', 'BMP')

#: Discard nothing the format allows keeping. What every edit uses today.
LOSSLESS = 'lossless'
#: The escape hatch: as good as a lossy encoder gets. Not used by any edit — see the
#: module docstring for the measurement that keeps it unused.
HIGH_QUALITY = 'high-quality'
POLICIES = (LOSSLESS, HIGH_QUALITY)

#: Extension the app writes for each format at import time (`_save_small_scrape_pair`).
#: Used by the tests that check a file's name still matches its bytes after an edit.
FORMAT_EXTENSIONS = {
    'PNG': ('.png',),
    'WEBP': ('.webp',),
    'JPEG': ('.jpg', '.jpeg'),
    'BMP': ('.bmp',),
}


def source_format(im: Image.Image, fallback: str = 'WEBP') -> str:
    """The format an edit of `im` should be written back as.

    `fallback` covers the formats an edit must not REFUSE but cannot round-trip
    sensibly (a legacy TIFF/GIF frame that somehow reached a dataset folder): those
    become lossless WEBP, which is what the pipeline used to produce anyway —
    without the lossy step. BMP is a supported static dataset format and therefore
    remains BMP.
    """
    fmt = (getattr(im, 'format', '') or '').upper()
    return fmt if fmt in EDITABLE_FORMATS else fallback


def format_for_path(path: str, im: Image.Image | None = None,
                    fallback: str = 'WEBP') -> str:
    """The format an edit writing to `path` should produce.

    The NAME wins when it promises a format we can write, because an edit that
    rewrites the whole file has no excuse to leave a `.png` holding WEBP bytes —
    and because a crop may write to a DIFFERENT destination than it read from
    (the reference editor reads the kept full-frame original and writes the
    derived crop). Falls back to the source image's own format, then `fallback`.

    The mirror deliberately does not use this: it flips pixels in place and stays
    minimal, so it inherits whatever mismatch a legacy file already had rather
    than converting it.
    """
    name = str(path).replace('\\', '/').rsplit('/', 1)[-1]
    ext = ('.' + name.rsplit('.', 1)[1].lower()) if '.' in name else ''
    for fmt, exts in FORMAT_EXTENSIONS.items():
        if ext in exts:
            return fmt
    return source_format(im, fallback) if im is not None else fallback


def save_params(im: Image.Image, fmt: str, policy: str, *,
                icc_profile: bytes | None = None):
    """Return `(image, save_kwargs)` encoding `im` as `fmt` under `policy`.

    `policy` is REQUIRED and stated by the operation, never defaulted: mirror and
    rotate publish a byte-identity promise, and a shared default would let a future
    tweak aimed at the crop break it invisibly.

    The image may be converted (mode narrowing the target format requires), so always
    save the RETURNED image, never the one passed in — and read its `.size` after the
    call if you check the encoded result against it.

    `icc_profile` must already be validated — LittleCMS chokes late on malformed
    profiles and Pillow copies arbitrary bytes through (see `_valid_icc_profile`).
    EXIF is deliberately NOT carried over: orientation is baked into the pixels by
    `ImageOps.exif_transpose` upstream, so re-attaching it would rotate twice.
    """
    if policy not in POLICIES:
        raise ValueError(f'unknown encoding policy: {policy!r}')
    fmt = (fmt or '').upper()
    kwargs: dict = {}
    if icc_profile and fmt != 'BMP':
        kwargs['icc_profile'] = icc_profile
    if fmt == 'PNG':
        # PNG has no lossy mode at all, so both policies land here. Native mode is
        # kept: the caller is responsible for narrowing it before any resampling.
        kwargs.update(compress_level=6)
    elif fmt == 'WEBP':
        # WEBP can carry alpha; RGB(A) preserves it while avoiding encoder-dependent
        # conversions for unusual legacy modes.
        im = im.convert('RGBA' if 'A' in im.getbands() else 'RGB')
        if policy == LOSSLESS:
            # method=4, not 6: measured 349 ms vs 4645 ms on a 1 MP photograph for a
            # ~1% size difference whose sign is content-dependent. Same pixels either
            # way — `method` is a search-effort knob, not a quality one.
            kwargs.update(lossless=True, quality=100, method=4)
        else:
            kwargs.update(quality=100, method=4)
    elif fmt == 'JPEG':
        im = im.convert('RGB')
        kwargs.update(quality=95 if policy == LOSSLESS else 92,
                      subsampling=0, optimize=True)
    elif fmt == 'BMP':
        # BMP is a lossless RGB container. Pillow's BMP writer does not offer a
        # reliable alpha/ICC round trip, so keep its portable core rather than
        # claiming metadata preservation we cannot prove.
        im = im.convert('RGB')
    else:
        raise ValueError(f'unsupported image format: {fmt or "unknown"}')
    return im, kwargs


def save_edit(im: Image.Image, fp, fmt: str, policy: str, *,
              icc_profile: bytes | None = None):
    """Write `im` to `fp` (path or file object) under `policy`. Returns `fmt`."""
    im, kwargs = save_params(im, fmt, policy, icc_profile=icc_profile)
    im.save(fp, fmt, **kwargs)
    return fmt


def resample_mode(im: Image.Image) -> str:
    """The mode an image must be in BEFORE a LANCZOS resize.

    Pillow silently falls back to nearest-neighbour on palette ('P') images and
    resampling a paletted crop that way is visibly worse than the lossy encoder we
    are removing. Alpha is kept when the source actually has it.
    """
    has_alpha = im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in getattr(im, 'info', {})
    return 'RGBA' if has_alpha else 'RGB'
