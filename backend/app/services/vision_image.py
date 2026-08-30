"""The one image gate every local vision provider goes through.

This lived inside ``vision_ollama`` as ``_ensure_ollama_decodable``. Nothing about
it is Ollama-specific: it exists because handing an arbitrary file to a vision
server is a DISCLOSURE, and the app promises that disclosure is pixels only. A
second provider that reimplemented — or skipped — this would quietly ship camera
EXIF/XMP/GPS to whatever endpoint the user typed, which is precisely the bug the
original docstring was written to prevent. So it moved here rather than being
copied, and both drivers import it.

``vision_ollama`` keeps its two old private names as aliases: they are referenced
by name in the existing suite, and a rename would have been churn with no reader.
"""
from __future__ import annotations

import logging
import warnings

from . import image_encoding, input_budget  # noqa: F401 - installs the shared budget

logger = logging.getLogger(__name__)

# Bounds the payload and the server-side decode cost. 1536 is well above what any
# of the supported vision models actually consumes, so this only ever trims the
# genuinely oversized.
VISION_MAX_SIDE = 1536


def ensure_vision_safe_jpeg(image_bytes: bytes, *, provider: str = 'vision') -> bytes | None:
    """Return image bytes any local vision server can definitely read, stripped.

    Every decodable format becomes a fresh, metadata-free JPEG (alpha flattened
    onto white, EXIF orientation baked, longest side bounded, quality 90). That
    makes captioning an explicit pixel-only disclosure even for source JPEG/PNG
    files. Fail closed: an undecodable or unsafe source returns ``None`` so raw
    camera bytes can never reach the configured (possibly remote) endpoint.

    ``provider`` only names who asked, for the log line — the treatment is
    identical for every provider, deliberately.
    """
    try:
        import io

        from PIL import Image, ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                # Same shared budget as Dataset import, resolved live: an image
                # the user was allowed to import must also be describable.
                image_encoding.validate_input_header_dimensions(
                    source, label='vision captioning')
                source.load()
                im = ImageOps.exif_transpose(source)
        if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != 'RGB':
            im = im.convert('RGB')
        # A fresh image has no inherited `info`; Pillow otherwise may carry
        # source metadata in surprising format-specific save paths.
        clean = Image.new('RGB', im.size)
        clean.paste(im)
        im = clean
        if max(im.size) > VISION_MAX_SIDE:
            im.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, 'JPEG', quality=90)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001 - never disclose raw bytes after a decode failure
        logger.warning('%s: refusing unsafe/unreadable image before send (%s)', provider, e)
        return None
