"""HEIC/HEIF decoding for the app process, so photos from a phone import.

Pillow reads no HEIF by itself, and neither do the browsers (Safari aside), the
trainers or the OpenCV-based workers. So HEIF is decoded only where an image
ENTERS the app, in this process, and converted on arrival (see
``face_dataset_service.import_store_image``): nothing downstream ever receives a
HEIF file, which is why no worker interpreter needs this decoder.

``pi-heif`` is the decode-only build of pillow-heif (BSD binding, LGPL libheif +
libde265 wheels, no encoder). It is a core requirement on every platform it
publishes wheels for (see backend/requirements.txt). Where it is missing, a HEIC
file fails as an unreadable image, exactly as it did before this module.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_registered: bool | None = None


def register() -> bool:
    """Register the HEIF opener with Pillow, once per process.

    Returns whether HEIF can be decoded. Never raises: a missing or broken
    native wheel must not stop the app from starting.
    """
    global _registered
    if _registered is None:
        try:
            from pi_heif import register_heif_opener
            register_heif_opener()
            _registered = True
        except Exception as exc:  # noqa: BLE001 - ImportError or a DLL that fails to load
            logger.info('HEIC/HEIF decoding unavailable: %s', exc)
            _registered = False
    return _registered
