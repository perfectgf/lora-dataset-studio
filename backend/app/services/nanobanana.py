"""Nano Banana (Gemini image API) variation generator for the face Dataset Maker.

Sends the reference photo + a variation prompt to the Gemini image model and
returns the generated image bytes. No GPU, no ComfyUI involvement — runs fully
off-device, so dataset generation can happen while local generations run.
SFW only by provider policy (fits the face-dataset use case by design).
"""
from __future__ import annotations
import base64
import logging
import os

import requests

from .. import config as cfg

logger = logging.getLogger(__name__)

# Nano Banana Pro (GA) — best-in-class face consistency. Overridable via env.
NANOBANANA_MODEL = os.environ.get('NANOBANANA_MODEL', 'gemini-3-pro-image')
_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _api_key():
    return cfg.secret('GEMINI_API_KEY')


def parse_image_response(data) -> bytes | None:
    """Extract the first inline image from a generateContent response."""
    try:
        for cand in data.get('candidates', []):
            for part in (cand.get('content') or {}).get('parts', []):
                inline = part.get('inlineData') or part.get('inline_data') or {}
                if inline.get('data'):
                    return base64.b64decode(inline['data'])
    except (TypeError, ValueError, KeyError):
        return None
    return None


def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1') -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.

    `ref_bytes` : une image (bytes) ou une LISTE d'images de la même personne
    (multi-références — gemini-3-pro-image accepte jusqu'à 14 images d'entrée et
    s'appuie sur toutes pour la cohérence d'identité). La principale en premier.
    `aspect_ratio` (ex. '1:1' visage, '3:4' buste/corps) évite de letterboxer les
    plans corps. Tries with imageConfig first (Pro models); on a 400 retries once
    with a slim payload for models that don't accept imageConfig."""
    key = _api_key()
    if not key:
        logger.warning("nanobanana: GEMINI_API_KEY missing in environment")
        return None
    mdl = model or NANOBANANA_MODEL
    refs = ref_bytes if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
    parts = [{"text": prompt}]
    for rb in refs:
        parts.append({"inlineData": {"mimeType": "image/webp",
                                     "data": base64.b64encode(rb).decode('ascii')}})
    payloads = [
        {"contents": [{"parts": parts}],
         "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                              "imageConfig": {"aspectRatio": aspect_ratio}}},
        {"contents": [{"parts": parts}],
         "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}},
    ]
    for i, payload in enumerate(payloads):
        try:
            r = requests.post(_API.format(model=mdl),
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=payload, timeout=(10, 180))
        except requests.RequestException as e:
            logger.warning(f"nanobanana: request error: {e}")
            return None
        if r.status_code == 400 and i == 0:
            continue  # retry without imageConfig
        if r.status_code != 200:
            logger.warning(f"nanobanana: HTTP {r.status_code}: {r.text[:300]}")
            return None
        img = parse_image_response(r.json())
        if img is None:
            logger.warning("nanobanana: no image in response (safety block or text-only)")
        return img
    return None
