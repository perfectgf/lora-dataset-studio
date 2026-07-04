"""ChatGPT image (OpenAI gpt-image-2) variation generator for the face Dataset Maker.

Same contract as `nanobanana.generate_variation` so the dataset fan-out treats
both API engines uniformly: reference photo(s) + variation prompt -> generated
image bytes (or None). Uses the multipart /images/edits endpoint (image[0] = the
base to edit, the rest = extra identity references, 16 max). No GPU, no ComfyUI —
SFW only by provider policy (moderation 400 -> None, the row just fails).
"""
from __future__ import annotations
import base64
import logging
import os

import requests

from .. import config as cfg

logger = logging.getLogger(__name__)

# gpt-image-2 = the only current model usable without OpenAI organization
# verification (gpt-image-1.5 / chatgpt-image-latest 403 without it).
CHATGPT_IMAGE_MODEL = os.environ.get('CHATGPT_IMAGE_MODEL', 'gpt-image-2')
# Dataset images are final training material -> default to 'high' (≈ Nano
# Banana's price point). Override with CHATGPT_IMAGE_QUALITY=medium to iterate.
CHATGPT_IMAGE_QUALITY = os.environ.get('CHATGPT_IMAGE_QUALITY', 'high')
_API = "https://api.openai.com/v1/images/edits"


def _api_key():
    return cfg.secret('OPENAI_API_KEY')


def size_for_aspect(aspect_ratio: str) -> str:
    """Map the dataset aspect strings ('1:1', '3:4', '16:9'…) onto the three
    sizes gpt-image supports. Portrait-ish -> 1024x1536, landscape-ish ->
    1536x1024, anything else (or unknown) -> square."""
    try:
        w, h = (int(x) for x in str(aspect_ratio).split(':', 1))
        if w > 0 and h > 0:
            if h > w:
                return '1024x1536'
            if w > h:
                return '1536x1024'
    except (ValueError, TypeError):
        pass
    return '1024x1024'


def parse_image_response(data) -> bytes | None:
    """Extract the first b64 image from an /images responses payload."""
    try:
        b64 = (data.get('data') or [{}])[0].get('b64_json')
        return base64.b64decode(b64) if b64 else None
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1') -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.

    `ref_bytes`: one image (bytes) or a LIST (primary first — it becomes the
    edit base; extras ride along as identity references, capped at 16 total).
    NB: gpt-image-2 does NOT accept `input_fidelity` (400) — never send it."""
    key = _api_key()
    if not key:
        logger.warning("chatgpt_image: no OpenAI API key (OPENAI_API_KEY)")
        return None
    refs = list(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
    refs = refs[:16]
    files = [('image[]', (f'ref{i}.webp', rb, 'image/webp')) for i, rb in enumerate(refs)]
    data = {
        'model': model or CHATGPT_IMAGE_MODEL,
        'prompt': prompt,
        'size': size_for_aspect(aspect_ratio),
        'quality': CHATGPT_IMAGE_QUALITY,
    }
    try:
        # 'high' renders take 1-3 min -> generous read timeout (connect stays short).
        r = requests.post(_API, headers={"Authorization": f"Bearer {key}"},
                          data=data, files=files, timeout=(10, 420))
    except requests.RequestException as e:
        logger.warning(f"chatgpt_image: request error: {e}")
        return None
    if r.status_code != 200:
        # 400 moderation/content-policy lands here too: the row fails, the user
        # can switch that shot to Klein (local) — mirrored from the skill notes.
        logger.warning(f"chatgpt_image: HTTP {r.status_code}: {r.text[:300]}")
        return None
    img = parse_image_response(r.json())
    if img is None:
        logger.warning("chatgpt_image: no image in response")
    return img
