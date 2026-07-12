"""ChatGPT image (OpenAI gpt-image-2) variation generator for the face Dataset Maker.

Same contract as `nanobanana.generate_variation` so the dataset fan-out treats
both API engines uniformly: reference photo(s) + variation prompt -> generated
image bytes (or None). Uses the multipart /images/edits endpoint (image[0] = the
base to edit, the rest = extra identity references, 16 max). No GPU, no ComfyUI —
SFW only by provider policy (moderation 400 -> None, the row just fails).
"""
from __future__ import annotations
import base64
import json
import logging
import os
import uuid

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

# --- Subscription lane (Codex OAuth) -----------------------------------------
# EXPERIMENTAL: renders gpt-image-2 on the user's ChatGPT subscription quota via
# the Codex Responses backend. Undocumented lane — may break if OpenAI closes it.
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
# The Codex lane accepts far fewer input images than /images/edits (16).
SUBSCRIPTION_MAX_REFS = 5
SUBSCRIPTION_ROUTER_MODEL = 'gpt-5.4-mini'   # routing model only; images are gpt-image-2


class SubscriptionQuotaExceeded(RuntimeError):
    """429 on the subscription lane: the plan's image quota is exhausted, so
    every later call in the batch would fail too. Callers stop the batch —
    None (row-level failure) and this (batch-level stop) are different channels."""


class SubscriptionUnavailable(RuntimeError):
    """The subscription lane was selected but the ChatGPT connection is gone
    (token expired and refresh failed mid-batch). Stop the batch — never fall
    back to the paid API key. Distinct from SubscriptionQuotaExceeded so the
    batch can show the right message."""


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


def _generate_via_api(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                      aspect_ratio: str = '1:1') -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.
    API-key lane: the multipart /images/edits endpoint.

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


def _use_subscription() -> bool:
    mode = cfg.get('engines.chatgpt_auth') or 'auto'
    if mode == 'api':
        return False
    if mode == 'subscription':
        return True
    from . import chatgpt_oauth
    return chatgpt_oauth.status()['connected']


def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1', force_lane: str | None = None) -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.
    Routes on engines.chatgpt_auth: API key (default lane) or ChatGPT
    subscription (Codex OAuth). Raises SubscriptionQuotaExceeded on a
    subscription-quota 429, and SubscriptionUnavailable if the subscription
    lane loses its token mid-call, so batch callers can stop instead of
    burning rows / silently falling back to the paid API key.

    `force_lane`: None -> decide from engines.chatgpt_auth (single-call
    callers); 'subscription' | 'api' -> pinned lane (batch callers pin once so
    a mid-batch disconnect can't reroute later rows onto the paid API key)."""
    use_sub = (force_lane == 'subscription') or (force_lane is None and _use_subscription())
    if use_sub:
        refs = list(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
        return _generate_via_subscription(refs, prompt, aspect_ratio)
    return _generate_via_api(ref_bytes, prompt, model, aspect_ratio)


def _image_from_output(output) -> bytes | None:
    for item in output or []:
        if item.get('type') == 'image_generation_call' and item.get('result'):
            try:
                return base64.b64decode(item['result'])
            except (ValueError, TypeError):
                return None
    return None


def _parse_sse_for_image(text: str) -> bytes | None:
    """Minimal SSE walk for the terminal image event — used when the backend
    answers text/event-stream despite stream:false."""
    for line in text.splitlines():
        if not line.startswith('data:'):
            continue
        try:
            evt = json.loads(line[5:].strip())
        except ValueError:
            continue
        if evt.get('type') == 'response.completed':
            return _image_from_output((evt.get('response') or {}).get('output'))
        if evt.get('type') == 'response.output_item.done':
            img = _image_from_output([evt.get('item') or {}])
            if img:
                return img
    return None


def _generate_via_subscription(refs: list, prompt: str, aspect_ratio: str) -> bytes | None:
    from . import chatgpt_oauth
    refs = refs[:SUBSCRIPTION_MAX_REFS]              # primary first, extras ride along
    content = [{'type': 'input_image',
                'image_url': 'data:image/webp;base64,' + base64.b64encode(rb).decode('ascii')}
               for rb in refs]
    content.append({'type': 'input_text', 'text': prompt})
    body = {
        'model': cfg.get('engines.chatgpt_subscription_model') or SUBSCRIPTION_ROUTER_MODEL,
        'input': [{'role': 'user', 'content': content}],
        'tools': [{'type': 'image_generation', 'size': size_for_aspect(aspect_ratio),
                   'quality': CHATGPT_IMAGE_QUALITY, 'moderation': 'auto'}],
        'tool_choice': 'required',
        # The Codex responses backend refuses to persist responses: without this
        # it 400s with {"detail":"Store must be set to false"}.
        'store': False,
        'stream': False,
    }
    for attempt in (0, 1):                           # attempt 1 = after a forced refresh
        token = chatgpt_oauth.access_token(force_refresh=bool(attempt))
        if not token:
            raise SubscriptionUnavailable(
                'ChatGPT connection lost — reconnect in Settings')
        headers = {'Authorization': f'Bearer {token}',
                   'chatgpt-account-id': chatgpt_oauth.account_id() or '',
                   'OpenAI-Beta': 'responses=experimental',
                   'originator': 'codex_cli_rs',
                   'session_id': str(uuid.uuid4())}
        try:
            # Same generous read timeout as the API lane: 'high' renders take minutes.
            r = requests.post(CODEX_RESPONSES_URL, headers=headers, json=body,
                              timeout=(10, 420))
        except requests.RequestException as e:
            logger.warning(f"chatgpt_image: subscription request error: {e}")
            return None
        if r.status_code == 401 and attempt == 0:
            continue
        if r.status_code == 429:
            raise SubscriptionQuotaExceeded(
                'ChatGPT subscription image quota reached — rerun in API-key mode '
                'or wait for your plan quota to reset')
        if r.status_code != 200:
            # Content-policy refusals land here too — the row fails, same as the API lane.
            logger.warning(f"chatgpt_image: subscription HTTP {r.status_code}: {r.text[:300]}")
            return None
        if 'text/event-stream' in (r.headers.get('content-type') or ''):
            img = _parse_sse_for_image(r.text)
        else:
            try:
                img = _image_from_output((r.json() or {}).get('output'))
            except ValueError:
                img = None
        if img is None:
            logger.warning("chatgpt_image: no image in subscription response")
        return img
    return None
