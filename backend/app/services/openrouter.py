"""OpenRouter image engine for the dataset generator.

WHY THIS EXISTS
---------------
The two API engines each need their own provider account: Nano Banana wants a
Google Gemini key, ChatGPT wants an OpenAI key (or a ChatGPT subscription).
People who already pay for OpenRouter — one account that fronts every provider —
had no way in (requested on GitHub #13 by jqs). This engine reaches the SAME
upstream models through that one key: OpenRouter's image API serves
`google/gemini-3-pro-image` (the Nano Banana model) and `openai/gpt-image-2`
(the ChatGPT one) among ~40 others, all through a single credit balance.

It is a THIRD engine, not a replacement: `nanobanana` and `chatgpt` are
unchanged, and which one a stored dataset row was generated with keeps its
historical value.

THE API
-------
`POST https://openrouter.ai/api/v1/images` — OpenRouter's own image endpoint
(NOT the OpenAI-compatible chat route). Reference images ride in
`input_references` as data URLs; the answer carries base64 bytes in
`data[].b64_json` with a `media_type`. Errors come back as
`{"error": {"code": …, "message": …}}`.

FAILING LOUDLY, ON PURPOSE
--------------------------
Every failure here RAISES with a named cause instead of returning None. A None
return is reserved for "the provider answered 200 and produced no image", i.e. a
refusal — which the caller already words correctly. This matters more than usual
because the user is spending money at a provider they chose: silently falling
back to another engine would bill them somewhere they did not intend, and a mute
failure would look like OpenRouter working badly rather than a key/credit/model
problem they can fix.

`OpenRouterFatal` marks the failures where EVERY other row of the batch would
fail identically (no key, rejected key, no credits, unknown model): the fan-out
catches it and stops the run instead of burning one call per row.

The API key is a secret: it only ever appears in the Authorization header. No
log line, no exception message and no error surfaced to the UI contains it or
any fragment of it.
"""
from __future__ import annotations

import base64
import logging

import requests

from .. import config as cfg
from .engine_errors import (EngineError, EngineFatal, EngineRefused,
                            provider_error_message as _error_message)

logger = logging.getLogger(__name__)

_API = 'https://openrouter.ai/api/v1/images'

# Default model: the same weights the Nano Banana engine calls directly, so
# switching engine is a change of BILLING, not of image quality. Kept as free
# text in config (engines.openrouter_model) — OpenRouter's catalogue moves fast
# and a new slug must never require a new release.
DEFAULT_MODEL = 'google/gemini-3-pro-image'

# Optional attribution headers OpenRouter uses for its app rankings. Public
# project identity only — nothing about the machine or the user.
_APP_URL = 'https://github.com/perfectgf/lora-dataset-studio'
_APP_TITLE = 'LoRA Dataset Studio'

_NO_KEY = ('no OpenRouter API key saved — add OPENROUTER_API_KEY in '
           'Settings > Image engines')


class OpenRouterError(EngineError):
    """A named OpenRouter failure. The message is user-facing and never carries
    the API key."""


class OpenRouterFatal(OpenRouterError, EngineFatal):
    """A failure that would repeat identically on every remaining row of a batch
    (missing/rejected key, no credits, unknown model). The fan-out stops the run
    on this rather than paying for the same refusal N times."""


class OpenRouterRefused(OpenRouterError, EngineRefused):
    """OpenRouter answered 200 and embedded a refusal in the body instead of an
    image. Not fatal: the next row may pass."""


def _api_key():
    return cfg.secret('OPENROUTER_API_KEY')


def get_model() -> str:
    """The configured model slug, or the default. Free text on purpose."""
    return (cfg.get('engines.openrouter_model') or '').strip() or DEFAULT_MODEL


def _raise_for_status(resp, *, ref_count: int, model: str) -> None:
    """Turn a non-200 into the most specific exception we can justify.

    401/403 = the key itself, 402 = the balance, 404 = the model slug: all three
    would fail the same way on every other row, so they are FATAL and stop the
    batch. 429 and 5xx are transient (rate limit, provider hiccup) and stay
    per-row, so one bad minute doesn't cancel a run that would have finished.

    A 400 is where a reference-count mismatch lands: models accept anywhere from
    1 to 16 references and that ceiling changes as OpenRouter's catalogue moves,
    so nothing here caps the list — we send what we were given and, if it is
    refused, SAY that the count may be the reason instead of silently dropping
    references the user expected to be used."""
    status = resp.status_code
    if status == 200:
        return
    detail = _error_message(resp)
    suffix = f': {detail}' if detail else ''
    if status in (401, 403):
        raise OpenRouterFatal(f'OpenRouter rejected the API key (HTTP {status}){suffix}')
    if status == 402:
        raise OpenRouterFatal(f'OpenRouter is out of credits (HTTP 402){suffix}')
    if status == 404:
        raise OpenRouterFatal(
            f'OpenRouter does not serve the model "{model}" (HTTP 404){suffix} — '
            'check the model in Settings > Image engines')
    if status == 429:
        raise OpenRouterError(f'OpenRouter rate-limited the request (HTTP 429){suffix}')
    if status == 400 and ref_count > 1:
        raise OpenRouterError(
            f'OpenRouter refused the request (HTTP 400){suffix} — note that '
            f'{ref_count} reference images were sent and "{model}" may accept fewer')
    raise OpenRouterError(f'OpenRouter returned HTTP {status}{suffix}')


def parse_image_response(data) -> bytes | None:
    """First image of an OpenRouter image response -> raw bytes, or None when the
    call succeeded without producing one (a refusal).

    Raises when the payload IS an image but one we cannot hand to the caller:
    the vector models answer with `image/svg+xml` inside the same b64_json field,
    and returning SVG markup where the caller expects pixels would surface much
    later as an unreadable-file error blaming the save step."""
    if not isinstance(data, dict):
        return None
    for entry in (data.get('data') or []):
        if not isinstance(entry, dict):
            continue
        b64 = entry.get('b64_json')
        if not b64:
            continue
        media = str(entry.get('media_type') or '').lower()
        if 'svg' in media:
            raise OpenRouterError(
                'OpenRouter returned a vector image (image/svg+xml) — pick a '
                'photographic model in Settings > Image engines')
        try:
            return base64.b64decode(b64)
        except (TypeError, ValueError) as e:
            raise OpenRouterError(f'OpenRouter returned an unreadable image payload: {e}')
    return None


def _raise_embedded_error(data, *, model: str) -> None:
    """OpenRouter can answer 200 and put the failure INSIDE the body — that is
    how a moderation block arrives (`error.code == 403`, with the reasons in
    `error.metadata.reasons`), and how an upstream provider outage arrives once
    the response has already started. Reading it costs nothing and is the whole
    difference between "the app came back empty" and "the model refused this,
    for this reason". Returns silently when the body says nothing: an
    unexplained blank stays unexplained rather than being given a cause."""
    if not isinstance(data, dict):
        return
    err = data.get('error')
    if not isinstance(err, dict):
        return
    msg = str(err.get('message') or '').strip()[:300]
    meta = err.get('metadata') if isinstance(err.get('metadata'), dict) else {}
    reasons = meta.get('reasons') if isinstance(meta.get('reasons'), list) else []
    reasons = ', '.join(str(r) for r in reasons if r)[:120]
    try:
        code = int(err.get('code'))
    except (TypeError, ValueError):
        code = 0
    if not (msg or reasons or code):
        return                          # an error key with nothing in it says nothing
    if code == 403 or reasons:
        detail = f' ({reasons})' if reasons else (f': {msg}' if msg else '')
        raise OpenRouterRefused(
            f'{model} refused this request{detail} — the moderation is the '
            "provider's, and LDS cannot turn it off")
    suffix = f': {msg}' if msg else ''
    raise OpenRouterError(
        f'OpenRouter answered without an image (error {code or "unknown"} in the '
        f'body){suffix} — nothing was generated for this image')


def generate_variation(ref_bytes: bytes | list[bytes], prompt: str, model: str | None = None,
                       aspect_ratio: str = '1:1') -> bytes | None:
    """Reference photo(s) + variation prompt -> generated image bytes, or None.

    Same contract as the Nano Banana and ChatGPT engines, so the whole fan-out
    stays engine-parametric. `ref_bytes` is one image or a list of images of the
    same subject, principal first; every one of them is sent.

    None now means one thing only, and a narrower one than it used to: OpenRouter
    answered 200, produced no image, and said NOTHING about why — the one case
    this engine genuinely cannot read. When the body embeds a reason (OpenRouter
    puts moderation blocks and mid-stream provider failures there, at 200), it is
    raised with that reason instead of being flattened into the same blank."""
    key = _api_key()
    if not key:
        # Deliberately an exception, not a None + warning: a missing key must
        # never read as "the provider refused your prompt".
        raise OpenRouterFatal(_NO_KEY)
    mdl = (model or '').strip() or get_model()
    refs = ref_bytes if isinstance(ref_bytes, (list, tuple)) else [ref_bytes]
    refs = [r for r in refs if r]
    references = [{'type': 'image_url',
                   'image_url': {'url': 'data:image/webp;base64,'
                                        + base64.b64encode(r).decode('ascii')}}
                  for r in refs]
    base = {'model': mdl, 'prompt': prompt}
    if references:
        base['input_references'] = references
    # Two attempts, same shape as the Nano Banana engine: aspect_ratio is a
    # normalized OpenRouter field but individual endpoints clamp to their own
    # subset and answer 400 when they don't take it at all. Retrying once without
    # it keeps a model that ignores framing usable instead of failing the row —
    # the image is still generated, just at the provider's own ratio.
    payloads = [dict(base, aspect_ratio=aspect_ratio), base]
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json',
               'HTTP-Referer': _APP_URL, 'X-OpenRouter-Title': _APP_TITLE}
    for i, payload in enumerate(payloads):
        try:
            r = requests.post(_API, headers=headers, json=payload, timeout=(10, 300))
        except requests.RequestException as e:
            raise OpenRouterError(f'could not reach OpenRouter: {e}')
        if r.status_code == 400 and i == 0 and len(payloads) > 1:
            logger.info('openrouter: %s refused aspect_ratio=%s, retrying without it',
                        mdl, aspect_ratio)
            continue                                   # retry without aspect_ratio
        _raise_for_status(r, ref_count=len(refs), model=mdl)
        try:
            data = r.json()
        except ValueError as e:
            raise OpenRouterError(f'OpenRouter returned a non-JSON response: {e}')
        img = parse_image_response(data)
        if img is None:
            logger.warning('openrouter: no image in response from %s '
                           '(safety block or text-only answer)', mdl)
            _raise_embedded_error(data, model=mdl)
            # Nothing readable in the body: it stays None, and the fan-out says
            # so in words ("a content-policy refusal and a transient API error
            # look identical here") instead of picking one.
        return img
    return None
