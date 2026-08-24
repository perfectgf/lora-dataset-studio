"""Nano Banana (Gemini image API) variation generator for the face Dataset Maker.

Sends the reference photo + a variation prompt to the Gemini image model and
returns the generated image bytes. No GPU, no ComfyUI involvement — runs fully
off-device, so dataset generation can happen while local generations run.
SFW only by provider policy (fits the face-dataset use case by design).

CHOOSING THE MODEL
------------------
`engines.nanobanana_model` is free text (Settings ▸ Image engines): Google ships
image models faster than this app ships releases, and a dropdown frozen into a
build would be stale the day it lands. Resolution order, read at CALL time so a
change in Settings applies without a restart:

    engines.nanobanana_model  >  NANOBANANA_MODEL (env)  >  DEFAULT_MODEL

The environment variable is deliberately still honoured, and above the built-in
default: it existed before the setting did and some installs set it. It is only
overridden when the user actually types a slug in Settings — i.e. by an explicit,
more recent choice. This is why the config default is BLANK rather than a copy of
DEFAULT_MODEL (see config.DEFAULTS['engines']).

FAILING LOUDLY
--------------
Failures raise a NAMED cause (see engine_errors) instead of returning None, and
the ones that would refuse every remaining row identically — rejected key,
unknown model, a model that cannot take reference images — are FATAL and stop the
batch. That matters most for the model field: a text-only or non-existent slug
used to surface as "empty response (often a content-policy refusal)", sending the
user to rewrite a prompt when the fix was one word in Settings.

THE OUTPUT FILTER (why a refusal is its own outcome)
----------------------------------------------------
Gemini screens the image it just produced, and when that screen trips the API
answers **200 OK with no image at all** — same shape as a success, minus the
picture. This engine used to hand that back as `None`, which the fan-out worded
as "empty response (often a content-policy refusal or a transient API error -
retry usually works)". That sentence guessed, and it guessed in the direction
that costs the most: a user whose request Google will refuse every time was told
to retry.

Three facts about this filter, established and worth stating plainly because
they change what the honest message is:

* It is **not configurable.** The four adjustable `safetySettings` categories act
  on the PROMPT. Nothing in the API — `BLOCK_NONE`, `OFF`, any threshold —
  turns off the screen on the returned image, and Google does not document it.
  LDS therefore cannot offer a switch, and must not imply one exists.
* It has **many false positives.** Ordinary requests get refused; the trip point
  is not something a user can reason about from the prompt text.
* It is **not deterministic.** The same prompt can pass on one call and be
  refused on the next. So "try again" is a coin toss, not a fix, and this code
  says so instead of promising a workaround it cannot deliver.

Separately: Google's usage policy forbids adult content on this engine, up to
and including account restriction. LDS does not route NSFW variations here (the
fan-out is fail-closed on that), and the refusal text names the policy rather
than hinting at a way around it.

`refusal_message()` is a pure function over the response body precisely so this
wording is testable without ever calling the real API.

WHAT EACH RETURN MEANS
----------------------
Refusals raise `NanoBananaRefused` (an `EngineRefused`, therefore NOT fatal — the
batch runs to the end and counts them). Malfunctions raise the named errors
above. `None` is now unreachable on this engine and only survives as the shared
signature. The API key never appears in a message or a log line.
"""
from __future__ import annotations
import base64
import logging
import os

import requests

from .. import config as cfg
from .engine_errors import (EngineError, EngineFatal, EngineRefused,
                            provider_error_message as _error_message)

logger = logging.getLogger(__name__)

# Nano Banana Pro (GA) — best-in-class face consistency. This is the FALLBACK,
# not a lock: see get_model() for the resolution order.
DEFAULT_MODEL = 'gemini-3-pro-image'
_ENV_VAR = 'NANOBANANA_MODEL'
_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_NO_KEY = ('no Gemini API key saved — add GEMINI_API_KEY in '
           'Settings > Image engines')

# Fragments Gemini uses when the request is wrong ABOUT THE MODEL rather than
# about this particular prompt: an unknown slug, a model that cannot answer with
# an image, or one that refuses image inputs (the dataset generator always sends
# reference images, so a text-only model can never work here). Matching on the
# provider's own words keeps a genuine per-prompt 400 from cancelling a batch.
_MODEL_FAULT_HINTS = (
    'modalit',              # "does not support the requested response modalities: image"
    'image input',
    'input image',
    'inline_data', 'inlinedata',
    'is not found', 'not found', 'not supported', 'unsupported',
    'does not support', 'is not supported',
)


class NanoBananaError(EngineError):
    """A named Nano Banana failure. User-facing text, never carries the key."""


class NanoBananaFatal(NanoBananaError, EngineFatal):
    """A failure that would repeat on every remaining row (rejected key, unknown
    model, a model that cannot take reference images)."""


class NanoBananaRefused(NanoBananaError, EngineRefused):
    """Gemini answered 200 and returned no image: its filter refused this one.
    Per-request, so the batch continues (see EngineRefused)."""


# finishReason values that mean "we made something and then refused to hand it
# over", plus the prompt-side blocks. Read as a MEMBERSHIP test with an unknown
# fallback: Google adds reasons without notice, and an unrecognised one must
# still read as a refusal (a 200 with no image is never a success) rather than
# fall through to a wrong sentence about the network.
_REFUSAL_FINISH_REASONS = {
    'IMAGE_SAFETY', 'SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'RECITATION',
    'IMAGE_PROHIBITED_CONTENT', 'IMAGE_RECITATION', 'IMAGE_OTHER', 'SPII',
}
# Said once, everywhere a refusal is reported. Deliberately claims nothing about
# a remedy: there is none to offer (see the module docstring).
_FILTER_IS_FIXED = 'not configurable — LDS cannot turn it off'


def _first(d, *names):
    """Gemini's REST envelope is camelCase, its protos are snake_case, and both
    spellings have been observed in the wild. Read either."""
    if not isinstance(d, dict):
        return None
    for n in names:
        v = d.get(n)
        if v not in (None, ''):
            return v
    return None


def refusal_detail(data) -> dict:
    """What a 200-with-no-image body actually says, as plain values.

    Returns {'scope', 'reason', 'text'} where scope is 'prompt' (Google refused
    the request before generating), 'image' (it generated, then withheld) or
    'unknown'. `reason` is Google's own code when it gave one. `text` is the
    model's written answer when it replied in words instead of pixels."""
    out = {'scope': 'unknown', 'reason': '', 'text': ''}
    if not isinstance(data, dict):
        return out
    feedback = _first(data, 'promptFeedback', 'prompt_feedback') or {}
    blocked = _first(feedback, 'blockReason', 'block_reason')
    if blocked:
        # Prompt-side: the request never reached image generation. This IS the
        # half of the safety stack the API exposes, so it is worth telling apart.
        out['scope'] = 'prompt'
        out['reason'] = str(blocked).strip()[:60]
        return out
    candidates = data.get('candidates')
    for cand in candidates if isinstance(candidates, list) else []:
        if not isinstance(cand, dict):
            continue
        reason = _first(cand, 'finishReason', 'finish_reason')
        if reason:
            reason = str(reason).strip()[:60]
            out['reason'] = reason
            if reason.upper() in _REFUSAL_FINISH_REASONS:
                out['scope'] = 'image'
        content = cand.get('content')
        parts = content.get('parts') if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else []:
            txt = isinstance(part, dict) and part.get('text')
            if txt and not out['text']:
                out['text'] = ' '.join(str(txt).split())[:160]
        if out['scope'] == 'image':
            break
    return out


def refusal_message(data) -> str:
    """The user-facing sentence for a 200 that carried no image.

    Kept SHORT on purpose: it lands in a dataset tile, which clamps to a few
    lines. The full explanation of why there is no workaround lives in the app's
    help topic and the README — this line's job is to name the cause correctly
    and never to invent a remedy."""
    d = refusal_detail(data)
    named = f' ({d["reason"]})' if d['reason'] else ''
    if d['scope'] == 'prompt':
        return (f'Google blocked the prompt before generating{named}. '
                'Nothing was produced.')
    if d['scope'] == 'image':
        return (f"Google's image filter refused this image{named}. "
                f'That filter is {_FILTER_IS_FIXED}.')
    if d['text']:
        # A text-only answer is a different animal from a filter block: the
        # model chose to reply in words. Relaying them beats paraphrasing.
        return f'Gemini answered with text instead of an image: "{d["text"]}"'
    return (f"Google's image filter refused this image{named or ' (no reason given)'}. "
            f'That filter is {_FILTER_IS_FIXED}.')


def _api_key():
    return cfg.secret('GEMINI_API_KEY')


def get_model() -> str:
    """The model this engine will ask for: setting > env var > built-in default.

    Read fresh on every call — a slug typed in Settings must apply to the very
    next generation, with no restart."""
    return ((cfg.get('engines.nanobanana_model') or '').strip()
            or (os.environ.get(_ENV_VAR) or '').strip()
            or DEFAULT_MODEL)


def _raise_for_status(resp, *, model: str) -> None:
    """Turn a non-200 into the most specific exception we can justify.

    401/403 (the key) and 404 (the slug) would fail every other row the same way,
    so they are FATAL. 429 and 5xx are transient and stay per-row, so one bad
    minute never cancels a run that would have finished. A 400 is the interesting
    one: it is fatal only when Gemini's message blames the MODEL — a request the
    user can never fix by retrying — and stays per-row otherwise."""
    status = resp.status_code
    if status == 200:
        return
    detail = _error_message(resp)
    suffix = f': {detail}' if detail else ''
    if status in (401, 403):
        raise NanoBananaFatal(f'Gemini rejected the API key (HTTP {status}){suffix}')
    if status == 404:
        raise NanoBananaFatal(
            f'Gemini does not serve the model "{model}" (HTTP 404){suffix} — '
            'check the model in Settings > Image engines')
    if status == 429:
        raise NanoBananaError(f'Gemini rate-limited the request (HTTP 429){suffix}')
    if status == 400 and any(h in detail.lower() for h in _MODEL_FAULT_HINTS):
        raise NanoBananaFatal(
            f'Gemini refused the request for model "{model}" (HTTP 400){suffix} — '
            'this engine always sends your reference images with the prompt, so '
            'the model must be an IMAGE model that accepts image input; check the '
            'model in Settings > Image engines')
    raise NanoBananaError(f'Gemini returned HTTP {status}{suffix}')


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
    with a slim payload for models that don't accept imageConfig.

    Every outcome that is not an image RAISES with the cause named: a filter
    refusal as NanoBananaRefused (per-request, the batch continues), a
    malfunction as NanoBananaError, and a cause that would repeat on every row as
    NanoBananaFatal. The `| None` in the signature is the shared engine contract,
    kept so the three engines stay interchangeable; this one no longer uses it."""
    key = _api_key()
    if not key:
        # An exception, not None: a missing key must never read to the user as
        # "the provider refused your prompt".
        raise NanoBananaFatal(_NO_KEY)
    mdl = (model or '').strip() or get_model()
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
            raise NanoBananaError(f'could not reach Gemini: {e}')
        if r.status_code == 400 and i == 0:
            continue  # retry without imageConfig
        _raise_for_status(r, model=mdl)
        try:
            data = r.json()
        except ValueError as e:
            raise NanoBananaError(f'Gemini returned a non-JSON response: {e}')
        img = parse_image_response(data)
        if img is None:
            # 200 + no image = Gemini declined. Raise instead of returning None
            # so the reason travels: the caller can tell a refusal from an
            # outage, and count it as one. Never fatal — the filter is not
            # deterministic, so the remaining rows still get their chance.
            msg = refusal_message(data)
            logger.warning("nanobanana: %s refused a request (%s)", mdl,
                           refusal_detail(data).get('reason') or 'no reason given')
            raise NanoBananaRefused(msg)
        return img
    return None
