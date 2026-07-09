"""Shared Ollama vision captioning helper.

Single responsibility: run one robust vision pass on an image via the local
Ollama server and return the caption text, or "" on any failure (non-fatal for
callers). Lifted from the parent project's seedance_routes extraction so both
the classify/caption passes of the face-dataset service can reuse it without
duplicating the Qwen3-VL quirks.
"""
from __future__ import annotations

import base64
import logging
import os as _os

import requests

from .. import config as cfg

logger = logging.getLogger(__name__)


def _ollama_url() -> str:
    # Total accessor: cfg.get() can return None (missing/corrupted config
    # section) and callers rstrip('/') the result unconditionally -- this
    # must never hand back None, or the never-raise contract below breaks.
    return cfg.get('ollama.url') or 'http://127.0.0.1:11434'


def get_vision_model() -> str:
    """Resolve the Ollama vision model: env ``VISION_OLLAMA_MODEL`` > config
    ``ollama.vision_model`` (defaults to 'huihui_ai/qwen3-vl-abliterated:8b', see
    config.DEFAULTS — the ABLITERATED/uncensored Qwen3-VL, needed because the vanilla
    'qwen3-vl:8b' refuses to describe the NSFW concept datasets this app captions).
    Lets the describe quality be upgraded (e.g. to the 30B variant) without code changes."""
    env = (_os.environ.get('VISION_OLLAMA_MODEL') or '').strip()
    if env:
        return env
    return cfg.get('ollama.vision_model') or 'huihui_ai/qwen3-vl-abliterated:8b'


def describe_image_ollama(image_bytes: bytes, prompt: str, *,
                          ollama_url: str | None = None,
                          model: str | None = None,
                          num_predict: int = 800,
                          num_ctx: int = 8192,
                          repeat_penalty: float = 1.1,
                          prefer_json: bool = False,
                          fmt: str | None = None,
                          keep_alive: str | int = 0,
                          timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Describe an image via Ollama vision. Returns the caption text, or "" on
    any failure (Ollama down, timeout, empty output) -- never raises.

    `timeout` is a (connect, read) tuple by default: fail fast (10s) when Ollama
    is unreachable so a caller never hangs, but allow a long read (120s) for a
    cold model load + inference. Pass a single float to use it for both phases.

    Qwen3-VL (abliterated) ALWAYS emits a `thinking` trace and cannot be made to
    skip it (think:false / `/no_think` are ignored by this checkpoint). The trace
    alone is ~900-1400 tokens, so `num_predict` must be large enough to cover the
    thinking AND the actual answer or the response comes back empty with
    `done_reason=length`. Callers that need a clean, complete answer (e.g. an SDXL
    tag line or a scene-exhaustive prose prompt) should pass a large budget
    (num_predict>=5000) so the rich answer survives AFTER the trace. We still fall
    back to the tail of `thinking` when `response` is empty. `num_ctx` defaults to
    8192 so the longer answer plus the thinking trace fit in context (the model
    card supports large contexts).

    `keep_alive` (défaut 0) : 0 décharge le modèle après CET appel (VRAM-safe,
    bon pour les appels isolés) ; un batch (caption/classify de N images) doit
    passer une durée (ex. '5m') pour garder le modèle chaud entre les images, PUIS
    appeler unload_vision_model() en fin de batch pour rendre la VRAM à ComfyUI.
    """
    try:
        url = (ollama_url or _ollama_url()).rstrip('/')
        b64 = base64.b64encode(image_bytes).decode('ascii')
        payload = {
            'model': model or get_vision_model(),
            'prompt': prompt,
            'images': [b64],
            'stream': False,
            'options': {'temperature': 0.3, 'num_ctx': int(num_ctx),
                        'num_predict': int(num_predict),
                        'repeat_penalty': float(repeat_penalty)},
            'keep_alive': keep_alive,
        }
        # `format='json'` constrains the response to valid JSON (Ollama grammar) —
        # stops the abliterated model from rambling prose instead of the object.
        if fmt:
            payload['format'] = fmt
        resp = requests.post(f'{url}/api/generate', json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        caption = (data.get('response') or '').strip()
        thinking = (data.get('thinking') or '').strip()
        # JSON callers: the structured object may land in EITHER `response` or
        # `thinking` (this checkpoint is non-deterministic about it). Return the
        # FULL field that contains an object so the caller's JSON extractor can
        # pull it out — the last-paragraph heuristic below would truncate it.
        if prefer_json:
            for cand in (caption, thinking):
                if '{' in cand:
                    return cand
            return caption or thinking
        if caption:
            return caption
        if thinking:
            done_reason = data.get('done_reason')
            logger.info('vision_ollama: response empty, falling back to thinking (done_reason=%s)',
                        done_reason)
            parts = [p.strip() for p in thinking.split('\n\n') if p.strip()]
            if not parts:
                return thinking
            # Graceful fallback: when the answer was truncated (done_reason='length')
            # the model never emitted a clean `response`, and the usable description is
            # the tail of a LONG thinking trace. Returning only the final paragraph
            # there collapses the scene to one sentence, so when the trace is genuinely
            # long we keep the last few paragraphs instead. A short trace's last
            # paragraph is already the whole answer, so we leave that case unchanged.
            if done_reason == 'length' and len(parts) > 3:
                return '\n\n'.join(parts[-3:])
            return parts[-1]
        return ''
    except Exception as e:
        logger.warning('vision_ollama: describe skipped: %s', e)
        return ''


def unload_vision_model(*, ollama_url: str | None = None, model: str | None = None) -> bool:
    """Décharge le modèle vision d'Ollama (libère la VRAM). À appeler à la FIN d'un
    batch caption/classify (où les appels ont gardé le modèle chaud via keep_alive)
    AVANT que ComfyUI reprenne le GPU, sinon le modèle resterait chargé et ComfyUI
    pourrait manquer de VRAM. Retente une fois car un unload raté = ~5 min résident
    (keep_alive). Retourne True si l'appel a réussi."""
    try:
        url = (ollama_url or _ollama_url()).rstrip('/')
        payload = {'model': model or get_vision_model(), 'keep_alive': 0}
    except Exception as e:
        logger.warning('vision_ollama: unload url/model resolution échouée : %s', e)
        return False
    for attempt in (1, 2):
        try:
            requests.post(f'{url}/api/generate', json=payload, timeout=(10, 30))
            return True
        except Exception as e:
            logger.warning('vision_ollama: unload attempt %d échoué : %s', attempt, e)
    return False
