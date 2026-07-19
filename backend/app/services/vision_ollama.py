"""Shared Ollama vision captioning helper.

Single responsibility: run one robust vision pass on an image via Ollama and
return the caption text. Ordinary best-effort calls return "" on failure;
caption batches that request local auto-start receive a clear exception if the
server still cannot caption. Lifted from the parent project's seedance_routes extraction so both
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
    return (cfg.get('ollama.url') or 'http://127.0.0.1:11434').strip().rstrip('/')


def _ollama_error_detail(exc: Exception) -> str:
    """Pull Ollama's own explanation out of a failed request. Ollama always
    answers a rejected /api/generate with a JSON body ({"error": "..."}) — e.g.
    a model with no image support, an architecture an older Ollama can't load, or
    a model that isn't pulled — but requests' HTTPError carries only the status
    line, so the reason is on the attached Response, not the exception string.

    Returns '' when there is no HTTP response at all (a connection/timeout error,
    where `exc.response` is None) so the caller can tell "Ollama rejected this"
    (has a body) from "Ollama was unreachable" (no response). Never raises."""
    resp = getattr(exc, 'response', None)
    if resp is None:
        return ''
    try:
        body = resp.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        msg = (body.get('error') or '').strip()
        if msg:
            return msg
    # Non-JSON body, or JSON without an 'error' key: fall back to the raw text so
    # the user still sees *something* actionable rather than a bare status code.
    try:
        text = (resp.text or '').strip()
    except Exception:
        text = ''
    return text[:300]


def _ollama_reject_message(exc: Exception) -> str:
    """User-facing one-liner when Ollama actively REJECTED a request (the server
    answered with a 4xx/5xx). '' when the failure wasn't an HTTP rejection (e.g. a
    connection error, which has no response/status) — the caller then keeps its
    unreachable/restart handling. Shape: 'Ollama rejected the request (HTTP 400):
    <exact error>' so the user can act without opening the log."""
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status is None:
        return ''
    detail = _ollama_error_detail(exc)
    return (f'Ollama rejected the request (HTTP {status}): {detail}' if detail
            else f'Ollama rejected the request (HTTP {status})')


# Ollama decodes the image bytes SERVER-SIDE before handing pixels to the model.
# Which decoder runs depends on the model's runtime: the llama.cpp runner (used by the
# GGUF vision models most users pull, e.g. huihui_ai/qwen3-vl-abliterated) loads images
# with stb_image, which handles JPEG/PNG/BMP/GIF/TGA but NOT WebP; Ollama's native Go
# engine only decodes the formats registered in image.Decode (gif/jpeg/png) unless the
# build blank-imports x/image/webp. Our datasets are stored as WebP (normalize_to_webp)
# and the Studio "Describe" pass also re-encodes to WebP, so on those builds the request
# fails with HTTP 400 "Failed to load image or audio file" (the exact llama.cpp reject) —
# issue #6, theotherbox122 on Ollama 0.32.0. Re-encoding anything that isn't already
# JPEG/PNG to JPEG at this single seam makes captioning work regardless of the Ollama /
# runner version, and passes JPEG/PNG through untouched so a batch of hundreds of images
# never pays for a needless re-encode.
_JPEG_MAGIC = b'\xff\xd8\xff'
_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
# Ollama vision encoders downscale to a small grid anyway; bound the JPEG we send so a
# huge source image (a raw import, not the ~1024px normalized dataset webp) doesn't bloat
# the base64 payload. Only applied when we already have the pixels open to re-encode.
_OLLAMA_MAX_SIDE = 1536


def _ensure_ollama_decodable(image_bytes: bytes) -> bytes:
    """Return image bytes Ollama's server-side decoder can definitely read.

    JPEG and PNG pass through byte-for-byte (both stb_image and Go's image.Decode read
    them, and skipping the re-encode keeps batch captioning cheap). Everything else —
    WebP above all — is re-encoded to JPEG (alpha flattened onto white, longest side
    bounded, quality 90). Best-effort: an undecodable blob is returned unchanged so the
    caller still surfaces Ollama's own error rather than crashing the never-raise
    contract."""
    head = image_bytes[:8] if image_bytes else b''
    if head.startswith(_JPEG_MAGIC) or head.startswith(_PNG_MAGIC):
        return image_bytes
    try:
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(image_bytes))
        im.load()
        if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != 'RGB':
            im = im.convert('RGB')
        if max(im.size) > _OLLAMA_MAX_SIDE:
            im.thumbnail((_OLLAMA_MAX_SIDE, _OLLAMA_MAX_SIDE), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, 'JPEG', quality=90)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001 - not-an-image / truncated: fall through to Ollama's error
        logger.warning('vision_ollama: could not re-encode image to JPEG (%s); '
                       'sending original bytes', e)
        return image_bytes


def get_vision_model() -> str:
    """Resolve the Ollama vision model: env ``VISION_OLLAMA_MODEL`` > config
    ``ollama.vision_model`` (defaults to 'huihui_ai/qwen3-vl-abliterated:8b-instruct', see
    config.DEFAULTS — the ABLITERATED/uncensored Qwen3-VL, needed because the vanilla
    'qwen3-vl:8b' refuses to describe the NSFW concept datasets this app captions).
    CRITICAL: use the '-instruct' tag, NOT plain ':8b' (which resolves to the THINKING
    variant). The Thinking model reasons out loud in the response on caption/omission tasks
    ("So the shot type is... Wait, is that the shared element?") - benchmarked 2/8 usable vs
    8/8 for -instruct, and ~8x slower (13s vs 1.6s/image). The 30b-a3b-instruct ties -instruct
    on quality at 3x the VRAM, so -instruct is the default; upgrade via config without code."""
    env = (_os.environ.get('VISION_OLLAMA_MODEL') or '').strip()
    if env:
        return env
    return cfg.get('ollama.vision_model') or 'huihui_ai/qwen3-vl-abliterated:8b-instruct'


def describe_image_ollama(image_bytes: bytes, prompt: str, *,
                          ollama_url: str | None = None,
                          model: str | None = None,
                          num_predict: int = 800,
                          num_ctx: int = 8192,
                          repeat_penalty: float = 1.1,
                          prefer_json: bool = False,
                          fmt: str | None = None,
                          keep_alive: str | int = 0,
                          auto_start_local: bool = False,
                          timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Describe an image via Ollama vision. Returns the caption text, or "" on
    failure for ordinary best-effort calls. With ``auto_start_local=True``, a
    stopped local server is started once and a persistent failure raises a
    user-facing RuntimeError.

    `timeout` is a (connect, read) tuple by default: fail fast (10s) when Ollama
    is unreachable so a caller never hangs, but allow a long read (120s) for a
    cold model load + inference. Pass a single float to use it for both phases.

    Model variant matters: the default is now the `-instruct` tag (NON-thinking) — it
    answers directly, no reasoning trace, so a modest `num_predict` suffices. The
    `-thinking` / plain `:8b` variant instead ALWAYS emits a `thinking` trace (~900-1400
    tokens) that can't be skipped (think:false / `/no_think` are ignored by that
    checkpoint); with it, `num_predict` must be large enough to cover the thinking AND the
    answer (>=5000) or the response comes back empty with `done_reason=length`. We still
    fall back to the tail of `thinking` when `response` is empty (harmless with instruct —
    that field is empty — and correct for the thinking variant). `num_ctx` defaults to 8192
    so a long answer (plus any thinking trace) fits in context.

    `keep_alive` (défaut 0) : 0 décharge le modèle après CET appel (VRAM-safe,
    bon pour les appels isolés) ; un batch (caption/classify de N images) doit
    passer une durée (ex. '5m') pour garder le modèle chaud entre les images, PUIS
    appeler unload_vision_model() en fin de batch pour rendre la VRAM à ComfyUI.
    """
    try:
        url = (ollama_url or _ollama_url()).strip().rstrip('/')
        # Normalize to a format Ollama's decoder can read (WebP -> JPEG); JPEG/PNG are
        # passed through untouched. Without this, WebP dataset bytes hit HTTP 400
        # "Failed to load image or audio file" on llama.cpp-backed runners (issue #6).
        b64 = base64.b64encode(_ensure_ollama_decodable(image_bytes)).decode('ascii')
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
        # If Ollama answered with a 4xx/5xx it told us WHY in the body — carry that
        # exact reason into both the log and the user-facing error. '' when the
        # failure had no HTTP response (connection/timeout), leaving the existing
        # unreachable/restart handling untouched.
        reject = _ollama_reject_message(e)
        if auto_start_local:
            # A rejection means the server DID answer, so starting a stopped server
            # can't fix it — surface Ollama's own reason now instead of retrying
            # into the same wall and reporting a generic "no caption after restart".
            if reject:
                logger.warning('vision_ollama: %s', reject)
                raise RuntimeError(reject) from e
            from . import ollama_control
            ready = ollama_control.ensure_captioning_ready()
            if not ready.get('ok'):
                raise RuntimeError(ready.get('error') or 'Ollama is unavailable') from e
            retried = describe_image_ollama(
                image_bytes, prompt, ollama_url=ollama_url, model=model,
                num_predict=num_predict, num_ctx=num_ctx,
                repeat_penalty=repeat_penalty, prefer_json=prefer_json, fmt=fmt,
                keep_alive=keep_alive, auto_start_local=False, timeout=timeout)
            if not retried:
                raise RuntimeError(
                    'Ollama did not return a caption after restart — check the configured '
                    'vision model and the application log.') from e
            return retried
        # Best-effort call: contract is to return "" — but still log the concrete
        # reason (previously only the opaque status code reached the log).
        logger.warning('vision_ollama: describe skipped: %s', reject or e)
        return ''


def generate_text_ollama(prompt: str, *,
                         ollama_url: str | None = None,
                         model: str | None = None,
                         num_predict: int = 400,
                         num_ctx: int = 4096,
                         repeat_penalty: float = 1.1,
                         keep_alive: str | int = 0,
                         timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Text-only generation via the SAME Ollama model as the vision seam (no image
    attached). Used to derive a SHORT caption from an already-stored long one — a pure
    text transform, so no GPU-heavy vision decode and no reason to pull in a second model.
    Reusing the abliterated Qwen3-VL matters: a vanilla text model would refuse to
    shorten the NSFW captions this app produces. Returns the text, or "" best-effort on
    any failure (the caller degrades to keeping the long caption). Same response/thinking
    extraction as describe_image_ollama so the -instruct answer is read correctly."""
    try:
        url = (ollama_url or _ollama_url()).strip().rstrip('/')
        payload = {
            'model': model or get_vision_model(),
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.3, 'num_ctx': int(num_ctx),
                        'num_predict': int(num_predict),
                        'repeat_penalty': float(repeat_penalty)},
            'keep_alive': keep_alive,
        }
        resp = requests.post(f'{url}/api/generate', json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        caption = (data.get('response') or '').strip()
        if caption:
            return caption
        thinking = (data.get('thinking') or '').strip()
        if thinking:
            parts = [p.strip() for p in thinking.split('\n\n') if p.strip()]
            return parts[-1] if parts else thinking
        return ''
    except Exception as e:
        logger.warning('vision_ollama: text generate skipped: %s', _ollama_reject_message(e) or e)
        return ''


def unload_vision_model(*, ollama_url: str | None = None, model: str | None = None) -> bool:
    """Décharge le modèle vision d'Ollama (libère la VRAM). À appeler à la FIN d'un
    batch caption/classify (où les appels ont gardé le modèle chaud via keep_alive)
    AVANT que ComfyUI reprenne le GPU, sinon le modèle resterait chargé et ComfyUI
    pourrait manquer de VRAM. Retente une fois car un unload raté = ~5 min résident
    (keep_alive). Retourne True si l'appel a réussi."""
    try:
        url = (ollama_url or _ollama_url()).strip().rstrip('/')
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
