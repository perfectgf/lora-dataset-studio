"""LM Studio driver — the second local LLM provider, beside Ollama.

Everything here was measured against a live LM Studio 0.4.23 rather than read off
the docs, because three of the things the docs and the bug tracker implied turned
out to be wrong for that build:

1. **Images go in as the STANDARD data: URI.** The open bug report #1752 says the
   OpenAI-compatible endpoint rejects `data:image/jpeg;base64,…` and implies bare
   base64 is the working form. On 0.4.23 it is the exact opposite: the data URI
   answers 200 and reads the image correctly, bare base64 returns
   ``400 {"error": "Invalid url."}``. A driver written from the bug report would
   fail every first call. So the data URI is what we send, and the bare form
   survives only as a one-shot fallback for the builds where the bug was live —
   triggered on that error, and remembered per process so the second image does
   not pay for the probe again.
2. **Residency has two different shapes.** ``/api/v1/models`` reports it as
   ``loaded_instances: [{id, config}]`` (no ``state`` field at all); ``/api/v0``
   reports ``state: "loaded" | "not-loaded"``. The OpenAI ``/v1/models`` carries
   NEITHER residency nor type — which is why :func:`probe_resident` answers
   ``unknown`` there instead of "nothing is loaded". That distinction is load
   bearing: the GPU fence is fail-closed, and reading "cannot tell" as "free"
   would hand ComfyUI a card another process is holding.
3. **The type spelling differs between surfaces** — ``embedding`` in v1,
   ``embeddings`` in v0 — so anything matching on it normalises first.

Two more measured behaviours shape the error messages: JIT loading is OFF by
default (a reachable server with nothing loaded answers 400 "No models loaded…",
before it even looks at the request), and there is no TTL by default (a loaded
model stays resident until something unloads it). Both are the NORMAL state right
after an install, so each gets a sentence that names the actual next action.
"""
from __future__ import annotations

import base64
import logging
from urllib.parse import urlsplit, urlunsplit

import requests

from .. import config as cfg
from . import vision_image

logger = logging.getLogger(__name__)

DEFAULT_URL = 'http://127.0.0.1:1234'

# The message LM Studio returns when the request shape it wants is not the one it
# got. Matched to decide whether the image-field fallback is worth one retry.
_INVALID_URL_MARKER = 'invalid url'
# What a reachable-but-empty server says. Recognised so the caller can be told to
# load a model rather than being handed a bare 400.
_NO_MODELS_MARKER = 'no models loaded'

# Which image field shape this server accepts, learned once per process.
# None = not yet known, True = data URI (every build measured so far), False = bare.
_data_uri_ok: bool | None = None


def _suffix_free(url: str) -> str:
    """Strip an API suffix a user may have pasted from LM Studio's own UI.

    The Developer tab advertises the server as ``http://localhost:1234/v1``, so
    that is the string people copy. Left alone it breaks twice over: the driver
    would compose ``…/v1/v1/chat/completions``, and the GPU fence classifies any
    URL carrying a path as ``unknown`` and refuses every local call with a
    message about Ollama. Both symptoms, one cause — so normalise at the door.
    """
    if not isinstance(url, str) or not url.strip():
        return ''
    parts = urlsplit(url.strip().rstrip('/'))
    path = parts.path or ''
    for suffix in ('/api/v1', '/api/v0', '/v1', '/v0'):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip('/'), '', ''))


def base_url() -> str:
    """The configured LM Studio origin, normalised, never None."""
    return _suffix_free(cfg.get('lmstudio.url') or DEFAULT_URL) or DEFAULT_URL


def _headers() -> dict:
    key = (cfg.get('lmstudio.api_key') or '').strip()
    return {'Authorization': f'Bearer {key}'} if key else {}


def get_vision_model() -> str:
    """The configured model id, or '' meaning "whatever this server has loaded".

    Unlike Ollama, LM Studio can only serve a model that is already loaded (JIT is
    off by default), so "the loaded one" is a genuinely useful default rather than
    a vague one — see :func:`resolve_model`.
    """
    return (cfg.get('lmstudio.vision_model') or '').strip()


def _get(path: str, *, url: str | None = None,
         timeout: tuple[float, float] | float = (5, 20)):
    return requests.get(f'{url or base_url()}{path}', headers=_headers(), timeout=timeout)


def _json_or_none(resp):
    try:
        if getattr(resp, 'status_code', 0) >= 400:
            return None
        return resp.json()
    except Exception:                      # noqa: BLE001 - a malformed body is "not this API"
        return None


def list_models(*, url: str | None = None,
                timeout: tuple[float, float] | float = (5, 20)) -> dict:
    """Everything the server knows about, normalised across its three surfaces.

    Returns ``{'ok', 'reachable', 'surface', 'models': [{id, type, loaded, ...}]}``.
    Never raises — an unreachable server is a state, not an error. ``surface``
    names which API answered, because what the caller can TRUST depends on it:
    only v1 and v0 carry type and residency.
    """
    out = {'ok': False, 'reachable': False, 'surface': None, 'models': []}

    # --- native v1 (LM Studio >= 0.4.0): richest, and the only one with configs
    data = _json_or_none(_get('/api/v1/models', url=url, timeout=timeout))
    if isinstance(data, dict) and isinstance(data.get('models'), list):
        out.update(ok=True, reachable=True, surface='v1', models=[
            {
                'id': m.get('key') or m.get('id') or '',
                'type': _norm_type(m.get('type')),
                'loaded': bool(m.get('loaded_instances')),
                # The unload endpoint's `instance_id` PARAMETER is fed from the
                # instance's `id` FIELD. They are not the same name.
                'instances': [i.get('id') for i in (m.get('loaded_instances') or [])
                              if isinstance(i, dict) and i.get('id')],
                'display_name': m.get('display_name') or '',
            }
            for m in data['models'] if isinstance(m, dict)
        ])
        return out

    # --- native v0: has `state` and `type`, no instance list
    data = _json_or_none(_get('/api/v0/models', url=url, timeout=timeout))
    if isinstance(data, dict) and isinstance(data.get('data'), list):
        out.update(ok=True, reachable=True, surface='v0', models=[
            {
                'id': m.get('id') or '',
                'type': _norm_type(m.get('type')),
                'loaded': str(m.get('state') or '') == 'loaded',
                'instances': [m.get('id')] if str(m.get('state') or '') == 'loaded' else [],
                'display_name': '',
            }
            for m in data['data'] if isinstance(m, dict)
        ])
        return out

    # --- OpenAI-compatible: ids only. No type, no residency — say so.
    data = _json_or_none(_get('/v1/models', url=url, timeout=timeout))
    if isinstance(data, dict) and isinstance(data.get('data'), list):
        out.update(ok=True, reachable=True, surface='openai', models=[
            {'id': m.get('id') or '', 'type': '', 'loaded': None,
             'instances': [], 'display_name': ''}
            for m in data['data'] if isinstance(m, dict)
        ])
    return out


def _norm_type(raw) -> str:
    """`embedding` (v1) and `embeddings` (v0) are the same thing. Say it once."""
    t = str(raw or '').strip().lower()
    return 'embeddings' if t.startswith('embedding') else t


def resolve_model(preferred: str | None = None, *, url: str | None = None) -> str:
    """The model id to send. Configured value wins; else the loaded one; else ''.

    Preferring what is LOADED is not a shortcut — with JIT off, a request naming
    an unloaded model fails no matter how correct the name is.
    """
    explicit = (preferred or get_vision_model() or '').strip()
    if explicit:
        return explicit
    listed = list_models(url=url)
    loaded = [m['id'] for m in listed['models'] if m.get('loaded')]
    if loaded:
        return loaded[0]
    vlms = [m['id'] for m in listed['models'] if m.get('type') in ('vlm', 'llm')]
    return vlms[0] if vlms else ''


def probe_resident(endpoint: str | None = None) -> tuple[str, list, dict]:
    """What is holding this server's GPU right now, in the fence's own vocabulary.

    Returns ``(state, names, meta)`` with ``state`` one of the four the fence
    branches on:

    ``models``  — these ids are resident.
    ``empty``   — the server answered on a surface that CAN report residency, and
                  nothing is loaded.
    ``down``    — nothing answered.
    ``unknown`` — something answered but not in a shape that reports residency
                  (the OpenAI surface, a proxy, a different product on the port).

    ``unknown`` must never collapse into ``empty``: the fence is fail-closed, and
    the whole point of it is to not hand the card to ComfyUI on a guess.
    """
    url = _suffix_free(endpoint) if endpoint else base_url()
    if not url:
        return 'unknown', [], {'reason': 'no endpoint configured'}
    try:
        listed = list_models(url=url)
    except Exception as exc:               # noqa: BLE001 - reported as a state
        return 'down', [], {'reason': str(exc)}
    if not listed['reachable']:
        return 'down', [], {'reason': 'no answer'}
    if listed['surface'] == 'openai':
        return 'unknown', [], {
            'reason': 'this server only answers the OpenAI-compatible API, '
                      'which reports neither model type nor residency'}
    resident = [i for m in listed['models'] for i in (m.get('instances') or [])]
    if resident:
        return 'models', resident, {'surface': listed['surface']}
    return 'empty', [], {'surface': listed['surface']}


def release(endpoint: str | None = None, model: str | None = None) -> bool:
    """Unload a resident model. Measured: this genuinely frees the VRAM.

    LM Studio has no TTL by default, so nothing expires on its own — which makes
    this the difference between a fence that can offer to free the card and one
    that can only ask the user to wait.
    """
    url = _suffix_free(endpoint) if endpoint else base_url()
    state, names, _ = probe_resident(url)
    targets = [model] if model else names
    if state != 'models' or not targets:
        return state == 'empty'
    ok = True
    for inst in targets:
        try:
            resp = requests.post(f'{url}/api/v1/models/unload',
                                 json={'instance_id': inst},
                                 headers=_headers(), timeout=(5, 30))
            if getattr(resp, 'status_code', 0) >= 400:
                logger.warning('vision_lmstudio: unload %s -> HTTP %s', inst,
                               resp.status_code)
                ok = False
        except Exception as exc:           # noqa: BLE001 - best effort, reported
            logger.warning('vision_lmstudio: unload %s failed: %s', inst, exc)
            ok = False
    return ok


def unload_vision_model(*, url: str | None = None, model: str | None = None) -> bool:
    """Name-compatible with the Ollama driver's entry point."""
    return release(url, model)


def failure_sentence(status: int | None, body: str) -> str:
    """One true sentence per situation, naming the gesture that fixes it.

    A bare 400 from a local server tells the user nothing; these four cases cover
    everything measured on a real install, and each names what to do next.
    """
    text = (body or '').lower()
    if status is None:
        return (f'LM Studio is not answering at {base_url()}. Open LM Studio, go to '
                'Developer and press Start Server (it listens on port 1234).')
    if _NO_MODELS_MARKER in text:
        return ('LM Studio is running but has no model loaded. Load a vision model '
                'in its Developer tab (or enable JIT loading) and try again.')
    if _INVALID_URL_MARKER in text:
        return ('LM Studio refused the image payload. This build wants a different '
                'encoding than the one tried first; retrying with the other form.')
    if status == 404:
        return ('LM Studio answered, but not on the API this needs. Check the URL in '
                'Settings ▸ Local tools points at the server root (no /v1 suffix).')
    return f'LM Studio returned HTTP {status}: {(body or "").strip()[:200]}'


def _fence_error_base():
    from .vision_ollama import LocalOllamaFenceError
    return LocalOllamaFenceError


class LocalLmStudioFenceError(_fence_error_base()):
    """A local inference lost its verified GPU ownership.

    Subclasses the Ollama one on purpose. That name is legacy — it means "the
    local-LLM fence refused" — and every handler in the app keys on it: the 409
    with `code: ollama_fence_blocked` in routes/_common.py is what makes the UI
    show its banner, offer "Run anyway", and replay the action once the card is
    free. A sibling class would have been caught by none of them, so an LM Studio
    user would have got a bare 500 exactly where the app has the best answer.
    """


def _admit(url: str, model: str) -> None:
    """Ask the GPU fence before loading anything onto a LOCAL card.

    Same gate the Ollama driver goes through, for the same reason: on one GPU a
    resident vision model and ComfyUI do not both fit, and an unfenced provider
    would win that race silently by loading first. Passing `provider` pins the
    wire format to this endpoint, so a later release speaks LM Studio's API even
    if the global setting has moved on since.
    """
    from . import ollama_gpu_fence
    scope = ollama_gpu_fence.mark_before_generate(url, model, provider='lmstudio')
    if scope == 'blocked':
        raise LocalLmStudioFenceError(ollama_gpu_fence.blocked_message())


def _chat(messages, *, model, max_tokens, temperature, timeout, url=None):
    payload = {'model': model, 'messages': messages,
               'max_tokens': max_tokens, 'temperature': temperature, 'stream': False}
    headers = {'Content-Type': 'application/json', **_headers()}
    return requests.post(f'{url or base_url()}/v1/chat/completions', json=payload,
                         headers=headers, timeout=timeout)


def _answer(resp) -> str:
    data = resp.json()
    return (((data.get('choices') or [{}])[0].get('message') or {}).get('content') or '').strip()


def _image_field(b64: str, data_uri: bool) -> str:
    return f'data:image/jpeg;base64,{b64}' if data_uri else b64


def describe_image(image_bytes: bytes, prompt: str, *,
                   url: str | None = None,
                   model: str | None = None,
                   num_predict: int = 800,
                   strict: bool = False,
                   timeout: tuple[float, float] | float = (10, 180)) -> str:
    """Describe an image through LM Studio. "" best-effort, or raises if strict.

    The image goes through the same gate every provider uses (fresh JPEG, no
    EXIF/GPS, bounded side) — a second provider that skipped it would quietly
    turn captioning back into a metadata disclosure.
    """
    global _data_uri_ok
    safe = vision_image.ensure_vision_safe_jpeg(image_bytes, provider='vision_lmstudio')
    if safe is None:
        if strict:
            raise RuntimeError('The image could not be read safely, so it was not sent.')
        return ''
    b64 = base64.b64encode(safe).decode()
    endpoint = _suffix_free(url) if url else base_url()
    target = resolve_model(model, url=endpoint)
    if not target:
        msg = failure_sentence(400, 'no models loaded')
        if strict:
            raise RuntimeError(msg)
        logger.warning('vision_lmstudio: describe skipped: %s', msg)
        return ''

    # Measured order: data URI first. The other form is tried once, only when the
    # server says the url field was invalid, and the answer is remembered.
    order = [True, False] if _data_uri_ok is not False else [False, True]
    if _data_uri_ok is True:
        order = [True]
    last_status, last_body = None, ''
    for use_data_uri in order:
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url',
             'image_url': {'url': _image_field(b64, use_data_uri)}},
        ]}]
        try:
            _admit(endpoint, target)
            resp = _chat(messages, model=target, max_tokens=num_predict,
                         temperature=0.2, timeout=timeout, url=endpoint)
        except LocalLmStudioFenceError:
            raise                          # the fence speaks for itself, 409 upstream
        except Exception as exc:           # noqa: BLE001 - reported below
            last_status, last_body = None, str(exc)
            break
        if getattr(resp, 'status_code', 0) < 400:
            _data_uri_ok = use_data_uri
            return _answer(resp)
        last_status, last_body = resp.status_code, resp.text or ''
        if _INVALID_URL_MARKER not in last_body.lower():
            break                          # a different problem: do not burn a retry

    msg = failure_sentence(last_status, last_body)
    if strict:
        raise RuntimeError(msg)
    logger.warning('vision_lmstudio: describe skipped: %s', msg)
    return ''


def generate_text(prompt: str, *,
                  url: str | None = None,
                  model: str | None = None,
                  num_predict: int = 400,
                  strict: bool = False,
                  timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Text-only generation through the same loaded model. Mirrors the Ollama seam."""
    endpoint = _suffix_free(url) if url else base_url()
    target = resolve_model(model, url=endpoint)
    if not target:
        msg = failure_sentence(400, 'no models loaded')
        if strict:
            raise RuntimeError(msg)
        logger.warning('vision_lmstudio: text generate skipped: %s', msg)
        return ''
    try:
        _admit(endpoint, target)
        resp = _chat([{'role': 'user', 'content': prompt}], model=target,
                     max_tokens=num_predict, temperature=0.2, timeout=timeout, url=endpoint)
    except LocalLmStudioFenceError:
        raise                              # the fence speaks for itself, 409 upstream
    except Exception as exc:               # noqa: BLE001 - reported below
        msg = failure_sentence(None, str(exc))
        if strict:
            raise RuntimeError(msg) from exc
        logger.warning('vision_lmstudio: text generate skipped: %s', msg)
        return ''
    if getattr(resp, 'status_code', 0) >= 400:
        msg = failure_sentence(resp.status_code, resp.text or '')
        if strict:
            raise RuntimeError(msg)
        logger.warning('vision_lmstudio: text generate skipped: %s', msg)
        return ''
    return _answer(resp)
