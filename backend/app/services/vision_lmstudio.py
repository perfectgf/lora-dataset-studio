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
import threading
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

# What LM Studio says when the JSON-mode spelling is not the one it wants.
_JSON_FORMAT_MARKER = "'response_format.type'"
# Which spelling this server accepts, learned once per process. Preference only --
# the other is always still tried, so pointing the app at a different build mid-
# session cannot strand it on a memo.
_json_format: str | None = None


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
    # The same refusals capabilities._validated_setup_http_base applies to every
    # other user-typed endpoint. Without them a password typed into the URL was
    # echoed back by /api/capabilities, and a scheme-less string produced a
    # nonsense request instead of an honest "unreachable".
    try:
        if (parts.scheme not in ('http', 'https') or not parts.hostname
                or parts.username is not None or parts.password is not None):
            return ''
        _ = parts.port                     # a malformed port is not a URL
    except (TypeError, ValueError):
        return ''
    path = parts.path or ''
    for suffix in ('/api/v1', '/api/v0', '/v1', '/v0'):
        # Case-insensitive: LM Studio's own docs write /v1, people paste /V1.
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip('/'), '', ''))


def base_url() -> str:
    """The configured LM Studio origin, normalised, never None."""
    return _suffix_free(cfg.get('lmstudio.url') or DEFAULT_URL) or DEFAULT_URL


def _headers() -> dict:
    # Through the secret store, never config.json: `cfg.secret` keeps it out of the
    # settings payload, out of the pasted diagnostic and out of a config a user
    # might attach to a bug report.
    key = (cfg.secret('LMSTUDIO_API_KEY') or '').strip()
    return {'Authorization': f'Bearer {key}'} if key else {}


def get_vision_model() -> str:
    """The configured model id, or '' meaning "whatever this server has loaded".

    Unlike Ollama, LM Studio can only serve a model that is already loaded (JIT is
    off by default), so "the loaded one" is a genuinely useful default rather than
    a vague one — see :func:`resolve_model`.
    """
    return (cfg.get('lmstudio.vision_model') or '').strip()


# What the last batch of requests learned about the server, beyond its body.
# `probe_resident` needs it: "nothing is listening" and "answered but I could not
# read it" are the same empty result and OPPOSITE verdicts for a fail-closed fence.
class _Reach:
    __slots__ = ('answered', 'refused', 'status', 'body')

    def __init__(self):
        self.answered = False      # an HTTP response came back, whatever its status
        self.refused = False       # the connection was actively refused
        # The last refusal seen, so a 401 from a mistyped token can say so instead
        # of being reported as "not answering — press Start Server", which sends
        # the user to restart a server that is already running.
        self.status = None
        self.body = ''


def _get(path: str, *, url: str | None = None, reach: '_Reach | None' = None,
         timeout: tuple[float, float] | float = (5, 20)):
    """One GET, total. Returns the response, or None if the request itself failed.

    Total on purpose: `list_models` documents "never raises", four callers were
    written on that promise, and the request was the one part of it left
    unguarded — so the single most likely failure (LM Studio not started, which
    this app cannot start for the user) escaped as a raw ConnectionError, past
    every failure sentence, out to a 500.
    """
    try:
        resp = requests.get(f'{url or base_url()}{path}', headers=_headers(),
                            timeout=timeout)
        if reach is not None:
            reach.answered = True
            status = getattr(resp, 'status_code', None)
            if isinstance(status, int) and status >= 400:
                reach.status, reach.body = status, (getattr(resp, 'text', '') or '')[:300]
        return resp
    except Exception as exc:               # noqa: BLE001 - reported as a state
        if reach is not None:
            from .ollama_gpu_fence import _connection_refused
            reach.refused = reach.refused or _connection_refused(exc)
        logger.debug('vision_lmstudio: GET %s failed: %s', path, exc)
        return None


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
    out = {'ok': False, 'reachable': False, 'surface': None, 'models': [],
           'answered': False, 'refused': False, 'last_status': None, 'last_body': ''}
    reach = _Reach()

    # --- native v1 (LM Studio >= 0.4.0): richest, and the only one with configs
    data = _json_or_none(_get('/api/v1/models', url=url, reach=reach, timeout=timeout))
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
            }
            for m in data['models'] if isinstance(m, dict)
        ])
        out['answered'], out['refused'] = reach.answered, reach.refused
        out['last_status'], out['last_body'] = reach.status, reach.body
        return out

    # --- native v0: has `state` and `type`, no instance list
    data = _json_or_none(_get('/api/v0/models', url=url, reach=reach, timeout=timeout))
    if isinstance(data, dict) and isinstance(data.get('data'), list):
        out.update(ok=True, reachable=True, surface='v0', models=[
            {
                'id': m.get('id') or '',
                'type': _norm_type(m.get('type')),
                'loaded': str(m.get('state') or '') == 'loaded',
                'instances': [m.get('id')] if str(m.get('state') or '') == 'loaded' else [],
            }
            for m in data['data'] if isinstance(m, dict)
        ])
        out['answered'], out['refused'] = reach.answered, reach.refused
        out['last_status'], out['last_body'] = reach.status, reach.body
        return out

    # --- OpenAI-compatible: ids only. No type, no residency — say so.
    data = _json_or_none(_get('/v1/models', url=url, reach=reach, timeout=timeout))
    if isinstance(data, dict) and isinstance(data.get('data'), list):
        out.update(ok=True, reachable=True, surface='openai', models=[
            {'id': m.get('id') or '', 'type': '', 'loaded': None,
             'instances': []}
            for m in data['data'] if isinstance(m, dict)
        ])
    out['answered'], out['refused'] = reach.answered, reach.refused
    out['last_status'], out['last_body'] = reach.status, reach.body
    return out


def _norm_type(raw) -> str:
    """`embedding` (v1) and `embeddings` (v0) are the same thing. Say it once."""
    t = str(raw or '').strip().lower()
    return 'embeddings' if t.startswith('embedding') else t


def _no_model_sentence(url: str | None = None) -> str:
    """Why there is no model to send to — reachability FIRST.

    Blaming the model on a server that never answered is the wrong sentence at the
    worst moment: it sends someone to load a model into an app they have not
    started. Both states end with "no model resolved", so the difference has to be
    asked for explicitly.
    """
    listed = list_models(url=url)
    if not listed['reachable']:
        return failure_sentence(None, 'unreachable')
    return failure_sentence(400, 'no models loaded')


def resolve_model(preferred: str | None = None, *, url: str | None = None) -> str:
    """The model id to send. Configured value wins; else the loaded one; else ''.

    Preferring what is LOADED is not a shortcut — with JIT off, a request naming
    an unloaded model fails no matter how correct the name is.
    """
    explicit = (preferred or get_vision_model() or '').strip()
    if explicit:
        return explicit
    listed = list_models(url=url)
    # An embedding model cannot chat, and keeping one resident is routine in LM
    # Studio (its own document chat loads one). Picked as "the loaded model" it
    # turns every caption into an opaque server error with nothing connecting it
    # to the cause. The pickers already filter this out — resolve_model was the
    # one place that forgot, on the DEFAULT path (vision_model is empty by design).
    usable = [m for m in listed['models'] if m.get('id')
              and m.get('type') != 'embeddings']
    for wanted in (('vlm',), ('llm', '')):
        loaded = [m['id'] for m in usable if m.get('loaded') and m.get('type') in wanted]
        if loaded:
            return loaded[0]
    loaded_any = [m['id'] for m in usable if m.get('loaded')]
    if loaded_any:
        return loaded_any[0]
    vlms = [m['id'] for m in usable if m.get('type') in ('vlm', 'llm')]
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
        return 'unknown', [], {'reason': str(exc)}
    if not listed['reachable']:
        # `down` is NOT a neutral state here: the fence reads it as proof the card
        # is free and admits ComfyUI. Only an actively refused connection proves
        # that. A timeout, a 401 from a token the user got wrong, a 500, a proxy —
        # all of those are a server that may well be holding 3 GB of VRAM, and
        # calling them `down` is how a fail-closed guard opens. The Ollama probe
        # already draws the line here; this now agrees with it.
        if listed.get('answered') or not listed.get('refused'):
            return 'unknown', [], {
                'reason': 'the server answered in a shape this app does not recognise'
                          if listed.get('answered') else 'no usable answer'}
        return 'down', [], {'reason': 'connection refused'}
    if listed['surface'] == 'openai':
        return 'unknown', [], {
            'reason': 'this server only answers the OpenAI-compatible API, '
                      'which reports neither model type nor residency'}
    resident = [i for m in listed['models'] for i in (m.get('instances') or [])]
    if resident:
        return 'models', resident, {'surface': listed['surface']}
    return 'empty', [], {'surface': listed['surface']}


_load_lock = threading.Lock()
# (endpoint, model) pairs this process has SEEN resident -- spares one listing per
# call on the hot path of a caption batch. Cleared when LDS itself unloads (the
# keep-warm lease, a consented eviction); an unload done inside LM Studio's own
# window simply makes the next call fail with the server's "no models loaded",
# which the next batch heals by loading again.
_seen_loaded: set = set()


def ensure_model_loaded(model: str, *, url: str | None = None,
                        timeout: tuple[float, float] | float = (10, 600)) -> tuple[bool, str]:
    """Make `model` resident, loading it OURSELVES when LM Studio has it on disk.

    This is the answer to a fair complaint from the first real install: "why do I
    have to keep loading a model? LDS is supposed to handle everything." It was
    right on principle AND on consistency -- this driver already starts the server
    and unloads models, so refusing to load one was not caution, just a gap.

    Measured on 0.4.23 before writing this:
      · POST /api/v1/models/load {"model": <key>} → 200 {"status": "loaded",
        "instance_id": ..., "load_time_seconds": ...}
      · loading an ALREADY-loaded model does not no-op -- it stacks a second
        instance (":2") and doubles the VRAM. Hence the residency check first,
        and the lock, so a concurrent batch cannot double-load either.
      · an unknown model → {"error": {"type": "model_not_found", ...}} -- turned
        into a sentence pointing at the download field (Settings ▸ Local tools),
        which lmstudio_download drives through the server's own job API.

    A load LDS performs is a residency LDS OWNS: it is registered with the GPU
    fence, so the keep-warm lease may legitimately unload it later and a ComfyUI
    hand-off can actually free the card. A model found already loaded stays
    BORROWED -- the user put it there, and this function never touches it.
    Returns (ok, detail); never raises.
    """
    endpoint = _suffix_free(url) if url else base_url()
    if (endpoint, model) in _seen_loaded:
        return True, 'already loaded'
    with _load_lock:
        if (endpoint, model) in _seen_loaded:      # loaded by the call we waited on
            return True, 'already loaded'
        listed = list_models(url=endpoint)
        if not listed['reachable']:
            return False, failure_sentence(listed.get('last_status'),
                                           listed.get('last_body') or '')
        if listed['surface'] == 'openai':
            # This surface reports no residency, so a blind load risks the double-
            # instance above -- and a JIT-enabled server behind it loads on its
            # own. Leave the request to the server, exactly as before.
            return True, 'residency unknown (OpenAI-compatible surface only)'
        row = next((m for m in listed['models'] if m.get('id') == model), None)
        if row is not None and row.get('loaded'):
            _seen_loaded.add((endpoint, model))
            return True, 'already loaded'
        try:
            resp = requests.post(f'{endpoint}/api/v1/models/load',
                                 json={'model': model},
                                 headers={'Content-Type': 'application/json', **_headers()},
                                 timeout=timeout)
        except requests.RequestException as exc:
            return False, failure_sentence(None, str(exc))
        if resp.status_code >= 400:
            body = resp.text or ''
            if 'model_not_found' in body:
                return False, (f'"{model}" is not downloaded yet — download it from '
                               'Settings ▸ Local tools (or inside LM Studio), or name '
                               'a downloaded one there.')
            return False, failure_sentence(resp.status_code, body)
        from . import ollama_gpu_fence
        ollama_gpu_fence.register_lds_load(endpoint, model)
        _seen_loaded.add((endpoint, model))
        try:
            seconds = float(resp.json().get('load_time_seconds') or 0)
            return True, f'loaded in {seconds:.1f}s'
        except (ValueError, AttributeError):
            return True, 'loaded'


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
        # `down` counts as released: a server that is not there holds no VRAM, and
        # calling that a FAILED release left the keep-warm lease outstanding and
        # ComfyUI blocked on a card nothing was using. `unknown` must stay False —
        # something answered and we could not read it, so we have proved nothing.
        return state in ('empty', 'down')
    ok = True
    _seen_loaded.difference_update({(url, t) for t in targets})
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
    """Release what LDS is entitled to release — through the fence, like Ollama's.

    NOT `release(url, model)`. With `model=None` that unloads every instance the
    server reports, and the bare form is the common one: seven caption batches end
    with `unload_vision_model()` and the keep-warm lease revokes with it. Under LM
    Studio each of those would have wiped the server clean — including an embedding
    model, or a chat model another application was serving from, that LDS never
    loaded. The fence is what knows the difference between what LDS put there and
    what it merely borrowed; `release()` stays the low-level primitive it reaches
    for once it has decided.
    """
    from . import ollama_gpu_fence
    released = ollama_gpu_fence.release_owned_models(ollama_url=url, model=model)
    # None = the fence considers this endpoint out of its scope (remote). LM Studio
    # on another machine holds no GPU here, so there is nothing to free.
    return True if released is None else released


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


def _json_response_format(kind: str) -> dict:
    if kind == 'json_object':
        return {'type': 'json_object'}
    # Permissive on purpose. The callers ask for "an object", never a named schema,
    # and LM Studio ENFORCES the grammar from this one -- measured stronger than
    # {'type': 'text'}, which only asks the prompt nicely and is free to drift.
    return {'type': 'json_schema',
            'json_schema': {'name': 'lds_object', 'schema': {'type': 'object'}}}


def _chat(messages, *, model, max_tokens, temperature, timeout, url=None, as_json=False,
          top_p=None, stop=None):
    """POST one chat completion. Retries ONCE on a rejected JSON-mode spelling.

    Ollama's `format='json'` has an OpenAI-compatible equivalent, and it is not
    optional: framing classify, head-crop and the watermark-bbox caller all PARSE
    the answer. Dropped silently, those passes get prose where they expect an
    object and fail as if the model were bad.

    Which spelling works is a per-BUILD fact, not a per-API one. Measured on
    0.4.23: `{'type': 'json_object'}` -- the historical OpenAI spelling, and what
    this driver shipped -- is refused with HTTP 400 "'response_format.type' must
    be 'json_schema' or 'text'", which took framing classify down on every call
    while captioning (no JSON mode) worked perfectly. Other builds accept the
    older spelling, so neither is hard-coded: the working one is preferred and
    the other stays one retry away.
    """
    global _json_format
    payload = {'model': model, 'messages': messages,
               'max_tokens': max_tokens, 'temperature': temperature, 'stream': False}
    # Both are OpenAI-standard, so they travel to whatever LM Studio is
    # fronting; sent only when a caller asked, so every existing pass keeps the
    # exact payload it was measured on.
    if top_p is not None:
        payload['top_p'] = float(top_p)
    if stop:
        payload['stop'] = list(stop)
    headers = {'Content-Type': 'application/json', **_headers()}
    target = f'{url or base_url()}/v1/chat/completions'
    if not as_json:
        return requests.post(target, json=payload, headers=headers, timeout=timeout)

    order = (['json_object', 'json_schema'] if _json_format == 'json_object'
             else ['json_schema', 'json_object'])
    resp = None
    for kind in order:
        resp = requests.post(target, json=dict(payload, response_format=_json_response_format(kind)),
                             headers=headers, timeout=timeout)
        if resp.status_code < 400:
            _json_format = kind
            return resp
        if _JSON_FORMAT_MARKER not in (resp.text or '').lower():
            return resp                    # a different problem: do not burn a retry
    return resp


def _answer(resp) -> str:
    data = resp.json()
    return (((data.get('choices') or [{}])[0].get('message') or {}).get('content') or '').strip()


def _image_field(b64: str, data_uri: bool) -> str:
    return f'data:image/jpeg;base64,{b64}' if data_uri else b64


def describe_frames(frames, prompt, *,
                    url: str | None = None,
                    model: str | None = None,
                    num_predict: int = 600,
                    timeout: tuple[float, float] | float = (10, 300)) -> str:
    """N frames of one shot -> a caption, or '' best-effort — the video door of
    :func:`describe_image`: one chat call, one image part per frame.

    Each frame passes the same safety gate; an unreadable one is dropped with a
    log rather than sinking the shot. temperature 0 on purpose — training
    captions, not conversation — and the same data-URI/bare-b64 memory the
    image door keeps."""
    global _data_uri_ok
    b64s = []
    for fb in frames:
        safe = vision_image.ensure_vision_safe_jpeg(fb, provider='vision_lmstudio')
        if safe is None:
            logger.warning('vision_lmstudio: a frame was unsafe or unreadable — dropped')
            continue
        b64s.append(base64.b64encode(safe).decode())
    if not b64s:
        return ''
    endpoint = _suffix_free(url) if url else base_url()
    target = resolve_model(model, url=endpoint)
    if not target:
        logger.warning('vision_lmstudio: describe_frames skipped: %s',
                       _no_model_sentence(endpoint))
        return ''
    loaded_ok, load_detail = ensure_model_loaded(target, url=endpoint)
    if not loaded_ok:
        logger.warning('vision_lmstudio: describe_frames skipped: %s', load_detail)
        return ''
    order = [True, False] if _data_uri_ok is not False else [False, True]
    if _data_uri_ok is True:
        order = [True]
    last_status, last_body = None, ''
    for use_data_uri in order:
        content = [{'type': 'text', 'text': prompt}] + [
            {'type': 'image_url', 'image_url': {'url': _image_field(b, use_data_uri)}}
            for b in b64s]
        messages = [{'role': 'user', 'content': content}]
        try:
            _admit(endpoint, target)
            resp = _chat(messages, model=target, max_tokens=num_predict,
                         temperature=0, timeout=timeout, url=endpoint,
                         as_json=False)
        except LocalLmStudioFenceError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported below
            last_status, last_body = None, str(exc)
            break
        if getattr(resp, 'status_code', 0) < 400:
            try:
                answer = _answer(resp)
            except Exception as exc:  # noqa: BLE001 - a proxy's HTML error page
                last_status, last_body = resp.status_code, str(exc)
                break
            _data_uri_ok = use_data_uri
            return answer
        last_status, last_body = resp.status_code, resp.text or ''
        if _INVALID_URL_MARKER not in last_body.lower():
            break
    logger.warning('vision_lmstudio: describe_frames failed: %s',
                   failure_sentence(last_status, last_body))
    return ''


def describe_image(image_bytes: bytes, prompt: str, *,
                   url: str | None = None,
                   model: str | None = None,
                   num_predict: int = 800,
                   as_json: bool = False,
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
        msg = _no_model_sentence(endpoint)
        if strict:
            raise RuntimeError(msg)
        logger.warning('vision_lmstudio: describe skipped: %s', msg)
        return ''
    loaded_ok, load_detail = ensure_model_loaded(target, url=endpoint)
    if not loaded_ok:
        if strict:
            raise RuntimeError(load_detail)
        logger.warning('vision_lmstudio: describe skipped: %s', load_detail)
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
                         temperature=0.2, timeout=timeout, url=endpoint,
                         as_json=as_json)
        except LocalLmStudioFenceError:
            raise                          # the fence speaks for itself, 409 upstream
        except Exception as exc:           # noqa: BLE001 - reported below
            last_status, last_body = None, str(exc)
            break
        if getattr(resp, 'status_code', 0) < 400:
            try:
                answer = _answer(resp)
            except Exception as exc:       # noqa: BLE001 - a proxy's HTML error page
                last_status, last_body = resp.status_code, str(exc)
                break
            _data_uri_ok = use_data_uri
            return answer
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
                  temperature: float = 0.2,
                  top_p: float | None = None,
                  stop: list[str] | None = None,
                  timeout: tuple[float, float] | float = (10, 120)) -> str:
    """Text-only generation through the same loaded model. Mirrors the Ollama seam."""
    endpoint = _suffix_free(url) if url else base_url()
    target = resolve_model(model, url=endpoint)
    if not target:
        msg = _no_model_sentence(endpoint)
        if strict:
            raise RuntimeError(msg)
        logger.warning('vision_lmstudio: text generate skipped: %s', msg)
        return ''
    loaded_ok, load_detail = ensure_model_loaded(target, url=endpoint)
    if not loaded_ok:
        if strict:
            raise RuntimeError(load_detail)
        logger.warning('vision_lmstudio: text generate skipped: %s', load_detail)
        return ''
    try:
        _admit(endpoint, target)
        resp = _chat([{'role': 'user', 'content': prompt}], model=target,
                     max_tokens=num_predict, temperature=float(temperature),
                     top_p=top_p, stop=stop, timeout=timeout, url=endpoint)
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
    try:
        return _answer(resp)
    except Exception as exc:               # noqa: BLE001 - same contract as above
        msg = failure_sentence(resp.status_code, str(exc))
        if strict:
            raise RuntimeError(msg) from exc
        logger.warning('vision_lmstudio: text generate skipped: %s', msg)
        return ''
