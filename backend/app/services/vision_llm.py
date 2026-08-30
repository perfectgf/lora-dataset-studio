"""One door in front of the local LLM, whichever one the user runs.

The whole app already funnels its vision and text calls through two functions in
``vision_ollama``. That narrow waist is why a second provider does not need a
refactor: this module sits in front of it and dispatches on ``local_llm.provider``,
and ``vision_ollama`` keeps its name and its behaviour as the Ollama *driver*.
Nothing that imports it has to change for Ollama users, and the default provider
is ``ollama`` — an existing install sees no difference at all.

Two accessors here that look incidental and are not: :func:`vision_concurrency`
and :func:`keep_warm_seconds`. Both settings exist per provider, and both are read
in modules (``vision_pool``, ``vision_keepalive``) that hard-coded the ``ollama.*``
keys. Without routing them, an LM Studio user would see two dials in Settings that
change nothing — the exact "lying control" the repo's rules forbid.

What does NOT map, stated rather than faked:

- Ollama takes a per-request ``keep_alive``; LM Studio has no per-request
  equivalent (and no TTL by default). Under LM Studio the keep-warm setting is
  honoured by NOT unloading between calls and unloading when the lease ends —
  :func:`unload_vision_model` is the lever, and it genuinely frees the card.
- ``num_ctx``, ``fmt``/``prefer_json`` and ``auto_start_local`` are Ollama-only.
  They are accepted and dropped for LM Studio rather than silently reinterpreted,
  because a JSON-format request that is quietly downgraded to free text produces a
  caption that parses as garbage instead of failing.
"""
from __future__ import annotations

import logging

from .. import config as cfg

logger = logging.getLogger(__name__)

OLLAMA = 'ollama'
LMSTUDIO = 'lmstudio'
PROVIDERS = (OLLAMA, LMSTUDIO)

# What each provider is called in a sentence shown to a user.
LABELS = {OLLAMA: 'Ollama', LMSTUDIO: 'LM Studio'}


def provider() -> str:
    """The configured local LLM provider, defaulting to Ollama.

    An unknown value falls back to Ollama rather than failing: a config written by
    a NEWER version of the app must never brick captioning on an older one.
    """
    raw = cfg.get('local_llm.provider')
    if not isinstance(raw, str):
        # Total by construction, like `vision_ollama._ollama_url`. cfg.get can hand
        # back anything a hand-edited config.json contains, and a provider lookup
        # that raises would take every caption, every head-crop and every prompt
        # helper down with it — for a setting whose safe answer is obvious.
        return OLLAMA
    raw = raw.strip().lower()
    return raw if raw in PROVIDERS else OLLAMA


def label(name: str | None = None) -> str:
    """'Ollama' / 'LM Studio' — for the sentences the user reads."""
    return LABELS.get(name or provider(), LABELS[OLLAMA])


def _driver(name: str | None = None):
    if (name or provider()) == LMSTUDIO:
        from . import vision_lmstudio
        return vision_lmstudio
    from . import vision_ollama
    return vision_ollama


def base_url(name: str | None = None) -> str:
    p = name or provider()
    if p == LMSTUDIO:
        from . import vision_lmstudio
        return vision_lmstudio.base_url()
    from . import vision_ollama
    return vision_ollama._ollama_url()


def vision_model(name: str | None = None) -> str:
    return _driver(name).get_vision_model()


def vision_concurrency(name: str | None = None) -> int:
    """How many vision calls a batch keeps in flight, for the ACTIVE provider."""
    key = f'{name or provider()}.vision_concurrency'
    try:
        return int(str(cfg.get(key)).strip())
    except (TypeError, ValueError):
        return 4


def keep_warm_seconds(name: str | None = None) -> int:
    """Seconds an isolated call may keep the model resident, for the ACTIVE provider."""
    key = f'{name or provider()}.vision_keep_warm_seconds'
    try:
        return int(str(cfg.get(key)).strip())
    except (TypeError, ValueError):
        return 120


def describe_image(image_bytes: bytes, prompt: str, **kw) -> str:
    """Image + prompt -> text, through whichever provider is configured."""
    if provider() == LMSTUDIO:
        from . import vision_lmstudio
        return vision_lmstudio.describe_image(
            image_bytes, prompt,
            url=kw.get('url') or kw.get('ollama_url'),
            model=kw.get('model'),
            num_predict=kw.get('num_predict', 800),
            # Ollama's fmt/prefer_json has an OpenAI equivalent, so it travels
            # rather than being dropped: eight call sites parse the answer.
            as_json=(kw.get('fmt') == 'json') or bool(kw.get('prefer_json')),
            strict=bool(kw.get('strict') or kw.get('auto_start_local')),
            timeout=kw.get('timeout', (10, 180)))
    from . import vision_ollama
    return vision_ollama.describe_image_ollama(image_bytes, prompt, **kw)


def generate_text(prompt: str, **kw) -> str:
    """Text -> text, through whichever provider is configured."""
    if provider() == LMSTUDIO:
        from . import vision_lmstudio
        return vision_lmstudio.generate_text(
            prompt,
            url=kw.get('url') or kw.get('ollama_url'),
            model=kw.get('model'),
            num_predict=kw.get('num_predict', 400),
            strict=bool(kw.get('strict')),
            timeout=kw.get('timeout', (10, 120)))
    from . import vision_ollama
    return vision_ollama.generate_text_ollama(prompt, **kw)


def unload_vision_model(**kw) -> bool:
    """Release the resident model. For LM Studio this really frees the VRAM."""
    if provider() == LMSTUDIO:
        from . import vision_lmstudio
        return vision_lmstudio.unload_vision_model(
            url=kw.get('url') or kw.get('ollama_url'), model=kw.get('model'))
    from . import vision_ollama
    return vision_ollama.unload_vision_model(**kw)


def probe_model(reachable=None, model: str | None = None) -> dict:
    """`{ok, detail}` — can the CONFIGURED provider caption right now?

    The passive readiness question, as opposed to :func:`ensure_ready`, which is
    allowed to act (start Ollama) before answering. Both surfaces gate their
    heavy passes on this: the bank refuses to START a watermark, framing or
    caption pass when it is false, and the dataset routes use it to decide
    whether auto head-crop can run.

    It exists because those gates called `probe_ollama_model` directly. That
    reads `ollama.url` and `ollama.vision_model`, so on an install running LM
    Studio — where Ollama is deliberately not running — every one of them
    answered "the vision model is not available" and the whole Bank stopped
    working, while captioning through the router worked fine two lines later.
    """
    # Only forward what was actually asked for. The six gates call this with no
    # arguments, and passing `reachable=None, model=None` down changed the CALL
    # SHAPE the existing suite stubs — two tests double `probe_ollama_model` with a
    # zero-argument lambda, which is exactly how the old call site looked. Routing
    # must be invisible to them.
    kw = {}
    if reachable is not None:
        kw['reachable'] = reachable
    if model is not None:
        kw['model'] = model
    if provider() == LMSTUDIO:
        from ..capabilities import probe_lmstudio_model
        return probe_lmstudio_model(**kw)
    from ..capabilities import probe_ollama_model
    return probe_ollama_model(**kw)


def load_model() -> dict:
    """Make the active provider's vision model resident. `{ok, model?, error?}`.

    LM Studio: resolve the model (configured, else the downloaded VLM) and load it
    through the server's own API -- the answer to "why do I have to keep loading a
    model?". Ollama needs none of this: it loads on demand, so the honest result
    is its ensure_captioning_ready, which can also start the daemon.
    """
    if provider() == LMSTUDIO:
        from . import vision_lmstudio
        target = vision_lmstudio.resolve_model()
        if not target:
            return {'ok': False,
                    'error': 'No vision model is downloaded in LM Studio yet — '
                             'download one there, then come back.'}
        ok, detail = vision_lmstudio.ensure_model_loaded(target)
        return ({'ok': True, 'model': target, 'detail': detail} if ok
                else {'ok': False, 'error': detail})
    from . import ollama_control
    got = ollama_control.ensure_captioning_ready()
    return {'ok': bool(got.get('ok')), 'error': got.get('error')}


def start_server() -> dict:
    """Start the configured provider's local server. `{ok, reachable, ...}`.

    ONE routed path, for the same reason `list_models` is one: the Setup step and
    the Settings card both press this, and two provider-specific endpoints would
    let the two buttons drift apart the way the model pickers did. Never raises;
    an install that cannot be started says why rather than throwing.

    Only ever reached from an explicit click. Nothing passive may call this.
    """
    if provider() == LMSTUDIO:
        from . import lmstudio_control
        return lmstudio_control.start_server()
    from . import ollama_control
    return ollama_control.start_ollama()


def ensure_ready(model: str | None = None) -> dict:
    """`{ok, error}` — can this provider caption right now, after doing what it can.

    The two providers differ in what "doing what it can" means, and pretending
    otherwise is how a button ends up lying:

    * Ollama can be STARTED from here, so a stopped daemon is a recoverable state
      and `ollama_control.ensure_captioning_ready` recovers it.
    * LM Studio has the same two recoveries now — its server can be started
      (lmstudio_control) and its model can be loaded (ensure_model_loaded). This
      branch used to be a bare probe, written when neither existed, and ✨ Enhance
      showed exactly what that costs: the gate refused "qwen/... is not loaded"
      BEFORE generate_text — where the auto-load lives — ever got the call. The
      user's answer was the right review: "I thought LDS loads it itself?"

    Only ever reached from an explicit user action (Enhance, Describe). Passive
    probes stay passive; this function's whole contract is that it may act.
    """
    if provider() == LMSTUDIO:
        from ..capabilities import probe_lmstudio_model
        from . import vision_lmstudio
        if not vision_lmstudio.list_models().get('reachable'):
            from . import lmstudio_control
            started = lmstudio_control.start_server()
            if not started.get('reachable'):
                return {'ok': False,
                        'error': started.get('error')
                        or f'LM Studio unreachable: {vision_lmstudio.base_url()}'}
        target = (model or '').strip() or vision_lmstudio.resolve_model()
        if not target:
            return {'ok': False,
                    'error': 'No vision model is downloaded in LM Studio yet — '
                             'download one there, then come back.'}
        loaded, detail = vision_lmstudio.ensure_model_loaded(target)
        if not loaded:
            return {'ok': False, 'error': detail}
        verdict = probe_lmstudio_model(model=model)
        return {'ok': verdict['ok'], 'error': None if verdict['ok'] else verdict['detail']}
    from . import ollama_control
    return ollama_control.ensure_captioning_ready(model)


def list_models() -> dict:
    """``{ok, reachable, models: [str]}`` — the shape the model pickers already read.

    Kept deliberately identical to what ``/api/ollama/models`` has always returned,
    so both surfaces' pickers (dataset AND bank) can switch endpoint without any
    change to how they read the answer.
    """
    if provider() == LMSTUDIO:
        from . import vision_lmstudio
        listed = vision_lmstudio.list_models()
        return {'ok': listed['ok'], 'reachable': listed['reachable'],
                'provider': LMSTUDIO,
                'models': [m['id'] for m in listed['models']
                           if m['id'] and m.get('type') != 'embeddings']}
    from . import ollama_control
    out = dict(ollama_control.list_models())
    out['provider'] = OLLAMA
    return out
