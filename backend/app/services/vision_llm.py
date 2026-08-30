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
    raw = (cfg.get('local_llm.provider') or OLLAMA).strip().lower()
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
    raw = cfg.get(key)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 4


def keep_warm_seconds(name: str | None = None) -> int:
    """Seconds an isolated call may keep the model resident, for the ACTIVE provider."""
    key = f'{name or provider()}.vision_keep_warm_seconds'
    raw = cfg.get(key)
    try:
        return int(raw)
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
