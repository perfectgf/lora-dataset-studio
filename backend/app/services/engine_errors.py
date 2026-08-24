"""Shared failure taxonomy for the three API image engines.

WHY THIS EXISTS
---------------
The OpenRouter engine shipped with a rule the two older engines did not follow:
every failure RAISES with a named cause, and the causes that would repeat
identically on every remaining row of a batch stop the run instead of paying for
the same refusal once per image. Nano Banana and ChatGPT swallowed everything
into `return None`, which the fan-out words as "empty response (often a
content-policy refusal)" — the wrong sentence for a rejected key, an unknown
model, or a model that cannot take reference images at all.

Making the engine model user-editable is what forced the alignment: a typo'd or
text-only model slug is now a routine, self-inflicted, fixable mistake, and it
must say so instead of sending the user to rewrite a prompt.

TWO LEVELS, ONE MEANING EACH
----------------------------
`EngineError`  — this image failed, for a named reason. The batch continues
                 (rate limit, provider hiccup, one refused request).
`EngineFatal`  — every remaining row would fail for the exact same reason (no
                 key, key rejected, no credits/quota, unknown or unusable
                 model). The fan-out stops the run on the first one.

`EngineRefused` — the provider answered SUCCESSFULLY and chose not to return an
                 image. This is not a malfunction and it is not fatal: the next
                 row may well succeed, so the batch continues. It exists because
                 "the provider refused you" and "the provider is broken" need
                 opposite remedies, and one shared sentence for both was a lie in
                 whichever direction it happened to be wrong.

`None` still means: the provider answered 200 and produced no image, WITHOUT
saying why. Engines that can read the refusal out of the response raise
`EngineRefused` instead — only the ones that genuinely cannot tell a refusal
from a hiccup are still allowed to shrug.

The per-engine subclasses below stay so a caller (and a test) can still name the
provider it is talking to; the fan-out catches only the two base classes.
No message in this hierarchy ever carries an API key or a fragment of one.
"""
from __future__ import annotations


class EngineError(RuntimeError):
    """A named image-engine failure. The message is user-facing (it lands in a
    tile's fail_reason) and never carries a secret."""


class EngineFatal(EngineError):
    """A failure that would repeat identically on every remaining row of a batch.
    Callers stop the run rather than re-ask a question already refused."""


class EngineRefused(EngineError):
    """The provider answered normally and declined to produce this image.

    Deliberately NOT an EngineFatal: a refusal is per-request. Providers that
    filter their own output are not deterministic about it, so stopping the run
    on the first refusal would throw away the rows that would have succeeded.
    Callers should count these separately from real failures — a batch that ends
    with "9 refused" is a policy outcome, not a broken app."""


def provider_error_message(resp) -> str:
    """The provider's own explanation for a failed request, trimmed.

    Gemini and OpenRouter both answer the documented envelope
    {"error": {"code", "message", ...}}; an edge/proxy failure can answer
    something else entirely, so fall back to the raw text. Handing back the
    provider's exact words is the point - a generic 'request failed' leaves
    the user with nothing to act on. One body for both engines (they were
    byte-identical copies); each keeps its historical `_error_message` name
    as an import alias so per-module monkeypatching still works."""
    try:
        body = resp.json()
    except Exception:                                  # noqa: BLE001 — non-JSON edge error
        body = None
    if isinstance(body, dict):
        err = body.get('error')
        if isinstance(err, dict):
            msg = str(err.get('message') or '').strip()
            if msg:
                return msg[:300]
        elif isinstance(err, str) and err.strip():
            return err.strip()[:300]
    try:
        return (resp.text or '').strip()[:300]
    except Exception:                                  # noqa: BLE001
        return ''
