"""Contention-driven `keep_alive` policy for the Ollama vision model.

The problem
-----------
Loading the vision model costs **12.8 s** on the reference machine; a warm
inference costs **0.52 s**. A cold call is therefore ~25x a warm one. Until now
`describe_image_ollama` defaulted to `keep_alive=0`, so every *isolated* call
(auto head-crop on a reference upload, a re-crop, a single watermark probe) paid
that 12.8 s in full — and a user cropping five references in a row paid it five
times.

The naive fix — "just keep it warm" — is worse than it looks. The vision model's
real footprint is **7.5 GB** (qwen3-vl 8b + an 8k KV cache), and a loaded ComfyUI
sits at ~19.1 GB of the 24.5 GB available. 19.1 + 7.5 = 26.6 > 24.5: they do not
both fit. Under WDDM, Windows does not raise an OOM on VRAM over-commitment — the
driver pages silently and `nvidia-smi` shows nothing. A measured vision pass with
ComfyUI resident ran at **6.99 s/img instead of 0.52 s/img (13.5x slower)**, with
Ollama itself reporting a 43%/57% CPU/GPU split. "No crash" is not "they
cohabit".

So holding 7.5 GB on speculation is only acceptable if we can **give it back on
demand**. That is what this module is: a *lease*, not a timer.

The design: revocation, not prediction
--------------------------------------
The tempting signal is "poll ComfyUI's `/queue` before deciding". It is the wrong
instrument, and this is the crux of the whole design:

  A poll answers "is somebody busy **right now**". The risk we are taking is
  "will somebody want the card **in the next 120 s**". A queue that is empty at
  decision time says nothing about the ComfyUI job the user fires ten seconds
  later — and by then the poll is long over. A point-in-time signal cannot cover
  the window it authorises.

The signals that *can* cover the window are the ones LDS controls, because LDS is
notified by construction rather than by asking:

  * `training_in_progress` — a system-state flag LDS itself writes.
  * `ImageGenerationQueue` — LDS's own ComfyUI job queue, a local table.

Both are plain database reads: no network, no timeout, nothing that can hang a
caption call. And both are *hookable*: `revoke()` below is called from the exact
code paths that take the GPU (`job_queue.process_one` right before submitting to
ComfyUI, `lora_training` right before spawning ai-toolkit), so a held lease is
handed back before the contender loads anything.

What this deliberately does NOT cover (stated honestly)
-------------------------------------------------------
Work submitted to ComfyUI by *something other than LDS* — ComfyUI's own web UI,
another app on the same machine. LDS gets no notification for that and, per the
argument above, polling would not help either. The exposure is bounded by
`warm_seconds()` (default 120 s, configurable, 0 disables the whole feature) and
by ComfyUI's own `load_models_gpu()`, which sizes its loads against the VRAM it
actually finds free. That degradation path is *reasoned*, not measured — see the
handover notes.

Also not covered, deliberately:

  * **A CUDA scoring pass** (face similarity / quality, `gpu_exclusive_vision_window`
    with `device == 'cuda'`). It is not in the contention signal and does not
    revoke. Two reasons: it announces itself with `vision_in_progress`, the very
    flag the isolated vision calls set for *themselves* — reading it would make
    the policy answer "contended" for the flagship head-crop case, which runs
    inside such a window. And its footprint (a few GB of torch) is nothing like
    ComfyUI's 19 GB, so a stale lease costs it some headroom for at most
    `warm_seconds()`, not a silent CPU spill. Revisit if the signal ever gains a
    dedicated flag.
  * **An LDS restart.** The lease is process-local state; after a restart the
    model may stay resident for the remainder of its Ollama-side TTL with no
    lease to revoke. Bounded by the same `warm_seconds()`.

Why "unknown" means "unload"
----------------------------
Every failure to read the contention signal resolves to *contended*. `keep_alive=0`
is exactly today's behaviour, so an unreadable signal degrades to the status quo
instead of to a 7.5 GB gamble. A policy that guesses "probably free" when it
cannot tell would trade a predictable reload for an unpredictable eviction, which
is the trade this module exists to avoid.
"""
from __future__ import annotations

import logging
import threading
import time

from .. import config as cfg

logger = logging.getLogger(__name__)

# `keep_alive` value meaning "unload the model as soon as this call returns".
UNLOAD = 0

# How long an uncontended isolated call may keep the model resident.
# 120 s covers a human clicking through a handful of reference crops (the burst
# this feature exists for) without holding 7.5 GB across a coffee break.
DEFAULT_WARM_SECONDS = 120
# Ceiling for the configured value. Past this the lease outlives any plausible
# burst and just becomes a VRAM squat waiting for a revocation that may never
# come (see the restart caveat above).
MAX_WARM_SECONDS = 600

_lock = threading.Lock()
_lease_until = 0.0  # time.monotonic() deadline of the lease we last granted


def warm_seconds(override=None) -> int:
    """Lease length in seconds. 0 (or any unusable value resolving to 0) turns
    the feature off entirely and restores the old always-unload behaviour.

    Total by construction, like `vision_pool.vision_concurrency`: the settings
    form stores whatever the user typed, so a blank, a word or a negative must
    degrade to something usable. Result is always an int in 0..MAX_WARM_SECONDS.
    """
    # Same reasoning as vision_pool.vision_concurrency: read the ACTIVE
    # provider's dial, or an LM Studio user gets a control that changes nothing.
    if override is not None:
        raw = override
    else:
        from . import vision_llm
        raw = cfg.get(f'{vision_llm.provider()}.vision_keep_warm_seconds')
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_WARM_SECONDS
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning('vision_keepalive: ignoring unusable '
                       'ollama.vision_keep_warm_seconds %r', raw)
        return DEFAULT_WARM_SECONDS
    return max(0, min(MAX_WARM_SECONDS, value))


def gpu_is_contended() -> bool:
    """Is another LDS GPU consumer running or waiting?

    True when a training run holds the card, or when LDS's own ComfyUI queue has
    any job that is not finished. Reads two local tables — never touches the
    network, so it cannot block a caption call. Any error answers True: see the
    "unknown means unload" note in the module docstring.
    """
    try:
        from ..job_queue import queue_manager
        if queue_manager._get_system_state('training_in_progress'):
            return True
        if queue_manager.has_comfyui_stalled_barrier():
            return True
        from ..models import ImageGenerationQueue
        pending = (ImageGenerationQueue.query
                   .filter(ImageGenerationQueue.status.in_(
                       ('pending', 'processing', 'sent_to_comfy',
                        'cancel_requested', 'stalled')))
                   .first())
        return pending is not None
    except Exception as exc:
        # No app/database context, a migration in flight, anything at all: the
        # safe answer is the current behaviour.
        logger.debug('vision_keepalive: contention signal unreadable (%s) '
                     '-> assuming contended', exc)
        return True


def keep_alive_for_isolated_call():
    """`keep_alive` an ISOLATED vision call should use.

    Returns `UNLOAD` when the card is contended (or when the signal cannot be
    read), otherwise an Ollama duration string, and records the lease so
    `revoke()` knows there is something to hand back.

    Batch passes do NOT go through here: they already know they are about to
    issue N calls and pass their own `keep_alive`, then unload in a `finally`.
    """
    seconds = warm_seconds()
    if seconds <= 0:
        return UNLOAD
    if gpu_is_contended():
        return UNLOAD
    global _lease_until
    with _lock:
        _lease_until = time.monotonic() + seconds
    return f'{seconds}s'


def lease_is_live() -> bool:
    """Whether a keep-warm lease granted by this process may still be resident."""
    with _lock:
        return time.monotonic() < _lease_until


def forget_lease() -> None:
    """Drop the lease without calling Ollama — for callers that already unloaded
    (batch passes end with their own `unload_vision_model()`)."""
    global _lease_until
    with _lock:
        _lease_until = 0.0


def revoke(reason: str = '') -> bool:
    """Hand the warm vision lease back only after Ollama confirmed the unload.

    A failed unload deliberately keeps the lease live. The ComfyUI queue uses
    this fact as a fail-closed gate: it must not submit a model-loading workflow
    while this process still believes Ollama owns the card.
    """
    if not lease_is_live():
        return False
    try:
        # Through the router: Ollama is asked to stop keeping the model warm,
        # LM Studio is actually told to unload it. Importing the Ollama driver by
        # name would end an LM Studio lease by talking to a daemon holding nothing.
        from .vision_llm import unload_vision_model
        ok = unload_vision_model()
    except Exception as exc:
        logger.warning('vision_keepalive: revoke failed (%s)', exc)
        return False
    if not ok:
        logger.warning('vision_keepalive: keeping the lease because Ollama did not confirm unload')
        return False
    forget_lease()
    logger.info('vision_keepalive: released the warm vision model%s',
                f' ({reason})' if reason else '')
    return True


def ensure_released_for_comfy(reason: str = 'ComfyUI job starting') -> bool:
    """Prove local Ollama no longer owns GPU memory before a ComfyUI handoff.

    A live warm lease is revoked once. The ownership fence then verifies the
    configured loopback Ollama runner is empty, including after an LDS restart
    where no in-memory lease exists. Its verified-empty cache makes later cells
    in the same ComfyUI batch hot: no unload and no repeated network probe.
    """
    if lease_is_live() and not revoke(reason):
        return False
    try:
        from . import ollama_gpu_fence
        return ollama_gpu_fence.ensure_released_for_comfy()
    except Exception:
        logger.exception('vision_keepalive: could not verify local Ollama release')
        return False
