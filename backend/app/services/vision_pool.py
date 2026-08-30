"""Bounded concurrency for the vision (Ollama) passes.

Why this exists
---------------
The bank's vision passes (watermark, framing, caption) call Ollama once per
image, one after the other, over banks of tens of thousands of files. Measured
on the reference machine, an average call takes ~1.7 s of which only a fraction
is GPU work: the rest is HTTP round-trip, base64 payload handling and
server-side request preparation. That part is *waiting*, and waiting overlaps.
The GPU is not saturated during these passes, which is what makes overlapping
them a real gain rather than a reshuffle: 24 images took 41.2 s sequentially and
20.3 s with 4 concurrent calls — **2.0x** — on the same sample and model.

Why a helper instead of a ThreadPoolExecutor at each call site
--------------------------------------------------------------
Three things have to hold at once, and a naive ``executor.map`` breaks all of
them:

1. **The SQLAlchemy session is not shareable across threads.** So the split here
   is absolute: worker threads do the network call and hand back a plain value;
   *every* database read and write stays on the thread that owns the job. That
   is enforced by shape, not by convention — a worker only ever sees whatever
   opaque item the caller pushed in.
2. **A pass must stay stoppable.** The bank's cooperative Stop has to keep
   working, and stopping must not lose an answer we already paid for. See
   ``map_vision``'s cancellation contract below.
3. **The concurrency has to be bounded and configurable.** Ollama serves a
   limited number of requests in parallel (``OLLAMA_NUM_PARALLEL``); past that
   it simply queues, so an unbounded pool buys nothing and only inflates
   per-call latency (hence a slower Stop) for no throughput.
"""
from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .. import config as cfg

logger = logging.getLogger(__name__)

# Default number of Ollama calls in flight per vision pass.
#
# 4 is the measured knee, not a guess. On the reference sample (24 images,
# qwen3-vl-abliterated:8b-instruct): 1 worker 41.2 s, 2 workers 27.5 s (1.50x),
# 4 workers 20.3 s (2.03x), 6 workers 18.2 s (2.27x), 8 workers 17.3 s (2.39x).
# Everything past 4 buys single-digit percentages while per-call latency climbs
# roughly linearly (1.7 s -> 3.3 s at 4, 5.1 s at 8) — and that latency is
# exactly the time a Stop has to wait for the calls already in flight.
#
# Correction to an earlier note here: this does NOT "match Ollama's own default
# parallelism". Ollama's `server/sched.go` keeps a deny-list of model families
# that are forced to `numParallel = 1` regardless of `OLLAMA_NUM_PARALLEL`, and
# `qwen3vl` is on it — so the server runs our calls one at a time. The measured
# 2.03x is therefore NOT GPU parallelism; it is the overlap of the parts that
# aren't GPU work (HTTP round-trip, base64 handling, request preparation). The
# number stands, the explanation didn't.
DEFAULT_VISION_CONCURRENCY = 4
# Hard ceiling for the configured value. Nothing breaks above it — Ollama just
# queues — but it would trade Stop responsiveness for no throughput, so the
# knob clamps instead of obeying.
MAX_VISION_CONCURRENCY = 16


def vision_concurrency(override=None) -> int:
    """How many Ollama calls a vision pass may keep in flight.

    ``override`` (a caller-supplied value) wins, otherwise
    ``ollama.vision_concurrency`` from the config, otherwise
    ``DEFAULT_VISION_CONCURRENCY``.

    Total by construction: the settings form stores whatever the user typed, so
    a blank, a word, a negative or an absurd number must degrade to something
    usable rather than sink the pass. The result is always an int clamped to
    1..MAX_VISION_CONCURRENCY, and 1 means "run exactly as before".
    """
    # The ACTIVE provider's dial, not Ollama's by definition: with a hard-coded
    # key an LM Studio user would see a Settings control that changes nothing.
    if override is not None:
        raw = override
    else:
        from . import vision_llm
        raw = cfg.get(f'{vision_llm.provider()}.vision_concurrency')
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_VISION_CONCURRENCY
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning('vision_pool: ignoring unusable ollama.vision_concurrency %r', raw)
        return DEFAULT_VISION_CONCURRENCY
    return max(1, min(MAX_VISION_CONCURRENCY, value))


_NOTHING = object()


def _guarded(work, item):
    """Run one unit of work on a worker thread, never letting it raise.

    An exception here would otherwise surface out of ``future.result()`` on the
    consuming thread and take the whole pass down with it. The passes already
    treat a single failed image as a counted error, and that must survive
    parallelisation, so the failure is carried back as a value.
    """
    try:
        return work(item), None
    except Exception as exc:  # noqa: BLE001 — one bad image never sinks the pass
        return None, exc


def map_vision(items, work, *, workers=None, should_cancel=None):
    """Run ``work(item)`` over ``items`` on a bounded pool, yielding
    ``(item, result, error)`` triples **in item order, on the calling thread**.

    ``items``   an iterable pulled on the CALLER's thread, lazily, one item per
                free slot. Anything the caller must do before a call — a DB
                read, resolving a path, dropping a stale blob — belongs in that
                iterable: it then runs on the owning thread, in order, and only
                for images the pass actually reaches.
    ``work``    called on a WORKER thread. Network and pure-CPU work only. It
                must not touch the SQLAlchemy session, the Flask app context, or
                any object whose attribute access could lazy-load from the DB.
    ``workers`` concurrency override; see ``vision_concurrency``. 1 runs fully
                inline, with no pool at all.
    ``should_cancel``  polled before every submission.

    Yields ``(item, result, None)`` on success and ``(item, None, exception)``
    when ``work`` raised, so the caller keeps its per-image error accounting.

    Cancellation contract
    ---------------------
    On cancel we stop *feeding* the pool immediately, then drain and yield what
    is already in flight. That is deliberate, and it is what makes Stop safe:

    * nothing is submitted after the flag flips, so no orphan calls are left
      running behind the pass;
    * every call that was already paid for is still handed to the caller, which
      still persists it — "what was processed stays processed", and a resumed
      pass does not redo it;
    * the window is exactly ``workers`` wide and every submitted item is
      immediately picked up by a thread, so nothing is ever pulled from
      ``items`` and then dropped unprocessed (which would waste the caller's
      per-item preparation — for the watermark pass, that preparation is
      destructive). Both paths test the flag BEFORE pulling, for that reason
      alone: at ``workers=1`` this is what stops a Stop from destroying the
      cleaned file of an image it will never analyse.

    The cost is the stop latency: draining takes about one call round (~3 s at
    the default concurrency) instead of the ~1.7 s a sequential pass waits for
    its current call. Seconds either way — and the alternative, abandoning
    in-flight answers, would silently re-do work on resume.
    """
    count = vision_concurrency(workers)
    iterator = iter(items)
    if count <= 1:
        while True:
            # The flag is tested BEFORE pulling, exactly like submit_next()
            # below. `for item in iterator` reads the item first and can only
            # test afterwards — which pulled one item through the caller's
            # preparation and then dropped it unprocessed on every Stop. That
            # preparation is destructive for the watermark pass (it discards
            # the cleaned blob), so the sequential path was deleting a user's
            # cleaned file and storing nothing in its place.
            if should_cancel and should_cancel():
                return
            item = next(iterator, _NOTHING)
            if item is _NOTHING:
                return
            result, error = _guarded(work, item)
            yield item, result, error
    # Abandoning the generator (a `break`, or an exception in the consumer)
    # exits this `with`, which joins the pool — a pass can never leave worker
    # threads calling Ollama behind it.
    # Carry only the Vision ownership token into workers. Deliberately do not
    # copy the full execution context: that could leak Flask's app/request or
    # SQLAlchemy state across threads.
    from ..gpu_window import bind_vision_window_context
    guarded_work = bind_vision_window_context(_guarded)

    with ThreadPoolExecutor(max_workers=count, thread_name_prefix='vision') as pool:
        inflight = deque()

        def submit_next():
            if should_cancel and should_cancel():
                return
            item = next(iterator, _NOTHING)
            if item is not _NOTHING:
                inflight.append((item, pool.submit(guarded_work, work, item)))

        for _ in range(count):
            submit_next()
        while inflight:
            item, future = inflight.popleft()
            result, error = future.result()
            # Refill BEFORE handing the result over, so the caller's database
            # write overlaps the next call instead of stalling the pool.
            submit_next()
            yield item, result, error
