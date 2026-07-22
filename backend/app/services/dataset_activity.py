"""In-memory per-dataset batch-activity registry.

Long batch operations on a dataset — watermark detect/clean, caption/re-caption,
face analysis, framing classification — run server-side inside a request thread.
The UI's "in progress" indicator used to be React-local state, so reloading the
page lost it while the server kept working. This registry lets the dataset payload
advertise a live ``activity`` object the front-end can RESTORE on reload and poll
to completion.

Design notes
------------
* **In-memory ONLY (no DB).** A batch dies with the process; on restart the
  registry is empty, so the (now-dead) indicator correctly disappears — nothing to
  clean up, no phantom "in progress" persisted anywhere.
* **Thread-safe.** A single module lock guards the store; these batches run in
  request threads and two datasets can be worked in parallel.
* **Crash-proof.** ``begin``/``end`` are meant to be used with ``try/finally`` so a
  batch that raises never leaves a phantom entry. As a belt-and-braces safety net a
  per-entry TTL is purged on every read: even if ``end`` were somehow skipped, the
  indicator can never outlive ``_TTL_SECONDS``.
* **One indicator per dataset.** The GPU-exclusive vision window serializes GPU
  passes (including watermark cleaning when CUDA is selected); CPU passes are
  guarded client-side by the hook's single-flight ``wrap`` on this
  single-local-user app — so overlapping
  kinds don't happen in normal use. Should two ever overlap (e.g. two browser
  tabs firing a GPU and a CPU pass at once), ``get`` returns the most recently
  STARTED one; the UI restores a single indicator, which is acceptable.
"""
import itertools
import threading
import time

# Kinds the UI knows how to restore. Kept as a documented allow-list so a typo in a
# begin() call is easy to spot (nothing enforces it — it's documentation + a guard
# for tests). 'generate' covers the ⚡ Generate-variations batch (Nano Banana /
# ChatGPT / Klein) — it keeps the Generate button (and every concurrent action)
# disabled for the WHOLE batch, not just the launch request.
KINDS = ('watermark_detect', 'watermark_clean', 'caption', 'recaption',
         'analyze_faces', 'classify', 'generate', 'improve')

# Kinds a user can gracefully STOP mid-batch (the ▶ Stop button). Only the
# per-image captioning passes qualify: the worker checks the cancel flag at each
# image boundary and stops cleanly, keeping what it already wrote. The others are
# either near-instant (classify), one-shot subprocess passes with no cooperative
# seam, or already Stop-able through their own path ('generate' → cancel_pending).
CANCELLABLE_KINDS = ('caption', 'recaption')

# The 'improve' kind is the server-side ✨ Klein upscale & improve batch. It is
# ALSO cooperatively stoppable, but through the ⏹ Stop generation button
# (cancel_pending) rather than the captioning Stop — so it gets its own arming
# scope instead of joining CANCELLABLE_KINDS, whose default is what the caption
# worker polls. Every kind here is disarmed by begin() so a leaked flag from a
# previous run can never cancel a fresh one.
IMPROVE_KINDS = ('improve',)
STOPPABLE_KINDS = CANCELLABLE_KINDS + IMPROVE_KINDS

# Safety TTL: an entry not touched for this long is purged on read even if end()
# never ran (process alive but the batch thread died without unwinding). 30 min is
# far longer than any real batch on a local dataset.
_TTL_SECONDS = 30 * 60

_lock = threading.Lock()
# dataset_id -> { token -> {kind, done, total, started_at, _touched} }
_active: dict = {}
# dataset_id -> set of KINDS for which a graceful stop has been REQUESTED. Armed by
# request_cancel (only for kinds actually live), read by the worker loop between
# items, and cleared on the next begin() of that kind so a leaked/stale arm can
# never cancel a fresh run. Per-kind rather than a single boolean: two independent
# Stop buttons feed it (captioning's ⏹ Stop and the generation ⏹ Stop, which also
# stops the 'improve' batch), and stopping one must never silently stop the other.
# Lives with _active under _lock; in-memory only (dies with the process).
_cancel: dict = {}
_counter = itertools.count(1)


def begin(dataset_id, kind, total=0, detail=None, engine=None):
    """Register a new in-progress batch on ``dataset_id`` and return an opaque token
    to pass to ``progress``/``bump``/``end``. ``total`` is the number of items the
    batch will process (0 when not enumerable up front)."""
    now = time.time()
    with _lock:
        # Fresh cancellable batch → disarm any stale/leaked stop request FOR THAT KIND
        # so it can never cancel this new run before it starts (a prior run that armed
        # the flag then crashed before it was consumed would otherwise poison this one).
        if kind in STOPPABLE_KINDS:
            _discard_cancel(dataset_id, (kind,))
        token = f'{dataset_id}:{kind}:{next(_counter)}'
        _active.setdefault(dataset_id, {})[token] = {
            'kind': kind, 'done': 0, 'total': int(total or 0),
            'started_at': now, '_touched': now,
        }
        if detail:
            _active[dataset_id][token]['detail'] = str(detail)
        if engine:
            _active[dataset_id][token]['engine'] = str(engine).lower()
    return token


def progress(token, done=None, total=None, detail=None):
    """Set the item counter (and optionally the total) for a running batch.
    No-op on an unknown/None token (already ended or purged)."""
    now = time.time()
    with _lock:
        entry = _entry(token)
        if entry is None:
            return
        if done is not None:
            entry['done'] = int(done)
        if total is not None:
            entry['total'] = int(total)
        if detail is not None:
            entry['detail'] = str(detail)
        entry['_touched'] = now


def bump(token, n=1):
    """Increment the item counter by ``n`` — convenience for per-image loops.
    No-op on an unknown/None token."""
    now = time.time()
    with _lock:
        entry = _entry(token)
        if entry is None:
            return
        entry['done'] += n
        entry['_touched'] = now


def end(token):
    """Remove a batch's entry. Idempotent (safe on an unknown/None token) so a
    ``finally``-block ``end`` never raises even if the entry was already purged."""
    dsid = _dsid_of(token)
    with _lock:
        bucket = _active.get(dsid)
        if not bucket:
            return
        bucket.pop(token, None)
        if not bucket:
            _active.pop(dsid, None)


def sync_pending(dataset_id, kind, pending, engine=None):
    """Reconcile a COUNT-tracked indicator of ``kind`` against a live in-flight
    total. Used where per-batch tracking isn't available — a Klein generate batch
    completes one job at a time on the job-queue monitor thread, and each
    completion callback holds only a ``job_id`` (no batch handle); completions can
    also be duplicated (retry) or bypassed entirely (Stop deletes the rows without
    a completion). So instead of a fragile per-batch job set we track the honest
    "how many are still in flight" number read straight from the DB:

    * ``pending > 0`` — ensure an entry exists, grow ``total`` to the high-water
      mark of items ever seen in flight, and set ``done = total - pending``.
    * ``pending <= 0`` — the batch is finished: clear the entry.

    Only ever touches the entry IT created (tagged ``_synced``), so it can coexist
    with a worker-owned ``begin``/``end`` entry of the same kind (e.g. an API batch)
    without corrupting it. Idempotent — safe to call on every enqueue and every
    completion. TTL purge (via ``get``) is the final safety net if a completion is
    lost and ``pending`` never reaches 0."""
    now = time.time()
    with _lock:
        bucket = _active.get(dataset_id) or {}
        tok = next((t for t, e in bucket.items()
                    if e['kind'] == kind and e.get('_synced')), None)
        if pending <= 0:
            if tok:
                bucket.pop(tok, None)
                if not bucket:
                    _active.pop(dataset_id, None)
            return
        if tok is None:
            bucket = _active.setdefault(dataset_id, {})
            tok = f'{dataset_id}:{kind}:{next(_counter)}'
            bucket[tok] = {'kind': kind, 'done': 0, 'total': int(pending),
                           'started_at': now, '_touched': now,
                           '_peak': int(pending), '_synced': True}
        entry = bucket[tok]
        if engine:
            entry['engine'] = str(engine).lower()
        entry['_peak'] = max(entry['_peak'], int(pending))
        entry['total'] = entry['_peak']
        entry['done'] = max(0, entry['_peak'] - int(pending))
        entry['_touched'] = now


def get(dataset_id):
    """Return the current activity on ``dataset_id`` as
    ``{kind, done, total, started_at}`` or ``None``. Purges TTL-expired entries
    first, so a leaked entry can never strand a phantom indicator. When several
    batches overlap, a WORKER-OWNED entry (begin/end) wins over a ``sync_pending``
    one, then the most recently STARTED one (see module note). Rationale: a synced
    entry is a *reconstruction* from the live in-flight count — it knows nothing of
    the batch that produced it. The ✨ improve batch owns a real handle AND drives
    in-flight generations, so both entries exist at once; the handle carries the
    honest done/total (250 images, not the 60 currently in flight)."""
    now = time.time()
    with _lock:
        bucket = _active.get(dataset_id)
        if not bucket:
            return None
        stale = [t for t, e in bucket.items() if now - e['_touched'] > _TTL_SECONDS]
        for t in stale:
            bucket.pop(t, None)
        if not bucket:
            _active.pop(dataset_id, None)
            return None
        entry = max(bucket.values(),
                    key=lambda e: (0 if e.get('_synced') else 1, e['started_at']))
        result = {'kind': entry['kind'], 'done': entry['done'],
                  'total': entry['total'], 'started_at': entry['started_at']}
        if entry.get('detail'):
            result['detail'] = entry['detail']
        if entry.get('engine'):
            result['engine'] = entry['engine']
        # A stop was requested but the worker hasn't reached the next item boundary
        # yet — the UI flips the Stop button to a disabled "Stopping…" state. Only
        # when the REPORTED batch is the one being stopped.
        if entry['kind'] in (_cancel.get(dataset_id) or ()):
            result['cancelling'] = True
        return result


def running(dataset_id, kinds):
    """True when a batch of one of ``kinds`` is live on ``dataset_id``. Used to refuse
    a second ✨ improve batch (-> 409) instead of racing two workers on one cap."""
    with _lock:
        return any(e['kind'] in kinds for e in (_active.get(dataset_id) or {}).values())


def request_cancel(dataset_id, kinds=CANCELLABLE_KINDS):
    """Ask the running batch(es) of ``kinds`` on ``dataset_id`` to stop at their next
    item boundary. Arms the flag ONLY for kinds actually live, so the caller (route)
    can answer 409 when there is nothing to stop. Idempotent: a second call while the
    same batch still runs simply re-arms (returns True again).

    We never interrupt an in-flight inference — the worker finishes the current item,
    then sees the flag and unwinds through the SAME cleanup as a normal finish (model
    unload, indicator end). Returns True when a batch was live and is now flagged."""
    with _lock:
        bucket = _active.get(dataset_id) or {}
        live = {e['kind'] for e in bucket.values() if e['kind'] in kinds}
        if not live:
            return False
        _cancel.setdefault(dataset_id, set()).update(live)
        return True


def cancel_requested(dataset_id, kinds=CANCELLABLE_KINDS):
    """True when a graceful stop is pending for one of ``kinds`` on ``dataset_id``.
    Called by the caption worker between images (default scope — the caption family)
    and by the improve worker with ``IMPROVE_KINDS``, plus by the routes to learn a
    pass ended because it was stopped. Cheap and lock-guarded — safe to poll in a
    tight per-item loop."""
    with _lock:
        return bool((_cancel.get(dataset_id) or set()) & set(kinds))


def clear_cancel(dataset_id, kinds=CANCELLABLE_KINDS):
    """Consume the stop flag of ``kinds`` for ``dataset_id`` (idempotent). The route
    clears it once the whole caption operation has unwound so it can never bleed into
    a later run — begin() also clears defensively, this just makes the intent explicit
    at the seam. Scoped by kind so unwinding a caption pass never disarms a concurrent
    improve batch that the user has just asked to stop."""
    with _lock:
        _discard_cancel(dataset_id, kinds)


def _discard_cancel(dataset_id, kinds):
    """Drop ``kinds`` from the armed stop set of ``dataset_id``. Caller holds ``_lock``."""
    armed = _cancel.get(dataset_id)
    if not armed:
        return
    armed.difference_update(kinds)
    if not armed:
        _cancel.pop(dataset_id, None)


def _entry(token):
    """The mutable entry dict for ``token``, or None. Caller holds ``_lock``."""
    return (_active.get(_dsid_of(token)) or {}).get(token)


def _dsid_of(token):
    try:
        return int(str(token).split(':', 1)[0])
    except (ValueError, AttributeError):
        return None


def reset():
    """Test helper: clear the whole registry between cases."""
    with _lock:
        _active.clear()
        _cancel.clear()
