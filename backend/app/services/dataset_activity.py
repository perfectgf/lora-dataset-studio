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
         'analyze_faces', 'classify', 'generate')

# Safety TTL: an entry not touched for this long is purged on read even if end()
# never ran (process alive but the batch thread died without unwinding). 30 min is
# far longer than any real batch on a local dataset.
_TTL_SECONDS = 30 * 60

_lock = threading.Lock()
# dataset_id -> { token -> {kind, done, total, started_at, _touched} }
_active: dict = {}
_counter = itertools.count(1)


def begin(dataset_id, kind, total=0, detail=None, engine=None):
    """Register a new in-progress batch on ``dataset_id`` and return an opaque token
    to pass to ``progress``/``bump``/``end``. ``total`` is the number of items the
    batch will process (0 when not enumerable up front)."""
    now = time.time()
    with _lock:
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
    batches overlap, the most recently STARTED one is returned (see module note)."""
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
        entry = max(bucket.values(), key=lambda e: e['started_at'])
        result = {'kind': entry['kind'], 'done': entry['done'],
                  'total': entry['total'], 'started_at': entry['started_at']}
        if entry.get('detail'):
            result['detail'] = entry['detail']
        if entry.get('engine'):
            result['engine'] = entry['engine']
        return result


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
