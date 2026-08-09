"""Background runner + result store for the concept face-mask PREVIEW.

Why this is not a plain synchronous POST any more
-------------------------------------------------
The preview used to be one blocking request. That made two things impossible at
once, and they are the same bug seen from two sides:

* nothing could be shown WHILE it ran — the button said "Looking for faces…" for
  the whole InsightFace pass (tens of seconds before image 1 even starts) and a
  crash looked identical to a slow run;
* nothing could be REJOINED. The browser tab that issued the request is the only
  holder of its handle, so leaving the page threw the pass away — and coming back
  offered to start a second full pass over the same images, possibly next to the
  first one still burning CPU.

Streaming the response would have fixed only the first. A server-side job fixes
both, so that is what this is.

Why not the two job registries already here
-------------------------------------------
* `dataset_activity` is the dataset-wide BUSY flag: entering it blocks generation
  on that dataset and lights the global activity banner. A read-only preview that
  writes nothing must not seize the dataset — see the gotcha noted in the wave
  report; a unified registry is a refactor, not this fix.
* `bank_jobs` is keyed by bank id in a single module dict; reusing it for dataset
  ids would collide two id namespaces in one keyspace.

So: the same design CONTRACT as bank_jobs (in-memory only, thread-safe, one live
job per key, inline under TESTING), keyed by dataset.

Staleness
---------
A stored preview is only true of the exact kept set it was computed from, so it
is stored with a fingerprint of that set (id + filename + size + mtime of every
kept image). Reads recompute it and flag `stale`. A preview shown as fresh after
the images changed would be worse than no preview at all — the user would trust
boxes drawn from photos that are no longer in the run.

Stopping, and why it comes with a resume
----------------------------------------
A Stop that threw the pass away would be a button that LOOKS like it saves time
and costs it: the detector load is a fixed price paid before image 1 (measured
~4-5 s warm on the reference machine, tens of seconds cold, plus a ~350 MB
download on the very first run), so a discard-everything Stop makes the second
attempt cost the whole pass again. The faces already found are the expensive
part, so they are kept: `remember_partial` banks them and the next start hands
the child only the images that are left.

The cache is guarded by the SAME fingerprint the result is (`fp`), not a second
mechanism of its own. Resuming onto a kept set that moved would draw boxes from
photos that are no longer in the run — strictly worse than starting over, which
is exactly the reasoning that put a fingerprint on the stored result in the first
place. A mismatch therefore drops the partial rather than repairing it.

Why it is written down, and how much of it
------------------------------------------
This used to be in-memory ONLY, on the grounds that "the worst case is one
recomputation". On a 150-image set that worst case is minutes of waiting for work
the user had already paid for once — and it hit the Stop/Resume bargain hardest:
the bank that makes a resume cheap died with the process, so the resume it was
banked FOR could never happen and Stop degraded into a discard.

So the two things that cost detector time — the published `result` and the
`partial` bank — are mirrored into a hidden sidecar beside the images they
describe (the same place the dataset's `.bank-analysis-cache` already lives), and
read back on first touch after a restart.

The `job` is deliberately NOT written down. Its thread died with the process, so
a restored "analyzing image 4 of 153" would be a progress bar nothing is driving
— and worse, the panel would refuse to start the pass that actually needs to run,
because `start()` joins a live job instead of duplicating it. A ghost is not a
lesser version of a running pass; it is the one state from which there is no way
forward.

Persisting changes WHERE this lives, not what it is allowed to claim: everything
read back is still gated by the same `fp` fingerprint, so a kept set that moved
while the server was down yields a labelled-stale result and an empty bank,
exactly as it does within one process.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import threading
import time

from .dataset_storage import dataset_path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_state: dict = {}          # dataset_id -> {'job': job|None, 'result': result|None}

# Hidden, and inside the dataset folder: deleting the dataset takes it with it,
# and no image lister will ever mistake it for a photo.
_STORE_NAME = '.face-mask-preview.json'

# A running job untouched for this long has lost its thread (the subprocess has
# its own, much shorter, watchdog) — treat it as dead so the UI can start again
# instead of waiting on a ghost forever.
_STALE_TTL = 60 * 60


def fingerprint(entries) -> str:
    """Fingerprint of the kept set. ``entries`` = iterable of
    (image_id, filename, size, mtime_ns). Sorted before hashing so row order
    never changes the answer."""
    parts = sorted(f'{i}|{n}|{s}|{m}' for (i, n, s, m) in entries)
    return hashlib.sha1('\n'.join(parts).encode('utf-8')).hexdigest()[:16]


def _store_path(dataset_id) -> str:
    """Where this dataset's preview is written. Resolves only — creating the
    directory is the writer's business, so a poll on a dataset that has none
    never brings one into existence."""
    return os.path.join(dataset_path(int(dataset_id)), _STORE_NAME)


def _read_store(dataset_id) -> dict:
    """The sidecar as {'result', 'partial'}, or empty.

    Never raises. A missing file is the normal case (nothing was ever computed);
    a corrupt one costs exactly one recomputation, which is strictly better than
    a training panel that will not open."""
    try:
        with open(_store_path(dataset_id), encoding='utf-8') as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.debug('unreadable face-mask preview store for dataset %s',
                     dataset_id, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(dataset_id, slot):
    """Mirror the durable half of `slot` to disk. Caller holds `_lock`.

    Written through a temp file and renamed: a process killed mid-write would
    otherwise leave a truncated file, and the next start would read half a
    preview rather than none. Never raises — a read-only or full disk must cost
    the persistence, not the pass that just succeeded."""
    payload = {'result': slot.get('result'), 'partial': slot.get('partial')}
    path = _store_path(dataset_id)
    tmp = f'{path}.tmp'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if payload['result'] is None and payload['partial'] is None:
            # Nothing durable left to say. Leaving a `{"result": null}` behind
            # would be a file that means the same as its absence.
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            return
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        logger.debug('could not persist the face-mask preview for dataset %s',
                     dataset_id, exc_info=True)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _slot(dataset_id) -> dict:
    """This dataset's state, hydrated from disk on first touch in this process.

    Hydration happens HERE rather than at each call site so no reader can forget
    it, and the PRESENCE of the slot is what marks it done — no second flag to
    keep in sync, and a poll every second does not re-read the file forever.
    That also makes `reset()` an honest restart: dropping the slot is exactly
    what makes the next touch read the disk again.

    Only the durable half comes back; `job` starts None, which is what a new
    process genuinely has. A stored half of the wrong shape (hand-edited file)
    is discarded rather than handed on, so no caller has to defend itself."""
    key = int(dataset_id)
    slot = _state.get(key)
    if slot is not None:
        return slot
    stored = _read_store(key)
    result, partial = stored.get('result'), stored.get('partial')
    slot = {'job': None,
            'result': result if isinstance(result, dict) else None,
            'partial': partial if isinstance(partial, dict) else None}
    _state[key] = slot
    return slot


def _live(job) -> bool:
    return bool(job) and not job['finished'] and time.time() - job['_touched'] < _STALE_TTL


def public(job) -> dict | None:
    """The job as the UI sees it — no private bookkeeping.

    `stopping` and `stopped` are separate states on purpose: between the click
    and the child actually winding up there is a real delay (it only looks at
    the sentinel between two images), and a button that snapped straight to
    "Stopped" while the pass was still running would be lying about the one
    thing the user is watching."""
    if not job:
        return None
    out = {k: job[k] for k in ('phase', 'done', 'total', 'error', 'finished')}
    out['stopping'] = bool(job.get('_stop')) and not job['finished']
    out['stopped'] = bool(job.get('stopped'))
    out['interrupted'] = bool(job.get('interrupted'))
    out['note'] = job.get('note')
    return out


def request_stop(dataset_id) -> bool:
    """Ask the live pass to wind up. True when there was one to ask.

    Only ever sets a flag: the worker polls it, hands the child the sentinel and
    keeps whatever came back. Nothing is killed from here — the results only
    exist inside the child until it prints them."""
    with _lock:
        job = _slot(dataset_id)['job']
        if not _live(job):
            return False
        job['_stop'] = True
        job['_touched'] = time.time()
        return True


def stop_requested(job) -> bool:
    """Polled from the worker thread — must stay cheap and lock-light."""
    return bool(job) and bool(job.get('_stop'))


def mark_stopped(job):
    """Record that this pass ended because it was ASKED to, not because it
    failed. Without it a stopped pass is indistinguishable from a finished one
    and the panel would claim a complete preview over a partial run."""
    with _lock:
        job['stopped'] = True
        job['_touched'] = time.time()


def mark_interrupted(job, message):
    """Record that this pass was cut short by something NOBODY asked for — the
    watchdog budget elapsing, or the child dying halfway.

    Its own state because both neighbours mislead here. `fail` reads as "this
    cannot work" for what is really "this needed more time", and it used to
    discard the pass with it; `stopped` would credit the user with a click they
    never made. What the panel has to say is the third thing: it was cut short,
    here is why, and here is what was kept."""
    with _lock:
        job['interrupted'] = True
        job['note'] = str(message)
        job['_touched'] = time.time()


def remember_partial(dataset_id, results, fp):
    """Bank the faces found by a pass that was stopped, under the fingerprint of
    the set they were computed on."""
    with _lock:
        slot = _slot(dataset_id)
        slot['partial'] = {'fp': fp, 'results': dict(results or {})}
        _write_store(dataset_id, slot)


def partial(dataset_id, fp) -> dict:
    """The banked detections for THIS exact kept set, or {}.

    A fingerprint mismatch drops the bank instead of trying to salvage the rows
    that might still match: the fingerprint covers the whole set at once, so a
    mismatch says the set moved and nothing in it can be trusted to still
    describe what would be trained."""
    with _lock:
        slot = _slot(dataset_id)
        banked = slot.get('partial')
        if not banked:
            return {}
        if fp is None or banked.get('fp') != fp:
            # The drop is persisted, not just forgotten: a bank the fingerprint
            # has already disqualified must not be resurrected by the next
            # restart and offered again as a resume.
            slot['partial'] = None
            _write_store(dataset_id, slot)
            return {}
        return dict(banked.get('results') or {})


def clear_partial(dataset_id):
    """Drop the bank — a pass that ran to completion has superseded it."""
    with _lock:
        slot = _slot(dataset_id)
        slot['partial'] = None
        _write_store(dataset_id, slot)


def get(dataset_id):
    """The live job for this dataset, or None."""
    with _lock:
        job = _slot(dataset_id)['job']
        return job if _live(job) else None


def snapshot(dataset_id, current_fp=None, total=None) -> dict:
    """What the panel needs on mount and on every poll: the running job (if any),
    the last result flagged `stale` when the kept set moved under it, and what a
    resume would be worth.

    `resume` is published rather than left implicit because the button has to
    NAME it: "Resume — 47 of 153 already analyzed" is a different offer from
    "Preview the mask", and a user who cannot see the credit has no reason to
    believe stopping was safe."""
    with _lock:
        slot = _slot(dataset_id)
        job, result = slot['job'], slot['result']
        out = {'job': public(job) if _live(job) or (job and job['finished']) else None}
        if result:
            out['result'] = {k: v for k, v in result.items() if k != 'fingerprint'}
            out['result']['stale'] = bool(
                current_fp is not None and result.get('fingerprint') != current_fp)
        else:
            out['result'] = None
        banked = slot.get('partial')
        # Same gate as the result's `stale`, one step stricter: a stale RESULT is
        # still shown (labelled), a stale BANK is worth nothing at all, because
        # resuming from it would silently mix boxes from images that left the set.
        if banked and current_fp is not None and banked.get('fp') == current_fp:
            out['resume'] = {'done': len(banked.get('results') or {}),
                             'total': int(total or 0)}
        else:
            out['resume'] = None
        return out


def start(app, dataset_id, fn, total=0, fp=None):
    """Start ``fn(job)`` for this dataset, or JOIN the one already running.

    Returns ``(job, started)``. ``started`` False means a pass was already in
    flight and this call did nothing — the second click, or the return to the
    page, costs zero InsightFace passes.
    """
    now = time.time()
    with _lock:
        slot = _slot(dataset_id)
        if _live(slot['job']):
            return slot['job'], False
        job = {'phase': 'starting', 'done': 0, 'total': int(total or 0),
               'error': None, 'finished': False, 'started_at': now,
               '_touched': now, '_fp': fp}
        slot['job'] = job

    def _run():
        try:
            with app.app_context():
                fn(job)
        except Exception as e:  # noqa: BLE001 — a background crash must surface, not vanish
            with _lock:
                job['error'] = f'{type(e).__name__}: {e}'
        finally:
            with _lock:
                job['finished'] = True
                job['_touched'] = time.time()

    # Under TESTING the job runs INLINE, exactly like bank_jobs: the suite uses a
    # per-connection sqlite:///:memory: DB, so a real worker thread would open a
    # fresh, EMPTY database.
    if app.config.get('TESTING'):
        _run()
    else:
        threading.Thread(target=_run, daemon=True,
                         name=f'facemask-preview-{dataset_id}').start()
    return job, True


def progress(job, record):
    """Apply one {'phase', 'done'?, 'total'?} record from the stderr reader."""
    if not job or not record:
        return
    with _lock:
        for key in ('phase', 'done', 'total'):
            if record.get(key) is not None:
                job[key] = record[key]
        job['_touched'] = time.time()


def set_result(dataset_id, result, fp=None):
    """Publish the finished preview. Kept until superseded — a preview the user
    walked away from is exactly the one they come back to look at."""
    with _lock:
        payload = dict(result or {})
        payload['fingerprint'] = fp
        payload['at'] = time.time()
        slot = _slot(dataset_id)
        slot['result'] = payload
        _write_store(dataset_id, slot)


def fail(job, message):
    with _lock:
        job['error'] = str(message)
        job['_touched'] = time.time()


def reset(dataset_id=None):
    """Test hook — drop one dataset's MODULE state, or all of it.

    Deliberately does not touch the sidecar: this is what a restart looks like
    from inside the process, and the tests use it as exactly that."""
    with _lock:
        if dataset_id is None:
            _state.clear()
        else:
            _state.pop(int(dataset_id), None)
