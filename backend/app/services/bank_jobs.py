"""In-memory background-job runner for the 🗃️ image bank.

Unlike dataset batches (dataset_activity), a bank pass runs over THOUSANDS of
files — holding the HTTP request open for its whole duration is not an option.
Each bank therefore gets at most ONE live background thread (scan / faces /
promote); the POST that starts it returns immediately and the UI polls the
bank payload, which embeds the job snapshot from here.

Same design contract as dataset_activity:
* **In-memory ONLY** — a job dies with the process; on restart the registry is
  empty and nothing phantom survives (raw scores already committed stay, so a
  re-run only pays for what's missing).
* **Thread-safe** — one module lock guards the store.
* **TTL-purged** — a finished snapshot is kept ~5 min so the UI can show the
  outcome, then purged on read; a running entry not touched for an hour is
  presumed dead and purged too (its thread would have to be truly stuck).
"""
import threading
import time
from contextlib import contextmanager

from . import job_eta

_lock = threading.Lock()
_jobs: dict = {}          # bank_id -> job dict (see start())
_FINISHED_TTL = 5 * 60    # finished snapshot lifetime
_STALE_TTL = 60 * 60      # running job with no progress for this long = dead

# Called after a pass function returns CLEANLY (no exception, no job error, not
# cancelled) with (bank_id, kind, detail), inside a fresh app context. Installed
# by image_bank_service at import to journal the run (note_pass_run); kept as a
# hook because this runner is lane-agnostic (the video lane shares it) and must
# not import a lane. Until it existed only ✂ semantic_dedup wrote its journal
# row, so a Launch-all report's "cancelled before it ran" stayed red forever
# over 👥/🚩 re-runs that had long since done the work.
on_complete = None


class BankJobBusy(Exception):
    """Another job is already running on this bank."""
    usage_error_code = 'busy'
    usage_error_reason = 'bank_busy'

    def __init__(self, kind):
        super().__init__(f'a {kind} job is already running on this bank')
        self.kind = kind


def _key(bank_id):
    """The registry slot for a bank id — TWO LANES KEY THIS DIFFERENTLY.

    Numeric ids are normalised to int, and that coercion is load-bearing: the
    image lane crosses a JSON boundary where the same Bank arrives as ``7`` or
    ``'7'``, and those have to be one slot or a second pass slips past the
    serialization.

    Anything NOT numeric passes through unchanged. The video lane deliberately
    keys on ``'video:<id>'`` (see ``video_bank_service.job_key``) so that video
    bank 1 and image bank 1 cannot occupy the same slot — their ids overlap by
    construction, and a collision would refuse a video pass in the name of an
    image pass the user cannot see. A bare ``int()`` here raised ValueError on
    that key, which the route layer turned into a 400: every pass in the video
    lane, from the probe onward, answered "bad request" the moment reservations
    landed. Two lanes, one registry, and nothing was holding the assumption they
    share — test_bank_jobs_key_namespaces.py now does.
    """
    try:
        return int(bank_id)
    except (TypeError, ValueError):
        return bank_id


def _reservation_keys(bank_id, reserve_ids=None):
    # `reserve_ids` stay strictly numeric: they are IMAGE ids reserved alongside
    # their bank, a different thing from the bank key itself.
    return tuple(dict.fromkeys(
        [_key(bank_id), *(int(value) for value in (reserve_ids or ())) ]))


def _drop_job_locked(job):
    """Remove every alias that still points at ``job`` (caller holds ``_lock``).

    Some old tests and, during a live upgrade, old in-memory entries do not have
    ``_keys``.  Identity-scanning the small registry is the safe fallback and
    also prevents deleting a newer job that has already reused one of the ids.
    """
    keys = tuple(job.get('_keys') or ())
    if not keys:
        keys = tuple(key for key, value in _jobs.items() if value is job)
    for key in keys:
        if _jobs.get(key) is job:
            _jobs.pop(key, None)


def _reserve_locked(bank_id, kind, total=0, reserve_ids=None):
    now = time.time()
    keys = _reservation_keys(bank_id, reserve_ids)
    for key in keys:
        cur = _jobs.get(key)
        if not cur:
            continue
        ttl = _FINISHED_TTL if cur.get('finished') else _STALE_TTL
        if now - cur.get('_touched', 0) > ttl:
            _drop_job_locked(cur)
            continue
        if not cur.get('finished'):
            raise BankJobBusy(cur.get('kind') or 'background')
        # A completed snapshot is useful only until a new generation reuses one
        # of its ids.  Drop all of its aliases before installing the replacement.
        _drop_job_locked(cur)
    job = {'kind': kind, 'done': 0, 'total': int(total or 0), 'error': None,
           'cancelled': False, 'finished': False, 'detail': None,
           'stop_cost': None, 'stop_wait': None,
           'started_at': now, '_touched': now, '_cancel_hook': None,
           'pipeline': None, '_keys': keys, '_launched': False,
           '_owner_thread': threading.get_ident(),
           '_eta': job_eta.new_state()}
    # One shared object under every participating Bank id is an atomic
    # multi-bank reservation.  No lock ordering/deadlock exists because the
    # registry lock is acquired once for the whole set.
    for key in keys:
        _jobs[key] = job
    return job


def reserve(bank_id, kind, total=0, reserve_ids=None):
    """Atomically reserve one or more Bank ids without starting a worker.

    Destination-building flows need this narrow two-phase API: flush a new Bank
    id (still invisible outside the transaction), reserve source + destination,
    commit the row, then call :func:`start`.  Ordinary callers keep using
    ``start`` directly.
    """
    with _lock:
        return _reserve_locked(bank_id, kind, total, reserve_ids)


def require_reservation(reservation, bank_id):
    """Validate an existing Bank reservation capability and keep it alive.

    Background passes sometimes call the same small mutation helpers as HTTP
    requests (the pipeline's auto-reject is the canonical example).  Those
    helpers must not bypass serialization merely because their caller already
    owns the Bank, nor may they try to reserve it again and deadlock/refuse
    themselves.  Exact object identity is the capability: a copied mapping, a
    finished/purged job, or a token for another Bank fails closed.
    """
    key = _key(bank_id)
    with _lock:
        valid = bool(
            isinstance(reservation, dict)
            and not reservation.get('finished')
            and key in tuple(reservation.get('_keys') or ())
            and _jobs.get(key) is reservation
        )
        if not valid:
            raise RuntimeError(
                'bank mutation reservation is missing, stale, or belongs to '
                'another bank')
        reservation['_touched'] = time.time()
        return reservation


@contextmanager
def mutation_lease(bank_id, kind, *, capability=None, total=0,
                   preserve_finished=True):
    """Serialize one synchronous Bank mutation with jobs and promotions.

    With no ``capability`` this installs a short-lived reservation atomically;
    another synchronous write or a background pass therefore wins or receives
    :class:`BankJobBusy`, never slips through a check-then-write window.  A job
    that already owns the Bank passes its exact reservation capability instead,
    which validates ownership without reacquiring the non-reentrant slot.
    """
    owned = capability is None
    previous = None
    key = _key(bank_id)
    if owned:
        with _lock:
            if preserve_finished:
                cur = _jobs.get(key)
                if cur and cur.get('finished'):
                    if time.time() - cur.get('_touched', 0) > _FINISHED_TTL:
                        _drop_job_locked(cur)
                    else:
                        # Displace only THIS alias. A completed multi-Bank job
                        # may still be the useful UI snapshot of its other Bank;
                        # dropping every alias here would erase that history.
                        previous = cur
                        if _jobs.get(key) is cur:
                            _jobs.pop(key, None)
            lease = _reserve_locked(key, kind, total=total)
    else:
        lease = require_reservation(capability, key)
    try:
        yield lease
    finally:
        if owned:
            with _lock:
                # Identity checks protect a newer owner if this synchronous
                # request somehow outlived its stale TTL.
                if _jobs.get(key) is lease:
                    _jobs.pop(key, None)
                if (previous is not None and key not in _jobs
                        and time.time() - previous.get('_touched', 0)
                        <= _FINISHED_TTL):
                    _jobs[key] = previous


def abort(job):
    """Release an unstarted/failed reservation, including every alias."""
    if not job:
        return
    with _lock:
        _drop_job_locked(job)


def launched(job) -> bool:
    """Whether ``start`` adopted and attempted to launch this reservation."""
    with _lock:
        return bool(job and job.get('_launched'))


def _adopt_reservation_locked(reservation, keys, kind):
    """Validate and return an explicitly supplied reservation (lock held)."""
    valid = bool(
        isinstance(reservation, dict)
        and not reservation.get('finished')
        and not reservation.get('_launched')
        and tuple(reservation.get('_keys') or ()) == keys
        and reservation.get('kind') == kind
        and all(_jobs.get(key) is reservation for key in keys)
    )
    if not valid:
        raise RuntimeError(
            'bank job reservation is missing, stale, or belongs to another job')
    return reservation


def start(app, bank_id, kind, fn, total=0, reserve_ids=None,
          reservation=None):
    """Run ``fn(job)`` in a daemon thread under an app context. One live job
    per bank — raises BankJobBusy otherwise. ``fn`` reports through
    ``progress``/``bump`` and should poll ``cancelled(job)`` between items.

    ``reservation`` is the object returned by :func:`reserve`; when supplied it
    is adopted only by exact identity under every reserved Bank id.  The legacy
    same-thread adoption remains temporarily available for callers that have not
    yet been migrated to pass the explicit capability.
    """
    keys = _reservation_keys(bank_id, reserve_ids)
    with _lock:
        cur = _jobs.get(_key(bank_id))
        if cur:
            ttl = _FINISHED_TTL if cur.get('finished') else _STALE_TTL
            if time.time() - cur.get('_touched', 0) > ttl:
                _drop_job_locked(cur)
                cur = None
        if reservation is not None:
            job = _adopt_reservation_locked(reservation, keys, kind)
        else:
            # Backward-compatible bridge for the two destination-building
            # service call sites. New callers must pass ``reservation``.
            adopt = bool(
                cur and not cur.get('finished') and not cur.get('_launched')
                and cur.get('_owner_thread') == threading.get_ident()
                and tuple(cur.get('_keys') or ()) == keys
                and cur.get('kind') == kind
            )
            job = cur if adopt else _reserve_locked(
                bank_id, kind, total=total, reserve_ids=reserve_ids)
        job['_launched'] = True
        job['_touched'] = time.time()

    # Only this explicit image-bank vocabulary is reported. The video lane and
    # third-party job names are not copied into product statistics.
    usage_ticket = None
    usage_context = None
    usage_action = {'scan': 'scan', 'caption': 'caption', 'faces': 'faces',
                    'score': 'score', 'framing': 'framing',
                    'promote': 'promote'}.get(kind)
    if usage_action and isinstance(_key(bank_id), int):
        from ..usage_statistics import begin_operation, operation_context, caption_context
        with app.app_context():
            usage_ticket = begin_operation('bank')
            usage_context = (caption_context() if usage_action == 'caption' else
                             operation_context(execution='local'))

    def _run():
        completed = False
        usage_error = ('operation_failed', 'unknown')
        try:
            with app.app_context():
                fn(job)
                completed = True
        except Exception as e:  # noqa: BLE001 — a background crash must surface in the UI
            from ..usage_statistics import error_categories
            usage_error = error_categories(e)
            with _lock:
                job['error'] = f'{type(e).__name__}: {e}'
        finally:
            with _lock:
                job['finished'] = True
                job['_touched'] = time.time()
            if usage_ticket is not None:
                from ..usage_statistics import finish_operation
                result = ('cancelled' if job.get('cancelled') else
                          'failed' if job.get('error') or not completed else 'completed')
                if result == 'completed' and job.get('_usage_result') in {'partial', 'failed'}:
                    result = job['_usage_result']
                finish_operation(usage_ticket, 'bank', usage_action, result,
                                 *(usage_error if result in {'partial', 'failed'} else ('none', 'none')),
                                 context=usage_context)
            # AFTER the snapshot is finished, never before: the journal write
            # commits, and a commit inside the pass's own context would race its
            # final progress read. A cancelled or errored run is deliberately
            # not journaled — the step's story has NOT moved on, and marking a
            # report row "re-run since" over a stop at 3% would be a lie.
            hook = on_complete
            if (hook is not None and completed
                    and not job.get('error') and not job.get('cancelled')):
                try:
                    with app.app_context():
                        hook(bank_id, kind, job.get('detail'))
                except Exception:  # noqa: BLE001 — bookkeeping never sinks a pass
                    pass

    # Under TESTING the job runs INLINE: the test suite uses a per-connection
    # sqlite:///:memory: DB, so a real worker thread would open a fresh, EMPTY
    # database — and assertions would race the thread anyway.
    if app.config.get('TESTING'):
        _run()
    else:
        try:
            threading.Thread(target=_run, daemon=True,
                             name=f'bank-{bank_id}-{kind}').start()
        except Exception:
            # Creating a destination and then failing to create its worker used
            # to leave every reserved id permanently busy until the one-hour TTL.
            abort(job)
            raise
    return job


def _observe_locked(job, now):
    """Feed the remaining-time estimator (caller holds ``_lock``).

    A few historical call sites hand a plain job mapping straight to a pass
    without going through ``reserve``/``start``; those have no estimator, and
    creating one here keeps them counting rather than crashing on a KeyError.
    """
    state = job.get('_eta')
    if state is None:
        state = job['_eta'] = job_eta.new_state()
    job_eta.observe(state, job.get('done') or 0, job.get('total') or 0, now)


def progress(job, done=None, total=None, detail=None):
    with _lock:
        if done is not None:
            job['done'] = int(done)
        if total is not None:
            job['total'] = int(total)
        if detail is not None:
            job['detail'] = str(detail)
        now = time.time()
        job['_touched'] = now
        _observe_locked(job, now)


def fail(job, message):
    """Stop a pass on a condition that is NOT an image problem — the source
    folder went away, a drive is unplugged. The runner's generic handler prefixes
    the exception type (`RuntimeError: ...`), which is noise in a toast, so a job
    that wants to explain itself in plain words sets the error here and returns."""
    with _lock:
        job['error'] = str(message)
        job['detail'] = str(message)
        job['_touched'] = time.time()


def bump(job, n=1):
    with _lock:
        job['done'] += n
        now = time.time()
        job['_touched'] = now
        _observe_locked(job, now)


def set_pipeline(job, snapshot):
    """Attach/replace the multi-step pipeline snapshot (step index/label and the
    per-step outcomes) carried alongside the plain done/total bar. Only the
    'pipeline' kind uses this; a copy is stored so later mutation is deliberate."""
    with _lock:
        job['pipeline'] = dict(snapshot) if snapshot is not None else None
        job['_touched'] = time.time()


def set_stop_notice(job, cost=None, wait=None):
    """Publish what pressing Stop RIGHT NOW costs, and what it then waits for.

    The Bank's Stop button used to be a bare label with no state: the click was
    honoured in 79 ms (this registry is in memory, ``cancel`` touches no
    database) but the only feedback lived in the bank payload, which takes
    seconds to rebuild while a pass is writing tens of thousands of rows. So the
    button looked untouched and people pressed it again — seven POSTs inside
    20 ms in one measured session.

    The front end can make the button answer the click on its own, but it cannot
    invent the two sentences that matter, because they are NOT properties of the
    pass — they are properties of the PHASE it is in. ✨ Score alone stops three
    different ways: during inference the computed scores are salvaged, during
    the write-back the stop lands at the end of the current commit batch, and
    during the style write it does not land at all because that step is written
    whole or not at all. Only the worker knows which of those is true this
    second, so it says so here and the bar relays it verbatim.

    ``cost`` is read BEFORE the click (what you keep, what you lose); ``wait``
    replaces it AFTER it (what the pass is finishing). Both are cleared by
    passing None — a phase that has nothing specific to promise stays SILENT
    rather than offering a generic reassurance it cannot back up.
    """
    with _lock:
        job['stop_cost'] = str(cost) if cost else None
        job['stop_wait'] = str(wait) if wait else None
        job['_touched'] = time.time()


def cancelled(job) -> bool:
    with _lock:
        # A few synchronous service-level callers (and older extensions) invoke
        # a pass with the historical plain job mapping instead of going through
        # ``start``.  Such a mapping has no reservation capability to lose, so
        # its explicit flag remains authoritative.  Every production job made
        # by ``reserve``/``start`` carries ``_keys`` and still fails closed when
        # its registry ownership is purged or replaced.
        if '_keys' not in job:
            if any(value is job for value in _jobs.values()):
                job['_touched'] = time.time()
            return bool(job.get('cancelled'))
        # A worker whose capability was purged/replaced must stop before it can
        # publish beside the newer owner. Active workers also heartbeat here;
        # long admission loops call this between items even before done advances.
        keys = tuple(job.get('_keys') or ())
        if not all(_jobs.get(key) is job for key in keys):
            return True
        job['_touched'] = time.time()
        return job['cancelled']


def set_cancel_hook(job, hook):
    """Register a callable invoked by cancel() — e.g. kill a subprocess so a
    cancel interrupts the current item, not just the loop between items."""
    with _lock:
        job['_cancel_hook'] = hook


def cancel(bank_id) -> bool:
    """Flag the bank's live job as cancelled (and fire its hook). False when
    there is nothing to cancel."""
    with _lock:
        job = _jobs.get(bank_id)
        if not job or job['finished']:
            return False
        job['cancelled'] = True
        job['_touched'] = time.time()
        hook = job['_cancel_hook']
    if hook:
        try:
            hook()
        except Exception:  # noqa: BLE001 — best effort; the loop flag still stands
            pass
    return True


def get(bank_id):
    """Snapshot for the payload: {kind, done, total, error, cancelled,
    finished, detail, started_at, pipeline, eta_state, eta_seconds, eta_scope,
    stop_cost, stop_wait} or None. Purges expired entries.

    ``stop_*`` is what the Stop button promises in the phase running right now —
    see :func:`set_stop_notice`.

    ``eta_*`` is how long the pass still has to run — see ``job_eta``. A
    FINISHED job never carries one: "about 20 minutes left" under a completed
    pass would be the loudest possible way to say the number means nothing."""
    now = time.time()
    with _lock:
        job = _jobs.get(bank_id)
        if not job:
            return None
        ttl = _FINISHED_TTL if job['finished'] else _STALE_TTL
        if now - job['_touched'] > ttl:
            _drop_job_locked(job)
            return None
        snap = {k: job[k] for k in ('kind', 'done', 'total', 'error',
                                    'cancelled', 'finished', 'detail',
                                    'started_at')}
        pipeline = job.get('pipeline')
        snap['pipeline'] = dict(pipeline) if pipeline else None
        # `.get` on purpose: a job mapping built before this field existed (an
        # in-memory entry surviving a live upgrade, or one of the historical
        # hand-built mappings some callers still pass) must not KeyError the
        # whole payload. A finished pass has no Stop button to describe, so it
        # carries no promise either.
        running = not (job['finished'] or job['error'])
        snap['stop_cost'] = job.get('stop_cost') if running else None
        snap['stop_wait'] = job.get('stop_wait') if running else None
        state = job.get('_eta')
        if state is None or job['finished'] or job['cancelled'] or job['error']:
            snap['eta_state'] = job_eta.ETA_NONE
            snap['eta_seconds'] = None
            snap['eta_scope'] = 'job'
        else:
            eta_state, seconds, scope = job_eta.read(state, now)
            snap['eta_state'] = eta_state
            snap['eta_seconds'] = seconds
            snap['eta_scope'] = scope
        return snap


def running(bank_id) -> bool:
    snap = get(bank_id)
    return bool(snap and not snap['finished'])


def reset():
    """Test helper: forget every job."""
    with _lock:
        _jobs.clear()
