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

_lock = threading.Lock()
_jobs: dict = {}          # bank_id -> job dict (see start())
_FINISHED_TTL = 5 * 60    # finished snapshot lifetime
_STALE_TTL = 60 * 60      # running job with no progress for this long = dead


class BankJobBusy(Exception):
    """Another job is already running on this bank."""
    def __init__(self, kind):
        super().__init__(f'a {kind} job is already running on this bank')
        self.kind = kind


def start(app, bank_id, kind, fn, total=0):
    """Run ``fn(job)`` in a daemon thread under an app context. One live job
    per bank — raises BankJobBusy otherwise. ``fn`` reports through
    ``progress``/``bump`` and should poll ``cancelled(job)`` between items."""
    now = time.time()
    with _lock:
        cur = _jobs.get(bank_id)
        if cur and not cur['finished'] and now - cur['_touched'] < _STALE_TTL:
            raise BankJobBusy(cur['kind'])
        job = {'kind': kind, 'done': 0, 'total': int(total or 0), 'error': None,
               'cancelled': False, 'finished': False, 'detail': None,
               'started_at': now, '_touched': now, '_cancel_hook': None}
        _jobs[bank_id] = job

    def _run():
        try:
            with app.app_context():
                fn(job)
        except Exception as e:  # noqa: BLE001 — a background crash must surface in the UI
            with _lock:
                job['error'] = f'{type(e).__name__}: {e}'
        finally:
            with _lock:
                job['finished'] = True
                job['_touched'] = time.time()

    # Under TESTING the job runs INLINE: the test suite uses a per-connection
    # sqlite:///:memory: DB, so a real worker thread would open a fresh, EMPTY
    # database — and assertions would race the thread anyway.
    if app.config.get('TESTING'):
        _run()
    else:
        threading.Thread(target=_run, daemon=True,
                         name=f'bank-{bank_id}-{kind}').start()
    return job


def progress(job, done=None, total=None, detail=None):
    with _lock:
        if done is not None:
            job['done'] = int(done)
        if total is not None:
            job['total'] = int(total)
        if detail is not None:
            job['detail'] = str(detail)
        job['_touched'] = time.time()


def bump(job, n=1):
    with _lock:
        job['done'] += n
        job['_touched'] = time.time()


def cancelled(job) -> bool:
    with _lock:
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
    finished, detail, started_at} or None. Purges expired entries."""
    now = time.time()
    with _lock:
        job = _jobs.get(bank_id)
        if not job:
            return None
        ttl = _FINISHED_TTL if job['finished'] else _STALE_TTL
        if now - job['_touched'] > ttl:
            _jobs.pop(bank_id, None)
            return None
        return {k: job[k] for k in ('kind', 'done', 'total', 'error',
                                    'cancelled', 'finished', 'detail',
                                    'started_at')}


def running(bank_id) -> bool:
    snap = get(bank_id)
    return bool(snap and not snap['finished'])


def reset():
    """Test helper: forget every job."""
    with _lock:
        _jobs.clear()
