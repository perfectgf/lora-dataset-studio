"""In-memory per-dataset reference-EDIT job registry + candidate file lifecycle.

Editing the reference photo is a SLOW (1-3 min) and PAID API call. Running it
inside the client's fetch made it die when a mobile tab was backgrounded ("Failed
to fetch") — and the paid result was lost. So the edit now runs in a server
background thread, exactly like the variation batch: the request returns at once,
the worker fills a CANDIDATE, and the client rediscovers it through the dataset
payload — which survives a tab sleep AND a reload.

This registry holds ONE pending edit per dataset:
  {status: running|ready|failed, engine, prompt, candidate_filename, error,
   token, started_at, _touched, _dir}
It is IN-MEMORY only (like dataset_activity) — a job dies with the process, which
is correct for a transient candidate awaiting Keep. The candidate FILE on disk is
reclaimed by a TTL sweep so a crashed process never leaks it.

Leak-proofing (what replaces the old client-held "browser GC"):
  * Keep      -> promote + delete the candidate file, clear the entry.
  * Discard   -> delete the candidate file, clear the entry.
  * abandoned -> TTL purge: lazy on get() (deletes the file) + a disk sweep on the
                 next start() (catches a file orphaned by a process crash).
A compare-and-set on ``token`` means a SUPERSEDED worker (a second edit started
before the first returned) never overwrites the newer job — set_ready/set_failed
return False and the stale worker deletes its own candidate instead.

Disk ownership: this module owns the candidate files end to end (create is done by
the service worker, delete lives here), so the lifecycle stays in one place.
"""
import itertools
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Distinct filename marker: DB-driven enumeration and the backup selector never
# pick these up (they list ref fields + image rows, never scan the dir), so a
# candidate can sit in the dataset dir without polluting the grid or a backup.
CANDIDATE_MARKER = '_datasetrefeditcand_'
_TTL_SECONDS = 30 * 60

_lock = threading.Lock()
_jobs: dict = {}                 # dataset_id -> entry dict
_counter = itertools.count(1)


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _public(e):
    """The client-facing shape embedded in the dataset payload."""
    return {'status': e['status'], 'engine': e.get('engine'),
            'prompt': e.get('prompt'), 'candidate_filename': e.get('candidate_filename'),
            'error': e.get('error'), 'started_at': e['started_at']}


def start(dataset_id, dsdir, engine, prompt):
    """Register a new RUNNING edit, SUPERSEDING any previous one for this dataset
    (its candidate file is deleted). Returns a unique token the worker echoes back
    on set_ready/set_failed — a stale worker whose token no longer matches is
    ignored. Also runs the on-start disk sweep for crash-orphaned candidates."""
    now = time.time()
    with _lock:
        prev = _jobs.get(dataset_id)
        token = next(_counter)
        _jobs[dataset_id] = {'status': 'running', 'engine': str(engine),
                             'prompt': str(prompt), 'candidate_filename': None,
                             'error': None, 'token': token,
                             'started_at': now, '_touched': now, '_dir': dsdir}
    # Disk I/O outside the lock.
    if prev and prev.get('candidate_filename') and prev.get('_dir'):
        _unlink(os.path.join(prev['_dir'], prev['candidate_filename']))
    sweep(dsdir)
    return token


def set_ready(dataset_id, token, candidate_filename):
    """Mark the job ready with its candidate file. Returns False when the job was
    superseded (token no longer current) — the caller then deletes its orphan."""
    with _lock:
        e = _jobs.get(dataset_id)
        if not e or e['token'] != token:
            return False
        e['status'] = 'ready'
        e['candidate_filename'] = candidate_filename
        e['_touched'] = time.time()
        return True


def set_failed(dataset_id, token, error):
    """Mark the job failed with a verbatim provider message. False if superseded."""
    with _lock:
        e = _jobs.get(dataset_id)
        if not e or e['token'] != token:
            return False
        e['status'] = 'failed'
        e['error'] = str(error)[:500]
        e['_touched'] = time.time()
        return True


def get(dataset_id):
    """The public entry for the payload, or None. LAZY TTL: an entry untouched for
    longer than the TTL is dropped here and its candidate file deleted — this is
    what makes an abandoned edit leak-proof without a cron (the payload calls get()
    on every poll)."""
    now = time.time()
    with _lock:
        e = _jobs.get(dataset_id)
        if not e:
            return None
        if now - e['_touched'] <= _TTL_SECONDS:
            return _public(e)
        _jobs.pop(dataset_id, None)
        expired = e
    if expired.get('candidate_filename') and expired.get('_dir'):
        _unlink(os.path.join(expired['_dir'], expired['candidate_filename']))
    return None


def peek(dataset_id):
    """Like get() but WITHOUT the TTL purge — used by keep/discard which must read
    the candidate even if it just aged past the TTL boundary (the user did claim
    it). None when there is no entry."""
    with _lock:
        e = _jobs.get(dataset_id)
        return _public(e) if e else None


def clear(dataset_id, dsdir=None):
    """Drop the entry and (when ``dsdir`` is given) delete its candidate file.
    Idempotent — used by Keep (after commit), Discard, and reference-mutation
    invalidation (crop/recrop/change)."""
    with _lock:
        e = _jobs.pop(dataset_id, None)
    if e and dsdir and e.get('candidate_filename'):
        _unlink(os.path.join(dsdir, e['candidate_filename']))


def sweep(dsdir):
    """Delete candidate files in ``dsdir`` older than the TTL. Catches a file left
    behind by a process that crashed mid-edit (no in-memory entry survives a
    restart, so only a disk scan can reclaim it). Never raises."""
    now = time.time()
    try:
        names = os.listdir(dsdir)
    except OSError:
        return
    for n in names:
        if CANDIDATE_MARKER in n:
            p = os.path.join(dsdir, n)
            try:
                if now - os.path.getmtime(p) > _TTL_SECONDS:
                    os.remove(p)
            except OSError:
                pass


def reset():
    """Test helper: clear the whole registry (does NOT touch files)."""
    with _lock:
        _jobs.clear()
