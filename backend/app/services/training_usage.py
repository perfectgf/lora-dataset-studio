"""Process-local, opt-in training receipts. No run identity is sent remotely.

The caller serializes completion with Stop using the training queue lock.
Restarted servers do not invent outcomes for watchers that no longer exist.
"""
import threading

from ..usage_statistics import finish_operation

_lock = threading.Lock()
_runs = {}


def register(run_token, ticket):
    if ticket is not None:
        with _lock:
            _runs[run_token] = [ticket, False]


def cancelled(run_token):
    with _lock:
        if run_token in _runs:
            _runs[run_token][1] = True


def discard(run_token):
    with _lock:
        _runs.pop(run_token, None)


def finish(run_token, returncode, *, resume_failed=False):
    with _lock:
        receipt = _runs.pop(run_token, None)
    if receipt is None:
        return
    ticket, was_cancelled = receipt
    if was_cancelled:
        result, code, reason = 'cancelled', 'none', 'none'
    elif resume_failed or returncode not in (0, None):
        result, code = 'failed', 'operation_failed'
        reason = 'training_resume_failed' if resume_failed else 'training_process_failed'
    elif returncode == 0:
        result, code, reason = 'completed', 'none', 'none'
    else:
        return  # An unknown exit code is not proof of completion.
    finish_operation(ticket, 'training', 'train_run', result, code, reason)
