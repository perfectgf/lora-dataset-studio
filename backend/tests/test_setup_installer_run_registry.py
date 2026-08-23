"""A worker thread can outlive its _runs entry, and must die quietly when it does.

The CI carried a recurring `KeyError: 'seedvr2_model'` warning: a download
thread from an earlier test kept appending progress after the fixture reset
`_runs`, and the error handler's own `_append` then raised the same KeyError it
was reporting. A cleared entry means nobody is watching that run any more —
the log line is dropped and the final state stamp is skipped, instead of the
thread being killed from inside its own exception handler (which in prod would
also eat the error report of a live download whose registry was cleared).
"""
from app import setup_installer as si


def test_appending_to_a_cleared_run_is_a_quiet_no_op():
    si._runs['ghost_action'] = si._new_run()
    si._runs.pop('ghost_action')
    si._append('ghost_action', 'progress after the registry moved on')
    si._finish_run('ghost_action', -1, 'error')
    assert 'ghost_action' not in si._runs


def test_append_still_logs_and_caps_for_a_live_run():
    si._runs['live_action'] = si._new_run()
    try:
        for i in range(si._LOG_MAX + 7):
            si._append('live_action', f'line {i}\n')
        log = si._runs['live_action']['log']
        assert len(log) == si._LOG_MAX
        assert log[-1] == f'line {si._LOG_MAX + 6}'
    finally:
        si._runs.pop('live_action', None)


def test_execute_survives_a_registry_cleared_under_it(monkeypatch):
    """The whole worker path: the registry vanishes mid-download AND the worker
    raises — _execute's handler must report nowhere rather than crash."""
    def worker(action):
        si._runs.clear()
        raise RuntimeError('boom mid-download')

    monkeypatch.setitem(si._WORKERS, 'ghost_worker', worker)
    si._runs['ghost_worker'] = si._new_run()
    si._execute('ghost_worker')
    assert 'ghost_worker' not in si._runs
