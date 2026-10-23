"""Local training outcomes respect consent and the Stop/watcher ownership boundary."""
import time

from flask import Flask
import pytest

from app.usage_statistics import UsageStatistics
from app.services import training_usage


@pytest.fixture
def receipt(tmp_path):
    service = UsageStatistics(tmp_path, project_token='phc_' + 'test' * 12, start_worker=False)
    service.set_enabled(True)
    ticket = (service, service.consent_generation(), time.monotonic())
    training_usage.register('test-run', ticket)
    yield service
    training_usage.discard('test-run')


@pytest.mark.parametrize('rc,cancelled,resume_failed,expected,reason', [
    (0, False, False, 'completed', 'none'),
    (1, False, False, 'failed', 'training_process_failed'),
    (78, False, True, 'failed', 'training_resume_failed'),
    (1, True, False, 'cancelled', 'none'),
    (None, False, False, None, None),
])
def test_terminal_outcome_once_without_run_identity(receipt, rc, cancelled, resume_failed, expected, reason):
    if cancelled:
        training_usage.cancelled('test-run')
    training_usage.finish('test-run', rc, resume_failed=resume_failed)
    training_usage.finish('test-run', rc, resume_failed=resume_failed)
    rows = receipt._state['events']
    if expected is None:
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]['properties']['action'] == 'train_run'
        assert rows[0]['properties']['result'] == expected
        assert rows[0]['properties']['error_reason'] == reason
        assert 'test-run' not in receipt.path.read_text(encoding='utf-8')


def test_reconsent_does_not_attribute_an_old_training_to_a_new_identity(receipt):
    receipt.set_enabled(False)
    receipt.set_enabled(True)
    training_usage.finish('test-run', 0)
    assert receipt.status()['queued_events'] == 0


def test_unobserved_or_unrelated_runs_do_not_create_or_cancel_an_outcome(receipt):
    training_usage.register('unobserved', None)
    training_usage.finish('unobserved', 0)
    training_usage.cancelled('another-run')
    training_usage.finish('test-run', 0)
    assert receipt._state['events'][0]['properties']['result'] == 'completed'


@pytest.mark.parametrize('rc', [0, 1])
def test_actual_watcher_reports_exit_under_the_stop_lock(receipt, monkeypatch, rc):
    from app.services import lora_training as lt
    class Proc:
        returncode = rc
        def wait(self):
            pass
    app = Flask(__name__)
    calls = []
    original = training_usage.finish
    def finish(*args, **kwargs):
        assert lt._queue_lock._is_owned()
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(training_usage, 'finish', finish)
    monkeypatch.setattr(lt, '_crash_payload', lambda *a: {'excerpt': {'text': 'test failure'}, 'log_tail': ''})
    monkeypatch.setattr(lt.queue_manager, '_set_system_state', lambda *a, **k: None)
    monkeypatch.setattr(lt, 'process_training_queue', lambda: None)
    lt._watch_training(app, Proc(), 'unused.log', 1, usage_run_token='test-run')
    assert len(calls) == 1
    assert receipt._state['events'][0]['properties']['result'] == ('completed' if rc == 0 else 'failed')


@pytest.mark.parametrize('stopped', [True, False])
def test_actual_stop_marks_only_a_confirmed_termination(receipt, monkeypatch, stopped):
    from types import SimpleNamespace
    from app.services import lora_training as lt
    state = {'training_pid': 4242, 'training_in_progress': True, 'training_run_token': 'test-run'}
    monkeypatch.setattr(lt.queue_manager, '_get_system_state', lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(lt, '_pid_alive', lambda pid: True)
    monkeypatch.setattr(lt, '_wait_for_training_process_exit', lambda pid: stopped)
    monkeypatch.setattr(lt.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(lt.os, 'kill', lambda *a: None)
    monkeypatch.setattr(lt, '_save_queue', lambda items: None)
    monkeypatch.setattr(lt, '_clear_training_identity', lambda **kw: None)
    assert lt.stop_training() is stopped
    training_usage.finish('test-run', 1)
    assert receipt._state['events'][0]['properties']['result'] == ('cancelled' if stopped else 'failed')
