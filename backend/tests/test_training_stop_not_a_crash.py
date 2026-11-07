"""A Stop ends ai-toolkit with a non-zero exit (taskkill /F). That exit is the Stop
itself, not a crash: no training_error, hence no crash banner and no ↻ Retry on the
Runs page, while the checkpoints it saved stay the run's. A real crash of a run
nobody stopped keeps the banner."""
from flask import Flask

from app.services import lora_training as lt


class _KilledProcess:
    returncode = 1

    @staticmethod
    def wait():
        return None


def _watch(monkeypatch, token):
    states = {}
    monkeypatch.setattr(lt, '_crash_payload', lambda *_a: {
        'dataset_id': 61, 'rc': 1, 'log_tail': '', 'excerpt': {'kind': 'error', 'text': 'killed'}})
    monkeypatch.setattr(lt, 'process_training_queue', lambda: None)
    monkeypatch.setattr(lt.queue_manager, '_set_system_state',
                        lambda key, value, **_kw: states.update({key: value}))
    lt._watch_training(Flask(__name__), _KilledProcess(), 'unused.log', 61, usage_run_token=token)
    return states


def test_a_stopped_run_is_not_reported_as_a_crash(monkeypatch):
    monkeypatch.setattr(lt, '_STOPPED_RUN_TOKENS', {'run-stopped'})
    assert 'training_error' not in _watch(monkeypatch, 'run-stopped')
    assert lt._STOPPED_RUN_TOKENS == set()


def test_a_run_nobody_stopped_still_reports_its_crash(monkeypatch):
    monkeypatch.setattr(lt, '_STOPPED_RUN_TOKENS', {'another-run'})
    assert _watch(monkeypatch, 'run-crashed')['training_error']['rc'] == 1
    assert lt._STOPPED_RUN_TOKENS == {'another-run'}
