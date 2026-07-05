import pytest


@pytest.fixture(autouse=True)
def _reset_runs():
    from app import setup_installer
    setup_installer._runs.clear()
    yield
    setup_installer._runs.clear()


def test_status_idle_when_never_started():
    from app import setup_installer
    assert setup_installer.status('ml_extras') == {'state': 'idle', 'returncode': None, 'log': []}


def test_start_unknown_action_raises():
    from app import setup_installer
    with pytest.raises(ValueError):
        setup_installer.start('rm_rf')


def test_start_sets_running(monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer, '_execute', lambda action: None)  # thread no-ops
    state = setup_installer.start('ml_extras')
    assert state['state'] == 'running'


def test_start_rejects_second_run():
    from app import setup_installer
    setup_installer._runs['ml_extras'] = {'state': 'running', 'returncode': None, 'log': []}
    with pytest.raises(setup_installer.AlreadyRunning):
        setup_installer.start('ml_extras')


def test_execute_success_clears_import_cache(monkeypatch):
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'ml_extras': lambda a: 0})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs['ml_extras'] = setup_installer._new_run()
    setup_installer._execute('ml_extras')
    assert setup_installer._runs['ml_extras']['state'] == 'success'
    assert setup_installer._runs['ml_extras']['returncode'] == 0
    assert calls == [1]


def test_execute_nonzero_is_error_and_skips_cache_clear(monkeypatch):
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'ml_extras': lambda a: 1})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs['ml_extras'] = setup_installer._new_run()
    setup_installer._execute('ml_extras')
    assert setup_installer._runs['ml_extras']['state'] == 'error'
    assert calls == []


def test_execute_worker_exception_is_captured(monkeypatch):
    from app import setup_installer
    def boom(a): raise RuntimeError('nope')
    monkeypatch.setattr(setup_installer, '_WORKERS', {'ml_extras': boom})
    setup_installer._runs['ml_extras'] = setup_installer._new_run()
    setup_installer._execute('ml_extras')
    assert setup_installer._runs['ml_extras']['state'] == 'error'
    assert setup_installer._runs['ml_extras']['returncode'] == -1
    assert any('nope' in line for line in setup_installer._runs['ml_extras']['log'])


def test_run_ml_extras_captures_output(monkeypatch):
    from app import setup_installer
    class FakeProc:
        stdout = iter(['Collecting rembg\n', 'Successfully installed\n'])
        returncode = 0
        def wait(self): return 0
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', lambda *a, **k: FakeProc())
    setup_installer._runs['ml_extras'] = setup_installer._new_run()
    rc = setup_installer._run_ml_extras('ml_extras')
    assert rc == 0
    assert any('rembg' in line for line in setup_installer._runs['ml_extras']['log'])


def test_run_ollama_model_streams(app, monkeypatch):
    from app import setup_installer, config
    class FakeResp:
        status_code = 200
        def iter_lines(self): return [b'{"status":"pulling"}', b'{"status":"success"}']
    with app.app_context():
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(setup_installer.requests, 'post', lambda *a, **k: FakeResp())
        setup_installer._runs['ollama_model'] = setup_installer._new_run()
        rc = setup_installer._run_ollama_model('ollama_model')
    assert rc == 0
    assert any('success' in line for line in setup_installer._runs['ollama_model']['log'])


def test_start_ollama_model_precondition(app):
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'ollama': {'url': '', 'vision_model': ''}})
        with pytest.raises(setup_installer.Precondition):
            setup_installer.start('ollama_model')
