import pytest


@pytest.fixture(autouse=True)
def _reset_runs():
    from app import setup_installer
    setup_installer._runs.clear()
    yield
    setup_installer._runs.clear()


def test_install_unknown_action_404(client):
    assert client.post('/api/setup/install/rm_rf').status_code == 404


def test_status_unknown_action_404(client):
    assert client.get('/api/setup/install/rm_rf/status').status_code == 404


def test_install_ml_extras_starts(client, monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer, 'start',
                        lambda a: {'state': 'running', 'returncode': None, 'log': []})
    r = client.post('/api/setup/install/ml_extras')
    assert r.status_code == 200 and r.get_json()['state'] == 'running'


def test_install_conflict_409(client, monkeypatch):
    from app import setup_installer
    def _raise(a): raise setup_installer.AlreadyRunning(a)
    monkeypatch.setattr(setup_installer, 'start', _raise)
    assert client.post('/api/setup/install/ml_extras').status_code == 409


def test_install_ollama_precondition_400(client, monkeypatch):
    from app import config, setup_installer
    config.save_config({'ollama': {'url': '', 'vision_model': ''}})
    # real start() runs the precondition check and raises before spawning a thread
    assert client.post('/api/setup/install/ollama_model').status_code == 400


def test_status_idle(client):
    r = client.get('/api/setup/install/ml_extras/status')
    assert r.status_code == 200 and r.get_json()['state'] == 'idle'
