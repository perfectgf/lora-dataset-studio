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


# --- "Install everything" orchestrator endpoints ---------------------------

def test_install_all_plan_endpoint(client, monkeypatch):
    from app import capabilities, setup_installer
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: {'python': {'ml_supported': True}})
    monkeypatch.setattr(setup_installer, 'install_all_plan', lambda caps: ['face_scoring', 'masks'])
    r = client.get('/api/setup/install-all/plan')
    assert r.status_code == 200 and r.get_json()['plan'] == ['face_scoring', 'masks']


def test_install_all_starts_plan(client, monkeypatch):
    from app import capabilities, setup_installer
    started = []
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: {})
    monkeypatch.setattr(setup_installer, 'start',
                        lambda a: (started.append(a) or {'state': 'running', 'returncode': None,
                                                          'log': [], 'progress': None,
                                                          'waiting_for': None,
                                                          'manual_command': ''}))
    r = client.post('/api/setup/install-all')
    body = r.get_json()
    assert r.status_code == 200
    # the {} snapshot -> the always-runnable extras (scrape stack + the three ML ones)
    assert body['plan'] == ['scrape_extras', 'face_scoring', 'masks', 'watermark_inpaint']
    assert set(body['statuses']) == set(body['plan'])
    assert started == body['plan']


def test_install_all_status_batches_requested_actions(client):
    r = client.get('/api/setup/install-all/status',
                   query_string={'actions': 'face_scoring,masks,not_real'})
    body = r.get_json()
    assert r.status_code == 200
    assert set(body['statuses']) == {'face_scoring', 'masks'}   # unknown dropped
    assert body['statuses']['face_scoring']['state'] == 'idle'


# --- ComfyUI directory validation endpoint (Setup Volet 1) -----------------

def test_validate_comfyui_dir_blank(client):
    r = client.get('/api/setup/comfyui-dir?path=')
    assert r.status_code == 200 and r.get_json()['status'] == 'empty'


def test_validate_comfyui_dir_valid(client, tmp_path):
    (tmp_path / 'main.py').touch()
    (tmp_path / 'models').mkdir()
    r = client.get('/api/setup/comfyui-dir', query_string={'path': str(tmp_path)})
    assert r.status_code == 200 and r.get_json()['status'] == 'valid'


def test_validate_comfyui_dir_nested_suggests_child(client, tmp_path):
    child = tmp_path / 'ComfyUI'
    child.mkdir()
    (child / 'main.py').touch()
    (child / 'models').mkdir()
    r = client.get('/api/setup/comfyui-dir', query_string={'path': str(tmp_path)})
    body = r.get_json()
    assert body['status'] == 'nested'
    assert body['suggestion'].endswith('ComfyUI')


def test_validate_comfyui_dir_missing(client, tmp_path):
    r = client.get('/api/setup/comfyui-dir', query_string={'path': str(tmp_path / 'nope')})
    assert r.get_json()['status'] == 'missing'
