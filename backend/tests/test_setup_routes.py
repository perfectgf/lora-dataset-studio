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
    from app import config
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


# --- Docker-managed service readiness --------------------------------------

def test_runtime_readiness_gpu_boot_polls_only_integrated_comfyui(
        client, monkeypatch):
    from app import capabilities
    calls = []

    monkeypatch.setenv('LDS_RUNTIME', 'docker-gpu')
    monkeypatch.setattr(capabilities.cfg, 'get', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(
        capabilities,
        '_http_ok',
        lambda url, timeout=3, **_kwargs: calls.append((url, timeout)) or False,
    )

    response = client.get('/api/setup/runtime-readiness')
    body = response.get_json()

    assert body == {
        'comfyui': {
            'mode': 'integrated', 'state': 'starting',
            'ready': False, 'poll': True,
        },
        'ollama': {
            'mode': 'unconfigured', 'state': 'unconfigured',
            'ready': False, 'poll': False,
        },
    }
    assert calls == [('http://127.0.0.1:8188/history', 1.0)]
    assert all(timeout <= 1.0 for _, timeout in calls)
    assert response.headers['Cache-Control'] == 'no-store'


def test_runtime_readiness_managed_services_turn_ready_without_full_probe(
        client, monkeypatch):
    from app import capabilities

    monkeypatch.setenv('LDS_RUNTIME', 'docker-gpu')
    monkeypatch.setattr(
        capabilities.cfg, 'get',
        lambda key, default=None: 'docker' if key == 'ollama.deployment_mode' else default)
    monkeypatch.setattr(capabilities, '_http_ok', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        capabilities,
        'probe',
        lambda *_args, **_kwargs: pytest.fail('full capability probe was called'),
    )

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert body['comfyui'] == {
        'mode': 'integrated', 'state': 'ready',
        'ready': True, 'poll': False,
    }
    assert body['ollama'] == {
        'mode': 'docker', 'state': 'ready',
        'ready': True, 'poll': False,
    }


def test_runtime_readiness_ollama_none_is_disabled_and_never_probed(
        client, monkeypatch):
    from app import capabilities
    calls = []

    monkeypatch.setenv('LDS_RUNTIME', 'desktop')
    # Native installs ignore Docker-only launcher choices.
    monkeypatch.setenv('LDS_OLLAMA_MODE', 'none')
    monkeypatch.setattr(
        capabilities.cfg, 'get', lambda *_args, **_kwargs: 'http://127.0.0.1:11434')
    monkeypatch.setattr(
        capabilities,
        '_http_ok',
        lambda url, **_kwargs: calls.append(url) or False,
    )

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert body == {
        'comfyui': {
            'mode': 'external', 'state': 'manual',
            'ready': False, 'poll': False,
        },
        'ollama': {
            'mode': 'local', 'state': 'unreachable',
            'ready': False, 'poll': False,
        },
    }
    assert calls == ['http://127.0.0.1:11434/api/tags']


def test_runtime_readiness_light_docker_never_invents_integrated_comfyui(
        client, monkeypatch):
    from app import capabilities
    calls = []

    monkeypatch.setenv('LDS_RUNTIME', 'docker')
    monkeypatch.setattr(capabilities.cfg, 'get', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(
        capabilities,
        '_http_ok',
        lambda url, **_kwargs: calls.append(url) or False,
    )

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert body['comfyui'] == {
        'mode': 'external', 'state': 'manual',
        'ready': False, 'poll': False,
    }
    assert body['ollama']['mode'] == 'unconfigured'
    assert body['ollama']['state'] == 'unconfigured'
    assert calls == []


def test_runtime_readiness_external_host_comfy_is_manual_not_starting(
        client, monkeypatch):
    from app import capabilities

    monkeypatch.setenv('LDS_RUNTIME', 'docker-external-comfy')
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'external')
    monkeypatch.setattr(capabilities.cfg, 'get', lambda *_args, **_kwargs: 'none')
    monkeypatch.setattr(
        capabilities,
        '_http_ok',
        lambda *_args, **_kwargs: pytest.fail('external ComfyUI must not be polled here'),
    )

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert body['comfyui'] == {
        'mode': 'external-host', 'state': 'manual',
        'ready': False, 'poll': False,
    }


def test_runtime_readiness_uses_persisted_mode_and_fixed_host_url(
        client, monkeypatch):
    from app import capabilities
    calls = []
    config_reads = []

    monkeypatch.setenv('LDS_RUNTIME', 'docker-external-comfy')
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'external')
    monkeypatch.setenv('LDS_OLLAMA_MODE', 'docker')
    monkeypatch.setenv('LDS_OLLAMA_URL', 'file:///ignored.sock')

    def fake_get(key, default=None):
        config_reads.append(key)
        if key == 'ollama.deployment_mode':
            return 'host'
        return pytest.fail(f'unexpected config read: {key}')

    monkeypatch.setattr(capabilities.cfg, 'get', fake_get)
    monkeypatch.setattr(
        capabilities, '_http_ok',
        lambda url, **_kwargs: calls.append(url) or False)

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert config_reads == ['ollama.deployment_mode']
    assert calls == ['http://host.docker.internal:11434/api/tags']
    assert body['ollama'] == {
        'mode': 'host', 'state': 'unreachable',
        'ready': False, 'poll': False,
    }
    assert 'url' not in body['ollama']


def test_runtime_readiness_docker_mode_ignores_arbitrary_saved_and_env_urls(
        client, monkeypatch):
    from app import capabilities
    calls = []

    monkeypatch.setenv('LDS_RUNTIME', 'docker-external-comfy')
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'external')
    monkeypatch.setenv('LDS_OLLAMA_URL', 'file:///ignored.sock')
    monkeypatch.setattr(
        capabilities.cfg, 'get',
        lambda key, default=None: 'docker' if key == 'ollama.deployment_mode'
        else pytest.fail(f'unexpected config read: {key}'))
    monkeypatch.setattr(
        capabilities, '_http_ok',
        lambda url, **_kwargs: calls.append(url) or False)

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert calls == ['http://ollama:11434/api/tags']
    assert body['ollama'] == {
        'mode': 'docker', 'state': 'starting',
        'ready': False, 'poll': True,
    }


def test_runtime_readiness_http_probe_is_bounded_streamed_and_closed(monkeypatch):
    from app import capabilities
    seen = {}

    class Response:
        status_code = 204
        closed = False

        def close(self):
            self.closed = True

    response = Response()

    def fake_get(url, **kwargs):
        seen.update(url=url, **kwargs)
        return response

    monkeypatch.setattr(capabilities.requests, 'get', fake_get)

    assert capabilities._http_ok(
        'http://ollama:11434/api/tags', timeout=99, readiness=True) is True
    assert seen == {
        'url': 'http://ollama:11434/api/tags',
        'timeout': 1.0,
        'allow_redirects': False,
        'stream': True,
    }
    assert response.closed is True


def test_runtime_readiness_response_never_exposes_configured_urls_or_paths(
        client, monkeypatch):
    from app import capabilities
    secret_url = 'http://user:password@private-host.internal:11434'
    secret_path = r'C:\private\ComfyUI'

    monkeypatch.setenv('LDS_RUNTIME', 'docker-gpu')
    monkeypatch.setenv('LDS_OLLAMA_MODE', 'host')
    monkeypatch.setattr(
        capabilities.cfg,
        'get',
        lambda key, *_args: ('host' if key == 'ollama.deployment_mode'
                             else secret_url if key == 'ollama.url' else secret_path),
    )
    monkeypatch.setattr(capabilities, '_http_ok', lambda *_args, **_kwargs: False)

    response = client.get('/api/setup/runtime-readiness')
    body = response.get_json()
    raw = response.get_data(as_text=True)

    assert response.status_code == 200
    assert set(body) == {'comfyui', 'ollama'}
    assert 'password' not in raw
    assert 'private-host' not in raw
    assert 'private' not in raw


@pytest.mark.parametrize(('runtime', 'docker_mode'), [
    ('docker-gpu', ''),
    ('docker-external-comfy', 'external'),
])
def test_portable_comfyui_start_is_refused_for_docker_owned_runtimes(
        client, monkeypatch, runtime, docker_mode):
    from app.services import comfyui_control

    monkeypatch.setenv('LDS_RUNTIME', runtime)
    if docker_mode:
        monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', docker_mode)
    else:
        monkeypatch.delenv('LDS_DOCKER_COMFY_MODE', raising=False)
    monkeypatch.setattr(
        comfyui_control,
        'start_comfyui',
        lambda: pytest.fail('Docker runtime must never spawn portable ComfyUI'),
    )

    response = client.post('/api/setup/comfyui/start')

    assert response.status_code == 409
    assert 'Docker' in response.get_json()['error']



@pytest.mark.parametrize(('mode', 'url'), [
    ('none', ''),
    ('host', 'http://host.docker.internal:11434'),
    ('docker', 'http://ollama:11434'),
])
def test_save_ollama_deployment_persists_only_fixed_contract(
        client, monkeypatch, mode, url):
    from app import capabilities
    from app.routes import setup as setup_routes
    saved = {}

    monkeypatch.setenv('LDS_RUNTIME', 'docker')
    monkeypatch.setattr(
        setup_routes.cfg, 'save_config',
        lambda partial: saved.update(partial) or partial)
    monkeypatch.setattr(
        capabilities, 'setup_runtime_readiness',
        lambda: {'ollama': {'mode': mode, 'state': 'ready',
                            'ready': True, 'poll': False}})

    response = client.put('/api/setup/ollama-deployment', json={'mode': mode})

    assert response.status_code == 200
    assert saved == {'ollama': {'deployment_mode': mode, 'url': url}}
    assert response.get_json()['config'] == saved


@pytest.mark.parametrize('payload', [
    None,
    {},
    {'mode': 'Docker'},
    {'mode': 'docker', 'url': 'http://attacker.invalid'},
    {'mode': 1},
])
def test_save_ollama_deployment_rejects_non_contract_payload(
        client, monkeypatch, payload):
    monkeypatch.setenv('LDS_RUNTIME', 'docker')
    response = client.put('/api/setup/ollama-deployment', json=payload)
    assert response.status_code == 400


def test_save_ollama_deployment_is_docker_only(client, monkeypatch):
    monkeypatch.setenv('LDS_RUNTIME', 'desktop')
    response = client.put('/api/setup/ollama-deployment', json={'mode': 'none'})
    assert response.status_code == 409


def test_ollama_cancel_endpoint_is_idempotent(client):
    from app import setup_installer

    setup_installer._runs['ollama_model'] = setup_installer._new_run()
    first = client.post('/api/setup/install/ollama_model/cancel')
    second = client.post('/api/setup/install/ollama_model/cancel')

    assert first.status_code == second.status_code == 200
    assert first.get_json()['cancel_requested'] is True
    assert second.get_json()['cancel_requested'] is True


def test_cancel_rejects_non_streamed_installs(client):
    response = client.post('/api/setup/install/ml_extras/cancel')
    assert response.status_code == 409
