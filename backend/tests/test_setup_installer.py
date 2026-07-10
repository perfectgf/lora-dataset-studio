import pytest


@pytest.fixture(autouse=True)
def _reset_runs():
    from app import setup_installer
    setup_installer._runs.clear()
    yield
    setup_installer._runs.clear()


def test_status_idle_when_never_started():
    from app import setup_installer
    s = setup_installer.status('ml_extras')
    assert s['state'] == 'idle' and s['returncode'] is None and s['log'] == []
    assert 'manual_command' in s   # always present so the UI can show a correct fallback


def test_manual_command_ml_extras_is_scoped_to_this_interpreter():
    """The manual fallback must target THIS app's interpreter (sys.executable),
    not a bare `pip` on PATH -- otherwise a copy-paste installs into the wrong
    environment and the extras stay unimportable."""
    import sys
    from app import setup_installer
    cmd = setup_installer.manual_command('ml_extras')
    assert sys.executable in cmd
    assert '-m pip install -r' in cmd
    assert 'requirements-ml.txt' in cmd
    assert not cmd.startswith('pip ')   # never bare pip

def test_manual_command_quotes_paths_with_spaces(monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer.sys, 'executable', r'C:\LoRA Dataset Studio\python\python.exe')
    cmd = setup_installer.manual_command('ml_extras')
    assert '"C:\\LoRA Dataset Studio\\python\\python.exe"' in cmd

def test_manual_command_ollama_model(app):
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'ollama': {'vision_model': 'qwen3-vl:8b'}})
        assert setup_installer.manual_command('ollama_model') == 'ollama pull qwen3-vl:8b'

def test_status_includes_manual_command_while_running():
    from app import setup_installer
    setup_installer._runs['ml_extras'] = {'state': 'running', 'returncode': None, 'log': ['x']}
    s = setup_installer.status('ml_extras')
    assert s['state'] == 'running' and 'requirements-ml.txt' in s['manual_command']


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


def test_execute_ollama_model_success_does_not_clear_cache(monkeypatch):
    """Verify ollama_model runs never trigger clear_import_cache, even on success."""
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'ollama_model': lambda a: 0})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs['ollama_model'] = setup_installer._new_run()
    setup_installer._execute('ollama_model')
    assert setup_installer._runs['ollama_model']['state'] == 'success'
    assert setup_installer._runs['ollama_model']['returncode'] == 0
    assert calls == []  # clear_import_cache never called


def test_append_respects_log_ring_buffer(monkeypatch):
    """Verify _append maintains a ring buffer of _LOG_MAX lines, keeping newest."""
    from app import setup_installer
    setup_installer._runs['test_action'] = setup_installer._new_run()
    action = 'test_action'

    # Append more than _LOG_MAX lines
    for i in range(setup_installer._LOG_MAX + 100):
        setup_installer._append(action, f'line_{i}\n')

    log = setup_installer._runs[action]['log']
    assert len(log) == setup_installer._LOG_MAX
    assert log[-1] == f'line_{setup_installer._LOG_MAX + 99}'  # newest kept
    assert log[0] == 'line_100'  # oldest kept (first 100 dropped)


def test_run_ollama_model_http_error(app, monkeypatch):
    """Verify _run_ollama_model handles HTTP errors (>= 400) and logs them."""
    from app import setup_installer, config

    class FakeResp:
        status_code = 500
        def iter_lines(self):
            return []

    with app.app_context():
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(setup_installer.requests, 'post', lambda *a, **k: FakeResp())
        setup_installer._runs['ollama_model'] = setup_installer._new_run()
        rc = setup_installer._run_ollama_model('ollama_model')

    assert rc == 1
    assert any('HTTP 500' in line for line in setup_installer._runs['ollama_model']['log'])


# --- Klein one-click downloads --------------------------------------------
# The four Klein assets download straight into the VALIDATED ComfyUI tree so the
# model listers pick them up with zero manual file-moving. A fake ComfyUI dir is
# just main.py + models/ (what capabilities._is_comfyui_dir checks).
import os


def _make_comfyui(root):
    base = root / 'ComfyUI'
    (base / 'models').mkdir(parents=True, exist_ok=True)
    (base / 'main.py').write_text('# fake ComfyUI entrypoint', encoding='utf-8')
    return base


class _FakeGet:
    """Stand-in for requests.get(..., stream=True) used as a context manager."""
    def __init__(self, status=200, payload=b'', capture=None):
        self._payload = payload
        self.status_code = status
        self.headers = {'content-length': str(len(payload))} if payload else {}
        self._capture = capture

    def __call__(self, url, **kw):
        if self._capture is not None:
            self._capture['url'] = url
            self._capture['headers'] = kw.get('headers')
        return self

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def iter_content(self, chunk_size=0):
        if self._payload:
            yield self._payload


def test_install_actions_include_klein_downloads():
    from app import setup_installer
    for a in ('klein_model', 'klein_lora', 'klein_text_encoder', 'klein_vae'):
        assert a in setup_installer.INSTALL_ACTIONS


def test_every_action_has_a_worker_and_manual_command(app):
    """Structural invariant: EVERY whitelisted action must have a worker (a
    missing entry died at runtime as \"error: 'scrape_extras'\" — the action was
    in INSTALL_ACTIONS/manual_command but absent from _WORKERS) and a non-empty
    manual fallback so the UI can always show a copy-paste alternative."""
    from app import setup_installer
    with app.app_context():
        for a in setup_installer.INSTALL_ACTIONS:
            assert a in setup_installer._WORKERS, f'no worker for {a}'
            assert setup_installer.manual_command(a), f'no manual command for {a}'


def test_run_scrape_extras_targets_scrape_requirements(monkeypatch):
    """The shared pip worker must install the file matching THE ACTION — not
    always requirements-ml.txt."""
    from app import setup_installer
    seen = {}
    class FakeProc:
        stdout = iter(())
        returncode = 0
        def wait(self): return 0
    def fake_popen(cmd, **kw):
        seen['cmd'] = cmd
        return FakeProc()
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', fake_popen)
    setup_installer._runs['scrape_extras'] = setup_installer._new_run()
    rc = setup_installer._run_ml_extras('scrape_extras')
    assert rc == 0
    assert any('requirements-scrape.txt' in str(part) for part in seen['cmd'])


def test_klein_dest_path_under_validated_base(app, tmp_path):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._klein_dest_path('klein_lora')
    assert dest == str(base / 'models' / 'loras' / 'klein'
                       / 'Flux2-Klein-9B-consistency-V2.safetensors')


def test_klein_model_dest_is_unet_klein(app, tmp_path):
    """The diffusion model must land in models/unet/klein/ -- that is the ONLY
    place capabilities._scan_models() detects a Klein UNET (base_name 'unet',
    subfolder named 'klein'). A wrong subfolder downloads 15 GB the app can't see."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._klein_dest_path('klein_model')
    assert dest.endswith(os.path.join('models', 'unet', 'klein',
                                      'flux-2-klein-9b-fp8.safetensors'))


def test_klein_dest_path_requires_valid_comfyui(app, tmp_path):
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(tmp_path / 'not-comfyui')}})
        with pytest.raises(setup_installer.Precondition):
            setup_installer._klein_dest_path('klein_lora')


def test_manual_command_klein_lora_is_curl_to_real_url(app, tmp_path):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        cmd = setup_installer.manual_command('klein_lora')
    assert cmd.startswith('curl -L -o ')
    assert 'huggingface.co/dx8152/Flux2-Klein-9B-Consistency' in cmd
    assert 'Flux2-Klein-9B-consistency-V2.safetensors' in cmd


def test_manual_command_klein_uses_placeholder_when_unconfigured(app, tmp_path):
    """No validated ComfyUI -> the copy-paste command still makes sense, pointing
    at a <ComfyUI> placeholder instead of raising."""
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': ''}})
        cmd = setup_installer.manual_command('klein_vae')
    assert '<ComfyUI>' in cmd
    assert 'flux2-vae.safetensors' in cmd


def test_run_klein_download_gated_401_logs_recovery_steps(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(status=401))
        setup_installer._runs['klein_model'] = setup_installer._new_run()
        rc = setup_installer._run_klein_download('klein_model')
    assert rc == 1
    log = setup_installer._runs['klein_model']['log']
    assert any('license-gated' in l for l in log)
    assert any('HF_TOKEN' in l for l in log)


def test_run_klein_download_streams_to_part_then_renames(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    payload = b'x' * (10 * 1024 * 1024)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(payload=payload))
        setup_installer._runs['klein_vae'] = setup_installer._new_run()
        rc = setup_installer._run_klein_download('klein_vae')
        dest = setup_installer._klein_dest_path('klein_vae')
    assert rc == 0
    assert os.path.isfile(dest)
    assert not os.path.exists(dest + '.part')   # atomic rename left no partial


def test_run_klein_download_sends_bearer_when_token_set(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    cap = {}
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setenv('HF_TOKEN', 'hf_secret')
        monkeypatch.setattr(setup_installer.requests, 'get',
                            _FakeGet(payload=b'z' * 1024, capture=cap))
        setup_installer._runs['klein_model'] = setup_installer._new_run()
        setup_installer._run_klein_download('klein_model')
    assert cap['headers'].get('Authorization') == 'Bearer hf_secret'


def test_run_klein_download_skips_when_already_present(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    def boom(*a, **k):
        raise AssertionError('network must not be hit when the file already exists')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._klein_dest_path('klein_lora')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as f:
            f.write(b'already downloaded')
        monkeypatch.setattr(setup_installer.requests, 'get', boom)
        setup_installer._runs['klein_lora'] = setup_installer._new_run()
        rc = setup_installer._run_klein_download('klein_lora')
    assert rc == 0
    assert any('already present' in l for l in setup_installer._runs['klein_lora']['log'])


def test_start_klein_blocks_on_low_disk(app, tmp_path, monkeypatch):
    import collections
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    Usage = collections.namedtuple('Usage', 'total used free')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        # 1 GB free vs 15 GB needed for the model -> precondition must block start()
        monkeypatch.setattr(setup_installer.shutil, 'disk_usage',
                            lambda p: Usage(0, 0, int(1e9)))
        with pytest.raises(setup_installer.Precondition):
            setup_installer.start('klein_model')


def test_start_klein_requires_valid_comfyui(app, tmp_path):
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(tmp_path / 'nope')}})
        with pytest.raises(setup_installer.Precondition):
            setup_installer.start('klein_lora')


def test_execute_klein_success_clears_model_caches(monkeypatch):
    from app import setup_installer
    import app.utils.comfyui as comfyui
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'klein_lora': lambda a: 0})
    monkeypatch.setattr(comfyui, 'clear_model_caches', lambda: calls.append(1))
    setup_installer._runs['klein_lora'] = setup_installer._new_run()
    setup_installer._execute('klein_lora')
    assert setup_installer._runs['klein_lora']['state'] == 'success'
    assert calls == [1]
