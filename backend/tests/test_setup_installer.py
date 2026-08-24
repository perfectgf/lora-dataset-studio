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
    environment and the extras stay unimportable. It installs the Flask-safe subset
    (requirements-ml.txt pinned as a -c constraint) with Pillow pinned, and NEVER
    the Pillow-incompatible extra (that one needs its own env)."""
    import sys
    from app import setup_installer
    cmd = setup_installer.manual_command('ml_extras')
    assert sys.executable in cmd
    assert '-m pip install' in cmd
    assert '-c ' in cmd and 'requirements-ml.txt' in cmd   # the file rides as a constraint
    assert 'simple-lama-inpainting' not in cmd             # the poison never lands here
    assert 'Pillow==' in cmd                               # app's Pillow pinned as a guard
    assert not cmd.startswith('pip ')   # never bare pip

def test_manual_command_quotes_paths_with_spaces(monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer.sys, 'executable', r'C:\LoRA Dataset Studio\python\python.exe')
    cmd = setup_installer.manual_command('ml_extras')
    assert '"C:\\LoRA Dataset Studio\\python\\python.exe"' in cmd

def test_ollama_model_has_no_cli_fallback():
    from app import setup_installer

    assert setup_installer.manual_command('ollama_model') == ''

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


def test_reinstall_reruns_worker_after_a_successful_install(monkeypatch):
    """Reinstall = re-clicking an already-green item (the Setup install menu's ↻ Reinstall).
    start() must RE-RUN the worker on a terminal (success/error) run — there is deliberately
    NO 'already installed, skip' gate, so a broken venv sitting behind a green capability can
    always be re-provisioned in place (the whole point of a reinstall button)."""
    from app import setup_installer
    calls = []
    monkeypatch.setattr(setup_installer, '_execute', lambda a: calls.append(a))  # thread no-ops
    setup_installer._pip_current = None
    setup_installer._pip_queue = []
    # A prior install of a venv-backed capability finished successfully.
    setup_installer._runs['masks'] = {'state': 'success', 'returncode': 0, 'log': [],
                                      'progress': None, 'waiting_for': None}
    state = setup_installer.start('masks')
    assert state['state'] == 'running'   # reset to running (not rejected as AlreadyRunning)
    assert calls == ['masks']            # the worker (which re-provisions/repairs) re-ran


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


def test_run_ollama_model_streams_progress_over_http(app, monkeypatch):
    from app import setup_installer, config
    seen = {}

    class FakeResp:
        status_code = 200
        closed = False

        def iter_content(self, chunk_size):
            assert chunk_size == setup_installer._OLLAMA_STREAM_CHUNK
            yield b'{"status":"pulling","completed":25,"total":100}\n'
            yield b'{"status":"success","completed":100,"total":100}\n'

        def close(self):
            self.closed = True

    response = FakeResp()

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return response

    with app.app_context():
        config.save_config({'ollama': {'url': 'http://o', 'vision_model': 'qwen3-vl:8b'}})
        monkeypatch.setattr(setup_installer.requests, 'post', fake_post)
        setup_installer._runs['ollama_model'] = setup_installer._new_run()
        rc = setup_installer._run_ollama_model('ollama_model')

    assert rc == 0
    assert seen['url'] == 'http://o/api/pull'
    assert seen['json'] == {'model': 'qwen3-vl:8b', 'stream': True}
    assert seen['stream'] is True and seen['allow_redirects'] is False
    assert setup_installer._runs['ollama_model']['progress']['pct'] == 100
    assert response.closed is True


def test_start_ollama_model_precondition(app):
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'ollama': {'url': '', 'vision_model': ''}})
        with pytest.raises(setup_installer.Precondition):
            setup_installer.start('ollama_model')


def test_execute_ollama_model_success_clears_probe_cache(monkeypatch):
    """A successful vision-model pull invalidates the probe cache so the Setup step /
    diagnostic flip to 'vision model ready' immediately, not after the 30 s TTL
    (issue #7: Setup kept saying 'not pulled' right after the pull finished).
    clear_import_cache() also resets the main probe cache — the one call that forces a
    fresh /api/tags check next probe."""
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'ollama_model': lambda a: 0})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs['ollama_model'] = setup_installer._new_run()
    setup_installer._execute('ollama_model')
    assert setup_installer._runs['ollama_model']['state'] == 'success'
    assert setup_installer._runs['ollama_model']['returncode'] == 0
    assert calls == [1]  # probe cache cleared exactly once on success


def test_execute_ollama_model_error_skips_cache_clear(monkeypatch):
    """A FAILED pull (non-zero rc) must not invalidate the probe cache — nothing new
    became available, so the previous verdict stands."""
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'ollama_model': lambda a: 1})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs['ollama_model'] = setup_installer._new_run()
    setup_installer._execute('ollama_model')
    assert setup_installer._runs['ollama_model']['state'] == 'error'
    assert calls == []


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


def test_every_action_has_a_worker_and_only_ollama_omits_manual_command(app):
    """Ollama pulls through its configured HTTP service; the container does not
    require a host CLI. Every other action retains its diagnostic fallback."""
    from app import setup_installer
    with app.app_context():
        for action in setup_installer.INSTALL_ACTIONS:
            assert action in setup_installer._WORKERS, f'no worker for {action}'
            if action == 'ollama_model':
                assert setup_installer.manual_command(action) == ''
            else:
                assert setup_installer.manual_command(action), f'no manual command for {action}'


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


def test_download_dest_path_under_validated_base(app, tmp_path):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._download_dest_path('klein_lora')
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
        dest = setup_installer._download_dest_path('klein_model')
    assert dest.endswith(os.path.join('models', 'unet', 'klein',
                                      'flux-2-klein-9b-kv-fp8.safetensors'))


def test_download_dest_path_requires_valid_comfyui(app, tmp_path):
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(tmp_path / 'not-comfyui')}})
        with pytest.raises(setup_installer.Precondition):
            setup_installer._download_dest_path('klein_lora')


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


def test_run_model_download_401_logs_recovery_steps(app, tmp_path, monkeypatch):
    """The KV UNET is a public download, but if HF ever denies access (re-gated, or
    a stale token was sent) a 401/403 must still log actionable recovery steps —
    that safety net is keyed on the spec's license_url, not the (now-False) gated
    flag."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(status=401))
        setup_installer._runs['klein_model'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('klein_model')
    assert rc == 1
    log = setup_installer._runs['klein_model']['log']
    assert any('denied access' in l for l in log)
    assert any('accept the licence' in l for l in log)
    assert any('HF_TOKEN' in l for l in log)


def test_run_model_download_accepts_legacy_unet_variant(app, tmp_path, monkeypatch):
    """An install that fetched the pre-KV model (flux-2-klein-9b-fp8.safetensors)
    must NOT be told to re-download the KV build: the legacy filename sitting in
    models/unet/klein/ counts as already installed (both resolve by name), so the
    network is never touched.

    The fixture writes a REAL safetensors header now: "already installed" means the
    loader can open it, so a garbage placeholder would (correctly) be treated as a
    corrupted download and re-fetched — see test_setup_presence_integrity."""
    import json
    import struct
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    def boom(*a, **k):
        raise AssertionError('network must not be hit when a legacy Klein UNET exists')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        legacy = os.path.join(os.path.dirname(setup_installer._download_dest_path('klein_model')),
                              'flux-2-klein-9b-fp8.safetensors')
        os.makedirs(os.path.dirname(legacy), exist_ok=True)
        _hdr = json.dumps({'w': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
        with open(legacy, 'wb') as f:
            f.write(struct.pack('<Q', len(_hdr)) + _hdr + b'\x00' * 64)
        monkeypatch.setattr(setup_installer.requests, 'get', boom)
        setup_installer._runs['klein_model'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('klein_model')
    assert rc == 0
    assert any('already present' in l and 'flux-2-klein-9b-fp8.safetensors' in l
               for l in setup_installer._runs['klein_model']['log'])


def test_run_model_download_streams_to_part_then_renames(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    # A structurally valid safetensors: the worker now VERIFIES what it wrote
    # before renaming it in (an auth wall answering 200 with an HTML page used to
    # land as a perfect-looking weight file), so a blob of 'x' is legitimately
    # rejected — see test_krea_install.test_a_login_page_served_as_200_is_rejected.
    import struct
    _hdr = b'{"__metadata__":{"lds":"test"}}'
    payload = struct.pack('<Q', len(_hdr)) + _hdr + b'\0' * (10 * 1024 * 1024)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(payload=payload))
        setup_installer._runs['klein_vae'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('klein_vae')
        dest = setup_installer._download_dest_path('klein_vae')
    assert rc == 0
    assert os.path.isfile(dest)
    assert not os.path.exists(dest + '.part')   # atomic rename left no partial


def test_run_model_download_sends_bearer_when_token_set(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    cap = {}
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        monkeypatch.setenv('HF_TOKEN', 'hf_secret')
        monkeypatch.setattr(setup_installer.requests, 'get',
                            _FakeGet(payload=b'z' * 1024, capture=cap))
        setup_installer._runs['klein_model'] = setup_installer._new_run()
        setup_installer._run_model_download('klein_model')
    assert cap['headers'].get('Authorization') == 'Bearer hf_secret'


def _safetensors_blob(size=1024):
    """A structurally VALID .safetensors: 8 little-endian bytes giving the JSON
    header length, the header, then payload. What model_integrity accepts."""
    import struct
    hdr = b'{"__metadata__":{"lds":"test"}}'
    return struct.pack('<Q', len(hdr)) + hdr + b'\0' * size


def test_run_model_download_skips_when_already_present(app, tmp_path, monkeypatch):
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    def boom(*a, **k):
        raise AssertionError('network must not be hit when the file already exists')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._download_dest_path('klein_lora')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as f:
            # A LOADABLE file. It used to be b'already downloaded' -- 18 bytes of
            # prose, which is precisely the shape this skip must no longer accept.
            f.write(_safetensors_blob())
        monkeypatch.setattr(setup_installer.requests, 'get', boom)
        setup_installer._runs['klein_lora'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('klein_lora')
    assert rc == 0
    assert any('already present' in l for l in setup_installer._runs['klein_lora']['log'])


def test_a_file_that_cannot_be_loaded_is_replaced_not_skipped(app, tmp_path, monkeypatch):
    """The dead end reported by zigzag4794 (Discord).

    His Klein UNET was on disk, in the right folder, at a plausible 9.5 GB, and
    capabilities reported it under `klein_invalid` with verdict
    `truncated_or_garbage` -- an interrupted download. The app's advice for that
    is "download it again", and the downloader answered "already present: <path>"
    and returned SUCCESS on any existing file. So the one remedy the UI offered
    could not work, and there was no way out from inside the app.

    Presence is not readability. The same validator the readiness probe uses gets
    asked first now, and a blocking verdict makes this a replacement."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    good = _safetensors_blob(4096)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._download_dest_path('klein_vae')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as f:
            # Declares a 4 GB JSON header it does not have: structurally garbage,
            # exactly what a cut-short download leaves behind.
            import struct
            f.write(struct.pack('<Q', 4 * 1024 ** 3) + b'\0' * 512)
        monkeypatch.setattr(setup_installer.requests, 'get', _FakeGet(payload=good))
        setup_installer._runs['klein_vae'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('klein_vae')
    log = setup_installer._runs['klein_vae']['log']
    assert rc == 0
    assert not any('already present' in l for l in log)
    assert any('cannot be loaded' in l for l in log)
    # It is replaced, but only by the copy that arrived — see
    # test_setup_download_replace_order.py for the ordering that guarantees it.
    assert any('until a fresh copy has actually downloaded' in l for l in log)
    with open(dest, 'rb') as f:
        assert f.read() == good        # the broken file is gone, the good one is in


def test_an_advisory_small_file_is_never_deleted(app, tmp_path, monkeypatch):
    """`too_small` is advisory: the file LOADS, it is merely suspicious. Deleting a
    user's weights on a hunch is not a repair, so the skip stands."""
    from app import setup_installer, config
    base = _make_comfyui(tmp_path)
    def boom(*a, **k):
        raise AssertionError('a loadable file must never be re-downloaded')
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        dest = setup_installer._download_dest_path('klein_model')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, 'wb') as f:
            f.write(_safetensors_blob(64))          # valid, but far under the 1 GB floor
        monkeypatch.setattr(setup_installer.requests, 'get', boom)
        setup_installer._runs['klein_model'] = setup_installer._new_run()
        rc = setup_installer._run_model_download('klein_model')
    assert rc == 0
    assert os.path.isfile(dest)
    assert any('already present' in l for l in setup_installer._runs['klein_model']['log'])


def test_a_broken_weight_is_re_queued_by_the_one_click_installs():
    """Before: `klein_missing` says nothing (the file is there), so the plan was
    empty, the Setup card said "everything is in place", and the engine stayed
    dark. A file no loader can open is not installed."""
    from app import setup_installer
    caps = {
        'face_scoring': True, 'masks': True, 'watermark_inpaint': True, 'scrape_deps': True,
        'ollama': {'reachable': False},
        'comfyui': {
            'dir_valid': True, 'reachable': True, 'klein_missing': [],
            'klein_invalid': [{'asset': 'klein_model', 'filename': 'k.safetensors',
                               'verdict': 'truncated_or_garbage', 'blocking': True}],
        },
    }
    assert setup_installer.install_all_plan(caps) == ['klein_model']

    # Same rule for the Krea group button, and the advisory verdict still doesn't count.
    krea = {'comfyui': {
        'dir_valid': True, 'reachable': True, 'krea_missing': [], 'krea_nodes_installed': True,
        'krea_nodes_missing': [],
        'krea_invalid': [
            {'asset': 'krea_model', 'filename': 'a.safetensors',
             'verdict': 'html_or_text', 'blocking': True},
            {'asset': 'krea_vae', 'filename': 'b.safetensors',
             'verdict': 'too_small', 'blocking': False},
        ]}}
    assert setup_installer.install_group_plan('krea', krea) == ['krea_model']


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


# --- watermark inpainting: scoped one-package install --------------------
# Installs JUST simple-lama-inpainting (the version floor read from
# requirements-ml.txt) into the SAME interpreter the LaMa wrapper resolves, so a
# user who already has the ML extras doesn't redo the whole step. Shown next to
# the Curate 🧽 tools; a success drops the probe import-cache (no restart).


def test_install_actions_include_watermark_inpaint(app):
    from app import setup_installer
    assert 'watermark_inpaint' in setup_installer.INSTALL_ACTIONS
    assert 'watermark_inpaint' in setup_installer._WORKERS
    with app.app_context():
        assert setup_installer.manual_command('watermark_inpaint')


def test_requirement_spec_reads_version_from_requirements_ml():
    """The version floor is the ONE written in requirements-ml.txt — parsed, never
    duplicated in the installer module (edit the file, the installer follows)."""
    from app import setup_installer
    import re
    spec = setup_installer._requirement_spec('simple-lama-inpainting')
    assert spec.replace(' ', '').startswith('simple-lama-inpainting')
    # Whatever the pin is, it must be the SAME text as in the requirements file.
    line = next(
        l.split('#', 1)[0].strip()
        for l in setup_installer._ML_REQUIREMENTS.read_text(encoding='utf-8').splitlines()
        if re.match(r'\s*simple[-_]lama[-_]inpainting', l, re.IGNORECASE)
    )
    assert spec == line


def test_requirement_spec_canonicalises_name_and_falls_back(tmp_path):
    """PEP 503 name folding (-_. + case) matches; an absent package returns the
    bare name (an unpinned install still works, it just isn't version-floored)."""
    from app import setup_installer
    # underscore/case variant still resolves to the file's dashed line
    assert setup_installer._requirement_spec('Simple_Lama_Inpainting') \
        == setup_installer._requirement_spec('simple-lama-inpainting')
    # missing line -> bare name fallback
    req = tmp_path / 'reqs.txt'
    req.write_text('numpy>=1.26,<2\n# a comment\n', encoding='utf-8')
    assert setup_installer._requirement_spec('nothere', requirements=req) == 'nothere'
    # and it reads the spec straight from an arbitrary file (single-source proof)
    req.write_text('foo-bar==1.2.3  # inline\n', encoding='utf-8')
    assert setup_installer._requirement_spec('foo_bar', requirements=req) == 'foo-bar==1.2.3'


def test_manual_command_watermark_inpaint_scoped_and_quoted(app):
    """Copy-paste command must target the WRAPPER's interpreter and quote the
    spec ('>=' is shell redirection unquoted). The version reflects the file."""
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'watermark': {'python': r'C:\ml env\python.exe'}})
        cmd = setup_installer.manual_command('watermark_inpaint')
        spec = setup_installer._requirement_spec('simple-lama-inpainting')
    assert '"C:\\ml env\\python.exe"' in cmd     # resolved interpreter, quoted for spaces
    assert '-m pip install' in cmd
    assert f'"{spec}"' in cmd                     # spec quoted whole
    assert not cmd.startswith('pip ')             # never bare pip


def _fake_popen_capturing(seen, returncode=0, lines=()):
    class FakeProc:
        stdout = iter(lines)
        def wait(self): return returncode
    FakeProc.returncode = returncode
    def fake_popen(cmd, **kw):
        seen['cmd'] = cmd
        return FakeProc()
    return fake_popen


def test_run_watermark_inpaint_targets_watermark_python(app, monkeypatch):
    """watermark.python wins the interpreter resolution and the pip target IS it,
    with the parsed spec and requirements-ml.txt pinned as a constraint (-c)."""
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 0))
    with app.app_context():
        config.save_config({'watermark': {'python': '/wm/py'},
                            'masks': {'python': '/masks/py'}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
        spec = setup_installer._requirement_spec('simple-lama-inpainting')
    assert rc == 0
    cmd = seen['cmd']
    assert cmd[0] == '/wm/py'                      # exact interpreter, not sys.executable
    assert cmd[1:4] == ['-m', 'pip', 'install']
    assert spec in cmd
    assert '-c' in cmd and any('requirements-ml.txt' in str(p) for p in cmd)


def test_run_watermark_inpaint_falls_back_to_masks_python(app, monkeypatch):
    """No dedicated watermark.python -> reuse the ML env (masks.python), matching
    the wrapper's own fallback chain."""
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 0))
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert seen['cmd'][0] == '/masks/py'


def test_run_watermark_inpaint_nonzero_returncode_propagates(app, monkeypatch):
    """A failing pip surfaces its non-zero return code (→ _execute marks 'error',
    the front shows the pip log tail — never a silent success)."""
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 1, lines=['ERROR: could not build\n']))
    with app.app_context():
        config.save_config({'watermark': {'python': '/wm/py'}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 1
    assert any('could not build' in l for l in
               setup_installer._runs['watermark_inpaint']['log'])


def test_execute_watermark_inpaint_success_clears_import_cache(monkeypatch):
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'watermark_inpaint': lambda a: 0})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
    setup_installer._execute('watermark_inpaint')
    assert setup_installer._runs['watermark_inpaint']['state'] == 'success'
    assert calls == [1]


def test_execute_watermark_inpaint_error_skips_cache_clear(monkeypatch):
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {'watermark_inpaint': lambda a: 1})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
    setup_installer._execute('watermark_inpaint')
    assert setup_installer._runs['watermark_inpaint']['state'] == 'error'
    assert calls == []


def test_execute_watermark_inpaint_success_invalidates_probe_cache(app, monkeypatch):
    """End-to-end: a successful install drops capabilities' import cache so the
    watermark_inpaint probe re-checks a freshly installed package NOW, not after
    the 600 s TTL (uses the REAL clear_import_cache, not a stub)."""
    from app import setup_installer, capabilities
    with app.app_context():
        # Poison the cache with a stale 'not installed' entry, then prove it's gone.
        capabilities._import_cache['watermark:x:import simple_lama_inpainting'] = (1e18, False)
        monkeypatch.setattr(setup_installer, '_WORKERS', {'watermark_inpaint': lambda a: 0})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        setup_installer._execute('watermark_inpaint')
    assert capabilities._import_cache == {}      # cache invalidated on success


def test_run_watermark_inpaint_verifies_import_after_install(app, monkeypatch, tmp_path):
    """A successful pip step is not enough: the installer IMPORTS the package in the
    target interpreter — that warms the heavy torch import so the probe fired right
    after (onDone → /api/capabilities) is warm rather than timing out and showing a
    '✗' seconds after a successful install."""
    from app import setup_installer, config
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_popen_capturing({}, 0))
    py = tmp_path / 'py.exe'; py.write_text('x')       # a real path so the isfile guard passes
    seen = {}
    def fake_run(cmd, **kw):
        seen['cmd'] = cmd
        class R:  # noqa: D401 - tiny stand-in for CompletedProcess
            returncode, stdout, stderr = 0, '', ''
        return R()
    monkeypatch.setattr(setup_installer.subprocess, 'run', fake_run)
    with app.app_context():
        config.save_config({'watermark': {'python': str(py)}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 0
    # Imported in the SAME interpreter pip targeted, with the exact probe
    # expression AND the isolation the real worker uses (services.infer_env):
    # a gate that verifies against a different sys.path than the pass will see
    # is a gate that can pass while the pass dies.
    assert seen['cmd'] == [str(py), '-s', '-c', 'import simple_lama_inpainting']


def test_run_watermark_inpaint_fails_when_package_does_not_import(app, monkeypatch, tmp_path):
    """pip 'already satisfied' but the package won't import (e.g. a torch/torchvision
    build mismatch pip never sees) → the install must NOT report success: the user gets
    the reason + a repair click instead of a silent ✗ capability. (JoyCaption #6 lesson.)"""
    from app import setup_installer, config
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_popen_capturing({}, 0))
    py = tmp_path / 'py.exe'; py.write_text('x')
    def fake_run(cmd, **kw):
        class R:
            returncode, stdout = 1, ''
            stderr = 'ImportError: DLL load failed while importing _C'
        return R()
    monkeypatch.setattr(setup_installer.subprocess, 'run', fake_run)
    with app.app_context():
        config.save_config({'watermark': {'python': str(py)}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 1
    log = setup_installer._runs['watermark_inpaint']['log']
    assert any('does not import' in l for l in log)
    assert any('DLL load failed' in l for l in log)   # the actionable stderr tail is surfaced


def test_run_watermark_inpaint_slow_import_still_succeeds(app, monkeypatch, tmp_path):
    """A cold import slower than the warm budget is 'still warming', not broken — the
    install stays successful (pip already succeeded) and the capability greens on the
    next probe; a slow first import must never fail a good install."""
    from app import setup_installer, config
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_popen_capturing({}, 0))
    py = tmp_path / 'py.exe'; py.write_text('x')
    def fake_run(cmd, **kw):
        raise setup_installer.subprocess.TimeoutExpired(cmd, setup_installer._WARM_IMPORT_TIMEOUT)
    monkeypatch.setattr(setup_installer.subprocess, 'run', fake_run)
    with app.app_context():
        config.save_config({'watermark': {'python': str(py)}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 0
    assert any('warming' in l for l in setup_installer._runs['watermark_inpaint']['log'])


# --- ML extras split per capability (face_scoring / masks) ----------------
# The monolithic `-r requirements-ml.txt` is now ALSO installable one capability
# at a time, so a user can install or REPAIR a single feature. Each scoped action
# targets the interpreter its own probe resolves and pins requirements-ml.txt as a
# -c constraint (numpy stays <2). The package->capability grouping lives in
# _CAPABILITY_PACKAGES; a test proves it covers every line in requirements-ml.txt.


def test_install_actions_include_face_scoring_and_masks(app):
    from app import setup_installer
    for a in ('face_scoring', 'masks'):
        assert a in setup_installer.INSTALL_ACTIONS
        assert a in setup_installer._WORKERS
        assert a in setup_installer._IMPORT_CACHE_ACTIONS   # flips the cap without a restart
        with app.app_context():
            assert setup_installer.manual_command(a)


def test_no_orphan_ml_package():
    """Anti-orphan invariant: EVERY package declared in requirements-ml.txt must be
    owned by at least one capability in _CAPABILITY_PACKAGES — a line added to the
    file but forgotten in the mapping would silently never be installed by any
    scoped action. Also proves the reverse (no mapped package is a typo absent from
    the file, which would install unpinned via the bare-name fallback)."""
    from app import setup_installer
    owned = {setup_installer._canon(p)
             for pkgs in setup_installer._CAPABILITY_PACKAGES.values() for p in pkgs}
    in_file = setup_installer._ml_requirement_names()
    assert in_file, 'requirements-ml.txt parsed empty — the mapping cannot be validated'
    orphans = in_file - owned
    assert not orphans, f'requirements-ml.txt packages mapped to no capability: {orphans}'
    phantom = owned - in_file
    assert not phantom, f'_CAPABILITY_PACKAGES names absent from requirements-ml.txt: {phantom}'


def test_run_face_scoring_targets_face_scoring_python(app, monkeypatch):
    """face_scoring.python wins the interpreter resolution (matching
    probe_face_scoring) and the pip target IS it, with insightface + onnxruntime
    from the file and requirements-ml.txt pinned as a -c constraint."""
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 0))
    with app.app_context():
        config.save_config({'face_scoring': {'python': '/fs/py'}})
        setup_installer._runs['face_scoring'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('face_scoring')
        insight = setup_installer._requirement_spec('insightface')
        onnx = setup_installer._requirement_spec('onnxruntime')
    assert rc == 0
    cmd = seen['cmd']
    assert cmd[0] == '/fs/py'                       # exact interpreter, not sys.executable
    assert cmd[1:4] == ['-m', 'pip', 'install']
    assert insight in cmd and onnx in cmd           # the version-pinned lines from the file
    assert '-c' in cmd and any('requirements-ml.txt' in str(p) for p in cmd)


def test_run_masks_targets_masks_python_and_installs_rembg(app, monkeypatch):
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 0))
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        setup_installer._run_ml_capability('masks')
        rembg = setup_installer._requirement_spec('rembg')
    cmd = seen['cmd']
    assert cmd[0] == '/masks/py'
    assert rembg in cmd
    assert '-c' in cmd and any('requirements-ml.txt' in str(p) for p in cmd)


def test_run_face_scoring_falls_back_to_sys_executable(app, monkeypatch):
    """No dedicated face_scoring.python -> install into THIS interpreter (same
    fallback probe_face_scoring uses)."""
    import sys
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 0))
    # Same reason as the Pillow test below: keep the captured command line the PIP one.
    monkeypatch.setattr(setup_installer, '_onnxruntime_provided', lambda p: False)
    monkeypatch.setattr(setup_installer, '_verify_capability_import', lambda a, p: True)
    with app.app_context():
        config.save_config({})   # no face_scoring.python
        setup_installer._runs['face_scoring'] = setup_installer._new_run()
        setup_installer._run_ml_capability('face_scoring')
    assert seen['cmd'][0] == sys.executable
    assert seen['cmd'][1:4] == ['-m', 'pip', 'install']


def test_run_ml_capability_nonzero_returncode_propagates(app, monkeypatch):
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 1, lines=['ERROR: no wheel\n']))
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        setup_installer._runs['masks'] = setup_installer._new_run()
        rc = setup_installer._run_ml_capability('masks')
    assert rc == 1
    assert any('no wheel' in l for l in setup_installer._runs['masks']['log'])


@pytest.mark.parametrize('action', ['face_scoring', 'masks'])
def test_execute_ml_capability_success_clears_import_cache(action, monkeypatch):
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {action: lambda a: 0})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs[action] = setup_installer._new_run()
    setup_installer._execute(action)
    assert setup_installer._runs[action]['state'] == 'success'
    assert calls == [1]


@pytest.mark.parametrize('action', ['face_scoring', 'masks'])
def test_execute_ml_capability_error_skips_cache_clear(action, monkeypatch):
    from app import setup_installer, capabilities
    calls = []
    monkeypatch.setattr(setup_installer, '_WORKERS', {action: lambda a: 1})
    monkeypatch.setattr(capabilities, 'clear_import_cache', lambda: calls.append(1))
    setup_installer._runs[action] = setup_installer._new_run()
    setup_installer._execute(action)
    assert setup_installer._runs[action]['state'] == 'error'
    assert calls == []


def test_manual_command_face_scoring_scoped_and_constrained(app):
    """Copy-paste command targets face_scoring's interpreter, quotes each spec
    ('>=' / '<' are shell redirection unquoted), and pins the -c constraint."""
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'face_scoring': {'python': r'C:\ml env\python.exe'}})
        cmd = setup_installer.manual_command('face_scoring')
        insight = setup_installer._requirement_spec('insightface')
    assert '"C:\\ml env\\python.exe"' in cmd        # resolved interpreter, quoted for spaces
    assert '-m pip install' in cmd
    assert f'"{insight}"' in cmd                     # spec quoted whole
    assert '-c ' in cmd and 'requirements-ml.txt' in cmd
    assert not cmd.startswith('pip ')               # never bare pip


def test_manual_command_masks_scoped(app):
    from app import setup_installer, config
    with app.app_context():
        config.save_config({'masks': {'python': '/masks/py'}})
        cmd = setup_installer.manual_command('masks')
        rembg = setup_installer._requirement_spec('rembg')
    assert '/masks/py' in cmd
    assert f'"{rembg}"' in cmd
    assert '-c ' in cmd and 'requirements-ml.txt' in cmd


# --- Flask-venv Pillow isolation ------------------------------------------
# The root of the "corrupted environment that survives updates" bug: simple-lama-
# inpainting hard-requires Pillow<10, so installing it into the app's own venv
# downgrades Pillow 12 and leaves a mixed/broken install. These lock in that NO
# Setup action can put that package (or any Pillow-incompatible one) into the Flask
# venv, and that every Flask-venv-targeted install pins Pillow so a transitive dep
# can't downgrade it either.


def test_flask_safe_ml_specs_exclude_the_incompatible_package():
    """The monolithic ml_extras package list is requirements-ml.txt MINUS the
    Pillow-incompatible extra — sourced from the file, so a new line is picked up."""
    from app import setup_installer
    specs = setup_installer._ml_requirement_specs(
        exclude=setup_installer._INCOMPATIBLE_CANON)
    joined = ' '.join(specs).lower()
    assert 'rembg' in joined and 'insightface' in joined     # the safe extras stay
    assert 'simple-lama-inpainting' not in joined            # the poison is dropped
    # and the poison IS still in the file (only excluded, not deleted — the version
    # floor for the dedicated install is read from there)
    all_specs = ' '.join(setup_installer._ml_requirement_specs()).lower()
    assert 'simple-lama-inpainting' in all_specs


def test_app_pillow_spec_is_pinned():
    from app import setup_installer
    spec = setup_installer._app_pillow_spec()
    assert spec.lower().startswith('pillow==')


def test_flask_pillow_guard_only_for_flask_venv(monkeypatch):
    import sys
    from app import setup_installer
    # the Flask venv (== sys.executable) is guarded...
    assert setup_installer._flask_pillow_guard(sys.executable) == [setup_installer._app_pillow_spec()]
    # ...a dedicated ML env is not (it may legitimately need pillow<10)
    assert setup_installer._flask_pillow_guard('/some/other/ml/python') == []


def test_run_ml_extras_installs_flask_safe_set_and_pins_pillow(monkeypatch):
    """The core guarantee: ml_extras installs the safe subset into THIS interpreter
    with Pillow pinned, never the Pillow-incompatible extra — so a full Setup can't
    downgrade the Flask venv's Pillow."""
    import sys
    from app import setup_installer
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 0))
    setup_installer._runs['ml_extras'] = setup_installer._new_run()
    rc = setup_installer._run_ml_extras('ml_extras')
    assert rc == 0
    cmd = seen['cmd']
    assert cmd[0] == sys.executable and cmd[1:4] == ['-m', 'pip', 'install']
    assert any('rembg' in str(p) for p in cmd)
    assert any('insightface' in str(p) for p in cmd)
    assert not any('simple' in str(p).lower() and 'lama' in str(p).lower() for p in cmd)
    assert '-c' in cmd and any('requirements-ml.txt' in str(p) for p in cmd)
    assert any(str(p).lower().startswith('pillow==') for p in cmd)   # Pillow guarded
    # the log directs the user to the safe way to add inpainting
    assert any('simple-lama-inpainting' in l and "own Python" in l
               for l in setup_installer._runs['ml_extras']['log'])


def test_run_watermark_inpaint_refuses_explicit_flask_venv(app, monkeypatch):
    """A user who EXPLICITLY points watermark.python at the app's own Python is
    refused (return 1, no pip) — installing simple-lama there would downgrade Pillow.
    The auto-provision path is NOT taken (we respect the user's explicit, if broken,
    value rather than silently overwriting it)."""
    import sys
    from app import setup_installer, config
    def boom(*a, **k):
        raise AssertionError('must not run pip against the Flask venv')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', boom)
    with app.app_context():
        config.save_config({'watermark': {'python': sys.executable}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 1
    log = setup_installer._runs['watermark_inpaint']['log']
    assert any('Pillow<10' in l for l in log)
    assert any('watermark.python' in l for l in log)


# --- watermark inpainting: one-click auto-provision -----------------------
# When nothing dedicated is configured the Install button BUILDS a dedicated
# 3.10-3.12 venv, installs into it, and records watermark.python — no manual venv,
# no setting to edit. The old "refuse + tell the user to make a venv" path is gone.


def _fake_venv_popen(seen, *, create_python=True):
    """Fake subprocess.Popen capturing every command. When it sees a `-m venv <dir>`
    command it creates <dir>/Scripts|bin/python(.exe) so the caller's isfile() check
    passes (a real venv would). pip commands just succeed. Also usable THROUGH
    subprocess.run (context-manager + communicate/poll): the watermark install runs a
    verification import via subprocess.run after pip, and it goes through this same fake."""
    import os
    class FakeProc:
        returncode = 0
        def __init__(self, cmd):
            self.args = cmd
            self.stdout = iter(())
            if create_python and cmd[1:3] == ['-m', 'venv']:
                from app import setup_installer
                py = setup_installer._venv_python(__import__('pathlib').Path(cmd[3]))
                os.makedirs(os.path.dirname(py), exist_ok=True)
                open(py, 'w').close()
        def wait(self): return 0
        # subprocess.run protocol (the post-install verify import uses run, not raw Popen)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def communicate(self, *a, **k): return ('', '')
        def poll(self): return 0
    def fake(cmd, **kw):
        seen.append(list(cmd))
        return FakeProc(cmd)
    return fake


def test_run_watermark_inpaint_auto_provisions_when_unconfigured(app, monkeypatch):
    """Nothing configured -> find a base Python, build the managed venv, install CPU
    torch + simple-lama into it, and SAVE watermark.python. No refusal, no manual step."""
    from app import setup_installer, config
    seen = []
    monkeypatch.setattr(setup_installer, '_find_base_python', lambda a: r'C:\pybase\python.exe')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_venv_popen(seen))
    with app.app_context():
        config.save_config({})   # nothing dedicated
        managed = setup_installer._watermark_env_python()
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
        saved = config.get('watermark.python')
    assert rc == 0
    # a venv was built from the base, then CPU torch, then simple-lama — all into it
    assert any(c[1:3] == ['-m', 'venv'] for c in seen)
    assert any(c[0] == r'C:\pybase\python.exe' and c[1:3] == ['-m', 'venv'] for c in seen)
    torch_cmd = next(c for c in seen if 'torch' in c)
    assert torch_cmd[0] == managed and '--index-url' in torch_cmd
    assert setup_installer._TORCH_CPU_INDEX in torch_cmd
    lama_cmd = next(c for c in seen if any('simple-lama' in str(p) for p in c))
    assert lama_cmd[0] == managed
    assert saved == managed          # watermark.python recorded -> probe resolves here


def test_run_watermark_inpaint_no_base_python_actionable_message(app, monkeypatch):
    """A truly bare machine (no Python 3.10-3.12) -> a short actionable message (install
    Python 3.12 / winget, then re-click) and NO pip. Never a copy-paste pip as the path."""
    from app import setup_installer, config
    def boom(*a, **k):
        raise AssertionError('no pip should run without a base Python')
    monkeypatch.setattr(setup_installer, '_find_base_python', lambda a: '')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', boom)
    with app.app_context():
        config.save_config({})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 1
    log = '\n'.join(setup_installer._runs['watermark_inpaint']['log'])
    assert 'Python 3.12' in log and 'winget' in log


def test_run_watermark_inpaint_respects_user_python_no_torch_no_overwrite(app, monkeypatch):
    """A user's OWN watermark.python is used verbatim: install simple-lama there,
    do NOT force-install CPU torch (never downgrade their CUDA build), and NEVER
    overwrite their value with the managed venv."""
    from app import setup_installer, config
    seen = []
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_venv_popen(seen))
    with app.app_context():
        config.save_config({'watermark': {'python': r'C:\ml\python.exe'}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
        saved = config.get('watermark.python')
    assert rc == 0
    assert not any('torch' in c and '--index-url' in c for c in seen)   # no forced CPU torch
    assert not any(c[1:3] == ['-m', 'venv'] for c in seen)              # no venv built
    lama_cmd = next(c for c in seen if any('simple-lama' in str(p) for p in c))
    assert lama_cmd[0] == r'C:\ml\python.exe'
    assert saved == r'C:\ml\python.exe'                                  # value untouched


def test_run_watermark_inpaint_idempotent_reuses_existing_env(app, monkeypatch):
    """A re-click after a successful provision reuses the SAME venv (no second venv
    built), so it repairs/upgrades in place — never a duplicate."""
    import os
    from app import setup_installer, config
    seen = []
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_venv_popen(seen))
    with app.app_context():
        managed = setup_installer._watermark_env_python()
        os.makedirs(os.path.dirname(managed), exist_ok=True)
        open(managed, 'w').close()                       # the venv already exists
        config.save_config({'watermark': {'python': managed}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 0
    assert not any(c[1:3] == ['-m', 'venv'] for c in seen)   # reused, not rebuilt
    assert any(any('simple-lama' in str(p) for p in c) for c in seen)


def test_run_watermark_inpaint_rebuilds_missing_managed_env(app, monkeypatch):
    """watermark.python still points at the managed venv but it was deleted -> rebuild
    it (self-heal) instead of failing on a dead interpreter path."""
    from app import setup_installer, config
    seen = []
    monkeypatch.setattr(setup_installer, '_find_base_python', lambda a: r'C:\pybase\python.exe')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_venv_popen(seen))
    with app.app_context():
        managed = setup_installer._watermark_env_python()   # does NOT exist on disk
        config.save_config({'watermark': {'python': managed}})
        setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
        rc = setup_installer._run_watermark_inpaint('watermark_inpaint')
    assert rc == 0
    assert any(c[1:3] == ['-m', 'venv'] for c in seen)       # rebuilt


# --- base-Python discovery (version checked by EXECUTION, not name) --------


def test_find_base_python_picks_first_in_range(monkeypatch):
    from app import setup_installer
    versions = {'/py313': (3, 13), '/py312': (3, 12), '/py310': (3, 10), '/broken': None}
    monkeypatch.setattr(setup_installer, '_base_python_candidates',
                        lambda: ['/broken', '/py313', '/py312', '/py310'])
    monkeypatch.setattr(setup_installer, '_python_minor', lambda e: versions[e])
    setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
    # first IN-RANGE wins: /broken (None) and /py313 (out of range) are skipped
    assert setup_installer._find_base_python('watermark_inpaint') == '/py312'


def test_find_base_python_empty_when_none_in_range(monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer, '_base_python_candidates', lambda: ['/py313', '/py39'])
    monkeypatch.setattr(setup_installer, '_python_minor',
                        lambda e: {'/py313': (3, 13), '/py39': (3, 9)}[e])
    setup_installer._runs['watermark_inpaint'] = setup_installer._new_run()
    assert setup_installer._find_base_python('watermark_inpaint') == ''


def test_base_python_candidates_includes_sys_executable(app):
    """sys.executable is a candidate BASE (the portable bundle's own 3.12 is a valid
    base with no other Python installed) — but only USED if its executed version is in
    range, which _find_base_python enforces separately."""
    import sys, os
    from app import setup_installer
    with app.app_context():
        cands = setup_installer._base_python_candidates()
    norm = [os.path.normcase(os.path.abspath(c)) for c in cands]
    assert os.path.normcase(os.path.abspath(sys.executable)) in norm


# --- pip serialization: one at a time, queued in click order --------------


def test_second_pip_action_is_queued_not_run(app, monkeypatch):
    """A second pip install requested while one runs is QUEUED (not rejected, not run
    concurrently) with waiting_for pointing at the running action."""
    from app import setup_installer
    monkeypatch.setattr(setup_installer, '_execute', lambda a: None)  # threads no-op
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()
    with app.app_context():
        s1 = setup_installer.start('face_scoring')
        s2 = setup_installer.start('masks')
    assert s1['state'] == 'running'
    assert s2['state'] == 'queued' and s2['waiting_for'] == 'face_scoring'
    assert setup_installer._pip_queue == ['masks']
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()


def test_release_pip_slot_starts_next_in_fifo_order(app, monkeypatch):
    from app import setup_installer
    started = []
    monkeypatch.setattr(setup_installer, '_execute', lambda a: started.append(a))
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()
    with app.app_context():
        setup_installer.start('face_scoring')   # runs
        setup_installer.start('masks')          # queued
        setup_installer.start('watermark_inpaint')  # queued behind masks
    assert setup_installer._pip_queue == ['masks', 'watermark_inpaint']
    setup_installer._release_pip_slot('face_scoring')
    assert setup_installer._pip_current == 'masks'
    assert setup_installer._runs['masks']['state'] == 'running'
    assert setup_installer._pip_queue == ['watermark_inpaint']
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()


def test_model_download_not_blocked_by_pip_queue(app, tmp_path, monkeypatch):
    """A Klein model download touches models/, not a venv — it must run in parallel
    with a pip install, never sit in the pip queue."""
    from app import setup_installer, config
    monkeypatch.setattr(setup_installer, '_execute', lambda a: None)
    monkeypatch.setattr(setup_installer, '_check_download_precondition', lambda a: None)
    setup_installer._pip_current = 'masks'   # simulate a pip install already running
    setup_installer._pip_queue.clear()
    base = _make_comfyui(tmp_path)
    with app.app_context():
        config.save_config({'comfyui': {'base_dir': str(base)}})
        s = setup_installer.start('klein_lora')
    assert s['state'] == 'running'                 # started, not queued
    assert 'klein_lora' not in setup_installer._pip_queue
    setup_installer._pip_current = None


# --- retry with backoff on a transient file-lock error (AV / indexer) ------


def test_run_pip_retries_on_transient_lock_then_succeeds(monkeypatch):
    """An Errno 13 / WinError-style lock (an antivirus holding a fresh file) is retried;
    the second attempt succeeds. pip is idempotent, so the rerun finishes the step."""
    from app import setup_installer
    monkeypatch.setattr(setup_installer.time, 'sleep', lambda s: None)
    attempts = {'n': 0}
    class FakeProc:
        def __init__(self, fail):
            self._fail = fail
            self.stdout = iter(["ERROR: Could not install: [Errno 13] Permission denied\n"]
                               if fail else ["Successfully installed\n"])
            self.returncode = 1 if fail else 0
        def wait(self): return self.returncode
    def fake_popen(cmd, **kw):
        attempts['n'] += 1
        return FakeProc(fail=(attempts['n'] == 1))
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', fake_popen)
    setup_installer._runs['masks'] = setup_installer._new_run()
    rc = setup_installer._run_pip('masks', ['py', '-m', 'pip', 'install', 'x'])
    assert rc == 0 and attempts['n'] == 2
    assert any('retrying' in l for l in setup_installer._runs['masks']['log'])


def test_run_pip_does_not_retry_a_real_build_failure(monkeypatch):
    """A genuine 'no wheel / build failed' error is NOT retryable — it returns at once
    (one attempt), so a doomed install doesn't spin 3x."""
    from app import setup_installer
    monkeypatch.setattr(setup_installer.time, 'sleep', lambda s: None)
    attempts = {'n': 0}
    class FakeProc:
        returncode = 1
        stdout = iter(['ERROR: Could not build wheels for insightface\n'])
        def wait(self): return 1
    def fake_popen(cmd, **kw):
        attempts['n'] += 1
        return FakeProc()
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', fake_popen)
    setup_installer._runs['masks'] = setup_installer._new_run()
    rc = setup_installer._run_pip('masks', ['py', '-m', 'pip', 'install', 'x'])
    assert rc == 1 and attempts['n'] == 1


def test_manual_command_watermark_inpaint_points_to_managed_env_when_unconfigured(app):
    """No dedicated env -> the debug/diagnostic command must NOT target the app's own
    Python (that would break Pillow); it points at the app-managed venv the Install
    button auto-builds. (This string is a debug aid now — the button installs itself.)"""
    import sys
    from app import setup_installer
    with app.app_context():
        cmd = setup_installer.manual_command('watermark_inpaint')
    assert 'envs' in cmd and 'watermark' in cmd            # the app-managed venv path
    assert sys.executable not in cmd                        # never the app's own Python
    assert 'pip install' in cmd and 'simple-lama-inpainting' in cmd
    assert not cmd.startswith('pip ')


# --- "Install everything" orchestrator (plan / start_all / batched status) -------
# install_all_plan is a PURE function of live capabilities: the missing components the
# app can install itself (ML extras, the vision model when Ollama is up, the Klein
# weights when a valid ComfyUI is set), in a deterministic order. start_all fans out to
# start() per action, so it inherits the pip-FIFO serialization and preconditions.


def _caps(**over):
    """A fully-installed capabilities snapshot (everything present) — each test flips
    just the pieces it needs MISSING, so the plan reflects exactly that gap."""
    caps = {
        'python': {'ml_supported': True},
        'scrape_deps': True,
        'face_scoring': True, 'masks': True, 'watermark_inpaint': True,
        'ollama': {'reachable': True, 'vision_model_ready': True, 'vision_model': 'qwen3-vl:8b'},
        'comfyui': {'dir_valid': True, 'klein_missing': []},
    }
    caps.update(over)
    return caps


def test_install_all_plan_empty_when_everything_installed():
    """Idempotence: a machine with every installable component already in place yields an
    EMPTY plan — 'Install everything' has nothing to do and the UI shows the done state."""
    from app import setup_installer
    assert setup_installer.install_all_plan(_caps()) == []


def test_install_all_plan_none_and_empty_caps_are_safe():
    """None (couldn't probe) folds to {} — never raises. With nothing detected, the ML
    tiles read missing (default present=falsey) and Ollama/ComfyUI are absent so their
    gated actions are skipped; only the always-runnable extras remain."""
    from app import setup_installer
    ungated = ['scrape_extras', 'face_scoring', 'masks', 'watermark_inpaint']
    assert setup_installer.install_all_plan(None) == ungated
    assert setup_installer.install_all_plan({}) == ungated


def test_install_all_plan_lists_missing_ml_extras():
    from app import setup_installer
    caps = _caps(face_scoring=False, masks=False, watermark_inpaint=False)
    assert setup_installer.install_all_plan(caps) == ['face_scoring', 'masks', 'watermark_inpaint']


def test_install_all_plan_skips_face_masks_on_unsupported_python():
    """face_scoring/masks install into the app's own Python — outside the ML wheel range
    they'd only source-build and fail, so the plan omits them. watermark_inpaint builds
    its OWN 3.10-3.12 venv, so it stays in even on a 3.14 interpreter."""
    from app import setup_installer
    caps = _caps(python={'ml_supported': False},
                 face_scoring=False, masks=False, watermark_inpaint=False)
    assert setup_installer.install_all_plan(caps) == ['watermark_inpaint']
    # scrape_extras is pure-python, so the SAME unsupported interpreter still gets it.
    caps = _caps(python={'ml_supported': False}, scrape_deps=False,
                 face_scoring=False, masks=False, watermark_inpaint=False)
    assert setup_installer.install_all_plan(caps) == ['scrape_extras', 'watermark_inpaint']


def test_install_all_plan_includes_missing_scrape_extras():
    """Regression: scrape_extras had a worker, an action id and a UI button but no entry
    in _INSTALL_ALL_ORDER, so 'Install everything' could never repair the scraper stack —
    a user missing one package (instaloader) was told everything was already in place and
    kept hitting the runtime error. Present deps => absent from the plan (idempotent)."""
    from app import setup_installer
    assert setup_installer.install_all_plan(_caps(scrape_deps=False)) == ['scrape_extras']
    assert setup_installer.install_all_plan(_caps(scrape_deps=True)) == []


def test_install_all_never_pulls_an_ollama_model_implicitly():
    from app import setup_installer

    for ollama in (
        {'reachable': True, 'vision_model_ready': False, 'vision_model': 'qwen3-vl:8b'},
        {'reachable': True, 'vision_model_ready': False, 'vision_model': ''},
        {'reachable': False, 'vision_model_ready': False, 'vision_model': 'qwen3-vl:8b'},
    ):
        assert 'ollama_model' not in setup_installer.install_all_plan(_caps(ollama=ollama))


def test_install_all_plan_klein_only_into_valid_comfyui():
    from app import setup_installer
    # valid dir + assets missing -> the missing assets, in canonical order (LoRA last)
    caps = _caps(comfyui={'dir_valid': True,
                          'klein_missing': ['klein_lora', 'klein_model', 'klein_vae']})
    plan = setup_installer.install_all_plan(caps)
    assert plan == ['klein_model', 'klein_vae', 'klein_lora']
    # same gap but NO valid ComfyUI folder -> never scatter multi-GB files, skip all
    caps = _caps(comfyui={'dir_valid': False,
                          'klein_missing': ['klein_model', 'klein_vae']})
    assert setup_installer.install_all_plan(caps) == []


def test_install_all_plan_full_order():
    """Everything missing: unattended installs stay ML -> Klein. The large Ollama
    model remains an explicit action on the Ollama Setup card."""
    from app import setup_installer
    caps = _caps(
        scrape_deps=False,
        face_scoring=False, masks=False, watermark_inpaint=False,
        ollama={'reachable': True, 'vision_model_ready': False, 'vision_model': 'qwen3-vl:8b'},
        comfyui={'dir_valid': True,
                 'klein_missing': ['klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora']})
    assert setup_installer.install_all_plan(caps) == [
        'scrape_extras', 'face_scoring', 'masks', 'watermark_inpaint',
        'klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora']


def test_start_all_queues_each_planned_action(app, monkeypatch):
    """start_all fans out to start() per planned action. Two ML pip installs -> the first
    runs, the second is QUEUED behind it (the existing FIFO), proving install-all reuses
    the serialization instead of racing two pips into one venv."""
    from app import setup_installer
    monkeypatch.setattr(setup_installer, '_execute', lambda a: None)  # threads no-op
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()
    caps = _caps(face_scoring=False, masks=False)
    with app.app_context():
        res = setup_installer.start_all(caps)
    assert res['plan'] == ['face_scoring', 'masks']
    assert res['statuses']['face_scoring']['state'] == 'running'
    assert res['statuses']['masks']['state'] == 'queued'
    assert res['statuses']['masks']['waiting_for'] == 'face_scoring'
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()


def test_start_all_empty_plan_starts_nothing(app):
    from app import setup_installer
    with app.app_context():
        res = setup_installer.start_all(_caps())
    assert res == {'plan': [], 'statuses': {}}


def test_start_all_already_running_action_is_reused_not_fatal(app, monkeypatch):
    """An action already in flight (AlreadyRunning) is reported with its live state, not
    raised — install-all must never abort because one piece is already installing."""
    from app import setup_installer
    monkeypatch.setattr(setup_installer, '_execute', lambda a: None)
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()
    setup_installer._runs['face_scoring'] = {'state': 'running', 'returncode': None,
                                             'log': [], 'progress': None, 'waiting_for': None}
    setup_installer._pip_current = 'face_scoring'
    caps = _caps(face_scoring=False)
    with app.app_context():
        res = setup_installer.start_all(caps)
    assert res['plan'] == ['face_scoring']
    assert res['statuses']['face_scoring']['state'] == 'running'
    setup_installer._pip_current = None
    setup_installer._pip_queue.clear()


def test_status_many_returns_only_known_actions():
    from app import setup_installer
    out = setup_installer.status_many(['face_scoring', 'not_an_action', 'klein_vae'])
    assert set(out) == {'face_scoring', 'klein_vae'}
    assert out['face_scoring']['state'] == 'idle'


def test_run_ml_capability_pins_pillow_when_targeting_flask_venv(app, monkeypatch):
    """A scoped face_scoring/masks install that falls back to the Flask venv pins
    Pillow too, so pulling insightface/rembg deps can't downgrade it."""
    import sys
    from app import setup_installer, config
    seen = {}
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_popen_capturing(seen, 0))
    # The worker also runs two SHORT probes around pip (does this env already have
    # an onnxruntime? does the capability import once pip is done?), and this crude
    # global Popen fake would capture their command lines instead of the pip one.
    # They have their own tests — stub them out and keep this one about pip.
    monkeypatch.setattr(setup_installer, '_onnxruntime_provided', lambda p: False)
    monkeypatch.setattr(setup_installer, '_verify_capability_import', lambda a, p: True)
    with app.app_context():
        config.save_config({})   # no masks.python -> Flask venv
        setup_installer._runs['masks'] = setup_installer._new_run()
        setup_installer._run_ml_capability('masks')
    cmd = seen['cmd']
    assert cmd[0] == sys.executable
    assert any(str(p).lower().startswith('pillow==') for p in cmd)


# --- bank scoring: a BORROWED interpreter is never installed into --------------
# bank_scoring.python is what the ⚡ "use a GPU Python you already have" picker
# writes, and that dialog promises twice that borrowed environments "are checked,
# never changed". The Install / ↻ Reinstall button used to take that same value
# as its install TARGET — i.e. pip into the user's ai-toolkit or ComfyUI venv.

def test_run_bank_scoring_repairs_managed_while_preserving_borrowed_selection(
        app, monkeypatch, tmp_path):
    """Install is never an operation on the selected borrowed Python.  It repairs
    LDS's managed venv successfully and leaves the GPU runtime selection intact."""
    from app import setup_installer, config
    borrowed = tmp_path / 'ai-toolkit' / 'venv' / 'Scripts' / 'python.exe'
    borrowed.parent.mkdir(parents=True)
    borrowed.touch()
    seen = []
    monkeypatch.setattr(setup_installer, '_find_base_python',
                        lambda a: r'C:\pybase\python.exe')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen',
                        _fake_venv_popen(seen))
    with app.app_context():
        config.save_config({'bank_scoring': {'python': str(borrowed)}})
        managed = setup_installer._bank_scoring_env_python()
        setup_installer._runs['bank_scoring'] = setup_installer._new_run()
        rc = setup_installer._run_bank_scoring('bank_scoring')
        selected = config.get('bank_scoring.python')
    assert rc == 0
    assert selected == str(borrowed)
    pip_commands = [c for c in seen if c[1:4] == ['-m', 'pip', 'install']]
    assert pip_commands
    assert all(c[0] == managed for c in pip_commands)
    assert all(c[0] != str(borrowed) for c in pip_commands)
    log = setup_installer._runs['bank_scoring']['log']
    assert any('keeping the selected borrowed' in l for l in log)
    assert any(managed in l for l in log)


def test_bank_scoring_manual_command_never_points_at_borrowed_python(app):
    from app import config, setup_installer
    borrowed = r'C:\ComfyUI\python_embeded\python.exe'
    with app.app_context():
        config.save_config({'bank_scoring': {'python': borrowed}})
        command = setup_installer.manual_command('bank_scoring')
        managed = setup_installer._bank_scoring_env_python()
    assert managed in command
    assert borrowed not in command


def test_run_bank_scoring_still_installs_into_the_managed_venv(app, monkeypatch):
    """With nothing configured the button builds LDS's venv, installs there and
    selects it as the Score runtime."""
    from app import setup_installer, config
    seen = []
    monkeypatch.setattr(setup_installer, '_find_base_python', lambda a: r'C:\pybase\python.exe')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_venv_popen(seen))
    with app.app_context():
        config.save_config({})
        managed = setup_installer._bank_scoring_env_python()
        setup_installer._runs['bank_scoring'] = setup_installer._new_run()
        rc = setup_installer._run_bank_scoring('bank_scoring')
        saved = config.get('bank_scoring.python')
    assert rc == 0
    torch_cmd = next(c for c in seen if 'torch' in c)
    assert torch_cmd[0] == managed and setup_installer._TORCH_CPU_INDEX in torch_cmd
    clip_cmd = next(c for c in seen if any('open_clip' in str(p) for p in c))
    assert clip_cmd[0] == managed
    assert saved == managed


# --- Interpreter identity: environments, not binaries ----------------------------
# On Linux every venv's bin/python is a SYMLINK to the base interpreter, so
# comparing interpreters with os.path.samefile answers "same base Python?" — True
# for any two venvs on the machine. Inside the GPU Docker image that mistook the
# app-managed bank-scoring env for the Flask venv and refused the watermark-detector
# and shot-detection installs with advice that could not work (the config key the
# refusal says to clear was already empty). The comparators now compare the venv
# DIRECTORY that owns the binary (marked by pyvenv.cfg). These tests rebuild the
# collision with HARDLINKS so it is exercised on every OS, symlinks or not.

def _hardlinked_venv(tmp_path, name, base):
    """A venv-SHAPED directory whose python binary is a hardlink to `base` — every
    env built this way answers True to os.path.samefile against the others, which
    is exactly the Linux symlink collision, reproduced portably (NTFS included)."""
    import os
    bindir = 'Scripts' if os.name == 'nt' else 'bin'
    exe = 'python.exe' if os.name == 'nt' else 'python'
    env = tmp_path / name
    (env / bindir).mkdir(parents=True)
    (env / 'pyvenv.cfg').write_text('home = anywhere\n')
    py = env / bindir / exe
    os.link(base, py)
    return str(py)


def test_venv_comparators_tell_two_envs_apart_when_their_binaries_are_one_file(
        tmp_path, monkeypatch):
    import os
    from app import setup_installer
    base = tmp_path / 'python-base'
    base.write_bytes(b'')
    try:
        flask_py = _hardlinked_venv(tmp_path, 'flask-venv', base)
        managed_py = _hardlinked_venv(tmp_path, 'managed-env', base)
    except OSError:
        pytest.skip('hardlinks unavailable on this filesystem')
    assert os.path.samefile(flask_py, managed_py)         # the collision is real
    monkeypatch.setattr(setup_installer.sys, 'executable', flask_py)
    assert setup_installer._is_flask_venv(flask_py)       # itself: still True
    assert not setup_installer._is_flask_venv(managed_py)  # another env: never
    assert not setup_installer._same_path(flask_py, managed_py)
    # A venv is not the bare base interpreter it was built from either.
    assert not setup_installer._is_flask_venv(str(base))


def test_two_binary_names_inside_one_env_are_the_same_interpreter(tmp_path):
    """bin/python vs bin/python3 of ONE env compare equal — the identity is the
    environment, not the file name."""
    import os
    from app import setup_installer
    base = tmp_path / 'python-base'
    base.write_bytes(b'')
    try:
        py = _hardlinked_venv(tmp_path, 'env', base)
    except OSError:
        pytest.skip('hardlinks unavailable on this filesystem')
    sibling = os.path.join(os.path.dirname(py),
                           'python3.exe' if os.name == 'nt' else 'python3')
    os.link(str(base), sibling)
    assert setup_installer._same_path(py, sibling)


def test_same_path_still_matches_a_not_yet_built_venv_by_string(tmp_path):
    """The pre-build compare (neither path exists yet) keeps working: a configured
    value equal to the managed path must count as managed BEFORE the env exists."""
    from app import setup_installer
    ghost = str(tmp_path / 'envs' / 'bank_scoring' / 'bin' / 'python')
    assert setup_installer._same_path(ghost, ghost)
    assert not setup_installer._same_path(
        ghost, str(tmp_path / 'other' / 'bin' / 'python'))


# --- torch and torchvision install as a PAIR from the CPU index -------------------
# torch alone from the CPU index leaves torchvision to resolve later from PyPI as
# open_clip/timm/simple-lama's dependency — and on Linux that wheel is built against
# a DIFFERENT torch, so the env dies at import with 'operator torchvision::nms does
# not exist' (the Docker GPU report; Dockerfile.gpu pairs them for the same reason).

def test_bank_scoring_install_pairs_torchvision_with_torch(app, monkeypatch):
    from app import setup_installer, config
    seen = []
    monkeypatch.setattr(setup_installer, '_find_base_python', lambda a: r'C:\pybase\python.exe')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_venv_popen(seen))
    with app.app_context():
        config.save_config({})
        setup_installer._runs['bank_scoring'] = setup_installer._new_run()
        rc = setup_installer._run_bank_scoring('bank_scoring')
    assert rc == 0
    torch_cmd = next(c for c in seen if 'torch' in c)
    assert 'torchvision' in torch_cmd
    assert setup_installer._TORCH_CPU_INDEX in torch_cmd


def test_manual_commands_pair_torchvision_with_torch_everywhere(app):
    """Every copy-paste command that puts torch into the managed env names
    torchvision on the same index line — the command documents what the Install
    button itself does, so the two must not drift."""
    from app import setup_installer
    with app.app_context():
        for action in ('bank_scoring', 'shot_detect', 'bank_siglip2',
                       'watermark_detect'):
            cmd = setup_installer.manual_command(action)
            assert 'torch torchvision --index-url' in cmd, action


# --- The transformers floor the video-caption worker depends on ------------------
# infer/video_caption_infer.py:103 imports Qwen3VLForConditionalGeneration, a class
# transformers only grew in 4.57.0 (huggingface/transformers PR #40795, shipped in
# the v4.57.0 release; the Qwen/Qwen3-VL-4B-Instruct card says `transformers==4.57.0`).
# Installing the bank-scoring step with a BARE `pip install transformers` is a no-op
# on a machine that already carries an older transformers ("already satisfied"), so
# the caption worker dies on ImportError and re-clicking the step never repairs it.
# Carrying the floor in the pip command is what turns that click into a repair.

def test_requirements_ml_floors_transformers_for_qwen3vl():
    """The floor lives in requirements-ml.txt (one place for every version), and it
    is at least 4.57 — below that `Qwen3VLForConditionalGeneration` does not exist."""
    from app import setup_installer
    spec = setup_installer._requirement_spec('transformers')
    assert spec != 'transformers', 'no floor in requirements-ml.txt — bare name fallback'
    floor = spec.split('>=', 1)[1].split(',')[0].strip()
    major, minor = (int(p) for p in floor.split('.')[:2])
    assert (major, minor) >= (4, 57), f'floor {floor} predates Qwen3-VL support'


def test_bank_scoring_manual_command_carries_the_transformers_floor(app):
    """The copy-paste command shows the floor, QUOTED — an unquoted '>=' is shell
    redirection and would write a file named '=4.57' instead of installing."""
    from app import setup_installer
    with app.app_context():
        cmd = setup_installer.manual_command('bank_scoring')
        spec = setup_installer._requirement_spec('transformers')
    assert f'"{spec}"' in cmd
    assert ' transformers ' not in cmd      # never the bare, no-op name


def test_run_bank_scoring_installs_the_floored_transformers(app, monkeypatch):
    """The REAL pip call carries the floor too, so re-clicking the step UPGRADES an
    old transformers instead of reporting 'already satisfied' and changing nothing."""
    from app import setup_installer, config
    seen = []
    monkeypatch.setattr(setup_installer, '_find_base_python', lambda a: r'C:\pybase\python.exe')
    monkeypatch.setattr(setup_installer.subprocess, 'Popen', _fake_venv_popen(seen))
    with app.app_context():
        config.save_config({})
        setup_installer._runs['bank_scoring'] = setup_installer._new_run()
        rc = setup_installer._run_bank_scoring('bank_scoring')
        spec = setup_installer._requirement_spec('transformers')
    assert rc == 0
    pkg_cmd = next(c for c in seen if any('open_clip' in str(p) for p in c))
    assert spec in pkg_cmd
    assert 'transformers' not in pkg_cmd     # the bare name must be GONE, not added to


def test_bank_scoring_verification_ignores_user_site(app, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app import setup_installer
    python = tmp_path / 'managed-python.exe'
    python.touch()
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stderr='')

    monkeypatch.setattr(setup_installer.subprocess, 'run', run)
    with app.app_context():
        setup_installer._runs['bank_scoring'] = setup_installer._new_run()
        assert setup_installer._verify_bank_scoring_import(
            'bank_scoring', str(python)) is True
    assert seen[0][:3] == [str(python), '-s', '-c']


def test_bank_siglip2_is_explicit_scoped_action(app):
    from app import config, setup_installer
    borrowed = r'C:\borrowed score\python.exe'
    assert 'bank_siglip2' in setup_installer.INSTALL_ACTIONS
    assert 'bank_siglip2' in setup_installer._WORKERS
    assert 'bank_siglip2' in setup_installer._PIP_ACTIONS
    assert 'bank_siglip2' not in setup_installer._INSTALL_ALL_ORDER
    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': borrowed},
            'bank_semantic': {'python': ''},
        })
        command = setup_installer.manual_command('bank_siglip2')
        managed = setup_installer._bank_scoring_env_python()
    assert managed in command
    assert borrowed not in command
    assert 'pip install torch' in command
    assert setup_installer._TORCH_CPU_INDEX in command
    assert 'transformers>=4.49' in command
    assert 'Pillow' in command
    assert 'hf_hub_download' in command


def test_bank_siglip2_manual_command_never_targets_a_borrowed_interpreter(app):
    """WHERE THE INDEX RUNS is not where we install.

    ``bank_semantic.python`` is a picker now: it can hold someone's ai-toolkit or
    ComfyUI venv, chosen so the index runs on their GPU. A repair line that read
    it would hand the user a command that pip-installs into that environment —
    the one thing this app never does. The managed venv is the only install
    target, whatever either interpreter key says."""
    from app import config, setup_installer
    borrowed_score = r'D:\borrowed score\python.exe'
    borrowed_semantic = r'D:\borrowed semantic\python.exe'
    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': borrowed_score},
            'bank_semantic': {'python': borrowed_semantic},
        })
        command = setup_installer.manual_command('bank_siglip2')
        managed = setup_installer._bank_scoring_env_python()
    assert managed in command
    assert borrowed_semantic not in command
    assert borrowed_score not in command


def test_bank_siglip2_install_ignores_borrowed_semantic_interpreter(
        app, monkeypatch, tmp_path):
    """The core guarantee of the semantic GPU picker, proved end to end.

    A user points the SigLIP 2 index at their ai-toolkit venv (execution), then
    hits Install/repair in Setup (installation). Every subprocess must run in the
    app-managed venv, none of them may name the borrowed path, and the borrowed
    choice must survive: overwriting it would silently drag the index back onto
    the CPU right after a repair."""
    from pathlib import Path
    from app import config, setup_installer
    from app.services import bank_semantic_models as assets
    borrowed = tmp_path / 'ai-toolkit' / 'venv' / 'Scripts' / 'python.exe'
    borrowed.parent.mkdir(parents=True)
    borrowed.touch()
    calls = []

    monkeypatch.setattr(
        setup_installer, '_run_pip',
        lambda action, command: calls.append(tuple(command)) or 0)
    monkeypatch.setattr(setup_installer, '_verify_capability_import',
                        lambda action, python: True)
    monkeypatch.setattr(assets, 'weights_present', lambda root=None: True)
    with app.app_context():
        config.save_config({'bank_semantic': {'python': str(borrowed)}})
        managed = setup_installer._bank_scoring_env_python()
        Path(managed).parent.mkdir(parents=True, exist_ok=True)
        Path(managed).touch()
        setup_installer._runs['bank_siglip2'] = setup_installer._new_run()
        assert setup_installer._run_bank_siglip2('bank_siglip2') == 0
        # Execution choice preserved — NOT repointed at the managed venv.
        assert config.get('bank_semantic.python') == str(borrowed)
    assert calls, 'the install must actually have run'
    assert all(command[0] == managed for command in calls)
    assert not any(str(borrowed) in str(part)
                   for command in calls for part in command)
    log = ' '.join(setup_installer._runs['bank_siglip2']['log'])
    assert 'LDS-managed environment' in log


def test_run_bank_siglip2_keeps_borrowed_score_and_installs_only_managed(
        app, monkeypatch, tmp_path):
    from pathlib import Path
    from app import config, setup_installer
    from app.services import bank_semantic_models as assets
    borrowed = tmp_path / 'other-tool' / 'Scripts' / 'python.exe'
    borrowed.parent.mkdir(parents=True)
    borrowed.touch()
    calls = []

    monkeypatch.setattr(
        setup_installer, '_run_pip',
        lambda action, command: calls.append(tuple(command)) or 0)
    monkeypatch.setattr(setup_installer, '_verify_capability_import',
                        lambda action, python: True)
    monkeypatch.setattr(assets, 'weights_present', lambda root=None: True)
    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': str(borrowed)},
            'bank_semantic': {'python': ''},
        })
        managed = setup_installer._bank_scoring_env_python()
        Path(managed).parent.mkdir(parents=True, exist_ok=True)
        Path(managed).touch()
        setup_installer._runs['bank_siglip2'] = setup_installer._new_run()
        assert setup_installer._run_bank_siglip2('bank_siglip2') == 0
        assert config.get('bank_scoring.python') == str(borrowed)
        assert config.get('bank_semantic.python') == managed
    assert len(calls) == 3
    assert all(command[0] == managed for command in calls)
    assert all(command[0] != str(borrowed) for command in calls)


def test_run_bank_siglip2_downloads_only_pinned_files(app, monkeypatch):
    from app import config, setup_installer
    from app.services import bank_semantic_models as assets
    calls = []
    monkeypatch.setattr(
        setup_installer, '_ensure_bank_scoring_env',
        lambda action, *, save_score_python=True: (
            setup_installer._bank_scoring_env_python()
            if save_score_python is False else
            (_ for _ in ()).throw(AssertionError('must not select Score'))))
    monkeypatch.setattr(
        setup_installer, '_run_pip',
        lambda action, command: calls.append(command) or 0)
    monkeypatch.setattr(setup_installer, '_verify_capability_import',
                        lambda action, python: True)
    monkeypatch.setattr(assets, 'weights_present', lambda root=None: True)
    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': ''},
            'bank_semantic': {'python': ''},
        })
        setup_installer._runs['bank_siglip2'] = setup_installer._new_run()
        assert setup_installer._run_bank_siglip2('bank_siglip2') == 0
        assert config.get('bank_semantic.python') == (
            setup_installer._bank_scoring_env_python())
    packages = next(command for command in calls
                    if 'transformers>=4.49' in command)
    assert 'Pillow' in packages
    download = next(command for command in calls if '-c' in command)
    payload = download[-1]
    for filename in assets.FILES:
        assert filename in payload
    assert assets.REVISION in payload


def test_run_bank_siglip2_reports_failure_when_semantic_python_cannot_be_saved(
        app, monkeypatch):
    from app import config, setup_installer
    from app.services import bank_semantic_models as assets
    calls = []
    managed = setup_installer._bank_scoring_env_python()
    monkeypatch.setattr(
        setup_installer, '_ensure_bank_scoring_env',
        lambda action, *, save_score_python=True: managed)
    monkeypatch.setattr(
        setup_installer, '_run_pip',
        lambda action, command: calls.append(tuple(command)) or 0)
    monkeypatch.setattr(setup_installer, '_verify_capability_import',
                        lambda action, python: True)
    monkeypatch.setattr(assets, 'weights_present', lambda root=None: True)

    def refuse_save(partial):
        assert partial == {'bank_semantic': {'python': managed}}
        raise OSError('config is read-only')

    with app.app_context():
        config.save_config({'bank_semantic': {'python': ''}})
        monkeypatch.setattr(setup_installer.cfg, 'save_config', refuse_save)
        setup_installer._runs['bank_siglip2'] = setup_installer._new_run()
        assert setup_installer._run_bank_siglip2('bank_siglip2') == 1
    assert len(calls) == 3, 'save is attempted only after packages and weights'
    log = setup_installer._runs['bank_siglip2']['log']
    assert any('could not be saved' in line for line in log)
    assert not any(line.startswith('SigLIP2 ready') for line in log)


# --- The decoder the shot-detect worker dies without ------------------------------
# infer/shot_detect_infer.py decodes with PyAV in the SAME environment as the
# model — `import av` runs before torch ever sees a frame. The install shipped
# torch + transnetv2 without av, and the capability probe did not import av
# either: install green, probe green, and then EVERY file of the first real
# bank answered "failed shot detection" (ModuleNotFoundError: No module named
# 'av', 246/246 files, found live the day the wave landed). These two tests
# pin the whole chain: what the worker imports, the installer installs and the
# probe checks.

def test_shot_detect_install_carries_the_decoder_its_worker_dies_without(
        app, monkeypatch, tmp_path):
    from app import setup_installer
    calls = []
    monkeypatch.setattr(setup_installer, '_run_pip',
                        lambda a, cmd: (calls.append(list(cmd)), 0)[1])
    monkeypatch.setattr(setup_installer, '_verify_shot_detect_import',
                        lambda a, p: True)
    managed = str(tmp_path / 'envs' / 'bank_scoring' / 'Scripts' / 'python.exe')
    monkeypatch.setattr(setup_installer, '_bank_scoring_env_python', lambda: managed)
    monkeypatch.setattr(setup_installer, '_ensure_bank_scoring_env',
                        lambda a, **k: managed)
    saved = {}
    monkeypatch.setattr(setup_installer.cfg, 'save_config',
                        lambda p: saved.update(p))
    with app.app_context():
        setup_installer._runs['shot_detect'] = setup_installer._new_run()
        rc = setup_installer._run_shot_detect('shot_detect')
    assert rc == 0
    flat = [arg for cmd in calls for arg in cmd]
    assert any(a == 'av' or a.startswith('av>') or a.startswith('av=')
               for a in flat), f'no av spec in the pip calls: {flat}'


def test_shot_detect_capability_probe_imports_what_the_worker_imports():
    """The probe and the worker must agree on the environment's contents; a
    probe that skips av says "ready" about a worker that cannot open one file."""
    from app import capabilities
    assert 'av' in capabilities.CAPABILITY_IMPORTS['shot_detect']
