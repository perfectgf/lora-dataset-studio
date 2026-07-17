"""GET /api/diagnostic — paste-safe bug-report payload.

The report is designed to be pasted as-is into a public issue or Discord
thread: secret VALUES must never appear (presence booleans only) and paths
are reduced to booleans.
"""
import json


def test_diagnostic_ok_and_shape(client):
    r = client.get('/api/diagnostic')
    assert r.status_code == 200
    j = r.get_json()
    assert j['app_version']
    assert isinstance(j['secrets_present'], dict)
    assert isinstance(j['capabilities'], dict)
    assert isinstance(j['capabilities']['engines'], dict)
    assert isinstance(j['config'], dict)
    assert isinstance(j['log_tail'], list)
    assert isinstance(j['generated_at'], int)


def test_diagnostic_never_leaks_secret_values(client, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'sk-SUPERSECRET-42')
    r = client.get('/api/diagnostic')
    body = r.get_data(as_text=True)
    assert 'sk-SUPERSECRET-42' not in body
    assert r.get_json()['secrets_present']['GEMINI_API_KEY'] is True


def test_diagnostic_has_no_absolute_paths(client):
    """Paths identify the machine/user — the payload carries *_set booleans
    instead. The log-derived sections are excluded: their lines may cite file
    names, which is why the UI warns to skim the report before posting."""
    j = client.get('/api/diagnostic').get_json()
    dumped = json.dumps({k: v for k, v in j.items()
                         if k not in ('log_tail', 'error_log')})
    assert ':\\\\' not in dumped and ':/' not in dumped


def test_diagnostic_includes_log_tail(client, app, tmp_path, monkeypatch):
    import os
    from pathlib import Path
    data_dir = Path(os.environ['LDS_DATA_DIR'])
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'app.log').write_text('line one\nline two\n', encoding='utf-8')
    j = client.get('/api/diagnostic').get_json()
    assert j['log_tail'][-1] == 'line two'


def test_diagnostic_exposes_ollama_vision_model_and_tags(client, app, monkeypatch):
    """The report carries the configured vision-model string AND the tags Ollama
    actually reports, so a 'vision_model=no' report can be triaged without a round
    trip: a truly-missing model vs one listed under a different identifier (issue #7)."""
    from app import capabilities, config
    with app.app_context():
        config.save_config({'ollama': {'url': 'http://o',
                                       'vision_model': 'huihui_ai/qwen3-vl-abliterated:8b-instruct'}})
    monkeypatch.setattr(capabilities, '_ollama_tags',
                        lambda *a, **k: ['gemma4:e2b-it-q4_K_M', 'qwen3-vl:8b-instruct',
                                         'huihui_ai/qwen3-vl-abliterated:8b-instruct'])
    j = client.get('/api/diagnostic').get_json()
    assert j['ollama']['vision_model'] == 'huihui_ai/qwen3-vl-abliterated:8b-instruct'
    assert 'huihui_ai/qwen3-vl-abliterated:8b-instruct' in j['ollama']['tags_seen']


def test_diagnostic_new_sections_shape(client):
    """The enriched payload ships the environment-health, per-engine, live-runtime,
    generation-error and error-log sections the three support cases needed."""
    j = client.get('/api/diagnostic').get_json()
    assert isinstance(j['python_ml'], dict) and 'ml_supported' in j['python_ml']
    assert isinstance(j['pillow'], dict) and 'healthy' in j['pillow']
    assert isinstance(j['disk'], dict)
    assert isinstance(j['comfyui_runtime'], dict)
    assert isinstance(j['generation_errors'], dict)
    assert isinstance(j['error_log'], list)
    caps = j['capabilities']
    assert isinstance(caps['klein_missing'], list)
    assert isinstance(caps['chatgpt_subscription'], bool)
    assert 'watermark_allow_crop' in j['config']


def test_error_log_reassembles_full_traceback(client, app):
    """The error section carries whole ERROR records — message AND every traceback
    frame — not the plain last-N-lines tail that cuts a stack in half. Surrounding
    INFO records are NOT dragged into it."""
    import os
    from pathlib import Path
    data_dir = Path(os.environ['LDS_DATA_DIR'])
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'app.log').write_text(
        '2026-07-17 12:00:00,000 INFO app.boot: started\n'
        '2026-07-17 12:00:01,000 ERROR app.routes.face: generation failed\n'
        'Traceback (most recent call last):\n'
        '  File "app/services/x.py", line 10, in run\n'
        '    raise ValueError("boom")\n'
        'ValueError: boom\n'
        '2026-07-17 12:00:02,000 INFO app.x: moved on\n',
        encoding='utf-8')
    el = '\n'.join(client.get('/api/diagnostic').get_json()['error_log'])
    assert 'ERROR app.routes.face: generation failed' in el
    assert 'Traceback (most recent call last):' in el
    assert 'ValueError: boom' in el
    assert 'app.boot: started' not in el      # the leading INFO record is not pulled in
    assert 'app.x: moved on' not in el        # nor the trailing one


def test_error_log_redacts_home_paths(client, app):
    """A traceback frame legitimately cites an absolute path — the home dir (OS
    account name) must be redacted to ~ before the report is pasted publicly."""
    import os
    from pathlib import Path
    data_dir = Path(os.environ['LDS_DATA_DIR'])
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'app.log').write_text(
        '2026-07-17 12:00:01,000 ERROR app.x: fail\n'
        '  File "C:\\Users\\somebody\\lds\\x.py", line 1, in run\n'
        'RuntimeError: nope\n', encoding='utf-8')
    body = client.get('/api/diagnostic').get_data(as_text=True)
    assert 'somebody' not in body


def test_generation_errors_report_most_recent_per_engine(client, app):
    """The last failed-generation reason PER engine (klein/chatgpt/nanobanana),
    newest wins — the 'it fails at every generation' cause instead of a guess."""
    from app.extensions import db
    from app.models import FaceDataset, FaceDatasetImage
    with app.app_context():
        ds = FaceDataset(name='n', trigger_word='t')
        db.session.add(ds)
        db.session.commit()
        for reason in ('klein: an older klein failure',
                       'klein: ComfyUI 409 klein_vae missing',    # newest klein
                       'chatgpt: 429 quota exceeded',
                       'nanobanana: 400 content policy'):
            db.session.add(FaceDatasetImage(dataset_id=ds.id, status='failed',
                                            fail_reason=reason))
        db.session.commit()
    ge = client.get('/api/diagnostic').get_json()['generation_errors']['engines']
    assert ge['klein'] == 'klein: ComfyUI 409 klein_vae missing'
    assert ge['chatgpt'] == 'chatgpt: 429 quota exceeded'
    assert ge['nanobanana'] == 'nanobanana: 400 content policy'


def test_generation_errors_redact_paths_and_studio_failure(client, app):
    """fail_reason / Studio error are path-redacted (no OS account name) and the last
    Studio failure is surfaced separately."""
    from app.extensions import db
    from app.models import FaceDataset, FaceDatasetImage, LoraTestImage
    with app.app_context():
        ds = FaceDataset(name='n', trigger_word='t')
        db.session.add(ds)
        db.session.commit()
        db.session.add(FaceDatasetImage(
            dataset_id=ds.id, status='failed',
            fail_reason='klein: saving the image failed at C:\\Users\\secretuser\\ds\\a.png'))
        db.session.add(LoraTestImage(
            dataset_id=ds.id, checkpoint='c', strength=1.0, status='failed',
            error='node error under /home/secretuser/comfy'))
        db.session.commit()
    r = client.get('/api/diagnostic')
    assert 'secretuser' not in r.get_data(as_text=True)
    ge = r.get_json()['generation_errors']
    assert ge['engines']['klein'].startswith('klein: saving the image failed at ~')
    assert ge['studio'] == 'node error under ~/comfy'


def test_generation_errors_never_leak_dataset_prompts(client, app):
    """A dataset's prompt lives in a SEPARATE column (variation_prompt) — only the
    engine error text (fail_reason) is ever exposed, never the prompt."""
    from app.extensions import db
    from app.models import FaceDataset, FaceDatasetImage
    with app.app_context():
        ds = FaceDataset(name='n', trigger_word='t')
        db.session.add(ds)
        db.session.commit()
        db.session.add(FaceDatasetImage(
            dataset_id=ds.id, status='failed', fail_reason='klein: 409 missing vae',
            variation_prompt='a very private SECRET-PROMPT-XYZ description'))
        db.session.commit()
    assert 'SECRET-PROMPT-XYZ' not in client.get('/api/diagnostic').get_data(as_text=True)


def test_pillow_health_reports_mixed_and_healthy(client, monkeypatch):
    """The self-heal verdict rides the same file-inspection check the boot repair
    uses: incompatible plugins -> MIXED, else healthy."""
    import bootstrap_dependencies
    from pathlib import Path
    monkeypatch.setattr(bootstrap_dependencies, 'incompatible_pillow_plugins',
                        lambda *a, **k: ('12.2.0', [Path('PngImagePlugin.py')]))
    assert client.get('/api/diagnostic').get_json()['pillow'] == {
        'version': '12.2.0', 'healthy': False, 'incompatible_plugins': ['PngImagePlugin.py']}
    monkeypatch.setattr(bootstrap_dependencies, 'incompatible_pillow_plugins',
                        lambda *a, **k: ('12.2.0', []))
    healthy = client.get('/api/diagnostic').get_json()['pillow']
    assert healthy['healthy'] is True and healthy['version'] == '12.2.0'


def test_comfyui_runtime_included_when_reachable(client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'comfyui_runtime',
                        lambda *a, **k: {'version': '0.3.30', 'gpu': 'RTX 4090',
                                         'vram_total_gb': 24.0, 'queue_running': 0,
                                         'queue_pending': 1})
    rt = client.get('/api/diagnostic').get_json()['comfyui_runtime']
    assert rt['version'] == '0.3.30' and rt['queue_pending'] == 1


def test_comfyui_runtime_empty_without_api_url(app):
    """No api_url -> no network, {} (never raises)."""
    from app import capabilities, config
    with app.app_context():
        config.save_config({'comfyui': {'api_url': ''}})
        assert capabilities.comfyui_runtime() == {}


def test_comfyui_runtime_parses_system_stats_and_queue(app, monkeypatch):
    from app import capabilities, config

    class _Resp:
        def __init__(self, data):
            self._d, self.status_code = data, 200

        def json(self):
            return self._d

    def fake_get(url, timeout=3):
        if url.endswith('/system_stats'):
            return _Resp({'system': {'comfyui_version': '0.3.30'},
                          'devices': [{'name': 'cuda:0 NVIDIA RTX 4090',
                                       'vram_total': 24 * 1024 ** 3,
                                       'vram_free': 12 * 1024 ** 3}]})
        if url.endswith('/queue'):
            return _Resp({'queue_running': [1], 'queue_pending': [1, 2]})
        raise AssertionError(url)

    with app.app_context():
        config.save_config({'comfyui': {'api_url': 'http://comfy'}})
        monkeypatch.setattr(capabilities.requests, 'get', fake_get)
        rt = capabilities.comfyui_runtime()
    assert rt['version'] == '0.3.30'
    assert rt['gpu'].startswith('cuda:0 NVIDIA')
    assert rt['vram_total_gb'] == 24.0 and rt['vram_free_gb'] == 12.0
    assert rt['queue_running'] == 1 and rt['queue_pending'] == 2


def test_diagnostic_redacts_user_paths_in_log_tail(client, app):
    """The log tail can legitimately cite absolute paths (e.g. "Checkpoint
    model directories not found: C:\\Users\\somebody\\ComfyUI\\models") — but
    this payload is pasted into a PUBLIC issue/Discord thread, so the Windows
    account / Unix username must be redacted to `~` (Windows single- AND
    double-backslash, plus the POSIX /home and /Users forms)."""
    import os
    from pathlib import Path
    data_dir = Path(os.environ['LDS_DATA_DIR'])
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / 'app.log').write_text(
        'Checkpoint model directories not found: C:\\Users\\somebody\\ComfyUI\\models\n'
        'escaped form: C:\\\\Users\\\\somebody\\\\ComfyUI\\models\n'
        'posix home: /home/somebody/x\n'
        'posix mac: /Users/somebody/x\n',
        encoding='utf-8')
    r = client.get('/api/diagnostic')
    body = r.get_data(as_text=True)
    assert 'somebody' not in body
    log_tail = r.get_json()['log_tail']
    assert all('~' in line for line in log_tail)
