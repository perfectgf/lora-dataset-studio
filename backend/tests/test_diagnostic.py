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
    instead. The log tail is excluded: log lines may cite file names, which is
    why the UI warns to skim the report before posting."""
    j = client.get('/api/diagnostic').get_json()
    dumped = json.dumps({k: v for k, v in j.items() if k != 'log_tail'})
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
