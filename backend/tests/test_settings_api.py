import pathlib
import pytest


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """GET /api/capabilities calls probe(), which hits every reachability
    probe. Stub the network/subprocess seams so this file never makes a
    real call, mirroring test_capabilities.py's isolation."""
    from app import capabilities
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    capabilities._import_cache.clear()
    monkeypatch.setattr(capabilities, '_http_ok', lambda *a, **k: False)
    monkeypatch.setattr(capabilities, '_import_ok', lambda *a, **k: False)
    yield
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    capabilities._import_cache.clear()


def test_get_settings_masks_secrets(client, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')
    data = client.get('/api/settings').get_json()
    assert data['secrets']['OPENAI_API_KEY'] is True
    assert 'sk-secret' not in str(data)

def test_put_settings_persists_config_and_secret(client, tmp_path):
    r = client.put('/api/settings', json={
        'config': {'ollama': {'url': 'http://127.0.0.1:11500'}},
        'secrets': {'GEMINI_API_KEY': 'g-123'}})
    assert r.status_code == 200
    assert r.get_json()['config']['ollama']['url'] == 'http://127.0.0.1:11500'
    assert r.get_json()['secrets']['GEMINI_API_KEY'] is True

def test_put_rejects_unknown_section(client):
    assert client.put('/api/settings', json={'config': {'nope': 1}}).status_code == 400

def test_put_rejects_non_object_section_value(client):
    """{"config": {"ollama": "x"}} would otherwise pass the top-level key-name
    check and let _deep_merge overwrite the whole 'ollama' section with a
    string, persistently corrupting config.json. Must be rejected AND leave
    the existing config untouched."""
    before = client.get('/api/settings').get_json()['config']['ollama']
    r = client.put('/api/settings', json={'config': {'ollama': 'x'}})
    assert r.status_code == 400
    after = client.get('/api/settings').get_json()['config']['ollama']
    assert after == before
    assert isinstance(after, dict) and 'url' in after

def test_put_rejects_non_object_config(client):
    assert client.put('/api/settings', json={'config': 'oops'}).status_code == 400

def test_put_rejects_non_object_secrets(client):
    assert client.put('/api/settings', json={'secrets': ['x']}).status_code == 400

def test_put_settings_autocorrects_portable_base_dir(client, tmp_path):
    """Saving a base_dir that points at the portable WRAPPER
    (...\\ComfyUI_windows_portable) must be rewritten to the nested ...\\ComfyUI
    that actually holds main.py + models/ -- otherwise every model lister scans an
    empty wrapper\\models and reports 'No checkpoint found' even though ComfyUI runs."""
    wrapper = tmp_path / 'ComfyUI_windows_portable'
    inner = wrapper / 'ComfyUI'
    inner.mkdir(parents=True)
    (inner / 'main.py').touch()
    (inner / 'models').mkdir()
    r = client.put('/api/settings', json={'config': {'comfyui': {'base_dir': str(wrapper)}}})
    assert r.status_code == 200
    saved = r.get_json()['config']['comfyui']['base_dir']
    assert pathlib.Path(saved) == inner   # auto-corrected down into the real install

def test_put_settings_keeps_valid_base_dir_unchanged(client, tmp_path):
    """A base_dir already pointing straight at a real ComfyUI install is left as-is."""
    base = tmp_path / 'Comfy'
    base.mkdir()
    (base / 'main.py').touch()
    (base / 'models').mkdir()
    r = client.put('/api/settings', json={'config': {'comfyui': {'base_dir': str(base)}}})
    assert pathlib.Path(r.get_json()['config']['comfyui']['base_dir']) == base

def test_capabilities_endpoint(client):
    caps = client.get('/api/capabilities').get_json()
    assert 'engines' in caps and 'studio_visible' in caps

def test_test_connection_unknown_target(client):
    assert client.post('/api/settings/test/nope').status_code == 404


def test_logs_tail_reads_app_log(client, tmp_path, monkeypatch):
    import os
    data_dir = os.environ['LDS_DATA_DIR']    # tmp dir set by the app fixture
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'app.log'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(f'line {i}' for i in range(500)) + '\n')
    d = client.get('/api/logs/tail?n=100').get_json()
    assert d['ok'] is True and d['file'] == 'app.log'
    assert len(d['lines']) == 100 and d['lines'][-1] == 'line 499'


def test_logs_tail_empty_when_no_log(client):
    d = client.get('/api/logs/tail').get_json()
    assert d == {'ok': True, 'file': None, 'lines': []}
