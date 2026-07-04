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

def test_capabilities_endpoint(client):
    caps = client.get('/api/capabilities').get_json()
    assert 'engines' in caps and 'studio_visible' in caps

def test_test_connection_unknown_target(client):
    assert client.post('/api/settings/test/nope').status_code == 404
