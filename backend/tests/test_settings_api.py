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


@pytest.fixture()
def _reset_update_cache():
    from app.routes import settings as sroutes
    sroutes._update_cache.update(ts=0.0, data=None)
    sroutes._git_check_cache.update(ts=0.0, data=None)
    yield
    sroutes._update_cache.update(ts=0.0, data=None)
    sroutes._git_check_cache.update(ts=0.0, data=None)


class _FakeResp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body or {}
    def json(self):
        return self._body


def test_update_check_detects_newer_release(client, monkeypatch, _reset_update_cache):
    import requests
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(200, {
        'tag_name': 'v9999.12.31', 'html_url': 'https://github.com/x/releases/tag/v9999.12.31'}))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is True and d['latest'] == '9999.12.31'
    assert d['url'].endswith('v9999.12.31')


def test_update_check_same_version_and_cache(client, monkeypatch, _reset_update_cache):
    import requests
    from app.version import APP_VERSION
    calls = []
    monkeypatch.setattr(requests, 'get',
                        lambda *a, **k: calls.append(1) or _FakeResp(200, {'tag_name': f'v{APP_VERSION}'}))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is False and d['latest'] == APP_VERSION
    client.get('/api/update/check')          # second call served from the 6h cache
    assert len(calls) == 1


def test_update_check_auto_fetches_git_then_serves_cache(client, monkeypatch, _reset_update_cache):
    """auto=1 (nav badge): the git-aware check RUNS (unlike the bare passive
    path) but is served from a TTL cache — SPA loads cost one fetch per 6 h."""
    from app.services import updater
    calls = []
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: True)
    monkeypatch.setattr(updater, 'git_update_status',
                        lambda root=None: calls.append(1) or {
                            'ok': True, 'is_git': True, 'update_available': True,
                            'behind': 2, 'current': '1.0'})
    d = client.get('/api/update/check?auto=1').get_json()
    assert d['update_available'] is True and d['behind'] == 2
    client.get('/api/update/check?auto=1')       # second auto call -> cache
    assert len(calls) == 1
    # a manual force check is always fresh AND refreshes the cache
    client.get('/api/update/check?force=1')
    assert len(calls) == 2


def test_update_check_bare_passive_never_fetches_git(client, monkeypatch, _reset_update_cache):
    """The bare passive path (no force, no auto) must not run the git check."""
    import requests
    from app.services import updater
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: True)
    monkeypatch.setattr(updater, 'git_update_status',
                        lambda root=None: (_ for _ in ()).throw(
                            AssertionError('git check must not run')))
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(404))
    d = client.get('/api/update/check').get_json()
    assert d['ok'] is True


def test_update_check_degrades_when_feed_unreachable(client, monkeypatch, _reset_update_cache):
    import requests
    def boom(*a, **k):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(requests, 'get', boom)
    d = client.get('/api/update/check').get_json()
    assert d['ok'] is True and d['update_available'] is False
    assert 'unreachable' in d['reason']


def test_update_check_private_repo_404(client, monkeypatch, _reset_update_cache):
    import requests
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(404))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is False and '404' in d['reason']


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


def test_chatgpt_oauth_routes(client, monkeypatch):
    from unittest.mock import patch
    from app.services import chatgpt_oauth
    with patch.object(chatgpt_oauth, 'login_start',
                      return_value={'ok': True, 'verification_url': 'https://x/device',
                                    'user_code': 'AB-12'}):
        r = client.post('/api/settings/chatgpt-oauth/start')
        assert r.status_code == 200 and r.get_json()['user_code'] == 'AB-12'
    with patch.object(chatgpt_oauth, 'login_start',
                      return_value={'ok': False, 'detail': 'network error'}):
        assert client.post('/api/settings/chatgpt-oauth/start').status_code == 502
    with patch.object(chatgpt_oauth, 'login_poll', return_value={'status': 'pending',
                                                                 'detail': None}):
        r = client.get('/api/settings/chatgpt-oauth/poll')
        assert r.status_code == 200 and r.get_json()['status'] == 'pending'
    with patch.object(chatgpt_oauth, 'import_codex_cli',
                      return_value={'ok': False, 'detail': 'no session'}):
        assert client.post('/api/settings/chatgpt-oauth/import-codex').status_code == 404
    with patch.object(chatgpt_oauth, 'import_codex_cli',
                      return_value={'ok': True, 'detail': 'imported'}):
        assert client.post('/api/settings/chatgpt-oauth/import-codex').status_code == 200
    r = client.post('/api/settings/chatgpt-oauth/logout')
    assert r.status_code == 200 and r.get_json()['ok'] is True


def test_put_settings_saves_chatgpt_auth_mode(client):
    r = client.put('/api/settings', json={'config': {'engines': {'chatgpt_auth': 'subscription'}}})
    assert r.status_code == 200
    assert r.get_json()['config']['engines']['chatgpt_auth'] == 'subscription'


# --- Server settings (host/port/LAN/access token) ----------------------------
def test_settings_payload_includes_server_defaults(client):
    cfg = client.get('/api/settings').get_json()['config']
    assert cfg['server'] == {'host': '127.0.0.1', 'port': 5050,
                             'require_token': False, 'access_token': ''}


def test_put_settings_saves_require_token(client):
    r = client.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    assert r.status_code == 200
    assert r.get_json()['config']['server']['require_token'] is True


def test_settings_restart_pins_saved_host_port_to_env(client, monkeypatch):
    """The launcher exports LDS_PORT, which otherwise wins over config forever.
    The restart route must stamp the SAVED host/port into env so the relaunch
    actually binds where the user asked (else the port field looks broken)."""
    import os
    from app.services import updater
    monkeypatch.setattr(updater, 'schedule_restart', lambda *a, **k: None)
    monkeypatch.delenv('LDS_PORT', raising=False)
    monkeypatch.delenv('LDS_HOST', raising=False)
    client.put('/api/settings', json={'config': {'server': {'host': '0.0.0.0', 'port': 5123}}})
    client.post('/api/settings/restart')
    assert os.environ['LDS_HOST'] == '0.0.0.0'
    assert os.environ['LDS_PORT'] == '5123'


def test_put_settings_saves_server_lan_and_port(client):
    r = client.put('/api/settings', json={'config': {'server': {'host': '0.0.0.0', 'port': 5001}}})
    assert r.status_code == 200
    assert r.get_json()['config']['server']['host'] == '0.0.0.0'
    assert r.get_json()['config']['server']['port'] == 5001


def test_runtime_reflects_what_run_py_stamped_on_boot(client, app):
    """Before run.py's __main__ block runs (dev/test boots go through create_app()
    directly), nothing has been bound yet -- the card must show that as 'unknown',
    never fabricate a value that looks like a real running bind."""
    rt = client.get('/api/settings').get_json()['runtime']
    assert (rt['host'], rt['port']) == (None, None)   # lan_ip is orthogonal (see its own test)
    app.config['LDS_BOUND_HOST'] = '0.0.0.0'
    app.config['LDS_BOUND_PORT'] = 5000
    rt = client.get('/api/settings').get_json()['runtime']
    assert (rt['host'], rt['port']) == ('0.0.0.0', 5000)


def test_runtime_can_differ_from_saved_config_until_restart(client, app):
    """Saving a new port must NOT retroactively change what's reported as running --
    that would lie about a bind change that hasn't taken effect yet."""
    app.config['LDS_BOUND_HOST'] = '127.0.0.1'
    app.config['LDS_BOUND_PORT'] = 5000
    client.put('/api/settings', json={'config': {'server': {'host': '0.0.0.0', 'port': 5001}}})
    data = client.get('/api/settings').get_json()
    assert data['config']['server']['port'] == 5001
    assert (data['runtime']['host'], data['runtime']['port']) == ('127.0.0.1', 5000)


def test_settings_runtime_includes_lan_ip(client):
    """The Server card builds a real copyable http://<ip>:port/ URL from this;
    it's the machine's primary LAN IPv4, or None (UI falls back to a placeholder)
    when offline / loopback-only. It must never be a loopback address."""
    runtime = client.get('/api/settings').get_json()['runtime']
    assert 'lan_ip' in runtime
    ip = runtime['lan_ip']
    assert ip is None or (isinstance(ip, str) and not ip.startswith('127.'))


def test_settings_runtime_includes_tailscale_ip(client):
    """The Server card offers a Tailscale URL beside the LAN one as the phone's
    off-perimeter path. It's a tailnet address (100.64.0.0/10) or None when the
    tunnel is down — never a bare LAN IP masquerading as a tailnet address."""
    from app.routes import settings as sroutes
    runtime = client.get('/api/settings').get_json()['runtime']
    assert 'tailscale_ip' in runtime
    ip = runtime['tailscale_ip']
    assert ip is None or sroutes._is_cgnat(ip)


def test_is_cgnat_classifies_tailscale_range():
    """Only 100.64.0.0/10 (Tailscale's CGNAT block) counts — a real LAN IP or a
    100.x address outside the block must not be mistaken for a tailnet address."""
    from app.routes import settings as sroutes
    assert sroutes._is_cgnat('100.87.119.32') is True     # in-block (real tailnet IP)
    assert sroutes._is_cgnat('100.64.0.1') is True        # lower edge
    assert sroutes._is_cgnat('100.127.255.254') is True   # upper edge
    assert sroutes._is_cgnat('100.63.255.255') is False   # just below the block
    assert sroutes._is_cgnat('100.128.0.1') is False      # just above the block
    assert sroutes._is_cgnat('192.168.1.162') is False    # a real LAN IP
    assert sroutes._is_cgnat('') is False
    assert sroutes._is_cgnat(None) is False


def test_settings_restart_triggers_schedule_restart(client, monkeypatch):
    from app.services import updater
    called = []
    monkeypatch.setattr(updater, 'schedule_restart', lambda *a, **k: called.append(1))
    r = client.post('/api/settings/restart')
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'restarting': True}
    assert called == [1]
