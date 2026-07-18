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

def test_put_settings_clears_skip_when_dir_provided(client, tmp_path):
    """Entering a ComfyUI directory annuls a prior "continue without ComfyUI" skip —
    the stored flag is cleared so it can't resurface if base_dir is later emptied."""
    from app import config
    config.save_config({'comfyui': {'setup_skipped': True}})
    r = client.put('/api/settings', json={'config': {'comfyui': {'base_dir': str(tmp_path / 'Comfy')}}})
    assert r.status_code == 200
    assert r.get_json()['config']['comfyui']['setup_skipped'] is False


def test_put_settings_skip_persists_when_dir_empty(client):
    """Saving the skip with no directory keeps the flag set (the wizard's skip flow)."""
    r = client.put('/api/settings', json={'config': {'comfyui': {'setup_skipped': True}}})
    assert r.status_code == 200
    assert r.get_json()['config']['comfyui']['setup_skipped'] is True


def test_put_settings_saves_scrape_credentials(client, monkeypatch):
    """Scrape credentials are presence-only and effective without restart."""
    import os
    monkeypatch.setenv('PEXELS_API_KEY', '')
    r = client.put('/api/settings', json={'secrets': {'REDDIT_CLIENT_ID': 'my-cid',
                                                      'CIVITAI_API_KEY': 'civ-key',
                                                      'PEXELS_API_KEY': 'pexels-key'}})
    assert r.status_code == 200
    secrets = r.get_json()['secrets']
    assert secrets['REDDIT_CLIENT_ID'] is True and secrets['CIVITAI_API_KEY'] is True
    assert secrets['PEXELS_API_KEY'] is True
    payload = str(r.get_json())
    assert 'my-cid' not in payload and 'pexels-key' not in payload  # presence only
    assert os.environ['REDDIT_CLIENT_ID'] == 'my-cid'      # effective immediately
    from app.scrape.sources import reddit
    from app.scrape.sources.civitai import civitai_api_key
    from app.scrape.sources.pexels import pexels_api_key
    assert reddit._client_id() == 'my-cid'
    assert civitai_api_key() == 'civ-key'
    assert pexels_api_key() == 'pexels-key'


def test_delete_scrape_credential_falls_back_to_shared_id(client, monkeypatch):
    """Removing the saved Reddit client id must drop it from the env too, so the
    source falls back to the shared gallery-dl id instead of a stale value."""
    import os
    from app.scrape.sources import reddit
    monkeypatch.setattr(reddit, 'resolve_cookies', lambda key: None)  # ignore any local admin file
    client.put('/api/settings', json={'secrets': {'REDDIT_CLIENT_ID': 'my-cid'}})
    r = client.delete('/api/settings/secret/REDDIT_CLIENT_ID')
    assert r.status_code == 200
    assert r.get_json()['secrets']['REDDIT_CLIENT_ID'] is False
    assert 'REDDIT_CLIENT_ID' not in os.environ
    assert reddit._client_id() == reddit._GDL_CLIENT_ID


def test_delete_pexels_api_key_clears_runtime_secret_without_leak(client):
    import os
    from app.scrape.sources.pexels import pexels_api_key

    client.put('/api/settings', json={'secrets': {'PEXELS_API_KEY': 'delete-me'}})
    assert pexels_api_key() == 'delete-me'
    r = client.delete('/api/settings/secret/PEXELS_API_KEY')

    assert r.status_code == 200
    assert r.get_json()['secrets']['PEXELS_API_KEY'] is False
    assert 'delete-me' not in str(r.get_json())
    assert 'PEXELS_API_KEY' not in os.environ
    assert pexels_api_key() is None


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

def test_put_settings_full_config_save_keeps_autoprovisioned_watermark_python(client):
    """A full-config Save (the frontend sends the WHOLE config) must not blank out an
    interpreter the installer auto-provisioned out-of-band. watermark.python has no UI
    input, so the browser always echoes it back as "" on a fresh install; if that blank
    reached save_config it would deep-merge over the dedicated venv path and the feature
    would read "NOT installed" forever despite a perfect install (the reported bug)."""
    from app import config
    config.save_config({'watermark': {'python': '/data/envs/watermark/py.exe'}})
    # The stale full-config Save the browser sends after the install finished.
    r = client.put('/api/settings', json={'config': {
        'watermark': {'allow_crop': True, 'device': 'auto', 'python': ''},
        'masks': {'python': ''},
    }})
    assert r.status_code == 200
    assert r.get_json()['config']['watermark']['python'] == '/data/envs/watermark/py.exe'
    # The user's other watermark edits still land.
    assert r.get_json()['config']['watermark']['allow_crop'] is True


def test_capabilities_endpoint(client):
    caps = client.get('/api/capabilities').get_json()
    assert 'engines' in caps and 'studio_visible' in caps

def test_test_connection_unknown_target(client):
    assert client.post('/api/settings/test/nope').status_code == 404


def test_test_connection_ollama_checks_model_not_just_reachability(client, monkeypatch):
    """POST /api/settings/test/ollama runs the unified probe (reachable AND vision
    model pulled), so the Test button can't be green while the diagnostic says the
    model isn't pulled (issue #7)."""
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_ollama', lambda: {'ok': True, 'detail': 'http://o'})
    # reachable but model absent -> honest failure through the SAME model probe
    monkeypatch.setattr(capabilities, 'probe_ollama_model',
                        lambda **k: {'ok': False, 'detail': 'model not pulled'})
    r = client.post('/api/settings/test/ollama').get_json()
    assert r['ok'] is False and 'not pulled' in r['detail']
    # reachable + present -> green
    monkeypatch.setattr(capabilities, 'probe_ollama_model',
                        lambda **k: {'ok': True, 'detail': 'ready'})
    assert client.post('/api/settings/test/ollama').get_json()['ok'] is True


# --- CSRF cookie freshness (long-lived SPA session) ---------------------------
# Flask-WTF time-limits the CSRF token (WTF_CSRF_TIME_LIMIT). The cookie used to
# be planted ONLY on GET /, so a tab left open past that limit kept echoing a
# stale token and every Save/Test POST failed with a cryptic HTML 400 until a hard
# refresh. An after_request hook now re-plants a fresh token on / and every /api
# response — including the CSRF-rejection 400 itself, so the client's one-shot
# retry can recover without a reload.

def _csrf_cookies(resp):
    """The Set-Cookie header(s) that (re)plant csrf_token, if any."""
    return [c for c in resp.headers.getlist('Set-Cookie') if c.startswith('csrf_token=')]


def test_after_request_plants_fresh_csrf_cookie_on_api(client):
    """Any /api response re-plants the csrf_token cookie, JS-readable (not
    HttpOnly, since the SPA must echo it back in the X-CSRFToken header)."""
    r = client.get('/api/health')
    assert r.status_code == 200
    planted = _csrf_cookies(r)
    assert planted, 'csrf_token cookie must be (re)planted on /api responses'
    assert 'HttpOnly' not in planted[0]
    assert 'SameSite=Lax' in planted[0]


def test_static_assets_do_not_replant_csrf_cookie(client):
    """The hook stays quiet on static assets (pure noise) — only / and /api."""
    r = client.get('/assets/does-not-exist.js')
    assert not _csrf_cookies(r)


@pytest.fixture()
def csrf_client(tmp_path, monkeypatch):
    """A client on an app with CSRF actually enforced (the default fixture turns
    it off). Mirrors conftest's app fixture env/cache isolation."""
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(_cfg, '_cache', None)
    from app import create_app
    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': True,
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    return application.test_client()


def test_csrf_rejection_carries_fresh_cookie_and_allows_retry(csrf_client):
    """A mutating POST with no/stale token is still rejected (400) — but that very
    rejection response re-plants a fresh csrf_token cookie, so the client can read
    it and replay the request once and succeed, with no hard refresh."""
    r = csrf_client.put('/api/settings', json={'config': {'ollama': {'url': 'http://x'}}})
    assert r.status_code == 400                      # token missing -> Flask-WTF rejects
    planted = _csrf_cookies(r)
    assert planted, 'the CSRF-rejection response must still refresh the token cookie'
    token = planted[0].split('csrf_token=', 1)[1].split(';', 1)[0]
    # Replay WITH the fresh token in the header (the session cookie rode along on
    # the test client's jar) -> accepted, no longer a 400.
    r2 = csrf_client.put('/api/settings', json={'config': {'ollama': {'url': 'http://x'}}},
                         headers={'X-CSRFToken': token})
    assert r2.status_code == 200
    assert r2.get_json()['config']['ollama']['url'] == 'http://x'


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


def test_update_apply_defers_changed_requirements_to_restart(client, monkeypatch):
    from app.services import updater
    monkeypatch.setattr(
        updater, 'apply_update',
        lambda: {'ok': True, 'changed': True, 'deps_changed': True},
    )
    calls = []
    monkeypatch.setattr(updater, 'schedule_restart',
                        lambda *a, **k: calls.append((a, k)))

    response = client.post('/api/update/apply')

    assert response.status_code == 200
    assert response.get_json()['restarting'] is True
    assert calls == [((), {'install_requirements': True})]


def test_update_check_reports_can_apply_for_zip_release(client, monkeypatch, _reset_update_cache):
    """A release with a ZIP asset -> can_apply True, so a packaged install can
    update in-app instead of only linking to the releases page."""
    import requests
    monkeypatch.setattr(requests, 'get', lambda *a, **k: _FakeResp(200, {
        'tag_name': 'v9999.12.31',
        'assets': [{'name': 'LoRA-Dataset-Studio-windows.zip',
                    'browser_download_url': 'https://x/win'}]}))
    d = client.get('/api/update/check').get_json()
    assert d['update_available'] is True and d['can_apply'] is True


def test_update_apply_zip_install_uses_release_updater(client, monkeypatch):
    """A NON-git install routes /update/apply to the release-ZIP updater (never the
    git pull path) and still defers changed requirements to the restart helper."""
    from app.services import updater
    monkeypatch.setattr(updater, 'is_git_checkout', lambda root=None: False)
    called = {}
    def fake_zip():
        called['zip'] = True
        return {'ok': True, 'changed': True, 'deps_changed': True,
                'from': '2026.07.16.1', 'to': '9999.01.01'}
    monkeypatch.setattr(updater, 'apply_zip_update', fake_zip)
    monkeypatch.setattr(updater, 'apply_update',
                        lambda: (_ for _ in ()).throw(AssertionError('git path must not run')))
    restart = []
    monkeypatch.setattr(updater, 'schedule_restart', lambda *a, **k: restart.append((a, k)))

    r = client.post('/api/update/apply')

    assert r.status_code == 200 and r.get_json()['restarting'] is True
    assert called.get('zip') is True
    assert restart == [((), {'install_requirements': True})]
