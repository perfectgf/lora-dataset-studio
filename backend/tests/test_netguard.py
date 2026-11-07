"""Network guard (app/netguard.py): loopback untouched; LAN open by default, token
gated only when the user opts in (server.require_token).

The app is single-user and unauthenticated by design. A home LAN is trusted, so a
`server.host: 0.0.0.0` bind is reachable without a token by default (no password to
type on a phone); turning on server.require_token restores the token gate for a
shared/untrusted network (audit 2026-07-12, angle securite). REMOTE_ADDR is
simulated via environ_base: the Flask test client defaults to 127.0.0.1.
"""
import os

import pytest

REMOTE = {'REMOTE_ADDR': '192.168.1.50'}


def _require_token(client):
    """Turn the opt-in token gate ON (persists server.require_token in the tmp config)."""
    client.put('/api/settings', json={'config': {'server': {'require_token': True}}})


def test_loopback_client_needs_no_token(client):
    r = client.get('/api/health')            # default REMOTE_ADDR = 127.0.0.1
    assert r.status_code == 200


def test_health_names_this_process_for_the_restart_reload(client):
    # After an update the page reloads only when the boot id changes: the old
    # process keeps answering "ok" for about a second before it exits.
    first, again = client.get('/api/health').json, client.get('/api/health').json
    assert first['ok'] is True and len(first['boot']) == 16 and again['boot'] == first['boot']


def test_remote_client_allowed_by_default_no_token(client):
    """require_token defaults OFF -> a LAN client gets in with no token (the whole
    point of trusted-LAN mode: reach it from a phone without typing a token)."""
    assert client.get('/api/health', environ_base=REMOTE).status_code == 200


def test_remote_client_blocked_when_token_required_but_none_configured(client):
    _require_token(client)
    r = client.get('/api/health', environ_base=REMOTE)
    assert r.status_code == 403
    assert 'LDS_ACCESS_TOKEN' in r.get_json()['error']


def test_remote_client_blocked_with_wrong_token(app, monkeypatch):
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    r = c.get('/api/health', environ_base=REMOTE,
              headers={'Authorization': 'Bearer nope'})
    assert r.status_code == 403


def test_remote_client_bearer_token_ok(app, monkeypatch):
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    r = c.get('/api/health', environ_base=REMOTE,
              headers={'Authorization': 'Bearer sekret'})
    assert r.status_code == 200


def test_query_token_sets_session_for_spa_fetches(app, monkeypatch):
    """First hit from a phone browser: ?token=... — then the SPA's fetches ride
    the signed session cookie without re-presenting the token."""
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    assert c.get('/api/health?token=sekret', environ_base=REMOTE).status_code == 200
    assert c.get('/api/health', environ_base=REMOTE).status_code == 200   # cookie remembered


def test_escape_hatch_env(app, monkeypatch):
    monkeypatch.setenv('LDS_ALLOW_UNAUTHENTICATED', '1')
    c = app.test_client()
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    assert c.get('/api/health', environ_base=REMOTE).status_code == 200


def test_public_runtime_forces_token_gate(app, monkeypatch):
    """LDS_PUBLIC=1: server.require_token stays OFF in config, but a non-loopback
    client is still refused. A public URL must never serve the unauthenticated UI,
    and toggling the setting off in Settings must not be able to open it."""
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = app.test_client()
    assert c.get('/api/health', environ_base=REMOTE).status_code == 403
    ok = c.get('/api/health', environ_base=REMOTE,
               headers={'Authorization': 'Bearer sekret'})
    assert ok.status_code == 200


def test_public_runtime_still_allows_loopback(app, monkeypatch):
    """The container healthcheck hits 127.0.0.1:5050/api/health with no token."""
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    assert app.test_client().get('/api/health').status_code == 200


def test_public_runtime_without_token_fails_closed(app, monkeypatch):
    """No token configured at all -> refuse, with the actionable message."""
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
    r = app.test_client().get('/api/health', environ_base=REMOTE)
    assert r.status_code == 403
    assert 'LDS_ACCESS_TOKEN' in r.get_json()['error']


def test_public_runtime_escape_hatch_still_wins(app, monkeypatch):
    """LDS_ALLOW_UNAUTHENTICATED stays the documented opt-out for setups with
    their own auth (VPN, authenticating reverse proxy)."""
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.setenv('LDS_ALLOW_UNAUTHENTICATED', '1')
    assert app.test_client().get('/api/health', environ_base=REMOTE).status_code == 200


def test_public_flag_is_not_truthy_for_arbitrary_values(app, monkeypatch):
    """Only the documented truthy set counts; '0' or a typo must not silently
    harden (or silently fail to harden) the app."""
    monkeypatch.setenv('LDS_PUBLIC', '0')
    assert app.test_client().get('/api/health', environ_base=REMOTE).status_code == 200


@pytest.mark.parametrize('peer', ['127.0.0.1', '::1', '::ffff:127.0.0.1', '', '192.168.1.50'])
def test_local_protection_denies_anonymous_sensitive_reads_and_writes(app, monkeypatch, peer):
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'test-local-credential')
    monkeypatch.setenv('LDS_ALLOW_UNAUTHENTICATED', '1')
    c = app.test_client()
    source = {'REMOTE_ADDR': peer}
    spoofed = {'Origin': 'http://localhost', 'X-Forwarded-For': '127.0.0.1'}
    for path in ('/', '/api/settings', '/api/csrf-token', '/api/plugins'):
        assert c.get(path, environ_overrides=source, headers=spoofed).status_code == 403
    assert c.put('/api/settings', json={'config': {'server': {'require_token': False}}},
                 environ_overrides=source, headers=spoofed).status_code == 403


def test_local_health_exception_is_only_a_real_loopback_read(app, monkeypatch):
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'test-local-credential')
    c = app.test_client()
    response = c.get('/api/health')
    assert response.status_code == 200 and set(response.json) == {'ok', 'boot'}
    assert c.head('/api/health').status_code == 200
    assert c.post('/api/health').status_code == 403
    for peer in ('', '192.168.1.50'):
        assert c.get('/api/health', environ_overrides={'REMOTE_ADDR': peer}).status_code == 403
    # Reading the health endpoint never creates an authenticated session.
    assert c.get('/api/settings').status_code == 403


def test_local_protection_without_a_credential_fails_closed(app, monkeypatch):
    from app import config as cfg
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
    app.config.pop('LDS_ACCESS_TOKEN', None)
    with app.app_context():
        cfg.save_config({'server': {'access_token': ''}})
    response = app.test_client().get('/api/settings')
    assert response.status_code == 403 and 'LDS_ACCESS_TOKEN' in response.json['error']


@pytest.mark.parametrize('header', ['Authorization', 'X-LDS-Token'])
def test_local_credential_authenticates_only_the_requesting_session(app, monkeypatch, header):
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'fake-local-credential')
    c = app.test_client()
    credential = 'fake-local-credential'
    value = f'Bearer {credential}' if header == 'Authorization' else credential
    assert c.get('/api/settings', headers={header: value}).status_code == 200
    assert c.get('/api/settings').status_code == 200
    assert app.test_client().get('/api/settings').status_code == 403


def test_local_browser_login_cleans_the_token_and_rejects_sessions_after_rotation(app, monkeypatch):
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'test-local-credential')
    c = app.test_client()
    login = c.get('/?token=test-local-credential')
    assert login.status_code == 303 and login.headers['Location'] == '/'
    assert login.headers['Cache-Control'] == 'no-store'
    assert login.headers['Referrer-Policy'] == 'no-referrer'
    assert c.get('/api/settings').status_code == 200
    again = c.get('/?token=test-local-credential')
    assert again.status_code == 303 and again.headers['Location'] == '/'
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'replacement-local-credential')
    assert c.get('/api/settings').status_code == 403
    assert c.get('/api/settings', headers={'X-LDS-Token': 'replacement-local-credential'}).status_code == 200


def test_a_legacy_session_flag_does_not_bypass_local_authentication(app, monkeypatch):
    from app.netguard import SESSION_FLAG
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'test-local-credential')
    c = app.test_client()
    with c.session_transaction() as state:
        state[SESSION_FLAG] = True
    assert c.get('/api/settings').status_code == 403
    assert c.get('/api/settings', headers={'X-LDS-Token': 'incorrect-\u00e9'}).status_code == 403


def _sign_in(client, token):
    assert client.get('/api/settings', headers={'X-LDS-Token': token}).status_code == 200


def test_runtime_tells_the_truth_about_the_local_lock(app, monkeypatch):
    monkeypatch.delenv('LDS_REQUIRE_LOCAL_TOKEN', raising=False)
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'test-local-credential')
    runtime = app.test_client().get('/api/settings').json['runtime']
    assert runtime['local_token_required'] is False and runtime['access_token'] is None

    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    with app.app_context():
        from app import config as cfg
        cfg.save_config({'server': {'access_token': 'saved-local-credential'}})
    client = app.test_client()
    _sign_in(client, 'test-local-credential')
    runtime = client.get('/api/settings').json['runtime']
    # The token the server accepts, even when the saved one differs from it.
    assert runtime['local_token_required'] is True
    assert runtime['access_token'] == 'test-local-credential'


def test_a_token_saved_under_the_local_lock_applies_at_once(app, monkeypatch):
    from app import config as cfg
    from app import netguard
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'test-local-credential')
    with app.app_context():
        cfg.save_config({'server': {'access_token': 'test-local-credential'}})
    saver, other = app.test_client(), app.test_client()
    _sign_in(saver, 'test-local-credential')
    _sign_in(other, 'test-local-credential')

    saved = saver.put('/api/settings', json={'config': {'server': {
        'access_token': 'replacement-local-credential'}}})

    assert saved.status_code == 200
    assert saved.json['runtime']['access_token'] == 'replacement-local-credential'
    assert saver.get('/api/settings').status_code == 200
    assert other.get('/api/settings').status_code == 403
    stale = app.test_client().get('/api/settings', headers={'X-LDS-Token': 'test-local-credential'})
    assert stale.status_code == 403
    assert app.test_client().get('/?token=replacement-local-credential').status_code == 303
    # A direct relaunch inherits this environment and keeps what it finds there.
    with app.app_context():
        assert netguard.ensure_access_token('127.0.0.1') == 'replacement-local-credential'


def test_saving_a_token_leaves_an_operator_token_and_the_unlocked_mode_alone(app, monkeypatch):
    from app import config as cfg
    monkeypatch.setenv('LDS_REQUIRE_LOCAL_TOKEN', '1')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'operator-local-credential')
    with app.app_context():
        cfg.save_config({'server': {'access_token': 'test-local-credential'}})
    client = app.test_client()
    _sign_in(client, 'operator-local-credential')
    saved = client.put('/api/settings', json={'config': {'server': {
        'access_token': 'replacement-local-credential'}}})
    assert saved.json['runtime']['access_token'] == 'operator-local-credential'
    assert client.get('/api/settings').status_code == 200

    monkeypatch.delenv('LDS_REQUIRE_LOCAL_TOKEN')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'test-local-credential')
    client.put('/api/settings', json={'config': {'server': {'access_token': 'another-local-credential'}}})
    assert os.environ['LDS_ACCESS_TOKEN'] == 'test-local-credential'
