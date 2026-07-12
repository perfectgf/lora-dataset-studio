"""chatgpt_oauth: token store, refresh, Codex CLI import. No real network."""
import json
import time
from unittest.mock import patch, MagicMock, Mock


def _tok(expires_in=3600, **over):
    base = {'access_token': 'at-1', 'refresh_token': 'rt-1', 'id_token': '',
            'account_id': 'acc-1', 'expires_at': time.time() + expires_in,
            'last_refresh': time.time()}
    base.update(over)
    return base


def test_status_disconnected_without_file(app):
    from app.services import chatgpt_oauth as oauth
    s = oauth.status()
    assert s == {'connected': False, 'email': None, 'plan': None}


def test_save_load_roundtrip_and_logout(app):
    from app.services import chatgpt_oauth as oauth
    oauth._save(_tok())
    assert oauth.status()['connected'] is True
    assert oauth.access_token() == 'at-1'
    assert oauth.account_id() == 'acc-1'
    oauth.logout()
    assert oauth.status()['connected'] is False
    assert oauth.access_token() is None


def test_access_token_refreshes_when_expired(app):
    from app.services import chatgpt_oauth as oauth
    oauth._save(_tok(expires_in=-10))                     # already expired
    resp = MagicMock(status_code=200)
    resp.json.return_value = {'access_token': 'at-2', 'refresh_token': 'rt-2',
                              'expires_in': 3600}
    with patch('app.services.chatgpt_oauth.requests.post', return_value=resp) as post:
        assert oauth.access_token() == 'at-2'
    body = post.call_args.kwargs['data']
    assert body['grant_type'] == 'refresh_token'
    assert body['client_id'] == oauth.CLIENT_ID
    assert body['refresh_token'] == 'rt-1'
    assert post.call_args.args[0] == oauth.TOKEN_URL
    # New tokens persisted: a fresh read sees at-2 without another refresh.
    with patch('app.services.chatgpt_oauth.requests.post') as post2:
        assert oauth.access_token() == 'at-2'
    post2.assert_not_called()


def test_refresh_http_failure_disconnects(app):
    from app.services import chatgpt_oauth as oauth
    oauth._save(_tok(expires_in=-10))
    resp = MagicMock(status_code=400, text='invalid_grant')
    with patch('app.services.chatgpt_oauth.requests.post', return_value=resp):
        assert oauth.access_token() is None
    assert oauth.status()['connected'] is False           # token file deleted


def test_refresh_network_error_keeps_session(app):
    """Transient offline must NOT force a reconnect: keep the stored session,
    return None for now (the row fails, the user retries)."""
    import requests as _rq
    from app.services import chatgpt_oauth as oauth
    oauth._save(_tok(expires_in=-10))
    with patch('app.services.chatgpt_oauth.requests.post',
               side_effect=_rq.ConnectionError('offline')):
        assert oauth.access_token() is None
    assert oauth.status()['connected'] is True            # file survived


def test_import_codex_cli(app, tmp_path, monkeypatch):
    from app.services import chatgpt_oauth as oauth
    codex = tmp_path / '.codex'
    codex.mkdir()
    (codex / 'auth.json').write_text(json.dumps({
        'OPENAI_API_KEY': None,
        'tokens': {'id_token': '', 'access_token': 'cx-at', 'refresh_token': 'cx-rt',
                   'account_id': 'cx-acc'},
        'last_refresh': '2026-07-01T00:00:00Z'}), encoding='utf-8')
    monkeypatch.setenv('CODEX_HOME', str(codex))
    out = oauth.import_codex_cli()
    assert out['ok'] is True
    tok = oauth._load()
    assert tok['refresh_token'] == 'cx-rt'
    assert tok['account_id'] == 'cx-acc'
    assert tok['expires_at'] == 0        # unknown expiry -> first use refreshes


def test_import_codex_cli_missing_or_malformed(app, tmp_path, monkeypatch):
    from app.services import chatgpt_oauth as oauth
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'nowhere'))
    assert oauth.import_codex_cli()['ok'] is False
    codex = tmp_path / '.codex'
    codex.mkdir()
    (codex / 'auth.json').write_text('not json', encoding='utf-8')
    monkeypatch.setenv('CODEX_HOME', str(codex))
    assert oauth.import_codex_cli()['ok'] is False
    (codex / 'auth.json').write_text(json.dumps({'OPENAI_API_KEY': 'sk-x'}), encoding='utf-8')
    out = oauth.import_codex_cli()
    assert out['ok'] is False            # API-key-only login: no ChatGPT tokens


def _post_router(routes):
    """Route mocked requests.post by URL substring -> MagicMock response."""
    def _post(url, **kwargs):
        for frag, resp in routes.items():
            if frag in url:
                # Return Mock objects as-is; only call if it's a real function
                if isinstance(resp, Mock):
                    return resp
                return resp if not callable(resp) else resp(url, **kwargs)
        raise AssertionError(f'unexpected POST {url}')
    return _post


def test_login_start_returns_code(app):
    from app.services import chatgpt_oauth as oauth
    uc = MagicMock(status_code=200)
    uc.json.return_value = {'device_auth_id': 'dev-1', 'user_code': 'ABCD-1234',
                            'interval': '5'}
    with patch('app.services.chatgpt_oauth.requests.post',
               side_effect=_post_router({'/deviceauth/usercode': uc})):
        out = oauth.login_start()
    assert out['ok'] is True
    assert out['user_code'] == 'ABCD-1234'
    assert out['verification_url'] == oauth.DEVICE_VERIFY_URL


def test_login_poll_pending_then_connected(app):
    from app.services import chatgpt_oauth as oauth
    uc = MagicMock(status_code=200)
    uc.json.return_value = {'device_auth_id': 'dev-1', 'user_code': 'ABCD-1234',
                            'interval': '5'}
    pending = MagicMock(status_code=403)
    with patch('app.services.chatgpt_oauth.requests.post',
               side_effect=_post_router({'/deviceauth/usercode': uc,
                                         '/deviceauth/token': pending})):
        oauth.login_start()
        assert oauth.login_poll()['status'] == 'pending'
    done = MagicMock(status_code=200)
    done.json.return_value = {'authorization_code': 'code-1',
                              'code_challenge': 'ch', 'code_verifier': 'ver-1'}
    exch = MagicMock(status_code=200)
    exch.json.return_value = {'access_token': 'at-9', 'refresh_token': 'rt-9',
                              'id_token': '', 'expires_in': 3600}
    with patch('app.services.chatgpt_oauth.requests.post',
               side_effect=_post_router({'/deviceauth/token': done,
                                         '/oauth/token': exch})) as post:
        assert oauth.login_poll()['status'] == 'connected'
    assert oauth.access_token() == 'at-9'
    exch_body = post.call_args.kwargs['data']            # last call = the code exchange
    assert exch_body['grant_type'] == 'authorization_code'
    assert exch_body['code'] == 'code-1'
    assert exch_body['code_verifier'] == 'ver-1'
    assert exch_body['redirect_uri'] == oauth.DEVICE_REDIRECT_URI


def test_login_poll_without_start_errors(app):
    from app.services import chatgpt_oauth as oauth
    oauth._clear_pending()
    assert oauth.login_poll()['status'] == 'error'


def test_login_poll_expires_after_ttl(app, monkeypatch):
    from app.services import chatgpt_oauth as oauth
    uc = MagicMock(status_code=200)
    uc.json.return_value = {'device_auth_id': 'dev-1', 'user_code': 'ABCD-1234',
                            'interval': '5'}
    with patch('app.services.chatgpt_oauth.requests.post',
               side_effect=_post_router({'/deviceauth/usercode': uc})):
        oauth.login_start()
    real_now = time.time()
    monkeypatch.setattr(oauth.time, 'time', lambda: real_now + 1000)
    assert oauth.login_poll()['status'] == 'error'
