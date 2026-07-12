"""chatgpt_oauth: token store, refresh, Codex CLI import. No real network."""
import json
import time
from unittest.mock import patch, MagicMock


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
