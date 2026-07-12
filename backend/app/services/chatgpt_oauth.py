"""ChatGPT subscription (Codex OAuth) token lifecycle for the ChatGPT engine.

Implements OpenAI's device-code login (the same flow `codex login --device-auth`
uses), token refresh, and one-click import of an existing Codex CLI session.
Tokens live in data/chatgpt_oauth.json — rotating machine-managed credentials,
NOT in .env (which holds static human-entered secrets).

EXPERIMENTAL LANE: the Codex Responses backend is undocumented and OpenAI may
close subscription image generation for third-party OAuth clients at any time.
Everything specific to it is contained in this module + the subscription path
of chatgpt_image.py; the API-key path is untouched.
"""
from __future__ import annotations
import base64
import json
import logging
import os
import threading
import time
from pathlib import Path

import requests

from .. import config as cfg

logger = logging.getLogger(__name__)

# Public OAuth constants, shipped in openai/codex (not secrets).
CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
AUTH_BASE = 'https://auth.openai.com'
TOKEN_URL = f'{AUTH_BASE}/oauth/token'
DEVICE_USERCODE_URL = f'{AUTH_BASE}/api/accounts/deviceauth/usercode'
DEVICE_TOKEN_URL = f'{AUTH_BASE}/api/accounts/deviceauth/token'
DEVICE_VERIFY_URL = f'{AUTH_BASE}/codex/device'
DEVICE_REDIRECT_URI = f'{AUTH_BASE}/deviceauth/callback'
# ChatGPT-specific claims (account id, plan) live under this JWT claim key.
JWT_AUTH_CLAIM = 'https://api.openai.com/auth'
_REFRESH_MARGIN = 300          # refresh when < 5 min to expiry
_DEVICE_LOGIN_TTL = 900        # the one-time code expires after 15 min

_lock = threading.Lock()


def _token_path() -> Path:
    return cfg._data_dir() / 'chatgpt_oauth.json'


def _load() -> dict | None:
    try:
        return json.loads(_token_path().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _save(tok: dict) -> None:
    p = _token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(tok, indent=2), encoding='utf-8')
    tmp.replace(p)


def logout() -> None:
    try:
        _token_path().unlink()
    except OSError:
        pass


def _decode_jwt(token: str) -> dict:
    try:
        payload = token.split('.')[1]
        raw = base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4))
        return json.loads(raw)
    except (IndexError, ValueError, TypeError):
        return {}


def _account_id_from_jwt(*jwts) -> str | None:
    for t in jwts:
        acc = (_decode_jwt(t or '').get(JWT_AUTH_CLAIM) or {}).get('chatgpt_account_id')
        if acc:
            return acc
    return None


def status() -> dict:
    tok = _load()
    if not tok or not tok.get('access_token'):
        return {'connected': False, 'email': None, 'plan': None}
    claims = _decode_jwt(tok.get('id_token') or '')
    return {'connected': True,
            'email': claims.get('email'),
            'plan': (claims.get(JWT_AUTH_CLAIM) or {}).get('chatgpt_plan_type')}


def account_id() -> str | None:
    tok = _load() or {}
    return tok.get('account_id') or _account_id_from_jwt(
        tok.get('id_token'), tok.get('access_token'))


def _refresh(tok: dict) -> dict | None:
    """Exchange the refresh token for a fresh access token. HTTP failure =
    revoked/expired grant -> drop the session (reconnect required). Network
    failure = transient -> keep the session, just fail this call."""
    rt = tok.get('refresh_token')
    if not rt:
        logout()
        return None
    try:
        r = requests.post(TOKEN_URL, data={'grant_type': 'refresh_token',
                                           'client_id': CLIENT_ID,
                                           'refresh_token': rt}, timeout=30)
    except requests.RequestException as e:
        logger.warning(f"chatgpt_oauth: refresh network error: {e}")
        return None
    if r.status_code != 200:
        logger.warning(f"chatgpt_oauth: refresh failed HTTP {r.status_code}: {r.text[:200]}")
        logout()
        return None
    j = r.json()
    out = dict(tok)
    out['access_token'] = j.get('access_token') or tok.get('access_token')
    out['refresh_token'] = j.get('refresh_token') or rt
    if j.get('id_token'):
        out['id_token'] = j['id_token']
    out['expires_at'] = time.time() + int(j.get('expires_in') or 3600)
    out['last_refresh'] = time.time()
    out['account_id'] = (tok.get('account_id')
                         or _account_id_from_jwt(out.get('id_token'), out.get('access_token')))
    _save(out)
    return out


def access_token(force_refresh: bool = False) -> str | None:
    with _lock:
        tok = _load()
        if not tok:
            return None
        if force_refresh or time.time() > float(tok.get('expires_at') or 0) - _REFRESH_MARGIN:
            tok = _refresh(tok)
        return (tok or {}).get('access_token')


def codex_auth_path() -> Path:
    return Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'auth.json'


def import_codex_cli() -> dict:
    """Copy an existing Codex CLI session (`codex login`). LDS refreshes its
    copy independently and never writes back to Codex's file."""
    p = codex_auth_path()
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except OSError:
        return {'ok': False, 'detail': f'no Codex CLI session found at {p}'}
    except ValueError:
        return {'ok': False, 'detail': f'unreadable auth file: {p}'}
    t = data.get('tokens') or {}
    if not (t.get('access_token') and t.get('refresh_token')):
        return {'ok': False,
                'detail': 'auth file has no ChatGPT tokens (API-key-only Codex login?)'}
    _save({'access_token': t['access_token'], 'refresh_token': t['refresh_token'],
           'id_token': t.get('id_token') or '',
           'account_id': t.get('account_id')
                         or _account_id_from_jwt(t.get('id_token'), t.get('access_token')),
           # Unknown expiry -> 0 forces a refresh on first use.
           'expires_at': 0, 'last_refresh': None, 'source': 'codex_cli'})
    return {'ok': True, 'detail': f'imported from {p}'}


# --- Device-code login --------------------------------------------------------
# OpenAI's Codex device flow (custom, not RFC 8628): request a one-time code,
# the user enters it at DEVICE_VERIFY_URL from ANY device, we poll until the
# server hands back an authorization code + the PKCE verifier it generated,
# then do a standard code exchange. Works when the LDS UI is accessed remotely
# (no localhost callback involved).
_pending: dict = {}


def _clear_pending() -> None:
    with _lock:
        _pending.clear()


def login_start() -> dict:
    try:
        r = requests.post(DEVICE_USERCODE_URL, json={'client_id': CLIENT_ID}, timeout=15)
    except requests.RequestException as e:
        return {'ok': False, 'detail': f'network error: {e}'}
    if r.status_code != 200:
        return {'ok': False, 'detail': f'device login unavailable (HTTP {r.status_code})'}
    j = r.json()
    with _lock:
        _pending.clear()
        _pending.update(device_auth_id=j.get('device_auth_id'),
                        user_code=j.get('user_code'), started=time.time())
    return {'ok': True, 'verification_url': DEVICE_VERIFY_URL,
            'user_code': j.get('user_code')}


def _exchange_code(code: str, verifier: str) -> dict | None:
    try:
        r = requests.post(TOKEN_URL, data={'grant_type': 'authorization_code',
                                           'client_id': CLIENT_ID, 'code': code,
                                           'code_verifier': verifier,
                                           'redirect_uri': DEVICE_REDIRECT_URI}, timeout=30)
    except requests.RequestException as e:
        logger.warning(f"chatgpt_oauth: code exchange network error: {e}")
        return None
    if r.status_code != 200:
        logger.warning(f"chatgpt_oauth: code exchange failed HTTP {r.status_code}: {r.text[:200]}")
        return None
    j = r.json()
    if not (j.get('access_token') and j.get('refresh_token')):
        return None
    return {'access_token': j['access_token'], 'refresh_token': j['refresh_token'],
            'id_token': j.get('id_token') or '',
            'account_id': _account_id_from_jwt(j.get('id_token'), j.get('access_token')),
            'expires_at': time.time() + int(j.get('expires_in') or 3600),
            'last_refresh': time.time(), 'source': 'device_code'}


def login_poll() -> dict:
    with _lock:
        pending = dict(_pending)
    if not pending.get('device_auth_id'):
        return {'status': 'error', 'detail': 'no device login in progress'}
    if time.time() - pending['started'] > _DEVICE_LOGIN_TTL:
        _clear_pending()
        return {'status': 'error', 'detail': 'device login expired (15 min) — start again'}
    try:
        r = requests.post(DEVICE_TOKEN_URL,
                          json={'device_auth_id': pending['device_auth_id'],
                                'user_code': pending['user_code']}, timeout=15)
    except requests.RequestException:
        return {'status': 'pending', 'detail': 'network hiccup — still waiting'}
    if r.status_code in (403, 404):
        return {'status': 'pending', 'detail': None}       # user hasn't finished yet
    if r.status_code != 200:
        _clear_pending()
        return {'status': 'error', 'detail': f'device login failed (HTTP {r.status_code})'}
    j = r.json()
    tok = _exchange_code(j.get('authorization_code'), j.get('code_verifier'))
    _clear_pending()
    if not tok:
        return {'status': 'error', 'detail': 'code exchange failed — start again'}
    _save(tok)
    return {'status': 'connected', 'detail': None}
