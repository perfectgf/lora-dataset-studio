"""Network access guard for non-loopback binds.

The app has NO user accounts (single local user by design): every route can read
API keys, launch GPU trainings or delete datasets. That is fine on 127.0.0.1 —
but `server.host` is configurable, and binding 0.0.0.0 (e.g. to reach the app
from a phone) would otherwise expose everything to the whole LAN.

Rule: requests from loopback clients are always allowed (the normal local flow,
untouched). Any OTHER client must present the access token — `run.py` generates
one automatically (and prints it) when the bind is non-loopback and none is set,
so the protection cannot be forgotten. Token sources, in order:
  - `Authorization: Bearer <token>` header
  - `X-LDS-Token: <token>` header
  - `?token=<token>` query parameter (first hit from a phone browser) — on
    success the flag is remembered in the signed session cookie so the SPA's
    subsequent fetches just work.

Escape hatch for setups with their own network isolation (VPN, reverse proxy
with auth, trusted Docker network): `LDS_ALLOW_UNAUTHENTICATED=1`.
"""
from __future__ import annotations
import ipaddress
import os
import secrets

from flask import jsonify, request, session

SESSION_FLAG = 'lds_token_ok'


def _is_loopback(addr: str | None) -> bool:
    if not addr:
        # No REMOTE_ADDR (unit tests, some WSGI shims): treat as local rather
        # than locking the single-user app out of itself.
        return True
    try:
        return ipaddress.ip_address(addr.split('%')[0]).is_loopback
    except ValueError:
        return False


def _presented_token() -> str | None:
    auth = request.headers.get('Authorization', '')
    if auth.lower().startswith('bearer '):
        return auth[7:].strip()
    return request.headers.get('X-LDS-Token') or request.args.get('token')


def install_network_guard(app):
    @app.before_request
    def _network_guard():
        if _is_loopback(request.remote_addr):
            return None
        if os.environ.get('LDS_ALLOW_UNAUTHENTICATED') == '1':
            return None
        token = os.environ.get('LDS_ACCESS_TOKEN') or app.config.get('LDS_ACCESS_TOKEN')
        if not token:
            # Non-loopback client but no token configured (custom WSGI launch that
            # bypassed run.py): fail CLOSED with an actionable message.
            return jsonify({'error': 'remote access requires an access token — '
                                     'set LDS_ACCESS_TOKEN (see README) or bind 127.0.0.1'}), 403
        if session.get(SESSION_FLAG):
            return None
        presented = _presented_token()
        if presented and secrets.compare_digest(str(presented), str(token)):
            session[SESSION_FLAG] = True   # signed cookie → the SPA's fetches follow
            return None
        return jsonify({'error': 'invalid or missing access token'}), 403
