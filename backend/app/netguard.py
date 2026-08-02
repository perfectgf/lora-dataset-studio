"""Network access guard for non-loopback binds.

The app has NO user accounts (single local user by design): every route can read
API keys, launch GPU trainings or delete datasets. That is fine on 127.0.0.1 —
but `server.host` is configurable, and binding 0.0.0.0 (e.g. to reach the app
from a phone) would otherwise expose everything to the whole LAN.

Rule: requests from loopback clients are always allowed (the normal local flow,
untouched, and the path the container healthcheck uses). A non-loopback (LAN)
client is allowed straight through UNLESS the user opted into a token via
Settings (`server.require_token`) — a home LAN is trusted by default, so no token
has to be typed on a phone.

That default is wrong exactly once: when the bind is reachable from the public
internet (a RunPod pod's proxy URL, a tunnel). `LDS_PUBLIC=1` says so, and forces
the gate on regardless of `server.require_token`, so turning the setting off
cannot open the app to everyone. `run.py` generates a token automatically (and
prints it) when none is set, so the gate can't be forgotten.
Token sources, in order:
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


_TRUTHY = {'1', 'true', 'yes', 'on'}


def public_bind() -> bool:
    """True when this process is served on a URL the public internet can reach.

    Declared by the deployment (the RunPod template, a tunnel, a reverse proxy) —
    never inferred from REMOTE_ADDR. Behind a proxy the peer address is the
    proxy's, and _is_loopback() answers True for a MISSING REMOTE_ADDR; a security
    property must not hang on either. Truthy set matches LDS_BIND_MANAGED's.

    Only hardens NON-loopback binds: a loopback-bound app behind a local reverse
    proxy still sees 127.0.0.1 and is allowed through, which is what
    LDS_ALLOW_UNAUTHENTICATED and the proxy's own auth are for.
    """
    return os.environ.get('LDS_PUBLIC', '').strip().lower() in _TRUTHY


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
        # LAN access is open by default (trusted home network); the token gate
        # only engages when the user turned it on in Settings. Read lazily so the
        # toggle takes effect on the next request without a restart.
        from . import config as cfg
        if not (public_bind() or cfg.get('server.require_token')):
            return None
        # config.server.access_token is read here too (not only the boot-time env)
        # so turning the gate on with a saved token works LIVE — no restart, unlike
        # the bind change. run.py still seeds the env token at boot for the custom
        # WSGI path that never writes config.
        token = (os.environ.get('LDS_ACCESS_TOKEN') or app.config.get('LDS_ACCESS_TOKEN')
                 or cfg.get('server.access_token'))
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
