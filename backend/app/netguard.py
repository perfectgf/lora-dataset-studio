"""Network access guard for non-loopback binds.

The app has NO user accounts (single local user by design): every route can read
API keys, launch GPU trainings or delete datasets. That is fine on 127.0.0.1 —
but `server.host` is configurable, and binding 0.0.0.0 (e.g. to reach the app
from a phone) would otherwise expose everything to the whole LAN.

By default, requests from loopback clients are allowed (the normal local flow
and the path the container healthcheck uses). A non-loopback (LAN)
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

`LDS_REQUIRE_LOCAL_TOKEN=1` requires authentication on local connections too,
regardless of these opt-outs. Only the loopback GET/HEAD health probe stays
public: it returns a boolean and a boot identifier, no application data.
This prevents anonymous local HTTP calls; it is not isolation from processes
that can read this account's token or session signing key.
"""
from __future__ import annotations
import hashlib
import hmac
import ipaddress
import os
import secrets
from urllib.parse import urlencode

from flask import jsonify, redirect, request, session

SESSION_FLAG = 'lds_token_ok'
SESSION_TOKEN = 'lds_token_binding'


_TRUTHY = {'1', 'true', 'yes', 'on'}


def local_token_required() -> bool:
    return os.environ.get('LDS_REQUIRE_LOCAL_TOKEN', '').strip().lower() in _TRUTHY


def local_browser_url(url: str, token: str | None) -> str:
    """Authenticate a launcher-opened local tab; ordinary launches keep their URL."""
    if local_token_required() and token:
        return url + '?' + urlencode({'token': token})
    return url


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
    def accepted():
        if request.method == 'GET' and request.path == '/' and request.args.get('token'):
            # Remove the credential even when this browser already has a valid
            # session (for example, opening the launcher's link a second time).
            response = redirect('/', code=303)
            response.headers['Cache-Control'] = 'no-store'
            response.headers['Referrer-Policy'] = 'no-referrer'
            return response
        return None

    @app.before_request
    def _network_guard():
        local = _is_loopback(request.remote_addr)
        strict = local_token_required()
        if local and not strict:
            return None
        # Keep process supervision and double-launch detection working. The
        # method/path and an actual loopback peer must all match; missing peer
        # information is never enough for this exception.
        if (strict and request.remote_addr and local and request.path == '/api/health'
                and request.method in {'GET', 'HEAD'}):
            return None
        if not strict and os.environ.get('LDS_ALLOW_UNAUTHENTICATED') == '1':
            return None
        # LAN access is open by default (trusted home network); the token gate
        # only engages when the user turned it on in Settings. Read lazily so the
        # toggle takes effect on the next request without a restart.
        from . import config as cfg
        if not (strict or public_bind() or cfg.get('server.require_token')):
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
            return jsonify({'error': 'access requires an access token — '
                                     'set LDS_ACCESS_TOKEN or start LDS with its launcher'}), 403
        # Bind a session to the current credential. A bare legacy flag, or a
        # session from before a token change, must not bypass local protection.
        key = app.secret_key
        if isinstance(key, str):
            key = key.encode('utf-8')
        binding = hmac.new(key, str(token).encode('utf-8'), hashlib.sha256).hexdigest()
        saved = session.get(SESSION_TOKEN)
        if (session.get(SESSION_FLAG) and isinstance(saved, str)
                and secrets.compare_digest(saved.encode('utf-8'), binding.encode('ascii'))):
            return accepted()
        presented = _presented_token()
        if presented and secrets.compare_digest(str(presented).encode('utf-8'), str(token).encode('utf-8')):
            session[SESSION_FLAG] = True   # signed cookie → the SPA's fetches follow
            session[SESSION_TOKEN] = binding
            return accepted()
        session.pop(SESSION_FLAG, None)
        session.pop(SESSION_TOKEN, None)
        return jsonify({'error': 'invalid or missing access token'}), 403


_LOOPBACK_HOSTS = {'127.0.0.1', 'localhost', '::1'}


def ensure_access_token(host: str) -> str | None:
    """Make sure netguard has a token to check, before the server starts.

    Returns the token when the gate is engaged (and generates + persists one if
    none existed), or None when no token is needed. Persisted in config.json
    rather than only stamped into the environment so it survives a restart
    instead of rotating every boot -- the Settings "Server" card reads it back
    from there to show and copy.
    """
    strict = local_token_required()
    if host in _LOOPBACK_HOSTS and not strict:
        return None
    if not strict and os.environ.get('LDS_ALLOW_UNAUTHENTICATED') == '1':
        return None
    from . import config as cfg
    if not (strict or public_bind() or cfg.get('server.require_token')):
        return None
    existing = os.environ.get('LDS_ACCESS_TOKEN')
    if existing:
        return existing
    token = cfg.get('server.access_token') or ''
    if not token:
        token = secrets.token_urlsafe(24)
        try:
            from .config import save_config
            save_config({'server': {'access_token': token}})
        except Exception:
            pass          # persistence failed (e.g. read-only/root-owned volume)
                          # -> fall back to an ephemeral in-process token
    os.environ['LDS_ACCESS_TOKEN'] = token
    return token
