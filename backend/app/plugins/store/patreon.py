"""Patreon access: shipped configuration, account link and membership status.

The Patreon service is an operator-run acquisition service (HTTP contract v1
plus its Patreon endpoints). LDS never handles Patreon credentials: the browser
authorizes on www.patreon.com and the service records the link server-side.
A callback page is never an entitlement; only the library response is, and the
service checks membership again on every download grant. Since protocol v1.1
the callback links nothing by itself: the Patreon page shows a code, and only
typing it in LDS (``confirm``) links the installation that asked.

``store/patreon.json`` ships with the app. ``LDS_STORE_PATREON_CONFIG`` points
a test instance at another file (HTTP is accepted for literal loopback only).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from flask import jsonify, request

from ... import config as cfg
from ..admin import is_admin, require_admin
from ..manifest import EXTERNAL_ID
from ..official import OFFICIAL_IDS
from .client import StoreConfig, StoreError, StoreNotConfigured, load_config, namespace_of, valid_namespace
from .commerce import CommerceClient, ServiceRefused, _origin

CONFIG_ENV = 'LDS_STORE_PATREON_CONFIG'
CONFIG_FORMAT = 'lds-patreon/1'
PATREON_ORIGIN = 'https://www.patreon.com'
AUTHORIZE_PREFIX = 'https://www.patreon.com/oauth2/authorize?'
MAX_CONFIG_BYTES = 256 * 1024
MAX_ROOT_BYTES = 512000
LINK_LIFETIME_SECONDS = 600
# `expires_at` of an authorization link only informs: the service enforces it.
# A local clock that lags or leads, up to a day (a time zone left wrong), must
# not refuse the link; a date further off still reads as a broken answer.
CLOCK_SKEW_SECONDS = 24 * 3600
STATUSES = frozenset({'active', 'inactive', 'verification_required'})
TIERS = frozenset({'member', 'pro'})
REASONS = (None, 'free_trial')
# The code of the Patreon page: 8 characters of this alphabet, shown XXXX-XXXX.
CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'
_CODE = re.compile(f'[{CODE_ALPHABET}]{{8}}')


class MembershipRequired(StoreError):
    """The service denied a download: no active membership (``acquisition_denied``)."""


class VerificationUnavailable(StoreError):
    """The service could not verify the membership recently (``verification_required``)."""


class PatreonRefusal(StoreError):
    """A refusal the person can act on. ``code`` names it for the interface."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


MEMBERSHIP_REQUIRED = ('Your Patreon membership is not active for this download. Join on Patreon, or connect '
                       'Patreon again if you just joined, then retry. Your installed files are kept.')
VERIFICATION_UNAVAILABLE = ('Patreon could not verify your membership right now. Members-only plugins require '
                            'verification; your files and the rest of LDS remain available.')
MALFORMED_CODE = 'Enter the 8-character code shown on the Patreon page, for example ABCD-2345.'
# The named errors of POST /v1/patreon/confirm.
CONFIRM_ERRORS = {
    'invalid_code': 'This code does not match the one shown on the Patreon page. Check it and try again.',
    'code_expired': 'This code has expired. Press Connect Patreon again to get a new one.',
    'too_many_attempts': 'Too many incorrect codes. Press Connect Patreon again to get a new code.',
    'no_pending_link': ('No Patreon approval is waiting for this installation. Press Connect Patreon, approve '
                        'LDS on Patreon, then enter the code shown there.'),
    'link_limit': ('This Patreon membership has connected the maximum number of new installations for the '
                   'last 30 days. Try again later.'),
}


@dataclass(frozen=True)
class PatreonConfig:
    service_url: str
    checkout_origin: str
    join_url: str | None
    root: bytes | None
    first_party_ids: frozenset
    plugin_ids: frozenset
    # Publishers whose external identifiers the catalog serves without a list:
    # a members-only plugin published later appears with no core update.
    plugin_namespaces: frozenset = frozenset()

    @property
    def commerce(self):
        return {'service_url': self.service_url, 'checkout_origin': self.checkout_origin}


def _url(value, *, maximum=2048):
    """An absolute HTTPS address without credentials, port, fragment or controls."""
    if (not isinstance(value, str) or not value or len(value) > maximum or '\\' in value
            or any(ord(c) < 33 or ord(c) == 127 for c in value)):
        raise ValueError('invalid address')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or ':' in parsed.netloc or parsed.fragment):
        raise ValueError('invalid address')
    return parsed


def join_page_url(value):
    if _url(value).netloc != 'www.patreon.com':
        raise ValueError('join address outside Patreon')
    return value


def request_page_url(value):
    _url(value)
    return value


def authorize_page_url(value):
    parsed = _url(value, maximum=4096)
    if (not value.startswith(AUTHORIZE_PREFIX) or parsed.netloc != 'www.patreon.com'
            or parsed.path != '/oauth2/authorize' or not parsed.query):
        raise ValueError('authorization address outside Patreon')
    return value


def _utc(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.utcoffset() != timedelta(0):
            raise ValueError
        return parsed
    except (ValueError, TypeError, AttributeError, OverflowError):
        raise StoreError('The Patreon service returned an invalid date.') from None


def _ids(value, *, first_party):
    if not isinstance(value, list) or len(value) > 100 or len(set(value)) != len(value):
        raise ValueError('invalid plugin scope')
    for pid in value:
        if not isinstance(pid, str) or (pid not in OFFICIAL_IDS if first_party else not EXTERNAL_ID.fullmatch(pid)):
            raise ValueError('invalid plugin scope')
    return frozenset(value)


def _namespaces(value):
    if (not isinstance(value, list) or len(value) > 20 or len(set(value)) != len(value)
            or any(not valid_namespace(name) for name in value)):
        raise ValueError('invalid plugin namespaces')
    return frozenset(value)


def _root(catalog, directory):
    if 'root' in catalog and 'root_path' in catalog:
        raise ValueError('ambiguous Patreon catalog root')
    if catalog.get('root') is not None:
        if not isinstance(catalog['root'], dict):
            raise ValueError('invalid Patreon catalog root')
        data = json.dumps(catalog['root'], sort_keys=True, separators=(',', ':')).encode('utf-8')
    elif catalog.get('root_path') is not None:
        if not isinstance(catalog['root_path'], str) or not catalog['root_path']:
            raise ValueError('invalid Patreon catalog root')
        path = Path(catalog['root_path'])
        data = (path if path.is_absolute() else directory / path).read_bytes()
    else:
        return None  # The catalog is not connected until its bootstrap root ships.
    if not 0 < len(data) <= MAX_ROOT_BYTES:
        raise ValueError('invalid Patreon catalog root')
    return data


def load_patreon_config():
    path = Path(os.environ.get(CONFIG_ENV) or cfg.REPO_ROOT / 'store' / 'patreon.json')
    if not path.is_file():
        raise StoreNotConfigured('Patreon access is not available on this installation.')
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise ValueError('configuration too large')
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('format') != CONFIG_FORMAT:
            raise ValueError('unsupported Patreon configuration')
        service = _origin(data['service_url'])
        checkout = _origin(data['checkout_origin'])
        if checkout != PATREON_ORIGIN:
            raise ValueError('the Patreon origin is pinned')
        join = data.get('join_url')
        join = None if join is None else join_page_url(join)
        catalog = data.get('catalog', {})
        if not isinstance(catalog, dict):
            raise ValueError('invalid Patreon catalog')
        config = PatreonConfig(service, checkout, join, _root(catalog, path.parent),
                               _ids(catalog.get('first_party_ids', []), first_party=True),
                               _ids(catalog.get('plugin_ids', []), first_party=False),
                               _namespaces(catalog.get('plugin_namespaces', [])))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StoreError('The Patreon access configuration is invalid.') from exc
    # RFC 2606 reserves .invalid: the shipped placeholder can never resolve.
    if (urlsplit(service).hostname or '').endswith('.invalid'):
        raise StoreNotConfigured('Patreon access is not connected on this installation yet.')
    return config


def _store_config(config, first_party, external, claimed=frozenset()):
    if config.root is None:
        raise StoreNotConfigured('The Patreon catalog is not connected on this installation yet.')
    try:
        primary = load_config().official_ids
    except StoreNotConfigured:
        primary = frozenset()
    if first_party & primary:
        raise StoreError('The Patreon catalog cannot replace primary first-party products.')
    selected = first_party | external
    namespaces = config.plugin_namespaces
    return StoreConfig(config.service_url + '/tuf/metadata/', config.service_url + '/tuf/targets/',
                       config.root, first_party, external, selected, (), provider='patreon',
                       external_namespaces=namespaces,
                       excluded_ids=frozenset(pid for pid in claimed if namespace_of(pid) in namespaces))


def patreon_plugin_source(claimed=frozenset()):
    """Plugin source for the configured identifiers and namespaces, minus the
    identifiers an operator source claims."""
    config = load_patreon_config()
    first_party, external = config.first_party_ids - claimed, config.plugin_ids - claimed
    if not first_party and not external and not config.plugin_namespaces:
        return None
    return _store_config(config, first_party, external, claimed)


def patreon_core_source():
    """Same signed repository, read for ``core-releases.json``."""
    config = load_patreon_config()
    return _store_config(config, config.first_party_ids, config.plugin_ids)


def confirmation_code(value):
    """The code typed by the person, as the service compares it: upper case,
    without the dash or spaces. A malformed code never reaches the service,
    where it would spend one of the five attempts."""
    if isinstance(value, str) and len(value) <= 32:
        code = re.sub(r'[\s-]', '', value).upper()
        if _CODE.fullmatch(code):
            return code
    raise PatreonRefusal(MALFORMED_CODE, 'malformed_code')


def parse_membership(value):
    """Strict reading of the library ``patreon`` object (PATREON-PROTOCOL v1.1).

    ``tier`` and ``valid_until`` are set only while the membership is active.
    ``valid_until`` is Patreon's next charge date: an indication for display,
    never an expiry the client enforces. ``pending``: an approval on Patreon
    waits for its code here. ``reason``: ``free_trial`` explains an inactive
    membership (free trials grant nothing unless the operator allows them)."""
    try:
        if not isinstance(value, dict):
            raise ValueError
        linked, status, tier = value.get('linked'), value.get('status'), value.get('tier')
        valid_until, pending, reason = value.get('valid_until'), value.get('pending'), value.get('reason')
        active = status == 'active'
        if (type(linked) is not bool or status not in STATUSES or (active and (not linked or tier not in TIERS))
                or (not active and (tier is not None or valid_until is not None))
                or type(pending) is not bool or reason not in REASONS):
            raise ValueError
        if valid_until is not None:
            _utc(valid_until)
        join = join_page_url(value.get('join_url'))
        request_url = value.get('request_url')
        if tier == 'pro':
            request_page_url(request_url)
        elif request_url is not None:
            raise ValueError
    except (ValueError, TypeError):
        raise StoreError('The Patreon membership response is invalid.') from None
    return {'linked': linked, 'status': status, 'tier': tier, 'valid_until': valid_until,
            'join_url': join, 'request_url': request_url, 'pending': pending, 'reason': reason}


class PatreonClient(CommerceClient):
    """The acquisition client bound to the Patreon service of ``store/patreon.json``."""

    def __init__(self, config=None):
        self.patreon = config or load_patreon_config()
        super().__init__(self.patreon.commerce)

    def download_target(self, *args, **kwargs):
        """Name the two refusals the service distinguishes; others stay generic."""
        try:
            return super().download_target(*args, **kwargs)
        except ServiceRefused as exc:
            if exc.code == 'acquisition_denied':
                raise MembershipRequired(MEMBERSHIP_REQUIRED) from None
            if exc.code == 'verification_required':
                raise VerificationUnavailable(VERIFICATION_UNAVAILABLE) from None
            raise

    def unlinked(self):
        return {'linked': False, 'status': 'inactive', 'tier': None, 'valid_until': None,
                'join_url': self.patreon.join_url, 'request_url': None, 'pending': False, 'reason': None}

    def library(self):
        credential = self.credential()
        if credential is None:
            # Reading never enrolls. Connect enrolls, then opens Patreon.
            return {'status': 'not_connected', 'items': [], 'patreon': self.unlinked()}
        value = self._json('GET', '/v1/library', credential=credential)
        items = value.get('items')
        if not isinstance(items, list) or len(items) > 1000:
            raise StoreError('The Patreon library response is invalid.')
        return {'status': 'ready', 'items': [self._item(item) for item in items],
                'patreon': parse_membership(value.get('patreon'))}

    def link(self):
        value = self._json('POST', '/v1/patreon/link', data={}, credential=self.credential(enroll=True))
        try:
            url = authorize_page_url(value.get('authorize_url'))
        except ValueError:
            raise StoreError('The Patreon authorization address is invalid.') from None
        remaining = (_utc(value.get('expires_at')) - datetime.now(timezone.utc)).total_seconds()
        if not -CLOCK_SKEW_SECONDS < remaining <= LINK_LIFETIME_SECONDS + CLOCK_SKEW_SECONDS:
            raise StoreError('The Patreon authorization link has an invalid expiry.')
        return {'authorize_url': url, 'expires_at': value['expires_at']}

    def confirm(self, code):
        """Finish linking with the code the Patreon page shows (v1.1). Only
        this installation is linked, whoever approved on Patreon."""
        code = confirmation_code(code)
        credential = self.credential()
        if credential is None:
            # Connect enrolls this installation; without it nothing can be pending.
            raise PatreonRefusal(CONFIRM_ERRORS['no_pending_link'], 'no_pending_link')
        try:
            value = self._json('POST', '/v1/patreon/confirm', data={'code': code}, credential=credential)
        except ServiceRefused as exc:
            if exc.code in CONFIRM_ERRORS:
                raise PatreonRefusal(CONFIRM_ERRORS[exc.code], exc.code) from None
            raise
        if value.get('status') != 'linked':
            raise StoreError('The Patreon service did not confirm the connection.')
        return {'status': 'linked'}

    def unlink(self):
        credential = self.credential()
        if credential is None:
            return {'status': 'unlinked'}
        value = self._json('POST', '/v1/patreon/unlink', data={}, credential=credential)
        if value.get('status') != 'unlinked':
            raise StoreError('The Patreon service did not confirm the disconnection.')
        return {'status': 'unlinked'}


def access_summary(client=None):
    """Membership for the UI, without plugin items: ``{status, patreon}``."""
    library = (client or PatreonClient()).library()
    return {'status': library['status'], 'patreon': library['patreon']}


def register_patreon_routes(bp):
    """Same administration and host CSRF as the commerce routes for every
    change. Reading the membership needs no administration, so a browser
    behind Docker's bridge or on the LAN still sees it (``can_manage`` false)."""
    def invoke(operation, *, administration=True):
        if administration and (denied := require_admin()) is not None:
            return denied
        try:
            from ..membership import current_access
            access = current_access()
            with access.lock:
                result = operation(PatreonClient())
                if 'patreon' in result:
                    access.observe(result)
            response = jsonify({**result, 'can_manage': is_admin()})
            response.headers['Cache-Control'] = 'no-store'
            return response
        except StoreNotConfigured as exc:
            from ..membership import current_access
            current_access().observe({'status': 'not_configured'})
            return jsonify(status='not_configured', message=str(exc), patreon=None, can_manage=is_admin()), 200
        except PatreonRefusal as exc:
            return jsonify(error=str(exc), code=exc.code, status='refused'), 409
        except (StoreError, OSError):
            from ..membership import current_access
            current_access().observe({'status': 'unavailable'})
            return jsonify(error='Patreon could not be reached or refused the request. Your files are kept; '
                                 'members-only plugins need an active verified membership.', status='unavailable', patreon=None,
                           can_manage=is_admin()), 409

    def changed(result):
        from ..membership import current_access
        access = current_access()
        access.forget()
        if result.get('status') == 'unlinked':
            access.observe({'status': 'not_connected'})
        from ...services import patreon_core
        patreon_core.forget_status()
        return result

    def empty_body():
        if request.get_json(silent=True) != {}:
            raise StoreError('The Patreon request is invalid.')

    @bp.get('/patreon')
    def patreon_status():
        return invoke(access_summary, administration=False)

    @bp.post('/patreon/link')
    def patreon_link():
        def link(client):
            empty_body()
            return changed(client.link())
        return invoke(link)

    @bp.post('/patreon/confirm')
    def patreon_confirm():
        def confirm(client):
            value = request.get_json(silent=True)
            if not isinstance(value, dict) or set(value) != {'code'}:
                raise StoreError('The Patreon request is invalid.')
            return changed(client.confirm(value['code']))
        return invoke(confirm)

    @bp.post('/patreon/unlink')
    def patreon_unlink():
        def unlink(client):
            empty_body()
            return changed(client.unlink())
        return invoke(unlink)
