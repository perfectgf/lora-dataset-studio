"""Members previews: what an installation without an active, linked Patreon
membership shows of the members features: a name, one sentence and a link to
Patreon. Text only. No code, flag or route of a members feature comes with a
preview, and the interface opens nothing but the Patreon page.

The list is signed content: the top-level ``member_previews`` key of
``catalog.json`` in the Patreon catalog (the TUF repository that
``store/patreon.json`` pins), written by the operator tool's ``set-previews``.
It changes without an app release; every catalog reader since the 2026.10.18
gateway ignores that key (``catalog.parse_catalog``).

An active, linked membership gets no previews, and neither does a membership
that cannot be verified right now. Without a readable signed list (offline, no
Patreon configuration, an expired or invalid catalog) nothing is shown and
nothing is said.
"""
from __future__ import annotations

import logging
import re
import threading
import time

from flask import Blueprint, jsonify

from .client import StoreError, StoreSession
from .patreon import join_page_url, load_patreon_config, patreon_core_source

log = logging.getLogger(__name__)

FORMAT = 'lds-member-previews/1'
CATALOG_KEY = 'member_previews'
MAX_ITEMS = 40
MAX_NAME = 40
MAX_SUMMARY = 160
MAX_URL = 2048
TTL_SECONDS = 600
RETRY_SECONDS = 60
ITEM_KEYS = frozenset({'id', 'name', 'summary', 'placement', 'url'})
REQUIRED_KEYS = frozenset({'id', 'name', 'summary', 'placement'})
_ID = re.compile(r'[a-z][a-z0-9_]{1,31}(?:\.[a-z][a-z0-9_]{1,31})?\Z')
_PLACEMENT = re.compile(r'[a-z][a-z0-9_]{0,23}(?:\.[a-z0-9_]{1,24}){0,3}\Z')
# Where this version shows previews. A newer catalog may name more places:
# this version leaves those entries out, one by one.
PLACEMENTS = frozenset({'studio.prompt', 'studio.options', 'training.options',
                        'dataset.crop', 'bank.crop', 'watermark.engines'})
# Test Studio options of one model family: studio.options.krea, studio.options.moss…
_FAMILY_PLACEMENT = re.compile(r'studio\.options\.[a-z][a-z0-9]{1,23}\Z')
# Controls, line separators and bidirectional overrides: a preview is plain text.
_UNSAFE = ((0x00, 0x1F), (0x7F, 0x9F), (0x2028, 0x2029), (0x202A, 0x202E), (0x2066, 0x2069))


def _plain(value, maximum):
    if not isinstance(value, str):
        raise ValueError
    text = value.strip()
    if (not text or len(text) > maximum
            or any(low <= ord(char) <= high for char in text for low, high in _UNSAFE)):
        raise ValueError
    return text


def known_placement(value):
    return value in PLACEMENTS or bool(_FAMILY_PLACEMENT.fullmatch(value))


def parse_member_previews(value):
    """Strict ``lds-member-previews/1``. Any invalid entry refuses the whole
    list, like the core release list; an entry for a place this version does
    not show (written for a newer app) is left out on its own."""
    try:
        if not isinstance(value, dict) or set(value) != {'format', 'items'} or value['format'] != FORMAT:
            raise ValueError
        items = value['items']
        if not isinstance(items, list) or len(items) > MAX_ITEMS:
            raise ValueError
        result, seen = [], set()
        for item in items:
            if not isinstance(item, dict) or not REQUIRED_KEYS <= set(item) or not set(item) <= ITEM_KEYS:
                raise ValueError
            ident, placement = item['id'], item['placement']
            if (not isinstance(ident, str) or not _ID.fullmatch(ident)
                    or not isinstance(placement, str) or not _PLACEMENT.fullmatch(placement)):
                raise ValueError
            preview = {'id': ident, 'name': _plain(item['name'], MAX_NAME),
                       'summary': _plain(item['summary'], MAX_SUMMARY), 'placement': placement}
            if 'url' in item:
                if not isinstance(item['url'], str) or len(item['url']) > MAX_URL:
                    raise ValueError
                preview['url'] = join_page_url(item['url'])
            if (placement, ident) in seen:
                raise ValueError
            seen.add((placement, ident))
            if known_placement(placement):
                result.append(preview)
    except (ValueError, TypeError, KeyError):
        raise StoreError('The members preview list is invalid.') from None
    return tuple(result)


_cache = {'until': 0.0, 'items': ()}
_lock = threading.Lock()


def forget():
    _cache.update(until=0.0, items=())


def _read_catalog():
    with StoreSession(patreon_core_source()) as session:
        document = session.catalog()
    value = document.get(CATALOG_KEY) if isinstance(document, dict) else None
    return () if value is None else parse_member_previews(value)


def catalog_previews():
    """The signed list, kept 10 minutes. ``()`` when it cannot be read, asked
    again after a minute, so an offline installation is not slowed down."""
    with _lock:
        now = time.monotonic()
        if now < _cache['until']:
            return _cache['items']
        try:
            items, keep = _read_catalog(), TTL_SECONDS
        # Best effort by design: whatever fails (network, TUF, configuration,
        # format), the screens simply show no preview.
        except Exception as exc:
            log.info('Members previews unavailable: %s', exc)
            items, keep = (), RETRY_SECONDS
        _cache.update(until=now + keep, items=items)
        return items


def visible_previews(access):
    """What this installation shows. Nothing for an active, linked membership
    or one that cannot be verified now (``membership_unavailable``); otherwise
    every signed preview, each with its Patreon link."""
    if access.get('code') != 'membership_required':
        return []
    items = catalog_previews()
    if not items:
        return []
    try:
        join = load_patreon_config().join_url
    except StoreError:
        join = None
    return [{**item, 'url': item.get('url') or join} for item in items if item.get('url') or join]


bp = Blueprint('member_previews', __name__)


@bp.get('/api/member-previews')
def member_previews_status():
    from ..membership import current_access
    response = jsonify({'format': FORMAT, 'items': visible_previews(current_access().status())})
    response.headers['Cache-Control'] = 'no-store'
    return response
