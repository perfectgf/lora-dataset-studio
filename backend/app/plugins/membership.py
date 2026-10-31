"""Runtime admission for members-only plugins, separate from installation state."""
from __future__ import annotations

import threading
import time

from flask import current_app, jsonify


CHECK_SECONDS = 15
REQUIRED = 'Connect an active Patreon membership in Plugins to use this members-only plugin. Your files are kept.'
UNAVAILABLE = 'Patreon membership could not be verified. Retry from Plugins. Your files are kept.'


def members_scope():
    """Identifiers and namespaces served by the configured members catalog."""
    try:
        from .store.patreon import load_patreon_config
        config = load_patreon_config()
    except (ImportError, ValueError):
        return frozenset(), frozenset({'patreon'})
    return config.first_party_ids | config.plugin_ids, config.plugin_namespaces | {'patreon'}


def requires_membership(record):
    if record is None:
        return False
    ids, namespaces = members_scope()
    # Keep the existing library protected even if the operator configuration is missing.
    return (record.id == 'local.creative_library' or record.id in ids
            or record.id.split('.', 1)[0] in namespaces)


class MemberAccess:
    def __init__(self):
        self.lock = threading.RLock()
        self._clock = time.monotonic
        self._value = None
        self._until = 0
        self._required = {}

    def required(self, record):
        if record is None:
            return False
        if record.id not in self._required:
            self._required[record.id] = requires_membership(record)
        return self._required[record.id]

    def forget(self):
        with self.lock:
            self._value, self._until = None, 0

    def observe(self, summary):
        with self.lock:
            member = summary.get('patreon') or {}
            allowed = (summary.get('status') == 'ready' and member.get('linked') is True
                       and member.get('status') == 'active')
            unavailable = summary.get('status') == 'unavailable' or member.get('status') == 'verification_required'
            self._value = {'allowed': allowed, 'code': None if allowed else (
                'membership_unavailable' if unavailable else 'membership_required'),
                'message': '' if allowed else UNAVAILABLE if unavailable else REQUIRED}
            self._until = self._clock() + CHECK_SECONDS
            return dict(self._value)

    def status(self):
        # Serialize refreshes with unlink/confirm: a delayed active response cannot
        # overwrite a completed disconnection. An outage never extends an old grant.
        with self.lock:
            if self._value is not None and self._clock() < self._until:
                return dict(self._value)
            from .store.patreon import access_summary
            from .store.client import StoreError, StoreNotConfigured
            try:
                summary = access_summary()
            except StoreNotConfigured:
                summary = {'status': 'not_configured'}
            except (StoreError, OSError):
                summary = {'status': 'unavailable'}
            return self.observe(summary)

    def snapshot(self):
        # The main plugin list must remain fast even if Patreon is unreachable.
        # Its separate membership endpoint refreshes access after the app mounts.
        if self.lock.acquire(blocking=False):
            try:
                if self._value is not None and self._clock() < self._until:
                    return dict(self._value)
            finally:
                self.lock.release()
        return {'allowed': False, 'code': 'membership_unavailable', 'message': UNAVAILABLE}

    def check(self, record, *, refresh=True):
        if not self.required(record):
            return {'allowed': True, 'code': None, 'message': ''}
        return self.status() if refresh else self.snapshot()


def current_access():
    registry = current_app.extensions.get('lds_plugins')
    if registry is not None:
        return registry.membership
    return current_app.extensions.setdefault('lds_membership', MemberAccess())


def payload(registry, *, refresh=False):
    ids = sorted(record.id for record in registry.records.values() if registry.membership.required(record)) if registry else []
    status = (registry.membership.status() if refresh else registry.membership.snapshot()) if ids else {
        'allowed': True, 'code': None, 'message': ''}
    return {'plugin_ids': ids, **status}


def refusal(access):
    response = jsonify(ok=False, code=access['code'], error=access['message'])
    response.headers['Cache-Control'] = 'no-store'
    return response, 503 if access['code'] == 'membership_unavailable' else 403
