"""Membership revocation closes existing handlers, including cached library reads."""
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

from app.plugins import hooks, membership, routes
from app.plugins.loader import load_plugins
from app.plugins.store import patreon
from app.plugins.store.client import StoreError
from test_plugin_registration import host, write_plugin  # noqa: F401

ACTIVE = {'status': 'ready', 'patreon': {'linked': True, 'status': 'active'}}
INACTIVE = {'status': 'ready', 'patreon': {'linked': False, 'status': 'inactive'}}
CODE = '''
from flask import Blueprint, current_app
bp = Blueprint('member_feature', __name__)
@bp.before_request
def before():
    current_app.extensions['entered'] = current_app.extensions.get('entered', 0) + 1
@bp.app_url_value_preprocessor
def early(*args):
    current_app.extensions['early'] = current_app.extensions.get('early', 0) + 1
@bp.route('/cached', methods=['GET', 'POST'])
def cached():
    return {'prompt': 'cached member content'}
def register(ctx):
    ctx.register_blueprint(bp, url_prefix='/api/member-alias')
    second = Blueprint('member_second', __name__)
    second.add_url_rule('/summary', 'summary', cached)
    ctx.register_blueprint(second)
    ctx.register_hook('caption.stamp', lambda text, img: text + ' member')
    ctx.register_hook('plugin.disable_blockers', lambda value, pid: value + ['work still running'])
'''


@pytest.fixture
def member_host(host, monkeypatch, request):
    app, csrf, root = host
    pid = getattr(request, 'param', 'patreon.feature')
    write_plugin(root / 'plugins', pid, package='lds_registration_member', code=CODE)
    app.register_blueprint(routes.bp)
    app.add_url_rule('/api/core', 'core', lambda: {'core': True})
    account = deepcopy(ACTIVE)

    class Client:
        def library(self):
            return deepcopy(account)

        def unlink(self):
            account.update(deepcopy(INACTIVE))
            return {'status': 'unlinked'}

        def confirm(self, code):
            account.update(deepcopy(ACTIVE))
            return {'status': 'linked'}

    monkeypatch.setattr(patreon, 'PatreonClient', Client)
    registry = load_plugins(app, csrf)
    return app, registry, account, pid


@pytest.mark.parametrize('member_host', ['patreon.feature', 'local.creative_library'], indirect=True)
@pytest.mark.parametrize('method', ['GET', 'POST'])
def test_unlink_immediately_blocks_cached_reads_aliases_and_callbacks_then_reconnects(member_host, method):
    app, registry, _, pid = member_host
    client = app.test_client()
    assert client.open('/api/member-alias/cached', method=method).json['prompt'] == 'cached member content'
    assert client.post('/api/plugins/store/patreon/unlink', json={}).status_code == 200
    entered, early = app.extensions['entered'], app.extensions['early']
    for url, verb in [('/api/member-alias/cached', method), (f'/api/plugins/{pid}/summary', 'GET')]:
        response = client.open(url, method=verb)
        assert response.status_code == 403
        assert response.json['code'] == 'membership_required'
        assert response.headers['Cache-Control'] == 'no-store'
        assert 'cached member content' not in response.text
    assert (app.extensions['entered'], app.extensions['early']) == (entered, early)
    assert client.get('/api/core').json == {'core': True}
    assert client.get('/api/plugins/membership').json['allowed'] is False
    assert registry.records[pid].state == 'loaded' and registry.records[pid].enabled
    assert client.post('/api/plugins/store/patreon/confirm', json={'code': 'ABCD-2345'}).status_code == 200
    assert client.get('/api/member-alias/cached').status_code == 200


def test_remote_revocation_and_failed_verification_do_not_extend_access(member_host, monkeypatch):
    app, registry, account, _ = member_host
    clock = [100.0]
    registry.membership._clock = lambda: clock[0]
    client = app.test_client()
    assert client.get('/api/member-alias/cached').status_code == 200
    account.update(deepcopy(INACTIVE))
    clock[0] += membership.CHECK_SECONDS + 1
    assert client.get('/api/member-alias/cached').status_code == 403
    account.update(deepcopy(ACTIVE))
    clock[0] += membership.CHECK_SECONDS + 1
    assert client.get('/api/member-alias/cached').status_code == 200

    def unavailable(*args):
        raise StoreError('offline')
    monkeypatch.setattr(patreon, 'access_summary', unavailable)
    clock[0] += membership.CHECK_SECONDS + 1
    assert client.get('/api/member-alias/cached').status_code == 503
    assert client.get('/api/core').status_code == 200


def test_membership_status_observation_revokes_runtime_cache(member_host):
    app, _, account, _ = member_host
    client = app.test_client()
    assert client.get('/api/member-alias/cached').status_code == 200
    account.update(deepcopy(INACTIVE))
    assert client.get('/api/plugins/store/patreon').json['patreon']['linked'] is False
    assert client.get('/api/member-alias/cached').status_code == 403


def test_member_features_stop_but_cleanup_hooks_remain_available(member_host):
    app, _, _, pid = member_host
    with app.app_context():
        assert hooks.run_filter('caption.stamp', 'caption', None) == 'caption member'
    app.test_client().post('/api/plugins/store/patreon/unlink', json={})
    with app.app_context():
        assert hooks.run_filter('caption.stamp', 'caption', None) == 'caption'
        assert hooks.run_filter('plugin.disable_blockers', [], pid) == ['work still running']


def test_public_plugins_never_request_patreon(monkeypatch):
    access = membership.MemberAccess()
    monkeypatch.setattr(access, 'status', lambda: pytest.fail('public plugin checked Patreon'))
    assert access.check(SimpleNamespace(id='video'))['allowed']
    assert access.check(SimpleNamespace(id='local.example'))['allowed']


def test_plugin_listing_and_public_requests_never_wait_for_patreon(member_host, monkeypatch):
    app, registry, _, pid = member_host
    monkeypatch.setattr(patreon, 'access_summary', lambda: pytest.fail('public request checked Patreon'))
    assert app.test_client().get('/api/core').status_code == 200
    value = membership.payload(registry)
    assert value['plugin_ids'] == [pid] and value['allowed'] is False


def test_delayed_active_refresh_cannot_restore_a_completed_unlink(member_host, monkeypatch):
    app, registry, _, _ = member_host
    started, release = threading.Event(), threading.Event()

    def slow_summary():
        started.set()
        assert release.wait(3)
        return deepcopy(ACTIVE)
    monkeypatch.setattr(patreon, 'access_summary', slow_summary)
    with_threads = []
    refresh = threading.Thread(target=lambda: with_threads.append(registry.membership.status()))
    unlink = threading.Thread(target=lambda: with_threads.append(
        app.test_client().post('/api/plugins/store/patreon/unlink', json={}).status_code))
    refresh.start()
    assert started.wait(3)
    unlink.start()
    release.set()
    refresh.join(3)
    unlink.join(3)
    assert not refresh.is_alive() and not unlink.is_alive()
    assert 200 in with_threads
    assert registry.membership.status()['allowed'] is False
