"""Patreon access client and routes: shipped configuration, strict reading of
the membership object, pinned Patreon addresses, administration + CSRF, and
the routing of Patreon products to the Patreon service. No network."""
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from flask import Blueprint, Flask, jsonify
from flask_wtf import CSRFProtect
from flask_wtf.csrf import generate_csrf
import pytest

from app.plugins.store import commerce, patreon, service
from app.plugins.store.catalog import members_releases, parse_catalog
from app.plugins.store.client import StoreConfig, StoreError, StoreNotConfigured, patreon_source
from public_store_fixtures import contract

SECRET = 's' * 43
CLIENT_ID = 'a' * 32
SERVICE = 'https://patreon-service.example.test'
JOIN = 'https://www.patreon.com/c/example'
MEMBER = {'linked': True, 'status': 'active', 'tier': 'member', 'valid_until': '2026-11-04T00:00:00Z',
          'join_url': JOIN, 'request_url': None, 'pending': False, 'reason': None}
PRO = {**MEMBER, 'tier': 'pro', 'request_url': 'https://requests.example.test/custom-plugin'}
CONFIG = patreon.PatreonConfig(SERVICE, 'https://www.patreon.com', JOIN, None, frozenset(), frozenset())


class Response:
    def __init__(self, body, status=200):
        self.body = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        self.status_code = status

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]

    def close(self):
        pass


@pytest.fixture
def remote(tmp_path, monkeypatch):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setattr(patreon, 'load_patreon_config', lambda: CONFIG)
    replies, calls = [], []

    class Session:
        trust_env = True

        def request(self, method, url, **options):
            assert self.trust_env is False and options['allow_redirects'] is False
            calls.append((method, url, options))
            return replies.pop(0) if replies else Response({'error': 'unavailable'}, 503)

        def close(self):
            pass
    monkeypatch.setattr(commerce.requests, 'Session', Session)
    return replies, calls


def connected(remote):
    remote[0].append(Response({'client_id': CLIENT_ID, 'client_secret': SECRET}, 201))
    client = patreon.PatreonClient()
    client.credential(enroll=True)
    return client


def write_config(tmp_path, monkeypatch, **over):
    value = {'format': 'lds-patreon/1', 'service_url': 'https://store.example.test',
             'checkout_origin': 'https://www.patreon.com', 'join_url': JOIN,
             'catalog': {'root': {'signed': {'_type': 'root'}}, 'first_party_ids': ['creature_battle', 'manga'],
                         'plugin_ids': []}}
    value.update(over)
    path = tmp_path / 'patreon-test.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    monkeypatch.setenv(patreon.CONFIG_ENV, str(path))
    return path


# --- Configuration -------------------------------------------------------------

def test_shipped_configuration_points_to_the_real_service(monkeypatch):
    monkeypatch.delenv(patreon.CONFIG_ENV, raising=False)
    shipped = json.loads((patreon.cfg.REPO_ROOT / 'store' / 'patreon.json').read_text(encoding='utf-8'))
    assert shipped['service_url'] == 'https://lds.perfectgf.fr'
    assert shipped['checkout_origin'] == 'https://www.patreon.com'
    # Installations that stop updating keep this file: later members-only
    # plugins reach them only through the namespace it ships.
    assert shipped['catalog']['plugin_namespaces'] == ['patreon']
    assert 'local.creative_library' in shipped['catalog']['plugin_ids']
    loaded = patreon.load_patreon_config()
    assert loaded.service_url == 'https://lds.perfectgf.fr'
    assert json.loads(loaded.root)['signed']['_type'] == 'root'


@pytest.mark.parametrize('url,ok', [('http://127.0.0.1:8765', True), ('https://store.example.test', True),
                                    ('http://localhost:8765', False), ('http://192.0.2.10:8765', False),
                                    ('https://store.example.test/api', False),
                                    ('https://user:pass@store.example.test', False)])
def test_test_instances_may_use_only_loopback_http(tmp_path, monkeypatch, url, ok):
    write_config(tmp_path, monkeypatch, service_url=url)
    if ok:
        assert patreon.load_patreon_config().service_url == url
    else:
        with pytest.raises(StoreError):
            patreon.load_patreon_config()


@pytest.mark.parametrize('over', [
    {'checkout_origin': 'https://checkout.example.test'},
    {'join_url': 'https://patreon.example.test/c/example'},
    {'join_url': 'http://www.patreon.com/c/example'},
    {'format': 'lds-patreon/2'},
    {'catalog': {'first_party_ids': ['not_an_lds_product']}},
    {'catalog': {'plugin_ids': ['missing_publisher']}},
    {'catalog': {'root': {}, 'root_path': 'root.json'}},
    # Namespaces: the publisher part of EXTERNAL_ID, never one the code reserves.
    {'catalog': {'plugin_namespaces': 'patreon'}},
    {'catalog': {'plugin_namespaces': ['patreon', 'patreon']}},
    {'catalog': {'plugin_namespaces': ['patreon.tools']}},
    {'catalog': {'plugin_namespaces': ['Patreon']}},
    {'catalog': {'plugin_namespaces': ['lds']}},
    {'catalog': {'plugin_namespaces': ['con']}},
])
def test_configuration_pins_patreon_and_an_explicit_scope(tmp_path, monkeypatch, over):
    write_config(tmp_path, monkeypatch, **over)
    with pytest.raises(StoreError):
        patreon.load_patreon_config()


def test_catalog_needs_its_root_and_cannot_replace_public_products(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, catalog={'root': None, 'first_party_ids': ['manga']})
    with pytest.raises(StoreNotConfigured):
        patreon.patreon_core_source()
    write_config(tmp_path, monkeypatch, catalog={'root': {'x': 1}, 'first_party_ids': ['video']})
    with pytest.raises(StoreError, match='cannot replace'):
        patreon.patreon_core_source()


def test_operator_sources_keep_the_identifiers_they_claim(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch)
    private = SimpleNamespace(selected_ids=frozenset({'manga'}))
    source = patreon_source([private])
    assert source.provider == 'patreon' and source.selected_ids == {'creature_battle'}
    assert source.metadata_url == 'https://store.example.test/tuf/metadata/'
    assert source.target_url == 'https://store.example.test/tuf/targets/'
    assert patreon_source([SimpleNamespace(selected_ids=frozenset({'manga', 'creature_battle'}))]) is None


def test_only_the_patreon_source_can_label_patreon_products():
    def catalog(price):
        data = contract(id='manga', name='Manga')
        return {'schema_version': 1, 'products': [{'id': 'manga', 'releases': [{
            'manifest': data, 'manifest_sha256': 'a' * 64, 'target': 'manga/1.0.0.ldsplugin', 'price': price}]}]}
    root = b'{}'
    member_price = {'kind': 'paid', 'product': 'patreon_lds', 'provider': 'patreon'}
    patreon_config = StoreConfig('https://s.example.test/m/', 'https://s.example.test/t/', root,
                                 frozenset({'manga'}), frozenset(), frozenset({'manga'}), (), provider='patreon')
    other = StoreConfig('https://s.example.test/m/', 'https://s.example.test/t/', root,
                        frozenset({'manga'}), frozenset(), frozenset({'manga'}))
    assert parse_catalog(catalog(member_price), patreon_config)['manga'][0].price == member_price
    for price in ({'kind': 'free'}, {'kind': 'paid', 'product': 'patreon_lds'}):
        with pytest.raises(StoreError):
            parse_catalog(catalog(price), patreon_config)
    with pytest.raises(StoreError):
        parse_catalog(catalog(member_price), other)


# --- Membership object and link ---------------------------------------------------

INACTIVE = {'linked': True, 'status': 'inactive', 'tier': None, 'valid_until': None, 'join_url': JOIN,
            'request_url': None, 'pending': False, 'reason': None}


@pytest.mark.parametrize('value', [MEMBER, PRO, INACTIVE, {**INACTIVE, 'status': 'verification_required'},
                                   {**INACTIVE, 'linked': False},
                                   # v1.1: an approval waiting for its code, and a free trial.
                                   {**INACTIVE, 'linked': False, 'pending': True},
                                   {**INACTIVE, 'reason': 'free_trial'}])
def test_valid_membership_objects_are_kept_exactly(value):
    assert patreon.parse_membership(value) == value


@pytest.mark.parametrize('value', [
    None, {**MEMBER, 'linked': 'yes'}, {**MEMBER, 'status': 'paused'}, {**MEMBER, 'tier': 'gold'},
    {**MEMBER, 'tier': None}, {**MEMBER, 'linked': False},
    # Contract: tier and valid_until (next charge date) are null unless active.
    {**MEMBER, 'status': 'inactive'}, {**INACTIVE, 'valid_until': '2026-11-04T00:00:00Z'},
    {**MEMBER, 'status': 'verification_required'},
    {**MEMBER, 'valid_until': '2026-11-04T00:00:00'}, {**MEMBER, 'valid_until': '2026-11-04T00:00:00+02:00'},
    {**MEMBER, 'join_url': 'https://evil.example.test/join'}, {**MEMBER, 'join_url': None},
    {**MEMBER, 'request_url': 'https://requests.example.test/form'},
    {**PRO, 'request_url': None}, {**PRO, 'request_url': 'http://requests.example.test/form'},
    {**PRO, 'request_url': 'https://requests.example.test/form#frag'},
    # v1.1: `pending` is a boolean, `reason` null or "free_trial".
    {**MEMBER, 'pending': None}, {**MEMBER, 'pending': 'false'}, {**MEMBER, 'pending': 0},
    {key: value for key, value in MEMBER.items() if key != 'pending'},
    {**INACTIVE, 'reason': 'trial'}, {**INACTIVE, 'reason': ''}, {**INACTIVE, 'reason': False},
])
def test_membership_object_is_read_strictly(value):
    with pytest.raises(StoreError):
        patreon.parse_membership(value)


def link_reply(url, seconds=600):
    return Response({'authorize_url': url,
                     'expires_at': (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()})


def test_link_enrolls_then_returns_the_pinned_authorization_page(remote):
    url = 'https://www.patreon.com/oauth2/authorize?response_type=code&client_id=x&state=y'
    remote[0].extend([Response({'client_id': CLIENT_ID, 'client_secret': SECRET}, 201), link_reply(url)])
    result = patreon.PatreonClient().link()
    assert result['authorize_url'] == url
    method, address, options = remote[1][-1]
    assert (method, address) == ('POST', SERVICE + '/v1/patreon/link')
    assert options['json'] == {} and options['headers']['Authorization'] == 'Bearer ' + CLIENT_ID + '.' + SECRET


@pytest.mark.parametrize('url', [
    'https://www.patreon.com/oauth2/authorize', 'http://www.patreon.com/oauth2/authorize?x=1',
    'https://patreon.com/oauth2/authorize?x=1', 'https://www.patreon.com.example.test/oauth2/authorize?x=1',
    'https://www.patreon.com:8443/oauth2/authorize?x=1', 'https://user@www.patreon.com/oauth2/authorize?x=1',
    'https://www.patreon.com/oauth2/authorize/../token?x=1', 'https://www.patreon.com/oauth2/authorize?x=1#f',
    'https://www.patreon.com/oauth2/authorize?x=1\\y', 'https://www.patreon.com/oauth2/authorize?x= 1', None])
def test_link_refuses_any_other_authorization_address(remote, url):
    client = connected(remote)
    remote[0].append(link_reply(url))
    with pytest.raises(StoreError):
        client.link()


@pytest.mark.parametrize('seconds,accepted', [
    (600, True), (-5, True),
    (600 + 15 * 60, True),     # this computer's clock 15 minutes behind
    (600 + 2 * 3600, True),    # a clock left on another time zone, behind...
    (600 - 3 * 3600, True),    # ...or ahead
    (3 * 86400, False), (-3 * 86400, False)])
def test_link_expiry_is_informative_and_tolerates_a_skewed_clock(remote, seconds, accepted):
    """`expires_at` only informs: a local clock that lags must not refuse the link."""
    client = connected(remote)
    url = 'https://www.patreon.com/oauth2/authorize?state=y'
    remote[0].append(link_reply(url, seconds))
    if accepted:
        assert client.link()['authorize_url'] == url
    else:
        with pytest.raises(StoreError):
            client.link()


def test_library_never_enrolls_and_unlink_without_link_needs_no_network(remote):
    client = patreon.PatreonClient()
    library = client.library()
    assert library['status'] == 'not_connected' and library['patreon']['join_url'] == JOIN
    assert client.unlink() == {'status': 'unlinked'}
    assert remote[1] == []


def test_unlink_requires_the_service_confirmation(remote):
    client = connected(remote)
    remote[0].append(Response({'status': 'unlinked'}))
    assert client.unlink() == {'status': 'unlinked'}
    assert remote[1][-1][1] == SERVICE + '/v1/patreon/unlink' and remote[1][-1][2]['json'] == {}
    remote[0].append(Response({'status': 'linked'}))
    with pytest.raises(StoreError):
        client.unlink()


def test_library_reads_items_and_membership(remote):
    client = connected(remote)
    item = {'plugin_id': 'manga', 'status': 'active', 'acquisition': 'all_published', 'updates': 'while_active',
            'offline': 'installed_continues', 'valid_until': None}
    remote[0].append(Response({'items': [item], 'patreon': PRO}))
    assert client.library() == {'status': 'ready', 'items': [item], 'patreon': PRO}
    remote[0].append(Response({'items': [item], 'patreon': {**PRO, 'request_url': None}}))
    with pytest.raises(StoreError):
        client.library()


def test_library_accepts_external_plugin_identifiers(remote):
    # Patreon sells external plugins too; their identifiers are publisher.name.
    client = connected(remote)
    external = {'plugin_id': 'studio.member_tool', 'status': 'active', 'acquisition': 'all_published',
                'updates': 'while_active', 'offline': 'installed_continues', 'valid_until': None}
    remote[0].append(Response({'items': [external], 'patreon': PRO}))
    assert client.library()['items'] == [external]
    remote[0].append(Response({'items': [{**external, 'plugin_id': 'studio.member.tool'}], 'patreon': PRO}))
    with pytest.raises(StoreError):
        client.library()


# --- Confirmation code (PATREON-PROTOCOL v1.1) ---------------------------------------

CONFIRM_REFUSALS = [('invalid_code', 403), ('code_expired', 403), ('too_many_attempts', 429),
                    ('no_pending_link', 404), ('link_limit', 429)]


def test_confirm_sends_the_code_of_the_patreon_page(remote):
    client = connected(remote)
    remote[0].append(Response({'status': 'linked'}))
    assert client.confirm(' abcd-2345 ') == {'status': 'linked'}
    method, address, options = remote[1][-1]
    assert (method, address) == ('POST', SERVICE + '/v1/patreon/confirm')
    assert options['json'] == {'code': 'ABCD2345'}
    assert options['headers']['Authorization'] == 'Bearer ' + CLIENT_ID + '.' + SECRET


@pytest.mark.parametrize('code,status', CONFIRM_REFUSALS)
def test_confirm_names_each_refusal(remote, code, status):
    client = connected(remote)
    remote[0].append(Response({'error': code}, status))
    with pytest.raises(patreon.PatreonRefusal) as refused:
        client.confirm('ABCD-2345')
    assert refused.value.code == code
    assert str(refused.value) == patreon.CONFIRM_ERRORS[code]


def test_each_confirmation_refusal_has_its_own_message():
    assert set(patreon.CONFIRM_ERRORS) == {code for code, _ in CONFIRM_REFUSALS}
    assert len(set(patreon.CONFIRM_ERRORS.values())) == len(CONFIRM_REFUSALS)


@pytest.mark.parametrize('code', ['', 'ABCD-234', 'ABCD-23456', 'ABCD-2O45', 'ABCD-1234', 'ABCD_2345',
                                  'X' * 40, None, 12345678])
def test_a_malformed_code_never_spends_an_attempt(remote, code):
    client = connected(remote)
    calls = len(remote[1])
    with pytest.raises(patreon.PatreonRefusal) as refused:
        client.confirm(code)
    assert refused.value.code == 'malformed_code' and len(remote[1]) == calls


def test_confirm_without_a_connection_needs_no_network(remote):
    with pytest.raises(patreon.PatreonRefusal) as refused:
        patreon.PatreonClient().confirm('ABCD-2345')
    assert refused.value.code == 'no_pending_link' and remote[1] == []


def test_confirm_requires_the_service_answer(remote):
    client = connected(remote)
    remote[0].append(Response({'status': 'pending'}))
    with pytest.raises(StoreError):
        client.confirm('ABCD-2345')


# --- Routes ---------------------------------------------------------------------------

def routes_client():
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY='test-session-signing-only')
    CSRFProtect(app)
    bp = Blueprint('patreon_test', __name__, url_prefix='/api/plugins/store')
    patreon.register_patreon_routes(bp)
    app.register_blueprint(bp)
    app.add_url_rule('/csrf', view_func=lambda: jsonify(token=generate_csrf()))
    return app.test_client()


def test_routes_require_admin_and_csrf_and_never_return_credentials(remote):
    client = routes_client()
    url = 'https://www.patreon.com/oauth2/authorize?state=abc'
    remote[0].extend([Response({'client_id': CLIENT_ID, 'client_secret': SECRET}, 201), link_reply(url)])
    assert client.post('/api/plugins/store/patreon/link', json={}).status_code == 400
    token = client.get('/csrf').json['token']
    # Reading needs no administration (Docker bridge, LAN browser); every change does.
    remote_read = client.get('/api/plugins/store/patreon', environ_base={'REMOTE_ADDR': '198.51.100.8'})
    assert remote_read.status_code == 200 and remote_read.json['can_manage'] is False
    assert remote_read.json['patreon']['linked'] is False
    assert client.post('/api/plugins/store/patreon/link', json={}, headers={'X-CSRFToken': token},
                       environ_base={'REMOTE_ADDR': '198.51.100.8'}).status_code == 403
    assert client.post('/api/plugins/store/patreon/link', json={'next': 'x'},
                       headers={'X-CSRFToken': token}).status_code == 409
    response = client.post('/api/plugins/store/patreon/link', json={}, headers={'X-CSRFToken': token})
    assert response.status_code == 200 and response.json['authorize_url'] == url
    assert SECRET not in response.get_data(as_text=True)
    assert response.headers['Cache-Control'] == 'no-store'
    status = client.get('/api/plugins/store/patreon')
    assert status.status_code == 409 and status.json['status'] == 'unavailable'  # No reply queued.


def test_confirm_route_requires_admin_and_csrf_and_names_the_refusal(remote):
    client = routes_client()
    connected(remote)
    path = '/api/plugins/store/patreon/confirm'
    assert client.post(path, json={'code': 'ABCD-2345'}).status_code == 400  # No CSRF token.
    headers = {'X-CSRFToken': client.get('/csrf').json['token']}
    assert client.post(path, json={'code': 'ABCD-2345'}, headers=headers,
                       environ_base={'REMOTE_ADDR': '198.51.100.8'}).status_code == 403
    assert client.post(path, json={'code': 'ABCD-2345', 'next': 'x'}, headers=headers).status_code == 409
    remote[0].append(Response({'error': 'code_expired'}, 403))
    refused = client.post(path, json={'code': 'ABCD-2345'}, headers=headers)
    assert refused.status_code == 409 and refused.json['code'] == 'code_expired'
    assert refused.json['error'] == patreon.CONFIRM_ERRORS['code_expired']
    remote[0].append(Response({'status': 'linked'}))
    linked = client.post(path, json={'code': 'abcd2345'}, headers=headers)
    assert linked.status_code == 200 and linked.json['status'] == 'linked'
    assert linked.headers['Cache-Control'] == 'no-store' and SECRET not in linked.get_data(as_text=True)
    assert remote[1][-1][2]['json'] == {'code': 'ABCD2345'}


def test_route_reports_not_configured_on_a_placeholder_service(client, tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, service_url='https://store.example.invalid')
    response = client.get('/api/plugins/store/patreon')
    assert response.status_code == 200 and response.json['status'] == 'not_configured'


# --- Plugins: the owning source selects the acquisition service -------------------

def test_patreon_grants_go_to_the_patreon_service(remote, tmp_path):
    client = connected(remote)
    content = b'patreon plugin archive'
    digest = hashlib.sha256(content).hexdigest()
    release = SimpleNamespace(target='manga/1.0.0.ldsplugin', manifest=SimpleNamespace(id='manga', version='1.0.0'))
    info = SimpleNamespace(path=release.target, length=len(content), hashes={'sha256': digest})
    remote[0].extend([Response({'token': 'g' * 43, 'target': release.target, 'sha256': digest,
                                'expires_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                                'download_url': SERVICE + '/v1/artifacts/' + release.target}),
                      Response(content)])
    assert client.credential() is not None
    path = commerce.download_release(release, info, tmp_path / 'manga.ldsplugin', provider='patreon')
    assert path.read_bytes() == content
    assert [call[1] for call in remote[1][-2:]] == [SERVICE + '/v1/grants', SERVICE + '/v1/artifacts/' + release.target]


def test_plan_asks_patreon_only_for_products_of_the_patreon_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))

    def release(pid):
        return SimpleNamespace(target=pid + '.ldsplugin', price={'kind': 'paid', 'product': 'p'},
                               manifest=SimpleNamespace(id=pid, version='1.0.0'),
                               payload=lambda pid=pid: {'manifest': {'id': pid, 'version': '1.0.0'}})

    @contextmanager
    def sessions(_plugin_id):
        yield []
    owners = {'manga': SimpleNamespace(config=SimpleNamespace(provider='patreon')),
              'paid_tool': SimpleNamespace(config=SimpleNamespace(provider=''))}
    monkeypatch.setattr(service, '_plan_sessions', sessions)
    monkeypatch.setattr(service, '_plan', lambda *a, **k: ([release('manga'), release('paid_tool')], 'x' * 64,
                                                            {'manga', 'paid_tool'}, owners))
    inactive = {'linked': True, 'status': 'inactive', 'tier': None, 'valid_until': None,
                'join_url': JOIN, 'request_url': None}

    class Patreon:
        def library(self):
            return {'status': 'ready', 'items': [], 'patreon': inactive}

    class Commerce:
        def library(self):
            return {'status': 'ready', 'items': [{'plugin_id': 'paid_tool', 'status': 'active'},
                                                 {'plugin_id': 'manga', 'status': 'active'}]}
    monkeypatch.setattr(service, 'PatreonClient', Patreon)
    monkeypatch.setattr(service, 'CommerceClient', Commerce)
    plan = service.preview_plan(None, ['manga', 'paid_tool'])
    # A purchase recorded by ANOTHER service never unlocks a Patreon product.
    assert plan['patreon_required'] == ['manga'] and plan['purchase_required'] == ['manga']
    assert plan['patreon_access'] == {'status': 'ready', 'patreon': inactive}


def _release(version, lds, issues=(), changelog=('Line',), patreon_product=False):
    price = ({'kind': 'paid', 'product': 'patreon_lds', 'provider': 'patreon'} if patreon_product
             else {'kind': 'free'})
    return {'manifest': {'version': version, 'compatibility': {'lds': lds}}, 'price': price,
            'changelog': list(changelog), 'compatibility_issues': [{'field': field} for field in issues]}


def test_membership_features_are_the_releases_only_the_members_core_installs():
    members = ('2026.10.18.3', '2026.10.18.2', '2026.10.18.1')
    cloud = [_release('1.0.9', '<2027,>=2026.10.18.1', ['compatibility.lds'], ['MOSS cloud training']),
             _release('1.0.7', '<2027,>=2026.9.4'), _release('1.0.6', '<2027,>=2026.9.4')]
    reserved = {'version': '1.0.9', 'changelog': ['MOSS cloud training'], 'available': False}
    # The public core: only core requirements (LDS range, plugin API) refuse them.
    assert members_releases(cloud, members, current='2026.10.18') == [reserved]
    forge = [_release('1.2.0', '>=2026.10.18.3', ['compatibility.lds', 'compatibility.api']),
             _release('1.0.0', '>=2026.9.4')]
    assert [item['version'] for item in members_releases(forge, members, current='2026.10.18')] == ['1.2.0']
    # A published members version builds on the public release of its date: it has them.
    installed = [{**cloud[0], 'compatibility_issues': []}, *cloud[1:]]
    assert members_releases(installed, members, current='2026.10.18.3') == [{**reserved, 'available': True}]
    # Nothing is claimed without the channel, for an unpublished members core,
    # for another refusal, or for a members-only plugin (reserved as a whole).
    assert members_releases(cloud, (), current='2026.10.18') == []
    assert members_releases([_release('2.0.0', '>=2026.11.2.1', ['compatibility.lds']), *cloud[1:]],
                            members, current='2026.10.18') == []
    assert members_releases([_release('1.0.9', '>=2026.10.18.1', ['compatibility.lds', 'compatibility.python']),
                             *cloud[1:]], members, current='2026.10.18') == []
    assert members_releases([_release('1.0.3', '>=2026.10.11', ['compatibility.api'], patreon_product=True)],
                            members, current='2026.10.18') == []


def test_update_all_leaves_locked_patreon_plugins_out_of_the_batch(monkeypatch, tmp_path):
    """One members-only plugin without an active membership must not block the
    other updates: it leaves the batch, named, and the rest can be confirmed."""
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    patreon_owner = SimpleNamespace(config=SimpleNamespace(provider='patreon'))
    public_owner = SimpleNamespace(config=SimpleNamespace(provider=''))
    owners = {'manga': patreon_owner, 'tool': public_owner, 'camera': public_owner}
    names = {'manga': 'Manga Studio', 'tool': 'Tool', 'camera': 'Camera'}

    def release(pid):
        price = {'kind': 'paid', 'product': 'patreon_lds', 'provider': 'patreon'} if pid == 'manga' else {'kind': 'free'}
        return SimpleNamespace(target=pid + '.ldsplugin', price=price,
                               manifest=SimpleNamespace(id=pid, version='2.0.0', name=names[pid]),
                               payload=lambda pid=pid: {'manifest': {'id': pid, 'name': names[pid], 'version': '2.0.0'}})

    @contextmanager
    def sessions(_plugin_id):
        yield []
    planned = []

    def plan(_sessions, _registry, plugin_id, _version, *, update_only=False):
        ids = [plugin_id] if isinstance(plugin_id, str) else list(plugin_id)
        planned.append(ids)
        return [release(pid) for pid in ids], 'x' * 64, set(ids), owners

    class Patreon:
        def library(self):
            return {'status': 'ready', 'items': [], 'patreon': INACTIVE}
    monkeypatch.setattr(service, '_plan_sessions', sessions)
    monkeypatch.setattr(service, '_plan', plan)
    monkeypatch.setattr(service, 'PatreonClient', Patreon)

    result = service.preview_plan(None, ['camera', 'manga', 'tool'], update_only=True)
    assert result['requested'] == ['camera', 'tool']
    assert [p['manifest']['id'] for p in result['packages']] == ['camera', 'tool']
    assert result['excluded'] == [{'id': 'manga', 'name': 'Manga Studio'}]
    assert result['purchase_required'] == [] and result['patreon_required'] == []
    assert planned[-1] == ['camera', 'tool']
    # Alone, or outside Update all, the plugin keeps the membership prompt.
    alone = service.preview_plan(None, ['manga'], update_only=True)
    assert alone['patreon_required'] == ['manga'] and 'excluded' not in alone
    chosen = service.preview_plan(None, ['manga', 'tool'])
    assert chosen['patreon_required'] == ['manga'] and 'excluded' not in chosen


@pytest.mark.parametrize('status,code,expected', [
    (403, 'acquisition_denied', patreon.MembershipRequired),
    (503, 'verification_required', patreon.VerificationUnavailable),
    (403, 'invalid_grant', commerce.ServiceRefused),
])
def test_grant_refusals_keep_their_meaning(remote, tmp_path, status, code, expected):
    """Inactive membership (Join / Connect) and stale verification (retry) are
    different problems for the person reading the message."""
    client = connected(remote)
    info = SimpleNamespace(path='manga/1.ldsplugin', length=4, hashes={'sha256': 'b' * 64})
    remote[0].append(Response({'error': code}, status))
    with pytest.raises(expected) as error:
        client.download_target('manga', '1.0.0', 'manga/1.ldsplugin', info, tmp_path / 'manga.ldsplugin')
    assert type(error.value) is expected
    assert not (tmp_path / 'manga.ldsplugin').exists()
