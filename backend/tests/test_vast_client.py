"""vast.ai REST client — pure requests, fully mocked, no network."""
import pytest


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


@pytest.fixture()
def vc(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import vast_client
    return vast_client


def test_search_offers_filters_and_sorts(vc, monkeypatch):
    seen = {}

    def fake_request(method, url, **kw):
        seen['method'], seen['url'], seen['json'] = method, url, kw.get('json')
        seen['auth'] = kw['headers']['Authorization']
        return FakeResp(200, {'offers': [
            {'id': 2, 'gpu_name': 'RTX 4090', 'dph_total': 0.55, 'gpu_ram': 24576},
            {'id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.30, 'gpu_ram': 24576},
        ]})

    monkeypatch.setattr(vc.requests, 'request', fake_request)
    offers = vc.search_offers(min_vram_gb=24, max_dph=0.8, min_inet_down_mbps=400)
    assert seen['method'] == 'POST' and seen['url'].endswith('/bundles/')
    assert seen['auth'] == 'Bearer k-test'
    body = seen['json']
    assert body['gpu_ram'] == {'gte': 24 * 1024}
    assert body['dph_total'] == {'lte': 0.8}
    assert body['verified'] == {'eq': True}
    assert body['rentable'] == {'eq': True}
    assert body['reliability'] == {'gte': 0.95}
    assert body['type'] == 'ondemand'
    assert body['inet_down'] == {'gte': 400}
    assert [o['offer_id'] for o in offers] == [1, 2]          # cheapest first
    assert offers[0]['gpu_ram_gb'] == 24.0


def test_create_instance_returns_contract_id(vc, monkeypatch):
    seen = {}

    def fake_request(method, url, **kw):
        seen['method'], seen['url'], seen['json'] = method, url, kw.get('json')
        return FakeResp(200, {'success': True, 'new_contract': 12345})

    monkeypatch.setattr(vc.requests, 'request', fake_request)
    iid = vc.create_instance(99, image='img:tag', env={'A': '1', '-p 8675:8675': '1'},
                             disk_gb=60, label='lds-7')
    assert iid == '12345'
    assert seen['method'] == 'PUT' and seen['url'].endswith('/asks/99/')
    assert seen['json']['image'] == 'img:tag'
    assert seen['json']['label'] == 'lds-7'
    assert seen['json']['disk'] == 60
    assert seen['json']['runtype'] == 'args'
    assert seen['json']['env']['-p 8675:8675'] == '1'


def test_create_instance_via_template(vc, monkeypatch):
    """Template launch (the smoke-validated path): the body carries ONLY
    template_hash_id + label + disk — env/ports/entrypoint come from the
    template server-side (an env override is rejected with 400 by vast)."""
    seen = {}

    def fake_request(method, url, **kw):
        seen['method'], seen['url'], seen['json'] = method, url, kw.get('json')
        return FakeResp(200, {'success': True, 'new_contract': 777})

    monkeypatch.setattr(vc.requests, 'request', fake_request)
    iid = vc.create_instance(99, disk_gb=48, label='lds-7',
                             template_hash='471ed5903d8cdb8e63b0d0e50f6cd519')
    assert iid == '777'
    assert seen['json'] == {'template_hash_id': '471ed5903d8cdb8e63b0d0e50f6cd519',
                            'label': 'lds-7', 'disk': 48}


def test_list_instances_uses_v1_endpoint(vc, monkeypatch):
    """v0 /instances/ answers 410 deprecated_endpoint since 2026-07-12."""
    seen = {}

    def fake_request(method, url, **kw):
        seen['url'] = url
        return FakeResp(200, {'instances': []})

    monkeypatch.setattr(vc.requests, 'request', fake_request)
    assert vc.list_instances() == []
    assert '/api/v1/instances/' in seen['url']


def test_get_instance_exposes_jupyter_token(vc, monkeypatch):
    payload = {'instances': {'id': 777, 'actual_status': 'running',
                             'public_ipaddr': '5.6.7.8', 'label': 'lds-9',
                             'jupyter_token': 'jtok-abc',
                             'ports': {'18675/tcp': [{'HostPort': '29739'}]}}}
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(200, payload))
    inst = vc.get_instance('777')
    assert inst['jupyter_token'] == 'jtok-abc'
    assert vc.derive_base_url(inst, 18675) == 'http://5.6.7.8:29739'


def test_get_instance_gone_returns_none(vc, monkeypatch):
    """vast answers 200 + {'instances': null} for a destroyed instance
    (observed live on 2026-07-12)."""
    monkeypatch.setattr(vc.requests, 'request',
                        lambda m, u, **kw: FakeResp(200, {'instances': None}))
    assert vc.get_instance('44625910') is None


def test_create_instance_failure_raises(vc, monkeypatch):
    monkeypatch.setattr(vc.requests, 'request',
                        lambda m, u, **kw: FakeResp(200, {'success': False, 'error': 'no capacity'}))
    with pytest.raises(vc.VastError):
        vc.create_instance(99, image='i', env={}, disk_gb=10, label='lds-x')


def test_list_and_get_instance(vc, monkeypatch):
    payload = {'instances': [{'id': 12345, 'actual_status': 'running',
                              'public_ipaddr': '1.2.3.4', 'label': 'lds-7',
                              'dph_total': 0.4,
                              'ports': {'8675/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '40123'}]}}]}
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(200, payload))
    insts = vc.list_instances()
    assert insts[0]['instance_id'] == '12345'
    assert insts[0]['label'] == 'lds-7'
    assert vc.get_instance('12345')['public_ipaddr'] == '1.2.3.4'
    assert vc.get_instance('999') is None


def test_destroy_is_idempotent_on_404(vc, monkeypatch):
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(404, {}))
    assert vc.destroy_instance('12345') is True


def test_destroy_5xx_returns_false(vc, monkeypatch):
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(500, {}))
    assert vc.destroy_instance('12345') is False


def test_derive_base_url(vc):
    inst = {'public_ipaddr': '1.2.3.4',
            'ports': {'8675/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '40123'}]}}
    assert vc.derive_base_url(inst, 8675) == 'http://1.2.3.4:40123'
    assert vc.derive_base_url({'public_ipaddr': None, 'ports': None}, 8675) is None
    assert vc.derive_base_url({'public_ipaddr': '1.2.3.4', 'ports': {}}, 8675) is None


def test_missing_key_raises(vc, monkeypatch):
    monkeypatch.delenv('VAST_API_KEY', raising=False)
    with pytest.raises(vc.VastError):
        vc.search_offers(min_vram_gb=24, max_dph=0.8)


def test_network_error_raises_vast_error(vc, monkeypatch):
    def boom(*a, **kw):
        raise vc.requests.ConnectionError('refused')
    monkeypatch.setattr(vc.requests, 'request', boom)
    with pytest.raises(vc.VastError, match='request failed'):
        vc.search_offers(min_vram_gb=24, max_dph=0.8)


def test_destroy_network_error_returns_false(vc, monkeypatch):
    def boom(*a, **kw):
        raise vc.requests.ConnectionError('refused')
    monkeypatch.setattr(vc.requests, 'request', boom)
    assert vc.destroy_instance('12345') is False
