"""vast.ai REST client — pure requests, fully mocked, no network."""
import pytest


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload) if text is None else text

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


def test_search_offers_quality_filters_and_host_fields(vc, monkeypatch):
    """Reliability floor + disk_bw filter reach the search body; offers expose
    machine_id and reliability for the selection layer (blacklist, preference)."""
    seen = {}

    def fake_request(method, url, **kw):
        seen['json'] = kw.get('json')
        return FakeResp(200, {'offers': [
            {'id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.30, 'gpu_ram': 24576,
             'machine_id': 43503, 'reliability2': 0.997},
        ]})

    monkeypatch.setattr(vc.requests, 'request', fake_request)
    offers = vc.search_offers(min_vram_gb=24, max_dph=0.8,
                              min_reliability=0.98, min_disk_bw_mbps=500)
    assert seen['json']['reliability'] == {'gte': 0.98}
    assert seen['json']['disk_bw'] == {'gte': 500}
    assert offers[0]['machine_id'] == 43503
    assert offers[0]['reliability'] == 0.997


def test_search_offers_optional_trust_filters(vc, monkeypatch):
    seen = []

    def fake_request(method, url, **kw):
        seen.append(kw['json'])
        return FakeResp(200, {'offers': []})

    monkeypatch.setattr(vc.requests, 'request', fake_request)
    vc.search_offers(min_vram_gb=24, max_dph=0.8,
                     verified_only=False, secure_cloud_only=False)
    vc.search_offers(min_vram_gb=24, max_dph=0.8,
                     verified_only=True, secure_cloud_only=True)

    assert 'verified' not in seen[0]
    assert 'datacenter' not in seen[0]
    assert seen[1]['verified'] == {'eq': True}
    assert seen[1]['datacenter'] == {'eq': True}


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
                             template_hash='471ed5903d8cdb8e63b0d0e50f6cd519',
                             image='vastai/ostris-ai-toolkit:new-tag')
    assert iid == '777'
    assert seen['json'] == {'template_hash_id': '471ed5903d8cdb8e63b0d0e50f6cd519',
                            'label': 'lds-7', 'disk': 48,
                            'image': 'vastai/ostris-ai-toolkit:new-tag'}


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


def test_get_instance_reports_the_image_the_pod_really_booted(vc, monkeypatch):
    """We ask for a machine by TEMPLATE, and a template is published by somebody
    else: what booted is a different fact from what we requested, and it is the
    only one that identifies the trainer that produced the weights. An instance
    that does not report it simply records nothing."""
    payload = {'instances': {'id': 778, 'actual_status': 'running',
                             'image_uuid': 'vastai/ostris-ai-toolkit:abc1234-cuda-12.9'}}
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(200, payload))
    assert vc.get_instance('778')['image_uuid'] == \
        'vastai/ostris-ai-toolkit:abc1234-cuda-12.9'

    silent = {'instances': {'id': 779, 'actual_status': 'running'}}
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(200, silent))
    assert vc.get_instance('779')['image_uuid'] is None


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


# --- what a refusal is allowed to say, and what it must never say ---------------

def test_a_refusal_carries_vast_s_own_words(vc, monkeypatch):
    """THE diagnostic fix: a non-200 used to be reported as 'HTTP 400 {}' because
    the body was parsed only on 200. The body IS the diagnosis."""
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(
        400, text='{"success": false, "msg": "disk_space 86 exceeds the 57 GB free"}'))
    with pytest.raises(vc.VastError) as excinfo:
        vc.create_instance(99, image='i', env={}, disk_gb=86, label='lds-x')
    assert 'HTTP 400' in str(excinfo.value)
    assert 'exceeds the 57 GB free' in str(excinfo.value)


def test_every_call_reports_the_body_not_just_the_code(vc, monkeypatch):
    monkeypatch.setattr(vc.requests, 'request',
                        lambda m, u, **kw: FakeResp(500, text='upstream is on fire'))
    for call in (lambda: vc.search_offers(min_vram_gb=24, max_dph=0.8),
                 lambda: vc.list_instances(),
                 lambda: vc.execute_command('7', 'echo hi')):
        with pytest.raises(vc.VastError, match='upstream is on fire'):
            call()


def test_a_refusal_that_echoes_our_request_never_leaks_a_secret(vc, monkeypatch):
    """vast can quote back the ask it refused — and our asks carry the pod's
    Hugging Face token, the training UI bearer, and the API key itself."""
    echoed = ('{"error": "bad request", "request": {"env": {"HF_TOKEN": '
              '"hf_LEAKEDsecretVALUE123", "AI_TOOLKIT_AUTH": "bearer-me-not"}, '
              '"headers": {"Authorization": "Bearer k-test"}}}')
    monkeypatch.setattr(vc.requests, 'request',
                        lambda m, u, **kw: FakeResp(400, text=echoed))
    with pytest.raises(vc.VastError) as excinfo:
        vc.create_instance(99, image='i', env={'HF_TOKEN': 'hf_LEAKEDsecretVALUE123'},
                           disk_gb=10, label='lds-x')
    message = str(excinfo.value)
    assert 'hf_LEAKEDsecretVALUE123' not in message
    assert 'bearer-me-not' not in message
    assert 'k-test' not in message
    assert 'bad request' in message, 'the useful half must survive the scrubbing'


def test_a_refusal_body_is_capped_not_pasted_whole(vc, monkeypatch):
    monkeypatch.setattr(vc.requests, 'request',
                        lambda m, u, **kw: FakeResp(400, text='x' * 5000))
    with pytest.raises(vc.VastError) as excinfo:
        vc.create_instance(99, image='i', env={}, disk_gb=10, label='lds-x')
    assert len(str(excinfo.value)) < 600


def test_an_oversized_onstart_is_refused_without_a_request(vc, monkeypatch):
    """vast caps the ask at 16384 characters and answers a plain 400 — which is
    how two cloud quantization launches died. Refusing here costs no round trip,
    and the sentence names the real problem instead of the status code."""
    calls = []
    monkeypatch.setattr(vc.requests, 'request',
                        lambda m, u, **kw: calls.append(u) or FakeResp(200, {
                            'success': True, 'new_contract': 1}))
    with pytest.raises(vc.VastError, match='onstart script is'):
        vc.create_instance(99, image='i', env={}, disk_gb=10, label='lds-x',
                           onstart='x' * (vc.MAX_ONSTART_CHARS + 1))
    assert calls == [], 'nothing may be sent — a rental could be created'
    # One character under the limit is a normal launch.
    assert vc.create_instance(99, image='i', env={}, disk_gb=10, label='lds-x',
                              onstart='x' * vc.MAX_ONSTART_CHARS) == '1'


def test_an_absurd_image_reference_is_refused_the_same_way(vc, monkeypatch):
    calls = []
    monkeypatch.setattr(vc.requests, 'request',
                        lambda m, u, **kw: calls.append(u) or FakeResp(200, {}))
    with pytest.raises(vc.VastError, match='image reference'):
        vc.create_instance(99, image='i' * (vc.MAX_IMAGE_CHARS + 1), env={},
                           disk_gb=10, label='lds-x')
    assert calls == []


# --- the disk floor: the difference between an offer and a rental ---------------

def test_search_asks_for_the_disk_the_rental_will_claim(vc, monkeypatch):
    """An ask whose `disk` exceeds the offer's free space is refused by vast, and
    the CHEAPEST offer is where free space runs out (live 2026-08-04: $0.081/h
    with 57 GB, against 19 others averaging 500+)."""
    seen = {}

    def fake_request(method, url, **kw):
        seen['json'] = kw.get('json')
        return FakeResp(200, {'offers': [
            {'id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.30, 'gpu_ram': 24576,
             'disk_space': 512.34, 'inet_down': 900.5},
        ]})

    monkeypatch.setattr(vc.requests, 'request', fake_request)
    offers = vc.search_offers(min_vram_gb=8, max_dph=0.8, min_disk_gb=86)
    assert seen['json']['disk_space'] == {'gte': 86}
    assert offers[0]["disk_space_gb"] == 512.3
    assert offers[0]['inet_down'] == 900.5


def test_an_offer_too_small_for_the_job_is_dropped_even_if_vast_returns_it(vc, monkeypatch):
    """Belt and braces: a silently-ignored predicate would hand back exactly the
    unrentable offers. An offer that does not publish its disk is kept."""
    monkeypatch.setattr(vc.requests, 'request', lambda m, u, **kw: FakeResp(200, {'offers': [
        {'id': 1, 'gpu_name': 'RTX 3080 Ti', 'dph_total': 0.08, 'gpu_ram': 12288,
         'disk_space': 57.4},
        {'id': 2, 'gpu_name': 'RTX 5070', 'dph_total': 0.11, 'gpu_ram': 12288,
         'disk_space': 717.6},
        {'id': 3, 'gpu_name': 'RTX 4090', 'dph_total': 0.34, 'gpu_ram': 24576},
    ]}))
    assert [o['offer_id'] for o in vc.search_offers(
        min_vram_gb=8, max_dph=0.8, min_disk_gb=86)] == [2, 3]
    # No floor asked -> no filtering, and no disk predicate in the body.
    assert len(vc.search_offers(min_vram_gb=8, max_dph=0.8)) == 3


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


def test_execute_command_names_the_provider_s_constraint_as_its_own_type(vc, monkeypatch):
    """vast's remote-exec endpoint is constrained to `ls`, `rm` and `du` (its own
    CLI's "available commands"), so a caller shipping a program is refused with
    HTTP 400 `invalid_args` — every time, on every pod, not as a transient. It
    gets a dedicated type because a pre-flight that cannot RUN and a pre-flight
    that FAILED call for opposite behaviour: the first must not ground the lane.
    Recognised by the provider's error CODE, never by its sentence."""
    def fake_request(method, url, **kw):
        return FakeResp(400, {'success': False, 'error': 'invalid_args',
                              'msg': 'Invalid command given.'})

    monkeypatch.setattr(vc, '_request', fake_request)
    with pytest.raises(vc.VastCommandUnsupported) as e:
        vc.execute_command(42, "python -c 'print(1)'")
    assert 'ls, rm and du' in str(e.value)
    # Still a VastError: callers whose whole feature IS the remote program keep
    # failing loudly on it, and nothing that catches VastError changes shape.
    assert isinstance(e.value, vc.VastError)


def test_execute_command_keeps_other_refusals_generic(vc, monkeypatch):
    """Only `invalid_args` means the command itself is unrunnable. A 400 for any
    other reason, or a 200 that carries no result URL, stays a plain VastError —
    reading every 400 as "unsupported" would silently turn a real failure into a
    skipped check."""
    def fake_request(method, url, **kw):
        return FakeResp(400, {'success': False, 'error': 'no_such_instance'})

    monkeypatch.setattr(vc, '_request', fake_request)
    with pytest.raises(vc.VastError) as e:
        vc.execute_command(42, 'ls -l')
    assert not isinstance(e.value, vc.VastCommandUnsupported)


def test_search_offers_sends_a_compute_floor_only_when_asked(vc, monkeypatch):
    """bf16 is not a speed on Turing, it is absent — so a lane whose recipe
    trains in bf16 has to filter on the GPU generation, not just on VRAM. vast
    reports it as an integer (750 Turing, 800 Ampere, 900 Hopper, 1200
    Blackwell) and the predicate goes in the search, before any rental.

    Sent ONLY when non-zero: every other lane's offer pool is a measured thing,
    and a floor arriving by default would quietly change what those lanes see."""
    seen = {}

    def fake_request(method, url, **kw):
        seen['json'] = kw.get('json')
        return FakeResp(200, {'offers': []})

    monkeypatch.setattr(vc, '_request', fake_request)
    vc.search_offers(min_vram_gb=48, max_dph=1.0, min_compute_cap=800)
    assert seen['json']['compute_cap'] == {'gte': 800}
    vc.search_offers(min_vram_gb=24, max_dph=1.0)
    assert 'compute_cap' not in seen['json']

