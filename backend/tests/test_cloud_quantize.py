"""Cloud quantization: the payload, the price, and the promise that the pod dies.

Nothing here rents anything or touches Hugging Face. What is asserted is the
exact onstart script the machine would receive, the refusals that happen BEFORE
a rental, and — the one that actually matters — that the instance is destroyed
on every path out, including the ones nobody plans for.

The rental half is written against the FIRST real attempt, which produced
``create_instance failed: HTTP 400 {}`` one second after the click: an offer
picked on price alone did not have the disk the job asks for, and a single
refusal ended everything. Those two are pinned here.
"""
import base64

import pytest

from app.services import cloud_quantize as cq
from app.services import vast_client


class _Sibling:
    def __init__(self, rfilename, size):
        self.rfilename = rfilename
        self.size = size
        self.lfs = None


class _Info:
    def __init__(self, siblings):
        self.siblings = siblings


class _Api:
    def __init__(self, siblings):
        self._siblings = siblings
        self.deleted = []

    def repo_info(self, **_kw):
        return _Info(self._siblings)

    def delete_file(self, **kw):
        self.deleted.append(kw.get('path_in_repo'))


FAKE_TOKEN = 'hf_zzUNIQUEsecret999'
MASTER = 'Krea_full_subject1_000002500.safetensors'
BF16_BYTES = 25_600_000_000
OFFER = {'offer_id': 77, 'gpu_name': 'RTX 3060', 'dph_total': 0.09,
         'inet_down': 1000, 'machine_id': 5, 'disk_space_gb': 512.0}
# A second machine of another class, a hair dearer: what the lane falls back to
# when the first one refuses the ask.
OFFER_B = {'offer_id': 78, 'gpu_name': 'RTX 4060', 'dph_total': 0.10,
           'inet_down': 900, 'machine_id': 6, 'disk_space_gb': 700.0}
# What vast actually answers when the ask does not fit the machine.
TAKEN = 'create_instance failed: HTTP 400 {"msg": "disk_space 86 exceeds free 57"}'


@pytest.fixture(autouse=True)
def _app_context(app):
    """system_state writes go through the DB — every path here stamps one."""
    with app.app_context():
        yield app


@pytest.fixture(autouse=True)
def _tokens(monkeypatch):
    monkeypatch.setattr(cq.cfg, 'secret',
                        lambda name, *a, **k: FAKE_TOKEN if name in
                        ('HF_CLOUD_TOKEN', 'HF_TOKEN', 'VAST_API_KEY') else None)


def _api(extra=()):
    return _Api([_Sibling(MASTER, BF16_BYTES), *extra])


def _plan(**kw):
    return cq.plan('me/krea-run-146', token=FAKE_TOKEN, _api=_api(kw.pop('extra', ())),
                   _offers=[OFFER], **kw)


# --- planning -------------------------------------------------------------------

def test_plan_prices_the_rental_and_names_both_files():
    planned = _plan()
    assert planned['weight_name'] == MASTER
    assert planned['output_name'] == 'Krea_full_subject1_000002500_fp8.safetensors'
    assert planned['source_bytes'] == BF16_BYTES
    # The user downloads ~10 GB instead of 25.6 GB — that is the whole point.
    assert 9e9 < planned['output_bytes_typical'] < 11e9
    assert planned['price_per_hour'] == 0.09
    assert planned['estimated_minutes'] >= 6
    assert 0 < planned['estimated_cost'] < 1.0, 'a minutes-long job must cost cents'
    # A hard ceiling is quoted up front, not discovered later.
    assert planned['max_minutes'] == cq.max_minutes() >= 5


def test_plan_refuses_to_rebuild_an_export_that_already_exists():
    with pytest.raises(cq.CloudQuantizeError, match='already in'):
        _plan(extra=(_Sibling('Krea_full_subject1_000002500_fp8.safetensors', 1),))


def test_plan_refuses_a_repository_with_no_master():
    with pytest.raises(cq.CloudQuantizeError, match='no full-precision'):
        cq.plan('me/empty', token=FAKE_TOKEN, _api=_Api([]), _offers=[OFFER])


def test_plan_refuses_a_malformed_repository_id():
    with pytest.raises(cq.CloudQuantizeError, match='owner/name'):
        cq.plan('not-a-repo', token=FAKE_TOKEN, _api=_api(), _offers=[OFFER])


def test_plan_ignores_an_existing_export_when_choosing_the_master():
    """An `_fp8` sibling is an export, never a quantization source."""
    api = _Api([_Sibling('Krea_a_fp8.safetensors', 10), _Sibling(MASTER, BF16_BYTES)])
    assert cq.plan('me/r', token=FAKE_TOKEN, _api=api, _offers=[OFFER])['weight_name'] == MASTER


def test_the_disk_request_holds_the_master_its_twin_and_the_cache():
    assert cq._disk_gb_for(BF16_BYTES) >= int(25.6 * 2.6) + 20
    assert cq._disk_gb_for(0) == 60          # never below a usable floor


# --- the script the machine actually receives ------------------------------------

def test_the_onstart_embeds_the_real_exporter_and_nothing_else():
    script = cq.build_onstart(_plan())
    payload = [line for line in script.splitlines() if 'base64 -d' in line][0]
    encoded = payload.split("'")[1]
    source = base64.b64decode(encoded).decode('utf-8')
    # THE invariant: one implementation. The pod runs the module the unit tests
    # exercise and that ComfyUI's own converter was fed.
    assert 'def export_scaled_fp8' in source
    assert 'scale_weight' in source
    assert 'LDS_FP8_RESULT' in source


def test_the_onstart_downloads_quantizes_uploads_and_reports_back():
    script = cq.build_onstart(_plan())
    assert 'hf_hub_download' in script
    assert "'me/krea-run-146'" in script
    assert f"filename='{MASTER}'" in script
    assert 'python fp8_export.py --src "$SRC"' in script
    assert '--budget-seconds' in script
    # It reports through the repo it is already authenticated for — no inbound
    # connection to the pod is ever needed.
    assert cq.RESULT_FILE in script
    assert 'upload_file' in script


def test_the_pod_installs_only_what_it_actually_imports():
    # `safetensors` was installed here until the exporter stopped memory-mapping
    # the checkpoint; it now reads and writes the format with plain file I/O, so
    # nothing the pod runs imports the package any more. What executes there is
    # fp8_export.py (json, os, struct, time, torch — plus huggingface_hub behind
    # --repo-id) and two heredocs that import huggingface_hub. An install nobody
    # imports is boot time on a machine billed by the hour.
    lines = [line for line in cq.build_onstart(_plan()).splitlines()
             if line.startswith('python -m pip install')]
    assert len(lines) == 1
    assert 'huggingface_hub>=0.30' in lines[0]
    assert 'safetensors' not in lines[0]


def test_the_master_is_only_dropped_when_explicitly_asked():
    assert '--drop-bf16' not in cq.build_onstart(_plan())
    assert '--drop-bf16' in cq.build_onstart(_plan(keep_bf16=False))


def test_the_token_travels_as_an_environment_variable_never_in_the_script():
    script = cq.build_onstart(_plan())
    # The secret reaches the pod through the container ENVIRONMENT (create_instance
    # env), never through a script the host stores and echoes back.
    assert FAKE_TOKEN not in script
    assert 'os.environ.get("HF_TOKEN")' in script


# --- the rental, and its guaranteed end -----------------------------------------

class _Vast:
    """Stands in for the whole client: searching, renting, destroying.

    Refusals are raised as the REAL VastError, because the retry loop this lane
    reuses (cloud_training.rent_with_fresh_offers) keys on that exact type.
    """

    def __init__(self, *, create_raises=None, offers=None, refuse=()):
        self.created = []
        self.destroyed = []
        self.instances = []
        self.searches = []
        self.attempts = []
        self.create_raises = create_raises      # raised on EVERY attempt
        self.refuse = set(refuse)               # offer ids that answer 400
        self.offers = [OFFER] if offers is None else list(offers)

    def search_offers(self, **kw):
        self.searches.append(kw)
        return [dict(o) for o in self.offers]

    def create_instance(self, offer_id, **kw):
        self.attempts.append(offer_id)
        if self.create_raises:
            raise self.create_raises
        if offer_id in self.refuse:
            raise vast_client.VastError(TAKEN)
        self.created.append((offer_id, kw))
        return '9001'

    def destroy_instance(self, instance_id):
        self.destroyed.append(str(instance_id))
        return True

    def list_instances(self):
        return self.instances


@pytest.fixture()
def vast(monkeypatch):
    fake = _Vast()
    monkeypatch.setattr(cq, 'vast_client', fake)
    return fake


def _drive(monkeypatch, vast, result, *, timeout=False):
    planned = _plan()
    api = _api()
    monkeypatch.setattr(cq, '_read_result', lambda *_a: None if timeout else result)
    clock = iter([0.0] + [i * 10.0 for i in range(1, 200)] + [10 ** 9] * 50)
    cq._drive(planned, FAKE_TOKEN, _api=api, _sleep=lambda _s: None,
              _now=lambda: next(clock))
    return api


def test_a_successful_job_reports_the_file_and_destroys_the_machine(monkeypatch, vast):
    api = _drive(monkeypatch, vast,
                 {'ok': True, 'uploaded': True, 'bytes_after': 10_100_000_000})
    state = cq.status()
    assert state['status'] == 'done'
    assert state['result']['uploaded'] is True
    assert vast.destroyed == ['9001']
    # The pod's report file does not stay behind in the user's repository.
    assert api.deleted == [cq.RESULT_FILE]
    # The rental carries this lane's label, which is what makes reaping possible.
    assert vast.created[0][1]['label'].startswith(cq.LABEL_PREFIX)
    assert vast.created[0][1]['env']['HF_TOKEN'] == FAKE_TOKEN


def test_a_failed_conversion_still_destroys_the_machine(monkeypatch, vast):
    _drive(monkeypatch, vast, {'ok': False, 'error': 'out of disk'})
    assert cq.status()['status'] == 'error'
    assert 'out of disk' in cq.status()['error']
    assert vast.destroyed == ['9001']


def test_a_pod_that_never_reports_is_destroyed_at_the_hard_deadline(monkeypatch, vast):
    _drive(monkeypatch, vast, None, timeout=True)
    state = cq.status()
    assert state['status'] == 'error'
    assert 'reported nothing' in state['error']
    assert 'nothing in your repository was changed' in state['error']
    assert vast.destroyed == ['9001']


def test_an_upload_that_never_happened_is_not_a_success(monkeypatch, vast):
    _drive(monkeypatch, vast, {'ok': True, 'uploaded': False})
    assert cq.status()['status'] == 'error'
    assert vast.destroyed == ['9001']


def test_a_rental_that_never_started_destroys_nothing_and_says_so(monkeypatch, vast):
    vast.create_raises = RuntimeError('offer taken')
    planned = _plan()
    cq._drive(planned, FAKE_TOKEN, _api=_api(), _sleep=lambda _s: None)
    assert cq.status()['status'] == 'error'
    assert vast.destroyed == []


# --- the incident: one refused offer used to be the end of the job -------------

def test_an_offer_that_refuses_the_ask_is_not_the_end_of_the_job(monkeypatch, vast):
    """The first real run died on 'HTTP 400 {}' one second after the click, with
    no machine and no reason. A refusal now costs one offer, not the job."""
    vast.offers = [OFFER, OFFER_B]
    vast.refuse = {77}
    _drive(monkeypatch, vast, {'ok': True, 'uploaded': True, 'bytes_after': 10_100_000_000})
    assert vast.attempts == [77, 78]            # refused, then a FRESH offer
    assert [c[0] for c in vast.created] == [78]
    assert cq.status()['status'] == 'done'
    assert vast.destroyed == ['9001']


def test_the_status_names_the_machine_that_was_actually_rented(monkeypatch, vast):
    """The quote is not a promise about WHICH box: renting re-searches, so what
    the panel reports from then on is the machine that is really billing."""
    vast.offers = [OFFER, OFFER_B]
    vast.refuse = {77}
    _drive(monkeypatch, vast, {'ok': True, 'uploaded': True})
    state = cq.status()
    assert state['gpu_name'] == 'RTX 4060'
    assert state['price_per_hour'] == 0.10      # the rented one, not the quoted one
    assert state['estimated_cost'] > 0


def test_when_every_machine_refuses_nothing_is_rented_and_vast_is_quoted(monkeypatch, vast):
    vast.offers = [OFFER, OFFER_B]
    vast.refuse = {77, 78}
    _drive(monkeypatch, vast, {'ok': True, 'uploaded': True})
    state = cq.status()
    assert state['status'] == 'error'
    assert vast.created == [] and vast.destroyed == []       # nothing was rented
    assert vast.attempts == [77, 78]                         # every offer tried once
    # The user is told what vast said, not 'HTTP 400 {}'.
    assert 'disk_space 86 exceeds free 57' in state['error']


def test_the_search_asks_for_the_disk_the_pod_will_claim(monkeypatch, vast):
    """The refused ask and the offer filter must be the same number, or the lane
    keeps picking machines that cannot hold the job."""
    _drive(monkeypatch, vast, {'ok': True, 'uploaded': True})
    disk = cq._disk_gb_for(BF16_BYTES)
    assert disk >= 86
    assert vast.searches[-1]['min_disk_gb'] == disk
    assert vast.created[0][1]['disk_gb'] == disk


def test_a_machine_dearer_than_the_quote_is_reported_not_rented(monkeypatch, vast):
    """The estimate is indicative, but not by an order of magnitude: a market
    that moved gets said out loud instead of billed."""
    vast.offers = [{**OFFER_B, 'dph_total': 0.55}]
    _drive(monkeypatch, vast, {'ok': True, 'uploaded': True})
    state = cq.status()
    assert state['status'] == 'error'
    assert vast.created == [] and vast.destroyed == []
    assert '0.550' in state['error'] and '0.090' in state['error']
    assert 'new estimate' in state['error']


def test_a_dearer_machine_within_the_margin_is_simply_rented(monkeypatch, vast):
    """Cents of drift are market noise — refusing them would be theatre."""
    vast.offers = [{**OFFER_B, 'dph_total': 0.11}]
    _drive(monkeypatch, vast, {'ok': True, 'uploaded': True})
    assert cq.status()['status'] == 'done'
    assert cq.status()['price_per_hour'] == 0.11


def test_the_quote_is_priced_by_the_same_rule_that_rents(monkeypatch):
    """plan() used to take offers[0] — the cheapest, which is exactly where the
    bait prices and the 57 GB disks live. Quoting one machine and renting
    another by a different rule is also how a price guard becomes a permanent
    refusal, so both go through _choose."""
    bait = {'offer_id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.05, 'machine_id': 1}
    honest = [{'offer_id': 2, 'gpu_name': 'RTX 3090', 'dph_total': 0.20, 'machine_id': 2},
              {'offer_id': 3, 'gpu_name': 'RTX 3090', 'dph_total': 0.22, 'machine_id': 3}]
    planned = cq.plan('me/krea-run-146', token=FAKE_TOKEN, _api=_api(),
                      _offers=[bait, *honest])
    assert planned['offer']['offer_id'] != 1
    assert planned['price_per_hour'] == 0.20


def test_reconcile_destroys_an_orphan_but_spares_a_live_job(vast):
    vast.instances = [
        {'instance_id': '111', 'label': cq.LABEL_PREFIX + 'abc'},
        {'instance_id': '222', 'label': 'someone-elses-run'},
    ]
    assert cq.reconcile_orphans() == ['111']
    assert vast.destroyed == ['111']

    vast.destroyed.clear()
    cq.queue_manager._set_system_state(cq._STATE_KEY, {
        'status': 'running', 'instance_id': '111', 'repo_id': 'r',
        'weight_name': 'w', 'output_name': 'o', 'source_bytes': 1,
        'output_bytes_typical': 1, 'price_per_hour': 0, 'estimated_cost': 0,
        'keep_bf16': True}, ttl_seconds=60)
    assert cq.reconcile_orphans() == []
    assert vast.destroyed == []
