"""Boot watchdog: the structural twin of the first-step watchdog (2026-07-28).

The boot wait already READ the pod's remote state — vast status, port
publication — and spent it on the phase line only, then decided on elapsed time
alone. A host honestly pulling a 26 GB container image was killed at 25 minutes
and blacklisted for three days, invisibly, while the evidence that it was
working sat one line above the kill.

Same rule as the first-step watchdog: a pod whose remote evidence advances is a
pod that progresses. Same shape too — an idle clock rearmed by evidence, an
absolute ceiling evaluated BEFORE the rearm, and a failure message that says
what was MEASURED.
"""
import pytest
from app.extensions import db

from test_cloud_training_monitor import ct, FakeRemote, _launch    # noqa: F401


def _coarse_clock(ct, monkeypatch, step=600.0):
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_now',
                        lambda: clock.__setitem__('t', clock['t'] + step) or clock['t'])
    return clock


def _instance(status, *, ports=False, msg=None):
    rec = {'instance_id': '777', 'actual_status': status,
           'public_ipaddr': '1.2.3.4', 'label': 'lds-x',
           'jupyter_token': 'jtok-vast'}
    if ports:
        rec['ports'] = {'18675/tcp': [{'HostPort': '40123'}]}
    if msg is not None:
        rec['status_msg'] = msg
    return rec


# --- the unit that decides what counts as evidence ----------------------------

def test_a_line_whose_only_moving_part_is_the_clock_is_not_evidence(ct):
    """The mistake we nearly shipped into the first-step watchdog: hashing a
    progress line whose elapsed time advances while nothing else does."""
    a = _instance('loading', msg='Pulling image 1.95GB/26.1GB [02:31<01:12:44]')
    b = _instance('loading', msg='Pulling image 1.95GB/26.1GB [09:47<06:31:02]')
    assert ct._boot_facts(a, 18675, None) == ct._boot_facts(b, 18675, None)


def test_more_pulled_bytes_and_a_published_port_are_evidence(ct):
    base = ct._boot_facts(_instance('loading', msg='Pulling image 1.9GB/26.1GB'),
                          18675, None)
    more = ct._boot_facts(_instance('loading', msg='Pulling image 8.4GB/26.1GB'),
                          18675, None)
    up = ct._boot_facts(_instance('running', ports=True), 18675,
                        'http://1.2.3.4:40123')
    assert more - base and up - base


# --- the run that used to die: an honest, slow image pull ---------------------

def test_a_pod_whose_boot_advances_is_never_killed_at_the_deadline(
        ct, app, client, monkeypatch):
    """RED before the fix. The pod spends five polls in 'loading' with its pull
    line advancing — far past the 25-min ready timeout on this coarse clock —
    then comes up. Judged on time alone it dies; judged on its own evidence it
    trains."""
    destroyed = []
    remote = FakeRemote(polls_to_complete=3)
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    calls = {'n': 0}

    def booting(iid):
        calls['n'] += 1
        if calls['n'] <= 5:
            return _instance('loading',
                             msg=f'Pulling image {calls["n"] * 5}.1GB/26.1GB '
                                 f'[0{calls["n"]}:10<01:44:02]')
        return _instance('running', ports=True, msg='container started')

    monkeypatch.setattr(ct.vast_client, 'get_instance', booting)
    _coarse_clock(ct, monkeypatch)
    with app.app_context():
        # The ceiling has its own test below; this one is about the rearm.
        ct.cfg.save_config({'cloud': {'boot_budget_minutes': 0}})
        ct._monitor(app, run_id)
        run = db.session.get(ct.CloudTrainingRun, run_id)
        assert 'become ready' not in (run.error or '').lower()
        assert remote.job_config is not None       # boot finished, job submitted
        assert ct._load_bad_hosts() == {}          # and nobody got exiled


# --- anti-masking: a pod that shows nothing must still die fast ---------------

def test_a_pod_that_shows_nothing_still_dies_on_the_idle_budget(
        ct, app, client, monkeypatch):
    """The guard that keeps the fix from being a cover-up. No new status, no
    port, no host line: this pod is money burning and must die on the 25-min
    budget, with the full three-day ban it always earned."""
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    polls = {'n': 0}

    def frozen(iid):
        polls['n'] += 1
        return _instance('loading', msg='waiting for host')

    monkeypatch.setattr(ct.vast_client, 'get_instance', frozen)
    _coarse_clock(ct, monkeypatch)
    with app.app_context():
        ct._monitor(app, run_id)
        run = db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'error'
        assert 'become ready' in (run.error or '').lower()
        assert destroyed == ['777']
        assert polls['n'] <= 6                     # died fast, on the IDLE clock
        # ... and the message says what was measured, not "never became ready".
        assert 'no boot progress' in (run.error or '').lower()
        assert 'loading' in (run.error or '')
        assert 'not published yet' in (run.error or '')
        # Dead host -> the full ban, no shortened TTL.
        entry = ct._load_bad_hosts()['43503']
        assert entry.get('ttl') is None


# --- the ceiling: a pod that will never finish still dies -----------------------

def test_the_boot_budget_caps_a_pod_that_would_never_finish(
        ct, app, client, monkeypatch):
    """A host at 200 kB/s advances its pull at every poll for a day. Rearming
    alone would let it ride to the runtime cap for zero training steps — the
    ceiling is what keeps the rearm honest."""
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    calls = {'n': 0}

    def crawling(iid):
        calls['n'] += 1
        return _instance('loading',
                         msg=f'Pulling image 1.{calls["n"]}GB/26.1GB')

    monkeypatch.setattr(ct.vast_client, 'get_instance', crawling)
    clock = _coarse_clock(ct, monkeypatch)
    with app.app_context():
        ct._monitor(app, run_id)
        run = db.session.get(ct.CloudTrainingRun, run_id)
        assert run.status == 'error'
        assert 'still booting after 90 min' in (run.error or '')
        assert destroyed == ['777']
        assert clock['t'] < 480 * 60               # well before the runtime cap


def test_a_host_killed_while_still_booting_is_skipped_for_hours_not_days(
        ct, app, client, monkeypatch):
    """The blacklist arbitrage. A host that was still visibly working when the
    ceiling cut it was SLOW, not broken — three days of invisible exile is the
    wrong price. It is still banned (a slow host is a bad deal right now, and
    the retry must land elsewhere), just for hours."""
    destroyed = []
    remote = FakeRemote()
    remote.is_ready = lambda: False
    ds_id, run_id = _launch(ct, app, client, monkeypatch, remote, destroyed)
    calls = {'n': 0}

    def crawling(iid):
        calls['n'] += 1
        return _instance('loading', msg=f'Pulling image 1.{calls["n"]}GB/26.1GB')

    monkeypatch.setattr(ct.vast_client, 'get_instance', crawling)
    clock = _coarse_clock(ct, monkeypatch)
    with app.app_context():
        ct._monitor(app, run_id)
        banned_at = clock['t']
        entry = ct._load_bad_hosts()['43503']
        # A short, explicit TTL — and the generic re-ban the retry path fires
        # right after did NOT silently upgrade it back to three days.
        assert entry['ttl'] == pytest.approx(6 * 3600)
        # Still banned an hour later, free again after seven.
        monkeypatch.setattr(ct, '_now', lambda: banned_at + 3600)
        assert '43503' in ct._load_bad_hosts()
        monkeypatch.setattr(ct, '_now', lambda: banned_at + 7 * 3600)
        assert '43503' not in ct._load_bad_hosts()
