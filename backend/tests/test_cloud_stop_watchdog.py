"""Stop that cannot lie + the out-of-monitor safety net.

Incident that motivated this file: a cloud run froze mid-run (its monitor
thread stopped making progress), the pod kept billing for hours, and the Stop
button answered {'ok': true} without terminating anything — request_stop only
set an in-process threading.Event that nobody was listening to any more.

Everything here is offline: vast_client is fully stubbed, no thread is
started, no dollar is spent.
"""
from datetime import datetime, timedelta

import pytest


@pytest.fixture()
def ct(app, monkeypatch):
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    cloud_training._stop_events.clear()
    cloud_training._monitor_threads.clear()
    yield cloud_training
    cloud_training._stop_events.clear()
    cloud_training._monitor_threads.clear()


def _mkrun(ct, **kw):
    fields = dict(dataset_id=1, status='training', vast_instance_id='90001',
                  vast_label='lds-1', job_name='j1', price_per_hour=0.669)
    fields.update(kw)
    run = ct.CloudTrainingRun(**fields)
    ct.db.session.add(run)
    ct.db.session.commit()
    return run


def _stub_destroy(ct, monkeypatch, ok=True):
    destroyed = []

    def fake(iid):
        destroyed.append(str(iid))
        if isinstance(ok, Exception):
            raise ok
        return ok

    monkeypatch.setattr(ct.vast_client, 'destroy_instance', fake)
    return destroyed


# --------------------------------------------------------------- Stop --

def test_stop_without_a_live_monitor_destroys_the_pod(ct, app, monkeypatch):
    """THE incident scenario: the run is 'training' in the database, no monitor
    thread is alive to observe the stop event, the user clicks Stop. The pod
    must be terminated for real and the run closed — never a complacent ok."""
    with app.app_context():
        stale = datetime.utcnow() - timedelta(hours=2)
        run = _mkrun(ct, updated_at=stale,
                     phase_detail='running: Generating images - 3/8')
        destroyed = _stub_destroy(ct, monkeypatch)

        res = ct.request_stop(run.id)

        assert res['ok'] is True
        assert res['mode'] == 'forced'
        assert destroyed == ['90001']
        ct.db.session.refresh(run)
        assert run.status == 'stopped'
        assert run.finished_at is not None


def test_stop_reports_failure_naming_the_instance(ct, app, monkeypatch):
    """vast refuses the termination -> the user is told, with the instance id
    to destroy by hand. The run is parked in error_pod_kept so boot/launch
    reconciliation reaps the pod later instead of forgetting it."""
    with app.app_context():
        run = _mkrun(ct, updated_at=datetime.utcnow() - timedelta(hours=2))
        _stub_destroy(ct, monkeypatch, ok=RuntimeError('vast 500'))

        res = ct.request_stop(run.id)

        assert res['ok'] is False
        assert '90001' in res['error']
        ct.db.session.refresh(run)
        assert run.status == 'error_pod_kept'
        assert '90001' in (run.error or '')


def test_stop_returns_false_when_destroy_returns_false(ct, app, monkeypatch):
    """destroy_instance answering False (not raising) is a failure too."""
    with app.app_context():
        run = _mkrun(ct, updated_at=datetime.utcnow() - timedelta(hours=2))
        _stub_destroy(ct, monkeypatch, ok=False)
        res = ct.request_stop(run.id)
        assert res['ok'] is False and '90001' in res['error']


def test_stop_hands_over_to_a_responsive_monitor(ct, app, monkeypatch):
    """A monitor that is alive AND writing keeps its graceful path: it stops
    the remote job and rescues the last checkpoint before terminating."""
    with app.app_context():
        run = _mkrun(ct, updated_at=datetime.utcnow())
        destroyed = _stub_destroy(ct, monkeypatch)
        ct._monitor_threads[run.id] = _FakeThread(alive=True)

        res = ct.request_stop(run.id)

        assert res['ok'] is True and res['mode'] == 'graceful'
        assert destroyed == []                       # the monitor does it
        assert ct._stop_event_for(run.id).is_set()
        ct.db.session.refresh(run)
        assert run.status == 'training'              # still the monitor's job
        assert run.stop_requested_at is not None     # deadline armed


def test_stop_on_no_active_run_is_an_explicit_no(ct, app):
    with app.app_context():
        _mkrun(ct, status='done')
        res = ct.request_stop(None)
        assert res['ok'] is False and res['error']


def test_stop_targets_only_the_given_run(ct, app, monkeypatch):
    with app.app_context():
        alive = _FakeThread(alive=True)
        r1 = _mkrun(ct, dataset_id=1)
        r2 = _mkrun(ct, dataset_id=2, vast_instance_id='90002')
        ct._monitor_threads[r1.id] = alive
        ct._monitor_threads[r2.id] = alive
        _stub_destroy(ct, monkeypatch)
        assert ct.request_stop(r1.id)['ok'] is True
        assert ct._stop_event_for(r1.id).is_set() is True
        assert ct._stop_event_for(r2.id).is_set() is False


class _FakeThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


# ---------------------------------------------------------- Watchdog --

def test_supervisor_cuts_a_frozen_training_run(ct, app, monkeypatch):
    """No database write for longer than the freeze threshold, in the phase
    where the monitor is supposed to write every poll -> the pod is terminated
    without asking the (possibly dead) monitor thread."""
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 45}})
        run = _mkrun(ct, updated_at=datetime.utcnow() - timedelta(minutes=50))
        destroyed = _stub_destroy(ct, monkeypatch)

        acted = ct.supervise_active_runs()

        assert [a['reason'] for a in acted] == ['freeze']
        assert destroyed == ['90001']
        ct.db.session.refresh(run)
        assert run.status == 'stopped' and 'no progress' in (run.phase_detail or '').lower()


def test_supervisor_spares_a_legitimately_slow_phase(ct, app, monkeypatch):
    """Staging a dataset / booting a pod / downloading the base weights are
    silent by construction — the watchdog must never fire on them.

    'uploading' was in this list until 2026-08-02 and deliberately left it: it
    now reports the bytes it pushes, so silence there means 'nothing reached
    the machine' rather than 'nobody wrote a row', and it is judged on its own
    much shorter limit (see test_cloud_upload_progress.py). The three phases
    below still have no evidence of their own and keep the generous floor."""
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 45}})
        destroyed = _stub_destroy(ct, monkeypatch)
        for status in ('preparing', 'provisioning', 'downloading'):
            _mkrun(ct, status=status,
                   updated_at=datetime.utcnow() - timedelta(minutes=90))
        assert ct.supervise_active_runs() == []
        assert destroyed == []


def test_the_final_checkpoint_download_is_never_judged_frozen(ct, app,
                                                             monkeypatch,
                                                             tmp_path):
    """THE worst possible loss: the run SUCCEEDED, and the pod is destroyed
    while we are pulling the checkpoint off it.

    test_supervisor_spares_a_legitimately_slow_phase covers the status
    'downloading' — but the row only reached that status AFTER the transfer
    returned, so during the transfer itself the run was still 'training' and
    judged on the 45-minute threshold with a frozen updated_at. This drives the
    real call path instead of a hand-made row.
    """
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 45}})
        run = _mkrun(ct, status='training')
        destroyed = _stub_destroy(ct, monkeypatch)
        seen = {}
        monkeypatch.setattr(ct, '_newest_remote_checkpoint',
                            lambda remote, job_id: {'path': 'out/j1/j1.safetensors',
                                                    'size': 88_000_000})

        def slow_fetch(r, remote, ckpt, **kw):
            # 50 minutes into a big transfer over a slow pod proxy: legitimate,
            # and nothing has written to the row since it started.
            r.updated_at = datetime.utcnow() - timedelta(minutes=50)
            ct.db.session.commit()
            seen['status'] = r.status
            seen['acted'] = ct.supervise_active_runs()
            dest = tmp_path / 'j1.safetensors'
            dest.write_bytes(b'x')
            return str(dest)

        monkeypatch.setattr(ct, '_fetch_checkpoint', slow_fetch)

        assert ct._try_download_checkpoint(run, object()) is True

        assert seen['status'] == 'downloading'   # active, and a SILENT phase
        assert seen['acted'] == []               # the supervisor left it alone
        assert destroyed == []                   # the pod survived the rescue
        ct.db.session.refresh(run)
        assert run.checkpoint_local_path.endswith('j1.safetensors')


def test_a_stop_is_not_deadlined_while_the_checkpoint_is_being_rescued(
        ct, app, monkeypatch):
    """A stop whose monitor is DOWNLOADING the result is not a wedged monitor:
    it is doing the most valuable part of the stop. The deadline still applies
    the moment it goes quiet (see the next test)."""
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 45}})
        _mkrun(ct, status='downloading', updated_at=datetime.utcnow(),
               stop_requested_at=datetime.utcnow() - timedelta(minutes=40))
        destroyed = _stub_destroy(ct, monkeypatch)

        assert ct.supervise_active_runs() == []
        assert destroyed == []


def test_a_download_that_goes_silent_is_still_deadlined(ct, app, monkeypatch):
    """The exemption is narrow on purpose: a monitor that died mid-transfer
    stops being spared one heartbeat later."""
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 45}})
        run = _mkrun(ct, status='downloading',
                     updated_at=datetime.utcnow() - timedelta(minutes=10),
                     stop_requested_at=datetime.utcnow() - timedelta(minutes=40))
        destroyed = _stub_destroy(ct, monkeypatch)

        acted = ct.supervise_active_runs()

        assert [a['reason'] for a in acted] == ['stop_deadline']
        assert destroyed == ['90001']
        ct.db.session.refresh(run)
        assert run.status == 'stopped'


def test_a_long_transfer_keeps_beating_the_progress_clock(ct, app, monkeypatch,
                                                          tmp_path):
    """The status flip alone buys a fixed floor; a transfer that reports its
    progress cannot look like silence at all — and the user sees a figure move
    instead of a frozen 'Downloading…'."""
    with app.app_context():
        run = _mkrun(ct, status='training')
        monkeypatch.setattr(ct, '_newest_remote_checkpoint',
                            lambda remote, job_id: {'path': 'out/j1/j1.safetensors',
                                                    'size': 88_000_000})
        beats = []

        def fetch_with_progress(r, remote, ckpt, on_progress=None, **kw):
            r.updated_at = datetime.utcnow() - timedelta(minutes=50)
            ct.db.session.commit()
            assert on_progress is not None, 'no progress callback was passed'
            silent_since = r.updated_at
            on_progress(44_000_000, 88_000_000)
            beats.append((silent_since, r.updated_at, r.phase_detail))
            dest = tmp_path / 'j1.safetensors'
            dest.write_bytes(b'x')
            return str(dest)

        monkeypatch.setattr(ct, '_fetch_checkpoint', fetch_with_progress)
        ct._try_download_checkpoint(run, object())

        (before, after, detail) = beats[0]
        assert (after - before).total_seconds() > 40 * 60   # the clock beat
        assert 'MB' in (detail or '')


def test_supervisor_spares_a_progressing_run(ct, app, monkeypatch):
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 45}})
        destroyed = _stub_destroy(ct, monkeypatch)
        _mkrun(ct, updated_at=datetime.utcnow() - timedelta(minutes=5))
        assert ct.supervise_active_runs() == []
        assert destroyed == []


def test_freeze_watchdog_can_be_turned_off(ct, app, monkeypatch):
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 0}})
        destroyed = _stub_destroy(ct, monkeypatch)
        _mkrun(ct, updated_at=datetime.utcnow() - timedelta(hours=6))
        assert ct.supervise_active_runs() == []
        assert destroyed == []


def test_supervisor_enforces_the_runtime_cap(ct, app, monkeypatch):
    """The cap used to be computed inside the monitor thread — the safety net
    died with the thing it protected. It now runs from outside."""
    with app.app_context():
        ct.cfg.save_config({'cloud': {'max_runtime_minutes': 60,
                                      'freeze_watchdog_minutes': 0}})
        run = _mkrun(ct, created_at=datetime.utcnow() - timedelta(minutes=200),
                     updated_at=datetime.utcnow())
        destroyed = _stub_destroy(ct, monkeypatch)

        acted = ct.supervise_active_runs()

        assert [a['reason'] for a in acted] == ['runtime_cap']
        assert destroyed == ['90001']
        ct.db.session.refresh(run)
        assert run.status == 'stopped'


def test_supervisor_enforces_the_stop_deadline(ct, app, monkeypatch):
    """A graceful stop handed to a monitor that then wedges must not linger:
    past the grace window the supervisor terminates the pod itself."""
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 0}})
        run = _mkrun(ct, updated_at=datetime.utcnow(),
                     stop_requested_at=datetime.utcnow() - timedelta(minutes=20))
        destroyed = _stub_destroy(ct, monkeypatch)

        acted = ct.supervise_active_runs()

        assert [a['reason'] for a in acted] == ['stop_deadline']
        assert destroyed == ['90001']
        ct.db.session.refresh(run)
        assert run.status == 'stopped'


def test_supervisor_never_raises(ct, app, monkeypatch):
    """A supervisor that dies is the bug we are fixing — its tick swallows
    everything."""
    with app.app_context():
        ct.cfg.save_config({'cloud': {'freeze_watchdog_minutes': 45}})
        _mkrun(ct, updated_at=datetime.utcnow() - timedelta(hours=3))
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: (_ for _ in ()).throw(RuntimeError('boom')))
        ct.supervise_active_runs()      # must not raise


def test_run_payload_exposes_the_idle_clock(ct, app):
    with app.app_context():
        run = _mkrun(ct, updated_at=datetime.utcnow() - timedelta(minutes=30))
        payload = ct._run_payload(run)
        assert payload['idle_seconds'] >= 29 * 60


# ------------------------------------------------------------- Route --

def test_stop_route_surfaces_the_failure(app, client, monkeypatch):
    from app.services import cloud_training as ct
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    with app.app_context():
        run = _mkrun(ct, updated_at=datetime.utcnow() - timedelta(hours=2))
        run_id = run.id
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: False)
    r = client.post('/api/dataset/train/cloud/stop', json={'run_id': run_id})
    body = r.get_json()
    assert r.status_code == 200
    assert body['ok'] is False and '90001' in body['error']


def test_stopping_can_ban_the_host_but_only_when_the_user_says_so(ct, app, monkeypatch):
    """"Do not rent this machine again", asked for by mr.arrow on Discord.

    It has to be a deliberate tick rather than an automatic rule. The app bans a
    host when a boot fails, stalls, or cannot serve its checkpoint — all of them
    failures it can classify. A pod that boots fine and then simply trains at
    half speed produces NO failure at all, so nothing here would ever ban it;
    the person watching the throughput is the only one who knows. And the
    reverse is just as true: stopping because you changed your mind says nothing
    about the box, which is why a plain stop must leave it alone.
    """
    banned = []
    with app.app_context():
        run = _mkrun(ct, updated_at=datetime.utcnow())
        _stub_destroy(ct, monkeypatch)
        ct._monitor_threads[run.id] = _FakeThread(alive=True)
        monkeypatch.setattr(ct, '_blacklist_run_host',
                            lambda r, reason, ttl_seconds=None: banned.append(reason))

        ct.request_stop(run.id)
        assert banned == [], 'a plain stop must never exile the machine'

    with app.app_context():
        run2 = _mkrun(ct, updated_at=datetime.utcnow())
        ct._monitor_threads[run2.id] = _FakeThread(alive=True)
        ct.request_stop(run2.id, ban_host=True)
        assert len(banned) == 1
        assert 'again' in banned[0].lower()
