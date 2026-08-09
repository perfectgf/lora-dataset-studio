"""Parallel cloud runs on one dataset: the same-family uniqueness guard is a
CONFIRMABLE refusal (PARALLEL_RUN: marker + allow_parallel_run), while the
fleet ceiling and the monthly budget stay hard blocks. vast_client and the
monitor thread are always mocked -- no network."""
import json

import pytest


@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(cloud_training, '_reconcile_before_launch', lambda a: None)
    return cloud_training


@pytest.fixture()
def seeded_dataset(app, client):
    return client.post('/api/dataset/create',
                       json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']


def _fake_export(monkeypatch, ct):
    monkeypatch.setattr(ct.lt, 'export_dataset_to_aitoolkit',
                        lambda uid, did, masked=True, dest_dir=None: dest_dir)
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds, **kw: 1200)
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **kw: None)


def test_second_same_family_launch_refused_with_marker(ct, app, seeded_dataset, monkeypatch):
    """Without allow_parallel_run the sibling refusal carries the PARALLEL_RUN:
    marker (the frontend's confirm contract) and NAMES the run it found."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        first = ct.launch_cloud_training('local', seeded_dataset)
        with pytest.raises(RuntimeError) as e:
            ct.launch_cloud_training('local', seeded_dataset)
        assert str(e.value).startswith('PARALLEL_RUN: ')
        assert f"#{first['run_id']}" in str(e.value)


def test_allow_parallel_run_launches_and_stamps_the_flag(ct, app, seeded_dataset, monkeypatch):
    """With the flag, the second same-family run launches; the flag is stamped
    into train_params (so the Runs hub can explain why two pods exist) and
    replayed by _confirmation_flags."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        second = ct.launch_cloud_training('local', seeded_dataset,
                                          allow_parallel_run=True)
        runs = ct.active_runs_for(seeded_dataset)
        assert len(runs) == 2
        params = json.loads(next(r for r in runs
                                 if r.id == second['run_id']).train_params)
        assert params['allow_parallel_run'] is True
        assert ct._confirmation_flags(params)['allow_parallel_run'] is True


def test_fleet_ceiling_still_hard_even_with_the_flag(ct, app, seeded_dataset, monkeypatch):
    """allow_parallel_run waives the SIBLING refusal only — the account-wide
    ceiling still blocks, with the historical message."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 1}})
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        with pytest.raises(RuntimeError, match='limit reached'):
            ct.launch_cloud_training('local', seeded_dataset,
                                     allow_parallel_run=True)


def test_budget_still_hard_even_with_the_flag(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2,
                                  'monthly_budget_usd': 5}})
    monkeypatch.setattr(ct, 'month_spend_usd', lambda: 9.0)
    with app.app_context():
        with pytest.raises(RuntimeError, match='budget reached'):
            ct.launch_cloud_training('local', seeded_dataset,
                                     allow_parallel_run=True)


def test_unknown_family_sibling_is_not_waivable(ct, app, seeded_dataset, monkeypatch):
    """A sibling whose family cannot be read (corrupt/pre-feature train_params)
    is AMBIGUOUS, not a same-family match — allow_parallel_run answers "yes,
    another <fam> run" and cannot cover a case the guard cannot even name.
    It stays a hard block, with no PARALLEL_RUN: marker."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        run = ct.active_runs_for(seeded_dataset)[0]
        run.train_params = json.dumps({})
        ct.db.session.commit()
        with pytest.raises(RuntimeError) as e:
            ct.launch_cloud_training('local', seeded_dataset,
                                     allow_parallel_run=True)
        assert not str(e.value).startswith('PARALLEL_RUN: ')


def test_run_for_addresses_one_run_and_checks_ownership(ct, app, seeded_dataset, monkeypatch):
    """run_for(run_id=) returns THAT run only when the (id, table) ownership
    holds — a video run sharing the integer id must not be served to a face
    dataset's caller. Without run_id it is latest_run_for, unchanged."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        first = ct.launch_cloud_training('local', seeded_dataset)
        second = ct.launch_cloud_training('local', seeded_dataset,
                                          allow_parallel_run=True)
        got = ct.run_for(seeded_dataset, run_id=first['run_id'])
        assert got is not None and got.id == first['run_id']
        # default = newest, exactly latest_run_for
        assert ct.run_for(seeded_dataset).id == second['run_id']
        # unknown id -> None, never a fallback to the newest
        assert ct.run_for(seeded_dataset, run_id=999999) is None
        # foreign table -> None
        from app.services import cloud_run_dataset as crd
        assert ct.run_for(seeded_dataset, run_id=first['run_id'],
                          dataset_table=crd.VIDEO) is None


def test_progress_route_404s_on_unknown_run_id(ct, app, client, seeded_dataset, monkeypatch):
    """?run_id= must answer for THAT run or not at all: an unknown id is 404,
    not the newest run wearing the wrong number."""
    _fake_export(monkeypatch, ct)
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
    r = client.get(f'/api/dataset/{seeded_dataset}/train/cloud/progress?run_id=999999')
    assert r.status_code == 404


def test_progress_route_serves_the_addressed_run(ct, app, client, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        first = ct.launch_cloud_training('local', seeded_dataset)
        ct.launch_cloud_training('local', seeded_dataset, allow_parallel_run=True)
    r = client.get(f'/api/dataset/{seeded_dataset}/train/cloud/progress'
                   f'?run_id={first["run_id"]}')
    assert r.status_code == 200
    assert r.get_json()['run_id'] == first['run_id']


def test_progress_route_404s_on_malformed_run_id(ct, app, client, seeded_dataset, monkeypatch):
    """?run_id=abc must not silently fall back to the newest run wearing the
    wrong number — request.args.get('run_id', type=int) would parse it as
    None and do exactly that."""
    _fake_export(monkeypatch, ct)
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
    r = client.get(f'/api/dataset/{seeded_dataset}/train/cloud/progress?run_id=abc')
    assert r.status_code == 404


def test_sample_route_404s_on_malformed_run_id(ct, app, client, seeded_dataset, monkeypatch):
    """Same silent-fallback trap as the progress route, on the sample route."""
    _fake_export(monkeypatch, ct)
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
    r = client.get(f'/api/dataset/{seeded_dataset}/train/cloud/sample/x.jpg?run_id=abc')
    assert r.status_code == 404


def test_checkpoint_route_404s_on_malformed_run_id(ct, app, client, seeded_dataset, monkeypatch):
    """Same silent-fallback trap as the progress/sample routes — this one
    hands out FILES, the worst place to serve a stranger's weights for a
    typo'd id."""
    _fake_export(monkeypatch, ct)
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
    r = client.get(f'/api/dataset/{seeded_dataset}/train/cloud/checkpoint?run_id=abc')
    assert r.status_code == 404


def test_frozen_generation_waits_for_the_lease(ct, app, seeded_dataset, monkeypatch):
    """A busy lease is retried until the deadline instead of failing on the
    spot — clicking Launch twice quickly is the feature's natural gesture, and
    'this dataset already has work in progress' would describe a problem the
    user does not have."""
    from app.services import dataset_activity
    real = dataset_activity.begin_exclusive
    calls = {'n': 0}

    def busy_then_free(*a, **kw):
        calls['n'] += 1
        return None if calls['n'] < 3 else real(*a, **kw)

    monkeypatch.setattr(dataset_activity, 'begin_exclusive', busy_then_free)
    sleeps = []
    monkeypatch.setattr(ct, '_wait_sleep', lambda s: sleeps.append(s))
    waits = {'n': 0}
    with app.app_context():
        out = ct._with_frozen_dataset_generation(
            'local', seeded_dataset, 'test', lambda: 'ran', wait_seconds=60,
            on_wait=lambda: waits.__setitem__('n', waits['n'] + 1))
    assert out == 'ran' and calls['n'] == 3
    # busy twice -> two retries -> two sleeps; on_wait is a heartbeat now, so
    # it ticks on EVERY iteration that goes on to sleep (not once) — a run
    # waiting close to wait_seconds must keep writing, or its monitor reads as
    # unresponsive to a Stop long before the wait is over.
    assert len(sleeps) == 2
    assert waits['n'] == 2


def test_frozen_generation_still_fails_fast_by_default(ct, app, seeded_dataset, monkeypatch):
    from app.services import dataset_activity
    calls = {'n': 0}

    def always_busy(*a, **kw):
        calls['n'] += 1
        return None

    def raise_if_called(s):
        raise AssertionError('default wait_seconds=0 must not sleep')

    monkeypatch.setattr(dataset_activity, 'begin_exclusive', always_busy)
    monkeypatch.setattr(ct, '_wait_sleep', raise_if_called)
    with app.app_context():
        with pytest.raises(dataset_activity.DatasetActivityBusy):
            ct._with_frozen_dataset_generation(
                'local', seeded_dataset, 'test', lambda: 'ran')
    # a single attempt, no retry budget
    assert calls['n'] == 1


def test_frozen_generation_raises_busy_after_deadline_expires(ct, app, seeded_dataset, monkeypatch):
    """A lease that never frees still ends the wait — the retry loop is
    bounded by wall-clock time, not by luck."""
    from app.services import dataset_activity
    monkeypatch.setattr(dataset_activity, 'begin_exclusive', lambda *a, **kw: None)
    clock = {'t': 0.0}
    monkeypatch.setattr(ct, '_wait_clock', lambda: clock['t'])

    def fake_sleep(s):
        clock['t'] += s

    monkeypatch.setattr(ct, '_wait_sleep', fake_sleep)
    with app.app_context():
        with pytest.raises(dataset_activity.DatasetActivityBusy):
            ct._with_frozen_dataset_generation(
                'local', seeded_dataset, 'test', lambda: 'ran',
                wait_seconds=5)
    assert clock['t'] >= 5


def test_frozen_generation_bounded_wait_does_not_block_on_a_held_lock(ct, app, seeded_dataset, monkeypatch):
    """The primary scenario for this feature: a sibling run HOLDS the ingest
    lock for its whole export. A waiting launch must not hang in an
    uninterruptible acquire past its own deadline — it must poll the lock
    with a bounded timeout and still get retry budget. Held from a SEPARATE
    thread: the lock is an RLock, so acquiring it again from this test's own
    thread would silently succeed and prove nothing."""
    import threading as _threading
    from app.services import dataset_activity

    monkeypatch.setattr(ct, '_wait_sleep', lambda s: None)
    lock = ct.fds._dataset_ingest_lock('local', seeded_dataset)
    held = _threading.Event()
    release = _threading.Event()

    def hold_lock():
        with lock:
            held.set()
            release.wait(10.0)

    holder = _threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    held.wait(5.0)
    try:
        with app.app_context():
            with pytest.raises(dataset_activity.DatasetActivityBusy):
                ct._with_frozen_dataset_generation(
                    'local', seeded_dataset, 'test', lambda: 'ran',
                    wait_seconds=3)
    finally:
        release.set()
        holder.join(timeout=5.0)


def test_frozen_generation_aborts_when_should_abort_fires(ct, app, seeded_dataset, monkeypatch):
    """A Stop during the wait must not leave the caller waiting up to
    wait_seconds and then exporting for a dead run."""
    from app.services import dataset_activity
    monkeypatch.setattr(dataset_activity, 'begin_exclusive', lambda *a, **kw: None)
    monkeypatch.setattr(ct, '_wait_sleep', lambda s: None)
    with app.app_context():
        with pytest.raises(RuntimeError, match='stop requested'):
            ct._with_frozen_dataset_generation(
                'local', seeded_dataset, 'test', lambda: 'ran',
                wait_seconds=60, should_abort=lambda: True)


def test_continue_cloud_run_forwards_allow_parallel_run(ct, app, seeded_dataset, monkeypatch):
    """The Runs hub's ▶ Continue relaunches through launch_cloud_training, so it
    hits the same sibling guard — the dialog's confirm must be able to answer
    it. A fresh allow_parallel_run=True widens the flags inherited from the
    source run (which may have been launched alone, stamped False)."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        first = ct.launch_cloud_training('local', seeded_dataset)
        run = ct.CloudTrainingRun.query.get(first['run_id'])
        run.status = 'done'
        ct.db.session.commit()
        monkeypatch.setattr(ct, '_run_staging_checkpoints', lambda r: [
            {'step': 500, 'filename': 'x_000000500.safetensors',
             'path': 'x_000000500.safetensors', 'source': 'local'}])
        seen = {}

        def fake_launch(user_id, dataset_id, **kw):
            seen.update(kw)
            return {'run_id': 4242}

        monkeypatch.setattr(ct, 'launch_cloud_training', fake_launch)
        ct.continue_cloud_run('local', run.id, allow_parallel_run=True)
        assert seen.get('allow_parallel_run') is True
        # and WITHOUT the fresh confirm, the source run's own answer replays
        seen.clear()
        ct.continue_cloud_run('local', run.id)
        assert seen.get('allow_parallel_run') is False


def test_continue_local_run_in_cloud_forwards_allow_parallel_run(
        ct, app, seeded_dataset, monkeypatch):
    """The dataset panel's Continue→Cloud lane is a normal cloud launch too:
    without the passthrough, the dialog's confirm loop resends a flag the
    service drops on the floor and the refusal comes back forever."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        monkeypatch.setattr(ct.lt, 'list_checkpoints',
                            lambda *a, **kw: [{'step': 500,
                                               'filename': 'f_000000500.safetensors',
                                               'final': False, 'record_id': None}])
        monkeypatch.setattr(ct.lt, 'checkpoint_file_path',
                            lambda *a, **kw: 'f_000000500.safetensors')
        seen = {}

        def fake_launch(user_id, dataset_id, **kw):
            seen.update(kw)
            return {'run_id': 4243}

        monkeypatch.setattr(ct, 'launch_cloud_training', fake_launch)
        ct.continue_local_run_in_cloud('local', seeded_dataset,
                                       allow_parallel_run=True)
        assert seen.get('allow_parallel_run') is True
        seen.clear()
        ct.continue_local_run_in_cloud('local', seeded_dataset)
        assert seen.get('allow_parallel_run') is False


def test_continue_routes_read_the_flag_from_the_request(
        ct, app, client, seeded_dataset, monkeypatch):
    """Both HTTP routes must hand the JSON flag to their service — the frontend
    confirm loop retries through THEM, not through the services directly."""
    from app.routes import training as troutes
    seen = {}

    def fake_continue_cloud(user_id, run_id, **kw):
        seen['cloud'] = kw.get('allow_parallel_run')
        return {}

    def fake_continue_local(user_id, dataset_id, **kw):
        seen['local'] = kw.get('allow_parallel_run')
        return {}

    monkeypatch.setattr(troutes.ct, 'continue_cloud_run', fake_continue_cloud)
    monkeypatch.setattr(troutes.ct, 'continue_local_run_in_cloud', fake_continue_local)
    client.post('/api/dataset/train/cloud/continue',
                json={'run_id': 1, 'allow_parallel_run': True})
    client.post(f'/api/dataset/{seeded_dataset}/train/cloud/continue-local',
                json={'extra_steps': 500, 'allow_parallel_run': True})
    assert seen.get('cloud') is True
    assert seen.get('local') is True


def test_auto_retry_of_a_run_with_a_live_confirmed_sibling_still_launches(
        ct, app, seeded_dataset, monkeypatch):
    """Run A launched alone (allow_parallel_run stamped False) and run B
    launched with the confirm (allow_parallel_run True) are both active on
    the same dataset/family. A's pod then fails transiently. The auto-retry
    replay must not raise the PARALLEL_RUN: refusal just because A's OWN
    stamped params never answered it — an auto-retry replaces a run whose
    pod is already dead, so the live sibling's same-family guard must not
    block it (the fleet ceiling and monthly budget still guard the spend)."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        first = ct.launch_cloud_training('local', seeded_dataset)
        ct.launch_cloud_training('local', seeded_dataset, allow_parallel_run=True)
        run_a = ct.CloudTrainingRun.query.get(first['run_id'])
        params = json.loads(run_a.train_params)
        assert params['allow_parallel_run'] is False   # A never confirmed
        run_a.status = 'error'
        run_a.vast_instance_id = 'old-pod'
        run_a.error = 'connection reset by peer'
        ct.db.session.commit()

        result = ct._maybe_auto_retry(run_a, run_a.error)

        assert result is not None
        assert not str(run_a.error or '').startswith(
            "PARALLEL_RUN") and 'PARALLEL_RUN' not in str(run_a.error or '')
        child = ct.CloudTrainingRun.query.get(result['run_id'])
        assert child is not None and child.id != run_a.id
        child_params = json.loads(child.train_params)
        assert child_params['allow_parallel_run'] is True
        assert child_params['auto_retry_of'] == run_a.id
