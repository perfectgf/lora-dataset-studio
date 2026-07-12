"""Launch validation, LEAK-SAFE provisioning (the property that matters),
stop request, and boot reconciliation. vast_client and the monitor thread are
always mocked -- no network, no thread started for real."""
import json
from datetime import datetime, timedelta

import pytest


@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training
    # never start the real monitor thread in launch tests
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    # launch_cloud_training now reconciles orphans on every call (so a user
    # coming back days later to a launch reaps an expired error_pod_kept pod
    # too, not just at boot) -- no-op that call site here so plain
    # launch/provision tests stay offline. Patching the seam (not
    # reconcile_orphans itself) leaves the reconcile-policy tests below,
    # which call reconcile_orphans() directly, exercising the real thing.
    monkeypatch.setattr(cloud_training, '_reconcile_before_launch', lambda a: None)
    return cloud_training


@pytest.fixture()
def seeded_dataset(app, client):
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']
    return ds_id


def _fake_export(monkeypatch, ct):
    monkeypatch.setattr(ct.lt, 'export_dataset_to_aitoolkit',
                        lambda uid, did, masked=True, dest_dir=None: dest_dir)
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds: 1200)
    # The seeded_dataset fixture has 0 kept images -- the real assert_trainable
    # (lora_training.py, already a standalone helper: dataset_id, train_type=None,
    # allow_caption_mismatch=False) requires >= 10, which is orthogonal to what
    # these launch/provision/reconcile tests exercise. Stub it out here so launch
    # reaches the orchestration code; the caption-mismatch contract itself is
    # covered by lora_training's own tests.
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **kw: None)


def test_launch_rejects_custom_base(ct, app, seeded_dataset, monkeypatch):
    with app.app_context():
        with pytest.raises(ValueError, match='local'):
            ct.launch_cloud_training('local', seeded_dataset, base_model='myBase.safetensors')


def test_launch_rejects_sdxl(ct, app, seeded_dataset):
    with app.app_context():
        with pytest.raises(ValueError, match='SDXL'):
            ct.launch_cloud_training('local', seeded_dataset, train_type='sdxl')


def test_launch_without_key_raises(app, seeded_dataset, monkeypatch):
    monkeypatch.delenv('VAST_API_KEY', raising=False)
    from app.services import cloud_training as ct
    with app.app_context():
        with pytest.raises(RuntimeError, match='key'):
            ct.launch_cloud_training('local', seeded_dataset)


def test_launch_creates_run_and_staging(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    with app.app_context():
        res = ct.launch_cloud_training('local', seeded_dataset)
        assert res['status'] == 'preparing'
        assert res['steps'] == 1200
        run = ct.get_active_run()
        assert run is not None and run.dataset_id == seeded_dataset
        assert run.vast_label == f"lds-{run.id}"
        assert run.job_name.startswith('lds')


def test_launch_refuses_second_active_run(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        with pytest.raises(RuntimeError, match='already'):
            ct.launch_cloud_training('local', seeded_dataset)


def test_provision_registers_instance(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    monkeypatch.setattr(ct.vast_client, 'search_offers',
                        lambda **kw: [{'offer_id': 9, 'gpu_name': 'RTX 4090',
                                       'dph_total': 0.4, 'gpu_ram_gb': 24.0}])
    monkeypatch.setattr(ct.vast_client, 'create_instance', lambda *a, **kw: '777')
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        run = ct.get_active_run()
        ct._provision(run)
        assert run.vast_instance_id == '777'
        assert run.price_per_hour == 0.4
        assert run.status == 'provisioning'
        # template mode: the auth token is vast's per-instance jupyter_token,
        # picked up during boot-wait -- empty right after provisioning
        assert run.auth_token == ''


def test_provision_no_offer_fails_cleanly(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: [])
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        run = ct.get_active_run()
        with pytest.raises(RuntimeError, match='offer'):
            ct._provision(run)


def test_provision_leak_safe_on_post_create_failure(ct, app, seeded_dataset, monkeypatch):
    """THE test: if anything fails after create_instance, the pod is destroyed."""
    _fake_export(monkeypatch, ct)
    destroyed = []
    monkeypatch.setattr(ct.vast_client, 'search_offers',
                        lambda **kw: [{'offer_id': 9, 'gpu_name': 'g', 'dph_total': 0.4,
                                       'gpu_ram_gb': 24.0}])
    monkeypatch.setattr(ct.vast_client, 'create_instance', lambda *a, **kw: '777')
    monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                        lambda iid: destroyed.append(iid) or True)
    # make the post-create registration explode
    monkeypatch.setattr(ct, '_register_instance',
                        lambda run, iid, offer, token: (_ for _ in ()).throw(OSError('db gone')))
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        run = ct.get_active_run()
        with pytest.raises(OSError):
            ct._provision(run)
        assert destroyed == ['777']


def test_reconcile_destroys_orphans_keeps_active(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    destroyed = []
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        run = ct.get_active_run()
        run.vast_instance_id = '111'
        ct.db.session.commit()
        monkeypatch.setattr(ct.vast_client, 'list_instances', lambda: [
            {'instance_id': '111', 'label': f'lds-{run.id}'},   # active -> keep
            {'instance_id': '222', 'label': 'lds-99'},          # orphan -> destroy
            {'instance_id': '333', 'label': 'other-app'},       # not ours -> keep
        ])
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: destroyed.append(iid) or True)
        n = ct.reconcile_orphans(app)
        assert destroyed == ['222']
        assert n == 1


def test_reconcile_without_key_is_noop(app, monkeypatch):
    monkeypatch.delenv('VAST_API_KEY', raising=False)
    from app.services import cloud_training as ct
    assert ct.reconcile_orphans(app) == 0


def test_reconcile_never_raises(ct, app, monkeypatch):
    """Boot must never be blocked: even an unexpected failure OUTSIDE the
    vast_client calls (db not ready, config error...) is swallowed and logged."""
    monkeypatch.setattr(ct, 'get_active_run',
                        lambda: (_ for _ in ()).throw(RuntimeError('db not ready')))
    monkeypatch.setattr(ct.vast_client, 'list_instances', lambda: [])
    assert ct.reconcile_orphans(app) == 0      # swallowed, boot not blocked


def test_reconcile_spares_recent_error_pod_kept(ct, app, monkeypatch):
    """A run left in 'error_pod_kept' deliberately keeps its pod alive so the
    user can recover the checkpoint by hand. Within cloud.max_runtime_minutes
    of run.finished_at, reconciliation must NOT destroy that pod -- otherwise
    the manual-recovery window would never actually exist."""
    destroyed = []
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='error_pod_kept',
                                  vast_instance_id='555', vast_label='lds-1',
                                  job_name='j', error='checkpoint download failed',
                                  finished_at=datetime.utcnow() - timedelta(minutes=10))
        ct.db.session.add(run)
        ct.db.session.commit()
        monkeypatch.setattr(ct.vast_client, 'list_instances',
                            lambda: [{'instance_id': '555', 'label': f'lds-{run.id}'}])
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: destroyed.append(iid) or True)
        n = ct.reconcile_orphans(app)
        assert destroyed == []
        assert n == 0
        # reconcile_orphans() ran its own nested app_context/session; the
        # mock's list_instances lambda (referencing run.id) forced an
        # implicit refresh -- and therefore a pinned read snapshot -- on
        # THIS (outer) session mid-call. expire_all() drops that pinned
        # snapshot so the assertions below see what was actually committed,
        # not a transaction-start-time view.
        ct.db.session.expire_all()
        kept = ct.CloudTrainingRun.query.get(run.id)
        assert kept.status == 'error_pod_kept'
        assert kept.error == 'checkpoint download failed'   # untouched


def test_reconcile_reaps_expired_error_pod_kept(ct, app, monkeypatch):
    """Past the recovery window, the kept pod IS destroyed like any other
    orphan, and the run is annotated -- but its terminal status must stay
    'error_pod_kept' (not flipped to something else)."""
    destroyed = []
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='error_pod_kept',
                                  vast_instance_id='555', vast_label='lds-1',
                                  job_name='j', error='checkpoint download failed',
                                  finished_at=datetime.utcnow() - timedelta(minutes=500))
        ct.db.session.add(run)
        ct.db.session.commit()
        monkeypatch.setattr(ct.vast_client, 'list_instances',
                            lambda: [{'instance_id': '555', 'label': f'lds-{run.id}'}])
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: destroyed.append(iid) or True)
        n = ct.reconcile_orphans(app)
        assert destroyed == ['555']
        assert n == 1
        # see the sibling test above for why expire_all() is needed here
        ct.db.session.expire_all()
        kept = ct.CloudTrainingRun.query.get(run.id)
        assert kept.status == 'error_pod_kept'               # terminal stays terminal
        assert kept.error.startswith('checkpoint download failed')
        assert kept.error.endswith('pod reaped after the recovery window')


def test_reconcile_error_pod_kept_absent_from_instances_is_noop(ct, app, monkeypatch):
    """The kept pod may already be gone (destroyed by hand, or a previous
    reconcile pass) -- if vast.ai no longer lists it, there is nothing to
    destroy or annotate."""
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=1, status='error_pod_kept',
                                  vast_instance_id='555', vast_label='lds-1',
                                  job_name='j', error='checkpoint download failed',
                                  finished_at=datetime.utcnow() - timedelta(minutes=500))
        ct.db.session.add(run)
        ct.db.session.commit()
        monkeypatch.setattr(ct.vast_client, 'list_instances', lambda: [])
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: (_ for _ in ()).throw(
                                AssertionError('nothing to destroy')))
        n = ct.reconcile_orphans(app)
        assert n == 0
        ct.db.session.expire_all()
        kept = ct.CloudTrainingRun.query.get(run.id)
        assert kept.error == 'checkpoint download failed'   # untouched


def test_reconcile_keeps_active_and_spares_error_pod_kept_together(ct, app, monkeypatch):
    """One reconcile pass must apply both policies at once: keep the truly
    active run's pod, spare the still-recoverable error_pod_kept pod, and
    destroy the plain orphan."""
    destroyed = []
    with app.app_context():
        active = ct.CloudTrainingRun(dataset_id=1, status='training',
                                     vast_instance_id='111', vast_label='lds-1',
                                     job_name='j1')
        kept_run = ct.CloudTrainingRun(dataset_id=2, status='error_pod_kept',
                                       vast_instance_id='555', vast_label='lds-2',
                                       job_name='j2', error='checkpoint download failed',
                                       finished_at=datetime.utcnow() - timedelta(minutes=10))
        ct.db.session.add_all([active, kept_run])
        ct.db.session.commit()
        active_id, kept_id = active.id, kept_run.id
        monkeypatch.setattr(ct.vast_client, 'list_instances', lambda: [
            {'instance_id': '111', 'label': f'lds-{active_id}'},   # active -> keep
            {'instance_id': '555', 'label': f'lds-{kept_id}'},     # recoverable -> spare
            {'instance_id': '222', 'label': 'lds-99'},             # orphan -> destroy
        ])
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: destroyed.append(iid) or True)
        n = ct.reconcile_orphans(app)
        assert destroyed == ['222']
        assert n == 1


def test_launch_respects_higher_concurrent_limit(ct, app, client, monkeypatch):
    """cloud.max_concurrent_runs=2 + 2 different datasets -> both launches
    succeed; a 3rd dataset trips the limit guard."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    ds1 = client.post('/api/dataset/create',
                      json={'name': 'A', 'trigger_word': 'a'}).get_json()['id']
    ds2 = client.post('/api/dataset/create',
                      json={'name': 'B', 'trigger_word': 'b'}).get_json()['id']
    ds3 = client.post('/api/dataset/create',
                      json={'name': 'C', 'trigger_word': 'c'}).get_json()['id']
    with app.app_context():
        ct.launch_cloud_training('local', ds1)
        ct.launch_cloud_training('local', ds2)
        with pytest.raises(RuntimeError, match='limit reached'):
            ct.launch_cloud_training('local', ds3)


def test_launch_refuses_same_dataset_twice_even_with_higher_limit(ct, app, client, monkeypatch):
    """The per-(dataset, family) uniqueness guard is independent of the
    concurrency cap: even with room under the limit, the SAME dataset cannot
    get a 2nd run of the SAME family (both launches default to zimage here)."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    ds1 = client.post('/api/dataset/create',
                      json={'name': 'A', 'trigger_word': 'a'}).get_json()['id']
    with app.app_context():
        ct.launch_cloud_training('local', ds1)
        with pytest.raises(RuntimeError, match='already has an active .*cloud run'):
            ct.launch_cloud_training('local', ds1)


def test_request_stop_targets_only_the_given_run(ct, app, client, monkeypatch):
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    ds1 = client.post('/api/dataset/create',
                      json={'name': 'A', 'trigger_word': 'a'}).get_json()['id']
    ds2 = client.post('/api/dataset/create',
                      json={'name': 'B', 'trigger_word': 'b'}).get_json()['id']
    with app.app_context():
        r1 = ct.launch_cloud_training('local', ds1)
        r2 = ct.launch_cloud_training('local', ds2)
        assert ct.request_stop(r1['run_id']) is True
        assert ct._stop_event_for(r1['run_id']).is_set() is True
        assert ct._stop_event_for(r2['run_id']).is_set() is False


def test_reconcile_keeps_multiple_actives_destroys_orphan(ct, app, monkeypatch):
    """Multi-run keep-set: TWO genuinely active runs (different datasets, both
    with a pod) must both be spared; only the true orphan is destroyed."""
    destroyed = []
    with app.app_context():
        active1 = ct.CloudTrainingRun(dataset_id=1, status='training',
                                      vast_instance_id='111', vast_label='lds-1',
                                      job_name='j1')
        active2 = ct.CloudTrainingRun(dataset_id=2, status='uploading',
                                      vast_instance_id='222', vast_label='lds-2',
                                      job_name='j2')
        ct.db.session.add_all([active1, active2])
        ct.db.session.commit()
        a1_id, a2_id = active1.id, active2.id
        monkeypatch.setattr(ct.vast_client, 'list_instances', lambda: [
            {'instance_id': '111', 'label': f'lds-{a1_id}'},   # active -> keep
            {'instance_id': '222', 'label': f'lds-{a2_id}'},   # active -> keep
            {'instance_id': '333', 'label': 'lds-99'},         # orphan -> destroy
        ])
        monkeypatch.setattr(ct.vast_client, 'destroy_instance',
                            lambda iid: destroyed.append(iid) or True)
        n = ct.reconcile_orphans(app)
        assert destroyed == ['333']
        assert n == 1


def test_launch_failure_frees_the_active_slot(ct, app, seeded_dataset, monkeypatch):
    """A mid-launch failure (after the row is created) must not strand a
    'preparing' row forever -- the run flips to 'error' so the single active
    slot is freed for the next launch."""
    monkeypatch.setattr(ct.lt, 'assert_trainable', lambda *a, **kw: None)
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds: 100)
    monkeypatch.setattr(ct.lt, 'export_dataset_to_aitoolkit',
                        lambda *a, **kw: (_ for _ in ()).throw(OSError('disk full')))
    with app.app_context():
        with pytest.raises(OSError):
            ct.launch_cloud_training('local', seeded_dataset)
        assert ct.get_active_run() is None        # slot freed
        run = ct.CloudTrainingRun.query.first()
        assert run.status == 'error' and 'disk full' in run.error


# --- Monthly budget guard: block LAUNCHES only, never kill a running pod ----

def _seed_finished_run(ct, price, start_h, end_h, dataset_id=999):
    """A terminal run UNAMBIGUOUSLY inside the current month: timestamps are
    anchored to the month start (created = month_start + start_h, finished =
    month_start + end_h), never to `now` — a now-relative seed run during the
    first UTC hours of the 1st would land in the PREVIOUS month and genuinely
    fail the spend assertions. cost = price x (end_h - start_h)."""
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1)
    run = ct.CloudTrainingRun(
        dataset_id=dataset_id, status='done', job_name='j', vast_label='lds-9',
        price_per_hour=price,
        created_at=month_start + timedelta(hours=start_h),
        finished_at=month_start + timedelta(hours=end_h))
    ct.db.session.add(run)
    ct.db.session.commit()
    return run


def test_budget_zero_never_blocks_launch(ct, app, seeded_dataset, monkeypatch):
    """monthly_budget_usd=0 (the default) means unlimited: heavy spend this
    month must not block anything."""
    _fake_export(monkeypatch, ct)
    with app.app_context():
        _seed_finished_run(ct, price=2.0, start_h=0, end_h=19)   # 19 h x $2 = $38
        res = ct.launch_cloud_training('local', seeded_dataset)
        assert res['status'] == 'preparing'


def test_budget_reached_blocks_launch(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'monthly_budget_usd': 3}})
    with app.app_context():
        # 0.5 $/h x 8 h = $4 spent >= $3 budget
        _seed_finished_run(ct, price=0.5, start_h=0, end_h=8)
        with pytest.raises(RuntimeError, match='budget'):
            ct.launch_cloud_training('local', seeded_dataset)


def test_budget_ignores_previous_month_runs(ct, app, seeded_dataset, monkeypatch):
    """Only runs STARTED since the 1st of the current month (UTC) count."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'monthly_budget_usd': 3}})
    with app.app_context():
        now = datetime.utcnow()
        month_start = datetime(now.year, now.month, 1)
        run = ct.CloudTrainingRun(
            dataset_id=999, status='done', job_name='j', vast_label='lds-9',
            price_per_hour=10.0,                            # $240 — last month
            created_at=month_start - timedelta(days=5),
            finished_at=month_start - timedelta(days=4))
        ct.db.session.add(run)
        ct.db.session.commit()
        res = ct.launch_cloud_training('local', seeded_dataset)
        assert res['status'] == 'preparing'


def test_cloud_status_reports_month_spend_budget_and_cap(ct, app, monkeypatch):
    ct.cfg.save_config({'cloud': {'monthly_budget_usd': 20}})
    with app.app_context():
        # 0.5 $/h x 4 h = $2.00
        _seed_finished_run(ct, price=0.5, start_h=0, end_h=4)
        # a priced-less run (crashed before provisioning) must count for $0
        _seed_finished_run(ct, price=None, start_h=0, end_h=1, dataset_id=998)
        s = ct.cloud_status()
        assert s['monthly_budget'] == 20
        assert s['month_spend'] == 2.0
        assert s['max_runtime_minutes'] == 480


# --- Per-(dataset, family) uniqueness: a zimage run and a krea run may share
# --- one dataset; two runs of the SAME family on one dataset may not. -------

def test_launch_allows_two_families_on_same_dataset(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        r1 = ct.launch_cloud_training('local', seeded_dataset, train_type='zimage')
        r2 = ct.launch_cloud_training('local', seeded_dataset, train_type='krea')
        assert r1['run_id'] != r2['run_id']
        assert len(ct.get_active_runs()) == 2


def test_launch_refuses_same_family_on_same_dataset(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 2}})
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset, train_type='krea')
        with pytest.raises(RuntimeError, match='already has an active krea cloud run'):
            ct.launch_cloud_training('local', seeded_dataset, train_type='krea')


def test_run_family_non_dict_json_degrades_to_none(ct, app, seeded_dataset):
    """train_params containing valid-but-non-dict JSON must yield None, never
    raise — one corrupt row would 500 cloud_status platform-wide."""
    from app.models import CloudTrainingRun
    with app.app_context():
        for bad in ('"x"', '[1]', '3'):
            run = CloudTrainingRun(dataset_id=seeded_dataset, status='error',
                                   vast_label='lds-x', train_params=bad)
            assert ct._run_family(run) is None
            assert ct._run_payload(run)['train_type'] is None


def test_launch_family_unknown_active_run_blocks_every_family(ct, app, seeded_dataset, monkeypatch):
    """An active run with no train_params (pre-feature row, or the 'preparing'
    window before the params are stamped) has an unknown family — out of
    caution it must block launches of ANY family on that dataset."""
    _fake_export(monkeypatch, ct)
    ct.cfg.save_config({'cloud': {'max_concurrent_runs': 3}})
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=seeded_dataset, status='training',
                                  vast_label='lds-1', job_name='j')   # train_params NULL
        ct.db.session.add(run)
        ct.db.session.commit()
        for fam in ('zimage', 'krea'):
            with pytest.raises(RuntimeError, match='already has an active'):
                ct.launch_cloud_training('local', seeded_dataset, train_type=fam)


def test_run_payload_carries_train_type(ct, app):
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=1, status='training', job_name='j', vast_label='lds-1',
            train_params=json.dumps({'train_type': 'krea', 'steps': 100}))
        ct.db.session.add(run)
        # defensive: corrupted params -> None, never a crash
        bad = ct.CloudTrainingRun(dataset_id=2, status='training', job_name='j2',
                                  vast_label='lds-2', train_params='{not json')
        ct.db.session.add(bad)
        ct.db.session.commit()
        assert ct._run_payload(run)['train_type'] == 'krea'
        assert ct._run_payload(bad)['train_type'] is None


def test_run_payload_carries_dataset_name_and_run_name(ct, app, client):
    ds = client.post('/api/dataset/create',
                     json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']
    with app.app_context():
        run = ct.CloudTrainingRun(dataset_id=ds, status='training', job_name='j',
                                  vast_label='lds-1', run_name='lola_krea')
        ct.db.session.add(run)
        ct.db.session.commit()
        p = ct._run_payload(run)
        assert p['dataset_name'] == 'Lola' and p['run_name'] == 'lola_krea'
        # a since-deleted dataset degrades to None, never a crash
        orphan = ct.CloudTrainingRun(dataset_id=999999, status='training',
                                     job_name='j', vast_label='lds-2')
        ct.db.session.add(orphan)
        ct.db.session.commit()
        assert ct._run_payload(orphan)['dataset_name'] is None


def test_all_runs_splits_active_and_recent(ct, app):
    with app.app_context():
        active = ct.CloudTrainingRun(dataset_id=1, status='training',
                                     job_name='j', vast_label='lds-1',
                                     price_per_hour=0.4)
        done1 = ct.CloudTrainingRun(dataset_id=2, status='done', job_name='j',
                                    vast_label='lds-2')
        done2 = ct.CloudTrainingRun(dataset_id=3, status='error', job_name='j',
                                    vast_label='lds-3')
        ct.db.session.add_all([active, done1, done2])
        ct.db.session.commit()
        out = ct.all_runs()
        assert [r['status'] for r in out['actives']] == ['training']
        # terminal runs, newest first
        assert [r['run_id'] for r in out['recent']] == [done2.id, done1.id]
        assert out['total_price_per_hour'] == 0.4
        assert 'month_spend' in out and 'monthly_budget' in out


def test_all_runs_respects_limit(ct, app):
    with app.app_context():
        for i in range(5):
            ct.db.session.add(ct.CloudTrainingRun(
                dataset_id=i, status='done', job_name='j', vast_label=f'lds-{i}'))
        ct.db.session.commit()
        assert len(ct.all_runs(limit=3)['recent']) == 3


# --- Launch-time GPU speed picker: requested_gpu is a preference, not a lock ---

def test_launch_stores_requested_gpu(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset, gpu_name='RTX 5090')
        run = ct.get_active_run()
        assert json.loads(run.train_params)['requested_gpu'] == 'RTX 5090'


def test_launch_without_gpu_name_omits_requested_gpu(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        run = ct.get_active_run()
        assert 'requested_gpu' not in json.loads(run.train_params)


def test_pick_offer_prefers_requested_class_cheapest():
    from app.services import cloud_training as ct
    offers = [                                    # already cheapest-first
        {'offer_id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.12},
        {'offer_id': 2, 'gpu_name': 'RTX 5090', 'dph_total': 0.60},
        {'offer_id': 3, 'gpu_name': 'RTX 5090', 'dph_total': 0.55},
    ]
    assert ct._pick_offer(offers, 'RTX 5090')['offer_id'] == 3   # cheapest 5090
    assert ct._pick_offer(offers, None)['offer_id'] == 1         # global cheapest


def test_pick_offer_falls_back_when_class_sold_out():
    from app.services import cloud_training as ct
    offers = [{'offer_id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.12}]
    # requested class no longer on the market -> cheapest overall
    assert ct._pick_offer(offers, 'RTX 5090')['offer_id'] == 1


def test_provision_honors_requested_gpu(ct, app, seeded_dataset, monkeypatch):
    _fake_export(monkeypatch, ct)
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: [
        {'offer_id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.12, 'gpu_ram_gb': 24.0},
        {'offer_id': 2, 'gpu_name': 'RTX 5090', 'dph_total': 0.60, 'gpu_ram_gb': 32.0},
    ])
    created = {}
    monkeypatch.setattr(ct.vast_client, 'create_instance',
                        lambda offer_id, **kw: created.setdefault('offer_id', offer_id) or '777')
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset, gpu_name='RTX 5090')
        run = ct.get_active_run()
        ct._provision(run)
        assert created['offer_id'] == 2          # the 5090, not the cheaper 3090
        assert run.price_per_hour == 0.60


def _offers_multi():
    return [
        {'offer_id': 1, 'gpu_name': 'RTX 3090', 'dph_total': 0.13, 'gpu_ram_gb': 24.0},
        {'offer_id': 2, 'gpu_name': 'RTX 3090', 'dph_total': 0.18, 'gpu_ram_gb': 24.0},
        {'offer_id': 3, 'gpu_name': 'RTX 5090', 'dph_total': 0.69, 'gpu_ram_gb': 32.0},
        {'offer_id': 4, 'gpu_name': 'RTX 4090', 'dph_total': 0.35, 'gpu_ram_gb': 24.0},
    ]


def test_gpu_tiers_groups_ranks_and_estimates(ct, app, seeded_dataset, monkeypatch):
    monkeypatch.setattr(ct.lt, 'default_steps', lambda ds: 3000)
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: _offers_multi())
    with app.app_context():
        out = ct.gpu_tiers('local', seeded_dataset, train_type='krea')
        tiers = out['tiers']
        assert out['steps'] == 3000 and out['family'] == 'krea'
        # one tier per GPU class, cheapest offer of each class kept
        names = [t['gpu_name'] for t in tiers]
        assert names == ['RTX 3090', 'RTX 4090', 'RTX 5090']    # slowest -> fastest
        by_name = {t['gpu_name']: t for t in tiers}
        assert by_name['RTX 3090']['dph_total'] == 0.13         # cheapest 3090, not 0.18
        assert by_name['RTX 3090']['offer_id'] == 1
        # faster GPU -> fewer estimated minutes; every tier priced & timed
        assert by_name['RTX 5090']['est_minutes'] < by_name['RTX 3090']['est_minutes']
        assert all(t['est_cost'] is not None and t['est_minutes'] > 0 for t in tiers)


def test_gpu_tiers_requires_key(app, seeded_dataset, monkeypatch):
    monkeypatch.delenv('VAST_API_KEY', raising=False)
    from app.services import cloud_training as ct
    with app.app_context():
        with pytest.raises(RuntimeError, match='key'):
            ct.gpu_tiers('local', seeded_dataset)


def test_gpu_tiers_rejects_sdxl(ct, app, seeded_dataset):
    with app.app_context():
        with pytest.raises(ValueError, match='SDXL'):
            ct.gpu_tiers('local', seeded_dataset, train_type='sdxl')


def test_cloud_progress_selects_run_by_family(ct, app, seeded_dataset, tmp_path):
    with app.app_context():
        def seed(fam, step, sub):
            staging = tmp_path / sub
            staging.mkdir()
            (staging / 'training.log').write_text(
                f'{step}%|##        | {step}/100 loss: 0.02', encoding='utf-8')
            run = ct.CloudTrainingRun(
                dataset_id=seeded_dataset, status='training', job_name=f'j-{fam}',
                vast_label='lds-x', staging_dir=str(staging),
                train_params=json.dumps({'train_type': fam, 'steps': 100}))
            ct.db.session.add(run)
            ct.db.session.commit()
        seed('zimage', 30, 'run_z')
        seed('krea', 60, 'run_k')                        # newest
        assert ct.cloud_progress('local', seeded_dataset, train_type='zimage')['step'] == 30
        assert ct.cloud_progress('local', seeded_dataset, train_type='krea')['step'] == 60
        # no filter -> newest run (behavior unchanged)
        assert ct.cloud_progress('local', seeded_dataset)['step'] == 60
        # family with no matching run -> fall back to the newest run
        assert ct.cloud_progress('local', seeded_dataset, train_type='sdxl')['step'] == 60
