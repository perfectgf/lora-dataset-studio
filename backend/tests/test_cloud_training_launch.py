"""Launch validation, LEAK-SAFE provisioning (the property that matters),
stop request, and boot reconciliation. vast_client and the monitor thread are
always mocked -- no network, no thread started for real."""
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
