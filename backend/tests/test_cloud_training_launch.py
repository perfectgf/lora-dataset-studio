"""Launch validation, LEAK-SAFE provisioning (the property that matters),
stop request, and boot reconciliation. vast_client and the monitor thread are
always mocked -- no network, no thread started for real."""
import pytest


@pytest.fixture()
def ct(app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training
    # never start the real monitor thread in launch tests
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
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
        assert run.auth_token and len(run.auth_token) >= 24


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
