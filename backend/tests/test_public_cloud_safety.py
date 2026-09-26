"""Rental identity checks against the real SDK/main ORM, with no external I/O."""
import importlib
import json
from datetime import timedelta
from dataclasses import FrozenInstanceError
from types import GeneratorType, SimpleNamespace

import pytest

from test_public_sdk_integration import activate, host, no_external_runtime  # noqa: F401


@pytest.fixture
def cloud(host, monkeypatch):
    assert activate(host, {'cloud_training'}).records['cloud_training'].state == 'loaded'
    ct = importlib.import_module('lds_cloud_training.cloud_training')
    monkeypatch.setenv('VAST_API_KEY', 'fictional-account-a')
    if hasattr(ct, '_rental_credentials'):
        monkeypatch.setattr(ct, '_rental_credentials', {})
    monkeypatch.setattr(ct, '_start_monitor_for_app', lambda *_a: None)
    with host[0].app_context():
        yield ct, host[0]


def row(ct, **kwargs):
    run = ct.CloudTrainingRun(dataset_id=1, status=kwargs.pop('status', 'preparing'),
                              train_params=json.dumps(kwargs.pop('params', {})), **kwargs)
    ct.db.session.add(run)
    ct.db.session.commit()
    if not run.vast_label:
        run.vast_label = f'lds-{run.id}'
        ct.db.session.commit()
    return run


def provider(monkeypatch, vc, pods=(), fail_create=False, created=None):
    calls = []

    def request(_session, method, url, **kwargs):
        calls.append((method, url, kwargs['headers']['Authorization']))
        data = {'instances': list(pods)}
        if '/asks/' in url:
            if created is not None:
                created.append(kwargs['json']['label'])
            if fail_create:
                raise vc.requests.Timeout('fictional-account-a timeout')
            data = {'success': True, 'new_contract': 701}
        elif method == 'GET' and '/instances/' in url and not url.endswith('/instances/'):
            iid = url.rstrip('/').split('/')[-1]
            data = {'instances': next((p for p in pods if str(p['id']) == iid), None)}
        return SimpleNamespace(status_code=200, text='', json=lambda: data)

    monkeypatch.setattr(vc.requests.sessions.Session, 'request', request)
    return calls


def provision_fakes(monkeypatch, ct):
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **_k: [{'offer_id': 1, 'dph_total': 0.1}])
    monkeypatch.setattr(ct, '_pick_offer', lambda offers, *_a, **_k: offers[0])
    monkeypatch.setattr(ct, '_filter_offers', lambda offers: offers)
    monkeypatch.setattr(ct, '_disk_gb_for', lambda *_a: 50)
    monkeypatch.setattr(ct, '_hf_token_for_mode', lambda *_a: '')


def test_unrecorded_foreign_training_pod_is_preserved(cloud, monkeypatch):
    ct, app = cloud
    calls = provider(monkeypatch, ct.vast_client, [{'id': 701, 'label': 'lds-123'}])
    assert ct.reconcile_orphans(app) == 0
    assert not [c for c in calls if c[0] == 'DELETE']


def test_create_registration_failure_uses_the_original_account(cloud, monkeypatch):
    ct, _ = cloud
    run = row(ct)
    calls = provider(monkeypatch, ct.vast_client)
    provision_fakes(monkeypatch, ct)

    def cannot_record(*_a):
        monkeypatch.setenv('VAST_API_KEY', 'fictional-account-b')
        raise RuntimeError('fake database write failure')

    monkeypatch.setattr(ct, '_register_instance', cannot_record)
    with pytest.raises(RuntimeError, match='fake database'):
        ct._provision(run)
    deletions = [c for c in calls if c[0] == 'DELETE']
    assert len(deletions) == 1
    assert deletions[0][2] == 'Bearer fictional-account-a'


def interrupted(cloud, monkeypatch):
    ct, _ = cloud
    run = row(ct)
    labels = []
    calls = provider(monkeypatch, ct.vast_client, fail_create=True, created=labels)
    provision_fakes(monkeypatch, ct)
    with pytest.raises(ct.vast_client.VastCreateUncertain):
        ct._provision(run)
    assert len(labels) == 1
    assert ct._is_training_label(labels[0]) and labels[0] != f'lds-{run.id}'
    assert ct._rental_identity(run)['pending']
    assert not run.vast_instance_id
    assert 'fictional-account-a' not in run.train_params
    return run, labels[0], calls


def test_interrupted_create_is_durable_and_does_not_authorize_another_rental(cloud, monkeypatch):
    ct, app = cloud
    run, label, calls = interrupted(cloud, monkeypatch)
    with pytest.raises(ct.vast_client.VastCreateUncertain):
        ct._provision(run)
    assert len([c for c in calls if c[0] == 'PUT']) == 1
    assert ct._finish(run, 'error', error='CREATE response lost') is False
    calls = provider(monkeypatch, ct.vast_client, [])
    assert ct.reconcile_orphans(app) == 0
    assert ct._pending_rental(run)
    assert ct._maybe_auto_retry(run, 'HTTP 500') is None
    assert not [c for c in calls if c[0] in ('PUT', 'DELETE')]
    assert run.vast_label == label
    with pytest.raises(RuntimeError, match='unresolved rental'):
        ct._assert_launch_guardrails(2, 'zimage', allow_parallel_run=True)


@pytest.mark.parametrize('terminal', [False, True])
def test_interrupted_create_recovers_only_its_unique_pod(cloud, monkeypatch, terminal):
    ct, app = cloud
    run, label, _ = interrupted(cloud, monkeypatch)
    if terminal:
        ct._finish(run, 'error', error='CREATE response lost')
    calls = provider(monkeypatch, ct.vast_client, [
        {'id': 701, 'label': label}, {'id': 702, 'label': f'lds-{run.id}'},
        {'id': 703, 'label': label + '\n'}, {'id': 704, 'label': 'lds-１２３'}])
    ct._rental_credentials.clear()  # Simulated process restart; only the matching configured key remains.
    assert ct.reconcile_orphans(app) == int(terminal)
    ct.db.session.refresh(run)
    assert run.vast_instance_id == '701'
    assert ct._rental_identity(run)['released'] is terminal
    assert [c[1].split('/')[-2] for c in calls if c[0] == 'DELETE'] == (['701'] if terminal else [])


def test_boot_resumes_an_observed_interrupted_rental_without_a_second_create(cloud, monkeypatch):
    ct, app = cloud
    run, label, _ = interrupted(cloud, monkeypatch)
    calls = provider(monkeypatch, ct.vast_client, [{'id': 701, 'label': label}])
    started = []
    monkeypatch.setattr(ct, '_start_monitor_for_app', lambda _app, rid: started.append(rid))
    ct.boot_recover(app)
    assert started == [run.id]
    assert not [c for c in calls if c[0] in ('PUT', 'DELETE')]


@pytest.mark.parametrize('ambiguity', ['provider_label', 'provider_id', 'local_id', 'local_label', 'wrong_label'])
def test_ambiguous_claims_preserve_pods(cloud, monkeypatch, ambiguity):
    ct, app = cloud
    run = row(ct, status='error', vast_instance_id='701')
    pods = [{'id': 701, 'label': run.vast_label}]
    if ambiguity == 'provider_label':
        pods.append({'id': 702, 'label': run.vast_label})
    elif ambiguity == 'provider_id':
        pods.append({'id': 701, 'label': 'lds-555'})
    elif ambiguity == 'local_id':
        row(ct, status='error', vast_instance_id='701', vast_label='lds-556')
    elif ambiguity == 'local_label':
        row(ct, status='error', vast_instance_id='702', vast_label=run.vast_label)
    else:
        pods[0]['label'] = 'lds-557'
    calls = provider(monkeypatch, ct.vast_client, pods)
    assert ct.reconcile_orphans(app) == 0
    assert not [c for c in calls if c[0] == 'DELETE']
    assert ct._rental_identity(run) is None


def test_multiple_pods_for_one_interrupted_intent_are_not_adopted(cloud, monkeypatch):
    ct, app = cloud
    run, label, _ = interrupted(cloud, monkeypatch)
    calls = provider(monkeypatch, ct.vast_client, [{'id': 701, 'label': label}, {'id': 702, 'label': label}])
    assert ct.reconcile_orphans(app) == 0
    assert run.vast_instance_id is None
    assert ct._pending_rental(run)
    assert not [c for c in calls if c[0] == 'DELETE']


def known_rental(cloud, monkeypatch):
    ct, _ = cloud
    run = row(ct)
    labels = []
    provider(monkeypatch, ct.vast_client, created=labels)
    provision_fakes(monkeypatch, ct)
    ct._provision(run)
    assert not ct._pending_rental(run)
    assert run.vast_instance_id == '701'
    calls = provider(monkeypatch, ct.vast_client, [{'id': 701, 'label': labels[0]}])
    return run, calls


def test_stop_after_settings_rotation_keeps_the_captured_account(cloud, monkeypatch):
    ct, _ = cloud
    run, calls = known_rental(cloud, monkeypatch)
    monkeypatch.setenv('VAST_API_KEY', 'fictional-account-b')
    assert ct._force_stop(run, 'Stop fixture')['ok']
    assert calls and {c[2] for c in calls} == {'Bearer fictional-account-a'}
    assert ct._rental_identity(run)['released']
    assert run.status == 'stopped'


def test_restart_with_another_key_never_rebinds_or_deletes(cloud, monkeypatch):
    ct, app = cloud
    run, calls = known_rental(cloud, monkeypatch)
    original = ct._rental_identity(run)
    ct._rental_credentials.clear()
    monkeypatch.setenv('VAST_API_KEY', 'fictional-account-b')
    assert not ct._force_stop(run, 'Stop fixture')['ok']
    assert ct.reconcile_orphans(app) == 0
    assert calls == []
    assert ct._rental_identity(run) == original


def test_monitor_scope_covers_indirect_client_calls_and_restores_default(cloud, monkeypatch):
    ct, app = cloud
    run, calls = known_rental(cloud, monkeypatch)

    def lifecycle(_app, _rid):
        monkeypatch.setenv('VAST_API_KEY', 'fictional-account-b')
        ct.vast_client.get_instance('701')

    monkeypatch.setattr(ct, '_monitor_with_credentials', lifecycle)
    ct._monitor(app, run.id)
    assert calls[-1][2] == 'Bearer fictional-account-a'
    ct.vast_client.get_instance('701')
    assert calls[-1][2] == 'Bearer fictional-account-b'


@pytest.mark.parametrize('expired', [False, True])
def test_known_kept_pod_retains_the_public_recovery_window(cloud, monkeypatch, expired):
    ct, app = cloud
    run, calls = known_rental(cloud, monkeypatch)
    ct._set(run, status='error_pod_kept', finished_at=ct.naive_utcnow() - timedelta(hours=10 if expired else 1))
    assert ct.reconcile_orphans(app) == int(expired)
    assert len([c for c in calls if c[0] == 'DELETE']) == int(expired)


def test_legacy_recorded_id_and_exact_label_can_still_be_supervised(cloud, monkeypatch):
    ct, _ = cloud
    run = row(ct, status='training', vast_instance_id='701', created_at=ct.naive_utcnow() - timedelta(hours=20))
    calls = provider(monkeypatch, ct.vast_client, [{'id': 701, 'label': run.vast_label}, {'id': 702, 'label': 'lds-999'}])
    result = ct.supervise_active_runs()
    assert result == [{'run_id': run.id, 'reason': 'runtime_cap', 'ok': True}]
    assert [c[1].split('/')[-2] for c in calls if c[0] == 'DELETE'] == ['701']
    assert ct._rental_identity(run)['unique'] is False


def test_captured_credentials_are_immutable_and_transport_errors_are_scrubbed(cloud, monkeypatch):
    ct, _ = cloud
    vc = ct.vast_client
    credential = vc.capture_credentials()
    assert 'fictional-account-a' not in repr(credential)
    with pytest.raises(FrozenInstanceError):
        credential.api_key = 'replacement'
    monkeypatch.setenv('VAST_API_KEY', 'fictional-account-b')

    def boom(session, *_a, **kwargs):
        assert session.trust_env is False and kwargs['allow_redirects'] is False
        raise vc.requests.Timeout('fictional-account-a fictional-account-b')

    monkeypatch.setattr(vc.requests.Session, 'request', boom)
    with pytest.raises(vc.VastError) as failure:
        vc.get_instance('701', credential=credential)
    assert 'fictional-account' not in str(failure.value)
    assert failure.value.__cause__ is None and failure.value.__suppress_context__


@pytest.mark.parametrize('payload', [{}, {'instances': None}, {'instances': [{'label': 'lds-1'}]}, {'instances': {}}])
def test_malformed_listings_cannot_prove_absence(cloud, monkeypatch, payload):
    ct, _ = cloud
    monkeypatch.setattr(ct.vast_client.requests.Session, 'request', lambda *_a, **_k:
                        SimpleNamespace(status_code=200, json=lambda: payload))
    with pytest.raises(ct.vast_client.VastError):
        ct.vast_client.list_instances()


def test_sdk_history_and_script_export_keep_existing_main_identities(cloud):
    from app.services import cloud_training as main, fp8_export
    from lds_sdk import cloud_history
    from lds_sdk.cloud_host.services import fp8_export as facade
    from lds_sdk.cloud_host.config import secret
    from app.config import secret as main_secret
    assert secret is main_secret
    assert cloud_history.latest_run_for is main.latest_run_for
    assert cloud_history.run_checkpoint_files is main.run_checkpoint_files
    assert facade.__file__ == fp8_export.__file__
    assert facade.fp8_name_for is fp8_export.fp8_name_for


@pytest.mark.parametrize('module_name,helper_name', [
    ('test_config', '_fresh'),
    ('test_comfy_folder_overrides', '_fresh'),
    ('test_krea_edit', '_fresh_config'),
    ('test_krea_default_base_election', '_fresh_config'),
    ('test_model_scanners_agree', 'tree'),
])
def test_config_fixture_resets_preserve_existing_sdk_imports(cloud, monkeypatch, tmp_path,
                                                            module_name, helper_name):
    """Reproduce the worker order: import SDK, reset another test's config, reuse SDK."""
    from app import config
    from lds_sdk.cloud_host import config as facade

    before = {name: getattr(config, name) for name in facade.__all__}
    helper = getattr(importlib.import_module(module_name), helper_name)
    helper = getattr(helper, '__wrapped__', helper)  # The scanner uses a yield fixture.
    result = helper(monkeypatch, tmp_path)
    try:
        if isinstance(result, GeneratorType):
            next(result)
        for name, original in before.items():
            assert getattr(config, name) is original, name
            assert getattr(facade, name) is original, name
    finally:
        if isinstance(result, GeneratorType):
            result.close()


def test_late_create_observation_restarts_monitoring(cloud, monkeypatch):
    ct, app = cloud
    run, label, _ = interrupted(cloud, monkeypatch)
    provider(monkeypatch, ct.vast_client, [])
    started = []
    monkeypatch.setattr(ct, '_start_monitor_for_app', lambda _a, rid: started.append(rid))
    ct.boot_recover(app)
    assert started == []
    provider(monkeypatch, ct.vast_client, [{'id': 701, 'label': label}])
    ct.reconcile_orphans(app)
    assert started == [run.id]


def test_lost_delete_response_is_recoverable_under_the_original_identity(cloud, monkeypatch):
    ct, app = cloud
    run, _ = known_rental(cloud, monkeypatch)
    original_destroy = ct.vast_client.destroy_instance
    monkeypatch.setattr(ct.vast_client, 'destroy_instance', lambda *_a, **_k: False)
    assert ct._finish(run, 'error', error='delete reply lost') is False
    assert ct._rental_identity(run)['delete_pending']
    monkeypatch.setattr(ct.vast_client, 'destroy_instance', original_destroy)
    calls = provider(monkeypatch, ct.vast_client, [])
    assert ct.reconcile_orphans(app) == 1
    ct.db.session.refresh(run)
    assert ct._rental_identity(run)['released']
    assert [c[0] for c in calls] == ['GET', 'GET', 'DELETE']


def test_dense_fetch_indirect_calls_use_the_rental_account(cloud, monkeypatch):
    ct, app = cloud
    run, calls = known_rental(cloud, monkeypatch)
    monkeypatch.setenv('VAST_API_KEY', 'fictional-account-b')
    monkeypatch.setattr(ct, '_make_remote', lambda _r: object())
    monkeypatch.setattr(ct, '_deliver_dense_locally', lambda *_a, **_k: ct.vast_client.get_instance('701'))
    ct._dense_fetch_worker(app, run.id)
    assert calls and {c[2] for c in calls} == {'Bearer fictional-account-a'}


def test_pending_terminal_rental_keeps_accruing_cost(cloud, monkeypatch):
    ct, _ = cloud
    run, _, _ = interrupted(cloud, monkeypatch)
    now = ct.naive_utcnow()
    ct._set(run, status='error', created_at=now - timedelta(hours=2), finished_at=now - timedelta(hours=1))
    assert ct.month_spend_usd() == pytest.approx(0.2, abs=0.001)


def test_off_recovery_preserves_cleanup_but_cannot_rent(cloud, monkeypatch):
    ct, _ = cloud
    run, calls = known_rental(cloud, monkeypatch)
    ct.cfg.save_config({'plugins': {'enabled': {'cloud_training': False}}})
    assert not ct.is_available('cloud_training')
    assert ct._force_stop(run, 'OFF cleanup')['ok']
    ct._set(run, status='error')
    launched = []
    monkeypatch.setattr(ct, 'launch_cloud_training', lambda *_a, **_k: launched.append(True))
    assert ct._maybe_auto_retry(run, 'pod disappeared') is None
    with pytest.raises(RuntimeError, match='disabled'):
        ct._provision(row(ct))
    with pytest.raises(RuntimeError, match='disabled'):
        ct._assert_launch_guardrails(2, 'zimage')
    assert launched == []
    assert not [c for c in calls if c[0] == 'PUT']


def test_disable_between_offer_search_and_create_blocks_the_rental(cloud, monkeypatch):
    ct, _ = cloud
    run = row(ct)
    calls = provider(monkeypatch, ct.vast_client)
    provision_fakes(monkeypatch, ct)

    def search(**_k):
        ct.cfg.save_config({'plugins': {'enabled': {'cloud_training': False}}})
        return [{'offer_id': 1, 'dph_total': 0.1}]

    monkeypatch.setattr(ct.vast_client, 'search_offers', search)
    with pytest.raises(RuntimeError, match='disabled'):
        ct._provision(run)
    assert not ct._pending_rental(run)
    assert calls == []


def test_video_http_cannot_erase_pending_rental_provenance(host, monkeypatch):
    loaded = activate(host, {'video', 'cloud_training'})
    assert loaded.records['video'].state == loaded.records['cloud_training'].state == 'loaded'
    ct = importlib.import_module('lds_cloud_training.cloud_training')
    app = host[0]
    monkeypatch.setenv('VAST_API_KEY', 'fictional-account-a')
    monkeypatch.setattr(ct, '_rental_credentials', {})
    from app.models import VideoDataset
    with app.app_context():
        ds = VideoDataset(user_id='local', name='Rental fixture', target_profile='wan22',
                          output_dir=str(host[2] / 'video'))
        ct.db.session.add(ds)
        ct.db.session.commit()
        run, label, _ = interrupted((ct, app), monkeypatch)
        ct._set(run, dataset_id=ds.id, dataset_table=ct.crd.VIDEO, status='error')
        assert ct.crd.owns(run, ds.id, ct.crd.VIDEO), (run.dataset_id, run.dataset_table, run.train_params)
        route = f'/api/video-dataset/{ds.id}/train/cloud/run/{run.id}'
        run_id = run.id
    response = app.test_client().delete(route)
    assert response.status_code == 409, response.get_data(as_text=True)
    assert 'cleanup is not confirmed' in response.json['error']
    with app.app_context():
        run = ct.db.session.get(ct.CloudTrainingRun, run_id)
        assert run is not None and ct._pending_rental(run)
        provider(monkeypatch, ct.vast_client, [{'id': 701, 'label': label}])
        assert ct.reconcile_orphans(app) == 1
        monkeypatch.setattr(ct, 'run_checkpoint_files', lambda _r: {})
        monkeypatch.setattr(ct, 'checkpoint_store_dir', lambda _r: None)
    assert app.test_client().delete(route).status_code == 200
    with app.app_context():
        assert ct.db.session.get(ct.CloudTrainingRun, run_id) is None


@pytest.mark.parametrize('method', ['launch_cloud_video_training', 'continue_cloud_video_run',
                                  'retry_cloud_video_run', 'delete_cloud_video_run', 'video_gpu_tiers'])
def test_video_cloud_dispatch_requires_its_active_provider(host, monkeypatch, method):
    assert activate(host, {'video'}).records['video'].state == 'loaded'
    from lds_sdk.video_host import cloud_video_training as facade
    from app.services import cloud_video_training as historical
    monkeypatch.setattr(historical, method, lambda *_a, **_k: pytest.fail('Historical host must not rent or delete'))
    with host[0].app_context(), pytest.raises(RuntimeError, match='disabled'):
        getattr(facade, method)()


def test_video_local_checkpoints_work_with_cloud_off(host, monkeypatch):
    assert activate(host, {'video'}).records['video'].state == 'loaded'
    vck = importlib.import_module('lds_video.video_checkpoints')
    from lds_sdk.video_host import cloud_video_training as facade
    from lds_sdk import cloud_training as history
    monkeypatch.setattr(vck, '_local_saves', lambda _ds: {'fixture_000000100.safetensors': '/fixture/local'})
    monkeypatch.setattr(vck.vtl, 'local_run_name', lambda _ds: 'fixture')
    monkeypatch.setattr(vck.vtl, 'save_root', lambda _ds: '/fixture')
    monkeypatch.setattr(vck.vtl, 'video_training_progress', lambda *_a: {'active': False})
    ds = SimpleNamespace(id=1, user_id='local', best_settings=None)
    with host[0].app_context():
        assert vck.local_group(ds, deployed={})['steps'][0]['step'] == 100
        assert facade.group_saves_by_step({'fixture_000000100.safetensors': '/fixture/local'})
        assert history._run_param(SimpleNamespace(train_params='{"steps": 100}'), 'steps') == 100
        with pytest.raises(RuntimeError, match='disabled'):
            history.month_spend_usd()


def test_sdk_cloud_cost_reads_the_guarded_product(cloud, monkeypatch):
    ct, _ = cloud
    from lds_sdk import cloud_training as facade
    from app.services import cloud_training as historical
    monkeypatch.setattr(historical, 'month_spend_usd', lambda: pytest.fail('Stale main billing path'))
    run, _, _ = interrupted(cloud, monkeypatch)
    now = ct.naive_utcnow()
    ct._set(run, status='error', created_at=now - timedelta(hours=2), finished_at=now - timedelta(hours=1))
    assert facade.month_spend_usd() == pytest.approx(0.2, abs=0.001)


@pytest.mark.parametrize('owner,table', [('another-user', 'video_dataset'), ('local', 'face_dataset')])
def test_video_http_refuses_foreign_user_or_dataset_type(host, owner, table):
    activate(host, {'video', 'cloud_training'})
    ct = importlib.import_module('lds_cloud_training.cloud_training')
    from app.models import VideoDataset, FaceDataset
    with host[0].app_context():
        video = VideoDataset(user_id=owner, name='Ownership fixture', target_profile='wan22',
                             output_dir=str(host[2] / 'video'))
        face = FaceDataset(user_id='local', name='Colliding fixture', trigger_word='fixture')
        ct.db.session.add_all([video, face])
        ct.db.session.commit()
        run = row(ct, status='done', dataset_table=table)
        route = f'/api/video-dataset/{video.id}/train/cloud/run/{run.id}'
        run_id = run.id
    assert host[0].test_client().delete(route).status_code == 404
    with host[0].app_context():
        assert ct.db.session.get(ct.CloudTrainingRun, run_id) is not None


def test_video_http_off_can_delete_released_history_and_serve_local_catalog(host):
    activate(host, {'video'})
    from app.models import VideoDataset, CloudTrainingRun
    from app.extensions import db
    with host[0].app_context():
        video = VideoDataset(user_id='local', name='OFF fixture', target_profile='wan22',
                             output_dir=str(host[2] / 'video'))
        db.session.add(video)
        db.session.commit()
        run = CloudTrainingRun(dataset_id=video.id, dataset_table='video_dataset', status='done')
        db.session.add(run)
        db.session.commit()
        route = f'/api/video-dataset/{video.id}/train/cloud/run/{run.id}'
    client = host[0].test_client()
    response = client.delete(route)
    assert response.status_code == 200, response.get_data(as_text=True)
    with host[0].app_context():
        assert db.session.get(CloudTrainingRun, run.id) is None
    assert client.get('/api/video/targets').status_code == 200


def test_durable_release_remains_idempotent_without_a_key(cloud, monkeypatch):
    ct, app = cloud
    run, calls = known_rental(cloud, monkeypatch)
    assert ct._finish(run, 'done')
    calls.clear()
    ct._rental_credentials.clear()
    monkeypatch.delenv('VAST_API_KEY')
    assert ct._destroy_run_instance(run)
    assert ct.reconcile_orphans(app) == 0
    assert calls == []


@pytest.mark.parametrize('payload,headers', [
    ({'instances': [], 'has_more': True}, {}),
    ({'instances': [], 'next': 'page-two'}, {}),
    ({'instances': [], 'total': 1}, {}),
    ({'instances': []}, {'Link': '<page-two>; rel="next"'}),
    ({'success': True, 'instances': [], 'instances_found': 0, 'total_instances': 1,
      'label_counts': {}, 'next_token': 'page-two'}, {}),
])
def test_paginated_fleet_is_not_an_ownership_observation(cloud, monkeypatch, payload, headers):
    ct, _ = cloud
    monkeypatch.setattr(ct.vast_client.requests.Session, 'request', lambda *_a, **_k:
                        SimpleNamespace(status_code=200, headers=headers, json=lambda: payload))
    with pytest.raises(ct.vast_client.VastError, match='incomplete'):
        ct.vast_client.list_instances()


@pytest.mark.parametrize('payload', [{'success': False}, {'error': 'refused'}, [], None])
def test_http_200_rejection_does_not_acknowledge_delete(cloud, monkeypatch, payload):
    ct, _ = cloud
    monkeypatch.setattr(ct.vast_client.requests.Session, 'request', lambda *_a, **_k:
                        SimpleNamespace(status_code=200, json=lambda: payload))
    assert ct.vast_client.destroy_instance('701') is False
