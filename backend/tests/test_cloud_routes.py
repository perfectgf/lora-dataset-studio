"""Cloud training routes: gating, forwarding, progress/status/stop/sample."""
import os
import pytest


def _mkds(client):
    return client.post('/api/dataset/create',
                       json={'name': 'Lola', 'trigger_word': 'lola'}).get_json()['id']


@pytest.fixture(autouse=True)
def _reset_capabilities_cache(monkeypatch):
    """Clear capabilities cache before each test so VAST_API_KEY env changes take effect."""
    from app import capabilities
    capabilities._cache = None
    capabilities._cache_ts = 0.0
    yield


def test_cloud_train_unconfigured_409_with_hint(client):
    ds = _mkds(client)
    r = client.post(f'/api/dataset/{ds}/train/cloud', json={})
    assert r.status_code == 409
    body = r.get_json()
    assert body['error'] == 'Cloud training is not configured'
    assert 'vast.ai' in body['hint']


def test_cloud_train_forwards_kwargs(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    seen = {}

    def fake_launch(user_id, dataset_id, **kw):
        seen.update(user_id=user_id, dataset_id=dataset_id, **kw)
        return {'run_id': 1, 'status': 'preparing', 'job_name': 'j', 'steps': 1200}

    monkeypatch.setattr('app.services.cloud_training.launch_cloud_training', fake_launch)
    r = client.post(f'/api/dataset/{ds}/train/cloud',
                    json={'steps': 500, 'variant': 'turbo', 'train_type': 'krea',
                          'masked': False})
    assert r.status_code == 200
    assert r.get_json()['ok'] is True
    assert seen['steps'] == 500 and seen['train_type'] == 'krea'
    assert seen['masked'] is False


def test_cloud_train_forwards_gpu_name(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    seen = {}

    def fake_launch(user_id, dataset_id, **kw):
        seen.update(kw)
        return {'run_id': 1, 'status': 'preparing', 'job_name': 'j', 'steps': 1200}

    monkeypatch.setattr('app.services.cloud_training.launch_cloud_training', fake_launch)
    client.post(f'/api/dataset/{ds}/train/cloud',
                json={'train_type': 'krea', 'gpu_name': 'RTX 5090'})
    assert seen['gpu_name'] == 'RTX 5090'


def test_cloud_offers_route_returns_tiers(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    seen = {}

    def fake_tiers(user_id, dataset_id, train_type=None, steps=None):
        seen.update(train_type=train_type, steps=steps)
        return {'tiers': [{'gpu_name': 'RTX 3090', 'dph_total': 0.13,
                           'est_minutes': 48, 'est_cost': 0.11, 'speed': 1.0}],
                'steps': 3000, 'family': 'krea', 'max_price_per_hour': 0.8}

    monkeypatch.setattr('app.services.cloud_training.gpu_tiers', fake_tiers)
    r = client.get(f'/api/dataset/{ds}/train/cloud/offers?train_type=krea&steps=3000')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['tiers'][0]['gpu_name'] == 'RTX 3090'
    assert seen['train_type'] == 'krea' and seen['steps'] == 3000


def test_cloud_offers_route_gated_when_unconfigured(client):
    ds = _mkds(client)
    r = client.get(f'/api/dataset/{ds}/train/cloud/offers')
    assert r.status_code == 409
    assert r.get_json()['error'] == 'Cloud training is not configured'


def test_cloud_train_value_error_maps_400(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    monkeypatch.setattr('app.services.cloud_training.launch_cloud_training',
                        lambda *a, **k: (_ for _ in ()).throw(ValueError('SDXL nope')))
    r = client.post(f'/api/dataset/{ds}/train/cloud', json={'train_type': 'sdxl'})
    assert r.status_code == 400


def test_cloud_status_unconfigured_is_open(client):
    r = client.get('/api/dataset/train/cloud/status')
    assert r.status_code == 200
    assert r.get_json()['configured'] is False


def test_cloud_runs_hub_endpoint(client, app, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        db.session.add_all([
            CloudTrainingRun(dataset_id=ds, status='training', job_name='j',
                             vast_label='lds-1', run_name='lola_krea'),
            CloudTrainingRun(dataset_id=ds, status='done', job_name='j',
                             vast_label='lds-2', run_name='lola_zimage'),
        ])
        db.session.commit()
    body = client.get('/api/dataset/train/cloud/runs?limit=5').get_json()
    assert body['configured'] is True
    assert [r['status'] for r in body['actives']] == ['training']
    assert body['recent'][0]['status'] == 'done'
    assert body['actives'][0]['dataset_name'] == 'Lola'


def test_cloud_runs_unconfigured_is_open_and_empty(client):
    body = client.get('/api/dataset/train/cloud/runs').get_json()
    assert body['configured'] is False
    assert body['actives'] == [] and body['recent'] == []


def test_cloud_progress_and_stop(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    monkeypatch.setattr('app.services.cloud_training.cloud_progress',
                        lambda uid, did, train_type=None: {
                            'active': True, 'phase': 'training',
                            'step': 5, 'total': 100, 'samples': []})
    r = client.get(f'/api/dataset/{ds}/train/cloud/progress')
    assert r.status_code == 200 and r.get_json()['phase'] == 'training'
    monkeypatch.setattr('app.services.cloud_training.request_stop', lambda run_id=None: True)
    assert client.post('/api/dataset/train/cloud/stop').get_json()['ok'] is True


def test_cloud_stop_forwards_run_id(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    seen = {}

    def fake_request_stop(run_id=None):
        seen['run_id'] = run_id
        return True

    monkeypatch.setattr('app.services.cloud_training.request_stop', fake_request_stop)
    r = client.post('/api/dataset/train/cloud/stop', json={'run_id': 42})
    assert r.status_code == 200 and r.get_json()['ok'] is True
    assert seen['run_id'] == 42
    # no body / no run_id -> still forwards (as None), compat with the old
    # "stop whatever is active" behavior
    seen.clear()
    r = client.post('/api/dataset/train/cloud/stop')
    assert r.status_code == 200 and r.get_json()['ok'] is True
    assert seen['run_id'] is None


def test_cloud_sample_served_from_staging(client, app, monkeypatch, tmp_path):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    staging = tmp_path / 'run_1'
    (staging / 'samples').mkdir(parents=True)
    (staging / 'samples' / '168__50_0.jpg').write_bytes(b'JPG')
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        run = CloudTrainingRun(dataset_id=ds, status='training', job_name='j',
                               vast_label='lds-1', staging_dir=str(staging))
        db.session.add(run)
        db.session.commit()
    r = client.get(f'/api/dataset/{ds}/train/cloud/sample/168__50_0.jpg')
    assert r.status_code == 200
    # traversal guard
    r = client.get(f'/api/dataset/{ds}/train/cloud/sample/..%2F..%2Fsecret.txt')
    assert r.status_code in (400, 404)


def test_cloud_checkpoint_download(client, app, monkeypatch, tmp_path):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    staging = tmp_path / 'run_2'
    staging.mkdir()
    ckpt = staging / 'j_000000100.safetensors'
    ckpt.write_bytes(b'CKPT')
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        run = CloudTrainingRun(dataset_id=ds, status='done', job_name='j',
                               vast_label='lds-2', staging_dir=str(staging),
                               checkpoint_local_path=str(ckpt))
        db.session.add(run)
        db.session.commit()
    r = client.get(f'/api/dataset/{ds}/train/cloud/checkpoint')
    assert r.status_code == 200
    assert r.data == b'CKPT'


# --- ?train_type= on progress/sample/checkpoint: resolve THAT family's run --

def _seed_family_runs(app, ds, tmp_path):
    """Two runs on the same dataset: zimage (step 30) then krea (step 60,
    NEWEST) — each with its own staging log, sample and checkpoint."""
    import json as _json
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        for fam, step, name in (('zimage', 30, 'zi'), ('krea', 60, 'kr')):
            staging = tmp_path / f'run_{name}'
            (staging / 'samples').mkdir(parents=True)
            (staging / 'training.log').write_text(
                f'{step}%|##        | {step}/100 loss: 0.02', encoding='utf-8')
            (staging / 'samples' / f'{name}__50_0.jpg').write_bytes(fam.encode())
            ckpt = staging / f'{name}.safetensors'
            ckpt.write_bytes(f'CKPT-{fam}'.encode())
            run = CloudTrainingRun(dataset_id=ds, status='done', job_name=f'j{name}',
                                   vast_label=f'lds-{name}', staging_dir=str(staging),
                                   checkpoint_local_path=str(ckpt),
                                   train_params=_json.dumps({'train_type': fam}))
            db.session.add(run)
            db.session.commit()


def test_cloud_progress_route_honors_train_type(client, app, monkeypatch, tmp_path):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    _seed_family_runs(app, ds, tmp_path)
    assert client.get(
        f'/api/dataset/{ds}/train/cloud/progress?train_type=zimage').get_json()['step'] == 30
    assert client.get(
        f'/api/dataset/{ds}/train/cloud/progress?train_type=krea').get_json()['step'] == 60
    # no filter -> newest run, behavior unchanged
    assert client.get(f'/api/dataset/{ds}/train/cloud/progress').get_json()['step'] == 60


def test_cloud_sample_route_honors_train_type(client, app, monkeypatch, tmp_path):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    _seed_family_runs(app, ds, tmp_path)
    r = client.get(f'/api/dataset/{ds}/train/cloud/sample/zi__50_0.jpg?train_type=zimage')
    assert r.status_code == 200 and r.data == b'zimage'
    r = client.get(f'/api/dataset/{ds}/train/cloud/sample/kr__50_0.jpg?train_type=krea')
    assert r.status_code == 200 and r.data == b'krea'
    # without the filter the NEWEST (krea) run's staging is used -> zi 404s
    assert client.get(f'/api/dataset/{ds}/train/cloud/sample/zi__50_0.jpg').status_code == 404


def test_cloud_checkpoint_route_honors_train_type(client, app, monkeypatch, tmp_path):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    _seed_family_runs(app, ds, tmp_path)
    r = client.get(f'/api/dataset/{ds}/train/cloud/checkpoint?train_type=zimage')
    assert r.status_code == 200 and r.data == b'CKPT-zimage'
    r = client.get(f'/api/dataset/{ds}/train/cloud/checkpoint?train_type=krea')
    assert r.status_code == 200 and r.data == b'CKPT-krea'
    # without the filter the newest run's checkpoint is served (unchanged)
    assert client.get(f'/api/dataset/{ds}/train/cloud/checkpoint').data == b'CKPT-krea'
