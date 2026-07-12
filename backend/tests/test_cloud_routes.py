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


def test_cloud_progress_and_stop(client, monkeypatch):
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    monkeypatch.setattr('app.services.cloud_training.cloud_progress',
                        lambda uid, did: {'active': True, 'phase': 'training',
                                          'step': 5, 'total': 100, 'samples': []})
    r = client.get(f'/api/dataset/{ds}/train/cloud/progress')
    assert r.status_code == 200 and r.get_json()['phase'] == 'training'
    monkeypatch.setattr('app.services.cloud_training.request_stop', lambda: True)
    assert client.post('/api/dataset/train/cloud/stop').get_json()['ok'] is True


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
