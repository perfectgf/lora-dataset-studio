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
                          'masked': False, 'allow_caption_quality': True})
    assert r.status_code == 200
    assert r.get_json()['ok'] is True
    assert seen['steps'] == 500 and seen['train_type'] == 'krea'
    assert seen['masked'] is False
    assert seen['allow_caption_quality'] is True


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

    def fake_tiers(user_id, dataset_id, train_type=None, variant=None, steps=None):
        seen.update(train_type=train_type, variant=variant, steps=steps)
        return {'tiers': [{'gpu_name': 'RTX 3090', 'dph_total': 0.13,
                           'est_minutes': 48, 'est_cost': 0.11, 'speed': 1.0}],
                'steps': 3000, 'family': 'krea', 'max_price_per_hour': 0.8}

    monkeypatch.setattr('app.services.cloud_training.gpu_tiers', fake_tiers)
    r = client.get(
        f'/api/dataset/{ds}/train/cloud/offers'
        '?train_type=krea&variant=base&steps=3000')
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['tiers'][0]['gpu_name'] == 'RTX 3090'
    assert seen == {'train_type': 'krea', 'variant': 'base', 'steps': 3000}


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


def test_cloud_retry_rechecks_mutated_dataset_without_legacy_bypass(
        client, app, monkeypatch):
    """An old successful/failed recipe cannot authorize today's bad export."""
    import json
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    monkeypatch.setattr(
        'app.services.cloud_training._reconcile_before_launch', lambda app: None)
    ds_id = _mkds(client)
    with app.app_context():
        from app.extensions import db
        from app.models import CloudTrainingRun, FaceDatasetImage
        # 15 kept = the krea family floor, so the retry reaches the UNCAPTIONED
        # guard rather than the readiness image-floor guard (which fires below 15).
        db.session.add_all([
            FaceDatasetImage(dataset_id=ds_id, status='keep',
                             filename=f'degraded-{i}.webp', caption=None)
            for i in range(15)
        ])
        run = CloudTrainingRun(
            dataset_id=ds_id, status='error', run_name='legacy-krea',
            train_params=json.dumps({
                'steps': 1000, 'variant': 'base', 'train_type': 'krea'}))
        db.session.add(run)
        db.session.commit()
        run_id = run.id

    response = client.post('/api/dataset/train/cloud/retry',
                           json={'run_id': run_id})
    assert response.status_code == 400
    assert response.get_json()['error'].startswith('UNCAPTIONED:')
    with app.app_context():
        from app.models import CloudTrainingRun
        assert CloudTrainingRun.query.count() == 1


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


def test_all_runs_unifies_local_and_cloud_history(app, client, monkeypatch):
    """The Runs hub history merges the provenance registry (local + cloud, with
    the per-launch settings snapshot) and enriches cloud rows from their
    CloudTrainingRun; legacy cloud runs without a registry row still appear."""
    import json
    ds = _mkds(client)
    with app.app_context():
        from app.extensions import db
        from app.models import CloudTrainingRun, TrainingRunRecord
        from app.services import cloud_training as ct
        # local launch record with a settings snapshot
        db.session.add(TrainingRunRecord(
            dataset_id=ds, family='krea', source='local', fingerprint='fp1', version=1,
            steps=2000, masked=True, settings=json.dumps({'rank': 32, 'resolution': [768, 1024]})))
        # cloud run + its registry row
        crun = CloudTrainingRun(dataset_id=ds, status='error', run_name='r', error='boom')
        db.session.add(crun)
        db.session.commit()
        db.session.add(TrainingRunRecord(
            dataset_id=ds, family='krea', source='cloud', fingerprint='fp1', version=1,
            steps=2000, masked=True, cloud_run_id=crun.id,
            settings=json.dumps({'rank': 48})))
        # legacy cloud run (predates the registry)
        db.session.add(CloudTrainingRun(dataset_id=ds, status='done', run_name='old'))
        db.session.commit()
        out = ct.all_runs(limit=10)
    recent = out['recent']
    assert len(recent) == 3
    sources = sorted(r['source'] for r in recent)
    assert sources == ['cloud', 'cloud', 'local']
    local_row = next(r for r in recent if r['source'] == 'local')
    assert local_row['settings'] == {'rank': 32, 'resolution': [768, 1024]}
    assert local_row['steps'] == 2000
    enriched = next(r for r in recent if r.get('error') == 'boom')
    assert enriched['status'] == 'error' and enriched['settings'] == {'rank': 48}
    # the registry's steps survive cloud enrichment (the pod row above never
    # stamped steps into train_params)
    assert enriched['steps'] == 2000
    legacy = next(r for r in recent if r.get('run_name') == 'old')
    assert legacy['settings'] is None and legacy['status'] == 'done'
    assert out['local_active'] is None


# --- Run preview thumbnails (Runs-hub cards) --------------------------------

def test_run_preview_serves_latest_cloud_sample(client, app, monkeypatch, tmp_path):
    """cloud-<id>/preview streams the NEWEST sample (highest step) of THAT
    run's staging, and all_runs stamps preview_url + saves on its row."""
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    ds = _mkds(client)
    staging = tmp_path / 'run_p'
    (staging / 'samples').mkdir(parents=True)
    (staging / 'samples' / '168__250_0.jpg').write_bytes(b'OLD')
    (staging / 'samples' / '169__500_1.jpg').write_bytes(b'NEWEST')
    (staging / 'samples' / 'notes.txt').write_bytes(b'not a sample')
    (staging / 'ck_000000500.safetensors').write_bytes(b'CK')
    (staging / 'final.safetensors').write_bytes(b'CK')
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        run = CloudTrainingRun(dataset_id=ds, status='done', job_name='j',
                               vast_label='lds-9', staging_dir=str(staging))
        db.session.add(run)
        db.session.commit()
        rid = run.id
    r = client.get(f'/api/dataset/train/runs/cloud-{rid}/preview')
    assert r.status_code == 200 and r.data == b'NEWEST'
    body = client.get('/api/dataset/train/cloud/runs?limit=5').get_json()
    row = next(x for x in body['recent'] if x.get('run_id') == rid)
    assert row['preview_url'] == f'/api/dataset/train/runs/cloud-{rid}/preview'
    assert row['saves'] == 2


def test_run_preview_local_record(client, app, monkeypatch, tmp_path):
    """rec-<id>/preview resolves a LOCAL run's stamped ai-toolkit run dir; its
    history row carries preview_url so the hub knows a thumbnail exists."""
    ds = _mkds(client)
    run_dir = tmp_path / 'lora_lola'
    (run_dir / 'samples').mkdir(parents=True)
    (run_dir / 'samples' / '170__100_0.jpg').write_bytes(b'LOCAL')
    from app.extensions import db
    from app.models import TrainingRunRecord
    with app.app_context():
        rec = TrainingRunRecord(dataset_id=ds, family='zimage', source='local',
                                fingerprint='fp', version=1, steps=1000,
                                masked=True)
        db.session.add(rec)
        db.session.commit()
        rec_id = rec.id
    monkeypatch.setattr('app.services.lora_training._run_dir',
                        lambda *a, **k: str(run_dir))
    r = client.get(f'/api/dataset/train/runs/rec-{rec_id}/preview')
    assert r.status_code == 200 and r.data == b'LOCAL'
    body = client.get('/api/dataset/train/cloud/runs?limit=5').get_json()
    row = next(x for x in body['recent'] if x.get('record_id') == rec_id)
    assert row['preview_url'] == f'/api/dataset/train/runs/rec-{rec_id}/preview'


def test_run_preview_unknown_or_sampleless_404(client, app, monkeypatch, tmp_path):
    """Unknown keys and runs that left no sample 404; their rows carry no
    preview_url (the hub falls back to the family tile)."""
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    assert client.get('/api/dataset/train/runs/cloud-999/preview').status_code == 404
    assert client.get('/api/dataset/train/runs/bogus/preview').status_code == 404
    assert client.get('/api/dataset/train/runs/rec-999/preview').status_code == 404
    ds = _mkds(client)
    staging = tmp_path / 'run_nosamples'
    staging.mkdir()
    from app.extensions import db
    from app.models import CloudTrainingRun
    with app.app_context():
        run = CloudTrainingRun(dataset_id=ds, status='error', job_name='j',
                               vast_label='lds-10', staging_dir=str(staging))
        db.session.add(run)
        db.session.commit()
        rid = run.id
    assert client.get(f'/api/dataset/train/runs/cloud-{rid}/preview').status_code == 404
    body = client.get('/api/dataset/train/cloud/runs?limit=5').get_json()
    row = next(x for x in body['recent'] if x.get('run_id') == rid)
    assert 'preview_url' not in row and row['saves'] == 0
