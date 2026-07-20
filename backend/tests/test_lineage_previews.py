"""Lab inline generation (slice 3): render one same-prompt/same-seed preview per
selected lineage checkpoint by REUSING the Test-Studio engine pinned to those
checkpoints at strength 1.0. The engine is mocked here (ComfyUI is never
contacted); the tests assert the pinning, the per-checkpoint preview storage, the
GPU serialization contract (training holding the GPU → 409), and the honest
'needs setup' path when no checkpoint is deployed for the family."""


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create',
                       json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _deployed(monkeypatch, entries):
    """Pretend these LoRAs are deployed (testable) for every dataset+family, so
    the resolver joins a lineage step to a loadable checkpoint without touching
    the real loras folder."""
    monkeypatch.setattr('app.services.lora_test_studio.list_test_checkpoints',
                        lambda ds, family=None: [{'filename': f, 'label': f} for f in entries])


def _mock_engine(monkeypatch, calls):
    def fake_create_run(user_id, dataset_id, checkpoints, strengths, **kw):
        calls['checkpoints'] = list(checkpoints)
        calls['strengths'] = list(strengths)
        calls['count'] = kw.get('count')
        calls['seed'] = kw.get('seed')
        calls['prompt'] = kw.get('prompt')
        calls['family'] = kw.get('family')
        return {'ids': list(range(101, 101 + len(checkpoints))),
                'seed': 777, 'count': 1, 'created': len(checkpoints)}
    monkeypatch.setattr('app.services.lora_test_studio.create_run', fake_create_run)


# --- step resolver -----------------------------------------------------------

def test_step_of_testable_parses_both_conventions(app):
    from app.services.cloud_training import _step_of_testable
    assert _step_of_testable('z image\\Nova-500.safetensors') == 500
    assert _step_of_testable('lora_EVA6938_000001000.safetensors') == 1000
    assert _step_of_testable('sdxl\\nova-1500.safetensors') == 1500
    assert _step_of_testable('lora_nova.safetensors') is None   # final, no step


# --- pinning + storage -------------------------------------------------------

def test_generate_pins_engine_to_checkpoints_at_strength_1(client, monkeypatch, app):
    ds = _create(client)
    _deployed(monkeypatch, ['z image\\Nova-500.safetensors', 'z image\\Nova-1000.safetensors'])
    calls = {}
    _mock_engine(monkeypatch, calls)
    from app.services import cloud_training as ct
    with app.app_context():
        out = ct.generate_checkpoint_previews(
            'local', ds,
            [{'record_id': 7, 'step': 500}, {'record_id': 7, 'step': 1000}],
            prompt='a portrait', seed=42, family='zimage')
    # Pinned to EXACTLY the two deployed checkpoints, each at strength 1.0, one image.
    assert calls['checkpoints'] == ['z image\\Nova-500.safetensors',
                                    'z image\\Nova-1000.safetensors']
    assert calls['strengths'] == [1.0]
    assert calls['count'] == 1
    assert calls['prompt'] == 'a portrait' and calls['seed'] == 42
    assert out['queued'] == 2 and out['needs_setup'] is False and out['skipped'] == []
    # One preview row per checkpoint, pointing at the reused Studio image rows.
    from app.models import CheckpointPreview
    with app.app_context():
        rows = {r.step: r for r in CheckpointPreview.query.filter_by(record_id=7).all()}
        assert set(rows) == {500, 1000}
        assert rows[500].lora_test_image_id == 101 and rows[1000].lora_test_image_id == 102
        assert rows[500].seed == 777 and rows[1000].seed == 777   # the engine's actual seed


def test_generate_skips_undeployed_and_flags_needs_setup(client, monkeypatch, app):
    ds = _create(client)
    _deployed(monkeypatch, ['z image\\Nova-500.safetensors'])   # only step 500 deployed
    calls = {}
    _mock_engine(monkeypatch, calls)
    from app.services import cloud_training as ct
    with app.app_context():
        out = ct.generate_checkpoint_previews(
            'local', ds, [{'record_id': 7, 'step': 500}, {'record_id': 7, 'step': 999}],
            family='zimage')
    assert out['queued'] == 1
    assert calls['checkpoints'] == ['z image\\Nova-500.safetensors']   # 999 not pinned
    assert out['skipped'] == [{'record_id': 7, 'step': 999, 'reason': 'not_deployed'}]

    with app.app_context():
        none_out = ct.generate_checkpoint_previews(
            'local', ds, [{'record_id': 7, 'step': 999}], family='zimage')
    assert none_out['needs_setup'] is True and none_out['queued'] == 0


# --- reading previews back ---------------------------------------------------

def test_checkpoint_previews_for_resolves_status_and_url(client, monkeypatch, app):
    ds = _create(client)
    from app.extensions import db
    from app.models import CheckpointPreview, LoraTestImage
    from app.services import cloud_training as ct
    with app.app_context():
        done = LoraTestImage(dataset_id=ds, checkpoint='z image\\Nova-500.safetensors',
                             strength=1.0, status='done', filename='out.png')
        pend = LoraTestImage(dataset_id=ds, checkpoint='z image\\Nova-1000.safetensors',
                             strength=1.0, status='pending')
        db.session.add_all([done, pend]); db.session.commit()
        db.session.add_all([
            CheckpointPreview(record_id=7, step=500, dataset_id=ds,
                              lora_test_image_id=done.id, seed=777),
            CheckpointPreview(record_id=7, step=1000, dataset_id=ds,
                              lora_test_image_id=pend.id, seed=777)])
        db.session.commit()
        pv = ct.checkpoint_previews_for(7)
    assert pv[500]['status'] == 'done'
    assert pv[500]['url'] == f'/api/dataset/{ds}/img/out.png'
    assert pv[1000]['status'] == 'pending' and pv[1000]['url'] is None


def test_lineage_node_pills_gain_testable_and_preview(client, monkeypatch, app):
    ds = _create(client)
    _deployed(monkeypatch, ['z image\\Nova-500.safetensors'])   # step 500 testable, 1000 not
    from app.extensions import db
    from app.models import TrainingRunRecord, CheckpointPreview, LoraTestImage
    from app.services import cloud_training as ct
    with app.app_context():
        rec = TrainingRunRecord(dataset_id=ds, family='zimage', source='local',
                                version=1, fingerprint='fp', steps=1000)
        db.session.add(rec); db.session.commit(); rid = rec.id
        img = LoraTestImage(dataset_id=ds, checkpoint='z image\\Nova-500.safetensors',
                            strength=1.0, status='done', filename='p.png')
        db.session.add(img); db.session.commit()
        db.session.add(CheckpointPreview(record_id=rid, step=500, dataset_id=ds,
                                         lora_test_image_id=img.id, seed=777))
        db.session.commit()
        # Feed the node two pills without needing files on disk.
        monkeypatch.setattr(ct, '_node_checkpoints',
                            lambda rec, crun: [{'step': 500, 'filename': 'a', 'present': True},
                                               {'step': 1000, 'filename': 'b', 'present': True}])
        node = ct._lineage_node(TrainingRunRecord.query.get(rid), None, rid, None)
    by_step = {c['step']: c for c in node['checkpoints']}
    assert by_step[500]['testable'] is True and by_step[1000]['testable'] is False
    assert by_step[500]['preview_url'] == f'/api/dataset/{ds}/img/p.png'
    assert by_step[500]['preview_status'] == 'done'
    assert 'preview_url' not in by_step[1000]   # never previewed → no claim


# --- route: GPU serialization + needs-setup ----------------------------------

def test_route_409_when_training_holds_gpu(client, monkeypatch):
    ds = _create(client)
    _deployed(monkeypatch, ['z image\\Nova-500.safetensors'])
    calls = {}
    _mock_engine(monkeypatch, calls)
    from app.job_queue import queue_manager
    with client.application.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
    resp = client.post(f'/api/dataset/{ds}/lineage/previews',
                       json={'checkpoints': [{'record_id': 7, 'step': 500}],
                             'family': 'zimage'})
    assert resp.status_code == 409 and 'GPU busy' in resp.get_json()['error']
    assert 'checkpoints' not in calls   # the engine was never reached


def test_route_409_needs_setup_when_nothing_deployed(client, monkeypatch):
    ds = _create(client)
    _deployed(monkeypatch, [])   # nothing deployed for the family
    _mock_engine(monkeypatch, {})
    resp = client.post(f'/api/dataset/{ds}/lineage/previews',
                       json={'checkpoints': [{'record_id': 7, 'step': 500}],
                             'family': 'zimage'})
    assert resp.status_code == 409
    assert resp.get_json()['needs_setup'] is True


def test_route_forwards_and_returns_queued(client, monkeypatch):
    ds = _create(client)
    _deployed(monkeypatch, ['z image\\Nova-500.safetensors'])
    _mock_engine(monkeypatch, {})
    resp = client.post(f'/api/dataset/{ds}/lineage/previews',
                       json={'checkpoints': [{'record_id': 7, 'step': 500}],
                             'prompt': 'p', 'seed': 5, 'family': 'zimage'})
    assert resp.status_code == 200 and resp.get_json()['queued'] == 1


def test_route_400_without_checkpoints(client):
    ds = _create(client)
    assert client.post(f'/api/dataset/{ds}/lineage/previews',
                       json={'checkpoints': []}).status_code == 400


def test_route_404_unknown_dataset(client):
    assert client.post('/api/dataset/999999/lineage/previews',
                       json={'checkpoints': [{'record_id': 1, 'step': 1}]}).status_code == 404
