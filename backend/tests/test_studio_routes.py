"""Studio blueprint + per-dataset lora-test routes: ComfyUI gating (only on
routes that actually enqueue a job) + service wiring.

Every test that touches gating patches `app.capabilities.probe`; tests that
exercise `create_run`/`create_comparison_run` patch the service layer instead
of the gate, since those are covered end-to-end by test_studio_service.py.
"""
import json


def _create(client, name='Nova', trigger='nova'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger}).get_json()['id']


def _comfy(monkeypatch, reachable):
    monkeypatch.setattr('app.capabilities.probe', lambda *a, **k: {'comfyui': {'reachable': reachable}})


# --- /api/studio/run gating ---------------------------------------------------

def test_studio_run_unreachable_comfyui_returns_409_with_hint(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.post('/api/studio/run', json={})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body['error'] == 'ComfyUI is not reachable'
    assert body['hint'] == 'Check the URL in Settings'


def test_studio_run_resume_unreachable_comfyui_returns_409(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.post('/api/studio/run/some-run-id/resume')
    assert resp.status_code == 409


def test_studio_run_reachable_forwards_to_service(client, monkeypatch):
    _comfy(monkeypatch, True)
    monkeypatch.setattr('app.services.lora_test_studio.create_comparison_run',
                        lambda *a, **k: {'created': 2, 'seed': 42, 'count': 1, 'run_id': 'r1'})
    resp = client.post('/api/studio/run', json={'selections': [{'dataset_id': 1, 'checkpoint': 'x'}]})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {'ok': True, 'created': 2, 'seed': 42, 'count': 1, 'run_id': 'r1'}


# --- /api/studio/run/<id>/status + /cancel work regardless of ComfyUI --------

def test_studio_run_status_unknown_returns_404_even_when_comfyui_down(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.get('/api/studio/run/does-not-exist/status')
    assert resp.status_code == 404


def test_studio_run_cancel_ungated_when_comfyui_down(client, monkeypatch):
    _comfy(monkeypatch, False)
    resp = client.post('/api/studio/run/does-not-exist/cancel')
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'cancelled': 0}


# --- /api/studio listings (ungated, no ComfyUI dependency) -------------------

def test_studio_checkpoints_and_recent_prompts_smoke(client):
    assert client.get('/api/studio/checkpoints').get_json() == {'loras': []}
    assert client.get('/api/studio/recent-prompts').get_json() == {'ok': True, 'prompts': []}


def test_studio_base_models_krea_type_returns_empty_list(client):
    resp = client.get('/api/studio/base-models?type=krea')
    assert resp.status_code == 200
    assert resp.get_json() == {'models': []}


# --- per-dataset lora-test/status --------------------------------------------

def test_lora_test_status_fresh_dataset_is_well_formed(client):
    ds_id = _create(client)
    resp = client.get(f'/api/dataset/{ds_id}/lora-test/status')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['checkpoints'] == [] and body['cells'] == [] and body['pending'] == 0


def test_lora_test_status_unknown_dataset_404(client):
    resp = client.get('/api/dataset/999999/lora-test/status')
    assert resp.status_code == 404


def test_lora_test_status_reachable_regardless_of_comfyui(client, monkeypatch):
    """History/status routes must stay viewable even with ComfyUI offline."""
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.get(f'/api/dataset/{ds_id}/lora-test/status')
    assert resp.status_code == 200


# --- per-dataset lora-test/run gating -----------------------------------------

def test_dataset_lora_test_run_unreachable_comfyui_returns_409(client, monkeypatch):
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/run', json={})
    assert resp.status_code == 409
    assert resp.get_json()['hint'] == 'Check the URL in Settings'


def test_dataset_lora_test_resume_unreachable_comfyui_returns_409(client, monkeypatch):
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/resume')
    assert resp.status_code == 409


# --- rate: valid ratings 1/-1/0 ok, invalid -> 400 ---------------------------

def test_rate_valid_ratings_accepted(client):
    ds_id = _create(client)
    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import LoraTestImage
        from app.config import LOCAL_USER
        img = LoraTestImage(dataset_id=ds_id, checkpoint='z image\\lora_nova_000001000.safetensors',
                            strength=1.0, status='done')
        svc.db.session.add(img)
        svc.db.session.commit()
        image_id = img.id
    for rating in (1, -1, 0):
        resp = client.post(f'/api/dataset/lora-test/image/{image_id}/rate', json={'rating': rating})
        assert resp.status_code == 200, (rating, resp.get_json())
        assert resp.get_json() == {'ok': True}


def test_rate_invalid_rating_returns_400(client):
    ds_id = _create(client)
    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import LoraTestImage
        img = LoraTestImage(dataset_id=ds_id, checkpoint='z image\\lora_nova_000001000.safetensors',
                            strength=1.0, status='done')
        svc.db.session.add(img)
        svc.db.session.commit()
        image_id = img.id
    resp = client.post(f'/api/dataset/lora-test/image/{image_id}/rate', json={'rating': 5})
    assert resp.status_code == 400
    assert resp.get_json() == {'ok': False, 'error': 'invalid'}


# --- best set/clear roundtrip persists into FaceDataset.best_settings -------

def test_best_set_then_clear_roundtrips_through_facedataset_best_settings(client, monkeypatch, tmp_path):
    from app import config
    base = tmp_path / 'Comfy'
    lora_dir = base / 'models' / 'loras' / 'z image'
    lora_dir.mkdir(parents=True)
    ck = 'z image\\lora_nova_000002000.safetensors'
    (lora_dir / 'lora_nova_000002000.safetensors').touch()
    config.save_config({'comfyui': {'base_dir': str(base)}})
    import app.utils.comfyui as comfyui_utils
    monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})

    ds_id = _create(client)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/best',
                       json={'checkpoint': ck, 'strength': 0.9})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert body['best_settings']['lora_filename'] == ck

    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDataset
        ds = svc.db.session.get(FaceDataset, ds_id)
        stored = json.loads(ds.best_settings)
        assert stored['zimage']['lora_filename'] == ck
        assert stored['zimage']['strength'] == 0.9

    resp = client.delete(f'/api/dataset/{ds_id}/lora-test/best')
    assert resp.status_code == 200
    with client.application.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDataset
        ds = svc.db.session.get(FaceDataset, ds_id)
        assert ds.best_settings is None


# --- score-faces: reaches the service, no ComfyUI gate ----------------------

def test_score_faces_reaches_service_without_comfyui_gate(client, monkeypatch):
    _comfy(monkeypatch, False)  # ComfyUI down -> must NOT be gated (CPU subprocess only)
    ds_id = _create(client)
    captured = {}

    def fake_score_faces(user_id, dataset_id, family=None):
        captured['user_id'] = user_id
        captured['dataset_id'] = dataset_id
        captured['family'] = family
        return {'ranking': []}
    monkeypatch.setattr('app.services.lora_test_studio.score_faces', fake_score_faces)
    resp = client.post(f'/api/dataset/{ds_id}/lora-test/score-faces', json={'family': 'zimage'})
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'ranking': []}
    assert captured == {'user_id': 'local', 'dataset_id': ds_id, 'family': 'zimage'}


# --- prompt delete: DB-only, no gate -----------------------------------------

def test_lora_test_prompt_delete_ungated_when_comfyui_down(client, monkeypatch):
    _comfy(monkeypatch, False)
    ds_id = _create(client)
    resp = client.delete(f'/api/dataset/{ds_id}/lora-test/prompt', json={'prompt': 'anything'})
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'deleted': 0}


# --- /api/index_config: deterministic field inventory ------------------------

def test_index_config_returns_documented_fields(client):
    resp = client.get('/api/index_config')
    assert resp.status_code == 200
    body = resp.get_json()
    # Exact field set: the only fields StudioGenerationSettings.jsx reads off
    # config (config.krea_loras / config.krea_samplers / config.krea_schedulers).
    assert set(body.keys()) == {'krea_loras', 'krea_samplers', 'krea_schedulers'}
    assert body['krea_loras'] == []
    assert 'er_sde' in body['krea_samplers']
    assert 'simple' in body['krea_schedulers']
