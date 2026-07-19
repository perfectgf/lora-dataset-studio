"""🧪 Caption Lab — per-candidate preview endpoint. Runs ONE caption config on ONE
image and returns the text WITHOUT persisting it (ephemeral A/B probe). The Ollama
vision seam is mocked so the pass is hermetic, exactly like the image-bank tests."""
import os

from PIL import Image


def _use_ollama_backend(app):
    """Force the Ollama backend so JoyCaption (ai-toolkit) is skipped in the preview."""
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': 'ollama'}})


def _mock_vision(monkeypatch, caption='a plain description', capture=None):
    """Mock describe_image_ollama; when `capture` is a dict, record the prompt it saw
    so a test can assert the vocabulary/instructions were appended."""
    from app.services import vision_ollama

    def fake_describe(image_bytes, prompt, *a, **k):
        if capture is not None:
            capture['prompt'] = prompt
        return caption

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


def _dataset_with_image(client, app, filename='a.png', caption=''):
    """Create a dataset + one kept image, writing the file to the dataset dir so the
    preview's on-disk check passes. Returns (dataset_id, image_id)."""
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'Lab', 'trigger_word': 'lab'}).get_json()['id']
    with app.app_context():
        from app.models import FaceDatasetImage
        from app.services import face_dataset_service as svc
        from app.services.dataset_storage import ensure_dataset_dir
        Image.new('RGB', (64, 64), (128, 128, 128)).save(
            os.path.join(ensure_dataset_dir(ds_id), filename))
        img = FaceDatasetImage(dataset_id=ds_id, status='keep', source='upload',
                               filename=filename, caption=caption)
        svc.db.session.add(img)
        svc.db.session.commit()
        return ds_id, img.id


def _preview(client, ds_id, img_id, **body):
    return client.post(
        f'/api/dataset/{ds_id}/image/{img_id}/caption/preview', json=body)


# --- happy path ---------------------------------------------------------------
def test_preview_returns_caption_without_persisting(client, app, monkeypatch):
    _use_ollama_backend(app)
    ds_id, img_id = _dataset_with_image(client, app, caption='ORIGINAL')
    _mock_vision(monkeypatch, caption='a candidate caption')

    r = _preview(client, ds_id, img_id)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['ok'] is True
    assert body['caption'] == 'a candidate caption'
    assert body['chars'] == len('a candidate caption')
    assert 'duration_ms' in body and body['cancelled'] is False

    # The stored caption must be untouched — a preview never writes.
    with app.app_context():
        from app.models import FaceDatasetImage
        from app.services import face_dataset_service as svc
        assert svc.db.session.get(FaceDatasetImage, img_id).caption == 'ORIGINAL'


def test_preview_appends_vocabulary_and_instructions(client, app, monkeypatch):
    _use_ollama_backend(app)
    ds_id, img_id = _dataset_with_image(client, app)
    capture = {}
    _mock_vision(monkeypatch, capture=capture)

    r = _preview(client, ds_id, img_id, vocabulary='explicit',
                 instructions='Name the visible clothing colors.')
    assert r.status_code == 200, r.get_json()
    # Both the vocabulary preset register and the free instructions ride in the prompt.
    assert 'crude anatomical terms' in capture['prompt']
    assert 'Name the visible clothing colors.' in capture['prompt']


def test_preview_default_backend_leaves_prompt_clean(client, app, monkeypatch):
    _use_ollama_backend(app)
    ds_id, img_id = _dataset_with_image(client, app)
    capture = {}
    _mock_vision(monkeypatch, capture=capture)

    r = _preview(client, ds_id, img_id)
    assert r.status_code == 200
    # No vocabulary, no instructions → the descriptive prompt is not augmented.
    assert 'crude anatomical terms' not in capture['prompt']


# --- guards -------------------------------------------------------------------
def test_preview_409_when_batch_in_progress(client, app, monkeypatch):
    _use_ollama_backend(app)
    ds_id, img_id = _dataset_with_image(client, app)
    _mock_vision(monkeypatch)
    from app.services import dataset_activity
    dataset_activity.begin(ds_id, 'caption', total=5)  # a real pass owns the GPU
    try:
        r = _preview(client, ds_id, img_id)
        assert r.status_code == 409
        assert 'batch' in r.get_json()['error'].lower()
    finally:
        dataset_activity.reset()


def test_preview_400_on_unknown_image(client, app):
    _use_ollama_backend(app)
    ds_id, _ = _dataset_with_image(client, app)
    r = _preview(client, ds_id, 999999)
    assert r.status_code == 400


def test_preview_400_on_invalid_backend(client, app):
    ds_id, img_id = _dataset_with_image(client, app)
    r = _preview(client, ds_id, img_id, backend='nonsense')
    assert r.status_code == 400


def test_preview_404_on_unknown_dataset(client, app):
    r = client.post('/api/dataset/424242/image/1/caption/preview', json={})
    assert r.status_code == 404


# --- cancel (service level: the stop path the route wires to Stop) ------------
def test_preview_cancel_returns_empty_flagged(client, app, monkeypatch):
    _use_ollama_backend(app)
    ds_id, img_id = _dataset_with_image(client, app)
    _mock_vision(monkeypatch, caption='should never be reached')
    with app.app_context():
        from app.config import LOCAL_USER
        from app.services import face_dataset_service as svc
        result = svc.preview_caption(LOCAL_USER, ds_id, img_id, backend='ollama',
                                     should_cancel=lambda: True)
    assert result['caption'] == ''
    assert result['cancelled'] is True
