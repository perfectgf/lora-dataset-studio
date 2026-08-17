"""Region touch-up: Klein prefill+reconstruct with a custom prompt, without
flagging the image as a watermark. GPU round-trip is mocked."""
import io
import os

import pytest
from PIL import Image, ImageDraw

from app.config import LOCAL_USER


def _img_bytes(color=(200, 30, 30), size=(64, 64), fmt='WEBP'):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, fmt)
    return buf.getvalue()


def _create(client, name='Touch', trigger='touch'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger})


def _kept_image(svc, ds_id, filename, *, size=(64, 64), color=(200, 30, 30),
                state=None):
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'wb') as fh:
        fh.write(_img_bytes(color=color, size=size))
    img = FaceDatasetImage(dataset_id=ds_id, source='import', status='keep',
                           filename=filename, framing='body',
                           watermark_state=state)
    svc.db.session.add(img)
    svc.db.session.commit()
    return img


def _stub_klein(monkeypatch, *, available=True, result=(True, None), recorder=None):
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: available)

    def _fake(user_id, path, boxes=None, **kwargs):
        if recorder is not None:
            recorder.append({'path': path, 'boxes': boxes, **kwargs})
            if result[0]:
                Image.new('RGB', (8, 8), (10, 200, 10)).save(path, 'WEBP')
        return result

    monkeypatch.setattr(wk, 'inpaint_mask_klein', _fake)


def test_compose_region_inpaint_prompt_leads_with_the_user_text():
    from app.services import face_dataset_service as svc
    out = svc.compose_region_inpaint_prompt('remove necklace')
    assert out.startswith('remove necklace. ')
    assert 'Reconstruct the marked area' in out
    assert 'No text, no logos' in out


def test_compose_region_inpaint_prompt_rejects_empty_and_too_long():
    from app.services import face_dataset_service as svc
    with pytest.raises(ValueError, match='required'):
        svc.compose_region_inpaint_prompt('   ')
    with pytest.raises(ValueError, match='required'):
        svc.compose_region_inpaint_prompt(None)
    with pytest.raises(ValueError, match='at most'):
        svc.compose_region_inpaint_prompt('x' * (svc.REGION_INPAINT_PROMPT_MAX + 1))


def test_inpaint_region_does_not_require_a_watermark_flag(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    calls = []
    _stub_klein(monkeypatch, recorder=calls)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = _kept_image(svc, ds.id, 'face.webp', state='none')
        before_state = img.watermark_state
        before_regions = img.watermark_regions
        path = svc._img_path(img)
        before = open(path, 'rb').read()
        out = svc.inpaint_region(
            LOCAL_USER, ds.id, img.id, [[0.2, 0.2, 0.4, 0.4]], 'remove necklace')
        assert out == {'ok': True, 'has_region_touchup': True}
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state == before_state
        assert row.watermark_regions == before_regions
        assert len(calls) == 1
        assert calls[0]['boxes'] == [[0.2, 0.2, 0.4, 0.4]]
        assert 'remove necklace' in calls[0]['prompt']
        stem, ext = os.path.splitext(path)
        assert os.path.exists(f'{stem}.orig{ext}')
        assert os.path.exists(f'{stem}.touchup{ext}')
        assert open(f'{stem}.touchup{ext}', 'rb').read() == before
        assert open(path, 'rb').read() != before


def test_inpaint_region_restore_brings_back_pre_touchup_pixels(app, monkeypatch):
    from app.services import face_dataset_service as svc
    _stub_klein(monkeypatch, recorder=[])
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = _kept_image(svc, ds.id, 'face.webp', color=(10, 20, 30))
        path = svc._img_path(img)
        before = open(path, 'rb').read()
        svc.inpaint_region(
            LOCAL_USER, ds.id, img.id, [[0.1, 0.1, 0.3, 0.3]], 'remove earrings')
        assert open(path, 'rb').read() != before
        out = svc.restore_region_inpaint(LOCAL_USER, ds.id, img.id)
        assert out['ok'] is True
        assert out['has_region_touchup'] is False
        assert open(path, 'rb').read() == before
        stem, ext = os.path.splitext(path)
        assert not os.path.exists(f'{stem}.touchup{ext}')
        from app.models import FaceDatasetImage
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state is None


def test_two_applies_reset_returns_the_first_original(app, monkeypatch):
    """A second apply must not overwrite the write-once snapshot, or Reset
    would only reach the file from just before that second apply."""
    from app.services import face_dataset_service as svc
    _stub_klein(monkeypatch, recorder=[])
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = _kept_image(svc, ds.id, 'face.webp', color=(10, 20, 30))
        path = svc._img_path(img)
        original = open(path, 'rb').read()
        svc.inpaint_region(
            LOCAL_USER, ds.id, img.id, [[0.1, 0.1, 0.3, 0.3]], 'remove earrings')
        after_first = open(path, 'rb').read()
        assert after_first != original
        svc.inpaint_region(
            LOCAL_USER, ds.id, img.id, [[0.4, 0.4, 0.6, 0.6]], 'remove necklace')
        stem, ext = os.path.splitext(path)
        snapshot = f'{stem}.touchup{ext}'
        assert os.path.exists(snapshot)
        assert open(snapshot, 'rb').read() == original
        out = svc.restore_region_inpaint(LOCAL_USER, ds.id, img.id)
        assert out['ok'] is True
        assert out['has_region_touchup'] is False
        assert open(path, 'rb').read() == original
        assert not os.path.exists(snapshot)


def test_inpaint_region_rejects_empty_prompt_and_regions(app, monkeypatch):
    from app.services import face_dataset_service as svc
    _stub_klein(monkeypatch, recorder=[])
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = _kept_image(svc, ds.id, 'face.webp')
        with pytest.raises(ValueError, match='required'):
            svc.inpaint_region(LOCAL_USER, ds.id, img.id, [[0.1, 0.1, 0.3, 0.3]], '  ')
        with pytest.raises(ValueError, match='at least one'):
            svc.inpaint_region(LOCAL_USER, ds.id, img.id, [], 'remove necklace')
        with pytest.raises(ValueError, match='region'):
            svc.inpaint_region(LOCAL_USER, ds.id, img.id, [[0, 0, 0, 0]], 'remove necklace')


def test_inpaint_region_unavailable_raises_without_touching_the_file(app, monkeypatch):
    from app.services import face_dataset_service as svc
    calls = []
    _stub_klein(monkeypatch, available=False, recorder=calls)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = _kept_image(svc, ds.id, 'face.webp')
        path = svc._img_path(img)
        before = open(path, 'rb').read()
        with pytest.raises(RuntimeError, match='not ready'):
            svc.inpaint_region(
                LOCAL_USER, ds.id, img.id, [[0.1, 0.1, 0.3, 0.3]], 'remove jewelry')
        assert calls == []
        assert open(path, 'rb').read() == before
        stem, ext = os.path.splitext(path)
        assert not os.path.exists(f'{stem}.touchup{ext}')


def test_inpaint_region_accepts_a_painted_mask(app, monkeypatch):
    import base64
    from app.services import face_dataset_service as svc
    calls = []
    _stub_klein(monkeypatch, recorder=calls)
    mask = Image.new('L', (64, 64), 0)
    ImageDraw.Draw(mask).ellipse((20, 20, 36, 36), fill=255)
    buf = io.BytesIO()
    mask.save(buf, 'PNG')
    data_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = _kept_image(svc, ds.id, 'face.webp')
        out = svc.inpaint_region(
            LOCAL_USER, ds.id, img.id, None, 'remove earrings', mask=data_url)
        assert out['ok'] is True
        assert calls[0]['boxes'] is None
        assert calls[0]['mask'].mode == 'L'
        assert calls[0]['mask'].getextrema()[1] == 255
        assert 'remove earrings' in calls[0]['prompt']


def test_region_inpaint_route_happy_path(client, app, monkeypatch):
    from app.routes import datasets as routes
    from app.services import face_dataset_service as svc
    monkeypatch.setattr(routes, '_klein_clean_preflight', lambda: None)
    _stub_klein(monkeypatch, recorder=[])
    ds_id = _create(client).get_json()['id']
    with app.app_context():
        img = _kept_image(svc, ds_id, 'face.webp')
        img_id = img.id
        path = svc._img_path(img)
        before = open(path, 'rb').read()
    resp = client.post(
        f'/api/dataset/{ds_id}/image/{img_id}/region-inpaint',
        json={'regions': [[0.2, 0.2, 0.5, 0.5]], 'prompt': 'remove makeup',
              'method': 'klein'})
    assert resp.status_code == 200 and resp.get_json()['ok'] is True
    listed = client.get(f'/api/dataset/{ds_id}').get_json()['images']
    row = next(i for i in listed if i['id'] == img_id)
    assert row['has_region_touchup'] is True
    with app.app_context():
        assert open(path, 'rb').read() != before
        stem, ext = os.path.splitext(path)
        assert os.path.exists(f'{stem}.touchup{ext}')
    rr = client.post(f'/api/dataset/{ds_id}/image/{img_id}/region-inpaint/restore', json={})
    assert rr.status_code == 200 and rr.get_json()['ok'] is True
    assert rr.get_json()['has_region_touchup'] is False
    with app.app_context():
        assert open(path, 'rb').read() == before
        stem, ext = os.path.splitext(path)
        assert not os.path.exists(f'{stem}.touchup{ext}')


def test_region_inpaint_route_400_without_prompt_or_regions(client, app, monkeypatch):
    from app.routes import datasets as routes
    from app.services import face_dataset_service as svc
    monkeypatch.setattr(routes, '_klein_clean_preflight', lambda: None)
    ds_id = _create(client).get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'face.webp').id
    missing_prompt = client.post(
        f'/api/dataset/{ds_id}/image/{img_id}/region-inpaint',
        json={'regions': [[0.2, 0.2, 0.5, 0.5]], 'method': 'klein'})
    assert missing_prompt.status_code == 400
    missing_regions = client.post(
        f'/api/dataset/{ds_id}/image/{img_id}/region-inpaint',
        json={'prompt': 'remove necklace', 'method': 'klein'})
    assert missing_regions.status_code == 400
    empty_prompt = client.post(
        f'/api/dataset/{ds_id}/image/{img_id}/region-inpaint',
        json={'regions': [[0.2, 0.2, 0.5, 0.5]], 'prompt': '  ', 'method': 'klein'})
    assert empty_prompt.status_code == 400


def test_region_inpaint_route_404_unknown_image(client, app, monkeypatch):
    from app.routes import datasets as routes
    monkeypatch.setattr(routes, '_klein_clean_preflight', lambda: None)
    ds_id = _create(client).get_json()['id']
    resp = client.post(
        f'/api/dataset/{ds_id}/image/999999/region-inpaint',
        json={'regions': [[0.2, 0.2, 0.5, 0.5]], 'prompt': 'remove necklace',
              'method': 'klein'})
    assert resp.status_code == 404


def test_region_inpaint_restore_404_without_snapshot(client, app):
    from app.services import face_dataset_service as svc
    ds_id = _create(client).get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'face.webp').id
    resp = client.post(
        f'/api/dataset/{ds_id}/image/{img_id}/region-inpaint/restore', json={})
    assert resp.status_code == 404
    assert 'touch-up' in (resp.get_json().get('error') or '')


def test_region_inpaint_route_503_when_training(client, app):
    from app.job_queue import queue_manager
    ds_id = _create(client).get_json()['id']
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=300)
    try:
        resp = client.post(
            f'/api/dataset/{ds_id}/image/1/region-inpaint',
            json={'regions': [[0.2, 0.2, 0.5, 0.5]], 'prompt': 'remove necklace',
                  'method': 'klein'})
        assert resp.status_code == 503 and 'GPU busy' in resp.get_json()['error']
    finally:
        with app.app_context():
            queue_manager._set_system_state('training_in_progress', None)


def test_region_inpaint_route_409_when_models_missing(client, app, monkeypatch):
    from app.routes import datasets as routes
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    from app import capabilities as caps_mod
    ds_id = _create(client).get_json()['id']
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', None)
        queue_manager._set_system_state('vision_in_progress', None)
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: ['klein_model'])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(caps_mod, 'resolve_comfyui_base',
                        lambda p: {'valid': True, 'resolved': p, 'nested': False})
    monkeypatch.setattr(routes, '_autostart_klein_downloads', lambda missing: ([], False))
    resp = client.post(
        f'/api/dataset/{ds_id}/image/1/region-inpaint',
        json={'regions': [[0.2, 0.2, 0.5, 0.5]], 'prompt': 'remove necklace',
              'method': 'klein'})
    assert resp.status_code == 409
    assert resp.get_json()['klein_missing'] == ['klein_model']
