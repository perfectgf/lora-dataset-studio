"""User control over the watermark cleaning method: the "Allow auto-crop" preference
and the review lightbox's per-image crop-vs-inpaint override. Kept in its own module
(test_watermarks.py is large and often touched in parallel).

The point of the feature: a border mark used to be CROPPED with no say from the user.
With allow_crop off (Settings, the batch bar, or a per-image "force inpaint") the same
mark is REPAINTED instead — nothing else about the routing changes, and the shipped
default (crop allowed) is preserved everywhere it isn't explicitly overridden.
"""
import io
import json
import os

from PIL import Image


def _img_bytes(color=(200, 30, 30), size=(1024, 1024), fmt='WEBP'):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, fmt)
    return buf.getvalue()


def _kept_image(svc, ds_id, filename, *, size=(1024, 1024), bbox=None, state='detected'):
    """A kept, 'detected' FaceDatasetImage backed by a real file (routing opens it)."""
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'wb') as fh:
        fh.write(_img_bytes(size=size))
    img = FaceDatasetImage(dataset_id=ds_id, source='import', status='keep',
                           filename=filename, framing='body', watermark_state=state,
                           watermark_bbox=json.dumps(bbox) if bbox is not None else None)
    svc.db.session.add(img)
    svc.db.session.commit()
    return img


# --- pure routing (_route_watermark) ---------------------------------------

def test_route_border_mark_inpaints_when_crop_disallowed(app):
    """The exact grief: a top-border mark routes to 'crop' by default, but to the inpaint
    route ('lama' here — small, off-center) once auto-crop is turned off."""
    from app.services import face_dataset_service as svc
    bbox = (0.0, 0.0, 0.3, 0.05)               # top band, small area (0.015)
    assert svc._route_watermark(bbox, 1024, 1024)[0] == 'crop'
    assert svc._route_watermark(bbox, 1024, 1024, allow_crop=False)[0] == 'lama'


def test_route_large_border_bar_needs_review_when_crop_disallowed(app):
    """A full-width bottom BAR is croppable by default; with crop off its area is too big
    for a safe LaMa repaint, so it honestly falls to manual review (Klein can still take
    it — that's the engine's job, not the router's)."""
    from app.services import face_dataset_service as svc
    bbox = (0.0, 0.85, 1.0, 1.0)               # bottom bar, area 0.15 > max inpaint area
    assert svc._route_watermark(bbox, 1024, 1024)[0] == 'crop'
    assert svc._route_watermark(bbox, 1024, 1024, allow_crop=False)[0] == 'review'


def test_route_non_border_mark_unaffected_by_allow_crop(app):
    """A small off-center mark never had a crop route; disabling crop is a no-op for it."""
    from app.services import face_dataset_service as svc
    bbox = (0.35, 0.35, 0.45, 0.45)
    assert svc._route_watermark(bbox, 1024, 1024)[0] == 'lama'
    assert svc._route_watermark(bbox, 1024, 1024, allow_crop=False)[0] == 'lama'


# --- clean_watermarks (LaMa mocked) ----------------------------------------

def test_clean_border_mark_inpainted_not_cropped_when_disallowed(app, monkeypatch):
    """allow_crop=False: the border mark is REPAINTED in place (LaMa) — the file keeps its
    original dimensions (never a crop) and the .orig is preserved for restore."""
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    seen = {}
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)

    def _fake_inpaint(path, bbox, timeout=300, device='cpu'):
        seen['bbox'] = bbox
        return True, None
    monkeypatch.setattr(watermark_lama, 'inpaint_watermark', _fake_inpaint)

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'wm.webp', size=(1024, 1024), bbox=[0.0, 0.0, 0.3, 0.05])
        path = svc._img_path(img)
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, allow_crop=False)
        assert err is None and counts['inpainted'] == 1 and counts['cropped'] == 0
        assert seen['bbox'] == [0.0, 0.0, 0.3, 0.05]
        with Image.open(path) as im:
            assert im.size == (1024, 1024)                       # NOT cropped
        stem, ext = os.path.splitext(path)
        assert os.path.exists(f'{stem}.orig{ext}')              # original preserved
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'cleaned'


def test_clean_default_still_crops_border(app, monkeypatch):
    """Default (allow_crop unset -> persisted preference, shipped True): unchanged crop."""
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermark',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not inpaint')))
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'wm.webp', size=(1024, 1024), bbox=[0.0, 0.0, 0.3, 0.05])
        path = svc._img_path(img)
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)       # allow_crop=None
        assert err is None and counts['cropped'] == 1
        with Image.open(path) as im:
            assert im.height < 1024                                 # cropped


def test_clean_resolves_persisted_preference_when_unset(app, monkeypatch):
    """allow_crop=None reads Settings' watermark.allow_crop: with it saved False, a plain
    clean (the batch button's call) repaints the border mark instead of cropping."""
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app import config as cfg
    from app.config import LOCAL_USER

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermark', lambda *a, **k: (True, None))
    with app.app_context():
        cfg.save_config({'watermark': {'allow_crop': False}})
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        _kept_image(svc, ds.id, 'wm.webp', size=(1024, 1024), bbox=[0.0, 0.0, 0.3, 0.05])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)       # None -> config False
        assert err is None and counts['inpainted'] == 1 and counts['cropped'] == 0


# --- HTTP route (svc.clean_watermarks spied) -------------------------------

def test_clean_route_forwards_allow_crop_false(client, monkeypatch):
    from app.services import face_dataset_service as svc
    seen = {}
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None, device=None, method=None, allow_crop='_absent': (
                            seen.update(allow_crop=allow_crop)
                            or ({'cropped': 0, 'inpainted': 1, 'inpainted_klein': 0,
                                 'needs_review': 0, 'failed': 0, 'skipped': 0}, None)))
    ds_id = client.post('/api/dataset/create', json={'name': 'R', 'trigger_word': 'r'}).get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'allow_crop': False})
    assert resp.status_code == 200 and resp.get_json()['inpainted'] == 1
    assert seen['allow_crop'] is False


def test_clean_route_omitting_allow_crop_does_not_forward_it(client, monkeypatch):
    """A plain {method:...} call keeps its exact old shape — allow_crop is NOT passed, so
    clean_watermarks falls back to the persisted preference on its own."""
    from app.services import face_dataset_service as svc
    seen = {}
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None, device=None, method=None, **kw: (
                            seen.update(kw=kw)
                            or ({'cropped': 1, 'inpainted': 0, 'inpainted_klein': 0,
                                 'needs_review': 0, 'failed': 0, 'skipped': 0}, None)))
    ds_id = client.post('/api/dataset/create', json={'name': 'R', 'trigger_word': 'r'}).get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={})
    assert resp.status_code == 200
    assert 'allow_crop' not in seen['kw']


def test_clean_route_rejects_non_bool_allow_crop(client):
    ds_id = client.post('/api/dataset/create', json={'name': 'R', 'trigger_word': 'r'}).get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'allow_crop': 'yes'})
    assert resp.status_code == 400


# --- payload + config + capabilities ---------------------------------------

def test_payload_exposes_route_nocrop_for_border_marks(app):
    """The lightbox reads BOTH routes: a border mark reports crop by default and its
    inpaint fallback under watermark_route_nocrop; a non-border mark reports the same
    route for both; a clean image carries neither."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'P', 'p')
        _kept_image(svc, ds.id, 'band.webp', bbox=[0.0, 0.0, 0.3, 0.05])       # crop -> lama
        _kept_image(svc, ds.id, 'small.webp', bbox=[0.35, 0.35, 0.45, 0.45])   # lama -> lama
        _kept_image(svc, ds.id, 'clean.webp', bbox=None, state='none')
        by_name = {i['filename']: i for i in svc.dataset_payload(LOCAL_USER, ds.id)['images']}
        assert by_name['band.webp']['watermark_route'] == 'crop'
        assert by_name['band.webp']['watermark_route_nocrop'] == 'lama'
        assert by_name['small.webp']['watermark_route'] == 'lama'
        assert by_name['small.webp']['watermark_route_nocrop'] == 'lama'
        assert by_name['clean.webp']['watermark_route'] is None
        assert by_name['clean.webp']['watermark_route_nocrop'] is None


def test_config_default_allow_crop_true(app):
    from app import config as cfg
    with app.app_context():
        assert cfg.get('watermark.allow_crop') is True


def test_capabilities_exposes_allow_crop(app, monkeypatch):
    from app import capabilities, config as cfg
    with app.app_context():
        assert capabilities.probe(force=True)['watermark_allow_crop'] is True
        cfg.save_config({'watermark': {'allow_crop': False}})
        assert capabilities.probe(force=True)['watermark_allow_crop'] is False
