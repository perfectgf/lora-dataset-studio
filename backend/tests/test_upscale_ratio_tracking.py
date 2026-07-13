"""Upscale-ratio tracking: the composition meter blind spot flagged from a Reddit
technical thread (crop-then-super-resolve biases training toward that local patch;
the discrete face/bust/body/back tally treated a native close-up and a heavily
enlarged crop as equivalent). face_crop_to_square_webp / crop_image now report the
LANCZOS enlargement factor (size / box_side) so it can be persisted per image and
surfaced per framing bucket via dataset_payload()['composition_upscaled'].
"""
import io

import pytest
from PIL import Image

from app.services import face_dataset_service as svc
from app.models import FaceDatasetImage
from app.config import LOCAL_USER


def _png(w, h):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


# --- face_crop_to_square_webp(return_scale=True) ----------------------------
def test_return_scale_reports_upscale_for_small_detected_box(app, monkeypatch):
    """A tight bbox on a modest source image yields a box far smaller than 1024,
    so the LANCZOS resize enlarges it (scale > 1) -- the exact pattern described in
    the thread: crop, then super-resolve, biasing training toward that patch."""
    with app.app_context():
        monkeypatch.setattr(svc, 'detect_head_bbox', lambda *a, **k: (0.4, 0.4, 0.6, 0.6))
        webp, detected, scale = svc.face_crop_to_square_webp(
            _png(1000, 1000), return_detected=True, return_scale=True)
        assert detected is True and webp[:4] == b'RIFF'
        assert scale > 2.5   # box_side=340 -> 1024/340 ~= 3.0


def test_return_scale_no_upscale_for_large_detected_box(app, monkeypatch):
    """A generously-sized source crop shouldn't be flagged: the box is already
    bigger than the 1024 target, so the resize downsamples (scale <= 1)."""
    with app.app_context():
        monkeypatch.setattr(svc, 'detect_head_bbox', lambda *a, **k: (0.2, 0.2, 0.8, 0.8))
        _webp, _detected, scale = svc.face_crop_to_square_webp(
            _png(3000, 3000), return_detected=True, return_scale=True)
        assert scale <= 1.0


def test_return_scale_is_additive_to_return_detected_alone():
    """Existing 2-tuple callers (the /ref route, recrop_reference_auto) must be
    unaffected -- return_scale defaults False and doesn't change that shape."""
    import inspect
    sig = inspect.signature(svc.face_crop_to_square_webp)
    assert sig.parameters['return_scale'].default is False


# --- import_images persists the ratio ---------------------------------------
def test_import_crop_stores_upscale_ratio(app, monkeypatch):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Amy', 'zchar_amy')
        monkeypatch.setattr(svc, 'detect_head_bbox', lambda *a, **k: (0.4, 0.4, 0.6, 0.6))
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png(1000, 1000)], crop=True)
        assert failed == 0 and len(ids) == 1
        img = svc.db.session.get(FaceDatasetImage, ids[0])
        assert img.framing == 'face'
        assert img.upscale_ratio > 2.5


def test_import_without_crop_leaves_upscale_ratio_none(app):
    """normalize_to_webp only shrinks (PIL .thumbnail never enlarges) -- the
    aspect-preserving import path carries no upscale-bias risk, so it stays NULL."""
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Bo', 'zchar_bo')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png(1200, 800)], crop=False)
        assert failed == 0
        img = svc.db.session.get(FaceDatasetImage, ids[0])
        assert img.upscale_ratio is None


# --- manual crop_image persists the ratio ------------------------------------
def test_manual_crop_image_stores_upscale_ratio(app):
    import os
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Cro', 'zchar_cro')
        d = svc._dataset_dir(ds.id)
        os.makedirs(d, exist_ok=True)
        buf = io.BytesIO(); Image.new('RGB', (1600, 1200), (90, 30, 30)).save(buf, 'PNG')
        open(os.path.join(d, 'w.webp'), 'wb').write(buf.getvalue())
        img = FaceDatasetImage(dataset_id=ds.id, filename='w.webp', status='keep', framing='face')
        svc.db.session.add(img); svc.db.session.commit()

        assert svc.crop_image(LOCAL_USER, img.id, 0, 0, 400, 400) is True
        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.upscale_ratio == pytest.approx(1024 / 400)


# --- dataset_payload surfaces composition_upscaled ---------------------------
def test_dataset_payload_flags_upscale_heavy_bucket(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Deb', 'zchar_deb')
        # 2 native face shots + 2 heavily-upscaled face crops (>= threshold).
        for ratio in (None, None, 2.0, 1.6):
            svc.db.session.add(FaceDatasetImage(
                dataset_id=ds.id, filename=f'f{ratio}.webp', status='keep',
                framing='face', upscale_ratio=ratio))
        # A rejected upscaled image must NOT count (mirrors the existing
        # composition tally, which already excludes reject/failed).
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename='rej.webp', status='reject',
            framing='face', upscale_ratio=5.0))
        svc.db.session.commit()

        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['composition']['face'] == 4
        assert payload['composition_upscaled']['face'] == 2
        ratios = sorted((i['upscale_ratio'] for i in payload['images'] if i['status'] == 'keep'),
                        key=lambda r: (r is not None, r))
        assert ratios == [None, None, 1.6, 2.0]


def test_dataset_payload_below_threshold_not_flagged(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Eli', 'zchar_eli')
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename='f.webp', status='keep',
            framing='face', upscale_ratio=svc.UPSCALE_WARN_THRESHOLD - 0.01))
        svc.db.session.commit()

        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        assert payload['composition']['face'] == 1
        assert payload['composition_upscaled']['face'] == 0
