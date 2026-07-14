"""Watermark auto-correction V1: bbox parsing, crop/LaMa/review routing, the detect
& clean batches, the routes (LaMa mocked present AND absent), the tuple error contract,
original preservation, and the additive columns."""
import io
import json
import os
import sqlite3

import pytest
from PIL import Image


def _img_bytes(color=(200, 30, 30), size=(64, 64), fmt='WEBP'):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, fmt)
    return buf.getvalue()


def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger})


def _kept_image(svc, ds_id, filename, *, size=(1024, 1024), state='detected', bbox=None):
    """A kept FaceDatasetImage backed by a REAL file on disk (routing/clean open it)."""
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'wb') as fh:
        fh.write(_img_bytes(size=size))
    img = FaceDatasetImage(dataset_id=ds_id, source='import', status='keep',
                           filename=filename, framing='body',
                           watermark_state=state,
                           watermark_bbox=json.dumps(bbox) if bbox is not None else None)
    svc.db.session.add(img)
    svc.db.session.commit()
    return img


# --- detect_watermark_bbox (mock vision) -----------------------------------

def test_bbox_present_true_scales_and_expands_margin(app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    monkeypatch.setattr(vo, 'describe_image_ollama',
                        lambda *a, **k: '{"present":true,"x1":100,"y1":200,"x2":300,"y2":260}')
    with app.app_context():
        bbox = svc.detect_watermark_bbox(b'x')
    m = svc._WATERMARK_BBOX_MARGIN
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert abs(x1 - (0.100 - m)) < 1e-6 and abs(y1 - (0.200 - m)) < 1e-6
    assert abs(x2 - (0.300 + m)) < 1e-6 and abs(y2 - (0.260 + m)) < 1e-6


def test_bbox_present_false_returns_none(app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: '{"present":false}')
    with app.app_context():
        assert svc.detect_watermark_bbox(b'x') is None


def test_bbox_swapped_corners_normalized(app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    # x1>x2 and y1>y2 -> min/max normalization (same box as the present-true test)
    monkeypatch.setattr(vo, 'describe_image_ollama',
                        lambda *a, **k: '{"present":true,"x1":300,"y1":260,"x2":100,"y2":200}')
    with app.app_context():
        bbox = svc.detect_watermark_bbox(b'x')
    m = svc._WATERMARK_BBOX_MARGIN
    assert bbox is not None and abs(bbox[0] - (0.1 - m)) < 1e-6 and abs(bbox[2] - (0.3 + m)) < 1e-6


def test_bbox_margin_clamped_to_unit_range(app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    monkeypatch.setattr(vo, 'describe_image_ollama',
                        lambda *a, **k: '{"present":true,"x1":0,"y1":0,"x2":1000,"y2":30}')
    with app.app_context():
        bbox = svc.detect_watermark_bbox(b'x')
    assert bbox == (0.0, 0.0, 1.0, min(1.0, 0.03 + svc._WATERMARK_BBOX_MARGIN))


def test_bbox_unparseable_or_empty_returns_none(app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    with app.app_context():
        monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: 'not json at all')
        assert svc.detect_watermark_bbox(b'x') is None
        monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: '')
        assert svc.detect_watermark_bbox(b'x') is None
        # present true but no usable box
        monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: '{"present":true}')
        assert svc.detect_watermark_bbox(b'x') is None


# --- _route_watermark (pure routing) ---------------------------------------

def test_route_top_border_crops_below_the_mark(app):
    from app.services import face_dataset_service as svc
    route, box = svc._route_watermark((0.0, 0.0, 1.0, 0.10), 1024, 1024)
    assert route == 'crop'
    assert box == (0, round(0.10 * 1024), 1024, 1024)


def test_route_bottom_border_crops_above_the_mark(app):
    from app.services import face_dataset_service as svc
    route, box = svc._route_watermark((0.0, 0.90, 1.0, 1.0), 1024, 1024)
    assert route == 'crop'
    assert box == (0, 0, 1024, round(0.90 * 1024))


def test_route_min_side_768_guard_blocks_thin_crop(app):
    from app.services import face_dataset_service as svc
    bbox = (0.0, 0.0, 0.5, 0.16)          # top-left strip, area 0.08
    # Tall enough → cropping keeps > 768 px → crop.
    assert svc._route_watermark(bbox, 1024, 1024)[0] == 'crop'
    # Too short: 900 - round(0.16*900)=756 < 768 → NOT crop, falls to inpaint.
    assert svc._route_watermark(bbox, 1024, 900)[0] == 'lama'


def test_route_small_offcenter_inpaints(app):
    from app.services import face_dataset_service as svc
    assert svc._route_watermark((0.35, 0.35, 0.45, 0.45), 1024, 1024)[0] == 'lama'


def test_route_large_needs_review(app):
    from app.services import face_dataset_service as svc
    assert svc._route_watermark((0.2, 0.2, 0.8, 0.8), 1024, 1024)[0] == 'review'


def test_route_center_overlap_needs_review(app):
    from app.services import face_dataset_service as svc
    # small area but straddling the exact center point → on-subject → review
    assert svc._route_watermark((0.45, 0.45, 0.55, 0.55), 1024, 1024)[0] == 'review'


# --- additive columns -------------------------------------------------------

def test_watermark_columns_migrated(app):
    from app.extensions import db
    from sqlalchemy import text
    with app.app_context():
        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset_image)'))}
        assert 'watermark_state' in cols and 'watermark_bbox' in cols


def test_watermark_regions_column_migrated(app):
    from app.extensions import db
    from sqlalchemy import text
    with app.app_context():
        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset_image)'))}
        assert 'watermark_regions' in cols


def test_watermark_regions_column_added_to_legacy_database(tmp_path, monkeypatch):
    legacy_db = tmp_path / 'legacy.db'
    with sqlite3.connect(legacy_db) as conn:
        conn.execute('''
            CREATE TABLE face_dataset_image (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                watermark_state VARCHAR(16),
                watermark_bbox TEXT
            )
        ''')
        conn.execute(
            'INSERT INTO face_dataset_image '
            '(id, dataset_id, watermark_state, watermark_bbox) VALUES (1, 7, ?, ?)',
            ('detected', '[0.1, 0.1, 0.2, 0.2]'),
        )

    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as cfg
    monkeypatch.setattr(cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(cfg, '_cache', None)
    from app import create_app
    from app.extensions import db
    from sqlalchemy import text
    legacy_app = create_app({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{legacy_db}',
    })

    with legacy_app.app_context():
        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset_image)'))}
        bbox = db.session.execute(text(
            'SELECT watermark_bbox FROM face_dataset_image WHERE id = 1'
        )).scalar_one()

    assert 'watermark_regions' in cols
    assert bbox == '[0.1, 0.1, 0.2, 0.2]'


# --- manual watermark region validation -----------------------------------

@pytest.mark.parametrize('regions', [
    [[0.1, 0.2, 0.1, 0.3]],          # zero width
    [[0.1, 0.2, 0.104, 0.3]],        # below minimum width
    [[0.1, 0.2, 0.1049, 0.3]],       # still below minimum width
    [[-0.1, 0.2, 0.3, 0.4]],         # outside image
    [[0.1, 0.2, float('nan'), 0.4]],  # non-finite
    [[True, 0.2, 0.3, 0.4]],         # bool is not a coordinate
    [[0.1, 0.2, 0.3]],               # wrong arity
    'not-a-list',
])
def test_normalize_watermark_regions_rejects_invalid(regions):
    from app.services import face_dataset_service as svc
    with pytest.raises(ValueError):
        svc.normalize_watermark_regions(regions)


def test_normalize_watermark_regions_rounds_and_preserves_separate_boxes():
    from app.services import face_dataset_service as svc
    assert svc.normalize_watermark_regions([
        [0.10004, 0.20004, 0.30006, 0.40006],
        [0.7, 0.7, 0.9, 0.9],
    ]) == [[0.1, 0.2, 0.3001, 0.4001], [0.7, 0.7, 0.9, 0.9]]


def test_normalize_watermark_regions_accepts_exact_minimum_side():
    from app.services import face_dataset_service as svc
    assert svc.normalize_watermark_regions([
        [0.1, 0.2, 0.105, 0.205],
    ]) == [[0.1, 0.2, 0.105, 0.205]]


def test_normalize_watermark_regions_controls_null_acceptance():
    from app.services import face_dataset_service as svc
    assert svc.normalize_watermark_regions(None) is None
    with pytest.raises(ValueError):
        svc.normalize_watermark_regions(None, allow_null=False)


# --- detect_watermarks batch ------------------------------------------------

def test_detect_watermarks_persists_state_and_bbox(app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    raws = ['{"present":true,"x1":0,"y1":0,"x2":100,"y2":50}', '{"present":false}']
    monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: raws.pop(0))
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'W', 'w')
        a = _kept_image(svc, ds.id, 'a.webp', state=None)
        b = _kept_image(svc, ds.id, 'b.webp', state=None)
        counts = svc.detect_watermarks(LOCAL_USER, ds.id)
        assert counts == {'detected': 1, 'none': 1, 'checked': 2}
        m = svc._WATERMARK_BBOX_MARGIN
        ra = svc.db.session.get(FaceDatasetImage, a.id)
        rb = svc.db.session.get(FaceDatasetImage, b.id)
        assert ra.watermark_state == 'detected'
        assert json.loads(ra.watermark_bbox) == [0.0, 0.0, round(0.1 + m, 4), round(0.05 + m, 4)]
        assert rb.watermark_state == 'none' and rb.watermark_bbox is None


def test_detect_watermarks_skips_on_empty_vision(app, monkeypatch):
    """Ollama down / empty answer must NOT falsely mark images 'none' (same careful
    handling as classify_images) — the state is left untouched so a retry can run."""
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: '')
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'W', 'w')
        a = _kept_image(svc, ds.id, 'a.webp', state=None)
        counts = svc.detect_watermarks(LOCAL_USER, ds.id)
        assert counts == {'detected': 0, 'none': 0, 'checked': 0}
        assert svc.db.session.get(FaceDatasetImage, a.id).watermark_state is None


# --- clean_watermarks routing (LaMa mocked) --------------------------------

def test_clean_crops_border_and_preserves_original(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'wm.webp', size=(1024, 1024), bbox=[0.0, 0.0, 1.0, 0.05])
        path = svc._img_path(img)
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)
        assert err is None and counts['cropped'] == 1
        stem, ext = os.path.splitext(path)
        assert os.path.exists(f'{stem}.orig{ext}')          # original preserved
        # cropped file is smaller in height (band removed), never taller
        with Image.open(path) as im:
            assert im.height < 1024
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'cleaned'


def test_clean_inpaints_small_offcenter(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    called = {}
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)

    def _fake_inpaint(path, bbox, timeout=300):
        called['path'] = path
        called['bbox'] = bbox
        return True, None

    monkeypatch.setattr(watermark_lama, 'inpaint_watermark', _fake_inpaint)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.35, 0.35, 0.45, 0.45])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)
        assert err is None and counts['inpainted'] == 1
        assert called['bbox'] == [0.35, 0.35, 0.45, 0.45]
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'cleaned'


def test_clean_inpaint_failure_surfaces_error_tuple(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermark',
                        lambda path, bbox, timeout=300: (False, {'kind': 'failed', 'detail': 'RuntimeError: boom'}))
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.35, 0.35, 0.45, 0.45])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)
        assert counts['failed'] == 1
        assert err == {'kind': 'failed', 'detail': 'RuntimeError: boom'}
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'failed'


def test_clean_lama_absent_skips_inpaint_no_error(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: False)

    def _boom(*a, **k):
        raise AssertionError('must not inpaint when LaMa is unavailable')

    monkeypatch.setattr(watermark_lama, 'inpaint_watermark', _boom)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.35, 0.35, 0.45, 0.45])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)
        assert err is None and counts['skipped'] == 1 and counts['inpainted'] == 0
        # left 'detected' so the crop-only pass didn't lose it
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'detected'


def test_clean_large_mark_needs_review_no_edit(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.2, 0.2, 0.8, 0.8])
        path = svc._img_path(img)
        before = os.path.getsize(path)
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)
        assert err is None and counts['needs_review'] == 1
        assert os.path.getsize(path) == before          # file untouched
        stem, ext = os.path.splitext(path)
        assert not os.path.exists(f'{stem}.orig{ext}')  # nothing preserved (nothing changed)
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'detected'


# --- routes -----------------------------------------------------------------

def test_detect_route_returns_counts(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'detect_watermarks',
                        lambda u, d, include_dismissed=False: {'detected': 1, 'none': 2, 'checked': 3})
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/detect')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] and body['detected'] == 1 and body['checked'] == 3


def test_detect_route_503_while_vision_flag_set(client, app):
    from app.job_queue import queue_manager
    ds_id = _create(client, 'R', 'r').get_json()['id']
    with app.app_context():
        queue_manager._set_system_state('vision_in_progress', True, ttl_seconds=300)
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/detect')
    assert resp.status_code == 503 and 'GPU busy' in resp.get_json()['error']


def test_clean_route_lama_present(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None: ({'cropped': 2, 'inpainted': 1, 'needs_review': 0,
                                       'failed': 0, 'skipped': 0}, None))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] and body['cropped'] == 2 and body['inpainted'] == 1 and body['error'] is None


def test_clean_route_lama_absent_reports_skipped(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None: ({'cropped': 1, 'inpainted': 0, 'needs_review': 0,
                                       'failed': 0, 'skipped': 3}, None))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    body = resp.get_json()
    assert body['ok'] and body['skipped'] == 3 and body['cropped'] == 1


def test_clean_route_surfaces_error(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None: ({'cropped': 0, 'inpainted': 0, 'needs_review': 0,
                                       'failed': 1, 'skipped': 0},
                                      {'kind': 'failed', 'detail': 'boom'}))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    body = resp.get_json()
    assert body['error'] == {'kind': 'failed', 'detail': 'boom'}


def test_watermark_routes_404_when_dataset_missing(client):
    assert client.post('/api/dataset/999999/watermarks/detect').status_code == 404
    assert client.post('/api/dataset/999999/watermarks/clean').status_code == 404


# --- manual watermark region route ----------------------------------------

def test_watermark_regions_route_replaces_override_and_returns_payload(client, app):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'Regions', 'regions').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'regions.webp', state='detected',
                             bbox=[0.1, 0.1, 0.2, 0.2]).id

    url = f'/api/dataset/{ds_id}/image/{img_id}/watermark-regions'
    first = [[0.2, 0.2, 0.3, 0.3]]
    replacement = [[0.4, 0.4, 0.5, 0.5], [0.7, 0.7, 0.9, 0.9]]
    assert client.put(url, json={'regions': first}).status_code == 200
    response = client.put(url, json={'regions': replacement})

    assert response.status_code == 200
    assert response.get_json() == {
        'ok': True,
        'watermark_regions': replacement,
        'effective_watermark_regions': replacement,
    }
    with app.app_context():
        img = svc.db.session.get(FaceDatasetImage, img_id)
        assert json.loads(img.watermark_regions) == replacement
        assert json.loads(img.watermark_bbox) == [0.1, 0.1, 0.2, 0.2]


def test_watermark_regions_route_stores_empty_override(client, app):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'Empty regions', 'empty-regions').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'empty.webp', state='detected',
                             bbox=[0.1, 0.1, 0.2, 0.2]).id

    response = client.put(
        f'/api/dataset/{ds_id}/image/{img_id}/watermark-regions',
        json={'regions': []},
    )

    assert response.status_code == 200
    assert response.get_json()['watermark_regions'] == []
    assert response.get_json()['effective_watermark_regions'] == []
    with app.app_context():
        assert svc.db.session.get(FaceDatasetImage, img_id).watermark_regions == '[]'


def test_watermark_regions_route_null_restores_detection(client, app):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'Reset regions', 'reset-regions').get_json()['id']
    bbox = [0.1, 0.1, 0.2, 0.2]
    with app.app_context():
        img = _kept_image(svc, ds_id, 'reset.webp', state='detected', bbox=bbox)
        img.watermark_regions = json.dumps([[0.3, 0.3, 0.4, 0.4]])
        svc.db.session.commit()
        img_id = img.id

    response = client.put(
        f'/api/dataset/{ds_id}/image/{img_id}/watermark-regions',
        json={'regions': None},
    )

    assert response.status_code == 200
    assert response.get_json()['watermark_regions'] is None
    assert response.get_json()['effective_watermark_regions'] == [bbox]
    with app.app_context():
        assert svc.db.session.get(FaceDatasetImage, img_id).watermark_regions is None


def test_watermark_regions_route_rejects_too_many_regions(client, app):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'Limit regions', 'limit-regions').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'limit.webp', state='detected',
                             bbox=[0.1, 0.1, 0.2, 0.2]).id
    regions = [[0.1, 0.1, 0.2, 0.2] for _ in range(33)]

    response = client.put(
        f'/api/dataset/{ds_id}/image/{img_id}/watermark-regions',
        json={'regions': regions},
    )

    assert response.status_code == 400


def test_watermark_regions_route_rejects_integer_beyond_float_range(client, app):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'Huge region', 'huge-region').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'huge.webp', state='detected',
                             bbox=[0.1, 0.1, 0.2, 0.2]).id

    response = client.put(
        f'/api/dataset/{ds_id}/image/{img_id}/watermark-regions',
        json={'regions': [[0.1, 0.1, 10 ** 309, 0.2]]},
    )

    assert response.status_code == 400


@pytest.mark.parametrize('body', [
    {},
    {'regions': [[0.4, 0.2, 0.3, 0.4]]},
    {'regions': [[True, 0.2, 0.3, 0.4]]},
])
def test_watermark_regions_route_rejects_missing_or_malformed_coordinates(client, app, body):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'Invalid regions', 'invalid-regions').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'invalid.webp', state='detected',
                             bbox=[0.1, 0.1, 0.2, 0.2]).id

    response = client.put(
        f'/api/dataset/{ds_id}/image/{img_id}/watermark-regions',
        json=body,
    )

    assert response.status_code == 400


def test_watermark_regions_route_returns_404_for_missing_or_foreign_image(client, app):
    from app.services import face_dataset_service as svc
    owned_id = _create(client, 'Owned dataset', 'owned').get_json()['id']
    foreign_id = _create(client, 'Foreign dataset', 'foreign').get_json()['id']
    with app.app_context():
        foreign_image_id = _kept_image(
            svc, foreign_id, 'foreign.webp', state='detected',
            bbox=[0.1, 0.1, 0.2, 0.2],
        ).id
    body = {'regions': [[0.2, 0.2, 0.3, 0.3]]}

    assert client.put(
        f'/api/dataset/{owned_id}/image/999999/watermark-regions', json=body,
    ).status_code == 404
    assert client.put(
        f'/api/dataset/{owned_id}/image/{foreign_image_id}/watermark-regions', json=body,
    ).status_code == 404


def test_set_watermark_regions_enforces_user_and_dataset_ownership(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        owned = svc.create_dataset(LOCAL_USER, 'Owned service dataset', 'owned-service')
        foreign = svc.create_dataset(LOCAL_USER, 'Foreign service dataset', 'foreign-service')
        foreign_image = _kept_image(
            svc, foreign.id, 'foreign-service.webp', state='detected',
            bbox=[0.1, 0.1, 0.2, 0.2],
        )
        regions = [[0.2, 0.2, 0.3, 0.3]]

        assert svc.set_watermark_regions(
            LOCAL_USER, owned.id, foreign_image.id, regions,
        ) is None
        assert svc.set_watermark_regions(
            'another-user', foreign.id, foreign_image.id, regions,
        ) is None


def test_set_watermark_regions_rechecks_detected_state_at_write(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Concurrent state', 'concurrent-state')
        img = _kept_image(svc, ds.id, 'concurrent.webp', state='detected',
                          bbox=[0.1, 0.1, 0.2, 0.2])
        img_id = img.id
        original_normalize = svc.normalize_watermark_regions

        def change_state_between_check_and_write(value, *, allow_null=True):
            current = svc.db.session.get(FaceDatasetImage, img_id)
            current.watermark_state = 'dismissed'
            svc.db.session.commit()
            return original_normalize(value, allow_null=allow_null)

        monkeypatch.setattr(
            svc, 'normalize_watermark_regions', change_state_between_check_and_write,
        )

        with pytest.raises(RuntimeError, match='no longer detected'):
            svc.set_watermark_regions(
                LOCAL_USER, ds.id, img_id, [[0.2, 0.2, 0.3, 0.3]],
            )

        current = svc.db.session.get(FaceDatasetImage, img_id)
        assert current.watermark_state == 'dismissed'
        assert current.watermark_regions is None


def test_watermark_regions_route_returns_409_when_image_not_detected(client, app):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'Clean image', 'clean-image').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'clean.webp', state='none').id

    response = client.put(
        f'/api/dataset/{ds_id}/image/{img_id}/watermark-regions',
        json={'regions': [[0.2, 0.2, 0.3, 0.3]]},
    )

    assert response.status_code == 409


# --- payload ----------------------------------------------------------------

def test_dataset_payload_exposes_watermark_fields(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'P', 'p')
        _kept_image(svc, ds.id, 'wm.webp', state='detected', bbox=[0.1, 0.1, 0.2, 0.15])
        payload = svc.dataset_payload(LOCAL_USER, ds.id)
        img = payload['images'][0]
        assert img['watermark_state'] == 'detected'
        assert img['watermark_bbox'] == [0.1, 0.1, 0.2, 0.15]
        assert img['watermark_regions'] is None
        assert img['effective_watermark_regions'] == [[0.1, 0.1, 0.2, 0.15]]


def test_dataset_payload_prefers_stored_watermark_regions(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    regions = [[0.2, 0.2, 0.3, 0.3], [0.7, 0.7, 0.9, 0.9]]
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Manual regions', 'manual-regions')
        img = _kept_image(svc, ds.id, 'manual.webp', state='detected',
                          bbox=[0.1, 0.1, 0.2, 0.2])
        img.watermark_regions = json.dumps(regions)
        svc.db.session.commit()

        payload_img = svc.dataset_payload(LOCAL_USER, ds.id)['images'][0]

        assert payload_img['watermark_regions'] == regions
        assert payload_img['effective_watermark_regions'] == regions


def test_payload_exposes_watermark_route_for_detected_only(app):
    """The review lightbox and the 🚩 tooltip read the exact planned action from the
    payload (no _route_watermark duplicated in JS). Only 'detected' rows carry it."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'P', 'p')
        # a full-width top band → 'crop'
        _kept_image(svc, ds.id, 'band.webp', size=(1024, 1024),
                    state='detected', bbox=[0.0, 0.0, 1.0, 0.05])
        # a small off-center mark → 'lama'
        _kept_image(svc, ds.id, 'small.webp', size=(1024, 1024),
                    state='detected', bbox=[0.35, 0.35, 0.45, 0.45])
        # not detected → no route
        _kept_image(svc, ds.id, 'clean.webp', state='none')
        by_name = {i['filename']: i for i in svc.dataset_payload(LOCAL_USER, ds.id)['images']}
        assert by_name['band.webp']['watermark_route'] == 'crop'
        assert by_name['small.webp']['watermark_route'] == 'lama'
        assert by_name['clean.webp']['watermark_route'] is None


# --- dismiss (false positive) ----------------------------------------------

def test_dismiss_watermarks_transitions_detected_only(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'D', 'd')
        a = _kept_image(svc, ds.id, 'a.webp', state='detected', bbox=[0.1, 0.1, 0.2, 0.2])
        b = _kept_image(svc, ds.id, 'b.webp', state='none')       # not detected → ignored
        n = svc.dismiss_watermarks(LOCAL_USER, ds.id, [a.id, b.id])
        assert n == 1
        assert svc.db.session.get(FaceDatasetImage, a.id).watermark_state == 'dismissed'
        assert svc.db.session.get(FaceDatasetImage, b.id).watermark_state == 'none'


def test_dismiss_watermarks_ignores_foreign_ids(app):
    """A stale/foreign id (another dataset) must never transition — same scoping as
    batch_image_action (ownership enforced on the dataset)."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    with app.app_context():
        d1 = svc.create_dataset(LOCAL_USER, 'One', 'one')
        d2 = svc.create_dataset(LOCAL_USER, 'Two', 'two')
        foreign = _kept_image(svc, d2.id, 'x.webp', state='detected', bbox=[0.1, 0.1, 0.2, 0.2])
        assert svc.dismiss_watermarks(LOCAL_USER, d1.id, [foreign.id]) == 0
        assert svc.db.session.get(FaceDatasetImage, foreign.id).watermark_state == 'detected'


def test_detect_skips_dismissed_and_include_dismissed_reexamines(app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    # Vision would flag anything it's asked about.
    monkeypatch.setattr(vo, 'describe_image_ollama',
                        lambda *a, **k: '{"present":true,"x1":0,"y1":0,"x2":100,"y2":50}')
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'W', 'w')
        fresh = _kept_image(svc, ds.id, 'a.webp', state=None)
        dismissed = _kept_image(svc, ds.id, 'b.webp', state='dismissed')
        counts = svc.detect_watermarks(LOCAL_USER, ds.id)
        # only the fresh image is examined; the dismissed one is skipped entirely
        assert counts == {'detected': 1, 'none': 0, 'checked': 1}
        assert svc.db.session.get(FaceDatasetImage, dismissed.id).watermark_state == 'dismissed'
        # explicit opt-in re-examines it → re-flagged
        counts2 = svc.detect_watermarks(LOCAL_USER, ds.id, include_dismissed=True)
        assert counts2['checked'] == 2
        assert svc.db.session.get(FaceDatasetImage, dismissed.id).watermark_state == 'detected'


# --- clean subset (image_ids) ----------------------------------------------

def test_clean_watermarks_subset_by_image_ids(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        a = _kept_image(svc, ds.id, 'a.webp', size=(1024, 1024), bbox=[0.0, 0.0, 1.0, 0.05])
        b = _kept_image(svc, ds.id, 'b.webp', size=(1024, 1024), bbox=[0.0, 0.0, 1.0, 0.05])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, image_ids=[a.id])
        assert err is None and counts['cropped'] == 1
        assert svc.db.session.get(FaceDatasetImage, a.id).watermark_state == 'cleaned'
        # b was NOT in the subset → left detected, untouched on disk
        assert svc.db.session.get(FaceDatasetImage, b.id).watermark_state == 'detected'


def test_clean_watermarks_empty_image_ids_cleans_nothing(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        a = _kept_image(svc, ds.id, 'a.webp', bbox=[0.0, 0.0, 1.0, 0.05])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, image_ids=[])
        assert err is None and sum(counts.values()) == 0
        assert svc.db.session.get(FaceDatasetImage, a.id).watermark_state == 'detected'


# --- new routes -------------------------------------------------------------

def test_dismiss_route_marks_and_validates(client, app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    ds_id = _create(client, 'R', 'r').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'a.webp', state='detected', bbox=[0.1, 0.1, 0.2, 0.2]).id
    # missing / empty list → 400
    assert client.post(f'/api/dataset/{ds_id}/watermarks/dismiss', json={}).status_code == 400
    assert client.post(f'/api/dataset/{ds_id}/watermarks/dismiss', json={'image_ids': []}).status_code == 400
    r = client.post(f'/api/dataset/{ds_id}/watermarks/dismiss', json={'image_ids': [img_id]})
    assert r.status_code == 200 and r.get_json()['dismissed'] == 1
    with app.app_context():
        assert svc.db.session.get(FaceDatasetImage, img_id).watermark_state == 'dismissed'


def test_dismiss_route_404_when_dataset_missing(client):
    assert client.post('/api/dataset/999999/watermarks/dismiss',
                       json={'image_ids': [1]}).status_code == 404


def test_clean_route_accepts_image_ids(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    seen = {}
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None: (seen.update(ids=image_ids)
                                                      or ({'cropped': 1, 'inpainted': 0, 'needs_review': 0,
                                                           'failed': 0, 'skipped': 0}, None)))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'image_ids': [7, 8]})
    assert resp.status_code == 200 and resp.get_json()['cropped'] == 1
    assert seen['ids'] == [7, 8]
    # a non-list image_ids is rejected
    assert client.post(f'/api/dataset/{ds_id}/watermarks/clean',
                       json={'image_ids': 'nope'}).status_code == 400


def test_detect_route_forwards_include_dismissed(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    seen = {}
    monkeypatch.setattr(svc, 'detect_watermarks',
                        lambda u, d, include_dismissed=False: (seen.update(inc=include_dismissed)
                                                               or {'detected': 0, 'none': 0, 'checked': 0}))
    client.post(f'/api/dataset/{ds_id}/watermarks/detect', json={'include_dismissed': True})
    assert seen.get('inc') is True


# --- LaMa service contract (subprocess mocked) -----------------------------

class _Proc:
    def __init__(self, stdout, returncode=0, stderr=''):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_inpaint_watermark_unavailable_never_shells_out(app, monkeypatch):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: False)

    def _boom(*a, **k):
        raise AssertionError('subprocess must not run when unavailable')

    monkeypatch.setattr('app.services.watermark_lama.subprocess.run', _boom)
    with app.app_context():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.webp')
            with open(p, 'wb') as fh:
                fh.write(_img_bytes())
            ok, err = watermark_lama.inpaint_watermark(p, [0.1, 0.1, 0.2, 0.2])
            assert ok is False and err['kind'] == 'unavailable'


def test_inpaint_watermark_parses_ok_line(app, monkeypatch):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr('app.services.watermark_lama.subprocess.run',
                        lambda *a, **k: _Proc('[lama] inpaint\n' + json.dumps({'ok': True})))
    with app.app_context():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.webp')
            with open(p, 'wb') as fh:
                fh.write(_img_bytes())
            ok, err = watermark_lama.inpaint_watermark(p, [0.1, 0.1, 0.2, 0.2])
            assert ok is True and err is None


def test_inpaint_watermark_crash_reports_stderr_tail(app, monkeypatch):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr('app.services.watermark_lama.subprocess.run',
                        lambda *a, **k: _Proc('', stderr='Traceback\nValueError: bad mask', returncode=1))
    with app.app_context():
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.webp')
            with open(p, 'wb') as fh:
                fh.write(_img_bytes())
            ok, err = watermark_lama.inpaint_watermark(p, [0.1, 0.1, 0.2, 0.2])
            assert ok is False and err['kind'] == 'failed' and err['detail'] == 'ValueError: bad mask'
