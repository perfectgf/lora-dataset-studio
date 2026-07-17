"""Watermark auto-correction V1: bbox parsing, crop/LaMa/review routing, the detect
& clean batches, the routes (LaMa mocked present AND absent), the tuple error contract,
original preservation, and the additive columns."""
import io
import json
import os
import sqlite3
from contextlib import contextmanager

import pytest
from PIL import Image


def _img_bytes(color=(200, 30, 30), size=(64, 64), fmt='WEBP'):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, fmt)
    return buf.getvalue()


def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger})


def _kept_image(svc, ds_id, filename, *, size=(1024, 1024), state='detected', bbox=None,
                regions=None):
    """A kept FaceDatasetImage backed by a REAL file on disk (routing/clean open it)."""
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'wb') as fh:
        fh.write(_img_bytes(size=size))
    img = FaceDatasetImage(dataset_id=ds_id, source='import', status='keep',
                           filename=filename, framing='body',
                           watermark_state=state,
                           watermark_bbox=json.dumps(bbox) if bbox is not None else None,
                           watermark_regions=json.dumps(regions) if regions is not None else None)
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


def test_completed_detection_clears_manual_regions_but_empty_vision_preserves_them(
        app, monkeypatch):
    from app.services import face_dataset_service as svc
    import app.services.vision_ollama as vo
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    raws = [
        '{"present":true,"x1":10,"y1":10,"x2":20,"y2":20}',
        '{"present":false}',
        '',
    ]
    monkeypatch.setattr(vo, 'describe_image_ollama', lambda *a, **k: raws.pop(0))
    monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Detection lifecycle', 'detect')
        old_regions = [[0.3, 0.3, 0.4, 0.4]]
        detected = _kept_image(
            svc, ds.id, 'detected.webp', bbox=[0.3, 0.3, 0.4, 0.4], regions=old_regions,
        )
        none = _kept_image(
            svc, ds.id, 'none.webp', bbox=[0.3, 0.3, 0.4, 0.4], regions=old_regions,
        )
        unavailable = _kept_image(
            svc, ds.id, 'unavailable-detect.webp', bbox=[0.3, 0.3, 0.4, 0.4],
            regions=old_regions,
        )

        counts = svc.detect_watermarks(LOCAL_USER, ds.id)

        assert counts == {'detected': 1, 'none': 1, 'checked': 2}
        assert svc.db.session.get(FaceDatasetImage, detected.id).watermark_regions is None
        assert svc.db.session.get(FaceDatasetImage, none.id).watermark_regions is None
        skipped = svc.db.session.get(FaceDatasetImage, unavailable.id)
        assert skipped.watermark_state == 'detected'
        assert json.loads(skipped.watermark_regions) == old_regions


def test_dismiss_clears_manual_regions_but_retains_legacy_bbox(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Dismiss lifecycle', 'dismiss')
        bbox = [0.1, 0.1, 0.2, 0.2]
        img = _kept_image(svc, ds.id, 'dismiss.webp', bbox=bbox, regions=[bbox])

        assert svc.dismiss_watermarks(LOCAL_USER, ds.id, [img.id]) == 1

        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state == 'dismissed'
        assert json.loads(row.watermark_bbox) == bbox
        assert row.watermark_regions is None


@pytest.mark.parametrize('via_batch', [False, True], ids=['single-status', 'batch-action'])
def test_reject_clears_all_watermark_metadata_before_restore(app, via_batch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Reject lifecycle', 'reject')
        bbox = [0.1, 0.1, 0.2, 0.2]
        img = _kept_image(svc, ds.id, 'reject.webp', bbox=bbox, regions=[bbox])

        if via_batch:
            assert svc.batch_image_action(LOCAL_USER, ds.id, [img.id], 'reject') == 1
        else:
            assert svc.set_image_status(LOCAL_USER, img.id, 'reject') is True

        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.status == 'reject'
        assert (row.watermark_state, row.watermark_bbox, row.watermark_regions) == (
            None, None, None,
        )
        assert svc.set_image_status(LOCAL_USER, img.id, 'keep') is True
        restored = svc.db.session.get(FaceDatasetImage, img.id)
        assert (restored.watermark_state, restored.watermark_bbox,
                restored.watermark_regions) == (None, None, None)


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


def test_clean_batches_multiple_lama_images_in_one_worker(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    calls = []
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)

    def _batch(jobs, *, device, timeout=900):
        calls.append((jobs, device))
        return {job['image_path']: (True, None) for job in jobs}

    monkeypatch.setattr(watermark_lama, 'inpaint_batch', _batch)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Batch', 'batch')
        _kept_image(svc, ds.id, 'a.webp', bbox=[0.7, 0.7, 0.8, 0.8])
        _kept_image(svc, ds.id, 'b.webp', bbox=[0.6, 0.6, 0.7, 0.7])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, device='cuda')
    assert err is None and counts['inpainted'] == 2
    assert len(calls) == 1 and calls[0][1] == 'cuda' and len(calls[0][0]) == 2


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


def test_clean_manual_regions_force_one_composite_inpaint_at_edge_and_clear_on_success(
        app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    calls = []
    regions = [[0.0, 0.0, 0.2, 0.05], [0.75, 0.8, 0.95, 0.9]]
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(
        watermark_lama, 'inpaint_watermarks',
        lambda path, bboxes, timeout=300: (calls.append((path, bboxes)) or (True, None)),
    )
    monkeypatch.setattr(
        watermark_lama, 'inpaint_watermark',
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError('manual regions must use the composite LaMa entry point')),
    )

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Manual clean', 'manual')
        legacy_bbox = [0.0, 0.0, 1.0, 0.05]  # would take the legacy crop route
        img = _kept_image(svc, ds.id, 'manual.webp', bbox=legacy_bbox, regions=regions)
        path = svc._img_path(img)
        before = open(path, 'rb').read()

        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, image_ids=[img.id])

        assert err is None
        assert counts == {
            'cropped': 0, 'inpainted': 1, 'inpainted_klein': 0, 'needs_review': 0,
            'failed': 0, 'skipped': 0,
        }
        assert calls == [(path, regions)]
        stem, ext = os.path.splitext(path)
        assert open(f'{stem}.orig{ext}', 'rb').read() == before
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state == 'cleaned'
        assert json.loads(row.watermark_bbox) == legacy_bbox
        assert row.watermark_regions is None


def test_clean_empty_manual_override_needs_review_without_touching_pixels(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(
        watermark_lama, 'inpaint_watermarks',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('empty override must not inpaint')),
    )
    monkeypatch.setattr(
        watermark_lama, 'inpaint_watermark',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('empty override must not inpaint')),
    )

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Empty manual clean', 'empty')
        img = _kept_image(
            svc, ds.id, 'empty.webp', bbox=[0.0, 0.0, 1.0, 0.05], regions=[],
        )
        path = svc._img_path(img)
        before = open(path, 'rb').read()

        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, image_ids=[img.id])

        assert err is None
        assert counts == {
            'cropped': 0, 'inpainted': 0, 'inpainted_klein': 0, 'needs_review': 1,
            'failed': 0, 'skipped': 0,
        }
        assert open(path, 'rb').read() == before
        stem, ext = os.path.splitext(path)
        assert not os.path.exists(f'{stem}.orig{ext}')
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state == 'detected'
        assert row.watermark_regions == '[]'


def test_clean_manual_regions_unavailable_preserves_retry_metadata(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    regions = [[0.1, 0.1, 0.2, 0.2]]
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: False)
    monkeypatch.setattr(
        watermark_lama, 'inpaint_watermarks',
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError('unavailable LaMa must not be invoked')),
    )

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Unavailable manual clean', 'unavailable')
        img = _kept_image(svc, ds.id, 'unavailable.webp', bbox=[0.1, 0.1, 0.2, 0.2],
                          regions=regions)
        path = svc._img_path(img)
        before = open(path, 'rb').read()

        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, image_ids=[img.id])

        assert err is None and counts['skipped'] == 1 and counts['inpainted'] == 0
        assert open(path, 'rb').read() == before
        stem, ext = os.path.splitext(path)
        assert not os.path.exists(f'{stem}.orig{ext}')
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state == 'detected'
        assert json.loads(row.watermark_regions) == regions


def test_clean_manual_regions_failure_preserves_retry_metadata(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    regions = [[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.8, 0.8]]
    failure = {'kind': 'failed', 'detail': 'RuntimeError: composite boom'}
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(
        watermark_lama, 'inpaint_watermarks',
        lambda path, bboxes, timeout=300: (False, failure),
    )

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Failed manual clean', 'failed')
        img = _kept_image(svc, ds.id, 'failed.webp', bbox=[0.1, 0.1, 0.2, 0.2],
                          regions=regions)
        path = svc._img_path(img)
        before = open(path, 'rb').read()

        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, image_ids=[img.id])

        assert counts['failed'] == 1 and counts['inpainted'] == 0
        assert err == failure
        assert open(path, 'rb').read() == before
        stem, ext = os.path.splitext(path)
        assert open(f'{stem}.orig{ext}', 'rb').read() == before
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state == 'detected'
        assert json.loads(row.watermark_regions) == regions


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
                        lambda u, d, image_ids=None, device=None, method=None: ({'cropped': 2, 'inpainted': 1, 'needs_review': 0,
                                       'failed': 0, 'skipped': 0}, None))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] and body['cropped'] == 2 and body['inpainted'] == 1 and body['error'] is None


def test_clean_route_lama_absent_reports_skipped(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None, device=None, method=None: ({'cropped': 1, 'inpainted': 0, 'needs_review': 0,
                                       'failed': 0, 'skipped': 3}, None))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    body = resp.get_json()
    assert body['ok'] and body['skipped'] == 3 and body['cropped'] == 1


def test_clean_route_surfaces_error(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None, device=None, method=None: ({'cropped': 0, 'inpainted': 0, 'needs_review': 0,
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
                        lambda u, d, image_ids=None, device=None, method=None: (seen.update(ids=image_ids)
                                                      or ({'cropped': 1, 'inpainted': 0, 'needs_review': 0,
                                                           'failed': 0, 'skipped': 0}, None)))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'image_ids': [7, 8]})
    assert resp.status_code == 200 and resp.get_json()['cropped'] == 1
    assert seen['ids'] == [7, 8]
    # a non-list image_ids is rejected
    assert client.post(f'/api/dataset/{ds_id}/watermarks/clean',
                       json={'image_ids': 'nope'}).status_code == 400


def test_clean_route_uses_gpu_window_only_for_cuda(client, app, monkeypatch):
    from app.routes import datasets as routes
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    ds_id = _create(client, 'R', 'r').get_json()['id']
    entered = []

    @contextmanager
    def _window(**kwargs):
        entered.append(kwargs)
        yield

    monkeypatch.setattr(routes, 'gpu_exclusive_vision_window', _window)
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None, device=None, method=None: (
                            {'cropped': 0, 'inpainted': 0, 'needs_review': 0,
                             'failed': 0, 'skipped': 0}, None))
    monkeypatch.setattr(watermark_lama, 'resolve_device', lambda requested=None: 'cpu')
    assert client.post(f'/api/dataset/{ds_id}/watermarks/clean').status_code == 200
    assert entered == []
    monkeypatch.setattr(watermark_lama, 'resolve_device', lambda requested=None: 'cuda')
    assert client.post(f'/api/dataset/{ds_id}/watermarks/clean').status_code == 200
    assert len(entered) == 1


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


def test_inpaint_watermarks_sends_one_composite_payload(app, monkeypatch, tmp_path):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    calls = []

    def _run(*args, **kwargs):
        calls.append((args, kwargs))
        return _Proc(json.dumps({'ok': True}))

    monkeypatch.setattr(watermark_lama.subprocess, 'run', _run)
    image_path = tmp_path / 'x.webp'
    image_path.write_bytes(_img_bytes())
    bboxes = [[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.9, 0.9]]

    ok, err = watermark_lama.inpaint_watermarks(image_path, bboxes)

    assert ok is True and err is None
    assert len(calls) == 1
    assert json.loads(calls[0][1]['input']) == {
        'image_path': str(image_path),
        'bboxes': bboxes,
        'device': 'cpu',
    }


def test_resolve_watermark_device_auto_and_explicit_cuda(app, monkeypatch):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, '_cuda_available', lambda: True)
    assert watermark_lama.resolve_device('auto') == 'cuda'
    assert watermark_lama.resolve_device('cuda') == 'cuda'
    assert watermark_lama.resolve_device('cpu') == 'cpu'

    monkeypatch.setattr(watermark_lama, '_cuda_available', lambda: False)
    assert watermark_lama.resolve_device('auto') == 'cpu'
    with pytest.raises(RuntimeError, match='CUDA.*not available'):
        watermark_lama.resolve_device('cuda')


def test_inpaint_batch_uses_one_worker_and_propagates_device(app, monkeypatch, tmp_path):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    first = tmp_path / 'a.webp'
    second = tmp_path / 'b.webp'
    first.write_bytes(_img_bytes())
    second.write_bytes(_img_bytes())
    seen = []

    def _run(*args, **kwargs):
        seen.append(json.loads(kwargs['input']))
        return _Proc(json.dumps({'ok': True, 'results': [
            {'image_path': str(first), 'ok': True},
            {'image_path': str(second), 'ok': True},
        ]}))

    monkeypatch.setattr(watermark_lama.subprocess, 'run', _run)
    results = watermark_lama.inpaint_batch([
        {'image_path': str(first), 'bboxes': [[0.1, 0.1, 0.2, 0.2]]},
        {'image_path': str(second), 'bboxes': [[0.3, 0.3, 0.4, 0.4]]},
    ], device='cuda')

    assert len(seen) == 1
    assert seen[0]['device'] == 'cuda' and len(seen[0]['jobs']) == 2
    assert results[str(first)] == (True, None)
    assert results[str(second)] == (True, None)


def test_inpaint_watermark_legacy_wrapper_sends_one_item_list(app, monkeypatch, tmp_path):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    payloads = []

    def _run(*args, **kwargs):
        payloads.append(json.loads(kwargs['input']))
        return _Proc(json.dumps({'ok': True}))

    monkeypatch.setattr(watermark_lama.subprocess, 'run', _run)
    image_path = tmp_path / 'x.webp'
    image_path.write_bytes(_img_bytes())
    bbox = [0.1, 0.1, 0.2, 0.2]

    ok, err = watermark_lama.inpaint_watermark(str(image_path), bbox)

    assert ok is True and err is None
    assert len(payloads) == 1
    assert payloads[0]['bboxes'] == [bbox]


@pytest.mark.parametrize(('bbox', 'expected'), [
    ([0.8, 0.9, 0.2, 0.1], [0.2, 0.1, 0.8, 0.9]),
    ([-0.1, 0.1, 0.2, 1.2], [0.0, 0.1, 0.2, 1.0]),
], ids=['swapped', 'partially-out-of-range'])
def test_inpaint_worker_legacy_bbox_normalizes_coordinates(bbox, expected):
    from infer.lama_infer import _payload_bboxes

    assert _payload_bboxes({'bbox': bbox}) == [expected]


@pytest.mark.parametrize('value', [
    float('inf'),
    float('-inf'),
    float('nan'),
], ids=['positive-infinity', 'negative-infinity', 'nan'])
def test_inpaint_worker_legacy_bbox_rejects_non_finite_values(value):
    from infer.lama_infer import _payload_bboxes

    with pytest.raises(ValueError, match='bbox values must be finite'):
        _payload_bboxes({'bbox': [0.1, 0.1, value, 0.9]})


@pytest.mark.parametrize(('bbox', 'expected'), [
    ([0.8, 0.9, 0.2, 0.1], [0.2, 0.1, 0.8, 0.9]),
    ([-0.1, 0.1, 0.2, 1.2], [0.0, 0.1, 0.2, 1.0]),
], ids=['swapped', 'partially-out-of-range'])
def test_inpaint_watermark_legacy_wrapper_normalizes_coordinates(
        app, monkeypatch, tmp_path, bbox, expected):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    payloads = []

    def _run(*args, **kwargs):
        payloads.append(json.loads(kwargs['input']))
        return _Proc(json.dumps({'ok': True}))

    monkeypatch.setattr(watermark_lama.subprocess, 'run', _run)
    image_path = tmp_path / 'x.webp'
    image_path.write_bytes(_img_bytes())

    ok, err = watermark_lama.inpaint_watermark(image_path, bbox)

    assert ok is True and err is None
    assert payloads == [{
        'image_path': str(image_path),
        'bboxes': [expected],
        'device': 'cpu',
    }]


@pytest.mark.parametrize(('bbox', 'detail'), [
    ([0.1, 0.1, float('inf'), 0.9], 'payload: bbox values must be finite'),
    ([0.1, 0.1, float('-inf'), 0.9], 'payload: bbox values must be finite'),
    ([0.1, 0.1, float('nan'), 0.9], 'payload: bbox values must be finite'),
    ([0.1, 0.1, 0.9], 'payload: bbox must have 4 values'),
    ([0.1, 0.1, 'oops', 0.9], "payload: could not convert string to float: 'oops'"),
], ids=['positive-infinity', 'negative-infinity', 'nan', 'wrong-arity', 'nonnumeric'])
def test_inpaint_watermark_invalid_legacy_bbox_returns_structured_failure(
        app, monkeypatch, tmp_path, bbox, detail):
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)

    def _boom(*args, **kwargs):
        raise AssertionError('invalid legacy bbox must not launch the worker')

    monkeypatch.setattr(watermark_lama.subprocess, 'run', _boom)
    image_path = tmp_path / 'x.webp'
    image_path.write_bytes(_img_bytes())

    assert watermark_lama.inpaint_watermark(image_path, bbox) == (
        False,
        {'kind': 'failed', 'detail': detail},
    )


def test_build_mask_marks_each_box_without_filling_space_between():
    from infer.lama_infer import build_mask

    mask = build_mask((100, 50), [
        [0.1, 0.2, 0.2, 0.4],
        [0.8, 0.6, 0.9, 0.8],
    ])

    assert mask.getpixel((15, 15)) == 255
    assert mask.getpixel((85, 35)) == 255
    assert mask.getpixel((50, 25)) == 0


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


# --- Klein inpaint (V2): crop geometry, byte-exact composite, method routing ----

def test_klein_crop_box_is_square_and_contains_the_mark():
    from app.services import watermark_klein as wk
    box = wk._klein_crop_box(1000, 1000, [[0.45, 0.45, 0.55, 0.55]])
    l, t, r, b = box
    assert r - l == b - t                          # square
    # contains the mark (450..550 px on both axes)
    assert l <= 450 and t <= 450 and r >= 550 and b >= 550
    assert 0 <= l and 0 <= t and r <= 1000 and b <= 1000


def test_klein_crop_box_slides_in_bounds_near_a_corner():
    from app.services import watermark_klein as wk
    box = wk._klein_crop_box(1000, 1000, [[0.0, 0.0, 0.1, 0.1]])
    l, t, r, b = box
    assert (l, t) == (0, 0)                         # slid flush to the corner
    assert r - l == b - t                           # still square
    assert r >= 100 and b >= 100                     # still contains the mark


def test_klein_crop_box_unions_multiple_regions():
    from app.services import watermark_klein as wk
    box = wk._klein_crop_box(1000, 1000, [[0.1, 0.1, 0.15, 0.15], [0.8, 0.8, 0.85, 0.85]])
    l, t, r, b = box
    # the crop must span both marks (100..850 px)
    assert l <= 100 and t <= 100 and r >= 850 and b >= 850


def test_composite_preserves_bytes_outside_the_mask_and_changes_them_inside():
    """THE preservation guarantee, verified byte-for-byte: every pixel where the
    composite mask is 0 keeps its ORIGINAL bytes; pixels under the mask change."""
    import numpy as np
    from PIL import ImageDraw, ImageFilter
    from app.services import watermark_klein as wk

    rng = np.random.default_rng(1234)
    W, H = 200, 200
    original = Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype='uint8'), 'RGB')
    crop_box = (50, 50, 150, 150)                  # 100x100 crop region
    filled = Image.new('RGB', (100, 100), (255, 0, 0))
    # A white rectangle in the crop's centre, feathered — the paste footprint.
    hard = Image.new('L', (100, 100), 0)
    ImageDraw.Draw(hard).rectangle([30, 30, 69, 69], fill=255)
    composite_mask = hard.filter(ImageFilter.GaussianBlur(6))

    result = wk.composite_inpaint(original, filled, crop_box, composite_mask)

    mask_full = np.zeros((H, W), dtype='uint8')
    mask_full[50:150, 50:150] = np.array(composite_mask)
    res = np.array(result)
    orig = np.array(original)
    # Byte-for-byte identical everywhere the mask is exactly 0.
    assert np.array_equal(res[mask_full == 0], orig[mask_full == 0])
    # And the fully-masked centre really changed (it became red).
    assert np.array_equal(res[mask_full == 255], np.broadcast_to(
        np.array([255, 0, 0], dtype='uint8'), res[mask_full == 255].shape))
    # A far corner (well outside the feather) is untouched.
    assert tuple(res[0, 0]) == tuple(orig[0, 0])


# --- Seam harmonization (kill Klein's tonal drift → no visible square) -----------

def test_seam_ring_excludes_other_zones_watermark_pixels():
    """A neighbouring mark's still-watermarked pixels must never enter this zone's ring —
    excluding the UNION of all masks keeps the sample pure original ground truth."""
    import numpy as np
    from PIL import ImageDraw
    from app.services import watermark_klein as wk
    W, H = 200, 200
    zone = Image.new('L', (W, H), 0)
    ImageDraw.Draw(zone).rectangle([40, 40, 60, 60], fill=255)
    other = Image.new('L', (W, H), 0)
    ImageDraw.Draw(other).rectangle([72, 40, 92, 60], fill=255)   # a near neighbour mark
    union = Image.new('L', (W, H), 0)
    ImageDraw.Draw(union).rectangle([40, 40, 60, 60], fill=255)
    ImageDraw.Draw(union).rectangle([72, 40, 92, 60], fill=255)

    ring_alone = wk._seam_ring(zone, zone)
    ring_excl = wk._seam_ring(zone, union)
    other_px = np.asarray(other) > 127
    # excluding the union keeps the neighbour's masked pixels out of the sample...
    assert not (ring_excl & other_px).any()
    # ...and only ever REMOVES ring pixels (the outside-band is otherwise the same).
    assert ring_excl.sum() < ring_alone.sum()
    assert not (ring_excl & (np.asarray(zone) > 127)).any()       # never samples the zone itself


def test_harmonize_seam_is_identity_when_no_drift():
    """Klein didn't drift (fill == original) → the correction is a byte-for-byte no-op."""
    import numpy as np
    from PIL import ImageDraw
    from app.services import watermark_klein as wk
    rng = np.random.default_rng(7)
    W, H = 128, 128
    crop = Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype='uint8'), 'RGB')
    zone = Image.new('L', (W, H), 0)
    ImageDraw.Draw(zone).rectangle([48, 48, 79, 79], fill=255)
    ring = wk._seam_ring(zone, zone)

    out = wk._harmonize_seam(crop, crop, ring)
    assert np.array_equal(np.asarray(out), np.asarray(crop))


def test_harmonize_seam_removes_a_flat_tonal_offset():
    """A uniform per-channel drift on a flat crop is fully removed — the corrected patch
    (mask included) is re-seated to the original's tone. Exercises the std≈0 gain guard."""
    import numpy as np
    from PIL import ImageDraw
    from app.services import watermark_klein as wk
    W, H = 100, 100
    original = Image.new('RGB', (W, H), (100, 110, 120))
    filled = Image.new('RGB', (W, H), (120, 130, 155))       # +20 / +20 / +35 tonal drift
    zone = Image.new('L', (W, H), 0)
    ImageDraw.Draw(zone).rectangle([40, 40, 59, 59], fill=255)
    ring = wk._seam_ring(zone, zone)

    out = np.asarray(wk._harmonize_seam(filled, original, ring))
    assert np.array_equal(out, np.broadcast_to(np.array([100, 110, 120], 'uint8'), out.shape))


def test_harmonize_seam_identity_when_ring_too_small():
    """Mark glued to the crop edges (no ground-truth ring) → fill returned UNCHANGED, never
    re-toned off a noisy estimate."""
    import numpy as np
    from app.services import watermark_klein as wk
    W, H = 80, 80
    filled = Image.new('RGB', (W, H), (30, 60, 90))
    original = Image.new('RGB', (W, H), (200, 200, 200))
    full = Image.new('L', (W, H), 255)                       # the whole crop is masked
    ring = wk._seam_ring(full, full)

    assert int(ring.sum()) < wk.KLEIN_HARMONIZE_MIN_RING
    out = wk._harmonize_seam(filled, original, ring)
    assert np.array_equal(np.asarray(out), np.asarray(filled))


def test_harmonize_then_composite_preserves_outside_bytes_and_changes_inside():
    """The full service wiring (per-zone harmonize → chained feathered composite) keeps every
    pixel outside the paste footprint byte-exact, while the masked centre really changes."""
    import numpy as np
    from PIL import ImageFilter
    from app.services import watermark_klein as wk
    rng = np.random.default_rng(99)
    W, H = 160, 160
    original = Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype='uint8'), 'RGB')
    crop_box = (30, 30, 130, 130)                            # 100x100 crop
    original_crop = original.crop(crop_box)
    filled = Image.fromarray(rng.integers(0, 256, (100, 100, 3), dtype='uint8'), 'RGB')
    boxes = [[0.40, 0.40, 0.50, 0.50]]                       # one small mark inside the crop

    union = wk._hard_mask(crop_box, W, H, boxes)
    result = original.convert('RGB').copy()
    for box in boxes:
        zone = wk._hard_mask(crop_box, W, H, [box])
        ring = wk._seam_ring(zone, union)
        corrected = wk._harmonize_seam(filled, original_crop, ring)
        zm = zone.filter(ImageFilter.GaussianBlur(wk.KLEIN_COMPOSITE_FEATHER_PX))
        result = wk.composite_inpaint(result, corrected, crop_box, zm)

    footprint = np.zeros((H, W), dtype='uint8')
    zm = union.filter(ImageFilter.GaussianBlur(wk.KLEIN_COMPOSITE_FEATHER_PX))
    footprint[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = np.asarray(zm)
    res, orig = np.asarray(result), np.asarray(original)
    assert np.array_equal(res[footprint == 0], orig[footprint == 0])   # outside → byte-exact
    assert not np.array_equal(res[footprint == 255], orig[footprint == 255])  # inside changed


def test_clean_inpaint_engine_mapping():
    from app.services import face_dataset_service as svc
    assert svc._clean_inpaint_engine('lama', 'auto') == 'lama'
    assert svc._clean_inpaint_engine('lama', 'lama') == 'lama'
    assert svc._clean_inpaint_engine('lama', 'klein') == 'klein'
    assert svc._clean_inpaint_engine('review', 'auto') == 'review'
    assert svc._clean_inpaint_engine('review', 'klein') == 'klein'   # review → actionable


def _stub_klein(monkeypatch, *, available=True, result=(True, None), recorder=None):
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: available)

    def _fake(user_id, path, boxes, **kwargs):
        if recorder is not None:
            recorder.append({'path': path, 'boxes': boxes})
        return result

    monkeypatch.setattr(wk, 'inpaint_watermark_klein', _fake)


def test_clean_klein_inpaints_the_lama_route_and_preserves_original(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    calls = []
    _stub_klein(monkeypatch, recorder=calls)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'k')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.35, 0.35, 0.45, 0.45])
        path = svc._img_path(img)
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, method='klein')
        assert err is None and counts['inpainted_klein'] == 1 and counts['inpainted'] == 0
        assert calls == [{'path': path, 'boxes': [[0.35, 0.35, 0.45, 0.45]]}]
        stem, ext = os.path.splitext(path)
        assert os.path.exists(f'{stem}.orig{ext}')          # original preserved
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'cleaned'


def test_clean_klein_makes_the_review_route_actionable(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    calls = []
    _stub_klein(monkeypatch, recorder=calls)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'k')
        # centre-overlapping mark → 'review' under LaMa, now cleaned under Klein
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.45, 0.45, 0.55, 0.55])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, method='klein')
        assert err is None and counts['inpainted_klein'] == 1 and counts['needs_review'] == 0
        assert calls == [{'path': svc._img_path(img), 'boxes': [[0.45, 0.45, 0.55, 0.55]]}]
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'cleaned'


def test_clean_klein_uses_manual_regions_and_clears_them(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    regions = [[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.8, 0.8]]
    calls = []
    _stub_klein(monkeypatch, recorder=calls)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'k')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.0, 0.0, 1.0, 0.05], regions=regions)
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, image_ids=[img.id], method='klein')
        assert err is None and counts['inpainted_klein'] == 1
        assert calls == [{'path': svc._img_path(img), 'boxes': regions}]
        row = svc.db.session.get(FaceDatasetImage, img.id)
        assert row.watermark_state == 'cleaned' and row.watermark_regions is None


def test_clean_klein_unavailable_skips_and_leaves_detected(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    calls = []
    _stub_klein(monkeypatch, available=False, recorder=calls)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'k')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.35, 0.35, 0.45, 0.45])
        path = svc._img_path(img)
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, method='klein')
        assert err is None and counts['skipped'] == 1 and counts['inpainted_klein'] == 0
        assert calls == []                                  # never attempted
        stem, ext = os.path.splitext(path)
        assert not os.path.exists(f'{stem}.orig{ext}')      # nothing preserved
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'detected'


def test_clean_klein_failure_surfaces_error_and_fails_the_row(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    failure = {'kind': 'failed', 'detail': 'ComfyUI KSampler: boom'}
    _stub_klein(monkeypatch, result=(False, failure))
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'K', 'k')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.35, 0.35, 0.45, 0.45])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id, method='klein')
        assert counts['failed'] == 1 and counts['inpainted_klein'] == 0
        assert err == failure
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'failed'


def test_clean_auto_method_never_calls_klein(app, monkeypatch):
    """Regression: the default method still routes through LaMa, untouched."""
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama, watermark_klein
    from app.config import LOCAL_USER
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermark', lambda *a, **k: (True, None))
    monkeypatch.setattr(watermark_klein, 'inpaint_watermark_klein',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('auto method must not call Klein')))
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'A', 'a')
        img = _kept_image(svc, ds.id, 'wm.webp', bbox=[0.35, 0.35, 0.45, 0.45])
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)   # method defaults to auto
        assert err is None and counts['inpainted'] == 1 and counts['inpainted_klein'] == 0


def test_klein_clean_disk_equals_display_equals_export_and_keeps_dimensions(
        client, app, monkeypatch):
    """LOCK the 'display ≠ download' grief: after a real Klein multi-zone clean the dataset
    file must keep the ORIGINAL dimensions (never the ~1MP upscaled crop), and the display
    URL, the disk file and the export ZIP must all carry the SAME full-frame composite — no
    crop / full-render leaks into any serving path."""
    import io
    import zipfile
    import numpy as np
    from app.routes import datasets as routes
    from app.services import face_dataset_service as svc
    from app.services import watermark_klein as wk
    from app.models import FaceDatasetImage

    # Deterministic Klein round-trip: prefill passes the crop through; the "render" brightens
    # the WHOLE crop by +60 — so a composite bypass would show up as a +60 shift over the
    # entire frame (and, when the crop clamps to the image, a wrong upscaled size).
    monkeypatch.setattr(routes, '_klein_clean_preflight', lambda: None)   # skip node/model 409
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    monkeypatch.setattr(wk, '_prefill_region', lambda scaled, boxes, device='cpu': (scaled, None))

    def _fake_klein(user_id, crop_img, *, seed, timeout=None):
        arr = np.asarray(crop_img.convert('RGB')).astype('int16') + 60
        return Image.fromarray(np.clip(arr, 0, 255).astype('uint8'), 'RGB'), None
    monkeypatch.setattr(wk, '_run_klein_job', _fake_klein)

    ds_id = _create(client, 'K', 'k').get_json()['id']
    with app.app_context():
        # a NON-square frame so any crop/reframe changes the aspect ratio, and two zones
        img = _kept_image(svc, ds_id, 'wm.webp', size=(900, 640),
                          bbox=[0.1, 0.1, 0.2, 0.2],
                          regions=[[0.10, 0.10, 0.22, 0.22], [0.60, 0.55, 0.74, 0.68]])
        img_id = img.id
        path = svc._img_path(img)
        orig = np.asarray(Image.open(path).convert('RGB'))

    r = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'method': 'klein'})
    assert r.status_code == 200 and r.get_json()['inpainted_klein'] == 1

    with app.app_context():
        img = svc.db.session.get(FaceDatasetImage, img_id)
        path = svc._img_path(img)
        filename = img.filename
        disk_bytes = open(path, 'rb').read()
        cleaned = np.asarray(Image.open(io.BytesIO(disk_bytes)).convert('RGB'))

    # (1) dimensions preserved — NOT the upscaled Klein crop
    assert cleaned.shape == orig.shape == (640, 900, 3)
    # (2) the composite was NOT bypassed: a full-render leak would shift the WHOLE frame by
    #     ~+60; only the two small zones may move, so the global mean barely changes.
    assert abs(float(cleaned.mean()) - float(orig.mean())) < 10
    # (3) the display URL serves EXACTLY the on-disk bytes
    disp = client.get(f'/api/dataset/{ds_id}/img/{filename}')
    assert disp.status_code == 200 and disp.data == disk_bytes
    # (4) the export ZIP carries the SAME pixels (PNG re-encode of the same file)
    exp = client.get(f'/api/dataset/{ds_id}/export')
    assert exp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(exp.data))
    png = next(n for n in zf.namelist() if n.endswith('.png'))
    zpix = np.asarray(Image.open(io.BytesIO(zf.read(png))).convert('RGB'))
    assert zpix.shape == orig.shape and np.array_equal(zpix, cleaned)

    # (5) restore brings back the exact original bytes AND dimensions
    rr = client.post(f'/api/dataset/{ds_id}/image/{img_id}/watermark-restore', json={})
    assert rr.status_code == 200
    with app.app_context():
        restored = np.asarray(Image.open(svc._img_path(
            svc.db.session.get(FaceDatasetImage, img_id))).convert('RGB'))
    assert restored.shape == orig.shape and np.array_equal(restored, orig)


# --- Klein clean route (method, preflight, 409/503) ------------------------------

def test_clean_route_forwards_method_klein(client, app, monkeypatch):
    from app.routes import datasets as routes
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(routes, '_klein_clean_preflight', lambda: None)
    seen = {}
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d, image_ids=None, method='auto': (
                            seen.update(method=method)
                            or ({'cropped': 0, 'inpainted': 0, 'inpainted_klein': 2,
                                 'needs_review': 0, 'failed': 0, 'skipped': 0}, None)))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'method': 'klein'})
    assert resp.status_code == 200 and resp.get_json()['inpainted_klein'] == 2
    assert seen['method'] == 'klein'


def test_clean_route_rejects_unknown_method(client):
    ds_id = _create(client, 'R', 'r').get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'method': 'nope'})
    assert resp.status_code == 400


def test_clean_route_klein_503_when_training(client, app):
    from app.job_queue import queue_manager
    ds_id = _create(client, 'R', 'r').get_json()['id']
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=300)
    try:
        resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'method': 'klein'})
        assert resp.status_code == 503 and 'GPU busy' in resp.get_json()['error']
    finally:
        with app.app_context():
            queue_manager._set_system_state('training_in_progress', None)


def test_clean_route_klein_409_when_models_missing(client, app, monkeypatch):
    from app.routes import datasets as routes
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    ds_id = _create(client, 'R', 'r').get_json()['id']
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', None)
        queue_manager._set_system_state('vision_in_progress', None)
    from app import capabilities as caps_mod
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: ['klein_model'])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    # a valid ComfyUI base so we reach the real "missing model" 409 (not the
    # "configure ComfyUI first" short-circuit), and neutralise the auto-download.
    monkeypatch.setattr(caps_mod, 'resolve_comfyui_base',
                        lambda p: {'valid': True, 'resolved': p, 'nested': False})
    monkeypatch.setattr(routes, '_autostart_klein_downloads', lambda missing: ([], False))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean', json={'method': 'klein'})
    assert resp.status_code == 409
    assert resp.get_json()['klein_missing'] == ['klein_model']


# --- Klein ComfyUI round-trip wiring (queue + output mocked) ---------------------

def test_run_klein_job_wires_workflow_and_returns_filled(app, monkeypatch, tmp_path):
    from app.services import watermark_klein as wk
    from app.services import klein_edit_helper as keh

    monkeypatch.setattr(wk, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(wk, '_comfy_output_dir', lambda: None)
    monkeypatch.setattr(keh, 'resolve_klein_unet', lambda selected=None: 'klein\\unet.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_vae', lambda: 'flux2-vae.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_text_encoder', lambda: 'qwen_3_8b_fp8mixed.safetensors')
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    captured = {}
    monkeypatch.setattr(wk.queue_manager, 'add_job',
                        lambda **kw: captured.update(kw) or kw['job_id'])
    monkeypatch.setattr(wk, '_wait_for_job',
                        lambda job_id, timeout: ('completed', 'wmklein_out.png', None))
    monkeypatch.setattr(wk, '_read_comfy_output', lambda filename: _img_bytes(size=(64, 64)))

    with app.app_context():
        crop = Image.new('RGB', (64, 64), (10, 20, 30))   # already pre-filled by the caller
        filled, err = wk._run_klein_job('local', crop, seed=7)

    assert err is None and filled is not None and filled.size == (64, 64)
    wf = captured['workflow_data']
    assert captured['metadata'] == {'model_name': 'watermark_klein'}
    assert wf['114']['inputs']['unet_name'] == 'klein\\unet.safetensors'
    assert wf['10']['inputs']['vae_name'] == 'flux2-vae.safetensors'
    assert wf['90']['inputs']['clip_name'] == 'qwen_3_8b_fp8mixed.safetensors'
    assert wf['77']['inputs']['seed'] == 7
    assert wf['77']['inputs']['denoise'] == wk.KLEIN_DENOISE
    assert wf['77']['inputs']['steps'] == wk.KLEIN_STEPS
    # full-edit: the KSampler samples the crop's own latent, no SetLatentNoiseMask node
    assert wf['77']['inputs']['latent_image'] == ['53', 0]
    assert 'setmask' not in wf and 'mask' not in wf
    assert wf['6']['inputs']['text'] == wk.KLEIN_INPAINT_PROMPT
    # the input crop was cleaned up after the run
    assert not list(tmp_path.glob('wmklein_*'))


def test_run_klein_job_raises_when_required_asset_missing(app, monkeypatch, tmp_path):
    from app.services import watermark_klein as wk
    from app.services import klein_edit_helper as keh
    monkeypatch.setattr(wk, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(keh, 'resolve_klein_unet', lambda selected=None: None)
    monkeypatch.setattr(keh, 'resolve_klein_vae', lambda: None)
    monkeypatch.setattr(keh, 'resolve_klein_text_encoder', lambda: None)
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: ['klein_model', 'klein_vae'])
    monkeypatch.setattr(wk.queue_manager, 'add_job',
                        lambda **kw: (_ for _ in ()).throw(AssertionError('must not enqueue')))
    with app.app_context():
        with pytest.raises(keh.KleinModelsMissing):
            wk._run_klein_job('local', Image.new('RGB', (32, 32)), seed=1)


# --- Prefill (LaMa worker, cv2 TELEA fallback, abort when neither) ---------------

def test_crop_boxes_norm_translates_into_crop_space():
    from app.services import watermark_klein as wk
    # crop is the top-right quadrant of a 1000x1000 image; a mark fully inside it.
    crop_box = (500, 0, 1000, 500)                      # cw = ch = 500
    boxes = wk._crop_boxes_norm(crop_box, 1000, 1000, [[0.6, 0.1, 0.7, 0.2]], expand_px=0)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    # x: (600-500)/500=0.2 .. (700-500)/500=0.4 ; y: 100/500=0.2 .. 200/500=0.4
    assert abs(x1 - 0.2) < 1e-9 and abs(x2 - 0.4) < 1e-9
    assert abs(y1 - 0.2) < 1e-9 and abs(y2 - 0.4) < 1e-9


def test_crop_boxes_norm_expand_grows_and_clamps_to_unit_range():
    from app.services import watermark_klein as wk
    # a mark hugging the crop's left/top edge: expand can't push coords below 0.
    crop_box = (0, 0, 500, 500)
    boxes = wk._crop_boxes_norm(crop_box, 500, 500, [[0.0, 0.0, 0.2, 0.2]], expand_px=10)
    x1, y1, x2, y2 = boxes[0]
    assert x1 == 0.0 and y1 == 0.0            # clamped, not negative
    assert x2 == (100 + 10) / 500 and y2 == (100 + 10) / 500   # grown by expand_px


def test_prefill_falls_back_to_telea_when_lama_absent(monkeypatch):
    """LaMa not installed → cv2 TELEA repaints the region (a clean, usable prefill)."""
    from app.services import watermark_klein as wk
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: False)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermarks',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('LaMa must not run when unavailable')))
    crop = Image.new('RGB', (128, 128), (40, 90, 160))
    prefilled, err = wk._prefill_region(crop, [[0.3, 0.3, 0.6, 0.6]])
    assert err is None and prefilled is not None and prefilled.size == (128, 128)


def test_prefill_aborts_cleanly_when_no_engine_available(monkeypatch):
    """Neither LaMa nor cv2 → a clean 'unavailable' abort (no Klein without a prefill)."""
    import sys
    from app.services import watermark_klein as wk
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: False)
    monkeypatch.setitem(sys.modules, 'cv2', None)   # force `import cv2` to raise ImportError
    prefilled, err = wk._prefill_region(Image.new('RGB', (64, 64)), [[0.2, 0.2, 0.5, 0.5]])
    assert prefilled is None
    assert err and err['kind'] == 'unavailable' and 'ML extras' in err['detail']


def test_prefill_uses_lama_worker_when_available(monkeypatch):
    """LaMa installed → the worker is the prefill (cv2 TELEA is only a fallback)."""
    from app.services import watermark_klein as wk
    from app.services import watermark_lama
    seen = {}

    def _fake_inpaint(path, bboxes, **kwargs):
        seen['path'] = path
        seen['bboxes'] = bboxes
        # emulate the worker: overwrite the temp file with a solid repaint
        Image.new('RGB', Image.open(path).size, (0, 0, 0)).save(path, 'WEBP')
        return True, None

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermarks', _fake_inpaint)
    monkeypatch.setattr(wk, '_prefill_telea',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('TELEA must not run when LaMa succeeded')))
    crop = Image.new('RGB', (96, 96), (200, 100, 50))
    prefilled, err = wk._prefill_region(crop, [[0.25, 0.25, 0.5, 0.5]])
    assert err is None and prefilled is not None and prefilled.size == (96, 96)
    assert seen['bboxes'] == [[0.25, 0.25, 0.5, 0.5]]


def test_inpaint_klein_harmonizes_the_drifted_patch_into_the_neighbourhood(monkeypatch, tmp_path):
    """End-to-end with prefill + Klein mocked: the real compositor must harmonize a drifted
    Klein patch to the surrounding tone (no square) instead of pasting the raw drift."""
    import numpy as np
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    # a uniform mid-grey photo: any un-corrected Klein drift would read as a clean square
    img = tmp_path / 'wm.webp'
    Image.fromarray(np.full((512, 512, 3), 120, dtype='uint8'), 'RGB').save(img, 'WEBP', quality=100)

    # prefill: pass the crop through untouched; Klein: brighten the whole crop by +40 (drift)
    monkeypatch.setattr(wk, '_prefill_region', lambda scaled, boxes, device='cpu': (scaled, None))

    def _fake_klein(user_id, crop_img, *, seed, timeout=None):
        arr = np.asarray(crop_img).astype('int16') + 40
        return Image.fromarray(np.clip(arr, 0, 255).astype('uint8'), 'RGB'), None
    monkeypatch.setattr(wk, '_run_klein_job', _fake_klein)

    ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]], seed=1)

    assert ok and err is None
    out = np.asarray(Image.open(img).convert('RGB'))
    centre = out[int(0.45 * 512), int(0.45 * 512), 0]
    assert abs(int(centre) - 120) <= 3        # +40 drift removed → matches the wall, not 160


def test_inpaint_klein_aborts_before_gpu_when_prefill_unavailable(app, monkeypatch, tmp_path):
    """A failed prefill must short-circuit — never enqueue a doomed Klein job."""
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    monkeypatch.setattr(wk, '_prefill_region',
                        lambda *a, **k: (None, {'kind': 'unavailable', 'detail': 'no engine'}))
    monkeypatch.setattr(wk, '_run_klein_job',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('must not reach the GPU without a prefill')))
    img = tmp_path / 'wm.webp'
    Image.new('RGB', (512, 512), (30, 30, 30)).save(img, 'WEBP')
    with app.app_context():
        ok, err = wk.inpaint_watermark_klein('local', str(img), [[0.4, 0.4, 0.5, 0.5]])
    assert ok is False and err == {'kind': 'unavailable', 'detail': 'no engine'}


# --- restore (undo a clean) -------------------------------------------------

def test_restore_recovers_cropped_original_and_keeps_orig(app, monkeypatch):
    """Clean (crop) then restore: the exact original bytes AND dimensions come back, the
    row is 'detected' again (re-cleanable), and the .orig sibling is kept."""
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = _kept_image(svc, ds.id, 'wm.webp', size=(1024, 1024), bbox=[0.0, 0.0, 1.0, 0.05])
        path = svc._img_path(img)
        before = open(path, 'rb').read()
        counts, err = svc.clean_watermarks(LOCAL_USER, ds.id)
        assert err is None and counts['cropped'] == 1
        with Image.open(path) as im:
            assert im.height < 1024                       # cropped in place
        assert open(path, 'rb').read() != before

        result = svc.restore_watermark_original(LOCAL_USER, ds.id, img.id)
        assert result is not None
        assert result['watermark_state'] == 'detected'
        assert result['watermark_route'] == 'crop'        # recomputed on the restored 1024²
        assert open(path, 'rb').read() == before          # exact original bytes back
        with Image.open(path) as im:
            assert im.size == (1024, 1024)                # original dimensions back
        stem, ext = os.path.splitext(path)
        assert os.path.exists(f'{stem}.orig{ext}')        # .orig kept (source of truth)
        assert svc.db.session.get(FaceDatasetImage, img.id).watermark_state == 'detected'


def test_restore_then_reclean_reuses_write_once_original(app, monkeypatch):
    """clean -> restore -> clean -> restore: _preserve_original is write-once, so the
    .orig is never clobbered by an already-edited image and the true original survives
    every cycle."""
    from app.services import face_dataset_service as svc
    from app.services import watermark_lama
    from app.config import LOCAL_USER

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Cycle', 'cycle')
        img = _kept_image(svc, ds.id, 'wm.webp', size=(1024, 1024), bbox=[0.0, 0.0, 1.0, 0.05])
        path = svc._img_path(img)
        stem, ext = os.path.splitext(path)
        orig = f'{stem}.orig{ext}'
        before = open(path, 'rb').read()

        assert svc.clean_watermarks(LOCAL_USER, ds.id)[0]['cropped'] == 1      # crop #1
        assert open(orig, 'rb').read() == before
        svc.restore_watermark_original(LOCAL_USER, ds.id, img.id)
        assert open(path, 'rb').read() == before

        assert svc.clean_watermarks(LOCAL_USER, ds.id)[0]['cropped'] == 1      # crop #2 (detected again)
        assert open(orig, 'rb').read() == before                              # write-once: still the original
        svc.restore_watermark_original(LOCAL_USER, ds.id, img.id)
        assert open(path, 'rb').read() == before                              # original recovered after N cycles


def test_restore_without_original_raises_filenotfound(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoOrig', 'noorig')
        img = _kept_image(svc, ds.id, 'clean.webp', state='cleaned')
        with pytest.raises(FileNotFoundError):
            svc.restore_watermark_original(LOCAL_USER, ds.id, img.id)


def test_restore_foreign_or_missing_returns_none(app):
    """A missing id, or an image that belongs to a DIFFERENT dataset, is a no-op None
    (never restores across datasets) — same ownership guard as set_watermark_regions."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'A', 'a')
        assert svc.restore_watermark_original(LOCAL_USER, ds.id, 999999) is None
        other = svc.create_dataset(LOCAL_USER, 'B', 'b')
        img = _kept_image(svc, other.id, 'x.webp', state='cleaned')
        assert svc.restore_watermark_original(LOCAL_USER, ds.id, img.id) is None


def _seed_cleaned_with_orig(svc, ds_id, filename, *, orig_bytes, current_bytes, bbox):
    """A 'cleaned' row backed by a shrunk file + a preserved <stem>.orig sibling."""
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), 'wb') as fh:
        fh.write(current_bytes)
    stem, ext = os.path.splitext(filename)
    with open(os.path.join(d, f'{stem}.orig{ext}'), 'wb') as fh:
        fh.write(orig_bytes)
    img = FaceDatasetImage(dataset_id=ds_id, source='import', status='keep',
                           filename=filename, framing='body',
                           watermark_state='cleaned', watermark_bbox=json.dumps(bbox))
    svc.db.session.add(img)
    svc.db.session.commit()
    return img


def test_watermark_restore_route_recovers_original(client, app):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    ds_id = _create(client, 'Restore', 'restore').get_json()['id']
    orig = _img_bytes(color=(10, 200, 10), size=(1024, 1024))
    with app.app_context():
        img = _seed_cleaned_with_orig(
            svc, ds_id, 'wm.webp',
            orig_bytes=orig,
            current_bytes=_img_bytes(color=(200, 10, 10), size=(1024, 300)),
            bbox=[0.0, 0.0, 1.0, 0.05])
        img_id = img.id
        path = svc._img_path(img)

    resp = client.post(f'/api/dataset/{ds_id}/image/{img_id}/watermark-restore', json={})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['watermark_state'] == 'detected'
    assert body['watermark_route'] == 'crop'
    with app.app_context():
        assert open(path, 'rb').read() == orig
        assert svc.db.session.get(FaceDatasetImage, img_id).watermark_state == 'detected'
        stem, ext = os.path.splitext(path)
        assert os.path.exists(f'{stem}.orig{ext}')


def test_watermark_restore_route_404_without_original(client, app):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'NoOrig', 'noorig').get_json()['id']
    with app.app_context():
        img_id = _kept_image(svc, ds_id, 'c.webp', state='cleaned').id
    resp = client.post(f'/api/dataset/{ds_id}/image/{img_id}/watermark-restore', json={})
    assert resp.status_code == 404
    assert 'original' in (resp.get_json().get('error') or '')


def test_watermark_restore_route_404_when_image_missing(client):
    ds_id = _create(client, 'X', 'x').get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/image/999999/watermark-restore', json={})
    assert resp.status_code == 404
