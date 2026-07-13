"""Watermark auto-correction V1: bbox parsing, crop/LaMa/review routing, the detect
& clean batches, the routes (LaMa mocked present AND absent), the tuple error contract,
original preservation, and the additive columns."""
import io
import json
import os

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
                        lambda u, d: {'detected': 1, 'none': 2, 'checked': 3})
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
                        lambda u, d: ({'cropped': 2, 'inpainted': 1, 'needs_review': 0,
                                       'failed': 0, 'skipped': 0}, None))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] and body['cropped'] == 2 and body['inpainted'] == 1 and body['error'] is None


def test_clean_route_lama_absent_reports_skipped(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d: ({'cropped': 1, 'inpainted': 0, 'needs_review': 0,
                                       'failed': 0, 'skipped': 3}, None))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    body = resp.get_json()
    assert body['ok'] and body['skipped'] == 3 and body['cropped'] == 1


def test_clean_route_surfaces_error(client, app, monkeypatch):
    from app.services import face_dataset_service as svc
    ds_id = _create(client, 'R', 'r').get_json()['id']
    monkeypatch.setattr(svc, 'clean_watermarks',
                        lambda u, d: ({'cropped': 0, 'inpainted': 0, 'needs_review': 0,
                                       'failed': 1, 'skipped': 0},
                                      {'kind': 'failed', 'detail': 'boom'}))
    resp = client.post(f'/api/dataset/{ds_id}/watermarks/clean')
    body = resp.get_json()
    assert body['error'] == {'kind': 'failed', 'detail': 'boom'}


def test_watermark_routes_404_when_dataset_missing(client):
    assert client.post('/api/dataset/999999/watermarks/detect').status_code == 404
    assert client.post('/api/dataset/999999/watermarks/clean').status_code == 404


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
