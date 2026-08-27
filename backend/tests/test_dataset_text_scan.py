"""🔤 Dataset text scan — the dataset half of the text-clean lane, and the
guards that stop a WATERMARK scan from erasing what the text pass wrote.

Two surfaces, one contract: same merge rules as the bank pass (zones fold into
watermark_regions, dismissed rows are left alone, 'error' rows retry on a
plain run), with the one deliberate difference documented on detect_text —
a 'cleaned' dataset row restarts from only the new text zones, because the
dataset clean REPLACED the file's pixels (the bank keeps a separate blob).
"""
import json
import os

from PIL import Image


def _create(client, name='Pages', trigger='pg'):
    r = client.post('/api/dataset/create',
                    json={'name': name, 'trigger_word': trigger})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id']


def _kept_image(svc, ds_id, filename, *, state=None, bbox=None, regions=None,
                text_state=None, size=(1024, 1024)):
    from app.models import FaceDatasetImage
    d = svc._dataset_dir(ds_id)
    os.makedirs(d, exist_ok=True)
    Image.new('RGB', size, (200, 30, 30)).save(os.path.join(d, filename), 'WEBP')
    img = FaceDatasetImage(
        dataset_id=ds_id, source='import', status='keep', filename=filename,
        framing='body', watermark_state=state,
        watermark_bbox=json.dumps(bbox) if bbox is not None else None,
        watermark_regions=json.dumps(regions) if regions is not None else None,
        text_state=text_state)
    svc.db.session.add(img)
    svc.db.session.commit()
    return img.id


def _row(app, image_id):
    from app.extensions import db
    from app.models import FaceDatasetImage
    with app.app_context():
        r = db.session.get(FaceDatasetImage, image_id)
        return {'watermark_state': r.watermark_state,
                'watermark_bbox': r.watermark_bbox,
                'watermark_regions': r.watermark_regions,
                'text_state': r.text_state}


def _ocr_ready(monkeypatch, ok=True):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_video_text',
                        lambda: {'ok': ok, 'detail': '' if ok else 'missing'})


def _fake_reader(monkeypatch, boxes_by_basename):
    from app.services import video_safe_zone

    def fake(frames, *, timeout=None, should_stop=None, on_progress=None,
             score_min=None):
        return {f['key']: [list(b) for b in
                           boxes_by_basename.get(os.path.basename(f['path']), [])]
                for f in frames}
    monkeypatch.setattr(video_safe_zone, 'read_text_boxes', fake)


TWO_LINES = [[0.30, 0.10, 0.70, 0.14, 0.97], [0.28, 0.16, 0.72, 0.20, 0.95]]


class TestDetectText:
    def test_route_flags_text_and_reports_counts(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            with_text = _kept_image(svc, ds, 'a.webp')
            without = _kept_image(svc, ds, 'b.webp')
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.webp': TWO_LINES, 'b.webp': []})
        r = client.post(f'/api/dataset/{ds}/text/detect', json={})
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        assert body['found'] == 1 and body['none'] == 1 and body['checked'] == 2
        assert body['stopped'] is False and body['uncovered'] == 0
        flagged = _row(app, with_text)
        assert flagged['watermark_state'] == 'detected'
        assert flagged['text_state'] == 'detected'
        assert len(json.loads(flagged['watermark_regions'])) == 1
        blank = _row(app, without)
        assert blank['text_state'] == 'none'
        assert blank['watermark_state'] is None

    def test_dismissed_rows_are_skipped_even_on_rescan(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            img = _kept_image(svc, ds, 'a.webp', state='dismissed')
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.webp': TWO_LINES})
        r = client.post(f'/api/dataset/{ds}/text/detect', json={'rescan': True})
        assert r.status_code == 200
        assert r.get_json()['checked'] == 0
        row = _row(app, img)
        assert row['watermark_state'] == 'dismissed'
        assert row['text_state'] is None

    def test_plain_run_resumes_rescan_rereads(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            img = _kept_image(svc, ds, 'a.webp', text_state='none')
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.webp': TWO_LINES})
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={}).get_json()['checked'] == 0
        assert _row(app, img)['text_state'] == 'none'
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={'rescan': True}).get_json()['found'] == 1
        assert _row(app, img)['text_state'] == 'detected'

    def test_existing_geometry_is_merged_not_replaced(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            img = _kept_image(svc, ds, 'a.webp', state='detected',
                              regions=[[0.05, 0.05, 0.15, 0.15]])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.webp': TWO_LINES})
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={}).status_code == 200
        regions = json.loads(_row(app, img)['watermark_regions'])
        assert [0.05, 0.05, 0.15, 0.15] in regions and len(regions) == 2

    def test_cleaned_row_restarts_from_text_zones_only(self, app, client, monkeypatch):
        # The dataset clean REPLACED the pixels, so the stored geometry
        # describes healed work — only the fresh text zones come back.
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            img = _kept_image(svc, ds, 'a.webp', state='cleaned',
                              bbox=[0.01, 0.90, 0.10, 0.99])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.webp': TWO_LINES})
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={}).status_code == 200
        row = _row(app, img)
        assert row['watermark_state'] == 'detected'
        regions = json.loads(row['watermark_regions'])
        assert len(regions) == 1                     # no resurrected old zone
        assert not any(b[1] >= 0.85 for b in regions)

    def test_engine_missing_is_a_503(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            _kept_image(svc, ds, 'a.webp')
        _ocr_ready(monkeypatch, ok=False)
        r = client.post(f'/api/dataset/{ds}/text/detect', json={})
        assert r.status_code == 503
        assert 'Setup' in r.get_json()['error']

    def test_cancel_route_without_a_scan_is_409(self, client):
        ds = _create(client)
        assert client.post(f'/api/dataset/{ds}/text/detect/cancel',
                           json={}).status_code == 409

    def test_unreadable_files_are_errors_and_reported_not_stopped(
            self, app, client, monkeypatch):
        # Same guard as the bank pass: a key the child never answered, with no
        # stop asked, is a file its reader could not open — a counted per-image
        # error the next plain run retries, never a phantom "Stopped".
        from app.services import face_dataset_service as svc
        from app.services import video_safe_zone
        ds = _create(client)
        with app.app_context():
            readable = _kept_image(svc, ds, 'ok.webp')
            broken = _kept_image(svc, ds, 'broken.webp')
        _ocr_ready(monkeypatch)

        def fake(frames, *, timeout=None, should_stop=None, on_progress=None,
             score_min=None):
            return {f['key']: [list(b) for b in TWO_LINES]
                    for f in frames
                    if os.path.basename(f['path']) == 'ok.webp'}
        monkeypatch.setattr(video_safe_zone, 'read_text_boxes', fake)
        r = client.post(f'/api/dataset/{ds}/text/detect', json={})
        assert r.status_code == 200
        body = r.get_json()
        assert body['stopped'] is False
        assert body['unreadable'] == 1
        assert _row(app, readable)['text_state'] == 'detected'
        assert _row(app, broken)['text_state'] == 'error'
        # The next plain run retries the errored row.
        _fake_reader(monkeypatch, {'ok.webp': TWO_LINES, 'broken.webp': TWO_LINES})
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={}).get_json()['found'] == 1
        assert _row(app, broken)['text_state'] == 'detected'


class TestWatermarkScanGuards:
    """A watermark scan runs AFTER a text scan: the text zones must survive it.
    Before these guards the vision pass reset watermark_regions on every row it
    judged — running 🧽 Find watermarks after 🔤 Find text erased the text work."""

    def _text_flagged(self, app, svc, ds, filename='a.webp'):
        with app.app_context():
            return _kept_image(svc, ds, filename, state='detected',
                               regions=[[0.28, 0.09, 0.72, 0.22]],
                               text_state='detected')

    def test_vision_none_verdict_keeps_text_zones_flagged(
            self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        import app.services.vision_ollama as vo
        ds = _create(client)
        img = self._text_flagged(app, svc, ds)
        monkeypatch.setattr(vo, 'describe_image_ollama',
                            lambda *a, **k: '{"present":false}')
        monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: None)
        with app.app_context():
            counts = svc.detect_watermarks(
                'local', ds, backend={'backend': 'vision', 'requested': 'vision',
                                      'fell_back': False, 'detail': ''})
        assert counts['none'] == 1                   # the watermark truth
        row = _row(app, img)
        assert row['watermark_state'] == 'detected'  # still repaintable
        assert json.loads(row['watermark_regions']) == [[0.28, 0.09, 0.72, 0.22]]

    def test_vision_found_box_folds_into_text_zones(
            self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        import app.services.vision_ollama as vo
        ds = _create(client)
        img = self._text_flagged(app, svc, ds)
        monkeypatch.setattr(
            vo, 'describe_image_ollama',
            lambda *a, **k: '{"present":true,"x1":20,"y1":900,"x2":120,"y2":980}')
        monkeypatch.setattr(vo, 'unload_vision_model', lambda *a, **k: None)
        with app.app_context():
            counts = svc.detect_watermarks(
                'local', ds, backend={'backend': 'vision', 'requested': 'vision',
                                      'fell_back': False, 'detail': ''})
        assert counts['detected'] == 1
        row = _row(app, img)
        regions = json.loads(row['watermark_regions'])
        assert [0.28, 0.09, 0.72, 0.22] in regions   # text zone survived
        assert any(b[1] >= 0.8 for b in regions)     # the found box joined it
        assert row['watermark_bbox'] is not None     # and stays readable as-is
