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

    def test_watermark_box_with_no_text_stays_none(self, app, client, monkeypatch):
        """Same guard as the bank pass: the verdict comes from the OCR lines,
        not the merged output — a page carrying only a watermark box is NOT a
        text page, and 'What to clean' routes the repaint on text_state."""
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            image_id = _kept_image(svc, ds, 'mark.webp', state='detected',
                                   bbox=[0.01, 0.90, 0.10, 0.99])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'mark.webp': []})
        r = client.post(f'/api/dataset/{ds}/text/detect', json={})
        assert r.status_code == 200, r.get_json()
        row = _row(app, image_id)
        assert row['text_state'] == 'none'
        assert row['watermark_state'] == 'detected'      # the box is kept…
        assert row['watermark_regions'] is None          # …and left as it was

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


class TestSampleParity:
    """The dataset's launch-window dials — full parity with the bank's."""

    def test_sample_counts_pages_that_need_reading_not_the_kept_pile(
            self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            done1 = _kept_image(svc, ds, 'a.webp', text_state='none')
            done2 = _kept_image(svc, ds, 'b.webp', text_state='none')
            fresh1 = _kept_image(svc, ds, 'c.webp')
            fresh2 = _kept_image(svc, ds, 'd.webp')
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'c.webp': TWO_LINES, 'd.webp': TWO_LINES})
        # limit=1 must read ONE page that needs reading (c), not stop on a/b.
        r = client.post(f'/api/dataset/{ds}/text/detect', json={'limit': 1})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()['checked'] == 1
        assert _row(app, fresh1)['text_state'] == 'detected'
        assert _row(app, fresh2)['text_state'] is None
        assert _row(app, done1)['text_state'] == 'none'
        assert _row(app, done2)['text_state'] == 'none'
        # The plain follow-up finishes exactly what the sample left.
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={}).get_json()['checked'] == 1
        assert _row(app, fresh2)['text_state'] == 'detected'

    def test_sample_with_redo_rereads_the_same_first_images(
            self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            first = _kept_image(svc, ds, 'a.webp', text_state='none')
            second = _kept_image(svc, ds, 'b.webp', text_state='none')
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.webp': TWO_LINES, 'b.webp': TWO_LINES})
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={'limit': 1, 'rescan': True}).status_code == 200
        assert _row(app, first)['text_state'] == 'detected'
        assert _row(app, second)['text_state'] == 'none'   # beyond the sample

    def test_bad_limit_is_a_400(self, client, monkeypatch):
        ds = _create(client)
        _ocr_ready(monkeypatch)
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={'limit': 0}).status_code == 400
        assert client.post(f'/api/dataset/{ds}/text/detect',
                           json={'limit': 'many'}).status_code == 400

    def test_image_payload_carries_text_state(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            _kept_image(svc, ds, 'a.webp', text_state='detected')
        detail = client.get(f'/api/dataset/{ds}').get_json()
        imgs = detail.get('images') or []
        assert imgs and imgs[0].get('text_state') == 'detected'


class TestTextCleanDataset:
    """Same graft, dataset surface: filler first on text rows, promotion on a
    full fill, LaMa leftovers, rectangle fallback when the filler is gone."""

    def _text_row(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            image_id = _kept_image(svc, ds, 'page.webp', state='detected',
                                   regions=[[0.3, 0.3, 0.6, 0.4]],
                                   text_state='detected')
        _ocr_ready(monkeypatch)
        return ds, image_id

    def _arm(self, monkeypatch, fill_result, lama_calls=None):
        from app.services import text_fill, watermark_lama

        def fake_fill(items, **kwargs):
            return {i['image_path']: dict(fill_result) for i in items}
        monkeypatch.setattr(text_fill, 'fill_batch', fake_fill)
        monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
        monkeypatch.setattr(
            watermark_lama, 'inpaint_watermarks',
            lambda path, bboxes, timeout=300, **k:
                ((lama_calls.append((path, bboxes)) if lama_calls is not None
                  else None) or (True, None)))
        monkeypatch.setattr(
            watermark_lama, 'inpaint_batch',
            lambda items, device='cpu':
                {i['image_path']: (True, None) for i in items})

    def test_full_fill_promotes_and_counts_text_filled(
            self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds, image_id = self._text_row(app, client, monkeypatch)
        self._arm(monkeypatch, {'ok': True, 'filled': 1, 'busy_boxes': []})
        with app.app_context():
            counts, err = svc.clean_watermarks('local', ds,
                                               image_ids=[image_id])
        assert err is None
        assert counts['text_filled'] == 1 and counts['inpainted'] == 0
        assert _row(app, image_id)['watermark_state'] == 'cleaned'

    def test_busy_leftovers_reach_lama_with_glyph_boxes(
            self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds, image_id = self._text_row(app, client, monkeypatch)
        busy = [[0.31, 0.31, 0.35, 0.36]]
        lama_calls = []
        self._arm(monkeypatch,
                  {'ok': True, 'filled': 1, 'busy_boxes': busy}, lama_calls)
        with app.app_context():
            counts, err = svc.clean_watermarks('local', ds,
                                               image_ids=[image_id])
        assert err is None
        assert counts['inpainted'] == 1 and counts['text_filled'] == 0
        assert len(lama_calls) == 1 and lama_calls[0][1] == busy
        assert _row(app, image_id)['watermark_state'] == 'cleaned'

    def test_filler_down_falls_back_to_rectangles(
            self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        from app.services import text_fill
        ds, image_id = self._text_row(app, client, monkeypatch)
        lama_calls = []
        self._arm(monkeypatch, {'ok': True}, lama_calls)

        def boom(items, **kwargs):
            raise RuntimeError('filler gone')
        monkeypatch.setattr(text_fill, 'fill_batch', boom)
        with app.app_context():
            counts, err = svc.clean_watermarks('local', ds,
                                               image_ids=[image_id])
        assert err is None
        assert counts['inpainted'] == 1
        assert lama_calls and lama_calls[0][1] == [[0.3, 0.3, 0.6, 0.4]]


class TestCleanTargetDataset:
    """🧽 'What to clean' on the dataset surface — the same page-level split as
    the bank: 'text' = pages 🔤 flagged (a page carrying both counts here),
    'watermark' = every other flagged page. Bank parity is the contract."""

    def _mixed_dataset(self, app, client, monkeypatch):
        """text_id is the MIXED page (text zones + an older detector box);
        mark_id is watermark-flagged only, with a border bbox so the crop
        route handles it without any engine."""
        from app.services import face_dataset_service as svc
        ds = _create(client)
        with app.app_context():
            text_id = _kept_image(svc, ds, 'page.webp', state='detected',
                                  bbox=[0.01, 0.90, 0.10, 0.99],
                                  regions=[[0.3, 0.3, 0.6, 0.4]],
                                  text_state='detected')
            mark_id = _kept_image(svc, ds, 'mark.webp', state='detected',
                                  bbox=[0.30, 0.00, 0.70, 0.04])
        _ocr_ready(monkeypatch)
        return ds, text_id, mark_id

    def _arm_fill(self, monkeypatch):
        from app.services import text_fill, watermark_lama
        monkeypatch.setattr(text_fill, 'fill_batch', lambda items, **k: {
            i['image_path']: {'ok': True, 'filled': len(i['regions']),
                              'busy_boxes': []} for i in items})
        monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
        monkeypatch.setattr(
            watermark_lama, 'inpaint_batch',
            lambda items, device='cpu':
                {i['image_path']: (True, None) for i in items})

    def test_target_text_cleans_only_text_pages(self, app, client, monkeypatch):
        from app.services import face_dataset_service as svc
        ds, text_id, mark_id = self._mixed_dataset(app, client, monkeypatch)
        self._arm_fill(monkeypatch)
        with app.app_context():
            counts, err = svc.clean_watermarks('local', ds, target='text')
        assert err is None
        assert counts['text_filled'] == 1 and counts['cropped'] == 0
        assert _row(app, text_id)['watermark_state'] == 'cleaned'
        assert _row(app, mark_id)['watermark_state'] == 'detected'  # untouched

    def test_target_watermark_leaves_text_pages_alone(
            self, app, client, monkeypatch):
        """The mixed page (text + detector box) belongs to the TEXT family —
        the page is the unit, so a watermark-only run must not touch it."""
        from app.services import face_dataset_service as svc
        ds, text_id, mark_id = self._mixed_dataset(app, client, monkeypatch)
        self._arm_fill(monkeypatch)
        with app.app_context():
            counts, err = svc.clean_watermarks('local', ds, target='watermark')
        assert err is None
        assert counts['cropped'] == 1 and counts['text_filled'] == 0
        assert _row(app, text_id)['watermark_state'] == 'detected'
        assert _row(app, mark_id)['watermark_state'] == 'cleaned'

    def test_route_validates_and_forwards_target(
            self, app, client, monkeypatch):
        ds, text_id, mark_id = self._mixed_dataset(app, client, monkeypatch)
        self._arm_fill(monkeypatch)
        r = client.post(f'/api/dataset/{ds}/watermarks/clean',
                        json={'target': 'bogus'})
        assert r.status_code == 400
        assert 'target' in r.get_json()['error']
        r = client.post(f'/api/dataset/{ds}/watermarks/clean',
                        json={'target': 'text'})
        assert r.status_code == 200, r.get_json()
        assert _row(app, text_id)['watermark_state'] == 'cleaned'
        assert _row(app, mark_id)['watermark_state'] == 'detected'  # untouched


class TestTextPreviewDataset:
    """The 🔤 launch window's result strip, dataset surface — the twin of the
    bank's /text/preview: flagged pages with their zones, oldest-id first (the
    sample's own order), polled by the window while a scan runs."""

    def _flagged(self, app, client, n=2):
        from app.services import face_dataset_service as svc
        ds = _create(client)
        ids = []
        with app.app_context():
            for i in range(n):
                ids.append(_kept_image(
                    svc, ds, f'p{i}.webp', state='detected',
                    regions=[[0.2, 0.1, 0.8, 0.2]], text_state='detected'))
        return ds, ids

    def test_preview_lists_flagged_pages_with_zones_id_asc(self, app, client):
        ds, ids = self._flagged(app, client)
        r = client.get(f'/api/dataset/{ds}/text/preview')
        assert r.status_code == 200
        data = r.get_json()
        assert data['total'] == 2
        assert [i['id'] for i in data['items']] == sorted(ids)
        for item in data['items']:
            assert item['filename']
            assert item['regions'] and all(len(b) == 4 for b in item['regions'])

    def test_rejected_pages_are_left_out(self, app, client):
        ds, ids = self._flagged(app, client)
        from app.extensions import db
        from app.models import FaceDatasetImage
        with app.app_context():
            db.session.get(FaceDatasetImage, ids[0]).status = 'reject'
            db.session.commit()
        data = client.get(f'/api/dataset/{ds}/text/preview').get_json()
        assert data['total'] == 1 and len(data['items']) == 1
        assert data['items'][0]['id'] == ids[1]

    def test_limit_is_honoured_and_total_still_counts_everything(
            self, app, client):
        ds, _ids = self._flagged(app, client)
        data = client.get(f'/api/dataset/{ds}/text/preview?limit=1').get_json()
        assert len(data['items']) == 1
        assert data['total'] == 2               # the strip's "of N" stays honest
        # Zero and garbage fall back to the default instead of erroring the
        # strip away (0 is "unset", not "show nothing") — bank contract.
        for bad in ('0', 'nope'):
            r = client.get(f'/api/dataset/{ds}/text/preview?limit={bad}')
            assert r.status_code == 200
            assert len(r.get_json()['items']) == 2

    def test_missing_dataset_is_a_404(self, client):
        assert client.get('/api/dataset/424242/text/preview').status_code == 404


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
