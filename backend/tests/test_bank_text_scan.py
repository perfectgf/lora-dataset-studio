"""🔤 Bank text scan — the OCR pass that feeds the watermark funnel.

What must hold: the pass folds its zones into the ONE channel the cleaning
levels consume (watermark_regions + 'detected'), it never loses geometry the
row already carried (a hand mask, the detector's box, the zones a previous
clean covered — losing those lets the next repaint resurrect the mark), it
never re-flags a row the user dismissed, and a stopped or engine-less run
leaves every unreached row exactly as it was.

The OCR engine itself is monkeypatched at the seam the job imports
(`video_safe_zone.read_text_boxes`) — these tests exercise the pass, not
onnxruntime; the seam's own child is covered by the video lane's tests.
"""
import json
import os

from PIL import Image


def _photo(size=1000, value=90):
    im = Image.new('RGB', (size, size), (value, value, value))
    return im


def _mkbank(client, tmp_path, names, name='TXT'):
    src = tmp_path / 'src'
    for rel in names:
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        _photo().save(str(p), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _rows(app, bank_id):
    """{basename: row-facts} — keyed, because inventory orders rows by NAME,
    not by the order the test wrote the files."""
    from app.models import BankImage
    with app.app_context():
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        return {os.path.basename(r.relpath): {
            'id': r.id,
            'watermark_state': r.watermark_state,
            'watermark_bbox': r.watermark_bbox,
            'watermark_regions': r.watermark_regions,
            'watermark_clean_method': r.watermark_clean_method,
            'watermark_fingerprint': r.watermark_fingerprint,
            'text_state': r.text_state} for r in rows}


def _ocr_ready(monkeypatch, ok=True):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_video_text',
                        lambda: {'ok': ok, 'detail': '' if ok else 'not installed'})


def _fake_reader(monkeypatch, boxes_by_basename, *, reachable=None):
    """read_text_boxes stand-in: answers per basename; `reachable` limits which
    frames the 'child' reached (absent key = never read, the seam's contract)."""
    from app.services import video_safe_zone

    def fake(frames, *, timeout=None, should_stop=None, on_progress=None,
             score_min=None):
        out = {}
        for f in frames:
            base = os.path.basename(f['path'])
            if reachable is not None and base not in reachable:
                continue
            out[f['key']] = [list(b) for b in boxes_by_basename.get(base, [])]
        return out
    monkeypatch.setattr(video_safe_zone, 'read_text_boxes', fake)


TWO_LINES = [[0.30, 0.10, 0.70, 0.14, 0.97], [0.28, 0.16, 0.72, 0.20, 0.95]]


class TestTextScan:
    def test_finds_text_flags_the_row_and_feeds_the_clean_pool(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg', 'clean.jpg'])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'page.jpg': TWO_LINES, 'clean.jpg': []})
        r = client.post(f'/api/bank/{bank_id}/text', json={})
        assert r.status_code == 202, r.get_json()
        rows = _rows(app, bank_id)
        page, clean = rows['page.jpg'], rows['clean.jpg']
        assert page['text_state'] == 'detected'
        assert page['watermark_state'] == 'detected'
        assert len(page['watermark_fingerprint'] or '') == 64
        regions = json.loads(page['watermark_regions'])
        assert len(regions) == 1               # two close lines -> one zone
        assert clean['text_state'] == 'none'
        assert clean['watermark_state'] is None    # no verdict invented
        assert clean['watermark_regions'] is None
        # The flagged row is now 🧽 Inpaint's work — the funnel, not a new lane.
        from app.services import image_bank_service as banks
        with app.app_context():
            pool = [row.id for row in banks._clean_pool_query(bank_id).all()]
        assert pool == [page['id']]

    def test_existing_detector_box_is_folded_in_not_lost(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        from app.extensions import db
        from app.models import BankImage
        with app.app_context():
            row = BankImage.query.filter_by(bank_id=bank_id).one()
            row.watermark_state = 'detected'
            row.watermark_bbox = json.dumps([0.01, 0.90, 0.10, 0.99])
            db.session.commit()
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'page.jpg': TWO_LINES})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        page = _rows(app, bank_id)['page.jpg']
        regions = json.loads(page['watermark_regions'])
        assert len(regions) == 2               # the corner mark + the text block
        assert any(b[1] >= 0.85 for b in regions)      # the old box survived

    def test_cleaned_row_is_reflagged_and_its_blob_dropped(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        from app.extensions import db
        from app.models import BankImage
        from app.services import image_bank_service as banks
        from app.models import ImageBank
        from app.services import bank_transfer_metadata as transfer
        with app.app_context():
            bank = db.session.get(ImageBank, bank_id)
            row = BankImage.query.filter_by(bank_id=bank_id).one()
            row.watermark_state = 'cleaned'
            row.watermark_clean_method = 'lama'
            row.watermark_bbox = json.dumps([0.01, 0.90, 0.10, 0.99])
            # A real cleaned row always carries the attestation the clean made
            # (_prepare_watermark_write): without it the generation guard
            # rightly refuses to trust — and purges — the stored geometry.
            row.watermark_fingerprint = transfer.content_fingerprint_path(
                banks.abs_image_path(bank, row))
            db.session.commit()
            blob = banks.clean_image_path(bank_id, row.id)
            blob.parent.mkdir(parents=True, exist_ok=True)
            _photo(64).save(str(blob), 'WEBP')
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'page.jpg': TWO_LINES})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        page = _rows(app, bank_id)['page.jpg']
        assert page['watermark_state'] == 'detected'
        assert page['watermark_clean_method'] is None
        assert not blob.exists()
        # The zone the previous clean covered rides along, so the next repaint
        # (which restarts from the source) cannot resurrect the old mark.
        regions = json.loads(page['watermark_regions'])
        assert any(b[1] >= 0.85 for b in regions)

    def test_dismissed_rows_are_never_reexamined(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        from app.extensions import db
        from app.models import BankImage
        from app.services import bank_transfer_metadata as transfer
        from app.services import image_bank_service as banks
        from app.models import ImageBank
        with app.app_context():
            bank = db.session.get(ImageBank, bank_id)
            row = BankImage.query.filter_by(bank_id=bank_id).one()
            row.watermark_state = 'dismissed'
            # The real dismiss flow attests the ruling against the current
            # bytes (_prepare_watermark_write); without it the shared clause
            # rightly re-examines a row whose file may have changed.
            row.watermark_fingerprint = transfer.content_fingerprint_path(
                banks.abs_image_path(bank, row))
            db.session.commit()
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'page.jpg': TWO_LINES})
        assert client.post(f'/api/bank/{bank_id}/text',
                           json={'rescan': True}).status_code == 202
        page = _rows(app, bank_id)['page.jpg']
        assert page['watermark_state'] == 'dismissed'
        assert page['text_state'] is None
        assert page['watermark_regions'] is None

    def test_todo_resumes_and_rescan_rereads(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.jpg': [], 'b.jpg': []})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        states = {k: v['text_state'] for k, v in _rows(app, bank_id).items()}
        assert states == {'a.jpg': 'none', 'b.jpg': 'none'}
        # Plain run again: nothing to do, nothing rewritten.
        _fake_reader(monkeypatch, {'a.jpg': TWO_LINES, 'b.jpg': TWO_LINES})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        states = {k: v['text_state'] for k, v in _rows(app, bank_id).items()}
        assert states == {'a.jpg': 'none', 'b.jpg': 'none'}
        # Rescan: both re-read, both flagged now.
        assert client.post(f'/api/bank/{bank_id}/text',
                           json={'rescan': True}).status_code == 202
        states = {k: v['text_state'] for k, v in _rows(app, bank_id).items()}
        assert states == {'a.jpg': 'detected', 'b.jpg': 'detected'}

    def test_engine_missing_is_a_503_before_any_row_is_touched(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        _ocr_ready(monkeypatch, ok=False)
        r = client.post(f'/api/bank/{bank_id}/text', json={})
        assert r.status_code == 503
        assert 'Setup' in r.get_json()['error']
        assert _rows(app, bank_id)['page.jpg']['text_state'] is None

    def test_engine_dying_mid_pass_leaves_unscanned_rows_untouched(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        _ocr_ready(monkeypatch)
        from app.services import video_safe_zone

        def boom(frames, **kwargs):
            raise RuntimeError('the text reader produced no result')
        monkeypatch.setattr(video_safe_zone, 'read_text_boxes', boom)
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        page = _rows(app, bank_id)['page.jpg']
        assert page['text_state'] is None
        assert page['watermark_state'] is None

    def test_unreadable_files_are_errors_not_a_cancelled_pass(
            self, app, client, tmp_path, monkeypatch):
        # A missing key with NO stop asked means the child could not READ the
        # file (a real bank hit this on every image at once — an unreadable
        # path — and the pass lied "cancelled — 0 with text"). The row is
        # marked 'error' (retried by the next plain run) and the pass says
        # done, with the shortfall counted.
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.jpg': TWO_LINES, 'b.jpg': TWO_LINES},
                     reachable={'a.jpg'})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        rows = _rows(app, bank_id)
        assert rows['a.jpg']['text_state'] == 'detected'
        assert rows['b.jpg']['text_state'] == 'error'
        from app.services import bank_jobs
        with app.app_context():
            detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
        assert detail.startswith('done')
        assert 'cancelled' not in detail
        assert '1 unreadable' in detail
        # The next plain run adopts the errored row (the todo clause keeps it).
        _fake_reader(monkeypatch, {'a.jpg': TWO_LINES, 'b.jpg': TWO_LINES})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        assert _rows(app, bank_id)['b.jpg']['text_state'] == 'detected'

    def test_a_vanished_source_folder_reads_as_missing_not_cancelled(
            self, app, client, tmp_path, monkeypatch):
        # The real 81-image incident: the bank's source folder was renamed
        # under it (a downloader re-padding its chapter names). The pass must
        # say "no longer on disk", touch nothing, and never start the OCR
        # child for files it can see are gone.
        import shutil
        bank_id, src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
        _ocr_ready(monkeypatch)
        from app.services import bank_jobs, video_safe_zone

        def never(frames, **kwargs):
            raise AssertionError('the OCR child must not run on missing files')
        monkeypatch.setattr(video_safe_zone, 'read_text_boxes', never)
        shutil.rmtree(src)
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        rows = _rows(app, bank_id)
        assert all(r['text_state'] is None for r in rows.values())
        with app.app_context():
            detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
        assert detail.startswith('done')
        assert 'no longer on disk' in detail
        assert 'cancelled' not in detail

    def test_a_real_stop_leaves_unreached_rows_unscanned(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
        _ocr_ready(monkeypatch)
        from app.services import bank_jobs, video_safe_zone

        def fake(frames, *, timeout=None, should_stop=None, on_progress=None,
             score_min=None):
            # The user stops mid-chunk: the child answers what it read and
            # never reaches the rest — absent keys under a REAL cancel.
            bank_jobs.cancel(bank_id)
            return {f['key']: [list(b) for b in TWO_LINES]
                    for f in frames
                    if os.path.basename(f['path']) == 'a.jpg'}
        monkeypatch.setattr(video_safe_zone, 'read_text_boxes', fake)
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        rows = _rows(app, bank_id)
        assert rows['b.jpg']['text_state'] is None     # unscanned, not 'error'
        with app.app_context():
            detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
        assert detail.startswith('cancelled')

    def test_hand_drawn_mask_survives_a_text_scan(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        from app.extensions import db
        from app.models import BankImage
        with app.app_context():
            row = BankImage.query.filter_by(bank_id=bank_id).one()
            row.watermark_state = 'detected'
            row.watermark_regions = json.dumps([[0.05, 0.05, 0.15, 0.15]])
            db.session.commit()
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'page.jpg': TWO_LINES})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        page = _rows(app, bank_id)['page.jpg']
        regions = json.loads(page['watermark_regions'])
        assert [0.05, 0.05, 0.15, 0.15] in regions
        assert len(regions) == 2

    def test_levels_payload_carries_the_text_block(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg', 'clean.jpg'])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'page.jpg': TWO_LINES, 'clean.jpg': []})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
        assert levels['text'] == {'scanned': 2, 'found': 1, 'unscanned': 0}


class TestTextClean:
    """The repaint level on text-flagged rows: filler first, LaMa only on the
    glyph-tight leftovers, and honest fallbacks when the filler is gone."""

    def _flag_text(self, app, client, tmp_path, monkeypatch, names=('page.jpg',)):
        bank_id, _src = _mkbank(client, tmp_path, list(names))
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {n: TWO_LINES for n in names})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        return bank_id

    def _arm_engines(self, monkeypatch, fill_result, lama_calls=None):
        from app.services import text_fill, watermark_lama

        def fake_fill(items, **kwargs):
            return {i['image_path']: dict(fill_result) for i in items}
        monkeypatch.setattr(text_fill, 'fill_batch', fake_fill)
        monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
        monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')

        def fake_batch(items, device='cpu'):
            if lama_calls is not None:
                lama_calls.extend(items)
            return {i['image_path']: (True, None) for i in items}
        monkeypatch.setattr(watermark_lama, 'inpaint_batch', fake_batch)

    def test_all_zones_filled_stamps_text_fill_and_counts_as_inpainted(
            self, app, client, tmp_path, monkeypatch):
        bank_id = self._flag_text(app, client, tmp_path, monkeypatch)
        lama_calls = []
        self._arm_engines(monkeypatch,
                          {'ok': True, 'filled': 1, 'busy_boxes': []},
                          lama_calls)
        r = client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={})
        assert r.status_code == 202, r.get_json()
        page = _rows(app, bank_id)['page.jpg']
        assert page['watermark_state'] == 'cleaned'
        assert page['watermark_clean_method'] == 'text_fill'
        assert lama_calls == []              # nothing left for LaMa
        levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
        assert levels['inpainted'] == 1
        from app.services import bank_jobs
        with app.app_context():
            detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
        assert 'text-filled' in detail

    def test_busy_leftovers_go_to_lama_with_glyph_boxes_not_the_rectangle(
            self, app, client, tmp_path, monkeypatch):
        bank_id = self._flag_text(app, client, tmp_path, monkeypatch)
        busy = [[0.4, 0.4, 0.45, 0.45]]
        lama_calls = []
        self._arm_engines(monkeypatch,
                          {'ok': True, 'filled': 1, 'busy_boxes': busy},
                          lama_calls)
        assert client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                           json={}).status_code == 202
        page = _rows(app, bank_id)['page.jpg']
        assert page['watermark_state'] == 'cleaned'
        assert page['watermark_clean_method'] == 'text_fill'
        assert len(lama_calls) == 1
        assert lama_calls[0]['bboxes'] == busy      # glyph boxes, not the zone

    def test_filler_down_falls_back_to_the_old_rectangle_route(
            self, app, client, tmp_path, monkeypatch):
        bank_id = self._flag_text(app, client, tmp_path, monkeypatch)
        from app.services import text_fill

        def boom(items, **kwargs):
            raise RuntimeError('filler gone')
        lama_calls = []
        self._arm_engines(monkeypatch, {'ok': True}, lama_calls)
        monkeypatch.setattr(text_fill, 'fill_batch', boom)
        assert client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                           json={}).status_code == 202
        page = _rows(app, bank_id)['page.jpg']
        assert page['watermark_state'] == 'cleaned'
        assert page['watermark_clean_method'] == 'lama'   # the pre-filler route
        assert len(lama_calls) == 1
        assert len(lama_calls[0]['bboxes'][0]) == 4       # full zones

    def test_undo_takes_a_text_fill_back_like_any_clean(
            self, app, client, tmp_path, monkeypatch):
        bank_id = self._flag_text(app, client, tmp_path, monkeypatch)
        self._arm_engines(monkeypatch,
                          {'ok': True, 'filled': 1, 'busy_boxes': []})
        assert client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                           json={}).status_code == 202
        r = client.post(f'/api/bank/{bank_id}/watermark/undo', json={})
        assert r.status_code == 200 and r.get_json()['restored'] == 1
        page = _rows(app, bank_id)['page.jpg']
        assert page['watermark_state'] == 'detected'
        assert page['watermark_clean_method'] is None


class TestSampleAndSensitivity:
    """The launch window's two dials: a deterministic first-N sample, and the
    stored sensitivity handed to the OCR seam."""

    def test_sample_reads_the_first_n_and_leaves_the_rest_unscanned(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg', 'c.jpg'])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.jpg': TWO_LINES, 'b.jpg': [], 'c.jpg': TWO_LINES})
        r = client.post(f'/api/bank/{bank_id}/text', json={'limit': 2})
        assert r.status_code == 202, r.get_json()
        rows = _rows(app, bank_id)
        assert rows['a.jpg']['text_state'] == 'detected'
        assert rows['b.jpg']['text_state'] == 'none'
        assert rows['c.jpg']['text_state'] is None          # beyond the sample
        from app.services import bank_jobs
        with app.app_context():
            detail = (bank_jobs.get(bank_id) or {}).get('detail') or ''
        assert detail.startswith('sample')
        assert 'Review' in detail
        levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
        assert levels['text'] == {'scanned': 2, 'found': 1, 'unscanned': 1}
        # A plain full run finishes exactly what the sample left.
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        assert _rows(app, bank_id)['c.jpg']['text_state'] == 'detected'

    def test_sample_with_redo_rereads_the_same_first_images(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg', 'c.jpg'])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.jpg': [], 'b.jpg': [], 'c.jpg': []})
        assert client.post(f'/api/bank/{bank_id}/text',
                           json={'limit': 2}).status_code == 202
        # New sensitivity, same sample: redo + limit re-reads a and b, not c.
        _fake_reader(monkeypatch, {'a.jpg': TWO_LINES, 'b.jpg': TWO_LINES,
                                   'c.jpg': TWO_LINES})
        assert client.post(f'/api/bank/{bank_id}/text',
                           json={'limit': 2, 'rescan': True}).status_code == 202
        rows = _rows(app, bank_id)
        assert rows['a.jpg']['text_state'] == 'detected'
        assert rows['b.jpg']['text_state'] == 'detected'
        assert rows['c.jpg']['text_state'] is None

    def test_bad_limit_is_a_400(self, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg'])
        _ocr_ready(monkeypatch)
        assert client.post(f'/api/bank/{bank_id}/text',
                           json={'limit': 0}).status_code == 400
        assert client.post(f'/api/bank/{bank_id}/text',
                           json={'limit': 'lots'}).status_code == 400

    def test_stored_sensitivity_reaches_the_ocr_seam_clamped(
            self, app, client, tmp_path, monkeypatch):
        from app.services import video_safe_zone
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg'])
        _ocr_ready(monkeypatch)
        seen = {}

        def fake(frames, *, timeout=None, should_stop=None, on_progress=None,
                 score_min=None):
            seen['score_min'] = score_min
            return {f['key']: [] for f in frames}
        monkeypatch.setattr(video_safe_zone, 'read_text_boxes', fake)
        r = client.put('/api/settings',
                       json={'config': {'text_scan': {'score_min': 0.35}}})
        assert r.status_code == 200, r.get_json()
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        assert seen['score_min'] == 0.35
        # A hand-edited absurd value degrades visibly, it does not abort.
        assert client.put('/api/settings',
                          json={'config': {'text_scan': {'score_min': 9}}}).status_code == 200
        with app.app_context():
            assert video_safe_zone.text_score_min() == 0.95

    def test_capabilities_payload_carries_the_stored_sensitivity(self, client):
        caps = client.get('/api/capabilities').get_json()
        assert caps['text_scan_score_min'] == 0.5


class TestWatermarkScanGuards:
    """A watermark scan AFTER a text scan must not undo the text pass: a 'none'
    verdict is about WATERMARKS and may not unflag zones still waiting for a
    repaint, and a found box must fold into the regions (regions win over the
    bbox at cleaning time, so a box left outside them is never repainted)."""

    def _text_flag(self, app, bank_id):
        from app.extensions import db
        from app.models import BankImage
        with app.app_context():
            row = BankImage.query.filter_by(bank_id=bank_id).one()
            row.watermark_state = 'detected'
            row.watermark_regions = json.dumps([[0.28, 0.09, 0.72, 0.22]])
            row.text_state = 'detected'
            db.session.commit()
            return row.id

    def _run_detector_scan(self, client, bank_id, monkeypatch, child_rows):
        """Drive the bank watermark scan down the DETECTOR route with a fake
        child: `child_rows(path)` -> (state, score, regions)."""
        from app.services import bank_transfer_metadata as transfer
        from app.services import watermark_detector
        from app import capabilities
        monkeypatch.setattr(
            watermark_detector, 'resolve_backend',
            lambda requested=None: {'requested': 'detector', 'backend': 'detector',
                                    'fell_back': False, 'detail': ''})
        monkeypatch.setattr(capabilities, 'watermark_detect_gpu_available',
                            lambda: False)

        def fake_scan(paths, *, device=None, locate=True, should_cancel=None,
                      cancel_file=None, info=None):
            for path in paths:
                state, score, regions = child_rows(path)
                yield (path, state, score, regions,
                       transfer.content_fingerprint_path(path), None)
        monkeypatch.setattr(watermark_detector, 'scan', fake_scan)
        r = client.post(f'/api/bank/{bank_id}/watermark', json={'rescan': True})
        assert r.status_code == 202, r.get_json()

    def test_none_verdict_keeps_text_zones_flagged(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        image_id = self._text_flag(app, bank_id)
        self._run_detector_scan(client, bank_id, monkeypatch,
                                lambda path: ('none', 0.05, []))
        page = _rows(app, bank_id)['page.jpg']
        assert page['id'] == image_id
        assert page['watermark_state'] == 'detected'     # still repaintable
        assert json.loads(page['watermark_regions']) == [[0.28, 0.09, 0.72, 0.22]]

    def test_found_box_folds_into_text_zones(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['page.jpg'])
        self._text_flag(app, bank_id)
        self._run_detector_scan(
            client, bank_id, monkeypatch,
            lambda path: ('detected', 0.97, [[0.02, 0.9, 0.12, 0.98]]))
        page = _rows(app, bank_id)['page.jpg']
        assert page['watermark_state'] == 'detected'
        regions = json.loads(page['watermark_regions'])
        assert [0.28, 0.09, 0.72, 0.22] in regions       # text zone survived
        assert any(b[1] >= 0.8 for b in regions)         # the box joined it
        assert page['watermark_bbox'] is not None
