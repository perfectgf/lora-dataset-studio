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

import pytest
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

    def fake(frames, *, timeout=None, should_stop=None, on_progress=None):
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

    def test_unreached_frames_stay_unscanned(
            self, app, client, tmp_path, monkeypatch):
        bank_id, _src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
        _ocr_ready(monkeypatch)
        _fake_reader(monkeypatch, {'a.jpg': TWO_LINES, 'b.jpg': TWO_LINES},
                     reachable={'a.jpg'})
        assert client.post(f'/api/bank/{bank_id}/text', json={}).status_code == 202
        rows = _rows(app, bank_id)
        assert rows['a.jpg']['text_state'] == 'detected'
        assert rows['b.jpg']['text_state'] is None     # absent key = never read

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
