"""Three findings from ONE real scan session, each with the fix that answers it.

The maintainer ran the new engine selector on a two-image test dataset (stock
photos tiled wall-to-wall with watermarks) and sent three screenshots:

1. vision engine → "0 watermark(s) found · 0 clean (of 0)" under a GREEN tick —
   the vision model was down, every call came back empty, and the loop skipped
   both images in silence. The bank's twin loop has counted this for a while;
   the dataset's never got the port.
2. detector @ 0.50 → flagged, but with ONE small corner box on marks tiled
   across the WHOLE frame — the per-box area cap (a deliberate guard against
   repainting an entire image) silently reduced "everywhere" to "one tile".
3. detector @ 0.99 → both ruled clean, with nothing on screen saying the top
   score was just under the bar.
"""
import json

import pytest

from app import config
from app.models import FaceDatasetImage, db
from app.services import face_dataset_service as svc
from app.services import watermark_detector


def _dataset_with_images(app, n=2):
    with app.app_context():
        ds = svc.create_dataset('local', f'wm-honesty-{n}', 'wmtrig')
        ids = []
        for i in range(n):
            img = FaceDatasetImage(dataset_id=ds.id, filename=f'i{i}.png',
                                   status='keep')
            db.session.add(img)
            db.session.commit()
            ids.append(img.id)
        return ds.id, ids


# --- 1. the silent-empty vision scan ----------------------------------------

def test_a_vision_scan_nobody_answered_says_so_instead_of_green_zero(
        app, client, monkeypatch, tmp_path):
    dataset_id, _ = _dataset_with_images(app)
    monkeypatch.setattr(svc, '_img_path', lambda img: str(tmp_path / 'x.png'))
    (tmp_path / 'x.png').write_bytes(b'fake')
    import os as _os
    monkeypatch.setattr(svc.os.path, 'exists', _os.path.exists)

    from app.services import vision_llm
    monkeypatch.setattr(vision_llm, 'describe_image', lambda *a, **k: '')
    monkeypatch.setattr(vision_llm, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'watermark_detect': {'backend': 'vision'}})

    r = client.post(f'/api/dataset/{dataset_id}/watermarks/detect', json={})
    d = r.get_json()
    assert r.status_code == 200 and d['ok'] is True
    assert d['checked'] == 0 and d['detected'] == 0
    assert d['unanswered'] == 2, 'the skipped-in-silence count the bank has had for a while'
    note = d['unanswered_note']
    assert 'no answer' in note and 'LM Studio' in note, (
        'the note must name the provider the user actually runs')


def test_a_partially_answered_vision_scan_counts_both_truths(app, client,
                                                             monkeypatch, tmp_path):
    dataset_id, _ = _dataset_with_images(app)
    monkeypatch.setattr(svc, '_img_path', lambda img: str(tmp_path / 'x.png'))
    (tmp_path / 'x.png').write_bytes(b'fake')

    from app.services import vision_llm
    answers = iter(['', json.dumps({'present': True, 'y1': 100, 'x1': 100,
                                    'y2': 300, 'x2': 300})])
    monkeypatch.setattr(vision_llm, 'describe_image',
                        lambda *a, **k: next(answers, ''))
    monkeypatch.setattr(vision_llm, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        config.save_config({'watermark_detect': {'backend': 'vision'}})

    d = client.post(f'/api/dataset/{dataset_id}/watermarks/detect', json={}).get_json()
    assert d['checked'] == 1 and d['detected'] == 1
    assert d['unanswered'] == 1


# --- 2. wall-to-wall marks stop shrinking to one tile ------------------------

def _infer_module():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'infer', 'watermark_detect_infer.py')
    spec = importlib.util.spec_from_file_location('watermark_detect_infer_h', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_wall_to_wall_claim_with_almost_nothing_located_stays_unlocated():
    """The original lie-by-omission, still guarded: raw boxes claim most of the
    frame (giant "text overlay" matches, dropped by the per-box cap) but the
    locator pinned under WALL_TO_WALL_MIN_ZONES actual tiles. Reporting the one
    surviving tile would read as "handled"; [] routes to 🔍 Review instead."""
    infer = _infer_module()
    claim = [[0, 0, 900, 800], [10, 15, 260, 135]]   # frame-sized junk + 1 tile
    assert infer._raw_coverage(claim, (1000, 1000)) > infer.WALL_TO_WALL_COVERAGE
    assert infer.effective_regions(claim, (1000, 1000)) == []


def test_a_well_located_tiling_reports_its_zones_instead_of_hiding_them():
    """The 2026-08-31 reversal, by the maintainer's mandate: when the sweep DID
    pin the tiles (the real stock photo localises 12 of ~25), those zones are
    the honest report — twelve boxes across the frame say "it is everywhere"
    better than zero did. The old rule blanked ANY coverage above the guard."""
    infer = _infer_module()
    tiles = [[x, y, x + 250, y + 120]
             for x in range(0, 1000, 250) for y in range(0, 1000, 200)]
    assert infer._raw_coverage(tiles, (1000, 1000)) > infer.WALL_TO_WALL_COVERAGE
    out = infer.effective_regions(tiles, (1000, 1000))
    assert len(out) >= infer.WALL_TO_WALL_MIN_ZONES, (
        'located tiles were hidden behind the wall-to-wall guard again')


def test_a_corner_mark_keeps_its_zone_exactly_as_before(app=None):
    infer = _infer_module()
    corner = [[700, 850, 950, 980]]
    out = infer.effective_regions(corner, (1000, 1000))
    assert out == [[0.7, 0.85, 0.95, 0.98]]


def test_the_coverage_is_bounded_and_junk_proof():
    infer = _infer_module()
    assert infer._raw_coverage([], (1000, 1000)) == 0.0
    assert infer._raw_coverage([[0, 0, 1000, 1000]] * 9, (1000, 1000)) == 1.0
    assert infer._raw_coverage([['x', 1, 2, 3], None], (1000, 1000)) == 0.0
    assert infer._raw_coverage([[0, 0, 10, 10]], (0, 0)) == 0.0


def test_the_locator_actually_routes_through_the_wall_to_wall_rule():
    """Source-as-text, because the real DINO cannot run here: the pure rule
    above is only worth its tests if regions() calls it. Reverting regions() to
    the bare merge would keep every pure test green over the original bug."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'infer', 'watermark_detect_infer.py')
    src = open(path, encoding='utf-8').read()
    assert ('return effective_regions(base, image.size, '
            'validate_boxes=validate)') in src, (
        'regions() no longer routes through the consensus + wall-to-wall decision')


# --- 3. zero flagged names the near-miss -------------------------------------

def test_zero_flagged_reports_the_highest_clean_score(app, client, monkeypatch,
                                                      tmp_path):
    dataset_id, _ = _dataset_with_images(app)
    monkeypatch.setattr(svc, '_img_path', lambda img: str(tmp_path / 'x.png'))
    (tmp_path / 'x.png').write_bytes(b'fake')

    def _fake_scan(paths, **kw):
        for i, p in enumerate(paths):
            yield p, 'none', 0.62 + i * 0.26, [], None, None   # best = 0.88

    monkeypatch.setattr(watermark_detector, 'scan', _fake_scan)
    monkeypatch.setattr(watermark_detector, 'resolve_backend',
                        lambda requested=None: {'requested': 'detector',
                                                'backend': 'detector',
                                                'fell_back': False, 'detail': ''})
    d = client.post(f'/api/dataset/{dataset_id}/watermarks/detect', json={}).get_json()
    assert d['detected'] == 0 and d['none'] == 2
    assert d['top_clean_score'] == pytest.approx(0.88), (
        'the one number separating "nothing there" from "everything under the bar"')
