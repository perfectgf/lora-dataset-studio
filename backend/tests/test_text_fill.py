"""text_fill_infer — the bubble-aware filler, exercised for real.

Skips without cv2/numpy in the test interpreter; the service-level grafts are
covered with the seam monkeypatched in the bank/dataset clean tests either
way. Every case here draws its own synthetic page, so the assertions measure
pixels, not opinions — and every path is non-ASCII, the exact Windows failure
mode this lane was bitten by once already.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

_INFER = Path(__file__).resolve().parents[1] / 'infer' / 'text_fill_infer.py'


def _run_child(payload):
    proc = subprocess.run(
        [sys.executable, str(_INFER)], input=json.dumps(payload),
        capture_output=True, text=True, encoding='utf-8', timeout=120)
    line = next((ln for ln in reversed((proc.stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    assert line, f'no JSON from child, stderr: {proc.stderr[-400:]}'
    return json.loads(line)


def _bubble_page(cv2, np, path):
    """A page with an outlined balloon and two text lines; the zone used in
    the tests deliberately OVERSHOOTS the outline, like real merged zones."""
    img = np.full((400, 400, 3), 230, np.uint8)
    cv2.ellipse(img, (200, 200), (150, 100), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (200, 200), (150, 100), 0, 0, 360, (0, 0, 0), 5)
    cv2.putText(img, 'HELLO', (120, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, 'WORLD', (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.imencode('.png', img)[1].tofile(str(path))
    return img


ZONE = [60 / 400, 110 / 400, 350 / 400, 280 / 400]


def test_bubble_is_emptied_and_the_outline_survives(tmp_path):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    page = tmp_path / 'bulle é' / 'page café.png'
    page.parent.mkdir()
    _bubble_page(cv2, np, page)
    out = _run_child({'items': [{'image_path': str(page), 'regions': [ZONE]}]})
    row = out['results'][str(page)]
    assert row['ok'] and row['filled'] == 1 and row['busy_boxes'] == []
    after = cv2.imdecode(np.fromfile(str(page), dtype=np.uint8), cv2.IMREAD_COLOR)
    # Text gone: nothing dark left where the lines were…
    assert int(after[150:250, 110:290].min()) >= 250
    # …and the outline arc above the text is still solid black.
    assert int(after[95:105, 190:210].min()) < 60


def test_busy_background_returns_merged_glyph_boxes_and_paints_nothing(tmp_path):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (300, 300, 3), np.uint8)
    cv2.putText(img, 'SFX', (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (5, 5, 5), 7)
    page = tmp_path / 'décor' / 'sfx.png'
    page.parent.mkdir()
    cv2.imencode('.png', img)[1].tofile(str(page))
    out = _run_child({'items': [{'image_path': str(page),
                                 'regions': [[0.1, 0.3, 0.9, 0.65]]}]})
    row = out['results'][str(page)]
    assert row['ok'] and row['filled'] == 0
    # Merged and capped: a handful of boxes, never a shotgun of grain
    # (pre-guard this exact page produced 495 one-pixel boxes).
    assert 1 <= len(row['busy_boxes']) <= 24
    after = cv2.imdecode(np.fromfile(str(page), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert (after == img).all(), 'busy path must not paint'


def test_light_on_dark_lettering_is_filled_too(tmp_path):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    img = np.full((300, 300, 3), 20, np.uint8)          # black balloon
    cv2.putText(img, 'BOO', (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255, 255, 255), 6)
    page = tmp_path / 'noir' / 'inversé.png'
    page.parent.mkdir()
    cv2.imencode('.png', img)[1].tofile(str(page))
    out = _run_child({'items': [{'image_path': str(page),
                                 'regions': [[0.15, 0.35, 0.85, 0.65]]}]})
    row = out['results'][str(page)]
    assert row['ok'] and row['filled'] == 1
    after = cv2.imdecode(np.fromfile(str(page), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert int(after[110:180, 60:250].max()) <= 30, 'white letters must be gone'


def test_unreadable_item_fails_alone_not_the_batch(tmp_path):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    good = tmp_path / 'ok.png'
    _bubble_page(cv2, np, good)
    out = _run_child({'items': [
        {'image_path': str(tmp_path / 'absent é.png'), 'regions': [ZONE]},
        {'image_path': str(good), 'regions': [ZONE]}]})
    bad_row = out['results'][str(tmp_path / 'absent é.png')]
    assert bad_row['ok'] is False and 'unreadable' in bad_row['error']
    assert out['results'][str(good)]['ok'] is True


def test_probe_covers_everything_the_filler_imports():
    """Issue #24's rule, applied to this worker: its runtime imports must all
    be in the video_text capability probe it is gated on."""
    import re
    from app import capabilities
    src = _INFER.read_text(encoding='utf-8')
    imported = set(re.findall(r'^\s*(?:import|from)\s+([A-Za-z_][\w]*)', src, re.M))
    imported -= {'__future__', 'json', 'os', 'sys', '_harness'}
    probe = capabilities.CAPABILITY_IMPORTS['video_text']
    for module in imported:
        assert module in probe, \
            f'the filler imports {module} and the video_text probe does not'
