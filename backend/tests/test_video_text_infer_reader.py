"""video_text_infer._read_bgr — the reader behind both text lanes.

The defect this pins: cv2.imread on Windows cannot open a non-ASCII path at
all (a findDecoder warning and None), and a bank living in an accented folder
fed the OCR child 81 files it silently could not read. The reader now falls
back to unicode-safe routes; this test walks the exact failing shape.

Skips without cv2/PIL/numpy in the test interpreter — the reader's callers are
covered by the lane tests either way; this one exercises the real decoders.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_INFER = Path(__file__).resolve().parents[1] / 'infer' / 'video_text_infer.py'


def _load_module():
    # By path, not by package: backend/infer is a script directory, and the
    # module imports _harness from its own folder.
    sys.path.insert(0, str(_INFER.parent))
    try:
        spec = importlib.util.spec_from_file_location('video_text_infer_ut', _INFER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(_INFER.parent))


def test_read_bgr_opens_a_non_ascii_path(tmp_path):
    pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    Image = pytest.importorskip('PIL.Image')
    folder = tmp_path / 'bank générée é'
    folder.mkdir()
    path = folder / 'photo café.jpg'
    Image.new('RGB', (60, 40), (10, 200, 30)).save(str(path), 'JPEG')

    mod = _load_module()
    image = mod._read_bgr(str(path))
    assert image is not None, 'accented path must be readable'
    assert image.shape[0] == 40 and image.shape[1] == 60
    # BGR, whichever route answered: the green pixel lands in channel 1.
    assert int(np.asarray(image)[20, 30, 1]) > 150


def test_read_bgr_answers_none_on_a_missing_file(tmp_path):
    pytest.importorskip('cv2')
    mod = _load_module()
    assert mod._read_bgr(str(tmp_path / 'nöpe.jpg')) is None
