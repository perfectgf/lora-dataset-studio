"""The captioner must not refuse images this install can import, view and train on.

The app's image INPUT budget became configurable (`image_input.max_side` /
`image_input.max_pixels`, shipped 64 Mi-pixels / 16384 px) precisely so a 24 MP
camera master would be accepted everywhere. `backend/infer/bank_image_guard.py`
was left behind at the OLD fixed 16 Mi-pixels / 8192 px, and it sits on the
CAPTIONING path (`joycaption_infer.py` reads every image through it). Measured on
a real run: 52 of 89 images refused with "bank image rejects images above 8192 px
per side or 16777216 pixels (got 3936x5905)", the batch finishing normally and
reporting only the 37 captions it wrote.

Two independent defects, one per half of this file:
  * the worker enforced a budget nobody configured;
  * a refused image was HANDLED but never counted, so the progress indicator
    stopped short of the total and the result named no reason.
"""
from __future__ import annotations

import importlib.util
import pathlib
import struct
import sys

import pytest


INFER = pathlib.Path(__file__).resolve().parents[1] / 'infer'


def _guard():
    """Load the guard the way its ML venv does: standalone, by path, no app."""
    if str(INFER) not in sys.path:
        sys.path.insert(0, str(INFER))
    spec = importlib.util.spec_from_file_location(
        'bank_image_guard_budget_test', INFER / 'bank_image_guard.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _compact_bmp(path: pathlib.Path, width: int, height: int) -> None:
    """A header-only BMP: tiny on disk, but it advertises the supplied raster —
    so the size checks run on the real numbers with no 22 Mi-pixel allocation."""
    file_header = struct.pack('<2sIHHI', b'BM', 54, 0, 0, 54)
    dib_header = struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0,
                             0, 0, 0, 0, 0)
    path.write_bytes(file_header + dib_header)


# One of the refused images from the reported run, and the arithmetic that
# matters here: 22 118 400 pixels
# is over the old 16 Mi-pixel SURFACE budget while both sides sit far under 8192,
# so it is the surface that refuses it — not the side, despite the message being
# read that way.
OVERSIZED = (3840, 5760)


def test_the_reported_image_is_over_the_surface_budget_not_the_side():
    guard = _guard()
    width, height = OVERSIZED
    assert width <= guard.MAX_SIDE and height <= guard.MAX_SIDE
    assert width * height > guard.MAX_PIXELS


def test_worker_without_a_parent_budget_keeps_its_shipped_ceiling(tmp_path, monkeypatch):
    """No environment = no app to ask: the conservative constants still apply."""
    monkeypatch.delenv('LDS_INFER_MAX_SIDE', raising=False)
    monkeypatch.delenv('LDS_INFER_MAX_PIXELS', raising=False)
    guard = _guard()
    path = tmp_path / 'big.bmp'
    _compact_bmp(path, *OVERSIZED)
    with pytest.raises(guard.BankImageGuardError, match='above 8192 px'):
        guard.read_validated_bank_image(str(path))


def test_worker_honours_the_budget_its_parent_configured(tmp_path, monkeypatch):
    """THE regression: an image the app's own budget accepts must reach the model."""
    guard = _guard()
    path = tmp_path / 'big.bmp'
    _compact_bmp(path, *OVERSIZED)
    monkeypatch.setenv('LDS_INFER_MAX_SIDE', '16384')
    monkeypatch.setenv('LDS_INFER_MAX_PIXELS', str(64 * 1024 * 1024))
    assert guard.read_validated_bank_image(str(path))
    assert guard.effective_limits() == (16384, 64 * 1024 * 1024)


def test_a_budget_of_zero_means_no_limit_on_that_dimension(tmp_path, monkeypatch):
    """0 is the app's documented "no limit", including past Pillow's own bomb
    threshold — which would otherwise refuse with a different message."""
    guard = _guard()
    path = tmp_path / 'huge.bmp'
    _compact_bmp(path, 20000, 20000)      # 400 Mi-pixels, over every default
    monkeypatch.setenv('LDS_INFER_MAX_SIDE', '0')
    monkeypatch.setenv('LDS_INFER_MAX_PIXELS', '0')
    assert guard.read_validated_bank_image(str(path))


def test_pillow_bomb_threshold_is_restored_after_a_raised_budget(tmp_path, monkeypatch):
    """The threshold is process-global; another task in this interpreter must not
    inherit a guard that was widened for one image."""
    from PIL import Image
    guard = _guard()
    path = tmp_path / 'huge.bmp'
    _compact_bmp(path, 20000, 20000)
    monkeypatch.setenv('LDS_INFER_MAX_SIDE', '0')
    monkeypatch.setenv('LDS_INFER_MAX_PIXELS', '0')
    before = Image.MAX_IMAGE_PIXELS
    guard.read_validated_bank_image(str(path))
    assert Image.MAX_IMAGE_PIXELS == before


@pytest.mark.parametrize('bogus', ['', '   ', 'lots', '-1', '1e9'])
def test_an_unusable_budget_falls_back_to_the_ceiling_not_to_no_limit(
        tmp_path, monkeypatch, bogus):
    """A typo in an environment variable must never silently disarm the guard."""
    guard = _guard()
    path = tmp_path / 'big.bmp'
    _compact_bmp(path, *OVERSIZED)
    monkeypatch.setenv('LDS_INFER_MAX_SIDE', bogus)
    monkeypatch.setenv('LDS_INFER_MAX_PIXELS', bogus)
    with pytest.raises(guard.BankImageGuardError, match='above 8192 px'):
        guard.read_validated_bank_image(str(path))


def test_the_message_names_only_the_limits_that_are_armed(tmp_path, monkeypatch):
    guard = _guard()
    path = tmp_path / 'big.bmp'
    _compact_bmp(path, *OVERSIZED)
    monkeypatch.setenv('LDS_INFER_MAX_SIDE', '0')          # side disarmed
    monkeypatch.setenv('LDS_INFER_MAX_PIXELS', str(16 * 1024 * 1024))
    with pytest.raises(guard.BankImageGuardError) as excinfo:
        guard.read_validated_bank_image(str(path))
    message = str(excinfo.value)
    assert '16777216 pixels' in message and 'px per side' not in message
    assert '3840x5760' in message


# --- the app side: the budget has to actually reach the subprocess -----------

def test_joycaption_subprocess_carries_the_configured_budget(app, monkeypatch):
    """Without this the two halves disagree in silence: the app accepts the image,
    the worker refuses it, and nothing connects the two facts."""
    from app.services import joycaption
    from app.services.input_budget import input_image_budget

    monkeypatch.setattr(joycaption, 'is_available', lambda: True)
    captured = {}

    class _Boom(OSError):
        pass

    def fake_popen(*args, **kwargs):
        captured['env'] = kwargs.get('env') or {}
        raise _Boom('not actually launching a model here')

    monkeypatch.setattr(joycaption.subprocess, 'Popen', fake_popen)
    with app.app_context():
        max_side, max_pixels = input_image_budget()
        assert joycaption.caption_images_joycaption([__file__]) == {}

    assert captured['env']['LDS_INFER_MAX_SIDE'] == str(max_side)
    assert captured['env']['LDS_INFER_MAX_PIXELS'] == str(max_pixels)


# --- a refused image is HANDLED: count it, and say why ----------------------

REFUSAL = ('bank image rejects images above 8192 px per side or 16777216 pixels '
           '(got 3840x5760)')


class _FakePopen:
    """Enough of Popen for the drain threads: one refusal on stdout, nothing else."""
    def __init__(self, stdout_text):
        import io as _io
        self.stdout = _io.StringIO(stdout_text)
        self.stderr = _io.StringIO('')
        self.stdin = _io.StringIO()
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def test_joycaption_hands_back_the_reason_an_image_was_refused(app, monkeypatch, tmp_path):
    """The worker already reports per-image errors on stdout; they used to die in
    the service, which returned captions only."""
    import json
    import app.services.joycaption as jc
    from PIL import Image

    image = tmp_path / 'a.webp'
    Image.new('RGB', (16, 12), (20, 30, 40)).save(image, 'WEBP')
    monkeypatch.setattr(jc, 'is_available', lambda: True)
    monkeypatch.setattr(jc.cfg, 'aitoolkit_path', lambda k: str(tmp_path / str(k)))
    stdout_text = json.dumps({'i': 1, 'path': str(image), 'error': REFUSAL}) + '\n'
    monkeypatch.setattr(jc.subprocess, 'Popen', lambda *a, **k: _FakePopen(stdout_text))

    errors: dict = {}
    with app.app_context():
        assert jc.caption_images_joycaption([str(image)], errors_out=errors) == {}
    assert errors == {str(image): REFUSAL}


def _dataset_with_kept_images(svc, local_user, count):
    """A real dataset with `count` kept images, each backed by a file on disk."""
    import io as _io
    import os
    from PIL import Image
    from app.models import FaceDatasetImage

    buf = _io.BytesIO()
    Image.new('RGB', (16, 12), (20, 30, 40)).save(buf, 'PNG')
    dataset = svc.create_dataset(local_user, 'SkipTest', 'skiptest')
    directory = svc._dataset_dir(dataset.id)
    os.makedirs(directory, exist_ok=True)
    for index in range(count):
        name = f'kept{index}.webp'
        with open(os.path.join(directory, name), 'wb') as handle:
            handle.write(buf.getvalue())
        svc.db.session.add(FaceDatasetImage(
            dataset_id=dataset.id, source='import', status='keep',
            filename=name, framing='face'))
    svc.db.session.commit()
    return dataset


def test_forced_joycaption_pass_counts_and_explains_what_it_refused(app, monkeypatch):
    """The second half of the reported symptom: with JoyCaption forced there is no
    Ollama phase behind it, so an image it refused is FINISHED. Leaving it
    uncounted froze the indicator short of the total ("37/89") on a pass that had
    actually run to the end, and the result named neither the number nor a reason.
    """
    from app.services import face_dataset_service as svc
    from app.services import dataset_activity as da
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config

    final = {}
    real_end = da.end

    def _capture_end(token):
        entry = da.get(da._dsid_of(token))
        if entry:
            final.update(entry)
        return real_end(token)

    monkeypatch.setattr(da, 'end', _capture_end)
    outcome: dict = {}
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})
        dataset = _dataset_with_kept_images(svc, LOCAL_USER, 3)
        monkeypatch.setattr(jc_mod, 'is_available', lambda: True)

        def _caption(paths, errors_out=None, **kwargs):
            if errors_out is not None:
                for refused in paths[1:]:
                    errors_out[refused] = REFUSAL
            return {paths[0]: 'a joycaption result'}

        monkeypatch.setattr(jc_mod, 'caption_images_joycaption', _caption)
        assert svc.caption_images(LOCAL_USER, dataset.id, outcome=outcome) == 1

    # The pass reached the end of its own list — no image left silently pending.
    assert final.get('done') == final.get('total') == 3
    assert outcome == {'skipped': 2, 'skipped_reason': REFUSAL}


def test_a_pass_that_refused_nothing_reports_no_skips(app, monkeypatch):
    """The clause has to be absent, not empty: "0 skipped" on a clean run is noise
    the user would learn to ignore."""
    from app.services import face_dataset_service as svc
    import app.services.joycaption as jc_mod
    from app.config import LOCAL_USER, save_config

    outcome: dict = {}
    with app.app_context():
        save_config({'captioning': {'backend': 'joycaption'}})
        dataset = _dataset_with_kept_images(svc, LOCAL_USER, 2)
        monkeypatch.setattr(jc_mod, 'is_available', lambda: True)
        monkeypatch.setattr(
            jc_mod, 'caption_images_joycaption',
            lambda paths, errors_out=None, **kw: {p: 'a caption' for p in paths})
        assert svc.caption_images(LOCAL_USER, dataset.id, outcome=outcome) == 2
    assert outcome == {}
