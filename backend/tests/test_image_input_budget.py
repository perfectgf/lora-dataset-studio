"""The image input budget is ONE configurable number, seen by every lane.

Two properties are pinned here, and they pull in opposite directions:

* it is a **setting** — a panorama refused at 8192 px must become importable by
  changing a value, including all the way to "no limit" (0);
* it stays **global** — the guard's own comment warns that a Bank scan,
  thumbnail or edit must not refuse what Dataset import accepted. A budget that
  only the import route honours would manufacture an image you can import and
  not look at, which is worse than the limit it replaced.

Plus the structural constraint: ``app/services/image_encoding.py`` is loaded BY
PATH under the ML interpreter (``backend/infer/lama_infer.py``), where the app
package does not exist. It may therefore never read config — the app injects a
provider instead. The last test loads that module the way the ML interpreter
does and proves it still imports and still guards.
"""
from __future__ import annotations

import importlib.util
import io
import pathlib

import pytest
from PIL import Image

from app.config import save_config


ENCODING_PATH = (pathlib.Path(__file__).resolve().parents[1]
                 / 'app' / 'services' / 'image_encoding.py')


class _Header:
    """A Pillow-like header. `load` must never be reached once it is refused."""
    format = 'JPEG'
    n_frames = 1

    def __init__(self, width, height):
        self.size = (width, height)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def load(self):
        raise AssertionError('a refused header was decoded anyway')

    def draft(self, *_args):
        raise AssertionError('a refused header reached draft')


# The image from the report: a legitimate panorama, over BOTH old bounds
# (10418 > 8192 px per side, and 21.9 Mi-pixels > 16 Mi).
PANORAMA = (10418, 2100)


def test_the_shipped_default_admits_a_panorama_and_a_modern_camera_master(app):
    from app.services import image_encoding as ie

    with app.app_context():
        assert ie.input_budget() == (16384, 64 * 1024 * 1024)
        assert ie.validate_input_header_dimensions(
            _Header(*PANORAMA), label='import') == PANORAMA
        # 61 MP full-frame master (Sony A7R V geometry): 57 Mi-pixels.
        assert ie.validate_input_header_dimensions(
            _Header(9504, 6336), label='import') == (9504, 6336)
        # ...and the budget is still a budget: 268 Mi-pixels is refused.
        with pytest.raises(ValueError):
            ie.validate_input_header_dimensions(_Header(16384, 16384), label='import')


def test_the_refusal_says_where_to_change_the_budget(app):
    """'reduce the image' was the only honest advice while the number was welded
    in. Now that it is not, a message that only says that is a dead end."""
    from app.services import image_encoding as ie

    with app.app_context():
        with pytest.raises(ValueError) as excinfo:
            ie.validate_input_header_dimensions(_Header(40000, 40000), label='import')
    message = str(excinfo.value)
    assert 'Settings' in message and 'Image size budget' in message
    assert '40000x40000' in message                     # what was actually seen
    assert 'reduce the image before import' in message  # the other way out remains


def test_zero_means_no_limit_on_each_axis_independently(app):
    from app.services import image_encoding as ie

    with app.app_context():
        save_config({'image_input': {'max_side': 0, 'max_pixels': 0}})
        assert ie.input_budget() == (0, 0)
        assert ie.input_budget_sentence() == 'any size (no limit)'
        # A raster no bounded budget would ever have admitted.
        assert ie.validate_input_header_dimensions(
            _Header(200000, 200000), label='import') == (200000, 200000)
        # Pillow's own bomb threshold must not quietly overrule the choice:
        # several call sites promote DecompressionBombWarning to an exception.
        assert Image.MAX_IMAGE_PIXELS is None

        # Only the pixel budget lifted: the side limit still bites, and alone.
        save_config({'image_input': {'max_side': 8192, 'max_pixels': 0}})
        assert ie.input_budget() == (8192, 0)
        assert ie.input_budget_sentence() == '8192 px per side'
        assert ie.validate_input_header_dimensions(
            _Header(8192, 8192), label='import') == (8192, 8192)   # 67 Mi-pixels, allowed
        with pytest.raises(ValueError, match='8192 px per side'):
            ie.validate_input_header_dimensions(_Header(8193, 10), label='import')

        # Only the side limit lifted.
        save_config({'image_input': {'max_side': 0, 'max_pixels': 16 * 1024 * 1024}})
        assert ie.validate_input_header_dimensions(
            _Header(100000, 100), label='import') == (100000, 100)
        with pytest.raises(ValueError, match='pixels'):
            ie.validate_input_header_dimensions(_Header(8192, 8192), label='import')


def test_a_zero_size_header_is_still_refused_when_the_budget_is_unlimited(app):
    """'No limit' removes a budget, not the sanity check: a 0x0 or non-integer
    header has no honest decode, whatever the setting says."""
    from app.services import image_encoding as ie

    with app.app_context():
        save_config({'image_input': {'max_side': 0, 'max_pixels': 0}})
        with pytest.raises(ValueError, match='unreadable image'):
            ie.validate_input_header_dimensions(_Header(0, 100), label='import')
        with pytest.raises(ValueError, match='unreadable image'):
            ie.validate_input_header_dimensions(_Header('wide', 100), label='import')


def test_an_unusable_configured_value_degrades_to_the_shipped_default(app):
    from app.services import image_encoding as ie

    with app.app_context():
        save_config({'image_input': {'max_side': 'huge', 'max_pixels': -1}})
        assert ie.input_budget() == (16384, 64 * 1024 * 1024)


def test_every_app_side_consumer_sees_the_SAME_raised_budget(app, monkeypatch, tmp_path):
    """The trap this whole change had to avoid: import accepts, a thumbnail or a
    vision pass refuses, and the user owns an image the app cannot show.

    Each consumer below is exercised on the SAME panorama header, once under the
    old fixed budget (all refuse) and once under a raised one (all accept)."""
    from app.services import face_dataset_service as fds
    from app.services import image_encoding as ie
    from app.services import vision_ollama
    from app.utils import comfy_fs

    # A REAL panorama-shaped PNG (9000x2000 = 17.2 Mi-pixels, over BOTH old
    # bounds): the vision lane imports PIL inside its own function, so it is
    # exercised with genuine bytes rather than a stub header.
    buffer = io.BytesIO()
    Image.new('RGB', (9000, 2000), (30, 60, 90)).save(buffer, 'PNG')
    real_png = buffer.getvalue()

    def bank_analysis_allows():
        return fds._bank_analysis_dimensions_allowed(_Header(*PANORAMA))

    def comfy_staging_allows(tmp_dir):
        """True when ComfyUI staging got PAST the header guard (our fake header
        then raises from load(), which is the proof it was admitted)."""
        # `comfy_fs.Image` IS the PIL module, so this patch is process-wide:
        # restore it before the next lane is measured, or that lane is measured
        # against this stub instead of its own input.
        real_open = comfy_fs.Image.open
        comfy_fs.Image.open = lambda *_a, **_k: _Header(*PANORAMA)
        try:
            comfy_fs.stage_input_image('src.png', 'dest.png', str(tmp_dir))
        except AssertionError:
            return True
        except ValueError as exc:
            assert 'Image size budget' in str(exc), exc
            return False
        finally:
            comfy_fs.Image.open = real_open
        return True

    def vision_allows():
        # Fails closed by contract (returns None) rather than raising, so the
        # verdict is the return value.
        return vision_ollama._ensure_ollama_decodable(real_png) is not None

    with app.app_context():
        save_config({'image_input': {'max_side': 8192,
                                     'max_pixels': 16 * 1024 * 1024}})
        assert ie.input_budget() == (8192, 16 * 1024 * 1024)
        assert bank_analysis_allows() is False
        assert comfy_staging_allows(tmp_path) is False
        assert vision_allows() is False
        assert fds.preserved_import_limits() == (8192, 16 * 1024 * 1024)
        policy = fds.import_encode_policy()
        assert policy['input_max_side'] == 8192
        assert policy['input_max_pixels'] == 16 * 1024 * 1024
        assert policy['preserve_max_side'] == 8192      # legacy alias moves too

        save_config({'image_input': {'max_side': 16384,
                                     'max_pixels': 64 * 1024 * 1024}})
        assert bank_analysis_allows() is True
        assert comfy_staging_allows(tmp_path) is True
        assert vision_allows() is True
        assert fds.preserved_import_limits() == (16384, 64 * 1024 * 1024)
        policy = fds.import_encode_policy()
        assert policy['input_max_side'] == 16384
        assert policy['preserve_max_pixels'] == 64 * 1024 * 1024
        # ...and the WebP NORMALISATION ceiling deliberately does NOT move: it
        # bounds what a WebP mode writes, not what may be decoded.
        assert policy['ceiling'] == 8192


def test_the_webp_ceiling_is_not_the_input_budget(app):
    from app.services import face_dataset_service as fds

    with app.app_context():
        save_config({'image_input': {'max_side': 0, 'max_pixels': 0}})
        assert fds.IMPORT_MAX_SIDE_CEILING == 8192
        assert fds.import_encode_policy()['ceiling'] == 8192


def test_the_capabilities_payload_publishes_the_configured_budget(client):
    assert client.put('/api/settings', json={
        'config': {'image_input': {'max_side': 32768,
                                   'max_pixels': 128 * 1024 * 1024}}}).status_code == 200
    capability = client.get('/api/capabilities').get_json()['dataset_import']
    assert capability['input_max_side'] == 32768
    assert capability['input_max_pixels'] == 128 * 1024 * 1024
    # The names released before this policy described anything but preserve.
    assert capability['preserve_max_side'] == 32768
    assert capability['preserve_max_pixels'] == 128 * 1024 * 1024


def test_the_budget_is_read_live_not_snapshotted_at_import(app):
    """A module-level snapshot would freeze the first value the process saw, and
    the symptom is the worst kind: a saved setting that only works after a
    restart, silently."""
    from app.services import image_encoding as ie

    with app.app_context():
        save_config({'image_input': {'max_side': 4096, 'max_pixels': 4 * 1024 * 1024}})
        with pytest.raises(ValueError):
            ie.validate_input_header_dimensions(_Header(*PANORAMA), label='import')
        save_config({'image_input': {'max_side': 32768, 'max_pixels': 128 * 1024 * 1024}})
        assert ie.validate_input_header_dimensions(
            _Header(*PANORAMA), label='import') == PANORAMA


def test_image_encoding_still_loads_by_path_with_no_app_and_keeps_the_defaults():
    """How `backend/infer/lama_infer.py` gets it: it doesn't, and that is stated.

    The ML interpreter loads this file directly, without the Flask app or its
    config on sys.path. It must import (a `cfg.get()` in there would be an
    ImportError at inference time) and it must still guard — with the shipped
    defaults, the conservative side of the split."""
    spec = importlib.util.spec_from_file_location('lds_image_encoding_probe',
                                                  ENCODING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.input_budget() == (module.DEFAULT_INPUT_MAX_SIDE,
                                     module.DEFAULT_INPUT_MAX_PIXELS)
    assert module.validate_input_header_dimensions(
        _Header(*PANORAMA), label='inpainting') == PANORAMA
    with pytest.raises(ValueError):
        module.validate_input_header_dimensions(_Header(40000, 40000), label='inpainting')
    # And it can be pointed at any budget by its own caller, without config.
    assert module.validate_input_header_dimensions(
        _Header(40000, 40000), label='inpainting', max_side=0, max_pixels=0) == (40000, 40000)


def test_a_broken_budget_provider_never_takes_the_app_down(app):
    """Total by construction: every image path runs through this, so a provider
    that raises must degrade to the defaults rather than refuse everything."""
    from app.services import image_encoding as ie

    previous = ie._budget_provider
    try:
        ie.set_input_budget_provider(lambda: (_ for _ in ()).throw(RuntimeError('boom')))
        assert ie.input_budget() == (16384, 64 * 1024 * 1024)
        ie.set_input_budget_provider(lambda: ('wide', None))
        assert ie.input_budget() == (16384, 64 * 1024 * 1024)
    finally:
        ie.set_input_budget_provider(previous)


def test_a_real_oversized_file_is_refused_before_pillow_decodes_it(app, tmp_path):
    """Not a stub: a genuine 4000x4000 PNG under a 4 Mi-pixel budget."""
    from app.services import image_bank_service as bank

    path = tmp_path / 'big.png'
    Image.new('RGB', (4000, 4000), (10, 20, 30)).save(path)
    with app.app_context():
        save_config({'image_input': {'max_side': 8192, 'max_pixels': 4 * 1024 * 1024}})
        with pytest.raises(ValueError, match='Image size budget'):
            with bank.safe_bank_source(str(path), label='bank image'):
                pass
        save_config({'image_input': {'max_side': 8192, 'max_pixels': 32 * 1024 * 1024}})
        with bank.safe_bank_source(str(path), label='bank image') as im:
            assert im.size == (4000, 4000)


def test_the_unlimited_warning_is_said_only_when_something_is_unlimited():
    from app.services import input_budget

    assert input_budget.unlimited_warning(16384, 64 * 1024 * 1024) == ''
    for pair in ((0, 64 * 1024 * 1024), (16384, 0), (0, 0)):
        assert 'fills memory' in input_budget.unlimited_warning(*pair)


@pytest.fixture(autouse=True)
def _restore_pillow_bomb_threshold():
    """`image_input.max_pixels = 0` clears Pillow's process-wide threshold on
    purpose. Put it back, or a later test in the same process inherits it."""
    saved = Image.MAX_IMAGE_PIXELS
    yield
    Image.MAX_IMAGE_PIXELS = saved


def test_the_capabilities_payload_publishes_the_request_limits_the_dropzone_batches_by(client):
    """A drop used to go up as ONE request and meet a bare 413 from five to
    eight high-resolution photos (_nofaceman, Discord). The dropzone now splits
    a drop by the server's two limits — and reads them from here, never from a
    copy kept in the front."""
    from app.services import face_dataset_service as fds
    capability = client.get('/api/capabilities').get_json()['dataset_import']
    assert capability['max_files_per_request'] == fds.IMPORT_MAX_FILES == 20
    assert capability['max_request_bytes'] == 512 * 1024 * 1024   # the IMPORT ceiling, not the generic web default


def test_the_import_route_and_the_capability_share_one_file_cap(app):
    """The route's refusal and the published limit cannot drift apart: both
    read IMPORT_MAX_FILES."""
    import inspect
    from app.routes import datasets as routes
    from app import capabilities
    assert 'svc.IMPORT_MAX_FILES' in inspect.getsource(routes.dataset_import)
    assert '_fds.IMPORT_MAX_FILES' in inspect.getsource(capabilities._dataset_import_policy)
