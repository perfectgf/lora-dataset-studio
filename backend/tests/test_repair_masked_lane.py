"""The MASKED (full-frame) repair lane — the geometry contributed in PR #37.

The crop lane cuts a square around the drawn box and magnifies it to ~1 MP, so
Klein reconstructs a necklace without ever seeing the face it sits on. This lane
sends the WHOLE frame plus a painted mask instead. What is pinned here is the
part that had to change on the way in — a full frame at any resolution is a
memory bill nobody agreed to — and the guarantee both lanes share: outside the
painted area, the file keeps its bytes.
"""
import base64
import io

import pytest
from PIL import Image

from app.services import watermark_klein as wk
from app.services import face_dataset_service as fds


# --- the cap ------------------------------------------------------------------

def test_a_huge_photo_is_scaled_down_before_it_reaches_the_model():
    """THE regression this lane arrived with. As contributed, the frame was only
    snapped to the latent stride, so a 24 MP photo went through Klein whole —
    about 12x the pixels the crop lane ever sends, on a machine that is also
    running ComfyUI and possibly a training run."""
    w, h = wk._mask_frame_size(6000, 4000)          # 24 MP
    assert (w * h) / 1_000_000 <= wk.KLEIN_MASK_MAX_MP + 0.25
    # …and the shape survives: a cap that squares the image would move the
    # painted area off its subject when the result is composited back.
    assert abs((w / h) - (6000 / 4000)) < 0.02


def test_a_small_photo_is_never_magnified():
    """The crop lane magnifies on purpose (a 200 px mark needs pixels to work
    with). A full frame does not: upsampling here would only invent detail the
    file never had, and then pay for it in VRAM."""
    w, h = wk._mask_frame_size(512, 512)
    assert w <= 512 and h <= 512


@pytest.mark.parametrize('size', [(6000, 4000), (1024, 768), (513, 977), (17, 4001)])
def test_every_size_snaps_to_the_latent_stride(size):
    w, h = wk._mask_frame_size(*size)
    assert w % wk.KLEIN_LATENT_MULT == 0 and h % wk.KLEIN_LATENT_MULT == 0
    assert w >= wk.KLEIN_LATENT_MULT and h >= wk.KLEIN_LATENT_MULT


# --- the guarantee ------------------------------------------------------------

def _png_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def test_pixels_outside_the_painted_area_keep_their_bytes(tmp_path, monkeypatch):
    """The whole reason this lane is allowed to touch the user's file in place."""
    src = tmp_path / 'shot.png'
    original = Image.new('RGB', (256, 256), (10, 200, 30))
    original.save(src)

    mask = Image.new('L', (256, 256), 0)
    for x in range(100, 160):
        for y in range(100, 160):
            mask.putpixel((x, y), 255)

    # Klein "returns" a frame that is red EVERYWHERE — so any pixel that ends up
    # red outside the painted square is a leak, not a repaint.
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    monkeypatch.setattr(wk, '_run_klein_mask_job',
                        lambda *a, **k: (Image.new('RGB', a[1].size, (255, 0, 0)), None))

    ok, err = wk.inpaint_mask_klein('u', str(src), mask=mask, seed=1)
    assert ok and err is None

    out = Image.open(src).convert('RGB')
    assert out.getpixel((128, 128)) == (255, 0, 0), 'the painted area was not repainted'
    # Well outside the feathered seam, the original colour must survive exactly.
    for probe in [(0, 0), (255, 255), (0, 255), (255, 0), (40, 40), (220, 220)]:
        assert out.getpixel(probe) == (10, 200, 30), f'pixel {probe} was rewritten'


def test_an_empty_mask_never_reaches_the_gpu(tmp_path, monkeypatch):
    src = tmp_path / 'shot.png'
    Image.new('RGB', (64, 64), (1, 2, 3)).save(src)
    monkeypatch.setattr(wk, 'is_available', lambda: True)

    def _boom(*a, **k):
        raise AssertionError('a blank mask must be refused before the round-trip')
    monkeypatch.setattr(wk, '_run_klein_mask_job', _boom)

    ok, err = wk.inpaint_mask_klein('u', str(src), mask=Image.new('L', (64, 64), 0))
    assert not ok and 'empty' in (err or {}).get('detail', '')


def test_boxes_still_work_on_this_lane(tmp_path, monkeypatch):
    """A drawn box is rasterized onto the full frame, so the two gestures meet
    here rather than growing two code paths that can drift apart."""
    src = tmp_path / 'shot.png'
    Image.new('RGB', (128, 128), (9, 9, 9)).save(src)
    seen = {}
    monkeypatch.setattr(wk, 'is_available', lambda: True)

    def _capture(user_id, frame, mask, **k):
        seen['extrema'] = mask.getextrema()
        return Image.new('RGB', frame.size, (7, 7, 7)), None
    monkeypatch.setattr(wk, '_run_klein_mask_job', _capture)

    ok, _ = wk.inpaint_mask_klein('u', str(src), [[0.25, 0.25, 0.5, 0.5]])
    assert ok and seen['extrema'][1] == 255


# --- the decoder --------------------------------------------------------------

def test_a_painted_mask_survives_the_round_trip_through_json():
    painted = Image.new('L', (40, 30), 0)
    painted.paste(255, (5, 5, 20, 20))
    out = fds.decode_repair_mask(_png_data_url(painted), (40, 30))
    assert out.size == (40, 30) and out.getextrema()[1] == 255


def test_a_mask_is_resized_to_the_frame_it_will_be_applied_to():
    """The browser paints at the size it displayed; the file may be larger."""
    painted = Image.new('L', (40, 30), 0)
    painted.paste(255, (5, 5, 20, 20))
    assert fds.decode_repair_mask(_png_data_url(painted), (400, 300)).size == (400, 300)


@pytest.mark.parametrize('bad, why', [
    ('', 'empty string'),
    (None, 'not a string'),
    ('data:image/png;base64', 'no comma'),
    ('data:image/png,notbase64', 'not declared base64'),
    ('!!!!not base64 at all!!!!', 'undecodable'),
])
def test_a_mask_that_is_not_one_is_refused(bad, why):
    with pytest.raises(ValueError):
        fds.decode_repair_mask(bad, (10, 10))


def test_an_unpainted_mask_says_so_instead_of_spending_a_gpu_minute():
    blank = _png_data_url(Image.new('L', (20, 20), 0))
    with pytest.raises(ValueError, match='paint the area'):
        fds.decode_repair_mask(blank, (20, 20))


def test_an_oversized_mask_is_refused_before_it_is_decoded():
    huge = 'data:image/png;base64,' + base64.b64encode(
        b'\x00' * (fds.REPAIR_MASK_MAX_BYTES + 1)).decode()
    with pytest.raises(ValueError, match='too large'):
        fds.decode_repair_mask(huge, (10, 10))


# --- the seam the other tests mock away -----------------------------------------
# Every test above stands in for `_run_klein_mask_job` wholesale, which is the
# right call for geometry — and is exactly why the lane shipped unable to enqueue
# anything. Inside that function `_await_klein_output` did not exist and
# `add_job` was called with the crop lane's arguments in the wrong positions:
# a NameError and a TypeError on the first real brush stroke, invisible to a
# suite that never entered the function. These tests enter it, and stand in one
# level lower — at ComfyUI's door.

def _klein_ready(monkeypatch):
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    monkeypatch.setattr(wk.keh, 'unet_for_job', lambda *a, **k: 'klein.safetensors')
    monkeypatch.setattr(wk.keh, '_unet_weight_dtype', lambda *a, **k: 'fp8_e4m3fn')
    monkeypatch.setattr(wk.keh, 'resolve_klein_vae', lambda *a, **k: 'vae.safetensors')
    monkeypatch.setattr(wk.keh, 'resolve_klein_text_encoder', lambda *a, **k: 'te.safetensors')
    monkeypatch.setattr(wk.keh, 'klein_missing_assets', lambda *a, **k: [])


def test_the_masked_lane_can_actually_enqueue_a_job(app, tmp_path, monkeypatch):
    """THE regression this file exists for. Calls the real function; only
    ComfyUI itself is stood in for."""
    _klein_ready(monkeypatch)
    monkeypatch.setattr(wk, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(wk, '_comfy_output_dir', lambda: str(tmp_path))

    seen = {}
    monkeypatch.setattr(wk.queue_manager, 'add_job',
                        lambda **kw: seen.update(kw) or 'queued')
    monkeypatch.setattr(wk, '_wait_for_job', lambda jid, t: ('completed', 'out.png', None))
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), (3, 4, 5)).save(buf, format='PNG')
    monkeypatch.setattr(wk, '_read_comfy_output', lambda name: buf.getvalue())

    with app.app_context():
        out, err = wk._run_klein_mask_job(
            'u', Image.new('RGB', (64, 64)), Image.new('L', (64, 64), 255), seed=7)

    assert err is None and out is not None, err
    assert out.size == (64, 64)
    # add_job is called BY NAME — the crop lane's positional order silently
    # became (job_type=user_id, user_id='...', workflow_data=...) once.
    assert seen['job_type'] == 'image'
    assert seen['user_id'] == 'u'
    assert seen['workflow_data']['51']['inputs']['image'].startswith('wmklein_mask_')
    assert seen['workflow_data']['52']['inputs']['image'].startswith('wmklein_frame_')


def test_a_failed_job_is_reported_not_swallowed(app, tmp_path, monkeypatch):
    _klein_ready(monkeypatch)
    monkeypatch.setattr(wk, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(wk.queue_manager, 'add_job', lambda **kw: 'queued')
    monkeypatch.setattr(wk, '_wait_for_job',
                        lambda jid, t: ('failed', None, 'the GPU said no'))

    with app.app_context():
        out, err = wk._run_klein_mask_job(
            'u', Image.new('RGB', (32, 32)), Image.new('L', (32, 32), 255), seed=1)
    assert out is None and 'the GPU said no' in err['detail']


def test_the_staged_copies_are_removed_even_when_the_job_fails(app, tmp_path, monkeypatch):
    """They are what the sweeper is a backstop FOR; the happy path must not be
    the only one that cleans up."""
    _klein_ready(monkeypatch)
    monkeypatch.setattr(wk, '_comfy_input_dir', lambda: str(tmp_path))
    monkeypatch.setattr(wk.queue_manager, 'add_job', lambda **kw: 'queued')
    monkeypatch.setattr(wk, '_wait_for_job', lambda jid, t: ('failed', None, 'nope'))

    with app.app_context():
        wk._run_klein_mask_job('u', Image.new('RGB', (32, 32)),
                               Image.new('L', (32, 32), 255), seed=1)
    leftovers = [f.name for f in tmp_path.iterdir() if f.name.startswith('wmklein_')]
    assert leftovers == [], f'staged copies left behind: {leftovers}'
