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
    from app.services import lanpaint_helper
    monkeypatch.setattr(wk, 'is_available', lambda: True)
    monkeypatch.setattr(wk.keh, 'unet_for_job', lambda *a, **k: 'klein.safetensors')
    monkeypatch.setattr(wk.keh, '_unet_weight_dtype', lambda *a, **k: 'fp8_e4m3fn')
    monkeypatch.setattr(wk.keh, 'resolve_klein_vae', lambda *a, **k: 'vae.safetensors')
    monkeypatch.setattr(wk.keh, 'resolve_klein_text_encoder', lambda *a, **k: 'te.safetensors')
    monkeypatch.setattr(wk.keh, 'klein_missing_assets', lambda *a, **k: [])
    # The LanPaint preflight answers "present" — the tests below are about the
    # job plumbing, and the preflight has tests of its own further down.
    monkeypatch.setattr(lanpaint_helper, 'lanpaint_missing_nodes', lambda: [])


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


# --- the LanPaint move (GitHub #43) ---------------------------------------------
# InpaintModelConditioning is for inpaint-TRAINED checkpoints; Klein is an edit
# model, and conditioning it that way smeared the painted area. The lane now
# runs LanPaint (training-free inpainting sampler) with the edit lane's proven
# ReferenceLatent conditioning, dilates the paint so edges rebuild, and sends a
# LOCALIZED mask as a native-resolution context crop.

def test_the_masked_graph_runs_lanpaint_not_inpaint_conditioning():
    """The shipped graph IS the fix: LanPaint sampler over a latent noise mask,
    the edit lane's ReferenceLatent pair, and no Fill-model conditioning."""
    import json
    graph = json.loads(wk.KLEIN_MASK_INPAINT_WORKFLOW_PATH.read_text(encoding='utf-8'))
    classes = {n['class_type'] for n in graph.values()}
    assert 'LanPaint_KSampler' in classes
    assert 'SetLatentNoiseMask' in classes
    assert 'ReferenceLatent' in classes
    assert 'InpaintModelConditioning' not in classes
    # Every node the service rewires must exist, or the guard fires at runtime.
    for node in wk._REQUIRED_MASK_NODES:
        assert node in graph, f'_REQUIRED_MASK_NODES names {node}, graph lacks it'
    # The mask must actually reach the sampler: LoadImageMask -> noise mask ->
    # the LanPaint latent input. A graph that loads the mask and wires it
    # nowhere would repaint the whole frame.
    assert graph['105']['inputs']['mask'] == ['51', 0]
    assert graph['77']['inputs']['latent_image'] == ['105', 0]


def test_missing_lanpaint_blocks_before_the_queue_with_the_fix_named(app, monkeypatch):
    """No LanPaint in this ComfyUI -> the job is refused BEFORE anything is
    staged or queued, and the message names the Setup install."""
    from app.services import lanpaint_helper
    _klein_ready(monkeypatch)
    monkeypatch.setattr(lanpaint_helper, 'lanpaint_missing_nodes',
                        lambda: ['LanPaint_KSampler'])
    monkeypatch.setattr(lanpaint_helper, 'lanpaint_node_pack_installed', lambda: False)

    def _boom(**kw):
        raise AssertionError('nothing may be queued when the sampler is absent')
    monkeypatch.setattr(wk.queue_manager, 'add_job', _boom)

    with app.app_context():
        out, err = wk._run_klein_mask_job(
            'u', Image.new('RGB', (64, 64)), Image.new('L', (64, 64), 255), seed=1)
    assert out is None and err['kind'] == 'nodes_missing'
    assert 'Setup' in err['detail'] and 'LanPaint' in err['detail']


def test_installed_but_not_restarted_says_restart_not_install(app, monkeypatch):
    """The pack on disk but not loaded is a RESTART — telling someone to install
    what they just watched install is the failure the Krea pack already fixed."""
    from app.services import lanpaint_helper
    _klein_ready(monkeypatch)
    monkeypatch.setattr(lanpaint_helper, 'lanpaint_missing_nodes',
                        lambda: ['LanPaint_KSampler'])
    monkeypatch.setattr(lanpaint_helper, 'lanpaint_node_pack_installed', lambda: True)
    with app.app_context():
        out, err = wk._run_klein_mask_job(
            'u', Image.new('RGB', (64, 64)), Image.new('L', (64, 64), 255), seed=1)
    assert out is None and err['kind'] == 'nodes_missing'
    assert 'restart ComfyUI' in err['detail']
    assert 'install it from' not in err['detail']


def test_the_paint_is_dilated_so_edges_can_rebuild():
    """BFL's own Erase dilates ~10 px for the same reason: a mask that stops on
    the anti-aliased boundary leaves a halo of the removed thing."""
    mask = Image.new('L', (200, 200), 0)
    mask.paste(255, (90, 90, 110, 110))
    grown = wk._dilate_mask(mask, wk.KLEIN_MASK_DILATE_PX)
    px = wk.KLEIN_MASK_DILATE_PX
    assert grown.getpixel((90 - px, 100)) == 255      # grew by exactly the margin
    assert grown.getpixel((90 - px - 2, 100)) == 0    # and no further
    assert grown.getpixel((100, 100)) == 255          # the paint itself survives


def test_a_localized_mask_travels_as_a_context_crop():
    """The point of the crop: a fingertip of paint on a large photo keeps its
    native pixels instead of riding a frame scaled to KLEIN_MASK_MAX_MP."""
    mask = Image.new('L', (4000, 3000), 0)
    mask.paste(255, (1900, 1400, 2100, 1600))         # 200 px of paint, centered
    box = wk._mask_crop_box(mask)
    l, t, r, b = box
    assert (l, t, r, b) != (0, 0, 4000, 3000)
    # The crop contains the paint plus real context on every side…
    assert l < 1900 and t < 1400 and r > 2100 and b > 1600
    # …and is small enough that _mask_frame_size will not scale it down.
    assert ((r - l) * (b - t)) / 1_000_000 <= wk.KLEIN_MASK_MAX_MP


def test_a_frame_wide_mask_still_sends_the_whole_frame():
    mask = Image.new('L', (1000, 800), 0)
    mask.paste(255, (50, 50, 950, 750))
    assert wk._mask_crop_box(mask) == (0, 0, 1000, 800)


def test_a_tiny_mark_still_gets_the_minimum_context():
    mask = Image.new('L', (4000, 3000), 0)
    mask.paste(255, (2000, 1500, 2010, 1510))         # 10 px speck
    l, t, r, b = wk._mask_crop_box(mask)
    assert (r - l) >= wk.KLEIN_MIN_CROP and (b - t) >= wk.KLEIN_MIN_CROP


def test_crop_composite_keeps_bytes_outside_the_paint(tmp_path, monkeypatch):
    """The preservation guarantee, re-proven on the NEW crop geometry: with a
    localized mask the model only ever sees a crop, and every pixel outside the
    (dilated, feathered) paint keeps its original bytes."""
    src = tmp_path / 'shot.png'
    Image.new('RGB', (1600, 1200), (10, 200, 30)).save(src)
    mask = Image.new('L', (1600, 1200), 0)
    mask.paste(255, (700, 500, 900, 700))

    seen = {}
    monkeypatch.setattr(wk, 'is_available', lambda: True)

    def _capture(user_id, frame, m, **k):
        seen['frame'] = frame.size
        return Image.new('RGB', frame.size, (255, 0, 0)), None
    monkeypatch.setattr(wk, '_run_klein_mask_job', _capture)

    ok, err = wk.inpaint_mask_klein('u', str(src), mask=mask, seed=1)
    assert ok and err is None
    # The model saw a crop, not the frame.
    assert seen['frame'][0] < 1600 and seen['frame'][1] < 1200
    out = Image.open(src).convert('RGB')
    assert out.getpixel((800, 600)) == (255, 0, 0)
    for probe in [(10, 10), (1590, 1190), (200, 1000), (1400, 100)]:
        assert out.getpixel(probe) == (10, 200, 30), f'pixel {probe} was rewritten'


def test_helper_pack_constants_match_the_installer():
    """lanpaint_helper detects the folder setup_installer clones — the two are
    pinned together so a rename in one cannot silently strand the other."""
    from app import setup_installer
    from app.services import lanpaint_helper
    spec = setup_installer._NODE_PACKS['lanpaint_nodes']
    assert spec['folder'] == lanpaint_helper.LANPAINT_NODE_PACK['pack']
    assert spec['repo'] == lanpaint_helper.LANPAINT_NODE_PACK['url']
    assert 'lanpaint_nodes' in setup_installer.INSTALL_ACTIONS


def test_the_sent_frame_is_prefilled_where_the_paint_is():
    """The frame is both the noise latent and the ReferenceLatent, and a
    reference that still shows the object makes cfg-1 Klein reproduce it —
    proven live on this lane (an earring came back almost untouched without the
    prefill, and removed clean with it). Masked pixels must change; unpainted
    pixels must not."""
    import numpy as np
    frame = Image.new('RGB', (128, 128), (40, 120, 60))
    frame.paste((250, 20, 20), (50, 50, 80, 80))       # the "object"
    mask = Image.new('L', (128, 128), 0)
    mask.paste(255, (48, 48, 82, 82))
    out = wk._prefill_mask(frame, mask)
    a, b = np.asarray(frame, dtype=int), np.asarray(out, dtype=int)
    # Inside the mask the object is gone (no longer saturated red)…
    assert b[64, 64, 0] < 200, 'the object survived the prefill'
    # …and outside it the frame is untouched.
    assert (a[:40, :] == b[:40, :]).all() and (a[:, 100:] == b[:, 100:]).all()


def test_an_empty_prefill_mask_returns_the_frame_unchanged():
    frame = Image.new('RGB', (32, 32), (1, 2, 3))
    out = wk._prefill_mask(frame, Image.new('L', (32, 32), 0))
    assert list(out.getdata()) == list(frame.getdata())
