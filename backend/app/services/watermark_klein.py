"""Watermark removal by PREFILL + Flux.2 Klein full-edit refine, crop-and-stitch — the
V2 sister of watermark_lama.

Why a second method: LaMa (V1) is non-generative and perfect outside the mask, but on
complex texture (skin / fabric / busy background) it smears — the "weird mask sections"
grief — and it can't touch a mark that sits ON the subject (those stay 'review'). Klein
reconstructs texture far better AND makes the on-subject case actionable.

Architecture — PREFILL then REFINE (GPU-derived 2026-07-17, see below):

  1. crop a padded square around the mark, upscale it to ~1 MP (the "magnifying glass" so
     a few-pixel mark in a 4K photo is big enough for the model to see);
  2. PREFILL the masked region of that crop — repaint the watermark away with the LaMa
     worker (fallback cv2 TELEA). The result is deliberately soft/blurry; that is fine,
     its ONLY job is to hand Klein a reference with NO watermark left in it;
  3. Klein REFINES the pre-filled crop as a native full-edit (VAEEncode the whole crop →
     ReferenceLatent + KSampler on that latent, denoise 1.0, cfg 1 — NO SetLatentNoiseMask,
     the improve-skin edit pattern). Klein regenerates real texture over the soft prefill;
  4. composite the refined crop back onto the ORIGINAL in pixel space, pasting ONLY the
     masked region (+ a few-px feather). Every pixel outside that footprint keeps its
     ORIGINAL bytes — that is THE preservation guarantee, and it holds no matter how far
     the model drifts across the (fully re-rendered) crop.

WHY the prefill is mandatory (empirical, on a real photo):
  * The masked-inpaint graph (SetLatentNoiseMask + DifferentialDiffusion, denoise 1.0)
    feeds the ORIGINAL crop as the ReferenceLatent. At cfg 1 (guidance-distilled Klein)
    the prompt barely counts against that reference — so the watermark, still visible in
    the reference, is REPRODUCED as ghost glyphs.
  * Pre-filling the reference (watermark gone) kills the ghosts — but if Klein is asked to
    only paint inside the mask it just copies the prefill's blur back. Handing Klein the
    pre-filled crop as a FULL edit lets it regenerate genuine texture over the soft patch,
    which is exactly its improve-details core competency. A Klein pass WITHOUT a prefill is
    proven ineffective, so there is no skip-prefill path — prefill or fail.

The ComfyUI round-trip goes through the shared queue_manager (serialized against training
/ vision by the worker's own gating), then this module reads the finished crop back and
does the composite locally. Same `(ok, error)` tuple contract as watermark_lama."""
from __future__ import annotations
import io
import logging
import math
import os
import random
import tempfile
import time
import uuid

from PIL import Image, ImageDraw, ImageFilter

from .. import config as cfg
from . import klein_edit_helper as keh
from ..job_queue import queue_manager
from ..utils.comfyui import load_workflow_local, fetch_output_image_bytes

logger = logging.getLogger(__name__)

KLEIN_INPAINT_WORKFLOW_PATH = cfg.BACKEND_DIR / 'workflows' / 'klein_inpaint.json'

# The prefill already removed the watermark, so the refine prompt is about RECONSTRUCTION,
# not removal: push Klein to regenerate real texture over the soft prefill and keep the
# rest of the crop identical (drift outside the mask is discarded by the composite anyway).
# Kept in sync with node 6 of klein_inpaint.json (a test asserts the wiring).
KLEIN_INPAINT_PROMPT = ('Reconstruct this photo as a clean, natural image: replace any '
                        'blurred, smudged or patched areas with sharp, realistic surface '
                        'texture (skin, fabric, hair, background) consistent with the '
                        'surrounding pixels. Keep the subject, pose, colours and composition '
                        'identical. No text, no logos, no watermarks.')

# Nodes this module rewires — fail loudly if the shipped workflow changes shape. Node 53
# (VAEEncode of the pre-filled crop) is the latent for BOTH the ReferenceLatent and the
# KSampler now (full-edit), so it is checked too even though its wiring is fixed in the JSON.
_REQUIRED_NODES = ('114', '10', '90', '52', '53', '6', '77', '9')

# --- Tunables (calibrated at the GPU smoke; the study left these open) ---------
# Crop = a square this many times the mark's larger side, so the model gets real
# surrounding context to reconstruct from. Everything outside the mask is discarded
# by the composite, so generous context is cheap (only VRAM/time, bounded by the
# ~1 MP upscale target).
KLEIN_CONTEXT_FACTOR = 2.5
KLEIN_MIN_CROP = 384            # never crop below this (a tiny mark still gets context)
KLEIN_TARGET_MP = 1.0          # upscale the crop to ~this many megapixels for the model
KLEIN_LATENT_MULT = 16         # Flux.2 latent stride — crop dims sent to ComfyUI snap to it
KLEIN_MASK_EXPAND_PX = 8       # grow the mark rectangle before prefill (cover its AA edge)
KLEIN_COMPOSITE_FEATHER_PX = 6  # feather of the pixel-space paste seam (crop-native res)
KLEIN_DENOISE = 1.0            # full-edit refine: the crop's own latent is fully noised and
                               # the (pre-filled, watermark-free) crop re-enters as the
                               # ReferenceLatent — anything below 1.0 would leak the noised
                               # crop back in, so 1.0 is required, not a tunable
KLEIN_STEPS = 8               # Klein 9b is guidance-distilled (improve-skin edit uses 5); a
                              # couple more for cleaner reconstructed texture over the prefill
KLEIN_TIMEOUT = 300           # per-image ComfyUI round-trip budget (seconds)

_POLL_INTERVAL = 1.0


def is_available() -> bool:
    """Klein inpaint is usable = ComfyUI reachable AND the required Klein assets are on
    disk. The custom-node preflight (network) is deferred to clean-time (one actionable
    409), same split as the Klein generate path."""
    try:
        from ..capabilities import probe_comfyui
        if not probe_comfyui()['ok']:
            return False
        missing = keh.klein_missing_assets()
        return not any(a in missing for a in keh.KLEIN_REQUIRED)
    except Exception:
        return False


# --- Pure geometry (unit-tested, no I/O) --------------------------------------

def _normalize_boxes(boxes) -> list[list[float]]:
    """Clamp/order a list of normalized [x1,y1,x2,y2] into valid unit-range boxes.
    Drops anything non-finite or degenerate. Returns [] when nothing usable."""
    out = []
    for box in boxes or []:
        try:
            vals = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
            continue
        x1, x2 = sorted((vals[0], vals[2]))
        y1, y2 = sorted((vals[1], vals[3]))
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append([x1, y1, x2, y2])
    return out


def _union_px(W, H, boxes) -> tuple[float, float, float, float]:
    l = min(b[0] for b in boxes) * W
    t = min(b[1] for b in boxes) * H
    r = max(b[2] for b in boxes) * W
    b_ = max(b[3] for b in boxes) * H
    return l, t, r, b_


def _klein_crop_box(W, H, boxes, *, context=KLEIN_CONTEXT_FACTOR, min_side=KLEIN_MIN_CROP):
    """A square crop (in original px) centered on the mark, padded to `context`× its
    larger side, clamped inside the image. Slides in-bounds rather than shrinking so it
    stays as square as the image allows. Always CONTAINS the union of `boxes`."""
    l, t, r, b = _union_px(W, H, boxes)
    cx, cy = (l + r) / 2, (t + b) / 2
    side = max(r - l, b - t) * context
    side = max(side, min_side)
    side = min(side, W, H)               # never exceed the image
    half = side / 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    if x0 < 0: x1 -= x0; x0 = 0
    if x1 > W: x0 -= (x1 - W); x1 = W
    if y0 < 0: y1 -= y0; y0 = 0
    if y1 > H: y0 -= (y1 - H); y1 = H
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def _hard_mask(crop_box, W, H, boxes, *, expand_px=KLEIN_MASK_EXPAND_PX) -> Image.Image:
    """Binary 'L' mask (white = inpaint) at crop-native resolution: each mark rectangle,
    translated into crop coordinates and grown by `expand_px`."""
    l, t, r, b = crop_box
    cw, ch = r - l, b - t
    mask = Image.new('L', (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        bx1 = x1 * W - l - expand_px
        by1 = y1 * H - t - expand_px
        bx2 = x2 * W - l + expand_px
        by2 = y2 * H - t + expand_px
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(cw, bx2), min(ch, by2)
        if bx2 > bx1 and by2 > by1:
            draw.rectangle([bx1, by1, bx2 - 1, by2 - 1], fill=255)
    return mask


def _crop_boxes_norm(crop_box, W, H, boxes, *, expand_px=KLEIN_MASK_EXPAND_PX):
    """The mark rectangles as normalized [x1,y1,x2,y2] WITHIN the crop (0..1), each grown
    by `expand_px` and clamped — the geometry the PREFILL repaints. Mirror of `_hard_mask`
    but returns boxes, because both prefill engines take rectangles (the LaMa worker's
    `inpaint_watermarks` bboxes, and the cv2 TELEA mask). Normalized so it is valid at any
    resolution (crop-native or the scaled crop). Drops anything degenerate."""
    l, t, r, b = crop_box
    cw, ch = r - l, b - t
    out = []
    for x1, y1, x2, y2 in boxes:
        bx1 = max(0.0, (x1 * W - l - expand_px) / cw)
        by1 = max(0.0, (y1 * H - t - expand_px) / ch)
        bx2 = min(1.0, (x2 * W - l + expand_px) / cw)
        by2 = min(1.0, (y2 * H - t + expand_px) / ch)
        if bx2 > bx1 and by2 > by1:
            out.append([bx1, by1, bx2, by2])
    return out


def _upscale_size(cw, ch, *, target_mp=KLEIN_TARGET_MP, mult=KLEIN_LATENT_MULT):
    """Target (w,h) for the crop sent to Klein: scale toward ~target_mp, snap to `mult`.
    Small crops are magnified; oversized crops are scaled DOWN to bound VRAM."""
    scale = math.sqrt((target_mp * 1_000_000) / max(1, cw * ch))
    w = max(mult, int(round(cw * scale / mult)) * mult)
    h = max(mult, int(round(ch * scale / mult)) * mult)
    return w, h


def composite_inpaint(original, filled_crop, crop_box, composite_mask) -> Image.Image:
    """Paste `filled_crop` back onto `original` ONLY where `composite_mask` (crop-native
    'L', feathered) is non-zero. Where the mask is 0 the destination pixel is preserved
    BYTE-FOR-BYTE (PIL paste short-circuits a 0 alpha) — this is the preservation
    guarantee. `original`/`filled_crop` are RGB; returns a new RGB image."""
    result = original.convert('RGB').copy()
    if filled_crop.size != composite_mask.size:
        filled_crop = filled_crop.resize(composite_mask.size, Image.LANCZOS)
    result.paste(filled_crop.convert('RGB'), (crop_box[0], crop_box[1]), composite_mask)
    return result


# --- Prefill (repaint the watermark away before Klein sees it) ----------------

def _prefill_telea(scaled_crop, crop_boxes):
    """cv2 TELEA fallback prefill: repaint the `crop_boxes` rectangles of `scaled_crop`.
    Fast but blurry — Klein regenerates the texture over it. cv2 lives in the SAME ML
    extras as the LaMa worker, so a missing cv2 means the extras aren't installed at all:
    return 'unavailable' (→ the clean skips the row, actionable "install ML extras"),
    not 'failed'. A genuine cv2 runtime error is 'failed'."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, {'kind': 'unavailable',
                      'detail': 'watermark prefill unavailable: install the ML extras '
                                '(LaMa / OpenCV) to use the Klein clean method'}
    try:
        rgb = np.array(scaled_crop.convert('RGB'))
        w, h = scaled_crop.size
        mask = np.zeros((h, w), dtype='uint8')
        for x1, y1, x2, y2 in crop_boxes:
            left = max(0, min(w - 1, int(x1 * w)))
            top = max(0, min(h - 1, int(y1 * h)))
            right = max(left + 1, min(w, int(math.ceil(x2 * w))))
            bottom = max(top + 1, min(h, int(math.ceil(y2 * h))))
            mask[top:bottom, left:right] = 255
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)), None
    except Exception as e:  # noqa: BLE001 — any cv2 failure is a clean, surfaced 'failed'
        return None, {'kind': 'failed', 'detail': f'cv2 TELEA prefill failed: {e}'}


def _prefill_region(scaled_crop, crop_boxes, *, device='cpu'):
    """Repaint the masked rectangles of `scaled_crop` so the crop handed to Klein shows NO
    watermark (Klein's ReferenceLatent is this crop — a reference that still contains the
    mark makes cfg=1 Klein reproduce it as ghost glyphs). Prefer the LaMa worker (plausible
    texture Klein then sharpens); fall back to cv2 TELEA; if neither engine is installed,
    return 'unavailable'. There is deliberately NO skip-prefill path.

    `scaled_crop` RGB, `crop_boxes` normalized [x1,y1,x2,y2] within the crop.
    Returns (prefilled_RGB_image, None) or (None, {'kind', 'detail'})."""
    if not crop_boxes:
        return None, {'kind': 'failed', 'detail': 'no prefill region inside the crop'}
    from . import watermark_lama
    if watermark_lama.is_available():
        fd, tmp = tempfile.mkstemp(suffix='.png', prefix='wmklein_prefill_')
        os.close(fd)
        try:
            scaled_crop.convert('RGB').save(tmp, 'PNG')
            ok, err = watermark_lama.inpaint_watermarks(tmp, crop_boxes, device=device)
            if ok:
                with Image.open(tmp) as im:
                    return im.convert('RGB').copy(), None
            # LaMa is installed but errored — degrade to TELEA (cv2 ships with it) rather
            # than fail the whole clean; log so a systematic LaMa problem is visible.
            logger.warning('watermark_klein: LaMa prefill failed (%s) — using cv2 TELEA', err)
        finally:
            _cleanup(tmp)
    return _prefill_telea(scaled_crop, crop_boxes)


# --- ComfyUI round-trip -------------------------------------------------------

def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


def _read_comfy_output(filename) -> bytes | None:
    """The finished crop, from the ComfyUI output dir if present, else the /view API
    (path-independent, like link_completed_dataset_image's fallback)."""
    out_dir = _comfy_output_dir()
    if out_dir:
        path = os.path.join(out_dir, filename)
        if os.path.isfile(path):
            try:
                with open(path, 'rb') as fh:
                    return fh.read()
            except OSError:
                pass
    return fetch_output_image_bytes(filename)


def _cleanup(*paths):
    for p in paths:
        if not p:
            continue
        try:
            os.remove(p)
        except OSError:
            pass


def _wait_for_job(job_id, timeout):
    """Block until the queue worker finishes `job_id`, returning
    (status, result_filename, error_message). `db.session.rollback()` each poll drops
    the request thread's stale snapshot so the worker thread's commits become visible
    (same cross-thread read pattern as link_completed_dataset_image)."""
    from ..models import ImageGenerationQueue
    from ..extensions import db
    deadline = time.monotonic() + timeout
    while True:
        db.session.rollback()
        row = (ImageGenerationQueue.query
               .filter_by(job_id=job_id)
               .first())
        if row is not None and row.status in ('completed', 'failed', 'cancelled'):
            return row.status, row.result_filename, row.error_message
        if time.monotonic() >= deadline:
            return 'timeout', None, None
        time.sleep(_POLL_INTERVAL)


def _run_klein_job(user_id, crop_img, *, seed, steps=KLEIN_STEPS,
                   denoise=KLEIN_DENOISE, timeout=KLEIN_TIMEOUT):
    """Enqueue one full-edit refine job on the PRE-FILLED `crop_img` and return
    (filled_crop_image, None) or (None, error). The crop must already be watermark-free —
    it becomes the KSampler latent AND the ReferenceLatent (no SetLatentNoiseMask). Isolated
    seam so tests can mock the GPU round-trip. Raises KleinModelsMissing if a required asset
    vanished between preflight and here (so the route can 409 + auto-download)."""
    workflow = load_workflow_local(str(KLEIN_INPAINT_WORKFLOW_PATH))
    if not workflow:
        return None, {'kind': 'failed', 'detail': 'failed to load klein_inpaint workflow'}
    for node in _REQUIRED_NODES:
        if node not in workflow:
            return None, {'kind': 'failed',
                          'detail': f'workflow node {node} missing — klein_inpaint.json changed'}
    unet = keh.resolve_klein_unet()
    vae = keh.resolve_klein_vae()
    te = keh.resolve_klein_text_encoder()
    missing = keh.klein_missing_assets()
    if any(a in missing for a in keh.KLEIN_REQUIRED):
        raise keh.KleinModelsMissing(missing)

    comfy_input = _comfy_input_dir()
    uid = uuid.uuid4().hex[:8]
    crop_name = f'wmklein_crop_{uid}.png'
    crop_path = os.path.join(comfy_input, crop_name)
    crop_img.convert('RGB').save(crop_path)

    workflow['114']['inputs']['unet_name'] = unet
    workflow['10']['inputs']['vae_name'] = vae
    workflow['90']['inputs']['clip_name'] = te
    workflow['52']['inputs']['image'] = crop_name
    workflow['6']['inputs']['text'] = KLEIN_INPAINT_PROMPT
    workflow['77']['inputs']['seed'] = int(seed)
    workflow['77']['inputs']['steps'] = max(1, int(steps))
    workflow['77']['inputs']['denoise'] = float(denoise)
    workflow['9']['inputs']['filename_prefix'] = f'wmklein_{uid}'

    job_id = str(uuid.uuid4())
    try:
        queue_manager.add_job(job_type='image', user_id=str(user_id), workflow_data=workflow,
                              prompt=KLEIN_INPAINT_PROMPT, job_id=job_id,
                              metadata={'model_name': 'watermark_klein'})
        status, filename, err_msg = _wait_for_job(job_id, timeout)
    finally:
        _cleanup(crop_path)

    if status != 'completed' or not filename:
        return None, {'kind': 'failed',
                      'detail': err_msg or f'klein inpaint {status}'}
    data = _read_comfy_output(filename)
    out_dir = _comfy_output_dir()
    if out_dir:
        _cleanup(os.path.join(out_dir, filename))   # temporary render, never user data
    if not data:
        return None, {'kind': 'failed',
                      'detail': 'finished crop could not be retrieved from ComfyUI'}
    try:
        filled = Image.open(io.BytesIO(data)).convert('RGB')
    except (OSError, ValueError) as e:
        return None, {'kind': 'failed', 'detail': f'unreadable klein output: {e}'}
    return filled, None


def inpaint_watermark_klein(user_id, image_path, boxes, *, seed=None, device='cpu',
                            timeout=KLEIN_TIMEOUT) -> tuple[bool, dict | None]:
    """Remove the watermark(s) at normalized `boxes` from `image_path` via PREFILL + Klein
    full-edit refine + pixel-space composite, overwriting the file in place (WEBP q92, same
    as LaMa; the caller preserves the .orig). Returns the `(ok, error)` tuple contract:
    `error` is None on success, else {'kind', 'detail'} (kind 'unavailable' when Klein or the
    prefill engine isn't ready, 'failed' otherwise). Preserves every pixel outside the
    mask+feather. `device` selects the prefill LaMa device ('cpu' by default so the pending
    ComfyUI GPU job runs alone; Klein itself always owns the GPU via ComfyUI)."""
    if not is_available():
        return False, {'kind': 'unavailable',
                       'detail': 'Klein inpaint is not ready (ComfyUI unreachable or models missing)'}
    try:
        original = Image.open(image_path).convert('RGB')
    except (OSError, ValueError) as e:
        return False, {'kind': 'failed', 'detail': f'unreadable image: {e}'}
    W, H = original.size
    norm = _normalize_boxes(boxes)
    if not norm:
        return False, {'kind': 'failed', 'detail': 'no valid watermark box'}

    crop_box = _klein_crop_box(W, H, norm)
    cw, ch = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    crop_img = original.crop(crop_box)
    scaled_size = _upscale_size(cw, ch)
    scaled_crop = crop_img.resize(scaled_size, Image.LANCZOS)

    hard = _hard_mask(crop_box, W, H, norm)
    composite_mask = hard.filter(ImageFilter.GaussianBlur(KLEIN_COMPOSITE_FEATHER_PX))

    # Prefill the mark away BEFORE Klein — its ReferenceLatent is this crop, so a leftover
    # watermark would be reproduced as ghost glyphs. No prefill → no Klein (abort).
    crop_boxes = _crop_boxes_norm(crop_box, W, H, norm)
    prefilled, err = _prefill_region(scaled_crop, crop_boxes, device=device)
    if err:
        return False, err

    seed = random.randint(0, 2 ** 63 - 1) if seed is None else int(seed)
    filled_scaled, err = _run_klein_job(user_id, prefilled, seed=seed, timeout=timeout)
    if err:
        return False, err
    filled_crop = (filled_scaled if filled_scaled.size == (cw, ch)
                   else filled_scaled.resize((cw, ch), Image.LANCZOS))
    result = composite_inpaint(original, filled_crop, crop_box, composite_mask)
    try:
        result.save(image_path, 'WEBP', quality=92)
    except (OSError, ValueError) as e:
        return False, {'kind': 'failed', 'detail': f'could not save cleaned image: {e}'}
    return True, None
