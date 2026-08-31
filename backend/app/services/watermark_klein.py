"""Watermark removal on Flux.2 Klein — the V2 sister of watermark_lama.

Why a second method: LaMa (V1) is non-generative and perfect outside the mask, but on
complex texture (skin / fabric / busy background) it smears — the "weird mask sections"
grief — and it can't touch a mark that sits ON the subject (those stay 'review'). Klein
reconstructs texture far better AND makes the on-subject case actionable.

TWO LANES BEHIND ONE ENTRY POINT. `inpaint_watermark_klein` is called both by the 🧽
watermark clean and by the ✦ Repair of a hand-drawn BOX, and since 2026-08-31 they no
longer do the same thing. The caller's free `prompt` is what separates them:

  * NO prompt → the 🧽 WATERMARK CLEAN. The WHOLE photo goes to Klein with the single
    instruction "remove watermark" and comes back entirely re-rendered (below).
  * a prompt → the ✦ BOX REPAIR ("remove the necklace", "fix this hand"). Unchanged:
    crop, prefill, refine, harmonize, paste back ONLY the box, so every pixel outside it
    keeps its ORIGINAL bytes. That byte-exactness is what that lane is sold on in the UI
    ("only the mark changes"), it is the one thing the ✦ Edit lane cannot do, so the
    clean's move to full-frame deliberately did not drag it along.

=== THE CLEAN: whole photo, 4 steps, "remove watermark" (maintainer, 2026-08-31) ======

The recipe: the ENTIRE photo (scaled down to KLEIN_MASK_MAX_MP if it is bigger,
dimensions snapped to KLEIN_LATENT_MULT, never magnified), its DETECTED ZONES ERASED
first (LaMa / cv2 TELEA, KLEIN_MASK_EXPAND_PX of margin, on the full frame) →
klein_inpaint.json, prompt `remove watermark`, steps 4, sampler euler, cfg 1, denoise
1.0, random seed → the render resampled back to the file's ORIGINAL dimensions and
written in place. No crop, no harmonization, no compositing. `boxes` no longer bounds
the repaint — it says "there is a mark HERE, do not hand it back", and everything the
detector missed is still cleaned by the pass itself.

WHAT THAT COSTS AND BUYS, measured on the maintainer's two test images the evening the
decision was taken (bench at 8 steps and the long reconstruction prompt, denoise 1.0):
  * the wall-to-wall TILED watermark — watermark score 0.69 → 0.15, 12 zones → 0,
    visually clean. The crop lane could never win this one: there was no "outside".
  * the 7-LOGO image — 7 zones → 3, one ghost glyph (a logo re-rendered as a moon in the
    sky), and the whole image re-rendered: mean deviation 37/255 on the pixels that
    carried no mark at all.
  * at denoise 0.4 and 0.7, on both images, NOTHING was removed. 1.0 is not a tunable
    here either — for a different reason than the box lane's.

Then re-measured on the SHIPPED recipe (4 steps, `remove watermark`, euler, cfg 1,
denoise 1.0), scored by the SigLIP2 / Grounding-DINO detector rather than by eye —
Klein 9B kv-fp8, same two images, before → after:
  * the wall-to-wall TILED one (1200×800) — watermark score 0.687 → 0.161, zones
    12 → 0, mean deviation 11/255 off the marks. Visually clean.
  * the 7-LOGO one (1488×992 after the cap) — score 0.875 → 0.490, zones 7 → 4. Four
    logos SURVIVE, partly as ghosts, and the whole photo is re-rendered (mean deviation
    29/255 off the marks, sky colours moved).
  * the denoise sweep, again: 0.4 and 0.7 remove nothing at all (7/7 and 9/12 zones
    still standing, score unmoved). 1.0 is required on this graph, not preferred.
THE GHOST-DISC FAILURE MODE, which is WHY the erase step exists. Run NAKED — the whole
photo, no zones erased — Klein does not delete that logo. It REINTERPRETS it: the
disc-with-a-mountain came back as a plausible MOON in the sky, twice in one pass (sky and
horizon), 0.878 → 0.492 with 17/255 of drift off the marks. Three independent runs, three
seeds, one of them straight through `_clean_full_frame` itself. A property of the recipe
on that kind of mark, not bad luck.

WORSE, AND THE REASON THIS IS IN CAPITALS: the detector scores that result at ZERO zones,
because a moon in a sky is not a watermark to a watermark detector. The pass would then
report 'cleaned' on a photo carrying two invented celestial bodies and nothing downstream
would disagree. Any future automatic "did the clean work?" gate must compare against the
SOURCE, never re-run the detector on the output.

WHAT FIXED IT, and what did not:
  * ERASE THE ZONES FIRST (TELEA on the full frame, margin, then the same pass) — no
    moon, no logo, scene faithful, score 0.44, 0 zones, drift 18/255. Shipped.
  * a naked pass with an explicit "do not add any moon…" prompt — no moons either, but
    a washed-out photo at 50/255 of drift. Rejected.
  * lowering the denoise — removes nothing at all (above). Not a lever.
The July doctrine held: a mark still present in the reference is reproduced, so take it
out of the reference. The full-frame pass keeps its whole point — it still clears what
the detector MISSED, which is what wins the tiled case (that one needs no erasing: its
mark has no "outside" to erase toward, and the naked pass already takes it to 0 zones).

An earlier note here claimed "all seven logos gone, no ghost" on a naked run. Two things
were wrong with it, and both are worth remembering: the FIRST measurement of that pair was
taken on a file the maintainer had already cleaned an hour before (same name, 2048×1368
instead of the original's 1365 — a bench input can change under you on a live instance),
and the second was a single seed on a different build read from a downscaled contact
sheet. One seed judged by eye does not establish the absence of anything.

THE SHAPE OF THE RESULT, then, and it is what the UI has to say: erase-then-re-render
clears the zones it was given AND what the detector missed, so it EXCELS on a repeated or
tiled mark — the case a box can never frame. What can still survive is a distinct mark
the detector never found, since nothing erased it from the reference. Look at the output;
↩ Restore original is one click away.

The trade is explicit and is the maintainer's call: the photo is REGENERATED, so the
"every byte outside the mark is original" guarantee does not hold on this lane any more,
and the removal is not guaranteed either — it is a generative pass, not a mask. What it
buys is reach: a mark the detector could not box is now removable at all. ↩ Restore
original (the preserved `.orig` sibling) is what a user who dislikes the re-render falls
back to, and it is untouched.

This REVERSES the July doctrine kept below. That doctrine was not wrong about what it
measured — a masked, box-scoped Klein pass really does reproduce the mark as ghost
glyphs without a prefill — it was answering a different question: how to repaint a BOX.
Asked to clean a photo instead, the same model with a short instruction and the whole
frame in view "works much better" (the maintainer's words, on their own images).

=== THE BOX REPAIR: PREFILL then REFINE (GPU-derived 2026-07-17, unchanged) ===========

This is the lane the crop / prefill / harmonize helpers below still serve:

  1. crop a padded square around the box, upscale it to ~1 MP (the "magnifying glass" so
     a few-pixel mark in a 4K photo is big enough for the model to see);
  2. PREFILL the masked region of that crop — repaint the watermark away with the LaMa
     worker (fallback cv2 TELEA). The result is deliberately soft/blurry; that is fine,
     its ONLY job is to hand Klein a reference with NO watermark left in it;
  3. Klein REFINES the pre-filled crop as a native full-edit (VAEEncode the whole crop →
     ReferenceLatent + KSampler on that latent, denoise 1.0, cfg 1 — NO SetLatentNoiseMask,
     the improve-skin edit pattern). Klein regenerates real texture over the soft prefill;
  4. HARMONIZE the refined crop's tone to the surrounding original at the seam (per zone),
     so the pasted patch has no visible tonal step — see "WHY harmonization" below;
  5. composite the harmonized crop back onto the ORIGINAL in pixel space, pasting ONLY the
     masked region (+ a few-px feather). Every pixel outside that footprint keeps its
     ORIGINAL bytes — that is THE preservation guarantee, and it holds no matter how far
     the model drifts across the (fully re-rendered) crop.

WHY the prefill is mandatory ON THAT LANE (empirical, on a real photo):
  * The masked-inpaint graph (SetLatentNoiseMask + DifferentialDiffusion, denoise 1.0)
    feeds the ORIGINAL crop as the ReferenceLatent. At cfg 1 (guidance-distilled Klein)
    the prompt barely counts against that reference — so the watermark, still visible in
    the reference, is REPRODUCED as ghost glyphs.
  * Pre-filling the reference (watermark gone) kills the ghosts — but if Klein is asked to
    only paint inside the mask it just copies the prefill's blur back. Handing Klein the
    pre-filled crop as a FULL edit lets it regenerate genuine texture over the soft patch,
    which is exactly its improve-details core competency. A Klein pass WITHOUT a prefill is
    proven ineffective, so there is no skip-prefill path — prefill or fail.
  * Read this as being about a BOX-SCOPED pass, which is all it was ever tested on. The
    clean above sends no box at all, has no "outside" to protect and no reference to
    poison, and is measured on its own terms.

WHY harmonization is mandatory (measured on a real photo, GPU 2026-07-17):
  * The full-edit re-renders the ENTIRE crop, so Klein's output drifts globally in tone,
    colour and contrast even OUTSIDE the mask (measured on the seam ring of a real 1024²
    crop: luma -8.6, per-channel means R -8.8 / G -7.7 / B -12.6, contrast std ratio ~1.3).
  * The composite pastes ONLY the masked rectangle, so that drift lands as a hard step at
    the paste boundary — the "visible square" grief, worst on soft flats (skin, walls,
    knit) where a constant offset reads as a clean rectangle. The feather smooths the alpha
    edge but CANNOT remove a tonal offset.
  * Fix: on the "seam ring" — the band of pixels just OUTSIDE the mask, where the original
    is the ground truth Klein was meant to reproduce — measure the per-channel mean/std of
    original vs fill; that difference IS the drift. Apply the inverse per-channel affine
    (gain = orig_std/fill_std, bias to match means) to the whole patch so it lines up with
    the neighbourhood. It is a linear transform, so it PRESERVES Klein's reconstructed
    texture; it only re-seats its tone. Done PER ZONE (each mark harmonized against its own
    local ring) so a multi-mark crop spanning different lighting corrects each seam locally.
    The correction touches the PATCH only — the byte-exact preservation guarantee is intact.

Every lane's ComfyUI round-trip goes through the shared queue_manager (serialized against
training / vision by the worker's own gating); this module then reads the finished render
back — and, on the two box lanes only, composites it locally. Same `(ok, error)` tuple
contract as watermark_lama, on all of them."""
from __future__ import annotations
import io
import logging
import math
import os
import random
import tempfile
import time
import uuid

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .. import config as cfg
from . import image_encoding
from . import klein_edit_helper as keh
from ..job_queue import queue_manager
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local, fetch_output_image_bytes

logger = logging.getLogger(__name__)

KLEIN_INPAINT_WORKFLOW_PATH = cfg.BACKEND_DIR / 'workflows' / 'klein_inpaint.json'
KLEIN_MASK_INPAINT_WORKFLOW_PATH = cfg.BACKEND_DIR / 'workflows' / 'klein_mask_inpaint.json'
# 2026-07-27: node 77 of that file sampled with `scheduler: "beta57"`, a value the
# third-party RES4LYF pack injects into ComfyUI's CORE scheduler list at import —
# so it worked on the machine the graph was captured on and refused to run
# ("Value not in list: scheduler") on every install without that pack, watermark
# cleaning included. Now `simple`, which exists everywhere. See
# backend/tests/test_workflow_portability.py, which fails offline if a value from
# somebody's custom nodes is ever pinned in a shipped graph again.

# The prefill already removed the watermark, so the refine prompt is about RECONSTRUCTION,
# not removal: push Klein to regenerate real texture over the soft prefill and keep the
# rest of the crop identical (drift outside the mask is discarded by the composite anyway).
# Kept in sync with node 6 of klein_inpaint.json (a test asserts the wiring).
KLEIN_INPAINT_PROMPT = ('Reconstruct this photo as a clean, natural image: replace any '
                        'blurred, smudged or patched areas with sharp, realistic surface '
                        'texture (skin, fabric, hair, background) consistent with the '
                        'surrounding pixels. Keep the subject, pose, colours and composition '
                        'identical. No text, no logos, no watermarks.')

# The CLEAN lane's prompt, and it is deliberately three words. The sentence above
# describes a repair job to a model that is being shown a pre-filled patch; asked to
# clean a whole photo, Klein answers the short instruction better than the long one
# (maintainer, 2026-08-31 — see the module docstring for the numbers). Short also means
# the queue row reads "remove watermark", which is what the user thinks they asked for.
KLEIN_CLEAN_PROMPT = 'remove watermark'

# Nodes this module rewires — fail loudly if the shipped workflow changes shape. Node 53
# (VAEEncode of the pre-filled crop) is the latent for BOTH the ReferenceLatent and the
# KSampler now (full-edit), so it is checked too even though its wiring is fixed in the JSON.
_REQUIRED_NODES = ('114', '10', '90', '52', '53', '6', '77', '9')
# Masked graph: LoadImageMask (51) feeding SetLatentNoiseMask (105) on the
# frame's own latent (53), sampled by LanPaint (77) with the proven edit-lane
# conditioning (prompt + empty-negative ReferenceLatents, 92/110). Same guard
# as above — a hand-edited workflow must fail loudly here rather than silently
# repaint the wrong thing.
_REQUIRED_MASK_NODES = ('114', '10', '90', '52', '51', '6', '100', '53', '92',
                        '110', '105', '175', '102', '77', '8', '9')

# --- Tunables (calibrated at the GPU smoke; the study left these open) ---------
# Everything down to KLEIN_MASK_MAX_MP below belongs to the CROP lanes (the ✦ box
# repair and the localized masked repair). The 🧽 clean uses none of it: it sends the
# whole frame, so it has no crop to size, no mask to grow and no seam to harmonize.
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

# --- Masked (full-frame) inpaint -----------------------------------------------
# The crop lane above hands Klein a square cut out around the box. That bounds
# VRAM, but the model then reconstructs a necklace or a pair of glasses without
# ever seeing the face they sit on, and the patch is resampled to ~1 MP on the
# way. Sending the WHOLE photo with a mask fixes both: full context, native
# framing, and `noise_mask` keeps the sampler off every unpainted pixel.
# (Lane contributed by OneCodingDude on GitHub, PR #37.)
KLEIN_MASK_STEPS = 4            # this graph is distilled harder than the crop one
# ...but a full frame is not free, and THIS is where the contributed lane needed
# changing: it snapped the image to the latent stride and sent it at whatever
# size it happened to be, so a 4K photo went through Klein whole — one 24 MP
# image is ~12x the pixels the crop lane ever sends, on a box that also has
# ComfyUI and possibly a training run on it. Above this bound the frame is
# scaled down for the model and the result composited back at full resolution.
# Unpainted pixels are copied from the ORIGINAL either way, so the cap costs
# detail only INSIDE the painted area, and only on images large enough to risk
# an out-of-memory failure halfway through an in-place edit.
# The 🧽 CLEAN sends whole frames too and shares this cap — with one difference
# that has to be said out loud: it composites nothing back, so on a photo above
# the cap the WHOLE image returns resampled (down to 2 MP, back up to its
# original dimensions) rather than only the painted area.
KLEIN_MASK_MAX_MP = 2.0
# Grow the painted mask before the model sees it — BFL's own Erase (the closest
# thing to an official reference for mask-driven Klein removal) recommends
# dilating ~10 px so the sampler has room to rebuild the object's edges; without
# it the reconstruction stops exactly on the anti-aliased boundary and leaves a
# halo of the removed thing. The composite pastes the dilated footprint, so the
# preservation guarantee moves out by the same 10 px — that is the point.
KLEIN_MASK_DILATE_PX = 10
# A LOCALIZED painted mask goes through crop-and-stitch: the crop (mask bbox ×
# KLEIN_CONTEXT_FACTOR) keeps native detail that a full-frame pass would scale
# away under KLEIN_MASK_MAX_MP. Above this fraction of the frame the crop stops
# paying for itself (it IS most of the frame) and the full frame is sent, as the
# contributed lane always did.
KLEIN_MASK_CROP_FRACTION = 0.6
KLEIN_HARMONIZE_RING_PX = 24    # width (crop-native) of the seam ring sampled to estimate
                                # Klein's tonal drift — local enough to track the neighbour
                                # tone, wide enough for a stable per-channel mean/std
KLEIN_HARMONIZE_MAX_GAIN = 1.5  # clamp the per-channel contrast correction so a high-variance
                                # ring (strong edges just outside the mask) can't over-punch
KLEIN_HARMONIZE_MIN_RING = 64   # too few ring pixels (mark glued to the crop edges) → skip
                                # the correction rather than trust a noisy estimate (identity)
KLEIN_DENOISE = 1.0            # full-edit refine: the crop's own latent is fully noised and
                               # the (pre-filled, watermark-free) crop re-enters as the
                               # ReferenceLatent — anything below 1.0 would leak the noised
                               # crop back in, so 1.0 is required, not a tunable
KLEIN_STEPS = 8               # Klein 9b is guidance-distilled (improve-skin edit uses 5); a
                              # couple more for cleaner reconstructed texture over the prefill
KLEIN_CLEAN_STEPS = 4         # the CLEAN lane's own budget: a whole photo, not a 1 MP crop,
                              # once per flagged image — and the maintainer's recipe says 4
KLEIN_SAMPLER = 'euler'       # written into node 77 rather than trusted to the shipped JSON,
KLEIN_CFG = 1.0               # so the recipe lives where the tests can read it. Both graphs
                              # already carried these values; pinning them changes no render.
KLEIN_TIMEOUT = 300           # per-image ComfyUI round-trip budget (seconds)

_POLL_INTERVAL = 1.0


def is_available() -> bool:
    """Klein inpaint is usable — the SAME verdict `caps.watermark_klein` publishes,
    read from the one place that computes it (klein_edit_helper.klein_engine_ready).

    It used to be a laxer copy: ComfyUI reachable + the three required files on
    disk, by name. That skipped the two gaps the engine badge already knew about —
    a pinned widget VALUE this ComfyUI does not accept, and a file that is present
    but is not loadable weights (the truncated 9.5 GB UNET / HTML licence page).
    Being laxer than the badge is not harmless: `is_available()` is what makes the
    cleaner fall back to LaMa, so it decided to hand ComfyUI a doomed job instead
    of degrading cleanly, and it words the "why not" the bank cleaner shows.
    The custom-node preflight (network) stays deferred to clean-time (one
    actionable 409), same split as the Klein generate path."""
    try:
        from ..capabilities import probe_comfyui
        return keh.klein_engine_ready(probe_comfyui()['ok'])
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


def _mask_frame_size(w, h, *, max_mp=KLEIN_MASK_MAX_MP, mult=KLEIN_LATENT_MULT):
    """Size to send a FULL frame at: snap to the latent stride, scaling down
    first if it is larger than `max_mp`. Never magnifies — a small photo goes as
    it is rather than being upsampled into detail the file does not have.
    Shared by the masked repair and the 🧽 clean: both send whole frames."""
    w, h = max(1, int(w)), max(1, int(h))
    mp = (w * h) / 1_000_000.0
    if max_mp and mp > max_mp:
        k = math.sqrt(max_mp / mp)
        w, h = max(1, int(round(w * k))), max(1, int(round(h * k)))
    sw = max(mult, int(round(w / mult)) * mult)
    sh = max(mult, int(round(h / mult)) * mult)
    return sw, sh


def _full_frame_mask(W, H, boxes) -> Image.Image:
    """Rasterize normalized boxes onto the FULL frame (white = repaint), so a
    drawn box and a painted mask meet here and share every step after it."""
    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        bx1, by1, bx2, by2 = x1 * W, y1 * H, x2 * W, y2 * H
        if bx2 > bx1 and by2 > by1:
            draw.rectangle([bx1, by1, bx2 - 1, by2 - 1], fill=255)
    return mask


def _mask_rgb(mask_img) -> Image.Image:
    """White = repaint, as an RGB PNG `LoadImageMask` reads on its red channel."""
    luma = mask_img.convert('L')
    return Image.merge('RGB', (luma, luma, luma))


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


# --- Seam harmonization (kill Klein's global tonal drift at the paste boundary) ---
# The full-edit re-renders the whole crop, so the refined patch drifts in tone/colour/
# contrast even outside the mask; pasting only the rectangle turns that drift into a
# visible square. These pure helpers re-seat the patch's tone to the surrounding original,
# measured on the ring of ground-truth pixels just outside the mask. See the module
# docstring ("WHY harmonization is mandatory") for the measured drift.

_RING_THRESHOLD = 8            # GaussianBlur band cutoff (0..255) that defines the ring width
_STD_EPS = 1e-3               # a flat ring (std≈0) → gain 1.0 (offset-only correction)


def _seam_ring(zone_mask, exclude_mask, *, ring_px=KLEIN_HARMONIZE_RING_PX):
    """Boolean array (crop-native, shape (H, W)) of the pixels to sample for one zone's
    seam correction: a band just OUTSIDE `zone_mask` (a single mark rectangle), grown by
    `ring_px`, MINUS every masked/watermark pixel (`exclude_mask` = the union of ALL zones
    in the crop). Subtracting the union keeps the sample pure ORIGINAL ground truth — a
    neighbouring mark's still-watermarked pixels never contaminate this zone's estimate.
    `zone_mask`/`exclude_mask` are 'L' masks (white = inpaint)."""
    import numpy as np
    grown = zone_mask.filter(ImageFilter.GaussianBlur(ring_px))
    band = np.asarray(grown) > _RING_THRESHOLD                 # zone interior + a band around it
    inpaint = np.asarray(exclude_mask.convert('L')) > 127
    return band & ~inpaint                                     # ...minus every mask → outside band


def _harmonize_seam(filled_crop, original_crop, ring, *,
                    max_gain=KLEIN_HARMONIZE_MAX_GAIN, min_ring=KLEIN_HARMONIZE_MIN_RING):
    """Re-seat `filled_crop`'s tone onto `original_crop` using the `ring` sample, returning a
    NEW RGB image the size of `filled_crop`. Pure numpy, deterministic.

    On the ring (ground-truth pixels Klein was meant to reproduce) measure per-channel
    mean/std of both images; the difference is Klein's drift. Apply the inverse affine
    `(fill - fill_mean) * gain + orig_mean`, gain = orig_std/fill_std (clamped, offset-only
    when the ring is flat), to the WHOLE patch — a linear map, so Klein's texture survives,
    only its tone moves. When the ring is too small (mark glued to the crop edges) there is
    no trustworthy estimate → return `filled_crop` UNCHANGED (identity)."""
    import numpy as np
    if int(ring.sum()) < min_ring:
        return filled_crop
    fill = np.asarray(filled_crop.convert('RGB'), dtype=np.float64)
    orig = np.asarray(original_crop.convert('RGB'), dtype=np.float64)
    out = np.empty_like(fill)
    for c in range(3):
        fo, oo = fill[..., c][ring], orig[..., c][ring]
        fmean, fstd = fo.mean(), fo.std()
        gain = oo.std() / fstd if fstd > _STD_EPS else 1.0
        gain = min(max(gain, 1.0 / max_gain), max_gain)
        out[..., c] = (fill[..., c] - fmean) * gain + oo.mean()
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), 'RGB')


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
                   denoise=KLEIN_DENOISE, timeout=KLEIN_TIMEOUT, klein_model=None,
                   prompt=None):
    """Enqueue one native full-edit job on `crop_img` and return (rendered_image, None) or
    (None, error). Whatever is passed becomes the KSampler latent AND the ReferenceLatent
    (no SetLatentNoiseMask) — a PRE-FILLED crop on the box-repair lane, the WHOLE photo on
    the clean lane. Isolated seam so tests can mock the GPU round-trip. Raises
    KleinModelsMissing if a required asset vanished between preflight and here (so the
    route can 409 + auto-download).

    `klein_model`: the bare file name this pass must run on — the DATASET's stored
    pick when the images belong to one. None keeps the historical auto-resolution,
    which is what a dataset that never chose and every bank (no dataset, nothing to
    inherit) get. A named model that has left the disk raises KleinModelGone rather
    than repainting on a neighbour: this lane overwrites the user's file in place,
    so a silent swap is not even reversible by regenerating."""
    workflow = load_workflow_local(str(KLEIN_INPAINT_WORKFLOW_PATH))
    if not workflow:
        return None, {'kind': 'failed', 'detail': 'failed to load klein_inpaint workflow'}
    for node in _REQUIRED_NODES:
        if node not in workflow:
            return None, {'kind': 'failed',
                          'detail': f'workflow node {node} missing — klein_inpaint.json changed'}
    unet = keh.unet_for_job(klein_model)
    vae = keh.resolve_klein_vae()
    te = keh.resolve_klein_text_encoder()
    missing = keh.klein_missing_assets()
    if any(a in missing for a in keh.KLEIN_REQUIRED):
        raise keh.KleinModelsMissing(missing)

    # Same filesystem hand-off as the generation lanes (utils/comfy_fs): the crop
    # is WRITTEN into ComfyUI's input folder, which a container/remote install may
    # not share. Guarded so the failure names the folder instead of dying on a raw
    # OSError halfway through a clean.
    uid = uuid.uuid4().hex[:8]
    crop_name = f'wmklein_crop_{uid}.png'
    try:
        comfy_input = comfy_fs.ensure_input_usable(_comfy_input_dir())
        crop_path = comfy_fs.stage_input_write(
            crop_name, lambda p: crop_img.convert('RGB').save(p), comfy_input)
    except comfy_fs.ComfyFolderUnavailable as exc:
        # 'unavailable', not 'failed': nothing was attempted on the GPU — Klein
        # simply cannot be reached over the filesystem from this process. The
        # message already names the folder and what to do about it.
        return None, {'kind': 'unavailable', 'detail': str(exc)}

    workflow['114']['inputs']['unet_name'] = unet
    # Same rule as the edit lane: the graph hardcodes fp8_e4m3fn, which is only
    # right for an fp8 build (keh._unet_weight_dtype).
    workflow['114']['inputs']['weight_dtype'] = keh._unet_weight_dtype(unet)
    workflow['10']['inputs']['vae_name'] = vae
    workflow['90']['inputs']['clip_name'] = te
    workflow['52']['inputs']['image'] = crop_name
    # A CALLER may steer what is painted back in. None keeps the watermark
    # reconstruction prompt, so the cleaning lane sends the exact same graph it
    # always did — a test pins that value. What the free prompt unlocks is the
    # ONE thing this machinery could not do: a masked, prompted repair that
    # leaves every pixel outside the box byte-identical, instead of the ✦ Edit
    # lane's full re-render. (Asked for by mr.arrow and .samexit on Discord.)
    text = (prompt or '').strip() or KLEIN_INPAINT_PROMPT
    workflow['6']['inputs']['text'] = text
    workflow['77']['inputs']['seed'] = int(seed)
    workflow['77']['inputs']['steps'] = max(1, int(steps))
    workflow['77']['inputs']['denoise'] = float(denoise)
    # Pinned from here, not read from the JSON: both lanes' recipes name a sampler and a
    # cfg, and a hand-edit of the shipped graph would otherwise move them silently.
    workflow['77']['inputs']['sampler_name'] = KLEIN_SAMPLER
    workflow['77']['inputs']['cfg'] = KLEIN_CFG
    workflow['9']['inputs']['filename_prefix'] = f'wmklein_{uid}'

    job_id = str(uuid.uuid4())
    try:
        queue_manager.add_job(job_type='image', user_id=str(user_id), workflow_data=workflow,
                              prompt=text, job_id=job_id,
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


def _run_klein_mask_job(user_id, frame_img, mask_img, *, seed, steps=KLEIN_MASK_STEPS,
                        denoise=KLEIN_DENOISE, timeout=KLEIN_TIMEOUT, klein_model=None,
                        prompt=None):
    """Enqueue ONE LanPaint masked-inpaint job: Klein sees `frame_img` whole and
    the LanPaint sampler regenerates only where `mask_img` is white. The caller
    neutralizes the masked region first (_prefill_mask) so the ReferenceLatent
    does not hand the object back. LanPaint replaced InpaintModelConditioning here
    because Klein is an edit model, not an inpaint-trained checkpoint, and the
    Fill-model conditioning smeared the masked region (GitHub #43); the sampler
    enforces the mask itself, so no inpaint training is needed. Kept as its own
    seam so tests can stand in for the GPU round-trip."""
    from . import lanpaint_helper
    missing = lanpaint_helper.lanpaint_missing_nodes()
    if missing:
        if lanpaint_helper.lanpaint_node_pack_installed():
            detail = ('the LanPaint sampler is installed but this ComfyUI has '
                      'not been restarted since — restart ComfyUI, then run the '
                      'repair again')
        else:
            detail = ('masked repair runs on the LanPaint sampler, which this '
                      'ComfyUI does not have yet — install it from Setup ▸ '
                      'Image generation ▸ LanPaint sampler nodes, restart '
                      'ComfyUI, then run the repair again')
        return None, {'kind': 'nodes_missing', 'detail': detail}
    workflow = load_workflow_local(str(KLEIN_MASK_INPAINT_WORKFLOW_PATH))
    if not workflow:
        return None, {'kind': 'failed',
                      'detail': 'failed to load klein_mask_inpaint workflow'}
    for node in _REQUIRED_MASK_NODES:
        if node not in workflow:
            return None, {'kind': 'failed',
                          'detail': f'workflow node {node} missing — klein_mask_inpaint.json changed'}
    unet = keh.unet_for_job(klein_model)
    vae = keh.resolve_klein_vae()
    te = keh.resolve_klein_text_encoder()
    missing = keh.klein_missing_assets()
    if any(a in missing for a in keh.KLEIN_REQUIRED):
        raise keh.KleinModelsMissing(missing)

    uid = uuid.uuid4().hex[:8]
    # SAME `wmklein_` family as the crop lane, because the orphan sweeper in
    # comfy_fs matches on the exact shape these names take. A staged file the
    # sweeper cannot recognise never gets collected when a job dies between
    # staging and the `finally` below — it just sits in ComfyUI's input folder.
    frame_name = f'wmklein_frame_{uid}.png'
    mask_name = f'wmklein_mask_{uid}.png'
    frame_path = mask_path = None
    try:
        comfy_input = comfy_fs.ensure_input_usable(_comfy_input_dir())
        frame_path = comfy_fs.stage_input_write(
            frame_name, lambda p: frame_img.convert('RGB').save(p), comfy_input)
        mask_path = comfy_fs.stage_input_write(
            mask_name, lambda p: _mask_rgb(mask_img).save(p), comfy_input)
    except comfy_fs.ComfyFolderUnavailable as exc:
        return None, {'kind': 'failed', 'detail': str(exc)}

    workflow['114']['inputs']['unet_name'] = unet
    workflow['114']['inputs']['weight_dtype'] = keh._unet_weight_dtype(unet)
    workflow['10']['inputs']['vae_name'] = vae
    workflow['90']['inputs']['clip_name'] = te
    workflow['52']['inputs']['image'] = frame_name
    workflow['51']['inputs']['image'] = mask_name
    text = (prompt or '').strip() or KLEIN_INPAINT_PROMPT
    workflow['6']['inputs']['text'] = text
    workflow['77']['inputs']['seed'] = int(seed)
    workflow['77']['inputs']['steps'] = max(1, int(steps))
    workflow['77']['inputs']['denoise'] = float(denoise)
    workflow['9']['inputs']['filename_prefix'] = f'wmkleinmask_{uid}'

    job_id = str(uuid.uuid4())
    status, filename, err_msg = None, None, None
    try:
        queue_manager.add_job(job_type='image', user_id=str(user_id),
                              workflow_data=workflow, prompt=text, job_id=job_id,
                              metadata={'model_name': 'watermark_klein_mask'})
        status, filename, err_msg = _wait_for_job(job_id, timeout)
    finally:
        for stale in (frame_path, mask_path):
            _cleanup(stale)

    if status != 'completed' or not filename:
        return None, {'kind': 'failed',
                      'detail': err_msg or f'klein masked inpaint {status}'}
    data = _read_comfy_output(filename)
    out_dir = _comfy_output_dir()
    if out_dir:
        _cleanup(os.path.join(out_dir, filename))  # temporary render, never user data
    if not data:
        return None, {'kind': 'failed',
                      'detail': 'finished frame could not be retrieved from ComfyUI'}
    try:
        return Image.open(io.BytesIO(data)).convert('RGB'), None
    except (OSError, ValueError) as e:
        return None, {'kind': 'failed', 'detail': f'unreadable klein output: {e}'}


def _dilate_mask(mask, px):
    """Grow the white region by `px` (see KLEIN_MASK_DILATE_PX). MaxFilter runs
    on a bbox-sized window only: on a large frame with a fingertip of paint the
    full-frame filter would spend seconds on pixels that are all zero anyway."""
    if px <= 0:
        return mask
    bbox = mask.getbbox()
    if not bbox:
        return mask
    l, t, r, b = bbox
    l, t = max(0, l - px), max(0, t - px)
    r, b = min(mask.width, r + px), min(mask.height, b + px)
    region = mask.crop((l, t, r, b)).filter(ImageFilter.MaxFilter(px * 2 + 1))
    out = mask.copy()
    out.paste(region, (l, t))
    return out


def _mask_crop_box(mask, *, factor=KLEIN_CONTEXT_FACTOR, min_side=KLEIN_MIN_CROP,
                   fraction=KLEIN_MASK_CROP_FRACTION):
    """The region worth SENDING for this mask: its bbox grown by the context
    factor, clamped to the frame — or the whole frame when that region would
    cover more than `fraction` of it (the crop then buys no detail back).
    Native detail is the point: a localized repair on a large photo used to
    travel at KLEIN_MASK_MAX_MP and come back soft inside the paint; cropped,
    the same pixels fit under the cap at (or near) native resolution."""
    W, H = mask.size
    bbox = mask.getbbox()
    if not bbox:
        return (0, 0, W, H)
    l, t, r, b = bbox
    cx, cy = (l + r) / 2.0, (t + b) / 2.0
    side = max((r - l) * factor, (b - t) * factor, float(min_side))
    half = side / 2.0
    cl, ct = int(max(0.0, cx - half)), int(max(0.0, cy - half))
    cr, cb = int(min(float(W), cx + half)), int(min(float(H), cy + half))
    if (cr - cl) * (cb - ct) > fraction * W * H:
        return (0, 0, W, H)
    return (cl, ct, cr, cb)


def _prefill_mask(frame, mask):
    """Neutralize the masked region of the SENT frame before it is encoded.

    The frame is both the noise latent AND the ReferenceLatent, and a reference
    that still shows the object makes cfg-1 Klein reproduce it — the exact
    ghost the crop lane's prefill kills (see "WHY the prefill is mandatory"
    above). Proven live on this lane too: asked to remove a small earring,
    LanPaint without a prefill handed the earring back almost untouched, and
    with one the same job removed it clean. cv2 TELEA on the mask's own shape
    (the rectangle prefill above cannot follow a painted stroke). When cv2 is
    not installed the frame goes as-is: a large mask still repairs well, only
    small-object removal degrades, and blocking the whole lane on the ML
    extras would cost more than it protects."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return frame
    try:
        rgb = np.array(frame.convert('RGB'))
        m = (np.array(mask.convert('L')) > 127).astype('uint8') * 255
        if not m.any():
            return frame
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, m, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
    except Exception:   # noqa: BLE001 — a prefill failure degrades, never blocks
        logger.debug('mask prefill failed; sending the frame unprefilled', exc_info=True)
        return frame


def inpaint_mask_klein(user_id, image_path, boxes=None, *, mask=None, seed=None,
                       device='cpu', timeout=KLEIN_TIMEOUT,
                       klein_model=None, prompt=None) -> tuple[bool, dict | None]:
    """Masked Klein inpaint, in place, on the smallest region worth sending.

    `mask` is an 'L' image (white = repaint); `boxes` (normalized) are
    rasterized onto the full frame when no mask is given, so the drawn-box
    gesture can use this lane too. The mask is dilated (KLEIN_MASK_DILATE_PX)
    so the sampler can rebuild the removed thing's edges, then a LOCALIZED
    mask travels as a context crop (_mask_crop_box) and a frame-wide one as
    the whole frame — either way capped at KLEIN_MASK_MAX_MP, never
    magnified — and the painted region is neutralized before encoding
    (_prefill_mask) so the reference no longer shows the object. Every
    unpainted pixel is composited back from the original, so the file keeps
    its original bytes outside the (dilated) painted area — the same
    guarantee the watermark crop lane makes.
    `device` is ignored; it exists so the two lanes share one call shape.
    """
    if not is_available():
        return False, {'kind': 'unavailable',
                       'detail': 'Klein inpaint is not ready (ComfyUI unreachable or models missing)'}
    try:
        original = Image.open(image_path).convert('RGB')
    except (OSError, ValueError) as e:
        return False, {'kind': 'failed', 'detail': f'unreadable image: {e}'}
    W, H = original.size
    if mask is not None:
        hard = mask.convert('L')
        if hard.size != (W, H):
            hard = hard.resize((W, H), Image.BILINEAR)
    else:
        norm = _normalize_boxes(boxes)
        if not norm:
            return False, {'kind': 'failed', 'detail': 'no valid inpaint mask'}
        hard = _full_frame_mask(W, H, norm)
    if hard.getextrema()[1] == 0:
        return False, {'kind': 'failed', 'detail': 'inpaint mask is empty'}
    hard = _dilate_mask(hard, KLEIN_MASK_DILATE_PX)

    box = _mask_crop_box(hard)
    full_frame = box == (0, 0, W, H)
    region = original if full_frame else original.crop(box)
    region_mask = hard if full_frame else hard.crop(box)

    sent_size = _mask_frame_size(*region.size)
    frame = region if region.size == sent_size else region.resize(sent_size, Image.LANCZOS)
    sent_mask = (region_mask if region_mask.size == sent_size
                 else region_mask.resize(sent_size, Image.BILINEAR))
    frame = _prefill_mask(frame, sent_mask)

    seed = random.randint(0, 2 ** 63 - 1) if seed is None else int(seed)
    filled, err = _run_klein_mask_job(user_id, frame, sent_mask, seed=seed, timeout=timeout,
                                      klein_model=klein_model, prompt=prompt)
    if err:
        return False, err
    if filled.size != region.size:
        filled = filled.resize(region.size, Image.LANCZOS)
    paste_mask = region_mask.filter(ImageFilter.GaussianBlur(KLEIN_COMPOSITE_FEATHER_PX))
    result = composite_inpaint(original, filled, box, paste_mask)
    try:
        image_encoding.save_edit(
            result, image_path, image_encoding.format_for_path(image_path, original),
            image_encoding.LOSSLESS)
    except (OSError, ValueError) as e:
        return False, {'kind': 'failed', 'detail': f'could not save repaired image: {e}'}
    return True, None


def compare_candidate_boxes(watermark_regions, watermark_bbox):
    """The zones ONE flagged row would hand the clean — the same derivation the
    batch uses (manual regions first, else the detected bbox), so the compare
    judges exactly what the pass will repaint. [] when the row carries neither.
    """
    import json as _json
    if watermark_regions:
        try:
            regions = _json.loads(watermark_regions)
        except (ValueError, TypeError):
            regions = None
        norm = _normalize_boxes(regions or [])
        if norm:
            return norm
    try:
        bbox = _json.loads(watermark_bbox) if watermark_bbox else None
    except (ValueError, TypeError):
        bbox = None
    if isinstance(bbox, list) and len(bbox) == 4:
        return _normalize_boxes([bbox])
    return []


def run_compare(user_id, rows, *, model, image_id=None, seed=None):
    """One compare call, surface-agnostic: pick the sample row, run ONE model.

    `rows` is an iterable of (id, label, path, regions_json, bbox_json) the ROUTE
    built from its own table — the only per-surface part. The rest (sample pick,
    box derivation, model validation, the preview run) must not fork per surface,
    or the two dialogs drift the way the fence sentences once did.

    Returns a JSON-ready dict, never raises: every failure is {'ok': False,
    'error': sentence} because the dialog renders errors inline, per candidate.
    """
    import base64
    from . import klein_edit_helper as keh

    name = (model or '').strip()
    if not name:
        return {'ok': False, 'error': 'name the Klein model to try'}
    if not keh.klein_model_on_disk(name):
        return {'ok': False,
                'error': f'"{name}" is not on disk any more — re-open the dialog '
                         'to refresh the list.'}
    pool = [r for r in rows if compare_candidate_boxes(r[3], r[4])]
    if not pool:
        return {'ok': False,
                'error': 'No flagged image with zones to repaint — run '
                         'Find watermarks first.'}
    if image_id is not None:
        row = next((r for r in pool if r[0] == image_id), None)
        if row is None:
            return {'ok': False, 'error': 'that image is not flagged any more'}
    else:
        row = pool[0]
    rid, label, path, regions_json, bbox_json = row
    if not path or not os.path.isfile(path):
        return {'ok': False, 'error': 'the flagged image file is missing on disk'}
    boxes = compare_candidate_boxes(regions_json, bbox_json)
    seed = random.randint(0, 2 ** 63 - 1) if seed is None else int(seed)
    started = time.monotonic()
    before, after, err = compare_preview(user_id, path, boxes,
                                         klein_model=name, seed=seed)
    if err:
        return {'ok': False, 'image_id': rid, 'model': name, 'seed': seed,
                'error': err.get('detail') or 'the repair failed'}
    return {'ok': True, 'image_id': rid, 'label': label, 'model': name, 'seed': seed,
            'seconds': round(time.monotonic() - started, 1),
            'before': base64.b64encode(before).decode(),
            'after': base64.b64encode(after).decode()}


def compare_preview(user_id, image_path, boxes, *, klein_model, seed,
                    timeout=KLEIN_TIMEOUT, max_side=896):
    """Run ONE model's inpaint on a THROWAWAY copy and hand back preview bytes.

    The judging half of "compare Klein models before the batch": the caller runs
    this once per candidate with the SAME image, boxes and seed, so the only
    variable across the grid is the model — which is the question being asked.

    Deliberately built on `inpaint_watermark_klein` itself rather than a
    parallel path: the preview must show what the real pass WILL do, and two
    implementations of one pass is how previews start lying. The original file
    is never opened for writing — the pass runs on a temp copy that is deleted
    in `finally`, and the copy is EXIF-uprighted first, exactly like the batch's
    own staging (the stored boxes live in the upright frame).

    Returns (before_jpeg_bytes, after_jpeg_bytes, err) — err carries the same
    {'kind','detail'} contract as the pass; both byte payloads are bounded to
    `max_side` so a 4K source does not ship two 4K frames per candidate.
    """
    import tempfile

    def _preview_bytes(img):
        img = img.convert('RGB')
        if max(img.size) > max_side:
            ratio = max_side / max(img.size)
            img = img.resize((max(1, round(img.width * ratio)),
                              max(1, round(img.height * ratio))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=88)
        return buf.getvalue()

    try:
        with Image.open(image_path) as raw:
            upright = ImageOps.exif_transpose(raw).convert('RGB')
    except (OSError, ValueError) as e:
        return None, None, {'kind': 'failed', 'detail': f'unreadable image: {e}'}
    before = _preview_bytes(upright)
    tmp = tempfile.NamedTemporaryFile(suffix='.webp', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        upright.save(tmp_path, 'WEBP', quality=95)
        ok, err = inpaint_watermark_klein(user_id, tmp_path, boxes, seed=seed,
                                          timeout=timeout, klein_model=klein_model)
        if not ok:
            return before, None, err or {'kind': 'failed', 'detail': 'the repair failed'}
        with Image.open(tmp_path) as res:
            after = _preview_bytes(res)
        return before, after, None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def inpaint_watermark_klein(user_id, image_path, boxes, *, seed=None, device='cpu',
                            timeout=KLEIN_TIMEOUT,
                            klein_model=None, prompt=None) -> tuple[bool, dict | None]:
    """Klein, on `image_path`, in place — the 🧽 watermark CLEAN or the ✦ BOX repair.

    ONE entry point, TWO lanes, and `prompt` is the switch (see the module docstring):

    * `prompt` empty → the CLEAN. `boxes` are ERASED on the whole photo first (LaMa /
      cv2 TELEA, no crop), then the WHOLE photo is re-rendered by Klein under the
      instruction "remove watermark" (4 steps, euler, cfg 1, denoise 1.0). The zones no
      longer bound the repaint — they keep Klein from re-inventing the mark it would
      otherwise see in its own reference — and the pass still clears marks the detector
      missed. The result is a new render of the entire frame at its original
      dimensions: nothing is byte-preserved any more, ↩ Restore original is the way back.
    * `prompt` given → the BOX repair: prefill, Klein refine of a padded crop, per-zone
      seam harmonization, and a feathered paste of the `boxes` footprint only. Every
      pixel outside it keeps its ORIGINAL bytes. Unchanged since 2026-07-17.

    Returns the `(ok, error)` tuple contract: `error` is None on success, else
    {'kind', 'detail'} — 'unavailable' when Klein or the prefill engine is not ready,
    'failed' otherwise. The file is overwritten losslessly in its own format; the caller
    is the one that preserved the `.orig` sibling.
    `device` selects the prefill LaMa device on BOTH lanes ('cpu' by default so the
    pending ComfyUI GPU job runs alone; Klein itself always owns the GPU via ComfyUI).
    `klein_model`: see _run_klein_job — the dataset's pick, or None (auto) when the
    caller has no dataset to inherit from."""
    if not is_available():
        return False, {'kind': 'unavailable',
                       'detail': 'Klein inpaint is not ready (ComfyUI unreachable or models missing)'}
    try:
        original = Image.open(image_path).convert('RGB')
    except (OSError, ValueError) as e:
        return False, {'kind': 'failed', 'detail': f'unreadable image: {e}'}
    norm = _normalize_boxes(boxes)
    if not norm:
        return False, {'kind': 'failed', 'detail': 'no valid watermark box'}
    if (prompt or '').strip():
        return _repair_boxes_crop_and_stitch(user_id, image_path, original, norm, seed=seed,
                                             device=device, timeout=timeout,
                                             klein_model=klein_model, prompt=prompt)
    return _clean_full_frame(user_id, image_path, original, norm, seed=seed, device=device,
                             timeout=timeout, klein_model=klein_model)


def _clean_full_frame(user_id, image_path, original, norm=None, *, seed=None,
                      device='cpu', timeout=KLEIN_TIMEOUT,
                      klein_model=None) -> tuple[bool, dict | None]:
    """The 🧽 clean (maintainer's 2026-08-31 recipe): ERASE the detected zones on the
    whole photo, hand that whole photo to Klein under "remove watermark", and write the
    render back at the file's own size.

    TWO STEPS, and the first one is not optional when zones are known. Without it, Klein
    does not delete a logo it can see in its own reference — it REINTERPRETS it, and the
    disc-with-a-mountain came back as a moon in the sky (measured, three runs, three
    seeds; the detector scores that 0 zones and never catches it). Erasing the zones
    first — TELEA/LaMa on the FULL frame, `KLEIN_MASK_EXPAND_PX` of margin, no crop and
    no compositing — removed every ghost while the pass kept clearing what the detector
    had MISSED. That is the July "poisoned reference" doctrine holding on this lane too,
    and it is why the clean is not simply the naked recipe.

    `norm` is the validated zone list. Empty/None is the DEGRADED case: the naked pass
    still runs (it is what clears a tiled mark, which has no zones worth erasing) and
    still removes marks, but the ghost-glyph risk above comes back with it. No caller
    reaches that path today — every one of them derives boxes before calling.

    The frame travels at KLEIN_MASK_MAX_MP at most, snapped to the latent stride and
    never magnified (`_mask_frame_size`, shared with the masked lane) — a 24 MP photo is
    not worth 12× the VRAM on a box that also runs ComfyUI and possibly a training run.
    The margin is applied in SENT pixels, where the erasing actually happens. The render
    comes back at that sent size and is resampled to the ORIGINAL dimensions, because a
    clean must never change the shape of a dataset image."""
    W, H = original.size
    sent_size = _mask_frame_size(W, H)
    frame = original if original.size == sent_size else original.resize(sent_size, Image.LANCZOS)

    if norm:
        # Boxes grown by the margin and re-normalized against the SENT frame — the
        # crop helper does exactly this when the "crop" is the whole picture.
        sw, sh = sent_size
        erase = _crop_boxes_norm((0, 0, sw, sh), sw, sh, norm)
        if erase:
            # Same engine order as the box lane (LaMa, else cv2 TELEA, else abort with
            # 'unavailable' so the caller skips the row and says "install the ML
            # extras"). Aborting is deliberate: with no engine the naked pass would
            # silently paint moons and report success.
            frame, err = _prefill_region(frame, erase, device=device)
            if err:
                return False, err

    seed = random.randint(0, 2 ** 63 - 1) if seed is None else int(seed)
    rendered, err = _run_klein_job(user_id, frame, seed=seed, steps=KLEIN_CLEAN_STEPS,
                                   denoise=KLEIN_DENOISE, timeout=timeout,
                                   klein_model=klein_model, prompt=KLEIN_CLEAN_PROMPT)
    if err:
        return False, err
    result = rendered if rendered.size == (W, H) else rendered.resize((W, H), Image.LANCZOS)
    try:
        image_encoding.save_edit(
            result, image_path, image_encoding.format_for_path(image_path, original),
            image_encoding.LOSSLESS)
    except (OSError, ValueError) as e:
        return False, {'kind': 'failed', 'detail': f'could not save cleaned image: {e}'}
    return True, None


def _repair_boxes_crop_and_stitch(user_id, image_path, original, norm, *, seed=None,
                                  device='cpu', timeout=KLEIN_TIMEOUT, klein_model=None,
                                  prompt=None) -> tuple[bool, dict | None]:
    """The ✦ prompted BOX repair: prefill + Klein full-edit refine of a padded crop +
    per-zone harmonization + feathered composite of the boxes' footprint only.

    Every pixel outside that footprint keeps its ORIGINAL bytes — the promise this lane
    is sold on, and the reason the 2026-08-31 move to full-frame cleaning stopped at the
    clean. `norm` is already normalized/validated by the caller."""
    W, H = original.size
    crop_box = _klein_crop_box(W, H, norm)
    cw, ch = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
    crop_img = original.crop(crop_box)
    scaled_size = _upscale_size(cw, ch)
    scaled_crop = crop_img.resize(scaled_size, Image.LANCZOS)

    union_hard = _hard_mask(crop_box, W, H, norm)   # all zones — the ring's exclusion mask

    # Prefill the mark away BEFORE Klein — its ReferenceLatent is this crop, so a leftover
    # watermark would be reproduced as ghost glyphs. No prefill → no Klein (abort).
    crop_boxes = _crop_boxes_norm(crop_box, W, H, norm)
    prefilled, err = _prefill_region(scaled_crop, crop_boxes, device=device)
    if err:
        return False, err

    seed = random.randint(0, 2 ** 63 - 1) if seed is None else int(seed)
    filled_scaled, err = _run_klein_job(user_id, prefilled, seed=seed, timeout=timeout,
                                        klein_model=klein_model, prompt=prompt)
    if err:
        return False, err
    filled_crop = (filled_scaled if filled_scaled.size == (cw, ch)
                   else filled_scaled.resize((cw, ch), Image.LANCZOS))

    # Per zone: harmonize the patch's tone to its OWN local ring, then paste only that
    # zone's feathered footprint. Chaining `composite_inpaint` keeps every untouched pixel
    # byte-exact (each paste writes only its mask), so multi-mark crops spanning different
    # lighting get each seam corrected locally instead of by one global (wrong) offset.
    original_crop = original.crop(crop_box)
    result = original.convert('RGB').copy()
    for box in norm:
        zone_hard = _hard_mask(crop_box, W, H, [box])
        ring = _seam_ring(zone_hard, union_hard)
        corrected = _harmonize_seam(filled_crop, original_crop, ring)
        zone_mask = zone_hard.filter(ImageFilter.GaussianBlur(KLEIN_COMPOSITE_FEATHER_PX))
        result = composite_inpaint(result, corrected, crop_box, zone_mask)
    try:
        # Preserve the file's format and encode without loss (`image_encoding`, the
        # rule mirror, rotate and crop follow). A WEBP q92 re-save re-compressed the WHOLE image
        # to repaint a few square centimetres — which silently contradicted the
        # "every pixel outside the footprint keeps its ORIGINAL bytes" guarantee
        # this method is built on.
        image_encoding.save_edit(
            result, image_path, image_encoding.format_for_path(image_path, original),
            image_encoding.LOSSLESS)
    except (OSError, ValueError) as e:
        return False, {'kind': 'failed', 'detail': f'could not save repaired image: {e}'}
    return True, None
