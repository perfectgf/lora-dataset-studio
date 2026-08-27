"""🔤 The text-zone filler — bubble-aware cleaning, on CPU, no model at all.

WHY THIS EXISTS. The repaint level used to hand LaMa the whole RECTANGLE of a
text zone. A rectangle overshoots a speech bubble by construction (the merge
pad that folds lines together also bites the outline), so LaMa — which
regenerates everything it covers — kept eating balloon outlines and cartouche
borders. The maintainer's pages showed exactly that: emptied bubbles with
half-dissolved rims.

WHAT IT DOES INSTEAD, per zone — the scanlation cleaners' method, measured on
real webtoon pages rather than borrowed as folklore:

  1. Ink mask (Otsu). Every ink component that TOUCHES the zone border is
     PRESERVED: a letter is a small closed shape inside the zone, while a
     balloon outline or the art always crosses the box edge. That one rule is
     what makes broken bubbles impossible.
  2. The background is judged in the RING right around the kept glyphs —
     inside the balloon by construction. Judging the whole rectangle mixes in
     the art and misclassifies every real bubble (measured).
  3. Near-pure background (≥60 % of ring pixels within ±2 of the dominant
     colour): flatten the whole glyph neighbourhood to that EXACT colour.
     Not a median, not an inpaint: these pages are pure-255 white with a
     252-254 JPEG-ringing veil around each letter, and any averaging fill
     leaves the text as a one-grey-level relief the eye reads as a ghost —
     found only because the maintainer preferred LaMa's output at 1:1.
  4. Genuinely graded background: TELEA inpaint over the deviating pixels.
  5. Busy background (art behind the text): nothing is painted here — the
     glyph-tight boxes are RETURNED so the caller can run LaMa on just the
     letters instead of the whole rectangle.

Needs cv2 + numpy only — both guaranteed by the `video_text` capability this
lane already requires (the probe imports them). No torch, no weights, no GPU.

Protocol (one JSON line in, one out — a batch, like video_text_infer):
  stdin  : {"items": [{"image_path": "<abs>", "regions": [[x0,y0,x1,y1], ...]},
            ...], "cancel_file": path|null}
  stdout : {"ok": true, "results": {"<image_path>": {"ok": bool,
            "filled": N, "busy_boxes": [[x0,y0,x1,y1], ...], "error": str?}},
            "stopped": bool}
           {"ok": false, "error": "<ExcType>: <message>"}
  stderr : "[fill] i/N" progress lines.

Regions and busy_boxes are NORMALISED [x0,y0,x1,y1] in 0..1 — the exact
contract the mask channel and the LaMa router already speak. Images are
rewritten IN PLACE (the caller hands staged copies, never user files), with
unicode-safe I/O throughout: plain cv2.imread/imwrite cannot open non-ASCII
paths on Windows, the exact failure that produced a phantom 'cancelled' on a
real bank.
"""
from __future__ import annotations
import json
import os
import sys
from _harness import _cancel_requested, _emit, _log

# The measured dials, named so the numbers carry their reasons:
# ±2 of the dominant ring colour counts as "that colour" (JPEG dither width),
# and 60 % of the ring at that colour is what separates a flat balloon from a
# graded one on real pages (measured 84-92 % on true bubbles).
MODE_TOLERANCE = 2
MODE_SHARE_MIN = 0.60
# A ring std below this is a plain background TELEA can rebuild; above it the
# text sits on art and only a glyph-tight repaint is honest.
RING_STD_MAX = 22.0
# Glyph mask growth: 5 px covers anti-aliased edges; the 11 px neighbourhood is
# what the flatten repaints (ringing veil included); the ring is sampled at
# 13 px. Growing past the ring would sample what we are about to repaint.
GLYPH_DILATE = 5
FLATTEN_DILATE = 11
RING_DILATE = 13
BUSY_BOX_PAD = 3
# Component floors: real glyphs at webtoon sizes are ≥ 3 px a side (a comma is
# 5-8 px at 720 px width); anything smaller is grain — measured: raw noise in
# a zone otherwise produced 495 one-pixel "glyphs" and 495 LaMa boxes.
MIN_GLYPH_SIDE = 3
MIN_GLYPH_AREA = 12
# Busy boxes are merged (transitive union) and capped: past this many separate
# spots the honest move is the WHOLE zone again — the caller's old behaviour,
# never worse — rather than a shotgun of tiny repaints.
BUSY_BOX_CAP = 24


def _read_bgr(path):
    import cv2
    import numpy as np
    try:
        image = cv2.imread(path)
    except Exception:  # noqa: BLE001
        image = None
    if image is not None:
        return image
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def _write_bgr(path, image) -> bool:
    """In-place, unicode-safe, format by extension. WebP is written LOSSLESS —
    the bank stages lossless WebP working copies and a lossy rewrite here
    would degrade every pixel of the page to repaint three bubbles."""
    import cv2
    import numpy as np
    ext = os.path.splitext(path)[1].lower() or '.png'
    params = []
    if ext == '.webp':
        params = [cv2.IMWRITE_WEBP_QUALITY, 101]      # >100 = lossless
    elif ext in ('.jpg', '.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, 97]
    try:
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            return False
        np.asarray(buf).tofile(path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _fill_zone(image, region, busy_boxes):
    """Clean ONE zone in place. Returns 'filled' | 'busy' | 'skipped'."""
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    x0, y0, x1, y1 = region
    px0, py0 = max(0, int(x0 * w)), max(0, int(y0 * h))
    px1, py1 = min(w, int(x1 * w)), min(h, int(y1 * h))
    if px1 - px0 < 4 or py1 - py0 < 4:
        return 'skipped'
    zone = image[py0:py1, px0:px1]
    grey = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    _t, ink = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if float((ink > 0).mean()) > 0.55:
        # More "ink" than background: light-on-dark lettering — flip so the
        # letters, not the ground, are the components we reason about.
        ink = cv2.bitwise_not(ink)
    n, labels, st, _c = cv2.connectedComponentsWithStats(ink, connectivity=8)
    zh, zw = ink.shape
    keep = np.zeros_like(ink)
    kept = 0
    glyph_stats = []
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if x == 0 or y == 0 or (x + cw) >= zw or (y + ch) >= zh:
            continue          # crosses the box edge: outline or art — preserved
        if cw < MIN_GLYPH_SIDE or ch < MIN_GLYPH_SIDE or area < MIN_GLYPH_AREA:
            continue          # grain, not lettering (see the floors above)
        keep[labels == i] = 255
        glyph_stats.append((x, y, cw, ch))
        kept += 1
    if not kept:
        # Nothing safely fillable (text glued to the outline, or the box holds
        # no closed ink at all). Hand the whole zone to the caller's repaint —
        # exactly what happened before this worker existed, never worse.
        busy_boxes.append([x0, y0, x1, y1])
        return 'busy'
    kernel = np.ones((GLYPH_DILATE, GLYPH_DILATE), np.uint8)
    keep = cv2.dilate(keep, kernel)
    ring = ((cv2.dilate(keep, np.ones((RING_DILATE, RING_DILATE), np.uint8)) > 0)
            & (keep == 0) & (ink == 0))
    bg = zone[ring].reshape(-1, 3)
    if not bg.size:
        busy_boxes.append([x0, y0, x1, y1])
        return 'busy'
    vals, counts = np.unique(bg, axis=0, return_counts=True)
    mode = vals[counts.argmax()]
    mode_share = float((np.abs(bg.astype(np.int16) - mode.astype(np.int16))
                        .max(axis=1) <= MODE_TOLERANCE).mean())
    ring_std = float(bg.std(axis=0).mean())
    near = cv2.dilate(keep, np.ones((FLATTEN_DILATE, FLATTEN_DILATE), np.uint8)) > 0
    if mode_share >= MODE_SHARE_MIN:
        zone[near | (keep == 255)] = mode
        return 'filled'
    if ring_std < RING_STD_MAX:
        dist = np.abs(zone.astype(np.int16) - mode.astype(np.int16)).max(axis=2)
        mask = (((dist > 4) & near) | (keep == 255)).astype(np.uint8) * 255
        zone[:] = cv2.inpaint(zone, mask, 5, cv2.INPAINT_TELEA)
        return 'filled'
    boxes = [[max(0, x - BUSY_BOX_PAD), max(0, y - BUSY_BOX_PAD),
              min(zw, x + cw + BUSY_BOX_PAD), min(zh, y + ch + BUSY_BOX_PAD)]
             for x, y, cw, ch in glyph_stats]
    boxes = _union_boxes(boxes)
    if len(boxes) > BUSY_BOX_CAP:
        busy_boxes.append([x0, y0, x1, y1])
        return 'busy'
    for bx0, by0, bx1, by1 in boxes:
        busy_boxes.append([(px0 + bx0) / w, (py0 + by0) / h,
                           (px0 + bx1) / w, (py0 + by1) / h])
    return 'busy'


def _union_boxes(boxes):
    """Union overlapping pixel boxes to a fixed point — glyphs of one word
    become one repaint box instead of a box per letter."""
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        for box in merged:
            for kept in out:
                if (min(kept[2], box[2]) > max(kept[0], box[0])
                        and min(kept[3], box[3]) > max(kept[1], box[1])):
                    kept[0] = min(kept[0], box[0])
                    kept[1] = min(kept[1], box[1])
                    kept[2] = max(kept[2], box[2])
                    kept[3] = max(kept[3], box[3])
                    changed = True
                    break
            else:
                out.append(box)
        merged = out
    return merged


def main() -> int:
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        _emit({'ok': False, 'error': f'bad json: {e}'})
        return 1
    items = [i for i in (req.get('items') or []) if i.get('image_path')]
    cancel_file = req.get('cancel_file') or None
    if not items:
        _emit({'ok': True, 'results': {}, 'stopped': False})
        return 0
    try:
        import cv2  # noqa: F401 — the dependency probe, like video_text_infer's
        import numpy  # noqa: F401
    except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
        _emit({'ok': False, 'error': f'fill deps missing: {type(e).__name__}: {e}'})
        return 1

    total = len(items)
    _log(f'[fill] 0/{total}')
    results = {}
    stopped = False
    for i, item in enumerate(items, start=1):
        if _cancel_requested(cancel_file):
            stopped = True
            break
        path = item['image_path']
        regions = [r for r in (item.get('regions') or [])
                   if isinstance(r, list) and len(r) >= 4]
        image = _read_bgr(path)
        if image is None:
            results[path] = {'ok': False, 'filled': 0, 'busy_boxes': [],
                             'error': 'unreadable image'}
            _log(f'[fill] {i}/{total}')
            continue
        busy_boxes = []
        filled = 0
        for region in regions:
            if _fill_zone(image, [float(v) for v in region[:4]],
                          busy_boxes) == 'filled':
                filled += 1
        if filled and not _write_bgr(path, image):
            results[path] = {'ok': False, 'filled': 0, 'busy_boxes': [],
                             'error': 'could not write image'}
            _log(f'[fill] {i}/{total}')
            continue
        results[path] = {'ok': True, 'filled': filled,
                         'busy_boxes': [[round(v, 4) for v in b]
                                        for b in busy_boxes]}
        _log(f'[fill] {i}/{total}')

    _emit({'ok': True, 'results': results, 'stopped': stopped})
    return 0


if __name__ == '__main__':
    sys.exit(main())
