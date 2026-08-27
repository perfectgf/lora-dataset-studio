"""🔤 Find text — turning OCR line boxes into the zones a repaint should cover.

The OCR engine (RapidOCR, the same one the Video bank's 🔳 Safe zone pass runs)
answers per LINE: a three-line speech bubble comes back as three thin boxes, a
sound effect as one or two, a subtitle as one per row. Repainting those raw
boxes would leave the leading between the lines untouched — stripes of the
original text's background inside an otherwise repainted bubble — and would
hand the mask editor a confetti of rectangles where the user sees ONE thing.

So this module merges: each line box is padded a little (the pad is also the
inpaint margin — it covers the anti-aliased edge and the outline strokes comic
lettering carries), overlapping boxes are unioned until nothing overlaps any
more, and the result is floored and capped to the exact contract
``normalize_watermark_regions`` enforces on hand-drawn masks — because that is
where these boxes are going. The cleaning levels, the mask editor, ↩ Undo and
the promote path all consume that one channel; this module's whole job is to
produce values that channel accepts.

Pure Python on purpose — no engine import, no I/O — so `node`-speed unit tests
exercise the geometry without onnxruntime, the same split the video lane keeps
between video_text_infer.py (the engine) and video_safe_zone_geometry.py (the
decisions).
"""
from __future__ import annotations

# The pad, in image fractions, applied to each OCR LINE box before merging.
# Two jobs with one number:
#   * it is the merge gap — two boxes closer than 2×pad become one zone, which
#     is what folds the lines of one bubble together while leaving the next
#     bubble (a much wider gap away) alone;
#   * it is the inpaint margin — comic lettering is outlined and anti-aliased,
#     and a mask cut exactly at the glyph edge leaves a ghost of the stroke.
# 0.02 of the image side ≈ 20 px on a 1000 px page. MEASURED, not guessed: on a
# real translated webtoon page the leading between two lines of ONE balloon
# reached 0.035 of the frame — above the 0.03 window a 0.015 pad gives, so the
# balloon split into two zones — while the gap between separate balloons was
# 0.39. The window has an order of magnitude of headroom before it could weld
# two balloons; the leading had almost none before it split one.
TEXT_MERGE_PAD = 0.02

# The two values below REPEAT the hand-mask contract they feed
# (face_dataset_service.WATERMARK_REGION_LIMIT / _MIN_SIDE) rather than import
# it: this module must stay importable without dragging the dataset service in,
# and test_text_regions.py pins the pairs so they cannot drift apart silently —
# the same discipline the backend/infer pairs follow.
TEXT_REGION_CAP = 32
TEXT_REGION_MIN_SIDE = 0.005


def _overlaps(a, b) -> bool:
    return (min(a[2], b[2]) > max(a[0], b[0])
            and min(a[3], b[3]) > max(a[1], b[1]))


def _sane(box):
    """One raw box as [x0,y0,x1,y1] floats clamped to 0..1, or None.

    OCR boxes arrive as 4-or-5-element lists (the fifth is the recognition
    score, already thresholded by the child — dropped here); stored regions
    arrive as clean 4-lists. Anything else — short rows, NaN, inverted or
    zero-area rectangles — is noise and is dropped rather than repaired:
    repairing a corrupt box invents a repaint zone nobody asked for.
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in box[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if not all(v == v and abs(v) != float('inf') for v in (x0, y0, x1, y1)):
        return None
    x0, y0 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, y0))
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _union_all(boxes):
    """Union overlapping boxes until nothing overlaps — a fixed point, not one
    sweep. One sweep is what the watermark locator does and it is enough there
    (three phrases produce three boxes over ONE mark); here line 1 ∪ line 2 can
    newly overlap line 3, so the loop runs until a full pass changes nothing.
    Terminates because every changing pass strictly shrinks the list."""
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        for box in merged:
            for kept in out:
                if _overlaps(kept, box):
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


def text_mask_regions(line_boxes, existing_regions=None, *,
                      pad=TEXT_MERGE_PAD, min_side=TEXT_REGION_MIN_SIDE,
                      cap=TEXT_REGION_CAP):
    """The FINAL region list for ``watermark_regions``, and how many zones the
    cap discarded: ``(regions, dropped)``.

    ``line_boxes`` are the OCR child's normalised per-line boxes (score in
    position 4 tolerated and ignored). ``existing_regions`` are zones the row
    ALREADY carries — a hand-drawn mask, a previous text scan, or the watermark
    detector's box being folded in. They join the union UNPADDED (they are
    final geometry, not glyph-tight lines) and are never silently lost: a scan
    that replaced a hand-drawn zone with only what the OCR saw would undo a
    correction the user believes landed, and — worse — re-flagging a cleaned
    image with ONLY the new text zones would let the next repaint, which always
    restarts from the source pixels, bring the original watermark back.

    ``dropped`` exists because the cap is a real cliff (the hand-mask channel
    accepts 32 zones), and a page with 40 text blocks must say "8 were not
    covered" somewhere rather than quietly repainting 32 and reporting done.
    The KEPT zones are the biggest ones — on a text-heavy page the big blocks
    are the bubbles and captions, the tail is stray single-word noise.
    """
    padded = []
    for box in (line_boxes or []):
        sane = _sane(box)
        if sane is None:
            continue
        x0, y0, x1, y1 = sane
        padded.append([max(0.0, x0 - pad), max(0.0, y0 - pad),
                       min(1.0, x1 + pad), min(1.0, y1 + pad)])
    for box in (existing_regions or []):
        sane = _sane(box)
        if sane is not None:
            padded.append(sane)
    merged = _union_all(padded)
    merged = [b for b in merged
              if (b[2] - b[0]) >= min_side and (b[3] - b[1]) >= min_side]
    merged.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    dropped = max(0, len(merged) - cap)
    return ([[round(v, 4) for v in b] for b in merged[:cap]], dropped)
