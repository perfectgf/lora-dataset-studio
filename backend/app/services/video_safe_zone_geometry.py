"""🔳 Where the picture actually IS — bands, burned-in text, and what is left.

PURE ARITHMETIC, no PIL, no OCR, no database. Everything here takes numbers and
returns numbers, which is what lets `node --test`'s Python counterpart exercise
the whole design on forged frames in milliseconds — the same split
`video_metrics.py` keeps for the quality pass and `video_clip_dedup` keeps for
the similarity one. The pass that decodes frames and runs the OCR engine lives
next door in `video_safe_zone.py`.

WHY THIS EXISTS AT ALL. Every public video-curation pipeline of the last two
years filters burned-in text FIRST, before anything expensive: Wan measures a
text-coverage ratio with a light OCR as step one of its cleaning, NVIDIA's Cosmos
Curator ships an "Artificial Text Filter" over PaddleOCR, and HunyuanVideo 1.5
crops away subtitles and logos and DROPS a clip whose crop keeps less than 60 %
of the frame. The reason is the same one that makes a corner watermark expensive:
a subtitle that sits in the same rectangle of every frame of every clip is the
most consistent thing in the dataset, so it is among the first things a LoRA
learns to draw — and unlike a watermark it also teaches the model that the bottom
sixth of a frame is a place where letters live.

WHAT THIS LANE DOES DIFFERENTLY, AND IT IS THE WHOLE DOCTRINE. Those pipelines
DROP. This measures and stores raw numbers; the verdict is derived at read time
against a cut the user moves (`video_metrics.verdicts`), so retuning re-sorts a
bank with nothing rescanned and nothing is ever deleted. The published figures
(60 % kept, "under 50 % left is hard to recover") are worth quoting and are
therefore in the panel HINTS — never in a default.

THREE FRAMES, AND WHY THE VOTE IS THE POINT. One frame cannot tell a subtitle
from a shop sign, or a letterbox from a fade-out. The same three instants the
embedding pass uses (`video_clip_search.frame_times`) are measured, and only what
survives ACROSS them is called structural:

  bands → the MIN per side. A band on all three frames is the container; a band
          on one is a frame that happened to fade. The min is also the only
          aggregation that cannot over-claim: it is exactly "how much band every
          frame has", so a safe zone built on it is safe on every frame.
  text  → a box needs a partner in ANOTHER frame (IoU ≥ TEXT_IOU) to count. A
          subtitle, a chyron and a text watermark stay put while their letters
          change; a shop sign in a panning shot moves, and a newspaper held up
          for one instant is in one frame only. Those are scene content and this
          module deliberately does not flag them.

THE FRONTIER WITH 🔖 WATERMARKS, stated once because the two look adjacent and
are not. That pass sends ONE frame to a SigLIP2 classifier and keeps a single
scalar `watermark_score` — it answers "does a logo sit on this footage", it
discards the regions it was given, and it needs a ~2 GB torch install. This one
answers "which rectangle of the frame is not picture", from bands plus OCR, on
CPU, in the app's own interpreter. A text watermark is seen by both, and gets two
different verdicts on purpose: one says "this shot is stamped", the other says
"crop here and you lose 14 % of the frame". Neither replaces the other, and this
module never writes a key that pass owns.
"""
from __future__ import annotations

# --- bands ------------------------------------------------------------------------

# Long side of the luma grid a band scan runs on. Bands are a MACRO feature — the
# image lane reduces to a small copy for the same reason (image_provenance
# _BARS_PROBE) — and reducing first does two things at once: it bounds the cost
# to a fixed few thousand comparisons whatever the source resolution, and it
# averages away the codec artefacts that otherwise end a scan one row early (a
# single stray bright pixel inside the padding of a heavily compressed rush).
BAND_PROBE = 160

# How flat a row/column has to be to count as band, as a spread in 0..255 luma.
# The image lane tests DARKNESS instead (`max(...) <= _BARS_DARK`), which is
# right for stills off a chat app and wrong here: a rush padded by an editor, a
# phone capture of a vertical video and a scan of a 4:3 broadcast are padded with
# grey and white at least as often as with black. Uniformity covers all of them
# with one rule. It also costs a known false positive — a genuinely flat sky, or
# the black surround of a dark-themed screen recording, reads as a band — which
# is why this cut ships with NO default and says so in its hint. The image lane
# reached the same conclusion from the other side and excludes its own `bars`
# flag from automatic rejection (image_bank_service._PIPELINE_EXCLUDED_REJECT_FLAGS).
BAND_TOLERANCE = 6

# The most one side may ever claim. Not a tuning knob — a statement that a "band"
# taking nearly half the frame is not a band. Past this the scan has walked into
# a fade, a title card or a flat colour, and the frame has no interior left to
# describe; `bands_of_grid` answers None for it and the vote simply ignores it.
MAX_BAND = 0.45


def _spread(values):
    return max(values) - min(values)


def bands_of_grid(rows, *, tolerance=BAND_TOLERANCE, cap=MAX_BAND):
    """{'top','bottom','left','right'} as fractions of ONE frame, or None.

    `rows` is a luma grid, row-major, values 0..255 — see `luma_grid` in
    video_safe_zone.py for the PIL adapter that produces one.

    None means "this frame cannot describe a container": every side ran to the
    cap, so the frame is uniform (a fade, a black slug, a plain title card).
    Returning zeros there would be a measurement — the claim that the frame is
    full of picture — and returning the caps would let one fading frame shrink a
    whole shot's safe zone. Neither is true, so the frame abstains.

    LETTERBOX AND PILLARBOX FALL OUT OF THE SAME RULE rather than being two code
    paths: a letterboxed frame has uniform ROWS at top and bottom while every
    column crosses the picture, and a pillarboxed one is the transpose. A frame
    with both is measured on both axes at once.
    """
    height = len(rows)
    width = len(rows[0]) if height else 0
    if height < 8 or width < 8:
        return None
    cols = [tuple(row[x] for row in rows) for x in range(width)]
    v_limit = max(1, int(height * cap))
    h_limit = max(1, int(width * cap))

    def walk(limit, flat):
        i = 0
        while i < limit and flat(i):
            i += 1
        return i

    top = walk(v_limit, lambda i: _spread(rows[i]) <= tolerance)
    bottom = walk(v_limit, lambda i: _spread(rows[height - 1 - i]) <= tolerance)
    left = walk(h_limit, lambda i: _spread(cols[i]) <= tolerance)
    right = walk(h_limit, lambda i: _spread(cols[width - 1 - i]) <= tolerance)
    if top >= v_limit and bottom >= v_limit and left >= h_limit and right >= h_limit:
        return None
    return {'top': top / height, 'bottom': bottom / height,
            'left': left / width, 'right': right / width}


SIDES = ('top', 'bottom', 'left', 'right')


def vote_bands(per_frame):
    """The bands of a SHOT, from its frames' bands — MIN per side, or None.

    The min, and the argument is worth keeping because the two obvious
    alternatives are both wrong here. A MEAN would report a band nobody ever saw:
    a clip whose middle frame fades would carry a permanent phantom letterbox of
    a third of the real thing. A MAX would let the single worst frame speak for
    the shot, which is the fade problem again in its strongest form. The MIN is
    the only one whose answer is TRUE OF EVERY FRAME — which is exactly the
    property a safe zone has to have, since the crop it justifies is applied to
    all of them.

    Frames that abstained (`bands_of_grid` → None) are dropped rather than read
    as zeros: "this frame is a fade" and "this frame is full of picture" must not
    collapse. None when every frame abstained — the shot is a fade, and
    `luma_min` is the metric that already says so.
    """
    usable = [b for b in (per_frame or []) if b]
    if not usable:
        return None
    return {side: min(b[side] for b in usable) for side in SIDES}


# ── TWO SURFACES, ONE VOCABULARY, TWO CONTRACTS ──────────────────────────────
# The image bank measures letterbox too (`image_provenance.bars_ratio`, the
# `bars_ratio` column, the `bank.bars_max` cut, the `bars` flag). This module
# deliberately shares its WORDS and deliberately does NOT share its shape, and
# CLAUDE.md's two-surfaces rule says to write down which is which — because the
# next person to notice the duplication will reach for the merge, and the merge
# silently deletes the half this lane needs.
#
# SHARED, and must stay shared:
#   * the name `bars_ratio` and the cut `bars_max`;
#   * the arithmetic that folds four sides into that one number,
#     `max(top + bottom, left + right)` — so 0.12 means the same thing on a
#     still and on a shot, and a user carries their calibration across.
#     Pinned by test_the_banded_share_is_the_larger_axis_and_matches_the_image_lane,
#     which reads the image lane's source.
#
# NOT shared, on purpose, and NOT a bug to reconcile:
#   * WHAT COUNTS AS A BAND. The image lane tests DARKNESS (`max(...) <=
#     _BARS_DARK`); this one tests UNIFORMITY. A still off a chat app is padded
#     black; a rush is padded black, white, or grey depending on who exported
#     it, so darkness alone misses most of them here. Uniformity is looser and
#     costs a known false positive (a flat sky) — which is why this cut ships
#     with no default and the image lane's ships 0.04.
#   * HOW MANY IMAGES ANSWER. There is one still; there are three frames, and
#     the per-side MIN across them is what separates a container from a fade.
#     A still has no fade to survive.
#   * THE SHAPE OF THE ANSWER. The image lane stores ONE scalar. This one stores
#     FOUR fractions (`safe_bands`) and derives the scalar from them, because
#     `safe_rect` has to know WHICH edges to cut — a scalar cannot say whether
#     0.24 is a letterbox or a pillarbox, and a crop needs to know.
#
# So: unify the vocabulary if it ever drifts. Do NOT collapse `vote_bands` into
# a single-image call, and do NOT reduce `safe_bands` to a scalar to "match" the
# image column — that throws away the input the safe zone is computed from.


def bars_ratio(bands):
    """The ONE number `bars_max` cuts on: the larger of the two banded shares.

    `max(top + bottom, left + right)`, deliberately IDENTICAL to the formula the
    image lane's `image_provenance.bars_ratio` uses — same question, same name,
    same arithmetic, so `0.12` means the same thing on a still and on a shot and
    a user who calibrated one cut can carry the number to the other. The two
    DETECT differently (that one tests darkness on one image, this one tests
    uniformity and votes across three) and that difference is the feature; a
    second meaning for the same number would not be.

    The larger rather than the sum, because letterbox and pillarbox are answers
    to the same question — "how much of this frame is container" — asked on two
    axes, and a 2.35:1 film in a 16:9 box has bands on one axis only.
    """
    if not bands:
        return None
    return max(bands['top'] + bands['bottom'], bands['left'] + bands['right'])


def band_rect(bands):
    """The bands as a rectangle, (x0, y0, x1, y1) in 0..1."""
    if not bands:
        return (0.0, 0.0, 1.0, 1.0)
    return (bands['left'], bands['top'], 1.0 - bands['right'], 1.0 - bands['bottom'])


# --- burned-in text ---------------------------------------------------------------

# How much two boxes must overlap to be called the same zone seen twice. 0.3, not
# the 0.5 an object detector would use, and the reason is what the boxes ARE: a
# subtitle keeps its position and its baseline while its WIDTH changes with the
# sentence, so two readings of one subtitle line routinely share a y-band and
# half a width. At 0.5 the second line of a conversation stops matching the first
# and a real subtitle track reads as a pile of isolated scene text. At 0.3 they
# match, while a shop sign in one corner and a chyron in the other still score 0.
TEXT_IOU = 0.3

# Frames a zone must appear in to count as burned INTO the footage rather than
# filmed IN it. Two, not three: shots under MIN_SPAN_FOR_THREE_S contribute only
# a couple of frames, and a rule that needed all three would silently exempt
# every short shot — the shots a cut-heavy bank is mostly made of.
TEXT_MIN_FRAMES = 2


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def structural_text(frames_boxes, *, min_frames=TEXT_MIN_FRAMES, iou=TEXT_IOU):
    """The text zones that are burned into the SOURCE — [(x0, y0, x1, y1), ...].

    `frames_boxes` is one list of normalised boxes per frame, in frame order.
    Boxes are linked to boxes in OTHER frames that overlap them by at least
    `iou`, transitively; a group seen in `min_frames` distinct frames is
    structural and is reported as the BOUNDING BOX of its members.

    THE BOUNDING BOX, not the average and not the first: a subtitle drifts a few
    pixels and grows a line, and a safe zone computed from an average would leave
    a sliver of letters inside it — which is the one outcome that makes the whole
    measurement worthless, because a sliver of letters trains just as well as a
    whole one.

    Linking across frames ONLY (never within one frame) is what keeps two
    subtitles stacked on one frame from merging into one tall box: they are two
    zones, and each finds its own partner in the next frame.

    `min_frames` IS CLAMPED to the number of frames there are, and that clamp is
    a real decision rather than defensive arithmetic. Shots under
    `video_clip_search.MIN_SPAN_FOR_THREE_S` contribute ONE frame, so an
    unclamped rule would report every sub-second shot as carrying no burned text
    — a hard zero, about footage it never got to compare. Between the two
    mistakes available on one frame, this takes the visible one: a shop sign in a
    0.9 s shot raises an advisory flag the user glances at and dismisses, while a
    subtitle silently reported as absent is never seen at all. `safe_zone_frames`
    is stored beside the reading so the difference stays legible.

    Returned sorted, so a stored measurement is byte-stable across runs and a
    test can compare it without sorting first.
    """
    frames_boxes = list(frames_boxes or [])
    min_frames = max(1, min(min_frames, len(frames_boxes)))
    flat = [(i, tuple(box)) for i, boxes in enumerate(frames_boxes)
            for box in (boxes or [])]
    parent = list(range(len(flat)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i][0] == flat[j][0]:
                continue                    # same frame — never linked, see above
            if _iou(flat[i][1], flat[j][1]) >= iou:
                parent[find(i)] = find(j)

    groups = {}
    for idx, (frame_i, box) in enumerate(flat):
        groups.setdefault(find(idx), []).append((frame_i, box))

    out = []
    for members in groups.values():
        if len({f for f, _ in members}) < min_frames:
            continue
        boxes = [b for _, b in members]
        out.append((min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes)))
    return sorted(out)


def union_area(boxes):
    """Exact area of the UNION of axis-aligned boxes, in 0..1.

    Exact, not a sum, because summing is wrong in the direction that matters:
    two OCR boxes over one subtitle line overlap heavily, and a summed
    `text_coverage` would double-count them and flag a clip whose text takes 3 %
    of the frame at a cut set for 5 %.

    A horizontal sweep — cut at every distinct y, merge the x-spans alive in each
    band — rather than the obvious compressed GRID. Both are exact and the grid
    is four lines shorter; it is also O(n³), and n is not always small. A shot
    off a screen recording, a subtitled film with a station logo and a scoreboard,
    a scan of a printed page: RapidOCR happily returns dozens of boxes per frame,
    and at n = 100 the grid is hundreds of millions of point-in-box tests — a
    single clip stalling the pass for minutes with the progress bar frozen on it.
    The sweep is O(n² log n) and lands in single-digit milliseconds there.
    """
    boxes = [b for b in (boxes or []) if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return 0.0
    cuts = sorted({b[1] for b in boxes} | {b[3] for b in boxes})
    total = 0.0
    for lower, upper in zip(cuts, cuts[1:]):
        height = upper - lower
        if height <= 0:
            continue
        alive = sorted((b[0], b[2]) for b in boxes
                       if b[1] <= lower and b[3] >= upper)
        width = 0.0
        reach = None
        for x0, x1 in alive:
            if reach is None or x0 > reach:
                width += x1 - x0
                reach = x1
            elif x1 > reach:
                width += x1 - reach
                reach = x1
        total += width * height
    return total


# --- the safe zone ----------------------------------------------------------------

def _area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _overlaps(rect, box):
    return (box[0] < rect[2] and box[2] > rect[0]
            and box[1] < rect[3] and box[3] > rect[1])


def safe_rect(bands, text_boxes):
    """(x0, y0, x1, y1) — the rectangle that holds no band and no burned text.

    Starts at the band rectangle and then cuts EDGES, cheapest cut first, until
    nothing structural is left inside. Edge cuts rather than the maximal empty
    rectangle, and the reason is what the answer is FOR: this number exists to
    tell a user what a crop would cost them, and a crop is four edges. The
    maximal-empty-rectangle answer is also routinely a tall thin sliver beside a
    centred logo — technically larger, useless as a training frame, and
    impossible to explain in a tooltip.

    Cheapest-first is what makes a subtitle cost its own height instead of half
    the picture: cutting a bottom-third caption from the bottom loses the caption
    band, while cutting it from the left loses everything left of it.

    A zone in the MIDDLE of the frame has no cheap cut, and that is not a bug to
    smooth over — it is the finding. The rectangle collapses towards nothing, the
    resulting `safe_area` is small, and that small number is the honest answer to
    "can this clip be saved by cropping": no.
    """
    rect = list(band_rect(bands))
    boxes = [tuple(b) for b in (text_boxes or [])]
    # Bounded rather than `while True`: each pass either removes a box or cannot,
    # so len(boxes) rounds is the ceiling — the bound is a guard against a
    # degenerate float, never a normal exit.
    for _ in range(len(boxes) + 1):
        live = [b for b in boxes if _overlaps(rect, b)]
        if not live:
            break
        best = None
        for box in live:
            for candidate in (
                    (box[2], rect[1], rect[2], rect[3]),      # cut from the left
                    (rect[0], rect[1], box[0], rect[3]),      # cut from the right
                    (rect[0], box[3], rect[2], rect[3]),      # cut from the top
                    (rect[0], rect[1], rect[2], box[1])):     # cut from the bottom
                loss = _area(rect) - _area(candidate)
                # Ties broken by the candidate itself, so the answer does not
                # depend on dict/set ordering — a stored measurement that moved
                # between runs would look like footage that changed.
                key = (round(loss, 9), candidate)
                if best is None or key < best[0]:
                    best = (key, candidate)
        rect = list(best[1])
        if _area(rect) <= 0:
            return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = (round(max(0.0, min(1.0, v)), 4) for v in rect)
    if x1 <= x0 or y1 <= y0:
        return (0.0, 0.0, 0.0, 0.0)
    return (x0, y0, x1, y1)


def safe_area(rect):
    """The share of the frame that rectangle keeps. The number the published
    figures are about: HunyuanVideo 1.5 discards a clip whose crop keeps under
    60 % of the frame, and under ~50 % there is not enough picture left to be
    worth the trouble. Both belong in the panel's hint and in NO default — they
    were set for a web-scale crawl, and a bank of deliberately-shot rushes is not
    one."""
    return _area(rect)
