"""The multi-scale consensus locate — every rule of the recipe, held pure.

The measured story (ground-truth bench: 12 stamped scenarios over 3 base
images, 57 marks in known positions, plus the maintainer's two test images).
The single full-frame DINO pass finds 4 of 7 logo copies and 17 of the 57
stamped marks. Sweeping tiles and keeping only boxes BOTH prompt phrasings
agree on lifts that a long way — but three of the rules AROUND the sweep were
throwing the extra findings straight back out, and each one had to be undone
before the depth showed up in the result. Reverting them one at a time from
what this branch ships, everything else held:

    rule reverted to what it was            rappel GT   precision   tiled photo
    tile floor 200px -> 250 (no 3x3 at 720)   46 -> 28    92 -> 86%   14
    merge by _same_mark -> bare overlap       46 -> 43    92 -> 95%   14 -> 9
    coverage union -> sum, guard 0.50 -> 0.40 46 -> 45    92 -> 100%  14

The last line is the one worth reading twice: reverting it SCORES BETTER on
precision, because the summed coverage saturated and the guard then blanked
every zone of every photo with fewer than three of them. Reporting nothing is
always precise. That is what put the corner-logo scenarios at 0 of 3 located,
and it is the whole reason the maintainer's verdict was "the detection is not
perfect".

On the maintainer's two images: the seven-logo photo reports 7 whole boxes
before and after, the tiled stock photo goes from 12 zones to 14, and the same
photo after a clean stays at 0.

Two rules joined them on 2026-08-31, both measured on the same bench, and both
about what a zone SAYS rather than where it is:

    what changed                        recall   precision   completeness
    geometry rescue (timid boxes)       46/57 =  46/50->43/47  0.71 -> 0.79
    untiled wall-to-wall claim          46/57 =  92% =         =

The first one exists because a zone that covers half a mark is a zone the
clean cannot finish: the emblem left outside the box was re-rendered as a ghost
glyph. "Completeness" is how much of a stamped mark its zone covers, and it is
measured against "spill" (how much of the zone falls outside every mark, 0.412
-> 0.416) — otherwise a rule that grows everything would score perfectly. The
three zones that disappear from the count are fragments of one logo fused into
one whole box, which is the point of the rule and not a cost of it.

The second one exists because on an image too small to tile the coverage union
is made of failed localisations, not of found marks: it read 0.59 on a stock
thumbnail and blanked the word the same pass had located at 0.65.

torch never loads here: the recipe's decisions (tile plan, window mapping,
consensus, merge, wall-to-wall rule) are pure functions, which is what makes
them testable at all — the GPU sweep only feeds them boxes.
"""
import importlib.util
import os

import pytest


def _infer():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'infer', 'watermark_detect_infer.py')
    spec = importlib.util.spec_from_file_location('watermark_detect_infer_dz', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the adaptive tile plan ---------------------------------------------------

def test_small_images_keep_the_legacy_single_pass_byte_for_byte():
    infer = _infer()
    assert infer.tile_plan((480, 360)) == [(1, infer.LOCATE_BOX_THRESHOLD)]


def test_big_images_add_every_grid_whose_tiles_stay_seeable():
    infer = _infer()
    # The maintainer's tiled stock photo: 800 short side / 3 = 267px tiles,
    # and the 3x3 sweep owns 11 of its 12 zones — the plan must include it.
    assert infer.tile_plan((1200, 800)) == [
        (1, infer.LOCATE_BOX_THRESHOLD), (2, infer.LOCATE_TILE_THRESHOLD),
        (3, infer.LOCATE_TILE_THRESHOLD)]
    assert infer.tile_plan((2048, 1365))[-1] == (3, infer.LOCATE_TILE_THRESHOLD)
    # 500/2 = 250 earns the 2x2; 500/3 = 167 is below the tile floor.
    assert infer.tile_plan((500, 900)) == [
        (1, infer.LOCATE_BOX_THRESHOLD), (2, infer.LOCATE_TILE_THRESHOLD)]


def test_an_ordinary_portrait_photo_earns_the_deep_sweep():
    """The shortfall that cost the most and was the hardest to see: at a 250px
    floor a 720px-wide photo got NO 3x3 sweep, because 720/3 = 240 missed by
    ten pixels. Both reference images clear either floor (1365/3, 800/3), so
    nothing in the bench noticed — while on the ground truth it was 17 of 57
    marks. Anything that raises the floor back over 240 gives them away again."""
    infer = _infer()
    for size in ((720, 1018), (720, 1294), (720, 1532)):
        assert infer.tile_plan(size)[-1] == (3, infer.LOCATE_TILE_THRESHOLD), (
            f'{size} lost its 3x3 sweep — 17 of the 57 ground-truth marks '
            'live in exactly that grid')


def test_the_seam_overlap_is_the_measured_one():
    """0.20 is the TOP of a curve swept in BOTH directions on the ground
    truth, not a direction followed until it stopped paying: at 0.12 the sweep
    finds two more stamped marks and pays fifteen false zones for them
    (precision 92% -> 74%), at 0.25 the tiled photo drops from 14 zones back to
    11 and precision falls to 76%. Moving it needs a better measurement, which
    is what this assertion is here to demand."""
    infer = _infer()
    assert infer.TILE_OVERLAP == 0.20
    # ...and it is a real growth: each window is a fifth of a tile wider per
    # side than its share, so a mark on a seam lands whole in one of them.
    x1, _y1, x2, _y2 = infer.tile_windows((1500, 900), 3)[0]
    assert (x2 - x1) - 500 == pytest.approx(500 * infer.TILE_OVERLAP, abs=1)


def test_tile_floor_is_above_the_full_frame_floor():
    """The whole reason tiles do not flood the scan with junk — a tile promotes
    texture to "a logo" at the full-frame floor (measured: coverage blew past
    the wall-to-wall guard and zeroed a 7-logo image)."""
    infer = _infer()
    assert infer.LOCATE_TILE_THRESHOLD > infer.LOCATE_BOX_THRESHOLD


def test_windows_cover_the_frame_and_overlap_their_seams():
    infer = _infer()
    size = (1500, 900)
    windows = infer.tile_windows(size, 3)
    assert len(windows) == 9
    assert all(0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]
               for x1, y1, x2, y2 in windows)
    # A mark sitting exactly on the inner seam (x = 500) lands WHOLE in the
    # first column's window thanks to the overlap growth.
    x2_first = windows[0][2]
    assert x2_first > 500, 'no seam overlap — a mark cut in two is found in neither half'
    assert infer.tile_windows(size, 1) == [(0, 0, 1500, 900)]


# --- the strict consensus -----------------------------------------------------

def test_consensus_keeps_agreed_spots_with_base_geometry_and_drops_solo_boxes():
    infer = _infer()
    base = [[0.10, 0.10, 0.20, 0.18],    # real mark, full box
            [0.55, 0.60, 0.66, 0.70]]    # rock texture the base prompt invented
    validate = [[0.12, 0.11, 0.19, 0.16],  # agrees on the mark (tighter box)
                [0.80, 0.05, 0.90, 0.12]]  # skin the validator alone named
    kept = infer._strict_consensus(base, validate)
    assert kept == [base[0]], (
        'consensus must keep the BASE geometry of agreed spots and drop both '
        'kinds of solo boxes — symmetric union measured 18 points worse on precision')


def test_effective_regions_routes_the_validator_when_given_one():
    infer = _infer()
    size = (1000, 1000)
    base = [[100, 100, 200, 180], [550, 600, 660, 700]]
    validate = [[120, 110, 190, 160]]
    out = infer.effective_regions(base, size, validate_boxes=validate)
    assert out == [[0.1, 0.1, 0.2, 0.18]]
    # None keeps the legacy single-set behaviour for old callers.
    legacy = infer.effective_regions(base, size)
    assert len(legacy) == 2


# --- the wall-to-wall rule, both edges ----------------------------------------

def test_enough_located_zones_survive_a_wall_to_wall_claim():
    infer = _infer()
    tiles = [[x, y, x + 240, y + 160]
             for x in range(0, 1000, 250) for y in range(0, 1000, 200)]
    assert infer._raw_coverage(tiles, (1000, 1000)) > infer.WALL_TO_WALL_COVERAGE
    out = infer.effective_regions(tiles, (1000, 1000))
    assert len(out) >= infer.WALL_TO_WALL_MIN_ZONES


def test_the_coverage_is_the_union_of_the_boxes_not_the_sum_of_their_areas():
    """The measurement that made a one-mark photo report nothing.

    The sweep sends fourteen windows through two prompts, so ONE mark comes
    back as a dozen boxes over the same spot. Summing their areas claimed the
    whole frame — the figure read 1.00 on eleven of the fifteen bench images,
    including a photo whose boxes really union to 0.27 — and the wall-to-wall
    guard below then reduced to "fewer than three zones located? report
    nothing", so ordinary photos came back from the scan with no box."""
    infer = _infer()
    one_mark_seen_twelve_times = [[100, 100, 300, 300]] * 12
    assert infer._raw_coverage(one_mark_seen_twelve_times,
                               (1000, 1000)) == pytest.approx(0.04)
    # ...and a frame-sized "text overlay" match still claims the frame, which
    # is the signal the guard exists for.
    assert infer._raw_coverage([[0, 0, 900, 800]], (1000, 1000)) > \
        infer.WALL_TO_WALL_COVERAGE
    # Partial overlap counts the shared strip once, not twice.
    assert infer._raw_coverage([[0, 0, 200, 100], [100, 0, 300, 100]],
                               (1000, 1000)) == pytest.approx(0.03)


def test_the_wall_to_wall_threshold_separates_the_measured_populations():
    """0.50, not the 0.40 it was: the two numbers are not comparable, because
    the old coverage saturated. Under the union, the 15-image bench splits into
    frame-sized "text overlay" claims (0.70 and up), the genuinely tiled photo
    (0.99), and photos carrying one or two isolated marks (0.41..0.50). The
    threshold has to sit ABOVE the isolated-mark population — left at 0.40 it
    went on blanking the corner-logo images, which is the regression this pass
    undoes — and BELOW the claims it exists to catch."""
    infer = _infer()
    # The measured shape of a corner-logo photo: ONE oversized "a text overlay"
    # match the per-box cap will drop, plus the two marks actually there. Its
    # union lands at 0.48 — between the two candidate thresholds, which is what
    # makes this a test and not a restatement of the constant.
    junk_plus_two_marks = [[0, 0, 680, 650],
                           [820, 880, 980, 970], [40, 700, 210, 830]]
    cover = infer._raw_coverage(junk_plus_two_marks, (1000, 1000))
    assert 0.40 < cover < 0.50, 'fixture no longer straddles the two thresholds'
    assert len(infer.effective_regions(junk_plus_two_marks, (1000, 1000))) == 2, (
        'the two located marks were blanked again — this is the corner-logo '
        'image coming back from the scan with no box')
    # ...while a real frame-wide claim with too little located still reports [].
    assert infer.effective_regions([[0, 0, 900, 800], [10, 15, 260, 135]],
                                   (1000, 1000)) == []


def test_one_mark_found_over_and_over_still_reports_its_zone():
    """End of the same bug, through the real decision point: a single corner
    logo the sweep found in every window it touched must NOT be blanked as a
    wall-to-wall claim. Measured: 3 stamped corner-logo scenarios, all three
    zeroed before, and the maintainer's own report was "the detection is not
    perfect" on exactly this shape."""
    infer = _infer()
    corner = [[820, 880, 980, 970]] * 14
    assert infer.effective_regions(corner, (1000, 1000)) == [
        [0.82, 0.88, 0.98, 0.97]]


def test_neighbouring_marks_of_a_tiling_stay_separate_zones():
    """The avalanche. Bare overlap merged any two boxes that touched, and the
    merged box GREW, so it reached the next neighbour: on the tiled stock photo
    91 correctly-found boxes collapsed to 10 blobs and the area cap threw 4 of
    those away. Finding more marks reported fewer zones."""
    infer = _infer()
    row = [[x, 400, x + 300, 520] for x in range(0, 900, 250)]   # 4, overlapping
    assert all(infer._overlaps(a, b) for a, b in zip(row, row[1:])), \
        'fixture must overlap, or it is not testing the merge at all'
    out = infer._merge_boxes(infer._normalise_boxes(row, (1000, 1000)))
    assert len(out) == len(row), (
        'a repeated mark fused into one blob again — that blob then dies on '
        'the per-box area cap and the image reports nothing')


def test_the_pieces_of_one_cut_mark_still_come_back_as_one_zone():
    """The other half of the same rule, and why it is not simply "merge less".
    Three phrasings over ONE mark nest inside each other; a tile seam splits a
    logo into an icon and a caption that barely clip. Both must come back as
    one box — the mask editor shows one rectangle per watermark, and ✂ crop
    routes on rows that located a single zone."""
    infer = _infer()
    nested = [[0.10, 0.10, 0.26, 0.22], [0.12, 0.11, 0.24, 0.20]]
    assert len(infer._merge_boxes(nested)) == 1
    icon = [0.10, 0.10, 0.20, 0.16]         # the glyph half
    caption = [0.12, 0.15, 0.26, 0.21]      # the words half, barely clipping
    assert infer._same_mark(icon, caption), 'one logo split in two'
    assert len(infer._merge_boxes([icon, caption])) == 1
    # Two marks whose union would be a big rectangle stay two, even touching.
    far = [[0.02, 0.02, 0.20, 0.14], [0.18, 0.12, 0.60, 0.40]]
    assert not infer._same_mark(*far)


def test_too_few_located_zones_under_a_wall_to_wall_claim_stay_unlocated():
    infer = _infer()
    claim = [[0, 0, 950, 700], [20, 30, 260, 140], [700, 800, 940, 930]]
    assert infer._raw_coverage(claim, (1000, 1000)) > infer.WALL_TO_WALL_COVERAGE
    assert infer.effective_regions(claim, (1000, 1000)) == [], (
        'two pinned tiles of a wall-to-wall mark read as "handled" — the '
        'lie-by-omission the guard exists for')


# --- the geometry rescue ------------------------------------------------------

def test_a_confirmed_zone_grows_to_cover_the_emblem_above_its_caption():
    """The case the rescue exists for, from the maintainer's seven-logo photo:
    the two-line caption is boxed confidently (0.502) and the emblem right
    above it only at 0.349 — under the tile floor — so the reported zone
    stopped at the text, the clean pre-erased only that, and the re-render put
    the emblem back as a ghost glyph. Same geometry, on a 1000x1000 frame."""
    infer = _infer()
    caption = [220, 550, 385, 620]           # confident: this IS the zone
    emblem = [260, 480, 350, 560]            # timid: only says how far it goes
    validate = [222, 553, 383, 590]          # the validator vouches for the spot
    without = infer.effective_regions([caption], (1000, 1000),
                                      validate_boxes=[validate])
    assert without == [[0.22, 0.55, 0.385, 0.62]], 'fixture drifted'
    grown = infer.effective_regions([caption], (1000, 1000),
                                    validate_boxes=[validate],
                                    weak_boxes=[caption, emblem])
    assert grown == [[0.22, 0.48, 0.385, 0.62]], (
        'the zone stops at the caption again — the emblem is outside it, the '
        'clean never erases it and the re-render regenerates it')


def test_the_rescue_never_invents_a_zone_of_its_own():
    """A spot no confident box named is not a mark. The timid boxes complete a
    geometry, they never create one — which is what keeps this away from the
    symmetric union measured at 72% precision against 90%."""
    infer = _infer()
    caption = [220, 550, 385, 620]
    elsewhere = [700, 700, 800, 780]         # timid, and nowhere near a zone
    out = infer.effective_regions([caption], (1000, 1000),
                                  validate_boxes=[caption],
                                  weak_boxes=[caption, elsewhere])
    assert out == [[0.22, 0.55, 0.385, 0.62]]


def test_the_rescue_refuses_to_grow_a_zone_into_the_picture():
    """A box that SWALLOWS the zone is a failed localisation, not the rest of
    the mark. _same_mark says yes to it (the zone nests inside), so the size
    cap is what stops it: past ONE_MARK_MAX_AREA a zone is no longer one mark,
    and the crop or inpaint it feeds would act on the photo."""
    infer = _infer()
    caption = [220, 550, 385, 620]
    swallows = [180, 500, 480, 720]          # 0.066 of the frame
    share = ((swallows[2] - swallows[0]) * (swallows[3] - swallows[1])) / 1e6
    assert infer.MAX_REGION_AREA > share > infer.ONE_MARK_MAX_AREA, (
        'fixture must survive the per-box cap and fail the one-mark cap')
    out = infer.effective_regions([caption], (1000, 1000),
                                  validate_boxes=[caption],
                                  weak_boxes=[caption, swallows])
    assert out == [[0.22, 0.55, 0.385, 0.62]]


def test_two_pieces_of_one_mark_that_grow_into_each_other_come_back_as_one():
    """Growing happens BEFORE the last merge on purpose: an icon half and a
    caption half the sweep found separately get bridged by a timid box, and
    the mask editor must then show ONE rectangle, not two overlapping ones."""
    infer = _infer()
    left = [100, 100, 200, 160]
    right = [300, 100, 400, 160]
    bridge = [180, 100, 320, 160]            # timid, joins the two halves
    apart = infer.effective_regions([left, right], (1000, 1000),
                                    validate_boxes=[left, right])
    assert len(apart) == 2, 'fixture must start as two zones'
    joined = infer.effective_regions([left, right], (1000, 1000),
                                     validate_boxes=[left, right],
                                     weak_boxes=[left, right, bridge])
    assert joined == [[0.1, 0.1, 0.4, 0.16]]


def test_the_rescue_floor_is_the_full_frame_floor_not_a_new_number():
    """A box confident enough to be reported on its own at full frame is
    confident enough to say how far a CONFIRMED mark extends. Swept both ways
    on the ground truth: at the tile floor the rescue is off (completeness
    0.71), at 0.20 and below the zones start growing over the picture instead
    of over marks (spill 0.416 -> 0.429 -> 0.451 while completeness flattens
    at 0.81)."""
    infer = _infer()
    assert infer.GEOMETRY_RESCUE_THRESHOLD == infer.LOCATE_BOX_THRESHOLD
    assert infer.GEOMETRY_RESCUE_THRESHOLD < infer.LOCATE_TILE_THRESHOLD, (
        'at or above the tile floor the rescue has nothing left to add')


# --- the wall-to-wall claim, once the sweep could not tile --------------------

def test_an_untiled_sweep_judges_the_claim_on_the_biggest_single_box():
    """The 474px stock thumbnails, measured. Their sweep is ONE forward, where
    DINO answers "the whole picture is a logo" two or three times; the union of
    those claims reads 0.59 and blanked the word the same pass had just
    located, at 0.65 confidence, dead centre. That union is not evidence of a
    tiling here — it is one failed localisation counted several times."""
    infer = _infer()
    size = (474, 270)
    assert infer.tile_plan(size) == [(1, infer.LOCATE_BOX_THRESHOLD)], (
        'fixture must be an image the sweep cannot tile')
    junk = [57, 25, 412, 229]                # 0.567 of the frame
    word = [104, 104, 370, 150]              # "shutterstock", across the middle
    footer = [144, 254, 330, 269]            # the site line under it
    raw = [junk, word, footer]
    assert infer._raw_coverage(raw, size) > infer.WALL_TO_WALL_COVERAGE, (
        'fixture no longer trips the union rule, so it tests nothing')
    assert infer._biggest_box_share(raw, size) < infer.WHOLE_FRAME_CLAIM
    out = infer.effective_regions(raw, size, validate_boxes=raw)
    assert len(out) == 2, (
        'the two located marks were blanked again — this is the stock '
        'thumbnail coming back "flagged, no position"')


def test_an_untiled_whole_frame_claim_still_reports_nothing():
    """The other edge, and why this is not simply "no guard on small images":
    a picture whose watermark really IS the whole picture produces one box over
    the whole frame (measured 0.983 on a 474px thumbnail of the tiled stock
    photo, against 0.86 and below on every other image of the bench). Two of
    its ten tiles located must not read as "handled"."""
    infer = _infer()
    size = (474, 316)
    everything = [0, 3, 474, 313]            # 0.98 of the frame
    tile_a = [146, 97, 290, 145]
    tile_b = [159, 142, 303, 191]
    raw = [everything, tile_a, tile_b]
    assert infer._biggest_box_share(raw, size) >= infer.WHOLE_FRAME_CLAIM
    assert infer.effective_regions(raw, size, validate_boxes=raw) == []


def test_a_tiled_sweep_keeps_judging_the_claim_on_the_union():
    """Nothing moves where the sweep CAN tile — the whole ground-truth bench
    lives there (46/57 recall, 92% precision, both unchanged). The same shape
    that now reports on a thumbnail still reports nothing at a tileable size,
    because there the union is made of what the tiles really found."""
    infer = _infer()
    size = (1000, 1000)
    assert len(infer.tile_plan(size)) > 1
    raw = [[0, 0, 900, 800], [10, 15, 260, 135]]
    assert infer._biggest_box_share(raw, size) < infer.WHOLE_FRAME_CLAIM
    assert infer.effective_regions(raw, size) == []


def test_the_biggest_box_share_is_bounded_and_junk_proof():
    infer = _infer()
    assert infer._biggest_box_share([], (1000, 1000)) == 0.0
    assert infer._biggest_box_share([[0, 0, 10, 10]], (0, 0)) == 0.0
    assert infer._biggest_box_share([['x', 1, 2, 3], None,
                                     [0, 0, 500, 500]], (1000, 1000)) == 0.25
    # Clamped like the coverage: a box DINO returned slightly outside the frame
    # claims the frame, not more.
    assert infer._biggest_box_share([[-20, -10, 1100, 1050]],
                                    (1000, 1000)) == 1.0
