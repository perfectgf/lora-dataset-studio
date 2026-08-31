"""The multi-scale consensus locate — every rule of the recipe, held pure.

The measured story (ground-truth bench: 12 stamped scenarios over 3 real
photos, plus the maintainer's two test images): the single full-frame DINO pass
finds 4 of 7 logo copies and 19% of stamped marks overall. Sweeping tiles at a
raised floor and keeping only boxes BOTH prompt phrasings agree on lifts recall
to 77% at 90% precision — the dropped boxes are exactly the rock-texture and
icicle shapes a single phrasing hallucinates inside tiles.

torch never loads here: the recipe's decisions (tile plan, window mapping,
consensus, wall-to-wall rule) are pure functions, which is what makes them
testable at all — the GPU sweep only feeds them boxes.
"""
import importlib.util
import os


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
    # 640/2 = 320 earns the 2x2; 640/3 = 213 is below the tile floor.
    assert infer.tile_plan((640, 900)) == [
        (1, infer.LOCATE_BOX_THRESHOLD), (2, infer.LOCATE_TILE_THRESHOLD)]


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
    tiles = [[x, y, x + 260, y + 120]
             for x in range(0, 1000, 300) for y in range(0, 1000, 250)]
    assert infer._raw_coverage(tiles, (1000, 1000)) > infer.WALL_TO_WALL_COVERAGE
    out = infer.effective_regions(tiles, (1000, 1000))
    assert len(out) >= infer.WALL_TO_WALL_MIN_ZONES


def test_too_few_located_zones_under_a_wall_to_wall_claim_stay_unlocated():
    infer = _infer()
    claim = [[0, 0, 950, 700], [20, 30, 260, 140], [700, 800, 940, 930]]
    assert infer._raw_coverage(claim, (1000, 1000)) > infer.WALL_TO_WALL_COVERAGE
    assert infer.effective_regions(claim, (1000, 1000)) == [], (
        'two pinned tiles of a wall-to-wall mark read as "handled" — the '
        'lie-by-omission the guard exists for')
