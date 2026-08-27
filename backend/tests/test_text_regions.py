"""text_regions — the geometry that turns OCR lines into repaint zones.

Pure-module tests: no app, no engine. The one cross-module concern is the
LAST class here, which pins this module's cap/floor to the hand-mask contract
it feeds — the two are deliberately duplicated (importing the dataset service
from text_regions would be a cycle), so a test is what keeps them one value.
"""
import math

from app.services.text_regions import (
    TEXT_MERGE_PAD, TEXT_REGION_CAP, TEXT_REGION_MIN_SIDE, text_mask_regions,
)


class TestMerging:
    def test_lines_of_one_bubble_become_one_zone(self):
        # Three lines, vertical gaps under 2*pad: one balloon, one zone.
        lines = [
            [0.30, 0.100, 0.70, 0.140],
            [0.28, 0.160, 0.72, 0.200],
            [0.35, 0.220, 0.65, 0.260],
        ]
        regions, dropped = text_mask_regions(lines)
        assert dropped == 0
        assert len(regions) == 1
        x0, y0, x1, y1 = regions[0]
        # The union covers every line, plus the pad on the outer edge.
        assert x0 <= 0.28 - TEXT_MERGE_PAD + 1e-9
        assert y0 <= 0.100 - TEXT_MERGE_PAD + 1e-9
        assert x1 >= 0.72 + TEXT_MERGE_PAD - 1e-9
        assert y1 >= 0.260 + TEXT_MERGE_PAD - 1e-9

    def test_two_far_apart_blocks_stay_two_zones(self):
        regions, _ = text_mask_regions([
            [0.10, 0.10, 0.30, 0.14],      # a bubble top-left
            [0.60, 0.80, 0.90, 0.86],      # a sound effect bottom-right
        ])
        assert len(regions) == 2

    def test_union_is_transitive_not_one_sweep(self):
        # A chain: 1 overlaps 2 only once padded, 1∪2 overlaps 3. A single
        # sweep in the wrong order leaves two zones; the fixed point leaves one.
        step = 2 * TEXT_MERGE_PAD * 0.9
        lines = [[0.4, 0.10 + i * (0.03 + step), 0.6, 0.13 + i * (0.03 + step)]
                 for i in range(5)]
        regions, _ = text_mask_regions(lines)
        assert len(regions) == 1

    def test_score_fifth_element_is_tolerated(self):
        regions, _ = text_mask_regions([[0.4, 0.4, 0.6, 0.45, 0.93]])
        assert len(regions) == 1
        assert all(len(r) == 4 for r in regions)

    def test_garbage_boxes_are_dropped_not_repaired(self):
        regions, dropped = text_mask_regions([
            [0.6, 0.4, 0.4, 0.45],          # inverted
            [0.1, 0.1, 0.1, 0.2],           # zero width
            [float('nan'), 0, 1, 1],        # NaN
            ['a', 0, 1, 1],                 # not numbers
            [0.2],                          # short row
        ])
        assert regions == []
        assert dropped == 0

    def test_output_is_clamped_rounded_and_area_sorted(self):
        regions, _ = text_mask_regions([
            [0.001, 0.001, 0.20, 0.05],     # pad would push past 0
            [0.30, 0.30, 0.90, 0.70],       # the big one
        ])
        assert len(regions) == 2
        big, small = regions
        assert (big[2] - big[0]) * (big[3] - big[1]) >= \
               (small[2] - small[0]) * (small[3] - small[1])
        for box in regions:
            assert all(0.0 <= v <= 1.0 for v in box)
            assert all(round(v, 4) == v for v in box)


class TestExistingRegions:
    def test_existing_zones_survive_the_merge(self):
        existing = [[0.05, 0.05, 0.15, 0.15]]
        regions, _ = text_mask_regions([[0.6, 0.6, 0.8, 0.65]], existing)
        assert len(regions) == 2
        assert [0.05, 0.05, 0.15, 0.15] in regions

    def test_existing_zones_are_not_padded(self):
        regions, _ = text_mask_regions([], [[0.2, 0.2, 0.4, 0.4]])
        assert regions == [[0.2, 0.2, 0.4, 0.4]]

    def test_overlapping_existing_zone_folds_into_the_text_zone(self):
        regions, _ = text_mask_regions(
            [[0.30, 0.30, 0.50, 0.35]], [[0.45, 0.28, 0.60, 0.40]])
        assert len(regions) == 1
        x0, y0, x1, y1 = regions[0]
        assert x1 >= 0.60 - 1e-9 and y0 <= 0.28 + 1e-9

    def test_no_text_but_existing_zones_returns_them(self):
        # A re-scan of a page whose text was hand-masked must not erase the mask.
        regions, dropped = text_mask_regions([], [[0.1, 0.1, 0.3, 0.3]])
        assert regions == [[0.1, 0.1, 0.3, 0.3]]
        assert dropped == 0


class TestCapAndFloor:
    def test_cap_keeps_the_biggest_and_counts_the_rest(self):
        # 40 disjoint zones of decreasing size: 32 kept, 8 reported dropped.
        lines = []
        for i in range(40):
            x = (i % 8) * 0.125 + 0.01
            y = (i // 8) * 0.2 + 0.01
            side = 0.08 - i * 0.001
            lines.append([x, y, x + side, y + side])
        regions, dropped = text_mask_regions(lines, pad=0.0)
        assert len(regions) == TEXT_REGION_CAP
        assert dropped == 8
        smallest_kept = min((r[2] - r[0]) for r in regions)
        assert smallest_kept >= 0.08 - 31 * 0.001 - 1e-9

    def test_specks_below_the_floor_are_dropped(self):
        regions, _ = text_mask_regions([[0.5, 0.5, 0.5015, 0.5015]], pad=0.0)
        assert regions == []

    def test_padded_speck_survives_because_pad_is_the_margin(self):
        # The default pad alone lifts a tiny OCR hit over the floor — wanted:
        # a one-glyph sound effect is small, its repaint zone is not.
        regions, _ = text_mask_regions([[0.5, 0.5, 0.503, 0.503]])
        assert len(regions) == 1


class TestContractWithTheHandMaskChannel:
    """The values these zones must satisfy are owned by the dataset service —
    duplicated here by design (import would cycle), pinned so they cannot
    drift. If one of these fails, someone moved a limit on one side only."""

    def test_cap_matches_normalize_watermark_regions(self):
        from app.services.face_dataset_service import WATERMARK_REGION_LIMIT
        assert TEXT_REGION_CAP == WATERMARK_REGION_LIMIT

    def test_floor_matches_normalize_watermark_regions(self):
        from app.services.face_dataset_service import WATERMARK_REGION_MIN_SIDE
        assert TEXT_REGION_MIN_SIDE == WATERMARK_REGION_MIN_SIDE

    def test_output_passes_the_channel_validator(self):
        from app.services.face_dataset_service import normalize_watermark_regions
        regions, _ = text_mask_regions([
            [0.30, 0.10, 0.70, 0.14],
            [0.28, 0.16, 0.72, 0.20],
            [0.60, 0.80, 0.90, 0.86],
        ], [[0.05, 0.05, 0.15, 0.15]])
        # Must not raise, must round-trip identically.
        assert normalize_watermark_regions(regions, allow_null=False) == regions

    def test_pad_is_finite_and_sane(self):
        assert math.isfinite(TEXT_MERGE_PAD) and 0 < TEXT_MERGE_PAD < 0.05
