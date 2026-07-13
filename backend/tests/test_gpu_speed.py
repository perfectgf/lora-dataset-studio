"""GPU speed heuristic feeding the launch-time time/cost estimates. Pure
functions, no app/network — just the ordering and scaling guarantees."""
from app.services import gpu_speed as gs


def test_faster_cards_have_higher_factor():
    assert gs.speed_factor('RTX 4090') > gs.speed_factor('RTX 3090')
    assert gs.speed_factor('RTX 5090') > gs.speed_factor('RTX 4090')
    assert gs.speed_factor('H100 SXM') > gs.speed_factor('RTX 5090')


def test_unknown_gpu_falls_back_to_baseline():
    assert gs.speed_factor('Totally New GPU 9000') == 1.0
    assert gs.speed_factor('') == 1.0
    assert gs.speed_factor(None) == 1.0


def test_longest_substring_match_wins():
    # 'rtx6000ada' (1.9) must win over a bare 'rtx' — and not be read as a
    # slow Ampere 'rtx 6000' had we tabulated one.
    assert gs.speed_factor('RTX 6000 Ada Generation') == 1.9
    # 4080 SUPER must take the measured 4080S factor, not plain 4080
    assert gs.speed_factor('RTX 4080S') == 1.56
    assert gs.speed_factor('RTX 4080') == 1.35


def test_matching_ignores_spacing():
    """vast writes both 'RTX 6000Ada' and 'RTX 6000 Ada' (observed live in the
    tier dialog 2026-07-13 — the spaceless form fell back to 3090 speed)."""
    assert gs.speed_factor('RTX 6000Ada') == 1.9
    assert gs.speed_factor('RTX 6000 Ada') == 1.9


def test_blackwell_pro_cards_are_tabulated():
    """'RTX PRO 5000/6000' rows showed 3090-class times in the launch dialog
    (fallback 1.0) with inflated total costs — they must be tabulated."""
    assert gs.speed_factor('RTX PRO 5000') == 1.9
    assert gs.speed_factor('RTX PRO 6000 WS') == 3.0
    assert gs.speed_factor('RTX PRO 6000 S') == 3.0
    assert gs.speed_factor('RTX PRO 6000 WS') > gs.speed_factor('RTX 5090')


def test_baselines_match_live_measurements():
    """Pinned to the values MEASURED on real pods (2026-07-13): zimage
    4.49 s/it and krea 8.84 s/it on RTX 3090 — the original guessed baselines
    (0.9/1.1 s) understated durations ~5-8x ('les calculs ne sont pas bons')."""
    assert gs.estimate_minutes('RTX 3090', 'zimage', 1000) == 1000 * 4.5 / 60
    assert gs.estimate_minutes('RTX 3090', 'krea', 1000) == 1000 * 8.8 / 60


def test_estimate_scales_with_steps_and_inversely_with_speed():
    slow = gs.estimate_minutes('RTX 3090', 'krea', 3000)
    fast = gs.estimate_minutes('RTX 5090', 'krea', 3000)
    assert slow > fast > 0
    # linear in steps
    assert gs.estimate_minutes('RTX 3090', 'krea', 6000) == \
        2 * gs.estimate_minutes('RTX 3090', 'krea', 3000)


def test_krea_slower_per_step_than_zimage():
    assert gs.estimate_minutes('RTX 3090', 'krea', 1000) > \
        gs.estimate_minutes('RTX 3090', 'zimage', 1000)


def test_zero_steps_is_zero_minutes():
    assert gs.estimate_minutes('RTX 4090', 'zimage', 0) == 0.0
