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
    # 'rtx 6000 ada' (1.9) must win over a bare 'rtx' — and not be read as a
    # slow Ampere 'rtx 6000' had we tabulated one.
    assert gs.speed_factor('RTX 6000 Ada Generation') == 1.9


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
