"""The pure shot-boundary rules — the module every re-threshold goes through.

Nothing here imports torch, numpy or av, and that is the point: the whole
reason the detector's per-frame probabilities are persisted is so that changing
a threshold costs no GPU and no decode. That promise is only real if the code
that turns probabilities into clips can run — and be tested — with none of the
video extras installed.

The vectors below are synthetic on purpose. A test that needed a real model
would need a GPU, and a rule that only holds on one corpus is not a rule.
"""
import pytest

from app.services import shot_boundaries as sb


# --- the transition rule (a faithful port, pinned against the child's) ---------

def test_predictions_to_scenes_finds_one_boundary():
    probs = [0.1] * 5 + [0.9] + [0.1] * 4
    assert sb.predictions_to_scenes(probs, 0.5) == [[0, 5], [6, 9]]


def test_predictions_to_scenes_threshold_is_strictly_greater_than():
    assert sb.predictions_to_scenes([0.1, 0.5, 0.1], 0.5) == [[0, 2]]


def test_predictions_to_scenes_of_a_single_shot_video():
    assert sb.predictions_to_scenes([0.1] * 20, 0.5) == [[0, 19]]


def test_predictions_to_scenes_of_nothing_is_empty():
    assert sb.predictions_to_scenes([], 0.5) == []


def test_raising_the_threshold_removes_boundaries_from_the_same_vector():
    """The whole point of persisting probabilities: one vector, many answers,
    no second decode."""
    probs = [0.0] * 4 + [0.6] + [0.0] * 4 + [0.95] + [0.0] * 4
    assert len(sb.predictions_to_scenes(probs, 0.5)) == 3
    assert len(sb.predictions_to_scenes(probs, 0.8)) == 2


def test_the_pure_port_agrees_with_the_worker_frame_for_frame():
    """Two copies of one rule in two interpreters is exactly how a re-threshold
    starts disagreeing with the detection that produced the cache. The vectors
    are shared with the worker's own test through this comparison, not through
    a hand-copied expectation."""
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'infer', 'shot_detect_infer.py')
    spec = importlib.util.spec_from_file_location('shot_detect_infer', path)
    infer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer)
    for probs in ([], [0.1] * 6, [0.1, 0.9, 0.1, 0.1, 0.9, 0.2],
                  [0.9] * 3, [0.0] * 4 + [0.6] + [0.0] * 4):
        for thr in (0.1, 0.5, 0.9):
            assert (sb.predictions_to_scenes(probs, thr)
                    == infer._predictions_to_scenes(probs, thr)), (probs, thr)


# --- which threshold wins ------------------------------------------------------

def test_the_file_threshold_beats_the_bank_which_beats_the_global():
    assert sb.resolve_threshold(0.7, 0.6, 0.5) == 0.7
    assert sb.resolve_threshold(None, 0.6, 0.5) == 0.6
    assert sb.resolve_threshold(None, None, 0.5) == 0.5


def test_an_out_of_range_threshold_is_clamped_never_refused():
    """Read on a hot path, mid-pass: a hand-edited config or a stale row must
    degrade to something usable rather than abort work already running."""
    assert sb.resolve_threshold(9, None, 0.5) == 1.0
    assert sb.resolve_threshold(-3, None, 0.5) == 0.0
    assert sb.resolve_threshold('nonsense', None, 0.5) == 0.5


# --- the minimum shot length, in SECONDS -------------------------------------

def test_min_shot_frames_converts_seconds_through_this_files_own_fps():
    """0.6 s is 15 frames at 25 fps and 36 at 60 — the whole reason the floor
    stopped being expressed in frames."""
    assert sb.min_shot_frames_for(25.0, min_seconds=0.6) == 15
    assert sb.min_shot_frames_for(60.0, min_seconds=0.6) == 36


def test_the_legacy_frame_floor_still_wins_when_it_is_the_only_one_set():
    """`shot_detect.min_shot_frames` sits in user config files today. It is
    never renamed away, and a user who set it and not the new key keeps the
    behaviour they configured."""
    assert sb.min_shot_frames_for(60.0, min_seconds=None, min_frames=5) == 5


def test_the_seconds_floor_wins_when_both_are_set():
    assert sb.min_shot_frames_for(25.0, min_seconds=0.6, min_frames=5) == 15


def test_min_shot_frames_never_goes_below_one():
    assert sb.min_shot_frames_for(25.0, min_seconds=0.0) == 1
    assert sb.min_shot_frames_for(0, min_seconds=0.6) == 1


# --- drop vs merge -------------------------------------------------------------

def test_drop_removes_the_sliver_and_renumbers_nothing():
    scenes = [[0, 40], [41, 42], [43, 90]]
    assert sb.apply_min_length(scenes, 5, 'drop') == [[0, 40], [43, 90]]


def test_merge_glues_the_sliver_onto_the_previous_shot():
    scenes = [[0, 40], [41, 42], [43, 90]]
    assert sb.apply_min_length(scenes, 5, 'merge') == [[0, 42], [43, 90]]


def test_merge_keeps_every_boundary_it_did_not_delete():
    """Merging removes ONE boundary — the one that produced the sliver — and
    moves no other. The surviving cuts are a strict subset of what the detector
    drew, which is what makes the gesture safe to offer by default one day."""
    scenes = [[0, 40], [41, 42], [43, 90], [91, 92], [93, 120]]
    merged = sb.apply_min_length(scenes, 5, 'merge')
    kept_starts = {s[0] for s in merged}
    assert kept_starts <= {s[0] for s in scenes}
    assert merged == [[0, 42], [43, 92], [93, 120]]


def test_merge_sends_a_leading_sliver_forward_because_there_is_no_previous():
    scenes = [[0, 2], [3, 60]]
    assert sb.apply_min_length(scenes, 5, 'merge') == [[0, 60]]


def test_merge_keeps_a_file_that_is_shorter_than_the_floor_in_one_piece():
    """Nothing to glue it to. Dropping it would leave the source with zero
    clips, which is a worse answer than one short one."""
    assert sb.apply_min_length([[0, 2]], 5, 'merge') == [[0, 2]]


def test_drop_of_everything_returns_nothing_and_says_so_by_being_empty():
    assert sb.apply_min_length([[0, 2]], 5, 'drop') == []


def test_an_unknown_policy_falls_back_to_drop():
    assert sb.apply_min_length([[0, 40], [41, 42]], 5, 'wat') == [[0, 40]]


# --- the second head: how WIDE was the transition ------------------------------

def test_a_hard_cut_is_a_narrow_plateau():
    all_probs = [0.0] * 10 + [0.9] + [0.0] * 10
    assert sb.transition_width(all_probs, 10) == 1


def test_a_dissolve_is_a_wide_plateau():
    all_probs = [0.0] * 10 + [0.6] * 18 + [0.0] * 10
    assert sb.transition_width(all_probs, 18) == 18


def test_the_width_is_zero_when_the_second_head_says_nothing_there():
    assert sb.transition_width([0.0] * 20, 10) == 0


def test_the_width_is_zero_when_there_is_no_second_head_at_all():
    """A cache written before the all-frames head was persisted is a legitimate
    state: no label, rather than a made-up one."""
    assert sb.transition_width(None, 10) == 0


def test_narrow_reads_as_a_cut_and_wide_as_a_dissolve():
    assert sb.label_transition(1) == 'cut'
    assert sb.label_transition(4) == 'cut'
    assert sb.label_transition(18) == 'dissolve'


def test_the_cut_dissolve_frontier_is_a_parameter_a_calibration_can_move():
    """The width->type rule is coherent with the published architecture and
    has never been measured on this corpus. It must be tunable without a code
    change the day someone measures it."""
    assert sb.label_transition(6, dissolve_min_frames=20) == 'cut'
    assert sb.label_transition(6, dissolve_min_frames=3) == 'dissolve'


# --- building clips ------------------------------------------------------------

def _single(peaks, n=120):
    probs = [0.0] * n
    for i in peaks:
        probs[i] = 0.95
    return probs


def test_build_clips_converts_frame_indices_to_pts_seconds():
    clips = sb.build_clips(_single([60]), fps=24.0, min_frames=1)
    assert clips[0]['start_s'] == 0.0
    assert clips[0]['end_s'] == pytest.approx(61 / 24)
    assert clips[1]['start_frame'] == 61


def test_build_clips_stamps_the_detector_id_on_every_row():
    clips = sb.build_clips(_single([60]), fps=24.0, min_frames=1)
    assert {c['detector'] for c in clips} == {'transnetv2'}


def test_build_clips_labels_the_boundary_at_each_end_of_a_shot():
    single = _single([60])
    all_probs = [0.0] * 120
    for i in range(52, 70):
        all_probs[i] = 0.7
    clips = sb.build_clips(single, all_probs, fps=24.0, min_frames=1)
    assert clips[0]['transition']['start'] is None          # the file's own start
    assert clips[0]['transition']['end'] == {'kind': 'dissolve', 'width': 18}
    assert clips[1]['transition']['start'] == {'kind': 'dissolve', 'width': 18}
    assert clips[1]['transition']['end'] is None            # the file's own end


def test_the_first_and_last_edges_of_a_file_are_never_transitions():
    """A file starts and ends; nothing dissolved into it. Labelling those as
    cuts would put a chip on every single clip and make the chip meaningless."""
    clips = sb.build_clips([0.0] * 40, [0.9] * 40, fps=24.0, min_frames=1)
    assert clips[0]['transition'] == {'start': None, 'end': None}


def test_trimming_a_dissolve_pulls_both_bounds_in_by_half_its_width():
    single = _single([60])
    all_probs = [0.0] * 120
    for i in range(52, 70):
        all_probs[i] = 0.7
    plain = sb.build_clips(single, all_probs, fps=24.0, min_frames=1)
    trimmed = sb.build_clips(single, all_probs, fps=24.0, min_frames=1,
                             trim_dissolves=True)
    assert trimmed[0]['end_frame'] == plain[0]['end_frame'] - 9
    assert trimmed[1]['start_frame'] == plain[1]['start_frame'] + 9


def test_trimming_leaves_hard_cuts_exactly_where_they_are():
    single = _single([60])
    all_probs = [0.0] * 120
    all_probs[60] = 0.9
    plain = sb.build_clips(single, all_probs, fps=24.0, min_frames=1)
    trimmed = sb.build_clips(single, all_probs, fps=24.0, min_frames=1,
                             trim_dissolves=True)
    assert [c['start_frame'] for c in trimmed] == [c['start_frame'] for c in plain]


def test_trimming_never_eats_a_shot_whole():
    """Two long dissolves around a short shot would trim it out of existence.
    A clip that survives the floor must survive the trim."""
    single = _single([40, 52])
    all_probs = [0.0] * 120
    for i in range(30, 62):
        all_probs[i] = 0.7
    trimmed = sb.build_clips(single, all_probs, fps=24.0, min_frames=5,
                             trim_dissolves=True)
    assert all(c['end_frame'] > c['start_frame'] for c in trimmed)


def test_build_clips_of_an_empty_vector_is_no_clips_not_a_crash():
    assert sb.build_clips([], fps=24.0, min_frames=1) == []


# --- the dry run ---------------------------------------------------------------

def test_the_sweep_answers_how_many_shots_each_threshold_would_give():
    probs = [0.0] * 4 + [0.6] + [0.0] * 4 + [0.95] + [0.0] * 4
    rows = sb.sweep(probs, [0.5, 0.8], fps=24.0, min_frames=1)
    assert rows == [{'threshold': 0.5, 'shots': 3}, {'threshold': 0.8, 'shots': 2}]


def test_the_sweep_counts_what_the_floor_would_actually_keep():
    """A preview that counted boundaries rather than surviving clips would
    promise shots the pass then drops — the exact complaint the metrics dry-run
    exists to avoid."""
    probs = [0.0] * 4 + [0.9] + [0.9] + [0.0] * 30
    rows = sb.sweep(probs, [0.5], fps=24.0, min_frames=20)
    assert rows == [{'threshold': 0.5, 'shots': 1}]
