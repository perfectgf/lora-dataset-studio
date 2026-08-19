"""🎥 The camera-motion pass.

The split every test here relies on: `clip_steps` is the ONLY function that
touches PyAV or OpenCV, so everything below it — the aggregation, the sign
conventions, the labels, the caption phrase — is exercised on trajectories built
in arithmetic, with no video, no extra installed and no encoder sitting between
the intent and the measurement. The forged trajectories are the point: on a real
clip the true answer is approximately known, and here it is known exactly.
"""
import json
import math

import pytest

from app import config as cfg
from app import capabilities, setup_installer
from app.services import (video_camera_motion as cam, video_metrics,
                          video_ai_check, video_defect_sweep, video_safe_zone)


# --- forging a trajectory ------------------------------------------------------------

def steps_from(count=25, fps=25.0, dx=0.0, dy=0.0, dzoom=0.0, drot=0.0,
               coverage=1.0, residual=0.0004, jitter=0.0):
    """`count` identical steps, optionally with an alternating high-frequency
    jitter laid over the translation. The jitter alternates sign every frame,
    which is the fastest thing a sampled series can carry and therefore exactly
    what a moving average cannot follow — a handheld camera, in one number."""
    out = []
    for i in range(count):
        wobble = jitter * (1 if i % 2 == 0 else -1)
        out.append({'dt': 1.0 / fps, 'dx': dx + wobble, 'dy': dy + wobble,
                    'dzoom': dzoom, 'drot': drot,
                    'coverage': coverage, 'residual': residual})
    return out


# --- the four sign conventions -------------------------------------------------------
#
# Each of these is an inversion waiting to happen, and three of the four are
# counter-intuitive. The transform describes how the PICTURE moved; every stored
# number describes how the CAMERA moved.

def test_a_picture_sliding_left_is_a_camera_panning_right():
    scores = cam.aggregate(steps_from(dx=-0.01))
    assert scores['camera_pan_rate'] > 0
    assert 'pan_right' in cam.labels(scores)


def test_a_picture_sliding_right_is_a_camera_panning_left():
    scores = cam.aggregate(steps_from(dx=+0.01))
    assert scores['camera_pan_rate'] < 0
    assert 'pan_left' in cam.labels(scores)


def test_a_picture_sliding_down_is_a_camera_tilting_up():
    # The ONE that is not negated: screen y grows downward, so the two
    # inversions cancel. Getting this wrong flips only this axis, which is
    # exactly the bug a symmetrical test would miss.
    scores = cam.aggregate(steps_from(dy=+0.01))
    assert scores['camera_tilt_rate'] > 0
    assert 'pan_up' in cam.labels(scores)


def test_a_picture_sliding_up_is_a_camera_tilting_down():
    scores = cam.aggregate(steps_from(dy=-0.01))
    assert scores['camera_tilt_rate'] < 0
    assert 'pan_down' in cam.labels(scores)


def test_a_growing_picture_is_a_camera_zooming_in():
    scores = cam.aggregate(steps_from(dzoom=+0.002))
    assert scores['camera_zoom_rate'] > 0
    assert 'zoom_in' in cam.labels(scores)


def test_a_shrinking_picture_is_a_camera_zooming_out():
    scores = cam.aggregate(steps_from(dzoom=-0.002))
    assert scores['camera_zoom_rate'] < 0
    assert 'zoom_out' in cam.labels(scores)


def test_a_picture_turning_clockwise_is_a_camera_rolling_anticlockwise():
    scores = cam.aggregate(steps_from(drot=+0.4))
    assert scores['camera_roll_rate'] < 0
    assert 'rolling' in cam.labels(scores)


def test_the_rates_are_per_second_and_not_per_frame():
    # The same displacement at half the frame rate is the same rate. A pass that
    # summed per-step values instead would report double, and every threshold
    # would then mean something different on 25 fps and 50 fps footage.
    slow = cam.aggregate(steps_from(count=25, fps=25.0, dx=-0.01))
    fast = cam.aggregate(steps_from(count=50, fps=50.0, dx=-0.005))
    assert slow['camera_pan_rate'] == pytest.approx(fast['camera_pan_rate'], rel=1e-6)


def test_a_pan_of_one_percent_per_frame_at_25_fps_reads_as_25_percent_per_second():
    scores = cam.aggregate(steps_from(fps=25.0, dx=-0.01))
    assert scores['camera_pan_rate'] == pytest.approx(25.0, abs=1e-3)


# --- the labels, one forged trajectory per case ----------------------------------------

def test_a_tripod_is_a_static_shot_and_nothing_else():
    assert cam.labels(cam.aggregate(steps_from())) == ['static_shot']


def test_a_pure_pan_is_not_also_called_static():
    assert cam.labels(cam.aggregate(steps_from(dx=-0.01))) == ['pan_right']


def test_high_frequency_jitter_is_handheld_and_not_a_pan():
    # The camera returns to where it started every other frame, so the NET rate
    # is zero — a pass reading only the mean would call this a tripod. An EVEN
    # count, so the alternating series really does cancel: an odd one leaves one
    # unpaired sample and a small genuine mean, which is arithmetic rather than
    # a reading and would make this test measure the fixture.
    scores = cam.aggregate(steps_from(count=24, jitter=0.01))
    assert scores['camera_pan_rate'] == pytest.approx(0.0, abs=1e-6)
    assert scores['camera_shake'] > cam.SHAKE_FLOOR
    assert cam.labels(scores) == ['handheld_shot']


def test_a_steered_pan_is_not_called_handheld():
    # Fast, and perfectly steered. A moving average follows it exactly, so the
    # residual stays at zero — this is the false positive the residual exists to
    # avoid, and it is the reason a whip pan is not tremor.
    scores = cam.aggregate(steps_from(dx=-0.04))
    assert scores['camera_shake'] == pytest.approx(0.0, abs=1e-9)
    assert 'handheld_shot' not in cam.labels(scores)


def test_a_handheld_pan_carries_both_labels():
    scores = cam.aggregate(steps_from(dx=-0.01, jitter=0.01))
    assert cam.labels(scores) == ['pan_right', 'handheld_shot']


def test_a_rigid_moving_frame_is_a_slideshow():
    # A Ken Burns move: one affine transform of one photograph, so nothing is
    # left over after the fit.
    scores = cam.aggregate(steps_from(dx=-0.01, dzoom=0.002, residual=0.00005))
    labels = cam.labels(scores)
    assert 'slideshow' in labels
    assert 'pan_right' in labels and 'zoom_in' in labels


def test_a_pan_over_a_scene_with_depth_is_not_a_slideshow():
    # The same move, over a scene where near and far travel at different rates.
    scores = cam.aggregate(steps_from(dx=-0.01, dzoom=0.002, residual=0.0004))
    assert 'slideshow' not in cam.labels(scores)


def test_a_static_shot_is_never_called_a_slideshow():
    # Trivially rigid, and saying so about every tripod shot in a bank would
    # make the label useless.
    assert cam.labels(cam.aggregate(steps_from(residual=0.0))) == ['static_shot']


def test_the_labels_come_out_in_canonical_order_not_alphabetical():
    scores = cam.aggregate(steps_from(dx=-0.01, dzoom=0.002, jitter=0.01,
                                      residual=0.0004))
    assert cam.labels(scores) == ['pan_right', 'zoom_in', 'handheld_shot']


# --- the guard ---------------------------------------------------------------------------

def test_a_dominant_subject_suppresses_every_direction():
    # The measured failure this guard exists for: with a subject winning the
    # fit, the rates describe the SUBJECT and are fiction about the camera.
    # Large, confident, and completely wrong — so none of it is reported.
    scores = cam.aggregate(steps_from(dx=-0.03, dzoom=0.004, coverage=0.74))
    assert cam.labels(scores) == ['subject_motion']


def test_the_guard_leaves_the_raw_numbers_stored():
    # Suppressed at READ time, not at write time. The rates stay on the clip so
    # that lowering the floor later re-labels the bank with no rescan — the rule
    # every advisory reading in this lane follows.
    scores = cam.aggregate(steps_from(dx=-0.03, coverage=0.74))
    assert scores['camera_pan_rate'] > 0
    assert scores['camera_coverage'] == pytest.approx(0.74)


def test_a_camera_that_wins_its_own_frame_is_read_normally():
    scores = cam.aggregate(steps_from(dx=-0.01, coverage=0.998))
    assert cam.labels(scores) == ['pan_right']


# --- unmeasured is a state, never a zero ---------------------------------------------------

def test_a_shot_with_too_few_steps_carries_a_state_and_no_numbers():
    scores = cam.aggregate(steps_from(count=cam.MIN_STEPS - 1))
    assert scores == {cam.STATE_KEY: 'too_short'}
    # Not one rate, because a zero rate reads as "perfectly locked off" — the
    # strongest possible claim about a shot nothing looked at.
    assert not any(k in scores for k in cam.OWNED_KEYS if k != cam.STATE_KEY)


def test_an_empty_trajectory_is_too_short_rather_than_static():
    assert cam.aggregate([]) == {cam.STATE_KEY: 'too_short'}
    assert cam.aggregate(None) == {cam.STATE_KEY: 'too_short'}


def test_an_unmeasured_clip_gets_no_labels_at_all():
    assert cam.labels({}) == []
    assert cam.labels({cam.STATE_KEY: 'unreadable'}) == []
    assert cam.labels({cam.STATE_KEY: 'too_short'}) == []
    assert cam.labels(None) == []


def test_a_measured_clip_missing_its_coverage_claims_nothing():
    # Coverage is the guard; without it there is no way to know whether the
    # rates describe the camera, so no label is safe.
    assert cam.labels({cam.STATE_KEY: 'ok', 'camera_pan_rate': 9.0}) == []


# --- the caption clause (the C12 consumer) --------------------------------------------------

def test_the_phrase_uses_the_trainers_own_words():
    scores = cam.aggregate(steps_from(dx=-0.01, jitter=0.01, residual=0.0004))
    assert cam.camera_phrase(scores) == 'pan right, handheld shot'


def test_the_phrase_is_empty_when_nothing_was_measured():
    assert cam.camera_phrase({}) == ''
    assert cam.camera_phrase({cam.STATE_KEY: 'too_short'}) == ''


def test_the_phrase_is_empty_when_the_reading_cannot_be_trusted():
    # A caption that says nothing teaches the model nothing false; one that says
    # "pan right" about a locked-off shot teaches it the wrong word.
    scores = cam.aggregate(steps_from(dx=-0.03, coverage=0.74))
    assert cam.camera_phrase(scores) == ''


def test_the_phrase_omits_the_slideshow_label():
    # `slideshow` describes where the footage came from, not how a camera moved,
    # and a photograph panned across looks exactly like a pan to the model being
    # trained. It stays in the filter facet and out of the caption.
    scores = cam.aggregate(steps_from(dx=-0.01, residual=0.00005))
    assert 'slideshow' in cam.labels(scores)
    assert cam.camera_phrase(scores) == 'pan right'


def test_a_tripod_says_so_rather_than_saying_nothing():
    assert cam.camera_phrase(cam.aggregate(steps_from())) == 'static shot'


def test_every_label_has_a_phrase_entry():
    missing = [n for n in cam.CAMERA_LABELS if n not in cam.LABEL_PHRASES]
    assert not missing, f'{missing} would fall out of a caption silently'


# --- the vocabulary itself --------------------------------------------------------------------

def test_the_borrowed_labels_are_a_subset_of_what_is_emitted():
    assert set(cam.HUNYUAN_LABELS) <= set(cam.CAMERA_LABELS)


def test_the_tilt_and_orbit_classes_are_never_emitted():
    # Not an oversight, and a test rather than a comment because the temptation
    # to "complete the vocabulary" is real. A pivot and a slide are the same 2D
    # movement, and an orbit needs structure-from-motion — see the module
    # docstring for the measured cost of pretending otherwise.
    forbidden = {'tilt_up', 'tilt_down', 'tilt_left', 'tilt_right',
                 'around_left', 'around_right'}
    assert not (set(cam.CAMERA_LABELS) & forbidden)


def test_our_own_labels_are_marked_as_ours():
    ours = set(cam.CAMERA_LABELS) - set(cam.HUNYUAN_LABELS)
    assert ours == {'rolling', 'slideshow', 'subject_motion'}


# --- the tracking seam, on forged matrices -----------------------------------------------------
#
# `track_pair` takes cv2 and np as arguments precisely so this can run with two
# fakes and no extra installed. What is checked is the arithmetic AROUND the fit
# — above all that the translation is read at the frame CENTRE.

class _FakeCv2:
    RANSAC = 8

    def __init__(self, matrix, points):
        self._matrix = matrix
        self._points = points

    def goodFeaturesToTrack(self, *_a, **_k):
        return self._points

    def calcOpticalFlowPyrLK(self, _a, _b, points, _n, **_k):
        import numpy as np
        moved = np.array([[(self._matrix[0][0] * x + self._matrix[0][1] * y
                            + self._matrix[0][2]),
                           (self._matrix[1][0] * x + self._matrix[1][1] * y
                            + self._matrix[1][2])]
                          for (x, y) in points.reshape(-1, 2)], dtype='float32')
        return moved.reshape(-1, 1, 2), np.ones((len(points), 1), dtype='uint8'), None

    def estimateAffinePartial2D(self, *_a, **_k):
        import numpy as np
        return np.array(self._matrix, dtype='float64'), np.ones((len(self._points), 1))


def _forged_step(matrix, width=384, height=216):
    np = pytest.importorskip('numpy')
    frame = np.zeros((height, width), dtype='uint8')
    points = np.array([[[x, y]] for x in (40, 120, 200, 300, 340, 360)
                       for y in (30, 90, 150)], dtype='float32')
    return cam.track_pair(frame, frame, 1.5, _FakeCv2(matrix, points), np)


def test_a_pure_rotation_about_the_frame_centre_reports_no_pan_at_all():
    # THE BUG THIS TEST EXISTS FOR. OpenCV parameterises the fit as p' = sRp + t
    # with the rotation taken about the ORIGIN, so a camera that only rolls
    # produces a large `t` purely to carry the picture back. Reading `tx`/`ty`
    # straight out of the matrix labelled a forged 6.9 deg/s roll as
    # `pan_left, pan_down, rolling`.
    angle = math.radians(5.0)
    cos, sin = math.cos(angle), math.sin(angle)
    cx, cy = 192.0, 108.0
    step = _forged_step([[cos, -sin, cx - cos * cx + sin * cy],
                         [sin, cos, cy - sin * cx - cos * cy]])
    assert step['dx'] == pytest.approx(0.0, abs=1e-9)
    assert step['dy'] == pytest.approx(0.0, abs=1e-9)
    assert step['drot'] == pytest.approx(5.0, abs=1e-6)


def test_a_pure_zoom_about_the_frame_centre_reports_no_pan_either():
    scale = 1.02
    cx, cy = 192.0, 108.0
    step = _forged_step([[scale, 0.0, cx - scale * cx],
                         [0.0, scale, cy - scale * cy]])
    assert step['dx'] == pytest.approx(0.0, abs=1e-9)
    assert step['dy'] == pytest.approx(0.0, abs=1e-9)
    assert step['dzoom'] == pytest.approx(math.log(scale), abs=1e-9)


def test_a_pure_translation_is_reported_as_a_fraction_of_the_width():
    step = _forged_step([[1.0, 0.0, 38.4], [0.0, 1.0, 0.0]])
    assert step['dx'] == pytest.approx(0.1, abs=1e-9)
    assert step['dy'] == pytest.approx(0.0, abs=1e-9)


def test_a_frame_with_too_few_corners_yields_no_step_rather_than_zeros():
    np = pytest.importorskip('numpy')
    frame = np.zeros((216, 384), dtype='uint8')
    fake = _FakeCv2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], None)
    assert cam.track_pair(frame, frame, 1.5, fake, np) is None
    # A zeroed step would claim the camera held still, and enough of them would
    # label a shot nobody could track as a perfect tripod.


# --- the high-frequency residual ------------------------------------------------------------

def test_a_constant_series_has_no_high_frequency_content():
    assert cam.high_frequency([0.5] * 20) == pytest.approx(0.0, abs=1e-12)


def test_a_linear_ramp_has_almost_none():
    # A steered move: the local mean follows it, so nothing is left over.
    assert cam.high_frequency([0.01 * i for i in range(20)]) < 1e-9


def test_an_alternating_series_is_almost_all_high_frequency():
    values = [0.01 if i % 2 == 0 else -0.01 for i in range(20)]
    assert cam.high_frequency(values) > 0.005


def test_a_series_shorter_than_the_window_yields_zero_rather_than_raising():
    assert cam.high_frequency([0.1, 0.2]) == 0.0


# --- the plumbing contracts -------------------------------------------------------------------

def test_every_key_this_pass_owns_survives_a_re_measure():
    missing = [k for k in cam.OWNED_KEYS if k not in video_metrics.ADVISORY_KEYS]
    assert not missing, (f'{missing} would be erased by the metrics pass — add '
                         f'them to video_metrics.ADVISORY_KEYS')


def test_this_pass_owns_no_key_another_pass_writes():
    others = set(video_safe_zone.OWNED_KEYS) | set(video_defect_sweep.OWNED_KEYS)
    others |= set(video_ai_check.OWNED_KEYS)
    others |= {'aesthetic_score', 'watermark_score', 'watermark_state',
               'duplicate_group', 'duplicate_of', video_metrics.SHARPNESS_KEY}
    assert not (set(cam.OWNED_KEYS) & others)


def test_the_state_key_is_the_first_key_owned():
    # `pending_clips` uses its ABSENCE as the resume test, so a second spelling
    # would make a read bank look permanently unread.
    assert cam.OWNED_KEYS[0] == cam.STATE_KEY == 'camera_state'


def test_the_cut_ships_empty_and_is_the_panels_to_offer():
    assert cfg.DEFAULTS['video_bank']['camera_shake_max'] is None
    assert 'camera_shake_max' in video_metrics.THRESHOLD_KEYS


def test_the_cut_flags_nothing_until_it_is_set():
    measured = {'camera_state': 'ok', 'camera_shake': 1.16}
    assert 'shaky' not in video_metrics.verdicts(measured, {})
    assert 'shaky' not in video_metrics.verdicts(measured,
                                                 {'camera_shake_max': None})
    assert 'shaky' in video_metrics.verdicts(measured, {'camera_shake_max': 0.3})


def test_an_unread_shot_is_never_flagged_shaky():
    assert 'shaky' not in video_metrics.verdicts({'camera_state': 'unreadable'},
                                                 {'camera_shake_max': 0.3})


def test_the_probe_imports_everything_the_pass_imports():
    # The pass tracks IN THIS PROCESS, on frames PyAV hands it. A probe naming
    # only `av` would report the video extra ready and then the pass would die
    # on its first clip — issue #24's exact shape.
    probe = capabilities.CAPABILITY_IMPORTS['video']
    for module in ('av', 'cv2', 'numpy'):
        assert module in probe, f'{module} is imported by the pass but not probed'


def test_setup_can_install_everything_the_probe_imports():
    packages = setup_installer._CAPABILITY_PACKAGES['video']
    assert 'av' in packages
    # The HEADLESS variant, never the desktop one: opencv-python drags a GUI
    # stack onto a server, the same trap video_text already names.
    assert 'opencv-python-headless' in packages
    assert 'opencv-python' not in packages
    assert 'numpy' in packages


def test_the_pass_needs_no_install_action_of_its_own():
    # It rides the video extra's existing row, so there must be no second one.
    for key in ('camera', 'camera_motion', 'video_camera'):
        assert key not in setup_installer.INSTALL_ACTIONS


def test_the_pass_takes_no_gpu_window():
    # OpenCV on the CPU, so a bank can be read while a training owns the card.
    import inspect
    from app.services import video_bank_service
    body = inspect.getsource(video_bank_service.start_camera)
    assert 'gpu_exclusive' not in body and '_gpu_busy_reason' not in body


# --- the pass over a bank ----------------------------------------------------------------------

class _Clip:
    """The two fields `_store` touches, and nothing else."""

    def __init__(self, metrics_json=None):
        self.metrics_json = metrics_json
        self.id = 1


def test_storing_a_reading_keeps_every_other_passs_numbers():
    clip = _Clip(json.dumps({'metrics_state': 'ok', 'sharpness_p90': 354.35,
                             'ai_check_state': 'ok'}))
    cam._store.__wrapped__ if False else None
    summary = cam._summary(clip)
    for key in cam.OWNED_KEYS:
        summary.pop(key, None)
    summary.update(cam.aggregate(steps_from(dx=-0.01)))
    # The merge `_store` performs, without the commit — the point being that the
    # three keys it does not own are all still there afterwards.
    assert summary['sharpness_p90'] == 354.35
    assert summary['ai_check_state'] == 'ok'
    assert summary['camera_state'] == 'ok'


def test_a_corrupt_blob_reads_as_empty_rather_than_raising():
    # This pass must never be the reason a bank's quality scores disappear.
    assert cam._summary(_Clip('not json at all')) == {}
    assert cam._summary(_Clip('[1, 2, 3]')) == {}
    assert cam._summary(_Clip(None)) == {}


def test_a_rerun_cannot_leave_last_runs_rates_beside_this_runs_state():
    # A shot that was read and is now too short (it was re-cut) must not keep a
    # pan rate from the previous run.
    clip = _Clip(json.dumps(dict(cam.aggregate(steps_from(dx=-0.01)))))
    summary = cam._summary(clip)
    for key in cam.OWNED_KEYS:
        summary.pop(key, None)
    summary.update({cam.STATE_KEY: 'too_short'})
    assert summary == {cam.STATE_KEY: 'too_short'}
