"""🎥 How the CAMERA moved — pan, tilt, zoom, roll, tripod, handheld, slideshow.

WHAT IT ANSWERS, and why it is a LABEL and not a cut. A video LoRA learns camera
language along with everything else, and the two people training on the same bank
want opposite halves of it: one is building a locked-off product shot and every
wobble is contamination, the other is training a handheld look and the wobble IS
the target. MotionDirector's whole premise is the second. So this pass never
rejects anything — it says what each shot DOES, and the person filtering decides
which half they wanted. One optional cut exists (`camera_shake_max`) for the
first user, ships empty like every other cut, and flags nothing until it is set.

THE VOCABULARY IS BORROWED, NOT INVENTED. The labels come from the fourteen
classes Hunyuan's camera classifier uses, because that is the vocabulary of the
model this app exists to train — a field that will later feed a caption's
`camera` clause has to speak the words the trainer already knows. Eight of the
fourteen are emitted. The other six are NOT, and each omission is a measurement
this cannot make rather than a shortcut:

  tilt_up / tilt_down / tilt_left / tilt_right
      A camera that PIVOTS and a camera that SLIDES produce the same 2D
      translation on the sensor. Nothing in a global 2D transform separates a
      yaw from a lateral truck, or a pitch from a crane — the difference is
      parallax, and parallax is depth. So the whole translation family is
      reported as `pan_*`, which is the honest superset, and `tilt_*` is never
      emitted rather than guessed at fifty-fifty.
  around_left / around_right
      An orbit is a translation along an arc with a counter-rotation. Its
      signature is precisely the parallax a single similarity cannot express;
      recovering it is structure-from-motion. CameraBench (NeurIPS 2025) puts
      the best geometric system, MegaSaM, at about 50 % AP on this class of
      question at MINUTES per clip — so the option is not "cheap or accurate",
      it is "expensive and still a coin flip". Omitted.

Three labels are this app's own, and are named as such wherever they surface:
`rolling` (roll about the optical axis — measured here, and absent from
Hunyuan's fourteen), `slideshow` (a still photograph being panned across, which
is not a camera move at all) and `subject_motion` (a GUARD, see below).

WHY NOT vidstab, WHICH IS ALREADY IN THE BINARY. `vidstabdetect` ships in the
ffmpeg this app resolves — verified on the bundled imageio-ffmpeg build, not
assumed — and it costs a fraction of a second per clip. It was still the wrong
tool, for a reason that only shows up once you read what it writes:

  * Its `.trf` holds LOCAL MOTION VECTORS, one bag per frame, not a global
    transform. The aggregation into (dx, dy, angle, zoom) happens inside
    `vidstabtransform` and is never written down. Using vidstab would mean
    reimplementing that aggregation over its output — so the arithmetic below
    would exist either way, on a worse input.
  * That input is quantised to WHOLE PIXELS and carries no rotation and no
    scale. Both are things this pass reports, and roll is the single
    measurement CameraBench singles out as the one everything else gets wrong.
  * It writes BINARY by default (magic `TRF1`); text needs `fileformat=ascii`,
    and the text of one five-second clip is 294 KB of vectors to parse.
  * It offers no inlier ratio, which is the number that separates a moving
    CAMERA from a moving SUBJECT — see `camera_coverage`. Without it the whole
    pass is unguarded.

Sparse Lucas-Kanade tracking plus a RANSAC similarity fit gives all four
components in sub-pixel precision AND the inlier ratio, in-process, at 0.07 s per
second of source. That is the design.

⚠️ THE GUARD IS THE PART TO READ TWICE. A RANSAC fit returns the DOMINANT
motion, and when a subject fills enough of the frame the dominant motion is the
SUBJECT. Measured, on a forged clip whose camera was a tripod and whose subject
was a textured block crossing 35 % of the frame: the fit reported a 8.47 %/s pan
and a -5.55 %/s zoom, both pure fiction, off a camera that never moved. What it
also reported was `camera_coverage` 0.739 against 0.998-1.000 for every clip
where the camera really was the dominant motion. So when coverage falls below
its floor NO direction is emitted — only `subject_motion` — because a confident
wrong answer about camera language is worse than no answer. This is the Koala
moving-surface rule with the polarity that matters here: the surface in motion
is not a defect, it is the reason the reading cannot be trusted.

UNMEASURED IS A STATE, NEVER A ZERO. A shot with too few usable steps carries
`camera_state: 'too_short'` and no numbers; one whose frames would not decode, or
whose picture is too flat to track (a wall, a black frame, a sky), carries
'unreadable'. Neither is a zero, because zero here reads as "perfectly locked
off" — the strongest possible claim about a shot nothing looked at.

THE OTHER SURFACE, named rather than assumed. The image bank has NO equivalent
and cannot have one: every number here is a rate between two instants, and a
still photograph has no second instant. This is the same legitimate divergence
`dup_frame_ratio` states in the defect sweep, and it needs no port.
"""
from __future__ import annotations

import json
import logging
import math

from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource

logger = logging.getLogger(__name__)

# --- what gets stored ---------------------------------------------------------------

STATE_KEY = 'camera_state'

# Every key this pass owns. Cleared wholesale by `_store` before a re-run writes,
# and pinned against video_metrics.ADVISORY_KEYS by a test — a key this writes
# that the advisory list does not carry is erased by the next quality scan, in
# silence, sending a whole bank back through a pass it has already paid for.
OWNED_KEYS = (STATE_KEY, 'camera_steps',
              'camera_pan_rate', 'camera_tilt_rate', 'camera_zoom_rate',
              'camera_roll_rate', 'camera_travel', 'camera_shake',
              'camera_coverage', 'camera_residual')

# --- the measurement's own constants ------------------------------------------------

# Width the analysis copy is reduced to, and it is NOT a free performance dial —
# it sets the sensitivity of the guard. `camera_coverage` counts points that fall
# within a RANSAC tolerance expressed in analysis pixels, so shrinking the frame
# shrinks every displacement toward that tolerance and a moving subject starts
# fitting the camera model. Measured on the forged subject-motion clip (a tripod,
# a block crossing 35 % of frame), coverage against the same clip's true value:
#
#     160 px   0.995     the guard is blind — indistinguishable from a tripod
#     256 px   0.911     borderline
#     384 px   0.739     clean separation from 0.998-1.000
#     512 px   0.635     no better separated, and 35 % slower
#
# 384 is where the guard starts working and the cost stops being worth paying.
# Note that video_metrics_scan's ANALYSIS_WIDTH is 160 for its own good reasons —
# this pass cannot ride that decode, and the table above is why.
ANALYSIS_WIDTH = 384

# Ceiling on how often the trajectory is sampled. Native rate below this, so 24
# and 25 fps material is read frame by frame; 50 and 60 fps material is read at
# 30 and pays half. Safe for the one statistic that cares about frequency:
# `camera_shake` is a residual against a local mean, not a spectrum, so tremor
# above the Nyquist limit folds to a lower frequency and STILL fails to be
# smooth. It would not be safe if this reported a frequency, and it does not.
MAX_SAMPLE_FPS = 30.0

# RANSAC reprojection tolerance, as a FRACTION of the analysis width rather than
# a pixel count. The two are calibrated together (see ANALYSIS_WIDTH) and a pixel
# constant would silently decalibrate the guard the day the width moves.
# 0.004 x 384 = 1.5 px.
RANSAC_TOLERANCE = 0.004

# Corners tracked per step, and the fewest that may survive the tracking before a
# step is discarded as unusable. A similarity has four degrees of freedom; six
# points is the smallest sample that leaves RANSAC anything to disagree about.
TRACK_POINTS = 200
MIN_POINTS = 6

# Frames in the moving average that defines "smooth". `camera_shake` is what a
# 5-frame mean cannot explain, which at 25 fps puts the boundary at about 2.5 Hz:
# below it a move is intentional, above it nobody is steering. Handheld tremor
# sits at 8-12 Hz and lands well inside the residual; a whip pan is fast but
# perfectly steered, so the mean follows it and it does NOT read as shake —
# measured at 0.037 on a 4.6 %/s forged pan, against 1.16 on forged tremor.
SMOOTH_WINDOW = 5

# Fewest steps an aggregate may be built from. Below it the mean of a rate and
# the spread of a residual are both noise, and a two-frame "shot" would get a
# confident label. At 25 fps this is a third of a second.
MIN_STEPS = 8

# --- the label thresholds, all internal ----------------------------------------------
#
# INTERNAL AND NOT SETTINGS, unlike every quality cut in this lane, because these
# do not reject anything — they name what a shot is. A user who disagrees with
# where "a pan" starts is not blocked by that disagreement: the raw rates are
# stored, the labels are derived from them AT READ TIME, and moving a constant
# here re-labels a whole bank with no rescan. The one number a user does control
# is `camera_shake_max`, which is a cut and therefore a setting.
#
# Units, once: every rate is a PERCENTAGE OF THE FRAME WIDTH per second, except
# the roll which is degrees per second. Width and not height, for both axes, so
# the same physical movement reads the same on 16:9 and on 9:16.

# A shot must drift by this much of its own width every second before it is
# called a pan — a tenth of the frame over a ten-second shot. Below it, a
# reading is the tracker's noise floor and a tripod's own micro-drift: the
# forged tripod measures 0.000 against 4.62 for a forged pan.
PAN_FLOOR = 1.5

# Forged zoom measures 5.36 %/s and forged Ken Burns 3.66; everything with no
# zoom in it measures at most 0.12, most of that the encoder's own rounding.
ZOOM_FLOOR = 1.0

# Degrees per second. A forged 6.9 deg/s roll reads 6.82; every clip with no
# roll reads under 0.05, except a parallax pan which manufactures 1.28 of
# apparent roll out of depth alone — the floor sits above that, deliberately.
ROLL_FLOOR = 2.0

# Percent of frame width, per frame, of high-frequency residual. Forged tremor
# measures 1.16; every smoothly-moved clip measures at most 0.10. An eleven-fold
# gap, and the floor sits in the middle of it on the log scale.
#
# ⚠️ It was set from a STRONG synthetic tremor (about 7 px of 1080p at 7-11 Hz).
# Gentle handheld — a braced operator, a stabilised phone — will sit under it and
# be called static. The raw number is stored, so that judgement is reversible;
# the label is the conservative half of it.
SHAKE_FLOOR = 0.30

# Median reprojection error, in percent of frame width, below which a MOVING shot
# is a photograph rather than a camera. A Ken Burns move is one affine transform
# of one picture and fits to within the tracker's own precision; a real camera
# moving through a scene with depth cannot, because near and far move at
# different rates. Measured: forged Ken Burns 0.0075, forged parallax pan 0.0405.
#
# ⚠️ THE FALSE POSITIVE IS REAL AND IS NAMED IN THE UI. A genuine pan across a
# scene with no depth — a flat wall, a horizon, a sky, a distant skyline — has no
# parallax either and reads as a slideshow. This says "the frame moved as one
# rigid picture", which a photograph always does and a flat scene also does.
SLIDESHOW_RESIDUAL = 0.020

# Share of tracked points the dominant transform must explain before any
# direction is reported at all. Below it only `subject_motion` is emitted — see
# the module docstring for the fiction a fit produces when a subject wins.
# Measured: 0.739 when a subject dominates, 0.998-1.000 when the camera does.
COVERAGE_FLOOR = 0.85

# --- the vocabulary ------------------------------------------------------------------

# CANONICAL ORDER, and it is the order labels are emitted and displayed in, not
# alphabetical: a card reading "pan right, zoom in, handheld shot" is a sentence
# and "handheld shot, pan right, zoom in" is a list. Frozen — these strings reach
# a stored caption and a saved filter, so renaming one needs an alias path.
CAMERA_LABELS = ('pan_left', 'pan_right', 'pan_up', 'pan_down',
                 'zoom_in', 'zoom_out', 'rolling',
                 'static_shot', 'handheld_shot', 'slideshow', 'subject_motion')

# Which of the above are Hunyuan's own words, and which are this app's. Kept as
# data rather than as a comment because the UI states the difference to the user
# and a test pins the split — an app-specific label quietly presented as part of
# the trainer's vocabulary is how a caption ends up with a word no model knows.
HUNYUAN_LABELS = ('pan_left', 'pan_right', 'pan_up', 'pan_down',
                  'zoom_in', 'zoom_out', 'static_shot', 'handheld_shot')

# What each label says in a caption clause. Hunyuan's own words for its own
# labels, plain English for ours.
LABEL_PHRASES = {
    'pan_left': 'pan left', 'pan_right': 'pan right',
    'pan_up': 'pan up', 'pan_down': 'pan down',
    'zoom_in': 'zoom in', 'zoom_out': 'zoom out',
    'rolling': 'rolling camera',
    'static_shot': 'static shot', 'handheld_shot': 'handheld shot',
    # Both deliberately silent in a caption, for opposite reasons. `slideshow`
    # describes where the footage CAME FROM, not how a camera moved — and a
    # photograph panned across looks, to the model being trained, exactly like a
    # pan, so the caption should say "pan right" and let the filter facet carry
    # the provenance. `subject_motion` is silent because it means the reading is
    # untrustworthy, and `camera_phrase` returns nothing at all in that case.
    'slideshow': '',
    'subject_motion': '',
}


# --- pure arithmetic over a trajectory ------------------------------------------------

def high_frequency(values, window=SMOOTH_WINDOW):
    """How much of a series a moving average of `window` cannot explain.

    The whole shake measurement, and it is deliberately a residual rather than a
    spectrum: a Fourier transform would need a constant sample rate this lane
    cannot promise (variable-frame-rate footage is most of what a phone or a
    screen recorder produces) and would answer a question nobody asked. What is
    wanted is one number saying "is anybody steering", and the spread of what a
    local mean misses is exactly that.

    Padded rather than truncated, so a short series keeps every sample: the
    alternative loses two frames at each end, which on a one-second shot is a
    sixth of the evidence.

    ⚠️ AND THE PADDING EXTRAPOLATES A SLOPE RATHER THAN REPEATING THE EDGE, which
    looks like a detail and is a false positive. What is smoothed here is the
    per-step displacement, so a steady pan is a CONSTANT series and is safe
    either way — but a move that ACCELERATES (a pan easing in, a zoom winding up,
    a camera being picked up) is a ramp, and a flat moving average cannot follow
    a ramp through a region padded with a repeated end value. The residual it
    leaves at each end grows with the acceleration, so easing into a move would
    read as tremor. Measured on a forged ramp of 0.01 per step: 0.002 of pure
    artefact with the naive padding, and zero with this one. An alternating
    series is untouched by the change — its slope across the window is zero, so
    the padding is the same constant the naive version used.

    For scale, what is left on real footage: a forged steady 4.6 %/s pan with no
    tremor in it reads 0.037 here, which is the tracker's and the encoder's own
    noise and sits an order of magnitude below SHAKE_FLOOR.
    """
    series = [float(v) for v in (values or [])]
    if len(series) < window or window < 2:
        return 0.0
    half = window // 2
    reach = min(window, len(series))
    head_slope = (series[reach - 1] - series[0]) / (reach - 1)
    tail_slope = (series[-1] - series[-reach]) / (reach - 1)
    padded = ([series[0] - head_slope * (half - i) for i in range(half)]
              + series
              + [series[-1] + tail_slope * (i + 1) for i in range(half)])
    smooth = [sum(padded[i:i + window]) / window for i in range(len(series))]
    residual = [series[i] - smooth[i] for i in range(len(series))]
    mean = sum(residual) / len(residual)
    variance = sum((r - mean) ** 2 for r in residual) / len(residual)
    return variance ** 0.5


def aggregate(steps):
    """The numbers ONE clip carries, from its per-step trajectory.

    `steps` is [{'dt', 'dx', 'dy', 'dzoom', 'drot', 'coverage', 'residual'}] as
    `clip_steps` produces it — every displacement already expressed as a
    FRACTION OF THE ANALYSIS WIDTH, so nothing here knows or cares what
    resolution the footage was.

    ⚠️ THE FOUR SIGN CONVENTIONS, because every one of them is an inversion
    waiting to happen and three of them are counter-intuitive. A step's transform
    describes how the PICTURE moved; every number stored here describes how the
    CAMERA moved, which is the opposite for translation and rotation and the same
    for scale:

      dx > 0   the picture slid RIGHT   → the camera panned LEFT
      dy > 0   the picture slid DOWN    → the camera tilted UP (screen y grows
                                          downward, so the two inversions cancel
                                          and this one alone is NOT negated)
      drot > 0 the picture turned CW    → the camera rolled COUNTER-clockwise
      scale> 1 the picture grew         → the camera zoomed IN (no inversion:
                                          both mean "closer")

    So `camera_pan_rate` is positive when the camera pans right,
    `camera_tilt_rate` when it tilts up, `camera_roll_rate` when it rolls
    clockwise, `camera_zoom_rate` when it zooms in. Pinned by tests built on
    transforms forged in arithmetic, where the true answer is known exactly and
    no encoder sits between the intent and the measurement.

    Returns `{'camera_state': 'too_short'}` and nothing else below MIN_STEPS.
    """
    usable = [s for s in (steps or []) if s and s.get('dt')]
    if len(usable) < MIN_STEPS:
        return {STATE_KEY: 'too_short'}
    span = sum(float(s['dt']) for s in usable)
    if span <= 0:
        return {STATE_KEY: 'too_short'}
    # Per-second rates from a per-step series: the mean step divided by the mean
    # interval, which is the total displacement over the total time. Not the mean
    # of per-step rates — on variable-frame-rate footage a short interval would
    # weigh as much as a long one and a single 2 ms gap would dominate the shot.
    fps = len(usable) / span
    mean = lambda key: sum(float(s[key]) for s in usable) / len(usable)
    dx, dy = mean('dx'), mean('dy')
    # PATH LENGTH, not net displacement, and it is the one stored number no
    # label reads (see `labels` for why a travel floor was dropped). It is kept
    # because it is the only number that survives a round trip: forged tremor
    # covers 24 % of the frame width every second and ends where it started, so
    # its net pan reads 0.22 and every rate above says "nothing happened". A
    # reader comparing travel against the rates can see that immediately.
    travel = sum(math.hypot(float(s['dx']), float(s['dy']))
                 for s in usable) / len(usable)
    shake = math.hypot(high_frequency([s['dx'] for s in usable]),
                       high_frequency([s['dy'] for s in usable]))
    return {
        STATE_KEY: 'ok',
        'camera_steps': len(usable),
        'camera_pan_rate': round(-dx * fps * 100.0, 4),
        'camera_tilt_rate': round(dy * fps * 100.0, 4),
        'camera_zoom_rate': round(mean('dzoom') * fps * 100.0, 4),
        'camera_roll_rate': round(-mean('drot') * fps, 4),
        'camera_travel': round(travel * fps * 100.0, 4),
        'camera_shake': round(shake * 100.0, 4),
        'camera_coverage': round(mean('coverage'), 4),
        'camera_residual': round(mean('residual') * 100.0, 4),
    }


def labels(scores):
    """The camera labels for one clip, in CAMERA_LABELS order. [] when unknown.

    Derived at READ time from the stored rates, which is what lets a threshold
    above move without re-decoding a single frame — the rule every advisory
    reading in this lane follows.

    THE GUARD COMES FIRST AND SUPPRESSES EVERYTHING. Below COVERAGE_FLOOR the
    dominant motion was not the camera, so the rates describe a subject and
    reporting a direction off them would be a confident lie (see the module
    docstring for the measured fiction). Only `subject_motion` is returned.

    A shot can carry SEVERAL labels — a handheld pan that also zooms is all
    three, and flattening that to one winner would throw away the two the reader
    was filtering for. `static_shot` is the exception: it is what "none of the
    above" is called, so it never appears beside another motion.
    """
    if not isinstance(scores, dict) or scores.get(STATE_KEY) != 'ok':
        return []
    got = lambda key: scores.get(key)
    coverage = got('camera_coverage')
    if coverage is None:
        return []
    if coverage < COVERAGE_FLOOR:
        return ['subject_motion']
    found = set()
    pan = got('camera_pan_rate')
    if pan is not None and abs(pan) >= PAN_FLOOR:
        found.add('pan_right' if pan > 0 else 'pan_left')
    tilt = got('camera_tilt_rate')
    if tilt is not None and abs(tilt) >= PAN_FLOOR:
        found.add('pan_up' if tilt > 0 else 'pan_down')
    zoom = got('camera_zoom_rate')
    if zoom is not None and abs(zoom) >= ZOOM_FLOOR:
        found.add('zoom_in' if zoom > 0 else 'zoom_out')
    roll = got('camera_roll_rate')
    if roll is not None and abs(roll) >= ROLL_FLOOR:
        found.add('rolling')
    shake = got('camera_shake')
    if shake is not None and shake >= SHAKE_FLOOR:
        found.add('handheld_shot')
    # A photograph rather than a camera, and only worth saying about a shot that
    # MOVES: a static frame is trivially one rigid picture, so calling every
    # tripod shot a slideshow would make the label useless.
    residual = got('camera_residual')
    if found and residual is not None and residual < SLIDESHOW_RESIDUAL:
        found.add('slideshow')
    # STATIC IS "NONE OF THE ABOVE", and deliberately not its own measurement.
    # The obvious-looking alternative — call a shot static when its total travel
    # is under a floor — was tried and dropped: travel is a sum of MAGNITUDES, so
    # per-step tracking noise never cancels in it. A forged pure zoom measures
    # 2.3 %/s of travel off a centre that does not move at all, which is sub-pixel
    # jitter accumulating, and any floor low enough to be meaningful would leave
    # a genuinely locked-off shot on real grainy footage with NO label at all.
    # Every jitter that could fake travel is high-frequency, and `camera_shake`
    # already reads it — so if no pan, tilt, zoom, roll or shake cleared its
    # floor, the camera was holding still, and that is the whole test.
    if not found:
        found.add('static_shot')
    return [name for name in CAMERA_LABELS if name in found]


def camera_phrase(scores):
    """The camera clause for one clip, e.g. 'pan right, handheld shot'. '' when
    there is nothing honest to say.

    WRITTEN FOR A CONSUMER THAT DOES NOT EXIST YET — the caption pass will put
    this in a caption's `camera` field, because a VLM cannot produce it (that was
    tried and refuted twice) and a trainer needs the words its own classifier
    uses. Shipped and tested now rather than later, so the day it is wired the
    vocabulary is already frozen and already correct.

    EMPTY, not a guess, in the two cases where a claim would be wrong: nothing
    measured, and `subject_motion` — a caption saying nothing about the camera is
    a caption the model learns nothing false from, while a caption saying "pan
    right" about a locked-off shot teaches the model the wrong word.
    """
    names = labels(scores)
    if not names or 'subject_motion' in names:
        return ''
    return ', '.join(p for p in (LABEL_PHRASES.get(n, '') for n in names) if p)


# --- the decode + tracking seam --------------------------------------------------------

def clip_steps(path, start_s, end_s, analysis_width=ANALYSIS_WIDTH):
    """The per-step trajectory of ONE shot. The only impure function here.

    Monkeypatched in tests, so every aggregate, every label and the caption
    phrase above are exercised with no PyAV, no OpenCV and no footage — the same
    split the defect sweep draws between its ffmpeg subprocess and its parsers.

    Raises on a shot that will not decode; the caller retires THAT shot and keeps
    going, because a bank is measured in bulk and one broken file must cost that
    file.

    ONE seek and then a forward decode of every frame in the window. Sampling by
    seek would be wrong here in a way it is not for the other passes: this
    measures what happens BETWEEN adjacent frames, so a gap in the series is not
    a smaller sample, it is a step whose displacement is silently several steps'
    worth and whose shake residual is meaningless.
    """
    import av
    import cv2
    import numpy as np

    tolerance = max(0.5, analysis_width * RANSAC_TOLERANCE)
    min_gap = 1.0 / MAX_SAMPLE_FPS - 1e-6
    steps = []
    previous = None
    previous_t = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        try:
            container.seek(int(float(start_s) / (stream.time_base or 1)),
                           stream=stream)
        except Exception:            # noqa: BLE001 — some streams refuse to
            pass                     # seek; decoding from zero still works
        for frame in container.decode(stream):
            ts = (float(frame.pts * stream.time_base)
                  if frame.pts is not None else 0.0)
            if ts < start_s:
                continue
            if ts >= end_s:
                break
            if previous_t is not None and ts - previous_t < min_gap:
                continue
            # Greyscale and reduced BEFORE any arithmetic, exactly as the metrics
            # scan does it: corner detection and optical flow both read intensity
            # only, and colour would triple the bytes for nothing.
            height = max(2, round(frame.height * analysis_width
                                  / max(frame.width, 1)))
            gray = frame.reformat(width=analysis_width, height=height,
                                  format='gray').to_ndarray()
            if previous is not None:
                step = track_pair(previous, gray, tolerance, cv2, np)
                if step is not None:
                    step['dt'] = ts - previous_t
                    steps.append(step)
            previous, previous_t = gray, ts
    return steps


def track_pair(first, second, tolerance, cv2, np):
    """One step: how the picture moved between two adjacent frames, or None.

    None rather than zeros whenever the answer would be invented — too few
    corners (a flat wall, a black frame, an overexposed sky), too few surviving
    the tracking, or a RANSAC that found no consensus. A dropped step shortens
    the series; a zeroed one would claim the camera held still, and enough of
    them would label a shot nobody could track as a perfect tripod.

    `cv2` and `np` are passed in rather than imported, so the caller owns the one
    import site and this stays callable from a test with two fakes.
    """
    corners = cv2.goodFeaturesToTrack(first, maxCorners=TRACK_POINTS,
                                      qualityLevel=0.01, minDistance=8,
                                      blockSize=7)
    if corners is None or len(corners) < MIN_POINTS:
        return None
    moved, status, _err = cv2.calcOpticalFlowPyrLK(first, second, corners, None,
                                                   winSize=(21, 21), maxLevel=3)
    if moved is None or status is None:
        return None
    kept = status.ravel() == 1
    src = corners[kept].reshape(-1, 2)
    dst = moved[kept].reshape(-1, 2)
    if len(src) < MIN_POINTS:
        return None
    # A SIMILARITY (translation, one rotation, one uniform scale) and not a full
    # homography, which is the tempting upgrade and the wrong one. A homography
    # has eight degrees of freedom and will happily absorb a moving subject into
    # a plausible perspective warp — the extra freedom is spent explaining
    # exactly what `camera_coverage` exists to catch. Four parameters is also all
    # the vocabulary needs: pan, tilt, zoom and roll, and nothing else.
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=tolerance,
        maxIters=2000, confidence=0.99)
    if matrix is None:
        return None
    a11, a12, tx = matrix[0]
    a21, a22, ty = matrix[1]
    scale = math.hypot(float(a11), float(a21))
    if scale <= 0:
        return None
    height, width = float(first.shape[0]), float(first.shape[1])
    # ⚠️ THE TRANSLATION IS READ AT THE FRAME CENTRE, NOT OFF `tx`/`ty`, and the
    # difference is not cosmetic. OpenCV parameterises the fit as p' = sRp + t
    # with the rotation and the scale taken about the ORIGIN — the top-left
    # corner. A camera that only ROLLS, about its own optical axis, therefore
    # produces a large `t` purely to carry the picture back from where rotating
    # about the corner sent it. Caught on a forged 6.9 deg/s roll about the frame
    # centre: `tx`/`ty` read as a 3.4 %/s pan and a 6.0 %/s tilt that no camera
    # performed, and the shot came out labelled `pan_left, pan_down, rolling`.
    # Where the CENTRE of the frame goes is the question actually being asked,
    # and it is zero for a pure roll and for a pure zoom by construction.
    centre_x, centre_y = width / 2.0, height / 2.0
    moved_x = float(a11) * centre_x + float(a12) * centre_y + float(tx)
    moved_y = float(a21) * centre_x + float(a22) * centre_y + float(ty)
    # Residual of EVERY tracked point, inlier or not, against the dominant model
    # — the median, so a handful of wild outliers move it far less than the
    # parallax of a whole scene does. This is the number that separates a
    # photograph from a camera; taking the mean would let three bad tracks
    # imitate depth.
    projected = (src @ matrix[:, :2].T) + matrix[:, 2]
    error = np.linalg.norm(projected - dst, axis=1)
    return {
        'dx': (moved_x - centre_x) / width,
        'dy': (moved_y - centre_y) / width,
        'dzoom': math.log(scale),
        'drot': math.degrees(math.atan2(float(a21), float(a11))),
        'coverage': float(inliers.mean()) if inliers is not None else 0.0,
        'residual': float(np.median(error)) / width,
    }


# --- the pass ---------------------------------------------------------------------------

def unavailable_reason():
    """None when this install can read camera motion, else the sentence saying
    why not.

    The decode extra's own probe, because this pass adds no environment of its
    own: it decodes with the PyAV this lane already requires and fits with the
    OpenCV that same extra now installs. Inventing a second sentence for one
    missing install is how a user installs twice.
    """
    from .. import capabilities
    verdict = capabilities.probe_video()
    if verdict['decode']:
        return None
    return (f'reading camera motion needs the video decode extra — '
            f'{verdict["detail"]}')


def pending_clips(bank_id, rescan=False):
    """The shots this pass would measure, oldest first.

    Only shots of a source that PROBED, like every reading pass here: an
    unreadable file has no frame to decode, and counting it as a failure on every
    run would make the pass look permanently broken.

    "Already measured" is a key in the blob rather than a column, like every
    other advisory pass in this lane — the resume test is a JSON read, and legacy
    databases need no migration to carry it.
    """
    rows = (VideoClip.query.filter_by(bank_id=bank_id)
            .join(VideoSource, VideoSource.id == VideoClip.source_id)
            .filter(VideoSource.probe_state == 'ok')
            .order_by(VideoClip.id.asc()).all())
    if rescan:
        return rows
    return [c for c in rows if STATE_KEY not in _summary(c)]


def run_camera_motion(bank_id, rescan=False, *, on_clip=None, should_stop=None):
    """Read every shot of a bank that has no camera reading yet.

    Returns {'measured', 'too_short', 'unreadable', 'error'}. Committed per clip,
    the resume contract every pass in this lane keeps: stopping loses nothing and
    a re-run pays only for the shots the first run had not reached.
    """
    out = {'measured': 0, 'too_short': 0, 'unreadable': 0, 'error': None}
    bank = db.session.get(VideoBank, bank_id)
    if bank is None:
        return out
    rows = pending_clips(bank_id, rescan)
    if not rows:
        return out
    from .video_bank_service import _abs_source_path
    relpaths = {s.id: s.relpath for s in
                VideoSource.query.filter_by(bank_id=bank_id).all()}

    for clip in rows:
        if should_stop is not None and should_stop():
            break
        path = _abs_source_path(bank, relpaths.get(clip.source_id) or '')
        if not path:
            _store(clip, {STATE_KEY: 'unreadable'})
            out['unreadable'] += 1
            if on_clip is not None:
                on_clip()
            continue
        try:
            steps = clip_steps(path, float(clip.start_s), float(clip.end_s))
        except Exception as e:       # noqa: BLE001 — one shot never sinks a pass
            logger.warning('camera motion: clip %s not read: %s', clip.id, e)
            out['error'] = f'{type(e).__name__}: {e}'
            _store(clip, {STATE_KEY: 'unreadable'})
            out['unreadable'] += 1
            if on_clip is not None:
                on_clip()
            continue
        summary = aggregate(steps)
        _store(clip, summary)
        if summary.get(STATE_KEY) == 'ok':
            out['measured'] += 1
        else:
            out['too_short'] += 1
        if on_clip is not None:
            on_clip()
    return out


def _store(clip, values):
    """Merge this pass's reading into the clip's blob and commit it.

    MERGE, not replace: metrics_json holds the quality scores an expensive pass
    produced plus seven other passes' verdicts, and overwriting it here would
    erase them silently. The keys this pass OWNS are cleared first, so a re-run
    that now finds a shot too short cannot leave last run's rates beside this
    run's state.
    """
    summary = _summary(clip)
    for key in OWNED_KEYS:
        summary.pop(key, None)
    summary.update(values)
    clip.metrics_json = json.dumps(summary)
    db.session.commit()


def _summary(clip):
    """The clip's stored measurements, parsed. A corrupt blob reads as an empty
    one — this pass must never be the reason a bank's quality scores disappear."""
    if not clip.metrics_json:
        return {}
    try:
        loaded = json.loads(clip.metrics_json)
    except (ValueError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
