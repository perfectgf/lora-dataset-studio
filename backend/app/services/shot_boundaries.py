"""Probabilities -> shots. The pure half of shot detection, with no model in it.

WHY THIS MODULE EXISTS AT ALL. TransNetV2 does not answer "where are the cuts".
It answers, for every frame, "how likely is a transition here" — and the cut
list is a THRESHOLD applied to that vector afterwards. Until now the vector was
computed inside the worker, thresholded there, and thrown away with the worker's
process, which made 0.5 an irreversible decision: disagreeing with it cost a
full GPU pass over the file. The vector is now persisted (services/shot_probs),
and everything downstream of it lives here, in a module that imports nothing but
the standard library. That is what makes a threshold sweep instant, offline, and
testable on an install with none of the video extras.

THE 0.5 DEFAULT IS A CONVENTION, NOT A MEASUREMENT. Souček & Lokoč's paper uses
it without justifying it anywhere, and no published curve exists for amateur
scraped footage — the population this bank was built for. So it stays the
default (changing it blind would be trading one unmeasured number for another),
and it becomes free to override per bank and per file, which is the honest
answer to "no single threshold is right for a mixed corpus".

TWO HEADS, AND THE SECOND ONE USED TO BE FREE AND WASTED. The network predicts
`single_frame_pred` (the centre frame of a transition — the head every published
F1 is measured on) and `all_frame_pred` (every frame the transition covers). The
worker computed both and dropped the second. Its width is the only signal
available anywhere that says whether a boundary is a hard cut or a dissolve, and
`label_transition` below turns it into exactly that.

Everything here works in FRAME INDICES and converts to seconds only at the very
end, through the caller's `fps`. Same reason the worker does: scraped video is
routinely variable-frame-rate, and the detector never saw a timestamp.
"""
from __future__ import annotations

DETECTOR_ID = 'transnetv2'

DEFAULT_THRESHOLD = 0.5

# 0.08 s at 60 fps — which is what `min_shot_frames=5` actually meant, and
# nobody chose that. The ecosystem's measured floor is 0.6 s (PySceneDetect's
# benchmark sweep, the only public one), expressed in SECONDS because a frame
# count means a different duration on every file in a mixed corpus.
DEFAULT_MIN_SHOT_SECONDS = 0.6
# The legacy key's default, kept verbatim so a config that never mentions
# either setting behaves the way it did before this module existed only where
# the legacy key is what the user actually wrote. See `min_shot_frames_for`.
LEGACY_MIN_SHOT_FRAMES = 5

SHORT_SHOT_POLICIES = ('drop', 'merge')
# 'drop' is the default because it is what shipped: a bank re-detected after an
# update must not silently gain different bounds than the one next to it.
DEFAULT_SHORT_SHOT_POLICY = 'drop'

# A frame belongs to a transition, for WIDTH purposes, when the all-frames head
# is above this. Deliberately low: the head's job is to cover the transition,
# and the tails of a long dissolve are exactly the frames a mid-height bar would
# cut off — which would make every dissolve read as a cut.
DISSOLVE_FLOOR = 0.1
# At or above this many frames a transition reads as a dissolve rather than a
# cut. A HYPOTHESIS, not a measurement: it follows from how the second head was
# trained (35 % hard cuts, 50 % dissolves) and it has never been checked against
# labelled amateur footage. It is a parameter of every function that uses it,
# and a config key, precisely so the day someone measures it on a real corpus
# the answer is a number in a settings file rather than a patch.
DEFAULT_DISSOLVE_MIN_FRAMES = 5


def predictions_to_scenes(probs, threshold=DEFAULT_THRESHOLD):
    """Per-frame transition probabilities -> [start, end] frame pairs, inclusive.

    A faithful port of TransNetV2's own `predictions_to_scenes`, in plain
    Python. The worker carries an identical copy (it must run in an interpreter
    that has no app package); a test compares the two vector by vector, because
    two copies of one rule drifting is exactly how a re-threshold would start
    contradicting the detection that filled the cache.

    `>` and not `>=`, matching upstream: a probability sitting exactly on the
    bar is not a transition.
    """
    if not probs:
        return []
    flags = [1 if p > threshold else 0 for p in probs]
    scenes = []
    t = -1
    t_prev = 0
    start = 0
    for i, t in enumerate(flags):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if not scenes:
        # Every frame cleared the bar. Upstream's own fallback is "the whole
        # file is one shot" rather than reporting nothing for a file that
        # unambiguously has content.
        return [[0, len(flags) - 1]]
    return scenes


def resolve_threshold(file_value, bank_value, global_value):
    """The threshold that applies to ONE file: its own, else its bank's, else
    the global default.

    Clamped, never refused. This is read mid-pass and mid-request; a nonsense
    value in a database row or a hand-edited config must degrade to something
    usable rather than abort work that is already running.
    """
    for value in (file_value, bank_value):
        clamped = _clamp01(value)
        if clamped is not None:
            return clamped
    return _clamp01(global_value) if _clamp01(global_value) is not None \
        else DEFAULT_THRESHOLD


def _clamp01(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:                       # NaN passes every comparison
        return None
    return min(1.0, max(0.0, number))


def min_shot_frames_for(fps, *, min_seconds=None, min_frames=None):
    """How many frames the shortest acceptable shot holds ON THIS FILE.

    `min_seconds` is the setting people should use; `min_frames` is the legacy
    `shot_detect.min_shot_frames`, which is sitting in config files today and is
    therefore never renamed away. When only the legacy key is set it WINS — a
    user who wrote 5 there gets 5, not a silently different floor after an
    update. When both are set, seconds win: setting the new key is the explicit
    act of moving on.
    """
    if min_seconds is None and min_frames is not None:
        try:
            return max(1, int(min_frames))
        except (TypeError, ValueError):
            return LEGACY_MIN_SHOT_FRAMES
    seconds = DEFAULT_MIN_SHOT_SECONDS if min_seconds is None else min_seconds
    try:
        seconds = float(seconds)
        rate = float(fps)
    except (TypeError, ValueError):
        return LEGACY_MIN_SHOT_FRAMES
    if rate <= 0 or seconds <= 0:
        return 1
    return max(1, int(round(seconds * rate)))


def apply_min_length(scenes, min_frames, policy=DEFAULT_SHORT_SHOT_POLICY):
    """Enforce the minimum shot length, by DROPPING slivers or by MERGING them.

    'drop' is what shipped: a two-frame shot is a mis-fire far more often than a
    real cut, and offering it as a clip is offering something untrainable.

    'merge' glues the sliver onto its previous neighbour (onto the NEXT one when
    it is the file's first shot, there being no previous). The distinction
    matters and no tool in this space makes it: dropping loses the footage,
    merging keeps it.

    WHY MERGING DOES NOT MOVE ANY EXISTING BOUNDARY. A merge deletes exactly one
    boundary — the one that produced the sliver — and every surviving cut stays
    on the frame the detector put it on. The result is always a strict SUBSET of
    the boundaries the detector drew, never a shifted version of them, so a
    merged bank and a dropped bank disagree about which cuts exist and never
    about where a cut is. Anything else (splitting the sliver between its two
    neighbours, say) would invent a boundary no head ever predicted.
    """
    floor = max(1, int(min_frames or 1))
    merging = policy == 'merge'
    out = []
    pending = None
    for scene in scenes or []:
        current = [pending[0], scene[1]] if pending else [scene[0], scene[1]]
        pending = None
        if (current[1] - current[0] + 1) >= floor:
            out.append(current)
        elif not merging:
            continue
        elif out:
            out[-1][1] = current[1]
        else:
            pending = current
    if pending:
        # A whole file shorter than the floor, under 'merge': there is nothing
        # to glue it to, and one short clip is a better answer than none.
        out.append(pending)
    return out


def transition_width(all_probs, frame_index, floor=DISSOLVE_FLOOR):
    """How many frames the transition around `frame_index` covers, per the
    all-frames head. 0 when that head says nothing there — or is not there.

    A cache written before this head was persisted is a legitimate state, and
    the honest answer for it is "no label", never a guessed one.
    """
    if not all_probs:
        return 0
    index = int(frame_index)
    if index < 0 or index >= len(all_probs):
        return 0
    if not all_probs[index] > floor:
        return 0
    left = index
    while left > 0 and all_probs[left - 1] > floor:
        left -= 1
    right = index
    last = len(all_probs) - 1
    while right < last and all_probs[right + 1] > floor:
        right += 1
    return right - left + 1


def label_transition(width, dissolve_min_frames=DEFAULT_DISSOLVE_MIN_FRAMES):
    """'cut' for a narrow transition, 'dissolve' for a wide one.

    UNMEASURED ON THIS CORPUS — see DEFAULT_DISSOLVE_MIN_FRAMES. The chip this
    feeds is advisory, like every other flag in the bank, and the frontier is a
    parameter so a calibration file can move it without touching this code.
    """
    try:
        frontier = max(1, int(dissolve_min_frames))
    except (TypeError, ValueError):
        frontier = DEFAULT_DISSOLVE_MIN_FRAMES
    return 'dissolve' if int(width) >= frontier else 'cut'


def _boundary(all_probs, centre, dissolve_min_frames):
    width = transition_width(all_probs, centre)
    if width <= 0:
        return None
    return {'kind': label_transition(width, dissolve_min_frames), 'width': width}


def build_clips(single_probs, all_probs=None, *, fps, threshold=DEFAULT_THRESHOLD,
                min_frames=1, policy=DEFAULT_SHORT_SHOT_POLICY,
                dissolve_min_frames=DEFAULT_DISSOLVE_MIN_FRAMES,
                trim_dissolves=False):
    """The whole rule, end to end: probabilities in, clip dicts out.

    Rows carry the same keys ``video_bank_service._insert_clips`` already reads
    (start_s/end_s/start_frame/end_frame/detector) plus ``transition``, the
    per-edge label.

    ``end_s`` reads ``end_frame + 1``: a shot's last frame is on screen for its
    own display duration, so the shot ends where the NEXT frame would begin.
    Using ``end_frame`` alone under-counts every clip by one frame.

    A file's own first and last edges are never labelled — nothing dissolved
    into the start of a file. A chip on every clip would say nothing.
    """
    scenes = apply_min_length(
        predictions_to_scenes(single_probs, threshold), min_frames, policy)
    if not scenes:
        return []
    last_frame = len(single_probs) - 1
    floor = max(1, int(min_frames or 1))
    clips = []
    for scene in scenes:
        start_frame, end_frame = int(scene[0]), int(scene[1])
        head = _boundary(all_probs, start_frame - 1, dissolve_min_frames) \
            if start_frame > 0 else None
        tail = _boundary(all_probs, end_frame, dissolve_min_frames) \
            if end_frame < last_frame else None
        if trim_dissolves:
            start_frame, end_frame = _trimmed(start_frame, end_frame, head, tail,
                                              floor)
        clips.append({
            'start_s': start_frame / fps,
            'end_s': (end_frame + 1) / fps,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'detector': DETECTOR_ID,
            'transition': {'start': head, 'end': tail},
        })
    return clips


def _trimmed(start_frame, end_frame, head, tail, floor):
    """Pull both bounds inside the dissolves that surround the shot.

    TransNetV2 places a boundary in the MIDDLE of a gradual transition
    (published Transition IoU 0.193), so half the dissolve lands inside each
    neighbour — a clip whose first frames are a cross-fade of the previous shot.
    Half the plateau on each side removes exactly that, and only for edges
    labelled 'dissolve': a hard cut has nothing to remove and moving it would be
    inventing a boundary.

    A trim that would leave less than the minimum shot length is DISCARDED
    whole, for that clip. A shot that survived the floor must not be trimmed out
    of existence — two long dissolves around a short shot would otherwise cross
    over and produce an inverted span.
    """
    start = start_frame + (head['width'] // 2 if head and head['kind'] == 'dissolve' else 0)
    end = end_frame - (tail['width'] // 2 if tail and tail['kind'] == 'dissolve' else 0)
    if (end - start + 1) < floor:
        return start_frame, end_frame
    return start, end


def sweep(single_probs, thresholds, *, fps, min_frames=1,
          policy=DEFAULT_SHORT_SHOT_POLICY):
    """"At threshold X you would get N shots", for several X, from one cache.

    Counts the clips that would SURVIVE, floor included — not the boundaries
    found. A preview that counted boundaries would promise shots the pass then
    drops, which is the exact complaint the metrics dry-run exists to answer.
    """
    rows = []
    for value in thresholds or []:
        thr = _clamp01(value)
        if thr is None:
            continue
        clips = build_clips(single_probs, None, fps=fps, threshold=thr,
                            min_frames=min_frames, policy=policy)
        rows.append({'threshold': thr, 'shots': len(clips)})
    return rows


def suggested_thresholds(current):
    """The ladder the dry-run offers: a fixed spread, plus whatever is in force.

    Fixed rather than derived from the vector, so the same file answers the same
    question tomorrow and two files can be compared against each other."""
    values = {round(v, 2) for v in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    resolved = _clamp01(current)
    if resolved is not None:
        values.add(round(resolved, 2))
    return sorted(values)
