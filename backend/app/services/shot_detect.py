"""The dedicated shot-boundary detector (TransNetV2) — parent half.

WHY THIS EXISTS. A cut is where a Bank clip STARTS and ENDS, and a naive
fixed-window cutter (every N seconds) routinely straddles a real cut, mixing
two shots into one training clip. TransNetV2 answers "is frame i a shot
boundary" directly, trained for exactly this, and beats PySceneDetect by a
wide margin on scraped content (measured 2026-08-03: 77.9 vs 55.75 F1 on
ClipShots) — the population this Bank was built for, not clean broadcast
footage.

THE CHILD (`backend/infer/shot_detect_infer.py`) RUNS IN A DEDICATED
"shot detect" interpreter (torch/transnetv2-pytorch are not in the Flask
venv), the same subprocess family as watermark_detect_infer.py and
bank_score_infer.py — this module never imports torch, only launches that
other interpreter as a subprocess and parses its JSON.

BOUNDARIES ARE PTS SECONDS, AND THAT IS THE CANONICAL FORM (see VideoClip's
own docstring on the video-bank branch for the full argument). TransNetV2
reasons in the indices of the frames it decoded; scraped video is routinely
variable-frame-rate, where index n names no stable instant while a
presentation timestamp does. `detect_shots()` converts using `fps_native` —
the caller's cached measurement if it has one, this module's OWN measurement
(from the exact decode pass that ran the detector) otherwise. The frame
indices travel alongside every clip, informative only: nothing here cuts from
them.

INDEPENDENCE FROM THE REST OF THE VIDEO LANE IS DELIBERATE. This module does
not import `video_probe`, `capabilities`, or the video-bank models — those
live on a sibling branch built in parallel. Every value they would otherwise
supply (the source's fps, whether the extra is even installed) is either an
explicit parameter here or left to the caller to resolve and hand in. Two
branches importing each other's work-in-progress is exactly how one half's
unfinished refactor breaks the other half's tests before either is done.

A MINIMUM SHOT LENGTH IS A FILTER, NOT A MERGE. A shot the detector drew at 2
frames is noise far more often than a real cut — a flash the network
mis-fired on, or a splice nobody would train on — but STITCHING it into a
neighbour would silently move that neighbour's boundary, a bigger and less
honest decision than simply not offering the sliver as a clip. Dropped shots
are gone from the returned list entirely; nothing else is renumbered around
them.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import threading

from .. import config as cfg
from . import shot_boundaries
from . import infer_env

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'shot_detect_infer.py')

# Persisted verbatim in VideoClip.detector. Duplicated in the child
# (shot_detect_infer.DETECTOR_ID) on purpose — the two run in different
# interpreters, and a single test asserts they never drift, the same
# safeguard the watermark detector's model ids already carry.
DETECTOR_ID = 'transnetv2'

# The child's per-video row shape and the names of its two probability heads.
# Duplicated from shot_detect_infer for the same reason DETECTOR_ID is — two
# interpreters, one contract — and pinned by a test that also checks the
# constant against what the child's `_detect_one` really returns, so a stub
# cannot stay green against a shape production has already left behind.
ROW_KEYS = ('path', 'state', 'shots', 'frame_count', 'fps_native', 'error',
            'probs')
PROBS_KEYS = ('single', 'all')

DEFAULT_THRESHOLD = 0.5
# Not a measured constant (unlike, say, the watermark detector's region-area
# cap) — no labelled sample exists for "how short is too short" here. 5
# frames rejects a single stray flash/glitch cut while keeping legitimate
# rapid cuts (compilations, montages) intact. A caller with a firmer opinion
# overrides it per call, or via `shot_detect.min_shot_frames` in config.
DEFAULT_MIN_SHOT_FRAMES = 5
DEFAULT_DEVICE = 'auto'


class ShotDetectUnavailable(RuntimeError):
    """The child could not start, or never got as far as loading TransNetV2 at
    all — an environment problem (missing extra, corrupt weights), not a bad
    file. The caller should surface an install hint, not mark one source as
    broken.

    DO NOT RENAME WITHOUT WARNING THE VIDEO-BANK BRANCH. Its shot-detect pass
    (branch feat/video-bank-service, commit ad83f255 at the time of writing)
    catches this by `type(exc).__name__` rather than `isinstance` — importing
    this module there to catch it properly would be circular, since that pass
    is the one calling `detect_shots()` here. A silent rename would not raise
    on that branch; it would just stop being recognised and start marking
    every source as an individual per-file failure again, which is the exact
    regression this exception exists to prevent (see the class docstring: a
    missing pip package must fail the PASS, not stamp `detect_state='error'`
    on hundreds of files that would have worked fine once it was installed)."""


class ShotDetectFileError(RuntimeError):
    """The model loaded fine but THIS file could not be processed — decode
    failure, or no frame rate available from either the caller or this
    module's own measurement to convert frame indices with. The caller should
    mark this one source failed and move on to the next, exactly what the
    child's own per-file contract promises."""


def detector_python() -> str:
    """The interpreter that runs the detector.

    Dedicated key first, then the bank-scoring environment — not a fallback
    but the COMMON case: that venv already carries torch, and building a
    second one would cost the user a second copy of it for no reason. Then
    the app's own Python, which simply probes unavailable on a normal
    install."""
    return (cfg.get('shot_detect.python')
            or cfg.get('bank_scoring.python')
            or sys.executable)


def threshold_default() -> float:
    """Clamped, never refused: read on a hot path mid-scan, a hand-edited
    config value must degrade to something usable rather than abort a pass
    already running."""
    try:
        value = float(cfg.get('shot_detect.threshold', DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(1.0, max(0.0, value))


def min_shot_frames_default() -> int:
    try:
        value = int(cfg.get('shot_detect.min_shot_frames', DEFAULT_MIN_SHOT_FRAMES))
    except (TypeError, ValueError):
        return DEFAULT_MIN_SHOT_FRAMES
    return max(1, value)


def legacy_min_shot_frames():
    """The RAW `shot_detect.min_shot_frames`, or None when the user never set it.

    Distinguishing "set to 5" from "unset" is the whole job of this function:
    an unset legacy key must not beat the new seconds-based floor, and a key
    somebody actually wrote must not silently change meaning under them. Never
    renamed — it is sitting in config files right now."""
    value = cfg.get('shot_detect.min_shot_frames', None)
    if value in (None, ''):
        return None
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return None


def min_shot_seconds_default():
    """The shortest shot worth offering, as a DURATION. See shot_boundaries for
    why 0.6 s and why not a frame count."""
    try:
        value = float(cfg.get('shot_detect.min_shot_seconds',
                              shot_boundaries.DEFAULT_MIN_SHOT_SECONDS))
    except (TypeError, ValueError):
        return shot_boundaries.DEFAULT_MIN_SHOT_SECONDS
    return max(0.0, value)


def short_shot_policy_default() -> str:
    """'drop' (what shipped) or 'merge'. An unrecognised value reads as 'drop':
    a typo in a config file must not invent a third behaviour."""
    value = str(cfg.get('shot_detect.short_shot_policy',
                        shot_boundaries.DEFAULT_SHORT_SHOT_POLICY) or '')
    return value if value in shot_boundaries.SHORT_SHOT_POLICIES \
        else shot_boundaries.DEFAULT_SHORT_SHOT_POLICY


def dissolve_min_frames_default() -> int:
    """Where 'cut' stops and 'dissolve' starts, in frames. Unmeasured on real
    footage — a setting on purpose, so a calibration is a number and not a
    patch (see shot_boundaries.label_transition)."""
    try:
        return max(1, int(cfg.get('shot_detect.dissolve_min_frames',
                                  shot_boundaries.DEFAULT_DISSOLVE_MIN_FRAMES)))
    except (TypeError, ValueError):
        return shot_boundaries.DEFAULT_DISSOLVE_MIN_FRAMES


def trim_dissolves_default() -> bool:
    """OFF by default, and it stays off until someone measures the width rule.
    Turning it on moves every dissolve boundary in the bank."""
    return bool(cfg.get('shot_detect.trim_dissolves', False))


def device_default() -> str:
    value = cfg.get('shot_detect.device', DEFAULT_DEVICE)
    return str(value) if value else DEFAULT_DEVICE


def _frame_to_seconds(frame_index, fps_native):
    return frame_index / fps_native


def _shots_to_clips(shots, fps_native, min_frames):
    """[start_frame, end_frame] index pairs (both inclusive — the detector's
    own convention) -> clip dicts in the canonical PTS-seconds form.

    `end_s` reads `end_frame + 1`, not `end_frame`: a shot's last frame is
    still on screen for its own display duration, so the shot's true end sits
    where the NEXT frame would begin. Using `end_frame` alone would
    under-count every single clip by one frame's worth of time.

    Shots shorter than `min_frames` are dropped, never merged — see the
    module docstring for why merging is the wrong call here.
    """
    clips = []
    for start_frame, end_frame in shots:
        if (end_frame - start_frame + 1) < min_frames:
            continue
        clips.append({
            'start_s': _frame_to_seconds(start_frame, fps_native),
            'end_s': _frame_to_seconds(end_frame + 1, fps_native),
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'detector': DETECTOR_ID,
        })
    return clips


def detect_shots(path, *, fps_native=None, threshold=None, min_shot_frames=None,
                  device=None, on_progress=None, cancel_file=None):
    """Detect shot boundaries in ONE video.

    Returns a list of clip dicts, in playback order:
        [{'start_s': float, 'end_s': float, 'start_frame': int,
          'end_frame': int, 'detector': 'transnetv2'}, ...]
    `start_s`/`end_s` are the canonical PTS-seconds bounds; `start_frame`/
    `end_frame` are always concrete ints (never None) — they only ever come
    from a shot the detector actually found on a successfully decoded file.

    `fps_native`, when given, OVERRIDES what this call would otherwise
    measure itself — pass the value already sitting on VideoSource to skip a
    second probe of the same file. Omit it and the conversion uses the fps
    this module measured on the SAME decode pass that ran the detector.

    Raises `ShotDetectUnavailable` when the extra is missing or the child
    never loaded a model at all. Raises `ShotDetectFileError` when the model
    loaded fine but THIS file could not be processed (corrupt video, or no
    frame rate available from either source to convert with). Both are
    RuntimeError subclasses, so a caller that wants to treat every failure
    the same way can catch RuntimeError once.
    """
    thr = threshold if threshold is not None else threshold_default()
    min_frames = (min_shot_frames if min_shot_frames is not None
                 else min_shot_frames_default())
    dev = device or device_default()

    raw = None
    for item in _run_worker([path], threshold=thr, device=dev,
                            cancel_file=cancel_file, on_progress=on_progress):
        raw = item
        break

    if raw is None:
        raise ShotDetectUnavailable(
            'the shot detector produced no result for this file')
    if raw.get('state') != 'ok':
        raise ShotDetectFileError(
            raw.get('error') or 'shot detection failed for this file')

    fps = fps_native or raw.get('fps_native')
    if not fps:
        raise ShotDetectFileError(
            'no frame rate available to convert frame indices to seconds')

    return _shots_to_clips(raw.get('shots') or [], fps, min_frames)


def _cut_options(fps_native, *, threshold=None, min_shot_frames=None,
                 min_shot_seconds=None, policy=None, dissolve_min_frames=None,
                 trim_dissolves=None):
    """Everything `shot_boundaries.build_clips` needs, resolved ONCE.

    Explicit argument beats bank/file override beats config default beats the
    module constant — the same ladder at every entry point, so a re-threshold
    from cache and a fresh pass cannot end up cutting to different rules."""
    legacy = legacy_min_shot_frames()
    if min_shot_frames is not None:
        min_frames = max(1, int(min_shot_frames))
    else:
        if min_shot_seconds is not None:
            seconds = min_shot_seconds
        elif cfg.get('shot_detect.min_shot_seconds', None) not in (None, ''):
            seconds = min_shot_seconds_default()
        elif legacy is not None:
            # Only the legacy key is set — leaving `seconds` None is what lets
            # `min_shot_frames_for` hand the user back the floor they wrote.
            seconds = None
        else:
            seconds = shot_boundaries.DEFAULT_MIN_SHOT_SECONDS
        min_frames = shot_boundaries.min_shot_frames_for(
            fps_native, min_seconds=seconds, min_frames=legacy)
    return {
        'threshold': (threshold if threshold is not None else threshold_default()),
        'min_frames': min_frames,
        'policy': policy or short_shot_policy_default(),
        'dissolve_min_frames': (dissolve_min_frames if dissolve_min_frames
                                is not None else dissolve_min_frames_default()),
        'trim_dissolves': (trim_dissolves if trim_dissolves is not None
                           else trim_dissolves_default()),
    }


def clips_from_probs(probs, *, fps_native, threshold=None, min_shot_frames=None,
                     min_shot_seconds=None, policy=None,
                     dissolve_min_frames=None, trim_dissolves=None):
    """Cut a CACHED probability vector into clips. No subprocess, no GPU, no
    decode — this is what makes a threshold change instant.

    `probs` is what services/shot_probs.load_probs returns. A vector with no
    all-frames head still cuts; it just carries no transition labels."""
    options = _cut_options(fps_native, threshold=threshold,
                           min_shot_frames=min_shot_frames,
                           min_shot_seconds=min_shot_seconds, policy=policy,
                           dissolve_min_frames=dissolve_min_frames,
                           trim_dissolves=trim_dissolves)
    return shot_boundaries.build_clips(
        (probs or {}).get('single') or [], (probs or {}).get('all'),
        fps=fps_native, **options)


def sweep_probs(probs, *, fps_native, thresholds, min_shot_frames=None,
                min_shot_seconds=None, policy=None):
    """"At threshold X you would get N shots" for several X, from the cache."""
    options = _cut_options(fps_native, min_shot_frames=min_shot_frames,
                           min_shot_seconds=min_shot_seconds, policy=policy)
    return shot_boundaries.sweep((probs or {}).get('single') or [], thresholds,
                                 fps=fps_native, min_frames=options['min_frames'],
                                 policy=options['policy'])


def detect_source(path, *, fps_native=None, threshold=None, min_shot_frames=None,
                  min_shot_seconds=None, policy=None, dissolve_min_frames=None,
                  trim_dissolves=None, device=None, on_progress=None,
                  cancel_file=None, want_probs=True):
    """Run the detector on ONE video and bring back everything it produced.

    Returns {'clips': [...], 'probs': {...}|None, 'fps_native': float,
             'frame_count': int|None}.

    `detect_shots` is kept as the thin "just the clips" wrapper it always was;
    this is where the probability cache and the transition labels come from.

    WHEN THE WORKER RETURNS A VECTOR, THE CLIPS ARE REBUILT FROM IT — the
    worker's own `shots` list is ignored. Two faithful ports of one rule exist
    (one per interpreter) and a test compares them frame by frame, but letting
    each feed a different half of the app would mean a re-threshold at the SAME
    value could disagree with the pass that filled the cache. One producer of
    clip bounds, always the same one.

    Raises the same two exceptions `detect_shots` does, for the same reasons.
    """
    raw = None
    thr = threshold if threshold is not None else threshold_default()
    for item in _run_worker([path], threshold=thr, device=device or device_default(),
                            cancel_file=cancel_file, on_progress=on_progress,
                            emit_probs=bool(want_probs)):
        raw = item
        break

    if raw is None:
        raise ShotDetectUnavailable(
            'the shot detector produced no result for this file')
    if raw.get('state') != 'ok':
        raise ShotDetectFileError(
            raw.get('error') or 'shot detection failed for this file')

    fps = fps_native or raw.get('fps_native')
    if not fps:
        raise ShotDetectFileError(
            'no frame rate available to convert frame indices to seconds')

    probs = raw.get('probs') or None
    if probs and probs.get('single'):
        clips = clips_from_probs(probs, fps_native=fps, threshold=thr,
                                 min_shot_frames=min_shot_frames,
                                 min_shot_seconds=min_shot_seconds,
                                 policy=policy,
                                 dissolve_min_frames=dissolve_min_frames,
                                 trim_dissolves=trim_dissolves)
    else:
        # No vector: an older worker, or a caller that did not want one. The
        # boundaries the child drew are all there is, and they are still right.
        options = _cut_options(fps, threshold=thr, min_shot_frames=min_shot_frames,
                               min_shot_seconds=min_shot_seconds, policy=policy)
        clips = _shots_to_clips(raw.get('shots') or [], fps, options['min_frames'])
        probs = None
    return {'clips': clips, 'probs': probs, 'fps_native': float(fps),
            'frame_count': raw.get('frame_count')}


def _run_worker(paths, *, threshold, device, cancel_file, on_progress,
                emit_probs=False):
    """Launch the child, stream back its per-video rows in input order.

    Raises ShotDetectUnavailable if the child never produced anything because
    the model failed to load — the one failure a caller must be able to tell
    apart from "ran fine, found nothing", because they lead to opposite
    decisions.
    """
    job = {
        'videos': list(paths),
        'threshold': threshold,
        'device': device,
        'cancel_file': cancel_file or '',
        'emit_probs': bool(emit_probs),
    }
    python = detector_python()
    env = infer_env.worker_env(python, PYTHONUTF8='1',
                           PYTHONIOENCODING='utf-8')
    try:
        proc = subprocess.Popen(
            infer_env.worker_argv(python, _SCRIPT),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as e:
        raise ShotDetectUnavailable(f'could not start the shot detector: {e}') from e

    tail = []

    def drain_stderr():
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                tail.append(line)
                del tail[:-20]
                if on_progress is not None:
                    try:
                        on_progress(line)
                    except Exception:    # noqa: BLE001 — a UI callback never sinks the pass
                        pass
    watcher = threading.Thread(target=drain_stderr, daemon=True)
    watcher.start()
    try:
        proc.stdin.write(json.dumps(job))
        proc.stdin.close()
    except OSError as e:
        proc.kill()
        raise ShotDetectUnavailable(
            f'the shot detector closed immediately: {e}') from e

    index = 0
    summary = None
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue          # a stray print is noise, never a failed pass
            if 'summary' in payload:
                summary = payload['summary'] or {}
                continue
            # The child answers in input order, but we key on the path it
            # echoes rather than our own counter: a desynchronised index would
            # attach one video's verdict to a DIFFERENT video's database row,
            # the single worst thing this module could do.
            path = payload.get('path')
            if path != (paths[index] if index < len(paths) else None):
                logger.warning('shot detector: out-of-order result for %r, '
                               'dropping it', path)
                continue
            index += 1
            yield payload
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        watcher.join(timeout=2)
    if summary is not None and not summary.get('ok') and index == 0:
        raise ShotDetectUnavailable(summary.get('error')
                                    or 'the shot detector could not load its model')
    if summary is None and index == 0:
        raise ShotDetectUnavailable(
            _last_line(tail) or 'the shot detector produced no output')


def _last_line(tail) -> str:
    return next((ln for ln in reversed(tail) if ln.strip()), '')
