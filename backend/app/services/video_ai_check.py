"""🤖 Does this shot MOVE like the world does? — the may-be-AI-generated flag.

WHY THIS EXISTS, and why now. The video bank fills by scraping, and a scraped
corpus in 2026 is not all camera footage any more. That matters for one measured
reason rather than for taste: Wan's curation work reports that even a small
minority of synthetic material — under a tenth of a corpus — measurably degrades
what a model trained on it learns. A generated clip is not a defect the eye
catches at thumbnail size, and none of the six passes before this one can see it:
🩻 Defects reads what a re-encode left behind, 🔳 Safe zone reads the container,
🎨 Look rates taste. A clean, well-lit, well-framed generated clip passes all
three.

WHAT IT MEASURES. One number per shot, `motion_irregularity`: how erratic the
motion is over a two-second window. Frames are encoded independently, the
distance between consecutive frame vectors is how much the picture changed, and
the statistic is the SPREAD of the change in that rate — the dispersion of
visual "acceleration". The arithmetic and its provenance are in
infer/video_ai_check_infer.py; this module owns the window, the aggregation, the
verdict and the honesty.

⚠️ THE POLARITY IS THE OPPOSITE OF THE INTUITIVE ONE, and getting it backwards
would invert the whole feature. A LOW score is the suspicious one: real footage
is full of small irregularities (a hand shakes, a subject accelerates, light
flickers, the sensor is noisy) and generated footage tends to be smoother than
the world. So the cut is a FLOOR — `motion_irregularity_floor` — and raising it
flags more shots, exactly like `sharpness_floor` and `aesthetic_floor` next to
it. The reference implementation makes the same claim in the only way that
cannot be misread: its evaluation ranks the RAW score with the REAL videos as
the positive class.

HOW GOOD IS IT, HONESTLY — this is shipped in the hint, not only here. The SAFE
Challenge (arXiv:2605.06912) evaluated detectors BLIND, on material the entrants
had not seen: the best system scored 0.86 balanced accuracy on untouched video
and 0.74 once it had been post-processed, and re-compression alone moved AUC from
0.88 to 0.77. A scraped bank is re-encoded by construction, so the honest order
of magnitude for this flag on this app's material is around 0.75 — three
in four — and not the ninety-somethings a detector's own paper reports on its own
benchmark. Two further limits, both real:

  * D3 was measured against the generators of 2023-24 (ModelScope, Gen2, Pika,
    LaVie, Sora, CogVideoX, OpenSora and a dozen more). Its entire thesis is
    that *current* generators cannot render second-order motion. Nothing in that
    evaluation says anything about Sora 2, Veo 3, Kling or Wan 2.5, and the
    claim is exactly the kind that decays.
  * It has a KNOWN inversion. On T2VZ, a generator whose output is incoherent
    and flickery, the reference scores 45.11 AP — BELOW chance. Chaotic
    generation reads as more real than clean generation. Cheap, glitchy or
    heavily stylised synthetic material is therefore the case this flag is worst
    at, and a hard cut inside the window has the same effect.

So it is ADVISORY and it says "may be". It flags, it never rejects, it is in no
default, and nothing in the app deletes or refuses a shot because of it.

WHY IT IS ITS OWN BUTTON. The lane already has the two shapes and this is
squarely the second. 🎨 Look rides 🔎 Find scenes because it costs no decode at
all — it reads vectors already on disk. This pass cannot ride anything, because
nothing in the app decodes what it needs: it wants SIXTEEN CONTIGUOUS frames at
8 fps in RGB at 224 px, and the four decodes that exist sample nothing like it —
🔎 embeds three frames chosen by position and sharpness across the whole shot at
256 px, 🗣 describes eight spread across the span at 448 px, 🔳 measures three at
768 px, and the quality scan reads the whole clip in GREYSCALE at 160 px wide.
A contiguous window is the entire premise of a temporal statistic; a spread
sample would measure the cuts between instants, not the motion inside one.

That leaves 🔳 Safe zone's rule, which applies unchanged: a pass that CONSUMES
nothing has no order to protect, so it earns its own button rather than a place
in a queue. It needs a shot's bounds and its source file, both of which exist the
moment detection has run.

UNMEASURED IS A STATE, NEVER A ZERO. A shot shorter than the window carries
`ai_check_state: 'too_short'` and no score — a zero would be the strongest
possible claim that a shot is synthetic. A shot whose frames could not be decoded
is 'unreadable'. Neither can ever be flagged, because `verdicts()` flags nothing
it has no measurement for.

ONE WINDOW PER SHOT, and the same window every time — see `window_times`. The
score's scale depends on how many samples the standard deviation is taken over,
so two shots measured over different frame counts are not on one scale and a
single cut across them would be a silent lie. Sixteen frames, or no measurement.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile

from .. import config as cfg
from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'video_ai_check_infer.py')

# The encoder. A CONSTANT and not a setting, unlike `video_caption.model`: that
# one is a choice between checkpoints that all do the same job differently, while
# this score has no absolute meaning and no calibration — swapping the encoder
# moves the scale under every number already stored and under any cut a user set
# against it. Changing this is a migration, not a preference.
MODEL_ID = 'microsoft/xclip-base-patch16'

# The window, and every constant in it comes from the reference implementation
# (see the infer script's docstring for the citation).
#
# FRAMES is 16 and is not free to move. The encoder passes messages between
# frames in groups of `config.num_frames` (8 for this checkpoint), so the count
# must be a multiple of 8 — and the standard deviation is taken over FRAMES-2
# samples, so changing it re-scales every score in every bank.
FRAMES = 16
SAMPLE_FPS = 8.0

# Distance from a shot boundary the window keeps, in seconds. The lane's own
# number (video_caption.EDGE_MARGIN_S) and the lane's own reason: a boundary is
# where a dissolve lives. Here it matters more than anywhere else, because a
# dissolve is precisely the kind of violently irregular motion that would read as
# "real" — it would push the score UP, hiding a generated clip rather than
# inventing one, which is the safer error and still an error.
EDGE_MARGIN_S = 0.25

# Long side handed to the encoder, and the fraction cropped off the LONGER edge
# before the resize. Both are the reference's, including the resize being
# non-aspect-preserving.
FRAME_SIDE = 224
CROP_FRACTION = 0.1

# The keys this pass owns in the shared metrics blob. Named once, and pinned
# against video_metrics.ADVISORY_KEYS by a test: a key this writes and that list
# does not carry is erased by the next quality scan, silently, sending a whole
# bank back through a pass it has already paid for.
STATE_KEY = 'ai_check_state'
FRAMES_KEY = 'ai_check_frames'
SCORE_KEY = 'motion_irregularity'
OWNED_KEYS = (STATE_KEY, FRAMES_KEY, SCORE_KEY)

# Shots per child invocation. Measured on this machine, on CPU: the child costs
# 5.6 s to reach its first answer (cold torch import, encoder load and one clip)
# and 0.71 s per clip after that, while the decode and preprocessing this side
# cost 0.12 s per clip. So the chunk is what amortises the fixed part: at 100
# shots it is under 7 % of the chunk, and there is nothing left to win by going
# higher. Bounded because this chunk's JPEGs sit on disk at once — 100 x 16
# frames at 224 px is ~20 MB, a scratch file rather than a second copy of the
# footage — and because the chunk is also the commit unit, so a pass stopped or
# killed keeps every shot it already measured.
#
# The whole pass therefore costs about 0.83 s per shot: a 3 000-shot bank is
# roughly forty minutes. On the CPU, always — see `score_chunk`.
CHUNK = 100

# Budget for one child: a per-shot allowance plus a floor that covers the cold
# torch import and a first-run weight download on a machine whose antivirus is
# scanning both. Same shape as the safe zone's, and for the same reason — a flat
# forfeit sized for a small chunk turns a slow machine's whole run into nothing,
# because a timeout returns no partial result.
_TIMEOUT_PER_CLIP_S = 6
_TIMEOUT_FLOOR_S = 900


def unavailable_reason():
    """None when this install can run the check, else the sentence saying why.

    The ✨ Score interpreter's own probe, deliberately. This pass runs torch and
    transformers, which is exactly what that environment is provisioned for and
    what 🎨 Look already borrows — inventing a second sentence for the same
    missing install is how a user installs twice."""
    from .clip_image_encoder import unavailable_reason as encoder_reason
    reason = encoder_reason()
    if reason is None:
        return None
    return reason.replace('frame embedding', 'the AI check')


def model_download_notice():
    """A sentence for the JOB LINE when the encoder is not on this machine yet,
    else ''. The download itself is allowed and happens on first use; doing it in
    SILENCE is not, because a pass sitting at 0/900 while several hundred
    megabytes cross someone's connection is indistinguishable from a hang."""
    from .video_caption import model_is_cached
    if model_is_cached(MODEL_ID):
        return ''
    return (f'{MODEL_ID} is not on this machine yet — the first run downloads it '
            f'before checking anything. Leave it running.')


# --- the window ------------------------------------------------------------------

def window_span_s():
    """How long the sampled window lasts. FRAMES instants at SAMPLE_FPS span
    FRAMES-1 intervals, not FRAMES — the fencepost that would make the minimum
    shot length wrong by an eighth of a second."""
    return (FRAMES - 1) / SAMPLE_FPS


def min_duration_s():
    """The shortest shot this pass can measure: the window plus a margin at BOTH
    ends. Shots below it carry 'too_short' and no score."""
    return window_span_s() + 2 * EDGE_MARGIN_S


def window_times(start_s, end_s):
    """The FRAMES timestamps to decode, ascending — or [] when the shot is too
    short to hold the window with a margin at each end.

    CENTRED in the shot rather than anchored at its start, which is the one
    choice here that is not the reference's. It samples the middle of a shot, at
    a maximum distance from both cuts, and it makes the margin symmetric for
    free: a head-anchored window only ever protects one boundary, and on a shot
    barely longer than the window it would run straight into the other.

    Contiguous at 8 fps, never spread across the span. Every other pass in this
    lane spreads its samples, and every one of them is answering a question about
    the shot as a whole; this one measures how motion evolves BETWEEN adjacent
    instants, and a spread sample would measure the jumps between moments rather
    than the movement inside one."""
    duration = float(end_s) - float(start_s)
    span = window_span_s()
    if duration < span + 2 * EDGE_MARGIN_S:
        return []
    first = float(start_s) + (duration - span) / 2.0
    return [first + i / SAMPLE_FPS for i in range(FRAMES)]


def crop_box(width, height, fraction=CROP_FRACTION):
    """(left, top, right, bottom) — the reference's centre crop, reproduced.

    It cuts `fraction` off EACH END of the longer edge and leaves the shorter one
    whole, so a 16:9 frame keeps its full height and 80 % of its width. Note that
    the paper describes this as cropping 10 % of the longer edge while the code
    removes 20 % of it; the CODE is what produced the published numbers, so the
    code is what this reproduces.

    A degenerate size (a 1-pixel frame, a zero) returns the whole frame rather
    than an empty box: an empty crop would raise inside the decode seam and
    retire a shot as unreadable for a reason that has nothing to do with it."""
    w = max(int(width or 0), 0)
    h = max(int(height or 0), 0)
    if w < 3 or h < 3:
        return (0, 0, w, h)
    cut = int((w if w >= h else h) * float(fraction))
    if 2 * cut >= (w if w >= h else h):
        return (0, 0, w, h)
    if w >= h:
        return (cut, 0, w - cut, h)
    return (0, cut, w, h - cut)


# --- the aggregation --------------------------------------------------------------

def irregularity(steps):
    """ONE number per shot from the per-adjacent-pair distances the child
    returned — the sample standard deviation of their successive differences.

    `steps[k]` is how much the picture changed between frame k and frame k+1.
    Their differences are how much that RATE changed, and the spread of those is
    the statistic: a subject moving at a perfectly constant speed produces
    constant steps, zero differences and a score of 0, while anything that
    accelerates unevenly — which is everything real — produces a positive one.

    BESSEL-CORRECTED (n-1), matching the reference's `torch.std` default and its
    Formula (8). With 14 samples the difference against the population form is
    about 4 % — small, and this app has no calibration of its own to absorb it,
    so it matches rather than improves.

    NOT normalised by the size of the steps, and that was measured rather than
    assumed. A coefficient of variation — this same spread divided by the mean
    step — is the obvious-looking improvement, because an absolute standard
    deviation ought to grow with how much a shot moves. Measured on ten forged
    clips (five constant-velocity pans against five of the same scene at the same
    mean speed with per-frame jitter, sensor noise and exposure flicker), encoded
    to H.264 and read back through the real pass:

        raw std (this)      AUC 0.840   smooth 0.657 / handheld 0.866   21/25
        std / mean step     AUC 0.800   smooth 0.214 / handheld 0.284   20/25

    So the normalisation is very nearly a constant divide and it loses a little.
    No gain, and it would take the port off the only variant anyone has measured
    against real generators. So the port stays a port.

    ⚠️ Any re-measurement here must first confirm the encoder's weights actually
    LOADED — see the infer script. A randomly initialised XCLIP separates these
    same two classes well enough to look like a working detector, so a number
    produced without that check says nothing about this decision.

    None for a series too short to have two differences, never 0.0. The caller
    stores a score only for a FULL window (see run_ai_check): a standard
    deviation over fewer samples is a different estimator on a different scale,
    and one cut spanning two scales would flag shots for their length."""
    values = [float(s) for s in (steps or []) if s is not None]
    if len(values) < 3:
        return None
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return variance ** 0.5


# --- the pass ---------------------------------------------------------------------

def pending_clips(bank_id, rescan=False):
    """The shots this pass would check, oldest first.

    Only shots of a source that PROBED: an unreadable file has no frame to
    decode, and counting it as unreadable on every run would make the pass look
    permanently broken. Deliberately NOT gated on `embed_state` the way 🎨 Look
    is — this reads pixels, not vectors, so requiring an embedding run first
    would be an invented dependency.

    "Already checked" is a key in the blob rather than a column, like every other
    advisory pass here: the resume test is a JSON read, and legacy databases need
    no migration to carry it."""
    rows = (VideoClip.query.filter_by(bank_id=bank_id)
            .join(VideoSource, VideoSource.id == VideoClip.source_id)
            .filter(VideoSource.probe_state == 'ok')
            .order_by(VideoClip.id.asc()).all())
    if rescan:
        return rows
    return [c for c in rows if STATE_KEY not in _summary(c)]


def _write_window(src_path, times, dest_dir, stem):
    """[jpeg path] for ONE shot's window, in decode order — the PyAV seam.

    Monkeypatched in tests so the suite runs with no video extra. Raises on a
    segment that cannot be decoded; the caller retires THAT shot and moves on.

    ONE seek and then a forward decode, unlike video_clip_search._write_frames
    which seeks per target. That one's three timestamps are seconds apart in a
    multi-gigabyte rush, so a seek each is the cheap way round; these sixteen are
    an eighth of a second apart and sit inside the same run of frames, so seeking
    between them would re-decode the same GOP sixteen times.

    Geometry happens HERE rather than in the child: a centre crop and a resize
    are pure arithmetic (`crop_box`), and a test can check them without torch,
    without transformers and without a model download."""
    import av
    from PIL import Image

    os.makedirs(dest_dir, exist_ok=True)
    out = []
    remaining = list(times)
    with av.open(str(src_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        try:
            container.seek(int(remaining[0] / (stream.time_base or 1)),
                           stream=stream)
        except Exception:            # noqa: BLE001 — some streams refuse to
            pass                     # seek; decoding from 0 still works
        for frame in container.decode(stream):
            if not remaining:
                break
            ts = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            if ts < remaining[0]:
                continue
            # The nearest frame AT OR AFTER each target, and a single decoded
            # frame may answer several targets on footage whose rate is below
            # SAMPLE_FPS. Consuming them in a loop rather than one per frame is
            # what stops an 8 fps source from silently producing a 9-frame
            # window that the caller would then have to reject.
            while remaining and ts >= remaining[0]:
                remaining.pop(0)
                path = os.path.join(dest_dir, f'{stem}_{len(out):02d}.jpg')
                img = frame.to_image().convert('RGB')
                img = img.crop(crop_box(img.width, img.height))
                # BILINEAR, not LANCZOS: the reference resizes through
                # albumentations, whose default is bilinear. The whole
                # preprocessing chain is reproduced rather than improved — see
                # the infer script's docstring.
                img = img.resize((FRAME_SIDE, FRAME_SIDE), Image.BILINEAR)
                # q95 rather than the lane's usual 88. The reference JPEG-encodes
                # at full size and downscales afterwards, which attenuates the
                # artefacts; encoding after the resize does not, so the quality
                # goes up to land in the same place. XCLIP's measured tolerance
                # runs the right way here (98.46 mAP at q100, 97.11 at q90).
                img.save(path, 'JPEG', quality=95)
                out.append(path)
    if len(out) < len(times):
        raise RuntimeError(f'only {len(out)} of {len(times)} frames decoded')
    return out


def score_chunk(payload, *, timeout=None):
    """{clip_id: [step distance, ...]} for a chunk — the MODEL seam.

    `payload` is [{'id': int, 'frames': [path, ...]}]. One subprocess in the
    ✨ Score interpreter, monkeypatched in tests so nothing here ever imports
    torch or downloads weights. Raises RuntimeError carrying the child's own
    words; the caller turns that into a result rather than a 500.

    CPU, ALWAYS, and it is a design decision rather than an omission. The card
    would help — the reference measures 0.056 s per clip on a 4090 against the
    0.71 s measured here — but this pass runs for tens of minutes over a bank,
    and taking the GPU-exclusive window for that long would unload ComfyUI and
    block a training start for the whole run, over an advisory flag. The
    opposite is the useful property: because it never touches the card, a bank
    can be checked WHILE a training owns it. Same trade, and the same sentence,
    as 🔳 Safe zone's. Setup also installs CPU torch into this interpreter, so on
    most installs there is no card here to take in the first place."""
    if not payload:
        return {}
    python = cfg.get('bank_scoring.python') or sys.executable
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'
    # Belt and braces with the child, which hides CUDA again before it imports
    # torch: a 16-frame base-ViT forward is not worth taking a card off a
    # training run for, and the parent takes no GPU window on its behalf.
    env['CUDA_VISIBLE_DEVICES'] = ''
    budget = timeout if timeout is not None else max(
        _TIMEOUT_FLOOR_S, _TIMEOUT_PER_CLIP_S * len(payload))
    request = json.dumps({'clips': payload, 'model': MODEL_ID,
                          'models_root': cfg.get('bank_scoring.models_root') or None})
    try:
        proc = subprocess.run(
            [python, _SCRIPT], input=request + '\n', capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=budget,
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        raise RuntimeError('the AI check timed out — check the ✨ Score '
                           'interpreter') from None
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'could not start the AI check: '
                           f'{type(e).__name__}: {e}') from None
    # Last line STARTING with '{' rather than blindly the last line — the same
    # scan the look score and person_mask use: a stray warning printed after the
    # payload must not turn a successful run into "no result".
    data = {}
    for text in reversed((proc.stdout or '').strip().splitlines()):
        if text.lstrip().startswith('{'):
            try:
                data = json.loads(text)
            except ValueError:
                data = {}
            break
    if not data:
        logger.warning('AI check: no JSON from the worker (rc=%s) stderr=%s',
                       proc.returncode, (proc.stderr or '')[-400:])
        raise RuntimeError('the AI check produced no result — check the '
                           '✨ Score interpreter')
    if not data.get('ok'):
        raise RuntimeError(str(data.get('error') or 'unknown AI-check error'))
    return {int(cid): list(values)
            for cid, values in (data.get('steps') or {}).items()}


def run_ai_check(bank_id, rescan=False, *, on_clip=None, should_stop=None):
    """Measure every shot of a bank that has no reading yet.

    Returns {'measured', 'too_short', 'unreadable', 'error'}. `error` is the
    encoder-could-not-load sentence and it is a RESULT rather than an exception,
    for the reason the watermark and safe-zone passes make it one: everything
    measured before it is real and kept, and a machine with no egress must not
    have its whole run reported as failed because a first-run download did not
    come back.
    """
    out = {'measured': 0, 'too_short': 0, 'unreadable': 0, 'error': None}
    bank = VideoBank.query.get(bank_id)
    if bank is None:
        return out
    rows = pending_clips(bank_id, rescan)
    if not rows:
        return out
    relpaths = {s.id: s.relpath for s in
                VideoSource.query.filter_by(bank_id=bank_id).all()}

    scratch = tempfile.mkdtemp(prefix='lds_aicheck_')
    try:
        for start in range(0, len(rows), CHUNK):
            if should_stop is not None and should_stop():
                break
            chunk = rows[start:start + CHUNK]
            payload, short, unreadable = _extract_chunk(bank, chunk, relpaths,
                                                        scratch)
            # The too-short and unreadable verdicts are written BEFORE the model
            # runs, and they are written even if the model then fails: they were
            # decided by the decode and re-deciding them on every run is how a
            # bank of two-second shots pays a torch import forever.
            for clip in short:
                _store(clip, {STATE_KEY: 'too_short'})
                out['too_short'] += 1
                if on_clip is not None:
                    on_clip()
            for clip in unreadable:
                _store(clip, {STATE_KEY: 'unreadable'})
                out['unreadable'] += 1
                if on_clip is not None:
                    on_clip()
            if not payload:
                _empty(scratch)
                continue
            try:
                steps = score_chunk([{'id': cid, 'frames': paths}
                                     for cid, paths, _clip in payload])
            except RuntimeError as e:
                logger.warning('video bank %s: AI check unavailable: %s',
                               bank_id, e)
                out['error'] = str(e)
                _empty(scratch)
                break
            for cid, _paths, clip in payload:
                # A FULL window or nothing: see `irregularity`. A clip the child
                # skipped keeps NO key at all, which is exactly what puts it back
                # in the next run's queue.
                series = steps.get(cid)
                if series is None or len(series) != FRAMES - 1:
                    continue
                value = irregularity(series)
                if value is None:
                    continue
                _store(clip, {STATE_KEY: 'ok', FRAMES_KEY: FRAMES,
                              SCORE_KEY: round(float(value), 4)})
                out['measured'] += 1
                if on_clip is not None:
                    on_clip()
            _empty(scratch)
    finally:
        _empty(scratch)
        try:
            os.rmdir(scratch)
        except OSError:
            pass
    return out


def _extract_chunk(bank, chunk, relpaths, scratch):
    """([(clip_id, [frame path], clip)], [too-short clip], [unreadable clip]).

    The three outcomes of a decode, kept apart on purpose. A shot too short to
    hold the window is a property of the CUT and will never change until it is
    re-cut; a shot whose frames would not decode is a property of the FILE. Both
    are honest states, and neither is a zero."""
    from .video_bank_service import _abs_source_path
    payload, short, unreadable = [], [], []
    for clip in chunk:
        times = window_times(clip.start_s, clip.end_s)
        if not times:
            short.append(clip)
            continue
        path = _abs_source_path(bank, relpaths.get(clip.source_id) or '')
        if not path:
            unreadable.append(clip)
            continue
        try:
            paths = _write_window(path, times, scratch, f'clip_{clip.id}')
        except Exception as e:      # noqa: BLE001 — one shot never sinks the pass
            logger.warning('AI check: clip %s window not decoded: %s', clip.id, e)
            unreadable.append(clip)
            continue
        payload.append((clip.id, paths, clip))
    return payload, short, unreadable


def _store(clip, values):
    """Merge this pass's reading into the clip's blob and commit it.

    MERGE, not replace: metrics_json holds the quality scores an expensive pass
    produced plus five other passes' verdicts, and overwriting it here would
    erase them silently. The keys this pass OWNS are cleared first, so a re-check
    that now finds a shot too short cannot leave last run's score beside this
    run's state.

    COMMITTED per clip, the resume contract every pass in this lane keeps."""
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


def _empty(scratch):
    """Drop this chunk's frames. Emptied per chunk rather than at the end, for
    the reason the safe zone empties its own: only one chunk's worth of JPEGs is
    ever on disk, whatever the size of the bank."""
    try:
        names = os.listdir(scratch)
    except OSError:
        return
    for name in names:
        try:
            os.unlink(os.path.join(scratch, name))
        except OSError:
            pass
