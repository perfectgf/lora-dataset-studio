"""🩻 The defect sweep — ONE ffmpeg pass per SOURCE FILE, three artefacts out of it.

WHAT IT ANSWERS. "Has this file already been through the mill?" Scraped and
re-uploaded footage carries damage that is invisible at thumbnail size and that a
LoRA learns first, because it is perfectly consistent across every frame of every
clip from the same source:

  duplicated frames  a 30 fps upload of 24 fps material, a frame-rate conversion,
                     a stall in the encoder. The model sees the same picture
                     twice and learns that this subject does not move.
  compression blocks the 8×8 macroblock grid showing through. Never legitimate,
                     never removable, and it survives every crop and resize the
                     training pipeline applies.
  soft edges         out of focus, or — far more common in scraped material —
                     upscaled from something smaller. See `blur_score` for why
                     the metrics pass cannot see this one at all.

WHY FFMPEG AND NOT PyAV, which is the opposite of every other pass here. These
three measurements already exist as ffmpeg filters (`mpdecimate`, `blockdetect`,
`blurdetect`) that run INSIDE the decode loop in C, on the full-size frame, for
a fraction of what the same arithmetic costs in Python. Re-implementing them over
PyAV frames would be slower, less accurate and ours to maintain. The binary is
already a dependency of the lane (the promotion encodes with it), it is invoked
as a separate process exactly as the promotion invokes it, and nothing here links
against it.

WHY ONE RUN PER SOURCE FILE AND NOT PER CLIP. Decoding is the cost; the filters
ride inside it. A file cut into forty shots would pay forty decodes of the same
bytes to answer a question about ONE encode — and these three defects are
properties of the FILE, not of the shot: the macroblock grid does not change at a
cut. So the file is decoded once, every frame is timestamped, and the readings
are mapped onto each clip's window afterwards. Same "one decode" argument
video_metrics_scan makes for its own pass, applied one level up.

BY PTS, NEVER BY FRAME INDEX × NOMINAL FPS. Every reading below is keyed on the
`pts_time` ffmpeg prints. A variable-frame-rate file — which is most of what a
phone or a screen recorder produces — has no constant to multiply, and the
container's own frame count is not reliable either (measured on this lane's own
corpus: files whose `nb_frames` was out by half). A timestamp is the only thing
both halves of this agree on.

ADVISORY, AND UNMEASURED IS A STATE. The raw numbers land in metrics_json beside
the quality scores; `dup_frames`, `blocky` and `blurry` are derived at read time
against `dup_frames_max`, `block_max` and `blur_max`, so moving a cut re-sorts
the bank with no rescan. A clip whose window caught no frame carries
`defect_state: 'unreadable'` and NO numbers — never zeros, which would claim a
clip is perfectly clean when nothing looked at it. Nothing is deleted, nothing is
rejected, no triage decision is touched.

THE OTHER SURFACE, named rather than assumed — the Bank asks two of these three
questions already, and answers them differently on purpose:

  dup_frame_ratio  has NO image equivalent and cannot have one. It is a temporal
                   defect and a still has no time. Legitimate divergence, full
                   stop.
  blur_score       ↔ `BankImage.detail_ratio`. Same question ("was this enlarged
                   from something smaller"), deliberately different method. The
                   Bank rebuilds one still at a ladder of sizes and reports the
                   smallest that still reconstructs it — accurate, and it costs
                   several resamples of a full-size image. Paying that per FRAME
                   is not on the table, while `blurdetect` rides inside a decode
                   that is happening anyway. Both carry the same honest limit,
                   stated in both places: neither can separate an enlargement
                   from a genuinely soft photograph.
  bit_rate / bpp   ↔ `BankImage.jpeg_quality`, and the parity here is exact —
                   both are read straight out of the container header, both
                   describe how hard the file was squeezed, and both are
                   DISPLAYED FACTS that raise no flag. The Bank's comment says
                   "a displayed FACT, never a flag"; this lane owes the same
                   restraint for the same reason, and now has a stronger one:
                   `block_score` MEASURES the damage these two predict.
  bars_ratio       already shared, in both directions — 🔳 Safe zone reuses the
                   Bank's arithmetic so a number calibrated on stills carries
                   over. Nothing here touches it.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time

from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource
from . import ffmpeg_tools
from .video_pass_scaffold import clip_summary as _summary, store_pass_result

logger = logging.getLogger(__name__)

# The key that says this pass has been here. Named once: `pending_sources` uses
# its ABSENCE as the resume test and `verdicts()` reads the numbers beside it, so
# a second spelling would make a swept bank look permanently unswept. Same choice
# as 🔳 Safe zone's `safe_zone_state` — a key in the blob rather than a column,
# so legacy databases need no migration to carry it.
STATE_KEY = 'defect_state'

# Every key this pass owns, so `_store` can clear a stale set wholesale rather
# than leaving half of a previous run's numbers beside a new state. Carried
# across a re-measure by video_metrics.ADVISORY_KEYS, which pins this list.
OWNED_KEYS = (STATE_KEY, 'defect_frames', 'dup_frame_ratio', 'block_score',
              'blur_score')

# How many frames per SECOND the two expensive filters look at.
#
# THE COST IS THE WHOLE REASON THIS CONSTANT EXISTS. Measured on this machine, on
# 1080p25: decoding alone is 1.2 s per minute of source, `mpdecimate` adds
# nothing measurable, and `blockdetect` + `blurdetect` on every frame cost 17.9 s
# and 42.0 s per minute — the full chain runs at 59.4 s per minute, which is real
# time. At four samples a second it is 8.9 s per minute, a 6.7× saving, and the
# numbers it produces are the same ones (checked against a full-rate run on the
# same files: the p90 of a blocky clip moved from 37.3 to 43.3, both far above a
# clean clip's 13).
#
# FOUR, and not fewer, because the aggregation below is a DECILE: a two-second
# shot yields eight samples, which is the fewest a tenth can be read off without
# collapsing into the extremum. Shots shorter than a quarter of a second can fall
# between two samples entirely — they then carry `defect_frames: 0` and no
# quality numbers, which is the honest answer and not a zero.
#
# NOT a resolution reduction, which is the obvious-looking saving and the wrong
# one: `blockdetect` looks for the 8×8 macroblock grid and `blurdetect` measures
# edge width IN PIXELS. Downscaling destroys both, and would reproduce exactly
# the blind spot `blur_score` exists to cover (see below).
QUALITY_SAMPLES_PER_S = 4

# Assumed frame rate when the probe could not read one. 25 is the value the
# metrics scan already falls back to; a second guess with a different number
# would make two passes disagree about the same file.
_ASSUMED_FPS = 25.0

# Budget for one file, per second of source, plus a floor that covers a cold
# start behind an on-access antivirus scan. Generous on purpose: the sweep runs
# at ~0.15 s per second of 1080p source, so 2.0 is a thirteenfold margin, and a
# timeout here throws away a whole file's decode.
_TIMEOUT_PER_SOURCE_S = 2.0
_TIMEOUT_FLOOR_S = 300.0

# How often the run is asked whether it should stop. A source file can be an
# hour long, so waiting for the process to end before honouring a Stop would make
# the button look broken.
_POLL_S = 0.25

# The three files ffmpeg writes, named relative to a scratch directory it is
# launched IN. Relative on purpose: a filter argument is split on ':' and '\',
# so a Windows absolute path inside `file=` needs two levels of escaping and is
# the classic way this breaks on one platform only. `cwd` costs nothing and has
# no escaping.
_ALL_FILE = 'all.txt'
_KEPT_FILE = 'kept.txt'
_QUALITY_FILE = 'quality.txt'


def sweep_chain(step: int) -> str:
    """The filter chain, as ONE linear graph. `step` is the framestep divisor.

    Reading it left to right is reading the design:

      metadata=add     tags every frame with a throwaway key. WITHOUT IT THE NEXT
                       FILTER PRINTS NOTHING — `metadata=print` emits a frame's
                       header only when that frame carries at least one key, and
                       a freshly decoded frame carries none. Measured, not
                       assumed: the first draft of this chain produced two empty
                       files and a third full one.
      print → all      every frame's timestamp. The denominator of the ratio, and
                       the only honest one on a variable-frame-rate file.
      mpdecimate       drops the frames it judges near-identical to the one
                       before. It exports NO metadata of its own (checked against
                       this build's own `-h filter=mpdecimate`), so the only way
                       to learn what it did is to count what survives it.
      print → kept     the survivors' timestamps. The numerator.
      framestep        one frame in `step`, so the two expensive filters run on a
                       sample rather than on everything. AFTER mpdecimate, so the
                       sample is spread over the file's DISTINCT frames — a clip
                       that is nine tenths frozen contributes its moving tenth
                       rather than fifty copies of one picture.
      blockdetect      the macroblock grid, as `lavfi.block`.
      blurdetect       edge width, as `lavfi.blur`.
      print → quality  both, per sampled frame, with its timestamp.

    `framestep` and not `fps`: the fps filter produces a CONSTANT rate and
    DUPLICATES frames to reach it, which after mpdecimate has just removed
    duplicates would hand the detectors the very frames the previous filter threw
    away. framestep only ever drops, and it leaves the original timestamps alone.
    """
    return (f'metadata=mode=add:key=lds.f:value=1,'
            f'metadata=mode=print:file={_ALL_FILE},'
            f'mpdecimate,'
            f'metadata=mode=print:file={_KEPT_FILE},'
            f'framestep=step={max(1, int(step))},'
            f'blockdetect,blurdetect,'
            f'metadata=mode=print:file={_QUALITY_FILE}')


def framestep_for(fps) -> int:
    """How many frames to skip so the expensive filters see ~4 a second.

    Derived from the file's OWN rate rather than fixed, because a fixed step
    means a 60 fps file is sampled two and a half times as often as a 24 fps one
    — the same clip length yielding a different number of readings, which is a
    difference in the measurement rather than in the material.
    """
    try:
        rate = float(fps or 0)
    except (TypeError, ValueError):
        rate = 0.0
    if rate <= 0:
        rate = _ASSUMED_FPS
    return max(1, round(rate / QUALITY_SAMPLES_PER_S))


def unavailable_reason():
    """None when this install can sweep, else the sentence saying why not.

    A REFUSAL, unlike 🔳 Safe zone's missing OCR engine, and the difference is
    that this pass has no half that survives: all three measurements come out of
    the one binary. Same shape as 🔖 Watermarks, which refuses without its
    detector rather than delivering a 202 and a job that dies.
    """
    verdict = ffmpeg_tools.ffmpeg_ready()
    if verdict['ok']:
        return None
    return (f'the defect sweep needs ffmpeg and it is not usable here — '
            f'{verdict["reason"]}')


# --- parsing: pure functions over CAPTURED output --------------------------------

def parse_records(text):
    """ffmpeg's `metadata=mode=print` output as [(pts_time, {key: value})].

    The format, one frame at a time:

        frame:12   pts:6144    pts_time:0.48
        lds.f=1
        lavfi.block=15.225635

    A frame whose timestamp ffmpeg could not express prints `pts_time:N/A`; it is
    DROPPED rather than given a position, because a reading with no timestamp
    cannot be attributed to a clip and guessing one would attribute it to the
    wrong shot. Anything else unparseable is skipped in the same spirit: this
    function is fed a file written by another program, and one odd line must cost
    that line rather than the file.
    """
    records = []
    current = None
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('frame:'):
            current = None
            marker = line.find('pts_time:')
            if marker < 0:
                continue
            try:
                seconds = float(line[marker + len('pts_time:'):].split()[0])
            except (ValueError, IndexError):
                continue          # 'N/A', or a format we do not recognise
            current = (seconds, {})
            records.append(current)
        elif current is not None and '=' in line:
            key, _sep, value = line.partition('=')
            try:
                current[1][key.strip()] = float(value)
            except ValueError:
                pass              # a non-numeric key: not ours, not a failure
    return records


def frame_times(text):
    """Just the timestamps, in order. What the two counting files are read for."""
    return [seconds for seconds, _values in parse_records(text)]


def _in_window(times, start_s, end_s):
    """The readings that fall inside one clip, as a count or a list.

    HALF-OPEN, [start, end). The metrics scan uses a closed interval because it
    decodes each clip separately and an overlap costs it nothing; here ONE pass
    feeds every clip of a file, and shot boundaries touch — a closed interval
    would count the frame on the seam into both neighbours and inflate both
    their denominators.
    """
    return [t for t in times if start_s <= t < end_s]


def percentile(values, p):
    """Linear-interpolation percentile, or None for an empty list.

    Its own copy rather than an import of video_metrics.percentile: this module
    is the ffmpeg seam and that one is the PyAV pass's arithmetic. They agree
    today by construction and a test pins them together — what must not happen is
    this module growing an import of a module it has nothing else to do with.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = p * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def summarise_clip(all_times, kept_times, quality, start_s, end_s):
    """The numbers ONE clip carries, from a whole file's readings.

    `quality` is [(pts_time, {'lavfi.block', 'lavfi.blur'})] as parse_records
    returns it. Returns `{'defect_state': 'unreadable'}` — and nothing else —
    when no frame at all landed in the window: a clip nobody could look at must
    not be storable as a clip with no defects.

    THE THREE AGGREGATIONS ARE NOT THE SAME SHAPE, which is the part worth
    reading. video_metrics' own docstring sets the rule — the model trains on
    every frame, so the question is which MOMENT it also learns — and each of
    these answers it differently:

      dup_frame_ratio → the SHARE of frames mpdecimate would drop. A share, for
        exactly the reason `freeze_ratio` is one: a duplicated stretch inside an
        otherwise lively shot leaves every average perfectly healthy, and only
        the proportion reveals it. It is NOT freeze_ratio in other clothes —
        that one reads motion vectors and answers "nothing MOVED", this one
        answers "the same picture was DELIVERED twice", and a 24-in-30 pulldown
        produces the second with no trace of the first.

      block_score → p90, the WORST tenth. Blocking is never a choice: no camera,
        lens or lighting produces a macroblock grid, so the only question worth
        asking is whether real blocking exists anywhere in the shot, and the
        worst decile answers it while ignoring the single spike a hard frame
        inside the shot can produce. This is `sharpness_p90`'s statistic with
        the polarity reversed — that one asks whether the GOOD exists, this one
        whether the BAD does.

      blur_score → p10, the SHARPEST tenth, and it is the one that would be
        wrong the other way round. Softness IS sometimes a choice: a fast pan, a
        shallow depth of field, a deliberate rack focus all smear edges, and a
        p90 would flag precisely the clips with the most interesting movement —
        the exact false positive `sharpness_p90` chose p90 to avoid. Measured
        here on a forged clip whose only defect was duplication: its blur p90
        read 7.62 (flagged) while its p10 read 4.55 (clean, correctly), because
        the still frames a freeze leaves behind read soft to an edge-width
        metric. Asking "is it soft even at its sharpest" is the only form of the
        question that does not punish movement.

    WHAT `blur_score` SEES THAT `sharpness_p90` CANNOT, since the two look like
    the same measurement and are not. The metrics pass computes its Laplacian
    variance on a 160-pixel-wide analysis copy — deliberately, because that
    Laplacian over a full frame costs more than the decode. That choice makes it
    structurally blind to everything that happens ABOVE 160 pixels. Measured, on
    three forged files carrying the same picture:

        native 1080p              sharpness_p90 354.35   blur_score 5.56
        1080p upscaled from 480p  sharpness_p90 353.69   blur_score 6.62
        1080p upscaled from 320p  sharpness_p90 353.72   blur_score 7.35

    Three identical sharpness readings for three very different files, because at
    160 pixels wide they ARE identical. `blurdetect` runs at full size inside
    ffmpeg's own decode loop and separates them. Upscaled footage is the single
    most common defect in scraped material and the metrics pass cannot see it at
    all — that is what this second opinion is for, and it is why the flag it
    feeds is named for the edges rather than for compression.

    THE HONEST LIMIT, since a measurement with no stated failure mode is a claim
    rather than a number: `lavfi.block`'s absolute value depends heavily on
    CONTENT, not only on damage. Measured across four synthetic scenes at ONE
    fixed quality it spanned three orders of magnitude, while the same scene
    across a quality ladder moved by under 4×. So the cut belongs to a bank of
    similar material and the dry run is how it gets chosen — which is the rule
    every cut in this panel already follows, stated out loud here because this
    number is more content-sensitive than most.
    """
    window_all = _in_window(all_times, start_s, end_s)
    if not window_all:
        # No frame in this clip's window: the file was swept, this shot was not
        # in it. Could be a clip whose bounds fell outside the decoded range, or
        # a window shorter than the gap between two frames.
        return {STATE_KEY: 'unreadable'}
    window_kept = _in_window(kept_times, start_s, end_s)
    samples = [values for seconds, values in quality
               if start_s <= seconds < end_s]
    blocks = [v['lavfi.block'] for v in samples if 'lavfi.block' in v]
    blurs = [v['lavfi.blur'] for v in samples if 'lavfi.blur' in v]

    out = {
        STATE_KEY: 'ok',
        # How many frames the QUALITY half actually looked at. Its own key rather
        # than an inference from the presence of the scores, because zero is a
        # real and reachable answer (a shot shorter than the sampling interval)
        # and a reader has to be able to tell it from a shot that was never swept.
        'defect_frames': len(samples),
        'dup_frame_ratio': round(
            (len(window_all) - len(window_kept)) / len(window_all), 4),
    }
    # ABSENT, not zero, when the sample caught nothing — the rule every score in
    # this lane follows, and the one that keeps a cut from firing on a
    # measurement that never happened.
    block = percentile(blocks, 0.90)
    if block is not None:
        out['block_score'] = round(block, 4)
    blur = percentile(blurs, 0.10)
    if blur is not None:
        out['blur_score'] = round(blur, 4)
    return out


# --- the seam ---------------------------------------------------------------------

def sweep_file(path, fps, *, should_stop=None, duration_s=None):
    """Run ffmpeg over ONE file. -> (all_times, kept_times, quality_records).

    The single subprocess of this module, and the only thing in it that touches
    the outside world — everything above is a pure function over the text this
    returns, which is what lets the parsing and the arithmetic be tested on
    CAPTURED output with no binary present.

    Raises RuntimeError with something readable when the run could not produce
    readings; the caller turns that into a per-FILE failure and keeps going,
    because a bank is swept in bulk and one broken file must cost that file.

    Launched IN a scratch directory so the three `file=` arguments can be plain
    relative names (see _ALL_FILE), which is why `path` must be absolute — a
    relative input would be resolved against that scratch directory and not
    found. Polled rather than waited on, so a Stop lands within a quarter second
    instead of at the end of an hour-long file.
    """
    binary = ffmpeg_tools.ffmpeg_path()
    if not binary:
        raise RuntimeError('no ffmpeg binary found')
    scratch = tempfile.mkdtemp(prefix='lds-vdf-')
    try:
        args = [binary, '-hide_banner', '-loglevel', 'error', '-nostdin',
                '-i', os.path.abspath(path), '-an',
                '-vf', sweep_chain(framestep_for(fps)), '-f', 'null', '-']
        budget = max(_TIMEOUT_FLOOR_S,
                     _TIMEOUT_PER_SOURCE_S * float(duration_s or 0))
        stopped, code, tail = _run_polled(args, scratch, budget, should_stop)
        if stopped:
            raise RuntimeError('stopped')
        texts = []
        for name in (_ALL_FILE, _KEPT_FILE, _QUALITY_FILE):
            full = os.path.join(scratch, name)
            try:
                with open(full, encoding='utf-8', errors='replace') as fh:
                    texts.append(fh.read())
            except OSError:
                texts.append('')
        if not texts[0].strip():
            # ffmpeg wrote nothing at all: a file it could not decode, a codec it
            # does not have, a truncated download. rc and its own last words are
            # the only actionable trace, exactly as the other drivers here report.
            raise RuntimeError(f'ffmpeg read no frames (exit {code})'
                               + (f': {tail}' if tail else ''))
        return frame_times(texts[0]), frame_times(texts[1]), parse_records(texts[2])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


_STDERR_FILE = 'stderr.txt'


def _run_polled(args, cwd, budget, should_stop):
    """(stopped, returncode, stderr tail). Terminates on a Stop or a timeout.

    NO PIPES, and that is not a detail. This waits by POLLING rather than by
    `communicate()`, because a source file can be an hour long and a Stop has to
    land in a quarter of a second rather than at the end of it. A poll loop and a
    pipe together are a deadlock: nobody drains the pipe, ffmpeg fills the OS
    buffer (~64 KB), blocks on its next write, and never exits — so a file that
    merely produced a lot of warnings would spin until the whole budget elapsed
    and then be reported as a failure. stderr goes to a FILE, which has no such
    ceiling, and stdout is discarded because every number this pass wants is
    already going to the three metadata files.

    `terminate` and not `kill`: ffmpeg closes its outputs on SIGTERM, and the
    metadata files are flushed by that same teardown — a hard kill would leave
    three truncated files that parse into a plausible, wrong answer. Nothing
    reads them after a stop today, but a partial file that LOOKS complete is the
    kind of thing a later change starts trusting.
    """
    err_path = os.path.join(cwd, _STDERR_FILE)
    stopped = False
    with open(err_path, 'w', encoding='utf-8') as err_file:
        proc = subprocess.Popen(
            args, cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=err_file,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.time() + budget
        while proc.poll() is None:
            if should_stop is not None and should_stop():
                stopped = True
                break
            if time.time() > deadline:
                logger.warning('defect sweep: ffmpeg exceeded its %.0fs budget',
                               budget)
                break
            time.sleep(_POLL_S)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:        # noqa: BLE001 — it is going away regardless
                proc.kill()
    try:
        with open(err_path, encoding='utf-8', errors='replace') as fh:
            tail = fh.read().strip()[-300:]
    except OSError:
        tail = ''
    return stopped, proc.returncode, tail


# --- the pass ---------------------------------------------------------------------

def pending_sources(bank_id, rescan=False):
    """[(VideoSource, [VideoClip, ...])] — the files this pass would work on.

    A file with NO clips is skipped, and that is not an optimisation: this pass
    writes its numbers onto clips, so a decode of a file nobody has cut yet would
    cost minutes and have nowhere to put its answer. Same reason 🔳 Safe zone
    works off clips rather than files.

    Only files that PROBED, like every reading pass here: an unreadable file has
    no frames, and counting it as a failure on every run would make the pass look
    permanently broken.

    A file is pending when ANY of its clips is — the unit of work is the file,
    and re-cutting one shot of a forty-shot rush should not re-decode the rush
    for the other thirty-nine, but it must not leave the new shot unmeasured
    either. Sweeping the whole file is what the one-decode argument buys.
    """
    sources = (VideoSource.query.filter_by(bank_id=bank_id)
               .filter(VideoSource.probe_state == 'ok')
               .order_by(VideoSource.id.asc()).all())
    clips_by_source = {}
    for clip in (VideoClip.query.filter_by(bank_id=bank_id)
                 .order_by(VideoClip.id.asc()).all()):
        clips_by_source.setdefault(clip.source_id, []).append(clip)
    out = []
    for src in sources:
        clips = clips_by_source.get(src.id) or []
        if not clips:
            continue
        if rescan or any(STATE_KEY not in _summary(c) for c in clips):
            out.append((src, clips))
    return out


def run_defects(bank_id, rescan=False, *, on_clip=None, should_stop=None,
                on_file=None):
    """Sweep every file with unswept shots. Returns
    {'measured', 'files', 'unreadable', 'error'}.

    `unreadable` counts CLIPS, so it can be read against `measured` — files are
    counted separately, because a user who sees "3 unreadable" after a run over
    two hundred shots needs to know whether that is three bad shots or one bad
    file with three shots in it.

    `error` is the last file-level failure sentence, kept as a RESULT rather than
    raised for the same reason 🔳 Safe zone keeps its text-reader failure as one:
    every file swept before it is real and is kept, and "one file of this bank is
    broken" and "this install cannot sweep" lead to opposite decisions.

    Same resume contract as every other pass: each clip's numbers are committed
    as they are produced, so stopping loses nothing and a re-run pays only for
    the files the first run had not reached.
    """
    bank = db.session.get(VideoBank, bank_id)
    if bank is None:
        return _empty_result()
    pending = pending_sources(bank_id, rescan)
    if not pending:
        return _empty_result()
    from .video_bank_service import _abs_source_path

    out = _empty_result()
    for src, clips in pending:
        if should_stop is not None and should_stop():
            break
        path = _abs_source_path(bank, src.relpath)
        try:
            if not path:
                raise RuntimeError('the file is not inside the bank folder')
            all_times, kept_times, quality = sweep_file(
                path, src.fps_native, should_stop=should_stop,
                duration_s=src.duration_s)
        except RuntimeError as e:
            if str(e) == 'stopped':
                break
            # ONE bad file costs that file. Its clips are left WITHOUT a state on
            # purpose: that absence is exactly what puts them back in the next
            # run's queue, rather than retiring them as swept-and-clean.
            logger.warning('defect sweep: source %s not swept: %s', src.id, e)
            out['error'] = str(e)
            continue
        except Exception as e:               # noqa: BLE001 — never sink the pass
            logger.warning('defect sweep: source %s failed: %s', src.id, e)
            out['error'] = f'{type(e).__name__}: {e}'
            continue
        out['files'] += 1
        for clip in clips:
            summary = summarise_clip(all_times, kept_times, quality,
                                     clip.start_s, clip.end_s)
            _store(clip, summary)
            if summary.get(STATE_KEY) == 'ok':
                out['measured'] += 1
            else:
                out['unreadable'] += 1
            if on_clip is not None:
                on_clip()
        if on_file is not None:
            on_file(src.relpath)
    return out


def _empty_result():
    return {'measured': 0, 'files': 0, 'unreadable': 0, 'error': None}


def _store(clip, values):
    """This pass's OWNED_KEYS through the shared merge-and-commit - see
    video_pass_scaffold.store_pass_result."""
    store_pass_result(clip, values, OWNED_KEYS)


