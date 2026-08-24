"""🔳 The safe-zone pass — decode three frames, measure the container, read the text.

WHAT IT ANSWERS. "How much of this shot is actually picture?" Two things eat a
frame and neither is visible at thumbnail size: the CONTAINER (letterbox and
pillarbox bands, from a re-encode, a vertical video padded into 16:9, a 4:3 scan)
and BURNED-IN TEXT (subtitles, chyrons, lower thirds, text watermarks). Both
survive a training crop, both are perfectly consistent across every frame of
every clip from the same source, and consistency is exactly what a LoRA learns
first. The arithmetic — bands, IoU clustering across frames, the resulting
rectangle — is next door in `video_safe_zone_geometry.py`, where it is testable
without onnxruntime; this module is the seam that gets it pixels and writes the
numbers back.

WHY IT DECODES ITS OWN FRAMES, which is the decision everything else follows
from. Nothing in this lane caches frames. The embedding pass writes its three
JPEGs to a temp folder and DELETES them the moment the vector exists
(`video_clip_search._discard`); what survives on disk is one 480 px thumbnail per
shot. So there is nothing to read for free, and the two designs built on that
assumption both fail:

  * riding 🔎 Find scenes and measuring its frames before they are discarded
    would tie this to a re-embed — hours of decoding — for any bank embedded
    before this shipped, and would run a CPU-heavy OCR inside the pass that holds
    the GPU window;
  * measuring the one cached thumbnail would give up the VOTE, and the vote is
    the whole discrimination: one frame cannot tell a subtitle from a shop sign,
    or a letterbox from a fade. 480 px is also thin for a subtitle.

So this pass owns its decode — three frames per shot at 768 px, the same
`_write_frames` seam 🔖 Watermarks uses, which is ONE file open per shot with
seek-then-decode-forward. Three frames from one open, not three opens.

WHY IT IS ITS OWN BUTTON. Every other advisory pass rides one: ✂ Duplicates and
🎨 Look consume what 🔎 Find scenes cached, so running them first would produce
an honest-looking empty answer. This one consumes NOTHING — a shot is measurable
the moment its source has probed — so there is no order to protect, and no reason
to make a user pay for an embedding run to learn that their footage is
letterboxed. It also degrades in a way a queued step could not report: the band
half needs no install at all, so an install with no OCR engine still gets its
bands measured and a visible `bars_only` state on every shot, instead of a
missing capability turning the whole thing off.

TWO HALVES, TWO INSTALL STORIES, ONE PASS.
  bands → PIL, in-process, free. No capability, no subprocess, no card.
  text  → RapidOCR in a CPU subprocess (see infer/video_text_infer.py). Absent,
          the pass still runs and says so.

ADVISORY, AND UNMEASURED IS A STATE. The raw numbers land in metrics_json beside
the quality scores; `letterboxed`, `burned_text` and `small_safe_zone` are
derived at read time against `bars_max`, `text_coverage_max` and `safe_area_min`,
so moving a cut re-sorts the bank with no rescan. A shot whose frames could not
be decoded carries `safe_zone_state: 'unreadable'` and NO numbers — never zeros,
which would be the app asserting the frame is entirely usable AND entirely
letterboxed at once. A shot measured without the OCR engine carries its bands and
NO text keys, so no text cut can fire on it. Nothing is deleted, nothing is
rejected, no triage decision is touched.

THE FRONTIER WITH 🔖 WATERMARKS, checked rather than assumed. That pass sends one
768 px frame to `watermark_detector.scan` and keeps `watermark_score` +
`watermark_state`; it explicitly discards the regions the detector returns, so it
stores no geometry at all. This pass writes only the seven `safe_*`/`text_*`/
`bars_*` keys and reads none of the watermark ones. They overlap in SUBJECT (a
text watermark is seen by both) and nowhere in code or in data — and that overlap
is wanted: one says "this shot is stamped", the other says "cropping it costs you
14 % of the frame".
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile

from .. import config as cfg
from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource
from . import video_safe_zone_geometry as geometry
from .video_pass_scaffold import clip_summary as _summary, empty_scratch as _empty, retry_when_idle as _retry_when_idle, store_pass_result

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'video_text_infer.py')

# Long side of the extracted frames — the constant somebody will find in six
# months and wonder about, so here is the arithmetic that chose it.
#
# THE FLOOR IS THE SUBTITLE. Broadcast and streaming subtitles are cut at
# roughly 4-5 % of frame height, so a 1080p source carries letters about 45 px
# tall; scaled to a 768 px long side (×0.71 on 16:9) they land around 32 px.
# Measured on this machine: RapidOCR read a 30 px line at 768×432 with 0.97
# confidence, so 768 sits just above the floor rather than on it. The embedding
# pass's 256 px would put that same line at ~10 px — GONE, and the OCR would
# then confidently report a bank with no text in it, which is the worst
# available outcome because it looks like a clean answer.
#
# THE CEILING IS THAT BIGGER BUYS NOTHING. RapidOCR's detector resizes to its
# own input internally, so handing it a native 4K frame pays ~7× the pixels of
# 1080p in decode and scratch disk to feed a downscale two functions later. Cost
# at 768 is already 0.61 s per warm frame on CPU (measured), three frames per
# shot; there is no budget to spend on pixels the model discards.
#
# Same number as 🔖 Watermarks' FRAME_LONG_SIDE, and that agreement is a
# coincidence of two similar constraints rather than a shared contract — a logo
# and a subtitle are both small things that vanish at thumbnail size. Neither
# constant should be changed because the other moved.
FRAME_LONG_SIDE = 768

# Shots per OCR child invocation. The engine costs ~1.4 s to import and build and
# ~0.6 s per frame on CPU (measured, 768 px), so 40 shots is ~120 frames, ~75 s
# of work, ~2 % startup overhead. Bounded because that chunk's JPEGs sit on disk
# at once — a whole bank of 768 px frames is not a scratch file, it is a second
# copy of the footage — and because the chunk is also the commit unit: a pass
# stopped or killed keeps every shot it already measured.
CHUNK = 40

# Recognition confidence a box must carry to count. RapidOCR scores 0..1 and
# reports the low end for texture it half-read as letters (foliage, brickwork,
# fabric patterns). 0.5 is the engine's own conventional floor; below it the
# boxes are noise, and noise that clusters across three frames of a static shot
# would be indistinguishable from a real subtitle.
TEXT_SCORE_MIN = 0.5

# Budget for one child, per frame, plus a floor that covers the cold import on a
# machine whose antivirus is scanning the ONNX models. Same shape as the face
# scorer's: a flat forfeit sized for a small chunk turns a slow machine's whole
# run into nothing, because a timeout returns no partial result.
_TIMEOUT_PER_FRAME_S = 6
_TIMEOUT_FLOOR_S = 300

_PROGRESS_RE = re.compile(r'\[text\] (\d+)/(\d+)')

# The key that says this pass has been here. Named once: `pending_clips` uses its
# ABSENCE as the resume test and `verdicts()` reads the numbers beside it, so a
# second spelling would make a measured bank look permanently unmeasured.
STATE_KEY = 'safe_zone_state'

# Every key this pass owns, so `_store` can clear a stale set wholesale rather
# than leaving half of a previous run's numbers beside a new state. Carried
# across a re-measure by video_metrics.ADVISORY_KEYS, which pins this list.
OWNED_KEYS = (STATE_KEY, 'safe_zone_frames', 'safe_bands', 'bars_ratio',
              'text_coverage', 'safe_rect', 'safe_area')


def text_engine_reason():
    """None when this install can read burned-in text, else the sentence saying
    why not.

    NOT a refusal for the pass — that is the whole point of the split. The band
    half needs nothing, so a missing OCR engine downgrades the result to
    `bars_only` and this sentence is what the job reports; it never turns the
    button off. Compare `video_watermark.unavailable_reason`, which IS a refusal,
    because that pass has nothing to offer without its detector.
    """
    from ..capabilities import probe_video_text
    probe = probe_video_text()
    if probe.get('ok'):
        return None
    return (f'burned-in text was not read — {probe.get("detail")}'
            if probe.get('detail') else 'burned-in text was not read')


# --- the seams -------------------------------------------------------------------

def _write_frames(src_path, times, dest_dir, stem, long_side=None):
    """The decode seam — `video_clip_search`'s, reused rather than copied, exactly
    as 🔖 Watermarks reuses it. That function owns the one-open-per-shot /
    seek-then-decode-forward contract, which is the part that is easy to get
    subtly wrong, and it takes a `long_side` override precisely so a second
    caller can ask for a bigger frame. Named here so tests can monkeypatch it
    without touching the search lane."""
    from .video_clip_search import _write_frames as write
    return write(src_path, times, dest_dir, stem, long_side=long_side)


def _clip_frame_times(clip):
    """The SAME three instants 🔎 Find scenes embeds, borrowed rather than
    re-derived. Two reasons, and the second is the one that matters: a shot's
    'key' frame is the metrics pass's ambassador, so this measurement and the
    look score describe the same three moments and can be read side by side —
    and re-deriving the rule here is how the two lists drift apart the day
    EDGE_MARGIN_S moves."""
    from .video_clip_search import _clip_frame_times as times
    return times(clip)


def luma_grid(path, probe=geometry.BAND_PROBE):
    """A JPEG as the row-major 0..255 luma grid `bands_of_grid` consumes, or None.

    Box-reduced on the way in (`Image.BOX`, the same reduction the image lane's
    bar scan uses): bands are a macro feature, the reduction bounds the cost to a
    fixed grid whatever the source resolution, and it averages away the codec
    artefacts inside the padding that otherwise end a scan one row early.
    """
    from PIL import Image
    try:
        with Image.open(path) as im:
            grey = im.convert('L')
            grey.thumbnail((probe, probe), Image.BOX)
            width, height = grey.size
            # tobytes(), not getdata(): identical bytes for mode 'L' (row-major,
            # no padding), faster, and getdata() is deprecated for removal in
            # Pillow 14 — this app pins Pillow 12 and will meet that.
            data = grey.tobytes()
    except Exception:  # noqa: BLE001 — one unreadable frame is not a 500
        return None
    if width < 8 or height < 8:
        return None
    return [data[y * width:(y + 1) * width] for y in range(height)]


def read_text_boxes(frames, *, timeout=None, should_stop=None, on_progress=None):
    """{key: [[x0,y0,x1,y1,score], ...]} for a chunk's frames — the MODEL seam.

    `frames` is [{'key', 'path'}]. One subprocess in the interpreter that has
    RapidOCR, monkeypatched in tests so nothing here ever imports onnxruntime.
    Raises RuntimeError carrying the child's own words; the caller turns that
    into a RESULT rather than a 500 — a missing OCR engine must not make a
    successful band measurement look like a failed pass.

    A key present with an EMPTY list means "read, no text". A key ABSENT means
    the child never reached that frame, and the clip goes back in the queue. The
    child guarantees that distinction; collapsing it here would retire a shot as
    "measured, no text" because a Stop landed on it.

    `on_progress(done, total)` is called from the stderr READER THREAD, so it may
    only touch in-memory state (`bank_jobs` is lock-guarded and qualifies). It
    exists because a chunk is 40 shots and ~75 s of OCR: without it the bar sits
    still for a minute and a quarter at a time, which reads as a hang.
    """
    from .infer_stream import run_infer_script, stderr_tail
    if not frames:
        return {}
    python = cfg.get('video_text.python') or _default_python()
    budget = timeout or max(_TIMEOUT_FLOOR_S,
                            120 + _TIMEOUT_PER_FRAME_S * len(frames))
    cancel_path, ask_stop, cleanup = _stop_plumbing()
    payload = json.dumps({'frames': frames, 'score_min': TEXT_SCORE_MIN,
                          'cancel_file': cancel_path}) + '\n'

    def _on_line(line):
        match = _PROGRESS_RE.search(line)
        if match and on_progress:
            on_progress(int(match.group(1)), int(match.group(2)))

    try:
        stdout, stderr_lines, rc, timed_out = run_infer_script(
            python, _SCRIPT, payload, budget, on_line=_on_line,
            should_stop=should_stop, on_stop=ask_stop)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'could not start the text reader: '
                           f'{type(e).__name__}: {e}') from None
    finally:
        cleanup()
    # Last line STARTING with '{' rather than blindly the last line — the same
    # scan video_aesthetic.score_frames does, and for the same reason: a stray
    # warning printed after the payload must not turn a successful run into
    # "no result".
    data = {}
    for text in reversed((stdout or '').strip().splitlines()):
        if text.lstrip().startswith('{'):
            try:
                data = json.loads(text)
            except ValueError:
                data = {}
            break
    if not data:
        # rc AND the child's own last stderr line. A worker that died before
        # emitting JSON (an onnxruntime DLL that would not load, an OOM, a wrong
        # `video_text.python`) leaves nothing else to act on, and a generic
        # sentence with no trace anywhere is not actionable by anyone. Same
        # convention as the look score and the ffmpeg drivers.
        logger.warning('safe zone: no JSON from the text reader (rc=%s, timed_out=%s) '
                       'stderr=%s', rc, timed_out, stderr_tail(stderr_lines))
        raise RuntimeError('the text reader produced no result — check the '
                           'burned-in text extra in Setup')
    if not data.get('ok'):
        raise RuntimeError(str(data.get('error') or 'unknown text-reader error'))
    return {key: [list(b) for b in (boxes or [])]
            for key, boxes in (data.get('boxes') or {}).items()}


def _default_python():
    import sys
    return sys.executable


def _stop_plumbing():
    """(sentinel path, ask, cleanup) — the house mechanism, same as face_mask's.

    The file is created only when a stop is actually ASKED: an existing file IS
    the request, so creating it up front would cancel the pass before it began,
    and a leftover one would silently cancel the next. Allocated even with no
    Stop button, because `run_infer_script`'s timeout watchdog sends the same
    request — a child that cannot be asked can only be killed, and a kill throws
    away every box it has read so far."""
    tmp = tempfile.mkdtemp(prefix='lds-vsz-')
    path = os.path.join(tmp, 'stop')

    def ask():
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write('stop')
        except OSError:
            logger.warning('safe zone: could not write the stop sentinel')

    return path, ask, (lambda: shutil.rmtree(tmp, ignore_errors=True))


# --- the pass ---------------------------------------------------------------------

def pending_clips(bank_id, rescan=False):
    """The shots this pass would work on, oldest first.

    Only shots of a source that PROBED: an unreadable file has no frame to
    extract, and counting it as unreadable on every run would make the pass look
    permanently broken. Deliberately NOT gated on `embed_state` the way 🎨 Look
    is — this pass reads pixels, not vectors, so requiring an embedding run first
    would be an invented dependency.

    "Already measured" is a key in the blob rather than a column: this verdict
    rides in metrics_json with the rest, so the resume test is a JSON read and
    legacy databases need no migration to carry it. Same choice, and the same
    reason, as the watermark pass's `watermark_state`.
    """
    q = (VideoClip.query.filter_by(bank_id=bank_id)
         .join(VideoSource, VideoSource.id == VideoClip.source_id)
         .filter(VideoSource.probe_state == 'ok'))
    rows = q.order_by(VideoClip.id.asc()).all()
    if rescan:
        return rows
    return _retry_when_idle(rows, STATE_KEY,
                            lambda c: STATE_KEY in _summary(c))


# --- the way back for a shot a pass gave up on -----------------------------------
# 'unreadable' is a HYPOTHESIS about a file, and every pass in this lane can form
# it for a reason that had nothing to do with the file — a decoder that stopped
# loading, a folder that moved, an interpreter an unrelated install broke. A
# hypothesis nothing ever re-tests becomes a fact by accident, and the shot is
# gone: the pass stops offering it, and its own button reports "nothing to do".
#
# So the pass offers them again, but only once it has NOTHING ELSE to do. On the
# normal path — a bank that grew — the second tier is never reached and costs
# nothing, which is what lets the recovery ride the button the user was already
# going to click instead of needing one of its own. See
# video_clip_search.pending_clips, where this rule was written first, after a
# bank of 861 shots retired itself in one pass.


def run_safe_zone(bank_id, rescan=False, *, on_clip=None, should_stop=None,
                  on_text_progress=None):
    """Measure the container and the burned-in text of every unmeasured shot.

    Returns {'measured', 'letterboxed', 'unreadable', 'text_frames', 'error'}
    — `text_frames` counts FRAMES the reader answered for, not shots, because
    that is the unit it works in and a counter that says "shots" while counting
    thirds of them is the kind of number nobody can reconcile with the log.
    `error` is the text-reader-could-not-run sentence, and it is a RESULT rather
    than an exception for the same reason the watermark pass makes it one: the
    band measurements of every shot this run touched are real and are kept, and
    the caller has to be able to tell "this install cannot read text" from "this
    bank carries none" — they lead to opposite decisions.

    `letterboxed` counts shots whose bands are not zero. It is NOT a verdict
    against `bars_max` (that cut is applied at read time, by the reader, against
    whatever the user has set right now) — it is the pass saying how much it
    found, so a run that reports "0 with bands" on obviously letterboxed footage
    is visibly wrong instead of quietly empty.

    The scratch folder goes on the way out whatever happened. 768 px JPEGs of a
    whole bank are a second copy of the footage, and a killed pass must not leave
    one behind.
    """
    bank = db.session.get(VideoBank, bank_id)
    if bank is None:
        return _empty_result()
    rows = pending_clips(bank_id, rescan)
    if not rows:
        return _empty_result()
    scratch = tempfile.mkdtemp(prefix=f'lds-vsz-{bank_id}-')
    try:
        return _measure_bank(bank, rows, scratch, on_clip, should_stop,
                             on_text_progress)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _empty_result():
    return {'measured': 0, 'letterboxed': 0, 'unreadable': 0, 'text_frames': 0,
            'error': None}


def _measure_bank(bank, rows, scratch, on_clip, should_stop, on_text_progress):
    relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                    .filter_by(bank_id=bank.id).all())
    out = _empty_result()
    # TWO different reasons the text can go unread, and they must not collapse.
    # This one is "this install has no OCR engine": known before the first
    # decode, permanent until somebody installs it, and `bars_only` is then the
    # honest answer for every shot of the run. The other one lives in the loop
    # below and gets the opposite treatment.
    #
    # Asked ONCE for the whole run rather than per chunk: a Setup install that
    # landed mid-pass would otherwise make the first half of a bank `bars_only`
    # and the second half 'ok', with nothing on screen explaining the split.
    text_reason = text_engine_reason()
    engine_missing = bool(text_reason)
    if engine_missing:
        out['error'] = text_reason

    for start in range(0, len(rows), CHUNK):
        if should_stop is not None and should_stop():
            break
        chunk = rows[start:start + CHUNK]
        extracted, frames = _extract_chunk(bank, chunk, relpaths, scratch)
        for clip in chunk:
            if clip.id in extracted:
                continue
            _store(clip, {STATE_KEY: 'unreadable'})
            out['unreadable'] += 1
            if on_clip is not None:
                on_clip()

        boxes = {}
        if frames and not engine_missing:
            try:
                boxes = read_text_boxes(frames, should_stop=should_stop,
                                        on_progress=on_text_progress)
                out['text_frames'] += len(boxes)
            except RuntimeError as e:
                # The OTHER reason, and it gets the OPPOSITE treatment. The
                # engine IS installed and its child died — an OOM, a killed
                # process, a DLL that unloaded. Writing `bars_only` here would
                # retire these shots on a machine that reads text perfectly
                # well, and nothing would ever queue them again. So this chunk
                # is abandoned whole: its shots keep NO state, which is exactly
                # what puts them back in the next run. A RESULT rather than a
                # raise, like the watermark pass — everything committed before
                # this stays, and the sentence travels to the job's own line.
                logger.warning('video bank %s: text reader stopped: %s', bank.id, e)
                out['error'] = str(e)
                break

        for clip in chunk:
            per_frame = extracted.get(clip.id)
            if per_frame is None:
                continue                            # already stored 'unreadable'
            summary = _summarise_clip(per_frame, boxes,
                                      text_ok=not engine_missing)
            if summary is None:
                # The text reader wound up before reaching this shot's frames —
                # a Stop, or a budget that elapsed. Left WITHOUT a state ON
                # PURPOSE: that absence is exactly what puts it back in the next
                # run's queue. Storing its bands alone would retire it as
                # `bars_only` on an install that can read text perfectly well.
                continue
            _store(clip, summary)
            out['measured'] += 1
            if summary.get('bars_ratio'):
                out['letterboxed'] += 1
            if on_clip is not None:
                on_clip()
        _empty(scratch)
    return out


def _extract_chunk(bank, chunk, relpaths, scratch):
    """({clip_id: [(key, bands|None), ...]}, [{'key','path'}, ...]).

    A clip ABSENT from the first dict is one whose frames could not be decoded —
    the caller retires it as 'unreadable'. A clip present carries one entry per
    frame it got, in decode order, each with the OCR key that frame will be
    reported under.

    ONE DECODE FEEDS BOTH HALVES, and that is a contract rather than a happy
    accident. The bands are measured off the very JPEGs whose paths go to the
    OCR child, in this loop, before anything else happens to them — so a shot is
    opened once and its three frames are written once, whatever the install can
    do with them. Splitting this into "a band pass" and "a text pass" would
    double the only expensive part of the whole feature for no gain, and it is
    the obvious-looking refactor, so `test_one_decode_feeds_both_halves` pins it.

    Extract first, then read the whole chunk's text in one child: the engine
    loads once per process, so interleaving one decode and one inference per clip
    would pay that load per clip. Bands are computed HERE, while the frame is on
    disk, because they cost a box-reduction and nothing else — sending them
    through the OCR child would make an install without it lose them, which is
    the whole reason the letterbox half survives a missing engine.
    """
    from .video_bank_service import _abs_source_path
    extracted = {}
    frames = []
    for clip in chunk:
        path = _abs_source_path(bank, relpaths.get(clip.source_id) or '')
        written = []
        if path:
            try:
                written = _write_frames(path, _clip_frame_times(clip), scratch,
                                        f'clip_{clip.id}',
                                        long_side=FRAME_LONG_SIDE)
            except Exception as e:      # noqa: BLE001 — one shot never sinks the pass
                logger.warning('safe zone: clip %s frames not extracted: %s',
                               clip.id, e)
                written = []
        if not written:
            continue
        per_frame = []
        for label, _seconds, jpeg in written:
            key = f'{clip.id}:{label}'
            grid = luma_grid(jpeg)
            per_frame.append((key, geometry.bands_of_grid(grid) if grid else None))
            frames.append({'key': key, 'path': jpeg})
        extracted[clip.id] = per_frame
    return extracted, frames


def _summarise_clip(per_frame, boxes_by_key, *, text_ok):
    """The numbers ONE shot carries, or None when it must stay in the queue.

    `per_frame` is [(ocr key, bands|None)] in decode order.

    Three shapes, and the difference between them is the whole honesty contract:

      'ok'        bands AND text measured. Every key present.
      'bars_only' the OCR engine could not run at all. Bands and their ratio are
                  stored; `text_coverage`, `safe_rect` and `safe_area` are
                  ABSENT, not zero — a safe zone computed without looking for
                  text would overstate itself, and `safe_area_min` would then
                  pass a clip on a measurement that never happened.
      None        the engine ran but did not reach every frame of this shot. NOT
                  a partial answer: two frames out of three cannot vote on the
                  third's subtitle, and a missing key is indistinguishable from
                  "no text" once it is written down.

    A fourth case hides inside 'ok': EVERY frame abstained from the band scan
    (all three are fades, a black slug, a title card). The bands are then
    unknown, so `safe_bands`, `bars_ratio`, `safe_rect` and `safe_area` are all
    left out — the last two claim to exclude a container nobody measured, and a
    `safe_area: 1.0` on a black clip is the app saying the whole frame is usable.
    `text_coverage` still rides, because it WAS measured, and `safe_zone_frames:
    0` is what says the rest was not.
    """
    bands = geometry.vote_bands([b for _key, b in per_frame])
    out = {STATE_KEY: 'ok' if text_ok else 'bars_only',
           'safe_zone_frames': sum(1 for _key, b in per_frame if b)}
    if bands is not None:
        # [top, bottom, left, right] — the order is geometry.SIDES and it is
        # part of the stored format, so a reader can rely on it and nobody may
        # reorder that tuple without an alias path. `safe_rect` is stored the
        # same way, as [x0, y0, x1, y1].
        out['safe_bands'] = [round(bands[side], 4) for side in geometry.SIDES]
        out['bars_ratio'] = round(geometry.bars_ratio(bands), 4)
    if not text_ok:
        return out

    if any(key not in boxes_by_key for key, _bands in per_frame):
        return None
    per_frame_boxes = [[box[:4] for box in boxes_by_key[key]]
                       for key, _bands in per_frame]
    structural = geometry.structural_text(per_frame_boxes)
    out['text_coverage'] = round(geometry.union_area(structural), 4)
    if bands is None:
        return out
    rect = geometry.safe_rect(bands, structural)
    out['safe_rect'] = [round(value, 4) for value in rect]
    out['safe_area'] = round(geometry.safe_area(rect), 4)
    return out


def _store(clip, values):
    """This pass's OWNED_KEYS through the shared merge-and-commit - see
    video_pass_scaffold.store_pass_result."""
    store_pass_result(clip, values, OWNED_KEYS)




