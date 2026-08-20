"""🔖 Which shots carry a watermark — the image lane's detector, one frame per shot.

WHY IT MATTERS HERE MORE THAN IT DOES FOR STILLS. Rushes come off stock sites,
screen recordings and other people's uploads, and a logo that sits in the same
corner of every frame of every clip is not noise to a LoRA — it is the most
consistent thing in the dataset, so it is the first thing the model learns to
draw. The user cannot catch that by scrolling a grid of 90 px thumbnails, which
is exactly the size at which a corner mark disappears.

WHY NOTHING NEW WAS BUILT. The image bank already ships a measured, licence-
checked detector (SigLIP2 ranks, Grounding DINO locates — never `ultralytics`,
which claims AGPL over trained weights and would contaminate this public repo).
It answers in milliseconds with a NUMBER, which is what makes a threshold
possible. This module is the seam that points it at video: extract one frame per
shot, hand the paths over, write the verdicts back.

WHY ONE FRAME, AND WHY THAT ONE. A watermark is a property of the SOURCE, not of
the instant — it is burned into every frame of the file, so a second frame buys
almost nothing and costs a decode per shot on a bank of thousands. The frame
chosen is the AMBASSADOR: the sharpest sanely-exposed frame the metrics pass
already measured. Not the first frame, which is whatever the cut landed on and is
disproportionately a dissolve; not the middle, which is a guess where a
measurement exists.

WHY IT IS EXTRACTED BIG. The embedding pass writes 256 px frames because CLIP
sees 224×224. A watermark is a few dozen pixels in a corner and is simply GONE at
that size — the classifier would then confidently report a clean bank. The two
passes ask the same decode seam for two different sizes on purpose, the same way
the metrics scan asks it for 160 px.

ADVISORY, AND TERNARY. The score lands in metrics_json next to the quality
numbers; the flag is derived at read time against ``watermark_max``, so moving
that cut re-sorts the bank with no rescan. A shot whose frame could not be
decoded, or that the detector could not judge, is 'unreadable' — never a zero,
which would be the app asserting the shot is clean. Nothing is deleted, nothing
is rejected, no triage decision is touched.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile

from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource

logger = logging.getLogger(__name__)

# Long side of the extracted frame. Bigger than the embed pass's 256 (see the
# module docstring) and bounded anyway: the detector's own preprocessor resizes
# to its input size, so extracting a full 4K frame would cost decode and disk to
# feed a downscale that happens two functions later.
FRAME_LONG_SIDE = 768

# Shots per detector child invocation. The models load once per process (~10 s),
# so this wants to be large; it is bounded only because the frames of one chunk
# sit on disk at once, and a whole bank of 768 px JPEGs is not a scratch file, it
# is a second copy of the footage.
CHUNK = 200


def unavailable_reason():
    """None when this install can look for watermarks, else the sentence saying
    why not — the same shape as the embed and caption passes, so the refusal
    arrives before a 202 rather than ten minutes into a job that dies on an
    import."""
    from ..capabilities import probe_watermark_detect
    probe = probe_watermark_detect()
    if probe.get('ok'):
        return None
    return (f'the watermark detector is not ready — {probe.get("detail")}'
            if probe.get('detail') else 'the watermark detector is not ready')


def ambassador_time(start_s, end_s, summary):
    """The second of the frame to look at, for ONE shot.

    The metrics pass's ``sharpest_frame_s`` when it falls inside the shot; the
    middle otherwise. A timestamp OUTSIDE the bounds belongs to a span that has
    since been re-cut — clamping it would invent a measurement, and the middle is
    the honest fallback. Same rule, and the same reason, as
    ``video_clip_search.frame_times``."""
    start = float(start_s)
    end = max(float(end_s), start)
    mid = start + (end - start) / 2.0
    if summary and summary.get('metrics_state') == 'ok':
        try:
            s = float(summary.get('sharpest_frame_s'))
        except (TypeError, ValueError):
            return mid
        if start <= s <= end:
            return s
    return mid


# --- the two heavy seams -------------------------------------------------------------

def _write_frames(src_path, times, dest_dir, stem, long_side=None):
    """The decode seam — ``video_clip_search``'s, reused rather than copied.

    That function already owns the one-open-per-shot / seek-then-decode-forward
    contract, which is the part that is easy to get subtly wrong, and it takes a
    `long_side` override precisely so a second caller can ask for a bigger frame.
    Named here so tests can monkeypatch it without touching the search lane."""
    from .video_clip_search import _write_frames as write
    return write(src_path, times, dest_dir, stem, long_side=long_side)


# The detector's per-image result, field by field, in yield order. Declared as
# DATA rather than left implicit in a `for a, b, c ... in` line, because this
# module does not own this contract and cannot see it change: the detector grew
# a `fingerprint` field between two branches, this pass kept unpacking five
# values, and the only symptom would have been a ValueError on the first shot of
# the first real run. The tests could not catch it either — they stub the
# detector, so the stub simply agreed with the stale assumption.
# test_video_watermark.py now pins this tuple against the real generator.
SCAN_FIELDS = ('path', 'state', 'score', 'regions', 'fingerprint', 'error')


def _scan_frames(paths, **kwargs):
    """The detector seam. A generator of one ``SCAN_FIELDS``-shaped tuple per
    image, in input order — the image lane's own contract, borrowed whole."""
    from .watermark_detector import scan
    return scan(paths, **kwargs)


# --- the pass -------------------------------------------------------------------------

def pending_clips(bank_id, rescan=False):
    """The shots this pass would work on, oldest first. Only shots of a source
    that PROBED: an unreadable file has no frame to extract, and counting it as
    unreadable on every run would make the pass look permanently broken."""
    q = (VideoClip.query.filter_by(bank_id=bank_id)
         .join(VideoSource, VideoSource.id == VideoClip.source_id)
         .filter(VideoSource.probe_state == 'ok'))
    rows = q.order_by(VideoClip.id.asc()).all()
    if rescan:
        return rows
    # "Already scanned" is a key in the blob, not a column: this verdict rides in
    # metrics_json with the rest, so the resume test is a JSON read. Cheap enough
    # — the alternative is a migration on a table that legacy databases already
    # created without it.
    return _retry_when_idle(rows, 'watermark_state',
                            lambda c: 'watermark_state' in _summary(c))


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
def _retry_when_idle(rows, state_key, done):
    """`done` first; when there are none, the ones that failed last time."""
    fresh = [c for c in rows if not done(c)]
    if fresh:
        return fresh
    return [c for c in rows if _summary(c).get(state_key) == 'unreadable']


def run_watermark(bank_id, rescan=False, *, on_clip=None, should_stop=None):
    """Look for a watermark on every shot of a bank that has not been looked at.

    Returns {'scanned', 'detected', 'unreadable', 'error'}. ``error`` is the
    detector-could-not-load sentence, and it is a RESULT rather than an
    exception: the shots judged before it must keep their verdicts, and the
    caller has to be able to tell "this install cannot do this" from "this bank
    is clean" — they lead to opposite decisions.

    The scratch folder goes on the way out whatever happened. 768 px JPEGs of a
    whole bank are a second copy of the footage, and a killed pass must not leave
    one behind.
    """
    bank = db.session.get(VideoBank, bank_id)
    if bank is None:
        return {'scanned': 0, 'detected': 0, 'unreadable': 0, 'error': None}
    rows = pending_clips(bank_id, rescan)
    if not rows:
        return {'scanned': 0, 'detected': 0, 'unreadable': 0, 'error': None}
    scratch = tempfile.mkdtemp(prefix=f'lds-vwm-{bank_id}-')
    try:
        return _scan_bank(bank, rows, scratch, on_clip, should_stop)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _scan_bank(bank, rows, scratch, on_clip, should_stop):
    from .video_bank_service import _abs_source_path
    relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                    .filter_by(bank_id=bank.id).all())
    scanned = detected = unreadable = 0
    error = None

    for start in range(0, len(rows), CHUNK):
        if should_stop is not None and should_stop():
            break
        chunk = rows[start:start + CHUNK]
        # Extract first, then scan the whole chunk in one child: the models load
        # once per process, so interleaving one decode and one inference per clip
        # would pay that load per clip.
        by_path = {}
        for clip in chunk:
            path = _abs_source_path(bank, relpaths.get(clip.source_id) or '')
            frame = None
            if path:
                t = ambassador_time(clip.start_s, clip.end_s, _summary(clip))
                try:
                    written = _write_frames(path, [('key', round(t, 3))], scratch,
                                            f'clip_{clip.id}',
                                            long_side=FRAME_LONG_SIDE)
                    frame = written[0][2] if written else None
                except Exception as e:      # noqa: BLE001 — one shot never sinks the pass
                    logger.warning('watermark: clip %s frame not extracted: %s',
                                   clip.id, e)
            if frame:
                by_path[frame] = clip
            else:
                # The verdict is committed with the clip that produced it — the
                # same resume contract the metrics scan keeps. A stopped pass
                # keeps everything it already learned.
                _store(clip, None, 'unreadable')
                unreadable += 1
                if on_clip is not None:
                    on_clip()

        if not by_path:
            _empty(scratch)
            continue
        # Anything still in `by_path` when this block ends got no verdict — the
        # detector stopped early, or was cancelled. Those clips are left WITHOUT
        # a watermark_state on purpose: that is exactly what puts them back in
        # the next run's queue.
        try:
            # `fingerprint` is read and dropped on purpose. It exists so the
            # image lane can tell a verdict about a file apart from a verdict
            # about a file that has since been edited — a real question there,
            # and a meaningless one here: the thing being scanned is a scratch
            # JPEG this pass wrote seconds ago and deletes on the way out. The
            # staleness question for a shot is its BOUNDS, and a re-cut already
            # clears the whole blob.
            for path, state, score, _regions, _fingerprint, _err in _scan_frames(
                    list(by_path), should_cancel=should_stop):
                clip = by_path.pop(path, None)
                if clip is None:
                    # The detector already refuses out-of-order results; a path we
                    # never sent would mean attaching one shot's verdict to
                    # another's row, so it is dropped rather than guessed at.
                    logger.warning('watermark: unexpected result for %r', path)
                    continue
                if state == 'error' or score is None:
                    _store(clip, None, 'unreadable')
                    unreadable += 1
                else:
                    _store(clip, float(score), 'ok')
                    scanned += 1
                    if state == 'detected':
                        detected += 1
                if on_clip is not None:
                    on_clip()
        except Exception as e:              # noqa: BLE001 — DetectorUnavailable and friends
            # A RESULT, not a raise: everything judged so far is already
            # committed, and the caller has to be able to tell "this install
            # cannot do this" from "this bank is clean".
            logger.warning('watermark: detector stopped: %s', e)
            error = str(e)
            break
        finally:
            _empty(scratch)

    return {'scanned': scanned, 'detected': detected, 'unreadable': unreadable,
            'error': error}


def _store(clip, score, state):
    """Merge the verdict into the clip's blob and commit it.

    MERGE, not replace: metrics_json holds the quality scores a much more
    expensive pass produced, and overwriting it here would erase them silently.
    COMMITTED per clip: this is the resume contract every pass in the lane keeps,
    and the one the detector's own generator shape exists to allow — a bank of
    thousands that is stopped, or whose detector dies, keeps every verdict it
    already earned."""
    summary = _summary(clip)
    summary['watermark_score'] = None if score is None else round(float(score), 4)
    summary['watermark_state'] = state
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
    """Drop this chunk's frames. Emptied per chunk rather than at the end: 768 px
    JPEGs of a whole bank are a second copy of the footage, and the point of
    chunking is that only one chunk's worth is ever on disk."""
    try:
        names = os.listdir(scratch)
    except OSError:
        return
    for name in names:
        try:
            os.unlink(os.path.join(scratch, name))
        except OSError:
            pass
