"""🎨 How each shot LOOKS — the image bank's aesthetic head, for free, on video.

WHY THIS EXISTS. A bank of rushes has a quality axis nothing in this lane
measured: not "is it sharp", not "does it move", but "is it nice to look at".
That is the axis every public video-curation pipeline filters on first — Wan runs
this exact LAION classifier as step one of its curation, and NVIDIA's Cosmos
Curator ships an aesthetic stage it deliberately leaves DISABLED by default,
offering a score-only mode instead. Both arrive at the same shape this lane
already had: produce the number, let the user choose the cut.

WHY IT COSTS NOTHING. 🔎 Find scenes already embedded three frames of every shot
into open_clip ViT-L/14 (openai) and wrote them to the bank's own .npz — the SAME
768-d space the image bank's ✨ Score pass rates. The improved-aesthetic-predictor
head is a 768→1 MLP over exactly that vector, so this pass DECODES NOTHING,
EMBEDS NOTHING and TOUCHES NO CARD: it hands the store to a small CPU subprocess
and writes the numbers back. The same argument that makes ✂ Duplicates instant.
Measured, on a forged store the size of a 3 000-shot bank (9 000 frames, 25.7 MB):
1.8 s end to end, cold subprocess and torch import included.

WHY IT RIDES THE EMBED PASS RATHER THAN A BUTTON OF ITS OWN. Two reasons, and
the second is the one that decided it. A shot cannot be rated before it is
embedded, so a separate button would be a second click that is *never* the right
first click. And a bank embedded before this shipped must not need a re-embed
(hours of decoding) to gain a score it can get from vectors already on disk — so
the step rates EVERY shot with vectors that has no rating yet, not just the ones
this run embedded. Re-clicking 🔎 Find scenes on a fully-embedded bank is
therefore the retrofit, and it costs one torch import.

NO WEIGHTS OF ITS OWN, NO INSTALL OF ITS OWN. Same head file, same URL, same
cache directory as the image lane (see infer/bank_score_infer.py) — an install
that has ever run ✨ Score already has it, and one that has not downloads the
same ~13 MB once. The capability is the one the embed pass already requires: the
✨ Score interpreter. There is nothing new for Setup to install.

ADVISORY, AND UNMEASURED IS A STATE. The raw score lands in metrics_json next to
the quality numbers; `low_aesthetic` is derived at read time against
`aesthetic_floor`, so moving that cut re-sorts the bank with no rescan. A shot
with no vectors carries NO key at all — never a 0, which on a 1..10 scale would
be the app asserting the shot is hideous. Nothing is deleted, nothing is
rejected, no triage decision is touched.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

from .. import config as cfg
from ..extensions import db
from ..models import VideoClip
from . import video_metrics
from . import infer_env
from .video_pass_scaffold import clip_summary as _summary

logger = logging.getLogger(__name__)

_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'video_aesthetic_infer.py')

# The child imports torch and downloads a 13 MB head on its first ever run. Same
# generosity as the frame encoder's start timeout, and for the same reason: a
# cold `import torch` on a machine that is also antivirus-scanning it is minutes,
# and a tighter bound would read as "the pass is broken" about one that is merely
# slow. Once warm the whole run is seconds.
TIMEOUT = 900

# The key this pass owns in the shared metrics blob. Named once: `pending_clips`
# uses its ABSENCE as the resume test, `verdicts()` reads it, and a second
# spelling of it would make a rated bank look permanently unrated.
SCORE_KEY = 'aesthetic_score'


def unavailable_reason():
    """None when this install can rate a bank, else the sentence saying why not.

    The frame encoder's own probe, deliberately: this runs in the ✨ Score
    interpreter, which is the very environment that produced the vectors being
    rated. If one cannot run, neither can the other, and inventing a second
    sentence for the same missing install is how a user installs twice."""
    from .clip_image_encoder import unavailable_reason as encoder_reason
    reason = encoder_reason()
    if reason is None:
        return None
    return reason.replace('frame embedding', 'the look score')


def pending_clips(bank_id, rescore=False):
    """The shots this pass would rate, oldest first.

    ONLY shots whose ``embed_state`` is 'ok'. That column is the authority over
    the store — the same rule ✂ Duplicates and 🔎 Search keep — and it is what
    stops a re-cut shot from being rated on three instants of a span it no longer
    has (a trim clears the column and leaves the vectors where they are).

    "Already rated" is a key in the blob rather than a column: this verdict rides
    in metrics_json with the rest, so the resume test is a JSON read and legacy
    databases need no migration to carry it. Same choice, for the same reason, as
    the watermark pass's ``watermark_state``.
    """
    rows = (VideoClip.query.filter_by(bank_id=bank_id, embed_state='ok')
            .order_by(VideoClip.id.asc()).all())
    if rescore:
        return rows
    return [c for c in rows if SCORE_KEY not in _summary(c)]


def score_frames(bank_id, *, timeout=TIMEOUT):
    """{clip_id: [per-frame score, ...]} for every vector in the bank's store.

    The MODEL seam — one subprocess in the ✨ Score interpreter, monkeypatched in
    tests so nothing here ever imports torch or downloads a head. Raises
    RuntimeError carrying the child's own words, because the caller turns that
    into a result rather than a 500 (see ``run_aesthetic``).

    The child is handed the STORE PATH, not the vectors. A bank of 3 000 shots is
    ~9 000 × 768 floats; serialising that as JSON is tens of megabytes and
    seconds of pure overhead for arithmetic that takes milliseconds, and the
    child already needs numpy to run the head at all.
    """
    from .video_clip_search import embed_cache_path
    store = embed_cache_path(bank_id)
    if not os.path.isfile(store):
        return {}
    python = cfg.get('bank_scoring.python') or sys.executable
    env = infer_env.worker_env(python, PYTHONUTF8='1')
    # Belt and braces with the child, which hides CUDA again before it imports
    # torch: this pass has no use for a card on any bank size, and must never be
    # the reason a training run loses one.
    env['CUDA_VISIBLE_DEVICES'] = ''
    payload = json.dumps({'store': str(store),
                          'models_root': cfg.get('bank_scoring.models_root') or None})
    try:
        proc = subprocess.run(
            infer_env.worker_argv(python, _SCRIPT),
            input=payload + '\n', capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=timeout,
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        raise RuntimeError('the look score timed out — check the ✨ Score '
                           'interpreter') from None
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f'could not start the look score: '
                           f'{type(e).__name__}: {e}') from None
    # Last line STARTING with '{' rather than blindly the last line — same scan
    # as person_mask's: a stray warning printed after the payload must not turn
    # a successful run into "no result".
    data = {}
    for text in reversed((proc.stdout or '').strip().splitlines()):
        if text.lstrip().startswith('{'):
            try:
                data = json.loads(text)
            except ValueError:
                data = {}
            break
    if not data:
        # The child's own words are the only diagnostic that exists for a worker
        # that died before emitting JSON (a torch DLL that would not load, an
        # OOM, a wrong `bank_scoring.python`). Same rc+stderr-tail convention as
        # person_mask and the ffmpeg drivers — a generic sentence with no trace
        # anywhere is not actionable by anyone.
        logger.warning('look score: no JSON from the worker (rc=%s) stderr=%s',
                       proc.returncode, (proc.stderr or '')[-400:])
        raise RuntimeError('the look score produced no result — check the '
                           '✨ Score interpreter')
    if not data.get('ok'):
        raise RuntimeError(str(data.get('error') or 'unknown look-score error'))
    # Keys arrive as JSON strings; the rest of this module works in clip ids.
    return {int(cid): list(values)
            for cid, values in (data.get('scores') or {}).items()}


def run_aesthetic(bank_id, rescore=False, *, on_clip=None, should_stop=None):
    """Rate every shot of a bank that has vectors and no rating yet.

    Returns {'rated', 'unrated', 'error'}. ``error`` is the head-could-not-load
    sentence, and it is a RESULT rather than an exception for the same reason the
    watermark pass makes it one: this runs at the tail of the embed pass, and a
    machine with no egress must not have its embedding run reported as failed
    because a 13 MB download did not come back.

    ``unrated`` counts shots this run left without a score and is reported rather
    than folded into a total: "every shot is rated" and "the store was missing
    half of them" are the same silence otherwise.
    """
    rows = pending_clips(bank_id, rescore)
    if not rows:
        return {'rated': 0, 'unrated': 0, 'error': None}
    if should_stop is not None and should_stop():
        return {'rated': 0, 'unrated': len(rows), 'error': None}
    try:
        scores = score_frames(bank_id)
    except RuntimeError as e:
        logger.warning('video bank %s: look score unavailable: %s', bank_id, e)
        return {'rated': 0, 'unrated': len(rows), 'error': str(e)}

    rated = unrated = 0
    for clip in rows:
        if should_stop is not None and should_stop():
            break
        value = video_metrics.aesthetic_of(scores.get(clip.id))
        if value is None:
            # A row whose vectors the store does not hold — the two can disagree
            # (an interrupted flush, a store pruned under a live bank). It is
            # left WITHOUT a key on purpose: that absence is exactly what puts it
            # back in the next run's queue, and writing None instead would retire
            # it as "rated, unrateable" forever.
            unrated += 1
            continue
        summary = _summary(clip)
        summary[SCORE_KEY] = round(float(value), 3)
        clip.metrics_json = json.dumps(summary)
        rated += 1
        if on_clip is not None:
            on_clip()
    # ONE commit, unlike the watermark pass's per-clip one. That pass commits per
    # clip because each verdict costs a decode and an inference it must not have
    # to pay twice; here the whole bank's arithmetic is already done in memory
    # before the first write, so a per-row commit would buy nothing and cost a
    # few thousand transactions.
    db.session.commit()
    return {'rated': rated, 'unrated': unrated, 'error': None}


