"""🔗 Does one shot hold ONE scene — the cut the detector missed, found for free.

WHAT IT ANSWERS. A shot detector cuts on a change big enough to see. The cuts it
misses are the soft ones — a dissolve, a match cut, a change of angle inside the
same room — and what they leave behind is a "shot" that is actually two. That
clip is the worst kind of training example: it teaches the model a transition
nobody asked for, and it is invisible in the grid because its thumbnail is one
of its two halves and looks perfectly fine.

WHY IT COSTS NOTHING. 🔎 Find scenes already embedded three frames of every shot
into a 768-d CLIP space and wrote them to the bank's own .npz. "Does this shot
change into another one" is a question about vectors already on disk, so this
pass DECODES NOTHING, EMBEDS NOTHING and STARTS NO SUBPROCESS — unlike 🎨 the
look score, which at least hands the store to a torch child. It is a dot product
per shot, in this process, over a store the same run just wrote. Measured on the
bank forged in tests, and on 337 real shots: the arithmetic is microseconds and
the whole pass is dominated by the database commit.

WHERE THE THRESHOLD COMES FROM, AND WHY PANDA-70M's DOES NOT TRANSPOSE
=====================================================================
The method is Panda-70M's, and its numbers are NOT reusable. Read
``splitting/event_stitching.py`` of snap-research/Panda-70M:

  * line 47-48   it samples each cutscene at 5 % and 95 % of its length;
  * line 70/76   ``diff = (start - end).pow(2).sum().sqrt()``, and a cutscene
                 with ``diff > transition_threshold`` is DISCARDED as holding a
                 transition — called with 1.0 at line 197;
  * line 131-132 the same distance, and an event with ``diff <
                 still_event_threshold`` (0.15, line 199) is dropped as one where
                 nothing happens.

So the polarity is the one physics gives you and not the one the note in our
roadmap gave: a LARGE divergence between the two ends means the span holds two
scenes, and a NEAR-ZERO one means nothing changed. What is measured there is a
EUCLIDEAN DISTANCE over IMAGEBIND-huge features (1024-d, line 41), not a cosine
over CLIP ViT-L/14. Those features are unit-normed, so the two are convertible —
``cos = 1 - d²/2`` — and the conversion is exactly what shows the numbers cannot
travel:

  * 0.15 (still)      → cosine 0.9887
  * 1.0  (transition) → cosine 0.5

Measured here, on this app's own encoder (open_clip ViT-L/14 openai, the ✨ Score
cache) and on 1 040 frames of 366 real shots from four real files: two frames of
DIFFERENT SOURCE FILES — unrelated pictures, no shared scene, no shared people —
score cosine 0.720 at the median and 0.501 at the FIRST PERCENTILE. A cut at 0.5
therefore sits below almost every genuinely unrelated pair CLIP can produce, and
would flag nothing at all, ever. That is the cone effect ViT-L/14 is known for,
and it is why the constant is re-derived below rather than converted.

WHAT WAS ACTUALLY MEASURED, AND HOW
The positives cannot be found by looking — a missed cut is by definition one the
detector did not report. So they were FORGED the only honest way: two ADJACENT
detected shots of one file, fused into a single pseudo-clip. That is precisely
the footage a missed cut leaves behind, and its first and last frames are the
frames this pass would embed for it. 362 fused pairs against 337 real shots:

    start↔end cosine   real shots  median 0.9234   p25 0.8739   p5 0.7412
                       fused pairs median 0.8348   p25 0.7778   p5 0.6562

⚠️ AND THE FIRST THING THAT LOOKED LIKE SIGNAL WAS DURATION. A fused pair is
twice as long as the shots it is made of, and a longer shot has more time to
drift on its own — measured, over 234 shots with both readings, the span cosine
correlates with DURATION at Spearman −0.41. Duration alone separates the two
sets at AUC 0.657. So the whole thing was re-measured against duration-MATCHED
real shots (each positive against a real shot of its own length ±20 %), and the
signal survives it: AUC 0.719, and the fused clip scores lower than its matched
twin in 74.0 % of pairs against 50 % for no signal at all.

That is an HONEST SEVENTY-TWO, not a classifier, and the defaults and the hint
say so. Duration-matched, at a cut of 0.80: 34.0 % of the missed cuts caught for
14.6 % of honest shots flagged. At 0.75: 18.5 % against 9.9 %. At 0.70: 9.1 %
against 4.1 % — the bottom of the distribution really is mostly two-scene clips,
which is what makes this worth having as a RANKING (open the twenty worst) even
though it is far too coarse to reject on. Hence: raw numbers stored, cut derived
at read time, EMPTY by default, and nothing anywhere deletes on it.

⚠️ OUR FRAMES ARE NOT AT 5 %/95 %, AND IT MATTERS IN OUR FAVOUR — A LITTLE.
``video_clip_search.frame_times`` takes its outer frames EDGE_MARGIN_S (0.25 s)
inside the shot's own bounds, and its middle frame at the metrics scan's sharpest
instant rather than at the centre. So this pass compares a slightly NARROWER span
than Panda does, which can only make a shot look more coherent than it is — the
error is on the side of not flagging, which is the side an advisory reading
should err on. The middle frame's arbitrary position is also why it is not part
of the verdict; see ``coherence_of``.

THE SECOND FLAG THAT WAS SPECIFIED AND IS DELIBERATELY NOT SHIPPED
=================================================================
The other half of Panda's rule — a near-1 similarity means nothing moves, a
slideshow or a false move — was measured on the same 234 shots and REFUTED. It
does not measure motion:

  * Spearman(span cosine, ``motion_mean``) = −0.18. Essentially nothing, against
    −0.41 for duration: the number tracks how LONG a shot is more than twice as
    strongly as it tracks whether anything moves in it.
  * The six shots that reach cosine ≥ 0.98 are NONE of them in the least-moving
    decile, and their motion sits at or above the corpus median (0.0063, 0.0061,
    0.0037 against a median 0.0051). They are short shots, not still ones.
  * The genuinely still shots — ``motion_mean`` 0.00013 to 0.00063, the ones
    `still` flags first — read 0.94 to 0.98 here, under any cut that would not
    also swallow half the bank. The one LONG still shot in the sample (22.7 s,
    ``motion_mean`` 0.00033) reads 0.844, which this pass's other flag would call
    a MISSED CUT.

The reason is not a calibration failure, it is what CLIP is: it encodes what a
frame is OF, not what moved in it. A person moving inside a fixed frame is the
same scene throughout and scores ~1; a locked-off shot that runs for twenty
seconds slowly stops being it. So a "nothing moves" flag built on this would be
shot LENGTH wearing a motion-shaped label — and worse than redundant beside two
passes that measure the real thing: `still` reads the codec's own motion vectors
(``video_metrics``), and `slideshow` reads how well a single homography explains
the whole frame (``video_camera_motion.SLIDESHOW_RESIDUAL``). Not shipped, and
this paragraph is the reason.

ADVISORY, AND UNMEASURED IS A STATE. A shot with no vectors carries NO key at
all — never a 1.0, which would be this pass asserting perfect coherence about a
shot nothing looked at, and the absence is what puts it back in the next run's
queue. A shot the store holds ONE frame for (anything under
``MIN_SPAN_FOR_THREE_S``, so under a second) carries ``coherence_state:
'one_frame'`` and no numbers: there is no second instant to compare, and a
sub-second shot has no room for a missed cut worth finding.

THE OTHER SURFACE, named rather than assumed. The image bank has no equivalent
and cannot have one: every number here is a comparison between two INSTANTS of
one clip, and a photograph has no second instant. Same legitimate divergence the
camera pass and the defect sweep's ``dup_frame_ratio`` already state, and it
needs no port.
"""
from __future__ import annotations

import json
import logging

from ..extensions import db
from ..models import VideoClip

logger = logging.getLogger(__name__)

# --- what gets stored ---------------------------------------------------------------

STATE_KEY = 'coherence_state'

# Every key this pass owns. Cleared wholesale by `_store` before a re-run writes,
# and pinned against video_metrics.ADVISORY_KEYS by a test — a key this writes
# that the advisory list does not carry is erased by the next quality scan, in
# silence, sending a whole bank back through a pass it has already paid for.
OWNED_KEYS = (STATE_KEY, 'coherence_frames', 'coherence_span',
              'coherence_min', 'coherence_span_s')


def load_vectors(bank_id):
    """{clip_id: [{label, time_s, vec}]} — the store 🔎 Find scenes wrote.

    A named seam over ``video_clip_search.load_embeddings``, exactly like the
    dedup pass's, so this can be exercised without numpy or an .npz on disk and
    so a reader can see in one line that it consumes someone else's output."""
    from .video_clip_search import load_embeddings
    return load_embeddings(bank_id)


# --- the arithmetic ------------------------------------------------------------------

def coherence_of(frames):
    """The numbers ONE shot carries, from the frame vectors the store holds.

    `frames` is [{'label', 'time_s', 'vec'}] as ``load_vectors`` returns it, in
    no guaranteed order — this sorts by time, because every number below is about
    what happened BETWEEN two instants and a store written by a future pass with
    a fourth frame must not change what "the ends" means.

    Returns None when the store holds nothing for the shot — the caller writes NO
    KEY in that case (see ``run_coherence``). ``{'coherence_state': 'one_frame'}``
    when it holds exactly one: that IS a measurement outcome (we looked, there is
    nothing to compare) and it retires the shot from the queue, unlike the
    absence above which re-queues it.

    ⚠️ THE VERDICT IS THE FIRST-TO-LAST PAIR ALONE, and the two obvious
    alternatives were measured and are worse. Over 362 forged missed cuts against
    337 real shots (see the module docstring for how the positives are built),
    separating the two sets:

        first↔last cosine        AUC 0.8135      ← stored as `coherence_span`
        MIN over the three pairs AUC 0.7490      ← stored as `coherence_min`
        MEAN over the three      AUC 0.7466

    The min is the tempting one — "the shot is only as coherent as its least
    similar pair" — and it loses by four points because two of the three pairs it
    minimises over involve the MIDDLE frame, which sits wherever the metrics
    scan found the sharpest instant. That position is arbitrary with respect to a
    cut: it can land either side of one, or at an unremarkable moment of a
    perfectly coherent shot, and either way it contributes a low pair that says
    nothing about whether the shot holds two scenes. The first-to-last pair is
    also the one Panda-70M itself uses (event_stitching.py line 131), and the one
    that spans the most time — which is the whole question.

    `coherence_min` is stored anyway and it is not dead weight: it is the number
    a reader compares against the span to see WHERE a shot changed. A shot whose
    span is low and whose min equals it changed once, between its ends; a shot
    whose min sits well below its span changed around its middle frame. Neither
    reading is a flag, and neither costs anything to keep.

    `coherence_span_s` is the elapsed time the compared pair actually covers, and
    it is stored because the module docstring's own measurement says it is the
    main confound: this cosine falls with elapsed time (Spearman −0.41) whether
    or not anything was cut. A reader — or a later version of this pass — cannot
    normalise for that without the seconds, and inferring them from the clip
    bounds would be wrong, because the embedded frames sit EDGE_MARGIN_S inside
    them.
    """
    import numpy as np
    usable = [f for f in (frames or []) if f is not None and f.get('vec') is not None]
    if not usable:
        return None
    if len(usable) < 2:
        return {STATE_KEY: 'one_frame'}
    ordered = sorted(usable, key=lambda f: float(f.get('time_s') or 0.0))
    # L2-normed here rather than trusted from the store, for the same reason the
    # dedup pass norms its own matrix: the embed pass writes what the encoder
    # returned, and one un-normed vector would silently produce a "cosine" above
    # 1 that no threshold means anything about.
    M = np.stack([np.asarray(f['vec'], dtype='float32') for f in ordered])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    sims = M @ M.T
    pairs = [float(sims[i][j])
             for i in range(len(ordered)) for j in range(i + 1, len(ordered))]
    return {
        STATE_KEY: 'ok',
        'coherence_frames': len(ordered),
        'coherence_span': round(float(sims[0][-1]), 4),
        'coherence_min': round(min(pairs), 4),
        'coherence_span_s': round(float(ordered[-1]['time_s'])
                                  - float(ordered[0]['time_s']), 3),
    }


# --- the pass ------------------------------------------------------------------------

def pending_clips(bank_id, recheck=False):
    """The shots this pass would read, oldest first.

    ONLY shots whose ``embed_state`` is 'ok', the same rule ✂ Duplicates and
    🎨 the look score keep: that column is the authority over the store, and it is
    what stops a re-cut shot from being judged on three instants of a span it no
    longer has (a trim clears the column and leaves the vectors where they are).

    "Already read" is a key in the blob rather than a column, like every advisory
    pass in this lane — the resume test is a JSON read, and legacy databases need
    no migration to carry it."""
    rows = (VideoClip.query.filter_by(bank_id=bank_id, embed_state='ok')
            .order_by(VideoClip.id.asc()).all())
    if recheck:
        return rows
    return [c for c in rows if STATE_KEY not in _summary(c)]


def run_coherence(bank_id, recheck=False, *, on_clip=None, should_stop=None):
    """Read every shot of a bank that has vectors and no coherence reading yet.

    Returns {'measured', 'one_frame', 'unmeasured'}. ``unmeasured`` counts shots
    the STORE had nothing for and is reported rather than folded into a total,
    the same promise ``video_aesthetic.run_aesthetic`` makes about ``unrated``:
    "every shot was read" and "the .npz was missing half of them" are the same
    silence otherwise, and a desynchronised store is exactly the failure this
    lane's flush order (vectors first, states second) makes possible on purpose.

    ONE commit at the end, like the aesthetic pass and unlike the camera one:
    the whole bank's arithmetic is done in memory before the first write, so a
    per-row commit would buy no resume that matters and cost a few thousand
    transactions.
    """
    out = {'measured': 0, 'one_frame': 0, 'unmeasured': 0}
    rows = pending_clips(bank_id, recheck)
    if not rows:
        return out
    vectors = load_vectors(bank_id)
    for clip in rows:
        if should_stop is not None and should_stop():
            break
        values = coherence_of(vectors.get(clip.id))
        if values is None:
            # A row whose vectors the store does not hold. Left WITHOUT a key on
            # purpose: that absence is what puts it back in the next run's queue,
            # and writing a state instead would retire it as "read, unreadable"
            # forever over what is usually a store one re-embed would repair.
            out['unmeasured'] += 1
            continue
        _store(clip, values)
        if values.get(STATE_KEY) == 'ok':
            out['measured'] += 1
        else:
            out['one_frame'] += 1
        if on_clip is not None:
            on_clip()
    db.session.commit()
    return out


def _store(clip, values):
    """Merge this pass's reading into the clip's blob.

    MERGE, not replace: metrics_json holds the quality scores an expensive pass
    produced plus eight other passes' verdicts, and overwriting it here would
    erase them silently. The keys this pass OWNS are cleared first, so a re-run
    that now finds one frame cannot leave last run's numbers beside this run's
    state."""
    summary = _summary(clip)
    for key in OWNED_KEYS:
        summary.pop(key, None)
    summary.update(values)
    clip.metrics_json = json.dumps(summary)


def _summary(clip):
    """The clip's stored measurements, parsed. A corrupt blob reads as an empty
    one — this pass MERGES into what eight other passes wrote and must never be
    the reason a bank's quality scores disappear."""
    if not clip.metrics_json:
        return {}
    try:
        loaded = json.loads(clip.metrics_json)
    except (ValueError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
