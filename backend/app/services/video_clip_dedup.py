"""✂ The same shot twice — near-duplicate clips, found in the vectors that exist.

WHY THIS EXISTS. A bank of rushes is full of retakes, and a training set built
from it counts each of them as a separate example. Ten near-identical takes of
one gesture do not teach a model ten things; they teach it one thing ten times as
loudly, which is how a LoRA ends up unable to do anything else. The user cannot
see that by scrolling — the takes are minutes apart in the grid and each looks
fine on its own.

WHY IT COSTS NOTHING. 🔎 Search already embedded three frames of every shot into
a 768-d CLIP space and wrote them to the bank's own .npz. Two shots being
near-identical is a question about vectors that are already on disk, so this pass
DECODES NOTHING and EMBEDS NOTHING: it reads that store, does a few thousand dot
products, and writes verdicts. Re-running it at another threshold is instant, for
the same reason the image lane's stage-2 dedup is instant.

WHY THE MAX OVER FRAME PAIRS, AND WHAT THAT COSTS. A shot is a span, and its
three embedded frames are three different pictures of it — a shot that starts on
a pan and settles shares its settled frames with the retake and nothing else. So
two shots are compared at their CLOSEST pair, which answers "does this bank hold
this moment twice" rather than "are these two shots the same throughout". The
honest cost of that choice: with three frames each, the comparison takes a MAX
over up to nine samples, so it reaches any given cut more easily than a
single-frame comparison would. That inflation is real, it is why the threshold is
exposed rather than baked in, and RAISING it is the remedy when a bank of
similar-looking material over-groups.

WHERE THE THRESHOLD COMES FROM — AND WHAT IS NOT MEASURED HERE.
Measured, in this project, on the IMAGE lane and over the SAME encoder and the
same 768-d space (open_clip ViT-L/14, the ✨ Score cache):

  * 0.96 is the cut the image bank ships for semantic near-duplicates — crops and
    re-compressed variants of one shot — calibrated against the push-down
    experiment's 7 316-image bank (see image_bank_service and
    config.DEFAULTS['bank']['semantic_dup_threshold']);
  * on a single-subject bank, two DIFFERENT photographs of the same person land
    at image-to-image cosine 0.60-0.89 (measured, image_bank_service.rank_similar).
    0.96 therefore sits far above "same subject, same style" and only catches
    pictures that are nearly the same picture.

NOT measured: any video-pair calibration. No corpus of labelled retakes exists in
this project yet, so this default is INHERITED by shared encoder rather than
derived from shots, and it is named as such here for the same reason
video_clip_search.HYBRID_CAPTION_WEIGHT is — presenting an inherited constant as
a measured one is the dishonest half of an otherwise honest feature. A user's own
bank is the calibration: the pass reports its group count, and the threshold is a
setting.

ADVISORY, LIKE EVERY VERDICT IN THIS LANE. Nothing is deleted, nothing is
rejected, no triage decision is touched. Each group keeps a REPRESENTATIVE (the
sharpest member) which carries no flag, and the others carry ``duplicate``. A
pass that flagged every member of a group would be telling the user to drop all
of them, which is the opposite of what it found.
"""
from __future__ import annotations

import json
import logging

from ..config import DEFAULTS
from ..extensions import db
from ..models import VideoClip
from . import video_metrics
from .video_pass_scaffold import clip_summary as _summary

logger = logging.getLogger(__name__)

# Inherited from the image lane's measured cut — see the module docstring. Read
# off the config defaults rather than retyped, so the two lanes cannot drift into
# two different meanings of "near-identical".
DEFAULT_THRESHOLD = DEFAULTS['bank']['semantic_dup_threshold']

# Rows of the frame matrix compared at once. Bounds the peak allocation of the
# similarity block independently of the bank's size — the same knob, and the same
# value, as the image lane's stage-2 dedup.
_ROW_BLOCK = 512


def configured_threshold() -> float:
    """The cut in force. Clamped rather than refused: this runs in the middle of
    a pass, and a config someone hand-edited to 5 must behave like "group
    nothing", visibly — not abort the job."""
    from .. import config as cfg
    try:
        value = float((cfg.get('video_bank') or {}).get('duplicate_threshold'))
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    return min(1.0, max(0.0, value))


def load_vectors(bank_id):
    """{clip_id: [{label, time_s, vec}]} — the store 🔎 Search wrote.

    A named seam over ``video_clip_search.load_embeddings`` so the pass can be
    exercised without numpy or an .npz on disk, and so a reader of this file can
    see in one line that it reads someone else's output rather than producing
    its own."""
    from .video_clip_search import load_embeddings
    return load_embeddings(bank_id)


# --- the arithmetic ---------------------------------------------------------------

def clip_similarity(frames_a, frames_b):
    """The cosine of the CLOSEST pair of frames between two shots, or None when
    either shot has no vectors.

    None, never 0.0: zero is a measurement ("nothing alike") and the absence of a
    vector is the absence of one. Collapsing them is how an un-embedded shot
    comes back looking checked-and-different."""
    if not frames_a or not frames_b:
        return None
    A = _matrix(frames_a)
    B = _matrix(frames_b)
    return float((A @ B.T).max())


def _matrix(frames):
    """L2-normed rows, so a dot product IS a cosine. Normed here rather than
    trusted from the store: the embed pass writes what the encoder returned, and
    one un-normed vector would silently outrank every honest comparison."""
    import numpy as np
    M = np.stack([np.asarray(f['vec'], dtype='float32') for f in frames])
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def group_clips(vectors, threshold):
    """[[clip_id, ...]] — the near-duplicate piles, each sorted, biggest first.

    UNION-FIND, the same shape (and for the same reason) as the image lane's
    stage-2 dedup: near-duplicate is not a pairing, it is a pile. Three takes of
    one slow pan can chain A~B~C while A~C stays under the cut, and reporting
    that as two overlapping pairs would ask the user to review the same footage
    twice and to keep two of the three.

    Groups of one are not groups and are not returned — a shot with no twin is
    not a finding.

    ONE MATRIX, NOT A PYTHON LOOP OVER PAIRS. The readable form — call
    ``clip_similarity`` for every pair — is O(n²) *interpreted*, and on a
    2 000-shot bank that is two million Python-level numpy calls: tens of
    minutes for arithmetic BLAS does in seconds. So every frame of the bank goes
    into one matrix, the similarities come out in row blocks, and each surviving
    frame pair is attributed back to the shots that own its two frames — the max
    over frame pairs then falls out of the union, since one pair above the cut is
    all it takes. Same shape as the image lane's stage-2 dedup.

    It is still quadratic in MEMORY BANDWIDTH, and that is the honest limit: the
    image lane can block its comparisons by style cluster and this has no such
    partition, so a very large bank pays a full n² matmul. Blocks of 512 rows
    keep the peak allocation flat whatever the bank's size.
    """
    import numpy as np
    ids = sorted(cid for cid, frames in vectors.items() if frames)
    parent = list(range(len(ids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if len(ids) >= 2:
        owner = []                      # frame row -> position in `ids`
        blocks = []
        for pos, cid in enumerate(ids):
            M = _matrix(vectors[cid])
            blocks.append(M)
            owner.extend([pos] * len(M))
        E = np.vstack(blocks)
        owner = np.asarray(owner)
        for i0 in range(0, len(E), _ROW_BLOCK):
            sims = E[i0:i0 + _ROW_BLOCK] @ E.T
            for a, b in np.argwhere(sims >= threshold):
                pa = int(owner[i0 + int(a)])
                pb = int(owner[int(b)])
                # `<` drops the diagonal, the mirror pairs, AND every pair of
                # frames belonging to the SAME shot — a shot is not its own
                # duplicate, and its own three frames are the most similar rows
                # in the whole matrix.
                if pa < pb:
                    union(pa, pb)
    piles = {}
    for i in range(len(ids)):
        piles.setdefault(find(i), []).append(ids[i])
    # Biggest pile first (that is the one worth opening), lowest id as the
    # tie-break so two runs over an unchanged bank number the groups identically.
    return sorted((sorted(p) for p in piles.values() if len(p) >= 2),
                  key=lambda p: (-len(p), p[0]))


def pick_representative(group, info):
    """The one member of a group that does NOT get flagged.

    The SHARPEST, read off ``sharpness_p90`` — the metrics pass's "is there real
    sharpness in this shot" number, and the same per-frame series whose argmax
    chooses the ambassador frame this pass compared. Among identical takes,
    softness is the thing that actually separates them and the thing a training
    set suffers from.

    An UNMEASURED shot never wins: it has no sharpness to offer, and keeping the
    one member of the pile whose quality nobody knows is the wrong trade. With
    nothing measured anywhere in the group the lowest clip id wins — arbitrary,
    but stable, so a re-run does not move the flag from one take to another.
    """
    def key(cid):
        sharp = (info.get(cid) or {}).get('sharpness')
        # (measured?, sharpness, -id) — maximised, so an unmeasured shot sits
        # below every measured one and ties fall to the lowest id.
        return (1 if sharp is not None else 0, sharp or 0.0, -cid)
    return max(group, key=key)


# --- the pass ----------------------------------------------------------------------

def run_dedup(bank_id, threshold=None, *, on_group=None, should_stop=None):
    """Group a bank's shots and write the advisory verdicts.

    Returns {'groups', 'flagged', 'evaluated', 'unevaluated'}. ``unevaluated`` is
    load-bearing and reported rather than folded into a total: a bank that was
    never embedded produces zero groups, and "no duplicates" and "nothing was
    compared" are the same sentence to a user who is not told which one happened.

    Every previous verdict is cleared first, so raising the cut UNFLAGS what no
    longer reaches it. A group left behind by an earlier run is a verdict nothing
    produced, and it is invisible — the user reads it as this run's answer.
    """
    t = float(configured_threshold() if threshold is None else threshold)
    vectors = load_vectors(bank_id)
    rows = VideoClip.query.filter_by(bank_id=bank_id).order_by(VideoClip.id).all()
    summaries = {r.id: _summary(r) for r in rows}
    # Only shots that still claim their vectors. `embed_state` is the authority
    # and the store is the data — the same rule the search keeps, and what stops
    # a re-cut shot from being grouped on three instants of a span it no longer
    # has.
    live = {r.id: vectors.get(r.id) for r in rows
            if r.embed_state == 'ok' and vectors.get(r.id)}

    # Clear first: a re-run recomputes wholly (see the docstring).
    dirty = set()
    for cid, summary in summaries.items():
        if summary.pop('duplicate_group', None) is not None:
            dirty.add(cid)
        if summary.pop('duplicate_of', 'absent') != 'absent':
            dirty.add(cid)

    info = {cid: {'sharpness': _sharpness(summaries[cid])} for cid in live}
    groups = group_clips(live, t)
    flagged = 0
    for gid, members in enumerate(groups, start=1):
        if should_stop is not None and should_stop():
            break
        keeper = pick_representative(members, info)
        for cid in members:
            summaries[cid]['duplicate_group'] = gid
            # None on the representative: the group is a fact about it too (the
            # grid says "kept out of 3"), but only the others are flagged.
            summaries[cid]['duplicate_of'] = None if cid == keeper else keeper
            dirty.add(cid)
            if cid != keeper:
                flagged += 1
        if on_group is not None:
            on_group()

    for r in rows:
        if r.id in dirty:
            summary = summaries[r.id]
            r.metrics_json = json.dumps(summary) if summary else None
    db.session.commit()
    return {'groups': len(groups), 'flagged': flagged,
            'evaluated': len(live), 'unevaluated': len(rows) - len(live)}




def _sharpness(summary):
    """The clip's sharpness, or None when it was never measured — including for a
    clip whose segment was unreadable, whose summary carries the key set to
    None."""
    if summary.get('metrics_state') != 'ok':
        return None
    value = summary.get(video_metrics.SHARPNESS_KEY)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
