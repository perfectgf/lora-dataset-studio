"""The shared skeleton of the video lane's advisory passes.

Every pass in this lane kept its own copy of the same moves - parse the
clip's metrics blob, order the pending pool, merge-and-commit a verdict,
drop a chunk's scratch frames. Measured on 2026-08-24: the copies had
drifted only in their docstrings (8/8 `_summary`, 4/4 `_retry_when_idle`,
4/6 `_store`, 3/3 `_empty` were logic-identical). One owner now, so the
NEXT pass starts from imports instead of a ninth copy.

Deliberately NOT here: `pending_clips` and `unavailable_reason`. Their
bodies genuinely differ per pass (each pool has its own gates, each
capability its own story) - a parameterized version would be a second
config language, not a deduplication. `video_temporal_coherence._store`
(no per-clip commit) and `video_watermark._store` (its own signature)
stay local for the same reason.
"""
import json
import os

from ..extensions import db


def clip_summary(clip):
    """The clip's stored measurements, parsed. A corrupt blob reads as an
    empty one - no pass in this lane may be the reason a bank's quality
    scores disappear."""
    if not clip.metrics_json:
        return {}
    try:
        loaded = json.loads(clip.metrics_json)
    except (ValueError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def retry_when_idle(rows, state_key, done):
    """`done` first; when there are none, the ones that failed last time."""
    fresh = [c for c in rows if not done(c)]
    if fresh:
        return fresh
    return [c for c in rows if clip_summary(c).get(state_key) == 'unreadable']


def store_pass_result(clip, values, owned_keys):
    """Merge one pass's reading into the clip's blob and commit it.

    MERGE, not replace: metrics_json holds the quality scores an expensive
    pass produced plus the other passes' verdicts, and overwriting it here
    would erase them silently. The keys the calling pass OWNS are cleared
    first, so a re-check that now finds a shot too short cannot leave last
    run's score beside this run's state.

    COMMITTED per clip - the resume contract every pass in this lane keeps."""
    summary = clip_summary(clip)
    for key in owned_keys:
        summary.pop(key, None)
    summary.update(values)
    clip.metrics_json = json.dumps(summary)
    db.session.commit()


def empty_scratch(scratch):
    """Drop this chunk's frames. Emptied per chunk rather than at the end:
    768 px JPEGs of a whole bank are a second copy of the footage, and the
    point of chunking is that only one chunk's worth is ever on disk."""
    try:
        names = os.listdir(scratch)
    except OSError:
        return
    for name in names:
        try:
            os.unlink(os.path.join(scratch, name))
        except OSError:
            pass
