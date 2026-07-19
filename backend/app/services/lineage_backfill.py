"""Best-effort reconstruction of lineage edges for continuations that ran
BEFORE the genealogy feature persisted them.

A continue-in-place (« cook this LoRA a bit longer ») has always created a new
TrainingRunRecord each time (via checkpoint_registry.register_launch), but the
parent_record_id / resumed_from EDGE was only persisted once the lineage feature
shipped. So a chain of continuations trained before that lands in the Runs-hub
🌳 tree as a row of isolated roots instead of one lineage.

This runs ONCE at boot (guarded by a SystemState flag), walks every training
LANE (dataset + family + base + variant) and stamps the missing edge ONLY when
the evidence is unambiguous — otherwise the record is left as a root. No edge is
ever invented.

Evidence and decision rule (conservative by design):
  A record ``cur`` is the continuation of the record ``prev`` that immediately
  precedes it in the SAME lane (ordered by creation) when, and only when, BOTH:
    * ``cur.steps > prev.steps`` — a continuation always targets more total steps
      than the checkpoint it resumed from (continue enforces +100 minimum). A
      fresh restart resets the counter, so it never strictly exceeds its
      predecessor's target on the same dataset.
    * ``cur.fingerprint == prev.fingerprint`` — an in-place continuation
      re-exports the CURRENT dataset; when the dataset was untouched between the
      two launches the fingerprint (hence the human version) is identical. This
      is the exact signature of a « cook it longer » chain.
  Anything else is left as a root, on purpose:
    * a different fingerprint (the dataset was edited between launches) is
      indistinguishable from a fresh restart on the reworked dataset — ambiguous;
    * a resume from an EARLIER epoch (target below the predecessor's) leaves only
      a `_superseded_<ts>` folder as evidence, not a durable DB signal — ambiguous.
  Each stamped edge is marked ``lineage_origin = 'backfill'`` so it is auditable
  and tellable apart from a natively-persisted edge.

Best-effort and idempotent: any failure is swallowed (boot must never block),
a record that already has a parent is never touched, and a second run — same
boot or a later one — is a no-op (the flag short-circuits it, and every
already-stamped edge is skipped anyway)."""
from __future__ import annotations

import json
import logging
from datetime import datetime

from ..extensions import db
from ..models import SystemState, TrainingRunRecord

logger = logging.getLogger(__name__)

# Bump when the reconstruction RULE improves, so a smarter pass re-runs once on
# an already-processed database (it still only fills edges that are still NULL).
BACKFILL_VERSION = 1
_STATE_KEY = 'lineage_backfill'


def _lane_key(rec) -> tuple:
    """The training lane a record belongs to. base_model is already normalized to
    '' (official base) by register_launch; variant is stored as-is."""
    return (rec.dataset_id, rec.family, rec.base_model or '', rec.variant)


def _load_state() -> dict:
    row = db.session.get(SystemState, _STATE_KEY)
    if row is None or not row.value:
        return {}
    try:
        return json.loads(row.value)
    except (TypeError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    row = db.session.get(SystemState, _STATE_KEY)
    if row is None:
        row = SystemState(key=_STATE_KEY)
        db.session.add(row)
    row.value = json.dumps(state)
    db.session.commit()


def reconstruct_edges() -> int:
    """Stamp every unambiguous missing lineage edge. Returns the number of edges
    reconstructed. Pure DB work — no filesystem, so it behaves identically on any
    deployment and is fully covered by tests. Never raises."""
    try:
        records = TrainingRunRecord.query.order_by(
            TrainingRunRecord.created_at.asc(), TrainingRunRecord.id.asc()).all()
    except Exception:
        logger.exception('lineage backfill: could not read records (skipped)')
        db.session.rollback()
        return 0

    lanes: dict[tuple, list] = {}
    for rec in records:
        lanes.setdefault(_lane_key(rec), []).append(rec)

    stamped = 0
    for lane in lanes.values():
        # `records` is already globally ordered by (created_at, id); a lane keeps
        # that order, so lane[i-1] is cur's immediate predecessor in the lane.
        for prev, cur in zip(lane, lane[1:]):
            if cur.parent_record_id is not None:
                continue                       # already linked — never overwrite
            if prev.steps is None or cur.steps is None:
                continue
            if cur.steps <= prev.steps:
                continue                       # not a strictly-growing target
            if cur.fingerprint != prev.fingerprint:
                continue                       # dataset changed -> ambiguous, leave root
            cur.parent_record_id = prev.id
            if cur.resumed_from is None:
                # The in-place continuation resumed from prev's final step; prev.steps
                # is that step. Honest best estimate, and the edge is marked backfill.
                cur.resumed_from = prev.steps
            cur.lineage_origin = 'backfill'
            stamped += 1

    if stamped:
        try:
            db.session.commit()
        except Exception:
            logger.exception('lineage backfill: commit failed (rolled back)')
            db.session.rollback()
            return 0
    return stamped


def run_if_needed() -> dict:
    """Boot entry point: reconstruct edges once, guarded by a persisted version
    flag so a normal restart never re-scans. Returns the recorded summary. Wrapped
    so a failure logs and returns quietly — it must never block startup."""
    try:
        state = _load_state()
        if int(state.get('version') or 0) >= BACKFILL_VERSION:
            return state                       # already done at this rule version
        edges = reconstruct_edges()
        summary = {'version': BACKFILL_VERSION, 'edges': edges,
                   'ran_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z'}
        _save_state(summary)
        if edges:
            logger.info('lineage backfill: reconstructed %d edge(s)', edges)
        return summary
    except Exception:
        logger.exception('lineage backfill failed (non-fatal, boot continues)')
        db.session.rollback()
        return {}


def summary() -> dict:
    """The last recorded backfill result, for the diagnostic payload. {} until the
    first pass has run."""
    try:
        return _load_state()
    except Exception:
        return {}
