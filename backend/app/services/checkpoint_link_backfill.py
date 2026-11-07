"""Attach the test images ALREADY on disk to the checkpoint that produced them.

From now on every launch writes ``lora_test_image.record_id`` / ``.step`` itself
(Test Studio, LoRA Canvas, comparison grid). This pass is what gives the same
link to everything generated before the columns existed, so the canvas gallery
opens on a real history instead of on an empty strip.

Four sources of evidence, in that order — and NOTHING else:

1. **The preview pointer.** ``checkpoint_preview`` already records
   ``(record_id, step) → lora_test_image_id`` for every image the Lab generated
   inline. That is not a heuristic, it is the link written at the time. One
   set-based UPDATE.

2. **The run tag baked into the deployed LoRA's name.** Deploying a checkpoint
   stamps the name with ``_rl<record id>`` (local) or ``_rc<cloud run id>``
   (cloud) — ``lora_training.parse_deployed_run`` reads it back — and the step is
   zero-padded in the middle of the same name
   (``cloud_training._step_of_testable``). A row whose ``checkpoint`` carries
   both is attributable exactly.

3. **The run tag ALONE, when the step is not in the name.** A step-less final
   save (``…_rl42.safetensors``) attributes itself to a run beyond doubt and to
   no step at all. Pass 2 threw that away with the step, so those images counted
   as "could not be traced back to a checkpoint" — true, and useless: the run WAS
   knowable. They now get ``record_id`` with ``step`` left NULL, which is exactly
   what the NAME tells. The run gallery shows them in a "step unknown" group;
   no checkpoint gallery picks them up, because those filter on a step — until
   pass 4 finds it.

4. **The run's lineage, for a final save's step.** That step was never unknown:
   the run's ◉ Graph pairs the step-less deployed file with its final pill
   (``final_save_step``, the join that pill shows "deployed" with). Every row
   that knows its run and not its step — pass 3's, and the ones an earlier
   version left — gets the step of the pill that shows its file; a file no pill
   shows (an older version, a copy since removed from ComfyUI) keeps its run
   only. This is the one pass that reads disk: when the ComfyUI LoRA folder, the
   relocated checkpoint store or a run's save list cannot be read, nothing is
   decided and the version is not recorded, so the next start tries again
   instead of filing "no pill" for good.

Everything else is LEFT UNLINKED. A name with no run tag (imported before
tagging existed, or a merge), a tag pointing at a run that no longer exists, a
cloud tag matching two records — all skipped. Those images stay invisible under
the nodes and the UI says how many there are (``unlinked_count``). Guessing would
put a stranger's image under someone's checkpoint, which is worse than an honest
gap. The member list a stacked cell carries in ``extra_loras`` is never rewritten.

Same discipline as ``lineage_backfill`` / ``framing_backfill``: version marker in
``SystemState``, idempotent, COUNT before UPDATE, set-based writes, and a failure
that never keeps the app from starting.
"""
from __future__ import annotations

import json
import logging
import os
from ..utils.timestamps import utc_stamp

from sqlalchemy import inspect, text

from ..extensions import db
from ..models import SystemState

logger = logging.getLogger(__name__)

# Bump when the RESOLUTION RULE improves, so a smarter pass re-runs once on a
# database it already visited (it still only fills links that are still NULL).
# 3: a final save's step, from its run's lineage (pass 4).
BACKFILL_VERSION = 3
_STATE_KEY = 'checkpoint_link_backfill'

# Pass 1 — the pointer written at generation time. Correlated subqueries rather
# than UPDATE ... FROM, which would need SQLite 3.33+ and buys nothing here.
# MIN(id) picks deterministically if a legacy database somehow holds two preview
# rows for one image; both would carry the same (record_id, step) anyway.
_POINTER_UPDATE_SQL = text(
    """
    UPDATE lora_test_image
       SET record_id = (SELECT p.record_id FROM checkpoint_preview p
                         WHERE p.lora_test_image_id = lora_test_image.id
                         ORDER BY p.id LIMIT 1),
           step      = (SELECT p.step FROM checkpoint_preview p
                         WHERE p.lora_test_image_id = lora_test_image.id
                         ORDER BY p.id LIMIT 1)
     WHERE lora_test_image.record_id IS NULL
       AND EXISTS (SELECT 1 FROM checkpoint_preview p
                    WHERE p.lora_test_image_id = lora_test_image.id)
    """
)

_POINTER_COUNT_SQL = text(
    """
    SELECT COUNT(*) FROM lora_test_image
     WHERE record_id IS NULL
       AND EXISTS (SELECT 1 FROM checkpoint_preview p
                    WHERE p.lora_test_image_id = lora_test_image.id)
    """
)

# Pass 2 works on DISTINCT checkpoint names, not on rows: a grid of 40 cells over
# 6 checkpoints costs 6 resolutions and 6 writes, whatever the library's size.
_UNLINKED_NAMES_SQL = text(
    """
    SELECT DISTINCT checkpoint FROM lora_test_image
     WHERE record_id IS NULL AND checkpoint IS NOT NULL AND checkpoint <> ''
    """
)

_LINK_BY_NAME_SQL = text(
    """
    UPDATE lora_test_image
       SET record_id = :rid, step = :step
     WHERE record_id IS NULL AND checkpoint = :ck AND dataset_id = :ds
    """
)

# Pass 3 — the run is known, the step is not. `step` is deliberately left NULL
# rather than defaulted to anything: a 0 or a guessed final step would put the
# image under a checkpoint pill that did not make it.
_LINK_RUN_ONLY_SQL = text(
    """
    UPDATE lora_test_image
       SET record_id = :rid
     WHERE record_id IS NULL AND checkpoint = :ck AND dataset_id = :ds
    """
)

# Pass 4 — the run is known, the step comes from its lineage. DISTINCT pairs, so
# a final save tested 200 times costs one lineage read. The UPDATE keys on the
# record already stored: a row is completed, never re-attributed.
_RUN_ONLY_PAIRS_SQL = text(
    """
    SELECT DISTINCT record_id, checkpoint FROM lora_test_image
     WHERE record_id IS NOT NULL AND step IS NULL
       AND checkpoint IS NOT NULL AND checkpoint <> ''
    """
)

_LINK_FINAL_STEP_SQL = text(
    """
    UPDATE lora_test_image
       SET step = :step
     WHERE record_id = :rid AND step IS NULL AND checkpoint = :ck
    """
)

_UNLINKED_COUNT_SQL = text(
    'SELECT COUNT(*) FROM lora_test_image WHERE record_id IS NULL')

_RUN_ONLY_COUNT_SQL = text(
    'SELECT COUNT(*) FROM lora_test_image WHERE record_id IS NOT NULL AND step IS NULL')


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


def _schema_supports_backfill() -> bool:
    """The columns this pass writes have to exist. They are added by the additive
    migration at boot, which runs first — but a database whose ALTER failed (a
    lock, a read-only volume) must skip the pass, not crash it."""
    insp = inspect(db.engine)
    names = set(insp.get_table_names())
    if 'lora_test_image' not in names or 'training_run_record' not in names:
        return False
    cols = {c['name'] for c in insp.get_columns('lora_test_image')}
    return {'record_id', 'step', 'checkpoint', 'dataset_id'} <= cols


def resolve_checkpoint_name(filename: str) -> tuple[int, int, int] | None:
    """(record_id, step, dataset_id) for a deployed LoRA filename, or None when
    the name does not attribute itself beyond doubt.

    Pure lookup over the run registry — no disk access, no folder scan. Returns
    None for: no run tag, no step in the name (a step-less final save), a tag
    pointing at a record that is gone, or a cloud tag matching more than one
    record (two records claiming one pod run: unattributable, so unattributed)."""
    from ..models import TrainingRunRecord
    from .lora_training import parse_deployed_run
    from .cloud_training import _step_of_testable

    source, run_id = parse_deployed_run(filename)
    if not source or run_id is None:
        return None
    step = _step_of_testable(filename)
    if step is None:
        return None
    if source == 'cloud':
        recs = TrainingRunRecord.query.filter_by(cloud_run_id=run_id).all()
        if len(recs) != 1:
            return None
        rec = recs[0]
    else:
        rec = db.session.get(TrainingRunRecord, run_id)
        if rec is None or rec.source == 'cloud':
            return None
    return rec.id, step, rec.dataset_id


def resolve_run_name(filename: str) -> tuple[int, int] | None:
    """(record_id, dataset_id) for a deployed LoRA filename whose RUN is knowable,
    whatever the step — or None.

    Deliberately weaker than ``resolve_checkpoint_name``: it answers the question
    "which run made this?" and stops there. A step-less final save answers it
    exactly; only the step is unknown, and an unknown step stays unknown."""
    from ..models import TrainingRunRecord
    from .lora_training import parse_deployed_run

    source, run_id = parse_deployed_run(filename)
    if not source or run_id is None:
        return None
    if source == 'cloud':
        recs = TrainingRunRecord.query.filter_by(cloud_run_id=run_id).all()
        if len(recs) != 1:
            return None
        rec = recs[0]
    else:
        rec = db.session.get(TrainingRunRecord, run_id)
        if rec is None or rec.source == 'cloud':
            return None
    return rec.id, rec.dataset_id


def _basename_key(name) -> str:
    return str(name or '').replace('\\', '/').rsplit('/', 1)[-1].lower()


def final_save_step(filename: str, record_id: int, dataset_id: int,
                    family: str | None = None, *, strict: bool = False) -> int | None:
    """The step of the save a deployed file IS, as its run's lineage shows it —
    or None when no pill of that run shows the file as deployed (an older
    version of the final save, a copy removed from ComfyUI since).

    This is the half of the answer a step-less final save is missing: ai-toolkit
    names it without a step (``…_rc192_v2``), so the name gives the run and
    nothing more, while the run's ◉ Graph already pairs that file with its final
    pill (``cloud_training._testable_for_record``). Asking that same join keeps a
    gallery and a pill from ever disagreeing about which checkpoint made an image.

    Unlike ``resolve_checkpoint_name`` this READS DISK — the ComfyUI LoRA folder
    and the run's save list. `family` picks the folder (the launch's family; else
    the name's folder, else the run's). Best-effort: None on any failure, unless
    `strict`, which lets a failed read raise for the boot pass that must tell
    "no pill" from "could not look"."""
    from ..models import TrainingRunRecord
    from ..utils.comfyui import family_of_lora
    from .cloud_training import _testable_for_record
    try:
        fam = family or family_of_lora(filename)
        if not fam:
            rec = db.session.get(TrainingRunRecord, record_id)
            fam = getattr(rec, 'family', None)
        if not fam:
            return None
        want = _basename_key(filename)
        by_step = _testable_for_record(dataset_id, fam, record_id, strict=strict)
        steps = {step for step, fn in by_step.items() if _basename_key(fn) == want}
        return steps.pop() if len(steps) == 1 else None
    except Exception:
        if strict:
            raise
        return None


def _listable(path) -> bool:
    if not path:
        return False
    try:
        with os.scandir(path) as entries:
            next(entries, None)
        return True
    except OSError:
        return False


def _unreadable_source() -> str | None:
    """The folder pass 4 needs and cannot read right now, or None.

    Both read as "nothing there" when they are missing — an unplugged drive, a
    ComfyUI not configured yet — and pass 4 would then file every final save as
    shown by no pill, for good. The checkpoint store is a cloud run's save list;
    it only counts when moved off the data folder, where its absence just means
    nothing was ever stored."""
    from .. import config as cfg
    if not _listable(cfg.comfyui_dir('loras')):
        return 'the ComfyUI LoRA folder'
    if (cfg.get('paths.checkpoints_dir') or '') and \
            not _listable(cfg.checkpoints_root(create=False)):
        return 'the checkpoint store'
    return None


def _link_final_saves(out: dict) -> None:
    """Pass 4: give every row that knows its run and not its step the step of the
    pill that shows its file (see final_save_step). Reads first, then one short
    write: no write lock is held across the disk reads."""
    pairs = [(r[0], r[1]) for r in db.session.execute(_RUN_ONLY_PAIRS_SQL)]
    if not pairs:
        return
    blocked = _unreadable_source()
    if blocked:
        out['deferred'] = blocked
        return
    found = []
    for rid, name in pairs:
        try:
            run_hit = resolve_run_name(name)
        except Exception:
            run_hit = None
        if not run_hit or run_hit[0] != rid:
            continue          # untagged, or tagged for another run: not ours to complete
        try:
            step = final_save_step(name, rid, run_hit[1], strict=True)
        except Exception:
            logger.debug('checkpoint link backfill: saves of run %s unreadable', rid,
                         exc_info=True)
            out['deferred'] = 'a run\'s saves'
            continue
        if step is not None:  # None: no pill shows this file, the run stays alone
            found.append((rid, name, step))
    for rid, name, step in found:
        out['by_final_save'] += db.session.execute(
            _LINK_FINAL_STEP_SQL, {'rid': rid, 'ck': name, 'step': step}).rowcount or 0
    if found:
        db.session.commit()


def backfill_checkpoint_links() -> dict:
    """Fill in every link the evidence supports. Returns
    {'by_pointer': n, 'by_name': n, 'by_run': n, 'by_final_save': n,
     'run_only': n, 'unlinked': n}, plus 'deferred' (what could not be read)
    when pass 4 has to wait for the next start. Never raises."""
    out = {'by_pointer': 0, 'by_name': 0, 'by_run': 0, 'by_final_save': 0,
           'run_only': 0, 'unlinked': 0}
    try:
        if not _schema_supports_backfill():
            logger.info('checkpoint link backfill: schema not ready (skipped)')
            return out

        # Read before writing: on most installs there is nothing to do, and this
        # way boot never opens a write transaction at all.
        if db.session.execute(_POINTER_COUNT_SQL).scalar():
            out['by_pointer'] = db.session.execute(_POINTER_UPDATE_SQL).rowcount or 0
            db.session.commit()

        names = [r[0] for r in db.session.execute(_UNLINKED_NAMES_SQL)]
        for name in names:
            try:
                hit = resolve_checkpoint_name(name)
            except Exception:
                hit = None                      # one unreadable name skips itself
            if hit:
                rid, step, ds_id = hit
                out['by_name'] += db.session.execute(
                    _LINK_BY_NAME_SQL,
                    {'rid': rid, 'step': step, 'ck': name, 'ds': ds_id}).rowcount or 0
                continue
            # The step is not in the name — but the run may still be. Record what
            # IS known instead of discarding the whole attribution.
            try:
                run_hit = resolve_run_name(name)
            except Exception:
                run_hit = None
            if not run_hit:
                continue
            rid, ds_id = run_hit
            out['by_run'] += db.session.execute(
                _LINK_RUN_ONLY_SQL,
                {'rid': rid, 'ck': name, 'ds': ds_id}).rowcount or 0
        if out['by_name'] or out['by_run']:
            db.session.commit()

        _link_final_saves(out)

        out['unlinked'] = db.session.execute(_UNLINKED_COUNT_SQL).scalar() or 0
        out['run_only'] = db.session.execute(_RUN_ONLY_COUNT_SQL).scalar() or 0
        return out
    except Exception:
        logger.exception('checkpoint link backfill: failed (skipped, nothing changed)')
        try:
            db.session.rollback()
        except Exception:
            pass
        return out


def unlinked_count() -> int:
    """How many test images carry no checkpoint link — the "N images not linked"
    counter the canvas shows so an incomplete history is stated, not hidden."""
    try:
        return db.session.execute(_UNLINKED_COUNT_SQL).scalar() or 0
    except Exception:
        return 0


def run_if_needed() -> dict:
    """Boot entry point: link once, guarded by a persisted version flag so a
    normal restart never re-scans. Never blocks startup."""
    try:
        state = _load_state()
        if int(state.get('version') or 0) >= BACKFILL_VERSION:
            return state
        res = backfill_checkpoint_links()
        if res.get('deferred'):
            # The version stays where it was: the next start runs the pass again.
            logger.info('checkpoint link backfill: %s cannot be read, final saves '
                        'keep their run without a step until the next start',
                        res['deferred'])
            return {'version': int(state.get('version') or 0), **res}
        summary_ = {'version': BACKFILL_VERSION,
                    'ran_at': utc_stamp(),
                    **res}
        _save_state(summary_)
        if (res['by_pointer'] or res['by_name'] or res['by_run']
                or res.get('by_final_save')):
            logger.info('checkpoint link backfill: %d image(s) linked by pointer, '
                        '%d by run tag, %d to a run without a step, '
                        '%d given their final save\'s step; %d still without a '
                        'step, %d left unlinked',
                        res['by_pointer'], res['by_name'], res['by_run'],
                        res.get('by_final_save', 0), res.get('run_only', 0),
                        res['unlinked'])
        return summary_
    except Exception:
        logger.exception('checkpoint link backfill failed (non-fatal, boot continues)')
        try:
            db.session.rollback()
        except Exception:
            pass
        return {}


def summary() -> dict:
    """The last recorded backfill result, for the diagnostic payload."""
    try:
        return _load_state()
    except Exception:
        return {}
