"""Best-effort reconstruction of lineage edges for continuations that ran before
the edge was persisted. Conservative: an edge is stamped only when the evidence
is unambiguous, and it is marked so it stays auditable."""
from datetime import datetime, timedelta

import pytest

from app.extensions import db
from app.models import SystemState, TrainingRunRecord
from app.services import lineage_backfill as bf


_T0 = datetime(2026, 7, 19, 3, 0, 0)


def _clear_flag():
    """create_app already ran the one-shot pass on the empty fixture DB (setting the
    flag with edges=0). Drop it so a test can exercise a not-yet-run database."""
    row = db.session.get(SystemState, 'lineage_backfill')
    if row is not None:
        db.session.delete(row)
        db.session.commit()


def _rec(dataset_id=1, family='zimage', base='', variant='turbo',
         steps=1000, fp='fpA', version=1, order=0, **kw):
    """A pre-feature record: no parent/resumed_from edge (as continuations stored
    them before the lineage feature). ``order`` sets a deterministic created_at."""
    r = TrainingRunRecord(
        dataset_id=dataset_id, family=family, source='local', base_model=base,
        variant=variant, steps=steps, fingerprint=fp, version=version,
        created_at=_T0 + timedelta(minutes=order), **kw)
    db.session.add(r)
    db.session.commit()
    return r


def test_reconstructs_cook_longer_chain(app):
    """Same lane, same fingerprint, strictly-growing step targets -> one linear
    lineage, each edge marked 'backfill' with resumed_from = parent's final step."""
    with app.app_context():
        a = _rec(steps=1000, order=0)
        b = _rec(steps=2000, order=1)
        c = _rec(steps=3000, order=2)
        assert bf.reconstruct_edges() == 2
        db.session.refresh(a); db.session.refresh(b); db.session.refresh(c)
        assert a.parent_record_id is None            # the root stays a root
        assert b.parent_record_id == a.id and b.resumed_from == 1000
        assert c.parent_record_id == b.id and c.resumed_from == 2000
        assert b.lineage_origin == 'backfill' and c.lineage_origin == 'backfill'
        assert a.lineage_origin is None


def test_edited_dataset_between_launches_is_left_root(app):
    """A different fingerprint means the dataset changed between the two launches —
    indistinguishable from a fresh restart on the reworked dataset. Ambiguous ->
    no edge (better an honest root than an invented parent)."""
    with app.app_context():
        a = _rec(steps=1000, fp='fpA', order=0)
        b = _rec(steps=2000, fp='fpB', order=1)     # dataset edited -> new fingerprint
        assert bf.reconstruct_edges() == 0
        db.session.refresh(b)
        assert b.parent_record_id is None and b.lineage_origin is None


def test_non_increasing_steps_is_left_root(app):
    """A fresh restart resets the step counter, so its target does not strictly
    exceed the predecessor's. Equal or lower target -> not a continuation."""
    with app.app_context():
        _rec(steps=1000, order=0)
        b = _rec(steps=1000, order=1)               # same target -> fresh restart
        c = _rec(steps=800, order=2)                # lower target
        assert bf.reconstruct_edges() == 0
        db.session.refresh(b); db.session.refresh(c)
        assert b.parent_record_id is None and c.parent_record_id is None


def test_lanes_never_cross_link(app):
    """Records only chain within one lane (dataset+family+base+variant). A run of a
    different family / variant / base / dataset is never a parent, even with a
    growing step count and matching fingerprint."""
    with app.app_context():
        z = _rec(family='zimage', steps=1000, order=0)
        k = _rec(family='krea', steps=2000, order=1)          # other family
        v = _rec(variant='base', steps=3000, order=2)         # other variant
        d = _rec(dataset_id=2, steps=4000, order=3)           # other dataset
        assert bf.reconstruct_edges() == 0
        for r in (k, v, d):
            db.session.refresh(r)
            assert r.parent_record_id is None
        db.session.refresh(z)
        assert z.parent_record_id is None


def test_chains_resume_after_an_ambiguous_gap(app):
    """A chain can pick back up after a break: an edited-dataset record is a root,
    and the next record continuing IT (same new fingerprint, growing target) is
    still linked to it — the break doesn't poison what follows."""
    with app.app_context():
        a = _rec(steps=1000, fp='fpA', order=0)
        b = _rec(steps=2000, fp='fpB', order=1)     # dataset edited -> root
        c = _rec(steps=3000, fp='fpB', order=2)     # continues b on the new dataset
        assert bf.reconstruct_edges() == 1
        db.session.refresh(b); db.session.refresh(c)
        assert b.parent_record_id is None
        assert c.parent_record_id == b.id and c.lineage_origin == 'backfill'


def test_never_overwrites_a_native_edge(app):
    """A natively-persisted edge in the middle of a chain is left untouched (its
    lineage_origin stays NULL); backfill only fills records whose parent is NULL."""
    with app.app_context():
        a = _rec(steps=1000, order=0)
        b = _rec(steps=2000, order=1)
        b.parent_record_id = a.id                   # already linked natively
        db.session.commit()
        c = _rec(steps=3000, order=2)
        assert bf.reconstruct_edges() == 1          # only c gets stamped
        db.session.refresh(b); db.session.refresh(c)
        assert b.parent_record_id == a.id and b.lineage_origin is None
        assert c.parent_record_id == b.id and c.lineage_origin == 'backfill'


def test_run_if_needed_is_one_shot(app):
    """The version flag makes the pass run once: a second call short-circuits and
    does not touch records added after the flag was set."""
    with app.app_context():
        _clear_flag()
        _rec(steps=1000, order=0)
        _rec(steps=2000, order=1)
        first = bf.run_if_needed()
        assert first['edges'] == 1 and first['version'] == bf.BACKFILL_VERSION
        # a later record — the flag stops it from being scanned again
        late = _rec(steps=3000, order=2)
        second = bf.run_if_needed()
        assert second['edges'] == 1                 # unchanged summary, no re-scan
        db.session.refresh(late)
        assert late.parent_record_id is None


def test_run_if_needed_reruns_when_rule_version_bumps(app, monkeypatch):
    """Bumping BACKFILL_VERSION lets an improved pass run once more; it still only
    fills edges that are still NULL."""
    with app.app_context():
        _clear_flag()
        _rec(steps=1000, order=0)
        b = _rec(steps=2000, order=1)
        bf.run_if_needed()
        db.session.refresh(b)
        assert b.parent_record_id is not None
        # a new record appears, then the rule version is bumped -> one more pass
        c = _rec(steps=3000, order=2)
        monkeypatch.setattr(bf, 'BACKFILL_VERSION', bf.BACKFILL_VERSION + 1)
        out = bf.run_if_needed()
        assert out['version'] == bf.BACKFILL_VERSION
        db.session.refresh(c)
        assert c.parent_record_id == b.id           # newly reachable edge filled


def test_empty_database_is_a_clean_noop(app):
    """A legacy database with no records: no edge, flag set, summary readable."""
    with app.app_context():
        out = bf.run_if_needed()
        assert out['edges'] == 0 and out['version'] == bf.BACKFILL_VERSION
        assert bf.summary()['edges'] == 0
        assert db.session.get(SystemState, 'lineage_backfill') is not None


def test_null_steps_never_link(app):
    """A record with no recorded step target (pre-steps pre-feature row) can't be
    placed on a step-growth chain -> left as a root, never guessed."""
    with app.app_context():
        _rec(steps=None, order=0)
        b = _rec(steps=2000, order=1)
        assert bf.reconstruct_edges() == 0
        db.session.refresh(b)
        assert b.parent_record_id is None
