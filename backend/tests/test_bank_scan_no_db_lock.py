"""🗃️ Image bank — the QUALITY SCAN lets the rest of the app work while it runs.

`test_bank_infer_no_db_lock.py` pins this invariant for the score, face and
watermark passes. There was no equivalent for the scan, and the scan is the pass
the owner reported as "Scan quality freezes LDS, I can't get to anything during
the scan". Two different holders were measured behind that one sentence, so there
are two halves here and they fail for different reasons:

  • THE WALK. Re-reading each row goes through `Session.get`, and SQLAlchemy
    flushes before a SELECT. That flush took the single write lock at the first
    image after a mutation and held it until the commit 25 images later, so
    every other WRITER queued behind it: a sort click went from 5.0 ms median to
    327.6 ms (p90 2 183 ms, worst 5 564 ms, one HTTP 500 out of 62 clicks).

  • THE TAIL. `rebuild_dup_groups` wrote a global clear plus one UPDATE per group
    — about 5 000 of them on a 50 000-image bank — inside ONE transaction, held 6
    to 9.5 s. Past SQLite's 5 s busy_timeout that is not slowness, it is
    `database is locked` for everybody else (3 attempts out of 3).

These tests measure the PROPERTY, not its supposed causes: another connection
writes to the same database while the pass runs, and either it gets through or it
does not. A version that batches its commits but still parks the lock somewhere
else fails here, which is the whole point — "it commits in batches now" is a
statement about the code, "the app still answers" is a statement about the app.
Each half is paired with a test that reproduces the OLD holder on the same
fixture and shows the probe catching it, so a green run can never mean the probe
was simply looking the wrong way.
"""
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest
from PIL import Image
from sqlalchemy import text


@pytest.fixture()
def file_db(tmp_path, monkeypatch):
    """(app, db_path, workdir) — a real app on a file-backed SQLite database.
    A `:memory:` one cannot be opened twice, so the lock this measures has to
    live in a file, exactly like a real install's."""
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    from app import config as cfg
    monkeypatch.setattr(cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(cfg, '_cache', None)
    from app import create_app
    from app.extensions import db

    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': False})
    with application.app_context():
        mode = db.session.execute(text('PRAGMA journal_mode')).scalar()
        db.session.commit()
    # Recorded, not forced: WAL removes reader/writer contention, never the
    # single write lock these tests are about.
    assert str(mode).lower() in ('wal', 'delete'), f'unexpected journal mode {mode!r}'
    return application, str(tmp_path / 'data' / 'studio.db'), tmp_path


def _concurrent_write(db_path, timeout=0.5):
    """A write from ANOTHER connection, as a live request would. None on success,
    else the SQLite error text ('database is locked' when it was blocked)."""
    con = sqlite3.connect(db_path, timeout=timeout)
    try:
        con.execute("UPDATE image_bank SET name = 'written during the pass'")
        con.commit()
        return None
    except sqlite3.OperationalError as e:
        return str(e)
    finally:
        con.close()


# --- half one: the walk ------------------------------------------------------
def _scan_writer_verdict(file_db, images=8, probe_at=3, flush_first=False):
    """Run the quality scan over a file-backed bank and report what a concurrent
    writer saw at the ``probe_at``-th row re-read.

    The probe sits at the TOP of the re-read, so what it reports is the lock
    state the PREVIOUS iterations left behind — which is exactly where the
    autoflush used to take it. ``flush_first`` puts the flush back, reproducing
    the old behaviour on the same fixture.
    """
    from app.extensions import db
    from app.services import image_bank_service as banks

    app, db_path, workdir = file_db
    seen = {}
    with app.app_context():
        src = workdir / 'src'
        src.mkdir(parents=True, exist_ok=True)
        for i in range(images):
            Image.new('RGB', (600, 600), (9 * i, 90, 160)).save(str(src / f'a{i}.jpg'))
        bank, _added = banks.create_bank('local', 'Dump', str(src))
        db.session.commit()
        real_live = banks._live_image
        calls = {'n': 0}

        def probing_live_image(image_id):
            i = calls['n']
            calls['n'] += 1
            if flush_first:
                db.session.flush()      # what autoflush used to do here
            if i == probe_at:
                seen['error'] = _concurrent_write(db_path)
            return real_live(image_id)

        with patch.object(banks, '_live_image', probing_live_image), \
             patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
             patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
             patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None):
            banks._scan_job(bank.id, False)({})
    assert 'error' in seen, 'the pass never reached the probed image'
    return seen['error']


def test_the_quality_scan_lets_other_writers_through_while_it_walks_the_bank(file_db):
    error = _scan_writer_verdict(file_db)
    assert error is None, (
        'the quality scan held the write lock between its commits — on a real '
        'bank every sort, every rename, every promotion queues behind it '
        f'({error})')


def test_the_walk_really_did_hold_the_lock_before(file_db):
    """The same fixture with the flush put back. Without this, the test above
    could be green because the probe never fires where the lock is taken."""
    error = _scan_writer_verdict(file_db, flush_first=True)
    assert error and 'locked' in error, (
        'the probe did not see the old holder — it is not measuring the write '
        f'lock the scan used to take (got {error!r})')


# --- half two: the tail (duplicate regrouping) -------------------------------
def _bank_of_duplicate_pairs(banks, db, tmp_path, pairs):
    """A bank of `2 * pairs` rows forming `pairs` duplicate groups of two — the
    shape that makes the regrouping write one UPDATE per group. Rows only, no
    files: the regrouping reads hashes out of the database and never touches the
    disk, so a fixture of files would only make the test slower."""
    from app.models import BankImage
    src = tmp_path / 'hashes'
    src.mkdir(parents=True, exist_ok=True)
    bank, _added = banks.create_bank('local', 'Hashes', str(src))
    rows = []
    for g in range(pairs):
        # Far apart between groups (the low 40 bits differ wildly), one bit apart
        # inside a group.
        base = (g * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        rows.append(BankImage(bank_id=bank.id, relpath=f'{g:06d}a.jpg',
                              status='pending', dhash=f'{base:016x}',
                              analysis_fingerprint='0' * 64))
        rows.append(BankImage(bank_id=bank.id, relpath=f'{g:06d}b.jpg',
                              status='pending', dhash=f'{base ^ 1:016x}',
                              analysis_fingerprint='0' * 64))
    db.session.bulk_save_objects(rows)
    db.session.commit()
    return bank.id


class _WriterPressure:
    """A second connection writing over and over, for as long as the pass runs.
    Records every refusal and the longest wait it ever had to sit through."""

    def __init__(self, db_path, timeout=0.5):
        self.db_path, self.timeout = db_path, timeout
        self.failures, self.waits, self.attempts = [], [], 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            err = _concurrent_write(self.db_path, self.timeout)
            self.waits.append(time.perf_counter() - t0)
            self.attempts += 1
            if err:
                self.failures.append(err)
            time.sleep(0.002)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        return False


def _legacy_rebuild_dup_groups(banks, db, BankImage, bank_id, d, probe=None):
    """The write half exactly as it was: clear + one UPDATE per group, all in ONE
    transaction. Kept here so the test can show what it is protecting against.
    ``probe`` is called with the transaction OPEN, after the first group has been
    written and before the commit."""
    rows = (db.session.query(BankImage.id, BankImage.dhash)
            .filter(BankImage.bank_id == bank_id, BankImage.dhash.isnot(None))
            .order_by(BankImage.id.asc()).all())
    ids = [r[0] for r in rows]
    groups = banks._dup_groups_from_hashes([int(r[1], 16) for r in rows], d)
    BankImage.query.filter_by(bank_id=bank_id).update(
        {'dup_group': None}, synchronize_session=False)
    seen = {}
    for gid, members in enumerate(groups, start=1):
        member_ids = [ids[i] for i in members]
        for i0 in range(0, len(member_ids), banks._SQL_IN_CHUNK):
            BankImage.query.filter(
                BankImage.id.in_(member_ids[i0:i0 + banks._SQL_IN_CHUNK])).update(
                {'dup_group': gid}, synchronize_session=False)
        if probe is not None and 'verdict' not in seen:
            seen['verdict'] = probe()
    db.session.commit()
    return len(groups), seen.get('verdict')


# 12 000 rows in 6 000 groups. Sized from a measurement, not a guess: the old
# write shape (one UPDATE statement per group, one transaction) held the write
# lock 0.9 s on this fixture and made a concurrent writer wait 626 ms; the new
# one (one executemany per batch, a pause between batches) runs in 0.25 s and
# tops out at 24 ms of wait.
_PAIRS = 6000


def test_the_duplicate_regrouping_lets_other_writers_through(file_db):
    """The property, under continuous pressure: a second connection writes for
    the whole phase and must never be refused."""
    from app.extensions import db
    from app.services import image_bank_service as banks

    app, db_path, workdir = file_db
    with app.app_context():
        bank_id = _bank_of_duplicate_pairs(banks, db, workdir, _PAIRS)
        # This fixture intentionally owns no image files: it measures the DB
        # write shape, not filesystem throughput.  Model a stable, attested
        # generation at the two exact seams the regrouping now revalidates.
        def analysis_path(_bank, row, *args, **kwargs):
            return str(workdir / 'hashes' / row.relpath)

        with patch.object(banks, 'analysis_image_path', analysis_path), \
             patch.object(
                 banks.bank_transfer_metadata, 'content_fingerprint_path',
                 lambda _path: '0' * 64), \
             _WriterPressure(db_path) as pressure:
            t0 = time.perf_counter()
            n = banks.rebuild_dup_groups(bank_id, max_distance=1)
            elapsed = time.perf_counter() - t0
    assert n == _PAIRS, f'the fixture did not produce {_PAIRS} groups (got {n})'
    assert pressure.attempts > 5, 'the writer never got to try'
    assert not pressure.failures, (
        f'{len(pressure.failures)} of {pressure.attempts} writes were refused '
        f'while the duplicate regrouping ran ({elapsed:.1f} s) — the app is '
        f'unusable for its whole duration: {pressure.failures[0]}')
    # And the WORST wait, because that is the shape of this failure: the old code
    # blocked one writer for 626 ms and let the rest through afterwards, so a
    # median would have shrugged at it.
    #
    # Expressed as a FRACTION of the phase, not in milliseconds. An absolute
    # budget measures how fast the machine is, not whether the lock is parked:
    # calibrated here (24 ms of a 0.25 s phase) it failed on a CI runner ~23x
    # slower, which measured 554 ms of a 5.7 s phase — the very same 0.10 share,
    # and a green property reported as a regression. The shape this guards
    # against is not slow, it is PARKED: the old code held the lock for 626 ms
    # of a 0.9 s phase, a share of 0.70. A 0.25 line sits clear of both by a wide
    # margin on any hardware. Sensitivity does not rest on this number anyway —
    # test_the_regrouping_really_did_lock_the_database_before proves the probe
    # catches the old shape deterministically, with no timing at all.
    worst = max(pressure.waits)
    share = worst / elapsed if elapsed > 0 else 0.0
    assert share < 0.25, (
        f'a writer waited {worst * 1000:.0f} ms of a {elapsed:.1f} s regrouping '
        f'({share:.0%} of it, {pressure.attempts} attempts) — the lock is still '
        'being parked')
    # There used to be a second, absolute ceiling here (`worst < 2.0`), meant
    # as a loose "felt by the user" guard. It was the stopwatch this test had
    # just sworn off: on 2026-08-21 a hosted CI runner stretched the whole
    # phase to 11.3 s, a writer sat through 2 015 ms of it — a share of 0.18,
    # comfortably inside the line above — and the ceiling turned that green
    # property into a red build. An absolute wait measures how slow the runner
    # is, never whether the lock is parked; the share does, and the sibling
    # test below proves the probe catches the parked shape with no timing at
    # all. So the share is the only line, on purpose.


def test_the_regrouping_really_did_lock_the_database_before(file_db):
    """The old write shape, probed from INSIDE its own open transaction.

    Green above means nothing unless the probe is shown catching what it is there
    to catch — and this is the deterministic way to show it: the clear plus one
    group are written, the transaction is still open, and the second connection
    is asked to write. On a real bank that transaction stayed open for 6 to 9.5 s
    over ~5 000 groups; here one group is enough to prove the lock exists, with
    no dependence on how fast the machine happens to be."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    app, db_path, workdir = file_db
    with app.app_context():
        bank_id = _bank_of_duplicate_pairs(banks, db, workdir, 40)
        # The file is writable when nothing holds it — otherwise the verdict
        # below would say "locked" about something else entirely.
        assert _concurrent_write(db_path) is None
        n, verdict = _legacy_rebuild_dup_groups(
            banks, db, BankImage, bank_id, 1,
            probe=lambda: _concurrent_write(db_path))
        assert n == 40
        # And it is writable again the moment the transaction ends.
        assert _concurrent_write(db_path) is None
    assert verdict and 'locked' in verdict, (
        'the old one-transaction regrouping blocked nobody — the probe is not '
        f'measuring the write lock (got {verdict!r})')
