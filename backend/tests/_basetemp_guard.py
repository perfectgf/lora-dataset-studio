"""Refuse to let two live pytest runs share one ``--basetemp``.

pytest DELETES an explicitly given ``--basetemp`` at startup: see
``_pytest/tmpdir.py``, ``TempPathFactory.getbasetemp`` — ``if basetemp.exists():
rm_rf(basetemp)``. Two runs pointed at the same path therefore destroy each
other's temporary trees mid-flight, and neither of them says so:

* the run that started FIRST loses, without warning, the files its tests are
  still using — the real git repositories of ``test_updater_history_rewrite``,
  the dataset folders of the preflight tests — and fails on an assertion that
  reads one of them back;
* the run that started SECOND dies with ``ERROR at setup`` on every single
  ``tmp_path`` fixture, because the directory it is busy deleting is held open
  by the first.

Both failures are intermittent (they need the two runs to overlap), independent
of test order, and green in isolation — which is exactly the shape everyone
reads as "flake". Measured on 2026-07-28: two suites sharing one ``--basetemp``
produced 1 phantom failure in ``test_updater_history_rewrite.py`` and 6 phantom
setup ERRORs in ``test_aitoolkit_remote.py``, none of which pointed at anything
wrong with the code or the tests.

So the collision is refused up front, loudly, naming the other run — a suite
that cannot run correctly must say so instead of producing archaeology.

The claim file is a SIBLING of the basetemp, never a child: pytest's own
``rm_rf`` of the basetemp would otherwise delete the very marker guarding it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# A claim older than this is presumed abandoned (a killed run leaves its file
# behind). Same order of magnitude as pytest's own numbered-dir lock timeout
# (3 h), and far longer than any run of this suite — the full backend suite is
# ~7 minutes.
STALE_AFTER_SECONDS = 3 * 60 * 60


def claim_path(basetemp) -> Path:
    basetemp = Path(str(basetemp))
    return basetemp.parent / (basetemp.name + '.pytest-owner')


def _read(path: Path) -> tuple[int | None, float | None]:
    """(pid, epoch) recorded in a claim file — (None, None) when unreadable."""
    try:
        raw = path.read_text(encoding='utf-8').strip().split()
        return int(raw[0]), float(raw[1])
    except (OSError, ValueError, IndexError):
        return None, None


def claim(basetemp, pid=None, now=None) -> str | None:
    """Take ownership of ``basetemp`` for this process.

    Returns ``None`` when the basetemp is ours to use, or a human message
    explaining the conflict when another live run already holds it. A stale
    claim (older than ``STALE_AFTER_SECONDS``) is taken over silently.
    """
    pid = os.getpid() if pid is None else pid
    now = time.time() if now is None else now
    path = claim_path(basetemp)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None            # can't guard it; never block a run over the guard
    payload = f'{pid} {now}\n'
    try:
        # O_EXCL: whoever creates the file wins, with no read-then-write race.
        with open(path, 'x', encoding='utf-8') as fh:
            fh.write(payload)
        return None
    except FileExistsError:
        pass
    except OSError:
        return None
    other_pid, started = _read(path)
    if other_pid == pid:
        return None            # our own claim (a re-entrant configure)
    if started is None or (now - started) > STALE_AFTER_SECONDS:
        try:
            path.write_text(payload, encoding='utf-8')
        except OSError:
            pass
        return None
    age = int(now - started)
    return (
        f'--basetemp={basetemp} is already in use by pytest pid {other_pid} '
        f'(started {age}s ago).\n'
        'pytest DELETES the basetemp it is given at startup, so sharing one '
        'between two runs makes each destroy the other\'s temporary files: the '
        'older run fails on files that vanished under it, the newer one ERRORs '
        'at setup on every tmp_path fixture. Both look like flakes and neither '
        'is one.\n'
        'Give this run its own --basetemp (append the pid, or just omit the '
        'option — pytest then picks a fresh numbered directory per run).\n'
        f'If no such run is alive, delete {claim_path(basetemp)}.'
    )


def release(basetemp, pid=None) -> bool:
    """Drop our claim. False when the file was not ours (never steal a release)."""
    pid = os.getpid() if pid is None else pid
    path = claim_path(basetemp)
    other_pid, _started = _read(path)
    if other_pid != pid:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True

def sweep_stale_siblings(basetemp, pid=None, now=None) -> list[str]:
    """Collect the basetemps dead runs left behind, next to ours.

    A killed pytest never reaches ``release()``: its claim file survives, and
    its basetemp keeps the gigabytes the suite wrote (images, exports, one
    copy per worker). Measured 2026-08-23: ~45 of them had accumulated on one
    machine - 105 GB - and under 10 GB of free disk the training preflight
    starts refusing launches, which reads as a merge regression five tests
    wide (nothing in the diff explains it).

    Scoped to what THIS guard provably owns: only a sibling whose
    ``.pytest-owner`` claim is stale (same ``STALE_AFTER_SECONDS`` the
    takeover in ``claim()`` uses) is deleted - directory and claim together -
    and an orphan claim whose directory is already gone is dropped too. A
    directory with NO claim is never touched: this scratch parent is shared,
    and everything here errs on the side of leaving foreign data alone.
    Best-effort like the rest of the guard: never blocks or fails a run.
    Returns the names it removed (for the tests).
    """
    import shutil
    pid = os.getpid() if pid is None else pid
    now = time.time() if now is None else now
    basetemp = Path(str(basetemp))
    removed = []
    try:
        entries = list(basetemp.parent.iterdir())
    except OSError:
        return removed
    for entry in entries:
        if entry.name == basetemp.name + '.pytest-owner':
            continue                       # ours, freshly written by claim()
        if not entry.name.endswith('.pytest-owner'):
            continue
        other_pid, started = _read(entry)
        if other_pid == pid:
            continue
        if started is not None and (now - started) <= STALE_AFTER_SECONDS:
            continue                       # a LIVE run - leave it alone
        target = entry.parent / entry.name[:-len('.pytest-owner')]
        if target == basetemp:
            continue
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            entry.unlink()
            removed.append(target.name)
        except OSError:
            continue
    return removed
