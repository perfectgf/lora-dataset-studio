"""The stale-basetemp sweep — scoped to what the guard provably owns.

The accumulation this closes was lived, not imagined: killed runs never
release their claim, ~45 basetemps once piled up 105 GB, and under 10 GB
free the training preflight refuses launches — five tests go red looking
exactly like a merge regression. The sweep's danger is the opposite one:
the scratch parent is SHARED, so it must never touch anything the guard
did not mark itself.
"""
import _basetemp_guard as guard

NOW = 1_000_000.0
STALE = NOW - guard.STALE_AFTER_SECONDS - 1
FRESH = NOW - 60


def _mk(tmp_path, name, *, claim_at=None, claim_pid=4242, with_dir=True):
    if with_dir:
        d = tmp_path / name
        d.mkdir()
        (d / 'gw0').mkdir()
        (d / 'gw0' / 'blob.bin').write_bytes(b'x' * 16)
    if claim_at is not None:
        (tmp_path / (name + '.pytest-owner')).write_text(
            f'{claim_pid} {claim_at}\n', encoding='utf-8')


def test_a_stale_claims_basetemp_goes_directory_and_claim_together(tmp_path):
    _mk(tmp_path, 'dead', claim_at=STALE)
    mine = tmp_path / 'mine'
    removed = guard.sweep_stale_siblings(mine, pid=1, now=NOW)
    assert removed == ['dead']
    assert not (tmp_path / 'dead').exists()
    assert not (tmp_path / 'dead.pytest-owner').exists()


def test_a_live_run_is_never_touched(tmp_path):
    _mk(tmp_path, 'alive', claim_at=FRESH)
    guard.sweep_stale_siblings(tmp_path / 'mine', pid=1, now=NOW)
    assert (tmp_path / 'alive').exists()
    assert (tmp_path / 'alive.pytest-owner').exists()


def test_a_directory_without_a_claim_is_foreign_data_and_stays(tmp_path):
    # The scratch parent is shared: only the guard's own marker authorizes a
    # deletion. No claim file = not provably ours = untouchable.
    _mk(tmp_path, 'someone-elses-folder', claim_at=None)
    removed = guard.sweep_stale_siblings(tmp_path / 'mine', pid=1, now=NOW)
    assert removed == []
    assert (tmp_path / 'someone-elses-folder').exists()


def test_an_orphan_claim_whose_directory_is_gone_is_dropped(tmp_path):
    _mk(tmp_path, 'ghost', claim_at=STALE, with_dir=False)
    removed = guard.sweep_stale_siblings(tmp_path / 'mine', pid=1, now=NOW)
    assert removed == ['ghost']
    assert not (tmp_path / 'ghost.pytest-owner').exists()


def test_our_own_claim_and_basetemp_survive_the_sweep(tmp_path):
    # claim() has just written our marker when the sweep runs from configure —
    # a sweep that ate its own caller would be the guard defeating itself.
    mine = tmp_path / 'mine'
    _mk(tmp_path, 'mine', claim_at=STALE, claim_pid=1)   # OUR pid, however old
    removed = guard.sweep_stale_siblings(mine, pid=1, now=NOW)
    assert removed == []
    assert mine.exists()
    assert (tmp_path / 'mine.pytest-owner').exists()


def test_an_unreadable_claim_counts_as_stale_and_is_collected(tmp_path):
    # _read() answers (None, None) for garbage; a marker nobody can parse
    # guards nothing and the directory behind it is a dead run's.
    _mk(tmp_path, 'garbled', claim_at=None)
    (tmp_path / 'garbled.pytest-owner').write_text('not a claim', encoding='utf-8')
    removed = guard.sweep_stale_siblings(tmp_path / 'mine', pid=1, now=NOW)
    assert removed == ['garbled']
    assert not (tmp_path / 'garbled').exists()


def test_a_missing_parent_never_raises(tmp_path):
    assert guard.sweep_stale_siblings(tmp_path / 'nope' / 'mine', pid=1, now=NOW) == []
