"""The .npz cache write survives a transient lock, and never strands its work.

WHY THESE TESTS DO NOT OPEN A FILE TO CREATE THE LOCK
The failure is Windows-only in the wild — POSIX lets you rename over an open
destination — but the CODE must be right everywhere, and a suite that only
proves itself on one OS proves nothing on the machines CI runs. So the lock is
injected by making ``os.replace`` raise the exact error Windows raises. That
also lets a test control precisely HOW MANY refusals precede the success, which
is the property under test: "it retried", not "it happened to work".

The salvage tests build real archives with numpy and truncate one on purpose,
because the thing being asserted — that a file cut mid-write is REJECTED — is a
property of decompressing it, not of any flag we could stub.
"""
import os
import pathlib
import sys

import pytest

np = pytest.importorskip('numpy')

INFER = pathlib.Path(__file__).resolve().parents[1] / 'infer'
if str(INFER) not in sys.path:
    sys.path.append(str(INFER))
import npz_atomic  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """The retry budget is ~9.5 s of wall clock. Record the naps, never take them."""
    taken = []
    monkeypatch.setattr(npz_atomic, '_sleep', taken.append)
    return taken


def _win_error(code):
    error = PermissionError(code, 'Access is denied')
    error.winerror = code
    return error


def _refuse_n_times(monkeypatch, count, code=5):
    """Make os.replace fail with a Windows "held by someone else" error `count`
    times, then behave. Returns the list of attempts for counting."""
    real = os.replace
    attempts = []

    def flaky(src, dst, *args, **kwargs):
        attempts.append((str(src), str(dst)))
        if len(attempts) <= count:
            raise _win_error(code)
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(npz_atomic.os, 'replace', flaky)
    return attempts


def _write_npz(path, count, key='paths'):
    np.savez_compressed(str(path), **{key: np.arange(count)})


# --- half one: the replace survives a transient lock ------------------------------
def test_save_survives_a_lock_that_clears(tmp_path, monkeypatch, _no_real_sleeping):
    destination = tmp_path / 'score_cache.npz'
    attempts = _refuse_n_times(monkeypatch, 3)

    npz_atomic.save_npz_atomic(destination, {'paths': np.arange(7)})

    assert len(attempts) == 4                      # 3 refusals, then through
    assert _no_real_sleeping == [0.1, 0.2, 0.4]    # the backoff, in order
    with np.load(str(destination)) as z:
        assert len(z['paths']) == 7
    # And it left nothing behind.
    assert not npz_atomic.orphan_temporaries(destination)


def test_a_lock_that_never_clears_names_the_cause_and_keeps_the_work(
        tmp_path, monkeypatch, _no_real_sleeping):
    destination = tmp_path / 'score_cache.npz'
    attempts = _refuse_n_times(monkeypatch, 999)

    with pytest.raises(npz_atomic.NpzReplaceLocked) as caught:
        npz_atomic.save_npz_atomic(destination, {'paths': np.arange(50)})

    assert len(attempts) == len(npz_atomic.REPLACE_DELAYS) + 1
    message = str(caught.value)
    # The message must send the user after the HOLDER, not after folder
    # permissions — "access denied" is what sent them there.
    assert 'another program is holding it open' in message
    assert 'antivirus' in message
    assert 'Nothing was lost' in message
    assert 'access denied' not in message.lower()
    # The finished archive is still on disk, under the name salvage looks for.
    orphans = npz_atomic.orphan_temporaries(destination)
    assert orphans and orphans[0] == caught.value.temporary
    with np.load(orphans[0]) as z:
        assert len(z['paths']) == 50


def test_a_sharing_violation_is_retried_too(tmp_path, monkeypatch):
    """WinError 32 is the same story ("someone holds it") from another layer."""
    destination = tmp_path / 'cache.npz'
    attempts = _refuse_n_times(monkeypatch, 2, code=32)
    npz_atomic.save_npz_atomic(destination, {'paths': np.arange(3)})
    assert len(attempts) == 3


def test_an_error_that_is_not_a_lock_is_not_retried(tmp_path, monkeypatch):
    """A bad path must fail immediately: retrying it only delays an honest error,
    and the temporary is cleaned up because it is litter, not a fallback."""
    destination = tmp_path / 'cache.npz'
    attempts = []

    def broken(src, dst, *args, **kwargs):
        attempts.append(src)
        raise OSError(22, 'Invalid argument')

    monkeypatch.setattr(npz_atomic.os, 'replace', broken)
    with pytest.raises(OSError) as caught:
        npz_atomic.save_npz_atomic(destination, {'paths': np.arange(3)})
    assert not isinstance(caught.value, npz_atomic.NpzReplaceLocked)
    assert len(attempts) == 1
    assert not npz_atomic.orphan_temporaries(destination)


def test_two_writers_never_share_a_temporary(tmp_path):
    destination = tmp_path / 'cache.npz'
    names = {npz_atomic.temporary_name(destination) for _ in range(50)}
    assert len(names) == 50
    assert all(name.endswith('.tmp.npz') for name in names)


# --- half two: the orphan is salvaged, or refused ---------------------------------
def _count(path):
    with np.load(str(path), allow_pickle=False) as z:
        return int(len(z['paths']))


def test_an_orphan_with_more_entries_is_promoted(tmp_path):
    destination = tmp_path / 'score_cache.npz'
    _write_npz(destination, 1800)
    _write_npz(str(destination) + '.tmp.npz', 1850)     # the old fixed name

    said = []
    recovered = npz_atomic.salvage_orphan_tmp(destination, _count, said.append)

    assert recovered == 1850
    assert _count(destination) == 1850
    assert not npz_atomic.orphan_temporaries(destination)
    assert any('recovered 1850 entries' in line for line in said)


def test_an_orphan_truncated_mid_write_is_refused_and_deleted(tmp_path):
    """The dangerous case: a run killed inside savez_compressed. The header is
    already there, so the file OPENS; only decompressing a member catches it."""
    destination = tmp_path / 'score_cache.npz'
    _write_npz(destination, 100)
    orphan = pathlib.Path(str(destination) + '.tmp.npz')
    _write_npz(orphan, 5000)
    whole = orphan.read_bytes()
    orphan.write_bytes(whole[:len(whole) // 2])         # cut mid-archive

    said = []
    recovered = npz_atomic.salvage_orphan_tmp(destination, _count, said.append)

    assert recovered == 0
    assert _count(destination) == 100                   # the good cache is intact
    assert not orphan.exists()
    assert any('discarded a leftover temporary' in line for line in said)


def test_an_orphan_with_fewer_entries_is_refused(tmp_path):
    destination = tmp_path / 'score_cache.npz'
    _write_npz(destination, 900)
    _write_npz(str(destination) + '.tmp.npz', 400)

    said = []
    assert npz_atomic.salvage_orphan_tmp(destination, _count, said.append) == 0
    assert _count(destination) == 900
    assert any('fewer than the 900 already cached' in line for line in said)


def test_the_newest_good_orphan_wins_and_the_rest_are_swept(tmp_path):
    """Writers that already used unique temporary names can leave several."""
    destination = tmp_path / 'semantic_cache.npz'
    _write_npz(destination, 10)
    older = tmp_path / 'semantic_cache.npz.111-aaa.tmp.npz'
    newer = tmp_path / 'semantic_cache.npz.222-bbb.tmp.npz'
    _write_npz(older, 300)
    _write_npz(newer, 200)
    os.utime(older, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    os.utime(newer, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))

    recovered = npz_atomic.salvage_orphan_tmp(destination, _count)

    # 200, not 300: the newest that passes wins. An older file with more rows is
    # an older SNAPSHOT, and promoting it would undo a newer run's work.
    assert recovered == 200
    assert _count(destination) == 200
    assert not older.exists()
    assert not npz_atomic.orphan_temporaries(destination)


def test_salvage_with_no_cache_in_place_still_promotes(tmp_path):
    destination = tmp_path / 'score_cache.npz'
    _write_npz(str(destination) + '.tmp.npz', 42)
    assert npz_atomic.salvage_orphan_tmp(destination, _count) == 42
    assert _count(destination) == 42


def test_salvage_leaves_an_unrelated_file_alone(tmp_path):
    destination = tmp_path / 'score_cache.npz'
    _write_npz(destination, 5)
    sidecar = tmp_path / 'score_cache.npz.count'
    sidecar.write_text('5', encoding='utf-8')
    other = tmp_path / 'face_cache.npz.tmp.npz'
    _write_npz(other, 99)

    assert npz_atomic.salvage_orphan_tmp(destination, _count) == 0
    assert sidecar.exists() and other.exists()
    assert _count(destination) == 5
