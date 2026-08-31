"""Claiming a just-written ComfyUI output must survive a locked source.

Completion used to shutil.move ComfyUI output into the dataset dir. A
rename cannot span volumes, so that is copy + unlink; unlink then hit WinError 32
while ComfyUI still held the PNG, and the tile stayed pending even though
dest already had the bytes. Dest present is success; source unlink is
best-effort.
"""
import os

from app.utils import comfy_fs


def _sharing_violation(msg='used by another process'):
    err = PermissionError(32, msg)
    err.winerror = 32
    return err


def test_claim_output_file_copies_and_removes_source(tmp_path):
    src = tmp_path / 'output' / 'shot.png'
    dst = tmp_path / 'datasets' / '14' / 'shot.png'
    src.parent.mkdir(parents=True)
    src.write_bytes(b'PNGDATA')

    assert comfy_fs.claim_output_file(src, dst) is True
    assert dst.read_bytes() == b'PNGDATA'
    assert not src.exists()


def test_claim_output_file_succeeds_when_source_stays_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_fs, '_OUTPUT_LOCK_RETRY_DELAY', 0)
    src = tmp_path / 'output' / 'locked.png'
    dst = tmp_path / 'datasets' / '14' / 'locked.png'
    src.parent.mkdir(parents=True)
    src.write_bytes(b'PNGDATA')

    def locked_unlink(path):
        if os.path.normpath(path) == os.path.normpath(src):
            raise _sharing_violation()
        raise AssertionError(f'unexpected unlink: {path}')

    monkeypatch.setattr(comfy_fs.os, 'unlink', locked_unlink)

    assert comfy_fs.claim_output_file(src, dst) is True
    assert dst.read_bytes() == b'PNGDATA'
    assert src.exists()  # leftover in ComfyUI output; tile still done


def test_claim_output_file_retries_unlink_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_fs, '_OUTPUT_LOCK_RETRY_DELAY', 0)
    src = tmp_path / 'output' / 'retry.png'
    dst = tmp_path / 'datasets' / '14' / 'retry.png'
    src.parent.mkdir(parents=True)
    src.write_bytes(b'PNGDATA')
    calls = {'n': 0}
    real_unlink = os.unlink

    def flaky_unlink(path):
        if os.path.normpath(path) != os.path.normpath(src):
            return real_unlink(path)
        calls['n'] += 1
        if calls['n'] < 2:
            raise _sharing_violation()
        return real_unlink(path)

    monkeypatch.setattr(comfy_fs.os, 'unlink', flaky_unlink)

    assert comfy_fs.claim_output_file(src, dst) is True
    assert dst.read_bytes() == b'PNGDATA'
    assert not src.exists()
    assert calls['n'] == 2


def test_claim_output_file_dest_already_there_without_src(tmp_path):
    dst = tmp_path / 'datasets' / '14' / 'already.png'
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b'PREVIOUS')
    ghost = tmp_path / 'output' / 'already.png'

    assert comfy_fs.claim_output_file(ghost, dst) is True
    assert dst.read_bytes() == b'PREVIOUS'


def test_claim_output_file_missing_both_returns_false(tmp_path):
    src = tmp_path / 'output' / 'gone.png'
    dst = tmp_path / 'datasets' / '14' / 'gone.png'
    assert comfy_fs.claim_output_file(src, dst) is False
    assert not dst.exists()


def test_claim_output_file_retries_a_locked_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_fs, '_OUTPUT_LOCK_RETRY_DELAY', 0)
    src = tmp_path / 'output' / 'busy.png'
    dst = tmp_path / 'datasets' / '14' / 'busy.png'
    src.parent.mkdir(parents=True)
    src.write_bytes(b'PNGDATA')
    calls = {'n': 0}
    real_copy2 = comfy_fs.shutil.copy2

    def flaky_copy2(source, dest, *a, **k):
        calls['n'] += 1
        if calls['n'] < 2:
            raise _sharing_violation()
        return real_copy2(source, dest, *a, **k)

    monkeypatch.setattr(comfy_fs.shutil, 'copy2', flaky_copy2)

    assert comfy_fs.claim_output_file(src, dst) is True
    assert dst.read_bytes() == b'PNGDATA'
    assert not src.exists()
    assert calls['n'] == 2


def test_claim_output_file_copy_failure_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_fs, '_OUTPUT_LOCK_RETRY_DELAY', 0)
    src = tmp_path / 'output' / 'nope.png'
    dst = tmp_path / 'datasets' / '14' / 'nope.png'
    src.parent.mkdir(parents=True)
    src.write_bytes(b'PNGDATA')
    monkeypatch.setattr(comfy_fs.shutil, 'copy2',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))

    assert comfy_fs.claim_output_file(src, dst) is False
    assert not dst.exists()
    assert src.exists()
