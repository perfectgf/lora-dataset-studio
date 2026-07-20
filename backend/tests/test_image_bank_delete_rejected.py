"""🗃️ Image bank — 🗑 "Delete rejected from disk".

This is the ONLY bank action that writes to the source folder, so the tests are
paranoid: rejected files must actually leave the disk, NON-rejected files must
survive byte-for-byte, rows must be dropped, per-file failures must be reported
without aborting the batch, and nothing outside the bank folder is ever touched.

Both removal modes are exercised deterministically by injecting a fake / absent
``send2trash`` module (real send2trash would move test files to the OS recycle
bin — a side effect and a non-determinism we don't want in the suite)."""
import os
import sys
import types

from PIL import Image


def _flat(value=128):
    return Image.new('RGB', (32, 32), (value, value, value))


def _mkbank(client, tmp_path, names):
    src = tmp_path / 'src'
    for rel in names:
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        _flat().save(str(p))
    r = client.post('/api/bank/create', json={'name': 'B', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _reject(client, bank_id, ids):
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': ids, 'status': 'reject'})


def _by_name(client, bank_id):
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    return {i['name']: i for i in imgs}


def _force_trash(monkeypatch):
    """Make the service take the send2trash path but route it to os.remove, so
    files really leave the source folder without touching the OS recycle bin."""
    fake = types.SimpleNamespace(send2trash=lambda p: os.remove(p))
    monkeypatch.setitem(sys.modules, 'send2trash', fake)


def _force_hard_delete(monkeypatch):
    """Make ``from send2trash import send2trash`` fail → the os.remove fallback."""
    monkeypatch.setitem(sys.modules, 'send2trash', None)


# --- the core contract: rejected gone, everything else intact ----------------
def test_delete_rejected_removes_only_rejected_files(client, tmp_path, monkeypatch):
    _force_trash(monkeypatch)
    bank_id, src = _mkbank(client, tmp_path, ['keep.jpg', 'undecided.jpg',
                                              'bad1.jpg', 'bad2.jpg'])
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['keep.jpg']['id']], 'status': 'keep'})
    _reject(client, bank_id, [by['bad1.jpg']['id'], by['bad2.jpg']['id']])

    r = client.post(f'/api/bank/{bank_id}/delete-rejected', json={})
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out['mode'] == 'trash'
    assert out['trashed'] == 2 and out['deleted'] == 0
    assert out['rows_removed'] == 2 and out['skipped'] == []

    # STRONG filesystem assertions — the whole point of the feature.
    assert not (src / 'bad1.jpg').exists()
    assert not (src / 'bad2.jpg').exists()
    assert (src / 'keep.jpg').exists()          # non-rejected untouched
    assert (src / 'undecided.jpg').exists()     # non-rejected untouched

    # Rows for the deleted files are gone; the survivors remain.
    names = set(_by_name(client, bank_id))
    assert names == {'keep.jpg', 'undecided.jpg'}


def test_hard_delete_fallback_when_send2trash_absent(client, tmp_path, monkeypatch):
    _force_hard_delete(monkeypatch)
    bank_id, src = _mkbank(client, tmp_path, ['keep.jpg', 'bad.jpg'])
    by = _by_name(client, bank_id)
    _reject(client, bank_id, [by['bad.jpg']['id']])

    out = client.post(f'/api/bank/{bank_id}/delete-rejected', json={}).get_json()
    assert out['mode'] == 'delete'
    assert out['deleted'] == 1 and out['trashed'] == 0
    assert not (src / 'bad.jpg').exists()
    assert (src / 'keep.jpg').exists()


def test_delete_rejected_noop_when_none_rejected(client, tmp_path, monkeypatch):
    _force_trash(monkeypatch)
    bank_id, src = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    out = client.post(f'/api/bank/{bank_id}/delete-rejected', json={}).get_json()
    assert out['rows_removed'] == 0
    assert out['deleted'] == 0 and out['trashed'] == 0 and out['already_absent'] == 0
    assert (src / 'a.jpg').exists() and (src / 'b.jpg').exists()


def test_already_absent_file_is_skipped_not_crashed(client, tmp_path, monkeypatch):
    _force_trash(monkeypatch)
    bank_id, src = _mkbank(client, tmp_path, ['keep.jpg', 'gone.jpg', 'bad.jpg'])
    by = _by_name(client, bank_id)
    _reject(client, bank_id, [by['gone.jpg']['id'], by['bad.jpg']['id']])
    # Someone deleted this one out from under us before we ran.
    os.remove(src / 'gone.jpg')

    out = client.post(f'/api/bank/{bank_id}/delete-rejected', json={}).get_json()
    assert out['trashed'] == 1              # bad.jpg
    assert out['already_absent'] == 1       # gone.jpg — reported, not a crash
    # Both rejected rows are cleaned (their files are gone either way).
    assert out['rows_removed'] == 2
    assert set(_by_name(client, bank_id)) == {'keep.jpg'}
    assert (src / 'keep.jpg').exists()


def test_permission_error_is_reported_and_batch_continues(client, tmp_path, monkeypatch):
    bank_id, src = _mkbank(client, tmp_path, ['bad1.jpg', 'bad2.jpg'])
    by = _by_name(client, bank_id)
    _reject(client, bank_id, [by['bad1.jpg']['id'], by['bad2.jpg']['id']])

    # bad1 refuses to delete (locked / read-only), bad2 goes through.
    real_remove = os.remove

    def flaky(p):
        if p.replace('\\', '/').endswith('bad1.jpg'):
            raise PermissionError('file in use')
        return real_remove(p)

    fake = types.SimpleNamespace(send2trash=flaky)
    monkeypatch.setitem(sys.modules, 'send2trash', fake)

    out = client.post(f'/api/bank/{bank_id}/delete-rejected', json={}).get_json()
    assert out['trashed'] == 1
    assert len(out['skipped']) == 1
    assert out['skipped'][0]['relpath'] == 'bad1.jpg'
    # The failed file keeps its row AND its file; the other is gone.
    assert out['rows_removed'] == 1
    assert (src / 'bad1.jpg').exists()
    assert not (src / 'bad2.jpg').exists()
    assert 'bad1.jpg' in set(_by_name(client, bank_id))


def test_delete_rejected_409_when_job_running(client, tmp_path, monkeypatch):
    _force_trash(monkeypatch)
    bank_id, src = _mkbank(client, tmp_path, ['bad.jpg'])
    by = _by_name(client, bank_id)
    _reject(client, bank_id, [by['bad.jpg']['id']])

    from app.services import bank_jobs
    monkeypatch.setattr(bank_jobs, 'running', lambda _bid: True)
    r = client.post(f'/api/bank/{bank_id}/delete-rejected', json={})
    assert r.status_code == 409
    assert (src / 'bad.jpg').exists()       # nothing deleted while a job runs


def test_delete_rejected_404_for_missing_bank(client, monkeypatch):
    _force_trash(monkeypatch)
    r = client.post('/api/bank/999999/delete-rejected', json={})
    assert r.status_code == 404


def test_delete_rejected_never_escapes_bank_folder(client, tmp_path, monkeypatch):
    _force_trash(monkeypatch)
    bank_id, src = _mkbank(client, tmp_path, ['bad.jpg'])
    by = _by_name(client, bank_id)
    _reject(client, bank_id, [by['bad.jpg']['id']])
    # A file OUTSIDE the bank folder that a poisoned relpath might point at.
    outside = tmp_path / 'precious.txt'
    outside.write_text('do not touch')

    from app.models import BankImage
    from app.extensions import db
    row = BankImage.query.filter_by(bank_id=bank_id).first()
    row.relpath = os.path.join('..', 'precious.txt')     # escape attempt
    db.session.commit()

    out = client.post(f'/api/bank/{bank_id}/delete-rejected', json={}).get_json()
    assert out['skipped'] and out['skipped'][0]['reason'] == 'unsafe_path'
    assert out['rows_removed'] == 0
    assert outside.exists()                 # the escaping path was refused
