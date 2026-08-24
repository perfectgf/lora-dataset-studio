"""🧹 Forget missing — drop the rows whose source file is really gone.

The folder-sync warning has TWO honest causes and used to offer one remedy.
A folder that MOVED keeps its files: 📦 Move folder… repoints the bank and
loses nothing. But when something deleted files IN PLACE — a downloader that
cleans up its own intermediates, a sync client, a by-hand tidy — the bank kept
their rows for ever: failing to load, muddying the counters, and consuming the
bank's image ceiling, with no button that lets go of them.

The dangerous half is the walk: an unplugged drive makes EVERY file read as
missing, so a forget that trusted that walk would erase a whole triage in one
click. Hence fail closed — an unavailable folder refuses the whole operation
and deletes nothing, and the count shown before the click comes from a walk
done NOW, never from the banner's possibly-stale cache.
"""
import os

from PIL import Image


def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def photo(size=128, seed=0):
    """Smooth structure — a stable dHash and non-degenerate quality metrics."""
    im = Image.new('L', (size, size))
    c, r2 = size / 2, (size / 3) ** 2
    im.putdata([min(255, int(150 * x / size + 50 * y / size) + seed
                    + (80 if (x - c) ** 2 + (y - c) ** 2 < r2 else 0))
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def _mkbank(client, root, rels, name='B'):
    src = root / 'src'
    for i, rel in enumerate(rels):
        _save(str(src / rel), photo(seed=i * 7))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _rows(app, bank_id):
    from app.models import BankImage
    with app.app_context():
        return sorted((r.relpath.replace('\\', '/'), r.status)
                      for r in BankImage.query.filter_by(bank_id=bank_id))


def _set_status(app, bank_id, rel, status):
    from app.models import BankImage, db
    with app.app_context():
        for r in BankImage.query.filter_by(bank_id=bank_id):
            if r.relpath.replace('\\', '/') == rel:
                r.status = status
        db.session.commit()


def test_preview_counts_missing_without_deleting(client, app, tmp_path):
    """{} is a report, not an action: fresh missing/present counts plus a
    recognisable sample, and every row still there afterwards."""
    rels = ['a.png', 'b.png', 'sub/c.png', 'd.png']
    bank_id, src = _mkbank(client, tmp_path, rels)
    os.remove(str(src / 'b.png'))
    os.remove(str(src / 'sub' / 'c.png'))

    r = client.post(f'/api/bank/{bank_id}/forget-missing', json={})
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d['applied'] is False
    assert (d['missing'], d['present']) == (2, 2), d
    sample = sorted(rel.replace('\\', '/') for rel in d['missing_sample'])
    assert sample == ['b.png', 'sub/c.png'], d

    assert len(_rows(app, bank_id)) == len(rels)


def test_confirm_drops_only_the_missing_rows(client, app, tmp_path):
    """The forget itself: exactly the rows whose file is gone disappear, the
    survivors keep their decisions untouched, and the folder-sync note stops
    reporting the ghosts (its cache is invalidated, not left to a cooldown)."""
    rels = [f'p{i}.png' for i in range(5)]
    bank_id, src = _mkbank(client, tmp_path, rels)
    _set_status(app, bank_id, 'p0.png', 'keep')     # survivor with a decision
    _set_status(app, bank_id, 'p3.png', 'reject')   # ghost-to-be with one too
    os.remove(str(src / 'p3.png'))
    os.remove(str(src / 'p4.png'))

    r = client.post(f'/api/bank/{bank_id}/forget-missing', json={'confirm': True})
    assert r.status_code == 200, r.get_json()
    d = r.get_json()
    assert d['applied'] is True
    assert (d['removed'], d['remaining']) == (2, 3), d

    rows = _rows(app, bank_id)
    assert [rel for rel, _ in rows] == ['p0.png', 'p1.png', 'p2.png'], rows
    assert dict(rows)['p0.png'] == 'keep', rows

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert (payload.get('folder_sync') or {}).get('missing') == 0, payload.get('folder_sync')


def test_unavailable_folder_refuses_and_deletes_nothing(client, app, tmp_path):
    """THE guard. A folder that cannot be walked reads as "everything is
    missing" — confirm must refuse in block instead of erasing the triage."""
    rels = [f'g{i}.png' for i in range(4)]
    bank_id, src = _mkbank(client, tmp_path, rels)
    os.rename(str(src), str(tmp_path / 'elsewhere'))

    for body in ({}, {'confirm': True}):
        r = client.post(f'/api/bank/{bank_id}/forget-missing', json=body)
        assert r.status_code == 400, r.get_json()
        assert 'unavailable' in r.get_json()['error'], r.get_json()

    assert len(_rows(app, bank_id)) == len(rels)


def test_nothing_missing_is_a_calm_zero(client, app, tmp_path):
    """A folder in sync forgets nothing and says so with numbers, both steps."""
    bank_id, _src = _mkbank(client, tmp_path, ['x.png', 'y.png'])

    d = client.post(f'/api/bank/{bank_id}/forget-missing', json={}).get_json()
    assert (d['missing'], d['present']) == (0, 2), d
    d = client.post(f'/api/bank/{bank_id}/forget-missing',
                    json={'confirm': True}).get_json()
    assert (d['removed'], d['remaining']) == (0, 2), d
    assert len(_rows(app, bank_id)) == 2


def test_unknown_bank_is_404(client):
    r = client.post('/api/bank/424242/forget-missing', json={})
    assert r.status_code == 404
