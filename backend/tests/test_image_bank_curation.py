"""🗃️ Image bank — CURATION selectors (turn a big dump into a good LoRA subset).

Two pure-CPU selectors that reuse the CLIP embeddings the ✨ Score pass cached
(no GPU, no re-scan — same contract as the semantic-dedup stage):

  • select-diverse  — farthest-point sampling: the N images that best COVER the
                      visual space (the antidote to 4 000 near-identical shots);
  • select-similar  — rank by cosine similarity to a REFERENCE bank image
                      ("keep what looks like THIS").

Both only ever return a SELECTION (a set of image ids the UI checks) — nothing
is mutated or deleted. We seed a synthetic score_cache.npz (as the scoring
subprocess would) so grouping runs WITHOUT torch. Background jobs run inline
under TESTING; these selectors are synchronous.
"""
import os

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


# --- factories (mirror the semantic-dedup suite) -----------------------------
def _flat(size=64, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, names, name='B'):
    src = tmp_path / 'src'
    for rel in names:
        os.makedirs(os.path.dirname(str(src / rel)), exist_ok=True)
        _flat(value=(hash(rel) % 200) + 20).save(str(src / rel))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


def _emb(*coords):
    """A 768-dim L2-normed vector with the given leading coordinates."""
    v = np.zeros(768, dtype='float32')
    v[:len(coords)] = coords
    v /= (np.linalg.norm(v) + 1e-8)
    return v


def _write_score_cache(app, bank_id, embs_by_name, state='ok'):
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        paths, states, arr, sigs = [], [], [], []
        for nm, e in embs_by_name.items():
            r = rows[nm]
            p = banks.abs_image_path(bank, r)
            paths.append(p)
            states.append(state)
            arr.append(np.asarray(e, dtype='float32'))
            st = os.stat(p)
            sigs.append(f'{st.st_size}:{st.st_mtime_ns}')
        cache_path = banks._score_cache_path(bank_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            paths=np.array(paths), states=np.array(states),
            aes=np.array([float('nan')] * len(paths), dtype='float32'),
            nsfw=np.array([float('nan')] * len(paths), dtype='float32'),
            embs=np.stack(arr).astype('float32'), sigs=np.array(sigs))


def _id_of(app, bank_id, name):
    with app.app_context():
        from app.models import BankImage
        rows = {os.path.basename(r.relpath): r.id
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        return rows[name]


def _names_of(app, bank_id, ids):
    with app.app_context():
        from app.models import BankImage
        by_id = {r.id: os.path.basename(r.relpath)
                 for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        return {by_id[i] for i in ids}


# --- diversity (farthest-point sampling) -------------------------------------
def test_diverse_picks_one_per_cluster(client, tmp_path, app):
    """4 tight clusters of near-identical embeddings; asking for 4 diverse images
    picks ONE from each cluster — never 4 near-duplicates of one cluster."""
    names = []
    embs = {}
    # cluster c along axis c, 3 members each with a tiny wobble.
    for c in range(4):
        for k in range(3):
            nm = f'c{c}_{k}.jpg'
            names.append(nm)
            base = [0.0, 0.0, 0.0, 0.0]
            base[c] = 1.0
            base[(c + 1) % 4] = 0.02 * k        # small intra-cluster wobble
            embs[nm] = _emb(*base)
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    r = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 4})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['pool'] == 12 and len(body['image_ids']) == 4
    picked = _names_of(app, bank_id, body['image_ids'])
    clusters = {nm.split('_')[0] for nm in picked}
    assert clusters == {'c0', 'c1', 'c2', 'c3'}     # one per cluster, full coverage


def test_diverse_is_deterministic(client, tmp_path, app):
    """Same pool + same n ⇒ byte-identical selection (lowest-id seed, id tie-break)."""
    names = [f'i{k}.jpg' for k in range(8)]
    embs = {nm: _emb(*np.random.RandomState(k).randn(5)) for k, nm in enumerate(names)}
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    a = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 5}).get_json()
    b = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 5}).get_json()
    assert a['image_ids'] == b['image_ids']
    assert a['image_ids'] == sorted(a['image_ids'])   # returned sorted


def test_diverse_whole_pool_when_n_exceeds(client, tmp_path, app):
    names = ['a.jpg', 'b.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0), 'b.jpg': _emb(0.0, 1.0)})
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 50}).get_json()
    assert body['pool'] == 2 and len(body['image_ids']) == 2


def test_diverse_hint_when_no_embeddings(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    r = client.post(f'/api/bank/{bank_id}/select-diverse', json={'n': 4})
    assert r.status_code == 400
    assert 'Score first' in r.get_json()['error']


def test_diverse_composes_with_filter_and_excludes_rejects(client, tmp_path, app):
    """The pool honours the grid filter (here a style_cluster) AND, with no status
    filter, drops rejected rows — you curate from what you might keep."""
    names = [f'i{k}.jpg' for k in range(6)]
    embs = {nm: _emb(*np.random.RandomState(k + 100).randn(5)) for k, nm in enumerate(names)}
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, embs)
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(), key=lambda r: r.id)
        rows[0].style_cluster = 1                 # only i0..i2 in style 1
        rows[1].style_cluster = 1
        rows[2].style_cluster = 1
        rows[2].status = 'reject'                 # i2 rejected → out of the pool
        db.session.commit()
    body = client.post(f'/api/bank/{bank_id}/select-diverse',
                       json={'n': 10, 'style': 1}).get_json()
    assert body['pool'] == 2                       # i0, i1 (i2 rejected)
    picked = _names_of(app, bank_id, body['image_ids'])
    assert picked == {'i0.jpg', 'i1.jpg'}


# --- reference similarity ----------------------------------------------------
def test_similar_ranks_by_cosine_to_reference(client, tmp_path, app):
    """Top-N most similar to the reference are its near-neighbours; the reference
    itself is always included (cosine 1.0), the far image is excluded at N=2."""
    names = ['ref.jpg', 'near.jpg', 'mid.jpg', 'far.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'ref.jpg':  _emb(1.0, 0.0),
        'near.jpg': _emb(0.98, np.sqrt(1 - 0.98 ** 2)),
        'mid.jpg':  _emb(0.80, np.sqrt(1 - 0.80 ** 2)),
        'far.jpg':  _emb(0.10, np.sqrt(1 - 0.10 ** 2))})
    ref_id = _id_of(app, bank_id, 'ref.jpg')
    body = client.post(f'/api/bank/{bank_id}/select-similar',
                       json={'ref_id': ref_id, 'n': 2}).get_json()
    assert _names_of(app, bank_id, body['image_ids']) == {'ref.jpg', 'near.jpg'}
    # results are score-ranked, reference first at cosine 1.0.
    assert body['results'][0]['id'] == ref_id
    assert body['results'][0]['score'] == pytest.approx(1.0, abs=1e-3)
    assert body['results'][1]['score'] > body['results'][-1]['score'] \
        if len(body['results']) > 2 else True


def test_similar_threshold_mode(client, tmp_path, app):
    """min_score keeps everything at/above the cosine cut, whatever the count."""
    names = ['ref.jpg', 'near.jpg', 'far.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    _write_score_cache(app, bank_id, {
        'ref.jpg':  _emb(1.0, 0.0),
        'near.jpg': _emb(0.95, np.sqrt(1 - 0.95 ** 2)),
        'far.jpg':  _emb(0.50, np.sqrt(1 - 0.50 ** 2))})
    ref_id = _id_of(app, bank_id, 'ref.jpg')
    body = client.post(f'/api/bank/{bank_id}/select-similar',
                       json={'ref_id': ref_id, 'min_score': 0.9}).get_json()
    assert _names_of(app, bank_id, body['image_ids']) == {'ref.jpg', 'near.jpg'}


def test_similar_requires_ref_id(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0)})
    r = client.post(f'/api/bank/{bank_id}/select-similar', json={'n': 4})
    assert r.status_code == 400
    assert 'ref_id' in r.get_json()['error']


def test_similar_hint_when_no_embeddings(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    ref_id = _id_of(app, bank_id, 'a.jpg')
    r = client.post(f'/api/bank/{bank_id}/select-similar',
                    json={'ref_id': ref_id, 'n': 2})
    assert r.status_code == 400
    assert 'Score first' in r.get_json()['error']


def test_similar_ref_without_embedding_is_400(client, tmp_path, app):
    """A reference that has no cached embedding (e.g. never scored) gets a clear
    error, not a silent empty selection."""
    names = ['ref.jpg', 'a.jpg', 'b.jpg']
    bank_id, _ = _mkbank(client, tmp_path, names)
    # Cache covers a/b but NOT ref.
    _write_score_cache(app, bank_id, {'a.jpg': _emb(1.0, 0.0),
                                      'b.jpg': _emb(0.0, 1.0)})
    ref_id = _id_of(app, bank_id, 'ref.jpg')
    r = client.post(f'/api/bank/{bank_id}/select-similar',
                    json={'ref_id': ref_id, 'n': 2})
    assert r.status_code == 400
    assert 'embedding' in r.get_json()['error']
