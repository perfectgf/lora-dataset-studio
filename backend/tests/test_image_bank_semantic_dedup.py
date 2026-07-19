"""🗃️ Image bank — stage-2 SEMANTIC near-duplicate dedup (crops / re-compressed
variants of the same shot the dHash misses). The heavy CLIP inference runs in the
dedicated scoring subprocess; here we exercise the pure-CPU stage-2 that reuses its
CACHED embeddings — grouping over synthetic embeddings, the stage-1 vs stage-2
distinction, threshold re-tri with no re-scan, the "run Score first" hint, the
pipeline step order + skip reason, resolution, and the score-cache staleness guard.
Background jobs run inline under TESTING (see bank_jobs.start)."""
import importlib.util
import os
import pathlib

import pytest
from PIL import Image

np = pytest.importorskip('numpy')


# --- factories ---------------------------------------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.jpg', '.jpeg')):
        im.save(path, 'JPEG', quality=92)
    else:
        im.save(path)


def _flat(size=128, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _photo(size=256):
    im = Image.new('L', (size, size))
    c, r2 = size / 2, (size / 3) ** 2
    im.putdata([min(255, int(150 * x / size + 50 * y / size)
                    + (80 if (x - c) ** 2 + (y - c) ** 2 < r2 else 0))
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _emb(*coords):
    """A 768-dim L2-normed vector with the given leading coordinates."""
    v = np.zeros(768, dtype='float32')
    v[:len(coords)] = coords
    v /= (np.linalg.norm(v) + 1e-8)
    return v


def _write_score_cache(app, bank_id, embs_by_name, state='ok', with_sig=True):
    """Fabricate the ✨ Score cache (score_cache.npz) directly, keyed by each row's
    absolute path — exactly what the scoring subprocess would have written — so
    stage-2 has embeddings to read WITHOUT any GPU/torch."""
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        paths, states, aes, nsfw, arr, sigs = [], [], [], [], [], []
        for name, e in embs_by_name.items():
            r = rows[name]
            p = banks.abs_image_path(bank, r)
            paths.append(p)
            states.append(state)
            aes.append(float('nan'))
            nsfw.append(float('nan'))
            arr.append(np.asarray(e, dtype='float32'))
            if with_sig:
                st = os.stat(p)
                sigs.append(f'{st.st_size}:{st.st_mtime_ns}')
            else:
                sigs.append('')
        cache_path = banks._score_cache_path(bank_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            paths=np.array(paths), states=np.array(states),
            aes=np.array(aes, dtype='float32'), nsfw=np.array(nsfw, dtype='float32'),
            embs=np.stack(arr).astype('float32'), sigs=np.array(sigs))


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


def _groups(app, bank_id):
    """{basename: semantic_dup_group} for every row."""
    with app.app_context():
        from app.models import BankImage
        return {os.path.basename(r.relpath): r.semantic_dup_group
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}


def _by_name(client, bank_id, **params):
    q = '&'.join(f'{k}={v}' for k, v in params.items())
    url = f'/api/bank/{bank_id}/images' + (f'?{q}' if q else '')
    return {i['name']: i for i in client.get(url).get_json()['images']}


# --- core grouping over synthetic embeddings ---------------------------------
def test_semantic_groups_same_shot_not_distinct_photos(client, tmp_path, app):
    """A shot + its crop (cosine 0.97) group; a distinct photo (0.85) stays out."""
    bank_id, _ = _mkbank(client, tmp_path, {
        'orig.jpg': _flat(value=10), 'crop.jpg': _flat(value=20),
        'other.jpg': _flat(value=30)})
    # orig↔crop = 0.97 (same shot, different framing); orig↔other = 0.85 (< 0.96).
    _write_score_cache(app, bank_id, {
        'orig.jpg': _emb(0.97, np.sqrt(1 - 0.97 ** 2), 0.0),
        'crop.jpg': _emb(1.0, 0.0, 0.0),
        'other.jpg': _emb(0.85, 0.0, np.sqrt(1 - 0.85 ** 2))})
    r = client.post(f'/api/bank/{bank_id}/semantic-dedup', json={})
    assert r.status_code == 202, r.get_json()
    g = _groups(app, bank_id)
    assert g['orig.jpg'] is not None and g['orig.jpg'] == g['crop.jpg']
    assert g['other.jpg'] is None
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['semantic_dup']['groups'] == 1
    assert payload['semantic_dup']['unresolved'] == 1


def test_stage1_and_stage2_are_distinct(client, tmp_path, app):
    """dHash (stage 1) groups an exact/resized copy; the semantic pass (stage 2)
    additionally groups a same-shot image dHash never linked."""
    big = _photo(256)
    small = big.resize((96, 96), Image.LANCZOS)     # same content → same dHash
    bank_id, _ = _mkbank(client, tmp_path, {
        'orig.jpg': big, 'resized.jpg': small, 'variant.jpg': _flat(value=200)})
    client.post(f'/api/bank/{bank_id}/scan', json={})   # stage 1 dHash groups
    by = _by_name(client, bank_id)
    # Stage 1: orig+resized share a dup_group; the flat 'variant' shares none.
    assert by['orig.jpg']['dup_group'] is not None
    assert by['orig.jpg']['dup_group'] == by['resized.jpg']['dup_group']
    assert by['variant.jpg']['dup_group'] != by['orig.jpg']['dup_group']
    # Stage 2: give all three the SAME-shot embedding — the variant, which dHash
    # never linked, now joins the semantic group.
    _write_score_cache(app, bank_id, {
        'orig.jpg': _emb(1.0), 'resized.jpg': _emb(0.99, np.sqrt(1 - 0.99 ** 2)),
        'variant.jpg': _emb(0.98, np.sqrt(1 - 0.98 ** 2))})
    client.post(f'/api/bank/{bank_id}/semantic-dedup', json={})
    g = _groups(app, bank_id)
    assert g['variant.jpg'] is not None
    assert g['orig.jpg'] == g['resized.jpg'] == g['variant.jpg']
    by = _by_name(client, bank_id)
    assert by['variant.jpg']['semantic_dup_group'] == by['orig.jpg']['semantic_dup_group']


def test_threshold_retri_without_rescan(client, tmp_path, app):
    """Re-running at a stricter threshold re-clusters the CACHED embeddings — the
    0.97 pair drops out at 0.99, with no re-scan / GPU work."""
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(10), 'b.jpg': _flat(20)})
    _write_score_cache(app, bank_id, {
        'a.jpg': _emb(1.0, 0.0), 'b.jpg': _emb(0.97, np.sqrt(1 - 0.97 ** 2))})
    client.post(f'/api/bank/{bank_id}/semantic-dedup', json={})
    assert _groups(app, bank_id)['a.jpg'] == _groups(app, bank_id)['b.jpg'] is not None
    # Stricter: 0.97 < 0.99 → no longer a near-dup, no group.
    client.post(f'/api/bank/{bank_id}/semantic-dedup', json={'threshold': 0.99})
    g = _groups(app, bank_id)
    assert g['a.jpg'] is None and g['b.jpg'] is None


def test_blocking_respects_config_fallback(app, tmp_path, client):
    """When style_threshold > semantic threshold the blocking guarantee can't hold,
    so grouping falls back to a single global block and still finds the pair."""
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(10), 'b.jpg': _flat(20)})
    _write_score_cache(app, bank_id, {
        'a.jpg': _emb(1.0, 0.0), 'b.jpg': _emb(0.97, np.sqrt(1 - 0.97 ** 2))})
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'bank': {'style_threshold': 0.99}})   # > 0.96 default
        from app.services import image_bank_service as banks
        n = banks.rebuild_semantic_dup_groups(bank_id)
    assert n == 1
    assert _groups(app, bank_id)['a.jpg'] == _groups(app, bank_id)['b.jpg']


# --- the "run Score first" hint ---------------------------------------------
def test_hint_when_no_embeddings(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat()})
    r = client.post(f'/api/bank/{bank_id}/semantic-dedup', json={})
    assert r.status_code == 400
    assert 'Score first' in r.get_json()['error']
    with app.app_context():
        from app.services import image_bank_service as banks
        assert banks.rebuild_semantic_dup_groups(bank_id) is None


# --- pipeline integration ----------------------------------------------------
def test_pipeline_step_order_and_skip_reason(client, tmp_path, app):
    from app.services import image_bank_service as banks
    steps = banks.PIPELINE_STEPS
    assert steps.index('semantic_dedup') == steps.index('score') + 1   # right after Score
    # A pipeline run of just semantic_dedup with no embeddings → skipped, with a
    # reason that names Score (never a mute ✗).
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat()})
    r = client.post(f'/api/bank/{bank_id}/pipeline', json={'steps': ['semantic_dedup']})
    assert r.status_code == 202
    report = client.get(f'/api/bank/{bank_id}').get_json()['pipeline_report']
    entry = next(e for e in report['steps'] if e['step'] == 'semantic_dedup')
    assert entry['status'] == 'skipped'
    assert 'Score' in entry['reason']


def test_pipeline_step_groups_when_embeddings_present(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _flat(10), 'b.jpg': _flat(20)})
    _write_score_cache(app, bank_id, {
        'a.jpg': _emb(1.0, 0.0), 'b.jpg': _emb(0.97, np.sqrt(1 - 0.97 ** 2))})
    r = client.post(f'/api/bank/{bank_id}/pipeline', json={'steps': ['semantic_dedup']})
    assert r.status_code == 202
    report = client.get(f'/api/bank/{bank_id}').get_json()['pipeline_report']
    entry = next(e for e in report['steps'] if e['step'] == 'semantic_dedup')
    assert entry['status'] == 'done'
    assert entry['counts']['semantic_groups'] == 1


# --- resolution --------------------------------------------------------------
def test_resolve_semantic_keep_best_and_manual(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {
        'big.jpg': _photo(256), 'smallcrop.jpg': _photo(96)})
    _write_score_cache(app, bank_id, {
        'big.jpg': _emb(1.0, 0.0), 'smallcrop.jpg': _emb(0.98, np.sqrt(1 - 0.98 ** 2))})
    client.post(f'/api/bank/{bank_id}/semantic-dedup', json={})
    # Keep-best prefers the larger image (no aesthetic score here) → 'big' wins.
    groups = client.get(f'/api/bank/{bank_id}/semantic-dup-groups').get_json()
    assert groups['total'] == 1
    best_id = groups['groups'][0]['best_id']
    best_name = next(i['name'] for i in groups['groups'][0]['images'] if i['id'] == best_id)
    assert best_name == 'big.jpg'
    r = client.post(f'/api/bank/{bank_id}/semantic-dups/resolve', json={'strategy': 'best'})
    assert r.get_json()['rejected'] == 1
    by = _by_name(client, bank_id)
    assert by['smallcrop.jpg']['status'] == 'reject'
    assert by['smallcrop.jpg']['reject_reason'] == 'semantic_dup'
    assert by['big.jpg']['status'] == 'pending'


def test_resolve_semantic_manual_keep_ids(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'x.jpg': _flat(10), 'y.jpg': _flat(20)})
    _write_score_cache(app, bank_id, {
        'x.jpg': _emb(1.0, 0.0), 'y.jpg': _emb(0.98, np.sqrt(1 - 0.98 ** 2))})
    client.post(f'/api/bank/{bank_id}/semantic-dedup', json={})
    yid = _by_name(client, bank_id)['y.jpg']['id']
    client.post(f'/api/bank/{bank_id}/semantic-dups/resolve', json={'keep_ids': [yid]})
    by = _by_name(client, bank_id)
    assert by['y.jpg']['status'] == 'pending'          # the one we kept
    assert by['x.jpg']['status'] == 'reject'


# --- score-cache staleness (embedding invalidation) --------------------------
def _infer_mod():
    spec = importlib.util.spec_from_file_location(
        'bank_score_infer',
        pathlib.Path(__file__).resolve().parents[1] / 'infer' / 'bank_score_infer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_file_signature_detects_change(tmp_path):
    mod = _infer_mod()
    p = str(tmp_path / 'f.bin')
    with open(p, 'wb') as fh:
        fh.write(b'hello')
    entry = ('ok', None, None, np.zeros(4), mod._file_sig(p))
    assert mod._is_stale(p, entry) is False
    with open(p, 'ab') as fh:            # change size → stale
        fh.write(b'more')
    assert mod._is_stale(p, entry) is True
    # A legacy 4-tuple (no signature) is never called stale.
    assert mod._is_stale(p, ('ok', None, None, np.zeros(4))) is False


def test_stale_embedding_dropped_from_semantic_pass(client, tmp_path, app):
    """An image edited at the same path after scoring must not group off its old
    embedding — the cached entry is invalidated by its signature."""
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _flat(10), 'b.jpg': _flat(20)})
    _write_score_cache(app, bank_id, {
        'a.jpg': _emb(1.0, 0.0), 'b.jpg': _emb(0.97, np.sqrt(1 - 0.97 ** 2))})
    # Edit b.jpg on disk (grow it) → its cached signature no longer matches.
    with open(str(src / 'b.jpg'), 'ab') as fh:
        fh.write(b'\x00' * 512)
    with app.app_context():
        from app.models import BankImage
        from app.services import image_bank_service as banks
        bank = banks.get_bank(_uid(), bank_id)
        embs = banks._load_score_embeddings(bank)
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        assert banks.abs_image_path(bank, rows['a.jpg']) in embs
        assert banks.abs_image_path(bank, rows['b.jpg']) not in embs   # stale, dropped
        # With b dropped, a has no partner → no semantic group.
        assert banks.rebuild_semantic_dup_groups(bank_id) == 0
