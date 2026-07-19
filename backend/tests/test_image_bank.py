"""🗃️ Image bank — inventory, quality scan, duplicate groups, triage statuses,
promotion. The bank references the source folder IN PLACE and must never write
to it; background jobs run inline under TESTING (see bank_jobs.start)."""
import io
import os
import random

import pytest
from PIL import Image, ImageFilter


# --- image factories ---------------------------------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.jpg', '.jpeg')):
        im.save(path, 'JPEG', quality=92)
    else:
        im.save(path)


def checkerboard(size=128, cell=8):
    im = Image.new('L', (size, size))
    im.putdata([255 if ((x // cell + y // cell) % 2) else 0
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def photo_like(size=256):
    """Smooth gradient + a bright disc: enough large-scale structure for a
    STABLE dHash across resizes (a checkerboard aliases at 9×8)."""
    im = Image.new('L', (size, size))
    c, r2 = size / 2, (size / 3) ** 2
    im.putdata([min(255, int(150 * x / size + 50 * y / size)
                    + (80 if (x - c) ** 2 + (y - c) ** 2 < r2 else 0))
                for y in range(size) for x in range(size)])
    return im.convert('RGB')


def noisy(size=128, seed=7):
    rng = random.Random(seed)
    im = Image.new('L', (size, size))
    im.putdata([rng.randrange(256) for _ in range(size * size)])
    return im.convert('RGB')


def flat(size=128, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, files, name='B'):
    """Write ``files`` = {relpath: PIL image or raw bytes} under a source dir,
    create the bank over it, return (bank_id, src_dir)."""
    src = tmp_path / 'src'
    for rel, im in files.items():
        p = src / rel
        if isinstance(im, (bytes, bytearray)):
            os.makedirs(p.parent, exist_ok=True)
            p.write_bytes(im)
        else:
            _save(str(p), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


# --- quality metrics (pure PIL) ---------------------------------------------
def test_quality_metrics_discriminate():
    from app.services.image_quality import quality_metrics
    sharp = quality_metrics(checkerboard())
    blurry = quality_metrics(checkerboard().filter(ImageFilter.GaussianBlur(6)))
    gray = quality_metrics(flat())
    grain = quality_metrics(noisy())
    # Sharpness: checkerboard ≫ its blurred copy ≫ flat gray.
    assert sharp['blur_score'] > 100 > blurry['blur_score'] >= gray['blur_score']
    # Noise: per-pixel random ≫ smooth surfaces.
    assert grain['noise_score'] > 15 > gray['noise_score']
    # Uniformity: flat gray is near zero, textured images are not.
    assert gray['uniformity_score'] < 12 < sharp['uniformity_score']


def test_quality_metrics_match_reference_laplacian():
    """The two-pass histogram trick must equal a direct signed Laplacian
    variance over the interior (PIL leaves the 1-px border unfiltered, so the
    implementation crops it — the reference does the same)."""
    from app.services.image_quality import quality_metrics
    im = photo_like(size=48).convert('L')
    px = list(im.getdata())
    w, h = im.size

    def at(x, y):
        return px[y * w + x]
    lap = [at(x, y - 1) + at(x - 1, y) + at(x + 1, y) + at(x, y + 1) - 4 * at(x, y)
           for y in range(1, h - 1) for x in range(1, w - 1)]
    mean = sum(lap) / len(lap)
    ref = sum((v - mean) ** 2 for v in lap) / len(lap)
    got = quality_metrics(im.convert('RGB'))['blur_score']
    assert got == pytest.approx(ref, rel=0.05, abs=2.0)


# --- inventory ---------------------------------------------------------------
def test_create_bank_walks_recursively(client, tmp_path, app):
    bank_id, src = _mkbank(client, tmp_path, {
        'a.jpg': checkerboard(), 'sub/b.png': flat(), 'sub/deep/c.webp': noisy(),
        'notes.txt': b'not an image',
    })
    with app.app_context():
        from app.models import BankImage
        rels = {r.relpath.replace('\\', '/') for r in
                BankImage.query.filter_by(bank_id=bank_id).all()}
    assert rels == {'a.jpg', 'sub/b.png', 'sub/deep/c.webp'}


def test_create_bank_validates_folder(client, tmp_path):
    r = client.post('/api/bank/create',
                    json={'name': 'X', 'folder': str(tmp_path / 'absent')})
    assert r.status_code == 400
    r = client.post('/api/bank/create', json={'name': '', 'folder': str(tmp_path)})
    assert r.status_code == 400


def test_banks_list(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': flat()})
    data = client.get('/api/banks').get_json()
    assert [b['id'] for b in data['banks']] == [bank_id]
    assert data['banks'][0]['total'] == 1


# --- quality scan ------------------------------------------------------------
def test_scan_scores_flags_and_unreadable(client, tmp_path):
    bank_id, src = _mkbank(client, tmp_path, {
        'sharp.jpg': checkerboard(), 'gray.jpg': flat(),
        'broken.jpg': b'\xff\xd8 definitely not a jpeg',
    })
    r = client.post(f'/api/bank/{bank_id}/scan', json={})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['counts']['scanned'] == 3
    assert payload['activity']['finished'] is True
    assert payload['flags']['unreadable'] == 1
    # Source folder untouched: exactly the three files we wrote.
    assert sorted(p.name for p in src.rglob('*') if p.is_file()) == \
        ['broken.jpg', 'gray.jpg', 'sharp.jpg']
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    by_name = {i['name']: i for i in imgs}
    assert 'blur' in by_name['gray.jpg']['flags']
    assert 'uniform' in by_name['gray.jpg']['flags']
    assert 'small' in by_name['sharp.jpg']['flags']      # 128 px < min_side 768
    assert 'blur' not in by_name['sharp.jpg']['flags']
    # The unreadable file is auto-rejected (it can never be promoted).
    assert by_name['broken.jpg']['status'] == 'reject'
    assert by_name['broken.jpg']['reject_reason'] == 'unreadable'


def test_flags_follow_threshold_changes_without_rescan(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'sharp.jpg': checkerboard()})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert 'blur' not in by['sharp.jpg']['flags']
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'bank': {'sharpness_min': 10 ** 9}})
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert 'blur' in by['sharp.jpg']['flags']            # no rescan needed


# --- duplicates --------------------------------------------------------------
def test_duplicate_groups_and_keep_best(client, tmp_path):
    big = photo_like(size=256)
    small = big.resize((96, 96), Image.LANCZOS)          # same content, downscaled
    bank_id, _src = _mkbank(client, tmp_path, {
        'orig.jpg': big, 'copy.jpg': small, 'other.jpg': noisy(),
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['dup'] == {'groups': 1, 'images': 2, 'unresolved': 1}
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    assert len(groups) == 1
    names = {i['name'] for i in groups[0]['images']}
    assert names == {'orig.jpg', 'copy.jpg'}
    # keep best = the higher-resolution member.
    best = next(i for i in groups[0]['images'] if i['id'] == groups[0]['best_id'])
    assert best['name'] == 'orig.jpg'
    r = client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'best'})
    assert r.get_json() == {'ok': True, 'resolved': 1, 'rejected': 1}
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['copy.jpg']['status'] == 'reject'
    assert by['copy.jpg']['reject_reason'] == 'duplicate'
    assert by['orig.jpg']['status'] == 'pending'         # keeper untouched
    assert client.get(f'/api/bank/{bank_id}').get_json()['dup']['unresolved'] == 0


def test_resolve_keep_first_and_manual_pick(client, tmp_path, app):
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {
        'a_first.jpg': im, 'b_copy.jpg': im, 'c_copy.jpg': im,
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    ids = {i['name']: i['id'] for i in groups[0]['images']}
    # Manual pick: keep c explicitly.
    r = client.post(f'/api/bank/{bank_id}/dups/resolve',
                    json={'keep_ids': [ids['c_copy.jpg']]})
    assert r.get_json()['rejected'] == 2
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['c_copy.jpg']['status'] == 'pending'
    assert by['a_first.jpg']['status'] == 'reject'
    # Reset then keep-first: lowest id (import order) wins.
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': list(ids.values()), 'status': 'pending'})
    r = client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'first'})
    assert r.get_json()['rejected'] == 2
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['a_first.jpg']['status'] == 'pending'
    assert by['b_copy.jpg']['status'] == 'reject'


def test_resolve_never_flips_a_manual_keep(client, tmp_path):
    im = checkerboard(size=256, cell=16)
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': im, 'b.jpg': im})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    groups = client.get(f'/api/bank/{bank_id}/dup-groups').get_json()['groups']
    ids = {i['name']: i['id'] for i in groups[0]['images']}
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [ids['b.jpg']], 'status': 'keep'})
    client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'first'})
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['b.jpg']['status'] == 'keep'               # manual keep survives


# --- flag application + statuses --------------------------------------------
def test_apply_flags_rejects_pending_only(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {
        'gray.jpg': flat(), 'gray2.jpg': flat(value=90), 'sharp.jpg': checkerboard(),
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    keep_id = next(i['id'] for i in imgs if i['name'] == 'gray.jpg')
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [keep_id], 'status': 'keep'})
    r = client.post(f'/api/bank/{bank_id}/apply-flags', json={'flags': ['uniform']})
    assert r.get_json()['rejected'] == {'uniform': 1}    # gray2 only — keep survives
    by = {i['name']: i for i in
          client.get(f'/api/bank/{bank_id}/images').get_json()['images']}
    assert by['gray.jpg']['status'] == 'keep'
    assert by['gray2.jpg']['status'] == 'reject'
    assert by['gray2.jpg']['reject_reason'] == 'uniform'
    assert by['sharp.jpg']['status'] == 'pending'


# --- listing filters + pagination -------------------------------------------
def test_images_filters_and_pagination(client, tmp_path):
    files = {f'n{i:02d}.jpg': checkerboard(size=128 + 2 * i) for i in range(5)}
    files['gray.jpg'] = flat()
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/scan', json={})
    page = client.get(f'/api/bank/{bank_id}/images?limit=2&offset=2').get_json()
    assert page['total'] == 6 and len(page['images']) == 2
    flagged = client.get(f'/api/bank/{bank_id}/images?flag=uniform').get_json()
    assert [i['name'] for i in flagged['images']] == ['gray.jpg']
    status = client.get(f'/api/bank/{bank_id}/images?status=reject').get_json()
    assert status['total'] == 0


def test_images_sort_by_resolution(client, tmp_path, app):
    """Sort=res_desc/res_asc orders by MEGAPIXELS (width×height), not width — a
    900×900 (810k px) outranks a wider 1200×300 (360k). Unscanned rows (width or
    height NULL) sink to the end in BOTH directions, and the sort composes with
    filters + pagination."""
    files = {f'{n}.jpg': checkerboard(size=64) for n in ('a', 'b', 'c', 'd', 'e')}
    bank_id, _src = _mkbank(client, tmp_path, files)
    # Set dimensions by hand (bypass the scan) so the areas are unambiguous.
    dims = {'a': (900, 900), 'b': (1200, 300), 'c': (1000, 1000),
            'd': (400, 400), 'e': (None, None)}   # e = unscanned → NULL area
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        for row in BankImage.query.filter_by(bank_id=bank_id).all():
            w, h = dims[row.relpath.split('.')[0]]
            row.width, row.height = w, h
        db.session.commit()

    def names(sort, **qs):
        params = '&'.join(f'{k}={v}' for k, v in {'sort': sort, **qs}.items())
        got = client.get(f'/api/bank/{bank_id}/images?{params}').get_json()
        return [i['name'].split('.')[0] for i in got['images']]

    # Descending megapixels: c(1M) > a(810k) > b(360k) > d(160k); NULL 'e' last.
    assert names('res_desc') == ['c', 'a', 'b', 'd', 'e']
    # Ascending: d < b < a < c; NULL 'e' still last (never first).
    assert names('res_asc') == ['d', 'b', 'a', 'c', 'e']
    # Composes with pagination — the top-2 of the descending order.
    page = client.get(f'/api/bank/{bank_id}/images?sort=res_desc&limit=2').get_json()
    assert [i['name'].split('.')[0] for i in page['images']] == ['c', 'a']
    assert page['total'] == 5
    # Composes with a status filter: reject the two largest, keep the sort.
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [i['id'] for i in
                              client.get(f'/api/bank/{bank_id}/images?sort=res_desc&limit=2')
                              .get_json()['images']], 'status': 'reject'})
    assert names('res_desc', status='pending') == ['b', 'd', 'e']
    # An unknown sort value is ignored (falls back to the default id order).
    assert names('bogus') == ['a', 'b', 'c', 'd', 'e']


def test_no_face_filter_only_matches_no_face_state(client, tmp_path, app):
    """The "No face" chip must show ONLY images where NO face was detected. The
    other non-scorable states (low_det / too_small / extreme_pose) DID find a
    face and must NOT appear — that regression surfaced photos with visible
    faces under a "No face" label. Unscanned rows (face_state NULL) stay out."""
    files = {f'n{i}.jpg': checkerboard(size=256) for i in range(6)}
    bank_id, _src = _mkbank(client, tmp_path, files)
    client.post(f'/api/bank/{bank_id}/scan', json={})
    states = ['no_face', 'low_det', 'too_small', 'extreme_pose', 'scorable', None]
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        for row, state in zip(rows, states):
            row.face_state = state
        db.session.commit()
    got = client.get(f'/api/bank/{bank_id}/images?flag=no_face').get_json()
    assert [i['face_state'] for i in got['images']] == ['no_face']


def test_face_device_resolves_cpu_without_cuda(app, monkeypatch):
    """Default device is CPU and the GPU window is never opened unless the face
    interpreter truly exposes CUDA. 'cuda' requested without CUDA degrades to
    CPU (so the parent won't serialize a CPU pass behind training)."""
    from app.services import image_bank_service as svc
    from app import capabilities
    with app.app_context():
        monkeypatch.setattr(capabilities, 'face_gpu_available', lambda: False)
        assert svc._resolve_face_device() == ('cpu', False)          # auto, no CUDA
        monkeypatch.setattr(svc.cfg, 'get',
                            lambda k, d=None: 'cuda' if k == 'face_scoring.device' else d)
        assert svc._resolve_face_device() == ('cpu', False)          # cuda asked, none
        monkeypatch.setattr(capabilities, 'face_gpu_available', lambda: True)
        assert svc._resolve_face_device() == ('cuda', True)          # cuda asked + CUDA


def test_face_embed_infer_cpu_providers_no_device():
    """The infer script must default to CPU-only providers — a bare CUDA-first
    list would grab the GPU the instant onnxruntime-gpu is present, outside the
    parent's GPU-exclusive window."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'infer' / 'face_embed_infer.py').read_text(encoding='utf-8')
    # No unconditional CUDA-first provider list survives.
    assert "['CUDAExecutionProvider', 'CPUExecutionProvider']}" not in src
    assert "device = str(req.get('device') or 'cpu').lower()" in src


# --- promotion ---------------------------------------------------------------
def test_promote_keeps_into_dataset(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.jpg': checkerboard(size=256, cell=16), 'b.jpg': noisy(size=256),
        'c.jpg': flat(size=256),
    })
    client.post(f'/api/bank/{bank_id}/scan', json={})
    imgs = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    keep_ids = [i['id'] for i in imgs if i['name'] in ('a.jpg', 'b.jpg')]
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': keep_ids, 'status': 'keep'})
    with app.app_context():
        from app.services import face_dataset_service as svc
        ds = svc.create_dataset('local', 'From bank', 'bnk')
        ds_id = ds.id
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_id})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['finished'] is True
    assert payload['activity']['error'] is None
    assert payload['counts']['promoted'] == 2
    with app.app_context():
        from app.models import FaceDatasetImage
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds_id).all()
        assert len(rows) == 2
        assert all(r2.source == 'import' and r2.status == 'keep' for r2 in rows)
    # Second promotion with nothing left to promote → 400.
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_id})
    assert r.status_code == 400


def test_promote_requires_dataset(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    r = client.post(f'/api/bank/{bank_id}/promote', json={})
    assert r.status_code == 400
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': 999})
    assert r.status_code == 400


# --- faces gate --------------------------------------------------------------
def test_faces_unavailable_is_503(client, tmp_path, monkeypatch):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    from app.services import face_similarity
    monkeypatch.setattr(face_similarity, 'is_available', lambda: False)
    r = client.post(f'/api/bank/{bank_id}/faces', json={})
    assert r.status_code == 503
    assert 'face scoring' in r.get_json()['error']


def test_cluster_assignment_orders_by_size():
    np = pytest.importorskip('numpy')
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        'face_embed_infer',
        pathlib.Path(__file__).resolve().parents[1] / 'infer' / 'face_embed_infer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    e = {}
    v1 = np.array([1.0] + [0.0] * 511, dtype='float32')
    v2 = np.array([0.0, 1.0] + [0.0] * 510, dtype='float32')
    cache = {
        'p1': ('scorable', 0.9, 0.5, v1), 'p2': ('scorable', 0.9, 0.5, v1),
        'p3': ('scorable', 0.9, 0.5, v1), 'p4': ('scorable', 0.9, 0.5, v2),
        'p5': ('scorable', 0.9, 0.5, v2), 'p6': ('no_face', 0.0, 0.0, v2 * 0),
    }
    out = mod._cluster(list(cache), cache, 0.45)
    assert out['p1'] == out['p2'] == out['p3'] == 1       # biggest cluster = id 1
    assert out['p4'] == out['p5'] == 2
    assert 'p6' not in out                                # no face → no cluster


# --- serving + deletion ------------------------------------------------------
def test_thumb_and_file_endpoints(client, tmp_path):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard(size=600)})
    img_id = client.get(f'/api/bank/{bank_id}/images').get_json()['images'][0]['id']
    r = client.get(f'/api/bank/{bank_id}/thumb/{img_id}')   # lazy — no scan yet
    assert r.status_code == 200
    with Image.open(io.BytesIO(r.data)) as t:
        assert max(t.size) <= 320
    r = client.get(f'/api/bank/{bank_id}/file/{img_id}')
    assert r.status_code == 200
    with Image.open(io.BytesIO(r.data)) as f:
        assert f.size == (600, 600)
    assert client.get(f'/api/bank/{bank_id}/thumb/99999').status_code == 404


def test_delete_bank_never_touches_source(client, tmp_path, app):
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    client.post(f'/api/bank/{bank_id}/scan', json={})
    assert client.delete(f'/api/bank/{bank_id}').status_code == 200
    assert (src / 'a.jpg').is_file()                     # source intact
    assert client.get(f'/api/bank/{bank_id}').status_code == 404
    with app.app_context():
        from app.models import BankImage
        assert BankImage.query.filter_by(bank_id=bank_id).count() == 0
        import app.config as cfg
        assert not (cfg.banks_root() / str(bank_id)).exists()


def test_busy_bank_answers_409(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': checkerboard()})
    from app.services import bank_jobs
    bank_jobs._jobs[bank_id] = {'kind': 'scan', 'done': 0, 'total': 1,
                                'error': None, 'cancelled': False,
                                'finished': False, 'detail': None,
                                'started_at': 0, '_touched': __import__('time').time(),
                                '_cancel_hook': None}
    r = client.post(f'/api/bank/{bank_id}/scan', json={})
    assert r.status_code == 409
    # cancel works, then the job is startable again
    assert client.post(f'/api/bank/{bank_id}/cancel', json={}).get_json()['ok'] is True
