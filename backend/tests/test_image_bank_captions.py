"""🗃️ Image bank — captioning + full-text search. Captions are written by the
same engines the datasets use (here the Ollama vision call is mocked so the pass
is hermetic), double as the bank's search text, and ride along to the dataset on
promotion. Background jobs run inline under TESTING (see bank_jobs.start)."""
import os

from PIL import Image


# --- factories (mirror test_image_bank) --------------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.jpg', '.jpeg')):
        im.save(path, 'JPEG', quality=92)
    else:
        im.save(path)


def _flat(size=64, value=128):
    return Image.new('RGB', (size, size), (value, value, value))


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _by_name(client, bank_id, **params):
    q = '&'.join(f'{k}={v}' for k, v in params.items())
    url = f'/api/bank/{bank_id}/images' + (f'?{q}' if q else '')
    return {i['name']: i for i in client.get(url).get_json()['images']}


def _use_ollama_backend(app):
    """Force the Ollama caption backend so JoyCaption (ai-toolkit) is skipped."""
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': 'ollama'}})


def _mock_vision(monkeypatch, caption_by_pixel):
    """Mock the Ollama vision seam caption_paths uses: describe returns a caption
    keyed by the image's top-left pixel value, unload is a no-op."""
    from app.services import vision_ollama

    def fake_describe(image_bytes, *a, **k):
        import io
        v = Image.open(io.BytesIO(image_bytes)).convert('L').getpixel((0, 0))
        return caption_by_pixel.get(v, '')

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


# --- caption pass + search ---------------------------------------------------
def test_caption_fills_rows_and_powers_search(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {
        'red.png': _flat(value=128), 'blue.png': _flat(value=60)})
    _mock_vision(monkeypatch, {128: 'a woman in a red dress', 60: 'a blue car on a street'})

    r = client.post(f'/api/bank/{bank_id}/caption', json={})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['finished'] is True
    assert payload['activity']['error'] is None

    by = _by_name(client, bank_id)
    assert by['red.png']['caption'] == 'a woman in a red dress'
    assert by['blue.png']['caption'] == 'a blue car on a street'

    # The caption is now searchable text.
    hits = client.get(f'/api/bank/{bank_id}/images?search=red%20dress').get_json()
    assert [i['name'] for i in hits['images']] == ['red.png']
    hits = client.get(f'/api/bank/{bank_id}/images?search=car').get_json()
    assert [i['name'] for i in hits['images']] == ['blue.png']
    # relpath is searchable too (the file name matches even without a caption match).
    hits = client.get(f'/api/bank/{bank_id}/images?search=blue.png').get_json()
    assert [i['name'] for i in hits['images']] == ['blue.png']


def test_search_composes_with_status_filter(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.png': _flat(value=128), 'b.png': _flat(value=60)})
    _mock_vision(monkeypatch, {128: 'a red dress', 60: 'a red hat'})
    client.post(f'/api/bank/{bank_id}/caption', json={})

    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.png']['id']], 'status': 'keep'})
    # search=red matches both; status=keep narrows to a.png.
    both = client.get(f'/api/bank/{bank_id}/images?search=red').get_json()
    assert both['total'] == 2
    narrowed = client.get(f'/api/bank/{bank_id}/images?search=red&status=keep').get_json()
    assert [i['name'] for i in narrowed['images']] == ['a.png']


def test_search_escapes_like_metacharacters(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.png': _flat(value=128), 'b.png': _flat(value=60)})
    _mock_vision(monkeypatch, {128: '50% off sign', 60: 'a plain wall'})
    client.post(f'/api/bank/{bank_id}/caption', json={})
    # A literal '%' must match itself, not act as a wildcard.
    hits = client.get(f'/api/bank/{bank_id}/images?search=50%25%20off').get_json()
    assert [i['name'] for i in hits['images']] == ['a.png']


def test_caption_only_missing_unless_force(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.png': _flat(value=128), 'b.png': _flat(value=60)})
    # Pre-caption a.png by hand; a non-force pass must leave it and only fill b.png.
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        row = BankImage.query.filter(BankImage.relpath.like('%a.png')).first()
        row.caption = 'kept as is'
        db.session.commit()
    _mock_vision(monkeypatch, {128: 'RECAPTIONED', 60: 'fresh caption for b'})

    client.post(f'/api/bank/{bank_id}/caption', json={})
    by = _by_name(client, bank_id)
    assert by['a.png']['caption'] == 'kept as is'
    assert by['b.png']['caption'] == 'fresh caption for b'

    # force=true overwrites the existing caption.
    client.post(f'/api/bank/{bank_id}/caption', json={'force': True})
    by = _by_name(client, bank_id)
    assert by['a.png']['caption'] == 'RECAPTIONED'


def test_caption_restricts_to_selected_ids(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.png': _flat(value=128), 'b.png': _flat(value=60)})
    _mock_vision(monkeypatch, {128: 'only a', 60: 'only b'})
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/caption', json={'image_ids': [by['a.png']['id']]})
    by = _by_name(client, bank_id)
    assert by['a.png']['caption'] == 'only a'
    assert by['b.png']['caption'] is None       # not in the selection


def test_caption_stops_at_image_boundary(client, tmp_path, app, monkeypatch):
    """The graceful-stop contract: a cancel mid-pass finishes the image being
    written, then leaves the rest untouched (never cuts an inference off)."""
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.png': _flat(value=128), 'b.png': _flat(value=60)})
    from app.services import bank_jobs, vision_ollama

    def fake_describe(image_bytes, *a, **k):
        import io
        v = Image.open(io.BytesIO(image_bytes)).convert('L').getpixel((0, 0))
        # Request cancel as soon as the first image is captioned; the loop must
        # then stop at the next image boundary, leaving the other uncaptioned.
        bank_jobs.cancel(bank_id)
        return 'first caption' if v == 128 else 'second caption'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)

    client.post(f'/api/bank/{bank_id}/caption', json={})
    captions = [i['caption'] for i in _by_name(client, bank_id).values()]
    non_null = [c for c in captions if c]
    assert len(non_null) == 1           # exactly one written, the rest left alone


def _striped(value, phase, size=256):
    """A distinct-dHash image (vertical bands, phase-shifted) carrying a top-left
    pixel `value` marker the mock keys off — two must NOT perceptual-dedupe on
    promotion (unlike two flat fills, which share dHash 0)."""
    im = Image.new('L', (size, size))
    im.putdata([255 if (((x // 32) + phase) % 2) else 0
                for _y in range(size) for x in range(size)])
    im = im.convert('RGB')
    im.putpixel((0, 0), (value, value, value))
    return im


def test_promotion_carries_captions_into_dataset(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {
        'a.png': _striped(128, phase=0), 'b.png': _striped(60, phase=1)})
    _mock_vision(monkeypatch, {128: 'caption for a', 60: 'caption for b'})
    client.post(f'/api/bank/{bank_id}/caption', json={})
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.png']['id'], by['b.png']['id']], 'status': 'keep'})
    with app.app_context():
        from app.services import face_dataset_service as svc
        ds_id = svc.create_dataset('local', 'From bank', 'bnk').id
    r = client.post(f'/api/bank/{bank_id}/promote', json={'dataset_id': ds_id})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['finished'] is True
    assert payload['activity']['error'] is None
    with app.app_context():
        from app.models import FaceDatasetImage
        caps = {r2.caption for r2 in
                FaceDatasetImage.query.filter_by(dataset_id=ds_id).all()}
    assert caps == {'caption for a', 'caption for b'}


# --- gates -------------------------------------------------------------------
def test_caption_gate_400_when_backend_none(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': 'none'}})
    r = client.post(f'/api/bank/{bank_id}/caption', json={})
    assert r.status_code == 400
    assert 'backend' in r.get_json()['error']


def test_caption_refuses_when_gpu_busy(client, tmp_path, app, monkeypatch):
    _use_ollama_backend(app)
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    from app.services import image_bank_service as banks
    monkeypatch.setattr(banks, '_gpu_busy_reason', lambda: 'training is running on the GPU')
    r = client.post(f'/api/bank/{bank_id}/caption', json={})
    assert r.status_code == 503
    assert 'training' in r.get_json()['error']
