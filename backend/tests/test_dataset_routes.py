import io
import zipfile

from PIL import Image


def _png_bytes(color=(255, 0, 0)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger})


def test_create_returns_ok_envelope(client):
    resp = _create(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert isinstance(body['id'], int)
    # the workspace payload is fetched separately — the envelope stays minimal like SRC
    full = client.get(f"/api/dataset/{body['id']}").get_json()
    assert full['name'] == 'Lola' and full['trigger_word'] == 'lola'


def test_create_requires_name_and_trigger(client):
    resp = client.post('/api/dataset/create', json={'name': '', 'trigger_word': ''})
    assert resp.status_code == 400


def test_list_contains_created_dataset(client):
    created = _create(client).get_json()
    resp = client.get('/api/dataset/list')
    assert resp.status_code == 200
    ids = [d['id'] for d in resp.get_json()['datasets']]
    assert created['id'] in ids


def test_get_unknown_id_404(client):
    resp = client.get('/api/dataset/999999')
    assert resp.status_code == 404


def test_variations_catalog(client):
    resp = client.get('/api/dataset/variations')
    assert resp.status_code == 200
    body = resp.get_json()
    assert 'catalog' in body and 'presets' in body
    assert 'zimage_12' in body['presets']


def test_ref_upload_multipart(client):
    ds_id = _create(client).get_json()['id']
    data = {'file': (io.BytesIO(_png_bytes()), 'ref.png')}
    resp = client.post(f'/api/dataset/{ds_id}/ref', data=data, content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['ref_filename']
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['ref_filename'] == body['ref_filename']


def test_ref_upload_unknown_dataset_404(client):
    data = {'file': (io.BytesIO(_png_bytes()), 'ref.png')}
    resp = client.post('/api/dataset/999999/ref', data=data, content_type='multipart/form-data')
    assert resp.status_code == 404


def test_export_zip_content_type(client):
    ds_id = _create(client, 'Zoe', 'zoe').get_json()['id']
    data = {'file': (io.BytesIO(_png_bytes()), 'ref.png')}
    client.post(f'/api/dataset/{ds_id}/ref', data=data, content_type='multipart/form-data')
    files = {'files': (io.BytesIO(_png_bytes((0, 255, 0))), 'img1.png')}
    client.post(f'/api/dataset/{ds_id}/import', data=files, content_type='multipart/form-data')
    imgs = client.get(f'/api/dataset/{ds_id}').get_json()['images']
    assert imgs, 'import should have produced at least one image row'

    resp = client.get(f'/api/dataset/{ds_id}/export')
    assert resp.status_code == 200
    assert resp.mimetype == 'application/zip'
    z = zipfile.ZipFile(io.BytesIO(resp.data))
    assert any(n.endswith('_000_ref.png') for n in z.namelist())


def test_export_no_kept_images_400(client):
    ds_id = _create(client, 'Empty', 'empty').get_json()['id']
    resp = client.get(f'/api/dataset/{ds_id}/export')
    assert resp.status_code == 400


def test_generate_chatgpt_no_key_accepts_and_creates_pending_rows(client, monkeypatch):
    """The service doesn't validate the API key up front — rows go pending and
    the background batch (which we don't wait for) will fail them later.

    The batch's Thread is stubbed out (same technique as the brief's
    test_api_fanout_creates_pending_rows) so this test only exercises the route
    contract and never races a real background thread against a real HTTP call:
    test_config.py's test_secrets_roundtrip leaks a fake OPENAI_API_KEY into
    process-wide os.environ via a bare `os.environ[...] = ...` (no monkeypatch),
    so an un-stubbed batch could otherwise fire a REAL (401) OpenAI request in
    a background thread racing this test's teardown. See task-8-report.md."""
    calls = []
    monkeypatch.setattr('app.services.face_dataset_service.threading.Thread',
                        lambda target, args=(), daemon=True: type('T', (), {'start': lambda s: calls.append(args)})())
    ds_id = _create(client, 'Nyx', 'nyx').get_json()['id']
    data = {'file': (io.BytesIO(_png_bytes()), 'ref.png')}
    client.post(f'/api/dataset/{ds_id}/ref', data=data, content_type='multipart/form-data')
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'generator': 'chatgpt',
        'variations': [{'label': 'Visage face, neutre', 'framing': 'face',
                        'prompt': 'close-up portrait, front view, neutral expression'}],
        'multiplier': 1,
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True and body['created'] == 1
    assert calls  # background batch was dispatched (never actually run)
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert len(payload['images']) == 1


def test_generate_klein_without_comfyui_returns_409(client):
    """klein_edit_helper (Task 14) isn't lifted yet -> the Klein path must
    surface a clean 409, not a raw 500."""
    ds_id = _create(client, 'Kai', 'kai').get_json()['id']
    data = {'file': (io.BytesIO(_png_bytes()), 'ref.png')}
    client.post(f'/api/dataset/{ds_id}/ref', data=data, content_type='multipart/form-data')
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'generator': 'klein',
        'variations': [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
        'multiplier': 1,
        'klein_model': 'some_model',
    })
    assert resp.status_code == 409


def test_image_status_invalid_returns_400(client):
    resp = client.post('/api/dataset/image/1/status', json={'status': 'nonsense'})
    assert resp.status_code == 400


def test_image_status_unknown_image_404(client):
    resp = client.post('/api/dataset/image/999999/status', json={'status': 'keep'})
    assert resp.status_code == 404


def test_delete_dataset(client):
    ds_id = _create(client, 'Trash', 'trash').get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/delete')
    assert resp.status_code == 200
    assert client.get(f'/api/dataset/{ds_id}').status_code == 404


def test_delete_unknown_dataset_404(client):
    resp = client.post('/api/dataset/999999/delete')
    assert resp.status_code == 404
