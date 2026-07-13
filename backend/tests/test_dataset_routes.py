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


def test_create_character_requires_trigger(client):
    """No kind (character, the default) still needs a trigger — it's the token
    that summons the identity. A blank trigger with a filled name is still 400,
    not a silent create (this is the exact shape the UI's Create button must
    stay disabled for)."""
    resp = client.post('/api/dataset/create', json={'name': 'Nolora', 'trigger_word': ''})
    assert resp.status_code == 400


def test_create_style_does_not_require_trigger(client):
    """A style LoRA has no trigger (it tints every image once loaded) — the
    service auto-generates a unique zsty_<id> placeholder for it, so the route
    must NOT reject an empty trigger_word for kind='style'."""
    resp = client.post('/api/dataset/create',
                       json={'name': 'Inkwash', 'trigger_word': '', 'kind': 'style'})
    assert resp.status_code == 200
    ds_id = resp.get_json()['id']
    full = client.get(f'/api/dataset/{ds_id}').get_json()
    assert full['trigger_word'] == f'zsty_{ds_id}'


def test_list_contains_created_dataset(client):
    created = _create(client).get_json()
    resp = client.get('/api/dataset/list')
    assert resp.status_code == 200
    ids = [d['id'] for d in resp.get_json()['datasets']]
    assert created['id'] in ids


def test_get_unknown_id_404(client):
    resp = client.get('/api/dataset/999999')
    assert resp.status_code == 404


def test_list_carries_library_stats(client, app):
    """The library page reads counts + trained families straight from /list —
    3 images (2 keep, 1 of them captioned, 1 reject) + one training run."""
    ds_id = _create(client, 'Stats', 'stats').get_json()['id']
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage, TrainingRunRecord
        db.session.add_all([
            FaceDatasetImage(dataset_id=ds_id, status='keep', source='upload',
                             filename='a.webp', caption='a caption'),
            FaceDatasetImage(dataset_id=ds_id, status='keep', source='upload',
                             filename='b.webp', caption=''),
            FaceDatasetImage(dataset_id=ds_id, status='reject', source='upload',
                             filename='c.webp', caption='x'),
            TrainingRunRecord(dataset_id=ds_id, family='krea', source='local',
                              fingerprint='test-fp', version=1),
        ])
        db.session.commit()
    entry = next(d for d in client.get('/api/dataset/list').get_json()['datasets']
                 if d['id'] == ds_id)
    assert entry['images_total'] == 3
    assert entry['images_kept'] == 2
    assert entry['images_captioned'] == 1
    assert entry['trained_families'] == ['krea']


def test_images_batch_route_validates_and_applies(client, app):
    ds_id = _create(client, 'Batch', 'batch').get_json()['id']
    # seed one committed image row through the service (no engine needed)
    with app.app_context():
        import os
        from app.services import face_dataset_service as svc
        from app.models import FaceDatasetImage
        d = svc._dataset_dir(ds_id); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'a.webp'), 'wb').write(_png_bytes())
        img = FaceDatasetImage(dataset_id=ds_id, filename='a.webp', status='pending', framing='face')
        svc.db.session.add(img); svc.db.session.commit()
        img_id = img.id
    assert client.post(f'/api/dataset/{ds_id}/images/batch',
                       json={'ids': [img_id], 'action': 'nope'}).status_code == 400
    assert client.post(f'/api/dataset/{ds_id}/images/batch',
                       json={'ids': [], 'action': 'keep'}).status_code == 400
    assert client.post('/api/dataset/999999/images/batch',
                       json={'ids': [img_id], 'action': 'keep'}).status_code == 404
    ok = client.post(f'/api/dataset/{ds_id}/images/batch',
                     json={'ids': [img_id], 'action': 'keep'})
    assert ok.status_code == 200 and ok.get_json() == {'ok': True, 'affected': 1}
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['images'][0]['status'] == 'keep'


def test_import_route_crop_0_skips_gpu_window(client, monkeypatch):
    """Character import with crop='0' keeps the original framing: import_images is
    called with crop=False and the GPU-exclusive vision window is NOT opened."""
    import app.routes.datasets as dr
    ds_id = _create(client, 'NoCrop', 'nocrop').get_json()['id']
    captured = {}

    def fake_import(user_id, dataset_id, files, crop=True, **kw):
        captured['crop'] = crop
        return [1], 0

    def boom(*a, **k):
        raise AssertionError('GPU window must not open when head-crop is off')

    monkeypatch.setattr(dr.svc, 'import_images', fake_import)
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', boom)
    resp = client.post(f'/api/dataset/{ds_id}/import',
                       data={'files': (io.BytesIO(_png_bytes()), 'a.png'), 'crop': '0'},
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    assert captured['crop'] is False


def test_import_route_reports_duplicates(client):
    """The /import envelope exposes the perceptual-duplicate count so the UI can
    say '2 imported · 1 duplicate skipped'. Concept dataset -> crop=False path
    (no GPU window to stub)."""
    from PIL import Image as PILImage
    def grad(direction):
        ramp = list(range(0, 256, 32))
        if direction == 'rtl':
            ramp = ramp[::-1]
        small = PILImage.new('L', (8, 8)); small.putdata([ramp[x] for _ in range(8) for x in range(8)])
        buf = io.BytesIO(); small.resize((800, 800), PILImage.BILINEAR).convert('RGB').save(buf, 'PNG')
        return buf.getvalue()
    ds_id = client.post('/api/dataset/create', json={
        'name': 'CDup', 'trigger_word': 'cdup', 'kind': 'concept',
        'concept_desc': 'a test concept'}).get_json()['id']
    data = {'files': [(io.BytesIO(grad('ltr')), 'a.png'),
                      (io.BytesIO(grad('ltr')), 'b.png'),
                      (io.BytesIO(grad('rtl')), 'c.png')]}
    resp = client.post(f'/api/dataset/{ds_id}/import', data=data,
                       content_type='multipart/form-data')
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['imported'] == 2 and body['duplicates'] == 1 and body['failed'] == 0


def test_captions_replace_route(client, app):
    ds_id = _create(client, 'Rep', 'rep').get_json()['id']
    with app.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDatasetImage
        svc.db.session.add(FaceDatasetImage(dataset_id=ds_id, filename='x.webp',
                                            status='keep', caption='a red dress'))
        svc.db.session.commit()
    assert client.post(f'/api/dataset/{ds_id}/captions/replace',
                       json={'find': '', 'replace': 'x'}).status_code == 400
    assert client.post('/api/dataset/999999/captions/replace',
                       json={'find': 'red', 'replace': 'blue'}).status_code == 404
    ok = client.post(f'/api/dataset/{ds_id}/captions/replace',
                     json={'find': 'red', 'replace': 'blue'})
    assert ok.status_code == 200 and ok.get_json() == {'ok': True, 'changed': 1}
    payload = client.get(f'/api/dataset/{ds_id}').get_json()
    assert payload['images'][0]['caption'] == 'a blue dress'


def test_captions_write_files_route(client, app):
    """💾 Write .txt files: kohya-style same-stem sidecars next to the KEPT
    captioned images only, trigger prepended like the export ZIP. Uncaptioned
    kept images are counted as skipped (not written), rejects are ignored,
    and a re-call after a caption edit overwrites the file (resync)."""
    import os
    ds_id = _create(client, 'Sidecar', 'sidetrig').get_json()['id']
    with app.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDatasetImage
        d = svc._dataset_dir(ds_id)
        for name in ('a.webp', 'b.webp', 'c.webp'):
            open(os.path.join(d, name), 'wb').write(_png_bytes())
        rows = [FaceDatasetImage(dataset_id=ds_id, filename='a.webp', status='keep',
                                 caption='a red dress'),
                FaceDatasetImage(dataset_id=ds_id, filename='b.webp', status='keep',
                                 caption=''),
                FaceDatasetImage(dataset_id=ds_id, filename='c.webp', status='reject',
                                 caption='rejected caption')]
        svc.db.session.add_all(rows); svc.db.session.commit()
        captioned_id = rows[0].id
    assert client.post('/api/dataset/999999/captions/write-files').status_code == 404
    resp = client.post(f'/api/dataset/{ds_id}/captions/write-files')
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True, 'written': 1, 'skipped_uncaptioned': 1}
    with app.app_context():
        from app.services import face_dataset_service as svc
        d = svc._dataset_dir(ds_id)
        with open(os.path.join(d, 'a.txt'), encoding='utf-8') as fh:
            assert fh.read() == 'sidetrig, a red dress'
        assert not os.path.exists(os.path.join(d, 'b.txt'))   # uncaptioned -> skipped
        assert not os.path.exists(os.path.join(d, 'c.txt'))   # reject -> ignored
    # Resync: edit the caption, call again -> same envelope, file overwritten.
    client.post(f'/api/dataset/image/{captioned_id}/caption', json={'caption': 'a blue dress'})
    resp2 = client.post(f'/api/dataset/{ds_id}/captions/write-files')
    assert resp2.get_json() == {'ok': True, 'written': 1, 'skipped_uncaptioned': 1}
    with app.app_context():
        from app.services import face_dataset_service as svc
        with open(os.path.join(svc._dataset_dir(ds_id), 'a.txt'), encoding='utf-8') as fh:
            assert fh.read() == 'sidetrig, a blue dress'


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


# --- Import from folder (kohya folder already on the server's disk) ----------
def _patterned_png(seed, base=(255, 255, 255)):
    """Distinct NON-uniform image: solid colors all share the same (zero) dHash
    and would read as perceptual duplicates of each other (cf. test_dataset_service)."""
    im = Image.new('RGB', (64, 64), base)
    for i in range(8):
        x = (seed * 13 + i * 7) % 56
        im.paste(((seed * 37) % 255, (i * 61) % 255, (seed * 7 + i * 29) % 255),
                 (x, i * 8, x + 8, i * 8 + 8))
    buf = io.BytesIO(); im.save(buf, 'PNG')
    return buf.getvalue()


def test_import_folder_route_images_captions_and_nonimage(client, app, tmp_path):
    """Kohya folder: images (any depth) + same-stem .txt sidecars become rows with
    captions, the caption lands on the MATCHING image, non-image files are ignored."""
    import os
    ds_id = _create(client, 'Folder', 'folder').get_json()['id']
    src = tmp_path / 'kohya'; (src / 'sub').mkdir(parents=True)
    (src / 'a.png').write_bytes(_patterned_png(1, base=(220, 30, 30)))     # red
    (src / 'a.txt').write_text('a red patterned square', encoding='utf-8')
    (src / 'sub' / 'b.png').write_bytes(_patterned_png(2, base=(30, 30, 220)))  # blue
    (src / 'notes.md').write_text('ignore me', encoding='utf-8')
    resp = client.post(f'/api/dataset/{ds_id}/import-folder', json={'path': str(src)})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['imported'] == 2 and body['captions'] == 1 and body['failed'] == 0
    with app.app_context():
        from app.services import face_dataset_service as svc
        from app.models import FaceDatasetImage
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds_id).all()
        assert len(rows) == 2   # notes.md ignored
        assert all(r.status == 'keep' and r.source == 'import' for r in rows)
        captioned = [r for r in rows if r.caption]
        assert len(captioned) == 1 and captioned[0].caption == 'a red patterned square'
        # the caption landed on the RED image (a.png), not the blue one
        with Image.open(os.path.join(svc._dataset_dir(ds_id), captioned[0].filename)) as im:
            r, _g, b = im.convert('RGB').resize((1, 1)).getpixel((0, 0))
        assert r > b


def test_import_folder_route_missing_or_bad_path(client, tmp_path):
    ds_id = _create(client, 'FolderBad', 'folderbad').get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/import-folder',
                       json={'path': str(tmp_path / 'does-not-exist')})
    assert resp.status_code == 400
    assert 'not found' in resp.get_json()['error']
    assert client.post(f'/api/dataset/{ds_id}/import-folder', json={}).status_code == 400
    assert client.post('/api/dataset/999999/import-folder',
                       json={'path': str(tmp_path)}).status_code == 404


def test_import_folder_route_reimport_dedupes(client, tmp_path):
    """Importing the same folder twice must not duplicate anything (dHash vs the
    dataset's existing rows)."""
    ds_id = _create(client, 'FolderDup', 'folderdup').get_json()['id']
    src = tmp_path / 'kohya'; src.mkdir()
    (src / 'a.png').write_bytes(_patterned_png(3))
    (src / 'b.png').write_bytes(_patterned_png(4))
    first = client.post(f'/api/dataset/{ds_id}/import-folder', json={'path': str(src)}).get_json()
    assert first['imported'] == 2
    again = client.post(f'/api/dataset/{ds_id}/import-folder', json={'path': str(src)}).get_json()
    assert again['imported'] == 0 and again['duplicates'] == 2
