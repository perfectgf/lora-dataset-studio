"""🗃️ Image bank — framing classification (face/bust/body/back) and the coverage
advice panel (idea by @antonp). The framing pass reuses the dataset Qwen3-VL
classifier (mocked here so the pass is hermetic); coverage is pure DB math over
the fields the passes persist, so it's seeded directly. Background jobs run inline
under TESTING (see bank_jobs.start)."""
import os

from PIL import Image

from app.services import image_bank_service as banks


# --- factories (mirror test_image_bank_scoring) ------------------------------
def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith(('.jpg', '.jpeg')):
        im.save(path, 'JPEG', quality=92)
    else:
        im.save(path)


def _flat(size=128, value=128):
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


def _set_fields(app, bank_id, **by_name):
    """Write arbitrary fields straight onto rows, keyed by file basename."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        rows = {os.path.basename(r.relpath): r
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        for name, vals in by_name.items():
            for k, v in vals.items():
                setattr(rows[name], k, v)
        db.session.commit()


def _seed_rows(app, bank_id, n, mk, _c=[0]):
    """Insert N synthetic rows (no files on disk needed for pure-DB coverage math)
    via ``mk(i) -> dict of BankImage fields``. Relpaths are globally unique across
    calls so several _seed_rows on one bank never collide."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        for i in range(n):
            _c[0] += 1
            db.session.add(BankImage(bank_id=bank_id, relpath=f'seed_{_c[0]}.png', **mk(i)))
        db.session.commit()


def _wipe(app, bank_id):
    """Drop the bootstrap scanned row(s) so a coverage pool holds ONLY seeds."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        BankImage.query.filter_by(bank_id=bank_id).delete()
        db.session.commit()


# --- framing pass (mocked classifier — hermetic) -----------------------------
def test_framing_gate_503_when_model_absent(client, tmp_path, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_ollama_model',
                        lambda *a, **k: {'ok': False, 'detail': 'not pulled'})
    r = client.post(f'/api/bank/{bank_id}/framing', json={})
    assert r.status_code == 503
    assert 'vision model' in r.get_json()['error']


def test_framing_refuses_when_gpu_busy(client, tmp_path, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_ollama_model', lambda *a, **k: {'ok': True})
    monkeypatch.setattr(banks, '_gpu_busy_reason', lambda: 'training is running on the GPU')
    r = client.post(f'/api/bank/{bank_id}/framing', json={})
    assert r.status_code == 503
    assert 'training' in r.get_json()['error']


def test_framing_job_classifies_persists_and_filters(client, tmp_path, app, monkeypatch):
    # Three distinct flat values so the mock can key each shot type off the pixel;
    # a fourth returns EMPTY (Ollama unreachable) and must stay NULL for a retry.
    bank_id, _ = _mkbank(client, tmp_path, {
        'face.png': _flat(value=128), 'body.png': _flat(value=60),
        'back.png': _flat(value=90), 'down.png': _flat(value=20)})
    from app import capabilities
    from app.services import vision_ollama
    monkeypatch.setattr(capabilities, 'probe_ollama_model', lambda *a, **k: {'ok': True})

    def fake_describe(image_bytes, *a, **k):
        import io
        v = Image.open(io.BytesIO(image_bytes)).convert('L').getpixel((0, 0))
        return {128: '{"framing":"face","angle":"front","expression":"neutral"}',
                60: '{"framing":"body","angle":"front","expression":"smile"}',
                90: '{"framing":"back","angle":"back","expression":"neutral"}',
                20: ''}.get(v, '')

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)

    r = client.post(f'/api/bank/{bank_id}/framing', json={})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['finished'] is True
    assert payload['activity']['error'] is None

    by = _by_name(client, bank_id)
    assert by['face.png']['framing'] == 'face'
    assert by['body.png']['framing'] == 'body'
    assert by['back.png']['framing'] == 'back'
    assert by['down.png']['framing'] is None            # empty → left for a retry

    # Facet counts on the payload.
    assert payload['framing'] == {'face': 1, 'bust': 0, 'body': 1, 'back': 1, 'unknown': 0}
    assert payload['counts']['framing_classified'] == 3

    # The framing chip filters the grid and composes with other facets.
    hits = client.get(f'/api/bank/{bank_id}/images?framing=face').get_json()
    assert [i['name'] for i in hits['images']] == ['face.png']
    hits = client.get(f'/api/bank/{bank_id}/images?framing=body').get_json()
    assert [i['name'] for i in hits['images']] == ['body.png']


def test_framing_rejected_images_skipped(client, tmp_path, app, monkeypatch):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat(value=128), 'b.png': _flat(value=60)})
    from app import capabilities
    from app.services import vision_ollama
    monkeypatch.setattr(capabilities, 'probe_ollama_model', lambda *a, **k: {'ok': True})
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: '{"framing":"face"}')
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.png']['id']], 'status': 'reject'})
    client.post(f'/api/bank/{bank_id}/framing', json={})
    by = _by_name(client, bank_id)
    assert by['a.png']['framing'] is None               # rejected → never touched
    assert by['b.png']['framing'] == 'face'


def test_framing_unknown_answer_stored(app, tmp_path, monkeypatch, client):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    from app import capabilities
    from app.services import vision_ollama
    monkeypatch.setattr(capabilities, 'probe_ollama_model', lambda *a, **k: {'ok': True})
    # A parseable JSON whose framing isn't one of the four → 'unknown' (terminal,
    # but re-runnable via rescan), distinct from an empty answer (retryable NULL).
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: '{"framing":"weird"}')
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    client.post(f'/api/bank/{bank_id}/framing', json={})
    assert _by_name(client, bank_id)['a.png']['framing'] == 'unknown'


# --- coverage advice (pure DB — seeded) --------------------------------------
def test_coverage_hint_when_framing_never_ran(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _wipe(app, bank_id)
    _seed_rows(app, bank_id, 25, lambda i: {'status': 'keep',
                                            'width': 1200, 'height': 1200})
    cov = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert cov['pool'] == 'kept'
    texts = ' '.join(a['text'] for a in cov['advice'])
    assert 'Framing pass' in texts                      # honest hint, never a mute ✗
    assert cov['framing_available'] is False


def test_coverage_flags_framing_imbalance(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _wipe(app, bank_id)
    # 18 face + 2 body kept = 90% face → the imbalance warning, suggesting the thin
    # axes (bust/back are 0%, body is 10%). Resolution comfortably above 1 MP.
    _seed_rows(app, bank_id, 18, lambda i: {'status': 'keep', 'framing': 'face',
                                            'width': 1500, 'height': 1500})
    _seed_rows(app, bank_id, 2, lambda i: {'status': 'keep', 'framing': 'body',
                                           'width': 1500, 'height': 1500})
    cov = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert cov['framing_available'] is True
    assert cov['framing'] == {'face': 18, 'bust': 0, 'body': 2, 'back': 0, 'unknown': 0}
    warn = [a for a in cov['advice'] if a['tone'] == 'warn']
    assert any('90% face' in a['text'] for a in warn)
    # The suggestion names the thin buckets (body/back/bust), not "face".
    imbal = next(a for a in warn if 'face shots' in a['text'])
    assert 'body' in imbal['text'] or 'back' in imbal['text']


def test_coverage_person_dominance_and_singletons(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _wipe(app, bank_id)
    # cluster 1 = 12 imgs (60%), then clusters 2..5 = one each (4 singletons).
    _seed_rows(app, bank_id, 12, lambda i: {'status': 'keep', 'face_cluster': 1,
                                            'width': 1200, 'height': 1200})
    _seed_rows(app, bank_id, 4, lambda i: {'status': 'keep', 'face_cluster': i + 2,
                                           'width': 1200, 'height': 1200})
    cov = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    texts = [a['text'] for a in cov['advice']]
    assert any('Person #1' in t and '75%' in t for t in texts)   # 12 / 16
    assert any('appear only once' in t for t in texts)


def test_coverage_low_resolution_share(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _wipe(app, bank_id)
    _seed_rows(app, bank_id, 15, lambda i: {'status': 'keep', 'width': 512, 'height': 512})
    _seed_rows(app, bank_id, 15, lambda i: {'status': 'keep', 'width': 2000, 'height': 2000})
    cov = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert any('under' in a['text'] and '1 MP' in a['text'] for a in cov['advice'])


def test_coverage_small_kept_set_warned(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _wipe(app, bank_id)
    _seed_rows(app, bank_id, 5, lambda i: {'status': 'keep', 'framing': 'face',
                                           'width': 1500, 'height': 1500})
    cov = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert any('Only 5' in a['text'] and a['tone'] == 'warn' for a in cov['advice'])


def test_coverage_falls_back_to_candidates_before_any_keep(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _wipe(app, bank_id)
    _seed_rows(app, bank_id, 25, lambda i: {'status': 'pending', 'framing': 'face',
                                            'width': 1500, 'height': 1500})
    _seed_rows(app, bank_id, 3, lambda i: {'status': 'reject', 'framing': 'body',
                                           'width': 1500, 'height': 1500})
    cov = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    # No keep yet → advises on non-rejected candidates (the rejects are excluded).
    assert cov['pool'] == 'candidates'
    assert cov['total'] == 25
    assert cov['framing']['body'] == 0


def test_coverage_empty_pool_is_gentle(client, tmp_path, app):
    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat()})
    _wipe(app, bank_id)
    _seed_rows(app, bank_id, 2, lambda i: {'status': 'reject'})
    cov = client.get(f'/api/bank/{bank_id}/coverage').get_json()
    assert cov['total'] == 0
    assert len(cov['advice']) == 1 and cov['advice'][0]['tone'] == 'info'


def test_coverage_404_when_bank_gone(client):
    assert client.get('/api/bank/999999/coverage').status_code == 404
