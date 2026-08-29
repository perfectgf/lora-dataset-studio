"""🧪 Caption Lab on the BANK — the mirror of test_caption_lab_preview.py.

Two surfaces, ONE contract. The bench that decides what a candidate means lives in
face_dataset_service.preview_caption_path, and both surfaces go through it: the last
test here pins that, because a second hand-maintained copy of the config validation is
exactly the divergence CLAUDE.md's "two surfaces of one product" section is about (the
face size gate shipped twice, drifted, and was reported on the side nobody fixed).

Also covers the write the bench needed: until now the Bank had no per-image caption
editor at all, so ✓ Keep this one had nowhere to put a winning candidate.
"""
import pytest
from PIL import Image


def _use_ollama_backend(app):
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': 'ollama'}})


def _mock_vision(monkeypatch, caption='a plain description', capture=None):
    from app.services import vision_ollama

    def fake_describe(image_bytes, prompt, *a, **k):
        if capture is not None:
            capture['prompt'] = prompt
            capture['model'] = k.get('model')
        return caption

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


def _bank_with_image(app, tmp_path, *, caption='', name='Lab bank'):
    """A bank scanned from a real folder holding one readable photo."""
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        from app.services import image_bank_service as banks

        folder = tmp_path / 'lab-bank-source'
        folder.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (128, 128), (90, 110, 130)).save(folder / 'shot.jpg')
        bank, added = banks.create_bank('local', name, str(folder))
        assert added == 1
        row = BankImage.query.filter_by(bank_id=bank.id).one()
        row.status = 'keep'
        row.caption = caption
        db.session.commit()
        return bank.id, row.id


def _preview(client, bank_id, image_id, **body):
    return client.post(
        f'/api/bank/{bank_id}/image/{image_id}/caption/preview', json=body)


# --- the bench ----------------------------------------------------------------

def test_preview_returns_a_caption_and_writes_nothing(client, app, tmp_path, monkeypatch):
    _use_ollama_backend(app)
    bank_id, image_id = _bank_with_image(app, tmp_path, caption='ORIGINAL')
    _mock_vision(monkeypatch, caption='a candidate caption')

    r = _preview(client, bank_id, image_id)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['ok'] is True
    assert body['caption'] == 'a candidate caption'
    assert body['chars'] == len('a candidate caption')
    assert 'duration_ms' in body and body['cancelled'] is False

    # A bench never writes — the same promise the dataset side makes.
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        assert db.session.get(BankImage, image_id).caption == 'ORIGINAL'


def test_preview_appends_the_register_and_the_extra_instructions(
        client, app, tmp_path, monkeypatch):
    _use_ollama_backend(app)
    bank_id, image_id = _bank_with_image(app, tmp_path)
    capture = {}
    _mock_vision(monkeypatch, capture=capture)

    r = _preview(client, bank_id, image_id,
                 vocabulary='clinical', length='concise',
                 instructions='mention the window')
    assert r.status_code == 200, r.get_json()
    prompt = capture['prompt']
    assert 'mention the window' in prompt
    # The register and the length ride in as instructions, exactly as on a dataset.
    assert len(prompt) > len('mention the window')


def test_preview_refuses_an_unknown_image_and_an_impossible_candidate(
        client, app, tmp_path, monkeypatch):
    _use_ollama_backend(app)
    bank_id, image_id = _bank_with_image(app, tmp_path)
    _mock_vision(monkeypatch)

    assert _preview(client, bank_id, 987654321).status_code == 404
    # An unknown BANK is a 404 too — the dataset twin answers 404 for an unknown dataset,
    # and the same mistake must not get two different codes on two surfaces of one product.
    assert _preview(client, 987654321, image_id).status_code == 404
    # 'none' is captioning disabled: a candidate that captions nothing is not a
    # candidate, and it is refused here rather than answered with an empty string.
    assert _preview(client, bank_id, image_id, backend='none').status_code == 400
    assert _preview(client, bank_id, image_id, backend='nope').status_code == 400
    assert _preview(client, bank_id, image_id, vocabulary='shouty').status_code == 400
    assert _preview(client, bank_id, image_id, length='enormous').status_code == 400


def test_preview_is_refused_while_a_pass_holds_the_bank(client, app, tmp_path, monkeypatch):
    """The bench owns the GPU for seconds, so it takes the bank lease like a pass.
    A pass already holding it must refuse the bench with `busy_kind`, not a bare 409:
    that field is what lets the UI say WHICH pass is in the way."""
    _use_ollama_backend(app)
    bank_id, image_id = _bank_with_image(app, tmp_path)
    _mock_vision(monkeypatch)
    from app.services import bank_jobs

    held = bank_jobs.reserve(bank_id, 'score')
    try:
        r = _preview(client, bank_id, image_id)
        assert r.status_code == 409, r.get_json()
        assert r.get_json().get('busy_kind') == 'score'
    finally:
        bank_jobs.abort(held)


# --- the write the bench needed ------------------------------------------------

def test_a_hand_written_bank_caption_lands_stamped_asserted(client, app, tmp_path):
    bank_id, image_id = _bank_with_image(app, tmp_path, caption='machine text')

    r = client.put(f'/api/bank/{bank_id}/image/{image_id}/caption',
                   json={'caption': '  a caption I wrote  '})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['caption'] == 'a caption I wrote'
    assert r.get_json()['caption_origin'] == 'asserted'

    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        from app.services import caption_origin
        row = db.session.get(BankImage, image_id)
        assert row.caption == 'a caption I wrote'
        # …and the protection the batch pass already honours now has a local writer.
        assert caption_origin.is_protected(row) is True


def test_clearing_a_bank_caption_drops_its_stamp_too(client, app, tmp_path):
    """A blanked caption that kept an 'asserted' label would be spared by a forced
    Re-caption forever — a row protected on the strength of text that is gone."""
    bank_id, image_id = _bank_with_image(app, tmp_path)
    client.put(f'/api/bank/{bank_id}/image/{image_id}/caption', json={'caption': 'mine'})

    r = client.put(f'/api/bank/{bank_id}/image/{image_id}/caption', json={'caption': ''})
    assert r.status_code == 200, r.get_json()
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        row = db.session.get(BankImage, image_id)
        assert not (row.caption or '')
        assert row.caption_origin is None


def test_the_caption_write_refuses_a_bad_body_and_an_unknown_image(client, app, tmp_path):
    bank_id, image_id = _bank_with_image(app, tmp_path)
    assert client.put(f'/api/bank/{bank_id}/image/987654321/caption',
                      json={'caption': 'x'}).status_code == 404
    assert client.put(f'/api/bank/{bank_id}/image/{image_id}/caption',
                      json={'caption': 42}).status_code == 400
    assert client.put(f'/api/bank/{bank_id}/image/{image_id}/caption',
                      json=[1, 2]).status_code == 400


# --- the parity pin ------------------------------------------------------------

@pytest.mark.parametrize('surface', ['dataset', 'bank'])
def test_both_surfaces_bench_through_the_same_definition_of_a_candidate(
        surface, client, app, tmp_path, monkeypatch):
    """WHAT THIS PROTECTS. A candidate is engine x vision model x register x length.
    If the Bank ever grows its own copy of that validation, the two surfaces drift and
    nothing fails — the user finds out when the same config means two things. So both
    routes are driven here and BOTH must land in preview_caption_path with the same
    keyword arguments; a copy would show up as a call that never arrives."""
    import os

    from app.services import face_dataset_service as datasets

    seen = []

    def spy(path, **kwargs):
        seen.append(kwargs)
        assert os.path.isfile(path), 'the surface must hand the bench a real file'
        return {'caption': 'x', 'chars': 1, 'duration_ms': 1, 'cancelled': False}

    monkeypatch.setattr(datasets, 'preview_caption_path', spy)
    body = {'backend': 'ollama', 'ollama_model': 'some-model',
            'vocabulary': 'clinical', 'length': 'concise', 'instructions': 'be brief'}

    if surface == 'bank':
        bank_id, image_id = _bank_with_image(app, tmp_path)
        r = client.post(f'/api/bank/{bank_id}/image/{image_id}/caption/preview', json=body)
    else:
        ds_id = client.post('/api/dataset/create',
                            json={'name': 'Lab', 'trigger_word': 'lab'}).get_json()['id']
        with app.app_context():
            from app.models import FaceDatasetImage
            from app.services.dataset_storage import ensure_dataset_dir
            Image.new('RGB', (64, 64), (20, 20, 20)).save(
                os.path.join(ensure_dataset_dir(ds_id), 'a.png'))
            img = FaceDatasetImage(dataset_id=ds_id, status='keep', source='upload',
                                   filename='a.png')
            datasets.db.session.add(img)
            datasets.db.session.commit()
            image_id = img.id
        r = client.post(f'/api/dataset/{ds_id}/image/{image_id}/caption/preview', json=body)

    assert r.status_code == 200, r.get_json()
    assert len(seen) == 1, f'{surface} did not reach the shared bench'
    for key, value in body.items():
        assert seen[0][key] == value, f'{surface} altered {key} on the way in'


# --- what the bench and the write may NOT reach ---------------------------------

def test_neither_route_can_reach_an_image_of_another_bank(client, app, tmp_path, monkeypatch):
    """The row is looked up with BOTH filters (bank_id AND id). Nothing failed when that
    was true, so nothing would fail if a later edit dropped one of them — which is the
    whole point of pinning it: a bench or a write that crosses banks would be silent."""
    _use_ollama_backend(app)
    _mock_vision(monkeypatch)
    bank_a, image_a = _bank_with_image(app, tmp_path / 'a', name='Bank A')
    bank_b, image_b = _bank_with_image(app, tmp_path / 'b', name='Bank B')
    assert bank_a != bank_b and image_a != image_b

    # …each is reachable from its OWN bank,
    assert _preview(client, bank_a, image_a).status_code == 200
    # …and invisible from the other one.
    assert _preview(client, bank_a, image_b).status_code == 404
    assert _preview(client, bank_b, image_a).status_code == 404
    assert client.put(f'/api/bank/{bank_a}/image/{image_b}/caption',
                      json={'caption': 'crossed'}).status_code == 404

    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        assert not (db.session.get(BankImage, image_b).caption or '')


def test_a_hand_written_bank_caption_is_capped_like_the_dataset_editor(client, app, tmp_path):
    """Every other caption writer in the app goes through _cap_caption. This one is the
    Bank's only local writer, and its text is read by the search, by promotion and by the
    training export — an unbounded row here would be the single exception."""
    from app.services.face_dataset_service import CAPTION_MAX_CHARS

    bank_id, image_id = _bank_with_image(app, tmp_path)
    r = client.put(f'/api/bank/{bank_id}/image/{image_id}/caption',
                   json={'caption': 'x' * (CAPTION_MAX_CHARS + 5000)})
    assert r.status_code == 200, r.get_json()
    assert len(r.get_json()['caption']) <= CAPTION_MAX_CHARS
