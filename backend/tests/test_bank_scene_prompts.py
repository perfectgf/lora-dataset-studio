"""🎬 Scenes from a bank — the bank's captions read as an ORDERED sequence of
prompts, the source the generation panels (Test Studio and the board's Generate
panel) offer as extra passes of the 📝 prompt axis.

The contract these tests hold:
  - bank order IS the sequence (row id ascending = import order);
  - a missing FRAMING is not a gate (a beat in the middle of a story must not
    silently disappear) — it rides as 'body';
  - a missing CAPTION skips, and is COUNTED, never guessed;
  - prompts stay under the shot importer's 500-char ceiling, cut on a word;
  - the route is read-only and 404s on a bank that does not exist.
"""


def _bank_with_captions(tmp_path, captions, *, framings=None, name='Chapter'):
    """A real bank of len(captions) images, captioned in order. Returns its id."""
    from PIL import Image
    from app.extensions import db
    from app.services import image_bank_service as banks
    from app.models import BankImage
    src = tmp_path / name
    src.mkdir()
    for i in range(len(captions)):
        Image.new('RGB', (60, 90), (i * 7 % 255, 80, 150)).save(str(src / f'p{i:03d}.jpg'))
    bank, _ = banks.create_bank('local', name, str(src))
    db.session.commit()
    rows = (BankImage.query.filter_by(bank_id=bank.id)
            .order_by(BankImage.id.asc()).all())
    for row, caption in zip(rows, captions):
        row.caption = caption
    if framings:
        for row, framing in zip(rows, framings):
            row.framing = framing
    db.session.commit()
    return bank.id


def test_scenes_come_out_in_bank_order(app, tmp_path):
    from app.services import image_bank_service as banks
    with app.app_context():
        bank_id = _bank_with_captions(tmp_path, ['street at dawn',
                                                 'close on her face',
                                                 'rooftop chase'])
        out = banks.export_scene_captions('local', bank_id)
    assert out['bank_name'] == 'Chapter'
    assert [s['prompt'] for s in out['scenes']] == [
        'street at dawn', 'close on her face', 'rooftop chase']
    # The label carries the sequence number, so the order stays readable
    # wherever the card lands.
    assert out['scenes'][2]['label'].startswith('Scene 3 — p002.jpg')
    assert out['skipped'] == {'no_caption': 0}


def test_a_page_without_a_caption_is_skipped_and_counted(app, tmp_path):
    """Never guessed: a captionless image cannot become a prompt, but the user
    must be told how many stayed behind rather than silently losing beats."""
    from app.services import image_bank_service as banks
    with app.app_context():
        bank_id = _bank_with_captions(tmp_path, ['one', '', '   ', 'four'])
        out = banks.export_scene_captions('local', bank_id)
    assert [s['prompt'] for s in out['scenes']] == ['one', 'four']
    assert out['skipped']['no_caption'] == 2
    # Numbering follows the KEPT scenes, not the bank rows.
    assert [s['label'].split(' — ')[0] for s in out['scenes']] == ['Scene 1', 'Scene 2']


def test_a_missing_framing_rides_as_body_instead_of_dropping_the_scene(app, tmp_path):
    """Unlike a shot-card export, framing is NOT a gate here: refusing a page
    would drop a beat from the middle of the sequence."""
    from app.services import image_bank_service as banks
    with app.app_context():
        bank_id = _bank_with_captions(
            tmp_path, ['a', 'b', 'c'], framings=['face', 'unknown', None])
        out = banks.export_scene_captions('local', bank_id)
    assert [s['framing'] for s in out['scenes']] == ['face', 'body', 'body']
    assert out['skipped']['no_caption'] == 0


def test_a_long_caption_is_cut_on_a_word_under_the_importer_ceiling(app, tmp_path):
    from app.services import image_bank_service as banks
    with app.app_context():
        long_caption = ('word ' * 200).strip()
        bank_id = _bank_with_captions(tmp_path, [long_caption])
        out = banks.export_scene_captions('local', bank_id)
    prompt = out['scenes'][0]['prompt']
    assert len(prompt) <= banks.SCENE_MAX_PROMPT
    assert prompt.endswith('…')
    assert 'wor…' not in prompt          # cut on a word boundary, never mid-word


def test_each_scene_carries_its_page_id_for_the_thumbnail(app, tmp_path):
    """Display-only: the panel renders the bank thumb of the page a scene came
    from. Generation payloads never carry it."""
    from app.models import BankImage
    from app.services import image_bank_service as banks
    with app.app_context():
        bank_id = _bank_with_captions(tmp_path, ['a', 'b'])
        out = banks.export_scene_captions('local', bank_id)
        ids = [r.id for r in (BankImage.query.filter_by(bank_id=bank_id)
                              .order_by(BankImage.id.asc()).all())]
    assert [s['image_id'] for s in out['scenes']] == ids


def test_scenes_route_serves_the_ordered_cards_and_404s_on_a_missing_bank(app, client, tmp_path):
    with app.app_context():
        bank_id = _bank_with_captions(tmp_path, ['panel one', 'panel two'])
    r = client.get(f'/api/bank/{bank_id}/scenes')
    assert r.status_code == 200
    body = r.get_json()
    assert body['bank_id'] == bank_id
    assert [s['prompt'] for s in body['scenes']] == ['panel one', 'panel two']
    assert client.get('/api/bank/999999/scenes').status_code == 404
