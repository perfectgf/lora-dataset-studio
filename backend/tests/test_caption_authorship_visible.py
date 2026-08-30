"""WHO wrote a caption, said where the caption is READ — and counted when it matters.

`caption_origin` has been stamped by every caption writer since caption_origin.py
landed, and it was used for exactly ONE decision: which rows a forced re-caption
spares.  The screens where a caption is read still showed an anonymous string, and
the run reported "427 captioned" over two halves written by two different engines.

These tests measure the PROPERTY on each side of the wire:
  * the payloads really carry the stamp, per image and per field;
  * the pass's own result names the writers, counted from the rows it STAMPED —
    not from the pool it started with, and not from the backend it was asked for;
  * the launch window's NSFW figures divide by what ✨ Score measured, never by the
    pile (an image nobody scored is unknown, not clean).
"""
import os
import random

from PIL import Image


# --- factories (same shape as test_caption_provenance.py) --------------------
def _flat(value=128, size=64):
    rnd = random.Random(value)
    im = Image.new('RGB', (size, size))
    px = im.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
    px[0, 0] = (value, value, value)
    return im


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / name
    for rel, im in files.items():
        os.makedirs(os.path.dirname(str(src / rel)), exist_ok=True)
        im.save(str(src / rel))
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


# --- the stamp reaches the screen -------------------------------------------
def test_the_bank_grid_payload_carries_who_wrote_each_caption(client, app, tmp_path):
    from app.extensions import db
    from app.models import BankImage
    from app.services import caption_origin

    bank_id, _ = _mkbank(client, tmp_path, {
        'a.png': _flat(11), 'b.png': _flat(12), 'c.png': _flat(13), 'd.png': _flat(14)})
    with app.app_context():
        by = {r.relpath: r for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        caption_origin.stamp(by['a.png'], 'joy wrote this', caption_origin.JOYCAPTION)
        caption_origin.stamp(by['b.png'], 'qwen wrote this', caption_origin.OLLAMA)
        caption_origin.stamp(by['c.png'], 'I wrote this', caption_origin.ASSERTED)
        by['d.png'].caption = 'nobody knows'      # a row that predates the column
        by['d.png'].caption_origin = None
        db.session.commit()

    rows = client.get(f'/api/bank/{bank_id}/images').get_json()['images']
    got = {r['name']: r.get('caption_origin') for r in rows}
    assert got == {'a.png': 'joycaption', 'b.png': 'ollama',
                   'c.png': 'asserted', 'd.png': None}
    # NULL travels as NULL. Defaulting it to an engine name would be the app
    # claiming to know something nobody recorded, on every legacy row at once.
    assert 'caption_origin' in rows[0]


def test_the_dataset_payload_attributes_each_of_the_two_captions_apart(
        client, app, tmp_path):
    """The short caption has its own writers, so it needs its own stamp on the wire."""
    from app.extensions import db
    from app.config import LOCAL_USER
    from app.models import FaceDataset, FaceDatasetImage
    from app.services import caption_origin

    with app.app_context():
        ds = FaceDataset(user_id=LOCAL_USER, name='D', kind='character',
                         trigger_word='trg')
        db.session.add(ds)
        db.session.commit()
        img = FaceDatasetImage(dataset_id=ds.id, filename='x.png', status='keep',
                               source='upload')
        db.session.add(img)
        db.session.commit()
        caption_origin.stamp(img, 'the long one', caption_origin.OLLAMA)
        caption_origin.stamp(img, 'the short one', caption_origin.ASSERTED,
                             field='caption_short')
        db.session.commit()
        ds_id = ds.id

    row = client.get(f'/api/dataset/{ds_id}').get_json()['images'][0]
    assert row['caption_origin'] == 'ollama'
    # The two must not answer for each other: a model wrote the long one and the
    # user wrote the short one, and one column cannot say both.
    assert row['caption_short_origin'] == 'asserted'


# --- the run says who wrote what --------------------------------------------
def test_the_pass_result_names_the_writers_and_counts_what_it_stamped(
        client, app, tmp_path, monkeypatch):
    """'auto' is a CHAIN, so the requested backend would mislabel half the bank.

    The mock captions two images through the JoyCaption seam and one through the
    Ollama one — the split the real chain produces — and the result line is
    asserted against the ROWS, so a note built from the requested backend or from
    the pool size fails here.
    """
    import io

    from app.services import face_dataset_service as fds

    bank_id, _ = _mkbank(client, tmp_path, {
        'j1.png': _flat(21), 'j2.png': _flat(22), 'o1.png': _flat(23)})

    def fake_caption_paths(paths, *a, on_caption=None, progress=None, **k):
        for p in paths:
            with open(p, 'rb') as fh:
                v = Image.open(io.BytesIO(fh.read())).convert('L').getpixel((0, 0))
            engine = 'ollama' if v == 23 else 'joycaption'
            on_caption(p, f'caption for {v}', engine)
        if progress:
            progress(len(paths), len(paths))

    monkeypatch.setattr(fds, 'caption_paths', fake_caption_paths)

    r = client.post(f'/api/bank/{bank_id}/caption', json={})
    assert r.status_code == 202
    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['activity']['error'] is None
    detail = payload['activity']['detail']
    assert '3 captioned' in detail
    # The two engines, named apart, in canonical order, with the numbers the rows
    # really carry — not "3 by the configured backend".
    # Parenthesised, so the comma-separated skip counts that follow can never read
    # as more writers.
    # The local engine is named NEUTRALLY: the stored origin is still 'ollama',
    # but it means 'the configured local provider' now, and a run served by LM
    # Studio used to be reported on this very line as Ollama's work.
    assert '(2 by JoyCaption, 1 by the local LLM vision model)' in detail
    assert 'Ollama' not in detail, 'the pass named one provider on a line both write'

    from app.models import BankImage
    with app.app_context():
        origins = sorted(r.caption_origin for r
                         in BankImage.query.filter_by(bank_id=bank_id).all())
    assert origins == ['joycaption', 'joycaption', 'ollama']


def test_a_write_whose_engine_reported_no_name_is_not_attributed_to_one(
        client, app, tmp_path, monkeypatch):
    """NULL means 'never recorded'.  It is counted, and it is counted APART."""
    from app.services import face_dataset_service as fds

    bank_id, _ = _mkbank(client, tmp_path, {'a.png': _flat(31), 'b.png': _flat(32)})

    def fake_caption_paths(paths, *a, on_caption=None, progress=None, **k):
        for p in paths:
            on_caption(p, 'a caption', None)      # an engine that names nobody
        if progress:
            progress(len(paths), len(paths))

    monkeypatch.setattr(fds, 'caption_paths', fake_caption_paths)
    assert client.post(f'/api/bank/{bank_id}/caption', json={}).status_code == 202
    detail = client.get(f'/api/bank/{bank_id}').get_json()['activity']['detail']
    assert '2 captioned' in detail
    assert 'engine did not report a name' in detail
    assert 'JoyCaption' not in detail and 'Ollama' not in detail


def test_the_writers_note_is_empty_when_nothing_was_written():
    """A pass that wrote nothing says nothing — never 'written by nobody'."""
    from app.services.image_bank_service import _caption_writers_note

    assert _caption_writers_note({}, 0) == ''
    assert _caption_writers_note({'joycaption': 0}, 0) == ''
    # An engine this build does not know is printed AS ITSELF rather than dropped.
    note = _caption_writers_note({'some-future-engine': 4}, 0)
    assert note == ' (4 by some-future-engine)'


# --- the launch window's figures --------------------------------------------
def test_the_nsfw_share_divides_by_what_was_measured_not_by_the_pile(
        client, app, tmp_path):
    """An image ✨ Score never reached is UNKNOWN, not SFW.

    Six images: three scored and all three flagged, three never scored.  Over the
    pile the share reads 50%; over what was measured it reads 100%, and only the
    second is a statement about anything.
    """
    from app.extensions import db
    from app.models import BankImage
    from app.services.image_bank_service import thresholds

    bank_id, _ = _mkbank(client, tmp_path, {
        f'{i}.png': _flat(40 + i) for i in range(6)})
    with app.app_context():
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(),
                      key=lambda r: r.relpath)
        limit = thresholds()['nsfw_max']
        for r in rows:
            r.status = 'keep'
        for r in rows[:3]:
            r.nsfw_score = min(1.0, float(limit) + 0.2)
        db.session.commit()

    caption = client.get(f'/api/bank/{bank_id}').get_json()['pass_scopes']['caption']
    assert caption['nsfw']['keep'] == 3
    assert caption['nsfw_measured']['keep'] == 3
    # The pile is 6 — the figure the UI must NOT divide by.
    assert client.get(f'/api/bank/{bank_id}').get_json()['counts']['keep'] == 6


def test_the_nsfw_figures_use_the_same_rule_as_the_grids_own_flag(
        client, app, tmp_path):
    """One definition of 'NSFW', not two.

    The window's number and the tile's amber chip must never disagree, so the
    payload figure is asserted against the per-image flag the grid renders rather
    than against a constant.
    """
    from app.extensions import db
    from app.models import BankImage
    from app.services.image_bank_service import thresholds

    bank_id, _ = _mkbank(client, tmp_path, {f'{i}.png': _flat(50 + i) for i in range(5)})
    with app.app_context():
        rows = sorted(BankImage.query.filter_by(bank_id=bank_id).all(),
                      key=lambda r: r.relpath)
        limit = thresholds()['nsfw_max']
        for r in rows:
            r.status = 'keep'
        rows[0].nsfw_score = min(1.0, limit + 0.3)
        rows[1].nsfw_score = min(1.0, limit + 0.1)
        rows[2].nsfw_score = max(0.0, limit - 0.1)     # measured, and under the bar
        db.session.commit()

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    flagged_tiles = sum(1 for im in client.get(f'/api/bank/{bank_id}/images')
                        .get_json()['images'] if 'nsfw' in (im.get('flags') or []))
    assert payload['pass_scopes']['caption']['nsfw']['keep'] == flagged_tiles == 2
    assert payload['pass_scopes']['caption']['nsfw_measured']['keep'] == 3
