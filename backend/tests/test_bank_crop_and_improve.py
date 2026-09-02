"""✂ Bank crop and ✨ Bank upscale & improve — the edits made IN the Bank.

Asked for by nofaceman on Discord (backed by mr.arrow): the Bank holds the
filtering and the curation, but reframing or upscaling a shot meant leaving it
for a Dataset and coming back through an export into a NEW bank.

The invariant is the SAME one the watermark cleaning is built on, and it is the
first thing asserted here: a bank is a read-only view over a folder that belongs
to the user, so an edit may never touch the source file. What is new, and what
the rest of this file pins, is the chain that makes an edit VISIBLE and HONEST:

  · the ONE resolver prefers the edited blob, ABOVE the watermark clean — an edit
    is made from what the user was looking at, so re-applying the clean or the
    turn to it would apply them twice;
  · `analysis_image_path` fails CLOSED on it, so no pass can measure the old
    pixels and file the result under the new image;
  · every measured lane is invalidated, which is the whole point of the request
    ("crop, then re-analyse, then curate, in one place");
  · the edited blob and the thumbnail carry the GENERATION in their name, so a
    re-crop cannot be served from a one-hour HTTP cache;
  · ↩ Revert deletes only what we made, and hands back the turn the edit baked in.

The ✨ pass needs ComfyUI and a GPU, so what is tested here is everything that
happens BEFORE the render: the pool, the refusals, and the fact that each of
them is answered once per pass rather than once per image.
"""
import os

import pytest
from PIL import Image


def _photo(w=1000, h=600, value=90):
    """A plain, readable image — these tests care about geometry, not content."""
    im = Image.new('RGB', (w, h), (value, value, value))
    for y in range(0, h, 40):           # structure, so a crop is visible
        for x in range(w):
            im.putpixel((x, y), (250, 250, 250))
    return im


def _mkbank(client, tmp_path, files, name='EDIT'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        im.save(str(p), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _ids(app, bank_id):
    from app.models import BankImage
    with app.app_context():
        return [r.id for r in BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all()]


def _row(app, image_id):
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        return db.session.get(BankImage, image_id)


def _fingerprint(path):
    """Bytes + mtime — what "the source was not touched" actually means."""
    return open(path, 'rb').read(), os.stat(path).st_mtime_ns


def _seed_measurements(app, image_id):
    """Give a row one representative of every measured lane, so the test can say
    whether an edit really did invalidate them."""
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        row = db.session.get(BankImage, image_id)
        row.quality_state = 'ok'
        row.blur_score = 123.0
        row.aesthetic_score = 7.5
        row.nsfw_score = 0.1
        row.framing = 'bust'
        row.medium = 'photo'
        row.face_state = 'scorable'
        row.face_det = 0.9
        # A person the USER declared is identity metadata, not a measurement:
        # it must survive every pixel-generation invalidation.
        row.face_cluster = 12
        row.face_cluster_origin = 'asserted'
        db.session.commit()


# --- ✂ Crop -----------------------------------------------------------------

def test_crop_never_writes_to_the_users_folder(client, app, tmp_path):
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]
    before = _fingerprint(str(src / 'a.jpg'))

    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 100, 'y': 50, 'w': 400, 'h': 300})
    assert r.status_code == 200, r.get_json()

    # The one promise the whole feature rests on.
    assert _fingerprint(str(src / 'a.jpg')) == before


def test_crop_publishes_the_box_and_the_resolver_serves_it(client, app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]

    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 100, 'y': 50, 'w': 400, 'h': 300})
    body = r.get_json()
    assert r.status_code == 200 and body['ok']
    assert body['edit_method'] == 'crop'
    assert body['edit_generation'] == 1
    # The row's displayed geometry is bound to the blob we just wrote — a grid
    # that lays tiles out from a NULL width collapses them.
    assert (body['width'], body['height']) == (400, 300)

    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = db.session.get(BankImage, image_id)
        blob = banks.edited_image_path(bank_id, image_id, row.edit_generation)
        assert blob.is_file()
        # THE resolver — the one every reader goes through (grid, /file, promote).
        assert banks.resolved_image_path(bank, row) == str(blob)
        with Image.open(str(blob)) as im:
            assert im.size == (400, 300)


def test_crop_encodes_once_and_only_the_box(client, app, tmp_path, monkeypatch):
    """nofaceman, back once the feature was out: "sometimes takes 3-5 seconds
    or more to do the cropping". Every crop paid TWO lossless WebP encodes, one
    of them of the WHOLE image — staged only to be reopened, cut and thrown
    away — at about a second per megapixel each. Pin the shape that makes it
    fast: one encode, of the box alone, and no full-frame blob left behind."""
    from app.services import image_bank_service as banks
    from app.services import image_encoding

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]

    encoded = []
    real_save_edit = image_encoding.save_edit

    def counting_save_edit(im, fp, fmt, policy, **kw):
        encoded.append((im.size, fmt, policy))
        return real_save_edit(im, fp, fmt, policy, **kw)
    monkeypatch.setattr(image_encoding, 'save_edit', counting_save_edit)

    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 100, 'y': 50, 'w': 400, 'h': 300})
    assert r.status_code == 200, r.get_json()

    assert encoded == [((400, 300), 'WEBP', image_encoding.LOSSLESS)], (
        'the crop must encode the box once — a full-frame (1000, 600) entry here '
        'is the staging copy coming back')
    with app.app_context():
        edited = banks.edited_image_path(bank_id, image_id, 1).parent
    assert sorted(p.name for p in edited.iterdir()) == [f'{image_id}.e1.webp']


def test_crop_cuts_the_displayed_pixels_exactly_and_ships_no_exif(client, app, tmp_path, monkeypatch):
    """The box is drawn on the image as the browser shows it — EXIF turn
    applied — so the cut must follow those pixels byte for byte (the blob is
    lossless), and must not carry the orientation tag that would make the
    browser turn the result a second time."""
    from PIL import ImageOps
    from app.services import image_bank_service as banks
    from app.services import image_encoding

    src = tmp_path / 'src'
    src.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6                       # shown turned: a 1000×600 file displays as 600×1000
    _photo().save(str(src / 'a.jpg'), 'JPEG', quality=92, exif=exif.tobytes())
    r = client.post('/api/bank/create', json={'name': 'TURNED', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    bank_id = r.get_json()['id']
    image_id = _ids(app, bank_id)[0]

    # The WebP writer never copies `info` on its own, so the blob cannot prove
    # the wash: look at the image HANDED to the encoder instead.
    handed = []
    real_save_edit = image_encoding.save_edit

    def capturing_save_edit(im, fp, fmt, policy, **kw):
        handed.append(im)
        return real_save_edit(im, fp, fmt, policy, **kw)
    monkeypatch.setattr(image_encoding, 'save_edit', capturing_save_edit)

    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 50, 'y': 120, 'w': 300, 'h': 500})
    assert r.status_code == 200, r.get_json()
    assert (r.get_json()['width'], r.get_json()['height']) == (300, 500)
    assert len(handed) == 1
    assert not handed[0].info.get('exif') and not handed[0].info.get('icc_profile'), (
        'the source metadata rode along into the cut — the browser would turn it twice')

    with Image.open(str(src / 'a.jpg')) as s:
        expected = ImageOps.exif_transpose(s).crop((50, 120, 350, 620)).convert('RGB')
    with app.app_context():
        blob = banks.edited_image_path(bank_id, image_id, 1)
    with Image.open(str(blob)) as b:
        b.load()
        assert b.size == (300, 500)
        assert not b.info.get('exif')
        assert b.convert('RGB').tobytes() == expected.tobytes()


def test_crop_keeps_the_alpha_of_a_transparent_source(client, app, tmp_path):
    """A PNG with holes stays a PNG with holes: the cut is RGBA when the
    source carries alpha, and the alpha channel is the source's own — neither
    flattened to RGB nor composited over black."""
    from app.services import image_bank_service as banks

    src = tmp_path / 'src'
    src.mkdir()
    im = _photo(400, 300).convert('RGBA')
    hole = Image.new('L', (400, 300), 255)
    for y in range(80, 160):
        for x in range(120, 260):
            hole.putpixel((x, y), 0)        # a transparent window inside the box
    im.putalpha(hole)
    im.save(str(src / 'holes.png'), 'PNG')
    r = client.post('/api/bank/create', json={'name': 'ALPHA', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    bank_id = r.get_json()['id']
    image_id = _ids(app, bank_id)[0]

    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 100, 'y': 50, 'w': 200, 'h': 150})
    assert r.status_code == 200, r.get_json()
    with app.app_context():
        blob = banks.edited_image_path(bank_id, image_id, 1)
    with Image.open(str(blob)) as b:
        b.load()
        assert b.mode == 'RGBA'
        expected = im.crop((100, 50, 300, 200))
        assert b.getchannel('A').tobytes() == expected.getchannel('A').tobytes()
        # Colour is compared where it can be seen: lossless WebP cleans the RGB
        # under fully transparent pixels (libwebp's default, `exact` off), and
        # that was already true of the blob before the cut went single-pass.
        seen = expected.getchannel('A').point(lambda a: 255 if a else 0)
        black = Image.new('RGB', expected.size)
        assert (Image.composite(b.convert('RGB'), black, seen).tobytes()
                == Image.composite(expected.convert('RGB'), black, seen).tobytes())


def test_crop_clamps_an_overflowing_box_and_says_so(client, app, tmp_path):
    """A box drawn past the edge is cut down to what exists, and the reply
    announces the CLAMPED size — that is what the grid lays the tile out from."""
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})     # 1000×600
    image_id = _ids(app, bank_id)[0]
    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 900, 'y': 500, 'w': 400, 'h': 300})
    assert r.status_code == 200, r.get_json()
    assert (r.get_json()['width'], r.get_json()['height']) == (100, 100)
    with app.app_context():
        blob = banks.edited_image_path(bank_id, image_id, 1)
    with Image.open(str(blob)) as b:
        assert b.size == (100, 100)


def test_a_failed_crop_leaves_the_previous_generation_in_place(client, app, tmp_path, monkeypatch):
    """The blob is published atomically: when the encode of a second crop dies
    half-way, the row still points at the first generation, the resolver still
    serves it, and no half-written file is left in edited/ for a reader to trip
    on."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks
    from app.services import image_encoding

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]
    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 100, 'y': 50, 'w': 400, 'h': 300})
    assert r.status_code == 200, r.get_json()

    def disk_full(im, fp, fmt, policy, **kw):
        raise OSError('disk full')
    monkeypatch.setattr(image_encoding, 'save_edit', disk_full)
    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 10, 'y': 10, 'w': 100, 'h': 100})
    assert r.status_code == 400
    assert 'could not be prepared for cropping' in r.get_json()['error']

    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = db.session.get(BankImage, image_id)
        assert row.edit_generation == 1
        first = banks.edited_image_path(bank_id, image_id, 1)
        assert banks.resolved_image_path(bank, row) == str(first)
        assert first.is_file()
        assert sorted(p.name for p in first.parent.iterdir()) == [f'{image_id}.e1.webp']


def test_crop_clears_the_measurements_so_the_next_pass_re_reads_it(client, app, tmp_path):
    """The half of the request that is easy to forget: nofaceman asked to crop
    AND re-analyse in the same place. A score read off the pre-crop pixels would
    describe an image this bank no longer holds."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]
    _seed_measurements(app, image_id)

    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 10, 'y': 10, 'w': 200, 'h': 200})
    assert r.status_code == 200, r.get_json()

    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = db.session.get(BankImage, image_id)
        for field in ('quality_state', 'blur_score', 'aesthetic_score',
                      'nsfw_score', 'framing', 'medium', 'face_state', 'face_det'):
            assert getattr(row, field) is None, field
        # …but not what the USER declared.
        assert row.face_cluster == 12
        assert row.face_cluster_origin == 'asserted'
        # And the passes must now read the CROP, not the source: fail-closed, so
        # a missing blob yields None rather than the original.
        assert banks.analysis_image_path(bank, row) == str(
            banks.edited_image_path(bank_id, image_id, row.edit_generation))


def test_a_second_crop_moves_the_blob_and_the_thumbnail_name(client, app, tmp_path):
    """A re-crop produces different pixels under an unchanged `edit_method`. If
    the generation were not in both names, the one-hour HTTP cache on /thumb
    would serve the FIRST crop — a re-crop that reads as "nothing happened"."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]

    # BOTH boxes are off-origin, and the first one's y is not a multiple of the
    # 40-row stripe: the region (60, 40, 260, 190) of the first crop and the
    # same region of the untouched source hold different pixels, so only a cut
    # made in the first generation can satisfy the comparison below. (With the
    # first box at the origin, "cut the raw source instead" passed unnoticed.)
    client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                json={'x': 100, 'y': 50, 'w': 500, 'h': 400})
    with app.app_context():
        row = db.session.get(BankImage, image_id)
        first_blob = banks.edited_image_path(bank_id, image_id, row.edit_generation)
        first_thumb = banks._thumb_path(bank_id, row)
    with Image.open(str(first_blob)) as im:
        first_pixels = im.convert('RGB')
        first_pixels.load()

    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 60, 'y': 40, 'w': 200, 'h': 150})
    assert r.get_json()['edit_generation'] == 2

    with app.app_context():
        row = db.session.get(BankImage, image_id)
        second_blob = banks.edited_image_path(bank_id, image_id, row.edit_generation)
        second_thumb = banks._thumb_path(bank_id, row)
        assert second_blob != first_blob
        assert second_thumb != first_thumb
        assert second_blob.is_file()
        # The superseded generation is pruned once the row points at the new one.
        assert not first_blob.exists()
        with Image.open(str(second_blob)) as im:
            # Cut from the FIRST crop, which is what the user was looking at.
            assert im.size == (200, 150)
            assert im.convert('RGB').tobytes() == first_pixels.crop((60, 40, 260, 190)).tobytes()


def test_a_crop_bakes_the_turn_in_and_revert_hands_it_back(client, app, tmp_path):
    """A box is drawn on the TURNED image, so the turn has to be applied before
    the box means anything — after which re-applying it would turn the pixels
    twice. The turn is not discarded, though: ↩ Revert gives it back."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo(1000, 600)})
    image_id = _ids(app, bank_id)[0]

    r = client.post(f'/api/bank/{bank_id}/rotate',
                    json={'ids': [image_id], 'degrees': 90})
    assert r.status_code == 200, r.get_json()

    # The turned image is 600×1000, so this box only exists after the turn.
    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                    json={'x': 0, 'y': 0, 'w': 600, 'h': 900})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['rotation'] == 0
    # The blob's size is the witness that the TURNED pixels were cut: on the
    # untouched 1000×600 source the same box would clamp to 600×600.
    assert (r.get_json()['width'], r.get_json()['height']) == (600, 900)
    with app.app_context():
        with Image.open(str(banks.edited_image_path(bank_id, image_id, 1))) as im:
            assert im.size == (600, 900)

    with app.app_context():
        row = db.session.get(BankImage, image_id)
        assert row.rotation is None            # already in the pixels
        assert row.edit_baked_rotation == 90   # …and remembered

    r = client.post(f'/api/bank/{bank_id}/edits/revert',
                    json={'image_ids': [image_id]})
    assert r.status_code == 200 and r.get_json()['reverted'] == 1
    with app.app_context():
        row = db.session.get(BankImage, image_id)
        assert row.edit_method is None
        assert row.rotation == 90              # given back, not silently dropped
        assert not banks.edited_image_path(bank_id, image_id, 1).exists()


def test_revert_puts_every_reader_back_on_the_source(client, app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as banks

    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]
    client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                json={'x': 0, 'y': 0, 'w': 300, 'h': 300})

    r = client.post(f'/api/bank/{bank_id}/edits/revert', json={})   # empty = all
    assert r.status_code == 200 and r.get_json()['reverted'] == 1

    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = db.session.get(BankImage, image_id)
        assert row.edit_method is None and row.edit_generation is None
        assert banks.resolved_image_path(bank, row) == str(src / 'a.jpg')


def test_undoing_the_watermark_clean_takes_the_edit_made_on_top_of_it(client, app, tmp_path):
    """An edit made on a cleaned image is made OF those pixels. Keeping it would
    leave an image the user is told is no longer cleaned, while the mark it is
    being restored to is still gone."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]

    # Stand where the cleaning pass leaves a row, without running a detector.
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        row = db.session.get(BankImage, image_id)
        raw = banks.abs_image_path(bank, row)
        banks._stage_clean_copy(bank_id, row, raw)
        row.watermark_state = 'cleaned'
        row.watermark_clean_method = 'crop'
        # The undo compares this against the source before touching anything: a
        # row whose file changed on disk keeps its cleaned copy.
        row.watermark_fingerprint = transfer.content_fingerprint_path(raw)
        db.session.commit()

    client.post(f'/api/bank/{bank_id}/image/{image_id}/crop',
                json={'x': 0, 'y': 0, 'w': 300, 'h': 300})
    with app.app_context():
        assert banks.edited_image_path(bank_id, image_id, 1).is_file()

    r = client.post(f'/api/bank/{bank_id}/watermark/undo', json={})
    assert r.status_code == 200, r.get_json()
    with app.app_context():
        row = db.session.get(BankImage, image_id)
        assert row.edit_method is None
        assert not banks.edited_image_path(bank_id, image_id, 1).exists()


@pytest.mark.parametrize('box', [
    {'x': 0, 'y': 0, 'w': 0, 'h': 100},        # empty
    {'x': 0, 'y': 0, 'w': 'wide', 'h': 100},   # not a number
    {'x': 5000, 'y': 0, 'w': 100, 'h': 100},   # a real box, entirely outside the 1000×600 image
])
def test_an_unusable_box_is_refused_and_changes_nothing(client, app, tmp_path, box):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]
    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop', json=box)
    assert r.status_code == 400
    # The same answer whether the box was empty before or after clamping — not
    # the generic "could not be prepared", which would send the user after a
    # broken file.
    assert r.get_json()['error'] == 'invalid crop box'
    with app.app_context():
        assert db.session.get(BankImage, image_id).edit_method is None
        edited = banks._edited_dir(bank_id)
    # Nothing was written — not a blob, not a half-written .part- either.
    assert not edited.exists() or not any(edited.iterdir())


def test_crop_needs_its_four_numbers_and_a_real_image(client, app, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _ids(app, bank_id)[0]
    r = client.post(f'/api/bank/{bank_id}/image/{image_id}/crop', json={'x': 0, 'y': 0})
    assert r.status_code == 400
    r = client.post(f'/api/bank/{bank_id}/image/999999/crop',
                    json={'x': 0, 'y': 0, 'w': 10, 'h': 10})
    assert r.status_code == 404


# --- ✨ Upscale & improve — everything that happens BEFORE the render ---------

def test_the_payload_counts_the_two_edits_apart_and_offers_the_pass_scope(client, app, tmp_path):
    """"12 cropped" and "12 upscaled" answer different questions, and one total
    would let a run of GPU-minutes hide behind a hand crop."""
    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo(), 'b.jpg': _photo()})
    ids = _ids(app, bank_id)
    client.post(f'/api/bank/{bank_id}/image/{ids[0]}/crop',
                json={'x': 0, 'y': 0, 'w': 200, 'h': 200})

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    # Under `counts`, with every other per-bank tally — the panel reads them
    # there and nowhere else (frontend/src/components/bank/bankEdits.js).
    assert payload['counts']['cropped'] == 1
    assert payload['counts']['improved'] == 0
    # The launch window renders a permanent "counting…" when a pass has no entry.
    scope = payload['pass_scopes']['improve']
    assert set(scope) == {'todo', 'all'}
    # A cropped image is NOT an improved one: it is still in the pool.
    assert sum(scope['todo'].values()) == 2


def test_an_already_improved_image_leaves_the_pool(client, app, tmp_path):
    """There is no "do it again" tick box on purpose: ↩ Revert is how a second
    attempt is asked for, and it is one click. So the pool must exclude what the
    pass already rendered, or a re-run silently pays for it twice."""
    from app.extensions import db
    from app.models import BankImage

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo(), 'b.jpg': _photo()})
    ids = _ids(app, bank_id)
    with app.app_context():
        row = db.session.get(BankImage, ids[0])
        row.edit_method = 'improve'
        row.edit_generation = 1
        db.session.commit()

    payload = client.get(f'/api/bank/{bank_id}').get_json()
    assert payload['counts']['improved'] == 1
    assert sum(payload['pass_scopes']['improve']['todo'].values()) == 1


def test_improve_refuses_an_empty_pool_before_it_starts_anything(client, app, tmp_path,
                                                                 monkeypatch):
    """One refusal per PASS, not one per image — which is the whole reason the
    preflight is in the launcher and not in the job thread."""
    from app.extensions import db
    from app.models import BankImage

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    ids = _ids(app, bank_id)
    with app.app_context():
        row = db.session.get(BankImage, ids[0])
        row.edit_method = 'improve'
        row.edit_generation = 1
        db.session.commit()

    # The engine preflight must never be reached: there is nothing to render, so
    # the pool check has to come first (and a machine with no Klein weights must
    # not be told to install them before being told there is nothing to do).
    from app.services import face_dataset_service as fds

    def _boom(engine):
        raise AssertionError('the preflight ran on an empty pool')
    monkeypatch.setattr(fds, '_improve_preflight', _boom)

    r = client.post(f'/api/bank/{bank_id}/improve', json={})
    assert r.status_code == 400
    assert 'Revert' in r.get_json()['error']


def test_improve_refuses_a_busy_gpu_with_503_and_starts_no_job(client, app, tmp_path,
                                                               monkeypatch):
    from app.services import face_dataset_service as fds
    from app.services import image_bank_service as banks

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    monkeypatch.setattr(fds, '_improve_preflight', lambda engine: None)
    monkeypatch.setattr(banks, '_gpu_busy_reason',
                        lambda: 'a training run is using the GPU')

    r = client.post(f'/api/bank/{bank_id}/improve', json={})
    assert r.status_code == 503
    assert 'training' in r.get_json()['error']
    # Nothing was queued: the bank is free for the next pass.
    assert client.get(f'/api/bank/{bank_id}').get_json().get('activity') in (None, {})


def test_a_missing_engine_keeps_its_itemized_409(client, app, tmp_path, monkeypatch):
    """The structured body is what lists the missing weights and starts their
    download. Flattening it into a plain error would cost the user the one thing
    that makes the refusal actionable."""
    from app.services import face_dataset_service as fds
    from app.services.klein_edit_helper import KleinModelsMissing

    bank_id, _ = _mkbank(client, tmp_path, {'a.jpg': _photo()})

    def _missing(engine):
        raise KleinModelsMissing(['klein_model'])
    monkeypatch.setattr(fds, '_improve_preflight', _missing)

    r = client.post(f'/api/bank/{bank_id}/improve', json={'engine': 'klein'})
    assert r.status_code == 409
    body = r.get_json()
    # Not the bare "another pass is running" shape — this 409 carries assets.
    assert 'busy_kind' not in body
    assert body.get('missing') or body.get('error')
