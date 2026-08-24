"""WHO wrote a caption, and what 🔄 Re-caption is therefore allowed to destroy.

These tests measure THE PROPERTY, not the arguments: they build one bank holding
machine-written captions, hand-written ones, captions with no recorded origin and
rejected rows, run the forced pass once, and then check ROW BY ROW which text
changed and which did not.  A test that only asserted "force=True was forwarded"
would pass just as happily on a pass that overwrites everything.
"""
import os
from app.extensions import db
import random
import sqlite3

from PIL import Image


# --- factories ---------------------------------------------------------------
def _flat(value=128, size=64):
    """A picture whose TOP-LEFT pixel is ``value`` (the mock captioner's key) and
    whose body is a value-dependent pattern — flat squares all share one perceptual
    hash, so the promotion paths would dedupe them down to a single row and the
    transfer assertions would pass on an empty result."""
    rnd = random.Random(value)
    im = Image.new('RGB', (size, size))
    px = im.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
    px[0, 0] = (value, value, value)
    return im


def _save(path, im):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)


def _mkbank(client, tmp_path, files, name='B'):
    src = tmp_path / name
    for rel, im in files.items():
        _save(str(src / rel), im)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _use_ollama_backend(app):
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': 'ollama'}})


def _mock_vision(monkeypatch, caption_by_pixel):
    """The Ollama seam caption_paths uses, keyed by the image's top-left pixel."""
    import io

    from app.services import vision_ollama

    def fake_describe(image_bytes, *a, **k):
        v = Image.open(io.BytesIO(image_bytes)).convert('L').getpixel((0, 0))
        return caption_by_pixel.get(v, '')

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


def _rows(app, bank_id):
    from app.models import BankImage
    with app.app_context():
        return {r.relpath: (r.caption, r.caption_origin, r.status)
                for r in BankImage.query.filter_by(bank_id=bank_id).all()}


def _seed_mixed_bank(client, app, tmp_path):
    """One bank, four rows, four different provenances — the whole point.

    Pixel values double as the mock captioner's key, so each row can be told
    apart by the text it would receive if it were rewritten.
    """
    from app.extensions import db
    from app.models import BankImage
    from app.services import caption_origin

    bank_id, _ = _mkbank(client, tmp_path, {
        'machine.png': _flat(10), 'mine.png': _flat(20),
        'unknown.png': _flat(30), 'binned.png': _flat(40),
        'blank.png': _flat(50)})
    with app.app_context():
        by = {r.relpath: r
              for r in BankImage.query.filter_by(bank_id=bank_id).all()}
        by['machine.png'].caption = 'written by a model'
        by['machine.png'].caption_origin = caption_origin.OLLAMA
        by['machine.png'].status = 'keep'
        by['mine.png'].caption = 'written by me'
        by['mine.png'].caption_origin = caption_origin.ASSERTED
        by['mine.png'].status = 'keep'
        # No origin at all: every row that predates the column looks like this.
        by['unknown.png'].caption = 'nobody knows who wrote this'
        by['unknown.png'].caption_origin = None
        by['unknown.png'].status = 'keep'
        # The bin is out of every pass's reach, whatever its caption says.
        by['binned.png'].caption = 'in the bin'
        by['binned.png'].caption_origin = caption_origin.ASSERTED
        by['binned.png'].status = 'reject'
        by['blank.png'].caption = None
        by['blank.png'].status = 'keep'
        db.session.commit()
    return bank_id


# --- the protection ----------------------------------------------------------
def test_forced_recaption_rewrites_everything_except_what_the_user_wrote(
        client, app, tmp_path, monkeypatch):
    _use_ollama_backend(app)
    bank_id = _seed_mixed_bank(client, app, tmp_path)
    _mock_vision(monkeypatch, {10: 'fresh 10', 20: 'fresh 20', 30: 'fresh 30',
                               40: 'fresh 40', 50: 'fresh 50'})

    r = client.post(f'/api/bank/{bank_id}/caption', json={'force': True})
    assert r.status_code == 202
    assert client.get(f'/api/bank/{bank_id}').get_json()['activity']['error'] is None

    after = _rows(app, bank_id)
    # Rewritten: a model wrote it, so nothing of the user's is lost.
    assert after['machine.png'][0] == 'fresh 10'
    assert after['machine.png'][1] == 'ollama'
    # SPARED — the whole feature. Text AND stamp untouched.
    assert after['mine.png'] == ('written by me', 'asserted', 'keep')
    # Rewritten: the decision on record is that an unrecorded origin is
    # re-captionable, because it can never be recovered and sparing it would
    # freeze this button on every bank that already exists.
    assert after['unknown.png'][0] == 'fresh 30'
    assert after['unknown.png'][1] == 'ollama'
    # The bin is untouched even though it is 'asserted' — a different rule,
    # older than this one, and this change must not have leaned on it.
    assert after['binned.png'] == ('in the bin', 'asserted', 'reject')
    # A blank row is captioned by a forced pass exactly as before.
    assert after['blank.png'][0] == 'fresh 50'


def test_the_escape_hatch_rewrites_the_users_own_captions_and_only_on_request(
        client, app, tmp_path, monkeypatch):
    _use_ollama_backend(app)
    bank_id = _seed_mixed_bank(client, app, tmp_path)
    _mock_vision(monkeypatch, {10: 'fresh 10', 20: 'fresh 20', 30: 'fresh 30',
                               40: 'fresh 40', 50: 'fresh 50'})

    r = client.post(f'/api/bank/{bank_id}/caption',
                    json={'force': True, 'include_asserted': True})
    assert r.status_code == 202
    after = _rows(app, bank_id)
    assert after['mine.png'][0] == 'fresh 20'
    assert after['mine.png'][1] == 'ollama'
    # Still not the bin.
    assert after['binned.png'][0] == 'in the bin'


def test_the_unforced_pass_is_byte_identical_to_what_it_always_was(
        client, app, tmp_path, monkeypatch):
    """No force, no option: only the rows with no caption at all are touched."""
    _use_ollama_backend(app)
    bank_id = _seed_mixed_bank(client, app, tmp_path)
    _mock_vision(monkeypatch, {10: 'fresh 10', 20: 'fresh 20', 30: 'fresh 30',
                               40: 'fresh 40', 50: 'fresh 50'})

    before = _rows(app, bank_id)
    assert client.post(f'/api/bank/{bank_id}/caption', json={}).status_code == 202
    after = _rows(app, bank_id)
    for name in ('machine.png', 'mine.png', 'unknown.png', 'binned.png'):
        assert after[name] == before[name], name
    assert after['blank.png'][0] == 'fresh 50'
    # …and the pass records WHICH engine wrote it, not which backend was asked for.
    assert after['blank.png'][1] == 'ollama'


def test_the_number_on_the_button_is_the_number_of_rows_that_change(
        client, app, tmp_path, monkeypatch):
    """The defect this whole row exists to avoid: announce N, act on M.

    The counts payload is what the button quotes; the pass is what happens.  They
    are asserted against each other here rather than each against a constant.
    """
    _use_ollama_backend(app)
    bank_id = _seed_mixed_bank(client, app, tmp_path)
    _mock_vision(monkeypatch, {10: 'fresh 10', 20: 'fresh 20', 30: 'fresh 30',
                               40: 'fresh 40', 50: 'fresh 50'})
    counts = client.get(f'/api/bank/{bank_id}').get_json()['counts']
    # The three figures the screen must never merge, per pile.
    assert counts['caption_asserted_keep'] == 1        # mine.png
    assert counts['caption_unrecorded_keep'] == 1      # unknown.png
    assert counts['caption_todo_keep'] == 1            # blank.png
    assert counts['keep'] == 4
    # What the button will say: pile minus what it spares.
    announced = counts['keep'] + counts['pending'] - (
        counts['caption_asserted_keep'] + counts['caption_asserted_pending'])

    before = _rows(app, bank_id)
    client.post(f'/api/bank/{bank_id}/caption', json={'force': True})
    after = _rows(app, bank_id)
    changed = sum(1 for k in after if after[k][0] != before[k][0])
    assert changed == announced == 3


def test_a_caption_typed_while_the_run_was_being_priced_still_wins(
        client, app, tmp_path, monkeypatch):
    """The launch filter prices the run; the job re-reads the rows.

    Both must apply the rule, or a caption written in the seconds between the two
    is destroyed by a run that had already decided it was fair game.  Exercised
    directly on the job, which is where the second filter lives.
    """
    from app.extensions import db
    from app.models import BankImage
    from app.services import caption_origin
    from app.services import image_bank_service as banks

    _use_ollama_backend(app)
    bank_id = _seed_mixed_bank(client, app, tmp_path)
    _mock_vision(monkeypatch, {10: 'fresh 10', 20: 'fresh 20', 30: 'fresh 30',
                               40: 'fresh 40', 50: 'fresh 50'})
    with app.app_context():
        # Priced while 'unknown.png' was anonymous; asserted a moment later.
        row = BankImage.query.filter_by(bank_id=bank_id,
                                        relpath='unknown.png').one()
        row.caption_origin = caption_origin.ASSERTED
        db.session.commit()
        job = banks._caption_job(bank_id, None, True, keep_asserted=True)
        job({'id': 'test', 'cancelled': False})
    after = _rows(app, bank_id)
    assert after['unknown.png'][0] == 'nobody knows who wrote this'


# --- the three copy paths ----------------------------------------------------
def _dataset_with_one_captioned_image(app, tmp_path, caption, origin):
    from app.extensions import db
    from app.services import face_dataset_service as datasets
    from app.services.dataset_storage import dataset_path

    ds = datasets.create_dataset('local', 'origin ds', 'trigger')
    src = tmp_path / 'ds-src.png'
    _save(str(src), _flat(200))
    with open(src, 'rb') as fh:
        ids, _ = datasets.import_images('local', ds.id, [fh.read()])
    from app.models import FaceDatasetImage
    row = db.session.get(FaceDatasetImage, ids[0])
    row.caption = caption
    row.caption_origin = origin
    row.status = 'keep'
    db.session.commit()
    assert os.path.isfile(os.path.join(str(dataset_path(ds.id)), row.filename))
    return ds, row


def test_dataset_to_bank_import_carries_who_wrote_the_caption(app, tmp_path):
    """THE bug this whole change exists for.

    A caption typed in the Dataset editor used to arrive in the Bank as an
    anonymous string, and the Bank's forced pass then had no way left to spare it.
    """
    from app.models import BankImage
    from app.services import caption_origin
    from app.services import image_bank_service as banks

    with app.app_context():
        ds, _ = _dataset_with_one_captioned_image(
            app, tmp_path, 'I typed this', caption_origin.ASSERTED)
        bank_id = banks.start_dataset_import(app, 'local', ds.id, 'from dataset')
        rows = BankImage.query.filter_by(bank_id=bank_id).all()
        assert len(rows) == 1
        assert rows[0].caption == 'I typed this'
        assert rows[0].caption_origin == 'asserted'


def test_bank_to_bank_copy_carries_it_too(client, app, tmp_path):
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id = _seed_mixed_bank(client, app, tmp_path)
    with app.app_context():
        ids = [r.id for r in BankImage.query.filter_by(bank_id=bank_id)
               .filter(BankImage.status == 'keep').all()]
        new_id = banks.start_bank_promote(app, 'local', bank_id, ids, 'copy')
        copied = {r.relpath: (r.caption, r.caption_origin)
                  for r in BankImage.query.filter_by(bank_id=new_id).all()}
    assert copied['mine.png'] == ('written by me', 'asserted')
    assert copied['machine.png'] == ('written by a model', 'ollama')
    assert copied['unknown.png'] == ('nobody knows who wrote this', None)


def test_promote_to_dataset_carries_it_back(client, app, tmp_path):
    """The return leg. Without it a Dataset -> Bank -> Dataset round-trip
    launders a hand-written caption into a rewritable one."""
    from app.models import BankImage, FaceDatasetImage
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    bank_id = _seed_mixed_bank(client, app, tmp_path)
    with app.app_context():
        ds = datasets.create_dataset('local', 'target', 'trg')
        ids = [r.id for r in BankImage.query.filter_by(bank_id=bank_id)
               .filter(BankImage.status == 'keep').all()]
        banks.start_promote(app, 'local', bank_id, ids, ds.id)
        got = sorted((r.caption, r.caption_origin)
                     for r in FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
                     if r.caption)
    assert ('written by me', 'asserted') in got
    assert ('written by a model', 'ollama') in got
    assert ('nobody knows who wrote this', None) in got


def test_an_unknown_origin_on_the_wire_is_stored_as_nothing(app, tmp_path):
    """import_images validates the stamp instead of trusting it.

    An unrecognised token would sit in the column forever, never matching
    'asserted' — a protection that silently is not one.
    """
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as datasets

    with app.app_context():
        ds = datasets.create_dataset('local', 'wire', 'trg')
        src = tmp_path / 'wire.png'
        _save(str(src), _flat(210))
        with open(src, 'rb') as fh:
            ids, _ = datasets.import_images(
                'local', ds.id, [fh.read()], captions=['text'],
                caption_origins=['definitely-not-a-real-value'])
        row = db.session.get(FaceDatasetImage, ids[0])
        assert row.caption == 'text'
        assert row.caption_origin is None


# --- the dataset-side writers ------------------------------------------------
def test_the_caption_editor_claims_authorship_and_clearing_it_gives_it_back(
        app, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as datasets

    with app.app_context():
        ds, row = _dataset_with_one_captioned_image(app, tmp_path, None, None)
        datasets.set_image_caption('local', row.id, 'my words', short='my short')
        row = db.session.get(FaceDatasetImage, row.id)
        assert (row.caption_origin, row.caption_short_origin) == ('asserted', 'asserted')
        # Emptying the box empties the stamp: an 'asserted' marker on a blank
        # caption would be a row every future pass skips, forever.
        datasets.set_image_caption('local', row.id, '', short='')
        row = db.session.get(FaceDatasetImage, row.id)
        assert (row.caption, row.caption_origin) == (None, None)
        assert (row.caption_short, row.caption_short_origin) == (None, None)


def test_a_backup_round_trip_does_not_lose_the_protection(app, tmp_path):
    """The silent, expensive one: the column names live only in _BACKUP_IMG_FIELDS.

    Dropping them there would not lose a caption — it would lose the protection on
    every hand-written caption, at the first restore, with nothing on screen.
    """
    from app.services import face_dataset_service as datasets

    assert 'caption_origin' in datasets._BACKUP_IMG_FIELDS
    assert 'caption_short_origin' in datasets._BACKUP_IMG_FIELDS


# --- the migration -----------------------------------------------------------
def test_the_migration_lands_on_an_existing_database_and_old_rows_stay_null(
        tmp_path, monkeypatch):
    """A database created BEFORE the column, opened by this build.

    The rows it already holds must come back NULL — not 'asserted' (which would
    freeze Re-caption on every bank in the wild) and not an engine name (which
    would be an attribution nobody measured, on the side that destroys work).
    """
    db_path = tmp_path / 'legacy.sqlite3'
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE bank_image (
            id INTEGER PRIMARY KEY, bank_id INTEGER NOT NULL,
            relpath TEXT NOT NULL, file_size INTEGER, status VARCHAR(10),
            caption TEXT);
        INSERT INTO bank_image (bank_id, relpath, status, caption)
            VALUES (1, 'old.png', 'keep', 'a caption from before all this');
        """)
    con.commit()
    con.close()

    from app import create_app
    monkeypatch.setenv('LDS_TESTING', '1')
    application = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
    })
    with application.app_context():
        from app.extensions import db as _db
        cols = {row[1] for row in _db.session.execute(
            __import__('sqlalchemy').text('PRAGMA table_info(bank_image)'))}
        assert 'caption_origin' in cols, 'the ALTER never ran'
        got = _db.session.execute(__import__('sqlalchemy').text(
            'SELECT caption, caption_origin FROM bank_image')).fetchone()
    assert got[0] == 'a caption from before all this'
    assert got[1] is None
    # …and the dataset side gained both of its columns the same way.
    con = sqlite3.connect(db_path)
    names = {r[1] for r in con.execute('PRAGMA table_info(face_dataset_image)')}
    con.close()
    assert {'caption_origin', 'caption_short_origin'} <= names


def test_the_vocabulary_is_frozen_and_borrowed_from_columns_that_exist(app):
    """Stored values never get renamed without an alias table (repo rule).

    They are also deliberately not new words: 'asserted' is
    BankImage.face_cluster_origin's, the engine names are watermark_source's idea.
    """
    from app.services import caption_origin

    assert caption_origin.ASSERTED == 'asserted'
    assert caption_origin.ENGINES == ('joycaption', 'ollama')
    # 'auto' is a CHAIN, not an engine: recording it would name a policy and
    # mislabel roughly half of an 'auto' run.
    assert caption_origin.engine_origin('auto') is None
    assert caption_origin.engine_origin(None) is None
    assert caption_origin.engine_origin('OLLAMA') == 'ollama'


def test_the_unprotected_clause_still_selects_rows_with_no_recorded_origin(app):
    """A NOT() over three-valued logic would drop every legacy row in silence.

    That is the bug the explicit clause exists to prevent, and it is invisible
    in every fixture where the column happens to be filled in.
    """
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import caption_origin

    with app.app_context():
        bank = ImageBank(user_id='local', name='clause', source_path='nowhere')
        db.session.add(bank)
        db.session.flush()
        db.session.add(BankImage(bank_id=bank.id, relpath='legacy.png', file_size=1,
                                 status='keep', caption='old text',
                                 caption_origin=None))
        db.session.add(BankImage(bank_id=bank.id, relpath='mine.png', file_size=1,
                                 status='keep', caption='my text',
                                 caption_origin='asserted'))
        db.session.commit()
        got = {r.relpath for r in BankImage.query.filter_by(bank_id=bank.id).filter(
            caption_origin.unprotected_clause(BankImage)).all()}
    assert got == {'legacy.png'}
