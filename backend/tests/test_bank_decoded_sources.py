"""🗃️ Image bank — HEIC/HEIF and AVIF sources, read through one decoded copy.

A bank indexes the user's folder IN PLACE and never writes to it, while its browser
view, its OpenCV face pass and its infer workers (possibly in a borrowed
interpreter) cannot read HEIC, and the trainers read no AVIF. So each such source
is decoded ONCE, in the app process, into a lossless PNG under the bank's own
working directory (`source_image_path`), and every reader takes that copy.

These tests pin the copy's lifetime (made on first use, named by the source's
SHA-256, replaced when the source changes, never served once the source is gone,
dropped with its row) and its readers (the resolvers, the poll's eligible set, the
embedding lookups, the face pass hand-off, the promotion to a dataset, /bank/file
and the thumbnail). Every test also proves the user's folder did not change.
"""
import hashlib
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from image_format_fixtures import STILL_AVIF, TINY_HEIC


def _folder_digest(folder):
    """{relative name: SHA-256} of every file in the user's folder."""
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}


def _jpeg(colour):
    out = io.BytesIO()
    Image.new('RGB', (64, 48), colour).save(out, 'JPEG', quality=90)
    return out.getvalue()


@pytest.fixture()
def bank_ctx(app, tmp_path):
    """A bank over a folder holding one HEIC, one AVIF and one JPEG, and the
    folder's digest taken before the app ever saw it."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    folder = tmp_path / 'phone'
    folder.mkdir()
    (folder / 'IMG_0001.HEIC').write_bytes(TINY_HEIC)
    (folder / 'web.avif').write_bytes(STILL_AVIF)
    (folder / 'plain.jpg').write_bytes(_jpeg((200, 40, 40)))
    before = _folder_digest(folder)
    with app.app_context():
        bank, added = banks.create_bank('local', 'Phone', str(folder))
        db.session.commit()
        assert added == 3
        rows = {row.relpath: row for row in BankImage.query.filter_by(bank_id=bank.id)}
        yield banks, bank, rows, folder, before


def test_without_a_heif_decoder_the_walk_leaves_heic_out(app, tmp_path, monkeypatch):
    """A platform pi-heif ships no wheel for: no row nothing could show."""
    from app.services import heif_decoder
    from app.services import image_bank_service as banks
    folder = tmp_path / 'phone'
    folder.mkdir()
    (folder / 'IMG_0001.HEIC').write_bytes(TINY_HEIC)
    (folder / 'web.avif').write_bytes(STILL_AVIF)
    (folder / 'plain.jpg').write_bytes(_jpeg((10, 120, 60)))
    monkeypatch.setattr(heif_decoder, 'register', lambda: False)
    with app.app_context():
        _bank, added = banks.create_bank('local', 'No HEIF decoder', str(folder))
    assert added == 2


def test_every_reader_takes_one_copy_made_on_first_use(bank_ctx):
    import numpy as np
    banks, bank, rows, folder, before = bank_ctx
    heic, avif, jpeg = rows['IMG_0001.HEIC'], rows['web.avif'], rows['plain.jpg']
    assert not banks._decoded_dir(bank.id).exists(), 'the inventory must decode nothing'

    copy = banks.resolved_image_path(bank, heic)
    assert copy == str(banks.decoded_image_path(
        bank.id, heic.id, hashlib.sha256(TINY_HEIC).hexdigest()))
    assert Path(copy).parent == banks._decoded_dir(bank.id)
    with Image.open(banks.abs_image_path(bank, heic)) as source, Image.open(copy) as png:
        assert png.format == 'PNG' and png.size == source.size == (96, 64)
        assert png.convert('RGB').tobytes() == source.convert('RGB').tobytes()
        assert png.info.get('icc_profile') == source.info.get('icc_profile')
    made = os.stat(copy).st_mtime_ns

    assert banks.analysis_image_path(bank, heic) == copy
    assert banks.source_image_path(bank, heic) == copy
    assert banks.analysis_image_path(bank, heic, materialize=False) == copy
    assert os.stat(copy).st_mtime_ns == made, 'a second reader re-made the copy'
    # A lookup never decodes: the AVIF has no copy yet, so it has no cache key yet.
    assert banks.analysis_image_path(bank, avif, materialize=False) is None
    assert banks.resolved_image_path(bank, avif).endswith('.png')
    assert banks.resolved_image_path(bank, jpeg) == banks.abs_image_path(bank, jpeg)

    _total, eligible = banks._semantic_eligible_paths(bank)
    assert copy in eligible and banks.abs_image_path(bank, jpeg) in eligible
    vector = np.ones(4, dtype=np.float32)
    ids, _E = banks._pool_embeddings(
        bank, {copy: vector, banks.abs_image_path(bank, jpeg): vector}, {})
    assert ids == sorted([heic.id, jpeg.id]), 'the score lookups lost the HEIC row'
    assert _folder_digest(folder) == before


def test_a_changed_source_gets_a_new_copy_and_a_vanished_one_is_never_served(bank_ctx):
    banks, bank, rows, folder, _before = bank_ctx
    heic = rows['IMG_0001.HEIC']
    first = banks.resolved_image_path(bank, heic)

    (folder / 'IMG_0001.HEIC').write_bytes(STILL_AVIF)       # the user replaced the photo
    second = banks.resolved_image_path(bank, heic)
    assert second == str(banks.decoded_image_path(
        bank.id, heic.id, hashlib.sha256(STILL_AVIF).hexdigest()))
    assert not os.path.exists(first), 'the previous generation outlived its source'

    (folder / 'IMG_0001.HEIC').unlink()                      # ...then deleted it
    assert banks.analysis_image_path(bank, heic) is None
    assert banks.source_image_path(bank, heic) == banks.abs_image_path(bank, heic)
    assert not os.path.isfile(banks.resolved_image_path(bank, heic))
    assert banks.forget_missing('local', bank.id)['removed'] == 1
    assert not os.path.exists(second), 'the copy outlived its row'


def test_an_undecodable_source_fails_closed_once_and_is_not_retried(bank_ctx, monkeypatch):
    """No pi-heif on this platform, or a corrupt file: analysis refuses the row,
    display falls back to the raw file, and the grid does not retry the decode
    (and log it) on every thumbnail request until the file changes."""
    banks, bank, rows, folder, _before = bank_ctx
    heic = rows['IMG_0001.HEIC']
    (folder / 'IMG_0001.HEIC').write_bytes(b'not a picture at all')
    attempts = []
    real = banks._materialize_decoded_copy
    monkeypatch.setattr(banks, '_materialize_decoded_copy',
                        lambda *args: attempts.append(args) or real(*args))
    for _ in range(3):
        assert banks.analysis_image_path(bank, heic) is None
        assert banks.resolved_image_path(bank, heic) == banks.abs_image_path(bank, heic)
    assert len(attempts) == 1
    (folder / 'IMG_0001.HEIC').write_bytes(TINY_HEIC)      # fixed: decoded again
    assert banks.analysis_image_path(bank, heic).endswith('.png') and len(attempts) == 2


def test_the_face_pass_hands_its_worker_the_decoded_copies(bank_ctx, monkeypatch):
    from app.extensions import db
    from app.models import BankImage
    from app.services import bank_transfer_metadata as transfer
    banks, bank, rows, folder, before = bank_ctx
    handed = []

    def fake_drive(job, python, script, payload, cache_path, progress_re, window, *,
                   include_cached=False):
        paths = json.loads(payload)['images']
        handed.extend(paths)
        return {'ok': True, 'clusters': {}, 'results': {
            p: {'state': 'ok', 'det': 0.9, 'fingerprint': transfer.content_fingerprint_path(p)}
            for p in paths}}, [], 0

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_drive)
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    monkeypatch.setattr(banks.bank_jobs, 'cancelled', lambda job: False)
    job = {'kind': 'faces', 'done': 0, 'total': 0, 'error': None, 'cancelled': False,
           'finished': False, 'detail': None, 'started_at': 0.0, '_touched': 0.0,
           '_cancel_hook': None, 'pipeline': None}
    banks._faces_job(bank.id)(job)

    assert job['error'] is None
    decoded = str(banks._decoded_dir(bank.id))
    copies = [p for p in handed if os.path.dirname(p) == decoded]
    assert len(handed) == 3 and len(copies) == 2 and all(p.endswith('.png') for p in copies)
    assert banks.abs_image_path(bank, rows['plain.jpg']) in handed
    assert all(db.session.get(BankImage, row.id).face_state for row in rows.values())
    assert _folder_digest(folder) == before


def test_promotion_to_a_dataset_takes_the_decoded_copy(bank_ctx):
    from app.extensions import db
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as datasets
    banks, bank, rows, folder, before = bank_ctx
    for row in rows.values():
        row.status = 'keep'
    db.session.commit()
    dataset = datasets.create_dataset('local', 'From phone', 'fromphone')
    progress = []
    with patch.object(banks.bank_jobs, 'cancelled', lambda _job: False), \
         patch.object(banks.bank_jobs, 'bump', lambda _job, _n=1: None), \
         patch.object(banks.bank_jobs, 'progress', lambda _job, **values: progress.append(values)):
        banks._promote_job('local', bank.id, [row.id for row in rows.values()], dataset.id)(object())

    assert progress[-1]['detail'].startswith('done — 3 imported'), progress[-1]
    promoted = {row.bank_image_id: row for row in
                FaceDatasetImage.query.filter_by(dataset_id=dataset.id)}
    assert sorted(os.path.splitext(row.filename)[1] for row in promoted.values()) == [
        '.jpg', '.png', '.png']
    for relpath in ('IMG_0001.HEIC', 'web.avif'):
        stored = Path(datasets._dataset_dir(dataset.id), promoted[rows[relpath].id].filename)
        assert stored.read_bytes() == Path(banks.resolved_image_path(bank, rows[relpath])).read_bytes()
    assert _folder_digest(folder) == before


def test_a_bank_to_bank_copy_carries_the_decoded_picture(app, bank_ctx):
    """The new bank gets the effective picture, as for a cleaned image, and reads
    that PNG as it is: no second decoded copy of a copy."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    banks, bank, rows, folder, before = bank_ctx
    heic = rows['IMG_0001.HEIC']
    expected = Path(banks.resolved_image_path(bank, heic)).read_bytes()
    new_id = banks.start_bank_promote(app, 'local', bank.id, [heic.id], 'Copies')
    new_bank = db.session.get(ImageBank, new_id)
    (copied,) = BankImage.query.filter_by(bank_id=new_id).all()
    assert Path(new_bank.source_path, copied.relpath).read_bytes() == expected
    assert banks.resolved_image_path(new_bank, copied) == banks.abs_image_path(new_bank, copied)
    assert banks.analysis_image_path(new_bank, copied) == banks.abs_image_path(new_bank, copied)
    assert not banks._decoded_dir(new_id).exists()
    assert _folder_digest(folder) == before


def test_the_browser_gets_the_decoded_copy_never_heic_bytes(app, bank_ctx):
    banks, bank, rows, folder, before = bank_ctx
    client = app.test_client()
    heic = rows['IMG_0001.HEIC']
    for query in ('', '?original=1'):
        response = client.get(f'/api/bank/{bank.id}/file/{heic.id}{query}')
        assert response.status_code == 200 and response.mimetype == 'image/png'
        assert response.data.startswith(b'\x89PNG')
        response.close()
    response = client.get(f'/api/bank/{bank.id}/thumb/{heic.id}')
    assert response.status_code == 200 and response.mimetype == 'image/webp'
    response.close()
    assert _folder_digest(folder) == before
