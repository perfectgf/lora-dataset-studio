"""🗃️ Image bank — deleting an image mid-pass skips that image, never the pass.

A bank pass reads its rows, then walks them for minutes or hours, committing as
it goes. `expire_on_commit` is on, so every row it has not reached yet is a lazy
re-SELECT — and rows CAN disappear underneath it: `delete_bank` cancels the live
job *cooperatively* and then drops every bank_image row immediately, so the
still-running thread keeps iterating over rows that are already gone
(`delete_rejected` does the same if the registry has aged the pass out as stale).

SQLAlchemy's answer to that is fatal by default: plain attribute access on an
expired row whose database row is gone raises ObjectDeletedError, and so does a
commit carrying a write staged on one. Either killed the WHOLE pass — one image
deleted at the wrong moment threw away the analysis of every image already
walked, with `ObjectDeletedError: ...` as the only thing the user was told.

Each test below deletes ONE image from the middle of a pass, through that pass's
own inference hook (the point where the real deletion lands: the pass is inside a
model call, not between two rows), and pins the same three things:

    the pass finished without an error, the surviving images were written,
    and the end-of-pass line SAYS one image was skipped.

The last one is not decoration. A pass that silently swallowed the row would pass
the first two assertions while telling the user it analysed N images when it
analysed N-1.
"""
import json

import pytest
from PIL import Image


# --- fixtures ---------------------------------------------------------------
def _bank(workdir, n=3):
    """A bank over ``n`` real image files, committed."""
    from app.extensions import db
    from app.services import image_bank_service as banks
    src = workdir / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (900, 900), (10 * i, 90, 160)).save(str(src / f'a{i}.jpg'))
    bank, _added = banks.create_bank('local', 'Dump', str(src))
    db.session.commit()
    return bank.id


def _image_ids(bank_id):
    from app.models import BankImage
    return [r.id for r in BankImage.query.filter_by(bank_id=bank_id)
            .order_by(BankImage.id.asc()).all()]


def _delete_image(image_id):
    """Delete one bank row the way another request would.

    A bulk DELETE with ``synchronize_session=False`` plus a commit reproduces
    exactly what the pass's session sees when a *different* session removed the
    row: it is gone from the database, while the pass still holds the (now
    expired) instance in its identity map. That is the whole trap — an ORM row
    object that looks alive and raises the moment anything reads it."""
    from app.extensions import db
    from app.models import BankImage
    BankImage.query.filter(BankImage.id == image_id).delete(
        synchronize_session=False)
    db.session.commit()


def _new_job(kind):
    """A job dict shaped like bank_jobs.start()'s, so progress/bump/detail are
    the real ones and the test can read the line the user would see."""
    return {'kind': kind, 'done': 0, 'total': 0, 'error': None, 'cancelled': False,
            'finished': False, 'detail': None, 'started_at': 0.0, '_touched': 0.0,
            '_cancel_hook': None, 'pipeline': None}


def _run(pass_fn, job):
    """Run a pass exactly as bank_jobs._run does — a crash becomes job['error']
    instead of propagating, which is how the user meets it (a red toast and a
    dead pass)."""
    try:
        pass_fn(job)
    except Exception as e:  # noqa: BLE001 — mirrors the real runner
        job['error'] = f'{type(e).__name__}: {e}'
    return job


def _assert_survived(job, skipped=1):
    assert job['error'] is None, (
        'one image deleted mid-pass killed the whole pass: ' + str(job['error']))
    assert job['detail'], 'the pass ended without saying anything'
    assert f'{skipped} skipped (deleted while the pass ran)' in job['detail'], (
        'the pass survived but never told the user an image had vanished — '
        f'it reported: {job["detail"]!r}')


@pytest.fixture()
def bank_ctx(app, tmp_path):
    """(bank_id, [image ids]) inside a live app context."""
    with app.app_context():
        bank_id = _bank(tmp_path)
        yield bank_id, _image_ids(bank_id)


# --- the two Ollama passes (watermark, framing) ------------------------------
def _run_vision_pass(monkeypatch, pass_fn, kind, victim, answer='{"present": false}'):
    """Drive a vision pass one image at a time, deleting ``victim`` from inside
    the FIRST model call — i.e. while the pass is waiting on the answer for the
    image before it, which is where the real click lands."""
    from app.services import vision_ollama, vision_pool
    calls = {'n': 0}

    def fake_describe(image_bytes, *a, **k):
        if calls['n'] == 0:
            _delete_image(victim)
        calls['n'] += 1
        return answer

    monkeypatch.setattr(vision_pool, 'vision_concurrency', lambda *a, **k: 1)
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake_describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    return _run(pass_fn, _new_job(kind))


def test_the_watermark_scan_skips_an_image_deleted_under_it(bank_ctx, monkeypatch):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, ids = bank_ctx
    job = _run_vision_pass(monkeypatch, banks._watermark_job(bank_id, False),
                           'watermark', victim=ids[1])
    _assert_survived(job)
    survivors = BankImage.query.filter(BankImage.id.in_([ids[0], ids[2]])).all()
    assert [r.watermark_state for r in survivors] == ['none', 'none'], (
        'the pass continued but stopped writing its verdicts')
    assert db.session.get(BankImage, ids[1]) is None


def test_the_framing_pass_skips_an_image_deleted_under_it(bank_ctx, monkeypatch):
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, ids = bank_ctx
    job = _run_vision_pass(monkeypatch, banks._framing_job(bank_id, False),
                           'framing', victim=ids[1], answer='{"framing": "face"}')
    _assert_survived(job)
    survivors = BankImage.query.filter(BankImage.id.in_([ids[0], ids[2]])).all()
    assert all(r.framing for r in survivors), (
        'the pass continued but stopped writing its classifications')


# --- the quality scan (thread pool, no model) --------------------------------
def test_the_quality_scan_skips_an_image_deleted_under_it(bank_ctx, monkeypatch):
    """The scan analyses files on a pool and writes the verdicts back on its own
    thread. The delete is fired from the progress bump right after the first
    image is written — i.e. between two write-backs, which is where a click
    during a long scan lands. (It cannot be fired from inside the analysis: that
    runs on a worker with no app context, as a real request never would.)"""
    from app.extensions import db
    from app.models import BankImage
    from app.services import bank_jobs, image_bank_service as banks

    bank_id, ids = bank_ctx
    real_bump = bank_jobs.bump
    calls = {'n': 0}

    def fake_bump(job, n=1):
        if calls['n'] == 0:
            _delete_image(ids[1])
        calls['n'] += 1
        return real_bump(job, n)

    monkeypatch.setattr(banks.bank_jobs, 'bump', fake_bump)
    job = _run(banks._scan_job(bank_id, False), _new_job('scan'))
    _assert_survived(job)
    assert all(db.session.get(BankImage, i).quality_state
               for i in (ids[0], ids[2])), (
        'the scan continued but stopped writing its measurements')


# --- the two subprocess passes (score, faces) --------------------------------
def _run_subprocess_pass(monkeypatch, banks, pass_fn, kind, victim, results):
    """Delete an image from inside the inference child — the hour-long wait the
    pass hands its path list to."""
    from app.services import bank_transfer_metadata as transfer

    # The real children bind every verdict to the exact bytes they decoded.
    # Keep this deletion-focused double faithful to that wire contract.
    for path, result in results.items():
        result.setdefault('fingerprint', transfer.content_fingerprint_path(path))

    def fake_drive(job, python, script, payload, cache_path, progress_re, window):
        _delete_image(victim)
        return {'ok': True, 'results': results, 'clusters': {}}, [], 0

    monkeypatch.setattr(banks, '_drive_infer_subprocess', fake_drive)
    monkeypatch.setattr(banks.bank_jobs, 'cancelled', lambda job: False)
    return _run(pass_fn, _new_job(kind))


def test_the_scoring_pass_skips_an_image_deleted_under_it(bank_ctx, monkeypatch,
                                                          tmp_path):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, ids = bank_ctx
    paths = {str((tmp_path / 'src' / f'a{i}.jpg')): {'state': 'ok', 'aesthetic': 6.0,
                                                     'nsfw': 0.1}
             for i in range(3)}
    monkeypatch.setattr('app.capabilities.bank_scoring_gpu_available', lambda: False)
    job = _run_subprocess_pass(monkeypatch, banks, banks._score_job(bank_id),
                               'score', ids[1], paths)
    _assert_survived(job)
    assert all(db.session.get(BankImage, i).aesthetic_score for i in (ids[0], ids[2])), (
        'the scoring pass continued but stopped writing its scores')
    # The count must come from the rows this loop WROTE, not from the child's
    # report. The child is handed three paths and scores three; only this loop
    # knows one of the images stopped existing. Reporting the child's number
    # produced "scored 3 image(s), 1 skipped" over a bank of three — two claims
    # that cannot both be true.
    assert 'scored 2 image(s)' in job['detail'], job['detail']


def test_the_face_pass_skips_an_image_deleted_under_it(bank_ctx, monkeypatch,
                                                       tmp_path):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, ids = bank_ctx
    paths = {str((tmp_path / 'src' / f'a{i}.jpg')): {'state': 'ok', 'det': 0.9}
             for i in range(3)}
    monkeypatch.setattr(banks, '_resolve_face_device', lambda: ('cpu', False))
    job = _run_subprocess_pass(monkeypatch, banks, banks._faces_job(bank_id),
                               'faces', ids[1], paths)
    _assert_survived(job)
    assert all(db.session.get(BankImage, i).face_state for i in (ids[0], ids[2])), (
        'the face pass continued but stopped writing its embeddings verdict')


# --- the caption pass -------------------------------------------------------
def test_the_caption_pass_skips_an_image_deleted_under_it(bank_ctx, monkeypatch):
    """Captions are written one at a time from a callback, so the delete lands
    between two captions — the shape a long overnight caption run really meets."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import face_dataset_service as fds
    from app.services import image_bank_service as banks

    bank_id, ids = bank_ctx

    # **_over absorbs the per-run engine/model overrides the pass now forwards. This
    # double is about SURVIVING A DELETION, not about the caption config, and pinning
    # the real function's exact signature here would make every new option a false
    # failure in a test that never looks at one.
    def fake_caption_paths(paths, extra_instructions=None, should_cancel=None,
                           on_caption=None, progress=None, **_over):
        for i, p in enumerate(paths):
            if i == 1:
                _delete_image(ids[1])
            on_caption(p, f'a caption for {i}')

    monkeypatch.setattr(fds, 'caption_paths', fake_caption_paths)
    job = _run(banks._caption_job(bank_id, None, False), _new_job('caption'))
    _assert_survived(job)
    assert 'done — 2 captioned' in job['detail'], job['detail']
    assert all(db.session.get(BankImage, i).caption for i in (ids[0], ids[2])), (
        'the caption pass continued but stopped storing captions')


# --- the two watermark cleaning levels --------------------------------------
def _flag_watermarks(ids, manual=False):
    """Put every image in the cleaning pool: flagged, with something to act on."""
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    for i in ids:
        row = db.session.get(BankImage, i)
        bank = db.session.get(ImageBank, row.bank_id)
        row.watermark_state = 'detected'
        row.watermark_bbox = json.dumps([0.0, 0.9, 1.0, 1.0])
        row.watermark_fingerprint = transfer.content_fingerprint_path(
            banks.abs_image_path(bank, row))
        if manual:
            row.watermark_regions = json.dumps([[0.0, 0.9, 1.0, 1.0]])
    db.session.commit()


def test_the_auto_crop_level_skips_an_image_deleted_under_it(bank_ctx, monkeypatch):
    from app.extensions import db
    from app.models import BankImage
    from app.services import face_dataset_service as fds
    from app.services import image_bank_service as banks

    bank_id, ids = bank_ctx
    _flag_watermarks(ids)
    # The deletion fires from inside the cut itself — the level's one engine
    # call, since it cuts straight from the source (no staging copy any more).
    real_cut = banks._cut_clean_copy
    calls = {'n': 0}

    def cut_then_delete(bank_id_, row, src_path, box):
        if calls['n'] == 0:
            _delete_image(ids[1])
        calls['n'] += 1
        return real_cut(bank_id_, row, src_path, box)

    monkeypatch.setattr(banks, '_cut_clean_copy', cut_then_delete)
    monkeypatch.setattr(fds, '_route_watermark',
                        lambda bbox, w, h, allow_crop=True: ('crop', bbox))
    job = _run(banks._watermark_crop_job(bank_id), _new_job('watermark_crop'))
    assert calls['n'] >= 1, 'the hook never fired — the level no longer cuts through _cut_clean_copy'
    _assert_survived(job)
    assert all(db.session.get(BankImage, i).watermark_state == 'cleaned'
               for i in (ids[0], ids[2])), (
        'the crop level continued but stopped marking images cleaned')


def test_the_inpaint_level_skips_an_image_deleted_under_its_batch(bank_ctx,
                                                                  monkeypatch):
    """LaMa repaints the whole selection in ONE batch, so its rows are held
    across a call that runs for minutes — the widest window of them all."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks
    from app.services import watermark_lama

    bank_id, ids = bank_ctx
    _flag_watermarks(ids, manual=True)     # a hand mask routes straight to LaMa

    def fake_batch(items, device='cpu'):
        _delete_image(ids[1])              # deleted while the batch was running
        return {it['image_path']: (True, None) for it in items}

    monkeypatch.setattr(watermark_lama, 'is_available', lambda: True)
    monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')
    monkeypatch.setattr(watermark_lama, 'inpaint_batch', fake_batch)
    job = _run(banks._watermark_inpaint_job(bank_id, 'auto'),
               _new_job('watermark_inpaint'))
    _assert_survived(job)
    assert all(db.session.get(BankImage, i).watermark_clean_method == 'lama'
               for i in (ids[0], ids[2])), (
        'the inpaint level continued but stopped recording its repaints')
    assert not banks.clean_image_path(bank_id, ids[1]).exists(), (
        'the repainted copy of a deleted image was left behind with no row '
        'pointing at it')
