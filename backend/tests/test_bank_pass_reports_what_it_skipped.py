"""🗃️ Image bank — a pass never reports its successes without its failures.

Reported on Discord (2026-08-07) by `_mr.arrow_` as "framing is not getting
persisted", then narrowed by him to the sentence this file exists for: "Running a
new framing added 4 new framed images, out of the 12 images it got through before
cancelling."

Nothing was broken about the persistence. His diagnostic showed, every ~5 s:

    ERROR app.gpu_window: vision GPU window renewal failed
    sqlite3.OperationalError: database is locked
    [SQL: UPDATE system_state SET value=?, updated_at=? WHERE key = ?]  ('vision_in_progress')

`database is locked` fails the renewal, `_admit_local_ollama` is fail-closed and
raises `LocalOllamaFenceError`, and the image is never shown to the model. Leaving
`framing` NULL there is CORRECT — it is what lets a re-run finish the job, where
writing 'unknown' would freeze an error into the bank. What was wrong is that the
pass said none of it: the cancelled branch named `classified` and nothing else, so
8 unclassified images out of 12 were reported as "4 classified so far", and the
only reading left to the user was that framing does not persist.

The tests below pin the REPORTING, on the rendered value of `job['detail']`:

    every ending of a pass — the finished one and the cancelled one — names
    every image the pass did not write, and says which kind of not-written.
"""
import pytest
from PIL import Image


# --- fixtures (same shape as test_bank_pass_survives_deleted_image) ----------
def _bank(workdir, n=12):
    from app.extensions import db
    from app.services import image_bank_service as banks
    src = workdir / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (900, 900), (10 * i, 90, 160)).save(str(src / f'a{i:02d}.jpg'))
    bank, _added = banks.create_bank('local', 'Dump', str(src))
    db.session.commit()
    return bank.id


def _new_job(kind):
    return {'kind': kind, 'done': 0, 'total': 0, 'error': None, 'cancelled': False,
            'finished': False, 'detail': None, 'started_at': 0.0, '_touched': 0.0,
            '_cancel_hook': None, 'pipeline': None}


def _run(pass_fn, job):
    try:
        pass_fn(job)
    except Exception as e:  # noqa: BLE001 — mirrors bank_jobs._run
        job['error'] = f'{type(e).__name__}: {e}'
    return job


@pytest.fixture()
def bank_ctx(app, tmp_path):
    with app.app_context():
        yield _bank(tmp_path)


def _framing_over_ollama(monkeypatch, describe):
    """Drive 📐 Framing one image at a time over a fake Ollama."""
    from app.services import vision_ollama, vision_pool
    monkeypatch.setattr(vision_pool, 'vision_concurrency', lambda *a, **k: 1)
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


# --- the reported run --------------------------------------------------------
def test_a_cancelled_framing_pass_names_the_images_the_gpu_window_lost(
        bank_ctx, monkeypatch):
    """4 classified, 8 refused by the GPU fence, then Stop — the exact run.

    The cancelled line has to carry the 8. Before this fix it read
    'cancelled — 4 classified so far' and stopped there, which is what turned a
    GPU-window outage into a bug report about persistence.
    """
    from app.models import BankImage
    from app.services import bank_jobs, image_bank_service as banks
    from app.services.vision_ollama import LocalOllamaFenceError

    bank_id = bank_ctx
    calls = {'n': 0}

    def describe(image_bytes, *a, **k):
        calls['n'] += 1
        if calls['n'] <= 4:
            return '{"framing": "face"}'
        if calls['n'] >= 12:
            job['cancelled'] = True     # the user hits Stop after 12 images
        raise LocalOllamaFenceError(
            'The Vision GPU window expired before Ollama could start safely.')

    _framing_over_ollama(monkeypatch, describe)
    monkeypatch.setattr(bank_jobs, 'cancelled', lambda j: bool(j['cancelled']))
    job = _new_job('framing')
    _run(banks._framing_job(bank_id, False), job)

    assert job['error'] is None, job['error']
    detail = job['detail']
    assert '4 classified' in detail, detail
    assert '8 not analysed' in detail, (
        'the cancelled pass named its 4 successes and hid the 8 images the GPU '
        f'window refused — it reported: {detail!r}')
    assert 'run the pass again' in detail, detail
    # And the row stays NULL, which is the behaviour the sentence now explains.
    assert BankImage.query.filter_by(bank_id=bank_id).filter(
        BankImage.framing.isnot(None)).count() == 4


def test_a_framing_storm_says_so_while_it_runs_not_only_at_the_end(
        bank_ctx, monkeypatch):
    """A live pass whose every call is refused must SAY it before the last line.

    The progress bar advancing over images that are all failing is what let a
    211-image bank burn through a whole run with nothing to show; the detail is
    already refreshed on screen, so it is where this belongs.
    """
    from app.services import bank_jobs, image_bank_service as banks
    from app.services.vision_ollama import LocalOllamaFenceError

    bank_id = bank_ctx
    seen = []
    real_progress = bank_jobs.progress

    def spy(job, done=None, total=None, detail=None):
        if detail is not None:
            seen.append(detail)
        return real_progress(job, done=done, total=total, detail=detail)

    def describe(image_bytes, *a, **k):
        raise LocalOllamaFenceError(
            'The Vision GPU window expired before Ollama could start safely.')

    _framing_over_ollama(monkeypatch, describe)
    monkeypatch.setattr(banks.bank_jobs, 'progress', spy)
    job = _run(banks._framing_job(bank_id, False), _new_job('framing'))

    assert job['error'] is None, job['error']
    mid = [d for d in seen[:-1] if 'could not reach the vision model' in d]
    assert mid, (
        'twelve images in a row failed to reach the model and the pass said '
        f'nothing until it was over — the lines it showed were: {seen!r}')
    assert 'in a row' in mid[0], mid[0]


def test_a_framing_pass_names_the_images_that_changed_under_it(bank_ctx,
                                                               monkeypatch):
    """`_prepare_analysis_write` refusing is a real outcome, not a silent one.

    An image whose bytes moved between the model call and the write-back is
    never stored — correctly, the answer describes something else. Uncounted, it
    let a pass over 12 images finish as 'done — 0 classified' with no clue.
    """
    from app.services import image_bank_service as banks

    bank_id = bank_ctx
    _framing_over_ollama(monkeypatch, lambda *a, **k: '{"framing": "face"}')
    monkeypatch.setattr(banks, '_prepare_analysis_write',
                        lambda row, path, fingerprint: False)
    job = _run(banks._framing_job(bank_id, False), _new_job('framing'))

    assert job['error'] is None, job['error']
    assert '12 skipped (the image changed while the pass ran)' in job['detail'], (
        'the pass wrote nothing for twelve images and reported none of them: '
        f'{job["detail"]!r}')


def test_a_framing_pass_where_ollama_answers_nothing_says_so(bank_ctx,
                                                             monkeypatch):
    """An empty answer leaves the row NULL — correctly — and used to be counted
    nowhere, so twelve unanswered images ended as 'done — 0 classified'. The
    watermark pass has counted this since it shipped; framing never did."""
    from app.services import image_bank_service as banks

    bank_id = bank_ctx
    _framing_over_ollama(monkeypatch, lambda *a, **k: '')
    job = _run(banks._framing_job(bank_id, False), _new_job('framing'))

    assert job['error'] is None, job['error']
    assert '12 not analysed (the vision model returned nothing' in job['detail'], (
        'every answer was empty and the pass reported none of it: '
        + repr(job['detail']))


def test_a_cancelled_framing_pass_names_unreadable_files_too(bank_ctx,
                                                             monkeypatch):
    """The cancelled line owes the SAME clauses as the finished one — including
    the failures that are not fence refusals."""
    from app.services import bank_jobs, image_bank_service as banks

    bank_id = bank_ctx
    calls = {'n': 0}

    def describe(image_bytes, *a, **k):
        calls['n'] += 1
        if calls['n'] <= 2:
            return '{"framing": "body"}'
        if calls['n'] >= 6:
            job['cancelled'] = True
        raise RuntimeError('decode failed')

    _framing_over_ollama(monkeypatch, describe)
    monkeypatch.setattr(bank_jobs, 'cancelled', lambda j: bool(j['cancelled']))
    job = _new_job('framing')
    _run(banks._framing_job(bank_id, False), job)

    assert job['error'] is None, job['error']
    assert '2 classified' in job['detail'], job['detail']
    assert 'unreadable' in job['detail'], (
        'a stopped pass hid the files it could not read: ' + repr(job['detail']))


# --- the sibling passes had the same hole ------------------------------------
def test_a_cancelled_caption_pass_names_the_images_it_did_not_write(bank_ctx,
                                                                    monkeypatch):
    from app.models import BankImage
    from app.services import bank_jobs, face_dataset_service as fds
    from app.services import image_bank_service as banks

    bank_id = bank_ctx
    ids = [r.id for r in BankImage.query.filter_by(bank_id=bank_id)
           .order_by(BankImage.id.asc()).all()]

    def _delete(image_id):
        from app.extensions import db
        BankImage.query.filter(BankImage.id == image_id).delete(
            synchronize_session=False)
        db.session.commit()

    def fake_caption_paths(paths, extra_instructions=None, should_cancel=None,
                           on_caption=None, progress=None, **_over):
        for i, p in enumerate(paths):
            if i == 2:
                _delete(ids[3])        # deleted while the caption before it ran
            if i >= 5:
                job['cancelled'] = True    # the user hits Stop
                return
            on_caption(p, f'a caption for {i}')

    monkeypatch.setattr(fds, 'caption_paths', fake_caption_paths)
    monkeypatch.setattr(bank_jobs, 'cancelled', lambda j: bool(j['cancelled']))
    job = _new_job('caption')
    _run(banks._caption_job(bank_id, None, False), job)

    assert job['error'] is None, job['error']
    assert 'cancelled —' in job['detail'], job['detail']
    assert '1 skipped (deleted while the pass ran)' in job['detail'], (
        'the stopped caption pass reported only what it wrote: '
        + repr(job['detail']))


def test_a_cancelled_watermark_scan_names_what_it_could_not_read(bank_ctx,
                                                                 monkeypatch):
    from app.services import bank_jobs, image_bank_service as banks

    bank_id = bank_ctx
    calls = {'n': 0}

    def describe(image_bytes, *a, **k):
        calls['n'] += 1
        if calls['n'] >= 5:
            job['cancelled'] = True
        raise RuntimeError('decode failed')

    _framing_over_ollama(monkeypatch, describe)
    monkeypatch.setattr(bank_jobs, 'cancelled', lambda j: bool(j['cancelled']))
    job = _new_job('watermark')
    _run(banks._watermark_job(bank_id, False), job)

    assert job['error'] is None, job['error']
    assert 'cancelled —' in job['detail'], job['detail']
    assert 'unreadable' in job['detail'], (
        'the stopped watermark scan hid every file it failed on: '
        + repr(job['detail']))


def test_a_framing_pass_quotes_the_fences_own_reason_not_the_gpu_window_sentence(
        bank_ctx, monkeypatch):
    """The fence refuses for more than one reason. When another tool holds the
    model, the historical "GPU window expired" clause named a remedy that was
    false for that case; the pass now quotes the refusal it actually got."""
    from app.services import image_bank_service as banks
    from app.services.ollama_gpu_fence import FENCE_BLOCKED_MESSAGE
    from app.services.vision_ollama import LocalOllamaFenceError

    bank_id = bank_ctx
    calls = {'n': 0}

    def describe(image_bytes, *a, **k):
        calls['n'] += 1
        if calls['n'] <= 2:
            return '{"framing": "face"}'
        raise LocalOllamaFenceError(FENCE_BLOCKED_MESSAGE)

    _framing_over_ollama(monkeypatch, describe)
    job = _new_job('framing')
    _run(banks._framing_job(bank_id, False), job)

    assert job['error'] is None, job['error']
    detail = job['detail']
    assert '2 classified' in detail, detail
    reason = FENCE_BLOCKED_MESSAGE.rstrip('.')
    assert f'10 not analysed ({reason} — run the pass again to finish them)' in detail, detail
    assert 'GPU window expired' not in detail, detail
