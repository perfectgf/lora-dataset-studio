"""A long Bank pass survives what used to empty it or end it.

Reported on 2026-11-15, on a bank of 36,000 images whose framing and caption
passes run for days:

* The machine slept for seven hours. On wake-up the Vision GPU window's persisted
  record had expired, every call was refused with "The Vision GPU window expired
  before Ollama could start safely", and the pass refused its 16,596 remaining
  images in six minutes, then ended "done".
* The disk was full for a while. One refused commit ended the caption step, under
  "This session is in 'prepared' state", which named nothing a user can act on.

Each test below replays one of these and pins what the pass now does instead.
"""
import os
import sqlite3
import time

import pytest
from PIL import Image
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError


def _refuse_commits(engine, armed):
    """Make the database refuse the next ``armed['n']`` commits the way a full disk
    does: the COMMIT itself fails, which leaves a session that is not rolled back
    in 'prepared' state. Returns the function that removes the refusal."""
    def _refuse(conn):
        if armed.get('n', 0) > 0:
            armed['n'] -= 1
            armed['refused'] = armed.get('refused', 0) + 1
            raise OperationalError('COMMIT', None, sqlite3.OperationalError('disk I/O error'))

    event.listen(engine, 'commit', _refuse)
    return lambda: event.remove(engine, 'commit', _refuse)


# --- the Vision GPU window ---------------------------------------------------------
def test_an_expired_reservation_is_written_again_while_its_window_runs(app):
    """What a sleep longer than the TTL leaves behind: no record, and a pass that
    still runs. No other owner could take the GPU meanwhile (the in-process fence
    stayed up), so the next renewal writes the record again."""
    import app.gpu_window as gw
    from app.job_queue import queue_manager
    with app.app_context():
        with gw.gpu_exclusive_vision_window(flag_ttl=300):
            token = queue_manager._get_system_state('vision_in_progress')
            queue_manager._set_system_state('vision_in_progress', token, ttl_seconds=-1)
            assert queue_manager._get_system_state('vision_in_progress') is None
            assert gw.renew_gpu_exclusive_vision_window() is True
            assert queue_manager._get_system_state('vision_in_progress') == token
        assert queue_manager._get_system_state('vision_in_progress') is None


def test_a_renewal_never_takes_the_reservation_of_another_owner(app):
    import app.gpu_window as gw
    from app.job_queue import queue_manager
    with app.app_context():
        with gw.gpu_exclusive_vision_window(flag_ttl=300):
            queue_manager._set_system_state('vision_in_progress', 'someone-else-token')
            assert gw.renew_gpu_exclusive_vision_window() is False
            assert queue_manager._get_system_state('vision_in_progress') == 'someone-else-token'
        assert queue_manager._get_system_state('vision_in_progress') == 'someone-else-token'
        queue_manager._set_system_state('vision_in_progress', None)


def test_a_worker_that_outlived_its_window_cannot_bring_the_reservation_back(app):
    """Writing an expired record again is for a window that still runs. A call that
    is still alive after its window exited must stay refused."""
    import app.gpu_window as gw
    from app.job_queue import queue_manager
    with app.app_context():
        with gw.gpu_exclusive_vision_window(flag_ttl=300):
            late_call = gw.bind_vision_window_context(gw.renew_gpu_exclusive_vision_window)
        assert late_call() is False
        assert queue_manager._get_system_state('vision_in_progress') is None


def test_a_refused_renewal_keeps_the_reservation_and_leaves_the_session_usable(app):
    """The write is refused, not the ownership: the stored record is still this
    window's and has not expired. And the session the renewal ran on, which is the
    pass's own on its thread, must be usable afterwards."""
    import app.gpu_window as gw
    from app.extensions import db
    from app.job_queue import queue_manager
    with app.app_context():
        armed = {'n': 0}
        remove = _refuse_commits(db.session.get_bind(), armed)
        try:
            with gw.gpu_exclusive_vision_window(flag_ttl=300):
                token = queue_manager._get_system_state('vision_in_progress')
                armed['n'] = 1
                assert gw.renew_gpu_exclusive_vision_window() is True
                assert armed.get('refused') == 1, 'the commit of the renewal was not refused'
                # Before the rollback this raised "This session is in 'prepared' state".
                assert db.session.execute(text('select 1')).scalar() == 1
                assert queue_manager._get_system_state('vision_in_progress') == token
        finally:
            remove()
        assert queue_manager._get_system_state('vision_in_progress') is None


def test_the_heartbeat_keeps_asking_after_a_renewal_that_failed(app, monkeypatch):
    """It used to stop for good at the first failure, so one refused write left a
    pass of several hours without any renewal."""
    import app.gpu_window as gw
    calls = []
    answers = iter([False, True])

    def renew(token, ttl):
        calls.append(time.monotonic())
        return next(answers, True)

    monkeypatch.setattr(gw, '_renew_owned_vision_token', renew)
    monkeypatch.setattr(gw, '_heartbeat_interval_seconds', lambda ttl: 0.01)
    with app.app_context():
        with gw.gpu_exclusive_vision_window(flag_ttl=300):
            deadline = time.monotonic() + 5
            while len(calls) < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
    assert len(calls) >= 3, f'the heartbeat stopped after {len(calls)} renewal(s)'


def test_the_failure_of_the_pass_is_reported_when_the_cleanup_fails_too(app, monkeypatch):
    """A database that refuses every write fails the pass AND the cleanup of its
    window. The pass's own error is the one a user can act on."""
    import app.gpu_window as gw
    with app.app_context():
        with pytest.raises(ValueError, match='the real cause'):
            with gw.gpu_exclusive_vision_window():
                monkeypatch.setattr(
                    gw, '_in_app_context',
                    lambda _callback: (_ for _ in ()).throw(RuntimeError('db cleanup')))
                raise ValueError('the real cause')
    assert gw.vision_gpu_window_blocks_gpu() is False


# --- the Bank passes, as the user reads them ----------------------------------------
def _images(folder, n):
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (64, 64), (10 * i + 20, 90, 160)).save(os.path.join(folder, f'img{i:02d}.png'))


def _use_ollama_backend(app):
    with app.app_context():
        import app.config as cfg
        cfg.save_config({'captioning': {'backend': 'ollama'}})


def _mkbank(client, tmp_path, n):
    src = tmp_path / 'src'
    _images(str(src), n)
    r = client.post('/api/bank/create', json={'name': 'B', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id']


def _over_ollama(monkeypatch, describe):
    """Drive a pass one image at a time over a fake Ollama."""
    from app.services import vision_ollama, vision_pool
    monkeypatch.setattr(vision_pool, 'vision_concurrency', lambda *a, **k: 1)
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)


def _refuse_from_the_first_answer(armed, answer, how_many):
    """A fake model call whose first answer arms the refusal: the commits that
    follow it, the pass's own, are the ones the database refuses."""
    def describe(image_bytes, *a, **k):
        if not armed.get('armed'):
            armed['armed'] = True
            armed['n'] = how_many
        return answer
    return describe


def test_a_caption_write_refused_once_is_tried_again_and_the_pass_goes_on(
        client, tmp_path, app, monkeypatch):
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks
    _use_ollama_backend(app)
    bank_id = _mkbank(client, tmp_path, 3)
    armed = {}
    _over_ollama(monkeypatch, _refuse_from_the_first_answer(armed, 'a plain wall', 1))
    monkeypatch.setattr(banks, '_write_retry_pause', lambda seconds, job: False)
    with app.app_context():
        remove = _refuse_commits(db.session.get_bind(), armed)
    try:
        r = client.post(f'/api/bank/{bank_id}/caption', json={})
    finally:
        remove()
    assert r.status_code == 202, r.get_json()
    activity = client.get(f'/api/bank/{bank_id}').get_json()['activity']
    assert activity['finished'] is True and activity['error'] is None, activity
    assert activity['detail'].startswith('done — 3 captioned'), activity['detail']
    assert armed.get('refused') == 1, 'no commit of the pass was refused'
    with app.app_context():
        assert BankImage.query.filter_by(bank_id=bank_id, caption='a plain wall').count() == 3


def test_a_caption_write_that_stays_refused_ends_the_pass_on_its_real_cause(
        client, tmp_path, app, monkeypatch):
    """A disk that stays full cannot be waited out. The pass ends, and its error
    says what was refused and what to check, not the state of a session."""
    from app.extensions import db
    from app.services import image_bank_service as banks
    _use_ollama_backend(app)
    bank_id = _mkbank(client, tmp_path, 3)
    armed = {}
    _over_ollama(monkeypatch, _refuse_from_the_first_answer(armed, 'a plain wall', 10 ** 6))
    monkeypatch.setattr(banks, '_write_retry_pause', lambda seconds, job: False)
    monkeypatch.setattr(banks, '_WRITE_PATIENCE_SECONDS', 0.0)
    with app.app_context():
        remove = _refuse_commits(db.session.get_bind(), armed)
    try:
        r = client.post(f'/api/bank/{bank_id}/caption', json={})
    finally:
        remove()
    assert r.status_code == 202, r.get_json()
    activity = client.get(f'/api/bank/{bank_id}').get_json()['activity']
    assert activity['finished'] is True
    error = activity['error'] or ''
    assert 'refused to save' in error and 'disk I/O error' in error, error
    assert 'free space' in error, error
    assert "'prepared' state" not in error, error


@pytest.fixture()
def bank_ctx(app, tmp_path):
    from app.extensions import db
    from app.services import image_bank_service as banks
    with app.app_context():
        src = tmp_path / 'framing-src'
        src.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            Image.new('RGB', (900, 900), (10 * i, 90, 160)).save(str(src / f'a{i:02d}.jpg'))
        bank, _added = banks.create_bank('local', 'Dump', str(src))
        db.session.commit()
        yield bank.id


def _new_job(kind):
    return {'kind': kind, 'done': 0, 'total': 0, 'error': None, 'cancelled': False,
            'finished': False, 'detail': None, 'started_at': 0.0, '_touched': 0.0,
            '_cancel_hook': None, 'pipeline': None}


def test_a_framing_write_refused_once_is_tried_again_and_the_pass_goes_on(bank_ctx, monkeypatch):
    from app.extensions import db
    from app.models import BankImage
    from app.services import bank_jobs, image_bank_service as banks
    bank_id = bank_ctx
    armed = {}
    _over_ollama(monkeypatch, _refuse_from_the_first_answer(armed, '{"framing": "face"}', 1))
    monkeypatch.setattr(banks, '_write_retry_pause', lambda seconds, job: False)
    monkeypatch.setattr(bank_jobs, 'cancelled', lambda j: bool(j['cancelled']))
    job = _new_job('framing')
    remove = _refuse_commits(db.session.get_bind(), armed)
    try:
        banks._framing_job(bank_id, False)(job)
    finally:
        remove()
    assert job['detail'].startswith('done — 6 classified'), job['detail']
    assert armed.get('refused') == 1, 'no commit of the pass was refused'
    assert BankImage.query.filter_by(bank_id=bank_id, framing='face').count() == 6
