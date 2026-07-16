import json, time
from unittest.mock import patch
import pytest


def test_add_job_inserts_pending(app):
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}}, prompt='p',
                                    metadata={'model_name': 'klein_edit_dataset'})
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'pending' and json.loads(row.job_metadata)['model_name'] == 'klein_edit_dataset'


def test_add_job_empty_workflow_raises(app):
    from app.job_queue import queue_manager
    with app.app_context():
        try:
            queue_manager.add_job(workflow_data={})
            assert False, 'expected ValueError'
        except ValueError:
            pass


def test_system_state_ttl(app):
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager._set_system_state('flag', True, ttl_seconds=1)
        assert queue_manager._get_system_state('flag') is True
        time.sleep(1.2)
        assert queue_manager._get_system_state('flag') is None
        queue_manager._set_system_state('k', {'a': 1})
        assert queue_manager._get_system_state('k') == {'a': 1}


def test_system_state_expired_read_deletes_row(app):
    """Expired TTL reads must lazily delete the row, not just mask it."""
    from app.job_queue import queue_manager
    from app.models import SystemState
    from app.extensions import db
    with app.app_context():
        queue_manager._set_system_state('flag', True, ttl_seconds=1)
        time.sleep(1.2)
        assert queue_manager._get_system_state('flag') is None
        assert db.session.get(SystemState, 'flag') is None


def test_system_state_none_deletes(app):
    from app.job_queue import queue_manager
    from app.models import SystemState
    from app.extensions import db
    with app.app_context():
        queue_manager._set_system_state('k', {'a': 1})
        queue_manager._set_system_state('k', None)
        assert queue_manager._get_system_state('k') is None
        assert db.session.get(SystemState, 'k') is None


def test_worker_completes_job_and_dispatches(app):
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    done = {}
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}}, prompt='p',
                                    metadata={'model_name': 'klein_edit_dataset'})
    with patch('app.job_queue._submit', return_value='prompt-1'), \
         patch('app.job_queue._poll_outputs', return_value=('out.png', False)), \
         patch('app.job_queue._dispatch_completion',
               side_effect=lambda job, fn, failed: done.update(fn=fn, failed=failed)):
        with app.app_context():
            queue_manager.process_one()          # synchronous single-step API for tests
            row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
            assert row.status == 'completed' and done == {'fn': 'out.png', 'failed': False}


def test_worker_dispatches_failed_on_poll_failure(app):
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    done = {}
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
    with patch('app.job_queue._submit', return_value='prompt-1'), \
         patch('app.job_queue._poll_outputs', return_value=(None, True)), \
         patch('app.job_queue._dispatch_completion',
               side_effect=lambda job, fn, failed: done.update(fn=fn, failed=failed)):
        with app.app_context():
            queue_manager.process_one()
            row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
            assert row.status == 'failed' and done == {'fn': None, 'failed': True}


def test_worker_survives_submit_exception(app):
    """A missing/broken seam (e.g. utils.comfyui not lifted yet) must fail the
    job, not crash the worker thread."""
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    done = {}
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
    with patch('app.job_queue._submit', side_effect=ImportError('no comfyui yet')), \
         patch('app.job_queue._dispatch_completion',
               side_effect=lambda job, fn, failed: done.update(fn=fn, failed=failed)):
        with app.app_context():
            queue_manager.process_one()
            row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
            assert row.status == 'failed' and done == {'fn': None, 'failed': True}


def test_process_one_returns_false_when_empty(app):
    from app.job_queue import queue_manager
    with app.app_context():
        assert queue_manager.process_one() is False


def test_process_one_skips_while_training_in_progress(app):
    """Jobs must stay pending (not be claimed/submitted) while training/vision
    holds the GPU; once the flag clears, the queue processes normally."""
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=60)
        with patch('app.job_queue._submit') as submit:
            assert not queue_manager.process_one()
            submit.assert_not_called()
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'pending'

        queue_manager._set_system_state('training_in_progress', None)
        with patch('app.job_queue._submit', return_value='prompt-1'), \
             patch('app.job_queue._poll_outputs', return_value=('out.png', False)), \
             patch('app.job_queue._dispatch_completion'):
            assert queue_manager.process_one() is True
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'completed'


def test_process_one_skips_while_vision_in_progress(app):
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager.add_job(workflow_data={'1': {}})
        queue_manager._set_system_state('vision_in_progress', True, ttl_seconds=60)
        with patch('app.job_queue._submit') as submit:
            assert not queue_manager.process_one()
            submit.assert_not_called()


def test_cancel_during_submit_window_not_resurrected(app):
    """A cancel landing between _submit() returning and the sent_to_comfy write
    must not be overwritten back to sent_to_comfy, and must never be polled."""
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})

        def _submit_then_cancel(workflow, client_id):
            queue_manager.cancel_job(client_id)  # race: cancel lands mid-submit
            return 'prompt-1'

        with patch('app.job_queue._submit', side_effect=_submit_then_cancel), \
             patch('app.job_queue._poll_outputs') as poll, \
             patch('app.utils.comfyui.cancel_comfyui_prompt') as interrupt:
            assert queue_manager.process_one() is True
            poll.assert_not_called()
            interrupt.assert_called_once_with('prompt-1', jid)
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'cancelled'


def test_dispatch_completion_crash_marks_linked_row_failed(app):
    """A link-callback crash must not strand the row as 'pending' forever -
    _dispatch_completion's except branch marks it failed as a fallback."""
    from app.job_queue import queue_manager
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Crash', 'crash')
        jid = queue_manager.add_job(workflow_data={'1': {}},
                                    metadata={'model_name': 'klein_edit_dataset'})
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending', job_id=jid)
        svc.db.session.add(img)
        svc.db.session.commit()

        with patch('app.job_queue._submit', return_value='prompt-1'), \
             patch('app.job_queue._poll_outputs', return_value=('out.png', False)), \
             patch('app.services.face_dataset_service.link_completed_dataset_image',
                   side_effect=RuntimeError('boom')):
            queue_manager.process_one()

        row = FaceDatasetImage.query.filter_by(job_id=jid).one()
        assert row.status == 'failed'


def test_cancel_pending(app):
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        assert queue_manager.cancel_job(jid) is True
        assert ImageGenerationQueue.query.filter_by(job_id=jid).one().status == 'cancelled'


def test_cancel_nonexistent_job_returns_false(app):
    from app.job_queue import queue_manager
    with app.app_context():
        assert queue_manager.cancel_job('does-not-exist') is False


def test_cancel_already_completed_job_returns_false(app):
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        row.update_status('completed', result_filename='x.png')
        from app.extensions import db
        db.session.commit()
        assert queue_manager.cancel_job(jid) is False


def test_boot_recovery_fails_stuck_jobs_and_dispatches(app):
    """Rows stuck in processing/sent_to_comfy past the timeout must be marked
    failed and dispatched with failed=True at boot."""
    from datetime import datetime, timedelta
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    from app.extensions import db
    done = {}
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}},
                                    metadata={'model_name': 'klein_edit_dataset'})
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        row.update_status('sent_to_comfy')
        row.last_heartbeat = datetime.utcnow() - timedelta(minutes=11)
        db.session.commit()

        queue_manager.init_app(app)
        with patch('app.job_queue._dispatch_completion',
                   side_effect=lambda job, fn, failed: done.update(fn=fn, failed=failed)):
            queue_manager._recover_stuck_jobs()

        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'failed'
        assert done == {'fn': None, 'failed': True}


def test_boot_recovery_leaves_fresh_jobs_alone(app):
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        row.update_status('processing')  # fresh heartbeat, not stuck
        from app.extensions import db
        db.session.commit()

        queue_manager.init_app(app)
        with patch('app.job_queue._dispatch_completion') as dispatch:
            queue_manager._recover_stuck_jobs()
            dispatch.assert_not_called()

        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'processing'


def test_start_stop_idempotent_and_clean(app):
    """start() must be safe to call twice and stop() must leave no thread running."""
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager.init_app(app)
        with patch('app.job_queue.JobQueueManager._recover_stuck_jobs'):
            queue_manager.start()
            worker_thread = queue_manager._thread
            queue_manager.start()  # idempotent: no second thread, no crash
            assert queue_manager._thread is worker_thread
            assert worker_thread.is_alive()
            queue_manager.stop()
            assert queue_manager._thread is None
            assert not worker_thread.is_alive()


def test_claim_on_pending_returns_true_and_sets_status(app):
    """_claim on a pending row must atomically set status='processing' and heartbeat."""
    from app.job_queue import queue_manager, _claim
    from app.models import ImageGenerationQueue
    from datetime import datetime
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        assert _claim(jid) is True
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'processing'
        assert row.started_at is not None
        assert row.last_heartbeat is not None


def test_claim_on_cancelled_returns_false_stays_cancelled(app):
    """_claim on a row already cancelled must return False and NOT change status."""
    from app.job_queue import queue_manager, _claim
    from app.models import ImageGenerationQueue
    from app.extensions import db
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        row.update_status('cancelled')
        db.session.commit()

        assert _claim(jid) is False
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'cancelled'


def test_cancel_during_claim_race_guard(app):
    """Simulate the race: _claim on a job that was cancelled after SELECT but before claim.
    The atomic _claim must fail, returning False, preventing submission to ComfyUI."""
    from app.job_queue import queue_manager, _claim
    from app.models import ImageGenerationQueue
    from app.extensions import db
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}},
                                    metadata={'model_name': 'klein_edit_dataset'})
        # Simulate: we SELECT the job, then another thread cancels it
        queue_manager.cancel_job(jid)

        # Now _claim should fail because the job is no longer pending
        assert _claim(jid) is False

        # Job stays cancelled
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert row.status == 'cancelled'


def test_poll_outputs_skips_temp_images_returns_output_type(app):
    """Real ComfyUI history: a PreviewImage node emits type='temp' upstream of
    the real SaveImage node — must not be mistaken for the result."""
    from app.job_queue import _poll_outputs
    history = {
        'prompt-1': {
            'outputs': {
                '9': {'images': [{'filename': 'preview.png', 'subfolder': '', 'type': 'temp'}]},
                '13': {'images': [{'filename': 'final.png', 'subfolder': '', 'type': 'output'}]},
            },
            'status': {'status_str': 'success', 'completed': True},
        }
    }
    with app.app_context():
        with patch('app.utils.comfyui.get_comfyui_history', return_value=history):
            filename, failed = _poll_outputs('prompt-1', timeout=1)
    assert (filename, failed) == ('final.png', False)


def test_poll_outputs_fails_fast_on_comfyui_error_status(app):
    from app.job_queue import _poll_outputs
    history = {'prompt-1': {'outputs': {}, 'status': {'status_str': 'error', 'completed': True}}}
    with app.app_context():
        with patch('app.utils.comfyui.get_comfyui_history', return_value=history):
            filename, failed = _poll_outputs('prompt-1', timeout=1)
    assert (filename, failed) == (None, True)


def test_poll_outputs_completed_with_no_outputs_fails(app):
    from app.job_queue import _poll_outputs
    history = {'prompt-1': {'outputs': {}, 'status': {'status_str': 'success', 'completed': True}}}
    with app.app_context():
        with patch('app.utils.comfyui.get_comfyui_history', return_value=history):
            filename, failed = _poll_outputs('prompt-1', timeout=1)
    assert (filename, failed) == (None, True)


def test_poll_outputs_all_temp_images_keeps_polling_then_times_out(app):
    """If every image found is still 'temp' (no real SaveImage output yet) and
    the job hasn't reported completed/error, polling must continue and only
    fail once the timeout elapses — never mistake a temp image for the result."""
    from app.job_queue import _poll_outputs
    history = {'prompt-1': {'outputs': {'9': {'images': [{'filename': 'p.png', 'type': 'temp'}]}},
                            'status': {}}}
    with app.app_context():
        with patch('app.utils.comfyui.get_comfyui_history', return_value=history), \
             patch('app.job_queue.POLL_INTERVAL_SECONDS', 0.01):
            filename, failed = _poll_outputs('prompt-1', timeout=0.05)
    assert (filename, failed) == (None, True)


def test_cancel_during_poll_exits_fast_and_interrupts_exact_prompt(app):
    """Stop must wake the history poll immediately and request a targeted
    ComfyUI cancellation instead of waiting for the next 2-second tick/timeout."""
    from app.job_queue import queue_manager, _poll_outputs
    from app.models import ImageGenerationQueue
    from app.extensions import db
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        row.update_status('sent_to_comfy', comfyui_prompt_id='prompt-cancel')
        db.session.commit()

        history_calls = 0

        def cancel_while_polling(_prompt_id):
            nonlocal history_calls
            history_calls += 1
            assert queue_manager.cancel_job(jid) is True
            return {}

        started = time.monotonic()
        with patch('app.utils.comfyui.get_comfyui_history', side_effect=cancel_while_polling), \
             patch('app.utils.comfyui.cancel_comfyui_prompt', return_value=True) as interrupt, \
             patch('app.job_queue.POLL_INTERVAL_SECONDS', 30):
            assert _poll_outputs('prompt-cancel', timeout=60) == (None, True)

        assert time.monotonic() - started < 1
        assert history_calls == 1
        interrupt.assert_called_once_with('prompt-cancel', jid)
        assert ImageGenerationQueue.query.filter_by(job_id=jid).one().status == 'cancelled'


def test_concurrent_expired_delete_guard(app):
    """_get_system_state on an expired row must survive losing a delete race:
    if a concurrent reader already removed the row, the flush raises
    (StaleDataError: 0 rows matched) and the guard must catch it, rollback,
    and return the default instead of crashing.

    Deterministic version of a real-threads race (see git history) that hit
    the guard reliably in isolation but flaked under full-suite scheduling.
    Here the conflict is injected directly instead of hoped for."""
    from app.job_queue import queue_manager
    from app.models import SystemState
    from app.extensions import db
    from sqlalchemy.orm.exc import StaleDataError
    from unittest.mock import patch

    with app.app_context():
        queue_manager._set_system_state('flag', True, ttl_seconds=-1)  # already expired, no sleep needed

        # Simulate a concurrent deleter having won the race: our own
        # commit() hits a 0-row DELETE and must recover, not raise.
        with patch.object(db.session, 'commit',
                           side_effect=StaleDataError(
                               "DELETE statement on table 'system_state' expected to "
                               "delete 1 row(s); 0 were matched.")):
            assert queue_manager._get_system_state('flag') is None

        # The failed commit was rolled back cleanly: a normal (unpatched) call
        # can still read and delete the row for real afterwards.
        assert queue_manager._get_system_state('flag') is None
        assert db.session.get(SystemState, 'flag') is None


# --- _submit unpacks the REAL queue_prompt_to_comfyui contract -------------
# The tests above all patch `_submit` with a bare string, so none of them
# exercise its actual body. queue_prompt_to_comfyui NEVER raises: it returns
# (response.json(), None) on success or (None, error) on failure. _submit must
# unpack that -- binding the raw tuple into the comfyui_prompt_id String column
# is a ProgrammingError that fails every real ComfyUI job.

def test_submit_unpacks_prompt_id_from_real_contract(app):
    from app.job_queue import _submit
    with app.app_context():
        with patch('app.utils.comfyui.queue_prompt_to_comfyui',
                   return_value=({'prompt_id': 'p1', 'number': 1, 'node_errors': {}}, None)):
            result = _submit({'1': {}}, 'client-1')
    assert result == 'p1'


def test_submit_raises_on_comfyui_error(app):
    from app.job_queue import _submit
    with app.app_context():
        with patch('app.utils.comfyui.queue_prompt_to_comfyui',
                   return_value=(None, 'WORKFLOW_INVALIDE (validation ComfyUI 400): bad')):
            with pytest.raises(RuntimeError, match='WORKFLOW_INVALIDE'):
                _submit({'1': {}}, 'client-1')


def test_process_one_completes_with_real_submit_contract(app):
    """End-to-end through the REAL _submit -> queue_prompt_to_comfyui contract
    (not the mocked-_submit shortcut the other tests use): a successful submit
    must advance to sent_to_comfy with a STRING prompt_id and complete."""
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
    with patch('app.utils.comfyui.queue_prompt_to_comfyui',
               return_value=({'prompt_id': 'p1'}, None)), \
         patch('app.job_queue._poll_outputs', return_value=('out.png', False)), \
         patch('app.job_queue._dispatch_completion'):
        with app.app_context():
            assert queue_manager.process_one() is True
            row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
            assert row.status == 'completed'
            assert row.comfyui_prompt_id == 'p1'


def test_poll_outputs_stashes_execution_error_on_job_row(app):
    """A ComfyUI runtime failure (e.g. wrong text encoder -> KSampler matmul
    error) must land on the job row's error_message so the failed tile can show
    WHY — the live repro was 'mat1 and mat2 shapes cannot be multiplied'."""
    from app.job_queue import queue_manager, _poll_outputs
    from app.models import ImageGenerationQueue
    history = {'prompt-err': {'outputs': {}, 'status': {
        'status_str': 'error', 'completed': False,
        'messages': [['execution_start', {}],
                     ['execution_error', {'node_id': '77', 'node_type': 'KSampler',
                                          'exception_message': 'mat1 and mat2 shapes cannot be multiplied (512x7680 and 12288x4096)\n\nTIPS: ...'}]],
    }}}
    with app.app_context():
        jid = queue_manager.add_job(workflow_data={'1': {}})
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        row.comfyui_prompt_id = 'prompt-err'
        from app.extensions import db
        db.session.commit()
        with patch('app.utils.comfyui.get_comfyui_history', return_value=history):
            filename, failed = _poll_outputs('prompt-err', timeout=1)
        assert (filename, failed) == (None, True)   # 2-tuple contract unchanged
        row = ImageGenerationQueue.query.filter_by(job_id=jid).one()
        assert 'KSampler' in row.error_message
        assert 'mat1 and mat2' in row.error_message


def test_failed_job_reason_reaches_dataset_tile(app):
    """process_one end-to-end on a runtime failure: the execution error stashed
    by the poll must flow through _dispatch_completion into the dataset row's
    fail_reason (not the generic 'see the server log' message)."""
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.extensions import db

    def fake_poll(prompt_id, timeout=None):
        job = ImageGenerationQueue.query.filter_by(comfyui_prompt_id=prompt_id).first()
        job.error_message = 'ComfyUI KSampler: mat1 and mat2 shapes cannot be multiplied'
        db.session.commit()
        return None, True

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'ErrProp', 'errprop')
        jid = queue_manager.add_job(workflow_data={'1': {}},
                                    metadata={'model_name': 'klein_edit_dataset'})
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending',
                               job_id=jid, klein_model='k.safetensors')
        db.session.add(img)
        db.session.commit()
        with patch('app.job_queue._submit', return_value='prompt-err2'), \
             patch('app.job_queue._poll_outputs', side_effect=fake_poll):
            assert queue_manager.process_one() is True
        refreshed = db.session.get(FaceDatasetImage, img.id)
        assert refreshed.status == 'failed'
        assert 'mat1 and mat2' in refreshed.fail_reason
