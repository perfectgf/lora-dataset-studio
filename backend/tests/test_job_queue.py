import json, time
from unittest.mock import patch


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


def test_concurrent_expired_delete_guard(app):
    """Concurrent _get_system_state on an expired row must not double-delete and crash."""
    from app.job_queue import queue_manager
    from app.models import SystemState
    from app.extensions import db
    from unittest.mock import MagicMock, patch
    import threading

    with app.app_context():
        queue_manager._set_system_state('flag', True, ttl_seconds=1)
        time.sleep(1.2)

        # Both threads will try to delete the expired row concurrently.
        # The second should catch the exception and rollback gracefully.
        results = []

        def read_and_delete():
            with app.app_context():
                val = queue_manager._get_system_state('flag')
                results.append(val)

        t1 = threading.Thread(target=read_and_delete)
        t2 = threading.Thread(target=read_and_delete)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should return None (default) without crashing
        assert results == [None, None]
        # Row should be deleted only once
        assert db.session.get(SystemState, 'flag') is None
