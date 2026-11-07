"""A queued local training waits for a busy GPU instead of being dropped.

The queue starts its next item through the same spawn transaction as a direct
launch, and that transaction refuses while a vision pass, a ComfyUI generation
or a ComfyUI recovery holds the card, or while the dataset is busy with a pass
of its own. The queue only looked at the vision FLAG, launched anyway, and on
the refusal removed the item: the training the user had queued was gone, with
an error nobody displayed. It now asks first, keeps an item refused for a
passing reason, and says what it is waiting for.

No real training: the launch is a stub, the owners are the arbiter's flags,
queue rows and activity registry.
"""
import pytest

ITEM = {'dataset_id': 4243, 'user_id': 'local', 'extra_steps': None}


def _queue(app, *items):
    from app.services import lora_training as lt
    with app.app_context():
        lt._save_queue([dict(item) for item in items])


def test_the_queue_waits_for_comfyui_work_then_starts_the_same_item(app, monkeypatch):
    from app.extensions import db
    from app.models import ImageGenerationQueue
    from app.services import lora_training as lt
    launched = []
    monkeypatch.setattr(lt, '_launch_queued_item', launched.append)
    _queue(app, ITEM)
    with app.app_context():
        db.session.add(ImageGenerationQueue(job_id='render-1', status='pending'))
        db.session.commit()
        assert lt._advance_training_queue() is None and launched == []
        assert len(lt.get_train_queue()) == 1, 'the item must keep its place'
        assert lt.training_status('local')['queue_waiting_for'] == 'an image generation in ComfyUI'
        ImageGenerationQueue.query.filter_by(job_id='render-1').delete()
        db.session.commit()
        assert lt._advance_training_queue() == 'launched:4243'
        assert [item['dataset_id'] for item in launched] == [4243]
        assert lt.get_train_queue() == []


def test_the_inprocess_vision_window_holds_the_queue(app, monkeypatch):
    from app import gpu_window
    from app.job_queue import GPU_ARBITER_LOCK
    from app.services import lora_training as lt
    monkeypatch.setattr(lt, '_launch_queued_item',
                        lambda item: pytest.fail('launched beside a live vision window'))
    _queue(app, ITEM)
    with GPU_ARBITER_LOCK:
        gpu_window._active_vision_window_tokens.add('live-worker-token')
    try:
        with app.app_context():
            assert lt._advance_training_queue() is None
            assert len(lt.get_train_queue()) == 1
            assert lt.training_status('local')['queue_waiting_for'] == 'a vision pass'
    finally:
        with GPU_ARBITER_LOCK:
            gpu_window._active_vision_window_tokens.discard('live-worker-token')
        _queue(app)


def test_a_passing_gpu_refusal_at_launch_keeps_the_item(app, monkeypatch):
    from app.gpu_window import GpuBusyError
    from app.services import lora_training as lt

    def refused(_item):
        raise GpuBusyError('ComfyUI has queued or active work', hold='generation')

    monkeypatch.setattr(lt, '_launch_queued_item', refused)
    _queue(app, ITEM)
    with app.app_context():
        assert lt._advance_training_queue() is None
        assert len(lt.get_train_queue()) == 1
        assert lt.queue_manager._get_system_state('training_queue_error') is None


def test_a_finished_run_hands_the_gpu_over_when_the_next_item_must_wait(app, monkeypatch):
    """The run before it ended (dead pid): the next item is refused for a passing
    reason. It keeps its place AND the dead run's fence is released, or the
    owner in the way could never finish."""
    from app.gpu_window import GpuBusyError
    from app.job_queue import queue_manager
    from app.services import lora_training as lt

    def refused(_item):
        raise GpuBusyError('a vision pass is using the GPU', hold='vision')

    monkeypatch.setattr(lt, '_launch_queued_item', refused)
    monkeypatch.setattr(lt, '_pid_alive', lambda pid: False)
    _queue(app, ITEM)
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=600)
        queue_manager._set_system_state('training_pid', 4242, ttl_seconds=600)
        assert lt._advance_training_queue() == 'released'
        assert len(lt.get_train_queue()) == 1
        assert not queue_manager._get_system_state('training_in_progress', False)
        _queue(app)


def test_a_busy_dataset_waits_without_holding_up_the_items_behind_it(app, monkeypatch):
    from app.services import dataset_activity
    from app.services import lora_training as lt
    launched = []
    monkeypatch.setattr(lt, '_launch_queued_item', launched.append)
    second = {**ITEM, 'dataset_id': 4244}
    _queue(app, ITEM, second)
    token = dataset_activity.begin(4243, 'caption', detail='captioning')
    try:
        with app.app_context():
            assert lt._advance_training_queue() == 'launched:4244'
            assert [item['dataset_id'] for item in lt.get_train_queue()] == [4243]
            assert lt.training_status('local')['queue_waiting_for'] == \
                'its dataset to finish the work it is doing'
    finally:
        dataset_activity.end(token)
        _queue(app)
