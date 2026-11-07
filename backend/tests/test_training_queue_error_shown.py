"""A queued training that fails to start is reported, never lost in silence.

The queue removes an item whose launch fails (so it cannot retry forever) and
records why in `training_queue_error` — which nothing ever read: the training
the user had queued simply vanished. The training status now carries it, and
the Training panel shows it.
"""


def test_a_queued_training_that_fails_to_start_is_reported(app, monkeypatch):
    from app.services import lora_training as lt

    def broken(_item):
        raise ValueError('custom base not converted')

    monkeypatch.setattr(lt, '_launch_queued_item', broken)
    with app.app_context():
        assert lt.training_status('local')['queue_error'] is None
        lt._save_queue([{'dataset_id': 4243, 'user_id': 'local', 'extra_steps': None}])
        assert lt._advance_training_queue() is None
        assert lt.get_train_queue() == []
        assert lt.training_status('local')['queue_error'] == {
            'dataset_id': 4243, 'name': '#4243', 'error': 'custom base not converted'}
