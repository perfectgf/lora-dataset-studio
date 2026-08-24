import uuid
from app.extensions import db

import pytest


_CONFIRM = {'confirmed_comfyui_restart': True}


def _stalled_studio_cell(app, *, run_id='recovery-run', known_prompt=False):
    """Create one linked Test Studio cell behind one durable barrier."""
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import LoraTestImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'Recovery', 'recovery')
        job_id = str(uuid.uuid4())
        cell = LoraTestImage(
            dataset_id=dataset.id,
            run_id=run_id,
            checkpoint='z image\\recovery.safetensors',
            strength=1.0,
            status='pending',
            job_id=job_id,
        )
        db.session.add(cell)
        db.session.flush()
        queue_manager.add_job(
            workflow_data={'1': {}},
            prompt='recovery prompt',
            job_id=job_id,
            user_id=str(LOCAL_USER),
            metadata={
                'model_name': 'zimage_lora_test',
                'is_lora_test': True,
                'dataset_id': dataset.id,
                'cell_id': cell.id,
                'run_id': run_id,
            },
            commit=False,
        )
        db.session.commit()
        if known_prompt:
            assert queue_manager._stall_comfy_job(
                job_id, 'known-prompt', allowed_statuses=('pending',))
        else:
            assert queue_manager._stall_unknown_comfy_job(
                job_id, allowed_statuses=('pending',))
        return dataset.id, cell.id, job_id


def _stalled_dataset_generation(app):
    """Create one ordinary dataset card behind an unknown-submit barrier."""
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.job_queue import queue_manager
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'Generate recovery', 'generate-recovery')
        job_id = str(uuid.uuid4())
        card = FaceDatasetImage(
            dataset_id=dataset.id, source='generated', status='pending',
            job_id=job_id)
        db.session.add(card)
        db.session.flush()
        queue_manager.add_job(
            workflow_data={'1': {}}, prompt='generation recovery prompt',
            job_id=job_id, user_id=str(LOCAL_USER),
            metadata={'model_name': 'klein_edit_dataset',
                      'dataset_id': dataset.id, 'image_id': card.id},
            commit=False)
        db.session.commit()
        assert queue_manager._stall_unknown_comfy_job(
            job_id, allowed_statuses=('pending',))
        return dataset.id, card.id, job_id


def _assert_still_stalled(app, cell_id, job_id, *, prompt_id=None):
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage

    with app.app_context():
        cell = db.session.get(LoraTestImage, cell_id)
        queue = ImageGenerationQueue.query.filter_by(job_id=job_id).one()
        assert cell.status == 'pending'
        assert cell.job_id == job_id
        assert queue.status == 'stalled'
        assert queue.comfyui_prompt_id == prompt_id
        assert queue_manager.get_comfyui_stalled_barrier() is not None


def test_unknown_submit_confirmation_cancels_exact_cell_barrier_and_job(app):
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.services import lora_test_studio as lts

    dataset_id, cell_id, job_id = _stalled_studio_cell(app)
    with app.app_context():
        payload = lts.studio_payload(LOCAL_USER, dataset_id)
        assert payload['comfyui_recovery'] == {
            'required': True,
            'kind': 'unknown_submit',
            'job_id': job_id,
            'cell_id': cell_id,
            'requires_comfyui_restart_confirmation': True,
        }

        assert lts.confirm_unknown_comfyui_restart(
            LOCAL_USER, dataset_id=dataset_id, restart_confirmed=True) == 1

        queue = ImageGenerationQueue.query.filter_by(job_id=job_id).one()
        cell = db.session.get(LoraTestImage, cell_id)
        assert queue.status == 'cancelled'
        assert queue.comfyui_prompt_id is None
        assert cell.status == 'cancelled'
        assert cell.job_id is None
        assert queue_manager.get_comfyui_stalled_barrier() is None


def test_dataset_stop_names_unknown_submit_and_confirm_route_recovers(
        client, monkeypatch):
    from app.job_queue import queue_manager
    from app.models import FaceDatasetImage, ImageGenerationQueue

    dataset_id, card_id, job_id = _stalled_dataset_generation(client.application)
    stopped = client.post(f'/api/dataset/{dataset_id}/cancel')
    assert stopped.status_code == 200
    assert stopped.get_json() == {
        'ok': True, 'cancelled': 0, 'recovery_pending': 1,
        'retry_pending': 0, 'restart_required': 1, 'recovery_error': 0,
    }
    with client.application.app_context():
        assert db.session.get(FaceDatasetImage, card_id).job_id == job_id
        assert queue_manager.get_comfyui_stalled_barrier() is not None

    missing = client.post(
        f'/api/dataset/{dataset_id}/confirm-comfyui-restart', json={})
    assert missing.status_code == 400

    monkeypatch.setattr(
        'app.capabilities.probe',
        lambda *, force=False: {'comfyui': {'reachable': bool(force)}})
    recovered = client.post(
        f'/api/dataset/{dataset_id}/confirm-comfyui-restart', json=_CONFIRM)
    assert recovered.status_code == 200
    assert recovered.get_json() == {'ok': True, 'cancelled': 1}
    with client.application.app_context():
        assert db.session.get(FaceDatasetImage, card_id) is None
        assert ImageGenerationQueue.query.filter_by(job_id=job_id).one().status == 'cancelled'
        assert queue_manager.get_comfyui_stalled_barrier() is None


def test_unknown_submit_recovery_requires_explicit_confirmation(app):
    from app.config import LOCAL_USER
    from app.services import lora_test_studio as lts

    dataset_id, cell_id, job_id = _stalled_studio_cell(app)
    with app.app_context(), pytest.raises(ValueError, match='Confirm that you restarted ComfyUI'):
        lts.confirm_unknown_comfyui_restart(LOCAL_USER, dataset_id=dataset_id)

    _assert_still_stalled(app, cell_id, job_id)


def test_unknown_submit_recovery_never_clears_a_known_prompt_barrier(app):
    from app.config import LOCAL_USER
    from app.services import lora_test_studio as lts

    dataset_id, cell_id, job_id = _stalled_studio_cell(app, known_prompt=True)
    with app.app_context(), pytest.raises(RuntimeError, match='no unknown ComfyUI submission'):
        lts.confirm_unknown_comfyui_restart(
            LOCAL_USER, dataset_id=dataset_id, restart_confirmed=True)

    _assert_still_stalled(app, cell_id, job_id, prompt_id='known-prompt')


def test_unknown_submit_recovery_is_scoped_to_exact_run(app):
    from app.config import LOCAL_USER
    from app.services import lora_test_studio as lts

    _dataset_id, cell_id, job_id = _stalled_studio_cell(app, run_id='right-run')
    with app.app_context(), pytest.raises(RuntimeError, match='does not belong'):
        lts.confirm_unknown_comfyui_restart(
            LOCAL_USER, run_id='wrong-run', restart_confirmed=True)

    _assert_still_stalled(app, cell_id, job_id)


def test_unknown_submit_recovery_rolls_back_job_barrier_and_cell_on_commit_failure(app, monkeypatch):
    from app.config import LOCAL_USER
    from app.extensions import db
    from app.services import lora_test_studio as lts

    dataset_id, cell_id, job_id = _stalled_studio_cell(app)
    with app.app_context():
        real_commit = db.session.commit

        def fail_commit():
            raise RuntimeError('simulated commit failure')

        monkeypatch.setattr(db.session, 'commit', fail_commit)
        try:
            with pytest.raises(RuntimeError, match='Could not record the ComfyUI recovery'):
                lts.confirm_unknown_comfyui_restart(
                    LOCAL_USER, dataset_id=dataset_id, restart_confirmed=True)
        finally:
            monkeypatch.setattr(db.session, 'commit', real_commit)

    _assert_still_stalled(app, cell_id, job_id)


def test_run_recovery_route_requires_confirmed_fresh_comfyui_and_clears_only_after_gate(client, monkeypatch):
    dataset_id, cell_id, job_id = _stalled_studio_cell(client.application, run_id='route-run')
    calls = []

    def probe(*, force=False):
        calls.append(force)
        return {'comfyui': {'reachable': True}}

    monkeypatch.setattr('app.capabilities.probe', probe)

    missing = client.post('/api/studio/run/route-run/confirm-comfyui-restart', json={})
    assert missing.status_code == 400
    assert calls == []
    _assert_still_stalled(client.application, cell_id, job_id)

    response = client.post('/api/studio/run/route-run/confirm-comfyui-restart', json=_CONFIRM)
    assert response.status_code == 200
    assert response.get_json() == {'ok': True, 'cancelled': 1, 'resumable': True}
    assert calls == [True]

    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage
    with client.application.app_context():
        assert ImageGenerationQueue.query.filter_by(job_id=job_id).one().status == 'cancelled'
        cell = db.session.get(LoraTestImage, cell_id)
        assert cell.status == 'cancelled' and cell.job_id is None
        assert queue_manager.get_comfyui_stalled_barrier() is None
        assert dataset_id == cell.dataset_id


def test_dataset_recovery_route_refuses_to_clear_while_comfyui_is_not_ready(client, monkeypatch):
    dataset_id, cell_id, job_id = _stalled_studio_cell(client.application)

    def probe(*, force=False):
        assert force is True
        return {'comfyui': {'reachable': False, 'hint': 'restart it'}}

    monkeypatch.setattr('app.capabilities.probe', probe)
    response = client.post(
        f'/api/dataset/{dataset_id}/lora-test/confirm-comfyui-restart',
        json=_CONFIRM,
    )
    assert response.status_code == 409
    _assert_still_stalled(client.application, cell_id, job_id)
