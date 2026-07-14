import io

import pytest
from PIL import Image


def _png(color=(255, 0, 0), size=(64, 64)):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger})


# --- gpu_window ---------------------------------------------------------
def test_nested_window_raises_gpu_busy_and_clears_flag_on_exit(app, monkeypatch):
    from app.gpu_window import gpu_exclusive_vision_window, GpuBusyError
    from app.job_queue import queue_manager
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', lambda *a, **k: True)
    with app.app_context():
        with gpu_exclusive_vision_window():
            assert queue_manager._get_system_state('vision_in_progress')  # flag is truthy (token string)
            with pytest.raises(GpuBusyError):
                with gpu_exclusive_vision_window():
                    pass
        assert queue_manager._get_system_state('vision_in_progress') is None


def test_flag_cleared_when_body_raises(app, monkeypatch):
    from app.gpu_window import gpu_exclusive_vision_window
    from app.job_queue import queue_manager
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', lambda *a, **k: True)
    with app.app_context():
        with pytest.raises(ValueError):
            with gpu_exclusive_vision_window():
                raise ValueError('boom')
        assert queue_manager._get_system_state('vision_in_progress') is None


def test_window_blocked_while_training_in_progress(app):
    from app.gpu_window import gpu_exclusive_vision_window, GpuBusyError
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True)
        with pytest.raises(GpuBusyError):
            with gpu_exclusive_vision_window():
                pass
        # the window raised BEFORE setting its own flag -- must not have set it
        assert queue_manager._get_system_state('vision_in_progress') is None


def test_free_comfyui_vram_called_best_effort_and_exception_swallowed(app, monkeypatch):
    from app.gpu_window import gpu_exclusive_vision_window
    calls = []

    def _raise(*a, **k):
        calls.append(True)
        raise RuntimeError('comfyui unreachable')

    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', _raise)
    with app.app_context():
        with gpu_exclusive_vision_window():
            pass  # must not raise even though free_comfyui_vram blew up
    assert calls, 'free_comfyui_vram should have been called'


def test_window_ownership_prevents_flag_stomp_on_re_acquisition(app, monkeypatch):
    from app.gpu_window import gpu_exclusive_vision_window
    from app.job_queue import queue_manager
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', lambda *a, **k: True)
    with app.app_context():
        with gpu_exclusive_vision_window():
            # Simulate flag expiry + re-acquisition by a different caller
            queue_manager._set_system_state('vision_in_progress', 'someone-else-token')
        # After exiting the window, the flag must still belong to the re-acquirer
        # (the exited window must not stomp it with None)
        assert queue_manager._get_system_state('vision_in_progress') == 'someone-else-token'
        # clean up for test isolation
        queue_manager._set_system_state('vision_in_progress', None)


def test_boot_recovery_clears_persisted_vision_lock(app):
    """A request dies with the server process, but its DB flag survives unless
    startup explicitly removes it. Restarting must never strand "GPU busy"."""
    from app.gpu_window import recover_stale_vision_window
    from app.job_queue import queue_manager

    with app.app_context():
        queue_manager._set_system_state('vision_in_progress', 'dead-process-token',
                                        ttl_seconds=1800)
        assert recover_stale_vision_window() is True
        assert queue_manager._get_system_state('vision_in_progress') is None
        assert recover_stale_vision_window() is False


# --- import_images(crop=True) head-crop ---------------------------------
def test_import_images_crop_true_produces_square_output(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER

    # bbox in the 0-1000 normalized space expected by detect_head_bbox
    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: '{"x1": 100, "y1": 100, "x2": 400, "y2": 400}')
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'H', 'h')
        raw = _png(size=(400, 600))  # portrait -- proves a real crop happened
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [raw], crop=True)
        assert len(ids) == 1 and failed == 0
        img = svc.db.session.get(FaceDatasetImage, ids[0])
        assert img.framing == 'face'
        with Image.open(svc._img_path(img)) as out:
            w, h = out.size
            assert w == h


# --- classify_images ------------------------------------------------------
def test_classify_images_sets_framing_from_vision(app, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: '{"framing": "body", "angle": "3/4", "expression": "smile"}')
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'I', 'i')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png(), _png((0, 255, 0))], crop=False)
        assert failed == 0
        n = svc.classify_images(LOCAL_USER, ds.id)
        assert n == 2
        rows = FaceDatasetImage.query.filter(FaceDatasetImage.id.in_(ids)).all()
        assert all(r.framing == 'body' for r in rows)
        assert all(r.variation_label == '3/4, smile' for r in rows)


# --- route-level: 503 while the vision window is held ----------------------
def test_classify_route_returns_503_while_vision_flag_set(client, app):
    from app.job_queue import queue_manager
    ds_id = _create(client, 'Kai', 'kai').get_json()['id']
    with app.app_context():
        queue_manager._set_system_state('vision_in_progress', True, ttl_seconds=300)
    resp = client.post(f'/api/dataset/{ds_id}/classify')
    assert resp.status_code == 503
    body = resp.get_json()
    assert 'GPU busy' in body['error']
