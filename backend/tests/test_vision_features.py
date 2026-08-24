import io

import pytest
from PIL import Image


def _png(color=(255, 0, 0), size=(64, 64)):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


def _vram_freed(*_args, **_kwargs):
    from app.utils.comfyui import ComfyVramFreeVerdict
    return ComfyVramFreeVerdict.FREED


def _create(client, name='Lola', trigger='lola'):
    return client.post('/api/dataset/create', json={'name': name, 'trigger_word': trigger})


# --- gpu_window ---------------------------------------------------------
def test_nested_window_raises_gpu_busy_and_clears_flag_on_exit(app, monkeypatch):
    from app.gpu_window import gpu_exclusive_vision_window, GpuBusyError
    from app.job_queue import queue_manager
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', _vram_freed)
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
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', _vram_freed)
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


def test_window_refuses_when_comfyui_vram_release_is_unconfirmed(app, monkeypatch):
    from app.gpu_window import (GpuBusyError, gpu_exclusive_vision_window,
                                vision_gpu_window_blocks_gpu)
    from app.job_queue import queue_manager
    calls = []

    def _raise(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError('comfyui unreachable')

    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', _raise)
    with app.app_context():
        with pytest.raises(GpuBusyError, match='did not confirm'):
            with gpu_exclusive_vision_window():
                pass
        assert queue_manager._get_system_state('vision_in_progress') is None
    assert calls, 'free_comfyui_vram should have been called'
    assert vision_gpu_window_blocks_gpu() is False


def test_window_ownership_prevents_flag_stomp_on_re_acquisition(app, monkeypatch):
    from app.gpu_window import gpu_exclusive_vision_window
    from app.job_queue import queue_manager
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', _vram_freed)
    with app.app_context():
        with gpu_exclusive_vision_window():
            # Simulate flag expiry + re-acquisition by a different caller
            queue_manager._set_system_state('vision_in_progress', 'someone-else-token')
        # After exiting the window, the flag must still belong to the re-acquirer
        # (the exited window must not stomp it with None)
        assert queue_manager._get_system_state('vision_in_progress') == 'someone-else-token'
        # clean up for test isolation
        queue_manager._set_system_state('vision_in_progress', None)


def test_teardown_releases_inprocess_fence_when_db_cleanup_raises(app, monkeypatch):
    import app.gpu_window as gw

    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram', _vram_freed)
    with app.app_context():
        with pytest.raises(RuntimeError, match='db cleanup'):
            with gw.gpu_exclusive_vision_window():
                monkeypatch.setattr(
                    gw, '_in_app_context',
                    lambda _callback: (_ for _ in ()).throw(RuntimeError('db cleanup')))
    assert gw.vision_gpu_window_blocks_gpu() is False


def _configure_aitoolkit(tmp_path, app):
    """Minimal fake ai-toolkit install, so launch_training gets past is_installed()
    and reaches the GPU guards we actually want to exercise."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


def test_direct_training_launch_refused_while_a_vision_pass_holds_the_gpu(
        app, tmp_path, monkeypatch):
    """The reciprocal half of the GPU-exclusive window.

    The window refuses to OPEN while a training runs. The mirror image — a
    training refusing to START while a vision pass holds the card — existed only
    on the QUEUE path (_advance_training_queue skips a due item while
    vision_in_progress); a direct launch walked straight through.

    That gap is not theoretical. Measured on a 24 GB card: with ~19 GB held by
    ComfyUI, Ollama cannot fit the 7.5 GB vision model and silently spills 43 %
    of it to the CPU — the vision pass runs 13.5x slower and the resident GPU
    work collapses 20-150x. Nothing OOMs, so nothing reports a failure: a
    training started here would just crawl for hours. Refuse it instead.
    """
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.gpu_window import GpuBusyError
    from app.job_queue import queue_manager
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'GW', 'zchar_gw')
        queue_manager._set_system_state('vision_in_progress', 'tok', ttl_seconds=300)
        try:
            with pytest.raises(GpuBusyError, match='vision pass'):
                lt.launch_training(LOCAL_USER, ds.id, check_captions=False)
            # ...but QUEUING stays legal: the queue exists precisely to hold work
            # until the card is free, and _advance_training_queue already refuses
            # to launch a due item while vision_in_progress. Refusing the enqueue
            # too would remove the only graceful way out the error message offers.
            assert lt.enqueue_training(LOCAL_USER, ds.id,
                                       extra_steps=100)['queued'] is True
        finally:
            queue_manager._set_system_state('vision_in_progress', None)
            lt.dequeue_training(ds.id)


def test_direct_training_refuses_a_live_inprocess_vision_fence_before_spawn(
        app, tmp_path, monkeypatch):
    from app import gpu_window
    from app.config import LOCAL_USER
    from app.gpu_window import GpuBusyError
    from app.job_queue import GPU_ARBITER_LOCK, queue_manager
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt, 'assert_interpreter_ready', lambda: None)
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda _path: type('usage', (), {'free': 500e9})())
    monkeypatch.setattr(lt.subprocess, 'Popen',
                        lambda *_args, **_kwargs: pytest.fail('training must not spawn'))
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Live Vision', 'live_vision')
        queue_manager._set_system_state('vision_in_progress', None)
        with GPU_ARBITER_LOCK:
            gpu_window._active_vision_window_tokens.add('live-worker-token')
        try:
            with pytest.raises(GpuBusyError, match='vision pass'):
                lt.launch_training(LOCAL_USER, ds.id, check_captions=False)
        finally:
            with GPU_ARBITER_LOCK:
                gpu_window._active_vision_window_tokens.discard('live-worker-token')


def test_training_launch_not_blocked_once_the_vision_pass_released(
        app, tmp_path, monkeypatch):
    """Non-regression: no vision flag (or an expired one) must never be the thing
    that refuses a launch — the guard is a lock, not a latch."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.gpu_window import GpuBusyError
    from app.job_queue import queue_manager
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'GW2', 'zchar_gw2')
        queue_manager._set_system_state('vision_in_progress', None)
        try:
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False)
        except GpuBusyError:                       # pragma: no cover - the bug
            pytest.fail('a released vision window must not block a launch')
        except Exception:
            pass  # any later failure (no real ai-toolkit to spawn) is fine here


def test_queue_advance_leaves_a_due_item_queued_while_a_vision_pass_runs(
        app, monkeypatch):
    """The other half of the same guarantee, on the queue path: a due item is
    NOT launched while the window is held, and — critically — it stays in the
    queue rather than being consumed, so it still runs once the pass ends."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    launched = []
    monkeypatch.setattr(lt, '_launch_queued_item',
                        lambda item: launched.append(item))
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', None)
        lt._save_queue([{'dataset_id': 4242, 'user_id': 'local', 'extra_steps': None}])
        try:
            queue_manager._set_system_state('vision_in_progress', 'tok', ttl_seconds=300)
            assert lt._advance_training_queue() is None
            assert launched == [], 'a vision pass must hold the training queue'
            assert len(lt.get_train_queue()) == 1, 'the item must not be consumed'
            # window released -> the SAME item goes, nothing was lost
            queue_manager._set_system_state('vision_in_progress', None)
            assert lt._advance_training_queue() == 'launched:4242'
            assert [i['dataset_id'] for i in launched] == [4242]
        finally:
            queue_manager._set_system_state('vision_in_progress', None)
            lt._save_queue([])


def test_train_route_answers_503_gpu_busy_during_a_vision_pass(
        app, client, tmp_path, monkeypatch):
    """User-visible contract: the ▶ button gets the same 503 'GPU busy' the
    vision routes return when training holds the card — symmetric, and never a
    silent launch that would crawl behind the pass."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.job_queue import queue_manager
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'GW3', 'zchar_gw3')
        ds_id = ds.id
        queue_manager._set_system_state('vision_in_progress', 'tok', ttl_seconds=300)
    try:
        r = client.post(f'/api/dataset/{ds_id}/train', json={})
        assert r.status_code == 503, r.get_data(as_text=True)
        body = r.get_json()
        assert body['error'] == 'GPU busy'
        assert 'vision pass' in body['detail']
    finally:
        with app.app_context():
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
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', _vram_freed)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'I', 'i')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png(), _png((0, 255, 0))], crop=False)
        assert failed == 0
        n = svc.classify_images(LOCAL_USER, ds.id)
        assert n == 2
        rows = FaceDatasetImage.query.filter(FaceDatasetImage.id.in_(ids)).all()
        assert all(r.framing == 'body' for r in rows)
        assert all(r.variation_label == '3/4, smile' for r in rows)


def test_classify_images_forces_json_format(app, monkeypatch):
    """Same grammar pin as Bank framing and head-crop: without fmt='json' a
    reasoning model rambles past the token budget and the pass reports 0
    classified while blaming Ollama."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.config import LOCAL_USER

    seen = {}

    def fake(_bytes, _prompt, **kw):
        seen.update(kw)
        return '{"framing": "bust"}'

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', fake)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', _vram_freed)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'J', 'j')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
        assert failed == 0
        assert svc.classify_images(LOCAL_USER, ds.id) == 1
        assert seen.get('fmt') == 'json'
        assert seen.get('prefer_json') is True


def test_classify_images_skips_already_tagged_unless_forced(app, monkeypatch):
    """Default fill leaves a known shot type alone; force overwrites it.

    After a crop the pixels changed and the stored framing is stale — that is
    the Bank's framing ``rescan``, ported to the dataset.
    """
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db

    answers = ['{"framing": "body"}']

    def _describe(*_a, **_k):
        return answers[0]

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama', _describe)
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', _vram_freed)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'I', 'i')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
        assert failed == 0
        assert svc.classify_images(LOCAL_USER, ds.id) == 1
        assert db.session.get(FaceDatasetImage, ids[0]).framing == 'body'

        answers[0] = '{"framing": "face"}'
        assert svc.classify_images(LOCAL_USER, ds.id) == 0
        assert db.session.get(FaceDatasetImage, ids[0]).framing == 'body'

        assert svc.classify_images(LOCAL_USER, ds.id, force=True) == 1
        assert db.session.get(FaceDatasetImage, ids[0]).framing == 'face'


def test_classify_images_fills_a_generated_row_with_no_framing(app, monkeypatch):
    """A crop clears framing on generated shots too; the fill pass must see them.

    Restricting the pool to source='import' was why a cropped catalog shot
    vanished from Composition and never came back without a full rescan.
    """
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: '{"framing": "face"}')
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', _vram_freed)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'G', 'g')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
        assert failed == 0
        row = db.session.get(FaceDatasetImage, ids[0])
        row.source = 'generated'
        row.framing = None
        db.session.commit()

        assert svc.classify_images(LOCAL_USER, ds.id) == 1
        assert db.session.get(FaceDatasetImage, ids[0]).framing == 'face'


def test_classify_images_force_includes_generated_rows(app, monkeypatch):
    """A tagged generated shot is left alone unless force=True (Bank rescan)."""
    from app.services import face_dataset_service as svc
    from app.services import vision_ollama
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.extensions import db

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: '{"framing": "face"}')
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', _vram_freed)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'G', 'g')
        ids, failed = svc.import_images(LOCAL_USER, ds.id, [_png()], crop=False)
        assert failed == 0
        row = db.session.get(FaceDatasetImage, ids[0])
        row.source = 'generated'
        row.framing = 'body'
        db.session.commit()

        assert svc.classify_images(LOCAL_USER, ds.id) == 0
        assert db.session.get(FaceDatasetImage, ids[0]).framing == 'body'

        assert svc.classify_images(LOCAL_USER, ds.id, force=True) == 1
        assert db.session.get(FaceDatasetImage, ids[0]).framing == 'face'


def test_classify_route_forwards_force(client, app, monkeypatch):
    from app.routes import datasets as routes

    seen = {}

    def _fake(_user_id, _dataset_id, force=False, report=None):
        seen['force'] = force
        if report is not None:
            report['attempted'] = 4
            report['unanswered'] = 0
        return 4

    monkeypatch.setattr(routes.svc, 'classify_images', _fake)
    ds_id = _create(client, 'Kai', 'kai').get_json()['id']
    resp = client.post(f'/api/dataset/{ds_id}/classify', json={'force': True})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['ok'] is True
    assert body['classified'] == 4
    assert body['attempted'] == 4
    assert seen['force'] is True

    seen.clear()
    resp = client.post(f'/api/dataset/{ds_id}/classify')
    assert resp.status_code == 200
    assert seen['force'] is False


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
