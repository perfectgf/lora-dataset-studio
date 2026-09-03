"""Click-to-ComfyUI latency: the pieces that were measured to cost seconds."""
import threading
import time

from app import capabilities
from app.routes import _common


def test_the_comfyui_gate_asks_comfyui_only_never_the_whole_app_probe(app, monkeypatch):
    """Measured on the maintainer's machine: capabilities.probe() is 24 s cold and
    cached 30 s; the ComfyUI probe alone is 30 ms. The gate must not touch the
    former, with or without `force`."""
    def boom(*a, **k):
        raise AssertionError('the generate gate ran the whole-app capabilities probe')
    monkeypatch.setattr(capabilities, 'probe', boom)
    monkeypatch.setattr(capabilities, 'probe_comfyui',
                        lambda: {'ok': True, 'detail': 'http://x', 'status': 'ok', 'hint': ''})
    with app.test_request_context():
        assert _common._require_comfyui() is None
        assert _common._require_comfyui(force=True) is None
    monkeypatch.setattr(capabilities, 'probe_comfyui',
                        lambda: {'ok': False, 'status': 'slow', 'detail': 'slow', 'hint': 'wait'})
    with app.test_request_context():
        body, status = _common._require_comfyui()
        assert status == 409 and 'too slowly' in body.get_json()['error']
    monkeypatch.setattr(capabilities, 'probe_comfyui',
                        lambda: {'ok': False, 'status': 'unreachable', 'detail': 'down', 'hint': 'Check the URL'})
    with app.test_request_context():
        body, status = _common._require_comfyui(force=True)
        assert status == 409 and 'not reachable' in body.get_json()['error']


def test_a_queued_job_wakes_the_worker_instead_of_waiting_the_poll_out(app, monkeypatch):
    from app import job_queue as jq
    qm = type(jq.queue_manager)()
    qm._app = app
    ticks = []
    monkeypatch.setattr(qm, 'process_one', lambda: ticks.append(time.perf_counter()) or False)
    monkeypatch.setattr(jq, 'IDLE_SLEEP_SECONDS', 5)          # a poll that would take 5 s
    qm._running = True
    t = threading.Thread(target=qm._run_loop, daemon=True)
    t.start()
    try:
        time.sleep(0.3)
        assert len(ticks) == 1, 'idle: one look, then asleep on the poll'
        kicked = time.perf_counter()
        qm._kick.set()                                          # what add_job does
        deadline = time.perf_counter() + 2
        while len(ticks) < 2 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert len(ticks) >= 2, 'the kick woke the loop'
        assert ticks[1] - kicked < 0.5, f'woke in {ticks[1] - kicked:.2f}s, not the 5 s poll'
    finally:
        qm._running = False
        qm._kick.set()
        t.join(timeout=2)


def test_add_job_kicks_the_worker(app, monkeypatch):
    from app import job_queue as jq
    qm = jq.queue_manager
    qm._kick.clear()
    monkeypatch.setattr(jq, 'require_comfyui_enqueue_ready', lambda: None)
    with app.app_context():
        qm.add_job(job_type='image', user_id='local', workflow_data={'1': {'class_type': 'X', 'inputs': {}}},
                   prompt='p', metadata={'model_name': 'video_lora_test', 'is_video_test': True})
    assert qm._kick.is_set()
