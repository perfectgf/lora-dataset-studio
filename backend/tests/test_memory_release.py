"""🧹 Free memory: the gesture behind the button beside the load readout.

Nothing here reaches a real ComfyUI, Ollama or psutil — the two levers and the
two readings are replaced, so the suite proves the ORDER (guard, unload,
release, re-read), the refusals and the arithmetic of the answer."""
import pytest

from app.services import memory_release as mr
from app.utils.comfyui import ComfyVramFreeVerdict


@pytest.fixture
def levers(monkeypatch):
    """Both levers recorded, readings scripted: first call = before, second = after."""
    from app.services import system_stats, vision_llm, cloud_training
    from app.utils import comfyui as cu
    state = {'calls': [], 'readings': [{'ram_used_gb': 43.8, 'ram_total_gb': 47.7, 'vram_used_gb': 15.6},
                                      {'ram_used_gb': 12.1, 'ram_total_gb': 47.7, 'vram_used_gb': 0.9}],
             'verdict': ComfyVramFreeVerdict.FREED, 'vision': True, 'busy': False, 'training': False}

    def stats(force=False):
        state['calls'].append(('stats', force))
        return dict(state['readings'].pop(0) if state['readings'] else {})
    monkeypatch.setattr(system_stats, 'machine_stats', stats)
    # The service imports the lever from utils.comfyui at call time.
    monkeypatch.setattr(cu, 'free_comfyui_vram', lambda: state['calls'].append(('comfy',)) or state['verdict'])
    monkeypatch.setattr(vision_llm, 'unload_vision_model',
                        lambda **kw: state['calls'].append(('vision',)) or state['vision'])
    monkeypatch.setattr(mr, 'comfyui_queue_busy', lambda: state['busy'])
    monkeypatch.setattr(cloud_training, 'training_in_progress', lambda: state['training'])
    return state


def test_the_gesture_runs_in_order_and_reports_what_it_measured(app, levers):
    with app.app_context():
        out = mr.free_memory(settle_seconds=0)
    assert [c[0] for c in levers['calls']] == ['stats', 'comfy', 'vision', 'stats']
    assert all(c[1] is True for c in levers['calls'] if c[0] == 'stats'), 'both readings bypass the shared cache'
    assert out['ok'] is True and out['comfyui'] == 'freed' and out['vision_released'] is True
    assert (out['ram_before_gb'], out['ram_after_gb'], out['freed_gb']) == (43.8, 12.1, 31.7)
    assert (out['vram_before_gb'], out['vram_after_gb']) == (15.6, 0.9)
    assert out['ram_total_gb'] == 47.7


def test_an_offline_comfyui_is_nothing_to_free_not_a_failure(app, levers):
    levers['verdict'] = ComfyVramFreeVerdict.COMFYUI_OFFLINE
    levers['vision'] = False
    levers['readings'] = [{'ram_used_gb': 10.0}, {'ram_used_gb': 10.0}]
    with app.app_context():
        out = mr.free_memory(settle_seconds=0)
    assert out['ok'] is True and out['comfyui'] == 'offline' and out['vision_released'] is False
    assert out['freed_gb'] == 0.0 and out['vram_before_gb'] is None


def test_the_verdict_survives_a_rebuilt_enum(app, levers, monkeypatch):
    """test_comfyui_utils reloads app.utils.comfyui, which rebuilds the verdict
    enum for every test that runs after it in the same worker; this file's
    fixture keeps the member it imported before. Mapped by member identity the
    service answered 'unknown' — the release runner drew that order, the local
    gate did not. The same rebuilt class, put in place for this test alone, is
    what pins the mapping by value, whichever worker the two files share."""
    from enum import Enum
    from app.utils import comfyui as cu
    rebuilt = Enum('ComfyVramFreeVerdict', [(m.name, m.value) for m in ComfyVramFreeVerdict])
    assert rebuilt.FREED is not ComfyVramFreeVerdict.FREED
    monkeypatch.setattr(cu, 'ComfyVramFreeVerdict', rebuilt)
    with app.app_context():
        out = mr.free_memory(settle_seconds=0)
    assert out['comfyui'] == 'freed'
    levers['verdict'] = ComfyVramFreeVerdict.COMFYUI_OFFLINE
    levers['readings'] = [{'ram_used_gb': 10.0}, {'ram_used_gb': 10.0}]
    with app.app_context():
        out = mr.free_memory(settle_seconds=0)
    assert out['comfyui'] == 'offline'


def test_a_machine_that_cannot_measure_still_answers(app, levers):
    levers['readings'] = [{}, {}]
    with app.app_context():
        out = mr.free_memory(settle_seconds=0)
    assert out['ram_before_gb'] is None and out['freed_gb'] is None and out['ok'] is True


def test_refused_while_comfyui_renders_or_a_training_runs(app, levers):
    levers['busy'] = True
    with app.app_context():
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(settle_seconds=0)
        assert 'ComfyUI is rendering' in str(e.value)
        assert levers['calls'] == [], 'refused BEFORE anything is unloaded'
        levers['busy'] = False
        levers['training'] = True
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(settle_seconds=0)
        assert 'training' in str(e.value)


def test_the_queue_probe_reads_running_and_pending_and_says_unknown_when_it_cannot(monkeypatch):
    from app.utils import comfyui as cu
    import app.services.memory_release as m
    monkeypatch.setattr(cu, 'api_address', lambda: 'http://127.0.0.1:8188')

    class R:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body
    answers = {}
    monkeypatch.setattr(m.requests, 'get', lambda url, **kw: answers['r'])
    answers['r'] = R(200, {'queue_running': [], 'queue_pending': []})
    assert m.comfyui_queue_busy() is False
    answers['r'] = R(200, {'queue_running': [[1]], 'queue_pending': []})
    assert m.comfyui_queue_busy() is True
    answers['r'] = R(200, {'queue_running': [], 'queue_pending': [[1], [2]]})
    assert m.comfyui_queue_busy() is True
    answers['r'] = R(500, {})
    assert m.comfyui_queue_busy() is None
    answers['r'] = R(200, {'nope': 1})
    assert m.comfyui_queue_busy() is None
    monkeypatch.setattr(cu, 'api_address', lambda: '')
    assert m.comfyui_queue_busy() is None


def test_the_route_answers_the_measurement_or_the_refusal(client, app, levers):
    r = client.post('/api/system/free-memory')
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['freed_gb'] == 31.7
    levers['busy'] = True
    r = client.post('/api/system/free-memory')
    assert r.status_code == 409 and 'ComfyUI is rendering' in r.get_json()['error']
