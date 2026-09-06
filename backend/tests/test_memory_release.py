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


# ── a refusal is not a wall (2026-09-06) ─────────────────────────────────────
# When the render on the card is LDS's own, the first press refuses and says
# a second press interrupts it; the second press does. A training, a prompt
# that is not LDS's, and a ComfyUI that cannot be asked keep refusing.

@pytest.fixture
def own_render(levers, monkeypatch):
    """ComfyUI busy with a render of LDS's own: the queue manager says so, and
    the interrupt it is asked for empties ComfyUI's queue (unless told not to)."""
    from types import SimpleNamespace
    from app.job_queue import queue_manager
    from app.services import live_studio
    state = {'state': 'own', 'verdict': 'interrupted', 'lets_go': True, 'interrupts': 0, 'expected': [],
             'live': None}
    row = SimpleNamespace(job_id='job-own', status='sent_to_comfy', comfyui_prompt_id='prompt-own')
    levers['busy'] = True
    monkeypatch.setattr(queue_manager, 'own_running_job',
                        lambda: (state['state'], row if state['state'] in ('own', 'unmapped') else None))

    def interrupt(expected=None):
        state['interrupts'] += 1
        state['expected'].append(expected)
        if state['verdict'] in ('interrupted', 'idle') and state['lets_go']:
            levers['busy'] = False   # ComfyUI let go, or the render had already ended
        return (state['verdict'], 'job-other' if state['verdict'] == 'other' else 'job-own')
    monkeypatch.setattr(queue_manager, 'interrupt_own_running', interrupt)
    monkeypatch.setattr(live_studio, 'current', lambda: state['live'])
    return state


def test_a_render_of_our_own_is_refused_once_with_the_offer_to_interrupt(app, levers, own_render):
    with app.app_context():
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(settle_seconds=0)
    assert e.value.can_interrupt is True and e.value.job_id == 'job-own'
    assert 'ComfyUI is rendering' in str(e.value) and 'Press 🧹 Free memory again' in str(e.value)
    assert own_render['interrupts'] == 0 and levers['calls'] == []


def test_the_second_press_interrupts_our_render_then_frees(app, levers, own_render):
    with app.app_context():
        out = mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
    assert own_render['interrupts'] == 1 and own_render['expected'] == ['job-own']
    assert out['ok'] is True and out['interrupted'] == 'job-own' and out['comfyui'] == 'freed'
    assert [c[0] for c in levers['calls']] == ['stats', 'comfy', 'vision', 'stats']


def test_the_second_press_is_bound_to_the_job_the_offer_named(app, levers, own_render):
    """Within the minute the next queued render could be the one on the card:
    a forced press for another job touches nothing and offers afresh, naming
    the job that runs now (found in review)."""
    with app.app_context():
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, job_id='job-earlier', settle_seconds=0)
        assert e.value.can_interrupt is True and e.value.job_id == 'job-own'
        assert 'Another clip' in str(e.value) and own_render['interrupts'] == 0
        own_render['verdict'] = 'other'
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
        assert e.value.can_interrupt is True and e.value.job_id == 'job-other'
    assert levers['calls'] == []


def test_a_render_that_ended_by_itself_lets_the_gesture_go_on(app, levers, own_render):
    own_render['verdict'] = 'idle'
    with app.app_context():
        out = mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
    assert out['ok'] is True and out['interrupted'] is None and out['comfyui'] == 'freed'


def test_the_local_live_channel_refuses_whatever_the_press(app, levers, own_render):
    from types import SimpleNamespace
    own_render['live'] = SimpleNamespace(state='running', params={'gpu': 'local'})
    with app.app_context():
        for interrupt in (False, True):
            with pytest.raises(mr.MemoryReleaseBusy) as e:
                mr.free_memory(interrupt=interrupt, job_id='job-own', settle_seconds=0)
            assert e.value.can_interrupt is False
            assert 'Stop the local Live channel' in str(e.value) and 'before freeing the memory' in str(e.value)
    assert own_render['interrupts'] == 0 and levers['calls'] == []


def test_an_own_render_whose_id_was_lost_is_refused_with_the_banner_not_as_foreign(app, levers, own_render):
    own_render['state'] = 'unmapped'
    with app.app_context():
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
    assert e.value.can_interrupt is False and 'recovery banner' in str(e.value)
    assert "not LDS's" not in str(e.value) and own_render['interrupts'] == 0


def test_the_unload_after_the_interrupt_runs_under_the_arbiter_lock(app, levers, own_render):
    """The wait runs without the lock (the dock keeps answering); the unload
    takes it, so the worker cannot send the next queued job between the
    re-check and the /free (found in review, both ways)."""
    import threading
    from app.job_queue import GPU_ARBITER_LOCK
    seen = []
    with app.app_context():
        from app.utils import comfyui as cu
        original = cu.free_comfyui_vram

        def free_under_lock():
            other = []
            t = threading.Thread(target=lambda: other.append(GPU_ARBITER_LOCK.acquire(timeout=0.2)))
            t.start(); t.join()
            seen.append(other[0])
            return original()
        cu.free_comfyui_vram = free_under_lock
        try:
            out = mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
        finally:
            cu.free_comfyui_vram = original
    assert out['ok'] is True and seen == [False], 'the arbiter lock was held during /free'
    assert GPU_ARBITER_LOCK.acquire(blocking=False) is True, 'released afterwards'
    GPU_ARBITER_LOCK.release()


def test_a_job_that_started_right_after_the_interrupt_is_never_unloaded_under(app, levers, own_render):
    from unittest.mock import patch
    own_render['lets_go'] = False   # the queue reads busy again: the next job
    with app.app_context():
        with patch('app.utils.comfyui.running_prompt_identity', return_value=('prompt-next', 'job-next')):
            with pytest.raises(mr.MemoryReleaseBusy) as e:
                mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
    assert 'started the next job' in str(e.value) and e.value.can_interrupt is False
    assert own_render['interrupts'] == 1 and levers['calls'] == []


def test_a_render_that_is_not_ours_is_never_interrupted_even_when_asked(app, levers, own_render):
    own_render['state'] = 'foreign'
    with app.app_context():
        for interrupt in (False, True):
            with pytest.raises(mr.MemoryReleaseBusy) as e:
                mr.free_memory(interrupt=interrupt, settle_seconds=0)
            assert e.value.can_interrupt is False and 'Press' not in str(e.value)
    assert own_render['interrupts'] == 0 and levers['calls'] == []


def test_an_unknown_running_prompt_is_a_plain_refusal(app, levers, own_render):
    own_render['state'] = 'unknown'
    with app.app_context():
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, settle_seconds=0)
    assert e.value.can_interrupt is False and own_render['interrupts'] == 0


def test_a_training_is_refused_whatever_the_press(app, levers, own_render):
    levers['training'] = True
    with app.app_context():
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, settle_seconds=0)
    assert 'training' in str(e.value) and e.value.can_interrupt is False
    assert own_render['interrupts'] == 0 and levers['calls'] == []


def test_a_mute_comfyui_or_one_that_does_not_let_go_is_said_as_such(app, levers, own_render):
    own_render['verdict'] = 'unknown'
    with app.app_context():
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
        assert 'did not answer' in str(e.value) and e.value.can_interrupt is False
        own_render['verdict'] = 'foreign'
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, job_id='job-own', settle_seconds=0)
        assert "not LDS's" in str(e.value) and e.value.can_interrupt is False
        own_render['verdict'] = 'interrupted'
        own_render['lets_go'] = False
        with pytest.raises(mr.MemoryReleaseBusy) as e:
            mr.free_memory(interrupt=True, job_id='job-own', interrupt_wait_seconds=0, settle_seconds=0)
        assert 'has not let go' in str(e.value)
        assert e.value.can_interrupt is True and e.value.job_id == 'job-own'
    assert own_render['interrupts'] == 3 and levers['calls'] == []


def test_the_route_carries_the_offer_and_the_second_press(client, app, levers, own_render):
    r = client.post('/api/system/free-memory', json={})
    assert r.status_code == 409
    body = r.get_json()
    assert body['can_interrupt'] is True and body['job_id'] == 'job-own'
    assert 'Press 🧹 Free memory again' in body['error']
    r = client.post('/api/system/free-memory', json={'interrupt': 'yes'})
    assert r.status_code == 400 and own_render['interrupts'] == 0
    r = client.post('/api/system/free-memory', json={'interrupt': True, 'job_id': 7})
    assert r.status_code == 400 and own_render['interrupts'] == 0
    r = client.post('/api/system/free-memory', json={'interrupt': True, 'job_id': 'job-earlier'})
    assert r.status_code == 409 and r.get_json()['job_id'] == 'job-own' and own_render['interrupts'] == 0
    r = client.post('/api/system/free-memory', json={'interrupt': True, 'job_id': 'job-own'})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()['interrupted'] == 'job-own' and own_render['interrupts'] == 1
