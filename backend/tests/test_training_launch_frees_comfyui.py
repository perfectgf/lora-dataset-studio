"""Before a LOCAL training takes the card, ComfyUI is asked to give it back.

Reported by acontentsheltie (Discord #help, 2026-09-05): a Krea 2 Raw run on an
RTX 3090 at 3 % GPU, RAM climbing, ETA around 300 hours. Every launch guard read
LDS's own state — its queue table, its vision flag, its Ollama fence — and none
asked ComfyUI what IT held, while ComfyUI keeps every model of the session
resident until it needs the room itself. The vision window has pulled `/free`
before every pass since July; the training launch never did (and the docstring
of memory_release.py already claimed it did).

Three properties, on BOTH launchers — the video lane duplicates the admission
block verbatim, so a fix that lands on one lane only is the next report:

* the request is made after the Ollama fence and before the child exists;
* it is best-effort: an unknown verdict or a raised exception does not stop the
  launch — a training does not need ComfyUI at all;
* the import stays inside the helper. conftest stubs the live POST by
  attribute (`app.utils.comfyui.free_comfyui_vram`); a module-level `from`
  import would bind the real function first and the unit suite would POST
  /free to whatever ComfyUI runs on the developer's machine.
"""
import pathlib
import re

import pytest

from app.config import LOCAL_USER
from test_training_service import _FakeProc, _configure_aitoolkit
from test_video_training_local import _FakeProc as _VideoFakeProc
from test_video_training_local import _aitoolkit as _video_aitoolkit
from test_video_training_local import _clear_fence, _video_dataset

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / 'app' / 'services'


class _Verdict:
    """Just the `.value` the helper reads off a ComfyVramFreeVerdict."""

    def __init__(self, value):
        self.value = value


def _body(path, name):
    src = pathlib.Path(path).read_text(encoding='utf-8')
    start = src.index(f'def {name}(')
    end = src.find('\ndef ', start + 1)
    return src[start:end if end > 0 else len(src)]


def _wire_image_launch(app, tmp_path, monkeypatch, *, free, calls):
    """Everything between launch_training() and the child, stubbed the way
    test_training_service does it — plus recorders on the three events whose
    ORDER is the contract: the Ollama fence, the ComfyUI request, the spawn."""
    from app.services import aitoolkit_state_bridge, checkpoint_registry, ollama_gpu_fence
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    monkeypatch.setattr(lt, 'assert_interpreter_ready', lambda: None)
    monkeypatch.setattr(lt, 'assert_free_disk', lambda *a, **k: None)
    monkeypatch.setattr(lt, '_assert_no_vision_pass_on_gpu', lambda: None)
    monkeypatch.setattr(lt.queue_manager, 'has_comfyui_work', lambda: False)
    monkeypatch.setattr(ollama_gpu_fence, 'ensure_released_for_comfy',
                        lambda: calls.append('ollama_fence') or True)
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram',
                        lambda *a, **k: calls.append('comfyui_free') or free(*a, **k))
    monkeypatch.setattr(aitoolkit_state_bridge, 'probe',
                        lambda _root: {'supported': False, 'reasons': ['test-disabled']})
    monkeypatch.setattr(checkpoint_registry, 'prepare_launch',
                        lambda *_a, **_k: {'manifest': [], 'snapshot': {}})
    monkeypatch.setattr(checkpoint_registry, 'prepared_generation_identity',
                        lambda prepared: {} if prepared is not None else None)
    monkeypatch.setattr(checkpoint_registry, 'register_launch', lambda *_a, **_k: object())

    def fake_popen(args, **_kwargs):
        # `lt.subprocess` IS the subprocess module, so this stub also catches the
        # nvidia-smi `subprocess.run` the VRAM readings make: only the child that
        # runs ai-toolkit counts as the spawn (same filter as test_training_service).
        if any(str(item).replace('\\', '/').endswith('/run.py') or str(item) == 'run.py'
               for item in args):
            calls.append('spawn')
        return _FakeProc(7171)
    monkeypatch.setattr(lt.subprocess, 'Popen', fake_popen)
    # The before/after readings are a separate seam (test_gpu_vram_used_gb_...);
    # here they must neither fork nvidia-smi nor depend on the box's card.
    from app.services import system_stats
    monkeypatch.setattr(system_stats, 'gpu_vram_used_gb', lambda: 1.5)
    monkeypatch.setattr(lt, '_record_training_process_identity', lambda _pid: None)
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)
    exported = tmp_path / 'export'
    exported.mkdir(exist_ok=True)
    monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit', lambda *_a, **_k: str(exported))
    with app.app_context():
        lt._clear_training_identity(ttl_seconds=1)
        ds = svc.create_dataset(LOCAL_USER, 'Card holder', 'card_holder')
        return ds.id


def _launch_image(app, ds_id):
    from app.services import lora_training as lt
    with app.app_context():
        try:
            return lt.launch_training(LOCAL_USER, ds_id, steps=500,
                                      check_captions=False, masked=False)
        finally:
            lt._clear_training_identity(ttl_seconds=1)


# --- the image lane -------------------------------------------------------------

def test_the_image_launch_asks_comfyui_between_the_ollama_fence_and_the_spawn(
        app, tmp_path, monkeypatch):
    calls = []
    ds_id = _wire_image_launch(app, tmp_path, monkeypatch,
                               free=lambda *a, **k: _Verdict('freed'), calls=calls)
    res = _launch_image(app, ds_id)
    assert res['started'] is True
    assert calls.count('comfyui_free') == 1 and calls.count('spawn') == 1
    fence = max(i for i, c in enumerate(calls) if c == 'ollama_fence')
    assert fence < calls.index('comfyui_free') < calls.index('spawn'), calls


def test_the_request_does_not_wait_the_helpers_full_ten_seconds(app, tmp_path, monkeypatch):
    """It runs under _queue_lock + GPU_ARBITER_LOCK, where Stop waits too. A live
    ComfyUI answers /free at once; only a dead host reaches the timeout, and it
    must not hold the card's admission for ten seconds to do so."""
    from app.services import lora_training as lt
    seen = {}
    ds_id = _wire_image_launch(app, tmp_path, monkeypatch,
                               free=lambda *a, **k: seen.update(k) or _Verdict('freed'),
                               calls=[])
    _launch_image(app, ds_id)
    assert seen.get('timeout') == lt._COMFYUI_FREE_TIMEOUT_S
    assert 0 < lt._COMFYUI_FREE_TIMEOUT_S < 10


def test_a_silent_comfyui_does_not_stop_the_launch(app, tmp_path, monkeypatch):
    calls = []
    ds_id = _wire_image_launch(app, tmp_path, monkeypatch,
                               free=lambda *a, **k: _Verdict('unknown'), calls=calls)
    res = _launch_image(app, ds_id)
    assert res['started'] is True and 'spawn' in calls


def test_a_raising_free_does_not_stop_the_launch(app, tmp_path, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError('ComfyUI exploded')
    calls = []
    ds_id = _wire_image_launch(app, tmp_path, monkeypatch, free=_boom, calls=calls)
    res = _launch_image(app, ds_id)
    assert res['started'] is True and 'spawn' in calls


# --- the video lane -------------------------------------------------------------

def test_the_video_launch_asks_too(app, tmp_path, monkeypatch):
    """Same admission block, copied verbatim into video_training_local — so the
    same request, at the same place: after the Ollama fence, before the child."""
    from app.services import system_stats
    from app.services import video_training_local as vtl
    calls = []
    _video_aitoolkit(monkeypatch, tmp_path)
    monkeypatch.setattr(system_stats, 'gpu_vram_used_gb', lambda: 1.5)
    monkeypatch.setattr('app.utils.comfyui.free_comfyui_vram',
                        lambda *a, **k: calls.append('comfyui_free') or _Verdict('freed'))

    def _spawn(argv, cwd, env, stdout):
        calls.append('spawn')
        return _VideoFakeProc()

    with app.app_context():
        _clear_fence()
        vid = _video_dataset(tmp_path)
        res = vtl.start_video_training('local', vid.id, steps=100, _spawn=_spawn)
        assert res['started'] is True
        _clear_fence()
    assert calls == ['comfyui_free', 'spawn']


# --- the source contracts: order, and the import that must stay lazy --------------

def _indent_of(body, line_start):
    m = re.search(r'^( *)' + re.escape(line_start), body, re.M)
    assert m, line_start
    return len(m.group(1))


def test_the_request_sits_after_the_ollama_fence_and_before_the_child_in_both_lanes():
    lanes = (
        ('image', _body(SRC_DIR / 'lora_training.py', '_lt_spawn_transaction'),
         'subprocess.Popen(', 'with _queue_lock, GPU_ARBITER_LOCK:', '_comfyui_free_report('),
        ('video', _body(SRC_DIR / 'video_training_local.py', 'start_video_training'),
         'proc = spawn(', 'with lt._queue_lock, GPU_ARBITER_LOCK:', 'lt._comfyui_free_report('),
    )
    for name, body, child_marker, with_marker, report_marker in lanes:
        fence = body.index('ensure_released_for_comfy()')
        asked = body.index('_comfyui_free_before_training(')
        child = body.index(child_marker)
        assert fence < asked < child, f'{name}: fence={fence} asked={asked} child={child}'
        assert body.index(report_marker) > child, name
        # ...and OUTSIDE the lock pair: the report line sits at the indentation of
        # the `with` line itself, dedented out of the block Stop waits on. A
        # rewrite that pulls it back under the locks reads green on the index
        # check above and red here.
        assert _indent_of(body, report_marker) == _indent_of(body, with_marker), name


def test_the_comfyui_import_stays_inside_the_helper():
    src = (SRC_DIR / 'lora_training.py').read_text(encoding='utf-8')
    module_level = [ln for ln in src.splitlines() if re.match(r'^(from|import)\s', ln)]
    assert not any('free_comfyui_vram' in ln for ln in module_level), \
        'a module-level import binds the real POST before conftest can stub it'
    helper = _body(SRC_DIR / 'lora_training.py', '_comfyui_free_before_training')
    assert 'from ..utils.comfyui import free_comfyui_vram' in helper


# --- the report half, and the reading it relies on ---------------------------------

def test_the_report_never_raises_past_the_spawn(monkeypatch):
    from app.services import lora_training as lt
    from app.services import system_stats

    def _broken():
        raise OSError('nvidia-smi went away')
    monkeypatch.setattr(system_stats, 'gpu_vram_used_gb', _broken)
    lt._comfyui_free_report(None)
    lt._comfyui_free_report({'asked_at': 'not a number', 'lane': 'image'})


@pytest.mark.gpu_reading
def test_gpu_vram_used_gb_is_one_fresh_reading_or_none(monkeypatch):
    from app.services import system_stats
    monkeypatch.setattr(system_stats, '_gpu_sample', lambda: (12, 3.4, 24.0, 55))
    assert system_stats.gpu_vram_used_gb() == 3.4
    monkeypatch.setattr(system_stats, '_gpu_sample', lambda: None)
    assert system_stats.gpu_vram_used_gb() is None
