"""One GPU answer for two surfaces (CLAUDE.md: Bank and Dataset parity).

The Bank's face pass and the dataset scorer run the SAME models and ask the same
question — "may I put InsightFace on the GPU for this run?". They used to answer
it in different places, and in fact differently: the Bank had a GPU lane behind
`face_scoring.device`, the dataset scorer was hard-pinned to CPU. That kind of
divergence is invisible until someone notices one surface is far slower than the
other, which is exactly how the face SIZE gate drifted apart before it.

Also pinned here: the GPU is EXCLUSIVE or it is nothing. The scorer was taken off
CUDA by "fix(gpu): serialize local inference and ComfyUI recovery" because it was
grabbing the card outside any arbiter; the lane only comes back THROUGH the
window that commit created.
"""
import pytest

from app import capabilities
from app.services import face_similarity as fs
from app.services import image_bank_service as banks


def _prefs(monkeypatch, value):
    original = capabilities.cfg.get
    monkeypatch.setattr(capabilities.cfg, 'get',
                        lambda k, *a, **kw: (value if k == 'face_scoring.device'
                                             else original(k, *a, **kw)))


def _gpu(monkeypatch, available):
    monkeypatch.setattr(capabilities, 'face_gpu_available', lambda: available)


# --- the shared answer -------------------------------------------------------

@pytest.mark.parametrize('pref,available,expected', [
    (None, True, ('cuda', True)),        # 'auto' is the default
    ('auto', True, ('cuda', True)),
    ('cuda', True, ('cuda', True)),
    ('cpu', True, ('cpu', False)),       # an explicit refusal outranks capability
    ('auto', False, ('cpu', False)),     # stock install: CPU onnxruntime
    ('cuda', False, ('cpu', False)),     # asking cannot conjure a provider
    ('nonsense', True, ('cpu', False)),  # an unreadable value never opens the GPU
])
def test_the_device_verdict(monkeypatch, pref, available, expected):
    _prefs(monkeypatch, pref)
    _gpu(monkeypatch, available)
    assert capabilities.resolve_face_device() == expected


def test_both_surfaces_read_the_same_answer(monkeypatch):
    """The Bank keeps its local name (its tests patch it) but no longer owns the
    RULE — so the two can no longer drift apart."""
    _prefs(monkeypatch, 'auto')
    _gpu(monkeypatch, True)
    assert banks._resolve_face_device() == capabilities.resolve_face_device()
    _gpu(monkeypatch, False)
    assert banks._resolve_face_device() == capabilities.resolve_face_device()


# --- the GPU is exclusive or it is nothing -----------------------------------

def _stub_run(monkeypatch, seen):
    monkeypatch.setattr(fs, 'is_available', lambda: True)
    monkeypatch.setattr(fs, '_scoring_python', lambda: 'python')

    def _run(python, payload, timeout, on_progress):
        import json as _json
        seen['payload'] = _json.loads(payload)
        return '{"ref_ok": true, "results": {}}', [], 0, False

    monkeypatch.setattr(fs, '_run_scorer', _run)


def test_a_cpu_pass_never_opens_the_gpu_window(app, monkeypatch, tmp_path):
    """Its original promise: it runs beside ComfyUI, bothering nobody."""
    ref = tmp_path / 'ref.png'; ref.write_bytes(b'x')
    img = tmp_path / 'a.png'; img.write_bytes(b'x')
    seen = {}
    _stub_run(monkeypatch, seen)
    monkeypatch.setattr(capabilities, 'resolve_face_device', lambda: ('cpu', False))

    def _boom(*a, **k):
        raise AssertionError('a CPU pass must not take the GPU window')

    import app.gpu_window as gw
    monkeypatch.setattr(gw, 'gpu_exclusive_vision_window', _boom)
    results, err = fs.score_dataset_faces(str(ref), [str(img)])
    assert err is None
    assert seen['payload']['device'] == 'cpu'


def test_a_gpu_pass_runs_inside_the_exclusive_window(app, monkeypatch, tmp_path):
    ref = tmp_path / 'ref.png'; ref.write_bytes(b'x')
    img = tmp_path / 'a.png'; img.write_bytes(b'x')
    seen = {}
    _stub_run(monkeypatch, seen)
    monkeypatch.setattr(capabilities, 'resolve_face_device', lambda: ('cuda', True))

    from contextlib import contextmanager
    entered = []

    @contextmanager
    def _window(flag_ttl=None):
        entered.append(flag_ttl)
        yield

    import app.gpu_window as gw
    monkeypatch.setattr(gw, 'gpu_exclusive_vision_window', _window)
    results, err = fs.score_dataset_faces(str(ref), [str(img)])
    assert err is None
    assert seen['payload']['device'] == 'cuda'
    assert entered, 'the GPU lane must go through the arbiter, not around it'


def test_a_busy_gpu_is_reported_as_busy_not_as_a_failure(app, monkeypatch, tmp_path):
    """And it does NOT quietly fall back to CPU: someone who chose the fast lane
    is owed the truth, not a slow pass wearing its name."""
    ref = tmp_path / 'ref.png'; ref.write_bytes(b'x')
    img = tmp_path / 'a.png'; img.write_bytes(b'x')
    seen = {}
    _stub_run(monkeypatch, seen)
    monkeypatch.setattr(capabilities, 'resolve_face_device', lambda: ('cuda', True))

    import app.gpu_window as gw

    def _busy(flag_ttl=None):
        raise gw.GpuBusyError('a vision task is already running')

    monkeypatch.setattr(gw, 'gpu_exclusive_vision_window', _busy)
    results, err = fs.score_dataset_faces(str(ref), [str(img)])
    assert results == {}
    assert err['kind'] == 'gpu_busy'
    assert 'busy' in err['detail']
