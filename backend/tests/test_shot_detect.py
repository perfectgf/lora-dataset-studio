"""The TransNetV2 shot-boundary detector — contracts that must not drift.

Not a test of the model itself (~33 MB of weights, and neither torch nor
transnetv2-pytorch is on CI). What IS tested: the frame-index -> PTS-seconds
conversion (the one arithmetic slip here would silently mis-cut every clip in
the app), the minimum-shot-length filter, the per-file failure isolation (one
corrupt source must cost that source and not the batch), and the streaming
subprocess protocol — the same shape already proven for bank_score_infer.py
and watermark_detect_infer.py.
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'infer'))

from app import config as cfg                       # noqa: E402
from app.services import shot_detect as sd           # noqa: E402


def _infer_module():
    """Import the CHILD script the way it actually runs — by path, with no app
    package on the way in. It must stay importable with nothing but the
    standard library, because the interpreter that runs it for real is not
    this one (torch/transnetv2-pytorch live in a dedicated venv)."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'infer', 'shot_detect_infer.py')
    spec = importlib.util.spec_from_file_location('shot_detect_infer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- identity: the two halves must never drift ---------------------------------

def test_detector_id_does_not_drift_between_parent_and_child():
    """The parent writes this into VideoClip.detector; the child hardcodes it
    for its own docstring/tests. They run in two different interpreters, so
    nothing but this test stops them disagreeing."""
    infer = _infer_module()
    assert infer.DETECTOR_ID == sd.DETECTOR_ID == 'transnetv2'


def test_child_declares_no_third_party_import_at_module_level():
    """Exec'd here in the Flask venv, which has neither torch nor av by
    contract (this dev machine happens to have both, CI has neither). Any
    heavy import sitting at module level would make importing the child itself
    require them, which breaks every other test in this file too."""
    module = _infer_module()
    assert module.DETECTOR_ID == 'transnetv2'


# --- the child's pure scene-boundary rule (no torch, no numpy) -----------------

def test_predictions_to_scenes_finds_one_boundary():
    infer = _infer_module()
    # 6 frames of shot A (frame 5 is the transition itself), 4 of shot B.
    probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
    assert infer._predictions_to_scenes(probs, threshold=0.5) == [[0, 5], [6, 9]]


def test_predictions_to_scenes_of_a_single_shot_video():
    infer = _infer_module()
    assert infer._predictions_to_scenes([0.1] * 20, threshold=0.5) == [[0, 19]]


def test_predictions_to_scenes_of_nothing_is_empty():
    infer = _infer_module()
    assert infer._predictions_to_scenes([], threshold=0.5) == []


def test_predictions_to_scenes_all_transition_frames_still_yields_one_shot():
    """TransNetV2's own fallback (which this is a faithful port of): a file
    that unambiguously has content must not report zero shots just because
    every frame cleared the bar."""
    infer = _infer_module()
    assert infer._predictions_to_scenes([0.9, 0.9, 0.9], threshold=0.5) == [[0, 2]]


def test_predictions_to_scenes_threshold_is_strictly_greater_than():
    """A probability sitting exactly ON the threshold must read as 'not a
    transition' (upstream uses `>`, not `>=`) — this port must not silently
    split one more shot than the reference implementation would."""
    infer = _infer_module()
    assert infer._predictions_to_scenes([0.1, 0.5, 0.1], threshold=0.5) == [[0, 2]]


# --- the child's decode seam (fake PyAV, no real video) -------------------------

class _FakeFrame:
    def __init__(self, value):
        self._value = value

    def reformat(self, width, height, format):
        assert (width, height, format) == (48, 27, 'rgb24')
        return self

    def to_ndarray(self, format):
        import numpy as np
        return np.full((27, 48, 3), self._value, dtype='uint8')


class _FakeStream:
    def __init__(self, average_rate):
        self.average_rate = average_rate


class _FakeContainer:
    def __init__(self, frame_values, average_rate=25.0, has_video=True):
        self._frame_values = frame_values
        video = [_FakeStream(average_rate)] if has_video else []
        self.streams = type('S', (), {'video': video})()
        self.closed = False

    def decode(self, stream):
        return (_FakeFrame(v) for v in self._frame_values)

    def close(self):
        self.closed = True


def test_read_frames_decodes_once_and_measures_fps(monkeypatch):
    infer = _infer_module()
    container = _FakeContainer([10, 20, 30], average_rate=29.97)
    monkeypatch.setattr(infer, '_open', lambda path: container)

    frames, fps_native, frame_count = infer._read_frames('/src/a.mp4')

    assert frame_count == 3
    assert fps_native == pytest.approx(29.97)
    assert frames.shape == (3, 27, 48, 3)
    assert container.closed is True


def test_read_frames_with_no_video_stream_raises(monkeypatch):
    """An .mp4 holding only audio opens fine and has nothing to cut."""
    infer = _infer_module()
    monkeypatch.setattr(infer, '_open',
                        lambda path: _FakeContainer([], has_video=False))
    with pytest.raises(Exception):
        infer._read_frames('/src/audio_only.mp4')


def test_read_frames_a_container_that_cannot_open_propagates(monkeypatch):
    def boom(path):
        raise OSError('moov atom not found')
    infer = _infer_module()
    monkeypatch.setattr(infer, '_open', boom)
    with pytest.raises(OSError):
        infer._read_frames('/src/broken.mp4')


# --- per-file isolation: _detect_one never raises -------------------------------

def test_detect_one_never_raises_on_a_broken_file(monkeypatch):
    infer = _infer_module()

    def boom(path):
        raise OSError('moov atom not found')
    monkeypatch.setattr(infer, '_read_frames', boom)

    row = infer._detect_one('/src/broken.mp4', model=object(), threshold=0.5)

    assert row['state'] == 'error'
    assert row['path'] == '/src/broken.mp4'
    assert 'moov atom' in row['error']
    assert row['shots'] == []


def test_detect_one_reports_shots_frame_count_and_fps(monkeypatch):
    infer = _infer_module()
    monkeypatch.setattr(infer, '_read_frames',
                        lambda path: ('FAKE_FRAMES', 25.0, 10))
    monkeypatch.setattr(infer, '_run_model',
                        lambda model, frames: ([0.1] * 5 + [0.9] + [0.1] * 4,
                                               [0.0] * 10))

    row = infer._detect_one('/src/a.mp4', model=object(), threshold=0.5)

    assert row == {'path': '/src/a.mp4', 'state': 'ok',
                   'shots': [[0, 5], [6, 9]], 'frame_count': 10,
                   'fps_native': 25.0, 'error': None, 'probs': None}


# --- the child's stdin/stdout protocol ------------------------------------------

def _run_main(monkeypatch, capsys, job, *, load_ok=True, rows=None):
    infer = _infer_module()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(job)))
    if load_ok:
        monkeypatch.setattr(infer, '_load_model', lambda device: object())
    else:
        def fail(device):
            raise ModuleNotFoundError("No module named 'transnetv2_pytorch'")
        monkeypatch.setattr(infer, '_load_model', fail)
    if rows is not None:
        it = iter(rows)
        monkeypatch.setattr(infer, '_detect_one',
                            lambda path, model, threshold, emit_probs=False:
                            next(it))
    rc = infer.main()
    return infer, rc


def test_main_emits_one_row_per_video_then_a_summary(monkeypatch, capsys):
    rows = [
        {'path': 'a.mp4', 'state': 'ok', 'shots': [[0, 9]], 'frame_count': 10,
         'fps_native': 24.0, 'error': None},
        {'path': 'b.mp4', 'state': 'ok', 'shots': [[0, 4]], 'frame_count': 5,
         'fps_native': 30.0, 'error': None},
    ]
    infer, rc = _run_main(monkeypatch, capsys,
                          {'videos': ['a.mp4', 'b.mp4'], 'threshold': 0.5},
                          rows=rows)

    assert rc == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert [l['path'] for l in lines[:2]] == ['a.mp4', 'b.mp4']
    assert lines[-1]['summary'] == {'ok': True, 'processed': 2, 'errors': 0,
                                    'device': 'auto'}


def test_main_a_broken_file_does_not_cost_the_rest_of_the_batch(monkeypatch, capsys):
    """The one contract this worker family exists to keep: a corrupt file
    among many must cost only itself."""
    rows = [
        {'path': 'a.mp4', 'state': 'error', 'shots': [], 'frame_count': None,
         'fps_native': None, 'error': 'RuntimeError: moov atom not found'},
        {'path': 'b.mp4', 'state': 'ok', 'shots': [[0, 9]], 'frame_count': 10,
         'fps_native': 24.0, 'error': None},
    ]
    infer, rc = _run_main(monkeypatch, capsys,
                          {'videos': ['a.mp4', 'b.mp4']}, rows=rows)

    assert rc == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert [l['state'] for l in lines[:2]] == ['error', 'ok']
    assert lines[-1]['summary'] == {'ok': True, 'processed': 2, 'errors': 1,
                                    'device': 'auto'}


def test_main_a_model_load_failure_emits_no_rows_at_all(monkeypatch, capsys):
    """A load failure IS the whole pass — the caller must be able to tell it
    apart from 'ran fine, every file happened to error', because they lead to
    opposite decisions (install hint vs. per-file retry)."""
    infer, rc = _run_main(monkeypatch, capsys, {'videos': ['a.mp4']}, load_ok=False)

    assert rc == 1
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]['summary']['ok'] is False
    assert 'transnetv2_pytorch' in lines[0]['summary']['error']


def test_main_stops_cleanly_on_cancel(monkeypatch, capsys, tmp_path):
    cancel_file = str(tmp_path / 'cancel')
    open(cancel_file, 'w', encoding='utf-8').close()
    infer, rc = _run_main(monkeypatch, capsys,
                          {'videos': ['a.mp4', 'b.mp4'], 'cancel_file': cancel_file},
                          rows=[])

    assert rc == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip()]
    # The sentinel already exists before the first video is even attempted.
    assert lines == [{'summary': {'ok': True, 'processed': 0, 'errors': 0,
                                  'device': 'auto'}}]


# --- parent: interpreter resolution & config-driven defaults -------------------

def test_detector_python_falls_back_to_the_scoring_environment(monkeypatch):
    """Not a fallback but the common case: the bank-scoring venv already
    carries torch, and a second dedicated env would cost the user a second
    copy of it for no reason."""
    values = {'shot_detect.python': '', 'bank_scoring.python': '/some/env/python'}
    monkeypatch.setattr(cfg, 'get', lambda key, *a: values.get(key, ''))
    assert sd.detector_python() == '/some/env/python'


def test_threshold_default_is_clamped_not_refused(monkeypatch):
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        5 if key == 'shot_detect.threshold' else default)
    assert sd.threshold_default() == 1.0
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        -2 if key == 'shot_detect.threshold' else default)
    assert sd.threshold_default() == 0.0
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        'nonsense' if key == 'shot_detect.threshold' else default)
    assert sd.threshold_default() == sd.DEFAULT_THRESHOLD


def test_min_shot_frames_default_never_goes_below_one(monkeypatch):
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        0 if key == 'shot_detect.min_shot_frames' else default)
    assert sd.min_shot_frames_default() == 1
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        'nonsense' if key == 'shot_detect.min_shot_frames' else default)
    assert sd.min_shot_frames_default() == sd.DEFAULT_MIN_SHOT_FRAMES


def test_device_default_falls_back_when_unset(monkeypatch):
    monkeypatch.setattr(cfg, 'get', lambda key, default=None: default)
    assert sd.device_default() == sd.DEFAULT_DEVICE


# --- parent: frame indices -> PTS seconds + the min-length filter --------------

def test_shots_to_clips_converts_frame_indices_to_pts_seconds():
    clips = sd._shots_to_clips([[0, 23]], fps_native=24.0, min_frames=1)
    assert clips == [{'start_s': 0.0, 'end_s': pytest.approx(1.0),
                      'start_frame': 0, 'end_frame': 23, 'detector': 'transnetv2'}]


def test_shots_to_clips_end_is_exclusive_of_the_next_frame():
    """24 frames at 24 fps span exactly 1.00 s — using end_frame alone (23/24)
    would under-count every clip by one frame's worth of time."""
    clips = sd._shots_to_clips([[24, 47]], fps_native=24.0, min_frames=1)
    assert clips[0]['start_s'] == pytest.approx(1.0)
    assert clips[0]['end_s'] == pytest.approx(2.0)


def test_shots_to_clips_drops_shots_shorter_than_the_floor():
    clips = sd._shots_to_clips([[0, 1], [10, 40]], fps_native=24.0, min_frames=5)
    assert len(clips) == 1
    assert clips[0]['start_frame'] == 10


def test_shots_to_clips_every_row_carries_the_detector_id():
    clips = sd._shots_to_clips([[0, 9], [10, 19]], fps_native=24.0, min_frames=1)
    assert all(c['detector'] == 'transnetv2' for c in clips)


# --- parent: the streaming subprocess protocol ----------------------------------

class _FakeProc:
    """Enough of Popen to exercise the protocol without a child."""

    def __init__(self, lines, rc=0):
        self.stdout = io.StringIO(''.join(lines))
        self.stderr = io.StringIO('')
        self.stdin = io.StringIO()
        self._rc = rc

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc

    def kill(self):
        pass


def _fake_popen(lines):
    def factory(*_a, **_kw):
        return _FakeProc(lines)
    return factory


def test_detect_shots_converts_to_pts_seconds(monkeypatch):
    lines = [
        json.dumps({'path': '/src/a.mp4', 'state': 'ok',
                    'shots': [[0, 23], [24, 47]], 'frame_count': 48,
                    'fps_native': 24.0, 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True, 'processed': 1, 'errors': 0,
                                'device': 'cpu'}}) + '\n',
    ]
    monkeypatch.setattr(sd.subprocess, 'Popen', _fake_popen(lines))

    clips = sd.detect_shots('/src/a.mp4')

    assert len(clips) == 2
    assert clips[0]['start_s'] == 0.0
    assert clips[0]['end_s'] == pytest.approx(1.0)
    assert clips[1]['start_s'] == pytest.approx(1.0)
    assert clips[1]['detector'] == 'transnetv2'


def test_detect_shots_prefers_the_callers_cached_fps(monkeypatch):
    """A caller passing an already-measured VideoSource.fps_native must win
    over whatever this decode pass measured on its own — the two should
    normally agree, but the caller's freshly-measured value is the one that
    matches its own database row if the source was re-encoded since."""
    lines = [
        json.dumps({'path': '/src/a.mp4', 'state': 'ok', 'shots': [[0, 9]],
                    'frame_count': 10, 'fps_native': 30.0, 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]
    monkeypatch.setattr(sd.subprocess, 'Popen', _fake_popen(lines))

    clips = sd.detect_shots('/src/a.mp4', fps_native=24.0)

    assert clips[0]['end_s'] == pytest.approx(10 / 24)


def test_detect_shots_raises_file_error_on_a_broken_file(monkeypatch):
    lines = [
        json.dumps({'path': '/src/broken.mp4', 'state': 'error', 'shots': [],
                    'frame_count': None, 'fps_native': None,
                    'error': 'OSError: moov atom not found'}) + '\n',
        json.dumps({'summary': {'ok': True, 'processed': 1, 'errors': 1}}) + '\n',
    ]
    monkeypatch.setattr(sd.subprocess, 'Popen', _fake_popen(lines))

    with pytest.raises(sd.ShotDetectFileError):
        sd.detect_shots('/src/broken.mp4')


def test_detect_shots_with_no_fps_anywhere_raises_file_error(monkeypatch):
    lines = [
        json.dumps({'path': '/src/a.mp4', 'state': 'ok', 'shots': [[0, 9]],
                    'frame_count': 10, 'fps_native': None, 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]
    monkeypatch.setattr(sd.subprocess, 'Popen', _fake_popen(lines))

    with pytest.raises(sd.ShotDetectFileError):
        sd.detect_shots('/src/a.mp4')


def test_detect_shots_raises_unavailable_when_the_model_never_loads(monkeypatch):
    lines = [json.dumps({'summary': {
        'ok': False, 'error': "No module named 'transnetv2_pytorch'"}}) + '\n']
    monkeypatch.setattr(sd.subprocess, 'Popen', _fake_popen(lines))

    with pytest.raises(sd.ShotDetectUnavailable):
        sd.detect_shots('/src/a.mp4')


def test_detect_shots_raises_unavailable_on_a_silent_child(monkeypatch):
    monkeypatch.setattr(sd.subprocess, 'Popen', _fake_popen([]))

    with pytest.raises(sd.ShotDetectUnavailable):
        sd.detect_shots('/src/a.mp4')


def test_detect_shots_a_result_for_the_wrong_file_is_dropped(monkeypatch):
    """The one thing this module must never do: attach one video's verdict to
    another video's database row."""
    lines = [
        json.dumps({'path': 'SOMETHING-ELSE.mp4', 'state': 'ok',
                    'shots': [[0, 9]], 'frame_count': 10, 'fps_native': 24.0,
                    'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]
    monkeypatch.setattr(sd.subprocess, 'Popen', _fake_popen(lines))

    with pytest.raises(sd.ShotDetectUnavailable):
        sd.detect_shots('/src/a.mp4')


def test_detect_shots_reports_progress_via_stderr(monkeypatch):
    class _ProgressProc(_FakeProc):
        def __init__(self):
            super().__init__([
                json.dumps({'path': '/src/a.mp4', 'state': 'ok',
                            'shots': [[0, 9]], 'frame_count': 10,
                            'fps_native': 24.0, 'error': None}) + '\n',
                json.dumps({'summary': {'ok': True}}) + '\n',
            ])
            self.stderr = io.StringIO('[shotdet] model ready\n[shotdet] 1/1 ok\n')

    monkeypatch.setattr(sd.subprocess, 'Popen', lambda *a, **kw: _ProgressProc())
    seen = []

    sd.detect_shots('/src/a.mp4', on_progress=seen.append)

    assert '[shotdet] model ready' in seen


class _CapturingStdin:
    def __init__(self, sink):
        self._sink = sink

    def write(self, data):
        self._sink['raw'] = data

    def close(self):
        pass


def test_detect_shots_sends_the_resolved_threshold_and_device(monkeypatch):
    sink = {}
    lines = [
        json.dumps({'path': '/src/a.mp4', 'state': 'ok', 'shots': [[0, 9]],
                    'frame_count': 10, 'fps_native': 24.0, 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]

    def factory(*_a, **_kw):
        proc = _FakeProc(lines)
        proc.stdin = _CapturingStdin(sink)
        return proc
    monkeypatch.setattr(sd.subprocess, 'Popen', factory)

    sd.detect_shots('/src/a.mp4', threshold=0.7, device='cuda')

    job = json.loads(sink['raw'])
    assert job == {'videos': ['/src/a.mp4'], 'threshold': 0.7, 'device': 'cuda',
                   'cancel_file': '', 'emit_probs': False}


def test_detect_shots_falls_back_to_config_defaults(monkeypatch):
    sink = {}
    lines = [
        json.dumps({'path': '/src/a.mp4', 'state': 'ok', 'shots': [[0, 9]],
                    'frame_count': 10, 'fps_native': 24.0, 'error': None}) + '\n',
        json.dumps({'summary': {'ok': True}}) + '\n',
    ]

    def factory(*_a, **_kw):
        proc = _FakeProc(lines)
        proc.stdin = _CapturingStdin(sink)
        return proc
    monkeypatch.setattr(sd.subprocess, 'Popen', factory)
    monkeypatch.setattr(cfg, 'get', lambda key, default=None: default)

    sd.detect_shots('/src/a.mp4')

    job = json.loads(sink['raw'])
    assert job['threshold'] == sd.DEFAULT_THRESHOLD
    assert job['device'] == sd.DEFAULT_DEVICE
