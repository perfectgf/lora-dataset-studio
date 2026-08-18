"""`detect_source` — the call that runs the detector AND brings the vector back.

`detect_shots` stays what it was (clips, nothing else) because callers and
tests depend on it. Everything the re-thresholding features need travels
through this second entry point instead: the clips, the probabilities that
produced them, and the fps used to convert.

The rule pinned hardest here: when the worker hands back probabilities, the
clips are rebuilt FROM those probabilities rather than from the worker's own
boundary list. Two ports of one rule exist; letting each feed a different half
of the app is how a re-threshold ends up disagreeing with the pass that filled
its cache.
"""
import io
import json

import pytest

from app import config as cfg
from app.services import shot_detect as sd


class _FakeProc:
    def __init__(self, lines):
        self.stdout = io.StringIO(''.join(lines))
        self.stderr = io.StringIO('')
        self.stdin = io.StringIO()

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _CapturingStdin:
    def __init__(self, sink):
        self._sink = sink

    def write(self, data):
        self._sink['raw'] = data

    def close(self):
        pass


def _worker(monkeypatch, row, sink=None):
    lines = [json.dumps(row) + '\n',
             json.dumps({'summary': {'ok': True}}) + '\n']

    def factory(*_a, **_kw):
        proc = _FakeProc(lines)
        if sink is not None:
            proc.stdin = _CapturingStdin(sink)
        return proc
    monkeypatch.setattr(sd.subprocess, 'Popen', factory)


def _row(single, every=None, fps=24.0, shots=None):
    return {'path': '/src/a.mp4', 'state': 'ok',
            'shots': shots if shots is not None else [[0, len(single) - 1]],
            'frame_count': len(single), 'fps_native': fps, 'error': None,
            'probs': {'single': single, 'all': every}}


def test_detect_source_returns_clips_the_vector_and_the_fps_it_used(monkeypatch):
    single = [0.0] * 5 + [0.9] + [0.0] * 4
    _worker(monkeypatch, _row(single, [0.0] * 10))

    out = sd.detect_source('/src/a.mp4', min_shot_frames=1)

    assert [c['start_frame'] for c in out['clips']] == [0, 6]
    assert out['probs']['single'] == pytest.approx(single)
    assert out['fps_native'] == 24.0


def test_detect_source_asks_the_worker_for_the_probabilities(monkeypatch):
    sink = {}
    _worker(monkeypatch, _row([0.0] * 10, [0.0] * 10), sink)

    sd.detect_source('/src/a.mp4')

    assert json.loads(sink['raw'])['emit_probs'] is True


def test_clips_are_rebuilt_from_the_vector_not_from_the_workers_shot_list(monkeypatch):
    """The worker's own list is ignored when a vector is available. Here the
    two disagree on purpose: only a parse of the VECTOR gives two clips."""
    single = [0.0] * 5 + [0.9] + [0.0] * 4
    _worker(monkeypatch, _row(single, [0.0] * 10, shots=[[0, 9]]))

    out = sd.detect_source('/src/a.mp4', min_shot_frames=1)

    assert len(out['clips']) == 2


def test_without_a_vector_the_workers_own_boundaries_are_used(monkeypatch):
    """A worker that was asked not to emit probabilities — or an older one —
    still produces clips. It just produces no labels and fills no cache."""
    row = _row([], shots=[[0, 23], [24, 47]], fps=24.0)
    row['probs'] = None
    row['frame_count'] = 48
    _worker(monkeypatch, row)

    out = sd.detect_source('/src/a.mp4', want_probs=False)

    assert [c['start_frame'] for c in out['clips']] == [0, 24]
    assert out['probs'] is None
    assert out['clips'][0]['end_s'] == pytest.approx(1.0)


def test_detect_source_labels_the_transitions_from_the_second_head(monkeypatch):
    single = [0.0] * 20
    single[10] = 0.9
    every = [0.0] * 20
    for i in range(4, 17):
        every[i] = 0.6
    _worker(monkeypatch, _row(single, every))

    out = sd.detect_source('/src/a.mp4', min_shot_frames=1)

    assert out['clips'][0]['transition']['end'] == {'kind': 'dissolve',
                                                    'width': 13}


def test_the_minimum_length_is_a_duration_converted_by_this_files_fps(monkeypatch):
    """0.6 s at 24 fps is 14 frames, so a 6-frame shot goes and a 30-frame one
    stays — a floor expressed in frames could not tell those apart across two
    files shot at different rates."""
    single = [0.0] * 60
    single[5] = 0.9
    _worker(monkeypatch, _row(single, fps=24.0))
    monkeypatch.setattr(cfg, 'get', lambda key, default=None: default)

    out = sd.detect_source('/src/a.mp4')

    assert [c['start_frame'] for c in out['clips']] == [6]


def test_the_legacy_frame_floor_is_still_honoured_when_it_is_the_only_one_set(
        monkeypatch):
    """`shot_detect.min_shot_frames` is in user config files today. It keeps
    working, with no rename and no silent change of meaning."""
    values = {'shot_detect.min_shot_frames': 5}
    monkeypatch.setattr(cfg, 'get',
                        lambda key, default=None: values.get(key, default))
    single = [0.0] * 60
    single[5] = 0.9
    _worker(monkeypatch, _row(single, fps=24.0))

    out = sd.detect_source('/src/a.mp4')

    assert [c['start_frame'] for c in out['clips']] == [0, 6]


def test_the_short_shot_policy_can_glue_the_sliver_on_instead_of_dropping_it(
        monkeypatch):
    values = {'shot_detect.short_shot_policy': 'merge'}
    monkeypatch.setattr(cfg, 'get',
                        lambda key, default=None: values.get(key, default))
    single = [0.0] * 60
    single[5] = 0.9
    _worker(monkeypatch, _row(single, fps=24.0))

    out = sd.detect_source('/src/a.mp4')

    assert [c['start_frame'] for c in out['clips']] == [0]
    assert out['clips'][0]['end_frame'] == 59


def test_an_explicit_threshold_reaches_both_the_worker_and_the_rebuild(monkeypatch):
    sink = {}
    single = [0.0] * 4 + [0.6] + [0.0] * 4 + [0.95] + [0.0] * 4
    _worker(monkeypatch, _row(single), sink)

    out = sd.detect_source('/src/a.mp4', threshold=0.8, min_shot_frames=1)

    assert json.loads(sink['raw'])['threshold'] == 0.8
    assert len(out['clips']) == 2


def test_a_broken_file_still_raises_the_per_file_error(monkeypatch):
    _worker(monkeypatch, {'path': '/src/a.mp4', 'state': 'error', 'shots': [],
                          'frame_count': None, 'fps_native': None,
                          'error': 'OSError: moov atom not found',
                          'probs': None})

    with pytest.raises(sd.ShotDetectFileError):
        sd.detect_source('/src/a.mp4')


# --- re-cutting a cached vector, with no worker at all -------------------------

def test_clips_from_probs_needs_no_subprocess(monkeypatch):
    """The whole point. If this ever spawns anything, the feature is a lie."""
    def forbidden(*_a, **_kw):
        raise AssertionError('a re-threshold must never start the detector')
    monkeypatch.setattr(sd.subprocess, 'Popen', forbidden)
    single = [0.0] * 5 + [0.9] + [0.0] * 4

    clips = sd.clips_from_probs({'single': single, 'all': None}, fps_native=24.0,
                                threshold=0.5, min_shot_frames=1)

    assert [c['start_frame'] for c in clips] == [0, 6]


def test_the_dry_run_answers_several_thresholds_from_one_vector():
    single = [0.0] * 4 + [0.6] + [0.0] * 4 + [0.95] + [0.0] * 4
    rows = sd.sweep_probs({'single': single}, fps_native=24.0,
                          thresholds=[0.5, 0.8], min_shot_frames=1)
    assert rows == [{'threshold': 0.5, 'shots': 3}, {'threshold': 0.8, 'shots': 2}]


# --- the settings ---------------------------------------------------------------

def test_min_shot_seconds_default_is_six_tenths_of_a_second(monkeypatch):
    monkeypatch.setattr(cfg, 'get', lambda key, default=None: default)
    assert sd.min_shot_seconds_default() == 0.6


def test_a_nonsense_min_shot_seconds_falls_back_rather_than_failing(monkeypatch):
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        'soon' if key == 'shot_detect.min_shot_seconds' else default)
    assert sd.min_shot_seconds_default() == 0.6


def test_an_unknown_short_shot_policy_reads_as_the_shipped_one(monkeypatch):
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        'staple' if key == 'shot_detect.short_shot_policy'
                        else default)
    assert sd.short_shot_policy_default() == 'drop'


def test_trimming_dissolves_is_off_unless_it_is_turned_on(monkeypatch):
    monkeypatch.setattr(cfg, 'get', lambda key, default=None: default)
    assert sd.trim_dissolves_default() is False
    monkeypatch.setattr(cfg, 'get', lambda key, default=None:
                        True if key == 'shot_detect.trim_dissolves' else default)
    assert sd.trim_dissolves_default() is True
