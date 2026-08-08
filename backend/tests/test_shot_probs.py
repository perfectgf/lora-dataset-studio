"""The per-frame probability cache — what makes a threshold free to change.

Two contracts live here. The STORE: probabilities survive the process that
computed them, round-trip whole, and a corrupt or missing file reads as "not
cached" rather than as an error. The PROTOCOL: the worker actually emits both
heads, and the row shape it emits is the row shape the parent reads — asserted
against the worker's own declared constant rather than against a hand-copied
literal, because a stub that agrees with a stale assumption stays green while
production breaks.
"""
import importlib.util
import io
import json
import os
import sys

import pytest

from app.services import shot_probs


def _infer_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'infer', 'shot_detect_infer.py')
    spec = importlib.util.spec_from_file_location('shot_detect_infer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the protocol: both heads, and one declared row shape ----------------------

def test_the_worker_declares_the_row_shape_the_parent_reads():
    """Two interpreters, one contract. The parent's reader and the worker's
    writer are compared to the same constant so neither can drift alone."""
    from app.services import shot_detect as sd
    infer = _infer_module()
    assert infer.ROW_KEYS == sd.ROW_KEYS
    assert infer.PROBS_KEYS == sd.PROBS_KEYS == ('single', 'all')


def test_the_declared_row_shape_is_what_detect_one_really_returns(monkeypatch):
    """The constant is only worth having if it is checked against the real
    function, not against another constant."""
    infer = _infer_module()
    monkeypatch.setattr(infer, '_read_frames', lambda path: ('FRAMES', 25.0, 10))
    monkeypatch.setattr(infer, '_run_model',
                        lambda model, frames: ([0.1] * 10, [0.1] * 10))
    row = infer._detect_one('/src/a.mp4', model=object(), threshold=0.5,
                            emit_probs=True)
    assert tuple(row.keys()) == infer.ROW_KEYS


def test_the_worker_emits_both_heads_when_asked(monkeypatch):
    infer = _infer_module()
    single = [0.0] * 5 + [0.9] + [0.0] * 4
    every = [0.0] * 4 + [0.6, 0.9, 0.6] + [0.0] * 3
    monkeypatch.setattr(infer, '_read_frames', lambda path: ('FRAMES', 25.0, 10))
    monkeypatch.setattr(infer, '_run_model', lambda model, frames: (single, every))

    row = infer._detect_one('/src/a.mp4', model=object(), threshold=0.5,
                            emit_probs=True)

    assert row['probs']['single'] == pytest.approx(single)
    assert row['probs']['all'] == pytest.approx(every)


def test_the_worker_stays_silent_about_probabilities_when_not_asked(monkeypatch):
    """The row shape does not change — only its content. A caller that wants
    boundaries and nothing else must not pay megabytes of JSON for them."""
    infer = _infer_module()
    monkeypatch.setattr(infer, '_read_frames', lambda path: ('FRAMES', 25.0, 10))
    monkeypatch.setattr(infer, '_run_model',
                        lambda model, frames: ([0.1] * 10, [0.1] * 10))

    row = infer._detect_one('/src/a.mp4', model=object(), threshold=0.5,
                            emit_probs=False)

    assert row['probs'] is None
    assert row['state'] == 'ok'


def test_emitted_probabilities_are_rounded_to_keep_the_line_small(monkeypatch):
    """A two-hour rush is a quarter of a million frames on one JSON line.
    Four decimals is far finer than any threshold anyone will ever set."""
    infer = _infer_module()
    monkeypatch.setattr(infer, '_read_frames', lambda path: ('FRAMES', 25.0, 2))
    monkeypatch.setattr(infer, '_run_model',
                        lambda model, frames: ([0.123456789, 0.5], [0.0, 0.0]))

    row = infer._detect_one('/src/a.mp4', model=object(), threshold=0.5,
                            emit_probs=True)

    assert row['probs']['single'] == [0.1235, 0.5]


def test_a_broken_file_carries_no_probabilities(monkeypatch):
    infer = _infer_module()

    def boom(path):
        raise OSError('moov atom not found')
    monkeypatch.setattr(infer, '_read_frames', boom)

    row = infer._detect_one('/src/broken.mp4', model=object(), threshold=0.5,
                            emit_probs=True)

    assert row['state'] == 'error' and row['probs'] is None
    assert tuple(row.keys()) == infer.ROW_KEYS


def test_main_passes_the_probability_request_down_to_each_file(monkeypatch):
    infer = _infer_module()
    seen = []
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(
        {'videos': ['a.mp4'], 'emit_probs': True})))
    monkeypatch.setattr(infer, '_load_model', lambda device: object())

    def spy(path, model, threshold, emit_probs):
        seen.append(emit_probs)
        return {k: None for k in infer.ROW_KEYS} | {'path': path, 'state': 'ok',
                                                    'shots': []}
    monkeypatch.setattr(infer, '_detect_one', spy)

    assert infer.main() == 0
    assert seen == [True]


# --- the store -----------------------------------------------------------------

def test_probabilities_round_trip_through_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    single = [0.0, 0.25, 0.9, 0.1]
    every = [0.0, 0.5, 0.8, 0.2]

    shot_probs.save_probs(1, 7, single, every)
    got = shot_probs.load_probs(1, 7)

    assert got['single'] == pytest.approx(single, abs=1e-4)
    assert got['all'] == pytest.approx(every, abs=1e-4)


def test_a_file_that_was_never_detected_has_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    assert shot_probs.load_probs(1, 999) is None


def test_a_corrupt_cache_reads_as_no_cache_not_as_an_error(tmp_path, monkeypatch):
    """The caller's answer to both is the same: run detection on this file."""
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    shot_probs.save_probs(1, 7, [0.1, 0.2], [0.1, 0.2])
    path = shot_probs.probs_path(1, 7)
    path.write_bytes(b'not an npz at all')

    assert shot_probs.load_probs(1, 7) is None


def test_a_cache_without_the_second_head_is_still_usable(tmp_path, monkeypatch):
    """The all-frames head can be absent — a source detected by a build that
    only stored the first one. Boundaries still re-threshold; only the
    transition labels are unavailable, which is the honest degradation."""
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    shot_probs.save_probs(1, 7, [0.1, 0.9, 0.1], None)

    got = shot_probs.load_probs(1, 7)

    assert got['single'] == pytest.approx([0.1, 0.9, 0.1], abs=1e-4)
    assert got['all'] is None


def test_forgetting_a_source_removes_its_cache_file(tmp_path, monkeypatch):
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    shot_probs.save_probs(1, 7, [0.1], [0.1])
    assert shot_probs.probs_path(1, 7).exists()

    shot_probs.forget(1, 7)

    assert not shot_probs.probs_path(1, 7).exists()
    assert shot_probs.load_probs(1, 7) is None


def test_forgetting_a_source_that_has_no_cache_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    shot_probs.forget(1, 404)


def test_each_source_gets_its_own_file_beside_the_banks_thumbnails(tmp_path,
                                                                   monkeypatch):
    """One file per source, not one per bank: a re-detection of ONE file must
    rewrite one small file, never a bank-wide store — the same reason a trim
    does not rewrite the embedding .npz."""
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    assert shot_probs.probs_path(1, 7) != shot_probs.probs_path(1, 8)
    assert shot_probs.probs_path(1, 7).parent.name == 'shot_probs'


def test_an_empty_vector_is_stored_and_read_back_as_empty(tmp_path, monkeypatch):
    """A file with no decodable frames is a legitimate 'detected, nothing
    there' — it must not be indistinguishable from 'never detected'."""
    monkeypatch.setattr(shot_probs, '_bank_dir', lambda bank_id: tmp_path)
    shot_probs.save_probs(1, 7, [], [])

    got = shot_probs.load_probs(1, 7)

    assert got is not None and got['single'] == []
