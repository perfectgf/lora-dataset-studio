"""A refused launch must not pay for the freeze it will never use.

`prepare_launch` hashes every image in the dataset, probes nvidia-smi and reads
the ai-toolkit revision. It deliberately runs OUTSIDE the queue/GPU-arbiter lock
pair (doing it under the lock is what once lost cloud runs to `database is
locked`), which means the authoritative "a training is already in progress" test
— which lives inside that pair — necessarily comes after it.

The cheap flag test at the top of `launch_training` covers the common case, but a
real dataset export sits between the two, and that takes minutes: long enough for
another launch to win the process slot. So the flag is re-read immediately before
the freeze. Two in-memory reads; the lock keeps the authority.

Asserted on the SOURCE because `launch_training` cannot be driven here without a
GPU, an ai-toolkit install and a real dataset export — and the property being
protected is an ORDER of statements, which is exactly what a rewrite silently
loses.
"""
import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / 'app' / 'services' / 'lora_training.py').read_text(encoding='utf-8')


def _launch_training_body() -> str:
    # launch_training's phases moved into _lt_* functions (2026-08-24, verbatim
    # extraction). The ordering contract spans the whole pipeline, so the body
    # under test is the moved phases concatenated in CALL order, trunk last —
    # index comparisons keep exactly their old meaning (the freeze lives in
    # _lt_prepare_job, the authoritative lock pair in _lt_spawn_transaction,
    # and the trunk carries neither marker).
    parts = []
    for name in ('_lt_prepare_job', '_lt_spawn_transaction', 'launch_training'):
        start = SRC.index(f'def {name}(')
        parts.append(SRC[start:SRC.index('\ndef ', start + 1)])
    return '\n'.join(parts)


def _atomic_export_body() -> str:
    start = SRC.index('def _export_and_freeze_local_dataset(')
    end = SRC.index('\ndef ', start + 1)
    return SRC[start:end]


def test_the_busy_flag_is_re_read_before_the_expensive_freeze():
    launch = _launch_training_body()
    atomic = _atomic_export_body()
    export = atomic.index('export_dataset_to_aitoolkit(')
    prepare = atomic.index('checkpoint_registry.prepare_launch(')
    busy_during_freeze = atomic.index(
        "queue_manager._get_system_state('training_in_progress'")
    assert export < busy_during_freeze < prepare, \
        'the busy flag must be re-read after export and before snapshot hashing'
    assert launch.index('_export_and_freeze_local_dataset(') \
        < launch.index('with _queue_lock, GPU_ARBITER_LOCK:')
    # …and the authoritative copy still lives under the queue/GPU lock pair,
    # after it. The order is part of the GPU admission contract.
    lock = launch.index('with _queue_lock, GPU_ARBITER_LOCK:')
    busy = [m.start() for m in re.finditer(
        r"queue_manager\._get_system_state\('training_in_progress'", launch)]
    assert any(b > lock for b in busy), \
        'the check inside the queue/GPU lock pair is the authority and must stay'
    assert launch.index('_export_and_freeze_local_dataset(') < lock, \
        'the freeze must stay OUTSIDE the lock — holding it there is what caused ' \
        '"database is locked" on the cloud launch path'
