"""No infer worker inherits the machine's user site-packages — and neither does
the probe that vouches for it.

THE INCIDENT THIS COMES FROM. A per-user site-packages directory is shared by
every interpreter of that Python version on the machine and sits AHEAD of the
interpreter's own on ``sys.path``. An ``eventlet`` left in one by an unrelated
project — imported transitively from ``open_clip``, and predating Python 3.12 —
made the CLIP image worker fail to start 855 times in a row. Nothing was
missing: the ✨ Score interpreter had every dependency it was asked for, and the
readiness probe said so, because the PROBE already ran with ``-s`` while the
WORKER did not.

That asymmetry is the thing these tests exist to prevent. A green card above a
worker that cannot start is worse than a red one: the failure lands per ITEM,
hours later, and reads as a fact about the user's data.
"""
import os
import sys

import pytest

from app.services import infer_env


# --- the helper itself ----------------------------------------------------------

def test_the_flag_lands_on_the_interpreter_not_on_the_script():
    """``python script -s`` hands the flag to the SCRIPT, which ignores it. The
    helper takes the trailing arguments rather than a whole argv so a caller
    cannot build that silently-useless command."""
    argv = infer_env.worker_argv(r'C:\py\python.exe', 'infer/x.py')

    assert argv == [r'C:\py\python.exe', '-s', 'infer/x.py']


def test_the_environment_carries_the_same_instruction_for_the_children():
    """``-s`` reaches the worker and stops there. A Python the worker itself
    starts — or a pip inside it, which is how the junk drawer gets refilled —
    only hears about it through the environment."""
    env = infer_env.worker_env()

    assert env['PYTHONNOUSERSITE'] == '1'


def test_asking_for_a_worker_environment_never_edits_the_app_s_own():
    """Callers mutate what they get back; handing out a view of os.environ would
    let one pass rewrite the whole process — including for ComfyUI and the
    training venv, whose environment is deliberately left alone."""
    before = dict(os.environ)

    env = infer_env.worker_env(PYTHONUTF8='1')
    env['LDS_SOMETHING'] = 'x'

    assert env['PYTHONUTF8'] == '1'
    assert 'LDS_SOMETHING' not in os.environ
    assert 'PYTHONNOUSERSITE' not in os.environ or before.get('PYTHONNOUSERSITE')
    assert dict(os.environ) == before


# --- the one interpreter this must NEVER be done to -------------------------------

def test_our_own_interpreter_keeps_its_user_site_because_that_may_be_where_we_live():
    """MEASURED, not cautious. On an install whose app runs on a system-wide
    Python rather than a venv, pip cannot write to that Python's site-packages
    and silently does a --user install instead: on such a machine `import torch`
    resolves inside the user site-packages and `python -s -c "import torch"`
    raises ModuleNotFoundError.

    There the user site IS the app's dependency store, and nothing can tell its
    packages from the junk. Isolating it would not protect that install — it
    would switch every ML feature off. So the flag is for interpreters we
    BORROWED, which earned their place by importing the dependencies from their
    own site-packages (scoring_python.select proves it with `-s` before it
    writes the config key). The interpreter in the incident was a borrowed one."""
    assert not infer_env.is_borrowed(sys.executable)
    assert infer_env.worker_argv(sys.executable, 'x.py') == [sys.executable, 'x.py']
    assert 'PYTHONNOUSERSITE' not in infer_env.worker_env(sys.executable)

    assert infer_env.is_borrowed(r'C:\ComfyUI\python_embeded\python.exe')
    assert infer_env.worker_env(
        r'C:\ComfyUI\python_embeded\python.exe')['PYTHONNOUSERSITE'] == '1'


def test_the_same_interpreter_written_two_ways_is_still_ours():
    """A config that spells our own interpreter differently — a relative path, a
    different case on Windows — must not become "borrowed" and lose its
    packages."""
    assert not infer_env.is_borrowed(sys.executable.upper())
    assert not infer_env.is_borrowed(sys.executable.lower())


# --- the workers ----------------------------------------------------------------

def _spy(monkeypatch, module, attr='Popen'):
    """Record the argv and env of the next launch, and hand back a dead child."""
    seen = {}

    def fake(argv, **kwargs):
        seen['argv'] = list(argv)
        seen['env'] = kwargs.get('env')
        raise OSError('not launching anything in a test')

    monkeypatch.setattr(getattr(module, 'subprocess'), attr, fake)
    return seen


BORROWED = r'C:\ComfyUI\python_embeded\python.exe'


def _borrow(keys):
    """Point the passes at a BORROWED interpreter — the only kind the flag is
    for, and the kind the incident happened on."""
    from app import config
    out = {}
    for key in keys:
        section, _, leaf = key.partition('.')
        out.setdefault(section, {})[leaf] = BORROWED
    config.save_config(out)


def test_the_frame_encoder_runs_without_the_user_site(app, monkeypatch):
    """THE regression. This is the exact worker whose start failed 855 times."""
    from app.services import clip_image_encoder

    seen = _spy(monkeypatch, clip_image_encoder)
    with app.app_context():
        _borrow(['bank_scoring.python'])
        with pytest.raises(clip_image_encoder.ImageEncodeError):
            clip_image_encoder.ImageEncoder()._start()

    assert seen['argv'][1] == '-s'
    assert seen['argv'][2].endswith('clip_image_embed_infer.py')
    assert seen['env']['PYTHONNOUSERSITE'] == '1'


def test_the_caption_worker_runs_without_the_user_site(app, monkeypatch):
    from app.services import video_caption_worker

    seen = _spy(monkeypatch, video_caption_worker)
    with app.app_context():
        _borrow(['bank_scoring.python'])
        with pytest.raises(Exception):
            video_caption_worker.CaptionWorker()._start()

    assert seen['argv'][1] == '-s'
    assert seen['env']['PYTHONNOUSERSITE'] == '1'


def test_the_shared_infer_streamer_runs_without_the_user_site(monkeypatch):
    """One seam, three passes: the face score, the concept face mask and the
    safe-zone text reader all launch through here."""
    from app.services import infer_stream

    seen = _spy(monkeypatch, infer_stream)
    with pytest.raises(OSError):
        infer_stream.run_infer_script('py.exe', 'infer/face_score_infer.py',
                                      '{}', 5)

    assert seen['argv'] == ['py.exe', '-s', 'infer/face_score_infer.py']
    assert seen['env']['PYTHONNOUSERSITE'] == '1'


def test_the_person_mask_worker_runs_without_the_user_site(app, monkeypatch):
    from app.services import person_mask

    seen = _spy(monkeypatch, person_mask, attr='run')
    monkeypatch.setattr(person_mask, 'is_available', lambda: True)
    with app.app_context():
        _borrow(['masks.python'])
        person_mask.generate_person_masks([__file__], 'out')

    assert seen['argv'][1] == '-s'
    assert seen['env']['PYTHONNOUSERSITE'] == '1'


@pytest.mark.parametrize('module_name, attr', [
    ('shot_detect', 'Popen'),
    ('watermark_detector', 'Popen'),
])
def test_the_video_detectors_keep_their_other_environment_settings(
        app, monkeypatch, module_name, attr):
    """Isolation is ADDED to what these already set, never instead of it: both
    still need PYTHONUTF8 to print a non-ASCII path without dying."""
    import importlib

    module = importlib.import_module(f'app.services.{module_name}')
    seen = _spy(monkeypatch, module, attr=attr)
    with app.app_context():
        _borrow(['shot_detect.python', 'watermark_detect.python'])
        with pytest.raises(Exception):
            if module_name == 'shot_detect':
                module.detect_shots(['a.mp4'])
            else:
                list(module._run_chunk(['a.jpg'], device='cpu', locate=False,
                                       should_cancel=None, cancel_file=None))

    assert seen['argv'][1] == '-s'
    assert seen['env']['PYTHONNOUSERSITE'] == '1'
    assert seen['env']['PYTHONUTF8'] == '1'


# --- what is deliberately NOT isolated -------------------------------------------

def test_the_training_venv_is_launched_exactly_as_the_user_arranged_it():
    """ai-toolkit's environment is the USER's. It may legitimately carry a
    ``pip install --user``, the training bridge launches it as it stands, and
    its readiness probe is unisolated to match (test_capabilities). Sanitising
    it here would answer a question about an environment that never runs."""
    from app import capabilities

    assert 'aitoolkit_torch' in capabilities._USER_SITE_IMPORT_KEYS
    assert 'aitoolkit_alive' in capabilities._USER_SITE_IMPORT_KEYS
    # PyAV is imported IN-PROCESS by Flask, which has no `-s` of its own.
    assert 'video_decode' in capabilities._USER_SITE_IMPORT_KEYS
