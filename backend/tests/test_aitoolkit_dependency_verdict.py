"""A venv that cannot train must not read as ready, and must be told what to run.

Two holes, both measured on the code these tests now pin, both reported by
acontentsheltie (Discord, RTX 3090):

1. The torch probe put `from accelerate import Accelerator` AND `Accelerator()`
   in one `try/except Exception: pass`. A venv with a working torch and no
   accelerate — or with an Accelerate config that raises — came back as
   `accelerator_device: null`, the verdict read the empty string as
   nothing-to-say, and the Test button went GREEN on an install that cannot
   train. ai-toolkit builds its Accelerator on the same bare call
   (`toolkit/accelerator.py`), so the probe failing IS the run failing.

2. Every `ModuleNotFoundError` was explained as an interpreter problem, under a
   headline hardcoded to "cannot import torch". For a missing `dotenv` the
   headline was false, the remedy (repoint the interpreter) was a dead end, and
   the one line that fixes it was never printed.

Pure units: no GPU, no ai-toolkit, no network. The only filesystem touched is a
tmp_path standing in for a checkout.
"""
from app.services.training_diagnostics import (
    DEPENDENCY_TITLE, INTERPRETER_TITLE, dependency_repair_command,
    has_aitoolkit_manager, interpreter_verdict, torch_cuda_verdict,
)

VENV = 'C:\\ai-toolkit\\venv\\Scripts\\python.exe'


def _seeing_payload(**over):
    """A probe answer where torch DOES see the card — the green case, so any red
    below comes from the accelerate half and nothing else."""
    payload = {'torch': '2.9.1+cu128', 'cuda': '12.8', 'cuda_available': True,
               'cuda_reason': '', 'capability': [8, 6], 'device_name': 'NVIDIA GeForce RTX 3090',
               'arch_list': ['sm_86'], 'torchvision': '0.24.1', 'accelerator_device': None,
               'accelerator_error': ''}
    payload.update(over)
    return payload


# --- 1. the null that used to pass for a green ---------------------------------

def test_accelerate_that_cannot_answer_is_not_a_green(tmp_path):
    """The exact shape measured on a venv with torch and no accelerate."""
    v = torch_cuda_verdict(
        _seeing_payload(accelerator_error="ModuleNotFoundError: No module named 'accelerate'"),
        venv_python=VENV, aitoolkit_dir=str(tmp_path))
    assert v is not None
    assert v['available'] is False, 'a venv that cannot build an Accelerator cannot train'
    # The reason travels: the user sees WHICH exception, not "something failed".
    assert 'accelerate' in v['message']
    assert 'ModuleNotFoundError' in v['message']
    # And a command, because the whole complaint was that none was ever given.
    assert v['command']


def test_a_raising_accelerate_config_reads_the_same_as_a_missing_one(tmp_path):
    """`accelerate` installed but `Accelerator()` raising is the same verdict:
    ai-toolkit calls it the same way, so the run dies either way."""
    v = torch_cuda_verdict(
        _seeing_payload(accelerator_error='ValueError: invalid distributed_type'),
        venv_python=VENV, aitoolkit_dir=str(tmp_path))
    assert v['available'] is False
    assert 'ValueError' in v['message']


def test_an_old_payload_without_the_field_keeps_its_green(tmp_path):
    """UNKNOWN still keeps the green. A cached answer from before this field
    existed must not turn into a refusal on the next launch."""
    payload = _seeing_payload()
    payload.pop('accelerator_error')
    v = torch_cuda_verdict(payload, venv_python=VENV, aitoolkit_dir=str(tmp_path))
    assert v['available'] is True


def test_a_null_device_with_no_reason_keeps_its_green(tmp_path):
    """The branch fires on a REASON the probe brought back, never on the mere
    absence of a device string."""
    v = torch_cuda_verdict(_seeing_payload(accelerator_error=''),
                           venv_python=VENV, aitoolkit_dir=str(tmp_path))
    assert v['available'] is True


def test_the_cpu_device_branch_still_wins_when_accelerate_did_answer(tmp_path):
    """Control: the pre-existing refusal is untouched, and an answer beats a
    reason — `cpu` gets the .env story, not the reinstall story."""
    v = torch_cuda_verdict(
        _seeing_payload(accelerator_device='cpu', accelerator_error='ignored'),
        venv_python=VENV, aitoolkit_dir=str(tmp_path))
    assert v['available'] is False
    assert 'ACCELERATE_USE_CPU' in v['message']
    assert not v['command'], 'nothing is installed wrong in that case'


# --- 2. which remedy, and is it alive in THIS checkout --------------------------

def test_the_manager_is_named_only_when_the_checkout_has_it(tmp_path):
    """`manager/` landed upstream on 2026-07-27. Older checkouts must not be
    handed `No module named manager` — the oldest installs are exactly the ones
    most likely to have an incomplete venv."""
    assert has_aitoolkit_manager(tmp_path) is False
    without = dependency_repair_command(str(tmp_path), VENV)
    assert 'manager install' not in without
    assert 'requirements.txt' in without

    (tmp_path / 'manager').mkdir()
    (tmp_path / 'manager' / '__main__.py').write_text('', encoding='utf-8')
    assert has_aitoolkit_manager(tmp_path) is True
    assert dependency_repair_command(str(tmp_path), VENV) == 'python -m manager install'


def test_the_pip_fallback_admits_it_installs_a_cpu_torch(tmp_path):
    """`pip install -r requirements.txt` is the command that CREATES the
    silent-CPU trap on Windows (torch is absent from ai-toolkit's requirements
    but pulled in by five of them, and PyPI serves a CPU-only wheel there). When
    we hand that line over we say so, instead of letting the next launch teach
    it."""
    v = torch_cuda_verdict(
        _seeing_payload(accelerator_error="ModuleNotFoundError: No module named 'accelerate'"),
        venv_python=VENV, aitoolkit_dir=str(tmp_path))
    assert 'CPU-only' in v['message']


def test_no_checkout_means_no_invented_command():
    assert dependency_repair_command(None, VENV) == ''
    assert dependency_repair_command('', VENV) == ''


# --- 3. the headline and the remedy for a missing package ----------------------

def test_a_missing_package_is_not_announced_as_a_torch_problem():
    v = interpreter_verdict(VENV, False, module='dotenv')
    assert v['title'] == DEPENDENCY_TITLE
    assert 'torch' not in v['title'], 'the headline used to say torch for every module'
    assert 'dotenv' in v['message']


def test_a_missing_package_gets_a_command_and_not_a_repointing():
    """The dead end: "point Settings at the Python you installed the
    requirements into" — when that IS the Python, and no other exists."""
    v = interpreter_verdict(VENV, False, module='oyaml', aitoolkit_dir='C:\\ai-toolkit')
    assert 'not the wrong interpreter' in v['message']
    assert 'Fix' in v['message'], 'the line that repairs it must be printed'
    assert v['command']


def test_torch_itself_keeps_the_interpreter_story():
    """GitHub #19 (strouder) is a real interpreter bug and its answer is
    unchanged: a Windows Store stub configured while a working venv sat next to
    run.py. No pip line there — repointing IS the fix."""
    v = interpreter_verdict(VENV, False, module='torch',
                            alternative='C:\\ai-toolkit\\venv\\Scripts\\python.exe')
    assert v['title'] == INTERPRETER_TITLE
    assert v['command'] == ''


def test_a_working_interpreter_elsewhere_still_wins_for_any_module():
    """When another interpreter provably has the module, naming it stays the
    first answer — that branch predates this change and is not weakened."""
    v = interpreter_verdict(VENV, False, module='dotenv',
                            alternative='C:\\other\\python.exe')
    assert 'A working interpreter WAS found' in v['message']


def test_no_verdict_at_all_when_the_import_was_never_proven_to_fail():
    """torch_ok True and None both mean "nothing to say" — a probe that did not
    answer is not a refusal."""
    assert interpreter_verdict(VENV, True, module='dotenv') is None
    assert interpreter_verdict(VENV, None, module='dotenv') is None
