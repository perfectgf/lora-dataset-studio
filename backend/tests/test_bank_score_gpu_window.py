"""🗃️ Image bank — ✨ Score only holds the GPU when it actually uses it.

The scoring extra installs CPU-only torch on purpose, and the child picks its
own device (`cuda if torch.cuda.is_available()`). The parent used to take the
GPU-exclusive window regardless — so on a CPU install the pass unloaded ComfyUI
and blocked every training start for the whole run (measured at ~57 min on a
9 000-image bank) while computing on the CPU anyway. The face pass had the right
shape all along; these tests pin the same contract onto Score.
"""
from contextlib import nullcontext
from unittest.mock import patch

import pytest
from PIL import Image


def _bank(tmp_path, n=2):
    from app.services import image_bank_service as banks
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (800, 800), (10 * i, 90, 160)).save(str(src / f'a{i}.jpg'))
    bank, _added = banks.create_bank('local', 'Dump', str(src))
    return bank.id


def _capture_window(app, bank_id):
    """Run the score job with the subprocess driver stubbed, and return the
    context manager the job handed it."""
    from app.services import image_bank_service as banks
    seen = {}

    def fake_drive(job, python, script, payload, cache_path, progress_re, window):
        seen['window'] = window
        return {'ok': True, 'results': {}}, [], 0

    with patch.object(banks, '_drive_infer_subprocess', fake_drive), \
         patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
         patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
         patch.object(banks.bank_jobs, 'progress', lambda job, **kw: seen.setdefault(
             'details', []).append(kw.get('detail'))), \
         patch.object(banks.bank_jobs, 'set_stop_notice',
                      lambda job, **kw: seen.setdefault('stop', []).append(kw)):
        banks._score_job(bank_id)(object())
    return seen


def test_a_cpu_scoring_pass_never_takes_the_gpu_window(app, tmp_path):
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(tmp_path)
        with patch('app.capabilities.bank_scoring_gpu_available', lambda: False):
            seen = _capture_window(app, bank_id)
        assert isinstance(seen['window'], type(nullcontext())), \
            'a CPU pass must not unload ComfyUI or block a training start'
        # …and it says so, so an hour-long bar is not read as a hang.
        assert any('CPU' in (d or '') for d in seen['details'])
        assert banks._resolve_score_device() == ('cpu', False)


def test_a_cuda_scoring_pass_still_takes_the_gpu_window(app, tmp_path):
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(tmp_path)
        with patch('app.capabilities.bank_scoring_gpu_available', lambda: True):
            seen = _capture_window(app, bank_id)
            assert banks._resolve_score_device() == ('cuda', True)
        assert not isinstance(seen['window'], type(nullcontext())), \
            'a GPU pass must stay serialized against training and vision'
        assert any('CUDA' in (d or '') for d in seen['details'])


def test_borrowed_gpu_score_does_not_make_managed_cpu_siglip_request_cuda(app):
    from app import config
    from app.services import image_bank_service as banks

    with app.app_context():
        config.save_config({
            'bank_scoring': {'python': '/borrowed/gpu/python'},
            'bank_semantic': {
                'python': '/managed/cpu/python', 'device': 'auto'},
        })
        with patch('app.capabilities.bank_scoring_gpu_available', lambda: True), \
             patch('app.capabilities.bank_siglip2_gpu_available', lambda: False):
            assert banks._resolve_score_device() == ('cuda', True)
            assert banks._resolve_semantic_device() == ('cpu', False)


def test_every_bank_child_ignores_the_user_site_not_just_the_score_one():
    """They share ONE borrowed interpreter, so they share its user
    site-packages — and one readiness probe, which runs isolated. Score alone
    carried the flag for a wave; the semantic worker was one unrelated
    `pip install --user` away from the same failure, under a green card."""
    from app.services import image_bank_service as banks

    python = r'C:\borrowed\python.exe'
    assert banks._infer_subprocess_argv(python, banks._SCORE_SCRIPT) == [
        python, '-s', banks._SCORE_SCRIPT]
    assert banks._infer_subprocess_argv(python, banks._SEMANTIC_SCRIPT) == [
        python, '-s', banks._SEMANTIC_SCRIPT]


def test_a_busy_gpu_no_longer_refuses_a_pass_that_runs_on_the_cpu(app, tmp_path):
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(tmp_path)
        started = {}
        with patch('app.capabilities.bank_scoring_gpu_available', lambda: False), \
             patch.object(banks, '_gpu_busy_reason', lambda: 'training is running'), \
             patch.object(banks, 'probe_bank_scoring', create=True), \
             patch('app.capabilities.probe_bank_scoring', lambda: {'ok': True}), \
             patch.object(banks.bank_jobs, 'start',
                          lambda *a, **k: started.setdefault('yes', True)):
            banks.start_score(app, 'local', bank_id)
        assert started.get('yes'), 'a CPU pass has no reason to wait for the GPU'


def test_a_busy_gpu_still_refuses_a_pass_that_would_use_it(app, tmp_path):
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(tmp_path)
        with patch('app.capabilities.bank_scoring_gpu_available', lambda: True), \
             patch.object(banks, '_gpu_busy_reason', lambda: 'training is running'), \
             patch('app.capabilities.probe_bank_scoring', lambda: {'ok': True}):
            with pytest.raises(RuntimeError, match='training is running'):
                banks.start_score(app, 'local', bank_id)


def test_the_payload_says_what_score_will_run_on_and_what_it_will_cost(
        app, tmp_path):
    """The install ships CPU torch on purpose; the user still has to be told,
    with a number, and only offered the fix when the machine has a card."""
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(tmp_path, n=3)
        with patch('app.capabilities.bank_scoring_gpu_available', lambda: False), \
             patch('app.capabilities.gpu_vram_gb', lambda: 24.0):
            info = banks.bank_payload('local', bank_id)['score_device']
        assert info['device'] == 'cpu' and info['gpu'] is False
        assert info['gpu_present'] is True          # a card is there to switch to
        assert info['eta_minutes'] is not None

        with patch('app.capabilities.bank_scoring_gpu_available', lambda: True):
            info = banks.bank_payload('local', bank_id)['score_device']
        assert info['device'] == 'cuda' and info['gpu'] is True
        assert info['eta_minutes'] is None          # nothing to warn about


def test_no_gpu_on_the_machine_is_reported_as_such(app, tmp_path):
    """Without a card there is nothing to suggest — the warning must not tell a
    laptop user to install a 2.5 GB CUDA build it cannot use."""
    from app.services import image_bank_service as banks

    with app.app_context():
        bank_id = _bank(tmp_path)
        with patch('app.capabilities.bank_scoring_gpu_available', lambda: False), \
             patch('app.capabilities.gpu_vram_gb', lambda: None):
            info = banks.bank_payload('local', bank_id)['score_device']
        assert info['device'] == 'cpu' and info['gpu_present'] is False
