"""What a dead training run is allowed to claim.

Both units under test are PURE — no GPU, no ai-toolkit, no filesystem — which is
the point: nobody on this project owns a Blackwell card, so the verdict logic is
exercised with simulated probe payloads instead of hardware.

The bug that started this (wannadecryptor, Discord, RTX 5070): the panel printed
the raw tail of training.log in red, and ai-toolkit's tail on an early death is a
harmless huggingface_hub `FutureWarning`. Hours lost on a deprecation notice.
"""
from unittest.mock import patch

from app.services.training_diagnostics import (
    CU128_INDEX_URL, MAX_EXCERPT_LINES, extract_error_excerpt, gated_repo_verdict,
    torch_arch_verdict, torch_cuda_verdict, torch_reinstall_command,
)

# The actual log the user was shown, trimmed. Nothing in it is an error.
FUTUREWARNING_LOG = (
    'C:\\Users\\somebody\\ai-toolkit\\venv\\Lib\\site-packages\\huggingface_hub\\constants.py:298: '
    'FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated. '
    'Please use `HF_XET_HIGH_PERFORMANCE` instead.\n'
    '  warnings.warn(\n'
    'Running 1 job\n'
    'Dataset: C:\\Users\\somebody\\datasets\\subject\n'
    ' - Preprocessing image dimensions\n'
    'Bucket sizes for C:\\Users\\somebody\\datasets\\subject:\n'
    '1024x1024: 42 files\n'
    'Caching latents to disk:   0%|          | 0/42 [00:00<?, ?it/s]\n'
)


# --- extract_error_excerpt ------------------------------------------------

def test_a_futurewarning_is_never_reported_as_the_cause():
    out = extract_error_excerpt(FUTUREWARNING_LOG)
    assert out['kind'] == 'none'          # -> the UI renders it neutral, not red
    assert out['headline'] == ''          # -> nothing is quoted as "the reason"
    assert 'FutureWarning' in out['text']  # still shown, as context


def test_the_excerpt_is_path_redacted_for_public_help_threads():
    out = extract_error_excerpt(FUTUREWARNING_LOG)
    assert 'somebody' not in out['text']
    assert '~\\ai-toolkit' in out['text'] or '~\\datasets' in out['text']


def test_a_traceback_wins_and_its_exception_line_is_the_headline():
    log = FUTUREWARNING_LOG + (
        'Traceback (most recent call last):\n'
        '  File "run.py", line 90, in <module>\n'
        '    main()\n'
        '  File "toolkit/train.py", line 12, in cache\n'
        '    latents = vae.encode(px)\n'
        'RuntimeError: CUDA error: no kernel image is available for execution on the device\n'
    )
    out = extract_error_excerpt(log)
    assert out['kind'] == 'traceback'
    assert out['headline'].startswith('RuntimeError: CUDA error: no kernel image')
    assert out['text'].startswith('Traceback (most recent call last):')
    # The warning noise above the traceback is dropped entirely.
    assert 'FutureWarning' not in out['text']


def test_the_last_traceback_wins_over_one_the_run_survived():
    log = ('Traceback (most recent call last):\n'
           'ValueError: first, caught\n'
           'retrying\n'
           'Traceback (most recent call last):\n'
           'OSError: second, fatal\n')
    out = extract_error_excerpt(log)
    assert out['headline'] == 'OSError: second, fatal'
    assert 'ValueError' not in out['text']


def test_an_error_line_without_a_traceback_is_quoted_with_context():
    log = ('loading model\n'
           'preparing buckets\n'
           'torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n'
           'exiting\n')
    out = extract_error_excerpt(log)
    assert out['kind'] == 'error'
    assert out['headline'].startswith('torch.cuda.OutOfMemoryError')
    assert 'preparing buckets' in out['text']   # context above kept
    assert 'exiting' in out['text']             # context below kept


def test_a_warning_line_is_skipped_even_when_it_says_error():
    log = ('some_module.py:12: UserWarning: ErrorHandler is deprecated, failed lookups ignored\n'
           'still going\n')
    out = extract_error_excerpt(log)
    assert out['kind'] == 'none' and out['headline'] == ''


def test_an_empty_log_claims_nothing():
    for empty in (None, '', '   \n\n  '):
        out = extract_error_excerpt(empty)
        assert out == {'kind': 'none', 'text': '', 'headline': ''}


def test_a_carriage_return_progress_bar_does_not_become_one_huge_line():
    log = 'Caching latents:   0%|\rCaching latents:  50%|\rCaching latents: 100%|\n'
    out = extract_error_excerpt(log)
    assert out['text'].count('\n') == 2       # three states, three lines
    assert max(len(line) for line in out['text'].split('\n')) < 60


def test_a_long_traceback_is_capped_and_says_so_instead_of_truncating_silently():
    frames = ''.join(f'  File "f{i}.py", line {i}, in fn\n' for i in range(60))
    log = 'Traceback (most recent call last):\n' + frames + 'RuntimeError: boom\n'
    out = extract_error_excerpt(log)
    lines = out['text'].split('\n')
    assert len(lines) <= MAX_EXCERPT_LINES
    assert 'more lines' in out['text'] and 'training.log' in out['text']
    assert lines[-1] == 'RuntimeError: boom'   # the exception line always survives


def test_the_cap_is_a_real_cap_even_at_absurdly_small_budgets():
    log = 'Traceback (most recent call last):\n' + ''.join(f'line {i}\n' for i in range(40))
    for budget in (1, 2, 3, 5):
        out = extract_error_excerpt(log, max_lines=budget)
        assert len(out['text'].split('\n')) <= max(budget, 3), budget


# --- torch_arch_verdict ---------------------------------------------------
# Simulated probe payloads: the verdict must never need a real card.

BLACKWELL = {'torch': '2.7.1+cu126', 'cuda': '12.6', 'cuda_available': True,
             'cuda_reason': '', 'capability': [12, 0],
             'device_name': 'NVIDIA GeForce RTX 5070',
             'arch_list': ['sm_50', 'sm_80', 'sm_86', 'sm_90', 'compute_90'],
             'torchvision': '0.22.1+cu126'}
ADA = {'torch': '2.7.1+cu126', 'cuda': '12.6', 'cuda_available': True, 'cuda_reason': '',
       'capability': [8, 9], 'device_name': 'NVIDIA GeForce RTX 4090',
       'arch_list': ['sm_50', 'sm_80', 'sm_86', 'sm_90'], 'torchvision': '0.22.1+cu126'}
# The payload the probe emitted before it learned to ask whether torch can SEE
# the card (2026-09-07): no `cuda_available` field at all. An older answer is
# unknown, never a verdict.
LEGACY = {'torch': '2.7.1+cu126', 'cuda': '12.6', 'capability': [8, 9],
          'device_name': 'NVIDIA GeForce RTX 4090',
          'arch_list': ['sm_50', 'sm_80', 'sm_86', 'sm_90']}
# The silent-CPU trap, as the probe reads it (acontentsheltie, Discord, RTX 3090):
# a CPU-only wheel has no CUDA at all — `torch.version.cuda` is None …
CPU_ONLY = {'torch': '2.13.0+cpu', 'cuda': None, 'cuda_available': False,
            'cuda_reason': '', 'capability': None, 'device_name': None,
            'arch_list': [], 'torchvision': '0.28.0+cpu'}
# … while a CUDA build the driver cannot serve keeps its CUDA version and torch
# says why, as a warning at the first `is_available()`.
DRIVER_TOO_OLD = {'torch': '2.13.0+cu130', 'cuda': '13.0', 'cuda_available': False,
                  'cuda_reason': ('CUDA initialization: The NVIDIA driver on your system '
                                  'is too old (found version 12040). Please update your GPU '
                                  'driver by downloading and installing a new version.'),
                  'capability': None, 'device_name': None, 'arch_list': [],
                  'torchvision': '0.28.0+cu130'}
AMPERE = {'torch': '2.9.1+cu128', 'cuda': '12.8', 'cuda_available': True, 'cuda_reason': '',
          'capability': [8, 6], 'device_name': 'NVIDIA GeForce RTX 3090',
          'arch_list': ['sm_50', 'sm_80', 'sm_86', 'sm_90'], 'torchvision': '0.24.1+cu128'}
VENV_PYTHON = 'C:\\Users\\somebody\\ai-toolkit\\venv\\Scripts\\python.exe'


def test_blackwell_on_a_stable_wheel_is_diagnosed_with_the_cu128_remedy():
    v = torch_arch_verdict(BLACKWELL, venv_python=VENV_PYTHON)
    assert v['supported'] is False and v['blackwell'] is True
    assert v['sm'] == 'sm_120' and v['built_up_to'] == 'sm_90'
    assert 'no kernel image' in v['message'] and 'RTX 5070' in v['message']
    assert CU128_INDEX_URL in v['command'] and '--force-reinstall' in v['command']
    # `--index-url` replaces PyPI: without --no-deps pip re-resolves numpy from
    # the PyTorch index and breaks every extension compiled against numpy 2.
    assert '--no-deps' in v['command']
    assert 'somebody' not in v['command']       # paste-safe like everything else


# --- torch_cuda_verdict ---------------------------------------------------
# Can torch SEE the card at all? ai-toolkit takes its device from Accelerate,
# which picks the CPU in silence when `torch.cuda.is_available()` is False —
# three Krea 2 runs on a 3090 never put a byte on the card, ETA 300 hours.

def test_a_cpu_only_wheel_is_named_and_gets_the_cuda_build_of_the_same_torch():
    v = torch_cuda_verdict(CPU_ONLY, venv_python=VENV_PYTHON)
    assert v['available'] is False and v['cpu_build'] is True and v['cuda'] is None
    assert 'CPU-only' in v['message'] and 'trains on the CPU' in v['message']
    assert 'hundreds of hours' in v['message']
    cmd = v['command']
    assert '--no-deps' in cmd and '--force-reinstall' in cmd and CU128_INDEX_URL in cmd
    # The pair the venv already resolved everything else against — the local
    # `+cpu` tag names the build being replaced and never reaches pip.
    assert 'torch==2.13.0 torchvision==0.28.0' in cmd and '+cpu' not in cmd
    assert 'somebody' not in cmd and 'somebody' not in v['message']


def test_a_cuda_build_the_driver_cannot_serve_quotes_torch_and_offers_both_ways_out():
    v = torch_cuda_verdict(DRIVER_TOO_OLD)
    assert v['available'] is False and v['cpu_build'] is False and v['cuda'] == '13.0'
    assert 'built for CUDA 13.0' in v['message']
    assert 'driver on your system is too old' in v['message']    # torch's own words
    assert 'Update the NVIDIA driver' in v['message']
    assert 'torch==2.13.0 torchvision==0.28.0' in v['command'] and '--no-deps' in v['command']
    assert '<ai-toolkit venv python>' in v['command']            # no interpreter known


def test_a_cuda_build_with_no_reason_names_the_usual_suspects():
    v = torch_cuda_verdict({**DRIVER_TOO_OLD, 'cuda_reason': ''})
    assert 'older than that CUDA release' in v['message']
    assert 'CUDA_VISIBLE_DEVICES' in v['message']


def test_torch_reason_is_redacted_before_it_reaches_a_help_thread():
    v = torch_cuda_verdict({**DRIVER_TOO_OLD,
                            'cuda_reason': 'failed at C:\\Users\\somebody\\venv\\x.py'})
    assert 'somebody' not in v['message'] and 'somebody' not in v['reason']


def test_a_torch_that_sees_the_card_is_cleared_with_no_command():
    v = torch_cuda_verdict(AMPERE)
    assert v['available'] is True and v['command'] == ''
    assert 'RTX 3090' in v['message']


def test_the_cuda_verdict_is_none_whenever_we_simply_do_not_know():
    for payload in (None, {}, {'error': 'timeout'}, LEGACY, 'garbage', 42):
        assert torch_cuda_verdict(payload) is None, payload


def test_the_reinstall_command_pins_the_pair_or_nothing():
    both = torch_reinstall_command(VENV_PYTHON, '2.13.0+cpu', '0.28.0+cpu')
    assert 'torch==2.13.0 torchvision==0.28.0' in both and 'somebody' not in both
    # Half a pin is worse than none: `--no-deps` would let a torchvision built for
    # another torch land next to the pinned one.
    half = torch_reinstall_command(None, '2.13.0+cpu', None)
    assert '==' not in half and 'torch torchvision' in half
    assert '<ai-toolkit venv python>' in half
    assert '--no-deps' in torch_reinstall_command(None)


def test_a_supported_card_is_cleared_across_minor_versions():
    # No stable wheel ships sm_89; the RTX 4090 runs the sm_86 kernels. Flagging
    # it would be a false alarm on the most common training GPU there is.
    v = torch_arch_verdict(ADA)
    assert v['supported'] is True and v['blackwell'] is False
    assert v['command'] == '' and 'ships kernels for it' in v['message']


def test_an_sm90a_style_arch_entry_is_still_understood():
    v = torch_arch_verdict({**ADA, 'arch_list': ['sm_80', 'sm_90a']})
    assert v is not None and v['built_up_to'] == 'sm_90'


def test_an_unsupported_but_non_blackwell_card_invents_no_command():
    v = torch_arch_verdict({**BLACKWELL, 'capability': [11, 0]})
    assert v['supported'] is False and v['blackwell'] is False
    assert v['command'] == ''      # we have no remedy we can honestly name


def test_the_verdict_is_none_whenever_we_simply_do_not_know():
    unknowns = [
        None,                                             # no probe at all
        {},                                               # empty payload
        {'error': 'torch not importable'},                # probe said so
        {**BLACKWELL, 'capability': None},                # CUDA saw no device
        {**BLACKWELL, 'capability': [12]},                # malformed
        {**BLACKWELL, 'capability': ['a', 'b']},          # malformed
        {**BLACKWELL, 'arch_list': []},                   # nothing to compare to
        {**BLACKWELL, 'arch_list': ['compute_90']},       # PTX only, no cubin list
        'not a dict',
    ]
    for payload in unknowns:
        assert torch_arch_verdict(payload) is None, payload


def test_without_a_known_interpreter_the_command_stays_a_placeholder():
    v = torch_arch_verdict(BLACKWELL)
    assert '<ai-toolkit venv python>' in v['command']


# --- the probe gate: cost discipline --------------------------------------

def test_every_card_pays_the_probe_once_because_the_cpu_only_trap_is_card_agnostic(app, tmp_path):
    """Until 2026-09-07 a card below compute 10.0 skipped the probe entirely: the
    only trap known then (Blackwell kernels) could not touch it. The second trap
    — a torch that cannot see the card, ai-toolkit training on the CPU without a
    word — touches every card, so an RTX 3090 now pays the import too. Once."""
    from app import capabilities
    fake_python = tmp_path / 'python.exe'
    fake_python.write_text('')
    capabilities._torch_probe_cache.clear()
    with app.app_context():
        with patch.object(capabilities, 'gpu_compute_capability', return_value=(8, 6)), \
             patch.object(capabilities.cfg, 'aitoolkit_path', return_value=fake_python), \
             patch.object(capabilities, '_torch_probe', return_value=CPU_ONLY) as probe:
            assert capabilities.aitoolkit_torch_info() == CPU_ONLY
            assert capabilities.aitoolkit_torch_info() == CPU_ONLY
            assert probe.call_count == 1
    capabilities._torch_probe_cache.clear()


def test_an_unknown_gpu_is_not_probed_and_asserts_nothing(app):
    from app import capabilities
    with app.app_context():
        with patch.object(capabilities, 'gpu_compute_capability', return_value=None), \
             patch.object(capabilities, '_torch_probe') as probe:
            assert capabilities.aitoolkit_torch_info() is None
        probe.assert_not_called()


def test_a_failed_probe_is_not_cached_as_a_fact(app, tmp_path):
    """A cold-import timeout must be retried next time, never remembered as
    'this venv has no torch'."""
    from app import capabilities
    fake_python = tmp_path / 'python.exe'
    fake_python.write_text('')
    capabilities._torch_probe_cache.clear()
    with app.app_context():
        with patch.object(capabilities, 'gpu_compute_capability', return_value=(12, 0)), \
             patch.object(capabilities.cfg, 'aitoolkit_path', return_value=fake_python), \
             patch.object(capabilities, '_torch_probe', return_value=None):
            assert capabilities.aitoolkit_torch_info() is None
        assert capabilities._torch_probe_cache == {}


def test_a_successful_probe_is_cached_so_a_relaunch_costs_nothing(app, tmp_path):
    from app import capabilities
    fake_python = tmp_path / 'python.exe'
    fake_python.write_text('')
    capabilities._torch_probe_cache.clear()
    with app.app_context():
        with patch.object(capabilities, 'gpu_compute_capability', return_value=(12, 0)), \
             patch.object(capabilities.cfg, 'aitoolkit_path', return_value=fake_python), \
             patch.object(capabilities, '_torch_probe', return_value=BLACKWELL) as probe:
            assert capabilities.aitoolkit_torch_info() == BLACKWELL
            assert capabilities.aitoolkit_torch_info() == BLACKWELL
            assert probe.call_count == 1
    capabilities._torch_probe_cache.clear()


# --- the preflight surface ------------------------------------------------
# Catching the trap BEFORE the launch is the point: the alternative is 20 minutes
# of setup for an opaque "ai-toolkit exited 1".

def _dataset(app, n_keep=20):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, 'Arch', 'arch_trig', train_type='zimage')
    for i in range(n_keep):
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename=f'k{i}.webp', status='keep', framing='half',
            caption=f'a nice varied caption with many words #{i}'))
    svc.db.session.commit()
    return ds


def test_the_preflight_warns_before_the_launch_is_wasted(app):
    from app.config import LOCAL_USER
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app)
        with patch.object(capabilities, 'aitoolkit_torch_info', return_value=BLACKWELL):
            r = lt.training_preflight(LOCAL_USER, ds.id)
    row = next((c for c in r['checks'] if c['id'] == 'torch_arch'), None)
    assert row is not None and row['status'] == 'warn'
    assert 'sm_120' in row['detail']
    assert len(row['detail']) < 110, 'the row sits in a list on a phone — keep it short'
    assert any('no kernel image' in w and CU128_INDEX_URL in w for w in r['warnings'])
    # A warning, never a blocker: we read a venv, we do not own the truth.
    assert not r['blockers'] and r['verdict'] != 'blocked'


def test_the_preflight_stays_silent_when_the_gpu_is_fine(app):
    from app.config import LOCAL_USER
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app)
        with patch.object(capabilities, 'aitoolkit_torch_info', return_value=ADA):
            r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not any(c['id'] == 'torch_arch' for c in r['checks'])


def test_an_unknown_probe_adds_no_row_and_no_warning(app):
    from app.config import LOCAL_USER
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app)
        with patch.object(capabilities, 'aitoolkit_torch_info', return_value=None):
            r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not any(c['id'] == 'torch_arch' for c in r['checks'])


def test_a_raising_probe_never_breaks_the_preflight(app):
    from app.config import LOCAL_USER
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app)
        with patch.object(capabilities, 'aitoolkit_torch_info', side_effect=OSError('boom')):
            r = lt.training_preflight(LOCAL_USER, ds.id)
    assert r['verdict'] in ('ready', 'warnings')     # the preflight still answered


# --- the silent-CPU trap on the same surfaces -------------------------------
# acontentsheltie's three logs, replayed as the probe payload. Unlike the arch
# row this one BLOCKS, and the ack cannot waive it: ai-toolkit would not fail,
# it would train on the CPU for days.

def test_the_preflight_blocks_a_run_that_would_train_on_the_cpu(app):
    from app.config import LOCAL_USER
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app)
        with patch.object(capabilities, 'aitoolkit_torch_info', return_value=CPU_ONLY):
            r = lt.training_preflight(LOCAL_USER, ds.id)
    row = next((c for c in r['checks'] if c['id'] == 'torch_cuda'), None)
    assert row is not None and row['status'] == 'fail' and row['scope'] == 'machine'
    assert row['bypassable'] is False
    assert 'CPU' in row['detail'] and '2.13.0+cpu' in row['detail']
    assert len(row['detail']) < 110, 'the row sits in a list on a phone — keep it short'
    assert r['verdict'] == 'blocked' and r['can_override'] is False
    # The warning line carries the whole story and the paste-safe pip line.
    assert any('CPU-only' in w and '--no-deps' in w and CU128_INDEX_URL in w
               for w in r['warnings'])
    # And the arch row has nothing to add: there is no card seen to have kernels for.
    assert not any(c['id'] == 'torch_arch' for c in r['checks'])


def test_the_preflight_says_nothing_about_cuda_when_torch_sees_the_card(app):
    from app.config import LOCAL_USER
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app)
        with patch.object(capabilities, 'aitoolkit_torch_info', return_value=AMPERE):
            r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not any(c['id'] in ('torch_cuda', 'torch_arch') for c in r['checks'])
    assert r['verdict'] != 'blocked'


def test_an_older_probe_payload_adds_no_cuda_row(app):
    """A payload without `cuda_available` is an unknown, not a verdict."""
    from app.config import LOCAL_USER
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        ds = _dataset(app)
        with patch.object(capabilities, 'aitoolkit_torch_info', return_value=LEGACY):
            r = lt.training_preflight(LOCAL_USER, ds.id)
    assert not any(c['id'] == 'torch_cuda' for c in r['checks'])


TRAINING_PYTHON = 'C:\\t\\ai-toolkit\\venv\\Scripts\\python.exe'


def test_a_launch_is_refused_when_torch_cannot_see_the_card(app):
    """The gate the preflight row mirrors: same sentence, same pip line — and
    it holds for every caller of assert_interpreter_ready (image, video)."""
    import pytest
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        with patch.object(capabilities, 'aitoolkit_interpreter_report',
                          return_value={'python': TRAINING_PYTHON, 'torch': True,
                                        'alternative': ''}), \
             patch.object(capabilities, 'aitoolkit_torch_info', return_value=CPU_ONLY):
            with pytest.raises(RuntimeError) as err:
                lt.assert_interpreter_ready()
    text = str(err.value)
    assert 'CPU-only' in text and 'trains on the CPU' in text
    assert '--no-deps' in text and CU128_INDEX_URL in text


def test_a_launch_goes_through_when_torch_sees_the_card_or_nobody_knows(app):
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        for info in (AMPERE, LEGACY, None):
            with patch.object(capabilities, 'aitoolkit_interpreter_report',
                              return_value={'python': TRAINING_PYTHON, 'torch': True,
                                            'alternative': ''}), \
                 patch.object(capabilities, 'aitoolkit_torch_info', return_value=info):
                lt.assert_interpreter_ready()       # no raise


def test_a_probe_that_raises_never_blocks_a_launch(app):
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        with patch.object(capabilities, 'aitoolkit_interpreter_report',
                          return_value={'python': TRAINING_PYTHON, 'torch': True,
                                        'alternative': ''}), \
             patch.object(capabilities, 'aitoolkit_torch_info', side_effect=OSError('boom')):
            lt.assert_interpreter_ready()           # no raise


# --- the one-line summary on the Runs page --------------------------------

def test_the_runs_page_summary_quotes_the_cause_not_the_last_line():
    from app.services.lora_training import local_error_message
    err = {'rc': 1, 'log_tail': FUTUREWARNING_LOG,
           'excerpt': extract_error_excerpt(FUTUREWARNING_LOG)}
    msg = local_error_message(err)
    assert msg == 'Training crashed (exit code 1). No error line in the log.'
    assert 'FutureWarning' not in msg


def test_a_legacy_payload_without_an_excerpt_is_re_analysed():
    """States live an hour: a crash recorded before this shipped must not fall
    back to the old 'last line = the cause' lie."""
    from app.services.lora_training import local_error_message
    msg = local_error_message({'rc': 1, 'log_tail': FUTUREWARNING_LOG})
    assert msg.endswith('No error line in the log.')
    msg2 = local_error_message({'rc': 1, 'log_tail': 'boom\nRuntimeError: it died\n'})
    assert msg2 == 'Training crashed (exit code 1). RuntimeError: it died'


# --- gated Hugging Face repo: 401 is NOT 403 ------------------------------
# SurpassHR (GitHub) hit this on Krea 2. huggingface_hub prints the SAME
# sentence for both statuses — "You must have access to it and be authenticated
# to access it" — and reading the sentence instead of the code produced a public
# answer telling him to request access he already had. The two are separated
# here, on the status code, with opposite remedies.

GATED_401_LOG = (
    'Running 1 job\n'
    'Traceback (most recent call last):\n'
    '  File "~/ai-toolkit/run.py", line 90, in <module>\n'
    '    main()\n'
    'huggingface_hub.errors.GatedRepoError: 401 Client Error. (Request ID: Root=1-abc)\n'
    '\n'
    'Cannot access gated repo for url '
    'https://huggingface.co/krea/Krea-2-Turbo/resolve/main/turbo.safetensors.\n'
    'Access to model krea/Krea-2-Turbo is restricted. You must have access to it and be '
    'authenticated to access it. Please log in.\n'
)

GATED_403_LOG = (
    'huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-def)\n'
    'Cannot access gated repo for url '
    'https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors.\n'
    'Access to model black-forest-labs/FLUX.1-dev is restricted and you are not in the '
    'authorized list. Visit https://huggingface.co/black-forest-labs/FLUX.1-dev to ask '
    'for access.\n'
)


def test_a_401_is_reported_as_not_authenticated_not_as_a_licence_problem():
    v = gated_repo_verdict(GATED_401_LOG)
    assert v['status'] == 401
    assert v['repo'] == 'krea/Krea-2-Turbo'
    low = v['message'].lower()
    assert 'not authenticated' in low
    assert 'settings' in low and 'api keys' in low
    # It must NOT send the user off to request access again — that was the wrong
    # answer this whole verdict exists to stop.
    assert 'accept the model licence' not in low
    assert 'ask for access' not in low


def test_a_403_is_reported_as_a_licence_not_yet_accepted():
    v = gated_repo_verdict(GATED_403_LOG)
    assert v['status'] == 403
    assert v['repo'] == 'black-forest-labs/FLUX.1-dev'
    assert v['url'] == 'https://huggingface.co/black-forest-labs/FLUX.1-dev'
    low = v['message'].lower()
    assert 'accept the model licence' in low
    assert 'another token will not help' in low


def test_a_401_with_a_token_already_saved_says_the_token_was_rejected():
    v = gated_repo_verdict(GATED_401_LOG, token_configured=True)
    assert 'rejected' in v['message'].lower()
    assert 'expired' in v['message'].lower()


def test_a_401_with_no_token_saved_says_to_paste_one():
    v = gated_repo_verdict(GATED_401_LOG, token_configured=False)
    assert 'no hugging face token is saved' in v['message'].lower()


def test_a_log_with_no_gated_refusal_gets_no_verdict():
    for log in (FUTUREWARNING_LOG, '', None, 'RuntimeError: CUDA out of memory\n',
                'huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error\n'):
        assert gated_repo_verdict(log) is None, log


def test_the_verdict_never_echoes_a_token():
    """Training logs get pasted into public help threads verbatim."""
    leaky = GATED_401_LOG + 'headers: {"Authorization": "Bearer hf_abcdefghijklmnop"}\n'
    v = gated_repo_verdict(leaky)
    assert 'hf_abcdefghijklmnop' not in str(v)
    assert 'hf_abcdefghijklmnop' not in extract_error_excerpt(leaky)['text']


def test_the_crash_payload_carries_the_gated_verdict(tmp_path):
    """End of the chain: what the watcher stores is what the panel renders."""
    from app.services.lora_training import _crash_payload
    log = tmp_path / 'training.log'
    log.write_text(GATED_401_LOG, encoding='utf-8')
    payload = _crash_payload(str(log), dataset_id=1, rc=1)
    assert payload['hf_gated']['status'] == 401
    assert payload['hf_gated']['repo'] == 'krea/Krea-2-Turbo'


# --- the interpreter itself (GitHub #19, strouder) -------------------------------
# An `aitoolkit.python` pointing at a Python without torch: the path exists, it
# runs, every folder check passes, and every run dies on
# `ModuleNotFoundError: No module named 'torch'` while the panel blames a missing
# base model or a Hugging Face token. The fix is not a stricter filter (conda, uv
# and portable installs are all legitimate interpreters) — it is naming the path.
TORCHLESS_LOG = (
    'Traceback (most recent call last):\n'
    '  File "D:\\AI\\ai-toolkit\\run.py", line 8, in <module>\n'
    '    import torch\n'
    "ModuleNotFoundError: No module named 'torch'\n"
)

STORE_PYTHON = 'C:\\Users\\somebody\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe'


def test_a_missing_module_is_read_off_the_log():
    from app.services.training_diagnostics import missing_module_in_log
    assert missing_module_in_log(TORCHLESS_LOG) == 'torch'
    # The LAST one wins, same rule as the traceback pick.
    assert missing_module_in_log(
        TORCHLESS_LOG + "ModuleNotFoundError: No module named 'diffusers'\n") == 'diffusers'


def test_a_log_without_a_missing_module_claims_nothing():
    from app.services.training_diagnostics import missing_module_in_log
    for log in (FUTUREWARNING_LOG, '', None, 'RuntimeError: CUDA out of memory\n'):
        assert missing_module_in_log(log) == '', log


def test_the_interpreter_verdict_names_the_path():
    """The whole point: the path, on screen, makes the mistake obvious."""
    from app.services.training_diagnostics import interpreter_verdict
    v = interpreter_verdict(STORE_PYTHON, False)
    assert 'WindowsApps' in v['message'] and 'python.exe' in v['message']
    # …but never the account name: this text gets pasted into public threads.
    assert 'somebody' not in v['message']


def test_a_windows_store_path_is_named_as_such_but_never_forbidden():
    from app.services.training_diagnostics import (
        WINDOWS_STORE_NOTE, interpreter_verdict, is_windows_store_python)
    assert is_windows_store_python(STORE_PYTHON)
    v = interpreter_verdict(STORE_PYTHON, False)
    assert v['windows_store'] is True
    assert WINDOWS_STORE_NOTE in v['message']
    # A verdict only ever follows a PROVEN failed import — the shape alone blocks
    # nothing (a real Python can be installed there).
    assert interpreter_verdict(STORE_PYTHON, True) is None


def test_conda_uv_and_portable_interpreters_are_not_called_store_stubs():
    """Anti-regression: every accepted install shape stays unremarkable."""
    from app.services.training_diagnostics import is_windows_store_python
    for path in ('C:\\miniconda3\\envs\\aitk\\python.exe',
                 'D:\\AI\\ai-toolkit\\python_embeded\\python.exe',
                 'D:\\AI\\ai-toolkit\\.venv\\Scripts\\python3.11.exe',
                 '/opt/uv/envs/aitk/bin/python',
                 'C:\\Python311\\python.exe'):
        assert not is_windows_store_python(path), path


def test_a_working_venv_next_door_is_offered():
    from app.services.training_diagnostics import interpreter_verdict
    v = interpreter_verdict(STORE_PYTHON, False,
                            alternative='D:\\AI\\ai-toolkit\\venv\\Scripts\\python.exe')
    assert 'D:\\AI\\ai-toolkit\\venv\\Scripts\\python.exe' in v['message']
    assert v['alternative'].endswith('python.exe')


def test_an_unknown_probe_is_never_a_verdict():
    """A cold `import torch` behind an antivirus times out; that is not a fact."""
    from app.services.training_diagnostics import interpreter_verdict
    assert interpreter_verdict(STORE_PYTHON, None) is None
    assert interpreter_verdict(STORE_PYTHON, True) is None


def test_the_verdict_does_not_mention_a_hugging_face_token():
    """The false lead that cost hours, and that made the REAL gate ambiguous."""
    from app.services.training_diagnostics import interpreter_verdict
    for alt in ('', 'D:\\AI\\ai-toolkit\\venv\\Scripts\\python.exe'):
        msg = interpreter_verdict(STORE_PYTHON, False, alternative=alt)['message'].lower()
        # A token is mentioned only to RULE IT OUT — never as a thing to go get.
        assert 'not a hugging face token problem' in msg
        assert 'api keys' not in msg
        assert 'huggingface.co' not in msg


def test_the_crash_payload_carries_the_interpreter_verdict(tmp_path):
    """End of the chain, exactly like the gated-repo one."""
    from app import capabilities
    from app.services.lora_training import _crash_payload
    log = tmp_path / 'training.log'
    log.write_text(TORCHLESS_LOG, encoding='utf-8')
    with patch.object(capabilities, 'aitoolkit_interpreter_report',
                      return_value={'python': STORE_PYTHON, 'torch': False,
                                    'alternative': ''}):
        payload = _crash_payload(str(log), dataset_id=1, rc=1)
    assert payload['interpreter']['module'] == 'torch'
    assert payload['interpreter']['windows_store'] is True
    assert 'WindowsApps' in payload['interpreter']['message']


def test_a_normal_crash_gets_no_interpreter_verdict(tmp_path):
    from app.services.lora_training import _crash_payload
    log = tmp_path / 'training.log'
    log.write_text('RuntimeError: CUDA out of memory\n', encoding='utf-8')
    payload = _crash_payload(str(log), dataset_id=1, rc=1)
    assert 'interpreter' not in payload


# --- the fast-download accelerator (GitHub #18, bobba84) ------------------------
# `HF_HUB_ENABLE_HF_TRANSFER=1` without `hf_transfer`/`hf_xet` aborts downloads
# with something that reads like a connection failure. The app never sets that
# variable — it arrives from the machine — so recognising it IS the whole remedy.
HF_TRANSFER_LOG = (
    'Downloading krea/Krea-2-Raw ...\n'
    'Error while downloading from https://cdn-lfs.hf.co/repos/aa/bb/model.safetensors: '
    'HTTP Status 416\n'
    'Consider disabling HF_HUB_ENABLE_HF_TRANSFER for better error handling.\n'
)

HF_TRANSFER_MISSING_LOG = (
    "RuntimeError: Fast download using 'hf_transfer' is enabled "
    "(HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available in "
    "your environment.\n"
)


def test_a_dead_accelerator_is_named_and_not_blamed_on_the_network():
    from app.services.training_diagnostics import hf_transfer_verdict
    for log in (HF_TRANSFER_LOG, HF_TRANSFER_MISSING_LOG):
        v = hf_transfer_verdict(log)
        assert v is not None, log
        assert 'HF_HUB_ENABLE_HF_TRANSFER=0' in v['message']
        assert 'hf_xet' in v['message']
        assert 'network is probably fine' in v['message']


def test_the_deprecation_warning_alone_is_never_a_verdict():
    """FUTUREWARNING_LOG mentions HF_HUB_ENABLE_HF_TRANSFER in a WARNING — the
    exact "a warning shown as a cause" bug this module exists to prevent."""
    from app.services.training_diagnostics import hf_transfer_verdict
    for log in (FUTUREWARNING_LOG, '', None, GATED_401_LOG,
                'RuntimeError: CUDA out of memory\n'):
        assert hf_transfer_verdict(log) is None, log


def test_the_crash_payload_carries_the_accelerator_verdict(tmp_path):
    from app.services.lora_training import _crash_payload
    log = tmp_path / 'training.log'
    log.write_text(HF_TRANSFER_LOG, encoding='utf-8')
    payload = _crash_payload(str(log), dataset_id=1, rc=1)
    assert 'HF_HUB_ENABLE_HF_TRANSFER=0' in payload['hf_transfer']['message']


# --- the launch gate: refuse BEFORE the run, naming the path --------------------

def test_a_launch_is_refused_when_the_interpreter_cannot_import_torch(app):
    """RED before this wave: the file existed, so `is_installed()` said yes, the
    run started and ai-toolkit died on `No module named 'torch'`."""
    import pytest
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        with patch.object(capabilities, 'aitoolkit_interpreter_report',
                          return_value={'python': STORE_PYTHON, 'torch': False,
                                        'alternative': ''}):
            with pytest.raises(RuntimeError) as err:
                lt.assert_interpreter_ready()
    assert 'WindowsApps' in str(err.value)          # the path is IN the refusal
    # and the Hugging Face red herring is explicitly ruled out, not offered.
    assert 'not a hugging face token problem' in str(err.value).lower()


def test_a_launch_is_never_blocked_on_an_answer_we_do_not_have(app):
    """True AND unknown both let the run through. A cold-import timeout must not
    cost someone a training run."""
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        for state in (True, None):
            with patch.object(capabilities, 'aitoolkit_interpreter_report',
                              return_value={'python': STORE_PYTHON, 'torch': state,
                                            'alternative': ''}):
                lt.assert_interpreter_ready()       # no raise


def test_a_broken_probe_never_blocks_a_launch(app):
    from app import capabilities
    from app.services import lora_training as lt
    with app.app_context():
        with patch.object(capabilities, 'aitoolkit_interpreter_report',
                          side_effect=OSError('probe exploded')):
            lt.assert_interpreter_ready()           # no raise
