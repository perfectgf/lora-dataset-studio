"""The lightweight test base and optional Torch runtime are both declared files.

The bug this pins is not a crash, it is a divergence. pytest-flask was installed on
a dev machine and absent from CI, so nine tests that forgot `with app.app_context():`
passed locally and failed on the release tag — the suite was green against an
environment nobody had written down. backend/pytest.ini neutralises that particular
plugin; this keeps the environment itself from drifting again. The heavyweight
CPU Torch runtime is a separate, pinned overlay which CI path-gates and releases
always install.

Pure text checks on files in the repo — no network, no pip.
"""
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DEV = _ROOT / 'backend' / 'requirements-dev.txt'
_TORCH_TESTS = _ROOT / 'backend' / 'requirements-torch-tests.txt'
_RUNTIME = _ROOT / 'backend' / 'requirements.txt'
_CI = _ROOT / '.github' / 'workflows' / 'ci.yml'
_RELEASE = _ROOT / '.github' / 'workflows' / 'release.yml'

_TORCH_PATH_SAMPLES = (
    'backend/app/training_bridge/sitecustomize.py',
    'backend/app/job_queue.py',
    'backend/app/services/aitoolkit_state_bridge.py',
    'backend/app/services/lora_training.py',
    'backend/app/services/run_environment.py',
    'backend/app/services/training_state_bundle.py',
    'backend/app/services/training_state_identity.py',
    'backend/app/routes/training.py',
    'backend/app/services/checkpoint_registry.py',
    'backend/tests/conftest.py',
    'backend/tests/test_aitoolkit_state_bridge.py',
    'backend/tests/test_training_state_bundle.py',
    'backend/requirements-torch-tests.txt',
    '.github/workflows/ci.yml',
)


def _lines(path):
    return [l.strip() for l in path.read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.strip().startswith('#')]


def _workflow_steps(text):
    """Split top-level workflow steps without depending on a YAML package."""
    starts = list(re.finditer(r'(?m)^(?P<indent>[ \t]*)-\s+(?=(?:name|uses):)', text))
    if not starts:
        return []
    step_indent = min(len(match.group('indent').expandtabs()) for match in starts)
    starts = [match for match in starts
              if len(match.group('indent').expandtabs()) == step_indent]
    return [text[match.start():(starts[index + 1].start()
                                 if index + 1 < len(starts) else len(text))]
            for index, match in enumerate(starts)]


def _install_steps(text, requirements_name):
    return [step for step in _workflow_steps(text)
            if requirements_name in step and re.search(r'\bpip\s+install\b', step)]


def _array_regexes(text, variable):
    assignment = re.search(
        rf'(?ms)\$?{re.escape(variable)}\s*=\s*@?\((.*?)^\s*\)', text)
    assert assignment, f'{variable} must remain an explicit auditable path list'
    patterns = re.findall(r"'([^'\r\n]+)'", assignment.group(1))
    assert patterns, f'{variable} contains no path patterns'
    return patterns


def _matches_any(path, patterns):
    return any(re.search(pattern, path) for pattern in patterns)


def _inline_torch_installs(text):
    """Find literal Torch package arguments, not the declared overlay filename."""
    without_overlay = text.replace('requirements-torch-tests.txt', '')
    return re.findall(
        r'(?im)^\s*(?!#).*\bpip\s+install\b[^\r\n]*'
        r'(?<![\w./-])torch(?:\[[^\]]+\])?(?=$|[\s<>=!~\'"`])',
        without_overlay,
    )


def test_dev_requirements_exists_and_pulls_the_runtime_file():
    assert _DEV.is_file(), 'backend/requirements-dev.txt is the one test environment'
    assert '-r requirements.txt' in _lines(_DEV)


def test_pytest_is_pinned_exactly_and_only_in_the_dev_file():
    """`pytest>=8.0` floating in the RUNTIME file was both a dependency end users
    never run and a version CI and a contributor could disagree on.

    The xdist runner is pinned for the same reason and listed here rather than
    passed inline in a workflow: the distribution mode decides the execution
    ORDER, so a contributor whose xdist differs from CI's is not reproducing
    CI's run."""
    dev = _lines(_DEV)
    assert [line for line in dev if line.lower().startswith('pytest')] == [
        'pytest==9.1.1',
        'pytest-xdist==3.8.0',
    ], 'pin the audited collector exactly — another pytest is not evidence about CI'
    assert not any(l.lower().startswith('pytest') for l in _lines(_RUNTIME)), \
        'test dependencies belong in requirements-dev.txt, not in the end-user install'


def test_torch_test_overlay_is_cpu_only_official_and_exactly_pinned():
    assert _TORCH_TESTS.is_file(), 'declare heavyweight test runtime separately'
    assert _lines(_TORCH_TESTS) == [
        '--index-url https://download.pytorch.org/whl/cpu',
        'torch==2.13.0+cpu',
    ], 'the test overlay must be only the exact CPU Torch wheel from the official index'
    assert 'setuptools==83.0.0' in _lines(_DEV), \
        'preinstall the audited build backend before switching to the Torch index'
    assert not any(line.lower().startswith('torch') for line in _lines(_DEV))
    assert not any(line.lower().startswith('torch') for line in _lines(_RUNTIME))


def test_the_test_only_ml_extras_are_declared_not_inlined_in_ci():
    """numpy/OpenCV make test_watermarks.py exercise real pixel maths. They were
    two literal pip arguments inside each workflow — i.e. a second, invisible
    requirements file that only CI had."""
    dev = ' '.join(_lines(_DEV))
    assert 'numpy>=1.26,<2' in dev
    assert 'opencv-python-headless' in dev


@pytest.mark.parametrize('workflow', [_CI, _RELEASE], ids=['ci', 'release'])
def test_workflows_install_the_dev_requirements_and_nothing_else(workflow):
    text = workflow.read_text(encoding='utf-8')
    assert 'requirements-dev.txt' in text, f'{workflow.name} must install the declared test env'
    # No loose `pip install "<package>"` for test extras alongside it: that is
    # exactly the shape the divergence took.
    stray = re.findall(r'pip install[^\n]*(?:numpy|opencv|pytest-)[^\n]*', text)
    assert not stray, f'{workflow.name} installs test deps inline: {stray}'
    assert not _inline_torch_installs(text), \
        f'{workflow.name} must install Torch through requirements-torch-tests.txt'


def test_ci_path_gates_the_torch_overlay_and_forces_the_outer_heavy_gate():
    text = _CI.read_text(encoding='utf-8')
    steps = _workflow_steps(text)
    installs = _install_steps(text, 'backend/requirements-torch-tests.txt')
    assert len(installs) == 1, 'CI must have one declared Torch-overlay install step'

    install = installs[0]
    condition = re.search(r'(?m)^\s*if:\s*(.+)$', install)
    assert condition, 'CI must not download Torch for every backend run'
    output = re.search(r'steps\.([\w-]+)\.outputs\.([\w-]+)', condition.group(1))
    assert output and re.search(r"==\s*['\"]true['\"]", condition.group(1)), \
        'the overlay install must depend on a true path-decision output'
    assert '--no-cache-dir' in install, 'the large Torch wheel must bypass pip cache'

    decision_id, decision_output = output.groups()
    decisions = [step for step in steps
                 if re.search(rf'(?m)^\s*id:\s*{re.escape(decision_id)}\s*$', step)]
    assert len(decisions) == 1
    assert f'{decision_output}=' in decisions[0], \
        'the conditional install must be driven by the declared path decision'

    decision_patterns = _array_regexes(decisions[0], 'torchPathPatterns')
    gate_steps = [step for step in steps if 'torch_path_patterns=(' in step]
    assert len(gate_steps) == 1, \
        'the push gate must explicitly recognize Torch-sensitive paths'
    gate = gate_steps[0]
    gate_patterns = _array_regexes(gate, 'torch_path_patterns')
    assert gate_patterns == decision_patterns, \
        'the outer heavy gate and overlay decision must use the same path contract'

    for path in _TORCH_PATH_SAMPLES:
        assert (_ROOT / path).is_file(), f'Torch path sample does not exist: {path}'
        assert _matches_any(path, gate_patterns), f'Torch-sensitive path is ungated: {path}'
    assert not _matches_any('frontend/src/components/LightButton.jsx', gate_patterns), \
        'a small ordinary UI change must not force the heavyweight Torch path'

    override = gate.find('torch_sensitive_paths')
    threshold = gate.find('$files" -ge')
    assert 0 <= override < threshold and 'run_heavy=true' in gate[override:threshold], \
        'Torch-sensitive pushes must force heavy CI before the size threshold'
    assert 'run_heavy=false' in gate[threshold:], \
        'a below-threshold non-Torch push must still be allowed to remain light'
    assert "if ($env:EVENT_NAME -eq 'workflow_dispatch')" in decisions[0]
    assert "Write-TorchDecision $true 'Manual run'" in decisions[0]
    assert "Write-TorchDecision $true 'No usable diff base'" in decisions[0]
    assert "Write-TorchDecision $false 'No Torch-sensitive changes'" in decisions[0]


def test_ci_cache_stays_lean_while_the_torch_wheel_bypasses_it():
    text = _CI.read_text(encoding='utf-8')
    setup_steps = [step for step in _workflow_steps(text)
                   if 'actions/setup-python@' in step and 'cache-dependency-path:' in step]
    assert len(setup_steps) == 1
    cache_step = setup_steps[0]
    cache_value = cache_step.split('cache-dependency-path:', 1)[1]
    assert 'backend/requirements-dev.txt' in cache_value
    assert 'backend/requirements-torch-tests.txt' not in cache_value, \
        'UI-only jobs must not restore a shared cache populated with the Torch wheel'
    torch_installs = _install_steps(text, 'backend/requirements-torch-tests.txt')
    assert len(torch_installs) == 1 and '--no-cache-dir' in torch_installs[0]


def test_release_always_installs_the_declared_torch_overlay():
    text = _RELEASE.read_text(encoding='utf-8')
    installs = _install_steps(text, 'backend/requirements-torch-tests.txt')
    assert len(installs) == 1, 'release must install the pinned Torch overlay once'
    assert not re.search(r'(?m)^\s*if:', installs[0]), \
        'release validation must never path-skip the Torch runtime tests'
    assert '--no-cache-dir' in installs[0], 'the large Torch wheel must bypass pip cache'


def test_release_tag_is_never_interpolated_into_powershell_source():
    text = _RELEASE.read_text(encoding='utf-8')
    expression = '${{ github.ref_name }}'
    interpolations = [line.strip() for line in text.splitlines()
                      if expression in line]
    assert interpolations == [
        f'RELEASE_REF_NAME: {expression}',
        f'RELEASE_TAG: {expression}   # stamps build_info.json in the bundle',
    ], 'Git refs may enter the workflow only as environment data, never script source'
    assert "-cnotmatch '^v[0-9]{4}\\.[0-9]{2}\\.[0-9]{2}" in text
    assert 'gh release create "$env:RELEASE_REF_NAME"' in text
    assert 'persist-credentials: false' in text


def test_pytest_ini_still_blocks_the_plugin_that_caused_the_red_release():
    ini = (_ROOT / 'backend' / 'pytest.ini').read_text(encoding='utf-8')
    assert '-p no:flask' in ini, \
        'without this, a machine carrying pytest-flask runs a different suite than CI'
