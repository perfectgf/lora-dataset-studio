"""Install AI Toolkit from Setup: the dry-run plan, the command plan and the
resumable state machine, driven by a simulated executor (nothing downloaded)."""
import io
import json
import zipfile

import pytest

from app import capabilities, config as cfg, setup_installer
from app.services import aitoolkit_install as aitk

COMMIT = aitk.AITOOLKIT_RELEASE['commit']
DIFFUSERS = ('git+https://github.com/huggingface/diffusers.git@'
             'c943837899b16cbae2f619b8dd4f7bb6f07dd81a')


def _facts(**over):
    facts = {'docker': False, 'pinokio': False, 'os': 'Windows', 'arch': 'amd64',
             'nvidia': {'gpus': [{'name': 'NVIDIA GeForce RTX 4090', 'compute_cap': '8.9'}],
                        'driver': '581.57', 'cuda': (13, 0)},
             'git': 'git', 'configured': {'ok': False, 'detail': 'aitoolkit.dir not configured'},
             'training': False,
             'python': {'path': 'C:/Python312/python.exe', 'version': '3.12', 'download': False}}
    facts.update(over)
    return facts


@pytest.fixture
def roomy(app, monkeypatch):
    monkeypatch.setattr(aitk, '_free_gb', lambda path: (200.0, 'X:\\'))
    yield


@pytest.mark.parametrize('over, state, words', [
    ({'docker': True}, 'unsupported', 'Docker'),
    ({'nvidia': None}, 'blocked', 'NVIDIA GPU'),
    ({'os': 'Darwin', 'nvidia': None}, 'blocked', 'macOS'),
    ({'nvidia': {'gpus': [{'name': 'GTX', 'compute_cap': '8.6'}], 'driver': '531', 'cuda': (12, 1)}},
     'blocked', 'too old'),
    ({'nvidia': {'gpus': [{'name': 'RTX 5090', 'compute_cap': '12.0'}], 'driver': '566', 'cuda': (12, 8)}},
     'blocked', '50-series'),
    ({'configured': {'ok': True, 'detail': 'D:\\ai-toolkit'}}, 'installed', 'already installed'),
    ({'training': True}, 'blocked', 'Local training is running'),
    ({'arch': 'arm64'}, 'blocked', 'Windows on ARM'),
])
def test_plan_refuses_with_a_reason(roomy, tmp_path, over, state, words):
    out = aitk.plan(str(tmp_path / 'ai-toolkit'), facts=_facts(**over))
    assert out['state'] == state
    assert words in out['reason']
    assert out['commands'], 'the dry run lists the commands whatever the verdict'


def test_driver_selects_the_cuda_build_like_upstream(roomy, tmp_path):
    older = _facts(nvidia={'gpus': [{'name': 'RTX 3090', 'compute_cap': '8.6'}], 'driver': '566',
                           'cuda': (12, 8)})
    out = aitk.plan(str(tmp_path / 'aitk'), facts=older)
    assert out['state'] == 'ready' and out['cuda_build'] == 'cu126'
    assert any('--index-url https://download.pytorch.org/whl/cu126' in c for c in out['commands'])
    assert aitk.plan(str(tmp_path / 'aitk'), facts=_facts())['cuda_build'] == 'cu130'


def test_destination_rules(roomy, tmp_path):
    existing = tmp_path / 'existing'
    existing.mkdir()
    (existing / 'run.py').touch()
    assert aitk.plan(str(existing), facts=_facts())['state'] == 'existing'
    busy = tmp_path / 'busy'
    busy.mkdir()
    (busy / 'notes.txt').write_text('mine')
    out = aitk.plan(str(busy), facts=_facts())
    assert out['state'] == 'blocked' and 'not empty' in out['reason']
    inside = aitk.plan(str(cfg.REPO_ROOT / 'ai-toolkit'), facts=_facts())
    assert inside['state'] == 'blocked' and 'outside the LDS folder' in inside['reason']
    assert aitk.plan('relative/ai-toolkit', facts=_facts())['state'] == 'blocked'
    spaced = aitk.plan(str(tmp_path / 'My Tools' / 'ai toolkit'), facts=_facts())
    assert spaced['state'] == 'ready'
    assert any(f'"{tmp_path / "My Tools" / "ai toolkit" / "venv"}"' in c for c in spaced['commands'])


def test_disk_space_is_checked_and_announced(app, tmp_path, monkeypatch):
    monkeypatch.setattr(aitk, '_free_gb', lambda path: (7.9, 'C:\\'))
    out = aitk.plan(str(tmp_path / 'ai-toolkit'), facts=_facts())
    assert out['state'] == 'blocked'
    assert f'about {aitk.MIN_FREE_GB} GB free on C:\\ (7.9 GB free now)' in out['reason']


def test_commands_follow_the_pinned_source(roomy, tmp_path):
    dest = tmp_path / 'ai-toolkit'
    git = aitk.plan(str(dest), facts=_facts())['commands']
    assert git[0].startswith('git clone --branch main https://github.com/ostris/ai-toolkit.git')
    assert git[1].endswith(f'reset --hard {COMMIT}')
    torch = next(c for c in git if 'torch==' in c and '--index-url' in c)
    assert 'torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0' in torch
    assert '--no-cache-dir' in torch and torch.endswith('whl/cu130')
    deps = next(c for c in git if ' -r ' in c)
    assert str(dest / 'requirements.txt') in deps and 'torch-constraints.txt' in deps
    archive = aitk.plan(str(dest), facts=_facts(git=''))
    assert archive['source'] == 'archive'
    assert archive['commands'][0] == f'download https://github.com/ostris/ai-toolkit/archive/{COMMIT}.zip'
    assert any('requirements-without-git.txt' in c for c in archive['commands'])
    assert 'git init' in archive['update_note'] and 'git pull' in aitk.update_note('git')


def test_requirements_without_git_install_the_same_ref(tmp_path):
    (tmp_path / 'requirements_base.txt').write_text(
        f'torchao==0.10.0\n{DIFFUSERS}\n#pip install git+https://github.com/x/y.git@refs/pull/1/head\n')
    (tmp_path / 'requirements.txt').write_text('-r requirements_base.txt\nscipy==1.12.0\n')
    lines = aitk.requirements_without_git(tmp_path / 'requirements.txt')
    assert ('diffusers @ https://github.com/huggingface/diffusers/archive/'
            'c943837899b16cbae2f619b8dd4f7bb6f07dd81a.zip') in lines
    assert 'scipy==1.12.0' in lines and 'torchao==0.10.0' in lines
    assert not [line for line in lines if 'git+' in line and not line.lstrip().startswith('#')]
    (tmp_path / 'other.txt').write_text('git+https://gitlab.com/a/b.git@main\n')
    with pytest.raises(aitk.InstallError, match='needs Git'):
        aitk.requirements_without_git(tmp_path / 'other.txt')


class FakeRunner:
    """Simulates git/venv/pip by their visible effects; `fail` maps a step to a code."""

    def __init__(self, fail=None, cancel_on=None, on_deps=None, fail_first=()):
        self.calls, self.fail, self.cancel_on, self.on_deps = [], fail or {}, cancel_on, on_deps
        self.fail_first = set(fail_first)

    @staticmethod
    def kind(argv):
        a = [str(x) for x in argv]
        if 'clone' in a:
            return 'clone'
        if 'reset' in a:
            return 'reset'
        if a[1:3] == ['-m', 'venv']:
            return 'venv'
        if 'pip' in a and '--upgrade' in a and a[-1] == 'pip':
            return 'pip-upgrade'
        if 'pip' in a and '--index-url' in a:
            return 'torch'
        if 'pip' in a and '-r' in a:
            return 'deps'
        if '-c' in a:
            return 'imports'
        return 'other'

    def run(self, argv, *, cwd=None, env=None):
        kind = self.kind(argv)
        self.calls.append((kind, [str(x) for x in argv], env))
        if kind == self.cancel_on:
            raise aitk.Cancelled()
        if kind in self.fail:
            return self.fail[kind]
        if kind in self.fail_first:
            self.fail_first.discard(kind)
            return 1
        a = [str(x) for x in argv]
        if kind == 'clone':
            staging = aitk.Path(a[-1])
            (staging / '.git' / 'info').mkdir(parents=True)
            (staging / '.git' / 'info' / 'exclude').write_text('# git ls-files --others\n')
            (staging / 'run.py').write_text('# ai-toolkit\n')
            (staging / 'requirements.txt').write_text(f'{DIFFUSERS}\nscipy==1.12.0\n')
        if kind == 'venv':
            python = aitk.venv_python(aitk.Path(a[-1]).parent)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text('')
        if kind == 'deps' and self.on_deps:
            self.on_deps()
        return 0

    def kinds(self):
        return [k for k, _a, _e in self.calls]


GOOD_PROBE = {'torch': '2.13.0+cu130', 'cuda': '13.0', 'cuda_available': True,
              'capability': [8, 9], 'device_name': 'NVIDIA GeForce RTX 4090',
              'accelerator_device': 'cuda', 'accelerator_error': ''}


def _install(dest, runner, facts=None, logs=None, **kw):
    logs = [] if logs is None else logs
    return aitk.install(str(dest), log=logs.append, runner=runner, facts=facts or _facts(),
                        find_python=lambda: 'C:/Python312/python.exe',
                        probe_torch=lambda py, cwd: dict(GOOD_PROBE), **kw)


def test_git_install_configures_the_trainer_and_probes_again(roomy, tmp_path):
    dest = tmp_path / 'Tools Folder' / 'ai-toolkit'
    runner, logs = FakeRunner(), []
    assert _install(dest, runner, logs=logs) == 0
    assert runner.kinds() == ['clone', 'reset', 'venv', 'pip-upgrade', 'torch', 'deps', 'imports']
    python = aitk.venv_python(dest)
    assert cfg.get('aitoolkit.dir') == str(dest) and cfg.get('aitoolkit.python') == str(python)
    assert capabilities.probe_aitoolkit()['ok']
    journal = aitk.load_journal()
    assert journal['state'] == 'done' and journal['done'] == list(aitk.STEPS)
    assert (dest / aitk.STAMP).is_file()
    assert f'/{aitk.STAMP}' in (dest / '.git' / 'info' / 'exclude').read_text()
    torch_env = next(e for k, _a, e in runner.calls if k == 'torch')
    assert torch_env['TMP'].startswith(str(dest)), 'pip downloads stay on the destination drive'
    constraints = (cfg.data_dir() / 'aitoolkit-install' / 'torch-constraints.txt').read_text()
    assert 'torch==2.13.0+cu130' in constraints
    assert any('PyTorch sees NVIDIA GeForce RTX 4090' in line for line in logs)


def test_a_failed_step_resumes_and_rebuilds_a_half_installed_torch(roomy, tmp_path):
    dest = tmp_path / 'ai-toolkit'
    with pytest.raises(aitk.InstallError, match='PyTorch could not be installed'):
        _install(dest, FakeRunner(fail={'torch': 1}))
    journal = aitk.load_journal()
    assert journal['state'] == 'error' and journal['done'] == ['source', 'python']
    assert not cfg.get('aitoolkit.dir'), 'nothing is configured before the checks pass'
    again = aitk.plan(str(dest), facts=_facts())
    assert again['state'] == 'ready' and again['resume']['done'] == ['source', 'python']
    runner = FakeRunner()
    _install(dest, runner)
    assert runner.kinds() == ['venv', 'pip-upgrade', 'torch', 'deps', 'imports']
    assert cfg.get('aitoolkit.dir') == str(dest)


def test_a_python_that_cannot_make_a_venv_falls_back_to_the_lds_python(roomy, tmp_path):
    runner = FakeRunner(fail_first={'venv'})
    _install(tmp_path / 'ai-toolkit', runner, fallback_python=lambda: 'C:/LDS/python-3.12/python.exe')
    bases = [a[0] for k, a, _e in runner.calls if k == 'venv']
    assert bases == ['C:/Python312/python.exe', 'C:/LDS/python-3.12/python.exe']
    assert cfg.get('aitoolkit.dir') == str(tmp_path / 'ai-toolkit')


def test_cancel_keeps_completed_steps(roomy, tmp_path, monkeypatch):
    dest = tmp_path / 'ai-toolkit'
    with pytest.raises(aitk.Cancelled):
        _install(dest, FakeRunner(cancel_on='deps'))
    assert aitk.load_journal()['state'] == 'cancelled'
    # PyTorch already takes its space: resuming must not demand the full amount again.
    monkeypatch.setattr(aitk, '_free_gb', lambda path: (6.0, 'X:\\'))
    assert aitk.plan(str(dest), facts=_facts())['state'] == 'ready'
    runner = FakeRunner()
    _install(dest, runner)
    assert runner.kinds() == ['deps', 'imports']


def test_a_trainer_configured_meanwhile_is_never_replaced(roomy, tmp_path):
    mine = tmp_path / 'my-toolkit'
    python = aitk.venv_python(mine)
    python.parent.mkdir(parents=True)
    python.write_text('')
    (mine / 'run.py').write_text('')
    logs = []
    runner = FakeRunner(on_deps=lambda: cfg.save_config(
        {'aitoolkit': {'dir': str(mine), 'python': str(python)}}))
    _install(tmp_path / 'ai-toolkit', runner, logs=logs)
    assert cfg.get('aitoolkit.dir') == str(mine)
    assert any('LDS kept that choice' in line for line in logs)


def test_without_git_the_pinned_archive_is_installed(roomy, tmp_path):
    dest = tmp_path / 'ai-toolkit'
    top = f'ai-toolkit-{COMMIT}/'

    def fetch(url, target, *, on_bytes, cancel_event):
        assert url == aitk.archive_url()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as bundle:
            bundle.writestr(top + 'run.py', '# ai-toolkit\n')
            bundle.writestr(top + 'requirements.txt', '-r requirements_base.txt\n')
            bundle.writestr(top + 'requirements_base.txt', f'{DIFFUSERS}\n')
        target.write_bytes(buffer.getvalue())
        on_bytes(len(buffer.getvalue()))

    runner = FakeRunner()
    _install(dest, runner, facts=_facts(git=''), fetch=fetch)
    assert 'clone' not in runner.kinds() and (dest / 'run.py').is_file()
    deps = next(a for k, a, _e in runner.calls if k == 'deps')
    flattened = aitk.Path(deps[deps.index('-r') + 1]).read_text()
    assert 'diffusers @ https://github.com/huggingface/diffusers/archive/' in flattened
    assert json.loads((dest / aitk.STAMP).read_text())['source'] == 'archive'


def test_process_runner_cancel_stops_the_running_child(tmp_path):
    import sys
    import threading
    import time
    event, lines = threading.Event(), []
    runner = aitk.ProcessRunner(lines.append, event)
    started = time.monotonic()

    def stop():
        time.sleep(0.5)
        event.set()
        runner.kill()

    threading.Thread(target=stop, daemon=True).start()
    with pytest.raises(aitk.Cancelled):
        runner.run([sys.executable, '-c', 'import time; print("started", flush=True); time.sleep(60)'])
    assert time.monotonic() - started < 20 and lines == ['started']


def test_setup_worker_streams_steps_and_cancel_reaches_the_child(app, monkeypatch, tmp_path):
    import threading
    import time
    stopped = threading.Event()

    class Child:
        def kill(self):
            stopped.set()

    def fake_install(dest, *, log, step, progress, cancel_event, register, **_hooks):
        register(Child())
        step(3, 6, 'Installing PyTorch')
        progress(1024, 0)
        log(f'into {dest}')
        assert cancel_event.wait(10)
        raise aitk.Cancelled()

    monkeypatch.setattr(aitk, 'plan', lambda dest=None, **_: {'state': 'ready', 'reason': '', 'dest': dest})
    monkeypatch.setattr(aitk, 'install', fake_install)
    monkeypatch.delitem(setup_installer._runs, 'aitoolkit_install', raising=False)
    with app.test_request_context():
        first = setup_installer.start('aitoolkit_install', {'dest': str(tmp_path / 'aitk')})
    assert first['state'] == 'running'
    for _ in range(100):
        status = setup_installer.status('aitoolkit_install')
        if status['step']:
            break
        time.sleep(0.05)
    assert status['step'] == {'index': 3, 'total': 6, 'label': 'Installing PyTorch'}
    assert status['progress']['done'] == 1024
    setup_installer.cancel('aitoolkit_install')
    assert stopped.wait(5)
    for _ in range(100):
        if setup_installer.status('aitoolkit_install')['state'] == 'cancelled':
            break
        time.sleep(0.05)
    assert setup_installer.status('aitoolkit_install')['state'] == 'cancelled'
    setup_installer._runs.pop('aitoolkit_install', None)


def test_http_plan_is_a_dry_run_and_start_refuses_with_its_reason(app, monkeypatch, tmp_path):
    monkeypatch.setattr(aitk, 'machine_facts', lambda: _facts(nvidia=None))
    monkeypatch.delitem(setup_installer._runs, 'aitoolkit_install', raising=False)
    client = app.test_client()
    dest = str(tmp_path / 'ai-toolkit')
    plan = client.get('/api/setup/aitoolkit-install/plan', query_string={'dest': dest})
    body = plan.get_json()
    assert plan.status_code == 200 and body['state'] == 'blocked' and 'NVIDIA GPU' in body['reason']
    assert body['commands'] and not (tmp_path / 'ai-toolkit').exists()
    started = client.post('/api/setup/install/aitoolkit_install', json={'dest': dest})
    assert started.status_code == 400 and 'NVIDIA GPU' in started.get_json()['error']
    assert 'aitoolkit_install' not in setup_installer._runs


def test_setup_installer_admits_it_as_a_cancellable_action(app, monkeypatch):
    monkeypatch.delitem(setup_installer._runs, 'aitoolkit_install', raising=False)
    assert setup_installer.known_action('aitoolkit_install')
    assert 'aitoolkit_install' not in setup_installer.install_all_plan({})
    assert not setup_installer._is_pip_action('aitoolkit_install')
    status = setup_installer.status('aitoolkit_install')
    assert status['state'] == 'idle' and status['step'] is None
    assert setup_installer.cancel('aitoolkit_install')['state'] == 'idle'
    monkeypatch.setattr(aitk, 'plan', lambda dest=None, **_: {'state': 'blocked', 'reason': 'no GPU here'})
    with app.test_request_context():
        with pytest.raises(setup_installer.Precondition, match='no GPU here'):
            setup_installer.start('aitoolkit_install', {'dest': 'D:/ai-toolkit'})
    assert 'aitoolkit_install' not in setup_installer._runs
