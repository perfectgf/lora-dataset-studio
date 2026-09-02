"""Focused safety contract for the LDS-owned ComfyUI portable launcher."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_launcher_state(tmp_path, monkeypatch):
    """Never let these tests write the CHECKOUT's own config.json.

    Most tests in this file take only ``tmp_path`` and ``monkeypatch`` — they
    exercise the launcher directly and have no reason to build a Flask app. That
    is exactly how they escaped the isolation ``conftest.app`` sets up: without
    it, ``LDS_CONFIG`` is unset, so ``_configure``'s ``save_config`` wrote the
    REAL config.json of whatever checkout the suite ran in, stamping
    ``comfyui.base_dir`` with a ``tmp_path`` that pytest deletes afterwards.

    The bill was paid by somebody else, later: the next app started from that
    checkout booted with "A part of your setup stopped working — ComfyUI folder
    was working before and is not responding now", pointing at a directory that
    no longer existed, and nothing connected that to having run the tests.

    Autouse and file-wide on purpose, rather than a fix inside ``_configure``:
    the leak is "a test in this file may call save_config", not "this one helper
    does", and the next test added here must be safe without remembering why.
    ``_cache`` is reset for the same reason conftest does it — config.py caches
    the loaded dict in a module global that is not keyed on LDS_CONFIG, so a
    stale one would survive the redirection.
    """
    from app import config as _cfg
    from app.services import comfyui_control
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'launcher-config.json'))
    monkeypatch.setattr(_cfg, '_cache', None)
    monkeypatch.setattr(comfyui_control, '_owned_process', None)


def test_these_tests_never_write_the_checkouts_config(tmp_path):
    """The guard that keeps the fix above from being undone by accident.

    Cheap and stable: it asserts the ABSENCE of a side effect on a file the
    suite has no business touching, so it cannot flake on timing or on whether
    a real ComfyUI happens to be installed."""
    from app import config
    repo_config = config.REPO_ROOT / 'config.json'
    before = repo_config.read_bytes() if repo_config.exists() else None
    _configure(_portable_layout(tmp_path))
    after = repo_config.read_bytes() if repo_config.exists() else None
    assert after == before, (
        'a launcher test wrote the checkout config.json — the next app started '
        'from this folder will boot with a dead ComfyUI path')


def _portable_layout(tmp_path):
    """Create the only portable shape the launcher is allowed to accept."""
    bundle = tmp_path / 'ComfyUI_windows_portable'
    base = bundle / 'ComfyUI'
    base.mkdir(parents=True)
    (base / 'main.py').write_text('# test ComfyUI entrypoint\n', encoding='utf-8')
    (base / 'models').mkdir()
    embedded = bundle / 'python_embeded'
    embedded.mkdir()
    (embedded / 'python.exe').write_bytes(b'not run by this test')
    # Marker only: no test and no production path ever executes it.
    (bundle / 'run_nvidia_gpu.bat').write_text('@echo off\n', encoding='utf-8')
    return base


def _configure(base, api_url='http://127.0.0.1:8188'):
    from app import config
    config.save_config({'comfyui': {'base_dir': str(base), 'api_url': api_url}})


def _no_spawn(*_args, **_kwargs):
    raise AssertionError('the portable launcher must not spawn for this input')


@pytest.mark.parametrize('api_url', [
    'http://192.168.1.20:8188',
    'https://127.0.0.1:8188',
    'http://localhost:8188',
    'http://[::1]:8188',
    'http://127.0.0.1:9999',
    'http://127.0.0.1:8188/',
    'http://127.0.0.1:8188/other',
])
def test_refuses_remote_or_nonstandard_comfyui_url(tmp_path, monkeypatch, api_url):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _configure(base, api_url)
    monkeypatch.setattr(comfyui_control, '_spawn', _no_spawn)

    result = comfyui_control.start_comfyui(wait_timeout=0)

    assert result['ok'] is False
    assert result['reachable'] is False
    assert str(base) not in result['error']


def test_refuses_an_invalid_or_nonportable_layout(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = tmp_path / 'some-source-checkout'
    base.mkdir()
    (base / 'main.py').touch()
    (base / 'models').mkdir()
    _configure(base)
    monkeypatch.setattr(comfyui_control, '_spawn', _no_spawn)

    result = comfyui_control.start_comfyui(wait_timeout=0)

    assert result['ok'] is False
    assert result['reachable'] is False


def test_refuses_a_symlinked_embedded_python(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    executable = base.parent / 'python_embeded' / 'python.exe'
    target = base.parent / 'python_embeded' / 'python-real.exe'
    executable.replace(target)
    try:
        executable.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip('symlinks are unavailable in this test environment')
    _configure(base)
    monkeypatch.setattr(comfyui_control, '_spawn', _no_spawn)

    result = comfyui_control.start_comfyui(wait_timeout=0)

    assert result['ok'] is False


def _declare_cli_flags(base, *flags):
    """Give the fake bundle the `comfy/cli_args.py` a real ComfyUI carries, so
    the launcher can read which memory flags this install understands."""
    (base / 'comfy').mkdir(exist_ok=True)
    body = '\n'.join(f'parser.add_argument("{f}", action="store_true")' for f in flags)
    (base / 'comfy' / 'cli_args.py').write_text(body + '\n', encoding='utf-8')


def test_spawn_uses_only_the_fixed_safe_argv(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _declare_cli_flags(base, '--fast-disk', '--disable-dynamic-vram')
    _configure(base)
    calls = []

    class RunningProcess:
        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return RunningProcess()

    # First check is before the lock, second is inside it, then the first poll wins.
    history = iter([
        comfyui_control._HISTORY_DOWN,
        comfyui_control._HISTORY_DOWN,
        comfyui_control._HISTORY_READY,
    ])
    monkeypatch.setattr(comfyui_control, '_history_state', lambda: next(history))
    monkeypatch.setattr(comfyui_control.subprocess, 'Popen', fake_popen)

    result = comfyui_control.start_comfyui(wait_timeout=1, poll_interval=0.05)

    assert result == {'ok': True, 'reachable': True}
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [
        os.path.normcase(str(base.parent / 'python_embeded' / 'python.exe')), '-s', 'main.py',
        '--windows-standalone-build', '--disable-auto-launch',
        '--preview-method', 'none',
        '--listen', '127.0.0.1', '--port', '8188',
        '--fast-disk',
    ]
    # The one memory flag the launcher owns, and the one it must not: measured on
    # the video lane's 43 GB weight set, `--fast-disk` took a clip from 348 s to
    # 21-30 s by reading weights from disk instead of staging them in RAM, while
    # `--disable-dynamic-vram` left the next clip at 315 s. The launcher's
    # comment carries the numbers; this pins the decision.
    assert argv.count('--fast-disk') == 1
    assert '--disable-dynamic-vram' not in argv
    assert kwargs['shell'] is False
    assert kwargs['stdin'] is comfyui_control.subprocess.DEVNULL
    assert kwargs['stdout'] is comfyui_control.subprocess.DEVNULL
    assert kwargs['stderr'] is comfyui_control.subprocess.DEVNULL
    assert kwargs['cwd'] == os.path.normcase(str(base))
    assert kwargs['close_fds'] is True
    assert not any(str(arg).lower().endswith(('.bat', '.cmd')) for arg in argv)
    assert '--cache-none' not in argv
    assert '--disable-smart-memory' not in argv
    assert '--force-fp16' not in argv
    assert '--fast' not in argv


def test_an_older_comfyui_that_lacks_the_flag_is_launched_without_it(tmp_path, monkeypatch):
    """argparse exits on an unknown flag before the server exists, and the
    launcher would report that as "couldn't start" on every pre-May-2026 bundle.
    The flag is read off the install's own cli_args.py — absent there, or the
    file itself absent, the command line is the fixed one it always was."""
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _configure(base)
    calls = []

    class RunningProcess:
        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        calls.append(argv)
        return RunningProcess()

    history = iter([comfyui_control._HISTORY_DOWN] * 2 + [comfyui_control._HISTORY_READY] * 4)
    monkeypatch.setattr(comfyui_control, '_history_state', lambda: next(history))
    monkeypatch.setattr(comfyui_control.subprocess, 'Popen', fake_popen)

    # No cli_args.py at all (the fixture's bare layout).
    assert comfyui_control.start_comfyui(wait_timeout=1, poll_interval=0.05)['ok'] is True
    assert '--fast-disk' not in calls[-1]
    assert calls[-1][-2:] == ['--port', '8188']

    # A cli_args.py that predates the flag.
    comfyui_control._owned_process = None
    _declare_cli_flags(base, '--disable-dynamic-vram')
    history = iter([comfyui_control._HISTORY_DOWN] * 2 + [comfyui_control._HISTORY_READY] * 4)
    assert comfyui_control.start_comfyui(wait_timeout=1, poll_interval=0.05)['ok'] is True
    assert '--fast-disk' not in calls[-1]
    assert len(calls) == 2

    # A file that only MENTIONS the name — a deprecation comment, a longer
    # flag — declares nothing: the launcher looks for the argparse declaration.
    comfyui_control._owned_process = None
    (base / 'comfy' / 'cli_args.py').write_text(
        '# --fast-disk was removed\nparser.add_argument("--fast-disk-cache", action="store_true")\n',
        encoding='utf-8')
    history = iter([comfyui_control._HISTORY_DOWN] * 2 + [comfyui_control._HISTORY_READY] * 4)
    assert comfyui_control.start_comfyui(wait_timeout=1, poll_interval=0.05)['ok'] is True
    assert '--fast-disk' not in calls[-1]

    # A real declaration in a file that is not clean UTF-8 (a fork edited in a
    # legacy code page): decoded leniently, the flag is still found and the
    # launch is not the one thing that breaks on a stray byte.
    comfyui_control._owned_process = None
    (base / 'comfy' / 'cli_args.py').write_bytes(
        b'# caf\xe9\nparser.add_argument("--fast-disk", action="store_true")\n')
    history = iter([comfyui_control._HISTORY_DOWN] * 2 + [comfyui_control._HISTORY_READY] * 4)
    assert comfyui_control.start_comfyui(wait_timeout=1, poll_interval=0.05)['ok'] is True
    assert calls[-1][-1] == '--fast-disk'


def test_timeout_with_live_child_reports_starting_and_never_spawns_twice(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _configure(base)
    launches = []

    class StillStarting:
        def poll(self):
            return None

    monkeypatch.setattr(comfyui_control, '_history_state',
                        lambda: comfyui_control._HISTORY_DOWN)
    monkeypatch.setattr(comfyui_control, '_spawn',
                        lambda layout: (launches.append(layout) or StillStarting()))

    first = comfyui_control.start_comfyui(wait_timeout=0)
    second = comfyui_control.start_comfyui(wait_timeout=0)

    assert first == {'ok': True, 'reachable': False, 'starting': True}
    assert second == {'ok': True, 'reachable': False, 'already_running': True,
                      'starting': True}
    assert len(launches) == 1


def test_already_listening_history_is_a_safe_noop(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _configure(base)
    monkeypatch.setattr(comfyui_control, '_history_state',
                        lambda: comfyui_control._HISTORY_READY)
    monkeypatch.setattr(comfyui_control, '_spawn', _no_spawn)

    assert comfyui_control.start_comfyui() == {
        'ok': True, 'reachable': True, 'already_running': True,
    }


def test_start_route_rejects_client_options_before_the_service(client, monkeypatch):
    from app.services import comfyui_control
    monkeypatch.setattr(comfyui_control, 'start_comfyui', _no_spawn)

    query = client.post('/api/setup/comfyui/start?argv=bad')
    body = client.post('/api/setup/comfyui/start', json={'argv': 'bad'})

    assert query.status_code == 400
    assert body.status_code == 400
    assert query.get_json()['error'] == 'This action does not accept options.'


@pytest.mark.parametrize('address', [
    '127.0.0.1', '::1', '::ffff:127.0.0.1',
    '192.0.2.20', '100.64.0.1',
])
def test_start_route_accepts_clients_allowed_by_the_global_access_guard(
        client, monkeypatch, address):
    from app.services import comfyui_control
    monkeypatch.setattr(comfyui_control, 'start_comfyui',
                        lambda: {'ok': True, 'reachable': True})

    response = client.post('/api/setup/comfyui/start',
                           environ_overrides={'REMOTE_ADDR': address})

    assert response.status_code == 200
    assert response.get_json() == {'ok': True, 'reachable': True}


def test_launch_error_never_returns_an_installation_path(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _configure(base)
    monkeypatch.setattr(comfyui_control, '_history_state',
                        lambda: comfyui_control._HISTORY_DOWN)

    def fail_with_private_path(_layout):
        raise OSError(f'cannot execute {base}')

    monkeypatch.setattr(comfyui_control, '_spawn', fail_with_private_path)
    result = comfyui_control.start_comfyui(wait_timeout=0)

    assert result['ok'] is False
    assert str(base) not in result['error']
    assert 'python_embeded' not in result['error']


def test_occupied_port_is_never_replaced(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _configure(base)
    monkeypatch.setattr(comfyui_control, '_history_state',
                        lambda: comfyui_control._HISTORY_OCCUPIED)
    monkeypatch.setattr(comfyui_control, '_spawn', _no_spawn)

    assert comfyui_control.start_comfyui() == {
        'ok': True, 'reachable': False, 'already_running': True, 'starting': True,
    }


class _HistoryResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, BaseException):
            raise self._body
        return self._body


def test_history_probe_fails_closed_except_for_a_connection_refusal(monkeypatch):
    from app.services import comfyui_control
    calls = []

    def ready_get(url, **kwargs):
        calls.append((url, kwargs))
        return _HistoryResponse(200, {})

    monkeypatch.setattr(comfyui_control.requests, 'get', ready_get)
    assert comfyui_control._history_state() == comfyui_control._HISTORY_READY
    assert calls == [(
        'http://127.0.0.1:8188/history',
        {'timeout': (1, 3), 'allow_redirects': False},
    )]

    monkeypatch.setattr(comfyui_control.requests, 'get',
                        lambda *_args, **_kwargs: _HistoryResponse(404, {}))
    assert comfyui_control._history_state() == comfyui_control._HISTORY_OCCUPIED
    monkeypatch.setattr(comfyui_control.requests, 'get',
                        lambda *_args, **_kwargs: _HistoryResponse(200, []))
    assert comfyui_control._history_state() == comfyui_control._HISTORY_OCCUPIED
    monkeypatch.setattr(comfyui_control.requests, 'get',
                        lambda *_args, **_kwargs: _HistoryResponse(200, ValueError('invalid JSON')))
    assert comfyui_control._history_state() == comfyui_control._HISTORY_OCCUPIED

    def timeout(*_args, **_kwargs):
        raise comfyui_control.requests.Timeout()

    monkeypatch.setattr(comfyui_control.requests, 'get', timeout)
    assert comfyui_control._history_state() == comfyui_control._HISTORY_OCCUPIED

    def refused(*_args, **_kwargs):
        raise comfyui_control.requests.ConnectionError()

    monkeypatch.setattr(comfyui_control.requests, 'get', refused)
    assert comfyui_control._history_state() == comfyui_control._HISTORY_DOWN


def test_history_probe_treats_connect_timeout_as_down(monkeypatch):
    from app.services import comfyui_control

    def connect_timeout(*_args, **_kwargs):
        raise comfyui_control.requests.ConnectTimeout()

    monkeypatch.setattr(comfyui_control.requests, 'get', connect_timeout)

    assert comfyui_control._history_state() == comfyui_control._HISTORY_DOWN


def test_history_probe_treats_read_timeout_as_occupied(monkeypatch):
    from app.services import comfyui_control

    def read_timeout(*_args, **_kwargs):
        raise comfyui_control.requests.ReadTimeout()

    monkeypatch.setattr(comfyui_control.requests, 'get', read_timeout)

    assert comfyui_control._history_state() == comfyui_control._HISTORY_OCCUPIED


def test_spawn_environment_excludes_lds_and_secret_variables(tmp_path, monkeypatch):
    from app.services import comfyui_control
    base = _portable_layout(tmp_path)
    _configure(base)
    calls = []

    class RunningProcess:
        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return RunningProcess()

    for secret_name in comfyui_control.cfg.SECRET_KEYS:
        monkeypatch.setenv(secret_name, 'private-value')
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'private-token')
    monkeypatch.setenv('LDS_TEST_ONLY', 'private-app-setting')
    monkeypatch.setenv('PYTHONPATH', 'untrusted-python-injection')
    monkeypatch.setenv('SystemRoot', 'C:\\Windows')
    history = iter([
        comfyui_control._HISTORY_DOWN,
        comfyui_control._HISTORY_DOWN,
        comfyui_control._HISTORY_READY,
    ])
    monkeypatch.setattr(comfyui_control, '_history_state', lambda: next(history))
    monkeypatch.setattr(comfyui_control.subprocess, 'Popen', fake_popen)

    assert comfyui_control.start_comfyui(wait_timeout=1, poll_interval=0.05) == {
        'ok': True, 'reachable': True,
    }
    assert len(calls) == 1
    child_env = calls[0][1]['env']
    names = {name.upper() for name in child_env}
    assert 'PATH' in names
    assert 'SYSTEMROOT' in names
    assert not (names & {name.upper() for name in comfyui_control.cfg.SECRET_KEYS})
    assert 'LDS_ACCESS_TOKEN' not in names
    assert 'PYTHONPATH' not in names
    assert not any(name.startswith('LDS_') for name in names)
