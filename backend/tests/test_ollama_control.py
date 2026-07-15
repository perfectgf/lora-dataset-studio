"""ollama_control.start_ollama: idempotence, not-installed, success, structured
failure, and the DETACHED spawn flags — all with the real Ollama untouched
(every network + Popen seam is mocked; no server is ever started)."""
import os
import subprocess

import pytest


class _FakeProc:
    """Minimal Popen stand-in: control whether poll() reports 'still running'."""
    def __init__(self, alive=True, stderr_path=None):
        self._alive = alive
        self.stderr_path = stderr_path

    def poll(self):
        return None if self._alive else 0


@pytest.fixture(autouse=True)
def _fast_no_sleep(monkeypatch):
    """Never actually sleep in the poll loop — keeps the timeout tests instant."""
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control.time, 'sleep', lambda *_: None)
    yield


def test_start_idempotent_when_already_running(app, monkeypatch):
    from app.services import ollama_control
    spawned = []
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: True)
    monkeypatch.setattr(ollama_control, '_spawn_detached',
                        lambda b: spawned.append(b) or _FakeProc())
    with app.app_context():
        r = ollama_control.start_ollama()
    assert r == {'ok': True, 'reachable': True, 'already_running': True}
    assert spawned == []          # a running server is NEVER double-spawned


def test_start_not_installed(app, monkeypatch):
    from app import capabilities
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: False)
    monkeypatch.setattr(capabilities, '_ollama_binary', lambda: '')
    with app.app_context():
        r = ollama_control.start_ollama()
    assert r['ok'] is False and r['reachable'] is False
    assert 'not installed' in r['error']


def test_start_success_after_poll(app, monkeypatch):
    """Server not up at first, spawn, then it answers -> ok/reachable."""
    from app import capabilities
    from app.services import ollama_control
    calls = {'n': 0}
    def reachable(url):
        calls['n'] += 1
        return calls['n'] > 1          # first check False (pre-spawn), then True
    monkeypatch.setattr(ollama_control, '_reachable', reachable)
    monkeypatch.setattr(capabilities, '_ollama_binary', lambda: r'C:\bin\ollama.exe')
    monkeypatch.setattr(ollama_control, '_spawn_detached', lambda b: _FakeProc(alive=True))
    with app.app_context():
        r = ollama_control.start_ollama(wait_timeout=5, poll_interval=0)
    assert r == {'ok': True, 'reachable': True}


def test_start_timeout_returns_structured_error_with_stderr(app, tmp_path, monkeypatch):
    """Spawned process dies without ever answering -> structured failure carrying
    the stderr tail read from the launch log."""
    from app import capabilities
    from app.services import ollama_control
    log = tmp_path / 'ollama.log'
    log.write_text('Error: listen tcp 127.0.0.1:11434: bind: address already in use\n',
                   encoding='utf-8')
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: False)
    monkeypatch.setattr(capabilities, '_ollama_binary', lambda: r'C:\bin\ollama.exe')
    monkeypatch.setattr(ollama_control, '_spawn_detached',
                        lambda b: _FakeProc(alive=False, stderr_path=str(log)))
    with app.app_context():
        r = ollama_control.start_ollama(wait_timeout=1, poll_interval=0)
    assert r['ok'] is False and r['reachable'] is False
    assert 'did not become reachable' in r['error']
    assert 'address already in use' in r['stderr']


def test_start_launch_oserror_is_structured(app, monkeypatch):
    from app import capabilities
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, '_reachable', lambda url: False)
    monkeypatch.setattr(capabilities, '_ollama_binary', lambda: r'C:\bin\ollama.exe')
    def boom(b):
        raise OSError('permission denied')
    monkeypatch.setattr(ollama_control, '_spawn_detached', boom)
    with app.app_context():
        r = ollama_control.start_ollama()
    assert r['ok'] is False
    assert 'could not launch Ollama' in r['error']


def test_spawn_detached_uses_detached_flags(monkeypatch, tmp_path):
    """The spawn must be DETACHED (survives the app) with no console window, and
    write to a temp FILE (never a PIPE that a chatty server could stall on)."""
    from app.services import ollama_control
    seen = {}
    class _P:
        pass
    def fake_popen(cmd, **kw):
        seen['cmd'] = cmd
        seen['kw'] = kw
        return _P()
    monkeypatch.setattr(ollama_control.subprocess, 'Popen', fake_popen)
    proc = ollama_control._spawn_detached(r'C:\bin\ollama.exe')
    assert seen['cmd'] == [r'C:\bin\ollama.exe', 'serve']
    kw = seen['kw']
    assert kw['close_fds'] is True
    assert kw['stdin'] == subprocess.DEVNULL
    assert kw['stderr'] == subprocess.STDOUT
    # stdout is a real file object (temp log), not a PIPE
    assert kw['stdout'] not in (subprocess.PIPE, subprocess.DEVNULL, None)
    if os.name == 'nt':
        flags = kw['creationflags']
        assert flags & 0x00000008    # DETACHED_PROCESS
        assert flags & 0x00000200    # CREATE_NEW_PROCESS_GROUP
        assert flags & 0x08000000    # CREATE_NO_WINDOW
    else:
        assert kw['start_new_session'] is True
    assert proc.stderr_path                      # log path recorded for a later tail read
    os.path.isfile(proc.stderr_path) and os.remove(proc.stderr_path)


def test_caption_ready_never_starts_local_process_for_remote_url(app, monkeypatch):
    from app.services import ollama_control
    from app import config
    with app.app_context():
        config.save_config({'ollama': {'url': 'http://remote-box:11434'}})
        monkeypatch.setattr(ollama_control, '_reachable', lambda url: False)
        monkeypatch.setattr(ollama_control, 'start_ollama',
                            lambda: (_ for _ in ()).throw(AssertionError('must not spawn')))
        result = ollama_control.ensure_captioning_ready()
    assert result['ok'] is False and 'remote' in result['error'].lower()
