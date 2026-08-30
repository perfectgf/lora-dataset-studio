"""The LM Studio Start button, held to what it promises.

Every case here is one the measured CLI can actually produce. The one that
matters most is the last: `lms server start` exits 0 and prints "Success!"
BEFORE the port necessarily answers, so an exit code is not a verdict. This
repo already has that rule for installs — never claim success without
re-running the probe — and a Start button is the same shape of promise.
"""
import subprocess

import pytest

from app import config
from app.services import lmstudio_control, vision_llm


@pytest.fixture
def as_lmstudio(app):
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://127.0.0.1:1234'}})
    yield


def _run(returncode=0, stdout='', stderr=''):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_a_running_server_is_never_restarted_under_the_user(app, monkeypatch, as_lmstudio):
    """Idempotent, and it says so. Restarting a live server would drop whatever
    model the user had loaded — the exact cost the button must not impose."""
    monkeypatch.setattr(lmstudio_control, '_reachable', lambda url, timeout=2.0: True)

    def never(*a, **kw):
        raise AssertionError('the CLI ran against a server that was already up')

    monkeypatch.setattr(lmstudio_control.subprocess, 'run', never)
    with app.app_context():
        assert lmstudio_control.start_server() == {
            'ok': True, 'reachable': True, 'already_running': True}


def test_a_remote_lm_studio_is_refused_rather_than_started_locally(app, monkeypatch):
    """Starting a LOCAL server for a URL pointing elsewhere is the worst outcome:
    the button appears to work, a second server really does start, and the app
    still talks to the machine that is down."""
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://10.0.0.5:1234'}})
    monkeypatch.setattr(lmstudio_control, '_reachable', lambda url, timeout=2.0: False)

    def never(*a, **kw):
        raise AssertionError('the CLI ran for a server on another machine')

    monkeypatch.setattr(lmstudio_control.subprocess, 'run', never)
    with app.app_context():
        out = lmstudio_control.start_server()
    assert out['ok'] is False and out['reachable'] is False
    assert 'not on this machine' in out['error']


def test_no_cli_says_how_to_get_one_instead_of_failing_blankly(app, monkeypatch, as_lmstudio):
    monkeypatch.setattr(lmstudio_control, '_reachable', lambda url, timeout=2.0: False)
    monkeypatch.setattr(lmstudio_control, 'lmstudio_cli', lambda: '')
    with app.app_context():
        out = lmstudio_control.start_server()
    assert out['ok'] is False
    # The remedy, not just the diagnosis: opening LM Studio once installs the tool.
    assert 'Developer' in out['error'] or 'installs the tool' in out['error']


def test_a_refusing_cli_is_quoted_rather_than_paraphrased(app, monkeypatch, as_lmstudio):
    """The CLI knows why far better than this app can guess, and a guess is what
    sends someone down the wrong path."""
    monkeypatch.setattr(lmstudio_control, '_reachable', lambda url, timeout=2.0: False)
    monkeypatch.setattr(lmstudio_control, 'lmstudio_cli', lambda: '/x/lms')
    monkeypatch.setattr(lmstudio_control.subprocess, 'run',
                        lambda *a, **kw: _run(1, stderr='port 1234 is already in use'))
    with app.app_context():
        out = lmstudio_control.start_server()
    assert out['ok'] is False and out['reachable'] is False
    assert 'port 1234 is already in use' in out['stderr']


def test_exit_zero_is_not_a_verdict(app, monkeypatch, as_lmstudio):
    """THE case this file exists for.

    `lms server start` prints "Success! Server is now running on port 1234" and
    returns 0. If the port never answers — a stuck bind, a server that came up on
    a different port — a button trusting that exit code turns green over an install
    that still cannot caption a single image.
    """
    monkeypatch.setattr(lmstudio_control, '_reachable', lambda url, timeout=2.0: False)
    monkeypatch.setattr(lmstudio_control, 'lmstudio_cli', lambda: '/x/lms')
    monkeypatch.setattr(lmstudio_control.subprocess, 'run',
                        lambda *a, **kw: _run(0, stdout='Success! Server is now running'))
    with app.app_context():
        out = lmstudio_control.start_server(wait_timeout=0.2, poll_interval=0.05)
    assert out['ok'] is False and out['reachable'] is False
    assert 'reported success' in out['error']


def test_the_configured_port_is_passed_so_it_starts_where_the_app_looks(
        app, monkeypatch):
    """`lms` otherwise reuses the port it last ran on. A user who moved LM Studio
    to 1235 in Settings would press Start, see it succeed, and still read "not
    answering" — the button working and the app still broken."""
    with app.app_context():
        config.save_config({'local_llm': {'provider': 'lmstudio'},
                            'lmstudio': {'url': 'http://127.0.0.1:1235/v1'}})
    seen = {}
    answers = iter([False, True])
    monkeypatch.setattr(lmstudio_control, '_reachable',
                        lambda url, timeout=2.0: next(answers, True))
    monkeypatch.setattr(lmstudio_control, 'lmstudio_cli', lambda: '/x/lms')

    def record(cmd, **kw):
        seen['cmd'] = cmd
        return _run(0)

    monkeypatch.setattr(lmstudio_control.subprocess, 'run', record)
    with app.app_context():
        assert lmstudio_control.start_server()['reachable'] is True
    assert seen['cmd'][:3] == ['/x/lms', 'server', 'start']
    assert seen['cmd'][3:] == ['--port', '1235'], 'the configured port was not passed'
    # ...and never --bind: widening what the server listens on is the user's
    # decision to make in LM Studio, not a side effect of a button labelled Start.
    assert '--bind' not in seen['cmd']


def test_the_start_is_routed_by_provider_not_hard_wired(app, monkeypatch):
    """One path, two servers. Two provider-specific endpoints would let the Setup
    button and the Settings button drift into starting different things — the
    divergence the model pickers already had to be rescued from."""
    from app.services import ollama_control
    monkeypatch.setattr(ollama_control, 'start_ollama',
                        lambda: {'ok': True, 'reachable': True, 'which': 'ollama'})
    monkeypatch.setattr(lmstudio_control, 'start_server',
                        lambda: {'ok': True, 'reachable': True, 'which': 'lmstudio'})
    for provider, expected in (('ollama', 'ollama'), ('lmstudio', 'lmstudio')):
        with app.app_context():
            config.save_config({'local_llm': {'provider': provider}})
            assert vision_llm.start_server()['which'] == expected


def test_the_route_answers_200_on_a_failure_too(app, client, monkeypatch, as_lmstudio):
    """A handled outcome, never a server fault: a 5xx makes apiFetch throw AND
    auto-toast a generic error on top of the specific one the body carries."""
    monkeypatch.setattr(lmstudio_control, 'start_server',
                        lambda: {'ok': False, 'reachable': False, 'error': 'nope'})
    r = client.post('/api/local-llm/start')
    assert r.status_code == 200
    assert r.get_json() == {'ok': False, 'reachable': False, 'error': 'nope'}


def test_the_cli_is_found_at_the_path_lm_studio_bootstraps_it_to(monkeypatch, tmp_path):
    """PATH first, then the fixed per-user location — which is the common case
    right after an install, in a shell whose PATH was never refreshed."""
    monkeypatch.setattr(lmstudio_control.shutil, 'which', lambda name: None)
    monkeypatch.setattr(lmstudio_control.Path, 'home', staticmethod(lambda: tmp_path))
    assert lmstudio_control.lmstudio_cli() == ''
    assert lmstudio_control.probe_installed() == {'ok': False, 'cli_path': ''}

    binary = tmp_path / '.lmstudio' / 'bin' / ('lms.exe' if lmstudio_control.os.name == 'nt'
                                               else 'lms')
    binary.parent.mkdir(parents=True)
    binary.write_text('', encoding='utf-8')
    assert lmstudio_control.lmstudio_cli() == str(binary)
    assert lmstudio_control.probe_installed()['ok'] is True

    # PATH still wins when it has one, so a user who put `lms` somewhere of their
    # own is not overridden by the default location.
    monkeypatch.setattr(lmstudio_control.shutil, 'which', lambda name: '/elsewhere/lms')
    assert lmstudio_control.lmstudio_cli() == '/elsewhere/lms'
