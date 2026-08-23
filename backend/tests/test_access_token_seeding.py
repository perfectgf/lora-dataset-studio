"""netguard.ensure_access_token(): the launcher-side half of the gate.

The two halves used to fail open together -- netguard let non-loopback clients
through unless server.require_token was on, and run.py only generated a token
under that same flag, which defaults off. A public bind needs a token to exist
whether or not the user ever opened Settings.
"""
import os
import pytest

from app import netguard


@pytest.fixture(autouse=True)
def _restore_ldsaccesstoken_env(monkeypatch):
    """ensure_access_token() stamps os.environ as a side effect (by design --
    run.py depends on it). monkeypatch.delenv(raising=False) only records an
    undo entry when the name was already present, so when a test below delenvs
    a variable that was never set, there's nothing to undo and the token
    written during the test leaks into every test that runs after this file --
    breaking tests/test_netguard.py whenever the suite runs in its default
    alphabetical order. Snapshot/restore unconditionally instead.

    monkeypatch.undo() is called first, deliberately, before the manual
    restore below: a mid-test delenv('LDS_ACCESS_TOKEN') call made AFTER the
    SUT has already set the var (test_second_call_reuses_the_persisted_token
    does exactly this) *is* tracked by monkeypatch, since the var exists at
    that point -- and monkeypatch's own autouse finalizer runs AFTER this one,
    which would silently re-leak the value it restores. Flushing monkeypatch's
    undo stack here, before our own restore, guarantees this fixture always
    has the last word regardless of fixture teardown order. undo() is safe to
    call more than once -- pytest's own call at monkeypatch's teardown is then
    a no-op.
    """
    had = 'LDS_ACCESS_TOKEN' in os.environ
    previous = os.environ.get('LDS_ACCESS_TOKEN')
    yield
    monkeypatch.undo()
    if had:
        os.environ['LDS_ACCESS_TOKEN'] = previous
    else:
        os.environ.pop('LDS_ACCESS_TOKEN', None)


def test_public_bind_generates_and_persists_a_token(app, monkeypatch):
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
    with app.app_context():
        token = netguard.ensure_access_token('0.0.0.0')
    assert token
    assert os.environ['LDS_ACCESS_TOKEN'] == token
    from app import config as cfg
    cfg._cache = None
    assert cfg.get('server.access_token') == token   # survives a restart


def test_second_call_reuses_the_persisted_token(app, monkeypatch):
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
    with app.app_context():
        first = netguard.ensure_access_token('0.0.0.0')
        monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
        second = netguard.ensure_access_token('0.0.0.0')
    assert first == second       # must not rotate every boot


def test_loopback_bind_seeds_nothing(app, monkeypatch):
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
    with app.app_context():
        assert netguard.ensure_access_token('127.0.0.1') is None
    assert 'LDS_ACCESS_TOKEN' not in os.environ


def test_trusted_lan_default_still_seeds_nothing(app, monkeypatch):
    """Not public, require_token off -> unchanged trusted-LAN behaviour."""
    monkeypatch.delenv('LDS_PUBLIC', raising=False)
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
    with app.app_context():
        assert netguard.ensure_access_token('0.0.0.0') is None


def test_escape_hatch_seeds_nothing(app, monkeypatch):
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.setenv('LDS_ALLOW_UNAUTHENTICATED', '1')
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)
    with app.app_context():
        assert netguard.ensure_access_token('0.0.0.0') is None


def test_persistence_failure_falls_back_to_ephemeral_token(app, monkeypatch):
    """A root-owned network volume can make save_config() raise OSError while
    writing config.json.tmp -- that must degrade to an ephemeral in-process
    token, not kill boot (see netguard.ensure_access_token's docstring)."""
    monkeypatch.setenv('LDS_PUBLIC', '1')
    monkeypatch.delenv('LDS_ACCESS_TOKEN', raising=False)

    from app import config as cfg

    def _boom(*args, **kwargs):
        raise OSError('read-only file system')

    monkeypatch.setattr(cfg, 'save_config', _boom)
    with app.app_context():
        token = netguard.ensure_access_token('0.0.0.0')
    assert token
    assert os.environ['LDS_ACCESS_TOKEN'] == token
