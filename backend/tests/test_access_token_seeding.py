"""netguard.ensure_access_token(): the launcher-side half of the gate.

The two halves used to fail open together -- netguard let non-loopback clients
through unless server.require_token was on, and run.py only generated a token
under that same flag, which defaults off. A public bind needs a token to exist
whether or not the user ever opened Settings.
"""
import os

from app import netguard


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
