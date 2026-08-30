"""Start LM Studio's local server on demand — powers the "Start LM Studio" button.

The sibling of `ollama_control`, and deliberately NOT a copy of it: the two
products are started in genuinely different ways, and pretending otherwise is
how a button ends up lying about what it did.

WHAT WAS MEASURED (LM Studio 0.4.23, on a machine with the desktop app running):

  · `lms server start` is a ONE-SHOT command. It exits 0 in ~400 ms having
    printed "Success! Server is now running on port 1234". Ollama's `serve` is a
    long-lived process this app spawns detached and then polls; `lms` is a
    control command that returns. So this module runs it and WAITS for it,
    rather than detaching — a detached spawn here would throw away the exit code
    and the error text, which are the only things that explain a failure.
  · A model loaded before a SERVER stop/start is STILL LOADED after it, so the
    button costs a running install nothing. It is NOT true of a cold start: with
    the LM Studio application fully closed, `lms server start` brings the server
    up (measured: 7.5 s from zero processes) with NOTHING loaded. Both are
    correct outcomes and the UI must not promise the first in the second case.
  · `lms server start --port <n>` exists, and without it the server reuses "the
    same port as the last time it was started" — which is not necessarily the
    port LDS is configured to talk to. We pass the configured one, so the button
    starts the server where the app will actually look for it.

WHAT IS NOT MEASURED, and therefore not claimed anywhere in the UI: whether
`lms server start` succeeds with the LM Studio desktop app closed. The failure
path surfaces the CLI's own stderr verbatim, so if it cannot, the user reads the
real reason rather than a sentence this file invented.

GUARD-RAIL: nothing here may run from a passive probe. It is only ever reached
through an explicit user click (POST /api/local-llm/start, routed by provider),
so the app can never
silently start a server behind the user's back.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_DEFAULT_URL = 'http://127.0.0.1:1234'
# `lms server start` returns in well under a second when it works. The ceiling is
# for the case where it returns 0 and the port is still binding.
_READY_TIMEOUT = 15.0
_POLL_INTERVAL = 0.4
# How long the CLI itself may take before we stop waiting on it. Generous next to
# the ~400 ms measured, because a first run may have to bootstrap.
_CLI_TIMEOUT = 60.0
_STDERR_TAIL = 2000     # chars of the CLI's own output surfaced on failure


def _url() -> str:
    from . import vision_lmstudio
    return vision_lmstudio.base_url() or _DEFAULT_URL


def _is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or '').lower()
    return host == 'localhost' or host == '::1' or host.startswith('127.')


def _port(url: str) -> int | None:
    """The port LDS expects to find the server on, or None to let `lms` choose.

    Passing it is what makes the button honest: `lms` otherwise reuses whatever
    port it last ran on, so a user who moved LM Studio to 1235 in Settings could
    press Start, watch it succeed, and still see "not answering".
    """
    try:
        return urlparse(url).port
    except ValueError:      # a malformed port in a hand-edited config
        return None


def lmstudio_cli() -> str:
    """Absolute path to the `lms` CLI if LM Studio is installed, else ''.

    Two signals, neither of which needs the server running:
      1. ``shutil.which('lms')`` — LM Studio's bootstrap adds the CLI to PATH.
      2. The per-user location that bootstrap writes to, ``~/.lmstudio/bin/lms``
         (``lms.exe`` on Windows) — same on every platform, and it covers a shell
         whose PATH was not refreshed since the install, which is the common case
         right after someone installs LM Studio and comes back to this page.
    First hit wins; never raises.
    """
    exe = shutil.which('lms')
    if exe:
        return exe
    name = 'lms.exe' if os.name == 'nt' else 'lms'
    try:
        cand = Path.home() / '.lmstudio' / 'bin' / name
        if cand.is_file():
            return str(cand)
    except OSError:
        pass
    return ''


def probe_installed() -> dict:
    """``{ok, cli_path}`` — is LM Studio installed, server up or not.

    Passive: it looks at the filesystem and nothing else. `ok` is what the UI
    reads as "installed but not running → offer a Start", the third state Ollama
    has always had and LM Studio could not have until this CLI was found.
    """
    exe = lmstudio_cli()
    return {'ok': bool(exe), 'cli_path': exe}


def _reachable(url: str, timeout: float = 2.0) -> bool:
    """Does something answer on the configured URL? Any HTTP answer counts.

    Deliberately not "answers correctly": this only has to distinguish a bound
    port from a dead one. Readiness (a model actually loaded) is a different
    question, asked by probe_lmstudio_model, and conflating the two here would
    make the Start button report failure for a server it started perfectly.
    """
    try:
        requests.get(f'{url}/api/v0/models', timeout=timeout)
        return True
    except requests.RequestException:
        return False


def _tail(text: str) -> str:
    text = (text or '').strip()
    return text[-_STDERR_TAIL:] if text else ''


def start_server(*, wait_timeout: float = _READY_TIMEOUT,
                 poll_interval: float = _POLL_INTERVAL) -> dict:
    """Idempotently bring LM Studio's local server up. Returns:
      already up           -> {ok:True,  reachable:True, already_running:True}
      started & answered   -> {ok:True,  reachable:True}
      remote URL           -> {ok:False, reachable:False, error:...}
      CLI not found        -> {ok:False, reachable:False, error:...}
      CLI failed           -> {ok:False, reachable:False, error:..., stderr:...}
      ran, never answered  -> {ok:False, reachable:False, error:..., stderr:...}
    Never raises."""
    url = _url()
    # Idempotent no-op: a running server must never be restarted under the user.
    if _reachable(url):
        return {'ok': True, 'reachable': True, 'already_running': True}

    # A configured remote LM Studio has no local server to start, and running
    # `lms` would start a DIFFERENT one on this machine that the app would then
    # not be pointing at — a button that appears to work and changes nothing.
    if not _is_loopback_url(url):
        return {'ok': False, 'reachable': False,
                'error': f'{url} is not on this machine, so there is no local server '
                         'to start. Start it where it runs, or point Settings at a '
                         'local LM Studio.'}

    exe = lmstudio_cli()
    if not exe:
        return {'ok': False, 'reachable': False,
                'error': "LM Studio's command-line tool was not found (no `lms` on PATH "
                         'or in the default per-user location). Open LM Studio once — it '
                         'installs the tool on first run — or start the server from its '
                         'Developer tab.'}

    cmd = [exe, 'server', 'start']
    port = _port(url)
    if port:
        cmd += ['--port', str(port)]
    # No --bind: the default is loopback, and widening what the server listens on
    # is the user's decision to make in LM Studio, never a side effect of a button
    # labelled Start.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_CLI_TIMEOUT,
                              stdin=subprocess.DEVNULL, check=False,
                              # Windows: no console window flashes on a click.
                              creationflags=0x08000000 if os.name == 'nt' else 0)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'reachable': False,
                'error': f'`lms server start` did not return within {int(_CLI_TIMEOUT)}s.'}
    except OSError as e:
        logger.warning('lmstudio start: launch failed: %s', e)
        return {'ok': False, 'reachable': False,
                'error': f'could not run the LM Studio CLI: {e}'}

    output = _tail(f'{proc.stdout}\n{proc.stderr}')
    if proc.returncode != 0:
        # The CLI's own words, not ours: it knows why far better than this file
        # can guess, and a guess here is what sends a user down the wrong path.
        out = {'ok': False, 'reachable': False,
               'error': 'LM Studio refused to start its server.'}
        if output:
            out['stderr'] = output
        return out

    # Exit 0 is NOT the answer. The rule this repo already applies to installs
    # applies here: never claim success without re-running the probe.
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if _reachable(url):
            return {'ok': True, 'reachable': True}
        time.sleep(poll_interval)
    if _reachable(url):
        return {'ok': True, 'reachable': True}

    out = {'ok': False, 'reachable': False,
           'error': f'The LM Studio CLI reported success, but nothing answered at {url} '
                    f'within {int(wait_timeout)}s.'}
    if output:
        out['stderr'] = output
    return out
