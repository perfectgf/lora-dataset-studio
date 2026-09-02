"""Safely start the LDS-owned ComfyUI portable instance on explicit request.

This module deliberately supports one narrow layout only: the NVIDIA Windows
portable bundle already configured in LDS.  It never invokes a user launcher or
modifies anything below that installation.  In particular, ``*.bat`` files are
only used as a passive layout marker and are never read, executed, or rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
import time

import requests

from .. import config as cfg


logger = logging.getLogger(__name__)

_LOCAL_API_URL = 'http://127.0.0.1:8188'
_READY_TIMEOUT = 45.0
_POLL_INTERVAL = 0.5
_REPARSE_POINT = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x0400)

# This is LDS's own child only.  We never look up, signal, terminate, or otherwise
# take ownership of a process that was not started by this module.
_start_lock = threading.Lock()
_owned_process = None


@dataclass(frozen=True)
class _PortableLayout:
    base_dir: Path
    python_exe: Path


def _canonical_path(path: str | os.PathLike[str]) -> Path:
    """Canonical, case-normalised path used for all process-affecting paths."""
    raw = os.path.abspath(os.fspath(path))
    return Path(os.path.normcase(os.path.realpath(raw)))


def _is_link_or_reparse(path: Path) -> bool:
    """Fail closed for symlinks, junctions, and any unreadable filesystem node."""
    try:
        info = os.lstat(path)
    except OSError:
        return True
    return bool(stat.S_ISLNK(info.st_mode) or (getattr(info, 'st_file_attributes', 0) & _REPARSE_POINT))


def _has_link_component(path: Path) -> bool:
    """Reject a link/reparse point at any component, not merely the final leaf."""
    try:
        absolute = Path(os.path.abspath(os.fspath(path)))
        if not absolute.is_absolute():
            return True
        parts = absolute.parts
        if not parts:
            return True
        current = Path(parts[0])
        if _is_link_or_reparse(current):
            return True
        for piece in parts[1:]:
            current = current / piece
            if _is_link_or_reparse(current):
                return True
    except (OSError, ValueError):
        return True
    return False


def _real_regular_file(path: Path) -> bool:
    try:
        return not _has_link_component(path) and path.is_file()
    except OSError:
        return False


def _real_directory(path: Path) -> bool:
    try:
        return not _has_link_component(path) and path.is_dir()
    except OSError:
        return False


def _safe_local_api_url(url: object) -> bool:
    """Accept only the exact IPv4 URL served by LDS's fixed bind and probe."""
    # Do not accept localhost/IPv6 aliases: this child always binds IPv4 only.
    return isinstance(url, str) and url == _LOCAL_API_URL


def _validated_portable_layout() -> _PortableLayout | None:
    """Return the one allowed NVIDIA portable layout, otherwise ``None``.

    ``capabilities.resolve_comfyui_base`` intentionally accepts several useful
    user-managed layouts.  That is right for model discovery, but too broad for an
    endpoint which launches a binary.  This independent validation is deliberately
    stricter and is always run immediately before spawning.
    """
    if not cfg.is_configured():
        return None
    raw_base = cfg.get('comfyui.base_dir')
    if not isinstance(raw_base, str) or not raw_base.strip():
        return None
    try:
        raw = Path(raw_base.strip())
        if not raw.is_absolute():
            return None
        # Portable bundles must be local filesystem paths.  UNC/device namespaces
        # would make the trusted local-launch boundary ambiguous.
        raw_text = os.path.abspath(os.fspath(raw))
        if raw_text.startswith('\\\\'):
            return None
        if _has_link_component(raw):
            return None
        base = _canonical_path(raw)
    except (OSError, TypeError, ValueError):
        return None

    # The normal portable layout is
    #   <bundle>/ComfyUI/main.py
    #   <bundle>/ComfyUI/models/
    #   <bundle>/python_embeded/python.exe
    # A source checkout, Desktop app, or arbitrary folder may remain perfectly
    # usable in LDS; it just remains independently launched by its owner.
    if base.name.casefold() != 'comfyui' or not _real_directory(base):
        return None
    bundle = base.parent
    main_py = base / 'main.py'
    models_dir = base / 'models'
    python_dir = bundle / 'python_embeded'
    python_exe = python_dir / 'python.exe'
    nvidia_marker = bundle / 'run_nvidia_gpu.bat'
    if not (
        _real_regular_file(main_py)
        and _real_directory(models_dir)
        and _real_directory(python_dir)
        and _real_regular_file(python_exe)
        # This is a passive identification marker only; it is never executed.
        and _real_regular_file(nvidia_marker)
    ):
        return None
    return _PortableLayout(base_dir=base, python_exe=_canonical_path(python_exe))


def launcher_status() -> dict:
    """Public, path-free capability metadata for the Setup button.

    No process is launched here.  It simply tells the UI whether the *saved*
    configuration has the narrow portable shape this launcher accepts.
    """
    return {
        'portable_supported': _validated_portable_layout() is not None,
        'local_api_safe': _safe_local_api_url(cfg.get('comfyui.api_url')),
    }


_HISTORY_READY = 'ready'
_HISTORY_OCCUPIED = 'occupied'
_HISTORY_DOWN = 'down'


def _history_state() -> str:
    """Classify only LDS's fixed local history endpoint without following redirects."""
    try:
        response = requests.get(f'{_LOCAL_API_URL}/history', timeout=(1, 3),
                                allow_redirects=False)
    except requests.ConnectTimeout:
        # On Windows, Requests can surface a missing loopback listener as a
        # ConnectTimeout rather than ConnectionError.  No TCP connection was
        # established, so the fixed local endpoint is safe to launch.
        return _HISTORY_DOWN
    except requests.ReadTimeout:
        # A connection was established but ComfyUI did not answer in time.
        # Treat the port as occupied and never start a competing process.
        return _HISTORY_OCCUPIED
    except requests.Timeout:
        # Fail closed for any unusual Timeout subtype whose connection phase is
        # unknown.
        return _HISTORY_OCCUPIED
    except requests.ConnectionError:
        return _HISTORY_DOWN
    except requests.RequestException:
        return _HISTORY_OCCUPIED
    if response.status_code != 200:
        return _HISTORY_OCCUPIED
    try:
        return _HISTORY_READY if isinstance(response.json(), dict) else _HISTORY_OCCUPIED
    except (AttributeError, TypeError, ValueError):
        return _HISTORY_OCCUPIED


_CHILD_ENV_ALLOWLIST = frozenset({
    'ALLUSERSPROFILE', 'APPDATA', 'COMSPEC', 'CUDA_PATH', 'HOMEDRIVE', 'HOMEPATH',
    'LOCALAPPDATA', 'NUMBER_OF_PROCESSORS', 'OS', 'PATH', 'PATHEXT',
    'PROCESSOR_ARCHITECTURE', 'PROCESSOR_IDENTIFIER', 'PROCESSOR_LEVEL',
    'PROCESSOR_REVISION', 'PROGRAMDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)',
    'PROGRAMW6432', 'PUBLIC', 'SYSTEMDRIVE', 'SYSTEMROOT', 'TEMP', 'TMP',
    'USERDOMAIN', 'USERNAME', 'USERPROFILE', 'WINDIR',
})
_CHILD_ENV_DENYLIST = frozenset(name.upper() for name in (*cfg.SECRET_KEYS, 'LDS_ACCESS_TOKEN'))


def _sanitized_child_env() -> dict[str, str]:
    """Pass only OS/runtime variables needed by the isolated portable child.

    Custom nodes must not inherit LDS credentials, proxy settings, arbitrary
    Python injection variables, or any ``LDS_*`` application environment values.
    """
    child = {}
    for name, value in os.environ.items():
        normalized = name.upper()
        if normalized.startswith('LDS_') or normalized in _CHILD_ENV_DENYLIST:
            continue
        if normalized in _CHILD_ENV_ALLOWLIST or normalized.startswith('CUDA_PATH_V'):
            child[name] = value
    return child


# Why the launcher passes `--fast-disk` (measured 2026-09-02, per node, on the
# maintainer's machine — 24 GB card, 48 GB of system RAM, models on an NVMe):
#
# The Video Test Studio's MiniMax H3 set weighs ~43 GB (DiT 21 + text encoder
# 16 + VAEs 6). ComfyUI's default loader stages every weight it offloads in
# system RAM, so on a machine whose RAM cannot hold the set beside everything
# else, a 56-frame clip cost 348 s — 319 s of them in the VAE decode, streamed
# layer by layer from a paged RAM with 1 GB of VRAM free, without one line of
# log — and the SECOND clip 302-315 s, the text encoder and the DiT chasing
# each other through the page file. `--fast-disk` tells ComfyUI to prefer the
# safetensors file over an unpinned RAM copy: the same clip took 30 s cold and
# 21-24 s warm, with 21 GB of RAM still free. It beat every other lever that
# was measured: `--disable-dynamic-vram` (72 s cold, 315 s warm), a `/free`
# before the clip (59 s) and a fresh process (63 s). On a machine with RAM to
# spare the flag costs nothing the OS file cache does not give back.
#
# `--disable-dynamic-vram` is deliberately NOT passed: it fixes the VAE decode
# and leaves the swap storm on the next clip, which is the worse half.
_FAST_DISK_FLAG = '--fast-disk'


def _memory_flags(layout: _PortableLayout) -> list:
    """The memory flags THIS ComfyUI understands — read from its own
    `comfy/cli_args.py`, never assumed.

    `--fast-disk` arrived in ComfyUI in May 2026. argparse answers an unknown
    flag with an exit before the server exists, which the launcher would then
    report as "ComfyUI couldn't start" on every older portable bundle — a
    regression for exactly the installs the flag was never going to help. The
    file is read as text (no import, nothing executed); an unreadable file
    means the fixed command line, as before.
    """
    try:
        text = (layout.base_dir / 'comfy' / 'cli_args.py').read_text(
            encoding='utf-8', errors='replace')
    except OSError:
        logger.debug('ComfyUI cli_args.py unreadable; launching without memory flags')
        return []
    # The DECLARATION, not a mention: a comment or a longer flag that happens
    # to contain the name must not put it on the command line.
    declared = re.search(r'add_argument\(\s*["\']' + re.escape(_FAST_DISK_FLAG) + r'["\']', text)
    flags = [_FAST_DISK_FLAG] if declared else []
    # Paste-safe: flag names only, never a path.
    logger.info('ComfyUI portable launch: memory flags %s', flags or 'none')
    return flags


def _spawn(layout: _PortableLayout):
    """Spawn only the fixed, LDS-owned ComfyUI command.  No user .bat is involved."""
    argv = [
        str(layout.python_exe), '-s', 'main.py',
        '--windows-standalone-build',
        '--disable-auto-launch',
        '--preview-method', 'none',
        '--listen', '127.0.0.1',
        '--port', '8188',
        *_memory_flags(layout),
    ]
    kwargs = {
        'shell': False,
        'stdin': subprocess.DEVNULL,
        'stdout': subprocess.DEVNULL,
        'stderr': subprocess.DEVNULL,
        'cwd': str(layout.base_dir),
        'close_fds': True,
        'env': _sanitized_child_env(),
    }
    if os.name == 'nt':
        # Avoid flashing a second console window when LDS is launched from one.
        kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
    return subprocess.Popen(argv, **kwargs)


def _clear_if_exited(process) -> None:
    global _owned_process
    try:
        exited = process.poll() is not None
    except Exception:  # pragma: no cover - defensive around unusual Popen mocks
        exited = True
    if exited:
        with _start_lock:
            if _owned_process is process:
                _owned_process = None


def _failure() -> dict:
    # Intentionally paste-safe: no path, argv, exception, or child log reaches HTTP.
    return {
        'ok': False,
        'reachable': False,
        'error': "ComfyUI couldn't start. Check its portable installation and try again.",
    }

def _busy() -> dict:
    """Paste-safe outcome when 8188 is already responding too slowly to probe."""
    return {'ok': True, 'reachable': False, 'already_running': True, 'starting': True}



def start_comfyui(*, wait_timeout: float = _READY_TIMEOUT,
                  poll_interval: float = _POLL_INTERVAL) -> dict:
    """Start the configured portable ComfyUI, or report a safe idempotent outcome.

    This function is only called from the explicit Setup POST route.  It never
    starts at boot, never writes under ComfyUI, and never kills a process.
    """
    if not _safe_local_api_url(cfg.get('comfyui.api_url')):
        return _failure()
    layout = _validated_portable_layout()
    if layout is None:
        return _failure()

    # A manually started, valid local ComfyUI remains fully independent: do not
    # replace it or claim its PID; simply report that the endpoint is ready.
    history = _history_state()
    if history == _HISTORY_READY:
        return {'ok': True, 'reachable': True, 'already_running': True}
    if history == _HISTORY_OCCUPIED:
        return _busy()

    global _owned_process
    with _start_lock:
        # Close the race between the first cheap probe and the lock acquisition.
        history = _history_state()
        if history == _HISTORY_READY:
            return {'ok': True, 'reachable': True, 'already_running': True}
        if history == _HISTORY_OCCUPIED:
            return _busy()
        if _owned_process is not None:
            try:
                running = _owned_process.poll() is None
            except Exception:  # pragma: no cover - defensive around unusual Popen mocks
                running = False
            if running:
                return {'ok': True, 'reachable': False, 'already_running': True,
                        'starting': True}
            _owned_process = None
        try:
            logger.info('ComfyUI portable launch attempted')
            _owned_process = _spawn(layout)
        except OSError:
            # Do not echo the exception: on Windows it normally includes the full
            # installation path, which this endpoint must never expose.
            logger.warning('ComfyUI portable launch failed')
            return _failure()
        process = _owned_process

    deadline = time.monotonic() + max(0.0, wait_timeout)
    while time.monotonic() < deadline:
        history = _history_state()
        if history == _HISTORY_READY:
            logger.info('ComfyUI portable readiness succeeded')
            return {'ok': True, 'reachable': True}
        try:
            returncode = process.poll()
            if returncode is not None:
                logger.warning('ComfyUI portable exited before readiness (code=%s)', returncode)
                _clear_if_exited(process)
                return _failure()
        except Exception:  # pragma: no cover - defensive around unusual Popen mocks
            _clear_if_exited(process)
            return _failure()
        # A bounded sleep prevents tight-looping against a still-loading ComfyUI.
        time.sleep(max(0.05, poll_interval))

    # Close the last-poll race without ever terminating the child.  Keeping a live
    # LDS-owned process in `_owned_process` prevents a timeout retry from spawning a
    # second competing instance; a later request can observe it as ready.
    history = _history_state()
    if history == _HISTORY_READY:
        logger.info('ComfyUI portable readiness succeeded')
        return {'ok': True, 'reachable': True}
    if history == _HISTORY_OCCUPIED:
        logger.warning('ComfyUI portable readiness timed out while the endpoint was occupied')
        return _busy()
    try:
        returncode = process.poll()
    except Exception:  # pragma: no cover - defensive around unusual Popen mocks
        returncode = None
    if returncode is not None:
        logger.warning('ComfyUI portable exited before readiness (code=%s)', returncode)
        _clear_if_exited(process)
        return _failure()
    logger.warning('ComfyUI portable readiness timed out')
    return {'ok': True, 'reachable': False, 'starting': True}


def can_start() -> bool:
    """Can this install actually LAUNCH ComfyUI from a button?

    Answered by the very check `start_comfyui` runs immediately before spawning,
    so a button gated on this cannot appear over an endpoint that would refuse.
    Deliberately narrow, and NOT the same question as "is ComfyUI installed":
    `capabilities.resolve_comfyui_base` accepts several user-managed layouts,
    which is right for finding models and too broad for starting a binary.

    False therefore covers the ordinary case where ComfyUI is perfectly fine and
    simply not ours to run — a Desktop install, a .bat the user wrote, another
    machine. Offering to start one of those would be a button that fails, which
    on a screen whose whole job is to unblock someone is worse than no button.
    """
    try:
        return _validated_portable_layout() is not None
    except Exception:                     # noqa: BLE001 — a probe is never fatal
        return False
