"""What the MACHINE looked like when a run was launched.

Two training runs can differ without a single dataset image or setting having
moved: a `git pull` in ai-toolkit changes the trainer itself, a torch/CUDA
upgrade changes the kernels, and re-downloading "Krea 2 Raw" can put a DIFFERENT
file behind the same name. None of that was recorded, so any comparison that
blamed the dataset for the gap was guessing.

Everything here is BEST-EFFORT:

  * nothing imports torch (a cold import costs seconds) — the version is READ
    out of the ai-toolkit venv's `torch/version.py`, a two-line file;
  * the ai-toolkit revision is one `git rev-parse`, hard-timeout 4 s, and only
    when that install is actually a git checkout;
  * the GPU name is one `nvidia-smi` query;
  * model artifacts used for exact-resume compatibility are identified by a
    complete SHA-256. A metadata cache avoids re-reading unchanged multi-GB
    files, but never treats size/mtime alone as an identity.

Every probe is memoised for `_TTL` seconds and every failure degrades to None —
an unknown is recorded as absent, never as a claim. A launch must never wait on,
or fail because of, provenance.

Paths are redacted (`~`) before they are stored: a snapshot ends up in the Share
config and in pasted diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import subprocess
import time
from pathlib import Path

from .. import config as cfg
from ..utils.redact import redact_user_paths

logger = logging.getLogger(__name__)

_TTL = 600.0                    # 10 min — none of this changes between two runs
_GIT_TIMEOUT = 4                # seconds; a slow/locked repo must not hold a launch
_SMI_TIMEOUT = 5
_HASH_CHUNK_BYTES = 8 << 20     # 8 MiB streaming reads for multi-GB model files
_REPARSE_POINT = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x0400)

_cache: dict = {}


def _no_window():
    return getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def _memo(key, producer):
    """Memoise `producer()` for _TTL seconds. A raising producer caches None so a
    broken probe is not retried on every launch."""
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and (now - hit[0]) < _TTL:
        return hit[1]
    try:
        value = producer()
    except Exception:
        logger.debug('environment probe %s failed', key, exc_info=True)
        value = None
    _cache[key] = (now, value)
    return value


def clear_cache() -> None:
    """Drop every memoised probe (tests, and Settings changing the ai-toolkit dir)."""
    _cache.clear()


# --- ai-toolkit revision -------------------------------------------------------

def _aitoolkit_probe():
    root = cfg.aitoolkit_path('dir')
    if not root or not Path(root).is_dir():
        return None
    if not (Path(root) / '.git').exists():
        # A zip/manual install has no revision to report. Honest silence.
        return None
    out = {}
    for key, args in (('commit', ['rev-parse', '--short=10', 'HEAD']),
                      ('committed_at', ['log', '-1', '--format=%cI'])):
        try:
            proc = subprocess.run(['git', '-C', str(root), *args],
                                  capture_output=True, text=True,
                                  timeout=_GIT_TIMEOUT,
                                  creationflags=_no_window())
        except (OSError, subprocess.SubprocessError):
            return out or None
        if proc.returncode != 0:
            return out or None
        value = (proc.stdout or '').strip()
        if value:
            out[key] = value
    return out or None


def aitoolkit_revision():
    """{'commit': 'a1b2c3d4e5', 'committed_at': '...'} for the ai-toolkit checkout
    that will train, or None (not configured / not a git checkout / git absent)."""
    return _memo('aitoolkit', _aitoolkit_probe)


# --- torch / CUDA (read, never imported) ---------------------------------------

_VERSION_RE = re.compile(r"^__version__\s*[:=][^'\"]*['\"]([^'\"]+)['\"]", re.MULTILINE)
_CUDA_RE = re.compile(r"^cuda\s*[:=][^'\"]*['\"]([^'\"]+)['\"]", re.MULTILINE)


def _site_packages_candidates(python_path: Path):
    """Plausible site-packages dirs for an interpreter, without running it.
    Windows keeps them at `<env>/Lib/site-packages`, POSIX at
    `<env>/lib/python3.X/site-packages` — both layouts are tried because LDS runs
    on Windows, on Linux and inside the Docker image."""
    env = python_path.parent.parent      # .../Scripts/python.exe -> env root
    yield env / 'Lib' / 'site-packages'
    lib = env / 'lib'
    if lib.is_dir():
        try:
            for child in sorted(lib.iterdir()):
                if child.name.startswith('python'):
                    yield child / 'site-packages'
        except OSError:
            return


def _torch_probe():
    python = cfg.aitoolkit_path('venv_python')
    if not python:
        return None
    for sp in _site_packages_candidates(Path(python)):
        vf = sp / 'torch' / 'version.py'
        try:
            text = vf.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        m = _VERSION_RE.search(text)
        if not m:
            continue
        out = {'torch': m.group(1)}
        c = _CUDA_RE.search(text)
        if c:
            out['cuda'] = c.group(1)
        return out
    return None


def torch_info():
    """{'torch': '2.7.0+cu128', 'cuda': '12.8'} of the ai-toolkit venv — the
    interpreter that TRAINS, not the one serving this request — or None.

    Deliberately parsed out of `torch/version.py` instead of imported: importing
    torch in a cold venv costs seconds and this runs on the launch path."""
    return _memo('torch', _torch_probe)


# --- GPU -----------------------------------------------------------------------

def _gpu_probe():
    # memory.total rides along in the SAME query — it costs nothing extra and it
    # is what lets the training panel advise "you can turn the memory savers off"
    # instead of making every user guess what their card takes. nvidia-smi prints
    # it as "24564 MiB"; anything unparseable is simply absent (never a claim).
    proc = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
         '--format=csv,noheader'],
        capture_output=True, text=True, timeout=_SMI_TIMEOUT,
        creationflags=_no_window())
    if proc.returncode != 0:
        return None
    line = ((proc.stdout or '').strip().splitlines() or [''])[0]
    parts = [p.strip() for p in line.split(',')]
    name = parts[0] if parts else ''
    if not name:
        return None
    out = {'name': name}
    if len(parts) > 1 and parts[1]:
        out['driver'] = parts[1]
    if len(parts) > 2:
        m = re.search(r'(\d+)', parts[2])
        if m:
            # Reported in MiB. Rounded to one decimal GiB — a 24 GB card reports
            # 24564 MiB (23.99 GiB), so flooring to an int would call it 23.
            out['vram_gb'] = round(int(m.group(1)) / 1024.0, 1)
    return out


def gpu_info():
    """{'name': 'NVIDIA GeForce RTX 4090', 'driver': '...', 'vram_gb': 24.0} or
    None (no NVIDIA card, nvidia-smi absent, a cloud launch from a GPU-less
    machine). `vram_gb` may be missing on its own if the driver didn't report it —
    callers must treat an absent value as "unknown", never as "small"."""
    return _memo('gpu', _gpu_probe)


def local_vram_gb():
    """Total VRAM of this machine's first NVIDIA GPU in GiB, or None when it can't
    be known. The ONLY consumer-facing use is advisory (see the memory-saving
    levers in lora_training): a None must always degrade to generic guidance, and
    a detected value never overrides a user's explicit choice."""
    gpu = gpu_info() or {}
    v = gpu.get('vram_gb')
    return v if isinstance(v, (int, float)) and v > 0 else None


# --- full model-artifact identity ---------------------------------------------

def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return bool(
        stat.S_ISLNK(info.st_mode)
        or (getattr(info, 'st_file_attributes', 0) & _REPARSE_POINT)
    )


def _assert_plain_path(path: Path) -> None:
    """Reject symlinks/junctions in an existing artifact path."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise OSError(f'model artifact path is unavailable: {current}') from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or (getattr(info, 'st_file_attributes', 0) & _REPARSE_POINT)
        ):
            raise OSError(f'model artifact path contains a link: {current}')


def _windows_change_time(descriptor: int) -> int | None:
    """NTFS ChangeTime for a live handle (unlike Python's creation-time ctime)."""
    if os.name != 'nt':
        return None
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class FILE_BASIC_INFO(ctypes.Structure):
            _fields_ = [
                ('CreationTime', ctypes.c_longlong),
                ('LastAccessTime', ctypes.c_longlong),
                ('LastWriteTime', ctypes.c_longlong),
                ('ChangeTime', ctypes.c_longlong),
                ('FileAttributes', wintypes.DWORD),
            ]

        info = FILE_BASIC_INFO()
        handle = msvcrt.get_osfhandle(descriptor)
        ok = ctypes.windll.kernel32.GetFileInformationByHandleEx(
            wintypes.HANDLE(handle),
            0,  # FileBasicInfo
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        return int(info.ChangeTime) if ok else None
    except (ImportError, OSError, ValueError, AttributeError):
        return None


def _file_change_token(descriptor: int, info) -> tuple:
    change_time = _windows_change_time(descriptor)
    if os.name == 'nt' and change_time is None:
        raise OSError(
            'filesystem change time is unavailable for safe model hash caching')
    if change_time is None:
        change_time = getattr(info, 'st_ctime_ns', None)
    if change_time is None:
        raise OSError(
            'filesystem change time is unavailable for safe model hash caching')
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(getattr(info, 'st_mtime_ns', int(info.st_mtime * 1_000_000_000))),
        int(change_time) if change_time is not None else None,
    )


def _open_regular_nofollow(path: Path):
    """Open one regular file and reject links/reparse points before hashing."""
    _assert_plain_path(path)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OSError('model artifact is not a regular file')
        if _is_link_or_reparse(path):
            raise OSError('model artifact is a link or reparse point')
        return os.fdopen(descriptor, 'rb', closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _full_file_signature(
        path: Path, *, trusted_link_root: Path | None = None) -> dict:
    original = path
    if _is_link_or_reparse(path):
        if trusted_link_root is None:
            raise OSError('model artifact is a link or reparse point')
        _assert_plain_path(path.parent)
        resolved = path.resolve(strict=True)
        trusted_link_root = _absolute_path(trusted_link_root)
        try:
            inside = (
                os.path.commonpath((trusted_link_root, resolved))
                == str(trusted_link_root))
        except ValueError:
            inside = False
        if not inside:
            raise OSError('pinned model link escapes its immutable blob store')
        path = resolved
    with _open_regular_nofollow(path) as stream:
        before_info = os.fstat(stream.fileno())
        before = _file_change_token(stream.fileno(), before_info)
        key = ('model-sha256', os.path.normcase(os.path.abspath(path)), *before)
        hit = _cache.get(key)
        if hit is not None:
            return dict(hit[1])
        digest = hashlib.sha256()
        while True:
            chunk = stream.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after_info = os.fstat(stream.fileno())
        after = _file_change_token(stream.fileno(), after_info)
        if before != after:
            raise OSError('model artifact changed while it was being hashed')
    hexdigest = digest.hexdigest()
    value = {
        'name': original.name,
        'kind': 'file',
        'size': int(before_info.st_size),
        'sha256': hexdigest,
        # Keep the historical short field for run comparison/UI consumers.
        'sig': hexdigest[:16],
        'sampled': False,
    }
    _cache[key] = (time.time(), dict(value))
    return value


def _directory_signature(
        root: Path, *, trusted_link_root: Path | None = None) -> dict:
    _assert_plain_path(root)
    if not root.is_dir() or _is_link_or_reparse(root):
        raise OSError('model artifact directory is unsafe')
    entries = []
    total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        _assert_plain_path(current_path)
        for directory in list(directories):
            child = current_path / directory
            if _is_link_or_reparse(child):
                raise OSError(f'model artifact directory contains a link: {child}')
        directories.sort()
        for filename in sorted(filenames):
            child = current_path / filename
            info = _full_file_signature(
                child, trusted_link_root=trusted_link_root)
            relative = child.relative_to(root).as_posix()
            entries.append({
                'path': relative,
                'size': info['size'],
                'sha256': info['sha256'],
            })
            total += info['size']
    encoded = json.dumps(
        entries, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')
    hexdigest = hashlib.sha256(encoded).hexdigest()
    return {
        'name': root.name,
        'kind': 'directory',
        'size': total,
        'files': len(entries),
        'sha256': hexdigest,
        'sig': hexdigest[:16],
        'sampled': False,
    }


def artifact_signature(path, *, strict=False):
    """Complete SHA-256 identity for one local file or directory artifact.

    Directories are a sorted manifest of relative path, size and full file hash.
    Links/reparse points are rejected so the identity names exactly the bytes the
    child will load. ``strict=False`` preserves the environment-snapshot
    best-effort contract; exact-state callers use ``strict=True``.
    """
    try:
        artifact = Path(path)
        if artifact.is_dir():
            return _directory_signature(artifact)
        return _full_file_signature(artifact)
    except (OSError, ValueError):
        if strict:
            raise
        return None


def _absolute_path(path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def pinned_hf_artifact_signature(
        path, repo_id: str, commit: str, *, strict=False):
    """Full hash of an immutable Hugging Face cache snapshot.

    Hub snapshots normally contain symlinks to the repository's content-addressed
    ``blobs`` directory. Those links are accepted only when the lexical load path
    is under ``models--<repo>/snapshots/<commit>`` and the resolved target stays
    inside that exact repository's blob store.
    """
    try:
        if not re.fullmatch(r'[0-9a-fA-F]{40,64}', str(commit or '')):
            raise OSError('invalid pinned Hugging Face commit')
        artifact = _absolute_path(path)
        parts = artifact.parts
        snapshot_index = None
        for index, part in enumerate(parts[:-1]):
            if (
                part == 'snapshots'
                and index + 1 < len(parts)
                and parts[index + 1].lower() == str(commit).lower()
            ):
                snapshot_index = index
        if snapshot_index is None:
            raise OSError('pinned model path is not under its commit snapshot')
        snapshot_root = Path(*parts[:snapshot_index + 2])
        model_root = snapshot_root.parent.parent
        expected = 'models--' + str(repo_id).replace('/', '--')
        if model_root.name != expected:
            raise OSError('pinned model path does not match its repository')
        try:
            inside = (
                os.path.commonpath((snapshot_root, artifact))
                == str(snapshot_root))
        except ValueError:
            inside = False
        if not inside:
            raise OSError('pinned model path escapes its commit snapshot')
        blobs = model_root / 'blobs'
        if artifact.is_dir():
            return _directory_signature(
                artifact, trusted_link_root=blobs)
        return _full_file_signature(
            artifact, trusted_link_root=blobs)
    except (OSError, ValueError):
        if strict:
            raise
        return None


def _resolve_base_path(base_model):
    """Absolute path of the base weights a launch will actually load, or None.

    An absolute value IS the file (custom weights). A ComfyUI-relative name is
    joined against the configured checkpoint roots — a bounded number of
    `isfile` probes, never the recursive walk `list_models` does. The official
    hosted base resolves to nothing local, which is a truthful None."""
    if not base_model:
        return None
    raw = str(base_model)
    if os.path.isabs(raw):
        return raw if os.path.isfile(raw) else None
    try:
        from . import comfy_model_paths as cmp
        roots = cmp.search_roots('checkpoints')
    except Exception:
        return None
    for root in roots:
        cand = os.path.join(root, raw)
        if os.path.isfile(cand):
            return cand
    return None


def base_model_identity(base_model):
    """Identity of the bytes behind a local base model.

    The full SHA-256 is cached by file identity plus OS change time; restoring
    size and mtime after a middle-byte mutation therefore cannot reuse a stale
    digest.
    """
    path = _resolve_base_path(base_model)
    if not path:
        return None
    info = artifact_signature(path)
    if info is not None:
        # The file name alone leaks nothing, but custom weights may live under the
        # home dir — and the folder is what identifies the file for a human.
        info['folder'] = redact_user_paths(os.path.dirname(path))
    return info


# --- the whole picture ---------------------------------------------------------

def capture(base_model=None) -> dict:
    """The environment stamp for one launch. Never raises; missing probes are
    simply absent keys, so `{}` means "this machine told us nothing" and the UI
    can say so instead of inventing."""
    env = {}
    try:
        from ..version import APP_VERSION
        env['app'] = APP_VERSION
    except Exception:
        pass
    at = aitoolkit_revision()
    if at:
        env['aitoolkit'] = at
    ti = torch_info()
    if ti:
        env.update(ti)
    gpu = gpu_info()
    if gpu:
        env['gpu'] = gpu.get('name')
        if gpu.get('driver'):
            env['gpu_driver'] = gpu['driver']
        if gpu.get('vram_gb'):
            env['gpu_vram_gb'] = gpu['vram_gb']
    base = base_model_identity(base_model)
    if base:
        env['base_file'] = base
    return env
