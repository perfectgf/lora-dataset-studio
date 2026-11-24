"""Provision LDS's private CPython base on an explicit Setup install only.

This is the same upstream used by the portable launcher. No system installer,
PATH changes, administrator rights, or packages installed into the host Python.
The shipped SHA-256 pins identify the original publisher archives on the LDS server.
"""
from ..timeout_settings import network_timeout, processing_timeout
import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tarfile
import tempfile
from urllib.parse import quote
import uuid

import requests

from .. import config as cfg


PYTHON_VERSION = '3.12'
_DOWNLOAD_PREFIX = 'https://lds.perfectgf.fr/models/tools/'
_PYTHON_BUILD = '3.12.16+20261123'
_RUNTIME_FILES = {
    'x86_64-pc-windows-msvc':
        ('75988f49393727d7e53b673d50088e7f809a629dd37cb59da3756479bc6ea5c1', 22388474),
    'x86_64-unknown-linux-gnu':
        ('c85811d35971d06612585a057db3b71806378df6ea8476275fb16aa5c9fe56a0', 34741061),
    'aarch64-unknown-linux-gnu':
        ('97bca1d0960240b721b6e58ec2dc080519fcd11f979bf64356434ae9d0f96947', 29460203),
    'x86_64-apple-darwin':
        ('da628e539cebac13f43a3dc42dd99826aeab254387d3bc43478a73267a362713', 24997878),
    'aarch64-apple-darwin':
        ('0de87cfc2649849da10a54084f87fd01fa1e653478e32eee92198c213550e15c', 25260089),
}
_MAX_ARCHIVE = 128 * 1024 * 1024
_MAX_EXTRACTED = 1024 * 1024 * 1024
_HEADERS = {'User-Agent': 'lora-dataset-studio-setup'}


def target_triple():
    arch = platform.machine().lower()
    arch = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(arch, arch)
    system = platform.system()
    if system == 'Windows' and arch == 'x86_64':
        return 'x86_64-pc-windows-msvc'
    if system == 'Darwin' and arch in ('x86_64', 'aarch64'):
        return f'{arch}-apple-darwin'
    if system == 'Linux' and arch in ('x86_64', 'aarch64'):
        if platform.libc_ver()[0] == 'glibc':
            return f'{arch}-unknown-linux-gnu'
    return ''


def status():
    """Cheap platform information, never a download or interpreter scan."""
    return {'python_version': PYTHON_VERSION, 'installation': 'automatic',
            'available': bool(target_triple()), 'isolated': True,
            'requires_admin': False}


def subprocess_env():
    """Pip cannot inherit a target/prefix/config pointing outside our venv."""
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('PIP_', 'PYTHON'))
           and key.upper() not in ('VIRTUAL_ENV', '__PYVENV_LAUNCHER__')}
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
    return env


def _python(root):
    return root / ('python.exe' if os.name == 'nt' else 'bin/python3')


def assert_owned_directory(path, parent):
    """Refuse links/junctions in an app-owned subtree before moving/writing it."""
    path, parent = Path(path).absolute(), Path(parent).resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError('Managed runtime destination is outside its data folder.')
    current = path
    while current != parent:
        if current.is_symlink() or getattr(current, 'is_junction', lambda: False)():
            raise ValueError('Managed runtime destination contains a link or junction.')
        current = current.parent
    if not path.resolve().is_relative_to(parent):
        raise ValueError('Managed runtime destination resolves outside its data folder.')


def _valid_python(python):
    try:
        result = subprocess.run(
            [str(python), '-I', '-c',
             'import sys,ssl,venv,ensurepip; '
             'sys.exit(0 if sys.version_info[:2] == (3,12) else 1)'],
            capture_output=True, timeout=processing_timeout(20),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _asset(triple):
    if triple not in _RUNTIME_FILES:
        raise ValueError('No managed Python build is available for this platform.')
    digest, size = _RUNTIME_FILES[triple]
    filename = f'cpython-{_PYTHON_BUILD}-{triple}-install_only_stripped.tar.gz'
    return {'name': filename, 'size': size, 'digest': 'sha256:' + digest,
            'browser_download_url': _DOWNLOAD_PREFIX + digest + '/' + quote(filename)}


def _download(asset, destination):
    digest = hashlib.sha256()
    size = 0
    with requests.get(asset['browser_download_url'], headers=_HEADERS,
                      timeout=network_timeout((15, 60)), stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('The LDS Python runtime server did not return an archive.')
        with destination.open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                if size > asset['size'] or size > _MAX_ARCHIVE:
                    raise ValueError('The Python download exceeded its declared size.')
                digest.update(chunk)
                output.write(chunk)
    if size != asset['size'] or digest.hexdigest() != asset['digest'][7:].lower():
        raise ValueError('Python download SHA-256 or size mismatch; nothing was installed.')


def _extract(archive, destination):
    """Bounded extraction, only normal files/directories and internal symlinks."""
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        if len(members) > 40000 or sum(m.size for m in members) > _MAX_EXTRACTED:
            raise ValueError('Python archive exceeds its extraction budget.')
        for member in members:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or not name.parts or name.parts[0] != 'python'
                    or '..' in name.parts or '\\' in member.name or ':' in member.name):
                raise ValueError('Unsafe Python archive path.')
            target = destination.joinpath(*name.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Python archive escapes its extraction folder.')
            if member.issym():
                link = PurePosixPath(member.linkname)
                if (link.is_absolute() or '\\' in member.linkname or ':' in member.linkname
                        or not target.parent.joinpath(*link.parts).resolve().is_relative_to(
                            (destination / 'python').resolve())):
                    raise ValueError('Unsafe Python archive link.')
            elif not member.isfile() and not member.isdir():
                raise ValueError('Unsupported Python archive member.')
        # Defer all links until files have been written, so no link can redirect a
        # later write. Check the parent each time as a second containment guard.
        for member in sorted(members, key=lambda m: m.issym()):
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Python archive link redirects another member.')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(member.linkname)


def ensure_python(log):
    """Return a validated managed Python, raising on failure for the install log.

    The previous runtime is preserved until a fully validated replacement exists.
    Staging is private and removed after every failure; retry starts cleanly.
    """
    parent = cfg.data_dir().resolve()
    root = parent / 'runtimes' / 'python-3.12'
    assert_owned_directory(root, parent)
    python = _python(root)
    if python.is_file() and _valid_python(python):
        log('Reusing the LDS-managed Python 3.12 runtime.')
        return str(python)
    triple = target_triple()
    if not triple:
        raise ValueError('Automatic ML Python is unavailable on this OS/architecture.')
    log('Preparing an isolated Python 3.12 for the selected quality tool…')
    asset = _asset(triple)
    log(f"Downloading {asset['name']} ({asset['size'] // (1024 * 1024)} MB).")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.python-setup-', dir=root.parent) as staging:
        staging = Path(staging)
        archive = staging / 'python.tar.gz'
        _download(asset, archive)
        log('Python archive SHA-256 verified; extracting…')
        _extract(archive, staging)
        candidate = staging / 'python'
        if not _valid_python(_python(candidate)):
            raise ValueError('Downloaded Python failed its SSL/venv runtime check.')
        backup = root.with_name(root.name + '.previous-' + uuid.uuid4().hex)
        if root.exists():
            root.rename(backup)
        try:
            candidate.rename(root)
            if not _valid_python(python):
                raise ValueError('Managed Python failed verification at its final path.')
        except Exception:
            if root.exists():
                assert_owned_directory(root, parent)
                shutil.rmtree(root)
            if backup.exists():
                backup.rename(root)
            raise
        # Keep the previous runtime if present: an existing venv might reference
        # its DLLs, and deleting it while a worker runs would break that worker.
    log('LDS-managed Python 3.12 is ready. The app Python was left unchanged.')
    return str(python)
