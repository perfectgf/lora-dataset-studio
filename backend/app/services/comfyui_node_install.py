"""Prepare a pinned node recipe in the configured ComfyUI installation.

Recipes are trusted server registrations, never request bodies. Every archive
is checked against its pinned SHA-256 before anything reaches custom_nodes.
On the validated Windows portable layout, whose Python LDS manages, this worker
also adds the recipe's missing wheels; it never upgrades an existing package,
runs install.py, or starts ComfyUI. A recipe without Python requirements is only
files, so it is prepared into the custom_nodes folder of any configured layout
(Desktop, manual, Linux), with the same path checks. A recipe with requirements
on another layout installs nothing: its plan lists the exact steps instead.
A disk receipt means prepared, not that node classes have loaded.
The caller must re-probe those classes after its separately controlled restart.
Managed node upgrades are deliberately refused until their migration is defined.
"""
from __future__ import annotations
from ..timeout_settings import network_timeout, processing_timeout

from configparser import ConfigParser, Error as ConfigError
from dataclasses import asdict, replace
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from time import monotonic
from urllib.parse import unquote, urlsplit
import zipfile

from filelock import FileLock, Timeout
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version
import requests
from urllib3.exceptions import HTTPError as TransportError

from ..plugins.environment import subprocess_env
from ..plugins.node_recipe import (
    BundledWheel as BundledWheel, NodeRecipe, NodeInstallError, NodeInstallManual, _ID,
    MAX_ARCHIVE_BYTES, _python_spec, _validate, _gpu_package,
)
from ..plugins.install import ArchiveError, _entry_relpath, _validated_entries
from . import comfyui_control as control
from . import plugin_tool_sources

RECEIPT = '.lds-node-prepared.json'
LOCK_NAME = '.lds-node-install.lock'
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_DOWNLOAD_SECONDS = 120
_LOCK = threading.RLock()
_PROBE = (
    'import json,sys,sysconfig,importlib.metadata as m; '
    'print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,'
    '"version":list(sys.version_info[:3]),"abi":sys.implementation.cache_tag,'
    '"install_paths":sysconfig.get_paths(),'
    '"packages":[[d.metadata["Name"],d.version] for d in m.distributions()]}))'
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _safe(path):
    """Check existing ancestors, including hard links on files, before writes."""
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        if os.path.lexists(item):
            if control._is_link_or_reparse(item):
                raise NodeInstallError('Linked custom-node paths cannot be managed by LDS.')
            if item.is_file() and item.stat().st_nlink != 1:
                raise NodeInstallError('Hard-linked custom-node files cannot be managed by LDS.')
    return path


def _target():
    layout = control._validated_portable_layout()
    if layout is None:
        raise NodeInstallError('Select a supported ComfyUI Windows portable installation first.')
    base, python = _safe(layout.base_dir), _safe(layout.python_exe)
    if not control._safe_local_api_url(control.cfg.get('comfyui.api_url')):
        raise NodeInstallError('The configured ComfyUI endpoint does not identify the supported local portable installation.')
    nodes = _safe(base / 'custom_nodes')
    if nodes.exists() and not nodes.is_dir():
        raise NodeInstallError('The ComfyUI custom_nodes path is not a directory.')
    return base, python, nodes


MANUAL_REASON = ('This node pack adds Python packages. LDS installs those automatically only on a '
                 'ComfyUI Windows portable, where it manages ComfyUI’s Python (Setup points at its '
                 'ComfyUI folder and the ComfyUI address is http://127.0.0.1:8188). Nothing was '
                 'installed. To install it yourself:')


def _python_layout():
    """The portable layout whose Python LDS manages, or None for any other layout."""
    layout = control._validated_portable_layout()
    if layout is None or not control._safe_local_api_url(control.cfg.get('comfyui.api_url')):
        return None
    return layout


def _mode(recipe):
    """'python' on the managed portable; elsewhere 'files' for a recipe that is
    only files, 'manual' for one that needs Python packages."""
    if _python_layout() is not None:
        return 'python'
    return 'manual' if recipe.requirements or recipe.wheels else 'files'


def _files_target():
    """The configured ComfyUI and its custom_nodes, whatever the layout."""
    from .. import capabilities
    resolved = capabilities.resolve_comfyui_base(control.cfg.get('comfyui.base_dir') or '')
    if not resolved['valid']:
        raise NodeInstallError('Select your ComfyUI folder in Setup first.')
    base = _safe(resolved['resolved'])
    nodes = _safe(base / 'custom_nodes')
    if nodes.exists() and not nodes.is_dir():
        raise NodeInstallError('The ComfyUI custom_nodes path is not a directory.')
    return base, nodes


def check_target():
    """Raise NodeInstallError when no configured ComfyUI can receive custom nodes."""
    if _python_layout() is not None:
        _target()
    else:
        _files_target()


def _python_hint(base):
    """ComfyUI's own interpreter when the folder shows one. Display only, never run."""
    for candidate in (base.parent / 'python_embeded' / 'python.exe',
                      base / '.venv' / 'Scripts' / 'python.exe', base / 'venv' / 'Scripts' / 'python.exe',
                      base / '.venv' / 'bin' / 'python', base / 'venv' / 'bin' / 'python'):
        if control._real_regular_file(candidate):
            return str(candidate)
    return None


def _manual_steps(recipe, base, destination):
    steps = [f'Download {recipe.url}',
             f'Check that the downloaded file has the SHA-256 {recipe.sha256}',
             f'Extract the folder {recipe.archive_prefix} from it into {destination.parent} '
             f'and rename it to {recipe.folder}' if recipe.archive_prefix
             else f'Extract its contents into {destination}']
    # Same rule as the managed install: a GPU package is never offered, ComfyUI keeps its own.
    gpu = [text for text in recipe.requirements if _gpu_package(canonicalize_name(Requirement(text).name))]
    wanted = [text for text in recipe.requirements if text not in gpu]
    if wanted:
        python = _python_hint(base)
        command = f'"{python}"' if python else 'python'
        where = '' if python else ', with the Python that starts ComfyUI'
        opencv = [text for text in wanted if canonicalize_name(Requirement(text).name) in _OPENCV]
        plain = [text for text in wanted if text not in opencv]
        if plain:
            packages = ' '.join(f'"{text}"' for text in plain)
            names = ', '.join(Requirement(text).name for text in plain)
            pronoun = 'it' if len(plain) == 1 else 'them'
            steps.append(f'If ComfyUI’s Python does not have {names} yet, add {pronoun}{where}: '
                         f'{command} -m pip install {packages}')
        # Its own step: someone with opencv-contrib-python would not see it in
        # "does not have opencv-python yet", and a second build breaks cv2.
        for text in opencv:
            steps.append('If ComfyUI’s Python has no OpenCV yet (opencv-python, opencv-contrib-python '
                         f'or a headless build), add it{where}: {command} -m pip install "{text}"')
    if gpu:
        steps.append('Keep the ' + ', '.join(Requirement(text).name for text in gpu)
                     + ' that ComfyUI already runs; LDS never installs GPU packages.')
    steps.append('Restart ComfyUI.')
    return steps


def manual_steps(recipe):
    """The steps that install `recipe` by hand into the configured ComfyUI."""
    try:
        base, nodes = _files_target()
    except NodeInstallError:
        base = Path('<ComfyUI>')
        nodes = base / 'custom_nodes'
    return _manual_steps(recipe, base, nodes / recipe.folder)


def prepared_version(recipe, destination):
    """The version an LDS receipt in `destination` records for this recipe, else None.

    Read-only. It lets a caller leave an older prepared copy alone, the rule for
    any existing folder, instead of asking for an upgrade that is not supported."""
    try:
        marker = _safe(Path(destination) / RECEIPT)
        if not marker.is_file() or marker.stat().st_size > 1024 * 1024:
            return None
        value = json.loads(marker.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if isinstance(value, dict) and value.get('recipe_id') == recipe.id and isinstance(value.get('version'), str):
        return value['version']
    return None


def _run(python, args):
    args, env = plugin_tool_sources.pip_install(args, subprocess_env())
    try:
        result = subprocess.run([str(python), '-I', *args], capture_output=True,
                                text=True, timeout=processing_timeout(300), env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NodeInstallError('The ComfyUI Python operation could not finish. Its installation was not marked prepared.') from exc
    # Child output can contain credentials, repository URLs or personal paths.
    # Do not relay it through an API/log. The caller receives a fixed diagnosis.
    if result.returncode:
        raise NodeInstallError('The ComfyUI Python dependency check or installation failed. No node was marked prepared.')
    return result.stdout


def _inventory(python):
    try:
        probe = json.loads(_run(python, ['-c', _PROBE]))
        if (Path(probe['executable']).resolve() != python.resolve()
                or Path(probe['prefix']).resolve() != python.parent.resolve()
                or not isinstance(probe['version'], list) or len(probe['version']) != 3
                or not isinstance(probe['abi'], str)):
            raise ValueError
        packages = {}
        paths = probe['install_paths']
        if not isinstance(paths, dict) or not {'purelib', 'platlib', 'scripts', 'data'} <= paths.keys():
            raise ValueError
        for name, value in paths.items():
            if not isinstance(name, str) or not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError
            destination = _safe(Path(value))
            if not destination.is_relative_to(python.parent):
                raise ValueError
        for name, version in probe['packages']:
            key = canonicalize_name(name)
            req = Requirement(f'{key}=={version}')
            if key in packages or req.url or req.marker or req.extras:
                raise ValueError
            packages[key] = version
        binding = {'python': str(python), 'version': probe['version'], 'abi': probe['abi'],
                   'install_paths': paths,
                   'executable_sha256': hashlib.sha256(python.read_bytes()).hexdigest()}
        return packages, binding
    except (ValueError, TypeError, KeyError, OSError) as exc:
        if isinstance(exc, NodeInstallError):
            raise
        raise NodeInstallError('The selected Python did not verify as ComfyUI’s own portable interpreter.') from exc


def _files(directory):
    out = {}
    for path in sorted(directory.rglob('*')):
        _safe(path)
        if path.is_dir():
            continue
        relative = path.relative_to(directory).as_posix()
        # ComfyUI may create bytecode when loading a successfully prepared pack.
        if relative == RECEIPT or '__pycache__' in path.relative_to(directory).parts:
            continue
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise NodeInstallError('The managed custom-node contents cannot be verified.')
        out[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _receipt(recipe, destination, binding):
    if not destination.exists():
        return None
    marker = _safe(destination / RECEIPT)
    try:
        if not destination.is_dir() or not marker.is_file() or marker.stat().st_size > 1024 * 1024:
            raise ValueError
        value = json.loads(marker.read_text(encoding='utf-8'))
        if (value['schema_version'] != 1 or value['recipe_id'] != recipe.id
                or value['target'] != binding or value['state'] != 'prepared'
                or not isinstance(value['owners'], list) or not value['owners']
                or not all(isinstance(o, str) and _ID.fullmatch(o) for o in value['owners'])):
            raise ValueError
        if value['version'] != recipe.version:
            raise NodeInstallError('A different managed node version is installed. Automatic shared-node upgrades are not supported yet.')
        fields = asdict(recipe)
        if _digest(fields) in plugin_tool_sources.H3MOD_RECIPE_DIGESTS:
            trusted = plugin_tool_sources.matches_h3mod_receipt(fields, value, _digest)
        else:
            variants = [fields]
            old_url = plugin_tool_sources.LEGACY_NODE_ORIGINS.get(recipe.sha256)
            if old_url and recipe.url.startswith(plugin_tool_sources.ROOT + '/'):
                # A mirror move preserves the prepared files and their owners.
                variants.append({**fields, 'url': old_url})
            identities = {_digest(variant) for variant in variants}
            if str(_python_spec(recipe.python)) == '>=3.10':
                for variant in variants:
                    identities.add(_digest({k: v for k, v in variant.items() if k != 'python'}))
            trusted = value['recipe_sha256'] in identities
        if not trusted or value['files'] != _files(destination):
            raise NodeInstallError('The managed node recipe or its files changed. Repair requires an explicit reviewed operation.')
        return value
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, NodeInstallError):
            raise
        raise NodeInstallError('An existing custom-node folder is not managed by this LDS recipe; it will not be replaced.') from exc


def _wheel_sources(recipe, wheel_root):
    """Only server-provided plugin provenance, never a path supplied by an API client."""
    if not recipe.wheels:
        return {}
    try:
        if wheel_root is None or not Path(wheel_root).is_absolute():
            raise ValueError
        root = _safe(wheel_root)
        if not root.is_dir():
            raise ValueError
        sources = {}
        for wheel in recipe.wheels:
            path = _safe(root / wheel.filename)
            if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError
            sources[wheel.filename] = path
        return sources
    except (ValueError, OSError, TypeError) as exc:
        raise NodeInstallError('The declared plugin wheel files are missing or have unsafe provenance.') from exc


def _read_wheel(path, digest, destination=None):
    """Bounded hash check, optionally snapshot to the host's private staging folder."""
    _safe(path)
    hasher, size = hashlib.sha256(), 0
    try:
        with path.open('rb') as source:
            if os.fstat(source.fileno()).st_nlink != 1:
                raise ValueError
            # At most one bounded wheel is held; no unverified source is given to pip.
            content = source.read(MAX_ARCHIVE_BYTES + 1)
            size = len(content)
            hasher.update(content)
        _safe(path)
        if size > MAX_ARCHIVE_BYTES or hasher.hexdigest() != digest:
            raise ValueError
        if destination is not None:
            with destination.open('xb') as output:
                output.write(content)
        return size
    except (OSError, ValueError) as exc:
        raise NodeInstallError('A declared plugin wheel is missing, linked or does not match its SHA-256.') from exc


def _bundled_wheels(recipe, wheel_root, temporary, binding):
    sources, bundled, total = _wheel_sources(recipe, wheel_root), {}, 0
    if not sources:
        return bundled
    stage = temporary / 'bundled-wheels'
    stage.mkdir()
    for wheel in recipe.wheels:
        archive = stage / wheel.filename
        total += _read_wheel(sources[wheel.filename], wheel.sha256, archive)
        if total > 256 * 1024 * 1024:
            raise NodeInstallError('The declared plugin wheels exceed the total size limit.')
        name, version, _, _ = parse_wheel_filename(wheel.filename)
        value = {'name': name, 'version': str(version), 'filename': wheel.filename, 'sha256': wheel.sha256}
        _wheel_paths(archive, value, binding)
        bundled[name] = {**value, 'archive': archive}
    return bundled


# Every OpenCV build installs the same `cv2` module, and two of them in one
# Python overwrite each other's files. So whichever build is installed answers
# for an OpenCV requirement, and a second build is never added.
_OPENCV = frozenset({'opencv-python', 'opencv-python-headless',
                     'opencv-contrib-python', 'opencv-contrib-python-headless'})


def _present(name, installed):
    """The installed distribution that answers for the requirement `name`, or None."""
    if name in installed:
        return name
    if name in _OPENCV:
        return next(iter(sorted(_OPENCV & installed.keys())), None)
    return None


def _serves(req, version):
    """Whether an installed `version` serves the recipe's exact pin `req`.

    THE DEPENDENCY RULE (maintainer's call, 2026-11-24). A recipe pins the
    version LDS installs when a package is missing. A version already installed
    with the same major release is accepted as it is and never replaced. With
    exact pins alone, almost every portable was refused, and two recipes pinning
    different numpy releases could never both install. Another major release is
    a conflict, and nothing is replaced.
    """
    if req.specifier.contains(version, prereleases=True):
        return True
    pins = [spec.version for spec in req.specifier if spec.operator == '==']
    try:
        return len(pins) == 1 and Version(version).major == Version(pins[0]).major
    except InvalidVersion:
        return False


def _dependency_plan(recipe, python, installed, temporary, bundled=None):
    bundled = bundled or {}
    missing = []
    for text in recipe.requirements:
        req = Requirement(text)
        present = _present(canonicalize_name(req.name), installed)
        if present is None:
            missing.append(text)
        elif not _serves(req, installed[present]):
            raise NodeInstallError('A node dependency conflicts with the existing ComfyUI environment. No installed package will be replaced.')
    if not missing:
        return []
    constraints = temporary / 'installed.txt'
    limits = [f'{name}=={version}' for name, version in sorted(installed.items())]
    # A constraint chooses the author's verified wheel only if resolution needs it.
    # It does not install unused bundled packages, or replace an installed version.
    limits.extend(f'{name} @ {wheel["archive"].as_uri()}' for name, wheel in sorted(bundled.items())
                  if name not in installed)
    constraints.write_text('\n'.join(limits), encoding='utf-8')
    report = temporary / 'pip-report.json'
    _run(python, ['-m', 'pip', '--isolated', 'install', '--dry-run', '--report', str(report),
                  '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                  '--no-input', '--disable-pip-version-check', '--no-cache-dir',
                  # Only what is missing: an accepted installed version differs
                  # from its pin, and the constraints keep it as it is.
                  '--constraint', str(constraints), *missing])
    try:
        data = json.loads(report.read_text(encoding='utf-8'))
        if data['version'] != '1' or not isinstance(data['install'], list):
            raise ValueError
        wheels, seen = [], set()
        for item in data['install']:
            name = canonicalize_name(item['metadata']['name'])
            version = item['metadata']['version']
            Requirement(f'{name}=={version}')
            url = item['download_info']['url']
            digest = item['download_info']['archive_info']['hashes']['sha256']
            if (name in installed or name in seen or _gpu_package(name)
                    or (name in _OPENCV and _OPENCV & (installed.keys() | seen))
                    or not re.fullmatch(r'[a-f0-9]{64}', digest)):
                raise ValueError
            if name in bundled:
                supplied = bundled[name]
                if (url != supplied['archive'].as_uri() or digest != supplied['sha256']
                        or version != supplied['version']):
                    raise ValueError
                source = {'filename': supplied['filename']}
            else:
                if not plugin_tool_sources.is_installer_wheel(name, url, digest):
                    raise ValueError
                source = {'url': url}
            seen.add(name)
            wheels.append({'name': name, 'version': version, **source, 'sha256': digest})
        resolved = {**installed, **{w['name']: w['version'] for w in wheels}}
        for r in map(Requirement, recipe.requirements):
            present = _present(canonicalize_name(r.name), resolved)
            if present is None or not _serves(r, resolved[present]):
                raise ValueError
        return sorted(wheels, key=lambda wheel: wheel['name'])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise NodeInstallError('The dependency plan would replace existing packages, add a GPU stack or a second '
                               'OpenCV build, or use unverifiable wheels. Nothing was installed.') from exc


def _layout_plan(recipe, owner, mode):
    """Plan for a layout whose Python LDS does not manage: files only, or steps."""
    base, nodes = _files_target()
    destination = _safe(nodes / recipe.folder)
    public = {'recipe_id': recipe.id, 'version': recipe.version,
              'expected_classes': list(recipe.expected_classes),
              'packages_to_add': [], 'restart_required': True}
    if mode == 'manual':
        if os.path.lexists(destination):
            raise NodeInstallError('An existing custom-node folder is not managed by this LDS recipe; it will not be replaced.')
        identity = {'recipe': asdict(recipe), 'owner': owner, 'mode': mode, 'base_dir': str(base)}
        return {'plan_id': _digest(identity), 'state': 'manual', **public,
                'manual_steps': _manual_steps(recipe, base, destination)}, None
    binding = {'layout': 'custom_nodes', 'base_dir': str(base)}
    receipt = _receipt(recipe, destination, binding)
    state = 'install' if receipt is None else 'current' if owner in receipt['owners'] else 'shared'
    identity = {'recipe': asdict(recipe), 'owner': owner, 'target': binding, 'receipt': receipt}
    return ({'plan_id': _digest(identity), 'state': state, **public},
            (None, destination, binding, {}, receipt, [], {}))


def _plan(recipe, owner, temporary, wheel_root=None):
    _validate(recipe, owner)
    # Presence, hashes and link provenance are checked even when dependencies already exist.
    sources = _wheel_sources(recipe, wheel_root)
    for wheel in recipe.wheels:
        _read_wheel(sources[wheel.filename], wheel.sha256)
    mode = _mode(recipe)
    if mode != 'python':
        return _layout_plan(recipe, owner, mode)
    base, python, nodes = _target()
    installed, binding = _inventory(python)
    required = _python_spec(recipe.python)
    try:
        actual = Version('.'.join(str(part) for part in binding['version']))
    except ValueError as exc:
        raise NodeInstallError('The ComfyUI Python version could not be verified.') from exc
    if not required.contains(actual, prereleases=True):
        raise NodeInstallError(f'ComfyUI uses Python {actual}; this node recipe requires Python {required}. '
                               'Select a compatible ComfyUI installation before preparing these nodes.')
    binding['base_dir'] = str(base)
    destination = _safe(nodes / recipe.folder)
    receipt = _receipt(recipe, destination, binding)
    _run(python, ['-m', 'pip', '--isolated', 'check'])
    bundled = _bundled_wheels(recipe, wheel_root, temporary, binding)
    wheels = _dependency_plan(recipe, python, installed, temporary, bundled)
    state = 'install' if receipt is None else 'current' if owner in receipt['owners'] else 'shared'
    identity = {'recipe': asdict(recipe), 'owner': owner, 'target': binding,
                'installed': installed, 'receipt': receipt, 'wheels': wheels}
    public = {'plan_id': _digest(identity), 'state': state, 'recipe_id': recipe.id,
              'version': recipe.version, 'expected_classes': list(recipe.expected_classes),
              'packages_to_add': [{k: w[k] for k in ('name', 'version')} for w in wheels],
              'restart_required': True}
    return public, (python, destination, binding, installed, receipt, wheels, bundled)


def plan(recipe: NodeRecipe, *, owner: str, wheel_root=None) -> dict:
    """Read target/inventory and resolve only missing dependencies, without writes to ComfyUI."""
    with _LOCK, tempfile.TemporaryDirectory(prefix='lds-node-plan-') as temporary:
        return _plan(recipe, owner, Path(temporary), wheel_root)[0]


def _download(recipe, archive):
    hasher, size = hashlib.sha256(), 0
    started = monotonic()
    try:
        with requests.get(recipe.url, stream=True, allow_redirects=False, timeout=network_timeout((10, 30)),
                          headers={'Accept-Encoding': 'identity'}) as response:
            if response.status_code != 200:
                raise NodeInstallError('The pinned node archive was not available without a redirect.')
            if int(response.headers.get('Content-Length', 0)) > MAX_ARCHIVE_BYTES:
                raise NodeInstallError('The custom-node archive exceeds the download limit.')
            if response.headers.get('Content-Encoding', 'identity').casefold() not in ('', 'identity'):
                raise NodeInstallError('The pinned archive must be served without HTTP content encoding.')
            with archive.open('xb') as target:
                while True:
                    if monotonic() - started > MAX_DOWNLOAD_SECONDS:
                        raise NodeInstallError('The custom-node download exceeded its time limit.')
                    # Unlike iter_content/read(amt), read1 returns available bytes
                    # without waiting to fill a chunk on a slow, continuous stream.
                    # The deadline can overshoot by at most the bounded socket read.
                    chunk = response.raw.read1(64 * 1024, decode_content=False)
                    if monotonic() - started > MAX_DOWNLOAD_SECONDS:
                        raise NodeInstallError('The custom-node download exceeded its time limit.')
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise NodeInstallError('The custom-node archive exceeds the download limit.')
                    hasher.update(chunk)
                    target.write(chunk)
    except (requests.RequestException, TransportError, OSError, ValueError) as exc:
        if isinstance(exc, NodeInstallError):
            raise
        raise NodeInstallError('The pinned custom-node archive could not be downloaded.') from exc
    if hasher.hexdigest() != recipe.sha256:
        raise NodeInstallError('The custom-node archive hash does not match the registered recipe.')


def _extract(recipe, archive, stage):
    prefix = recipe.archive_prefix + '/' if recipe.archive_prefix else ''
    try:
        with zipfile.ZipFile(archive) as zf:
            entries = _validated_entries(zf)
            selected = []
            for info, relative in entries:
                if info.is_dir() and relative == recipe.archive_prefix:
                    continue
                if prefix and not relative.startswith(prefix):
                    raise ArchiveError('unexpected archive root')
                relative = relative[len(prefix):]
                if (relative == RECEIPT or info.file_size > MAX_FILE_BYTES
                        or info.file_size > max(info.compress_size, 1) * 200):
                    raise ArchiveError('reserved metadata or unsafe expansion')
                selected.append((info, relative))
            if not any(r == '__init__.py' and not i.is_dir() for i, r in selected):
                raise ArchiveError('missing node package entry point')
            stage.mkdir()
            for info, relative in selected:
                target = _safe(stage / relative)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as source, target.open('xb') as output:
                        copied = 0
                        while chunk := source.read(64 * 1024):
                            copied += len(chunk)
                            if copied > info.file_size:
                                raise ArchiveError('archive entry expanded beyond its declared size')
                            output.write(chunk)
    except (ArchiveError, zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise NodeInstallError('The custom-node archive has unsafe paths, links, metadata or expanded contents.') from exc


def _wheel_paths(archive, wheel, binding):
    """Map wheel destinations before pip can overwrite another distribution's modules."""
    paths = binding['install_paths']
    try:
        with zipfile.ZipFile(archive) as zf:
            entries = [(info, rel) for info, rel in _validated_entries(zf) if not info.is_dir()]
            metadata_paths = [rel for _, rel in entries if re.fullmatch(r'[^/]+\.dist-info/METADATA', rel)]
            if len(metadata_paths) != 1:
                raise ValueError
            info_dir = metadata_paths[0].split('/')[0]

            def text(name):
                info = zf.getinfo(name)
                if info.file_size > 1024 * 1024:
                    raise ValueError
                return zf.read(info).decode('utf-8')

            metadata = Parser().parsestr(text(metadata_paths[0]))
            descriptor = Parser().parsestr(text(info_dir + '/WHEEL'))
            pure = descriptor['Root-Is-Purelib']
            info_name, info_version = info_dir.removesuffix('.dist-info').rsplit('-', 1)
            if (canonicalize_name(metadata['Name']) != wheel['name'] or metadata['Version'] != wheel['version']
                    or canonicalize_name(info_name) != wheel['name'] or Version(info_version) != Version(wheel['version'])
                    or descriptor['Wheel-Version'] != '1.0' or pure not in ('true', 'false')):
                raise ValueError
            if any(Requirement(value).url for value in metadata.get_all('Requires-Dist', [])):
                raise ValueError
            library = Path(paths['purelib' if pure == 'true' else 'platlib'])
            destinations = set()
            for _, relative in entries:
                parts = relative.split('/')
                if parts[0].endswith('.data'):
                    if (parts[0] != info_dir.removesuffix('.dist-info') + '.data' or len(parts) < 3
                            or parts[1] not in ('purelib', 'platlib', 'data')):
                        # Arbitrary wheel scripts/headers require a separate qualified mapping.
                        raise ValueError
                    target = Path(paths[parts[1]]).joinpath(*parts[2:])
                else:
                    target = library / relative
                if target.suffix.casefold() == '.pth' or target.name in ('sitecustomize.py', 'usercustomize.py'):
                    raise ValueError
                destinations.add(_safe(target))
            # pip also writes metadata which need not occur in the archive.
            destinations.add(library / info_dir)
            for name in ('INSTALLER', 'REQUESTED', 'direct_url.json', 'RECORD'):
                destinations.add(library / info_dir / name)
            entry_file = info_dir + '/entry_points.txt'
            if any(rel == entry_file for _, rel in entries):
                parser = ConfigParser(interpolation=None)
                parser.optionxform = str
                parser.read_string(text(entry_file))
                for group in ('console_scripts', 'gui_scripts'):
                    for name in parser[group] if parser.has_section(group) else ():
                        if (not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}', name)
                                or _entry_relpath(name) != name or '/' in name
                                or os.path.splitext(name)[1].casefold().startswith('.py')
                                or name.casefold() in ('pip', 'easy_install')):
                            raise ValueError
                        for suffix in ('', '.exe', '-script.py', '.exe.manifest'):
                            destinations.add(_safe(Path(paths['scripts']) / (name + suffix)))
            return destinations
    except (ValueError, TypeError, KeyError, OSError, ConfigError, zipfile.BadZipFile) as exc:
        raise NodeInstallError('A dependency wheel has unsupported paths, metadata or interpreter hooks. Nothing was installed.') from exc


def _free_wheel_paths(destinations):
    for destination in destinations:
        _safe(destination)
        if destination.exists() or any(parent.is_file() for parent in destination.parents):
            raise NodeInstallError('A dependency wheel would overwrite an existing ComfyUI file. Nothing was installed.')


def _verified_wheels(recipe, wheels, binding, temporary, bundled=None):
    destinations, local, total = set(), [], 0
    for index, wheel in enumerate(wheels):
        filename = wheel.get('filename') or unquote(urlsplit(wheel['url']).path.rsplit('/', 1)[-1])
        if _entry_relpath(filename) != filename or '/' in filename:
            raise NodeInstallError('A dependency wheel has an invalid archive name.')
        directory = temporary / f'wheel-{index}'
        directory.mkdir()
        archive = directory / filename
        if 'filename' in wheel:
            _read_wheel(bundled[wheel['name']]['archive'], wheel['sha256'], archive)
        else:
            _download(replace(recipe, url=wheel['url'], sha256=wheel['sha256']), archive)
        total += archive.stat().st_size
        if total > 256 * 1024 * 1024:
            raise NodeInstallError('The custom-node dependency download plan is too large.')
        additions = _wheel_paths(archive, wheel, binding)
        if destinations & additions:
            raise NodeInstallError('Two dependency wheels would write the same ComfyUI files. Nothing was installed.')
        destinations.update(additions)
        local.append(f'{archive.as_uri()} --hash=sha256:{wheel["sha256"]}')
    _free_wheel_paths(destinations)
    return local, destinations


def _write_receipt(directory, value):
    target = _safe(directory / RECEIPT)
    temporary = _safe(directory / (RECEIPT + '.tmp'))
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def prepare(recipe: NodeRecipe, *, owner: str, plan_id: str, log=None, wheel_root=None) -> dict:
    """Execute the still-current plan; installation errors do not promise pip rollback."""
    emit = log if callable(log) else lambda _message: None
    _validate(recipe, owner)
    with _LOCK:
        mode = _mode(recipe)
        if mode == 'manual':
            public = plan(recipe, owner=owner, wheel_root=wheel_root)
            if not isinstance(plan_id, str) or public['plan_id'] != plan_id:
                raise NodeInstallError('The custom-node plan or target changed. Review a fresh installation plan.')
            raise NodeInstallManual(MANUAL_REASON, public['manual_steps'])
        if mode == 'python':
            base, _, _ = _target()
            lock_path = _safe(base.parent / LOCK_NAME)
        else:
            # Not the parent of ComfyUI, which may belong to someone else (a
            # Documents folder, a container's root). custom_nodes is ignored by
            # ComfyUI's own .gitignore and its loader skips a file that is not
            # .py, so the lock a POSIX filelock leaves behind there is inert.
            base, nodes = _files_target()
            nodes.mkdir(exist_ok=True)
            lock_path = _safe(nodes / LOCK_NAME)
        try:
            with FileLock(str(lock_path), timeout=0):
                return _prepare(recipe, owner, plan_id, emit, base, wheel_root)
        except Timeout as exc:
            raise NodeInstallError('Another LDS custom-node installation is using this ComfyUI environment.') from exc


def _prepare(recipe, owner, plan_id, emit, locked_base, wheel_root=None):
    with tempfile.TemporaryDirectory(prefix='lds-node-prepare-') as temporary:
        temporary = Path(temporary)
        public, internal = _plan(recipe, owner, temporary, wheel_root)
        if not isinstance(plan_id, str) or public['plan_id'] != plan_id or internal is None:
            raise NodeInstallError('The custom-node plan or target changed. Review a fresh installation plan.')
        python, destination, binding, installed, receipt, wheels, bundled = internal
        if binding['base_dir'] != str(locked_base):
            raise NodeInstallError('The ComfyUI target changed before its installation lock was acquired.')
        if receipt is None:
            emit('Downloading and checking the pinned custom-node archive.')
            archive, stage = temporary / 'nodes.zip', temporary / 'pack'
            _download(recipe, archive)
            _extract(recipe, archive, stage)
        else:
            stage = destination
        prepared_files = _files(stage)
        if python is None:
            # Files only: no interpreter is involved, so the folder is what is rechecked.
            base_now, _ = _files_target()
            if str(base_now) != binding['base_dir']:
                raise NodeInstallError('The ComfyUI environment changed during preparation. Review a fresh plan.')
            stage_parent = base_now
        else:
            local_wheels, wheel_destinations = _verified_wheels(recipe, wheels, binding, temporary, bundled)
            # Revalidate target and inventory after the download, before touching its environment.
            base_now, python_now, _ = _target()
            fresh, fresh_binding = _inventory(python_now)
            fresh_binding['base_dir'] = str(base_now)
            if fresh_binding != binding or fresh != installed:
                raise NodeInstallError('The ComfyUI environment changed during preparation. Review a fresh plan.')
            sources = _wheel_sources(recipe, wheel_root)
            for wheel in recipe.wheels:
                _read_wheel(sources[wheel.filename], wheel.sha256)
            stage_parent = base_now.parent
        touched = False
        try:
            if python is not None:
                if wheels:
                    emit('Adding the verified missing wheels to ComfyUI Python; existing packages are pinned.')
                    locked = temporary / 'wheels.txt'
                    locked.write_text('\n'.join(local_wheels), encoding='utf-8')
                    _free_wheel_paths(wheel_destinations)
                    touched = True
                    _run(python, ['-m', 'pip', '--isolated', 'install', '--no-deps', '--no-index',
                                  '--only-binary=:all:', '--require-hashes', '--no-input',
                                  '--disable-pip-version-check', '--no-cache-dir', '--no-compile', '-r', str(locked)])
                _run(python, ['-m', 'pip', '--isolated', 'check'])
                after, after_binding = _inventory(python)
                after_binding['base_dir'] = str(base_now)
                expected = {**installed, **{w['name']: w['version'] for w in wheels}}
                final_base, final_python, _ = _target()
                if (after != expected or after_binding != binding
                        or final_base != base_now or final_python != python):
                    raise NodeInstallError('The dependency inventory changed unexpectedly.')
            if (_files(stage) != prepared_files
                    or (receipt is not None and _receipt(recipe, destination, binding) != receipt)):
                raise NodeInstallError('The custom-node files or receipt changed during preparation.')
            owners = sorted(set((receipt or {}).get('owners', [])) | {owner})
            value = {'schema_version': 1, 'recipe_id': recipe.id, 'version': recipe.version,
                     'recipe_sha256': _digest(asdict(recipe)), 'target': binding,
                     'owners': owners, 'state': 'prepared', 'files': prepared_files}
            _write_receipt(stage, value)
            if receipt is None:
                # Stage on the destination volume, outside custom_nodes so ComfyUI cannot import it.
                import shutil
                with tempfile.TemporaryDirectory(prefix='.lds-node-stage-', dir=stage_parent) as sibling:
                    final_stage = Path(sibling) / 'pack'
                    shutil.copytree(stage, final_stage)
                    _safe(destination)
                    destination.parent.mkdir(exist_ok=True)
                    if destination.exists():
                        raise NodeInstallError('The destination appeared during preparation; it was not replaced.')
                    final_stage.rename(destination)
        except (NodeInstallError, OSError) as exc:
            raise NodeInstallError(
                'Custom-node preparation did not finish. Missing dependencies may have been added; '
                'no dependency rollback is claimed. Review the environment before retrying.' if touched
                else 'Custom-node preparation did not finish. No dependency installation was performed.',
                dependencies_may_have_changed=touched) from exc
        emit('Node files are prepared. Recheck the expected classes after the controlled ComfyUI restart.')
        return {**public, 'state': 'prepared', 'restart_required': True}
