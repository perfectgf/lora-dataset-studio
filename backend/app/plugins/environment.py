"""Plugin environments use Setup's existing pip queue, runner and log.

The owning marker distinguishes a repairable LDS environment from a directory
the user placed there. A success receipt is written only after the interpreter,
the installed package files and the dependency graph have all been checked.
"""
from __future__ import annotations
from ..timeout_settings import processing_timeout

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from .. import config as cfg

ACTION_PREFIX = 'plugin_environment:'
MARKER = '.lds-plugin-environment.json'
RECEIPT = '.lds-plugin-ready.json'
# Receipts certify this worker policy, separately from the host ABI. Changing
# it creates a new v2 environment and invalidates legacy readiness receipts.
CPU_RUNTIME_POLICY = 'cpu-python-3.10-3.12-v1'
_RUNNING = {}


class EnvironmentError(ValueError):
    pass


def action_id(plugin_id):
    return ACTION_PREFIX + plugin_id


def wants_environment(record, registry=None):
    manifest = record.manifest
    return bool(manifest and (
        (manifest.requirements and not manifest.in_process_requirements) or (registry and any(
            spec['plugin'] == record.id and spec.get('python', 'plugin') == 'plugin'
            and not callable(spec.get('run')) and (spec.get('packages') or spec.get('requirements'))
            for spec in registry.install_actions.values()))))


def check_enabled(record, *, loaded=False):
    if getattr(record, 'pending_restart', False):
        raise EnvironmentError('Restart LDS to load the replaced plugin before installing its environment.')
    if (not record.enabled or (cfg.get('plugins.enabled') or {}).get(record.id) is False
            or record.disabled_by or record.state in ('disabled', 'incompatible', 'misplaced')
            or (loaded and record.state != 'loaded')):
        raise EnvironmentError('Turn this plugin on and restart LDS before installing or running its environment.')


def _inside(path, root):
    root = Path(root).resolve()
    target = Path(os.path.abspath(path))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise EnvironmentError('The plugin environment must stay inside its own data folder.') from exc
    current = root
    for part in relative.parts:
        current = current / part
        # A venv's bin/python symlink is legitimate; directories and our own
        # metadata/requirements files are not allowed to redirect writes.
        if current.resolve() != current:
            raise EnvironmentError('A linked plugin environment path cannot be managed by Setup.')
    return target


def environment_dir(record):
    from .loader import plugin_data_dir
    from .storage import StorageError
    anchor = cfg.data_dir().resolve()
    try:
        data = _inside(plugin_data_dir(record), anchor)
    except StorageError as exc:
        # A refused persistent path belongs on this plugin's diagnostic card;
        # it must not take down the manager or the startup UI registry request.
        raise EnvironmentError(str(exc)) from exc
    if record.manifest.contract.get('schema_version', 1) >= 2:
        # A different package/dependency set gets a different interpreter. Old
        # environments remain usable after code rollback and are never repaired
        # in place for the next release. Keep them out of mutable data snapshots.
        key = hashlib.sha256(json.dumps(_signature(record), sort_keys=True).encode()).hexdigest()
        return _inside(anchor / 'plugin-environments' / record.id / key, anchor)
    return _inside(data / '.venv', anchor)


def _metadata(path):
    try:
        if path.stat().st_nlink != 1:
            raise EnvironmentError('Linked plugin environment metadata cannot be managed by Setup.')
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _owned_dir(record, *, create=False):
    env_dir = environment_dir(record)
    if env_dir.exists() and not env_dir.is_dir():
        raise EnvironmentError('The plugin environment path is occupied by a file.')
    marker = _inside(env_dir / MARKER, env_dir.parent.parent)
    owner = _metadata(marker)
    if owner is not None and owner != {'plugin': record.id}:
        raise EnvironmentError('The existing environment belongs to another plugin.')
    if env_dir.exists() and owner is None and any(env_dir.iterdir()):
        raise EnvironmentError('The existing environment is not managed by LDS. Move it aside before installing.')
    venv_config = _inside(env_dir / 'pyvenv.cfg', env_dir.parent.parent)
    if venv_config.exists() and venv_config.stat().st_nlink != 1:
        raise EnvironmentError('Linked plugin environment metadata cannot be managed by Setup.')
    if create:
        env_dir.mkdir(parents=True, exist_ok=True)
        if owner is None:
            with marker.open('x', encoding='utf-8') as stream:
                json.dump({'plugin': record.id}, stream)
    return env_dir


def interpreter(record, *, ready=True):
    from ..setup_installer import _venv_python
    env_dir = _owned_dir(record)
    python = Path(_venv_python(env_dir))
    if not python.is_file() or not (env_dir / 'pyvenv.cfg').is_file():
        raise EnvironmentError('No environment yet — install the Python environment on this plugin’s card in Setup ▸ Plugins.')
    if ready and _metadata(_inside(env_dir / RECEIPT, env_dir.parent.parent)) != _signature(record):
        raise EnvironmentError('The plugin environment needs installation or repair from Setup ▸ Plugins.')
    return str(python)


def _specs(values):
    result = []
    for value in values:
        value = str(value).split(' #', 1)[0].strip()
        if not value or value.startswith('#'):
            continue
        # Markers are evaluated by pip, but the requirement before one must
        # remain a package spec. Includes, flags, local files and URLs could
        # silently change the destination or replace the CPU package source.
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise EnvironmentError('Plugin requirements must contain package names and versions, without pip options, URLs or local files.') from exc
        if requirement.url:
            raise EnvironmentError('Plugin requirements must contain package names and versions, without pip options, URLs or local files.')
        # Preserve existing receipt fingerprints for already-valid requirements.
        # pip accepts the same PEP 508 spelling; normalization is unnecessary.
        result.append(value)
    return result


def requirements(record, file=None):
    if file is None and record.manifest.in_process_requirements:
        return []
    name = file if file is not None else record.manifest.requirements
    if not name:
        return []
    path = _inside(Path(record.dir).resolve() / name, Path(record.dir).resolve())
    try:
        if path.stat().st_nlink != 1:
            raise EnvironmentError('Linked plugin requirements cannot be installed by Setup.')
        return _specs(path.read_text(encoding='utf-8-sig').splitlines())
    except UnicodeError as exc:
        raise EnvironmentError('The plugin requirements file must use UTF-8 text. Reinstall a corrected plugin archive.') from exc
    except OSError as exc:
        raise EnvironmentError('The plugin requirements file is missing or unreadable. Reinstall its archive.') from exc


def _signature(record):
    content = json.dumps(requirements(record), ensure_ascii=True)
    signature = {'plugin': record.id, 'version': record.manifest.version,
                 'runtime_policy': CPU_RUNTIME_POLICY,
                 'requirements_sha256': hashlib.sha256(content.encode()).hexdigest()}
    if record.manifest.contract.get('schema_version', 1) >= 2:
        signature['python_abi'] = sys.implementation.cache_tag
        signature['platform'] = sys.platform
    return signature


def summary(record, registry):
    if not wants_environment(record, registry):
        return None
    result = {'action': action_id(record.id), 'ready': False, 'can_install': False, 'reason': ''}
    try:
        if action_id(record.id) in registry.install_actions or action_id(record.id) in registry.model_downloads:
            raise EnvironmentError('A plugin action conflicts with the reserved environment action. Reinstall a corrected plugin archive.')
        _owned_dir(record)
        requirements(record)
        try:
            interpreter(record)
            result['ready'] = True
        except EnvironmentError as exc:
            result['reason'] = str(exc)
        check_enabled(record)
        result['can_install'] = True
    except EnvironmentError as exc:
        result['reason'] = str(exc)
    return result


def action_spec(action, registry):
    if not action.startswith(ACTION_PREFIX) or not registry:
        return None
    record = registry.records.get(action[len(ACTION_PREFIX):])
    if not record or not wants_environment(record, registry):
        return None
    if action in registry.install_actions or action in registry.model_downloads:
        return None
    return {'plugin': record.id, 'label': f'{record.manifest.name}: Python environment',
            'python': 'plugin', 'environment': True}


def subprocess_env():
    # pip --isolated ignores option environment variables, but still reads an
    # explicit PIP_CONFIG_FILE, whose target/prefix can redirect every write.
    # Disable all pip config files, including machine and per-venv config.
    # Ordinary transport settings (HTTPS_PROXY, NO_PROXY, CA bundles) remain.
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith('PIP_') and key.upper() not in ('PYTHONHOME', 'PYTHONPATH')}
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
    return env


def running(plugin_id):
    return bool(_RUNNING.get(plugin_id))


def installer():
    """Require the host installer bridge before admitting managed operations."""
    from .. import setup_installer
    if not callable(getattr(setup_installer, 'plugin_install_busy', None)):
        raise EnvironmentError('Managed plugin environments require the plugin installer integration.')
    return setup_installer


@contextmanager
def worker(record):
    """Reserve this product's work, independently of who owns its interpreter."""
    from .lifecycle import state_change_lock
    host_installer = installer()
    with state_change_lock:
        check_enabled(record, loaded=True)
        if host_installer.plugin_install_busy(record.id):
            raise EnvironmentError('The plugin environment is being installed. Wait for Setup to finish.')
        _RUNNING[record.id] = _RUNNING.get(record.id, 0) + 1
    try:
        yield
    finally:
        with state_change_lock:
            _RUNNING[record.id] -= 1
            if not _RUNNING[record.id]:
                del _RUNNING[record.id]


@contextmanager
def use(record):
    with worker(record):
        python = interpreter(record)
        _verify_interpreter(python, environment_dir(record))
        yield python


def _verify_interpreter(python, env_dir):
    from ..setup_installer import _VENV_PY_MIN, _VENV_PY_MAX
    probe = subprocess.run([python, '-I', '-c',
                            'import json,sys; print(json.dumps([sys.prefix,sys.base_prefix,list(sys.version_info[:2])]))'],
                           capture_output=True, text=True, timeout=processing_timeout(30), env=subprocess_env())
    try:
        prefix, base, version = json.loads(probe.stdout)
        valid = probe.returncode == 0 and Path(prefix).resolve() == env_dir.resolve() and prefix != base
        compatible = _VENV_PY_MIN <= tuple(version) <= _VENV_PY_MAX
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise EnvironmentError('The plugin interpreter did not verify as its own isolated environment.')
    if not compatible:
        raise EnvironmentError('Plugin CPU workers require Python 3.10–3.12. Move this plugin’s old environment aside, then install its environment again; its packages have not been changed.')


# Runs in the plugin interpreter. pip keeps an installed version whose files
# were deleted or changed, and pip check reads only metadata, so each package
# is compared with the sizes and hashes pip wrote to its RECORD file. RECORD is
# parsed directly: Distribution.files omits missing files on Python 3.12.
_INTEGRITY_PROBE = r'''
import base64, csv, hashlib, json, re, sys
from importlib import metadata
norm = lambda name: re.sub(r'[-_.]+', '-', str(name or '')).lower()
found = {}
for dist in metadata.distributions():
    try:
        found.setdefault(norm(dist.metadata['Name']), dist)
    except Exception:
        continue
def intact(dist):
    record = dist.read_text('RECORD')
    if record is None:
        return False
    for row in csv.reader(record.splitlines()):
        if len(row) < 2 or '=' not in row[1]:
            continue
        mode, _, value = row[1].partition('=')
        try:
            digest = hashlib.new(mode)
        except ValueError:
            continue
        path = dist.locate_file(row[0])
        if len(row) > 2 and row[2] and path.stat().st_size != int(row[2]):
            return False
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        if base64.urlsafe_b64encode(digest.digest()).rstrip(b'=').decode() != value:
            return False
    return True
def state(name):
    if norm(name) not in found:
        return 'absent'
    try:
        return 'ok' if intact(found[norm(name)]) else 'damaged'
    except Exception:
        return 'damaged'
print(json.dumps({name: state(name) for name in sys.argv[1:]}))
'''


def _package_states(python, names):
    try:
        probe = subprocess.run([python, '-I', '-c', _INTEGRITY_PROBE, *names], capture_output=True, text=True,
                               timeout=processing_timeout(600), env=subprocess_env())
        states = json.loads(probe.stdout) if probe.returncode == 0 else None
    except (subprocess.TimeoutExpired, ValueError):
        states = None
    if not isinstance(states, dict) or set(states) != set(names):
        raise EnvironmentError('The installed package files could not be checked. Move this plugin’s old environment aside, then install its environment again.')
    return states


def _cpu_plan(report):
    plan = []
    for item in report.get('install', []):
        metadata = item.get('metadata') or {}
        name = re.sub(r'[-_.]+', '-', str(metadata.get('name', '')).lower())
        version = str(metadata.get('version', ''))
        _check_cpu_package(name, version)
        download = item.get('download_info') or {}
        url = download.get('url', '')
        digest = (download.get('archive_info') or {}).get('hashes', {}).get('sha256')
        if not url.startswith('https://') or not re.fullmatch(r'[a-f0-9]{64}', str(digest or '')):
            raise EnvironmentError('The resolved package plan has no verified HTTPS archive.')
        plan.append((name, f'{url} --hash=sha256:{digest}'))
    return plan


def _check_cpu_package(name, version):
    if (not name or not version or name.startswith(('nvidia-', 'cuda-', 'cupy', 'jax-cuda', 'tensorrt'))
            or name in ('triton', 'pytorch-triton', 'onnxruntime-gpu', 'tensorflow-gpu', 'paddlepaddle-gpu')
            or (name in ('torch', 'torchvision', 'torchaudio') and not version.endswith('+cpu'))):
        raise EnvironmentError('This environment requests or contains GPU packages. Setup only installs CPU dependencies and will not replace a GPU environment; the plugin needs CPU-compatible requirements.')


def _check_existing_packages(python):
    result = subprocess.run([python, '-I', '-m', 'pip', '--isolated', 'list', '--format=json'],
                            capture_output=True, text=True, timeout=processing_timeout(60), env=subprocess_env())
    if result.returncode != 0:
        raise EnvironmentError('The installed package inventory could not be checked. Repair the plugin environment.')
    for package in json.loads(result.stdout):
        _check_cpu_package(re.sub(r'[-_.]+', '-', package['name'].lower()), package['version'])


def install(action, record, *, extra_packages=(), extra_requirements=None):
    si = installer()
    env = subprocess_env()

    def run(cmd):
        return si._run_pip(action, cmd, env=env)

    check_enabled(record)
    packages = requirements(record) + _specs(extra_packages)
    if extra_requirements:
        packages += requirements(record, extra_requirements)
    signature = _signature(record)
    env_dir = _owned_dir(record, create=True)
    receipt = _inside(env_dir / RECEIPT, env_dir.parent.parent)
    if receipt.exists() and receipt.stat().st_nlink != 1:
        raise EnvironmentError('Linked plugin environment metadata cannot be managed by Setup.')
    python = si._venv_python(env_dir)
    if not Path(python).is_file() or not (env_dir / 'pyvenv.cfg').is_file():
        # A 3.14 host can create a venv successfully while its ML wheels cannot
        # install. Choose the supported base BEFORE creating the worker venv.
        base = si._find_base_python(action)
        if not base:
            return 1
        si._append(action, 'Creating this plugin’s isolated Python environment…')
        if run([base, '-I', '-m', 'venv', str(env_dir)]) != 0:
            return 1
    _verify_interpreter(python, env_dir)
    if receipt.exists():
        receipt.unlink()
    if run([python, '-I', '-m', 'ensurepip', '--upgrade']) != 0:
        return 1
    si._ensure_modern_pip(action, python, isolated_env=env)
    _check_existing_packages(python)
    if packages:
        with tempfile.TemporaryDirectory(prefix='lds-plugin-plan-') as temp:
            report = Path(temp) / 'report.json'
            si._append(action, 'Checking the CPU dependency plan before installing packages…')
            if run([python, '-I', '-m', 'pip', '--isolated', 'install', '--dry-run',
                    '--ignore-installed', '--report', str(report),
                    '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                    '--extra-index-url', si._TORCH_CPU_INDEX, *packages]) != 0:
                return 1
            plan = _cpu_plan(json.loads(report.read_text(encoding='utf-8')))
            if plan:
                names = [name for name, _ in plan]
                damaged = {name for name, state in _package_states(python, names).items() if state == 'damaged'}
                if damaged:
                    si._append(action, 'Reinstalling packages whose installed files are missing or changed: '
                                       + ', '.join(sorted(damaged)))
                # Only damaged packages are forced; intact ones keep pip's
                # already-satisfied shortcut.
                for force in (False, True):
                    lines = [line for name, line in plan if (name in damaged) == force]
                    if not lines:
                        continue
                    locked = Path(temp) / ('repair.txt' if force else 'requirements.txt')
                    locked.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                    if run([python, '-I', '-m', 'pip', '--isolated', 'install', '--no-deps',
                            *(['--force-reinstall'] if force else []),
                            '--no-index', '--require-hashes', '-r', str(locked)]) != 0:
                        return 1
                broken = sorted(name for name, state in _package_states(python, names).items() if state != 'ok')
                if broken:
                    si._append(action, f'Installed files of {", ".join(broken)} are still missing or changed right after '
                                       'installing. Security software or a failing disk can do this; check its quarantine, '
                                       'then repair this environment again.')
                    return 1
    if run([python, '-I', '-m', 'pip', '--isolated', 'check']) != 0:
        return 1
    _verify_interpreter(python, env_dir)
    check_enabled(record)
    if signature != _signature(record):
        raise EnvironmentError('The plugin changed during installation. Restart LDS and install its environment again.')
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(signature, stream)
    si._append(action, 'Plugin environment ready. Python, installed package files and dependency compatibility verified.')
    return 0
