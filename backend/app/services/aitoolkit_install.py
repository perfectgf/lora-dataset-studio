"""Install AI Toolkit for local training, from Setup, on explicit request.

Public core: nothing here imports a member-only module. The installer owns ONE
new folder. It never writes into an existing AI Toolkit, never replaces a
configured one that works, and only deletes what it can prove it created (its
staging folder, or the Python environment inside a folder carrying its stamp).

Steps, each idempotent. The journal under data/aitoolkit-install/ records the
completed ones, so Install after a failure or a cancel resumes where it stopped:

  source  git clone of branch main, reset to the pinned commit; without git,
          the GitHub archive of that same commit
  python  a venv inside the folder, from a Python 3.10-3.12
  torch   the CUDA build AI Toolkit documents for that commit and this driver
  deps    requirements.txt, with that torch build held by a constraint
  check   imports, then the torch/CUDA probe the training preflight uses
  config  aitoolkit.dir and aitoolkit.python, then the probes run again
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import psutil

from .. import config as cfg

# AI Toolkit as LDS installs it, read at the source rather than in a local clone:
# github.com/ostris/ai-toolkit at 4a7f28532ed6 (main on 2026-11-10, version 0.13.42).
#   README "Manual installation" (Windows and Linux):
#     pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0
#       --index-url https://download.pytorch.org/whl/cu130
#     pip install -r requirements.txt
#   manager/spec.py: TORCH, PYTORCH_INDEX and _cuda_flavor -- cu130 for a driver
#     that supports CUDA 13.0, cu126 from CUDA 12.6 when no Blackwell card is
#     present, otherwise a driver update; Python 3.12 recommended (README: >=3.10).
# Move every field together with `commit`, after reading both files again.
AITOOLKIT_RELEASE = {
    'repo': 'https://github.com/ostris/ai-toolkit',
    'branch': 'main',
    'commit': '4a7f28532ed6fe81d9f8c00718878bb3b8739015',
    'version': '0.13.42',
    'torch': (('torch', '2.13.0'), ('torchvision', '0.28.0'), ('torchaudio', '2.11.0')),
    'torch_index': 'https://download.pytorch.org/whl/',
    'cuda_builds': (((13, 0), 'cu130'), ((12, 6), 'cu126')),
    'python': ((3, 10), (3, 12)),
}

# Free space needed on the destination drive before anything is downloaded, by
# PyTorch build. A real install on Windows (2026-11-15, cu130) left 4.5 GB on
# disk, 4.4 GB of it the environment. The peak comes while pip unpacks PyTorch:
# its wheel (1.92 GB for cu130, 2.59 GB for cu126, HEAD requests on
# download.pytorch.org) is still in pip's temporary folder, which stays on the
# same drive (TMP points inside the destination): about 6 GB for cu130. cu126
# was not installed for real; scaled from its wheel, about 8 GB. Each figure
# keeps about 2 GB free at that peak. The first estimate, 12 GB for every
# build, refused machines that had the room.
MIN_FREE_GB = {'cu130': 8, 'cu126': 10}
# Still needed to resume once PyTorch is installed: the requirements and pip's
# temporary files (roughly 2-3 GB).
RESUME_FREE_GB = 4

VENV_DIR = 'venv'            # the README's name; config.aitoolkit_derived_python finds it
STAMP = '.lds-aitoolkit-install.json'
TMP_DIR = '.lds-install-tmp'
_ARCHIVE_MAX = 512 * 1024 * 1024
_EXTRACT_MAX = 2 * 1024 * 1024 * 1024
_NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

STEPS = ('source', 'python', 'torch', 'deps', 'check', 'config')
# What image and video training import, as the trainer update checks it.
# torchaudio is installed with the pinned trio but not a gate: AI Toolkit's own
# manager treats it as maintenance-mode and version-agnostic.
_IMPORTS = 'import torch, torchvision, diffusers, transformers, accelerate, safetensors'


class Cancelled(Exception):
    pass


class InstallError(Exception):
    """A refusal or failure whose text is shown to the user as is."""


# --- Machine facts -------------------------------------------------------------

def _run_quiet(argv, timeout=15):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def nvidia_driver():
    """GPUs and the highest CUDA version the driver supports, as AI Toolkit's
    manager/detect.py reads them, or None when nvidia-smi does not answer."""
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    csv = _run_quiet([smi, '--query-gpu=name,driver_version,compute_cap',
                      '--format=csv,noheader'])
    if not csv:
        csv = _run_quiet([smi, '--query-gpu=name,driver_version', '--format=csv,noheader'])
    if not csv:
        return None
    gpus, driver = [], ''
    for line in csv.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            gpus.append({'name': parts[0], 'compute_cap': parts[2] if len(parts) > 2 else ''})
            driver = parts[1]
    if not gpus:
        return None
    # Windows/WDDM prints "CUDA UMD Version:" on recent drivers, Linux "CUDA Version:".
    match = re.search(r'CUDA(?:\s+[A-Z]+)?\s+Version:\s*([0-9]+)\.([0-9]+)', _run_quiet([smi]) or '')
    return {'gpus': gpus, 'driver': driver,
            'cuda': (int(match.group(1)), int(match.group(2))) if match else None}


def cuda_build(nvidia):
    """(build, '') for the PyTorch flavour this driver can run, or (None, refusal)."""
    if not nvidia or not nvidia.get('gpus'):
        return None, ('AI Toolkit trains on an NVIDIA GPU, and none was found on this '
                      'machine (nvidia-smi did not answer). Install the NVIDIA driver if '
                      'the card is there, or use Cloud training.')
    cuda = nvidia.get('cuda')
    if cuda is None:
        # Upstream's rule when the driver is present but its CUDA level unknown.
        return AITOOLKIT_RELEASE['cuda_builds'][0][1], ''
    caps = []
    for gpu in nvidia['gpus']:
        with contextlib.suppress(TypeError, ValueError):
            caps.append(float(gpu.get('compute_cap')))
    newest = AITOOLKIT_RELEASE['cuda_builds'][0][1]
    for minimum, build in AITOOLKIT_RELEASE['cuda_builds']:
        if tuple(cuda) >= minimum:
            if build != newest and any(c >= 12.0 for c in caps):
                return None, (f'This RTX 50-series card needs the {newest} build of PyTorch, '
                              f'and this NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]} only. '
                              'Update the driver (580 or later), then check again.')
            return build, ''
    return None, (f'This NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]}, too old for the '
                  f'PyTorch {dict(AITOOLKIT_RELEASE["torch"])["torch"]} that AI Toolkit uses. '
                  'Update the driver (580 or later), then check again.')


def _local_training_running() -> bool:
    try:
        from . import lora_training as lt
        state = lt.queue_manager._get_system_state
        if not state('training_in_progress', False):
            return False
        return not lt._training_process_is_definitely_dead(state('training_pid', None))
    except Exception:
        return False


def _python_preview():
    """The base Python a venv would be made from, WITHOUT downloading anything."""
    from .. import setup_installer as si
    low, high = AITOOLKIT_RELEASE['python']
    for exe in si._base_python_candidates():
        ver = si._python_minor(exe)
        if ver is not None and low <= ver <= high:
            return {'path': exe, 'version': f'{ver[0]}.{ver[1]}', 'download': False}
    from . import managed_python
    if managed_python.target_triple():
        return {'path': '', 'version': managed_python.PYTHON_VERSION, 'download': True}
    return None


def machine_facts() -> dict:
    from .. import capabilities
    from .updater import is_pinokio_runtime
    git = shutil.which('git')
    return {
        'docker': capabilities.setup_is_docker_runtime(),
        'pinokio': is_pinokio_runtime(),
        'os': platform.system(),
        'arch': platform.machine().lower(),
        'nvidia': nvidia_driver(),
        'git': git or '',
        'configured': capabilities.probe_aitoolkit(),
        'training': _local_training_running(),
        'python': _python_preview(),
    }


# --- Journal -------------------------------------------------------------------

def _journal_path() -> Path:
    return cfg.data_dir() / 'aitoolkit-install' / 'state.json'


def load_journal():
    try:
        return json.loads(_journal_path().read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None


def _write_journal(journal):
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(journal, indent=2), encoding='utf-8')
    os.replace(pending, path)


def _identity(pid):
    try:
        return {'pid': pid, 'created': psutil.Process(pid).create_time()}
    except psutil.Error:
        return None


def _alive(identity) -> bool:
    if not identity:
        return False
    try:
        return abs(psutil.Process(identity['pid']).create_time() - identity['created']) < 0.01
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except (psutil.AccessDenied, KeyError, TypeError):
        return True     # unreadable is not proof that it stopped


def _kill_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.Error:
        return
    for proc in (*reversed(children), parent):
        with contextlib.suppress(psutil.Error):
            proc.kill()


def _pending(journal, dest) -> bool:
    return bool(journal and journal.get('state') != 'done'
                and os.path.normcase(journal.get('dest') or '') == os.path.normcase(str(dest)))


# --- Destination ---------------------------------------------------------------

def _is_dir_empty(path: Path) -> bool:
    try:
        return path.is_dir() and not any(path.iterdir())
    except OSError:
        return False


def default_destination(facts=None, journal=None) -> Path:
    """Beside the LDS folder: the disk the user already chose for this kind of
    tool, outside LDS's own files (an update or a ZIP swap never reaches it),
    and a plain AI Toolkit checkout its own tools can still manage. Pinokio
    keeps its apps in one folder it lists itself, so there the home folder is
    used. An unfinished install keeps its folder."""
    journal = load_journal() if journal is None else journal
    if journal and journal.get('state') != 'done' and journal.get('dest'):
        return Path(journal['dest'])
    pinokio = bool((facts or {}).get('pinokio'))
    base = Path.home() if pinokio else cfg.REPO_ROOT.resolve().parent
    for index in range(1, 10):
        cand = base / ('ai-toolkit' if index == 1 else f'ai-toolkit-{index}')
        if not cand.exists() or _is_dir_empty(cand):
            return cand
    return base / 'ai-toolkit'


def _free_gb(path: Path):
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1e9, str(probe.anchor or probe)
    except OSError:
        return None, str(path.anchor or path)


def _destination_problem(dest: Path, journal):
    """(state, message) for a folder that cannot receive a fresh install, else None."""
    if not dest.is_absolute():
        return 'blocked', 'Use a full folder path for AI Toolkit.'
    if not dest.name:
        return 'blocked', 'Choose a folder for AI Toolkit, not the root of a drive.'
    repo = cfg.REPO_ROOT.resolve()
    try:
        resolved = dest.resolve()
    except OSError:
        resolved = dest
    if resolved == repo or resolved.is_relative_to(repo):
        return 'blocked', ('Choose a folder outside the LDS folder: LDS updates replace '
                           'the files there.')
    if dest.exists() and not dest.is_dir():
        return 'blocked', 'This path is a file. Choose a folder for AI Toolkit.'
    if (dest / 'run.py').is_file() and not _pending(journal, dest):
        return 'existing', ('This folder already holds AI Toolkit. Select it with '
                            '“Use this folder” instead of installing it again.')
    if dest.is_dir() and not _is_dir_empty(dest) and not _pending(journal, dest):
        return 'blocked', ('This folder is not empty. Choose a new or empty folder: LDS '
                           'never installs over existing files.')
    return None


# --- Plan (the dry run) ----------------------------------------------------------

def _q(part) -> str:
    part = str(part)
    return f'"{part}"' if (' ' in part or not part) else part


def _cmd(argv) -> str:
    return ' '.join(_q(a) for a in argv)


def venv_python(dest: Path) -> Path:
    return (dest / VENV_DIR / 'Scripts' / 'python.exe' if os.name == 'nt'
            else dest / VENV_DIR / 'bin' / 'python')


def torch_specs():
    return [f'{name}=={version}' for name, version in AITOOLKIT_RELEASE['torch']]


def torch_constraints(build):
    return [f'{name}=={version}+{build}' for name, version in AITOOLKIT_RELEASE['torch']]


def _index(build) -> str:
    return AITOOLKIT_RELEASE['torch_index'] + build


def _staging(dest: Path) -> Path:
    return dest.parent / f'.{dest.name}.lds-partial'


def archive_url() -> str:
    return f"{AITOOLKIT_RELEASE['repo']}/archive/{AITOOLKIT_RELEASE['commit']}.zip"


def commands(dest: Path, *, source, build, base_python):
    """The exact commands for these choices, in order, for the dry run and the log."""
    repo, commit = AITOOLKIT_RELEASE['repo'] + '.git', AITOOLKIT_RELEASE['commit']
    staging, py = _staging(dest), venv_python(dest)
    build = build or '<cuda build>'
    out = []
    if source == 'git':
        out += [_cmd(['git', 'clone', '--branch', AITOOLKIT_RELEASE['branch'], repo, staging]),
                _cmd(['git', '-C', staging, 'reset', '--hard', commit]),
                f'move {_q(staging)} -> {_q(dest)}']
    else:
        out += [f'download {archive_url()}', f'extract -> {_q(dest)}']
    out.append(_cmd([base_python or '<LDS-managed Python 3.12>', '-m', 'venv', dest / VENV_DIR]))
    out.append(_cmd([py, '-m', 'pip', 'install', '--no-cache-dir', *torch_specs(),
                     '--index-url', _index(build)]))
    requirements = (dest / 'requirements.txt' if source == 'git'
                    else _journal_path().parent / 'requirements-without-git.txt')
    out.append(_cmd([py, '-m', 'pip', 'install', '--no-cache-dir', '-r', requirements,
                     '-c', _journal_path().parent / 'torch-constraints.txt',
                     *_find_links(build)]))
    out.append(_cmd([py, '-c', _IMPORTS]))
    out.append(f'settings: aitoolkit.dir = {_q(dest)} ; aitoolkit.python = {_q(py)}')
    return out


def _find_links(build):
    # Per-package pages, as AI Toolkit's own manager does: they keep the pinned
    # CUDA build resolvable without turning the PyTorch index into a second
    # source for numpy, pillow and the rest.
    out = []
    for name, _version in AITOOLKIT_RELEASE['torch']:
        out += ['--find-links', f'{_index(build)}/{name}/']
    return out


def update_note(source) -> str:
    if source == 'git':
        return ('This is a normal Git checkout of AI Toolkit on its main branch. To update '
                'it later, finish local training, then run `git pull --ff-only` in its '
                'folder followed by `venv\\Scripts\\python -m pip install -r requirements.txt` '
                '(Linux: `venv/bin/python`).')
    return ('Git was not found, so LDS downloads a fixed copy of AI Toolkit '
            f"{AITOOLKIT_RELEASE['version']} that cannot update itself. To update it later, "
            'install Git, then in its folder run `git init`, `git remote add origin '
            f"{AITOOLKIT_RELEASE['repo']}.git`, `git fetch origin main` and "
            '`git checkout -f -B main origin/main`, then reinstall its requirements with '
            '`venv\\Scripts\\python -m pip install -r requirements.txt` (Linux: '
            '`venv/bin/python`). Your venv, outputs and downloaded models stay in place.')


def min_free_gb(build) -> int:
    """Free space a first install needs; the largest figure when the build is unknown."""
    return MIN_FREE_GB.get(build) or max(MIN_FREE_GB.values())


def _resume_done(journal, build) -> list:
    """The completed steps a resume really keeps, for the plan and the install alike,
    so the screen never lists as done a step that Resume runs again. A driver that
    now needs another CUDA build keeps the download alone (PyTorch must follow the
    driver); an environment whose PyTorch install did not finish is rebuilt, because
    a half-installed torch is not trusted."""
    done = list((journal or {}).get('done') or [])
    if (journal or {}).get('cuda_build') not in (None, build):
        done = [s for s in done if s == 'source']
    if 'torch' not in done:
        done = [s for s in done if s != 'python']
    return done


def plan(dest=None, *, facts=None) -> dict:
    """What Install would do here, without side effects -- the dry run.

    state: ready | installed | existing | blocked | unsupported | running.
    The commands are listed whatever the state, so the dry run always shows them."""
    facts = machine_facts() if facts is None else facts
    journal = load_journal()
    # A path pasted from Explorer often keeps its quotes.
    raw = (dest or '').strip().strip('"\'').strip() if isinstance(dest, str) else ''
    target = Path(os.path.expanduser(raw)) if raw else default_destination(facts, journal)
    pending = _pending(journal, target)
    source = (journal or {}).get('source') if pending else ('git' if facts.get('git') else 'archive')
    build, gpu_problem = cuda_build(facts.get('nvidia'))
    python = facts.get('python') or None
    free, drive = _free_gb(target)
    # Resuming after PyTorch is in place only needs room for the requirements:
    # asking for the full amount again would block the very repair it offers.
    kept = _resume_done(journal, build) if pending else []
    need = RESUME_FREE_GB if 'torch' in kept else min_free_gb(build)
    out = {
        'dest': str(target), 'default_dest': str(default_destination(facts, journal)),
        'source': source, 'git': facts.get('git') or '',
        'version': AITOOLKIT_RELEASE['version'], 'commit': AITOOLKIT_RELEASE['commit'],
        'repo': AITOOLKIT_RELEASE['repo'],
        'cuda_build': build, 'torch': dict(AITOOLKIT_RELEASE['torch']),
        'gpus': [g.get('name') for g in ((facts.get('nvidia') or {}).get('gpus') or [])],
        'python': python, 'disk': {'free_gb': None if free is None else round(free, 1),
                                    'need_gb': need, 'drive': drive},
        'resume': {'done': kept, 'state': journal.get('state')} if pending else None,
        'update_note': update_note(source),
        'pinokio': bool(facts.get('pinokio')),
        'commands': commands(target, source=source, build=build,
                             base_python=(python or {}).get('path')),
        'state': 'ready', 'reason': '',
    }

    def verdict(state, reason):
        out.update(state=state, reason=reason)
        return out

    if facts.get('docker'):
        return verdict('unsupported', (
            'LDS is running in Docker. AI Toolkit cannot be installed inside this '
            'container: its GPU environment would be lost on every rebuild. Use Cloud '
            'training, or install AI Toolkit with its own instructions on a machine that '
            'runs LDS natively.'))
    configured = facts.get('configured') or {}
    if configured.get('ok'):
        return verdict('installed', f"AI Toolkit is already installed at {configured.get('detail')}. "
                                    'Nothing to do here.')
    if pending and journal.get('state') == 'running' and _alive(journal.get('owner')):
        return verdict('running', 'AI Toolkit is being installed into this folder.')
    if (facts.get('os') or '') == 'Darwin':
        return verdict('blocked', 'AI Toolkit training through LDS needs an NVIDIA GPU, '
                                  'which macOS does not support. Use Cloud training.')
    if gpu_problem:
        return verdict('blocked', gpu_problem)
    if (facts.get('os') or '') == 'Windows' and facts.get('arch') in ('arm64', 'aarch64'):
        return verdict('blocked', (
            "Windows on ARM needs AI Toolkit's own installer, which sets up the x64 stack "
            'its PyTorch build requires: run `python -m manager install` in an AI Toolkit '
            'checkout, then select its folder.'))
    if facts.get('training'):
        return verdict('blocked', 'Local training is running. Install AI Toolkit after it finishes.')
    problem = _destination_problem(target, journal)
    if problem:
        return verdict(*problem)
    if pending and journal.get('commit') != AITOOLKIT_RELEASE['commit']:
        return verdict('blocked', (
            'An unfinished install from another LDS version is in this folder. Choose '
            'another folder, or delete this one first.'))
    if free is not None and free < need:
        return verdict('blocked', (
            f'AI Toolkit needs about {need} GB free on {drive} ({free:.1f} GB free now). '
            'Free some space or choose a folder on another drive.'))
    if python is None:
        return verdict('blocked', ('No Python 3.10-3.12 was found and LDS cannot download one '
                                   'for this system. Install Python 3.12, then check again.'))
    return out


# --- Requirements without git ---------------------------------------------------

_GITHUB_GIT = re.compile(
    r'^(?:(?P<name>[A-Za-z0-9_.\-]+)\s*@\s*)?git\+https://github\.com/'
    r'(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+?)(?:\.git)?@(?P<ref>[\w.\-/]+?)'
    r'(?:#egg=(?P<egg>[\w.\-]+))?$')


def requirements_without_git(path: Path, _seen=None) -> list:
    """requirements.txt flattened, each GitHub `git+` pin turned into the archive
    of the SAME ref, so pip installs the identical source without git."""
    seen = set() if _seen is None else _seen
    path = Path(path)
    key = os.path.normcase(str(path.resolve()))
    if key in seen:
        return []
    seen.add(key)
    out = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line.startswith(('-r ', '--requirement ')):
            out += requirements_without_git(path.parent / line.split(None, 1)[1].strip(), seen)
            continue
        if line.startswith('#') or 'git+' not in line:
            out.append(raw)
            continue
        match = _GITHUB_GIT.match(re.split(r'\s+#', line, maxsplit=1)[0].strip())
        if not match:
            raise InstallError(f'This AI Toolkit requirement needs Git: {line}. Install Git, '
                               'then click Install again.')
        name = match['name'] or match['egg'] or match['repo']
        out.append(f"{name} @ https://github.com/{match['owner']}/{match['repo']}/archive/"
                   f"{match['ref']}.zip")
    return out


# --- Running commands ------------------------------------------------------------

def subprocess_env(tmp=None) -> dict:
    from . import managed_python
    env = {k: v for k, v in managed_python.subprocess_env().items()
           if not k.upper().startswith(('GIT_', 'CONDA_'))}
    env.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never', GIT_LFS_SKIP_SMUDGE='1',
               PIP_NO_INPUT='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONUNBUFFERED='1',
               PYTHONIOENCODING='utf-8')
    if tmp:
        env.update(TMP=str(tmp), TEMP=str(tmp), TMPDIR=str(tmp))
    return env


class ProcessRunner:
    """One child at a time, its output streamed to the log, killable by cancel()."""

    def __init__(self, log, cancel_event, on_child=None):
        self.log, self.cancel_event, self.on_child = log, cancel_event, on_child
        self.process = None
        self._lock = threading.Lock()

    def run(self, argv, *, cwd=None, env=None) -> int:
        if self.cancel_event.is_set():
            raise Cancelled()
        try:
            proc = subprocess.Popen([str(a) for a in argv], cwd=cwd, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                    errors='replace', creationflags=_NO_WINDOW)
        except OSError as exc:
            raise InstallError(f'Could not start {Path(str(argv[0])).name}: {exc}') from exc
        with self._lock:
            self.process = proc
        if self.on_child:
            self.on_child(_identity(proc.pid))
        try:
            for line in proc.stdout:
                self.log(line.rstrip())
            rc = proc.wait()
        finally:
            proc.stdout.close()
            with self._lock:
                self.process = None
            if self.on_child:
                self.on_child(None)
        if self.cancel_event.is_set():
            raise Cancelled()
        return rc

    def kill(self):
        with self._lock:
            proc = self.process
        if proc is not None and proc.poll() is None:
            _kill_tree(proc.pid)


def _http_fetch(url, target: Path, *, on_bytes, cancel_event):
    import requests
    from ..timeout_settings import network_timeout
    done = 0
    with requests.get(url, stream=True, timeout=network_timeout((15, 60)),
                      headers={'User-Agent': 'lora-dataset-studio-setup'}) as response:
        response.raise_for_status()
        with target.open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                if cancel_event.is_set():
                    raise Cancelled()
                done += len(chunk)
                if done > _ARCHIVE_MAX:
                    raise InstallError('The AI Toolkit download is larger than expected; nothing was installed.')
                output.write(chunk)
                on_bytes(done)
    return done


def _extract_archive(archive: Path, staging: Path, commit: str) -> Path:
    """Unpack GitHub's archive: one top folder, regular files only, contained."""
    top = f'ai-toolkit-{commit}'
    staging.mkdir(parents=True, exist_ok=True)
    root = staging.resolve()
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > 60000 or sum(i.file_size for i in infos) > _EXTRACT_MAX:
            raise InstallError('The AI Toolkit archive exceeds its extraction budget.')
        for info in infos:
            parts = Path(info.filename.replace('\\', '/')).parts
            if (not parts or parts[0] != top or '..' in parts or ':' in info.filename
                    or info.filename.startswith(('/', '\\'))):
                raise InstallError('The AI Toolkit archive has an unexpected layout; nothing was installed.')
            target = staging.joinpath(*parts)
            if not target.resolve().is_relative_to(root):
                raise InstallError('The AI Toolkit archive escapes its folder; nothing was installed.')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    return staging / top


def _rename(source: Path, target: Path, attempts=6):
    # A freshly written tree can be held for a moment by an antivirus scan.
    for attempt in range(attempts):
        try:
            if target.is_dir() and _is_dir_empty(target):
                target.rmdir()
            source.rename(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)


def _rmtree(path: Path):
    # Git keeps its objects read-only; Windows refuses to delete those as is.
    def retry(func, name, _exc):
        with contextlib.suppress(OSError):
            os.chmod(name, 0o700)
            func(name)
    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


def _tree_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total


@contextlib.contextmanager
def _download_meter(tmp: Path, progress):
    """pip prints no progress through a pipe; the size of its temporary download
    (TMP points at `tmp`) is the honest byte count while PyTorch arrives."""
    stop = threading.Event()

    def sample():
        while not stop.wait(1.5):
            progress(_tree_size(tmp), 0)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
        progress(None, None)


# --- The install -------------------------------------------------------------------

def _noop(*_args, **_kwargs):
    return None


def _labels(source, build):
    torch = dict(AITOOLKIT_RELEASE['torch'])['torch']
    version, short = AITOOLKIT_RELEASE['version'], AITOOLKIT_RELEASE['commit'][:7]
    return {
        'source': (f'Cloning AI Toolkit {version} ({short}) with Git' if source == 'git'
                   else f'Downloading AI Toolkit {version} ({short})'),
        'python': 'Creating its Python environment',
        'torch': f'Installing PyTorch {torch} for CUDA ({build}), about 2-3 GB to download',
        'deps': "Installing AI Toolkit's requirements",
        'check': 'Checking that PyTorch sees your GPU',
        'config': 'Saving the AI Toolkit folder and Python in Settings',
    }


def install(dest, *, log, step=None, progress=None, cancel_event=None, find_python=None,
            runner=None, fetch=None, probe_torch=None, facts=None, register=None,
            fallback_python=None):
    """Run (or resume) the install into `dest`. Returns 0; raises InstallError with
    the user-facing reason, or Cancelled. Collaborators are injectable for tests;
    `register(runner)` hands the caller the runner its cancel button stops, and
    `fallback_python()` supplies LDS's own Python when the machine's cannot make
    a venv (a Linux Python without its venv package)."""
    step, progress = step or _noop, progress or _noop
    cancel_event = cancel_event or threading.Event()
    checked = plan(dest, facts=facts)
    if checked['state'] != 'ready':
        raise InstallError(checked['reason'] or 'AI Toolkit cannot be installed here.')
    dest = Path(checked['dest'])
    source, build = checked['source'], checked['cuda_build']
    journal = load_journal()
    if _pending(journal, dest):
        if _alive(journal.get('child')):
            log(f"Stopping the unfinished step left by a previous LDS session (process {journal['child']['pid']}).")
            _kill_tree(journal['child']['pid'])
        if journal.get('cuda_build') not in (None, build):
            log(f"The NVIDIA driver now needs {build} instead of {journal['cuda_build']}: "
                'rebuilding the Python environment.')
        done = _resume_done(journal, build)
        journal.update(done=done)
        log(f"Resuming the install in {dest}. Completed: {', '.join(done) or 'nothing yet'}.")
    else:
        if journal and journal.get('state') != 'done' and journal.get('dest'):
            log(f"An earlier unfinished install remains in {journal['dest']}; LDS leaves it in place.")
        journal = {'dest': str(dest), 'source': source, 'commit': AITOOLKIT_RELEASE['commit'],
                   'version': AITOOLKIT_RELEASE['version'], 'cuda_build': build, 'done': []}
    journal.update(state='running', owner=_identity(os.getpid()), cuda_build=build, error='')
    _write_journal(journal)

    def on_child(identity):
        journal['child'] = identity
        _write_journal(journal)

    runner = runner or ProcessRunner(log, cancel_event, on_child)
    if register:
        register(runner)
    labels = _labels(source, build)
    log(f"AI Toolkit {AITOOLKIT_RELEASE['version']} -> {dest}")
    log(f"PyTorch {dict(AITOOLKIT_RELEASE['torch'])['torch']} {build} from {_index(build)} "
        f"(AI Toolkit's own instructions at {AITOOLKIT_RELEASE['commit'][:7]})")
    git = checked['git'] or 'git'
    work = {
        'source': lambda: _step_source(dest, source, git, runner, log, progress, cancel_event, fetch),
        'python': lambda: _step_python(dest, runner, log, find_python, fallback_python),
        'torch': lambda: _step_torch(dest, build, runner, progress),
        'deps': lambda: _step_deps(dest, source, build, runner, progress),
        'check': lambda: _step_check(dest, runner, log, probe_torch),
        'config': lambda: _step_config(dest, log),
    }
    try:
        for index, name in enumerate(STEPS, 1):
            if name in journal['done']:
                continue
            if cancel_event.is_set():
                raise Cancelled()
            step(index, len(STEPS), labels[name])
            log(f'{index}/{len(STEPS)} · {labels[name]}…')
            work[name]()
            journal['done'].append(name)
            _write_journal(journal)
        journal.update(state='done', child=None)
        _write_journal(journal)
    except Cancelled:
        journal.update(state='cancelled', child=None)
        _write_journal(journal)
        log('Cancelled. Click Install again to continue from the last completed step.')
        raise
    except Exception as exc:
        journal.update(state='error', child=None, error=str(exc)[-500:])
        _write_journal(journal)
        raise
    finally:
        with contextlib.suppress(OSError):
            _rmtree(dest / TMP_DIR)
    log('AI Toolkit is installed and selected in Settings > Local tools.')
    log(update_note(source))
    return 0


def _step_source(dest, source, git, runner, log, progress, cancel_event, fetch):
    staging = _staging(dest)
    if staging.exists():
        log('Removing the unfinished download from the previous attempt.')
        _rmtree(staging)
    dest.parent.mkdir(parents=True, exist_ok=True)
    commit = AITOOLKIT_RELEASE['commit']
    env = subprocess_env()
    if source == 'git':
        rc = runner.run([git, 'clone', '--branch', AITOOLKIT_RELEASE['branch'],
                         AITOOLKIT_RELEASE['repo'] + '.git', staging], env=env)
        if rc:
            raise InstallError(f'Git could not download AI Toolkit (exit {rc}). Check the '
                               'connection, then click Install again.')
        if runner.run([git, '-C', staging, 'reset', '--hard', commit], env=env):
            runner.run([git, '-C', staging, 'fetch', 'origin', commit], env=env)
            if runner.run([git, '-C', staging, 'reset', '--hard', commit], env=env):
                raise InstallError('The AI Toolkit revision LDS installs is no longer available '
                                   'upstream. Update LDS, then click Install again.')
        tree = staging
    else:
        archive = dest.parent / f'.{dest.name}.lds-download.zip'
        try:
            (fetch or _http_fetch)(archive_url(), archive,
                                   on_bytes=lambda n: progress(n, 0), cancel_event=cancel_event)
        except (Cancelled, InstallError):
            raise
        except Exception as exc:
            raise InstallError(f'AI Toolkit could not be downloaded ({exc}). Check the '
                               'connection, then click Install again.') from exc
        finally:
            progress(None, None)
        try:
            tree = _extract_archive(archive, staging, commit)
        except zipfile.BadZipFile as exc:
            raise InstallError('The AI Toolkit download is damaged. Click Install again.') from exc
        finally:
            with contextlib.suppress(OSError):
                archive.unlink()
    if not (tree / 'run.py').is_file() or not (tree / 'requirements.txt').is_file():
        raise InstallError('The downloaded AI Toolkit has no run.py or requirements.txt.')
    (tree / STAMP).write_text(json.dumps({'commit': commit, 'source': source,
                                          'installed_by': 'LoRA Dataset Studio'}), encoding='utf-8')
    exclude = tree / '.git' / 'info' / 'exclude'
    if exclude.parent.is_dir():
        with exclude.open('a', encoding='utf-8') as fh:
            fh.write(f'\n/{STAMP}\n/{TMP_DIR}/\n/{VENV_DIR}/\n')
    _rename(tree, dest)
    if staging.exists():
        _rmtree(staging)


def _step_python(dest, runner, log, find_python, fallback_python=None):
    if not (dest / STAMP).is_file():
        raise InstallError('This folder was not created by the LDS installer. Choose a new folder.')
    venv = dest / VENV_DIR
    if venv.exists():
        log('Rebuilding the unfinished Python environment from the previous attempt.')
        _rmtree(venv)
    base = find_python() if find_python else ''
    if not base:
        raise InstallError('No Python 3.10-3.12 is available to create the AI Toolkit '
                           'environment. Check the connection and the free space, then '
                           'click Install again.')
    if runner.run([base, '-m', 'venv', venv], env=subprocess_env()):
        managed = ''
        if fallback_python:
            log(f'{base} could not create a venv; using the LDS-managed Python instead.')
            _rmtree(venv)
            with contextlib.suppress(Exception):
                managed = fallback_python() or ''
        if (not managed or os.path.normcase(managed) == os.path.normcase(str(base))
                or runner.run([managed, '-m', 'venv', venv], env=subprocess_env())):
            raise InstallError(f'Python could not create the AI Toolkit environment with {base}. '
                               'On Linux, install the python3-venv package, then click Install again.')
    python = venv_python(dest)
    if not python.is_file():
        raise InstallError('The new AI Toolkit environment has no Python. Click Install again.')
    runner.run([python, '-m', 'pip', 'install', '--upgrade', 'pip'], env=subprocess_env())


def _pip_env(dest):
    tmp = dest / TMP_DIR
    _rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp, subprocess_env(tmp)


def _step_torch(dest, build, runner, progress):
    tmp, env = _pip_env(dest)
    with _download_meter(tmp, progress):
        rc = runner.run([venv_python(dest), '-m', 'pip', 'install', '--no-cache-dir', *torch_specs(),
                         '--index-url', _index(build)], cwd=dest, env=env)
    if rc:
        raise InstallError(f'PyTorch could not be installed (exit {rc}). Check the connection and '
                           'the free space, then click Install again to continue.')


def _step_deps(dest, source, build, runner, progress):
    folder = _journal_path().parent
    folder.mkdir(parents=True, exist_ok=True)
    constraints = folder / 'torch-constraints.txt'
    constraints.write_text('\n'.join(torch_constraints(build)) + '\n', encoding='utf-8')
    requirements = dest / 'requirements.txt'
    if source != 'git':
        flattened = folder / 'requirements-without-git.txt'
        flattened.write_text('\n'.join(requirements_without_git(requirements)) + '\n', encoding='utf-8')
        requirements = flattened
    tmp, env = _pip_env(dest)
    with _download_meter(tmp, progress):
        rc = runner.run([venv_python(dest), '-m', 'pip', 'install', '--no-cache-dir',
                         '-r', requirements, '-c', constraints, *_find_links(build)],
                        cwd=dest, env=env)
    if rc:
        raise InstallError(f"AI Toolkit's requirements could not be installed (exit {rc}). Resolve "
                           'the problem reported above, then click Install again to continue.')


def _step_check(dest, runner, log, probe_torch):
    python = venv_python(dest)
    if runner.run([python, '-c', _IMPORTS], cwd=dest, env=subprocess_env()):
        raise InstallError('AI Toolkit\'s packages do not import in the new environment (see the '
                           'lines above). Click Install again to repair it.')
    from .. import capabilities
    from .training_diagnostics import torch_cuda_verdict
    probe = probe_torch or (lambda py, cwd: capabilities._torch_probe(str(py), timeout=300, cwd=str(cwd)))
    info = probe(python, dest)
    verdict = torch_cuda_verdict(info, venv_python=python, aitoolkit_dir=dest)
    if verdict and not verdict.get('available'):
        raise InstallError(verdict['message'])
    if info and info.get('device_name'):
        log(f"PyTorch sees {info['device_name']}.")


def _step_config(dest, log):
    from .. import capabilities
    python = venv_python(dest)
    current = capabilities.probe_aitoolkit()
    if current.get('ok') and os.path.normcase(str(cfg.aitoolkit_path('dir'))) != os.path.normcase(str(dest)):
        log(f"Settings already point at a working AI Toolkit ({current.get('detail')}); LDS kept "
            f'that choice. The new copy is ready at {dest}.')
        return
    cfg.save_config({'aitoolkit': {'dir': str(dest), 'python': str(python)}})
    capabilities.clear_import_cache()
    check = capabilities.probe_aitoolkit()
    if not check.get('ok'):
        raise InstallError(f"Settings were saved, but AI Toolkit is not ready: {check.get('detail')}")
    log(f'Settings > Local tools now use {dest}.')
