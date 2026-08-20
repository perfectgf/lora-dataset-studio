"""Find a Python already on this machine that can run a heavy pass on the GPU.

The scoring extra deliberately installs CPU-only torch (Setup builds it a small
private venv rather than pushing a ~2.5 GB CUDA download on people who have no
card). On a machine that *does* have a card that default costs hours: CLIP
ViT-L/14 measures ~336 ms/image on the CPU against ~15 ms on a recent GPU.

The obvious fix — a button that pip-installs a CUDA torch — is the one we
deliberately do NOT build: it means a 2.5 GB download plus picking a wheel index
against the driver, and getting that wrong is exactly how a `--index-url`
install shredded someone's numpy. The better move is the opposite one: a machine
that trains LoRAs every day ALREADY has a proven CUDA Python (ai-toolkit's venv,
ComfyUI's, a conda env). Point the scoring pass at it instead of building a
third one.

What makes this honest rather than hopeful:

* **Every dependency, not just CUDA.** ``bank_score_infer.py`` needs torch AND
  open_clip AND transformers/timm AND numpy/Pillow. An interpreter with a
  perfect CUDA torch but no ``open_clip`` will die mid-pass. So the probe reports
  a state PER DEPENDENCY and the UI can say "ai-toolkit has CUDA but is missing
  open_clip" instead of an opaque no.
* **Read-only.** We never install anything into an environment we did not build.
  The ai-toolkit venv runs the user's training; silently pip-installing into it
  would be unacceptable. When something is missing we name it and hand over the
  exact command — the user decides.
* **Known candidates only.** Interpreters the app already knows about, plus a
  path the user types. Sweeping the disk would be slow and fragile.
* **Fail safe.** Nothing proven -> nothing changes, and the pass keeps running
  where it runs today. ``select()`` refuses any interpreter it could not verify.

The probe really imports every dependency, in an isolated interpreter that
ignores the process owner's user-site packages (``python -s``).  Merely finding
a module spec is not enough: a package can be present yet fail while importing
one of its native or transitive dependencies.  That exact false positive used
to mark a borrowed ComfyUI Python as GPU-ready while Score itself stayed off.

**Two features, one detector.** ✨ Score was the first caller; the SigLIP2
semantic index is the second, and it needs a DIFFERENT dependency list (no
open_clip, no timm) written to a DIFFERENT config key. Everything else — the
candidates, the read-only rule, the fail-safe selection — is identical, so the
module is parametrised by an :class:`InterpreterProfile` rather than copied.
Telling someone "ComfyUI's Python cannot run the semantic index" because it
lacks ``open_clip`` would be a lie: that worker never imports it.

**Borrowing is about WHERE A PASS RUNS, never about where Setup installs.**
Selecting an interpreter here writes one config key and nothing else. Setup's
install actions resolve their target from the app-managed environment on their
own and never read these keys — see ``setup_installer._bank_semantic_install_python``
and ``test_setup_installer.py``. That separation is the whole reason it is safe
to point the semantic engine at someone's ai-toolkit venv.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from .. import config as cfg
from . import infer_env

# Everything backend/infer/bank_score_infer.py imports to complete a pass, in
# report order. `pip` is the name to install, `module` is what actually gets
# imported — they differ for three of the six, which is precisely why a
# copy-pasteable command has to be generated rather than guessed.
SCORING_DEPS = (
    {'module': 'torch', 'pip': 'torch', 'label': 'PyTorch'},
    {'module': 'open_clip', 'pip': 'open_clip_torch', 'label': 'OpenCLIP'},
    {'module': 'transformers', 'pip': 'transformers', 'label': 'Transformers'},
    {'module': 'timm', 'pip': 'timm', 'label': 'timm'},
    {'module': 'numpy', 'pip': 'numpy', 'label': 'NumPy'},
    {'module': 'PIL', 'pip': 'Pillow', 'label': 'Pillow'},
)

# Everything backend/infer/bank_semantic_infer.py imports. A strict SUBSET of the
# scoring list, plus one thing a module name cannot answer: ``Siglip2Model``
# only exists from transformers 4.49. A 2024 ComfyUI venv imports `transformers`
# perfectly and then dies on the model class — exactly the false positive this
# probe exists to refuse, which is why a dep may also demand a SYMBOL.
# `spec` is what the copy-pasteable repair line must say when a bare package
# name would be a no-op on an environment that already holds an older copy.
SEMANTIC_DEPS = (
    {'module': 'torch', 'pip': 'torch', 'label': 'PyTorch'},
    {'module': 'transformers', 'pip': 'transformers', 'spec': 'transformers>=4.49',
     'attrs': ('Siglip2Model',), 'label': 'Transformers (SigLIP2-capable)'},
    {'module': 'numpy', 'pip': 'numpy', 'label': 'NumPy'},
    {'module': 'PIL', 'pip': 'Pillow', 'label': 'Pillow'},
)

# Everything backend/infer/watermark_detect_infer.py imports: torch, transformers
# for BOTH halves of the cascade — the SigLIP2 ranker (SiglipForImageClassification)
# and the Grounding-DINO locator (AutoModelForZeroShotObjectDetection) — and
# Pillow via the shared bank_image_guard. Both model classes are demanded as
# SYMBOLS for the same reason the semantic list demands Siglip2Model: a 2024
# transformers imports fine and then dies on the class, which is precisely the
# false positive the probe exists to refuse.
WATERMARK_DEPS = (
    {'module': 'torch', 'pip': 'torch', 'label': 'PyTorch'},
    {'module': 'transformers', 'pip': 'transformers', 'spec': 'transformers>=4.40',
     'attrs': ('SiglipForImageClassification', 'AutoModelForZeroShotObjectDetection'),
     'label': 'Transformers (SigLIP + Grounding-DINO capable)'},
    {'module': 'PIL', 'pip': 'Pillow', 'label': 'Pillow'},
)


class InterpreterProfile(NamedTuple):
    """One feature's answer to "what must this Python be able to do, and where
    is the choice stored". `feature` is user-facing copy: it lands verbatim in
    the refusal a user reads, so it names the pass, not the module."""
    key: str
    config_key: str
    feature: str
    deps: tuple


PROFILES = {
    'scoring': InterpreterProfile(
        'scoring', 'bank_scoring.python', '✨ Score', SCORING_DEPS),
    'semantic': InterpreterProfile(
        'semantic', 'bank_semantic.python', 'the SigLIP 2 semantic index',
        SEMANTIC_DEPS),
    'watermark_detect': InterpreterProfile(
        'watermark_detect', 'watermark_detect.python',
        'the 🚩 watermark detector', WATERMARK_DEPS),
}
DEFAULT_PROFILE = PROFILES['scoring']


def get_profile(profile=None) -> InterpreterProfile:
    """Accept a profile, its key, or nothing (Score — the original caller, kept
    as the default so every existing call site keeps its exact meaning)."""
    if isinstance(profile, InterpreterProfile):
        return profile
    key = str(profile or DEFAULT_PROFILE.key)
    try:
        return PROFILES[key]
    except KeyError:
        raise ValueError(f'unknown interpreter profile: {key!r}') from None


# The probe answers for the UNION of every profile: one subprocess, one cache,
# both features served. Order follows SCORING_DEPS so the payload a Score client
# already parses is byte-identical.
_ALL_DEPS = SCORING_DEPS + SEMANTIC_DEPS + WATERMARK_DEPS
_DEP_MODULES = tuple(dict.fromkeys(d['module'] for d in _ALL_DEPS))
_DEP_ATTRS = tuple(sorted({(d['module'], attr) for d in _ALL_DEPS
                           for attr in d.get('attrs', ())}))

# A cold `import torch` on a fresh machine (antivirus scanning ~300 MB of native
# DLLs) runs tens of seconds. Generous, because a timeout here reads as "this
# interpreter is unusable" and that would be a lie about a working venv.
PROBE_TIMEOUT = 90
_PROBE_TTL = 600
_probe_cache = {}     # normalised path -> (ts, info|None)

_PROBE_CODE = (
    'import importlib as _i, json, sys\n'
    'mods = ' + repr(list(_DEP_MODULES)) + '\n'
    'found = {}\n'
    'loaded = {}\n'
    'for m in mods:\n'
    '    try:\n'
    '        loaded[m] = _i.import_module(m)\n'
    '        found[m] = True\n'
    '    except Exception:\n'
    '        found[m] = False\n'
    'cuda, device, torch_version = False, None, None\n'
    'if found.get("torch"):\n'
    '    try:\n'
    '        torch = loaded["torch"]\n'
    '        torch_version = torch.__version__\n'
    '        cuda = bool(torch.cuda.is_available())\n'
    '        if cuda:\n'
    '            device = torch.cuda.get_device_name(0)\n'
    '    except Exception:\n'
    '        found["torch"] = False\n'
    # A module that imports is not proof the SYMBOL a worker needs is in it.
    'attrs = ' + repr([list(a) for a in _DEP_ATTRS]) + '\n'
    'symbols = {}\n'
    'for m, a in attrs:\n'
    '    try:\n'
    '        symbols[m + ":" + a] = getattr(loaded[m], a) is not None\n'
    '    except Exception:\n'
    '        symbols[m + ":" + a] = False\n'
    'print(json.dumps({"python": "%d.%d.%d" % sys.version_info[:3],\n'
    '                  "modules": found, "symbols": symbols, "cuda": cuda,\n'
    '                  "device_name": device,\n'
    '                  "torch_version": torch_version}))\n'
)


def _norm(path) -> str:
    return os.path.normcase(os.path.abspath(str(path or '')))


def clear_cache() -> None:
    """Forget every probe result. Called by the rescan action so a user who just
    ran the suggested pip command sees the truth instead of a 10-minute-old
    'missing open_clip'."""
    _probe_cache.clear()


def _run_probe(python: str):
    """Raw probe facts for `python`, or None when we could not learn anything
    (interpreter missing/broken, cold-import timeout, garbage on stdout). None is
    UNKNOWN — never a claim that the interpreter is unusable."""
    try:
        proc = subprocess.run(
            infer_env.worker_argv(python, '-c', _PROBE_CODE),
            capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=PROBE_TIMEOUT,
            env=infer_env.worker_env(python),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception:      # noqa: BLE001 — OSError, TimeoutExpired, anything
        return None
    if proc.returncode != 0:
        return None
    try:
        info = json.loads(((proc.stdout or '').strip().splitlines() or [''])[-1])
    except Exception:      # noqa: BLE001
        return None
    return info if isinstance(info, dict) else None


def probe(python: str, force=False):
    """Cached _run_probe. A None result is NOT cached: a cold-import timeout must
    not freeze a working venv into 'unreachable' for ten minutes."""
    key = _norm(python)
    if not key:
        return None
    now = time.time()
    hit = _probe_cache.get(key)
    if hit and not force and (now - hit[0]) < _PROBE_TTL:
        return hit[1]
    info = _run_probe(python)
    if info is not None:
        _probe_cache[key] = (now, info)
    return info


def _quote(p: str) -> str:
    return f'"{p}"' if ' ' in str(p) else str(p)


# Where an interpreter sits INSIDE an environment folder. venv and conda do not
# agree, and neither does a portable bundle, so all the shapes are tried and the
# one that ANSWERS wins — we never decide from the folder's look. Ordered
# cheapest-first; on a POSIX box the .exe entries simply never match.
_INTERPRETER_SPOTS = (
    ('Scripts', 'python.exe'),      # Windows venv / virtualenv
    ('bin', 'python'),              # POSIX venv, and conda envs on Linux/macOS
    ('bin', 'python3'),
    ('python.exe',),                # conda env root, portable python_embeded
    ('python',),
)


def _clean_path(raw: str) -> str:
    """A path as a HUMAN gives it: pasted from a file manager (wrapped in
    quotes), copied out of a terminal (trailing space), or dragged in. Only
    transport noise is removed — the path itself, spaces, accents and all, is
    left exactly as typed. Never resolves, never guesses."""
    p = (raw or '').strip().strip('\r\n')
    if len(p) >= 2 and p[0] == p[-1] and p[0] in '"\'':
        p = p[1:-1].strip()
    return p


def interpreters_in(folder) -> list:
    """Every path inside `folder` that COULD be an interpreter, in probe order.

    Deliberately not a verdict: it only says where to knock. A venv, a conda
    env, a portable bundle and a bare folder holding a python.exe all get their
    shapes tried; whether any of them answers is the probe's business, not a
    guess from the folder's name or contents. Empty list on anything unreadable
    — a folder we cannot list is not an error, just no candidates."""
    out = []
    try:
        root = Path(folder)
        if not root.is_dir():
            return []
    except OSError:
        return []
    for spot in _INTERPRETER_SPOTS:
        cand = root.joinpath(*spot)
        try:
            if cand.is_file():
                out.append(str(cand))
        except OSError:
            continue
    return out


def resolve_entered_path(raw: str) -> list:
    """What to probe for a path a user typed, best guess first.

    Accepts BOTH an interpreter and an environment folder, because both are what
    people have at hand: "C:\\miniconda3\\envs\\ml" and ".../envs/ml/python.exe"
    must lead to the same place. A file is taken at its word (any name — `python`,
    `python3.11`, `python.exe`, a shim); a folder is expanded to the shapes above.
    Returns [] for a path that exists as neither, so the caller can say so
    precisely instead of probing nothing and calling it a failure."""
    p = _clean_path(raw)
    if not p:
        return []
    try:
        path = Path(p)
        if path.is_file():
            return [str(path)]
        if path.is_dir():
            return interpreters_in(path)
    except OSError:
        return []
    # Doesn't exist (yet): hand it back anyway so the probe produces the honest
    # "did not answer" verdict against the exact string the user typed.
    return [p]


def _dep_present(dep, mods, symbols) -> bool:
    """A dependency counts as present only when its module imported AND every
    symbol the worker needs is really in it. A probe payload with no `symbols`
    map (an older cached answer, a hand-built fixture) cannot prove a symbol, so
    a dep that demands one is reported MISSING — fail safe, like everything else
    here: the cost of a wrong 'ready' is an import error an hour into a pass."""
    if not mods.get(dep['module']):
        return False
    return all(bool((symbols or {}).get(f"{dep['module']}:{attr}"))
               for attr in dep.get('attrs', ()))


def _install_spec(dep) -> str:
    """What to hand pip. A bare package name is a no-op on an environment that
    already carries an older copy, so a dep with a version floor names it."""
    spec = dep.get('spec') or dep['pip']
    return f'"{spec}"' if spec != dep['pip'] else spec


def describe(python: str, info, profile=None) -> dict:
    """Turn raw probe facts into the verdict the UI renders, for one profile.

    status ∈
      'unreachable' — the interpreter did not answer (missing, broken, timeout).
      'incomplete'  — it answered, but the pass would crash on an import.
      'cpu_only'    — every dependency is there, torch just has no usable CUDA.
      'gpu_ready'   — every dependency is there AND torch sees the GPU.

    `missing` names the modules that aren't there and `install_command` is the
    exact line to fix it — we never run it ourselves.
    """
    prof = get_profile(profile)
    out = {
        'path': str(python), 'status': 'unreachable', 'cuda': False,
        'device_name': None, 'python_version': None, 'torch_version': None,
        'profile': prof.key,
        'deps': [dict(d, present=False) for d in prof.deps],
        'missing': [d['pip'] for d in prof.deps],
        'install_command': '', 'usable': False, 'gpu': False,
        'detail': 'this interpreter did not answer — check the path',
    }
    if not info:
        return out
    mods = info.get('modules') or {}
    symbols = info.get('symbols') or {}
    deps = [dict(d, present=_dep_present(d, mods, symbols)) for d in prof.deps]
    missing = [d for d in deps if not d['present']]
    out.update({
        'deps': deps,
        'missing': [d['pip'] for d in missing],
        'cuda': bool(info.get('cuda')),
        'device_name': info.get('device_name') or None,
        'python_version': info.get('python') or None,
        'torch_version': info.get('torch_version') or None,
    })
    if missing:
        names = ', '.join(d['label'] for d in missing)
        out['status'] = 'incomplete'
        out['install_command'] = (f'{_quote(python)} -m pip install '
                                  + ' '.join(_install_spec(d) for d in missing))
        out['detail'] = (
            f'has CUDA but is missing {names}' if out['cuda']
            else f'missing {names}')
        return out
    out['usable'] = True
    if out['cuda']:
        out['status'] = 'gpu_ready'
        out['gpu'] = True
        card = out['device_name'] or 'a CUDA GPU'
        out['detail'] = f'ready — runs on {card}'
    else:
        out['status'] = 'cpu_only'
        out['detail'] = 'ready, but torch here has no usable CUDA — runs on the CPU'
    return out


def _comfyui_pythons() -> list:
    """Interpreter paths a ComfyUI install may use. ComfyUI is installed half a
    dozen ways in the wild — a venv inside the folder, the portable bundle's
    python_embeded (which sits NEXT TO the ComfyUI folder, not inside), a conda
    env, the system Python. We look in the places tied to the folder we were
    given and let the probe decide; a shape we don't know about is not excluded,
    it just isn't guessed — the user can still enter it by hand."""
    base = (cfg.get('comfyui.base_dir') or '').strip()
    if not base:
        return []
    root = Path(base)
    out = []
    for env_dir in (root / 'venv', root / '.venv',
                    root / 'python_embeded', root.parent / 'python_embeded'):
        out.extend(interpreters_in(env_dir))
    return out


def candidates(profile=None) -> list:
    """Interpreters worth probing, best-known first: [{path, source, label}].

    Deliberately NOT a disk sweep — only Pythons the app already knows about
    (plus whatever the user types, handled by the caller). Deduplicated on the
    normalised path so the currently-selected one doesn't appear twice, and
    filtered to files that exist: an entry we cannot even find is noise, not
    information. The app's own interpreter is always last — it is what runs the
    pass today, so it belongs in the list as the way back."""
    from .. import setup_installer
    prof = get_profile(profile)
    seen, out = set(), []

    def add(path, source, label):
        if not path:
            return
        key = _norm(path)
        if not key or key in seen:
            return
        try:
            if not Path(path).is_file():
                return
        except OSError:
            return
        seen.add(key)
        out.append({'path': str(path), 'source': source, 'label': label})

    # The DESCRIPTIVE sources go in first on purpose. The interpreter in use is
    # usually one of them, and "ai-toolkit — the environment that trains your
    # LoRAs" tells the user far more than "currently used": which one is selected
    # is already carried by `selected` (and an "In use" badge). The configured
    # path is added afterwards only when it matches nothing we recognise.
    try:
        add(setup_installer._bank_scoring_env_python(), 'managed',
            "The app's own scoring environment")
    except Exception:      # noqa: BLE001 — a data-dir hiccup must not empty the list
        pass
    try:
        add(cfg.aitoolkit_path('venv_python'), 'aitoolkit',
            'ai-toolkit — the environment that trains your LoRAs')
    except Exception:      # noqa: BLE001
        pass
    for p in _comfyui_pythons():
        add(p, 'comfyui', 'ComfyUI')
    add((cfg.get('masks.python') or '').strip(), 'masks',
        "The app's masking environment")
    add((cfg.get('watermark.python') or '').strip(), 'watermark',
        "The app's inpainting environment")
    add(sys.executable, 'app', "The app's own Python")
    # For any feature OTHER than Score, the interpreter Score already borrows is
    # the single most likely right answer — it is the one the user already proved
    # on this machine. It gets a row of its own so "use the same Python as ✨
    # Score" is one click and not a path to copy by hand. Deduplication makes it
    # a no-op when Score runs on something already listed above.
    if prof.key != 'scoring':
        add((cfg.get('bank_scoring.python') or '').strip(), 'scoring',
            'The Python ✨ Score already uses')
    add((cfg.get(prof.config_key) or '').strip(), 'configured',
        f'Currently used for {prof.feature}')
    return out


def default_python(profile=None) -> str:
    """What the pass runs in when nothing is selected — the exact fallback its
    own resolver applies, not a guess. Kept next to the resolvers it mirrors so
    a divergence is one grep away (``bank_semantic_models.semantic_python``)."""
    prof = get_profile(profile)
    if prof.key in ('semantic', 'watermark_detect'):
        # Both resolvers fall back to Score's interpreter before the app's own
        # (``bank_semantic_models.semantic_python``,
        #  ``watermark_detector.detector_python``).
        return (str(cfg.get('bank_scoring.python') or '').strip()
                or sys.executable)
    return sys.executable


def detect(force=False, extra_path='', profile=None) -> dict:
    """The whole picture for the picker: every candidate with its per-dependency
    verdict, which one is selected, and whether the selected one reaches the GPU.

    `extra_path` is a path the user typed — probed like any other candidate and
    reported even when it does not exist (that IS the answer they need). Never
    raises: a candidate that explodes degrades to 'unreachable'."""
    prof = get_profile(profile)
    selected = (cfg.get(prof.config_key) or '').strip()
    entries = list(candidates(prof))
    known = {_norm(e['path']) for e in entries}
    # A hand-typed path is a FIRST-CLASS route, not a fallback: most installs
    # out there have neither ai-toolkit nor ComfyUI where we look (or at all),
    # and for those users this field IS the feature. A folder is expanded to
    # every interpreter shape it could hold, so "my conda env" works as well as
    # "my conda env's python.exe".
    resolved = resolve_entered_path(extra_path)
    entered_keys = {_norm(p) for p in resolved}
    for p in resolved:
        if _norm(p) not in known:
            known.add(_norm(p))
            entries.append({'path': p, 'source': 'manual',
                            'label': 'The path you entered'})
    # Probed in PARALLEL: each candidate costs a cold `import torch`, which is
    # seconds of native-DLL loading (and antivirus scanning) that spends its time
    # in a subprocess, not holding the GIL. Serially, four interpreters made the
    # dialog take the better part of a minute to open for the first time; this
    # makes the wait the slowest one instead of the sum. Order is preserved.
    def probe_one(entry):
        try:
            return describe(entry['path'], probe(entry['path'], force=force), prof)
        except Exception:      # noqa: BLE001 — a broken candidate is a row, not a 500
            return describe(entry['path'], None, prof)

    with ThreadPoolExecutor(max_workers=min(8, len(entries) or 1)) as pool:
        verdicts = list(pool.map(probe_one, entries))
    out = []
    for entry, verdict in zip(entries, verdicts):
        verdict.update(source=entry['source'], label=entry['label'],
                       selected=bool(selected) and _norm(entry['path']) == _norm(selected),
                       # Marks the row the typed path landed on — INCLUDING one we
                       # already knew about. Without it, entering a path the list
                       # already holds looks like the button did nothing, which is
                       # the worst answer for the route most users depend on.
                       entered=_norm(entry['path']) in entered_keys)
        out.append(verdict)
    return {
        'selected': selected,
        'profile': prof.key,
        # No explicit selection = the pass runs wherever the resolver lands.
        # Naming it keeps "what am I on right now" answerable in both states —
        # and for the semantic engine that is NOT sys.executable: its resolver
        # still falls back to Score's key for configs written before it had one.
        'default_python': default_python(prof),
        # Is there an NVIDIA card here AT ALL (nvidia-smi, cached ~10 min)? Drives
        # the WORDING, never a refusal. A machine with no card — or an AMD/Intel
        # one — has nothing to fix, and a screen talking to it about CUDA is pure
        # noise. It can still borrow an interpreter that already has the
        # packages; it just isn't sold a speed-up it cannot have.
        'nvidia_present': nvidia_present(),
        # What became of a path the user typed: '' when nothing was typed,
        # 'resolved' when it produced at least one row, 'no_interpreter' when the
        # folder exists but holds nothing that could be one. Silence is the wrong
        # answer to someone who just typed a path and pressed a button.
        'entered_status': ('' if not _clean_path(extra_path)
                           else 'resolved' if resolved else 'no_interpreter'),
        'interpreters': out,
    }


def nvidia_present() -> bool:
    """True when this machine has an NVIDIA card we could ever use. False for no
    GPU, an AMD/Intel GPU, or no driver — three situations that are STATES, not
    errors. Never raises: gpu_vram_gb() returns None for 'unknown', which we
    read as 'assume nothing to offer' rather than as a failure."""
    from ..capabilities import gpu_vram_gb
    try:
        return gpu_vram_gb() is not None
    except Exception:      # noqa: BLE001
        return False


class SelectionError(ValueError):
    """A refused selection, carrying the verdict so the caller can show WHY."""

    def __init__(self, message, verdict=None):
        super().__init__(message)
        self.verdict = verdict


def _save_selection(config_key: str, value: str) -> None:
    """Write one dotted config key without touching its siblings. The key names
    are stored in user configs (``bank_scoring.python``, ``bank_semantic.python``)
    and are never renamed — they are split here, never rebuilt from a label."""
    section, _, leaf = config_key.partition('.')
    cfg.save_config({section: {leaf: value}})


def select(path: str, profile=None) -> dict:
    """Point one feature at `path` (or back at the app default when blank).

    Verifies FIRST and refuses anything it could not prove — an interpreter that
    is missing open_clip would fail an hour into a Score pass, and one whose
    transformers predates ``Siglip2Model`` would fail the same way on the
    semantic index. On success the capability caches are dropped so
    ``bank_scoring_gpu_available()`` / ``bank_siglip2_gpu_available()`` and the
    buttons agree with the new choice immediately, with no restart.

    This is an EXECUTION choice and nothing else. No Setup action reads the key
    it writes to pick an install target."""
    from .. import capabilities
    prof = get_profile(profile)
    target = (path or '').strip()
    if not target:
        _save_selection(prof.config_key, '')
        capabilities.clear_import_cache()
        return {'selected': '', 'reverted': True, 'profile': prof.key}
    verdict = describe(target, probe(target, force=True), prof)
    if not verdict['usable']:
        if verdict['status'] == 'unreachable':
            raise SelectionError(
                'That path did not answer as a Python interpreter — '
                'nothing was changed.', verdict)
        raise SelectionError(
            f"That Python cannot run {prof.feature}: {verdict['detail']}. "
            'Nothing was changed — install the missing packages there first.',
            verdict)
    _save_selection(prof.config_key, target)
    capabilities.clear_import_cache()
    return {'selected': target, 'reverted': False, 'verdict': verdict,
            'profile': prof.key}
