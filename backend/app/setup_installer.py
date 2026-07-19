"""Setup installer: run whitelisted, self-contained installs in a background
thread and expose their live state for polling. Actions:

  ml_extras          -> pip install -r backend/requirements-ml.txt (the app's own venv):
                        installs ALL the ML extras at once — kept for a first-time setup
  face_scoring       -> pip install JUST the face-scoring packages (insightface + onnx-
                        runtime, versions read from requirements-ml.txt) into the inter-
                        preter probe_face_scoring resolves — install/repair ONE feature
  masks              -> pip install JUST the person-mask package (rembg) into the inter-
                        preter probe_masks resolves — install/repair ONE feature
  watermark_inpaint  -> install the watermark-inpainting package (simple-lama-inpainting,
                        version floor read from requirements-ml.txt) into a dedicated
                        3.10-3.12 interpreter. When the user has configured one it is used;
                        otherwise the action AUTO-PROVISIONS one — finds a base Python
                        3.10-3.12, builds an isolated venv under the data dir, installs CPU
                        torch + simple-lama into it, and records it as watermark.python. No
                        manual venv, no setting to edit (the package needs Pillow<10 and can
                        never share the app's Pillow-12 venv)
  (face_scoring/masks/watermark_inpaint all follow the same shape: ML interpreter resolved
   per capability, requirements-ml.txt pinned as a -c constraint, probe cache invalidated
   on success so the capability flips without a restart.)
  ollama_model       -> stream Ollama's /api/pull for the configured vision model
  klein_model        -> download the Klein 9B (KV) fp8 diffusion model into
                        <ComfyUI>/models/unet/klein/ — a PUBLIC download (no token). The KV
                        build caches the reference images' KV pairs on the first denoising
                        step, so multi-reference editing (the dataset engine's whole job) runs
                        up to 2.5x faster at identical quality. A 401 still logs recovery
                        steps as a safety net (see license_url below)
  klein_lora         -> download the consistency LoRA into <ComfyUI>/models/loras/klein/
  klein_text_encoder -> qwen_3_8b_fp8mixed into <ComfyUI>/models/text_encoders/
  klein_vae          -> flux2-vae into <ComfyUI>/models/vae/

No shell, no client-supplied arguments: each action's command/URL/destination is fixed.

Pip actions are SERIALIZED (one at a time, second request queued in click order): two
pip processes writing the same environment race on a shared package's dist-info and
corrupt it — proven by repro (two concurrent installs of one big binary package into
one venv fail 6/6 with WinError 2 / Errno 13). Each pip run also retries once on a
transient file-lock error (an antivirus holding a fresh file). Model downloads and the
ollama pull don't touch a venv, so they stay parallel.
"""
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import requests

from . import capabilities
from . import config as cfg

logger = logging.getLogger(__name__)

# Fixed catalog of the Klein downloads (re-checked 2026-07-17): all four files are
# PUBLIC downloads. The default UNET is the KV-cache build (flux-2-klein-9b-kv-fp8):
# it caches the reference images' KV pairs on the first denoising step, so multi-
# reference editing (the dataset engine's whole job) runs up to 2.5x faster at
# identical quality — same VAE/text-encoder. Unlike the plain 9b-fp8 repo (which is
# license-gated → 401 without a token), the KV repo is NOT access-gated: HF serves it
# publicly (verified: API gated=false, resolve → public CDN). The FLUX Non-Commercial
# License still governs USE. `license_url` is kept so a future re-gating (or a stale
# token) still degrades into actionable recovery steps rather than a bare 401.
# `legacy_names` = earlier default filenames still accepted as "already installed",
# so an install that fetched the pre-KV model never re-downloads ~10 GB (both variants
# resolve by name at generate time — see klein_edit_helper.resolve_klein_unet).
_KLEIN_DOWNLOADS = {
    'klein_model': {
        'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8/resolve/main/flux-2-klein-9b-kv-fp8.safetensors',
        'dest': ('unet', 'klein', 'flux-2-klein-9b-kv-fp8.safetensors'),
        'min_free_gb': 15, 'gated': False,
        'license_url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8',
        'legacy_names': ('flux-2-klein-9b-fp8.safetensors',),
    },
    'klein_lora': {
        'url': 'https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency/resolve/main/Flux2-Klein-9B-consistency-V2.safetensors',
        'dest': ('loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors'),
        'min_free_gb': 1, 'gated': False,
    },
    'klein_text_encoder': {
        'url': 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors',
        'dest': ('text_encoders', 'qwen_3_8b_fp8mixed.safetensors'),
        'min_free_gb': 12, 'gated': False,
    },
    'klein_vae': {
        'url': 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors',
        'dest': ('vae', 'flux2-vae.safetensors'),
        'min_free_gb': 2, 'gated': False,
    },
}

INSTALL_ACTIONS = ('ml_extras', 'scrape_extras', 'ollama_model',
                   'face_scoring', 'masks', 'watermark_inpaint',
                   'bank_scoring') + tuple(_KLEIN_DOWNLOADS)

_ML_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-ml.txt'
_SCRAPE_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-scrape.txt'
# pip -r installers share one worker; both target THIS interpreter (the scrape
# stack runs in-process, so any other environment would be invisible to the app).
_PIP_REQUIREMENTS = {'ml_extras': _ML_REQUIREMENTS, 'scrape_extras': _SCRAPE_REQUIREMENTS}
# The single package the watermark-inpaint scoped install adds. The NAME lives
# here (an identifier), but the VERSION SPEC is parsed from requirements-ml.txt
# so there's exactly one place a version floor is ever written.
_WATERMARK_PKG = 'simple-lama-inpainting'

# Bank scoring extra: CLIP (open_clip) + the NSFW classifier (transformers/timm)
# for the aesthetic/NSFW/style pass. Installed into a dedicated auto-provisioned
# venv with CPU torch — never the Flask venv (torch is heavy and version-touchy).
# These are NOT in requirements-ml.txt (which the monolithic ml_extras installs
# into the Flask venv); they live here and install only through the bank_scoring
# action, same isolation as the watermark torch install.
_BANK_SCORING_PKGS = ('open_clip_torch', 'transformers', 'timm', 'safetensors',
                      'huggingface_hub')

# The app's core requirements — Pillow is PINNED here (Pillow==12.x). An install
# that targets the Flask venv appends this pin so pip can never downgrade Pillow to
# satisfy an ML dependency (see _flask_pillow_guard).
_APP_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements.txt'

# ML packages that must NEVER install into the Flask (app's own) venv: their pins
# would drag Pillow below the version the app REQUIRES (simple-lama-inpainting
# hard-requires pillow<10 vs the app's Pillow 12). They install ONLY into a
# dedicated ML interpreter (watermark.python / masks.python — a separate 3.10-3.12
# env); with none configured the install is refused with an actionable message,
# never forced into the Flask venv. That silent Pillow downgrade is the root of the
# "corrupted Python environment that survives updates" bug this module guards.
_FLASK_VENV_INCOMPATIBLE = frozenset({_WATERMARK_PKG})

# --- ML extras, split per capability -------------------------------------------
# requirements-ml.txt is a FLAT pip file (not grouped by feature), so the
# package->capability grouping lives HERE. The VERSIONS are never duplicated: each
# package's exact requirement line is read from requirements-ml.txt via
# _requirement_spec(), and that same file rides along as a `-c` constraint so a
# scoped install can't bump numpy past insightface's <2 ABI ceiling. A dedicated
# test (test_no_orphan_ml_package) asserts EVERY line in requirements-ml.txt is
# owned by at least one capability below — a package added to the file but
# forgotten here would silently never be installed by any scoped action.
#
#   face_scoring  insightface (face embeddings) + onnxruntime (its runtime). numpy
#                 is pinned <2 *for insightface's* ABI; opencv-python-headless is
#                 the server-safe cv2 that insightface & rembg both pull — listed so
#                 the scoped install prefers the headless variant, matching the
#                 monolithic `-r` install.
#   masks         rembg (u2net background removal), + the same shared numpy /
#                 headless-opencv floor.
#   watermark_inpaint  simple-lama-inpainting (has its own dedicated worker below;
#                 listed here only so the anti-orphan test sees its package covered).
_CAPABILITY_PACKAGES = {
    'face_scoring': ('insightface', 'onnxruntime', 'numpy', 'opencv-python-headless'),
    'masks': ('rembg', 'numpy', 'opencv-python-headless'),
    'watermark_inpaint': (_WATERMARK_PKG,),
}
# The capabilities served by the GENERIC per-capability pip worker
# (_run_ml_capability). watermark_inpaint keeps its own worker, so it's excluded.
_CAPABILITY_ML_ACTIONS = ('face_scoring', 'masks')

# Actions whose success makes a NEW importable package appear -> the probe
# import-cache must be dropped so the capability flips without waiting out the
# 600 s TTL (ml_extras/scrape_extras via -r, the scoped per-capability installs).
_IMPORT_CACHE_ACTIONS = (frozenset(_PIP_REQUIREMENTS)
                         | set(_CAPABILITY_ML_ACTIONS)
                         | {'watermark_inpaint', 'bank_scoring'})

# Actions that invoke pip and therefore MUST NOT run concurrently: two pip processes
# writing the same environment race on a shared package's files/dist-info and corrupt
# it (proven by repro: two concurrent installs of one big binary package into one venv
# fail 6/6 with WinError 2 / Errno 13 on the package's dist-info). All the default ML
# installs target the app's own venv (no dedicated python), so these are serialized to
# ONE at a time; a second request is QUEUED in click order. Model downloads and the
# ollama pull touch models/ or the network, not a venv, so they are NOT here and keep
# running in parallel.
_PIP_ACTIONS = (frozenset(_PIP_REQUIREMENTS)
                | set(_CAPABILITY_ML_ACTIONS)
                | {'watermark_inpaint', 'bank_scoring'})

# Transient file-lock errors an install can hit even without concurrency: an antivirus
# or the search indexer briefly holding a just-written file at the moment pip renames
# it (classically Bitdefender on Windows -> Errno 13; a sharing violation -> WinError
# 32; access denied -> WinError 5). These are retryable: pip is idempotent, so rerunning
# finishes the interrupted step. A genuine "no wheel / build failed" error does NOT match
# and is surfaced immediately.
_RETRYABLE_PIP_ERR = re.compile(
    r'Errno 13|Permission denied|WinError 5\b|WinError 32|WinError 2\b|being used by another process',
    re.IGNORECASE)
_PIP_RETRIES = 3          # total attempts on a retryable error
_PIP_RETRY_BACKOFF = 3    # seconds * attempt number between tries

_LOG_MAX = 400  # ring-buffer the log so a chatty pip can't grow unbounded

_lock = threading.Lock()
_runs = {}  # action -> {'state', 'returncode', 'log', 'progress', 'waiting_for'}
# Pip serialization (guarded by _lock): the single action currently occupying the pip
# worker, and the FIFO of actions waiting their turn (click order).
_pip_current = None
_pip_queue = []


class AlreadyRunning(Exception):
    pass


class Precondition(Exception):
    pass


def _new_run():
    return {'state': 'running', 'returncode': None, 'log': [], 'progress': None,
            'waiting_for': None}


def _append(action, line):
    log = _runs[action]['log']
    log.append(line.rstrip('\n'))
    if len(log) > _LOG_MAX:
        del log[:-_LOG_MAX]


def _set_progress(action, done, total):
    """Publish a live byte-progress snapshot for a streaming download, separate
    from the text log (so a smooth % bar never spams the log). `total` may be 0
    when the server sends no content-length -> pct is None (indeterminate)."""
    run = _runs.get(action)
    if run is None:
        return
    run['progress'] = {
        'done': done,
        'total': total,
        'pct': (done * 100 // total) if total else None,
    }


def _quote(p: str) -> str:
    # Quote paths with spaces so the manual command is copy-paste-safe: the
    # portable bundle can be extracted under e.g. C:\Users\...\LoRA Dataset Studio\.
    return f'"{p}"' if ' ' in p else p


def _canon(name: str) -> str:
    """PEP 503 canonical form: -_. all fold to a single dash, case-insensitive."""
    return re.sub(r'[-_.]+', '-', name).lower()


# Canonical names of the Flask-venv-incompatible packages, for membership tests.
_INCOMPATIBLE_CANON = frozenset(_canon(n) for n in _FLASK_VENV_INCOMPATIBLE)


def _requirement_spec(name: str, requirements=_ML_REQUIREMENTS) -> str:
    """The full requirement line for `name` as written in a requirements file
    (e.g. 'simple-lama-inpainting>=0.1.2') — the version floor lives in ONE place
    (requirements-ml.txt), never duplicated in this module. Package-name match is
    canonicalised (PEP 503: -_. all fold together, case-insensitive) and tolerant
    of version/marker/extras suffixes. Falls back to the bare name if the file or
    line is missing (an unpinned `pip install <name>` still works)."""
    canon = _canon(name)
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()   # drop comments / blank lines
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]   # name before any spec/marker
            if _canon(token) == canon:
                return line
    except OSError:
        pass
    return name


def _ml_requirement_names(requirements=_ML_REQUIREMENTS) -> set:
    """Canonical names of every package declared in a requirements file (comments
    and blank lines dropped). Used by the anti-orphan test to prove each ML package
    is mapped to a capability in _CAPABILITY_PACKAGES."""
    names = set()
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]
            names.add(_canon(token))
    except OSError:
        pass
    return names


def _ml_requirement_specs(*, exclude=frozenset(), requirements=_ML_REQUIREMENTS) -> list:
    """Requirement lines from requirements-ml.txt in FILE ORDER, dropping any whose
    canonical name is in `exclude`. One source of truth for the ML versions — the
    monolithic ml_extras install builds its Flask-safe package list from here."""
    out = []
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]
            if _canon(token) in exclude:
                continue
            out.append(line)
    except OSError:
        pass
    return out


def _app_pillow_spec() -> str:
    """The Pillow pin from requirements.txt (e.g. 'Pillow==12.2.0') — the version the
    Flask venv MUST keep. Appended as an explicit requirement to any install that
    targets the Flask venv so pip REFUSES (clean error) rather than silently
    DOWNGRADES Pillow to satisfy an ML dependency. Bare-name fallback still blocks
    the known-bad <10 downgrade if the pin can't be parsed."""
    spec = _requirement_spec('Pillow', requirements=_APP_REQUIREMENTS)
    return spec if spec.lower() != 'pillow' else 'Pillow>=10'


def _is_flask_venv(python: str) -> bool:
    """True when `python` resolves to the app's OWN interpreter (the Flask venv) —
    the environment whose Pillow must never be downgraded. Case/separator-insensitive
    on Windows; never raises."""
    try:
        return os.path.samefile(python, sys.executable)
    except OSError:
        return (os.path.normcase(os.path.abspath(python))
                == os.path.normcase(os.path.abspath(sys.executable)))


def _flask_pillow_guard(python: str) -> list:
    """Pillow pin to append to a pip install ONLY when it targets the Flask venv:
    pip then can't silently downgrade the app's Pillow (it keeps it or fails clean).
    A dedicated ML env is exempt — it may legitimately need pillow<10 for
    simple-lama-inpainting."""
    return [_app_pillow_spec()] if _is_flask_venv(python) else []


def _watermark_python() -> str:
    """Interpreter the watermark LaMa wrapper resolves. Reuse the wrapper's OWN
    resolver (watermark.python > masks.python > sys.executable) so the install
    target and the later import can never drift apart."""
    from .services import watermark_lama
    return watermark_lama.lama_python()


def _capability_python(action) -> str:
    """Interpreter a scoped ML install targets — MUST match the resolution its
    matching probe uses, so the install target and the later import can't drift:
      face_scoring -> face_scoring.python  (see capabilities.probe_face_scoring)
      masks        -> masks.python         (see capabilities.probe_masks)
      watermark_inpaint -> the wrapper chain (watermark.python > masks.python)."""
    if action == 'watermark_inpaint':
        return _watermark_python()
    return cfg.get(f'{action}.python') or sys.executable


def manual_command(action) -> str:
    """The exact command that reproduces an install BY HAND, scoped to THIS app's
    own interpreter (sys.executable). A copy-paste then targets the SAME
    environment the app imports from -- the portable bundle's python\\python.exe or
    the dev venv -- instead of whatever bare `pip` happens to be first on PATH
    (which is the whole point of the user's question: a plain `pip install` would
    land in the wrong environment and the extras would never be importable)."""
    if action == 'ml_extras':
        # Install EVERYTHING in requirements-ml.txt EXCEPT the Pillow-incompatible
        # extra (that one needs its own env — see watermark_inpaint below), into this
        # interpreter, with Pillow PINNED so pip can't downgrade the app's Pillow.
        specs = ' '.join(f'"{s}"' for s in _ml_requirement_specs(exclude=_INCOMPATIBLE_CANON))
        guard = ' '.join(f'"{g}"' for g in _flask_pillow_guard(sys.executable))
        cmd = (f'{_quote(sys.executable)} -m pip install {specs} '
               f'-c {_quote(str(_ML_REQUIREMENTS))}')
        return f'{cmd} {guard}' if guard else cmd
    if action in _PIP_REQUIREMENTS:        # scrape_extras: pure-python -r install
        return f'{_quote(sys.executable)} -m pip install -r {_quote(str(_PIP_REQUIREMENTS[action]))}'
    if action in _CAPABILITY_ML_ACTIONS:
        # One scoped capability (face_scoring | masks): the exact version-pinned
        # lines from requirements-ml.txt, quoted (the '>=' / '<' are shell
        # redirection unquoted), plus that file as a -c constraint. Interpreter =
        # the same one the capability's probe resolves; when that's the Flask venv,
        # Pillow is pinned too so the scoped install can't downgrade it either.
        python = _capability_python(action)
        specs = ' '.join(f'"{_requirement_spec(p)}"' for p in _CAPABILITY_PACKAGES[action])
        guard = ' '.join(f'"{g}"' for g in _flask_pillow_guard(python))
        cmd = (f'{_quote(python)} -m pip install {specs} '
               f'-c {_quote(str(_ML_REQUIREMENTS))}')
        return f'{cmd} {guard}' if guard else cmd
    if action == 'watermark_inpaint':
        # Quote the spec: the '>=' in 'simple-lama-inpainting>=0.1.2' is shell
        # redirection unquoted. Interpreter = the wrapper's resolved python — but
        # NEVER the Flask venv (simple-lama needs pillow<10 and would break the app).
        # When nothing dedicated is configured the Install button AUTO-BUILDS a
        # dedicated venv; this debug/diagnostic line points at that managed venv.
        python = _watermark_python()
        spec = _requirement_spec(_WATERMARK_PKG)
        if _is_flask_venv(python):
            python = _watermark_env_python()
        return f'{_quote(python)} -m pip install "{spec}"'
    if action == 'bank_scoring':
        # The dedicated managed venv (auto-built) + CPU torch + the CLIP/NSFW stack.
        python = cfg.get('bank_scoring.python') or _bank_scoring_env_python()
        pkgs = ' '.join(_BANK_SCORING_PKGS)
        return (f'{_quote(python)} -m pip install torch --index-url {_TORCH_CPU_INDEX}  '
                f'&&  {_quote(python)} -m pip install {pkgs}')
    if action == 'ollama_model':
        model = (cfg.get('ollama.vision_model') or '').strip() or '<vision-model>'
        return f'ollama pull {model}'
    if action in _KLEIN_DOWNLOADS:
        spec = _KLEIN_DOWNLOADS[action]
        try:
            dest = _klein_dest_path(action)
        except Precondition:
            dest = os.path.join('<ComfyUI>', 'models', *spec['dest'])
        return f'curl -L -o "{dest}" "{spec["url"]}"'
    return ''


def status(action) -> dict:
    run = _runs.get(action)
    cmd = manual_command(action)
    if run is None:
        return {'state': 'idle', 'returncode': None, 'log': [], 'progress': None,
                'waiting_for': None, 'manual_command': cmd}
    return {'state': run['state'], 'returncode': run['returncode'],
            'log': list(run['log']), 'progress': run.get('progress'),
            # 'queued' -> which action it's waiting behind (the UI shows an honest
            # "waiting for another install" instead of a dead-looking button).
            'waiting_for': run.get('waiting_for'),
            # Kept for the diagnostic/debug log only — no longer shown as a user
            # "run this by hand" path (installs auto-recover or repair on re-click).
            'manual_command': cmd}


def start(action) -> dict:
    if action not in INSTALL_ACTIONS:
        raise ValueError(f'unknown action: {action}')
    global _pip_current
    with _lock:
        run = _runs.get(action)
        if run and run['state'] in ('running', 'queued'):
            raise AlreadyRunning(action)
        if action == 'ollama_model':
            _check_ollama_precondition()
        if action in _KLEIN_DOWNLOADS:
            _check_klein_precondition(action)
        _runs[action] = _new_run()
        if action in _PIP_ACTIONS and _pip_current is not None:
            # A pip install already owns the worker -> queue this one (FIFO, click
            # order) instead of racing it into the same environment. It starts on its
            # own when the current install finishes (see _release_pip_slot).
            _runs[action]['state'] = 'queued'
            _runs[action]['waiting_for'] = _pip_current
            _pip_queue.append(action)
            return status(action)
        if action in _PIP_ACTIONS:
            _pip_current = action
    threading.Thread(target=_execute, args=(action,), daemon=True).start()
    return status(action)


def _release_pip_slot(finished):
    """A pip action finished: free the worker and launch the next queued pip action
    (FIFO). Model downloads / ollama pulls never touch these globals."""
    global _pip_current
    nxt = None
    with _lock:
        if _pip_current == finished:
            _pip_current = None
        if _pip_queue and _pip_current is None:
            nxt = _pip_queue.pop(0)
            _pip_current = nxt
            run = _runs.get(nxt)
            if run is not None:
                run['state'] = 'running'
                run['waiting_for'] = None
    if nxt is not None:
        threading.Thread(target=_execute, args=(nxt,), daemon=True).start()


def _check_ollama_precondition():
    if not (cfg.get('ollama.url') or '').strip():
        raise Precondition('ollama.url not configured')
    if not (cfg.get('ollama.vision_model') or '').strip():
        raise Precondition('ollama.vision_model not configured')


def _klein_dest_path(action) -> str:
    """Absolute destination for a Klein download, under the VALIDATED ComfyUI
    models root. Raises Precondition when base_dir isn't a real install (we must
    never scatter multi-GB files under a wrong folder)."""
    r = capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')
    if not r['valid']:
        raise Precondition('point the app at a valid ComfyUI folder first (Setup, ComfyUI step)')
    spec = _KLEIN_DOWNLOADS[action]
    return os.path.join(r['resolved'], 'models', *spec['dest'])


def _check_klein_precondition(action):
    dest = _klein_dest_path(action)
    spec = _KLEIN_DOWNLOADS[action]
    try:
        free_gb = shutil.disk_usage(os.path.dirname(os.path.dirname(dest))).free / 1e9
        if free_gb < spec['min_free_gb']:
            raise Precondition(f'not enough disk space: {free_gb:.1f} GB free, '
                               f"~{spec['min_free_gb']} GB needed for this file")
    except OSError:
        pass   # unknown -> never block on a stat failure


# --- "Install everything" orchestrator -----------------------------------------
# One click that queues every install the app can run ITSELF right now — the missing
# ML extras, the Ollama vision model, and the Klein weights — instead of walking the
# user through each step. It never installs ComfyUI/Ollama themselves nor pastes API
# keys (those are external / credentials), so the plan is deliberately the subset whose
# preconditions are already satisfiable. Firing order is grouped by capability area for
# a coherent "X / N" progress display; the real scheduling still comes from start()
# (pip serialized FIFO, model downloads parallel), so the order here is cosmetic.
_INSTALL_ALL_ORDER = ('face_scoring', 'masks', 'watermark_inpaint', 'ollama_model',
                      'klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora')


def _action_needed(action, caps) -> bool:
    """Is `action` both MISSING and satisfiable right now, from live capabilities?
    Pure (caps in, bool out) — the single rule install_all_plan is built from."""
    if action in ('face_scoring', 'masks'):
        # These install into the app's OWN interpreter, so they need it inside the ML
        # wheel range (3.10-3.12); on a newer Python they'd only source-build and fail,
        # so "Install everything" skips them (the per-feature tile still explains why).
        if not (caps.get('python') or {}).get('ml_supported', True):
            return False
        return not caps.get(action)
    if action == 'watermark_inpaint':
        # Auto-provisions its own 3.10-3.12 venv, so it's runnable on any interpreter.
        return not caps.get('watermark_inpaint')
    if action == 'ollama_model':
        # Only when Ollama is already reachable AND a model name is configured (the pull
        # needs a target) — Ollama itself can't be auto-installed here.
        o = caps.get('ollama') or {}
        return bool(o.get('reachable') and not o.get('vision_model_ready')
                    and (o.get('vision_model') or '').strip())
    if action in _KLEIN_DOWNLOADS:
        # Only into a VALIDATED ComfyUI tree (never scatter multi-GB files under a wrong
        # folder). klein_missing already lists exactly the asset actions still absent
        # (required trio + recommended LoRA).
        c = caps.get('comfyui') or {}
        return bool(c.get('dir_valid')) and action in (c.get('klein_missing') or [])
    return False


def install_all_plan(caps) -> list:
    """The ordered list of install actions 'Install everything' will queue for these
    capabilities — every MISSING component whose preconditions are already met. Pure and
    deterministic (order = _INSTALL_ALL_ORDER) so it can be tested and drives the global
    progress count. Empty => everything the app can install itself is already in place."""
    caps = caps or {}
    return [a for a in _INSTALL_ALL_ORDER if _action_needed(a, caps)]


def start_all(caps) -> dict:
    """Queue every action in install_all_plan(caps). Each start() applies the SAME rules
    as a single install (pip queued FIFO so two never race one venv; model downloads run
    in parallel; per-action preconditions enforced), so this is just a fan-out. An action
    already in flight (AlreadyRunning) reuses its live state; one momentarily unsatisfiable
    (Precondition) is reported as an error row rather than aborting the whole batch. Returns
    the plan + each action's status so the caller can render 'X / N' without re-deriving it."""
    plan = install_all_plan(caps)
    statuses = {}
    for action in plan:
        try:
            statuses[action] = start(action)
        except AlreadyRunning:
            statuses[action] = status(action)
        except (Precondition, ValueError) as e:
            statuses[action] = {'state': 'error', 'returncode': None, 'log': [str(e)],
                                'progress': None, 'waiting_for': None,
                                'manual_command': manual_command(action)}
    return {'plan': plan, 'statuses': statuses}


def status_many(actions) -> dict:
    """Per-action status for a set of actions (the live 'Install everything' plan), so the
    UI polls ONE endpoint instead of one request per action. Unknown names are dropped."""
    return {a: status(a) for a in actions if a in INSTALL_ACTIONS}


def _execute(action):
    try:
        rc = _WORKERS[action](action)
        _runs[action]['returncode'] = rc
        _runs[action]['state'] = 'success' if rc == 0 else 'error'
        if action in _IMPORT_CACHE_ACTIONS and rc == 0:
            try:
                capabilities.clear_import_cache()
            except Exception:
                # never downgrade a successful install; surface at debug only
                logger.debug('clear_import_cache failed after %s', action, exc_info=True)
        if action == 'ollama_model' and rc == 0:
            # A successful vision-model pull must flip the Setup step / diagnostic
            # 'vision model ready' probe NOW, not after the 30 s probe-cache TTL —
            # otherwise the Setup keeps saying "the vision model isn't pulled yet"
            # right after the pull the user just watched finish (issue #7).
            # clear_import_cache() also resets the main probe cache, so it's the one
            # call that forces a fresh /api/tags check on the next probe.
            try:
                capabilities.clear_import_cache()
            except Exception:
                logger.debug('probe-cache clear failed after ollama_model', exc_info=True)
        if action in _KLEIN_DOWNLOADS and rc == 0:
            # The training-base/model listers cache their scans 5 min — a freshly
            # downloaded model must show up on the next probe, not in 5 minutes.
            try:
                from .utils import comfyui
                comfyui.clear_model_caches()
            except Exception:
                logger.debug('clear_model_caches failed after %s', action, exc_info=True)
    except Exception as e:  # never let a worker thread die silently
        _append(action, f'error: {e}')
        _runs[action]['returncode'] = -1
        _runs[action]['state'] = 'error'
    finally:
        # Always hand the pip worker to the next queued install, even on failure — a
        # crashed install must not wedge the queue behind it.
        if action in _PIP_ACTIONS:
            _release_pip_slot(action)


def _run_pip(action, cmd) -> int:
    """Run a pip command, streaming its output to the ring log, with a bounded retry
    on a TRANSIENT file-lock error (an antivirus/indexer holding a just-written file —
    Errno 13 / WinError 5|32|2). pip is idempotent, so a rerun finishes the interrupted
    step. A genuine build/resolution failure doesn't match _RETRYABLE_PIP_ERR and is
    returned immediately. Concurrency is already prevented by the pip queue; this is the
    single-process defence (the Bitdefender-style lock users without a queue still hit)."""
    rc = -1
    for attempt in range(1, _PIP_RETRIES + 1):
        buf = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            _append(action, line)
            buf.append(line)
        proc.wait()
        rc = proc.returncode
        if rc == 0:
            return 0
        if attempt < _PIP_RETRIES and any(_RETRYABLE_PIP_ERR.search(l) for l in buf):
            wait = _PIP_RETRY_BACKOFF * attempt
            _append(action, f'transient file-lock error (an antivirus or indexer may be '
                            f'holding a fresh file); retrying in {wait}s '
                            f'[{attempt}/{_PIP_RETRIES - 1}]')
            time.sleep(wait)
            continue
        return rc
    return rc


def _run_ml_extras(action) -> int:
    """pip-install worker for the two bundle actions (name kept for callers/tests):
      scrape_extras -> `pip install -r requirements-scrape.txt` (pure python) into THIS venv
      ml_extras     -> the Flask-SAFE ML extras only (face-scoring + masks packages),
                       into THIS venv, with Pillow PINNED so no dependency can
                       downgrade it. The Pillow-incompatible extra (simple-lama-
                       inpainting, which hard-requires pillow<10) is NEVER installed
                       here — it goes to a dedicated ML interpreter (see
                       _run_watermark_inpaint), so a full Setup can't corrupt the
                       Flask venv's Pillow. That corruption is the "environment that
                       survives updates" bug this split closes at the source.
    """
    if action != 'ml_extras':
        # scrape_extras: pure-python, safe to install straight into this interpreter.
        return _run_pip(action, [sys.executable, '-m', 'pip', 'install', '-r',
                                 str(_PIP_REQUIREMENTS.get(action, _ML_REQUIREMENTS))])

    # ml_extras (insightface/numpy<2/onnx) has no wheels outside Python 3.10–3.12;
    # on a newer interpreter pip source-builds and fails with a cryptic numpy
    # conflict. Lead the log with a plain-English explanation + the fix so the
    # traceback that follows is already contextualized.
    ps = capabilities.python_ml_status()
    if not ps['ml_supported']:
        for line in (
            '=' * 64,
            f"NOTE: this app runs on Python {ps['version']}, but the ML extras",
            f"need Python {ps['ml_range']} (insightface / numpy<2 / onnxruntime",
            "publish no wheels for newer versions → pip will try to BUILD them",
            "from source and the install will likely fail below.",
            "",
            "These extras are OPTIONAL — they only add face-resemblance scoring",
            "and background masking. You can:",
            "  1. Skip them (the app works without them), or",
            "  2. Install them into a separate Python 3.11/3.12 venv and set",
            "     face_scoring.python + masks.python to it in Settings.",
            '=' * 64,
        ):
            _append(action, line)
    # Flask-safe subset (everything except the Pillow-incompatible extra) + Pillow
    # pinned so a transitive dep can never downgrade the app's Pillow.
    specs = _ml_requirement_specs(exclude=_INCOMPATIBLE_CANON)
    cmd = ([sys.executable, '-m', 'pip', 'install', *specs,
            '-c', str(_ML_REQUIREMENTS)] + _flask_pillow_guard(sys.executable))
    rc = _run_pip(action, cmd)
    # The Pillow-incompatible extra is deliberately absent from the Flask venv: say
    # so and how to add it safely, so "install everything" never silently half-does
    # the job (and never breaks Pillow doing it). The 'Install inpainting' button now
    # BUILDS a dedicated Python for it automatically — no manual venv to create.
    for pkg in sorted(_FLASK_VENV_INCOMPATIBLE):
        _append(action, f"note: {pkg} is NOT installed into the app's own Python "
                        f"(it needs Pillow<10, which would break the app). Click the "
                        f"'Install inpainting' button to enable it — it builds a "
                        f"dedicated Python for you automatically.")
    return rc


# --- Auto-provisioned watermark venv -------------------------------------------
# simple-lama-inpainting hard-requires Pillow<10, so it can never share the app's
# Pillow-12 venv. When the user hasn't pointed watermark.python at a dedicated
# 3.10-3.12 interpreter, the Install button BUILDS one for them: find a base Python
# 3.10-3.12 on the machine, create an isolated venv under the app's data dir, install
# CPU torch + simple-lama-inpainting into it, and record its interpreter as
# watermark.python so the probe + wrapper resolve there. No manual venv, no setting to
# edit. Idempotent: a re-click reuses/repairs the same venv; a user's own
# watermark.python is always respected and never overwritten.
_VENV_PY_MIN = (3, 10)   # mirrors capabilities._ML_PY_MIN/_MAX (the ML wheel range):
_VENV_PY_MAX = (3, 12)   # torch / simple-lama publish wheels for CPython 3.10-3.12.
# CPU torch, installed EXPLICITLY into the managed venv: reliable and small on every OS
# (no CUDA toolkit, no multi-GB download), and watermark inpainting only repaints small
# masked regions where CPU is fine. watermark.device='auto' resolves to CPU when CUDA is
# absent, so the env works with zero config. A user who wants GPU points watermark.python
# at their own CUDA env — where we DON'T force CPU torch (we never downgrade their build).
_TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
# Budget for the post-install verification import (see _verify_watermark_import). Far
# longer than the capability probe's 60 s ceiling on purpose: importing simple-lama pulls
# in torch + torchvision + opencv (~430 MB of native code, a single 291 MB torch_cpu.dll),
# and the FIRST cold import on a fresh machine — real-time AV scanning brand-new DLLs — can
# run minutes. We pay that once, here, so the probe fired right after the install is warm.
_WARM_IMPORT_TIMEOUT = 300


def _watermark_env_dir():
    """The app-managed watermark venv directory (deterministic, under the data dir), so
    a re-click resolves the SAME venv — idempotent build/repair, never a duplicate."""
    return cfg.data_dir() / 'envs' / 'watermark'


def _venv_python(env_dir) -> str:
    return str(env_dir / ('Scripts' if os.name == 'nt' else 'bin')
              / ('python.exe' if os.name == 'nt' else 'python'))


def _watermark_env_python() -> str:
    """Absolute path to the app-managed watermark venv's python (may not exist yet)."""
    return _venv_python(_watermark_env_dir())


def _same_path(a, b) -> bool:
    """True when two paths point at the same interpreter. samefile when both exist,
    else a case/separator-insensitive compare (so a not-yet-built venv path matches)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return (os.path.normcase(os.path.abspath(a or ''))
                == os.path.normcase(os.path.abspath(b or '')))


def _python_minor(exe: str):
    """(major, minor) reported by RUNNING `exe` — never trusted from its name/path —
    or None when it can't be executed. Short timeout, no console window."""
    try:
        proc = subprocess.run(
            [exe, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    m = re.match(r'^(\d+)\.(\d+)\s*$', proc.stdout or '')
    return (int(m.group(1)), int(m.group(2))) if m else None


def _base_python_candidates() -> list:
    """Interpreters to try as the BASE for `-m venv`, in reliability order. Names/paths
    only — each is version-checked by EXECUTION before use. We never install into these;
    we only spawn an isolated venv from one (its site-packages are never touched)."""
    cands = []
    if os.name == 'nt':
        # 1. Windows launcher: explicit 3.12 > 3.11 > 3.10 (resolve the tag to a path).
        launcher = shutil.which('py')
        if launcher:
            for tag in ('3.12', '3.11', '3.10'):
                try:
                    p = subprocess.run([launcher, f'-{tag}', '-c',
                                        'import sys; print(sys.executable)'],
                                       capture_output=True, text=True, timeout=15,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    exe = (p.stdout or '').strip()
                    if p.returncode == 0 and exe:
                        cands.append(exe)
                except (OSError, subprocess.SubprocessError):
                    pass
    # 2. On PATH.
    for name in ('python3.12', 'python3.11', 'python3.10', 'python3', 'python'):
        exe = shutil.which(name)
        if exe:
            cands.append(exe)
    # 3. Standard per-user / system install locations (Windows).
    if os.name == 'nt':
        for root in (os.environ.get('LOCALAPPDATA', ''), os.environ.get('PROGRAMFILES', ''),
                     os.environ.get('PROGRAMFILES(X86)', ''), 'C:\\'):
            if not root:
                continue
            for ver in ('312', '311', '310'):
                cands.append(os.path.join(root, 'Programs', 'Python', f'Python{ver}', 'python.exe'))
                cands.append(os.path.join(root, f'Python{ver}', 'python.exe'))
    # 4. Pythons the app already knows — used ONLY as a venv base. sys.executable (the
    #    app's own 3.12 venv) is a perfect base on a portable-bundle machine that has no
    #    other Python installed: `-m venv` from it makes a fresh, empty env, so the app's
    #    Pillow 12 is never touched.
    cands.append(sys.executable)
    for key in ('face_scoring.python', 'masks.python'):
        v = (cfg.get(key) or '').strip()
        if v:
            cands.append(v)
    try:
        ai = cfg.aitoolkit_path('venv_python')
        if ai:
            cands.append(str(ai))
    except Exception:
        pass
    # Dedupe, preserving order (normcase for Windows path equality).
    seen, out = set(), []
    for c in cands:
        key = os.path.normcase(os.path.abspath(c)) if c else ''
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _find_base_python(action) -> str:
    """First candidate interpreter whose REAL (executed) version is 3.10-3.12, else ''.
    Logs the chosen base so a report can see exactly what was used."""
    for exe in _base_python_candidates():
        ver = _python_minor(exe)
        if ver is not None and _VENV_PY_MIN <= ver <= _VENV_PY_MAX:
            _append(action, f'found base Python {ver[0]}.{ver[1]}: {exe}')
            return exe
    return ''


def _ensure_watermark_env(action) -> str:
    """Build (or reuse) the app-managed watermark venv and record it as watermark.python.
    Returns the venv python on success, '' on failure (an actionable one-liner is logged).
    Idempotent: an existing venv is reused; a missing one is (re)built."""
    env_dir = _watermark_env_dir()
    env_python = _venv_python(env_dir)
    if not os.path.isfile(env_python):
        base = _find_base_python(action)
        if not base:
            for line in (
                'No Python 3.10-3.12 was found to build the inpainting environment '
                '(simple-lama-inpainting needs Pillow<10, so it must live in its own '
                'Python, never the app\'s).',
                'Install Python 3.12, then click Install again:',
                '  python.org/downloads  (tick "Add python.exe to PATH")',
                '  or:  winget install Python.Python.3.12',
            ):
                _append(action, line)
            return ''
        _append(action, f'building the watermark environment at {env_dir}')
        try:
            env_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _append(action, f'could not create the data folder: {e}')
            return ''
        # venv creation is quick; stream it through the same retry helper so an AV lock
        # on a freshly-written pyvenv file is retried rather than failing the whole build.
        rc = _run_pip(action, [base, '-m', 'venv', str(env_dir)])
        if rc != 0 or not os.path.isfile(env_python):
            _append(action, 'could not create the environment — see the log above')
            return ''
        _append(action, 'environment ready')
    else:
        _append(action, f'reusing the watermark environment at {env_dir}')
    # Record it so the probe + wrapper resolve here and a re-click repairs the SAME env.
    # Only reached when nothing dedicated was configured, so this never overrides a
    # user-set watermark.python.
    try:
        cfg.save_config({'watermark': {'python': env_python}})
    except Exception as e:
        _append(action, f'warning: could not save watermark.python ({e}); '
                        'the environment still works for this run')
    return env_python


def _pip_install_watermark(action, python, *, managed: bool) -> int:
    """Install simple-lama-inpainting into `python` (a dedicated 3.10-3.12 env). The
    version floor is read from requirements-ml.txt (single source of truth), which also
    rides along as a -c constraint so pulling torch can't bump numpy past insightface's
    <2 ceiling. For the app-managed venv (managed=True) CPU torch is installed FIRST and
    explicitly (small/reliable/cross-OS); a user's OWN env keeps whatever torch it has —
    we never downgrade a CUDA build there."""
    spec = _requirement_spec(_WATERMARK_PKG)
    _append(action, f'target interpreter: {python}')
    if managed:
        _append(action, 'installing CPU torch (download.pytorch.org/whl/cpu)')
        rc = _run_pip(action, [python, '-m', 'pip', 'install', 'torch',
                               '--index-url', _TORCH_CPU_INDEX, '-c', str(_ML_REQUIREMENTS)])
        if rc != 0:
            _append(action, f'torch install failed (rc={rc}) — see the log above')
            return rc
    _append(action, f'installing {spec}  (constraints: requirements-ml.txt)')
    return _run_pip(action, [python, '-m', 'pip', 'install', spec, '-c', str(_ML_REQUIREMENTS)])


def _verify_watermark_import(action, python) -> bool:
    """Actually IMPORT simple_lama_inpainting in the target interpreter once the pip
    step reports success. Two jobs, one import:

    1. HONESTY. pip 'Requirement already satisfied' proves the distribution is on disk,
       NOT that it loads — the same gap that let JoyCaption read 'ready' then crash with
       ModuleNotFoundError (issue #6). A torch/torchvision build mismatch pip can't see
       fails only at import. If the import errors, the install is NOT usable, so we fail
       it (the UI shows the reason + a repair click) instead of reporting success while
       the capability stays a silent ✗.
    2. WARMING. This is the app's heaviest probe import (~430 MB of native code, a single
       291 MB torch_cpu.dll). On a fresh machine the first cold import — real-time AV
       scanning brand-new DLLs — can exceed the capability probe's 60 s subprocess ceiling,
       so the probe fired right after this install (onDone → /api/capabilities) would time
       out and show '✗ Watermark inpainting' seconds after a fully successful install (the
       probe would flip green only on a LATER, warm probe). Doing that first cold import
       HERE, once, with a generous budget, leaves the OS/AV cache warm so the following
       probe is fast → green with no restart, as the one-click flow promises.

    Returns True = ready (import OK) or merely slow (a cold import past the budget is
    'still warming', never a reason to fail a good install). False = a genuine import
    error → the caller fails the install. Never raises."""
    if not os.path.isfile(python):
        return True   # no interpreter to check (should not happen post-install) — leave rc as-is
    _append(action, 'verifying the install (first import — this also warms it, so the '
                    'capability turns green without a restart)…')
    try:
        proc = subprocess.run([python, '-c', 'import simple_lama_inpainting'],
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=_WARM_IMPORT_TIMEOUT,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        _append(action, 'still warming up (the first import is slow on a fresh machine) — '
                        'the capability turns green on its own shortly; no restart needed')
        return True   # slow, not broken — keep the install successful
    except OSError as e:
        _append(action, f'could not run the verification import ({e}) — skipping the check')
        return True   # couldn't check — don't punish a pip install that succeeded
    if proc.returncode == 0:
        _append(action, 'import OK — watermark inpainting is ready')
        return True
    _append(action, 'installed, but simple_lama_inpainting does not import in this '
                    'environment yet — the install is not usable:')
    for line in (proc.stderr or '').strip().splitlines()[-4:]:
        _append(action, f'  {line}')
    return False


def _run_watermark_inpaint(action) -> int:
    """Install simple-lama-inpainting (LaMa) into a dedicated 3.10-3.12 interpreter —
    NEVER the Flask venv (the package hard-requires Pillow<10, which would downgrade and
    break the app's Pillow 12).

    When the user has pointed watermark.python (or masks.python) at a real dedicated env,
    install there. When NOTHING dedicated is configured, AUTO-PROVISION: build an isolated
    venv under the app's data dir and record it as watermark.python. This is the one-click
    replacement for the old refuse-with-instructions path — the user never creates a venv
    or edits a setting. Idempotent: a re-click reuses/repairs the same venv (and rebuilds
    it if it went missing); a user-set watermark.python is always respected."""
    managed_python = _watermark_env_python()
    configured = (cfg.get('watermark.python') or cfg.get('masks.python') or '').strip()
    # Auto-provision when nothing dedicated is configured, OR when the ONLY thing
    # configured is our own managed venv and it has gone missing (rebuild it).
    rebuild_managed = (bool(configured) and _same_path(configured, managed_python)
                       and not os.path.isfile(managed_python))
    if not configured or rebuild_managed:
        python = _ensure_watermark_env(action)
        if not python:
            return 1
    else:
        python = configured
        if _is_flask_venv(python):
            for line in (
                "watermark.python points at the app's own Python, but simple-lama-",
                "inpainting requires Pillow<10 and would break the app's Pillow 12.",
                "Nothing was installed. Clear watermark.python (and masks.python) and",
                "click Install again — the app will build a dedicated Python for you.",
                f"(refused target — the app's own interpreter: {sys.executable})",
            ):
                _append(action, line)
            return 1
    rc = _pip_install_watermark(action, python, managed=_same_path(python, managed_python))
    # A successful pip step is necessary but not sufficient: confirm the package actually
    # imports in `python` (and warm that heavy import so the probe fired right after is
    # green with no restart). A hard import error fails the install so it never reports
    # success over a silent ✗ capability.
    if rc == 0 and not _verify_watermark_import(action, python):
        return 1
    return rc


# --- Auto-provisioned bank-scoring venv ----------------------------------------
# The CLIP + NSFW stack (torch, open_clip, transformers, timm) is heavy and
# version-touchy, so it lives in its OWN app-managed venv rather than the Flask
# venv — same isolation and one-click build/repair as the watermark venv.
def _bank_scoring_env_dir():
    return cfg.data_dir() / 'envs' / 'bank_scoring'


def _bank_scoring_env_python() -> str:
    return _venv_python(_bank_scoring_env_dir())


def _ensure_bank_scoring_env(action) -> str:
    """Build (or reuse) the app-managed bank-scoring venv and record it as
    bank_scoring.python. Returns the venv python on success, '' on failure (an
    actionable one-liner is logged). Idempotent — mirrors _ensure_watermark_env."""
    env_dir = _bank_scoring_env_dir()
    env_python = _venv_python(env_dir)
    if not os.path.isfile(env_python):
        base = _find_base_python(action)
        if not base:
            for line in (
                'No Python 3.10-3.12 was found to build the bank-scoring environment '
                '(the CLIP aesthetic/NSFW stack installs into its own Python, never '
                "the app's).",
                'Install Python 3.12, then click Install again:',
                '  python.org/downloads  (tick "Add python.exe to PATH")',
                '  or:  winget install Python.Python.3.12',
            ):
                _append(action, line)
            return ''
        _append(action, f'building the bank-scoring environment at {env_dir}')
        try:
            env_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _append(action, f'could not create the data folder: {e}')
            return ''
        rc = _run_pip(action, [base, '-m', 'venv', str(env_dir)])
        if rc != 0 or not os.path.isfile(env_python):
            _append(action, 'could not create the environment — see the log above')
            return ''
        _append(action, 'environment ready')
    else:
        _append(action, f'reusing the bank-scoring environment at {env_dir}')
    try:
        cfg.save_config({'bank_scoring': {'python': env_python}})
    except Exception as e:
        _append(action, f'warning: could not save bank_scoring.python ({e}); '
                        'the environment still works for this run')
    return env_python


def _run_bank_scoring(action) -> int:
    """Install the bank-scoring stack (CPU torch + open_clip + transformers + timm)
    into a dedicated 3.10-3.12 interpreter — NEVER the Flask venv. Auto-provisions a
    managed venv when nothing is configured; respects a user-set bank_scoring.python.
    Verifies the import at the end so a pip-success-but-import-fail never reports a
    ready capability over a silent ✗ (same honesty gate as the watermark install)."""
    managed_python = _bank_scoring_env_python()
    configured = (cfg.get('bank_scoring.python') or '').strip()
    rebuild_managed = (bool(configured) and _same_path(configured, managed_python)
                       and not os.path.isfile(managed_python))
    if not configured or rebuild_managed:
        python = _ensure_bank_scoring_env(action)
        if not python:
            return 1
    else:
        python = configured
        if _is_flask_venv(python):
            for line in (
                "bank_scoring.python points at the app's own Python, but the CLIP/NSFW",
                "stack is heavy and installs into its own Python. Nothing was installed.",
                "Clear bank_scoring.python and click Install again — the app builds a",
                "dedicated Python for you.",
                f"(refused target — the app's own interpreter: {sys.executable})",
            ):
                _append(action, line)
            return 1
    managed = _same_path(python, managed_python)
    _append(action, f'target interpreter: {python}')
    if managed:
        _append(action, 'installing CPU torch (download.pytorch.org/whl/cpu)')
        rc = _run_pip(action, [python, '-m', 'pip', 'install', 'torch',
                               '--index-url', _TORCH_CPU_INDEX])
        if rc != 0:
            _append(action, f'torch install failed (rc={rc}) — see the log above')
            return rc
    _append(action, f"installing {', '.join(_BANK_SCORING_PKGS)}")
    rc = _run_pip(action, [python, '-m', 'pip', 'install', *_BANK_SCORING_PKGS])
    if rc == 0 and not _verify_bank_scoring_import(action, python):
        return 1
    return rc


def _verify_bank_scoring_import(action, python) -> bool:
    """Import torch/open_clip/transformers in the target env once pip reports done —
    HONESTY (a torch/torchvision mismatch fails only at import) and WARMING (a heavy
    cold import that would time out the 60 s capability probe fired right after). A
    timeout is 'still warming', never a failure. Mirrors _verify_watermark_import."""
    if not os.path.isfile(python):
        return True
    _append(action, 'verifying the install (first import — this also warms it, so the '
                    'capability turns green without a restart)…')
    try:
        proc = subprocess.run([python, '-c', 'import torch, open_clip, transformers'],
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=_WARM_IMPORT_TIMEOUT,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        _append(action, 'still warming up (the first import is slow on a fresh machine) — '
                        'the capability turns green on its own shortly; no restart needed')
        return True
    except OSError as e:
        _append(action, f'could not run the verification import ({e}) — skipping the check')
        return True
    if proc.returncode == 0:
        _append(action, 'import OK — bank scoring is ready')
        return True
    _append(action, 'installed, but the bank-scoring stack does not import in this '
                    'environment yet — the install is not usable:')
    for line in (proc.stderr or '').strip().splitlines()[-4:]:
        _append(action, f'  {line}')
    return False


def _run_ml_capability(action) -> int:
    """Install JUST the packages ONE ML capability needs (face_scoring | masks)
    into the interpreter that capability's probe resolves — so a user can install
    or REPAIR a single feature without the monolithic `-r requirements-ml.txt`.
    Versions come solely from requirements-ml.txt (via _requirement_spec) and that
    file rides along as a `-c` constraint, so pulling insightface/rembg deps can
    never bump numpy past the <2 ABI ceiling and break the other ML capabilities.
    Same shape as _run_watermark_inpaint (resolved ML python, -c constraint)."""
    python = _capability_python(action)
    specs = [_requirement_spec(p) for p in _CAPABILITY_PACKAGES[action]]
    # face_scoring pulls insightface, which only has wheels for Python 3.10–3.12.
    # When targeting THIS interpreter (no dedicated env) and it's out of range,
    # lead with the plain-English reason so the pip source-build failure below is
    # already contextualised — same courtesy the monolithic ml_extras worker gives.
    if action == 'face_scoring' and python == sys.executable:
        ps = capabilities.python_ml_status()
        if not ps['ml_supported']:
            _append(action, f"NOTE: Python {ps['version']} is outside the ML wheel "
                            f"range {ps['ml_range']} — insightface has no wheel here, "
                            "so pip will try to build it and likely fail. Install into a "
                            "separate 3.11/3.12 env and set face_scoring.python instead.")
    _append(action, f'target interpreter: {python}')
    _append(action, f"installing {', '.join(specs)}  (constraints: requirements-ml.txt)")
    # When this capability targets the Flask venv (no dedicated python), pin Pillow
    # so pulling insightface/rembg deps can't downgrade the app's Pillow either.
    return _run_pip(action, [python, '-m', 'pip', 'install', *specs,
                             '-c', str(_ML_REQUIREMENTS), *_flask_pillow_guard(python)])


def _klein_present_in_extra(action) -> bool:
    """Is the Klein asset for `action` already on disk under an extra_model_paths.yaml
    root? We still DOWNLOAD into the base is-default tree (dest is unchanged, per the
    "install location doesn't move" rule) — this only skips a redundant multi-GB fetch
    when the file already lives somewhere ComfyUI will load it. Accepts the canonical
    filename AND any earlier default name (`legacy_names`): an install that fetched the
    pre-KV UNET into an extra root still resolves it by name, so it must not re-download.
    EXTRA roots only (base presence is the os.path.isfile(dest) + _klein_variant_already_present
    checks), so with no yaml this is a no-op and behaviour is identical."""
    spec = _KLEIN_DOWNLOADS[action]
    dest_parts = spec['dest']                 # e.g. ('unet','klein','flux-2-...safetensors')
    comfy_type = dest_parts[0]                # 'unet'|'loras'|'text_encoders'|'vae'
    subdirs = dest_parts[1:-1]                # e.g. ('klein',) for the UNET, () otherwise
    names = (dest_parts[-1], *(spec.get('legacy_names') or ()))
    try:
        from .services import comfy_model_paths
        return any(os.path.isfile(os.path.join(root, *subdirs, name))
                   for root in comfy_model_paths.extra_roots(comfy_type)
                   for name in names)
    except Exception:
        logger.debug('extra-path klein presence check failed for %s', action, exc_info=True)
        return False


def _klein_variant_already_present(action):
    """Basename of a previously-accepted filename for `action` already on disk in the
    BASE dest folder (today: the pre-KV Klein UNET flux-2-klein-9b-fp8.safetensors),
    else None. When the default download filename changes, an install that fetched the
    old one stays valid — both variants resolve by name at generate time — so either
    counts as "already installed" instead of re-fetching ~10 GB. (extra_model_paths
    roots are covered by _klein_present_in_extra, which accepts the same alternates.)
    None when the spec lists no `legacy_names` (every other action)."""
    spec = _KLEIN_DOWNLOADS[action]
    alts = spec.get('legacy_names') or ()
    if not alts:
        return None
    try:
        dest_dir = os.path.dirname(_klein_dest_path(action))
    except Precondition:
        return None
    for name in alts:
        if os.path.isfile(os.path.join(dest_dir, name)):
            return name
    return None


def _run_klein_download(action) -> int:
    """Stream one Klein asset into the validated ComfyUI tree. Writes to a .part
    file then renames (a killed download never leaves a half file the model
    scanners would pick up). Progress lines land in the ring log (~every 512 MB).
    An access-denied repo (401/403) with a license_url -> actionable recovery steps, rc 1."""
    spec = _KLEIN_DOWNLOADS[action]
    dest = _klein_dest_path(action)
    if os.path.isfile(dest):
        _append(action, f'already present: {dest}')
        return 0
    variant = _klein_variant_already_present(action)
    if variant:
        _append(action, f'already present ({variant}) — an earlier Klein UNET build is '
                        'installed and still resolves; skipping download')
        return 0
    if _klein_present_in_extra(action):
        _append(action, 'already available via a configured extra_model_paths.yaml root - skipping download')
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {}
    token = cfg.secret('HF_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    _append(action, f"downloading {spec['url']}")
    _append(action, f'-> {dest}')
    part = dest + '.part'
    try:
        with requests.get(spec['url'], stream=True, timeout=(10, 120),
                          headers=headers, allow_redirects=True) as resp:
            if resp.status_code in (401, 403):
                if spec.get('gated') or spec.get('license_url'):
                    # Normally public (KV UNET); a 401/403 here means HF is denying
                    # access anyway (re-gated, or a stale HF_TOKEN was sent) -> the
                    # fix is still: accept the licence + provide a valid token.
                    _append(action, f'HTTP {resp.status_code} - Hugging Face denied access to this file.')
                    _append(action, f"1. Open {spec['license_url']} and accept the licence (free)")
                    _append(action, '2. Create a read token at https://huggingface.co/settings/tokens')
                    _append(action, '3. Paste it as HF_TOKEN in Settings -> API keys, then retry')
                    _append(action, '   (or download the file manually into the folder above)')
                else:
                    _append(action, f'HTTP {resp.status_code}')
                return 1
            if resp.status_code >= 400:
                _append(action, f'HTTP {resp.status_code}')
                return 1
            total = int(resp.headers.get('content-length') or 0)
            done = 0
            next_mark = 0
            _set_progress(action, 0, total)   # show the bar from the first byte
            with open(part, 'wb') as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    _set_progress(action, done, total)   # live % for the UI bar (every chunk)
                    if done >= next_mark:                 # coarse milestone in the text log
                        pct = f' ({done * 100 // total}%)' if total else ''
                        _append(action, f'{done / 1e9:.2f} / {total / 1e9:.2f} GB{pct}')
                        next_mark = done + 512 * 1024 * 1024
        if total and done < total:
            _append(action, f'incomplete download ({done}/{total} bytes) - retry')
            os.remove(part)
            return 1
        os.replace(part, dest)
        _append(action, f'done -> {dest}')
        return 0
    except requests.RequestException as e:
        _append(action, f'network error: {e}')
        try:
            os.remove(part)
        except OSError:
            pass
        return 1


def _run_ollama_model(action) -> int:
    url = (cfg.get('ollama.url') or '').rstrip('/')
    model = cfg.get('ollama.vision_model') or ''
    resp = requests.post(f'{url}/api/pull', json={'name': model, 'stream': True},
                         stream=True, timeout=None)
    if resp.status_code >= 400:
        _append(action, f'HTTP {resp.status_code}')
        return 1
    for line in resp.iter_lines():
        if line:
            _append(action, line.decode('utf-8', 'replace') if isinstance(line, bytes) else str(line))
    return 0


_WORKERS = {**{a: _run_ml_extras for a in _PIP_REQUIREMENTS},   # ml_extras + scrape_extras
            'ollama_model': _run_ollama_model,
            **{a: _run_ml_capability for a in _CAPABILITY_ML_ACTIONS},  # face_scoring + masks
            'watermark_inpaint': _run_watermark_inpaint,
            'bank_scoring': _run_bank_scoring,
            **{a: _run_klein_download for a in _KLEIN_DOWNLOADS}}
# Structural invariant: every whitelisted action MUST have a worker — a missing
# entry surfaces as a cryptic "error: '<action>'" KeyError at runtime (live
# repro: scrape_extras was added to INSTALL_ACTIONS but not here).
assert set(INSTALL_ACTIONS) == set(_WORKERS), \
    f'INSTALL_ACTIONS/_WORKERS mismatch: {set(INSTALL_ACTIONS) ^ set(_WORKERS)}'
