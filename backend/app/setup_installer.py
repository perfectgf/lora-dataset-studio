"""Setup installer: run whitelisted, self-contained installs in a background
thread and expose their live state for polling. Actions:

  ml_extras          -> pip install -r backend/requirements-ml.txt (the app's own venv):
                        installs ALL the ML extras at once — kept for a first-time setup
  face_scoring       -> pip install JUST the face-scoring packages (insightface + onnx-
                        runtime, versions read from requirements-ml.txt) into the inter-
                        preter probe_face_scoring resolves — install/repair ONE feature
  masks              -> pip install JUST the person-mask package (rembg) into the inter-
                        preter probe_masks resolves — install/repair ONE feature
  watermark_inpaint  -> pip install JUST the watermark-inpainting package (simple-lama-
                        inpainting, version floor read from requirements-ml.txt) into the
                        interpreter the LaMa wrapper resolves — the scoped install shown
                        next to the Curate 🧽 tools, so a user who already has rembg/
                        insightface doesn't redo the whole ML extras step
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
"""
import logging
import os
import re
import shutil
import subprocess
import sys
import threading

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
                   'face_scoring', 'masks', 'watermark_inpaint') + tuple(_KLEIN_DOWNLOADS)

_ML_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-ml.txt'
_SCRAPE_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-scrape.txt'
# pip -r installers share one worker; both target THIS interpreter (the scrape
# stack runs in-process, so any other environment would be invisible to the app).
_PIP_REQUIREMENTS = {'ml_extras': _ML_REQUIREMENTS, 'scrape_extras': _SCRAPE_REQUIREMENTS}
# The single package the watermark-inpaint scoped install adds. The NAME lives
# here (an identifier), but the VERSION SPEC is parsed from requirements-ml.txt
# so there's exactly one place a version floor is ever written.
_WATERMARK_PKG = 'simple-lama-inpainting'

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
                         | set(_CAPABILITY_ML_ACTIONS) | {'watermark_inpaint'})
_LOG_MAX = 400  # ring-buffer the log so a chatty pip can't grow unbounded

_lock = threading.Lock()
_runs = {}  # action -> {'state', 'returncode', 'log'}


class AlreadyRunning(Exception):
    pass


class Precondition(Exception):
    pass


def _new_run():
    return {'state': 'running', 'returncode': None, 'log': [], 'progress': None}


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
        # NEVER the Flask venv (simple-lama needs pillow<10 and would break the app),
        # so when nothing dedicated is configured point at a separate env instead.
        python = _watermark_python()
        spec = _requirement_spec(_WATERMARK_PKG)
        if _is_flask_venv(python):
            return f'"<a separate Python 3.10-3.12>" -m pip install "{spec}"'
        return f'{_quote(python)} -m pip install "{spec}"'
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
                'manual_command': cmd}
    return {'state': run['state'], 'returncode': run['returncode'],
            'log': list(run['log']), 'progress': run.get('progress'),
            'manual_command': cmd}


def start(action) -> dict:
    if action not in INSTALL_ACTIONS:
        raise ValueError(f'unknown action: {action}')
    with _lock:
        run = _runs.get(action)
        if run and run['state'] == 'running':
            raise AlreadyRunning(action)
        if action == 'ollama_model':
            _check_ollama_precondition()
        if action in _KLEIN_DOWNLOADS:
            _check_klein_precondition(action)
        _runs[action] = _new_run()
    threading.Thread(target=_execute, args=(action,), daemon=True).start()
    return status(action)


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
        proc = subprocess.Popen(
            [sys.executable, '-m', 'pip', 'install', '-r',
             str(_PIP_REQUIREMENTS.get(action, _ML_REQUIREMENTS))],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            _append(action, line)
        proc.wait()
        return proc.returncode

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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        _append(action, line)
    proc.wait()
    rc = proc.returncode
    # The Pillow-incompatible extra is deliberately absent from the Flask venv: say
    # so and how to add it safely, so "install everything" never silently half-does
    # the job (and never breaks Pillow doing it).
    for pkg in sorted(_FLASK_VENV_INCOMPATIBLE):
        _append(action, f"note: {pkg} is NOT installed into the app's own Python "
                        f"(it needs Pillow<10, which would break the app). Enable it "
                        f"with the 'Install inpainting' button after pointing "
                        f"watermark.python at a separate Python 3.10-3.12 env.")
    return rc


def _run_watermark_inpaint(action) -> int:
    """Install JUST the watermark-inpainting package (simple-lama-inpainting, plus
    its torch/opencv deps) into the interpreter the LaMa wrapper resolves — which
    MUST NOT be the Flask venv: the package hard-requires Pillow<10 and installing
    it into the app's own environment would downgrade (break) Pillow 12. When no
    dedicated ML interpreter is configured (watermark.python / masks.python), that
    resolver falls back to the Flask venv → refuse cleanly with an actionable
    message instead of corrupting Pillow. That refusal IS the graceful degradation
    for the 'corrupted env that survives updates' bug: nothing installs, the app
    stays intact, and the log says exactly how to enable the feature safely.

    On a real dedicated env: a user who already ran the ML extras step keeps
    rembg/insightface (pip skips the already-satisfied ones). The version floor is
    READ from requirements-ml.txt (single source of truth), and that file rides
    along as a CONSTRAINT (-c) so pulling torch can never bump numpy past
    insightface's <2 ceiling and silently break face scoring."""
    python = _watermark_python()
    if _is_flask_venv(python):
        for line in (
            "watermark inpainting needs its OWN Python: simple-lama-inpainting requires",
            "Pillow<10, and installing it into the app's own environment would downgrade",
            "Pillow 12 and break the app. Nothing was installed.",
            "Fix: create a separate Python 3.10-3.12 environment, then set its",
            "python(.exe) as watermark.python in Settings and click Install again.",
            f"(refused target — the app's own interpreter: {sys.executable})",
        ):
            _append(action, line)
        return 1
    spec = _requirement_spec(_WATERMARK_PKG)
    _append(action, f'target interpreter: {python}')
    _append(action, f'installing {spec}  (constraints: requirements-ml.txt)')
    proc = subprocess.Popen(
        [python, '-m', 'pip', 'install', spec, '-c', str(_ML_REQUIREMENTS)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        _append(action, line)
    proc.wait()
    return proc.returncode


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
    proc = subprocess.Popen(
        [python, '-m', 'pip', 'install', *specs, '-c', str(_ML_REQUIREMENTS),
         *_flask_pillow_guard(python)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        _append(action, line)
    proc.wait()
    return proc.returncode


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
            **{a: _run_klein_download for a in _KLEIN_DOWNLOADS}}
# Structural invariant: every whitelisted action MUST have a worker — a missing
# entry surfaces as a cryptic "error: '<action>'" KeyError at runtime (live
# repro: scrape_extras was added to INSTALL_ACTIONS but not here).
assert set(INSTALL_ACTIONS) == set(_WORKERS), \
    f'INSTALL_ACTIONS/_WORKERS mismatch: {set(INSTALL_ACTIONS) ^ set(_WORKERS)}'
