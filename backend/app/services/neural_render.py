"""✨ Neural render — NVIDIA DLSS 5 Neural Rendering over finished video clips.

WHAT IT IS
The DLSS 5 model (NGX feature 18, the file ``nvngx_dlssnr.dll``) re-renders a
frame's materials and lighting: skin, hair and fabric gain structure the source
only implied. It was built for games, where the engine hands it motion vectors
and depth — but the model only REQUIRES a colour frame and an output surface
(its own log says "Missing Color or Output parameter" and nothing about the
rest), which is what makes a plain video a valid input. This module runs it
over a finished clip and writes a new clip.

WHAT IT NEEDS, AND WHO SUPPLIES WHAT
* Windows and an NVIDIA GPU with a recent display driver: the model is a D3D12
  PE and its temporal mode uses the driver's Optical Flow engine. There is no
  Linux or Docker path, and this is the first capability of the app gated by
  the operating system — ``status()`` says so in words rather than showing a
  button that cannot work.
* The BRIDGE: two small MIT-licensed DLLs from the ComfyUI-DLSS5-NR project
  (github.com/lisitskyaa/ComfyUI-DLSS5-NR), which is the open-source host this
  app drives — an in-process D3D12 bridge and the caller shim the model's
  caller check demands. Setup downloads them from the pinned release
  (``BRIDGE_RELEASE``: exact URL, size and SHA-256, so a re-uploaded asset is
  refused rather than trusted).
* The MODEL: ``nvngx_dlssnr.dll`` is NVIDIA's. It is not downloaded, not
  linked and not looked for anywhere but ``runtime_dir()``. The user places
  their own copy there; the Setup card says exactly where. NVIDIA ships the
  model for the RTX 50 series; community builds for older cards exist and
  their terms are the user's to weigh — the app only reports whether a file
  is present, and relays the model's own refusal when it will not run.

HOW A CLIP IS RENDERED
Never in this process. ``backend/infer/dlss5nr_infer.py`` is launched per clip
(see its docstring for the two hangs that decided this) and reports progress
as JSON lines; cancelling a render is killing that process. The interpreter is
the video extra's (``video.python``), the one that has numpy.

THE TWO SURFACES
* The video DATASET: a render REPLACES the clip in place — the dataset IS its
  flat folder of .mp4 + .txt, read by every trainer as-is, so a second file
  next to the clip would be trained on. The original is kept in
  ``backup_dir()``, OUTSIDE the dataset, write-once: a clip rendered twice is
  rendered from its original both times (no stacking), and 🩹 Restore puts the
  original back. Rendering keeps the caption, the row and the provenance
  columns untouched — only pixels change.
* The Video Test STUDIO: a NEW row, never an edit of the compared clip — the
  same rule as ↗ Smooth, with ``nr_of`` pointing at the source clip.
* The video BANK gets no verb: its clips are time ranges over a rush and the
  bank writes no media (``VideoClip``: "the thumbnail is the ONLY media the
  bank writes"). A render belongs to a clip that exists as a file.

THE DIALS
``tone`` and ``structure`` (0–2, default 1) are the model's own local tone and
local structure strengths; ``automask`` its automatic mask. ``intensity``,
``skin``, ``preset`` and ``style`` are NOT exposed: swept in both directions
through this bridge, they changed nothing (bit-identical output). ``tone``
matters more than it looks — at its default the model relights flat art and
greys pure whites; at 0 the tones stay and only structure is added, which is
what the "keep tones" preset in the UI sets.

``temporal`` keeps the model's history across frames with motion vectors the
driver estimates. A scene cut resets the history so the previous shot is never
smeared into the next (threshold re-measured on this app's clips: Merserk's
0.24 fired on nothing here). It needs a frame at least ``TEMPORAL_MIN_WIDTH`` px wide
(bisected: 700 fails, 704 passes, whatever the height) — ``auto`` picks it
when the clip allows it and falls back to still mode otherwise, and the
choice is reported.

Credits: the scene-cut reset and the "video-only encode, then mux the source's
audio and metadata back" shape follow Merserk's dlss5-visual-enhancer (MIT,
github.com/Merserk/dlss5-visual-enhancer). Measurements behind the constants
and the dial choices were taken on an RTX 4090, September 2026.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from .. import config as cfg
from . import bank_jobs, ffmpeg_tools, infer_env

logger = logging.getLogger(__name__)

# ── The bridge release this app installs ────────────────────────────────────
# Pinned by tag AND content: GitHub lets an author re-upload an asset under the
# same tag, so the URL alone is not an identity. A mismatch is a refusal, never
# a warning.
BRIDGE_PROJECT_URL = 'https://github.com/lisitskyaa/ComfyUI-DLSS5-NR'
BRIDGE_RELEASE = {
    'version': '0.3.0',
    'url': ('https://github.com/lisitskyaa/ComfyUI-DLSS5-NR/releases/download/'
            'v0.3.0/ComfyUI-DLSS5-NR-v0.3.0-windows-x64.zip'),
    'size': 176746,
    'sha256': '3b7d52507a5548d10c3f60f9a5ea4cc5eb2fd9bd715b536533df55887a1d907f',
    # zip member -> path under runtime_dir()
    'members': {
        'ComfyUI-DLSS5-NR/native/bin/dlss5nr_bridge.dll': 'dlss5nr_bridge.dll',
        'ComfyUI-DLSS5-NR/runtime/caller/nvngx.dll_comfy.dll': os.path.join('caller', 'nvngx.dll_comfy.dll'),
        'ComfyUI-DLSS5-NR/LICENSE': 'LICENSE-ComfyUI-DLSS5-NR.txt',
    },
}
BRIDGE_FILE = 'dlss5nr_bridge.dll'
SHIM_FILE = os.path.join('caller', 'nvngx.dll_comfy.dll')
MODEL_FILE = 'nvngx_dlssnr.dll'
# A real model is ~165 MB; a forwarder or a stub of the same name is ~100 KB.
# The size floor turns "the wrong file under the right name" into a sentence.
MODEL_MIN_BYTES = 50 * 1024 * 1024

# Optical Flow's width floor — measured on the bridge, see the module docstring.
# Mirrored in infer/dlss5nr_infer.py; test_neural_render pins the two together.
TEMPORAL_MIN_WIDTH = 704
SCENE_CUT_DEFAULT = 0.10   # see infer/dlss5nr_infer.py for the measurement behind it
CRF_DEFAULT = 17

DEFAULT_PARAMS = {'tone': 1.0, 'structure': 1.0, 'automask': False,
                  'temporal': 'auto', 'scene_cut': SCENE_CUT_DEFAULT,
                  # The levers the model does not expose (measured, see the
                  # child): extrapolation past its answer, extra passes, a 2x
                  # working size delivered at the clip's size.
                  'strength': 1.0, 'passes': 1, 'scale': 1}
STRENGTH_MAX = 3.0
PASSES_MAX = 3
TEMPORAL_MODES = ('auto', 'on', 'off')

JOB_KIND = 'neural_render'
_STUDIO_THREADS = {}   # clip id -> thread, for the "already rendering" refusal


class NeuralRenderError(ValueError):
    """A refusal with a sentence the UI can show as-is."""


# ── Where things live ───────────────────────────────────────────────────────

def runtime_dir(create=False) -> Path:
    root = cfg.data_dir() / 'dlss5nr'
    if create:
        (root / 'caller').mkdir(parents=True, exist_ok=True)
    return root


def backup_dir(dataset_id) -> Path:
    """Originals of rendered dataset clips — OUTSIDE the dataset folder on
    purpose (every trainer reads that folder whole)."""
    return cfg.data_dir() / 'video_nr_backup' / str(int(dataset_id))


def worker_python() -> str:
    """The interpreter that runs the child: the video extra's, which carries
    numpy — the only Python dependency the child has."""
    return cfg.get('video.python') or sys.executable


# ── Status: what is here, what is missing, said in words ────────────────────

def _driver_files() -> dict:
    """The two driver-installed files the bridge needs: the NGX core
    (``_nvngx.dll``, discovered from the DriverStore) and the Optical Flow API
    (``nvofapi64.dll``, temporal mode only). Their presence is what "an NVIDIA
    display driver is installed" looks like from disk."""
    sysroot = os.environ.get('SystemRoot') or r'C:\Windows'
    system32 = os.path.join(sysroot, 'System32')
    nvof = os.path.isfile(os.path.join(system32, 'nvofapi64.dll'))
    ngx = os.path.isfile(os.path.join(system32, '_nvngx.dll')) or bool(glob.glob(
        os.path.join(system32, 'DriverStore', 'FileRepository', 'nv_dispi.inf_amd64_*',
                     '_nvngx.dll')))
    return {'ngx': ngx, 'nvof': nvof}


def runtime_files(root=None) -> dict:
    root = Path(root) if root else runtime_dir()
    model = root / MODEL_FILE
    model_size = model.stat().st_size if model.is_file() else 0
    return {
        'bridge': (root / BRIDGE_FILE).is_file(),
        'shim': (root / SHIM_FILE).is_file(),
        'model': model.is_file() and model_size >= MODEL_MIN_BYTES,
        'model_present_but_small': model.is_file() and model_size < MODEL_MIN_BYTES,
        'model_size': model_size,
    }


def status(root=None, os_name=None, driver=None, worker_ok=None, ffmpeg_ok=None) -> dict:
    """The capability, as the Setup card and the two verbs read it.

    Every absence is a SENTENCE naming the gesture that fixes it, and the
    verdicts are kept apart because they are fixed differently: an OS cannot
    be installed, a driver comes from NVIDIA, the bridge from Setup's button,
    the model from the user. ``ready`` is the single verdict the verbs read.
    """
    os_ok = (os_name or os.name) == 'nt'
    drv = driver if driver is not None else (_driver_files() if os_ok else {'ngx': False, 'nvof': False})
    # The child needs numpy in the interpreter it runs under. The app's own
    # requirements do not carry numpy; the video extra does, in the same
    # interpreter this lane resolves — so the question is the video lane's
    # decode probe, asked with its own cache key (one subprocess per TTL, not
    # one per poll). Injectable for tests, which must not probe the machine.
    if worker_ok is None:
        if os_ok:
            from .. import capabilities as _caps
            worker_ok = _caps._cached_import('video_decode', worker_python(),
                                             _caps.CAPABILITY_IMPORTS['video'])
        else:
            worker_ok = False
    files = runtime_files(root)
    root_str = str(Path(root) if root else runtime_dir())
    missing = []
    if not os_ok:
        missing.append('Windows — the DLSS 5 model is a Direct3D 12 library and runs nowhere else')
    elif not drv['ngx']:
        missing.append('an NVIDIA display driver (the NGX runtime it installs was not found)')
    if os_ok and not worker_ok:
        missing.append('the video extra (numpy for the render process) — install it from Setup')
    # ffmpeg reads and writes the clip. Asked through the one definition of
    # 'the encoder works' (a RUN, cached), never a path check.
    if ffmpeg_ok is None:
        ffmpeg_ok = bool(ffmpeg_tools.ffmpeg_ready()['ok']) if os_ok else False
    if os_ok and not ffmpeg_ok:
        missing.append('ffmpeg (the video extra installs it) — install it from Setup')
    if not files['bridge'] or not files['shim']:
        missing.append('the neural rendering bridge — install it from Setup')
    if files['model_present_but_small']:
        missing.append(f'{MODEL_FILE} in {root_str} is not the model (a real one is about 165 MB)')
    elif not files['model']:
        missing.append(f'your own copy of {MODEL_FILE}, placed in {root_str}')
    return {
        'ready': not missing,
        'os_ok': os_ok,
        'driver_ngx': bool(drv['ngx']),
        'driver_nvof': bool(drv['nvof']),
        'worker': bool(worker_ok),
        'ffmpeg': bool(ffmpeg_ok),
        'bridge': bool(files['bridge'] and files['shim']),
        'model': bool(files['model']),
        'model_size': files['model_size'],
        'runtime_dir': root_str,
        'model_file': MODEL_FILE,
        'bridge_version': BRIDGE_RELEASE['version'],
        'bridge_url': BRIDGE_PROJECT_URL,
        'temporal_min_width': TEMPORAL_MIN_WIDTH,
        'missing': missing,
    }


# ── Dials ───────────────────────────────────────────────────────────────────

def normalize_params(raw) -> dict:
    """Validate a request's dials into the exact dict the child is given.
    Unknown keys are ignored, out-of-range values are refused with the range."""
    raw = raw or {}
    out = dict(DEFAULT_PARAMS)
    for key in ('tone', 'structure'):
        if key in raw and raw[key] is not None:
            try:
                val = float(raw[key])
            except (TypeError, ValueError):
                raise NeuralRenderError(f'{key} must be a number between 0 and 2')
            if not 0.0 <= val <= 2.0:
                raise NeuralRenderError(f'{key} must be between 0 and 2')
            out[key] = round(val, 3)
    if 'automask' in raw and raw['automask'] is not None:
        out['automask'] = bool(raw['automask'])
    if 'temporal' in raw and raw['temporal'] is not None:
        mode = str(raw['temporal']).lower()
        if mode not in TEMPORAL_MODES:
            raise NeuralRenderError("temporal must be 'auto', 'on' or 'off'")
        out['temporal'] = mode
    if 'strength' in raw and raw['strength'] is not None:
        try:
            k = float(raw['strength'])
        except (TypeError, ValueError):
            raise NeuralRenderError(f'strength must be a number between 0 and {STRENGTH_MAX:g}')
        if not 0.0 <= k <= STRENGTH_MAX:
            raise NeuralRenderError(f'strength must be between 0 and {STRENGTH_MAX:g}')
        out['strength'] = round(k, 2)
    if 'passes' in raw and raw['passes'] is not None:
        try:
            n = int(raw['passes'])
        except (TypeError, ValueError):
            raise NeuralRenderError(f'passes must be a whole number between 1 and {PASSES_MAX}')
        if not 1 <= n <= PASSES_MAX:
            raise NeuralRenderError(f'passes must be between 1 and {PASSES_MAX}')
        out['passes'] = n
    if 'scale' in raw and raw['scale'] is not None:
        if str(raw['scale']) not in ('1', '2'):
            raise NeuralRenderError('scale must be 1 or 2')
        out['scale'] = int(raw['scale'])
    if 'scene_cut' in raw and raw['scene_cut'] is not None:
        try:
            cut = float(raw['scene_cut'])
        except (TypeError, ValueError):
            raise NeuralRenderError('scene_cut must be a number between 0 and 1')
        if not 0.0 <= cut <= 1.0:
            raise NeuralRenderError('scene_cut must be between 0 and 1')
        out['scene_cut'] = cut
    return out


def decide_temporal(mode, width, nvof=True, passes=1) -> tuple[bool, str]:
    """Whether THIS clip is rendered with history, and why in one clause.
    ``on`` below the floor is a refusal, not a silent downgrade — the user
    asked for something the clip cannot have."""
    if mode == 'off':
        return False, 'still mode'
    if passes > 1:
        # Extra passes feed the model its own answer; a frame history would
        # then describe the wrong picture. Asked for explicitly, it is a refusal
        # the dialog already words; on auto it is simply still mode.
        if mode == 'on':
            raise NeuralRenderError('temporal mode and extra passes exclude each other — choose one')
        return False, 'still mode (extra passes)'
    if not nvof:
        if mode == 'on':
            raise NeuralRenderError('temporal mode needs NVIDIA Optical Flow, which this driver does not provide')
        return False, 'still mode (no Optical Flow in this driver)'
    if width is not None and width < TEMPORAL_MIN_WIDTH:
        if mode == 'on':
            raise NeuralRenderError(
                f'temporal mode needs a clip at least {TEMPORAL_MIN_WIDTH} px wide (this one is {width}) — '
                'choose still mode or leave it on auto')
        return False, f'still mode (narrower than {TEMPORAL_MIN_WIDTH} px)'
    return True, 'temporal mode'


# ── Probing a clip ──────────────────────────────────────────────────────────

def clip_dimensions(path) -> tuple[int, int] | None:
    ffmpeg = ffmpeg_tools.ffmpeg_path()
    if not ffmpeg:
        return None
    ffprobe = os.path.join(os.path.dirname(ffmpeg), os.path.basename(ffmpeg).replace('ffmpeg', 'ffprobe'))
    if not os.path.isfile(ffprobe):
        ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return None
    try:
        out = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                              '-show_entries', 'stream=width,height', '-of', 'json', str(path)],
                             capture_output=True, text=True, encoding='utf-8', errors='replace',
                             timeout=30)
        stream = (json.loads(out.stdout or '{}').get('streams') or [{}])[0]
        return int(stream['width']), int(stream['height'])
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None


# ── The render itself: one child per clip ───────────────────────────────────

def worker_argv(src, dst, params, temporal_on, ffmpeg=None) -> list:
    script = str(cfg.BACKEND_DIR / 'infer' / 'dlss5nr_infer.py')
    return infer_env.worker_argv(
        worker_python(), script,
        '--src', str(src), '--dst', str(dst), '--runtime', str(runtime_dir()),
        '--ffmpeg', str(ffmpeg or ffmpeg_tools.ffmpeg_path() or 'ffmpeg'),
        '--tone', str(params['tone']), '--structure', str(params['structure']),
        '--automask', '1' if params['automask'] else '0',
        '--temporal', 'on' if temporal_on else 'off',
        '--scene-cut', str(params['scene_cut']), '--crf', str(CRF_DEFAULT),
        '--strength', str(params.get('strength', 1.0)), '--passes', str(params.get('passes', 1)),
        '--scale', str(params.get('scale', 1)))


def render_video(src, dst, params, *, on_progress=None, cancel=None, timeout_s=None) -> dict:
    """Render ONE file into another. Blocks; raises NeuralRenderError with the
    child's own sentence on any failure. ``cancel`` is polled between lines and
    kills the child when it answers True — the only way to stop a native model
    mid-frame. Returns the child's ``done`` event (frames, mode, timings)."""
    st = status()
    if not st['ready']:
        raise NeuralRenderError('neural rendering is not set up: ' + '; '.join(st['missing']))
    if not ffmpeg_tools.ffmpeg_path():
        raise NeuralRenderError('ffmpeg is needed to read and write the clip')
    dims = clip_dimensions(src)
    width = dims[0] if dims else None
    temporal_on, mode_note = decide_temporal(params['temporal'], width, nvof=st['driver_nvof'],
                                             passes=int(params.get('passes', 1)))
    argv = worker_argv(src, dst, params, temporal_on)
    env = infer_env.worker_env(worker_python(), PYTHONUTF8='1')
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                            text=True, encoding='utf-8', errors='replace',
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    started = time.time()
    result = None
    error = None
    # stderr is drained on its own thread: a child that writes a long traceback
    # there while we block on stdout would deadlock the pipe.
    err_lines = []
    drain = threading.Thread(target=lambda: err_lines.extend(proc.stderr.read().splitlines()[-40:]),
                             daemon=True)
    drain.start()
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get('event')
            if kind == 'error':
                error = event.get('message') or 'the render failed'
            elif kind == 'done':
                result = event
            elif on_progress is not None:
                on_progress(event)
            if cancel is not None and cancel():
                proc.kill()
                raise NeuralRenderError('cancelled')
            if timeout_s and time.time() - started > timeout_s:
                proc.kill()
                raise NeuralRenderError(f'the render took longer than {int(timeout_s)} s and was stopped')
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        proc.wait(timeout=30) if proc.poll() is None else None
        drain.join(timeout=5)
    if error:
        raise NeuralRenderError(error)
    if result is None:
        tail = ' '.join(err_lines[-3:]).strip()
        raise NeuralRenderError('the render stopped without a result' + (f': {tail[-300:]}' if tail else ''))
    result['mode_note'] = mode_note
    return result


# ── The bridge install (Setup's button) ─────────────────────────────────────

def install_bridge(log=None, fetch=None) -> int:
    """Download the pinned bridge release, refuse anything but the pinned
    bytes, and unpack the two DLLs (plus the project's licence) into
    runtime_dir(). Returns 0 on success, 1 otherwise; every step is logged as
    a sentence. ``fetch(url) -> bytes`` exists for tests."""
    say = log or (lambda line: logger.info('dlss5nr bridge: %s', line))
    rel = BRIDGE_RELEASE
    say(f"downloading the neural rendering bridge v{rel['version']} from {rel['url']}")
    try:
        if fetch is None:
            import urllib.request
            req = urllib.request.Request(rel['url'], headers={'User-Agent': 'lora-dataset-studio'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        else:
            data = fetch(rel['url'])
    except Exception as exc:  # noqa: BLE001 — a download fails a hundred ways, all reported the same
        say(f'download failed: {exc}')
        return 1
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != rel['size'] or digest != rel['sha256']:
        say(f"refused: the download is {len(data)} bytes, sha256 {digest[:16]}…, "
            f"not the pinned release ({rel['size']} bytes, {rel['sha256'][:16]}…). "
            'The asset may have been re-uploaded — this app only installs the bytes it verified.')
        return 1
    root = runtime_dir(create=True)
    try:
        with zipfile.ZipFile(__import__('io').BytesIO(data)) as zf:
            names = set(zf.namelist())
            for member, rel_path in rel['members'].items():
                if member not in names:
                    say(f'refused: {member} is not in the archive')
                    return 1
            for member, rel_path in rel['members'].items():
                target = root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + '.part')
                with zf.open(member) as src, open(tmp, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(tmp, target)
                say(f'installed {rel_path}')
    except (zipfile.BadZipFile, OSError) as exc:
        say(f'unpacking failed: {exc}')
        return 1
    files = runtime_files(root)
    if not (files['bridge'] and files['shim']):
        say('the bridge files are still missing after the install')
        return 1
    say(f"bridge v{rel['version']} installed in {root}")
    if not files['model']:
        say(f'next: place your {MODEL_FILE} in {root} — this app does not download it')
    return 0


# ── Dataset clips: render in place, keep the original ───────────────────────

def job_key(dataset_id) -> str:
    """The bank_jobs slot. A string on purpose, like the video bank's
    ``video:<id>``: dataset 3 and image bank 3 must never share a slot."""
    return f'vds:{int(dataset_id)}'


def dataset_job(dataset_id):
    return bank_jobs.get(job_key(dataset_id))


def cancel_dataset_job(dataset_id) -> bool:
    return bank_jobs.cancel(job_key(dataset_id))


def _dataset_and_rows(user_id, dataset_id, clip_ids=None):
    from ..models import VideoDatasetClip
    from . import video_bank_service as vbs
    ds = vbs.get_video_dataset(user_id, dataset_id)
    if ds is None or not ds.output_dir:
        return None, []
    q = VideoDatasetClip.query.filter(VideoDatasetClip.dataset_id == ds.id)
    if clip_ids:
        ids = {int(i) for i in clip_ids if str(i).lstrip('-').isdigit()}
        q = q.filter(VideoDatasetClip.id.in_(ids))
    return ds, q.order_by(VideoDatasetClip.filename).all()


def rendered_clip_ids(user_id, dataset_id) -> list:
    """Which clips of this dataset currently play a RENDER (their original is in
    the backup folder). Derived from disk, not stored: the backup IS the state."""
    ds, rows = _dataset_and_rows(user_id, dataset_id)
    if ds is None:
        return []
    root = backup_dir(ds.id)
    return [r.id for r in rows if (root / r.filename).is_file()]


def start_dataset_render(app, user_id, dataset_id, clip_ids, params) -> dict:
    """Queue the in-place render of ``clip_ids`` (all clips when empty). One job
    per dataset, like every bank pass; the snapshot is read from dataset_job."""
    params = normalize_params(params)
    st = status()
    if not st['ready']:
        raise NeuralRenderError('neural rendering is not set up: ' + '; '.join(st['missing']))
    ds, rows = _dataset_and_rows(user_id, dataset_id, clip_ids)
    if ds is None:
        raise NeuralRenderError('dataset not found')
    if not rows:
        raise NeuralRenderError('no clip to render')
    targets = [(r.id, r.filename) for r in rows]
    out_dir = ds.output_dir
    dataset_ident = ds.id

    def _run(job):
        done = 0
        failed = []
        current = {'proc_cancel': False}
        bank_jobs.set_cancel_hook(job, lambda: current.__setitem__('proc_cancel', True))
        for clip_id, filename in targets:
            if bank_jobs.cancelled(job):
                break
            bank_jobs.progress(job, done=done, total=len(targets), detail=f'rendering {filename}')
            try:
                _render_one_in_place(dataset_ident, out_dir, filename, params,
                                     cancel=lambda: bank_jobs.cancelled(job) or current['proc_cancel'])
            except NeuralRenderError as exc:
                if str(exc) == 'cancelled':
                    break
                failed.append(f'{filename}: {exc}')
                logger.warning('neural render: dataset %s %s failed: %s', dataset_ident, filename, exc)
            done += 1
            bank_jobs.progress(job, done=done, total=len(targets))
        if failed:
            bank_jobs.fail(job, f'{len(failed)} of {len(targets)} clips failed — first: {failed[0]}')

    bank_jobs.start(app, job_key(dataset_ident), JOB_KIND, _run, total=len(targets))
    return {'queued': len(targets), 'params': params}


def render_record(params, result=None) -> dict:
    """What a render is remembered by: the dials asked for, plus — once the
    child answered — the frame mode actually used and the cost per frame.
    `temporal` stays the request ('auto'); `temporal_used` is the fact."""
    rec = {k: params[k] for k in ('tone', 'structure', 'automask', 'temporal',
                                   'strength', 'passes', 'scale') if k in params}
    if result:
        rec['temporal_used'] = bool(result.get('temporal'))
        if result.get('mean_ms') is not None:
            rec['ms_per_frame'] = round(float(result['mean_ms']), 1)
        if result.get('frames') is not None:
            rec['frames'] = int(result['frames'])
    return rec


def sidecar_path(dataset_id, filename) -> Path:
    """Where a dataset clip's render record lives: next to its kept original,
    outside the dataset folder (a trainer must never find a .json there)."""
    return backup_dir(dataset_id) / (os.path.basename(str(filename)) + '.nr.json')


def rendered_clip_params(user_id, dataset_id) -> dict:
    """{clip id: render record} for every clip of the dataset that plays a
    render and whose record survived. Read from disk like rendered_clip_ids —
    the backup folder IS the state."""
    ds, rows = _dataset_and_rows(user_id, dataset_id)
    if ds is None:
        return {}
    out = {}
    for r in rows:
        path = sidecar_path(ds.id, r.filename)
        if not (backup_dir(ds.id) / r.filename).is_file() or not path.is_file():
            continue
        try:
            out[r.id] = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
    return out


def _render_one_in_place(dataset_id, out_dir, filename, params, cancel=None) -> dict:
    """Render ONE dataset clip over itself.

    The ORIGINAL is the source, always: the first render copies the clip into
    the backup folder (write-once — an existing backup is never overwritten),
    and every render reads from there. So a clip rendered twice with different
    dials is two renders of the original, never a render of a render. The
    output lands next to the clip under a temporary name and replaces it in
    one ``os.replace``: the folder never holds a half-written .mp4 under the
    clip's name, and a trainer walking it mid-render sees the old file or the
    new one, never a truncated one."""
    clip_path = os.path.join(out_dir, filename)
    if not os.path.isfile(clip_path):
        raise NeuralRenderError('the clip is no longer on disk')
    root = backup_dir(dataset_id)
    root.mkdir(parents=True, exist_ok=True)
    backup = root / filename
    if not backup.is_file():
        shutil.copy2(clip_path, backup)
    fd, tmp_out = tempfile.mkstemp(prefix='.nr-', suffix='.mp4.part', dir=out_dir)
    os.close(fd)
    try:
        result = render_video(str(backup), tmp_out, params, cancel=cancel)
        os.replace(tmp_out, clip_path)
        try:
            sidecar_path(dataset_id, filename).write_text(
                json.dumps(render_record(params, result)), encoding='utf-8')
        except OSError:
            logger.warning('neural render: could not record the dials of %s', filename)
    finally:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
    return result


def forget_backups(dataset_id, filenames=None) -> int:
    """Drop the kept originals of clips that left the dataset (every backup when
    ``filenames`` is None — the dataset itself is gone). A backup without its
    clip can never be restored: a re-promotion creates a NEW dataset id, so
    the folder would only ever grow. Called by the routes that remove clips
    and delete datasets, after the removal succeeded."""
    root = backup_dir(dataset_id)
    if not root.is_dir():
        return 0
    if filenames is None:
        shutil.rmtree(root, ignore_errors=True)
        return 1
    dropped = 0
    for name in filenames:
        path = root / os.path.basename(str(name))
        if path.is_file():
            try:
                path.unlink()
                dropped += 1
            except OSError:
                pass
        record = root / (os.path.basename(str(name)) + '.nr.json')
        if record.is_file():
            try:
                record.unlink()
            except OSError:
                pass
    try:
        root.rmdir()          # only when empty
    except OSError:
        pass
    return dropped


def original_clip_path(user_id, dataset_id, clip_id) -> str | None:
    """The kept ORIGINAL of a rendered dataset clip, for the side-by-side
    player — None when the clip is unknown or plays no render (no backup).
    The filename comes from the row, never from the request, and the backup
    folder is the app's own: containment is by construction."""
    ds, rows = _dataset_and_rows(user_id, dataset_id, [clip_id])
    if ds is None or not rows:
        return None
    path = backup_dir(ds.id) / os.path.basename(rows[0].filename)
    return str(path) if path.is_file() else None


def restore_dataset_clips(user_id, dataset_id, clip_ids=None) -> dict:
    """🩹 Put the originals back (all rendered clips when ``clip_ids`` is empty).
    The backup is MOVED over the clip, so a restored clip has no backup and is
    reported as not rendered — the two facts cannot drift apart."""
    ds, rows = _dataset_and_rows(user_id, dataset_id, clip_ids)
    if ds is None:
        raise NeuralRenderError('dataset not found')
    if bank_jobs.running(job_key(ds.id)):
        raise NeuralRenderError('a render is running on this dataset — stop it first')
    root = backup_dir(ds.id)
    restored = 0
    for row in rows:
        backup = root / row.filename
        if not backup.is_file():
            continue
        clip_path = os.path.join(ds.output_dir, row.filename)
        try:
            os.replace(backup, clip_path)
            restored += 1
            sidecar = sidecar_path(ds.id, row.filename)
            if sidecar.is_file():
                sidecar.unlink()
        except OSError as exc:
            logger.warning('neural render: restore of %s refused: %s', row.filename, exc)
    return {'restored': restored}


# ── Studio clips: a new row ─────────────────────────────────────────────────

def start_studio_render(app, user_id, clip_id, params) -> dict:
    """✨ Render a finished studio clip as a NEW clip (never an edit — the
    studio exists to compare). The row is written first, in ``pending``; a
    daemon thread renders and flips it to ``done`` or ``failed`` with the
    child's own sentence in ``error``."""
    from ..extensions import db
    from ..models import VideoTestClip
    from . import video_test_studio as vts

    params = normalize_params(params)
    st = status()
    if not st['ready']:
        raise NeuralRenderError('neural rendering is not set up: ' + '; '.join(st['missing']))
    src = VideoTestClip.query.filter_by(id=int(clip_id)).first()
    if src is None:
        raise NeuralRenderError('clip not found')
    if src.status != 'done' or not src.filename:
        raise NeuralRenderError('that clip has not finished rendering yet')
    src_path = os.path.join(str(vts.clips_dir()), os.path.basename(src.filename))
    if not os.path.isfile(src_path):
        raise NeuralRenderError('that clip is no longer on disk')
    live = _STUDIO_THREADS.get(src.id)
    if live is not None and live.is_alive():
        raise NeuralRenderError('that clip is already being rendered')

    out_name = f'{vts.new_prefix(user_id)}_nr_{uuid.uuid4().hex[:6]}.mp4'
    clip = VideoTestClip(
        run_id=src.run_id, dataset_id=src.dataset_id, job_id=None,
        status='pending', prompt=src.prompt, mode=src.mode,
        source_image=src.source_image, seed=src.seed, steps=src.steps,
        frames=src.frames, megapixels=src.megapixels, fps=src.fps,
        base_model=src.base_model, lora=src.lora, lora_strength=src.lora_strength,
        turbo=bool(src.turbo), sparse=src.sparse, latent_upscale=bool(src.latent_upscale),
        vfi_of=src.vfi_of, nr_of=src.id, nr_params=json.dumps(params))
    db.session.add(clip)
    db.session.commit()
    new_id = clip.id
    dst_path = os.path.join(str(vts.clips_dir()), out_name)

    def _run():
        with app.app_context():
            row = VideoTestClip.query.filter_by(id=new_id).first()
            # ⏱ This lane never goes through the queue, so nothing stamps it:
            # measured here, so a rendered clip carries a time its source can
            # be compared against — the very use of ⇔ Compare. Monotonic: a
            # local duration, not two database stamps.
            t0 = time.monotonic()
            try:
                result = render_video(src_path, dst_path, params)
                if row is not None:
                    row.filename = out_name
                    row.status = 'done'
                    row.error = None
                    row.nr_params = json.dumps(render_record(params, result))
                    logger.info('neural render: studio clip %s -> %s (%s, %s frames, %.1f ms/frame)',
                                src.id, new_id, result.get('mode_note'), result.get('frames'),
                                result.get('mean_ms') or 0)
            except NeuralRenderError as exc:
                if row is not None:
                    row.status = 'failed'
                    row.error = str(exc)
            except Exception as exc:  # noqa: BLE001 — the row must never stay pending
                logger.exception('neural render: studio clip %s crashed', new_id)
                if row is not None:
                    row.status = 'failed'
                    row.error = f'unexpected failure: {exc}'
            if row is not None:
                row.render_seconds = round(time.monotonic() - t0, 1)
            db.session.commit()

    thread = threading.Thread(target=_run, name=f'neural-render-{new_id}', daemon=True)
    _STUDIO_THREADS[src.id] = thread
    thread.start()
    return {'clip_id': new_id, 'params': params}
