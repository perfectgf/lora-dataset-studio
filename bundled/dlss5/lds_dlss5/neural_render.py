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
Never in this process. ``infer/dlss5nr_infer.py`` is launched per clip
(see its docstring for the two hangs that decided this) and reports progress
as JSON lines; cancelling a render kills and reaps that process even while a
native frame remains silent. The interpreter and encoder belong to DLSS's
managed environment or its explicit ``plugins.dlss5.python`` override.

THE SURFACES
* The standalone DLSS studio accepts imported video files, keeps each original
  and writes a separate result. It works without any other product installed.
* The following Video integrations are optional; their media/table adapters
  stay with Video while DLSS owns the engine, routes and finishing dialogs.
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
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from lds_sdk import config as cfg
from lds_sdk import workers as infer_env
from . import runtime as ffmpeg_tools
from lds_sdk.dlss5 import NeuralRenderError, ComparisonBusyError, ComparisonTooLargeError

logger = logging.getLogger(__name__)

# The lane's infer scripts live beside the package (bundled/video/infer/),
# not in the core's backend/infer/: resolved from this file, wherever the
# plugin is installed.
_INFER_DIR = Path(__file__).resolve().parents[1] / 'infer'

# ── The bridge release this app installs ────────────────────────────────────
# Pinned by tag AND content: GitHub lets an author re-upload an asset under the
# same tag, so the URL alone is not an identity. A mismatch is a refusal, never
# a warning.
BRIDGE_PROJECT_URL = 'https://github.com/lisitskyaa/ComfyUI-DLSS5-NR'
BRIDGE_RELEASE = {
    'version': '0.3.0',
    'url': ('https://lds.perfectgf.fr/models/tools/'
            '3b7d52507a5548d10c3f60f9a5ea4cc5eb2fd9bd715b536533df55887a1d907f/'
            'ComfyUI-DLSS5-NR-v0.3.0-windows-x64.zip'),
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


# ── Where things live ───────────────────────────────────────────────────────

def runtime_dir(create=False) -> Path:
    root = cfg.data_dir() / 'dlss5nr'
    if create:
        (root / 'caller').mkdir(parents=True, exist_ok=True)
    return root


def worker_python() -> str:
    return ffmpeg_tools.interpreter()


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
            worker_ok = ffmpeg_tools.worker_ready()
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
        missing.append('the DLSS 5 Python engine — prepare it in Plugins > DLSS 5 > Settings')
    # ffmpeg reads and writes the clip. Asked through the one definition of
    # 'the encoder works' (a RUN, cached), never a path check.
    if ffmpeg_ok is None:
        ffmpeg_ok = bool(ffmpeg_tools.ffmpeg_ready()['ok']) if os_ok else False
    if os_ok and not ffmpeg_ok:
        missing.append('the DLSS 5 video encoder — prepare the engine in Plugins > DLSS 5 > Settings')
    if not files['bridge'] or not files['shim']:
        missing.append('the neural rendering bridge — install it from Setup')
    if files['model_present_but_small']:
        missing.append(f'{MODEL_FILE} in {root_str} is not the model (a real one is about 165 MB)')
    elif not files['model']:
        missing.append(f'your own copy of {MODEL_FILE}, placed in {root_str}')
    return {
        'available': True,
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


# ── ⇔ The comparison, as ONE file ───────────────────────────────────────────
# The side-by-side player answers "did this do anything?" on screen and nowhere
# else. A file answers it everywhere: a post, a message, a note kept beside the
# clip. So this encodes the picture the player shows — the two clips in one
# frame, in step by construction, one timeline instead of two players.

# `shortest=1` belongs to the FILTER, not to the muxer — see comparison_argv.
HSTACK = 'hstack=inputs=2:shortest=1'
COMPARISON_CRF = 18            # near-transparent: the point of this file is detail
COMPARISON_MAX_BYTES = 512 * 1024 * 1024
COMPARISON_TIMEOUT_S = 900

# ONE encode at a time, and the second caller is told to come back rather than
# queued. Measured on this app's own clips: a 2816×800 pair of 485 frames takes
# 2.1 s alone and 17.4 s when six of them run at once — six ffmpeg processes do
# not share a machine, they divide it, and the RAM each finished file sits in
# multiplies by the same number. Same shape as the timeline GIF next door
# (checkpoint_timeline._GIF_RENDER_GATE), for the same reason.
_COMPARISON_GATE = threading.BoundedSemaphore(1)
FONT_CANDIDATES = (
    r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\segoeui.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/TTF/DejaVuSans.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
)


def comparison_font() -> str | None:
    """The first usable font file, or None when the labels have to be dropped.

    This ``isfile`` is what holds the promise, and it cannot be replaced by
    trusting ffmpeg: measured, a ``fontfile=`` pointing at a path that does not
    exist does NOT fail — both resolvable builds here are ``--enable-fontconfig``
    and drawtext quietly falls back to a substitute face, exit 0, pixels drawn.
    So a stale entry in the list below would silently change the typeface rather
    than drop the label, and an exit code can never be read as "my font was
    used".
    """
    return next((f for f in FONT_CANDIDATES if os.path.isfile(f)), None)


def graph_value(path) -> str:
    """A path as a filtergraph VALUE.

    Three parsers read this string — the graph, the filter description, then the
    option — and each eats one level of escaping, so a Windows drive colon needs
    TWO backslashes to reach drawtext. Measured on the bundled ffmpeg 7.1:
    ``C\\\\:/Windows/…`` parses, while ``C\\:/Windows/…`` and ``C\\:\\\\Windows\\\\…``
    both die with "No option name near". Backslashes become forward slashes for
    the same reason; a path with no colon comes back unchanged.
    """
    return str(path).replace('\\', '/').replace(':', r'\\:')


def label_filter(text, font) -> str:
    """One caption, centred at the bottom of its pane, on a box so it stays
    readable over a white frame. The text is stripped of the three characters
    that would end the option rather than escaped: these labels are ours, not
    the user's, and a caption is not worth an escaping bug."""
    safe = str(text).replace('\\', '').replace("'", '').replace(':', ' ')
    return (f"drawtext=fontfile={graph_value(font)}:text='{safe}':fontcolor=white:"
            'fontsize=h/22:box=1:boxcolor=black@0.55:boxborderw=10:'
            'x=(w-text_w)/2:y=h-text_h-24')


def comparison_argv(left, right, out, *, left_label, right_label,
                    font=None, ffmpeg=None) -> list:
    """The one command that builds the side-by-side file.

    ``-map_metadata -1`` is not tidiness. A studio clip carries ComfyUI's ENTIRE
    workflow in its ``comment`` tag — every prompt and every absolute path,
    ``C:\\Users\\<name>\\…`` included — and ffmpeg copies that to the output by
    default. This file exists to be handed to other people, so it starts with no
    metadata at all. (Measured: the tag was there, in full, before this flag was.)

    ``-map 0:a?`` keeps the left clip's sound when it has one and asks for
    nothing when it does not.

    The pair ends with the SHORTER clip, and that takes ``shortest=1`` on the
    filter — ``-shortest`` alone does not do it. Measured on a deliberate
    mismatch (5.17 s against 2 s): the output ran the full 5.17 s with the short
    side FROZEN on its last frame, because ``-shortest`` is a muxer option and
    ``hstack`` had already padded the short input long before the muxer saw it
    (frame-to-frame delta on the right pane: 1.49 while it plays, 0.009 after).
    ``-shortest`` is kept beside it for the audio stream.
    """
    ff = str(ffmpeg or ffmpeg_tools.ffmpeg_path() or 'ffmpeg')
    if font:
        graph = (f'[0:v]{label_filter(left_label, font)}[l];'
                 f'[1:v]{label_filter(right_label, font)}[r];'
                 f'[l][r]{HSTACK}[v]')
    else:
        graph = f'[0:v][1:v]{HSTACK}[v]'
    return [ff, '-y', '-hide_banner', '-i', str(left), '-i', str(right),
            '-filter_complex', graph, '-map', '[v]', '-map', '0:a?',
            '-map_metadata', '-1',
            '-c:v', 'libx264', '-crf', str(COMPARISON_CRF), '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-shortest', str(out)]


def build_comparison(left, right, *, left_label, right_label, ffmpeg=None,
                     timeout_s=None) -> bytes:
    """The side-by-side clip's BYTES.

    Built into a temp file — an mp4 with ``+faststart`` has to be seekable, so a
    pipe is not an option — then read back and deleted here. The bytes are what
    the route sends, and the temp file is gone before this returns: measured on a
    threaded Flask, ``send_file(path)`` plus ``call_on_close`` to delete it never
    fires on a SUCCESSFUL download (eight requests, eight orphans, 77 MB), which
    is the trap ``routes/training.py`` already documents — "some servers close
    that wrapper without invoking Response.call_on_close". Deleting the file
    inside the view is not the alternative either: Windows refuses to unlink a
    file the response still holds open (PermissionError 13). Reading it and
    dropping it here is the shape that leaves nothing behind, on either OS.
    """
    for path, side in ((left, 'left'), (right, 'right')):
        if not path or not os.path.isfile(str(path)):
            raise NeuralRenderError(f'the {side} clip is not on disk any more')
    if not ffmpeg_tools.ffmpeg_path() and not ffmpeg:
        raise NeuralRenderError('ffmpeg is needed to build the comparison — '
                                'prepare the DLSS 5 engine in its plugin settings')
    if not _COMPARISON_GATE.acquire(blocking=False):
        raise ComparisonBusyError('another comparison is being built — try again '
                                  'in a moment')
    # EVERYTHING after the acquire is inside the try, mkdtemp included. It was
    # outside for one commit, and a temp dir that cannot be made — a full disk,
    # a %TEMP% that is gone or read-only — walked out with the slot still held:
    # every later export answered 429 until the server was restarted. The
    # release belongs to a `finally` that starts at the acquire, not at the
    # first line that happens to look risky.
    out = None
    try:
        out = Path(tempfile.mkdtemp(prefix='lds-compare-'))
        dst = out / 'comparison.mp4'
        argv = comparison_argv(left, right, dst, left_label=left_label,
                               right_label=right_label, font=comparison_font(),
                               ffmpeg=ffmpeg)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8',
                                  errors='replace',
                                  timeout=timeout_s or COMPARISON_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise NeuralRenderError('building the comparison took too long and was stopped')
        except OSError as exc:
            # The path resolved and the binary is not there: the video extra was
            # removed, or an ffmpeg on PATH moved. `ffmpeg_path()` answers with a
            # directory entry, never with a working encoder — so this is a
            # sentence, not a 500 out of subprocess.
            raise NeuralRenderError(
                'ffmpeg could not be run — reprepare the DLSS 5 engine in its plugin settings '
                f'({exc.__class__.__name__})')
        if proc.returncode != 0 or not dst.is_file():
            tail = (proc.stderr or '').strip().splitlines()[-1:] or ['ffmpeg failed']
            raise NeuralRenderError(f'the comparison could not be built: {tail[0]}')
        size = dst.stat().st_size
        if size > COMPARISON_MAX_BYTES:
            raise ComparisonTooLargeError(
                f'the comparison came out at {size // (1024 * 1024)} MB, past the '
                f'{COMPARISON_MAX_BYTES // (1024 * 1024)} MB this route will hold '
                'in memory — compare a shorter clip')
        return dst.read_bytes()
    finally:
        if out is not None:
            shutil.rmtree(out, ignore_errors=True)
        _COMPARISON_GATE.release()


# ── The render itself: one child per clip ───────────────────────────────────

def worker_argv(src, dst, params, temporal_on, ffmpeg=None) -> list:
    script = str(_INFER_DIR / 'dlss5nr_infer.py')
    return infer_env.isolated_worker_argv(
        worker_python(), script,
        '--src', str(src), '--dst', str(dst), '--runtime', str(runtime_dir()),
        '--ffmpeg', str(ffmpeg or ffmpeg_tools.ffmpeg_path() or 'ffmpeg'),
        '--tone', str(params['tone']), '--structure', str(params['structure']),
        '--automask', '1' if params['automask'] else '0',
        '--temporal', 'on' if temporal_on else 'off',
        '--scene-cut', str(params['scene_cut']), '--crf', str(CRF_DEFAULT),
        '--strength', str(params.get('strength', 1.0)), '--passes', str(params.get('passes', 1)),
        '--scale', str(params.get('scale', 1)))


def _render_video(src, dst, params, *, on_progress=None, cancel=None, timeout_s=None) -> dict:
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
    env = infer_env.isolated_worker_env(worker_python(), PYTHONUTF8='1')
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
    stopped = threading.Event()
    stop_reason = []

    def watch():
        # stdout can remain silent inside a native frame. Cancellation must
        # not depend on the next JSON line arriving from that same frame.
        while not stopped.wait(0.1):
            reason = ('cancelled' if cancel is not None and cancel() else
                      'The render exceeded its time limit.' if timeout_s and
                      time.time() - started > timeout_s else None)
            if reason:
                stop_reason.append(reason)
                if proc.poll() is None:
                    proc.kill()
                return

    watchdog = threading.Thread(target=watch, name='dlss5-cancel', daemon=True)
    watchdog.start()
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
        stopped.set()
        watchdog.join(timeout=1)
        try:
            proc.stdout.close()
        except OSError:
            pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        drain.join(timeout=5)
    if stop_reason:
        raise NeuralRenderError(stop_reason[0])
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
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
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



def render_video(src, dst, params, **kwargs):
    """Own the worker and the shared GPU until the native child is reaped."""
    from lds_sdk.video_host.gpu import gpu_exclusive_vision_window, GpuBusyError
    try:
        with ffmpeg_tools.use_worker(), gpu_exclusive_vision_window():
            return _render_video(src, dst, params, **kwargs)
    except GpuBusyError as exc:
        raise NeuralRenderError(str(exc)) from exc
