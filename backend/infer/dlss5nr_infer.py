"""✨ Neural render — ONE video through NVIDIA's DLSS 5 Neural Rendering model.

THE CHILD. This script runs in its own process, launched by
``app/services/neural_render.py``, and never inside the Flask process — two
reasons, both measured before this file existed:

* the bridge's ``dlss5nr_shutdown()`` hangs after a render (five minutes, until
  the caller gave up), so a process that owns a D3D12 device can only be
  finished by exiting it — hence ``os._exit`` at the bottom, never a tidy
  shutdown;
* a hang in a native DLL is not something a Python thread can be rescued from,
  and the parent must stay able to cancel a render: killing a process works,
  interrupting a stuck ctypes call does not.

Dependencies: numpy and an ffmpeg binary. Nothing else — no torch, no cv2, no
Flask. The interpreter is whichever one the parent hands it (the video extra's,
which carries numpy), see ``neural_render.worker_python``.

WHAT IS FED TO THE MODEL, AND WHY THAT IS ENOUGH
The model (NGX feature 18, ``nvngx_dlssnr.dll``) evaluates with a colour frame
and an output surface; depth and motion vectors are optional — the model logs
"Missing Color or Output parameter" and nothing about the others. The bridge
(``dlss5nr_bridge.dll``, MIT, github.com/lisitskyaa/ComfyUI-DLSS5-NR) never
binds depth; in temporal mode it estimates motion vectors with NVIDIA's Optical
Flow engine from the raw frames. ``still`` mode resets the history on every
frame and needs nothing but the frame.

TEMPORAL MODE HAS A WIDTH FLOOR. The Optical Flow session refuses frames
narrower than 704 px (bisected: 700 fails, 704 passes, height is irrelevant —
1080×1920 passes, 576×1024 fails). The parent decides the mode; this script only
reports the failure honestly if asked for temporal below that floor.

PROTOCOL. One JSON object per stdout line: ``init`` (bridge/GPU/NVOF),
``frame`` every few frames (index, count, milliseconds), ``done`` (stats) or
``error`` (message). Exit code 0 on ``done``, 1 on ``error``. ffmpeg's stderr
is captured into the failure message; it never shares our stdout.

Credits: scene-cut detection (mean absolute difference of the downscaled grey
frame, history reset above a threshold) follows ``TemporalGuideGenerator`` in
Merserk's dlss5-visual-enhancer (MIT, github.com/Merserk/dlss5-visual-enhancer),
and so does the "encode video only, then mux audio and metadata back from the
source" shape of the output.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

# NVIDIA's Optical Flow session refuses anything narrower than this; measured by
# bisection on the bridge (see the module docstring). Kept here AND in
# neural_render.TEMPORAL_MIN_WIDTH — the parent gates, the child re-checks, and a
# contract test pins both to the same number.
TEMPORAL_MIN_WIDTH = 704
# Scene-cut threshold on the mean absolute difference of the downscaled grey
# frame (0..1): above it the temporal history is reset so the model does not
# smear the previous shot into this one. Merserk's 0.24 was measured on a
# different grey (full-res, RGBA): on THIS thumbnail a real cut between two
# witness clips scored 0.12 and the busiest single-shot frame pair 0.072, so
# 0.24 fired on nothing. 0.10 sits between the two — one witness pair, so it is
# a dial (`--scene-cut`), not a law. A cut missed smears a few frames; a reset
# fired for nothing costs one frame of history.
SCENE_CUT_DEFAULT = 0.10
FLOW_WIDTH = 160   # the grey thumbnail used for the cut test (width; height follows)


def pump_stderr(stream, sink, limit=40):
    """Drain a child's stderr on its own thread, keeping the last ``limit``
    lines. NOT optional: a pipe nobody reads fills at 4 KB on Windows and the
    child then blocks on its next write — ffmpeg's encoder does exactly that
    after ~35 s of libx264 at crf 17 (measured), and the render hangs with a
    truncated file on disk. ``shot_detect.py`` drains the same way."""
    def _run():
        try:
            for raw in iter(stream.readline, b''):
                line = raw.decode('utf-8', 'replace').rstrip()
                if line:
                    sink.append(line)
                    if len(sink) > limit:
                        del sink[:-limit]
        except (OSError, ValueError):
            pass
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def emit(event, **fields):
    sys.stdout.write(json.dumps({'event': event, **fields}) + '\n')
    sys.stdout.flush()


def fail(message, code=1):
    emit('error', message=str(message))
    sys.stdout.flush()
    os._exit(code)


def _probe(ffprobe, path):
    out = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries',
         'stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames',
         '-show_entries', 'format=duration', '-of', 'json', path],
        capture_output=True, text=True, encoding='utf-8', errors='replace')
    if out.returncode != 0:
        raise RuntimeError(f'ffprobe failed: {out.stderr.strip()[-400:]}')
    data = json.loads(out.stdout or '{}')
    video = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
    if video is None:
        raise RuntimeError('no video stream in the source')
    audio = any(s.get('codec_type') == 'audio' for s in data.get('streams', []))
    rate = video.get('avg_frame_rate') or video.get('r_frame_rate') or '24/1'
    if rate in ('0/0', '0'):
        rate = video.get('r_frame_rate') or '24/1'
    try:
        frames = int(video.get('nb_frames') or 0)
    except (TypeError, ValueError):
        frames = 0
    return {'width': int(video['width']), 'height': int(video['height']),
            'rate': rate, 'frames': frames, 'audio': audio,
            'duration': float((data.get('format') or {}).get('duration') or 0)}


def _load_bridge(runtime):
    bridge = os.path.join(runtime, 'dlss5nr_bridge.dll')
    caller = os.path.join(runtime, 'caller')
    for d in (runtime, caller):
        if os.path.isdir(d):
            os.add_dll_directory(d)
    lib = ctypes.WinDLL(bridge)
    F = ctypes.POINTER(ctypes.c_float)
    lib.dlss5nr_init.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_char_p, ctypes.c_int]
    lib.dlss5nr_init.restype = ctypes.c_int
    lib.dlss5nr_process.argtypes = [
        F, F, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,                       # style, preset
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,  # intensity, tone, structure, skin
        ctypes.c_int, ctypes.c_int, ctypes.c_int,         # automask, reset, temporal
        ctypes.c_char_p, ctypes.c_int]
    lib.dlss5nr_process.restype = ctypes.c_int
    for name in ('dlss5nr_version', 'dlss5nr_gpu_name'):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_char_p
    for name in ('dlss5nr_nvof_available', 'dlss5nr_nvof_grid', 'dlss5nr_nvof_perf'):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_int
    return lib


def _channel_order(np, fin, fout):
    """RGB or BGR? The stock model has been seen answering in BGRA order and the
    Ada-patched builds in RGBA. Decided ONCE, on the first frame, by which
    reading keeps the low-frequency colour closest to the source — the model
    relights and retextures, it never swaps red and blue globally."""
    h, w = fin.shape[:2]
    sy, sx = max(1, h // 128), max(1, w // 128)
    ref = fin[::sy, ::sx]
    raw = fout[::sy, ::sx]
    swp = raw[..., ::-1]
    score = lambda a: float(np.abs(a - ref).mean() + np.abs(a.mean((0, 1)) - ref.mean((0, 1))).mean())  # noqa: E731
    return 'rgb' if score(raw) <= score(swp) else 'bgr'


def decode_filter(src_w, src_h, rw, rh):
    """The decoder's -vf: Lanczos to the model's working size when it differs
    from the source (an odd edge, or a 2x render); 'null' otherwise."""
    return f'scale={rw}:{rh}:flags=lanczos' if (rw, rh) != (src_w, src_h) else 'null'


def encode_filter(rw, rh, w, h):
    """The encoder's -vf: back to the file's size after a 2x render, so a
    dataset keeps the size its target profile gave every clip."""
    return ['-vf', f'scale={w}:{h}:flags=lanczos'] if (rw, rh) != (w, h) else []


def apply_strength(np, fin, fout, strength):
    """out = in + k * (model - in). k = 1 is the model's picture; above it
    carries on past it (what the game mod's 'Detail strength' does, and why the
    game looked transformed at 1.43); below 1 fades it towards the input."""
    if abs(strength - 1.0) < 1e-6:
        return fout
    return np.clip(fin + np.float32(strength) * (fout - fin), 0.0, 1.0)


def _grey_thumb(np, frame_u8, w, h):
    tw = min(FLOW_WIDTH, w)
    sx = max(1, w // tw)
    sy = sx
    small = frame_u8[::sy, ::sx]
    return small.astype(np.float32).mean(axis=2) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--runtime', required=True)
    ap.add_argument('--ffmpeg', required=True)
    ap.add_argument('--tone', type=float, default=1.0)
    ap.add_argument('--structure', type=float, default=1.0)
    ap.add_argument('--automask', type=int, default=0)
    ap.add_argument('--temporal', choices=('on', 'off'), default='off')
    ap.add_argument('--scene-cut', type=float, default=SCENE_CUT_DEFAULT)
    ap.add_argument('--crf', type=int, default=17)
    # The three levers the model does not expose but the eye needs — measured
    # on a photoreal frame: the default adds ~7 % fine detail, strength x2
    # ~17 %, x3 ~30 %, two passes ~12 %, rendering at 2x ~12 % at 1:1.
    ap.add_argument('--strength', type=float, default=1.0)   # out = in + k * (model - in); 1 = the model's answer
    ap.add_argument('--passes', type=int, default=1)         # feed the answer back through the model
    ap.add_argument('--scale', type=int, default=1)          # render at 2x, deliver at the clip's size
    ap.add_argument('--gpu-index', type=int, default=0)
    ap.add_argument('--progress-every', type=int, default=6)
    args = ap.parse_args()

    if os.name != 'nt':
        fail('DLSS 5 Neural Rendering needs Windows (a D3D12 model)')
    try:
        import numpy as np
    except ImportError as exc:
        fail(f'numpy is missing in this interpreter: {exc}')

    ffmpeg = args.ffmpeg
    ffprobe = os.path.join(os.path.dirname(ffmpeg), 'ffprobe' + ('.exe' if ffmpeg.lower().endswith('.exe') else ''))
    if not os.path.isfile(ffprobe):
        ffprobe = shutil.which('ffprobe') or ffprobe
    try:
        meta = _probe(ffprobe, args.src)
    except Exception as exc:  # noqa: BLE001 — every refusal is reported the same way
        fail(exc)

    # Even dimensions: yuv420p cannot hold an odd edge, and the model does not
    # care either way, so the decode pads nothing and scales by at most a pixel.
    w = meta['width'] - meta['width'] % 2
    h = meta['height'] - meta['height'] % 2
    scale = 2 if int(args.scale) >= 2 else 1
    passes = max(1, min(3, int(args.passes)))
    strength = max(0.0, min(3.0, float(args.strength)))
    # The model works on rw x rh (2x when asked); the file keeps w x h — a
    # dataset's clips must all keep the size the target profile gave them.
    rw, rh = w * scale, h * scale
    temporal = args.temporal == 'on'
    if passes > 1 and temporal:
        # A second pass feeds the model its own answer; the history the
        # temporal contract keeps would then describe the wrong frame, and
        # switching contracts per frame rebuilds the feature every time.
        fail('extra passes run in still mode — set temporal off (or auto)')
    if temporal and w < TEMPORAL_MIN_WIDTH:
        fail(f'temporal mode needs a frame at least {TEMPORAL_MIN_WIDTH} px wide '
             f'(this one is {w}) — render it in still mode instead')

    try:
        lib = _load_bridge(args.runtime)
    except OSError as exc:
        fail(f'the neural rendering bridge could not be loaded: {exc}')
    err = ctypes.create_string_buffer(4096)
    t0 = time.perf_counter()
    if not lib.dlss5nr_init(int(args.gpu_index), args.runtime, err, len(err)):
        fail(err.value.decode('utf-8', 'replace') or 'the neural rendering runtime refused to start')
    nvof = bool(lib.dlss5nr_nvof_available())
    emit('init', bridge=lib.dlss5nr_version().decode('utf-8', 'replace'),
         gpu=lib.dlss5nr_gpu_name().decode('utf-8', 'replace'), nvof=nvof,
         init_ms=round((time.perf_counter() - t0) * 1000), width=w, height=h,
         frames=meta['frames'], temporal=temporal)
    if temporal and not nvof:
        fail('temporal mode needs NVIDIA Optical Flow (nvofapi64.dll from the display driver)')

    F = ctypes.POINTER(ctypes.c_float)
    nbytes = rw * rh * 3
    vf = decode_filter(meta['width'], meta['height'], rw, rh)
    dec = subprocess.Popen(
        [ffmpeg, '-v', 'error', '-i', args.src, '-vf', vf, '-f', 'rawvideo',
         '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tmp_dir = tempfile.mkdtemp(prefix='lds-nr-')
    video_only = os.path.join(tmp_dir, 'video.mp4')
    enc = subprocess.Popen(
        [ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{rw}x{rh}', '-r', meta['rate'], '-i', '-',
         *encode_filter(rw, rh, w, h),
         '-c:v', 'libx264', '-preset', 'medium', '-crf', str(int(args.crf)),
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart', video_only],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    dec_log, enc_log = [], []
    dec_pump = pump_stderr(dec.stderr, dec_log)
    enc_pump = pump_stderr(enc.stderr, enc_log)

    order = None
    prev_grey = None
    done = 0
    resets = 0
    ms_total = 0.0
    ms_max = 0.0
    try:
        while True:
            buf = dec.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            u8 = np.frombuffer(buf, np.uint8).reshape(rh, rw, 3)
            fin = np.ascontiguousarray(u8.astype(np.float32) / 255.0)
            fout = np.empty_like(fin)
            reset = 1
            if temporal:
                grey = _grey_thumb(np, u8, rw, rh)
                if prev_grey is not None and prev_grey.shape == grey.shape:
                    cut = float(np.abs(grey - prev_grey).mean())
                    reset = 1 if cut > args.scene_cut else 0
                    resets += reset
                prev_grey = grey
            e = ctypes.create_string_buffer(4096)
            t1 = time.perf_counter()
            ok = lib.dlss5nr_process(
                fin.ctypes.data_as(F), fout.ctypes.data_as(F), rw, rh,
                1, 3,                                   # style, preset: inert, kept at the bridge's defaults
                ctypes.c_float(1.0), ctypes.c_float(float(args.tone)),
                ctypes.c_float(float(args.structure)), ctypes.c_float(-1.0),
                1 if args.automask else 0, reset, 1 if temporal else 0, e, len(e))
            ms = (time.perf_counter() - t1) * 1000
            if not ok:
                raise RuntimeError(f'frame {done}: ' + (e.value.decode('utf-8', 'replace') or 'the model refused the frame'))
            if order is None:
                order = _channel_order(np, fin, fout)
            if order == 'bgr':
                fout = np.ascontiguousarray(fout[..., ::-1])
            for _ in range(passes - 1):
                again = np.empty_like(fout)
                e2 = ctypes.create_string_buffer(4096)
                t2 = time.perf_counter()
                ok = lib.dlss5nr_process(
                    fout.ctypes.data_as(F), again.ctypes.data_as(F), rw, rh,
                    1, 3, ctypes.c_float(1.0), ctypes.c_float(float(args.tone)),
                    ctypes.c_float(float(args.structure)), ctypes.c_float(-1.0),
                    1 if args.automask else 0, 1, 0, e2, len(e2))
                ms += (time.perf_counter() - t2) * 1000
                if not ok:
                    raise RuntimeError(f'frame {done}, extra pass: ' + (e2.value.decode('utf-8', 'replace') or 'the model refused the frame'))
                fout = np.ascontiguousarray(again[..., ::-1]) if order == 'bgr' else again
            fout = apply_strength(np, fin, fout, strength)
            out_u8 = (np.clip(fout, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            enc.stdin.write(out_u8.tobytes())
            done += 1
            ms_total += ms
            ms_max = max(ms_max, ms)
            if done % max(1, args.progress_every) == 0:
                emit('frame', done=done, total=meta['frames'] or None, ms=round(ms, 1))
    except Exception as exc:  # noqa: BLE001
        for p in (dec, enc):
            try:
                p.kill()
            except OSError:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        fail(exc)
    finally:
        try:
            dec.stdout.close()
        except OSError:
            pass
        try:
            enc.stdin.close()
        except OSError:
            pass

    dec.wait()
    enc.wait()
    dec_pump.join(timeout=5)
    enc_pump.join(timeout=5)
    dec_err = '\n'.join(dec_log).strip()
    enc_err = '\n'.join(enc_log).strip()
    if done == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        fail('no frame could be decoded from the source' + (f': {dec_err[-300:]}' if dec_err else ''))
    if enc.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        fail(f'the encoder failed: {enc_err[-400:]}')

    # Mux the source's audio and metadata back around the rendered video: the
    # render touched pixels and nothing else. Copied streams only — no audio
    # re-encode, so a silent clip stays silent and a track keeps its codec.
    mux = subprocess.run(
        [ffmpeg, '-v', 'error', '-y', '-i', video_only, '-i', args.src,
         '-map', '0:v:0', '-map', '1:a?', '-c', 'copy', '-map_metadata', '1',
         '-movflags', '+faststart', '-shortest', args.dst],
        capture_output=True, text=True, encoding='utf-8', errors='replace')
    if mux.returncode != 0 and meta['audio']:
        # An audio track the container cannot carry as-is: keep the pictures.
        mux = subprocess.run(
            [ffmpeg, '-v', 'error', '-y', '-i', video_only, '-map', '0:v:0',
             '-c', 'copy', '-movflags', '+faststart', args.dst],
            capture_output=True, text=True, encoding='utf-8', errors='replace')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if mux.returncode != 0:
        fail(f'writing the output failed: {mux.stderr.strip()[-400:]}')

    emit('done', frames=done, width=w, height=h, temporal=temporal, resets=resets,
         strength=strength, passes=passes, scale=scale,
         channel_order=order, mean_ms=round(ms_total / done, 1), max_ms=round(ms_max, 1),
         audio=bool(meta['audio']))
    sys.stdout.flush()
    # The bridge's shutdown hangs after a render (measured); the process is the
    # unit of cleanup here, so leave without calling it.
    os._exit(0)


if __name__ == '__main__':
    main()
