"""TransNetV2 shot-boundary detection worker — the video lane's only consumer
of torch and `transnetv2-pytorch`.

WHY THIS WORKER DECODES ITSELF INSTEAD OF CALLING THE LIBRARY'S OWN
`predict_video()` / `detect_scenes()`. Those high-level methods shell out to an
`ffmpeg` COMMAND (via the `ffmpeg-python` wrapper the package depends on) and,
for their timestamps, separately call `ffmpeg.probe()` — which is `ffprobe`, a
second binary. Both assume a full ffmpeg toolchain sitting on PATH. This app
solved that exact problem once already, the hard way (see video_probe.py's
docstring on the video-bank branch): the bundled `imageio-ffmpeg` ships ffmpeg
WITHOUT ffprobe, so anything that shells out to either binary works on a
developer's machine — which has a full install on PATH — and breaks on
precisely the install the video extra exists for. So this worker decodes with
PyAV instead — already the video lane's chosen decode path — straight down to
the network's fixed 48x27 input geometry, and never touches the library's own
video/probe helpers. TransNetV2 itself is used only for what only it can do:
`predict_frames()` (windowed inference over an already-decoded frame tensor).
Its `predictions_to_scenes` transition rule is reimplemented below in plain
Python (a faithful port, not a rewrite) so it can be unit tested with no torch
import at all — the same discipline every seam in this file follows.

THE NETWORK IS NEVER THE BOTTLENECK, DECODING IS. TransNetV2 runs on 48x27
frames — a few KB each — so a decode-plus-resize is the entire per-frame cost
that matters here. This worker decodes each source video exactly ONCE: the
frames handed to the model are the same reformatted frames read from the
container in a single pass, never re-opened or re-scaled for a second purpose.

BOUNDARIES COMING OUT OF THIS FILE ARE STILL FRAME INDICES. Converting them to
PTS seconds is deliberately NOT this file's job: TransNetV2 reasons only in
indices of the frames it was handed, and scraped video is routinely
variable-frame-rate, where frame index n names no stable instant. That
conversion belongs to the parent (services/shot_detect.py), which has the
caller's measured `fps_native` — often already sitting on a VideoSource row
from an earlier probe. This worker still reports what IT measured from its own
decode pass (`fps_native` per video below), as the fallback the parent uses
when the caller has no cached value to hand it.

Weights ship INSIDE the `transnetv2-pytorch` wheel (~33 MB, MIT) — the
constructor below never downloads anything and works fully offline.

BOTH HEADS ARE REPORTED, NOT JUST THE ONE BOUNDARIES ARE DRAWN FROM. The
network predicts `single_frame_pred` (the centre frame of a transition) and
`all_frame_pred` (every frame the transition covers). This worker used to
compute the second and drop it on the floor. It is the only signal anywhere
that says how WIDE a transition is — hard cut or slow dissolve — and it costs
nothing extra, the forward pass having already produced it. Both vectors go
back to the parent when it asks (`emit_probs`), which persists them; from then
on a different threshold is a re-read of a small file rather than a second
pass over the GPU.

Protocol (same streaming shape as watermark_detect_infer.py — one line per
file, because a bank-wide pass must be able to report partial progress and
must not let one corrupt source cost the rest):
  stdin  : ONE json object
           {"videos": [abs paths], "threshold": 0.5, "device": "auto"|"cuda"|"cpu",
            "cancel_file": path|null, "emit_probs": bool}
  stdout : ONE json line PER VIDEO, in input order, holding exactly ROW_KEYS:
           {"path": str, "state": "ok"|"error",
            "shots": [[start_frame, end_frame], ...],
            "frame_count": int|null, "fps_native": float|null, "error": str|null,
            "probs": {"single": [float], "all": [float]}|null}
           then a final line {"summary": {"ok": bool, "processed": int,
            "errors": int, "device": str, "error"?: str}}.
  stderr : "[shotdet] i/N state" progress lines + load diagnostics.

A per-file failure is a row with state "error", never a dead pass: one corrupt
video among four hundred must cost that video and not the scan.
"""
from __future__ import annotations
import json
import os
import sys

# Persisted verbatim in VideoClip.detector by the parent
# (services/shot_detect.DETECTOR_ID). Duplicated on purpose — the two run in
# different interpreters, and a single test asserts they never drift, the
# same safeguard the watermark detector's model ids already carry.
DETECTOR_ID = 'transnetv2'

# The EXACT keys of one per-video row, in order. Declared as a constant rather
# than left implicit in a dict literal because the reader lives in another
# interpreter: a test asserts this tuple against what `_detect_one` really
# returns AND against the parent's own copy (services/shot_detect.ROW_KEYS), so
# neither half can quietly grow, lose or rename a key. The lesson is paid for —
# a stub built from a hand-copied shape stayed green for a whole wave while
# production had already moved on.
ROW_KEYS = ('path', 'state', 'shots', 'frame_count', 'fps_native', 'error',
            'probs')
# The two heads, named the same way on both sides of the pipe.
PROBS_KEYS = ('single', 'all')

# Emitted probabilities are rounded to this many decimals. A two-hour rush is a
# quarter of a million frames on ONE json line; four decimals is three orders of
# magnitude finer than any threshold anyone will ever set, and it roughly halves
# the line.
_PROB_DECIMALS = 4

# TransNetV2's fixed input geometry (H, W, C) — see the library's own
# `_input_size` attribute. Not configurable: the trained weights are for this
# exact shape.
_INPUT_SIZE = (27, 48, 3)


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _emit(obj):
    print(json.dumps(obj), flush=True)


def _cancel_requested(cancel_file):
    """The parent drops this sentinel file to ask for a clean stop between
    videos, rather than killing a process holding a loaded model."""
    return bool(cancel_file) and os.path.exists(cancel_file)


def _predictions_to_scenes(probs, threshold=0.5):
    """Per-frame transition probabilities -> [start, end] frame-index pairs
    (both inclusive), in playback order.

    A faithful port of TransNetV2's own `predictions_to_scenes` staticmethod
    to plain Python floats/ints, not a reimplementation from scratch. The
    upstream version lives in a module that `import torch`s at its own top
    level, so it cannot be exercised without torch installed; this copy has no
    import at all, which is what lets the transition rule be unit tested on an
    install with none of the video extras.

    `threshold` is used as-is (`>`, not `>=` — matching upstream exactly, so
    this port never draws one more boundary than the reference
    implementation would); clamping a hand-edited config value is the
    caller's job, not this pure function's.
    """
    if not probs:
        return []
    flags = [1 if p > threshold else 0 for p in probs]
    scenes = []
    t = -1
    t_prev = 0
    start = 0
    for i, t in enumerate(flags):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])
    if not scenes:
        # Every frame cleared the bar. TransNetV2's own fallback is "the whole
        # clip is one shot" rather than reporting zero shots for a file that
        # unambiguously has content.
        return [[0, len(flags) - 1]]
    return scenes


def _open(path):
    """The single PyAV decode seam — imported lazily so this module stays
    importable on an install with no video extra at all (mirrors
    video_probe._open on the video-bank branch)."""
    import av
    return av.open(path)


def _read_frames(path):
    """Decode `path` ONCE, straight to TransNetV2's 48x27 RGB input geometry.

    Returns (frames, fps_native, frame_count):
      frames      — numpy uint8 array, shape [T, 27, 48, 3]
      fps_native  — float measured from THIS decode pass, or None if the
                    container carries no usable rate (the parent falls back to
                    this only when its caller has no cached value)
      frame_count — T

    Raises on any decode failure, a stream with no video, or zero decodable
    frames — the caller (`_detect_one`) turns any of that into a per-file
    'error' row, exactly the way a corrupt image is handled elsewhere in this
    app; nothing here is fatal to the batch.
    """
    import numpy as np
    height, width = _INPUT_SIZE[0], _INPUT_SIZE[1]
    container = _open(path)
    try:
        streams = container.streams.video
        if not streams:
            raise RuntimeError('no video stream')
        stream = streams[0]
        fps_native = float(stream.average_rate) if stream.average_rate else None
        frames = []
        for frame in container.decode(stream):
            small = frame.reformat(width=width, height=height, format='rgb24')
            frames.append(small.to_ndarray(format='rgb24'))
        if not frames:
            raise RuntimeError('no decodable frames')
        return np.stack(frames).astype('uint8'), fps_native, len(frames)
    finally:
        try:
            container.close()
        except Exception:               # noqa: BLE001
            pass


def _load_model(device):
    """Lazy TransNetV2 load — the only place this worker imports torch.
    Weights are bundled in the wheel, so this never touches the network."""
    from transnetv2_pytorch import TransNetV2
    return TransNetV2(device=device)


def _run_model(model, frames):
    """The only place this worker touches a torch tensor. Returns BOTH heads as
    plain list[float]:

      single — the transition's centre frame. Boundaries are drawn from this,
               and every published F1 for this network measures it.
      all    — every frame the transition covers. Not a boundary signal; its
               WIDTH is what distinguishes a hard cut from a dissolve, and it
               is already computed by the same forward pass, so returning it
               costs a `.tolist()` and nothing else.
    """
    import numpy as np
    import torch
    tensor = torch.from_numpy(np.asarray(frames, dtype='uint8')).to(model.device)
    single_frame_pred, all_frame_pred = model.predict_frames(tensor, quiet=True)
    return (single_frame_pred.detach().cpu().tolist(),
            all_frame_pred.detach().cpu().tolist())


def _rounded(values):
    return [round(float(v), _PROB_DECIMALS) for v in values]


def _detect_one(path, model, threshold, emit_probs=False):
    """Decode + run + threshold ONE video. Never raises: a broken file becomes
    an 'error' row, exactly like the rest of this worker family, so one bad
    source never sinks the batch it was queued with.

    The row ALWAYS holds every key in ROW_KEYS, in that order — `probs` is None
    rather than absent when the parent did not ask for it. A row whose shape
    depends on the request is a row every reader has to guess at."""
    try:
        frames, fps_native, frame_count = _read_frames(path)
        single, every = _run_model(model, frames)
        shots = _predictions_to_scenes(single, threshold)
        probs = ({'single': _rounded(single), 'all': _rounded(every)}
                 if emit_probs else None)
        return {'path': path, 'state': 'ok', 'shots': shots,
                'frame_count': frame_count, 'fps_native': fps_native,
                'error': None, 'probs': probs}
    except Exception as e:               # noqa: BLE001 — one file, not the batch
        return {'path': path, 'state': 'error', 'shots': [],
                'frame_count': None, 'fps_native': None,
                'error': f'{type(e).__name__}: {e}', 'probs': None}


def main() -> int:
    try:
        job = json.loads(sys.stdin.read() or '{}')
    except ValueError as e:
        _emit({'summary': {'ok': False, 'error': f'unreadable job: {e}'}})
        return 1
    videos = [str(p) for p in (job.get('videos') or [])]
    threshold = float(job.get('threshold') or 0.5)
    cancel_file = job.get('cancel_file') or ''
    device = str(job.get('device') or 'auto')
    emit_probs = bool(job.get('emit_probs'))

    try:
        model = _load_model(device)
    except Exception as e:               # noqa: BLE001 — a load failure IS the pass
        _emit({'summary': {'ok': False, 'error': f'{type(e).__name__}: {e}'}})
        return 1
    _log(f'[shotdet] model ready ({DETECTOR_ID}, device={device})')

    processed = errors = 0
    for i, path in enumerate(videos, 1):
        if _cancel_requested(cancel_file):
            _log(f'[shotdet] cancelled at {i - 1}/{len(videos)}')
            break
        row = _detect_one(path, model, threshold, emit_probs)
        processed += 1
        if row['state'] == 'error':
            errors += 1
        _emit(row)
        _log(f'[shotdet] {i}/{len(videos)} {row["state"]}')
    _emit({'summary': {'ok': True, 'processed': processed, 'errors': errors,
                       'device': device}})
    return 0


if __name__ == '__main__':
    sys.exit(main())
