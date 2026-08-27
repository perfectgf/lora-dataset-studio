"""🔳 Where the letters are — RapidOCR over already-extracted frames, on CPU.

WHY RapidOCR AND NOT THE OBVIOUS ALTERNATIVES. The two heads this repo already
ships could both be pointed at text and both would be the wrong tool. The
watermark detector (SigLIP2 + Grounding DINO) answers "is a mark present" with a
scalar and needs ~2 GB of torch; the caption model reads a scene, not a
rectangle. What this pass needs is BOXES — a position and a size, three times per
shot, thousands of times per bank — and that is a detector's job, not a
classifier's or a VLM's.

RapidOCR is the one that fits without a new environment. It is Apache-2.0 (this
repo is public, and `ultralytics`' AGPL-over-weights claim is exactly why the
watermark detector is not built on it), it runs on the CPU onnxruntime the app
ALREADY installs for face scoring and masks, and its 1.4.x wheels carry the
PP-OCRv4 ONNX models INSIDE the package — 16 MB, no download, so this works on a
machine with no egress on the day it is installed. It is also what the pipelines
this pass copies use: Cosmos Curator's "Artificial Text Filter" is PaddleOCR, and
these are the same PP-OCR weights without the Paddle runtime.

THE CARD IS NEVER TAKEN. onnxruntime CPU only, deliberately: the whole point of
this pass is that it can measure a bank while a training run owns the GPU. There
is no `auto` branch to get wrong.

Measured on this machine, 768 px frames, CPU: 0.86 s to import, 0.55 s to build
the engine, 0.61 s per frame warm. Three frames per shot puts a thousand-shot
bank around half an hour — real money, which is why the pass has its own button
rather than riding somebody else's, and why the parent reports progress per
frame instead of per chunk.

Protocol (one JSON line in, one JSON line out — a batch, not a warm worker):
  stdin  : {"frames": [{"key": "<clip id>:<label>", "path": "<abs jpeg>"}, ...],
            "score_min": 0.5, "cancel_file": path|null}
  stdout : {"ok": true, "boxes": {"<key>": [[x0,y0,x1,y1,score], ...]},
            "read": N, "stopped": bool}
           {"ok": false, "error": "<ExcType>: <message>"}
  stderr : "[text] i/N" progress lines — the parent's only progress source.

BOXES COME BACK NORMALISED, 0..1, axis-aligned. Normalised because the parent
compares boxes ACROSS frames and a shot's frames are all the same shape but no
particular size, and because a stored measurement in pixels stops meaning
anything the day the extraction size changes. Axis-aligned because RapidOCR
returns a quadrilateral (it handles rotated text) and every consumer here — IoU
clustering, union area, edge cuts — is a rectangle problem; the bounding box of
the quad is the conservative choice, which is the right direction for a
measurement that justifies a crop.

WHAT IS DELIBERATELY NOT DONE HERE: no clustering, no coverage, no verdict. Those
are product decisions with arguments attached and they live in
app/services/video_safe_zone_geometry.py, where the tests can exercise them
without onnxruntime — the same split video_aesthetic_infer.py keeps from
video_metrics.aesthetic_of.
"""
from __future__ import annotations
import json
import os
import sys
from _harness import _cancel_requested, _emit, _log

# Hidden BEFORE anything heavy is imported. onnxruntime's CPU build ignores it,
# but an install that has replaced it with onnxruntime-gpu (the app never does,
# the user may) would otherwise quietly take a card a training run is using.
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _read_bgr(path):
    """The image as a BGR array, or None — three readers, cheapest first.

    cv2.imread leads: the historical reader, and the one that applies EXIF
    orientation (the boxes must live in the VISUAL geometry the repaint
    consumes). But on Windows it cannot open a non-ASCII path at all —
    measured: an accented folder name returns None with a findDecoder
    warning — which the image lane hits on real banks; video frames never
    did, their temp dir is ASCII. So a None falls through to PIL, which is
    unicode-safe and whose exif_transpose keeps the same visual geometry.
    PIL missing or refusing still falls through to np.fromfile + imdecode:
    unicode-safe too, no EXIF, better than dropping the file — for the pages
    and screenshots this lane reads, rotation EXIF is rare anyway.

    Imports live inside on purpose, like main()'s own cv2 import: this module
    must stay importable for tests without any of the heavy deps present.
    """
    import cv2
    try:
        image = cv2.imread(path)
    except Exception:  # noqa: BLE001
        image = None
    if image is not None:
        return image
    try:
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            return np.asarray(rgb)[:, :, ::-1].copy()
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def _boxes_of(result, width, height, score_min):
    """RapidOCR's [(quad, text, score), ...] as normalised [x0,y0,x1,y1,score].

    The TEXT is read and dropped. It is user content — a name on a chyron, a
    location on a lower third — and this pass answers a geometric question, so
    keeping the string would put footage transcripts in a database that has no
    use for them. The score is kept because it is what makes `score_min`
    meaningful.
    """
    out = []
    for entry in (result or []):
        try:
            quad, _text, score = entry[0], entry[1], entry[2]
            score = float(score)
        except (TypeError, ValueError, IndexError):
            continue
        if score < score_min:
            continue
        try:
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
        except (TypeError, ValueError, IndexError):
            continue
        if not xs or not ys:
            continue
        x0 = max(0.0, min(xs) / width)
        y0 = max(0.0, min(ys) / height)
        x1 = min(1.0, max(xs) / width)
        y1 = min(1.0, max(ys) / height)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append([round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4),
                    round(score, 3)])
    return out


def main() -> int:
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        _emit({'ok': False, 'error': f'bad json: {e}'})
        return 1
    frames = [f for f in (req.get('frames') or [])
              if f.get('path') and f.get('key')]
    score_min = float(req.get('score_min') or 0.0)
    cancel_file = req.get('cancel_file') or None

    if not frames:
        _emit({'ok': True, 'boxes': {}, 'read': 0, 'stopped': False})
        return 0

    try:
        # cv2 unused HERE on purpose (reading moved into _read_bgr): this
        # import is the dependency PROBE. Without it a missing cv2 would
        # surface as "every file unreadable" instead of the clean
        # "OCR deps missing" JSON the parent turns into an install hint.
        import cv2  # noqa: F401
        from rapidocr_onnxruntime import RapidOCR
    except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
        # The whole reason this is a subprocess: the parent turns this sentence
        # into a visible "text was not measured" state on every shot it touched,
        # and the pass still delivers its band measurements.
        _emit({'ok': False, 'error': f'OCR deps missing: {type(e).__name__}: {e}'})
        return 1

    try:
        engine = RapidOCR()
    except Exception as e:  # noqa: BLE001
        _emit({'ok': False, 'error': f'OCR engine unavailable: {type(e).__name__}: {e}'})
        return 1

    total = len(frames)
    _log(f'[text] 0/{total}')
    boxes = {}
    stopped = False
    for i, frame in enumerate(frames, start=1):
        if _cancel_requested(cancel_file):
            stopped = True
            break
        path = frame['path']
        # Read here rather than through the engine's own path handling: the
        # parent needs the boxes normalised, which needs the frame's size, and
        # reading the array here is the only place that knows it. _read_bgr
        # answers None on an unreadable file instead of raising, so a single
        # truncated JPEG leaves its frame without boxes rather than sinking
        # the chunk — and its fallbacks keep non-ASCII paths readable, which
        # plain cv2.imread is not on Windows.
        image = _read_bgr(path)
        if image is None:
            _log(f'[text] {i}/{total}')
            continue
        height, width = image.shape[0], image.shape[1]
        try:
            result, _elapse = engine(image)
        except Exception as e:  # noqa: BLE001 — one frame never sinks a chunk
            _log(f'[text] frame failed: {type(e).__name__}: {e}')
            result = None
        # ALWAYS keyed, empty list included. A frame that was read and holds no
        # text and a frame the run never reached are the same silence otherwise,
        # and the parent's whole resume contract is built on telling them apart:
        # the first is a measurement, the second must go back in the queue.
        boxes[frame['key']] = _boxes_of(result, width, height, score_min)
        _log(f'[text] {i}/{total}')

    _emit({'ok': True, 'boxes': boxes, 'read': len(boxes), 'stopped': stopped})
    return 0


if __name__ == '__main__':
    sys.exit(main())
