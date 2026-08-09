"""Face-mask generator — InsightFace (antelopev2), run by the DEDICATED ML
interpreter (insightface/numpy are not in the Flask venv). Same subprocess
pattern as face_score_infer.py / mask_infer.py.

WHAT THIS WRITES IS THE OPPOSITE POLARITY OF mask_infer.py.
  mask_infer.py  -> person WHITE on black background  (learn the subject, not the room)
  this script    -> face BLACK on a white frame       (learn the whole scene, not the identity)

ai-toolkit reads the PNG as a per-pixel LOSS WEIGHT (dataloader_mixins: white -> 1.0,
black -> mask_min_value; SDTrainer: loss = loss * mask). So a black face does NOT
erase pixels and does NOT paint anything into the image — it tells the trainer "do
not correct me here". That is why this is a loss mask and not a blur: a blurred face
would BE the regression target and the LoRA would learn to produce blur.

stdin  : {"images": [paths...], "out_dir": path|null, "expand": float,
          "cancel_file": path|null}
stdout : last line = JSON {"ok", "written", "cancelled", "results": {path: {...}}}
Logs -> stderr.

STOPPING (`cancel_file`)
  The parent asks for a stop by CREATING that file; this script notices it
  between two images and exits by PRINTING WHAT IT ALREADY FOUND, with
  `cancelled: true`. Being SIGKILLed would have been simpler and would have
  thrown that away: the results only exist in this process until the final JSON
  line, so a killed child means every face found so far is lost and a relaunch
  re-detects them. The boxes are the expensive part, so they are what survives.

  The check is between images ONLY. Nothing can interrupt `FaceAnalysis.prepare`
  from the outside, so a stop asked during the load lands at the top of the loop
  instead — the first thing the loop does, before image 1. That is why the
  parent states the cost of stopping DURING the load separately: there, nothing
  has been analysed yet and the load itself is what is lost.

PROGRESS PROTOCOL (stderr, read by app/services/face_mask.py):
  `[facemask] phase=<name>` then `[facemask] i/N ...` per image, same idiom as
  face_score_infer's `[face] i/N`. The phases exist because the per-image counter
  alone LIES about the wait: importing onnxruntime and preparing antelopev2 costs
  tens of seconds BEFORE image 1, and a bar frozen at 0/N for that long is
  indistinguishable from a hang — the exact confusion this reports away. Phases
  are only named when they really cost: `starting` (interpreter + heavy imports),
  `downloading` (models absent -> insightface will fetch ~350 MB), `loading`
  (FaceAnalysis.prepare), `detecting` (the per-image loop).

  A third shape carries the RESULT rather than the count: `[facemask] item {...}`,
  one per image, holding that image's path/state/boxes. It exists so the parent
  can bank detections WHILE the pass runs, instead of owning nothing until the
  final JSON line — see _item().

`out_dir` null = DETECT ONLY (no file written): the preview path asks for the raw
boxes and grows them client-side, so moving the expand slider redraws instantly
instead of paying for another InsightFace pass.
"""
import json
import os
import sys

# The face box InsightFace returns runs eyes-to-chin. Growing it around its centre
# is what turns it into a head box; the upward bias buys the hair, which sits
# entirely above that centre. Same two constants the app already uses to crop a
# head for a reference photo (face_crop_to_square_webp: pad 1.7, 10% shift up).
_SHIFT_UP = 0.10
# Above this fraction of the frame the mask stops being a good idea: ai-toolkit
# renormalises the mask to mean 1.0, so masking most of the image multiplies the
# loss on what little is left (70% masked -> x2.7 on the rest = a silent learning
# -rate bump for that sample). Such an image is left UNMASKED and reported.
_MAX_COVERAGE = 0.5
# Softens the ellipse edge. The mask is resampled to 1/8 resolution by the trainer
# anyway, so a hard edge buys nothing and a soft one avoids a brutal weight step.
_FEATHER_FRAC = 0.03


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _phase(name):
    """Announce a named stage of the run. Unbuffered: a phase line that arrives
    after the phase is over is worse than none."""
    _log(f'[facemask] phase={name}')


def _item(path, record):
    """Publish ONE image's result the moment it is known.

    Everything else this script computes lives in `results` until the final JSON
    line, so anything that ends the process before that line takes the whole pass
    with it — a parent watchdog, a kill, a machine that reboots. The graceful
    wind-up above only covers the case where someone ASKS; this covers the cases
    nobody asked for, by letting the parent bank as the pass goes.

    Deliberately a SEPARATE line from the `i/N` counter: those drive the progress
    bar and their grammar is under test, so they are left exactly as they were."""
    try:
        payload = json.dumps({'path': path, **record})
    except (TypeError, ValueError):
        return      # a result that cannot be serialised must not kill the pass
    _log(f'[facemask] item {payload}')


def _models_present(models_root=None) -> bool:
    """True when antelopev2's .onnx files are already on disk. Drives the
    `downloading` phase: insightface silently fetches ~350 MB on first use, and
    that download looks EXACTLY like a slow detection from the outside. Checks
    the nested layout too (see _repair_nested_antelopev2) so a fresh auto-extract
    that has not been flattened yet still counts as present, not as a re-download."""
    import glob
    root = models_root or os.path.join(os.path.expanduser('~'), '.insightface')
    outer = os.path.join(root, 'models', 'antelopev2')
    return bool(glob.glob(os.path.join(outer, '*.onnx'))
                or glob.glob(os.path.join(outer, 'antelopev2', '*.onnx')))


def cancel_requested(cancel_file) -> bool:
    """True once the parent has created the sentinel. A FILE rather than a signal
    because this is a plain stdin/stdout child on Windows too, where there is no
    usable POSIX signal for "wind up cleanly"."""
    return bool(cancel_file) and os.path.exists(cancel_file)


def dilate_box(box, expand, shift_up=_SHIFT_UP):
    """Grow a face box into a head box. PURE — mirrored verbatim by the frontend
    preview (frontend/src/utils/faceMaskBox.js) so what the user sees drawn is what
    the trainer will be given. Any change here changes there."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 - (y2 - y1) * shift_up
    hw = (x2 - x1) * expand / 2.0
    hh = (y2 - y1) * expand / 2.0
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        images = payload.get('images') or []
        out_dir = payload.get('out_dir') or None
        expand = float(payload.get('expand') or 2.0)
        models_root = payload.get('models_root') or None
        cancel_file = payload.get('cancel_file') or None
    except Exception as e:  # noqa: BLE001 — must exit as clean JSON, never a mute traceback
        print(json.dumps({"ok": False, "error": f"payload: {e}"}))
        return 1
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # First line out of the child: everything above is microseconds, everything
    # below is tens of seconds.
    _phase('starting')
    try:
        import cv2
        import numpy as np  # noqa: F401 — insightface needs it importable
        from PIL import Image, ImageDraw, ImageFilter
        from insightface.app import FaceAnalysis
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"import: {type(e).__name__}: {e}"}))
        return 1

    # antelopev2.zip ships a nested folder; reuse the repair the scoring script
    # already owns rather than growing a second copy of that trap.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from face_score_infer import _repair_nested_antelopev2
        _repair_nested_antelopev2(models_root)
    except Exception:  # noqa: BLE001 — repair is best-effort, never fatal
        pass

    # Announced BEFORE FaceAnalysis(), because that call is what triggers the
    # fetch: said afterwards it would arrive several minutes late, on a bar the
    # user already read as "stuck".
    _phase('downloading' if not _models_present(models_root) else 'loading')
    try:
        kwargs = {'name': 'antelopev2', 'providers': ['CPUExecutionProvider']}
        if models_root:
            kwargs['root'] = models_root
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"model load failed: {type(e).__name__}: {e}"}))
        return 1

    def detect_all(img):
        """EVERY face, not the biggest one. A concept photo routinely holds two or
        three people, and masking only the largest would leave the others' identity
        at full loss weight — worse than not masking, because the unmasked faces
        then dominate what the LoRA learns about faces."""
        faces = app.get(img) or []
        if faces:
            return faces, 1.0, 0
        # Padding rescue: SCRFD misses a face that fills the frame — exactly the
        # case we must not miss. Same 25% border the scoring script uses.
        h, w = img.shape[:2]
        pad = int(0.25 * max(h, w))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return (app.get(padded) or []), 1.0, pad

    results, written, cancelled = {}, 0, False
    _phase('detecting')
    for i, p in enumerate(images, 1):
        # BEFORE the work, not after: this is also where a stop asked during the
        # model load is honoured, and there it must cost zero images.
        if cancel_requested(cancel_file):
            cancelled = True
            _log(f'[facemask] stopped at {i - 1}/{len(images)}')
            break
        try:
            img = cv2.imread(p)
            if img is None:
                results[p] = {"state": "unreadable", "boxes": []}
                _item(p, results[p])
                _log(f'[facemask] {i}/{len(images)} unreadable')
                continue
            h, w = img.shape[:2]
            faces, _, pad = detect_all(img)
            # Boxes are reported NORMALISED (0-1, relative to the real image) so the
            # preview can draw them on an <img> of any displayed size, and the pad
            # rescue's offset is undone here rather than leaking into the UI.
            boxes = []
            for f in faces:
                x1, y1, x2, y2 = (float(v) for v in f.bbox[:4])
                x1, x2 = (x1 - pad) / w, (x2 - pad) / w
                y1, y2 = (y1 - pad) / h, (y2 - pad) / h
                boxes.append([x1, y1, x2, y2])
            if not boxes:
                # No face is a NORMAL outcome, not an error: a concept dataset may
                # legitimately hold no people at all. An all-white mask is an exact
                # no-op (white -> weight 1.0 everywhere), so the run is byte-for-byte
                # an unmasked one for this image.
                results[p] = {"state": "no_face", "boxes": []}
                if out_dir:
                    name = os.path.splitext(os.path.basename(p))[0] + '.png'
                    Image.new('L', (w, h), 255).save(os.path.join(out_dir, name), 'PNG')
                    written += 1
                _item(p, results[p])
                _log(f'[facemask] {i}/{len(images)} no_face')
                continue

            mask = Image.new('L', (w, h), 255)
            draw = ImageDraw.Draw(mask)
            covered = 0.0
            for b in boxes:
                dx1, dy1, dx2, dy2 = dilate_box(b, expand)
                px = (dx1 * w, dy1 * h, dx2 * w, dy2 * h)
                covered += max(0.0, (min(dx2, 1.0) - max(dx1, 0.0))) * \
                    max(0.0, (min(dy2, 1.0) - max(dy1, 0.0)))
                draw.ellipse([px[0], px[1], px[2], px[3]], fill=0)
            state = "masked"
            if covered > _MAX_COVERAGE:
                # Renormalisation would turn this into a hidden LR bump; refuse and say so.
                state = "too_large"
                mask = Image.new('L', (w, h), 255)
            elif out_dir:
                r = max(1, int(min(w, h) * _FEATHER_FRAC))
                mask = mask.filter(ImageFilter.GaussianBlur(radius=r))
            results[p] = {"state": state, "boxes": boxes, "coverage": round(covered, 4)}
            if out_dir:
                name = os.path.splitext(os.path.basename(p))[0] + '.png'
                mask.save(os.path.join(out_dir, name), 'PNG')
                written += 1
            _item(p, results[p])
            _log(f'[facemask] {i}/{len(images)} {state} faces={len(boxes)}')
        except Exception as e:  # noqa: BLE001 — one bad image must not kill the pass
            results[p] = {"state": "error", "error": str(e), "boxes": []}
            _item(p, results[p])
            _log(f'[facemask] {i}/{len(images)} ERROR {e}')
    # `ok` stays TRUE on a stop: the pass did not fail, it was asked to end early
    # and it is handing back everything it managed to compute. Reporting it as an
    # error would push the parent down the failure path and discard exactly the
    # work this whole mechanism exists to keep.
    print(json.dumps({"ok": True, "written": written, "cancelled": cancelled,
                      "results": results}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
