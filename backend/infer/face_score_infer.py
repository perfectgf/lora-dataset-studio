"""Face similarity scorer — InsightFace antelopev2, lance dans un interprete DEDIE
(insightface y est installe, PAS dans le venv Flask). CPU par DEFAUT (provider CPU +
ctx_id=-1) -> pas de GPU, ne touche pas ComfyUI. Le parent peut demander
{"device": "cuda"}, mais SEULEMENT depuis la fenetre GPU exclusive (cf. le
resolveur partage capabilities.resolve_face_device).
Protocole stdin: {"ref": path, "images": [paths], "models_root": path|null} -> stdout
UNE ligne JSON
{"ref_ok": bool, "results": {path: {state, sim?, det, bbox_frac, yaw, zoomed}}}.
Logs -> stderr.
Gating 3-etats + padding rescue (valide empiriquement sur test3) + zoom rescue."""
from __future__ import annotations
import json, sys

DET_MIN, YAW_MAX = 0.50, 40.0

# Identity size floor, in ABSOLUTE PIXELS on the SHORT side of the detected box.
# It used to be BBOX_MIN = 0.06, a fraction of the image AREA (~25% of the linear
# dimension), which made identifiability depend on the CAMERA rather than on the
# face: the same 300 px head passes on a 1 Mpx photo and fails on a 24 Mpx one.
# On a dataset built of full-body and bust shots that gate filed nearly every
# image 'too_small' — never scored, so 🎯 Auto-triage could never rule on it
# (reported on Discord). The Bank's own face pass moved off the fraction for the
# same reason and landed on the same number; the rationale is written out at
# face_embed_infer.FACE_PX_MIN, which this deliberately mirrors. Kept as two
# constants rather than one import because face_embed_infer imports THIS module
# (_repair_nested_antelopev2), so the dependency only runs one way.
FACE_PX_MIN = 64.0

# --- Zoom rescue -------------------------------------------------------------
# The size floor alone is not enough, because the DETECTOR is the other half of
# the problem: FaceAnalysis fits the whole frame into det_size (640x640) before
# it looks. A 120 px head in a 4000 px-wide full-body shot is ~19 px by the time
# SCRFD sees it — missed outright, or found with landmarks too coarse to align a
# usable 112x112 recognition crop. Re-running detection on a CROP around the head
# hands the model those pixels at their native resolution, which is the whole
# point: the information was always in the file, only the framing hid it.
ZOOM_PAD = 0.6          # bbox sides added around the box (hair, chin, ears)
# Whole-image 2x retry when the first pass found NOTHING. Only worth it when the
# frame is big enough for the det_size downscale to be what hid the face: at or
# under 1280 px the detector already saw it at >=50% scale, so a miss is a real
# miss and the retry would only cost CPU.
ZOOM_RETRY_MIN_SIDE = 1280
# Detector input size — shared with the Bank's embed pass for the same
# reason as every constant above it: same question, same answer. Pinned by
# test_face_score_zoom_rescue.
DET_SIZE = (640, 640)


def _log(m): print(m, file=sys.stderr, flush=True)


def _verdict(det, face_px, yaw):
    """The state of one detected face.

    ``face_px`` is the SHORT side of the box in SOURCE pixels — what the 112x112
    aligned recognition crop is actually limited by, so a 300x40 sliver reads as
    what it is rather than as a large face. Order matters: the reason reported is
    the first that applies. NaN inputs are answered conservatively but in
    opposite directions, because they mean opposite things — an unmeasured SIZE
    is not a pass, while an unmeasured POSE is not a turned head."""
    if det < DET_MIN:
        return 'low_det'
    if not (face_px >= FACE_PX_MIN):        # NaN (no size) reads as too small
        return 'too_small'
    if abs(0.0 if yaw != yaw else yaw) > YAW_MAX:
        return 'extreme_pose'
    return 'scorable'


def _repair_nested_antelopev2(models_root=None):
    """L'antelopev2.zip d'insightface 0.7.3 contient un DOSSIER RACINE (contrairement
    a buffalo_l) : l'auto-extract pose les .onnx dans .../models/antelopev2/antelopev2/,
    or FaceAnalysis globbe NON-recursivement -> 0 modele charge -> AssertionError
    (`'detection' in self.models`). CHAQUE install fraiche en auto-download est
    touchee, et ca ne s'auto-repare jamais (le dossier externe existe, insightface
    ne re-telecharge pas). On aplatit une fois pour toutes ici."""
    import glob, os, shutil
    root = models_root or os.path.join(os.path.expanduser('~'), '.insightface')
    outer = os.path.join(root, 'models', 'antelopev2')
    inner = os.path.join(outer, 'antelopev2')
    if not os.path.isdir(inner) or glob.glob(os.path.join(outer, '*.onnx')):
        return
    moved = 0
    for f in glob.glob(os.path.join(inner, '*.onnx')):
        shutil.move(f, outer)
        moved += 1
    try:
        os.rmdir(inner)
    except OSError:
        pass  # reliquats (zip...) — sans consequence
    if moved:
        _log(f"[face] repaired nested antelopev2 layout ({moved} model(s) moved up)")


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({"ref_ok": False, "results": {}, "error": f"bad json: {e}"})); return 1
    ref = req.get("ref"); images = [str(p) for p in (req.get("images") or [])]
    models_root = req.get("models_root") or None
    # 'cpu' (default) or 'cuda'. The PARENT decides — it is the only side that
    # knows whether it opened the GPU-exclusive window, and putting the model on
    # CUDA outside that window is exactly the unserialized GPU grab that was
    # removed here in "fix(gpu): serialize local inference and ComfyUI recovery".
    want_cuda = str(req.get("device") or "cpu").lower() == "cuda"
    if not ref or not images:
        print(json.dumps({"ref_ok": False, "results": {}, "error": "missing ref/images"})); return 1

    import numpy as np, cv2
    from insightface.app import FaceAnalysis
    _repair_nested_antelopev2(models_root)
    import onnxruntime as _ort
    # Ask for CUDA only if the parent requested it AND this interpreter really
    # exposes it: the stock face extra ships CPU onnxruntime, and listing a
    # provider that does not exist makes onnxruntime fall back MUTELY — the pass
    # would then run on CPU while the parent held ComfyUI paused for nothing.
    use_cuda = want_cuda and 'CUDAExecutionProvider' in _ort.get_available_providers()
    if want_cuda and not use_cuda:
        _log('[face] CUDA requested but CUDAExecutionProvider is unavailable - running on CPU')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda
                 else ['CPUExecutionProvider'])
    try:
        kwargs = {'name': 'antelopev2', 'providers': providers}
        if models_root:
            kwargs['root'] = models_root
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=0 if use_cuda else -1, det_size=DET_SIZE)
    except Exception as e:
        # Un crash de chargement (modeles absents/corrompus) doit sortir en JSON
        # propre — pas en traceback muet que le parent resume en « pas de JSON ».
        print(json.dumps({"ref_ok": False, "results": {},
                          "error": f"model load failed: {type(e).__name__}: {e}"}))
        return 1
    _log(f"[face] device={'cuda' if use_cuda else 'cpu'} "
         f"providers: {_ort.get_available_providers()}")

    def biggest(faces):
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None

    def _yaw(f):
        # NaN = never measured, which _verdict must not read as a turned head.
        return float(f.pose[1]) if getattr(f, "pose", None) is not None else float('nan')

    def _short_side(f, scale=1.0):
        return float(min(f.bbox[2]-f.bbox[0], f.bbox[3]-f.bbox[1]) / scale)

    def detect(img):
        f = biggest(app.get(img))
        if f is None:  # padding rescue : SCRFD rate les gros plans plein cadre
            h, w = img.shape[:2]; pad = int(0.25 * max(h, w))
            f2 = biggest(app.get(cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))))
            if f2 is not None:
                f2._padded = True
                return f2
        return f

    def zoom_detect(img, f, scale=1.0):
        """Re-detect on a crop centred on `f`, at the source image's resolution.

        `scale` maps `f`'s coordinates back to source pixels (see the 2x retry in
        analyze). The crop itself never rescales, so the returned face is already
        measured in source pixels."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (v / scale for v in f.bbox)
        half = max(x2 - x1, y2 - y1) * (0.5 + ZOOM_PAD)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cx1, cy1 = max(0, int(cx - half)), max(0, int(cy - half))
        cx2, cy2 = min(w, int(cx + half)), min(h, int(cy + half))
        crop = img[cy1:cy2, cx1:cx2]
        return biggest(app.get(crop)) if crop.size else None

    def analyze(path):
        img = cv2.imread(path)
        if img is None: return {"state": "unreadable"}
        h, w = img.shape[:2]
        f = detect(img)
        scale = 1.0
        if f is None and max(h, w) > ZOOM_RETRY_MIN_SIDE:
            # Nothing at det_size(640). On a frame this big that is usually the
            # downscale rather than the absence of a face — a head in a full-body
            # shot is a few pixels once the whole frame is squeezed into 640.
            f = biggest(app.get(cv2.resize(img, None, fx=2.0, fy=2.0,
                                           interpolation=cv2.INTER_CUBIC)))
            scale = 2.0
        if f is None: return {"state": "no_face"}
        # copyMakeBorder ADDS pixels without rescaling them, so a padded face
        # keeps its pixel size; only the AREA FRACTION has to be divided back by
        # the padded/original ratio to stay a fraction of the original image.
        frac_div = 1.0
        if getattr(f, "_padded", False):
            pad = int(0.25 * max(h, w)); frac_div = (w + 2*pad) * (h + 2*pad) / (w * h)
        area = (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]) / (scale * scale)
        bbox_frac = float(area / (w * h) / frac_div)
        det, yaw = float(f.det_score), _yaw(f)
        state = _verdict(det, _short_side(f, scale), yaw)
        zoomed = False
        # One look at the head's own resolution before a size/detection verdict
        # stands. NOT for 'extreme_pose': a profile stays a profile however close
        # you crop, and embedding one merges people instead of separating them.
        # Skipped after a padding rescue too — that path only fires on a face
        # that already fills the frame, so there is nothing left to zoom into.
        if state in ('too_small', 'low_det') and not getattr(f, "_padded", False):
            z = zoom_detect(img, f, scale)
            if z is not None:
                zdet, zyaw = float(z.det_score), _yaw(z)
                if _verdict(zdet, _short_side(z), zyaw) == 'scorable':
                    f, det, yaw, state, zoomed = z, zdet, zyaw, 'scorable', True
                    bbox_frac = float((z.bbox[2]-z.bbox[0]) * (z.bbox[3]-z.bbox[1])
                                      / (w * h))
        return {"state": state, "det": round(det, 3), "bbox_frac": round(bbox_frac, 4),
                "yaw": None if yaw != yaw else round(yaw, 1), "zoomed": zoomed,
                "_emb": f.normed_embedding}

    ref_res = analyze(ref)
    ref_emb = ref_res.pop("_emb", None)
    if ref_emb is None:
        print(json.dumps({"ref_ok": False, "results": {},
                          "error": f"ref unusable: {ref_res.get('state')}"})); return 1

    results = {}
    for i, p in enumerate(images, 1):
        try:
            r = analyze(p); emb = r.pop("_emb", None)
            if r["state"] == "scorable" and emb is not None:
                r["sim"] = round(float(np.dot(ref_emb, emb)), 4)
            results[p] = r
            _log(f"[face] {i}/{len(images)} {r['state']} sim={r.get('sim')}"
                 f"{' (zoomed)' if r.get('zoomed') else ''}")
        except Exception as e:
            results[p] = {"state": "error", "error": str(e)}
            _log(f"[face] {i}/{len(images)} ERROR {e}")
    print(json.dumps({"ref_ok": True, "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
