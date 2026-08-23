"""Bank face pass — InsightFace antelopev2 embeddings + person clustering, run
in the DEDICATED ML interpreter (insightface/numpy are not in the Flask venv).
Device is chosen by the parent: 'cpu' (default — never touches the GPU/ComfyUI)
or 'cuda' (only when onnxruntime-gpu is present; the parent then runs this pass
inside its GPU-exclusive window). CUDA requested but unavailable → CPU fallback.

Protocol (same family as face_score_infer.py):
  stdin  : {"images": [abs paths], "models_root": path|null,
            "cache": abs path to a .npz|null, "threshold": 0.45,
            "device": "cpu"|"cuda", "require_yaw": bool,
            "regate_too_small": bool,
            "groups": [{"name": str, "images": [abs paths]}]  # OPTIONAL}
  stdout : ONE JSON line {"ok": bool,
            "results": {path: {state, det, bbox_frac, yaw|null}},
            "clusters": {path: int}, "used_gpu": bool, "error"?: str,
            "group_clusters": {name: {path: int}}  # only when groups was given}
  stderr : "[embed] i/N <state>" progress lines (the parent streams these to
            drive the UI progress bar).

Embeddings are CACHED in the .npz (parallel arrays paths/embs/states/dets/
bfracs/yaws/bpx/sigs/hashes) and written incrementally every CACHE_EVERY images — killing the
pass mid-way loses at most that slice, and re-clustering at another threshold is
then near-instant.

``yaws`` and ``bpx`` were added after the fact and every cache written before
each of them lacks it. Those entries load with NaN — for the yaw, reported as
``yaw: null`` ("not measured", never 0.0, which would read as a perfectly frontal
face); for ``bpx``, the pixel size of the face this entry's verdict was taken on.
Both are ADDITIVE: nobody ever has to re-embed a bank because a new number
joined the tuple.

Two OPT-IN flags ask for cached entries to be RE-DETECTED, because neither
number is recoverable from a stored embedding. Both are off by default, so an
ordinary resume never turns into hours of re-detection:

  * ``require_yaw`` — entries with no measured angle. It is what the app's ⤢
    backfill sets.
  * ``regate_too_small`` — entries a PRE-``bpx`` build filed as ``too_small``.
    Their verdict came from the old gate, a fraction of the image AREA, which
    dropped most faces in an ordinary photo library (see FACE_PX_MIN below);
    the pixel size that would settle them under the new floor was never stored
    and CANNOT be recomputed from the fraction alone (the image dimensions are
    not in the cache either), so the only honest answer is to look again.
    Deliberately narrow: only ``too_small`` entries with no ``bpx`` qualify.
    Legacy entries under any other state keep the verdict they hold — the new
    floor exists to ADD faces to their person, never to demote a face that
    already clusters.

Update path for a bank that was scanned before this floor existed: a re-run of
👥 Group by person re-detects NOTHING by itself. ``_needs_work``/``_is_stale``
only reach an image whose bytes changed (stat signature + SHA-256), so every
already-cached verdict — including every old ``too_small`` — is returned as-is;
images ADDED to the bank, or edited, get the new floor immediately, and so does
any bank scanned for the first time. Freeing the faces an old scan skipped
therefore takes a caller passing ``regate_too_small``; nothing in the app sets
it yet (the ⤢ backfill is the shape that action would take).

Clustering = union-find over cosine ≥ threshold on the
L2-normed embeddings of the SCORABLE faces (biggest face per image — a group
photo clusters by its dominant face); cluster ids are 1-based, ordered by
cluster size descending, singletons included (a person seen once is still a
cluster of one)."""
from __future__ import annotations
import hashlib
import json
import os
import sys

CACHE_EVERY = 50
# Detection confidence and head-turn gates — the same values face_score_infer
# uses. Its SIZE gate is deliberately no longer the same: that pass answers
# "is this face usable for TRAINING", this one answers "is this face
# identifiable", and those are different questions about the same pixels.
DET_MIN, YAW_MAX = 0.50, 40.0
# Identity size floor, in ABSOLUTE PIXELS on the SHORT side of the detected box.
# It used to be BBOX_MIN = 0.06, a fraction of the image AREA (≈25% of the
# linear dimension), which made identifiability depend on the camera rather than
# on the face: the same 300 px head passes on a 1 Mpx photo and fails on a
# 24 Mpx one. Pointed at an ordinary photo library — full-body shots, groups,
# scenes — that gate sent almost every face to 'too_small', so it never got an
# embedding and never joined its person.
# 64 px is where the measurement itself stops meaning anything: ArcFace
# (antelopev2) recognises from a 112x112 aligned crop, so a face under ~64 px is
# upscaled by more than 2x into the model and its embedding drifts far enough to
# merge different people. Below the floor the answer would be noise; above it,
# a small face is merely a small face.
FACE_PX_MIN = 64.0
# --- Zoom rescue -------------------------------------------------------------
# The floor is only half the fix — the DETECTOR is the other half: FaceAnalysis
# fits the whole frame into det_size (640x640) before it looks, so a head in a
# full-body shot arrives a few pixels wide however many pixels the file holds.
# A second detection on a CROP around the head (or on a 2x upscale when nothing
# was found on a big frame) hands the model the resolution that was in the file
# all along. Mirrors face_score_infer, where the rationale is written out in
# full; same constants on both surfaces, pinned together by
# test_face_score_zoom_rescue.py.
ZOOM_PAD = 0.6
ZOOM_RETRY_MIN_SIDE = 1280

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_image_guard import read_validated_bank_image  # noqa: E402
from face_score_infer import _repair_nested_antelopev2  # noqa: E402
import npz_atomic  # noqa: E402


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _cancel_requested(cancel_file):
    """The parent drops this sentinel file to ask for a clean stop, so the pass
    flushes its cache and exits between images instead of being SIGKILLed
    mid-compute (which would lose up to CACHE_EVERY images)."""
    return bool(cancel_file) and os.path.exists(cancel_file)


def _write_count(cache_path, n):
    """Plain-text sidecar (``<cache>.count``) with how many images are cached so
    far. The Flask parent has no numpy to read the .npz, so this is how a stopped
    pass can still report an honest "N cached (M remaining)" — even in the rare
    case it had to be hard-killed before it could print its own cancel line."""
    if not cache_path:
        return
    try:
        with open(cache_path + '.count', 'w', encoding='utf-8') as f:
            f.write(str(int(n)))
    except OSError:
        pass


def _file_sig(path):
    """Cheap runtime invalidation signature, aligned with the Score cache."""
    try:
        st = os.stat(path)
        return f'{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return ''


def _file_hash(path):
    try:
        before = _file_sig(path)
        if not before:
            return b''
        digest = hashlib.sha256()
        with open(path, 'rb') as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.digest() if _file_sig(path) == before else b''
    except OSError:
        return b''


def _load_cache(path):
    """{path: (state, det, bbox_frac, emb, yaw, sig, hash, face_px)}. A cache
    written before the yaw or the bpx array existed loads that slot as NaN —
    additive, so no user ever has to re-embed a bank just because a new number
    was added to the tuple."""
    import numpy as np
    out = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with np.load(path, allow_pickle=False) as z:
            paths, states = z['paths'], z['states']
            embs, dets, bfracs = z['embs'], z['dets'], z['bfracs']
            yaws = z['yaws'] if 'yaws' in z.files else None
            bpx = z['bpx'] if 'bpx' in z.files else None
            sigs = z['sigs'] if 'sigs' in z.files else None
            hashes = z['hashes'] if 'hashes' in z.files else None
            if (hashes is not None
                    and (hashes.shape != (len(paths), 32)
                         or hashes.dtype != np.dtype('uint8'))):
                raise ValueError('invalid cache hash shape')
        for i, p in enumerate(paths):
            digest = hashes[i].tobytes() \
                if hashes is not None else b''
            if digest == b'\0' * 32:
                digest = b''
            out[str(p)] = (states[i], float(dets[i]), float(bfracs[i]), embs[i],
                           float(yaws[i]) if yaws is not None else float('nan'),
                           str(sigs[i]) if sigs is not None else '', digest,
                           float(bpx[i]) if bpx is not None else float('nan'))
    except Exception as e:  # noqa: BLE001 — a corrupt cache = recompute, never fatal
        _log(f'[embed] cache unreadable, recomputing: {e}')
        return {}
    return out


def _save_cache(path, cache):
    import numpy as np
    if not path:
        return
    if not cache:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    paths = list(cache)
    npz_atomic.save_npz_atomic(path, dict(
        paths=np.array(paths),
        states=np.array([cache[p][0] for p in paths]),
        dets=np.array([cache[p][1] for p in paths], dtype='float32'),
        bfracs=np.array([cache[p][2] for p in paths], dtype='float32'),
        yaws=np.array([cache[p][4] for p in paths], dtype='float32'),
        bpx=np.array([_cache_px(cache[p]) for p in paths], dtype='float32'),
        embs=np.stack([cache[p][3] for p in paths]).astype('float32'),
        sigs=np.array([_cache_sig(cache[p]) for p in paths]),
        hashes=np.frombuffer(b''.join(
            _cache_hash(cache[p]) or (b'\0' * 32) for p in paths),
            dtype='uint8').reshape(len(paths), 32)))


def _flush_cache(path, cache):
    """Save, and never let a HELD cache file kill the pass — the twin of the same
    helper in bank_score_infer. The finished archive stays on disk under its
    temporary name, and the next run salvages it."""
    try:
        _save_cache(path, cache)
    except npz_atomic.NpzReplaceLocked as error:
        _log(f'[embed] {error}')


def _salvage_cache(path):
    """Promote a temporary left by an interrupted run, if it reads back clean.

    ``_load_cache`` is the validator on purpose: it applies exactly the parsing a
    real read applies and touches every array, so a file truncated by a kill
    mid-write fails it instead of being promoted."""
    if not path:
        return
    npz_atomic.salvage_orphan_tmp(path, lambda p: len(_load_cache(p)), _log)


def _cache_sig(entry):
    return entry[5] if len(entry) > 5 else ''


def _cache_hash(entry):
    value = entry[6] if len(entry) > 6 else b''
    return value if isinstance(value, bytes) and len(value) == 32 else b''


def _cache_px(entry):
    """Short side, in pixels, of the face this entry's verdict was taken on —
    NaN for an entry cached before the size was stored ("not measured", and NOT
    something bbox_frac can be back-solved into: the image dimensions it was
    divided by are not in the cache)."""
    return float(entry[7]) if len(entry) > 7 else float('nan')


def _verdict(det, face_px, yaw):
    """The state of one detected face.

    ``face_px`` is the SHORT side of the detected box in pixels, measured on
    whatever image the detector was handed. The padding-rescue path needs no
    correction here: cv2.copyMakeBorder ADDS pixels, it does not rescale them,
    so a face keeps its pixel size in padded coordinates — unlike bbox_frac,
    which has to be divided back by the padded/original area ratio to stay a
    fraction of the ORIGINAL image."""
    if det < DET_MIN:
        return 'low_det'
    if not (face_px >= FACE_PX_MIN):     # NaN (no size) reads as too small
        return 'too_small'
    # No pose at all must not gate: NaN is "not measured", not a turned head.
    if abs(0.0 if yaw != yaw else yaw) > YAW_MAX:
        return 'extreme_pose'
    return 'scorable'


def _is_stale(path, entry):
    stored = _cache_sig(entry)
    digest = _cache_hash(entry)
    if not stored or not digest:
        return True
    current = _file_sig(path)
    return (not current or current != stored
            or _file_hash(path) != digest)


def _cluster(order, cache, threshold):
    """{path: 1-based cluster id} for the scorable faces of ``order``."""
    import numpy as np
    scorable = [p for p in order
                if p in cache and cache[p][0] == 'scorable' and cache[p][3] is not None]
    if not scorable:
        return {}
    E = np.stack([cache[p][3] for p in scorable]).astype('float32')
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)   # normed already; belt & braces
    n = len(scorable)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    chunk = 512
    for i0 in range(0, n, chunk):
        sims = E[i0:i0 + chunk] @ E.T
        for a, b in np.argwhere(sims >= threshold):
            a += i0
            if a < b:
                union(int(a), int(b))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda m: (-len(m), m[0]))
    out = {}
    for cid, members in enumerate(ordered, start=1):
        for i in members:
            out[scorable[i]] = cid
    return out


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                          'error': f'bad json: {e}'}))
        return 1
    images = [str(p) for p in (req.get('images') or [])]
    if not images:
        print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                          'error': 'no images'}))
        return 1
    models_root = req.get('models_root') or None
    cache_path = req.get('cache') or None
    cancel_file = req.get('cancel_file') or None
    threshold = float(req.get('threshold') or 0.45)
    device = str(req.get('device') or 'cpu').lower()   # 'cpu' | 'cuda'
    # Opt-in only: re-detect the entries a pre-yaw build cached, because the
    # angle cannot be recovered from a stored embedding. See the module docstring.
    require_yaw = bool(req.get('require_yaw'))
    # Opt-in only, and narrow: re-detect the entries a pre-bpx build filed as
    # 'too_small' under the old area-fraction gate. See the module docstring.
    regate_too_small = bool(req.get('regate_too_small'))

    used_gpu = False   # set when the model actually loads on CUDA below
    # Claim any embedding work a previous run finished but could not rename into
    # place (a held destination) or never got to rename at all (a hard kill).
    _salvage_cache(cache_path)
    cache = _load_cache(cache_path)

    def _needs_work(p):
        if p not in cache:
            return True
        if _is_stale(p, cache[p]):
            return True
        entry = cache[p]
        if require_yaw and entry[4] != entry[4]:                 # NaN test
            return True
        # A legacy 'too_small' is the ONLY verdict the pixel floor may overturn,
        # and only while no pixel size was ever stored for it. Everything else
        # keeps the state it holds.
        px = _cache_px(entry)
        return regate_too_small and str(entry[0]) == 'too_small' and px != px

    todo = [p for p in images if _needs_work(p)]
    _write_count(cache_path, len(images) - len(todo))
    _log(f'[embed] {len(images)} image(s), {len(images) - len(todo)} cached')

    if todo:
        import cv2
        import numpy as np  # noqa: F401 — insightface needs it importable
        from insightface.app import FaceAnalysis
        _repair_nested_antelopev2(models_root)
        # Provider selection is EXPLICIT per requested device — a bare
        # ['CUDAExecutionProvider', ...] would silently grab the GPU the moment
        # onnxruntime-gpu is present, outside the parent's GPU-exclusive window.
        # cpu → CPU only (ctx_id=-1); cuda → try CUDA, fall back to CPU (logged).
        import onnxruntime as ort
        avail = ort.get_available_providers()
        used_gpu = device == 'cuda' and 'CUDAExecutionProvider' in avail
        if device == 'cuda' and not used_gpu:
            _log('[embed] CUDA requested but CUDAExecutionProvider unavailable '
                 '(install onnxruntime-gpu) — falling back to CPU')
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if used_gpu else ['CPUExecutionProvider'])
        _log(f'[embed] providers={avail} device={device} used_gpu={used_gpu}')
        try:
            kwargs = {'name': 'antelopev2', 'providers': providers}
            if models_root:
                kwargs['root'] = models_root
            app = FaceAnalysis(**kwargs)
            app.prepare(ctx_id=0 if used_gpu else -1, det_size=(640, 640))
        except Exception as e:  # noqa: BLE001 — must exit as clean JSON, not a mute traceback
            print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                              'error': f'model load failed: {type(e).__name__}: {e}'}))
            return 1

        def biggest(faces):
            return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) \
                if faces else None

        def zoom_detect(img, f, lin_scale=1.0):
            """Re-detect on a crop centred on `f`, at the source resolution.

            `lin_scale` maps `f`'s coordinates back to source pixels (see the
            2x retry below). The crop itself never rescales, so the returned
            face is already measured in source pixels — mirrors
            face_score_infer.zoom_detect."""
            h, w = img.shape[:2]
            x1, y1, x2, y2 = (v / lin_scale for v in f.bbox)
            half = max(x2 - x1, y2 - y1) * (0.5 + ZOOM_PAD)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cx1, cy1 = max(0, int(cx - half)), max(0, int(cy - half))
            cx2, cy2 = min(w, int(cx + half)), min(h, int(cy + half))
            crop = img[cy1:cy2, cx1:cx2]
            return biggest(app.get(crop)) if crop.size else None

        import numpy as _np
        zero = _np.zeros(512, dtype='float32')
        done_since_save = 0
        uncached_changed = 0
        if _cancel_requested(cancel_file):   # cancelled during the model load
            cached = len(images) - len(todo)
            _write_count(cache_path, cached)
            print(json.dumps({'ok': True, 'cancelled': True,
                              'cached': cached, 'remaining': len(todo)}))
            return 0
        for i, p in enumerate(todo, 1):
            changed_while_reading = False
            signature = ''
            payload_hash = b''
            result = None
            try:
                # Read a bounded, validated snapshot rather than opening the
                # live Bank path with cv2.  The parent can only preflight a
                # path; it cannot stop that path being replaced before this
                # dedicated interpreter starts.
                signature = _file_sig(p)
                payload = read_validated_bank_image(p)
                payload_hash = hashlib.sha256(payload).digest()
                if not signature or _file_sig(p) != signature:
                    changed_while_reading = True
                    raise RuntimeError('image changed while it was read')
                img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    result = ('unreadable', 0.0, 0.0, zero, float('nan'),
                              signature, payload_hash, float('nan'))
                else:
                    h, w = img.shape[:2]
                    f = biggest(app.get(img))
                    padded = False
                    pad_scale = 1.0    # padded/original AREA ratio (bbox_frac only)
                    lin_scale = 1.0    # linear upscale of the 2x retry below
                    if f is None:   # padding rescue: SCRFD misses full-frame closeups
                        pad = int(0.25 * max(h, w))
                        f = biggest(app.get(cv2.copyMakeBorder(
                            img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))))
                        if f is not None:
                            padded = True
                            pad_scale = (w + 2 * pad) * (h + 2 * pad) / (w * h)
                    if f is None and max(h, w) > ZOOM_RETRY_MIN_SIDE:
                        # Nothing at det_size(640). On a frame this big that is
                        # usually the downscale rather than the absence of a
                        # face — a head in a full-body shot is a few pixels once
                        # the whole frame is squeezed into 640 (mirrors
                        # face_score_infer.analyze).
                        f = biggest(app.get(cv2.resize(
                            img, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_CUBIC)))
                        lin_scale = 2.0
                    if f is None:
                        result = ('no_face', 0.0, 0.0, zero, float('nan'),
                                  signature, payload_hash, float('nan'))
                    else:
                        # Sides in SOURCE pixels: /lin_scale undoes the 2x retry.
                        # The padding rescue needs no correction here — it ADDS
                        # pixels without rescaling them (see _verdict).
                        side_w = float(f.bbox[2] - f.bbox[0]) / lin_scale
                        side_h = float(f.bbox[3] - f.bbox[1]) / lin_scale
                        # bbox_frac stays a fraction of the ORIGINAL image (the
                        # /pad_scale undoes the padding rescue) — it is still
                        # stored and still read downstream. It is only no longer
                        # the thing the size VERDICT is taken on.
                        bbox_frac = float(side_w * side_h / (w * h) / pad_scale)
                        face_px = min(side_w, side_h)
                        det = float(f.det_score)
                        # The pose estimate: kept BOTH as the 'extreme_pose'
                        # gate (its original and only use) and, now, as a stored
                        # measurement — it is what the app's head-angle facet is
                        # made of. No pose at all stays NaN rather than becoming
                        # 0.0, which would claim a perfectly frontal face.
                        pose = getattr(f, 'pose', None)
                        yaw = float(pose[1]) if pose is not None else float('nan')
                        state = _verdict(det, face_px, yaw)
                        # One look at the head's own resolution before a size or
                        # detection verdict stands — the stored embedding then
                        # comes from the aligned crop those pixels deserve. NOT
                        # for 'extreme_pose': a profile stays a profile however
                        # close you crop, and embedding one MERGES people. And
                        # not after a padding rescue — that face already fills
                        # the frame. Mirrors face_score_infer.analyze.
                        if state in ('too_small', 'low_det') and not padded:
                            z = zoom_detect(img, f, lin_scale)
                            if z is not None:
                                zpose = getattr(z, 'pose', None)
                                zyaw = (float(zpose[1]) if zpose is not None
                                        else float('nan'))
                                zdet = float(z.det_score)
                                zpx = float(min(z.bbox[2] - z.bbox[0],
                                                z.bbox[3] - z.bbox[1]))
                                if _verdict(zdet, zpx, zyaw) == 'scorable':
                                    f, det, yaw = z, zdet, zyaw
                                    state, face_px = 'scorable', zpx
                                    bbox_frac = float(
                                        (z.bbox[2] - z.bbox[0])
                                        * (z.bbox[3] - z.bbox[1]) / (w * h))
                        result = (state, round(det, 3), round(bbox_frac, 4),
                                  f.normed_embedding.astype('float32'),
                                  yaw if yaw != yaw else round(yaw, 2), signature,
                                  payload_hash, round(face_px, 1))
                # Inference may be much slower than the validated read.  Commit
                # its local result only while the live path still identifies
                # the bytes captured above.
                if not signature or _file_sig(p) != signature:
                    changed_while_reading = True
                    raise RuntimeError('image changed while it was analysed')
                cache[p] = result
            except Exception as e:  # noqa: BLE001 — one broken file must not sink the pass
                # The validated read can itself fail because a live Bank file
                # was replaced or removed.  Compare the identity captured
                # before the read even on that exception path; otherwise an
                # ERROR could be signed for replacement bytes no model saw.
                if not signature or _file_sig(p) != signature:
                    changed_while_reading = True
                if changed_while_reading:
                    # Never bless the replacement with an ERROR computed from
                    # different bytes. Missing entry means the next run retries.
                    cache.pop(p, None)
                    uncached_changed += 1
                else:
                    cache[p] = ('error', 0.0, 0.0, zero, float('nan'), signature,
                                payload_hash, float('nan'))
                _log(f'[embed] {i}/{len(todo)} ERROR {e}')
                continue
            finally:
                done_since_save += 1
                if cache_path and done_since_save >= CACHE_EVERY:
                    _flush_cache(cache_path, cache)
                    _write_count(
                        cache_path,
                        len(images) - len(todo) + i - uncached_changed)
                    done_since_save = 0
            _log(f'[embed] {i}/{len(todo)} '
                 f'{cache[p][0] if p in cache else "changed; retry next run"}')
            if _cancel_requested(cancel_file):   # clean stop between images
                if cache_path:
                    _flush_cache(cache_path, cache)
                cached = sum(path in cache for path in images)
                _write_count(cache_path, cached)
                print(json.dumps({'ok': True, 'cancelled': True,
                                  'cached': cached, 'remaining': len(todo) - i}))
                return 0
        if cache_path:
            _flush_cache(cache_path, cache)
            _write_count(cache_path, sum(path in cache for path in images))

    results = {}
    for p in images:
        entry = cache.get(p) or ('error', 0.0, 0.0, None, float('nan'), '', b'',
                                 float('nan'))
        state, det, bfrac, _emb, yaw = entry[:5]
        digest = _cache_hash(entry)
        results[p] = {'state': str(state), 'det': float(det), 'bbox_frac': float(bfrac),
                      'fingerprint': digest.hex() if digest else None,
                      # null, never 0.0 — "not measured" is its own answer.
                      'yaw': None if yaw != yaw else float(yaw)}
    clusters = _cluster(images, cache, threshold)
    out = {'ok': True, 'results': results, 'clusters': clusters,
           'used_gpu': used_gpu}
    # Optional per-GROUP clustering, for the caller asking "is EACH of these
    # folders one person?" — a question the flat clustering above cannot answer:
    # it would happily merge two folders into a single cluster and let that read
    # as "consistent". Same threshold, same union-find, run once per group, and
    # riding on the SAME child call so a scan over forty folders costs one
    # subprocess and one model load — none at all when every image is cached.
    groups = req.get('groups') or []
    if groups:
        out['group_clusters'] = {
            str(g.get('name')): _cluster(
                [str(p) for p in (g.get('images') or [])], cache, threshold)
            for g in groups}
    print(json.dumps(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
