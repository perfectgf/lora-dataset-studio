"""Bank face pass — InsightFace antelopev2 embeddings + person clustering, run
in the DEDICATED ML interpreter (insightface/numpy are not in the Flask venv).
Device is chosen by the parent: 'cpu' (default — never touches the GPU/ComfyUI)
or 'cuda' (only when onnxruntime-gpu is present; the parent then runs this pass
inside its GPU-exclusive window). CUDA requested but unavailable → CPU fallback.

Protocol (same family as face_score_infer.py):
  stdin  : {"images": [abs paths], "models_root": path|null,
            "cache": abs path to a .npz|null, "threshold": 0.45,
            "device": "cpu"|"cuda"}
  stdout : ONE JSON line {"ok": bool, "results": {path: {state, det, bbox_frac}},
            "clusters": {path: int}, "used_gpu": bool, "error"?: str}
  stderr : "[embed] i/N <state>" progress lines (the parent streams these to
            drive the UI progress bar).

Embeddings are CACHED in the .npz (parallel arrays paths/embs/states/dets/
bfracs) and written incrementally every CACHE_EVERY images — killing the pass
mid-way loses at most that slice, and re-clustering at another threshold is
then near-instant. Clustering = union-find over cosine ≥ threshold on the
L2-normed embeddings of the SCORABLE faces (biggest face per image — a group
photo clusters by its dominant face); cluster ids are 1-based, ordered by
cluster size descending, singletons included (a person seen once is still a
cluster of one)."""
from __future__ import annotations
import json
import os
import sys

CACHE_EVERY = 50
DET_MIN, BBOX_MIN, YAW_MAX = 0.50, 0.06, 40.0   # same gates as face_score_infer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_score_infer import _repair_nested_antelopev2  # noqa: E402


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


def _load_cache(path):
    import numpy as np
    out = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with np.load(path, allow_pickle=False) as z:
            paths, states = z['paths'], z['states']
            embs, dets, bfracs = z['embs'], z['dets'], z['bfracs']
        for i, p in enumerate(paths):
            out[str(p)] = (states[i], float(dets[i]), float(bfracs[i]), embs[i])
    except Exception as e:  # noqa: BLE001 — a corrupt cache = recompute, never fatal
        _log(f'[embed] cache unreadable, recomputing: {e}')
        return {}
    return out


def _save_cache(path, cache):
    import numpy as np
    if not path or not cache:
        return
    paths = list(cache)
    tmp = path + '.tmp.npz'   # .npz suffix so numpy never appends its own
    np.savez_compressed(
        tmp,
        paths=np.array(paths),
        states=np.array([cache[p][0] for p in paths]),
        dets=np.array([cache[p][1] for p in paths], dtype='float32'),
        bfracs=np.array([cache[p][2] for p in paths], dtype='float32'),
        embs=np.stack([cache[p][3] for p in paths]).astype('float32'))
    os.replace(tmp, path)


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

    used_gpu = False   # set when the model actually loads on CUDA below
    cache = _load_cache(cache_path)
    todo = [p for p in images if p not in cache]
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

        import numpy as _np
        zero = _np.zeros(512, dtype='float32')
        done_since_save = 0
        if _cancel_requested(cancel_file):   # cancelled during the model load
            cached = len(images) - len(todo)
            _write_count(cache_path, cached)
            print(json.dumps({'ok': True, 'cancelled': True,
                              'cached': cached, 'remaining': len(todo)}))
            return 0
        for i, p in enumerate(todo, 1):
            try:
                img = cv2.imread(p)
                if img is None:
                    cache[p] = ('unreadable', 0.0, 0.0, zero)
                else:
                    h, w = img.shape[:2]
                    f = biggest(app.get(img))
                    if f is None:   # padding rescue: SCRFD misses full-frame closeups
                        pad = int(0.25 * max(h, w))
                        f = biggest(app.get(cv2.copyMakeBorder(
                            img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))))
                        scale = (w + 2 * pad) * (h + 2 * pad) / (w * h) if f is not None else 1.0
                    else:
                        scale = 1.0
                    if f is None:
                        cache[p] = ('no_face', 0.0, 0.0, zero)
                    else:
                        area = (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        bbox_frac = float(area / (w * h) / scale)
                        det = float(f.det_score)
                        yaw = float(f.pose[1]) if getattr(f, 'pose', None) is not None else 0.0
                        state = 'scorable'
                        if det < DET_MIN:
                            state = 'low_det'
                        elif bbox_frac < BBOX_MIN:
                            state = 'too_small'
                        elif abs(yaw) > YAW_MAX:
                            state = 'extreme_pose'
                        cache[p] = (state, round(det, 3), round(bbox_frac, 4),
                                    f.normed_embedding.astype('float32'))
            except Exception as e:  # noqa: BLE001 — one broken file must not sink the pass
                cache[p] = ('error', 0.0, 0.0, zero)
                _log(f'[embed] {i}/{len(todo)} ERROR {e}')
                continue
            finally:
                done_since_save += 1
                if cache_path and done_since_save >= CACHE_EVERY:
                    _save_cache(cache_path, cache)
                    _write_count(cache_path, len(images) - len(todo) + i)
                    done_since_save = 0
            _log(f'[embed] {i}/{len(todo)} {cache[p][0]}')
            if _cancel_requested(cancel_file):   # clean stop between images
                if cache_path:
                    _save_cache(cache_path, cache)
                cached = len(images) - len(todo) + i
                _write_count(cache_path, cached)
                print(json.dumps({'ok': True, 'cancelled': True,
                                  'cached': cached, 'remaining': len(todo) - i}))
                return 0
        if cache_path:
            _save_cache(cache_path, cache)
            _write_count(cache_path, len(images))

    results = {}
    for p in images:
        state, det, bfrac, _emb = cache.get(p) or ('error', 0.0, 0.0, None)
        results[p] = {'state': str(state), 'det': float(det), 'bbox_frac': float(bfrac)}
    clusters = _cluster(images, cache, threshold)
    print(json.dumps({'ok': True, 'results': results, 'clusters': clusters,
                      'used_gpu': used_gpu}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
