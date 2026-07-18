"""Bank scoring pass — CLIP ViT-L/14 aesthetic + NSFW classifier + style
embeddings, run in the DEDICATED "bank scoring" ML interpreter (torch/open_clip/
transformers are not in the Flask venv). Same subprocess family as
face_embed_infer.py; CUDA when available, else CPU.

One CLIP forward per image feeds THREE outputs, so a mixed dump can be triaged by
"nice vs ugly", "SFW vs NSFW" and "grouped by visual style" from a single pass:

  * aesthetic — the LAION improved-aesthetic MLP head on the (L2-normed) CLIP
    ViT-L/14 image embedding → ~1..10 (higher = nicer). The head weights download
    once from the public LAION repo and cache under `models_root` (or the HF cache).
  * nsfw      — Marqo/nsfw-image-detection-384 (a small self-contained timm/
    transformers classifier) → P(NSFW) in 0..1. A SEPARATE tiny model load, kept in
    the same subprocess so the pass stays one queued job.
  * style     — the CLIP image embedding itself (L2-normed), cached in the .npz and
    union-find clustered by cosine ≥ style_threshold, exactly like the face pass
    clusters by identity. Cluster ids are 1-based, biggest first.

Protocol (same shape as face_embed_infer.py):
  stdin  : {"images": [abs paths], "models_root": path|null,
            "cache": abs path to a .npz|null, "style_threshold": 0.6}
  stdout : ONE JSON line {"ok": bool,
            "results": {path: {state, aesthetic?, nsfw?}},
            "clusters": {path: int}, "error"?: str}
  stderr : "[score] i/N <state>" progress lines the parent streams to the UI.

Each of the three heads degrades independently: if the aesthetic weights can't be
fetched the pass still returns nsfw + style (aesthetic omitted), and vice-versa —
a broken single head never sinks the whole pass. Embeddings + scores are cached in
the .npz and written every CACHE_EVERY images, so killing the pass mid-way loses at
most that slice and re-clustering at another threshold is near-instant."""
from __future__ import annotations
import json
import os
import sys

CACHE_EVERY = 50

# LAION improved-aesthetic-predictor (the canonical 7-layer MLP over a
# L2-normalized CLIP ViT-L/14 image embedding). Public, ~13 MB, downloaded once.
_AESTHETIC_URL = ('https://github.com/christophschuhmann/improved-aesthetic-predictor/'
                  'raw/main/sac+logos+ava1-l14-linearMSE.pth')
_AESTHETIC_FILE = 'sac+logos+ava1-l14-linearMSE.pth'
_NSFW_MODEL = 'Marqo/nsfw-image-detection-384'


def _log(m):
    print(m, file=sys.stderr, flush=True)


# --- caching (parallel-array .npz, same idea as the face cache) ----------------
def _load_cache(path):
    import numpy as np
    out = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with np.load(path, allow_pickle=False) as z:
            paths, states = z['paths'], z['states']
            aes, nsfw, embs = z['aes'], z['nsfw'], z['embs']
        for i, p in enumerate(paths):
            a = float(aes[i])
            n = float(nsfw[i])
            out[str(p)] = (str(states[i]),
                           None if a != a else a,      # NaN sentinel = "not scored"
                           None if n != n else n,
                           embs[i])
    except Exception as e:  # noqa: BLE001 — a corrupt cache = recompute, never fatal
        _log(f'[score] cache unreadable, recomputing: {e}')
        return {}
    return out


def _save_cache(path, cache):
    import numpy as np
    if not path or not cache:
        return
    paths = list(cache)
    nan = float('nan')
    tmp = path + '.tmp.npz'
    np.savez_compressed(
        tmp,
        paths=np.array(paths),
        states=np.array([cache[p][0] for p in paths]),
        aes=np.array([nan if cache[p][1] is None else cache[p][1] for p in paths],
                     dtype='float32'),
        nsfw=np.array([nan if cache[p][2] is None else cache[p][2] for p in paths],
                      dtype='float32'),
        embs=np.stack([cache[p][3] for p in paths]).astype('float32'))
    os.replace(tmp, path)


# --- style clustering (union-find over cosine, testable without torch) ---------
def _cluster_style(order, cache, threshold):
    """{path: 1-based style-cluster id} over the embeddings of the SCORED images of
    ``order`` (state 'ok' and a non-zero embedding). Biggest cluster first; a style
    seen once is still a cluster of one — same contract as the face clustering."""
    import numpy as np
    usable = [p for p in order
              if p in cache and cache[p][0] == 'ok' and cache[p][3] is not None
              and float(np.abs(cache[p][3]).sum()) > 0]
    if not usable:
        return {}
    E = np.stack([cache[p][3] for p in usable]).astype('float32')
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    n = len(usable)
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
            out[usable[i]] = cid
    return out


# --- model heads ---------------------------------------------------------------
def _aesthetic_mlp():
    """The improved-aesthetic-predictor MLP (768→1) as an nn.Module."""
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(768, 1024), nn.Dropout(0.2),
                nn.Linear(1024, 128), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.Dropout(0.1),
                nn.Linear(64, 16), nn.Linear(16, 1))

        def forward(self, x):
            return self.layers(x)

    return _MLP()


def _load_aesthetic_head(models_root, device):
    """(module, ok). Downloads the LAION head weights once (cached under models_root
    or the default HF hub cache), returns (None, False) on any failure so the pass
    still yields nsfw + style."""
    import torch
    try:
        cache_dir = os.path.join(models_root or _default_cache(), 'bank_scoring')
        os.makedirs(cache_dir, exist_ok=True)
        dest = os.path.join(cache_dir, _AESTHETIC_FILE)
        if not os.path.isfile(dest):
            _log('[score] fetching aesthetic head weights (once)…')
            import urllib.request
            urllib.request.urlretrieve(_AESTHETIC_URL, dest + '.part')
            os.replace(dest + '.part', dest)
        head = _aesthetic_mlp()
        # weights_only: the head is a plain tensor state_dict — never unpickle
        # arbitrary objects from a downloaded file.
        state = torch.load(dest, map_location='cpu', weights_only=True)
        head.load_state_dict(state)
        head.to(device).eval()
        return head, True
    except Exception as e:  # noqa: BLE001
        _log(f'[score] aesthetic head unavailable ({type(e).__name__}: {e}) — '
             'aesthetic scores skipped')
        return None, False


def _load_nsfw(device):
    """((model, processor, nsfw_index), ok). Marqo NSFW classifier; degrades to
    (None, False) so a fetch/load failure only drops the nsfw column."""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        proc = AutoImageProcessor.from_pretrained(_NSFW_MODEL)
        model = AutoModelForImageClassification.from_pretrained(_NSFW_MODEL)
        model.to(device).eval()
        # Find the NSFW label index robustly (id2label wording varies by revision).
        id2label = getattr(model.config, 'id2label', {}) or {}
        nsfw_idx = 0
        for idx, label in id2label.items():
            if 'nsfw' in str(label).lower():
                nsfw_idx = int(idx)
                break
        return (model, proc, nsfw_idx), True
    except Exception as e:  # noqa: BLE001
        _log(f'[score] NSFW model unavailable ({type(e).__name__}: {e}) — '
             'nsfw scores skipped')
        return None, False


def _default_cache():
    from pathlib import Path
    return str(Path.home() / '.cache' / 'lds')


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
    style_threshold = float(req.get('style_threshold') or 0.6)

    cache = _load_cache(cache_path)
    todo = [p for p in images if p not in cache]
    _log(f'[score] {len(images)} image(s), {len(images) - len(todo)} cached')

    if todo:
        try:
            import numpy as np  # noqa: F401
            import open_clip
            import torch
            from PIL import Image
        except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
            print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                              'error': f'ML deps missing: {type(e).__name__}: {e}'}))
            return 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='openai',
                cache_dir=(os.path.join(models_root, 'open_clip') if models_root else None))
            clip_model.to(device).eval()
        except Exception as e:  # noqa: BLE001
            print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                              'error': f'CLIP load failed: {type(e).__name__}: {e}'}))
            return 1
        aes_head, aes_ok = _load_aesthetic_head(models_root, device)
        nsfw_bundle, nsfw_ok = _load_nsfw(device)
        zero = np.zeros(768, dtype='float32')
        done_since_save = 0
        for i, p in enumerate(todo, 1):
            try:
                with Image.open(p) as im:
                    im = im.convert('RGB')
                    with torch.no_grad():
                        tens = preprocess(im).unsqueeze(0).to(device)
                        emb = clip_model.encode_image(tens)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb_np = emb.cpu().numpy()[0].astype('float32')
                        aesthetic = None
                        if aes_ok:
                            aesthetic = round(float(aes_head(emb)[0][0].item()), 3)
                        nsfw = None
                        if nsfw_ok:
                            model, proc, nsfw_idx = nsfw_bundle
                            inp = proc(images=im, return_tensors='pt').to(device)
                            logits = model(**inp).logits
                            probs = torch.softmax(logits, dim=-1)[0]
                            nsfw = round(float(probs[nsfw_idx].item()), 4)
                    cache[p] = ('ok', aesthetic, nsfw, emb_np)
            except Exception as e:  # noqa: BLE001 — one broken file never sinks the pass
                cache[p] = ('error', None, None, zero)
                _log(f'[score] {i}/{len(todo)} ERROR {e}')
                continue
            finally:
                done_since_save += 1
                if cache_path and done_since_save >= CACHE_EVERY:
                    _save_cache(cache_path, cache)
                    done_since_save = 0
            _log(f'[score] {i}/{len(todo)} {cache[p][0]}')
        if cache_path:
            _save_cache(cache_path, cache)

    results = {}
    for p in images:
        state, aesthetic, nsfw, _emb = cache.get(p) or ('error', None, None, None)
        entry = {'state': str(state)}
        if aesthetic is not None:
            entry['aesthetic'] = float(aesthetic)
        if nsfw is not None:
            entry['nsfw'] = float(nsfw)
        results[p] = entry
    clusters = _cluster_style(images, cache, style_threshold)
    print(json.dumps({'ok': True, 'results': results, 'clusters': clusters}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
