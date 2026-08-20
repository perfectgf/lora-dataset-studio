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
            "cache": abs path to a .npz|null, "style_threshold": 0.6,
            "rescore": false}
  stdout : ONE JSON line {"ok": bool,
            "results": {path: {state, aesthetic?, nsfw?}},
            "clusters": {path: int}|null, "computed": int, "reused": int,
            "error"?: str}
  stderr : "[score] i/N <state>" progress lines the parent streams to the UI, and
           "[phase] <sentence>" lines for the steps that have no per-image
           counter (cache write, style grouping) — the parent shows those
           verbatim instead of leaving a full bar and a stale count behind.

Each of the three heads degrades independently: if the aesthetic weights can't be
fetched the pass still returns nsfw + style (aesthetic omitted), and vice-versa —
a broken single head never sinks the whole pass. Embeddings + scores are cached in
the .npz and written every CACHE_EVERY images, so killing the pass mid-way loses at
most that slice.

RESUMING is what the cache is for, and it has two halves:
  * a path already cached (and unchanged on disk) is never embedded again — that
    is the cheap half, and it is why the payload stays the WHOLE bank instead of
    "the unscored rows". The clustering below needs every embedding, so shrinking
    the payload would not save inference, it would only make the style partition
    wrong;
  * a path cached while a head was DOWN carries a hole (aesthetic or nsfw None)
    and state 'ok'. Those used to be cached forever: the entry existed, so the
    image was skipped, so the missing score never came back — a permanent gap
    from a transient download failure. They are now retried whenever the head
    that was missing is available again, and only then (a head that is still
    down leaves them exactly as they are, so no run ever churns).

``rescore: true`` ignores the cache entirely — the explicit "recompute everything"
lane (new model, thresholds you no longer trust). The normal pass never does this.

Style clustering is NOT near-instant: measured 7.6 s over 5 000 images and 181 s
over 23 000 (n² cosine + a Python pass over every pair above the threshold). It is
therefore skipped on the cancel path — a stopped pass has 15 s to answer before the
parent kills it, and spending three minutes there would throw away the scores it
was trying to save."""
from __future__ import annotations
import hashlib
import io
import json
import os
import sys

CACHE_EVERY = 50

# LAION improved-aesthetic-predictor (the canonical 7-layer MLP over a
# L2-normalized CLIP ViT-L/14 image embedding). Public, ~13 MB, downloaded once.
_AESTHETIC_URL = ('https://github.com/christophschuhmann/improved-aesthetic-predictor/'
                  'raw/main/sac+logos+ava1-l14-linearMSE.pth')
_AESTHETIC_FILE = 'sac+logos+ava1-l14-linearMSE.pth'
# The `l14` in that filename is ViT-L/14 AS OPENAI RELEASED IT, not a family:
# the MLP was fitted on that one embedding space. Any other 768-d CLIP feeds it
# without error and yields a plausible, meaningless score — so the pair lives
# here, beside the checkpoint that requires it, and
# tests/test_clip_text_model_contract.py fails if the scoring pass drifts off it.
_AESTHETIC_EXPECTS = ('ViT-L-14', 'openai')
_NSFW_MODEL = 'Marqo/nsfw-image-detection-384'


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402
from bank_image_guard import read_validated_bank_image  # noqa: E402
import npz_atomic  # noqa: E402


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _phase(sentence):
    """Announce a step that has NO per-image counter, in words the UI shows as-is.

    The parent forwards any ``[phase] …`` line straight into the job's detail and
    clears the counter with it (see _PHASE_RE in image_bank_service). So this
    sentence is user-facing English, and it states what a Stop would cost RIGHT
    HERE when that differs from the rest of the pass — the whole point being that
    the Stop button stays offered through steps where it destroys different
    things."""
    _log(f'[phase] {sentence}')


def _cancel_requested(cancel_file):
    """The parent drops this sentinel file to ask for a clean stop, so the pass
    flushes its cache and exits between images instead of being SIGKILLed
    mid-compute (which would lose up to CACHE_EVERY images)."""
    return bool(cancel_file) and os.path.exists(cancel_file)


def _write_count(cache_path, n):
    """Plain-text sidecar (``<cache>.count``) with how many images are scored so
    far. The Flask parent has no numpy to read the .npz, so this is how a stopped
    pass can still report an honest "N scored (M remaining)" — even in the rare
    case it had to be hard-killed before it could print its own cancel line."""
    if not cache_path:
        return
    try:
        with open(cache_path + '.count', 'w', encoding='utf-8') as f:
            f.write(str(int(n)))
    except OSError:
        pass


def _file_sig(path):
    """A cheap identity signature for a file — size + mtime (ns). A cached entry
    whose signature no longer matches the file on disk is STALE (the image was
    edited/replaced at the same path) and must be re-scored, so a stale embedding
    never lingers behind a semantic near-duplicate group. '' when the file is
    unreachable (leaves the entry as-is rather than churning it every run)."""
    try:
        st = os.stat(path)
        return f'{st.st_size}:{st.st_mtime_ns}'
    except OSError:
        return ''


def _file_hash(path):
    """Raw SHA-256 of the live path, or ``b''`` when it cannot be read."""
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


# --- caching (parallel-array .npz, same idea as the face cache) ----------------
# Cache tuple: (state, aesthetic|None, nsfw|None, emb, sig, sha256). ``sig`` is
# the cheap runtime invalidator; the raw 32-byte digest is transfer authority.
def _load_cache(path):
    import numpy as np
    out = {}
    if not path or not os.path.isfile(path):
        return out
    try:
        with np.load(path, allow_pickle=False) as z:
            paths, states = z['paths'], z['states']
            aes, nsfw, embs = z['aes'], z['nsfw'], z['embs']
            # 'sigs' is additive — a cache written before signatures shipped has
            # none, so those entries carry an empty sig (never treated as stale).
            sigs = z['sigs'] if 'sigs' in z.files else [''] * len(paths)
            hashes = z['hashes'] if 'hashes' in z.files else None
            if (hashes is not None
                    and (hashes.shape != (len(paths), 32)
                         or hashes.dtype != np.dtype('uint8'))):
                raise ValueError('invalid cache hash shape')
        for i, p in enumerate(paths):
            a = float(aes[i])
            n = float(nsfw[i])
            digest = hashes[i].tobytes() \
                if hashes is not None else b''
            if digest == b'\0' * 32:
                digest = b''
            out[str(p)] = (str(states[i]),
                           None if a != a else a,      # NaN sentinel = "not scored"
                           None if n != n else n,
                           embs[i], str(sigs[i]), digest)
    except Exception as e:  # noqa: BLE001 — a corrupt cache = recompute, never fatal
        _log(f'[score] cache unreadable, recomputing: {e}')
        return {}
    return out


def _cache_sig(entry):
    """The signature of a cache tuple, tolerant of a legacy 4-tuple (no sig)."""
    return entry[4] if len(entry) > 4 else ''


def _cache_hash(entry):
    value = entry[5] if len(entry) > 5 else b''
    return value if isinstance(value, bytes) and len(value) == 32 else b''


def _is_stale(path, entry):
    """True when the file at ``path`` differs from what the cache recorded — a
    same-path edit. An empty stored/current sig (unreadable then or now) is never
    called stale, so we never thrash a file we cannot stat."""
    stored = _cache_sig(entry)
    digest = _cache_hash(entry)
    # A legacy entry remains readable, but a deliberate Score pass re-computes
    # it once so every result written back can carry cryptographic byte
    # authority.  Stat equality alone cannot distinguish a same-size/mtime edit.
    if not stored or not digest:
        return True
    current = _file_sig(path)
    return not current or current != stored or _file_hash(path) != digest


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
    nan = float('nan')
    npz_atomic.save_npz_atomic(path, dict(
        paths=np.array(paths),
        states=np.array([cache[p][0] for p in paths]),
        aes=np.array([nan if cache[p][1] is None else cache[p][1] for p in paths],
                     dtype='float32'),
        nsfw=np.array([nan if cache[p][2] is None else cache[p][2] for p in paths],
                      dtype='float32'),
        embs=np.stack([cache[p][3] for p in paths]).astype('float32'),
        sigs=np.array([_cache_sig(cache[p]) for p in paths]),
        hashes=np.frombuffer(b''.join(
            _cache_hash(cache[p]) or (b'\0' * 32) for p in paths),
            dtype='uint8').reshape(len(paths), 32)))


def _flush_cache(path, cache):
    """Save, and NEVER let a held cache file kill the pass.

    Losing this write costs a re-score; dying here costs the whole run. The
    finished archive stays on disk under its temporary name and the next pass
    salvages it (``_salvage_cache``), so the honest thing is to tell the user
    what happened and keep going."""
    try:
        _save_cache(path, cache)
    except npz_atomic.NpzReplaceLocked as error:
        _phase(str(error))


def _salvage_cache(path):
    """Promote a temporary left by an interrupted run, if it reads back clean.

    ``_load_cache`` is the validator on purpose: it is exactly the parsing a real
    read applies, and it touches every array, so a file truncated by a kill
    mid-write fails it instead of being promoted."""
    if not path:
        return
    npz_atomic.salvage_orphan_tmp(path, lambda p: len(_load_cache(p)), _log)


# --- style clustering (union-find over cosine, testable without torch) ---------
def _cluster_style(order, cache, threshold, should_stop=None):
    """{path: 1-based style-cluster id} over the embeddings of the SCORED images of
    ``order`` (state 'ok' and a non-zero embedding). Biggest cluster first; a style
    seen once is still a cluster of one — same contract as the face clustering.

    ``should_stop`` is polled between blocks and makes this return None: the ids are
    a PARTITION of the bank, so a half-finished one is not "some clusters", it is a
    wrong answer. None means "write nothing", which leaves the previous partition
    intact. It matters because this is the slow tail of the pass (181 s over 23 000
    images), so a Stop very often lands right here."""
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
        if should_stop is not None and should_stop():
            return None
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
    """(module, ok, reason). Downloads the LAION head weights once (cached under
    models_root or the default HF hub cache), returns (None, False, why) on any
    failure so the pass still yields nsfw + style. `why` is a one-line
    "<ExcType>: <message>" the parent puts in front of the user: this fetch is the
    first network call of a pass, so on a machine with no egress it is also the
    reason every score comes back empty — and "unavailable" alone left the user
    with a completed pass, no scores, and nothing to act on."""
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
        return head, True, None
    except Exception as e:  # noqa: BLE001
        reason = _reason(e)
        _log(f'[score] aesthetic head unavailable ({reason}) — '
             'aesthetic scores skipped')
        return None, False, reason


def _reason(exc) -> str:
    """One short "<ExcType>: <message>" line, safe to show in the UI. Trimmed
    because some hub/urllib errors carry a multi-line body with a URL and a
    traceback hint, and this ends up inside a one-line activity sentence."""
    msg = ' '.join(str(exc).split())
    if len(msg) > 160:
        msg = msg[:157] + '…'
    return f'{type(exc).__name__}: {msg}' if msg else type(exc).__name__


def _load_nsfw(device):
    """((model, processor, nsfw_index), ok, reason). Marqo NSFW classifier;
    degrades to (None, False, why) so a fetch/load failure only drops the nsfw
    column — while still saying why (see _load_aesthetic_head)."""
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
        return (model, proc, nsfw_idx), True, None
    except Exception as e:  # noqa: BLE001
        reason = _reason(e)
        _log(f'[score] NSFW model unavailable ({reason}) — nsfw scores skipped')
        return None, False, reason


def _default_cache():
    from pathlib import Path
    return str(Path.home() / '.cache' / 'lds')


def _results_from_cache(images, cache, cached_only=False):
    """{path: {state, aesthetic?, nsfw?}} for the parent's write-back.

    ``cached_only`` drops the paths this run never reached — the cancel payload
    must describe the work that was PAID for and nothing else, so the parent can
    write it without blanking rows it knows nothing about."""
    out = {}
    for p in images:
        entry = cache.get(p)
        if entry is None:
            if cached_only:
                continue
            entry = ('error', None, None, None, '')
        digest = _cache_hash(entry)
        row = {'state': str(entry[0]),
               'fingerprint': digest.hex() if digest else None}
        if entry[1] is not None:
            row['aesthetic'] = float(entry[1])
        if entry[2] is not None:
            row['nsfw'] = float(entry[2])
        out[p] = row
    return out


def _incomplete(cache, path):
    """True when the cached entry scored fine but is MISSING a head's value — the
    signature of an image embedded while that head could not be downloaded."""
    entry = cache.get(path)
    return bool(entry) and entry[0] == 'ok' and (entry[1] is None or entry[2] is None)


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
    style_threshold = float(req.get('style_threshold') or 0.6)
    # The explicit "recompute everything" lane. Never the normal pass: the normal
    # pass resumes, and a button that silently ignored the cache would make every
    # relaunch cost the full bank again.
    rescore = bool(req.get('rescore'))

    # Before anything else: a previous run may have finished a cache write and
    # then lost the rename to a file lock (or been killed mid-flush). That work
    # is paid GPU time sitting in a temporary; claim it before deciding what is
    # left to do. Done even on the rescore lane, which is also where leftover
    # temporaries get swept instead of accumulating forever.
    _salvage_cache(cache_path)
    cache = {} if rescore else _load_cache(cache_path)
    # Re-score anything not cached OR whose file changed on disk since (a same-path
    # edit invalidates the stale embedding/scores).
    todo = [p for p in images if p not in cache or _is_stale(p, cache[p])]
    todo_set = set(todo)
    # Cached entries with a hole a head could not fill last time. Whether they are
    # really worth re-running depends on which heads load THIS time, so the list is
    # only a candidate list here (see ``retry`` below).
    holed = [p for p in images if p not in todo_set and _incomplete(cache, p)]
    reused = len(images) - len(todo)
    _write_count(cache_path, reused)
    _log(f'[score] {len(images)} image(s), {reused} cached')

    computed = fresh = 0
    # Why a head produced nothing, per head. Empty when every head loaded — and
    # when no work ran at all, since heads are only loaded for real work.
    head_errors = {}
    if todo or holed:
        try:
            import numpy as np  # noqa: F401
            import open_clip
            import torch
            from PIL import Image
        except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
            print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                              'error': import_report.import_failure(e)}))
            return 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Heads BEFORE CLIP, on purpose: whether a holed entry is worth re-running
        # is exactly "did the head that was missing come back?", and the answer
        # decides whether the ~1 GB CLIP load is needed at all. Loading CLIP first
        # would charge that load to every relaunch of a bank whose head is
        # permanently unavailable, for zero work.
        aes_head, aes_ok, aes_why = _load_aesthetic_head(models_root, device)
        nsfw_bundle, nsfw_ok, nsfw_why = _load_nsfw(device)
        head_errors = {k: v for k, v in (('aesthetic', aes_why),
                                         ('nsfw', nsfw_why)) if v}
        retry = [p for p in holed
                 if (aes_ok and cache[p][1] is None)
                 or (nsfw_ok and cache[p][2] is None)]
        retry_set = set(retry)
        work = todo + retry
        if retry:
            _log(f'[score] {len(retry)} cached image(s) missing a head that is '
                 f'available again — re-scoring those too')
        if not work:
            _log('[score] nothing to compute — every image is cached and complete')
    else:
        work = []

    if work:
        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='openai',
                cache_dir=(os.path.join(models_root, 'open_clip') if models_root else None))
            clip_model.to(device).eval()
        except Exception as e:  # noqa: BLE001
            print(json.dumps({'ok': False, 'results': {}, 'clusters': {},
                              'error': f'CLIP load failed: {type(e).__name__}: {e}'}))
            return 1
        zero = np.zeros(768, dtype='float32')
        done_since_save = 0
        if _cancel_requested(cancel_file):   # cancelled during the model load
            _write_count(cache_path, reused)
            print(json.dumps({'ok': True, 'cancelled': True,
                              'cached': reused, 'remaining': len(images) - reused,
                              'computed': 0, 'reused': reused,
                              'results': _results_from_cache(images, cache, True),
                              'clusters': None}))
            return 0
        for i, p in enumerate(work, 1):
            # A retried entry keeps the value of any head that is STILL down —
            # overwriting a good aesthetic score with None because the nsfw model
            # came back would trade one hole for another.
            keep = cache.get(p) if p in retry_set else None
            signature = ''
            payload_hash = b''
            changed_while_scoring = False
            try:
                # The bytes were validated from the open descriptor immediately
                # before this decode.  Never re-open the mutable Bank path.
                signature = _file_sig(p)
                payload = read_validated_bank_image(p)
                payload_hash = hashlib.sha256(payload).digest()
                if not signature or _file_sig(p) != signature:
                    changed_while_scoring = True
                    raise RuntimeError('image changed while it was read')
                with Image.open(io.BytesIO(payload)) as im:
                    im = im.convert('RGB')
                    with torch.no_grad():
                        tens = preprocess(im).unsqueeze(0).to(device)
                        emb = clip_model.encode_image(tens)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                        emb_np = emb.cpu().numpy()[0].astype('float32')
                        aesthetic = keep[1] if keep else None
                        if aes_ok:
                            aesthetic = round(float(aes_head(emb)[0][0].item()), 3)
                        nsfw = keep[2] if keep else None
                        if nsfw_ok:
                            model, proc, nsfw_idx = nsfw_bundle
                            inp = proc(images=im, return_tensors='pt').to(device)
                            logits = model(**inp).logits
                            probs = torch.softmax(logits, dim=-1)[0]
                            nsfw = round(float(probs[nsfw_idx].item()), 4)
                    result = ('ok', aesthetic, nsfw, emb_np, signature, payload_hash)
                # CLIP and the heads can take long enough for a live Bank path
                # to be replaced.  Publish only while it still identifies the
                # validated bytes that produced ``result``.
                if _file_sig(p) != signature:
                    changed_while_scoring = True
                    raise RuntimeError('image changed while it was scored')
                cache[p] = result
            except Exception as e:  # noqa: BLE001 — one broken file never sinks the pass
                if not signature or _file_sig(p) != signature:
                    changed_while_scoring = True
                # A hole-retry that fails KEEPS its entry: the embedding in it is
                # good work, and downgrading it to ('error', zero) because the file
                # is momentarily unreadable would drop the image out of the style
                # partition to fix nothing.
                if changed_while_scoring:
                    cache.pop(p, None)
                    if p in retry_set:
                        reused = max(reused - 1, 0)
                elif keep is None:
                    cache[p] = ('error', None, None, zero, signature, payload_hash)
                _log(f'[score] {i}/{len(work)} ERROR {e}')
                continue
            finally:
                if p in todo_set and p in cache:
                    fresh += 1
                computed += 1
                done_since_save += 1
                if cache_path and done_since_save >= CACHE_EVERY:
                    _flush_cache(cache_path, cache)
                    _write_count(cache_path, reused + fresh)
                    done_since_save = 0
            _log(f'[score] {i}/{len(work)} {cache[p][0]}')
            if _cancel_requested(cancel_file):   # clean stop between images
                if cache_path:
                    _flush_cache(cache_path, cache)
                _write_count(cache_path, reused + fresh)
                # results, NOT clusters. The scores here are paid GPU work and the
                # parent writes them; the style partition is 181 s of n² away and
                # the parent kills us after 15 s, so asking for it would lose both.
                print(json.dumps({
                    'ok': True, 'cancelled': True,
                    'cached': reused + fresh,
                    'remaining': len(images) - reused - fresh,
                    'computed': computed, 'reused': reused,
                    'results': _results_from_cache(images, cache, True),
                    'clusters': None}))
                return 0
        if cache_path:
            _phase(f'saving the score cache ({len(cache)} image(s))…')
            _flush_cache(cache_path, cache)
            _write_count(cache_path, reused + fresh)

    results = _results_from_cache(images, cache)
    # EVERYTHING BELOW IS MUTE WORK, and that is what these phase lines are for.
    # The per-image counter stopped at N/N when the last image was scored, and
    # the two steps after it (a compressed 70 MB cache write, then an n² pass over
    # every embedding) take MINUTES on a large bank with the bar sitting at 100 %.
    # A pass that says nothing while it works reads as a hung one — measured on a
    # 21 000-image bank: ~4 minutes of silence behind a full bar.
    _phase(f'grouping styles over {len(images)} image(s) — the slow tail of this '
           'pass; Stop now keeps every score already computed but discards the '
           'grouping, which can only be redone whole')
    clusters = _cluster_style(images, cache, style_threshold,
                              lambda: _cancel_requested(cancel_file))
    if clusters is None:
        # Stopped inside the clustering — everything is scored and cached, only the
        # partition is missing. Say so instead of dying under the watchdog kill.
        print(json.dumps({'ok': True, 'cancelled': True,
                          'cached': len(images), 'remaining': 0,
                          'computed': computed, 'reused': reused,
                          'results': _results_from_cache(images, cache, True),
                          'clusters': None}))
        return 0
    # One more mute step worth naming: this JSON carries a line per image, and
    # building it over tens of thousands of them is not instant either.
    _phase(f'handing {len(results)} result(s) back to the app…')
    print(json.dumps({'ok': True, 'results': results, 'clusters': clusters,
                      'computed': computed, 'reused': reused,
                      'head_errors': head_errors}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
