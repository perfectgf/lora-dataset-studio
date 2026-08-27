"""🎨 The LAION aesthetic head over vectors that ALREADY EXIST — no decode, no CLIP.

The cheapest member of this folder, and the reason it can exist at all is that
the video lane already embedded three frames of every shot into the SAME space
the image bank's ✨ Score pass uses: open_clip ViT-L/14 (openai), 768-d,
L2-normalised. The improved-aesthetic-predictor head is a 768→1 MLP over exactly
that vector, so rating a whole bank is one small matmul over an .npz that is
already on disk — no video is opened, no frame is decoded, CLIP is never loaded.

WHY IT IS A SEPARATE SCRIPT AND NOT A BRANCH OF bank_score_infer.py. That one
takes IMAGE PATHS: it opens files, runs CLIP, runs the NSFW classifier and
maintains its own embedding cache. Every one of those steps is exactly what this
must not do. What the two DO share is the head itself, and that is imported from
there rather than copied — one URL, one filename, one MLP shape, one cache
directory. A second copy of those four facts is how a video score and an image
score end up on two different scales while both look plausible.

THE CARD IS NEVER TAKEN. This is ~1 GFLOP per thousand frames; CUDA is hidden
before torch is imported so a bank can be rated while a training run owns the
GPU. The only real cost is the torch import itself, which is why the parent skips
this whole subprocess when every shot already has a score.

Protocol (one JSON line in, one JSON line out — this is a batch, not a worker):
  stdin  : {"store": "<abs path to a clip_embeddings.npz>",
            "models_root": path|null}
  stdout : {"ok": true, "scores": {"<clip_id>": [float, ...]}, "frames": N}
           {"ok": false, "error": "<ExcType>: <message>"}
  stderr : "[look] …" progress lines; the parent does not parse them.

PER-FRAME SCORES, NOT A PER-CLIP ONE. How several frames become one number is a
product decision with an argument attached (see `video_metrics.aesthetic_of`),
and it belongs in the pure module the tests can exercise without torch — not in
a subprocess nobody can run in CI.
"""
from __future__ import annotations
import json
import os
import sys
# A ._pth-pinned interpreter (ComfyUI portable's python_embeded) does not put
# this script's directory on sys.path — restore it or the import below dies
# there. See _harness.py for the whole story.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _harness import _emit, _log

# Hidden BEFORE torch is imported — the only thing that actually keeps this off
# a card a training run is using. Same two-locks-on-one-door reflex as the frame
# encoder, except here there is no "auto" branch at all: this pass has no use
# for a GPU on any bank size.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402
import bank_score_infer  # noqa: E402 — the head, imported rather than copied

# The arrays this reads out of the store `video_clip_search.save_embeddings`
# writes. Named here so a reader sees the borrowed contract, and pinned against
# that writer by tests/test_video_aesthetic.py — the store also carries `labels`
# and `times`, which a per-frame score has no use for.
STORE_ARRAYS = ('clip_ids', 'vecs')

# Rows per forward pass. The head is tiny, but a 30 000-frame bank in one tensor
# is a 90 MB float32 allocation plus its activations for no gain.
BLOCK = 2048


def main() -> int:
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        _emit({'ok': False, 'error': f'bad json: {e}'})
        return 1
    store = req.get('store') or ''
    models_root = req.get('models_root') or None
    if not store or not os.path.isfile(store):
        # Not an error the user should ever see: the parent only launches this
        # when the embed pass has written a store. Answered rather than crashed
        # so the caller gets one sentence instead of a traceback on stderr.
        _emit({'ok': False, 'error': 'no embedding store at that path'})
        return 1

    try:
        import numpy as np
        import torch
    except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
        _emit({'ok': False, 'error': import_report.import_failure(e)})
        return 1

    try:
        with np.load(store, allow_pickle=False) as z:
            missing = [k for k in STORE_ARRAYS if k not in z.files]
            if missing:
                raise ValueError(f'embedding store has no {", ".join(missing)}')
            ids = np.asarray(z['clip_ids'])
            vecs = np.asarray(z['vecs'], dtype='float32')
        if len(ids) != len(vecs):
            raise ValueError('embedding store arrays are not row-aligned')
    except Exception as e:  # noqa: BLE001
        _emit({'ok': False, 'error': f'unreadable store: {type(e).__name__}: {e}'})
        return 1

    if not len(ids):
        _emit({'ok': True, 'scores': {}, 'frames': 0})
        return 0

    head, ok, why = bank_score_infer._load_aesthetic_head(models_root, 'cpu')
    if not ok:
        # The head is a one-time download from a public repo. On a machine with
        # no egress this is the whole reason every score comes back empty, so
        # the reason travels — "unavailable" alone leaves a finished pass, no
        # scores, and nothing to act on. Same sentence the image lane surfaces.
        _emit({'ok': False, 'error': why or 'aesthetic head unavailable'})
        return 1

    # RE-NORMALISED here rather than trusted from the store. The encoder writes
    # what it normalised, but this head was fitted on unit-length ViT-L/14
    # features and a single un-normed row would come back with a confident,
    # meaningless rating — the same defensive reflex as video_clip_dedup._matrix.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-8)

    scores = {}
    _log(f'[look] rating {len(vecs)} frame(s)')
    with torch.no_grad():
        for start in range(0, len(vecs), BLOCK):
            block = torch.from_numpy(vecs[start:start + BLOCK])
            out = head(block).squeeze(-1).tolist()
            for offset, value in enumerate(out):
                cid = str(int(ids[start + offset]))
                scores.setdefault(cid, []).append(round(float(value), 3))
    _emit({'ok': True, 'scores': scores, 'frames': int(len(vecs))})
    return 0


if __name__ == '__main__':
    sys.exit(main())
