"""🤖 D3's second-order temporal statistic, over one 2-second window of a shot.

WHAT THIS COMPUTES, and it is ten lines of arithmetic. Each frame of a short
contiguous window is encoded independently into one vector; the L2 distance
between consecutive vectors is how much the picture CHANGED between those two
instants; the difference of those distances is how much that rate of change
itself changed. Real footage varies that second quantity constantly — a hand
shakes, a subject accelerates, light flickers, the sensor is noisy. A generated
clip, on the evidence of the 2023-24 generators the method was measured on,
does not: its motion is smoother than the world's.

    D3: Training-free AI-Generated Video Detection Using Second-Order Features
    (Chende Zheng et al., ICCV 2025) — arXiv:2508.00701,
    github.com/Zig-HS/D3, MIT. Formula (8), and models/D3_model.py.

PORTED RATHER THAN DEPENDED ON. The reference repo is a research harness: it
pins numpy 2.3.3 (this app's ceiling is <2, for insightface's ABI), it drags
moviepy 1.x to read a duration ffprobe already knows, and its albumentations pin
is incompatible with its own augmentation call. MIT lets us take the ten lines
that matter and leave the harness, so that is what this is — the arithmetic,
under this app's own dependency floors.

THE REDUCTION ORDER IS THE PART TO GET RIGHT, and the intuitive reading of the
paper's prose gets it wrong. It is NOT the std of the second differences of the
feature VECTORS. The vector is collapsed to a SCALAR distance first, and the
differencing happens on that 1-D series:

    f[t]        per-frame feature, 768-d, NOT normalised
    d[k]        = ||f[k+1] - f[k]||          k = 0 .. T-2     (T-1 scalars)
    a[j]        = d[j+1] - d[j]              j = 0 .. T-3     (T-2 scalars)
    score       = std(a), Bessel-corrected

Differencing the vectors and taking a std per dimension is a different
statistic, unvalidated by anybody, and it looks equally plausible in a diff.
`test_video_ai_check.py` pins this order against hand-computed values.

WHY XCLIP-B/16 AND NOT PLAIN CLIP, which would be simpler. The reference
measures ten encoders and reports that the CHOICE barely matters on clean
footage (CLIP-B/16 97.00 mAP against XCLIP-B/16's 97.72 on GenVideo). It stops
being true the moment the footage has been re-encoded, which for a scraped bank
is always: under JPEG requantisation at q60 CLIP-B/16 falls from 97.82 to 78.81
while XCLIP-B/16 goes 98.46 → 94.50 (their Table 6). The authors' own reading is
that second-order features are only as stable as the feature space under them.
Our corpus is re-encoded by construction, so that column decides it.

⚠️ XCLIP IS NOT A PER-FRAME ENCODER, which is a real property of this port and
not a detail. Every layer of `XCLIPVisionModel` passes a "message token" between
frames — `transformers` calls the block `CrossFrameAttentionBlock` — in groups of
exactly `config.num_frames` CONSECUTIVE items of the batch (8 for this
checkpoint). So a frame's vector depends on its neighbours, and:

  * the frame count MUST be a multiple of 8 or the reshape raises. That is why
    the reference hardcodes 8-or-16 and why FRAMES is fixed at 16 upstream.
  * several clips may share one forward pass ONLY while each contributes a
    multiple of 8 frames — otherwise a group would straddle two clips and mix
    one shot's motion into another's. `_encode` asserts it rather than trusting
    the caller; `test_batching_never_straddles_two_clips` pins it.

⚠️ AND THE TOWER MUST BE TAKEN OFF THE WHOLE MODEL, never loaded on its own. The
checkpoint stores its vision tensors under a `vision_model.` prefix, so the bare
`XCLIPVisionModel.from_pretrained(id)` matches nothing and hands back a RANDOM
encoder — silently, with plausible weight statistics and plausible-looking
scores. Measured, and guarded at load time; see `main`.

The preprocessing reproduces the reference EXACTLY, including one thing that is
plainly a bug: it reads frames with `cv2.imread` (BGR) and hands them to a
CLIP-family encoder trained on RGB, then normalises with ImageNet statistics
rather than CLIP's own. Both are wrong on their face.

THE FIX WAS MEASURED AND NOT TAKEN, which is the honest version of this comment.
On ten forged clips built here — five constant-velocity pans against five of the
same scenes at the same mean speed with per-frame jitter, sensor noise and
exposure flicker, all encoded to H.264 and read back through the real pass:

    BGR + ImageNet (the reference)   AUC 0.840   smooth 0.657 / handheld 0.866
    RGB + ImageNet                   AUC 0.840   smooth 0.607 / handheld 0.981
    RGB + CLIP's own statistics      AUC 0.640   smooth 0.729 / handheld 0.934

So "obviously wrong" is not obviously better: the colour swap changes nothing
here, and using CLIP's own statistics — the most defensible-looking of the three
— is the one that clearly hurts. Ten synthetic clips are not a validation set
either way. What IS on the other side of the scale is 40 subsets of real
generators, and the compression-robustness table this port was chosen for, all
measured on BGR. Deviating would keep the citation while dropping the only thing
the citation guarantees. So the port stays faithful, the measurement is written
down, and changing it is a one-line edit waiting on a labelled corpus this app
does not have. Pinned by a test so it cannot drift by accident.

⚠️ These numbers were taken with the encoder's weights VERIFIED as loaded (0
missing keys). Anyone re-running this must do the same: a randomly initialised
XCLIP still separates these two classes, so a comparison made without that check
measures the forgery and not the encoder.

Protocol (one JSON line in, one JSON line out — a batch, not a warm worker):
  stdin  : {"clips": [{"id": 12, "frames": ["<abs jpeg>", ...]}, ...],
            "model": "microsoft/xclip-base-patch16", "models_root": path|null}
  stdout : {"ok": true, "steps": {"12": [d0, ...]}, "clips": N, "device": "cpu"}
           {"ok": false, "error": "<ExcType>: <message>"}
  stderr : "[aicheck] …" progress lines; the parent does not parse them.

STEPS OUT, NOT A SCORE. The child returns the per-adjacent-pair distances and
stops there. Turning T-1 distances into one number per shot is a product
decision with an argument attached (see `video_ai_check.irregularity`), and it
belongs in the pure module a test can exercise without torch — the same split
`video_aesthetic_infer` makes with `video_metrics.aesthetic_of`.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402

# Hidden BEFORE torch is imported, exactly like the look score's child. This is a
# 16-frame forward pass over a base-sized ViT; it is not worth interrupting a
# training run for, and the parent takes no GPU window on its behalf.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# The reference's preprocessing constants, named rather than inlined so the test
# that pins them against the paper has something to read.
#
# IMAGENET, not CLIP's own (0.4815, 0.4578, 0.4082)/(0.2686, 0.2613, 0.2758) —
# see the module docstring. `albumentations.Normalize(..., max_pixel_value=255)`
# is (x/255 - mean) / std, which is what `_preprocess` computes.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Frames per forward pass. The cross-frame block groups by config.num_frames, so
# this is a count of GROUPS: it must stay a multiple of the frames-per-clip the
# parent sends (16), which is why it is expressed that way rather than as a
# round number somebody would later "tidy" to 50.
CLIPS_PER_FORWARD = 4


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _emit(obj):
    print(json.dumps(obj), flush=True)


def _preprocess(paths):
    """[T, 3, H, W] float32, the reference's own pipeline.

    BGR because `cv2.imread` is BGR and the reference never converts; ImageNet
    statistics because that is what its albumentations transform uses. Both are
    reproduced on purpose — see the module docstring.

    The geometry (centre crop, 224x224 resize) happened in the PARENT, where it
    is pure arithmetic a test can read without PIL or torch. This function only
    turns JPEGs into a normalised tensor."""
    import numpy as np
    from PIL import Image

    rows = []
    for path in paths:
        with Image.open(path) as im:
            arr = np.asarray(im.convert('RGB'), dtype=np.float32)
        # RGB -> BGR. One line, and the whole reason it is a line rather than
        # nothing: see the docstring. `.copy()` because the negative stride a
        # reversed view carries is not something torch will accept.
        arr = arr[:, :, ::-1].copy()
        arr /= 255.0
        arr -= np.asarray(IMAGENET_MEAN, dtype=np.float32)
        arr /= np.asarray(IMAGENET_STD, dtype=np.float32)
        rows.append(arr.transpose(2, 0, 1))
    return np.stack(rows)


def step_distances(features):
    """[||f[k+1] - f[k]||] for a [T, D] array — the FIRST-order series.

    Pure numpy and module-level on purpose: this is half the statistic, and a
    test can forge features and check it without torch anywhere near the
    machine. The parent owns the second half (`video_ai_check.irregularity`).

    L2 and not cosine, which the reference also implements. Cosine is bounded and
    therefore comparable across encoders — but the authors measure it 6 points
    worse on GenVideo (91.30 against 97.72), and this app pins ONE encoder, so
    the comparability buys nothing here and the accuracy is real."""
    import numpy as np
    f = np.asarray(features, dtype='float32')
    if f.ndim != 2 or len(f) < 2:
        return []
    return [float(v) for v in np.linalg.norm(f[1:] - f[:-1], axis=1)]


def _encode(model, batch, per_clip):
    """[T, 768] pooled features for a stacked batch of several clips' frames.

    `per_clip` is the frame count each clip contributed. Every one of them must
    be a multiple of the model's `num_frames`, or the cross-frame message block
    would mix two shots inside one group — see the module docstring."""
    num_frames = int(getattr(model.config, 'num_frames', 8) or 8)
    bad = [n for n in per_clip if n % num_frames]
    if bad:
        raise ValueError(f'frames per clip must be a multiple of {num_frames}, '
                         f'got {sorted(set(bad))}')
    # Imported AFTER the guard on purpose: a mis-batched call is a programming
    # error the parent must hear about identically on every machine, and torch is
    # absent from CI. Importing first turned that ValueError into a
    # ModuleNotFoundError and made the contract untestable without the GPU stack.
    import torch

    with torch.no_grad():
        out = model(pixel_values=torch.from_numpy(batch))
    # `.pooler_output` is post_layernorm(CLS) — 768-d, NOT projected into the
    # joint text space and NOT L2-normalised. Both are the reference's choice and
    # both matter: a normalised feature would make every step distance a chord on
    # the unit sphere and flatten exactly the magnitude this measures.
    return out.pooler_output.detach().cpu().numpy()


def main() -> int:
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        _emit({'ok': False, 'error': f'bad json: {e}'})
        return 1
    clips = req.get('clips') or []
    model_id = (req.get('model') or '').strip()
    models_root = req.get('models_root') or None
    if not model_id:
        _emit({'ok': False, 'error': 'no model id'})
        return 1
    if not clips:
        _emit({'ok': True, 'steps': {}, 'clips': 0, 'device': 'cpu'})
        return 0

    try:
        import numpy as np  # noqa: F401 — used by _preprocess/step_distances
        import torch  # noqa: F401 — a probe: a missing torch must answer as import_failure here, not as a transformers traceback later
        from transformers import XCLIPModel
    except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
        _emit({'ok': False, 'error': import_report.import_failure(e)})
        return 1

    try:
        kwargs = {'cache_dir': models_root} if models_root else {}
        # ⚠️ THE WHOLE MODEL IS LOADED AND THE VISION TOWER TAKEN OFF IT, rather
        # than `XCLIPVisionModel.from_pretrained(id)` — which is the obvious call,
        # is what the reference does, and IS SILENTLY WRONG HERE.
        #
        # The published checkpoint is a COMPOSITE: every vision tensor is stored
        # under a `vision_model.` prefix, next to a text tower and a temporal
        # head. Handed to the bare vision class, those keys match nothing.
        # Measured on transformers 5.14.1, against the raw safetensors:
        #
        #   XCLIPVisionModel.from_pretrained(id, config=vision_config)
        #                                     -> 343 missing keys, RANDOM weights
        #   XCLIPVisionModel.from_pretrained(id)
        #                                     -> 343 missing keys, RANDOM weights
        #   XCLIPModel.from_pretrained(id).vision_model
        #                                     -> 0 missing keys, matches the file
        #
        # And it does not raise, warn usefully, or look wrong downstream: a
        # freshly initialised ViT has entirely plausible weight statistics
        # (patch-embedding std 0.0200 against the checkpoint's 0.0189) and still
        # separates jittery motion from smooth motion, because almost any
        # projection preserves some of that structure. Two forged clips scored
        # 0.09 against 0.28 through the RANDOM encoder — a result that reads as
        # a working feature. Nothing but a comparison against the checkpoint
        # itself catches this, which is why the guard below is not optional.
        #
        # WHY THE REFERENCE GETS AWAY WITH IT: it pins `transformers==4.57.0`,
        # and on 4.57.6 the bare vision class does resolve the prefix — measured,
        # 0 missing keys. Our floor is `transformers>=4.57` with no ceiling
        # (setup_installer._bank_scoring_specs), so a new install today gets 5.x
        # and the same call silently stops working. Porting arithmetic out of a
        # pinned research harness means the pins do not come with it, and this is
        # what that costs. The call below is correct on BOTH versions, which is
        # why it is preferred to adding a ceiling nothing else here wants.
        #
        # The text tower and temporal head are instantiated and dropped. They
        # cost no extra download — the checkpoint is one file either way — only
        # a moment of RAM, which is the price of loading weights that are real.
        model, info = XCLIPModel.from_pretrained(
            model_id, output_loading_info=True, **kwargs)
        missing = info.get('missing_keys') or ()
        if missing:
            # REFUSED rather than reported: an encoder whose weights did not
            # arrive produces numbers, and they are indistinguishable from
            # measurements. A pass that stops with a reason costs a run; one that
            # stores noise costs the user's trust in every score in the bank.
            raise RuntimeError(
                f'{len(missing)} weight(s) of {model_id} did not load, so the '
                f'encoder is randomly initialised and every score would be '
                f'noise (first: {sorted(missing)[0]})')
        model = model.vision_model
        model.eval()
    except Exception as e:  # noqa: BLE001
        # The weights are a first-run download from a public repo. On a machine
        # with no egress this is the whole reason every shot comes back
        # unmeasured, so the reason travels — "unavailable" alone leaves a
        # finished pass, no scores, and nothing to act on.
        _emit({'ok': False, 'error': f'could not load {model_id}: '
                                     f'{type(e).__name__}: {e}'})
        return 1

    steps = {}
    _log(f'[aicheck] {len(clips)} clip(s) on cpu')
    for start in range(0, len(clips), CLIPS_PER_FORWARD):
        group = clips[start:start + CLIPS_PER_FORWARD]
        try:
            tensors = [_preprocess(c.get('frames') or []) for c in group]
            batch = np.concatenate(tensors)
            features = _encode(model, batch, [len(t) for t in tensors])
        except Exception as e:  # noqa: BLE001 — one group never sinks the pass
            # Reported per group and NOT fatal: the clips of this group simply
            # come back without an entry, which the parent already reads as "put
            # it back in the queue". A whole bank must not be lost to one
            # unreadable JPEG.
            _log(f'[aicheck] group at {start} failed: {type(e).__name__}: {e}')
            continue
        offset = 0
        for clip, tensor in zip(group, tensors):
            take = len(tensor)
            steps[str(clip.get('id'))] = step_distances(
                features[offset:offset + take])
            offset += take
    _emit({'ok': True, 'steps': steps, 'clips': len(steps), 'device': 'cpu'})
    return 0


if __name__ == '__main__':
    sys.exit(main())
