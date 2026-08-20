"""CLIP image tower as a WARM WORKER — embed frames one at a time, load once.

The third member of the CLIP family in this project, and the one that closes the
loop for video. ``bank_score_infer.py`` embeds a fixed LIST of images in one shot
(it also runs the aesthetic and NSFW heads, which is why it is a batch job);
``clip_text_infer.py`` is a warm worker for the TEXT tower. This is the image
tower, in the warm-worker shape.

WHY WARM RATHER THAN BATCH. The video embedding pass extracts frames itself, in
the Flask process, because PyAV lives there and not in the ML interpreter. Frames
therefore arrive as they are decoded, and a batch script would force the parent
to choose between two bad options: decode the WHOLE bank to a temp folder first
(gigabytes of JPEGs before a single vector exists, and nothing resumable), or pay
the ~8 s model load once per batch. A warm worker lets the parent decode a shot,
embed it, commit it, and move on — which is exactly the per-clip resume contract
every other pass in this lane already keeps.

⚠️ THE MODEL SPEC IS A CONTRACT, NOT A CHOICE.
``MODEL_NAME``/``PRETRAINED`` MUST stay byte-identical to what
``bank_score_infer.py`` and ``clip_text_infer.py`` use. Two vectors from
different CLIP configurations are not comparable: the dot product still returns
plausible numbers and a ranking that means nothing — no error, no crash, and
invisible on results that look roughly right. A video search ranks these vectors
against a query encoded by the TEXT worker, so a drift here silently breaks the
whole feature. ``backend/tests/test_clip_text_model_contract.py`` fails if the
three files disagree.

Protocol:
  stdin,  once      : {"models_root": path|null, "device": "auto"|"cpu"}
  stdout, once ready: {"ok": true, "ready": true, "dim": 768, "device": "cuda"}
  stdin,  per frame : {"image": "abs path to a jpg"}\n
  stdout, per frame : {"ok": true, "vector": [768 floats]}\n
                      {"ok": false, "error": "..."}\n   ← one bad frame, worker lives
  EOF on stdin      : clean exit.
A fatal load failure prints {"ok": false, "ready": false, "error": ...} and exits.

THE GPU IS OPTIONAL AND ASKED FOR. Unlike the text tower (small, and its whole
cost is the load), embedding thousands of frames is real compute: ~336 ms/frame
on the CPU against ~15 ms on a card, measured on this exact checkpoint. So the
parent decides — it takes the GPU-exclusive window when it is going to use the
card, and passes device="cpu" when it is not, in which case CUDA is hidden from
the child before torch is imported so a pass can never quietly grab the card a
training run is using.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402

# MUST match bank_score_infer.py and clip_text_infer.py — see the contract note.
MODEL_NAME = 'ViT-L-14'
PRETRAINED = 'openai'


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _emit(obj):
    print(json.dumps(obj), flush=True)


def main() -> int:
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        _emit({'ok': False, 'ready': False, 'error': f'bad json: {e}'})
        return 1
    models_root = req.get('models_root') or None
    want = str(req.get('device') or 'auto').lower()

    # Hidden BEFORE torch is imported — this is what actually keeps an embedding
    # pass off the card when the parent did not take the GPU window for it.
    if want == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    try:
        import open_clip
        import torch
        from PIL import Image
    except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
        _emit({'ok': False, 'ready': False,
               'error': import_report.import_failure(e)})
        return 1

    device = 'cuda' if (want != 'cpu' and torch.cuda.is_available()) else 'cpu'
    _log(f'[embed] loading CLIP {MODEL_NAME}/{PRETRAINED} ({device})…')
    try:
        # Same call, same cache_dir as the other two: one download, one checkpoint.
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED,
            cache_dir=(os.path.join(models_root, 'open_clip') if models_root else None))
        model.to(device).eval()
    except Exception as e:  # noqa: BLE001
        _emit({'ok': False, 'ready': False,
               'error': f'CLIP load failed: {type(e).__name__}: {e}'})
        return 1

    def _encode(path):
        with Image.open(path) as im:
            im = im.convert('RGB')
            with torch.no_grad():
                emb = model.encode_image(preprocess(im).unsqueeze(0).to(device))
                # L2-normalised exactly like the other two towers, so a dot
                # product IS the cosine and all three stay comparable.
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.cpu().numpy()[0].astype('float32')

    dim = int(getattr(model.visual, 'output_dim', 0) or 768)
    _emit({'ok': True, 'ready': True, 'dim': dim, 'device': device})
    _log('[embed] ready')

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({'ok': False, 'error': f'bad json: {e}'})
            continue
        path = str(msg.get('image') or '')
        if not path:
            _emit({'ok': False, 'error': 'no image path'})
            continue
        try:
            vec = _encode(path)
        except Exception as e:  # noqa: BLE001 — one bad frame never kills the worker
            _emit({'ok': False, 'error': f'{type(e).__name__}: {e}'})
            continue
        _emit({'ok': True, 'vector': [float(x) for x in vec]})
    _log('[embed] stdin closed — exiting')
    return 0


if __name__ == '__main__':
    sys.exit(main())
