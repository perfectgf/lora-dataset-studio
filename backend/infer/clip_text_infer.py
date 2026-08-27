"""CLIP text tower — turn a written query into the SAME 768-d space the ✨ Score
pass already put every bank image in, run in the dedicated "bank scoring" ML
interpreter (torch/open_clip are not in the Flask venv).

This is the missing half of bank_score_infer.py. That script loads CLIP
ViT-L/14 and calls `encode_image`; the model it loads carries a text tower too,
which nothing has ever used. Here we load the same checkpoint and call
`encode_text` instead — so "brunette outdoors, wide shot" becomes a vector that
can be cosine-compared against the cached image embeddings with no new model, no
new download and no new dependency.

⚠️ THE MODEL SPEC IS A CONTRACT, NOT A CHOICE.
``MODEL_NAME``/``PRETRAINED`` below MUST stay byte-identical to what
``bank_score_infer.py`` passes to ``create_model_and_transforms``. Two vectors
produced by different CLIP configurations are not comparable: the dot product
still returns plausible-looking numbers and a ranking that means nothing — a
silent failure with no error and no crash, invisible on "roughly right" results.
``backend/tests/test_clip_text_model_contract.py`` fails if the two drift apart.

Known, deliberate consequence: open_clip warns that the `openai` ViT-L-14 weights
were trained with QuickGELU while `'ViT-L-14'` (no `-quickgelu` suffix) builds a
plain-GELU model. Every cached bank embedding in the wild was produced that way,
so the text side reproduces it EXACTLY. Fixing the activation is a separate
migration (it invalidates every scored bank), never a side effect of this file.

Protocol — a WARM WORKER, because loading CLIP costs ~8 s and encoding a
sentence costs ~20 ms (measured, CPU). The parent starts one, keeps it for a
short idle window, and pays the load once per session rather than once per
search:

  stdout, once ready : {"ok": true, "ready": true, "dim": 768}
  stdin,  per query  : {"text": "brunette outdoors, wide shot"}\n
  stdout, per query  : {"ok": true, "vector": [768 floats]}\n
                       {"ok": false, "error": "..."}\n
  EOF on stdin       : clean exit.
A fatal load failure prints {"ok": false, "ready": false, "error": ...} and exits.

CPU ON PURPOSE. The text tower is small and one sentence is microseconds of
compute — the entire cost is the model LOAD. Taking the GPU for that would make
a search race the training run and the captioning pass this project already
serialises against each other, for no speed the user could notice. So CUDA is
hidden from the child outright and the parent never takes a GPU window.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402

# MUST match bank_score_infer.py — see the contract note above.
MODEL_NAME = 'ViT-L-14'
PRETRAINED = 'openai'


def main() -> int:
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        _emit({'ok': False, 'ready': False, 'error': f'bad json: {e}'})
        return 1
    models_root = req.get('models_root') or None

    # Hidden BEFORE torch is imported — this is what actually keeps a text search
    # off the card while a LoRA trains.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    try:
        import open_clip
        import torch
    except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
        _emit({'ok': False, 'ready': False,
               'error': import_report.import_failure(e)})
        return 1

    _log(f'[text] loading CLIP {MODEL_NAME}/{PRETRAINED} (CPU)…')
    try:
        # Same call, same cache_dir as bank_score_infer.py, so both halves resolve
        # the SAME checkpoint out of the SAME directory — a bank that has been
        # scored has already paid this download.
        model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED,
            cache_dir=(os.path.join(models_root, 'open_clip') if models_root else None))
        model.to('cpu').eval()
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    except Exception as e:  # noqa: BLE001
        _emit({'ok': False, 'ready': False,
               'error': f'CLIP load failed: {type(e).__name__}: {e}'})
        return 1

    def _encode(text):
        with torch.no_grad():
            emb = model.encode_text(tokenizer([text]))
            # L2-normalised exactly like the image side, so a dot product IS the
            # cosine and the two halves stay comparable.
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy()[0].astype('float32')

    dim = int(model.text_projection.shape[-1]) if hasattr(model, 'text_projection') \
        and getattr(model.text_projection, 'shape', None) is not None else 768
    _emit({'ok': True, 'ready': True, 'dim': dim})
    _log('[text] ready')

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({'ok': False, 'error': f'bad json: {e}'})
            continue
        text = str(msg.get('text') or '')
        if not text.strip():
            _emit({'ok': False, 'error': 'empty text'})
            continue
        try:
            vec = _encode(text)
        except Exception as e:  # noqa: BLE001 — one bad query never kills the worker
            _emit({'ok': False, 'error': f'text encoding failed: {type(e).__name__}: {e}'})
            continue
        _emit({'ok': True, 'vector': [float(x) for x in vec]})
    _log('[text] stdin closed — exiting')
    return 0


if __name__ == '__main__':
    sys.exit(main())
