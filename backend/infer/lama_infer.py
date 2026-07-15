"""Watermark inpainting — simple-lama-inpainting (LaMa), lance par l'interprete ML
DEDIE (le paquet est absent du venv Flask). Meme pattern subprocess que
face_score_infer.py / mask_infer.py.

stdin  : {"image_path": path, "bboxes": [[x1, y1, x2, y2], ...]}
         ou l'ancien champ "bbox" (coordonnees normalisees [0,1])
stdout : DERNIERE ligne = JSON {"ok": bool, "error"?: str}
Logs -> stderr.

LaMa est NON-generatif : seuls les pixels du rectangle masque changent. Le device
est fourni par l'appelant, qui prend la fenetre GPU exclusive quand CUDA est choisi.
"""
import json
import math
import os
import sys

def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def build_mask(size, bboxes):
    """Construit un masque binaire unique couvrant tous les rectangles normalises."""
    from PIL import Image, ImageDraw

    width, height = size
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in bboxes:
        left = max(0, min(width - 1, int(x1 * width)))
        top = max(0, min(height - 1, int(y1 * height)))
        right = max(left + 1, min(width, int(math.ceil(x2 * width))))
        bottom = max(top + 1, min(height, int(math.ceil(y2 * height))))
        draw.rectangle((left, top, right - 1, bottom - 1), fill=255)
    return mask


def _payload_bboxes(req):
    if 'bboxes' in req:
        raw_bboxes = req['bboxes']
    else:
        raw_bbox = req['bbox']
        if not isinstance(raw_bbox, (list, tuple)):
            raise ValueError('bbox must have 4 values')
        bbox = [float(value) for value in raw_bbox]
        if len(bbox) != 4:
            raise ValueError('bbox must have 4 values')
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError('bbox values must be finite')
        x1, y1, x2, y2 = bbox
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        raw_bboxes = [[
            max(0.0, min(1.0, left)),
            max(0.0, min(1.0, top)),
            max(0.0, min(1.0, right)),
            max(0.0, min(1.0, bottom)),
        ]]
    if not isinstance(raw_bboxes, list) or not raw_bboxes:
        raise ValueError('bboxes must be a non-empty list')

    bboxes = []
    for index, raw_bbox in enumerate(raw_bboxes):
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise ValueError(f'bboxes[{index}] must have 4 values')
        bbox = [float(value) for value in raw_bbox]
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError(f'bboxes[{index}] values must be finite')
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f'bboxes[{index}] must be normalized and ordered')
        bboxes.append(bbox)
    return bboxes


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw) if raw.strip() else {}
        requested_device = str(req.get('device') or 'cpu').lower()
        if requested_device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('device must be auto, cuda or cpu')
        raw_jobs = req.get('jobs')
        if raw_jobs is None:
            raw_jobs = [{'image_path': req['image_path'], 'bboxes': _payload_bboxes(req)}]
            batch = False
        else:
            if not isinstance(raw_jobs, list) or not raw_jobs:
                raise ValueError('jobs must be a non-empty list')
            raw_jobs = [{'image_path': job['image_path'], 'bboxes': _payload_bboxes(job)} for job in raw_jobs]
            batch = True
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"payload: {e}"}))
        return 1
    if requested_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        from PIL import Image
        import torch
        from simple_lama_inpainting import SimpleLama
    except Exception as e:
        # import KO (paquet absent / torch casse) -> JSON propre, pas de traceback muet.
        print(json.dumps({"ok": False, "error": f"import: {type(e).__name__}: {e}"}))
        return 1
    try:
        cuda = bool(torch.cuda.is_available())
        if requested_device == 'cuda' and not cuda:
            raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
        actual_device = 'cuda' if requested_device in ('auto', 'cuda') and cuda else 'cpu'
        lama = SimpleLama()
        results = []
        for job in raw_jobs:
            image_path = job['image_path']
            try:
                if not image_path or not os.path.isfile(image_path):
                    raise FileNotFoundError('image not found')
                img = Image.open(image_path).convert('RGB')
                W, H = img.size
                mask = build_mask((W, H), job['bboxes'])
                _log(f"[lama] inpaint {W}x{H} boxes={len(job['bboxes'])} device={actual_device}")
                result = lama(img, mask)
                result.convert('RGB').save(image_path, 'WEBP', quality=92)
                results.append({'image_path': image_path, 'ok': True})
            except Exception as e:
                results.append({'image_path': image_path, 'ok': False,
                                'error': f'{type(e).__name__}: {e}'})
        if batch:
            print(json.dumps({'ok': True, 'device': actual_device, 'results': results}))
        elif results[0]['ok']:
            print(json.dumps({'ok': True, 'device': actual_device}))
        else:
            print(json.dumps({'ok': False, 'error': results[0]['error']}))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
