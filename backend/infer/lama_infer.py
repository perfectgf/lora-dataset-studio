"""Watermark inpainting — simple-lama-inpainting (LaMa), lance par l'interprete ML
DEDIE (le paquet est absent du venv Flask). Meme pattern subprocess que
face_score_infer.py / mask_infer.py.

stdin  : {"image_path": path, "bbox": [x1, y1, x2, y2]}  (bbox normalise [0,1])
stdout : DERNIERE ligne = JSON {"ok": bool, "error"?: str}
Logs -> stderr.

LaMa est NON-generatif : seuls les pixels du rectangle masque changent, le reste de
l'image est bit-pour-bit conserve. Tourne sur CPU (CUDA_VISIBLE_DEVICES vide ci-dessous)
pour ne PAS entrer en concurrence GPU avec ComfyUI/un training — c'est l'appelant qui
garantit que LaMa tourne HORS de la fenetre vision.
"""
import json
import os
import sys

# Forcer le CPU AVANT tout import torch : SimpleLama() choisit cuda si torch.cuda.
# is_available(), or on veut rester CPU (hors fenetre GPU). Vider la variable rend le
# GPU invisible a torch -> is_available() False -> device cpu.
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw) if raw.strip() else {}
        image_path = req['image_path']
        bbox = [float(v) for v in req['bbox']]
        if len(bbox) != 4:
            raise ValueError('bbox must have 4 values')
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"payload: {e}"}))
        return 1
    if not image_path or not os.path.isfile(image_path):
        print(json.dumps({"ok": False, "error": "image not found"}))
        return 1
    try:
        from PIL import Image, ImageDraw
        from simple_lama_inpainting import SimpleLama
    except Exception as e:
        # import KO (paquet absent / torch casse) -> JSON propre, pas de traceback muet.
        print(json.dumps({"ok": False, "error": f"import: {type(e).__name__}: {e}"}))
        return 1
    try:
        img = Image.open(image_path).convert('RGB')
        W, H = img.size
        x1, y1, x2, y2 = bbox
        # normalise [0,1] -> px, clamp, coins ordonnes.
        px1, px2 = sorted((int(round(x1 * W)), int(round(x2 * W))))
        py1, py2 = sorted((int(round(y1 * H)), int(round(y2 * H))))
        px1, px2 = max(0, px1), min(W, px2)
        py1, py2 = max(0, py1), min(H, py2)
        if px2 <= px1 or py2 <= py1:
            print(json.dumps({"ok": False, "error": "empty mask box"}))
            return 1
        # Masque binaire 1 canal : 255 = a repeindre (contrat simple-lama).
        mask = Image.new('L', (W, H), 0)
        ImageDraw.Draw(mask).rectangle([px1, py1, px2 - 1, py2 - 1], fill=255)
        _log(f"[lama] inpaint {W}x{H} box=({px1},{py1},{px2},{py2})")
        result = SimpleLama()(img, mask)
        # Ecrit EN PLACE (l'appelant a deja preserve l'original). WEBP q92 comme le
        # reste du pipeline (normalize_to_webp / crop).
        result.convert('RGB').save(image_path, 'WEBP', quality=92)
        print(json.dumps({"ok": True}))
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
