"""Person-mask generator — rembg (u2net), lance par le python_embeded de ComfyUI
(rembg absent du venv Flask). Meme pattern subprocess que face_score_infer.py.

stdin  : {"images": [paths...], "out_dir": path}
stdout : derniere ligne = JSON {"ok": bool, "written": N, "results": {path: state}}
Logs -> stderr. Le masque est la matte u2net BRUTE (L, blanc=personne, noir=fond,
bords doux pour les cheveux) sauvee en PNG sous le MEME nom de base que l'image —
la convention mask_path d'ai-toolkit (le fond est ensuite pondere par
mask_min_value cote training, pas ici).
"""
import json
import os
import sys


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        images = payload.get('images') or []
        out_dir = payload['out_dir']
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"payload: {e}"}))
        return 1
    os.makedirs(out_dir, exist_ok=True)
    try:
        from PIL import Image
        from rembg import new_session, remove
        session = new_session('u2net')  # telecharge ~/.u2net/u2net.onnx au 1er run
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"rembg init: {e}"}))
        return 1
    results, written = {}, 0
    for i, p in enumerate(images, 1):
        try:
            with Image.open(p) as im:
                mask = remove(im.convert('RGB'), session=session, only_mask=True)
            name = os.path.splitext(os.path.basename(p))[0] + '.png'
            mask.convert('L').save(os.path.join(out_dir, name), 'PNG')
            results[p] = 'ok'
            written += 1
        except Exception as e:
            results[p] = f'error: {e}'
        _log(f'[mask] {i}/{len(images)} {results[p]}')
    print(json.dumps({"ok": True, "written": written, "results": results}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
