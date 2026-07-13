"""Watermark inpainting via simple-lama-inpainting (LaMa), lance dans un interprete
DEDIE (le paquet est absent du venv Flask). Meme pattern subprocess que
person_mask.py / face_similarity.py. CPU (torch CPU, GPU masque cote infer) -> ne
touche PAS le GPU/ComfyUI, tourne HORS de la fenetre vision.

LaMa est NON-generatif : seuls les pixels du rectangle masque changent, le reste de
l'image reste identique. Sert la V1 de la correction automatique des watermarks : les
bbox HORS bande de bord mais d'aire <= 10% sont repeintes ici (les bbox de bord sont
croppees en PIL pur, sans ce module)."""
from __future__ import annotations
import json
import logging
import os
import subprocess
import sys

from .. import config as cfg

logger = logging.getLogger(__name__)

# lama_infer.py vit dans backend/infer/ (pas app/services/).
_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'lama_infer.py')


def _lama_python() -> str:
    # Cle dediee, sinon on reutilise le python ML existant (rembg/insightface), sinon
    # l'interpreteur courant. simple-lama vit dans le MEME extra ML (requirements-ml.txt).
    return cfg.get('watermark.python') or cfg.get('masks.python') or sys.executable


def is_available() -> bool:
    from ..capabilities import probe_watermark_inpaint
    return probe_watermark_inpaint()['ok']


def _stderr_tail(proc) -> str:
    return next((ln.strip() for ln in reversed((proc.stderr or '').splitlines())
                 if ln.strip()), '')


def inpaint_watermark(image_path, bbox, timeout: int = 300):
    """Repeint (LaMa) le rectangle `bbox` (normalise [x1,y1,x2,y2]) de `image_path`,
    EN PLACE. Retourne (ok: bool, error|None) — meme contrat que score_dataset_faces :
    `error` = None quand ca a tourne, sinon {'kind', 'detail'} : 'unavailable' (extra
    ML absent) | 'failed' (subprocess/JSON casse ou inpaint KO — detail = derniere
    ligne exploitable). NON-fatal mais JAMAIS muet (un inpaint casse doit dire POURQUOI,
    pas passer pour un succes)."""
    if not image_path or not os.path.isfile(image_path):
        return False, {'kind': 'failed', 'detail': 'image not found'}
    if not is_available():
        return False, {'kind': 'unavailable',
                       'detail': 'watermark inpainting is not installed (ML extras)'}
    payload = json.dumps({"image_path": image_path, "bbox": list(bbox)})
    try:
        proc = subprocess.run([_lama_python(), _SCRIPT], input=payload,
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning('watermark_lama: subprocess echec : %s', e)
        return False, {'kind': 'failed', 'detail': str(e)}
    line = next((ln for ln in reversed((proc.stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        tail = _stderr_tail(proc)
        logger.warning('watermark_lama: pas de JSON (rc=%s) stderr=%s',
                       proc.returncode, (proc.stderr or '')[-400:])
        return False, {'kind': 'failed',
                       'detail': tail or f'inpainter produced no output (rc={proc.returncode})'}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning('watermark_lama: JSON illisible : %s', e)
        return False, {'kind': 'failed', 'detail': f'unreadable inpainter output: {e}'}
    if not data.get('ok'):
        detail = data.get('error') or 'inpaint failed'
        logger.warning('watermark_lama: echec : %s', detail)
        return False, {'kind': 'failed', 'detail': detail}
    return True, None
