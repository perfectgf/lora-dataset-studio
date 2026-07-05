"""Génération de masques « personne » via rembg (u2net), en SUBPROCESS dans un
interprete DEDIE (rembg absent du venv Flask). Meme pattern que
app/services/face_similarity.py. CPU (onnxruntime) → ne touche pas le GPU/ComfyUI.

Sert le MASKED TRAINING (méthode jandordoe) : un masque par image d'entraînement,
le fond pondéré à mask_min_value (0.1) côté ai-toolkit → l'identité se lie au
sujet, pas au décor."""
from __future__ import annotations
import json
import logging
import os
import subprocess
import sys

from .. import config as cfg

logger = logging.getLogger(__name__)

# mask_infer.py vit dans backend/infer/ (pas app/services/).
_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'mask_infer.py')


def _mask_python() -> str:
    return cfg.get('masks.python') or sys.executable


def is_available() -> bool:
    from ..capabilities import probe_masks
    return probe_masks()['ok']


def generate_person_masks(image_paths, out_dir, timeout: int = 1200) -> dict:
    """Génère un masque PNG (même nom de base) par image dans `out_dir`.
    Retourne {'ok': bool, 'written': N, 'results': {path: state}}.
    Vide ({}) si indisponible/échec (JAMAIS bloquant — un entraînement sans
    masques reste un entraînement valide)."""
    image_paths = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    if not image_paths or not is_available():
        return {}
    payload = json.dumps({"images": image_paths, "out_dir": out_dir})
    try:
        proc = subprocess.run([_mask_python(), _SCRIPT], input=payload,
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning('person_mask: subprocess échec : %s', e)
        return {}
    line = next((ln for ln in reversed((proc.stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        logger.warning('person_mask: pas de JSON (rc=%s) stderr=%s',
                       proc.returncode, (proc.stderr or '')[-400:])
        return {}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning('person_mask: JSON illisible : %s', e)
        return {}
    if not data.get('ok'):
        logger.warning('person_mask: échec : %s', data.get('error'))
        return {}
    return data
