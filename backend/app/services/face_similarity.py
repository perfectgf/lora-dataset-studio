"""Scoring de ressemblance faciale via InsightFace antelopev2, en SUBPROCESS dans un
interprete DEDIE (insightface absent du venv Flask). Meme pattern que
app/services/joycaption.py. CPU -> ne touche pas le GPU/ComfyUI."""
from __future__ import annotations
import json
import logging
import os
import subprocess

from .. import config as cfg

logger = logging.getLogger(__name__)

# face_score_infer.py vit dans backend/infer/ (pas app/services/).
_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'face_score_infer.py')


def _scoring_python() -> str:
    import sys
    return cfg.get('face_scoring.python') or sys.executable


def is_available() -> bool:
    from ..capabilities import probe_face_scoring
    return probe_face_scoring()['ok']


def score_dataset_faces(ref_path, image_paths, timeout: int = 900) -> dict:
    """{path: {state, sim?, det, bbox_frac, yaw}}. Vide si indispo/echec (non-fatal)."""
    image_paths = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    if not ref_path or not os.path.isfile(ref_path) or not image_paths or not is_available():
        return {}
    payload = json.dumps({"ref": ref_path, "images": image_paths,
                          "models_root": cfg.get('face_scoring.models_root') or None})
    try:
        proc = subprocess.run([_scoring_python(), _SCRIPT], input=payload,
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning('face_similarity: subprocess echec : %s', e)
        return {}
    line = next((ln for ln in reversed((proc.stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        logger.warning('face_similarity: pas de JSON (rc=%s) stderr=%s',
                       proc.returncode, (proc.stderr or '')[-400:])
        return {}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning('face_similarity: JSON illisible : %s', e)
        return {}
    if not data.get('ref_ok'):
        logger.warning('face_similarity: ref inutilisable : %s', data.get('error'))
        return {}
    return data.get('results') or {}
