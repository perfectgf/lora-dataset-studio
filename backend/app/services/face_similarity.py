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


def _stderr_tail(proc) -> str:
    """Derniere ligne non vide de stderr — pour un crash Python c'est la ligne
    `SomeError: ...` du traceback, exactement ce qu'un humain veut lire."""
    return next((ln.strip() for ln in reversed((proc.stderr or '').splitlines())
                 if ln.strip()), '')


def score_dataset_faces(ref_path, image_paths, timeout: int = 900):
    """Retourne ({path: {state, sim?, det, bbox_frac, yaw}}, error|None).

    `error` est None quand le scorer a tourne, sinon {'kind', 'detail'} :
    'unavailable' (extras ML absents), 'failed' (subprocess/JSON casse — detail
    = derniere ligne du traceback), 'ref_unusable' (la reference n'a pas de
    visage exploitable). Les echecs restent NON-fatals ({} + error) mais
    doivent etre VISIBLES : les avaler en {} muet transformait un scorer casse
    en « Face scoring done — 0/14 » avec toast vert (user-reported)."""
    image_paths = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    if not ref_path or not os.path.isfile(ref_path) or not image_paths:
        return {}, None
    if not is_available():
        return {}, {'kind': 'unavailable',
                    'detail': 'face scoring is not installed (Quality tools step in Setup)'}
    payload = json.dumps({"ref": ref_path, "images": image_paths,
                          "models_root": cfg.get('face_scoring.models_root') or None})
    try:
        proc = subprocess.run([_scoring_python(), _SCRIPT], input=payload,
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning('face_similarity: subprocess echec : %s', e)
        return {}, {'kind': 'failed', 'detail': str(e)}
    line = next((ln for ln in reversed((proc.stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        tail = _stderr_tail(proc)
        logger.warning('face_similarity: pas de JSON (rc=%s) stderr=%s',
                       proc.returncode, (proc.stderr or '')[-400:])
        return {}, {'kind': 'failed',
                    'detail': tail or f'scorer produced no output (rc={proc.returncode})'}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning('face_similarity: JSON illisible : %s', e)
        return {}, {'kind': 'failed', 'detail': f'unreadable scorer output: {e}'}
    if not data.get('ref_ok'):
        logger.warning('face_similarity: ref inutilisable : %s', data.get('error'))
        return {}, {'kind': 'ref_unusable',
                    'detail': data.get('error') or 'no usable face in the reference photo'}
    return data.get('results') or {}, None
