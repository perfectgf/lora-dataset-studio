"""JoyCaption Beta One — captioning de dataset LoRA via subprocess.

Le modèle (Llava 8B NF4) tourne dans le PYTHON DU VENV ai-toolkit (torch+transformers
+bitsandbytes), pas le Python de Flask — même pattern que la conversion zimage. On
caption tout le dataset en UN seul chargement de modèle (batch), sinon recharger le
8B par image serait inexploitable. Non-fatal : en cas d'indispo/échec, retourne {} et
le caller (`face_dataset_service.caption_images`) retombe sur Qwen3-VL (ou honore le
backend choisi dans les réglages)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from .. import config as cfg

logger = logging.getLogger(__name__)

# joycaption_infer.py vit dans backend/infer/ (pas app/services/).
_SCRIPT = cfg.BACKEND_DIR / 'infer' / 'joycaption_infer.py'


def is_available() -> bool:
    """JoyCaption est utilisable si le venv ai-toolkit ET le script existent."""
    venv = cfg.aitoolkit_path('venv_python')
    return bool(venv) and venv.exists() and _SCRIPT.exists()


def caption_images_joycaption(paths, prompt: str | None = None,
                              max_tokens: int = 300, timeout: int = 1800) -> dict:
    """Caption une LISTE d'images en un seul chargement de modèle.
    Retourne {chemin: caption}. Vide si indispo/échec (non-fatal)."""
    paths = [p for p in (paths or []) if p and os.path.isfile(p)]
    if not paths or not is_available():
        return {}
    payload = json.dumps({'images': paths, 'prompt': prompt, 'max_tokens': max_tokens})
    venv_python = str(cfg.aitoolkit_path('venv_python'))
    script = str(_SCRIPT)
    # HF_HOME = même cache que l'entraînement (modèle déjà téléchargé là).
    env = dict(os.environ, HF_HOME=str(cfg.aitoolkit_path('hf_home')), PYTHONIOENCODING='utf-8')
    started = time.monotonic()
    logger.info('joycaption: starting batch (%d image(s), timeout=%ss)', len(paths), timeout)
    try:
        proc = subprocess.run(
            [venv_python, script], input=payload, env=env,
            cwd=os.path.dirname(script), capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=timeout,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        logger.error('joycaption: timed out after %.1fs while processing %d image(s)',
                     time.monotonic() - started, len(paths))
        return {}
    except OSError as e:
        logger.error('joycaption: could not start subprocess after %.1fs: %s',
                     time.monotonic() - started, e)
        return {}
    out = (proc.stdout or '').strip()
    # La sortie JSON est la dernière ligne `{…}` (les logs vont sur stderr).
    line = next((ln for ln in reversed(out.splitlines()) if ln.strip().startswith('{')), '')
    if not line:
        logger.warning('joycaption: pas de JSON (rc=%s) stderr=%s',
                       proc.returncode, (proc.stderr or '')[-400:])
        return {}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning('joycaption: JSON illisible : %s', e)
        return {}
    if data.get('errors'):
        logger.info('joycaption: %d erreur(s) image : %s',
                    len(data['errors']), list(data['errors'].values())[:3])
    captions = {k: (v or '').strip() for k, v in (data.get('captions') or {}).items() if v}
    logger.info('joycaption: batch finished (%d/%d captioned, elapsed=%.1fs)',
                len(captions), len(paths), time.monotonic() - started)
    return captions
