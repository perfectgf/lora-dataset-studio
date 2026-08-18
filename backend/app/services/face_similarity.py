"""Scoring de ressemblance faciale via InsightFace antelopev2, en SUBPROCESS dans un
interprete DEDIE (insightface absent du venv Flask). Meme pattern que
app/services/joycaption.py. CPU par DEFAUT -> ne touche pas le GPU/ComfyUI ;
GPU seulement si face_scoring.device l'autorise ET que CUDA existe vraiment, et
alors DANS la fenetre GPU exclusive."""
from __future__ import annotations
import json
import logging
import os
import re
from contextlib import nullcontext

from ..gpu_window import GpuBusyError

from .. import config as cfg
from .infer_stream import run_infer_script, stderr_tail as _tail

logger = logging.getLogger(__name__)

# face_score_infer.py vit dans backend/infer/ (pas app/services/).
_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'face_score_infer.py')

# The scorer already announces every image it finishes on stderr — nobody read
# it, so the pass showed 0/N for its whole duration and then jumped to N/N. Same
# mechanism as the Bank's embedding pass (image_bank_service._PROGRESS_RE over
# face_embed_infer's "[embed] i/N"): one regex, one drain thread.
_PROGRESS_RE = re.compile(r'\[face\] (\d+)/(\d+)')


def _scoring_python() -> str:
    import sys
    return cfg.get('face_scoring.python') or sys.executable


def is_available() -> bool:
    from ..capabilities import probe_face_scoring
    return probe_face_scoring()['ok']


def _stderr_tail(lines) -> str:
    """Derniere ligne non vide de stderr — pour un crash Python c'est la ligne
    `SomeError: ...` du traceback, exactement ce qu'un humain veut lire."""
    return _tail(lines)


def _run_scorer(python, payload, timeout, on_progress):
    """Run face_score_infer, streaming its `[face] i/N` lines to ``on_progress``.

    Returns ``(stdout, stderr_lines, returncode, timed_out)``. The Popen/drain
    plumbing lives in infer_stream.run_infer_script, shared with the concept
    face-mask preview; only the line grammar is ours."""
    def _on_line(line):
        m = _PROGRESS_RE.search(line)
        if m and on_progress:
            on_progress(int(m.group(1)), int(m.group(2)))

    return run_infer_script(python, _SCRIPT, payload, timeout, _on_line)


# Budget temps par image, en secondes. antelopev2 sur CPU tourne autour de
# 0.3-1 s/image selon la taille ; 3 s laisse de la marge sur une machine lente
# sans jamais bloquer une session entiere. Le forfait couvre le chargement du
# modele (le plus gros cout fixe du subprocess).
_TIMEOUT_PER_IMAGE_S = 3
_TIMEOUT_FLOOR_S = 900


def default_timeout(n_images: int) -> int:
    """Budget d'un run de scoring, en secondes. Le timeout etait un forfait de
    900 s dimensionne pour le seul set GARDE ; depuis que la passe couvre aussi
    la pile de triage (les variations generees non encore ✓/✕), un gros dataset
    peut depasser ce forfait — et un timeout ne rend AUCUN resultat partiel, donc
    la passe entiere serait perdue. Le budget suit donc le nombre d'images."""
    return max(_TIMEOUT_FLOOR_S,
               120 + _TIMEOUT_PER_IMAGE_S * max(0, int(n_images or 0)))


def score_dataset_faces(ref_path, image_paths, timeout: int | None = None,
                        on_progress=None):
    """Retourne ({path: {state, sim?, det, bbox_frac, yaw, zoomed}}, error|None).

    `on_progress(done, total)` — optionnel — est appelé à chaque image finie par
    le scorer, depuis un thread de lecture (donc PAS dans un contexte Flask :
    n'y touchez qu'à de l'état en mémoire, comme dataset_activity). Sans lui la
    passe reste exactement ce qu'elle était.

    `error` est None quand le scorer a tourne, sinon {'kind', 'detail'} :
    'unavailable' (extras ML absents), 'failed' (subprocess/JSON casse — detail
    = derniere ligne du traceback), 'ref_unusable' (la reference n'a pas de
    visage exploitable), 'gpu_busy' (voie GPU demandee, carte prise). Les echecs restent NON-fatals ({} + error) mais
    doivent etre VISIBLES : les avaler en {} muet transformait un scorer casse
    en « Face scoring done — 0/14 » avec toast vert (user-reported)."""
    image_paths = [p for p in (image_paths or []) if p and os.path.isfile(p)]
    if not ref_path or not os.path.isfile(ref_path) or not image_paths:
        return {}, None
    if not is_available():
        return {}, {'kind': 'unavailable',
                    'detail': 'face scoring is not installed (Quality tools step in Setup)'}
    if timeout is None:
        timeout = default_timeout(len(image_paths))
    # Same verdict as the Bank's face pass, from the same resolver: two surfaces
    # running the same models must not disagree about the GPU (CLAUDE.md, the
    # Bank/Dataset parity rule). 'cpu' stays the answer on every stock install —
    # the face extra ships CPU onnxruntime — so this changes nothing until the
    # user puts onnxruntime-gpu in that interpreter.
    from ..capabilities import resolve_face_device
    device, use_gpu = resolve_face_device()
    payload = json.dumps({"ref": ref_path, "images": image_paths,
                          "models_root": cfg.get('face_scoring.models_root') or None,
                          "device": device})
    # GPU is EXCLUSIVE or it is nothing. The window unloads ComfyUI and holds off
    # a training start for the whole pass, which is precisely why this scorer was
    # taken off CUDA in "fix(gpu): serialize local inference and ComfyUI
    # recovery" back when no such window existed. A CPU pass stays out of it and
    # keeps its original promise: it runs alongside ComfyUI, bothering nobody.
    # Built INSIDE the try: a @contextmanager only raises on __enter__, but a
    # refusal raised at construction time would otherwise escape this function
    # as an unhandled error instead of the honest 'gpu_busy' below.
    try:
        if use_gpu:
            from ..gpu_window import gpu_exclusive_vision_window
            window = gpu_exclusive_vision_window(flag_ttl=1800)
        else:
            window = nullcontext()
        with window:
            stdout, stderr_lines, returncode, timed_out = _run_scorer(
                _scoring_python(), payload, timeout, on_progress)
    except GpuBusyError as e:
        # The card is taken (a training, a Studio grid). Say so instead of
        # failing obscurely — and do NOT silently fall back to CPU: the user
        # asked for the fast lane, a slow one pretending to be it is worse.
        logger.info('face_similarity: GPU occupe : %s', e)
        return {}, {'kind': 'gpu_busy',
                    'detail': f'the GPU is busy ({e}) - retry when it frees up, '
                              f'or set face_scoring.device to cpu'}
    except OSError as e:
        logger.warning('face_similarity: subprocess echec : %s', e)
        return {}, {'kind': 'failed', 'detail': str(e)}
    if timed_out:
        logger.warning('face_similarity: timeout apres %ss', timeout)
        return {}, {'kind': 'failed',
                    'detail': f'face scoring timed out after {timeout}s '
                              f'({len(image_paths)} image(s))'}
    line = next((ln for ln in reversed((stdout or '').splitlines())
                 if ln.strip().startswith('{')), '')
    if not line:
        tail = _stderr_tail(stderr_lines)
        logger.warning('face_similarity: pas de JSON (rc=%s) stderr=%s',
                       returncode, ' | '.join(stderr_lines))
        return {}, {'kind': 'failed',
                    'detail': tail or f'scorer produced no output (rc={returncode})'}
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
