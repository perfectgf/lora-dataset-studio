"""JoyCaption Beta One — captioning de dataset LoRA via subprocess.

Le modèle (Llava 8B NF4) tourne dans le PYTHON DU VENV ai-toolkit (torch+transformers
+bitsandbytes), pas le Python de Flask — même pattern que la conversion zimage. On
caption tout le dataset en UN seul chargement de modèle (batch), sinon recharger le
8B par image serait inexploitable. Non-fatal : en cas d'indispo/échec, retourne {} et
le caller (`face_dataset_service.caption_images`) retombe sur Qwen3-VL (ou honore le
backend choisi dans les réglages)."""
from __future__ import annotations

import collections
import json
import logging
import os
import subprocess
import threading
import time

from .. import config as cfg

logger = logging.getLogger(__name__)

# joycaption_infer.py vit dans backend/infer/ (pas app/services/).
_SCRIPT = cfg.BACKEND_DIR / 'infer' / 'joycaption_infer.py'


def availability() -> dict:
    """Single source of truth for JoyCaption readiness: the capability probe.
    Returns {ok, detail} — `detail` names what's missing (the exact pip command
    for the ai-toolkit venv when the deps aren't importable). Delegating here means
    is_available() can't drift from what the Settings UI advertises (issue #6: the
    old filesystem-only check said "ready" while transformers was absent)."""
    from .. import capabilities
    return capabilities.probe_joycaption()


def is_available() -> bool:
    """True only when the ai-toolkit venv + script exist AND the venv can import the
    JoyCaption deps (transformers/bitsandbytes/accelerate) — otherwise the batch
    subprocess would ModuleNotFoundError. See availability() for the reason string."""
    return availability()['ok']


def _reflect_stage(line: str, activity_token) -> None:
    """Mirror one of the infer script's own ``[joycaption] …`` markers into the dataset
    activity indicator so the UI shows a live stage ("first run: downloading …",
    "model loaded", per-image progress) instead of a frozen "Loading…". Raw Hugging Face
    tqdm download bars (no ``[joycaption]`` prefix) go to the log only, keeping the UI
    detail readable. Non-fatal: never let progress reflection break captioning."""
    if not activity_token or not line.startswith('[joycaption]'):
        return
    try:
        from . import dataset_activity
        dataset_activity.progress(activity_token, detail=line[len('[joycaption]'):].strip()[:200])
    except Exception:  # noqa: BLE001
        pass


def caption_images_joycaption(paths, prompt: str | None = None,
                              max_tokens: int = 300, timeout: int = 1800,
                              activity_token=None, should_cancel=None) -> dict:
    """Caption une LISTE d'images en un seul chargement de modèle.
    Retourne {chemin: caption}. Vide si indispo/échec (non-fatal).

    Le stderr du subprocess est STREAMÉ ligne-à-ligne vers le log de l'app EN DIRECT
    (thread lecteur) : au PREMIER run le modèle 8B NF4 (~7 Go) se télécharge depuis
    Hugging Face, et sans ce flux l'app semblait gelée (issue #6 — l'utilisateur croyait
    que rien ne se passait). Chargement du modèle, progression du download et erreurs
    apparaissent désormais au fil de l'eau. ``activity_token`` (optionnel) reflète en plus
    les jalons dans l'indicateur d'activité du dataset.

    ``should_cancel`` (optionnel) : polled at each image BOUNDARY for a graceful Stop.
    stdout is streamed per image, so each caption already delivered is KEPT; when the flag
    trips, the subprocess is killed (no half-decoded image is interrupted) and the captions
    gathered so far are returned — the SAME "keep what's written, stop the rest" contract as
    the Ollama loop. Without it the whole batch was uninterruptible: Stop flipped the UI to
    "Stopping…" while JoyCaption kept captioning every image to the end."""
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
        proc = subprocess.Popen(
            [venv_python, script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=os.path.dirname(script), text=True,
            encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except OSError as e:
        logger.error('joycaption: could not start subprocess after %.1fs: %s',
                     time.monotonic() - started, e)
        return {}

    # Drain both pipes in threads: stderr is logged live (first-run download visibility),
    # stdout is streamed LINE BY LINE and parsed as each per-image caption lands. Reading
    # both concurrently avoids the pipe-buffer deadlock a naive proc.wait() would hit on a
    # chatty subprocess, and lets us enforce the timeout via proc.wait() while the readers
    # keep draining. Streaming stdout (rather than one end-of-run .read()) is what makes a
    # graceful Stop keep the captions already produced.
    captions: dict[str, str] = {}
    errors: dict[str, str] = {}
    cancelled = {'flag': False}
    stderr_tail: collections.deque = collections.deque(maxlen=25)

    def _consume_json_line(line: str) -> None:
        """Parse one stdout JSON line: a per-image {i,path,caption|error}, or the final
        {captions,errors} aggregate (merged defensively for a stale worker)."""
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            return
        if not isinstance(obj, dict):
            return
        if 'path' in obj:
            p = obj['path']
            if obj.get('caption'):
                captions[p] = str(obj['caption']).strip()
            elif obj.get('error'):
                errors[p] = str(obj['error'])
        elif 'captions' in obj:
            for p, cap in (obj.get('captions') or {}).items():
                if cap and p not in captions:
                    captions[p] = str(cap).strip()
            errors.update(obj.get('errors') or {})

    def _drain_stdout():
        try:
            for raw in proc.stdout:
                line = raw.strip()
                if line.startswith('{'):
                    _consume_json_line(line)
                    # Graceful Stop seam: we just captured an image's caption, so killing
                    # here loses nothing already produced. The next generate() (possibly
                    # long) never starts.
                    if should_cancel and not cancelled['flag'] and should_cancel():
                        cancelled['flag'] = True
                        logger.info('joycaption: stop requested — killing batch after '
                                    '%d caption(s)', len(captions))
                        try:
                            proc.kill()
                        except Exception:  # noqa: BLE001
                            pass
                        break
        except Exception:  # noqa: BLE001
            pass

    def _drain_stderr():
        try:
            for raw in proc.stderr:
                line = raw.rstrip('\n')
                if not line:
                    continue
                stderr_tail.append(line)
                logger.info('joycaption[sub]: %s', line)
                _reflect_stage(line, activity_token)
        except Exception:  # noqa: BLE001
            pass

    t_out = threading.Thread(target=_drain_stdout, daemon=True)
    t_err = threading.Thread(target=_drain_stderr, daemon=True)
    t_out.start()
    t_err.start()
    try:
        proc.stdin.write(payload)
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        # The dominant cause of a first-run timeout is the ~7 GB model download, not a
        # hang — say so, and note the download is cached so a re-run resumes instead of
        # restarting from zero. Anything already streamed is still returned below.
        logger.error('joycaption: timed out after %.1fs while processing %d image(s) — '
                     'if this was the FIRST run the ~7 GB model was still downloading; '
                     'the partial download is cached, so just run captioning again to '
                     'resume. Last subprocess output: %s',
                     time.monotonic() - started, len(paths),
                     ' | '.join(list(stderr_tail)[-5:]) or '(none)')
    t_out.join(timeout=5)
    t_err.join(timeout=5)

    # `captions`/`errors` were filled by the stdout drain as each per-image line arrived, so
    # a graceful Stop (or a timeout) still returns everything produced so far.
    result = {k: v for k, v in captions.items() if v}
    if errors:
        logger.info('joycaption: %d erreur(s) image : %s',
                    len(errors), list(errors.values())[:3])
    if not result and not cancelled['flag'] and not errors:
        logger.warning('joycaption: pas de captions (rc=%s) stderr=%s',
                       proc.returncode, ' | '.join(list(stderr_tail)[-6:]))
    logger.info('joycaption: batch %s (%d/%d captioned, elapsed=%.1fs)',
                'stopped' if cancelled['flag'] else 'finished',
                len(result), len(paths), time.monotonic() - started)
    return result
