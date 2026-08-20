"""Conversion d'un checkpoint Z-Image ComfyUI (.safetensors single-file) vers le
format diffusers attendu par ai-toolkit, pour entraîner un LoRA sur un merge
custom.

S'appuie sur convert_comfy_zimage_to_diffusers.py (mapping OFFICIEL ComfyUI,
gate validé : 0 clé manquante), lancé avec le python d'ai-toolkit (diffusers
requis). Conversion lourde (~12 Go, quelques minutes) → faite une fois et mise
en cache sous <aitoolkit dir>/converted/<name>/, en thread d'arrière-plan.

Lifted from the parent project's app/services/zimage_convert.py for LoRA
Dataset Studio: the module-level CONVERTED_ROOT (hardcoded `F:\\AI\\aitoolkit\\
converted`) and the COMFYUI_OUTPUT_DIR-derived models root become live
`cfg`-backed accessors (no machine-specific default paths, per the plan's
Global Constraints) -- the converted-cache root now lives under the
configured ai-toolkit dir instead of a separate hardcoded drive.
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import threading

from .. import config as cfg
from ..job_queue import queue_manager
from ..utils.comfy_names import local_model_path
from . import infer_env
from .lora_training import (_aitoolkit_dir, _hf_home, _venv_python,
                            assert_free_disk, MIN_FREE_GB_CONVERT)

logger = logging.getLogger(__name__)

_CONVERTER = str(cfg.BACKEND_DIR / 'infer' / 'convert_comfy_zimage_to_diffusers.py')
_CONVERT_KEY = 'zimage_base_convert'  # system_state : statut de conversion en cours
_convert_lock = threading.Lock()      # sérialise l'acquisition du verrou de conversion


def _converted_root():
    return _aitoolkit_dir() / 'converted'


def _official_config() -> str | None:
    """config.json du transformer Z-Image-Turbo officiel (cache HF d'ai-toolkit).
    Présent dès qu'un entraînement officiel a tourné une fois."""
    g = glob.glob(os.path.join(str(_hf_home()), 'hub', 'models--Tongyi-MAI--Z-Image-Turbo',
                               'snapshots', '*', 'transformer', 'config.json'))
    return g[0] if g else None


def _resolve_merge(z_model: str) -> str | None:
    """Chemin absolu du .safetensors ComfyUI depuis une valeur z_model
    (ex. 'z image\\bigLove_zt3.safetensors'). Anti path-traversal : refuse '..'
    et les chemins absolus, et CONFINE la CIBLE sous models/ - un z_model forgé
    ne peut pas sortir du dossier des modèles.

    Le confinement est vérifié LEXICALEMENT sur `cand` (construit sous root_real
    à partir d'une valeur sans '..' ni chemin absolu) : il reste sur le même
    disque, donc commonpath ne crashe jamais. On NE fait PAS commonpath sur le
    realpath : celui-ci peut légitimement traverser un disque via une jonction
    Windows (models/ symlinké vers un 2e DD, cas courant pour les gros modèles),
    ce qui levait `ValueError: Paths don't have the same drive` et cassait la
    conversion. realpath ne sert donc qu'à tester l'existence et à retourner le
    chemin réel dont le sous-process a besoin.

    Les racines fouillées sont celles de ComfyUI lui-même — ``<base>/models/{unet,
    diffusion_models}`` PLUS toute racine ``diffusion_models`` déclarée dans
    ``extra_model_paths.yaml`` — dans l'ordre de priorité de ComfyUI (``is_default``
    d'abord). Avant ce correctif, un merge Z-Image rangé sous une racine yaml était
    tout simplement inconvertible : présent sur le disque, chargé par ComfyUI tous
    les jours, et « base model not found on disk » ici. `unet` et `diffusion_models`
    ne sont plus deux dossiers en dur mais le MÊME type canonique côté ComfyUI
    (folder_paths.map_legacy), ce que ``search_roots`` applique déjà.

    Sans yaml, les racines sont exactement les deux dossiers historiques, dans le
    même ordre : la résolution est inchangée au bit près."""
    if not z_model or os.path.isabs(z_model) or '..' in z_model.replace('\\', '/'):
        return None
    if not cfg.comfyui_dir('models'):
        return None
    # Chemin du SYSTÈME DE FICHIERS LOCAL, pas un widget ComfyUI : séparateur
    # os.sep, jamais un backslash en dur. Sous Linux, `os.path.join(root, 'unet',
    # 'z image\\x.safetensors')` n'échoue pas — il fabrique le nom d'un fichier
    # unique contenant un backslash, donc introuvable, donc « conversion
    # impossible » sans explication (famille du bug GitHub #21, 1Tomber).
    rel = local_model_path(z_model)
    base = os.path.basename(rel)
    from . import comfy_model_paths
    roots = comfy_model_paths.search_roots('diffusion_models')
    for root in roots:
        root_real = os.path.realpath(str(root))
        root_key = os.path.normcase(os.path.normpath(root_real))
        for want in (rel, os.path.join('z image', base)):
            cand = os.path.join(root_real, want)
            cand_key = os.path.normcase(os.path.normpath(cand))
            if os.path.commonpath([root_key, cand_key]) != root_key:
                continue
            real = os.path.realpath(cand)
            if os.path.isfile(real):
                return real
    # Second passage, insensible à la casse : la valeur stockée porte la casse du
    # template ou du picker ('Z image\\'), le disque porte la sienne ('z image').
    # Windows ne fait pas la différence, Linux (et tout entraînement cloud) si.
    for root in roots:
        root_real = os.path.realpath(str(root))
        for want in (rel, os.path.join('z image', base)):
            hit = comfy_model_paths.ci_resolve(root_real, want)
            if hit:
                real = os.path.realpath(hit)
                if os.path.isfile(real):
                    return real
    return None


def _safe_name(z_model: str) -> str:
    """Nom du dossier de conversion dérivé du chemin COMPLET (sous-dossier inclus),
    pas du seul basename - sinon deux merges homonymes dans des sous-dossiers
    différents écraseraient la même conversion."""
    rel = z_model.replace('\\', '/').rsplit('.', 1)[0]
    safe = ''.join(c if (c.isalnum() or c in '_-') else '_' for c in rel).strip('_')
    return safe or 'base'


def converted_dir(z_model: str) -> str:
    return str(_converted_root() / _safe_name(z_model))


def is_converted(z_model: str) -> bool:
    d = os.path.join(converted_dir(z_model), 'transformer')
    return (os.path.isfile(os.path.join(d, 'diffusion_pytorch_model.safetensors'))
            and os.path.isfile(os.path.join(d, 'config.json')))


def convert(z_model: str) -> str:
    """Convertit (BLOQUANT, plusieurs minutes). Retourne le dossier diffusers
    racine (à passer en name_or_path). Lève ValueError si échec."""
    if is_converted(z_model):
        return converted_dir(z_model)
    merge = _resolve_merge(z_model)
    if not merge:
        raise ValueError(f'base model not found on disk: {z_model}')
    official_config_path = _official_config()
    if not official_config_path:
        raise ValueError("config.json for Z-Image-Turbo is missing from the HF cache - first run "
                         "a training on the official base (this downloads the model)")
    out = converted_dir(z_model)
    os.makedirs(out, exist_ok=True)
    logger.info(f'conversion base {z_model} -> {out}')
    proc = subprocess.run(infer_env.worker_argv(_venv_python(), _CONVERTER, merge,
                                                official_config_path, '--save', out),
                          capture_output=True, text=True, timeout=2400,
                          env=infer_env.worker_env(_venv_python()))
    if not is_converted(z_model):
        tail = (proc.stdout or '')[-600:] + ' | ' + (proc.stderr or '')[-600:]
        raise ValueError(f'conversion failed: {tail}')
    return out


# --- Conversion en arrière-plan + statut (poll UI) ----------------------------
def convert_status() -> dict:
    return queue_manager._get_system_state(_CONVERT_KEY, {}) or {}


def start_convert_async(app, z_model: str) -> None:
    """Lance la conversion dans un thread daemon ; statut suivi dans system_state
    (running/done/error). Refuse si une conversion tourne déjà."""
    if not _resolve_merge(z_model):
        raise ValueError(f'base model not found: {z_model}')
    # ~12 Go écrits : refuser tout de suite plutôt qu'un crash à 90 % qui laisse
    # un dossier diffusers incomplet (is_converted=False mais 10 Go consommés).
    assert_free_disk(_converted_root(), MIN_FREE_GB_CONVERT, 'the base conversion (~12 GB)')
    # Acquisition ATOMIQUE du verrou (check-then-set sous lock) : empêche deux
    # conversions 12 Go concurrentes (double-clic / 2 datasets en même temps).
    with _convert_lock:
        if convert_status().get('status') == 'running':
            raise ValueError('a conversion is already in progress')
        queue_manager._set_system_state(_CONVERT_KEY, {'z_model': z_model, 'status': 'running'},
                                        ttl_seconds=3600)

    def _run():
        with app.app_context():
            try:
                convert(z_model)
                queue_manager._set_system_state(
                    _CONVERT_KEY, {'z_model': z_model, 'status': 'done'}, ttl_seconds=3600)
                logger.info(f'conversion base terminée : {z_model}')
            except Exception as e:
                queue_manager._set_system_state(
                    _CONVERT_KEY, {'z_model': z_model, 'status': 'error', 'error': str(e)},
                    ttl_seconds=3600)
                logger.error(f'conversion base échouée ({z_model}) : {e}')

    threading.Thread(target=_run, daemon=True).start()
