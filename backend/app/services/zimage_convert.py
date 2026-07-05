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
from .lora_training import _aitoolkit_dir, _hf_home, _venv_python

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
    et les chemins absolus, et CONFINE le résultat sous models/ (realpath +
    commonpath) - un z_model forgé ne peut pas sortir du dossier des modèles."""
    if not z_model or os.path.isabs(z_model) or '..' in z_model.replace('\\', '/'):
        return None
    root = cfg.comfyui_dir('models')
    if not root:
        return None
    root_real = os.path.realpath(str(root))
    rel = z_model.replace('/', '\\')
    base = os.path.basename(rel)
    for sub in ('unet', 'diffusion_models'):
        for cand in (os.path.join(root_real, sub, rel), os.path.join(root_real, sub, 'z image', base)):
            real = os.path.realpath(cand)
            if os.path.isfile(real) and os.path.commonpath([root_real, real]) == root_real:
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
        raise ValueError("config.json Z-Image-Turbo absent du cache HF - lance d'abord "
                         "un entraînement sur la base officielle (télécharge le modèle)")
    out = converted_dir(z_model)
    os.makedirs(out, exist_ok=True)
    logger.info(f'conversion base {z_model} -> {out}')
    proc = subprocess.run([str(_venv_python()), _CONVERTER, merge, official_config_path, '--save', out],
                          capture_output=True, text=True, timeout=2400)
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
