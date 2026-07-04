# app/utils/zimage_helper.py
"""Shared Z-Image (ZTurbo) workflow injection helper.

Factorizes the Z-Image parameter injection that previously lived inline in
``app/main/routes.py`` (the ``is_zturbo_workflow`` branch) so the /generate
route AND the LoRA test studio configure the ZTurbo workflow through ONE code
path that stays in sync with the workflow JSON shape:

    node 1  = UNETLoader          (z_model)
    node 4  = CLIPTextEncode      (positive prompt)
    node 6  = EmptySD3LatentImage (width/height/batch)
    node 7  = BasicScheduler      (z_steps)
    node 9  = CFGGuider           (z_cfg)
    node 10 = RandomNoise         (seed)
    node 13 = SaveImage           (filename_prefix)

LoRA chaining goes through ``inject_zimage_loras`` (which repoints the model
consumers 7 + 9 to the end of the LoraLoaderModelOnly chain).
"""
from __future__ import annotations

import logging
import os

from .comfyui import get_zimage_loras, get_zimage_models, inject_zimage_loras

logger = logging.getLogger(__name__)

# ZTurbo workflow node ids (see WORKFLOW_ZTURBO_PATH JSON).
ZT_NODE_UNET = "1"
ZT_NODE_PROMPT = "4"
ZT_NODE_NEGATIVE = "5"   # CLIPTextEncode négatif (inerte à cfg=1, agit dès z_cfg>1)
ZT_NODE_LATENT = "6"
ZT_NODE_STEPS = "7"
ZT_NODE_CFG = "9"
ZT_NODE_SEED = "10"
ZT_NODE_SAVE = "13"


def apply_zimage_settings(workflow, *, z_steps=None, z_cfg=None, z_model=None,
                          z_loras=None, prompt=None, negative=None, seed=None, width=None,
                          height=None, batch_size=None, filename_prefix=None,
                          allowed_models=None, allowed_loras=None):
    """Inject Z-Image (ZTurbo) parameters into ``workflow`` (mutated in place).

    Semantics mirror the historical /generate route block exactly:
      - ``z_steps`` clamped to [1, 50], written only if node 7 carries 'steps';
      - ``z_cfg`` clamped to [1.0, 15.0], written only if node 9 carries 'cfg';
      - ``z_model`` validated against ``allowed_models`` (defaults to
        ``get_zimage_models()``) before touching node 1 — path-injection guard;
      - ``z_loras`` = [{filename, strength}] whitelisted against
        ``allowed_loras`` (defaults to the real loras/z image/ folder) and
        chained via ``inject_zimage_loras``.

    Non-numeric ``z_steps``/``z_cfg``/``seed`` raise TypeError/ValueError —
    the caller decides whether to abort or ignore (the /generate route logs a
    warning and keeps the workflow defaults, like before the refactor).

    The remaining keyword args (prompt/seed/dims/filename_prefix) are the
    studio-side extras; ``None`` leaves the workflow value untouched.

    Returns ``{'z_model_used', 'z_loras_used', 'loras_injected'}`` where
    ``z_model_used`` is the basename of the applied model (or None) and
    ``z_loras_used`` is the [{name, strength}] list of LoRAs actually applied
    (or None) — both in the exact format the /generate history logging expects.
    """
    info = {'z_model_used': None, 'z_loras_used': None, 'loras_injected': 0}

    if z_steps not in (None, ''):
        if ZT_NODE_STEPS in workflow and "steps" in workflow[ZT_NODE_STEPS].get("inputs", {}):
            workflow[ZT_NODE_STEPS]["inputs"]["steps"] = max(1, min(50, int(z_steps)))

    if z_cfg not in (None, ''):
        if ZT_NODE_CFG in workflow and "cfg" in workflow[ZT_NODE_CFG].get("inputs", {}):
            workflow[ZT_NODE_CFG]["inputs"]["cfg"] = max(1.0, min(15.0, float(z_cfg)))

    z_model = (z_model or '').strip()
    if z_model and ZT_NODE_UNET in workflow and "unet_name" in workflow[ZT_NODE_UNET].get("inputs", {}):
        models = allowed_models if allowed_models is not None else get_zimage_models()
        if z_model in models:
            workflow[ZT_NODE_UNET]["inputs"]["unet_name"] = z_model
            info['z_model_used'] = os.path.basename(z_model)
            logger.info(f"Z-Image: model -> {z_model}")

    if z_loras:
        allowed = (allowed_loras if allowed_loras is not None
                   else {l['filename'] for l in get_zimage_loras()})
        n_inj = inject_zimage_loras(workflow, z_loras, allowed)
        applied = [l for l in z_loras if isinstance(l, dict) and l.get('filename') in allowed]
        if applied:
            info['z_loras_used'] = [{'name': os.path.basename(l['filename']),
                                     'strength': l.get('strength', 1.0)} for l in applied]
        info['loras_injected'] = n_inj
        if n_inj:
            logger.info(f"Z-Image: injected {n_inj} LoRA(s)")

    # --- Studio-side extras (None = leave the workflow value untouched) -----
    if prompt is not None and ZT_NODE_PROMPT in workflow:
        workflow[ZT_NODE_PROMPT]["inputs"]["text"] = prompt
    # Négatif : écrit UNIQUEMENT s'il est non vide (comme la route generate) — un
    # négatif vide laisse le défaut du workflow. Inerte à cfg=1, agit dès z_cfg>1.
    if negative and ZT_NODE_NEGATIVE in workflow and "text" in workflow[ZT_NODE_NEGATIVE].get("inputs", {}):
        workflow[ZT_NODE_NEGATIVE]["inputs"]["text"] = negative
    if seed is not None and ZT_NODE_SEED in workflow:
        workflow[ZT_NODE_SEED]["inputs"]["noise_seed"] = int(seed)
    if width is not None and ZT_NODE_LATENT in workflow:
        workflow[ZT_NODE_LATENT]["inputs"]["width"] = int(width)
    if height is not None and ZT_NODE_LATENT in workflow:
        workflow[ZT_NODE_LATENT]["inputs"]["height"] = int(height)
    if batch_size is not None and ZT_NODE_LATENT in workflow:
        workflow[ZT_NODE_LATENT]["inputs"]["batch_size"] = int(batch_size)
    if filename_prefix is not None and ZT_NODE_SAVE in workflow:
        workflow[ZT_NODE_SAVE]["inputs"]["filename_prefix"] = filename_prefix

    return info
