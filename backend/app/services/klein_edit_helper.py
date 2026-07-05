"""Focused helper to enqueue a SINGLE Flux.2 Klein edit job.

Reused by the face-dataset fan-out. Builds a minimal job from the same
single-edit workflow /generate_edit uses in single mode (WORKFLOW_IMPROVE_SKIN_PATH,
nodes 52=LoadImage, 145=TextBox1 prompt, 77=KSampler seed, 9=SaveImage,
114=UNETLoader Klein). Deliberately a small focused helper rather than a refactor
of the large live generate_edit route.

Lifted from the parent project's app/services/klein_edit_helper.py for LoRA
Dataset Studio: SRC's module-level COMFYUI_INPUT_DIR/COMFYUI_OUTPUT_DIR constants
become live `cfg.comfyui_dir(...)` calls (config.json changes take effect without
a restart, and an unconfigured ComfyUI raises RuntimeError instead of writing
into a hardcoded path), and the hardcoded consistency-LoRA filename/strength are
now the `klein.consistency_lora` / `klein.consistency_strength` settings.
"""
from __future__ import annotations
import logging
import os
import random
import shutil
import uuid

from .. import config as cfg
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_IMPROVE_SKIN_PATH = cfg.BACKEND_DIR / 'workflows' / 'improve skin.json'

# Nodes this helper rewires — fail LOUDLY if the workflow file changes shape
# instead of silently enqueuing a job with the wrong source/prompt/model.
_REQUIRED_NODES = ('52', '145', '77', '9', '114')


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


def enqueue_klein_edit(user_id, source_filename, edit_prompt, klein_model=None,
                       extra_metadata=None, lora_strength=None, source_path=None):
    """Copy the source into ComfyUI input, configure the single Klein edit
    workflow, and enqueue it. Returns the app job_id. Raises ValueError on a
    missing source / unloadable workflow / missing required node, RuntimeError
    if ComfyUI isn't configured.
    `lora_strength` overrides `klein.consistency_strength` (clamped [0.0, 1.5]).
    `source_path` overrides the default ComfyUI output dir/<source_filename> lookup
    so callers with per-dataset storage can pass the full path directly."""
    if source_path is None:
        out_dir = _comfy_output_dir()
        if out_dir is None:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(out_dir, source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f"source image not found: {source_filename}")
    comfy_input_dir = _comfy_input_dir()
    workflow = load_workflow_local(str(WORKFLOW_IMPROVE_SKIN_PATH))
    if not workflow:
        raise ValueError("failed to load Klein edit workflow")
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f"workflow node {node} missing — improve skin.json a changé")

    uid = uuid.uuid4().hex[:8]
    comfy_input = f"edit_source_{uid}_{source_filename}"
    shutil.copy2(source_path, os.path.join(comfy_input_dir, comfy_input))

    workflow["52"]["inputs"]["image"] = comfy_input
    workflow["145"]["inputs"]["text1"] = edit_prompt
    workflow["77"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    workflow["9"]["inputs"]["filename_prefix"] = f"{user_id}_DatasetFace"
    # The improve-skin workflow's default UNET file is not guaranteed present on
    # disk → fall back to the first available Flux.2 Klein model so node 114 never
    # references a missing file (silent generation failure).
    if not klein_model:
        try:
            from ..utils.comfyui import get_flux2_klein_models
            models = get_flux2_klein_models()
            if models:
                klein_model = models[0]['filename']
        except Exception:
            pass
    if klein_model:
        workflow["114"]["inputs"]["unet_name"] = klein_model

    # Inject the consistency LoRA between the UNET (114) and the existing LoRA
    # node (139) → chain: 114 -> consistency -> 139 -> rest. Improves face fidelity.
    # Skipped (degraded but functional) if the LoRA file or node 139 is missing.
    consistency_lora = (cfg.get('klein.consistency_lora') or '').replace('/', os.sep)
    consistency_strength = cfg.get('klein.consistency_strength')
    lora_dir = cfg.comfyui_dir('loras')
    lora_path = (os.path.join(str(lora_dir), consistency_lora)
                if lora_dir and consistency_lora else None)
    if "139" not in workflow:
        logger.warning("workflow node 139 missing — consistency LoRA injection skipped")
    elif not lora_path or not os.path.exists(lora_path):
        logger.warning(f"consistency LoRA not found at {lora_path} — injection skipped")
    else:
        strength = consistency_strength
        if lora_strength is not None:
            strength = max(0.0, min(1.5, float(lora_strength)))
        workflow["ds_consistency_lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": consistency_lora,
                       "strength_model": strength, "model": ["114", 0]},
            "_meta": {"title": "Dataset consistency LoRA"},
        }
        workflow["139"]["inputs"]["model"] = ["ds_consistency_lora", 0]

    job_id = str(uuid.uuid4())
    meta = {"model_name": "klein_edit_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt=edit_prompt, job_id=job_id, metadata=meta)
    return job_id
