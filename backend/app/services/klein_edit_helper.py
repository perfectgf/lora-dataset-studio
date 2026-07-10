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
_REQUIRED_NODES = ('52', '145', '77', '9', '114', '10', '90')

# The Klein pipeline's model dependencies, keyed by the setup_installer download
# action that provides each. REQUIRED = the graph is invalid without it (block +
# auto-download); RECOMMENDED = quality only (the consistency LoRA — degrade).
KLEIN_REQUIRED = ('klein_model', 'klein_text_encoder', 'klein_vae')
KLEIN_RECOMMENDED = ('klein_lora',)

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')


class KleinModelsMissing(Exception):
    """A graph-critical Klein asset (UNET / text-encoder / VAE) is not on disk, so
    a valid job can't be built. `.missing` lists ALL absent assets (incl. the
    optional consistency LoRA) as setup_installer action names, so the caller can
    auto-download them instead of firing a doomed ComfyUI job."""
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__('Klein models missing: ' + ', '.join(self.missing))


def _models_root():
    d = cfg.comfyui_dir('models')
    return str(d) if d else None


# Canonical filenames = the exact files the Setup installer downloads (single
# source of truth: setup_installer._KLEIN_DOWNLOADS dest names). Matching MUST be
# canonical-first with NARROW token fallbacks: on a ComfyUI shared with other
# apps, models/text_encoders/ holds many families' encoders and a loose
# "contains 'qwen'" match wired Z-Image's qwen3vl_4b TE (sorts before
# qwen_3_8b_fp8mixed: '3' < '_') into the Klein graph -> KSampler died on
# "mat1 and mat2 shapes cannot be multiplied". Wrong model >> missing model:
# missing triggers the auto-download, wrong fails at runtime with a cryptic error.
def _canonical_name(action):
    from .. import setup_installer
    return setup_installer._KLEIN_DOWNLOADS[action]['dest'][-1]


def _find_model_file(subparts, canonical, tokens):
    """Model file under <models>/<subparts...>: the canonical filename if present,
    else the first (sorted) name containing any of the NARROW tokens. None when
    nothing matches — never a blind first-file-in-folder guess."""
    root = _models_root()
    if not root:
        return None
    folder = os.path.join(root, *subparts)
    try:
        names = sorted(n for n in os.listdir(folder)
                       if n.lower().endswith(_MODEL_SUFFIXES))
    except OSError:
        return None
    if canonical in names:
        return canonical
    for n in names:
        if any(tok in n.lower() for tok in tokens):
            return n
    return None


def resolve_klein_unet(selected=None):
    """ComfyUI-relative `unet_name` for node 114, or None if no Klein model is on
    disk. Scans models/unet/klein/ and models/diffusion_models/klein/ (the folders
    the capability gate and the Setup downloads use) and returns the value WITH its
    subfolder prefix (e.g. 'klein\\flux-2-klein-9b-fp8.safetensors'): a UNETLoader
    lists files relative to models/unet, so the bare filename the picker sends is
    not loadable on its own — the missing 'klein\\' prefix is the whole bug.
    Inside klein/ every file IS a Klein UNET, so the picker's choice wins, then
    the canonical download, then the first file."""
    root = _models_root()
    if not root:
        return None
    bare_pick = os.path.basename(selected) if selected else None
    canonical = _canonical_name('klein_model')
    for base in ('unet', 'diffusion_models'):
        folder = os.path.join(root, base, 'klein')
        try:
            names = sorted(n for n in os.listdir(folder)
                           if n.lower().endswith(_MODEL_SUFFIXES))
        except OSError:
            continue
        if not names:
            continue
        if bare_pick and bare_pick in names:
            pick = bare_pick
        elif canonical in names:
            pick = canonical
        else:
            pick = names[0]
        return os.path.join('klein', pick)
    return None


def resolve_klein_vae():
    """`vae_name` for node 10 — canonical flux2-vae.safetensors, else a narrow
    flux2-vae token match (covers the 'flux2_vae.safetensors.safetensors'
    double-extension variant some installs carry). Never e.g. qwen_image_vae."""
    return _find_model_file(('vae',), _canonical_name('klein_vae'),
                            ('flux2-vae', 'flux2_vae', 'flux-2-vae', 'flux_2_vae'))


def resolve_klein_text_encoder():
    """`clip_name` for node 90 — canonical qwen_3_8b_fp8mixed.safetensors, else a
    narrow qwen_3_8b token match. NEVER a bare 'qwen' match: qwen3vl_* (Z-Image)
    and qwen_2.5_vl_* (Qwen-Image) encoders live in the same folder and produce
    incompatible embeddings."""
    return _find_model_file(('text_encoders',), _canonical_name('klein_text_encoder'),
                            ('qwen_3_8b', 'qwen3_8b', 'qwen-3-8b'))


def _consistency_lora():
    """(relative_name, absolute_path) of the configured consistency LoRA, or
    (name, None) when the loras dir is unset."""
    name = (cfg.get('klein.consistency_lora') or '').replace('/', os.sep)
    lora_dir = cfg.comfyui_dir('loras')
    if not (lora_dir and name):
        return name or None, None
    return name, os.path.join(str(lora_dir), name)


def klein_missing_assets():
    """Which Klein assets are NOT on disk, as setup_installer action names (a
    subset of KLEIN_REQUIRED + KLEIN_RECOMMENDED). Drives both the generate-time
    block and the auto-download."""
    missing = []
    if not resolve_klein_unet():
        missing.append('klein_model')
    if not resolve_klein_text_encoder():
        missing.append('klein_text_encoder')
    if not resolve_klein_vae():
        missing.append('klein_vae')
    _, lora_path = _consistency_lora()
    if not (lora_path and os.path.exists(lora_path)):
        missing.append('klein_lora')
    return missing


def _bypass_node(workflow, node_id, passthrough_input):
    """Delete node_id and reconnect every consumer of its output slot 0 to the
    node's own `passthrough_input` upstream. Used to drop a LoRA loader whose file
    is absent (its consumers wire straight to its `model` input) so ComfyUI never
    fails validation on a missing LoRA."""
    node = workflow.get(node_id)
    if not node:
        return
    upstream = node.get('inputs', {}).get(passthrough_input)
    if upstream is None:
        return
    for other in workflow.values():
        for k, v in list(other.get('inputs', {}).items()):
            if isinstance(v, list) and len(v) == 2 and v[0] == node_id:
                other['inputs'][k] = upstream
    workflow.pop(node_id, None)


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


def enqueue_klein_edit(user_id, source_filename, edit_prompt, klein_model=None,
                       extra_metadata=None, lora_strength=None, source_path=None,
                       extra_ref_paths=None):
    """Copy the source into ComfyUI input, configure the single Klein edit
    workflow, and enqueue it. Returns the app job_id. Raises ValueError on a
    missing source / unloadable workflow / missing required node, RuntimeError
    if ComfyUI isn't configured.
    `lora_strength` overrides `klein.consistency_strength` (clamped [0.0, 1.5]);
    0 disables the consistency LoRA entirely (it anchors composition, and even
    mid strengths can suppress big restagings).
    `source_path` overrides the default ComfyUI output dir/<source_filename> lookup
    so callers with per-dataset storage can pass the full path directly.
    `extra_ref_paths`: additional identity reference images (the dataset's extra
    refs) chained as native ReferenceLatent nodes — Klein consumes several
    references natively, and extra angles of the same face lock identity better
    than the single primary ref."""
    if source_path is None:
        out_dir = _comfy_output_dir()
        if out_dir is None:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(out_dir, source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f"source image not found: {source_filename}")
    workflow = load_workflow_local(str(WORKFLOW_IMPROVE_SKIN_PATH))
    if not workflow:
        raise ValueError("failed to load Klein edit workflow")
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f"workflow node {node} missing — improve skin.json has changed")

    # Resolve every loader node against what is ACTUALLY installed — the lifted
    # workflow hardcodes the developer's own ComfyUI filenames (see module
    # docstring), none of which match a fresh install. Block BEFORE copying the
    # source / enqueuing when a graph-critical asset is absent, so the caller can
    # auto-download it instead of firing a job every tile of which would fail.
    unet_ref = resolve_klein_unet(klein_model)
    vae_ref = resolve_klein_vae()
    te_ref = resolve_klein_text_encoder()
    missing = klein_missing_assets()
    if any(a in missing for a in KLEIN_REQUIRED):
        raise KleinModelsMissing(missing)

    comfy_input_dir = _comfy_input_dir()
    uid = uuid.uuid4().hex[:8]
    comfy_input = f"edit_source_{uid}_{source_filename}"
    shutil.copy2(source_path, os.path.join(comfy_input_dir, comfy_input))

    workflow["52"]["inputs"]["image"] = comfy_input
    workflow["145"]["inputs"]["text1"] = edit_prompt
    workflow["77"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    # UNIQUE prefix per job: SaveImage numbers files from what's currently in
    # ComfyUI's output folder, and the app MOVES each result out right after
    # completion — with a shared prefix the counter kept re-issuing the same
    # name (live repro: 4 different seeds/prompts all saved as
    # local_DatasetFace_00002_.png), so every tile displayed the same file,
    # each new generation overwriting the previous one in the dataset dir.
    workflow["9"]["inputs"]["filename_prefix"] = f"{user_id}_DatasetFace_{uid}"
    workflow["114"]["inputs"]["unet_name"] = unet_ref
    workflow["10"]["inputs"]["vae_name"] = vae_ref
    workflow["90"]["inputs"]["clip_name"] = te_ref

    # Multi-reference (native): chain each extra identity ref into the POSITIVE
    # conditioning — Klein reads several ReferenceLatent nodes natively. The
    # primary ref stays first and strongest (2 MP, node 92); extras add identity
    # signal from other angles at 1 MP. cfg=1 → the negative chain (110) is
    # ignored by the sampler, so it is left untouched. A missing extra file is
    # skipped silently (same tolerance as the Nano Banana multi-ref path).
    prev = "92"
    for i, ref_path in enumerate(extra_ref_paths or [], start=1):
        if not ref_path or not os.path.exists(ref_path):
            logger.warning(f"klein multi-ref: extra ref missing on disk: {ref_path}")
            continue
        ref_input = f"edit_ref{i}_{uid}_{os.path.basename(ref_path)}"
        shutil.copy2(ref_path, os.path.join(comfy_input_dir, ref_input))
        load_id, scale_id = f"ds_ref{i}_load", f"ds_ref{i}_scale"
        enc_id, lat_id = f"ds_ref{i}_encode", f"ds_ref{i}_latent"
        workflow[load_id] = {"class_type": "LoadImage", "inputs": {"image": ref_input},
                             "_meta": {"title": f"Extra identity ref {i}"}}
        workflow[scale_id] = {"class_type": "ImageScaleToTotalPixels",
                              "inputs": {"upscale_method": "lanczos", "megapixels": 1,
                                         "resolution_steps": 1, "image": [load_id, 0]},
                              "_meta": {"title": f"Scale extra ref {i}"}}
        workflow[enc_id] = {"class_type": "VAEEncode",
                            "inputs": {"pixels": [scale_id, 0], "vae": ["10", 0]},
                            "_meta": {"title": f"Encode extra ref {i}"}}
        workflow[lat_id] = {"class_type": "ReferenceLatent",
                            "inputs": {"conditioning": [prev, 0], "latent": [enc_id, 0]},
                            "_meta": {"title": f"Reference latent extra {i}"}}
        prev = lat_id
    if prev != "92":
        workflow["77"]["inputs"]["positive"] = [prev, 0]

    # Inject the consistency LoRA between the UNET (114) and the base LoRA node
    # (139) → chain 114 -> consistency -> 139. NOTE: this LoRA anchors STRUCTURE
    # (composition/background) — its own guide recommends ~0.5 and warns 0.8-1.0
    # can stop edits from applying. Strength 0 (slider fully left) skips the
    # node entirely so the base edit behaviour is reachable. Also skipped
    # (degraded but functional) if the LoRA file or node 139 is missing.
    consistency_lora, lora_path = _consistency_lora()
    strength = cfg.get('klein.consistency_strength')
    if lora_strength is not None:
        strength = max(0.0, min(1.5, float(lora_strength)))
    if "139" not in workflow:
        logger.warning("workflow node 139 missing — consistency LoRA injection skipped")
    elif not lora_path or not os.path.exists(lora_path):
        logger.warning(f"consistency LoRA not found at {lora_path} — injection skipped")
    elif not strength or float(strength) <= 0:
        logger.info("consistency LoRA strength 0 — injection skipped (LoRA off)")
    else:
        workflow["ds_consistency_lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": consistency_lora,
                       "strength_model": strength, "model": ["114", 0]},
            "_meta": {"title": "Dataset consistency LoRA"},
        }
        workflow["139"]["inputs"]["model"] = ["ds_consistency_lora", 0]

    # The base style LoRA in node 139 (klein\realistic.safetensors) belongs to the
    # source app's ComfyUI and is NOT part of the Klein install — bypass it when
    # its file is absent so ComfyUI doesn't fail validation on a missing LoRA. The
    # consistency LoRA injected above (if any) stays in the chain.
    base_lora = (workflow.get("139", {}).get("inputs", {}).get("lora_name") or '').replace('/', os.sep)
    loras_dir = cfg.comfyui_dir('loras')
    base_lora_path = os.path.join(str(loras_dir), base_lora) if (loras_dir and base_lora) else None
    if "139" in workflow and (not base_lora_path or not os.path.exists(base_lora_path)):
        logger.info("base LoRA %r absent — bypassing node 139", base_lora)
        _bypass_node(workflow, "139", "model")

    job_id = str(uuid.uuid4())
    meta = {"model_name": "klein_edit_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt=edit_prompt, job_id=job_id, metadata=meta)
    return job_id
