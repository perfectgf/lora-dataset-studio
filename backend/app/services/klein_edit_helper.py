"""Focused helper to enqueue a SINGLE Flux.2 Klein edit job.

Reused by the face-dataset fan-out. Builds a minimal job from the same
single-edit workflow /generate_edit uses in single mode (WORKFLOW_IMPROVE_SKIN_PATH,
nodes 52=LoadImage, 6=CLIPTextEncode prompt, 77=KSampler seed, 9=SaveImage,
114=UNETLoader Klein). Deliberately a small focused helper rather than a refactor
of the large live generate_edit route.

Node preflight: the shipped 'improve skin.json' is written to use ONLY core +
comfy_extras nodes (the prompt text goes straight into the CLIPTextEncode widget,
and the image-size wiring skips the rgthree 'Any Switch' passthroughs) so a stock
ComfyUI with zero custom-node packs can run it. `klein_missing_nodes()` re-checks
this against the live /object_info as a safety net — if a reverted/edited workflow
or a future change reintroduces a custom node the target ComfyUI lacks, the route
answers one actionable "install pack X, restart ComfyUI" 409 instead of ComfyUI's
raw 400 'missing_node_type'.

Lifted from the parent project's app/services/klein_edit_helper.py for LoRA
Dataset Studio: SRC's module-level COMFYUI_INPUT_DIR/COMFYUI_OUTPUT_DIR constants
become live `cfg.comfyui_dir(...)` calls (config.json changes take effect without
a restart, and an unconfigured ComfyUI raises RuntimeError instead of writing
into a hardcoded path), and the hardcoded consistency-LoRA filename/strength are
now the `klein.consistency_lora` / `klein.consistency_strength` settings.

SUPPORTS extra_model_paths.yaml (Option A): all model resolution functions now
search both the main ComfyUI models/ folder AND any extra paths defined under
'unet', 'vae', 'text_encoders', or 'loras' in the YAML file.
FALLBACK: if YAML is missing, reads from config.json -> comfyui.extra_model_paths.
"""
from __future__ import annotations
import logging
import os
import random
import shutil
import time
import uuid

# yaml is required for extra_model_paths.yaml support
try:
    import yaml
except ImportError:
    yaml = None
    logging.getLogger(__name__).warning("PyYAML not installed - extra_model_paths.yaml support disabled")

from .. import config as cfg
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_IMPROVE_SKIN_PATH = cfg.BACKEND_DIR / 'workflows' / 'improve skin.json'

# Nodes this helper rewires — fail LOUDLY if the workflow file changes shape
# instead of silently enqueuing a job with the wrong source/prompt/model.
# Node 6 = CLIPTextEncode: the per-job prompt is written straight into its `text`
# widget (the RES4LYF TextBox1 node 145 that used to hold it was removed so the
# graph needs no custom-node packs).
_REQUIRED_NODES = ('52', '6', '77', '9', '114', '10', '90')

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


def _get_comfy_root():
    """Infer the ComfyUI root folder from the models directory."""
    models_root = _models_root()
    if models_root:
        return os.path.dirname(models_root)
    return None


# Cache for parsed extra_model_paths.yaml to avoid repeated file I/O.
_EXTRA_PATHS_CACHE = None


def _map_key_to_canonical(key):
    """Map YAML keys to canonical model types."""
    if key in ('unet', 'diffusion_models'):
        return 'unet'
    if key in ('text_encoders', 'clip'):
        return 'text_encoders'
    if key == 'vae':
        return 'vae'
    if key == 'loras':
        return 'loras'
    return None


def _get_extra_paths_for_type(model_type):
    """Return a list of absolute directory paths for the given model type
    ('unet', 'vae', 'text_encoders', 'loras') as defined in extra_model_paths.yaml.
    Returns an empty list if the file is missing, unparseable, or the type is absent.
    
    Mappings: 'diffusion_models' -> 'unet', 'clip' -> 'text_encoders' to support
    common StabilityMatrix and custom setups.
    
    Also supports both top-level keys: 'extra_model_paths' (standard) and 'Comfyui'
    (as used by some custom configs).
    
    FALLBACK: if YAML fails, read from config.json -> comfyui.extra_model_paths.
    """
    global _EXTRA_PATHS_CACHE
    if yaml is None:
        return []

    if _EXTRA_PATHS_CACHE is None:
        _EXTRA_PATHS_CACHE = {}
        comfy_root = _get_comfy_root()
        if not comfy_root:
            logger.debug("Could not determine ComfyUI root for extra_model_paths.yaml")
            # Fallback: try config
            return _get_extra_paths_from_config(model_type)

        yaml_path = os.path.join(comfy_root, 'extra_model_paths.yaml')
        if not os.path.exists(yaml_path):
            logger.debug("extra_model_paths.yaml not found at %s", yaml_path)
            return _get_extra_paths_from_config(model_type)

        try:
            with open(yaml_path, 'r', encoding='utf-8-sig') as f:  # handle BOM
                data = yaml.safe_load(f) or {}
            # Find the top-level dict: try 'extra_model_paths' first, then 'Comfyui' (case-insensitive)
            top_key = None
            if 'extra_model_paths' in data:
                top_key = 'extra_model_paths'
            else:
                for key in data:
                    if key.lower() == 'comfyui':
                        top_key = key
                        break
            if top_key is None:
                logger.warning("extra_model_paths.yaml missing 'extra_model_paths' or 'Comfyui' top-level key")
                return _get_extra_paths_from_config(model_type)

            extra_data = data[top_key]
            if not isinstance(extra_data, dict):
                logger.warning("extra_model_paths.yaml top-level value is not a dict")
                return _get_extra_paths_from_config(model_type)

            # If top_key is 'Comfyui', treat the entire extra_data as the paths dict (flat structure)
            # Otherwise (standard 'extra_model_paths'), it may have sections, so we flatten them.
            if top_key.lower() == 'comfyui':
                # Flat structure: keys like 'clip', 'diffusion_models' are directly paths.
                for key, path in extra_data.items():
                    canonical = _map_key_to_canonical(key)
                    if canonical and isinstance(path, str):
                        abs_path = os.path.abspath(path)
                        if os.path.isdir(abs_path):
                            _EXTRA_PATHS_CACHE.setdefault(canonical, []).append(abs_path)
                        else:
                            logger.warning("Extra path %s for key %s does not exist", abs_path, key)
            else:
                # Standard nested structure: sections like 'comfyui', 'a111' each have their own paths.
                for section_name, paths in extra_data.items():
                    if not isinstance(paths, dict):
                        continue
                    for key, path in paths.items():
                        canonical = _map_key_to_canonical(key)
                        if canonical and isinstance(path, str):
                            abs_path = os.path.abspath(path)
                            if os.path.isdir(abs_path):
                                _EXTRA_PATHS_CACHE.setdefault(canonical, []).append(abs_path)
                            else:
                                logger.warning("Extra path %s for key %s does not exist", abs_path, key)

        except Exception as e:
            logger.warning("Could not parse extra_model_paths.yaml: %s", e)
            return _get_extra_paths_from_config(model_type)

    return _EXTRA_PATHS_CACHE.get(model_type, [])


def _get_extra_paths_from_config(model_type):
    """Fallback: read extra paths from config.json -> comfyui.extra_model_paths."""
    extra = cfg.get('comfyui.extra_model_paths')
    if not isinstance(extra, dict):
        return []
    # Map config keys (unet, text_encoders, vae, loras) to our internal type
    # Also accept aliases like 'diffusion_models' or 'clip'
    mapping = {
        'unet': 'unet',
        'diffusion_models': 'unet',
        'text_encoders': 'text_encoders',
        'clip': 'text_encoders',
        'vae': 'vae',
        'loras': 'loras',
    }
    # Direct match using the exact key from config
    for key, canonical in mapping.items():
        if canonical == model_type:
            val = extra.get(key)
            if val and isinstance(val, str) and os.path.isdir(val):
                return [os.path.abspath(val)]
    # Also check if the model_type itself is a key
    if model_type in extra and os.path.isdir(extra[model_type]):
        return [os.path.abspath(extra[model_type])]
    return []


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
    """Search for a model file in the main models/ folder AND in any extra paths
    defined in extra_model_paths.yaml for the given model type.
    subparts: tuple of folder fragments under models/, e.g. ('unet',) or ('text_encoders',)
    canonical: the exact filename we prefer
    tokens: fallback substrings if canonical is not found
    Returns the filename (not full path) if found, else None."""
    root = _models_root()
    if not root:
        return None

    model_type = subparts[0] if subparts else None
    # Base directories: main models/ folder first, then extras
    base_dirs = [root]
    extra_dirs = _get_extra_paths_for_type(model_type)
    if extra_dirs:
        base_dirs.extend(extra_dirs)

    for base in base_dirs:
        if base == root:
            # For the main root, always append subparts
            folder = os.path.join(base, *subparts)
            try:
                names = sorted(n for n in os.listdir(folder) if n.lower().endswith(_MODEL_SUFFIXES))
            except OSError:
                continue
            if canonical in names:
                return canonical
            for n in names:
                if any(tok in n.lower() for tok in tokens):
                    return n
        else:
            # For extra paths: search the base folder directly first,
            # then try with model_type appended.
            folders_to_try = [base]
            # Only append model_type if the basename doesn't already match
            basename_lower = os.path.basename(base).lower()
            # Remove underscores and hyphens for comparison
            normalized_basename = basename_lower.replace('_', '').replace('-', '')
            normalized_model_type = model_type.replace('_', '').replace('-', '')
            if normalized_model_type not in normalized_basename:
                folders_to_try.append(os.path.join(base, model_type))
            # Also try the classic appended path (in case the user uses the exact subfolder)
            folders_to_try.append(os.path.join(base, model_type))

            for folder in folders_to_try:
                try:
                    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(_MODEL_SUFFIXES))
                except OSError:
                    continue
                if canonical in names:
                    return canonical
                for n in names:
                    if any(tok in n.lower() for tok in tokens):
                        return n

    return None


def _klein_unet_folders():
    """Scan both the main models/ and extra_model_paths.yaml locations for
    Klein UNET folders. Returns a list of (subfolder_prefix, [filenames]).
    For main paths, the prefix is the subfolder (e.g., 'klein' or 'Flux2 klein').
    For extra paths, the prefix may be '' (empty) for bare files, or the subfolder name.
    This mirrors how ComfyUI's model picker sees files.
    
    NEW: Also scans the top-level folders (the base of unet/diffusion_models)
    for files that contain 'klein' in their name, to support models placed directly
    in the root (e.g., StabilityMatrix diffusionmodels/)."""
    root = _models_root()
    out = []

    # --- Scan main models/ folder ---
    if root:
        for base_name in ('unet', 'diffusion_models'):
            base_dir = os.path.join(root, base_name)
            # 1. Check the base folder itself for any file with 'klein' in the name
            try:
                all_files = sorted(n for n in os.listdir(base_dir)
                                   if n.lower().endswith(_MODEL_SUFFIXES))
                klein_files = [f for f in all_files if 'klein' in f.lower()]
                if klein_files:
                    out.append(('', klein_files))  # empty prefix = bare filename
            except OSError:
                pass

            # 2. Check subfolders that contain 'klein' in their name
            try:
                subs = sorted(d for d in os.listdir(base_dir)
                              if 'klein' in d.lower() and os.path.isdir(os.path.join(base_dir, d)))
            except OSError:
                continue
            for sub in subs:
                try:
                    names = sorted(n for n in os.listdir(os.path.join(base_dir, sub))
                                   if n.lower().endswith(_MODEL_SUFFIXES))
                except OSError:
                    continue
                if names:
                    out.append((sub, names))

    # --- Scan extra_model_paths.yaml locations ---
    extra_dirs = _get_extra_paths_for_type('unet')
    for extra_base in extra_dirs:
        # 1. Check the extra folder itself for bare model files
        try:
            all_files = sorted(n for n in os.listdir(extra_base)
                               if n.lower().endswith(_MODEL_SUFFIXES))
            klein_files = [f for f in all_files if 'klein' in f.lower()]
            if klein_files:
                out.append(('', klein_files))
        except OSError:
            pass

        # 2. Check subfolders inside the extra folder that contain 'klein'
        try:
            subs = sorted(d for d in os.listdir(extra_base)
                          if 'klein' in d.lower() and os.path.isdir(os.path.join(extra_base, d)))
        except OSError:
            continue
        for sub in subs:
            try:
                names = sorted(n for n in os.listdir(os.path.join(extra_base, sub))
                               if n.lower().endswith(_MODEL_SUFFIXES))
            except OSError:
                continue
            if names:
                out.append((sub, names))

    return out


def resolve_klein_unet(selected=None):
    """ComfyUI-relative `unet_name` for node 114, or None if no Klein model is on
    disk. Searches both main models/ and extra_model_paths.yaml locations.
    Returns the value WITH its subfolder prefix if it lives under main models/,
    or just the bare filename if it comes from an extra path or the top-level unet/.
    Preference: the picker's choice (searched across ALL locations), then the
    canonical download, then the first file found."""
    folders = _klein_unet_folders()
    if not folders:
        return None

    bare_pick = os.path.basename(selected) if selected else None
    # 1. Try to match the explicitly selected filename
    if bare_pick:
        for sub, names in folders:
            if bare_pick in names:
                # If sub is empty (extra path top-level), join gives just bare_pick
                return os.path.join(sub, bare_pick)

    # 2. Try the canonical filename
    canonical = _canonical_name('klein_model')
    for sub, names in folders:
        if canonical in names:
            return os.path.join(sub, canonical)

    # 3. Fallback to the first file found
    sub, names = folders[0]
    return os.path.join(sub, names[0])


def resolve_klein_vae():
    """`vae_name` for node 10 — canonical flux2-vae.safetensors, else a narrow
    flux2-vae token match (covers the 'flux2_vae.safetensors.safetensors'
    double-extension variant some installs carry). Searches main and extras."""
    return _find_model_file(('vae',), _canonical_name('klein_vae'),
                            ('flux2-vae', 'flux2_vae', 'flux-2-vae', 'flux_2_vae'))


def resolve_klein_text_encoder():
    """`clip_name` for node 90 — canonical qwen_3_8b_fp8mixed.safetensors, else a
    narrow qwen_3_8b token match. NEVER a bare 'qwen' match: qwen3vl_* (Z-Image)
    and qwen_2.5_vl_* (Qwen-Image) encoders live in the same folder and produce
    incompatible embeddings. Searches main and extras."""
    return _find_model_file(('text_encoders',), _canonical_name('klein_text_encoder'),
                            ('qwen_3_8b', 'qwen3_8b', 'qwen-3-8b'))


def _consistency_lora():
    """(relative_name, absolute_path) of the configured consistency LoRA, or
    (name, None) if not found. Searches the main ComfyUI loras folder and
    any extra loras folders from extra_model_paths.yaml.
    The name is the relative path under any of these folders (e.g., 'klein/consistency.safetensors')."""
    name = (cfg.get('klein.consistency_lora') or '').replace('/', os.sep)
    if not name:
        return None, None

    # 1. Check main loras folder
    main_lora_dir = cfg.comfyui_dir('loras')
    if main_lora_dir:
        full_path = os.path.join(str(main_lora_dir), name)
        if os.path.exists(full_path):
            return name, full_path

    # 2. Check extra loras folders
    extra_dirs = _get_extra_paths_for_type('loras')
    for extra_base in extra_dirs:
        full_path = os.path.join(extra_base, name)
        if os.path.exists(full_path):
            # ComfyUI's LoraLoader only loads from main loras folder.
            # We'll log a warning and return the filename (which likely fails).
            logger.warning(
                "Consistency LoRA found in extra path %s but ComfyUI only loads LoRAs from the main loras folder. "
                "Please move it to %s or use a symlink.", full_path, main_lora_dir
            )
            return os.path.basename(name), full_path

    # Not found
    return None, None


def klein_missing_assets():
    """Which Klein assets are NOT on disk, as setup_installer action names (a
    subset of KLEIN_REQUIRED + KLEIN_RECOMMENDED). Drives both the generate-time
    block and the auto-download. Now checks across main and extra paths."""
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


# --- Custom-node preflight -------------------------------------------------
# The class_types the shipped 'improve skin.json' historically pulled from custom
# packs, mapped to the pack that ships each + its GitHub page. Setup installs
# neither, and no doc mentioned them, so a fresh install hit a raw ComfyUI 400
# 'missing_node_type' on the first generation. The workflow no longer references
# these (strategy A), so this map is the safety net for a reverted/edited
# workflow — and the generic fallback (pack/url = None) covers any OTHER custom
# node a future change might introduce.
KLEIN_NODE_PACKS = {
    'TextBox1': ('RES4LYF', 'https://github.com/ClownsharkBatwing/RES4LYF'),
    'Any Switch (rgthree)': ('rgthree-comfy', 'https://github.com/rgthree/rgthree-comfy'),
}


def _workflow_class_types(workflow):
    return {n.get('class_type') for n in (workflow or {}).values()
            if isinstance(n, dict) and n.get('class_type')}


# Success-only TTL cache for the node preflight on the SHIPPED workflow:
# /object_info is a multi-MB payload, so don't re-fetch it for every tile
# regenerate click. Only an "all nodes present" verdict is cached (node packs
# don't uninstall mid-session); a miss or an unreachable probe is NEVER cached,
# so the "install the pack, restart ComfyUI, retry" flow re-probes immediately.
_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def klein_missing_nodes(workflow=None):
    """[{class_type, pack, url}] for every node class the Klein edit workflow needs
    that the target ComfyUI does NOT expose (i.e. absent from its /object_info
    keys). Loads the shipped 'improve skin.json' when no `workflow` is given.

    FAIL-OPEN: returns [] when /object_info can't be fetched (fetch returns None)
    — a transient probe failure must never block generation (mirrors the Test
    Studio node preflight in lora_test_studio.preflight_family)."""
    global _nodes_ok_until
    shipped = workflow is None
    if shipped:
        if time.time() < _nodes_ok_until:
            return []
        workflow = load_workflow_local(str(WORKFLOW_IMPROVE_SKIN_PATH)) or {}
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = []
    for ct in sorted(_workflow_class_types(workflow) - available):
        pack, url = KLEIN_NODE_PACKS.get(ct, (None, None))
        out.append({'class_type': ct, 'pack': pack, 'url': url})
    if shipped and not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def format_missing_nodes_message(missing_nodes):
    """Human sentence for a Klein node-missing 409: each missing class_type with the
    pack that provides it + its GitHub link, then the fix instruction. Reused by
    the datasets route's existing Klein-missing error path."""
    bits = []
    for n in missing_nodes:
        ct, pack, url = n.get('class_type'), n.get('pack'), n.get('url')
        if pack and url:
            bits.append(f"{ct} (from {pack}: {url})")
        elif pack:
            bits.append(f"{ct} (from {pack})")
        else:
            bits.append(str(ct))
    return ("Your ComfyUI is missing custom node(s) this Klein workflow needs: "
            + '; '.join(bits) + ". Install via ComfyUI-Manager, then restart ComfyUI.")


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
                       extra_ref_paths=None, sampler_steps=None,
                       base_lora_strength=None):
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
    # Prompt into the CLIPTextEncode widget directly (node 6). The old RES4LYF
    # TextBox1 (node 145) that used to carry it was dropped to de-depend the graph
    # from custom-node packs; node 6's `text` is a plain STRING input.
    workflow["6"]["inputs"]["text"] = edit_prompt
    workflow["77"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    if sampler_steps is not None:
        workflow["77"]["inputs"]["steps"] = max(1, int(sampler_steps))
    if base_lora_strength is not None and "139" in workflow:
        workflow["139"]["inputs"]["strength_model"] = float(base_lora_strength)
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
        # Note: lora_name must be a path relative to the main loras folder.
        # If we are using extra paths, we may need to use a symlink or copy.
        # We'll use the name as is; if it's not in main loras, ComfyUI will fail.
        # The check above already warns.
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