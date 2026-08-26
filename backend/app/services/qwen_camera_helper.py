"""📷 Camera angles — the ComfyUI side: resolve the weights, submit one view.

The lane's REASON for existing is in services/camera_angles.py (short version:
Klein turns the subject, this turns the camera). What lives here is the same
job every other local engine does — stage the source into ComfyUI's input,
point each loader node at what is actually on disk, enqueue — written against
the Qwen-Image-Edit-2511 graph instead of the Klein one.

THE RESOLUTION RULE IS NOT NEGOTIABLE, and it is the reason this module reuses
klein_edit_helper's finder rather than growing its own: **a wrong model is far
worse than a missing one.** Missing triggers the Setup download; wrong runs and
fails inside the sampler with "mat1 and mat2 shapes cannot be multiplied", or —
worse — succeeds and produces a picture nobody can explain. That is not
hypothetical here: `models/text_encoders/` on a shared ComfyUI holds at least
three Qwen encoders that are NOT interchangeable —

    qwen_3_8b_fp8mixed          -> FLUX.2 Klein
    qwen3vl_4b_fp8_scaled       -> Z-Image / Krea 2
    qwen_2.5_vl_7b_fp8_scaled   -> Qwen-Image-Edit  (this lane)

— and a naive "first file containing qwen" match picks the wrong one every
time. So: canonical filename first, then a NARROW token, never a first-file
guess.

WHAT THIS LANE DOES NOT OWN. The VAE. `qwen_image_vae.safetensors` is already
installed by the Krea 2 lane (setup_installer._KREA_DOWNLOADS['krea_vae']) and
it is the same file — Qwen-Image and Krea 2 share it. Declaring a second
download of the same bytes under another key would make Setup offer the same
gigabyte twice and let the two copies drift; this lane resolves the one that is
there and says so in `camera_missing_assets`.
"""
from __future__ import annotations
import logging
import os
import random
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager
from .klein_edit_helper import (_lora_abs, normalize_rel_model_name,
                                resolve_model_ref)

logger = logging.getLogger(__name__)

WORKFLOW_CAMERA_PATH = cfg.BACKEND_DIR / 'workflows' / 'qwen_camera_angles.json'

# Nodes the shipped graph MUST still contain. Checked before anything is staged
# or enqueued: a template edited into a different shape has to fail by name here
# rather than produce a job whose every tile dies in ComfyUI.
_REQUIRED_NODES = ('108', '109', '102', '93', '95', '41', '107', '105',
                   '112', '106', '9')

# Setup keys, in the order a missing-assets message should read them.
# `krea_vae` and not a `camera_vae` of our own: the Qwen VAE is ALREADY a Setup
# download (the Krea 2 lane installs the identical file to the identical place).
# Declaring a second key for the same bytes would make the Setup screen offer
# the same gigabyte twice and let two copies drift apart — so this lane names
# the button that already exists.
CAMERA_VAE_ACTION = 'krea_vae'
CAMERA_REQUIRED = ('camera_model', 'camera_lora', 'camera_text_encoder',
                   CAMERA_VAE_ACTION)
# Not required: the lane runs without it, just slower (4 steps -> 20+). Kept out
# of CAMERA_REQUIRED on purpose, so a user who deleted it still gets pictures.
CAMERA_RECOMMENDED = ('camera_speed_lora',)

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')


class CameraModelsMissing(Exception):
    """Graph-critical Qwen assets are absent. Carries the Setup action keys, so
    the route can offer the very buttons that install them (same contract as
    KleinModelsMissing — the 409 the frontend already knows how to read)."""

    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__('camera-angle models missing: ' + ', '.join(self.missing))


def _spec(action):
    """The download spec for one Setup key, from the merged catalog.

    Merged, not `_CAMERA_DOWNLOADS`, because one of this lane's four required
    assets is NOT this lane's download: the Qwen VAE arrives with Krea 2 (see
    the module docstring). Reading the merged table is what lets `camera_vae`
    below name the action that actually installs it."""
    from .. import setup_installer
    return setup_installer._MODEL_DOWNLOADS[action]


def _canonical(action):
    return _spec(action)['dest'][-1]


def _rel_dest(action):
    """The ComfyUI-relative name of a download's destination, subfolder included
    ('qwen/Qwen-...safetensors'). What a loader node wants."""
    return os.path.join(*_spec(action)['dest'][1:])


def _scan(comfy_type, canonical, tokens, rel_dest=None):
    """The ComfyUI-relative name of a model for `comfy_type`, or None.

    Order, and every step of it is deliberate:
      1. the canonical destination INCLUDING its subfolder, under any search
         root — the file Setup installs, found where Setup puts it;
      2. the canonical BARE filename at the root of a search folder — someone
         who dropped it in by hand;
      3. the first (sorted) name matching a NARROW token, root first, then ONE
         level of subfolders.
    Never a blind first-file guess.

    ⚠️ Step 3 descends one level, and that is the difference from the Klein
    lane's finder (which lists the root only). It has to: every asset here ships
    into a `qwen/` subfolder, and a scan that could not see into it would report
    an installed model as missing and offer to download 20 GB again. One level,
    not a full walk — `models/loras` on a working install holds tens of
    thousands of files across dozens of families, and a recursive scan of it on
    every status poll is a stall, not a search.
    """
    roots = list(comfy_model_paths.search_roots(comfy_type))
    if rel_dest:
        want = normalize_rel_model_name(rel_dest)
        for root in roots:
            if os.path.isfile(os.path.join(root, want)):
                return want
    for root in roots:
        if os.path.isfile(os.path.join(root, canonical)):
            return canonical
    for root in roots:
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for n in names:
            if (n.lower().endswith(_MODEL_SUFFIXES)
                    and any(t in n.lower() for t in tokens)):
                return n
        for sub in names:
            subdir = os.path.join(root, sub)
            if not os.path.isdir(subdir):
                continue
            try:
                inner = sorted(os.listdir(subdir))
            except OSError:
                continue
            for n in inner:
                if (n.lower().endswith(_MODEL_SUFFIXES)
                        and any(t in n.lower() for t in tokens)):
                    return normalize_rel_model_name(os.path.join(sub, n))
    return None


def _configured(comfy_type, cfg_key):
    """A user-pinned model for this slot, respelled as a loader wants it, or None."""
    value = (cfg.get(cfg_key) or '').strip()
    return resolve_model_ref(comfy_type, value) if value else None


def resolve_camera_unet():
    """`unet_name` for node 108 — the pin, then the canonical repack, then a
    narrow token. `qwen_image_edit` is narrow enough to exclude Qwen-Image
    (text-to-image) and every other family; the 2511 generation is tried first,
    because the LoRA was trained on it and a 2509 build would load happily and
    quietly under-perform."""
    pinned = _configured('diffusion_models', 'camera.unet')
    if pinned:
        return pinned
    for tokens in (('qwen_image_edit_2511', 'qwen-image-edit-2511'),
                   ('qwen_image_edit', 'qwen-image-edit')):
        found = _scan('diffusion_models', _canonical('camera_model'), tokens,
                      rel_dest=_rel_dest('camera_model'))
        if found:
            return found
    return None


def resolve_camera_text_encoder():
    """`clip_name` for node 93 — the Qwen 2.5-VL 7B encoder, and ONLY that one.
    See the module docstring: two other Qwen encoders live in the same folder
    and produce embeddings this graph cannot use."""
    return (_configured('text_encoders', 'camera.text_encoder')
            or _scan('text_encoders', _canonical('camera_text_encoder'),
                     ('qwen_2.5_vl_7b', 'qwen2.5_vl_7b', 'qwen2_5_vl_7b'),
                     rel_dest=_rel_dest('camera_text_encoder')))


def resolve_camera_vae():
    """`vae_name` for node 95. The Krea 2 lane's file — same bytes, one copy."""
    return (_configured('vae', 'camera.vae')
            or _scan('vae', _canonical(CAMERA_VAE_ACTION),
                     ('qwen_image_vae', 'qwen-image-vae'),
                     rel_dest=_rel_dest(CAMERA_VAE_ACTION)))


def _resolve_lora(action, cfg_key, tokens):
    """(relative name, absolute path) for one of the lane's LoRAs. The path is
    None when the file is not on disk — the NAME is still returned, so a log or
    a refusal can print what was looked for rather than an empty string."""
    pinned = (cfg.get(cfg_key) or '').strip()
    if pinned:
        rel = resolve_model_ref('loras', pinned)
        return normalize_rel_model_name(rel), _lora_abs(rel)
    found = _scan('loras', _canonical(action), tokens, rel_dest=_rel_dest(action))
    if found:
        return found, _lora_abs(found)
    return normalize_rel_model_name(_rel_dest(action)), None


def resolve_camera_lora():
    """The Multiple-Angles LoRA. Without it the base model still edits — it just
    ignores `<sks>` and answers the way any edit model does, by turning the
    subject. That is precisely the failure this lane exists to avoid, so it is
    REQUIRED rather than recommended: a silently angle-less "camera view" would
    look like a success."""
    return _resolve_lora('camera_lora', 'camera.angles_lora',
                         ('multiple-angles', 'multiple_angles'))


def resolve_camera_speed_lora():
    """The 4-step Lightning LoRA. Optional: absent, the graph runs at 20 steps
    with the same weights, roughly five times slower."""
    return _resolve_lora('camera_speed_lora', 'camera.speed_lora',
                         ('lightning-4steps', 'lightning_4steps', 'lightning-8steps'))


# Steps to run when the speed LoRA is NOT installed. The distilled 4-step regime
# is a property of that LoRA, not of the base model; keeping 4 without it would
# render noise, which is the kind of "configured for the developer's machine"
# failure this repo has already paid for once.
STEPS_WITH_SPEED_LORA = 4
STEPS_WITHOUT_SPEED_LORA = 20


def camera_missing_assets():
    """Setup action keys for every camera-angle asset that is NOT on disk.

    Ordered as CAMERA_REQUIRED then CAMERA_RECOMMENDED, so a message built from
    it reads worst-first."""
    missing = []
    if not resolve_camera_unet():
        missing.append('camera_model')
    if not resolve_camera_lora()[1]:
        missing.append('camera_lora')
    if not resolve_camera_text_encoder():
        missing.append('camera_text_encoder')
    if not resolve_camera_vae():
        missing.append(CAMERA_VAE_ACTION)
    if not resolve_camera_speed_lora()[1]:
        missing.append('camera_speed_lora')
    return missing


def camera_ready():
    """True when a view can actually be rendered right now."""
    return not any(a in camera_missing_assets() for a in CAMERA_REQUIRED)


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_camera_view(user_id, source_filename, source_path, pose_prompt,
                        extra_metadata=None, seed=None,
                        model_name='qwen_camera_angle'):
    """Stage the source, point the graph at the installed weights, enqueue ONE
    view. Returns the app job_id.

    One view per job on purpose. A single job rendering eight poses would share
    a seed, a failure and a queue slot: one bad pose would take the other seven
    with it, and the gallery could not show the first result until the last one
    finished. The picker's "8 views" is therefore eight jobs, and the pictures
    arrive one by one.
    """
    if not source_path or not os.path.exists(source_path):
        raise ValueError(f'source image not found: {source_filename}')
    workflow = load_workflow_local(str(WORKFLOW_CAMERA_PATH))
    if not workflow:
        raise ValueError('failed to load the camera-angle workflow')
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(
                f'workflow node {node} missing — qwen_camera_angles.json has changed')

    unet_ref = resolve_camera_unet()
    te_ref = resolve_camera_text_encoder()
    vae_ref = resolve_camera_vae()
    angles_lora, angles_path = resolve_camera_lora()
    missing = camera_missing_assets()
    if any(a in missing for a in CAMERA_REQUIRED):
        raise CameraModelsMissing(missing)

    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    staged_source = comfy_fs.stage_input_image(
        source_path, f'camera_source_{uid}_{stem}.png', comfy_input_dir)
    comfy_input = os.path.basename(staged_source)
    staged_inputs = [comfy_input]

    workflow['41']['inputs']['image'] = comfy_input
    workflow['112']['inputs']['prompt'] = pose_prompt
    workflow['108']['inputs']['unet_name'] = unet_ref
    workflow['93']['inputs']['clip_name'] = te_ref
    workflow['95']['inputs']['vae_name'] = vae_ref
    workflow['109']['inputs']['lora_name'] = angles_lora
    workflow['106']['inputs']['seed'] = (int(seed) if seed is not None
                                         else random.randint(0, 2 ** 64 - 1))

    # The speed LoRA is the only optional node. Absent, it is removed from the
    # chain AND the step count is raised to match — dropping it while leaving 4
    # steps is what would render noise.
    speed_lora, speed_path = resolve_camera_speed_lora()
    if speed_path:
        workflow['102']['inputs']['lora_name'] = speed_lora
        workflow['106']['inputs']['steps'] = STEPS_WITH_SPEED_LORA
    else:
        logger.info('camera lane: speed LoRA absent — %d steps instead of %d',
                    STEPS_WITHOUT_SPEED_LORA, STEPS_WITH_SPEED_LORA)
        workflow['94']['inputs']['model'] = ['109', 0]      # skip node 102
        workflow.pop('102', None)
        workflow['106']['inputs']['steps'] = STEPS_WITHOUT_SPEED_LORA

    # Unique per job: SaveImage numbers from what is in ComfyUI's output folder
    # and the app MOVES each result out, so a shared prefix makes the counter
    # re-issue one name and every view overwrite the last (measured on the Klein
    # lane — four different prompts all saved as ..._00002_.png).
    workflow['9']['inputs']['filename_prefix'] = f'{user_id}_CameraAngle_{uid}'

    job_id = str(uuid.uuid4())
    # Two stamps for one workflow, because the completion routes ON THIS NAME:
    # 'qwen_camera_angle' rides `is_lora_test` back to the gallery table, while
    # 'qwen_camera_dataset' is in DATASET_IMAGE_JOB_NAMES and lands as a
    # FaceDatasetImage. One name for both would strand one lane's results —
    # the exact bug test_dataset_job_harvest exists to catch.
    meta = {'model_name': model_name}
    if extra_metadata:
        meta.update(extra_metadata)
    meta['staged_inputs'] = staged_inputs
    queue_manager.add_job(job_type='image', user_id=str(user_id), workflow_data=workflow,
                          prompt=pose_prompt, job_id=job_id, metadata=meta)
    return job_id
