"""The Video Test Studio — playing a trained video LoRA back as a clip.

The video lane could train a LoRA and hand back a `.safetensors`, and that was
where it stopped. Judging the result meant leaving the app: copy the file into
ComfyUI by hand, open a graph someone else wrote, guess which knobs the training
recipe implied. This module closes that loop the way the image Test Studio
already does for image LoRAs — same idea, different medium.

WHAT THIS IS A PORT OF, AND WHY THAT MATTERS
--------------------------------------------
Nothing here was invented. The MiniMax H3 image-to-video graph, the turbo
distillation LoRA, the third-party 10Eros base, the sparse-attention grafts and
the latent upscale all come from a video pipeline that has been generating real
clips for months, and every constant below was measured there — several of them
the hard way, on runs that had to be interrupted. The comments carry those
measurements because a value without its measurement is a value the next person
will "clean up".

THE ONE DESIGN DECISION WORTH STATING: GRAFTS, NOT WORKFLOW VARIANTS
--------------------------------------------------------------------
`workflows/minimax_h3_i2v.json` holds eighteen nodes and no options. Turbo,
LoRA, sparse attention and upscale are grafted onto that graph at build time
(ids 600-810), never shipped as separate JSON files. Four options that combine
freely would otherwise be sixteen workflow files, fifteen of which nobody would
ever open again, and a bug fixed in one would live on in the other fifteen.

The build is a PURE function. It takes what it needs — including whether the
10Eros weight is on disk — and returns a graph. No Flask, no filesystem, no
database: the whole option matrix is testable without a GPU, which is the only
reason the pitfalls below can be pinned by tests at all.
"""

import sqlalchemy as sa
import logging
import json
import os
import subprocess
import re
import uuid

from lds_video import video_reference_catalog as vrc

logger = logging.getLogger(__name__)


N_CLIP = '13'           # CLIPLoader — Qwen3-VL text encoder
N_VAE_AUDIO = '24'


VIDEO_VAE_FP16 = 'minimax_h3_video_vae_fp16.safetensors'


from lds_sdk.h3_render import (  # noqa: F401 — retain the Video module's existing aliases
    sage_available as sage_available,
    vdn_node_compatible as vdn_node_compatible,
    ATTN_EDGE_PCT as ATTN_EDGE_PCT,
    ATTN_FAST_BACKENDS as ATTN_FAST_BACKENDS,
    ATTN_KITCHEN as ATTN_KITCHEN,
    ATTN_PYTORCH as ATTN_PYTORCH,
    ATTN_SAGE as ATTN_SAGE,
    BASE_EROS as BASE_EROS,
    BASE_LIGHT as BASE_LIGHT,
    BASE_OFFICIAL as BASE_OFFICIAL,
    BLOCK_ATTN_CLASS as BLOCK_ATTN_CLASS,
    BLOCK_ATTN_INSTALL_ACTION as BLOCK_ATTN_INSTALL_ACTION,
    CARD_SETUP_PATH as CARD_SETUP_PATH,
    DARETIES_LORA as DARETIES_LORA,
    DEFAULT_STEPS as DEFAULT_STEPS,
    FAST_DISK_MIN_COMFYUI as FAST_DISK_MIN_COMFYUI,
    FAST_DISK_RAM_FLOOR_GB as FAST_DISK_RAM_FLOOR_GB,
    FRAMES_DEFAULT as FRAMES_DEFAULT,
    FRAMES_MAX as FRAMES_MAX,
    FRAMES_MIN as FRAMES_MIN,
    H3_HOST_RAM_GB as H3_HOST_RAM_GB,
    LIGHT_MIN_COMFY as LIGHT_MIN_COMFY,
    LIGHT_MIN_COMFY_LABEL as LIGHT_MIN_COMFY_LABEL,
    LORA_EXT as LORA_EXT,
    LORA_SUBDIR as LORA_SUBDIR,
    MP_DEFAULT as MP_DEFAULT,
    MP_MAX as MP_MAX,
    MP_MIN as MP_MIN,
    N_ACCEL_LORA as N_ACCEL_LORA,
    N_BLOCK_ATTN as N_BLOCK_ATTN,
    N_COND as N_COND,
    N_CREATE_VIDEO as N_CREATE_VIDEO,
    N_DECODE_AUDIO as N_DECODE_AUDIO,
    N_DECODE_VIDEO as N_DECODE_VIDEO,
    N_GUIDER as N_GUIDER,
    N_LOAD_END as N_LOAD_END,
    N_LOAD_IMAGE as N_LOAD_IMAGE,
    N_NOISE as N_NOISE,
    N_REF_SPARSE_MEMORY as N_REF_SPARSE_MEMORY,
    N_SAGE as N_SAGE,
    N_SAMPLER as N_SAMPLER,
    N_SAMPLER_SELECT as N_SAMPLER_SELECT,
    N_SAVE as N_SAVE,
    N_SCALE as N_SCALE,
    N_SCHEDULER as N_SCHEDULER,
    N_SHIFT as N_SHIFT,
    N_SIZE as N_SIZE,
    N_SPARSE as N_SPARSE,
    N_TEST_LORA as N_TEST_LORA,
    N_TURBO_LORA as N_TURBO_LORA,
    N_TURBO_SAMPLER as N_TURBO_SAMPLER,
    N_UNET as N_UNET,
    N_UPSCALE as N_UPSCALE,
    N_UPSCALE_PARAMS as N_UPSCALE_PARAMS,
    N_UPSCALE_SPLIT as N_UPSCALE_SPLIT,
    N_VAE_VIDEO as N_VAE_VIDEO,
    N_VDN as N_VDN,
    OPTIONAL_WEIGHTS as OPTIONAL_WEIGHTS,
    OPTION_NODE_PACKS as OPTION_NODE_PACKS,
    PARASYTE_LORA as PARASYTE_LORA,
    REQUIRED_WEIGHTS as REQUIRED_WEIGHTS,
    SAGE_CLASS as SAGE_CLASS,
    SAGE_PACK as SAGE_PACK,
    SPARSE_BACKEND as SPARSE_BACKEND,
    SPARSE_MODES as SPARSE_MODES,
    SPARSE_PRESETS as SPARSE_PRESETS,
    TARGET_KEY as TARGET_KEY,
    TURBO_LORA as TURBO_LORA,
    TURBO_STEPS as TURBO_STEPS,
    UNFETCHABLE as UNFETCHABLE,
    UPSCALE_MODEL as UPSCALE_MODEL,
    VDN_BRANCH_FILE as VDN_BRANCH_FILE,
    VDN_BRANCH_FILE_INT8 as VDN_BRANCH_FILE_INT8,
    VDN_CLASS as VDN_CLASS,
    VDN_FOLDER as VDN_FOLDER,
    VDN_LIGHT_ADVICE as VDN_LIGHT_ADVICE,
    VDN_LIGHT_ADVICE_VRAM_GB as VDN_LIGHT_ADVICE_VRAM_GB,
    VDN_MODES as VDN_MODES,
    VDN_NODE_INPUTS as VDN_NODE_INPUTS,
    VDN_SETUP_PATH as VDN_SETUP_PATH,
    VDN_STAGE as VDN_STAGE,
    VDN_STAGE_REQUIRED as VDN_STAGE_REQUIRED,
    VDN_STEPS as VDN_STEPS,
    VDN_WINDOW_LATENT_FRAMES as VDN_WINDOW_LATENT_FRAMES,
    VIDEO_VAE_INT8 as VIDEO_VAE_INT8,
    WORKFLOW_FILENAME as WORKFLOW_FILENAME,
    DYNAMIC_VRAM_FORCE as _DYNAMIC_VRAM_FORCE,
    DYNAMIC_VRAM_OFF_MODES as _DYNAMIC_VRAM_OFF_MODES,
    DYNAMIC_VRAM_SWITCH as _DYNAMIC_VRAM_SWITCH,
    FAST_DISK_FLAG as _FAST_DISK_FLAG,
    FRAME_MOD as _FRAME_MOD,
    FRAME_OFFSET as _FRAME_OFFSET,
    HIGH_RAM_FLAG as _HIGH_RAM_FLAG,
    ROW_PRESENT as _ROW_PRESENT,
    VERSION_RE as _VERSION_RE,
    drop_sage as _drop_sage,
    graft_block_attention as _graft_block_attention,
    graft_latent_upscale as _graft_latent_upscale,
    graft_reference_accel as _graft_reference_accel,
    graft_sparse as _graft_sparse,
    graft_stock_accel as _graft_stock_accel,
    graft_test_lora as _graft_test_lora,
    graft_turbo as _graft_turbo,
    graft_vdn as _graft_vdn,
    insert_into_chain as _insert_into_chain,
    loras_write_dir as _loras_write_dir,
    make_t2v as _make_t2v,
    model_readers as _model_readers,
    parse_comfy_version as _parse_comfy_version,
    profile as _profile,
    vdn_row_present as _vdn_row_present,
    video_runs as _video_runs,
    weight_present as _weight_present,
    attention_spec as attention_spec,
    block_attention_backends as block_attention_backends,
    block_attention_missing_nodes as block_attention_missing_nodes,
    block_attention_pack_installed as block_attention_pack_installed,
    clamp_megapixels as clamp_megapixels,
    comfyui_launch_facts as comfyui_launch_facts,
    comfyui_vram_gb as comfyui_vram_gb,
    deploy_checkpoint as deploy_checkpoint,
    deploy_file as deploy_file,
    deployed_loras as deployed_loras,
    eros_on_disk as eros_on_disk,
    import_external_lora as import_external_lora,
    knows_fast_disk as knows_fast_disk,
    last_frame_command as last_frame_command,
    launch_advice as launch_advice,
    light_on_disk as light_on_disk,
    light_status as light_status,
    light_usable as light_usable,
    load_base_workflow as load_base_workflow,
    local_attention_backends as local_attention_backends,
    new_prefix as new_prefix,
    normalise_mode as normalise_mode,
    normalise_sparse as normalise_sparse,
    option_availability as option_availability,
    reference_accel_spec as reference_accel_spec,
    reference_sparse_compatible as reference_sparse_compatible,
    reference_weight_name as reference_weight_name,
    registered_classes as registered_classes,
    snap_frames as snap_frames,
    studio_ready as studio_ready,
    trained_loras as trained_loras,
    undeploy_lora as undeploy_lora,
    vdn_branch_active as vdn_branch_active,
    vdn_latent_frames as vdn_latent_frames,
    vdn_node_checkpoints as vdn_node_checkpoints,
    vdn_node_unwritten_required as vdn_node_unwritten_required,
    vdn_roots as vdn_roots,
    vdn_stage_missing as vdn_stage_missing,
    vdn_stage_missing_under as vdn_stage_missing_under,
    vdn_stage_present as vdn_stage_present,
    workflow_path as workflow_path,
)
from .h3_chimera import (
    ACCELERATIONS as ACCELERATIONS, ACCEL_IDS as ACCEL_IDS,
    accel_spec as accel_spec, accelerations_status as accelerations_status,
    build_workflow, missing_weights as missing_weights, normalise_accel as normalise_accel,
)
from .h3_performance import reference_status as reference_status
from .clipproj import preflight as preflight


class VideoStudioAssetsMissing(Exception):
    """Raised by the preflight when the graph cannot run on this install.

    Carries what is missing so the caller can name each file and node instead of
    letting ComfyUI answer "Value not in list: unet_name" three screens later.
    """

    def __init__(self, missing_files, missing_nodes):
        self.missing_files = list(missing_files)
        self.missing_nodes = list(missing_nodes)
        super().__init__('video studio assets missing: '
                         f'{len(self.missing_files)} file(s), '
                         f'{len(self.missing_nodes)} node(s)')


import shutil


def clips_dir(create=True):
    """Where finished test clips live, away from ComfyUI's output directory.

    Same reasoning as the image studio's per-dataset folder: the clip has to
    survive whatever ComfyUI does with its own outputs, and be served by the app
    under a path it controls.
    """
    from lds_sdk.video_host import config as cfg
    root = cfg.data_dir() / 'video_tests'
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def enqueue_clip(user_id, *, prompt, mode='i2v', image=None, end_image=None, lora=None,
                 lora_strength=1.0, run_id=None, dataset_id=None, seed=None,
                 steps=None, frames=None, megapixels=MP_DEFAULT, aspect='auto',
                 turbo=False, accel=None, eros=False, light=False, sparse='',
                 latent_upscale=False,
                 references=None, ref_base='official', ref_image_size='match', refmods=False,
                 fused=False, h3_attention='auto', h3_spectrum=False,
                 h3_video_vae='fp16', h3_video_writer='native',
                 source_ratio=None, skip_preflight=False, continues=None,
                 _prepared=None, _record=None, _remote=None) -> dict:
    """Build the graph, record the clip, queue the job — in that order.

    The row is written BEFORE the queue insert and in the SAME transaction, so a
    job can never exist without the row that explains what it was: the queue
    monitor's completion callback resolves by `job_id`, and a missing row means
    a finished clip nobody can attribute.
    """
    from lds_video.models import db
    from lds_sdk.video_runtime import queue as queue_manager
    from lds_video.models import VideoTestClip
    from lds_video import video_references as refs

    mode = normalise_mode(mode)
    if _remote is not None and (mode != 'ref2va' or _prepared is not None or latent_upscale):
        raise ValueError('The battle GPU supports Reference clips without latent upscale.')
    from lds_video.h3_refmods import validate as validate_refmods
    validate_refmods(mode, image, references, refmods)
    # Reference/text canvases are chosen independently of the inputs. Keep the
    # same normalized choice the builder uses, so Reuse cannot inherit a later
    # clip's format. An image-to-video canvas follows its source image.
    aspect = str(aspect or 'auto').strip().lower()
    if mode == 'i2v' or aspect not in ('auto', 'portrait', 'landscape', 'square'):
        aspect = 'auto'
    references = (refs.validate_references(references, user_id=user_id, enforce_limits=not refmods)
                  if mode == 'ref2va' or refmods else [])
    # A reference clip continues WITH its references (2026-09-07): the take
    # keeps the cast it was made with, and the seam is the H3 guide at frame 0
    # -- the parent's last frame, staged like any other first frame guide. The
    # blanket refusal this replaces was written on 2026-09-04, with the
    # reference mode itself and before those guides existed.
    # The picture is what makes it a continuation: without it the join would
    # cut to a clip that starts wherever the sampler took it, so the launch
    # says what is missing rather than rendering a jump.
    if mode == 'ref2va' and continues and not image:
        raise ValueError('A reference continuation starts on the last frame of the clip '
                         'it continues: stage it as the first frame guide.')

    from lds_video.video_best_settings import resolve_lora
    if _prepared is None:
        dataset_id, run_id = resolve_lora(user_id, lora, run_id=run_id, dataset_id=dataset_id)
    else:
        # Internal batch preparation already verified the exact checkpoint and
        # froze its weight. HTTP routes never accept these private arguments.
        dataset_id, run_id = _prepared['origin']

    # ONE /object_info read for the whole launch, and it decides two things:
    # whether SageAttention goes into the graph at all, and (through the
    # preflight below) whether an armed option's nodes are there. Reading it
    # twice would let a ComfyUI that restarts between the two answer differently
    # for the same clip.
    # ⏭ A continuation: the clip this one will be joined behind. Checked
    # before anything is queued — a parent that is not done has no last frame.
    parent = None
    if continues:
        parent = VideoTestClip.query.filter_by(id=int(continues)).first()
        # Two different situations, two sentences: a clip that is GONE is not
        # a clip that has not finished, and telling somebody to wait for a
        # render they deleted is a refusal they cannot act on. The continuation
        # outlives the card (it is kept in this browser), so this is reachable
        # by deleting the parent and coming back (verification, 2026-09-07).
        if parent is None:
            raise ValueError(f'clip #{int(continues)} no longer exists — drop the continuation '
                             'to launch a clip of its own')
        if parent.status != 'done' or not parent.filename:
            raise ValueError('the clip to continue has not finished rendering')
        if parent.user_id not in (None, str(user_id)):
            raise ValueError('the clip to continue belongs to another workspace')
    if _prepared is None:
        classes = set() if _remote is not None else registered_classes()
        light_ok, light_note = light_usable() if light else (False, '')
        built = build_workflow(
            prompt=prompt, mode=mode, image=image, end_image=end_image, seed=seed, steps=steps,
            frames=frames, megapixels=megapixels, aspect=aspect, lora=lora,
            lora_strength=lora_strength, turbo=turbo, accel=accel, eros=eros,
            eros_on_disk=eros_on_disk() if eros else False,
            light=light, light_on_disk=light_ok, light_note=light_note, sparse=sparse,
            latent_upscale=latent_upscale, source_ratio=source_ratio,
            references=references, ref_base=ref_base, ref_image_size=ref_image_size, refmods=refmods,
            fused=fused, h3_attention=h3_attention, h3_spectrum=h3_spectrum,
            h3_video_vae=h3_video_vae, h3_video_writer=h3_video_writer, performance_classes=classes,
            sage=sage_available(classes), filename_prefix=new_prefix(user_id))
    else:
        built = _prepared['built']
    if _remote is None:
        if built['generation_settings'].get('fused'):
            name = built['workflow'][N_UNET]['inputs']['unet_name']
            built['workflow'][N_UNET]['inputs']['unet_name'] = reference_weight_name(('diffusion_models', 'unet'), name) or name
        for node in built['workflow'].values():
            if node.get('class_type') == 'VAELoader':
                name = node['inputs'].get('vae_name')
                node['inputs']['vae_name'] = reference_weight_name(('vae',), name) or name
    if mode == 'ref2va' and _remote is None:
        if ref_base == 'light':
            _argv, _ram, version = comfyui_launch_facts()
            parsed = _parse_comfy_version(version)
            if parsed is not None and parsed < LIGHT_MIN_COMFY:
                raise ValueError(f'Reference W4A8 needs ComfyUI {LIGHT_MIN_COMFY_LABEL} or later.')
        _resolve_reference_weights(built['workflow'])
    if _remote is not None:
        _remote.validate_workflow(built['workflow'])
        built['generation_settings'].update(execution='battle_cloud', cloud_session_id=_remote.id)
    elif not skip_preflight:
        preflight(built['workflow'])

    job_id = str(uuid.uuid4())
    clip = VideoTestClip(
        run_id=run_id, dataset_id=dataset_id, job_id=job_id, status='pending',
        prompt=prompt, mode=mode, user_id=str(user_id), aspect=aspect,
        references_json=json.dumps(references) if references else None,
        ref_base=ref_base if mode == 'ref2va' else None,
        ref_image_size=ref_image_size if mode == 'ref2va' else None,
        source_image=image, end_image=(end_image or None), seed=built['seed'], steps=built['steps'],
        frames=built['frames'], megapixels=built['megapixels'],
        fps=float(_profile().get('fps') or 24.0), base_model=built['base'],
        lora=lora, lora_strength=built['generation_settings']['lora_strength'],
        generation_settings=json.dumps(built['generation_settings']),
        turbo=(built['accel'] == 'turbo'), accel=(built['accel'] or None),
        sparse=normalise_sparse(sparse), latent_upscale=bool(latent_upscale),
        continues_of=(parent.id if parent else None))
    db.session.add(clip)
    db.session.flush()          # mint the id inside the same transaction
    try:
        refs.keep_clip_references(clip.id, references, user_id=user_id)
        if _record is not None:
            _record(clip)
    except Exception:
        db.session.rollback()
        raise
    if _remote is not None:
        db.session.commit()
        try:
            keep_clip_frames(clip.id, image=image, end_image=end_image)
            _remote.submit(clip.id, built['workflow'], references=references, image=image,
                           end_image=end_image, lora=lora, user_id=user_id)
        except Exception:
            db.session.rollback()
            link_completed_clip(job_id, None, failed=True, reason='The cloud clip could not be started. Retry this turn.')
            raise
        return {'clip_id': clip.id, 'job_id': job_id, 'seed': built['seed'],
                'frames': built['frames'], 'steps': built['steps'], 'notes': built['notes'],
                'execution': 'battle_cloud', 'cloud_session_id': _remote.id}
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=built['workflow'], prompt=prompt or '',
                          job_id=job_id,
                          metadata={'model_name': 'video_lora_test',
                                    'is_video_test': True,
                                    'clip_id': clip.id,
                                    # Spared by the boot sweep while the
                                    # job is live (see job_queue); never
                                    # under `staged_inputs`, which drops
                                    # per job — a last frame is shared.
                                    'inputs_in_use': ([n for n in (image, end_image) if n]
                                                      + [r['name'] for r in references])},
                          commit=False)
    db.session.commit()
    # 🎬 The clip keeps its own copy of the pictures it was conditioned on:
    # ComfyUI's input folder is swept, the app's clips folder is not.
    keep_clip_frames(clip.id, image=image, end_image=end_image)
    logger.info('video studio: queued clip %s (%s)', clip.id,
                ', '.join(built['notes']) or 'no options')
    return {'clip_id': clip.id, 'job_id': job_id, 'seed': built['seed'],
            'frames': built['frames'], 'steps': built['steps'],
            'notes': built['notes']}


def _resolve_reference_weights(workflow):
    for node in workflow.values():
        inputs = node.get('inputs', {})
        if node.get('class_type') == 'UNETLoader':
            key, folders = 'unet_name', ('diffusion_models', 'unet')
        elif node.get('class_type') == vrc.REF_LORA_CLASS:
            key, folders = 'lora_name', ('loras',)
        else:
            continue
        inputs[key] = reference_weight_name(folders, inputs[key]) or inputs[key]


VFI_CKPT = 'rife49.pth'
VFI_MULTIPLIER = 2
VFI_CLEAR_CACHE_EVERY = 16
VFI_CRF = 19

N_VFI_LOAD, N_VFI_RIFE, N_VFI_SAVE = 'v1', 'v2', 'v3'


def build_vfi_workflow(*, video_path, fps, multiplier=VFI_MULTIPLIER,
                       filename_prefix=None) -> dict:
    """The interpolation graph for ONE finished clip. Pure — no disk, no queue.

    The source is given as an ABSOLUTE path: the clip lives in this app's own
    folder (clips_dir), never in ComfyUI's output, and VHS_LoadVideoPath takes a
    path rather than a name precisely so a file outside that tree can be read.
    The output rate is the source's times the multiplier, which is what makes
    this a SMOOTHING rather than a slow motion. RIFE inserts between pairs:
    N input frames produce (N - 1) * multiplier + 1 frames, so the encoded
    duration is slightly shorter (by less than one source frame).
    """
    rate = round(float(fps or 24) * int(multiplier), 3)
    return {
        N_VFI_LOAD: {
            'class_type': 'VHS_LoadVideoPath',
            'inputs': {'video': str(video_path), 'force_rate': 0,
                       'custom_width': 0, 'custom_height': 0,
                       'frame_load_cap': 0, 'skip_first_frames': 0,
                       'select_every_nth': 1, 'format': 'AnimateDiff'},
        },
        N_VFI_RIFE: {
            'class_type': 'RIFE VFI',
            'inputs': {'ckpt_name': VFI_CKPT,
                       'frames': [N_VFI_LOAD, 0],
                       'clear_cache_after_n_frames': VFI_CLEAR_CACHE_EVERY,
                       'multiplier': int(multiplier), 'fast_mode': True,
                       'ensemble': True, 'scale_factor': 2},
        },
        N_VFI_SAVE: {
            'class_type': 'VHS_VideoCombine',
            'inputs': {'images': [N_VFI_RIFE, 0], 'frame_rate': rate,
                       # VHS lazily reads this optional audio output. Its
                       # combine node also accepts a source with no audio.
                       'audio': [N_VFI_LOAD, 2],
                       'loop_count': 0,
                       'filename_prefix': filename_prefix or 'lds_vfi',
                       'format': 'video/h264-mp4', 'pix_fmt': 'yuv420p',
                       'crf': VFI_CRF, 'save_metadata': True,
                       'trim_to_audio': False, 'pingpong': False,
                       'save_output': True},
        },
    }


def interpolate_clip(user_id, clip_id, multiplier=VFI_MULTIPLIER) -> dict:
    """↗ Smooth a finished clip: queue a RIFE pass over its own file.

    A NEW row, never an edit of the old one: the smoothed clip is a different
    artefact with a different frame rate, and overwriting the original would
    destroy the comparison the studio exists for. It carries the source's
    settings so the card still says what made it, plus `vfi_of` so the pair
    stays readable.

    Refused rather than queued when the interpolation nodes are absent — the
    job would otherwise die inside ComfyUI with a message nobody sees.
    """
    from lds_video.models import db
    from lds_sdk.video_runtime import queue as queue_manager
    from lds_video.models import VideoTestClip

    src = VideoTestClip.query.filter_by(id=int(clip_id)).first()
    if src is not None and src.user_id not in (None, str(user_id)):
        raise ValueError('clip not found')
    if src is None:
        raise ValueError('clip not found')
    if src.status != 'done' or not src.filename:
        raise ValueError('that clip has not finished rendering yet')
    path = os.path.join(str(clips_dir()), os.path.basename(src.filename))
    if not os.path.isfile(path):
        raise ValueError('that clip is no longer on disk')
    if src.frames is not None and 0 < src.frames < 2:
        raise ValueError('Smooth needs at least 2 frames in the source clip')

    classes = registered_classes()
    spec = OPTION_NODE_PACKS['vfi']
    if classes is not None:
        gone = [c for c in spec['classes'] if c not in classes]
        if gone:
            raise ValueError(
                f"this ComfyUI has no {', '.join(gone)} — install "
                f"{spec['pack']} ({spec['search']} in ComfyUI-Manager) and "
                f"try again")

    mult = max(2, min(8, int(multiplier or VFI_MULTIPLIER)))
    job_id = str(uuid.uuid4())
    workflow = build_vfi_workflow(video_path=path, fps=src.fps or 24,
                                  multiplier=mult,
                                  filename_prefix=new_prefix(user_id))
    clip = VideoTestClip(
        run_id=src.run_id, dataset_id=src.dataset_id, job_id=job_id,
        generation_settings=src.generation_settings,
        status='pending', prompt=src.prompt, mode=src.mode, aspect=src.aspect or 'auto',
        user_id=str(user_id), references_json=src.references_json,
        ref_base=src.ref_base, ref_image_size=src.ref_image_size,
        accel=src.accel,
        source_image=src.source_image, end_image=getattr(src, 'end_image', None),
        seed=src.seed, steps=src.steps,
        # RIFE writes each endpoint once, plus mult-1 frames between each pair.
        # Unknown legacy counts stay unknown; the source row is never edited.
        frames=(src.frames - 1) * mult + 1 if src.frames and src.frames > 0 else None,
        megapixels=src.megapixels, fps=round(float(src.fps or 24) * mult, 3),
        base_model=src.base_model, lora=src.lora,
        lora_strength=src.lora_strength, turbo=bool(src.turbo),
        sparse=src.sparse, latent_upscale=bool(src.latent_upscale),
        vfi_of=src.id)
    db.session.add(clip)
    db.session.flush()
    from lds_video import video_references as refs
    try:
        refs.keep_clip_references(clip.id, refs.references_of(src), user_id=user_id)
    except Exception:
        db.session.rollback()
        raise
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=src.prompt or '',
                          job_id=job_id,
                          metadata={'model_name': 'video_lora_test',
                                    'is_video_test': True,
                                    'clip_id': clip.id},
                          commit=False)
    db.session.commit()
    logger.info('video studio: queued VFI x%s of clip %s as %s',
                mult, src.id, clip.id)
    return {'clip_id': clip.id, 'job_id': job_id, 'multiplier': mult,
            'fps': clip.fps}


def link_completed_clip(job_id, filename, failed=False, reason=None, *, render_seconds=None, postprocess=True):
    """Attach a finished ComfyUI job to its clip row.

    Runs in the queue monitor thread, whose session may hold a stale read
    snapshot — hence the rollback-and-re-read before concluding the row is
    absent, exactly as the image studio's callback does.

    The mp4 comes back under the history's `images` key like any other output
    (that is how `SaveVideo` reports), so nothing in the polling path had to
    change for video.
    """
    from lds_video.models import db
    from lds_video.models import VideoTestClip
    clip = VideoTestClip.query.filter_by(job_id=job_id).first()
    if clip is None:
        db.session.rollback()
        clip = VideoTestClip.query.filter_by(job_id=job_id).first()
    if clip is None:
        logger.warning('video studio: no clip row for job %s', job_id)
        return
    if clip.status != 'pending':
        logger.info('video studio: clip %s already %s — late completion ignored',
                    clip.id, clip.status)
        return
    # Read before the row settles, on the failure path too: a clip that died
    # after four minutes of rendering says something a bare "failed" does not.
    clip.render_seconds = _render_seconds(job_id) if render_seconds is None else round(max(0, render_seconds), 1)
    if failed:
        clip.status = 'failed'
        clip.error = (reason or 'Generation failed (see the server log in '
                                'Settings for the ComfyUI error).')
        db.session.commit()
        return
    clip.filename = filename
    clip.status = 'done'
    _bring_clip_home(filename)
    clip_id = clip.id
    # Committed BEFORE any join: from here the row is a valid render (the
    # part), and the encode below runs with no write transaction open. It ran
    # inside this one once — measured: every other writer got "database is
    # locked" for the length of ffmpeg (up to its 600 s timeout).
    db.session.commit()
    if not postprocess:
        # Cloud settles the row under its short cancellation lock, then joins
        # locally outside that lock so Stop can release the paid GPU at once.
        return clip_id
    postprocess_completed_clip(clip_id)


def _run_ffmpeg(cmd, timeout=600):
    """The app's subprocess convention for ffmpeg (video_bank_service has the
    same): no console window in the frozen Windows build, utf-8 stderr with
    replacement, and a timeout."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=timeout,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


STAGED_FRAME_NAME = re.compile(r'^lds_vstudio_[0-9a-f]{10}\.png$')
FRAME_CHECKPOINT = 'start frame'
FRAMES_DATASET_NAME = 'Video Test Studio · frames'
FRAMES_DATASET_LEGACY_NAMES = ('Video Test Studio · start frames',)
FRAME_COPIES_MAX = 100


def staged_frame_path(name: str, role: str = 'start') -> str:
    """The file behind a staged start (or last) frame — by its NAME only, under ComfyUI's
    input folder. Anything that is not the exact shape `/source` hands out is
    refused before the filesystem is touched: the name comes from the client,
    and a name that walks is how a viewer becomes a file reader."""
    from lds_sdk.video_host import config as cfg
    base = os.path.basename(str(name or ''))
    if not STAGED_FRAME_NAME.match(base):
        raise ValueError('not a staged start frame')
    folder = cfg.comfyui_dir('input')
    if not folder:
        raise ValueError("ComfyUI's input folder is not configured — set the ComfyUI folder in Settings")
    path = os.path.join(str(folder), base)
    if not os.path.isfile(path):
        raise LookupError(f'that {role} frame is no longer staged — pick it again')
    return path


def frames_dataset_id(user_id) -> int:
    """The holding dataset for the Video Test Studio's frames — start frames
    opened in the viewer, and every finished clip's collected last frame —
    created on first use. It is an ordinary dataset: it shows in the Datasets
    list under its name, and deleting it drops every frame with it. Found
    under a name it had before, it is renamed, so no install ends up with two."""
    from lds_sdk import video_frames
    return video_frames.ensure_collection(user_id, FRAMES_DATASET_NAME,
        legacy_names=FRAMES_DATASET_LEGACY_NAMES, trigger_word='videoframe')


def _as_int(value, what):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{what} must be a number') from None


def adopt_frame(dataset_id=None, *, image=None, gallery_image_id=None, user_id='local'):
    """🔍 Give a start frame a library row, so the shared viewer's verbs can
    address it.

    The viewer every surface of the app opens on a picture (✨ improve, 🔍
    upscale, ✦ repair, 📷 camera angles, 📤 Civitai) works on `lora_test_image`
    rows — that is what its routes resolve. A staged start frame is a file in
    ComfyUI's input folder with no row, so the SAME buttons the Gallery shows
    would be dead on it. Adopting it means: the picture copied into a dataset's
    folder (where every library image lives and is served from) and one row
    that says what it is — `derivation_kind` VIDEO_START_FRAME, so it is never
    read as a cell of the Test Studio.

    Two ways in: a picture the Gallery already owns (`gallery_image_id` — the
    row itself, no copy) or a staged name (`image`; a clip's last frame arrives
    this way too, `/clip/<id>/last-frame` having staged it). `dataset_id` is
    the page's image dataset when the Studio was opened from one; None means
    the holding dataset (`frames_dataset_id`).

    CONTENT-ADDRESSED: the copy is named by the hash of the bytes, so the same
    picture opened twice, or staged from two tabs, is ONE row and an improve
    result lands next to it rather than next to a duplicate. And the name is
    re-checked against the bytes: ✦ Repair rewrites a library file in place,
    so a copy whose bytes moved on is left alone and the same source takes the
    next free name — re-picking the original must not reopen the repaired one.
    """
    import hashlib
    from lds_sdk import video_frames
    if gallery_image_id is not None and gallery_image_id != '':
        row = video_frames.gallery(user_id, _as_int(gallery_image_id, 'gallery_image_id'))
        if row is None or not row.filename:
            raise LookupError('that generated image is not in the gallery any more')
        return row
    if not image:
        raise ValueError('name a staged frame or a gallery image')
    restage_frame(image)                  # back from the clip's copy if the sweep took it
    src = staged_frame_path(image, role=frame_role(image))
    if dataset_id is None or dataset_id == '':
        ds_id = frames_dataset_id(user_id)
    else:
        ds_id = _as_int(dataset_id, 'dataset_id')
        if video_frames.collection(user_id, ds_id) is None:
            raise LookupError('dataset not found')
    with open(src, 'rb') as fh:
        digest = hashlib.sha1(fh.read()).hexdigest()[:16]
    folder = str(video_frames.dataset_directory(user_id, ds_id))

    def same_bytes(path):
        try:
            with open(path, 'rb') as fh:
                return hashlib.sha1(fh.read()).hexdigest()[:16] == digest
        except OSError:
            return False

    dest = None
    for n in range(FRAME_COPIES_MAX):
        cand = f'vsframe_{digest}.png' if n == 0 else f'vsframe_{digest}_{n + 1}.png'
        if not os.path.isfile(os.path.join(folder, cand)) or same_bytes(os.path.join(folder, cand)):
            dest = cand
            break
    if dest is None:
        raise ValueError('too many copies of this frame')
    target = os.path.join(folder, dest)
    # Copy first, then look for the row: a row whose file went missing gets
    # its file back and stays ONE row, rather than a second row on the same name.
    if not os.path.isfile(target):
        shutil.copy2(src, target)
    existing = video_frames.existing(user_id, ds_id, dest)
    if existing is not None:
        return existing
    return video_frames.create_frame(user_id, ds_id, filename=dest,
        checkpoint=FRAME_CHECKPOINT, prompt='Start frame of the Video Test Studio — a staged start frame.',
        derivation_kind='video_start_frame')


def clip_frame_sidecar(clip_id, which) -> str:
    """`clip_<id>_first.png` / `clip_<id>_end.png` beside the mp4: the clip's
    own copy of the picture it starts from / ends on (`clip_<id>_last.png` is
    the extracted last frame, see `last_frame_png`)."""
    return os.path.join(str(clips_dir()), f'clip_{int(clip_id)}_{which}.png')


def keep_clip_frames(clip_id, *, image=None, end_image=None) -> None:
    """🎬 Copy the clip's staged frames from ComfyUI's input folder to the app's
    own clips folder, at enqueue time — the one moment they are surely there.

    ComfyUI's input folder is not ours: the boot sweep clears every staged
    picture older than 48 h (`comfy_fs`), and before it did, a start frame
    staged once stayed there for good (415 MB on one install). The clip's
    folder IS ours, lives as long as the clip, and is where ↻ Reuse finds the
    picture again (`restage_frame`). Best-effort: a copy that fails is a
    warning, never a launch refused.
    """
    from lds_sdk.video_host import config as cfg
    folder = cfg.comfyui_dir('input')
    if not folder:
        return
    for which, name in (('first', image), ('end', end_image)):
        base = os.path.basename(str(name or ''))
        if not base:
            continue
        src = os.path.join(str(folder), base)
        try:
            if os.path.isfile(src):
                shutil.copy2(src, clip_frame_sidecar(clip_id, which))
        except OSError as exc:
            logger.warning('video studio: the %s frame of clip %s was not kept: %s', which, clip_id, exc)


def keep_frames_of_existing_clips() -> tuple[int, set]:
    """🎬 Once per boot, BEFORE the sweep: clips rendered before the copies
    existed get theirs from ComfyUI's input folder, while their frames are
    still there. Without this, the first boot after the update would have
    cleared every frame an existing clip could be reused from. Returns how
    many copies were made and the NAMES that could not be copied (disk
    full, a file held open): the sweep spares those, so a copy that failed
    never costs the only picture left (found in verification, 2026-09-04).
    A clip whose copy exists, or whose frame is already gone, costs one stat."""
    from lds_sdk.video_host import config as cfg
    from lds_video.models import VideoTestClip
    folder = cfg.comfyui_dir('input')
    if not folder or not os.path.isdir(str(folder)):
        return 0, set()
    made = 0
    failed = set()
    rows = VideoTestClip.query.filter(sa.or_(VideoTestClip.source_image.isnot(None),
                                             VideoTestClip.end_image.isnot(None))).all()
    for clip in rows:
        for which, name in (('first', clip.source_image), ('end', clip.end_image)):
            base = os.path.basename(str(name or ''))
            if not base or not STAGED_FRAME_NAME.match(base):
                continue
            side = clip_frame_sidecar(clip.id, which)
            src = os.path.join(str(folder), base)
            if os.path.isfile(side) or not os.path.isfile(src):
                continue
            try:
                shutil.copy2(src, side)
                made += 1
            except OSError as exc:
                failed.add(base)
                logger.warning('video studio: the %s frame of clip %s was not kept: %s', which, clip.id, exc)
    if made:
        logger.info('video studio: kept the frames of %s existing clip(s) beside them', made)
    return made, failed


def frame_role(name) -> str:
    """'last' when the staged name is known as a clip's END frame and not as
    any clip's start — for the sentence a viewer gets on a frame that is
    gone; 'start' otherwise."""
    from lds_video.models import VideoTestClip
    base = os.path.basename(str(name or ''))
    if not base:
        return 'start'
    if VideoTestClip.query.filter(VideoTestClip.source_image == base).first() is not None:
        return 'start'
    if VideoTestClip.query.filter(VideoTestClip.end_image == base).first() is not None:
        return 'last'
    return 'start'


def restage_frame(name) -> bool:
    """A staged frame back in ComfyUI's input folder, from the copy a clip kept.

    ↻ Reuse and the viewer address a frame by its staged NAME; once the boot
    sweep has cleared it (48 h), the name points at nothing and ComfyUI would
    refuse the graph a minute later. So: present → True; gone but kept by a
    clip that started from it or ended on it → copied back under the same
    name, True; gone and kept by nobody (a clip deleted since, a frame staged
    before the copies existed) → False, and the caller says "pick it again".
    A name that is not one this app staged is not judged here (True).
    """
    from lds_sdk.video_host import config as cfg
    from lds_video.models import VideoTestClip
    base = os.path.basename(str(name or ''))
    if not base or not STAGED_FRAME_NAME.match(base):
        return True
    from lds_sdk.video_host import comfy_fs
    folder = cfg.comfyui_dir('input')
    if not folder:
        return False
    dest = os.path.join(str(folder), base)
    if os.path.isfile(dest):
        return True
    # The folder is checked, never created: an unmounted or read-only input
    # folder must raise the named ComfyFolderUnavailable (-> 409 with the
    # Settings hint) exactly as staging does, not be conjured up locally
    # where ComfyUI would never read it.
    comfy_fs.ensure_input_usable(folder)
    for which, column in (('first', VideoTestClip.source_image), ('end', VideoTestClip.end_image)):
        for clip in VideoTestClip.query.filter(column == base).order_by(VideoTestClip.id.desc()).all():
            side = clip_frame_sidecar(clip.id, which)
            if not os.path.isfile(side):
                continue
            try:
                shutil.copy2(side, dest)
            except OSError as exc:
                logger.warning('video studio: %s could not be restaged from clip %s: %s', base, clip.id, exc)
                continue
            logger.info('video studio: restaged %s from clip %s', base, clip.id)
            return True
    return False


PREVIOUS_PARTS_MAX = 3


def previous_parts(clip_id, limit=PREVIOUS_PARTS_MAX) -> list:
    """⏭ The prompts of the parts a continuation follows, most recent first:
    the clip being continued, then the one IT continued, up to `limit` — what
    the ✨ writers are handed so the next part carries the take on instead of
    starting over. An unknown id yields nothing; a chain that loops cannot run
    past `limit`; a part without a prompt is skipped, not counted."""
    from lds_video.models import VideoTestClip
    from lds_sdk.video_host.config import LOCAL_USER
    out = []
    seen = set()
    try:
        cur = int(clip_id)
    except (TypeError, ValueError):
        return out
    while cur and cur not in seen and len(out) < limit:
        seen.add(cur)
        clip = VideoTestClip.query.filter_by(id=cur).first()
        if clip is None or clip.user_id not in (None, str(LOCAL_USER)):
            break
        if str(clip.prompt or '').strip():
            out.append(str(clip.prompt))
        cur = clip.continues_of
    return out


def last_frame_png(clip_id) -> str:
    """The clip's last frame as a PNG next to its mp4 (`clip_<id>_last.png`),
    extracted once and kept; the picture the next clip starts from."""
    from lds_video.models import VideoTestClip
    from lds_sdk.video_host.config import LOCAL_USER
    from lds_sdk.video_host import ffmpeg_tools
    clip = VideoTestClip.query.filter_by(id=int(clip_id)).first()
    if clip is None or clip.user_id not in (None, str(LOCAL_USER)):
        raise LookupError('clip not found')
    if clip.status != 'done' or not clip.filename:
        raise ValueError('that clip has not finished rendering yet')
    root = str(clips_dir())
    src = os.path.join(root, os.path.basename(clip.filename))
    if not os.path.isfile(src):
        raise ValueError('that clip is no longer on disk')
    dst = os.path.join(root, f'clip_{int(clip_id)}_last.png')
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    ffmpeg = ffmpeg_tools.ffmpeg_path()
    if not ffmpeg:
        raise ValueError('ffmpeg is needed to read the last frame — install the video extra from Setup')
    # Extracted under a name of its own, then published in one rename: the
    # continuation and the Rendered clip tab's preview can ask for this
    # picture within a second of each other, and two ffmpegs writing the
    # same path could leave a half-written PNG in the Gallery for good
    # (found in verification, 2026-09-04). Same extension, so ffmpeg still
    # picks the PNG muxer from the name.
    tmp = os.path.join(root, f'.clip_{int(clip_id)}_last.{uuid.uuid4().hex[:8]}.png')
    try:
        r = _run_ffmpeg(last_frame_command(ffmpeg, src, tmp), timeout=120)
        if r.returncode != 0 or not os.path.isfile(tmp):
            raise ValueError(f'the last frame could not be read: {(r.stderr or "")[-300:]}')
        os.replace(tmp, dst)
    finally:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return dst


def _probe_media(ffmpeg, path) -> dict:
    """What ffmpeg sees in a file, read off its own `-i` banner: whether it
    has a sound track, and how long it plays. The join needs both — a source
    (or an older smoothed clip) can be silent, and a silent side joined to a
    sounding one must be padded with silence of ITS
    length, not muted along with the other side (found in verification,
    2026-09-03: the blind `-an` fallback threw the new part's sound away)."""
    r = _run_ffmpeg([ffmpeg, '-hide_banner', '-i', path], timeout=60)
    text = (r.stderr or '') + (getattr(r, 'stdout', '') or '')
    audio = re.search(r'Stream #\d+:\d+.*?: Audio', text) is not None
    m = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', text)
    duration = None
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return {'audio': audio, 'duration': duration}


def continuation_command(ffmpeg, parent, part, dst, width, height, fps, *, part_fps=None,
                         parent_audio=True, part_audio=True, parent_seconds=None,
                         part_seconds=None):
    """Parent, then the new part: one video, at the parent's cadence.

    The part's FIRST frame is dropped — it is the parent's last frame, the
    picture the part was conditioned on, and kept it freezes the join for one
    frame — and its audio is trimmed by the same 1/fps (the PART's fps: a
    smoothed parent plays at 48 while its part rendered at 24) so the two stay
    in step. Both sides are forced to one constant cadence (`fps=`): concat
    accepts mixed rates and writes a variable-rate file whose stated fps then
    lies to Smooth and to the card. The part is scaled to the parent's size in
    case the dials changed between the two.

    Sound: each side that has a track keeps it (resampled to one format, or
    concat refuses the pair); a side without one is padded with silence of
    its own length, so the other side's sound survives. Neither side sounding
    → no track. Metadata is dropped: a studio clip carries its whole
    generation graph in a tag, and the join is not that graph."""
    fps = float(fps or 24)
    part_fps = float(part_fps or fps)
    trim = 1.0 / part_fps
    rate = f'{fps:g}'
    graph = (f'[1:v]trim=start_frame=1,setpts=PTS-STARTPTS,'
             f'scale={int(width)}:{int(height)}:flags=lanczos,setsar=1,fps={rate}[v1];'
             f'[0:v]setsar=1,fps={rate}[v0];[v0][v1]concat=n=2:v=1:a=0[v]')
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', parent, '-i', part]
    fmt = 'aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo'
    if parent_audio or part_audio:
        if parent_audio:
            a0 = f'[0:a]{fmt}[a0]'
        else:
            if parent_seconds is None:
                raise ValueError('a silent parent needs its length to be padded with silence')
            a0 = f'anullsrc=r=48000:cl=stereo:d={max(float(parent_seconds), 0.01):.3f}[a0]'
        if part_audio:
            a1 = f'[1:a]atrim=start={trim:.6f},asetpts=PTS-STARTPTS,{fmt}[a1]'
        else:
            if part_seconds is None:
                raise ValueError('a silent part needs its length to be padded with silence')
            a1 = f'anullsrc=r=48000:cl=stereo:d={max(float(part_seconds) - trim, 0.01):.3f}[a1]'
        graph += f';{a0};{a1};[a0][a1]concat=n=2:v=0:a=1[a]'
        cmd += ['-filter_complex', graph, '-map', '[v]', '-map', '[a]', '-c:a', 'aac', '-b:a', '128k']
    else:
        cmd += ['-filter_complex', graph, '-map', '[v]', '-an']
    cmd += ['-c:v', 'libx264', '-crf', '19', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-map_metadata', '-1', dst]
    return cmd


def _join_continuation(clip_id) -> bool:
    """⏭ The finished part joined behind the clip it continues, as THIS clip's
    file: what the card plays is the whole, the parent followed by the new
    motion. The parent stays as it is; the part is not kept apart.

    Called AFTER the completion is committed, and committing on its own: the
    row is already 'done' with the part as its file — a valid render — while
    ffmpeg runs, so no write lock is held for the length of an encode (the
    Bank's passes keep the same rule, `_release_db_before_inference`). A join
    that fails, or raises, leaves the part as the clip and says so in
    `error`, which the card shows under its lineage line."""
    from lds_video.models import db
    from lds_video.models import VideoTestClip
    from lds_sdk.video_host import ffmpeg_tools

    def _fail(msg):
        row = VideoTestClip.query.filter_by(id=int(clip_id)).first()
        if row is not None:
            row.error = f'continuation not joined: {msg}'[:300]
            db.session.commit()
        return False

    # 1. What the join needs, read in one short transaction.
    clip = VideoTestClip.query.filter_by(id=int(clip_id)).first()
    if clip is None or not clip.continues_of or not clip.filename:
        return False
    parent = VideoTestClip.query.filter_by(id=int(clip.continues_of)).first()
    root = str(clips_dir())
    part = os.path.join(root, os.path.basename(clip.filename))
    if parent is None or not parent.filename:
        return _fail('the clip it continues is gone')
    src = os.path.join(root, os.path.basename(parent.filename))
    if not os.path.isfile(src) or not os.path.isfile(part):
        return _fail('a file is missing')
    ffmpeg = ffmpeg_tools.ffmpeg_path()
    if not ffmpeg:
        return _fail('ffmpeg is not available')
    parent_id, parent_frames, part_frames = parent.id, parent.frames, clip.frames
    fps = float(parent.fps or clip.fps or 24)
    part_fps = float(clip.fps or fps)
    try:
        from PIL import Image
        with Image.open(last_frame_png(parent_id)) as im:
            width, height = im.size
    except Exception as exc:  # noqa: BLE001 — no size, no scaling target
        return _fail(str(exc))
    db.session.commit()   # nothing dirty — this only ends the read before the encode

    # 2. The encode, with no transaction open.
    stem, _ = os.path.splitext(os.path.basename(part))
    out_name = f'{stem}_joined.mp4'
    dst = os.path.join(root, out_name)
    try:
        p_src, p_part = _probe_media(ffmpeg, src), _probe_media(ffmpeg, part)
        parent_audio, part_audio = p_src['audio'], p_part['audio']
        # A silent side is padded with silence of its own length; when that
        # length cannot be read the pair goes without sound rather than out of step.
        if ((not parent_audio and part_audio and p_src['duration'] is None)
                or (parent_audio and not part_audio and p_part['duration'] is None)):
            parent_audio = part_audio = False
        r = _run_ffmpeg(continuation_command(
            ffmpeg, src, part, dst, width, height, fps, part_fps=part_fps,
            parent_audio=parent_audio, part_audio=part_audio,
            parent_seconds=p_src['duration'], part_seconds=p_part['duration']))
        if r.returncode != 0 or not os.path.isfile(dst):
            raise RuntimeError((r.stderr or '')[-300:] or 'ffmpeg wrote nothing')
    except Exception as exc:  # noqa: BLE001 — a timeout, a vanished binary, a refused graph: the part stays
        try:
            os.remove(dst)
        except OSError:
            pass
        # A timeout's own text is the whole command line — the two paths alone
        # outrun the 300 characters the card can show; say the one fact.
        msg = (f'ffmpeg timed out after {exc.timeout:.0f} s'
               if isinstance(exc, subprocess.TimeoutExpired) else str(exc))
        logger.warning('video studio: clip %s could not be joined behind clip %s: %s',
                       clip_id, parent_id, msg)
        return _fail(msg)

    # 3. Name the joint on the row and commit — then, and only then, let the part go.
    row = VideoTestClip.query.filter_by(id=int(clip_id)).first()
    if row is None:
        try:
            os.remove(dst)
        except OSError:
            pass
        return False
    row.filename = out_name
    if parent_frames and part_frames:
        # The part's frames, minus the dropped one, at the parent's cadence.
        row.frames = int(parent_frames) + int(round((int(part_frames) - 1) * fps / part_fps))
    row.fps = fps
    row.error = None
    db.session.commit()
    try:
        os.remove(part)
    except OSError:
        pass
    logger.info('video studio: clip %s joined behind clip %s -> %s', clip_id, parent_id, out_name)
    return True


def _render_seconds(job_id):
    """Seconds the queue spent on a job: from the worker's claim (`started_at`,
    stamped when the job is taken, so the wait in the queue is excluded) to the
    moment it settled (`completed_at`). ComfyUI's model loading is inside that
    window on purpose — it is what the user waited for, and on a machine whose
    RAM cannot hold the weights it is most of the number.

    None whenever the queue cannot say: no job row (a clip settled by hand), a
    stamp missing, a clock that went backwards — and a job the queue CANCELLED.
    That last one is the ComfyUI-restart path: the job stalls with its
    `started_at` kept, the barrier waits for the user, and `completed_at` is
    stamped when the barrier is reconciled, hours later if need be. The
    difference then measures the outage, not the render, and a card saying
    "failed after 6 h" would be a lie. A number here is a measurement, never a
    guess.
    """
    from lds_sdk.video_runtime import job as queue_job
    job = queue_job(job_id)
    if job is None or job.status not in ('completed', 'failed'):
        return None
    if not job.started_at or not job.completed_at:
        return None
    secs = (job.completed_at - job.started_at).total_seconds()
    return round(secs, 1) if secs >= 0 else None


def _bring_clip_home(filename):
    """Bring the finished mp4 out of ComfyUI's output dir into the app's own.

    Disk claim first, HTTP `/view` fetch as the fallback: a ComfyUI pointed at a
    custom output directory (`--output-directory`, the desktop app's setting)
    has a path this app cannot know, but its API serves the file regardless.
    A failure here is not fatal — the row keeps the filename and the clip is
    still readable from ComfyUI.
    """
    from lds_sdk.video_host import studio as lts
    from lds_sdk.video_host import comfy_fs
    dst = os.path.join(str(clips_dir()), filename)
    if os.path.exists(dst):
        return
    out_dir = lts.comfy_output_dir()
    src = os.path.join(out_dir, filename) if out_dir else None
    try:
        # Same claim as the two image lanes: copy into place, treat dest-present
        # as success, unlink the source only if it lets us. A mp4 ComfyUI has
        # just flushed is the likeliest of our outputs to still be held open,
        # and `shutil.move`'s copy+unlink fallback raised AFTER the bytes had
        # landed — the half-claimed clip then satisfied the `dst` guard above
        # for good, and the /view fallback below was skipped by the raise.
        if src and comfy_fs.claim_output_file(src, dst):
            return
        from lds_sdk.video_host.comfyui import fetch_output_image_bytes
        data = fetch_output_image_bytes(filename)
        if data:
            with open(dst, 'wb') as fh:
                fh.write(data)
        else:
            logger.warning('video studio: %s is not on disk and /view returned '
                           'nothing — the clip stays with ComfyUI', filename)
    except OSError:
        logger.exception('video studio: could not bring %s home', filename)




def postprocess_completed_clip(clip_id):
    """Join a committed continuation, without extracting unused last frames."""
    from lds_video.models import db
    from lds_video.models import VideoTestClip
    clip = db.session.get(VideoTestClip, clip_id)
    if clip is None or clip.status != 'done':
        return
    continues = getattr(clip, 'continues_of', None)
    if continues:
        # ⏭ The part becomes the whole: parent, then this render.
        _join_continuation(clip_id)
    # last_frame_png remains lazy: Continue, an explicit frame preview or
    # frame selection extracts it when needed, without adding Gallery rows.
