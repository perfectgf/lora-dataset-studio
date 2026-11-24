"""H3 rendering primitives shared by independent products (SDK 1.10).

Graph construction, model inspection and LoRA deployment contain no Video Bank,
dataset, clip history, route or product activation. Callers own their queue jobs.
"""
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import uuid
from . import h3_targets as video_targets
from . import h3_reference_catalog as vrc
logger = logging.getLogger(__name__)
N_VAE_AUDIO = '24'

WORKFLOW_FILENAME = 'minimax_h3_i2v.json'

N_UNET = '6'            # UNETLoader — the base, swapped by the 10Eros option

N_VAE_VIDEO = '11'

N_LOAD_IMAGE = '114'    # LoadImage — dropped in t2v

N_SCALE = '119'         # ImageScaleToTotalPixels — carries the megapixel dial

N_SIZE = '120'          # GetImageSize — dropped in t2v

N_COND = '104'          # MiniMaxH3ImageToVideo — prompt, length, first_frame

N_GUIDER = '16'

N_NOISE = '15'          # RandomNoise — the seed

N_SAMPLER_SELECT = '17'

N_SCHEDULER = '9'       # BasicScheduler — steps, and a second model reader

N_SAMPLER = '14'

N_DECODE_VIDEO = '10'

N_DECODE_AUDIO = '23'

N_CREATE_VIDEO = '91'   # CreateVideo — fps

N_SAVE = '92'           # SaveVideo — filename_prefix

N_SAGE = '600'          # PathchSageAttentionKJ

N_TURBO_LORA = '601'

N_TURBO_SAMPLER = '603'

N_ACCEL_LORA = '602'    # a stock-loader acceleration LoRA (Parasyte, DARE-TIES)

N_SHIFT = '604'         # MiniMaxH3SigmaShift — the shift those LoRAs were tuned at

N_VDN = '605'           # ApplyVDNH3 — the hybrid-attention stage, on the head of the chain

N_TEST_LORA = '610'     # the LoRA under test — 606-609 are the style LoRAs upstream

N_LOAD_END = '620'      # LoadImage for the LAST frame — grafted only when one is staged

N_BLOCK_ATTN = '630'    # LDSMiniMaxH3BlockAttentionSplit — the attention backend, per block

N_UPSCALE = '800'

N_UPSCALE_PARAMS = '801'

N_UPSCALE_SPLIT = '802'

N_SPARSE = '810'

N_REF_SPARSE_MEMORY = '811'

BASE_OFFICIAL = 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'

BASE_EROS = ('10Eros_Max_h3_fl2va_beta2_pruned_int8_convrot_skip_edges'
             '.safetensors')

BASE_LIGHT = 'minimax_h3_fl2va_pruned-w4a8_convrot_pruned.safetensors'

LIGHT_MIN_COMFY = (0, 31, 0)

LIGHT_MIN_COMFY_LABEL = '0.31.0'

TURBO_LORA = 'minimax_h3_turbo_v4_step600_ema.safetensors'

TURBO_STEPS = 6

VIDEO_VAE_INT8 = 'minimax_h3_video_vae_int8_convrot.safetensors'

BLOCK_ATTN_CLASS = 'LDSMiniMaxH3BlockAttentionSplit'

BLOCK_ATTN_INSTALL_ACTION = 'h3_attention_nodes'

ATTN_PYTORCH = 'pytorch attention'

ATTN_KITCHEN = 'comfy kitchen attention'

ATTN_SAGE = 'sage attention'

ATTN_FAST_BACKENDS = (ATTN_SAGE, ATTN_KITCHEN)

ATTN_EDGE_PCT = 20.0    # first/last blocks kept on pytorch: jacokon's own fallback advice

PARASYTE_LORA = 'H3-PK-Parasyte-Turbo.safetensors'

DARETIES_LORA = 'minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro.safetensors'

VDN_STAGE = 'stage-dmd-step-250'               # the 8-step release directory

VDN_FOLDER = 'vdn'                             # ComfyUI/models/vdn — the folder the pack registers

VDN_BRANCH_FILE = 'linear_branch/model.safetensors'

VDN_BRANCH_FILE_INT8 = 'linear_branch/model_int8_convrot_comfyui.safetensors'

VDN_STAGE_REQUIRED = ('model_spec.json',
                      'adapters/default/adapter_config.json',
                      'adapters/default/adapter_model.safetensors',
                      'adapters/turbo/adapter_config.json',
                      'adapters/turbo/adapter_model.safetensors')

VDN_STEPS = 8

VDN_CLASS = 'ApplyVDNH3'

VDN_NODE_INPUTS = ('vdn_checkpoint', 'apply_turbo_adapter', 'strength', 'lora_mode',
                   'branch_weights', 'retain_buffers', 'attention_backend', 'verbose')

VDN_WINDOW_LATENT_FRAMES = 10

VDN_MODES = ('merge', 'bypass')
VDN_LIGHT_ADVICE_VRAM_GB = 28.0


ACCELERATIONS = (
    {'id': 'turbo', 'label': 'larryvrh Turbo v4', 'file': TURBO_LORA, 'steps': TURBO_STEPS,
     'arena': '#1 · I2V 1103 / T2V 1110', 'author': 'larryvrh', 'license': 'apache-2.0',
     'action': 'h3_turbo_lora', 'pack': 'turbo', 'strength': 1.0, 'shift': None,
     'hint': 'A distillation LoRA with its own sampler — a different model, not a faster one.'},
    {'id': 'parasyte', 'label': 'Parasyte Turbo', 'file': PARASYTE_LORA, 'steps': TURBO_STEPS,
     'arena': '#2 · I2V 1106 / T2V 1094', 'author': 'Plaguekind', 'license': 'MIT',
     'action': 'h3_parasyte_lora', 'pack': None, 'strength': 4.0, 'shift': (8.0, 3.0),
     'hint': "Plaguekind's LoRA on the stock sampler: strength 4, shift 8/3, euler. MIT."},
    {'id': 'dareties', 'label': 'DARE-TIES merge', 'file': DARETIES_LORA, 'steps': TURBO_STEPS,
     'arena': '#3 · I2V 1107 / T2V 1085', 'author': 'silveroxides', 'license': 'not stated',
     'action': 'h3_dareties_lora', 'pack': None, 'strength': 0.8, 'shift': (8.0, 3.0),
     'hint': 'LightX2V v0.1 and larryvrh v4 merged (DARE-TIES): strength 0.8, shift 8/3, euler. '
             'Its author states no license.'},
    # Not an arena row: it postdates the arena, and it is not a LoRA on the
    # same sampler but a different attention. `arena` is empty on purpose —
    # the panel prints a rank only when there is one.
    {'id': 'vdn', 'label': 'VDN-H3 hybrid attention', 'file': VDN_STAGE, 'steps': VDN_STEPS,
     'arena': '', 'author': 'OpenVDN', 'license': 'apache-2.0 (port) · MiniMax H3 community (weights)',
     'action': 'h3_vdn_stage', 'pack': 'vdn', 'strength': 1.0, 'shift': None,
     'hint': "OpenVDN's linear-attention branch over the base: 8 steps on the base's own "
             'shift, near-lossless against dense H3 by its authors, and its cost grows with '
             'clip length instead of its square. About twice the turbo time per clip on '
             'one card — compare quality and long clips.'},
)

VDN_LIGHT_ADVICE = ('On a card under 28 GB tick the Lighter base (W4A8) with it: the 21 GB '
                    'base leaves no room and the render pages for minutes.')

CARD_SETUP_PATH = 'Setup › 🎬 Video Test Studio'

VDN_SETUP_PATH = 'Setup › Install or repair individually › Video acceleration: VDN-H3'

ACCEL_IDS = tuple(a['id'] for a in ACCELERATIONS)

def accel_spec(accel) -> dict:
    for a in ACCELERATIONS:
        if a['id'] == accel:
            return a
    raise ValueError(f'unknown acceleration: {accel!r}')

def normalise_accel(accel, turbo=False) -> str:
    """The acceleration in force: the one named, `turbo` for a caller that still
    speaks the boolean, '' for the dense base. Anything else is refused — a
    string that reaches a loader must be one this module wrote."""
    key = str(accel or '').strip().lower()
    if key in ACCEL_IDS:
        return key
    if key in ('', 'none', 'off', 'false', '0'):
        return 'turbo' if turbo else ''
    raise ValueError(f'unknown acceleration: {accel!r}')

def vdn_latent_frames(frames) -> int:
    """ComfyUI's own count (comfy_extras.nodes_minimax_h3.video_latent_t) for a
    snapped clip length: 22 → 7, 39 → 12, 56 → 17."""
    n = int(frames)
    return 2 if n <= 5 else ((n - 5) // 17) * 5 + 2

def vdn_branch_active(frames) -> bool:
    """Does VDN's linear branch carry anything at this length? Below
    VDN_WINDOW_LATENT_FRAMES the trained window already covers the whole
    clip and the node renders dense attention — plain H3, at branch cost."""
    return vdn_latent_frames(frames) > VDN_WINDOW_LATENT_FRAMES



def comfyui_vram_gb(timeout=3):
    """The first CUDA device's total VRAM, in GB, from /system_stats — None when
    ComfyUI is not configured, silent, or reports no device. Never raises."""
    from lds_sdk import config as cfg
    import requests
    api = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api:
        return None
    try:
        r = requests.get(f'{api}/system_stats', timeout=timeout, allow_redirects=False)
        if r.status_code != 200:
            return None
        devices = (r.json() or {}).get('devices') or []
        total = (devices[0] or {}).get('vram_total') if devices else None
    except Exception:
        return None
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
        return None
    # Binary gigabytes, the unit the card is sold in: a 4090's 25 756 696 576
    # bytes read "24 GB", as the advice and the threshold say.
    return total / 1024 ** 3

def normalise_mode(mode):
    mode = str(mode or 'i2v').strip().lower()
    if mode not in ('i2v', 't2v', 'ref2va'):
        raise ValueError('Choose image-to-video, text-to-video or references.')
    return mode

def reference_accel_spec(accel):
    key = str(accel or '').strip().lower()
    if key in ('', 'none', 'off', 'false', '0'):
        return None
    for spec in vrc.REF_ACCELERATIONS:
        if key == spec['id']:
            return spec
    raise ValueError('Reference mode needs Ref4, Ref8, VDN-H3, or the dense base.')

DEFAULT_STEPS = 20

SPARSE_PRESETS = {
    'default': {'video_budget': 0.3, 'early_steps': 2, 'early_kv': 0.5,
                'late_steps': 2, 'late_kv': 0.5},
    'conservative': {'video_budget': 0.5, 'early_steps': 2, 'early_kv': 0.8,
                     'late_steps': 2, 'late_kv': 0.8},
    'max': {'video_budget': 0.3, 'early_steps': 2, 'early_kv': 0.5,
            'late_steps': 2, 'late_kv': 0.5},
}

SPARSE_MODES = ('', 'default', 'conservative', 'max')

SPARSE_BACKEND = 'Kitchen INT8'

UPSCALE_MODEL = 'minimax_h3_latent_upscaler_3d_bf16.safetensors'

TARGET_KEY = 'minimax_h3'

def _profile() -> dict:
    """The shared catalogue entry for H3, or {} on an unknown key.

    Everything the studio takes from the catalogue goes through here: fps, the
    default clip length, the legal-length rule. Restating any of them locally is
    how a clip generated in the studio and a clip cut for training start
    disagreeing about what H3 accepts.
    """
    return video_targets.get(TARGET_KEY) or {}

_FRAME_MOD, _FRAME_OFFSET = 17, 5

FRAMES_MIN, FRAMES_MAX = 22, 362

FRAMES_DEFAULT = 56

MP_MIN, MP_MAX, MP_DEFAULT = 0.1, 2.0, 0.3

LORA_SUBDIR = os.path.join('h3', 'lds')

def workflow_path():
    """Absolute path of the embedded H3 graph."""
    return str(Path(__file__).resolve().parent / 'workflows' / WORKFLOW_FILENAME)

def load_base_workflow() -> dict:
    """The embedded graph, freshly parsed.

    Read from disk on every build rather than cached at import: a build MUTATES
    the graph it is handed, and a shared dict would carry one run's turbo nodes
    into the next run that asked for none.
    """
    with open(workflow_path(), 'r', encoding='utf-8') as fh:
        return json.load(fh)

def snap_frames(requested) -> int:
    """Round a requested frame count to the nearest length H3's VAE accepts.

    Nothing downstream objects to an illegal count — the VAE floors it in latent
    space and no exception is ever raised — so this is the only place the rule
    is enforced. `video_targets.is_legal_frames` is the authority on what legal
    means; this function only has to land on it.
    """
    try:
        want = int(requested)
    except (TypeError, ValueError):
        want = FRAMES_DEFAULT
    snapped = round((want - _FRAME_OFFSET) / _FRAME_MOD) * _FRAME_MOD + _FRAME_OFFSET
    return max(FRAMES_MIN, min(FRAMES_MAX, int(snapped)))

def clamp_megapixels(value) -> float:
    try:
        mp = float(value)
    except (TypeError, ValueError):
        mp = MP_DEFAULT
    return max(MP_MIN, min(MP_MAX, mp))

def normalise_sparse(mode) -> str:
    """'' for off, or one of the author's levels. An unknown string is OFF, not
    a guess: a typo that silently selected a sampling mode would be invisible in
    the output and blamed on the LoRA."""
    m = str(mode or '').strip().lower()
    return m if m in SPARSE_PRESETS else ''

def build_workflow(*, prompt, mode='i2v', image=None, end_image=None, seed=None, steps=None,
                   frames=None, megapixels=MP_DEFAULT, aspect='auto',
                   fps=None, lora=None, lora_strength=1.0, turbo=False,
                   accel=None, eros=False, eros_on_disk=False, sparse='',
                   light=False, light_on_disk=False, light_note='',
                   latent_upscale=False, source_ratio=None, sage=True,
                   attention=None, video_vae=None, vdn_mode=None,
                   references=None, ref_base='official', ref_image_size='match',
                   filename_prefix='lds_video_test') -> dict:
    """One MiniMax H3 clip, as a ComfyUI graph.

    `accel` names one of ACCELERATIONS ('' or None for the dense base); `turbo`
    is the older boolean and means `accel='turbo'` when `accel` is not given.

    `eros_on_disk` is passed IN rather than probed here so the whole option
    matrix stays testable without a 21 GB file — and so the fail-open path (ask
    for 10Eros, get the official base plus a note) is exercised by a test rather
    than by a user whose download had not finished.

    `light` / `light_on_disk` are the same pair for the 🪶 W4A8 base, with the
    same fail-open; `light_note` is the caller's reason when the base is not
    usable (a server older than the format), so the fail-open line says why.
    A clip has ONE base, and precedence goes to what is actually THERE:
    10Eros when its file is, else W4A8 when its file is, else the official
    base — each with a note naming what was asked for and not applied,
    rather than one flag silently shadowing the other.

    `end_image` is the staged picture the clip ENDS on. H3 is a first-last-
    to-video model: `MiniMaxH3ImageToVideo` takes `last_frame` as a keyframe at
    the clip's last index and resizes it to the canvas itself, so the graft is
    one LoadImage wired into that input — in t2v too, where the clip then
    resolves from the prompt onto the picture.

    `source_ratio` (width/height of the picked image) only matters when the
    latent upscale is armed and there is an image to measure; without it the
    upscale keeps the node's own target size.

    `sage=False` takes SageAttention out. It is a speed patch that needs the
    `sageattention` GPU package, which LDS never installs, so the graph has to be
    able to do without it — the caller passes `sage_available()`, which keeps it
    only when the target ComfyUI registers KJNodes' patch AND imports the package.

    `attention` — the per-block backend switch (`attention_spec`): the node is
    grafted right after the base, before every LoRA, with the middle stack on
    a fast backend and the ends on pytorch. None = no graft, which is what a
    target that does not list the class gets. `video_vae` — a VAE file name
    to put in place of the graph's fp16 one (the int8 repack of the real-time
    preset); None keeps the template's. `vdn_mode` — how VDN-H3 applies its
    adapters: None is `merge`, the mode the port requires for the 8-step
    stage (VDN_MODES); `bypass` exists for an A/B, never as a launch default.
    """
    wf = load_base_workflow()
    notes = []
    mode = normalise_mode(mode)
    reference_ratio = None
    if mode == 'ref2va':
        from lds_sdk import h3_reference_graph as refs
        ref_spec = next((a for a in vrc.REF_BASES if a['id'] == ref_base), None)
        if ref_spec is None:
            raise ValueError('Unknown reference base model.')
        if not isinstance(references, list) or not references:
            raise ValueError('Add at least one reference.')
        counts = dict.fromkeys(refs.LIMITS, 0)
        for ref in references:
            if (not isinstance(ref, dict) or not isinstance(ref.get('kind'), str)
                    or ref['kind'] not in refs.LIMITS):
                raise ValueError('Invalid reference kind.')
            refs._safe_name(ref.get('name'))
            counts[ref['kind']] += 1
        if any(counts[k] > refs.LIMITS[k] for k in counts):
            raise ValueError('Too many references.')
        reference_ratio = refs.reference_format_ratio(references)
        if turbo or eros or light:
            raise ValueError('Choose the base and acceleration from the reference mode controls.')
        ref_accel = reference_accel_spec(accel)

    frames = snap_frames(FRAMES_DEFAULT if frames is None else frames)
    megapixels = clamp_megapixels(megapixels)
    sparse = normalise_sparse(sparse)

    wf[N_COND]['inputs']['prompt'] = str(prompt or '')
    wf[N_COND]['inputs']['length'] = frames
    wf[N_SCALE]['inputs']['megapixels'] = megapixels
    wf[N_CREATE_VIDEO]['inputs']['fps'] = float(
        fps or _profile().get('fps') or 24.0)
    wf[N_SAVE]['inputs']['filename_prefix'] = filename_prefix

    # Seed. The template ships `noise_seed: 42` hard-coded, and a graph that
    # never writes node 15 renders the SAME clip from the same prompt every
    # time — which reads as "the model is stuck", not as "the seed never moved".
    if seed is None or (isinstance(seed, int) and seed < 0):
        seed = random.randint(0, 999_999_999_999_999)
    wf[N_NOISE]['inputs']['noise_seed'] = int(seed)

    # Before any graft: everything below reads "the chain", and Sage is part of
    # it or is not.
    if not sage:
        _drop_sage(wf)
        notes.append('sage: not usable on this ComfyUI (KJNodes or the sageattention package missing) '
                     '— running without it')

    if video_vae:
        wf[N_VAE_VIDEO]['inputs']['vae_name'] = str(video_vae)
        notes.append(f'video vae: {video_vae}')

    if mode in ('t2v', 'ref2va'):
        # ⏭ A reference CONTINUATION renders at its parent's shape: the route
        # hands the seam picture's ratio, and it wins over the shape dial and
        # over a format reference. A part that came back another shape is
        # damaged twice — the H3 guide centre-crops the seam into the new
        # canvas before the render, then the join scales the result into the
        # parent's frame: a 16:9 parent continued in portrait keeps a third of
        # its last frame and comes back stretched (verification, 2026-09-07).
        canvas_ratio = source_ratio if (mode == 'ref2va' and source_ratio) else reference_ratio
        _make_t2v(wf, megapixels=megapixels, aspect=aspect, ratio=canvas_ratio)
        notes.append('t2v: image branch removed')
    else:
        wf[N_LOAD_IMAGE]['inputs']['image'] = str(image or '')
    if mode == 'ref2va':
        refs.graft(wf, references=references, image=image, end_image=end_image,
                   ref_image_size=ref_image_size)
        wf[N_UNET]['inputs']['unet_name'] = ref_spec['file']
        notes.append(f"references: {len(references)}, base: {ref_spec['label']}")
    elif end_image:
        wf[N_LOAD_END] = {'class_type': 'LoadImage', 'inputs': {'image': str(end_image)}}
        wf[N_COND]['inputs']['last_frame'] = [N_LOAD_END, 0]
        notes.append('end frame: the clip ends on the staged picture')

    # ── Order of the grafts is the contract ──────────────────────────────────
    # base swap → turbo → tested LoRA → sparse → upscale.
    #
    # The base swap goes first so every log line below can name the base that
    # actually ran. The sparse graft goes before the upscale because the upscale
    # reads whatever the model chain currently ends in — reversing them gives
    # the upscale a model that has not been patched yet, with no error anywhere.
    # One clip, one base — and precedence goes to what is actually THERE, not
    # to the flag: asking for 10Eros while its file is still downloading must
    # not throw away a W4A8 that IS on disk, and no note may claim a base
    # that did not run.
    if eros and eros_on_disk:
        wf[N_UNET]['inputs']['unet_name'] = BASE_EROS
        notes.append('base: 10Eros-Max (third-party finetune)')
        if light:
            notes.append('base: W4A8 not applied — 10Eros chosen')
    elif light and light_on_disk:
        wf[N_UNET]['inputs']['unet_name'] = BASE_LIGHT
        notes.append(f'base: W4A8 ConvRot, 12.5 GB (needs ComfyUI {LIGHT_MIN_COMFY_LABEL}+)')
        if eros:
            notes.append('base: 10Eros requested but absent — W4A8 used')
    else:
        # Fail OPEN to the official base. A box can legitimately be ticked
        # while the file is still downloading, and a graph ComfyUI refuses at
        # validation ("Value not in list: unet_name") is a worse answer than a
        # clip on the official base plus a line saying so — and, for W4A8,
        # saying WHY when the caller knows (a server older than the format).
        if eros:
            notes.append('base: 10Eros requested but absent — official base used')
        if light:
            notes.append(f"base: W4A8 requested but {light_note or 'absent'} — official base used")

    # The attention switch goes right after the base and before every LoRA:
    # it patches the blocks' attention calls, and a LoRA patches weights —
    # the two compose, but the graph reads clearer when the model's own
    # mechanics come first and the adapters after.
    if attention:
        _graft_block_attention(wf, attention)
        notes.append(f"attention: {attention.get('middle')} (edges {attention.get('edge', ATTN_PYTORCH)}, "
                     f"{attention.get('head_pct', ATTN_EDGE_PCT):g}/{attention.get('tail_pct', ATTN_EDGE_PCT):g} %)")

    accel = (ref_accel['id'] if ref_accel else '') if mode == 'ref2va' else normalise_accel(accel, turbo)
    if accel == 'vdn' and sparse:
        # Both own `blocks.*.attn` — the sparse pack abandons its path when
        # another patch is there, and VDN's own README says the same about
        # attention overrides on ITS path. Refused, not silently dropped: a
        # clip that says "sparse: max" while rendering dense is the kind of
        # note this lane has already paid for once.
        raise ValueError('VDN-H3 and sparse attention patch the same attention path — '
                         'pick one of the two.')
    if mode == 'ref2va' and ref_accel and accel != 'vdn':
        _graft_reference_accel(wf, ref_accel)
    elif accel == 'turbo':
        _graft_turbo(wf)
    elif accel == 'vdn':
        vdn_mode = str(vdn_mode or 'merge').strip().lower()
        if vdn_mode not in VDN_MODES:
            raise ValueError(f'unknown VDN-H3 adapter mode: {vdn_mode!r}')
        _graft_vdn(wf, vdn_mode)
    elif accel and mode != 'ref2va':
        _graft_stock_accel(wf, accel_spec(accel))
    if steps is not None:
        # An explicit step count always wins, including over the turbo default:
        # the panel showing 6 while the graph runs 4 is the exact failure this
        # pipeline already paid for once.
        wf[N_SCHEDULER]['inputs']['steps'] = max(4, min(40, int(steps)))
    # The note reads the count AFTER the override: a log that said "steps=6"
    # while the graph ran the user's 4 was seen on the first real launch.
    if mode == 'ref2va' and ref_accel and accel != 'vdn':
        notes.append(f"accel: {ref_accel['label']}, Euler, shift 12/3")
    elif accel == 'turbo':
        notes.append(f'turbo: larryvrh distillation, steps={wf[N_SCHEDULER]["inputs"]["steps"]}')
    elif accel == 'vdn':
        notes.append(f"accel: VDN-H3 {VDN_STAGE}, adapters {wf[N_VDN]['inputs']['lora_mode']}, "
                     f"euler, base shift, steps={wf[N_SCHEDULER]['inputs']['steps']}")
        if mode == 'ref2va':
            notes.append('VDN window coverage depends on reference tokens as well as generated frames; '
                         'inspect the VDN runtime layout for the active attention path')
        elif not vdn_branch_active(frames):
            notes.append(f'VDN window covers all {vdn_latent_frames(frames)} latent frames at '
                         f'{frames} frames: the node renders dense attention (no branch benefit)')
        if N_SAGE in wf or attention:
            # The windows run exact SDPA on purpose (the port's README):
            # attention patches only reach the token refiner and the dense
            # fallback. Said here so the "attention: sage" line above cannot
            # read as the backend the clip was sampled with.
            notes.append('attention patches (Sage, block switch) reach the token refiner and '
                         "the dense fallback only — VDN's windows run exact SDPA")
    elif accel:
        spec = accel_spec(accel)
        notes.append(f"accel: {spec['label']} @ {spec['strength']:g}, shift "
                     f"{spec['shift'][0]:g}/{spec['shift'][1]:g}, euler, "
                     f"steps={wf[N_SCHEDULER]['inputs']['steps']}")

    if lora:
        # The applied force comes BACK from the graft rather than being
        # re-derived for the log: the graft coerces and clamps, and a note that
        # re-read the raw request would announce a strength the graph does not
        # carry (and crash outright on a non-numeric one).
        applied = _graft_test_lora(wf, lora, lora_strength)
        notes.append(f'lora: {lora} @ {applied:g}')

    if sparse:
        _graft_sparse(wf, sparse, upscale_armed=bool(latent_upscale))
        notes.append(f'sparse: {sparse}')

    if latent_upscale:
        _graft_latent_upscale(wf, megapixels=megapixels, source_ratio=source_ratio)
        notes.append('latent upscale x2')

    effective_mode = mode
    effective_aspect = str(aspect or '').strip().lower()
    if effective_aspect not in ('portrait', 'landscape', 'square'):
        effective_aspect = 'landscape'
    settings = {
        'mode': effective_mode, 'aspect': effective_aspect if effective_mode != 'i2v' else 'auto',
        'seed': int(seed), 'frames': frames, 'megapixels': megapixels,
        'base_model': wf[N_UNET]['inputs']['unet_name'],
        'steps': wf[N_SCHEDULER]['inputs']['steps'],
        'lora': lora, 'lora_strength': applied if lora else None,
        'accel': accel, 'sparse': sparse, 'latent_upscale': bool(latent_upscale),
    }
    if mode == 'ref2va':
        settings.update(ref_base=ref_base, ref_image_size=ref_image_size, references=references,
                        canvas_width=wf[N_COND]['inputs']['width'],
                        canvas_height=wf[N_COND]['inputs']['height'])
    return {'workflow': wf, 'seed': int(seed), 'frames': frames,
            'megapixels': megapixels, 'notes': notes,
            'base': wf[N_UNET]['inputs']['unet_name'],
            'steps': wf[N_SCHEDULER]['inputs']['steps'],
            'accel': accel, 'generation_settings': settings}

def _graft_reference_accel(wf, spec):
    _insert_into_chain(wf, N_ACCEL_LORA, {
        'class_type': vrc.REF_LORA_CLASS,
        'inputs': {'model': [N_UNET, 0], 'lora_name': spec['file'],
                   'strength': float(spec['strength'])},
    }, [N_UNET, 0])
    _insert_into_chain(wf, N_SHIFT, {
        'class_type': 'MiniMaxH3SigmaShift',
        'inputs': {'model': [N_ACCEL_LORA, 0], 'shift_video': 12.0, 'shift_audio': 3.0},
    }, [N_ACCEL_LORA, 0])
    wf[N_SAMPLER_SELECT]['inputs']['sampler_name'] = 'euler'
    wf[N_SCHEDULER]['inputs'].update(scheduler='simple', steps=spec['steps'])

def _model_readers(wf, ref, *, skip=()):
    """Every node whose `model` input reads `ref`."""
    return [nid for nid, node in wf.items()
            if nid not in skip and (node.get('inputs') or {}).get('model') == ref]

def _insert_into_chain(wf, node_id, node, head):
    """Put a patch node between `head` and everyone who was reading `head`.

    Written once, because getting it wrong is silent in a specific way: the
    guider path and the BasicScheduler both read the model, and patching only
    the one you were thinking about leaves the sigmas computed from an unpatched
    model while the sampling runs on the patched one. The graph is valid, the job
    is green, and the clip is mush.

    Naming the readers instead of listing them by id is also what lets Sage be
    optional — with or without it in the graph, "everyone downstream" is the same
    sentence.
    """
    readers = _model_readers(wf, head, skip=(node_id,))
    wf[node_id] = node
    for nid in readers:
        wf[nid]['inputs']['model'] = [node_id, 0]

def _drop_sage(wf):
    """Take SageAttention out of the chain, and out of the graph.

    Two callers, one behaviour. It goes when the target ComfyUI does not have
    the node at all (KJNodes is a pack, and one that pulls pip dependencies —
    the base graph must not need it), and it goes when sparse attention is armed
    (H3-Optimizations >= 0.2.16 refuses to compose with an attention override it
    does not own: it abandons the sparse path, keeps Sage, and logs a warning —
    no error, no red job, the mode simply renders dense).

    Removed rather than left unconsumed: an unread node costs nothing to EXECUTE,
    but a node whose class this install does not register is a validation risk
    for no benefit. Its readers move onto its own upstream, so nothing else in
    the chain notices.
    """
    if N_SAGE not in wf:
        return
    upstream = (wf[N_SAGE].get('inputs') or {}).get('model')
    if upstream is not None:
        for nid in _model_readers(wf, [N_SAGE, 0], skip=(N_SAGE,)):
            wf[nid]['inputs']['model'] = upstream
    wf.pop(N_SAGE, None)

def _make_t2v(wf, *, megapixels, aspect, ratio=None):
    """Text-to-video: unplug the image branch.

    `first_frame` is OPTIONAL on `MiniMaxH3ImageToVideo`, so t2v is the same
    node with one input removed — but the graph then has nothing to measure the
    canvas from, which is why width/height become explicit here.
    """
    for nid in (N_LOAD_IMAGE, N_SCALE, N_SIZE):
        wf.pop(nid, None)
    wf[N_COND]['inputs'].pop('first_frame', None)
    from_reference = ratio is not None
    if ratio is None:
        ratio = {'landscape': 16 / 9, 'portrait': 9 / 16,
                 'square': 1.0}.get(str(aspect or '').strip().lower(), 16 / 9)
    height = int(round((megapixels * 1_000_000 / ratio) ** 0.5 / 32) * 32)
    width = int(round(height * ratio / 32) * 32)
    if from_reference and max(width, height) > 16384:
        raise ValueError('The reference video format exceeds the model canvas limit at this '
                         'resolution; lower megapixels or turn off matching its format.')
    wf[N_COND]['inputs']['width'] = max(32, width)
    wf[N_COND]['inputs']['height'] = max(32, height)

def attention_spec(backends, middle=None, edge_pct=ATTN_EDGE_PCT):
    """The graft for a ComfyUI that lists these backends on the switch, or
    None when nothing faster than pytorch is there (a graft that only asks for
    pytorch would cost a node for nothing).

    `middle` names a backend to insist on; absent or not listed, the fastest
    listed one is taken, in ATTN_FAST_BACKENDS order."""
    listed = [str(b) for b in (backends or [])]
    chosen = middle if middle in listed else next((b for b in ATTN_FAST_BACKENDS if b in listed), None)
    if not chosen or chosen == ATTN_PYTORCH:
        return None
    return {'middle': chosen, 'edge': ATTN_PYTORCH,
            'head_pct': float(edge_pct), 'tail_pct': float(edge_pct)}

def block_attention_backends(node_info) -> list:
    """The backend names the switch offers on a given ComfyUI, read out of
    the node's own /object_info entry (its `middle_backend` combo). [] when
    the node is not registered there — never a guess."""
    try:
        combo = ((node_info or {}).get('input') or {}).get('required', {}).get('middle_backend')
        options = combo[0] if isinstance(combo, (list, tuple)) and combo else []
        return [str(o) for o in options if isinstance(o, str)]
    except (AttributeError, TypeError, IndexError):
        return []

def local_attention_backends() -> list:
    """What THIS machine's ComfyUI offers on the switch — [] when the pack is
    not installed (Setup has the button) or ComfyUI is not reachable."""
    from lds_sdk.video_host.comfyui import fetch_node_info
    return block_attention_backends(fetch_node_info(BLOCK_ATTN_CLASS))

def block_attention_missing_nodes():
    """[class] when the target ComfyUI does not register the switch; [] when it
    does OR when /object_info is unreachable (fail open: a probe failure is not
    a verdict). Same shape as the Krea sampler's helper."""
    from lds_sdk.video_host.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    return [] if BLOCK_ATTN_CLASS in available else [BLOCK_ATTN_CLASS]

def block_attention_pack_installed() -> bool:
    """Is a CURRENT copy of the shipped folder in ComfyUI's custom_nodes?
    Disk-only, like the Krea sampler's: ComfyUI registers nodes at startup, so
    a copy that landed a minute ago is "installed, restart ComfyUI", not
    "missing"."""
    from lds_sdk.video_host import setup as setup_installer
    try:
        return setup_installer.bundled_pack_state(BLOCK_ATTN_INSTALL_ACTION) == 'current'
    except Exception:  # noqa: BLE001 — an unreadable folder is "not that we know of"
        return False

def _graft_block_attention(wf, spec):
    """🔴 The per-block attention switch onto the head of the model chain.

    Placed between the UNET and everyone who reads it (the same
    `_insert_into_chain` rule as every graft here: the guider path and the
    scheduler must both see the patched model). Whatever LoRA is grafted
    after this reads the switched model, and so the sampling runs on the
    chosen backend whatever else is mounted.
    """
    _insert_into_chain(wf, N_BLOCK_ATTN, {
        'class_type': BLOCK_ATTN_CLASS,
        'inputs': {'model': [N_UNET, 0],
                   'edge_backend': str(spec.get('edge') or ATTN_PYTORCH),
                   'middle_backend': str(spec.get('middle') or ATTN_PYTORCH),
                   'head_pct': float(spec.get('head_pct', ATTN_EDGE_PCT)),
                   'tail_pct': float(spec.get('tail_pct', ATTN_EDGE_PCT))},
    }, [N_UNET, 0])

def _graft_turbo(wf):
    """⚡ The 4-step distillation LoRA and its double-clock sampler.

    The sampler is not optional decoration. Video and audio are denoised on
    DIFFERENT schedules (shift 12 and shift 3); a single-calendar sampler
    over-samples the audio at four steps and audibly breaks it.

    Both the guider path and the scheduler have to be moved onto the patched
    model. Moving only one leaves the sigmas computed from an unpatched model
    while the sampling runs on the patched one — a graph that runs, and renders
    mush.
    """
    _insert_into_chain(wf, N_TURBO_LORA, {
        'class_type': 'MiniMaxH3TurboLoRA',
        'inputs': {'model': [N_UNET, 0], 'lora_name': TURBO_LORA,
                   'strength': 1.0, 'low_vram': False},
    }, [N_UNET, 0])
    wf[N_TURBO_SAMPLER] = {'class_type': 'MiniMaxH3TurboSampler', 'inputs': {}}
    wf[N_SAMPLER]['inputs']['sampler'] = [N_TURBO_SAMPLER, 0]
    wf[N_SCHEDULER]['inputs']['steps'] = TURBO_STEPS

def _graft_stock_accel(wf, spec):
    """⚡ An acceleration LoRA through the STOCK loader, and the sigma shift it
    was tuned at, on the stock euler sampler.

    Both nodes go INTO the chain (the loader, then the shift) so the guider
    path and the scheduler read the same patched model — the rule every graft
    here follows, for the reason _insert_into_chain states. The shift is a
    core ComfyUI node (comfy_extras.nodes_minimax_h3): no pack to install.
    """
    _insert_into_chain(wf, N_ACCEL_LORA, {
        'class_type': 'LoraLoaderModelOnly',
        'inputs': {'model': [N_UNET, 0], 'lora_name': spec['file'],
                   'strength_model': float(spec['strength'])},
    }, [N_UNET, 0])
    video, audio = spec['shift']
    _insert_into_chain(wf, N_SHIFT, {
        'class_type': 'MiniMaxH3SigmaShift',
        'inputs': {'model': [N_ACCEL_LORA, 0], 'shift_video': float(video),
                   'shift_audio': float(audio)},
    }, [N_ACCEL_LORA, 0])
    wf[N_SAMPLER_SELECT]['inputs']['sampler_name'] = 'euler'
    wf[N_SCHEDULER]['inputs']['scheduler'] = 'simple'
    wf[N_SCHEDULER]['inputs']['steps'] = int(spec['steps'])

def _graft_vdn(wf, lora_mode='merge'):
    """⚡ VDN-H3 on the head of the model chain.

    One node, between the base and everyone who reads it (the guider path and
    the scheduler — the same both-readers rule as every graft here). The node
    patches each block's attention and applies the two adapters (`lora_mode`,
    see VDN_MODES: bypass keeps the adaln half on the pruned bases this lane
    offers, merge is the port's default), so it sits UNDER the LoRA under test,
    which then patches on top — the composition the pack's README describes.

    The values are the released model's, not ours: turbo adapter on, strength
    1.0, the trained window (nothing in the Advanced node's ablations), the
    grouped exact-SDPA windows, and `auto` for what the node decides from the
    VRAM it finds after the base is loaded (branch resident or streamed per
    block, scratch retained or transient). No sigma shift node: the official
    inference config samples on the base's own 12/3 grid, which is what the
    template already carries. Eight steps, the schedule the turbo adapter was
    distilled for — the node's own warning is that mixing the two schedules
    degrades output.
    """
    _insert_into_chain(wf, N_VDN, {
        'class_type': VDN_CLASS,
        'inputs': {'model': [N_UNET, 0], 'vdn_checkpoint': VDN_STAGE,
                   'apply_turbo_adapter': True, 'strength': 1.0, 'lora_mode': str(lora_mode),
                   'branch_weights': 'auto', 'retain_buffers': 'auto',
                   'attention_backend': 'grouped', 'verbose': False},
    }, [N_UNET, 0])
    wf[N_SAMPLER_SELECT]['inputs']['sampler_name'] = 'euler'
    wf[N_SCHEDULER]['inputs']['scheduler'] = 'simple'
    wf[N_SCHEDULER]['inputs']['steps'] = VDN_STEPS

def _graft_test_lora(wf, lora_name, strength):
    """The LoRA under test, chained onto the head of the model chain.

    Loaded through the STANDARD `LoraLoaderModelOnly`: an ai-toolkit LoRA ships
    `diffusion_model.`-prefixed keys, which is exactly what that loader expects
    and exactly what the turbo node would break — that node re-prefixes, giving
    `diffusion_model.diffusion_model.blocks…`, which matches nothing and is
    dropped key by key WITHOUT a word. Measured on this family: 0/208 modules
    through the dedicated node, 208/208 through the standard one.

    Head of chain is the turbo node when it is mounted, the UNET otherwise, and
    every model reader downstream of Sage is moved onto the result — the same
    both-readers rule as the turbo graft, for the same reason.

    Returns the strength that was actually applied, after coercion and clamping.
    """
    if N_TURBO_LORA in wf:
        head = [N_TURBO_LORA, 0]
    elif N_ACCEL_LORA in wf:
        head = [N_ACCEL_LORA, 0]
    elif N_VDN in wf:
        head = [N_VDN, 0]
    else:
        head = [N_UNET, 0]
    try:
        force = float(strength)
    except (TypeError, ValueError):
        force = 1.0
    # Beyond ±2 a rank 8-16 LoRA destroys the shot before it expresses anything.
    force = max(-2.0, min(2.0, force))
    _insert_into_chain(wf, N_TEST_LORA, {
        'class_type': 'LoraLoaderModelOnly',
        'inputs': {'model': head, 'lora_name': lora_name,
                   'strength_model': force},
    }, head)
    return force

def _graft_sparse(wf, mode, *, upscale_armed):
    """⚡ Sparse attention, and the two things that make it not free.

    FIRST — SAGE HAS TO LEAVE THE CHAIN. H3-Optimizations ≥ 0.2.16 refuses to
    compose sparse attention with an `optimized_attention_override` it does not
    own: if the model reaching this node has already been patched by Sage (600),
    the pack ABANDONS the sparse path, keeps Sage, and logs a warning. No error,
    no red job — just the ⚡ mode quietly rendering dense. Naming an explicit
    backend does not protect: the guard runs before the backend is consulted.

    We excise Sage instead of hard-wiring a target for the sparse node, because
    what sits around 600 varies — LoRAs upstream, and other patches downstream.
    Rewiring 600's consumers onto 600's own upstream removes Sage alone and
    preserves everything else. The node stays in the graph and is simply not
    consumed; ComfyUI does not execute a node nobody reads.

    SECOND — WHERE THE SPARSITY LANDS. With turbo there are only a handful of
    sampling steps, so the node's "dense edges" already cover the whole schedule
    and the sparsity bites the very steps that decide the composition — which
    shows up as a prompt that stopped being respected. The base pass is also the
    cheap one; the time lives in the upscale. So:

      * upscale armed → the base stays DENSE (the guider does not read the
        sparse node) and only the upscale samples sparse;
      * no upscale → sparse applies to the base, because it is the only pass
        there is, and the adherence cost is the price of the mode;
      * `max` → sparse on BOTH passes, adherence drift accepted by whoever
        ticked it. It is never the default.
    """
    settings = SPARSE_PRESETS[mode]
    _drop_sage(wf)
    head = wf[N_GUIDER]['inputs']['model']
    if any(node.get('class_type') == vrc.REF_LORA_CLASS for node in wf.values()):
        # Ref4/Ref8 add their delta through each projection's forward hook.
        # Optimized QKV/MLP weight reads bypass it, so keep those forwards while
        # letting the attention kernel sparsify only the target-video tokens.
        wf[N_REF_SPARSE_MEMORY] = {
            'class_type': 'H3MemoryOptimization',
            'inputs': {'model': head, 'fused_qkv': 'off', 'mlp_memory': 'off',
                       'chunk_rows': 4096, 'preserve_precision': True,
                       'precision_mode': 'Preserve native', 'qkv_streaming_mode': 'Off'},
        }
        head = [N_REF_SPARSE_MEMORY, 0]
    wf[N_SPARSE] = {
        'class_type': 'H3SparseAttentionAdvanced',
        'inputs': {'model': head, 'backend': SPARSE_BACKEND, **settings},
    }
    if not (upscale_armed and mode != 'max'):
        wf[N_GUIDER]['inputs']['model'] = [N_SPARSE, 0]

def _graft_latent_upscale(wf, *, megapixels, source_ratio=None):
    """🔬 Enlarge in latent space, before anything is decoded.

    The model is read OFF THE GUIDER, never hard-coded: by this point it points
    at Sage, at a LoRA, or at the sparse node, depending on what was armed.
    Writing `['600', 0]` here would silently drop the user's LoRA. When the
    sparse node exists it is the head, deliberately — the base was left dense so
    the prompt survives, and the acceleration belongs on the pass that costs the
    minutes.

    NO SPATIAL TILING. Wired once, and the output was not an enlarged clip but a
    MOSAIC: each tile had resampled the entire scene instead of its own portion.
    The node says as much itself — "leave unconnected to sample each chunk whole
    (no tiling)". Tiling exists to fit in VRAM one does not have; the temporal
    split stays, because it carries clip LENGTH, not picture content.
    """
    model = ([N_SPARSE, 0] if N_SPARSE in wf
             else wf[N_GUIDER]['inputs']['model'])
    conditioning = wf[N_GUIDER]['inputs'].get('conditioning', [N_COND, 0])

    target_w, target_h = 1280, 704          # the node's own defaults
    if not source_ratio:
        # Text-only: no picture to measure, but `_make_t2v` has already made
        # the canvas explicit on the conditioning node — read the shape off
        # it. Left to the defaults, a portrait clip was enlarged towards a
        # landscape target (found in verification, 2026-09-04). In i2v the
        # two are links to GetImageSize, not numbers, and nothing changes.
        w = wf[N_COND]['inputs'].get('width')
        h = wf[N_COND]['inputs'].get('height')
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            source_ratio = w / h
    if source_ratio:
        try:
            ratio = float(source_ratio)
            side = (megapixels * 1_000_000 * ratio) ** 0.5
            target_w = max(64, int(side * 2) // 32 * 32)
            target_h = max(64, int(side / max(ratio, 1e-6) * 2) // 32 * 32)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    wf[N_UPSCALE_PARAMS] = {
        'class_type': 'MMH3LatentUpscaleWithModelParams',
        'inputs': {'model_name': UPSCALE_MODEL, 'width': target_w,
                   'height': target_h, 'device': 'cuda', 'precision': 'bf16'},
    }
    wf[N_UPSCALE_SPLIT] = {
        'class_type': 'MMH3TemporalSplitParams',
        'inputs': {'chunk_length': 136, 'temporal_overlap': 17,
                   'anchor_strength': 0.999},
    }
    wf[N_UPSCALE] = {
        'class_type': 'MMH3UltimateUpscale',
        'inputs': {'model': model, 'conditioning': conditioning,
                   'latent': [N_SAMPLER, 0], 'noise': [N_NOISE, 0],
                   'sampler': [N_SAMPLER_SELECT, 0], 'sigmas': [N_SCHEDULER, 0],
                   'cfg': 1.0,
                   'latent_upscale_param': [N_UPSCALE_PARAMS, 0],
                   'temporal_split_param': [N_UPSCALE_SPLIT, 0]},
    }
    # BOTH decoders move onto the enlarged latent. Rewiring one would leave the
    # video and its own audio coming from two different latents.
    wf[N_DECODE_VIDEO]['inputs']['samples'] = [N_UPSCALE, 0]
    wf[N_DECODE_AUDIO]['inputs']['samples'] = [N_UPSCALE, 0]

def new_prefix(user_id=None) -> str:
    """A SaveVideo prefix unique per clip.

    ComfyUI's own counter restarts from zero when it restarts, so a prefix that
    only carried a user id produced repeat filenames across sessions — and a
    repeat filename is a stale clip served out of the browser cache under a new
    run's name.
    """
    who = f'{user_id}_' if user_id is not None else ''
    return f'{who}lds_video_test_{uuid.uuid4().hex[:8]}'

logger = logging.getLogger(__name__)
N_VAE_AUDIO = '24'

def _loras_write_dir():
    """Where the studio DEPLOYS a trained video LoRA, created on demand.

    ComfyUI's loader lists files under its own roots and nothing else, so a
    checkpoint sitting in the app's checkpoint store is invisible to it however
    valid it is. Deployment is a copy into the FIRST loras root — the one
    ComfyUI writes to and lists first.
    """
    from lds_sdk.video_host import comfy_model_paths
    roots = comfy_model_paths.search_roots('loras')
    if not roots:
        return None
    dest = os.path.join(str(roots[0]), LORA_SUBDIR)
    os.makedirs(dest, exist_ok=True)
    return dest

def deployed_loras() -> list:
    """Every video LoRA ComfyUI can already load, in LoraLoader form.

    Scans the app's own deployment folder AND the surrounding `h3/` namespace,
    because a user who dropped a LoRA there by hand has a perfectly good LoRA
    that the picker refusing to list would send back to a file explorer.
    """
    from lds_sdk.video_host import comfy_model_paths
    out, seen = [], set()
    for root in comfy_model_paths.search_roots('loras'):
        for sub in (LORA_SUBDIR, 'h3'):
            folder = os.path.join(str(root), sub)
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                continue
            for name in names:
                if not name.lower().endswith('.safetensors'):
                    continue
                rel = os.path.join(sub, name)
                key = rel.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append({'filename': rel,
                            'label': name[:-len('.safetensors')],
                            'source': 'deployed'})
    return out

def _video_runs():
    """Cloud training runs that trained on a VIDEO dataset, newest first.

    `dataset_table` is the discriminator: the video and face lanes share the id
    space of `cloud_training_run`, so filtering on the id alone would offer a
    face LoRA as a video one.
    """
    from lds_sdk import cloud_runs
    return [run for run in cloud_runs.all_runs(newest_first=True)
            if run.dataset_table == 'video_dataset']

def trained_loras() -> list:
    """The LoRAs this app has trained on video, and whether each is deployed.

    One entry per checkpoint file, newest run first, carrying the run id and the
    filename so the caller can ask for a deploy. `deployed_as` is the
    LoraLoader-form name when a copy is already in ComfyUI, else None — the
    picker uses it to offer "Test" instead of "Deploy & test".

    A run whose checkpoint directory is gone (the video lane grew a DELETE)
    contributes nothing, which is the honest reading of an empty directory.
    """
    from lds_sdk import run_history as rg
    deployed = {os.path.basename(e['filename']).lower(): e['filename']
                for e in deployed_loras()}
    out = []
    for run in _video_runs():
        try:
            files = rg.run_checkpoint_files(run)
        except Exception:       # a run row pointing at an unreadable path
            continue
        for name in sorted(files):
            stem = (name[:-len('.safetensors')]
                    if name.endswith('.safetensors') else name)
            out.append({
                'run_id': run.id,
                'dataset_id': run.dataset_id,
                'filename': name,
                'label': stem,
                'run_status': run.status,
                'deployed_as': deployed.get(name.lower()),
                'source': 'trained',
            })
    return out

def deploy_checkpoint(run_id, filename) -> str:
    """Copy one trained checkpoint into ComfyUI's loras folder, and name it.

    Returns the LoraLoader-form name (`h3/lds/<file>`) the graph should use.

    Idempotent by size: re-testing a LoRA does not re-copy 300 MB every time,
    while a checkpoint that was overwritten upstream still refreshes.

    `filename` never reaches the filesystem unchecked — `run_checkpoint_path`
    resolves by basename only and returns None for anything else, which is what
    keeps a request from walking out of the store.
    """
    from lds_sdk import cloud_runs
    from lds_sdk import run_history as rg
    run = cloud_runs.get(int(run_id))
    if run is None or run.dataset_table != 'video_dataset':
        raise ValueError('video training run not found')
    src = rg.run_checkpoint_path(run, filename)
    if not src or not os.path.isfile(src):
        raise ValueError('checkpoint file not found')
    return deploy_file(src)

def deploy_file(src) -> str:
    """The copy itself — one resolved checkpoint file into the app's folder
    under ComfyUI's loras root. Split from `deploy_checkpoint` so the dataset
    workspace can deploy a LOCAL run's save (which no CloudTrainingRun row can
    resolve) through the exact same folder and naming the Studio lists."""
    if not src or not os.path.isfile(src):
        raise ValueError('checkpoint file not found')
    dest_dir = _loras_write_dir()
    if not dest_dir:
        raise ValueError('ComfyUI loras folder is not configured')
    dst = os.path.join(dest_dir, os.path.basename(src))
    if not (os.path.isfile(dst)
            and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy2(src, dst)
        logger.info('video studio: deployed %s into %s',
                    os.path.basename(src), LORA_SUBDIR)
    return os.path.join(LORA_SUBDIR, os.path.basename(src))

def undeploy_lora(deployed_as) -> str:
    """⏏ Move one deployed copy OUT of ComfyUI's loras folder, into the trash.

    Takes the LoraLoader-form name `deploy_file` answered (`h3/lds/<file>`)
    and nothing else: a name outside the app's own subfolder is refused, so a
    LoRA the user dropped by hand under `h3/` — which the picker lists as
    deployed on purpose — can never be trashed by a click in a dataset list.
    Basename-only inside that folder, for the same reason every resolver here
    is. Returns the trashed path."""
    from lds_sdk.video_host import comfy_model_paths
    from lds_sdk.video_host import trash
    rel = os.path.normpath(str(deployed_as or ''))
    sub, name = os.path.split(rel)
    own = os.path.normcase(os.path.normpath(LORA_SUBDIR))
    if (not name or os.path.normcase(sub) != own
            or not name.lower().endswith('.safetensors')):
        raise ValueError('only a LoRA the app deployed itself can be undeployed '
                         'from here')
    for root in comfy_model_paths.search_roots('loras'):
        path = os.path.join(str(root), LORA_SUBDIR, name)
        if os.path.isfile(path):
            return trash.send_to_trash(path, context='video_lora_undeploy')
    raise ValueError('that LoRA is not in ComfyUI\'s loras folder any more')

LORA_EXT = '.safetensors'

def import_external_lora(src_path=None, upload=None, filename=None) -> dict:
    """Copy a LoRA the user already has into ComfyUI's folder, and name it.

    Two ways in, because the two are different situations: a PATH (the file is
    on this machine — nothing crosses HTTP, which matters at 300 MB) and an
    UPLOAD (it is on the phone, or on the machine driving the browser).

    Refuses rather than guesses:
      * anything that is not a .safetensors — the loader reads nothing else,
        and an entry that fails at generation time is worse than no entry;
      * a name that could resolve outside the folder (traversal, drive letter,
        rooted path) — the same sanitizer the canvas uses;
      * a DIFFERENT file already sitting under that name. Overwriting it would
        silently change what every clip generated with that name meant, and the
        picker would keep showing one label for two different weights. Same
        name AND same size is treated as already imported, so re-importing is
        free — the idempotence deploy_checkpoint already applies.

    Returns {'filename': 'h3/lds/<name>', 'label': <stem>, 'bytes': n,
    'already': bool} — the LoraLoader-form name the graph will use.
    """
    from lds_sdk.video_host.studio import is_unsafe_external_lora_name as _is_unsafe_external_lora_name
    name = os.path.basename(str(filename or src_path or '')).strip()
    if not name or _is_unsafe_external_lora_name(name):
        raise ValueError('that file name cannot be used')
    if not name.lower().endswith(LORA_EXT):
        raise ValueError(f'a LoRA is a {LORA_EXT} file — this one is not, and '
                         f'ComfyUI would not load it')
    dest_dir = _loras_write_dir()
    if not dest_dir:
        raise ValueError('ComfyUI loras folder is not configured')
    dst = os.path.join(dest_dir, name)

    if src_path is not None:
        src = os.path.abspath(str(src_path))
        if not os.path.isfile(src):
            raise ValueError('that file is not on this machine')
        size = os.path.getsize(src)
        if os.path.isfile(dst):
            if os.path.getsize(dst) == size:
                return {'filename': os.path.join(LORA_SUBDIR, name),
                        'label': name[:-len(LORA_EXT)], 'bytes': size,
                        'already': True}
            raise ValueError(f'a different {name} is already in the folder — '
                             f'rename yours, so the two stay tellable apart')
        shutil.copy2(src, dst)
    else:
        if upload is None:
            raise ValueError('attach a file, or give a path on this machine')
        if os.path.isfile(dst):
            # An upload has no size to compare before it is written, so the
            # collision is refused outright rather than after 300 MB.
            raise ValueError(f'{name} is already in the folder — rename yours, '
                             f'or use the one that is there')
        upload.save(dst)
        size = os.path.getsize(dst)
    logger.info('video studio: imported %s into %s', name, LORA_SUBDIR)
    return {'filename': os.path.join(LORA_SUBDIR, name),
            'label': name[:-len(LORA_EXT)],
            'bytes': os.path.getsize(dst), 'already': False}

def eros_on_disk() -> bool:
    """Is the 10Eros weight actually on THIS machine?

    Read at build time and passed into the pure builder, so the fail-open path
    is a decision with a reason rather than a graph ComfyUI rejects.
    """
    from lds_sdk.video_host import comfy_model_paths
    for kind in ('diffusion_models', 'unet'):
        for root in comfy_model_paths.search_roots(kind):
            if os.path.isfile(os.path.join(str(root), BASE_EROS)):
                return True
    return False

def light_on_disk() -> bool:
    """🪶 Is the W4A8 base on THIS machine? Same role as `eros_on_disk`, through
    the shared scanner (every root, case-insensitively) like the accelerations."""
    return _weight_present(('diffusion_models', 'unet'), BASE_LIGHT)

def _parse_comfy_version(text):
    """'0.34.0', 'v0.31.0' or '0.31.0-dev' -> (0, 34, 0); anything else -> None."""
    m = re.match(r'\s*v?(\d+)\.(\d+)(?:\.(\d+))?', str(text or ''))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))

def light_status(comfy_version=None) -> dict:
    """🪶 Whether the Render panel may offer the lighter base, and the sentence
    for a greyed box.

    Two things can say no, and the panel must not guess which: the file is not
    on disk (Setup fetches it), or the running ComfyUI predates the format
    (the file is there and the server would refuse it at load — the worse
    failure, because it happens minutes into a queue). `comfy_version` is the
    string /system_stats echoes, read ONCE by the caller alongside the launch
    advice; None or unreadable is silence, and silence is not a no — same rule
    as the node packs."""
    on_disk = light_on_disk()
    parsed = _parse_comfy_version(comfy_version)
    comfy_ok = None if parsed is None else parsed >= LIGHT_MIN_COMFY
    if not on_disk:
        hint = 'Not on this machine — Setup can fetch it (12.5 GB).'
    elif comfy_ok is False:
        hint = (f'Needs ComfyUI {LIGHT_MIN_COMFY_LABEL} or newer to read its 4-bit '
                f'format; this one is {comfy_version}. Update ComfyUI first.')
    else:
        hint = ''
    return {'on_disk': on_disk, 'comfy_version': comfy_version, 'comfy_ok': comfy_ok,
            'min_comfy': LIGHT_MIN_COMFY_LABEL,
            'available': on_disk and comfy_ok is not False, 'hint': hint}

def light_usable():
    """🪶 The launch-time twin of `light_status`: (usable, why not).

    The panel's verdict was taken at another moment, on another request; the
    launch replays BOTH halves — the file and the server's version — because
    a graph naming a weight the server cannot read fails minutes into the
    queue, which is the failure the verdict exists to prevent. One extra
    /system_stats read, and only when the box is ticked."""
    _argv, _ram, version = comfyui_launch_facts()
    st = light_status(version)
    if st['available']:
        return True, ''
    if not st['on_disk']:
        return False, 'absent'
    return False, f'needs ComfyUI {LIGHT_MIN_COMFY_LABEL}+ (this one is {version})'

def reference_sparse_compatible(info):
    """Older packs silently ignore the controls protecting Ref LoRA forwards."""
    inputs = (info or {}).get('input') or {}
    fields = {**inputs.get('required', {}), **inputs.get('optional', {})}
    for field, value in (('qkv_streaming_mode', 'Off'), ('mlp_memory', 'off'),
                         ('precision_mode', 'Preserve native')):
        spec = fields.get(field)
        if not isinstance(spec, (list, tuple)) or not spec:
            return False
        choices = spec[0] if isinstance(spec[0], list) else (
            spec[1].get('options', []) if len(spec) > 1 and isinstance(spec[1], dict) else [])
        if value not in choices:
            return False
    return True

def preflight(workflow):
    """Refuse a run whose graph this install cannot execute.

    TWO checks, because the shared scanner cannot see all of this lane.

    It knows the loaders in `_STUDIO_MODEL_LOADERS` — UNET, CLIP, VAE, LoRA —
    and it compares the graph's node classes against /object_info. What it
    cannot know is the latent upscaler: that weight is read by a THIRD-PARTY
    loader, from `models/latent_upscale_models/`, a folder ComfyUI itself does
    not define. A run armed with the upscale and missing that file passes every
    shared check and then dies mid-job, which is the failure this whole preflight
    exists to prevent. So it is checked here, by name, first.

    Raising the studio's own exception type means the existing structured 409
    renders this lane's gaps with no new plumbing on either side.
    """
    from lds_sdk.video_host import studio as lts
    if N_REF_SPARSE_MEMORY in workflow:
        from lds_sdk.video_host.comfyui import fetch_node_info
        if not reference_sparse_compatible(fetch_node_info('H3MemoryOptimization')):
            raise ValueError('Sparse attention with Ref4/Ref8 needs the current H3-Optimizations '
                             'memory controls. Update that pack and restart ComfyUI, or choose '
                             'dense attention. If ComfyUI is offline, reconnect it and retry.')
    # The reference bypass loader names a real LoRA even though it is not in
    # the shared scanner's stock loader table. Scan its equivalent stock input
    # as well, retaining the real node for class availability checks.
    reference_loaders = [n for n in workflow.values() if n.get('class_type') == vrc.REF_LORA_CLASS]
    if reference_loaders:
        scanner = dict(workflow)
        for index, node in enumerate(reference_loaders):
            scanner[f'reference_asset_{index}'] = {
                'class_type': 'LoraLoaderModelOnly', 'inputs': {
                    'lora_name': node['inputs']['lora_name'], 'strength_model': 1.0,
                },
            }
        workflow = scanner
    if N_UPSCALE_PARAMS in workflow:
        wanted = (workflow[N_UPSCALE_PARAMS].get('inputs') or {}).get('model_name')
        if wanted and not _weight_present(('latent_upscale_models',), wanted):
            raise lts.StudioAssetsMissing(
                'h3video',
                [{'path': f'models/latent_upscale_models/{wanted}',
                  'kind': 'latent upscaler',
                  'hint': 'the latent upscale needs this file; place it in that '
                          'folder, or turn the option off'}],
                [])
    if N_VDN in workflow:
        # ⚡ The VDN stage is read by a THIRD-PARTY node from a folder ComfyUI
        # itself does not define (models/vdn), so the shared scanner cannot
        # see it either: checked here by name, like the upscaler. The node's
        # declared inputs are read too — an older pack lacks one the graph
        # writes, and ComfyUI's answer to that is a refusal after the queue.
        gone = vdn_stage_missing()
        if gone:
            raise lts.StudioAssetsMissing(
                'h3video',
                [{'path': f'models/{VDN_FOLDER}/{rel}', 'kind': 'VDN-H3 stage',
                  'hint': f'Setup downloads the VDN-H3 stage ({VDN_SETUP_PATH}), '
                          'or choose another acceleration'} for rel in gone],
                [])
        from lds_sdk.video_host.comfyui import fetch_node_info
        info = fetch_node_info(VDN_CLASS)
        unwritten = vdn_node_unwritten_required(info)
        if unwritten:
            raise ValueError('This ComfyUI-VDN-H3 declares required inputs this app does not '
                             f"write ({', '.join(unwritten)}): update the app, or choose another "
                             'acceleration.')
        listed = vdn_node_checkpoints(info)
        if listed is not None and VDN_STAGE not in listed:
            raise ValueError(f'The VDN-H3 node does not list the {VDN_STAGE} stage although its '
                             f'files are under models/{VDN_FOLDER}/: that ComfyUI reads another '
                             'models root — put the stage under the one its loras folder sits in.')
    lts.preflight_family('h3video', [workflow])

def reference_weight_name(folders, filename):
    """Return the loader's real relative name, including an h3/ subfolder."""
    from lds_sdk.video_host import comfy_model_paths as cmp
    for folder in folders:
        try:
            roots = cmp.search_roots(folder)
            found = cmp.scan_family_tree(roots, ('',),
                root_file_accept=lambda name: name.lower() == filename.lower(),
                accept=lambda name: name.lower() == filename.lower())
        except (OSError, ValueError):
            found = []
        if found:
            return found[0]
    return None

def reference_status(classes=None, comfy_version=None):
    """Readiness belongs to the task and selected base, never to FL2VA."""
    shared_missing = [m for m in missing_weights()
                      if m['required'] and m['filename'] != BASE_OFFICIAL]
    required_nodes = {'MiniMaxH3ReferenceToVideo', 'LoadImage', 'LoadVideo', 'Video Slice',
                      'GetVideoComponents', 'LoadAudio'}
    missing_nodes = sorted(required_nodes - classes) if classes is not None else []
    version = _parse_comfy_version(comfy_version)
    bases = []
    for spec in vrc.REF_BASES:
        found = reference_weight_name(('diffusion_models', 'unet'), spec['file'])
        version_ok = spec['id'] != 'light' or version is None or version >= LIGHT_MIN_COMFY
        available = bool(found) and version_ok
        base_missing = list(shared_missing)
        if not found:
            base_missing.append({'action': spec['action'], 'filename': spec['file'],
                                 'what': spec['label'], 'required': True,
                                 'place_in': 'models/diffusion_models/'})
        bases.append({**spec, 'weight_present': bool(found), 'available': available,
                      'loader_name': found, 'ready': available and classes is not None
                      and not shared_missing and not missing_nodes,
                      'missing_weights': base_missing, 'comfy_ok': version_ok})
    accels = []
    for spec in vrc.REF_ACCELERATIONS:
        if spec['id'] == 'vdn':
            pack = option_availability(classes)['vdn']
            present = vdn_stage_present()
            accels.append({**spec, 'weight_present': present, 'loader_name': None,
                           'available': present and pack['available'] is not False,
                           'nodes_action': None, 'pack': pack, 'setup_path': VDN_SETUP_PATH})
            continue
        found = reference_weight_name(('loras',), spec['file'])
        nodes_ok = classes is not None and vrc.REF_LORA_CLASS in classes
        accels.append({**spec, 'weight_present': bool(found), 'loader_name': found,
                       'available': bool(found) and nodes_ok, 'nodes_action': vrc.REF_NODE_ACTION})
    missing = list(shared_missing)
    for spec in [*bases, *accels]:
        if not spec['weight_present']:
            missing.append({'action': spec['action'], 'filename': spec['file'],
                            'what': spec['label'], 'required': False,
                            'place_in': ('models/vdn/' if spec.get('id') == 'vdn' else
                                         'models/loras/' if spec in accels else 'models/diffusion_models/')})
    node_available = classes is not None and vrc.REF_LORA_CLASS in classes
    if classes is not None and not node_available:
        missing_nodes = [*missing_nodes, vrc.REF_LORA_CLASS]
    return {'limits': dict(vrc.REF_LIMITS), 'bases': bases, 'accelerations': accels,
            'ready': any(b['ready'] for b in bases), 'missing_weights': missing,
            'missing_nodes': missing_nodes, 'ref_image_sizes': ['match', 'max'],
            'node_action': vrc.REF_NODE_ACTION, 'node_available': node_available,
            'default_base': 'official', 'default_accel': 'ref8', 'default_steps': DEFAULT_STEPS,
            'nodes_known': classes is not None, 'video_fps': 24,
            'video_seconds': {'min': 2, 'max': 15}, 'audio_seconds': {'min': 0.2, 'max': 15}}

def last_frame_command(ffmpeg, src, dst):
    """The last frame of `src` as a PNG: seek into the last second and take the
    first frame of the REVERSED tail — the true last frame whatever the exact
    duration, where `-sseof` alone can land past it and write nothing."""
    return [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-sseof', '-1', '-i', src,
            '-vf', 'reverse', '-frames:v', '1', '-update', '1', dst]

REQUIRED_WEIGHTS = (
    ('h3_base', ('diffusion_models', 'unet'), BASE_OFFICIAL,
     'the model itself'),
    ('h3_text_encoder', ('text_encoders', 'clip'),
     'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'the prompt encoder'),
    ('h3_video_vae', ('vae',), 'minimax_h3_video_vae_fp16.safetensors',
     'the picture decoder'),
    ('h3_audio_vae', ('vae',), 'minimax_h3_audio_vae_fp32.safetensors',
     'the sound decoder'),
)

OPTIONAL_WEIGHTS = (
    ('h3_turbo_lora', ('loras',), TURBO_LORA, 'turbo'),
    # ⚡ The other two accelerations of the Render panel (arena rows 2 and 3).
    ('h3_parasyte_lora', ('loras',), PARASYTE_LORA, 'parasyte'),
    ('h3_dareties_lora', ('loras',), DARETIES_LORA, 'dareties'),
    # ⚡ VDN-H3: a directory, not a file. The row names the branch file INSIDE
    # the stage (relative to models/vdn — `_weight_present` joins it under
    # every root), one row per precision so each Setup button reads its own
    # file; the acceleration itself is available with either branch present
    # (`vdn_stage_present`). Neither is in the card's one-click plan: 5.5 GB
    # for an opt-in that also needs a node pack is a row, never a default.
    ('h3_vdn_stage', (VDN_FOLDER,), f'{VDN_STAGE}/{VDN_BRANCH_FILE}', 'vdn'),
    ('h3_vdn_stage_int8', (VDN_FOLDER,), f'{VDN_STAGE}/{VDN_BRANCH_FILE_INT8}', 'vdn_int8'),
    # ⚠️ No setup action for this one, on purpose — see UNFETCHABLE below.
    (None, ('latent_upscale_models',), UPSCALE_MODEL, 'latent_upscale'),
    # Nor for the third-party base: it is somebody else's finetune, it is
    # 20 GB, and nothing in the app needs it. Present on disk, the box works.
    (None, ('diffusion_models', 'unet'), BASE_EROS, 'eros'),
    # 🪶 Unlike 10Eros this one HAS a button: it is the official weights
    # re-quantized, apache-2.0 on the hub, and it is what lets a 24 GB card
    # breathe — a lane feature, not somebody else's finetune.
    ('h3_base_light', ('diffusion_models', 'unet'), BASE_LIGHT, 'light'),
    # 🔴 The int8 video VAE of the live channel's real-time preset — absent, the
    # fp16 one decodes; nothing else changes.
    ('h3_video_vae_int8', ('vae',), VIDEO_VAE_INT8, 'video_vae_int8'),
)

UNFETCHABLE = {
    UPSCALE_MODEL: 'models/latent_upscale_models/',
    BASE_EROS: 'models/diffusion_models/',
}

OPTION_NODE_PACKS = {
    'turbo': {
        'pack': 'ComfyUI-MiniMax-H3-Turbo',
        'url': 'https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo',
        'search': 'MiniMax H3 Turbo',
        'classes': ('MiniMaxH3TurboLoRA', 'MiniMaxH3TurboSampler'),
    },
    'sparse': {
        'pack': 'H3-Optimizations',
        'url': 'https://github.com/Zironic/H3-Optimizations',
        'search': 'H3 Optimizations',
        'classes': ('H3SparseAttentionAdvanced',),
    },
    # ⚡ VDN-H3: the native port of OpenVDN's hybrid attention (Apache-2.0,
    # declares no pip dependency — the one signal this table asks of a pack).
    'vdn': {
        'pack': 'ComfyUI-VDN-H3',
        'url': 'https://github.com/Saganaki22/ComfyUI-VDN-H3',
        'search': 'VDN-H3',
        'classes': (VDN_CLASS,),
    },
    # ↗ VFI is two packs, and both are named: the interpolator and the video
    # helper that reads a finished mp4 back in and writes the smoothed one out.
    'vfi': {
        'pack': 'ComfyUI-Frame-Interpolation + VideoHelperSuite',
        'url': 'https://github.com/Fannovel16/ComfyUI-Frame-Interpolation',
        'search': 'Frame Interpolation',
        'classes': ('RIFE VFI', 'VHS_LoadVideoPath', 'VHS_VideoCombine'),
    },
    'latent_upscale': {
        'pack': 'Comfyui-MMH3-UltimateUpscale',
        'url': 'https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale',
        'search': 'MMH3 Ultimate Upscale',
        'classes': ('MMH3UltimateUpscale', 'MMH3LatentUpscaleWithModelParams',
                    'MMH3TemporalSplitParams'),
    },
}

SAGE_CLASS = 'PathchSageAttentionKJ'

SAGE_PACK = {
    'pack': 'ComfyUI-KJNodes',
    'url': 'https://github.com/kijai/ComfyUI-KJNodes',
    'search': 'KJNodes',
}

def _weight_present(subfolders, filename) -> bool:
    """Is this file under any of ComfyUI's roots for those subfolders?

    Case-insensitively, and across EVERY root (the yaml's extra paths included),
    like the loader that will be handed the name — a weight deployed into an
    extra_model_paths root is present, whatever the app would have chosen.
    """
    from lds_sdk.video_host import comfy_model_paths

    def current(path):
        if not os.path.isfile(path):
            return False
        if filename == VIDEO_VAE_INT8:
            from lds_sdk.h3_downloads import H3_DOWNLOADS
            try:
                return os.path.getsize(path) not in H3_DOWNLOADS['h3_video_vae_int8']['superseded_bytes']
            except OSError:
                return False
        return True

    for sub in subfolders:
        try:
            roots = comfy_model_paths.search_roots(sub)
        except Exception:
            roots = []
        for root in roots:
            if current(os.path.join(str(root), filename)):
                return True
            try:
                names = {n.lower(): n for n in os.listdir(str(root))}
            except OSError:
                continue
            if filename.lower() in names and current(os.path.join(str(root), names[filename.lower()])):
                return True
    return False

def vdn_roots() -> list[str]:
    """Every `models/vdn` folder the VDN node will look in, derived the way the
    pack derives it (vdn_h3/spec.py `register_folder`): a `vdn` folder BESIDE
    each loras root ComfyUI knows — the default one and every extra root a
    yaml declares under `loras:` — plus any root a yaml declares under `vdn:`
    outright. Reading only the `vdn:` key made a stage in a standard extra
    tree load in ComfyUI while Setup called it absent and offered 5.5 GB."""
    from lds_sdk.video_host import comfy_model_paths
    out = []

    def add(path):
        p = os.path.normpath(str(path))
        if p not in out:
            out.append(p)
    try:
        for r in comfy_model_paths.search_roots('loras'):
            add(os.path.join(os.path.dirname(os.path.normpath(str(r))), VDN_FOLDER))
    except Exception:
        pass
    try:
        for r in comfy_model_paths.search_roots(VDN_FOLDER):
            add(r)
    except Exception:
        pass
    return out

def vdn_stage_missing() -> list[str]:
    """What the VDN stage still lacks, as paths relative to models/vdn — [] when
    a complete stage sits under any ComfyUI root: a branch file in EITHER
    precision (the node reads both and picks under VRAM pressure) plus the spec
    and the two adapters it refuses to run without. The most complete root
    wins, so a stage half-downloaded elsewhere never masks a whole one."""
    best = None
    for root in vdn_roots():
        gone = vdn_stage_missing_under(root)
        if best is None or len(gone) < len(best):
            best = gone
        if not gone:
            break
    return [f'{VDN_STAGE}/{f}' for f in (best if best is not None else [VDN_BRANCH_FILE, *VDN_STAGE_REQUIRED])]

def vdn_stage_missing_under(root) -> list[str]:
    """The files the stage under ONE models/vdn root lacks (relative to the
    stage): a branch in either precision, the spec, the two adapters."""
    stage = os.path.join(str(root), VDN_STAGE)
    gone = [f for f in VDN_STAGE_REQUIRED if not os.path.isfile(os.path.join(stage, f))]
    if not any(os.path.isfile(os.path.join(stage, f)) for f in (VDN_BRANCH_FILE, VDN_BRANCH_FILE_INT8)):
        gone.insert(0, VDN_BRANCH_FILE)
    return gone

def vdn_stage_present() -> bool:
    return not vdn_stage_missing()

def vdn_node_unwritten_required(info) -> list[str]:
    """The REQUIRED inputs the registered ApplyVDNH3 declares that the graph
    does not write — [] when every one is covered, or when there is no entry
    to read (silence is not a verdict).

    That direction, and only that one: ComfyUI keeps an input the node does
    not declare out of the call (execution.get_input_data) and refuses a
    prompt only for a declared required input it lacks (validate_inputs,
    `required_input_missing`) — so an older pack without `retain_buffers`
    runs this graph fine, and a newer pack with a required input this app
    has never heard of is what fails, minutes into the queue."""
    inputs = (info or {}).get('input') or {}
    required = inputs.get('required') or {}
    written = set(VDN_NODE_INPUTS) | {'model'}
    return sorted(name for name in required if name not in written)

def vdn_node_checkpoints(info):
    """The stage directories the node lists in its `vdn_checkpoint` combo, or
    None when the entry does not carry one (unreadable, older shape)."""
    try:
        combo = ((info or {}).get('input') or {}).get('required', {}).get('vdn_checkpoint')
        options = combo[0] if isinstance(combo, (list, tuple)) and combo else None
        return [str(o) for o in options] if isinstance(options, (list, tuple)) else None
    except (AttributeError, TypeError, IndexError):
        return None

def _vdn_row_present(branch_rel) -> bool:
    if vdn_stage_missing():
        return False
    return any(os.path.isfile(os.path.join(root, VDN_STAGE, branch_rel)) for root in vdn_roots())

_ROW_PRESENT = {
    # The main row answers for the acceleration: a complete stage with either
    # branch (an INT8-only install is a working VDN-H3, not a red row for life).
    'h3_vdn_stage': vdn_stage_present,
    # The INT8 row answers for its own file, inside a complete stage.
    'h3_vdn_stage_int8': lambda: _vdn_row_present(VDN_BRANCH_FILE_INT8),
}

def missing_weights() -> list[dict]:
    """Every weight this lane wants and cannot find, required ones first.

    Each entry carries what the Setup screen needs to act (`action`, or None
    when the app will not fetch it and `place_in` says where to put it by hand)
    and what a human needs to decide (`what` — 'the prompt encoder' beats
    'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors').
    """
    out = []
    for action, subs, filename, what in REQUIRED_WEIGHTS:
        if not _weight_present(subs, filename):
            out.append({'action': action, 'filename': filename, 'what': what,
                        'required': True, 'place_in': f'models/{subs[0]}/'})
    for action, subs, filename, what in OPTIONAL_WEIGHTS:
        if action and not _setup_action_available(action):
            continue
        # A row that is a stage answers for the whole stage (_ROW_PRESENT),
        # so Setup and the Render panel cannot disagree about it.
        present = _ROW_PRESENT[action]() if action in _ROW_PRESENT else _weight_present(subs, filename)
        if not present:
            out.append({'action': action, 'filename': filename,
                        'what': what, 'required': False,
                        'place_in': UNFETCHABLE.get(filename, f'models/{subs[0]}/')})
    return out

def registered_classes():
    """The class set the target ComfyUI actually registers, or None.

    None means the probe could not be made — never an empty set: an unreachable
    /object_info would otherwise read as "this install has no nodes at all" and
    grey out every option on a machine that has them.
    """
    from lds_sdk.video_host.comfyui import fetch_object_info_classes
    try:
        return fetch_object_info_classes()
    except Exception:
        return None

def option_availability(classes=None) -> dict:
    """{option: {available, pack, url, search, nodes}} — what each checkbox needs.

    `available` is None when /object_info could not be read: the UI keeps
    offering the option, because a probe that could not run is not a verdict.
    Same fail-open rule as the preflight.

    `pack`/`url`/`search` travel with every entry, available or not, so the panel
    can say WHICH pack to install without a second table to keep in step.
    """
    if classes is None:
        classes = registered_classes()
    out = {}
    for option, spec in OPTION_NODE_PACKS.items():
        row = {'pack': spec['pack'], 'url': spec['url'], 'search': spec['search']}
        if classes is None:
            out[option] = {**row, 'available': None, 'nodes': []}
            continue
        gone = [c for c in spec['classes'] if c not in classes]
        out[option] = {**row, 'available': not gone, 'nodes': gone}
    return out

def accelerations_status(classes=None, vram_gb=None) -> list:
    """The Render panel's acceleration choices, resolved against THIS machine:
    the weight on disk and, for larryvrh's and VDN-H3's, the node pack too.
    `available` is what the panel can offer; `action` is the Setup button that
    fetches the weight; `pack` travels for the choices that need code
    installed. `vram_gb` adds lighter-base advice to VDN-H3's hint without
    changing availability or vetoing the selected base at launch."""
    packs = option_availability(classes)
    out = []
    for a in ACCELERATIONS:
        if not _setup_action_available(a['action']):
            continue
        # A LoRA is one file under loras/; the VDN stage is a directory whose
        # completeness is its own question (either branch, plus the adapters).
        weight = vdn_stage_present() if a['id'] == 'vdn' else _weight_present(('loras',), a['file'])
        pack = packs.get(a['pack']) if a['pack'] else None
        pack_ok = None if pack is None else pack['available']
        hint = a['hint']
        needs_light = bool(a['id'] == 'vdn' and vram_gb is not None
                           and vram_gb < VDN_LIGHT_ADVICE_VRAM_GB)
        if needs_light:
            hint = f'{hint} {VDN_LIGHT_ADVICE}'
        out.append({
            'id': a['id'], 'label': a['label'], 'arena': a['arena'], 'steps': a['steps'],
            'author': a['author'], 'license': a['license'], 'file': a['file'],
            'action': a['action'], 'hint': hint, 'weight_present': bool(weight),
            'pack': pack, 'needs_light': needs_light,
            'setup_path': VDN_SETUP_PATH if a['id'] == 'vdn' else f"{CARD_SETUP_PATH} › {a['label']}",
            # A probe that could not run is not a no — same rule as the packs.
            'available': bool(weight) and pack_ok is not False,
        })
    return out


def _setup_action_available(action):
    """Older packages must not advertise preparation their installer cannot run."""
    from flask import has_app_context
    if not has_app_context():
        return True
    from .setup import known_action
    return known_action(action)

def studio_ready(missing=None) -> bool:
    """Can a clip be rendered right now, with the default options off?

    THE verdict — capabilities, the options route and the enqueue preflight all
    ask this one function, so no surface can decide readiness from a different
    subset of the gaps.
    """
    if missing is None:
        missing = missing_weights()
    return not any(m['required'] for m in missing)

H3_HOST_RAM_GB = 43

FAST_DISK_RAM_FLOOR_GB = 60

FAST_DISK_MIN_COMFYUI = (0, 23, 0)

_FAST_DISK_FLAG = '--fast-disk'

_HIGH_RAM_FLAG = '--high-ram'

_DYNAMIC_VRAM_SWITCH = '--disable-dynamic-vram'

_DYNAMIC_VRAM_FORCE = '--enable-dynamic-vram'

_DYNAMIC_VRAM_OFF_MODES = ('--novram', '--highvram', '--gpu-only', '--cpu')

_VERSION_RE = re.compile(r'(\d+)\.(\d+)(?:\.(\d+))?')

def knows_fast_disk(comfyui_version) -> bool:
    """Does a ComfyUI reporting this version string declare `--fast-disk`?

    Reads the leading `major.minor[.patch]` out of whatever the server sent
    ("0.30.1", "v0.23.0", "0.30.1+16"); anything else is "cannot tell" = False.
    """
    m = _VERSION_RE.search(str(comfyui_version or ''))
    if not m:
        return False
    return tuple(int(g or 0) for g in m.groups()) >= FAST_DISK_MIN_COMFYUI

def launch_advice(argv, ram_total_gb, comfyui_version=None):
    """What to change on the command that starts ComfyUI, or None. Pure.

    {flag, add, remove, ram_total_gb, weights_gb}: `flag` is `--fast-disk`;
    `add` says whether it is missing from the line (False = already there);
    `remove` names `--disable-dynamic-vram` when the launcher carries it and
    nothing forces the loader back on, because the flag does nothing until that
    switch is gone — a card that only said "add --fast-disk" to such a launcher
    would send the user to a change that changes nothing.

    None covers every case where nothing should be said: no argv (an instance
    too old to echo it, or unreachable), a ComfyUI that predates the flag, no
    RAM figure or enough RAM, `--high-ram`, a memory mode with no dynamic
    loader, and a line that already has what it needs.
    """
    if not isinstance(argv, (list, tuple)) or not argv:
        return None
    if not knows_fast_disk(comfyui_version):
        return None
    if not isinstance(ram_total_gb, (int, float)) or isinstance(ram_total_gb, bool):
        return None
    if ram_total_gb <= 0 or ram_total_gb >= FAST_DISK_RAM_FLOOR_GB:
        return None
    flags = {str(a).split('=', 1)[0].strip() for a in argv}
    if _HIGH_RAM_FLAG in flags:
        return None
    forced_on = _DYNAMIC_VRAM_FORCE in flags
    if not forced_on and any(m in flags for m in _DYNAMIC_VRAM_OFF_MODES):
        return None
    switched_off = (not forced_on) and _DYNAMIC_VRAM_SWITCH in flags
    has_flag = _FAST_DISK_FLAG in flags
    if has_flag and not switched_off:
        return None
    return {'flag': _FAST_DISK_FLAG, 'add': not has_flag,
            'remove': _DYNAMIC_VRAM_SWITCH if switched_off else None,
            'ram_total_gb': round(float(ram_total_gb), 1),
            'weights_gb': H3_HOST_RAM_GB}

def comfyui_launch_facts(timeout=3):
    """(argv, ram_total_gb, version) of the RUNNING ComfyUI, from /system_stats.

    NETWORK — one short GET, kept out of the pure `launch_advice` so the
    decision stays testable without a server. Each field is None on its own
    when the server did not send it (an older instance echoes its argv without
    a RAM figure, and that argv is still worth having); all three are None when
    ComfyUI is not configured, does not answer, or answers nonsense. Never
    raises.
    """
    from lds_sdk.video_host import config as cfg
    import requests
    api = (cfg.get('comfyui.api_url') or '').rstrip('/')
    if not api:
        return None, None, None
    try:
        r = requests.get(f'{api}/system_stats', timeout=timeout, allow_redirects=False)
        if r.status_code != 200:
            return None, None, None
        payload = r.json()
        system = (payload.get('system') if isinstance(payload, dict) else None) or {}
    except Exception:
        return None, None, None
    argv = system.get('argv')
    ram = system.get('ram_total')
    version = system.get('comfyui_version')
    ram_gb = (ram / 1024 ** 3) if isinstance(ram, (int, float)) and not isinstance(ram, bool) and ram > 0 else None
    return ((list(argv) if isinstance(argv, (list, tuple)) else None), ram_gb,
            (str(version) if version else None))

def vdn_node_compatible(info) -> bool:
    return not vdn_node_unwritten_required(info)

def sage_available(classes=None) -> bool:
    """Whether to keep the SageAttention patch in the graph.

    Fails CLOSED, unlike everything else here: when /object_info cannot be read
    we build WITHOUT Sage. A graph missing a speed patch renders correctly and a
    little slower; a graph naming a node the install does not have is refused
    outright. The asymmetry is the whole reason this one is not `is not None`.

    KJNodes' class alone is not enough: KJNodes registers the patch without the
    `sageattention` package and imports it only when the patch runs, so every
    render failed on a ComfyUI that had KJNodes and not the package. Sage stays
    only when that ComfyUI's Python is known to import it.
    """
    if classes is None:
        classes = registered_classes()
    if not classes or SAGE_CLASS not in classes:
        return False
    from lds_sdk.video_host.comfyui import sageattention_importable
    return sageattention_importable(classes)

# Public engine primitives; private names remain legacy Video compatibility aliases.
profile = _profile
weight_present = _weight_present

DYNAMIC_VRAM_FORCE = _DYNAMIC_VRAM_FORCE

DYNAMIC_VRAM_OFF_MODES = _DYNAMIC_VRAM_OFF_MODES

DYNAMIC_VRAM_SWITCH = _DYNAMIC_VRAM_SWITCH

FAST_DISK_FLAG = _FAST_DISK_FLAG

FRAME_MOD = _FRAME_MOD

FRAME_OFFSET = _FRAME_OFFSET

HIGH_RAM_FLAG = _HIGH_RAM_FLAG

ROW_PRESENT = _ROW_PRESENT

VERSION_RE = _VERSION_RE

drop_sage = _drop_sage

graft_block_attention = _graft_block_attention

graft_latent_upscale = _graft_latent_upscale

graft_reference_accel = _graft_reference_accel

graft_sparse = _graft_sparse

graft_stock_accel = _graft_stock_accel

graft_test_lora = _graft_test_lora

graft_turbo = _graft_turbo

graft_vdn = _graft_vdn

insert_into_chain = _insert_into_chain

loras_write_dir = _loras_write_dir

make_t2v = _make_t2v

model_readers = _model_readers

parse_comfy_version = _parse_comfy_version

vdn_row_present = _vdn_row_present

video_runs = _video_runs

__all__ = [
    'ACCELERATIONS',
    'ACCEL_IDS',
    'ATTN_EDGE_PCT',
    'ATTN_FAST_BACKENDS',
    'ATTN_KITCHEN',
    'ATTN_PYTORCH',
    'ATTN_SAGE',
    'BASE_EROS',
    'BASE_LIGHT',
    'BASE_OFFICIAL',
    'BLOCK_ATTN_CLASS',
    'BLOCK_ATTN_INSTALL_ACTION',
    'CARD_SETUP_PATH',
    'DARETIES_LORA',
    'DEFAULT_STEPS',
    'DYNAMIC_VRAM_FORCE',
    'DYNAMIC_VRAM_OFF_MODES',
    'DYNAMIC_VRAM_SWITCH',
    'FAST_DISK_FLAG',
    'FAST_DISK_MIN_COMFYUI',
    'FAST_DISK_RAM_FLOOR_GB',
    'FRAMES_DEFAULT',
    'FRAMES_MAX',
    'FRAMES_MIN',
    'FRAME_MOD',
    'FRAME_OFFSET',
    'H3_HOST_RAM_GB',
    'HIGH_RAM_FLAG',
    'LIGHT_MIN_COMFY',
    'LIGHT_MIN_COMFY_LABEL',
    'LORA_EXT',
    'LORA_SUBDIR',
    'MP_DEFAULT',
    'MP_MAX',
    'MP_MIN',
    'N_ACCEL_LORA',
    'N_BLOCK_ATTN',
    'N_COND',
    'N_CREATE_VIDEO',
    'N_DECODE_AUDIO',
    'N_DECODE_VIDEO',
    'N_GUIDER',
    'N_LOAD_END',
    'N_LOAD_IMAGE',
    'N_NOISE',
    'N_REF_SPARSE_MEMORY',
    'N_SAGE',
    'N_SAMPLER',
    'N_SAMPLER_SELECT',
    'N_SAVE',
    'N_SCALE',
    'N_SCHEDULER',
    'N_SHIFT',
    'N_SIZE',
    'N_SPARSE',
    'N_TEST_LORA',
    'N_TURBO_LORA',
    'N_TURBO_SAMPLER',
    'N_UNET',
    'N_UPSCALE',
    'N_UPSCALE_PARAMS',
    'N_UPSCALE_SPLIT',
    'N_VAE_AUDIO',
    'N_VAE_VIDEO',
    'N_VDN',
    'OPTIONAL_WEIGHTS',
    'OPTION_NODE_PACKS',
    'PARASYTE_LORA',
    'REQUIRED_WEIGHTS',
    'ROW_PRESENT',
    'SAGE_CLASS',
    'SAGE_PACK',
    'SPARSE_BACKEND',
    'SPARSE_MODES',
    'SPARSE_PRESETS',
    'TARGET_KEY',
    'TURBO_LORA',
    'TURBO_STEPS',
    'UNFETCHABLE',
    'UPSCALE_MODEL',
    'VDN_BRANCH_FILE',
    'VDN_BRANCH_FILE_INT8',
    'VDN_CLASS',
    'VDN_FOLDER',
    'VDN_LIGHT_ADVICE',
    'VDN_LIGHT_ADVICE_VRAM_GB',
    'VDN_HEAVY_BASE_MIN_VRAM_GB',
    'vdn_base_fits',
    'vdn_launch_guard',
    'VDN_MODES',
    'VDN_NODE_INPUTS',
    'VDN_SETUP_PATH',
    'VDN_STAGE',
    'VDN_STAGE_REQUIRED',
    'VDN_STEPS',
    'VDN_WINDOW_LATENT_FRAMES',
    'VERSION_RE',
    'VIDEO_VAE_INT8',
    'WORKFLOW_FILENAME',
    'accel_spec',
    'accelerations_status',
    'attention_spec',
    'block_attention_backends',
    'block_attention_missing_nodes',
    'block_attention_pack_installed',
    'build_workflow',
    'clamp_megapixels',
    'comfyui_launch_facts',
    'comfyui_vram_gb',
    'deploy_checkpoint',
    'deploy_file',
    'deployed_loras',
    'drop_sage',
    'eros_on_disk',
    'graft_block_attention',
    'graft_latent_upscale',
    'graft_reference_accel',
    'graft_sparse',
    'graft_stock_accel',
    'graft_test_lora',
    'graft_turbo',
    'graft_vdn',
    'import_external_lora',
    'insert_into_chain',
    'knows_fast_disk',
    'last_frame_command',
    'launch_advice',
    'light_on_disk',
    'light_status',
    'light_usable',
    'load_base_workflow',
    'local_attention_backends',
    'loras_write_dir',
    'make_t2v',
    'missing_weights',
    'model_readers',
    'new_prefix',
    'normalise_accel',
    'normalise_mode',
    'normalise_sparse',
    'option_availability',
    'parse_comfy_version',
    'preflight',
    'profile',
    'reference_accel_spec',
    'reference_sparse_compatible',
    'reference_status',
    'reference_weight_name',
    'registered_classes',
    'sage_available',
    'snap_frames',
    'studio_ready',
    'trained_loras',
    'undeploy_lora',
    'vdn_branch_active',
    'vdn_latent_frames',
    'vdn_node_checkpoints',
    'vdn_node_compatible',
    'vdn_node_unwritten_required',
    'vdn_roots',
    'vdn_row_present',
    'vdn_stage_missing',
    'vdn_stage_missing_under',
    'vdn_stage_present',
    'video_runs',
    'weight_present',
    'workflow_path',
]


# SDK 1.10 callers may still invoke these names. Memory pressure is advice;
# an installed base is never vetoed by the host's reported VRAM capacity.
VDN_HEAVY_BASE_MIN_VRAM_GB = VDN_LIGHT_ADVICE_VRAM_GB


def vdn_base_fits(base_file, vram_gb, light_requested=False, light_reason=''):
    return True, ''


def vdn_launch_guard(built, *, light=False, vram_gb=None):
    return None


# Compatibility with the original public SDK exports.
_LEGACY_EXPORTS = {'ACCELERATIONS': ('app.services.video_test_studio', 'ACCELERATIONS'),
 'ACCEL_IDS': ('app.services.video_test_studio', 'ACCEL_IDS'),
 'BASE_EROS': ('app.services.video_test_studio', 'BASE_EROS'),
 'BASE_OFFICIAL': ('app.services.video_test_studio', 'BASE_OFFICIAL'),
 'DARETIES_LORA': ('app.services.video_test_studio', 'DARETIES_LORA'),
 'DEFAULT_STEPS': ('app.services.video_test_studio', 'DEFAULT_STEPS'),
 'DYNAMIC_VRAM_FORCE': ('app.services.video_test_studio',
                        '_DYNAMIC_VRAM_FORCE'),
 'DYNAMIC_VRAM_OFF_MODES': ('app.services.video_test_studio',
                            '_DYNAMIC_VRAM_OFF_MODES'),
 'DYNAMIC_VRAM_SWITCH': ('app.services.video_test_studio',
                         '_DYNAMIC_VRAM_SWITCH'),
 'FAST_DISK_FLAG': ('app.services.video_test_studio', '_FAST_DISK_FLAG'),
 'FAST_DISK_MIN_COMFYUI': ('app.services.video_test_studio',
                           'FAST_DISK_MIN_COMFYUI'),
 'FAST_DISK_RAM_FLOOR_GB': ('app.services.video_test_studio',
                            'FAST_DISK_RAM_FLOOR_GB'),
 'FRAMES_DEFAULT': ('app.services.video_test_studio', 'FRAMES_DEFAULT'),
 'FRAMES_MAX': ('app.services.video_test_studio', 'FRAMES_MAX'),
 'FRAMES_MIN': ('app.services.video_test_studio', 'FRAMES_MIN'),
 'H3_HOST_RAM_GB': ('app.services.video_test_studio', 'H3_HOST_RAM_GB'),
 'HIGH_RAM_FLAG': ('app.services.video_test_studio', '_HIGH_RAM_FLAG'),
 'LORA_EXT': ('app.services.video_test_studio', 'LORA_EXT'),
 'LORA_SUBDIR': ('app.services.video_test_studio', 'LORA_SUBDIR'),
 'MP_DEFAULT': ('app.services.video_test_studio', 'MP_DEFAULT'),
 'MP_MAX': ('app.services.video_test_studio', 'MP_MAX'),
 'MP_MIN': ('app.services.video_test_studio', 'MP_MIN'),
 'N_ACCEL_LORA': ('app.services.video_test_studio', 'N_ACCEL_LORA'),
 'N_COND': ('app.services.video_test_studio', 'N_COND'),
 'N_CREATE_VIDEO': ('app.services.video_test_studio', 'N_CREATE_VIDEO'),
 'N_DECODE_AUDIO': ('app.services.video_test_studio', 'N_DECODE_AUDIO'),
 'N_DECODE_VIDEO': ('app.services.video_test_studio', 'N_DECODE_VIDEO'),
 'N_GUIDER': ('app.services.video_test_studio', 'N_GUIDER'),
 'N_LOAD_IMAGE': ('app.services.video_test_studio', 'N_LOAD_IMAGE'),
 'N_NOISE': ('app.services.video_test_studio', 'N_NOISE'),
 'N_SAGE': ('app.services.video_test_studio', 'N_SAGE'),
 'N_SAMPLER': ('app.services.video_test_studio', 'N_SAMPLER'),
 'N_SAMPLER_SELECT': ('app.services.video_test_studio', 'N_SAMPLER_SELECT'),
 'N_SAVE': ('app.services.video_test_studio', 'N_SAVE'),
 'N_SCALE': ('app.services.video_test_studio', 'N_SCALE'),
 'N_SCHEDULER': ('app.services.video_test_studio', 'N_SCHEDULER'),
 'N_SHIFT': ('app.services.video_test_studio', 'N_SHIFT'),
 'N_SIZE': ('app.services.video_test_studio', 'N_SIZE'),
 'N_SPARSE': ('app.services.video_test_studio', 'N_SPARSE'),
 'N_TEST_LORA': ('app.services.video_test_studio', 'N_TEST_LORA'),
 'N_TURBO_LORA': ('app.services.video_test_studio', 'N_TURBO_LORA'),
 'N_TURBO_SAMPLER': ('app.services.video_test_studio', 'N_TURBO_SAMPLER'),
 'N_UNET': ('app.services.video_test_studio', 'N_UNET'),
 'N_UPSCALE': ('app.services.video_test_studio', 'N_UPSCALE'),
 'N_UPSCALE_PARAMS': ('app.services.video_test_studio', 'N_UPSCALE_PARAMS'),
 'N_UPSCALE_SPLIT': ('app.services.video_test_studio', 'N_UPSCALE_SPLIT'),
 'N_VAE_VIDEO': ('app.services.video_test_studio', 'N_VAE_VIDEO'),
 'OPTIONAL_WEIGHTS': ('app.services.video_test_studio', 'OPTIONAL_WEIGHTS'),
 'OPTION_NODE_PACKS': ('app.services.video_test_studio', 'OPTION_NODE_PACKS'),
 'PARASYTE_LORA': ('app.services.video_test_studio', 'PARASYTE_LORA'),
 'REQUIRED_WEIGHTS': ('app.services.video_test_studio', 'REQUIRED_WEIGHTS'),
 'SAGE_CLASS': ('app.services.video_test_studio', 'SAGE_CLASS'),
 'SAGE_PACK': ('app.services.video_test_studio', 'SAGE_PACK'),
 'SPARSE_BACKEND': ('app.services.video_test_studio', 'SPARSE_BACKEND'),
 'SPARSE_MODES': ('app.services.video_test_studio', 'SPARSE_MODES'),
 'SPARSE_PRESETS': ('app.services.video_test_studio', 'SPARSE_PRESETS'),
 'TARGET_KEY': ('app.services.video_test_studio', 'TARGET_KEY'),
 'TURBO_LORA': ('app.services.video_test_studio', 'TURBO_LORA'),
 'TURBO_STEPS': ('app.services.video_test_studio', 'TURBO_STEPS'),
 'UNFETCHABLE': ('app.services.video_test_studio', 'UNFETCHABLE'),
 'UPSCALE_MODEL': ('app.services.video_test_studio', 'UPSCALE_MODEL'),
 'VERSION_RE': ('app.services.video_test_studio', '_VERSION_RE'),
 'WORKFLOW_FILENAME': ('app.services.video_test_studio', 'WORKFLOW_FILENAME'),
 'accel_spec': ('app.services.video_test_studio', 'accel_spec'),
 'accelerations_status': ('app.services.video_test_studio',
                          'accelerations_status'),
 'build_workflow': ('app.services.video_test_studio', 'build_workflow'),
 'clamp_megapixels': ('app.services.video_test_studio', 'clamp_megapixels'),
 'comfyui_launch_facts': ('app.services.video_test_studio',
                          'comfyui_launch_facts'),
 'deploy_checkpoint': ('app.services.video_test_studio', 'deploy_checkpoint'),
 'deploy_file': ('app.services.video_test_studio', 'deploy_file'),
 'deployed_loras': ('app.services.video_test_studio', 'deployed_loras'),
 'drop_sage': ('app.services.video_test_studio', '_drop_sage'),
 'eros_on_disk': ('app.services.video_test_studio', 'eros_on_disk'),
 'graft_latent_upscale': ('app.services.video_test_studio',
                          '_graft_latent_upscale'),
 'graft_sparse': ('app.services.video_test_studio', '_graft_sparse'),
 'graft_stock_accel': ('app.services.video_test_studio', '_graft_stock_accel'),
 'graft_test_lora': ('app.services.video_test_studio', '_graft_test_lora'),
 'graft_turbo': ('app.services.video_test_studio', '_graft_turbo'),
 'import_external_lora': ('app.services.video_test_studio',
                          'import_external_lora'),
 'insert_into_chain': ('app.services.video_test_studio', '_insert_into_chain'),
 'knows_fast_disk': ('app.services.video_test_studio', 'knows_fast_disk'),
 'last_frame_command': ('app.services.video_test_studio', 'last_frame_command'),
 'launch_advice': ('app.services.video_test_studio', 'launch_advice'),
 'load_base_workflow': ('app.services.video_test_studio', 'load_base_workflow'),
 'loras_write_dir': ('app.services.video_test_studio', '_loras_write_dir'),
 'make_t2v': ('app.services.video_test_studio', '_make_t2v'),
 'missing_weights': ('app.services.video_test_studio', 'missing_weights'),
 'model_readers': ('app.services.video_test_studio', '_model_readers'),
 'new_prefix': ('app.services.video_test_studio', 'new_prefix'),
 'normalise_accel': ('app.services.video_test_studio', 'normalise_accel'),
 'normalise_sparse': ('app.services.video_test_studio', 'normalise_sparse'),
 'option_availability': ('app.services.video_test_studio',
                         'option_availability'),
 'preflight': ('app.services.video_test_studio', 'preflight'),
 'profile': ('app.services.video_test_studio', '_profile'),
 'registered_classes': ('app.services.video_test_studio', 'registered_classes'),
 'sage_available': ('app.services.video_test_studio', 'sage_available'),
 'snap_frames': ('app.services.video_test_studio', 'snap_frames'),
 'studio_ready': ('app.services.video_test_studio', 'studio_ready'),
 'trained_loras': ('app.services.video_test_studio', 'trained_loras'),
 'undeploy_lora': ('app.services.video_test_studio', 'undeploy_lora'),
 'video_runs': ('app.services.video_test_studio', '_video_runs'),
 'weight_present': ('app.services.video_test_studio', '_weight_present'),
 'workflow_path': ('app.services.video_test_studio', 'workflow_path')}

def __getattr__(name):
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module
    module, symbol = _LEGACY_EXPORTS[name]
    return getattr(import_module(module), symbol)

__all__ = list(dict.fromkeys([*__all__, *_LEGACY_EXPORTS]))
