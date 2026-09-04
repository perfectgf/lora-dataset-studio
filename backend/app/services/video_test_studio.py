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

import json
import os
import subprocess
import random
import re
import uuid

from . import video_targets

# ── The base graph ───────────────────────────────────────────────────────────
# 18 nodes: loaders (6/13/11/24), the image branch (114 → 119 → 120), the H3
# conditioner (104), sampling (15/16/17/9/14), decode (10/23) and mux (91/92),
# plus SageAttention (600) sitting between the UNET and everything that reads
# the model.
WORKFLOW_FILENAME = 'minimax_h3_i2v.json'

# Node ids of the base graph that the grafts read or rewrite. Named because
# `workflow["9"]` five hundred lines from here says nothing about which node
# just had its step count changed.
N_UNET = '6'            # UNETLoader — the base, swapped by the 10Eros option
N_CLIP = '13'           # CLIPLoader — Qwen3-VL text encoder
N_VAE_VIDEO = '11'
N_VAE_AUDIO = '24'
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

# Grafted ids. The ranges are deliberately spread out: a collision would
# overwrite a node with no message at all, in a dict where every key is a
# string that looks like every other key.
N_TURBO_LORA = '601'
N_TURBO_SAMPLER = '603'
N_ACCEL_LORA = '602'    # a stock-loader acceleration LoRA (Parasyte, DARE-TIES)
N_SHIFT = '604'         # MiniMaxH3SigmaShift — the shift those LoRAs were tuned at
N_TEST_LORA = '610'     # the LoRA under test — 606-609 are the style LoRAs upstream
N_UPSCALE = '800'
N_UPSCALE_PARAMS = '801'
N_UPSCALE_SPLIT = '802'
N_SPARSE = '810'

# ── The weights this graph names ─────────────────────────────────────────────
# The OFFICIAL base. Kept as a constant rather than read off the JSON because
# the 10Eros swap has to be able to say which base a run actually used.
BASE_OFFICIAL = 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'

# 🔥 10Eros-Max — a THIRD-PARTY finetune of H3 (cicalooo), int8 convrot, in the
# `skip_edges` variant (blocks 0/1/48/49 kept in BF16, which its own SKIP_EDGES
# note calls "the safer starting point when comparing quality").
#
# Never elected in silence: the option is off by default and the official base
# stays the graph's. It also imposes its own faces, which is exactly wrong when
# the thing being tested is whether YOUR LoRA reproduces an identity — the UI
# says so, and the identity recipe measured on this pipeline uses the official
# base for that reason.
BASE_EROS = ('10Eros_Max_h3_fl2va_beta2_pruned_int8_convrot_skip_edges'
             '.safetensors')

# ⚡ The 6-step distillation LoRA (larryvrh v4, step 600 EMA — the top row of the
# multimodalart H3 acceleration arena at 6 steps), applied through its OWN node, not
# a standard loader. Both halves of that sentence were paid for:
#
#   * the file uses H3's bare key naming (`blocks.0.attn…`) plus 102 adaln keys.
#     `MiniMaxH3TurboLoRA` re-prefixes the keys AND re-injects the adaln that the
#     pruned base collapsed into a curve. Hand the same file to a standard
#     loader and the adaln half is simply missing.
#   * a standard loader MERGES into the weights (`add_patches`). On an int8
#     base that pushes modules into lowvram patches and re-quantises on every
#     forward pass — measured elsewhere at step 1 unfinished after 6 min 37,
#     against ~9 s/step without. The dedicated node runs the LoRA alongside the
#     base instead (bypass), immune to where the weights happen to sit.
TURBO_LORA = 'minimax_h3_turbo_v4_step600_ema.safetensors'

# Six, not four. The model card is explicit — "4 steps is the recommended
# MINIMUM; 4-8 is the useful range. 6-8 steps look noticeably better than 4" —
# and at 4 steps with fast motion this checkpoint trails ghosting.
TURBO_STEPS = 6

# ⚡ The accelerations the Render panel offers: the top three rows of the
# multimodalart MiniMax-H3 acceleration arena (human preference Elo, ~7 400
# votes per task, 95 % intervals about ±26 — so the three are statistical
# ties), every one of them at 6 steps and about 4× the 28-step reference.
# `turbo` is larryvrh's LoRA through its OWN nodes (a LoRA node and the
# double-clock sampler, see _graft_turbo). The other two are ordinary LoRAs
# for the stock loader, run on the stock euler sampler at the sigma shift
# their cards and the arena ran them at (video 8, audio 3; the base's own
# grid is 12/3). Strengths are the arena's verified settings, not 1.0: the
# Parasyte file's alpha convention wants 4-5, the merge's card says 0.6-0.8.
PARASYTE_LORA = 'H3-PK-Parasyte-Turbo.safetensors'
DARETIES_LORA = 'minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro.safetensors'
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
)
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

# Without turbo the base is not distilled and needs a real schedule.
DEFAULT_STEPS = 20

# ⚡ Sparse attention (H3-Optimizations, Zironic). Both levels are the AUTHOR's,
# not ours: `default` is the node's own defaults, `conservative` raises the
# budget and holds the schedule's edges denser — the exact shape of the node's
# "Denser Early/Late" toggle applied to a higher budget.
#
# `max` carries the same numbers as `default` and differs only in WHERE it is
# applied (see `_graft_sparse`): it is the one level that lets the sparse graph
# touch the base sampling pass as well.
SPARSE_PRESETS = {
    'default': {'video_budget': 0.3, 'early_steps': 2, 'early_kv': 0.5,
                'late_steps': 2, 'late_kv': 0.5},
    'conservative': {'video_budget': 0.5, 'early_steps': 2, 'early_kv': 0.8,
                     'late_steps': 2, 'late_kv': 0.8},
    'max': {'video_budget': 0.3, 'early_steps': 2, 'early_kv': 0.5,
            'late_steps': 2, 'late_kv': 0.5},
}
SPARSE_MODES = ('', 'default', 'conservative', 'max')

# Backend left at the node's embedded default. FROST BF16 exists on SM89 but an
# explicitly named backend FAILS when it is unavailable, and this graph is also
# meant to run on a rented GPU nobody inspected.
SPARSE_BACKEND = 'Kitchen INT8'

# 🔬 The latent upscaler. H3 emits an INTERLEAVED latent — 24 video channels and
# 32 audio channels in one tensor — which image upscalers cannot read; they have
# to decode to pixels, enlarge, and re-encode, which is where the re-muxed audio
# and the tile seams come from. This one enlarges inside the model's own domain
# and the audio passes through untouched.
UPSCALE_MODEL = 'minimax_h3_latent_upscaler_3d_bf16.safetensors'

# The profile key in the shared target catalogue. The studio takes fps and the
# legal clip lengths from there rather than restating them, so a clip generated
# here and a clip cut for training cannot drift apart.
TARGET_KEY = 'minimax_h3'


def _profile() -> dict:
    """The shared catalogue entry for H3, or {} on an unknown key.

    Everything the studio takes from the catalogue goes through here: fps, the
    default clip length, the legal-length rule. Restating any of them locally is
    how a clip generated in the studio and a clip cut for training start
    disagreeing about what H3 accepts.
    """
    return video_targets.get(TARGET_KEY) or {}

# H3's VAE packs 17 pixel frames per chunk, so a legal length is ≡ 5 (mod 17).
_FRAME_MOD, _FRAME_OFFSET = 17, 5
# Generation reaches further than training does. The catalogue stops at 209
# because that is where TRAINING clip lengths stop being useful; the model
# itself renders to ~15 s at 24 fps, and refusing that here would cap the studio
# at 8.7 s for a reason that has nothing to do with the studio.
FRAMES_MIN, FRAMES_MAX = 22, 362

# And the default is the STUDIO's, not the catalogue's. The catalogue's
# `frame_default` is 39 because that is how long a TRAINING clip should be; the
# same preset carries 107 on its preview line, and reading the wrong one of those
# two numbers has already cost this project a wrong default once. A test clip
# wants enough motion to judge (39 frames is 1.6 s — barely a gesture) without
# paying the ~2.7x per-step cost of 107, so it sits between them.
FRAMES_DEFAULT = 56

# Resolution, in megapixels at node 119. The ceiling is the MODEL's, not a
# card's: the sweet spot for faces sits near 1.0 MP and the machine that runs
# the job decides what it can hold.
MP_MIN, MP_MAX, MP_DEFAULT = 0.1, 2.0, 0.3

# Where the studio deploys a trained video LoRA so ComfyUI can list it. `h3` is
# the subfolder the H3 LoRAs of this ecosystem already live in; `lds` keeps the
# app's own deployments distinct from files the user put there by hand.
LORA_SUBDIR = os.path.join('h3', 'lds')


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


def workflow_path():
    """Absolute path of the embedded H3 graph."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'workflows', WORKFLOW_FILENAME)


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


# ── The build ────────────────────────────────────────────────────────────────

def build_workflow(*, prompt, mode='i2v', image=None, seed=None, steps=None,
                   frames=None, megapixels=MP_DEFAULT, aspect='auto',
                   fps=None, lora=None, lora_strength=1.0, turbo=False,
                   accel=None, eros=False, eros_on_disk=False, sparse='',
                   latent_upscale=False, source_ratio=None, sage=True,
                   filename_prefix='lds_video_test') -> dict:
    """One MiniMax H3 clip, as a ComfyUI graph.

    `accel` names one of ACCELERATIONS ('' or None for the dense base); `turbo`
    is the older boolean and means `accel='turbo'` when `accel` is not given.

    `eros_on_disk` is passed IN rather than probed here so the whole option
    matrix stays testable without a 21 GB file — and so the fail-open path (ask
    for 10Eros, get the official base plus a note) is exercised by a test rather
    than by a user whose download had not finished.

    `source_ratio` (width/height of the picked image) only matters when the
    latent upscale is armed and there is an image to measure; without it the
    upscale keeps the node's own target size.

    `sage=False` takes SageAttention out. It is a speed patch from a pack the
    installer deliberately does not fetch (it declares pip dependencies, and
    this app never pip-installs a third-party requirements file into someone
    else's environment), so the graph has to be able to do without it — the
    caller passes what the target ComfyUI actually registers.
    """
    wf = load_base_workflow()
    notes = []

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
        notes.append('sage: absent from this ComfyUI — running without it')

    if str(mode or 'i2v').lower() == 't2v':
        _make_t2v(wf, megapixels=megapixels, aspect=aspect)
        notes.append('t2v: image branch removed')
    else:
        wf[N_LOAD_IMAGE]['inputs']['image'] = str(image or '')

    # ── Order of the grafts is the contract ──────────────────────────────────
    # base swap → turbo → tested LoRA → sparse → upscale.
    #
    # The base swap goes first so every log line below can name the base that
    # actually ran. The sparse graft goes before the upscale because the upscale
    # reads whatever the model chain currently ends in — reversing them gives
    # the upscale a model that has not been patched yet, with no error anywhere.
    if eros:
        if eros_on_disk:
            wf[N_UNET]['inputs']['unet_name'] = BASE_EROS
            notes.append('base: 10Eros-Max (third-party finetune)')
        else:
            # Fail OPEN to the official base. The box can legitimately be
            # ticked while 21.7 GB are still downloading, and a graph ComfyUI
            # refuses at validation ("Value not in list: unet_name") is a worse
            # answer than a clip on the official base plus a line saying so.
            notes.append('base: 10Eros requested but absent — official base used')

    accel = normalise_accel(accel, turbo)
    if accel == 'turbo':
        _graft_turbo(wf)
    elif accel:
        _graft_stock_accel(wf, accel_spec(accel))
    if steps is not None:
        # An explicit step count always wins, including over the turbo default:
        # the panel showing 6 while the graph runs 4 is the exact failure this
        # pipeline already paid for once.
        wf[N_SCHEDULER]['inputs']['steps'] = max(4, min(40, int(steps)))
    # The note reads the count AFTER the override: a log that said "steps=6"
    # while the graph ran the user's 4 was seen on the first real launch.
    if accel == 'turbo':
        notes.append(f'turbo: larryvrh distillation, steps={wf[N_SCHEDULER]["inputs"]["steps"]}')
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

    return {'workflow': wf, 'seed': int(seed), 'frames': frames,
            'megapixels': megapixels, 'notes': notes,
            'base': wf[N_UNET]['inputs']['unet_name'],
            'steps': wf[N_SCHEDULER]['inputs']['steps'],
            'accel': accel}


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


def _make_t2v(wf, *, megapixels, aspect):
    """Text-to-video: unplug the image branch.

    `first_frame` is OPTIONAL on `MiniMaxH3ImageToVideo`, so t2v is the same
    node with one input removed — but the graph then has nothing to measure the
    canvas from, which is why width/height become explicit here.
    """
    for nid in (N_LOAD_IMAGE, N_SCALE, N_SIZE):
        wf.pop(nid, None)
    wf[N_COND]['inputs'].pop('first_frame', None)
    ratio = {'landscape': 16 / 9, 'portrait': 9 / 16,
             'square': 1.0}.get(str(aspect or '').strip().lower(), 16 / 9)
    height = int(round((megapixels * 1_000_000 / ratio) ** 0.5 / 32) * 32)
    width = int(round(height * ratio / 32) * 32)
    wf[N_COND]['inputs']['width'] = max(32, width)
    wf[N_COND]['inputs']['height'] = max(32, height)


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


# ═══════════════════════════════════════════════════════════════════════════
# The runtime half: what is on disk, what ComfyUI can see, and the queue.
# Everything above this line is pure; everything below touches the world.
# ═══════════════════════════════════════════════════════════════════════════

import logging
import shutil

logger = logging.getLogger(__name__)


def _loras_write_dir():
    """Where the studio DEPLOYS a trained video LoRA, created on demand.

    ComfyUI's loader lists files under its own roots and nothing else, so a
    checkpoint sitting in the app's checkpoint store is invisible to it however
    valid it is. Deployment is a copy into the FIRST loras root — the one
    ComfyUI writes to and lists first.
    """
    from . import comfy_model_paths
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
    from . import comfy_model_paths
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
    from ..models import CloudTrainingRun
    return (CloudTrainingRun.query
            .filter(CloudTrainingRun.dataset_table == 'video_dataset')
            .order_by(CloudTrainingRun.id.desc())
            .all())


def trained_loras() -> list:
    """The LoRAs this app has trained on video, and whether each is deployed.

    One entry per checkpoint file, newest run first, carrying the run id and the
    filename so the caller can ask for a deploy. `deployed_as` is the
    LoraLoader-form name when a copy is already in ComfyUI, else None — the
    picker uses it to offer "Test" instead of "Deploy & test".

    A run whose checkpoint directory is gone (the video lane grew a DELETE)
    contributes nothing, which is the honest reading of an empty directory.
    """
    from . import cloud_training as ct
    deployed = {os.path.basename(e['filename']).lower(): e['filename']
                for e in deployed_loras()}
    out = []
    for run in _video_runs():
        try:
            files = ct.run_checkpoint_files(run)
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
    from ..models import CloudTrainingRun
    from . import cloud_training as ct
    run = CloudTrainingRun.query.filter_by(id=int(run_id)).first()
    if run is None or run.dataset_table != 'video_dataset':
        raise ValueError('video training run not found')
    src = ct.run_checkpoint_path(run, filename)
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
    from . import comfy_model_paths, trash
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


# A LoRA is one safetensors file. The extension is not decoration here: it is
# what ComfyUI's LoraLoader reads, and copying a .ckpt or a .pt into the folder
# would put an entry in the picker that fails at generation time.
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
    from .lora_test_studio import _is_unsafe_external_lora_name
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
    from . import comfy_model_paths
    for kind in ('diffusion_models', 'unet'):
        for root in comfy_model_paths.search_roots(kind):
            if os.path.isfile(os.path.join(str(root), BASE_EROS)):
                return True
    return False


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
    from . import lora_test_studio as lts
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
    lts.preflight_family('h3video', [workflow])


def clips_dir(create=True):
    """Where finished test clips live, away from ComfyUI's output directory.

    Same reasoning as the image studio's per-dataset folder: the clip has to
    survive whatever ComfyUI does with its own outputs, and be served by the app
    under a path it controls.
    """
    from .. import config as cfg
    root = cfg.data_dir() / 'video_tests'
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def enqueue_clip(user_id, *, prompt, mode='i2v', image=None, lora=None,
                 lora_strength=1.0, run_id=None, dataset_id=None, seed=None,
                 steps=None, frames=None, megapixels=MP_DEFAULT, aspect='auto',
                 turbo=False, accel=None, eros=False, sparse='', latent_upscale=False,
                 source_ratio=None, skip_preflight=False, continues=None) -> dict:
    """Build the graph, record the clip, queue the job — in that order.

    The row is written BEFORE the queue insert and in the SAME transaction, so a
    job can never exist without the row that explains what it was: the queue
    monitor's completion callback resolves by `job_id`, and a missing row means
    a finished clip nobody can attribute.
    """
    from ..extensions import db
    from ..job_queue import queue_manager
    from ..models import VideoTestClip

    # Keep the T2V canvas choice with the clip so Reuse cannot inherit a later
    # render's format. I2V follows its source image instead of a canvas preset.
    mode = 't2v' if str(mode or 'i2v').lower() == 't2v' else 'i2v'
    aspect = str(aspect or 'auto').strip().lower()
    if mode == 'i2v' or aspect not in ('auto', 'portrait', 'landscape', 'square'):
        aspect = 'auto'

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
        if parent is None or parent.status != 'done' or not parent.filename:
            raise ValueError('the clip to continue has not finished rendering')
    classes = registered_classes()
    built = build_workflow(
        prompt=prompt, mode=mode, image=image, seed=seed, steps=steps,
        frames=frames, megapixels=megapixels, aspect=aspect, lora=lora,
        lora_strength=lora_strength, turbo=turbo, accel=accel, eros=eros,
        eros_on_disk=eros_on_disk() if eros else False, sparse=sparse,
        latent_upscale=latent_upscale, source_ratio=source_ratio,
        sage=sage_available(classes), filename_prefix=new_prefix(user_id))
    if not skip_preflight:
        preflight(built['workflow'])

    job_id = str(uuid.uuid4())
    clip = VideoTestClip(
        run_id=run_id, dataset_id=dataset_id, job_id=job_id, status='pending',
        prompt=prompt, mode=mode, aspect=aspect,
        source_image=image, seed=built['seed'], steps=built['steps'],
        frames=built['frames'], megapixels=built['megapixels'],
        fps=float(_profile().get('fps') or 24.0), base_model=built['base'],
        lora=lora, lora_strength=float(lora_strength) if lora else None,
        turbo=(built['accel'] == 'turbo'), accel=(built['accel'] or None),
        sparse=normalise_sparse(sparse), latent_upscale=bool(latent_upscale),
        continues_of=(parent.id if parent else None))
    db.session.add(clip)
    db.session.flush()          # mint the id inside the same transaction
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=built['workflow'], prompt=prompt or '',
                          job_id=job_id,
                          metadata={'model_name': 'video_lora_test',
                                    'is_video_test': True,
                                    'clip_id': clip.id},
                          commit=False)
    db.session.commit()
    logger.info('video studio: queued clip %s (%s)', clip.id,
                ', '.join(built['notes']) or 'no options')
    return {'clip_id': clip.id, 'job_id': job_id, 'seed': built['seed'],
            'frames': built['frames'], 'steps': built['steps'],
            'notes': built['notes']}


# ↗ VFI. The checkpoint, the multiplier and every dial below are read from the
# maintainer's image generator (workflows/video-generation/vfi.json) rather than
# chosen here: the two apps drive the same ComfyUI, and a clip smoothed in one
# should be the clip smoothed in the other. rife49 is what that graph loads;
# `fast_mode` and `ensemble` are its settings; the cache is cleared every 16
# frames, which is what keeps a 200-frame clip inside VRAM.
VFI_CKPT = 'rife49.pth'
VFI_MULTIPLIER = 2
VFI_CLEAR_CACHE_EVERY = 16
# h264 at crf 19, yuv420p — the same container the generator writes, so the
# smoothed clip plays anywhere the original did.
VFI_CRF = 19

N_VFI_LOAD, N_VFI_RIFE, N_VFI_SAVE = 'v1', 'v2', 'v3'


def build_vfi_workflow(*, video_path, fps, multiplier=VFI_MULTIPLIER,
                       filename_prefix=None) -> dict:
    """The interpolation graph for ONE finished clip. Pure — no disk, no queue.

    The source is given as an ABSOLUTE path: the clip lives in this app's own
    folder (clips_dir), never in ComfyUI's output, and VHS_LoadVideoPath takes a
    path rather than a name precisely so a file outside that tree can be read.
    The output rate is the source's times the multiplier, which is what makes
    this a SMOOTHING rather than a slow motion — the clip keeps its duration and
    gains frames.
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
    from ..extensions import db
    from ..job_queue import queue_manager
    from ..models import VideoTestClip

    src = VideoTestClip.query.filter_by(id=int(clip_id)).first()
    if src is None:
        raise ValueError('clip not found')
    if src.status != 'done' or not src.filename:
        raise ValueError('that clip has not finished rendering yet')
    path = os.path.join(str(clips_dir()), os.path.basename(src.filename))
    if not os.path.isfile(path):
        raise ValueError('that clip is no longer on disk')

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
        status='pending', prompt=src.prompt, mode=src.mode,
        aspect=src.aspect or 'auto', accel=src.accel,
        source_image=src.source_image, seed=src.seed, steps=src.steps,
        # The frame COUNT grows with the rate, so the clip lasts exactly as
        # long — RIFE inserts between frames, it does not slow anything down.
        frames=(src.frames or 0) * mult if src.frames else None,
        megapixels=src.megapixels, fps=float(src.fps or 24) * mult,
        base_model=src.base_model, lora=src.lora,
        lora_strength=src.lora_strength, turbo=bool(src.turbo),
        sparse=src.sparse, latent_upscale=bool(src.latent_upscale),
        vfi_of=src.id)
    db.session.add(clip)
    db.session.flush()
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


def link_completed_clip(job_id, filename, failed=False, reason=None):
    """Attach a finished ComfyUI job to its clip row.

    Runs in the queue monitor thread, whose session may hold a stale read
    snapshot — hence the rollback-and-re-read before concluding the row is
    absent, exactly as the image studio's callback does.

    The mp4 comes back under the history's `images` key like any other output
    (that is how `SaveVideo` reports), so nothing in the polling path had to
    change for video.
    """
    from ..extensions import db
    from ..models import VideoTestClip
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
    clip.render_seconds = _render_seconds(job_id)
    if failed:
        clip.status = 'failed'
        clip.error = (reason or 'Generation failed (see the server log in '
                                'Settings for the ComfyUI error).')
        db.session.commit()
        return
    clip.filename = filename
    clip.status = 'done'
    _bring_clip_home(filename)
    continues = getattr(clip, 'continues_of', None)
    clip_id = clip.id
    # Committed BEFORE any join: from here the row is a valid render (the
    # part), and the encode below runs with no write transaction open. It ran
    # inside this one once — measured: every other writer got "database is
    # locked" for the length of ffmpeg (up to its 600 s timeout).
    db.session.commit()
    if continues:
        # ⏭ The part becomes the whole: parent, then this render.
        _join_continuation(clip_id)


# ── ⏭ Continue: the last frame as the next start frame, the clips joined ──

def _run_ffmpeg(cmd, timeout=600):
    """The app's subprocess convention for ffmpeg (video_bank_service has the
    same): no console window in the frozen Windows build, utf-8 stderr with
    replacement, and a timeout."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=timeout,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def last_frame_command(ffmpeg, src, dst):
    """The last frame of `src` as a PNG: seek into the last second and take the
    first frame of the REVERSED tail — the true last frame whatever the exact
    duration, where `-sseof` alone can land past it and write nothing."""
    return [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-sseof', '-1', '-i', src,
            '-vf', 'reverse', '-frames:v', '1', '-update', '1', dst]


def last_frame_png(clip_id) -> str:
    """The clip's last frame as a PNG next to its mp4 (`clip_<id>_last.png`),
    extracted once and kept; the picture the next clip starts from."""
    from ..models import VideoTestClip
    from . import ffmpeg_tools
    clip = VideoTestClip.query.filter_by(id=int(clip_id)).first()
    if clip is None:
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
    # Concurrent previews and continuations must never see a partial PNG. Keep
    # the previous complete image until ffmpeg succeeds, then publish in place.
    tmp = os.path.join(root, f'.clip_{int(clip_id)}_last.{uuid.uuid4().hex[:8]}.png')
    try:
        r = _run_ffmpeg(last_frame_command(ffmpeg, src, tmp), timeout=120)
        if r.returncode != 0 or not os.path.isfile(tmp):
            raise ValueError(f'the last frame could not be read: {(r.stderr or "")[-300:]}')
        os.replace(tmp, dst)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return dst


def _probe_media(ffmpeg, path) -> dict:
    """What ffmpeg sees in a file, read off its own `-i` banner: whether it
    has a sound track, and how long it plays. The join needs both — a smoothed
    clip has no sound (its VHS_VideoCombine is given pictures only), and a
    silent side joined to a sounding one must be padded with silence of ITS
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
    from ..extensions import db
    from ..models import VideoTestClip
    from . import ffmpeg_tools

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
    from ..models import ImageGenerationQueue
    job = ImageGenerationQueue.query.filter_by(job_id=job_id).first()
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
    from . import lora_test_studio as lts
    from ..utils import comfy_fs
    dst = os.path.join(str(clips_dir()), filename)
    if os.path.exists(dst):
        return
    out_dir = lts._comfy_output_dir()
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
        from ..utils.comfyui import fetch_output_image_bytes
        data = fetch_output_image_bytes(filename)
        if data:
            with open(dst, 'wb') as fh:
                fh.write(data)
        else:
            logger.warning('video studio: %s is not on disk and /view returned '
                           'nothing — the clip stays with ComfyUI', filename)
    except OSError:
        logger.exception('video studio: could not bring %s home', filename)


# ═══════════════════════════════════════════════════════════════════════════
# What a machine needs before this lane can render anything — and what it is
# still missing right now.
#
# The Setup screen turns each key below into a button, so these names are not
# labels: they are `setup_installer.INSTALL_ACTIONS` entries. A name that does
# not exist there is a dead end by construction, which a contract test catches.
# ═══════════════════════════════════════════════════════════════════════════

# (setup action, ComfyUI subfolders it may live in, filename, why it is needed).
# REQUIRED first: a message built from this list reads worst-first.
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

# Optional weights: each one belongs to ONE checkbox, and its absence disables
# that checkbox rather than the lane.
OPTIONAL_WEIGHTS = (
    ('h3_turbo_lora', ('loras',), TURBO_LORA, 'turbo'),
    # ⚡ The other two accelerations of the Render panel (arena rows 2 and 3).
    ('h3_parasyte_lora', ('loras',), PARASYTE_LORA, 'parasyte'),
    ('h3_dareties_lora', ('loras',), DARETIES_LORA, 'dareties'),
    # ⚠️ No setup action for this one, on purpose — see UNFETCHABLE below.
    (None, ('latent_upscale_models',), UPSCALE_MODEL, 'latent_upscale'),
    # Nor for the third-party base: it is somebody else's finetune, it is
    # 20 GB, and nothing in the app needs it. Present on disk, the box works.
    (None, ('diffusion_models', 'unet'), BASE_EROS, 'eros'),
)

# The weights the app will NOT fetch for you, and why — stated here rather than
# discovered as a silent gap:
#   * the latent upscaler has no authoritative home. The node's own README names
#     the FOLDER it reads and no download; the copies on the model hub are
#     community re-uploads whose provenance cannot be checked. Shipping an
#     installer button that pulls an unverifiable 0.6 GB file into somebody
#     else's ComfyUI is not a thing this app should do, so the preflight names
#     the folder instead and the checkbox says the file is missing.
#   * the 10Eros base is a third-party finetune, opt-in by design.
UNFETCHABLE = {
    UPSCALE_MODEL: 'models/latent_upscale_models/',
    BASE_EROS: 'models/diffusion_models/',
}

# Custom-node packs, per option. The BASE graph is deliberately absent from this
# table: it runs on a stock ComfyUI, which is what lets a new user render their
# first clip with nothing but the weights.
#
# SageAttention is absent too, and that is the same decision seen from the other
# side: it is a speed patch, its pack declares pip dependencies, and this app
# never pip-installs a third-party requirements file — so it is used when the
# target ComfyUI already has it and skipped when it does not.
# WHY THESE ARE LINKS AND NOT BUTTONS (maintainer's call, 2026-08-31)
# "Downloading models is fine, but we do not take responsibility for breaking a
# ComfyUI install." A weight is an inert file; a custom node is code ComfyUI
# imports at startup, and one bad import takes the server down for every other
# lane. So the app names the pack, links it, and lets the user install it on the
# ComfyUI side — where they can see what they are adding.
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

# SageAttention. Not an option and not a checkbox: a speed patch the graph keeps
# when the target ComfyUI has it and drops when it does not. Linked for the same
# reason as the three above, and installed the same way — by the user.
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
    from . import comfy_model_paths
    for sub in subfolders:
        try:
            roots = comfy_model_paths.search_roots(sub)
        except Exception:
            roots = []
        for root in roots:
            if os.path.isfile(os.path.join(str(root), filename)):
                return True
            try:
                names = {n.lower() for n in os.listdir(str(root))}
            except OSError:
                continue
            if filename.lower() in names:
                return True
    return False


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
        if not _weight_present(subs, filename):
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
    from ..utils.comfyui import fetch_object_info_classes
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


def accelerations_status(classes=None) -> list:
    """The Render panel's acceleration choices, resolved against THIS machine:
    the weight on disk and, for larryvrh's, its node pack too. `available` is
    what the panel can offer; `action` is the Setup button that fetches the
    weight; `pack` travels for the one choice that needs code installed."""
    packs = option_availability(classes)
    out = []
    for a in ACCELERATIONS:
        weight = _weight_present(('loras',), a['file'])
        pack = packs.get(a['pack']) if a['pack'] else None
        pack_ok = None if pack is None else pack['available']
        out.append({
            'id': a['id'], 'label': a['label'], 'arena': a['arena'], 'steps': a['steps'],
            'author': a['author'], 'license': a['license'], 'file': a['file'],
            'action': a['action'], 'hint': a['hint'], 'weight_present': bool(weight),
            'pack': pack,
            # A probe that could not run is not a no — same rule as the packs.
            'available': bool(weight) and pack_ok is not False,
        })
    return out


def sage_available(classes=None) -> bool:
    """Whether to keep the SageAttention patch in the graph.

    Fails CLOSED, unlike everything else here: when /object_info cannot be read
    we build WITHOUT Sage. A graph missing a speed patch renders correctly and a
    little slower; a graph naming a node the install does not have is refused
    outright. The asymmetry is the whole reason this one is not `is not None`.
    """
    if classes is None:
        classes = registered_classes()
    return bool(classes) and SAGE_CLASS in classes


def studio_ready(missing=None) -> bool:
    """Can a clip be rendered right now, with the default options off?

    THE verdict — capabilities, the options route and the enqueue preflight all
    ask this one function, so no surface can decide readiness from a different
    subset of the gaps.
    """
    if missing is None:
        missing = missing_weights()
    return not any(m['required'] for m in missing)


# ⏱ How ComfyUI was STARTED decides more than any dial on this screen.
#
# The H3 set the graph loads weighs about 43 GB (DiT int8 21 + text encoder
# nvfp4 16 + the two VAEs 6), and ComfyUI's default loader keeps a copy of every
# weight it offloads in system RAM. On a machine whose RAM cannot hold that set
# beside the OS and the desktop, the models page through the swap file at every
# node change — measured per node on a 48 GB machine: a 56-frame clip took
# 348 s (319 s of them decoding the VAE), the next one 302-315 s. Started with
# `--fast-disk`, ComfyUI reads the weights back from the safetensors files
# instead: the same clip took 30 s cold and 21-24 s warm, RAM left alone.
#
# LDS's own launcher passes the flag (`comfyui_control._spawn`). A ComfyUI the
# user starts some other way — a .bat, a Desktop install, another machine —
# is not ours to configure, so the Studio asks the running instance what it
# was started with (`/system_stats` echoes its argv, its RAM and its version)
# and SAYS so when the flag is missing on a machine that needs it. Every
# "cannot tell" case stays silent: advice built on a guess would name a flag
# the user may already be passing, or one their ComfyUI does not know.
H3_HOST_RAM_GB = 43
# psutil reports the RAM the OS can use, a little under the nominal size: the
# 48 GB machine above reads 47.7, a 64 GB one about 63.7. The floor sits under
# the nominal 64 GB it means, or that class of machine would get the card the
# comment above says it should not. Calibrated on one point (48 GB of RAM,
# 43 GB of weights); nothing in between has been measured.
FAST_DISK_RAM_FLOOR_GB = 60
# `--fast-disk` was declared in ComfyUI v0.23.0 (2026-06-01). argparse answers
# an unknown flag with an exit before the server exists, so advising it to an
# older instance would stop that ComfyUI from starting — the exact failure the
# launcher guards against by reading cli_args.py. Below this version, or
# without one, the advice stays silent.
FAST_DISK_MIN_COMFYUI = (0, 23, 0)
_FAST_DISK_FLAG = '--fast-disk'
# `--high-ram` is the user saying "I prefer the page file to model loading" —
# the opposite choice, made on purpose, whatever the loader. Never argued with.
_HIGH_RAM_FLAG = '--high-ram'
# `--fast-disk` only steers ComfyUI's DYNAMIC loader (its whole effect runs
# through the vbar path). With that loader off the flag is inert, so advising
# it alone would send the user to a change that changes nothing. ComfyUI's own
# rule (`enables_dynamic_vram`): `--enable-dynamic-vram` forces the loader on;
# otherwise `--disable-dynamic-vram` or any of the memory modes below turns it
# off. The switch is what a launcher tuned for an older ComfyUI still carries
# (the maintainer's own did, calibrated on a 20 GB image model, and ComfyUI now
# announces it as "will be removed soon") — that one is named, to be removed.
# The modes are a choice, and the advice stays silent rather than argue.
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
    from .. import config as cfg
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
