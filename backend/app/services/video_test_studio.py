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
import random
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

# ⚡ The 4-step distillation LoRA (larryvrh), applied through its OWN node, not
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
                   eros=False, eros_on_disk=False, sparse='',
                   latent_upscale=False, source_ratio=None, sage=True,
                   filename_prefix='lds_video_test') -> dict:
    """One MiniMax H3 clip, as a ComfyUI graph.

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

    if turbo:
        _graft_turbo(wf)
        notes.append(f'turbo: 4-step distillation, steps={wf[N_SCHEDULER]["inputs"]["steps"]}')
    if steps is not None:
        # An explicit step count always wins, including over the turbo default:
        # the panel showing 6 while the graph runs 4 is the exact failure this
        # pipeline already paid for once.
        wf[N_SCHEDULER]['inputs']['steps'] = max(4, min(40, int(steps)))

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
            'steps': wf[N_SCHEDULER]['inputs']['steps']}


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
    head = [N_TURBO_LORA, 0] if N_TURBO_LORA in wf else [N_UNET, 0]
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
                 turbo=False, eros=False, sparse='', latent_upscale=False,
                 source_ratio=None, skip_preflight=False) -> dict:
    """Build the graph, record the clip, queue the job — in that order.

    The row is written BEFORE the queue insert and in the SAME transaction, so a
    job can never exist without the row that explains what it was: the queue
    monitor's completion callback resolves by `job_id`, and a missing row means
    a finished clip nobody can attribute.
    """
    from ..extensions import db
    from ..job_queue import queue_manager
    from ..models import VideoTestClip

    # ONE /object_info read for the whole launch, and it decides two things:
    # whether SageAttention goes into the graph at all, and (through the
    # preflight below) whether an armed option's nodes are there. Reading it
    # twice would let a ComfyUI that restarts between the two answer differently
    # for the same clip.
    classes = registered_classes()
    built = build_workflow(
        prompt=prompt, mode=mode, image=image, seed=seed, steps=steps,
        frames=frames, megapixels=megapixels, aspect=aspect, lora=lora,
        lora_strength=lora_strength, turbo=turbo, eros=eros,
        eros_on_disk=eros_on_disk() if eros else False, sparse=sparse,
        latent_upscale=latent_upscale, source_ratio=source_ratio,
        sage=sage_available(classes), filename_prefix=new_prefix(user_id))
    if not skip_preflight:
        preflight(built['workflow'])

    job_id = str(uuid.uuid4())
    clip = VideoTestClip(
        run_id=run_id, dataset_id=dataset_id, job_id=job_id, status='pending',
        prompt=prompt, mode=('t2v' if str(mode).lower() == 't2v' else 'i2v'),
        source_image=image, seed=built['seed'], steps=built['steps'],
        frames=built['frames'], megapixels=built['megapixels'],
        fps=float(_profile().get('fps') or 24.0), base_model=built['base'],
        lora=lora, lora_strength=float(lora_strength) if lora else None,
        turbo=bool(turbo), sparse=normalise_sparse(sparse),
        latent_upscale=bool(latent_upscale))
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
    if failed:
        clip.status = 'failed'
        clip.error = (reason or 'Generation failed (see the server log in '
                                'Settings for the ComfyUI error).')
        db.session.commit()
        return
    clip.filename = filename
    clip.status = 'done'
    _bring_clip_home(filename)
    db.session.commit()


def _bring_clip_home(filename):
    """Move the finished mp4 out of ComfyUI's output dir into the app's own.

    Disk move first, HTTP `/view` fetch as the fallback: a ComfyUI pointed at a
    custom output directory (`--output-directory`, the desktop app's setting)
    has a path this app cannot know, but its API serves the file regardless.
    A failure here is not fatal — the row keeps the filename and the clip is
    still readable from ComfyUI.
    """
    from . import lora_test_studio as lts
    dst = os.path.join(str(clips_dir()), filename)
    if os.path.exists(dst):
        return
    out_dir = lts._comfy_output_dir()
    src = os.path.join(out_dir, filename) if out_dir else None
    try:
        if src and os.path.exists(src):
            shutil.move(src, dst)
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
