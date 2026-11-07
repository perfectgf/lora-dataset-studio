"""LoRA Test Studio: checkpoint/strength sweeps with fixed seeds and prompts.

Generate comparison grids for a trained dataset, collect votes and persist
winning settings. Commit each row before enqueueing to avoid orphan jobs.
Queue metadata marks is_lora_test; completion/failure/cancellation links through
link_completed_test_image at the same hook as dataset jobs. Move completed files
into the dataset folder. Runs are free, with one active run per dataset, and
cannot start while training or vision owns the GPU.

Cell counts have no hard cap: the serial queue and UI count/time estimate let
users choose the workload. MAX_TEST_IMAGES is a frontend warning threshold,
not an enforced limit.

Workflow paths resolve under cfg.BACKEND_DIR/workflows; ComfyUI output uses a
live accessor. This single-user app omits the original project's ownership and
gallery-history subsystems. Exact dataset trigger boundaries identify deployed
checkpoints, and every test-image row belongs to the local user."""
from __future__ import annotations
from ..utils.timestamps import naive_utcnow

from dataclasses import dataclass
import itertools
import json
import logging
import math
import os
import random
import re
import uuid

from .. import config as cfg
from ..extensions import db
from ..gpu_window import GpuBusyError
from ..models import FaceDataset, ImageGenerationQueue, LoraTestImage
from . import face_dataset_service as fds, trash
from . import lora_training as lt
from ..job_queue import GPU_ARBITER_LOCK, queue_manager
from ..utils.comfyui import (FAMILY_LABELS, KREA_ALLOWED_SAMPLERS, KREA_ALLOWED_SCHEDULERS,
                             KREA_HIRES_DENOISE, KREA_HIRES_MAX_SCALE,
                             KREA_ALLOWED_WEIGHT_DTYPES, apply_optimal_sampler_params,
                             family_of_lora, format_trained_lora_label, get_family_loras,
                             get_krea_loras,
                             get_krea_models, get_sdxl_loras, get_zimage_loras,
                             get_zimage_models,
                             load_workflow_local, resolve_checkpoint_ckpt_name)
from ..utils.zimage_helper import apply_zimage_settings

logger = logging.getLogger(__name__)

# ✨ The derivation kind of a row that lives in lora_test_image WITHOUT being a
# Test Studio cell: the Upscale & improve result produced from the ◉ Canvas
# lightbox. It has to be in this table because `canvas_image_node.image_id` is a
# `lora_test_image.id` and the board must be able to pin it — but no checkpoint×
# strength sweep ever made it.
CANVAS_IMAGE_IMPROVE = 'canvas_image_improve'


def _is_cell():
    """The predicate `_cells()` applies, for the queries that cannot START from it.

    A join that selects columns from another table (see
    `measured_seconds_per_image`) builds its own query and cannot be handed a
    `LoraTestImage.query`. It still needs the SAME rule, so the rule is written
    once here and both spellings share it — two hand-written copies of
    `derivation_kind IS NULL` is how the two would eventually disagree.
    """
    return LoraTestImage.derivation_kind.is_(None)


def _cells():
    """EVERY query in this module that means "the Test Studio cells" starts here.

    A derived row (see CANVAS_IMAGE_IMPROVE) is in the same table and must not be
    read as a cell. Auditing the 21 read sites found TEN that break otherwise,
    and none of them is theoretical:

      * `_active_run_count` counts pending rows with no file, so an improve still
        rendering answered "a test run is already in progress" and BLOCKED every
        new Test Studio launch;
      * the resume path picks up cancelled/failed rows by dataset and would have
        re-queued a failed improve as a Z-Image cell — wrong workflow, wrong
        engine;
      * `cell_scores` / `model_net_scores` aggregate by (checkpoint, strength, …),
        so a 👍 given to an UPSCALE in the gallery became a vote for the
        checkpoint that did not produce it;
      * `best_cell` and its per-checkpoint sibling take `order_by(id.desc())`, so
        the improvement — newest by construction — became the representative
        image of the winning config;
      * the face-scoring pass would spend GPU on it and `face_ranking` would
        average its score into its checkpoint's.

    So the filter is not a detail of one query, it is the meaning of "cell", and
    it lives in ONE place. `test_canvas_image_improve.py` forbids a bare
    `LoraTestImage.query` anywhere else in this module, so a future reader cannot
    reintroduce the leak by writing the obvious thing. A query that legitimately
    needs every row carries a `lds-allow-bare-lora-test-query:` comment saying
    why — the four that exist today all resolve ONE row by `job_id`, and the
    completion callback is among them: filtering derived rows out of it would
    leave every improvement pending forever.

    ⚠️ WHERE THIS HELPER MUST **NOT** BE USED — this boundary is the feature.
    `services/cloud_training.py` (checkpoint_gallery, run_gallery,
    canvas_image_nodes) and `services/gallery_download.py` read the SAME table and
    deliberately do NOT filter these rows out: showing the improvement in the
    gallery it was made from, next to its source, and letting it be pinned onto
    the board, IS what the ✨ button was asked for. "Harmonising" by applying this
    helper there would silently delete the feature, so a test pins both
    directions: absent from the studio's cells, present in the checkpoint gallery
    and pinnable.
    """
    return LoraTestImage.query.filter(_is_cell())


# Hard image cap per run (roughly 4-6 minutes of GPU time with Z-Image Turbo).
MAX_TEST_IMAGES = 24
# Guest checkpoints (a LoRA file that is not in this dataset's trigger-matched
# pool) share the Canvas plugin-node cap: enough for a mine-vs-theirs grid,
# not enough for a picker dump to explode the matrix.
MAX_GUEST_CHECKPOINTS = 16
GUEST_LABEL_PREFIX = 'Theirs · '

# One shared LoRA strength range governs both Studio sweeps and the leading
# weight of Blend stacks, which both reach LoraTestImage.strength through
# build_matrix. Different bounds would reject an entire blend unexpectedly.
# The upper bound increased from 4 to 5 on 2026-08-08 to support stronger styles
# and undertrained LoRAs; the -2 lower bound supports inverse slider effects.
# Update browser mirrors in the same commit:
# frontend/src/components/dataset/studio/loraStack.js (COMBINE_MAX_WEIGHT)
# frontend/src/components/dataset/studio/constants.js (STRENGTH_CHOICES_EXTENDED).
MIN_LORA_STRENGTH = -2.0
MAX_LORA_STRENGTH = 5.0

# Default identity prompt; substitute the dataset trigger word.
IDENTITY_PROMPT_TEMPLATE = "{trigger}, close-up portrait, neutral expression, looking at camera"

# ZTurbo workflow resolution.
TEST_WIDTH, TEST_HEIGHT = 832, 1216

# Workflow paths for the bundled image-generation graphs.
WORKFLOW_ZTURBO_PATH = cfg.BACKEND_DIR / 'workflows' / 'ZImage_bigLove_ZT3_optimal.json'
WORKFLOW_HQ_PATH = cfg.BACKEND_DIR / 'workflows' / 'image_real_HQ.json'
WORKFLOW_KREA_TURBO_PATH = cfg.BACKEND_DIR / 'workflows' / 'krea2_turbo.json'
WORKFLOW_KREA_IMG2IMG_PATH = cfg.BACKEND_DIR / 'workflows' / 'krea2_turbo_img2img.json'
WORKFLOW_FLUX2KLEIN_PATH = cfg.BACKEND_DIR / 'workflows' / 'flux2_klein_t2i.json'


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


# Test aspect ratios at about one megapixel, in multiples of 64; framing can affect LoRA appearance.
TEST_ASPECTS = {
    '9:16': (832, 1216),
    '3:4':  (896, 1152),
    '1:1':  (1024, 1024),
    '4:3':  (1152, 896),
    '16:9': (1216, 832),
}
# SDXL uses the same ratios with the long side capped at 1024. Z-Image's
# 1216-pixel long-side buckets can distort SDXL merges/DMD checkpoints.
# Keep dimensions in multiples of 64.
TEST_ASPECTS_SDXL = {
    '9:16': (576, 1024),
    '3:4':  (768, 1024),
    '1:1':  (1024, 1024),
    '4:3':  (1024, 768),
    '16:9': (1024, 576),
}
# Studio formats -> Generate aspectRatio values (mirrors
# react-frontend/src/components/dataset/studio/constants.js:ASPECT_TO_GENERATE).
_STUDIO_ASPECT_TO_GENERATE = {
    '9:16': 'portrait', '3:4': 'portrait', '1:1': 'square',
    '4:3': 'landscape', '16:9': 'landscape',
}
_MODE_LABEL_BY_FAMILY = {**FAMILY_LABELS, 'krea': 'Krea 2 Turbo'}
DEFAULT_ASPECT = '9:16'
# Resolution tiers match resolution.py/_TIERS; None preserves the legacy fixed dimensions.
RESOLUTION_TIERS = ('fast', 'standard', 'hq', 'max')
# Map the five Studio ratios to compute_tier_dims names such as square and landscape.
_ASPECT_TO_TIER_RATIO = {
    '1:1': 'square', '4:3': 'landscape', '3:4': 'portrait',
    '16:9': 'widescreen', '9:16': 'tall',
}


def _aspect_dims(aspect, train_type=None, resolution_tier=None, resolution_multiplier=1.0):
    """Resolve width/height for a ratio, defaulting on unknown input.

    A supplied fast/standard/hq/max tier uses compute_tier_dims and a clamped
    1.0-1.9 multiplier. Otherwise use family-specific legacy tables; the multiplier
    does not affect them. SDXL tier output caps its long side at 1024*multiplier
    in multiples of 64, avoiding dimensions that distort SDXL merges/DMD models."""
    if resolution_tier in RESOLUTION_TIERS:
        named = _ASPECT_TO_TIER_RATIO.get(aspect)
        if named:
            from ..utils.resolution import clamp_multiplier, compute_tier_dims
            w, h = compute_tier_dims(named, resolution_tier, resolution_multiplier)
            if (train_type or '').lower() == 'sdxl':
                # Scale the SDXL cap too, matching the frontend instead of silently cancelling its multiplier.
                ceiling = 1024.0 * clamp_multiplier(resolution_multiplier)
                longest = max(w, h)
                if longest > ceiling:
                    sc = ceiling / longest
                    w = max(64, int(round(w * sc / 64)) * 64)
                    h = max(64, int(round(h * sc / 64)) * 64)
            return w, h
    table = TEST_ASPECTS_SDXL if (train_type or '').lower() == 'sdxl' else TEST_ASPECTS
    return table.get(aspect, table[DEFAULT_ASPECT])

# Optional CFG/step axes retain distilled defaults (CFG 1, eight steps) for
# Z-Image Turbo, Krea Turbo and distilled SDXL checkpoints. Sweeps help compare
# identity consistency across settings.
DEFAULT_CFG = 1.0
DEFAULT_STEPS = 8
# Additive only — these lists are echoed into the Studio pickers and a value that
# disappears would strand a persisted selection. 3.5/4.0/5.0 and 30/50 exist so the
# NON-distilled Z-Image Base defaults below are reachable from the picker at all.
CFG_CHOICES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
# 25 is the sample-step count a dense Krea 2 run previews with — see
# KREA_RAW_DEFAULTS below; without it in the picker the recommended setting for a
# full-model artifact would not be selectable at all.
STEPS_CHOICES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 25, 30, 32, 40, 50]

# --- Per-BASE-MODEL sampler defaults (bobba84, GitHub #18) --------------------
# Z-Image ships in two flavours that need opposite sampler settings, and the app
# used to hand both the same one: picking "Z-Image Base" in the Test Studio landed
# on cfg 1 / 8 steps, which are Turbo's numbers. Turbo is guidance-DISTILLED — cfg 1
# is correct there and ruinous on Base, where it means "no guidance at all". A user
# trying Base with those settings concludes the model is bad.
#
# PROVENANCE OF THE BASE NUMBERS: ComfyUI's own Z-Image day-0 announcement states
# Z-Image-Base "requires 30-50 steps with cfg 3~5 for optimal quality". We take the
# CONSERVATIVE end of the step range (30, the cheapest of the recommended window)
# and the middle of the cfg range (4.0). These are documented starting points, NOT
# values this project measured — they are defaults for an axis the user can and
# should sweep, which is the entire point of the Studio grid.
ZIMAGE_TURBO_DEFAULTS = {'cfg': DEFAULT_CFG, 'steps': DEFAULT_STEPS}
ZIMAGE_BASE_DEFAULTS = {'cfg': 4.0, 'steps': 30}

# Whole-word phrases (see zimage_model_resolver._phrase: separators and case are
# normalised away) that identify a build. Distilled markers are checked FIRST, so a
# name carrying both stays on today's behaviour rather than flipping to slow+guided.
_ZIMAGE_DISTILLED_PHRASES = ('turbo', 'zt1', 'zt2', 'zt3', 'distill', 'distilled',
                             'lightning', 'lightx2v', 'step')
_ZIMAGE_BASE_PHRASES = ('base', 'deturbo', 'de turbo', 'raw')


def zimage_build_of(model_name) -> str:
    """'turbo' | 'base' | 'unknown' for a Z-Image UNET filename, read from its NAME —
    the only signal available, since these are loose files a user downloaded. 'unknown'
    deliberately keeps the historical Turbo defaults: the overwhelming majority of
    Z-Image checkpoints in the wild are Turbo finetunes, and changing the defaults for
    an unrecognised name would be a regression for everyone who is fine today."""
    from .zimage_model_resolver import _phrase
    key = _phrase(_basename(model_name))
    if any(f' {p} ' in key for p in _ZIMAGE_DISTILLED_PHRASES):
        return 'turbo'
    if any(f' {p} ' in key for p in _ZIMAGE_BASE_PHRASES):
        return 'base'
    return 'unknown'


def zimage_model_defaults(model_name) -> dict:
    """{'cfg', 'steps'} for ONE Z-Image base model. Turbo/unknown -> today's values."""
    return dict(ZIMAGE_BASE_DEFAULTS if zimage_build_of(model_name) == 'base'
                else ZIMAGE_TURBO_DEFAULTS)


# --- Krea 2: the same trap, one family over ------------------------------------
# The full-model (dense) lane delivers a RAW Krea 2 checkpoint — undistilled, and
# it needs a real CFG and a real step count. The Studio's family defaults are
# Turbo's (cfg 1 / 8 steps), and applied to a Raw model they render a blurry
# sketch that reads as "the fine-tune failed". Reported after the first dense run
# was tested that way.
#
# The numbers are not invented here: they are the sample settings the run's OWN
# preview sheet was rendered with, imported from the training recipe so a change
# there moves the test lane with it.
_KREA_DISTILLED_PHRASES = ('turbo', 'distill', 'distilled', 'lightning',
                           'lightx2v', 'step', 'schnell')
# 'full' and 'fp8' cover this app's own dense deliveries (Krea_full_<trigger>…
# and its _fp8 twin); 'raw'/'base'/'undistilled' cover Krea-2-Raw derivatives.
_KREA_RAW_PHRASES = ('raw', 'base', 'full', 'undistilled', 'fp8')


def krea_build_of(model_name) -> str:
    """'turbo' | 'raw' | 'unknown' for a Krea 2 checkpoint, read from its NAME.

    Same contract as ``zimage_build_of``: distilled markers win, and 'unknown'
    keeps today's Turbo defaults — most Krea checkpoints in the wild are Turbo
    finetunes and changing their defaults would be a regression.
    """
    from .zimage_model_resolver import _phrase
    key = _phrase(_basename(model_name))
    if any(f' {p} ' in key for p in _KREA_DISTILLED_PHRASES):
        return 'turbo'
    if any(f' {p} ' in key for p in _KREA_RAW_PHRASES):
        return 'raw'
    return 'unknown'


def krea_model_defaults(model_name) -> dict:
    """{'cfg', 'steps'} for ONE Krea 2 checkpoint. Turbo/unknown -> today's values."""
    if krea_build_of(model_name) != 'raw':
        return {'cfg': DEFAULT_CFG, 'steps': DEFAULT_STEPS}
    from . import lora_training as lt
    return {'cfg': float(lt.FULL_TRANSFORMER_SAMPLE_GUIDANCE),
            'steps': int(lt.FULL_TRANSFORMER_SAMPLE_STEPS)}


def studio_model_defaults(family, models) -> dict:
    """{model_value: {'cfg', 'steps'}} for the bases the Studio offers, so the front
    can seed its axes from the SELECTED base instead of one family-wide constant.
    Z-Image and Krea 2 both ship a distilled and an undistilled build that need
    opposite sampler settings; SDXL returns nothing and keeps
    `default_cfg`/`default_steps`. The shape is per-family on purpose so the next
    family that needs it has nowhere else to put it."""
    fam = (family or '').lower()
    resolver = {'zimage': zimage_model_defaults, 'krea': krea_model_defaults}.get(fam)
    if fam in TRAINED_IMAGE_FAMILIES or fam == 'flux2klein':
        resolver = lambda model: studio_family_defaults(fam)
    if resolver is None:
        return {}
    out = {}
    for m in models or []:
        value = m.get('value') if isinstance(m, dict) else m
        if value:
            out[value] = resolver(value)
    return out


def _basename(path: str) -> str:
    """Basename tolerant to ComfyUI's backslash-relative LoRA paths."""
    return (path or '').replace('\\', '/').rsplit('/', 1)[-1]


def _checkpoint_display_label(filename, known=None) -> str:
    """Grid / ranking label for one checkpoint. Trigger-matched files keep the
    trained-epoch parse; anything else (a guest from models/loras) is prefixed
    so it groups as Theirs instead of sorting into the middle of your steps."""
    trained = format_trained_lora_label(filename)
    stem = _basename(filename).rsplit('.', 1)[0]
    if known is not None and filename not in known:
        return GUEST_LABEL_PREFIX + (trained or stem)
    return trained or stem


def _run_family_map(rows) -> dict:
    """run_id → family taken from any member whose path has a family folder.

    Guest files at the loras root have family_of_lora = None; they inherit
    their run's family so they stay on the same grid as the epochs they were
    compared against, instead of collapsing to the historical 'zimage' fallback
    and vanishing from a Krea studio."""
    out = {}
    for r in rows:
        rid = getattr(r, 'run_id', None)
        if not rid or rid in out:
            continue
        fam = family_of_lora(getattr(r, 'checkpoint', None))
        if fam:
            out[rid] = fam
    return out


def _row_family(row, by_run, default='zimage'):
    """Folder prefix, else the run's foldered sibling, else `default`.

    Pass `default=None` to distinguish 'still unknown' from zimage — a
    guest-only run has no sibling to inherit from and must not collapse
    to zimage or it vanishes from a Krea studio."""
    return (family_of_lora(getattr(row, 'checkpoint', None))
            or by_run.get(getattr(row, 'run_id', None))
            or default)


def _filter_rows_by_family(rows, family, default='zimage'):
    """Keep cells of one pipeline. Folder prefix wins; unfoldered guests
    inherit their run. A guest-only run (every file at the loras root) has
    no foldered sibling to inherit from — those cells stay on the family
    tab that is open, instead of collapsing to `default` and vanishing
    from a Krea studio. Unprefixed historical names that ARE this
    dataset's stay zimage, as they always have."""
    if not family:
        return list(rows)
    fam = family.lower()
    by_run = _run_family_map(rows)
    known_by_ds = {}
    out = []
    for r in rows:
        r_fam = _row_family(r, by_run, default=None)
        if r_fam is None:
            dsid = getattr(r, 'dataset_id', None)
            if dsid not in known_by_ds:
                ds = db.session.get(FaceDataset, dsid) if dsid else None
                known_by_ds[dsid] = _known_checkpoints(ds)
            if getattr(r, 'checkpoint', None) not in known_by_ds[dsid]:
                out.append(r)
                continue
            r_fam = default
        if r_fam == fam:
            out.append(r)
    return out


def _accept_guest_checkpoints(guests) -> set:
    """Fail-closed: a filename not in this dataset's trigger-matched pool must
    still resolve under a loras root, and must not climb out of it. Same
    messages as Canvas `external_loras` so the two free-text channels cannot
    disagree. Returns the set to union into `allowed`."""
    uniq = []
    seen = set()
    for fn in guests or []:
        if not fn or fn in seen:
            continue
        seen.add(fn)
        uniq.append(fn)
    if len(uniq) > MAX_GUEST_CHECKPOINTS:
        raise ValueError(
            f'at most {MAX_GUEST_CHECKPOINTS} LoRAs from outside this dataset '
            'can be in one run')
    out = set()
    for fn in uniq:
        if _is_unsafe_external_lora_name(fn):
            raise ValueError(f'invalid external LoRA name: {fn}')
        if not _resolve_lora_abs_path(fn):
            raise ValueError(f'external LoRA not found: {fn}')
        out.add(fn)
    return out


def _wilson_lower_bound(likes: int, voted: int, z: float = 1.96) -> float:
    """Return the 95% Wilson lower bound for the positive-vote rate, or zero.

    Ranking by raw counts favors often-tested configurations; raw rates favor
    single-vote results. Wilson balances rate and confidence: 2/2 (0.34) beats
    6/10 (0.31), while 5/5 (0.57) beats 2/2."""
    if voted <= 0:
        return 0.0
    p = likes / voted
    z2 = z * z
    denom = 1.0 + z2 / voted
    centre = p + z2 / (2 * voted)
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * voted)) / voted)
    return (centre - margin) / denom


def identity_prompt(ds, with_trigger=True) -> str:
    """Build the default prompt, omitting the trigger when with_trigger=False.

    An empty custom prompt must not reintroduce a trigger through fallback text
    while the UI claims "no trigger"."""
    if not with_trigger:
        return IDENTITY_PROMPT_TEMPLATE.format(trigger='').lstrip(', ')
    return IDENTITY_PROMPT_TEMPLATE.format(trigger=(ds.trigger_word or '').strip())


def _trigger_pattern(trigger):
    """`trigger` as a whole token, case-insensitive; group 1 is the boundary before it."""
    return re.compile(r'(^|[^0-9A-Za-z])' + re.escape(trigger) + r'(?=[^0-9A-Za-z]|$)', re.IGNORECASE)


def _prompt_with_trigger(prompt, trigger_word):
    """Prepend the dataset trigger unless empty or already a whole token.

    Match case-insensitively to avoid duplication. Apply only during workflow
    construction; keep persisted prompts raw for the recent-prompts menu."""
    p = (prompt or '').strip()
    t = (trigger_word or '').strip()
    if not p or not t:
        return p
    if _trigger_pattern(t).search(p):
        return p
    return f'{t}, {p}'


def _learned_trigger(lora_name, trigger):
    """`trigger` spelled as the LoRA file `lora_name` learned it.

    Training writes the dataset trigger through lora_training._safe_trigger, in every
    family, so "Double pen" is learned as "Double_pen", and a text encoder splits the
    two differently: sending the typed words never activates the LoRA. The file's own
    record (lora_training.recorded_trigger) is the authority, which also covers a
    trigger renamed after training and a guest LoRA's own word. A LoRA that records no
    trigger keeps the historical injection: `trigger` as typed."""
    if not lora_name or not (trigger or '').strip():
        return trigger
    path = _resolve_lora_abs_path(lora_name)
    return (lt.recorded_trigger(path) if path else None) or trigger


def _respell_trigger(prompt, typed, learned):
    """`prompt` with each whole-token `typed` trigger respelled `learned`. The default
    identity prompt carries the typed trigger, and so may a written prompt: that word is
    the trigger, so it is respelled in place instead of getting a second copy prefixed."""
    t, u = (learned or '').strip(), (typed or '').strip()
    if not t or not u or t.casefold() == u.casefold() or _trigger_pattern(t).search(prompt):
        return prompt
    return _trigger_pattern(u).sub(lambda m: m.group(1) + t, prompt)


def _inject_triggers(prompt, checkpoint, trigger_word, extra_loras=None):
    """The prompt a cell sends: each trigger spelled as its LoRA learned it, head first.

    `trigger_word` is the tested checkpoint's dataset trigger, or a list headed by it
    (combine mode); a stacked member's trigger belongs to the `extra_loras` entry that
    carries it as `trigger`. Respells the typed triggers, then prefixes the missing ones."""
    if trigger_word is None:
        return _prompt_with_triggers(prompt, None)
    typed = [trigger_word] if isinstance(trigger_word, str) else list(trigger_word)
    owners = {}
    for entry in extra_loras or ():
        if isinstance(entry, dict) and entry.get('trigger') and entry.get('filename'):
            owners.setdefault(entry['trigger'], entry['filename'])
    learned = [_learned_trigger(checkpoint if i == 0 else owners.get(t), t)
               for i, t in enumerate(typed)]
    p = (prompt or '').strip()
    for t, u in zip(learned, typed):
        p = _respell_trigger(p, u, t)
    return _prompt_with_triggers(p, learned)


def _prompt_with_triggers(prompt, trigger_words):
    """Same as `_prompt_with_trigger` but for a STACK of LoRA loaded together
    (combine mode): every trigger is prefixed, in selection order, so
    `["a", "b"], "portrait"` gives `"a, b, portrait"`.

    Accepts a bare string (the historical single-LoRA call) or any iterable.
    Folding runs right-to-left because each step prepends; the dedup of
    `_prompt_with_trigger` also collapses two LoRA that share a trigger, so a
    stack never emits the same token twice."""
    if trigger_words is None or isinstance(trigger_words, str):
        return _prompt_with_trigger(prompt, trigger_words)
    p = (prompt or '').strip()
    for t in reversed(list(trigger_words)):
        p = _prompt_with_trigger(p, t)
    return p


# Discover deployable families in UI order, using shared FAMILY_LABELS.
# Cover every family in lora_training._FAMILY_SUBDIR; the parity contract guards
# this. Missing families previously caused deployed Klein files to be searched
# in the Z-Image folder and incorrectly reported as undeployed (issue #52).
FAMILIES = ('zimage', 'sdxl', 'krea', 'flux', 'flux2klein', 'anima', 'qwenimage21')

# Distinguish families whose LoRAs can be discovered from families the Studio
# can generate with. Generation requires a complete workflow, settings adapter
# and base allowlist. Conflating these lists exposed an unusable Klein Generate
# choice that silently failed through the Z-Image path (issue #53).
TRAINED_IMAGE_FAMILIES = ('flux', 'anima', 'qwenimage21')
GENERATION_FAMILIES = ('zimage', 'sdxl', 'krea', 'flux2klein') + TRAINED_IMAGE_FAMILIES


def studio_family_defaults(family):
    """Defaults shared by the axes, persisted cells and graph builders."""
    if family in TRAINED_IMAGE_FAMILIES:
        from ..utils.trained_image_workflows import family_defaults
        defaults = family_defaults(family)
        return {'cfg': defaults['cfg'], 'steps': defaults['steps']}
    if family == 'flux2klein':
        return {'cfg': 1.0, 'steps': 4}
    return {'cfg': DEFAULT_CFG, 'steps': DEFAULT_STEPS}


def generation_capabilities(family):
    caps = {'negative_prompt': family in ('zimage', 'anima', 'qwenimage21', 'krea')}
    if family == 'krea':
        # Krea 2 Turbo samples at CFG 1, where ComfyUI never evaluates the
        # negative branch. The panel offers the field once a cell runs above
        # CFG 1 (a Raw base, or a CFG axis value): the only cells it can change.
        caps['negative_prompt_needs_cfg'] = True
    return caps


def family_base_models(family):
    """Family-specific filenames for every launch, replay and winning preset."""
    if family == 'sdxl':
        return [m['filename'] for m in list_sdxl_base_models()]
    if family == 'krea':
        return [None] + get_krea_models()
    if family == 'flux2klein':
        from ..utils.comfyui import get_flux2_klein_models
        return [None] + [m['filename'] if isinstance(m, dict) else m
                         for m in get_flux2_klein_models()]
    if family in TRAINED_IMAGE_FAMILIES:
        from .trained_image_models import list_family_models
        return [m['filename'] for m in list_family_models(family)]
    if family == 'zimage':
        return get_zimage_models()
    raise _no_generation_lane(family)


def _require_family_bases(family):
    models = family_base_models(family)
    if not models:
        label = FAMILY_LABELS.get(family, family)
        raise ValueError(f'no {label} model available — configure its models in Settings or Setup')
    return models


def _select_base_models(models, z_model=None, z_models=None, *, family=None):
    """An explicit unavailable base must never silently select another model."""
    if z_models is not None and not isinstance(z_models, (list, tuple)):
        raise ValueError('base models must be a list')
    requested = list(z_models) if z_models else ([z_model] if z_model else [])
    if not requested and family in TRAINED_IMAGE_FAMILIES:
        from .trained_image_models import resolve_family_assets
        requested = [resolve_family_assets(family)['diffusion_model']]
    requested = [None if m in ('', None) else m for m in requested]
    invalid = [m for m in requested if m not in models]
    if invalid:
        raise ValueError('selected base model is unavailable for this family — choose an installed base')
    return list(dict.fromkeys(requested)) or [models[0]]


def can_generate_with(family: str) -> bool:
    """Return whether Studio has a generation path for this family."""
    return (family or 'zimage').lower() in GENERATION_FAMILIES


def _no_generation_lane(family: str) -> ValueError:
    """Explain the requested family's missing generation path.

    Name that family instead of reporting an unrelated Z-Image model error after
    a silent fallback."""
    label = FAMILY_LABELS.get((family or '').lower(), family or 'this family')
    return ValueError(
        f'generating from the board is not supported for {label} yet — its '
        'LoRAs train, deploy and show up as deployed, but the studio has no '
        f'{label} generation workflow. Use a family that has one, or generate '
        'from the dataset instead.')


def _pool_for_family(family: str) -> list[dict]:
    """Read the selected family's deployed LoRA pool through its own folder.

    Use get_family_loras and the shared family-subfolder table. Unknown families
    return [] rather than silently searching Z-Image. The parity contract ensures
    no deployable family is missing."""
    f = (family or 'zimage').lower()
    if f == 'sdxl':
        return get_sdxl_loras()
    if f == 'krea':
        return get_krea_loras()
    if f == 'zimage':
        return get_zimage_loras()
    if f in FAMILIES:
        return get_family_loras(f)
    return []


def _trigger_token_match(norm: str, trigger: str) -> bool:
    """Match a trigger followed by '_'/'-' or the end of the normalized name.

    Require a complete token so a short trigger cannot claim checkpoints belonging
    to a longer trigger that shares its prefix."""
    if not norm.startswith(trigger):
        return False
    rest = norm[len(trigger):]
    return rest == '' or rest[0] in ('_', '-')


def _trigger_match_checkpoints(ds, family=None) -> list[dict]:
    """List checkpoints matching the dataset's exact canonical trigger boundary.

    Support both legacy <trigger>-<step> and ai-toolkit lora_<trigger>_<step> names,
    case-insensitively. The selected family chooses the pool; otherwise use the
    dataset's train_type. Return [{filename, label}] in LoraLoader form.
    Canonicalize with lt._safe_trigger, exactly as training/deployment do, so
    multiword triggers match underscore-normalized filenames. Never independently
    reimplement this slugging rule."""
    trigger = lt._safe_trigger(ds).lower()
    if not trigger:
        return []
    fam = (family or getattr(ds, 'train_type', None) or 'zimage').lower()
    pool = _pool_for_family(fam)
    out = []
    for lora in pool:
        base = _basename(lora['filename'])
        stem = base.rsplit('.', 1)[0]
        norm = stem.lower()
        if norm.startswith('lora_'):  # accept ai-toolkit's raw prefix
            norm = norm[len('lora_'):]
        if _trigger_token_match(norm, trigger):
            # Pass the dataset's REAL trigger (not the safe/lowercased match form) so
            # a multi-token trigger like `leg_behind` labels faithfully instead of
            # splitting into `leg · behind` (the deployed filename can't disambiguate
            # the trigger's own underscores from the field separators on its own).
            entry = {'filename': lora['filename'],
                     'label': format_trained_lora_label(
                         lora['filename'], fam,
                         trigger=getattr(ds, 'trigger_word', None)) or stem}
            # Discreet retrofit badge for a mislabelled deploy: read the file's
            # REAL arch and flag it when it contradicts the folder's family, so a
            # wrong-family checkpoint is visible in the picker (not silently no-op).
            _p = _resolve_lora_abs_path(lora['filename'])
            _detected = lt.detect_lora_arch(_p) if _p else None
            if lt.lora_arch_conflicts(_detected, fam):
                entry['arch_mismatch'] = _detected
                entry['arch_label'] = lt._LORA_ARCH_LABEL.get(_detected, _detected)
            out.append(entry)
    return out


def list_test_checkpoints(ds, family=None) -> list[dict]:
    """List the selected family's testable checkpoints for this dataset.

    The caller already scopes ds to the local user. Return [{filename, label}]
    with filenames in LoraLoader form; no cross-user ownership filter is needed."""
    return _trigger_match_checkpoints(ds, family)


def _known_checkpoints(ds, family=None) -> set:
    """Filenames this dataset owns in `family` (or every family, if omitted).
    Used to decide the Theirs · prefix: a guest is anything we generated that
    is not in this set."""
    if ds is None:
        return set()
    if family:
        return {c['filename'] for c in list_test_checkpoints(ds, family)}
    out = set()
    for fam in FAMILIES:
        out |= {c['filename'] for c in list_test_checkpoints(ds, fam)}
    return out


def available_families(ds) -> list[dict]:
    """List trained families with at least one matching, testable deployed checkpoint.

    Return [{family, label, count}] in FAMILIES order, or [] if none. Datasets may
    appear in multiple families. Only expose families with a complete generation
    path in this selector; other deployed families remain discoverable elsewhere."""
    out = []
    for fam in FAMILIES:
        if not can_generate_with(fam):
            continue
        n = len(list_test_checkpoints(ds, fam))
        if n:
            out.append({'family': fam, 'label': FAMILY_LABELS.get(fam, fam), 'count': n})
    return out


def permanent_lora_candidates(family) -> list[dict]:
    """List family-pool style/utility LoRAs eligible for always-on Studio use.

    Exclude names beginning with lora_, which represent trained checkpoint axes.
    Return shared [{filename, label}] entries using the pool's display names."""
    out = []
    for lora in _pool_for_family(family):
        base = _basename(lora['filename'])
        if base.lower().startswith('lora_'):
            continue  # trained checkpoint: a test axis rather than an always-on utility
        out.append({'filename': lora['filename'],
                    'label': lora.get('displayName') or base.rsplit('.', 1)[0]})
    return out


def _resolve_family(ds, requested, families=None) -> str:
    """Choose a present requested family, then a present persisted family, then
    any present family; finally fall back to raw train_type if none exist.
    Avoid showing an empty family when another contains usable LoRAs."""
    fams = available_families(ds) if families is None else families
    keys = [f['family'] for f in fams]
    req = (requested or '').lower()
    if req in keys:
        return req
    default = (getattr(ds, 'train_type', None) or 'zimage').lower()
    if default in keys:
        return default
    return keys[0] if keys else default


def list_sdxl_base_models() -> list[dict]:
    """SDXL checkpoints usable as test bases: the same ones as Generate.
    Returns [{filename, label}]."""
    from ..utils.comfyui import get_checkpoint_models
    out = []
    for m in get_checkpoint_models():
        name = m.get('name')
        if name:
            out.append({'filename': name, 'label': name.split('\\')[-1]})
    return out


def list_all_testable_checkpoints(user_id) -> list[dict]:
    """Aggregate standalone-picker entries for every local dataset/family pair.

    Datasets can have deployments in multiple family folders. available_families
    uses those folders rather than the scalar train_type. Return dataset identity,
    trigger, family/label and checkpoints; train_type equals the entry's family
    for the frontend badge."""
    out = []
    datasets = (FaceDataset.query.filter_by(user_id=str(user_id))
                .order_by(FaceDataset.id.asc()).all())
    for ds in datasets:
        for fam in available_families(ds):   # one family/label/count entry per present family
            cks = list_test_checkpoints(ds, fam['family'])
            if not cks:
                continue
            out.append({'dataset_id': ds.id, 'dataset_name': ds.name,
                        'lora_label': ds.trigger_word or ds.name,
                        'trigger_word': ds.trigger_word,
                        'family': fam['family'],
                        'family_label': fam['label'],
                        'train_type': fam['family'],   # frontend badge/gate uses this entry's family
                        'checkpoints': cks})
    return out


# --- Guards ------------------------------------------------------------------
def _comfyui_recovery_target() -> dict | None:
    """Return the exact Studio view owning the durable ComfyUI barrier.

    The barrier is global, so the dataset currently displayed by the UI may not
    be the one that needs recovery.  Only expose a navigation target when the
    durable owner, queue row and unfinished Studio cell still agree.
    """
    owner = queue_manager.get_comfyui_stalled_barrier()
    if not isinstance(owner, dict) or not isinstance(owner.get('job_id'), str):
        return None
    job_id = owner['job_id']
    queue_row = ImageGenerationQueue.query.filter_by(job_id=job_id, status='stalled').first()
    # lds-allow-bare-lora-test-query: resolved by job_id — a stalled ComfyUI job
    # must be matched to whatever row owns it, derived rows included.
    cells = (LoraTestImage.query
             .filter_by(job_id=job_id, status='pending')
             .filter(LoraTestImage.filename.is_(None)).limit(2).all())
    if queue_row is None or len(cells) != 1:
        return None
    cell = cells[0]
    if (('dataset_id' in owner and owner.get('dataset_id') != str(cell.dataset_id))
            or ('run_id' in owner and owner.get('run_id') != cell.run_id)
            or ('cell_id' in owner and owner.get('cell_id') != str(cell.id))):
        return None
    kind = owner.get('kind', 'prompt')
    if kind == 'unknown_submit':
        if queue_row.comfyui_prompt_id is not None or owner.get('prompt_id') is not None:
            return None
    elif (kind != 'prompt'
          or str(queue_row.comfyui_prompt_id or '') != str(owner.get('prompt_id') or '')):
        return None
    return {
        'dataset_id': cell.dataset_id,
        'run_id': cell.run_id,
        'family': family_of_lora(cell.checkpoint) or 'zimage',
        'kind': kind,
    }


def gpu_busy_reason() -> str | None:
    """Return a human error when the GPU is held by a long-running exclusive
    task (LoRA training / vision pass), else None. The queue itself serializes
    normal generations, so no further locking is needed."""
    if queue_manager._get_system_state('training_in_progress', False):
        return "LoRA training in progress - the studio is unavailable (GPU busy)."
    if queue_manager._get_system_state('vision_in_progress', False):
        return "Vision pass in progress (GPU busy) - try again in a moment."
    if queue_manager.has_comfyui_stalled_barrier():
        target = _comfyui_recovery_target()
        if target and target['kind'] == 'unknown_submit':
            return ('A Test Studio image is paused because its ComfyUI submission outcome is unknown. '
                    'Restart ComfyUI, open the paused test, confirm the restart, then resume it.')
        return ('A Test Studio image is paused because ComfyUI stopped answering. '
                'Open the paused test, click Stop to recover it, then Resume.')
    return None


def _active_run_count(dataset_id=None) -> int:
    """Find pending cells without files. dataset_id=None checks globally for
    multi-LoRA comparisons; a supplied ID enforces one active run per dataset."""
    q = (_cells()
         .filter_by(status='pending')
         .filter(LoraTestImage.filename.is_(None)))
    if dataset_id is not None:
        q = q.filter_by(dataset_id=dataset_id)
    return q.count()


def _queue_activity(rows) -> dict:
    """Real queue state for the in-flight Test Studio cells in ``rows``.

    ``LoraTestImage.status`` intentionally stays ``pending`` until the output
    callback links a file, so it cannot distinguish waiting from GPU work. The
    linked ``ImageGenerationQueue`` row can. ``pending`` remains the historical
    total of unfinished cells for API compatibility; ``queued`` and
    ``generating`` split that total during the normal queue lifecycle.
    """
    live = [r for r in rows if r.status == 'pending' and not r.filename]
    job_ids = {r.job_id for r in live if r.job_id}
    queue_rows = (ImageGenerationQueue.query
                  .filter(ImageGenerationQueue.job_id.in_(job_ids)).all()
                  if job_ids else [])
    queue_by_job = {q.job_id: q for q in queue_rows}
    raw_by_job = {job_id: q.status for job_id, q in queue_by_job.items()}

    def _display_status(raw):
        if raw == 'pending':
            return 'queued'
        if raw in ('processing', 'sent_to_comfy'):
            return 'generating'
        if raw in ('cancel_requested', 'stalled'):
            return 'stalled'
        return raw

    queue_status = {job_id: _display_status(status)
                    for job_id, status in raw_by_job.items()}
    queue_error = {
        job_id: q.error_message for job_id, q in queue_by_job.items()
        if q.status == 'stalled' and isinstance(q.error_message, str) and q.error_message.strip()
    }
    queued = sum(1 for r in live if raw_by_job.get(r.job_id) == 'pending')
    generating = sum(1 for r in live
                     if raw_by_job.get(r.job_id) in ('processing', 'sent_to_comfy'))
    return {
        'pending': len(live),
        'queued': queued,
        'generating': generating,
        # Alias for consumers that use queue terminology rather than UI copy.
        'running': generating,
        'queue_status': queue_status,
        'queue_error': queue_error,
    }


def _unknown_submit_recovery(rows, activity):
    """UI-safe recovery metadata for the one exact paused Studio cell.

    A generic stalled tile may still have a known ComfyUI prompt and must use
    remote reconciliation instead. Expose this action only when the durable raw
    barrier, queue state, and linked cell all agree on an unknown submission.
    """
    owner = queue_manager.get_comfyui_stalled_barrier()
    if (owner is None or owner.get('kind') != 'unknown_submit'
            or owner.get('prompt_id') is not None
            or not isinstance(owner.get('job_id'), str)):
        return None
    job_id = owner['job_id']
    matching = [row for row in rows
                if row.status == 'pending' and not row.filename and row.job_id == job_id]
    if len(matching) != 1 or activity['queue_status'].get(job_id) != 'stalled':
        return None
    cell = matching[0]
    try:
        cell_id = int(owner.get('cell_id'))
    except (TypeError, ValueError):
        return None
    if str(cell_id) != owner.get('cell_id') or cell.id != cell_id:
        return None
    if (('dataset_id' in owner and owner.get('dataset_id') != str(cell.dataset_id))
            or ('run_id' in owner and owner.get('run_id') != cell.run_id)):
        return None
    return {
        'required': True,
        'kind': 'unknown_submit',
        'job_id': job_id,
        'cell_id': cell.id,
        'requires_comfyui_restart_confirmation': True,
    }


def build_matrix(checkpoints, strengths, aspects=None, cfgs=None, steps_list=None, steps2_list=None,
                 *, family=None) -> list[tuple]:
    """Validate and materialize the checkpoint/strength/aspect grid.

    Require nonempty checkpoint and strength axes, with deduplication preserving
    order. Strengths use the shared bounds: zero is the base-model control;
    negative values reverse the LoRA effect. Validate aspect ratios, defaulting
    to 9:16. Do not cap the cell count: users see the serial workload estimate.
    Blend leading weights also pass through this function into the strength
    column, so COMBINE_MAX_WEIGHT must use the same upper bound."""
    defaults = studio_family_defaults(family)
    cps = [c for c in (checkpoints or []) if isinstance(c, str) and c.strip()]
    sts = []
    for s in (strengths or []):
        try:
            v = round(float(s), 2)
        except (TypeError, ValueError):
            raise ValueError(f'invalid strength: {s!r}')
        if not MIN_LORA_STRENGTH <= v <= MAX_LORA_STRENGTH:
            raise ValueError(
                f'strength out of range [{MIN_LORA_STRENGTH}, {MAX_LORA_STRENGTH}]: {v}')
        if v not in sts:
            sts.append(v)
    asp = []
    for a in (aspects or []):
        if a in TEST_ASPECTS and a not in asp:
            asp.append(a)
    if not asp:
        asp = [DEFAULT_ASPECT]
    cfs = []
    for v in (cfgs or []):
        try:
            fv = round(float(v), 2)
        except (TypeError, ValueError):
            continue
        if 1.0 <= fv <= 15.0 and fv not in cfs:
            cfs.append(fv)
    if not cfs:
        cfs = [defaults['cfg']]
    if family == 'flux2klein':
        cfs = [1.0]
    sps = []
    for v in (steps_list or []):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= iv <= 50 and iv not in sps:
            sps.append(iv)
    if not sps:
        sps = [defaults['steps']]
    # Optional SDXL steps2 controls its detail pass; None reuses first-pass steps. Z-Image has no second pass.
    sps2 = []
    for v in (steps2_list or []):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= iv <= 50 and iv not in sps2:
            sps2.append(iv)
    if not sps2:
        sps2 = [None]
    if not cps or not sts:
        raise ValueError('at least one checkpoint and one strength are required')
    # No cell cap: the serial queue workload and time estimate are shown before launch.
    return [(c, s, a, cf, sp, sp2)
            for c in cps for s in sts for a in asp for cf in cfs for sp in sps for sp2 in sps2]


def _krea_zero_strength_first(items, run_family, strength_of) -> list:
    """Stable-partition Krea around controls with the tested LoRA switched off.

    Moving every exact-zero tested-LoRA cell before its non-zero counterparts
    avoids returning to a tested-LoRA-free graph after Krea has begun loading
    the tested LoRAs. Both partitions keep their original order
    (base-model/checkpoint/aspect axes included), negative strengths remain in
    the non-zero partition, and other model families are deliberately left
    byte-for-byte ordered as before. Permanent/batch LoRAs remain untouched and
    may still patch a zero-strength control.
    """
    planned = list(items)
    if run_family != 'krea':
        return planned
    zero = []
    nonzero = []
    for item in planned:
        (zero if strength_of(item) == 0.0 else nonzero).append(item)
    return zero + nonzero


# Describe converts an image to a test prompt using Ollama vision; enforce the service upload limit too.
STUDIO_DESCRIBE_MAX_BYTES = 20 * 1024 * 1024

# Describe scene, pose, framing and clothing directly as a generation prompt.
# Do not describe identity or add a trigger: the LoRA supplies identity and
# Studio injects the trigger at workflow construction, keeping stored text raw.
STUDIO_DESCRIBE_PROMPT = (
    "You are writing a TEXT-TO-IMAGE GENERATION PROMPT that would recreate this image.\n\n"
    "ABSOLUTE RULE - never describe WHO the person is. Do not mention identity or any "
    "identity-fixing trait: hair (its length, colour, style, texture), face shape, facial "
    "features, eye colour, eyebrows, nose, lips, jawline, skin tone or texture, freckles, "
    "age, gender, or ethnicity. Refer to a person only as \"the subject\". Do not invent a "
    "name and do not add any trigger word or token.\n\n"
    "DO describe, the way a prompt would: the shot type and framing (close-up, "
    "three-quarter, full-body, wide), the pose and body position, the expression and gaze "
    "as a state (smiling, looking at the viewer, eyes closed), the clothing and accessories "
    "with their colours, the setting or location, and the lighting and mood.\n\n"
    "Output ONE compact paragraph of plain natural-language prose, ready to paste as a "
    "generation prompt, beginning with the shot type and framing. Output only the prompt "
    "itself - no preamble, no \"Here is\", no quotation marks, no commentary.")


def describe_test_prompt(image_bytes: bytes) -> str:
    """Describe an uploaded image into a ready-to-paste Studio TEST PROMPT via the
    Ollama vision model (the same abliterated Qwen3-VL the app captions with, so NSFW
    passes). Resizes to <=1024 long side (like captioning) before the call, force-starts
    a stopped LOCAL Ollama. Whether the model stays resident afterwards is decided by
    CONTENTION (services/vision_keepalive.py): with a generation queued or a training
    running, ComfyUI gets its VRAM back immediately, exactly as before; on an otherwise
    idle card the model is leased warm so describing several images in a row doesn't pay
    the 12.8 s cold load every time. The lease is revoked the moment the queue picks up
    a job.

    Raises ValueError on a missing / oversized / unreadable (non-image) upload, and
    RuntimeError when Ollama is unavailable or rejects the request (its own reason is
    carried straight through via describe_image_ollama's auto_start_local path)."""
    if not image_bytes:
        raise ValueError('no image provided')
    if len(image_bytes) > STUDIO_DESCRIBE_MAX_BYTES:
        raise ValueError(f'image too large (max {STUDIO_DESCRIBE_MAX_BYTES // (1024 * 1024)} MB)')
    try:
        webp = fds.normalize_to_webp(image_bytes, size=1024)
    except Exception as e:
        raise ValueError('unreadable image — expected a webp, png or jpg file') from e
    # The /describe-image route owns the one GPU-exclusive Vision window. Keep
    # this service callable without recursively claiming it a second time.
    from .vision_llm import describe_image as describe_image_ollama
    from .vision_keepalive import keep_alive_for_isolated_call
    text = describe_image_ollama(
        webp, STUDIO_DESCRIBE_PROMPT, num_predict=500, auto_start_local=True,
        keep_alive=keep_alive_for_isolated_call())
    text = (text or '').strip().strip('"').strip()
    if not text:
        raise RuntimeError(
            'The vision model returned an empty description — check the configured '
            'vision model in Settings and the application log.')
    return text


STUDIO_ENHANCE_MAX_CHARS = 4000
# Text-only prompt enrichment must not invent identity or modify trigger words.
# Studio injects the trigger during workflow construction, preventing duplicated
# or corrupted tokens from the language model.
STUDIO_ENHANCE_PROMPT = (
    "You are rewriting a TEXT-TO-IMAGE GENERATION PROMPT so it renders better.\n\n"
    "Keep every subject, action, clothing item, setting and camera choice the author "
    "already wrote — you enrich, you do not replace. Add what a good prompt states and "
    "this one leaves out: shot type and framing, pose, lighting, background, mood, "
    "lens/photographic quality.\n\n"
    "ABSOLUTE RULES - do not describe WHO the person is: no hair, face, eyes, skin, age, "
    "gender or ethnicity (a LoRA supplies the identity). Do not add, repeat, translate or "
    "invent any trigger word, token or name. Do not add negatives, weights, "
    "parentheses-emphasis or LoRA tags.\n\n"
    "Output ONE compact paragraph of plain natural-language prose, ready to paste. Output "
    "only the prompt itself - no preamble, no \"Here is\", no quotation marks, no "
    "commentary.\n\n"
    "PROMPT TO ENHANCE:\n{prompt}")


def enhance_test_prompt(prompt: str, model: str | None = None) -> str:
    """Enrich a Studio test prompt with a LOCAL Ollama text model — by default the same
    abliterated model the app captions with (a vanilla model refuses the NSFW prompts
    this app produces), through the SAME client as captioning (`vision_ollama`); no
    second Ollama seam exists. `model` is the ⚙️ Enhance-options override: that exact
    model is then verified and used instead, and the readiness error names IT — not the
    Settings default the call never touched.

    A stopped LOCAL Ollama is started on demand, exactly like Describe. Whether the
    model stays resident afterwards is decided by contention (vision_keepalive), so
    enhancing three prompts in a row doesn't pay the cold load three times.

    Raises ValueError on an empty/oversized prompt, and RuntimeError when Ollama is
    unreachable, has no usable model, or answers nothing — the caller maps those to
    400 / 409 so the button never fails silently on an install without Ollama."""
    p = (prompt or '').strip()
    if not p:
        raise ValueError('write a prompt first — there is nothing to enhance')
    if len(p) > STUDIO_ENHANCE_MAX_CHARS:
        raise ValueError(f'prompt too long to enhance (max {STUDIO_ENHANCE_MAX_CHARS} characters)')
    from .vision_keepalive import keep_alive_for_isolated_call
    from .vision_llm import ensure_ready, generate_text as generate_text_ollama, label
    ready = ensure_ready(model)
    if not ready.get('ok'):
        # The remedy is not the same word for the two providers: an Ollama model is
        # PULLED, an LM Studio one is LOADED in its app. Saying "load" to an Ollama
        # user was a regression this wave introduced; saying "pull" to an LM Studio
        # user names an action their server does not have.
        if model:
            fix = (' — pick another model from the ✨ Enhance ⚙️ options, or load this '
                   'one in LM Studio first.' if label() == 'LM Studio'
                   else ' — pick another model from the ✨ Enhance ⚙️ options, or pull '
                        'this one first.')
        else:
            fix = (f' — Enhance needs the local {label()} model configured in '
                   'Settings › Local tools.')
        raise RuntimeError((ready.get('error') or f'{label()} is unavailable') + fix)
    text = generate_text_ollama(STUDIO_ENHANCE_PROMPT.format(prompt=p), model=model,
                                num_predict=500,
                                keep_alive=keep_alive_for_isolated_call(), strict=True)
    text = (text or '').strip().strip('"').strip()
    if not text:
        raise RuntimeError(
            f'The model returned an empty prompt — check the configured {label()} model '
            'in Settings and the application log.')
    return text


# --- Workflow build + enqueue -------------------------------------------------
def apply_sdxl_lora_test_settings(workflow, *, base_ckpt, lora_name, strength,
                                  prompt, seed, width, height, cfg=None, steps=None,
                                  steps2=None, batch_size=1, filename_prefix=None,
                                  allowed_bases=None, allowed_loras=None,
                                  detail_amount=None):
    """Configure an SDXL HQ cell in place with an allowlisted base and LoRA.

    Node 1 loads the checkpoint; node 25 applies the tested LoRA. Set prompt,
    seed, dimensions and steps. First-pass node 5 uses steps; detail-pass scheduler
    57 uses steps2 or falls back to steps. Reject non-allowlisted paths."""
    if allowed_bases is not None and base_ckpt not in allowed_bases:
        raise ValueError(f"unknown SDXL checkpoint: {base_ckpt}")
    if allowed_loras is not None and lora_name not in allowed_loras:
        raise ValueError(f"unknown SDXL LoRA: {lora_name}")

    def _set(node_id, key, value):
        n = workflow.get(node_id)
        if isinstance(n, dict) and key in n.get("inputs", {}):
            n["inputs"][key] = value

    # Resolve the checkpoint basename to the relative path required by ComfyUI, including subfolders.
    _set("1", "ckpt_name", resolve_checkpoint_ckpt_name(base_ckpt))
    _set("25", "lora_name", lora_name)
    _set("25", "strength_model", float(strength))
    _set("25", "strength_clip", float(strength))
    _set("3", "text", prompt)
    _set("5", "seed", int(seed))
    if steps is not None:
        _set("5", "steps", int(steps))          # pass 1 (KSampler)
    # pass 2 (detail daemon, node 57): use steps2 when provided, otherwise steps.
    _pass2 = steps2 if steps2 is not None else steps
    if _pass2 is not None:
        _set("57", "steps", int(_pass2))
    if cfg is not None:
        _set("5", "cfg", float(cfg))
    _set("6", "width", int(width))
    _set("6", "height", int(height))
    _set("6", "batch_size", int(batch_size))
    # Find DetailDaemonSamplerNode by class. The slider is effective detail with
    # fade=0; clamp to [0,1], preserving workflow defaults for None. SDXL-safe values
    # are approximately 0-0.25.
    if detail_amount is not None:
        try:
            _da = max(0.0, min(1.0, float(detail_amount)))
        except (TypeError, ValueError):
            _da = None
        if _da is not None:
            for _n in workflow.values():
                if (isinstance(_n, dict) and _n.get("class_type") == "DetailDaemonSamplerNode"
                        and "detail_amount" in _n.get("inputs", {})):
                    _n["inputs"]["detail_amount"] = _da
    if filename_prefix is not None:
        _set("9", "filename_prefix", filename_prefix)


def _resolve_lora_rel_by_basename(basename):
    """Loras-root-relative name (the exact string a LoraLoader wants) of the FIRST
    file whose basename matches `basename` across every loras search root, or None.
    Lets a workflow-wired accelerator LoRA be found WHEREVER the user keeps it —
    root, a differently-named subfolder, an extra_model_paths loras root — instead of
    depending on the developer's own subfolder. Reuses the same disk view as the
    picker/probe (comfy_model_paths.list_models), so anything the app lists is
    resolvable here; [] with no ComfyUI configured → None."""
    from . import comfy_model_paths
    target = (basename or '').lower()
    if not target:
        return None
    for rel, _ab in comfy_model_paths.list_models('loras'):
        if os.path.basename(rel).lower() == target:
            return rel
    return None


def _bypass_lora_loader(workflow, node_id):
    """Delete a two-output LoraLoader (model + clip) and reconnect each consumer of
    its model output (slot 0) / clip output (slot 1) to that node's OWN upstream model
    / clip inputs, so ComfyUI never fails validation on the missing LoRA. The two-slot
    form of klein_edit_helper._bypass_node (which handles a single model output)."""
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        return
    upstream = {0: node.get('inputs', {}).get('model'),
                1: node.get('inputs', {}).get('clip')}
    for other in workflow.values():
        if not isinstance(other, dict):
            continue
        for k, v in list(other.get('inputs', {}).items()):
            if (isinstance(v, list) and len(v) == 2 and v[0] == node_id
                    and upstream.get(v[1]) is not None):
                other['inputs'][k] = upstream[v[1]]
    workflow.pop(node_id, None)


def _apply_sdxl_accelerator(workflow):
    """Make the SDXL HQ workflow's DMD2 accelerator LoRA independent of the dev's own
    ComfyUI layout. The template wires 'DMD2\\dmd2_sdxl_4step_lora_fp16.safetensors' (a
    personal subfolder) at strength 1.0 — a SPEED/quality accelerator, NOT a
    graph-critical asset like the base checkpoint / VAE / text encoder. So:
      * resolve it by canonical basename across every loras root (a user who keeps the
        public DMD2 LoRA under any other folder still gets it wired — mission's #1
        preference), and
      * BYPASS the loader when it is absent EVERYWHERE, so a fresh SDXL Studio degrades
        to a plain render instead of hard-blocking the whole family on a file that only
        exists on the dev's disk (mirrors the Klein node-139 bypass).
    Distilled base checkpoints (the workflow's design point) render unchanged without
    it; a full SDXL checkpoint renders softer — the honest trade-off vs. a blocked grid.
    Idempotent and shape-agnostic: matches the DMD2 loader by `lora_name`, not node id."""
    for nid, node in list(workflow.items()):
        if not isinstance(node, dict) or node.get('class_type') != 'LoraLoader':
            continue
        ref = str(node.get('inputs', {}).get('lora_name') or '')
        if 'dmd2' not in ref.lower():
            continue
        # Fast path: the wired path is already on disk (dev, or anyone who placed it
        # there) → leave it, skip the loras walk that a grid would repeat per cell.
        if _resolve_lora_abs_path(ref):
            break
        found = _resolve_lora_rel_by_basename(os.path.basename(ref.replace('\\', '/')))
        if found:
            node['inputs']['lora_name'] = found
        else:
            _bypass_lora_loader(workflow, nid)
        break
    return workflow


# WHAT THE « Official » ENTRY LOADS, AND WHY IT IS NO LONGER A FILENAME
# ---------------------------------------------------------------------
# It used to be one: `krea2_turbo_fp8.safetensors`, the basename frozen into
# krea2_turbo.json's node 20 and repeated here as a constant. Two defects.
#
#   * It is NOT the file Setup installs. `setup_installer` fetches Comfy-Org's own
#     `krea2_turbo_fp8_scaled.safetensors`; ComfyUI validates a loader widget by
#     exact string match against the list it publishes, and those two names are
#     not the same string. So on an install that simply followed Setup, the
#     Studio's default base named a file that is not there and the whole prompt
#     was refused ("Value not in list: unet_name") before a step ran. It worked
#     only on machines that happened to have that community repack.
#   * That repack carries tensors this family does not declare. Measured on the
#     real header: 432 tensors against the family's 430, the two extras being
#     `last.down.weight` / `last.up.weight` `[6144, 6144]`, which its own
#     `__metadata__` (`egg_format`, `egg_w`, `egg_h`, `egg_c`) describes as an
#     embedded image — ~75 MB of picture shipped inside a base model.
#
# So the default is ELECTED from what is on disk, by the ranking documented on
# `krea_edit_helper.resolve_krea_unet` and shared with it — the Generate resolver
# and this picker must never elect different files out of the same folder.

def krea_default_base():
    """ComfyUI-relative name of the base the « Official » entry loads, or None
    when nothing on disk qualifies — then node 20 keeps the workflow's own value,
    which is the historical behaviour and the case the missing-asset preflight
    already owns.

    Elected out of `get_krea_models()`, the very list this screen offers: ranking
    a name the picker does not list would elect a base its own whitelist refuses."""
    try:
        from .krea_edit_helper import elect_krea_base
        return elect_krea_base(get_krea_models())
    except Exception:                       # noqa: BLE001 — never fatal to a render
        logger.exception('Krea default base election failed')
        return None


def krea_default_base_entry() -> dict:
    """The « Official » row of the Krea base pickers: ``{value, label, note, source}``.

    ``value`` stays ``''`` forever — it is persisted on run rows and read back, so
    it is an id, not a label. What CHANGES with the disk is what the row says: the
    name stays "Official" only while the elected base IS the file Setup installs.
    Anything else is named for what it is, so a base nobody chose is never
    presented as the official one.

    ``source`` is the file that row will actually load (None when node 20 keeps the
    workflow's own value). Callers need it because the sampler defaults belong to
    THAT file: a Raw build elected as the default must not be offered with the
    Turbo numbers — cfg 1 / 8 steps on an undistilled base renders a blurry sketch
    people read as a failed training (GitHub #18, bobba84, on the Z-Image side)."""
    from .krea_edit_helper import KREA_ASSETS, KREA_CANONICAL_UNET
    entry = {'value': '', 'label': 'Official – Krea 2 Turbo', 'note': None,
             'source': None}
    elected = krea_default_base()
    if not elected:
        return entry
    entry['source'] = elected
    bare = _basename(elected)
    if bare.lower() == KREA_CANONICAL_UNET.lower():
        return entry
    entry['label'] = f'Default – {bare.rsplit(".", 1)[0]}'
    notes = [f'The Krea 2 Turbo base Setup installs ({KREA_CANONICAL_UNET}) is not '
             f'on this machine, so the Studio renders on the best Krea 2 build it '
             f'found here: {bare}.']
    health = _krea_base_health(elected)
    if health and health.get('note'):
        notes.append(health['note'])
    # A note that only DIAGNOSES leaves the reader with a fact and no gesture, so
    # it ends on the action and on where the file lands — the same folder every
    # other Krea message names.
    notes.append(f'To render on the official base instead: Setup ▸ Install ▸ Krea 2 '
                 f'downloads {KREA_CANONICAL_UNET} into '
                 f'{KREA_ASSETS["krea_model"]["path"]}, and this entry goes back to '
                 f'it on its own — nothing else to change.')
    entry['note'] = ' '.join(notes)
    return entry


def _krea_base_health(rel_name):
    """`model_integrity.base_health` for an elected base, but ONLY when the verdict
    is worth a sentence — a file that announces it carries something other than
    weights. A plain fp8 cast is a normal thing to render on and says nothing here
    (the TRAINING picker is where precision earns a warning). None when the file
    cannot be resolved or read."""
    try:
        from . import comfy_model_paths, model_integrity
        path = comfy_model_paths.resolve_model_file('diffusion_models', rel_name)
        if not path:
            return None
        health = model_integrity.base_health(path)
        return health if health['rank'] == model_integrity.HEALTH_FOREIGN_PAYLOAD else None
    except Exception:                       # noqa: BLE001 — advisory only
        return None


def krea_alt_base_models() -> list:
    """Local Krea alternatives to the "Official" base: checkpoints from
    get_krea_models() excluding the elected default. An empty list leaves the
    selectors hidden, preserving the historical behavior.

    Compare BASENAMES: a base copied into both the root and a subdirectory must
    not appear twice under different labels."""
    default = krea_default_base()
    bare = _basename(default).lower() if default else None
    return [m for m in get_krea_models() if not bare or _basename(m).lower() != bare]


def apply_klein_lora_test_settings(workflow, *, lora_name, strength, prompt, seed,
                                   width, height, cfg=None, steps=None, batch_size=1,
                                   filename_prefix=None, allowed_loras=None,
                                   base_model=None, allowed_bases=None):
    """Configure a test cell using the FLUX.2 Klein text-to-image workflow.

    Elsewhere Klein edits images (variations, improve, inpaint). This path alone
    generates from a prompt: Test Studio compares checkpoints using identical
    prompts and seeds without a source image. Its graph therefore uses
    EmptyFlux2LatentImage and entirely vanilla nodes, unlike the Krea path
    (see test_workflow_portability).

    Resolve the UNET, text encoder and VAE through the same functions as the
    other Klein paths. Names embedded in shipped JSON are only valid on the
    machine where the workflow was captured, which previously made other graphs
    misrepresent their assets.

    Load the tested LoRA in model-only mode (node 29): character LoRAs produced
    by ai-toolkit for Klein train only the transformer.

    `base_model` overrides the elected local Klein UNET, just as for Krea/SDXL/
    Z-Image. Validate it against `allowed_bases` to prevent path injection, as
    for the LoRA. Clamp cfg to 1.0: Klein 9B is guidance-distilled and diverges
    above that value."""
    if allowed_loras is not None and lora_name not in allowed_loras:
        raise ValueError(f'unknown Klein LoRA: {lora_name}')
    if base_model and allowed_bases is not None and base_model not in allowed_bases:
        raise ValueError(f'unknown Klein base model: {base_model}')

    def _set(node_id, key, value):
        n = workflow.get(node_id)
        if isinstance(n, dict) and key in n.get('inputs', {}):
            n['inputs'][key] = value

    from . import klein_edit_helper as keh
    unet = base_model or keh.unet_for_job()
    if unet:
        _set('20', 'unet_name', unet)
        _set('20', 'weight_dtype', keh._unet_weight_dtype(unet))
    te = keh.resolve_klein_text_encoder()
    if te:
        _set('21', 'clip_name', te)
    vae = keh.resolve_klein_vae()
    if vae:
        _set('22', 'vae_name', vae)

    _set('29', 'lora_name', lora_name)
    _set('29', 'strength_model', float(strength))
    _set('23', 'text', prompt)
    for node in ('25', '31'):                    # latent ET ModelSamplingFlux
        _set(node, 'width', int(width))
        _set(node, 'height', int(height))
    _set('25', 'batch_size', int(batch_size))
    _set('26', 'seed', int(seed))
    if steps is not None:
        _set('26', 'steps', max(1, min(50, int(steps))))
    # Do not guard with `if cfg is not None`: this value comes from a Studio
    # sweep axis. Allowing cfg > 1 on a guidance-distilled model would produce
    # burnt-out cells that users could mistake for a bad checkpoint.
    _set('26', 'cfg', 1.0)
    if filename_prefix is not None:
        _set('28', 'filename_prefix', filename_prefix)
    return workflow


def krea_hires_settings() -> dict:
    """{'scale', 'steps', 'denoise'} for the optional second Krea pass, read from
    `krea_hires.*`. Ready to splat onto `apply_krea_lora_test_settings`.

    A scale that does not enlarge comes back as `scale: None`, not 1.0: None is
    the OFF state `inject_krea_hires_fix` understands as "add no node", so ONE
    place decides what off looks like and a neutral value can never turn into a
    phantom pass. A malformed setting degrades to OFF rather than to some
    arbitrary factor — this is read once per cell, and a bad config quietly
    quadrupling the cost of a whole grid is the expensive kind of wrong.

    `steps` 0/absent -> None -> pass 2 inherits pass 1's count."""
    from ..utils.comfyui import KREA_HIRES_MAX_SCALE, KREA_HIRES_DENOISE
    # Every dial is read whatever `scale` says: a run that arms the pass from the
    # panel while the global switch is off (the shipped default) still owes the
    # configured rewrite and step count — only `scale` carries the OFF state.
    try:
        scale = float(cfg.get('krea_hires.scale'))
    except (TypeError, ValueError):
        scale = 1.0
    # NaN first: every comparison against it is False, so `scale <= 1.0` would
    # let it through and the clamp below would hand back the CEILING. A corrupt
    # setting has to mean OFF, never "on, at maximum".
    scale_off = (not math.isfinite(scale)) or scale <= 1.0
    try:
        denoise = float(cfg.get('krea_hires.denoise'))
    except (TypeError, ValueError):
        denoise = KREA_HIRES_DENOISE
    if not math.isfinite(denoise):
        denoise = KREA_HIRES_DENOISE
    denoise = max(0.05, min(1.0, denoise))
    # int() of an infinite float raises OverflowError — neither TypeError nor
    # ValueError — so the finiteness check has to come BEFORE the conversion, or
    # a corrupt value takes every Krea run down with a 500 instead of degrading.
    try:
        steps_raw = float(cfg.get('krea_hires.steps') or 0)
    except (TypeError, ValueError):
        steps_raw = 0.0
    steps = int(steps_raw) if math.isfinite(steps_raw) else 0
    return {'scale': None if scale_off else min(KREA_HIRES_MAX_SCALE, scale),
            'steps': steps if steps > 0 else None,
            'denoise': denoise}


def krea_hires_defaults() -> dict:
    """The `krea_hires.*` setting as NUMBERS, for the Studio panel to display as
    its "Settings default". Distinct from `krea_hires_settings()`, whose OFF
    comes back as `scale: None` for the injector: a panel needs the 1.0 to show
    "off", not a None it would have to translate. Malformed -> the shipped
    default, never a crash on a page that only wants to print a label."""
    out = {}
    for key, shipped in (('scale', 1.0), ('denoise', KREA_HIRES_DENOISE), ('steps', 0)):
        try:
            v = float(cfg.get(f'krea_hires.{key}'))
        except (TypeError, ValueError):
            v = shipped
        out[key] = v if math.isfinite(v) else shipped
    out['steps'] = int(out['steps'])
    return out


def _krea_hires_for_cell(hires_scale, hires_denoise) -> dict:
    """The three `hires_*` kwargs for ONE cell: the run's own values when the
    panel sent some, the `krea_hires.*` setting otherwise.

    Per-run `hires_scale` 1.0 is an explicit OFF — it wins over a setting that
    says 1.5, because that is what "off for this run" means. None (the panel
    left it alone) defers to the setting, which is the "starting point, not a
    lock" contract every other per-run knob on this panel follows. `steps` has
    no per-run form and always follows the setting."""
    base = krea_hires_settings()
    if hires_scale is not None:
        try:
            v = float(hires_scale)
        except (TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v):
            base['scale'] = None if v <= 1.0 else min(KREA_HIRES_MAX_SCALE, v)
    if hires_denoise is not None and base['scale'] is not None:
        try:
            v = float(hires_denoise)
        except (TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v):
            base['denoise'] = max(0.05, min(1.0, v))
    return {f'hires_{k}': v for k, v in base.items()}


def apply_krea_lora_test_settings(workflow, *, lora_name, strength, prompt, seed,
                                  width, height, cfg=None, steps=None, batch_size=1,
                                  filename_prefix=None, allowed_loras=None, extra_loras=None,
                                  sampler=None, scheduler=None,
                                  weight_dtype=None,
                                  base_model=None, allowed_bases=None,
                                  sampler_preset=None, hires_scale=None,
                                  hires_steps=None, hires_denoise=None, negative=None):
    """Configure a Krea 2 Turbo test cell: inject the tested LoRA after the
    UNETLoader (node 20 -> KSampler node 26), plus prompt/negative/seed/dims/steps/cfg.
    `extra_loras` are additional always-on style/utility LoRAs in the same chain,
    applied unchanged to this cell outside the batch axis. Krea has one pass
    (no steps2).

    Mutate in place. Raise ValueError if the tested LoRA is outside its allowlist
    (path-injection protection).

    `base_model` overrides the workflow's default local Krea UNET in node 20,
    using the same mechanism as SDXL (`base_ckpt`) / Z-Image (`z_model`). None
    leaves the node unchanged. Validate against `allowed_bases`, as for the LoRA."""
    if allowed_loras is not None and lora_name not in allowed_loras:
        raise ValueError(f"unknown Krea LoRA: {lora_name}")
    if base_model and allowed_bases is not None and base_model not in allowed_bases:
        raise ValueError(f"unknown Krea base model: {base_model}")

    def _set(node_id, key, value):
        n = workflow.get(node_id)
        if isinstance(n, dict) and key in n.get("inputs", {}):
            n["inputs"][key] = value

    if base_model:
        _set("20", "unet_name", base_model)
    else:
        # « Official » : elect the base rather than trust the filename frozen into
        # the workflow JSON — see krea_default_base for what that literal actually
        # named. Server-elected, so it does not go through `allowed_bases` (that
        # whitelist guards a USER-supplied value against path injection). None →
        # node untouched, exactly as before.
        elected = krea_default_base()
        if elected:
            _set("20", "unet_name", elected)

    _set("23", "text", prompt)                    # prompt (CLIPTextEncode Krea)
    if negative:
        # Node 24 feeds the KSampler's negative input. ComfyUI only evaluates it
        # above CFG 1, so a Turbo cell at CFG 1 renders exactly as before.
        _set("24", "text", str(negative))
    _set("25", "width", int(width))
    _set("25", "height", int(height))
    _set("25", "batch_size", int(batch_size))
    _set("26", "seed", int(seed))
    if steps is not None:
        _set("26", "steps", max(1, min(50, int(steps))))
    if cfg is not None:
        _set("26", "cfg", max(1.0, min(10.0, float(cfg))))
    # Validate sampler/scheduler (node 26) and UNET precision (node 20) against
    # the same allowlists as generation; ignore unknown values to prevent injection.
    if sampler in KREA_ALLOWED_SAMPLERS:
        _set("26", "sampler_name", sampler)
    if scheduler in KREA_ALLOWED_SCHEDULERS:
        _set("26", "scheduler", scheduler)
    if weight_dtype in KREA_ALLOWED_WEIGHT_DTYPES:
        _set("20", "weight_dtype", weight_dtype)
    if filename_prefix is not None:
        _set("28", "filename_prefix", filename_prefix)
    # Tested and always-on LoRAs share one node 20 -> 26 chain, as in Krea
    # generation. `allowed` contains the entire Krea pool, including always-on LoRAs.
    from ..utils.comfyui import inject_krea_loras
    requested = [{"filename": lora_name, "strength": float(strength)}]
    for e in (extra_loras or []):
        fn = str((e or {}).get("filename") or "")
        if not fn:
            continue
        try:
            st = float(e.get("strength", 1.0))
        except (TypeError, ValueError):
            st = 1.0
        requested.append({"filename": fn, "strength": st})
    allowed = set(allowed_loras) if allowed_loras is not None else {r["filename"] for r in requested}
    if allowed_loras is not None:
        # `allowed_loras` is a FAMILY-POOL scan (krea/ subfolder only): always-on
        # AND external (Canvas plugin node) entries in `extra_loras` were already
        # validated (path-injection / fail-closed) before reaching here, so this
        # whitelist must not re-filter them out — the same silent-drop bug the
        # Z-Image path guards against at `allowed_loras=(set(...) | {...})` above.
        # Without this union, `inject_krea_loras` below drops every extra whose
        # filename lives outside krea/ with NO error: persisted on the cell's
        # JSON, never mounted in the graph.
        allowed |= {r["filename"] for r in requested[1:]}
    inject_krea_loras(workflow, requested, allowed=allowed)
    # Apply hi-res fix (second latent pass) after injecting LoRAs. Pass two
    # clones pass one, including `model`; cloning earlier would silently connect
    # it to the bare UNETLoader and omit LoRAs. Apply it before the sampler preset,
    # which redirects consumers of the replaced node, now LatentUpscaleBy. This
    # lets both transformations compose without knowing about each other.
    # hires_scale None or <= 1 disables it: add no nodes and preserve rendering.
    from ..utils.comfyui import inject_krea_hires_fix
    inject_krea_hires_fix(workflow, hires_scale, steps=hires_steps,
                          denoise=hires_denoise)
    # Preset sampler LAST, and it has to stay last: it reads KSampler.model, which
    # both injections above rewrite. Ahead of them it would wire the guider to the
    # bare UNETLoader and drop the whole stack, silently. `sampler_preset` None/off
    # (the default) leaves the KSampler exactly as it is — which is also what keeps
    # the shipped node OPTIONAL: no preset, no class in the graph, nothing to
    # install. Pinned by test_krea_preset_sampler_contract.
    from ..utils.comfyui import inject_krea_preset_sampler
    inject_krea_preset_sampler(workflow, sampler_preset)


# --- Node-class resolution (variant-tolerant custom nodes) --------------------
# Some ComfyUI custom nodes register under DIFFERENT class names across installs
# (a pack rename, a fork, a locally-edited copy). Our workflow JSON can only carry
# ONE class string, so a target ComfyUI that has the node under another name would
# fail the preflight (409 "install pack X") AND fail every tile if enqueued — even
# though the capability is right there. NODE_CLASS_ALIASES maps the CANONICAL class
# (the exact string our templates carry) to the alternative name(s) the SAME node
# is known to register as. Consumed by BOTH the preflight (a required class counts
# as present when the canonical OR any alias is in /object_info) and the cell builder
# (rewrites a node's class_type to whichever variant the target actually exposes, so
# the enqueued graph validates). Add an entry only for a node we have SEEN register
# under two names — never a speculative alias.
NODE_CLASS_ALIASES = {
    # No SHIPPED graph names this class any more (the Krea rebalance was retired on
    # 2026-09-02 — see utils/comfyui, "Retired"). Kept because it is the one alias
    # we have actually SEEN: the published pack (nova452/ComfyUI-Conditioning-
    # Rebalance) registers ConditioningKrea2Rebalance, some installs register the
    # very same node as Krea2RebalanceConditioning — and a user's own template or a
    # Canvas plugin graph may still carry it. The resolver stays generic.
    'ConditioningKrea2Rebalance': ('Krea2RebalanceConditioning',),
}


def _node_class_present(class_type, available):
    """True when `available` (a set of /object_info class names) exposes `class_type`
    OR any of its known aliases (NODE_CLASS_ALIASES). Presence only — the caller
    decides what a miss means; pass a real set (never None) so 'probe failed' stays a
    separate, fail-open decision at the call site."""
    if class_type in available:
        return True
    return any(alt in available for alt in NODE_CLASS_ALIASES.get(class_type, ()))


def _resolve_workflow_node_classes(workflow, available):
    """Rewrite each node whose CANONICAL class_type is absent from `available` but a
    known ALIAS is present, to that alias — so the ENQUEUED graph names the class the
    target ComfyUI actually registers (the resolver philosophy, applied to node
    classes). No-op when `available` is falsy (probe failed / not threaded → fail
    open: keep the canonical name, the preflight/per-tile path still reports a true
    miss) or when the canonical class is already present. Mutates in place; returns
    `workflow`."""
    if not available:
        return workflow
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type')
        aliases = NODE_CLASS_ALIASES.get(ct)
        if not aliases or ct in available:
            continue
        for alt in aliases:
            if alt in available:
                node['class_type'] = alt
                break
    return workflow


def _target_node_classes():
    """The target ComfyUI's /object_info class set, fetched ONCE per run so the grid's
    per-cell class resolution doesn't re-pull the (large) payload per tile. None when
    ComfyUI can't be reached — callers fail open (keep canonical names)."""
    from ..utils.comfyui import fetch_object_info_classes
    return fetch_object_info_classes()


def _build_cell_workflow(user_id, checkpoint, strength, prompt, seed, z_model,
                         allowed_loras, width=TEST_WIDTH, height=TEST_HEIGHT,
                         cfg=None, steps=None, steps2=None, dataset_id=None, train_type='zimage',
                         extra_loras=None, negative=None, sampler=None,
                         scheduler=None, weight_dtype=None,
                         detail_amount=None, trigger_word=None, available_classes=None,
                         sampler_preset=None, hires_scale=None, hires_denoise=None):
    """Load the ZTurbo (Z-Image) / HQ (SDXL) / Krea workflow and configure a cell.
    `extra_loras` are always-on style/utility LoRAs applied to this cell alongside
    the tested checkpoint, outside the batch axis. Raise ValueError if the
    workflow file cannot be loaded.

    `available_classes` is the target ComfyUI /object_info class set (from
    `_target_node_classes()`, fetched once per run). When provided, rewrite
    variant custom-node classes to the registered names (NODE_CLASS_ALIASES),
    so the graph validates on installations using alternative names. None keeps
    the canonical names (fail open).

    filename_prefix includes dataset_id and a short UUID per cell. Without
    these, ComfyUI's counter (reset on restart) produced identical filenames
    across datasets (`{uid}_LoraTest_00022_`), causing browser-cache collisions
    and showing one LoRA's images in another's studio. The UUID also prevents
    collisions within a dataset when rerunning after a ComfyUI restart."""
    # Inject trigger words here, only when building the graph; retain the raw
    # prompt in storage. `trigger_word` may be a list, one per stacked LoRA, each
    # spelled as its LoRA learned it (_inject_triggers).
    prompt = _inject_triggers(prompt, checkpoint, trigger_word, extra_loras)
    ds_tag = f"d{dataset_id}_" if dataset_id is not None else ""
    fname = f"{user_id}_{ds_tag}LoraTest_{uuid.uuid4().hex[:8]}"
    extra_loras = extra_loras or []
    family = (train_type or 'zimage').lower()
    if family in TRAINED_IMAGE_FAMILIES:
        from ..utils.trained_image_workflows import build_trained_image_workflow
        from .trained_image_models import resolve_family_assets
        bases = family_base_models(family)
        assets = resolve_family_assets(family, z_model)
        return build_trained_image_workflow(
            family, base_model=z_model, assets=assets,
            loras=[{'filename': checkpoint, 'strength': strength}] + list(extra_loras),
            allowed_bases=set(bases),
            allowed_loras=set(allowed_loras) | {e['filename'] for e in extra_loras},
            prompt=prompt, negative=negative or '', seed=seed,
            width=width, height=height, cfg=cfg, steps=steps,
            sampler=sampler, scheduler=scheduler, weight_dtype=weight_dtype,
            batch_size=1, filename_prefix=fname, available_classes=available_classes)
    if (train_type or 'zimage').lower() == 'sdxl':
        workflow = load_workflow_local(str(WORKFLOW_HQ_PATH))
        if not workflow:
            raise ValueError('HQ workflow not found/unreadable')
        from ..utils.comfyui import get_checkpoint_models, inject_sdxl_loras
        allowed_bases = {m.get('name') for m in get_checkpoint_models() if m.get('name')}
        allowed_sdxl_loras = {l['filename'] for l in get_sdxl_loras()}
        # As in normal SDXL generation, set sampler/scheduler/cfg and toggle DMD2
        # according to the base: on for DMD-distilled checkpoints such as bigLove/mop,
        # off for full SDXL. Otherwise output breaks. Apply before test injection so
        # Studio cfg/steps axes can override these defaults.
        apply_optimal_sampler_params(workflow, z_model)
        apply_sdxl_lora_test_settings(
            workflow, base_ckpt=z_model, lora_name=checkpoint, strength=strength,
            prompt=prompt, seed=seed, width=width, height=height, cfg=cfg, steps=steps,
            steps2=steps2, batch_size=1, filename_prefix=fname,
            allowed_bases=allowed_bases, allowed_loras=allowed_sdxl_loras,
            detail_amount=detail_amount,
        )
        if extra_loras:  # always-on entries chained after Style LoRA (node 25)
            inject_sdxl_loras(workflow, extra_loras, {e['filename'] for e in extra_loras})
        # DMD2 accelerator (node 10): resolve across loras roots or bypass when absent,
        # so the SDXL Studio never depends on the dev's personal 'DMD2\' subfolder nor
        # hard-blocks on a quality-only accelerator. Runs LAST so it reads node 10's
        # real upstream after any always-on chaining.
        _apply_sdxl_accelerator(workflow)
        return _resolve_workflow_node_classes(workflow, available_classes)
    if (train_type or 'zimage').lower() == 'krea':
        workflow = load_workflow_local(str(WORKFLOW_KREA_TURBO_PATH))
        if not workflow:
            raise ValueError('Krea workflow not found/unreadable')
        allowed_krea = {l['filename'] for l in get_krea_loras()}
        apply_krea_lora_test_settings(
            workflow, lora_name=checkpoint, strength=strength, prompt=prompt,
            seed=seed, width=width, height=height, cfg=cfg, steps=steps,
            batch_size=1, filename_prefix=fname, allowed_loras=allowed_krea,
            extra_loras=extra_loras,
            sampler=sampler, scheduler=scheduler, weight_dtype=weight_dtype,
            sampler_preset=sampler_preset, negative=negative,
            # Optional local Krea base (z_model, as for SDXL/Z-Image). None keeps the
            # workflow UNET. The disk-scan allowlist prevents path injection.
            base_model=z_model, allowed_bases=set(get_krea_models()),
            # Hi-res fix uses the explicit run setting, otherwise the global
            # krea_hires.* setting. Resolve here, in the builder shared by the Studio
            # grid, Canvas and resume, so run overrides and Settings defaults are
            # arbitrated once. Default off preserves the existing graph.
            **_krea_hires_for_cell(hires_scale, hires_denoise),
        )
        # Resolve NODE_CLASS_ALIASES against the target ComfyUI: rewrite class_type
        # to the registered variant so the queued graph validates. When
        # available_classes is None, keep the canonical name.
        return _resolve_workflow_node_classes(workflow, available_classes)
    if (train_type or 'zimage').lower() == 'flux2klein':
        workflow = load_workflow_local(str(WORKFLOW_FLUX2KLEIN_PATH))
        if not workflow:
            raise ValueError('FLUX.2 Klein workflow not found/unreadable')
        from ..utils.comfyui import get_flux2_klein_models
        apply_klein_lora_test_settings(
            workflow, lora_name=checkpoint, strength=strength, prompt=prompt,
            seed=seed, width=width, height=height, cfg=cfg, steps=steps,
            batch_size=1, filename_prefix=fname,
            allowed_loras=set(allowed_loras),
            base_model=z_model,
            allowed_bases={m['filename'] if isinstance(m, dict) else m
                           for m in get_flux2_klein_models()},
        )
        # Blend and permanent LoRAs belong to the same graph as the tested head.
        model_ref = ['29', 0]
        for index, entry in enumerate(extra_loras):
            name = f'lds_klein_lora_{index}'
            workflow[name] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
                'model': model_ref, 'lora_name': entry['filename'],
                'strength_model': float(entry.get('strength', 1.0))}}
            model_ref = [name, 0]
        workflow['31']['inputs']['model'] = model_ref
        return _resolve_workflow_node_classes(workflow, available_classes)
    # Close this file's third silent Z-Image fallback, for the same reason as
    # the others: an unsupported family previously inherited another workflow,
    # such as a Z-Image graph loading a Klein LoRA. Base dispatchers already
    # reject this upstream; this guard prevents new families slipping through.
    if not can_generate_with(train_type):
        raise _no_generation_lane(train_type)
    workflow = load_workflow_local(str(WORKFLOW_ZTURBO_PATH))
    if not workflow:
        raise ValueError('ZTurbo workflow not found/unreadable')
    apply_zimage_settings(
        workflow,
        z_model=z_model,
        z_loras=[{'filename': checkpoint, 'strength': strength}] + list(extra_loras),
        prompt=prompt,
        negative=negative,
        seed=seed,
        width=width, height=height, batch_size=1,
        z_cfg=cfg, z_steps=steps,
        filename_prefix=fname,
        # Include always-on LoRAs in the allowlist or inject_zimage_loras would filter them out.
        allowed_loras=(set(allowed_loras) | {e['filename'] for e in extra_loras}) if extra_loras else allowed_loras,
    )
    return _resolve_workflow_node_classes(workflow, available_classes)


def _enqueue_cell(user_id, dataset_id, workflow, prompt, job_id=None, commit=True,
                  *, cell_id=None, run_id=None) -> str:
    """Enqueue one serialized Test Studio cell with durable cell identity.

    ``job_id`` is minted before the cell insert. ``cell_id`` / ``run_id`` are
    deliberately copied into queue metadata once the cell has an id, so a paused
    ComfyUI prompt can be shown and recovered without guessing which grid tile it
    belongs to. ``commit=False`` retains the one-transaction cell + queue insert.
    """
    job_id = job_id or str(uuid.uuid4())
    metadata = {
        'model_name': 'zimage_lora_test',
        'is_lora_test': True,
        'dataset_id': dataset_id,
    }
    if cell_id is not None:
        metadata['cell_id'] = int(cell_id)
        cell = db.session.get(LoraTestImage, cell_id)
        if cell is not None:
            dataset = db.session.get(FaceDataset, cell.dataset_id)
            metadata['family'] = (family_of_lora(cell.checkpoint)
                                  or getattr(dataset, 'train_type', None) or 'zimage')
            metadata['base_model'] = cell.z_model
    if run_id:
        metadata['run_id'] = str(run_id)
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=prompt, job_id=job_id,
                          metadata=metadata, commit=commit)
    return job_id


def _persist_and_enqueue_cell(img, user_id, dataset_id, prompt, build_workflow) -> str:
    """Insert ONE grid cell and its queue job in a SINGLE transaction, and return
    its job_id.

    Why one commit and not zero (a single commit for the whole grid): a grid is
    enqueued cell by cell and an enqueue failure at cell 20/50 must LEAVE the 19
    already-queued cells in the database — a batch commit would roll their rows back
    while their jobs stay in the queue (orphan jobs, ghost tiles). Why not three
    (the historical shape: insert row, enqueue, re-write row with its job_id): each
    commit takes SQLite's write lock, and a 50-cell grid firing 150 of them back to
    back is exactly the profile that starves a concurrent writer into
    'database is locked'.

    The workflow is built before taking the GPU arbiter: resolving files/nodes
    must not hold local GPU scheduling.  The arbiter is then acquired *before*
    the first cell write and retained through the cell + queue commit.  This
    keeps the only safe lock order (GPU_ARBITER_LOCK -> SQLite transaction) and
    prevents a recovery barrier from appearing after ``add_job(commit=False)``
    checked readiness but before this transaction becomes durable.

    On failure the half-built transaction is rolled back (dropping the queue row that
    may already have been staged) and the cell is re-inserted as 'failed' with the
    reason, so the caller's `raise` still surfaces a visible, explained tile."""
    job_id = str(uuid.uuid4())
    img.job_id = job_id

    try:
        workflow = build_workflow()
    except Exception as e:
        # Workflow construction has not touched the queue transaction.  Roll
        # back any read/autoflush side effect from a builder before recording
        # the explained failure marker.
        db.session.rollback()
        img.job_id = None
        img.status = 'failed'
        img.error = str(e)[:400] or 'enqueue failed'   # say WHY, not a mute red tile
        db.session.add(img)
        db.session.commit()
        raise

    with GPU_ARBITER_LOCK:
        db.session.add(img)
        try:
            # A flush is not a commit: it gives the queue metadata the exact cell id
            # while preserving the one-transaction insert invariant below.
            db.session.flush()
            _enqueue_cell(user_id, dataset_id, workflow, prompt, job_id=job_id,
                          commit=False, cell_id=img.id, run_id=img.run_id)
            db.session.commit()
        except Exception as e:
            # rollback expunges the pending cell + job rows; the cell object goes back
            # to transient and can be re-added as the failed marker.  Keep the outer
            # arbiter through this replacement commit too: never acquire it after a
            # SQLite write transaction has begun.
            db.session.rollback()
            img.job_id = None
            img.status = 'failed'
            img.error = str(e)[:400] or 'enqueue failed'
            db.session.add(img)
            db.session.commit()
            raise
    return job_id


def _sanitize_gen_knobs(run_family, *, negative=None, sampler=None, scheduler=None,
                        weight_dtype=None,
                        detail_amount=None, resolution_tier=None, resolution_multiplier=None,
                        init_image=None, denoise=None, sampler_preset=None,
                        hires_scale=None, hires_denoise=None,
                        finish_sharpen=None, finish_grain=None) -> dict:
    """Normalize and validate a run's global generation settings (Generate parity).
    Filter by family: a Krea sampler has no meaning in Z-Image. Return a dict
    ready to persist on LoraTestImage and pass to `_build_cell_workflow`.
    Out-of-scope or non-allowlisted values become None, keeping workflow defaults.

    An empty `negative` becomes None; clamp `denoise` to 0.05..1.0;
    `resolution_tier` must belong to RESOLUTION_TIERS."""
    fam = (run_family or 'zimage').lower()
    neg = ((negative or '').strip() or None) if generation_capabilities(fam)['negative_prompt'] else None
    smp = sampler if (fam == 'krea' and sampler in KREA_ALLOWED_SAMPLERS) else None
    # Custom sampler preset. Its allowlist is deliberately separate from
    # KREA_ALLOWED_SAMPLERS: they share a UI menu but target different graph
    # locations. Writing a preset into `sampler_name` would make ComfyUI reject
    # an unknown sampler. Unknown presets become None, using standard KSampler.
    from ..utils.comfyui import KREA_SAMPLER_PRESETS
    smp_preset = (sampler_preset
                  if (fam == 'krea' and sampler_preset in KREA_SAMPLER_PRESETS)
                  else None)
    sch = scheduler if (fam == 'krea' and scheduler in KREA_ALLOWED_SCHEDULERS) else None
    wdt = weight_dtype if (fam == 'krea' and weight_dtype in KREA_ALLOWED_WEIGHT_DTYPES) else None
    dta = None
    if fam == 'sdxl' and detail_amount is not None:
        try:
            dta = max(0.0, min(1.0, float(detail_amount)))
        except (TypeError, ValueError):
            dta = None
    tier = resolution_tier if resolution_tier in RESOLUTION_TIERS else None
    # Clamp the resolution multiplier to [1.0, 1.9], default 1.0. It applies
    # only to tier-based resolution; the fixed table leaves it at 1.0, with no effect.
    from ..utils.resolution import clamp_multiplier
    mult = clamp_multiplier(resolution_multiplier if resolution_multiplier is not None else 1.0)
    den = None
    if fam == 'krea' and denoise is not None:
        try:
            den = max(0.05, min(1.0, float(denoise)))
        except (TypeError, ValueError):
            den = None
    ini = ((init_image or '').strip() or None) if fam == 'krea' else None

    # Hi-res fix (Krea only). Three states, and None is the one that matters:
    #   None  -> the krea_hires.* setting decides at build time ("not set here")
    #   1.0   -> OFF for this run, whatever the setting says
    #   >1    -> this factor, clamped to the injector's ceiling
    # A value that does not parse, or is not finite, is "not set here" — never a
    # silent 1.0 (which would override a Settings default the user relies on)
    # and never a silent maximum (`min(MAX, nan)` IS the maximum; every
    # comparison against NaN is False and the clamp hands the ceiling back).
    from ..utils.comfyui import KREA_HIRES_MAX_SCALE
    hrs = None
    if fam == 'krea' and hires_scale is not None:
        try:
            v = float(hires_scale)
        except (TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v):
            hrs = 1.0 if v <= 1.0 else min(KREA_HIRES_MAX_SCALE, v)
    hrd = None
    if fam == 'krea' and hires_denoise is not None:
        try:
            v = float(hires_denoise)
        except (TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v):
            hrd = max(0.05, min(1.0, v))

    # Finishing (Krea only, engine-agnostic passes only). None/<=0 = off = NULL
    # on the row: there is no setting to fall back to for a Studio cell, so
    # "not set here" and "off" are the same state and share one representation.
    def _finish(value, ceiling):
        if fam != 'krea' or value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v) or v <= 0:
            return None
        return min(ceiling, v)
    fsh = _finish(finish_sharpen, 3.0)
    fgr = _finish(finish_grain, 0.2)
    return {'negative': neg, 'sampler': smp, 'sampler_preset': smp_preset,
            'scheduler': sch, 'weight_dtype': wdt,
            'detail_amount': dta, 'resolution_tier': tier,
            'resolution_multiplier': mult, 'init_image': ini, 'denoise': den,
            'hires_scale': hrs, 'hires_denoise': hrd,
            'finish_sharpen': fsh, 'finish_grain': fgr}


# --- Studio preflight (model files on disk + custom nodes in ComfyUI) ---------
# Klein already preflights its assets (KleinModelsMissing → 409 + auto-download);
# Krea/SDXL/Z-Image did NOT — the studio workflows hardcode the developer's own
# VAE / text-encoder / accelerator-LoRA names (none of which exist on
# a fresh install), so a fresh user launched a grid and every tile failed ComfyUI
# validation SILENTLY (empty grid, no reason). This block gives each family the
# same up-front check: verify (a) every model file the BUILT workflow references
# is on disk (via the exact filenames the workflow will send — zero divergence),
# and (b) every custom node the workflow uses exists in the target ComfyUI
# (/object_info), and raises StudioAssetsMissing so the route answers ONE
# actionable 409 instead.

class StudioAssetsMissing(Exception):
    """A Studio family's workflow references model files not on disk, or custom
    nodes the target ComfyUI doesn't expose, so every grid tile would fail ComfyUI
    validation and land as a silently-empty cell. Raised BEFORE any row/job is
    created so the caller can answer one actionable 409 (same spirit as Klein's
    KleinModelsMissing).

    `.family` = pipeline key ('zimage'/'sdxl'/'krea'); `.missing_files` =
    [{path, kind}] with `path` a display path like 'models/vae/…'; `.missing_nodes`
    = [class_type]; `.invalid_files` = [{path, kind, reason}] for a referenced model
    that IS on disk but is NOT real, loadable weights (an HTML gate page saved as
    .safetensors, a truncated download) — the same silent-empty-tile failure as a
    missing file, but the fix is 'delete + re-download', not 'place the file'."""
    def __init__(self, family, missing_files, missing_nodes, invalid_files=None):
        self.family = family
        self.missing_files = list(missing_files)
        self.missing_nodes = list(missing_nodes)
        self.invalid_files = list(invalid_files or [])
        n_f, n_n, n_i = len(self.missing_files), len(self.missing_nodes), len(self.invalid_files)
        super().__init__(f'{family} studio assets missing: {n_f} file(s), '
                         f'{n_n} node(s), {n_i} invalid file(s)')


class StudioArchMismatch(Exception):
    """A selected checkpoint's REAL architecture (read from its safetensors header)
    contradicts the family whose pipeline the Studio would run it under. ComfyUI
    silently drops every incompatible LoRA key, so the entire grid renders as if
    the LoRA were off (strength 0) with no error anywhere — the 2026-07-13
    incident (a Z-Image LoRA mislabelled Krea produced 117 no-op tiles). Raised
    BEFORE any row/job is created so the caller answers one actionable 409 (same
    spirit as StudioAssetsMissing).

    `.family` = the Studio's pipeline key; `.detected` = the checkpoint's real
    family; `.checkpoint` = the LoraLoader-form path that mismatched."""
    def __init__(self, family, detected, checkpoint):
        self.family = family
        self.detected = detected
        self.checkpoint = checkpoint
        super().__init__(f'{checkpoint} is a {detected} LoRA, not {family}')


def _is_unsafe_external_lora_name(fn) -> bool:
    """True if `fn` could resolve OUTSIDE a loras root once handed to
    `_ci_resolve` (path traversal / drive-letter / rooted path). `os.path.isabs`
    alone is not enough: on Windows it is False for a POSIX-style rooted path
    like '/abs/x.safetensors' (no drive letter), yet `_ci_resolve` still walks
    it as a normal — if odd — first component, and `_resolve_lora_abs_path`
    would otherwise `lstrip(os.sep)` it into something that LOOKS validated.
    So every rooted form is rejected explicitly, not inferred from `isabs`."""
    s = str(fn or '')
    if not s or os.path.isabs(s) or s.startswith(('/', '\\')) or ':' in s:
        return True
    return any(part == '..' for part in s.replace('\\', '/').split('/'))


def _resolve_lora_abs_path(checkpoint) -> str | None:
    """Absolute path of a LoraLoader-form checkpoint ('<subfolder>\\name.safetensors',
    relative to models/loras), resolved case-INSENSITIVELY (the workflow paths
    carry mixed casing — 'z image', 'Krea' — and a case-sensitive cloud FS must
    still find the file). None when ComfyUI's loras dir isn't configured or the
    file can't be located.

    Searched across EVERY loras root in ComfyUI's own priority order (the yaml's
    included), like the loader node this path is handed to — a LoRA deployed into
    an extra_model_paths root, or into the old default one before GitHub #25, must
    resolve either way."""
    rel = str(checkpoint or '').replace('\\', os.sep).replace('/', os.sep).lstrip(os.sep)
    if not rel:
        return None
    from . import comfy_model_paths
    try:
        roots = comfy_model_paths.search_roots('loras')
    except Exception:
        roots = []
    for loras in roots:
        found = _ci_resolve(str(loras), rel)
        if found and os.path.isfile(found):
            return found
    return None


def _preflight_checkpoint_arch(run_family, checkpoints):
    """Raise StudioArchMismatch if any selected checkpoint's REAL arch (safetensors
    header) contradicts `run_family`. family_of_lora keys off the FOLDER, which is
    exactly the blind spot a mislabelled deploy exploits — so we read the header
    here. Undetectable/foreign headers pass (no false block); only a POSITIVE
    cross-namespace contradiction stops the run."""
    for cp in checkpoints:
        p = _resolve_lora_abs_path(cp)
        if not p:
            continue
        detected = lt.detect_lora_arch(p)
        if lt.lora_arch_conflicts(detected, run_family):
            raise StudioArchMismatch(run_family, detected, cp)


# ComfyUI loader class_type -> (input keys carrying a model FILENAME, the models/
# subfolders that loader lists files from, human kind). A loader lists files
# relative to ONE of these subfolders; the file counts as present if it resolves
# under any (a UNET lives in unet/ OR diffusion_models/ on shared installs). Only
# loaders the studio workflows actually use are mapped.
_STUDIO_MODEL_LOADERS = {
    'UNETLoader': (('unet_name',), ('unet', 'diffusion_models'), 'diffusion model'),
    'CheckpointLoaderSimple': (('ckpt_name',), ('checkpoints',), 'checkpoint'),
    'VAELoader': (('vae_name',), ('vae',), 'VAE'),
    'CLIPLoader': (('clip_name',), ('text_encoders', 'clip'), 'text encoder'),
    'DualCLIPLoader': (('clip_name1', 'clip_name2'), ('text_encoders', 'clip'), 'text encoder'),
    'LoraLoader': (('lora_name',), ('loras',), 'LoRA'),
    'LoraLoaderModelOnly': (('lora_name',), ('loras',), 'LoRA'),
}


def _models_root():
    try:
        d = cfg.comfyui_dir('models')
    except Exception:
        return None
    return str(d) if d else None


def _ci_resolve(root, rel):
    """The real absolute path of root/rel with each component matched
    case-INSENSITIVELY below `root`, or None when no such entry exists. ComfyUI on
    Windows is case-insensitive and the workflow templates carry mixed folder casing
    (node refs 'Z image\\…' / 'Krea\\…' vs the on-disk 'z image' / 'krea') — a
    case-sensitive filesystem (cloud) must NOT read those as missing. `root` is
    assumed to exist."""
    cur = root
    for part in rel.split(os.sep):
        if not part or part == '.':
            continue
        nxt = os.path.join(cur, part)
        if os.path.exists(nxt):
            cur = nxt
            continue
        try:
            match = next((e for e in os.listdir(cur) if e.lower() == part.lower()), None)
        except OSError:
            return None
        if match is None:
            return None
        cur = os.path.join(cur, match)
    return cur if os.path.exists(cur) else None


def _ci_join_exists(root, rel):
    """os.path.exists(root/rel) with each component matched case-INSENSITIVELY below
    `root` — the boolean form of _ci_resolve (see it for why casing is tolerated)."""
    return _ci_resolve(root, rel) is not None


def _resolve_model_abs(models_root, subfolders, ref):
    """Absolute path of a PRESENT loader ref (mirrors _model_file_present's search:
    models_root/<subfolder>/ then extra_model_paths roots, case-insensitive), or None
    when it isn't on disk. Lets the preflight read the exact file ComfyUI would open
    and check it is real weights, not just present."""
    rel_ref = (ref or '').replace('\\', os.sep).replace('/', os.sep).lstrip(os.sep)
    if not rel_ref:
        return None
    for sub in subfolders:
        p = _ci_resolve(models_root, os.path.join(sub, rel_ref))
        if p:
            return p
    try:
        from . import comfy_model_paths
        for sub in subfolders:
            for root in comfy_model_paths.extra_roots(sub):
                p = _ci_resolve(root, rel_ref)
                if p:
                    return p
    except Exception:
        pass
    return None


def _model_file_present(models_root, subfolders, ref):
    """True if `ref` (a loader value, possibly with its own subfolder prefix)
    resolves to a real file under models_root/<subfolder>/ for any candidate
    subfolder, OR under an extra_model_paths.yaml root for those types (where the
    extra dir IS the type root, so `ref` resolves directly beneath it). With no yaml
    only the base models_root is checked, so this is unchanged. This keeps the Studio
    preflight from raising a false 'missing file' 409 for a model that lives in an
    extra path — ComfyUI resolves it natively at run time from the same yaml."""
    rel_ref = (ref or '').replace('\\', os.sep).replace('/', os.sep).lstrip(os.sep)
    if not rel_ref:
        return True  # empty ref = loader left at a wired default upstream — not our miss
    if any(_ci_join_exists(models_root, os.path.join(sub, rel_ref)) for sub in subfolders):
        return True
    try:
        from . import comfy_model_paths
        for sub in subfolders:
            for root in comfy_model_paths.extra_roots(sub):
                if _ci_join_exists(root, rel_ref):
                    return True
    except Exception:
        pass
    return False


def _scan_workflow_assets(workflow, models_root):
    """(missing_files, invalid_files, class_types) for a BUILT cell workflow.
    missing_files = [{path, kind}] for every model-loader reference NOT on disk;
    invalid_files = [{path, kind, reason}] for a reference that IS on disk but is not
    real, loadable weights (an HTML gate page saved as .safetensors, a truncated
    download) — this would fail ComfyUI validation and leave every tile silently
    EMPTY, the same silent-failure class as a missing file, so the preflight owns it
    too. Both are skipped entirely when models_root is unknown (the base-pool guards
    already caught that case). class_types = every node class in the graph."""
    missing, invalid, classes = [], [], set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type')
        if not ct:
            continue
        classes.add(ct)
        spec = _STUDIO_MODEL_LOADERS.get(ct)
        if not (spec and models_root):
            continue
        keys, subfolders, kind = spec
        inputs = node.get('inputs', {}) if isinstance(node.get('inputs'), dict) else {}
        for k in keys:
            ref = inputs.get(k)
            if not isinstance(ref, str) or not ref.strip():
                continue
            display = f'models/{subfolders[0]}/{ref}'.replace('\\', '/')
            abs_path = _resolve_model_abs(models_root, subfolders, ref)
            if abs_path is None:
                entry = {'path': display, 'kind': kind}
                # A resolver that came up empty leaves `_meta.lds_missing_hint` on the
                # node saying WHAT it accepted and WHERE it looked (see
                # utils/zimage_helper._resolve_zimage_assets). Carrying it into the 409
                # is what keeps the preflight honest now that resolution is automatic:
                # "this file is missing" alone would suggest the exact name is required,
                # when a dozen spellings would have done.
                meta = node.get('_meta')
                hint = meta.get('lds_missing_hint') if isinstance(meta, dict) else None
                if hint:
                    entry['hint'] = str(hint)
                if entry not in missing:
                    missing.append(entry)
                continue
            # Present — but is it real weights? Only a BLOCKING verdict (an HTML/text
            # file or a truncated header) counts: those can't load at all. No size
            # floor here — a legitimately small Studio VAE/LoRA must not be flagged.
            from . import model_integrity
            verdict = model_integrity.validate_model_file(abs_path)
            if verdict['blocking']:
                entry = {'path': display, 'kind': kind, 'reason': verdict['reason']}
                if entry not in invalid:
                    invalid.append(entry)
    return missing, invalid, classes


def preflight_family(family, workflows):
    """Raise StudioAssetsMissing if the target ComfyUI is missing any model file or
    custom node the family's BUILT workflow(s) need, OR if a referenced model file is
    present but not real weights (an HTML gate page saved as .safetensors, a truncated
    download — it would fail ComfyUI validation and leave every tile silently empty).
    `workflows` = representative built cell workflow(s) (one per base) — checking the
    ACTUAL built graph means zero divergence from what will be enqueued. Best-effort:
    only raises on a CONCRETE absence/invalidity; a build that couldn't be produced or
    an unreachable /object_info fails OPEN (the per-tile error capture still surfaces
    the reason).
    """
    models_root = _models_root()
    missing_files, invalid_files, all_classes = [], [], set()
    for wf in workflows:
        if not wf:
            continue
        mf, inv, classes = _scan_workflow_assets(wf, models_root)
        for e in mf:
            if e not in missing_files:
                missing_files.append(e)
        for e in inv:
            if e not in invalid_files:
                invalid_files.append(e)
        all_classes |= classes
    # Custom nodes: compare the graph's class_types to /object_info. Fail-OPEN when
    # it can't be fetched (None) — never block on a transient probe failure. A class
    # counts as present when the target exposes it OR a known alias (a node registered
    # under a permuted/forked name is the SAME capability — cf. NODE_CLASS_ALIASES),
    # so we never 409 an install that has the node under an alternative class name.
    missing_nodes = []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is not None and all_classes:
        missing_nodes = sorted(c for c in all_classes if not _node_class_present(c, available))
    if missing_files or invalid_files or missing_nodes:
        raise StudioAssetsMissing(family, missing_files, missing_nodes, invalid_files)


# Custom-node class_types the Studio family workflows pull from community packs,
# mapped to the pack that ships each + how to find it in ComfyUI-Manager. Turns a
# bare "node X is missing" 409 into an actionable "install pack Y (search: Z), then
# restart ComfyUI". Same spirit as klein_edit_helper.KLEIN_NODE_PACKS; an unknown
# node simply gets no hint (the generic message still shows), never an error.
STUDIO_NODE_PACKS = {
    # Krea 2 conditioning rebalance — RETIRED from the shipped graphs on 2026-09-02
    # (utils/comfyui, "Retired"). The hint stays: a user template or a Canvas plugin
    # graph may still name the class, and the pack is published by BOTH the original
    # nova452/ComfyUI-Conditioning-Rebalance and the huwhitememes fork under one key.
    'ConditioningKrea2Rebalance': {
        'pack': 'ComfyUI-Conditioning-Rebalance',
        'url': 'https://github.com/nova452/ComfyUI-Conditioning-Rebalance',
        'search': 'Krea 2 Conditioning',
    },
    # Krea 2 preset sampler (the optional custom-sampling lane). Unlike every other
    # entry here there is no `url` and no `search`: the code is already on the
    # user's disk, shipped with the app, and the fix is a button on the Setup
    # screen rather than a trip to ComfyUI-Manager. The banner branches on the
    # missing `url` — sending someone to search a manager for a pack that is not
    # published anywhere would be a dead end dressed as an instruction.
    'LDSKrea2PresetSampler': {
        'pack': 'Krea 2 preset sampler',
        'setup': 'the Setup screen, under Krea 2',
    },
    # Detail Daemon sampler (node 57 of image_real_HQ.json, the SDXL family's pass
    # 2) — EVERY fresh SDXL Studio install needs this pack, so its absence must
    # name the pack, not just the class (GitHub #36, KingyWolf).
    'DetailDaemonSamplerNode': {
        'pack': 'ComfyUI-Detail-Daemon',
        'url': 'https://github.com/Jonseed/ComfyUI-Detail-Daemon',
        'search': 'Detail Daemon',
    },
    # ── The Video Test Studio's MiniMax H3 graph ──────────────────────────────
    # Four packs, and only the first is needed for a plain clip: the others come
    # with an option the user ticked. Naming them separately is the difference
    # between "install this pack to use ⚡ Turbo" and a bare class name for a
    # node the user never asked for by name.
    'PathchSageAttentionKJ': {
        'pack': 'ComfyUI-KJNodes',
        'url': 'https://github.com/kijai/ComfyUI-KJNodes',
        'search': 'KJNodes',
    },
    'MiniMaxH3TurboLoRA': {
        'pack': 'ComfyUI-MiniMax-H3-Turbo',
        'url': 'https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo',
        'search': 'MiniMax H3 Turbo',
    },
    'MiniMaxH3TurboSampler': {
        'pack': 'ComfyUI-MiniMax-H3-Turbo',
        'url': 'https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo',
        'search': 'MiniMax H3 Turbo',
    },
    'H3SparseAttentionAdvanced': {
        'pack': 'H3-Optimizations',
        'url': 'https://github.com/Zironic/H3-Optimizations',
        'search': 'H3 Optimizations',
    },
    'MMH3UltimateUpscale': {
        'pack': 'Comfyui-MMH3-UltimateUpscale',
        'url': 'https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale',
        'search': 'MMH3 Ultimate Upscale',
    },
    'MMH3LatentUpscaleWithModelParams': {
        'pack': 'Comfyui-MMH3-UltimateUpscale',
        'url': 'https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale',
        'search': 'MMH3 Ultimate Upscale',
    },
    'MMH3TemporalSplitParams': {
        'pack': 'Comfyui-MMH3-UltimateUpscale',
        'url': 'https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale',
        'search': 'MMH3 Ultimate Upscale',
    },
}


def studio_missing_node_hints(nodes):
    """[{class_type, pack, url, search}] for each missing node class that maps to a
    known community pack (unknown classes omitted). Lets a studio_missing 409 name
    WHAT to install instead of only the bare class_type — same spirit as Klein's
    format_missing_nodes_message."""
    out = []
    for ct in nodes or []:
        pk = STUDIO_NODE_PACKS.get(ct)
        if pk:
            out.append({'class_type': ct, **pk})
    return out


def _preflight_run(user_id, run_family, checkpoint, bases, allowed, prompt, seed,
                   dataset_id, trigger_word, sampler_preset=None):
    """Build a representative cell workflow for `run_family` (one per distinct base
    in `bases`) and run `preflight_family` on it. Raises StudioAssetsMissing when
    the target ComfyUI can't run the grid. A representative build that itself fails
    is skipped (the enqueue loop would surface that path's own error).

    `sampler_preset` has to be threaded in even though the preflight cares about
    NOTHING else in the gen knobs: it is the one setting that changes which node
    CLASSES the graph names. Left out, the representative build carries a plain
    KSampler, the preflight finds nothing missing, and the run is enqueued to fail
    tile by tile on a ComfyUI validation error — the exact grid of mute failures
    this preflight exists to replace. Any future knob that adds a node class owes
    the same thread.

    The hi-res fix is NOT threaded, on purpose: it adds core classes only
    (LatentUpscaleBy, KSampler), which no install can be missing."""
    wfs = []
    seen = set()
    for base in (bases or [None]):
        key = base or ''
        if key in seen:
            continue
        seen.add(key)
        try:
            wfs.append(_build_cell_workflow(
                user_id, checkpoint, 1.0, prompt or '', seed or 1, base, allowed,
                dataset_id=dataset_id, train_type=run_family, trigger_word=trigger_word,
                sampler_preset=sampler_preset))
        except Exception as e:  # noqa: BLE001 — a bad representative build ≠ a missing asset
            if run_family in TRAINED_IMAGE_FAMILIES:
                raise
            logger.warning('studio preflight: representative build failed (base=%r): %s', base, e)
    preflight_family(run_family, wfs)


# --- Run lifecycle -----------------------------------------------------------
def _batch_lora_axis(batch_loras, run_family) -> list:
    """Validate the batch axis with the same path-injection rules as always-on
    LoRAs. Return [None, {filename,strength}, ...], where None is the reference
    cell without the LoRA. Deduplicate and cap at four LoRAs to limit GPU cost."""
    perm_allowed = {c['filename'] for c in permanent_lora_candidates(run_family)}
    entries = []
    for e in (batch_loras or []):
        fn = str((e or {}).get('filename') or '')
        if fn not in perm_allowed or any(x['filename'] == fn for x in entries):
            continue
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        entries.append({'filename': fn, 'strength': st})
    return [None] + entries[:4] if entries else [None]


# A prompt batch remains one run. Consecutive launches are rejected by
# "a test run is already in progress", and GPU execution is serial anyway.
# The batch is therefore an axis, like formats or cfg: one cell per prompt
# with the same checkpoints, settings and seed.
#
# There is no cell-count cap (see the module header and build_matrix): the
# queue is serial and users see the count and duration estimate before launch.
# An early version capped prompts at 24 without a measured reason. At 33,
# the request is still only a few KB against a 64 MB limit; prompt is unbounded
# TEXT, queue depth is unbounded, and result views do not truncate. GPU time
# is the real cost, communicated numerically before launch rather than by
# arbitrarily rejecting one of the six axes multiplied by the run.


def _prompt_axis(prompts, fallback) -> list:
    """Return the selected prompt axis, cleaned and deduplicated in input order.
    An empty list becomes `[fallback]`, preserving the single prompt-field
    behavior. `fallback` may be None when each cell should use its dataset's
    identity prompt for a comparison across datasets."""
    seen, out = set(), []
    for p in (prompts or []):
        if not isinstance(p, str):
            continue
        s = p.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out or [fallback]


# --- Measured machine throughput -------------------------------------------
# The UI formerly hardcoded "~12 s/image", accurate for Z-Image Turbo on a
# 4090 but misleading on slower cards: a promised 20-minute sweep could take
# two hours. The queue already records every job's started_at/completed_at;
# use that existing evidence to estimate actual throughput.
_PACE_SCAN_ROWS = 200      # maximum queue rows scanned, indexed by completed_at
_PACE_SAMPLE_SIZE = 30     # enough retained samples for a stable median
_PACE_MIN_SAMPLES = 3      # below this count, retain the UI default
_PACE_MIN_SECONDS = 0.5    # lower guard: a cell finished in 0.1 s did not render
_PACE_MAX_SECONDS = 900.0  # upper guard: exclude a machine sleeping during a job
                           # to avoid an eight-hour image distorting the estimate
DEFAULT_SECONDS_PER_IMAGE = 12.0   # historical fallback when no measurements exist


def measured_seconds_per_image(family=None) -> float | None:
    """Return the median test-generation duration observed on this machine.

    Use the median rather than the mean: a job delayed by a model download
    would inflate the mean for many later runs. `family` restricts samples to
    that pipeline, since Krea and Z-Image Turbo have different costs. Return
    None when there are too few samples, so the caller marks its default as
    approximate instead of inventing precision from two measurements."""
    try:
        rows = (db.session.query(ImageGenerationQueue.started_at,
                                 ImageGenerationQueue.completed_at,
                                 LoraTestImage.checkpoint)
                .join(LoraTestImage, LoraTestImage.job_id == ImageGenerationQueue.job_id)
                .filter(ImageGenerationQueue.status == 'completed',
                        ImageGenerationQueue.started_at.isnot(None),
                        ImageGenerationQueue.completed_at.isnot(None),
                        # ✨ An Upscale & improve job is NOT a test generation and
                        # its duration says nothing about the pace of a sweep: it
                        # is a 2 MP Klein edit or a SeedVR2 restoration, minutes
                        # where a Turbo cell takes seconds. It passes the family
                        # filter below (a derived row copies its source's
                        # `checkpoint`) and sits well inside the 0.5-900 s window,
                        # so with a 30-sample median a handful of them visibly
                        # inflate the duration this estimate promises before a
                        # launch. Same rule as _cells(), same predicate.
                        _is_cell())
                .order_by(ImageGenerationQueue.completed_at.desc())
                .limit(_PACE_SCAN_ROWS).all())
    except Exception:                      # legacy database missing one of the columns
        logger.debug('pace: queue timings unreadable', exc_info=True)
        return None
    secs = []
    for started, completed, checkpoint in rows:
        if family and family_of_lora(checkpoint or '') not in (None, family):
            continue
        try:
            d = (completed - started).total_seconds()
        except (TypeError, AttributeError):
            continue
        if _PACE_MIN_SECONDS <= d <= _PACE_MAX_SECONDS:
            secs.append(d)
        if len(secs) >= _PACE_SAMPLE_SIZE:
            break
    if len(secs) < _PACE_MIN_SAMPLES:
        return None
    secs.sort()
    mid = len(secs) // 2
    median = secs[mid] if len(secs) % 2 else (secs[mid - 1] + secs[mid]) / 2
    return round(median, 1)


def checkpoint_origins(checkpoints, explicit=None, family=None) -> dict:
    """{deployed filename: (record_id, step)} — WHICH training checkpoint each
    selected LoRA came from, so every cell can record it on its row instead of
    the app re-deriving it from the filename on every render (the heuristic that
    already shipped a bug, see LoraTestImage.record_id).

    `explicit` is the mapping a caller that ALREADY knows the answer provides —
    the LoRA Canvas, where the user picked a lineage pill, so the run and the
    step are the identity of what was clicked. It always wins.

    Without it the origin is read back from the run tag the DEPLOY stamped into
    the name (`_rl<record>` / `_rc<cloud run>` + the zero-padded step): the Test
    Studio picks a filename out of a folder and has no other handle. That tag was
    written by the app, not inferred from a trigger word — and a name that
    carries none resolves to (None, None), i.e. an honestly unlinked cell.

    A run's FINAL save is deployed with the tag and NO step (`…_rc192_v2`). Its
    step is the one the run's lineage pairs that file with — the final pill that
    shows it as deployed (`final_save_step`; `family`, the launch's, names the
    ComfyUI folder that pairing reads). A file no pill shows keeps its run with
    no step, as the boot backfill records it. Keeping neither left every image
    tested from a final save out of the Canvas and the checkpoint galleries.

    Resolved ONCE per distinct filename: a 40-cell grid over 6 checkpoints costs
    6 lookups."""
    out = {}
    for cp in checkpoints or []:
        if cp in out:
            continue
        hint = (explicit or {}).get(cp)
        if hint:
            try:
                out[cp] = (int(hint['record_id']), int(hint['step']))
                continue
            except (KeyError, TypeError, ValueError):
                pass                     # malformed hint → fall through to the tag
        try:
            from .checkpoint_link_backfill import (
                final_save_step, resolve_checkpoint_name, resolve_run_name)
            hit = resolve_checkpoint_name(cp)
            if hit is None:
                run = resolve_run_name(cp)
                if run:
                    hit = (run[0], final_save_step(cp, run[0], run[1], family))
        except Exception:                # a registry read must never fail a launch
            hit = None
        out[cp] = (hit[0], hit[1]) if hit else (None, None)
    return out


def _batch_lora_label(row):
    """Return a cell's readable batch LoRA name, or None, for grid/lightbox badges.
    Read the batch:true entry in its extra_loras JSON."""
    try:
        for e in json.loads(row.extra_loras or '[]'):
            if isinstance(e, dict) and e.get('batch'):
                return _basename(e.get('filename', '')).rsplit('.', 1)[0]
    except (ValueError, TypeError):
        pass
    return None


def _combined_lora_labels(row) -> list:
    """Readable names of LoRAs stacked with the cell's LoRA (`combined:true`
    extra_loras entries), used for the grid/lightbox "+ X" badge. Return an
    empty list for a cell without a stack.

    Runs launched from the stack view write `filename`/`dataset_id`/`trigger`.
    Cell JSON is frozen at creation, so older runs have only `label`/`weight`;
    missing keys become None and composition remains visible without a trigger.

    `record_id`/`step` identify the member's source board checkpoint. They follow
    the same rule: stored when known at launch, None on older runs. Missing
    provenance is valid for an older stack. Readers must distinguish "no parent"
    from "unknown parent" instead of inventing one."""
    out = []
    try:
        for e in json.loads(row.extra_loras or '[]'):
            if isinstance(e, dict) and e.get('combined'):
                name = _basename(e.get('filename', '')).rsplit('.', 1)[0]
                out.append({'label': format_trained_lora_label(e.get('filename', '')) or name,
                            'weight': e.get('strength'),
                            'filename': e.get('filename') or None,
                            'dataset_id': e.get('dataset_id'),
                            'record_id': e.get('record_id'),
                            'step': e.get('step'),
                            'trigger': e.get('trigger') or None})
    except (ValueError, TypeError):
        pass
    return out


def stack_of_row(row) -> list | None:
    """Return a cell's ordered stack composition, or None when it has no stack.

    The head LoRA is the cell itself (`checkpoint` and `strength`);
    create_comparison_run reduces the strengths axis to the head weight in
    combine mode. Subsequent members are `combined:true` entries. Read the head
    trigger from its dataset: unlike stacked triggers, it is not frozen in JSON."""
    combined = _combined_lora_labels(row)
    if not combined:
        return None
    ds = db.session.get(FaceDataset, row.dataset_id)
    head = {'label': (format_trained_lora_label(row.checkpoint)
                      or _basename(row.checkpoint or '').rsplit('.', 1)[0]),
            'weight': row.strength, 'filename': row.checkpoint,
            'dataset_id': row.dataset_id,
            # The head's origin is already stored in columns linking it to a board
            # checkpoint. Include it here so every stack member, including the head,
            # uses the same representation.
            'record_id': row.record_id, 'step': row.step,
            'trigger': (getattr(ds, 'trigger_word', None) or None) if ds else None,
            'head': True}
    return [head] + [{**c, 'head': False} for c in combined]


def _stack_signature(members) -> str:
    """Identify a stack independently of weights by its sorted filenames.
    Runs with the same signature are weight variants of the same stack and can
    be shown side by side."""
    return '|'.join(sorted(str((m or {}).get('filename') or '') for m in (members or [])))


# Bound the variant scan instead of searching the dataset's entire history.
# A stack run has few cells (1 x count x batch), so several hundred rows
# comfortably cover a tuning session.
_STACK_SCAN_ROWS = 600


def _shared_cell(r) -> dict:
    """Serialize a Studio cell first as an ordinary gallery image.

    The shared cloud_training.gallery_image serializer provides parity: Studio
    and Gallery read the same prompt, seed, checkpoint, extra LoRAs, base,
    sampler, derivation and camera-pose facts. Studio-specific keys extend this
    base rather than replace it. Three cell payloads previously diverged into
    three shapes and displayed inconsistent information.
    Import lazily because cloud_training already imports this module at runtime."""
    from . import cloud_training as ct
    cell = ct.gallery_image(r)
    if not r.filename:
        cell['url'] = None   # pending/failed: no file, no misleading URL
    return cell


def stack_variants(run_id, rows, limit=8) -> list:
    """Return runs of the same stack (same LoRAs, possibly different weights),
    newest first, including the current run marked `active`.

    Each variant includes its weight vector, cells (voted on by cell ID), and
    vote summary to compare alternative weights. Limit to `limit` variants and
    `_STACK_SCAN_ROWS` scanned rows. Frequently rerun stacks show only recent
    variants; a variant extending beyond the scan window has truncated cells."""
    members = stack_of_row(rows[0]) if rows else None
    if not members:
        return []
    sig = _stack_signature(members)
    head_ds = members[0].get('dataset_id')
    scanned = (_cells()
               .filter(LoraTestImage.dataset_id == head_ds,
                       LoraTestImage.extra_loras.isnot(None))
               .order_by(LoraTestImage.id.desc()).limit(_STACK_SCAN_ROWS).all())
    # Group by (run, weight vector), not just run. Before blend sweeps each
    # run had one combination; now one run may contain several. Grouping only
    # by run would mislabel all images with the first cell's weights. With one
    # weight per LoRA, the vector is constant and grouping remains unchanged.
    def _weight_vector(row):
        comp = stack_of_row(row)
        return tuple((m.get('filename'), m.get('weight')) for m in (comp or []))

    groups = {}
    for r in scanned:
        if not r.run_id:
            continue
        groups.setdefault((r.run_id, _weight_vector(r)), []).append(r)
    # The current run is independent of the scan window: reinsert each of its
    # combinations under its own key.
    for r in rows:
        groups.setdefault((run_id, _weight_vector(r)), [])
        if r not in groups[(run_id, _weight_vector(r))]:
            groups[(run_id, _weight_vector(r))].append(r)

    out = []
    for (rid, _vector), grp in groups.items():
        # `limit` must never evict the displayed run: its columns are what users
        # are viewing, and a sweep can have several. Cap only the other variants.
        if len(out) >= limit and rid != run_id:
            continue
        # Cells without run_id in legacy databases do not form one run. Grouping
        # them would fabricate a phantom variant from unrelated generations.
        if not rid or not grp:
            continue
        cells = sorted(grp, key=lambda x: x.id)
        comp = stack_of_row(cells[0])
        if not comp or _stack_signature(comp) != sig:
            continue
        out.append({
            'run_id': rid,
            'active': rid == run_id,
            'weights': [{'label': m['label'], 'weight': m['weight'],
                         'filename': m['filename']} for m in comp],
            'likes': sum(1 for c in cells if c.rating == 1),
            'dislikes': sum(1 for c in cells if c.rating == -1),
            'done': sum(1 for c in cells if c.status == 'done' and c.filename),
            # Extend the shared cloud_training.gallery_image serializer so Studio
            # shows the same prompt, LoRAs, base and sampler facts as Gallery. Keep
            # block-specific keys layered on top of those shared image facts.
            'cells': [{**_shared_cell(c),
                       'label': _basename(c.checkpoint or '').rsplit('.', 1)[0],
                       'filename': c.filename, 'status': c.status,
                       'error': c.error if c.status == 'failed' else None} for c in cells],
        })
    # Current run first, then scan order from newest to oldest.
    out.sort(key=lambda v: not v['active'])
    return out


@dataclass(frozen=True)
class StudioGenSettings:
    """The generation settings a Studio run carries, one object instead of
    a 24-keyword tail on every signature (the audit's 31-parameter debt).

    Axes (checkpoints, strengths) and per-entry-point extras (family,
    origins, prompts, combine) stay explicit parameters on purpose: they
    are what each call is ABOUT. Everything here is the shared "how" -
    validated downstream by _sanitize_gen_knobs exactly as before, so a
    None keeps meaning "the family default".

    Frozen: a run's settings are decided at the door; nothing downstream
    may quietly edit them for one cell."""
    seed: object = None
    prompt: object = None
    z_model: object = None
    z_models: object = None
    aspects: object = None
    cfgs: object = None
    steps_list: object = None
    steps2_list: object = None
    count: object = 1
    permanent_loras: object = None
    batch_loras: object = None
    negative: object = None
    sampler: object = None
    # Custom Krea sampler preset. None disables it and uses standard KSampler.
    sampler_preset: object = None
    scheduler: object = None
    weight_dtype: object = None
    detail_amount: object = None
    resolution_tier: object = None
    resolution_multiplier: object = None
    init_image: object = None
    denoise: object = None
    # Krea hi-res fix, per run. None = the krea_hires.* setting; 1.0 = off for
    # THIS run whatever the setting says. Steps are not per-run: they follow
    # the setting (0 = inherit pass 1), one dial fewer on a panel that is
    # already a wall.
    hires_scale: object = None
    hires_denoise: object = None
    # App-side finishing on the finished cell. None/0 = off. Krea only, like
    # the pass that inspired it; no colour match here — a Studio cell is
    # text-to-image, there is no "before" image to match to.
    finish_sharpen: object = None
    finish_grain: object = None
    # Trigger word checkbox: False prevents prefixing the dataset trigger.
    # None/True preserves the historical injection during graph construction.
    inject_trigger: object = None

    @classmethod
    def from_payload(cls, d):
        """Build from a request JSON dict, using the exact wire names the
        routes have always read (steps/steps2 are the wire names of
        steps_list/steps2_list). Only known keys are read - a payload
        typo cannot smuggle a setting in."""
        return cls(
            seed=d.get('seed'),
            prompt=d.get('prompt'),
            z_model=d.get('z_model'),
            z_models=d.get('z_models'),
            aspects=d.get('aspects'),
            cfgs=d.get('cfgs'),
            steps_list=d.get('steps'),
            steps2_list=d.get('steps2'),
            count=d.get('count'),
            permanent_loras=d.get('permanent_loras'),
            batch_loras=d.get('batch_loras'),
            negative=d.get('negative'),
            sampler=d.get('sampler'),
            sampler_preset=d.get('sampler_preset'),
            scheduler=d.get('scheduler'),
            weight_dtype=d.get('weight_dtype'),
            detail_amount=d.get('detail_amount'),
            resolution_tier=d.get('resolution_tier'),
            resolution_multiplier=d.get('resolution_multiplier'),
            init_image=d.get('init_image'),
            denoise=d.get('denoise'),
            hires_scale=d.get('hires_scale'),
            hires_denoise=d.get('hires_denoise'),
            finish_sharpen=d.get('finish_sharpen'),
            finish_grain=d.get('finish_grain'),
            inject_trigger=d.get('inject_trigger'))


def create_run(user_id, dataset_id, checkpoints, strengths, settings=None, *,
               family=None, origins=None, prompts=None) -> dict:
    """Validate and materialize the grid, then enqueue every cell.

    `prompts` is an axis: render each configuration once per checked history
    prompt. Missing/empty means a single `prompt`, preserving previous behavior.

    A filename outside this dataset's trigger-matched pool is a guest checkpoint
    (a LoRA trained elsewhere in models/loras). It gets its own cell with the
    same prompt, seed and strength axis, rather than being stacked on every
    cell. Guests fail closed on unsafe names or missing files and are capped
    at MAX_GUEST_CHECKPOINTS.

    Each cell row and queue job land in one commit (`_persist_and_enqueue_cell`).
    An enqueue failure marks that row 'failed' and re-raises; previously enqueued
    cells retain their rows and jobs. Return
    {'created', 'seed', 'count', 'run_id', 'ids'}."""
    # One object at the door; the body below is verbatim from the flat-
    # signature era, so it reads the same locals it always has.
    settings = settings or StudioGenSettings()
    seed = settings.seed
    prompt = settings.prompt
    z_model = settings.z_model
    z_models = settings.z_models
    aspects = settings.aspects
    cfgs = settings.cfgs
    steps_list = settings.steps_list
    steps2_list = settings.steps2_list
    count = settings.count
    permanent_loras = settings.permanent_loras
    batch_loras = settings.batch_loras
    negative = settings.negative
    sampler = settings.sampler
    sampler_preset = settings.sampler_preset
    scheduler = settings.scheduler
    weight_dtype = settings.weight_dtype
    detail_amount = settings.detail_amount
    resolution_tier = settings.resolution_tier
    resolution_multiplier = settings.resolution_multiplier
    init_image = settings.init_image
    denoise = settings.denoise
    hires_scale = settings.hires_scale
    hires_denoise = settings.hires_denoise
    finish_sharpen = settings.finish_sharpen
    finish_grain = settings.finish_grain
    # Unchecked Trigger word means False: send the prompt without the dataset
    # trigger. Persist per cell so resume reproduces that choice.
    inject_trigger = settings.inject_trigger is not False
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not (ds.trigger_word or '').strip():
        raise ValueError('trigger word is required')

    reason = gpu_busy_reason()
    if reason:
        raise GpuBusyError(reason)
    if _active_run_count(dataset_id):
        raise ValueError('a test run is already in progress on this dataset - '
                         'wait for it to finish or cancel')

    # Derive the run family from selected checkpoints in one loras/<family>
    # folder. The frontend selects one family at a time. A run cannot mix
    # ZIT/SDXL/Krea because bases and workflows differ. Use `family` as fallback
    # for legacy renamed checkpoints without a folder prefix.
    cps_in = [c for c in (checkpoints or []) if isinstance(c, str) and c.strip()]
    if not cps_in:
        raise ValueError('at least one checkpoint is required')
    fams = {family_of_lora(c) for c in cps_in}
    fams.discard(None)
    if len(fams) > 1:
        raise ValueError('a test run cannot mix multiple families (ZIT/SDXL/Krea)')
    run_family = (next(iter(fams), None) or family or getattr(ds, 'train_type', None) or 'zimage').lower()

    allowed = {c['filename'] for c in list_test_checkpoints(ds, run_family)}
    guests = [c for c in cps_in if c not in allowed]
    if guests:
        allowed |= _accept_guest_checkpoints(guests)

    # Apply always-on style/utility LoRAs to every cell, outside the batch axis.
    # Validate against family candidates to prevent path injection; clamp strength.
    perm_allowed = {c['filename'] for c in permanent_lora_candidates(run_family)}
    extra_loras = []
    for e in (permanent_loras or []):
        fn = str((e or {}).get('filename') or '')
        if fn not in perm_allowed:
            continue
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        extra_loras.append({'filename': fn, 'strength': st})
    # Batch axis: render each configuration once without, then once with each
    # checked batch LoRA. Always-on LoRAs above apply to every cell.
    batch_axis = _batch_lora_axis(batch_loras, run_family)

    # Global run generation settings (Generate parity), validated and gated by family.
    knobs = _sanitize_gen_knobs(
        run_family, negative=negative, sampler=sampler, scheduler=scheduler,
        sampler_preset=sampler_preset,
        weight_dtype=weight_dtype,
        detail_amount=detail_amount, resolution_tier=resolution_tier,
        resolution_multiplier=resolution_multiplier,
        init_image=init_image, denoise=denoise,
        hires_scale=hires_scale, hires_denoise=hires_denoise,
        finish_sharpen=finish_sharpen, finish_grain=finish_grain)

    cells = build_matrix(checkpoints, strengths, aspects, cfgs, steps_list, steps2_list,
                         family=run_family)

    models = _require_family_bases(run_family)
    valid_models = _select_base_models(models, z_model, z_models, family=run_family)

    try:
        seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
    except (TypeError, ValueError):
        raise ValueError(f'invalid seed: {seed!r}')

    # Generate N distinct seeds per configuration, shared across configurations
    # for fair comparisons with identical seeds. Bound N to 1..4.
    try:
        count = max(1, min(int(count or 1), 4))
    except (TypeError, ValueError):
        count = 1
    _MAX = 2**31 - 1
    seeds = [1 + ((seed + i - 1) % _MAX) for i in range(count)]  # distinct values in [1, 2^31-1]

    # Use the optional custom prompt or default identity prompt, without the
    # trigger when unchecked. The fallback must honor the same choice so empty
    # prompts do not contradict the "no trigger" metadata.
    prompt = (prompt or '').strip() or identity_prompt(ds, with_trigger=inject_trigger)
    # Prompt batch: no selection gives [prompt], preserving the previous path.
    # Preflight and logging use the first prompt.
    prompt_axis = _prompt_axis(prompts, prompt)
    prompt = prompt_axis[0]

    # Architecture guard: family_of_lora derives family from the folder, which
    # could misclassify a Z-Image LoRA deployed under loras/krea and silently do
    # nothing. Inspect every selected checkpoint's actual header architecture
    # before creating rows, returning an actionable 409 on mismatch.
    _preflight_checkpoint_arch(run_family, cps_in)
    # Preflight every model and custom node required on the target ComfyUI.
    # Build a representative graph per base before creating rows. New installs
    # receive one actionable 409 instead of a grid of silent failures.
    # Previously only Klein had preflight; Krea and SDXL did not.
    _preflight_run(user_id, run_family, cells[0][0], valid_models, allowed,
                   prompt, seeds[0], dataset_id,
                   ds.trigger_word if inject_trigger else None,
                   sampler_preset=knobs['sampler_preset'])

    # Read target ComfyUI classes once for the grid. Builders resolve
    # NODE_CLASS_ALIASES to actual registered names; None on a failed probe
    # preserves canonical names.
    available_classes = _target_node_classes()
    # WHICH lineage checkpoint each selected LoRA is, stamped on every cell it
    # produces (see checkpoint_origins) — the canvas gallery reads these columns,
    # it never re-parses a filename.
    origin_of = checkpoint_origins(cps_in, origins, family=run_family)
    # One opaque id per invocation is the strict boundary of a render-equivalent
    # timeline.  A later launch with the same prompt/seed/settings must never be
    # spliced into this one; resume_run reuses the ids already stored on rows.
    run_id = uuid.uuid4().hex
    # Partition the complete base-model × cell plan, not each base separately:
    # after Krea starts applying a tested LoRA, it must not return to a
    # tested-LoRA-off control merely because the optional base axis advanced.
    cell_plan = _krea_zero_strength_first(
        ((zm, cell) for zm in valid_models for cell in cells),
        run_family,
        lambda planned: planned[1][1],
    )
    ids = []
    for zm, cell in cell_plan:
        checkpoint, strength, cell_aspect, cell_cfg, cell_steps, cell_steps2 = cell
        # Format/CFG/steps (passes 1 and 2) are full multi-select test axes.
        width, height = _aspect_dims(cell_aspect, run_family, knobs['resolution_tier'],
                                     knobs['resolution_multiplier'])
        for batch_lora in batch_axis:  # Batch axis: without, then with each selected LoRA
          row_extra = extra_loras + ([{**batch_lora, 'batch': True}] if batch_lora else [])
          wf_extra = extra_loras + ([batch_lora] if batch_lora else [])
          cell_extra_json = json.dumps(row_extra) if row_extra else None
          for cell_prompt in prompt_axis:  # Prompt batch axis: one pass per selected prompt
           for cell_seed in seeds:  # N images per configuration with different seeds, displayed as a cell strip
            img = LoraTestImage(dataset_id=dataset_id, checkpoint=checkpoint,
                                strength=strength, seed=cell_seed, run_seed=seed,
                                run_id=run_id,
                                status='pending', z_model=zm, aspect=cell_aspect,
                                prompt=cell_prompt, cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                extra_loras=cell_extra_json,
                                negative=knobs['negative'], sampler=knobs['sampler'],
                                sampler_preset=knobs['sampler_preset'],
                                scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                detail_amount=knobs['detail_amount'],
                                resolution_tier=knobs['resolution_tier'],
                                resolution_multiplier=knobs['resolution_multiplier'],
                                init_image=knobs['init_image'], denoise=knobs['denoise'],
                                hires_scale=knobs['hires_scale'],
                                hires_denoise=knobs['hires_denoise'],
                                finish_sharpen=knobs['finish_sharpen'],
                                finish_grain=knobs['finish_grain'],
                                # Use NULL when checked (the default), preserving the row's original
                                # values from before this column existed.
                                inject_trigger=None if inject_trigger else False,
                                record_id=origin_of.get(checkpoint, (None, None))[0],
                                step=origin_of.get(checkpoint, (None, None))[1])
            _persist_and_enqueue_cell(
                img, user_id, dataset_id, cell_prompt,
                lambda: _build_cell_workflow(user_id, checkpoint, strength,
                                             cell_prompt, cell_seed, zm, allowed,
                                             width=width, height=height,
                                             cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                             dataset_id=dataset_id,
                                             train_type=run_family, extra_loras=wf_extra,
                                             negative=knobs['negative'], sampler=knobs['sampler'],
                                             sampler_preset=knobs['sampler_preset'],
                                             scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                             detail_amount=knobs['detail_amount'],
                                             hires_scale=knobs['hires_scale'],
                                             hires_denoise=knobs['hires_denoise'],
                                             trigger_word=(ds.trigger_word
                                                           if inject_trigger else None),
                                             available_classes=available_classes))
            ids.append(img.id)
    logger.info(f"lora-test: run {run_id} dataset {dataset_id} -> {len(ids)} cell(s) "
                f"({len(valid_models)} model(s), {len(prompt_axis)} prompt(s)), "
                f"base seed {seed} ×{count}")
    return {'created': len(ids), 'seed': seed, 'count': count,
            'run_id': run_id, 'ids': ids}


# A combined-stack weight uses the shared range's upper bound above.
# The head weight passes through build_matrix; a higher blend cap would
# fail the run instead of clamping it. Browser counterpart: COMBINE_MAX_WEIGHT
# in frontend/src/components/dataset/studio/loraStack.js.
COMBINE_MAX_WEIGHT = MAX_LORA_STRENGTH


def _combine_weight(sel) -> float:
    """A combined-stack LoRA weight: 0..COMBINE_MAX_WEIGHT, rounded to two decimals.
    Default to 1.0 for missing or unreadable values."""
    try:
        return max(0.0, min(COMBINE_MAX_WEIGHT,
                            round(float((sel or {}).get('weight', 1.0)), 2)))
    except (TypeError, ValueError):
        return 1.0


def _combine_weights(sel) -> list:
    """Weights swept for this LoRA: the `weights` list when supplied (Blend panel
    weight checkboxes), otherwise the scalar `weight`.

    Always nonempty, clamped to 0..COMBINE_MAX_WEIGHT, rounded to two decimals,
    and deduplicated in input order. A selection providing only `weight` (an
    older client or a newer frontend falling back to an older backend) returns
    exactly one value. Sweeping is additive and preserves existing semantics."""
    raw = (sel or {}).get('weights')
    if not isinstance(raw, (list, tuple)) or not raw:
        return [_combine_weight(sel)]
    out = []
    for v in raw:
        try:
            w = max(0.0, min(COMBINE_MAX_WEIGHT, round(float(v), 2)))
        except (TypeError, ValueError):
            continue
        if w not in out:
            out.append(w)
    return out or [_combine_weight(sel)]


def _cmp_resolve_run_family(selections):
    """Validate comparison-run inputs, extracted unchanged on 2026-08-23.
    Reject empty selections, busy GPU, an active run or mixed families, then
    resolve the run family and allowed bases. Return (run_type, models)."""
    if not selections:
        raise ValueError('no LoRA selected')
    reason = gpu_busy_reason()
    if reason:
        raise GpuBusyError(reason)
    if _active_run_count():
        raise ValueError('a test run is already in progress - wait for it to finish or cancel')
    # Derive family from checkpoint folders (family_of_lora), not ds.train_type:
    # a dataset can train multiple families. A run uses one family because bases
    # and workflows differ. Resolve its base before entering the loop.
    fams = {family_of_lora(str(sel.get('checkpoint') or '')) for sel in (selections or [])}
    fams.discard(None)
    if len(fams) > 1:
        # Name the selected families in errors. A generic ZIT/SDXL/Krea message
        # hid which choices conflicted, especially in combine mode.
        named = ' + '.join(_MODE_LABEL_BY_FAMILY.get(f, f) for f in sorted(fams))
        raise ValueError(
            f'a test run cannot mix LoRA families ({named}) — they need different '
            'base models and workflows. Keep one family per run.')
    run_type = (next(iter(fams), None) or 'zimage').lower()
    models = _require_family_bases(run_type)
    return run_type, models


def _cmp_seed_and_prompts(models, z_model, z_models, seed, count, prompt,
                          prompts, *, family=None):
    """Resolve the seed/base/prompt axes, extracted unchanged.
    Validate requested bases (list preferred, scalar for backward compatibility),
    choose the supplied or random seed, bound count, and derive the seed sequence
    and prompt batch. Return (valid_models, seed, count, seeds, prompt_axis)."""
    # Base models form a sweep axis exactly as in create_run. Canvas's
    # BASE MODEL (MULTI) previously launched only one silently. Prefer z_models,
    # then legacy scalar z_model, then the first available base.
    valid_models = _select_base_models(models, z_model, z_models, family=family)
    try:
        seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
    except (TypeError, ValueError):
        raise ValueError(f'invalid seed: {seed!r}')
    try:
        count = max(1, min(int(count or 1), 4))
    except (TypeError, ValueError):
        count = 1
    _MAX = 2**31 - 1
    seeds = [1 + ((seed + i - 1) % _MAX) for i in range(count)]
    common_prompt = (prompt or '').strip() or None
    # Prompt batch: one pass per checked prompt. No selection gives
    # [common_prompt], or [None] if absent, so each cell uses its own dataset's
    # identity prompt as before.
    prompt_axis = _prompt_axis(prompts, common_prompt)
    return valid_models, seed, count, seeds, prompt_axis


def _cmp_collect_extra_loras(run_type, permanent_loras, external_loras):
    """Collect LoRAs stacked on every cell, extracted unchanged.
    Validate always-on entries against the family pool (silently skip invalid
    entries), then validate Canvas externals fail-closed: reject traversal before
    resolution, fail on missing files, and cap at 16. Return
    (extra_loras, externals); only externals feed the architecture preflight."""
    # Validate always-on style/utility LoRAs against the family to prevent path
    # injection, then apply to every cell, just as in create_run.
    perm_allowed = {c['filename'] for c in permanent_lora_candidates(run_type)}
    extra_loras = []
    for e in (permanent_loras or []):
        fn = str((e or {}).get('filename') or '')
        if fn not in perm_allowed:
            continue
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        extra_loras.append({'filename': fn, 'strength': st})
    # 🔌 External LoRAs (Canvas plugin nodes): ANY models/loras file, stacked on
    # top of every cell via the same extra_loras channel. Unlike always-on LoRA
    # they are NOT restricted to the family pool, so validation is fail-closed:
    # a name that does not resolve under a loras root is a hard error, never a
    # silent skip.
    externals = []
    for e in (external_loras or []):
        fn = str((e or {}).get('filename') or '').strip()
        if not fn or any(x['filename'] == fn for x in externals):
            continue
        # Path-traversal guard: `external_loras` is the FIRST free-text channel
        # to reach `_resolve_lora_abs_path` → `_ci_resolve` (every other caller —
        # permanent/batch/checkpoint — is gated by a disk-scan allowlist first).
        # `_ci_resolve` walks each component checking `os.path.exists` and treats
        # '..' as an ordinary component, so it happily climbs OUT of the loras
        # root. Checked BEFORE the resolve call, not after: a name that escapes
        # must never even get a "not found" vs "found" answer.
        if _is_unsafe_external_lora_name(fn):
            raise ValueError(f'invalid external LoRA name: {fn}')
        if not _resolve_lora_abs_path(fn):
            raise ValueError(f'external LoRA not found: {fn}')
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        externals.append({'filename': fn, 'strength': st, 'external': True})
        if len(externals) >= 16:   # same cap as the PUT route + the board's UI
            break
    extra_loras.extend(externals)
    return extra_loras, externals


def _cmp_cell_knobs(run_type, batch_loras,
                    negative, sampler, scheduler, weight_dtype,
                    detail_amount, resolution_tier,
                    resolution_multiplier, init_image, denoise,
                    sampler_preset=None, hires_scale=None, hires_denoise=None,
                    finish_sharpen=None, finish_grain=None):
    """Resolve cell settings, extracted unchanged: the batch axis and global
    settings validated and gated by family. Return (batch_axis, knobs)."""
    # Batch axis: render each configuration without, then with each checked
    # batch LoRA, using the same mechanism as create_run.
    batch_axis = _batch_lora_axis(batch_loras, run_type)
    # Global generation settings (Generate parity), validated and gated by family.
    knobs = _sanitize_gen_knobs(
        run_type, negative=negative, sampler=sampler, scheduler=scheduler,
        sampler_preset=sampler_preset,
        weight_dtype=weight_dtype,
        detail_amount=detail_amount, resolution_tier=resolution_tier,
        resolution_multiplier=resolution_multiplier,
        init_image=init_image, denoise=denoise,
        hires_scale=hires_scale, hires_denoise=hires_denoise,
        finish_sharpen=finish_sharpen, finish_grain=finish_grain)
    return batch_axis, knobs


def _cmp_preflight(user_id, run_type, selections, externals, valid_models,
                   prompt_axis, seeds, inject_trigger=True, sampler_preset=None):
    """Run preflights, extracted unchanged: compare every checkpoint's actual
    architecture (including externals) with the family, then try the family
    workflow on the first valid selection. Raise one actionable 409 before
    creating any rows. Return the memoized `_dataset_and_checkpoints` closure
    shared by the stack and cell loop: one LoRA scan per dataset per call."""
    # Architecture guard, as in create_run: each selected checkpoint's header
    # must match the run family. Otherwise ComfyUI silently drops it, yielding
    # a no-op grid. Check before creating any rows for an actionable 409.
    _preflight_checkpoint_arch(
        run_type,
        [s.get('checkpoint') for s in selections if s.get('checkpoint')]
        + [x['filename'] for x in externals])
    # One LoRA scan per dataset. `list_test_checkpoints` walks the family's whole
    # LoRA folder (and stats every match): its result only depends on (dataset, family),
    # so a 24-cell grid over 8 checkpoints of the same dataset re-scanned that folder 9
    # times for one identical answer. Memoised for the duration of THIS call only — the
    # deployed set can change between two runs.
    _ckpt_memo = {}

    def _dataset_and_checkpoints(ds_id):
        """(dataset, allowed checkpoint filenames) for this run's family, scanned once."""
        if ds_id not in _ckpt_memo:
            _ds = fds.get_dataset(user_id, ds_id)
            _allowed = {c['filename'] for c in list_test_checkpoints(_ds, run_type)} if _ds else set()
            _ckpt_memo[ds_id] = (_ds, _allowed)
        return _ckpt_memo[ds_id]

    # Preflight, as in create_run: verify that target ComfyUI can execute this
    # family's workflow using the first valid selection. The run has one family;
    # check before creating rows to return one actionable 409.
    for _sel in selections:
        _pf_ds, _pf_allowed = _dataset_and_checkpoints(_sel.get('dataset_id'))
        if not _pf_ds:
            continue
        _pf_cp = _sel.get('checkpoint')
        if _pf_cp in _pf_allowed:
            _preflight_run(user_id, run_type, _pf_cp, valid_models, _pf_allowed,
                           prompt_axis[0] or identity_prompt(_pf_ds, with_trigger=inject_trigger),
                           seeds[0], _sel.get('dataset_id'),
                           (getattr(_pf_ds, 'trigger_word', None)
                            if inject_trigger else None),
                           sampler_preset=sampler_preset)
            break
    return _dataset_and_checkpoints


def _cmp_expand_stack(combine, selections, origin_of,
                      _dataset_and_checkpoints):
    """Expand stack mode, extracted unchanged: revalidate members against their
    own dataset's deployed checkpoints, stamp provenance, collect triggers,
    expand the Cartesian product of weights, and reduce selections to the head
    LoRA. Return (combine, selections, stack_triggers, combos, members).
    In ordinary comparison mode, pass values through unchanged."""
    # --- Stack mode (combine) --------------------------------------------------
    # Comparison gives each selection its own cells, one LoRA alone per image.
    # Combine describes one stack: the first LoRA remains the tested head, owning
    # the grid column and dataset used for the default prompt. Chain subsequent
    # members in the same graph through extra_loras, already supported by the
    # Z-Image, Krea and SDXL injectors. Revalidate each member against its own
    # dataset's deployed checkpoints: graph assembly accepts permissive extras,
    # so path-injection protection belongs here.
    #
    # Each selection may supply multiple `weights`; render their Cartesian
    # product as configurations within the same run. A single weight per LoRA,
    # including legacy clients sending only `weight`, gives one combination
    # and preserves previous behavior. LoraTestImage.strength stores the head
    # weight and extra_loras JSON stores member weights, so each cell already
    # identifies its combination without another database column.
    combine = bool(combine) and len(selections) > 1
    # [(stack_extra, stack_row)] per combination, aligned with combos.
    stack_triggers = []
    combos = [None]
    members = []
    if combine:
        for sel in selections[1:]:
            _ds_i, _allowed_i = _dataset_and_checkpoints(sel.get('dataset_id'))
            if not _ds_i:
                raise ValueError(f"dataset {sel.get('dataset_id')} not found")
            fn = sel.get('checkpoint')
            if fn not in _allowed_i:
                raise ValueError(f'unknown checkpoint for {_ds_i.name}: {fn}')
            # Generation provenance identifies this member's source board checkpoint.
            # origin_of has already resolved every selected checkpoint, including
            # members, while the clicked source or current deployment tag is known.
            # Without this, only the head's parent would be known even though a blend
            # has several parents.
            _origin_i = origin_of.get(fn, (None, None))
            members.append({'filename': fn, 'weights': _combine_weights(sel),
                            'dataset_id': _ds_i.id,
                            'record_id': _origin_i[0], 'step': _origin_i[1],
                            'trigger': getattr(_ds_i, 'trigger_word', None) or None})
            if getattr(_ds_i, 'trigger_word', None):
                stack_triggers.append(_ds_i.trigger_word)
        # Each combination is (head weight, member 1 weight, ...). The last LoRA
        # varies fastest, matching the panel's displayed order.
        head_weights = _combine_weights(selections[0])
        combos = [tuple(c) for c in itertools.product(
            head_weights, *[m['weights'] for m in members])]
        selections = selections[:1]
    return combine, selections, stack_triggers, combos, members


def _cmp_build_cell_plan(valid_models, selections, combos, strengths,
                         aspects, cfgs, steps_list, steps2_list, run_type):
    """Build the cell plan, extracted unchanged: run_id and the base-major
    product of bases x selections x combinations x matrix, followed by Krea's
    stable partition placing LoRA-off controls first. Return (run_id, cell_plan)."""
    run_id = uuid.uuid4().hex
    # Materialize the original selection-major plan, then stable-partition it
    # once for Krea. Zero tested-LoRA-off controls across *all* selected
    # checkpoints therefore finish before the first non-zero tested-LoRA cell,
    # while each group's checkpoint-major and strength order stays unchanged.
    cell_plan = []
    # Base-major order, as in create_run: one base preserves the original plan;
    # multiple bases appear consecutively instead of interleaving.
    for zm in valid_models:
        for sel in selections:
            checkpoint = sel.get('checkpoint')
            for combo in combos:
                # Stacks give each LoRA its own weight, so replace the strengths axis
                # with this combination's head weight.
                combo_strengths = [combo[0]] if combo is not None else strengths
                for cell in build_matrix([checkpoint], combo_strengths, aspects, cfgs,
                                         steps_list, steps2_list, family=run_type):
                    cell_plan.append((sel, cell, combo, zm))
    cell_plan = _krea_zero_strength_first(
        cell_plan, run_type, lambda planned: planned[1][1])
    return run_id, cell_plan


def _cmp_enqueue_cells(user_id, cell_plan, members, _dataset_and_checkpoints,
                       knobs, run_type, batch_axis, prompt_axis, seeds, seed,
                       run_id, extra_loras, origin_of,
                       combine, stack_triggers, available_classes,
                       inject_trigger=True):
    """Materialize the plan, extracted unchanged. For each planned cell, persist
    the identified stack, aspect dimensions, batch/prompt/seed axes and
    LoraTestImage row, then enqueue its workflow builder with the base bound
    as a default rather than captured late. Return the created IDs."""
    ids = []
    for sel, cell, combo, zm in cell_plan:
        # Member weights for this combination. Graph-facing stack_extra keeps the
        # always-on format plus the member's trigger (spelled at graph build as that
        # LoRA learned it); only the persisted copy carries member identity so the
        # stack view can recover its dataset and trigger without guessing.
        stack_extra, stack_row = [], []
        for i, m in enumerate(members):
            entry = {'filename': m['filename'], 'strength': combo[i + 1]}
            stack_extra.append({**entry, 'trigger': m['trigger']})
            # Only the persisted copy carries provenance; stack_extra retains the
            # always-on format (and the trigger) the workflow builder reads.
            stack_row.append({**entry, 'combined': True,
                              'dataset_id': m['dataset_id'], 'trigger': m['trigger'],
                              'record_id': m['record_id'], 'step': m['step']})
        ds, allowed = _dataset_and_checkpoints(sel.get('dataset_id'))
        if not ds:
            raise ValueError(f"dataset {sel.get('dataset_id')} not found")
        checkpoint = sel.get('checkpoint')
        if checkpoint not in allowed:
            raise ValueError(f'unknown checkpoint for {ds.name}: {checkpoint}')
        cp, strength, cell_aspect, cell_cfg, cell_steps, cell_steps2 = cell
        width, height = _aspect_dims(cell_aspect, run_type, knobs['resolution_tier'],
                                     knobs['resolution_multiplier'])
        for batch_lora in batch_axis:  # Batch axis: without, then with each selected LoRA
          row_extra = extra_loras + stack_row + ([{**batch_lora, 'batch': True}] if batch_lora else [])
          wf_extra = extra_loras + stack_extra + ([batch_lora] if batch_lora else [])
          cell_extra_json = json.dumps(row_extra) if row_extra else None
          for axis_prompt in prompt_axis:  # Prompt batch axis: one pass per selected prompt
           cell_prompt = axis_prompt or identity_prompt(ds, with_trigger=inject_trigger)
           for cell_seed in seeds:
            img = LoraTestImage(dataset_id=ds.id, checkpoint=cp, strength=strength,
                                seed=cell_seed, run_seed=seed, run_id=run_id,
                                status='pending', z_model=zm, aspect=cell_aspect,
                                prompt=cell_prompt, cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                extra_loras=cell_extra_json,
                                negative=knobs['negative'], sampler=knobs['sampler'],
                                sampler_preset=knobs['sampler_preset'],
                                scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                detail_amount=knobs['detail_amount'],
                                resolution_tier=knobs['resolution_tier'],
                                resolution_multiplier=knobs['resolution_multiplier'],
                                init_image=knobs['init_image'], denoise=knobs['denoise'],
                                hires_scale=knobs['hires_scale'],
                                hires_denoise=knobs['hires_denoise'],
                                finish_sharpen=knobs['finish_sharpen'],
                                finish_grain=knobs['finish_grain'],
                                inject_trigger=None if inject_trigger else False,
                                record_id=origin_of.get(cp, (None, None))[0],
                                step=origin_of.get(cp, (None, None))[1])
            _persist_and_enqueue_cell(
                img, user_id, ds.id, cell_prompt,
                # Bind `zm` as a default, not a late capture: graphs are built later,
                # and late binding would give every cell the last cell's base.
                lambda _zm=zm: _build_cell_workflow(user_id, cp, strength, cell_prompt,
                                     cell_seed, _zm, allowed, width=width,
                                     height=height, cfg=cell_cfg, steps=cell_steps,
                                     steps2=cell_steps2, dataset_id=ds.id,
                                     train_type=run_type, extra_loras=wf_extra,
                                     negative=knobs['negative'], sampler=knobs['sampler'],
                                     sampler_preset=knobs['sampler_preset'],
                                     scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                     detail_amount=knobs['detail_amount'],
                                     hires_scale=knobs['hires_scale'],
                                     hires_denoise=knobs['hires_denoise'],
                                     # Combined stack: all triggers, head first.
                                     # An unchecked Trigger word box disables
                                     # all triggers, including stacked ones.
                                     trigger_word=(([ds.trigger_word] + stack_triggers
                                                    if combine else ds.trigger_word)
                                                   if inject_trigger else None),
                                     available_classes=available_classes))
            ids.append(img.id)
    return ids


def create_comparison_run(user_id, selections, strengths, settings=None, *,
                          combine=None, prompts=None, external_loras=None) -> dict:
    """Launch one comparison run across LoRAs. `selections` contains
    [{dataset_id, checkpoint}]. Entries may carry `record_id`/`step`, which the
    LoRA Canvas knows from the selected checkpoint. Stamp these unchanged on
    cells, or read origin from the deployment tag (see checkpoint_origins).
    All cells share run_id and seed for fairness. Use the shared `prompt` when
    provided, otherwise each cell's dataset identity_prompt with its own trigger.
    One selection produces a single-LoRA run.

    Generate parity (2026-07-01): all cells share always-on LoRAs, SDXL steps2
    and global negative/sampler/scheduler/precision/detail/tier settings, gated
    and validated by family through _sanitize_gen_knobs.

    `combine=True` with at least two selections switches from comparison
    (one LoRA alone per cell) to a stack: load selected LoRAs together with
    each selection's `weight` and inject all dataset triggers. Replace the
    `strengths` axis with the head LoRA's weight, since each member has its own
    weight. A run still uses one family: mixing Krea and SDXL is rejected with
    a message naming the incompatible families and their bases/workflows.

    `external_loras` (Canvas plugin nodes) accepts [{filename, strength}] from
    any models/loras file and stacks them on every cell through `extra_loras`,
    without restricting them to the family pool. Missing files are hard errors,
    never silent skips; architecture preflight covers them like checkpoints."""
    # Same door as create_run: one object in, the verbatim body below keeps
    # reading the locals it always has.
    settings = settings or StudioGenSettings()
    seed = settings.seed
    prompt = settings.prompt
    z_model = settings.z_model
    z_models = settings.z_models
    aspects = settings.aspects
    cfgs = settings.cfgs
    steps_list = settings.steps_list
    steps2_list = settings.steps2_list
    count = settings.count
    permanent_loras = settings.permanent_loras
    batch_loras = settings.batch_loras
    negative = settings.negative
    sampler = settings.sampler
    sampler_preset = settings.sampler_preset
    scheduler = settings.scheduler
    weight_dtype = settings.weight_dtype
    detail_amount = settings.detail_amount
    resolution_tier = settings.resolution_tier
    resolution_multiplier = settings.resolution_multiplier
    init_image = settings.init_image
    denoise = settings.denoise
    hires_scale = settings.hires_scale
    hires_denoise = settings.hires_denoise
    finish_sharpen = settings.finish_sharpen
    finish_grain = settings.finish_grain
    # Trigger word checkbox follows create_run: False means the raw prompt.
    inject_trigger = settings.inject_trigger is not False
    run_type, models = _cmp_resolve_run_family(selections)
    valid_models, seed, count, seeds, prompt_axis = _cmp_seed_and_prompts(
        models, z_model, z_models, seed, count, prompt, prompts, family=run_type)
    extra_loras, externals = _cmp_collect_extra_loras(
        run_type, permanent_loras, external_loras)
    batch_axis, knobs = _cmp_cell_knobs(
        run_type, batch_loras, negative,
        sampler, scheduler, weight_dtype,
        detail_amount, resolution_tier, resolution_multiplier, init_image,
        denoise, sampler_preset=sampler_preset,
        hires_scale=hires_scale, hires_denoise=hires_denoise,
        finish_sharpen=finish_sharpen, finish_grain=finish_grain)

    _dataset_and_checkpoints = _cmp_preflight(
        user_id, run_type, selections, externals, valid_models,
        prompt_axis, seeds, inject_trigger=inject_trigger,
        sampler_preset=knobs['sampler_preset'])

    # Read target ComfyUI classes once per run (see create_run) and resolve
    # NODE_CLASS_ALIASES to the actual registered names.
    available_classes = _target_node_classes()
    # Resolve each selected LoRA's origin (run and step) from explicit Canvas
    # input or its deployment tag. Resolve once per distinct filename.
    origin_of = checkpoint_origins(
        [s.get('checkpoint') for s in selections if s.get('checkpoint')],
        {s['checkpoint']: s for s in selections
         if s.get('checkpoint') and s.get('record_id') is not None
         and s.get('step') is not None},
        family=run_type)

    combine, selections, stack_triggers, combos, members = _cmp_expand_stack(
        combine, selections, origin_of, _dataset_and_checkpoints)

    run_id, cell_plan = _cmp_build_cell_plan(
        valid_models, selections, combos, strengths, aspects, cfgs,
        steps_list, steps2_list, run_type)

    ids = _cmp_enqueue_cells(
        user_id, cell_plan, members, _dataset_and_checkpoints, knobs,
        run_type, batch_axis, prompt_axis, seeds, seed, run_id,
        extra_loras, origin_of, combine, stack_triggers,
        available_classes, inject_trigger=inject_trigger)
    # Use len(members), not len(stack_extra), which now lives inside the loop
    # and would represent the last combination or be absent for an empty plan.
    # Log the combination count to explain unexpectedly large image totals.
    logger.info(f"lora-test: {'combined' if combine else 'comparison'} run {run_id} -> "
                f"{len(ids)} cell(s), {len(selections) + len(members)} LoRA, "
                f"{len(combos) if combine else 1} combinaison(s), "
                f"{len(prompt_axis)} prompt(s), seed {seed}")
    return {'created': len(ids), 'seed': seed, 'count': count, 'run_id': run_id, 'ids': ids}


def _run_owned(user_id, run_id) -> bool:
    """Single-user app: every run belongs to the local user - no cross-user
    ownership DB to consult (SRC checked every cell's dataset against
    `user_id`)."""
    return True


def cancel_run(user_id, dataset_id=None, run_id=None) -> int:
    """Cancel only cells whose exact ComfyUI work is safely gone.

    The entire selection and cancellation sweep holds the same GPU arbiter as
    ``process_one``. A worker therefore cannot claim the next grid cell between
    two safe cancellations. An uncertain prompt stays attached to its pending
    cell and is rendered as paused until a later Cancel can reconcile it.
    """
    if run_id is not None:
        if not _run_owned(user_id, run_id):
            return 0
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            return 0

    with GPU_ARBITER_LOCK:
        if run_id is not None:
            rows = (_cells()
                    .filter_by(run_id=run_id, status='pending')
                    .filter(LoraTestImage.filename.is_(None)).all())
        else:
            rows = (_cells()
                    .filter_by(dataset_id=dataset_id, status='pending')
                    .filter(LoraTestImage.filename.is_(None)).all())

        cancelled = 0
        for img in rows:
            if not img.job_id:
                img.status = 'cancelled'
                cancelled += 1
                continue
            try:
                safe = queue_manager.cancel_job(img.job_id, str(user_id), 'image')
            except Exception:
                logger.exception('lora-test: could not safely cancel queue job %s', img.job_id)
                safe = False
            if not safe:
                # Recover a request interrupted after the queue row committed but
                # before this cell could be committed. A terminal cancelled queue
                # row is durable proof that clearing this cell is safe.
                queue_row = ImageGenerationQueue.query.filter_by(job_id=img.job_id).first()
                safe = queue_row is not None and queue_row.status == 'cancelled'
            if not safe:
                continue
            img.status = 'cancelled'
            img.job_id = None
            cancelled += 1

        if cancelled:
            db.session.commit()
        return cancelled


def confirm_unknown_comfyui_restart(user_id, *, dataset_id=None, run_id=None,
                                    restart_confirmed=False) -> int:
    """Make exactly one unknown-submit Test Studio cell resumable again.

    A user must explicitly confirm an external ComfyUI restart at the route
    boundary. We then cancel only the stalled queue job identified by the raw
    barrier and its one linked pending cell in the same commit. Known prompt
    barriers keep their stricter remote reconciliation path.
    """
    if restart_confirmed is not True:
        raise ValueError('Confirm that you restarted ComfyUI before clearing this paused job.')
    if (dataset_id is None) == (run_id is None):
        raise ValueError('choose exactly one Test Studio run or dataset')
    if run_id is not None:
        if not _run_owned(user_id, run_id):
            raise ValueError('run not found')
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            raise ValueError('dataset not found')

    with GPU_ARBITER_LOCK:
        owner = queue_manager.get_comfyui_stalled_barrier()
        if (owner is None or owner.get('kind') != 'unknown_submit'
                or not isinstance(owner.get('job_id'), str)
                or owner.get('prompt_id') is not None):
            raise RuntimeError('There is no unknown ComfyUI submission awaiting restart confirmation.')
        job_id = owner['job_id']

        # The queue job is the authoritative remote identity. The cell match is
        # deliberately exact too: metadata protects current rows, while job_id
        # protects legacy rows created before cell_id/run_id were persisted.
        queue_job = (ImageGenerationQueue.query
                     .filter_by(job_id=job_id, user_id=str(user_id), status='stalled')
                     .filter(ImageGenerationQueue.comfyui_prompt_id.is_(None)).first())
        if queue_job is None:
            raise RuntimeError('The paused ComfyUI job changed; refresh its status before retrying.')

        try:
            cell_id = int(owner.get('cell_id'))
        except (TypeError, ValueError):
            raise RuntimeError('This paused job has no exact Test Studio cell to recover.')
        if str(cell_id) != owner.get('cell_id'):
            raise RuntimeError('This paused job has an invalid Test Studio cell identity.')

        # lds-allow-bare-lora-test-query: resolved by job_id (and an explicit id)
        # — recovery identifies ONE known row, whatever kind it is.
        cell_query = (LoraTestImage.query.filter_by(id=cell_id, job_id=job_id, status='pending')
                      .filter(LoraTestImage.filename.is_(None)))
        if run_id is not None:
            cell_query = cell_query.filter_by(run_id=str(run_id))
        else:
            cell_query = cell_query.filter_by(dataset_id=dataset_id)
        cell = cell_query.first()
        if cell is None:
            raise RuntimeError('The paused ComfyUI job does not belong to this Test Studio view.')
        for key, value in (('dataset_id', str(cell.dataset_id)), ('run_id', cell.run_id)):
            if key in owner and owner.get(key) != value:
                raise RuntimeError('The paused ComfyUI job identity does not match its Test Studio cell.')

        if not queue_manager.confirm_unknown_comfyui_restart(
                job_id, str(user_id), restart_confirmed=True, commit=False):
            db.session.rollback()
            raise RuntimeError('The paused ComfyUI job changed; refresh its status before retrying.')
        cell.status = 'cancelled'
        cell.job_id = None
        cell.error = None
        try:
            db.session.commit()  # Test Studio cell + exact queue job + raw barrier
        except Exception as exc:
            db.session.rollback()
            logger.exception('lora-test: could not confirm unknown ComfyUI restart for %s', job_id)
            raise RuntimeError('Could not record the ComfyUI recovery; nothing was resumed.') from exc
        return 1


def resume_run(user_id, dataset_id=None, run_id=None) -> dict:
    """Resume a stopped run by re-enqueuing all 'cancelled'/'failed' cells with
    their stored prompt/seed/model/format/strength settings.

    When `run_id` is provided, target that run; otherwise retain the historical
    `dataset_id` scope."""
    if run_id is not None:
        if not _run_owned(user_id, run_id):
            raise ValueError('run not found')
        reason = gpu_busy_reason()
        if reason:
            raise GpuBusyError(reason)
        if _active_run_count():
            raise ValueError('a test run is already in progress')
        rows = (_cells().filter_by(run_id=run_id)
                .filter(LoraTestImage.status.in_(['cancelled', 'failed'])).all())
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            raise ValueError('dataset not found')
        reason = gpu_busy_reason()
        if reason:
            raise GpuBusyError(reason)
        if _active_run_count(dataset_id):
            raise ValueError('a test run is already in progress')
        rows = (_cells().filter_by(dataset_id=dataset_id)
                .filter(LoraTestImage.status.in_(['cancelled', 'failed'])).all())
    if not rows:
        raise ValueError('no cell to resume')
    # A run_id can span datasets. Resolve and cache each cell's dataset, and
    # derive family from its checkpoint folder rather than dataset train_type,
    # which can differ for a dataset trained with several pipelines. Cache the
    # allowlist by (dataset, family).
    ds_cache, allowed_cache = {}, {}
    base_cache = {}

    def _ds(did):
        if did not in ds_cache:
            ds_cache[did] = fds.get_dataset(user_id, did)
        return ds_cache[did]

    def _allowed(did, fam):
        key = (did, fam)
        if key not in allowed_cache:
            d = _ds(did)
            allowed_cache[key] = {c['filename'] for c in list_test_checkpoints(d, fam)} if d else set()
        return allowed_cache[key]
    # Read target ComfyUI classes once for resume and resolve NODE_CLASS_ALIASES
    # to actual registered names, as in create_run.
    available_classes = _target_node_classes()
    n = 0
    for img in rows:
        cell_ds = _ds(img.dataset_id)
        # Checkpoint folder (train_type fallback) determines family, allowlist, base, dimensions and workflow.
        cell_family = (family_of_lora(img.checkpoint)
                       or getattr(cell_ds, 'train_type', None) or 'zimage').lower()
        allowed = _allowed(img.dataset_id, cell_family)
        if not cell_ds or img.checkpoint not in allowed:
            continue  # skip a missing dataset/checkpoint
        # A replay uses the saved base or fails explicitly if it was removed.
        # Selecting the first remaining model would silently change the experiment.
        try:
            if cell_family not in base_cache:
                base_cache[cell_family] = _require_family_bases(cell_family)
            z_model = _select_base_models(base_cache[cell_family], img.z_model,
                                         family=cell_family)[0]
        except ValueError as exc:
            img.status = 'failed'
            img.error = str(exc)[:400]
            db.session.commit()
            continue
        aspect = img.aspect if img.aspect in TEST_ASPECTS else DEFAULT_ASPECT
        # Persisted resolution tier and multiplier reproduce initial dimensions.
        # Legacy cells without the column use the fixed table and multiplier 1.0.
        width, height = _aspect_dims(aspect, cell_family, getattr(img, 'resolution_tier', None),
                                     getattr(img, 'resolution_multiplier', None) or 1.0)
        prompt = ((img.prompt or '').strip()
                  or identity_prompt(cell_ds,
                                     with_trigger=getattr(img, 'inject_trigger', None) is not False))
        seed = img.seed or random.randint(1, 2**31 - 1)
        # Reapply each cell's stored always-on LoRAs unchanged on resume.
        try:
            cell_extra = json.loads(img.extra_loras) if img.extra_loras else None
        except (json.JSONDecodeError, TypeError):
            cell_extra = None
        # Read stacked member triggers from persisted combined entries so resume
        # uses the launch prompt. Previously only the head trigger survived;
        # enqueue uses [head] + stack_triggers.
        _stack_trigs = [e.get('trigger') for e in (cell_extra or [])
                        if isinstance(e, dict) and e.get('combined') and e.get('trigger')]
        try:
            # Read all global settings from the cell for faithful Generate-parity resume.
            workflow = _build_cell_workflow(user_id, img.checkpoint, img.strength,
                                            prompt, seed, z_model, allowed,
                                            width=width, height=height,
                                            cfg=img.cfg, steps=img.steps, steps2=img.steps2,
                                            dataset_id=img.dataset_id,
                                            train_type=cell_family, extra_loras=cell_extra,
                                            negative=getattr(img, 'negative', None),
                                            sampler=getattr(img, 'sampler', None),
                                            sampler_preset=getattr(img, 'sampler_preset', None),
                                            scheduler=getattr(img, 'scheduler', None),
                                            weight_dtype=getattr(img, 'weight_dtype', None),
                                            detail_amount=getattr(img, 'detail_amount', None),
                                            # Per-run hi-res fix persisted on the cell: a
                                            # resume renders what it replaces, not what
                                            # Settings says today. NULL (legacy cell) =
                                            # the setting, exactly as at its first run.
                                            hires_scale=getattr(img, 'hires_scale', None),
                                            hires_denoise=getattr(img, 'hires_denoise', None),
                                            # Preserve an unchecked Trigger word choice
                                            # (False column) on resume; otherwise use
                                            # head and stack triggers as at enqueue.
                                            trigger_word=(None
                                                          if getattr(img, 'inject_trigger', None) is False
                                                          else ([getattr(cell_ds, 'trigger_word', None)]
                                                                + _stack_trigs
                                                                if _stack_trigs
                                                                else getattr(cell_ds, 'trigger_word', None))),
                                            available_classes=available_classes)
            if cell_family in TRAINED_IMAGE_FAMILIES:
                preflight_family(cell_family, [workflow])
            job_id = _enqueue_cell(user_id, img.dataset_id, workflow, prompt,
                                   cell_id=img.id, run_id=img.run_id)
            img.status = 'pending'
            img.filename = None
            img.job_id = job_id
            img.seed = seed
            img.error = None  # clean slate on a successful re-enqueue
            db.session.commit()
            n += 1
        except Exception as e:
            img.status = 'failed'
            img.error = str(e)[:400] or 'resume failed'
            db.session.commit()
    return {'resumed': n}


# --- Completion linking (called from job_queue) --------------------------------
def _cleanup_output_file(filename, failed):
    """Best-effort removal of an orphaned OUTPUT_DIR file from a completed job
    whose row is no longer valid."""
    if failed or not filename:
        return
    out_dir = _comfy_output_dir()
    if not out_dir:
        return
    try:
        p = os.path.join(out_dir, filename)
        if os.path.isfile(p):
            os.remove(p)
    except OSError:
        pass


def _photo_finish_or_none(img):
    """The finishing module, or None with ONE plain log line when it cannot load.

    numpy lives in requirements-ml, not in the base install (start.bat). Imported
    only once a stage is actually ON — so an install that never turned finishing
    on never imports it — and an install that turned it on without the ML
    requirements gets a sentence naming the fix, not a traceback per cell."""
    try:
        from ..utils import photo_finish
        return photo_finish
    except ImportError as exc:
        logger.warning('lora-test link: finishing pass skipped for image %s — %s '
                       '(install the ML requirements to enable it)',
                       getattr(img, 'id', None), exc)
        return None


def _finish_test_image(img, dst):
    """The app-side finishing pass over a just-linked LoraTestImage file.

    Two kinds of row land here and they are finished DIFFERENTLY, on purpose:

    * an ordinary Studio cell (derivation_kind NULL): sharpen/grain from the
      cell's OWN persisted knobs (the panel), off when the row says nothing.
      No colour match — it is text-to-image, there is no "before" to match.
    * a ◉ Canvas ✨ improve (CANVAS_IMAGE_IMPROVE): exactly what the dataset
      ✨ improve gets — `improve.*` from Settings, with the PARENT as the colour
      reference — so the two surfaces the same button lives on finish the same
      way. The engine is read off the row: `improve_profile` is written for a
      Klein pass and left NULL for SeedVR2 (lora_test_studio.improve_canvas_image),
      which is the one fact that decides whether colour matching may run.

    Anything else (📷 camera angles, unknown kinds) is left alone. Wrapped
    whole: a finishing pass is a nicety, and it may never be the reason a render
    the user waited for is lost."""
    try:
        kind = getattr(img, 'derivation_kind', None)
        if kind is None:
            sharpen = getattr(img, 'finish_sharpen', None) or 0.0
            grain = getattr(img, 'finish_grain', None) or 0.0
            if sharpen <= 0 and grain <= 0:
                return
            # The grain's colour mix is the one dial the cell does not carry: it
            # follows `improve.grain_saturation`, read through the SAME clamp the
            # dataset lane uses (NaN-safe, honours 0 = luminance-only) rather than
            # a bare `or 0.2` that would turn a legitimate 0 into 0.2.
            saturation = fds._finishing_profile('klein')['grain_saturation']
            photo_finish = _photo_finish_or_none(img)
            if photo_finish is None:
                return
            photo_finish.apply_to_file(
                dst, seed=int(img.id or 0), sharpen=float(sharpen), grain=float(grain),
                grain_saturation=saturation)
            return
        if kind != CANVAS_IMAGE_IMPROVE:
            return
        engine = 'klein' if getattr(img, 'improve_profile', None) else 'seedvr2'
        profile = fds._finishing_profile(engine)
        if not any(profile[k] > 0 for k in ('colour_strength', 'sharpen', 'grain')):
            return
        photo_finish = _photo_finish_or_none(img)
        if photo_finish is None:
            return
        reference_path = None
        if profile['colour_strength'] > 0 and getattr(img, 'parent_image_id', None):
            # lds-allow-bare-lora-test-query: the parent of a derived row, by id.
            parent = db.session.get(LoraTestImage, img.parent_image_id)
            if parent is not None and parent.filename:
                candidate = os.path.join(fds._dataset_dir(parent.dataset_id), parent.filename)
                if os.path.isfile(candidate):
                    reference_path = candidate
        photo_finish.apply_to_file(
            dst, reference_path=reference_path, seed=int(img.id or 0),
            colour_strength=profile['colour_strength'], sharpen=profile['sharpen'],
            grain=profile['grain'], grain_saturation=profile['grain_saturation'])
    except Exception:
        logger.exception('lora-test link: finishing pass failed for image %s',
                         getattr(img, 'id', None))


def link_completed_test_image(job_id, filename, failed=False, reason=None):
    """Attach a finished studio job to its LoraTestImage row.

    Mirror of link_completed_dataset_image: runs in the queue monitor thread
    whose SQLAlchemy session may hold a STALE read snapshot - if the first
    lookup misses, rollback (end the transaction) and re-read on a fresh
    snapshot before concluding the row doesn't exist.
    `reason` (the job row's error_message: a ComfyUI 400 validation body / node
    execution error / timeout) is persisted on the failed cell so the tile can
    say WHY it's empty instead of a mute red square (P0-b)."""
    # lds-allow-bare-lora-test-query: resolved by job_id. This is the completion
    # callback and it MUST see derived rows — an ✨ improve started from the canvas
    # lands through exactly this path, so filtering it out here would leave every
    # improvement pending forever.
    img = LoraTestImage.query.filter_by(job_id=job_id).first()
    if img is None:
        db.session.rollback()  # drop the stale read snapshot, then re-read
        # lds-allow-bare-lora-test-query: same lookup, fresh snapshot.
        img = LoraTestImage.query.filter_by(job_id=job_id).first()
    if img is None:
        logger.warning(f"lora-test link: no LoraTestImage row for job {job_id}")
        _cleanup_output_file(filename, failed)  # job without a valid row (cancelled/resumed): orphaned output
        return
    # Finalize only still-pending cells. A late completion from a cancelled or
    # resumed row (new job_id or nonpending status) must not overwrite the valid
    # run. Discard its output file instead of moving it.
    if img.status != 'pending':
        logger.info(f"lora-test link: row {img.id} already {img.status} for job {job_id} - skipped")
        _cleanup_output_file(filename, failed)
        return
    if failed:
        img.status = 'failed'
        img.error = (reason
                     or 'Generation failed (see 🪵 Server log in Settings for the ComfyUI error).')
    else:
        img.filename = filename
        img.status = 'done'
        # Bring the completed file into the per-dataset dir (served by
        # /api/dataset/<id>/img/<filename>, cleaned with the dataset). Copy from
        # ComfyUI's output dir (a cross-volume path or a just-written lock
        # cannot shutil.move); dest present is enough even if the source stays.
        # If the file isn't on disk — custom output path, or none configured —
        # fetch it over the /view API instead (path-independent). See GH #2.
        from ..utils import comfy_fs
        dst = os.path.join(fds._dataset_dir(img.dataset_id), filename)
        out_dir = _comfy_output_dir()
        src = os.path.join(out_dir, filename) if out_dir else None
        claimed = comfy_fs.claim_output_file(src, dst) if src else os.path.isfile(dst)
        if not claimed:
            from ..utils.comfyui import fetch_output_image_bytes
            data = fetch_output_image_bytes(filename)
            if data:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'wb') as f:
                    f.write(data)
            else:
                # The result vanished (not on disk, /view fetch failed) — mark the
                # cell failed WITH a reason rather than leaving a 'done' row whose
                # <img> would 404 into a mute broken tile (P0-b, mirrors the dataset
                # fan-out's fail path).
                img.filename = None
                img.status = 'failed'
                img.error = ('The finished image could not be retrieved from ComfyUI '
                             '(not on disk, and the /view API fetch failed).')
                logger.warning(f"lora-test link: file not on disk and /view API fetch failed for {filename}")
        # The app-side finishing pass, on the file now at `dst` — only when the
        # link succeeded (a row just flipped to 'failed' above has no file to
        # finish). Off by default; see _finish_test_image for which rows get what.
        if img.status == 'done':
            _finish_test_image(img, dst)
    db.session.commit()


# --- ✨ Upscale & improve, from the ◉ Canvas lightbox --------------------------
# The board's pictures are `LoraTestImage` rows, so the dataset improve route
# (`/api/dataset/image/<id>/improve`, which resolves a `FaceDatasetImage`) cannot
# serve them: the two tables have INDEPENDENT id spaces, and passing a board id to
# it does not 404 — it finds a real, unrelated dataset image and improves that
# one. A silent wrong answer is the worst failure available here, which is why
# this lane exists instead of a one-line reuse of the other route.
#
# What it does NOT re-implement: the engine choice, the readiness preflight and
# the Klein/SeedVR2 hand-off are `face_dataset_service`'s, called as-is. So the
# 409 bodies that offer to install SeedVR2 on demand are the same ones, and a
# change to either engine reaches both surfaces at once.

# Worded as the user reads them, so the button can explain itself BEFORE the
# click instead of surfacing an error after it.
IMPROVE_SOURCE_GONE = 'that image is no longer in the library'
REPAIR_NOT_DONE = 'this image is still rendering'
REPAIR_FILE_GONE = 'that image file is no longer on disk'
REPAIR_NEEDS_PROMPT = 'a prompt is required - say what should be painted in that area'


def repair_generated_image(user_id, image_id, boxes, prompt, *,
                           seed=None, mask=None) -> dict | None:
    """Repaint a drawn zone of a GENERATED image from a free prompt.

    Asked for by .samexit on Discord: "add the inpaint feature immediately after
    the first generation, to avoid having to completely regenerate the image just
    to fix a small detail". Until now a bad hand or a stray object meant throwing
    the whole picture away and rolling the dice again.

    The dataset lane already does this (face_dataset_service.repair_image_region);
    what a generated image lacks is a FaceDatasetImage row to hang it on. So the
    row is addressed by its LoraTestImage id and the filename is resolved HERE —
    the client never names a path. That is deliberate: this call overwrites a file
    in place, and a client-supplied name is how an overwrite becomes an arbitrary
    write.

    Returns None when the image is unknown or not the caller's (so the route
    answers 404 without leaking which of the two it was). Raises ValueError with
    one of the REPAIR_* sentences, or keh.KleinModelGone.
    """
    row = db.session.get(LoraTestImage, image_id)
    if row is None:
        return None
    ds = fds.get_dataset(user_id, row.dataset_id)
    if not ds:
        return None
    if row.status != 'done' or not row.filename:
        raise ValueError(REPAIR_NOT_DONE)
    path = os.path.join(fds._dataset_dir(row.dataset_id), row.filename)
    if not os.path.isfile(path):
        raise ValueError(REPAIR_FILE_GONE)
    text = (prompt or '').strip()
    if not text:
        raise ValueError(REPAIR_NEEDS_PROMPT)

    from . import watermark_klein
    from . import klein_edit_helper as keh
    if not watermark_klein.is_available():
        raise ValueError('Klein is not ready (ComfyUI unreachable or models missing)')
    klein_model = fds.dataset_klein_model(ds)
    if klein_model and not keh.klein_model_on_disk(klein_model):
        raise keh.KleinModelGone(klein_model)

    # The SAME safety as the dataset lane, reused rather than re-implemented: an
    # upright disposable sibling (the boxes were drawn against EXIF orientation),
    # the file preserved before anything is written, and the edit promoted only
    # once Klein succeeded. A failed repair leaves the picture exactly as it was.
    staged = fds._stage_oriented_watermark_edit(path)
    if not staged:
        raise ValueError('could not stage the image (EXIF orientation)')
    # A painted mask is sized against the STAGED frame — the upright image the
    # browser drew it on. Same decoder as the dataset lane, not a second copy.
    pil_mask = None
    if mask is not None:
        try:
            pil_mask = fds.decode_repair_mask_for(staged, mask)
        except ValueError:
            fds._discard_staged_watermark_edit(staged)
            raise
    if not fds._preserve_original(path):
        fds._discard_staged_watermark_edit(staged)
        raise ValueError('could not preserve the original; your file was left unchanged')
    # One step of undo, from the CURRENT pixels (fds.repair_snapshot_path explains
    # why this is not the write-once .orig sibling).
    fds.take_repair_snapshot(path)
    try:
        # A brush stroke goes through the full frame (Klein sees the whole
        # picture); a drawn box keeps the cheaper crop-and-stitch lane, which
        # the free `text` is what selects — the same call with NO prompt is the
        # 🧽 clean, and that one re-renders the entire photo (2026-08-31).
        if pil_mask is not None:
            ok, err = watermark_klein.inpaint_mask_klein(
                user_id, staged, mask=pil_mask, seed=seed,
                klein_model=klein_model, prompt=text)
        else:
            ok, err = watermark_klein.inpaint_watermark_klein(
                user_id, staged, boxes, seed=seed, klein_model=klein_model, prompt=text)
        if not ok:
            raise ValueError((err or {}).get('detail') or 'the repair failed')
        if not fds._promote_staged_watermark_edit(staged, path):
            raise ValueError('the repair rendered but could not be written back')
    finally:
        fds._discard_staged_watermark_edit(staged)
    return {'ok': True, 'filename': row.filename}


def undo_generated_repair(user_id, image_id) -> dict | None:
    """↩ Undo the last ✦ Repair of a GENERATED image.

    Same one-step contract as the dataset lane, and the same reason: an inpaint
    is a dice roll, so iterating on the sentence has to be cheap. Asked for by a
    user on Discord after the repair shipped.
    """
    row = db.session.get(LoraTestImage, image_id)
    if row is None:
        return None
    if not fds.get_dataset(user_id, row.dataset_id):
        return None
    if not row.filename:
        return None
    path = os.path.join(fds._dataset_dir(row.dataset_id), row.filename)
    if not os.path.isfile(path):
        raise ValueError(REPAIR_FILE_GONE)
    return {'ok': True, 'undone': fds.undo_repair_at(path), 'filename': row.filename}


IMPROVE_FILE_GONE = 'that image file is no longer on disk'
IMPROVE_NOT_DONE = 'this image is still rendering'


def improve_canvas_image(user_id, image_id, engine=None):
    """Queue one non-destructive ✨ Upscale & improve of a board image.

    The source row and its file are never modified: the result is a SEPARATE row
    that copies `record_id`/`step`, so it appears in the same checkpoint gallery,
    right next to the picture it was made from, and can be pinned onto the board —
    which is the whole request.

    Returns ``{'candidate_id', 'job_id', 'engine'}``, or ``None`` when the image
    is not the caller's. Clicking twice while one is in flight returns the SAME
    candidate rather than spending the GPU on a duplicate. A candidate that
    FAILED does not block a new attempt — pressing ✨ again is how you retry,
    because the Test Studio's own resume path deliberately no longer picks these
    rows up (it would re-queue them as Z-Image cells, which is the wrong engine
    and the wrong workflow).

    Improves CHAIN: a finished improve result (or a 📷 camera view) is as valid
    a source as the render it came from, so Klein detail can be followed by a
    SeedVR2 resolution pass on the same picture. Each pass is its own click and
    its own row; only a source still rendering is refused.
    """
    row = db.session.get(LoraTestImage, image_id)
    if row is None:
        return None
    ds = fds.get_dataset(user_id, row.dataset_id)
    if not ds:
        return None
    # A derived row is a legitimate source — asked for from the gallery on a
    # phone, refused with "cannot be improved again", and the refusal had no
    # ground to stand on: an upscale of an upscale (or of a 📷 camera view) is
    # still the SAME picture, only worked further — nothing compounds the way a
    # camera view OF a camera view invents a second backdrop. It is also the
    # only way to run Klein detail THEN SeedVR2 resolution on one picture.
    # Every chained candidate copies record_id/step/checkpoint from its source,
    # so it lands in the same gallery, and the in-flight idempotence below is
    # keyed on THIS row's id — a chain cannot collide with its parent's slot.
    if row.status != 'done' or not row.filename:
        raise ValueError(IMPROVE_NOT_DONE)
    source_path = os.path.join(fds._dataset_dir(row.dataset_id), row.filename)
    if not os.path.isfile(source_path):
        raise ValueError(IMPROVE_FILE_GONE)

    # Idempotent while one is ACTUALLY in flight (pending). A finished candidate
    # does not block a second one: unlike the dataset lane there is no keep/reject
    # review to force here — the result is just another picture in the gallery.
    # lds-allow-bare-lora-test-query: this one wants the EXACT OPPOSITE of
    # _cells() — it looks for derived rows on purpose.
    active = (LoraTestImage.query
              .filter_by(parent_image_id=row.id, derivation_kind=CANVAS_IMAGE_IMPROVE,
                         status='pending')
              .order_by(LoraTestImage.id.desc()).first())
    if active is not None and active.job_id:
        return {'candidate_id': active.id, 'job_id': active.job_id,
                'engine': engine or ''}

    engine = fds.resolve_improve_engine(engine)
    fds._improve_preflight(engine)          # same refusals, same actionable 409s
    prompt = fds._improve_prompt() if engine == 'klein' else ''
    # The knobs this pass will run with, computed ONCE: the same dict is stored
    # on the candidate (improve_profile) and handed to the engine, so stored
    # can never drift from executed. SeedVR2 has no knobs to record.
    profile = fds._improve_enqueue_profile(ds) if engine == 'klein' else None
    preset_rows = (profile or {}).get('generation_loras') or []
    # The preset NAME the rows came from — recorded only when it actually
    # resolved: a stale name ran as "none", and storing it would restore a
    # pick that did not decide this picture.
    preset_name = ((cfg.get('klein.improve_lora_preset') or '')
                   if preset_rows else '')

    candidate = LoraTestImage(
        dataset_id=row.dataset_id,
        # NOT NULL columns, and honest: this IS an upscale of that checkpoint's
        # render at that strength.
        checkpoint=row.checkpoint, strength=row.strength,
        status='pending', filename=None,
        # WHERE it shows up: same checkpoint, same step → same gallery, next to
        # its source. `run_id` stays NULL, which is what keeps it out of the
        # checkpoint timeline (it filters `run_id IS NOT NULL`) — a 2 MP upscale
        # spliced into an epoch-by-epoch morph would be a lie.
        record_id=row.record_id, step=row.step, run_id=None,
        seed=row.seed,           # the download name keeps the lineage
        # The pass that ACTUALLY ran. A SeedVR2 restoration sends no prompt at
        # all, so storing Klein's instruction on one would put a sentence on
        # screen that had no effect on the picture (same rule as the dataset lane).
        prompt=(prompt[:500] if engine == 'klein'
                else 'SeedVR2 upscale (no prompt — restoration pass)'),
        # Deliberately NOT copied: z_model, aspect, cfg, steps, sampler, scheduler.
        # None of them decided this image — the improve profile did — and the
        # lightbox renders them as "Made with", where a copied value would be a
        # lie about how the picture was produced. `extra_loras` follows the same
        # rule from the other side: it is set (below) to the improve preset's
        # OWN rows when one is chained, because those DID decide the picture —
        # and left NULL otherwise, never copied from the source render.
        parent_image_id=row.id,
        derivation_kind=CANVAS_IMAGE_IMPROVE,
        extra_loras=(json.dumps(
            [{'filename': r['file'], 'strength': r['strength']}
             for r in preset_rows]) if preset_rows else None),
        # Everything else the pass ran with, for ↩ "Use these improve
        # settings" — keys are a frontend contract (improveSettingsRestore.js).
        improve_profile=(json.dumps({
            'engine': 'klein',
            'klein_model': profile['klein_model'],          # None = the auto pin
            'consistency_strength': profile['lora_strength'],
            'steps': profile['sampler_steps'],
            'base_lora_strength': profile['base_lora_strength'],
            'megapixels': profile['output_megapixels'],
            'lora_preset': preset_name,
        }) if profile is not None else None),
    )
    db.session.add(candidate)
    db.session.commit()                      # row BEFORE enqueue: no orphan job
    candidate_id = candidate.id

    try:
        job_id = fds._enqueue_improve(
            engine, user_id=user_id, source=row, source_path=source_path,
            prompt=prompt, label='', dataset=ds, profile=profile,
            # `is_lora_test` is what routes the finished job back to
            # link_completed_test_image (job_queue._dispatch_completion checks it
            # FIRST, before the model_name branch), so the result lands in THIS
            # table instead of being looked up as a dataset image that does not
            # exist.
            extra_metadata={
                'is_lora_test': True,
                'dataset_id': str(row.dataset_id),
                'cell_id': candidate_id,
                'derivation_kind': CANVAS_IMAGE_IMPROVE,
                'parent_image_id': row.id,
                'action': 'upscale_improve',
                'improve_engine': engine,
            })
    except Exception:
        # No ghost row. A candidate left `pending` with no file is exactly the
        # shape `_active_run_count` counts, so a failed enqueue that kept its row
        # would have been a permanent "a test run is already in progress" if the
        # derivation filter ever slipped. Belt and braces on the worst case.
        stale = db.session.get(LoraTestImage, candidate_id)
        if stale is not None:
            db.session.delete(stale)
            db.session.commit()
        raise

    saved = db.session.get(LoraTestImage, candidate_id)
    if saved is None:                        # cancelled mid-enqueue
        return None
    saved.job_id = job_id
    db.session.commit()
    return {'candidate_id': candidate_id, 'job_id': job_id, 'engine': engine}


# --- 📷 Camera angles ---------------------------------------------------------
# The same instant, seen from somewhere else. Why this is not one of the shot
# catalog's `angle` variations, and why it needs its own base model, is written
# once in services/camera_angles.py — read that before changing anything here.
#
# ⚠️ Written into user databases. Renaming it strands every row already there.
CAMERA_ANGLE = 'camera_angle'


def camera_views_for_canvas_image(user_id, image_id, poses):
    """Queue one render per requested camera position, from ONE library picture.

    Returns ``{'views': [{'candidate_id', 'job_id', 'pose', 'label'}], 'queued'}``
    or ``None`` when the image is not the caller's.

    Shape follows ✨ improve deliberately — separate rows, source never touched,
    each result landing in the same gallery next to the picture it came from —
    with one difference that matters: **one job per view.** A single job
    rendering eight poses would share a seed, a queue slot and a failure; one
    bad pose would take the other seven with it, and nothing would appear until
    the last one finished. Eight jobs means the first picture arrives in about
    twelve seconds and a failure costs one view.

    Unlike improve there is NO idempotency guard. Two presses of ✨ on the same
    picture would produce two indistinguishable upscales, so the second is
    refused; two presses here are a legitimate request for a second take of the
    same angle (a different seed, a different guess at the hidden side), which
    is exactly what someone building a dataset does. The cost is bounded by
    camera_angles.MAX_VIEWS_PER_RUN instead.
    """
    from . import camera_angles as ca
    from . import qwen_camera_helper as qch

    row = db.session.get(LoraTestImage, image_id)
    if row is None:
        return None
    ds = fds.get_dataset(user_id, row.dataset_id)
    if not ds:
        return None
    if row.derivation_kind == CAMERA_ANGLE:
        # A view of a view compounds two invented backdrops: the second pass
        # re-invents what the first pass already invented, and the result is
        # sold as a photograph of the original scene.
        raise ValueError(ca.ALREADY_DERIVED)
    # ✨ An improve result is NOT refused, and the distinction is the point: an
    # upscale is the SAME scene from the SAME viewpoint, only cleaner — which
    # makes it the best source this lane can get, not a compounded guess. The
    # first version of this guard refused every `derivation_kind`, and pointed
    # at a real library it was wrong on sight: the newest six tiles were all
    # improve results, so the verb read as broken on the pictures people
    # actually keep.
    if row.status != 'done' or not row.filename:
        raise ValueError(ca.SOURCE_NOT_DONE)
    source_path = os.path.join(fds._dataset_dir(row.dataset_id), row.filename)
    if not os.path.isfile(source_path):
        raise ValueError(ca.SOURCE_FILE_GONE)

    wanted = ca.normalize_requested(poses)      # raises on empty / unknown / too many

    # Refuse BEFORE creating any row when the weights are absent, so a missing
    # model cannot leave a dataset full of failed tiles — the lesson the Klein
    # lane already paid for (preflight in generate_variations).
    missing = qch.camera_missing_assets()
    if not qch.camera_ready(missing):
        raise qch.CameraModelsMissing(missing)

    views = []
    for azimuth, elevation, distance in wanted:
        pose = ca.pose_id(azimuth, elevation, distance)
        prompt = ca.pose_prompt(azimuth, elevation, distance)
        candidate = LoraTestImage(
            dataset_id=row.dataset_id,
            # NOT NULL, and honest: this IS that checkpoint's render, re-shot.
            checkpoint=row.checkpoint, strength=row.strength,
            status='pending', filename=None,
            # Same checkpoint and step → same gallery, next to its source.
            # `run_id` stays NULL so a camera view never splices itself into the
            # epoch-by-epoch timeline, which filters on it.
            record_id=row.record_id, step=row.step, run_id=None,
            seed=row.seed,
            # The prompt that ACTUALLY ran, in the LoRA's own grammar. The
            # lightbox renders it as "Made with", so anything else here would be
            # a sentence that did not decide the picture.
            prompt=prompt[:500],
            parent_image_id=row.id,
            derivation_kind=CAMERA_ANGLE,
        )
        db.session.add(candidate)
        db.session.commit()                    # row BEFORE enqueue: no orphan job
        candidate_id = candidate.id
        try:
            job_id = qch.enqueue_camera_view(
                user_id=str(user_id), source_filename=row.filename,
                source_path=source_path, pose_prompt=prompt,
                # `is_lora_test` is what routes the finished job back to
                # link_completed_test_image — without it the completion looks the
                # result up as a dataset image that does not exist.
                extra_metadata={
                    'is_lora_test': True,
                    'dataset_id': str(row.dataset_id),
                    'cell_id': candidate_id,
                    'derivation_kind': CAMERA_ANGLE,
                    'parent_image_id': row.id,
                    'action': 'camera_angle',
                    'camera_pose': pose,
                })
        except Exception:
            # No ghost row: a candidate left pending with no file is exactly what
            # _active_run_count counts, so a failed enqueue that kept its row
            # would read as "a test run is already in progress" forever.
            stale = db.session.get(LoraTestImage, candidate_id)
            if stale is not None:
                db.session.delete(stale)
                db.session.commit()
            # Views already queued keep theirs — they are real work in flight,
            # and dropping them would waste GPU already spent.
            if views:
                logger.warning('camera angles: %s queued before %s failed',
                               len(views), pose)
                break
            raise
        saved = db.session.get(LoraTestImage, candidate_id)
        if saved is None:                      # cancelled mid-enqueue
            continue
        saved.job_id = job_id
        saved.camera_pose = pose
        db.session.commit()
        views.append({'candidate_id': candidate_id, 'job_id': job_id,
                      'pose': pose,
                      'label': ca.pose_label(azimuth, elevation, distance)})
    return {'views': views, 'queued': len(views)}


# --- Rating + best settings ---------------------------------------------------
def _owned_test_image(user_id, image_id):
    """Single-user app: no cross-user ownership check (SRC compared the
    image's dataset.user_id against `user_id`) - just the row lookup."""
    return db.session.get(LoraTestImage, image_id)


def image_render_status(user_id, image_id):
    """One library image's render state — the heartbeat the ✨ modal polls.

    Deliberately tiny: {status, url, error} is everything a "is my improve
    done yet" question needs, and nothing a 4-second poll should pay more
    for. None when the row is not the caller's."""
    row = db.session.get(LoraTestImage, image_id)
    if row is None:
        return None
    ds = fds.get_dataset(user_id, row.dataset_id)
    if not ds:
        return None
    return {
        'id': row.id, 'status': row.status,
        'url': (f'/api/dataset/{row.dataset_id}/img/{row.filename}'
                if row.filename else None),
        'error': row.error if row.status == 'failed' else None,
    }


def rate_image(user_id, image_id, rating) -> bool:
    if rating not in (1, -1, 0):
        return False
    img = _owned_test_image(user_id, image_id)
    if not img:
        return False
    img.rating = rating
    db.session.commit()
    return True


def _model_label(z_model):
    return _basename(z_model).rsplit('.', 1)[0] if z_model else None


# Below this vote count, flag a statistically fragile sample in the UI.
# Wilson sorting already penalizes small samples; the flag simply makes
# that uncertainty visible.
LOW_CONFIDENCE_MIN = 3


def cell_scores(dataset_id, family=None) -> list[dict]:
    """Score each configuration (checkpoint, strength, format, model, cfg, steps)
    across all its images and runs. Model belongs to the key, so results from
    different models do not merge into one cell.

    Optional `family` restricts cells to the pipeline inferred from checkpoint
    folders, keeping ZIT/SDXL/Krea results separate for a multi-family dataset.
    Legacy checkpoints without a folder prefix count as 'zimage'.

    Expose net likes-minus-dislikes as `score`, but sort by `rank`: the Wilson
    lower bound of the like rate combines rate and confidence, avoiding bias
    toward configurations with more tests. Best-first order is rank descending,
    vote count descending (confidence), strength ascending (anti-overfitting)."""
    rows = _cells().filter_by(dataset_id=dataset_id).all()
    # Failed cells produced no image and can't be judged — exclude them so a broken
    # config doesn't inflate the 'images' denominator or otherwise pollute the
    # ranking / best-config pick (P0-b).
    rows = [r for r in rows if r.status != 'failed']
    if family:
        rows = _filter_rows_by_family(rows, family)
    agg = {}
    for r in rows:
        key = (r.checkpoint, r.strength, r.aspect, r.z_model, r.cfg, r.steps, r.steps2)
        e = agg.setdefault(key, {'checkpoint': r.checkpoint, 'strength': r.strength,
                                 'aspect': r.aspect, 'z_model': r.z_model,
                                 'z_model_label': _model_label(r.z_model),
                                 'cfg': r.cfg, 'steps': r.steps, 'steps2': r.steps2,
                                 'score': 0, 'likes': 0, 'dislikes': 0,
                                 'images': 0, 'voted': 0, 'rank': 0.0})
        e['images'] += 1
        if r.rating == 1:
            e['likes'] += 1
            e['voted'] += 1
        elif r.rating == -1:
            e['dislikes'] += 1
            e['voted'] += 1
    for e in agg.values():
        e['score'] = e['likes'] - e['dislikes']
        e['rank'] = round(_wilson_lower_bound(e['likes'], e['voted']), 4)
        # Approval rate (likes/voted): None without votes to avoid a misleading 0/0.
        e['like_rate'] = round(e['likes'] / e['voted'], 4) if e['voted'] else None
        # Flag a vote sample too small for confidence.
        e['low_confidence'] = e['voted'] < LOW_CONFIDENCE_MIN
    return sorted(agg.values(),
                  key=lambda e: (-e['rank'], -e['voted'], e['strength']))


def model_net_scores(dataset_id) -> dict:
    """Net sentiment per model (likes minus dislikes across its images), exposed
    for display. best_cell uses the rate instead (see _model_like_rates)."""
    rows = _cells().filter_by(dataset_id=dataset_id).all()
    net = {}
    for r in rows:
        if r.rating == 1:
            net[r.z_model] = net.get(r.z_model, 0) + 1
        elif r.rating == -1:
            net[r.z_model] = net.get(r.z_model, 0) - 1
    return net


def _model_like_rates(scores) -> dict:
    """Like rate per model (likes/voted), aggregated across its configurations,
    for evaluating overall model ratings. Return {model: rate|None}; None means
    no votes."""
    acc = {}
    for e in scores:
        likes, voted = acc.get(e['z_model'], (0, 0))
        acc[e['z_model']] = (likes + e['likes'], voted + e['voted'])
    return {m: (likes / voted if voted else None) for m, (likes, voted) in acc.items()}


def model_comparison(dataset_id, scores=None) -> list[dict]:
    """Aggregate votes per base model (z_model) for fair comparisons.
    Rank by rate (Wilson lower bound), not raw totals, which favor models tested
    more often. Each entry exposes images/voted and low_confidence to make the
    sample size visible.

    Accept shared `scores` (see best_cell) to avoid rescanning the table."""
    scores = cell_scores(dataset_id) if scores is None else scores
    acc = {}
    for e in scores:
        a = acc.setdefault(e['z_model'], {
            'z_model': e['z_model'], 'z_model_label': e['z_model_label'],
            'likes': 0, 'dislikes': 0, 'images': 0, 'voted': 0, 'checkpoints': set()})
        a['likes'] += e['likes']
        a['dislikes'] += e['dislikes']
        a['images'] += e['images']
        a['voted'] += e['voted']
        a['checkpoints'].add(e['checkpoint'])
    out = []
    for a in acc.values():
        out.append({
            'z_model': a['z_model'], 'z_model_label': a['z_model_label'],
            'likes': a['likes'], 'dislikes': a['dislikes'],
            'net': a['likes'] - a['dislikes'],
            'images': a['images'], 'voted': a['voted'],
            'like_rate': round(a['likes'] / a['voted'], 4) if a['voted'] else None,
            'wilson': round(_wilson_lower_bound(a['likes'], a['voted']), 4),
            'low_confidence': a['voted'] < LOW_CONFIDENCE_MIN,
            'n_checkpoints': len(a['checkpoints']),
        })
    out.sort(key=lambda m: (-m['wilson'], -m['voted']))
    return out


def checkpoint_model_breakdown(dataset_id, scores=None) -> list[dict]:
    """For each (checkpoint, z_model), return generated/voted counts and like rate.
    These denominators expose thin samples when one LoRA has been tested more
    on one base than another. Sort by checkpoint label, then descending rate.

    Accept shared `scores` (see best_cell)."""
    scores = cell_scores(dataset_id) if scores is None else scores
    known = _known_checkpoints(db.session.get(FaceDataset, dataset_id))
    acc = {}
    for e in scores:
        key = (e['checkpoint'], e['z_model'])
        a = acc.setdefault(key, {
            'checkpoint': e['checkpoint'],
            'label': _checkpoint_display_label(e['checkpoint'], known),
            'z_model': e['z_model'], 'z_model_label': e['z_model_label'],
            'likes': 0, 'dislikes': 0, 'images': 0, 'voted': 0})
        a['likes'] += e['likes']
        a['dislikes'] += e['dislikes']
        a['images'] += e['images']
        a['voted'] += e['voted']
    out = []
    for a in acc.values():
        a['net'] = a['likes'] - a['dislikes']
        a['like_rate'] = round(a['likes'] / a['voted'], 4) if a['voted'] else None
        a['low_confidence'] = a['voted'] < LOW_CONFIDENCE_MIN
        out.append(a)
    out.sort(key=lambda a: (a['label'], -(a['like_rate'] or 0), -a['voted']))
    return out


def best_cell(dataset_id, scores=None) -> dict | None:
    """Recommend a configuration from votes.

    1. Candidates have positive net sentiment (likes > dislikes).
    2. Rank by descending Wilson `rank` (rate x confidence).
    3. Break ties by descending vote count, descending global model like rate,
       then ascending strength.

    Model sentiment is a tiebreaker, not a filter: a clearly better configuration
    is never excluded merely because its model performs moderately elsewhere.
    Return None until a configuration is liked.

    Accept precomputed `scores` to avoid rescanning the table. studio_payload
    shares one cell_scores result across best_cell/best_preset/best_per_checkpoint."""
    scores = cell_scores(dataset_id) if scores is None else scores
    candidates = [e for e in scores if e['likes'] > e['dislikes']]
    if not candidates:
        return None
    rates = _model_like_rates(scores)

    def model_pref(m):
        r = rates.get(m)
        return r if r is not None else 0.5  # a model without votes is neutral
    candidates.sort(key=lambda e: (-e['rank'], -e['voted'],
                                   -model_pref(e['z_model']), e['strength']))
    return candidates[0]


def best_preset(dataset_id, scores=None) -> dict | None:
    """The recommended configuration (best_cell, including model), enriched with
    a representative image's prompt/seed/filename from that exact configuration."""
    bc = best_cell(dataset_id, scores=scores)
    if not bc:
        return None
    img = (_cells()
           .filter_by(dataset_id=dataset_id, checkpoint=bc['checkpoint'],
                      strength=bc['strength'], aspect=bc.get('aspect'),
                      z_model=bc.get('z_model'), cfg=bc.get('cfg'),
                      steps=bc.get('steps'), steps2=bc.get('steps2'), status='done')
           .order_by(LoraTestImage.id.desc()).first())
    return {
        **bc,
        'label': _checkpoint_display_label(bc['checkpoint'],
                                           _known_checkpoints(
                                               db.session.get(FaceDataset, dataset_id))),
        'prompt': getattr(img, 'prompt', None) if img else None,
        'seed': img.seed if img else None,
        'filename': img.filename if img else None,
    }


def best_per_checkpoint(dataset_id, scores=None) -> list[dict]:
    """Return the best setting per checkpoint, since votes vary across models.
    For each checkpoint with at least one net-positive configuration, select
    its best configuration using best_cell's Wilson ordering and attach a
    representative image. Sort by descending rank.

    Accept shared `scores` (see best_cell) to avoid rescanning the table."""
    scores = cell_scores(dataset_id) if scores is None else scores
    candidates = [e for e in scores if e['likes'] > e['dislikes']]
    if not candidates:
        return []
    rates = _model_like_rates(scores)

    def model_pref(m):
        r = rates.get(m)
        return r if r is not None else 0.5
    candidates.sort(key=lambda e: (-e['rank'], -e['voted'],
                                   -model_pref(e['z_model']), e['strength']))
    best_by_cp = {}
    for e in candidates:  # already sorted: the first entry per checkpoint is best
        best_by_cp.setdefault(e['checkpoint'], e)
    out = []
    for bc in best_by_cp.values():
        img = (_cells()
               .filter_by(dataset_id=dataset_id, checkpoint=bc['checkpoint'],
                          strength=bc['strength'], aspect=bc.get('aspect'),
                          z_model=bc.get('z_model'), cfg=bc.get('cfg'),
                          steps=bc.get('steps'), steps2=bc.get('steps2'), status='done')
               .order_by(LoraTestImage.id.desc()).first())
        out.append({**bc,
                    'label': _checkpoint_display_label(bc['checkpoint'],
                                                       _known_checkpoints(
                                                           db.session.get(FaceDataset, dataset_id))),
                    'prompt': getattr(img, 'prompt', None) if img else None,
                    'seed': img.seed if img else None,
                    'filename': img.filename if img else None})
    out.sort(key=lambda e: -e['rank'])
    return out


def _best_map(ds) -> dict:
    """Read best_settings as a {family: settings} map. For backward compatibility,
    attach a legacy flat object (identified by top-level `lora_filename`) to the
    dataset's train_type. Return {} for empty or unreadable values."""
    if not ds.best_settings:
        return {}
    try:
        data = json.loads(ds.best_settings)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if 'lora_filename' in data:  # ancien format plat (mono-famille)
        return {(getattr(ds, 'train_type', None) or 'zimage').lower(): data}
    return data


def _best_for_family(ds, family) -> dict | None:
    """Return the saved setting for this family, or None."""
    return _best_map(ds).get((family or 'zimage').lower())


def best_settings_lora_filenames(ds) -> list[str]:
    """Every LoRA filename this dataset pins as a ★ best setting — a LIST, one
    entry per family, because the pin is stored per family (a dataset can have a
    winning ZIT combo and a winning SDXL one at the same time).

    This is what the "you are about to delete the pinned LoRA" guard-rail needs.
    Readers used to reach for `best_settings.lora_filename` straight off the
    payload, which only ever matched the LEGACY flat format: since the pin became
    a {family: setting} map that key does not exist any more, so the ⚠ line
    silently stopped appearing for every modern pin. Going through _best_map
    covers both shapes at once. Order is deterministic (family order as stored),
    duplicates collapsed."""
    out: list[str] = []
    for setting in _best_map(ds).values():
        if not isinstance(setting, dict):
            continue
        fn = setting.get('lora_filename')
        if fn and fn not in out:
            out.append(str(fn))
        # Pinning a stack pins every member. Deleting the second LoRA breaks the
        # winning setting just as deleting the head would, so deletion guards must
        # check all members.
        for member in setting.get('stack') or []:
            mfn = member.get('lora_filename') if isinstance(member, dict) else None
            if mfn and mfn not in out:
                out.append(str(mfn))
    return out


def set_best_settings(user_id, dataset_id, checkpoint, strength,
                      z_model=None, cfg=None, steps=None, steps2=None, aspect=None,
                      stack=None) -> dict:
    """Persist the full winning checkpoint, strength, model, cfg, steps(1+2), and
    format configuration. Save by family, inferred from the checkpoint folder,
    so one dataset can retain distinct ZIT, SDXL and Krea settings. The checkpoint
    must belong to its family's allowlist; validate a supplied model against
    bases of the matching type (a fixed Krea base ignores the model). Return
    the saved setting."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    family = (family_of_lora(checkpoint) or getattr(ds, 'train_type', None) or 'zimage').lower()
    allowed = {c['filename'] for c in list_test_checkpoints(ds, family)}
    if checkpoint not in allowed:
        raise ValueError('unknown checkpoint for this dataset')
    try:
        strength = round(float(strength), 2)
    except (TypeError, ValueError):
        raise ValueError(f'invalid strength: {strength!r}')
    if not 0.05 <= strength <= 4.0:
        raise ValueError(f'strength out of range: {strength}')
    allowed_bases = set(family_base_models(family))
    z_model = z_model or None  # '' (Krea Official entry) means the default: NULL
    if z_model and z_model not in allowed_bases:
        raise ValueError('selected base model is unavailable for this family')
    try:
        cfg = round(float(cfg), 2) if cfg is not None else None
    except (TypeError, ValueError):
        cfg = None
    try:
        steps = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        steps = None
    try:
        steps2 = int(steps2) if steps2 is not None else None
    except (TypeError, ValueError):
        steps2 = None
    aspect = aspect if aspect in TEST_ASPECTS else None
    # A stack's winning setting includes all its weights. Keep the head in
    # lora_filename/strength so existing Canvas pins, Apply actions, deletion
    # guards and workspace badges keep working. Add other members in `stack`.
    # Revalidate each member against its own dataset's deployed checkpoints:
    # request data must never become a source of arbitrary paths.
    stack_out = []
    for member in (stack or []):
        if not isinstance(member, dict):
            raise ValueError('invalid stack member')
        member_ds = fds.get_dataset(user_id, member.get('dataset_id'))
        if not member_ds:
            raise ValueError('unknown dataset in stack')
        member_fn = member.get('lora_filename') or member.get('filename')
        member_family = (family_of_lora(member_fn)
                         or getattr(member_ds, 'train_type', None) or 'zimage').lower()
        if member_fn not in {c['filename'] for c in list_test_checkpoints(member_ds, member_family)}:
            raise ValueError('unknown checkpoint in stack')
        stack_out.append({'lora_filename': member_fn, 'dataset_id': member_ds.id,
                          'weight': _combine_weight({'weight': member.get('weight')}),
                          'trigger': getattr(member_ds, 'trigger_word', None) or None})
    best = {
        'lora_filename': checkpoint,
        'strength': strength,
        'z_model': z_model,
        'cfg': cfg,
        'steps': steps,
        'steps2': steps2,
        'aspect': aspect,
        'family': family,
        'decided_at': naive_utcnow().isoformat(),
        # Omit the key rather than writing [] for a single LoRA, preserving the
        # same shape for older and current single-LoRA settings.
        **({'stack': stack_out} if stack_out else {}),
    }
    best_map = _best_map(ds)
    best_map[family] = best
    ds.best_settings = json.dumps(best_map)
    db.session.commit()
    return best


def clear_best_settings(user_id, dataset_id, family=None) -> bool:
    """Clear saved settings. With `family`, clear only that family; otherwise
    clear all. Idempotent when no setting exists."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if family:
        m = _best_map(ds)
        m.pop((family or '').lower(), None)
        ds.best_settings = json.dumps(m) if m else None
    else:
        ds.best_settings = None
    db.session.commit()
    return True


# --- Objective facial scoring (automatic best epoch) -----------------------
def score_faces(user_id, dataset_id, family=None) -> dict:
    """Score each completed family cell against the dataset reference using
    InsightFace (antelopev2 in a CPU subprocess, without touching the GPU).
    Persist face_score/face_state per cell, then return checkpoint rankings.

    Automate objective epoch selection: render checkpoints with a fixed seed
    (as Studio already does), then choose the best measured face score instead
    of the latest epoch. Idempotent: rescoring overwrites existing scores."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Third InsightFace lane, same single rule (fds.face_scoring_block_reason).
    # Returned in the shape the panel already renders (scoring_error) so the button
    # explains itself instead of scoring 0 cells in green.
    blocked = fds.face_scoring_block_reason(ds)
    if blocked:
        return {'scored': 0, 'total': 0, 'ranking': [],
                'scoring_error': {'kind': 'subject_not_photographic', 'detail': blocked}}
    if not ds.ref_filename:
        raise ValueError('reference photo missing')
    ref_path = fds._ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference photo missing')
    eff = _resolve_family(ds, family, available_families(ds))
    rows = (_cells().filter_by(dataset_id=dataset_id, status='done')
            .filter(LoraTestImage.filename.isnot(None)).all())
    rows = _filter_rows_by_family(rows, eff)
    ds_dir = fds._dataset_dir(dataset_id)
    by_path = {}
    for r in rows:
        p = os.path.join(ds_dir, r.filename)
        if os.path.exists(p):
            by_path[p] = r
    if not by_path:
        return {'scored': 0, 'total': 0, 'scoring_error': None, 'ranking': []}
    from .face_similarity import score_dataset_faces
    # Expose scoring_error ({kind, detail} | None) in the toast. A broken
    # scorer must explain why instead of showing a successful "done - 0/14".
    results, scoring_error = score_dataset_faces(ref_path, list(by_path.keys()))
    scored = 0
    for p, r in by_path.items():
        res = results.get(p)
        if not res:
            continue
        r.face_state = res.get('state')
        r.face_score = res.get('sim')
        scored += 1
    db.session.commit()
    logger.info(f"lora-test: score-faces dataset {dataset_id} ({eff}) -> "
                f"{scored}/{len(by_path)} cell(s) scored")
    return {'scored': scored, 'total': len(by_path), 'scoring_error': scoring_error,
            'ranking': face_ranking(dataset_id, eff)}


def face_ranking(dataset_id, family) -> list:
    """Rank checkpoints by mean facial similarity for previously scored cells in
    the given family. Return [{checkpoint, label, avg, n}], best first; the
    frontend marks the first as the best epoch."""
    rows = (_cells().filter_by(dataset_id=dataset_id)
            .filter(LoraTestImage.face_score.isnot(None)).all())
    rows = _filter_rows_by_family(rows, family)
    agg = {}
    for r in rows:
        a = agg.setdefault(r.checkpoint, [0.0, 0])
        a[0] += float(r.face_score)
        a[1] += 1
    known = _known_checkpoints(db.session.get(FaceDataset, dataset_id), family)
    out = [{'checkpoint': cp,
            'label': _checkpoint_display_label(cp, known),
            'avg': round(s / n, 4), 'n': n}
           for cp, (s, n) in agg.items()]
    out.sort(key=lambda e: (-e['avg'], -e['n']))
    return out


def delete_prompt(user_id, dataset_id, prompt) -> int:
    """Delete one prompt’s Test Studio cells only after safe queue cancellation.

    A pending cell can be cancelled locally. A sent/running prompt must first be
    proven absent by ``queue_manager.cancel_job``; otherwise its row and files
    remain intact. This avoids turning an ambiguous ComfyUI request into an
    orphaned GPU job while the user is trying to delete the prompt.
    """
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    p = (prompt or '').strip()
    if not p:
        return 0
    dataset_dir = fds._dataset_path(dataset_id)

    with GPU_ARBITER_LOCK:
        rows = _cells().filter_by(dataset_id=dataset_id, prompt=p).all()
        if not rows:
            return 0
        # Do every safe cancellation before touching a file. A refusal retains
        # the prompt identity so the normal ComfyUI recovery flow can reconcile
        # it later; earlier safe cancellations are harmless and visible.
        for row in rows:
            if row.job_id and row.status not in ('done', 'failed', 'cancelled'):
                if not queue_manager.cancel_job(row.job_id, str(user_id), 'image'):
                    raise ValueError(
                        'ComfyUI work for this prompt is still running or needs recovery; '
                        'recover ComfyUI, cancel the paused cell, then try again.')

        moved = []
        seen_paths = set()
        try:
            for row in rows:
                if row.filename:
                    fp = os.path.join(dataset_dir, row.filename)
                    path_key = os.path.normcase(os.path.abspath(fp))
                    if path_key not in seen_paths and os.path.exists(fp):
                        destination = trash.send_to_trash(
                            fp, context=f'dataset-{dataset_id}-studio-prompt')
                        moved.append((destination, fp))
                        seen_paths.add(path_key)
                db.session.delete(row)
            db.session.commit()
        except Exception:
            db.session.rollback()
            for destination, original in reversed(moved):
                fds._restore_from_trash(destination, original)
            raise
    n = len(rows)
    logger.info(f"lora-test: prompt deleted from dataset {dataset_id} -> {n} cell(s)")
    return n

# --- Payload (poll) ------------------------------------------------------------
def studio_payload(user_id, dataset_id, family=None) -> dict | None:
    """Everything the Studio panel needs in one poll, scoped to one family.

    Resolve the user's selected ZIT/SDXL/Krea `family` to an available family
    for this dataset. Scope checkpoints, grid, scores, best settings and bases
    to it, keeping results separate when a dataset trains with multiple
    pipelines. `available_families` lists selector choices; `family` returns
    the resolved family."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return None
    fams = available_families(ds)
    eff = _resolve_family(ds, family, fams)
    rows_all = (_cells().filter_by(dataset_id=dataset_id)
                .order_by(LoraTestImage.id.asc()).all())
    # Grid cells use the resolved family, inferred from checkpoints; guests
    # without folders inherit the run family.
    rows = _filter_rows_by_family(rows_all, eff)
    activity = _queue_activity(rows_all)
    best = _best_for_family(ds, eff)
    known = _known_checkpoints(ds, eff)
    # Base pool for the resolved family: SDXL checkpoints as {value,label},
    # fixed Krea workflow UNET without a selector, otherwise Z-Image models.
    # train_type is the resolved family used by frontend pickers and handoff.
    base_note = None
    # Family CFG/steps fallback. Krea adjusts to the actual elected base:
    # Turbo settings (cfg 1, eight steps) on a non-distilled default produce
    # blurry sketches that users could mistake for failed training.
    defaults = studio_family_defaults(eff)
    default_cfg, default_steps = defaults['cfg'], defaults['steps']
    if eff == 'sdxl':
        z_models = [{'value': m['filename'], 'label': m['label']}
                    for m in list_sdxl_base_models()]
    elif eff == 'krea':
        # Local Krea alternatives exclude the elected default. The leading empty
        # value still means default, with a label naming a non-Setup file when used.
        # No local alternatives means an empty list and hidden selector, as before.
        # Still expose base_note: that installation particularly needs the warning.
        _krea_entry = krea_default_base_entry()
        base_note = _krea_entry['note']
        if _krea_entry['source']:
            _d = krea_model_defaults(_krea_entry['source'])
            default_cfg, default_steps = _d['cfg'], _d['steps']
        _alts = krea_alt_base_models()
        z_models = ([{'value': '', 'label': _krea_entry['label']}]
                    + [{'value': m, 'label': _basename(m).rsplit('.', 1)[0]} for m in _alts]
                    if _alts else [])
    else:
        z_models = [{'value': m or '', 'label': (_basename(m).rsplit('.', 1)[0]
                                                if m else 'Default model')}
                    for m in family_base_models(eff)]
    generation_state = {}
    if eff in TRAINED_IMAGE_FAMILIES:
        from .trained_image_models import generation_readiness
        readiness = generation_readiness(eff)
        generation_state = {'generation_readiness': readiness,
                            'default_model': (readiness.get('assets') or {}).get('diffusion_model', '')}
        if readiness.get('config_error'):
            base_note = readiness['config_error']
    return {
        **generation_state,
        'checkpoints': list_test_checkpoints(ds, eff),
        'trigger_word': ds.trigger_word,
        'train_type': eff,
        'family': eff,
        'generation_capabilities': generation_capabilities(eff),
        # Trained families for the dataset selector: [{family,label,count}].
        'available_families': fams,
        # Always-on style/utility LoRAs available for this family, outside the batch axis.
        'permanent_loras': permanent_lora_candidates(eff),
        'prompt': identity_prompt(ds),
        'z_models': z_models,
        # Explain an unusual default base, such as a missing Setup Krea file or
        # an elected file without weights. Otherwise None hides the frontend note.
        'base_note': base_note,
        'aspects': list(TEST_ASPECTS.keys()),
        'default_aspect': DEFAULT_ASPECT,
        'cfg_choices': [1.0] if eff == 'flux2klein' else CFG_CHOICES,
        'default_cfg': default_cfg,
        'steps_choices': STEPS_CHOICES, 'default_steps': default_steps,
        # Per-BASE-MODEL cfg/steps, keyed by the same `value` as `z_models` (bobba84,
        # GitHub #18): Z-Image Base is not guidance-distilled and must not inherit
        # Turbo's cfg 1 / 8 steps. `default_cfg`/`default_steps` stay the fallback for
        # every base not listed here, so an older frontend behaves exactly as before.
        'model_defaults': studio_model_defaults(eff, z_models),
        # Expose the second detail-daemon pass only for SDXL's two-pass HQ workflow.
        # Otherwise NULL hides the frontend's second steps picker.
        'steps2_choices': (STEPS_CHOICES if eff == 'sdxl' else None),
        'default_steps2': (DEFAULT_STEPS if eff == 'sdxl' else None),
        'max_images': MAX_TEST_IMAGES,
        # Observed median throughput for this machine and pipeline, or null when
        # history is too short. This replaces a universal "~12 s/image" estimate.
        'seconds_per_image': measured_seconds_per_image(eff),
        # Extend the shared cloud_training.gallery_image serializer (see
        # stack_variants), retaining Gallery facts plus grid-specific keys. run_id
        # was always stored by create_run but previously omitted here, making the
        # grid infer runs from run_seed+prompt and display N prompts as N runs.
        'cells': [{**_shared_cell(r),
                   'label': _checkpoint_display_label(r.checkpoint, known),
                   'filename': r.filename, 'run_seed': r.run_seed,
                   'status': r.status,
                   'queue_status': activity['queue_status'].get(r.job_id),
                   'queue_error': activity['queue_error'].get(r.job_id),
                   'z_model': r.z_model,
                   'z_model_label': (_basename(r.z_model).rsplit('.', 1)[0] if r.z_model else None),
                   'steps2': r.steps2,
                   'batch_lora': _batch_lora_label(r),
                   'combined_loras': _combined_lora_labels(r),
                   # Why the tile is empty (failed cells only) → shown on hover (P0-b).
                   'error': r.error if r.status == 'failed' else None,
                   'face_state': r.face_state}
                  for r in rows],
        # Share one family-filtered cell_scores table scan across best_cell,
        # best_preset and best_per_checkpoint instead of four identical scans.
        'scores': (_scores := cell_scores(dataset_id, family=eff)),
        'best_cell': best_cell(dataset_id, scores=_scores),
        'best_preset': best_preset(dataset_id, scores=_scores),
        'best_per_model': best_per_checkpoint(dataset_id, scores=_scores),
        # Fair base comparisons by z_model, plus checkpoint/base breakdowns.
        'model_comparison': model_comparison(dataset_id, scores=_scores),
        'checkpoint_breakdown': checkpoint_model_breakdown(dataset_id, scores=_scores),
        # Objective facial checkpoint ranking from scored cells (best epoch).
        'face_ranking': face_ranking(dataset_id, eff),
        'pending': activity['pending'],
        'queued': activity['queued'],
        'generating': activity['generating'],
        'running': activity['running'],
        # Resumable stopped/failed cells across the dataset.
        'resumable': sum(1 for r in rows_all if r.status in ('cancelled', 'failed')),
        # Distinct recent prompts for reloading/rerunning: global to the user
        # across all datasets and families.
        'recent_prompts': user_recent_prompts(ds.user_id),
        'gpu_busy': gpu_busy_reason(),
        'comfyui_recovery': _unknown_submit_recovery(rows, activity),
        'comfyui_recovery_target': _comfyui_recovery_target(),
        'best_settings': best,
    }


def lora_net_scores(run_id) -> list[dict]:
    """Rank LoRAs within a run by aggregating cell votes per dataset_id (one LoRA).
    Sort by descending net score (likes minus dislikes), then descending likes."""
    rows = _cells().filter_by(run_id=run_id).filter(
        LoraTestImage.filename.isnot(None)).all()
    agg = {}
    for r in rows:
        a = agg.setdefault(r.dataset_id, {'dataset_id': r.dataset_id, 'likes': 0,
                                          'dislikes': 0, 'voted': 0, 'total': 0,
                                          'lora_label': format_trained_lora_label(r.checkpoint)
                                          or _basename(r.checkpoint).rsplit('.', 1)[0]})
        a['total'] += 1
        if r.rating == 1: a['likes'] += 1; a['voted'] += 1
        elif r.rating == -1: a['dislikes'] += 1; a['voted'] += 1
    for a in agg.values():
        a['net'] = a['likes'] - a['dislikes']
        a['wilson'] = _wilson_lower_bound(a['likes'], a['voted'])
        ds = db.session.get(FaceDataset, a['dataset_id'])
        a['dataset_name'] = ds.name if ds else f"#{a['dataset_id']}"
    return sorted(agg.values(), key=lambda a: (a['net'], a['likes']), reverse=True)


def studio_payload_run(user_id, run_id) -> dict | None:
    """Payload for one single- or multi-LoRA run, queried by run_id and enriched
    with per-LoRA rankings and the list of participating LoRAs."""
    rows = (_cells().filter_by(run_id=run_id)
            .order_by(LoraTestImage.id.asc()).all())
    if not rows:
        return None
    ds_ids = {r.dataset_id for r in rows}
    owned = {d.id for d in FaceDataset.query.filter(FaceDataset.user_id == str(user_id),
             FaceDataset.id.in_(ds_ids)).all()}
    if ds_ids - owned:
        return None
    activity = _queue_activity(rows)
    def _lbl(d):
        return next((_basename(r.checkpoint).rsplit('.', 1)[0] for r in rows if r.dataset_id == d), str(d))
    def _name(d):
        ds = db.session.get(FaceDataset, d); return ds.name if ds else str(d)
    return {
        'run_id': run_id,
        'loras': [{'dataset_id': d, 'lora_label': _lbl(d), 'dataset_name': _name(d)}
                  for d in sorted(ds_ids)],
        # Extend the shared cloud_training.gallery_image serializer, using the
        # same contract as studio_payload above.
        'cells': [{**_shared_cell(r),
                   'label': _basename(r.checkpoint).rsplit('.', 1)[0],
                   'filename': r.filename, 'run_seed': r.run_seed, 'status': r.status,
                   'queue_status': activity['queue_status'].get(r.job_id),
                   'queue_error': activity['queue_error'].get(r.job_id),
                   'z_model': r.z_model, 'steps2': r.steps2,
                   'batch_lora': _batch_lora_label(r),
                   'combined_loras': _combined_lora_labels(r),
                   'error': r.error if r.status == 'failed' else None} for r in rows],
        'lora_ranking': lora_net_scores(run_id),
        # Stack run composition (each LoRA, weight and trigger), plus other runs
        # of the same stack. Per-LoRA rankings would mislead because a stack has
        # only one tested head; the frontend shows composition instead.
        # `stack` is None for a comparison run.
        'stack': (_stack := stack_of_row(rows[0])),
        'stack_variants': stack_variants(run_id, rows) if _stack else [],
        'pending': activity['pending'],
        'queued': activity['queued'],
        'generating': activity['generating'],
        'running': activity['running'],
        'resumable': sum(1 for r in rows if r.status in ('cancelled', 'failed')),
        'gpu_busy': gpu_busy_reason(),
        'comfyui_recovery': _unknown_submit_recovery(rows, activity),
        'comfyui_recovery_target': _comfyui_recovery_target(),
    }


def _recent_prompts(rows, limit=None) -> list[dict]:
    """Distinct used prompts, newest first, with thumbnails and image counts.
    Prefer a liked image generated with the prompt, otherwise the latest
    completed image. `thumb_dataset_id` identifies the thumbnail's dataset when
    rows span several datasets. `limit=None` returns all distinct prompts in
    `rows`; the arbitrary limit of ten was removed at the user's request.
    Only user_recent_prompts' scan of the latest 1500 cells remains bounded.
    Return [{prompt, thumbnail(filename|None), thumb_dataset_id, thumb_rating, count}]."""
    seen = {}  # prompt -> dict, inserted newest to oldest
    for r in sorted(rows, key=lambda x: -x.id):  # newest first
        p = (r.prompt or '').strip()
        if not p:
            continue
        if p not in seen:
            if limit is not None and len(seen) >= limit:
                continue
            seen[p] = {'prompt': p, 'thumbnail': None, 'thumb_dataset_id': None,
                       'thumb_rating': 0, 'count': 0}
        e = seen[p]
        if r.filename:
            e['count'] += 1
            if r.rating == 1 and e['thumb_rating'] != 1:      # prefer the most recent liked image
                e['thumbnail'], e['thumb_rating'] = r.filename, 1
                e['thumb_dataset_id'] = r.dataset_id
            elif e['thumbnail'] is None:                       # otherwise use the first completed image seen (the most recent)
                e['thumbnail'], e['thumb_rating'] = r.filename, (r.rating or 0)
                e['thumb_dataset_id'] = r.dataset_id
    return list(seen.values())


def user_recent_prompts(user_id, limit=None) -> list[dict]:
    """The user's recent test prompts across all datasets, as requested on
    2026-07-03: prompt/preset history is shared across datasets.
    `limit=None` returns all distinct prompts; the limit of ten was removed
    at the user's request. Scan at most the latest 1500 cells for performance.
    Each entry includes `thumb_dataset_id` so the frontend builds the correct
    dataset thumbnail URL."""
    ds_ids = [d.id for d in FaceDataset.query.filter_by(user_id=str(user_id)).all()]
    if not ds_ids:
        return []
    rows = (_cells().filter(LoraTestImage.dataset_id.in_(ds_ids))
            .order_by(LoraTestImage.id.desc()).limit(1500).all())
    return _recent_prompts(rows, limit=limit)


def delete_prompt_everywhere(user_id, prompt) -> int:
    """Delete a recent prompt and its test cells/images across all the user's
    datasets, matching deletion from the global history list."""
    p = (prompt or '').strip()
    if not p:
        return 0
    n = 0
    for d in FaceDataset.query.filter_by(user_id=str(user_id)).all():
        try:
            n += delete_prompt(user_id, d.id, p)
        except ValueError:
            continue
    return n


def enhance_with_instructions(instructions: str, model: str | None = None) -> str:
    """Run bounded caller-owned writing instructions through the normal local writer."""
    if not isinstance(instructions, str) or not instructions.strip() or len(instructions) > 30000:
        raise ValueError('Writing instructions must contain at most 30000 characters.')
    from .vision_keepalive import keep_alive_for_isolated_call
    from .vision_llm import ensure_ready, generate_text as generate_text_ollama, label
    ready = ensure_ready(model)
    if not ready.get('ok'):
        # The remedy is not the same word for the two providers: an Ollama model is
        # PULLED, an LM Studio one is LOADED in its app. Saying "load" to an Ollama
        # user was a regression this wave introduced; saying "pull" to an LM Studio
        # user names an action their server does not have.
        if model:
            fix = (' — pick another model from the ✨ Enhance ⚙️ options, or load this '
                   'one in LM Studio first.' if label() == 'LM Studio'
                   else ' — pick another model from the ✨ Enhance ⚙️ options, or pull '
                        'this one first.')
        else:
            fix = (f' — Enhance needs the local {label()} model configured in '
                   'Settings › Local tools.')
        raise RuntimeError((ready.get('error') or f'{label()} is unavailable') + fix)
    text = generate_text_ollama(instructions, model=model,
                                num_predict=500,
                                keep_alive=keep_alive_for_isolated_call(), strict=True)
    text = (text or '').strip().strip('"').strip()
    if not text:
        raise RuntimeError(
            f'The model returned an empty prompt — check the configured {label()} model '
            'in Settings and the application log.')
    return text
