"""Turning a built video dataset into an ai-toolkit job config — and reading back
what a two-expert model writes when it saves.

This is the video sibling of `lora_training._build_job_config_*`, kept in its own
module because it shares none of their inputs. Those builders read a `FaceDataset`
and its `train_settings` column; there is no such column on `video_dataset`, and
routing a video run through helpers that all `getattr(ds, ..., None)` would mean a
config assembled entirely from defaults that nobody could see. The entities are
different, so the builder is separate.

Like `video_targets`, this module is pure: a catalogue lookup, some arithmetic and
string work. No ffmpeg, no PyAV, no torch, no database. A cloud launcher can build
a config from a snapshot on a machine that has no trainer installed at all.

WHERE THE VALUES COME FROM
--------------------------
An end-to-end smoke run has been through this path: a dataset built by the video
Bank, handed to the local ai-toolkit, trained to decreasing loss (0.28 -> 0.23) at
81 frames and 384 px. The non-obvious settings below are that run's, and the two
shipped ai-toolkit example configs it derives from. Where no such source exists —
a base repository for LTX or MiniMax — this module RAISES rather than invents one.
A guessed repo id does not fail at build time; it fails after the GPU is rented.

WHAT IT COSTS, MEASURED
-----------------------
That run filled 24 GB and took 170-185 s per step. `low_vram` is why: it keeps one
expert on the CPU and shuttles it over PCIe every boundary switch. It is the
default here because a 24 GB card cannot do without it, and it is a PARAMETER
because the whole reason to move a video run to a rented 80 GB pod is to turn it
off.
"""
import math
import os
import re

from . import video_targets


class VideoTrainingUnsupported(ValueError):
    """This video dataset cannot be turned into a training config, and the reason
    is stated. Raised at BUILD time, on purpose: every one of these conditions
    would otherwise surface as a trainer crash or — worse — a silent misread, on a
    pod that is already being paid for."""


# Bases stated by an ai-toolkit example config shipped on this machine, keyed by
# the arch. Deliberately short. The two Wan entries are the only video bases any
# local source names; every other arch demands an explicit `base_model` from the
# caller. `wan21` covers both the 1.3B and the 14B in one catalogue profile, so
# the flagship is the default and a caller wanting the 1.3B names it.
_VERIFIED_BASES = {
    'wan21': 'Wan-AI/Wan2.1-T2V-14B-Diffusers',
    'wan22_14b': 'ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16',
    # NOT `MiniMaxAI/MiniMax-H3`. The trainer resolves every weight file from the
    # Comfy-Org repack — `COMFY_REPO` in its own minimax_h3.py — and touches the
    # original repository only for the tokenizer's tiny config files. This is the
    # one field where naming the "obvious" upstream repo costs a 43 GB download
    # before anything says a word.
    'minimax_h3': 'Comfy-Org/MiniMax-H3',
}

# What one architecture's weights are ALREADY stored as. `qfloat8` is the value
# every image family here carries and it is the wrong one for a pre-quantised
# checkpoint: ai-toolkit does not refuse a mismatched qtype, it re-quantises the
# tensors layer by layer into that format on every load. For MiniMax H3 that is a
# 21 GB int8-ConvRot transformer and a 16 GB nvfp4 text encoder converted into
# something they were not saved as, for nothing. Both strings are real backends
# (`CONVROT_QTYPES` / `NVFP4_QTYPES` in ai-toolkit's ostris quantizer registry).
_DEFAULT_QTYPE_TE = 'qfloat8'
_QTYPES = {
    'minimax_h3': {'qtype': 'convrot8', 'qtype_te': 'nvfp4'},
}

# The noise schedule each architecture was distilled on, from ai-toolkit's own job
# presets. Three arches, three answers — `linear` is Wan's, and it is only the
# default here because that is the one this app has a finished run behind. Getting
# this wrong does not crash; it trains against the wrong schedule and the loss
# curve looks fine.
_DEFAULT_TIMESTEP_TYPE = 'linear'
_TIMESTEP_TYPES = {
    'minimax_h3': 'shift',
    'ltx2': 'weighted',
    'ltx2.3': 'weighted',
}

# Arches whose ai-toolkit preset caches encoded clips to disk. It matters most
# where the transformer is largest: H3 leaves roughly two gigabytes of a 24 GB
# card free, and encoding on the fly keeps the video VAE resident beside it for
# the whole run. Wan's proven run did not cache, so Wan is left as it trained.
_CACHE_LATENTS_ARCHES = ('minimax_h3', 'ltx2.3')

# Preview sampler settings, keyed by arch. H3 is GUIDANCE-DISTILLED: its released
# pipeline applies no CFG at all, so Wan's 3.5 is not a harmless default here —
# above 1 it degrades the very preview a caller would judge the LoRA by.
_DEFAULT_SAMPLE_GUIDANCE = (3.5, 4)
_SAMPLE_GUIDANCE = {
    'minimax_h3': (1, 28),
}

# The multiplier ai-toolkit's presets give the audio branch of a joint model.
# Which arches HAVE one is not decided here — `video_targets.wants_audio` owns
# that, from the same source the exporter reads when it decides to mux.
_AUDIO_LOSS_MULTIPLIER = 1.0

# Accuracy-recovery adapters, keyed by arch. The 4-bit path is only usable WITH
# one: `quantize: true` + `qtype: uint4` alone is a measurably worse run, so the
# adapter travels with the qtype string rather than being a separate opt-in.
_RECOVERY_ADAPTERS = {
    'wan22_14b': ('uint4|ostris/accuracy_recovery_adapters/'
                  'wan22_14b_t2i_torchao_uint4.safetensors'),
}

# The de-distillation recipe, per arch, and it is ai-toolkit's OWN default rather
# than a setting we chose. H3 ships guidance-distilled; a LoRA trained on it raw
# runs into what ostris named "distillation breakdown", and upstream's answer came
# in two halves that are meant to travel together: a contrastive guidance loss
# (2026-08-04) and a frozen "training adapter" LoRA loaded beside the model
# (2026-08-06, made the default training method the same day). The commit that
# turned both on at once says it plainly — they yield "better results than one or
# the other".
#
# The adapter is never merged: `minimax_h3.load_training_adapter` keeps it live
# and frozen because the transformer is pre-quantized and a merge would resample
# every int8 scale. The sampler switches it off for previews.
#
# GATED, and this is the whole reason the caller passes a flag instead of us
# reading a config: `assistant_lora_path` reaches a per-arch loader that only
# exists from 2026-08-06. On anything older the generic path
# (toolkit/assistant_lora.py) is Flux-only and raises "Only Flux models can load
# assistant adapters currently" — a crash, not a downgrade. The local lane probes
# the installed toolkit; the cloud lane knows its pinned image.
_TRAINING_ADAPTERS = {
    'minimax_h3': {
        'assistant_lora_path': ('ostris/minimax_h3_training_adapter/'
                                'minimax_h3_training_adapter_v1.safetensors'),
        'guidance_loss_target': 3.5,
    },
}

# Module names the LoRA must NOT wrap, per arch. `ignore_if_contains` is a plain
# substring skip over module names (toolkit/lora_special.py), old and generic, so
# it costs nothing on an ai-toolkit that predates the reason for it.
#
# The reason, for H3, is upstream's own: ostris removed `adaln_proj` from H3's
# network modules on 2026-08-04 and the shipped preset has carried the exclusion
# ever since. Size is not the argument but it is what makes it visible - in run
# #166's checkpoint that single module holds a [96768, 16] B matrix per block,
# the largest tensor in the file, spent on a modulation projection rather than on
# anything the LoRA is being asked to learn. That run predates this line and its
# checkpoint carries the modules; the next one will not.
_IGNORED_MODULES = {
    'minimax_h3': ('adaln_proj',),
}

# The two-expert (MoE) arches. Two consequences, and they are unrelated to each
# other: training needs `switch_boundary_every` to alternate between the experts,
# and SAVING writes two files per checkpoint instead of one.
_MOE_ARCHES = ('wan22_14b', 'wan22_14b_i2v')

# From ai-toolkit's own 24 GB example and from the run that trained. Not a tuning
# knob we picked: a boundary this tight is what keeps both experts learning
# together instead of the run becoming two half-trainings in sequence.
_SWITCH_BOUNDARY_EVERY = 10

# The suffixes `wan22_14b_model.save_lora` appends when it splits a state dict on
# `.transformer_1.` / `.transformer_2.`. Order matters only for readability; the
# parser matches whichever is present.
_STAGE_SUFFIXES = ('high_noise', 'low_noise')

# ai-toolkit zero-pads the step to 9 digits. 6 is the floor every consumer in this
# app already used, and it is what keeps a trailing '_v3' or '_rc74' from reading
# as a step.
_STEP_RE = re.compile(r'_(\d{6,})$')


def is_multistage_arch(arch) -> bool:
    """Does this architecture save its LoRA as a high-noise / low-noise PAIR?

    True for the Wan 2.2 14B MoE arches. `split_multistage_loras` defaults to True
    in `toolkit/config_modules.py`, so this is the shape a caller gets unless it
    goes out of its way to ask for a combined file — and the combined file is not
    what any downstream loader here expects."""
    return str(arch or '') in _MOE_ARCHES


def split_checkpoint_name(filename):
    """``(step, stage)`` for one saved checkpoint filename.

    `step` is None for the run's FINAL save, which carries no number in either
    world — that is how a caller flags it final. `stage` is `'high_noise'`,
    `'low_noise'`, or None for a single-file arch.

    THIS EXISTS BECAUSE OF THE ANCHOR. Every step read in this app was
    ``_(\\d{6,})\\.safetensors$``, and `save_lora` builds its pair by rewriting
    `.safetensors` into `_high_noise.safetensors`. The step is therefore no longer
    at the end of the stem, the regex misses, and each consumer's `or target`
    fallback labels EVERY intermediate save with the run's total step count — six
    identical pills, and a "continue from step 50" that resumes from step 100.
    Nothing raises; the numbers are just wrong.

    The stage is only recognised at the very end of the stem, so a dataset called
    "low noise study" does not turn all of its saves into low-noise halves."""
    stem = os.path.basename(str(filename or ''))
    if stem.lower().endswith('.safetensors'):
        stem = stem[:-len('.safetensors')]
    stage = None
    for suffix in _STAGE_SUFFIXES:
        if stem.endswith('_' + suffix):
            stage = suffix
            stem = stem[:-(len(suffix) + 1)]
            break
    m = _STEP_RE.search(stem)
    return (int(m.group(1)) if m else None), stage


def restage_checkpoint_name(base: str, step, stage) -> str:
    """Rebuild a checkpoint filename from a new stem plus the step and stage read
    off the original — the inverse of `split_checkpoint_name`.

    The mirror into the local run folder needs this. Rebuilding from the step
    alone gives BOTH halves of a pair the same name, and the second copy is then
    refused as a collision with the first — one expert of every checkpoint lost,
    with a log line that says the local file was protected."""
    parts = [base]
    if step is not None:
        parts.append(f'{int(step):09d}')
    if stage:
        parts.append(stage)
    return '_'.join(parts) + '.safetensors'


def _resolution_for(width, height, size_multiple, max_pixels=None):
    """The single scalar that asks ai-toolkit for exactly this clip's pixels.

    `resolution` is not a width. `toolkit/buckets.get_bucket_for_image_size` reads
    it as a pixel CAP — `max_pixels = resolution ** 2`, then
    `target = min(total_pixels, max_pixels)` — and rescales the clip's own aspect
    ratio under it. So the faithful scalar for a w x h clip is the geometric mean
    sqrt(w*h): the cap equals the clip's own pixel count, the scaler is exactly
    1.0, and nothing is resized in either direction.

    Floored to the target's size multiple, never rounded up: the multiple is the
    VAE's spatial stride, and rounding up would ask for more pixels than the clips
    contain and upscale data that does not exist.

    `max_pixels` is the target's own canvas cap, where it has one — MiniMax H3
    caps the AREA at 768*1344, which no step can express (1920x1088 is a clean
    multiple of 32 and still out of spec). Since the scalar IS a pixel cap once
    squared, it is floored under the model's too. Only a clamp DOWN ever happens
    here, which is a downscale of real data rather than an invention of pixels."""
    step = size_multiple or 1
    side = math.sqrt(width * height)
    if max_pixels:
        side = min(side, math.sqrt(max_pixels))
    return max(step, int(side // step) * step)


def build_job_config(video_ds, dataset_folder: str, steps: int,
                     training_folder=None, base_model=None, low_vram=True,
                     sample_prompts=None, save_every=None,
                     max_step_saves_to_keep=2, rank=16,
                     training_adapter=False) -> dict:
    """The ai-toolkit job config for training on a built video dataset.

    `training_adapter` says whether the ai-toolkit that will RUN this config can
    load a per-arch training adapter — a capability that arrived on 2026-08-06 and
    that this function cannot check for itself, because the toolkit it is building
    for is not always the one it is running on. The local lane probes the
    installed copy; the cloud lane knows the tag its pod boots. Left False the
    config is exactly what it was before the recipe existed, which is what an
    older toolkit needs: the alternative is not a slower run, it is a raised
    "Only Flux models can load assistant adapters currently".

    `video_ds` needs the `video_dataset` columns and nothing else: `name`,
    `target_profile`, `frames`, `fps`, `width`, `height`. It is duck-typed rather
    than a model import so a cloud launcher can pass a frozen snapshot of a row
    whose dataset may since have been rebuilt.

    `dataset_folder` is used VERBATIM. A video dataset is already the shape
    ai-toolkit wants — a flat folder of .mp4 files with homonym .txt captions — so
    unlike the image families there is no export to re-run; the caller passes the
    local folder, and `_cloudify_job_config` rewrites it to the pod path with the
    same string swap it does for every other family.

    `training_folder` is the same cloud seam as the image families: supplied, it is
    used as-is so a launch needs no local ai-toolkit; omitted, the caller is
    expected to have resolved it (this module has no run-root of its own).

    `low_vram` defaults True because 24 GB cannot do without it, and exists as a
    parameter because turning it off is the measured reason to rent a bigger GPU:
    the smoke run spent 170-185 s per step shuttling the idle expert over PCIe.

    `sample_prompts` defaults to none, and with none `disable_sampling` is True. A
    video preview is minutes of paid GPU per prompt and proves nothing about the
    dataset. Asked for, previews are rendered at the DATASET's own frame count and
    fps — a preview at another rate is not a preview of this LoRA.
    """
    key = getattr(video_ds, 'target_profile', None)
    profile = video_targets.get(key)
    if profile is None:
        raise VideoTrainingUnsupported(
            f'unknown video target profile {key!r} — this build has no rules for '
            'it, and Wan\'s would be a guess')
    arch = profile['aitk_arch']
    if not arch:
        raise VideoTrainingUnsupported(
            f'the {key!r} video target declares no ai-toolkit architecture, so no '
            'training config can be written for it — pick a catalogued target')

    frames = getattr(video_ds, 'frames', None)
    if not frames or not video_targets.is_legal_frames(key, frames):
        raise VideoTrainingUnsupported(
            f'{frames} frames is not a length {profile["label"]} can ingest — its '
            'VAE would silently drop the trailing frames of every clip')

    width = getattr(video_ds, 'width', None)
    height = getattr(video_ds, 'height', None)
    if not width or not height:
        raise VideoTrainingUnsupported(
            'this video dataset recorded no clip size, so the training resolution '
            'cannot be derived — rebuild it, or pass the size explicitly')

    name_or_path = base_model or _VERIFIED_BASES.get(arch)
    if not name_or_path:
        raise VideoTrainingUnsupported(
            f'no verified base model is known for {profile["label"]} — name the '
            'base repository to train it (nothing installed here states one, and '
            'a guessed repository id only fails once the pod is paid for)')

    qtypes = _QTYPES.get(arch, {})
    model = {
        'arch': arch,
        'name_or_path': name_or_path,
        'quantize': True,
        'quantize_te': True,
        'qtype_te': qtypes.get('qtype_te', _DEFAULT_QTYPE_TE),
        'low_vram': bool(low_vram),
    }
    # An accuracy-recovery adapter, where one exists, wins: it is a 4-bit path
    # chosen deliberately, not a description of how the file on disk is stored.
    qtype = _RECOVERY_ADAPTERS.get(arch) or qtypes.get('qtype')
    if qtype:
        model['qtype'] = qtype
    if is_multistage_arch(arch):
        # Both experts, explicitly. Under low_vram ai-toolkit unloads whichever
        # one is idle; training only one of them yields half a LoRA that no
        # downstream loader here knows how to complete.
        model['model_kwargs'] = {'train_high_noise': True, 'train_low_noise': True}

    train = {
        'batch_size': 1,
        'steps': int(steps),
        'gradient_accumulation': 1,
        'train_unet': True,
        'train_text_encoder': False,
        'gradient_checkpointing': True,
        'noise_scheduler': 'flowmatch',
        'timestep_type': _TIMESTEP_TYPES.get(arch, _DEFAULT_TIMESTEP_TYPE),
        'optimizer': 'adamw8bit',
        'lr': 1e-4,
        'optimizer_params': {'weight_decay': 1e-4},
        'dtype': 'bf16',
        'disable_sampling': not sample_prompts,
        # NOT `unload_text_encoder`. ai-toolkit offers exactly two routes to 24 GB
        # and they are not interchangeable: unloading encodes the TRIGGER WORD
        # ONLY and reuses that one embedding for every clip, discarding every
        # caption in the dataset without a word. Caching pre-encodes the real
        # captions. A captioned-clip lane can only ever use the second.
        'cache_text_embeddings': True,
    }
    if is_multistage_arch(arch):
        train['switch_boundary_every'] = _SWITCH_BOUNDARY_EVERY

    fps = getattr(video_ds, 'fps', None) or profile['fps'] or 16
    dataset_block = {
        'folder_path': dataset_folder,
        'caption_ext': 'txt',
        'caption_dropout_rate': 0.05,
        'num_frames': int(frames),
        # The rate the clips were CUT to, and not decoration. On a joint
        # audio/video model the waveform is stretched to
        # `num_frames / dataset fps`; with no fps stated it is fitted to the
        # SOURCE duration instead and drifts against the frames it accompanies.
        'fps': int(fps),
        # Explicitly off, even though it is ai-toolkit's default and even though
        # its own H3 preset turns it ON. This lane cuts clips to an exact legal
        # count and states it; letting the loader recount off the file would put a
        # length nobody validated back in charge — and it also forbids any batch
        # size above 1.
        'auto_frame_count': False,
        # ai-toolkit's default, stated rather than inherited, because what makes
        # it safe is a property of OUR clips and nothing in the config says so.
        # True means the loader takes `num_frames` frames spread evenly across
        # the whole file: on a clip cut to exactly that count it is the identity,
        # and on a clip one frame too long it silently speeds the motion up. The
        # cutter guarantees the exact count (see video_clip_export.clip_command,
        # which refuses a span too short and passes `-frames:v`), so this lane
        # meets the precondition ai-toolkit's own comment asks for: "trim your
        # videos to the desired length and use shrink_video_to_frames(default)".
        # Flipping it to False would NOT be the safer read - it switches the
        # loader to pulling frames at `fps` from a RANDOM start frame.
        'shrink_video_to_frames': True,
        'resolution': [_resolution_for(width, height, profile['size_multiple'],
                                       profile['max_pixels'])],
    }
    if video_targets.wants_audio(key):
        # Opt-in per dataset, and defaulting to False: omitted, the audio branch
        # of a joint model trains on nothing while the config still looks whole.
        dataset_block['do_audio'] = True
        train['audio_loss_multiplier'] = _AUDIO_LOSS_MULTIPLIER
    if arch in _CACHE_LATENTS_ARCHES:
        dataset_block['cache_latents_to_disk'] = True

    # Upstream's own de-distillation recipe, when the target has one and the
    # toolkit on the other end can run it. Both halves or neither: the commit
    # that made them the default turned them on together.
    recipe = _TRAINING_ADAPTERS.get(arch) if training_adapter else None
    if recipe:
        model['assistant_lora_path'] = recipe['assistant_lora_path']
        train['do_guidance_loss'] = True
        train['guidance_loss_target'] = recipe['guidance_loss_target']

    proc = {
        'type': 'sd_trainer',
        # 'output' is ai-toolkit's own default, relative to its root, and is what
        # the run that trained carried. Emitting the caller's None instead would
        # put a literal null in the config and fail inside the trainer.
        'training_folder': training_folder or 'output',
        'device': 'cuda:0',
        'network': _network_block(arch, rank),
        'save': {
            'dtype': 'float16',
            'save_every': int(save_every or max(1, int(steps) // 2)),
            'max_step_saves_to_keep': int(max_step_saves_to_keep),
        },
        'datasets': [dataset_block],
        'train': train,
        'model': model,
    }
    if sample_prompts:
        guidance, sample_steps = _SAMPLE_GUIDANCE.get(arch,
                                                      _DEFAULT_SAMPLE_GUIDANCE)
        proc['sample'] = {
            'sampler': 'flowmatch',
            'sample_every': proc['save']['save_every'],
            'width': int(width),
            'height': int(height),
            'num_frames': int(frames),
            'fps': int(fps),
            'prompts': list(sample_prompts),
            'neg': '',
            'seed': 42,
            'guidance_scale': guidance,
            'sample_steps': sample_steps,
        }
    return {
        'job': 'extension',
        'config': {
            'name': job_name_for(video_ds),
            'process': [proc],
        },
    }


# The day `load_training_adapter` landed upstream. Not decoration: it is the line
# between "the recipe is skipped" and "the run raises Only Flux models can load
# assistant adapters currently".
_TRAINING_ADAPTER_SINCE = '2026-08-06'

# vastai publishes its ai-toolkit images as `<sha>-YYYY-MM-DD-cuda-x.y`, and that
# date is the ONLY signal about a pod's toolkit that exists before the rental.
_IMAGE_DATE_RE = re.compile(r'-(\d{4}-\d{2}-\d{2})-')


def image_supports_training_adapter(image_tag) -> bool:
    """Is this pinned pod image new enough to load a training adapter?

    A pod cannot be probed before it is rented, so the tag is what there is. Ours
    is pinned well past the cutoff, but the pin is CONFIGURABLE — someone who
    moves it back to chase a different trainer must not also, silently, arm a
    recipe their image cannot run.

    A tag with no readable date answers False. A skipped recipe costs quality on
    one run; a wrong answer costs the run itself, after the money is spent."""
    found = _IMAGE_DATE_RE.search(str(image_tag or ''))
    return bool(found) and found.group(1) >= _TRAINING_ADAPTER_SINCE


def suggested_steps(clip_count) -> int:
    """A step count sized to the DATASET, for the UI to prefill - never a value
    this module applies on its own.

    The face lane has had this shape for months (steps per image, clamped, with
    the rationale shown beside the field); the video lane launched with a fixed
    number instead, and the only public measurements there are say that is the
    wrong shape: on the same recipe, 1,500 steps won at 53 clips and the SAME
    recipe needed 5,000 at 176 (fal, blind-judged H3 realism runs). Those two
    wins sit on one line - 28.3 and 28.4 steps per clip - so the slope here is
    28, rounded UP to the hundred, which lands exactly on both measured points.

    Clamped, and each bound has a reason rather than a shrug: the floor is 1000,
    this lane's historic fixed default, so no dataset is ever suggested LESS
    than what every run before this function got; the cap is 5000, the largest
    run any measurement supports - suggesting past the evidence would be a
    guess dressed as a recommendation, and long cloud runs also walk into the
    max-runtime safety cap.

    Calibrated on H3 and offered lane-wide: the principle (steps scale with the
    data) is model-agnostic, the constant is not, which is why the UI presents
    it as a suggestion beside an editable field and never as this target's
    number."""
    try:
        count = max(0, int(clip_count or 0))
    except (TypeError, ValueError):
        count = 0
    return min(5000, max(1000, (28 * count + 99) // 100 * 100))


def training_adapter_for(arch):
    """The de-distillation recipe this arch has, or None. Public so the capability
    probe can ask "is there anything to gate?" before touching the filesystem."""
    return _TRAINING_ADAPTERS.get(str(arch or ''))


def _network_block(arch, rank) -> dict:
    """The LoRA network, plus the modules this architecture leaves alone.

    `network_kwargs` is omitted entirely when an arch has nothing to exclude,
    rather than emitted empty: an empty dict is a claim that the question was
    considered and answered "none", and only the catalogue above gets to make
    that claim."""
    block = {'type': 'lora', 'linear': int(rank), 'linear_alpha': int(rank)}
    ignored = _IGNORED_MODULES.get(arch)
    if ignored:
        block['network_kwargs'] = {'ignore_if_contains': list(ignored)}
    return block


def job_name_for(video_ds) -> str:
    """A filesystem- and route-safe job name for this dataset. It becomes the save
    root on the pod and the stem of every checkpoint file, so it may only contain
    what ai-toolkit's own upload route preserves ([A-Za-z0-9._-])."""
    raw = (getattr(video_ds, 'name', None) or '').strip()
    safe = ''.join(c if (c.isalnum() or c in '_-') else '_' for c in raw).strip('_')
    return f'video_{safe}' if safe else f'video_dataset{getattr(video_ds, "id", 0)}'


# Kept as the name the image families use, so a reader grepping for the family of
# builders finds this one too.
_build_job_config_video = build_job_config
