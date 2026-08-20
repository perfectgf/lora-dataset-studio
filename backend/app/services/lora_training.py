"""Automatisation de l'entraînement LoRA Z-Image via ai-toolkit.

L'app prépare (export dataset + job-config) et lance l'UI ai-toolkit ; elle ne
réimplémente pas l'entraîneur. Pause GPU via le flag system_state
`training_in_progress` honoré par le superviseur ComfyUI.

Lifted from the parent project's app/services/lora_training.py (1288 lines)
for LoRA Dataset Studio: SRC's module-level AITOOLKIT_DIR/HF_HOME/DATASETS_DIR/
OUTPUT_DIR/LORA_DEST_DIR* constants become live `cfg.aitoolkit_path(...)` /
`cfg.comfyui_dir(...)` accessors below, each raising a clean RuntimeError when
its backend isn't configured yet (so config.json edits apply without a
restart, and routes can map the RuntimeError to a 409). `UI_URL` (ai-toolkit's
web UI, unused - this app drives the CLI) and the whole ownership subsystem
(`record_lora_ownership`, the ownership-filtered checkpoint listing) are
dropped - single local user, cf. plan's Global Constraints.
"""
from __future__ import annotations
import filecmp
import functools
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import stat
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from .. import config as cfg
from ..models import FaceDataset, FaceDatasetImage
from ..job_queue import GPU_ARBITER_LOCK, queue_manager
from . import cloud_run_dataset as _crd
from . import dataset_activity, face_dataset_service as fds, face_mask, trash
from .person_mask import generate_person_masks

logger = logging.getLogger(__name__)

# Résolution + VRAM Krea 2 (modèle 12B). MESURÉ 2026-06-26 : à 1024 SANS unload TE la VRAM
# sature (24,0/24,5 Go) → ~180 s/it (ETA ~7 j, inexploitable) ; à 768 → 3,5 s/it (~50× plus
# rapide → goulot = ACTIVATIONS, pas le streaming des poids). Stratégie qualité : on GARDE 1024
# mais on libère le Qwen3-VL via cache_text_embeddings + unload_text_encoder (~4-8 Go) pour
# tenir sans offload. Si 1024 sature encore → baisser ce SEUL curseur à 896 (mesurer), puis 768
# (cadence prouvée). Curseur de tuning #1, un seul endroit.
KREA_TRAIN_RESOLUTION = 1024

# Dense checkpoints are roughly 26 GB.  These values are intentionally NOT
# inherited from hidden LoRA advanced settings: one recoverable checkpoint at a
# predictable cadence bounds disk use while preserving restartability.
FULL_TRANSFORMER_SAVE_EVERY = 250
FULL_TRANSFORMER_SAMPLE_EVERY = 250
# How many dense saves live side by side. Named (rather than the literal it used
# to be) because the Hugging Face storage pre-check multiplies by it: the
# forecast and the job config must never disagree about how many ~26 GB objects
# a run produces.
FULL_TRANSFORMER_MAX_STEP_SAVES = 1
# The two OFFICIAL Krea 2 repositories, used by the LoRA lane and the dense lane
# alike. FULL_TRANSFORMER_BASE keeps its name because it is also the dense
# DEFAULT (an unset `train_variant` means Raw, cf. _krea_is_raw) — but it is no
# longer the only base a dense run can carry: see _krea_name_or_path.
FULL_TRANSFORMER_BASE = 'krea/Krea-2-Raw'
KREA_TURBO_BASE = 'krea/Krea-2-Turbo'
FULL_TRANSFORMER_VAE = 'Qwen/Qwen-Image-2512'

# --- the dense knobs that are NOT locked --------------------------------------
# The dense recipe stays server-owned because its geometry is what makes a 12B
# transformer fit in 80 GB: optimizer (adafactor), batch 1, bf16 and gradient
# checkpointing are load-bearing, and changing any one of them turns a working
# run into an out-of-memory crash an hour in. The ones below are different —
# they change what the run PRODUCES, not whether it fits:
#
#   * preview prompts  — the four generic defaults show nothing about the actual
#     dataset, which makes the mid-run sample sheet useless for deciding whether
#     to stop. Reuses the LoRA `sample_prompts` key on purpose: same dataset,
#     same subject, one place to edit (and no new stored id to alias later).
#   * learning rate    — bounded around the validated 1e-6.
#   * resolution       — 1024 (default) or 768; 768 is the measured fallback
#     when a run has to be cheaper, not a quality choice.
#   * checkpoint every / keep — the storage lever, and the one the Hugging Face
#     pre-check multiplies by. Nothing here may drift from what the job config
#     emits; both read the resolvers below.
FULL_TRANSFORMER_LR = 1e-6
FULL_TRANSFORMER_LR_MIN = 1e-7
FULL_TRANSFORMER_LR_MAX = 5e-6
FULL_TRANSFORMER_RESOLUTION_CHOICES = (768, 1024)
FULL_TRANSFORMER_SAVE_EVERY_MIN = 100
FULL_TRANSFORMER_SAVE_EVERY_MAX = 5000
FULL_TRANSFORMER_MAX_STEP_SAVES_MAX = 3

# --- WHICH ai-toolkit these verdicts describe ---------------------------------
# Dense training and LoRA training do NOT run the same ai-toolkit, and reading
# the wrong one is the easiest way to ship a setting that lies.
#
#   * LoRA (local lane) runs the ai-toolkit checkout on the USER'S machine. That
#     one moves: it is whatever they last pulled. Nothing here describes it.
#   * DENSE (this recipe) is cloud-only and runs the ai-toolkit baked into the
#     vast.ai pod. The machine the user's dataset is actually trained on is a
#     remote one nobody here can `git log`.
#
# So the line references below are against the ai-toolkit commit the dense pod
# image carries, NOT against any local checkout:
FULL_TRANSFORMER_AITOOLKIT_COMMIT = '4625406'      # ai-toolkit, dated 2026-07-12
# `config.py` pins that same commit in `cloud.image`; the test suite fails if the
# two ever drift, because a new image is a new trainer and every verdict below
# has to be re-checked against it. Two honest caveats that a pin cannot fix:
#   - the DEFAULT launch path is `cloud.template_hash` (vast.ai's own "Ostris AI
#     Toolkit" template), and its contents are published by a third party — it
#     can move without this repo changing. `cloud.image` is only the raw-image
#     fallback. Nothing in a run record stamps which trainer actually ran, which
#     is a real gap, flagged rather than silently patched here.
#   - to bound that risk, every verdict below was ALSO re-checked against a
#     three-weeks-newer ai-toolkit (2026-08-03). All six came out identical, so
#     none of them hinges on one snapshot — but "identical at both ends of a
#     three-week window" is evidence, not a guarantee.
#
# --- the three quality levers, and why the other two are NOT here -------------
# Every value below was traced through the code path a `krea2` run actually
# takes — `extensions_built_in/sd_trainer/SDTrainer.py` and
# `jobs/process/BaseSDTrainProcess.py` — not merely found in ai-toolkit's
# `toolkit/config_modules.py`. That distinction is the whole point: a key can
# exist in the config dataclass, be parsed without complaint, and never be read
# on this architecture. A setting that quietly does nothing is worse than no
# setting, because it makes a run look tuned when it is not.
#
#   * gradient accumulation — READ. BaseSDTrainProcess pulls
#     `train.gradient_accumulation` batches into one list per step (L2380) and
#     SDTrainer runs a backward per batch before a single optimiser step
#     (L2122-2155). With batch_size pinned at 1, this is the ONLY way to make
#     the effective batch bigger than one image, and it costs no VRAM: the
#     forwards are sequential, the gradient buffer already exists (batches wait
#     on the CPU, a few hundred MB of host RAM at 8). It costs TIME — N
#     accumulations means an N× longer run and an N× bigger pod bill, which is
#     why `dense_time_multiplier` travels with the setting.
#     NB the optimiser is adafactor, whose update is RMS-normalised
#     (toolkit/optimizers/adafactor.py L343-345), so accumulating summed rather
#     than averaged gradients does not inflate the step size the way it would
#     under plain SGD. The LR bounds above stay valid.
#   * LR schedule / warmup — READ. `train.lr_scheduler` (+ `lr_scheduler_params`)
#     builds a torch scheduler (BaseSDTrainProcess L2054-2065) stepped every
#     iteration (SDTrainer L2169). It reaches adafactor because ai-toolkit forces
#     `relative_step=False` / `scale_parameter=False` (toolkit/optimizer.py
#     L88-96), so adafactor reads `param_group['lr']` straight back
#     (adafactor.py L193-203) — the value the scheduler rewrites.
#     Warmup is wired for `constant_with_warmup` ONLY: the torch schedulers
#     behind `cosine`/`constant` do not accept `num_warmup_steps` and would
#     raise (toolkit/scheduler.py L6-40). Same rule as the LoRA lane.
#   * timestep type — READ. flowmatch training re-draws the timestep schedule
#     from `train.timestep_type` (BaseSDTrainProcess L1178-1207 →
#     toolkit/samplers/custom_flowmatch_sampler.py L107-219), and `weighted`
#     additionally re-weights the loss (SDTrainer L836-853). Krea 2 is a
#     flow-matching model (krea2.py L181) so all of that applies.
#     `shift`/`flux_shift` are deliberately NOT offered even though they parse:
#     the trainer derives the shift from an image-token count that assumes
#     `unet.config.patch_size` (BaseSDTrainProcess L1195-1200), and Krea 2's
#     denoiser config names that field `patch` (krea2/src/mmdit.py L92-106), so
#     patch_size silently falls back to 1 and the token count comes out 4× the
#     value Krea 2's own scheduler was calibrated against (krea2.py L86-89).
#     A mis-shifted noise schedule is exactly the "looks tuned, is not" trap.
#
# Refused, with the reason, so nobody re-litigates it from the key list alone:
#   * EMA — supported, and fatal here. ExponentialMovingAverage clones every
#     trained parameter on the training device (toolkit/ema.py L60-63), and each
#     save/sample clones them a SECOND time (store() before copy_to(), ema.py
#     L171-176, BaseSDTrainProcess L491-497). In dense mode the parameter set is
#     the whole 12B transformer (no `network`, so the optimiser holds the
#     transformer itself — BaseSDTrainProcess L777-789 reads exactly those param
#     groups), so that is roughly +26 GB resident and +26 GB more at every
#     checkpoint, on top of an unquantised bf16 model and its gradients. It would
#     not survive its first save. (The LoRA lane offers EMA because there it
#     averages a few hundred MB of adapter.)
#   * min_snr_gamma — supported for epsilon models, crashes on this one, and
#     crashes LATE. SDTrainer L920-922 calls `apply_snr_weight`, which needs
#     `all_snr` or `alphas_cumprod` off the noise scheduler
#     (toolkit/train_tools.py L642-654). Krea 2 trains on
#     CustomFlowMatchEulerDiscreteScheduler (krea2.py L223-224, L450), which
#     inherits diffusers' FlowMatchEulerDiscreteScheduler and defines neither.
#     ai-toolkit does TRY to attach the table up front (SDTrainer L276 →
#     train_tools L623-639) but that helper swallows its own exception, so the
#     failure does not surface at startup: it surfaces as an uncaught
#     AttributeError on the first loss, an hour into a paid pod.
FULL_TRANSFORMER_GRAD_ACCUM = 1
# 8 is the ceiling because the ceiling is a bill, not a memory limit: at 8 a
# 3000-step dense run takes eight times as long on a rented 80 GB GPU. Somebody
# who needs more smoothing than that should be training on fewer, better images.
FULL_TRANSFORMER_GRAD_ACCUM_CHOICES = (1, 2, 4, 8)
FULL_TRANSFORMER_TIMESTEP_TYPE = 'linear'
FULL_TRANSFORMER_TIMESTEP_TYPE_CHOICES = ('linear', 'sigmoid', 'weighted')
FULL_TRANSFORMER_LR_SCHEDULE = 'constant'
FULL_TRANSFORMER_LR_SCHEDULE_CHOICES = ('constant', 'constant_with_warmup', 'cosine')
FULL_TRANSFORMER_WARMUP = 100
FULL_TRANSFORMER_WARMUP_MIN = 10
FULL_TRANSFORMER_WARMUP_MAX = 1000
# train_settings keys. NEW names (never reused from the LoRA lane), so a dataset
# that switches modes cannot inherit a value that meant something else.
DENSE_SETTING_KEYS = ('dense_lr', 'dense_resolution', 'dense_save_every',
                      'dense_max_step_saves', 'dense_fp8_export',
                      'dense_keep_bf16', 'dense_grad_accum',
                      'dense_lr_schedule', 'dense_warmup',
                      'dense_timestep_type')


def _dense_bool(ds, key, default=True) -> bool:
    v = _train_settings(ds).get(key)
    return v if isinstance(v, bool) else default


def dense_fp8_export_enabled(ds) -> bool:
    """Produce the ~10 GB ComfyUI-loadable fp8 twin at the end of a dense run.

    ON by default: nobody generates with a 26 GB bf16 transformer, and doing the
    conversion on the pod (where the file already is) costs minutes, while doing
    it at home costs a 26 GB download first."""
    return _dense_bool(ds, 'dense_fp8_export', True)


def dense_keep_bf16_master(ds) -> bool:
    """Keep the bf16 master NEXT TO the fp8 export instead of replacing it.

    ON by default, and that default is not a preference: fp8 is a lossy,
    one-way export. A dense run that only delivered fp8 can never be continued,
    re-trained, merged or re-quantized differently — the master is the only
    thing that makes those possible."""
    return _dense_bool(ds, 'dense_keep_bf16', True)


def _dense_lr(ds) -> float:
    v = _train_settings(ds).get('dense_lr')
    if isinstance(v, (int, float)) and not isinstance(v, bool) \
            and FULL_TRANSFORMER_LR_MIN <= float(v) <= FULL_TRANSFORMER_LR_MAX:
        return float(v)
    return FULL_TRANSFORMER_LR


def _dense_resolution(ds) -> int:
    v = _train_settings(ds).get('dense_resolution')
    return v if v in FULL_TRANSFORMER_RESOLUTION_CHOICES else KREA_TRAIN_RESOLUTION


def _dense_save_every(ds) -> int:
    v = _train_settings(ds).get('dense_save_every')
    if isinstance(v, int) and not isinstance(v, bool) \
            and FULL_TRANSFORMER_SAVE_EVERY_MIN <= v <= FULL_TRANSFORMER_SAVE_EVERY_MAX:
        return v
    return FULL_TRANSFORMER_SAVE_EVERY


def _dense_max_step_saves(ds) -> int:
    v = _train_settings(ds).get('dense_max_step_saves')
    if isinstance(v, int) and not isinstance(v, bool) \
            and 1 <= v <= FULL_TRANSFORMER_MAX_STEP_SAVES_MAX:
        return v
    return FULL_TRANSFORMER_MAX_STEP_SAVES


def _dense_grad_accum(ds) -> int:
    v = _train_settings(ds).get('dense_grad_accum')
    if isinstance(v, int) and not isinstance(v, bool) \
            and v in FULL_TRANSFORMER_GRAD_ACCUM_CHOICES:
        return v
    return FULL_TRANSFORMER_GRAD_ACCUM


def _dense_timestep_type(ds) -> str:
    v = _train_settings(ds).get('dense_timestep_type')
    return v if v in FULL_TRANSFORMER_TIMESTEP_TYPE_CHOICES \
        else FULL_TRANSFORMER_TIMESTEP_TYPE


def _dense_lr_schedule(ds) -> str:
    v = _train_settings(ds).get('dense_lr_schedule')
    return v if v in FULL_TRANSFORMER_LR_SCHEDULE_CHOICES \
        else FULL_TRANSFORMER_LR_SCHEDULE


def _dense_warmup(ds) -> int:
    v = _train_settings(ds).get('dense_warmup')
    if isinstance(v, int) and not isinstance(v, bool) \
            and FULL_TRANSFORMER_WARMUP_MIN <= v <= FULL_TRANSFORMER_WARMUP_MAX:
        return v
    return FULL_TRANSFORMER_WARMUP


def _dense_lr_schedule_fields(ds) -> dict:
    """`{}` for the shipped `constant` schedule — and that emptiness is load-
    bearing. ai-toolkit's TrainConfig already defaults `lr_scheduler` to
    'constant' (config_modules.py L371), so omitting the key entirely keeps the
    emitted job config byte-for-byte what it was before this lever existed.
    `num_warmup_steps` travels ONLY with `constant_with_warmup`: the torch
    schedulers behind the other choices reject it (toolkit/scheduler.py)."""
    schedule = _dense_lr_schedule(ds)
    if schedule == FULL_TRANSFORMER_LR_SCHEDULE:
        return {}
    out = {'lr_scheduler': schedule}
    if schedule == 'constant_with_warmup':
        out['lr_scheduler_params'] = {'num_warmup_steps': _dense_warmup(ds)}
    return out


def dense_time_multiplier(ds) -> int:
    """How many times longer accumulation alone makes this run — which is also
    how many times bigger the pod bill gets, since the pod is billed by the hour.

    Accumulation-only on purpose: it is the one lever whose cost is exactly a
    multiplication (N sequential forward/backward passes per optimiser step,
    same steps, same checkpoints). Resolution also moves the clock, but not by a
    factor anyone can state without measuring, so it is not folded in here — a
    made-up second factor would poison the one number that IS exact.

    Exported (not private) because it has to be sayable BEFORE a machine is
    rented; the recipe card is the honest place to learn it, not the invoice."""
    return _dense_grad_accum(ds)


def dense_images_per_step(ds) -> int:
    """The effective batch: `batch_size` (pinned at 1) × gradient accumulation.
    This is what the lever is FOR — on a 6 000-image dataset, a gradient
    computed from one image is a very noisy estimate of the right direction."""
    return _dense_grad_accum(ds)


# --- how to TEST what a dense run delivers -------------------------------------
# The dense artifact is Krea 2 RAW: undistilled, and it needs a real CFG and a
# real step count. The family's inference defaults are Turbo defaults (a handful
# of steps at CFG 1) — applied to a Raw checkpoint they produce a blurry sketch
# and read as "the training failed". These are the values the run's OWN preview
# sheet was rendered with (sample.guidance_scale / sample.sample_steps in the
# emitted job config), which is why they are defined once, here, and referenced
# by the recipe, the panel, the run card and the test lane rather than retyped.
FULL_TRANSFORMER_SAMPLE_GUIDANCE = 4
FULL_TRANSFORMER_SAMPLE_STEPS = 25
FULL_TRANSFORMER_SAMPLE_GUIDANCE_RANGE = (3.5, 5.0)
FULL_TRANSFORMER_SAMPLE_STEPS_RANGE = (20, 30)


def dense_inference_hint(ds=None) -> dict:
    """The settings to generate with from a dense artifact, in one shape.

    Consumed by the training panel, the run card and the test/generation lane so
    a single change of the recipe moves every surface at once.

    The NUMBERS never change with the base — undistilled guidance and 20-30
    steps are what a dense Krea 2 artifact wants either way, and that is also
    the published replacement recipe for a distilled model that has been fully
    fine-tuned. The SENTENCE does: calling a Turbo-based run "a RAW
    (undistilled) model" would state as fact the very thing nobody has
    measured. ``ds`` is optional because two callers hold a delivered artifact
    rather than a dataset; without one the wording stays the Raw default, which
    is the default recipe."""
    lo_cfg, hi_cfg = FULL_TRANSFORMER_SAMPLE_GUIDANCE_RANGE
    lo_steps, hi_steps = FULL_TRANSFORMER_SAMPLE_STEPS_RANGE
    settings = (f'CFG ~{FULL_TRANSFORMER_SAMPLE_GUIDANCE} '
                f'({lo_cfg:g}-{hi_cfg:g}) and {lo_steps}-{hi_steps} steps')
    base = 'krea2_raw'
    note = (f'This is a RAW (undistilled) Krea 2 model. Test it at {settings} '
            '— Turbo-style few-step settings will look blurry.')
    if ds is not None and _train_type(ds) == 'krea':
        if _is_custom_weights(getattr(ds, 'train_base_model', None)):
            base = 'custom'
            note = (f'Test this full model at {settings}. If the checkpoint you '
                    'fine-tuned was a few-step build, check for yourself whether '
                    'it still works at low step counts — training moves the '
                    'weights that behaviour lives in.')
        elif not _krea_is_raw(ds):
            base = 'krea2_turbo'
            note = (f'Trained from Krea 2 Turbo. Test it at {settings}, the way '
                    'an undistilled model is tested: full fine-tuning moves the '
                    'weights the few-step behaviour lives in. How much of that '
                    'behaviour is left is exactly what has not been measured — '
                    'try your usual Turbo settings too and compare.')
    return {
        'base': base,
        'guidance_scale': FULL_TRANSFORMER_SAMPLE_GUIDANCE,
        'steps': FULL_TRANSFORMER_SAMPLE_STEPS,
        'guidance_scale_range': [lo_cfg, hi_cfg],
        'steps_range': [lo_steps, hi_steps],
        'note': note,
    }


def dense_max_step_saves_for(ds) -> int:
    """THE number the Hugging Face storage forecast multiplies a ~26 GB
    checkpoint by. Exported because two callers outside this module (the cloud
    launch pre-check and the Settings storage card) must never re-derive it —
    a forecast that disagrees with the emitted job config is the exact failure
    that cost run #146 its last 250 steps. ``None`` = no dataset in hand, i.e.
    the shipped default."""
    return FULL_TRANSFORMER_MAX_STEP_SAVES if ds is None else _dense_max_step_saves(ds)

# Persisted/API contract. Keep this deliberately tiny: accepting aliases here
# would make provenance ambiguous and could silently turn a requested dense run
# back into a LoRA. Legacy/NULL rows resolve to the historical LoRA behaviour.
TRAINING_MODES = ('lora', 'full_transformer')


def normalize_training_mode(value) -> str:
    """Return one canonical training mode, or reject the request explicitly."""
    mode = 'lora' if value is None else value
    if mode not in TRAINING_MODES:
        raise ValueError("training_mode must be 'lora' or 'full_transformer'")
    return mode


def training_mode(ds, override=None) -> str:
    """Effective mode for a dataset/action; missing legacy state means LoRA."""
    value = getattr(ds, 'training_mode', None) if override is None else override
    return normalize_training_mode(value)


def _is_full_transformer(ds, override=None) -> bool:
    return training_mode(ds, override) == 'full_transformer'


def _assert_local_training_mode(ds, requested=None) -> str:
    """Reject dense training before any local export/spawn side effect."""
    mode = training_mode(ds, requested)
    if mode == 'full_transformer':
        raise ValueError(
            'full_transformer training is cloud-only; choose Cloud training or switch to LoRA')
    # An explicit LoRA selection switches a dataset back from a previously
    # persisted cloud dense mode before build_job_config reads the row.
    if (requested is not None and hasattr(ds, 'training_mode')
            and getattr(ds, 'training_mode', None) != mode):
        ds.training_mode = mode
        fds.db.session.commit()
    return mode

# The local-training state is a durable GPU ownership fence. It must never
# expire while a surviving ai-toolkit child may still own VRAM; only an exact
# process identity check is allowed to release it after a restart.
_TRAIN_STATE_TTL = None

# Serialises the entire mutable local-launch preparation (dataset export, fixed
# job/config paths, live-lane context and spawn).  Exact continuation acquires
# this before its short queue-lock sections, establishing one global lock order:
# launch transaction -> queue ownership -> GPU arbiter.
_launch_transaction_lock = threading.RLock()


def _serial_local_launch(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        with _launch_transaction_lock:
            return function(*args, **kwargs)
    return guarded


# --- Path accessors (replace SRC's module-level AITOOLKIT_DIR/HF_HOME/... constants) --

def _aitoolkit_dir():
    d = cfg.aitoolkit_path('dir')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _hf_home():
    d = cfg.aitoolkit_path('hf_home')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _hf_hub_cache() -> Path:
    """The cache_dir consumed by huggingface_hub under the child process HF_HOME."""
    return Path(_hf_home()) / 'hub'


def _datasets_dir():
    d = cfg.aitoolkit_path('datasets')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _output_dir():
    d = cfg.aitoolkit_path('output')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _venv_python():
    p = cfg.aitoolkit_path('venv_python')
    if not p:
        raise RuntimeError('ai-toolkit is not configured')
    return p


def _jobs_dir():
    d = cfg.aitoolkit_path('jobs')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    d.mkdir(parents=True, exist_ok=True)
    return d


def _machine_hf_token_files() -> list:
    """Every place `hf auth login` may have written a token on THIS machine, best
    first. Mirrors `huggingface_hub.constants` (read from 0.36): `$HF_HOME/token`,
    `$XDG_CACHE_HOME/huggingface/token`, `~/.cache/huggingface/token`.

    All three are listed rather than only the first: the app's own process may
    already run with `HF_HOME` pointed at the ai-toolkit cache, while the shell
    where the user typed `hf auth login` had none — the CLI token is then in the
    plain default home and only the last candidate finds it.

    Pure string math; huggingface_hub is NOT imported (it lives in the ai-toolkit
    venv, not necessarily in ours), and no home path is assumed — Linux, macOS
    and Windows all fall out of `expanduser`."""
    def _norm(p):
        return os.path.expanduser(os.path.expandvars(p))

    homes = []
    env_home = (os.environ.get('HF_HOME') or '').strip()
    if env_home:
        homes.append(_norm(env_home))
    xdg = (os.environ.get('XDG_CACHE_HOME') or '').strip()
    if xdg:
        homes.append(os.path.join(_norm(xdg), 'huggingface'))
    homes.append(os.path.join(_norm('~'), '.cache', 'huggingface'))

    out = []
    for h in homes:
        cand = os.path.join(h, 'token')
        if cand not in out:
            out.append(cand)
    return out


def training_subprocess_env(hf_home=None) -> dict:
    """The environment the LOCAL ai-toolkit process is launched with.

    `HF_HOME` routes base/adapter weights onto the configured disk and
    `PYTHONIOENCODING` keeps unicode logs from dying on cp1252.

    Hugging Face authentication, in the order huggingface_hub itself resolves it:

    1. the token saved in Settings ▸ API keys is injected EXPLICITLY, exactly
       like the cloud lane does (`cloud_training`) — the local lane used to rely
       on it happening to sit in this process's `os.environ`, which is an
       implementation detail of `cfg.secret`, not a contract;
    2. failing that, whatever the machine already carries is preserved: an
       `HF_TOKEN` in the environment (ai-toolkit's own `.env`, loaded by its
       `run.py`, is how this worked for the people it worked for) …
    3. … and, crucially, the token file `hf auth login` wrote. huggingface_hub
       reads it at `$HF_HOME/token`, so overriding `HF_HOME` for the cache HID
       a perfectly valid login and produced a 401 on gated bases (Krea 2,
       FLUX.1-dev, FLUX.2 Klein) — reported by SurpassHR on GitHub. `HF_TOKEN_PATH`
       is a separate variable that wins over `HF_HOME`, so we pin it at the real
       file. Relocating a CACHE must never log the user out.

    Never logs, and never copies a token anywhere but into this env dict.
    """
    env = dict(os.environ, HF_HOME=str(hf_home if hf_home is not None else _hf_home()),
               PYTHONIOENCODING='utf-8')
    token = (cfg.secret('HF_TOKEN') or '').strip()
    if token:
        env['HF_TOKEN'] = token
        return env
    if (env.get('HF_TOKEN') or '').strip():
        return env                                   # already authenticated by env
    if (env.get('HF_TOKEN_PATH') or '').strip():
        return env                                   # user pinned it: respect it
    try:
        ours = os.path.join(env['HF_HOME'], 'token')
        # A login made WITH our HF_HOME already resolves — never redirect it away.
        if os.path.isfile(ours):
            return env
        for cand in _machine_hf_token_files():
            if cand != ours and os.path.isfile(cand):
                env['HF_TOKEN_PATH'] = cand
                break
    except OSError:
        pass                                         # unreadable home: change nothing
    return env


# ComfyUI-side destinations (deploy target for a trained LoRA, and the SDXL base
# checkpoint pool). Distinct error message from the aitoolkit accessors above:
# a dataset can be trainable (aitoolkit OK) while ComfyUI itself is unconfigured,
# and the two are gated independently by the Settings/capabilities probe.
#
# The loras root comes from `comfy_model_paths.write_root`, NOT from
# `cfg.comfyui_dir('loras')`: the latter only knows the explicit override and
# <base>/models/loras, so an `extra_model_paths.yaml` declaring the real loras
# folder was honoured when READING and ignored when WRITING — deploys and the
# "open LoRA folder" button both landed in the default folder (GitHub #25,
# Geekswordsman). Everything a user can still find on disk keeps being read from
# EVERY root (see `_lora_family_dirs`), so a LoRA deployed before this change is
# still listed, resolvable and deletable.
def _loras_root():
    from . import comfy_model_paths
    d = comfy_model_paths.write_root('loras')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d


# Family -> its subfolder under a loras root. Single source for the deploy
# accessors AND for the multi-root read helpers below.
_FAMILY_SUBDIR = {'zimage': 'z image', 'sdxl': 'sdxl', 'krea': 'krea',
                  'flux': 'flux', 'flux2klein': 'flux2klein', 'anima': 'anima'}


def _lora_dest_dir_zimage():
    return os.path.join(_loras_root(), 'z image')


def _lora_dest_dir_sdxl():
    return os.path.join(_loras_root(), 'sdxl')


def _lora_dest_dir_krea():
    return os.path.join(_loras_root(), 'krea')


def _lora_dest_dir_flux():
    return os.path.join(_loras_root(), 'flux')


def _lora_dest_dir_flux2klein():
    return os.path.join(_loras_root(), 'flux2klein')


def _lora_dest_dir_anima():
    return os.path.join(_loras_root(), 'anima')


def _lora_family_dirs(fam: str) -> list[str]:
    """Every folder where a LoRA of this family can be found, write folder FIRST
    then the other roots ComfyUI searches, de-duplicated.

    Writing needs one folder; reading must not lose sight of the others. Without
    this, changing the deploy root would have orphaned every LoRA already deployed
    under `<base>/models/loras` — still on disk, still loadable by ComfyUI, but
    gone from the "IN COMFYUI" list and undeletable from the app."""
    from . import comfy_model_paths
    sub = _FAMILY_SUBDIR.get((fam or '').lower(), 'z image')
    roots = []
    try:
        roots.append(_loras_root())
    except RuntimeError:
        pass
    roots += comfy_model_paths.search_roots('loras')
    out, seen = [], set()
    for root in roots:
        p = os.path.join(root, sub)
        key = os.path.normcase(os.path.normpath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _resolve_deployed_path(fam: str, filename: str) -> str | None:
    """Absolute path of a deployed LoRA named in LoraLoader form
    (`<family subfolder>\\name.safetensors`), searched across every loras root.
    Fail-closed on path traversal: the resolved file must sit inside the family
    folder of the root it was found in. None when it is nowhere."""
    rel = str(filename or '').replace('\\', os.sep).replace('/', os.sep).strip(os.sep)
    if not rel:
        return None
    for d in _lora_family_dirs(fam):
        root = os.path.abspath(d)
        cand = os.path.abspath(os.path.join(os.path.dirname(root), rel))
        try:
            inside = os.path.commonpath([cand, root]) == root
        except ValueError:            # different drives
            continue
        if inside and os.path.isfile(cand):
            return cand
    return None


def _sdxl_checkpoints_dir():
    d = cfg.comfyui_dir('models')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d / 'checkpoints'


def is_installed() -> bool:
    """ai-toolkit est-il installé (venv python présent) ?"""
    p = cfg.aitoolkit_path('venv_python')
    return bool(p) and p.is_file()


def assert_interpreter_ready() -> None:
    """Refuse a launch whose interpreter provably cannot `import torch`, NAMING
    the path — the only fact that makes the failure obvious in one second.

    `is_installed()` above only asks whether a file is there. A path that exists,
    runs, and carries none of ai-toolkit's dependencies passes it, the run starts,
    and ai-toolkit dies on `ModuleNotFoundError: No module named 'torch'` while the
    panel suggests a missing base model or a Hugging Face token (GitHub #19,
    strouder — an `aitoolkit.python` pointing at the Windows Store python stub
    while a working venv sat next to run.py).

    Refuses ONLY on a proven False. An unknown probe (cold-import timeout) lets
    the launch through: blocking a real training run on an answer we do not have
    would be a worse bug than the one this fixes. RuntimeError -> 409 (a backend
    availability problem, not a bad request)."""
    from .. import capabilities
    from .training_diagnostics import interpreter_verdict
    try:
        report = capabilities.aitoolkit_interpreter_report()
    except Exception:
        return                                   # a broken probe never blocks a run
    verdict = interpreter_verdict(report['python'], report['torch'],
                                  alternative=report['alternative'])
    if verdict:
        raise RuntimeError(verdict['message'])


def _aitoolkit_supports_krea() -> bool:
    """L'ai-toolkit installé connaît-il l'arch Krea 2 ? C'est CRITIQUE : ai-toolkit
    fait `if ModelClass.arch == config.arch` puis, sans match, retombe
    SILENCIEUSEMENT sur le loader SD legacy (get_model.py:get_model_class) - aucune
    erreur levée. Une config `arch:'krea2'` sur un ai-toolkit pas à jour chargerait
    donc Krea-2-Turbo comme un checkpoint SD et planterait de façon confuse. On
    scanne les sources d'archs (extensions_built_in) ; lecture fraîche → dès que
    le mainteneur fait `git pull`, la détection passe à True sans redémarrage.

    On exige l'arch EXACTE `arch = "krea2"` (la chaîne émise par _build_job_config_krea),
    pas la simple sous-chaîne « krea » : sinon une mention incidente (commentaire,
    variable) ferait un FAUX POSITIF, et surtout si l'arch upstream diffère (ex.
    « krea2_turbo ») la garde donnerait un feu vert alors que get_model_class ne
    matcherait pas → fallback SD silencieux, précisément ce qu'on veut empêcher."""
    root = cfg.aitoolkit_path('dir')
    if not root:
        return False
    ext_root = root / 'extensions_built_in'
    if not ext_root.is_dir():
        return False
    pat = re.compile(r'arch\s*=\s*[\'"]krea2[\'"]')
    for dp, _dn, files in os.walk(str(ext_root)):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            try:
                with open(os.path.join(dp, fn), encoding='utf-8', errors='ignore') as fh:
                    if pat.search(fh.read()):
                        return True
            except OSError:
                continue
    return False


def _aitoolkit_supports_flux2klein() -> bool:
    """L'ai-toolkit installé connaît-il FLUX.2 Klein ? Même enjeu CRITIQUE que
    _aitoolkit_supports_krea (lire son commentaire) : les archs flux2_klein_4b/9b
    sont des EXTENSIONS (extensions_built_in/diffusion_models/flux2), pas des archs
    cœur comme 'flux' — un ai-toolkit pas à jour ne les connaît pas et
    get_model_class retomberait SILENCIEUSEMENT sur le loader SD legacy → LoRA
    corrompu. On exige l'arch EXACTE `arch = "flux2_klein_4b"` ou `"..._9b"` (les
    chaînes émises par _build_job_config_flux2klein), jamais la sous-chaîne
    « klein » seule — une mention incidente ferait un faux positif. Lecture
    fraîche : un `git pull` du mainteneur passe la détection à True sans restart."""
    root = cfg.aitoolkit_path('dir')
    if not root:
        return False
    ext_root = root / 'extensions_built_in'
    if not ext_root.is_dir():
        return False
    pat = re.compile(r'arch\s*=\s*[\'"]flux2_klein_(?:4b|9b)[\'"]')
    for dp, _dn, files in os.walk(str(ext_root)):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            try:
                with open(os.path.join(dp, fn), encoding='utf-8', errors='ignore') as fh:
                    if pat.search(fh.read()):
                        return True
            except OSError:
                continue
    return False


def _aitoolkit_supports_anima() -> bool:
    """L'ai-toolkit installé connaît-il Anima ? Même enjeu CRITIQUE que
    _aitoolkit_supports_krea (lire son commentaire) : l'arch 'anima' est une
    EXTENSION (extensions_built_in/diffusion_models/anima, PR ostris/ai-toolkit
    #860 mergée le 2026-07-15), pas une arch cœur — un ai-toolkit antérieur ne la
    connaît pas et get_model_class retomberait SILENCIEUSEMENT sur le loader SD
    legacy → LoRA corrompu. On exige l'arch EXACTE `arch = "anima"` (la chaîne
    émise par _build_job_config_anima). Lecture fraîche : un `git pull` du
    mainteneur passe la détection à True sans restart. ⚠️ Cette garde vérifie la
    PRÉSENCE de l'arch dans les sources, PAS la version de diffusers : Anima exige
    aussi un diffusers récent (AnimaModularPipeline/CosmosTransformer3DModel) —
    un checkout à jour mais un venv ancien lèvera un ImportError au chargement
    (même angle mort que krea2/flux2_klein)."""
    root = cfg.aitoolkit_path('dir')
    if not root:
        return False
    ext_root = root / 'extensions_built_in'
    if not ext_root.is_dir():
        return False
    pat = re.compile(r'arch\s*=\s*[\'"]anima[\'"]')
    for dp, _dn, files in os.walk(str(ext_root)):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            try:
                with open(os.path.join(dp, fn), encoding='utf-8', errors='ignore') as fh:
                    if pat.search(fh.read()):
                        return True
            except OSError:
                continue
    return False


def _aitoolkit_supports_automagic3() -> bool:
    """Whether this checkout can resolve the Automagic3 optimizer.

    Automagic3 is newer than the other optimizer choices and older ai-toolkit
    checkouts fail only when constructing the optimizer.  Detect the concrete
    implementation and registry entry so local preflight/launch can fail early
    with an actionable update instruction.
    """
    root = cfg.aitoolkit_path('dir')
    if not root:
        return False
    implementation = root / 'toolkit' / 'optimizers' / 'automagic3.py'
    registry = root / 'toolkit' / 'optimizer.py'
    if not implementation.is_file() or not registry.is_file():
        return False
    try:
        return bool(re.search(
            r"(?:lower_type|optimizer_type)\s*==\s*['\"]automagic3['\"]",
            registry.read_text(encoding='utf-8', errors='ignore')))
    except OSError:
        return False


def _safe_trigger(ds) -> str:
    t = (ds.trigger_word or f'dataset{ds.id}').strip()
    return ''.join(c if (c.isalnum() or c in '_-') else '_' for c in t) or f'dataset{ds.id}'


def _train_type(ds, family=None) -> str:
    """Famille de modèle entraînée : 'zimage' (défaut/None), 'sdxl', 'krea',
    'flux' ou 'flux2klein'.
    `family` (override) prime sur le train_type persisté quand fourni (non vide) -
    c'est ce qui permet au sélecteur de famille de l'UI de piloter la lecture des
    runs/checkpoints/déploiements SANS écraser le train_type persisté du dataset."""
    return ((family or None) or getattr(ds, 'train_type', None) or 'zimage').lower()


def _lora_dest_dir(ds, family=None) -> str:
    """Dossier loras ComfyUI où DÉPLOYER le LoRA entraîné, routé par famille :
    krea → loras/krea/ (pour qu'il apparaisse dans le menu de génération Krea via
    get_krea_loras), sdxl → loras/sdxl/, zimage (défaut) → « z image/ ». Garde les
    familles séparées (un LoRA Krea ne doit pas polluer le Test Studio Z-Image)."""
    fam = _train_type(ds, family)
    if fam == 'sdxl':
        return str(_lora_dest_dir_sdxl())
    if fam == 'krea':
        return str(_lora_dest_dir_krea())
    if fam == 'flux':
        return str(_lora_dest_dir_flux())
    if fam == 'flux2klein':
        return str(_lora_dest_dir_flux2klein())
    if fam == 'anima':
        return str(_lora_dest_dir_anima())
    return str(_lora_dest_dir_zimage())


def _sdxl_base_choices() -> set:
    """Whitelist serveur des bases SDXL = basenames des checkpoints ComfyUI.
    include_hidden=True pour ne pas exclure un checkpoint masqué légitime, et
    pour récupérer une forme stable quelle que soit la variante de retour.

    Union with every checkpoint reachable through ``extra_model_paths.yaml``:
    get_checkpoint_models only knows ``<base>/models/checkpoints``, so on a
    portable / Stability-Matrix / A1111-shared install this whitelist rejected
    ('unknown SDXL checkpoint') bases that ComfyUI itself loads. Strictly ADDITIVE
    — with no yaml the extra set is empty and this is byte-for-byte the old one."""
    from ..utils.comfyui import get_checkpoint_models
    out = set()
    for c in (get_checkpoint_models(include_hidden=True) or []):
        out.add(c['name'] if isinstance(c, dict) else c)
    try:
        from . import comfy_model_paths
        for rel, _ab in comfy_model_paths.list_models('checkpoints'):
            out.add(os.path.basename(rel))
    except Exception:
        pass
    return out


def _sdxl_base_path(base_model: str) -> str:
    """Résout le .safetensors SDXL parmi TOUTES les racines `checkpoints` que
    ComfyUI utiliserait (``<base>/models/checkpoints`` + les racines déclarées dans
    ``extra_model_paths.yaml``), dans l'ordre de priorité de ComfyUI. get_checkpoint_models
    APLATIT en basename (l'info de sous-dossier - ex. Biglove/ - est perdue) → la
    recherche par basename est conservée. Refuse chemin absolu / '..' (anti-traversal ;
    la whitelist amont _sdxl_base_choices garantit déjà un basename connu).

    Raises ValueError NAMING the file when nothing matches. The old code returned the
    bare name here, which is what made this hole expensive rather than merely wrong:
    ai-toolkit received `bigLove_photo5.safetensors`, resolved it against its OWN
    working directory, and died with a message about a path the user never typed. A
    model we cannot find has to be said here, by name, while we still know which name
    was asked for."""
    name = str(base_model or '')
    parts = name.replace('\\', '/').split('/')
    if os.path.isabs(name) or '..' in parts:
        raise ValueError('invalid SDXL base path')
    _sdxl_checkpoints_dir()   # raises the explicit 'ComfyUI is not configured'
    from . import comfy_model_paths
    hit = comfy_model_paths.resolve_model_file('checkpoints', name)
    if hit:
        return hit
    raise ValueError(
        f'SDXL base checkpoint not found: {name} - looked in every ComfyUI '
        'checkpoints folder, including the ones declared in extra_model_paths.yaml')


# --- Custom weights (V1 « Custom weights… », local-only) ----------------------
# A base VALUE that is a free ABSOLUTE local path to a .safetensors is the
# opt-in custom-weights field: krea/flux/flux2klein/sdxl load it as name_or_path
# (same architecture, TE/VAE still official for the non-sdxl families). It is
# distinguished from a ComfyUI-relative base name (SDXL whitelist basename,
# Z-Image merge value) purely by being ABSOLUTE — those are never absolute. Only
# the families below expose it; Z-Image keeps its own conversion path untouched.
CUSTOM_WEIGHTS_FAMILIES = ('sdxl', 'krea', 'flux', 'flux2klein')
# SDXL is the ONLY family where ai-toolkit honours a top-level vae_path /
# te_name_or_path override (stable_diffusion_model.py). Every other family
# bundles its TE/VAE (Z-Image extras_name_or_path, Klein's hardcoded MISTRAL_PATH
# → a silent no-op) so exposing them there would lie — strict per-family whitelist.
VAE_TE_OVERRIDE_FAMILIES = ('sdxl',)


def _is_custom_weights(value) -> bool:
    """True when `value` is the opt-in custom-weights path (a free ABSOLUTE local
    path), as opposed to a ComfyUI-relative base/merge name or the official base."""
    return bool(value) and os.path.isabs(str(value))


def assert_trainable_base_file(path) -> dict:
    """Refuse a base the trainer CANNOT LOAD — and only that one — at selection.

    The community publishes fp8/int8 repacks of every popular base (~10 GB
    instead of ~26 GB) and they are the files most people already have on disk.
    What makes one unusable is its FORMAT, not its bit width, and the two forms
    behave differently (measured — see model_integrity's block comment):

    * a STRUCTURED export (ComfyUI scaled fp8 / comfy_quant, int8 repacks, this
      app's own fp8 twin) ships extra dequantization tensors, and ai-toolkit
      loads a base with ``load_state_dict(..., strict=True)`` — the load itself
      raises, immediately. Refused here, so the failure lands when the file is
      PICKED rather than after the dataset export and (in the cloud lane) after
      a GPU has been rented, for a few kilobytes of header.
    * a BARE cast adds no key of its own; the loader up-casts it to the training
      dtype and nothing in the PACKING stands in the way. Allowed —
      `model_integrity.base_precision_warning` states what it costs instead. It
      can still be refused at load for an unrelated reason (a tensor the
      architecture does not declare); this guard reads the packing only, and its
      wording is careful not to promise otherwise.

    An earlier version of this docstring claimed the trainer "dies deep in the
    first optimizer step". It does not, for either form, and that sentence was
    used to justify scoping decisions elsewhere — hence the detail here.

    Returns the report (``checked=False`` = unreadable header → deliberately
    permissive: the integrity validator owns "this file is broken", and refusing
    a base nobody could inspect would be worse than the failure it prevents).
    """
    from . import model_integrity
    report = model_integrity.quantization_report(path)
    if not report.get('trainable_as_base', True):
        raise ValueError(model_integrity.QUANT_REFUSAL)
    return report


_SAFETENSORS_MAX_HEADER = 64 * 1024 * 1024   # 64 MB — a real header is < ~10 MB


def _read_safetensors_header(path):
    """(`__metadata__` dict, tensor-NAME set) of a .safetensors file, read from
    its header WITHOUT loading a single weight (8-byte LE length + JSON metadata
    block). Raises ValueError when the file isn't a readable safetensors
    container. The metadata block is where ai-toolkit stamps ss_base_model_version
    (the strongest architecture signal); the tensor names are the fallback sniff."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError('file too short to be a safetensors container')
            n = struct.unpack('<Q', raw)[0]
            if n <= 0 or n > _SAFETENSORS_MAX_HEADER:
                raise ValueError('implausible safetensors header length')
            blob = fh.read(n)
            if len(blob) != n:
                raise ValueError('truncated safetensors header')
            meta = json.loads(blob.decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise ValueError(f'not a readable .safetensors file ({e})')
    if not isinstance(meta, dict):
        raise ValueError('not a readable .safetensors file (header is not an object)')
    md = meta.get('__metadata__')
    if not isinstance(md, dict):
        md = {}
    return md, {k for k in meta if k != '__metadata__'}


def _safetensors_tensor_keys(path) -> set:
    """The tensor NAMES of a .safetensors file, read from its header WITHOUT
    loading a single weight. Raises ValueError when the file isn't a readable
    safetensors container."""
    return _read_safetensors_header(path)[1]


def _detect_safetensors_arch(keys) -> str | None:
    """Best-effort architecture family from tensor NAMES only. Returns one of
    'sdxl' | 'sd15' | 'flux' | 'krea2', or None when undetectable. 'flux' covers
    BOTH FLUX.1 and FLUX.2 Klein — their DiT stream blocks are named identically,
    so a name-only sniff cannot tell them apart (an honest V1 limitation; a wrong
    FLUX.1↔FLUX.2 file still fails loudly at load, on a shape mismatch)."""
    def has(sub):
        return any(sub in k for k in keys)
    # SDXL LDM single-file checkpoint: the tell is the SECOND (OpenCLIP bigG) text
    # encoder — SD1.5 has a single encoder under cond_stage_model.
    if has('conditioner.embedders.1.'):
        return 'sdxl'
    if has('cond_stage_model.'):
        return 'sd15'
    # Krea2 SingleStreamDiT MMDiT: 'txtfusion' is unique to it.
    if has('txtfusion.'):
        return 'krea2'
    # FLUX-family DiT (FLUX.1 / FLUX.2 Klein): double + single stream blocks
    # (BFL layout) or the diffusers export naming.
    if (has('double_blocks.') and has('single_blocks.')) \
            or has('single_transformer_blocks.'):
        return 'flux'
    return None


# --- Trained-LoRA architecture detector (the deploy/Studio guardrail) ---------
# The base sniff above targets full UNET checkpoints (a BASE); a TRAINED LoRA has
# a different, prefixed key layout (lora_A/lora_B, lokr_w*, kohya lora_unet_*). A
# wrong-arch LoRA is invisible to ComfyUI: it drops every incompatible key
# SILENTLY, so the whole grid renders as if the LoRA were off (the 2026-07-13
# incident — a Z-Image LoRA mislabelled Krea produced 117 no-op tiles). We read
# the real arch from the header and check it wherever a LoRA is deployed or run.
#
# Verdict = FAMILY key ('zimage'|'sdxl'|'krea'|'flux'|'flux2klein') or None
# (undetectable → callers MUST NOT block; the guarantee is simply absent).
_LORA_ARCH_LABEL = {'zimage': 'Z-Image', 'sdxl': 'SDXL', 'krea': 'Krea 2',
                    'flux': 'FLUX.1', 'flux2klein': 'FLUX.2 Klein'}
# Key-namespace GROUP: two families in the SAME group share the tensor namespace,
# so a wrong file loads its keys (a version mismatch then fails LOUDLY on a shape
# error, not silently). Different groups = disjoint names = SILENT drop = the
# danger we block. FLUX.1 and FLUX.2 Klein share the double/single-stream layout,
# so they're one group (a name-only sniff can't tell them apart anyway).
_LORA_ARCH_NAMESPACE = {'zimage': 'zimage', 'sdxl': 'sdxl', 'krea': 'krea',
                        'flux': 'flux', 'flux2klein': 'flux', 'anima': 'anima'}


def _family_from_base_model_version(value) -> str | None:
    """Map ai-toolkit's ss_base_model_version metadata to a FAMILY key. Real
    values observed on deployed LoRAs (C:\\ai-toolkit: toolkit/metadata.py stamps
    'sdxl_1.0'/'sd_1.5'/'sd_2.1'; each newer arch's get_base_model_version returns
    'zimage' / 'krea2' / 'flux' / 'flux2_klein_4b' / 'flux2_klein_9b' / 'anima').
    SD1.5/2.1 and any foreign value → None (not one of our trainable families)."""
    v = str(value or '').strip().lower()
    if not v:
        return None
    if 'anima' in v:                     # AnimaModel.get_base_model_version() → 'anima'
        return 'anima'
    if v.startswith(('flux2_klein', 'flux2klein')):
        return 'flux2klein'
    if v.startswith('flux'):
        return 'flux'
    if 'zimage' in v or 'z_image' in v or 'z-image' in v:
        return 'zimage'
    if 'krea' in v:                      # 'krea2'
        return 'krea'
    if v.startswith(('sdxl', 'sd_xl')):
        return 'sdxl'
    return None


def _lora_arch_from_keys(keys) -> str | None:
    """Best-effort FAMILY from a trained LoRA's tensor NAMES (fallback when the
    metadata is absent/foreign). Signatures verified against real deployed LoRAs:
      - kohya SD/SDXL: 'lora_unet_*' / 'lora_te*' prefixes                → sdxl
      - FLUX-family DiT: 'double_blocks.'/'single_blocks.' (BFL) or the
        diffusers 'single_transformer_blocks.' — FLUX.1 AND FLUX.2 Klein  → flux
      - Krea2 SingleStreamDiT: 'txtfusion' is unique to it (present even
        in a header-only stub); or diffusion_model.blocks.*.attn.{wk,wq,gate} → krea
      - Z-Image NextDiT: 'diffusion_model.layers.*' (adaLN / attention.to_*) → zimage
      - Anima (Cosmos Predict2 DiT): the ComfyUI-converted keys use
        'diffusion_model.llm_adapter.*' (text conditioner) and, inside
        'diffusion_model.blocks.*', the Cosmos-specific 'self_attn.q_proj' /
        'cross_attn.*' / 'adaln_modulation_self_attn' names (PR #860's
        _convert_diffusers_lora_key_to_comfy) → anima. These are DISJOINT from
        Krea's 'diffusion_model.blocks.*.attn.wk/wq/gate' + 'txtfusion', so the
        two never cross-detect (regression guard test pins both directions).
    A name-only sniff can't separate FLUX.1 from FLUX.2 Klein → 'flux' for both."""
    def has(sub):
        return any(sub in k for k in keys)
    if has('lora_unet_') or has('lora_te'):
        return 'sdxl'
    if has('double_blocks.') or has('single_blocks.') \
            or has('single_transformer_blocks.'):
        return 'flux'
    if has('txtfusion') or (has('diffusion_model.blocks.')
                            and (has('.attn.wk') or has('.attn.wq')
                                 or has('.attn.gate'))):
        return 'krea'
    # Anima BEFORE the generic zimage 'layers.' check. Its signature keys off the
    # Cosmos-specific tensor names so it can never match a Krea block (which uses
    # .attn.wk/wq/gate and has no llm_adapter / self_attn.q_proj / adaln_modulation).
    if has('diffusion_model.llm_adapter.') or has('adaln_modulation_self_attn') \
            or (has('diffusion_model.blocks.') and has('self_attn.q_proj')):
        return 'anima'
    if has('diffusion_model.layers.'):
        return 'zimage'
    return None


# Header reads are pure functions of the file bytes; a deployed LoRA never mutates
# in place. Cache the verdict by (abspath, mtime_ns, size) so repeated listing /
# preflight passes read each header at most once.
_LORA_ARCH_CACHE: dict = {}


def detect_lora_arch(path) -> str | None:
    """The real FAMILY of a trained LoRA .safetensors, read from its header
    WITHOUT loading a single weight. Returns 'zimage'|'sdxl'|'krea'|'flux'|
    'flux2klein', or None when undetectable (unreadable/foreign header, or a
    layout we don't recognize) — callers treat None as 'no guarantee, do not
    block'. Never raises. Metadata (ss_base_model_version) wins over the tensor
    sniff; only the sniff can appear when the metadata was stripped."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _LORA_ARCH_CACHE:
        return _LORA_ARCH_CACHE[key]
    fam = None
    try:
        md, keys = _read_safetensors_header(path)
        fam = _family_from_base_model_version(md.get('ss_base_model_version'))
        if fam is None:
            fam = _lora_arch_from_keys(keys)
    except ValueError:
        fam = None
    _LORA_ARCH_CACHE[key] = fam
    return fam


def lora_arch_conflicts(detected, family) -> bool:
    """True only when a POSITIVELY-detected LoRA arch cannot be loaded by
    `family`'s pipeline (different key namespace → ComfyUI drops every key
    SILENTLY → the LoRA is a no-op). None/unknown on either side → False (never a
    false block). flux vs flux2klein share a namespace (a wrong version fails
    LOUDLY on a shape error) → not a conflict."""
    if not detected:
        return False
    dg = _LORA_ARCH_NAMESPACE.get(detected)
    fg = _LORA_ARCH_NAMESPACE.get((family or '').lower())
    if dg is None or fg is None:
        return False
    return dg != fg


_FAMILY_EXPECTED_ARCH = {'sdxl': 'sdxl', 'krea': 'krea2',
                         'flux': 'flux', 'flux2klein': 'flux', 'anima': 'anima'}
_ARCH_LABEL = {'sdxl': 'an SDXL', 'sd15': 'a Stable Diffusion 1.5',
               'flux': 'a FLUX', 'krea2': 'a Krea 2', 'anima': 'an Anima'}
# THE family display name — one map, used by preflight_custom_paths,
# foreign_base_message, training_preflight and the run-summary lines.
#
# It was defined TWICE in this module (here, and again ~4000 lines below). Python
# keeps the LAST binding, so at runtime the second one always won and this one was
# dead code — nothing was ever mislabelled. What was lost was READABILITY: this
# copy had no 'zimage' entry, so anyone reading the callers here concluded that
# Z-Image falls through to its raw key, and a fix applied to this map would have
# had no effect at all. The two are merged, with 'zimage' kept (the runtime
# behaviour of the surviving definition), and the lower one deleted.
_FAMILY_LABEL = {'zimage': 'Z-Image', 'sdxl': 'SDXL', 'krea': 'Krea 2',
                 'flux': 'FLUX.1', 'flux2klein': 'FLUX.2 Klein', 'anima': 'Anima'}
# Caption FORM each family is prompted with — the only input of the
# MISMATCH_CAPTION guard (assert_trainable). Three values:
#   'booru'  the model is tag-native (SDXL booru checkpoints, e.g. bigLove);
#   'prose'  natural language only (the default for every family not listed);
#   None     HYBRID — both forms are first-class, so no mismatch can exist.
# Anima is the hybrid case: its model card documents booru tags AND natural
# language as supported prompting styles (Cosmos-Predict2 2B backbone, Qwen LLM
# text encoder — an LLM text encoder is what makes both readable). LDS used to
# lump it with 'prose' by omission and refused booru-captioned anima datasets;
# that was half a truth, and the mirror image of the "booru only" half someone
# else had. Anything absent from this table keeps the historical prose default.
_EXPECTED_CAPTION_FORM = {'sdxl': 'booru', 'anima': None}
# Confirmable-refusal marker (mirrors UNCAPTIONED:/MISMATCH_CAPTION:): the UI
# strips it, asks window.confirm, and retries with allow_unverified_weights.
_UNVERIFIED_MARKER = 'CUSTOM_WEIGHTS_UNVERIFIED: '


def _looks_like_local_path(s) -> bool:
    """A te_name_or_path may be a HF repo id ('org/name') OR a local dir/file.
    Treat it as LOCAL (and therefore existence-checkable) only when it is an
    absolute path, already exists, or carries a Windows backslash — a bare
    'org/name' repo id stays unverifiable (accepted as-is)."""
    s = str(s)
    return bool(s) and (os.path.isabs(s) or os.path.exists(s) or '\\' in s)


def preflight_custom_paths(family, weights=None, vae_path=None, te_path=None,
                           allow_unverified_weights=False) -> None:
    """Validate the custom base/vae/te BEFORE any run dir or spawn (guardrail).

    HARD failures (→ ValueError, mapped to 400): a provided path that does not
    exist, or a .safetensors whose header can't be parsed. A file whose
    architecture can't be POSITIVELY matched to `family` raises a CONFIRMABLE
    ValueError (the _UNVERIFIED_MARKER) unless `allow_unverified_weights` — the
    same confirm-and-retry contract as UNCAPTIONED. vae_path/te_path are only
    ever passed for SDXL (the caller enforces the per-family whitelist)."""
    fam_label = _FAMILY_LABEL.get(family, family)
    if _is_custom_weights(weights):
        if not os.path.isfile(weights):
            raise ValueError(f'custom weights file not found: {weights}')
        # Second seam for the same refusal the SELECTOR already makes: a base
        # persisted before that guard shipped, or restored from a preset/share,
        # must not reach a rented GPU either. Hard, never confirmable — an
        # inference-only export cannot be trained on at any confidence level.
        assert_trainable_base_file(weights)
        keys = _safetensors_tensor_keys(weights)   # raises on unreadable header
        detected = _detect_safetensors_arch(keys)
        expected = _FAMILY_EXPECTED_ARCH.get(family)
        if expected is None or detected != expected:
            if not allow_unverified_weights:
                if detected and detected in _ARCH_LABEL:
                    why = (f'this file looks like {_ARCH_LABEL[detected]} checkpoint, '
                           f'not {fam_label}')
                else:
                    why = (f'cannot verify this file matches {fam_label} — it carries '
                           f'no recognizable {fam_label} signature')
                raise ValueError(f'{_UNVERIFIED_MARKER}{why}.')
    # VAE override (SDXL): a local file/dir must exist; a .safetensors must parse.
    if vae_path:
        if not os.path.exists(vae_path):
            raise ValueError(f'VAE file not found: {vae_path}')
        if os.path.isfile(vae_path) and str(vae_path).endswith('.safetensors'):
            _safetensors_tensor_keys(vae_path)     # raises on unreadable header
    # TE override (SDXL): a LOCAL path must exist; a bare HF repo id is accepted.
    if te_path and _looks_like_local_path(te_path) and not os.path.exists(te_path):
        raise ValueError(f'text-encoder path not found: {te_path}')


# Sentinelle « base non fournie » : distingue l'absence d'argument (→ base
# PERSISTÉE du dataset) de la valeur '' (= base officielle, un choix explicite).
_PERSISTED = object()


def _base_tag_for(base_model) -> str:
    """Suffixe de run pour une base EXPLICITE ('' / None = officiel → '')."""
    if not base_model:
        return ''
    base = os.path.basename(str(base_model).replace('\\', '/')).rsplit('.', 1)[0]
    safe = ''.join(c if (c.isalnum() or c in '_-') else '_' for c in base)
    return f'_{safe}' if safe else ''


def _base_tag(ds) -> str:
    """Suffixe de run dérivé de la base d'entraînement PERSISTÉE (vide = officiel).
    Isole les checkpoints d'un run sur merge de ceux du run officiel du même
    dataset (sinon ai-toolkit auto-resume depuis le mauvais base → mélange)."""
    return _base_tag_for(getattr(ds, 'train_base_model', None))


def official_base_repo(ds, family=None, variant=_PERSISTED):
    """The Hugging Face repo a CLOUD pod will download for the OFFICIAL base of this
    recipe, or None when the run uses custom local weights (nothing to fetch) or a
    family whose base is not an HF repo.

    Exists for one reason: a gated repo the account has not been granted access to
    fails with a 403 **on the pod**, i.e. after a GPU has been rented and paid for.
    Three runs were burned that way on krea/Krea-2-Raw. Resolving the repo here lets
    the launch check the gate BEFORE reserving anything.

    Mirrors the recipe decisions (`_krea_is_raw`, `_flux2klein_is_9b`, the Z-Image
    variant matrix) rather than restating them, so a recipe change cannot silently
    leave this behind."""
    fam = _train_type(ds, family)
    weights = getattr(ds, 'train_base_model', None)
    if _is_custom_weights(weights) or (weights and os.path.isfile(str(weights))):
        return None                       # local file: the pod never asks HF for it
    if fam == 'krea':
        return 'krea/Krea-2-Raw' if _krea_is_raw(ds, variant) else 'krea/Krea-2-Turbo'
    if fam == 'flux':
        return 'black-forest-labs/FLUX.1-dev'
    if fam == 'flux2klein':
        return ('black-forest-labs/FLUX.2-klein-base-9B' if _flux2klein_is_9b(ds, variant)
                else 'black-forest-labs/FLUX.2-klein-base-4B')
    if fam == 'zimage':
        var = str((getattr(ds, 'train_variant', None) if variant is _PERSISTED else variant)
                  or 'turbo').lower()
        if var == 'deturbo':
            return ZIMAGE_DETURBO_BASE
        return ZIMAGE_BASE if var == 'base' else ZIMAGE_TURBO_BASE
    if fam == 'anima':
        return ANIMA_BASE           # public, non-gated → the pre-rent HEAD returns 200
    return None


KREA_BASE_LABEL = 'Krea-2-Turbo'   # mirrors name_or_path 'krea/Krea-2-Turbo'
# Flux a une seule base officielle (FLUX.1-dev). Sans point dans le label (sinon
# _base_tag_for le prendrait pour une extension et tronquerait à « FLUX ») → tag
# stable '_FLUX-1-dev' qui isole les runs/LoRA Flux des runs Z-Image officiels
# (tag vide) au même trigger — même garde anti-collision que Krea (cf. _dest_base_tag).
FLUX_BASE_LABEL = 'FLUX-1-dev'
# FLUX.2 Klein a DEUX bases officielles (4B et 9B) → tags DISTINCTS obligatoires :
# les poids 4B et 9B sont incompatibles, et un même trigger entraîné sur les deux
# variantes partagerait sinon le même dossier de run (auto-resume croisé → LoRA
# corrompu) et le même nom de LoRA déployé. Sans point dans les labels (même piège
# d'extension que FLUX_BASE_LABEL : _base_tag_for tronque après un '.').
FLUX2KLEIN_BASE_LABELS = {'4b': 'FLUX2-Klein-4B', '9b': 'FLUX2-Klein-9B'}
# Anima (circlestone-labs, Cosmos Predict2 DiT, 2B) has a single OFFICIAL base and
# it is PUBLIC/non-gated (unlike Krea/FLUX/Klein) — the pre-rent HEAD in
# cloud_training returns 200, never a 403. Base tag has no dot (same extension
# trap as FLUX_BASE_LABEL: _base_tag_for truncates after a '.') so official Anima
# runs stay isolated from Z-Image runs of the same trigger (both would otherwise
# carry an empty tag → shared run folder → ai-toolkit cross-resume).
ANIMA_BASE = 'circlestone-labs/Anima-Base-v1.0-Diffusers'
ANIMA_BASE_LABEL = 'Anima-Base'
# Default preview negative for Anima, verbatim from ai-toolkit options.ts (PR #860,
# entry 'anima'). The score_1..3 / "artist name" tags are the anime-model convention
# the upstream UI ships — kept as the default so previews match ai-toolkit's own.
ANIMA_SAMPLE_NEG = ('worst quality, low quality, score_1, score_2, score_3, blurry, '
                    'jpeg artifacts, sepia, signature, artist name')

# Z-Image recipes are intentionally centralized here.  Before this guardrail,
# ``train_variant='base'`` / ``'deturbo'`` only removed the Turbo training
# adapter while the model itself silently stayed ``Z-Image-Turbo``.  That is the
# exact incompatible combination that produced unusable runs: a distilled base
# trained as if it were non-distilled.  Keep every launch path and the generated
# ai-toolkit config on this single matrix.
ZIMAGE_TURBO_BASE = 'Tongyi-MAI/Z-Image-Turbo'
ZIMAGE_BASE = 'Tongyi-MAI/Z-Image'
ZIMAGE_DETURBO_BASE = 'ostris/Z-Image-De-Turbo'
ZIMAGE_TURBO_TRAINING_ADAPTER = (
    'ostris/zimage_turbo_training_adapter/'
    'zimage_turbo_training_adapter_v2.safetensors')
ZIMAGE_RECIPE_VERSION = 1


def zimage_training_recipe(variant=None, base_model=None) -> dict:
    """Resolve and validate one complete Z-Image training recipe.

    Official selections are a strict matrix: Turbo uses the official distilled
    pipeline plus adapter v2; Base uses the real non-distilled Tongyi pipeline;
    De-Turbo uses Ostris' transformer and borrows tokenizer/TE/VAE from the
    official Turbo pipeline via ``extras_name_or_path``.  A converted local
    checkpoint keeps its own transformer and also borrows those shared extras;
    the explicitly declared variant determines whether adapter v2 is required.

    This helper is pure on purpose, so local launch, cloud reservation, queueing,
    provenance, and config generation can all consult the same contract before
    any GPU process/pod is started.
    """
    var = str(variant or 'turbo').strip().lower()
    if var not in ('turbo', 'base', 'deturbo'):
        raise ValueError(
            f"invalid Z-Image variant '{variant}' — expected turbo, base or deturbo")
    custom = (str(base_model).strip() if base_model else '')

    if custom:
        effective_base = custom
        extras = ZIMAGE_TURBO_BASE
    elif var == 'turbo':
        effective_base, extras = ZIMAGE_TURBO_BASE, None
    elif var == 'base':
        effective_base, extras = ZIMAGE_BASE, None
    else:
        # ostris/Z-Image-De-Turbo ships the transformer rather than a complete
        # pipeline, so ai-toolkit needs the shared tokenizer/TE/VAE explicitly.
        effective_base, extras = ZIMAGE_DETURBO_BASE, ZIMAGE_TURBO_BASE

    if var == 'turbo':
        adapter = ZIMAGE_TURBO_TRAINING_ADAPTER
        sample_steps, guidance, timestep_type = 8, 1, 'sigmoid'
    elif var == 'deturbo':
        adapter = None
        sample_steps, guidance, timestep_type = 25, 3, 'weighted'
    else:
        adapter = None
        sample_steps, guidance, timestep_type = 35, 4, 'weighted'

    return {
        'recipe_version': ZIMAGE_RECIPE_VERSION,
        'variant': var,
        'effective_base': effective_base,
        'extras_name_or_path': extras,
        'training_adapter': adapter,
        'sample_steps': sample_steps,
        'guidance_scale': guidance,
        'timestep_type': timestep_type,
        'custom_base': bool(custom),
    }


def assert_zimage_custom_recipe_confirmed(family, base_model, variant,
                                           allow_unverified_weights=False) -> None:
    """Require an explicit acknowledgement for unverifiable custom recipes.

    Safetensors architecture keys can identify Z-Image, but they cannot prove
    whether a custom transformer is distilled.  Declaring such a file as Base
    or De-Turbo therefore needs the same explicit confirmation used by the
    existing custom-weight preflight.  Turbo remains allowed because its safe
    recipe always installs the required training adapter.
    """
    if ((family or '').lower() == 'zimage' and str(base_model or '').strip()
            and str(variant or 'turbo').strip().lower() in ('base', 'deturbo')
            and not allow_unverified_weights):
        raise ValueError(
            f'{_UNVERIFIED_MARKER}a custom Z-Image checkpoint declared as '
            f'{str(variant).strip().lower()} cannot be verified as distilled '
            'or non-distilled. Confirm this recipe explicitly before export.')


# Families whose custom base is a free ABSOLUTE local file: a RELATIVE name there
# can only have been picked on another family (their builders gate on
# `_is_custom_weights`, so they ignore it outright — see the `name_or_path` lines
# in _build_job_config_krea/_flux/_flux2klein/_anima).
_ABSOLUTE_BASE_FAMILIES = ('krea', 'flux', 'flux2klein', 'anima')


def foreign_base_reason(family, base_model) -> str | None:
    """Why `base_model` provably cannot belong to `family` — or None.

    `train_base_model` is ONE column shared by every family, so a base picked for
    Z-Image stays attached when the family becomes Krea 2. The two shapes below
    are decidable without touching the disk or a ComfyUI config, which is what
    makes them safe to act on:

    * a RELATIVE name on a family whose custom lane is an absolute file path — it
      can only be a Z-Image merge (or an SDXL checkpoint) name;
    * an ABSOLUTE path on Z-Image, whose custom lane is a ComfyUI merge NAME that
      gets converted to diffusers first.

    SDXL is deliberately excluded: its bases are relative basenames too, and
    telling them apart from a Z-Image merge needs a configured ComfyUI — on an
    install that has none, every legitimate SDXL base would read as foreign.
    Companion of `assert_zimage_custom_recipe_confirmed`: same "the recipe must be
    coherent before anything is spawned or uploaded" job, one family-scope earlier.
    """
    base = str(base_model or '').strip()
    if not base:
        return None
    fam = (family or '').lower()
    if fam in _ABSOLUTE_BASE_FAMILIES and not _is_custom_weights(base):
        return 'relative_base_on_absolute_family'
    if fam == 'zimage' and _is_custom_weights(base):
        return 'absolute_base_on_zimage'
    return None


def foreign_base_message(family, base_model) -> str | None:
    """The human sentence for `foreign_base_reason`, or None when coherent.
    Names the family it can't belong to AND what actually happens, because
    "unavailable" was read as "my file is gone" when the file was fine."""
    if not foreign_base_reason(family, base_model):
        return None
    label = _FAMILY_LABEL.get(family, family)
    name = os.path.basename(str(base_model).replace('\\', '/'))
    return (f'“{name}” was chosen for another model family, not {label} — a '
            f'{label} run cannot load it, so this run uses the official '
            f'{label} base. Pick a {label} base to change that.')


def zimage_recipe_diagnostic(family, variant, effective_base=None,
                             training_adapter=None, recipe_version=None) -> dict | None:
    """Read-only safety annotation for Runs payloads, including legacy rows.

    Historical Turbo configs were coherent, but historical Base/De-Turbo rows
    without a stamped recipe were built from Z-Image-Turbo *without* its adapter.
    Mark them visibly; never stop, delete, or otherwise mutate an old/active run.
    """
    if (family or '').lower() != 'zimage':
        return None
    var = (variant or 'turbo').lower()
    if not recipe_version:
        if var in ('base', 'deturbo'):
            return {
                'status': 'legacy_incompatible',
                'warning': ('Legacy Z-Image Base/De-Turbo run: it predates the recipe '
                            'guardrail and may have trained Z-Image-Turbo without the '
                            'required adapter. The run was not stopped or modified.'),
            }
        return {'status': 'legacy_inferred_turbo', 'warning': None}

    try:
        official = {ZIMAGE_TURBO_BASE, ZIMAGE_BASE, ZIMAGE_DETURBO_BASE}
        custom_base = effective_base if effective_base not in official else None
        expected = zimage_training_recipe(var, custom_base)
    except ValueError as exc:
        return {'status': 'incompatible', 'warning': str(exc)}
    mismatch = (effective_base != expected['effective_base']
                or training_adapter != expected['training_adapter'])
    if mismatch:
        return {
            'status': 'incompatible',
            'warning': ('Stamped Z-Image recipe does not match the safe '
                        f"{var} base/adapter matrix."),
        }
    return {'status': 'safe', 'warning': None}


def _krea_is_raw(ds, variant=_PERSISTED) -> bool:
    """Krea 2 training base. `train_variant` 'base'/'raw' → Krea-2-Raw (non-distilled,
    the official recommendation « train on Raw, validate on Turbo » — best quality,
    the LoRA transfers to Turbo at inference); 'turbo' → Krea-2-Turbo + Ostris adapter
    (VRAM-friendly). Default RAW when unset — that's the chosen product default, so the
    tag and the job-config never disagree even if train_variant was never persisted."""
    selected = (getattr(ds, 'train_variant', None)
                if variant is _PERSISTED else variant)
    return str(selected or 'base').lower() in ('base', 'raw')


def _krea_name_or_path(ds, variant=_PERSISTED) -> str:
    """The base ai-toolkit LOADS for a Krea 2 run — the single resolver for both
    the LoRA lane and the dense lane.

    It exists because those two lanes used to answer this question differently:
    the LoRA branch read the selection, the dense branch pinned the constant. A
    dense run could therefore be launched, named and stamped "Turbo" while
    training Raw. One function, one answer, and a test asserts the emitted
    ``model.name_or_path`` per recipe rather than the absence of an exception.

    Custom weights (an ABSOLUTE local path) win over the variant: the file IS
    the base. The TE/VAE stay official — Krea bundles them, and ai-toolkit's
    krea2 arch takes no override for either."""
    base = getattr(ds, 'train_base_model', None)
    if _is_custom_weights(base):
        return str(base)
    return FULL_TRANSFORMER_BASE if _krea_is_raw(ds, variant) else KREA_TURBO_BASE


def _flux2klein_is_9b(ds, variant=_PERSISTED) -> bool:
    """FLUX.2 Klein model size. `train_variant` '9b' → the 9B base (32-48 GB VRAM,
    the cloud-first lane); anything else → the 4B base (16-24 GB, the local lane).
    Default 4B when unset — the chosen product default (mirrors _default_variant_for),
    so the run tag and the job-config never disagree even if train_variant was
    never persisted."""
    selected = (getattr(ds, 'train_variant', None)
                if variant is _PERSISTED else variant)
    return str(selected or '4b').lower() == '9b'


def _default_variant_for(family) -> str:
    """Variante par défaut d'une famille quand aucune n'est fournie NI persistée :
    Krea → 'base' (Raw, reco officielle), FLUX.2 Klein → '4b' (la voie locale
    16-24 Go ; le 9B est la voie cloud), sinon 'turbo'. Utilisé par tous les
    chemins de lancement (direct / file / reprise / cloud) pour que le défaut
    tienne de bout en bout, pas seulement quand l'UI envoie explicitement la variante."""
    fam = family or 'zimage'
    if fam == 'krea':
        return 'base'
    if fam == 'flux2klein':
        return '4b'
    return 'turbo'


def _valid_variants_for(family) -> tuple:
    """Variantes acceptées au lancement, PAR FAMILLE : flux2klein n'a que ses deux
    tailles de modèle ('4b'/'9b') ; les familles historiques gardent l'enum
    turbo/base/deturbo (comportement inchangé). Une variante hors liste retombe
    sur le défaut de la famille (jamais d'erreur) : c'est ce qui neutralise une
    variante PERSISTÉE d'une autre famille quand l'utilisateur change de type
    (ex. un dataset ex-Krea avec train_variant='base' lancé en flux2klein)."""
    return ('4b', '9b') if (family or 'zimage') == 'flux2klein' \
        else ('turbo', 'base', 'deturbo')


# --- Réglages ai-toolkit avancés, éditables par dataset (persistés en JSON dans
#     `train_settings`). Absent/NULL → défaut family-aware issu de la recherche
#     (cf. Research vault 2026-07-10). Toute valeur hors des listes autorisées
#     retombe sur le défaut : on ne pousse JAMAIS une config invalide à ai-toolkit. ---
_DEFAULT_RANK = {'zimage': 16, 'krea': 32, 'sdxl': 32, 'flux': 16, 'flux2klein': 16, 'anima': 32}   # Z-Image reste 16 (choix user) ; Krea/SDXL 32 ; Flux/FLUX.2 Klein 16 (défaut des exemples officiels) ; Anima 32 (defaultLinearRank ai-toolkit options.ts, PR #860)
# FLUX.2 Klein STYLE only : linear 128 (+ Conv2d 64) — la recette dominante du sweep
# Calvin Herbst (64 runs, fév. 2026) ET l'exemple de training officiel BFL, tous deux
# sur les dims 128/64/64/32 (ratio 4:2:2:1). Les AUTRES kinds Klein gardent 16.
_KLEIN_STYLE_RANK = 128
_RANK_CHOICES = (8, 16, 24, 32, 48, 64)
# multi-échelle par défaut ; '768' seul = LE levier basse-VRAM (Krea 12B : 1024
# sature un 24 GB à ~180 s/it, 768 mesuré ~3,5 s/it — cf. commentaire de tête).
_RES_CHOICES = {
    '512,768': [512, 768],
    '768,1024': [768, 1024],
    '1024': [1024],
    '768': [768],
}
_SAVE_CHOICES = (250, 500, 1000)
# --- Expert levers (train_settings, ALL default to current behaviour when absent,
#     so a newcomer who never touches them gets the exact same config as before) ---
_DROPOUT_CHOICES = (0.05, 0.1, 0.15, 0.2, 0.3)          # LoRA network dropout ; absent = off
_ALPHA_CHOICES = (1, 2, 4, 8, 16, 24, 32, 48, 64)       # alpha découplé du rank ; absent = dérivé
_TIMESTEP_TYPE_CHOICES = ('sigmoid', 'linear', 'weighted', 'shift')  # pondération flowmatch ; SDXL le désactive
# ⚠️ `shift` sur Krea : l'entraîneur calcule le décalage depuis un nombre de
# tokens qui suppose `unet.config.patch_size`, alors que le denoiser de Krea 2
# nomme ce champ `patch` → patch_size retombe à 1 et le compte sort 4× trop
# grand. Constaté dans l'ai-toolkit LOCAL (c'est bien lui qui exécute la voie
# LoRA) ET dans l'image figée du pod : ça ne dépend pas de la version. Laissé
# tel quel ici — le retirer changerait un réglage déjà stocké chez des gens —
# mais la voie DENSE ne le propose pas (cf. FULL_TRANSFORMER_TIMESTEP_TYPE_CHOICES).
# Les deux voies ne tournent PAS sur le même ai-toolkit : local (mouvant) pour
# les LoRA, image du pod (épinglée) pour le dense.
_DEFAULT_TIMESTEP = {'zimage': 'sigmoid', 'krea': 'linear', 'flux': 'sigmoid',
                     'flux2klein': 'weighted', 'anima': 'weighted'}   # ce que « Auto » résout (sdxl : aucun) ; flux subject → sigmoid (reco ai-toolkit) ; flux2klein → weighted (défaut canonique options.ts, PAS sigmoid) ; anima → weighted (défaut options.ts PR #860)
# Batch 2 — optimiseur / planning du LR / batch effectif (valeurs VÉRIFIÉES dans
# ai-toolkit : get_optimizer + toolkit/scheduler.py). CAME n'est PAS supporté.
_OPTIMIZER_CHOICES = (
    'adamw8bit', 'adafactor', 'automagic', 'automagic2', 'automagic3', 'prodigy')
_LR_SCHEDULER_CHOICES = ('constant', 'linear', 'cosine', 'cosine_with_restarts', 'constant_with_warmup')
_WARMUP_CHOICES = (50, 100, 200, 500)          # num_warmup_steps ; UNIQUEMENT avec constant_with_warmup
_GRAD_ACCUM_CHOICES = (1, 2, 4)
# Network variant + EMA — both VÉRIFIÉS arch-génériques dans ai-toolkit installé :
#   - network.type='lokr' : LoRASpecialNetwork choisit LokrModule pour TOUTE arch
#     (toolkit/lora_special.py L384 `elif self.network_type.lower() == "lokr"`) et
#     'lokr' est dans le Literal NetworkType (toolkit/config_modules.py L165). Aucune
#     famille exclue → PAS de whitelist. NB : use_old_lokr_format diffère selon l'arch
#     (nommage des poids seulement, pas le support) — krea2/flux2_klein = nouveau
#     format, zimage/sdxl/flux = ancien ; les deux s'entraînent et se chargent.
#   - train.ema_config={use_ema, ema_decay} : knob niveau TrainConfig, arch-agnostique
#     (config_modules.py L525-533 + EMAConfig L794-797, défaut ema_decay=0.999).
_NETWORK_TYPE_CHOICES = ('lora', 'lokr')
_LOKR_FACTOR_CHOICES = (4, 8, 16, 32)
_EMA_CHOICES = (0.99, 0.999)
# Krea Raw community-recipe controls. They are real ai-toolkit TrainConfig
# fields, but intentionally Krea-scoped in LDS: the report that motivated them
# concerns Krea 2 and we do not turn one anecdotal recipe into a global default.
_CONTENT_OR_STYLE_CHOICES = ('balanced', 'style', 'content')
_DIFFERENTIAL_GUIDANCE_SCALE_RANGE = (0.1, 10.0)
_LOSS_TYPE_CHOICES = ('mse',)
_QTYPE_CHOICES = ('qfloat8', 'float8', 'int8')
_SAVE_DTYPE_CHOICES = ('float16', 'bf16')
_OFFLOADING_PERCENT_RANGE = (0.0, 1.0)

# --- Memory-saving levers (quantisation + low-VRAM streaming) --------------------
# Community request (GitHub issue #14, bobba84): the recipes hard-coded quantize /
# quantize_te / low_vram, calibrated so a 12B DiT fits in 24 GB. On a card with MORE
# than the target, that calibration is a tax nobody asked for — quantisation costs
# precision and low_vram costs a lot of speed (it streams blocks CPU↔GPU).
#
# VÉRIFIÉ dans l'ai-toolkit installé : `quantize`, `quantize_te`, `qtype` et
# `low_vram` sont des champs de ModelConfig (toolkit/config_modules.py L658-662),
# arch-agnostiques, tous à False/'qfloat8' par défaut. Donc AUCUNE whitelist par
# famille : le levier existe partout, et sa valeur par défaut reste celle que la
# recette de CHAQUE famille a calibrée (table ci-dessous). Un utilisateur qui n'y
# touche pas obtient un job-config byte-for-byte identique à avant.
#
# `qtype` n'est PAS exposé : il ne s'applique que quand la quantisation est ON, et
# 'qfloat8' est déjà l'option la plus fidèle qu'ai-toolkit offre là (int8/uint8
# échangent de la qualité contre de la place). Un knob qui ne peut que dégrader
# n'est pas un choix, c'est un piège.
_MEMORY_SETTING_KEYS = ('quantize', 'quantize_te', 'low_vram')
# Human names, mirrored from the panel's MEMORY_LABELS (memorySavingAdvice.js) so
# a preflight sentence names the checkbox the user has to go and tick back.
_MEMORY_LABELS = {'quantize': 'Quantise base model',
                  'quantize_te': 'Quantise text encoder',
                  'low_vram': 'Low-VRAM streaming'}

# Ce que chaque famille émet quand l'utilisateur ne choisit rien. NE PAS TOUCHER :
# la majorité du parc est à 24 Go ou moins et c'est ce qui fait tenir l'entraînement.
_DEFAULT_MEMORY_SAVING = {
    'zimage':     {'quantize': True,  'quantize_te': True,  'low_vram': True},
    'krea':       {'quantize': True,  'quantize_te': True,  'low_vram': True},
    'flux':       {'quantize': True,  'quantize_te': True,  'low_vram': True},
    'flux2klein': {'quantize': True,  'quantize_te': True,  'low_vram': True},
    # 2B DiT — les defaults options.ts d'ai-toolkit sont déjà « pas de quantisation ».
    # Le levier reste offert dans l'AUTRE sens : une petite carte peut l'activer.
    'anima':      {'quantize': False, 'quantize_te': False, 'low_vram': False},
    'sdxl':       {'quantize': False, 'quantize_te': False, 'low_vram': False},
}

# VRAM (Gio) qu'il faut RAISONNABLEMENT pour entraîner cette famille sans
# quantisation ni streaming basse-VRAM. ESTIMÉ, pas mesuré carte par carte :
# poids du DiT en bf16 (2 octets × paramètres) + ~6 Gio d'activations/optimiseur
# LoRA/marge. Sert UNIQUEMENT à formuler un conseil ; ne bloque jamais rien.
#   zimage 6B → ~12 + 6      krea/flux 12B → ~24 + 6
#   flux2klein 9B → ~18 + 6  flux2klein 4B → ~8 + 6
_UNQUANTISED_VRAM_GB = {'zimage': 18, 'krea': 30, 'flux': 30,
                        'flux2klein': 24, 'flux2klein_4b': 14,
                        'anima': 10, 'sdxl': 10}


def _memory_saving_defaults(ds, family) -> dict:
    """Les trois défauts calibrés de la famille (copie — l'appelant les mute)."""
    return dict(_DEFAULT_MEMORY_SAVING.get(family or '',
                                           _DEFAULT_MEMORY_SAVING['zimage']))


def _memory_flag_eff(ds, key: str, default: bool) -> bool:
    """Valeur EFFECTIVE d'un levier mémoire : le booléen stocké s'il y en a un,
    sinon le défaut de la famille. Tri-état volontaire — `False` STOCKÉ doit
    survivre (c'est précisément la demande « disable »), donc on teste le type et
    jamais la véracité, contrairement à `dual_captions` où falsy = clé retirée."""
    v = _train_settings(ds).get(key)
    return v if isinstance(v, bool) else default


def _model_memory_block(ds, family) -> dict:
    """Fragment `model` à fusionner dans la recette de chaque famille.

    Forme d'émission choisie pour que le défaut reste BYTE-IDENTIQUE à l'existant :
      * `quantize` / `quantize_te` : toujours émis (les 6 recettes les émettaient
        déjà, dans les deux sens) ;
      * `low_vram` : émis SEULEMENT quand True — le défaut ModelConfig est False,
        donc omettre == False, et anima/sdxl (qui ne l'émettaient pas) ne gagnent
        pas une clé ;
      * `qtype` : émis seulement si au moins une quantisation est active — sans
        quantisation la clé ne veut rien dire.
    """
    d = _memory_saving_defaults(ds, family)
    q = _memory_flag_eff(ds, 'quantize', d['quantize'])
    qte = _memory_flag_eff(ds, 'quantize_te', d['quantize_te'])
    lv = _memory_flag_eff(ds, 'low_vram', d['low_vram'])
    s = _train_settings(ds)
    out = {'quantize': q, 'quantize_te': qte}
    if lv:
        out['low_vram'] = True
    if q or qte:
        out['qtype'] = (s.get('qtype') if s.get('qtype') in _QTYPE_CHOICES
                        else 'qfloat8')
    if s.get('qtype_te') in _QTYPE_CHOICES:
        out['qtype_te'] = s['qtype_te']
    if isinstance(s.get('layer_offloading'), bool):
        out['layer_offloading'] = s['layer_offloading']
    if s.get('layer_offloading') is True:
        for key in ('layer_offloading_transformer_percent',
                    'layer_offloading_text_encoder_percent'):
            value = s.get(key)
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and _OFFLOADING_PERCENT_RANGE[0] <= float(value)
                    <= _OFFLOADING_PERCENT_RANGE[1]):
                out[key] = float(value)
    return out


def _unquantised_vram_need(ds, family) -> int:
    """Estimation Gio pour tourner sans quantisation. FLUX.2 Klein a deux tailles
    de base (9B/4B) : le 4B tient beaucoup plus bas, le dire serait faux sinon."""
    if family == 'flux2klein' and not _flux2klein_is_9b(ds):
        return _UNQUANTISED_VRAM_GB['flux2klein_4b']
    return _UNQUANTISED_VRAM_GB.get(family or '', 24)


def _memory_saving_advice(ds, family) -> dict:
    """Conseil INDEXÉ SUR LA CARTE RÉELLE pour les leviers mémoire.

    `verdict` :
      * 'unknown'  — pas de nvidia-smi, GPU non-NVIDIA, machine sans carte, ou
                     famille non quantisée par défaut : texte générique côté UI ;
      * 'can_disable' — la VRAM détectée couvre le besoin estimé sans quantisation ;
      * 'keep_on'  — elle ne le couvre pas.

    Conseiller n'est PAS décider : rien ici n'écrit dans train_settings, rien ne
    bloque un lancement, et un échec de sonde (fail-open, mémoïsé 10 min côté
    run_environment) retombe simplement sur 'unknown'."""
    try:
        from . import run_environment
        vram = run_environment.local_vram_gb()
        gpu = (run_environment.gpu_info() or {}).get('name')
    except Exception:                                   # sonde absolument jamais fatale
        vram, gpu = None, None
    need = _unquantised_vram_need(ds, family)
    if not vram:
        verdict = 'unknown'
    elif vram + 0.5 >= need:        # 0.5 Gio : nvidia-smi rapporte 23.99 pour « 24 Go »
        verdict = 'can_disable'
    else:
        verdict = 'keep_on'
    return {'verdict': verdict, 'vram_gb': vram, 'gpu': gpu,
            'unquantised_vram_gb': need}


def memory_saving_risk(ds, family) -> dict | None:
    """Which of the family's CALIBRATED memory savers this run has switched off,
    or None when the recipe is intact.

    Why a warning and not a per-family memory (the `train_family_bases` treatment
    given to the base and the variant):

    * a base model is meaningless outside its family — a Z-Image merge is not a
      thing a Krea run can even load — so remembering it per family is the only
      way to state a truth. `quantize=False` is not like that: it is a statement
      about the CARD ("mine is big enough"), and the card does not change when
      the family does. What changes is whether it still suffices. Stashing the
      flag per family would answer a question nobody asked and would silently
      re-enable quantisation on the way back, which is the same silence in the
      other direction;
    * this is provenance-BLIND on purpose. Someone who unticks "Quantise base
      model" directly on Krea 2 with a 24 GB card is in exactly the same danger
      as someone who unticked it on Anima and switched family. A memory only ever
      catches the second one. One check catches both.

    Only the True→False direction is reported: switching a saver ON where the
    family default is off (anima/sdxl) costs precision and speed, never a run.
    """
    d = _memory_saving_defaults(ds, family)
    disabled = [k for k in _MEMORY_SETTING_KEYS
                if d[k] and not _memory_flag_eff(ds, k, d[k])]
    if not disabled:
        return None
    advice = _memory_saving_advice(ds, family)
    return {'family': family, 'disabled': disabled,
            'unquantised_vram_gb': advice['unquantised_vram_gb'],
            'vram_gb': advice['vram_gb'], 'gpu': advice['gpu'],
            'verdict': advice['verdict']}


# Settings whose meaning is bound to the FAMILY, not to the machine or the
# dataset — the ones that get the `train_family_bases` treatment (a per-family
# memory in `train_family_settings`, see face_dataset_service.set_train_type).
#
# `timestep_type` qualifies and the memory levers deliberately do not:
#   * 'weighted' is the canonical flowmatch schedule of FLUX.2 Klein and Anima,
#     'sigmoid' of Z-Image and FLUX.1, 'linear' of Krea 2. Picking one is picking
#     a family's recipe; carrying it over silently changes the LoRA that comes
#     out, with no error, no slowdown and nothing to observe afterwards. There is
#     no sentence a warning could add — "your LoRA is different" is not
#     actionable — so the honest fix is to stop carrying it;
#   * the memory levers are a statement about the card, and get a warning
#     instead. See memory_saving_risk for the full argument;
#   * `resolution` stays global too: 768 and 1024 mean the same thing on every
#     family, the value is a deliberate quality/VRAM trade-off a user restates
#     rarely, and the one dangerous combination (1024 on a 12B, small card) is
#     already a preflight row. Adding it here would mean silently re-raising
#     someone's 768 back to 768,1024 on a family switch — a NEW silent change.
_FAMILY_SCOPED_SETTING_KEYS = ('timestep_type',)

# Private provenance stored beside the explicit settings of an applied preset.
# It is deliberately not part of TRAIN_SETTING_KEYS and never appears in exports:
# it only prevents a family/kind/variant-scoped recipe from continuing to affect a
# dataset after that context changes.
_ACTIVE_PRESET_SCOPE_KEY = '_active_preset_scope'


class _TrainContextView:
    """Read-only dataset view with the exact selections for one action.

    A falsey value can be meaningful here: ``base_model=''`` explicitly selects
    the official base.  Sentinels therefore distinguish "not supplied" from an
    empty UI selection instead of falling through to the mutable dataset row.
    """

    def __init__(self, ds, family=None, variant=None, *,
                 base_model=_PERSISTED, mode=_PERSISTED,
                 train_slider=_PERSISTED):
        self._ds = ds
        self._family = family
        self._variant = variant
        self._base_model = base_model
        self._mode = mode
        self._train_slider = train_slider

    @property
    def train_type(self):
        return self._family if self._family is not None else self._ds.train_type

    @property
    def train_variant(self):
        return self._variant if self._variant is not None else self._ds.train_variant

    @property
    def train_base_model(self):
        return (self._ds.train_base_model if self._base_model is _PERSISTED
                else self._base_model)

    @property
    def training_mode(self):
        return (getattr(self._ds, 'training_mode', None)
                if self._mode is _PERSISTED else self._mode)

    @property
    def train_slider(self):
        return (getattr(self._ds, 'train_slider', None)
                if self._train_slider is _PERSISTED else self._train_slider)

    def __getattr__(self, name):
        return getattr(self._ds, name)


def _train_context_view(ds, family=None, variant=None, *,
                        base_model=_PERSISTED, training_mode=_PERSISTED,
                        train_slider=_PERSISTED):
    if ds is None:
        return None
    fam = _train_type(ds, family)
    selected = str(
        variant or getattr(ds, 'train_variant', None)
        or _default_variant_for(fam)).strip().lower()
    return _TrainContextView(
        ds, fam, selected, base_model=base_model, mode=training_mode,
        train_slider=train_slider)


def _preset_scope_matches(ds, scope) -> bool:
    if not isinstance(scope, dict):
        return True
    family = str(getattr(ds, 'train_type', None) or 'zimage').strip().lower()
    kind = str(getattr(ds, 'kind', None) or 'character').strip().lower()
    variant = str(
        getattr(ds, 'train_variant', None)
        or _default_variant_for(family)).strip().lower()
    if scope.get('train_type') and scope.get('train_type') != family:
        return False
    if scope.get('dataset_kind') and scope.get('dataset_kind') != kind:
        return False
    variants = scope.get('variants')
    if isinstance(variants, list) and variants and variant not in variants:
        return False
    return True


def clear_active_preset_settings(settings: dict) -> dict:
    """Drop an applied preset and every setting that replacement owned."""
    scope = settings.pop(_ACTIVE_PRESET_SCOPE_KEY, None)
    if isinstance(scope, dict):
        for key in scope.get('keys') or []:
            if key in TRAIN_SETTING_KEYS:
                settings.pop(key, None)
    return settings


def _train_settings(ds) -> dict:
    """Parse le blob JSON `train_settings` en dict (jamais lève ; {} si absent/cassé)."""
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(d, dict):
        return {}
    scope = d.get(_ACTIVE_PRESET_SCOPE_KEY)
    if isinstance(scope, dict) and not _preset_scope_matches(ds, scope):
        # Return an inert VIEW while preserving the stored provenance. If the
        # user comes back to the scoped context the recipe is restored; outside
        # it, not one of its owned settings reaches preflight/config/step policy.
        d = dict(d)
        for key in scope.get('keys') or []:
            if key in TRAIN_SETTING_KEYS:
                d.pop(key, None)
    return d


def _klein_style(ds, family) -> bool:
    """FLUX.2 Klein STYLE LoRA — la SEULE combinaison famille×kind qui s'écarte du
    schéma linear-only / alpha=rank : réseau 128/64 + Conv2d 64/32 (ratio 4:2:2:1).
    Le sweep Herbst (64 runs) et l'exemple officiel BFL convergent dessus ; les
    autres kinds Klein (character/concept/slider) gardent le défaut linear-only."""
    return (family or '') == 'flux2klein' and fds.is_style(ds)


def _default_rank_for(ds, family) -> int:
    """Family default rank — except in slider mode, where public concept sliders
    ship at rank 4-8 (a slider is a low-rank direction, not an identity), and
    FLUX.2 Klein style, whose researched recipe is a 128-dim linear + Conv2d LoRA."""
    if slider_mode_enabled(ds):
        return _SLIDER_DEFAULT_RANK
    if _klein_style(ds, family):
        return _KLEIN_STYLE_RANK
    return _DEFAULT_RANK.get(family, 32)


def _lora_rank(ds, family) -> int:
    r = _train_settings(ds).get('rank')
    return r if r in _RANK_CHOICES else _default_rank_for(ds, family)


def _lora_alpha(rank, family, ds=None) -> int:
    """Alpha dérivé du rank. Défaut alpha = rank (échelle 1.0) pour zimage/krea/flux.
    Trois écarts délibérés, tous sourcés : SDXL = rank/2 (« demi-force », recherche) ;
    FLUX.2 Klein style = rank/2 (dims 128/64 du sweep Herbst + exemple BFL) ; slider =
    alpha 4 fixe (notebook Ostris « bigger is not always better, especially for
    sliders » — rank 8 / alpha 4, échelle 0.5)."""
    if ds is not None and slider_mode_enabled(ds):
        return _SLIDER_DEFAULT_ALPHA
    if family == 'sdxl' or _klein_style(ds, family):
        return max(1, rank // 2)
    return rank


def _numeric_choice(value, choices):
    """A persisted numeric choice, accepting only real integer values.

    ``bool`` is an ``int`` subclass and legacy JSON can also contain floats that
    compare equal to an integer choice (for example ``1.0 == 1``).  Neither is a
    valid stored configuration value, so keep the effective readers aligned with
    API ingress and only propagate exact integers.
    """
    return value if type(value) is int and value in choices else None


def _lora_alpha_eff(ds, rank, family) -> int:
    """Alpha EFFECTIF : un `alpha` explicite dans train_settings prime sur le dérivé.
    Découpler alpha du rank = levier de LR « doux » (échelle effective = alpha/rank).
    En mode slider, l'utilisateur peut ainsi remettre alpha 8 (défaut 4) via ce knob."""
    a = _numeric_choice(_train_settings(ds).get('alpha'), _ALPHA_CHOICES)
    return a if a is not None else _lora_alpha(rank, family, ds)


def _network_type_eff(ds) -> str:
    """'lora' (défaut) ou 'lokr' — validé contre l'enum ai-toolkit ; inconnu → 'lora'.
    LoKr est arch-générique (LokrModule sur toutes les familles), aucune garde
    par famille nécessaire."""
    t = _train_settings(ds).get('network_type')
    return t if t in _NETWORK_TYPE_CHOICES else 'lora'


def _lokr_factor_eff(ds) -> int | None:
    """Explicit LoKr decomposition factor, or None for ai-toolkit's auto choice.

    `lokr_factor=-1` is ai-toolkit's auto mode. LDS keeps the user-facing setting
    absent in that case so existing LoKr runs retain their prior behaviour; a
    shipped recipe can opt into a known factor (notably Krea Raw's factor 16).
    """
    v = _train_settings(ds).get('lokr_factor')
    return v if isinstance(v, int) and not isinstance(v, bool) and v in _LOKR_FACTOR_CHOICES else None


def _lokr_full_rank_eff(ds) -> bool:
    """The explicitly recorded LoKr full-rank mode, defaulting to the LDS-safe False.

    This is deliberately not an editable advanced setting.  It only lets a cloud
    continuation replay a provenance snapshot made by an older/other LDS run that
    explicitly used full-rank LoKr; otherwise forcing False would silently change
    the checkpoint topology while claiming to resume it.
    """
    return _train_settings(ds).get('lokr_full_rank') is True


def _network_block(ds, rank, family) -> dict:
    """Bloc `network` LoRA/LoKr partagé par les 5 job-configs : type + rank + alpha
    (override-aware) + dropout optionnel (régularisateur anti-overfit, clé omise quand
    off). A normal LDS LoKr run pins `lokr_full_rank=False` because ai-toolkit has
    changed its implicit default; an explicit value from a frozen continuation
    snapshot is replayed verbatim so the weights' topology is not changed.
    `lokr_factor` remains auto unless explicitly chosen."""
    network_type = _network_type_eff(ds)
    net = {'type': network_type, 'linear': rank,
           'linear_alpha': _lora_alpha_eff(ds, rank, family)}
    if network_type == 'lokr':
        net['lokr_full_rank'] = _lokr_full_rank_eff(ds)
        factor = _lokr_factor_eff(ds)
        if factor is not None:
            net['lokr_factor'] = factor
    s = _train_settings(ds)
    explicit_conv = _numeric_choice(s.get('conv'), _RANK_CHOICES)
    if explicit_conv is not None and net['type'] == 'lora':
        net['conv'] = explicit_conv
        conv_alpha = _numeric_choice(s.get('conv_alpha'), _ALPHA_CHOICES)
        net['conv_alpha'] = conv_alpha if conv_alpha is not None else explicit_conv
    elif _klein_style(ds, family) and net['type'] == 'lora':
        # FLUX.2 Klein STYLE : ajoute un LoRA Conv2d aux moitiés du linear (conv_alpha
        # au quart) → dims 128/64/64/32 au rank par défaut. Combo dominant du sweep
        # Herbst (64 runs, fév. 2026) et de l'exemple de training officiel BFL. Clés
        # ai-toolkit VÉRIFIÉES : NetworkConfig lit conv/conv_alpha au même niveau que
        # linear/linear_alpha (toolkit/config_modules.py). LoKr garde le linear-only.
        net['conv'] = max(1, rank // 2)
        net['conv_alpha'] = max(1, rank // 4)
    d = _train_settings(ds).get('dropout')
    if isinstance(d, (int, float)) and d in _DROPOUT_CHOICES:
        net['dropout'] = d
    return net


def _save_dtype_eff(ds) -> str:
    value = _train_settings(ds).get('save_dtype')
    return value if value in _SAVE_DTYPE_CHOICES else 'float16'


def _dataset_cache_text_embeddings(ds, default=None) -> dict:
    """Dataset-level cache override; absent keeps each family's old emission."""
    value = _train_settings(ds).get('cache_text_embeddings')
    if isinstance(value, bool):
        return {'cache_text_embeddings': value}
    if isinstance(default, bool):
        return {'cache_text_embeddings': default}
    return {}


# These families cache text embeddings in their untouched LDS recipe.  An
# explicit per-preset override may now change that fact, so callers that care
# about the effective capability must use ``_cache_text_embeddings_eff`` rather
# than checking this tuple directly.
DUAL_CAPTION_UNSUPPORTED_FAMILIES = ('krea', 'anima')


def _cache_text_embeddings_eff(ds, family) -> bool:
    value = _train_settings(ds).get('cache_text_embeddings')
    if isinstance(value, bool):
        return value
    return (family or '').lower() in DUAL_CAPTION_UNSUPPORTED_FAMILIES


def _train_serializer_fields(ds) -> dict:
    """Optional real TrainConfig fields, omitted for untouched datasets."""
    s = _train_settings(ds)
    out = {}
    decay = s.get('weight_decay')
    if (isinstance(decay, (int, float)) and not isinstance(decay, bool)
            and 0 <= float(decay) <= 1):
        out['optimizer_params'] = {'weight_decay': float(decay)}
    if s.get('loss_type') in _LOSS_TYPE_CHOICES:
        out['loss_type'] = s['loss_type']
    return out


def _timestep_type_eff(ds, default: str) -> str:
    """Pondération des timesteps : override la valeur family-default si l'utilisateur en
    a choisi une valide (gardé à l'enum ai-toolkit ; inconnu → le défaut)."""
    t = _train_settings(ds).get('timestep_type')
    return t if t in _TIMESTEP_TYPE_CHOICES else default


def _optimizer_eff(ds) -> str:
    o = _train_settings(ds).get('optimizer')
    return o if o in _OPTIMIZER_CHOICES else 'adamw8bit'


_DEFAULT_LR = 1e-4                         # base LR for every non-adaptive optimizer
_RESUME_LR_FACTORS = (0.5, 0.1)           # ▶ Continue "half (polish)" / "tenth (gentle finish)"


def _lr_from_settings(s: dict) -> float:
    """Effective LR for a settings dict (live train_settings OR a frozen run
    snapshot). Prodigy drives the LR itself → the lr≈1.0 convention always wins,
    even over a stale stored value. Otherwise an explicit `learning_rate` (set on
    a resume via the LR factor knob) overrides the family-fixed 1e-4 default."""
    opt = s.get('optimizer')
    opt = opt if opt in _OPTIMIZER_CHOICES else 'adamw8bit'
    if opt.startswith('prodigy'):
        return 1.0
    lr = s.get('learning_rate')
    return float(lr) if isinstance(lr, (int, float)) and lr > 0 else _DEFAULT_LR


def _lr_eff(ds) -> float:
    return _lr_from_settings(_train_settings(ds))


def resolve_resume_lr(settings: dict, lr_factor) -> float | None:
    """Turn a ▶ Continue LR *factor* into the absolute `learning_rate` to persist for
    the continuation, given the run's effective ``settings`` (live dict or snapshot).
    ``None`` / 1 = keep current (no change). A real factor (½/⅒) scales the run's
    current effective LR — a 1e-4 run continues at 5e-5 or 1e-5. Refused loudly on a
    Prodigy run: it adapts the LR itself (lr=1), so there is no base rate to scale and
    a factor would be meaningless — same "regime-bound settings are refused, never
    silently swallowed" contract as the rest of validate_resume_overrides."""
    if lr_factor in (None, 1, 1.0):
        return None
    if lr_factor not in _RESUME_LR_FACTORS:
        raise ValueError(f'lr_factor must be one of {_RESUME_LR_FACTORS} (or 1 to keep current)')
    opt = settings.get('optimizer')
    opt = opt if opt in _OPTIMIZER_CHOICES else 'adamw8bit'
    if opt.startswith('prodigy'):
        raise ValueError(
            'the learning-rate factor cannot apply to a Prodigy run — Prodigy adapts '
            'the LR itself (lr=1), so there is no base rate to halve or tenth')
    return _lr_from_settings(settings) * float(lr_factor)


def _grad_accum(ds) -> int:
    g = _numeric_choice(_train_settings(ds).get('grad_accum'), _GRAD_ACCUM_CHOICES)
    return g if g is not None else 1


def _lr_sched_fields(ds) -> dict:
    """{} par défaut (= 'constant' d'ai-toolkit). Sinon {lr_scheduler [+ lr_scheduler_params
    {num_warmup_steps} pour constant_with_warmup]} à fusionner dans le bloc train. Le warmup
    n'est câblé QUE pour constant_with_warmup : les schedulers torch (cosine/linear/constant)
    n'acceptent pas num_warmup_steps → le passer les ferait planter (cf. toolkit/scheduler.py)."""
    s = _train_settings(ds).get('lr_scheduler')
    if s not in _LR_SCHEDULER_CHOICES or s == 'constant':
        return {}
    out = {'lr_scheduler': s}
    if s == 'constant_with_warmup':
        w = _train_settings(ds).get('warmup')
        out['lr_scheduler_params'] = {'num_warmup_steps': w if w in _WARMUP_CHOICES else 100}
    return out


def _ema_eff(ds):
    """Décroissance EMA choisie (0.99/0.999) ou None (= off). Inconnu → None."""
    v = _train_settings(ds).get('ema')
    return v if v in _EMA_CHOICES else None


def _ema_fields(ds) -> dict:
    """{} par défaut (= ai-toolkit use_ema=False) → à fusionner dans le bloc `train`.
    Sinon {ema_config: {use_ema, ema_decay}} : moyenne mobile exponentielle des poids,
    checkpoints plus lisses (clés VÉRIFIÉES config_modules.py EMAConfig L794-797)."""
    v = _ema_eff(ds)
    if v is None:
        return {}
    return {'ema_config': {'use_ema': True, 'ema_decay': v}}


def _content_or_style_eff(ds) -> str:
    """Krea's ai-toolkit `train.content_or_style`, defaulting to balanced."""
    v = _train_settings(ds).get('content_or_style')
    return v if v in _CONTENT_OR_STYLE_CHOICES else 'balanced'


def _differential_guidance_enabled(ds) -> bool:
    return _train_settings(ds).get('do_differential_guidance') is True


def _differential_guidance_scale_eff(ds) -> float:
    """Validated Differential Guidance multiplier, defaulting to ai-toolkit's 3."""
    v = _train_settings(ds).get('differential_guidance_scale')
    lo, hi = _DIFFERENTIAL_GUIDANCE_SCALE_RANGE
    if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= float(v) <= hi:
        return float(v)
    return 3.0


def _content_or_style_fields(ds) -> dict:
    """Explicit generic TrainConfig content/style balance, omitted by default."""
    value = _train_settings(ds).get('content_or_style')
    return ({'content_or_style': value}
            if value in _CONTENT_OR_STYLE_CHOICES else {})


def _krea_recipe_fields(ds) -> dict:
    """Optional Krea 2 community-recipe controls for ai-toolkit's TrainConfig.

    LDS emits no extra keys for an untouched dataset, preserving the existing
    ai-toolkit defaults. A preset that pins Balanced or Differential Guidance
    emits exactly what it announced, so the run config and provenance stay
    reproducible.
    """
    out = _content_or_style_fields(ds)
    if _differential_guidance_enabled(ds):
        out['do_differential_guidance'] = True
        out['differential_guidance_scale'] = _differential_guidance_scale_eff(ds)
    return out


# --- Slider LoRA mode (Beta) -----------------------------------------------------
# Backed by ai-toolkit's MODERN `concept_slider` extension (extensions_built_in/
# concept_slider/ConceptSliderTrainer.py, uid 'concept_slider', extends
# DiffusionTrainer) — NOT the legacy `type: slider`/TrainSliderProcess path.
# The trainer learns ONE bipolar LoRA (multiplier +1/−1) from a positive/negative
# prompt pair (LECO-style guided loss); the dataset's kept images are only a
# DENOISING SUBSTRATE (ConceptSliderTrainer.get_guided_loss reads noisy_latents
# from the normal training batch; captions are encoded but IGNORED by the loss).
# A dataset therefore stays REQUIRED — that's why slider is a per-dataset MODE,
# not a dataset kind. Schema of the emitted `slider:` block = the exact kwargs of
# ConceptSliderTrainerConfig (guidance_strength, anchor_strength, positive_prompt,
# negative_prompt, target_class, anchor_class).
_SLIDER_DEFAULT_RANK = 8            # public concept sliders ship at rank 4-8
_SLIDER_DEFAULT_ALPHA = 4           # Ostris slider notebook: rank 8 / alpha 4 (scale 0.5),
                                    # "bigger is not always better, especially for sliders"
_SLIDER_DEFAULT_GUIDANCE = 3.0      # ConceptSliderTrainerConfig default
_SLIDER_DEFAULT_ANCHOR_STRENGTH = 1.0
_SLIDER_GUIDANCE_RANGE = (1.0, 10.0)
_SLIDER_ANCHOR_STRENGTH_RANGE = (0.0, 10.0)
_SLIDER_PROMPT_MAX_LEN = 500
# Fixed step target: a slider direction is prompt-defined, so dataset size does
# not drive convergence (images are substrate only). Ostris' own slider recipe
# uses 500 steps and "rarely goes over 1000"; the modern trainer trains both
# polarities per step, so 1000 is a safe honest default with early checkpoints.
SLIDER_DEFAULT_STEPS = 1000
# Substrate floor: the batch only needs a few varied images to sample latents
# from. 4 = hard floor (variety of noising targets), 12+ recommended.
TRAIN_MIN_IMAGES_SLIDER = (4, 12)


def _slider_settings(ds) -> dict:
    """Parse the `train_slider` JSON blob (never raises; {} when absent/broken)."""
    raw = getattr(ds, 'train_slider', None)
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def slider_mode_enabled(ds) -> bool:
    """True when this dataset trains in Slider LoRA mode (Beta)."""
    return bool(_slider_settings(ds).get('enabled'))


def effective_slider_settings(ds) -> dict:
    """Slider mode state + resolved knobs for the TrainingPanel (and tests)."""
    s = _slider_settings(ds)
    return {
        'enabled': bool(s.get('enabled')),
        'positive': s.get('positive') or '',
        'negative': s.get('negative') or '',
        'target_class': s.get('target_class') or '',
        'anchor': s.get('anchor') or '',
        'guidance': _slider_guidance(ds),
        'anchor_strength': _slider_anchor_strength(ds),
        'default_rank': _SLIDER_DEFAULT_RANK,
        'default_alpha': _SLIDER_DEFAULT_ALPHA,   # Ostris slider recipe (scale 0.5)
        'default_steps': SLIDER_DEFAULT_STEPS,
        'min_images': TRAIN_MIN_IMAGES_SLIDER[0],
    }


def _slider_guidance(ds) -> float:
    try:
        v = float(_slider_settings(ds).get('guidance'))
    except (TypeError, ValueError):
        return _SLIDER_DEFAULT_GUIDANCE
    lo, hi = _SLIDER_GUIDANCE_RANGE
    return v if lo <= v <= hi else _SLIDER_DEFAULT_GUIDANCE


def _slider_anchor_strength(ds) -> float:
    try:
        v = float(_slider_settings(ds).get('anchor_strength'))
    except (TypeError, ValueError):
        return _SLIDER_DEFAULT_ANCHOR_STRENGTH
    lo, hi = _SLIDER_ANCHOR_STRENGTH_RANGE
    return v if lo <= v <= hi else _SLIDER_DEFAULT_ANCHOR_STRENGTH


def _clean_slider_prompt(value, field) -> str:
    txt = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(txt) > _SLIDER_PROMPT_MAX_LEN:
        raise ValueError(f'{field} is too long (max {_SLIDER_PROMPT_MAX_LEN} chars)')
    return txt


def update_slider_settings(user_id, dataset_id, patch: dict) -> dict:
    """Validate + merge a slider-mode patch {enabled?, positive?, negative?,
    target_class?, anchor?, guidance?, anchor_strength?} into `train_slider`.
    Free-text fields are whitespace-normalized and length-capped; numeric knobs
    are range-validated (invalid → 400, never a silent clamp). Returns the
    effective slider settings."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur = _slider_settings(ds)
    if 'enabled' in patch:
        if patch['enabled']:
            if _is_full_transformer(ds):
                raise ValueError(
                    'Slider mode cannot be enabled during full_transformer '
                    'training. Switch the training mode to LoRA first.')
            cur['enabled'] = True
        else:
            cur.pop('enabled', None)
    for key, label in (('positive', 'positive prompt'), ('negative', 'negative prompt'),
                       ('target_class', 'target class'), ('anchor', 'anchor prompt')):
        if key in patch:
            txt = _clean_slider_prompt(patch[key], label)
            if txt:
                cur[key] = txt
            else:
                cur.pop(key, None)
    for key, rng, default in (
            ('guidance', _SLIDER_GUIDANCE_RANGE, _SLIDER_DEFAULT_GUIDANCE),
            ('anchor_strength', _SLIDER_ANCHOR_STRENGTH_RANGE,
             _SLIDER_DEFAULT_ANCHOR_STRENGTH)):
        if key in patch:
            v = patch[key]
            if v in (None, '', 'auto'):
                cur.pop(key, None)
                continue
            try:
                fv = round(float(v), 2)
            except (TypeError, ValueError):
                raise ValueError(f'{key} must be a number in [{rng[0]}, {rng[1]}]')
            if not rng[0] <= fv <= rng[1]:
                raise ValueError(f'{key} out of range [{rng[0]}, {rng[1]}]: {fv}')
            if fv == default:
                cur.pop(key, None)
            else:
                cur[key] = fv
    ds.train_slider = json.dumps(cur) if cur else None
    fds.db.session.commit()
    return effective_slider_settings(ds)


def _aitoolkit_supports_concept_slider() -> bool:
    """Does the installed ai-toolkit ship the `concept_slider` extension? Same
    stakes as _aitoolkit_supports_krea (read its comment): an unknown process
    type crashes the job at boot with a confusing traceback. We require the
    exact extension uid string emitted by _apply_slider_overrides. Fresh read:
    a maintainer `git pull` flips the detection without a restart."""
    root = cfg.aitoolkit_path('dir')
    if not root:
        return False
    ext_root = root / 'extensions_built_in'
    if not ext_root.is_dir():
        return False
    pat = re.compile(r'uid\s*=\s*[\'"]concept_slider[\'"]')
    for dp, _dn, files in os.walk(str(ext_root)):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            try:
                with open(os.path.join(dp, fn), encoding='utf-8', errors='ignore') as fh:
                    if pat.search(fh.read()):
                        return True
            except OSError:
                continue
    return False


def _train_res(ds) -> list:
    return _RES_CHOICES.get(_train_settings(ds).get('resolution'), [768, 1024])


def _resolution_is_explicit(ds) -> bool:
    """Did the user PICK a resolution (vs. riding the family default)? A stored
    value outside the valid set counts as "not chosen"."""
    return _train_settings(ds).get('resolution') in _RES_CHOICES


def _effective_resolution(ds) -> list:
    """The resolution list this run will ACTUALLY emit.

    Slider mode defaults to 768 only: the concept_slider loss runs several
    prediction passes per step, so its VRAM peak sits far above a normal run —
    multi-scale 768+1024 really OOMs on 24 GB (the first reported slider run died
    with 'bad allocation' at step 21 when Discord grabbed some VRAM). This is a
    DEFAULT, not a clamp: an explicit user resolution is always obeyed."""
    if slider_mode_enabled(ds) and not _resolution_is_explicit(ds):
        return [768]
    return _train_res(ds)


def _save_every(ds) -> int:
    v = _train_settings(ds).get('save_every')
    return v if v in _SAVE_CHOICES else 250


# Combien de saves intermédiaires ai-toolkit CONSERVE pendant le run (local et
# cloud) : au-delà, il supprime les plus anciens lui-même. L'historique (10)
# laissait s'accumuler ~10 Go de checkpoints par run Krea.
_MAX_SAVES_CHOICES = (2, 3, 4, 6, 10)


def _max_step_saves(ds) -> int:
    v = _train_settings(ds).get('max_step_saves')
    return v if v in _MAX_SAVES_CHOICES else 4


# --- Prompts de preview (sample) -----------------------------------------------
# ai-toolkit génère une image par prompt tous les `sample_every` steps pendant le
# run (dossier .../samples), pour voir le LoRA converger. Les défauts historiques
# décrivaient un VISAGE (« close-up portrait, headshot… ») — hors sujet pour un
# dataset « concept ». D'où un défaut distinct selon le kind, et un override total
# par l'utilisateur (Advanced options → Preview prompts).
_SAMPLE_EVERY_CHOICES = (100, 250, 500, 1000)
_MAX_SAMPLE_PROMPTS = 8   # 1 image générée / prompt / palier → borne le coût des previews

_DEFAULT_SAMPLE_PROMPTS_CHARACTER = [
    '{trigger}, close-up portrait, neutral expression',
    '{trigger}, headshot, soft studio light',
    '{trigger}, full body, walking outdoors, smiling',
    '{trigger}, sitting in a cafe, casual outfit',
]
# Un concept n'est pas un visage : on l'exerce seul sous quelques cadrages neutres
# (le vocabulaire « portrait / headshot » tirerait un LoRA non-visage hors sujet).
_DEFAULT_SAMPLE_PROMPTS_CONCEPT = [
    '{trigger}',
    '{trigger}, high detail, sharp focus',
    '{trigger}, wide shot',
    '{trigger}, cinematic lighting',
]


# Un style n'a PAS de trigger : le LoRA teinte toute image dès qu'il est chargé.
# Les previews sont donc des scènes génériques variées — si le style s'y voit,
# l'entraînement prend ; le vocabulaire portrait/headshot tirerait hors sujet.
_DEFAULT_SAMPLE_PROMPTS_STYLE = [
    'a woman reading in a sunlit cafe',
    'a city street at night, rain',
    'a mountain landscape, wide shot',
    'a still life of fruit on a wooden table',
]


def _default_sample_prompts(ds) -> list:
    if fds.is_style(ds):
        return list(_DEFAULT_SAMPLE_PROMPTS_STYLE)
    return list(_DEFAULT_SAMPLE_PROMPTS_CONCEPT if fds.is_concept(ds)
                else _DEFAULT_SAMPLE_PROMPTS_CHARACTER)


def _inject_trigger(prompt: str, trigger: str) -> str:
    """Une preview DOIT solliciter le LoRA : si la ligne ne mentionne pas déjà le
    trigger (insensible à la casse), on le préfixe — sinon l'image teste le modèle
    de base, pas l'entraînement en cours."""
    p = (prompt or '').strip()
    if not trigger:
        return p
    if not p:
        return trigger
    return p if trigger.lower() in p.lower() else f'{trigger}, {p}'


def _strip_style_trigger(prompt: str, trigger: str) -> str:
    """Remove the placeholder or a legacy *leading* internal run id.

    Style is an always-on training mode: ``trigger_word`` is retained solely for
    collision-free run/file names. A legacy custom preview may nevertheless still
    begin ``trigger, content``. Never remove an ordinary content word in the
    middle (e.g. trigger ``ink`` in ``an ink illustration``).
    """
    value = (prompt or '').replace('{trigger}', ' ')
    if trigger:
        if value.strip(' .!?:;,').strip().casefold() == trigger.casefold():
            value = ''
        else:
            value = re.sub(
                rf'^\s*{re.escape(trigger)}\s*[,;:.!?]\s*', '', value,
                count=1, flags=re.IGNORECASE)
    value = re.sub(r'\s*,\s*,+', ', ', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip(' ,')


def _resolved_default_sample_prompts(ds, trigger) -> list:
    """Défauts (selon le kind) avec `{trigger}` substitué — pour l'aperçu UI."""
    if fds.is_style(ds):   # style : pas de trigger, jamais injecté
        return list(_default_sample_prompts(ds))
    return [_inject_trigger(l.replace('{trigger}', trigger), trigger)
            for l in _default_sample_prompts(ds)]


def _sample_prompts(ds, trigger) -> list:
    """Prompts de preview effectifs : liste custom de train_settings si présente,
    sinon défaut selon le kind. `{trigger}` (placeholder explicite) ET le trigger en
    clair sont gérés ; le trigger est auto-préfixé s'il manque. Toujours ≥1 prompt,
    ≤_MAX_SAMPLE_PROMPTS (borne le nombre d'images générées par palier)."""
    raw = _train_settings(ds).get('sample_prompts')
    tmpl = raw if (isinstance(raw, list)
                   and any(isinstance(x, str) and x.strip() for x in raw)) \
        else _default_sample_prompts(ds)
    # STYLE : aucun trigger — le LoRA teinte tout, une preview générique le
    # sollicite déjà. Le token persisté ne sert qu'à nommer/isoler le run.
    style = fds.is_style(ds)
    out = []
    for line in tmpl:
        if not isinstance(line, str) or not line.strip():
            continue
        if style:
            resolved = _strip_style_trigger(line, trigger)
            raw_trigger = (getattr(ds, 'trigger_word', None) or '').strip()
            if raw_trigger and raw_trigger.casefold() != trigger.casefold():
                resolved = _strip_style_trigger(resolved, raw_trigger)
            if not resolved:
                continue
            out.append(resolved)
        else:
            resolved = line.replace('{trigger}', trigger).strip(', ') or line
            out.append(_inject_trigger(resolved, trigger))
        if len(out) >= _MAX_SAMPLE_PROMPTS:
            break
    if out:
        return out
    return [_default_sample_prompts(ds)[0]] if style else [_inject_trigger('', trigger)]


def _sample_every(ds) -> int:
    v = _train_settings(ds).get('sample_every')
    return v if v in _SAMPLE_EVERY_CHOICES else 250


def person_masking_enabled(ds) -> bool:
    """Person masking resolved for a run: the dataset's stored opt-in (default ON),
    minus the two server guards that already force it off at export time — a
    concept/style set (a person mask erases what is being taught) and slider mode
    (the guided slider loss never reads batch.mask_tensor)."""
    return fds.person_masking_enabled(ds) and not slider_mode_enabled(ds)


def resolve_masked(ds, requested=None) -> bool:
    """THE resolution of `masked` for a launch — one implementation, every lane.

    An EXPLICIT boolean on the request still wins, and that is deliberate: the
    canvas ▶ Continue replays the SOURCE run's own frozen flag, and a cloud
    retry/continue replays the stamped run params. Those are per-RUN facts and
    must not be re-read from a dataset that has since been edited. Absent (None)
    — every fresh launch from the panel, the queue and the scheduler — resolves
    to the dataset's persisted setting instead of the old hardcoded True."""
    if isinstance(requested, bool):
        return requested
    return person_masking_enabled(ds)


def resolve_masked_for(user_id, dataset_id, requested=None) -> bool:
    """`resolve_masked` for the routes, which hold ids rather than the ORM row.
    An unknown dataset falls back to the historical default (the launch below
    raises 'dataset not found' on its own — this must not raise first)."""
    if isinstance(requested, bool):
        return requested
    return person_masking_enabled(fds.get_dataset(user_id, dataset_id))


def launch_settings_snapshot(ds, family=None, masked=None) -> dict:
    """Les réglages EFFECTIFS envoyés à ai-toolkit pour CE lancement — défauts
    résolus, pas les choix stockés. Stampé dans le registre de provenance
    (TrainingRunRecord.settings) par chaque launch local et cloud ; la page
    Runs l'affiche par run (« quels réglages sont partis ? »). Compact : les
    leviers experts n'apparaissent que s'ils dévient du défaut."""
    fam = family or _train_type(ds)
    mode = training_mode(ds)
    if mode == 'full_transformer':
        # Dense Krea is a different artifact and a different optimiser recipe,
        # not a LoRA with a few toggles changed.  Keep its provenance free of
        # rank/alpha/network keys: those fields would claim adapter geometry that
        # the emitted config intentionally does not contain.
        dense_ds = (_train_context_view(
            ds, fam, getattr(ds, 'train_variant', None),
            base_model=getattr(ds, 'train_base_model', None),
            training_mode=mode) if fam != _train_type(ds) else ds)
        _assert_full_transformer_recipe(dense_ds)
        return {
            'training_mode': 'full_transformer',
            'artifact_kind': 'full_transformer',
            'model_arch': 'krea2',
            # Read from the SAME resolver as the emitted config. It used to be
            # the Raw constant, which was harmless only for as long as a dense
            # run could not be anything else — the Runs page would otherwise
            # claim "Krea 2 Raw" over a Turbo or custom-base run.
            'effective_base': _krea_name_or_path(dense_ds),
            'vae_path': FULL_TRANSFORMER_VAE,
            **({'base_weights': str(getattr(dense_ds, 'train_base_model', None))}
               if _is_custom_weights(getattr(dense_ds, 'train_base_model', None))
               else {}),
            'resolution': [_dense_resolution(dense_ds)],
            'caption_dropout_rate': 0.05,
            'cache_latents_to_disk': True,
            'cache_text_embeddings': True,
            'save_every': _dense_save_every(dense_ds),
            'max_step_saves': _dense_max_step_saves(dense_ds),
            'save_dtype': 'bf16',
            'batch_size': 1,
            'grad_accum': _dense_grad_accum(dense_ds),
            'train_unet': True,
            'train_text_encoder': False,
            'unload_text_encoder': True,
            'gradient_checkpointing': True,
            'noise_scheduler': 'flowmatch',
            'timestep_type': _dense_timestep_type(dense_ds),
            'optimizer': 'adafactor',
            'lr': _dense_lr(dense_ds),
            # Stamped EFFECTIVE, like every other value in this block: 'constant'
            # is what the run uses even though the emitted config leaves the key
            # out. Provenance answers "what did this run train with?", not "what
            # bytes were in the file".
            'lr_scheduler': _dense_lr_schedule(dense_ds),
            **({'warmup': _dense_warmup(dense_ds)}
               if _dense_lr_schedule(dense_ds) == 'constant_with_warmup' else {}),
            'dtype': 'bf16',
            'quantize': False,
            'quantize_te': False,
            'low_vram': False,
            'sample_every': _dense_save_every(dense_ds),
            'guidance_scale': FULL_TRANSFORMER_SAMPLE_GUIDANCE,
            'sample_steps': FULL_TRANSFORMER_SAMPLE_STEPS,
            'sample_prompts': _sample_prompts(dense_ds, _safe_trigger(dense_ds)),
            'fp8_export': dense_fp8_export_enabled(dense_ds),
            'keep_bf16_master': dense_keep_bf16_master(dense_ds),
            'trigger': _safe_trigger(dense_ds),
            'masked': (bool(masked) if isinstance(masked, bool)
                       else person_masking_enabled(dense_ds)),
        }
    rank = _lora_rank(ds, fam)
    zrecipe = (zimage_training_recipe(getattr(ds, 'train_variant', None),
                                      getattr(ds, 'train_base_model', None))
               if fam == 'zimage' else None)
    snap = {
        'training_mode': 'lora',
        'artifact_kind': 'lora',
        'rank': rank,
        'alpha': _lora_alpha_eff(ds, rank, fam),
        # The resolution ACTUALLY emitted (slider mode defaults to 768 only), so
        # provenance / ⎘ Share config never disagree with what the job used.
        'resolution': _effective_resolution(ds),
        'save_every': _save_every(ds),
        'max_step_saves': _max_step_saves(ds),
        'optimizer': _optimizer_eff(ds),
        'lr': _lr_eff(ds),
    }
    network = _network_block(ds, rank, fam)
    if network['type'] == 'lora':
        # ``None`` is an explicit topology fact for new records: linear-only.
        # Older records lacking these keys remain unknown and permissive.
        snap['conv'] = network.get('conv')
        snap['conv_alpha'] = network.get('conv_alpha')
    if slider_mode_enabled(ds):
        # Slider mode (Beta): the prompt pair IS the recipe — it must travel in
        # provenance and ⎘ Share config. Like Style, the trigger is only an
        # internal run/file identifier here (no activation token), so it is
        # deliberately absent.
        sc = _slider_settings(ds)
        snap['slider_mode'] = True
        snap['slider'] = {
            'positive_prompt': sc.get('positive') or '',
            'negative_prompt': sc.get('negative') or '',
            'target_class': sc.get('target_class') or '',
            'anchor_class': sc.get('anchor') or '',
            'guidance_strength': _slider_guidance(ds),
            'anchor_strength': _slider_anchor_strength(ds),
        }
    elif fds.is_style(ds):
        # The persisted trigger is only an internal run/file identifier for a
        # Style LoRA. It is deliberately absent from provenance/share snapshots
        # so nobody mistakes it for an activation token.
        snap['style_mode'] = 'always_on'
        snap['effective_caption_dropout'] = _effective_style_caption_dropout(fam)
    else:
        # Character/concept LoRAs do require this token to reproduce inference.
        snap['trigger'] = _safe_trigger(ds)
    if fam != 'sdxl':
        timestep_default = (zrecipe['timestep_type'] if zrecipe
                            else _DEFAULT_TIMESTEP.get(fam, 'sigmoid'))
        snap['timestep_type'] = _timestep_type_eff(ds, timestep_default)
    if zrecipe:
        # Persist what actually went to ai-toolkit.  ``training_adapter=None`` is
        # deliberate and useful provenance for Base/De-Turbo, not a missing key.
        snap.update({'recipe_version': zrecipe['recipe_version'],
                     'effective_base': zrecipe['effective_base'],
                     'training_adapter': zrecipe['training_adapter']})
    # Provenance: the ACTUAL custom paths that went to ai-toolkit (weights + the
    # SDXL-only VAE/TE overrides). Surfaced in the Runs hub and the ⎘ Share config
    # (both redact the home-dir prefix via redact_user_paths — no identity leaks).
    _weights = getattr(ds, 'train_base_model', None)
    if _is_custom_weights(_weights):
        snap['base_weights'] = _weights
    if fam in VAE_TE_OVERRIDE_FAMILIES:
        if getattr(ds, 'train_vae_path', None):
            snap['vae_path'] = ds.train_vae_path
        if getattr(ds, 'train_te_path', None):
            snap['te_name_or_path'] = ds.train_te_path
    s = _train_settings(ds)
    for k in ('dropout', 'sample_every'):
        if s.get(k):
            snap[k] = s[k]
    # Recipe levers, ALWAYS stamped with their effective value — including the
    # default. They used to be omitted when they matched the default, which reads
    # as "compact" until you compare two runs: an absent `ema` was then
    # indistinguishable from a run recorded before the key existed, so the panel
    # could not say which of two runs had EMA on — the exact question the EMA
    # experiment exists to answer. An explicit 'off' costs one line and cannot be
    # misread. The cloud run stamps this same snapshot.
    snap['network_type'] = _network_type_eff(ds)
    if snap['network_type'] == 'lokr':
        # ai-toolkit has changed the implicit `lokr_full_rank` default across
        # releases. LDS pins False in the emitted job so rank/alpha stay real;
        # stamp both facts to make a shared recipe reproducible.
        snap['lokr_full_rank'] = _lokr_full_rank_eff(ds)
        factor = _lokr_factor_eff(ds)
        snap['lokr_factor'] = factor if factor is not None else 'auto'
    em = _ema_eff(ds)
    snap['ema'] = em if em is not None else 'off'
    if fam == 'krea' or s.get('content_or_style') in _CONTENT_OR_STYLE_CHOICES:
        snap['content_or_style'] = _content_or_style_eff(ds)
    if fam == 'krea':
        dg_enabled = _differential_guidance_enabled(ds)
        snap['do_differential_guidance'] = dg_enabled
        snap['differential_guidance_scale'] = (
            _differential_guidance_scale_eff(ds) if dg_enabled else 'off')
    snap['lr_scheduler'] = s.get('lr_scheduler') if s.get('lr_scheduler') in _LR_SCHEDULER_CHOICES else 'constant'
    snap['warmup'] = s.get('warmup') if s.get('warmup') in _WARMUP_CHOICES else 0
    snap['grad_accum'] = _grad_accum(ds)
    # Fixed at 1 by every family's recipe today. Recorded anyway: the day it stops
    # being 1, the runs on either side of the change have to be comparable.
    snap['batch_size'] = 1
    # Stamped EFFECTIVE, like `ema` and the memory keys: a recipe that caches text
    # embeddings cannot train a second caption (see _dual_captions_unsupported_reason),
    # so recording the preference there would make the run comparison claim two runs
    # differ by dual captions when the trainer saw exactly the same captions.
    snap['dual_captions'] = (bool(s.get('dual_captions'))
                             and not _cache_text_embeddings_eff(ds, fam))
    # Face masking is a per-run FACT, not a preference: two concept runs that
    # differ only by it are not the same experiment, so it is stamped effective
    # (concept-only, hence the kind check) exactly like `ema` and the memory keys.
    snap['mask_faces'] = bool(s.get('mask_faces')) and fds.is_concept(ds)
    # Person masking, same reasoning as `mask_faces` right above: two runs of the
    # same dataset that differ only by it are NOT the same experiment, and until
    # this line the value lived in a browser and appeared in no snapshot at all —
    # so a run comparison could not tell them apart. Stamped EFFECTIVE (the export
    # guards for concept/style and slider mode are already folded in), and the
    # per-run override wins when a replay carries one.
    snap['masked'] = (bool(masked) if isinstance(masked, bool)
                      else person_masking_enabled(ds))
    # Memory strategy — stamped with its EFFECTIVE value for the same reason as
    # `ema` above: two runs of the same dataset can differ only by these, and a
    # quantised run and a full-precision one are NOT the same experiment. Absent
    # would be indistinguishable from "recorded before the keys existed".
    _memdef = _memory_saving_defaults(ds, fam)
    for _k in _MEMORY_SETTING_KEYS:
        snap[_k] = _memory_flag_eff(ds, _k, _memdef[_k])
    for _k in (
        'weight_decay', 'loss_type', 'qtype', 'qtype_te',
        'layer_offloading', 'layer_offloading_transformer_percent',
        'layer_offloading_text_encoder_percent', 'cache_text_embeddings',
        'save_dtype', 'preset_steps_per_image', 'preset_steps_min',
        'preset_steps_max', 'preset_steps_fixed',
    ):
        if _k in s:
            snap[_k] = s[_k]
    return snap


def effective_train_settings(ds, family=None) -> dict:
    """Réglages pour la famille courante — ce que « Advanced options » affiche et
    ce que build_job_config enverra. `rank` = choix STOCKÉ (None = auto/défaut) pour
    que le select re-coche « Auto » ; `effective_rank`/`alpha`/`default_rank` = ce
    qui sera réellement utilisé (pour le libellé explicatif)."""
    fam = family or _train_type(ds)
    s = _train_settings(ds)
    stored_rank = s.get('rank') if s.get('rank') in _RANK_CHOICES else None
    eff_rank = stored_rank if stored_rank else _default_rank_for(ds, fam)
    res = s.get('resolution')
    trig = _safe_trigger(ds)
    stored_prompts = s.get('sample_prompts')
    network = _network_block(ds, eff_rank, fam)
    return {'rank': stored_rank,                       # None → Auto (défaut family-aware)
            'effective_rank': eff_rank,                # ce qui part à ai-toolkit
            'alpha': _lora_alpha_eff(ds, eff_rank, fam),   # alpha EFFECTIF (override-aware) — libellé
            'default_rank': _default_rank_for(ds, fam),
            # --- Expert levers (None/off = comportement actuel ; le select recoche « Auto ») ---
            'alpha_setting': _numeric_choice(s.get('alpha'), _ALPHA_CHOICES),
            'default_alpha': _lora_alpha(eff_rank, fam, ds),
            'alpha_choices': list(_ALPHA_CHOICES),
            'dropout': s.get('dropout') if s.get('dropout') in _DROPOUT_CHOICES else None,
            'dropout_choices': list(_DROPOUT_CHOICES),
            'timestep_type': s.get('timestep_type') if s.get('timestep_type') in _TIMESTEP_TYPE_CHOICES else None,
            'timestep_type_choices': list(_TIMESTEP_TYPE_CHOICES),
            'default_timestep_type': _DEFAULT_TIMESTEP.get(fam),   # None pour sdxl → contrôle masqué
            'timestep_type_supported': fam != 'sdxl',
            'optimizer': s.get('optimizer') if s.get('optimizer') in _OPTIMIZER_CHOICES else None,   # None → adamw8bit
            'optimizer_choices': list(_OPTIMIZER_CHOICES),
            'automagic3_supported': _aitoolkit_supports_automagic3(),
            # Effective LR (absolute) this run trains at — the ▶ Continue dialog's LR
            # factor shows the resulting rate and hides the knob when it's adaptive
            # (Prodigy). Not an editable Advanced-options control; read-only here.
            'learning_rate': _lr_eff(ds),
            'lr_scheduler': s.get('lr_scheduler') if s.get('lr_scheduler') in _LR_SCHEDULER_CHOICES else None,  # None → constant
            'lr_scheduler_choices': list(_LR_SCHEDULER_CHOICES),
            'warmup': s.get('warmup') if s.get('warmup') in _WARMUP_CHOICES else None,
            'warmup_choices': list(_WARMUP_CHOICES),
            'grad_accum': _numeric_choice(s.get('grad_accum'), _GRAD_ACCUM_CHOICES),   # None → 1
            'grad_accum_choices': list(_GRAD_ACCUM_CHOICES),
            'network_type': s.get('network_type') if s.get('network_type') in _NETWORK_TYPE_CHOICES else None,  # None → lora
            'network_type_choices': list(_NETWORK_TYPE_CHOICES),
            # LoKr is arch-generic in ai-toolkit → offered on every family. The flag
            # mirrors timestep_type_supported so the UI can gate a future family with
            # one line; today it is always True (no family refuses lokr).
            'network_type_supported': True,
            'lokr_factor': (_lokr_factor_eff(ds) if _network_type_eff(ds) == 'lokr'
                            else None),
            'lokr_factor_choices': list(_LOKR_FACTOR_CHOICES),
            'lokr_full_rank': (network.get('lokr_full_rank')
                               if network['type'] == 'lokr' else False),
            'conv': network.get('conv'),
            'conv_alpha': network.get('conv_alpha'),
            'ema': s.get('ema') if s.get('ema') in _EMA_CHOICES else None,   # None → off
            'ema_choices': list(_EMA_CHOICES),
            # The four fields below are deliberately Krea-only in LDS. ai-toolkit
            # accepts them broadly, but they exist here to make the Krea Raw LoKr
            # community preset transparent rather than silently storing invisible
            # settings on unrelated families. Values survive a family switch and
            # return when the user comes back to Krea, like other advanced knobs.
            'krea_recipe_supported': fam == 'krea',
            'content_or_style': (s.get('content_or_style')
                                 if fam == 'krea' and s.get('content_or_style') in _CONTENT_OR_STYLE_CHOICES
                                 else None),
            'content_or_style_choices': list(_CONTENT_OR_STYLE_CHOICES),
            'content_or_style_default': 'balanced',
            'do_differential_guidance': (s.get('do_differential_guidance') is True
                                         if fam == 'krea' else False),
            'differential_guidance_scale': (_differential_guidance_scale_eff(ds)
                                            if fam == 'krea' else None),
            # Dual long+short captioning (ai-toolkit short_and_long_captions). Boolean,
            # default OFF. Local training only for now (the cloud pod's dataset upload
            # skips the JSON caption file), so the recipe strips it on the cloud path.
            'dual_captions': bool(s.get('dual_captions')),
            # Concept face masking (issue #15). `mask_faces` = the stored opt-in;
            # `mask_faces_supported` = whether this dataset can use it at all
            # (concept only), so the panel states the reason instead of hiding it.
            'mask_faces': bool(s.get('mask_faces')) and fds.is_concept(ds),
            'mask_faces_supported': fds.is_concept(ds),
            'mask_faces_concept_conflict': fds.concept_face_conflict(ds),
            # Person masking (background at 10 % loss weight). `masked` = the value
            # this dataset will train with, resolved (default ON, forced OFF for
            # concept/style and slider mode); `masked_supported` = whether the
            # toggle can do anything at all here, so the panel states the reason
            # instead of hiding the control; `masked_stored` = the RAW tri-state
            # (None = never answered) the one-time localStorage carry-over reads.
            'masked': person_masking_enabled(ds),
            'masked_supported': not fds.is_conceptual(ds) and not slider_mode_enabled(ds),
            'masked_stored': fds.person_masking_stored(ds),
            # --- Memory strategy (issue #14) -----------------------------------
            # `memory_saving` = le choix STOCKÉ par clé (None = « Auto », le panel
            # recoche le défaut de la famille) ; `memory_saving_default` = ce que
            # la recette calibrée émet ; `memory_saving_effective` = ce qui partira
            # vraiment. `memory_advice` porte le conseil indexé sur la carte réelle
            # (verdict/vram_gb/gpu) — purement consultatif, jamais appliqué seul.
            'memory_saving': {k: (s.get(k) if isinstance(s.get(k), bool) else None)
                              for k in _MEMORY_SETTING_KEYS},
            'memory_saving_default': _memory_saving_defaults(ds, fam),
            'memory_saving_effective': {
                k: _memory_flag_eff(ds, k, _memory_saving_defaults(ds, fam)[k])
                for k in _MEMORY_SETTING_KEYS},
            'memory_advice': _memory_saving_advice(ds, fam),
            # None when the family's calibrated recipe is intact; otherwise names
            # WHICH savers are off and what this family needs without them. The
            # panel states it next to the checkboxes, the preflight repeats it
            # before the GPU (or the rented pod) is paid for. Both read the same
            # function — one rule, two surfaces.
            'memory_risk': memory_saving_risk(ds, fam),
            # Serializer-backed preset primitives. Stored None means the family/
            # ai-toolkit default remains in force; effective save dtype is explicit.
            'weight_decay': s.get('weight_decay'),
            'loss_type': (s.get('loss_type')
                          if s.get('loss_type') in _LOSS_TYPE_CHOICES else None),
            'qtype': s.get('qtype') if s.get('qtype') in _QTYPE_CHOICES else None,
            'qtype_te': (s.get('qtype_te')
                         if s.get('qtype_te') in _QTYPE_CHOICES else None),
            'layer_offloading': (s.get('layer_offloading')
                                 if isinstance(s.get('layer_offloading'), bool)
                                 else None),
            'layer_offloading_transformer_percent':
                s.get('layer_offloading_transformer_percent'),
            'layer_offloading_text_encoder_percent':
                s.get('layer_offloading_text_encoder_percent'),
            'cache_text_embeddings': (s.get('cache_text_embeddings')
                                      if isinstance(s.get('cache_text_embeddings'), bool)
                                      else None),
            'save_dtype': _save_dtype_eff(ds),
            'preset_steps_policy': _preset_steps_policy(ds),
            # Family label, so the panel can name the family in that sentence
            # without shipping a second copy of the map.
            'family_label': _FAMILY_LABEL.get(fam, fam),
            'resolution': res if res in _RES_CHOICES else '768,1024',
            # `resolution` above is the STORED choice (default label when unset);
            # these two report what the run will actually train at — slider mode
            # defaults to 768 only, so the panel control + summary stay truthful.
            'resolution_explicit': res in _RES_CHOICES,
            'effective_resolution': _effective_resolution(ds),
            'save_every': _save_every(ds),
            'max_step_saves': _max_step_saves(ds),
            'max_step_saves_choices': list(_MAX_SAVES_CHOICES),
            'sample_every': _sample_every(ds),
            # liste STOCKÉE brute (telle que tapée) ou [] → textarea vide = « défauts ».
            'sample_prompts': stored_prompts if isinstance(stored_prompts, list) else [],
            # défaut résolu (kind + trigger courant) : placeholder/aperçu quand vide.
            'sample_prompts_default': _resolved_default_sample_prompts(ds, trig),
            'sample_every_choices': list(_SAMPLE_EVERY_CHOICES),
            'max_sample_prompts': _MAX_SAMPLE_PROMPTS,
            # --- full-model (dense) recipe: every unlocked value, the two
            # delivery switches, and what the choice costs. Always present so
            # the panel can render the locked card and its editable half from
            # ONE payload; `*_stored` is the raw choice (None = "default", so
            # the control re-checks Auto), the bare key is what the run will
            # actually use.
            **_dense_settings_payload(ds)}


def _dense_settings_payload(ds) -> dict:
    s = _train_settings(ds)
    return {
        'dense_lr': _dense_lr(ds),
        'dense_lr_stored': s.get('dense_lr') if isinstance(s.get('dense_lr'), (int, float))
        and not isinstance(s.get('dense_lr'), bool) else None,
        'dense_lr_default': FULL_TRANSFORMER_LR,
        'dense_lr_min': FULL_TRANSFORMER_LR_MIN,
        'dense_lr_max': FULL_TRANSFORMER_LR_MAX,
        'dense_resolution': _dense_resolution(ds),
        'dense_resolution_default': KREA_TRAIN_RESOLUTION,
        'dense_resolution_choices': list(FULL_TRANSFORMER_RESOLUTION_CHOICES),
        'dense_save_every': _dense_save_every(ds),
        'dense_save_every_default': FULL_TRANSFORMER_SAVE_EVERY,
        'dense_save_every_min': FULL_TRANSFORMER_SAVE_EVERY_MIN,
        'dense_save_every_max': FULL_TRANSFORMER_SAVE_EVERY_MAX,
        'dense_max_step_saves': _dense_max_step_saves(ds),
        'dense_max_step_saves_default': FULL_TRANSFORMER_MAX_STEP_SAVES,
        'dense_max_step_saves_max': FULL_TRANSFORMER_MAX_STEP_SAVES_MAX,
        'dense_grad_accum': _dense_grad_accum(ds),
        'dense_grad_accum_default': FULL_TRANSFORMER_GRAD_ACCUM,
        'dense_grad_accum_choices': list(FULL_TRANSFORMER_GRAD_ACCUM_CHOICES),
        # The price tag, next to the control that sets it. Accumulation is the
        # only dense lever that spends money instead of memory, and the panel
        # must be able to say "≈3× longer, ≈3× the pod bill" at the moment of
        # choosing — not leave it to be discovered on the invoice.
        'dense_time_multiplier': dense_time_multiplier(ds),
        'dense_images_per_step': dense_images_per_step(ds),
        'dense_lr_schedule': _dense_lr_schedule(ds),
        'dense_lr_schedule_default': FULL_TRANSFORMER_LR_SCHEDULE,
        'dense_lr_schedule_choices': list(FULL_TRANSFORMER_LR_SCHEDULE_CHOICES),
        'dense_warmup': _dense_warmup(ds),
        'dense_warmup_default': FULL_TRANSFORMER_WARMUP,
        'dense_warmup_min': FULL_TRANSFORMER_WARMUP_MIN,
        'dense_warmup_max': FULL_TRANSFORMER_WARMUP_MAX,
        # Warmup steps only reach ai-toolkit on this one schedule; the panel
        # gates its control on the same fact rather than re-deriving it.
        'dense_warmup_applies': _dense_lr_schedule(ds) == 'constant_with_warmup',
        'dense_timestep_type': _dense_timestep_type(ds),
        'dense_timestep_type_default': FULL_TRANSFORMER_TIMESTEP_TYPE,
        'dense_timestep_type_choices': list(FULL_TRANSFORMER_TIMESTEP_TYPE_CHOICES),
        'dense_fp8_export': dense_fp8_export_enabled(ds),
        'dense_keep_bf16': dense_keep_bf16_master(ds),
        # What the delivery will weigh, so the panel can say it BEFORE the pod is
        # rented rather than after the 403. One source of arithmetic with the
        # Hugging Face pre-check (hf_storage.dense_storage_forecast).
        'dense_storage_plan': dense_storage_plan(ds),
        # What to generate with once the model lands — the SAME numbers the run
        # previews with, so the panel never hard-codes a second version of them.
        'dense_inference_hint': dense_inference_hint(ds),
    }


def dense_storage_plan(ds) -> dict:
    """How many objects of what size ONE dense run will put on Hugging Face.

    Deliberately expressed as the PEAK, not the steady state: even when the bf16
    master is dropped, ai-toolkit pushes it first and the fp8 twin is uploaded
    before it is deleted. A forecast that quoted the post-cleanup total would be
    accurate about the wrong moment — the moment that refuses a push is the peak.
    """
    from . import hf_storage
    from . import fp8_export
    keeps = _dense_max_step_saves(ds)
    checkpoint, source = hf_storage.dense_checkpoint_bytes()
    fp8 = (fp8_export.estimate_fp8_bytes(checkpoint)
           if dense_fp8_export_enabled(ds) else 0)
    return {'keeps': keeps, 'checkpoint_bytes': checkpoint,
            'checkpoint_source': source,
            # `fp8_bytes` is the forecast CEILING (it has to round up or it stops
            # protecting anything); `fp8_typical_bytes` is what to SHOW someone
            # asking how big their download will be. Same file, two questions.
            'fp8_bytes': fp8,
            'fp8_typical_bytes': (fp8_export.typical_fp8_bytes(checkpoint)
                                  if dense_fp8_export_enabled(ds) else 0),
            'keep_bf16': dense_keep_bf16_master(ds),
            'peak_bytes': checkpoint * keeps + fp8,
            'resident_bytes': (checkpoint * keeps if dense_keep_bf16_master(ds)
                               else 0) + fp8}


def _training_selection_candidate(ds, patch: dict, requested_mode) -> dict:
    """Validate the mode/family/base/variant tuple without mutating ``ds``.

    The TrainingPanel saves those four controls together.  Dense Krea only has
    one legal tuple, so validating the mode against yesterday's persisted base
    before applying today's explicit ``base_model=''`` would reject a perfectly
    valid save.  Build one exact candidate first, validate it, then let the
    caller perform a single database commit.
    """
    disable_slider = False
    if 'disable_slider_for_full_transformer' in patch:
        disable_slider = patch['disable_slider_for_full_transformer']
        if not isinstance(disable_slider, bool):
            raise ValueError(
                'disable_slider_for_full_transformer must be true or false')

    current_family = _train_type(ds)
    family = current_family
    if 'train_type' in patch:
        raw_family = patch.get('train_type')
        if not isinstance(raw_family, str):
            raise ValueError('train_type must be a supported model family')
        family = raw_family.strip().lower()
        if family not in fds.TRAIN_TYPES:
            raise ValueError('train_type must be one of ' + ', '.join(fds.TRAIN_TYPES))
    family_changed = family != current_family

    if family_changed:
        remembered_base, remembered_variant = fds.remembered_family_base(ds, family)
        base_model = remembered_base if remembered_base is not None else ''
        variant = remembered_variant or _default_variant_for(family)
    else:
        base_model = getattr(ds, 'train_base_model', None) or ''
        variant = (getattr(ds, 'train_variant', None)
                   or _default_variant_for(family))

    if 'base_model' in patch:
        raw_base = patch.get('base_model')
        if raw_base is not None and not isinstance(raw_base, str):
            raise ValueError('base_model must be a string or empty')
        base_model = (raw_base or '').strip()
        # Custom weights only: a ComfyUI-relative name addresses a catalog entry
        # this app installed, an absolute path is a file the user picked.
        if _is_custom_weights(base_model):
            assert_trainable_base_file(base_model)
    if 'variant' in patch:
        raw_variant = patch.get('variant')
        if not isinstance(raw_variant, str) or not raw_variant.strip():
            raise ValueError('variant must be a supported model variant')
        variant = raw_variant.strip().lower()
    else:
        variant = str(variant or _default_variant_for(family)).strip().lower()
    if family == 'krea' and variant == 'raw':
        variant = 'base'
    if variant not in _valid_variants_for(family):
        raise ValueError(
            f'variant must be one of {", ".join(_valid_variants_for(family))} '
            f'for {family}')

    mode = (normalize_training_mode(requested_mode)
            if requested_mode is not None else training_mode(ds))
    if disable_slider and mode != 'full_transformer':
        raise ValueError(
            'disable_slider_for_full_transformer requires '
            "training_mode='full_transformer'")
    candidate_slider = _PERSISTED
    if disable_slider:
        slider_settings = _slider_settings(ds)
        slider_settings.pop('enabled', None)
        candidate_slider = (json.dumps(slider_settings)
                            if slider_settings else None)
    candidate = _train_context_view(
        ds, family, variant, base_model=base_model, training_mode=mode,
        train_slider=candidate_slider)
    _assert_full_transformer_recipe(candidate)
    return {'family': family, 'base_model': base_model, 'variant': variant,
            'mode': mode, 'family_changed': family_changed,
            'disable_slider': disable_slider,
            'train_slider': candidate_slider}


def update_train_settings(user_id, dataset_id, patch: dict, *, _settings=None) -> dict:
    """Valide + fusionne un patch {rank?, resolution?, save_every?, sample_every?,
    sample_prompts?} dans train_settings. Une clé à None/'auto'/vide est RETIRÉE
    (retour au défaut). Retourne les réglages effectifs pour la famille courante."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # `training_mode` is a first-class dataset column, not an ai-toolkit expert
    # knob and not part of presets. It shares this endpoint so the TrainingPanel
    # can persist the selector atomically with any advanced-options patch.
    requested_training_mode = None
    selection = None
    if _settings is None:
        if 'training_mode' in patch:
            requested_training_mode = normalize_training_mode(patch['training_mode'])
        if any(key in patch for key in
               ('training_mode', 'train_type', 'base_model', 'variant',
                'disable_slider_for_full_transformer')):
            selection = _training_selection_candidate(
                ds, patch, requested_training_mode)
            if (selection['family_changed']
                    and any(key in patch for key in TRAIN_SETTING_KEYS)):
                raise ValueError(
                    'change train_type separately from advanced training settings')
    # ``_settings`` is the preset path's private, unpersisted candidate.  Reusing
    # this validator keeps every acceptance/rejection rule identical while a
    # preset validates its complete replacement before making one DB write.
    cur = _train_settings(ds) if _settings is None else _settings
    if 'rank' in patch:
        r = patch['rank']
        if r in (None, 'auto'):
            cur.pop('rank', None)
        elif r in _RANK_CHOICES:
            cur['rank'] = r
        else:
            raise ValueError(f'rank must be one of {_RANK_CHOICES} (or auto)')
    if 'resolution' in patch:
        v = patch['resolution']
        if v in _RES_CHOICES:
            cur['resolution'] = v
        else:
            raise ValueError(f'resolution must be one of {list(_RES_CHOICES)}')
    if 'save_every' in patch:
        v = patch['save_every']
        if v in _SAVE_CHOICES:
            cur['save_every'] = v
        else:
            raise ValueError(f'save_every must be one of {_SAVE_CHOICES}')
    if 'max_step_saves' in patch:
        v = patch['max_step_saves']
        if v in (None, 'auto'):
            cur.pop('max_step_saves', None)
        elif v in _MAX_SAVES_CHOICES:
            cur['max_step_saves'] = v
        else:
            raise ValueError(f'max_step_saves must be one of {_MAX_SAVES_CHOICES}')
    if 'sample_every' in patch:
        v = patch['sample_every']
        if v in _SAMPLE_EVERY_CHOICES:
            cur['sample_every'] = v
        else:
            raise ValueError(f'sample_every must be one of {_SAMPLE_EVERY_CHOICES}')
    if 'sample_prompts' in patch:
        v = patch['sample_prompts']
        # Accepte aussi une string multi-lignes (une par prompt) pour le confort UI.
        if isinstance(v, str):
            v = v.splitlines()
        if v in (None, ''):
            cur.pop('sample_prompts', None)               # vide → retour aux défauts kind-aware
        elif isinstance(v, list):
            cleaned = [str(x).strip() for x in v if str(x).strip()][:_MAX_SAMPLE_PROMPTS]
            if cleaned:
                cur['sample_prompts'] = cleaned
            else:
                cur.pop('sample_prompts', None)
        else:
            raise ValueError('sample_prompts must be a list of strings (or empty to reset)')
    # --- the unlocked half of the dense recipe --------------------------------
    # Bounded, never free-form: the bounds are what keep "editable" from meaning
    # "able to burn 80 GB of rented GPU on a value that cannot converge".
    if 'dense_lr' in patch:
        v = patch['dense_lr']
        if v in (None, '', 'auto'):
            cur.pop('dense_lr', None)
        elif (isinstance(v, (int, float)) and not isinstance(v, bool)
                and FULL_TRANSFORMER_LR_MIN <= float(v) <= FULL_TRANSFORMER_LR_MAX):
            cur['dense_lr'] = float(v)
        else:
            raise ValueError(
                f'dense_lr must be between {FULL_TRANSFORMER_LR_MIN:g} and '
                f'{FULL_TRANSFORMER_LR_MAX:g} (or auto)')
    if 'dense_resolution' in patch:
        v = patch['dense_resolution']
        if v in (None, '', 'auto'):
            cur.pop('dense_resolution', None)
        elif v in FULL_TRANSFORMER_RESOLUTION_CHOICES:
            cur['dense_resolution'] = v
        else:
            raise ValueError('dense_resolution must be one of '
                             f'{list(FULL_TRANSFORMER_RESOLUTION_CHOICES)} (or auto)')
    if 'dense_save_every' in patch:
        v = patch['dense_save_every']
        if v in (None, '', 'auto'):
            cur.pop('dense_save_every', None)
        elif (isinstance(v, int) and not isinstance(v, bool)
                and FULL_TRANSFORMER_SAVE_EVERY_MIN <= v <= FULL_TRANSFORMER_SAVE_EVERY_MAX):
            cur['dense_save_every'] = v
        else:
            raise ValueError(
                f'dense_save_every must be between {FULL_TRANSFORMER_SAVE_EVERY_MIN} '
                f'and {FULL_TRANSFORMER_SAVE_EVERY_MAX} steps (or auto)')
    if 'dense_max_step_saves' in patch:
        v = patch['dense_max_step_saves']
        if v in (None, '', 'auto'):
            cur.pop('dense_max_step_saves', None)
        elif (isinstance(v, int) and not isinstance(v, bool)
                and 1 <= v <= FULL_TRANSFORMER_MAX_STEP_SAVES_MAX):
            cur['dense_max_step_saves'] = v
        else:
            raise ValueError('dense_max_step_saves must be between 1 and '
                             f'{FULL_TRANSFORMER_MAX_STEP_SAVES_MAX} (or auto)')
    if 'dense_grad_accum' in patch:
        v = patch['dense_grad_accum']
        if v in (None, '', 'auto'):
            cur.pop('dense_grad_accum', None)
        elif (isinstance(v, int) and not isinstance(v, bool)
                and v in FULL_TRANSFORMER_GRAD_ACCUM_CHOICES):
            cur['dense_grad_accum'] = v
        else:
            # The message says what the refusal protects: this is the one dense
            # value whose upper bound is a budget, not a memory limit.
            raise ValueError(
                'dense_grad_accum must be one of '
                f'{list(FULL_TRANSFORMER_GRAD_ACCUM_CHOICES)} (or auto) — each '
                'step becomes that many images, so the run (and the rented GPU '
                'it is billed on) takes that many times longer')
    if 'dense_lr_schedule' in patch:
        v = patch['dense_lr_schedule']
        if v in (None, '', 'auto'):
            cur.pop('dense_lr_schedule', None)
        elif v in FULL_TRANSFORMER_LR_SCHEDULE_CHOICES:
            cur['dense_lr_schedule'] = v
        else:
            raise ValueError(
                'dense_lr_schedule must be one of '
                f'{list(FULL_TRANSFORMER_LR_SCHEDULE_CHOICES)} (or auto)')
    if 'dense_warmup' in patch:
        v = patch['dense_warmup']
        if v in (None, '', 'auto'):
            cur.pop('dense_warmup', None)
        elif (isinstance(v, int) and not isinstance(v, bool)
                and FULL_TRANSFORMER_WARMUP_MIN <= v <= FULL_TRANSFORMER_WARMUP_MAX):
            cur['dense_warmup'] = v
        else:
            raise ValueError(
                f'dense_warmup must be between {FULL_TRANSFORMER_WARMUP_MIN} and '
                f'{FULL_TRANSFORMER_WARMUP_MAX} steps (or auto), and it only '
                "applies to the 'constant_with_warmup' schedule")
    if 'dense_timestep_type' in patch:
        v = patch['dense_timestep_type']
        if v in (None, '', 'auto'):
            cur.pop('dense_timestep_type', None)
        elif v in FULL_TRANSFORMER_TIMESTEP_TYPE_CHOICES:
            cur['dense_timestep_type'] = v
        else:
            # 'shift' is the value people ask for and the one that lands here:
            # say why it is missing instead of letting it read as an oversight.
            raise ValueError(
                'dense_timestep_type must be one of '
                f'{list(FULL_TRANSFORMER_TIMESTEP_TYPE_CHOICES)} (or auto); '
                "shift-based schedules are not offered for Krea 2 because the "
                'trainer would compute their shift from a four-times-too-large '
                'token count')
    for _flag in ('dense_fp8_export', 'dense_keep_bf16'):
        if _flag in patch:
            v = patch[_flag]
            if v in (None, '', 'auto'):
                cur.pop(_flag, None)
            elif isinstance(v, bool):
                cur[_flag] = v
            else:
                raise ValueError(f'{_flag} must be true or false')
    if 'dropout' in patch:
        v = patch['dropout']
        if v in (None, 0, 0.0, 'off', ''):
            cur.pop('dropout', None)                       # off → clé retirée
        elif v in _DROPOUT_CHOICES:
            cur['dropout'] = v
        else:
            raise ValueError(f'dropout must be one of {_DROPOUT_CHOICES} (or off)')
    if 'alpha' in patch:
        v = patch['alpha']
        if v in (None, 'auto'):
            cur.pop('alpha', None)                         # auto → alpha dérivé du rank
        elif type(v) is int and v in _ALPHA_CHOICES:
            cur['alpha'] = v
        else:
            raise ValueError(f'alpha must be one of {_ALPHA_CHOICES} (or auto)')
    if 'timestep_type' in patch:
        v = patch['timestep_type']
        if v in (None, 'auto', ''):
            cur.pop('timestep_type', None)                 # auto → défaut family-aware
        elif v in _TIMESTEP_TYPE_CHOICES:
            cur['timestep_type'] = v
        else:
            raise ValueError(f'timestep_type must be one of {_TIMESTEP_TYPE_CHOICES} (or auto)')
    if 'optimizer' in patch:
        v = patch['optimizer']
        if v in (None, 'auto', '', 'adamw8bit'):
            cur.pop('optimizer', None)                     # défaut → clé retirée
        elif v in _OPTIMIZER_CHOICES:
            if (v == 'automagic3'
                    and _numeric_choice(cur.get('grad_accum'),
                                        _GRAD_ACCUM_CHOICES) not in (None, 1)):
                raise ValueError(
                    'Automagic3 does not support gradient accumulation above 1; '
                    'set grad_accum to 1/auto or choose another optimizer')
            cur['optimizer'] = v
        else:
            raise ValueError(f'optimizer must be one of {_OPTIMIZER_CHOICES} (or auto)')
    if 'lr_scheduler' in patch:
        v = patch['lr_scheduler']
        if v in (None, 'auto', '', 'constant'):
            cur.pop('lr_scheduler', None)                  # constant = défaut → clé retirée
        elif v in _LR_SCHEDULER_CHOICES:
            cur['lr_scheduler'] = v
        else:
            raise ValueError(f'lr_scheduler must be one of {_LR_SCHEDULER_CHOICES} (or auto)')
    if 'warmup' in patch:
        v = patch['warmup']
        if v in (None, 0, 'off', ''):
            cur.pop('warmup', None)
        elif v in _WARMUP_CHOICES:
            cur['warmup'] = v
        else:
            raise ValueError(f'warmup must be one of {_WARMUP_CHOICES} (or off)')
    if 'grad_accum' in patch:
        v = patch['grad_accum']
        if v in (None, 'auto') or (type(v) is int and v == 1):
            cur.pop('grad_accum', None)                    # 1 = défaut → clé retirée
        elif type(v) is int and v in _GRAD_ACCUM_CHOICES:
            if v > 1 and cur.get('optimizer') == 'automagic3':
                raise ValueError(
                    'Automagic3 does not support gradient accumulation above 1; '
                    'set grad_accum to 1/auto or choose another optimizer')
            cur['grad_accum'] = v
        else:
            raise ValueError(f'grad_accum must be one of {_GRAD_ACCUM_CHOICES} (or auto)')
    if 'network_type' in patch:
        v = patch['network_type']
        if v in (None, 'auto', '', 'lora'):
            cur.pop('network_type', None)                  # lora = défaut → clé retirée
        elif v in _NETWORK_TYPE_CHOICES:
            cur['network_type'] = v
        else:
            raise ValueError(f'network_type must be one of {_NETWORK_TYPE_CHOICES} (or auto)')
    if 'lokr_factor' in patch:
        v = patch['lokr_factor']
        if v in (None, 'auto', '', -1):
            cur.pop('lokr_factor', None)                   # -1 = ai-toolkit auto factor
        elif isinstance(v, int) and not isinstance(v, bool) and v in _LOKR_FACTOR_CHOICES:
            cur['lokr_factor'] = v
        else:
            raise ValueError(f'lokr_factor must be one of {_LOKR_FACTOR_CHOICES} (or auto)')
    if 'lokr_full_rank' in patch:
        v = patch['lokr_full_rank']
        if isinstance(v, bool):
            cur['lokr_full_rank'] = v
        elif v in (None, 'auto', ''):
            cur.pop('lokr_full_rank', None)
        else:
            raise ValueError('lokr_full_rank must be true, false or auto')
    for key, choices in (('conv', _RANK_CHOICES), ('conv_alpha', _ALPHA_CHOICES)):
        if key not in patch:
            continue
        v = patch[key]
        if v in (None, 'auto', '', 0):
            cur.pop(key, None)
        elif type(v) is int and v in choices:
            cur[key] = v
        else:
            raise ValueError(f'{key} must be one of {choices} (or auto)')
    if 'ema' in patch:
        v = patch['ema']
        if v in (None, 'off', '', 0, 0.0):
            cur.pop('ema', None)                           # off → clé retirée
        elif v in _EMA_CHOICES:
            cur['ema'] = v
        else:
            raise ValueError(f'ema must be one of {_EMA_CHOICES} (or off)')
    if 'content_or_style' in patch:
        v = patch['content_or_style']
        if v in (None, 'auto', ''):
            cur.pop('content_or_style', None)
        elif v in _CONTENT_OR_STYLE_CHOICES:
            # Keep an explicit 'balanced' setting when a preset provides it: the
            # preset then remains self-describing even though it matches ai-toolkit's
            # default today.
            cur['content_or_style'] = v
        else:
            raise ValueError(f'content_or_style must be one of {_CONTENT_OR_STYLE_CHOICES} (or auto)')
    if 'do_differential_guidance' in patch:
        v = patch['do_differential_guidance']
        if not isinstance(v, bool):
            raise ValueError('do_differential_guidance must be true or false')
        if v:
            cur['do_differential_guidance'] = True
        else:
            cur.pop('do_differential_guidance', None)
    if 'differential_guidance_scale' in patch:
        v = patch['differential_guidance_scale']
        lo, hi = _DIFFERENTIAL_GUIDANCE_SCALE_RANGE
        if v in (None, 'auto', '', 'off'):
            cur.pop('differential_guidance_scale', None)
        elif (isinstance(v, (int, float)) and not isinstance(v, bool)
              and lo <= float(v) <= hi):
            cur['differential_guidance_scale'] = float(v)
        else:
            raise ValueError(
                f'differential_guidance_scale must be between {lo:g} and {hi:g} (or auto)')
    if 'dual_captions' in patch:
        # Plain boolean lever: truthy stores True, anything falsy drops the key so OFF is
        # byte-identical to a dataset that never touched it.
        if patch['dual_captions']:
            cur['dual_captions'] = True
        else:
            cur.pop('dual_captions', None)
    if 'mask_faces' in patch:
        # Concept face masking (issue #15, shivdbz2010). Same plain-boolean contract
        # as dual_captions: falsy drops the key, so OFF is byte-identical to a
        # dataset that never heard of the option. The CONCEPT-only restriction is
        # enforced where it matters (face_masking_enabled + the export guard), not
        # here — a preset carrying the key must stay applicable to any dataset.
        if patch['mask_faces']:
            cur['mask_faces'] = True
        else:
            cur.pop('mask_faces', None)
    if 'masked' in patch:
        # Person masking. TRI-STATE, like the memory keys and unlike dual_captions /
        # mask_faces: this lever's default is ON, so an explicit False is a VALUE
        # that must be stored — dropping it would silently re-enable masking. Only
        # None/'auto'/'' clears the key back to the default.
        v = patch['masked']
        if isinstance(v, bool):
            cur['masked'] = v
        elif v in (None, 'auto', ''):
            cur.pop('masked', None)
        else:
            raise ValueError('masked must be true, false or auto')
    for _mk in _MEMORY_SETTING_KEYS:
        if _mk not in patch:
            continue
        v = patch[_mk]
        # Tri-state: an explicit False is a VALUE (the whole point of issue #14),
        # so it must be stored, not dropped like a falsy `dual_captions`. Only
        # None/'auto'/'' clear the key back to the family's calibrated default.
        if isinstance(v, bool):
            cur[_mk] = v
        elif v in (None, 'auto', ''):
            cur.pop(_mk, None)
        else:
            raise ValueError(f'{_mk} must be true, false or auto')
    if 'learning_rate' in patch:
        # Not a general Advanced-options control: the family-fixed 1e-4 (or the
        # Prodigy lr=1 convention) is the default, and only the ▶ Continue dialog's
        # LR factor writes an explicit absolute rate here (see resolve_resume_lr).
        v = patch['learning_rate']
        if v in (None, 'auto', ''):
            cur.pop('learning_rate', None)                 # back to the family-fixed default
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            cur['learning_rate'] = float(v)
        else:
            raise ValueError('learning_rate must be a positive number (or auto)')
    if 'weight_decay' in patch:
        v = patch['weight_decay']
        if v in (None, 'auto', '', 'off'):
            cur.pop('weight_decay', None)
        elif (isinstance(v, (int, float)) and not isinstance(v, bool)
              and 0 <= float(v) <= 1):
            cur['weight_decay'] = float(v)
        else:
            raise ValueError('weight_decay must be between 0 and 1 (or auto)')
    if 'loss_type' in patch:
        v = patch['loss_type']
        if v in (None, 'auto', ''):
            cur.pop('loss_type', None)
        elif v in _LOSS_TYPE_CHOICES:
            cur['loss_type'] = v
        else:
            raise ValueError(f'loss_type must be one of {_LOSS_TYPE_CHOICES} (or auto)')
    for key in ('qtype', 'qtype_te'):
        if key not in patch:
            continue
        v = patch[key]
        if v in (None, 'auto', ''):
            cur.pop(key, None)
        elif v in _QTYPE_CHOICES:
            cur[key] = v
        else:
            raise ValueError(f'{key} must be one of {_QTYPE_CHOICES} (or auto)')
    for key in ('layer_offloading', 'cache_text_embeddings'):
        if key not in patch:
            continue
        v = patch[key]
        if isinstance(v, bool):
            cur[key] = v
        elif v in (None, 'auto', ''):
            cur.pop(key, None)
        else:
            raise ValueError(f'{key} must be true, false or auto')
    for key in ('layer_offloading_transformer_percent',
                'layer_offloading_text_encoder_percent'):
        if key not in patch:
            continue
        v = patch[key]
        lo, hi = _OFFLOADING_PERCENT_RANGE
        if v in (None, 'auto', ''):
            cur.pop(key, None)
        elif (isinstance(v, (int, float)) and not isinstance(v, bool)
              and lo <= float(v) <= hi):
            cur[key] = float(v)
        else:
            raise ValueError(f'{key} must be between {lo:g} and {hi:g} (or auto)')
    if 'save_dtype' in patch:
        v = patch['save_dtype']
        if v in (None, 'auto', ''):
            cur.pop('save_dtype', None)
        elif v in _SAVE_DTYPE_CHOICES:
            cur['save_dtype'] = v
        else:
            raise ValueError(f'save_dtype must be one of {_SAVE_DTYPE_CHOICES} (or auto)')
    for key in ('preset_steps_per_image', 'preset_steps_min',
                'preset_steps_max', 'preset_steps_fixed'):
        if key not in patch:
            continue
        v = patch[key]
        if v in (None, 'auto', ''):
            cur.pop(key, None)
        elif type(v) is int and v > 0:
            cur[key] = v
        else:
            raise ValueError(f'{key} must be a positive integer (or auto)')
    if _settings is not None:
        return cur
    if selection and selection['family_changed']:
        # Stash/restore the family-scoped base/settings in-memory only.  The
        # explicit base/variant below and train_settings above then join it in
        # the endpoint's one authoritative commit.
        fds.set_train_type(
            user_id, dataset_id, selection['family'], commit=False,
            target_training_mode=selection['mode'])
        cur = _train_settings(ds)
    if selection:
        if 'train_type' in patch or selection['family_changed']:
            ds.train_type = selection['family']
        if 'base_model' in patch or selection['family_changed']:
            ds.train_base_model = selection['base_model'] or None
        if 'variant' in patch or selection['family_changed']:
            ds.train_variant = selection['variant']
        if selection['disable_slider']:
            ds.train_slider = selection['train_slider']
    ds.train_settings = json.dumps(cur) if cur else None
    if requested_training_mode is not None:
        ds.training_mode = requested_training_mode
    try:
        fds.db.session.commit()
    except Exception:
        # A family transition and Slider disable can touch several columns.
        # Never leave their in-memory half-state visible after a failed commit.
        fds.db.session.rollback()
        raise
    return effective_train_settings(ds)


# Every key update_train_settings knows how to validate — KEEP IN SYNC when a
# new expert lever is added above. This is what makes presets schema-tolerant:
# a preset key outside this list is IGNORED (and reported), never fatal.
TRAIN_SETTING_KEYS = ('rank', 'resolution', 'save_every', 'max_step_saves',
                      'sample_every', 'sample_prompts', 'dropout', 'alpha',
                      'timestep_type', 'optimizer', 'lr_scheduler', 'warmup',
                      'grad_accum', 'network_type', 'lokr_factor',
                      'lokr_full_rank', 'conv', 'conv_alpha', 'ema',
                      'content_or_style', 'do_differential_guidance',
                      'differential_guidance_scale', 'dual_captions',
                      'mask_faces', 'masked', 'learning_rate', 'weight_decay',
                      'loss_type', 'qtype', 'qtype_te', 'layer_offloading',
                      'layer_offloading_transformer_percent',
                      'layer_offloading_text_encoder_percent',
                      'cache_text_embeddings', 'save_dtype',
                      'preset_steps_per_image', 'preset_steps_min',
                      'preset_steps_max', 'preset_steps_fixed',
                      # The unlocked half of the dense recipe. Present here so a
                      # shared/exported preset can carry it too — the values are
                      # bounded, and a LoRA run simply never reads them.
                      *DENSE_SETTING_KEYS,
                      *_MEMORY_SETTING_KEYS)

# The ONLY settings a resume/continue may change. ai-toolkit rebuilds the job
# config from scratch on every launch, so a re-read setting is honored on resume —
# but the LoRA weights being resumed have a fixed shape (rank/alpha/network) and
# the run has a fixed training regime (optimizer, schedule, resolution). Changing
# any of those mid-run either fails to load the checkpoint (shape mismatch) or
# silently trains a different recipe than the one that produced the weights.
# Cadence and preview prompts touch neither: save/sample cadence only decides WHEN
# to snapshot, and sample prompts only affect the preview images. timestep_type is
# the deliberate exception: it changes no weight shape and re-weighting the noise
# levels ON PURPOSE is a documented continuation recipe (train balanced, then
# resume low-noise-leaning to polish fine texture) — the user picks it explicitly
# in the Continue dialog, so it is a stated intent, never a silent drift. SDXL
# ignores it (flowmatch weighting does not apply there) — a harmless no-op.
# lr_factor is the second deliberate exception: it scales the LR of the continuation
# only (½ polish / ⅒ gentle finish — the LR pendant of the low-noise timestep
# recipe), touches no weight shape, and resolves to an explicit `learning_rate`
# via resolve_resume_lr (refused on a Prodigy run, whose LR is self-adaptive).
#
# The memory-saving levers (quantize / quantize_te / low_vram) are deliberately
# NOT in this list, and that is a decision, not an oversight. They ARE harmless to
# a resume — VÉRIFIÉ: quantisation is applied to the BASE modules, which are frozen
# (`param.requires_grad = False` + `freeze(orig_module)` in ai-toolkit's
# util/quantize.py), while the resumed checkpoint holds only the LoRA weights,
# whose shape is fixed by `network` (rank/alpha/type) and untouched here. But this
# tuple is the whitelist of what the ▶ Continue DIALOG may send, and the dialog has
# no control for them; the persisted Advanced-options value is already re-read on
# every resume (ai-toolkit rebuilds the job config from scratch), so a user who
# turns quantisation off and hits ▶ Continue already gets it. Listing them would
# add untested surface with no caller.
RESUME_SAFE_SETTING_KEYS = ('save_every', 'sample_every', 'sample_prompts',
                            'timestep_type', 'lr_factor')


def validate_resume_overrides(overrides) -> dict:
    """Validate the safe-subset settings a continue/resume is allowed to change
    (see RESUME_SAFE_SETTING_KEYS) and return a cleaned patch. Raises ValueError
    on any forbidden key or bad value — a resume must refuse to touch anything
    that would mismatch the checkpoint's weights or change its training regime.
    Value validation mirrors update_train_settings so local (persisted) and cloud
    (per-run snapshot) apply identical rules."""
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError('overrides must be an object')
    forbidden = [k for k in overrides if k not in RESUME_SAFE_SETTING_KEYS]
    if forbidden:
        raise ValueError(
            'these settings cannot change when continuing a run — they must match '
            f'the checkpoint being resumed: {", ".join(sorted(forbidden))}')
    patch = {}
    if 'save_every' in overrides:
        v = overrides['save_every']
        if v not in _SAVE_CHOICES:
            raise ValueError(f'save_every must be one of {_SAVE_CHOICES}')
        patch['save_every'] = v
    if 'sample_every' in overrides:
        v = overrides['sample_every']
        if v not in _SAMPLE_EVERY_CHOICES:
            raise ValueError(f'sample_every must be one of {_SAMPLE_EVERY_CHOICES}')
        patch['sample_every'] = v
    if 'sample_prompts' in overrides:
        v = overrides['sample_prompts']
        if isinstance(v, str):
            v = v.splitlines()                              # multi-line convenience
        if v in (None, ''):
            patch['sample_prompts'] = None                  # reset to kind-aware defaults
        elif isinstance(v, list):
            cleaned = [str(x).strip() for x in v if str(x).strip()][:_MAX_SAMPLE_PROMPTS]
            patch['sample_prompts'] = cleaned or None
        else:
            raise ValueError('sample_prompts must be a list of strings (or empty to reset)')
    if 'timestep_type' in overrides:
        v = overrides['timestep_type']
        if v not in _TIMESTEP_TYPE_CHOICES:
            raise ValueError(f'timestep_type must be one of {_TIMESTEP_TYPE_CHOICES}')
        patch['timestep_type'] = v
    if 'lr_factor' in overrides:
        # A FACTOR (½/⅒) scaling the run's current LR, kept as-is in the patch: the
        # caller (continue_training / continue_cloud_run) resolves it to an absolute
        # `learning_rate` against the run's own optimizer + LR, where the Prodigy
        # refusal lives (resolve_resume_lr). Here we only vet the value; 1 / keep
        # is dropped so a no-op never persists a redundant learning_rate.
        v = overrides['lr_factor']
        if v in (None, 1, 1.0):
            pass                                            # keep current — no override
        elif v in _RESUME_LR_FACTORS:
            patch['lr_factor'] = v
        else:
            raise ValueError(f'lr_factor must be one of {_RESUME_LR_FACTORS} (or 1 to keep current)')
    return patch

# Built-in quick presets: shipped with the app (every install sees them),
# read-only, versioned with the code. Every (family × dataset kind) has one
# general-purpose quick preset; narrowly-scoped, source-labelled recipes may sit
# alongside it. Character locks an identity, Style absorbs a look (route-owned
# catalogue), Concept generalizes an object/pose/composition. Every value is SOURCED —
# research vault first (Tech-IA notes), the installed ai-toolkit's own defaults
# second (ui/src/app/jobs/new/options.ts + config/examples), 2026 community
# consensus third — never intuition; per-preset comments carry the source.
# Steps stay adaptive (recommended_steps owns them per kind). A source-labelled
# recipe may pin the family-default learning rate explicitly when that is part of
# its published configuration. A test asserts every builtin applies with zero
# ignored/rejected keys, so a drifting choice-list can't silently break them.

# Identity AND flexibility probes — overfit (waxy skin, frozen pose) shows here
# first. One probe sheet per checkpoint: on character sets the quality comes
# from picking the earliest checkpoint that holds the identity, not from
# exotic hyper-parameters.
_CHARACTER_SAMPLE_PROMPTS = [
    '{trigger}, close-up portrait, neutral expression, soft studio light',
    '{trigger}, headshot, golden hour sunlight, slight smile',
    '{trigger}, bust shot, profile view, window light',
    '{trigger}, full body, walking outdoors in a park, casual jeans and t-shirt',
    '{trigger}, full body, elegant evening dress, dim moody lighting',
    '{trigger}, sitting at a cafe table, laughing, candid photo',
    '{trigger}, sportswear, stretching in a gym, harsh fluorescent light',
    '{trigger}, wide shot, standing on a beach at dusk, wind in hair',
]

# Probes exercise the concept across framings, contexts and lighting: a concept
# LoRA that only reproduces its training context has overfit.
_CONCEPT_SAMPLE_PROMPTS = [
    '{trigger}',
    '{trigger}, close-up, high detail, sharp focus',
    '{trigger}, wide shot showing the full scene',
    '{trigger}, in an unusual setting, outdoors',
    '{trigger}, soft natural window light',
    '{trigger}, night scene, artificial light',
    '{trigger}, seen from a high angle',
    '{trigger}, cinematic composition, shallow depth of field',
]


def _character_preset_settings(rank, alpha, resolution='768,1024',
                               timestep_type=None, ema=None):
    """Character quick-preset baseline: a save + probe sheet every 250 steps
    with ten snapshots kept — all sweet-spot candidates for the earliest
    checkpoint that holds the likeness (save/sample 250 = the ai-toolkit
    canonical cadence, train_lora_flux_24gb.yaml)."""
    out = {
        'rank': rank,
        'alpha': alpha,
        'resolution': resolution,
        'save_every': 250,
        'max_step_saves': 10,
        'sample_every': 250,
        'sample_prompts': list(_CHARACTER_SAMPLE_PROMPTS),
    }
    if timestep_type:
        out['timestep_type'] = timestep_type
    if ema:
        out['ema'] = ema
    return out


def _concept_preset_settings(rank, alpha, resolution='768,1024',
                             timestep_type=None):
    """Concept quick-preset baseline. Runs are long (sub-linear 475·√n steps,
    authoritative in recommended_steps), so save/sample every 500 halves the
    preview GPU cost while max_step_saves keeps the N most RECENT saves
    (ai-toolkit deletes the oldest): 10×500 spans the last 5000 steps — the
    whole run at the small anchor, the second half at the large one. Alpha is
    rank/2 across families: the concept research (vault 2026-06-22) recommends
    'Alpha dim/2' for object/pose/composition — generalize, don't memorize."""
    out = {
        'rank': rank,
        'alpha': alpha,
        'resolution': resolution,
        'save_every': 500,
        'max_step_saves': 10,
        'sample_every': 500,
        'sample_prompts': list(_CONCEPT_SAMPLE_PROMPTS),
    }
    if timestep_type:
        out['timestep_type'] = timestep_type
    return out


BUILTIN_TRAIN_PRESETS = [
    # --- Character (one per family) --------------------------------------
    # Historical ID kept stable (tests, muscle memory). rank/alpha 32/32:
    # "Every Krea-2 source uses 32/32" (vault ai-toolkit settings 2026-07-10,
    # HIGH confidence; musubi + RunComfy agree). timestep_type deliberately
    # NOT pinned: it falls to the family default (linear) because linear-vs-
    # sigmoid is the vault's one unresolved Krea conflict — "don't ship blind".
    {
        'id': 'builtin-krea-character',
        'name': 'Krea 2 · Character',
        'train_type': 'krea',
        'dataset_kind': 'character',
        'variants': ['base', 'raw', 'turbo'],
        'builtin': True,
        'description': 'Every Krea-2 source agrees on rank 32/32; multi-scale '
                       '768/1024 with a probe sheet every 250 steps to catch '
                       'the identity sweet-spot early.',
        'settings': _character_preset_settings(32, 32),
    },
    # Community report, not a universal result:
    # https://www.reddit.com/r/StableDiffusion/comments/1v2vsqm/
    # almost_perfect_likeness_in_750_steps_krea_2_lokr/
    # The linked Pastebin configuration was later deleted. The post specifies
    # LoKr factor 16 but not linear rank/alpha, so LDS retains its verified Krea
    # Character 32/32 baseline instead of inventing missing values. `base` is
    # Krea-2-Raw in LDS; Turbo is excluded from THIS preset's variant list
    # because the community report was never validated on it, not because
    # Turbo LoRA training is broken. Full-model/dense training likewise refuses
    # Turbo because dense-on-Turbo is untested, unlike Turbo LoRA training
    # which works fine — see _assert_full_transformer_recipe.
    {
        'id': 'builtin-krea-raw-lokr-likeness',
        'name': 'Krea 2 Raw · LoKr likeness',
        'train_type': 'krea',
        'dataset_kind': 'character',
        'variants': ['base', 'raw'],
        'builtin': True,
        'community': True,
        'description': 'Community Krea-2 Raw starting recipe: LoKr factor 16, '
                       '768 px, Automagic v2, sigmoid, Balanced and Differential '
                       'Guidance ×3. Inspect the early checkpoints; it is not a '
                       'guarantee for every dataset.',
        'settings': {
            **_character_preset_settings(32, 32, resolution='768', timestep_type='sigmoid'),
            'network_type': 'lokr',
            'lokr_factor': 16,
            'optimizer': 'automagic2',
            'learning_rate': 1e-4,
            'content_or_style': 'balanced',
            'do_differential_guidance': True,
            'differential_guidance_scale': 3.0,
        },
    },
    {
        'id': 'builtin-krea-raw-character-balanced',
        'name': 'Krea 2 Raw · Character Balanced',
        'train_type': 'krea',
        'dataset_kind': 'character',
        'variants': ['base', 'raw'],
        'builtin': True,
        'approved': True,
        'community': True,
        'confidence': 'medium',
        'evidence_label': 'community-tested',
        'source_url': (
            'https://www.reddit.com/r/StableDiffusion/comments/1upiocf/'
            'character_loras_with_krea2_again/'),
        'recommended_images': {'min': 40, 'max': 60, 'target': 50},
        'recommended_steps': {'per_image': 50, 'min': 2000, 'max': 3000},
        'checkpoint_targets': [1500, 2000, 2500, 3000],
        'caption_guidance': (
            'Describe visible identity, pose, framing and lighting; keep the trigger '
            'consistent and avoid inferred traits.'),
        'limitations': [
            'The community source did not publish rank or alpha; LDS therefore uses '
            'its Krea Character defaults (32/32).',
            'Community-tested starting point, not a guarantee for every face or dataset.',
        ],
        'description': (
            'Community Krea Raw character recipe using Automagic3, sigmoid and '
            'Balanced at 1024. The source did not publish rank/alpha, so LDS '
            'transparently falls back to its 32/32 Krea defaults.'),
        'settings': {
            'resolution': '1024',
            'save_every': 500,
            'max_step_saves': 10,
            'sample_every': 500,
            'sample_prompts': list(_CHARACTER_SAMPLE_PROMPTS),
            'optimizer': 'automagic3',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'timestep_type': 'sigmoid',
            'content_or_style': 'balanced',
            'preset_steps_per_image': 50,
            'preset_steps_min': 2000,
            'preset_steps_max': 3000,
        },
    },
    {
        'id': 'builtin-krea-raw-character-lokr-fast',
        'name': 'Krea 2 Raw · Character LoKr Fast',
        'train_type': 'krea',
        'dataset_kind': 'character',
        'variants': ['base', 'raw'],
        'builtin': True,
        'approved': True,
        'community': True,
        'confidence': 'medium',
        'evidence_label': 'community-tested',
        'source_url': (
            'https://www.reddit.com/r/StableDiffusion/comments/1uyk9fz/'
            'struggling_with_krea2_lora_training_looking_for/'),
        'recommended_images': {'min': 20, 'max': 40, 'target': 30},
        'recommended_steps': {'per_image': 100, 'min': 1500, 'max': 3000},
        'checkpoint_targets': [1500, 2000, 2500, 3000],
        'caption_guidance': (
            'Use concise identity captions with varied pose, distance and lighting; '
            'keep the same trigger in every caption.'),
        'limitations': [
            'Full-rank LoKr produces a different checkpoint topology from normal LoKr.',
            'Fast convergence can overfit; compare the 250-step checkpoints.',
        ],
        'description': (
            'Fast Krea Raw full-rank LoKr recipe: factor 4, Automagic2, EMA 0.99, '
            'MSE and Differential Guidance ×3, with cached text embeddings.'),
        'settings': {
            'resolution': '1024',
            'save_every': 250,
            'max_step_saves': 10,
            'sample_every': 250,
            'sample_prompts': list(_CHARACTER_SAMPLE_PROMPTS),
            'network_type': 'lokr',
            'lokr_full_rank': True,
            'lokr_factor': 4,
            'optimizer': 'automagic2',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'timestep_type': 'sigmoid',
            'content_or_style': 'balanced',
            'loss_type': 'mse',
            'ema': 0.99,
            'do_differential_guidance': True,
            'differential_guidance_scale': 3.0,
            'cache_text_embeddings': True,
            'preset_steps_per_image': 100,
            'preset_steps_min': 1500,
            'preset_steps_max': 3000,
        },
    },
    {
        'id': 'builtin-krea-raw-style-compact',
        'name': 'Krea 2 Raw · Style Compact',
        'train_type': 'krea',
        'dataset_kind': 'style',
        'variants': ['base', 'raw'],
        'builtin': True,
        'approved': True,
        'community': True,
        'confidence': 'medium',
        'evidence_label': 'community-tested',
        'source_url': (
            'https://www.reddit.com/r/StableDiffusion/comments/1uzuypa/'
            'made_another_style_lora_on_krea2/'),
        'recommended_images': {'min': 37, 'max': 70, 'target': 50},
        'recommended_steps': {'fixed': 2250},
        'checkpoint_targets': [1000, 1500, 2000, 2250],
        'caption_guidance': (
            'Caption image content only: never name the style, and keep each caption '
            'under 50 words.'),
        'limitations': [
            'Fixed 2250-step policy is tuned for the reported 37–70 image range.',
            'Rank 16 is compact and may underfit unusually broad styles.',
        ],
        'description': (
            'Compact rank-16 Krea Raw style LoRA at 512/768, fixed at 2250 '
            'steps with content-only probes every 250 steps. Alpha is derived.'),
        'settings': {
            'rank': 16,
            'resolution': '512,768',
            'save_every': 250,
            'max_step_saves': 10,
            'sample_every': 250,
            'sample_prompts': [
                'a woman reading in a sunlit cafe',
                'a city street at night, rain, neon reflections',
                'a mountain landscape, wide shot, morning mist',
                'a still life of fruit on a wooden table',
                'a cozy interior, warm lamp light',
                'a runner mid-stride on a bridge, motion',
                'a cat sleeping on a windowsill',
                'a modern building facade, strong shadows',
            ],
            'preset_steps_fixed': 2250,
        },
    },
    {
        'id': 'builtin-krea-raw-concept-16gb',
        'name': 'Krea 2 Raw · General Concept (16 GB reported)',
        'train_type': 'krea',
        'dataset_kind': 'concept',
        'variants': ['base', 'raw'],
        'builtin': True,
        'approved': True,
        'community': True,
        'confidence': 'medium',
        'evidence_label': 'community-tested',
        'source_url': (
            'https://www.reddit.com/r/StableDiffusion/comments/1v9yl1u/'
            'krea_2_lora_training_the_very_easy_guide_for_16gb/'),
        'recommended_images': {'min': 50, 'max': 60, 'target': 55},
        'recommended_steps': {'per_image': 55, 'min': 3000, 'max': 3250},
        'checkpoint_targets': [2000, 2500, 3000],
        'caption_guidance': (
            'Describe the concept and its visible context precisely; vary composition, '
            'viewpoint and lighting rather than repeating one caption.'),
        'limitations': [
            'Reported on a 16 GB setup; memory use is not guaranteed across drivers '
            'or ai-toolkit revisions.',
            'Layer offloading trades speed for memory and does not replace the normal '
            'VRAM preflight.',
        ],
        'description': (
            'Community Krea Raw concept recipe reported on 16 GB: Automagic3, '
            'int8 quantization, 50% layer offloading and cached text embeddings. '
            'Reported fit is not guaranteed.'),
        'settings': {
            'resolution': '1024',
            'save_every': 500,
            'max_step_saves': 10,
            'sample_every': 500,
            'sample_prompts': list(_CONCEPT_SAMPLE_PROMPTS),
            'optimizer': 'automagic3',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'timestep_type': 'sigmoid',
            'content_or_style': 'balanced',
            'low_vram': True,
            'layer_offloading': True,
            'layer_offloading_transformer_percent': 0.5,
            'layer_offloading_text_encoder_percent': 0.5,
            'cache_text_embeddings': True,
            'qtype': 'int8',
            'qtype_te': 'int8',
            'preset_steps_per_image': 55,
            'preset_steps_min': 3000,
            'preset_steps_max': 3250,
        },
    },
    {
        'id': 'builtin-zimage-turbo-character-balanced',
        'name': 'Z-Image Turbo · Character Balanced',
        'train_type': 'zimage',
        'dataset_kind': 'character',
        'variants': ['turbo'],
        'builtin': True,
        'approved': True,
        'community': True,
        'confidence': 'medium',
        'evidence_label': 'community-tested',
        'source_url': (
            'https://www.reddit.com/r/StableDiffusion/comments/1q1ahx9/'
            'some_zimageturbo_training_presets_for_12gb_vram/'),
        'recommended_images': {'min': 30, 'max': 40, 'target': 35},
        'recommended_steps': {'per_image': 100, 'min': 3000, 'max': 4000},
        'checkpoint_targets': [3000, 3250, 3500, 3750, 4000],
        'caption_guidance': (
            'Keep the trigger stable; describe pose, framing, expression and lighting '
            'without repeating fixed identity traits.'),
        'limitations': [
            'Turbo-only recipe; do not apply its float8/offloading balance to Base or '
            'De-Turbo checkpoints.',
            'Conv LoRA increases adapter size relative to the linear-only default.',
        ],
        'description': (
            'Z-Image Turbo character recipe: LoRA 32/32 plus Conv 16/16, '
            'sigmoid/Balanced, 512/768, float8 model and text encoder, BF16 saves.'),
        'settings': {
            'rank': 32,
            'alpha': 32,
            'conv': 16,
            'conv_alpha': 16,
            'resolution': '512,768',
            'save_every': 250,
            'max_step_saves': 10,
            'sample_every': 250,
            'sample_prompts': list(_CHARACTER_SAMPLE_PROMPTS),
            'learning_rate': 1e-4,
            'timestep_type': 'sigmoid',
            'content_or_style': 'balanced',
            'qtype': 'float8',
            'qtype_te': 'float8',
            'save_dtype': 'bf16',
            'cache_text_embeddings': True,
            'preset_steps_per_image': 100,
            'preset_steps_min': 3000,
            'preset_steps_max': 4000,
        },
    },
    # 32/32 is the AI-Toolkit-community default and the "lower-regret choice
    # for hard faces" (vault 2026-07-10; options.ts + neurocanvas ship 32) —
    # drop rank to 16 manually for small clean frontal sets. sigmoid pinned:
    # "matches Ostris' subject guidance exactly" (vault, confirmed ✓).
    {
        'id': 'builtin-character-zimage',
        'name': 'Z-Image · Character',
        'train_type': 'zimage',
        'dataset_kind': 'character',
        'variants': [],
        'builtin': True,
        'description': 'Rank 32/32 (community default, lower-regret for hard '
                       'faces) with sigmoid timesteps per Ostris subject '
                       'guidance; probe every 250 steps and pick the earliest '
                       'checkpoint that holds the likeness.',
        'settings': _character_preset_settings(32, 32, timestep_type='sigmoid'),
    },
    # SDXL keeps its deliberate half-strength alpha: "SDXL's existing 32/16 is
    # a deliberate valid choice — keep it" (vault 2026-07-10). Native 1024
    # single-scale (vault Z-Image-vs-SDXL 2026-06-14). No timestep_type: SDXL
    # is ddpm — flowmatch weighting does not apply (options.ts disables it).
    {
        'id': 'builtin-character-sdxl',
        'name': 'SDXL · Character',
        'train_type': 'sdxl',
        'dataset_kind': 'character',
        'variants': [],
        'builtin': True,
        'description': 'The researched SDXL recipe: rank 32 with half-strength '
                       'alpha 16 at native 1024 — finer detail capture, with '
                       '250-step probes for checkpoint picking.',
        'settings': _character_preset_settings(32, 16, resolution='1024'),
    },
    # The canonical ai-toolkit FLUX recipe, verbatim from
    # config/examples/train_lora_flux_24gb.yaml: lora 16/16, EMA on at 0.99
    # ("Recommended to leave on"), save/sample 250. sigmoid = Ostris' subject
    # guidance ("for just subject, change to sigmoid"), confirmed by the vault.
    {
        'id': 'builtin-character-flux1',
        'name': 'FLUX.1 dev · Character',
        'train_type': 'flux',
        'dataset_kind': 'character',
        'variants': [],
        'builtin': True,
        'description': "ai-toolkit's canonical FLUX recipe: rank 16/16 with "
                       'sigmoid timesteps and EMA 0.99 (shipped default), '
                       'probing every 250 steps.',
        'settings': _character_preset_settings(16, 16, timestep_type='sigmoid',
                                               ema=0.99),
    },
    # No vault coverage for Klein training yet. rank 16/16 = the family default
    # and the community starting point ("start 16, try 32 if underfitting" —
    # RunComfy Klein guide); BFL's official page gives step/LR envelopes only.
    # sigmoid is EXTRAPOLATED from generic ai-toolkit subject guidance
    # (weighted default → sigmoid for character) — not Klein-verified. Both
    # sizes share hyper-parameters; only VRAM differs (apatero 4B-vs-9B).
    {
        'id': 'builtin-character-klein',
        'name': 'FLUX.2 Klein · Character',
        'train_type': 'flux2klein',
        'dataset_kind': 'character',
        'variants': ['4b', '9b'],
        'builtin': True,
        'description': 'Community starting point for Klein identity: rank '
                       '16/16, sigmoid timesteps (generic subject guidance — '
                       'no Klein-specific study yet), 250-step probes.',
        'settings': _character_preset_settings(16, 16, timestep_type='sigmoid'),
    },
    # No Anima-specific study exists yet (model shipped mid-2026): extrapolated
    # from ai-toolkit's own defaults (options.ts entry 'anima', PR #860) — rank 32
    # (defaultLinearRank) with weighted timesteps (the entry's canonical default,
    # NOT subject-tuned sigmoid). Drop rank to 16 manually for small clean sets.
    {
        'id': 'builtin-character-anima',
        'name': 'Anima · Character',
        'train_type': 'anima',
        'dataset_kind': 'character',
        'variants': [],
        'builtin': True,
        'description': "ai-toolkit's Anima defaults: rank 32/32 with weighted "
                       'timesteps — no Anima-specific study yet, extrapolated '
                       'from options.ts; probe every 250 steps.',
        'settings': _character_preset_settings(32, 32, timestep_type='weighted'),
    },
    # --- Concept / composition (one per family) --------------------------
    # Historical ID kept stable. rank 16 / alpha 8: concept research (vault
    # 2026-06-22) — object/concept rank 16-32 with alpha dim/2. weighted
    # pinned: sigmoid is the app's zimage default but it is subject-tuned;
    # weighted is the arch-canonical options.ts default and the community's
    # non-character recommendation.
    {
        'id': 'builtin-concept',
        'name': 'Z-Image · Concept',
        'train_type': 'zimage',
        'dataset_kind': 'concept',
        'variants': [],
        'builtin': True,
        'description': 'Generalize, don\'t memorize: rank 16 with half alpha '
                       'and weighted timesteps, probing across framings and '
                       'lighting every 500 steps.',
        'settings': _concept_preset_settings(16, 8, timestep_type='weighted'),
    },
    # Concept research: simple concepts rank 4-8, complex 16-32 → 16 is the
    # mid-choice; alpha dim/2 (vault 2026-06-22). Native 1024, no flowmatch
    # timestep on SDXL.
    {
        'id': 'builtin-concept-sdxl',
        'name': 'SDXL · Concept',
        'train_type': 'sdxl',
        'dataset_kind': 'concept',
        'variants': [],
        'builtin': True,
        'description': 'Rank 16 with half alpha 8 at native 1024 — enough '
                       'capacity for objects and poses without absorbing the '
                       'whole scene; probes every 500 steps.',
        'settings': _concept_preset_settings(16, 8, resolution='1024'),
    },
    # The vault is EXPLICIT that no Krea concept recipe exists ("seuls
    # style/character/object documentés" — Krea-2 note 2026-06-29), so this
    # extrapolates the family canon: rank 32 (every Krea-2 source) × the
    # generic concept alpha dim/2 rule. linear pinned = the Krea-canonical
    # timestep (ai-toolkit options.ts l.1050 + RunComfy Krea recipe).
    {
        'id': 'builtin-concept-krea',
        'name': 'Krea 2 · Concept',
        'train_type': 'krea',
        'dataset_kind': 'concept',
        'variants': [],
        'builtin': True,
        'description': 'No published Krea concept recipe exists — extrapolated '
                       "from the family's rank-32 canon with half alpha 16 and "
                       "Krea's linear timesteps.",
        'settings': _concept_preset_settings(32, 16, timestep_type='linear'),
    },
    # rank 16 = the canonical FLUX example dimension; alpha dim/2 per the
    # concept research. 'shift' is the repo's only composition guidance
    # (train_lora_flex2_24gb.yaml: "shift works well for training fast and
    # learning composition and style; for just subject → sigmoid") — Flex
    # lineage, flagged as such.
    {
        'id': 'builtin-concept-flux1',
        'name': 'FLUX.1 dev · Concept',
        'train_type': 'flux',
        'dataset_kind': 'concept',
        'variants': [],
        'builtin': True,
        'description': "Rank 16 with half alpha 8; 'shift' timesteps per "
                       "Ostris ('works well for learning composition and "
                       "style'), probing every 500 steps.",
        'settings': _concept_preset_settings(16, 8, timestep_type='shift'),
    },
    # No Klein concept source exists anywhere (flagged by the 2026 web sweep):
    # extrapolated from the Klein style side — composition wants capacity, so
    # rank 32 with the generic alpha dim/2 rule, and Klein's canonical
    # weighted timesteps (options.ts) rather than subject-tuned sigmoid.
    {
        'id': 'builtin-concept-klein',
        'name': 'FLUX.2 Klein · Concept',
        'train_type': 'flux2klein',
        'dataset_kind': 'concept',
        'variants': ['4b', '9b'],
        'builtin': True,
        'description': 'No Klein-specific concept research yet — extrapolated '
                       'from Klein style: rank 32, half alpha 16, weighted '
                       'timesteps, 500-step probes.',
        'settings': _concept_preset_settings(32, 16, timestep_type='weighted'),
    },
    # No Anima concept source exists: extrapolate the family canon (rank 32, the
    # options.ts defaultLinearRank) with the generic concept alpha dim/2 rule and
    # Anima's canonical weighted timesteps — same reasoning as the Krea/Klein
    # concept presets, flagged as extrapolated.
    {
        'id': 'builtin-concept-anima',
        'name': 'Anima · Concept',
        'train_type': 'anima',
        'dataset_kind': 'concept',
        'variants': [],
        'builtin': True,
        'description': 'No Anima-specific concept research yet — extrapolated '
                       "from ai-toolkit's rank-32 default with half alpha 16 "
                       'and weighted timesteps, 500-step probes.',
        'settings': _concept_preset_settings(32, 16, timestep_type='weighted'),
    },
    # Legacy generic Style alias. The API hides it from GET and resolves its ID
    # to a family-specific built-in at apply time. Keep the raw entry only for
    # older callers/tests that imported BUILTIN_TRAIN_PRESETS directly. Style
    # steps are family/variant-aware (50 steps/image with researched envelopes),
    # not the Concept formula. Content-only previews contain no activation token.
    {
        'id': 'builtin-style',
        'name': 'Style — recommended',
        'train_type': 'zimage',
        'builtin': True,
        'settings': {
            'resolution': '768,1024',
            'save_every': 500,
            'max_step_saves': 10,
            'sample_every': 500,
            'sample_prompts': [
                'a woman reading in a sunlit cafe',
                'a city street at night, rain, neon reflections',
                'a mountain landscape, wide shot, morning mist',
                'a still life of fruit on a wooden table',
                'a cozy interior, warm lamp light',
                'a runner mid-stride on a bridge, motion',
                'a cat sleeping on a windowsill',
                'a modern building facade, strong shadows',
            ],
        },
    },
]


def snapshot_train_settings(user_id, dataset_id) -> dict:
    """The dataset's RAW explicit settings (what a preset captures) — only the
    keys the user actually changed, not the effective/derived view. Invalid
    legacy non-integer values for numeric controls are omitted so a newly saved
    preset cannot perpetuate bool/float numeric ambiguity."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    settings = _train_settings(ds)
    return {key: value for key, value in settings.items()
            if key in TRAIN_SETTING_KEYS and (
                (key == 'alpha' and _numeric_choice(value, _ALPHA_CHOICES) is not None)
                or (key == 'grad_accum'
                    and _numeric_choice(value, _GRAD_ACCUM_CHOICES) is not None)
                or key not in ('alpha', 'grad_accum'))}


def apply_train_settings_dict(user_id, dataset_id, settings: dict, *,
                              preset_scope=None):
    """REPLACE the dataset's explicit settings with a preset's dict, running
    every key through the validated update_train_settings path. Content is
    never fatal: unknown keys (newer/older app versions) are ignored, invalid
    values collected — both reported so the UI can say what didn't land.
    Returns (effective_settings, ignored_keys, rejected)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    ignored = sorted(k for k in settings if k not in TRAIN_SETTING_KEYS)
    rejected = []
    candidate = {}                    # a preset REPLACES, it doesn't overlay
    for k in TRAIN_SETTING_KEYS:
        if k not in settings:
            continue
        try:
            update_train_settings(user_id, dataset_id, {k: settings[k]},
                                  _settings=candidate)
        except ValueError as e:
            rejected.append({'key': k, 'reason': str(e)})
    if isinstance(preset_scope, dict):
        variants = [
            str(v).strip().lower() for v in (preset_scope.get('variants') or [])
            if str(v).strip()
        ]
        selected_variant = str(
            preset_scope.get('selected_variant') or '').strip().lower()
        if variants and selected_variant in variants:
            # Applying a variant-scoped preset and showing its effective values
            # must be one atomic server truth; otherwise the still-persisted old
            # variant would immediately make the just-applied recipe inert.
            ds.train_variant = selected_variant
        candidate[_ACTIVE_PRESET_SCOPE_KEY] = {
            'preset_id': preset_scope.get('preset_id'),
            'train_type': str(
                preset_scope.get('train_type') or ds.train_type or 'zimage'
            ).strip().lower(),
            'dataset_kind': str(
                preset_scope.get('dataset_kind') or getattr(ds, 'kind', None)
                or 'character'
            ).strip().lower(),
            'variants': variants,
            'keys': sorted(k for k in candidate if k in TRAIN_SETTING_KEYS),
        }
    # A concurrent Train observes either the old preset or this fully validated
    # replacement — never the previous clear plus a prefix of its keys.
    ds.train_settings = json.dumps(candidate) if candidate else None
    fds.db.session.commit()
    return effective_train_settings(ds), ignored, rejected


def _dest_base_tag(ds, base_model=_PERSISTED, family=None,
                   variant=_PERSISTED) -> str:
    """Deployment-name suffix, family-aware. Like _base_tag, but for Krea
    (which has no base column - always Krea-2-Turbo) falls back to a constant tag
    so the LoRA carries the model name like SDXL does. `family` override permet au
    sélecteur UI de router vers Krea même si le train_type persisté diffère."""
    tag = _base_tag(ds) if base_model is _PERSISTED else _base_tag_for(base_model)
    fam = _train_type(ds, family)
    if fam == 'zimage':
        # All three recipes are incompatible, including official Turbo versus
        # the old suffix-less legacy folder. Give every recipe a distinct
        # folder/deployed filename. Custom
        # bases already carry their base tag; suffix the recipe there as well so
        # switching adapter policy can never auto-resume a different recipe.
        var = (getattr(ds, 'train_variant', None) if variant is _PERSISTED
               else variant) or 'turbo'
        var = str(var).lower()
        if var == 'turbo':
            tag += '_turbo' if tag else '_Z-Image-Turbo'
        elif var == 'base':
            tag += '_base' if tag else '_Z-Image-Base'
        elif var == 'deturbo':
            tag += '_deturbo' if tag else '_Z-Image-De-Turbo'
    if not tag and fam == 'krea':
        # Raw and Turbo are DIFFERENT base checkpoints → distinct tags so their
        # run folders / deployed LoRA names never collide (same trigger, same
        # family, but incompatible weights would otherwise share a folder).
        tag = _base_tag_for(
            'Krea-2-Raw' if _krea_is_raw(ds, variant) else KREA_BASE_LABEL)
    # Même garde pour Flux : sa base officielle donne un tag vide, qui télescoperait
    # un run Z-Image officiel du même trigger (même dossier `u{user}_{trigger}` →
    # ai-toolkit auto-resume le mauvais run, poids mélangés). Le tag `_FLUX-1-dev`
    # isole le run et le LoRA déployé de la famille Z-Image.
    if not tag and fam == 'flux':
        tag = _base_tag_for(FLUX_BASE_LABEL)
    # FLUX.2 Klein : même garde, mais le tag encode AUSSI la variante (4B vs 9B
    # sont deux checkpoints incompatibles) — sans ça, deux runs du même trigger
    # sur les deux tailles partageraient dossier de run et nom déployé.
    if not tag and fam == 'flux2klein':
        tag = _base_tag_for(
            FLUX2KLEIN_BASE_LABELS[
                '9b' if _flux2klein_is_9b(ds, variant) else '4b'])
    # Anima : même garde que Flux — sa base officielle unique donne un tag vide qui
    # télescoperait un run Z-Image officiel du même trigger. Le tag `_Anima-Base`
    # isole le run et le LoRA déployé.
    if not tag and fam == 'anima':
        tag = _base_tag_for(ANIMA_BASE_LABEL)
    return tag + _custom_combo_hash(ds, base_model, family)


def _custom_combo_hash(ds, base_model=_PERSISTED, family=None) -> str:
    """Short hash of the full (custom weights, VAE, TE) TRIPLET, appended to the
    run tag so two different custom combos NEVER share a run folder (ai-toolkit
    auto-resumes from the folder — a shared one would blend incompatible weights).
    Empty when nothing custom is in play, so every official/whitelist run keeps
    its exact historical folder name. VAE/TE only count for SDXL (the only family
    that honours them) — a stale value on another family can't perturb its tag."""
    fam = _train_type(ds, family)
    weights = getattr(ds, 'train_base_model', None) if base_model is _PERSISTED else base_model
    vae = getattr(ds, 'train_vae_path', None) if fam in VAE_TE_OVERRIDE_FAMILIES else None
    te = getattr(ds, 'train_te_path', None) if fam in VAE_TE_OVERRIDE_FAMILIES else None
    # Z-Image custom bases can be relative ComfyUI model names as well as
    # absolute filesystem paths. Hash the complete normalized identifier in
    # both cases; basename-only tags would collide for two same-named files in
    # different folders.
    zimage_custom = fam == 'zimage' and bool(str(weights or '').strip())
    if not (zimage_custom or _is_custom_weights(weights) or vae or te):
        return ''
    norm_weights = (os.path.normcase(os.path.normpath(str(weights)))
                    if weights else '')
    raw = f'{norm_weights}|{vae or ""}|{te or ""}'
    return '_h' + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]


# Source-run tag appended to a DEPLOYED LoRA's name so two runs that produce the
# same trigger/step/base/dataset-version never collapse onto ONE ComfyUI file
# (the overwrite report: importing step 2500 of run A, then step 2500 of run B,
# silently replaced A's LoRA). Cloud runs tag with their CloudTrainingRun id
# (`_rc49`), local runs with their TrainingRunRecord id (`_rl12`) — matching the
# ☁/💻 #N chips on the Runs page. Parsed back out (parse_deployed_run) so the
# "in ComfyUI" list can show each file's source run. Display labels also keep
# the raw `rcN`/`rlN` token (comfyui.format_trained_lora_label) so Test Studio
# can tell two same-version runs apart where no chip is shown.
_DEPLOYED_RUN_TAG_RE = re.compile(r'_r([cl])(\d+)(?:_v\d+)?(?:\.[^.]+)?$')


def _run_tag(run_source, run_id) -> str:
    """`_rc<id>` (cloud) / `_rl<id>` (local); '' when the run id is unknown (the
    legacy import path) — the deployed name then stays untagged, as before."""
    if not run_id:
        return ''
    return f"_r{'c' if run_source == 'cloud' else 'l'}{int(run_id)}"


def parse_deployed_run(filename):
    """(source, run_id) recovered from a deployed LoRA's run tag, or (None, None)
    for files imported before run tagging existed (legacy — still listed and
    deletable, just shown as 'run unknown'). Anchored to the END of the name so a
    base tag that merely looks like `..._rl2000` in the middle never matches."""
    m = _DEPLOYED_RUN_TAG_RE.search(os.path.basename(str(filename)))
    if not m:
        return None, None
    return ('cloud' if m.group(1) == 'c' else 'local'), int(m.group(2))


def _run_name(ds, base_model=_PERSISTED, family=None,
              variant=_PERSISTED) -> str:
    """Nom de dossier de run unique par (user, trigger, base, FAMILLE) - évite qu'un
    même trigger_word chez deux datasets partage/écrase les dossiers, isole un run
    sur base custom du run officiel, ET isole les familles entre elles. `base_model`
    absent → base persistée ; fourni (même '') → cette base précise.

    Fix B (2026-07-01) : le tag vient de `_dest_base_tag` (et non `_base_tag`), donc
    un run **Krea** porte le suffixe `_Krea-2-Turbo` dans le NOM DE DOSSIER. Sans ça,
    Z-Image base-officielle (tag vide) et Krea (base vide) au même trigger tombaient
    dans le même dossier `u{user}_{trigger}` → ai-toolkit mélangeait les deux runs et
    l'import récupérait le mauvais checkpoint. Z-Image ajoute désormais aussi la
    recette (Turbo/Base/De-Turbo), afin de ne jamais reprendre des poids issus
    d'une matrice base/adapter incompatible."""
    tag = _dest_base_tag(ds, base_model, family, variant)
    # Slider mode gets its own run folder: ai-toolkit AUTO-RESUMES from the
    # training_folder, so a slider run sharing the normal run's folder would
    # resume subject-LoRA weights into slider training (and vice versa).
    slider = '_slider' if slider_mode_enabled(ds) else ''
    return f'u{ds.user_id}_{_safe_trigger(ds)}{tag}{slider}'


def find_run_collision(user_id, dataset_id, base_model=_PERSISTED,
                       variant=_PERSISTED):
    """Autre dataset du MÊME user qui produirait le même dossier de run
    (`u{user}_{trigger}{base_tag}`) que (dataset_id, base_model). C'est la source
    de collision : ai-toolkit auto-resume depuis ce dossier → LoRA mélangés, et
    deux lancements simultanés corrompent l'`optimizer.pt` partagé (incident
    Test/Test 2, 2026-06-16). Retourne le FaceDataset en conflit, ou None.

    La clé de collision est le dossier complet (trigger + base + recette/variante).
    On compare le run-name CIBLE (base/variante en cours de sélection) aux
    run-names PERSISTÉS des autres datasets du user."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return None
    target = _run_name(ds, variant=variant) if base_model is _PERSISTED \
        else _run_name(ds, base_model, variant=variant)
    # Enumerated through `fds.list_datasets` — the library's own definition of
    # "the datasets that exist" — rather than a raw FaceDataset query. A refusal
    # is only actionable if the user can OPEN the dataset it names: colliding
    # with a row that is not in the library blocks a training run with an error
    # nobody can act on. Today the two sets are identical; keeping the single
    # source means a future listing rule (hidden/archived rows) is honoured here
    # for free instead of being a second place someone must remember.
    for o in fds.list_datasets(ds.user_id):
        if int(o.id) != int(ds.id) and _run_name(o) == target:
            return o
    return None


def _masks_dir(dataset_folder: str) -> str:
    """Dossier des masques d'un export (convention mask_path ai-toolkit : dossier
    frère, mêmes noms de fichiers)."""
    return f'{dataset_folder}_masks'


# Historical person-mask weight (méthode jandordoe). A CONSTANT, not the new
# configurable knob: `face_mask.min_weight` governs face masking, and letting it
# silently re-weight every character run's background would be a behaviour change
# nobody asked for.
_PERSON_MASK_MIN_VALUE = 0.1

# Written beside the masks by the exporter so `_mask_fields` can tell WHICH mask it
# is looking at. The two polarities share one folder (ai-toolkit takes a single PNG
# per image), so the folder alone cannot say whether 'black' means background or
# face — and they do not want the same weight. ai-toolkit resolves masks by exact
# `<image basename><ext>` lookup, so this extra file is never mistaken for one.
_MASK_META_FILENAME = '_mask_meta.json'


def _write_mask_meta(masks_dir: str, kind: str, min_value: float) -> None:
    try:
        with open(os.path.join(masks_dir, _MASK_META_FILENAME), 'w', encoding='utf-8') as fh:
            json.dump({'kind': kind, 'mask_min_value': float(min_value)}, fh)
    except OSError:
        pass  # best-effort: a missing sidecar degrades to the historical weight


def _mask_fields(dataset_folder: str) -> dict:
    """Champs `mask_path`/`mask_min_value` à fusionner dans l'entrée datasets de la
    job-config SI des masques ont été exportés.

    Deux polarités possibles, même dossier :
      - `person` (méthode jandordoe) : sujet blanc, fond noir pondéré à 10 %.
      - `face` (issue #15) : visage noir, reste blanc — l'acte s'apprend, pas l'identité.
    Le poids appliqué vient du sidecar écrit à la génération ; sans sidecar (export
    d'une version antérieure) on retombe sur le 10 % historique, jamais sur le
    réglage du masque visage.
    Dossier absent/vide → {} (l'entraînement reste strictement l'historique)."""
    md = _masks_dir(dataset_folder)
    try:
        if not (os.path.isdir(md) and any(f.lower().endswith('.png') for f in os.listdir(md))):
            return {}
    except OSError:
        return {}
    min_value = _PERSON_MASK_MIN_VALUE
    try:
        with open(os.path.join(md, _MASK_META_FILENAME), encoding='utf-8') as fh:
            meta = json.load(fh)
        if isinstance(meta, dict) and isinstance(meta.get('mask_min_value'), (int, float)):
            min_value = float(meta['mask_min_value'])
    except (OSError, ValueError, TypeError):
        pass
    return {'mask_path': md, 'mask_min_value': min_value}


# ai-toolkit reads dual long+short captions ONLY from a JSON caption file (folder_path
# points at the file, keys are image paths, values {caption, caption_short}); the .txt
# sidecar path cannot carry a short. We keep writing the .txt sidecars too so the cloud
# path (which strips dual) and any manual inspection still work.
_DUAL_CAPTION_FILENAME = '_captions.json'


def _dual_caption_json_path(dataset_folder) -> str:
    """Absolute path of the dual-caption JSON inside an export folder — the single source
    of truth shared by the exporter (writes it) and build_job_config (points folder_path
    at it). Forward-slashed so it matches the JSON keys and needs no backslash escaping."""
    return (str(dataset_folder).rstrip('/\\') + '/' + _DUAL_CAPTION_FILENAME)


# --- Export encoding (2026-08-03) ---------------------------------------------
# The exporter used to re-encode EVERY master to lossless PNG. On a 6 211-image
# style dataset that turned 3.6 GB of masters into 23.7 GB of staging and burned
# 24 min of CPU before a single byte reached the network — measured, not guessed
# — which is what filled the disk ([Errno 28] with no pod created), what left a
# half-hour window in which any app restart killed the run, and what made the
# upload "12 422 files and 24 GB" that RemoteAiToolkit.upload_dataset already
# documents. The re-encode exists for exactly two reasons: bake EXIF orientation
# into the pixels (an upright JPEG must never train sideways) and hand the
# trainer a format it reads. A master that has NO EXIF block at all needs
# neither the baking nor the metadata stripping, so its own bytes go straight
# through.
#
# Extensions verified at the source, not assumed: ai-toolkit scans
# `img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']` (toolkit/dataloader_mixins.py)
# and pairs a mask by STEM + any of those, so a copied .jpg still finds its .png
# mask; the pod uploader's own whitelist (_DATA_EXTS) already carries the same four.
_TRAINER_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

# Free space demanded on top of the estimate: captions, masks, the samples dir a
# cloud run puts beside the dataset, and the plain fact that an export that ends
# on a disk with nothing left is a broken export.
_EXPORT_DISK_MARGIN_BYTES = 2 * 1024 ** 3


def _export_copy_is_safe(im, ext) -> bool:
    """Can this master's own bytes be handed to the trainer untouched?

    Four conditions, all necessary: an extension ai-toolkit reads; NO EXIF block
    at all (nothing to bake into pixels, and nothing — GPS, camera, timestamps —
    to ship to a rented pod); already RGB (palette/CMYK/alpha must be converted);
    a single frame. Anything else keeps the historical PNG re-encode."""
    try:
        exif = im.getexif()
    except Exception:                       # a corrupt EXIF block: re-encode
        return False
    return (ext in _TRAINER_IMAGE_EXTS and not len(exif)
            and im.mode == 'RGB' and int(getattr(im, 'n_frames', 1) or 1) == 1)


def _export_image_bytes(src) -> int:
    """Bytes `src` will occupy in the export, without writing anything.

    A copied master costs exactly its file size; a re-encoded one is bounded by
    its raw RGB size (PNG of a photo lands well under that). Unreadable headers
    fall back to the file size rather than failing the estimate — the export
    itself skips those files anyway."""
    try:
        with Image.open(src) as im:
            if _export_copy_is_safe(im, os.path.splitext(src)[1].lower()):
                return os.path.getsize(src)
            w, h = im.size
            return int(w) * int(h) * 3
    except OSError:
        try:
            return os.path.getsize(src)
        except OSError:
            return 0


def _assert_export_fits(out, srcs) -> None:
    """Refuse an export that cannot fit BEFORE writing its first file.

    Half a written export is worse than none: it fails with a bare
    "[Errno 28] No space left on device" (twice, live, on runs that had already
    spent 20 minutes), it leaves the partial copy behind, and it names neither
    the size it wanted nor the space there was."""
    need = int(sum(_export_image_bytes(s) for s in srcs) * 1.05) + _EXPORT_DISK_MARGIN_BYTES
    try:
        free = shutil.disk_usage(out).free
    except OSError:
        return                              # cannot measure -> do not invent a refusal
    if free >= need:
        return
    raise ValueError(
        f'not enough free disk space to export this dataset: about '
        f'{need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free where the export '
        f'goes ({out}). Free some space — finished cloud runs keep their staging '
        f'copy until you clean it up (🧹 in the cloud run list) — then relaunch.')


def export_dataset_to_aitoolkit(user_id, dataset_id, masked: bool = True, dest_dir=None,
                                masked_faces: bool = True) -> str:
    """Écrit les images `keep` en paires image/.txt dans
    DATASETS_DIR/<trigger>. L'image garde ses octets d'origine (.jpg/.webp/.png)
    quand le trainer les lit tels quels et qu'aucun EXIF n'est à appliquer, sinon
    elle est ré-encodée en .png — cf `_export_copy_is_safe`. Character/concept = trigger + caption éditée ; Style
    always-on = caption de contenu seule (le trigger interne n'est jamais exporté).
    Retourne le dossier.

    `masked` (défaut ON) : génère aussi un masque « personne » par image (rembg
    u2net, subprocess CPU - cf app/services/person_mask) dans `<dossier>_masks` →
    la job-config passe en MASKED TRAINING (fond à 10 %). Échec des masques =
    jamais bloquant : l'entraînement part simplement sans masques (loggé).

    `dest_dir` (cloud seam) : exporte LÀ au lieu de DATASETS_DIR/<run_name> - ne
    requiert PAS ai-toolkit configuré localement (pas d'appel à _datasets_dir()).
    Défaut (None) = comportement historique inchangé."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # TWO masks exist, of OPPOSITE polarity, and the guard below must not confuse
    # them (issue #15, shivdbz2010):
    #   person mask -> subject white, background black. Learn the subject, not the room.
    #   face mask   -> face black, everything else white. Learn the scene, not the identity.
    # The historical guard was written when only the first existed, so it read
    # "concept => no masks at all". A FACE mask does not have the defect it guards
    # against: it removes the identity and keeps the act. Hence a per-polarity guard.
    face_masked = bool(masked_faces) and fds.face_masking_enabled(ds)
    if masked and fds.is_conceptual(ds):
        # A person-mask would erase the very thing we want the LoRA to learn (the
        # recurring act for a concept; the whole-image rendering for a style - which
        # lives as much in backgrounds as in people). Force masked training OFF for
        # concept AND style datasets even if the caller/UI asked for it -- server guard.
        logger.info('dataset %s %s -> masked training forced OFF (server guard)',
                    dataset_id, ds.kind)
        masked = False
    if face_masked and not fds.is_concept(ds):
        # Character wants its identity learned; a Style must learn how it renders a
        # face. Only a concept benefits. face_masking_enabled already refuses both,
        # this is the belt to its braces (a caller may pass masked_faces=True).
        logger.info('dataset %s %s -> face masking forced OFF (server guard)',
                    dataset_id, ds.kind)
        face_masked = False
    if face_masked and slider_mode_enabled(ds):
        # Same reason the person mask is refused in slider mode: the guided slider
        # loss never reads batch.mask_tensor, so the masks would only burn export time.
        logger.info('dataset %s -> slider mode: face masking forced OFF (server guard)',
                    dataset_id)
        face_masked = False
    if face_masked and masked:
        # Belt and braces: one mask PNG per image, so the two polarities cannot
        # share a folder. Unreachable today (a concept never keeps `masked`), but
        # the day someone lifts the concept guard this must not silently pick one.
        logger.info('dataset %s -> face masking wins over person masking', dataset_id)
        masked = False
    if masked and slider_mode_enabled(ds):
        # Slider mode: the guided slider loss never reads the masked-loss path
        # (ConceptSliderTrainer.get_guided_loss ignores batch.mask_tensor), so
        # generating person masks would only burn export time -- server guard.
        logger.info('dataset %s -> slider mode: masked training forced OFF (server guard)',
                    dataset_id)
        masked = False
    trigger = _safe_trigger(ds)
    out = str(dest_dir) if dest_dir else str(_datasets_dir() / _run_name(ds))
    if os.path.isdir(out):
        # Derived training export cache, recreated below from the dataset source.
        # It is not user-authored data, so bypassing Trash is intentional.
        shutil.rmtree(out)  # ré-export propre
    masks_out = _masks_dir(out)
    if os.path.isdir(masks_out):
        # Derived masks are regenerated from exported images on every export.
        shutil.rmtree(masks_out)  # jamais de masques périmés (ré-export ou toggle OFF)
    os.makedirs(out, exist_ok=True)
    kept = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None)).all())
    if not kept:
        raise ValueError('no kept images to export')
    _assert_export_fits(out, [p for p in
                             (os.path.join(fds._dataset_dir(img.dataset_id), img.filename)
                              for img in kept) if os.path.isfile(p)])
    n = 0
    exported = []
    dual = fds.dual_captions_enabled(ds)
    dual_entries = {}
    for img in kept:
        src = os.path.join(fds._dataset_dir(img.dataset_id), img.filename)
        if not os.path.isfile(src):
            continue
        stem = f'{trigger}_{n:03d}'
        ext = os.path.splitext(src)[1].lower()
        # Dataset masters can retain their native JPEG/PNG/WebP/BMP bytes. A
        # master the trainer already reads, carrying no EXIF at all, is copied
        # byte for byte; anything else gets the historical disposable PNG, which
        # bakes EXIF orientation into pixels and drops metadata so an upright
        # JPEG never trains sideways. See _export_copy_is_safe.
        with Image.open(src) as source:
            verbatim = _export_copy_is_safe(source, ext)
            dst = os.path.join(out, f'{stem}{ext if verbatim else ".png"}')
            if not verbatim:
                ImageOps.exif_transpose(source).convert('RGB').save(dst, 'PNG')
        if verbatim:
            shutil.copyfile(src, dst)
        exported.append(dst)
        cap = fds.style_content_caption(ds, img.caption)
        body = cap if fds.is_style(ds) else (f'{trigger}, {cap}' if cap else trigger)
        with open(os.path.join(out, f'{stem}.txt'), 'w', encoding='utf-8') as fh:
            fh.write(body)
        if dual:
            # Short variant: the stored caption_short, degrading to the long caption when a
            # short was never derived (short==long is a harmless no-op augmentation). The
            # trigger is prepended exactly like the long (style stays content-only). JSON
            # key = the image path ai-toolkit will open (forward-slashed).
            short_src = (img.caption_short or '').strip() or img.caption
            scap = fds.style_content_caption(ds, short_src)
            sbody = scap if fds.is_style(ds) else (f'{trigger}, {scap}' if scap else trigger)
            dual_entries[dst.replace('\\', '/')] = {'caption': body, 'caption_short': sbody}
        n += 1
    if n == 0:
        raise ValueError('no valid image file found on disk')
    if dual and dual_entries:
        with open(_dual_caption_json_path(out), 'w', encoding='utf-8') as fh:
            json.dump(dual_entries, fh, ensure_ascii=False, indent=2)
    masked_ok = False
    if masked:
        # generate_person_masks returns a DICT ({"ok", "written", "results"}, or {}
        # on any failure/unavailability) -- a non-empty dict is always truthy, so a
        # verbatim `if wrote:` on the return value would never take the cleanup
        # branch. Read the actual count instead.
        res = generate_person_masks(exported, masks_out)
        wrote = int(res.get('written') or 0) if isinstance(res, dict) else 0
        if wrote:
            masked_ok = True
            _write_mask_meta(masks_out, 'person', _PERSON_MASK_MIN_VALUE)
            logger.info(f'export dataset {dataset_id}: {wrote}/{n} masque(s) personne -> {masks_out}')
        else:
            logger.warning(f'export dataset {dataset_id}: masques indisponibles - training SANS masked loss')
            if os.path.isdir(masks_out):
                # Failed/incomplete derived mask cache; safe to destroy directly.
                shutil.rmtree(masks_out, ignore_errors=True)
    face_masked_ok = False
    face_coverage = None
    if face_masked:
        # Same contract as the person masks: a dict (possibly {}), read the count —
        # never `if res:`, which a non-empty failure dict would satisfy.
        res = face_mask.generate_face_masks(exported, masks_out)
        wrote = int(res.get('written') or 0) if isinstance(res, dict) else 0
        if wrote:
            face_masked_ok = True
            face_coverage = face_mask.coverage_summary(res.get('results') or {})
            _write_mask_meta(masks_out, 'face', face_mask.min_weight())
            logger.info('export dataset %s: %s/%s masque(s) visage -> %s (%s)',
                        dataset_id, wrote, n, masks_out, face_coverage)
        else:
            logger.warning('export dataset %s: masques visage indisponibles '
                           '(insightface absent ?) - training SANS masked loss', dataset_id)
            if os.path.isdir(masks_out):
                shutil.rmtree(masks_out, ignore_errors=True)
    # A REQUESTED masked run that produced no masks (rembg missing, or generation
    # crashed at runtime) silently trains UNMASKED. Record it per-run so the live
    # progress view can warn — instead of the fallback being invisible. `masked` is
    # the FINAL intent: concept/style were already forced OFF above (by design), so
    # they never set this flag.
    queue_manager._set_system_state('training_masks_skipped',
                                    bool((masked and not masked_ok)
                                         or (face_masked and not face_masked_ok)),
                                    ttl_seconds=_TRAIN_STATE_TTL)
    # Face-mask COVERAGE is a safety figure, not a statistic: a partially masked set
    # is the worst case, because the faces that stayed unmasked are then the only
    # ones carrying loss weight and end up over-represented. Publish it per run so
    # the progress view can say "masked on 32 of 40" instead of staying silent.
    queue_manager._set_system_state('training_face_mask_coverage', face_coverage,
                                    ttl_seconds=_TRAIN_STATE_TTL)
    logger.info(f'export dataset {dataset_id} -> {out} ({n} paires)')
    return out


def _export_and_freeze_local_dataset(user_id, dataset_id, *, masked, base_model):
    """Create one coherent local-training export and provenance snapshot.

    Dataset mutations use the same ingest lock and consult the exclusive
    activity below.  Keeping both the ai-toolkit export and
    ``prepare_launch`` inside that reservation prevents a run from training on
    generation A while its LDS record describes generation B.
    """
    lock = fds._dataset_ingest_lock(user_id, dataset_id)
    with lock:
        token = dataset_activity.begin_exclusive(
            dataset_id, 'training_export',
            detail='freezing the Dataset for training')
        if token is None:
            raise dataset_activity.DatasetActivityBusy(
                'This dataset already has work in progress. Wait for it to '
                'finish before launching training.')
        heartbeat_stop = threading.Event()

        def keep_reservation_alive():
            while not heartbeat_stop.wait(30.0):
                dataset_activity.progress(token)

        heartbeat = threading.Thread(
            target=keep_reservation_alive, daemon=True,
            name=f'dataset-{dataset_id}-training-export-heartbeat')
        heartbeat.start()
        try:
            dataset_folder = export_dataset_to_aitoolkit(
                user_id, dataset_id, masked=masked)
            # A cloud/other-lane launch can win while a large local export is
            # being written.  Refuse before the expensive snapshot if so; the
            # final queue/GPU lock remains authoritative for spawning.
            if (queue_manager._get_system_state('training_in_progress', False)
                    and not _training_process_is_definitely_dead(
                        queue_manager._get_system_state('training_pid', None))):
                raise ValueError(
                    'a training is already in progress - wait for it to finish '
                    'or queue this dataset')
            from . import checkpoint_registry
            prepared = checkpoint_registry.prepare_launch(
                user_id, dataset_id, base_model=base_model)
            if checkpoint_registry.prepared_generation_identity(prepared) is None:
                raise RuntimeError(
                    'could not freeze the Dataset provenance for training; no '
                    'run was started — retry after checking the backend log')
            return dataset_folder, prepared
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1.0)
            dataset_activity.end(token)


# --- Overrides STYLE (communs aux familles) ------------------------------------
_STYLE_CAPTION_DROPOUT = 0.05


def _effective_style_caption_dropout(family: str | None, process: dict | None = None) -> float:
    """Caption dropout really applied by ai-toolkit for an always-on Style.

    Krea caches frozen text embeddings before training. Caption dropout after that
    cache is ineffective/misleading, so explicitly use zero. Other families retain
    the conservative 5% generic style baseline.
    """
    fam = (family or '').lower()
    if fam == 'krea':
        # Every LDS Krea recipe currently caches embeddings. Keep the process probe
        # for future recipes that may disable it, while snapshots (process=None)
        # still report the current effective Krea value.
        cached = process is None or any(
            bool(d.get('cache_text_embeddings')) for d in process.get('datasets', ()))
        if cached:
            return 0.0
    return _STYLE_CAPTION_DROPOUT


def _apply_style_overrides(ds, process: dict, family: str | None = None) -> dict:
    """Apply the always-on Style contract to one ai-toolkit process. No-op otherwise."""
    if not fds.is_style(ds):
        return process
    fam = _train_type(ds, family)
    # Internal identifier only: it may name folders/configs, never condition the
    # model or leak into sidecars/sample prompts.
    process.pop('trigger_word', None)
    dropout = _effective_style_caption_dropout(fam, process)
    for d in process.get('datasets', ()):
        d['caption_dropout_rate'] = dropout
    # Timestep choice is family/variant-specific. Never erase a resolved safe
    # recipe merely because the dataset kind is Style.
    return process


def _dual_captions_unsupported_reason(process: dict) -> str | None:
    """Why this ai-toolkit process cannot train dual captions. None = it can.

    ai-toolkit caches ONE embedding per image: TextEmbeddingCachingMixin.cache_text_embeddings
    encodes `file_item.caption` (the LONG one) and nothing else, and once the encoder is
    unloaded SDTrainer takes the `if unload_text_encoder or is_caching_text_embeddings`
    branch, which feeds the model `batch.prompt_embeds` and never looks at the prompt
    strings again. The short caption has, literally, nowhere to go.

    It does not merely go unused — it crashes the run. The caching pass reaches
    load_caption() through get_text_embedding_info_dict() WITHOUT the JSON caption dict,
    so each item is filled from its .txt sidecar (long only) and `raw_caption_short` stays
    None; load_caption() then short-circuits on the real per-batch call ("we already
    loaded it"), `caption_short` is never computed, and the doubled prompt list handed to
    inject_trigger_into_prompt contains None → AttributeError at the first step, after the
    weights download and the whole caching pass. Reported on krea as GitHub issue #22 by
    1Tomber; anima emits the identical pair.

    So the combination is refused at config time rather than patched with a placeholder
    short caption: a placeholder would buy a green run that trains exactly like a
    long-caption run while claiming otherwise."""
    train = process.get('train') or {}
    if any(d.get('cache_text_embeddings') for d in process.get('datasets', ())):
        return ('this family pre-caches its text embeddings (one embedding per image) '
                'to free the VRAM its text encoder would hold')
    if train.get('unload_text_encoder'):
        return ('this family unloads its text encoder after caching, so no second '
                'caption can be encoded')
    return None


def _apply_dual_captions(ds, process: dict, dataset_folder) -> dict:
    """Wire ai-toolkit dual long+short captioning onto one process. No-op unless the
    dataset opted in. When on, the FIRST dataset block points at the JSON caption file
    (the only format a short caption can be read from — the .txt sidecar path cannot) and
    train.short_and_long_captions turns on the batch-doubling that trains each image with
    BOTH captions. caption_ext is left as-is; it is ignored once folder_path is a JSON file.

    Local-only for now: the cloud pod's dataset upload skips the JSON, so the cloud path
    (_cloudify_job_config) reverts this back to the historical folder + .txt sidecars.

    No-op as well on a family that caches its text embeddings — emitting the pair is what
    crashed issue #22. The run then trains on the long caption alone (the .txt sidecars,
    trigger included), and the preflight says so before the launch."""
    if not fds.dual_captions_enabled(ds):
        return process
    if _dual_captions_unsupported_reason(process):
        return process
    datasets = process.get('datasets') or []
    if not datasets:
        return process
    datasets[0]['folder_path'] = _dual_caption_json_path(dataset_folder)
    process.setdefault('train', {})['short_and_long_captions'] = True
    return process


def _apply_slider_overrides(ds, process: dict, family: str | None = None) -> dict:
    """Turn one family process into a Slider LoRA (Beta) job. No-op when the
    dataset is not in slider mode.

    Emission is validated against ai-toolkit's modern `concept_slider` extension
    (extensions_built_in/concept_slider/ConceptSliderTrainer.py):
    - process type = the extension uid 'concept_slider' (replaces 'sd_trainer');
    - `slider:` block = the exact ConceptSliderTrainerConfig kwargs;
    - `trigger_word` dropped — the slider is prompt-defined, there is no
      activation token (the ostris UI disables that section for slider jobs,
      ui/src/app/jobs/new/options.ts:1190);
    - `anchor_class` emitted only when set: ConceptSliderTrainerConfig defaults
      it to None (anchors off). NB: the ostris UI default emits '' which the
      trainer treats as an ACTIVE anchor on the unconditional prompt — we keep
      the trainer's native default instead and document the field in the UI;
    - masks stripped: the guided slider loss never reads the masked-loss path
      (ConceptSliderTrainer.get_guided_loss ignores batch.mask_tensor);
    - Z-Image: text-embedding cache force-disabled + batch_size already 1 —
      the community workaround for ai-toolkit issue #554 (broken embedding
      cache on the zimage slider path);
    - previews become a bipolar sweep: the SAME prompt sampled at network
      multipliers −2/−1/+1/+2 (SampleItem.network_multiplier, applied per image
      in BaseSDTrainProcess.sample), so collapse is visible early."""
    fam = _train_type(ds, family)
    if not slider_mode_enabled(ds):
        return process
    sc = _slider_settings(ds)
    process['type'] = 'concept_slider'
    process.pop('trigger_word', None)
    slider = {
        'guidance_strength': _slider_guidance(ds),
        'anchor_strength': _slider_anchor_strength(ds),
        'positive_prompt': (sc.get('positive') or '').strip(),
        'negative_prompt': (sc.get('negative') or '').strip(),
        'target_class': (sc.get('target_class') or '').strip(),
    }
    anchor = (sc.get('anchor') or '').strip()
    if anchor:
        slider['anchor_class'] = anchor
    process['slider'] = slider
    # Slider VRAM default: emit 768 only unless the user explicitly chose a
    # resolution. The slider loss makes several prediction passes per step, so
    # multi-scale 768+1024 peaks far higher and OOMs on 24 GB. Recomputed here
    # (not read off the family process) so the same rule covers every family and
    # matches launch_settings_snapshot's stamped resolution.
    res = _effective_resolution(ds)
    for d in process.get('datasets', ()):
        # Substrate only: masked loss is dead code in the slider loss path, and
        # person masks would just burn CPU time at export.
        d.pop('mask_path', None)
        d.pop('mask_min_value', None)
        d['resolution'] = res
        if fam == 'zimage':
            d['cache_text_embeddings'] = False   # issue #554 workaround
    # Bipolar preview sheet. Base prompt: the user's first custom sample prompt
    # if any, else the target class ("a photo of a <class>"), else the positive
    # prompt (a detail-style slider with an empty target class affects
    # everything, so the positive side is the most telling preview).
    custom = _train_settings(ds).get('sample_prompts')
    base_prompt = ''
    if isinstance(custom, list) and custom:
        base_prompt = str(custom[0]).strip()
    if not base_prompt:
        tc = (sc.get('target_class') or '').strip()
        base_prompt = f'a photo of a {tc}' if tc else (sc.get('positive') or '').strip()
    sample = process.get('sample')
    if isinstance(sample, dict) and base_prompt:
        sample.pop('prompts', None)
        sample['samples'] = [{'prompt': base_prompt, 'network_multiplier': m}
                             for m in (-2, -1, 1, 2)]
    return process


def _assert_full_transformer_recipe(ds) -> None:
    """What a Krea 2 dense run may and may not be built from.

    Two refusals used to live here and no longer do — Turbo, and a custom base.
    Both were SCOPE decisions, and this docstring already said so; what kept
    them was not the risk of a broken checkpoint but the risk of a MISLABELLED
    run, because the dense config pinned its base to a constant. That is the
    part that was fixed first: `_krea_name_or_path` now resolves the base from
    the selection for the dense lane exactly as for the LoRA lane, and the
    provenance stamp reads the same resolver. A dense run can no longer
    announce Turbo and train Raw, so the refusal it justified has nothing left
    to protect.

    What travels with the lifted Turbo refusal, as a WARNING rather than a
    wall (the product decision: warn and let through):

    - The dense lane still loads NO `assistant_lora_path`, on Turbo included.
      That is deliberate. ai-toolkit can subtract an adapter
      (`merge_out(w) == merge_in(-w)`) but nothing wires that into the dense
      save path, and a LoRA-shaped subtraction would not cover the
      normalisation/modulation tensors a dense run moves anyway. Adding the
      de-distillation adapter here would introduce the exact defect the old
      guard feared; leaving it out cannot.
    - Where dense-on-distilled HAS been measured publicly (Z-Image-Turbo,
      FLUX.2 Klein) the cost is speed, not validity: the checkpoint stays
      structurally sound and progressively stops being a few-step model,
      drifting back toward real guidance and ~25-30 steps. Erosion, not a
      cliff. Nothing equivalent has been measured for Krea 2, by us or by
      anyone — which is what the UI says, without predicting either outcome.

    What survives, and why each one is a different kind of statement:

    * family — Krea 2 only. The dense recipe (no `network` block, adafactor,
      the 80 GB geometry) is written against `arch='krea2'`.
    * Slider — a slider LoRA IS an adapter with a direction; there is no dense
      equivalent of a network multiplier.
    * a base that is not an absolute local path — a ComfyUI-relative name
      addresses another family's catalog and is silently ignored by the krea2
      loader. The LoRA lane drops it; the dense lane refuses it instead,
      because "ignored" here means training something other than what the
      panel shows.
    * a STRUCTURED fp8/int8 export as the base — the only MECHANICAL limit in
      this list, and the reason it is enforced by `assert_trainable_base_file`
      rather than restated: ai-toolkit loads a base with
      ``load_state_dict(..., strict=True)`` and a scaled_fp8 export carries
      `.scale_weight` tensors the architecture never declares, so the load
      raises. A BARE cast (same tensor names, reduced dtype) is up-cast by the
      trainer and stays allowed. Refusing the two alike would repeat, in the
      other direction, the mistake this function is being corrected for.
    """
    if not _is_full_transformer(ds):
        return
    if _train_type(ds) != 'krea':
        raise ValueError('full_transformer training is supported only for Krea 2')
    if slider_mode_enabled(ds):
        raise ValueError('full_transformer training is incompatible with Slider LoRA mode')
    base = str(getattr(ds, 'train_base_model', None) or '').strip()
    if base:
        if not _is_custom_weights(base):
            raise ValueError(
                'full_transformer training needs either the official Krea 2 base '
                'or the full path to a local .safetensors checkpoint')
        # Mechanical, not scope: a structured fp8/int8 export cannot be loaded.
        assert_trainable_base_file(base)


def build_job_config(ds, dataset_folder: str, steps: int = 3000, training_folder=None) -> dict:
    """Job-config ai-toolkit pour la recette Z-Image validée (Turbo/Base/De-Turbo).
    Clés alignées sur ce que génère
    l'UI ai-toolkit (ui/src/app/jobs/new/options.ts) + structure LoRA 24 Go de
    référence - vérifiées au runtime contre la version installée (cf. spec §3).
    Points non négociables : arch='zimage', base/adapter résolus uniquement par
    ``zimage_training_recipe``, quantize qfloat8 + low_vram pour tenir sur 24 Go
    — ces trois-là sont désormais les DÉFAUTS (inchangés) d'un tri-état surchargeable
    par dataset (_model_memory_block), pas des constantes : cf. issue #14.

    SDXL (train_type='sdxl') part dans une branche dédiée (_build_job_config_sdxl) -
    le chemin zimage ci-dessous reste strictement inchangé.

    `training_folder` (cloud seam) : utilisé TEL QUEL comme process.training_folder
    dans les 3 familles - aucun appel à _output_dir() (pas d'ai-toolkit local requis).
    Défaut (None) = comportement historique inchangé (`_run_root(ds)`) - c'est aussi
    le dossier où atterrit training.log, l'invariant que « 📂 Run folder » ouvre."""
    mode = training_mode(ds)
    if mode == 'full_transformer' and _train_type(ds) != 'krea':
        _assert_full_transformer_recipe(ds)
    if _train_type(ds) == 'sdxl':
        cfg_ = _build_job_config_sdxl(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0], 'sdxl')
        _apply_slider_overrides(ds, cfg_['config']['process'][0], 'sdxl')
        _apply_dual_captions(ds, cfg_['config']['process'][0], dataset_folder)
        return cfg_
    if _train_type(ds) == 'krea':
        cfg_ = _build_job_config_krea(ds, dataset_folder, steps, training_folder=training_folder)
        # Dense training must stay free of every LoRA-specific post-processor.
        # The dedicated builder already emits the complete conservative recipe.
        if mode == 'full_transformer':
            return cfg_
        _apply_style_overrides(ds, cfg_['config']['process'][0], 'krea')
        _apply_slider_overrides(ds, cfg_['config']['process'][0], 'krea')
        _apply_dual_captions(ds, cfg_['config']['process'][0], dataset_folder)
        return cfg_
    if _train_type(ds) == 'flux':
        cfg_ = _build_job_config_flux(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0], 'flux')
        _apply_slider_overrides(ds, cfg_['config']['process'][0], 'flux')
        _apply_dual_captions(ds, cfg_['config']['process'][0], dataset_folder)
        return cfg_
    if _train_type(ds) == 'flux2klein':
        cfg_ = _build_job_config_flux2klein(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0], 'flux2klein')
        _apply_slider_overrides(ds, cfg_['config']['process'][0], 'flux2klein')
        _apply_dual_captions(ds, cfg_['config']['process'][0], dataset_folder)
        return cfg_
    if _train_type(ds) == 'anima':
        cfg_ = _build_job_config_anima(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0], 'anima')
        _apply_slider_overrides(ds, cfg_['config']['process'][0], 'anima')
        _apply_dual_captions(ds, cfg_['config']['process'][0], dataset_folder)
        return cfg_
    trigger = _safe_trigger(ds)
    base_model = getattr(ds, 'train_base_model', None)
    recipe = zimage_training_recipe(getattr(ds, 'train_variant', None), base_model)

    # Base : officielle (repo HF diffusers) OU merge ComfyUI converti en diffusers.
    model = {'arch': 'zimage', **_model_memory_block(ds, 'zimage')}
    if recipe['custom_base']:
        from .zimage_convert import converted_dir
        model['name_or_path'] = converted_dir(base_model)       # dossier diffusers converti
    else:
        model['name_or_path'] = recipe['effective_base']
    if recipe['extras_name_or_path']:
        model['extras_name_or_path'] = recipe['extras_name_or_path']
    if recipe['training_adapter']:
        model['assistant_lora_path'] = recipe['training_adapter']
    _zrank = _lora_rank(ds, 'zimage')   # défaut 16 (choix user) ; éditable via train_settings

    cfg_ = {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_run_root(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _zrank, 'zimage'),
                'save': {'dtype': _save_dtype_eff(ds), 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    # 5% de dropout caption : le modèle voit parfois le trigger seul,
                    # ce qui renforce l'association trigger→identité (reco LoRA de
                    # sujet ; l'identité doit vivre dans le trigger, pas les mots).
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    **_dataset_cache_text_embeddings(ds),
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    # 'sigmoid' = reco runbook pour un LoRA de sujet (l'exemple
                    # ai-toolkit confirme : "for just subject, change to sigmoid").
                    'timestep_type': _timestep_type_eff(ds, recipe['timestep_type']),
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_train_serializer_fields(ds),
                    **_content_or_style_fields(ds),
                    **_lr_sched_fields(ds),
                    **_ema_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',   # cohérence avec SDXL : défaut ai-toolkit = False (booléen) → fragile
                    'sample_every': _sample_every(ds),
                    'guidance_scale': recipe['guidance_scale'],
                    'sample_steps': recipe['sample_steps'],
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }
    _apply_style_overrides(ds, cfg_['config']['process'][0], 'zimage')
    _apply_slider_overrides(ds, cfg_['config']['process'][0], 'zimage')
    _apply_dual_captions(ds, cfg_['config']['process'][0], dataset_folder)
    return cfg_


def _build_job_config_krea(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit pour Krea 2. Deux bases selon `train_variant` (cf.
    _krea_is_raw), toutes deux arch='krea2', alignées sur l'UI ai-toolkit
    (ui/src/app/jobs/new/options.ts) :

    - RAW (défaut, reco officielle « train on Raw, validate on Turbo ») :
      name_or_path='krea/Krea-2-Raw' (non distillé), AUCUN assistant_lora_path (rien
      à dé-distiller), previews en CFG 4 / 25 steps (le Raw a besoin d'un vrai CFG).
      1er run = download des poids Raw (~24 Go) et run > 4 h → d'où _TRAIN_STATE_TTL 12 h.
    - TURBO (opt-in, VRAM-friendly) : name_or_path='krea/Krea-2-Turbo' + l'adapter de
      training Ostris (retiré à l'inférence, comme Z-Image), previews CFG 1 / 8 steps.

    Commun : quantize qfloat8 + low_vram pour tenir sur 24 Go (défauts surchargeables,
    cf. _model_memory_block — 12B non quantisé = ~30 Gio). ⚠️ Requiert ai-toolkit
    À JOUR (commit « Add support for Krea2 », arch 'krea2') sinon l'arch est inconnue
    (garde _aitoolkit_supports_krea). Réseau = 'lora' : VÉRIFIÉ canonique 2026-06-26.
    Résolution KREA_TRAIN_RESOLUTION (1024, TE déchargé) car 768 seul tenait sinon."""
    trigger = _safe_trigger(ds)
    if _is_full_transformer(ds):
        _assert_full_transformer_recipe(ds)
        return {
            'job': 'extension',
            'config': {
                # Krea 2 Community License requires derivative model names to
                # begin with "Krea"; the job/save root inherits this value.
                'name': f'Krea_full_{trigger}',
                'process': [{
                    'type': 'sd_trainer',
                    'training_folder': (training_folder if training_folder
                                        else str(_run_root(ds))),
                    'device': 'cuda:0',
                    'trigger_word': trigger,
                    # Deliberately NO `network` key: ai-toolkit interprets its
                    # absence as optimisation of the actual transformer weights.
                    'save': {
                        'dtype': 'bf16',
                        'save_every': _dense_save_every(ds),
                        'max_step_saves_to_keep': _dense_max_step_saves(ds),
                    },
                    'datasets': [{
                        'folder_path': dataset_folder,
                        'caption_ext': 'txt',
                        'caption_dropout_rate': 0.05,
                        'cache_latents_to_disk': True,
                        'cache_text_embeddings': True,
                        'resolution': [_dense_resolution(ds)],
                        **_mask_fields(dataset_folder),
                    }],
                    'train': {
                        'batch_size': 1,
                        'steps': steps,
                        # `steps` counts OPTIMISER steps, so accumulation does
                        # not change how many checkpoints a run produces — it
                        # changes how many images each step learned from, and
                        # how long the run takes. Neither the save cadence nor
                        # the storage forecast moves.
                        'gradient_accumulation': _dense_grad_accum(ds),
                        'train_unet': True,
                        'train_text_encoder': False,
                        'unload_text_encoder': True,
                        'gradient_checkpointing': True,
                        'noise_scheduler': 'flowmatch',
                        'timestep_type': _dense_timestep_type(ds),
                        'optimizer': 'adafactor',
                        'lr': _dense_lr(ds),
                        **_dense_lr_schedule_fields(ds),
                        'dtype': 'bf16',
                    },
                    'model': {
                        'arch': 'krea2',
                        # The SELECTION, not a constant: Raw, Turbo, or the
                        # absolute path to a local checkpoint. Same resolver as
                        # the LoRA branch and as the provenance stamp, so a run
                        # cannot be named/stamped one base and trained on
                        # another. NO assistant_lora_path on Turbo here — see
                        # _assert_full_transformer_recipe.
                        'name_or_path': _krea_name_or_path(ds),
                        'quantize': False,
                        'low_vram': False,
                        'quantize_te': False,
                        'model_kwargs': {'vae_path': FULL_TRANSFORMER_VAE},
                    },
                    'sample': {
                        'sampler': 'flowmatch',
                        'neg': '',
                        # Previews follow the checkpoint cadence: a probe sheet
                        # that does not line up with a save cannot be used to
                        # pick which save to keep.
                        'sample_every': _dense_save_every(ds),
                        'guidance_scale': FULL_TRANSFORMER_SAMPLE_GUIDANCE,
                        'sample_steps': FULL_TRANSFORMER_SAMPLE_STEPS,
                        'prompts': _sample_prompts(ds, trigger),
                    },
                }],
            },
        }
    is_raw = _krea_is_raw(ds)
    _krank = _lora_rank(ds, 'krea')   # défaut 32/32 (recherche) ; éditable via train_settings
    # Custom weights (local-only, same krea2 arch) override name_or_path; the TE/VAE
    # stay official (Krea bundles them). The variant still drives the adapter/CFG.
    model = {
        'arch': 'krea2',
        'name_or_path': _krea_name_or_path(ds),
        **_model_memory_block(ds, 'krea'),
    }
    # Adapter de dé-distillation : Turbo UNIQUEMENT (le Raw est déjà non distillé →
    # rien à retirer ; le charger dessus dégraderait le training).
    if not is_raw:
        model['assistant_lora_path'] = ('ostris/krea2_turbo_training_adapter/'
                                        'krea2_turbo_training_adapter_v1.safetensors')
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_run_root(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _krank, 'krea'),
                'save': {'dtype': _save_dtype_eff(ds), 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    # Pré-cache les embeddings du Qwen3-VL pour pouvoir le DÉCHARGER pendant le
                    # training (cf. unload_text_encoder) → libère ~4-8 Go → 1024 tient sans offload.
                    # Valide ici car train_text_encoder=False (sorties figées → cachables sans perte).
                    **_dataset_cache_text_embeddings(ds, default=True),
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    **({'unload_text_encoder': True}
                       if _dataset_cache_text_embeddings(
                           ds, default=True)['cache_text_embeddings'] else {}),
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    'timestep_type': _timestep_type_eff(ds, 'linear'),  # défaut canonique krea2 (options.ts)
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_train_serializer_fields(ds),
                    **_lr_sched_fields(ds),
                    **_ema_fields(ds),
                    **_krea_recipe_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    # Turbo (distillé) : cfg 1 / 8 steps ; Raw (non distillé) : cfg 4 / 25 steps.
                    'guidance_scale': 4 if is_raw else 1,
                    'sample_steps': 25 if is_raw else 8,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


def _build_job_config_flux(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit pour FLUX.1-dev (arch='flux'). Valeurs VÉRIFIÉES contre
    l'ai-toolkit installé : `ui/.../options.ts` (entrée 'flux' : name_or_path
    'black-forest-labs/FLUX.1-dev', quantize + quantize_te True, sampler /
    noise_scheduler 'flowmatch') ET le notebook officiel `FLUX_1_dev_LoRA_Training`
    (linear/alpha 16, lr 1e-4, previews guidance 4 / 20 steps).

    arch='flux' est une arch CŒUR d'ai-toolkit (toolkit/config_modules.py) — supportée
    par tout ai-toolkit, donc AUCUNE garde de version (contrairement à krea2, extension).
    FLUX.1-dev est un modèle GATED sur Hugging Face : le 1er run télécharge ~24 Go et
    exige un HF_TOKEN ayant accepté la licence (même mécanique que Krea, aussi gated).

    VRAM : Flux est un DiT 12B (même classe que Krea 2). On ajoute low_vram + qfloat8
    (comme Krea, dont la mesure LDS a montré la nécessité à 24 Go) au-dessus des defaults
    options.ts — curseur basse-VRAM = la résolution 768 (cf. _train_res / KREA_TRAIN)."""
    trigger = _safe_trigger(ds)
    _frank = _lora_rank(ds, 'flux')   # défaut 16 (exemple flux officiel) ; éditable via train_settings
    # Custom weights (local-only, same flux arch) override name_or_path; TE/VAE stay
    # official (ai-toolkit's flux loader resolves them from the official repo).
    _fbase = getattr(ds, 'train_base_model', None)
    model = {
        'arch': 'flux',
        'name_or_path': (_fbase if _is_custom_weights(_fbase)
                         else 'black-forest-labs/FLUX.1-dev'),
        **_model_memory_block(ds, 'flux'),
    }
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_run_root(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _frank, 'flux'),
                'save': {'dtype': _save_dtype_eff(ds), 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    **_dataset_cache_text_embeddings(ds),
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    # 'sigmoid' = reco LoRA de SUJET pour les modèles flowmatch (l'exemple
                    # flux d'ai-toolkit documente ce choix ; identique à Z-Image).
                    'timestep_type': _timestep_type_eff(ds, 'sigmoid'),
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_train_serializer_fields(ds),
                    **_content_or_style_fields(ds),
                    **_lr_sched_fields(ds),
                    **_ema_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    'guidance_scale': 4,   # FLUX.1-dev : guidance ~4 (notebook officiel)
                    'sample_steps': 20,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


def _build_job_config_flux2klein(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit pour FLUX.2 Klein. Deux tailles selon `train_variant`
    (cf. _flux2klein_is_9b) : arch='flux2_klein_4b' (défaut, voie locale 16-24 Go)
    ou 'flux2_klein_9b' (32-48 Go, voie cloud surtout). Valeurs VÉRIFIÉES contre
    l'ai-toolkit installé : `ui/.../options.ts` (entrées flux2_klein_4b/9b) et
    `extensions_built_in/diffusion_models/flux2/flux2_klein_model.py`.

    Divergences vs le chemin flux (options.ts fait foi) :
    - timestep_type 'weighted' — le défaut canonique des deux entrées Klein
      (PAS 'sigmoid' comme flux/zimage) ;
    - model_kwargs {'match_target_res': False} — clé propre à cette arch,
      absente du chemin flux ;
    - base NON distillée (flux2_is_guidance_distilled=False côté ai-toolkit) →
      les previews utilisent un VRAI CFG : guidance 4 / 25 steps (les défauts
      « non distillé » de l'UI ai-toolkit — même duo que Krea Raw), là où
      FLUX.1-dev (guidance-distillé) sample en guidance 4 / 20 steps.

    Les deux name_or_path sont des modèles GATED sur Hugging Face : accepter la
    licence + HF_TOKEN avant le 1er run, même mécanique que FLUX.1-dev et Krea.
    ⚠️ Contrairement à 'flux' (arch CŒUR), flux2_klein_* sont des EXTENSIONS →
    garde de version obligatoire (_aitoolkit_supports_flux2klein) sinon
    get_model_class retombe en silence sur le loader SD legacy (LoRA corrompu).
    quantize/low_vram/qfloat8 comme les autres familles ; curseur basse-VRAM =
    la résolution 768 (cf. _train_res)."""
    trigger = _safe_trigger(ds)
    is_9b = _flux2klein_is_9b(ds)
    _fkrank = _lora_rank(ds, 'flux2klein')   # défaut 16 ; éditable via train_settings
    # Custom weights (local-only, same flux2_klein arch) override name_or_path; the
    # TE (Mistral, hardcoded MISTRAL_PATH in ai-toolkit) and VAE stay official.
    _fkbase = getattr(ds, 'train_base_model', None)
    model = {
        'arch': 'flux2_klein_9b' if is_9b else 'flux2_klein_4b',
        'name_or_path': (_fkbase if _is_custom_weights(_fkbase)
                         else ('black-forest-labs/FLUX.2-klein-base-9B' if is_9b
                               else 'black-forest-labs/FLUX.2-klein-base-4B')),
        **_model_memory_block(ds, 'flux2klein'),
        'model_kwargs': {'match_target_res': False},
    }
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_run_root(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _fkrank, 'flux2klein'),
                'save': {'dtype': _save_dtype_eff(ds), 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    **_dataset_cache_text_embeddings(ds),
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    'timestep_type': _timestep_type_eff(ds, 'weighted'),
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_train_serializer_fields(ds),
                    **_content_or_style_fields(ds),
                    **_lr_sched_fields(ds),
                    **_ema_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    # Base non distillée → vrai CFG (cf. docstring) : 4 / 25 steps.
                    'guidance_scale': 4,
                    'sample_steps': 25,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


def _build_job_config_anima(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit pour Anima (arch='anima'). Modèle circlestone-labs
    Anima (Cosmos Predict2 DiT 2B + text encoder Qwen3 + conditionneur T5 + VAE
    Qwen-Image). Valeurs VÉRIFIÉES contre la PR ostris/ai-toolkit #860 (mergée le
    2026-07-15), entrée `anima` de `ui/.../options.ts` :
    - name_or_path 'circlestone-labs/Anima-Base-v1.0-Diffusers' (PUBLIC, non gated) ;
    - quantize / quantize_te False (defaults options.ts — le DiT ne fait que 2B,
      contrairement aux 12B krea/flux qui forcent qfloat8) ;
    - noise_scheduler / sampler 'flowmatch', timestep_type 'weighted' (défaut
      canonique de l'entrée, PAS 'sigmoid') ;
    - negative de preview = ANIMA_SAMPLE_NEG (tags anime score_1..3, défaut UI).

    ⚠️ arch 'anima' = EXTENSION (extensions_built_in/diffusion_models/anima), PAS
    une arch cœur → garde de version obligatoire (_aitoolkit_supports_anima) sinon
    get_model_class retombe en silence sur le loader SD legacy (LoRA corrompu).
    Anima exige aussi un diffusers récent (AnimaModularPipeline) — angle mort de la
    garde, documenté dans _aitoolkit_supports_anima.

    VRAM : 2B → modeste. On garde quand même cache_latents_to_disk +
    cache_text_embeddings + unload_text_encoder (générique ai-toolkit, valide car
    train_text_encoder=False → sorties du Qwen3 figées, cachables sans perte), ce
    qui décharge le TE après caching et laisse de la marge sur les petites cartes.
    guidance 4 / 25 steps pour les previews : base NON distillée (vrai CFG), même
    duo que Krea Raw — extrapolé faute de chiffre publié spécifique à Anima."""
    trigger = _safe_trigger(ds)
    _arank = _lora_rank(ds, 'anima')   # défaut 32 (defaultLinearRank options.ts) ; éditable via train_settings
    # Custom weights (local-only, same anima arch) override name_or_path; the TE
    # (Qwen3) / conditioner (T5) / VAE stay official (ai-toolkit's anima loader
    # resolves them from the official pipeline).
    _abase = getattr(ds, 'train_base_model', None)
    model = {
        'arch': 'anima',
        'name_or_path': (_abase if _is_custom_weights(_abase) else ANIMA_BASE),
        **_model_memory_block(ds, 'anima'),
    }
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_run_root(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _arank, 'anima'),
                'save': {'dtype': _save_dtype_eff(ds), 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    # Pré-cache les embeddings du Qwen3 pour pouvoir le DÉCHARGER pendant
                    # le training (unload_text_encoder) → libère de la VRAM. Valide car
                    # train_text_encoder=False (sorties figées → cachables sans perte).
                    **_dataset_cache_text_embeddings(ds, default=True),
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    **({'unload_text_encoder': True}
                       if _dataset_cache_text_embeddings(
                           ds, default=True)['cache_text_embeddings'] else {}),
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    'timestep_type': _timestep_type_eff(ds, 'weighted'),  # défaut canonique anima (options.ts)
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_train_serializer_fields(ds),
                    **_content_or_style_fields(ds),
                    **_lr_sched_fields(ds),
                    **_ema_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': ANIMA_SAMPLE_NEG,
                    'sample_every': _sample_every(ds),
                    'guidance_scale': 4,
                    'sample_steps': 25,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


def _build_job_config_sdxl(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit arch='sdxl' - valeurs VÉRIFIÉES dans ai-toolkit
    ui/.../options.ts (entrée 'sdxl', 2026-06-14) : quantize/quantize_te False,
    noise_scheduler/sampler 'ddpm', timestep_type DÉSACTIVÉ, guidance 6. Base =
    checkpoint SDXL ComfyUI local (single-file, pas de conversion)."""
    trigger = _safe_trigger(ds)
    base_model = getattr(ds, 'train_base_model', None)
    if not base_model:
        raise ValueError('SDXL: a base checkpoint is required')
    # A ComfyUI-whitelist basename resolves under models/checkpoints; a free
    # ABSOLUTE path is the opt-in custom-weights file (validated by the launch
    # preflight, so it bypasses the basename whitelist deliberately).
    name_or_path = base_model if _is_custom_weights(base_model) else _sdxl_base_path(base_model)
    model = {'arch': 'sdxl', 'name_or_path': name_or_path,
             **_model_memory_block(ds, 'sdxl')}
    # SDXL is the only family where ai-toolkit honours these top-level overrides
    # (stable_diffusion_model.py). Emitted only when set; TE may be a local path
    # or a HF repo id (AutoModel.from_pretrained accepts both).
    _svae = getattr(ds, 'train_vae_path', None)
    _ste = getattr(ds, 'train_te_path', None)
    if _svae:
        model['vae_path'] = _svae
    if _ste:
        model['te_name_or_path'] = _ste
    _srank = _lora_rank(ds, 'sdxl')   # défaut 32 ; alpha = rank/2 (demi-force, conservé)
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_run_root(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _srank, 'sdxl'),
                'save': {'dtype': _save_dtype_eff(ds), 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    **_dataset_cache_text_embeddings(ds),
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'ddpm',   # SDXL = epsilon/DDPM (≠ flowmatch Z-Image)
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_train_serializer_fields(ds),
                    **_content_or_style_fields(ds),
                    **_lr_sched_fields(ds),
                    **_ema_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'ddpm',
                    # neg='' EXPLICITE : sans cette clé, ai-toolkit met neg=False (booléen) et le
                    # tokenizer CLIP de transformers 5.x rejette [False] → ValueError au sample
                    # baseline (« text input must be of type str »). SDXL crashait juste avant la
                    # 1re step. '' est un str valide → sample sans négatif (voulu pour un LoRA sujet).
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    'guidance_scale': 6,
                    'sample_steps': 28,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


_CK_RE = re.compile(r'_(\d{4,})\.safetensors$')


def _run_root(ds, base_model=_PERSISTED, family=None, variant=_PERSISTED):
    """ai-toolkit's `training_folder` for this run — the run's TOP folder.

    It holds `training.log` (we open it before spawning, so it exists from the
    first second of a run) and, once ai-toolkit reaches its first save, the
    `lora_<trigger>` save_root below. Two distinct folders were both being
    called "the run folder" at nine call sites; this is the top one, `_run_dir`
    is the save_root, and no caller should rebuild either by hand again."""
    return _output_dir() / _run_name(ds, base_model, family, variant)


def _run_log_path(ds, base_model=_PERSISTED, family=None, variant=_PERSISTED) -> str:
    """Where the local run's `training.log` is written and read. Single source
    of truth: the writer, the progress reader and « 📂 Run folder » must never
    disagree on it (they did — a crashed run's log looked missing)."""
    return str(_run_root(ds, base_model, family, variant) / 'training.log')


def _run_dir(user_id, dataset_id, base_model=_PERSISTED, family=None,
             variant=_PERSISTED) -> str:
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # ai-toolkit écrit ses checkpoints/samples dans <training_folder>/<name>/
    # où name = 'lora_<trigger>' (cf. build_job_config). On pointe ce sous-dossier.
    # `base_model` cible le run d'une base PRÉCISE (sélection UI) ; `family` cible la
    # famille sélectionnée (Krea vs Z-Image) - sans quoi le panneau montre les
    # checkpoints du mauvais run quand deux familles partagent le même trigger.
    return str(_run_root(ds, base_model, family, variant)
               / f'lora_{_safe_trigger(ds)}')


def open_training_folder(user_id, dataset_id, target='loras', family=None,
                         base_model=_PERSISTED, variant=_PERSISTED) -> str:
    """Ouvre dans l'explorateur de fichiers du POSTE (app locale mono-utilisateur,
    le navigateur tourne sur la même machine) le dossier demandé :
    'loras' → dossier d'import ComfyUI de la famille (loras/krea, loras/sdxl,
    loras/z image) ; 'run' → dossier HAUT du run courant (base+famille) : il porte
    training.log, et les checkpoints sont dans son sous-dossier lora_<trigger> ;
    'dataset' → dossier des images du dataset (data/datasets/<id>/ — où « 💾 Write
    .txt files » dépose les captions sidecar ; aucune dépendance ai-toolkit).
    Cibles FIXES résolues côté serveur — le client n'envoie jamais de chemin.
    Crée le dossier au besoin (avant un premier import il n'existe pas encore).
    Retourne le chemin ouvert."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if target == 'run':
        # The run's TOP folder, not the `lora_<trigger>` save_root below it.
        # That save_root is created by ai-toolkit at its first save, so a run
        # that died at boot has none — opening it used to CREATE an empty
        # folder and reveal that, while `training.log` sat one level up, in
        # the very folder the failure message sends people to (reported by
        # wannadecryptor on Discord). The top folder shows the log AND leads
        # to the checkpoints.
        path = str(_run_root(ds, base_model, family, variant))
    elif target == 'loras':
        path = _lora_dest_dir(ds, family)
    elif target == 'dataset':
        path = fds._dataset_dir(dataset_id)
    else:
        raise ValueError('unknown folder target')
    os.makedirs(path, exist_ok=True)
    if os.name == 'nt':
        os.startfile(path)                                   # Explorateur Windows
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])
    logger.info('open folder (%s): %s', target, path)
    return path


def list_checkpoints(user_id, dataset_id, base_model=_PERSISTED, family=None,
                     variant=_PERSISTED) -> list[dict]:
    """Checkpoints .safetensors du run de la base+famille données (absentes → persistées),
    triés par step croissant. Retour: [{step:int, filename:str, final?:bool}].

    Inclut le fichier FINAL `lora_<trigger>.safetensors` (écrit à la fin d'un run
    abouti, SANS numéro de step) : c'est le résultat terminé, et le regex numéroté
    l'excluait → le LoRA fini était invisible/non importable depuis le panneau."""
    run = _run_dir(user_id, dataset_id, base_model, family, variant)
    if not os.path.isdir(run):
        return []
    out = []
    for f in os.listdir(run):
        m = _CK_RE.search(f)
        if m:
            out.append({'step': int(m.group(1)), 'filename': f})
    out.sort(key=lambda c: c['step'])
    # Fichier final (run = .../lora_<trigger> → lora_<trigger>.safetensors).
    final_name = os.path.basename(run) + '.safetensors'
    if os.path.isfile(os.path.join(run, final_name)):
        last = out[-1]['step'] if out else 0
        out.append({'step': last, 'filename': final_name, 'final': True})
    # Provenance annotation: which dataset VERSION most plausibly produced
    # each file (newest registry record older than the file). Pre-feature
    # datasets have no records -> no annotation, shape unchanged otherwise.
    from . import checkpoint_registry
    ds = fds.get_dataset(user_id, dataset_id)
    fam = _train_type(ds, family) if ds else None
    effective_base = (
        getattr(ds, 'train_base_model', None)
        if base_model is _PERSISTED else base_model)
    effective_variant = (
        getattr(ds, 'train_variant', None)
        if variant is _PERSISTED else variant)
    effective_variant = (
        str(effective_variant or _default_variant_for(fam)).strip().lower()
        if fam else effective_variant)
    for c in out:
        try:
            rec = checkpoint_registry.record_for_mtime(
                dataset_id, fam,
                os.path.getmtime(os.path.join(run, c['filename'])),
                base_model=effective_base or '',
                variant=effective_variant,
                source=('local', 'legacy'))
        except OSError:
            rec = None
        if rec is not None:
            c['version'] = rec.version
            c['source'] = rec.source
            # The record that PRODUCED this file. A continuation resolves its
            # lineage parent (and the LoRA geometry it must keep) from HERE —
            # `run_id` below is a cloud id for a cloud record, which is not a
            # record key and cannot be used for either.
            c['record_id'] = rec.id
            c['trained_at'] = rec.created_at.isoformat() if rec.created_at else None
            # Run identity for the ☁/💻 #N chip + deep-link on the local group
            # header — the same run the deployed file will be tagged with.
            if rec.source == 'cloud' and rec.cloud_run_id:
                c['run_id'], c['run_source'] = rec.cloud_run_id, 'cloud'
            else:
                c['run_id'], c['run_source'] = rec.id, 'local'
            # The FINAL save has no step in its name, so it was filed under the
            # last NUMBERED one (2750 for a 3000-step run) — where the dedup of
            # the ▶ Continue list swallowed it, making the run's real end
            # unresumable from here. The ◉ Graph reads a cloud run's staging and
            # numbers the same file at the run's target (3000): two views, two
            # truths, and a pill the panel then refused. Number it like the graph
            # does — its own run's target — whenever the record knows better.
            if c.get('final') and (rec.steps or 0) > c['step']:
                c['step'] = rec.steps
    # Exact resume is checkpoint-specific.  Old saves intentionally remain
    # usable, but are labelled weights-only instead of inheriting ai-toolkit's
    # mutable optimizer.pt by accident.  Verification is done here so the UI
    # never advertises a bundle whose bytes no longer match its manifest.
    from . import training_state_bundle
    try:
        state_by_step = {}
        for inspection in training_state_bundle.list_bundles(
                run, verify=True, include_invalid=True):
            if inspection.completed_step is not None:
                state_by_step.setdefault(inspection.completed_step, inspection)
    except Exception:
        logger.warning('could not inspect training-state bundles for %s',
                       run, exc_info=True)
        state_by_step = {}
    unavailable_reason = None
    try:
        from . import aitoolkit_state_bridge
        status_path = Path(run).parent / '.lds-state' / 'bridge-status.json'
        bridge_status = aitoolkit_state_bridge.read_status(status_path)
        if not isinstance(bridge_status, dict):
            raise ValueError('missing or invalid bridge status')
        bridge_reasons = bridge_status.get('reasons')
        if (bridge_status.get('status') in ('unsupported', 'save_unsupported')
                and isinstance(bridge_reasons, list)):
            bridge_reasons = [
                str(reason).strip() for reason in bridge_reasons
                if str(reason).strip()
            ]
            if bridge_reasons:
                unavailable_reason = (
                    'Full-state resume is unavailable for this run: '
                    + '; '.join(bridge_reasons))
    except (OSError, ValueError, TypeError):
        pass
    for c in out:
        state = state_by_step.get(int(c['step']))
        if state is not None:
            c['resume_state'] = state.to_ui_dict()
        else:
            c['resume_state'] = {
                'bundle_id': None,
                'status': 'missing',
                'integrity': 'unchecked',
                'state_level': 'weights',
                'reason': unavailable_reason or (
                    'Legacy checkpoint: only the LoRA weights were saved; '
                    'optimizer, scheduler, RNG and dataloader state are unavailable.'),
                'size_bytes': 0,
                'capabilities': ['weights'],
            }
    out.sort(key=lambda c: (c['step'], bool(c.get('final'))))
    return out


def has_local_checkpoints(user_id, dataset_id, base_model=_PERSISTED,
                          family=None, variant=_PERSISTED) -> bool:
    """Cheap local-training evidence without bundle verification.

    Baseline backfill needs to know only whether a public checkpoint exists.
    Keeping this scan separate avoids hashing every exact-state bundle twice in
    one checkpoints API request while preserving the required ordering:
    baseline first, fully annotated listing second.
    """
    run = _run_dir(user_id, dataset_id, base_model, family, variant)
    try:
        final_name = os.path.basename(run) + '.safetensors'
        with os.scandir(run) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.name == final_name or _CK_RE.search(entry.name):
                    return True
    except OSError:
        return False
    return False


def resume_source_checkpoint(checkpoints, step):
    """The save a resume at `step` actually loads, out of a list_checkpoints()
    list — so a continuation can read ITS provenance (`record_id`) and ITS
    geometry instead of guessing from the lane. Ties (a numbered save and the
    bare final at the same step) prefer the numbered file, the same rule every
    ▶ Continue path already uses. None when nothing sits at that step."""
    matches = [c for c in (checkpoints or []) if c.get('step') == step]
    if not matches:
        return None
    return min(matches, key=lambda c: bool(c.get('final')))


_LEGACY_LOKR_FULL_RANK_RESUME_ERROR = (
    'cannot continue this legacy LoKr checkpoint because its `lokr_full_rank` '
    'topology was never recorded. ai-toolkit defaults have changed, so forcing '
    'a mode now could load the weights into a different network. Start a fresh '
    'LoKr run, or continue from a checkpoint with recorded LoKr topology.')


def legacy_lokr_resume_error(parent_geometry, fallback_settings=None):
    """Return a hard-stop reason when a resume would guess LoKr full-rank mode.

    A recorded LoKr parent is authoritative: it must carry a real boolean
    ``lokr_full_rank`` value.  When a pre-registry parent has no adapter type,
    inspect the frozen/live settings that the continuation would otherwise emit;
    a legacy LoKr setting without that fact is equally unsafe.  A known LoRA
    parent stays untouched here (the normal type-conflict check handles a
    LoRA-to-LoKr switch separately).
    """
    geometry = parent_geometry if isinstance(parent_geometry, dict) else {}
    parent_type = geometry.get('network_type')
    candidate = (geometry if parent_type == 'lokr'
                 else fallback_settings if parent_type not in _NETWORK_TYPE_CHOICES
                 else None)
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except (TypeError, ValueError):
            candidate = None
    if (isinstance(candidate, dict)
            and candidate.get('network_type') == 'lokr'
            and not isinstance(candidate.get('lokr_full_rank'), bool)):
        return _LEGACY_LOKR_FULL_RANK_RESUME_ERROR
    return None


def describe_geometry_conflict(parent_geometry, rank, alpha, *, network_type=None,
                               lokr_factor=None, lokr_full_rank=None,
                               conv=None, conv_alpha=None):
    """Explain a known checkpoint/topology mismatch, or return ``None``.

    Adapter type, rank/alpha, and LoKr's decomposition parameters all determine
    the tensors a checkpoint can load.  The parent geometry is intentionally
    partial: missing provenance keys are legacy *unknowns*, not today's defaults,
    so only facts explicitly recorded for the checkpoint are enforced.
    """
    geo = parent_geometry or {}
    parent_type = geo.get('network_type')
    if (parent_type in _NETWORK_TYPE_CHOICES
            and network_type in _NETWORK_TYPE_CHOICES
            and parent_type != network_type):
        return (f'this checkpoint was trained as {"LoKr" if parent_type == "lokr" else "LoRA"}, '
                f'and this run would use {"LoKr" if network_type == "lokr" else "LoRA"}. '
                'The adapter type is fixed by the checkpoint weights; restore the '
                'original network type in Training settings, or start a fresh run.')
    want_r, want_a = geo.get('rank'), geo.get('alpha')
    bad = ((want_r is not None and rank is not None and int(want_r) != int(rank))
           or (want_a is not None and alpha is not None and int(want_a) != int(alpha)))
    if bad:
        return (f'this checkpoint was trained at rank {want_r} / alpha {want_a}, and '
                f'this run would use rank {rank} / alpha {alpha}. A LoRA\'s rank is '
                'fixed by its weights — continuing it at another rank cannot load '
                'them. Set rank and alpha back in Training settings, or start a '
                'fresh run instead of continuing.')
    if parent_type == network_type == 'lokr':
        parent_factor = geo.get('lokr_factor')
        if (parent_factor is not None and lokr_factor is not None
                and parent_factor != lokr_factor):
            return (f'this LoKr checkpoint was trained with factor {parent_factor}, and '
                    f'this run would use factor {lokr_factor}. LoKr factor changes the '
                    'checkpoint tensor geometry; restore the original factor or start '
                    'a fresh run.')
        if ('lokr_full_rank' in geo and isinstance(lokr_full_rank, bool)
                and geo['lokr_full_rank'] != lokr_full_rank):
            return ('this LoKr checkpoint was trained with '
                    f'lokr_full_rank={geo["lokr_full_rank"]}, and this run would use '
                    f'lokr_full_rank={lokr_full_rank}. Full-rank mode changes the '
                    'checkpoint tensor geometry; restore the original mode or start '
                    'a fresh run.')
    if parent_type == network_type == 'lora':
        for key, current, label in (
                ('conv', conv, 'Conv rank'),
                ('conv_alpha', conv_alpha, 'Conv alpha')):
            if key not in geo or geo[key] == current:
                continue
            before = 'off' if geo[key] is None else geo[key]
            after = 'off' if current is None else current
            return (f'this LoRA checkpoint was trained with {label} {before}, and '
                    f'this run would use {label} {after}. Conv LoRA changes the '
                    'checkpoint tensor geometry; restore the original Conv settings '
                    'or start a fresh run.')
    return None


def checkpoint_file_path(user_id, dataset_id, filename, base_model=_PERSISTED,
                         family=None, variant=_PERSISTED):
    """Absolute path of ONE local run-dir checkpoint for a browser download, or
    None if it isn't a real save of that run. Anti path-traversal: the filename
    must appear in this run's list_checkpoints (the same whitelist import_checkpoint
    uses), so `..`/absolute paths never resolve. Powers the ◉ Graph's per-checkpoint
    ⬇ for local runs, mirroring the cloud endpoint's staging serve."""
    run = _run_dir(user_id, dataset_id, base_model, family, variant)
    if not os.path.isdir(run):
        return None
    allowed = {c['filename'] for c in list_checkpoints(
        user_id, dataset_id, base_model, family, variant)}
    if filename not in allowed:
        return None
    path = os.path.join(run, filename)
    return path if os.path.isfile(path) else None


def import_checkpoint(user_id, dataset_id, filename, base_model=_PERSISTED, family=None,
                      src_dir=None, version=None, variant=_PERSISTED,
                      run_id=None, run_source=None, return_meta=False):
    """Copie le checkpoint choisi vers le dossier loras de ComfyUI : loras/z image/
    pour Z-Image, loras/sdxl/ pour SDXL, loras/krea/ pour Krea (routage par famille,
    pour ne pas polluer le Test Studio Z-Image). Anti path-traversal :
    le filename doit appartenir à la liste des checkpoints du run.

    Le nom de DESTINATION encode la base et la recette d'entraînement : ai-toolkit
    écrit toujours `lora_<trigger>_<step>.safetensors` quel que soit le modèle de
    base (le `name` du job n'est pas base-aware), donc un LoRA entraîné sur un
    merge ComfyUI et un autre entraîné sur la base officielle produisent des
    fichiers IDENTIQUES qui, une fois copiés dans le dossier partagé de ComfyUI,
    sont indiscernables et s'écrasent au même step. On insère ici le tag du merge
    (`lora_<trigger>_<step>_<merge>_<recipe>.safetensors`) pour les rendre
    reconnaissables ET éviter la collision. Le fichier
    source ai-toolkit n'est pas renommé (l'auto-resume continue de fonctionner).

    `base_model`/`family` ciblent le run d'une base+famille précises (sélection UI) ;
    absents → persistés. Run dir, whitelist, dossier ET suffixe de destination
    utilisent la MÊME base+famille → cohérent (un LoRA Krea part bien en loras/krea).

    `src_dir` (cloud seam) : le checkpoint est lu LÀ (dossier de staging où le pod a
    déposé le résultat téléchargé) au lieu du run ai-toolkit local - aucun besoin
    d'ai-toolkit configuré (ni _run_dir(), ni list_checkpoints(), qui appellent tous
    deux _output_dir()). La whitelist ici est PUREMENT anti-traversal : tout
    .safetensors réellement présent dans src_dir est autorisé (pas de filtre de
    forme _CK_RE — le checkpoint FINAL d'un run abouti, `lora_<trigger>.safetensors`,
    n'a pas de suffixe de step et doit passer). Défaut (None) = comportement
    historique inchangé."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if src_dir:
        run_dir = str(src_dir)
        try:
            allowed = {f for f in os.listdir(run_dir)
                       if f.lower().endswith('.safetensors')}
        except OSError:
            allowed = set()
    else:
        run_dir = _run_dir(user_id, dataset_id, base_model, family, variant)
        allowed = {c['filename'] for c in list_checkpoints(
            user_id, dataset_id, base_model, family, variant)}
    if filename not in allowed:
        raise ValueError('unknown checkpoint')
    # Arch guard: read the LoRA's REAL family from its header and refuse a deploy
    # that would land it in the wrong ComfyUI folder. ComfyUI silently drops every
    # incompatible key, so a Z-Image LoRA copied under loras/krea/ tests as a pure
    # no-op with no error anywhere (the 2026-07-13 incident). Undetectable header →
    # pass (no false block); only a POSITIVE cross-namespace mismatch stops here.
    fam_target = _train_type(ds, family)
    detected = detect_lora_arch(os.path.join(run_dir, filename))
    if lora_arch_conflicts(detected, fam_target):
        det_lbl = _LORA_ARCH_LABEL.get(detected, detected)
        tgt_lbl = _LORA_ARCH_LABEL.get(fam_target, fam_target)
        raise ValueError(
            f'this file is a {det_lbl} LoRA — deploy it under the {det_lbl} '
            f'family, not {tgt_lbl}.')
    # Déploiement routé par famille : sdxl → loras/sdxl, krea → loras/krea, sinon
    # « z image » (ne pollue pas le Test Studio Z-Image ; un LoRA Krea atterrit
    # directement dans le dossier lu par le menu de génération Krea).
    dest_dir = _lora_dest_dir(ds, family)
    os.makedirs(dest_dir, exist_ok=True)
    tag = _dest_base_tag(ds, base_model, family, variant)
    # Dataset-version suffix (_v3): makes successive dataset states
    # distinguishable in the ComfyUI/Test Studio dropdowns AND prevents a
    # cloud/local re-run of a CHANGED dataset from silently overwriting the
    # deployed LoRA of the previous version. `version` is passed explicitly by
    # the cloud import (the run knows its version); local imports resolve the
    # file's run via the provenance registry (file mtime vs launch times).
    # No registry rows (pre-feature datasets) -> no suffix, names unchanged.
    # run_tag_*: the identity of the run that produced this checkpoint, encoded
    # into the deployed name (_rc<id>/_rl<id>) so two runs of the SAME recipe do
    # not overwrite each other. Cloud imports pass it from the route; local
    # imports resolve it from the same provenance record that gives the version.
    run_tag_source, run_tag_id = run_source, run_id
    if version is None and not src_dir:
        from . import checkpoint_registry
        try:
            mtime = os.path.getmtime(os.path.join(run_dir, filename))
            rec = checkpoint_registry.record_for_mtime(
                dataset_id, _train_type(ds, family), mtime,
                base_model=(
                    getattr(ds, 'train_base_model', None)
                    if base_model is _PERSISTED else base_model) or '',
                variant=str(
                    (getattr(ds, 'train_variant', None)
                     if variant is _PERSISTED else variant)
                    or _default_variant_for(_train_type(ds, family))
                ).strip().lower(),
                source=('local', 'legacy'))
        except OSError:
            rec = None
        if rec is not None:
            version = rec.version
            if run_tag_id is None:
                # A cloud launch recorded locally addresses by its pod-run id
                # (matches the cloud import path AND the ☁ #N chip); everything
                # else is a local record -> its TrainingRunRecord id.
                if rec.source == 'cloud' and rec.cloud_run_id:
                    run_tag_source, run_tag_id = 'cloud', rec.cloud_run_id
                else:
                    run_tag_source, run_tag_id = 'local', rec.id
    stem, ext = os.path.splitext(filename)
    # Cloud jobs are named `lds<run>_u<user>_<trigger>_<base>` on the pod, so
    # their checkpoints arrive as `lds12_ulocal_tata_cv_Krea-2-Raw_000000250`.
    # Deployed as-is, that stem is invisible to every trigger-prefix matcher
    # (Test Studio's `lora_<trigger>_…` whitelist, labels) — "my cloud
    # checkpoints are unusable", user-reported — and the deploy suffix used to
    # re-append a base tag the stem already carried. Normalize to the LOCAL
    # ai-toolkit convention at deploy time: `lora_<trigger>[_<step>]`, rebuilt
    # from the dataset's own trigger (no string surgery on the tag).
    if re.match(r'^lds\d+_u[0-9A-Za-z]+_', stem):
        step = re.search(r'_(\d{6,10})$', stem)
        stem = f'lora_{_safe_trigger(ds)}' + (f'_{step.group(1)}' if step else '')
    # <base tag><run tag>_v<N>: the run tag sits BEFORE the version suffix so the
    # existing `_v\d+$` strip in list_imported_checkpoints still finds it.
    runtag = _run_tag(run_tag_source, run_tag_id)
    suffix = f'{tag}{runtag}' + (f'_v{int(version)}' if version else '')
    dest_name = f'{stem}{suffix}{ext}' if suffix else filename
    dest = os.path.join(dest_dir, dest_name)
    src_path = os.path.join(run_dir, filename)
    # Last-resort anti-clobber: the run tag already keeps distinct runs apart, so
    # a same-name collision means DIFFERENT content is already there — a legacy
    # untagged import, or (theoretically) a run-id reuse. Never overwrite it in
    # silence: take the next free `_N` suffix (the _dest_base_tag philosophy).
    # Identical bytes (an idempotent re-import) are left to overwrite in place —
    # no proliferation of `_2`, `_3` copies on repeated clicks.
    collision = False
    if os.path.exists(dest) and not _same_file(src_path, dest):
        root, ext2 = os.path.splitext(dest_name)
        i = 2
        while os.path.exists(os.path.join(dest_dir, f'{root}_{i}{ext2}')):
            i += 1
        dest_name = f'{root}_{i}{ext2}'
        dest = os.path.join(dest_dir, dest_name)
        collision = True
    shutil.copy2(src_path, dest)
    logger.info(f'import checkpoint {filename} -> {dest}'
                + (' (renamed to avoid overwriting a different LoRA)'
                   if collision else ''))
    if return_meta:
        return {'dest': dest, 'name': dest_name, 'collision': collision}
    return dest


def _same_file(a, b) -> bool:
    """True when two files hold identical bytes (size + content). Used so a
    re-import of the very same checkpoint overwrites in place instead of
    spawning a `_2` copy, while genuinely different content is never clobbered."""
    try:
        return filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def list_imported_checkpoints(user_id, dataset_id, family=None) -> list[dict]:
    """LoRA de CE dataset déjà déployés dans le dossier loras de la FAMILLE demandée
    (chargeables par le Test Studio / la page generate). [{filename, label}].
    `family` (sélecteur UI) prime sur le train_type persisté : sans ça, la liste
    « IN COMFYUI (loras/…) » montrait toujours la famille persistée (ex. Krea) même
    quand l'utilisateur regardait la page Z-Image ou SDXL.

    Single-user app: no ownership DB to filter against (SRC's list_test_checkpoints
    consulted lora_ownership to hide LoRA belonging to OTHER users) -- everything on
    disk that matches this dataset's trigger boundary IS this dataset's checkpoint.
    A direct filesystem scan of the family's deploy folder replaces that call.
    `filename` is returned in LoraLoader form (family-subfolder\\name.safetensors),
    matching delete_imported_checkpoint's path resolution."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return []
    fam = _train_type(ds, family)
    prefix = f'lora_{_safe_trigger(ds)}'
    # EVERY loras root, not just the deploy one: a LoRA deployed before the app
    # learned to read extra_model_paths.yaml (GitHub #25) sits in the old default
    # folder, and dropping it from this list would make an existing file
    # undeletable from the app for no reason the user can see.
    dirs = [d for d in _lora_family_dirs(fam) if os.path.isdir(d)]
    if not dirs:
        return []
    from ..utils.comfyui import format_trained_lora_label
    # Cloud-trained checkpoints are auto-imported into the same folder but
    # named after the pod job (`lds<N>_<run>…`), not `lora_<trigger>…` — the
    # prefix filter alone hid them from the "IN COMFYUI" list even though the
    # files were right there (user-observed 2026-07-13). Accept any filename
    # that IS a known cloud checkpoint of THIS dataset.
    cloud_names = set()
    cloud_prefixes = set()
    try:
        from ..models import CloudTrainingRun
        for r in CloudTrainingRun.query.filter_by(dataset_id=dataset_id).all():
            if r.checkpoint_local_path:
                cloud_names.add(os.path.basename(r.checkpoint_local_path))
            # Every staging file of this run starts with its pod-job prefix
            # (`lds<id>_…`, see cloud_training job_name). Matching on the prefix
            # covers EVERY harvested epoch AND survives the `_<base_tag>` +
            # `_v<N>` suffixes import_checkpoint appends to the deployed name —
            # the exact-basename match above misses both (user-observed
            # 2026-07-13: imports succeeded but "in ComfyUI" stayed at 0).
            cloud_prefixes.add(f'lds{r.id}_')
    except Exception:
        pass
    subfolder = _FAMILY_SUBDIR.get(fam, 'z image')
    out, seen = [], set()
    for dest_dir, fn in sorted(((d, fn) for d in dirs for fn in os.listdir(d)),
                               key=lambda pair: pair[1]):
        if not fn.lower().endswith('.safetensors'):
            continue
        # Same name in two roots = ComfyUI loads the higher-priority one; list it
        # once, from that same root, so the app never shows a file ComfyUI shadows.
        if fn in seen:
            continue
        seen.add(fn)
        # deployed cloud names may carry the _v<N> dataset-version suffix —
        # strip it before matching against the staging basenames
        stem = re.sub(r'_v\d+(?=\.safetensors$)', '', fn)
        if not _trigger_boundary(fn, prefix) \
                and fn not in cloud_names and stem not in cloud_names \
                and not any(fn.startswith(p) for p in cloud_prefixes):
            continue
        # Pass the dataset's real trigger so a multi-token trigger (`leg_behind`)
        # labels faithfully rather than truncating to `leg` and leaking `behind`
        # into the base tag (the deployed filename alone can't tell the trigger's
        # underscores from the field separators).
        entry = {'filename': os.path.join(subfolder, fn),
                 'label': format_trained_lora_label(
                     fn, fam, trigger=getattr(ds, 'trigger_word', None)) or fn}
        # Source run of this deployed file (☁/💻 #N chip). Files imported before
        # run tagging carry no tag -> (None, None): shown as "run unknown", never
        # renamed retroactively (they stay listed and deletable exactly as-is).
        rsrc, rid = parse_deployed_run(fn)
        if rid is not None:
            entry['run_id'], entry['run_source'] = rid, rsrc
        # Retrofit signal for already-deployed files: if the header's real arch
        # contradicts THIS folder's family, flag it (mislabelled imports from the
        # pre-6952b11 wrong-arch bug) so the panel can badge it. No file is moved.
        detected = detect_lora_arch(os.path.join(dest_dir, fn))
        if lora_arch_conflicts(detected, fam):
            entry['arch_mismatch'] = detected
            entry['arch_label'] = _LORA_ARCH_LABEL.get(detected, detected)
        out.append(entry)
    return out


def list_all_deployed_checkpoints(user_id) -> list[dict]:
    """EVERY LoRA this app has deployed into ComfyUI — across every dataset and
    every family — so one screen can undeploy a pile of them in one pass.

    WHY THIS EXISTS. Undeploying was reachable one pill at a time, buried in a
    node's popover, with no way to see how many were deployed at all. Asked for
    by the maintainer: "a button that lists every deployed LoRA and lets me tick
    the ones to undeploy, to make it faster."

    WHY IT DELEGATES INSTEAD OF SCANNING THE FOLDERS ITSELF, which would be a
    dozen lines shorter: `list_imported_checkpoints` is the one place that knows
    which file in `loras/<family>/` BELONGS TO THIS APP — the `lora_<trigger>`
    boundary plus the cloud runs' own staging prefixes. A plain directory scan
    would also return the LoRA the user downloaded from Civitai and dropped in
    the same folder, and this list feeds a DELETE button. Offering someone their
    own files to remove, in a screen labelled "undeploy what the app deployed",
    is the one failure this must not have.

    Cost: one call per (dataset, family). Each returns [] immediately when the
    family folder does not exist, which is the common case — a machine trains one
    or two families, not six.

    Entries carry their dataset so the caller can address the undeploy route,
    which is dataset-scoped: {dataset_id, dataset_name, family, filename, label}
    plus whatever `list_imported_checkpoints` already stamps (run_id/run_source,
    arch_mismatch).
    """
    out, seen = [], set()
    for ds in fds.list_datasets(user_id) or []:
        if ds.id is None:
            continue
        for fam in _FAMILY_SUBDIR:
            try:
                rows = list_imported_checkpoints(user_id, ds.id, family=fam)
            except Exception:
                # One unreadable family folder must not empty the whole list —
                # the other five are still actionable.
                logger.exception('deployed list: dataset %s family %s failed',
                                 ds.id, fam)
                continue
            for row in rows:
                # ONE row per file on disk. Two datasets sharing a trigger word
                # both claim the same deployed file (the boundary match is on the
                # name, and the name is all there is); listing it twice would
                # offer the same removal twice and fail the second click.
                key = (fam, os.path.normcase(row.get('filename') or ''))
                if key in seen:
                    continue
                seen.add(key)
                out.append({**row, 'dataset_id': ds.id,
                            'dataset_name': ds.name, 'family': fam})
    return out


def lora_deploy_dir(user_id, dataset_id, family=None) -> str:
    """Absolute ComfyUI loras deploy folder for this dataset's FAMILY. Raises
    RuntimeError when ComfyUI is unconfigured (the caller — the full-backup
    service — treats that as "can't place restored LoRA here", never fatal).
    Public wrapper so backup/restore never reaches into `_lora_dest_dir`."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    return _lora_dest_dir(ds, family)


def deployed_lora_paths(user_id, dataset_id, family=None) -> list[str]:
    """Absolute paths of THIS dataset's LoRA files already deployed in the
    family's ComfyUI loras folders, for the "include trained LoRAs" backup toggle.
    [] when ComfyUI is unconfigured or nothing is deployed. Reuses
    list_imported_checkpoints so cloud-named epochs are captured too."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return []
    fam = _train_type(ds, family)
    out = []
    for c in list_imported_checkpoints(user_id, dataset_id, family=family):
        p = _resolve_deployed_path(fam, c.get('filename', ''))
        if p:
            out.append(p)
    return out


def deployed_file_present(user_id, dataset_id, filename, family=None) -> bool:
    """Is this deployed name still a real file in the family's loras folders?

    `delete_imported_checkpoint` answers 'unknown checkpoint' to two situations
    that are NOT the same thing for the person who clicked: the file was deleted
    by hand since the list was drawn (nothing to do — they already have the
    outcome they wanted), or the name is not one this app deployed (a refusal
    they should see). The whitelist alone cannot tell them apart, because a file
    that is gone drops out of the whitelist too. The disk can.
    """
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return False
    try:
        return bool(_resolve_deployed_path(_train_type(ds, family), filename))
    except Exception:
        return False


def delete_imported_checkpoint(user_id, dataset_id, filename, family=None) -> str:
    """Supprime un checkpoint déployé du dossier loras de ComfyUI. Garde-fous :
    le filename doit appartenir aux checkpoints importés du dataset (whitelist,
    famille-scopée) ET le chemin résolu doit rester dans le dossier loras de la
    FAMILLE sélectionnée (z image / sdxl / krea) - anti path-traversal, fail-closed.
    `family` (menu UI) prime sur le train_type persisté, comme la liste affichée."""
    ds = fds.get_dataset(user_id, dataset_id)
    allowed = {c['filename'] for c in list_imported_checkpoints(user_id, dataset_id, family=family)}
    if filename not in allowed:
        raise ValueError('unknown checkpoint')
    # ds is guaranteed truthy here: an unowned/missing dataset makes
    # list_imported_checkpoints return [] above, which already raised.
    # Searched across every loras root, so a LoRA deployed before the app read
    # extra_model_paths.yaml (GitHub #25) is deletable where it actually is.
    dest = _resolve_deployed_path(_train_type(ds, family), filename)
    if not dest:
        raise ValueError('file not found')
    # trash, never destroy: a wrong click on a deployed LoRA is recoverable
    # until 'Empty trash' in Settings.
    from . import trash
    trash.send_to_trash(dest, context=f'lora_ds{dataset_id}')
    logger.info(f'trashed imported checkpoint {dest}')
    return os.path.basename(dest)


def _local_training_active_for(dataset_id) -> bool:
    """True while THIS dataset trains locally — its run dir is being written
    (deleting a checkpoint ai-toolkit is about to rewrite invites corruption)."""
    try:
        if not queue_manager._get_system_state('training_in_progress'):
            return False
        active_ds = queue_manager._get_system_state('training_dataset_id')
        return active_ds is not None and int(active_ds) == int(dataset_id)
    except Exception:
        return False


def delete_checkpoint(user_id, dataset_id, filename, base_model=_PERSISTED,
                      family=None, variant=_PERSISTED) -> str:
    """Move ONE run-dir checkpoint to the trash. Whitelisted against
    list_checkpoints (anti path-traversal), refused while this dataset trains
    locally. Returns the trashed filename."""
    if _local_training_active_for(dataset_id):
        raise ValueError('this dataset is training right now — stop the run '
                         'before deleting its checkpoints')
    allowed = {c['filename'] for c in
               list_checkpoints(user_id, dataset_id, base_model, family, variant)}
    if filename not in allowed:
        raise ValueError('unknown checkpoint')
    run_dir = _run_dir(user_id, dataset_id, base_model, family, variant)
    from . import trash
    trash.send_to_trash(os.path.join(run_dir, filename),
                        context=f'ckpt_ds{dataset_id}')
    return filename


def cleanup_checkpoints(user_id, dataset_id, keep, base_model=_PERSISTED,
                        family=None, variant=_PERSISTED) -> dict:
    """'Clean up this run': trash every run-dir checkpoint NOT in `keep`
    (typically the final + the best-epoch pick). Returns {'removed', 'kept'}."""
    if _local_training_active_for(dataset_id):
        raise ValueError('this dataset is training right now — stop the run '
                         'before cleaning its checkpoints')
    keep_set = {str(k) for k in (keep or [])}
    run_dir = _run_dir(user_id, dataset_id, base_model, family, variant)
    from . import trash
    removed = 0
    for c in list_checkpoints(user_id, dataset_id, base_model, family, variant):
        if c['filename'] in keep_set:
            continue
        try:
            trash.send_to_trash(os.path.join(run_dir, c['filename']),
                                context=f'cleanup_ds{dataset_id}')
            removed += 1
        except OSError as e:
            logger.warning('cleanup: could not trash %s: %s', c['filename'], e)
    return {'removed': removed, 'kept': sorted(keep_set)}


def _dir_size(path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def dataset_disk_usage(user_id, dataset_id, base_model=_PERSISTED, family=None,
                       variant=_PERSISTED) -> dict:
    """Where this dataset's training bytes live: the selected run dir, the
    cloud staging dirs of its runs, and its deployed LoRA. Best-effort."""
    out = {'run_dir_bytes': 0, 'cloud_staging_bytes': 0, 'deployed_bytes': 0}
    try:
        rd = _run_dir(user_id, dataset_id, base_model, family, variant)
        if os.path.isdir(rd):
            out['run_dir_bytes'] = _dir_size(rd)
    except Exception:
        pass
    try:
        from ..models import CloudTrainingRun
        for r in CloudTrainingRun.query.filter_by(dataset_id=dataset_id).all():
            if r.staging_dir and os.path.isdir(r.staging_dir):
                out['cloud_staging_bytes'] += _dir_size(r.staging_dir)
    except Exception:
        pass
    try:
        ds = fds.get_dataset(user_id, dataset_id)
        fam = _train_type(ds, family)
        for c in list_imported_checkpoints(user_id, dataset_id, family=family):
            p = _resolve_deployed_path(fam, c['filename'])
            try:
                out['deployed_bytes'] += os.path.getsize(p) if p else 0
            except OSError:
                pass
    except Exception:
        pass
    out['total_bytes'] = sum(v for k, v in out.items() if k.endswith('_bytes'))
    return out


def _trigger_boundary(name: str, prefix: str) -> bool:
    """`name` commence par `prefix` ET la suite est vide ou commence par `_`/`.` -
    frontière de trigger EXACTE. Évite que « Lola » attrape « Lola2 »/« Lola69382 »
    (le caractère après le préfixe doit être un séparateur, pas un chiffre/lettre)."""
    if not name.startswith(prefix):
        return False
    rest = name[len(prefix):]
    return rest == '' or rest[0] in '_.'


def purge_training_artifacts(user_id, trigger_safe) -> list[str]:
    """Supprime TOUS les artefacts d'entraînement d'un (user, trigger), appelé à la
    suppression d'un dataset : LoRA déployés dans ComfyUI (z image + sdxl + krea), run
    ai-toolkit (output/), export (datasets/) et job config (config/generated/).

    Sécurité : matching sur la FRONTIÈRE EXACTE du trigger (jamais un sibling type
    Lola vs Lola2) ; les noms viennent d'os.listdir (bare, pas de path-traversal) ;
    trigger vide → no-op (sinon `u{user}_` balaierait tout). Retourne les chemins
    retirés (pour log/affichage). Idempotent : un 2e appel ne retire plus rien.

    Each backend (ComfyUI loras dir / ai-toolkit output+datasets dirs) is probed
    independently -- an unconfigured backend just yields no roots to sweep for
    that step instead of aborting the whole purge (this runs from
    face_dataset_service.delete_dataset as best-effort cleanup)."""
    trigger_safe = (trigger_safe or '').strip()
    if not trigger_safe or user_id in (None, ''):
        return []
    removed: list[str] = []
    run_prefix = f'u{user_id}_{trigger_safe}'    # ex. u1_Lola69382
    lora_prefix = f'lora_{trigger_safe}'         # ex. lora_Lola69382
    # 1) LoRA déployés dans ComfyUI (z image + sdxl + krea + flux + flux2klein + anima
    # séparés), dans CHAQUE racine loras : un LoRA déployé avant le correctif #25
    # vit dans l'ancien dossier par défaut et doit partir avec le dataset.
    lora_roots = []
    for fam in _FAMILY_SUBDIR:
        lora_roots += _lora_family_dirs(fam)
    for root in lora_roots:
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if fn.endswith('.safetensors') and _trigger_boundary(fn, lora_prefix) and os.path.isfile(p):
                try:
                    trash.send_to_trash(p, context=f'training-{trigger_safe}')
                    removed.append(p)
                except OSError as e:
                    logger.warning('purge: trash %s échoué : %s', p, e)
    # 2) run output + 3) export datasets (dossiers entiers)
    output_datasets_roots = []
    for accessor in (_output_dir, _datasets_dir):
        try:
            output_datasets_roots.append(str(accessor()))
        except RuntimeError:
            pass
    for root in output_datasets_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if _trigger_boundary(name, run_prefix) and os.path.isdir(p):
                try:
                    trash.send_to_trash(p, context=f'training-{trigger_safe}')
                    removed.append(p)
                except OSError as e:
                    logger.warning('purge: trash %s échoué : %s', p, e)
    # 4) job configs : nommés d'après le run name (base/famille), donc un même
    #    trigger peut en avoir plusieurs (ex. un run zimage + un run krea). On
    #    balaie tout config dont le stem est sur la frontière de ce trigger,
    #    comme les étapes 2-3 pour les dossiers.
    try:
        jobs_dir = str(_jobs_dir())
    except RuntimeError:
        jobs_dir = None
    if jobs_dir and os.path.isdir(jobs_dir):
        for fn in os.listdir(jobs_dir):
            if not fn.endswith('.json'):
                continue
            p = os.path.join(jobs_dir, fn)
            if _trigger_boundary(fn[:-len('.json')], run_prefix) and os.path.isfile(p):
                try:
                    trash.send_to_trash(p, context=f'training-{trigger_safe}')
                    removed.append(p)
                except OSError as e:
                    logger.warning('purge: trash %s échoué : %s', p, e)
    logger.info('purge_training_artifacts u%s/%s : %d artefact(s) retiré(s)',
                user_id, trigger_safe, len(removed))
    return removed


def _rename_plan(root, old_prefix, new_prefix, *, want_dir=False, suffix=None) -> list:
    """Entries of `root` on the EXACT boundary of old_prefix, paired with their
    renamed path. `suffix` restricts to one extension and is stripped before the
    boundary test (a job config's boundary lives in its stem, not in '.json')."""
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        if suffix and not name.endswith(suffix):
            continue
        src = os.path.join(root, name)
        if os.path.isdir(src) != want_dir:
            continue
        stem = name[:-len(suffix)] if suffix else name
        if not _trigger_boundary(stem, old_prefix):
            continue
        out.append((src, os.path.join(root, new_prefix + name[len(old_prefix):])))
    return out


def _rename_inside_run(run_dir, old_trigger_safe, new_trigger_safe) -> list:
    """Rename the `lora_<trigger>` subfolder and every `lora_<trigger>*` weights file
    INSIDE a run folder that was just renamed.

    ai-toolkit stamps the trigger at three levels, not one:
        ulocal_<trigger>_<base>/ lora_<trigger>/ lora_<trigger>_000000250.safetensors
    Moving only the outer folder therefore fixed nothing the user could see: importing
    a checkpoint deploys it under the SOURCE FILE's stem (`import_checkpoint`), so the
    LoRA still landed in ComfyUI under the old trigger. Reported after renaming a
    style dataset — the label changed, the deployed name did not.

    Best-effort per entry: a locked file is skipped, not fatal. Depth 1 is deliberate
    — that is the exact shape ai-toolkit writes, and walking deeper would risk
    renaming unrelated payloads (samples, optimizer state) that merely share a prefix."""
    moved = []
    try:
        entries = sorted(os.listdir(run_dir))
    except OSError:
        return moved
    for name in entries:
        if not _trigger_boundary(os.path.splitext(name)[0], f'lora_{old_trigger_safe}'):
            continue
        src = os.path.join(run_dir, name)
        dest = os.path.join(run_dir, f'lora_{new_trigger_safe}'
                            + name[len(f'lora_{old_trigger_safe}'):])
        if os.path.exists(dest):
            logger.warning('rename: %s already exists — left in place', dest)
            continue
        try:
            os.rename(src, dest)
            moved.append((src, dest))
            if os.path.isdir(dest):        # the lora_<trigger>/ folder: its files too
                moved += _rename_inside_run(dest, old_trigger_safe, new_trigger_safe)
        except OSError as e:
            logger.warning('rename: %s -> %s échoué : %s', src, dest, e)
    return moved


def rename_training_artifacts(user_id, old_trigger_safe, new_trigger_safe) -> dict:
    """Rename every training artefact of a (user, trigger) onto a NEW trigger —
    the mirror of purge_training_artifacts, for an edit instead of a delete.

    Without this, changing a dataset's trigger word orphaned everything it had
    already produced: the deployed LoRA, the ai-toolkit run folder, the export and
    the job config all keep the OLD trigger in their name, so they no longer match
    the dataset that made them (and a later run under the new trigger starts from
    an empty folder while the old one lingers as dead weight).

    Covers the same four backends as the purge: deployed LoRAs (all five family
    dirs), run output/, export datasets/, and config/generated/. Same safety
    rules too — exact trigger-boundary matching (never a sibling: Lola vs Lola2),
    bare os.listdir names (no path traversal), empty trigger is a no-op.

    PLANNED IN FULL, THEN EXECUTED: a half-renamed set is worse than none at all
    (artefacts split across two triggers with no record of the split), so any
    destination that already exists aborts the whole rename and nothing is moved.
    Returns {'renamed': [(src, dest)], 'conflicts': [dest], 'ok': bool}; ok is
    False only when a conflict blocked it. Idempotent: a second call finds
    nothing left under the old trigger and is a successful no-op.

    Each backend is probed independently — an unconfigured one (no ComfyUI dir
    yet) simply yields no roots to sweep instead of aborting the rename."""
    old_trigger_safe = (old_trigger_safe or '').strip()
    new_trigger_safe = (new_trigger_safe or '').strip()
    if (not old_trigger_safe or not new_trigger_safe
            or old_trigger_safe == new_trigger_safe or user_id in (None, '')):
        return {'renamed': [], 'conflicts': [], 'ok': True}

    old_run, new_run = f'u{user_id}_{old_trigger_safe}', f'u{user_id}_{new_trigger_safe}'
    old_lora, new_lora = f'lora_{old_trigger_safe}', f'lora_{new_trigger_safe}'

    def _roots(accessors):
        roots = []
        for accessor in accessors:
            try:
                roots.append(str(accessor()))
            except RuntimeError:
                pass                      # backend not configured yet -> nothing to sweep
        return roots

    plan = []
    # 1) deployed LoRAs in ComfyUI (zimage + sdxl + krea + flux + flux2klein + anima),
    # in every loras root — see purge_training_artifacts.
    for fam in _FAMILY_SUBDIR:
        for root in _lora_family_dirs(fam):
            plan += _rename_plan(root, old_lora, new_lora, suffix='.safetensors')
    # 2) run output + 3) export datasets (whole folders)
    for root in _roots((_output_dir, _datasets_dir)):
        plan += _rename_plan(root, old_run, new_run, want_dir=True)
    # 4) job configs, keyed by run name (one trigger can have several families)
    for root in _roots((_jobs_dir,)):
        plan += _rename_plan(root, old_run, new_run, suffix='.json')

    conflicts = [dest for _, dest in plan if os.path.exists(dest)]
    if conflicts:
        logger.warning('rename_training_artifacts u%s %s->%s : abandon, %d collision(s)',
                       user_id, old_trigger_safe, new_trigger_safe, len(conflicts))
        return {'renamed': [], 'conflicts': conflicts, 'ok': False}

    renamed = []
    for src, dest in plan:
        try:
            os.rename(src, dest)
            renamed.append((src, dest))
            if os.path.isdir(dest):
                renamed += _rename_inside_run(dest, old_trigger_safe, new_trigger_safe)
        except OSError as e:
            # Best-effort like the purge: a locked file (an open LoRA, a folder held
            # by a viewer) is logged and skipped rather than aborting mid-way, which
            # would leave the set split with no way to tell which half moved.
            logger.warning('rename: %s -> %s échoué : %s', src, dest, e)
    logger.info('rename_training_artifacts u%s %s->%s : %d artefact(s) renommé(s)',
                user_id, old_trigger_safe, new_trigger_safe, len(renamed))
    return {'renamed': renamed, 'conflicts': [], 'ok': True}


def _configure_exact_state_dataloaders(job_cfg):
    """Make the job's loader cursor synchronously replayable by the bridge."""
    # A worker process may prefetch sampler indices and consume its own RNG ahead
    # of the optimizer boundary. That state cannot be reconstructed after a
    # crash, so exact bundles deliberately use the main process.
    for process in job_cfg.get('config', {}).get('process', ()):
        for dataset in process.get('datasets', ()):
            dataset['num_workers'] = 0
    return job_cfg


def write_job_config(
        ds, dataset_folder: str, steps: int = 3000,
        *, exact_state_bridge: bool = False, job_config=None) -> str:
    job_cfg = (
        job_config
        if isinstance(job_config, dict)
        else build_job_config(ds, dataset_folder, steps=steps))
    if exact_state_bridge:
        _configure_exact_state_dataloaders(job_cfg)
    # Name by the base/family-aware run name, NOT the trigger alone: a zimage run
    # and a krea run of the same trigger have distinct run names everywhere else
    # (training_folder, dataset_folder), so keying this file by trigger only made
    # the second launch silently clobber the first's config record.
    path = _jobs_dir() / f'{_run_name(ds)}.json'
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(job_cfg, fh, indent=2)
    return str(path)


def _style_steps_policy(ds, train_type=None, variant=None) -> dict:
    """Resolve the researched Style step envelope for one family/variant."""
    fam = _train_type(ds, train_type)
    selected = str(variant or getattr(ds, 'train_variant', None)
                   or _default_variant_for(fam)).strip().lower()
    if fam == 'flux2klein':
        low, high, recipe = 1200, 3000, 'klein'
    elif fam == 'krea':
        if selected in ('base', 'raw'):
            low, high, recipe = 2000, 3000, 'krea_raw'
            selected = 'raw'
        else:
            # _krea_is_raw maps every non-Raw legacy value to the Turbo recipe.
            low, high, recipe = 1000, 2000, 'krea_turbo'
            selected = 'turbo'
    elif fam == 'zimage' and selected == 'turbo':
        low, high, recipe = 1000, 2000, 'zimage_turbo'
    else:
        low, high, recipe = 1500, 3000, 'general'
    return {'train_type': fam, 'variant': selected, 'min_steps': low,
            'max_steps': high, 'recipe': recipe}


def _preset_steps_policy(ds) -> dict | None:
    """Validated built-in step policy stored with the applied preset."""
    s = _train_settings(ds)
    fixed = s.get('preset_steps_fixed')
    if type(fixed) is int and fixed > 0:
        return {'preset_steps_fixed': fixed, 'fixed_steps': fixed,
                'recipe': 'preset_fixed'}
    per_image = s.get('preset_steps_per_image')
    if type(per_image) is not int or per_image <= 0:
        return None
    low = s.get('preset_steps_min')
    high = s.get('preset_steps_max')
    low = low if type(low) is int and low > 0 else 1
    high = high if type(high) is int and high > 0 else 2_000_000_000
    if low > high:
        low, high = high, low
    return {
        'preset_steps_per_image': per_image,
        'preset_steps_min': low,
        'preset_steps_max': high,
        'per_image': per_image,
        'min_steps': low,
        'max_steps': high,
        'recipe': 'preset_per_image',
    }


def recommended_steps(dataset_id, train_type=None, variant=None) -> int:
    """Steps cibles selon le *type* de dataset — la recette suit le dataset, pas l'inverse.

    Character (défaut) : ~120 steps/image, bornés [1500, 3500]. On verrouille une
    identité sur un petit set curé (~100-150 vues/image, consensus des guides
    ai-toolkit/Z-Image) ; un 3000 fixe surentraînait les petits datasets et
    sous-entraînait les gros. À 25 images (preset équilibré) ça redonne 3000.

    Concept : échelle SOUS-LINÉAIRE (√n), bornée [2000, 12000]. Un concept
    doit généraliser, pas mémoriser : plus le set grossit, moins chaque image doit
    être vue. Appliquer le taux « character » (120/img) à 400 images donnerait
    48 000 steps (overfit garanti) ; le clamp à 3500 donnait l'inverse (sous-
    entraîné). 475·√n colle aux deux points d'ancrage du consensus : ~30-40 images
    de concept → ~3000 steps, ~400 images → ~9500 steps (~24 vues/image).

    Style : cible 50 steps/image, arrondie AU-DESSUS à la centaine, puis bornée
    par la recette effective : Klein [1200,3000], Krea Raw [2000,3000], Krea ou
    Z-Image Turbo [1000,2000], autres [1500,3000]. ``train_type``/``variant``
    sont optionnels et rétrocompatibles ; fournis par les routes ils décrivent le
    lancement en cours plutôt qu'un ancien choix persisté.
    """
    ds = _train_context_view(
        FaceDataset.query.get(dataset_id), train_type, variant)
    n = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep').count()
    if ds is not None and slider_mode_enabled(ds):
        # Slider (mode, Beta) : la direction est définie par les prompts, pas par
        # les images (substrat) → cible FIXE, indépendante de n. Ancrage : la
        # recette slider d'Ostris (500 steps, « rarement au-dessus de 1000 ») ;
        # le trainer moderne entraîne les deux polarités à chaque step.
        return SLIDER_DEFAULT_STEPS
    policy = _preset_steps_policy(ds) if ds is not None else None
    if policy:
        if 'fixed_steps' in policy:
            return policy['fixed_steps']
        target = n * policy['per_image']
        return max(policy['min_steps'], min(policy['max_steps'], target))
    if ds is not None and fds.is_style(ds):
        policy = _style_steps_policy(ds, train_type, variant)
        target = int(math.ceil((50 * max(n, 1)) / 100.0) * 100)
        return max(policy['min_steps'], min(policy['max_steps'], target))
    if ds is not None and fds.is_concept(ds):
        target = int(round(475 * math.sqrt(max(n, 1)), -2))
        return max(2000, min(12000, target))
    target = int(round(n * 120, -2))  # ~120 steps/image, arrondi à la centaine
    return max(1500, min(3500, target))


def default_steps(ds, train_type=None, variant=None) -> int:
    """Adaptive step count for a dataset — single source of truth shared by
    local launch_training and cloud training (parity guarantee). Thin ds-based
    wrapper over recommended_steps(dataset_id) (the calc used by launch_training
    when steps=None) so callers holding the ds object don't need the id."""
    return recommended_steps(ds.id, train_type=train_type, variant=variant)


def recommended_steps_info(dataset_id, train_type=None, variant=None) -> dict:
    """Version « transparente » de recommended_steps pour l'UI : le nombre + le
    pourquoi, afin que l'app apprenne au débutant au lieu de décider en boîte
    noire. Ne mute rien."""
    ds = _train_context_view(
        FaceDataset.query.get(dataset_id), train_type, variant)
    n = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep').count()
    kind = (ds.kind or 'character') if ds is not None else 'character'
    steps = recommended_steps(dataset_id, train_type=train_type, variant=variant)
    if ds is not None and slider_mode_enabled(ds):
        rationale = (f"Slider (Beta) — fixed {steps} steps. The slider direction is "
                     "prompt-defined, so dataset size doesn't drive convergence (the "
                     f"{n} kept images are only a denoising substrate). Ostris' slider "
                     "recipe uses 500-1000 steps; watch the ± preview sheet and stop "
                     "early if the sweep collapses.")
        return {'steps': steps, 'kind': kind, 'n_images': n, 'rationale': rationale,
                'slider': True}
    policy = _preset_steps_policy(ds) if ds is not None else None
    if policy:
        if 'fixed_steps' in policy:
            rationale = (
                f"{kind.title()} preset — fixed {steps} steps, independent of the "
                f"{n} kept images; inspect the announced checkpoints.")
        else:
            views = round(steps / n, 1) if n else 0
            rationale = (
                f"{kind.title()} preset — {policy['per_image']} steps/image, clamped "
                f"{policy['min_steps']}–{policy['max_steps']} ({n} kept images, "
                f"~{views}/img here).")
        return {'steps': steps, 'kind': kind, 'n_images': n,
                'rationale': rationale, **policy}
    if kind == 'style' and ds is not None:
        policy = _style_steps_policy(ds, train_type, variant)
        views = round(steps / n, 1) if n else 0
        rationale = (f"Style — {n} images kept. 50 steps/image, rounded up to 100, "
                     f"then clamped {policy['min_steps']}–{policy['max_steps']} for "
                     f"{policy['train_type']} {policy['variant']} (~{views}/img here). "
                     "Use checkpoints to select the visual peak before overfitting.")
        return {'steps': steps, 'kind': kind, 'n_images': n, 'rationale': rationale,
                **policy}
    if kind == 'concept':
        views = round(steps / n, 1) if n else 0
        rationale = (f"Concept — {n} images kept. Sublinear scaling (475·√n, "
                     f"clamped 2000–12000): the bigger the set, the fewer views per "
                     f"image (~{views}/img here), so the LoRA generalizes the concept "
                     f"instead of memorizing shots. Variety matters more than count.")
    else:
        rationale = (f"Character — {n} images kept. ~120 steps/image (clamped "
                     f"1500–3500): a small curated set seen many times locks the "
                     f"identity without drifting.")
    return {'steps': steps, 'kind': kind, 'n_images': n, 'rationale': rationale}


def _normalize_style_caption(value) -> str:
    """Comparison form for Style-quality guards (not an exported caption rewrite)."""
    collapsed = re.sub(r'\s+', ' ', str(value or '')).strip()
    return collapsed.strip(' .!?:;,').strip().casefold()


def _style_caption_quality_from_rows(ds, rows) -> dict:
    """Analyze the two catastrophic Style-caption patterns using stored captions.

    Stored Style captions must describe image content only. The dataset trigger is
    an internal run id, while identical sidecars provide no per-image conditioning.
    This pure helper is shared by preflight and the authoritative launch guard.
    """
    captions = [(row.caption or '').strip() for row in rows
                if (row.caption or '').strip()]
    trigger = _normalize_style_caption(getattr(ds, 'trigger_word', None))
    normalized = [_normalize_style_caption(caption) for caption in captions]
    trigger_only_count = sum(1 for caption in normalized
                             if trigger and caption == trigger)
    all_identical = len(normalized) > 1 and len(set(normalized)) == 1
    issues = []
    if trigger_only_count:
        issues.append(
            f'{trigger_only_count} Style caption(s) contain only the internal run id; '
            'captions must describe the visible content instead.')
    if all_identical:
        issues.append(
            'all Style captions are identical; each image needs its own content description.')
    return {
        'caption_count': len(captions),
        'distinct_caption_count': len(set(normalized)),
        'trigger_only_count': trigger_only_count,
        'all_identical': all_identical,
        'issues': issues,
    }


def style_caption_quality(dataset_id) -> dict:
    """Public read-only Style quality report used by routes/tests and preflight."""
    ds = FaceDataset.query.get(dataset_id)
    if ds is None or not fds.is_style(ds):
        return {'caption_count': 0, 'distinct_caption_count': 0,
                'trigger_only_count': 0, 'all_identical': False, 'issues': []}
    rows = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep').all()
    return _style_caption_quality_from_rows(ds, rows)


# --- Preflight d'entraînement (garde-fous, lecture seule) -----------------------
# Plancher DUR / recommandé par famille. Sous le plancher → blocker ; entre les
# deux → warning à confirmer. 10 images fixes pour tout le monde sous-estimait
# SDXL (booru, plus gourmand en variété) et laissait passer des runs voués au
# surapprentissage.
TRAIN_MIN_IMAGES = {'zimage': (12, 20), 'sdxl': (20, 30), 'krea': (15, 20), 'flux': (15, 20),
                    'flux2klein': (15, 20)}
# (_FAMILY_LABEL used to be re-declared here, shadowing the definition near the
#  top of the module. Merged there — see the comment on it.)
# VRAM mesurée : Krea 2 (12B) sature un 24 GB à 1024 (cf. KREA_TRAIN_RESOLUTION). Flux
# est un DiT de même classe (12B) → même seuil recommandé.
_KREA_MIN_VRAM_GB = 24
# flux2klein est VOLONTAIREMENT absent : le check est variant-aveugle (la variante
# se choisit au lancement, après ce preflight) et le défaut 4B tient en 16-24 Go —
# un warning « il faut ~24 GB » serait un faux positif sur la voie locale normale.
# Le 9B (32-48 Go) est la voie cloud ; un seuil 24 le sous-estimerait de toute façon.
_VRAM24_FAMILIES = ('krea', 'flux')   # familles 12B qui recommandent ~24 GB à 1024


def training_preflight(user_id, dataset_id, train_type=None, variant=None,
                       lane=None, masked=None, training_mode=None,
                       base_model=_PERSISTED) -> dict:
    """Pre-launch sanity report: {'blockers': [...], 'warnings': [...]}. Blockers
    stop the launch (too few images for the family); warnings ask for one explicit
    confirm in the UI. Pure reads — never mutates, never raises on probe failures
    (an unknown GPU must not block a run).

    Émet AUSSI `checks` (liste structurée {id,label,status,detail,target}) +
    `verdict` ('ready'|'warnings'|'blocked') pour la pastille de préparation du
    workspace — construits DANS LA MÊME PASSE que blockers/warnings (une seule
    source de vérité, aucune règle dupliquée). `target` = id de section du
    workspace (gf-generate/gf-images) où corriger — None quand rien à cibler.
    NB : le check 'captioned' (images gardées sans caption) est un fail dans
    `checks` (assert_trainable refusera le launch) mais volontairement PAS un
    blocker ici — le flux modal existant (launch → erreur explicite) est conservé.

    ``lane`` ('local' default, or 'cloud') says WHERE the run will execute. Every
    row carries a ``scope``: 'dataset' (a property of the images/captions, true
    wherever the job runs) or 'machine' (a read of THIS box — its GPU, its
    ai-toolkit venv). A cloud lane drops the 'machine' rows and their warning
    lines: that hardware will not run the job, and on a machine with no local
    training environment at all they would fire on every single cloud launch —
    which is exactly how users learn to click through warnings without reading
    them. Default 'local' keeps the historical payload byte-for-byte.

    ``masked`` says whether the caller intends MASKED training (person masks). It
    is a client-side preference the server cannot read, so it is passed in; None
    (the default) means "not stated" and the person-mask row is omitted entirely —
    warning about a mask nobody asked for is exactly the noise that teaches people
    to click through preflights."""
    from .face_variations import caption_has_identity_leak
    stored_ds = fds.get_dataset(user_id, dataset_id)
    if not stored_ds:
        raise ValueError('dataset not found')
    ttype = _train_type(stored_ds, train_type)
    mode = normalize_training_mode(
        training_mode if training_mode is not None
        else getattr(stored_ds, 'training_mode', None))
    ds = _train_context_view(
        stored_ds, ttype, variant, base_model=base_model,
        training_mode=mode)
    label = _FAMILY_LABEL.get(ttype, ttype)
    blockers, warnings = [], []
    checks = []
    hf_cloud_token_status = None
    # `warnings` is a flat list of strings (the modal renders it verbatim), so the
    # lane filter cannot recognise a machine-scope line by reading it. Record the
    # indices as they are appended instead — the only reliable pairing.
    machine_warning_ix = set()

    def _machine_warn(msg):
        machine_warning_ix.add(len(warnings))
        warnings.append(msg)

    def _check(cid, clabel, status, detail, target=None, bypassable=None, hint=None,
               scope='dataset'):
        # `bypassable` (fail rows only): True = a QUALITY guard-rail the explicit
        # "Continue anyway" ack can waive; False = a physical impossibility the ack
        # never covers. `hint` = the honest one-line risk shown next to the ack.
        # `scope`: 'dataset' = a property of the images/captions (true on any lane);
        # 'machine' = a read of THIS box, meaningless when the job runs elsewhere;
        # 'cloud' = a remote-lane prerequisite such as the dedicated HF token.
        entry = {'id': cid, 'label': clabel, 'status': status,
                 'detail': detail, 'target': target, 'scope': scope}
        if bypassable is not None:
            entry['bypassable'] = bool(bypassable)
        if hint:
            entry['hint'] = hint
        checks.append(entry)

    rows = FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all()
    kept = [r for r in rows if r.status == 'keep' and r.filename]
    n = len(kept)
    if (_optimizer_eff(ds) == 'automagic3'
            and (lane or 'local') != 'cloud'
            and not _aitoolkit_supports_automagic3()):
        message = ("Automagic3 is selected, but this ai-toolkit checkout does not "
                   "provide it. Update ai-toolkit (git pull) or choose another optimizer.")
        _machine_warn(message)
        _check('automagic3', 'Automagic3 available', 'fail', message,
               'gf-training', bypassable=False, scope='machine')
    # CONCEPT / STYLE : plusieurs dimensions ci-dessous (équilibre de composition,
    # fuite d'identité) sont des heuristiques de LoRA PERSONNAGE sans objet quand
    # l'invariant du set n'est pas une identité — on les saute pour ne pas générer
    # de faux avertissements.
    concept = fds.is_conceptual(ds)
    style = fds.is_style(ds)
    # SLIDER mode (Beta) : les images ne sont qu'un SUBSTRAT de débruitage et les
    # captions sont ignorées par la loss slider → plancher d'images réduit, gardes
    # caption/composition/identité sans objet ; la vraie exigence est la paire de
    # prompts qui définit la direction du slider.
    slider = slider_mode_enabled(ds)

    # Dense Krea is intentionally a separate, cloud-only lane. Surface every
    # physical incompatibility in the normal structured preflight instead of
    # letting a paid pod discover it after provisioning.
    if mode == 'full_transformer':
        dense_issues = []
        if (lane or 'local') != 'cloud':
            dense_issues.append('full_transformer training is cloud-only')
        if ttype != 'krea':
            dense_issues.append('it is supported only for Krea 2')
        if slider:
            dense_issues.append('Slider LoRA mode must be disabled')
        _dense_base = str(getattr(ds, 'train_base_model', None) or '').strip()
        if _dense_base:
            if not _is_custom_weights(_dense_base):
                dense_issues.append('a custom base must be the full path to a '
                                    'local .safetensors checkpoint')
            else:
                try:
                    assert_trainable_base_file(_dense_base)
                except ValueError as exc:
                    dense_issues.append(str(exc))
        if dense_issues:
            message = '; '.join(dense_issues)
            blockers.append(message)
            _check('training_mode', 'Dense training compatibility', 'fail',
                   message, 'gf-training', bypassable=False)
        else:
            # Names the base this run will REALLY train — by file name for a
            # custom checkpoint, never its full path: preflight details are
            # pasted into Discord and GitHub issues.
            _dense_target = (os.path.basename(_dense_base)
                             if _is_custom_weights(_dense_base)
                             else _krea_name_or_path(ds))
            _check('training_mode', 'Dense training compatibility', 'ok',
                   f'{_dense_target} full transformer training will run '
                   'in the cloud')
        # Turbo is allowed and UNMEASURED. Said here, in the preflight the user
        # reads before renting, not after — and phrased as what is unknown: no
        # promise of a result, no prediction of failure.
        if ttype == 'krea' and not _krea_is_raw(ds) and not _is_custom_weights(_dense_base):
            _check('dense_turbo', 'Full-model training on Krea 2 Turbo', 'warn',
                   'Krea officially recommends training a LoRA on Raw and applying '
                   'it to Turbo. Dense training on a distilled base has not been '
                   'measured here; on other distilled models it has been seen to '
                   'erode few-step behaviour, so the result may need real guidance '
                   'and more steps.', scope='cloud')

        # Reuse the launch's definitive credential validator.  This may contact
        # Hugging Face, but it never reserves a pod/GPU; a paid run must not be
        # the first place an absent token, wrong token type/scope, or unaccepted
        # Krea licence is discovered.
        try:
            from . import cloud_training as cloud
            hf_cloud_token_status = cloud.full_transformer_token_preflight(
                required_base_repo=official_base_repo(ds, ttype))
            if not isinstance(hf_cloud_token_status, dict):
                raise RuntimeError('invalid token preflight response')
        except Exception:
            hf_cloud_token_status = {
                'ok': False,
                'configured': bool(cfg.secret('HF_CLOUD_TOKEN')),
                'error': ('HF_CLOUD_TOKEN could not be validated. Configure a '
                          'Hugging Face token with Krea read and repository '
                          'write access; fine-grained is recommended and global '
                          'write is accepted with a warning.'),
            }
        if hf_cloud_token_status.get('ok'):
            namespace = hf_cloud_token_status.get('namespace')
            warning = hf_cloud_token_status.get('warning')
            detail = warning or ('Dedicated HF_CLOUD_TOKEN validated'
                                 + (f' for {namespace}' if namespace else ''))
            _check('hf_cloud_token', 'Hugging Face cloud token',
                   'warn' if warning else 'ok',
                   detail, scope='cloud')
        else:
            detail = (hf_cloud_token_status.get('error')
                      or 'HF_CLOUD_TOKEN is missing or invalid')
            blockers.append(detail)
            _check(
                'hf_cloud_token', 'Hugging Face cloud token', 'fail',
                detail, 'gf-training', bypassable=False,
                hint=('Add HF_CLOUD_TOKEN in Settings with read access to '
                      'krea/Krea-2-Raw and repository write access. A '
                      'fine-grained token is recommended; global write is '
                      'accepted with a warning.'),
                scope='cloud')

    # 1) minimum d'images par famille (slider : plancher substrat réduit)
    floor, reco = (TRAIN_MIN_IMAGES_SLIDER if slider
                   else TRAIN_MIN_IMAGES.get(ttype, (12, 20)))
    if n < floor:
        blockers.append(
            f'{n} kept image(s) — slider training still needs {floor}+ images as a '
            'denoising substrate. Import a few varied images.' if slider else
            f'{n} kept image(s) — the hard minimum for a {label} LoRA is {floor}. '
            'Generate or import more before training.')
        _check('images', 'Enough images', 'fail',
               (f'{n} kept — slider substrate needs {floor}+ varied images' if slider
                else f'{n} kept — the hard minimum for {label} is {floor}'), 'gf-generate',
               # With no image kept there is literally nothing to train on (ai-toolkit
               # would crash) → a physical impossibility the ack can't cover. One or
               # more kept but below the floor is a quality guard-rail: waivable.
               bypassable=(n >= 1),
               hint=(f'{n} substrate image(s) is under {floor} — a slider trained this '
                     'thin will be unstable; you can proceed, but expect a weak slider.'
                     if slider else
                     f'{n} image(s) is well under {floor} — a {label} LoRA trained this '
                     'thin will likely overfit and generalize poorly. The minimum exists '
                     f'because {label} needs visual variety; proceed only to experiment.'))
    elif n < reco:
        warnings.append(
            f'{n} kept image(s) — {reco}+ varied substrate images recommended for a '
            'slider run.' if slider else
            f'{n} kept image(s) — {reco} recommended for a solid {label} LoRA.')
        _check('images', 'Enough images', 'warn',
               (f'{n} kept — {reco}+ varied substrate images recommended' if slider
                else f'{n} kept — {reco}+ recommended for a solid {label} LoRA'),
               'gf-generate')
    else:
        _check('images', 'Enough images', 'ok',
               f'{n} kept ({reco}+ recommended)' + (' — substrate only in slider mode'
                                                    if slider else ''))

    # 1bis) SLIDER : la paire de prompts est LE prérequis (assert_trainable refuse
    # le launch sans elle) + rappel honnête que les captions ne s'entraînent pas.
    if slider:
        sc = _slider_settings(ds)
        pos = (sc.get('positive') or '').strip()
        neg = (sc.get('negative') or '').strip()
        if not pos or not neg:
            missing = ' and '.join([w for w, v in (('positive', pos), ('negative', neg))
                                    if not v])
            blockers.append(f'Slider mode is ON but the {missing} prompt is empty — '
                            'the pair defines the two ends of the slider.')
            _check('slider_prompts', 'Slider prompt pair', 'fail',
                   f'{missing} prompt missing — set it in the training panel', 'gf-training',
                   # No prompt pair = no slider direction to learn: ai-toolkit has
                   # nothing to optimise → physical impossibility, never waivable.
                   bypassable=False)
        else:
            _check('slider_prompts', 'Slider prompt pair', 'ok',
                   f'“{pos[:60]}” ↔ “{neg[:60]}”')
        _check('captioned', 'Captions', 'ok',
               'captions are ignored by the slider loss (images are substrate only)')

    # 2) équilibre de composition — heuristique PERSONNAGE (viser un mix face/bust/body/
    # back pour rendre un visage à toutes les distances). Sans objet pour un CONCEPT (il
    # s'apprend sur les cadrages tels quels), et un dataset non classé (framing=None) y
    # déclencherait un faux « tout en gros plan visage » → on saute pour les concepts.
    if n and not concept and not slider:
        comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
        for r in kept:
            if r.framing in comp:
                comp[r.framing] += 1
        _comp_ok = True
        if comp['bust'] + comp['body'] + comp['back'] == 0:
            warnings.append('every kept image is a face shot — the LoRA will struggle to '
                            'render busts and full-body scenes.')
            _check('composition', 'Framing balance', 'warn',
                   'all kept images are face shots — add bust/body shots', 'gf-generate')
            _comp_ok = False
        if fds.is_body_fidelity(ds) and comp['body'] == 0:
            warnings.append('body fidelity is ON but there is no full-body shot — the body '
                            "can't be learned without body images.")
            if _comp_ok:
                _check('composition', 'Framing balance', 'warn',
                       'body fidelity is ON but there is no full-body shot', 'gf-generate')
                _comp_ok = False
        if _comp_ok:
            _check('composition', 'Framing balance', 'ok',
                   f"face {comp['face']} · bust {comp['bust']} · body {comp['body']} · back {comp['back']}")

    # 3bis) toutes les gardées ont une caption — WARN, plus un mur : le launch
    # demande un confirm (« train anyway ») au lieu de refuser (UNCAPTIONED:
    # dans assert_trainable). Les captions restent fortement recommandées.
    uncaptioned = sum(1 for r in kept if not (r.caption or '').strip())
    if n and not slider:
        if uncaptioned:
            caption_policy = ('content captions are required for always-on Style'
                              if style else 'captions are strongly recommended')
            warnings.append(f'{uncaptioned}/{n} kept image(s) have no caption — '
                            f'{caption_policy}; launching will ask you to confirm.')
            _check('captioned', 'Every kept image captioned', 'warn',
                   f'{uncaptioned}/{n} kept image(s) have no caption — {caption_policy}; '
                   'launching asks to confirm', 'gf-images')
        else:
            _check('captioned', 'Every kept image captioned', 'ok', f'{n}/{n} captioned')

    # 3ter) DUAL CAPTIONS vs effective recipe. A family default or a selected preset
    # may pre-cache text embeddings and unload the encoder: the short caption then
    # has no encoder to read it, and emitting it used to crash at step 1 (issue #22).
    # Say so before launch instead of pretending the second caption will train.
    # Slider mode is excluded: its guided loss ignores captions entirely, and the
    # 'captioned' row above already says so — a second row promising two wordings would
    # contradict it.
    if fds.dual_captions_enabled(ds) and not slider:
        if _cache_text_embeddings_eff(ds, ttype):
            warnings.append(
                f'Dual captions are ON but {label} cannot train them — it pre-caches its '
                'text embeddings and unloads the text encoder, so only the long caption is '
                'encoded. The run will train on the long caption alone.')
            _check('dual_captions', 'Dual captions', 'warn',
                   f'{label} caches its text embeddings — the short caption is ignored and '
                   'the run trains on the long one alone', 'gf-training')
        else:
            _check('dual_captions', 'Dual captions', 'ok',
                   f'{label} trains each image on both its long and its short caption')

    # 3) captions suspectes (trop courtes / dupliquées) — sans objet en slider mode
    caps = [(r.caption or '').strip() for r in kept if (r.caption or '').strip()]
    if caps and not slider:
        _cap_ok = True
        if style:
            quality = _style_caption_quality_from_rows(ds, kept)
            if quality['issues']:
                warnings.extend(quality['issues'])
                _check('caption_quality', 'Caption quality', 'warn',
                       ' '.join(quality['issues']), 'gf-images')
                _cap_ok = False
        short = sum(1 for c in caps if len(c.split()) < 8)
        if short / len(caps) > 0.3:
            warnings.append(f'{short}/{len(caps)} caption(s) are very short (<8 words) — '
                            'weak captions weaken prompt control.')
            if _cap_ok:
                _check('caption_quality', 'Caption quality', 'warn',
                       f'{short}/{len(caps)} captions are very short (<8 words)', 'gf-images')
            _cap_ok = False
        if not style and len(set(c.lower() for c in caps)) < len(caps) * 0.7:
            warnings.append('many captions are identical — the model learns nothing from '
                            'repeated text; re-caption for variety.')
            if _cap_ok:
                _check('caption_quality', 'Caption quality', 'warn',
                       'many captions are identical — re-caption for variety', 'gf-images')
                _cap_ok = False
        if _cap_ok:
            _check('caption_quality', 'Caption quality', 'ok',
                   'varied, ≥8 words')

    # 4) fuite d'identité — on RETIENT les images fautives (pas juste le compte) pour
    # que l'UI liste lesquelles au moment du preflight, éditables sur place.
    # CONCEPT : décrire l'identité (visage/cheveux/corps) est VOULU — c'est le concept,
    # pas le visage, qui se lie au trigger → la « fuite d'identité » n'a aucun sens ici.
    # On saute entièrement cette dimension (comme le badge caption_leak du payload), sinon
    # CHAQUE caption concept déclenche un faux avertissement au preflight.
    body = fds.is_body_fidelity(ds)
    appearance = (fds.caption_options(ds).get('appearance') or None)
    leak_images = [] if (concept or slider) else [
        {'id': r.id, 'filename': r.filename, 'caption': (r.caption or '').strip()}
        for r in kept
        if (r.caption or '').strip()
        and caption_has_identity_leak((r.caption or '').strip(), body=body,
                                     appearance=appearance)]
    if leak_images:
        warnings.append(f'{len(leak_images)} caption(s) still describe the identity (face/hair'
                        f'{"/body marks" if body else ""}) — it will bind to those words '
                        'instead of the trigger. Re-caption or edit them.')
        _check('leaks', 'No identity leaks', 'warn',
               f'{len(leak_images)} caption(s) describe hair/face/skin — identity will bind '
               'to those words, not the trigger', 'gf-images')
    elif caps and not concept and not slider:
        _check('leaks', 'No identity leaks', 'ok', '0 leaking caption')

    # 5) quasi-doublons parmi les kept (dHash pairwise, n<=~60 -> négligeable). On
    # retient les PAIRES (leurs deux images) pour que l'UI montre lesquelles rejeter.
    # Slider : sans objet (le substrat n'est pas mémorisé) → on saute le scan.
    dup_pairs = []
    if not slider:
        try:
            hp = []  # [(row, dhash)] pour les kept lisibles sur disque
            for r in kept:
                p = fds._img_path(r)
                if p and os.path.exists(p):
                    with Image.open(p) as im:
                        hp.append((r, fds._dhash(im)))
            for i in range(len(hp)):
                for j in range(i + 1, len(hp)):
                    if fds._hamming(hp[i][1], hp[j][1]) <= fds.SCRAPE_DHASH_MAX_DISTANCE:
                        ra, rb = hp[i][0], hp[j][0]
                        dup_pairs.append({'a': {'id': ra.id, 'filename': ra.filename},
                                          'b': {'id': rb.id, 'filename': rb.filename}})
            if dup_pairs:
                warnings.append(f'{len(dup_pairs)} pair(s) of kept images are near-duplicates — '
                                'the model overfits repeated content; reject one of each pair.')
                _check('duplicates', 'No near-duplicates', 'warn',
                       f'{len(dup_pairs)} near-duplicate pair(s) — reject one of each', 'gf-images')
            elif n:
                _check('duplicates', 'No near-duplicates', 'ok', '0 pair')
        except Exception:
            pass   # best-effort: an unreadable file must not block the preflight

    # 11) images encore en attente de tri (elles ne s'entraînent PAS)
    untriaged = sum(1 for r in rows if r.status == 'pending' and r.filename)
    if untriaged:
        warnings.append(f'{untriaged} image(s) still await triage (✓/✕) — they will NOT '
                        'be part of the training.')
        _check('triage', 'Everything triaged', 'warn',
               f'{untriaged} image(s) still await ✓/✕ — they will NOT train', 'gf-images')
    elif rows:
        _check('triage', 'Everything triaged', 'ok', 'no image awaiting ✓/✕')

    # 6bis) MEMORY SAVERS switched off under a family whose recipe needs them.
    #
    # This is the row that did not exist while quantize/quantize_te/low_vram were
    # stored globally and applied to whatever family came next: a `False` set on
    # Anima or SDXL (2B — where it IS the default) followed a family switch onto a
    # 12B DiT and produced a Krea/FLUX config with `quantize: false`, no
    # `low_vram` and no `qtype` — the calibrated recipe gone, and nothing said.
    # Deliberately provenance-blind (see memory_saving_risk): setting it here
    # directly is the same danger, so it earns the same sentence.
    #
    # scope='dataset': the flags travel WITH the job, so they matter on the cloud
    # lane too — that is where the mistake costs rented GPU-hours in real money,
    # which is the last place to drop the row.
    _mem_risk = memory_saving_risk(ds, ttype)
    if _mem_risk:
        _off = ', '.join(_MEMORY_LABELS[k] for k in _mem_risk['disabled'])
        _need = _mem_risk['unquantised_vram_gb']
        # The local card's verdict only speaks for the LOCAL lane. On the cloud
        # the job runs on a pod we have not rented yet, so a big local card
        # proves nothing about it and the requirement is stated as a requirement.
        if (lane or 'local') == 'cloud':
            warnings.append(
                f'{_off} switched off for a {label} run — {label} needs roughly {_need} GB '
                f'of VRAM without them. Rent a pod with at least that, or turn them back '
                'on in Advanced options ▸ Memory saving.')
            _check('memory_saving', 'Memory saving', 'warn',
                   f'{_off} off — the pod needs ~{_need} GB VRAM for {label}', 'gf-training')
        elif _mem_risk['verdict'] == 'can_disable':
            _check('memory_saving', 'Memory saving', 'ok',
                   f'{_off} off — {_mem_risk["gpu"] or "this GPU"} has '
                   f'{_mem_risk["vram_gb"]} GB, over the ~{_need} GB {label} needs without them')
        else:
            _seen = (f'this GPU reports {_mem_risk["vram_gb"]} GB'
                     if _mem_risk['vram_gb'] else 'this machine reports no usable GPU')
            warnings.append(
                f'{_off} switched off for a {label} run — {label} needs roughly {_need} GB '
                f'of VRAM without them and {_seen}. The run does not fail cleanly there: it '
                'slows to a crawl for hours while the driver pages to system RAM. Turn them '
                'back on in Advanced options ▸ Memory saving.')
            _check('memory_saving', 'Memory saving', 'warn',
                   f'{_off} off — {label} needs ~{_need} GB without them, {_seen}',
                   'gf-training')

    # 7) VRAM (Krea 2 mesuré à 24 GB ; None = inconnu, jamais bloquant)
    try:
        from .. import capabilities
        vram = capabilities.gpu_vram_gb()
        if vram is not None and ttype in _VRAM24_FAMILIES and vram < _KREA_MIN_VRAM_GB:
            # Only advise dropping to 768 when 768 is not ALREADY the choice —
            # telling someone to make a change they made is how a preflight
            # teaches people to click through it.
            _at_1024 = max(_effective_resolution(ds)) > 768
            _fix = ('Drop the resolution to 768 in Advanced options to fit.' if _at_1024
                    else 'You are already at 768, the low-VRAM resolution for this family; '
                         'keep the memory savers on too.')
            _machine_warn(f'{label} training needs ~{_KREA_MIN_VRAM_GB} GB of VRAM at 1024 '
                          f'— this GPU reports {vram} GB; expect OOM or extreme slowness. '
                          + _fix)
            _check('vram', 'GPU memory', 'warn',
                   f'{label} needs ~{_KREA_MIN_VRAM_GB} GB VRAM at 1024 — this GPU reports '
                   f'{vram} GB' + ('' if _at_1024 else ' (already at 768)'),
                   scope='machine')
    except Exception:
        pass

    # 8) torch build vs GPU architecture — the RTX 50 (Blackwell) trap. Stable
    # PyTorch wheels stop at sm_90; `torch.cuda.is_available()` stays True, the
    # run builds its buckets, then dies at the first real computation. Catching
    # it HERE is the whole point: the alternative is 20 minutes of setup for an
    # opaque "ai-toolkit exited 1". A warning, not a blocker — the verdict is a
    # read of the venv, and an unknown probe (None) says nothing at all.
    try:
        from .. import capabilities
        from .training_diagnostics import torch_arch_verdict
        arch = torch_arch_verdict(capabilities.aitoolkit_torch_info(),
                                  venv_python=cfg.aitoolkit_path('venv_python'))
        if arch and not arch['supported']:
            _machine_warn(arch['message']
                          + (f' Fix: {arch["command"]}' if arch['command'] else ''))
            # Keep the row SHORT — it sits in a one-line list next to ten other
            # checks, on a phone too. The full explanation + fix is the warning.
            _check('torch_arch', 'PyTorch supports this GPU', 'warn',
                   f'torch {arch["torch"]} has no {arch["sm"]} kernels — the run '
                   'dies at the first GPU computation', scope='machine')
    except Exception:
        pass   # a probe failure must never block or fake a diagnosis

    # 9) Face masking asked for, but the detector isn't installed.
    #
    # InsightFace is an OPTIONAL extra by decision (a few hundred MB nobody should
    # be made to download), so its absence is a NORMAL state, not an anomaly — this
    # is a warning, never a blocker. What is NOT acceptable is what happened before:
    # the export silently dropped the masks and the run trained unmasked, with the
    # user only finding out from a flag on the progress view, GPU-hours in. So the
    # decision is posed HERE, once, before the launch: install it, or continue
    # unmasked on purpose.
    #
    # The condition distinguishes "impossible for lack of a tool" from "refused BY
    # DESIGN": face_masking_enabled() already returns False for a Character or a
    # Style (their identity/rendering must be learned, masking would amputate it),
    # and slider mode forces it off because the guided slider loss never reads
    # batch.mask_tensor. Installing InsightFace would change nothing in those cases,
    # so they stay silent — warning there would be pure noise.
    if not slider and fds.face_masking_enabled(ds):
        try:
            from . import face_mask
            face_mask_ok = face_mask.is_available()
        except Exception:
            face_mask_ok = None      # probe blew up -> say nothing, never block
        if face_mask_ok is False:
            warnings.append(
                'Mask faces is ON, but face detection (InsightFace) is not installed — '
                'this run would train UNMASKED, with the faces at full loss weight. '
                'Install it from the Mask faces option in Advanced training options '
                '(~400 MB, a few minutes), or continue and train unmasked on purpose.')
            _check('face_mask', 'Face masking ready', 'warn',
                   'InsightFace is not installed — this run trains unmasked')
        elif face_mask_ok:
            _check('face_mask', 'Face masking ready', 'ok',
                   'InsightFace found — the faces will be weighted down')

    # 9bis) Person masking set to ON, but rembg isn't installed.
    # Retenu de la premiere ecriture de cette ligne (issue #24) : la sonde derriere
    # rembg est un import en sous-processus dont le TIMEOUT s'effondre en False
    # (capabilities._cached_import), et un `import rembg` a froid a ete MESURE a
    # ~20 s. Un refus dur transformerait donc une machine lente en machine qui ne
    # peut plus lancer du tout : c'est la seconde raison, independante, pour
    # laquelle cette ligne avertit et ne bloque jamais.
    #
    # THE reason `masked` became a stored dataset setting. While it lived in the
    # browser's localStorage the server only learned it at launch, so this badge
    # could not say what it says now: that the dataset is set to train masked and
    # will not, because the mask backend is missing. Users found out from a flag
    # on the progress view, GPU-hours in.
    #
    # Same shape as the face-mask row above and for the same reasons: rembg is an
    # optional ML extra, so its absence is a NORMAL state — a warning, never a
    # blocker (a run without masks is a valid run). person_masking_enabled()
    # already returns False for concept/style and slider mode, where masks are
    # refused BY DESIGN and installing rembg would change nothing, so those stay
    # silent instead of emitting pure noise.
    # resolve_masked, pas person_masking_enabled : un appelant qui EXPRIME une
    # intention explicite (le panneau qui rejoue le drapeau gele d'un run, une
    # relance cloud) doit etre cru — avertir « le dataset est en masque » a qui
    # vient de dire « lance sans masque » serait un contresens. Sans intention
    # (le badge de preparation, qui n'en a pas), on lit le reglage du dataset :
    # c'est exactement ce que ce chantier rend possible.
    # …ET les gardes de CONCEPTION, toujours. Une intention explicite decide de
    # l'OPT-IN de l'utilisateur, jamais des cas ou le masque est refuse par
    # construction : sur un concept/style le masque effacerait ce qu'on enseigne,
    # et en mode slider la perte guidee ne lit jamais le masque. Sans cette
    # seconde moitie, un `masked=True` explicite sur un concept enverrait
    # installer rembg pour un run qui ne s'en servira jamais.
    if resolve_masked(ds, masked) and not slider and not concept and not style:
        try:
            from . import person_mask
            person_mask_ok = person_mask.is_available()
        except Exception:
            person_mask_ok = None    # probe blew up -> say nothing, never block
        if person_mask_ok is False:
            warnings.append(
                'Masked training is ON for this dataset, but the person-mask backend '
                '(rembg) is not installed — this run would train UNMASKED, with the '
                'background at full loss weight. Install the ML extras from the Setup '
                'tab, or turn Masked off in the training panel to train unmasked on '
                'purpose.')
            _check('person_mask', 'Masked training ready', 'warn',
                   'rembg is not installed — this dataset is set to masked but the run '
                   'trains unmasked', 'gf-training')
        elif person_mask_ok:
            _check('person_mask', 'Masked training ready', 'ok',
                   'rembg found — the background will be weighted down to 10%')

    # Lane filter — BEFORE the verdict, so a cloud launch whose only complaint was
    # this machine's GPU comes back a clean 🟢 instead of a warning nobody can act
    # on. Note what stays: face_mask is machine-INSTALLED but dataset-SCOPED, since
    # the masks are generated locally at export and uploaded with the images
    # (cloud_training uploads `<staging>_masks`) — InsightFace missing here means
    # the PAID run trains unmasked. Local origin, cloud consequence: it must show.
    if (lane or 'local') == 'cloud':
        checks = [c for c in checks if c.get('scope') != 'machine']
        warnings = [w for i, w in enumerate(warnings) if i not in machine_warning_ix]

    # Verdict agrégé pour la pastille : un fail = 🔴, sinon un warn = 🟡, sinon 🟢.
    statuses = {c['status'] for c in checks}
    verdict = ('blocked' if 'fail' in statuses
               else 'warnings' if 'warn' in statuses else 'ready')

    # « Continue anyway » : proposé UNIQUEMENT quand il y a ≥1 blocker ET que TOUS
    # sont contournables (garde-fous qualité). Un seul blocker physique (0 image,
    # paire de prompts slider absente) → l'option disparaît et le launch reste refusé
    # même avec l'ack. override_hint = la ligne de risque honnête à afficher sous la case.
    fail_checks = [c for c in checks if c['status'] == 'fail']
    can_override = bool(fail_checks) and all(c.get('bypassable') for c in fail_checks)
    override_hint = ' '.join(c['hint'] for c in fail_checks
                             if c.get('hint')) if can_override else ''

    return {'blockers': blockers, 'warnings': warnings,
            # Détail « lesquelles » pour l'UI : images dont la caption fuit, et paires
            # quasi-doublons — le message reste agrégé, mais on peut drill-down + agir.
            'leak_images': leak_images, 'dup_pairs': dup_pairs,
            'checks': checks, 'verdict': verdict,
            # can_override : la case « Continue anyway » n'est offerte que quand c'est True
            # (miroir exact du garde serveur assert_trainable/allow_not_ready).
            'can_override': can_override, 'override_hint': override_hint,
            # Echoed so the modal can say WHERE this run is headed (and, implicitly,
            # why no GPU-memory row is listed) without re-deriving it client-side.
            'lane': ('cloud' if (lane or 'local') == 'cloud' else 'local'),
            'training_mode': mode,
            'hf_cloud_token_status': hf_cloud_token_status,
            'kept': n, 'floor': floor, 'recommended': reco}


# --- Garde-fou espace disque ---------------------------------------------------
# Un run plein (10 checkpoints ~0,3-2 Go + latents/samples) et une conversion
# diffusers (~12 Go) qui crashent à 90 % pour cause de disque plein laissent des
# artefacts corrompus. On refuse AVANT, avec un message actionnable.
MIN_FREE_GB_TRAIN = 10
MIN_FREE_GB_CONVERT = 15


def free_disk_gb(path) -> float | None:
    """Free space (GB) on the drive holding `path` (climbs to the nearest existing
    parent — the target dir may not exist yet). None if it can't be determined
    (never blocks on a stat failure)."""
    try:
        p = os.path.abspath(str(path))
        while p and not os.path.exists(p):
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return None


def assert_free_disk(path, min_gb, what) -> None:
    """Raise ValueError when the drive holding `path` has under `min_gb` GB free."""
    free = free_disk_gb(path)
    if free is not None and free < min_gb:
        raise ValueError(
            f'not enough disk space for {what}: {free:.1f} GB free on the target drive, '
            f'~{min_gb} GB needed - free up space and retry')


def _log_tail(path: str, n: int = 30) -> str:
    """Dernières `n` lignes d'un fichier log (pour remonter une erreur ai-toolkit)."""
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            return ''.join(fh.readlines()[-n:]).strip()
    except OSError:
        return '(log illisible)'


# Scanning window for the excerpt: wide enough that a full traceback (frames +
# exception line) is never cut in half, while `log_tail` keeps its historical
# 30-line shape for anything still reading that field.
_ERROR_SCAN_LINES = 200


def _crash_payload(log_path, dataset_id, rc) -> dict:
    """The `training_error` state a crashed local run leaves behind: the tail
    (path-redacted — this text is shown to the user and pasted into public help
    threads), the excerpt that actually EXPLAINS the failure, and, when the venv
    says so, the GPU-architecture verdict that turns "exited 1" into something
    the user can act on. Best-effort throughout: nothing here may raise inside
    the watcher thread, and an unknown probe adds no key at all."""
    from ..utils.redact import redact_tokens, redact_user_paths
    from .training_diagnostics import (extract_error_excerpt, gated_repo_verdict,
                                       hf_transfer_verdict, interpreter_verdict,
                                       missing_module_in_log, torch_arch_verdict)
    wide = _log_tail(log_path, _ERROR_SCAN_LINES)
    payload = {'dataset_id': dataset_id, 'rc': rc,
               'log_tail': redact_tokens(redact_user_paths(_log_tail(log_path)))[-1500:],
               'excerpt': extract_error_excerpt(wide)}
    # A gated-base refusal is a PROVEN cause with a precise remedy — and 401 and
    # 403 have opposite remedies, which the raw HF sentence conflates.
    try:
        gated = gated_repo_verdict(wide, token_configured=bool(cfg.secret('HF_TOKEN')))
        if gated:
            payload['hf_gated'] = {k: gated[k] for k in ('status', 'repo', 'url',
                                                         'title', 'message')}
    except Exception:
        pass
    # A dead fast-download accelerator looks exactly like a network fault, and the
    # app never sets that variable — so saying so is the whole remedy (GitHub #18,
    # bobba84).
    try:
        transfer = hf_transfer_verdict(wide)
        if transfer:
            payload['hf_transfer'] = transfer
    except Exception:
        pass
    # A `ModuleNotFoundError` in the log is a PROVEN interpreter problem, and the
    # one fact the log itself never carries is WHICH Python produced it. The
    # module is read off the log — no subprocess is spawned from the watcher
    # thread — and the "which venv works instead" hint costs a probe only here,
    # on a run that already failed (GitHub #19, strouder).
    try:
        module = missing_module_in_log(wide)
        if module:
            from .. import capabilities
            report = capabilities.aitoolkit_interpreter_report()
            verdict = interpreter_verdict(
                report['python'] or cfg.aitoolkit_path('venv_python'),
                False, alternative=report['alternative'], module=module)
            if verdict:
                payload['interpreter'] = {k: verdict[k] for k in (
                    'python', 'module', 'windows_store', 'alternative',
                    'title', 'message')}
    except Exception:
        pass
    try:
        from .. import capabilities
        arch = torch_arch_verdict(capabilities.aitoolkit_torch_info(),
                                  venv_python=cfg.aitoolkit_path('venv_python'))
        if arch and not arch['supported']:
            payload['gpu_arch'] = {'message': arch['message'], 'command': arch['command']}
    except Exception:
        pass
    return payload


def _exact_resume_bridge_failure(status_path, rc):
    """Return a concrete pre-training bridge failure, else ``None``."""
    from . import aitoolkit_state_bridge

    status = aitoolkit_state_bridge.read_status(status_path) if status_path else None
    if isinstance(status, dict) and status.get('training_started') is True:
        return None
    status_name = status.get('status') if isinstance(status, dict) else None
    reasons = status.get('reasons') if isinstance(status, dict) else None
    if isinstance(reasons, list):
        reasons = [str(reason).strip() for reason in reasons if str(reason).strip()]
    else:
        reasons = []
    if reasons:
        return '; '.join(reasons)
    if rc == 78 or status_name == 'bootstrap_error':
        return 'the exact-state bridge failed during interpreter bootstrap'
    if status_name == 'restored':
        return 'the exact state was restored, but training failed before its first optimizer boundary'
    if status_name == 'patched':
        return 'the exact-state bridge failed before it could restore the bundle'
    if rc in (0, None):
        return 'the exact-state process ended before its first optimizer boundary'
    return 'the exact-state bridge failed before training began'


def _clear_exact_resume_journal_after_boundary(transaction) -> bool:
    if not isinstance(transaction, dict):
        return False
    journal_path = transaction.get('journal_path')
    if not journal_path:
        return False
    from . import aitoolkit_state_bridge
    status = aitoolkit_state_bridge.read_status(
        transaction.get('status_path'))
    if not isinstance(status, dict) or status.get('training_started') is not True:
        return False
    try:
        _delete_exact_resume_journal(journal_path)
    except Exception:
        logger.exception(
            'could not clear exact-resume journal after optimizer boundary')
        return False
    transaction['journal_path'] = None
    return True


def _mark_exact_resume_launching(journal_path, run_token) -> None:
    """Persist pre-spawn intent or clear the fence while no child can exist."""
    try:
        _update_exact_resume_journal(
            journal_path,
            phase='launching',
            run_token=run_token,
        )
    except Exception:
        # This helper is called immediately before Popen. A failure here proves
        # no child was spawned, so retaining the just-published GPU fence would
        # deadlock every later launch after continue() restores the source lane.
        _clear_training_identity(ttl_seconds=1)
        raise


def _watch_training(
        app, proc, log_path, dataset_id, exact_resume_transaction=None) -> None:
    """Thread daemon : attend la fin du process ai-toolkit puis fait avancer la
    file (libère ComfyUI / lance le suivant) DÈS la fin, sans dépendre du polling
    client. Sur un crash (rc≠0), remonte la fin du log. process_training_queue()
    reste le filet de secours si Flask redémarre (le watcher meurt, le flag est
    rattrapé au prochain poll ou à la récupération de démarrage)."""
    try:
        if exact_resume_transaction and callable(getattr(proc, 'poll', None)):
            while True:
                _clear_exact_resume_journal_after_boundary(
                    exact_resume_transaction)
                rc = proc.poll()
                if rc is not None:
                    break
                time.sleep(0.25)
        else:
            proc.wait()
            rc = proc.returncode
        _clear_exact_resume_journal_after_boundary(
            exact_resume_transaction)
    except Exception:
        return
    try:
        with app.app_context():
            bridge_reason = None
            if exact_resume_transaction:
                bridge_reason = _exact_resume_bridge_failure(
                    exact_resume_transaction.get('status_path'), rc)
            if rc not in (0, None) or bridge_reason:
                payload = _crash_payload(log_path, dataset_id, rc)
                if bridge_reason:
                    try:
                        # A dead exact child leaves a short interval where its PID
                        # fence is conclusively dead but its source lane still
                        # needs rollback.  Exclude every new local launch while
                        # moving that lane back; otherwise a competitor could
                        # spawn from the fresh lane and have it renamed underneath
                        # the new process.
                        with _launch_transaction_lock, _queue_lock:
                            _rollback_unlaunched_exact_resume(
                                exact_resume_transaction['training_folder'],
                                exact_resume_transaction['archived'])
                            journal_path = exact_resume_transaction.get(
                                'journal_path')
                            if journal_path:
                                _delete_exact_resume_journal(journal_path)
                    except Exception:
                        logger.exception(
                            'full-state resume rollback failed for dataset %s',
                            dataset_id)
                        rollback_message = (
                            'The full-state bridge failed before training began, '
                            'and LDS could not automatically restore the archived '
                            'source run. Open the run folder before retrying.')
                    else:
                        rollback_message = (
                            'The full-state bridge failed before training began. '
                            'LDS restored the original run, so no source checkpoint '
                            'was lost and the continuation can be retried.')
                    payload['exact_resume'] = {
                        'rolled_back': rollback_message.startswith(
                            'The full-state bridge failed before training began. LDS'),
                        'reason': bridge_reason,
                        'message': rollback_message,
                    }
                    payload['excerpt'] = {
                        'kind': 'error',
                        'headline': 'Full-state resume did not start',
                        'text': f'{rollback_message}\n\nBridge: {bridge_reason}',
                    }
                logger.error("Entraînement ai-toolkit dataset %s terminé en ERREUR (rc=%s). "
                             "Cause probable :\n%s", dataset_id, rc,
                             payload['excerpt']['text'] or payload['log_tail'])
                # Surface l'erreur à l'UI (sinon un crash = juste « terminé » silencieux).
                queue_manager._set_system_state('training_error', payload, ttl_seconds=3600)
            else:
                logger.info("Entraînement ai-toolkit dataset %s terminé (rc=%s).", dataset_id, rc)
            process_training_queue()  # libère le GPU / enchaîne la file immédiatement
    except Exception as e:
        logger.warning("watcher training : post-traitement échoué : %s", e)


def archive_previous_run(ds) -> str | None:
    """Écarte le dossier du run existant (rename en `*_archived_<horodatage>`,
    jamais de suppression) pour que le prochain lancement reparte de ZÉRO au lieu
    de l'auto-resume ai-toolkit — le cas « j'ai remanié le dataset, je veux un
    LoRA neuf ». Les checkpoints archivés restent sur disque (récupérables à la
    main) et tombent avec le dataset : le nom garde le préfixe `lora_<trigger>`
    donc purge_training_artifacts les balaie aussi. Les copies déjà importées
    dans ComfyUI (loras/<famille>) ne sont pas touchées. None si aucun run."""
    run_dir = _run_root(ds)
    if not run_dir.is_dir():
        return None
    dest = f'{run_dir}_archived_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    try:
        os.rename(run_dir, dest)
    except OSError as e:
        # Dossier verrouillé (ex. antivirus, explorateur ouvert) → message actionnable.
        raise ValueError(f'could not archive the previous run ({e}) - close anything '
                         f'using "{run_dir}" and retry')
    logger.info('fresh training: previous run archived -> %s', dest)
    return dest


def _refuse_unresolved_exact_resume_transactions() -> None:
    """Recover dead exact resumes, or keep launch admission fail-closed.

    Callers already hold ``_launch_transaction_lock`` and take ``_queue_lock``
    around this helper.  A journal with a live/indeterminate child is an active
    lane owner; an invalid journal likewise requires operator recovery before
    another child may touch any mutable run lane.
    """
    if _reconcile_exact_resume_journals():
        raise ValueError(
            'a full-state resume transaction is still active or requires '
            'operator recovery before another local training can start')


@_serial_local_launch
def launch_training(user_id, dataset_id, steps: int | None = None, check_captions: bool = True,
                    base_model=None, variant: str | None = None, train_type: str | None = None,
                    allow_caption_mismatch: bool = False, masked: bool | None = None,
                    fresh: bool = False, allow_uncaptioned: bool = False,
                    allow_caption_quality: bool = False,
                    vae_path=_PERSISTED, te_path=_PERSISTED,
                    allow_unverified_weights: bool = False,
                    allow_not_ready: bool = False,
                    parent_record_id=None, resumed_from=None,
                    training_mode=None,
                    _state_restore_bundle=None,
                    _state_bridge_required: bool = False,
                    _state_resume_training_folder=None,
                    _state_resume_archived=None,
                    _state_model_pins=None,
                    _state_resume_journal=None) -> dict:
    """Export + config + pause ComfyUI (flag) + lance l'entraînement ai-toolkit
    en CLI headless (`run.py <config>`).

    ``steps`` = step cible (None → calculé par recommended_steps selon le nombre
    d'images). ai-toolkit reprend AUTOMATIQUEMENT depuis le dernier checkpoint
    présent dans le training_folder (get_latest_save_path), donc relancer avec un
    steps > dernier_step continue l'entraînement. ``fresh=True`` écarte d'abord le
    run existant (archive_previous_run) → repart de zéro sur le dataset actuel.

    Retourne {pid, config_path, log_path}. Raises RuntimeError if ai-toolkit isn't
    installed/configured (route maps this to 409, not 400 - it's a backend
    availability problem, not a bad request)."""
    # The decorator owns the launch transaction before this first action.  Exact
    # continuation passes its newly-created journal only after the outer
    # transaction has reconciled every older one; all other launches must recover
    # or refuse orphaned exact lanes before touching dataset/config/run state.
    if not _state_resume_journal:
        with _queue_lock:
            _refuse_unresolved_exact_resume_transactions()
    if not is_installed():
        raise RuntimeError('ai-toolkit is not configured')
    # The interpreter EXISTS; can it actually run ai-toolkit? Asked here, before a
    # whole dataset is exported, so a torch-less Python is named now instead of
    # surfacing minutes later as an opaque crash (GitHub #19, strouder).
    assert_interpreter_ready()
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    _assert_local_training_mode(ds, training_mode)
    # Disque plein à mi-run = checkpoints corrompus ; refuser AVANT d'exporter.
    assert_free_disk(_output_dir(), MIN_FREE_GB_TRAIN, 'a training run')
    # Garde-fou anti double-lancement : un entraînement DÉJÀ vivant (flag levé +
    # pid en vie) → refuser. Deux process sur le même GPU/dossier corrompent
    # l'optimizer partagé (incident Test/Test 2). Un pid mort avec flag encore
    # levé (avance de file) passe : on ne bloque que sur un process réellement vivant.
    if (queue_manager._get_system_state('training_in_progress', False)
            and not _training_process_is_definitely_dead(
                queue_manager._get_system_state('training_pid', None))):
        raise ValueError('a training is already in progress - wait for it to finish or queue this dataset')
    # Cheap refusal BEFORE the dataset export below: re-exporting a whole dataset
    # only to reject the launch under the spawn lock would burn minutes of disk
    # for nothing. The authoritative copy of this check lives in that lock.
    _assert_no_vision_pass_on_gpu()
    if check_captions:
        assert_trainable(dataset_id, train_type=train_type,
                         allow_caption_mismatch=allow_caption_mismatch,
                         allow_uncaptioned=allow_uncaptioned,
                         allow_caption_quality=allow_caption_quality,
                         allow_not_ready=allow_not_ready,
                         variant=variant)
    # Base d'entraînement : None/'' = officielle ; sinon un merge ComfyUI qui DOIT
    # avoir été converti en diffusers d'abord (gate). On persiste le choix sur le
    # dataset → _run_name/_run_dir/list_checkpoints deviennent base-aware (run isolé).
    base_model = (base_model or '').strip() or None
    variant = (variant or '').strip().lower()
    # La famille de CE lancement vient du param train_type s'il est donné, sinon du
    # dataset — c'est elle qui fixe l'enum de variantes valide (flux2klein : 4b/9b ;
    # les autres : turbo/base/deturbo) et le défaut (Krea → Raw, flux2klein → 4B).
    launch_fam = _train_type(ds, train_type)
    recipe = None
    if launch_fam == 'zimage':
        # Strict (no silent coercion): one typo must never select a different
        # base/adapter recipe.  This validation runs before export/config/spawn.
        recipe = zimage_training_recipe(
            variant or _default_variant_for(launch_fam), base_model)
        variant = recipe['variant']
    elif variant not in _valid_variants_for(launch_fam):
        variant = _default_variant_for(launch_fam)
    launch_view = _train_context_view(ds, launch_fam, variant)
    if train_type is not None:
        ds.train_type = train_type
    # Conversion diffusers : UNIQUEMENT pour Z-Image (SDXL = single-file direct,
    # pas de conversion → on ne bloque pas sur is_converted).
    if base_model and _train_type(ds) == 'zimage':
        from .zimage_convert import is_converted
        if not is_converted(base_model):
            raise ValueError('custom base not converted - prepare it first (button "Convert base")')
    # SDXL : la base vient brute du body → whitelist serveur (anti path-traversal,
    # comme prepare-base le fait pour Z-Image). Refus immédiat si inconnue. Un
    # chemin ABSOLU est le champ « Custom weights… » (validé par le preflight
    # ci-dessous) → il contourne délibérément la whitelist de basenames.
    if (base_model and _train_type(ds) == 'sdxl' and not _is_custom_weights(base_model)
            and base_model not in _sdxl_base_choices()):
        raise ValueError('unknown SDXL checkpoint')
    # --- Custom base/vae/te : whitelist STRICTE par famille + preflight avant spawn.
    # VAE/TE ne sont honorés QUE par SDXL (ai-toolkit) → refuser explicitement pour
    # toute autre famille (jamais d'ignore silencieux). `_PERSISTED` = « non fourni
    # par l'appelant » → on garde la valeur persistée (continue/queue) ; une valeur
    # explicite (même vide) remplace. Une famille non-SDXL n'emporte jamais de VAE/TE.
    _prov_vae = vae_path is not _PERSISTED and (vae_path or '').strip()
    _prov_te = te_path is not _PERSISTED and (te_path or '').strip()
    if launch_fam not in VAE_TE_OVERRIDE_FAMILIES:
        if _prov_vae or _prov_te:
            raise ValueError('VAE / text-encoder overrides are SDXL-only')
        eff_vae = eff_te = None
    else:
        eff_vae = (ds.train_vae_path if vae_path is _PERSISTED
                   else ((vae_path or '').strip() or None))
        eff_te = (ds.train_te_path if te_path is _PERSISTED
                  else ((te_path or '').strip() or None))
    # Preflight (fichier existe, header safetensors lisible, sniff d'arch) — un
    # sniff non concluant lève un refus CONFIRMABLE (_UNVERIFIED_MARKER), levé par
    # `allow_unverified_weights` exactement comme UNCAPTIONED.
    preflight_custom_paths(launch_fam, weights=base_model, vae_path=eff_vae,
                           te_path=eff_te,
                           allow_unverified_weights=allow_unverified_weights)
    assert_zimage_custom_recipe_confirmed(
        launch_fam, base_model, variant,
        allow_unverified_weights=allow_unverified_weights)
    # Krea 2 : refuser TÔT si l'ai-toolkit installé n'a pas l'arch krea2 (sinon
    # fallback silencieux vers le loader SD legacy → mauvais modèle, plantage confus).
    if _train_type(ds) == 'krea' and not _aitoolkit_supports_krea():
        raise ValueError(
            "ai-toolkit doesn't support Krea 2 yet (krea2 arch missing) - "
            "update it (git pull) before training a Krea LoRA.")
    # FLUX.2 Klein : même garde que Krea (archs d'EXTENSION, fallback SD silencieux
    # sur un ai-toolkit pas à jour → LoRA corrompu, cf. _aitoolkit_supports_flux2klein).
    if _train_type(ds) == 'flux2klein' and not _aitoolkit_supports_flux2klein():
        raise ValueError(
            "ai-toolkit doesn't support FLUX.2 Klein yet (flux2_klein arch missing) - "
            "update it (git pull) before training a FLUX.2 Klein LoRA.")
    # Anima : même garde (arch d'EXTENSION, PR #860 mergée le 2026-07-15) — un
    # ai-toolkit antérieur retombe en silence sur le loader SD legacy → LoRA corrompu.
    if _train_type(ds) == 'anima' and not _aitoolkit_supports_anima():
        raise ValueError(
            "ai-toolkit doesn't support Anima yet (anima arch missing) - "
            "update it (git pull) before training an Anima LoRA.")
    if (_optimizer_eff(launch_view) == 'automagic3'
            and not _aitoolkit_supports_automagic3()):
        raise ValueError(
            "ai-toolkit doesn't support Automagic3 yet - update it (git pull) "
            "or choose another optimizer before training.")
    # Slider mode (Beta) : the modern `concept_slider` trainer is an ai-toolkit
    # EXTENSION — an older install would crash at job boot on the unknown process
    # type. Refuse early with the fix, like the krea2/flux2klein arch guards.
    if slider_mode_enabled(ds) and not _aitoolkit_supports_concept_slider():
        raise ValueError(
            "ai-toolkit doesn't ship the concept_slider trainer - "
            "update it (git pull) before training a Slider LoRA.")
    # Garde-fou anti-collision de dossier : un AUTRE dataset du user avec le même
    # (trigger, base, recette) écrirait dans le même run → LoRA mélangés. Refuser AVANT de
    # persister/lancer, en nommant le conflit pour que l'utilisateur change un trigger.
    clash = find_run_collision(user_id, dataset_id, base_model=base_model,
                               variant=variant)
    if clash:
        raise ValueError(
            f"training collision: dataset '{clash.name}' (#{clash.id}) already uses "
            f"the same trigger '{ds.trigger_word}' on the same base - they would write "
            f"to the same folder. Change the trigger_word of one of the two before training.")
    ds.train_base_model = base_model
    ds.train_variant = variant
    # Persist the resolved SDXL VAE/TE overrides (None on every other family) so the
    # run-dir tag, the config, and continue/queue replays all read the same triplet.
    ds.train_vae_path = eff_vae
    ds.train_te_path = eff_te
    fds.db.session.commit()
    # The bridge is an opt-in overlay around one inspected ai-toolkit source
    # shape.  Unknown revisions keep training normally but cannot claim exact
    # checkpoints.  A requested restore is fail-closed.
    from . import aitoolkit_state_bridge
    _bridge_probe = aitoolkit_state_bridge.probe(_aitoolkit_dir())
    _bridge_candidate = bool(
        _bridge_probe.get('supported')
        and _bridge_probe.get('aitoolkit_revision'))
    if _state_bridge_required and not _bridge_candidate:
        reasons = '; '.join(_bridge_probe.get('reasons') or ())
        raise ValueError(
            'full-state resume is unavailable for this ai-toolkit installation'
            + (f': {reasons}' if reasons else
               ' because its revision/lifecycle cannot be verified'))
    # Repartir de zéro : écarter le run existant APRÈS la persistance base/variante
    # (_run_name lit les valeurs persistées → on archive bien LE run qui serait repris).
    # Serialise the first live-lane mutation with exact continuation.  An exact
    # resume holds this same re-entrant lock from archive/seed through Popen, so
    # a concurrent fresh launch cannot move or replace its newly seeded lane.
    with _queue_lock:
        archived = archive_previous_run(ds) if fresh else None
    # Steps adaptatifs si non imposés ; sinon override borné (jamais < 500).
    steps = (default_steps(ds, train_type=launch_fam, variant=variant)
             if steps is None else max(500, int(steps)))
    # masked : masques personne exportés à côté du dataset → la job-config passe
    # en masked training (fond 10 %). OFF ou indispo = historique. `None` (the
    # default, and what every fresh launch now sends) = read the dataset's stored
    # setting; an explicit bool is a per-RUN override replayed by ▶ Continue.
    masked = resolve_masked(ds, masked)
    _bridge_model_pins = None
    if _bridge_candidate:
        try:
            from . import training_state_identity
            _pin_probe_config = build_job_config(
                ds, '<lds-dataset>', steps=steps)
            _configure_exact_state_dataloaders(_pin_probe_config)
            _bridge_model_pins = training_state_identity.pin_job_model_artifacts(
                _pin_probe_config,
                cache_dir=_hf_hub_cache(),
                token=cfg.secret('HF_TOKEN'),
                pins=_state_model_pins,
            )
        except Exception as exc:
            if _state_bridge_required:
                raise ValueError(
                    f'full-state resume model pinning could not be established: {exc}'
                ) from exc
            logger.warning(
                'exact-state bridge disabled: model inputs could not be pinned: %s',
                exc, exc_info=True)
            _bridge_candidate = False
            _bridge_model_pins = None
    dataset_folder, _prepared = _export_and_freeze_local_dataset(
        user_id, dataset_id, masked=masked, base_model=base_model)
    _job_config = build_job_config(ds, dataset_folder, steps=steps)
    if _bridge_candidate:
        _configure_exact_state_dataloaders(_job_config)
        try:
            training_state_identity.pin_job_model_artifacts(
                _job_config,
                cache_dir=_hf_hub_cache(),
                token=cfg.secret('HF_TOKEN'),
                pins=_bridge_model_pins,
            )
        except Exception as exc:
            if _state_bridge_required:
                raise ValueError(
                    f'full-state resume model pinning could not be established: {exc}'
                ) from exc
            logger.warning(
                'exact-state bridge disabled while applying model pins: %s',
                exc, exc_info=True)
            _bridge_candidate = False
            _job_config = build_job_config(ds, dataset_folder, steps=steps)
    config_path = write_job_config(
        ds, dataset_folder, steps=steps,
        exact_state_bridge=_bridge_candidate,
        job_config=_job_config)
    # Environnement du sous-process d'entraînement (HF_HOME + auth Hugging Face,
    # cf. training_subprocess_env). Jamais shell=True ; args en liste.
    env = training_subprocess_env()
    # Context/status files live inside the mutable run lane.  Protect their
    # publication with the same lock as lane archive/seed and final admission;
    # otherwise a launch which passed the cheap preflight earlier could overwrite
    # an exact resume's context before either request reaches the spawn lock.
    with _queue_lock:
        run_dir = _run_root(
            ds, base_model=base_model, family=launch_fam, variant=variant)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = _run_log_path(
            ds, base_model=base_model, family=launch_fam, variant=variant)
    run_token = secrets.token_hex(16)
    # The Dataset manifest/snapshot was frozen atomically with the export above;
    # only the short registry write remains for the spawn transaction below.
    from . import checkpoint_registry
    _bridge_identity_path = None
    _bridge_status_path = None
    with _queue_lock:
        # A competing request may have completed its expensive freeze while this
        # one waited. Re-check ownership before publishing context/status into
        # the live lane; the final spawn check would be too late because those
        # files are already consumed by the winning child.
        if (queue_manager._get_system_state('training_in_progress', False)
                and not _training_process_is_definitely_dead(
                    queue_manager._get_system_state('training_pid', None))):
            raise ValueError(
                'a training is already in progress - wait for it to finish or '
                'queue this dataset')
        if _bridge_candidate:
            try:
                from . import training_state_identity
                with open(config_path, encoding='utf-8') as _config_file:
                    _job_config = json.load(_config_file)
                _job_config['_lds_state'] = {'masked': bool(masked)}
                _bridge_identity = training_state_identity.build_identity(
                    job_config=_job_config,
                    prepared=_prepared,
                    toolkit_probe=_bridge_probe,
                    python_path=_venv_python(),
                )
                _bridge_identity_path = training_state_identity.write_identity(
                    run_dir / '.lds-state' / 'context.json',
                    _bridge_identity,
                )
                _bridge_status_path = run_dir / '.lds-state' / 'bridge-status.json'
                env = aitoolkit_state_bridge.subprocess_environment(
                    aitoolkit_dir=_aitoolkit_dir(),
                    status_file=_bridge_status_path,
                    identity_file=_bridge_identity_path,
                    restore_dir=_state_restore_bundle,
                    keep=2,
                    strict=bool(_state_restore_bundle),
                    base=env,
                )
            except Exception as exc:
                if _state_bridge_required:
                    raise ValueError(
                        'full-state resume compatibility could not be established: '
                        f'{exc}'
                    ) from exc
                logger.warning(
                    'exact-state bridge disabled for this launch: %s', exc,
                    exc_info=True)
                _bridge_candidate = False
    # The training queue lock serializes launch/Stop ownership; the shared GPU
    # arbiter also covers vision's check -> flag handoff. Keep this lock order
    # everywhere training launches: queue ownership first, GPU admission second.
    # The verified Ollama handoff is intentionally *inside* both locks: releasing
    # it before acquiring the shared arbiter would let a vision pass claim the
    # GPU in the interval before Popen.
    with _queue_lock, GPU_ARBITER_LOCK:
        if (queue_manager._get_system_state('training_in_progress', False)
                and not _training_process_is_definitely_dead(
                    queue_manager._get_system_state('training_pid', None))):
            raise ValueError(
                'a training is already in progress - wait for it to finish or '
                'queue this dataset')
        # Authoritative: a vision pass may have grabbed the window during the
        # export above. Checked under the SAME lock as the live-run test so the
        # two GPU owners can never both believe they won the card.
        _assert_no_vision_pass_on_gpu()
        # ComfyUI keeps rendering while its worker polls outside this lock. The
        # durable queue state is therefore the admission record for an already
        # running prompt as well as for a pending one.
        if queue_manager.has_comfyui_work():
            from ..gpu_window import GpuBusyError
            raise GpuBusyError(
                'ComfyUI has queued or active work, so local training cannot take the GPU. '
                'Wait for it to finish or cancel it safely first.')
        try:
            from .ollama_gpu_fence import ensure_released_for_comfy
            ollama_released = ensure_released_for_comfy()
        except Exception as exc:
            logger.exception('could not verify Ollama GPU release before training')
            from ..gpu_window import GpuBusyError
            raise GpuBusyError(
                'Could not verify that Ollama released the GPU before local training. '
                'Check Ollama, then try again.') from exc
        if not ollama_released:
            from ..gpu_window import GpuBusyError
            raise GpuBusyError(
                'Ollama still owns the GPU, so local training cannot start safely. '
                'Wait for the vision task to finish or unload it, then try again.')
        # Provenance registry: record WHICH dataset version this launch trains on
        # only after this request has won the process slot.
        # Honest provenance: a launch waved through despite a readiness blocker
        # records « acknowledged_not_ready » in its settings snapshot (surfaced in
        # the Runs-hub Share config) — discreet, so a thin run is explainable later.
        _launch_settings = launch_settings_snapshot(ds, masked=masked)
        if allow_not_ready and isinstance(_launch_settings, dict):
            _launch_settings = {**_launch_settings, 'acknowledged_not_ready': True}
        _run_record = checkpoint_registry.register_launch(
            user_id, dataset_id, family=launch_fam, source='local',
            base_model=base_model or '', variant=variant, masked=bool(masked),
            steps=int(steps), settings=_launch_settings, prepared=_prepared,
            parent_record_id=parent_record_id, resumed_from=resumed_from)
        if _run_record is None:
            raise RuntimeError(
                'could not persist the Dataset provenance for local training; '
                'no training process was started')
        queue_manager._set_system_state('training_error', None, ttl_seconds=1)
        identity = {
            'training_in_progress': True,
            'training_dataset_id': int(dataset_id),
            'training_target_step': int(steps),
            'training_run_token': run_token,
            'training_train_type': launch_fam,
            'training_slider_mode': slider_mode_enabled(ds),
            'training_variant': variant,
            'training_base_model': base_model or '',
            'training_effective_base': (
                recipe.get('effective_base') if recipe else (base_model or None)),
            'training_training_adapter': (
                recipe.get('training_adapter') if recipe else None),
            'training_recipe_version': (
                recipe.get('recipe_version') if recipe else None),
        }
        logf = None
        proc = None
        try:
            # Publish the durable admission identity and launching intent in the
            # same pre-spawn failure region as Popen. A queue write can fail after
            # ``training_in_progress`` but before the rest of the identity; every
            # such partial fence must be cleared while we still know no child
            # exists.
            for key, value in identity.items():
                queue_manager._set_system_state(
                    key, value, ttl_seconds=_TRAIN_STATE_TTL)
            if _state_resume_journal:
                # Durable intent precedes Popen. If Flask dies after spawn but
                # before the journal's PID update, reconciliation keeps the
                # transaction fail-closed.
                _mark_exact_resume_launching(
                    _state_resume_journal, run_token)
            logf = open(log_path, 'w', encoding='utf-8')
            proc = subprocess.Popen(
                [str(_venv_python()), 'run.py', config_path],
                cwd=str(_aitoolkit_dir()), env=env, shell=False,
                stdout=logf, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as e:
            if logf is not None:
                try:
                    logf.close()
                except OSError:
                    pass
            try:
                _clear_training_identity(ttl_seconds=None)
            except Exception:
                logger.exception(
                    'could not clear partial pre-spawn training identity')
            if isinstance(e, (FileNotFoundError, OSError)):
                raise ValueError(f"could not start training: {e}") from e
            raise
        # Popen success is the irreversible launch boundary.  From this point on,
        # no persistence failure may escape to continue_training(), whose
        # pre-spawn exception path restores the archived lane.  The already
        # published run token and launching journal intentionally remain a
        # fail-closed GPU fence if richer PID identity cannot be persisted.
        birth_time = None
        try:
            birth_time = _record_training_process_identity(proc.pid)
        except Exception:
            logger.exception(
                'could not persist spawned training process identity; '
                'keeping the GPU fence fail-closed')
        if _state_resume_journal:
            try:
                _update_exact_resume_journal(
                    _state_resume_journal,
                    phase='spawned',
                    pid=int(proc.pid),
                    pid_create_time=birth_time,
                    run_token=run_token,
                )
            except Exception:
                # The child already exists: never raise into continue() and
                # roll its source lane back underneath a live process. The
                # earlier launching intent + durable fence remain recoverable.
                logger.exception(
                    'could not attach process identity to exact-resume journal')
    # Watcher event-driven : libère ComfyUI / enchaîne la file dès la fin du
    # process (le poll de /train/status reste le filet de secours).
    _exact_resume_transaction = None
    if (_state_restore_bundle and _state_resume_training_folder
            and _state_resume_archived):
        _exact_resume_transaction = {
            'training_folder': str(_state_resume_training_folder),
            'archived': str(_state_resume_archived),
            'status_path': (
                str(_bridge_status_path) if _bridge_status_path is not None else None),
            'journal_path': (
                str(_state_resume_journal)
                if _state_resume_journal is not None else None),
        }
    try:
        from flask import current_app
        threading.Thread(target=_watch_training,
                         args=(current_app._get_current_object(), proc, log_path,
                               int(dataset_id), _exact_resume_transaction),
                         daemon=True).start()
    except Exception as e:
        logger.warning("watcher training non démarré : %s", e)
    return {'started': True, 'pid': proc.pid, 'config_path': config_path, 'steps': steps,
            'dataset_folder': dataset_folder, 'log_path': log_path,
            'fresh': bool(fresh), 'archived_run': archived,
            'run_token': run_token}


def _validate_resume_contract(resume_mode, state_bundle_id):
    mode = str(resume_mode or '').strip().lower()
    if mode not in ('weights_only', 'full_state'):
        raise ValueError("resume_mode must be 'weights_only' or 'full_state'")
    if mode == 'full_state' and not state_bundle_id:
        raise ValueError('full-state resume requires state_bundle_id')
    if mode == 'weights_only' and state_bundle_id is not None:
        raise ValueError(
            'state_bundle_id is only valid with resume_mode=full_state')
    return mode


def _exact_resume_identity(ds, user_id, dataset_id, base, family, variant,
                           target_steps, masked):
    """Current context + compatibility spec, built before any run is moved."""
    from . import aitoolkit_state_bridge, checkpoint_registry
    from . import training_state_identity
    probe = aitoolkit_state_bridge.probe(_aitoolkit_dir())
    if not probe.get('supported') or not probe.get('aitoolkit_revision'):
        reasons = '; '.join(probe.get('reasons') or ())
        raise ValueError(
            'full-state resume is unavailable for this ai-toolkit installation'
            + (f': {reasons}' if reasons else
               ' because its revision/lifecycle cannot be verified'))
    launch_view = _train_context_view(
        ds, family, variant, base_model=base)
    job_config = build_job_config(
        launch_view, '<lds-dataset>', steps=target_steps,
        training_folder='<lds-training-folder>')
    _configure_exact_state_dataloaders(job_config)
    pins = training_state_identity.pin_job_model_artifacts(
        job_config,
        cache_dir=_hf_hub_cache(),
        token=cfg.secret('HF_TOKEN'),
    )
    job_config['_lds_state'] = {'masked': bool(resolve_masked(ds, masked))}
    prepared = checkpoint_registry.prepare_launch(
        user_id, dataset_id, base_model=base)
    try:
        identity = training_state_identity.build_identity(
            job_config=job_config,
            prepared=prepared,
            toolkit_probe=probe,
            python_path=_venv_python(),
        )
    except Exception as exc:
        raise ValueError(
            f'full-state resume compatibility could not be established: {exc}'
        ) from exc
    return identity, training_state_identity.compatibility_spec(identity), pins


_EXACT_RESUME_JOURNAL_SCHEMA = 'lds.exact-resume-transaction/v1'
_EXACT_RESUME_JOURNAL_DIR = '.lds-exact-resume-transactions'
_EXACT_RESUME_JOURNAL_RE = re.compile(r'^[0-9a-f]{32}\.json$')
_REPARSE_POINT = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x0400)


def _exact_resume_journal_root() -> Path:
    return Path(_output_dir()) / _EXACT_RESUME_JOURNAL_DIR


def _absolute_lexical(path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _fsync_directory(path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _journal_path(transaction_id: str) -> Path:
    if not re.fullmatch(r'[0-9a-f]{32}', str(transaction_id or '')):
        raise ValueError('invalid exact-resume transaction id')
    return _exact_resume_journal_root() / f'{transaction_id}.json'


def _validate_exact_resume_journal(path, value) -> dict:
    """Validate every path before a journal can drive a rename."""
    path = _absolute_lexical(path)
    root = _absolute_lexical(_exact_resume_journal_root())
    if path.parent != root or not _EXACT_RESUME_JOURNAL_RE.fullmatch(path.name):
        raise ValueError('exact-resume journal path is outside its fixed root')
    if not isinstance(value, dict):
        raise ValueError('exact-resume journal is not an object')
    if value.get('schema') != _EXACT_RESUME_JOURNAL_SCHEMA:
        raise ValueError('unknown exact-resume journal schema')
    if value.get('transaction_id') != path.stem:
        raise ValueError('exact-resume journal id mismatch')
    output = _absolute_lexical(_output_dir())
    training_folder = _absolute_lexical(value.get('training_folder') or '')
    archived = _absolute_lexical(value.get('archived') or '')
    status_path = _absolute_lexical(value.get('status_path') or '')
    try:
        inside_training = (
            os.path.commonpath((output, training_folder)) == str(output))
        inside_archive = (
            os.path.commonpath((output, archived)) == str(output))
    except ValueError:
        inside_training = inside_archive = False
    if not inside_training or not inside_archive:
        raise ValueError('exact-resume journal path escapes output root')
    if (
        archived.parent != training_folder.parent
        or not archived.name.startswith(
            f'{training_folder.name}_superseded_')
    ):
        raise ValueError('exact-resume archive is not the run sibling')
    expected_status = training_folder / '.lds-state' / 'bridge-status.json'
    if status_path != expected_status:
        raise ValueError('exact-resume status path does not match the live run')
    phase = value.get('phase')
    if phase not in ('prepared', 'seeded', 'launching', 'spawned'):
        raise ValueError('invalid exact-resume journal phase')
    normalized = dict(value)
    normalized.update({
        'training_folder': str(training_folder),
        'archived': str(archived),
        'status_path': str(status_path),
    })
    return normalized


def _write_exact_resume_journal(path, value) -> Path:
    from ..training_bridge.lds_aitk_bridge_contract import atomic_json_nofollow

    target = _absolute_lexical(path)
    normalized = _validate_exact_resume_journal(target, value)
    atomic_json_nofollow(target, normalized)
    return target


def _read_exact_resume_journal(path) -> dict:
    from ..training_bridge.lds_aitk_bridge_contract import read_json_nofollow

    target = _absolute_lexical(path)
    value = read_json_nofollow(target, max_bytes=64 << 10)
    return _validate_exact_resume_journal(target, value)


def _update_exact_resume_journal(path, **changes) -> dict:
    value = _read_exact_resume_journal(path)
    value.update(changes)
    _write_exact_resume_journal(path, value)
    return value


def _delete_exact_resume_journal(path) -> None:
    target = _absolute_lexical(path)
    root = _absolute_lexical(_exact_resume_journal_root())
    if target.parent != root or not _EXACT_RESUME_JOURNAL_RE.fullmatch(target.name):
        raise ValueError('refusing to delete an untrusted journal path')
    if os.name != 'nt' and os.unlink in os.supports_dir_fd:
        flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        try:
            directory_fd = os.open(root, flags)
        except FileNotFoundError:
            return
        try:
            try:
                info = os.stat(
                    target.name, dir_fd=directory_fd,
                    follow_symlinks=False)
            except FileNotFoundError:
                return
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise OSError('exact-resume journal target is unsafe')
            os.unlink(target.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return
    if _journal_path_is_unsafe(root):
        raise OSError('exact-resume journal parent is unsafe')
    try:
        info = os.lstat(target)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or (getattr(info, 'st_file_attributes', 0) & _REPARSE_POINT)
    ):
        raise OSError('exact-resume journal target is unsafe')
    os.unlink(target)
    _fsync_directory(root)


def _new_exact_resume_journal(
        dataset_id, training_folder: Path, archived: Path) -> tuple[Path, dict]:
    transaction_id = secrets.token_hex(16)
    path = _journal_path(transaction_id)
    training_folder = _absolute_lexical(training_folder)
    archived = _absolute_lexical(archived)
    value = {
        'schema': _EXACT_RESUME_JOURNAL_SCHEMA,
        'transaction_id': transaction_id,
        'dataset_id': int(dataset_id),
        'training_folder': str(training_folder),
        'archived': str(archived),
        'status_path': str(
            training_folder / '.lds-state' / 'bridge-status.json'),
        'phase': 'prepared',
        'created_at': datetime.now().isoformat(),
    }
    _write_exact_resume_journal(path, value)
    return path, value


def _copy_verified_exact_checkpoint(source: Path, destination: Path, record) -> None:
    """Copy one manifest artifact from one opened handle while re-verifying it."""
    if _journal_path_is_unsafe(source.parent):
        raise ValueError('the exact-state bundle path is unsafe')
    try:
        source_info = os.lstat(source)
    except OSError as exc:
        raise ValueError(
            'the exact-state bundle checkpoint vanished before seeding') from exc
    if (
        not stat.S_ISREG(source_info.st_mode)
        or stat.S_ISLNK(source_info.st_mode)
        or (getattr(source_info, 'st_file_attributes', 0) & _REPARSE_POINT)
    ):
        raise ValueError('the exact-state bundle checkpoint is not a regular file')
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    source_fd = os.open(source, flags)
    temporary = destination.with_name(
        f'.{destination.name}.{secrets.token_hex(8)}.tmp')
    output_fd = None
    try:
        opened = os.fstat(source_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_size) != int(record.size_bytes)
        ):
            raise ValueError(
                'the exact-state bundle checkpoint changed before seeding')
        # Windows otherwise opens this descriptor in text mode.  ``os.write``
        # then expands every LF byte to CRLF, silently corrupting arbitrary
        # binary safetensors even though the source-side digest still matches.
        out_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        out_flags |= getattr(os, 'O_BINARY', 0)
        out_flags |= getattr(os, 'O_NOFOLLOW', 0)
        output_fd = os.open(temporary, out_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 8 << 20)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_fd, view)
                view = view[written:]
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = None
        if (
            copied != int(record.size_bytes)
            or digest.hexdigest() != record.sha256
        ):
            raise ValueError(
                'the exact-state bundle checkpoint changed before seeding')
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        os.close(source_fd)
        if output_fd is not None:
            os.close(output_fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _archive_and_seed_exact_bundle(
        user_id, dataset_id, base, family, variant, chosen_filename,
        bundle_id, inspection):
    """Move the source run aside and seed a fresh run from its verified bundle.

    The canonical bundle stays immutable in the archived run and the bridge reads
    it there.  Only its public safetensors checkpoint is copied into the new
    save_root so ai-toolkit creates the same network before raw parameters/state
    are restored.
    """
    from . import training_state_bundle
    from ..training_bridge.lds_aitk_bridge_contract import ARTIFACT_FILENAMES

    ds = fds.get_dataset(user_id, dataset_id)
    trigger = _safe_trigger(ds)
    training_folder = _run_root(ds, base, family, variant)
    save_root = training_folder / f'lora_{trigger}'
    if not training_folder.is_dir():
        raise ValueError('run folder missing - cannot restore full training state')
    token = secrets.token_hex(4)
    archived = Path(
        f'{training_folder}_superseded_'
        f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{token}')
    journal_path, _ = _new_exact_resume_journal(
        dataset_id, training_folder, archived)
    try:
        os.rename(training_folder, archived)
        _fsync_directory(training_folder.parent)
    except Exception:
        _delete_exact_resume_journal(journal_path)
        raise
    try:
        archived_save_root = archived / f'lora_{trigger}'
        bundle = training_state_bundle.resolve_bundle_path(
            archived_save_root, bundle_id, require_exists=True)
        checkpoint_name = ARTIFACT_FILENAMES['public_checkpoint']
        record = next(
            (item for item in inspection.artifacts
             if item.name == checkpoint_name),
            None)
        if record is None:
            raise ValueError(
                'the exact-state bundle has no public checkpoint artifact')
        source = bundle.joinpath(*record.path.split('/'))
        save_root.mkdir(parents=True, exist_ok=False)
        seeded = save_root / chosen_filename
        _copy_verified_exact_checkpoint(source, seeded, record)
        os.utime(seeded, None)
        logger.info(
            'full-state continuation: seeded %s; source run -> %s',
            chosen_filename, archived)
        _update_exact_resume_journal(journal_path, phase='seeded')
        return str(archived), str(bundle), str(journal_path)
    except Exception:
        # Roll back without deleting evidence: if the fresh folder contains any
        # partial work, move it beside the archive before restoring the source.
        if training_folder.exists():
            failed = Path(
                f'{training_folder}_failed_full_state_'
                f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{token}')
            os.rename(training_folder, failed)
        os.rename(archived, training_folder)
        _fsync_directory(training_folder.parent)
        _delete_exact_resume_journal(journal_path)
        raise


def _rollback_unlaunched_exact_resume(training_folder, archived):
    """Restore the source lane when launch failed before a child process existed."""
    training_folder = Path(training_folder)
    archived = Path(archived)
    if not archived.is_dir():
        return
    if training_folder.exists():
        failed = Path(
            f'{training_folder}_failed_full_state_'
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{secrets.token_hex(4)}')
        os.rename(training_folder, failed)
    os.rename(archived, training_folder)
    _fsync_directory(training_folder.parent)


def _launch_exact_resume_transaction(
        user_id, dataset_id, base, family, variant, chosen_filename,
        bundle_id, inspection, training_folder, launch_kwargs,
        *, allow_dead_predecessor=False):
    """Archive, seed and launch one exact continuation under one ownership lock.

    Validation and hashing deliberately stay outside this lock, but the first
    live-lane rename through Popen (or pre-spawn rollback) is indivisible with
    respect to every local launch/stop path.  This prevents a competing child
    from taking ownership of the fresh lane before this transaction's exception
    handler decides whether it is still safe to restore the archive.
    """
    with _launch_transaction_lock:
        with _queue_lock:
            _refuse_unresolved_exact_resume_transactions()
            if queue_manager._get_system_state('training_in_progress', False):
                previous_is_dead = _training_process_is_definitely_dead(
                    queue_manager._get_system_state('training_pid', None))
                if not (allow_dead_predecessor and previous_is_dead):
                    raise ValueError('a training is already in progress')
            archived, exact_bundle_path, journal_path = (
                _archive_and_seed_exact_bundle(
                    user_id, dataset_id, base, family, variant, chosen_filename,
                    bundle_id, inspection))
        try:
            result = launch_training(
                user_id,
                dataset_id,
                **launch_kwargs,
                _state_restore_bundle=exact_bundle_path,
                _state_bridge_required=True,
                _state_resume_training_folder=training_folder,
                _state_resume_archived=archived,
                _state_resume_journal=journal_path,
            )
        except Exception:
            # launch_training is required never to propagate after Popen. Thus an
            # exception here is proof that no child exists. The transaction lock
            # excludes competing launches; the queue lock makes the lane rename
            # atomic with Stop/recovery ownership.
            with _queue_lock:
                _rollback_unlaunched_exact_resume(training_folder, archived)
                _delete_exact_resume_journal(journal_path)
            raise
        return result, archived, exact_bundle_path, journal_path


def _seed_continuation_from(user_id, dataset_id, base, family, variant,
                            chosen_filename) -> str:
    """Prepare an explicit weights-only LOCAL continuation.

    ai-toolkit otherwise auto-resumes from mutable sidecar state in the current
    run folder.  That would make a UI choice labelled "weights only" untrue even
    for the latest checkpoint.  Move the whole source run aside (never delete it)
    and seed only the chosen public checkpoint into a fresh save_root.  Optimizer,
    scheduler, scaler, RNG and dataloader state therefore restart deliberately.
    Returns the archived folder path.
    """
    ds = fds.get_dataset(user_id, dataset_id)
    trigger = _safe_trigger(ds)
    training_folder = _run_root(ds, base, family, variant)
    if not training_folder.is_dir():
        raise ValueError('run folder missing - cannot seed the earlier checkpoint')
    dest = f'{training_folder}_superseded_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    try:
        os.rename(training_folder, dest)
    except OSError as e:
        raise ValueError(f'could not set the later checkpoints aside ({e}) - close '
                         f'anything using "{training_folder}" and retry')
    save_root = training_folder / f'lora_{trigger}'
    src = os.path.join(dest, f'lora_{trigger}', chosen_filename)
    if not os.path.isfile(src):
        os.rename(dest, training_folder)                    # roll back — never half-moved
        raise ValueError('chosen checkpoint vanished before seeding')
    save_root.mkdir(parents=True, exist_ok=True)
    seeded = save_root / chosen_filename
    shutil.copy2(src, seeded)
    os.utime(seeded, None)                                  # newest by ctime → ai-toolkit picks it
    logger.info('continuation: seeded %s into fresh save_root, superseded run -> %s',
                chosen_filename, dest)
    return dest


def continue_training(user_id, dataset_id, extra_steps: int = 1000,
                      base_model=_PERSISTED, variant=None, train_type=None,
                      masked=None, allow_unverified_weights=False,
                      allow_caption_mismatch=False, allow_uncaptioned=False,
                      allow_caption_quality=False, from_step=None, overrides=None,
                      resume_mode='weights_only', state_bundle_id=None,
                      allow_not_ready=False, _allow_dead_predecessor=False,
                      training_mode='lora') -> dict:
    """Reprend l'entraînement d'une base et vise ``step_de_reprise + extra_steps``.
    ai-toolkit auto-resume depuis le training_folder ; il faut donc qu'au moins un
    checkpoint existe POUR CETTE BASE.

    ``from_step`` absent → reprise depuis le DERNIER checkpoint (comportement
    historique, relance en place). Fourni → reprise depuis CE step précis ; s'il est
    INFÉRIEUR au dernier, on repart d'un checkpoint plus ancien SANS rien détruire :
    le run est archivé de côté et seul le checkpoint choisi est semé dans un dossier
    propre (_seed_continuation_from). ``overrides`` = sous-ensemble sûr de réglages
    (cadence de sauvegarde/preview, prompts de preview) appliqué avant la reprise ;
    tout autre réglage est refusé (il romprait la compatibilité des poids).

    `base_model` absent → base persistée du dataset (ex. file d'attente). Fourni
    (sélection UI) → on reprend le run DE CETTE base précise : sinon on proposait
    « Continuer » sur une base sans run et on relançait en fait l'ancienne base."""
    # Validate the caller-controlled restore contract before interpreter probes,
    # dataset reads, archives or settings writes.
    resume_mode = _validate_resume_contract(resume_mode, state_bundle_id)
    # Queue advancement calls this while the previous run's flag is still set
    # (so ComfyUI never grabs the GPU between jobs).  Only a *live* PID blocks;
    # a dead predecessor is precisely the normal queued-continue transition.
    if queue_manager._get_system_state('training_in_progress', False):
        previous_is_dead = _training_process_is_definitely_dead(
            queue_manager._get_system_state('training_pid', None))
        if not (_allow_dead_predecessor and previous_is_dead):
            raise ValueError('a training is already in progress')
    # Same interpreter gate as a fresh launch: a resume spawns the very same
    # ai-toolkit command, so a torch-less Python must be named here too.
    assert_interpreter_ready()
    # Validate the safe-subset overrides BEFORE any side effect: a forbidden key
    # (rank/alpha/optimizer/…) must fail loudly with nothing archived or persisted.
    override_patch = validate_resume_overrides(overrides)
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    _assert_local_training_mode(ds, training_mode)
    base = (ds.train_base_model if ds else None) if base_model is _PERSISTED else base_model
    fam = _train_type(ds, train_type) if ds else train_type
    var = (variant or (ds.train_variant if ds else None)
           or _default_variant_for(fam))
    if fam == 'zimage':
        var = zimage_training_recipe(var, base)['variant']
    assert_zimage_custom_recipe_confirmed(
        fam, base, var, allow_unverified_weights=allow_unverified_weights)
    # A resume re-exports the CURRENT mutable dataset. A checkpoint proves only
    # that an older snapshot trained successfully; it cannot waive caption/image
    # guards after rows were edited, removed or re-captioned.
    assert_trainable(dataset_id, train_type=fam,
                     allow_caption_mismatch=allow_caption_mismatch,
                     allow_uncaptioned=allow_uncaptioned,
                     allow_caption_quality=allow_caption_quality,
                     allow_not_ready=allow_not_ready,
                     variant=var)
    cks = list_checkpoints(user_id, dataset_id, base_model=base,
                           family=fam, variant=var)
    if not cks:
        raise ValueError("no checkpoint to resume for this base - run a training first")
    latest = max(c['step'] for c in cks)
    # Which checkpoint to resume FROM. Default = the latest.
    # A specific step lets the user restart from an earlier, better epoch.
    if from_step is None:
        resume_step = latest
        matches = [c for c in cks if c['step'] == resume_step]
        chosen = min(matches, key=lambda c: bool(c.get('final')))
    else:
        try:
            resume_step = int(from_step)
        except (TypeError, ValueError):
            raise ValueError('from_step must be an integer step')
        matches = [c for c in cks if c['step'] == resume_step]
        if not matches:
            avail = sorted({c['step'] for c in cks})
            raise ValueError(
                f'no checkpoint at step {resume_step} for this run (available: {avail})')
        # Ties (a numbered save and the bare final at the same step): prefer the
        # numbered file — it carries a clean step and is never the run's live final.
        chosen = min(matches, key=lambda c: bool(c.get('final')))
    # Lineage: the record this continuation resumes FROM is the record that
    # PRODUCED the file being loaded (list_checkpoints stamps `record_id` on every
    # save), NOT merely the newest record of the lane. One lane holds several runs,
    # and their saves coexist in the same run dir: attributing the child to the
    # newest record drew an edge to a run whose weights were never touched — and,
    # worse, one whose rank could differ, which is what makes the graph's claim
    # physically impossible. Falls back to the lane's newest record when the file
    # carries no stamp (pre-registry saves). Best-effort like all provenance — a
    # resolution failure leaves the edge NULL and NEVER blocks the continuation.
    from . import checkpoint_registry
    _src_ck = resume_source_checkpoint(cks, resume_step)
    try:
        _parent = checkpoint_registry.record_by_id((_src_ck or {}).get('record_id'))
        if _parent is None:
            _parent = checkpoint_registry.newest_record_for(dataset_id, fam, base or '', var)
    except Exception:
        _parent = None
    # …and the geometry those weights were trained with is not negotiable. The
    # local lane trains from the dataset's PERSISTED settings, so it cannot quietly
    # inherit the parent's rank without rewriting the user's own settings behind
    # their back (the cloud lane can — it carries a per-run snapshot). Refuse
    # loudly, here: nothing has been archived, persisted or launched yet.
    _parent_geometry = checkpoint_registry.network_geometry(_parent)
    _legacy_lokr_error = legacy_lokr_resume_error(
        _parent_geometry, getattr(ds, 'train_settings', None))
    if _legacy_lokr_error:
        raise ValueError(_legacy_lokr_error)
    _live_geometry = launch_settings_snapshot(ds, fam)
    _conflict = describe_geometry_conflict(
        _parent_geometry,
        _live_geometry['rank'], _live_geometry['alpha'],
        network_type=_live_geometry.get('network_type'),
        lokr_factor=_live_geometry.get('lokr_factor'),
        lokr_full_rank=_live_geometry.get('lokr_full_rank'),
        conv=_live_geometry.get('conv'),
        conv_alpha=_live_geometry.get('conv_alpha'))
    if _conflict:
        raise ValueError(_conflict)
    try:
        extra = max(100, int(extra_steps))
    except (TypeError, ValueError):
        extra = 1000
    target_steps = resume_step + extra

    # Exact continuation must preserve every trajectory-shaping setting. Save
    # and sample boundaries affect ai-toolkit's main-vs-regularisation loader
    # selection, so cadence changes are not merely presentation changes.
    if resume_mode == 'full_state' and (
            'lr_factor' in override_patch
            or 'timestep_type' in override_patch
            or 'save_every' in override_patch
            or 'sample_every' in override_patch):
        raise ValueError(
            'full-state resume cannot change learning rate, timestep type, '
            'save cadence or sample cadence; '
            'keep the original trajectory or choose weights-only resume')

    exact_inspection = None
    exact_bundle_path = None
    exact_model_pins = None
    exact_journal_path = None
    archived = None
    if resume_mode == 'full_state':
        from . import training_state_bundle
        advertised = (chosen.get('resume_state') or {}).get('bundle_id')
        if advertised != state_bundle_id:
            raise ValueError(
                'the selected state bundle does not belong to this checkpoint')
        identity, expected, exact_model_pins = _exact_resume_identity(
            ds, user_id, dataset_id, base, fam, var, target_steps, masked)
        del identity
        save_root = _run_dir(
            user_id, dataset_id, base_model=base, family=fam, variant=var)
        try:
            exact_inspection = training_state_bundle.verify_bundle(
                save_root, state_bundle_id, expected=expected)
        except training_state_bundle.IncompatibleBundleError as exc:
            raise ValueError(
                'full-state bundle is incompatible with the current training '
                f'context ({exc.reason}); choose weights-only resume') from exc
        except training_state_bundle.InvalidBundleError as exc:
            raise ValueError(
                'full-state bundle is missing or corrupt '
                f'({exc.reason}); choose weights-only resume') from exc
        if exact_inspection.completed_step != resume_step:
            raise ValueError(
                'the selected state bundle does not belong to this checkpoint')
    # Resolve the LR factor (½/⅒) against THIS run's current effective settings into
    # an absolute learning_rate, refusing loudly on a Prodigy run BEFORE any side
    # effect (nothing archived or persisted). Keep current (factor absent) is a no-op.
    lr_factor = override_patch.pop('lr_factor', None)
    if lr_factor is not None:
        override_patch['learning_rate'] = resolve_resume_lr(_train_settings(ds), lr_factor)
    # Apply the safe overrides (cadence / preview prompts / LR) before building the job.
    if override_patch:
        update_train_settings(user_id, dataset_id, override_patch)
    # Reprendre AVEC la base/variante ciblée - sinon launch_training les remettrait
    # à l'officiel et ai-toolkit reprendrait depuis le mauvais run. vae/te restent
    # _PERSISTED (on garde le triplet du run). A custom Base/De-Turbo declaration
    # still requires the caller's explicit confirmation; an old run existing is
    # not proof that the custom transformer has the declared distillation type.
    needs_explicit_z_recipe = (
        fam == 'zimage' and bool(str(base or '').strip())
        and var in ('base', 'deturbo'))
    # Historical resumes already bypassed the generic custom-weight sniff: the
    # checkpoint proves that this exact local base was accepted for the run.
    # Preserve that behavior except for custom Z-Image Base/De-Turbo, whose
    # distillation recipe cannot be inferred and therefore needs a fresh,
    # explicit server-side acknowledgement.
    launch_allow_unverified = (allow_unverified_weights
                               or not needs_explicit_z_recipe)
    training_folder = (
        _run_root(ds, base, fam, var)
        if resume_mode == 'full_state' else None)
    launch_kwargs = {
        'steps': target_steps,
        'check_captions': False,
        'base_model': base,
        'variant': var,
        'train_type': fam,
        'masked': masked,
        'allow_caption_mismatch': allow_caption_mismatch,
        'allow_uncaptioned': allow_uncaptioned,
        'allow_caption_quality': allow_caption_quality,
        'allow_not_ready': allow_not_ready,
        'allow_unverified_weights': launch_allow_unverified,
        'parent_record_id': (_parent.id if _parent else None),
        'resumed_from': resume_step,
        'training_mode': training_mode,
        '_state_model_pins': (
            exact_model_pins if resume_mode == 'full_state' else None),
    }
    # Both modes launch into a fresh lane. Weights-only seeds exactly one public
    # checkpoint. Full-state keeps archive, seed, launch and any proven-pre-spawn
    # rollback inside one queue ownership transaction.
    if resume_mode == 'full_state':
        res, archived, exact_bundle_path, exact_journal_path = (
            _launch_exact_resume_transaction(
                user_id,
                dataset_id,
                base,
                fam,
                var,
                chosen['filename'],
                state_bundle_id,
                exact_inspection,
                training_folder,
                launch_kwargs,
                allow_dead_predecessor=_allow_dead_predecessor,
            )
        )
    else:
        with _launch_transaction_lock:
            with _queue_lock:
                _refuse_unresolved_exact_resume_transactions()
                if queue_manager._get_system_state(
                        'training_in_progress', False):
                    previous_is_dead = _training_process_is_definitely_dead(
                        queue_manager._get_system_state('training_pid', None))
                    if not (_allow_dead_predecessor and previous_is_dead):
                        raise ValueError('a training is already in progress')
                archived = _seed_continuation_from(
                    user_id, dataset_id, base, fam, var, chosen['filename'])
            res = launch_training(
                user_id, dataset_id, **launch_kwargs)
    res['resumed_from'] = resume_step
    res['target_steps'] = target_steps
    res['resume_mode'] = resume_mode
    res['archived_run'] = archived
    if resume_mode == 'full_state':
        res['state_bundle_id'] = state_bundle_id
    return res


def stop_training(expected_dataset_id=None, expected_run_token=None,
                  expected_dataset_table=None) -> bool:
    """Kill the local training process, then release its GPU ownership fence.

    The final state transition is deliberately fail-closed: a non-zero
    taskkill result, a missing PID, or an unavailable PID probe leaves the
    training fence in place. Releasing it without proof would let Vision or
    ComfyUI allocate the GPU while ai-toolkit may still be running.

    `expected_dataset_table` completes `expected_dataset_id`, which is an integer
    two tables now share. Omitted, it means `face_dataset` — so the image lane's
    Stop button keeps refusing, rather than killing, a video run of the same id.
    An unconditional stop (no expected id at all) is unchanged and still stops
    whatever is running: it is the "get off my GPU" button, and it is not asked
    to know what it is stopping.
    """
    # Keep the launch lock order: training ownership first, then shared GPU
    # admission. This makes the final clear atomic with Vision/ComfyUI admission.
    with _queue_lock, GPU_ARBITER_LOCK:
        current_id = queue_manager._get_system_state('training_dataset_id', None)
        current_token = queue_manager._get_system_state('training_run_token', None)
        current_table = (queue_manager._get_system_state(
            'training_dataset_table', None) or _crd.FACE)
        in_progress = bool(queue_manager._get_system_state(
            'training_in_progress', False))
        if expected_dataset_id is not None:
            try:
                same_run = int(current_id) == int(expected_dataset_id)
            except (TypeError, ValueError):
                same_run = False
            if same_run:
                same_run = current_table == (expected_dataset_table or _crd.FACE)
            if not in_progress or not same_run:
                return False
        if expected_run_token is not None:
            token_ok = bool(current_token) and secrets.compare_digest(
                str(current_token), str(expected_run_token))
            if not in_progress or not token_ok:
                return False

        pid = queue_manager._get_system_state('training_pid', None)
        pid_alive = _pid_alive(pid) if in_progress else False
        if in_progress and pid_alive is None:
            logger.error(
                'stop_training: cannot prove whether training pid %r is still alive; '
                'keeping the GPU fence', pid)
            return False

        if pid_alive:
            # Recheck the birth-time identity immediately before the PID-only OS
            # kill. If the old training exited and Windows recycled this PID between
            # probes, the replacement is never a permitted taskkill target.
            rechecked_pid_alive = _pid_alive(pid)
            if rechecked_pid_alive is None:
                logger.error(
                    'stop_training: cannot re-prove training pid %r immediately before kill; ',
                    'keeping the GPU fence', pid)
                return False
            pid_alive = rechecked_pid_alive

        if pid_alive:
            try:
                if os.name == 'nt':
                    # /T terminates ai-toolkit children such as dataloaders too.
                    completed = subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(int(pid))],
                        shell=False, capture_output=True)
                    if getattr(completed, 'returncode', 0) != 0:
                        logger.warning(
                            'stop_training: taskkill pid %s failed (rc=%s); '
                            'keeping the GPU fence', pid,
                            getattr(completed, 'returncode', None))
                        return False
                else:
                    os.kill(int(pid), 15)
            except (ValueError, OSError) as exc:
                logger.warning('stop_training: kill pid %s failed: %s', pid, exc)
                return False

            if not _wait_for_training_process_exit(pid):
                logger.warning(
                    'stop_training: pid %s did not become conclusively dead; '
                    'keeping the GPU fence', pid)
                return False

        # Stop is intentional: clear the queue before publishing the idle state.
        _save_queue([])
        _clear_training_identity(ttl_seconds=None)
        return True


def _dataset_name(dataset_id):
    if dataset_id is None:
        return None
    ds = FaceDataset.query.get(int(dataset_id))
    return ds.name if ds else f'#{dataset_id}'


def kept_uncaptioned_count(dataset_id) -> int:
    """Nombre d'images GARDÉES (status keep) sans caption - bloque l'entraînement."""
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .with_entities(FaceDatasetImage.caption).all())
    return sum(1 for (caption,) in rows if not (caption or '').strip())


def assert_trainable(dataset_id, train_type=None, allow_caption_mismatch=False,
                     allow_uncaptioned=False, allow_caption_quality=False,
                     variant=None, allow_not_ready=False) -> None:
    """Lève ValueError si le dataset n'est pas prêt : trop peu d'images gardées,
    captions manquantes, ou STYLE de caption incohérent avec le type de modèle
    (SDXL booru-native attend des tags booru ; Z-Image attend de la prose ; anima
    est HYBRIDE — les deux formes sont natives, donc aucun mismatch n'existe pour
    elle, cf. _EXPECTED_CAPTION_FORM). Le
    `train_type` effectif est passé par l'appelant car il n'est persisté qu'APRÈS
    cet appel. `allow_caption_mismatch=True` = override explicite (bouton « forcer »).
    `allow_uncaptioned=True` = confirm explicite « train anyway » : les captions
    manquantes ne sont plus un mur, juste un « êtes-vous sûr ? » (demande
    utilisateur — pouvoir expérimenter), le préfixe UNCAPTIONED: déclenche le
    confirm côté front comme MISMATCH_CAPTION:. Pour Style, les captions de contenu
    restent la règle (always-on, sans trigger). ``allow_caption_quality=True`` lève
    séparément le garde trigger-only/toutes-identiques. ``variant`` est accepté pour
    garder une signature family/variant homogène avec les recommandations de steps.

    ``allow_not_ready=True`` = case « Continue anyway » du panneau de préparation :
    lève le garde-fou QUALITÉ du plancher d'images par famille (marqueur NOT_READY:,
    miroir de la pastille de readiness). Les IMPOSSIBILITÉS PHYSIQUES ne sont JAMAIS
    levées par ce flag : 0 image gardée (rien à entraîner) et, en mode slider, la
    paire de prompts absente (aucune direction à apprendre) — ai-toolkit planterait."""
    kept = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep').count()
    ds_ = FaceDataset.query.get(dataset_id)
    if ds_ is not None and slider_mode_enabled(ds_):
        # Slider mode (Beta): images are only a denoising substrate — a small
        # varied set is enough — and captions are encoded but IGNORED by the
        # slider loss, so every caption guard below is meaningless here. What
        # IS required: the prompt pair that defines the slider direction.
        floor = TRAIN_MIN_IMAGES_SLIDER[0]
        # Physical impossibility: an empty substrate leaves nothing to denoise —
        # never waivable by the readiness ack.
        if kept == 0:
            raise ValueError(
                'slider training needs a few images as a denoising substrate — '
                'none are kept')
        # Below the substrate floor is a QUALITY guard-rail: the explicit ack
        # (allow_not_ready) lets a thin experiment through.
        if kept < floor and not allow_not_ready:
            raise ValueError(
                f'NOT_READY: only {kept} kept image(s) — slider training still '
                f'needs at least {floor} as a denoising substrate. '
                'Continue anyway to train with too few.')
        # The prompt pair is a physical requirement (no direction to learn) —
        # never waivable, even with the readiness ack.
        sc = _slider_settings(ds_)
        if not (sc.get('positive') or '').strip() or not (sc.get('negative') or '').strip():
            raise ValueError(
                'slider mode needs both a positive and a negative prompt — '
                'they define the two ends of the slider')
        return
    # Effective family drives the per-family image floor (the readiness blocker).
    # Resolve it up here so the floor guard runs before the caption guards below —
    # and reuse it for the caption style↔type check further down.
    ttype = (train_type or '').strip().lower()
    if not ttype:
        ttype = (getattr(ds_, 'train_type', None) or 'zimage').lower() if ds_ else 'zimage'
    floor = TRAIN_MIN_IMAGES.get(ttype, (12, 20))[0]
    label = _FAMILY_LABEL.get(ttype, ttype)
    # Physical impossibility: an empty dataset can't train (never waivable).
    if kept == 0:
        raise ValueError('no kept images — keep at least a few before training')
    # Below the per-family floor is a QUALITY guard-rail (the readiness blocker):
    # waivable by the explicit « Continue anyway » ack, refused otherwise.
    if kept < floor and not allow_not_ready:
        raise ValueError(
            f'NOT_READY: only {kept} kept image(s) — the minimum for a {label} LoRA '
            f'is {floor}. Continue anyway to train with too few (expect overfitting).')
    style = fds.is_style(ds_)
    missing = kept_uncaptioned_count(dataset_id)
    if missing and not allow_uncaptioned:
        policy = ('Style captions must describe the visible content of every image'
                  if style else 'Captions are strongly recommended')
        raise ValueError(
            f"UNCAPTIONED: {missing} kept image(s) have no caption (including whitespace). "
            f"{policy} — confirm explicitly to train anyway.")
    if style and not allow_caption_quality:
        quality = style_caption_quality(dataset_id)
        if quality['issues']:
            raise ValueError('CAPTION_QUALITY: ' + ' '.join(quality['issues']) +
                             ' Re-caption the dataset, or confirm explicitly to train anyway.')
    if allow_caption_mismatch:
        return
    # Garde-fou style ↔ type : un LoRA SDXL entraîné sur des captions PROSE = mismatch
    # booru-native → « images disjointes » (recherche 2026-06-14) ; et l'inverse pour Z-Image.
    # `ttype` a déjà été résolu en tête (plancher d'images) — on le réutilise.
    # `None` = famille HYBRIDE : les deux formes sont first-class, aucun mismatch
    # n'existe, le garde se tait (voir _EXPECTED_CAPTION_FORM).
    expected = _EXPECTED_CAPTION_FORM.get(ttype, 'prose')
    if expected is None:
        return
    from .face_variations import caption_style
    caps = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.caption.isnot(None)).all())
    sample = [c.caption for c in caps if c.caption and c.caption.strip()][:12]
    if sample:
        booru_n = sum(1 for s in sample if caption_style(s) == 'booru')
        actual = 'booru' if booru_n * 2 >= len(sample) else 'prose'   # vote majoritaire
        if actual != expected:
            if expected == 'booru':
                # A CONCEPT dataset cannot follow the usual advice. Its captions
                # come from _caption_concept, which has no booru variant and does
                # not even take a `mode`, and the prose/booru selector is hidden
                # on conceptual datasets — so "re-caption in 'Booru tags' mode"
                # names a mode the user cannot reach from anywhere. Sending
                # someone hunting for it is worse than the refusal itself.
                # (Style datasets are fine: caption_prompt_for_style IS
                # mode-aware, and an SDXL style dataset already defaults to
                # booru, so it never lands here.)
                if fds.is_concept(ds_):
                    raise ValueError(
                        "MISMATCH_CAPTION: this SDXL dataset has PROSE captions, but a booru "
                        "model (bigLove type) is prompted with tags. Concept captions are only "
                        "produced as prose today — there is no 'Booru tags' mode to switch to "
                        "on a concept dataset. Either train this concept on a prose family "
                        "(Z-Image, Krea 2, FLUX.1, FLUX.2 Klein, Anima), or force the training "
                        "and expect the quality loss a booru-native base takes from prose.")
                raise ValueError(
                    "MISMATCH_CAPTION: this SDXL dataset has PROSE captions, but a booru "
                    "model (bigLove type) is prompted with tags. Re-caption in 'Booru tags' mode "
                    "before training, or force the training.")
            # `label`, not a hard-coded "Z-Image": this branch fires for EVERY
            # prose family (Krea 2, FLUX.1, FLUX.2 Klein…), and telling a Klein
            # user they have a Z-Image dataset is the same disease as the anima
            # bug above — a claim frozen into a string where the truth is
            # per-family. `label` was already resolved for the image floor.
            raise ValueError(
                f"MISMATCH_CAPTION: this {label} dataset has booru TAG captions, but {label} "
                "expects prose. Re-caption in 'Prose' mode, or force the training.")


def _assert_no_vision_pass_on_gpu():
    """Refuse to start a local training while a vision pass owns the GPU.

    The mirror of gpu_exclusive_vision_window's own 'training is running' check.
    Until now this half only existed on the QUEUE path (_advance_training_queue
    skips a due item while `vision_in_progress`); a direct launch — the ▶ button,
    a retry, a continue — walked straight past it.

    It matters because the two do NOT share the card gracefully. Measured on a
    24 GB card with ~19 GB already resident: the vision model (7.5 GB with its
    context) no longer fits, Ollama silently spills ~43 % of it to the CPU, the
    vision pass runs ~13.5x slower and the resident GPU work drops 20-150x.
    Nothing raises and nothing OOMs — so a training started here would simply
    crawl for hours with no error to explain it. Refusing is the kinder failure.

    The persisted flag is TTL-bounded and cleared at startup. A process-local
    companion stays closed while a currently running Vision window is alive,
    including a transient heartbeat failure.
    """
    from ..gpu_window import GpuBusyError, vision_gpu_window_blocks_gpu
    try:
        active_vision_window = vision_gpu_window_blocks_gpu()
    except Exception as exc:
        raise GpuBusyError(
            'Could not confirm whether a vision task owns the GPU; training was not started safely.'
        ) from exc

    if active_vision_window or queue_manager._get_system_state('vision_in_progress', False):
        raise GpuBusyError(
            'a vision pass (captioning, watermark or framing) is using the GPU - '
            'training would fight it for VRAM instead of failing outright, so it '
            'has to wait. Stop the pass, or queue this dataset and it will start '
            'by itself when the pass is done.')

    if queue_manager.has_comfyui_stalled_barrier():
        raise GpuBusyError(
            'ComfyUI recovery is required before local training can take the GPU. '
            'Recover ComfyUI, cancel the paused Test Studio job, then resume it.')


def is_local_run_active(dataset_id) -> bool:
    """True when the single-flight LOCAL trainer is mid-run on THIS dataset.
    delete_dataset uses it (alongside cloud_training.active_runs_for) to refuse
    deleting a dataset whose training is still running."""
    if not queue_manager._get_system_state('training_in_progress', False):
        return False
    cur = queue_manager._get_system_state('training_dataset_id', None)
    return cur is not None and str(cur) == str(dataset_id)


def training_status(user_id=None) -> dict:
    cur_id = queue_manager._get_system_state('training_dataset_id', None)
    in_progress = bool(queue_manager._get_system_state('training_in_progress', False))
    cur_table = (queue_manager._get_system_state('training_dataset_table', None)
                 or _crd.FACE)
    current = None
    if in_progress and cur_id is not None and cur_table == _crd.VIDEO:
        # A video run. Resolved here rather than below because every line of the
        # face branch — the family, the variant, the base model, the Z-Image
        # recipe diagnostic — is a question about a `face_dataset` row that this
        # id does not name. Answering them from the colliding face row is how the
        # run ends up on the wrong page under the wrong name.
        from ..models import VideoDataset
        vds = VideoDataset.query.get(int(cur_id))
        current = {
            'dataset_id': cur_id,
            'dataset_table': cur_table,
            'name': vds.name if vds else f'video dataset {cur_id}',
            'run_token': queue_manager._get_system_state('training_run_token', None),
            'train_type': 'video',
            'target_profile': vds.target_profile if vds else None,
            'slider_mode': False,
            'variant': None,
            'base_model': None,
            'effective_base': None,
            'training_adapter': None,
            'recipe_version': None,
            'recipe_status': None,
            'recipe_warning': None,
        }
    elif in_progress and cur_id is not None:
        ds = FaceDataset.query.get(int(cur_id))
        fam = (queue_manager._get_system_state('training_train_type', None)
               or (_train_type(ds) if ds else None))
        variant = (queue_manager._get_system_state('training_variant', None)
                   or (getattr(ds, 'train_variant', None) if ds else None))
        base_model = queue_manager._get_system_state(
            'training_base_model', getattr(ds, 'train_base_model', None) if ds else None)
        effective_base = queue_manager._get_system_state(
            'training_effective_base', None)
        adapter = queue_manager._get_system_state(
            'training_training_adapter', None)
        recipe_version = queue_manager._get_system_state(
            'training_recipe_version', None)
        diag = zimage_recipe_diagnostic(
            fam, variant, effective_base, adapter, recipe_version)
        current = {
            'dataset_id': cur_id,
            'dataset_table': cur_table,
            'name': ds.name if ds else _dataset_name(cur_id),
            'run_token': queue_manager._get_system_state(
                'training_run_token', None),
            'train_type': fam,
            # Slider LoRA (Beta) run — the progress UI labels it honestly.
            'slider_mode': bool(queue_manager._get_system_state(
                'training_slider_mode', False)),
            'variant': variant,
            'base_model': base_model,
            'effective_base': effective_base,
            'training_adapter': adapter,
            'recipe_version': recipe_version,
            'recipe_status': diag.get('status') if diag else None,
            'recipe_warning': diag.get('warning') if diag else None,
        }
    return {'in_progress': in_progress,
            'installed': is_installed(),
            'pid': queue_manager._get_system_state('training_pid', None),
            'current': current,
            # Dernier crash d'entraînement (rc≠0) remonté par le watcher, pour l'UI.
            'error': queue_manager._get_system_state('training_error', None),
            'queue': train_queue_view(user_id) if user_id is not None else []}


# --- Retry d'un run LOCAL raté (page Runs) --------------------------------------
# Local runs carry no status column: their launch is recorded once in the
# provenance registry (TrainingRunRecord, source='local'), and the ONLY signal
# that one crashed is the transient global `training_error` the watcher writes on
# rc≠0 — cleared on the next launch, TTL-capped. Local training is single-flight,
# so at most one local run is "failed" at a time: the newest local record of the
# dataset the last crash points at. This mirrors the cloud ↻ Retry, which replays
# the stamped launch params (CloudTrainingRun.train_params) on a fresh pod.

def last_local_error() -> dict | None:
    """The last local-training crash, as {dataset_id, rc, log_tail}, or None."""
    err = queue_manager._get_system_state('training_error', None)
    return err if isinstance(err, dict) else None


def local_error_message(err) -> str:
    """One-line, paste-safe summary of a local crash for the Runs page. Quotes the
    line that EXPLAINS the crash (last traceback / error line), never the last
    line of the log — which is very often a harmless FutureWarning. When the log
    holds no error line at all, the summary stays honest and says just that."""
    if not isinstance(err, dict):
        return 'Training crashed.'
    from .training_diagnostics import extract_error_excerpt
    rc = err.get('rc')
    excerpt = err.get('excerpt')
    if not isinstance(excerpt, dict):     # legacy payload written before excerpts
        excerpt = extract_error_excerpt(err.get('log_tail') or '')
    headline = (excerpt.get('headline') or '').strip()[:300]
    base = f'Training crashed (exit code {rc}).' if rc is not None else 'Training crashed.'
    return f'{base} {headline}' if headline else f'{base} No error line in the log.'


def _failed_local_record_id() -> int | None:
    """Registry id of the local run the last crash belongs to, or None. Pure
    error→record mapping (ignores whether a NEW run is now in progress — that
    refusal belongs to launch_training's collision guard)."""
    err = last_local_error()
    if not err or err.get('dataset_id') is None:
        return None
    from ..models import TrainingRunRecord
    rec = (TrainingRunRecord.query
           .filter_by(dataset_id=int(err['dataset_id']), source='local')
           .order_by(TrainingRunRecord.id.desc()).first())
    return rec.id if rec else None


def failed_local_run() -> tuple | None:
    """(record_id, message) of the local run the Runs page should offer a ↻ Retry
    for, or None. None while a run is in progress: its launch cleared the error
    state, so nothing is "failed" during a live run."""
    if queue_manager._get_system_state('training_in_progress', False):
        return None
    rec_id = _failed_local_record_id()
    if rec_id is None:
        return None
    return rec_id, local_error_message(last_local_error())


# The confirmable pre-flight refusals a launch can be waved through, and the ONE
# list the local retry lane forwards. Every one of them is a refusal the Start
# flow already lets the user answer; a retry that could not answer them was a
# dead button (GitHub #23). Mirrors cloud_training._CONFIRMATION_FLAGS — the
# cloud lane replays them from the run's stored pod params, the local lane asks
# again (its record stores no consent, and the dataset it re-exports is live).
CONFIRMATION_FLAGS = (
    'allow_caption_mismatch',
    'allow_uncaptioned',
    'allow_caption_quality',
    'allow_unverified_weights',
    'allow_not_ready',
)


def retry_local_run(user_id, record_id, **confirmations) -> dict:
    """↻ Retry a FAILED local run: a REAL launch_training replaying the identity
    params stamped for that launch (family / variant / base / masked / steps) —
    same guardrails as any launch (GPU-collision refusal, normal preflight, no
    bypass), not a resurrection of a dead process. The live dataset (images,
    captions, advanced + slider settings) is the source of truth and is replayed
    as-is, so a slider run re-emits its slider recipe — now with the 768-only
    default that keeps its VRAM peak under 24 GB.

    ``**confirmations`` = the CONFIRMATION_FLAGS the caller answered for THIS
    retry (all False by default). They are not read back from the failed record:
    the record stores no consent, and the dataset being re-exported is the live
    one, so an answer given at the first launch describes a dataset that may no
    longer exist. Unknown keys are refused rather than silently dropped — a
    typo'd flag name must not read as "the user did not confirm"."""
    unknown = set(confirmations) - set(CONFIRMATION_FLAGS)
    if unknown:
        raise ValueError(f'unknown confirmation flag(s): {", ".join(sorted(unknown))}')
    from ..models import TrainingRunRecord
    rec = fds.db.session.get(TrainingRunRecord, int(record_id))
    if rec is None:
        raise ValueError('unknown training run')
    if rec.source != 'local':
        raise ValueError('only a local run can be retried here')
    # Run-in-progress → the exact collision message launch_training would raise,
    # surfaced before any preflight work.
    if (queue_manager._get_system_state('training_in_progress', False)
            and not _training_process_is_definitely_dead(
                queue_manager._get_system_state('training_pid', None))):
        raise ValueError('a training is already in progress - wait for it to finish or queue this dataset')
    if _failed_local_record_id() != rec.id:
        raise ValueError('this run has no recorded failure to retry')
    return launch_training(
        user_id, rec.dataset_id, steps=rec.steps,
        base_model=(rec.base_model or None), variant=rec.variant,
        train_type=rec.family, masked=bool(rec.masked), training_mode='lora',
        **{k: bool(confirmations.get(k)) for k in CONFIRMATION_FLAGS})


# --- Suivi de progression (log tail + loss curve + samples) -------------------
# ai-toolkit redirige tqdm dans training.log : les mises à jour sont séparées par
# des \r sur une même « ligne », d'où le split sur [\r\n]. Un segment type :
#   lora_x:   2%|▏| 60/3000 [01:23<1:07:41, 1.38s/it, lr: 1.0e+00 loss: 3.412e-01]
_PROG_STEP_RE = re.compile(r'(\d+)/(\d+)')
_PROG_LOSS_RE = re.compile(r'loss[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)')
_PROG_SPEED_RE = re.compile(r'([\d.]+\s*(?:s/it|it/s))')
_PROG_ETA_RE = re.compile(r'<\s*([\d:]+)\s*,')
_SAMPLE_RE = re.compile(r'__(\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$', re.IGNORECASE)
_PROG_LOG_MAX_BYTES = 4 * 1024 * 1024   # tail cap: 3000 tqdm updates ≈ 0.5 MB
_PROG_CURVE_MAX_POINTS = 200
_PROG_SAMPLES_MAX = 24

# The SAME log also carries huggingface_hub's byte-counter bars while the pod
# pulls the base weights — the phase that costs the most and reported the least:
#   raw.safetensors:   7%|▋| 1.95G/26.3G [15:30<2:37:06, 2.58MB/s]
# Two independent reasons to recognise them.
#  * They are not steps. '0.00/26.3G' matched _PROG_STEP_RE as "0/26", so a run
#    downloading its base model showed 'step 0 / 26' — a plausible-looking
#    number that was pure noise (verified on the run-121 log, 2026-07-28).
#  * They ARE the answer to "is it downloading or is it frozen?", which nothing
#    surfaced: two users waited hours in front of a fixed sentence.
# The discriminator is the rate unit: tqdm prints B/s only for byte bars
# (it/s or s/it for step bars). '?B/s' (no estimate yet) counts too.
_DOWNLOAD_PROG_RE = re.compile(
    r'(?P<label>[^\r\n|]{1,80}?):\s*(?P<percent>\d{1,3})%\|[^|]*\|\s*'
    r'(?P<done>[\d.]+\s*[kKMGTP]?)/(?P<total>[\d.]+\s*[kKMGTP]?)\s*'
    r'\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*(?P<speed>[\d.?]+\s*[kKMGTP]?B/s)\s*\]')
_DOWNLOAD_RATE_RE = re.compile(r'[\d.?]\s*[kKMGTP]?B/s')


def _parse_training_log(text: str) -> dict:
    """Extract (step, total, loss, speed, eta, loss_curve) from raw log text.
    Pure function — unit-testable without a real run."""
    out = {'step': None, 'total': None, 'loss': None, 'speed': None, 'eta': None,
           'loss_curve': []}
    curve = []
    for seg in re.split(r'[\r\n]+', text):
        lm = _PROG_LOSS_RE.search(seg)
        # Only trust real tqdm segments ('%|' bar or a loss postfix) — the log also
        # contains incidental 'X/Y' text (dataset counts, resolutions) that must not
        # be read as progress.
        if '%|' not in seg and not lm:
            continue
        # A byte bar is a download, not a training step: '0.00/26.3G' read as
        # step 0 of 26 is worse than no number at all.
        if _DOWNLOAD_RATE_RE.search(seg):
            continue
        sm = None
        for sm in _PROG_STEP_RE.finditer(seg):
            pass                             # last step/total occurrence of the segment
        if not sm:
            continue
        step, total = int(sm.group(1)), int(sm.group(2))
        if total <= 0 or step > total:
            continue                         # e.g. '1024x1024' image sizes, not progress
        out['step'], out['total'] = step, total
        if lm:
            try:
                loss = float(lm.group(1))
            except ValueError:
                continue
            out['loss'] = loss
            if not curve or curve[-1][0] != step:
                curve.append([step, loss])
        spm = _PROG_SPEED_RE.search(seg)
        if spm:
            out['speed'] = spm.group(1).strip()
        em = _PROG_ETA_RE.search(seg)
        if em:
            out['eta'] = em.group(1)
    # Downsample evenly so the payload stays small on long runs.
    if len(curve) > _PROG_CURVE_MAX_POINTS:
        stride = len(curve) / _PROG_CURVE_MAX_POINTS
        curve = [curve[int(i * stride)] for i in range(_PROG_CURVE_MAX_POINTS - 1)] + [curve[-1]]
    out['loss_curve'] = curve
    return out


def parse_download_progress(text: str) -> dict | None:
    """The LAST byte-counter bar in the log, or None when there is none.

    Pure function. Returns the figures exactly as tqdm printed them — strings,
    not converted numbers: '1.95G' is what the log says, and inventing
    1.95 × 1024³ bytes from it would be a number the log never contained (the
    unit divisor is the producer's choice, not ours). The caller displays them
    and compares `done` for movement; neither needs a conversion.

    Degrades on purpose: this format belongs to huggingface_hub/tqdm and can
    change or be absent (some phases print no bar at all). Anything that does
    not match EVERY field is not guessed at — it yields None, and the caller
    keeps whatever it was already showing."""
    last = None
    for seg in re.split(r'[\r\n]+', text or ''):
        m = _DOWNLOAD_PROG_RE.search(seg)
        if m:
            last = m
    if last is None:
        return None
    label = last.group('label').strip()
    # tqdm re-prints the bar with a bumped elapsed even while the byte counter
    # is frozen (measured on the run-121 log: 1.95G at 15:11, then at 15:30).
    # 'done' is therefore the only field that means "it moved".
    return {'label': label[-60:] or 'download',
            'percent': min(100, int(last.group('percent'))),
            'done': last.group('done').strip(),
            'total': last.group('total').strip(),
            'elapsed': last.group('elapsed'),
            'eta': None if '?' in last.group('eta') else last.group('eta'),
            'speed': (None if '?' in last.group('speed')
                      else last.group('speed').strip())}


_DOWNLOAD_UNIT = {'': 1.0, 'k': 1e3, 'K': 1e3, 'M': 1e6,
                  'G': 1e9, 'T': 1e12, 'P': 1e15}


def download_bytes_seen(text: str) -> float | None:
    """Total bytes downloaded across EVERY bar in the log, or None when the log
    holds no download bar at all. Same regex, same log, different question from
    parse_download_progress(): that one answers "what do I SHOW the user" and
    therefore reports the last bar verbatim; this one answers "did the pod
    MOVE", which no single bar can answer.

    Why a sum and not the last bar: huggingface_hub fetches several files at
    once, so consecutive tails end on DIFFERENT bars. A watchdog comparing the
    last bar alone would read A(1.0G), B(2.0G), A(1.0G)… as endless movement
    while both files sit frozen — the exact false progress a kill decision must
    never be built on. Keyed by label, summed, so only a file that genuinely
    advanced can raise the total.

    The unit divisor is assumed decimal. That is a guess about a producer's
    formatting choice (see parse_download_progress), so the result is used for
    two things only — a `>` comparison against the previous poll, and a rounded
    human label in a failure message — never as an exact byte count."""
    per_label = {}
    for seg in re.split(r'[\r\n]+', text or ''):
        for m in _DOWNLOAD_PROG_RE.finditer(seg):
            raw = m.group('done').strip()
            unit = raw[-1] if raw and raw[-1] in _DOWNLOAD_UNIT else ''
            try:
                value = float(raw[:-1] if unit else raw) * _DOWNLOAD_UNIT[unit]
            except ValueError:
                continue                      # not a number we can compare
            per_label[m.group('label').strip()] = value
    return sum(per_label.values()) if per_label else None


def _samples_dir(user_id, dataset_id, base_model=_PERSISTED, family=None,
                 variant=_PERSISTED) -> str:
    return os.path.join(
        _run_dir(user_id, dataset_id, base_model, family, variant), 'samples')


def list_training_samples(user_id, dataset_id, base_model=_PERSISTED, family=None,
                          limit=_PROG_SAMPLES_MAX,
                          variant=_PERSISTED) -> list[dict]:
    """Sample previews ai-toolkit writes every sample_every steps
    (<run>/samples/<ts>__<step>_<promptidx>.jpg). Newest steps first, capped
    (limit=None → all, for the best-epoch scoring pass)."""
    d = _samples_dir(user_id, dataset_id, base_model, family, variant)
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        m = _SAMPLE_RE.search(f)
        if m:
            out.append({'filename': f, 'step': int(m.group(1)), 'prompt_idx': int(m.group(2))})
    out.sort(key=lambda s: (-s['step'], s['prompt_idx']))
    return out if limit is None else out[:limit]


def score_checkpoint_samples(user_id, dataset_id, base_model=_PERSISTED, family=None,
                             variant=_PERSISTED) -> dict:
    """Best-epoch selection (jandordoe method): every training sample is an output
    of the LoRA at its step — scoring their face similarity vs the dataset
    reference (insightface, CPU, one subprocess for the whole set) tells which
    step holds the identity best. The recommended checkpoint is the saved one
    closest to that step.

    Returns {'available': bool, 'reason'?: str, 'steps': [{'step','mean_sim','n'}],
    'best_step': int|None, 'checkpoint': str|None} — never raises on missing
    prerequisites, the UI shows `reason` instead."""
    from . import face_similarity
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Same InsightFace lane as the dataset pass -> same single rule. Ranking epochs
    # by a similarity the model cannot measure would recommend a checkpoint at
    # random while looking authoritative; `reason` is already what the UI shows.
    blocked = fds.face_scoring_block_reason(ds)
    if blocked:
        return {'available': False, 'reason': blocked}
    if not ds.ref_filename:
        return {'available': False, 'reason': 'this dataset has no reference photo'}
    ref_path = os.path.join(fds._dataset_dir(ds.id), ds.ref_filename)
    if not face_similarity.is_available():
        return {'available': False,
                'reason': 'face scoring is not installed (Quality tools step in Setup)'}
    samples = list_training_samples(
        user_id, dataset_id, base_model, family, limit=None, variant=variant)
    if not samples:
        return {'available': False, 'reason': 'no training samples yet (they appear every 250 steps)'}
    sdir = _samples_dir(user_id, dataset_id, base_model, family, variant)
    paths = [os.path.join(sdir, s['filename']) for s in samples]
    results, scoring_error = face_similarity.score_dataset_faces(ref_path, paths)
    if not results:
        detail = (scoring_error or {}).get('detail')
        return {'available': False,
                'reason': f'face scoring failed: {detail}' if detail
                else 'face scoring failed (see server log)'}
    by_step = {}
    for s, p in zip(samples, paths):
        r = results.get(p)
        if r and r.get('state') == 'scorable' and r.get('sim') is not None:
            by_step.setdefault(s['step'], []).append(float(r['sim']))
    steps = [{'step': st, 'mean_sim': round(sum(v) / len(v), 4), 'n': len(v)}
             for st, v in sorted(by_step.items())]
    if not steps:
        return {'available': False, 'reason': 'no scorable face in the samples'}
    best = max(steps, key=lambda s: s['mean_sim'])
    # Map the winning sample step to the CLOSEST saved checkpoint (samples every
    # 250 steps, checkpoints every 500 — they rarely align exactly).
    cks = list_checkpoints(user_id, dataset_id, base_model, family, variant)
    ck = min(cks, key=lambda c: abs(c['step'] - best['step']))['filename'] if cks else None
    return {'available': True, 'steps': steps, 'best_step': best['step'], 'checkpoint': ck}


def training_progress(user_id, dataset_id, base_model=_PERSISTED, family=None,
                      variant=_PERSISTED) -> dict:
    """Live view of a run: parsed log progress + sample listing. Never raises on a
    missing/unreadable log (a run that hasn't started writing yet is normal) —
    only on an unknown dataset (route → 404 via get_dataset)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur_id = queue_manager._get_system_state('training_dataset_id', None)
    active = (bool(queue_manager._get_system_state('training_in_progress', False))
              and cur_id is not None and int(cur_id) == int(dataset_id)
              and not _training_process_is_definitely_dead(
                  queue_manager._get_system_state('training_pid', None)))
    log_path = _run_log_path(ds, base_model, family, variant)
    parsed = {'step': None, 'total': None, 'loss': None, 'speed': None, 'eta': None,
              'loss_curve': []}
    # A local run downloads its base weights too (the first run of a family
    # pulls tens of GB from Hugging Face), and showed the same motionless
    # 'Starting up...' the cloud card showed. Same parser, same degradation.
    download = None
    log_exists = os.path.isfile(log_path)
    if log_exists:
        try:
            size = os.path.getsize(log_path)
            with open(log_path, encoding='utf-8', errors='replace') as fh:
                if size > _PROG_LOG_MAX_BYTES:
                    fh.seek(size - _PROG_LOG_MAX_BYTES)
                text = fh.read()
            parsed = _parse_training_log(text)
            download = parse_download_progress(text)
        except OSError:
            log_exists = False
    return {'active': active, 'log_exists': log_exists, **parsed,
            'download': download,
            'masks_skipped': bool(active and queue_manager._get_system_state('training_masks_skipped', False)),
            'samples': list_training_samples(
                user_id, dataset_id, base_model, family, variant=variant)}


# --- File d'attente d'entraînement -------------------------------------------
TRAIN_QUEUE_KEY = 'lora_train_queue'

# Sérialise TOUS les read-modify-write de la file dans ce process. Le verrou est
# réentrant pour que l'avancement de file puisse continuer à appeler des helpers
# de queue sans risque de deadlock. Les preflights d'enqueue restent volontairement
# hors du verrou : seule la courte transaction duplicate-check/read/write est
# critique.
_queue_lock = threading.RLock()

_TRAIN_IDENTITY_KEYS = (
    'training_pid', 'training_pid_create_time', 'training_dataset_id',
    # WHICH TABLE `training_dataset_id` POINTS INTO. Absent means `face_dataset`,
    # the only meaning it could have had before the video lane existed. Without
    # it the fence is one integer shared by two tables: face dataset #3 and video
    # dataset #3 both exist, so a video run would show up under a face dataset's
    # name and be killed by that dataset's Stop button.
    'training_dataset_table',
    'training_target_step',
    'training_run_token', 'training_train_type', 'training_variant',
    'training_base_model', 'training_effective_base', 'training_slider_mode',
    'training_training_adapter', 'training_recipe_version',
)


def _clear_training_identity(ttl_seconds=None) -> None:
    for key in _TRAIN_IDENTITY_KEYS:
        queue_manager._set_system_state(key, None, ttl_seconds=ttl_seconds)
    queue_manager._set_system_state(
        'training_in_progress', False, ttl_seconds=ttl_seconds)


def _training_process_create_time(pid) -> float | None:
    """Read a process birth time without ever treating an error as a death."""
    try:
        import psutil
        return float(psutil.Process(int(pid)).create_time())
    except Exception as exc:
        logger.warning('Could not capture training process identity for pid %r: %s', pid, exc)
        return None


def _record_training_process_identity(pid) -> float | None:
    """Persist a PID plus birth time so a later PID reuse cannot be killed."""
    queue_manager._set_system_state(
        'training_pid', int(pid), ttl_seconds=_TRAIN_STATE_TTL)
    birth_time = _training_process_create_time(pid)
    if birth_time is None:
        logger.error(
            'training pid %s has no durable birth-time identity; keeping its GPU '
            'fence fail-closed after a restart', pid)
        return None
    queue_manager._set_system_state(
        'training_pid_create_time', birth_time, ttl_seconds=_TRAIN_STATE_TTL)
    return birth_time


def _pid_alive_with_birth(pid, expected_raw) -> bool | None:
    """Return True (exact training child), False (confirmed old child gone), or None.

    A persisted PID alone is not safe after Flask restarts because Windows can
    reuse it. The stored process creation time turns a reused PID into a
    confirmed dead old training identity, never a process to terminate.
    """
    if not pid:
        return None
    try:
        import psutil
        process = psutil.Process(int(pid))
        current_birth_time = float(process.create_time())
    except Exception as exc:
        # psutil.NoSuchProcess is the one reliable proof that this exact PID is
        # gone. Everything else (access denied, bad state, import failure) keeps
        # the durable GPU fence in place.
        try:
            import psutil
            no_such_process = psutil.NoSuchProcess
        except Exception:
            no_such_process = ()
        if no_such_process and isinstance(exc, no_such_process):
            return False
        logger.warning('Could not inspect training pid %r: %s', pid, exc)
        return None

    try:
        expected_birth_time = float(expected_raw)
    except (TypeError, ValueError):
        logger.error(
            'training pid %s lacks a durable birth-time identity; keeping the GPU '
            'fence fail-closed', pid)
        return None

    if abs(current_birth_time - expected_birth_time) > 0.01:
        logger.warning(
            'training pid %s was reused (expected birth %.6f, got %.6f); '
            'the old training is confirmed gone and will never be taskkilled',
            pid, expected_birth_time, current_birth_time)
        return False
    return True


def _pid_alive(pid) -> bool | None:
    expected_raw = queue_manager._get_system_state(
        'training_pid_create_time', None)
    return _pid_alive_with_birth(pid, expected_raw)


def _training_process_is_definitely_dead(pid) -> bool:
    """Only a trustworthy negative PID probe may release the GPU fence."""
    return _pid_alive(pid) is False


def _wait_for_training_process_exit(pid, timeout_seconds=5.0) -> bool:
    """Bounded proof that a successful kill actually released this PID."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        if _training_process_is_definitely_dead(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def get_train_queue() -> list:
    q = queue_manager._get_system_state(TRAIN_QUEUE_KEY, [])
    return q if isinstance(q, list) else []


def _save_queue(q: list) -> None:
    queue_manager._set_system_state(TRAIN_QUEUE_KEY, q, ttl_seconds=None)


def _queued_training_mode(item: dict) -> str:
    """Frozen queue mode; pre-feature or damaged rows stay LoRA-safe."""
    value = item.get('training_mode', 'lora')
    return value if value in TRAINING_MODES else 'lora'


def enqueue_training(user_id, dataset_id, extra_steps=None,
                     base_model=_PERSISTED, variant=None, train_type=None,
                     allow_caption_mismatch=False, not_before=None, masked=None,
                     steps=None, allow_uncaptioned=False,
                     allow_caption_quality=False,
                     vae_path=_PERSISTED, te_path=_PERSISTED,
                     allow_unverified_weights=False, allow_not_ready=False,
                     training_mode=None) -> dict:
    """Ajoute un dataset à la file (lancé à la fin du training courant).

    `base_model`/`variant` permettent de CHOISIR explicitement la base du job en
    file (absent → base persistée). Sans ça, on ne pouvait pas choisir le modèle
    d'un job mis en file pendant qu'un autre entraînement tourne (le sélecteur
    était masqué et l'enqueue réutilisait silencieusement la base persistée).

    `steps` = cible ABSOLUE de steps pour un lancement neuf (None → adaptatif via
    recommended_steps). À NE PAS confondre avec `extra_steps` (mode « continuer »
    = +N steps depuis le dernier checkpoint). Snapshotté dans la file pour que le
    lancement différé respecte le même plafond (ex. « s'arrêter à 2000 »)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    mode = _assert_local_training_mode(ds, training_mode)
    # Every queued job re-exports the CURRENT dataset, including +N checkpoint
    # resumes. Validate now and again when the item reaches the GPU.
    assert_trainable(dataset_id, train_type=train_type,
                     allow_caption_mismatch=allow_caption_mismatch,
                     allow_uncaptioned=allow_uncaptioned,
                     allow_caption_quality=allow_caption_quality,
                     allow_not_ready=allow_not_ready,
                     variant=variant)
    if train_type is not None:
        ds.train_type = train_type
        fds.db.session.commit()
    ttype = _train_type(ds)
    base = (ds.train_base_model if base_model is _PERSISTED else base_model) or None
    var = str(variant or ds.train_variant or _default_variant_for(ttype)).strip().lower()
    recipe = None
    if ttype == 'zimage':
        # Reject a bad recipe now, not hours later when the queued item reaches
        # the GPU.  launch_training validates again as the authoritative gate.
        recipe = zimage_training_recipe(var, base)
        var = recipe['variant']
    elif var not in _valid_variants_for(ttype):
        var = _default_variant_for(ttype)
    queue_view = _train_context_view(
        ds, ttype, var, base_model=base, training_mode=mode)
    assert_zimage_custom_recipe_confirmed(
        ttype, base, var,
        allow_unverified_weights=allow_unverified_weights)
    # Base custom (merge) Z-Image = doit être convertie AVANT (SDXL = single-file
    # direct, pas de conversion → on saute la vérif). Refus immédiat et lisible.
    if extra_steps is None and base and ttype == 'zimage':
        from .zimage_convert import is_converted
        if not is_converted(base):
            raise ValueError('custom base not converted - prepare it first (button "Convert base")')
    # SDXL : whitelist serveur de la base (anti path-traversal). Un chemin ABSOLU
    # = « Custom weights… » (validé par le preflight) → contourne la whitelist.
    if base and ttype == 'sdxl' and not _is_custom_weights(base) and base not in _sdxl_base_choices():
        raise ValueError('unknown SDXL checkpoint')
    # Custom vae/te : whitelist STRICTE par famille (SDXL-only), persistance et
    # preflight — même contrat qu'au lancement, pour ne pas mettre en file un job
    # voué à un refus 400 (ou à un chemin fantôme) au moment de son démarrage.
    _q_prov_vae = vae_path is not _PERSISTED and (vae_path or '').strip()
    _q_prov_te = te_path is not _PERSISTED and (te_path or '').strip()
    if ttype not in VAE_TE_OVERRIDE_FAMILIES:
        if _q_prov_vae or _q_prov_te:
            raise ValueError('VAE / text-encoder overrides are SDXL-only')
        eff_vae = eff_te = None
    else:
        eff_vae = (ds.train_vae_path if vae_path is _PERSISTED
                   else ((vae_path or '').strip() or None))
        eff_te = (ds.train_te_path if te_path is _PERSISTED
                  else ((te_path or '').strip() or None))
    if extra_steps is None:
        preflight_custom_paths(ttype, weights=base, vae_path=eff_vae, te_path=eff_te,
                               allow_unverified_weights=allow_unverified_weights)
    # Persist vae/te so the deferred launch (and the continue path) read the same
    # triplet the run-dir tag was computed with.
    ds.train_vae_path = eff_vae
    ds.train_te_path = eff_te
    fds.db.session.commit()
    # Krea 2 : même garde qu'au lancement - pas de mise en file d'un job qui
    # tomberait dans le fallback SD legacy faute d'arch krea2 dans l'ai-toolkit.
    if ttype == 'krea' and not _aitoolkit_supports_krea():
        raise ValueError(
            "ai-toolkit doesn't support Krea 2 yet (krea2 arch missing) - "
            "update it (git pull) before queuing a Krea LoRA.")
    # FLUX.2 Klein : même garde qu'au lancement (archs d'extension, cf. launch).
    if ttype == 'flux2klein' and not _aitoolkit_supports_flux2klein():
        raise ValueError(
            "ai-toolkit doesn't support FLUX.2 Klein yet (flux2_klein arch missing) - "
            "update it (git pull) before queuing a FLUX.2 Klein LoRA.")
    # Anima : même garde qu'au lancement (arch d'extension, PR #860).
    if ttype == 'anima' and not _aitoolkit_supports_anima():
        raise ValueError(
            "ai-toolkit doesn't support Anima yet (anima arch missing) - "
            "update it (git pull) before queuing an Anima LoRA.")
    if (_optimizer_eff(queue_view) == 'automagic3'
            and not _aitoolkit_supports_automagic3()):
        raise ValueError(
            "ai-toolkit doesn't support Automagic3 yet - update it (git pull) "
            "or choose another optimizer before queuing.")
    # Même garde-fou de collision qu'au lancement : pas de mise en file d'un job
    # qui partagerait le dossier de run d'un autre dataset (même trigger + base + recette).
    clash = find_run_collision(user_id, dataset_id, base_model=base, variant=var)
    if clash:
        raise ValueError(f"training collision with '{clash.name}' (#{clash.id}): "
                         f"same trigger + same base. Change the trigger_word before queuing.")
    # Snapshot de la base/variante/type CHOISIE au moment de la mise en file (le
    # lancement différé doit garder CE choix, pas relancer sur l'officiel/zimage).
    # `not_before` (ISO, heure locale serveur) = entraînement PROGRAMMÉ : le job
    # reste en file jusqu'à l'échéance ; s'il devient dû pendant qu'un autre
    # entraînement tourne, il attend simplement son tour (jamais d'erreur).
    # Cible de steps ABSOLUE (plafond choisi côté UI) - coercition défensive : un
    # '' / 0 / non-numérique retombe sur None (= adaptatif), jamais de crash JSON.
    try:
        steps_target = int(steps) if steps else None
    except (TypeError, ValueError):
        steps_target = None
    item = {'dataset_id': int(dataset_id), 'user_id': str(user_id), 'extra_steps': extra_steps,
            'base_model': base, 'variant': var, 'train_type': ttype,
            # Execution mode is a queued-run fact.  A later panel save must not
            # reinterpret an already planned LoRA as dense training.
            'training_mode': mode,
            # Resolved HERE, at enqueue: the queue item freezes what the user saw
            # when they queued it (like base/variant/steps just above). `None` =
            # no explicit request → the dataset's stored setting.
            'not_before': not_before, 'masked': resolve_masked(ds, masked),
            'steps': steps_target,
            # SDXL custom overrides ride along so the deferred launch reproduces
            # the exact triplet (they're also persisted on ds above).
            'vae_path': eff_vae, 'te_path': eff_te,
            # Confirmation flags must survive the wait; launch re-runs the same
            # authoritative guards when the queued item reaches the GPU.
            'allow_caption_mismatch': bool(allow_caption_mismatch),
            'allow_uncaptioned': bool(allow_uncaptioned),
            'allow_caption_quality': bool(allow_caption_quality),
            'allow_unverified_weights': bool(allow_unverified_weights),
            # « Continue anyway » ack survives the wait; the launch re-runs the
            # authoritative floor guard when the queued item reaches the GPU.
            'allow_not_ready': bool(allow_not_ready)}
    if recipe:
        item.update({'recipe_version': recipe['recipe_version'],
                     'effective_base': recipe['effective_base'],
                     'training_adapter': recipe['training_adapter']})
    # Ne verrouiller qu'après tous les preflights potentiellement coûteux. La
    # lecture, le contrôle anti-doublon et l'écriture doivent former UNE opération
    # atomique, sinon deux requêtes concurrentes peuvent perdre un item ou accepter
    # deux fois le même dataset.
    with _queue_lock:
        q = get_train_queue()
        if any(int(it.get('dataset_id', -1)) == int(dataset_id) for it in q):
            return {'queued': False, 'reason': 'already queued'}
        q.append(item)
        _save_queue(q)
        position = len(q)
    return {'queued': True, 'position': position, 'not_before': not_before}


def dequeue_training(dataset_id) -> int:
    # Même transaction atomique que l'enqueue : sans le verrou, deux suppressions
    # simultanées peuvent chacune réécrire leur ancien snapshot et ressusciter
    # l'item supprimé par l'autre requête.
    with _queue_lock:
        q = get_train_queue()
        new = [it for it in q if int(it.get('dataset_id', -1)) != int(dataset_id)]
        _save_queue(new)
        return len(q) - len(new)


def train_queue_view(user_id) -> list:
    out = []
    for it in get_train_queue():
        ds = fds.get_dataset(it.get('user_id', user_id), it.get('dataset_id'))
        bm = it.get('base_model')
        base_label = (os.path.basename(str(bm).replace('\\', '/')).rsplit('.', 1)[0]
                      if bm else 'Official')
        out.append({'dataset_id': it.get('dataset_id'),
                    'name': ds.name if ds else f"#{it.get('dataset_id')}",
                    'extra_steps': it.get('extra_steps'),
                    # Cible de steps absolue choisie à la mise en file (None = adaptatif).
                    'steps': it.get('steps'),
                    'base_model': bm, 'base_label': base_label,
                    'train_type': it.get('train_type'),
                    'variant': it.get('variant'),
                    'training_mode': _queued_training_mode(it),
                    'recipe_version': it.get('recipe_version'),
                    'effective_base': it.get('effective_base'),
                    'training_adapter': it.get('training_adapter'),
                    # Échéance de programmation (ISO local) - None = dès que possible.
                    'not_before': it.get('not_before')})
    return out


def _launch_queued_item(item) -> None:
    ds_id = item['dataset_id']
    uid = item.get('user_id')
    extra = item.get('extra_steps')
    # Queue rows written before training modes existed are historical LoRA
    # plans.  Missing/corrupt state therefore degrades to LoRA, never dense.
    mode = _queued_training_mode(item)
    if extra:
        continue_training(
            uid, ds_id, extra_steps=extra,
            base_model=item.get('base_model'), variant=item.get('variant'),
            train_type=item.get('train_type'),
            masked=item.get('masked', True),
            allow_caption_mismatch=bool(
                item.get('allow_caption_mismatch')),
            allow_uncaptioned=bool(item.get('allow_uncaptioned')),
            allow_caption_quality=bool(
                item.get('allow_caption_quality')),
            allow_not_ready=bool(item.get('allow_not_ready')),
            allow_unverified_weights=bool(
                item.get('allow_unverified_weights')),
            training_mode=mode,
            # _advance_training_queue deliberately keeps the dead predecessor's
            # flag raised until the next PID is published (no ComfyUI GPU flap).
            _allow_dead_predecessor=True)
    else:
        launch_training(uid, ds_id, steps=item.get('steps'),
                        base_model=item.get('base_model'),
                        # None → launch_training applique le défaut family-aware (Krea → Raw).
                        variant=item.get('variant'),
                        train_type=item.get('train_type'),
                        masked=item.get('masked', True),
                        allow_caption_mismatch=bool(item.get('allow_caption_mismatch')),
                        allow_uncaptioned=bool(item.get('allow_uncaptioned')),
                        allow_caption_quality=bool(item.get('allow_caption_quality')),
                        allow_not_ready=bool(item.get('allow_not_ready')),
                        # SDXL custom overrides snapshotted at enqueue time; the file
                        # was already preflighted, so re-clear the confirmable gate.
                        vae_path=item.get('vae_path', _PERSISTED),
                        te_path=item.get('te_path', _PERSISTED),
                        allow_unverified_weights=bool(item.get('allow_unverified_weights')),
                        training_mode=mode)


def _journal_path_is_unsafe(path: Path) -> bool:
    """Reject links/reparses/non-directories in every existing ancestor."""
    output = _absolute_lexical(_output_dir())
    target = _absolute_lexical(path)
    try:
        if os.path.normcase(os.path.commonpath((output, target))) != os.path.normcase(
                str(output)):
            return True
        relative = target.relative_to(output)
    except (ValueError, OSError):
        return True
    current = output
    candidates = [current]
    for part in relative.parts:
        current = current / part
        candidates.append(current)
    for candidate in candidates:
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            # Missing descendants are not links. Every existing ancestor was
            # checked before reaching the first absent component.
            return False
        if (
            stat.S_ISLNK(info.st_mode)
            or (getattr(info, 'st_file_attributes', 0) & _REPARSE_POINT)
            or not stat.S_ISDIR(info.st_mode)
        ):
            return True
    return False


def _rearm_exact_resume_fence(value, pid=None, birth_time=None) -> None:
    queue_manager._set_system_state(
        'training_in_progress', True, ttl_seconds=_TRAIN_STATE_TTL)
    facts = {
        'training_dataset_id': value.get('dataset_id'),
        'training_run_token': value.get('run_token'),
        'training_pid': pid,
        'training_pid_create_time': birth_time,
    }
    for key, fact in facts.items():
        if fact is not None:
            queue_manager._set_system_state(
                key, fact, ttl_seconds=_TRAIN_STATE_TTL)


def _matching_journal_process_identity(value) -> tuple[object, object]:
    pid = value.get('pid')
    birth_time = value.get('pid_create_time')
    if pid is not None and birth_time is not None:
        return pid, birth_time
    if not queue_manager._get_system_state('training_in_progress', False):
        return pid, birth_time
    persisted_token = queue_manager._get_system_state(
        'training_run_token', None)
    journal_token = value.get('run_token')
    token_matches = bool(
        journal_token and persisted_token == journal_token)
    dataset_matches = (
        not journal_token
        and queue_manager._get_system_state(
            'training_dataset_id', None) == value.get('dataset_id'))
    if token_matches or dataset_matches:
        return (
            queue_manager._get_system_state('training_pid', pid),
            queue_manager._get_system_state(
                'training_pid_create_time', birth_time),
        )
    return pid, birth_time


def _clear_matching_exact_resume_fence(value) -> None:
    persisted_token = queue_manager._get_system_state(
        'training_run_token', None)
    journal_token = value.get('run_token')
    if journal_token:
        if persisted_token == journal_token:
            _clear_training_identity(ttl_seconds=1)
    elif persisted_token is None:
        _clear_training_identity(ttl_seconds=1)


def _reconcile_exact_resume_journals() -> bool:
    """Recover exact-resume lane moves after watcher/Flask/process failure.

    Returns True when at least one live or indeterminate transaction keeps the
    GPU fence fail-closed.
    """
    try:
        root = _exact_resume_journal_root()
        entries = list(os.scandir(root))
    except FileNotFoundError:
        return False
    except RuntimeError:
        # The ordinary durable PID fence still works when ai-toolkit/output is
        # temporarily unconfigured; there cannot be a discoverable journal root.
        return False
    except OSError:
        logger.exception('could not inspect exact-resume recovery journals')
        return True
    held = False
    candidates = []
    for entry in entries:
        if not _EXACT_RESUME_JOURNAL_RE.fullmatch(entry.name):
            continue
        try:
            regular = entry.is_file(follow_symlinks=False)
        except OSError:
            logger.exception(
                'could not validate exact-resume journal candidate %s',
                entry.path)
            held = True
            continue
        if not regular:
            # A matching non-regular entry is untrusted recovery evidence. It
            # cannot mask later real journals, and it must not make admission
            # look clear merely because no-follow validation refused it.
            logger.error(
                'refusing non-regular exact-resume journal candidate %s',
                entry.path)
            held = True
            continue
        candidates.append(entry)
    # Scan every validated candidate. The directory is private/local and a hard
    # cap here is unsafe: enough junk names could otherwise push a live journal
    # beyond the cap and make recovery incorrectly release launch admission.
    for entry in candidates:
        journal_path = Path(entry.path)
        try:
            value = _read_exact_resume_journal(journal_path)
        except Exception:
            logger.exception(
                'refusing invalid exact-resume journal %s', journal_path)
            held = True
            continue
        from . import aitoolkit_state_bridge
        status = aitoolkit_state_bridge.read_status(value['status_path'])
        if isinstance(status, dict) and status.get('training_started') is True:
            try:
                _delete_exact_resume_journal(journal_path)
            except Exception:
                logger.exception(
                    'could not clear completed exact-resume journal %s',
                    journal_path)
                held = True
            continue

        training_folder = Path(value['training_folder'])
        archived = Path(value['archived'])
        if _journal_path_is_unsafe(training_folder) or _journal_path_is_unsafe(archived):
            logger.error(
                'refusing exact-resume rollback through unsafe run path: %s',
                journal_path)
            held = True
            _rearm_exact_resume_fence(value)
            continue
        # Already rolled back (watcher won before Flask restarted), or the
        # prepared intent was persisted before the source rename happened.
        if training_folder.is_dir() and not archived.exists():
            try:
                _delete_exact_resume_journal(journal_path)
            except Exception:
                logger.exception(
                    'could not clear idempotent exact-resume journal %s',
                    journal_path)
                held = True
            continue

        pid, birth_time = _matching_journal_process_identity(value)
        if pid is not None:
            alive = _pid_alive_with_birth(pid, birth_time)
            if alive is not False:
                held = True
                _rearm_exact_resume_fence(
                    value, pid=pid, birth_time=birth_time)
                continue
        elif value.get('phase') == 'launching':
            # Flask may have died immediately after Popen returned but before a
            # PID could be attached to either durable record. Absence of a PID is
            # therefore never negative launch proof: keep the lane and GPU fence
            # fail-closed until explicit operator recovery or stronger evidence.
            held = True
            _rearm_exact_resume_fence(value)
            continue
        elif value.get('phase') == 'spawned':
            held = True
            _rearm_exact_resume_fence(value)
            continue

        if not archived.is_dir():
            # Neither a source archive nor a trusted live child can be proven.
            # Keep evidence and the fence; guessing would risk data loss.
            held = True
            _rearm_exact_resume_fence(
                value, pid=pid, birth_time=birth_time)
            continue
        try:
            _rollback_unlaunched_exact_resume(
                training_folder, archived)
            _delete_exact_resume_journal(journal_path)
            _clear_matching_exact_resume_fence(value)
            logger.warning(
                'recovered pre-boundary exact resume for dataset %s',
                value.get('dataset_id'))
        except Exception:
            logger.exception(
                'exact-resume restart rollback failed for %s', journal_path)
            held = True
            _rearm_exact_resume_fence(
                value, pid=pid, birth_time=birth_time)
    return held


def recover_training_fence() -> str | None:
    """Reconcile the durable local-training GPU fence at boot or poll time."""
    with _launch_transaction_lock, _queue_lock, GPU_ARBITER_LOCK:
        return _advance_training_queue()


def process_training_queue() -> str | None:
    """Advance queued training through the same boot-safe GPU recovery path."""
    return recover_training_fence()


def _snapshot_final_checkpoint(dataset_id, step, base_model=_PERSISTED,
                               family=None, variant=_PERSISTED) -> str | None:
    """Copie le final bare `lora_<trigger>.safetensors` vers son nom NUMÉROTÉ
    `lora_<trigger>_<step:09d>.safetensors`. ai-toolkit écrit le résultat final SANS
    numéro de step ; sans ce snapshot :
      - continuer un entraînement écrase ce final sans aucune trace (perte) ;
      - list_checkpoints sous-estime le step de reprise (il compte le bare au DERNIER
        numéro existant, pas à son vrai step) → `continue_training` repart trop bas.
    Le snapshot rend chaque final permanent ET visible à son vrai step. Idempotent
    (ne réécrit jamais un numéroté existant). Retourne le nom créé, ou None."""
    try:
        step = int(step)
    except (TypeError, ValueError):
        return None
    if step <= 0 or dataset_id is None:
        return None
    ds = FaceDataset.query.get(int(dataset_id))
    if not ds:
        return None
    trigger = _safe_trigger(ds)
    run = str(_run_root(ds, base_model, family, variant) / f'lora_{trigger}')
    final = os.path.join(run, f'lora_{trigger}.safetensors')
    numbered = os.path.join(run, f'lora_{trigger}_{step:09d}.safetensors')
    if not os.path.isfile(final) or os.path.exists(numbered):
        return None
    try:
        shutil.copy2(final, numbered)
        logger.info('snapshot final → %s (step %d)', numbered, step)
        return os.path.basename(numbered)
    except OSError as e:
        logger.warning('snapshot final échoué : %s', e)
        return None


def _due_index(q) -> int | None:
    """Index du premier job DÛ de la file : sans `not_before`, ou dont l'échéance
    (ISO, heure locale serveur) est atteinte. Un job PROGRAMMÉ pour plus tard ne
    bloque pas ceux placés derrière lui. `not_before` illisible → dû (fail-open)."""
    now = datetime.now()
    for i, it in enumerate(q):
        nb = it.get('not_before')
        if not nb:
            return i
        try:
            if datetime.fromisoformat(str(nb)) <= now:
                return i
        except (TypeError, ValueError):
            return i
    return None


def _advance_training_queue() -> str | None:
    # First action under the launch locks: recover any exact-resume lane move
    # whose watcher vanished with Flask. This must precede queue admission.
    if _reconcile_exact_resume_journals():
        return None
    flag = bool(queue_manager._get_system_state('training_in_progress', False))
    pid = queue_manager._get_system_state('training_pid', None)
    vision_busy = bool(queue_manager._get_system_state('vision_in_progress', False))
    q = get_train_queue()

    if flag:
        pid_alive = _pid_alive(pid)
        if pid_alive is not False:
            # A failed PID probe is not evidence that ai-toolkit released
            # the GPU, so it deliberately keeps the same durable fence.
            # Re-arm the 4h TTLs on every poll: without this, a training run
            # longer than 4h would see these flags silently expire mid-run,
            # and the GPU gate (job_queue / gpu_busy_reason) would think
            # nothing is running and let the queue/vision grab the GPU back.
            queue_manager._set_system_state(
                'training_in_progress', True, ttl_seconds=_TRAIN_STATE_TTL)
            for key in _TRAIN_IDENTITY_KEYS:
                value = queue_manager._get_system_state(key, None)
                if value is not None:
                    queue_manager._set_system_state(
                        key, value, ttl_seconds=_TRAIN_STATE_TTL)
            return None  # toujours en cours
        # Process mort alors que le flag est levé → training terminé.
        # Snapshot du final en nom NUMÉROTÉ (immuable) AVANT d'enchaîner/libérer :
        # sinon un futur « continuer » écrase ce final sans trace. Idempotent, et ce
        # point tourne aussi via le poll /train/status (robuste à un restart Flask).
        try:
            _snapshot_final_checkpoint(
                queue_manager._get_system_state('training_dataset_id', None),
                queue_manager._get_system_state('training_target_step', None),
                base_model=queue_manager._get_system_state(
                    'training_base_model', _PERSISTED),
                family=queue_manager._get_system_state(
                    'training_train_type', None),
                variant=queue_manager._get_system_state(
                    'training_variant', _PERSISTED))
        except Exception as e:
            logger.warning('snapshot final (advance) échoué : %s', e)
        due = _due_index(q)
        if due is not None and not vision_busy:
            nxt = q[due]
            try:
                _launch_queued_item(nxt)  # remet le flag + un nouveau pid (pas de flap GPU)
                _save_queue(q[:due] + q[due + 1:])  # retirer SEULEMENT après lancement réussi
                logger.info(f"File training : terminé → lancement dataset {nxt['dataset_id']}")
                return f"next:{nxt['dataset_id']}"
            except Exception as e:
                # Échec → on retire l'item (évite une boucle infinie) mais on
                # SURFACE l'erreur au lieu de la perdre silencieusement.
                _save_queue(q[:due] + q[due + 1:])
                queue_manager._set_system_state(
                    'training_queue_error',
                    {'dataset_id': nxt.get('dataset_id'), 'error': str(e)}, ttl_seconds=3600)
                logger.error(f"File training : échec lancement {nxt.get('dataset_id')}: {e}")
                return None
        # File vide (ou uniquement des jobs programmés plus tard) → libérer le GPU
        # (le superviseur relance ComfyUI ; le ticker relancera le job à l'échéance).
        _clear_training_identity(ttl_seconds=1)
        logger.info("File training : terminé, aucune suite due → flag libéré")
        return 'released'

    due = _due_index(q)
    if due is not None and not vision_busy:
        nxt = q[due]
        try:
            _launch_queued_item(nxt)
            _save_queue(q[:due] + q[due + 1:])  # retirer SEULEMENT après lancement réussi
            logger.info(f"File training : lancement dataset {nxt['dataset_id']}")
            return f"launched:{nxt['dataset_id']}"
        except Exception as e:
            _save_queue(q[:due] + q[due + 1:])
            queue_manager._set_system_state(
                'training_queue_error',
                {'dataset_id': nxt.get('dataset_id'), 'error': str(e)}, ttl_seconds=3600)
            logger.error(f"File training : échec lancement {nxt.get('dataset_id')}: {e}")
            return None
    return None


# --- Programmation d'entraînements (jour + heure) -----------------------------
_scheduler_started = False


def start_training_scheduler(app, interval_seconds=60):
    """Ticker de fond : avance la file toutes les `interval_seconds` MÊME sans
    navigateur ouvert. Sans lui, seuls le poll /train/status et le watcher de fin
    de process faisaient avancer la file - un entraînement programmé à 3 h du
    matin ne serait jamais parti. Idempotent (un seul thread par process)."""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True
    try:
        with app.app_context():
            recover_training_fence()
    except Exception as exc:
        logger.warning('training boot recovery failed closed: %s', exc)

    def _tick():
        import time
        while True:
            time.sleep(interval_seconds)
            try:
                with app.app_context():
                    process_training_queue()
            except Exception as e:  # jamais fatal - le tick suivant réessaie
                logger.debug('training scheduler tick: %s', e)

    threading.Thread(target=_tick, daemon=True, name='train-scheduler').start()
    logger.info('Training scheduler démarré (tick %ss)', interval_seconds)
