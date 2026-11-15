"""Setup installer: run whitelisted, self-contained installs in a background
thread and expose their live state for polling. Actions:

  ml_extras          -> face scoring and masks in LDS's managed quality Python
  face_scoring       -> verify and install the selected face-scoring packages
  masks              -> verify and install the selected person-mask packages
  watermark_inpaint  -> verify and install in LDS's isolated watermark Python
  ollama_model       -> stream Ollama's /api/pull for the configured vision model
  klein_model        -> download the Klein 9B (KV) fp8 diffusion model into
                        <ComfyUI>/models/unet/klein/ — a PUBLIC download (no token). The KV
                        build caches the reference images' KV pairs on the first denoising
                        step, so multi-reference editing (the dataset engine's whole job) runs
                        up to 2.5x faster at identical quality. A 401 still logs recovery
                        steps as a safety net (see license_url below)
  klein_lora         -> download the consistency LoRA into <ComfyUI>/models/loras/klein/
  klein_text_encoder -> qwen_3_8b_fp8mixed into <ComfyUI>/models/text_encoders/
  klein_vae          -> flux2-vae into <ComfyUI>/models/vae/
  krea_model         -> the Krea 2 Turbo base into <ComfyUI>/models/diffusion_models/krea/
  krea_text_encoder  -> qwen3vl_4b_fp8_scaled into <ComfyUI>/models/text_encoders/
  krea_vae           -> qwen_image_vae into <ComfyUI>/models/vae/
  krea_identity_lora -> the Krea 2 Identity Edit LoRA (its author's file, served by the
                        LDS download server) into <ComfyUI>/models/loras/krea/
  krea_nodes         -> prepare the pinned node pack in a supported ComfyUI portable;
                        existing unmanaged folders are preserved. Restart and a loaded-node
                        check are required before the engine can be considered ready.

No shell, no client-supplied arguments: each action's command/URL/destination is fixed.

Pip actions are SERIALIZED (one at a time, second request queued in click order): two
pip processes writing the same environment race on a shared package's dist-info and
corrupt it — proven by repro (two concurrent installs of one big binary package into
one venv fail 6/6 with WinError 2 / Errno 13). Each pip run also retries once on a
transient file-lock error (an antivirus holding a fresh file). Model downloads and the
ollama pull don't touch a venv, so they stay parallel.
"""
from .timeout_settings import network_timeout, processing_timeout
import contextlib
import importlib
import json
import logging
import os
import re
import hashlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import requests

from . import capabilities
from . import config as cfg
from .utils.redact import redact_tokens, redact_user_paths
from .services import infer_env, managed_python, plugin_tool_sources
from .version import APP_VERSION

logger = logging.getLogger(__name__)

# Fixed catalog of the Klein downloads (re-checked 2026-07-17): all four files are
# PUBLIC downloads. The default UNET is the KV-cache build (flux-2-klein-9b-kv-fp8):
# it caches the reference images' KV pairs on the first denoising step, so multi-
# reference editing (the dataset engine's whole job) runs up to 2.5x faster at
# identical quality — same VAE/text-encoder. Unlike the plain 9b-fp8 repo (which is
# license-gated → 401 without a token), the KV repo is NOT access-gated: HF serves it
# publicly (verified: API gated=false, resolve → public CDN). The FLUX Non-Commercial
# License still governs USE. `license_url` is kept so a future re-gating (or a stale
# token) still degrades into actionable recovery steps rather than a bare 401.
# `legacy_names` = earlier default filenames still accepted as "already installed",
# so an install that fetched the pre-KV model never re-downloads ~10 GB (both variants
# resolve by name at generate time — see klein_edit_helper.resolve_klein_unet).
_KLEIN_DOWNLOADS = {
    'klein_model': {
        'url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8/resolve/main/flux-2-klein-9b-kv-fp8.safetensors',
        'dest': ('unet', 'klein', 'flux-2-klein-9b-kv-fp8.safetensors'),
        'min_free_gb': 15, 'gated': False,
        'license_url': 'https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8',
        'legacy_names': ('flux-2-klein-9b-fp8.safetensors',),
    },
    'klein_lora': {
        'url': 'https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency/resolve/main/Flux2-Klein-9B-consistency-V2.safetensors',
        'dest': ('loras', 'klein', 'Flux2-Klein-9B-consistency-V2.safetensors'),
        'min_free_gb': 1, 'gated': False,
    },
    # The detail LoRA node 139 of the improve workflow loads. It shipped as a
    # hardcoded filename the graph expected to already exist, so on any machine
    # without it the node was silently BYPASSED — the "Upscale & improve"
    # enhancement strength then moved nothing, with no way to tell. Downloading it
    # like every other Klein asset is what makes that setting mean something.
    # Same author as the consistency LoRA above; Apache-2.0, so linking the
    # original source is enough — the file is never re-hosted here.
    'klein_enhancement_lora': {
        'url': 'https://huggingface.co/dx8152/Flux2-Klein-9B-Enhanced-Details/resolve/main/realistic.safetensors',
        'dest': ('loras', 'klein', 'realistic.safetensors'),
        'min_free_gb': 1, 'gated': False,
        'license_url': 'https://huggingface.co/dx8152/Flux2-Klein-9B-Enhanced-Details',
    },
    'klein_text_encoder': {
        'url': 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors',
        'dest': ('text_encoders', 'qwen_3_8b_fp8mixed.safetensors'),
        'min_free_gb': 12, 'gated': False,
    },
    'klein_vae': {
        'url': 'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors',
        'dest': ('vae', 'flux2-vae.safetensors'),
        'min_free_gb': 2, 'gated': False,
    },
}

# Krea 2 Identity Edit — the SECOND local engine's weights. Same worker, same
# ".part then rename" streaming, same precondition as Klein: the ONLY thing that
# was ever missing here was a destination mapping (the engine shipped with a
# "place these four files yourself" message, i.e. five manual gestures).
#
# URL survey 2026-07-27 (anonymous HTTP, no token): the three Hugging Face files
# live in ONE public repo, `Comfy-Org/Krea-2` (API gated=false), and each
# `resolve/main/...` answered 200 with the full content-length. Measurements are
# a photograph of one moment — the worker therefore keeps the SAME 401/403
# recovery path as Klein, so a future re-gating degrades into actionable steps
# instead of a bare error.
#
# `dest[0]` is 'diffusion_models', NOT 'unet': both are the same ComfyUI folder
# type, but resolve_krea_unet scans `search_roots('diffusion_models')` for a
# 'krea'-named subfolder — 'krea' is exactly what it looks for.
#
# BASE VARIANT: Turbo, not Raw (~13 GB each, we install ONE). Two reasons, both
# in the code: krea_edit_helper.build_workflow pins cfg 1.0 / 10 steps /
# euler+simple — the guidance-distilled few-step regime Turbo IS — and
# resolve_krea_unet already prefers a 'turbo' build over a 'raw' one, so the file
# we fetch is the file the resolver would pick anyway. Someone who wants Raw
# drops it in the same folder and points krea.base_model at it.
_KREA_DOWNLOADS = {
    'krea_model': {
        'url': 'https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors',
        'dest': ('diffusion_models', 'krea', 'krea2_turbo_fp8_scaled.safetensors'),
        'min_free_gb': 15, 'gated': False, 'min_bytes': 1024 ** 3,
        'license_url': 'https://huggingface.co/Comfy-Org/Krea-2',
    },
    'krea_text_encoder': {
        # Canonical name — resolve_krea_text_encoder matches it EXACTLY first.
        'url': 'https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors',
        'dest': ('text_encoders', 'qwen3vl_4b_fp8_scaled.safetensors'),
        'min_free_gb': 7, 'gated': False, 'min_bytes': 256 * 1024 ** 2,
        'license_url': 'https://huggingface.co/Comfy-Org/Krea-2',
    },
    'krea_vae': {
        'url': 'https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors',
        'dest': ('vae', 'qwen_image_vae.safetensors'),
        'min_free_gb': 1, 'gated': False, 'min_bytes': 8 * 1024 ** 2,
        'license_url': 'https://huggingface.co/Comfy-Org/Krea-2',
    },
    # The identity LoRA is a community file published on Civitai. LDS serves its
    # own unmodified copy of it, so installing Krea 2 Edit depends on no other
    # site's login; the author's page (`license_url`) stays the reference for the
    # terms. `auth: 'none'`: neither the Hugging Face token nor a Civitai key is
    # ever sent to the LDS server. The size and the SHA256 are the published
    # file's, checked before the download takes its place. The filename matches
    # the krea.identity_lora default so the resolver finds it by canonical name.
    'krea_identity_lora': {
        'url': 'https://lds.perfectgf.fr/models/krea/krea2_identity_edit_v1_2.safetensors',
        'dest': ('loras', 'krea', 'krea2_identity_edit_v1_2.safetensors'),
        'min_free_gb': 3, 'gated': False, 'auth': 'none',
        'min_bytes': 512 * 1024, 'expected_bytes': 1828256432,
        'sha256': '6adf9a69cc9502d286db7b69964d37da7e9cfe4b05b4d004bc275f087d3fd3cf',
        'license_url': 'https://civitai.com/models/2761113',
    },
}

# SeedVR2 — the fidelity upscaler (issue #32, SurpassHR). Two files only, and
# the small one is the DEFAULT on purpose: the 3B FP8 build is 3.4 GB and the
# pack's own guidance puts it at 8-12 GB of VRAM, which is the card most people
# have. Someone with more drops a 7B build in the same folder and points
# `seedvr2.model` at it — seedvr2_helper.resolve_seedvr2_dit picks up anything
# present, so the bigger builds need no second install action.
#
# URL survey 2026-08-02 (anonymous HTTP HEAD, no token): `numz/SeedVR2_comfyUI`
# answers 200 with the full content-length on every file, API `gated=false`,
# licence apache-2.0 — the same licence as ByteDance's own SeedVR2 weights. The
# 401/403 recovery path is kept anyway, exactly like Klein's: a measurement is a
# photograph of one moment, and a future re-gating must degrade into actionable
# steps rather than a bare error.
#
# `dest[0]` is 'SEEDVR2' — the folder the node pack itself registers under
# ComfyUI's models dir (SEEDVR2_FOLDER_NAME in its constants.py), and the same
# string seedvr2_helper.MODEL_FOLDER searches.


# 📷 Camera angles — the weights that move the CAMERA rather than the subject.
#
# WHY A SECOND BASE MODEL AT ALL, when a 9 GB one is already installed. Measured
# on this repo's own Klein lane (2026-08-25, one reference, seed held constant):
# asked for a profile or a back view, Klein turns the PERSON and leaves the room
# exactly where it was — every phrasing tried, English, an explicit "only the
# photographer moves", the Chinese cinematography terms, with and without the
# one camera LoRA that exists for Klein 9B. The backdrop delta stayed at 11-14
# (noise) against 64-80 for a real viewpoint change. The Qwen LoRA below was
# trained on 3000+ gaussian-splatting renders — pairs where the subject cannot
# move and the background must — and reaches 64-80 on all eight angles tried.
# That inversion is the feature; it is not reachable by prompting harder.
#
# URL survey 2026-08-26 (anonymous HTTP HEAD, no token): all four answer 200 and
# the signed CDN URL carries `user_id=public` — none is access-gated. The
# 401/403 recovery path is kept anyway, like every other catalog here: a
# measurement is a photograph of one moment.
#
# NOT LISTED, on purpose: the Qwen image VAE. `krea_vae` above already installs
# the identical file to the identical destination — a second key for the same
# bytes would put the same gigabyte on the Setup screen twice and let two copies
# drift. services/qwen_camera_helper.CAMERA_VAE_ACTION names that button instead.
#
# `dest[0]` is 'diffusion_models' for the model and 'loras' for the adapters,
# both under a `qwen/` subfolder — the same shape the Klein and Krea lanes use,
# and what qwen_camera_helper._scan looks for first.


# Every streamed model download, whatever engine it belongs to. The worker,
# destination resolution, disk precondition and extra_model_paths de-duplication
# are engine-agnostic; only the catalog entries differ.
# MiniMax H3 - the Video Test Studio's engine. Four required files, 39.5 GB
# measured on disk, all from Comfy-Org's own conversion (verified against the
# hub's file list: not gated, and these exact paths). They land in ComfyUI's
# standard subfolders, which are also the paths the repository uses - so the
# training lane's WEIGHT_FOOTPRINTS and these entries name the same four files
# and cannot drift into two different opinions of what H3 needs.
#
# `min_free_gb` is the file's own size plus room to write it, not the lane's
# total: each action is downloaded on its own, and a machine that can take three
# of them should be told about the fourth rather than refused up front.


from .services.trained_image_models import model_downloads as _studio_model_downloads

_STUDIO_DOWNLOADS = _studio_model_downloads()
_MODEL_DOWNLOADS = {**_KLEIN_DOWNLOADS, **_KREA_DOWNLOADS, **_STUDIO_DOWNLOADS}

# Custom-node packs the app can install itself. The first git-cloned
# dependencies this app installs at all, so the rules are written down rather
# than implied:
#   * the URL is a CONSTANT here, never derived from user input, and the clone
#     runs as an argument list (no shell) with a timeout;
#   * the destination is <validated ComfyUI>/custom_nodes/<folder> — resolved
#     through the same capabilities.resolve_comfyui_base every other install
#     uses, and REFUSED (never guessed) when no valid ComfyUI is configured;
#   * an existing folder is left strictly alone (a user may have patched it);
#   * git may be absent (ZIP installs of ComfyUI have none), so a codeload ZIP
#     is the fallback — and if both fail the log says what to do by hand;
#   * ComfyUI only registers nodes at STARTUP, so a successful install reports
#     "restart ComfyUI", it never claims the engine is ready.
# `pip`: the pack declares `dependencies = []` (pyproject, checked 2026-07-27),
# so a clone is enough. We deliberately do NOT pip-install a third-party
# requirements file into the app's environment — if one appears the log says so
# and leaves the call to the user.
_NODE_PACKS = {
    'krea_nodes': {
        'pack': 'comfyui-krea2edit',
        'repo': 'https://github.com/lbouaraba/comfyui-krea2edit',
        'zip': 'https://lds.perfectgf.fr/models/tools/ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171/comfyui-krea2edit.zip',
        'sha256': 'ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171',
        'folder': 'comfyui-krea2edit',
    },
    # LanPaint: the training-free inpainting sampler the masked Repair lane
    # runs on (services/lanpaint_helper explains why it replaced
    # InpaintModelConditioning — GitHub #43). GPL-3.0, like ComfyUI itself and
    # installed the same way every ComfyUI custom node is: into the USER'S
    # ComfyUI, at their click. pyproject declares zero dependencies (checked
    # 2026-08-25, v2.1.0), so a clone is enough — same contract as the pack
    # above.
    'lanpaint_nodes': {
        'pack': 'LanPaint',
        'repo': 'https://github.com/scraed/LanPaint',
        'zip': 'https://lds.perfectgf.fr/models/tools/bc79a3466fb607d7ee70076b03059d11c14391cee25ad40f6afefc4604628d2f/LanPaint.zip',
        'sha256': 'bc79a3466fb607d7ee70076b03059d11c14391cee25ad40f6afefc4604628d2f',
        'folder': 'LanPaint',
    },
}

# WHY THE VIDEO STUDIO'S NODE PACKS ARE NOT HERE (maintainer's call, 2026-08-31)
# --------------------------------------------------------------------------
# "Downloading models is fine, but we do not take responsibility for breaking a
# ComfyUI install." A weight is an inert file in a models folder: worst case it
# is unused. A custom node is CODE that ComfyUI imports at startup, and one bad
# import takes the whole server down for every other lane the user has — a cost
# they did not agree to when they clicked a button in this app.
#
# So the three optional packs of the video lane (MiniMax-H3-Turbo,
# H3-Optimizations, MMH3-UltimateUpscale) and SageAttention (ComfyUI-KJNodes)
# are NAMED and LINKED by the studio, and installed by the user on the ComfyUI
# side, through ComfyUI-Manager or a clone. `video_test_studio.OPTION_NODE_PACKS`
# holds those links; nothing here fetches them.
#
# The two packs above (krea/lanpaint) predate that rule and keep their buttons —
# what changed is the direction, not a retrofit of somebody's working install.

# Node packs the app SHIPS (backend/comfy_nodes/<folder>), installed by COPY into
# the user's ComfyUI instead of fetched from a remote. See
# backend/comfy_nodes/README.md for the contract these folders sign.
#
# Why a second registry rather than a `local: True` flag on _NODE_PACKS: the two
# differ on the rule that matters most, which is what to do when the folder is
# already there. A third-party pack is LEFT ALONE (the user may have pinned a
# version of somebody else's code). Ours is OVERWRITTEN when the stamp inside it
# is not this app version — nobody pins ours, and a stale copy is how a user ends
# up running last month's sampler against this month's graph. One flag would have
# hidden an inverted rule inside a shared code path.
_BUNDLED_NODE_PACKS = {
    'moss_nodes': {
        'pack': 'MOSS-Image loaders',
        'folder': 'lds_moss_image',
    },
    'krea_sampler_nodes': {
        'pack': 'Krea 2 preset sampler',
        'folder': 'lds_krea_sampler',
    },
}

# The stamp file written into a deployed folder. Its presence is ALSO the
# permission to delete that folder: the installer only ever removes a directory
# it can prove it wrote itself.
_BUNDLED_STAMP = '.lds-version'

# Every action that lands something in <ComfyUI>/custom_nodes, whatever its
# source. The post-install cache clears and the ComfyUI-folder precondition apply
# to all of them; only the FETCH differs between the two registries.
_ALL_NODE_PACKS = tuple(_NODE_PACKS) + tuple(_BUNDLED_NODE_PACKS)

# aitoolkit_install puts the trainer itself into a NEW folder of its own
# (services/aitoolkit_install). Public core: no member lock. Not a pip-FIFO
# action -- it builds its own venv, so it never races the app's environment --
# and never part of "Install everything": several GB land only on an explicit
# click, with a chosen folder.
INSTALL_ACTIONS = ('aitoolkit_install', 'ml_extras', 'ollama_model', 'face_scoring', 'masks',
                   'watermark_inpaint', 'bank_scoring', 'bank_siglip2',
                   'watermark_detect', 'video_text') + tuple(_MODEL_DOWNLOADS) + _ALL_NODE_PACKS
# Actions that can be stopped mid-run: their worker owns the only child process.
_CANCELLABLE_ACTIONS = ('ollama_model', 'aitoolkit_install')

_ML_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-ml.txt'
_SCRAPE_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements-scrape.txt'
# pip -r installers share one worker; both target THIS interpreter (the scrape
# stack runs in-process, so any other environment would be invisible to the app).
_PIP_REQUIREMENTS = {'ml_extras': _ML_REQUIREMENTS, 'scrape_extras': _SCRAPE_REQUIREMENTS}
# The single package the watermark-inpaint scoped install adds. The NAME lives
# here (an identifier), but the VERSION SPEC is parsed from requirements-ml.txt
# so there's exactly one place a version floor is ever written.
_WATERMARK_PKG = 'simple-lama-inpainting'

# Bank scoring extra: CLIP (open_clip) + the NSFW classifier (transformers/timm)
# for the aesthetic/NSFW/style pass. Installed into a dedicated auto-provisioned
# venv with CPU torch — never the Flask venv (torch is heavy and version-touchy).
# These are NOT in requirements-ml.txt (which the monolithic ml_extras installs
# into the Flask venv); they live here and install only through the bank_scoring
# action, same isolation as the watermark torch install.
# These are NOT all in requirements-ml.txt, so most install unpinned; the ones that
# ARE (transformers, for Qwen3-VL) get their floor from there via _requirement_spec
# — see _bank_scoring_specs().
_BANK_SCORING_PKGS = ('open_clip_torch', 'transformers', 'timm', 'safetensors',
                      'huggingface_hub')

# The app's core requirements — Pillow is PINNED here (Pillow==12.x). An install
# that targets the Flask venv appends this pin so pip can never downgrade Pillow to
# satisfy an ML dependency (see _flask_pillow_guard).
_APP_REQUIREMENTS = cfg.BACKEND_DIR / 'requirements.txt'

# ML packages that must NEVER install into the Flask (app's own) venv: their pins
# would drag Pillow below the version the app REQUIRES (simple-lama-inpainting
# hard-requires pillow<10 vs the app's Pillow 12). They install ONLY into a
# dedicated ML interpreter (watermark.python / masks.python — a separate 3.10-3.12
# env); with none configured the install is refused with an actionable message,
# never forced into the Flask venv. That silent Pillow downgrade is the root of the
# "corrupted Python environment that survives updates" bug this module guards.
_FLASK_VENV_INCOMPATIBLE = frozenset({_WATERMARK_PKG})

# --- ML extras, split per capability -------------------------------------------
# requirements-ml.txt is a FLAT pip file (not grouped by feature), so the
# package->capability grouping lives HERE. The VERSIONS are never duplicated: each
# package's exact requirement line is read from requirements-ml.txt via
# _requirement_spec(), and that same file rides along as a `-c` constraint so a
# scoped install can't bump numpy past insightface's <2 ABI ceiling. A dedicated
# test (test_no_orphan_ml_package) asserts EVERY line in requirements-ml.txt is
# owned by at least one capability below — a package added to the file but
# forgotten here would silently never be installed by any scoped action.
#
#   face_scoring  insightface (face embeddings) + onnxruntime (its runtime). numpy
#                 is pinned <2 *for insightface's* ABI; opencv-python-headless is
#                 the server-safe cv2 that insightface & rembg both pull — listed so
#                 the scoped install prefers the headless variant, matching the
#                 monolithic `-r` install.
#   masks         rembg (u2net background removal) + onnxruntime (the runtime it
#                 RUNS on), + the same shared numpy / headless-opencv floor.
#                 onnxruntime is listed EXPLICITLY, not via a `rembg[cpu]` extra:
#                 rembg ≥2.0.50 imports onnxruntime at module load but stopped
#                 declaring it, so a scoped masks install resolved every package
#                 it knew about, reported success, and left `import rembg` dying
#                 on ModuleNotFoundError — the capability stayed ✗ with no reason
#                 shown (issue #24, 1Tomber). The explicit name is the durable
#                 choice: the `[cpu]`/`[gpu]` extras did not exist before 2.0.50
#                 and would silently resolve to nothing on an older pin, while
#                 the plain name is pinned once in requirements-ml.txt like every
#                 other ML package. _drop_provided_onnxruntime() below keeps it
#                 from stepping on a GPU build the user already has.
#   watermark_inpaint  simple-lama-inpainting (has its own dedicated worker below;
#                 listed here only so the anti-orphan test sees its package covered).
_CAPABILITY_PACKAGES = {
    'face_scoring': ('insightface', 'onnxruntime', 'numpy', 'opencv-python-headless'),
    'masks': ('rembg', 'onnxruntime', 'numpy', 'opencv-python-headless'),
    'watermark_inpaint': (_WATERMARK_PKG,),
    # 🎬 The video lane, split across two environments on purpose.
    #   video       decoding (PyAV, imported IN-PROCESS by Flask, so it must land
    #               in the app's own interpreter) plus a bundled static ffmpeg
    #               binary, which is what lets a user who has never installed
    #               ffmpeg export a dataset. Small; the generic ML worker handles it.
    #   shot_detect TransNetV2, which drags torch — so it rides the environment
    #               bank scoring already manages instead of costing a second
    #               ~2.5 GB copy. Its own worker, like watermark_detect; listed
    #               here so the anti-orphan test sees its package covered.
    #               It carries `av` OF ITS OWN even though `video` installs the
    #               same package: the two land in DIFFERENT interpreters. `video`
    #               puts PyAV in the app's own Python because Flask imports it
    #               in-process; shot detection runs in the bank-scoring
    #               environment, where `shot_detect_infer._open()` is the single
    #               decode seam. Without this line the install reported success
    #               and the capability stayed off — the probe imports av, so it
    #               kept failing in an environment nothing had put av into.
    #   video  `opencv-python-headless` and `numpy` are named for the camera
    #          pass, which tracks and fits in the app's own interpreter. The
    #          HEADLESS variant for the reason video_text names it below — the
    #          desktop `opencv-python` drags a GUI stack onto a server — and
    #          numpy explicitly because a scoped install must resolve it even
    #          when the headline package's metadata is vague.
    'video': ('imageio-ffmpeg', 'av', 'opencv-python-headless', 'numpy'),
    'shot_detect': ('transnetv2-pytorch', 'av'),
    #   video_text  RapidOCR, for the safe-zone pass's burned-in-text half. It
    #               lands in the SAME interpreter as face_scoring and masks (the
    #               app's own by default) because it is the same kind of extra:
    #               CPU onnxruntime, no torch, no second 2.5 GB copy of anything.
    #               `onnxruntime` and `numpy` are named here for the reason the
    #               masks line names them — a scoped install must resolve them
    #               even when the headline package's own metadata is vague, and
    #               _drop_provided_onnxruntime() still keeps this from stepping
    #               on a GPU build the user installed themselves.
    #               `opencv-python-headless` is named because RapidOCR depends on
    #               the DESKTOP `opencv-python`, which drags a GUI stack onto a
    #               server: naming the headless variant makes pip prefer it, the
    #               same trick face_scoring and masks already use for the same
    #               transitive dependency.
    # pillow: the worker's unicode-path reader falls back to PIL (cv2.imread
    # cannot open non-ASCII paths on Windows). Always present in the app's own
    # Python — the app itself requires it — listed so a custom video_text
    # interpreter gets it installed rather than probing ✗ unrepairably.
    'video_text': ('rapidocr-onnxruntime', 'onnxruntime', 'numpy',
                   'opencv-python-headless', 'pillow'),
    #   bank_scoring  has its own worker and its own package tuple
    #                 (_BANK_SCORING_PKGS); only the ONE package whose version
    #                 floor matters is declared in requirements-ml.txt, so it is
    #                 named here too — otherwise the anti-orphan test below sees
    #                 an unowned line. Same bookkeeping-only role as shot_detect.
    'bank_scoring': ('transformers',),
}
# The capabilities served by the GENERIC per-capability pip worker
# (_run_ml_capability). watermark_inpaint keeps its own worker, so it's excluded.
_CAPABILITY_ML_ACTIONS = ('face_scoring', 'masks', 'video', 'video_text')
_MANAGED_CAPABILITY_ACTIONS = (*_CAPABILITY_ML_ACTIONS, 'shot_detect')

# Actions whose success makes a NEW importable package appear -> the probe
# import-cache must be dropped so the capability flips without waiting out the
# 600 s TTL (ml_extras/scrape_extras via -r, the scoped per-capability installs).
_IMPORT_CACHE_ACTIONS = (frozenset(_PIP_REQUIREMENTS)
                         | set(_CAPABILITY_ML_ACTIONS)
                         | {'watermark_inpaint', 'bank_scoring', 'bank_siglip2',
                            'watermark_detect', 'shot_detect'})

# Actions that invoke pip and therefore MUST NOT run concurrently: two pip processes
# writing the same environment race on a shared package's files/dist-info and corrupt
# it (proven by repro: two concurrent installs of one big binary package into one venv
# fail 6/6 with WinError 2 / Errno 13 on the package's dist-info). All the default ML
# installs target the app's own venv (no dedicated python), so these are serialized to
# ONE at a time; a second request is QUEUED in click order. Model downloads and the
# ollama pull touch models/ or the network, not a venv, so they are NOT here and keep
# running in parallel.
_PIP_ACTIONS = (frozenset(_PIP_REQUIREMENTS)
                | set(_CAPABILITY_ML_ACTIONS)
                # watermark_detect installs into the SAME venv bank_scoring owns
                # (that sharing is the whole point — it saves a second 2.5 GB
                # torch), so it must share the pip queue too or the two race on
                # one environment's dist-info.
                | {'watermark_inpaint', 'bank_scoring', 'bank_siglip2',
                   'watermark_detect', 'shot_detect'})

# Transient file-lock errors an install can hit even without concurrency: an antivirus
# or the search indexer briefly holding a just-written file at the moment pip renames
# it (classically Bitdefender on Windows -> Errno 13; a sharing violation -> WinError
# 32; access denied -> WinError 5). These are retryable: pip is idempotent, so rerunning
# finishes the interrupted step. A genuine "no wheel / build failed" error does NOT match
# and is surfaced immediately.
_RETRYABLE_PIP_ERR = re.compile(
    r'Errno 13|Permission denied|WinError 5\b|WinError 32|WinError 2\b|being used by another process',
    re.IGNORECASE)
_PIP_RETRIES = 3          # total attempts on a retryable error
_PIP_RETRY_BACKOFF = 3    # seconds * attempt number between tries

_LOG_MAX = 400  # ring-buffer the log so a chatty pip can't grow unbounded
_OLLAMA_CONNECT_TIMEOUT = 5
_OLLAMA_READ_TIMEOUT = 45
_OLLAMA_STREAM_CHUNK = 8192
_OLLAMA_MAX_LINE = 64 * 1024

_lock = threading.Lock()
_runs = {}  # action -> {'state', 'returncode', 'log', 'progress', 'waiting_for'}
# Pip serialization (guarded by _lock): the single action currently occupying the pip
# worker, and the FIFO of actions waiting their turn (click order).
_pip_current = None
_pip_queue = []


class AlreadyRunning(Exception):
    pass


class Precondition(Exception):
    pass


class Cancelled(Exception):
    pass


def _new_run():
    return {'state': 'running', 'returncode': None, 'log': [], 'progress': None,
            'waiting_for': None, 'cancel_event': threading.Event(), 'response': None,
            'runtime_notice': None, 'step': None}


def _append(action, line):
    # A worker thread can outlive its registry entry: the tests reset _runs
    # between cases while a download thread is still draining, and clearing
    # runs mid-flight is one registry write away in prod too. A cleared entry
    # means nobody is watching this run any more — drop the line rather than
    # killing the thread (the CI's recurring `KeyError: 'seedvr2_model'`
    # warning was this, raised from the error handler's own _append).
    run = _runs.get(action)
    if run is None:
        return
    log = run['log']
    if action == 'aitoolkit_install' or plugin_action_spec(action) is not None:
        line = redact_user_paths(redact_tokens(line))
    log.append(line.rstrip('\n'))
    if len(log) > _LOG_MAX:
        del log[:-_LOG_MAX]


def _finish_run(action, returncode, state):
    """Stamp a worker's final state, tolerating an entry cleared under it —
    same contract as _append, for the same orphaned-thread reason."""
    run = _runs.get(action)
    if run is None:
        return
    run['returncode'] = returncode
    run['state'] = state


def _note(action, line):
    """_append for the presence checks, which are ALSO called outside a run (the
    install plan and the tests ask them directly). No run -> no log, no KeyError."""
    if action in _runs:
        _append(action, line)


def _set_progress(action, done, total):
    """Publish a live byte-progress snapshot for a streaming download, separate
    from the text log (so a smooth % bar never spams the log). `total` may be 0
    when the server sends no content-length -> pct is None (indeterminate)."""
    run = _runs.get(action)
    if run is None:
        return
    run['progress'] = None if done is None else {
        'done': done,
        'total': total,
        'pct': (done * 100 // total) if total else None,
    }


def _set_step(action, index, total, label):
    """Which stage of a multi-step install is running ("3 of 6 · Installing
    PyTorch…"), next to the byte progress of the download inside it."""
    run = _runs.get(action)
    if run is not None:
        run['step'] = {'index': index, 'total': total, 'label': label}


def _quote(p: str) -> str:
    # Quote paths with spaces so the manual command is copy-paste-safe: the
    # portable bundle can be extracted under e.g. C:\Users\...\LoRA Dataset Studio\.
    return f'"{p}"' if ' ' in p else p


def _canon(name: str) -> str:
    """PEP 503 canonical form: -_. all fold to a single dash, case-insensitive."""
    return re.sub(r'[-_.]+', '-', name).lower()


# Canonical names of the Flask-venv-incompatible packages, for membership tests.
_INCOMPATIBLE_CANON = frozenset(_canon(n) for n in _FLASK_VENV_INCOMPATIBLE)


def _requirement_spec(name: str, requirements=_ML_REQUIREMENTS) -> str:
    """The full requirement line for `name` as written in a requirements file
    (e.g. 'simple-lama-inpainting>=0.1.2') — the version floor lives in ONE place
    (requirements-ml.txt), never duplicated in this module. Package-name match is
    canonicalised (PEP 503: -_. all fold together, case-insensitive) and tolerant
    of version/marker/extras suffixes. Falls back to the bare name if the file or
    line is missing (an unpinned `pip install <name>` still works)."""
    canon = _canon(name)
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()   # drop comments / blank lines
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]   # name before any spec/marker
            if _canon(token) == canon:
                return line
    except OSError:
        pass
    return name


def _ml_requirement_names(requirements=_ML_REQUIREMENTS) -> set:
    """Canonical names of every package declared in a requirements file (comments
    and blank lines dropped). Used by the anti-orphan test to prove each ML package
    is mapped to a capability in _CAPABILITY_PACKAGES."""
    names = set()
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]
            names.add(_canon(token))
    except OSError:
        pass
    return names


def _ml_requirement_specs(*, exclude=frozenset(), requirements=_ML_REQUIREMENTS) -> list:
    """Requirement lines from requirements-ml.txt in FILE ORDER, dropping any whose
    canonical name is in `exclude`. One source of truth for the ML versions — the
    scoped installers build their constrained package lists from here."""
    out = []
    try:
        for raw in requirements.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            token = re.split(r'[<>=!~;\[\s]', line, maxsplit=1)[0]
            if _canon(token) in exclude:
                continue
            out.append(line)
    except OSError:
        pass
    return out


def _bank_scoring_specs() -> list:
    """_BANK_SCORING_PKGS with every version floor requirements-ml.txt knows about
    applied (bare name for the rest). The floor is what makes re-clicking ✨ Score a
    REPAIR: `pip install transformers` is a no-op against an already-installed older
    transformers, while `pip install "transformers>=4.57"` upgrades it — and 4.57 is
    where `Qwen3VLForConditionalGeneration` (infer/video_caption_infer.py:103, the
    video-caption worker) first exists. Callers building a SHELL string must quote
    each spec; '>=' unquoted is redirection."""
    return [_requirement_spec(p) for p in _BANK_SCORING_PKGS]


def _app_pillow_spec() -> str:
    """The Pillow pin from requirements.txt (e.g. 'Pillow==12.2.0') — the version the
    Flask venv MUST keep. Appended as an explicit requirement to any install that
    targets the Flask venv so pip REFUSES (clean error) rather than silently
    DOWNGRADES Pillow to satisfy an ML dependency. Bare-name fallback still blocks
    the known-bad <10 downgrade if the pin can't be parsed."""
    spec = _requirement_spec('Pillow', requirements=_APP_REQUIREMENTS)
    return spec if spec.lower() != 'pillow' else 'Pillow>=10'


def _venv_root(python: str) -> str:
    """The venv directory that OWNS `python` (its grandparent, when a pyvenv.cfg
    marks it as one), resolved and case-normalised — or '' for a non-venv path.

    This is the identity that matters when two interpreter paths are compared:
    on Linux a venv's bin/python is a SYMLINK to the base interpreter, so
    resolving the BINARY (os.path.samefile) answers "same base Python?", never
    "same environment?". Every venv on the machine then collapses into one —
    which is how the Flask-venv guard mistook the app-managed bank-scoring env
    for the app's own venv inside the GPU Docker image and refused installs that
    were the whole point of the button. The DIRECTORY is still resolved (a data
    dir reached through a mount symlink must match itself); only the binary is
    taken at face value. Conda envs carry no pyvenv.cfg and return '', keeping
    their comparisons on the old samefile path. Never raises."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(python or '')))
        if root and os.path.isfile(os.path.join(root, 'pyvenv.cfg')):
            return os.path.normcase(os.path.realpath(root))
    except OSError:
        pass
    return ''


def _is_flask_venv(python: str) -> bool:
    """True when `python` resolves to the app's OWN interpreter (the Flask venv) —
    the environment whose Pillow must never be downgraded. Compares ENVIRONMENTS,
    not binaries (see _venv_root): two venvs are the same only when they are the
    same directory, and a venv is never the same environment as a bare system
    Python, even the one it was built from. Case/separator-insensitive on
    Windows; never raises."""
    own, other = _venv_root(sys.executable), _venv_root(python)
    if own or other:
        return bool(own) and bool(other) and own == other
    try:
        return os.path.samefile(python, sys.executable)
    except OSError:
        return (os.path.normcase(os.path.abspath(python))
                == os.path.normcase(os.path.abspath(sys.executable)))


def _flask_pillow_guard(python: str) -> list:
    """Pillow pin to append to a pip install ONLY when it targets the Flask venv:
    pip then can't silently downgrade the app's Pillow (it keeps it or fails clean).
    A dedicated ML env is exempt — it may legitimately need pillow<10 for
    simple-lama-inpainting."""
    return [_app_pillow_spec()] if _is_flask_venv(python) else []


def _watermark_python() -> str:
    """Interpreter the watermark LaMa wrapper resolves. Reuse the wrapper's OWN
    resolver (watermark.python > masks.python > sys.executable) so the install
    target and the later import can never drift apart."""
    from .services import watermark_lama
    return watermark_lama.lama_python()


def _capability_python(action) -> str:
    """Interpreter a scoped ML install targets — MUST match the resolution its
    matching probe uses, so the install target and the later import can't drift:
      face_scoring -> face_scoring.python  (see capabilities.probe_face_scoring)
      masks        -> masks.python         (see capabilities.probe_masks)
      watermark_inpaint -> the wrapper chain (watermark.python > masks.python)."""
    if action == 'watermark_inpaint':
        return _watermark_python()
    return cfg.get(f'{action}.python') or sys.executable


def _manual_pip(python, args, *, index=plugin_tool_sources.PYPI):
    cmd = f'{_quote(python)} -m pip install {args} --index-url {index} --isolated'
    if 'transnetv2-pytorch' in args:
        cmd += ' --constraint ' + _quote(plugin_tool_sources.MODEL_CONSTRAINTS)
    return cmd


def manual_command(action) -> str:
    """The exact command that reproduces an install BY HAND, scoped to THIS app's
    own interpreter (sys.executable). A copy-paste then targets the SAME
    environment the app imports from -- the portable bundle's python\\python.exe or
    the dev venv -- instead of whatever bare `pip` happens to be first on PATH
    (which is the whole point of the user's question: a plain `pip install` would
    land in the wrong environment and the extras would never be importable)."""
    if action in ('ml_extras', 'face_scoring', 'masks', 'video_text', 'watermark_inpaint'):
        return '(Use Install in Setup: LDS creates and verifies the isolated Python first.)'
    if action == 'aitoolkit_install':
        # Polled with every status read, so never the plan itself (it runs
        # nvidia-smi and Python probes); GET /api/setup/aitoolkit-install/plan
        # lists the exact commands for this machine.
        return '(Use Install in Setup: LDS checks the GPU, the folder and the free space first.)'
    spec = plugin_action_spec(action)
    if spec is not None:
        if callable(spec.get('run')):
            return f"(run from Setup: {spec.get('label') or action})"
        try:
            command, _ = plugin_tool_sources.pip_install(_plugin_action_command(spec), {})
            return ' '.join(_quote(part) for part in [*command, '--isolated'])
        except Precondition as e:
            return f'({e})'
    if action in _PIP_REQUIREMENTS:        # scrape_extras: pure-python -r install
        return _manual_pip(sys.executable, '-r ' + _quote(str(_PIP_REQUIREMENTS[action])))
    if action in _CAPABILITY_ML_ACTIONS:
        # One scoped capability (face_scoring | masks): the exact version-pinned
        # lines from requirements-ml.txt, quoted (the '>=' / '<' are shell
        # redirection unquoted), plus that file as a -c constraint. Interpreter =
        # the same one the capability's probe resolves; when that's the Flask venv,
        # Pillow is pinned too so the scoped install can't downgrade it either.
        python = _capability_python(action)
        specs = ' '.join(f'"{_requirement_spec(p)}"' for p in _CAPABILITY_PACKAGES[action])
        guard = ' '.join(f'"{g}"' for g in _flask_pillow_guard(python))
        args = f'{specs} -c {_quote(str(_ML_REQUIREMENTS))}'
        return _manual_pip(python, f'{args} {guard}' if guard else args)
    if action == 'watermark_inpaint':
        # Quote the spec: the '>=' in 'simple-lama-inpainting>=0.1.2' is shell
        # redirection unquoted. Interpreter = the wrapper's resolved python — but
        # NEVER the Flask venv (simple-lama needs pillow<10 and would break the app).
        # When nothing dedicated is configured the Install button AUTO-BUILDS a
        # dedicated venv; this debug/diagnostic line points at that managed venv.
        python = _watermark_python()
        spec = _requirement_spec(_WATERMARK_PKG)
        if _is_flask_venv(python):
            python = _watermark_env_python()
        return _manual_pip(python, f'"{spec}"')
    if action == 'bank_scoring':
        # The dedicated managed venv (auto-built) + CPU torch + the CLIP/NSFW stack.
        # This command documents what the Install button itself does.  A borrowed
        # Score interpreter is a runtime selection, never an install target.
        # The specs keep their version floors: a bare package name is a no-op on
        # an environment that already carries an older copy (see
        # test_requirements_ml_floors_transformers_for_qwen3vl).
        python = _bank_scoring_env_python()
        pkgs = ' '.join(f'"{s}"' for s in _bank_scoring_specs())
        return '  &&  '.join((_manual_pip(python, 'torch torchvision', index=_TORCH_CPU_INDEX), _manual_pip(python, pkgs)))
    if action == 'shot_detect':
        # One line, and no weights step: transnetv2-pytorch carries its own inside
        # the wheel. Targets the scoring environment because of torch.
        python = (cfg.get('shot_detect.python') or cfg.get('bank_scoring.python')
                  or _bank_scoring_env_python())
        pkgs = f'"{_requirement_spec("transnetv2-pytorch")}" "{_requirement_spec("av")}"'
        return '  &&  '.join((_manual_pip(python, 'torch torchvision', index=_TORCH_CPU_INDEX), _manual_pip(python, pkgs)))
    if action == 'bank_siglip2':
        # ALWAYS the LDS-managed environment — exactly what the Install button
        # does, and deliberately blind to ``bank_semantic.python``. That key is
        # now a picker ("run the index in the Python that already has CUDA"), so
        # reading it here would turn a repair into a pip install inside someone
        # else's ai-toolkit or ComfyUI venv. Where it RUNS and where we INSTALL
        # are two different questions; this line only ever answers the second.
        python = _bank_semantic_install_python()
        from .services import bank_semantic_models as assets
        root = assets.models_root()
        pulls = '; '.join(
            f"d(repo_id='{assets.MODEL_ID}', filename='{name}', "
            f"revision='{assets.REVISION}', cache_dir=r'{root}')"
            for name in assets.FILES)
        packages = _manual_pip(python, '"transformers>=4.49" huggingface_hub safetensors sentencepiece Pillow')
        weights = f'{_quote(python)} -c "from huggingface_hub import hf_hub_download as d; {pulls}"'
        return '  &&  '.join((_manual_pip(python, 'torch torchvision', index=_TORCH_CPU_INDEX), packages, weights))
    if action == 'watermark_detect':
        # Packages then weights. The weights line names the FILES on purpose —
        # a bare `snapshot_download` of the SigLIP2 repo pulls its training
        # checkpoints and costs 2.4 GB instead of 371 MB (measured).
        python = _watermark_detect_python()
        from .services import watermark_detector
        root = watermark_detector.models_root() or '<data>/models/watermark_detect'
        pulls = '  &&  '.join(
            f'{_quote(python)} -c "from huggingface_hub import hf_hub_download as d; '
            + '; '.join(f"d(repo_id='{repo}', filename='{name}', cache_dir=r'{root}')"
                        for name in meta['files'])
            + '"'
            for repo, meta in watermark_detector.MODEL_FILES.items())
        packages = _manual_pip(python, 'transformers huggingface_hub safetensors')
        return '  &&  '.join((_manual_pip(python, 'torch torchvision', index=_TORCH_CPU_INDEX), packages, pulls))
    if action == 'ollama_model':
        # The Studio container need not have an Ollama CLI; this action is HTTP-only.
        return ''
    if action == 'dlss5nr_bridge':
        from .services import neural_render
        return (f'curl -L -o bridge.zip "{neural_render.BRIDGE_RELEASE["url"]}"  '
                f'&&  unzip bridge.zip -d "{neural_render.runtime_dir()}"')
    spec = model_download_spec(action)
    if spec is not None:
        try:
            dest = _download_dest_path(action)
            root = _comfyui_root()
        except Precondition:
            dest = os.path.join('<ComfyUI>', 'models', *spec['dest'])
            root = '<ComfyUI>'
        # A stage download names every file it lands, the branch first: the
        # manual equivalent of one button is the whole directory, not one file.
        lines = [f'curl -L -o "{dest}" "{spec["url"]}"']
        for comp in spec.get('companions') or ():
            lines.append(f'curl -L -o "{os.path.join(root, "models", *comp["dest"])}" "{comp["url"]}"')
        return '  &&  '.join(lines)
    if action in _NODE_PACKS:
        spec = _NODE_PACKS[action]
        try:
            dest = _node_pack_dest(action)
        except Precondition:
            dest = os.path.join('<ComfyUI>', 'custom_nodes', spec['folder'])
        return (f'Download {spec["zip"]}, verify SHA-256 {spec["sha256"]}, then extract the archive folder '
                f'to "{dest}" and restart ComfyUI.')
    if action in _BUNDLED_NODE_PACKS:
        # No remote to fetch: the "manual command" is the copy this action performs.
        try:
            dest = _bundled_pack_dest(action)
        except Precondition:
            dest = os.path.join('<ComfyUI>', 'custom_nodes',
                                _BUNDLED_NODE_PACKS[action]['folder'])
        return f'copy "{_bundled_pack_source(action)}" -> "{dest}"'
    return ''


def status(action) -> dict:
    run = _runs.get(action)
    cmd = manual_command(action)
    if run is None:
        return {'state': 'idle', 'returncode': None, 'log': [], 'progress': None,
                'waiting_for': None, 'cancel_requested': False,
                'manual_command': cmd, 'runtime_notice': None, 'step': None}
    return {'state': run['state'], 'returncode': run['returncode'],
            'log': list(run['log']), 'progress': run.get('progress'),
            'step': dict(run['step']) if run.get('step') else None,
            # 'queued' -> which action it's waiting behind (the UI shows an honest
            # "waiting for another install" instead of a dead-looking button).
            'waiting_for': run.get('waiting_for'),
            'cancel_requested': bool(run.get('cancel_event')
                                     and run['cancel_event'].is_set()),
            # Kept for the diagnostic/debug log only — no longer shown as a user
            # "run this by hand" path (installs auto-recover or repair on re-click).
            'manual_command': cmd,
            'runtime_notice': dict(run['runtime_notice']) if run.get('runtime_notice') else None}


def start(action, options=None) -> dict:
    # The same short admission lock protects plugin removal/disable/replace.
    # Install work itself remains in the existing FIFO worker, outside it.
    # `options` is read by the one action that takes any (the AI Toolkit
    # destination folder); every other action ignores it.
    from .plugins.lifecycle import state_change_lock
    with state_change_lock:
        return _start_locked(action, options)


def cancel(action) -> dict:
    """Request cancellation of an install whose interruption is safe.

    Pip/model-file workers are not process-safe to interrupt. Ollama's pull is:
    closing its response releases a blocked reader while the event handles the
    race before/after the response is registered. The AI Toolkit install is too:
    it writes only its own new folder, journals every completed step and
    rebuilds an interrupted one, so its worker stops its current child process.
    """
    if action not in _CANCELLABLE_ACTIONS:
        raise Precondition('only the Ollama model pull and the AI Toolkit install can be cancelled')
    response = stopper = None
    with _lock:
        run = _runs.get(action)
        if run is None or run.get('state') != 'running':
            return status(action)
        event = run.get('cancel_event')
        if event is None:
            event = threading.Event()
            run['cancel_event'] = event
        event.set()
        response = run.get('response')
        stopper = run.get('stopper')
    if response is not None:
        try:
            response.close()
        except Exception:
            pass
    if stopper is not None:
        try:
            stopper()
        except Exception:
            logger.debug('could not stop the child process of %s', action, exc_info=True)
    return status(action)


def _release_pip_slot(finished):
    """A pip action finished: free the worker and launch the next queued pip action
    (FIFO). Model downloads / ollama pulls never touch these globals."""
    global _pip_current
    nxt = None
    with _lock:
        if _pip_current == finished:
            _pip_current = None
        if _pip_queue and _pip_current is None:
            nxt = _pip_queue.pop(0)
            _pip_current = nxt
            run = _runs.get(nxt)
            if run is not None:
                run['state'] = 'running'
                run['waiting_for'] = None
    if nxt is not None:
        threading.Thread(target=_execute, args=(nxt,), daemon=True).start()


def _ollama_pull_base_url() -> str:
    raw = cfg.get('ollama.url') or ''
    url = capabilities._validated_setup_http_base(raw)
    if not url:
        raise Precondition('ollama.url must be an HTTP(S) origin without credentials or a path')
    return url


def _check_ollama_precondition():
    _ollama_pull_base_url()
    if not (cfg.get('ollama.vision_model') or '').strip():
        raise Precondition('ollama.vision_model not configured')


def _comfyui_root() -> str:
    """The VALIDATED ComfyUI install root every install writes into. Raises
    Precondition when base_dir isn't a real install — we must never scatter
    multi-GB files, nor clone third-party code, under a wrong folder."""
    r = capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')
    if not r['valid']:
        raise Precondition('point the app at a valid ComfyUI folder first (Setup, ComfyUI step)')
    return r['resolved']


def _download_dest_path(action) -> str:
    """Absolute destination for a model download, under the validated ComfyUI
    models root."""
    spec = model_download_spec(action)
    root = _comfyui_root()
    models_root = os.path.join(root, 'models')
    if action in _STUDIO_DOWNLOADS and cfg.get('comfyui.models_dir'):
        # Studio discovery reads the configured models root. Downloading into
        # the install's default directory would leave those assets invisible.
        # Keep validating the ComfyUI install before honoring its override.
        models_root = os.path.abspath(os.fspath(cfg.comfyui_dir('models')))
    return os.path.join(models_root, *spec['dest'])


def _node_pack_dest(action) -> str:
    """Absolute destination folder for a custom-node pack: THIS install's
    <ComfyUI>/custom_nodes/<pack folder>. The folder name is a constant from
    _NODE_PACKS, never anything a request supplied."""
    return os.path.join(_comfyui_root(), 'custom_nodes', _NODE_PACKS[action]['folder'])


def _bundled_pack_source(action) -> str:
    """Where the shipped folder lives inside THIS install: backend/comfy_nodes/<folder>.

    `packaging/build_release_zip.ps1` robocopies all of `backend/` into the
    release, so this path is as valid in a downloaded ZIP as in a git checkout —
    which is the reason the folder lives under backend/ and not at the repo root."""
    return str(cfg.BACKEND_DIR / 'comfy_nodes' / _BUNDLED_NODE_PACKS[action]['folder'])


def _bundled_pack_dest(action) -> str:
    """Absolute destination for a shipped pack: <validated ComfyUI>/custom_nodes/<folder>.
    Raises Precondition (via _comfyui_root) when ComfyUI's folder isn't set yet."""
    return os.path.join(_comfyui_root(), 'custom_nodes',
                        _BUNDLED_NODE_PACKS[action]['folder'])


def _bundled_pack_stamp(dest) -> str | None:
    """The app version recorded inside a deployed folder, or None when there is no
    stamp — which also means "we did not put this here", and the installer must
    not delete it."""
    try:
        with open(os.path.join(dest, _BUNDLED_STAMP), encoding='utf-8') as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _bundled_pack_state(action) -> str:
    """'absent' | 'current' | 'stale' | 'foreign' for the deployed copy.

    'foreign' is a real folder under our name that carries no stamp: either a
    hand-installed copy or a leftover. It is reported, never overwritten — the
    app does not get to silently delete something in the user's ComfyUI that it
    cannot prove it wrote."""
    try:
        dest = _bundled_pack_dest(action)
    except Precondition:
        return 'absent'
    if not os.path.isdir(dest):
        return 'absent'
    stamp = _bundled_pack_stamp(dest)
    if stamp is None:
        return 'foreign'
    return 'current' if stamp == APP_VERSION else 'stale'


def _check_download_precondition(action):
    dest = _download_dest_path(action)
    spec = model_download_spec(action)
    # Check the nearest existing ancestor even when nested model folders
    # do not exist yet on a fresh installation.
    probe = os.path.dirname(dest)
    while probe and not os.path.isdir(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    try:
        free_gb = shutil.disk_usage(probe).free / 1e9
        if free_gb < spec['min_free_gb']:
            raise Precondition(f'not enough disk space: {free_gb:.1f} GB free, '
                               f"~{spec['min_free_gb']} GB needed for this file")
    except OSError:
        pass


# --- "Install everything" orchestrator -----------------------------------------
# One click that queues every install the app can run ITSELF right now — the missing
# ML extras, the Ollama vision model, and the Klein weights — instead of walking the
# user through each step. It never installs ComfyUI/Ollama themselves nor pastes API
# keys (those are external / credentials), so the plan is deliberately the subset whose
# preconditions are already satisfiable. Firing order is grouped by capability area for
# a coherent "X / N" progress display; the real scheduling still comes from start()
# (pip serialized FIFO, model downloads parallel), so the order here is cosmetic.
_INSTALL_ALL_ORDER = ('scrape_extras', 'face_scoring', 'masks', 'watermark_inpaint',
                      'klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora',
                      'klein_enhancement_lora')


def _broken_or_missing(missing, invalid) -> set:
    """Asset actions that need (re)downloading: absent from disk, OR present under
    the resolved name but not loadable (capabilities' `*_invalid`, blocking only).

    A corrupted file is not "installed". Judging these lists on `*_missing` alone
    is what let a one-click install plan NOTHING while the engine stayed dark — the
    file was there, so nothing looked missing. Mirrored in the front by
    useSetupSteps.brokenOrMissing; this is the authority both plans recompute."""
    out = set(missing or [])
    for i in (invalid or []):
        if isinstance(i, dict) and i.get('blocking') and i.get('asset'):
            out.add(i['asset'])
    return out


def _action_needed(action, caps) -> bool:
    """Is `action` both MISSING and satisfiable right now, from live capabilities?
    Pure (caps in, bool out) — the single rule install_all_plan is built from."""
    if action in ('face_scoring', 'masks'):
        runtime = (caps.get('python') or {}).get('managed_ml')
        if runtime is not None and not runtime.get('available', False):
            return False
        return not caps.get(action)
    if action == 'watermark_inpaint':
        # Auto-provisions its own 3.10-3.12 venv, so it's runnable on any interpreter.
        return not caps.get('watermark_inpaint')
    if action == 'ollama_model':
        # Only when Ollama is already reachable AND a model name is configured (the pull
        # needs a target) — Ollama itself can't be auto-installed here.
        #
        # And only when Ollama is the SELECTED provider. Offering to pull an Ollama
        # model to someone running LM Studio installs several GB they will never use,
        # and it would happen exactly when they are most likely to click: a machine
        # that still has Ollama running answers `reachable` perfectly well.
        #
        # There is deliberately no LM Studio counterpart here. Its own
        # POST /api/v1/models/download exists, but the matching progress endpoint does
        # not on 0.4.23 (`GET /api/v1/models/download/status` -> "Unexpected endpoint"),
        # so an install action for it would be a multi-gigabyte download with no
        # progress and no cancel — worse than what LM Studio's own app already does
        # well. Models are downloaded there; the Setup card says so.
        if (caps.get('local_llm') or {}).get('provider', 'ollama') != 'ollama':
            return False
        o = caps.get('ollama') or {}
        return bool(o.get('reachable') and not o.get('vision_model_ready')
                    and (o.get('vision_model') or '').strip())
    if action in _KLEIN_DOWNLOADS:
        # Only into a VALIDATED ComfyUI tree (never scatter multi-GB files under a wrong
        # folder). klein_missing already lists exactly the asset actions still absent
        # (required trio + recommended LoRA).
        c = caps.get('comfyui') or {}
        return bool(c.get('dir_valid')) and action in _broken_or_missing(
            c.get('klein_missing'), c.get('klein_invalid'))
    # The Krea 2 Edit assets are DELIBERATELY absent from this plan even though
    # they are one-click installable everywhere else. "Install everything" runs
    # unattended from a Setup button, and Krea is ~20 GB on top of Klein's ~20 —
    # fetching a SECOND engine nobody asked for is hostile on a metered link or a
    # small disk. Klein is the app's default engine (the generate route falls back
    # to it), Krea is an explicit pick. So Krea installs on intent instead: the
    # per-asset buttons in Setup, the "Install Krea 2 Edit" group button, and the
    # auto-start when a user actually selects the engine and presses Generate
    # (routes/datasets._krea_missing_response) — the same trigger Klein has.
    return False


def install_all_plan(caps) -> list:
    """The ordered list of install actions 'Install everything' will queue for these
    capabilities — every listed MISSING core component whose preconditions are met. Pure and
    deterministic (order = _INSTALL_ALL_ORDER) so it can be tested and drives the global
    progress count. Plugin components remain explicit in their own preparation pages."""
    caps = caps or {}
    # Plugin preparation stays explicit on its owner's page. Keep this boundary even
    # if a future migration leaves a formerly core action in the global ordering.
    return [a for a in _INSTALL_ALL_ORDER
            if a in INSTALL_ACTIONS and a not in _PLUGIN_MANAGED_ACTIONS
            and plugin_action_spec(a) is None
            and not (model_download_spec(a) or {}).get('plugin')
            and known_action(a) and _action_needed(a, caps)]


def start_all(caps) -> dict:
    """Queue every action in install_all_plan(caps). Each start() applies the SAME rules
    as a single install (pip queued FIFO so two never race one venv; model downloads run
    in parallel; per-action preconditions enforced), so this is just a fan-out. An action
    already in flight (AlreadyRunning) reuses its live state; one momentarily unsatisfiable
    (Precondition) is reported as an error row rather than aborting the whole batch. Returns
    the plan + each action's status so the caller can render 'X / N' without re-deriving it."""
    plan = install_all_plan(caps)
    statuses = {}
    for action in plan:
        try:
            statuses[action] = start(action)
        except AlreadyRunning:
            statuses[action] = status(action)
        except (Precondition, ValueError) as e:
            statuses[action] = {'state': 'error', 'returncode': None, 'log': [str(e)],
                                'progress': None, 'waiting_for': None,
                                'manual_command': manual_command(action)}
    return {'plan': plan, 'statuses': statuses}


# --- Named install groups ------------------------------------------------------
# One engine = one button, without dragging that engine into the unattended
# "Install everything" plan. The Krea group is the node pack FIRST (it is a
# ~1 MB clone; getting it out of the way means the only thing left to wait for is
# bytes) then the four weights.
#
# SeedVR2 has NO pack action: its node pack declares thirteen pip dependencies
# that belong in ComfyUI's interpreter, which this app does not own and must
# never pip into (see seedvr2_helper's module docstring). Cloning it alone would
# land a pack that fails to import, so the pack is explained and only the two
# weights are installed here.
_INSTALL_GROUPS = {'krea': ('krea_nodes', 'krea_model', 'krea_text_encoder',
                             'krea_vae', 'krea_identity_lora')}

# Which capabilities keys hold each group's gaps, and which member (if any) is
# the node-pack install. Written down per group rather than branched on the
# group name, so adding the next engine is one row.
_GROUP_CAPS_KEYS = {'krea': {
    'missing': 'krea_missing', 'invalid': 'krea_invalid', 'pack_action': 'krea_nodes',
    'nodes_missing': 'krea_nodes_missing', 'nodes_installed': 'krea_nodes_installed'}}


def install_group_plan(group, caps=None) -> list:
    """The actions a named group would queue: its members MINUS what is already
    installed, in a fixed order. `caps` is the live capabilities payload (each
    group's gaps come from the comfyui.* keys named in _GROUP_CAPS_KEYS); with
    none it plans the whole group. Pure."""
    members = install_groups().get(group)
    if not members:
        return []
    if caps is None:
        return list(members)
    keys = group_caps_keys(group)
    c = (caps or {}).get('comfyui') or {}
    if not c.get('dir_valid'):
        return []                      # nowhere to install into — never guess a path
    missing_assets = _broken_or_missing(c.get(keys['missing']), c.get(keys['invalid']))
    # Does the pack need INSTALLING? Three states, and the difference matters:
    #   on disk                -> no. Missing nodes then mean a ComfyUI RESTART, and
    #                             re-running the installer would only log "already
    #                             installed" and teach the user nothing.
    #   nodes reported missing -> yes.
    #   nodes reported present -> no (a pack installed under another folder name,
    #                             e.g. through the ComfyUI Manager, must not be
    #                             cloned a second time).
    #   ComfyUI unreachable    -> the node probe fails OPEN (it reports nothing
    #                             missing because it could not ask). Not on disk +
    #                             no answer = install it; a stopped ComfyUI must not
    #                             silently drop the pack from a one-click install.
    #   no pack action        -> the group installs weights only.
    if not keys['pack_action']:
        needs_pack = False
    elif c.get(keys['nodes_installed']):
        needs_pack = False
    elif c.get(keys['nodes_missing']):
        needs_pack = True
    else:
        needs_pack = not c.get('reachable')
    return [a for a in members
            if (a == keys['pack_action'] and needs_pack) or a in missing_assets]


def start_group(group, caps=None) -> dict:
    """Prepare a named function, checking node dependencies before its models.

    Plugin groups use the same ownership and preflight gate as an explicit
    preparation batch. Workers revalidate their plans before installation.
    """
    plan = install_group_plan(group, caps)
    if not plan:
        return {'plan': [], 'statuses': {}}
    registry = _plugin_registry()
    plugin_group = registry.install_groups.get(group) if registry else None
    if plugin_group:
        from .plugins import preparation
        from .plugins.loader import external_dir
        return preparation.start(plugin_group['plugin'], {'actions': plan},
                                 registry=registry, root=external_dir())
    for action in plan:
        spec = plugin_action_spec(action) or {}
        preflight = spec.get('node_preflight')
        if callable(preflight):
            try:
                preflight()
            except ValueError as exc:
                raise Precondition(str(exc)) from exc
    statuses = {}
    for action in plan:
        try:
            statuses[action] = start(action)
        except AlreadyRunning:
            statuses[action] = status(action)
        except (Precondition, ValueError) as e:
            statuses[action] = {'state': 'error', 'returncode': None, 'log': [str(e)],
                                'progress': None, 'waiting_for': None,
                                'manual_command': manual_command(action)}
    return {'plan': plan, 'statuses': statuses}


def status_many(actions) -> dict:
    """Per-action status for a set of actions (the live 'Install everything' plan), so the
    UI polls ONE endpoint instead of one request per action. Unknown names are dropped."""
    return {a: status(a) for a in actions if known_action(a)}


def _execute(action):
    try:
        rc = _worker_for(action)(action)
        _finish_run(action, rc, 'success' if rc == 0 else 'error')
        if (action in _IMPORT_CACHE_ACTIONS or plugin_action_spec(action) is not None) and rc == 0:
            try:
                capabilities.clear_import_cache()
            except Exception:
                # never downgrade a successful install; surface at debug only
                logger.debug('clear_import_cache failed after %s', action, exc_info=True)
        if action == 'ollama_model' and rc == 0:
            # A successful vision-model pull must flip the Setup step / diagnostic
            # 'vision model ready' probe NOW, not after the 30 s probe-cache TTL —
            # otherwise the Setup keeps saying "the vision model isn't pulled yet"
            # right after the pull the user just watched finish (issue #7).
            # clear_import_cache() also resets the main probe cache, so it's the one
            # call that forces a fresh /api/tags check on the next probe.
            try:
                capabilities.clear_import_cache()
            except Exception:
                logger.debug('probe-cache clear failed after ollama_model', exc_info=True)
        if (model_download_spec(action) is not None or action in _ALL_NODE_PACKS) and rc == 0:
            # The training-base/model listers cache their scans 5 min and
            # /object_info is cached per API address — a freshly downloaded model
            # (or an installed node pack, once ComfyUI has been restarted) must
            # show up on the next probe, not after the TTL. clear_model_caches
            # drops both, which is exactly why a node-pack install calls it too:
            # otherwise the engine card would keep reporting the OLD node list for
            # minutes after the restart and look like a failed install.
            try:
                from .utils import comfyui
                comfyui.clear_model_caches()
            except Exception:
                logger.debug('clear_model_caches failed after %s', action, exc_info=True)
        if action in _ALL_NODE_PACKS and rc == 0:
            # Both node caches only ever hold a POSITIVE answer, so clearing
            # them regardless of which pack just landed costs one probe each
            # and can never turn a present pack into a missing one.
            try:
                from .services import krea_edit_helper
                krea_edit_helper.clear_nodes_cache()
            except Exception:
                logger.debug('krea node-cache clear failed after %s', action, exc_info=True)
            try:
                from .services import lanpaint_helper
                lanpaint_helper.clear_nodes_cache()
            except Exception:
                logger.debug('lanpaint node-cache clear failed after %s', action, exc_info=True)
            try:
                from .services import krea_sampler_helper
                krea_sampler_helper.clear_nodes_cache()
            except Exception:
                logger.debug('krea sampler node-cache clear failed after %s', action,
                             exc_info=True)
    except Cancelled:
        _append(action, 'cancelled by user')
        _finish_run(action, None, 'cancelled')
    except Exception as e:  # never let a worker thread die silently
        _append(action, f'error: {e}')
        _finish_run(action, -1, 'error')
    finally:
        # Always hand the pip worker to the next queued install, even on failure — a
        # crashed install must not wedge the queue behind it.
        if _is_pip_action(action):
            _release_pip_slot(action)


def _run_pip(action, cmd, *, env=None) -> int:
    """Run a pip command, streaming its output to the ring log, with a bounded retry
    on a TRANSIENT file-lock error (an antivirus/indexer holding a just-written file —
    Errno 13 / WinError 5|32|2). pip is idempotent, so a rerun finishes the interrupted
    step. A genuine build/resolution failure doesn't match _RETRYABLE_PIP_ERR and is
    returned immediately. Concurrency is already prevented by the pip queue; this is the
    single-process defence (the Bitdefender-style lock users without a queue still hit).
    Plugin environments pass their sanitized env through every retry; callers
    without one retain the historical process environment."""
    if env is None and action in ('ml_extras', 'face_scoring', 'masks', 'video_text',
                                  'watermark_inpaint', 'bank_scoring',
                                  'bank_siglip2', 'watermark_detect', 'shot_detect'):
        env = managed_python.subprocess_env()
    cmd, env = plugin_tool_sources.pip_install(cmd, env)
    rc = -1
    for attempt in range(1, _PIP_RETRIES + 1):
        buf = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for line in proc.stdout:
            _append(action, line)
            buf.append(line)
        proc.wait()
        rc = proc.returncode
        if rc == 0:
            return 0
        if attempt < _PIP_RETRIES and any(_RETRYABLE_PIP_ERR.search(l) for l in buf):
            wait = _PIP_RETRY_BACKOFF * attempt
            _append(action, f'transient file-lock error (an antivirus or indexer may be '
                            f'holding a fresh file); retrying in {wait}s '
                            f'[{attempt}/{_PIP_RETRIES - 1}]')
            time.sleep(wait)
            continue
        return rc
    return rc


def _run_ml_extras(action) -> int:
    """Legacy shared-quality button; heavy engines remain explicit selections."""
    if action != 'ml_extras':
        return _run_pip(action, [sys.executable, '-m', 'pip', 'install', '-r',
                                 str(_PIP_REQUIREMENTS[action])])
    return _install_quality_tools(action, ('face_scoring', 'masks'))


# --- Auto-provisioned watermark venv -------------------------------------------
# simple-lama-inpainting hard-requires Pillow<10, so it can never share the app's
# Pillow-12 venv. When the user hasn't pointed watermark.python at a dedicated
# 3.10-3.12 interpreter, the Install button BUILDS one for them: find a base Python
# 3.10-3.12 on the machine, create an isolated venv under the app's data dir, install
# CPU torch + simple-lama-inpainting into it, and record its interpreter as
# watermark.python so the probe + wrapper resolve there. No manual venv, no setting to
# edit. Idempotent: a re-click reuses/repairs the same venv; a user's own
# watermark.python is always respected and never overwritten.
_VENV_PY_MIN = (3, 10)   # mirrors capabilities._ML_PY_MIN/_MAX (the ML wheel range):
_VENV_PY_MAX = (3, 12)   # torch / simple-lama publish wheels for CPython 3.10-3.12.
# CPU torch, installed EXPLICITLY into the managed venv: reliable and small on every OS
# (no CUDA toolkit, no multi-GB download), and watermark inpainting only repaints small
# masked regions where CPU is fine. watermark.device='auto' resolves to CPU when CUDA is
# absent, so the env works with zero config. A user who wants GPU points watermark.python
# at their own CUDA env — where we DON'T force CPU torch (we never downgrade their build).
_TORCH_CPU_INDEX = plugin_tool_sources.CPU
# Budget for the post-install verification import (see _verify_watermark_import). Far
# longer than the capability probe's 60 s ceiling on purpose: importing simple-lama pulls
# in torch + torchvision + opencv (~430 MB of native code, a single 291 MB torch_cpu.dll),
# and the FIRST cold import on a fresh machine — real-time AV scanning brand-new DLLs — can
# run minutes. We pay that once, here, so the probe fired right after the install is warm.
_WARM_IMPORT_TIMEOUT = 300


def _install_cpu_torch_pair(action, python, *, constraints=None) -> int:
    """Install torch AND torchvision together from _TORCH_CPU_INDEX into a managed
    environment. Always the PAIR, never torch alone: the stacks that land in these
    envs afterwards (open_clip_torch, timm, simple-lama-inpainting) depend on
    torchvision, and left to pip that torchvision resolves from PyPI — where the
    Linux wheel is built against a DIFFERENT torch than the CPU-index one already
    present. The mismatch imports into `RuntimeError: operator torchvision::nms
    does not exist` and the whole env is unusable (reported from the GPU Docker
    image, whose rebuilt bank-scoring env failed exactly this way; Dockerfile.gpu
    names the same trap for the image venv and pairs them for the same reason).
    Windows never surfaced it because PyPI's Windows torchvision wheels are CPU
    builds. One index, both names: pip resolves a matched pair, and the call is a
    no-op when a matched pair is already there."""
    _append(action, 'installing CPU torch + torchvision '
                    'from the LDS CPU wheel index if needed')
    cmd = [python, '-m', 'pip', 'install', 'torch', 'torchvision',
           '--index-url', _TORCH_CPU_INDEX]
    if constraints:
        # A PATH, never the raw requirements file: the one caller that constrains
        # this install (the watermark env) must not inherit the app's Pillow pin.
        cmd += ['-c', str(constraints)]
    rc = _run_pip(action, cmd)
    if rc != 0:
        _append(action, f'torch install failed (rc={rc}) — see the log above')
    return rc


# `python -m venv` seeds a new env with the pip BUNDLED IN THE BASE PYTHON, not a
# current one. A 3.10 base ships pip 21.x, which rejects several of today's wheels
# ("inconsistent Name: expected 'typing-extensions', but metadata has
# 'typing_extensions'"), falls back to their sdists, and then cannot even build
# those against the CPU torch index (it has no flit_core) — all four optional
# installs died exactly this way on one user's machine. pip normalises that name
# check from 23.x, so anything older gets ONE upgrade. Checked on creation AND on
# reuse: existing installs out there already carry the old pip.
_MIN_PIP = (23, 1)


def _pip_version(python, *, isolated_env=None):
    """(major, minor) of `python`'s pip — read by RUNNING it — or None when it
    cannot be read (missing/fake interpreter, no pip module)."""
    try:
        cmd = ([python, '-I', '-m', 'pip', '--isolated', '--version'] if isolated_env is not None
               else [python, '-m', 'pip', '--version'])
        proc = subprocess.run(cmd, env=isolated_env,
                              capture_output=True, text=True, timeout=processing_timeout(30),
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    parts = (proc.stdout or '').split()
    if len(parts) < 2 or parts[0] != 'pip':
        return None
    nums = parts[1].split('.')
    try:
        major = int(nums[0])
        minor = int(nums[1]) if len(nums) > 1 else 0
    except ValueError:
        return None
    return major, minor


def _ensure_modern_pip(action, python, *, isolated_env=None) -> None:
    """Upgrade a MANAGED venv's pip once when it is too old for current wheels.

    Only ever called on the app's own venvs — never on a user-configured
    interpreter, whose environment is theirs. Non-fatal by design: an unreadable
    version or a failed upgrade logs and moves on, leaving the install to behave
    exactly as it did before this guard existed. Plugin environments pass
    isolated_env to isolate both the version probe and the upgrade."""
    ver = (_pip_version(python, isolated_env=isolated_env) if isolated_env is not None
           else _pip_version(python))
    if ver is None or ver >= _MIN_PIP:
        return
    _append(action, f'pip {ver[0]}.{ver[1]} in this environment is too old for '
                    "today's packages — upgrading it once")
    if isolated_env is not None:
        rc = _run_pip(action, [python, '-I', '-m', 'pip', '--isolated', 'install', '--upgrade', 'pip'],
                      env=isolated_env)
    else:
        rc = _run_pip(action, [python, '-m', 'pip', 'install', '--upgrade', 'pip'])
    if rc != 0:
        _append(action, 'pip upgrade failed — continuing with the bundled pip '
                        '(the install may still hit the old-pip wheel refusal)')


def _watermark_env_dir():
    """The app-managed watermark venv directory (deterministic, under the data dir), so
    a re-click resolves the SAME venv — idempotent build/repair, never a duplicate."""
    return cfg.data_dir() / 'envs' / 'watermark'


def _venv_python(env_dir) -> str:
    return str(env_dir / ('Scripts' if os.name == 'nt' else 'bin')
              / ('python.exe' if os.name == 'nt' else 'python'))


def _watermark_env_python() -> str:
    """Absolute path to the app-managed watermark venv's python (may not exist yet)."""
    return _venv_python(_watermark_env_dir())


def _same_path(a, b) -> bool:
    """True when two paths point at the same interpreter ENVIRONMENT. Venv pythons
    compare by the venv directory that owns them (see _venv_root) — never by
    resolving the binary, which on Linux collapses every venv into its symlinked
    base and made a borrowed interpreter indistinguishable from the managed env.
    Non-venv paths keep samefile when both exist, else a case/separator-insensitive
    compare (so a not-yet-built venv path matches)."""
    ra, rb = _venv_root(a or ''), _venv_root(b or '')
    if ra or rb:
        return bool(ra) and bool(rb) and ra == rb
    try:
        return os.path.samefile(a, b)
    except OSError:
        return (os.path.normcase(os.path.abspath(a or ''))
                == os.path.normcase(os.path.abspath(b or '')))


def _python_minor(exe: str):
    """(major, minor) reported by RUNNING `exe` — never trusted from its name/path —
    or None when it can't be executed. Short timeout, no console window."""
    try:
        proc = subprocess.run(
            [exe, '-I', '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'],
            capture_output=True, text=True, timeout=processing_timeout(15),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    m = re.match(r'^(\d+)\.(\d+)\s*$', proc.stdout or '')
    return (int(m.group(1)), int(m.group(2))) if m else None


def _base_python_candidates() -> list:
    """Interpreters to try as the BASE for `-m venv`, in reliability order. Names/paths
    only — each is version-checked by EXECUTION before use. We never install into these;
    we only spawn an isolated venv from one (its site-packages are never touched)."""
    cands = []
    if os.name == 'nt':
        # 1. Windows launcher: explicit 3.12 > 3.11 > 3.10 (resolve the tag to a path).
        launcher = shutil.which('py')
        if launcher:
            for tag in ('3.12', '3.11', '3.10'):
                try:
                    p = subprocess.run([launcher, f'-{tag}', '-c',
                                        'import sys; print(sys.executable)'],
                                       capture_output=True, text=True, timeout=processing_timeout(15),
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    exe = (p.stdout or '').strip()
                    if p.returncode == 0 and exe:
                        cands.append(exe)
                except (OSError, subprocess.SubprocessError):
                    pass
    # 2. On PATH.
    for name in ('python3.12', 'python3.11', 'python3.10', 'python3', 'python'):
        exe = shutil.which(name)
        if exe:
            cands.append(exe)
    # 3. Standard per-user / system install locations (Windows).
    if os.name == 'nt':
        for root in (os.environ.get('LOCALAPPDATA', ''), os.environ.get('PROGRAMFILES', ''),
                     os.environ.get('PROGRAMFILES(X86)', ''), 'C:\\'):
            if not root:
                continue
            for ver in ('312', '311', '310'):
                cands.append(os.path.join(root, 'Programs', 'Python', f'Python{ver}', 'python.exe'))
                cands.append(os.path.join(root, f'Python{ver}', 'python.exe'))
    # 4. Pythons the app already knows — used ONLY as a venv base. sys.executable (the
    #    app's own 3.12 venv) is a perfect base on a portable-bundle machine that has no
    #    other Python installed: `-m venv` from it makes a fresh, empty env, so the app's
    #    Pillow 12 is never touched.
    cands.append(sys.executable)
    for key in ('face_scoring.python', 'masks.python'):
        v = (cfg.get(key) or '').strip()
        if v:
            cands.append(v)
    try:
        ai = cfg.aitoolkit_path('venv_python')
        if ai:
            cands.append(str(ai))
    except Exception:
        pass
    # Dedupe, preserving order (normcase for Windows path equality).
    seen, out = set(), []
    for c in cands:
        key = os.path.normcase(os.path.abspath(c)) if c else ''
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _find_base_python(action) -> str:
    """Reuse a compatible local base, otherwise provision our private Python."""
    for exe in _base_python_candidates():
        ver = _python_minor(exe)
        if ver is not None and _VENV_PY_MIN <= ver <= _VENV_PY_MAX:
            _append(action, f'found base Python {ver[0]}.{ver[1]}: {exe}')
            return exe
    try:
        return managed_python.ensure_python(lambda line: _append(action, line))
    except Exception as exc:
        _append(action, f'Could not prepare the managed Python: {exc}')
        _append(action, 'Nothing was installed in the app Python. Check the connection '
                        'and free disk space, then click Install again to retry.')
        return ''


def _ensure_watermark_env(action) -> str:
    return _ensure_managed_ml_env(action, _watermark_env_dir())


# The pins requirements-ml.txt carries FOR THE APP that this environment must
# never inherit. Pillow is the whole reason the watermark venv exists: the app
# needs 12 (the burned-in-text reader's unicode-path fallback), and
# simple-lama-inpainting refuses anything above 9. Handing pip both at once is
# an unsatisfiable request, and pip answers with ResolutionImpossible naming
# "The user requested (constraint) pillow<13,>=12" — GitHub #59, on an install
# that had worked until the app pinned its own Pillow in requirements-ml.txt.
# The constraint file is still worth passing: it is what keeps a torch pull from
# bumping numpy past insightface's <2 ceiling in a user's OWN environment.
_WATERMARK_CONSTRAINT_EXCLUDES = frozenset({_canon('pillow')})


@contextlib.contextmanager
def _watermark_constraint_file():
    """requirements-ml.txt MINUS the pins the watermark env cannot honour.

    Yields a path to a temporary constraints file, or None when the source file
    cannot be read (then the caller passes no -c at all — an unconstrained
    install still works, a crashed one does not).

    A filtered COPY rather than a second checked-in file on purpose: the floors
    stay in one place, so a version bump in requirements-ml.txt reaches this
    environment too, and nobody has to remember a parallel list exists.
    """
    lines = _ml_requirement_specs(exclude=_WATERMARK_CONSTRAINT_EXCLUDES)
    if not lines:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix='lds-watermark-') as tmp:
        path = os.path.join(tmp, 'constraints.txt')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        yield path


def _pip_install_watermark(action, python, *, managed: bool) -> int:
    """Install simple-lama-inpainting into `python` (a dedicated 3.10-3.12 env). The
    version floor is read from requirements-ml.txt (single source of truth), which also
    rides along as a -c constraint so pulling torch can't bump numpy past insightface's
    <2 ceiling — MINUS the app's own Pillow pin, which this environment exists to
    contradict (see _WATERMARK_CONSTRAINT_EXCLUDES). For the app-managed venv
    (managed=True) CPU torch is installed FIRST and explicitly (small/reliable/
    cross-OS); a user's OWN env keeps whatever torch it has — we never downgrade a
    CUDA build there."""
    spec = _requirement_spec(_WATERMARK_PKG)
    _append(action, f'target interpreter: {python}')
    with _watermark_constraint_file() as constraints:
        if managed:
            # simple-lama-inpainting depends on torchvision, so the pair matters here
            # exactly as it does for the bank-scoring stack (see _install_cpu_torch_pair).
            rc = _install_cpu_torch_pair(action, python, constraints=constraints)
            if rc != 0:
                return rc
        _append(action, f'installing {spec}  (constraints: requirements-ml.txt '
                        'without the app Pillow pin)')
        cmd = [python, '-m', 'pip', 'install', spec]
        if constraints:
            cmd += ['-c', constraints]
        return _run_pip(action, cmd)


def _verify_watermark_import(action, python) -> bool:
    """A package install is ready only after the actual worker import succeeds."""
    return _verify_capability_import('watermark_inpaint', python, log_action=action)


def _run_watermark_inpaint(action) -> int:
    """Repair LDS's isolated LaMa environment, preserving borrowed selections."""
    python = _ensure_watermark_env(action)
    if not python:
        return 1
    rc = _pip_install_watermark(action, python, managed=True)
    if rc != 0 or not _verify_watermark_import(action, python):
        return rc or 1
    return 0 if _select_managed_python(action, 'watermark', python) else 1


# --- Auto-provisioned bank-scoring venv ----------------------------------------
# The CLIP + NSFW stack (torch, open_clip, transformers, timm) is heavy and
# version-touchy, so it lives in its OWN app-managed venv rather than the Flask
# venv — same isolation and one-click build/repair as the watermark venv.
def _bank_scoring_env_dir():
    return cfg.data_dir() / 'envs' / 'bank_scoring'


def _bank_scoring_env_python() -> str:
    return _venv_python(_bank_scoring_env_dir())


def _ensure_bank_scoring_env(action, *, save_score_python=True) -> str:
    # Keep the keyword for product callers. Selection is saved by the installing
    # action only AFTER its imports pass, never while creating an empty venv.
    return _ensure_managed_ml_env(action, _bank_scoring_env_dir())


def _run_bank_scoring(action) -> int:
    """Install the bank-scoring stack (CPU torch + open_clip + transformers + timm)
    into the app's OWN bank-scoring venv — never the Flask venv, and never an
    environment the app did not build. A borrowed ``bank_scoring.python`` is only
    a runtime selection (written by the GPU picker): installing or repairing the
    managed environment keeps that selection intact and never invokes pip in it.
    Verifies the managed import at the end so a pip-success-but-import-fail never
    reports success (same honesty gate as the watermark install)."""
    managed_python = _bank_scoring_env_python()
    configured = (cfg.get('bank_scoring.python') or '').strip()
    borrowed = bool(configured) and not _same_path(configured, managed_python)
    python = _ensure_bank_scoring_env(
        action, save_score_python=not borrowed)
    if not python:
        return 1
    if not _same_path(python, managed_python):
        _append(action, 'internal error: Bank scoring did not resolve to the '
                        'LDS-managed environment; nothing was installed')
        return 1
    if borrowed:
        _append(action, f'keeping the selected borrowed Score interpreter unchanged: '
                        f'{configured}')
        _append(action, 'Install/repair targets only the LDS-managed environment below.')
    # Past this point the target is always the app-managed venv.
    _append(action, f'target interpreter: {python}')
    rc = _install_cpu_torch_pair(action, python)
    if rc != 0:
        return rc
    specs = _bank_scoring_specs()
    _append(action, f"installing {', '.join(specs)}")
    rc = _run_pip(action, [python, '-m', 'pip', 'install', *specs])
    if rc == 0 and not _verify_bank_scoring_import(action, python):
        return 1
    if rc == 0 and not _select_managed_python(action, 'bank_scoring', python):
        return 1
    return rc


def _verify_bank_scoring_import(action, python) -> bool:
    """A package install is ready only after the actual worker import succeeds."""
    return _verify_capability_import('bank_scoring', python, log_action=action)


def _bank_semantic_install_python() -> str:
    """Where SigLIP2 is INSTALLED. Always the app-managed Bank ML venv.

    Deliberately takes no argument and reads no interpreter key: ``bank_scoring.python``
    and ``bank_semantic.python`` say where a pass RUNS, and both can point at an
    environment the user built (ai-toolkit's, ComfyUI's). Installing into one of
    those is the one thing this app never does, so the install target is derived
    from the data folder and nothing else. Enforced by
    ``test_bank_siglip2_install_ignores_borrowed_semantic_interpreter``."""
    return _bank_scoring_env_python()


def _run_bank_siglip2(action) -> int:
    """Install the optional SigLIP2 semantic engine and its pinned checkpoint.

    It always targets LDS's managed Bank ML venv. Score — and now the semantic
    index itself — may keep using a borrowed CUDA interpreter: this action
    neither installs into it nor repoints it. ``bank_semantic.python`` is only
    written when nothing was borrowed, and only after packages and every pinned
    weight are ready; a user who chose a GPU Python for the index keeps it.
    """
    from .services import bank_semantic_models as assets

    managed_python = _bank_semantic_install_python()
    configured = (cfg.get('bank_semantic.python') or '').strip()
    borrowed = bool(configured) and not _same_path(configured, managed_python)

    python = _ensure_bank_scoring_env(action, save_score_python=False)
    if not python:
        return 1
    if not _same_path(python, managed_python):
        _append(action, 'internal error: SigLIP2 did not resolve to the LDS-managed '
                        'Bank environment; nothing was installed')
        return 1
    if borrowed:
        _append(action, f'keeping the selected borrowed semantic interpreter '
                        f'unchanged: {configured}')
        _append(action, 'Install/repair targets only the LDS-managed environment below.')

    _append(action, f'target interpreter: {python}')
    rc = _install_cpu_torch_pair(action, python)
    if rc != 0:
        return rc
    rc = _run_pip(action, [python, '-m', 'pip', 'install',
                           'transformers>=4.49', 'huggingface_hub', 'safetensors',
                           'sentencepiece', 'Pillow'])
    if rc != 0:
        return rc
    if not _verify_capability_import(action, python):
        return 1

    root = assets.models_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _append(action, f'could not create {root}: {e}')
        return 1
    _append(action, f'downloading {assets.MODEL_ID} (~{assets.DOWNLOAD_MB} MB, '
                    f'Apache-2.0) to {root}')
    _append(action, f'pinned revision: {assets.REVISION}')
    code = (
        'import json, sys\n'
        'from huggingface_hub import hf_hub_download\n'
        'repo, revision, root, files = json.loads(sys.argv[1])\n'
        'for name in files:\n'
        '    hf_hub_download(repo_id=repo, filename=name, revision=revision, cache_dir=root)\n'
        'print("ok")\n'
    )
    payload = json.dumps([
        assets.MODEL_ID, assets.REVISION, str(root), list(assets.FILES)])
    rc = _run_pip(action, [python, '-c', code, payload])
    if rc != 0:
        _append(action, 'SigLIP2 download did not finish — click Install again to resume')
        return rc
    if not assets.weights_present(root):
        _append(action, 'download returned success but at least one pinned model file is missing')
        return 1
    if not _select_managed_python(action, 'bank_semantic', managed_python):
        return 1
    _append(action, 'SigLIP2 ready — each Bank can now choose it without deleting CLIP')
    return 0


def _watermark_detect_python() -> str:
    """The install/repair target, independent of the selected runtime."""
    return _bank_scoring_env_python()


def _run_watermark_detect(action) -> int:
    """Install the dedicated watermark DETECTOR: the packages (torch +
    transformers, into the app's own bank-scoring venv) and then the weights.

    Both halves matter and they fail differently. Packages missing = "the extra
    is not installed". Weights missing = "it is installed but the first scan will
    die on a network error an hour in" — which is why the capability probe checks
    the model cache too, and why this worker downloads them here rather than
    lazily on first use."""
    managed_python = _bank_scoring_env_python()
    configured = (cfg.get('watermark_detect.python') or '').strip()
    if configured and not _same_path(configured, managed_python):
        _append(action, 'The selected external detector runtime is read-only; '
                        'installing into the managed Bank environment.')
    python = _ensure_bank_scoring_env(action, save_score_python=False)
    if not python:
        return 1
    if _is_flask_venv(python):
        for line in (
            "The detector needs torch, which never installs into the app's own Python.",
            'Nothing was installed. Clear watermark_detect.python and click Install',
            'again — the app builds a dedicated Python for you.',
        ):
            _append(action, line)
        return 1
    _append(action, f'target interpreter: {python}')
    rc = _install_cpu_torch_pair(action, python)
    if rc != 0:
        return rc
    rc = _run_pip(action, [python, '-m', 'pip', 'install', 'transformers',
                           'huggingface_hub', 'safetensors'])
    if rc != 0:
        return rc
    if not _verify_watermark_detect_import(action, python):
        return 1
    rc = _download_watermark_detect_models(action, python)
    if rc != 0:
        return rc
    return 0 if _select_managed_python(action, 'watermark_detect', python) else 1


def _verify_watermark_detect_import(action, python) -> bool:
    """The detector must import successfully before Setup can publish it ready."""
    return _verify_capability_import('watermark_detect', python, log_action=action)


def _download_watermark_detect_models(action, python) -> int:
    """Fetch the two model repos, NAMING every file.

    Never a bare snapshot_download. Measured on 2026-08-03: the SigLIP2 repo also
    publishes its training checkpoints (checkpoint-712/, checkpoint-1424/,
    optimizer.pt) next to a model that weighs 371 MB, so a whole-repo pull costs
    2.4 GB, and the Grounding DINO repo would add a duplicate pytorch_model.bin.
    Listing the files brings the download to ~0.9 GB — for identical behaviour."""
    from .services import watermark_detector
    root = watermark_detector.models_root()
    if not root:
        _append(action, 'could not resolve where to store the detector weights')
        return 1
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as e:
        _append(action, f'could not create {root}: {e}')
        return 1
    _append(action, f'downloading the detector weights (~{watermark_detector.DOWNLOAD_MB} MB) '
                    f'to {root}')
    for repo, meta in watermark_detector.MODEL_FILES.items():
        _append(action, f'  {repo} — {meta["license"]}, {meta["role"]}')
        code = (
            'import sys, json\n'
            'from huggingface_hub import hf_hub_download\n'
            'repo, root, files = json.loads(sys.argv[1])\n'
            'for name in files:\n'
            '    hf_hub_download(repo_id=repo, filename=name, cache_dir=root)\n'
            'print("ok")\n'
        )
        payload = json.dumps([repo, root, list(meta['files'])])
        rc = _run_pip(action, [python, '-c', code, payload])
        if rc != 0:
            _append(action, f'could not download {repo} (rc={rc}) — the detector stays '
                            'unavailable and the vision model keeps doing the work')
            return rc
    _append(action, 'weights ready — 🚩 Find watermarks now uses the detector')
    return 0


def _run_ml_capability(action) -> int:
    """Quality tools install in managed Python; in-process extras keep their owner."""
    if action in _CAPABILITY_ML_ACTIONS:
        return _install_quality_tools(action, (action,))
    python = _capability_python(action)
    specs = _drop_provided_onnxruntime(
        action, python, [_requirement_spec(p) for p in _CAPABILITY_PACKAGES[action]])
    _append(action, f'target interpreter: {python}')
    _append(action, f"installing {', '.join(specs)}  (constraints: requirements-ml.txt)")
    # When this capability targets the Flask venv (no dedicated python), pin Pillow
    # so pulling insightface/rembg deps can't downgrade the app's Pillow either.
    rc = _run_pip(action, [python, '-m', 'pip', 'install', *specs,
                           '-c', str(_ML_REQUIREMENTS), *_flask_pillow_guard(python)])
    if rc == 0 and not _verify_capability_import(action, python):
        return 1
    return rc


# onnxruntime ships under several DIFFERENT distribution names that all provide
# the same `onnxruntime` module and cannot coexist in one environment:
# onnxruntime (CPU), onnxruntime-gpu (CUDA), onnxruntime-directml,
# onnxruntime-silicon. pip does not know they conflict, so `pip install
# onnxruntime` into an env that already has the GPU build "succeeds" and quietly
# leaves the user on CPU — a performance regression they would never be told
# about. Any variant satisfies rembg and insightface equally well, so the rule is
# simple: we only add onnxruntime when the target interpreter cannot import one.
_ONNXRUNTIME_CANON = _canon('onnxruntime')
# An `import onnxruntime` that is going to fail fails instantly (there is nothing
# to load). A slow one means a real, large runtime IS being loaded. So this probe
# is short on purpose, and a timeout counts as PRESENT.
_ONNXRUNTIME_PROBE_TIMEOUT = 60


def _onnxruntime_provided(python) -> bool:
    """Can `python` already import onnxruntime (under ANY distribution name)?"""
    if not os.path.isfile(python):
        return False
    try:
        proc = subprocess.run([python, '-c', 'import onnxruntime'],
                              capture_output=True, timeout=processing_timeout(_ONNXRUNTIME_PROBE_TIMEOUT),
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        return True    # slow import == a real runtime is loading; do not overwrite it
    except Exception:
        # Could not ask at all (unlaunchable interpreter, …) -> install it. A missing
        # runtime is the bug being fixed here; a broken interpreter has bigger
        # problems than which onnxruntime build it carries. Never raises: this only
        # shapes a pip command line.
        return False
    return proc.returncode == 0


def _drop_provided_onnxruntime(action, python, specs) -> list:
    """Remove the onnxruntime requirement when the target env already provides one.
    Idempotent by design: on the very many installs that already have onnxruntime
    (every machine where face scoring or the monolithic ML extras were installed)
    this makes the added dependency a no-op instead of a reinstall, and on a
    machine carrying onnxruntime-gpu it protects that build."""
    keep, dropped = [], False
    for spec in specs:
        name = re.split(r'[<>=!~;\[\s]', spec, maxsplit=1)[0]
        if _canon(name) == _ONNXRUNTIME_CANON and _onnxruntime_provided(python):
            dropped = True
            continue
        keep.append(spec)
    if dropped:
        _append(action, 'onnxruntime already imports in this environment '
                        '(any of the CPU / GPU / DirectML builds works) — leaving it '
                        'untouched so a GPU build is not replaced by the CPU one.')
    return keep


def _verify_capability_import(action, python, *, log_action=None) -> bool:
    """Re-run the capability's OWN probe import once pip reports done, and say what
    happened in the install log.

    This is the honesty gate for the scoped ML installs. pip's "Requirement already
    satisfied" proves distributions are on disk; it proves nothing about whether the
    feature loads. A masks install could therefore report success, every package
    resolved, while `import rembg` died on a runtime nobody had listed — leaving
    "✓ installed successfully" next to "✗ Not installed" with no reason anywhere
    (issue #24, 1Tomber). The import expression comes from
    capabilities.CAPABILITY_IMPORTS, i.e. literally the one the probe runs, so the
    two can never drift apart again.

    An import cannot speak for everything an action installs, though. When the
    action delivers something else too — `video` ships a BINARY next to its
    package — it registers an extra check in _CAPABILITY_EXTRA_CHECKS and this
    gate runs it after the import. Per action, never globally: every other
    capability IS fully described by its import and must not be made stricter as
    a side effect.

    It also WARMS the import, like the watermark/bank verifications: the capability
    probe fires seconds later and its first cold import can be slow enough to time
    out and read ✗ on a perfectly good install.

    Only a completed, successful import is ready. Missing interpreters, timeouts
    and launch failures leave the action retryable and cannot publish a selection."""
    expr = capabilities.CAPABILITY_IMPORTS.get(action)
    capability = action
    action = log_action or action
    if not expr or not os.path.isfile(python):
        _append(action, 'The verification interpreter or import is missing; retry Install.')
        return False
    _append(action, 'verifying the install (running the same import the capability '
                    'check runs — this also warms it, so it turns green without a restart)…')
    try:
        proc = subprocess.run(infer_env.worker_argv(python, '-c', expr),
                              capture_output=True, text=True,
                              encoding='utf-8', errors='replace',
                              timeout=processing_timeout(_WARM_IMPORT_TIMEOUT),
                              env=infer_env.worker_env(python),
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        _append(action, 'still warming up (the first import is slow on a fresh machine) — '
                        'verification did not finish; click Install again to retry')
        return False
    except Exception as e:
        _append(action, f'could not run the verification import ({e}) — retry Install')
        return False
    if proc.returncode == 0:
        extra = _CAPABILITY_EXTRA_CHECKS.get(capability)
        if extra is None:
            _append(action, f'import OK — {_CAPABILITY_LABEL.get(action, action)} is ready')
            return True
        # This action delivers something an import cannot answer for. Saying
        # "ready" here and then failing two lines down is the confusion this
        # gate exists to remove, so the import result is announced as the HALF it
        # actually proves, and the check that owns the other half speaks next.
        _append(action, f'import OK — {_CAPABILITY_LABEL.get(action, action)} loads; '
                        f'now checking the rest of what this install promises…')
        return extra(action, python)
    stderr = proc.stderr or ''
    _append(action, f'pip finished, but {_CAPABILITY_LABEL.get(action, action)} still does '
                    f'not load in this environment — the capability stays OFF:')
    missing = _MISSING_MODULE_RE.search(stderr)
    if missing:
        # The single most useful line in the whole chain, and the one the old
        # flow threw away: WHICH module is missing. Lead with it — the stderr
        # tail below is the proof, this is the answer.
        _append(action, f"  missing module: {missing.group(1)} — it is not installed in "
                        f"{python}. Click Install again to retry the managed install.")
    for line in stderr.strip().splitlines()[-4:]:
        _append(action, f'  {line}')
    return False


_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
_CAPABILITY_LABEL = {'face_scoring': 'face scoring', 'masks': 'person masks',
                      'bank_scoring': 'bank scoring',
                      'bank_siglip2': 'SigLIP2 Bank semantics',
                      'watermark_inpaint': 'watermark inpainting',
                      # `video` is TWO halves and the import proves only the
                      # first, so its label names that half and never the extra.
                      'video': 'video decoding (PyAV)'}


def _verify_video_encoder(action, python) -> bool:
    """The OTHER half of the `video` extra: an ffmpeg binary that runs.

    `video` installs two things — PyAV for reading and imageio-ffmpeg for writing
    — and an import can only speak for the first. So the action could install a
    package whose bundled binary never arrived, watch `import av` succeed, and
    announce "✓ installed successfully" while Setup's "Video bank — clip
    encoding" row stayed ✗ behind the very same ↻ button. The user then reruns
    the install that already worked, which is exactly the wrong-half reinstall
    probe_video() splits its three rows to prevent.

    Judged by capabilities' own definition (ffmpeg_tools.ffmpeg_ready), in the
    process that will do the judging later — resolution happens IN FLASK, not in
    `python`, because PyAV is what lives in the capability's interpreter while
    ffmpeg is resolved in-process by whoever encodes.
    """
    # pip just wrote a package into this interpreter's site-packages; without
    # this the still-running process can keep a cached "no such module" view of
    # that directory and report a missing encoder that is in fact installed.
    importlib.invalidate_caches()
    from .services import ffmpeg_tools
    status = ffmpeg_tools.ffmpeg_ready(force=True)
    if status['ok']:
        _append(action, f"clip encoding ready — ffmpeg at "
                        f"{redact_user_paths(status['path'] or '')}")
        return True
    _append(action, 'HALF INSTALLED: your videos can be read, but clips cannot be '
                    'ENCODED yet, so a bank still cannot be exported to a dataset.')
    _append(action, f"  {status['reason']}")
    if not _same_path(python, sys.executable):
        # A dedicated video.python cannot fix encoding: Flask resolves ffmpeg in
        # its OWN process. Worth saying, because the install genuinely succeeded.
        _append(action, '  note: imageio-ffmpeg went into the interpreter above, but the '
                        "app resolves ffmpeg inside its own Python — clear video.python, "
                        'or put ffmpeg on PATH, so the encoder is visible to the app.')
    _append(action, '  fix: run this install again with network access (imageio-ffmpeg '
                    'fetches its binary), or install ffmpeg yourself and put it on '
                    'PATH — then click ↻ once more. Decoding stays available either way.')
    return False


# Post-install checks that go BEYOND the probe import, per action. A dict and not
# a branch inside the gate: every other capability is fully described by its
# import, and must not become stricter because this one is not.
_CAPABILITY_EXTRA_CHECKS = {'video': _verify_video_encoder}


def _is_blocking_invalid(path, spec) -> bool:
    """Is the file at `path` a declared superseded revision or impossible to load
    (an HTML licence page, a truncated/garbage download)? `too_small` is NOT counted, and a checker
    that cannot answer says False — no skip is ever turned into a re-download on a
    guess.

    This is the "a file resolves, therefore the asset is installed" hole, and it had
    FOUR doors. 54e5011 shut the one at `dest`; the other three below skip the
    download because SOME OTHER file resolves (a legacy filename, a file under an
    extra_model_paths root, a hand-placed Krea asset) and none of them looked at
    that file either — so the corrupted-weight dead end simply came back through a
    different door. Same validator, same rule, all four."""
    try:
        if os.path.getsize(path) in spec.get('superseded_bytes', ()):
            return True
        from .services import model_integrity
        res = model_integrity.validate_model_file(path, min_bytes=spec.get('min_bytes'))
    except Exception:
        logger.debug('integrity check failed for %s', path, exc_info=True)
        return False
    return bool(res['blocking'])


def _download_present_in_extra(action) -> bool:
    """Is the asset for `action` already on disk under an extra_model_paths.yaml
    root? We still DOWNLOAD into the base is-default tree (dest is unchanged, per the
    "install location doesn't move" rule) — this only skips a redundant multi-GB fetch
    when the file already lives somewhere ComfyUI will load it. Accepts the canonical
    filename AND any earlier default name (`legacy_names`): an install that fetched the
    pre-KV UNET into an extra root still resolves it by name, so it must not re-download.
    EXTRA roots only (base presence is the os.path.isfile(dest) + _variant_already_present
    checks), so with no yaml this is a no-op and behaviour is identical.

    A blocking-invalid file out there does NOT count as present — it is exactly the
    file the loader would open, so skipping on it leaves the user in the dead end
    they came to Setup to escape. Nothing under a user's own extra root is deleted
    though: the download lands in the base dest as always, and the broken copy is
    named in the log so it can be removed by hand (deleting inside a tree the app
    does not own is a bigger promise than this function should make)."""
    spec = model_download_spec(action)
    dest_parts = spec['dest']                 # e.g. ('unet','klein','flux-2-...safetensors')
    comfy_type = dest_parts[0]                # 'unet'|'loras'|'text_encoders'|'vae'
    subdirs = dest_parts[1:-1]                # e.g. ('klein',) for the UNET, () otherwise
    names = (dest_parts[-1], *(spec.get('legacy_names') or ()))
    try:
        found = [os.path.join(root, *subdirs, name)
                 for root in (spec['extra_roots']() if spec.get('extra_roots') else _extra_roots_for(comfy_type))
                 for name in names
                 if os.path.isfile(os.path.join(root, *subdirs, name))]
    except Exception:
        logger.debug('extra-path klein presence check failed for %s', action, exc_info=True)
        return False
    usable = [p for p in found if not _is_blocking_invalid(p, spec)]
    for p in found:
        if p not in usable:
            _note(action, f'ignoring an unusable copy under an extra_model_paths root: {p}')
    return bool(usable)


def _variant_already_present(action, condemned=None):
    """Basename of a previously-accepted filename for `action` already on disk in the
    BASE dest folder (today: the pre-KV Klein UNET flux-2-klein-9b-fp8.safetensors),
    else None. When the default download filename changes, an install that fetched the
    old one stays valid — both variants resolve by name at generate time — so either
    counts as "already installed" instead of re-fetching ~10 GB. (extra_model_paths
    roots are covered by _download_present_in_extra, which accepts the same alternates.)
    None when the spec lists no `legacy_names` (every other action).

    "Still resolves" has to mean "still LOADS": a truncated legacy UNET resolves by
    name just as well as a good one, so accepting it on presence alone re-opened
    the dead end `dest` was fixed for. This folder is the app's own install tree
    (same tree `dest` lives in) and the resolver may well prefer the legacy name
    over the fresh download, so an unloadable variant does have to go — but NOT
    here and now. It sits at its own path, which `os.replace(part, dest)` will
    never overwrite, so it is collected into `condemned` and deleted by the caller
    once the fresh copy has actually landed. Deleting it up front turned a failed
    download into "the user now has nothing at all"."""
    spec = model_download_spec(action)
    alts = spec.get('legacy_names') or ()
    if not alts:
        return None
    try:
        dest_dir = os.path.dirname(_download_dest_path(action))
    except Precondition:
        return None
    for name in alts:
        path = os.path.join(dest_dir, name)
        if not os.path.isfile(path):
            continue
        reason = _unloadable_reason(action, path, spec)
        if not reason:
            return name
        _note(action, f'an earlier build is here under {name} but cannot be loaded: {reason}')
        if condemned is not None:
            condemned.append(path)
    return None


def _civitai_key():
    """The Civitai API key, read through the SAME resolver the scraper uses
    (env CIVITAI_API_KEY > the admin cookies dir > a legacy token file) so there
    is ONE Civitai credential in the app, not a second competing setting. The
    scrape package pulls optional dependencies, so an import failure degrades to
    the Settings-managed secret rather than breaking the download."""
    try:
        from .scrape.sources.civitai import civitai_api_key
        key = civitai_api_key()
        if key:
            return key
    except Exception:
        logger.debug('civitai_api_key() unavailable — falling back to the stored secret',
                     exc_info=True)
    return cfg.secret('CIVITAI_API_KEY') or None


def _download_auth(spec):
    """(headers, provider) for a download. A provider's token is NEVER sent to
    another host: the HF bearer only goes to Hugging Face URLs, the Civitai key
    only to Civitai, and a file LDS serves itself ('none') gets no credential.
    No credential at all is a legitimate case for both — public
    files download fine and a 401/403 is handled below."""
    provider = spec.get('auth', 'hf')
    if provider == 'none':
        return {}, provider
    token = _civitai_key() if provider == 'civitai' else cfg.secret('HF_TOKEN')
    return ({'Authorization': f'Bearer {token}'} if token else {}), provider


# Where the user creates a credential, per provider, for the 401/403 recovery
# steps. Same shape as the Hugging Face path that already existed.
_AUTH_RECOVERY = {
    'hf': ('Hugging Face', 'https://huggingface.co/settings/tokens', 'HF_TOKEN',
           'accept the licence on the model page (free), then'),
    'civitai': ('Civitai', 'https://civitai.com/user/account', 'CIVITAI_API_KEY',
                'sign in — Civitai requires an account for part of its catalogue '
                '(NSFW, early access, creator restrictions) — then'),
}


def _verify_downloaded_model(action, dest, spec, provider='hf') -> bool:
    """Is the file we just wrote actually loadable weights? An auth wall answers
    200 with an HTML page and the browser filename, which lands as a perfectly
    named `.safetensors` that ComfyUI then dies on ("Expecting value: line 1
    column 1"). Header-only check, the same validator the readiness probe uses.
    A blocking verdict DELETES the file — leaving it would make every later probe
    report the asset as installed. Advisory `too_small` is logged, not fatal.

    Callers pass the `.part` file, BEFORE it takes the real name: a gate page that
    already overwrote the previous copy would leave the user with strictly less
    than they started with, which is the one outcome this whole path exists to
    avoid."""
    if spec.get('sha256'):
        import hashlib
        try:
            expected_size = spec.get('expected_bytes')
            if expected_size and os.path.getsize(dest) != expected_size:
                raise ValueError('download size does not match the published file')
            digest = hashlib.sha256()
            _append(action, 'verifying the downloaded file SHA256')
            with open(dest, 'rb') as downloaded:
                for chunk in iter(lambda: downloaded.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != spec['sha256']:
                raise ValueError('SHA256 does not match the published file')
        except (OSError, ValueError) as exc:
            _append(action, f'download verification failed: {exc}; retry the download')
            # This is the temporary download, before replacement of any existing file.
            try:
                os.remove(dest)
            except OSError:
                pass
            return False
    try:
        from .services import model_integrity
        res = model_integrity.validate_model_file(dest, min_bytes=spec.get('min_bytes'))
    except Exception:
        logger.debug('integrity check failed for %s', action, exc_info=True)
        return True                     # never fail an install on the checker itself
    if res['ok']:
        return True
    if not res['blocking']:
        _append(action, f"warning: {res['reason']}")
        return True
    host, key_url, key_name, _verb = _AUTH_RECOVERY.get(provider, _AUTH_RECOVERY['hf'])
    _append(action, f"the downloaded file is not usable weights: {res['reason']}")
    _append(action, f'{host} most likely answered with a login/licence page instead of the '
                    f'file. Create an API key at {key_url} and paste it as {key_name} in '
                    'Settings -> API keys, then retry.')
    _append(action, 'the unusable file has been deleted, so nothing broken is left behind.')
    try:
        os.remove(dest)
    except OSError:
        pass
    return False


def _resolver_backed_assets():
    """{action: (missing_fn, invalid_fn)} for every engine whose OWN resolvers can
    answer "is this installed?". Built lazily so importing this module never drags
    in the engine helpers (and their ComfyUI probes)."""
    from .services import krea_edit_helper
    out = {a: (krea_edit_helper.krea_missing_assets,
               krea_edit_helper.krea_invalid_assets) for a in _KREA_DOWNLOADS}
    from functools import partial
    from .services import trained_image_models as studio_models
    for family, specs in studio_models.ASSET_SPECS.items():
        for slot in specs:
            out[studio_models.install_action(family, slot)] = (
                partial(studio_models.missing_assets, family),
                partial(studio_models.invalid_assets, family))
    return out


def _krea_asset_already_installed(action) -> bool:
    """RETROFIT guard: someone who placed a Krea or SeedVR2 asset by hand, under
    their own file name, anywhere ComfyUI registers, must not see it
    re-downloaded. The
    engine's own resolvers already answer "is this installed?" for exactly the
    file a generate would load, so we ask them rather than test one hardcoded
    path. Klein keeps its filename-based checks above (its resolver accepts a
    wider set and would suppress a legitimate first install).

    "The resolver finds it" is not "the loader can open it": krea_missing_assets()
    answers presence, and a hand-placed file that is an HTML gate page or a
    truncated download passes it. So the resolver's OWN integrity verdict
    (krea_invalid_assets, blocking only — the same list capabilities greys the
    engine on) vetoes the skip. Nothing is deleted: the file sits under a name and
    a folder the user chose, and the download goes to the canonical dest anyway."""
    try:
        entry = _resolver_backed_assets().get(action)
        if not entry:
            return False
        missing_fn, invalid_fn = entry
        if action in missing_fn():
            return False
        broken = next((i for i in invalid_fn()
                       if i['asset'] == action and i['blocking']), None)
        if broken:
            _note(action, f"the file already resolving for this asset cannot be loaded: "
                          f"{broken['reason']}")
            return False
        return True
    except Exception:
        logger.debug('resolver presence check failed for %s', action, exc_info=True)
        return False


def _unloadable_reason(action, path, spec):
    """Why the file at `path` is unusable weights, or None if it is keepable. PURE
    CHECK — it deletes nothing, which is the whole point of splitting it out.

    Replacement is limited to explicitly declared superseded revisions or a
    blocking verdict (model_integrity: an HTML gate page, or a header the file is
    too short to satisfy), which is a file no loader can open under any
    circumstances. Advisory `too_small` is the user's business. A failing checker
    is never grounds to condemn either — no answer means keep."""
    try:
        if os.path.getsize(path) in spec.get('superseded_bytes', ()):
            return 'superseded model revision; download the current official file'
        from .services import model_integrity
        res = model_integrity.validate_model_file(path, min_bytes=spec.get('min_bytes'))
    except Exception:
        logger.debug('integrity check failed for %s', action, exc_info=True)
        return None
    if res['ok'] or not res['blocking']:
        return None
    return res['reason']


def _drop_condemned(action, paths, keep=None):
    """Delete files judged unloadable — called ONLY once something better is proven
    to exist (a fresh download that landed, or another copy that does load). The
    ordering IS the feature: a broken weight is useless but it surprises nobody,
    while an empty folder after a re-download that never happened does.

    `keep` is the path a successful download has just rewritten in place (via
    os.replace) — condemning it was about the OLD bytes, which are already gone."""
    kept = os.path.normcase(os.path.abspath(keep)) if keep else None
    for path in paths:
        if kept and os.path.normcase(os.path.abspath(path)) == kept:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except OSError as e:
            _note(action, f'could not delete it ({e}) — remove it by hand: {path}')
            continue
        _note(action, f'removed the unusable file: {path}')


def _run_model_download(action) -> int:
    """Fetch the primary asset and its declared companions, preserving any valid files already present."""
    rc = _run_primary_download(action)
    if rc != 0:
        return rc
    complete = model_download_spec(action).get('complete')
    if complete and complete():
        # The whole stage is already where the node will read it — under an
        # extra root, say: nothing to add to this install's tree.
        _append(action, 'the stage is already complete where ComfyUI reads it — nothing else to fetch')
        return 0
    return _run_companion_downloads(action)


# --- Custom-node pack install --------------------------------------------------
# Bounded so a hung network can never wedge the install worker thread.
_GIT_CLONE_TIMEOUT_S = 300
_ZIP_TIMEOUT = (10, 120)


def _node_pack_already_there(action, dest) -> bool:
    """A non-empty destination folder means the pack is ALREADY installed (or the
    user put something of their own there). We never overwrite it: someone may
    have patched the pack, pinned a commit, or installed it through the ComfyUI
    Manager. Idempotent by design — re-clicking Install is safe."""
    try:
        return os.path.isdir(dest) and any(os.scandir(dest))
    except OSError:
        return False


def _zip_node_pack(action, spec, dest) -> bool:
    """Fetch the pack's pinned commit ZIP, check its SHA-256, and move its single
    top-level folder into place. Extracts to a sibling temp folder first, so a
    failure never leaves a partial pack ComfyUI would try to import."""
    import tempfile
    import zipfile
    if not spec.get('sha256'):
        # Every catalog entry is pinned; an unpinned one is a catalog bug.
        _append(action, 'This node pack has no pinned checksum. Nothing was installed.')
        return False
    _append(action, f"downloading {spec['zip']}")
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix='.lds_nodepack_', dir=parent)
    archive = os.path.join(tmp_dir, 'pack.zip')
    try:
        with requests.get(spec['zip'], stream=True, timeout=network_timeout(_ZIP_TIMEOUT),
                          allow_redirects=False) as resp:
            if resp.status_code != 200:
                _append(action, f'HTTP {resp.status_code} downloading the ZIP')
                return False
            with open(archive, 'wb') as fh:
                digest = hashlib.sha256()
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != spec['sha256']:
            _append(action, 'The node archive checksum did not match. Nothing was installed.')
            return False
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_dir)
        roots = [n for n in os.listdir(tmp_dir)
                 if n != 'pack.zip' and os.path.isdir(os.path.join(tmp_dir, n))]
        if len(roots) != 1:
            _append(action, f'unexpected ZIP layout ({len(roots)} top-level folders) — '
                            'install the pack manually, see the link above')
            return False
        shutil.move(os.path.join(tmp_dir, roots[0]), dest)
        return True
    except (requests.RequestException, OSError, zipfile.BadZipFile) as e:
        _append(action, f'ZIP install failed: {e}')
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _deploy_bundled_pack(action, log=None) -> tuple[bool, str]:
    """Copy backend/comfy_nodes/<folder> into the user's ComfyUI. (ok, message).

    Shared by the Setup button and the boot-time refresh, so the two can never
    disagree about what "installed" means. `log` is an optional one-arg callable
    for progress lines (the Setup run log); the boot path passes None.

    Replaces rather than merges: a `copytree(..., dirs_exist_ok=True)` over an old
    version leaves behind any file the new version dropped, and a stray module in
    a ComfyUI package folder is imported all the same. The delete is gated on the
    stamp, so the only directory this can ever remove is one a previous run of
    this same function wrote."""
    def say(line):
        if log:
            log(line)

    src = _bundled_pack_source(action)
    if not os.path.isdir(src):
        # Only reachable from a truncated install (the folder is copied verbatim
        # into every release). Say which folder, so a support answer is one line.
        return False, (f'the shipped node folder is missing from this install '
                       f'({os.path.basename(src)}) — reinstall the app files')
    dest = _bundled_pack_dest(action)
    state = _bundled_pack_state(action)

    if state == 'current':
        return True, f'already up to date ({APP_VERSION})'
    if state == 'foreign':
        return False, ('a folder of that name already exists in your ComfyUI and was '
                       'not put there by this app — it was left untouched. Remove or '
                       'rename it if you want the shipped version.')
    if state == 'stale':
        say(f'replacing the previous copy ({_bundled_pack_stamp(dest)} -> {APP_VERSION})')
        try:
            shutil.rmtree(dest)
        except OSError as e:
            return False, f'could not remove the previous copy: {e}'

    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # __pycache__ is the user's ComfyUI's business, not ours to seed: a .pyc
        # compiled by a different interpreter version is at best ignored and at
        # worst imported in preference to the source next to it.
        shutil.copytree(src, dest,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        with open(os.path.join(dest, _BUNDLED_STAMP), 'w', encoding='utf-8') as fh:
            fh.write(APP_VERSION)
    except OSError as e:
        # Never leave a half-copied package behind: ComfyUI would import it and
        # report a broken node rather than a missing one, which is a much harder
        # thing for a user to describe.
        shutil.rmtree(dest, ignore_errors=True)
        return False, f'could not copy the node into ComfyUI: {e}'
    return True, f'installed {_BUNDLED_NODE_PACKS[action]["pack"]} ({APP_VERSION})'


def _run_bundled_node_pack(action) -> int:
    """Setup action for a node pack this app ships. No network, no git: the source
    is already on disk beside the code that installs it.

    Like every node install, success does not mean the node is usable yet —
    ComfyUI registers nodes at startup only."""
    try:
        _bundled_pack_dest(action)
    except Precondition as e:
        _append(action, f'{e}')
        _append(action, "the node has to go inside YOUR ComfyUI's custom_nodes folder, "
                        "and the app doesn't know where that is yet — nothing was installed.")
        return 1
    ok, message = _deploy_bundled_pack(action, log=lambda line: _append(action, line))
    _append(action, message)
    if not ok:
        return 1
    _append(action, 'restart ComfyUI for it to load.')
    return 0


def refresh_bundled_node_packs() -> dict:
    """Re-deploy every shipped pack whose installed copy is out of date. {action: message}
    for the ones that were actually touched (empty when there is nothing to do).

    Called at boot. This is the piece that makes an app update reach the node:
    "Update & restart" replaces the app's files and nothing else — it has no
    business writing to the user's ComfyUI on its own — so without this, someone
    who installed the node once would keep the version they first clicked, forever,
    while the app's graph moved on. Absent copies are LEFT absent: installing is
    the user's decision, refreshing what they already chose is not a new one."""
    out = {}
    for action in _BUNDLED_NODE_PACKS:
        try:
            if _bundled_pack_state(action) != 'stale':
                continue
            ok, message = _deploy_bundled_pack(action)
            out[action] = message
            logger.info('bundled node pack %s: %s', action, message)
        except Exception:       # noqa: BLE001 — boot must not die over a node folder
            logger.warning('bundled node pack %s: refresh failed', action, exc_info=True)
    return out


def _run_node_pack(action) -> int:
    """Install the reviewed node archive or reuse a previously prepared folder."""
    spec = _NODE_PACKS[action]
    try:
        dest = _node_pack_dest(action)
    except Precondition as e:
        _append(action, f'{e}')
        _append(action, "the pack has to go inside YOUR ComfyUI's custom_nodes folder, and "
                        "the app doesn't know where that is yet — nothing was installed.")
        return 1
    # The recognized portable uses the pinned, verified recipe. Other layouts
    # retain their existing acquisition path until they have a runtime adapter.
    # Never fall back to an unverified download after a managed preparation fails.
    from .services import comfyui_control, comfyui_node_install, comfyui_node_recipes
    recipe = comfyui_node_recipes.CORE_RECIPES.get(action)
    managed = (recipe is not None and comfyui_control._validated_portable_layout() is not None)
    existing = _node_pack_already_there(action, dest)
    if managed and (not existing or os.path.lexists(os.path.join(dest, comfyui_node_install.RECEIPT))):
        try:
            plan = comfyui_node_install.plan(recipe, owner='lds.core')
            comfyui_node_install.prepare(recipe, owner='lds.core', plan_id=plan['plan_id'],
                                         log=lambda line: _append(action, str(line)))
        except comfyui_node_install.NodeInstallError as exc:
            _append(action, str(exc))
            return 1
        _append(action, 'Prepared the verified Krea node pack. RESTART ComfyUI when it is idle; '
                        'LDS will check the loaded nodes before marking the engine ready.')
        return 0
    if existing:
        _append(action, f'already installed: {dest}')
        _append(action, 'left untouched (an existing folder may be a version you chose). '
                        'If ComfyUI still reports the nodes as missing, restart ComfyUI.')
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ok = _zip_node_pack(action, spec, dest)
    if not ok:
        _append(action, f"couldn't install {spec['pack']} automatically.")
        _append(action, manual_command(action))
        return 1
    _append(action, f"installed {spec['pack']} -> {dest}")
    # The pack declares no dependencies today; if a future version adds some, say
    # so instead of silently pip-installing third-party requirements into the app.
    reqs = os.path.join(dest, 'requirements.txt')
    if os.path.isfile(reqs):
        _append(action, 'note: this pack now ships a requirements.txt. The app does not '
                        'install third-party Python packages for you — install them into '
                        "your ComfyUI's Python if the nodes fail to load.")
    _append(action, '⚠ RESTART ComfyUI now — it only registers custom nodes at startup, so '
                    'the engine stays marked "nodes missing" until you do.')
    return 0


def _ollama_cancelled(action) -> bool:
    run = _runs.get(action) or {}
    event = run.get('cancel_event')
    return bool(event and event.is_set())


def _iter_bounded_ollama_lines(response):
    """Split Ollama NDJSON without allowing one unterminated line to grow forever."""
    pending = bytearray()
    for chunk in response.iter_content(chunk_size=_OLLAMA_STREAM_CHUNK):
        if not chunk:
            continue
        if isinstance(chunk, str):
            chunk = chunk.encode('utf-8')
        pending.extend(chunk)
        while True:
            separator = pending.find(b'\n')
            if separator < 0:
                if len(pending) > _OLLAMA_MAX_LINE:
                    raise ValueError('Ollama response line is too large')
                break
            if separator > _OLLAMA_MAX_LINE:
                raise ValueError('Ollama response line is too large')
            line = bytes(pending[:separator]).rstrip(b'\r')
            del pending[:separator + 1]
            if line:
                yield line
    if len(pending) > _OLLAMA_MAX_LINE:
        raise ValueError('Ollama response line is too large')
    if pending:
        yield bytes(pending).rstrip(b'\r')


def _safe_ollama_text(value) -> str:
    if not isinstance(value, str):
        return ''
    return re.sub(r'[\x00-\x1f\x7f]+', ' ', value).strip()[:300]


def _run_ollama_model(action) -> int:
    url = _ollama_pull_base_url()
    model = (cfg.get('ollama.vision_model') or '').strip()
    response = None
    last_status = ''
    last_total = 0
    saw_success = False
    if _ollama_cancelled(action):
        raise Cancelled()
    try:
        response = requests.post(
            f'{url}/api/pull',
            json={'model': model, 'stream': True},
            stream=True,
            allow_redirects=False,
            timeout=network_timeout((_OLLAMA_CONNECT_TIMEOUT, _OLLAMA_READ_TIMEOUT)),
        )
        run = _runs.get(action)
        if run is not None:
            run['response'] = response
        if _ollama_cancelled(action):
            raise Cancelled()
        if 300 <= response.status_code < 400:
            _append(action, 'Ollama refused an unexpected HTTP redirect.')
            return 1
        if not 200 <= response.status_code < 300:
            _append(action, f'Ollama returned HTTP {response.status_code}.')
            return 1

        for raw_line in _iter_bounded_ollama_lines(response):
            if _ollama_cancelled(action):
                raise Cancelled()
            try:
                payload = json.loads(raw_line.decode('utf-8'))
            except (UnicodeDecodeError, ValueError):
                _append(action, 'Ollama returned invalid streaming JSON.')
                return 1
            if not isinstance(payload, dict):
                _append(action, 'Ollama returned an invalid streaming event.')
                return 1
            error = _safe_ollama_text(payload.get('error'))
            if error:
                _append(action, f'Ollama error: {error}')
                return 1
            completed = payload.get('completed')
            total = payload.get('total')
            if (type(completed) is int and type(total) is int
                    and completed >= 0 and total >= 0
                    and (not total or completed <= total)):
                _set_progress(action, completed, total)
                last_total = total or last_total
            status_text = _safe_ollama_text(payload.get('status'))
            if status_text and status_text != last_status:
                _append(action, status_text)
                last_status = status_text
            if status_text.lower() == 'success':
                saw_success = True

        if _ollama_cancelled(action):
            raise Cancelled()
        if not saw_success:
            _append(action, 'Ollama closed the pull stream before reporting success.')
            return 1
        if last_total:
            _set_progress(action, last_total, last_total)
        return 0
    except Cancelled:
        raise
    except requests.exceptions.Timeout:
        if _ollama_cancelled(action):
            raise Cancelled()
        _append(action, 'Ollama stopped sending progress before the 45-second read timeout.')
        return 1
    except requests.exceptions.RequestException:
        if _ollama_cancelled(action):
            raise Cancelled()
        _append(action, 'Could not reach Ollama for the model pull.')
        return 1
    except Exception:
        if _ollama_cancelled(action):
            raise Cancelled()
        _append(action, 'Ollama returned an invalid or interrupted pull stream.')
        return 1
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        run = _runs.get(action)
        if run is not None and run.get('response') is response:
            run['response'] = None


def _run_shot_detect(action) -> int:
    """Install TransNetV2 — the shot-boundary detector the video bank cuts with.

    It goes into the environment bank scoring already manages, not the app's own
    Python, for one reason: it needs torch. A second torch is ~2.5 GB the user
    gains nothing from, and the watermark detector already settled this question
    the same way. The capability probe resolves the interpreter through the same
    chain, so the install target and the later import cannot drift.

    Unlike the watermark detector there is no weights step: transnetv2-pytorch
    ships its ~33 MB weights inside the wheel, so "installed" really does mean
    "usable offline". That is most of why it was chosen over the alternatives.
    """
    managed_python = _bank_scoring_env_python()
    configured = (cfg.get('shot_detect.python') or '').strip()
    if configured and not _same_path(configured, managed_python):
        # A BORROWED environment — checked, never changed (the ⚡ picker's promise).
        for line in (
            'shot_detect.python points at an environment this app did not create,',
            'so nothing was installed into it — borrowed environments are checked,',
            'never changed. To add the detector there yourself, run:',
            f'  "{configured}" -m pip install torch transnetv2-pytorch av',
            'Or clear shot_detect.python and click Install again — the app then uses',
            'its own scoring environment, which already has torch.',
        ):
            _append(action, line)
        return 1
    python = _ensure_bank_scoring_env(action, save_score_python=False)
    if not python:
        return 1
    if _is_flask_venv(python):
        for line in (
            "Shot detection needs torch, which never installs into the app's own Python.",
            'Nothing was installed. Clear shot_detect.python and click Install again —',
            'the app builds a dedicated Python for you.',
        ):
            _append(action, line)
        return 1
    _append(action, f'target interpreter: {python}')
    rc = _install_cpu_torch_pair(action, python)
    if rc != 0:
        return rc
    # av rides along because the WORKER decodes with PyAV in this same
    # environment (infer/shot_detect_infer.py imports av before torch sees a
    # frame). Without it the model loads, the probe used to say ready, and
    # every file failed with ModuleNotFoundError: av — 246/246 on the first
    # real bank this install met.
    rc = _run_pip(action, [python, '-m', 'pip', 'install',
                           _requirement_spec('transnetv2-pytorch'),
                           _requirement_spec('av')])
    if rc != 0:
        return rc
    if not _verify_shot_detect_import(action, python):
        return 1
    try:
        cfg.save_config({'shot_detect': {'python': python}})
    except Exception as e:      # noqa: BLE001
        _append(action, f'warning: could not save shot_detect.python ({e}); '
                        'the environment still works for this run')
    return 0


def _verify_shot_detect_import(action, python) -> bool:
    """Run the SAME import the probe will, in the target environment, once pip
    says it is done — otherwise an install reports success while the capability
    stays off with no reason shown anywhere. A timeout is 'still warming', never
    a failure."""
    if not os.path.isfile(python):
        return True
    _append(action, 'verifying the install (first import — this also warms it)…')
    try:
        proc = subprocess.run(
                              infer_env.worker_argv(
                                  python, '-c', 'import torch, transnetv2_pytorch, av'),
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=processing_timeout(_WARM_IMPORT_TIMEOUT),
                              env=infer_env.worker_env(python),
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired:
        _append(action, 'still warming up — the capability turns green on its own '
                        'shortly; no restart needed')
        return True
    if proc.returncode == 0:
        return True
    _append(action, 'the packages installed but the import still fails:')
    for line in (proc.stderr or '').strip().splitlines()[-8:]:
        _append(action, f'  {line}')
    return False


def _run_dlss5nr_bridge(action) -> int:
    """✨ The neural rendering bridge: a pinned zip, verified by size and SHA-256,
    two DLLs unpacked under the app's data folder. Everything it says goes to
    the run log, including the one thing it cannot do — the model file."""
    from .services import neural_render
    return neural_render.install_bridge(log=lambda line: _append(action, line))


def _run_aitoolkit_install(action):
    """The trainer, into the folder admitted at start (see aitoolkit_install)."""
    from .services import aitoolkit_install
    run = _runs[action]

    def progress(done, total):
        _set_progress(action, done, total)

    def register(runner):
        run['stopper'] = runner.kill

    with run['app'].app_context():
        try:
            return aitoolkit_install.install(
                run['options']['dest'], log=lambda line: _append(action, line),
                step=lambda index, total, label: _set_step(action, index, total, label),
                progress=progress, cancel_event=run['cancel_event'],
                find_python=lambda: _find_base_python(action), register=register,
                fallback_python=lambda: managed_python.ensure_python(
                    lambda line: _append(action, line)))
        except aitoolkit_install.Cancelled:
            raise Cancelled() from None
        finally:
            run['stopper'] = None


def _aitoolkit_install_options(options) -> dict:
    """Admit an AI Toolkit install: the dry-run plan must say 'ready' for the
    requested folder (or the default one). Raises Precondition with its reason."""
    from .services import aitoolkit_install
    dest = options.get('dest') if isinstance(options, dict) else None
    if dest is not None and not isinstance(dest, str):
        raise Precondition('dest must be a folder path')
    checked = aitoolkit_install.plan(dest)
    if checked['state'] != 'ready':
        raise Precondition(checked['reason'] or 'AI Toolkit cannot be installed here.')
    return {'dest': checked['dest']}


_WORKERS = {'aitoolkit_install': _run_aitoolkit_install,
            'ml_extras': _run_ml_extras,
            'ollama_model': _run_ollama_model,
            **{a: _run_ml_capability for a in _CAPABILITY_ML_ACTIONS if a in INSTALL_ACTIONS},
            'watermark_inpaint': _run_watermark_inpaint,
            'bank_scoring': _run_bank_scoring,
            'bank_siglip2': _run_bank_siglip2,
            'watermark_detect': _run_watermark_detect,
            **{a: _run_model_download for a in _MODEL_DOWNLOADS},
            **{a: _run_node_pack for a in _NODE_PACKS},
            **{a: _run_bundled_node_pack for a in _BUNDLED_NODE_PACKS}}
# Structural invariant: every whitelisted action MUST have a worker — a missing
# entry surfaces as a cryptic "error: '<action>'" KeyError at runtime (live
# repro: scrape_extras was added to INSTALL_ACTIONS but not here).
assert set(INSTALL_ACTIONS) == set(_WORKERS), \
    f'INSTALL_ACTIONS/_WORKERS mismatch: {set(INSTALL_ACTIONS) ^ set(_WORKERS)}'


# Generic plugin preparation state, independent of installed products.
_PLUGIN_MANAGED_ACTIONS = frozenset(('scrape_extras', 'video', 'shot_detect'))
_COMPANION_LOCKS = {}


def _plugin_registry():
    from .plugins.registry import active
    return active()


def model_download_spec(action):
    """Return the core or registered plugin asset specification, without downloading it."""
    spec = _MODEL_DOWNLOADS.get(action)
    if spec is not None:
        return spec
    registry = _plugin_registry()
    return registry.model_downloads.get(action) if registry else None


def _managed_action_enabled(action) -> bool:
    if action not in _PLUGIN_MANAGED_ACTIONS:
        return True
    registry = _plugin_registry()
    if registry is None:
        return True  # Pure installer utilities before an app loads its plugins.
    owner = registry.owner_of('install_actions', action)
    record = registry.records.get(owner)
    return bool(record and record.enabled and record.state == 'loaded'
                and (cfg.get('plugins.enabled') or {}).get(owner) is not False)


def known_action(action) -> bool:
    """Whitelist check for the routes: the core's actions or a plugin's."""
    if action in INSTALL_ACTIONS:
        return _managed_action_enabled(action)
    registry = _plugin_registry()
    spec = plugin_action_spec(action)
    model = registry.model_downloads.get(action) if registry else None
    return bool(registry and ((model is not None and _plugin_action_enabled(model))
                              or (spec is not None and (spec.get('environment') or _plugin_action_enabled(spec)))))


def plugin_action_spec(action):
    """What a plugin registered through ``ctx.register_install_action`` —
    ``{'plugin', 'label', 'packages', 'requirements', 'models', 'run', 'python', 'verify'}``
    — or None for a core action or a model download."""
    registry = _plugin_registry()
    from .plugins.environment import action_spec
    return (action_spec(action, registry) or registry.install_actions.get(action)) if registry else None


def _plugin_action_enabled(spec) -> bool:
    """Offers obey the same owner-state gate as execution; retain specs for busy tracking."""
    registry = _plugin_registry()
    record = registry.records.get(spec['plugin']) if registry else None
    if record is None:
        return False
    from .plugins.environment import EnvironmentError, check_enabled
    try:
        check_enabled(record, loaded=not spec.get('environment'))
    except EnvironmentError:
        return False
    return True


def plugin_actions_catalog() -> dict:
    """Every Setup action a plugin added, with the label the screen shows —
    the core's labels live in the frontend's own table; a plugin's arrive
    through this (GET /api/setup/actions)."""
    registry = _plugin_registry()
    if registry is None:
        return {}
    out = {}
    for key, spec in registry.install_actions.items():
        if not _plugin_action_enabled(spec):
            continue
        kind = 'node_pack' if spec.get('node_pack') else 'run' if callable(spec.get('run')) else 'pip'
        out[key] = {'label': spec.get('label') or key, 'plugin': spec['plugin'], 'kind': kind,
                    'python': 'comfyui' if spec.get('node_pack') else spec.get('python') or 'plugin'}
    for key, spec in registry.model_downloads.items():
        if not _plugin_action_enabled(spec):
            continue
        out.setdefault(key, {'label': spec.get('label') or key, 'plugin': spec['plugin'],
                             'kind': 'download', 'python': None})
    from .plugins.environment import action_id, action_spec
    for record in registry.records.values():
        key = action_id(record.id)
        spec = action_spec(key, registry)
        if spec:
            out[key] = {'label': spec['label'], 'plugin': record.id, 'kind': 'pip', 'python': 'plugin'}
    return out


def _plugin_action_is_pip(spec) -> bool:
    return bool(spec) and not callable(spec.get('run')) and bool(
        spec.get('environment') or spec.get('packages') or spec.get('requirements'))


def _is_pip_action(action) -> bool:
    """The pip FIFO's membership: the core's pip actions and a plugin's pip
    action alike — two installs must never race one environment."""
    return action in _PIP_ACTIONS or _plugin_action_is_pip(plugin_action_spec(action))


def _plugin_action_python(spec) -> str:
    """The interpreter a plugin's pip action installs into: the app's own
    (``python='app'`` — pure-Python wheels the app imports in-process, the
    scrape stack's case) or the plugin's own environment, which Setup ▸ Plugins
    builds under the plugin's data folder. ``python='capability'`` resolves the
    owned host capability's configured interpreter, including its app fallback.
    This read-only diagnostic refuses
    a missing environment; starting the action provisions it in the FIFO."""
    if spec.get('python') == 'capability':
        return _capability_python(spec['capability'])
    if (spec.get('python') or 'plugin') == 'app':
        return sys.executable
    registry = _plugin_registry()
    record = registry.records.get(spec['plugin']) if registry else None
    if record is None:
        raise Precondition(f"plugin {spec['plugin']!r} is not loaded")
    from .plugins.environment import EnvironmentError, interpreter
    try:
        return interpreter(record)
    except EnvironmentError as exc:
        raise Precondition(str(exc)) from exc


def _plugin_action_command(spec) -> list:
    cmd = [_plugin_action_python(spec), '-m', 'pip', 'install', *spec.get('packages', ())]
    if spec.get('requirements'):
        cmd += ['-r', str(spec['requirements'])]
    if spec.get('python') == 'capability':
        if not _ML_REQUIREMENTS.is_file():
            raise Precondition('The LDS ML constraints are missing. Repair LDS before installing extras.')
        cmd += ['-c', str(_ML_REQUIREMENTS), *_flask_pillow_guard(cmd[0])]
    if spec.get('python') == 'app' or (spec.get('python') == 'capability' and _is_flask_venv(cmd[0])):
        if not _APP_REQUIREMENTS.is_file():
            raise Precondition('The LDS dependency requirements are missing. Repair LDS before installing extras.')
        # Host requirements can include extras, which pip forbids in constraints.
        # Resolve them alongside the plugin to preserve both host pins and extras.
        cmd += ['-r', str(_APP_REQUIREMENTS)]
    return cmd


def _run_plugin_action(action) -> int:
    """The worker of a plugin's install action: its own ``run(log=...)`` (an
    int returncode, None read as success), or the pip install its spec
    describes, streamed to the same ring log as the core's."""
    spec = plugin_action_spec(action)
    if spec is None:
        raise KeyError(action)
    registry = _plugin_registry()
    record = registry.records.get(spec['plugin']) if registry else None
    from .plugins import environment
    if record is None:
        raise Precondition('This plugin is no longer installed. Restart LDS and try again.')
    environment.check_enabled(record, loaded=not spec.get('environment'))
    if spec.get('python') == 'capability':
        # Host capabilities have a fixed managed recipe. The product contributes
        # its action, while LDS owns Python selection, FIFO and verification.
        worker = _run_shot_detect if action == 'shot_detect' else _run_ml_capability
        return _verify_plugin_action(action, spec, worker(action))
    run = spec.get('run')
    if callable(run):
        rc = run(log=lambda line: _append(action, str(line)))
        rc = int(rc or 0)
        return _verify_plugin_action(action, spec, rc)
    if not _plugin_action_is_pip(spec):
        _append(action, 'nothing to install: the action names no packages, requirements or run()')
        return 1
    if spec.get('python', 'plugin') == 'plugin':
        rc = environment.install(action, record, extra_packages=spec.get('packages', ()),
                                 extra_requirements=spec.get('requirements'))
    else:
        command = _plugin_action_command(spec)
        _append(action, f'target interpreter: {command[0]}')
        rc = _run_pip(action, command)
        if rc == 0 and spec.get('python') == 'capability':
            if not _verify_capability_import(spec['capability'], command[0]):
                return 1
    return _verify_plugin_action(action, spec, rc)


def _verify_plugin_action(action, spec, rc):
    verify = spec.get('verify')
    if rc == 0 and callable(verify):
        importlib.invalidate_caches()
        if not verify():
            _append(action, f"Post-install check failed — {spec.get('label') or action} is not ready. "
                            'Repair this component and try again.')
            return 1
    return rc


def install_groups() -> dict:
    """group -> member actions: the core's groups, then the plugins'."""
    out = dict(_INSTALL_GROUPS)
    registry = _plugin_registry()
    if registry:
        for key, spec in registry.install_groups.items():
            out.setdefault(key, spec['members'])
    return out


def group_caps_keys(group) -> dict:
    """The capability keys a group's plan reads its gaps from, every key
    present (None = the group has no such lane)."""
    keys = _GROUP_CAPS_KEYS.get(group)
    if keys is None:
        registry = _plugin_registry()
        spec = registry.install_groups.get(group) if registry else None
        keys = spec['caps_keys'] if spec else {}
    return {'missing': None, 'invalid': None, 'pack_action': None,
            'nodes_missing': None, 'nodes_installed': None, **keys}


def _worker_for(action):
    # Product recipes override the retained legacy recipe only while loaded.
    if plugin_action_spec(action) is not None:
        return _run_plugin_action
    worker = _WORKERS.get(action)
    if worker is None and model_download_spec(action) is not None:
        worker = _run_model_download
    if worker is None and plugin_action_spec(action) is not None:
        worker = _run_plugin_action
    if worker is None:
        raise KeyError(action)
    return worker


def _start_locked(action, options=None) -> dict:
    if not known_action(action):
        raise ValueError(f'unknown action: {action}')
    global _pip_current
    admitted = None
    if action == 'aitoolkit_install':
        # Its plan runs nvidia-smi and Python probes: check it BEFORE taking the
        # registry lock every status poll needs, then re-check the run below.
        with _lock:
            run = _runs.get(action)
            if run and run['state'] in ('running', 'queued'):
                raise AlreadyRunning(action)
        admitted = _aitoolkit_install_options(options)
    with _lock:
        run = _runs.get(action)
        if run and run['state'] in ('running', 'queued'):
            raise AlreadyRunning(action)
        if admitted is None:
            check_start_preconditions(action)
        _runs[action] = _new_run()
        if admitted is not None:
            _runs[action]['options'] = admitted
        if action == 'aitoolkit_install':
            from flask import current_app
            _runs[action]['app'] = current_app._get_current_object()
        if _is_pip_action(action) and _pip_current is not None:
            # A pip install already owns the worker -> queue this one (FIFO, click
            # order) instead of racing it into the same environment. It starts on its
            # own when the current install finishes (see _release_pip_slot).
            _runs[action]['state'] = 'queued'
            _runs[action]['waiting_for'] = _pip_current
            _pip_queue.append(action)
            return status(action)
        if _is_pip_action(action):
            _pip_current = action
    threading.Thread(target=_execute, args=(action,), daemon=True).start()
    return status(action)


def check_start_preconditions(action):
    """Check an admitted action without creating a run, for selected batches too.

    start() repeats these checks immediately before creating its reservation.
    A batch checks every member first, so an invalid later member cannot start
    an unrelated partial preparation.
    """
    if not known_action(action):
        raise ValueError(f'unknown action: {action}')
    if action == 'aitoolkit_install':
        _aitoolkit_install_options(None)
    if action == 'ollama_model':
        _check_ollama_precondition()
    if model_download_spec(action) is not None:
        _check_download_precondition(action)
    if action in _NODE_PACKS:
        _node_pack_dest(action)
    if action in _BUNDLED_NODE_PACKS:
        _bundled_pack_dest(action)
    spec = plugin_action_spec(action)
    if spec is not None:
        from .plugins import environment
        record = _plugin_registry().records.get(spec['plugin'])
        try:
            environment.check_enabled(record, loaded=not spec.get('environment'))
            if environment.running(record.id):
                raise environment.EnvironmentError('Wait for this plugin’s running script to finish before installing its environment.')
            if _plugin_action_is_pip(spec) and spec.get('python', 'plugin') == 'plugin':
                environment._owned_dir(record)
                environment.requirements(record)
                environment._specs(spec.get('packages', ()))
                if spec.get('requirements'):
                    environment.requirements(record, spec['requirements'])
        except environment.EnvironmentError as exc:
            raise Precondition(str(exc)) from exc
        if spec.get('node_pack'):
            from .services import comfyui_node_install
            try:
                comfyui_node_install._target()
            except comfyui_node_install.NodeInstallError as exc:
                raise Precondition(str(exc)) from exc


def plugin_install_busy(plugin_id) -> bool:
    """An admitted install keeps its source and environment until it finishes."""
    with _lock:
        return any(run['state'] in ('running', 'queued')
                   and ((plugin_action_spec(action) or {}).get('plugin') == plugin_id
                        or (model_download_spec(action) or {}).get('plugin') == plugin_id
                        or plugin_id in run.get('preparation_plugins', ()))
                   for action, run in _runs.items())


def _quality_env_dir():
    return cfg.data_dir() / 'envs' / 'quality'


def _managed_env_valid(python):
    """Verify the running Python actually belongs to this isolated venv."""
    try:
        result = subprocess.run(
            [python, '-I', '-c', 'import sys,json,pip; '
             'print(json.dumps([list(sys.version_info[:2]),sys.prefix,sys.base_prefix]))'],
            capture_output=True, text=True, timeout=processing_timeout(20),
            env=managed_python.subprocess_env(),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        version, prefix, base = json.loads(result.stdout)
        expected = os.path.dirname(os.path.dirname(python))
        return (result.returncode == 0 and _VENV_PY_MIN <= tuple(version) <= _VENV_PY_MAX
                and os.path.normcase(os.path.abspath(prefix)) == os.path.normcase(expected)
                and os.path.normcase(prefix) != os.path.normcase(base))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return False


def _ensure_managed_ml_env(action, env_dir) -> str:
    """Create/reuse only a validated LDS-owned venv; repair broken ones safely."""
    import uuid
    env_dir = env_dir.absolute()
    python = _venv_python(env_dir)
    try:
        managed_python.assert_owned_directory(env_dir, cfg.data_dir())
        if _is_flask_venv(python):
            raise ValueError('Refusing to install ML tools into the app Python.')
        if os.path.isfile(python) and _managed_env_valid(python):
            _append(action, f'reusing the managed {env_dir.name} environment')
            _ensure_modern_pip(action, python)
            return python
        base = _find_base_python(action)
        if not base:
            return ''
        backup = env_dir.with_name(env_dir.name + '.previous-' + uuid.uuid4().hex)
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        if env_dir.exists():
            env_dir.rename(backup)
            _append(action, 'The previous broken environment was kept for recovery.')
        try:
            _append(action, f'building the managed {env_dir.name} environment')
            rc = _run_pip(action, [base, '-m', 'venv', str(env_dir)])
            if rc != 0 or not os.path.isfile(python) or not _managed_env_valid(python):
                raise ValueError('The managed environment failed its Python version check.')
        except Exception:
            if env_dir.exists():
                managed_python.assert_owned_directory(env_dir, cfg.data_dir())
                shutil.rmtree(env_dir)
            if backup.exists():
                backup.rename(env_dir)
            raise
        _ensure_modern_pip(action, python)
        return python
    except Exception as exc:
        _append(action, f'Could not prepare the quality environment: {exc}')
        _append(action, 'Click Install again to retry; the app and external Python '
                        'environments were left unchanged.')
        return ''


def _bank_runtime_notice(action, python, selected, *, imports_ok=None, selection_failed=False):
    """Receipt of a verified managed install, not a GPU/worker health verdict.

    Paths belong to the local selection UI. Diagnostic text stays path-free.
    Keep the receipt on the run so status polling never imports or computes.
    """
    profile = {'bank_scoring': 'scoring', 'bank_siglip2': 'semantic'}.get(action)
    run = _runs.get(action)
    if profile is None or run is None:
        return
    run['runtime_notice'] = {
        'profile': profile, 'managed_python': python, 'effective_python': selected,
        'uses_managed': _same_path(selected, python), 'managed_installed': True,
        'selected_imports_ok': imports_ok, 'compute_tested': False,
        'selection_failed': selection_failed,
    }


def _select_managed_python(action, key, python) -> bool:
    """Publish a ready runtime, preserving an explicit external selection."""
    configured = (cfg.get(f'{key}.python') or '').strip()
    selection_location = ('the Python picker in Bank → Passes or below the repair result'
                          if action in ('bank_scoring', 'bank_siglip2') else 'Settings')
    if configured and not _same_path(configured, python) and not _is_flask_venv(configured):
        _append(action, f'Keeping the selected borrowed {key} interpreter unchanged: '
                        f'{configured}. The managed environment is ready for selection '
                        f'in {selection_location}.')
        feature = {'watermark': 'watermark_inpaint', 'bank_semantic': 'bank_siglip2'}.get(key, key)
        imports_ok = _verify_capability_import(feature, configured, log_action=action)
        _bank_runtime_notice(action, python, configured, imports_ok=imports_ok)
        if not imports_ok:
            _append(action, 'The managed install is ready, but the selected external '
                            'runtime failed its import check. That external runtime '
                            'was not modified. Select the managed interpreter in '
                            f'{selection_location}.')
            return False
        return True
    try:
        cfg.save_config({key: {'python': python}})
        _bank_runtime_notice(action, python, python, imports_ok=True)
        return True
    except Exception as exc:
        _bank_runtime_notice(action, python, configured or python, selection_failed=True)
        _append(action, f'Could not select the installed {key} environment: {exc}. '
                        'Click Install again to retry.')
        return False


def _install_quality_tools(action, features) -> int:
    python = _ensure_managed_ml_env(action, _quality_env_dir())
    if not python:
        return 1
    names = tuple(dict.fromkeys(name for feature in features
                               for name in _CAPABILITY_PACKAGES[feature]))
    specs = _drop_provided_onnxruntime(action, python, [_requirement_spec(n) for n in names])
    _append(action, f'target interpreter: {python}')
    _append(action, 'Installing CPU quality packages from prebuilt wheels; '
                    'no local compiler or changes to the app Python.')
    rc = _run_pip(action, [python, '-m', 'pip', 'install', '--only-binary=:all:',
                           *specs, '-c', str(_ML_REQUIREMENTS)])
    if rc != 0:
        return rc
    for feature in features:
        if not _verify_capability_import(feature, python, log_action=action):
            return 1
    for feature in features:
        if not _select_managed_python(action, feature, python):
            return 1
    return 0


def _extra_roots_for(comfy_type):
    """Generic configured extra model roots; product-derived roots are callbacks."""
    from .services import comfy_model_paths
    return comfy_model_paths.extra_roots(comfy_type)


def _log_denied(action, status, license_url, provider):
    """The recovery steps for a 401/403, the same four lines whichever file of
    an action the host refused — a companion can be gated while its main file
    was not (the INT8 row's adapters live on another repository)."""
    if provider == 'none':
        # A file LDS serves itself takes no credential: there is no key to create.
        _append(action, f'HTTP {status} - the LDS download server refused this file. Retry later.')
        return
    host, key_url, key_name, verb = _AUTH_RECOVERY.get(provider, _AUTH_RECOVERY['hf'])
    _append(action, f'HTTP {status} - {host} denied access to this file.')
    if license_url:
        _append(action, f'1. Open {license_url} and {verb} continue')
    _append(action, f'2. Create an API key at {key_url}')
    _append(action, f'3. Paste it as {key_name} in Settings -> API keys, then retry')
    _append(action, '   (or download the file manually into the folder above)')


def _companion_dest_path(comp) -> str:
    return os.path.join(_comfyui_root(), 'models', *comp['dest'])


def _companion_unusable_reason(comp, path):
    """Why the companion at `path` cannot be kept, or None. A JSON file is
    kept when it parses; a weight when the shared validator does not condemn
    it — an HTML page named adapter_model.safetensors is the failure mode."""
    if comp.get('kind') == 'json':
        try:
            with open(path, encoding='utf-8') as fh:
                json.load(fh)
            return None
        except (OSError, ValueError) as exc:
            return f'not a readable JSON file ({exc})'
    return _unloadable_reason('companion', path, comp)


def _companion_lock(companions):
    with _lock:
        key = tuple(sorted(tuple(item['dest']) for item in companions))
        return _COMPANION_LOCKS.setdefault(key, threading.Lock())


def _run_companion_downloads(action) -> int:
    """Fetch every companion of `action` that is absent or unusable, each to a
    .part then renamed, verified by its own kind. Progress restarts per file;
    the log names each. rc 1 on the first failure — a stage is whole or it is
    not, and the branch that already landed stays."""
    spec = model_download_spec(action)
    companions = spec.get('companions') or ()
    if not companions:
        return 0
    with _companion_lock(companions):
        return _fetch_companions(action, spec, companions)


def _fetch_companions(action, spec, companions) -> int:
    headers, provider = _download_auth(spec)
    headers = {**headers, 'Accept-Encoding': 'identity'}
    for comp in companions:
        dest = _companion_dest_path(comp)
        part = dest + '.part'
        try:
            if os.path.isfile(dest):
                reason = _companion_unusable_reason(comp, dest)
                if not reason:
                    _append(action, f'already present: {dest}')
                    continue
                _append(action, f'the companion already here cannot be used: {reason} — replacing it')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _append(action, f"downloading {comp['url']}")
            _append(action, f'-> {dest}')
            with requests.get(comp['url'], stream=True, timeout=network_timeout((10, 120)),
                              headers=headers, allow_redirects=True) as resp:
                if resp.status_code in (401, 403):
                    _log_denied(action, resp.status_code,
                                comp.get('license_url') or spec.get('license_url'), provider)
                    return 1
                if resp.status_code >= 400:
                    _append(action, f'HTTP {resp.status_code} on {os.path.basename(dest)}')
                    return 1
                _stream_model_download(action, resp, comp['url'], part, headers)
            if comp.get('kind') == 'json':
                reason = _companion_unusable_reason(comp, part)
                if reason:
                    _append(action, f'download verification failed: {reason}; retry the download')
                    os.remove(part)
                    return 1
            elif not _verify_downloaded_model(action, part, comp, provider):
                return 1
            os.replace(part, dest)
            _append(action, f'done -> {dest}')
        except requests.RequestException as e:
            _append(action, f'network error: {e}')
            _discard_part(part)
            return 1
        except OSError as e:
            # A filesystem refusal (a file held open, a full disk, a folder
            # that vanished) ends the run with a readable line, not a
            # traceback in the button.
            _append(action, f'could not write {os.path.basename(dest)}: {e} — retry the download')
            _discard_part(part)
            return 1
    return 0


def _discard_part(part):
    try:
        os.remove(part)
    except OSError:
        pass


def _stream_model_download(action, response, url, part, headers):
    from .services.model_download import stream_model
    return stream_model(
        response, url, part, headers=headers, timeout=network_timeout((10, 120)),
        progress=lambda done, total: _set_progress(action, done, total),
        log=lambda line: _append(action, line),
    )


def _run_primary_download(action) -> int:
    """Stream one model asset (Klein or Krea) into the validated ComfyUI tree.
    Writes to a .part file then renames (a killed download never leaves a half
    file the model scanners would pick up), then verifies the result is real
    weights. Aggregate byte progress is published for the UI. An
    access-denied host (401/403) -> actionable recovery steps for THAT provider,
    rc 1."""
    spec = model_download_spec(action)
    dest = _download_dest_path(action)
    # Files judged unusable, deleted ONLY once a replacement exists (see below).
    condemned = []
    if os.path.isfile(dest):
        # "Already present" used to end the story here, on ANY existing file. That
        # made the one remedy the app suggests for a corrupted weight — download it
        # again — a no-op that reported success: the file stayed broken, every
        # screen kept certifying it, and there was no way out of the loop from
        # inside the app (zigzag4794, Discord: a truncated 9.5 GB Klein UNET).
        # So the same validator the readiness probe uses gets asked first, and a
        # BLOCKING verdict (an HTML licence page, a truncated/garbage file) makes
        # this a replacement instead of a skip. The advisory `too_small` never
        # condemns anything — a small-but-loadable file is the user's, not ours.
        reason = _unloadable_reason(action, dest, spec)
        if not reason:
            _append(action, f'already present: {dest}')
            return 0
        diagnosis = ('needs replacing' if reason.startswith('superseded model revision')
                     else 'cannot be loaded')
        _append(action, f'the file already here {diagnosis}: {reason}')
        # It is NOT deleted now. `dest` is written by os.replace(part, dest) at the
        # end of a successful download, which overwrites it atomically, so there is
        # nothing to clear beforehand — and clearing it beforehand is exactly how a
        # 401, an expired token or a dead host turned "you have a broken file" into
        # "you have no file". It only goes if a good copy takes its place.
        _append(action, 'it stays where it is until a fresh copy has actually downloaded')
        condemned.append(dest)
    variant = _variant_already_present(action, condemned)
    if variant:
        # A loadable copy is proven present, so the condemned files can go now:
        # nothing here depends on a download that may never happen.
        _drop_condemned(action, condemned)
        _append(action, f'already present ({variant}) — an earlier build is '
                        'installed and still resolves; skipping download')
        return 0
    if _download_present_in_extra(action):
        _drop_condemned(action, condemned)
        _append(action, 'already available via a configured extra_model_paths.yaml root - skipping download')
        return 0
    if _krea_asset_already_installed(action):
        _drop_condemned(action, condemned)
        _append(action, 'already installed — the engine already resolves this asset from a '
                        'file you have; skipping download')
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers, provider = _download_auth(spec)
    headers = {**headers, 'Accept-Encoding': 'identity'}
    _append(action, f"downloading {spec['url']}")
    _append(action, f'-> {dest}')
    part = dest + '.part'
    try:
        with requests.get(spec['url'], stream=True, timeout=network_timeout((10, 120)),
                          headers=headers, allow_redirects=True) as resp:
            if resp.status_code in (401, 403):
                if spec.get('gated') or spec.get('license_url'):
                    # Normally public; a 401/403 here means the host is denying
                    # access anyway (re-gated, region-restricted, or a stale token
                    # was sent) -> the fix is: get an account/licence + a valid key.
                    _log_denied(action, resp.status_code, spec.get('license_url'), provider)
                else:
                    _append(action, f'HTTP {resp.status_code}')
                return 1
            if resp.status_code >= 400:
                _append(action, f'HTTP {resp.status_code}')
                return 1
            _stream_model_download(action, resp, spec['url'], part, headers)
        # Verify BEFORE the rename: a 200-with-a-login-page must not have already
        # taken the place of whatever was there.
        if not _verify_downloaded_model(action, part, spec, provider):
            return 1
        os.replace(part, dest)
        # The replacement is on disk and verified: NOW the old copies may go. `dest`
        # itself was already overwritten atomically above, so it is spared here —
        # removing it would delete the file we just downloaded.
        _drop_condemned(action, condemned, keep=dest)
        _append(action, f'done -> {dest}')
        return 0
    except requests.RequestException as e:
        _append(action, f'network error: {e}')
        _discard_part(part)
        return 1
    except OSError as e:
        _append(action, f'could not write {os.path.basename(dest)}: {e} — retry the download')
        _discard_part(part)
        return 1
