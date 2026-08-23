"""Krea 2 Identity Edit — the SECOND local generation engine, next to Klein.

WHAT IT IS
----------
Krea 2 Identity Edit is an identity-preserving edit LoRA for the Krea 2 base
model, driven by the `comfyui-krea2edit` custom-node pack. Given ONE reference
photo it re-stages the subject (angle, framing, light, background) while keeping
the face, the body morphology and the permanent markings — WITHOUT any character
LoRA. That is exactly the bootstrap case the dataset builder needs: a character
who has no LoRA yet.

WHY A HAND-BUILT GRAPH (no shipped workflow JSON)
-------------------------------------------------
Klein loads `workflows/improve skin.json`, a file lifted from a developer's own
ComfyUI whose loader nodes hardcode filenames that exist on no fresh install —
`klein_edit_helper` then has to re-resolve every one of them. Building the Krea
graph in code removes that whole class of bug: every loader value comes from a
resolver, so there is nothing to drift.

THE GRAPH (validated against /object_info on a live install)
-----------------------------------------------------------
    UNETLoader -> LoraLoaderModelOnly(identity edit LoRA) -> Krea2EditModelPatch -> KSampler
    LoadImage  -> VAEEncode -> source_latent  ^                positive: Krea2EditGroundedEncode(prompt, image)
                                                              negative: Krea2EditGroundedEncode('',     image)
    CLIPLoader(qwen3-vl 4B, type='krea2') · VAELoader(qwen image vae) · EmptySD3LatentImage

An OPTIONAL second reference joins the patch as `source_image_b` and both
encodes as `image_b` — the pack's only `_b` slot, and it wants a DIFFERENT
subject (a person or a scene), never another angle of the same face. See
build_workflow for the two traps it carries (no second VAEEncode; ref_boost
changes hands).

BOTH custom nodes are mandatory:
  * `Krea2EditGroundedEncode` is what makes the text encoder actually SEE the
    reference. With a plain `CLIPTextEncode` the reference has no effect at all
    (measured — not a quality nuance, the identity simply does not transfer).
    The NEGATIVE branch is grounded too, exactly like the pack's own reference
    workflow.
  * `Krea2EditModelPatch` injects the source latent (+ the anti-blur pixel path)
    into the model.

MEASURED CONSTRAINTS (2026-07-25, live install — do not "simplify" these away)
------------------------------------------------------------------------------
  * cfg 1.0, 8 steps, euler/simple, edit LoRA at 1.0 — the calibrated
    dataset-restaging profile.
  * Dataset variations can request their catalog card's aspect ratio and stay <=
    2 MP. The installed v1.2 node's ``fit`` geometry supports that target even
    when it differs from the source; free-prompt reference edits keep the source
    aspect by omitting a requested ratio.
  * `BigLoveKreaEdit1_fp8mixed` renders PURE NOISE with this LoRA (it is not a
    Krea 2 Raw/Turbo checkpoint). It is excluded from base-model resolution.

EVERY PIECE IS NOW AUTO-INSTALLED (setup_installer: `krea_model`,
`krea_text_encoder`, `krea_vae`, `krea_identity_lora` and `krea_nodes` — the
first custom-node pack this app clones). Selecting the engine and pressing
Generate fires them, exactly like Klein; Setup ▸ Install has the same button.
This module keeps only the JUDGEMENT (what is missing, what is present but not
loadable, which node classes are absent) — the fetching lives in setup_installer.
The state of the world behind that decision, RE-MEASURED 2026-07-27 (read this
before re-quoting it — the previous version of this paragraph was a photograph of
one moment that became an architecture decision for weeks):

  * The original claim — "weights we have no verified direct URL for" — is
    OBSOLETE. All three Hugging Face pieces live in ONE public, NON-GATED repo,
    `Comfy-Org/Krea-2`, under the exact canonical filenames the resolvers below
    look for: `diffusion_models/krea2_turbo_fp8_scaled.safetensors` (13.1 GB),
    `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` (5.2 GB) and
    `vae/qwen_image_vae.safetensors` (254 MB). Measured: anonymous HTTP 200 on
    each, no token, `gated=false`.
  * The identity LoRA (Civitai 2761113) also downloaded anonymously on that date
    — 307 to a signed CDN URL, then real safetensors bytes, no API key.
  * The node pack declares `dependencies = []`, so installing it is a clone, not
    a pip run.

None of it sits behind a licence checkbox a human has to tick in a browser. That
said, an OPEN download today is not a guarantee: the installer sends a Civitai
key when the user has one, tries without when they do not, and turns a 401/403
into instructions — because Civitai gates part of its catalogue with rules that
have changed before and vary by country.

The preflight still publishes the full `_studio_missing_response` shape (every
gap, its expected path inside the user's ComfyUI, the page it comes from): a
machine that cannot download — offline, proxied, no valid ComfyUI folder — must
still get a complete answer. Never a silent failure, never an inert button.

If you are about to conclude "Krea can't be auto-installed", re-run the
measurement first — that sentence has already been wrong once, for weeks.
"""
from __future__ import annotations
import logging
import os
import random
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

ENGINE_ID = 'krea'
ENGINE_LABEL = 'Krea 2 Edit'

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')

# The two node classes the graph cannot be built without, and the pack shipping
# them. Same shape as lora_test_studio.STUDIO_NODE_PACKS so the 409 body and the
# banner that renders it are identical to the Studio one.
KREA_NODE_CLASSES = ('Krea2EditModelPatch', 'Krea2EditGroundedEncode')
KREA_NODE_PACK = {
    'pack': 'comfyui-krea2edit',
    'url': 'https://github.com/lbouaraba/comfyui-krea2edit',
    'search': 'krea2edit',
}

# Where each asset belongs inside a ComfyUI install, for the "place it here"
# message. Display paths only — the real lookup goes through comfy_model_paths,
# so an extra_model_paths.yaml root works exactly the same.
KREA_ASSETS = {
    'krea_model': {
        'kind': 'Krea 2 base model (Raw or Turbo)',
        'path': 'models/diffusion_models/Krea/',
        'source': 'https://huggingface.co/Comfy-Org/Krea-2/tree/main/diffusion_models',
    },
    'krea_identity_lora': {
        'kind': 'Krea 2 Identity Edit LoRA',
        'path': 'models/loras/krea/krea2_identity_edit_v1_2.safetensors',
        'source': 'https://civitai.com/models/2761113',
    },
    'krea_text_encoder': {
        'kind': 'Qwen3-VL 4B text encoder',
        'path': 'models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/Krea-2/tree/main/text_encoders',
    },
    'krea_vae': {
        'kind': 'Qwen Image VAE',
        'path': 'models/vae/qwen_image_vae.safetensors',
        'source': 'https://huggingface.co/Comfy-Org/Krea-2/tree/main/vae',
    },
}
# Every asset is graph-critical: unlike Klein there is no "quality only" piece —
# without the identity LoRA the model is not an edit model at all.
KREA_REQUIRED = tuple(KREA_ASSETS)


class KreaModelsMissing(Exception):
    """A Krea 2 Edit asset (base model / identity LoRA / text encoder / VAE) is
    not on disk and/or the custom-node pack is absent, so no valid job can be
    built. Raised BEFORE any row or job is created, so the route answers ONE
    actionable 409 instead of a grid of tiles each dying on ComfyUI validation.

    `.missing` = asset keys (subset of KREA_REQUIRED); `.missing_nodes` =
    class_type strings the target ComfyUI does not expose."""

    def __init__(self, missing, missing_nodes=None):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('Krea 2 Edit assets missing: '
                         + ', '.join(self.missing + self.missing_nodes))


class KreaPinnedModelMissing(Exception):
    """A Krea model file the user PINNED in Settings is not on disk.

    Deliberately NOT the same error as KreaModelsMissing, and deliberately not a
    silent fallback to automatic resolution.

    WHY THE ENGINE REFUSES INSTEAD OF FALLING BACK
    ----------------------------------------------
    This resolver used to log a warning and elect a base model on its own when
    `krea.base_model` named a file it could not find. On a shared ComfyUI that is
    how an entire training ran on a third-party finetune nobody chose: the
    Settings field showed one filename, the graph loaded another, and the only
    symptom was images that were subtly not the model asked for — visible after
    the expensive part, not before it.

    A setting that quietly resolves to something OTHER than what it displays is
    the worst failure mode available to us. So a pin that cannot be resolved is a
    STOP, naming the file, and clearing the field is the explicit gesture that
    goes back to auto-detection.

    `.gaps` = [{'slot', 'key', 'configured'}]."""

    def __init__(self, gaps):
        self.gaps = list(gaps or [])
        super().__init__(
            'Krea 2 Edit: pinned model file not found — '
            + '; '.join(f"{g['key']} = {g['configured']!r}" for g in self.gaps)
            + '. Pick a file that exists, or clear the field to use auto-detection.')


# --- Resolution -------------------------------------------------------------
# Canonical-name-first with NARROW token fallbacks, scanning roots in ComfyUI's
# own priority order — the same discipline as klein_edit_helper. A WRONG model is
# worse than a missing one: missing produces an actionable message, wrong dies at
# sample time on a shape mismatch or renders noise.

def _listings(comfy_type):
    out = []
    for folder in comfy_model_paths.search_roots(comfy_type):
        try:
            out.append((folder, sorted(n for n in os.listdir(folder)
                                       if n.lower().endswith(_MODEL_SUFFIXES))))
        except OSError:
            continue
    return out


def _find_model_file(comfy_type, canonical, tokens):
    """Bare filename for a ComfyUI folder type: the canonical name if present in
    ANY search root, else the first (sorted) name containing a NARROW token.
    None when nothing matches — never a blind first-file guess."""
    listings = _listings(comfy_type)
    if any(canonical in names for _root, names in listings):
        return canonical
    for _root, names in listings:
        for n in names:
            if any(tok in n.lower() for tok in tokens):
                return n
    return None


# Checkpoints that carry 'krea' in their name but are NOT a Krea 2 Raw/Turbo
# base: the identity-edit LoRA renders PURE NOISE on top of them (measured on
# BigLoveKreaEdit1_fp8mixed). Excluding them here is what stops the resolver from
# silently picking one on a shared ComfyUI and blaming the user for the result.
KREA_INCOMPATIBLE_TOKENS = ('biglove',)


def _krea_unet_folders():
    """(prefix, [model files]) candidates for the Krea 2 base UNET across every
    diffusion-model search root. `prefix` is a 'krea'-named subfolder (e.g.
    'Krea', 'krea2') or '' for a file dropped straight into the root of a search
    folder — flat / Stability-Matrix installs have no subfolder, and
    os.path.join('', name) == name is exactly what UNETLoader loads. Mirrors
    klein_edit_helper._klein_unet_folders so anything listed is loadable.

    NOTHING IS FILTERED OUT HERE. This walk used to drop every file matching
    `KREA_INCOMPATIBLE_TOKENS`, which meant a build sitting in the user's own
    `Krea/` folder was absent from every list with nothing on screen saying it
    existed or why it was gone. Choosing is the user's call: a measured warning
    belongs next to the name, not in place of it. `elect_krea_base` still keeps a
    flagged build from WINNING an automatic election, which is a different
    question — see there."""
    return comfy_model_paths.scan_family_folders(
        comfy_model_paths.search_roots('diffusion_models'), ('krea',),
        suffixes=_MODEL_SUFFIXES)


def _krea_base_compatible(name):
    low = (name or '').lower()
    return not any(tok in low for tok in KREA_INCOMPATIBLE_TOKENS)


# The basename Setup installs for this family: Comfy-Org's own Krea 2 Turbo
# build. This is the ONE filename the resolver knows, and it is deliberately not
# a "list of known community files" — it is this app's installation contract,
# declared in `setup_installer._KREA_DOWNLOADS['krea_model']` and repeated here so
# the default is the file Setup fetches. `test_krea_install` asserts the two agree,
# so a changed download can never leave the resolver pointing at a name nobody
# installs.
KREA_CANONICAL_UNET = 'krea2_turbo_fp8_scaled.safetensors'

# The sampling regime the Krea graphs are built for. `build_workflow` pins cfg 1.0
# and a few steps, which is what a guidance-distilled ("turbo") build IS; a "raw"
# build renders mush at those settings. So the regime outranks file quality: a
# clean full-precision Raw checkpoint is a WORSE default here than a turbo one,
# and that is not a preference, it is what the graph can drive.
#
# Read that again before "improving" the order, because it is counter-intuitive
# and the failure it prevents is silent: the BEST file at the WRONG sampling
# regime produces failed images, and the user concludes the model is bad. A
# blurry sketch is never read as "this base wanted 25 steps" — it is read as
# "the training did not work".
_KREA_REGIME_TOKENS = ('turbo', 'raw')


def _krea_regime_tier(name: str) -> int:
    """0 = the canonical file Setup installs, 1 = a 'turbo' build, 2 = a 'raw'
    build, 3 = anything else Krea-shaped. Name-level only; the tie inside a tier
    is broken by the file's header (see `_krea_candidates`)."""
    low = (name or '').lower()
    if low == KREA_CANONICAL_UNET.lower():
        return 0
    for i, token in enumerate(_KREA_REGIME_TOKENS):
        if token in low:
            return 1 + i
    return 1 + len(_KREA_REGIME_TOKENS)


def _krea_health_rank(rel_name: str) -> int:
    """Header verdict for one candidate: full precision < plain cast < packed
    export < a file whose own metadata says it carries something that is not
    weights. Unreadable / unresolvable → treated as a plain cast, i.e. neither
    promoted nor punished for being uninspectable."""
    from . import model_integrity
    path = comfy_model_paths.resolve_model_file('diffusion_models', rel_name)
    if not path:
        return model_integrity.HEALTH_BARE_CAST
    return model_integrity.base_health(path)['rank']


def elect_krea_base(candidates):
    """The Krea 2 base to use as a default, out of `candidates` — an ordered list
    of ComfyUI-relative names (``'Krea\\\\x.safetensors'``, or a bare filename at a
    root). None when the list is empty or holds nothing loadable.

    ONE ranking, shared by every surface that has to choose a Krea base, so the
    Generate resolver and the Test Studio can never elect different files from the
    same folder. The order is documented on `resolve_krea_unet`. It reads headers
    ONLY to break a tie inside the best regime tier, so a single-candidate install
    pays no I/O.

    The list this is applied to is the CALLER's — the Studio ranks what its own
    picker offers, Generate ranks what its own resolver found. Ranking a name the
    caller does not list would elect a base its own whitelist then refuses."""
    ranked = [(name, i) for i, name in enumerate(candidates or [])
              if name and comfy_model_paths.is_loadable_model(name)]
    if not ranked:
        return None
    # A build flagged by KREA_INCOMPATIBLE_TOKENS is ELECTABLE but never PREFERRED.
    # The two halves are deliberate. The user picking one is their call and is not
    # filtered anywhere. An automatic election is not a choice at all, and one made
    # in silence has already sent a run onto a third-party finetune nobody noticed
    # until the results were wrong -- so a flagged build only wins when it is the
    # only thing on disk, and the surfaces that name the running base say so.
    preferred = [c for c in ranked
                 if _krea_base_compatible(os.path.basename(str(c[0]).replace('\\', '/')))]
    ranked = preferred or ranked
    tiers = {name: _krea_regime_tier(os.path.basename(str(name).replace('\\', '/')))
             for name, _i in ranked}
    best = min(tiers[name] for name, _i in ranked)
    short = [(name, i) for name, i in ranked if tiers[name] == best]
    if len(short) > 1:
        short.sort(key=lambda c: (_krea_health_rank(c[0]), c[1]))
    return short[0][0]


def resolve_krea_unet(selected=None):
    """ComfyUI-relative `unet_name` for the UNETLoader, WITH its subfolder prefix
    (e.g. 'Krea\\krea2_turbo_fp8_scaled.safetensors'), or None when no compatible
    Krea 2 base is on disk.

    THE ELECTION, in order:

      1. the explicit pick (`selected`, or the `krea.base_model` setting — matched
         on its BASENAME across every krea folder, so a value copied from a
         listing still resolves). A user's choice is never second-guessed;
      2. the SAMPLING REGIME the graph needs: the canonical file Setup installs,
         then a 'turbo' build, then a 'raw' one, then anything else;
      3. inside that tier, what the file's HEADER says: full precision, then a
         plain low-precision cast, then a packed inference export, then — last,
         always — a file whose own metadata declares it carries tensors that are
         not weights (`model_integrity.base_health`). Last, not excluded: an
         install where that file is the ONLY Krea base still gets a default
         rather than a broken engine;
      4. the scan order of the candidate list itself (roots in ComfyUI's own
         priority order, names sorted), so the same install always resolves the
         same file — which is what makes a regenerate reproduce its original
         render.

    Step 3 reads headers, so it runs ONLY when more than one candidate survives
    step 2 — a single-candidate install pays nothing."""
    folders = _krea_unet_folders()
    if not folders:
        return None
    pick = selected or cfg.get('krea.base_model') or ''
    bare_pick = os.path.basename(str(pick).replace('/', os.sep).replace('\\', os.sep))
    if bare_pick:
        for sub, names in folders:
            if bare_pick in names:
                return os.path.join(sub, bare_pick)
        # An explicit pick that is NOT there stops here. It used to fall through
        # to the election below with a log line nobody reads — see
        # KreaPinnedModelMissing for the run that cost us. Returning None makes
        # `krea_missing_assets` report the gap and `preflight` refuse, naming the
        # file the user asked for instead of quietly loading another one.
        logger.warning('krea base model %r is not on disk — refusing to elect a '
                       'different one', pick)
        return None
    return elect_krea_base([os.path.join(sub, n) for sub, names in folders
                            for n in names])


def resolve_krea_text_encoder():
    """`clip_name` for the CLIPLoader (type='krea2'). Canonical
    qwen3vl_4b_fp8_scaled.safetensors, else a NARROW qwen3-vl-4b token match.
    Never a bare 'qwen' match — qwen_3_8b (Klein) and qwen_2.5_vl (Qwen-Image)
    live in the same folder and produce incompatible embeddings."""
    return _find_model_file('text_encoders', 'qwen3vl_4b_fp8_scaled.safetensors',
                            ('qwen3vl_4b', 'qwen3_vl_4b', 'qwen3-vl-4b'))


def resolve_krea_vae():
    """`vae_name` for the VAELoader — canonical qwen_image_vae.safetensors, else a
    narrow qwen-image-vae token match. Never flux2-vae (Klein's)."""
    return _find_model_file('vae', 'qwen_image_vae.safetensors',
                            ('qwen_image_vae', 'qwen-image-vae', 'qwenimage_vae'))


# The LoRA the whole engine hangs on. Config-named first (so a user with a
# different filename just points at it), then a narrow recursive scan of the
# loras roots — the file is commonly dropped in loras/ root or loras/krea/, and
# a rename ('...v1_2' -> '...v1.2') must not make the engine look uninstalled.
_IDENTITY_LORA_TOKENS = ('krea2_identity_edit', 'krea2-identity-edit',
                         'krea2_identity', 'identity_edit')


def _lora_abs(rel_name):
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots('loras'):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def resolve_krea_identity_lora():
    """(relative_name, absolute_path) of the Krea 2 Identity Edit LoRA, or
    (None, None) when it isn't on disk. The relative name is what a LoraLoader
    wants (it stays relative to whichever registered loras root holds it)."""
    configured = (cfg.get('krea.identity_lora') or '').replace('/', os.sep)
    if configured:
        found = _lora_abs(configured)
        if found:
            return configured, found
        # A pin the USER chose is a stop, not a licence to load a different
        # LoRA: the identity LoRA IS the face transfer, and swapping it silently
        # changes every output. But the SHIPPED default is not a choice — it is
        # the canonical download name, and "not there under this name, so scan
        # for the renamed file" is the whole reason that fallback exists. Only a
        # value moved off the default refuses; the default keeps scanning.
        if _is_user_pin('krea.identity_lora'):
            logger.warning('krea.identity_lora %r is not on disk — refusing to '
                           'scan for a different one', configured)
            return None, None
        logger.info('krea.identity_lora %r not found — scanning for the LoRA by name',
                    configured)
    # list_models mirrors ComfyUI's own get_filename_list: every loras search
    # root, recursive, de-duplicated, in priority order — so the rel_name we
    # return is exactly the string a LoraLoader accepts, wherever the file lives.
    for rel, path in sorted(comfy_model_paths.list_models('loras')):
        if any(tok in os.path.basename(rel).lower() for tok in _IDENTITY_LORA_TOKENS):
            return rel, path
    return None, None


def krea_missing_assets():
    """Which Krea assets are NOT on disk, as KREA_ASSETS keys. Disk-only,
    network-free — safe for the readiness probe."""
    missing = []
    if not resolve_krea_unet():
        missing.append('krea_model')
    if not resolve_krea_identity_lora()[0]:
        missing.append('krea_identity_lora')
    if not resolve_krea_text_encoder():
        missing.append('krea_text_encoder')
    if not resolve_krea_vae():
        missing.append('krea_vae')
    return missing


# --- Present-but-INVALID assets ---------------------------------------------
# Exactly Klein's mechanism (klein_edit_helper.klein_invalid_assets), because
# nothing about the failure is Klein-specific. The reason WHY was wrong until
# 2026-07-27: this used to say the Krea 2 base sits behind a Hugging Face licence
# gate and the identity LoRA behind a Civitai login. Re-measured — neither is
# true (see the module docstring: both download anonymously). The check stays,
# because the failure it catches never depended on that story: any interrupted,
# proxied, rate-limited or error-page download saves HTML or a half file to
# <name>.safetensors. It passes krea_missing_assets ("the file is there")
# and then dies at generate time on a raw ComfyUI
# `UNETLoader: Expecting value: line 1 column 1 (char 0)`. A truncated download
# is worse still — it renders silently distorted images with no error anywhere.
#
# Advisory `too_small` floors: deliberately far under the real sizes so a
# legitimate file can never trip them. The identity LoRA is genuinely small
# (tens of MB) and the VAE ~250 MB, so their floors only catch a near-empty stub;
# the structural cases (HTML page, truncation) need no floor at all.
KREA_MIN_BYTES = {
    'krea_model': 1024 ** 3,                # 1 GB   (real ≈ 12-20 GB)
    'krea_text_encoder': 256 * 1024 ** 2,   # 256 MB (real ≈ 4-5 GB)
    'krea_vae': 8 * 1024 ** 2,              # 8 MB   (real ≈ 250 MB)
    'krea_identity_lora': 512 * 1024,       # 512 KB (real ≈ tens of MB)
}


def _abs_under_roots(comfy_type, rel_name):
    """Absolute path of a <comfy_type>-relative model name under the FIRST search
    root that holds it, else None. Mirrors how ComfyUI itself resolves a loader
    value, so the file we validate is exactly the one the loader node would open."""
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(comfy_type):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _krea_asset_paths():
    """{KREA_ASSETS key: absolute path} for each Krea asset PRESENT on disk. The
    resolvers return ComfyUI-relative loader names; this maps each back to the
    concrete file so model_integrity can read its header. Absent assets are
    omitted — krea_missing_assets owns 'missing', this owns 'present'."""
    paths = {}
    for key, comfy_type, rel in (
            ('krea_model', 'diffusion_models', resolve_krea_unet()),
            ('krea_text_encoder', 'text_encoders', resolve_krea_text_encoder()),
            ('krea_vae', 'vae', resolve_krea_vae())):
        p = _abs_under_roots(comfy_type, rel)
        if p:
            paths[key] = p
    _rel, lora_path = resolve_krea_identity_lora()
    if lora_path:
        paths['krea_identity_lora'] = lora_path
    return paths


def krea_invalid_assets():
    """Krea assets that ARE on disk under the resolved name but are NOT real,
    loadable weights — the state BETWEEN 'missing' and 'ready'.

    Returns ``[{asset, filename, verdict, blocking, reason}]``, the same shape
    klein_invalid_assets publishes, so the front end needs no second format.
    Header-only + cached (model_integrity), so it is cheap enough for the
    readiness probe. ``blocking`` True means the file cannot load at all
    (html_or_text / truncated) → the actionable 'delete & re-download'; False is
    the advisory ``too_small``."""
    from . import model_integrity
    out = []
    for asset, path in _krea_asset_paths().items():
        res = model_integrity.validate_model_file(path, min_bytes=KREA_MIN_BYTES.get(asset))
        if res['ok']:
            continue
        out.append({'asset': asset, 'filename': res['filename'],
                    'verdict': res['verdict'], 'blocking': res['blocking'],
                    'reason': res['reason']})
    return out


# --- Custom-node preflight ---------------------------------------------------
# Success-only TTL cache, same contract as klein_missing_nodes: /object_info is
# the heaviest probe in the app, node packs don't uninstall mid-session, and a
# miss is NEVER cached so "install the pack, restart ComfyUI, retry" re-probes at
# once. FAIL-OPEN when ComfyUI can't be reached: a transient probe failure must
# never block a generation (the job would still fail, with ComfyUI's own error).
_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def krea_missing_nodes():
    """[class_type] of the Krea 2 Edit nodes the target ComfyUI does not expose.
    [] when they are all present OR when /object_info is unreachable."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in KREA_NODE_CLASSES if c not in available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def clear_nodes_cache():
    """Drop the success-TTL so the next probe re-asks /object_info. Called right
    after the node pack is installed: the cache only ever holds a POSITIVE result,
    but a stale positive would hide a pack the user removed, and clearing costs
    one probe."""
    global _nodes_ok_until
    _nodes_ok_until = 0.0


def krea_node_pack_installed():
    """Is the pack's folder present in this ComfyUI's custom_nodes? Disk-only.

    This is what separates "you have to install the pack" from "the pack is
    installed, ComfyUI just hasn't been restarted yet" — ComfyUI registers nodes
    at STARTUP, so /object_info keeps reporting them missing until then, and
    without this distinction the app would tell someone to install what they just
    installed. False whenever ComfyUI's folder isn't configured/valid: we then
    genuinely do not know."""
    from .. import capabilities
    r = capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')
    if not r['valid']:
        return False
    folder = os.path.join(r['resolved'], 'custom_nodes', KREA_NODE_PACK['pack'])
    try:
        return os.path.isdir(folder) and any(os.scandir(folder))
    except OSError:
        return False


def krea_node_hints(nodes):
    """[{class_type, pack, url, search}] for each missing node — the shape
    `_studio_missing_response` already publishes and StudioPreflightBanner
    already renders, so the front needs no second format."""
    return [{'class_type': ct, **KREA_NODE_PACK} for ct in (nodes or [])]


def missing_file_entries(missing):
    """[{path, kind, source}] for each missing asset key — again the Studio
    `files` shape, so one banner covers both engines."""
    out = []
    for key in missing or []:
        meta = KREA_ASSETS.get(key)
        if meta:
            out.append({'path': meta['path'], 'kind': meta['kind'],
                        'source': meta['source']})
    return out


# The two Krea slots a user can PIN to an exact file, and the picker slot each
# one feeds.
KREA_PIN_KEYS = {
    'base_model': ('krea.base_model', 'krea_base_model'),
    'identity_lora': ('krea.identity_lora', 'krea_identity_lora'),
}


def _is_user_pin(key: str) -> bool:
    """Is this config value something the USER chose, as opposed to the value the
    app ships?

    The distinction is not pedantry, it is the difference between a helpful
    refusal and a bricked install. `krea.base_model` defaults to '' — any value
    is a choice. `krea.identity_lora` defaults to the canonical DOWNLOAD name, so
    every install on earth has it "set" while nobody typed it; treating that as a
    pin would turn the renamed-download recovery into a hard stop for everyone.
    Klein's consistency LoRA has exactly the same shape (see
    klein_edit_helper.klein_pin_gaps)."""
    section, _, field = key.partition('.')
    raw = str(cfg.get(key) or '').strip()
    if not raw:
        return False
    shipped = str((cfg.DEFAULTS.get(section) or {}).get(field, '') or '').strip()
    return raw != shipped


def krea_pin_gaps():
    """``[{'slot', 'key', 'configured'}]`` for every Krea model file pinned in
    Settings that is NOT on disk. Empty on an install that pins nothing (the
    normal case) and on one whose pins all resolve.

    Read by capabilities (to keep the engine dark and let the UI say WHY) and by
    preflight (to refuse a run). One function, so the badge in Settings, the
    engine card and the refusal can never disagree about which file is missing."""
    gaps = []
    raw_base = (cfg.get('krea.base_model') or '').strip()
    if raw_base and resolve_krea_unet() is None:
        gaps.append({'slot': 'base_model', 'key': 'krea.base_model',
                     'configured': raw_base})
    raw_lora = (cfg.get('krea.identity_lora') or '').strip()
    if _is_user_pin('krea.identity_lora') and resolve_krea_identity_lora()[0] is None:
        gaps.append({'slot': 'identity_lora', 'key': 'krea.identity_lora',
                     'configured': raw_lora})
    return gaps


def preflight():
    """Raise KreaModelsMissing when the engine cannot run. No auto-download: see
    the module docstring — the honest answer is a named gap, not a fake installer.

    A PINNED file that is absent is raised first and separately: "your base model
    is missing" and "the file you chose is not there" send the user to two
    different places, and only the second one is about something they typed."""
    gaps = krea_pin_gaps()
    if gaps:
        raise KreaPinnedModelMissing(gaps)
    missing = krea_missing_assets()
    nodes = krea_missing_nodes()
    if missing or nodes:
        raise KreaModelsMissing(missing, nodes)


# --- Output geometry ---------------------------------------------------------
# Imported, never re-implemented: services/output_geometry.py is the SINGLE
# calculation Klein and Krea both size their results with, so "the two engines
# produce the same canvas" is a property of the code rather than a coincidence
# between two call sites. `variation_output_megapixels` is the shared dial; the
# no-ratio path here is still the free reference edit's source geometry.
from .output_geometry import (            # noqa: E402  (module layout: see above)
    MAX_OUTPUT_MP, fit_output_size, source_size as _source_size, variation_output_megapixels,
)


# --- Settings ----------------------------------------------------------------

def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


GROUNDING_PX_MIN, GROUNDING_PX_MAX = 512, 1536


def grounding_px():
    """THE consistency <-> prompt-adherence dial, in pixels.

    It is the resolution the reference is shown to the Qwen3-VL encoder at. LOW =
    the model reads less of the reference and follows the PROMPT (more variety in
    pose/outfit/scene, weaker likeness); HIGH = it reads more of the reference and
    RESEMBLES it (stronger likeness, but it also starts copying the pose and the
    outfit it was told to change).

    512 is the measured dataset default: it retains identity while allowing pose
    and expression instructions to win. Clamped and snapped to a multiple of 64
    (the encoder's patch grid)."""
    v = _clamp(cfg.get('krea.grounding_px'), GROUNDING_PX_MIN, GROUNDING_PX_MAX, 512.0)
    return int(round(v / 64) * 64)


def _steps():
    return int(_clamp(cfg.get('krea.steps'), 1, 50, 8.0))


def _identity_strength():
    return _clamp(cfg.get('krea.identity_lora_strength'), 0.0, 1.5, 1.0)


def _ref_boost():
    return _clamp(cfg.get('krea.ref_boost'), 0.0, 10.0, 0.25)


# --- Optional always-on generation LoRAs -------------------------------------
# Hard caps, mirroring klein_edit_helper.MAX_GENERATION_LORAS /
# MAX_GENERATION_LORA_PRESETS. The two lanes are deliberate copies (like
# comfyui.inject_krea_loras / inject_zimage_loras / inject_sdxl_loras are): the
# shapes match, the clamps do not. Keep the numbers in step by hand if one moves.
MAX_GENERATION_LORAS = 8          # LoRAs chained per preset
MAX_GENERATION_LORA_PRESETS = 12  # named presets

# Strength ceiling — comfyui.inject_krea_loras' clamp, deliberately NOT Klein's
# 1.5. The utility LoRAs this feature exists for (filter-bypass) have no effect
# below ~10, so a 1.5 ceiling would ship a knob that cannot reach. This is the
# anti-absurd guard only; the 0-6 UX range lives on the front-end slider.
LORA_STRENGTH_MAX = 20.0
DEFAULT_ROW_STRENGTH = 1.0


def configured_generation_lora_presets():
    """Sanitized `krea.generation_lora_presets`: ordered
    [{name, loras: [{file, strength}]}] with blank/duplicate names and
    blank/malformed rows dropped, strengths clamped to [0, LORA_STRENGTH_MAX]
    (junk -> DEFAULT_ROW_STRENGTH), rows capped at MAX_GENERATION_LORAS per
    preset and presets at MAX_GENERATION_LORA_PRESETS.

    THE single source of truth for which files may chain and in what order — a
    /generate request can only NAME a preset from here, never define files or an
    order."""
    raw = cfg.get('krea.generation_lora_presets')
    out, seen = [], set()
    for p in (raw if isinstance(raw, list) else []):
        if not isinstance(p, dict):
            continue
        name = p.get('name')
        name = name.strip() if isinstance(name, str) else ''
        if not name or name in seen:
            continue
        rows = []
        loras = p.get('loras')
        for e in (loras if isinstance(loras, list) else []):
            if not isinstance(e, dict):
                continue
            f = e.get('file')
            f = f.strip() if isinstance(f, str) else ''
            if not f:
                continue
            s = e.get('strength')
            s = float(s) if isinstance(s, (int, float)) else DEFAULT_ROW_STRENGTH
            rows.append({'file': f, 'strength': max(0.0, min(LORA_STRENGTH_MAX, s))})
            if len(rows) >= MAX_GENERATION_LORAS:
                break
        seen.add(name)
        out.append({'name': name, 'loras': rows})
        if len(out) >= MAX_GENERATION_LORA_PRESETS:
            break
    return out


def resolve_generation_lora_preset(preset_name):
    """Ordered [{file, strength}] rows of the CONFIG preset named `preset_name`.

    Fail-closed by design: blank/None/non-string -> [] (no preset picked, the
    default run); an unknown name -> [] with a warning, because a stale UI or a
    preset renamed in Settings must degrade to 'no extra LoRAs' rather than block
    a dataset run. Rows are copies — a caller mutating them cannot corrupt the
    next resolution."""
    name = preset_name.strip() if isinstance(preset_name, str) else ''
    if not name:
        return []
    for p in configured_generation_lora_presets():
        if p['name'] == name:
            return [dict(e) for e in p['loras']]
    logger.warning('krea generation-LoRA preset %r is not configured — '
                   'running with no extra LoRAs', name)
    return []


def _existing_generation_lora_rows(rows, identity_lora=None):
    """The subset of `rows` that can actually chain, in order, capped at
    MAX_GENERATION_LORAS.

    This is where the disk is touched — build_workflow stays pure. Degradation is
    PER ROW and never raises: a preset whose third LoRA was moved still applies
    its other two, because a dataset run must not die over an optional LoRA. Each
    dropped row says why in the log, so a silent no-op is diagnosable.

    `identity_lora`: the resolved relative name already loaded at node 4 (fixed
    identity slot, strength krea.identity_lora_strength). A row pointing at the
    SAME file would chain the identical LoRA a second time — not a no-op like a
    missing file, but the two strengths summing into one delta well past what
    the file was trained for (an 0.8 row on top of the identity slot's default
    1.0 measured as visible macro-blocking, reported by waltm on Discord). Path
    comparison is normcase+normpath so a '/' vs '\\' or case difference can't
    dodge the guard."""
    identity_key = (os.path.normcase(os.path.normpath(identity_lora))
                    if identity_lora else None)
    out = []
    for entry in (rows or []):
        if len(out) >= MAX_GENERATION_LORAS:
            break
        if not isinstance(entry, dict):
            continue
        name = (entry.get('file') or '')
        name = name.replace('/', os.sep) if isinstance(name, str) else ''
        if not name:
            continue
        try:
            strength = max(0.0, min(LORA_STRENGTH_MAX, float(entry.get('strength'))))
        except (TypeError, ValueError):
            strength = 0.0
        if strength <= 0:
            logger.info('krea generation LoRA %r at strength 0 — skipped (row off)',
                        name)
            continue
        if identity_key and os.path.normcase(os.path.normpath(name)) == identity_key:
            logger.warning('krea generation LoRA %r is the identity-edit LoRA — '
                           'skipped (already applied at krea.identity_lora_strength, '
                           'a second copy would double-stack it)', name)
            continue
        if not _lora_abs(name):
            logger.warning('krea generation LoRA %r not found under any loras root '
                           '— skipped', name)
            continue
        out.append({'file': name, 'strength': strength})
    return out


# --- Graph -------------------------------------------------------------------

def build_workflow(source_image, prompt, *, unet, clip, vae, lora_name,
                   width, height, seed, steps=None, grounding=None,
                   ref_boost=None, lora_strength=None, fit_mode='fit',
                   filename_prefix='krea_edit', generation_loras=None,
                   extra_source_image=None):
    """The ComfyUI API-format graph. Pure function of its arguments — no config
    read, no disk access — so a test can assert the exact wiring without a
    ComfyUI, and every loader value is one a resolver produced.

    cfg is pinned to 1.0 and the sampler to euler/simple: the pack's reference
    workflow, and a guidance-distilled model ignores anything else. The NEGATIVE
    branch is a grounded encode of the EMPTY prompt, not a bare CLIPTextEncode —
    that is what the reference workflow does and what the model expects.

    `generation_loras` (optional): ordered [{file, strength}] rows of the run's
    always-on LoRA preset, chained between the identity LoRA (4) and the model
    patch (7). Rows arrive ALREADY existence-checked and clamped from
    enqueue_krea_edit — this function must stay pure, so it neither reads config
    nor touches the disk. None/[] leaves the graph exactly as it was.

    `extra_source_image` (optional): the ComfyUI input filename of a SECOND
    reference. The node pack exposes exactly one extra slot (`source_image_b` /
    `image_b`) — there is no third, so this is a single name, not a list."""
    steps = 8 if steps is None else max(1, int(steps))
    grounding = 512 if grounding is None else int(grounding)
    ref_boost = 0.25 if ref_boost is None else float(ref_boost)
    lora_strength = 1.0 if lora_strength is None else float(lora_strength)
    g = {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': unet, 'weight_dtype': 'default'},
              '_meta': {'title': 'Krea 2 base model'}},
        '2': {'class_type': 'CLIPLoader',
              'inputs': {'clip_name': clip, 'type': 'krea2', 'device': 'default'},
              '_meta': {'title': 'Qwen3-VL text encoder'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': vae}},
        '4': {'class_type': 'LoraLoaderModelOnly',
              'inputs': {'lora_name': lora_name, 'strength_model': lora_strength,
                         'model': ['1', 0]},
              '_meta': {'title': 'Krea 2 Identity Edit LoRA'}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': source_image}},
        '6': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['5', 0], 'vae': ['3', 0]}},
        # Source latent in context + the pixel path that keeps the result sharp.
        '7': {'class_type': 'Krea2EditModelPatch',
              'inputs': {'model': ['4', 0], 'source_latent': ['6', 0],
                         'ref_boost': ref_boost, 'ref_boost_a': 1.0,
                         'fit_mode': fit_mode, 'vae': ['3', 0],
                         'source_image': ['5', 0]}},
        '8': {'class_type': 'Krea2EditGroundedEncode',
              'inputs': {'clip': ['2', 0], 'prompt': prompt, 'image': ['5', 0],
                         'grounding_px': grounding},
              '_meta': {'title': 'Positive (grounded on the reference)'}},
        '9': {'class_type': 'Krea2EditGroundedEncode',
              'inputs': {'clip': ['2', 0], 'prompt': '', 'image': ['5', 0],
                         'grounding_px': grounding},
              '_meta': {'title': 'Negative (grounded, empty prompt)'}},
        '10': {'class_type': 'EmptySD3LatentImage',
               'inputs': {'width': int(width), 'height': int(height), 'batch_size': 1}},
        '11': {'class_type': 'KSampler',
               'inputs': {'seed': seed, 'steps': steps, 'cfg': 1.0,
                          'sampler_name': 'euler', 'scheduler': 'simple',
                          'denoise': 1.0, 'model': ['7', 0], 'positive': ['8', 0],
                          'negative': ['9', 0], 'latent_image': ['10', 0]}},
        '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['3', 0]}},
        '13': {'class_type': 'SaveImage',
               'inputs': {'filename_prefix': filename_prefix, 'images': ['12', 0]}},
    }
    # --- Optional SECOND reference (the pack's `_b` slots) ---------------------
    # Wired EXACTLY like the pack's own two-reference workflow
    # (workflows/krea2_identity_edit.json): the extra image reaches the patch and
    # BOTH grounded encodes — the negative included, because that is the trained
    # unconditional.
    #
    # Two things here are counter-intuitive and were read off the node source, not
    # guessed. Do not "complete" either of them:
    #
    #  1. NO second VAEEncode. The pack also exposes `source_latent_b`, and its
    #     reference workflow connects it — but `Krea2EditModelPatch.patch` rebuilds
    #     `src` from the PIXEL path whenever `vae` + `source_image` are both
    #     connected, which this graph always does. `source_latent_b` is therefore
    #     read only when the pixel path is off; adding it here would encode an
    #     image on every render for a value the node never looks at.
    #
    #  2. `ref_boost_a` is raised to the SAME value as `ref_boost`. The node
    #     applies `ref_boost` to the LAST reference and `ref_boost_a` to the
    #     FIRST. With one reference the primary IS the last one, so it gets the
    #     configured krea.ref_boost. Adding a second reference silently demotes
    #     the primary to "first" — without this line the user's tuned value would
    #     jump onto the second reference and the primary would fall back to the
    #     hardcoded 1.0. Keeping both equal means adding an angle changes what the
    #     model sees, never how hard it pulls on the reference already there.
    if extra_source_image:
        g['14'] = {'class_type': 'LoadImage',
                   'inputs': {'image': extra_source_image},
                   '_meta': {'title': 'Second reference (a different subject / scene)'}}
        g['7']['inputs']['source_image_b'] = ['14', 0]
        g['7']['inputs']['ref_boost_a'] = ref_boost
        g['8']['inputs']['image_b'] = ['14', 0]
        g['9']['inputs']['image_b'] = ['14', 0]
    # Always-on generation LoRAs: chained AFTER the identity LoRA and BEFORE the
    # model patch, in list order, so the patch (and through it the KSampler) sees
    # the whole stack. Named node keys — this graph is ours, unlike Klein's
    # foreign JSON, so they cannot collide with '1'..'13' and they read in a
    # ComfyUI error message.
    prev = '4'
    for i, entry in enumerate((generation_loras or [])[:MAX_GENERATION_LORAS], start=1):
        node_id = f'gen_lora_{i}'
        g[node_id] = {
            'class_type': 'LoraLoaderModelOnly',
            'inputs': {'lora_name': entry['file'],
                       'strength_model': float(entry['strength']),
                       'model': [prev, 0]},
            '_meta': {'title': f"Generation LoRA {i}: "
                               f"{os.path.basename(entry['file'])}"},
        }
        prev = node_id
    if prev != '4':
        g['7']['inputs']['model'] = [prev, 0]
    return g


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_krea_edit(user_id, source_filename, edit_prompt, source_path=None,
                      extra_metadata=None, krea_model=None, aspect_ratio=None,
                      generation_loras=None, extra_ref_paths=None,
                      output_megapixels=None):
    """Copy the reference into ComfyUI's input folder, build the Krea 2 Edit
    graph against what is ACTUALLY installed, and enqueue it. Returns the app
    job_id.

    Raises KreaModelsMissing when an asset or a node is absent (checked BEFORE
    anything is copied or queued), ValueError on a missing source, RuntimeError
    when ComfyUI isn't configured.

    `output_megapixels`: pixel budget for the result. None with an
    `aspect_ratio` (a dataset card) spends the shared
    `variations.output_megapixels` dial — the same one the Klein lane spends, so
    the two engines fill a dataset with identically sized shots.

    `extra_ref_paths`: the SECOND subject, in order — one is used, because
    `Krea2EditModelPatch` and `Krea2EditGroundedEncode` each expose a single
    extra slot (`_b`) and no more. Anything beyond it is dropped here rather than
    silently at graph-build time. A path that no longer exists is skipped with a
    log line, never fatal: an edit must not die over an optional reference.

    WHAT belongs here, per the pack: a DIFFERENT subject — "scene first, subject
    second", "two distinct people ... subject B on the `_b` inputs". Another
    angle of the SAME person is off-label and comes back duplicated. That is why
    the caller feeds this from the edit DIALOG and never from the dataset's extra
    references, which are angles of one face by construction (see
    face_dataset_service.LOCAL_EDIT_REF_SUPPORT). Wiring it to the dataset pool
    was this feature's first shape, and it guaranteed the wrong photo.

    `generation_loras`: ordered [{file, strength}] rows of the run's always-on
    LoRA preset (already resolved from config by the caller — a request only ever
    names a preset). Rows whose file is absent, or whose strength is 0, are
    dropped individually with a log line."""
    if source_path is None:
        out_dir = cfg.comfyui_dir('output')
        if not out_dir:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(str(out_dir), source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f'source image not found: {source_filename}')

    preflight()
    unet = resolve_krea_unet(krea_model)
    clip = resolve_krea_text_encoder()
    vae = resolve_krea_vae()
    lora_name, _lora_path = resolve_krea_identity_lora()

    # Same filesystem hand-off (and the same guard) as the Klein lane: the URL
    # being up says nothing about ComfyUI's input folder being reachable from here.
    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    source_stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    staged_source = comfy_fs.stage_input_image(
        source_path, f'krea_source_{uid}_{source_stem}.png', comfy_input_dir)
    comfy_input = os.path.basename(staged_source)

    # The single second subject, staged the same way. Only the basename is logged:
    # a dataset path names a machine and the log is what users paste into a bug
    # report.
    extra_input = None
    for candidate in (extra_ref_paths or []):
        if not candidate or not os.path.exists(candidate):
            # Says "skipped", not "running with the primary alone": a caller may
            # hand over several candidates and a later one can still be usable.
            logger.warning('krea: extra reference is not on disk — skipped: %s',
                           os.path.basename(str(candidate or '')))
            continue
        staged_extra = comfy_fs.stage_input_image(
            candidate, f'krea_ref_b_{uid}.png', comfy_input_dir)
        extra_input = os.path.basename(staged_extra)
        break

    # Measure the exact sanitized, EXIF-oriented PNG handed to the graph, never
    # the raw camera raster whose aspect can be transposed by orientation 5–8.
    # A dataset card requests its own output ratio AND spends the shared
    # `variations.output_megapixels` budget; free-prompt reference edits omit the
    # ratio and keep this source geometry at the model's own ceiling.
    budget = (float(output_megapixels) if output_megapixels is not None
              else variation_output_megapixels() if aspect_ratio
              else MAX_OUTPUT_MP)
    width, height = fit_output_size(*_source_size(staged_source), max_mp=budget,
                                    requested_aspect=aspect_ratio)
    workflow = build_workflow(
        comfy_input, edit_prompt, unet=unet, clip=clip, vae=vae,
        lora_name=lora_name, width=width, height=height,
        seed=random.randint(0, 2 ** 64 - 1), steps=_steps(),
        grounding=grounding_px(), ref_boost=_ref_boost(),
        lora_strength=_identity_strength(),
        generation_loras=_existing_generation_lora_rows(generation_loras, identity_lora=lora_name),
        extra_source_image=extra_input,
        # UNIQUE prefix per job: SaveImage numbers from what is currently in the
        # output folder and the app moves each result out right after completion,
        # so a shared prefix makes the counter re-issue the same name (the Klein
        # bug — every tile displaying the same file).
        filename_prefix=f'{user_id}_DatasetKrea_{uid}')

    job_id = str(uuid.uuid4())
    meta = {'model_name': 'krea_identity_edit_dataset'}
    if extra_metadata:
        meta.update(extra_metadata)
    # Dropped again when the job ends — the second reference is staged too, so it
    # has to be listed or it would linger in ComfyUI's input folder for good.
    meta['staged_inputs'] = [comfy_input] + ([extra_input] if extra_input else [])
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=edit_prompt,
                          job_id=job_id, metadata=meta)
    return job_id


# The PROMPT side of this engine (the outfit palette, the markings lock, the
# wrapper itself) lives in face_variations — the pure, DB-free, Flask-free module
# that owns every prompt in the app. Nothing prompt-shaped belongs here.
