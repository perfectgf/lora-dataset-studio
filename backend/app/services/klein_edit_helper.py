"""Focused helper to enqueue a SINGLE Flux.2 Klein edit job.

Reused by the face-dataset fan-out. Builds a minimal job from the same
single-edit workflow /generate_edit uses in single mode (WORKFLOW_IMPROVE_SKIN_PATH,
nodes 52=LoadImage, 6=CLIPTextEncode prompt, 77=KSampler seed, 9=SaveImage,
114=UNETLoader Klein). Deliberately a small focused helper rather than a refactor
of the large live generate_edit route.

Node preflight: the shipped 'improve skin.json' is written to use ONLY core +
comfy_extras nodes (the prompt text goes straight into the CLIPTextEncode widget,
and the image-size wiring skips the rgthree 'Any Switch' passthroughs) so a stock
ComfyUI with zero custom-node packs can run it. `klein_missing_nodes()` re-checks
this against the live /object_info as a safety net — if a reverted/edited workflow
or a future change reintroduces a custom node the target ComfyUI lacks, the route
answers one actionable "install pack X, restart ComfyUI" 409 instead of ComfyUI's
raw 400 'missing_node_type'.

Widget VALUES are part of that portability, and this is where it was missed
(2026-07-27, reported by IndependentProcess0 on Reddit). Node 77 used to sample
with `scheduler: "beta57"`, which is NOT a ComfyUI scheduler: the RES4LYF pack
appends it to the CORE list at import (`SCHEDULER_NAMES.append("beta57")`), so on
the machine this workflow was captured on even a plain KSampler accepted it — and
on every install without that pack, generation died with ComfyUI's raw
"Value not in list: scheduler". The graph contained no third-party NODE, so the
preflight above saw nothing wrong. It now uses `simple`, which every ComfyUI has
and which four other shipped graphs already use. Do not "improve" it back to a
value that works on your machine without checking it against a stock install —
`backend/tests/test_workflow_portability.py` enforces exactly that, offline.

Lifted from the parent project's app/services/klein_edit_helper.py for LoRA
Dataset Studio: SRC's module-level COMFYUI_INPUT_DIR/COMFYUI_OUTPUT_DIR constants
become live `cfg.comfyui_dir(...)` calls (config.json changes take effect without
a restart, and an unconfigured ComfyUI raises RuntimeError instead of writing
into a hardcoded path), and the hardcoded consistency-LoRA filename/strength are
now the `klein.consistency_lora` / `klein.consistency_strength` settings.
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
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

WORKFLOW_IMPROVE_SKIN_PATH = cfg.BACKEND_DIR / 'workflows' / 'improve skin.json'

# Nodes this helper rewires — fail LOUDLY if the workflow file changes shape
# instead of silently enqueuing a job with the wrong source/prompt/model.
# Node 6 = CLIPTextEncode: the per-job prompt is written straight into its `text`
# widget (the RES4LYF TextBox1 node 145 that used to hold it was removed so the
# graph needs no custom-node packs).
_REQUIRED_NODES = ('52', '6', '77', '9', '114', '10', '90')

# The Klein pipeline's model dependencies, keyed by the setup_installer download
# action that provides each. REQUIRED = the graph is invalid without it (block +
# auto-download); RECOMMENDED = quality only (the consistency LoRA — degrade).
KLEIN_REQUIRED = ('klein_model', 'klein_text_encoder', 'klein_vae')
KLEIN_RECOMMENDED = ('klein_lora', 'klein_enhancement_lora')
# The detail LoRA node 139 of the improve workflow loads. Kept as a constant
# because the node is BYPASSED when the file is absent, so its presence is what
# decides whether the "Upscale & improve" enhancement strength does anything.
ENHANCEMENT_LORA_NAME = os.path.join('klein', 'realistic.safetensors')

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')


class KleinModelsMissing(Exception):
    """A graph-critical Klein asset (UNET / text-encoder / VAE) is not on disk, so
    a valid job can't be built. `.missing` lists ALL absent assets (incl. the
    optional consistency LoRA) as setup_installer action names, so the caller can
    auto-download them instead of firing a doomed ComfyUI job."""
    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__('Klein models missing: ' + ', '.join(self.missing))


# Canonical filenames = the exact files the Setup installer downloads (single
# source of truth: setup_installer._KLEIN_DOWNLOADS dest names). Matching MUST be
# canonical-first with NARROW token fallbacks: on a ComfyUI shared with other
# apps, models/text_encoders/ holds many families' encoders and a loose
# "contains 'qwen'" match wired Z-Image's qwen3vl_4b TE (sorts before
# qwen_3_8b_fp8mixed: '3' < '_') into the Klein graph -> KSampler died on
# "mat1 and mat2 shapes cannot be multiplied". Wrong model >> missing model:
# missing triggers the auto-download, wrong fails at runtime with a cryptic error.
def _canonical_name(action):
    from .. import setup_installer
    return setup_installer._KLEIN_DOWNLOADS[action]['dest'][-1]


def _find_model_file(comfy_type, canonical, tokens):
    """Model file for a ComfyUI folder type ('vae' / 'text_encoders'): the canonical
    filename if present in ANY search root, else the first (sorted) name containing a
    NARROW token, scanning roots in ComfyUI's priority order (base <models>/<type>
    first, then extra_model_paths roots). None when nothing matches — never a blind
    first-file guess. Returns the bare filename: a VAE/CLIP loader lists these files
    relative to the (registered) folder that holds them, so the bare name loads.
    With no extra_model_paths.yaml this scans exactly the one base folder as before."""
    listings = []
    for folder in comfy_model_paths.search_roots(comfy_type):
        try:
            listings.append(sorted(n for n in os.listdir(folder)
                                   if n.lower().endswith(_MODEL_SUFFIXES)))
        except OSError:
            continue
    if any(canonical in names for names in listings):
        return canonical
    for names in listings:
        for n in names:
            if any(tok in n.lower() for tok in tokens):
                return n
    return None


def _klein_unet_folders():
    """(prefix, [model files]) candidates for the Klein UNET across each diffusion-
    model search root (base models/unet + models/diffusion_models, then any
    extra_model_paths roots) — mirrors capabilities._scan_models, so anything the
    picker lists is resolvable here. `prefix` is a 'klein'-named subfolder (e.g.
    'klein', 'Flux2 klein') OR '' for a file dropped straight into the root of a
    search folder: flat / Stability-Matrix installs place the model in
    diffusion_models/ with no klein/ subfolder, and os.path.join('', name) == name
    is exactly what UNETLoader loads for a root-level file. Per root, subfolders
    first (sorted), then the root entry; across roots, base unet/ first (the
    canonical download location), then shared-install folders, then extra roots.
    With no extra_model_paths.yaml the roots are exactly [unet, diffusion_models]."""
    out = []
    for base_dir in comfy_model_paths.search_roots('diffusion_models'):
        try:
            entries = os.listdir(base_dir)
        except OSError:
            continue
        subs = sorted(d for d in entries
                      if 'klein' in d.lower() and os.path.isdir(os.path.join(base_dir, d)))
        for sub in subs:
            try:
                names = sorted(n for n in os.listdir(os.path.join(base_dir, sub))
                               if n.lower().endswith(_MODEL_SUFFIXES))
            except OSError:
                continue
            if names:
                out.append((sub, names))
        root_names = sorted(n for n in entries
                            if 'klein' in n.lower() and n.lower().endswith(_MODEL_SUFFIXES)
                            and os.path.isfile(os.path.join(base_dir, n)))
        if root_names:
            out.append(('', root_names))
    return out


class KleinModelGone(ValueError):
    """A model was NAMED for this job and is not on disk any more.

    Deliberately not a fallback. `resolve_klein_unet(selected)` degrades an
    unknown pick to the canonical download, which is the right behaviour when the
    pick is a stale browser preference nobody typed — but not when it is a stored
    dataset setting: the user chose a model, the job silently ran on a different
    one, and the result looks fine, so nothing ever tells them. Named exception so
    the routes can say WHICH file went missing rather than "improve failed"."""

    def __init__(self, name):
        self.name = name
        super().__init__(
            f'the Klein model chosen for this dataset is no longer on disk: {name} '
            '— pick another one, or put the file back')


def klein_model_on_disk(selected):
    """The loader-relative `unet_name` for the BARE model file `selected` (e.g.
    'klein\\flux-2-klein-9b-fp8.safetensors'), or None when no search root holds a
    file by that name. This is the STRICT half of resolve_klein_unet: it answers
    "is exactly this file there?" and never falls back to another one.

    Scans the same roots as the picker's list (capabilities._scan_models) so a
    model the UI offers is always one this can name — see
    tests/test_improve_klein_model_choice.py, which asserts both ends layout by
    layout."""
    bare = os.path.basename(selected) if selected else None
    if not bare:
        return None
    for sub, names in _klein_unet_folders():
        if bare in names:
            return os.path.join(sub, bare)
    return None


def resolve_klein_unet(selected=None):
    """ComfyUI-relative `unet_name` for node 114, or None if no Klein model is on
    disk. Returns the value WITH its subfolder prefix (e.g.
    'klein\\flux-2-klein-9b-fp8.safetensors' or 'Flux2 klein\\...-kv-fp8.safetensors'):
    a UNETLoader lists files relative to models/unet (or diffusion_models), so the
    bare filename the picker sends is not loadable on its own — the missing
    subfolder prefix was the original bug. Preference: the picker's choice
    (searched across ALL klein folders), then the canonical download, then the
    first file found."""
    folders = _klein_unet_folders()
    if not folders:
        return None
    bare_pick = os.path.basename(selected) if selected else None
    if bare_pick:
        for sub, names in folders:
            if bare_pick in names:
                return os.path.join(sub, bare_pick)
    canonical = _canonical_name('klein_model')
    for sub, names in folders:
        if canonical in names:
            return os.path.join(sub, canonical)
    # Last-resort pick: only from files a loader can actually OPEN. Listing and
    # loading are different questions, and conflating them let the Krea resolver
    # choose a .gguf on its own (see comfy_model_paths.is_loadable_model). An
    # explicit pick above is still honoured as-is -- if the user names a file,
    # the failure must be about THEIR file, not silently about another one.
    for sub, names in folders:
        for n in names:
            if comfy_model_paths.is_loadable_model(n):
                return os.path.join(sub, n)
    return None


def unet_for_job(klein_model=None):
    """The `unet_name` ONE Klein job must load, given what it was told to run on.

    The single place that turns "this job names a model" / "this job names none"
    into a loader value, so every Klein lane answers the question identically:

    * a NAMED model is a promise — `klein_model_on_disk` or nothing. A file that
      has left the disk raises KleinModelGone instead of quietly resolving to the
      canonical download, because the result of that swap is indistinguishable
      from a correct one;
    * NO name keeps the historical auto-resolution, byte for byte — which is what
      a dataset that never chose (and every bank, which has nothing to choose
      with) must keep getting.

    Extracted from enqueue_klein_edit so the watermark-inpaint lane could stop
    calling resolve_klein_unet() with no argument: it was the last place where a
    stored dataset pick was ignored AND a vanished model was swapped in silence."""
    if not klein_model:
        return resolve_klein_unet()
    unet_ref = klein_model_on_disk(klein_model)
    if not unet_ref:
        raise KleinModelGone(klein_model)
    return unet_ref


def resolve_klein_vae():
    """`vae_name` for node 10 — canonical flux2-vae.safetensors, else a narrow
    flux2-vae token match (covers the 'flux2_vae.safetensors.safetensors'
    double-extension variant some installs carry). Never e.g. qwen_image_vae."""
    return _find_model_file('vae', _canonical_name('klein_vae'),
                            ('flux2-vae', 'flux2_vae', 'flux-2-vae', 'flux_2_vae'))


def resolve_klein_text_encoder():
    """`clip_name` for node 90 — canonical qwen_3_8b_fp8mixed.safetensors, else a
    narrow qwen_3_8b token match. NEVER a bare 'qwen' match: qwen3vl_* (Z-Image)
    and qwen_2.5_vl_* (Qwen-Image) encoders live in the same folder and produce
    incompatible embeddings."""
    return _find_model_file('text_encoders', _canonical_name('klein_text_encoder'),
                            ('qwen_3_8b', 'qwen3_8b', 'qwen-3-8b'))


def _lora_abs(rel_name):
    """Absolute path of a loras-relative name under the FIRST loras search root that
    holds it (base models/loras, then extra_model_paths loras roots), else None. The
    name passed to a LoraLoader stays relative to whichever root contains it — ComfyUI
    lists it identically from any registered loras folder. No yaml → base root only."""
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots('loras'):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _configured_lora(cfg_key):
    """(relative_name, absolute_path) of a loras-relative LoRA named by config key
    `cfg_key`. The path is where it was FOUND (base or an extra_model_paths loras
    root); when absent it is the base-expected path (for a stable log message), or
    None when no loras root exists. The relative name is unchanged (e.g.
    'klein\\Flux2-...safetensors') — it carries 'klein/' when the file lives under
    loras/klein/, which is exactly the string a LoraLoader wants regardless of
    which registered loras root holds it."""
    name = (cfg.get(cfg_key) or '').replace('/', os.sep)
    if not name:
        return None, None
    found = _lora_abs(name)
    if found:
        return name, found
    roots = comfy_model_paths.search_roots('loras')
    return name, (os.path.join(roots[0], name) if roots else None)


def _consistency_lora():
    """(relative_name, absolute_path) of the configured consistency LoRA."""
    return _configured_lora('klein.consistency_lora')


# Hard caps on the optional generation-LoRA presets (Idea by @waltm): bound
# the graph size and the Settings UI alike — the UI shows the same numbers.
MAX_GENERATION_LORAS = 8          # LoRAs chained per preset
MAX_GENERATION_LORA_PRESETS = 12  # named presets


def configured_generation_lora_presets():
    """Sanitized `klein.generation_lora_presets`: ordered
    [{name, loras: [{file, strength}]}] with blank/duplicate names and
    blank/malformed rows dropped, strengths clamped to [0, 1.5] (junk -> the
    0.6 default), rows capped at MAX_GENERATION_LORAS per preset and presets
    capped at MAX_GENERATION_LORA_PRESETS. This is the single source of truth
    for WHICH files may chain and in WHAT order — a /generate request can only
    NAME a preset that exists here, never define files or an order."""
    raw = cfg.get('klein.generation_lora_presets')
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
            s = float(s) if isinstance(s, (int, float)) else 0.6
            rows.append({'file': f, 'strength': max(0.0, min(1.5, s))})
            if len(rows) >= MAX_GENERATION_LORAS:
                break
        seen.add(name)
        out.append({'name': name, 'loras': rows})
        if len(out) >= MAX_GENERATION_LORA_PRESETS:
            break
    return out


def resolve_generation_lora_preset(preset_name):
    """Ordered [{file, strength}] rows of the CONFIG preset named
    `preset_name` — the per-run request can only NAME a preset, never define
    files/strengths/order (fail-closed). Blank/None -> [] (no preset picked,
    the default run); an unknown name -> [] with a warning (a stale UI or a
    renamed preset must degrade to 'no extra LoRAs', never block the run)."""
    name = preset_name.strip() if isinstance(preset_name, str) else ''
    if not name:
        return []
    for p in configured_generation_lora_presets():
        if p['name'] == name:
            return [dict(e) for e in p['loras']]
    logger.warning("generation-LoRA preset %r not found in config — ignored", name)
    return []


def klein_missing_assets():
    """Which Klein assets are NOT on disk, as setup_installer action names (a
    subset of KLEIN_REQUIRED + KLEIN_RECOMMENDED). Drives both the generate-time
    block and the auto-download."""
    missing = []
    if not resolve_klein_unet():
        missing.append('klein_model')
    if not resolve_klein_text_encoder():
        missing.append('klein_text_encoder')
    if not resolve_klein_vae():
        missing.append('klein_vae')
    _, lora_path = _consistency_lora()
    if not (lora_path and os.path.exists(lora_path)):
        missing.append('klein_lora')
    if not _lora_abs(ENHANCEMENT_LORA_NAME):
        missing.append('klein_enhancement_lora')
    return missing


# Minimum plausible on-disk size per Klein asset, for model_integrity's advisory
# `too_small` floor (a partial download / broken symlink that still carries a
# complete header). Deliberately conservative — well under the real size — so it
# never cries wolf on a legitimate file: an fp8 9B UNET is ~9-11 GB and the 8B
# text-encoder ~8 GB, while the VAE (~330 MB) and the consistency LoRA (tens of
# MB) are genuinely small, so their floors only catch a near-empty stub. The hard
# "this isn't weights at all" cases (an HTML gate page, a truncated download) are
# caught structurally and need no floor.
KLEIN_MIN_BYTES = {
    'klein_model': 1024 ** 3,               # 1 GB   (real ≈ 9-11 GB)
    'klein_text_encoder': 512 * 1024 ** 2,  # 512 MB (real ≈ 8 GB)
    'klein_vae': 8 * 1024 ** 2,             # 8 MB   (real ≈ 330 MB)
    'klein_lora': 512 * 1024,               # 512 KB (real ≈ tens of MB)
}


def _abs_under_roots(comfy_type, rel_name):
    """Absolute path of a <comfy_type>-relative model name (as the resolvers return
    it — bare or subfolder-prefixed) under the FIRST search root that holds it, else
    None. Mirrors how ComfyUI itself resolves a loader value, so the file we validate
    is exactly the one the loader node would open."""
    if not rel_name:
        return None
    for root in comfy_model_paths.search_roots(comfy_type):
        cand = os.path.join(root, rel_name)
        if os.path.exists(cand):
            return cand
    return None


def _klein_asset_paths():
    """{setup_installer action name: absolute path} for each Klein asset PRESENT on
    disk. The resolvers return ComfyUI-relative loader names; this maps each back to
    the concrete file so model_integrity can read its header. Absent assets are
    omitted — klein_missing_assets() owns the 'missing' case; this owns 'present'."""
    paths = {}
    unet = resolve_klein_unet()
    if unet:
        p = _abs_under_roots('diffusion_models', unet)   # same roots _klein_unet_folders scans
        if p:
            paths['klein_model'] = p
    te = resolve_klein_text_encoder()
    if te:
        p = _abs_under_roots('text_encoders', te)
        if p:
            paths['klein_text_encoder'] = p
    vae = resolve_klein_vae()
    if vae:
        p = _abs_under_roots('vae', vae)
        if p:
            paths['klein_vae'] = p
    _, lora_path = _consistency_lora()
    if lora_path and os.path.exists(lora_path):
        paths['klein_lora'] = lora_path
    enhancement = _lora_abs(ENHANCEMENT_LORA_NAME)
    if enhancement:
        paths['klein_enhancement_lora'] = enhancement
    return paths


def klein_invalid_assets():
    """Klein assets that ARE on disk under the resolved name but are NOT real,
    loadable weights — the state BETWEEN 'missing' and 'ready'. This is the case a
    plain existence check misses: a licence-gated model downloaded from a browser
    without the licence/token saves the HTML gate PAGE to <name>.safetensors, which
    passes Setup ('the file is there') and then dies at generate time on
    ``UNETLoader: Expecting value: line 1 column 1 (char 0)`` (json.loads choking on
    ``<!doctype html>``); a truncated / half-symlinked download instead renders
    silently distorted images.

    Returns ``[{asset, filename, verdict, blocking, reason}]`` — ``asset`` the
    setup_installer action name — and ``[]`` when every present asset is
    structurally valid. Header-only + cached (model_integrity), so it is cheap
    enough for the readiness probe. ``blocking`` True means the file can't load at
    all (html_or_text / truncated) → the actionable 'delete & re-download'; False is
    the advisory ``too_small`` (structurally intact but suspiciously small)."""
    from . import model_integrity
    out = []
    for asset, path in _klein_asset_paths().items():
        res = model_integrity.validate_model_file(path, min_bytes=KLEIN_MIN_BYTES.get(asset))
        if res['ok']:
            continue
        out.append({'asset': asset, 'filename': res['filename'],
                    'verdict': res['verdict'], 'blocking': res['blocking'],
                    'reason': res['reason']})
    return out


def klein_blocking_invalid(invalid=None) -> bool:
    """Is a REQUIRED Klein asset present but unloadable? Advisory `too_small` does
    not count — a small-but-loadable file is the user's, not ours."""
    return any(i['blocking'] and i['asset'] in KLEIN_REQUIRED
               for i in (klein_invalid_assets() if invalid is None else invalid))


def klein_engine_ready(comfy_ok, *, missing=None, invalid=None, unsupported_enums=None) -> bool:
    """THE Klein readiness verdict — ONE implementation, four conditions.

    It lived inline in capabilities (as `klein_ready`, behind `engines.klein` and
    `watermark_klein`) while watermark_klein.is_available() judged the same engine
    on presence alone. Two sincere answers to one question is how a truncated
    weight gets a green button on one screen and a refusal on the next — the exact
    shape of the Setup incident (54e5011). The looser copy also decided a SILENT
    fallback: the bank/dataset cleaner drops to LaMa when Klein is "unavailable",
    so an over-permissive verdict meant asking ComfyUI to load a file it cannot
    open instead of degrading cleanly.

    Callers that already hold the raw ingredients (capabilities recomputes them for
    its payload) pass them in; everyone else lets this fetch them. Each probe is
    cheap: disk listdir for the assets, a header read + cache for integrity, a
    cached /object_info for the widget values (which fails OPEN — an unreachable
    ComfyUI is already answered by `comfy_ok`)."""
    if not comfy_ok:
        return False
    enums = klein_unsupported_enums() if unsupported_enums is None else unsupported_enums
    if enums:
        return False
    gaps = klein_missing_assets() if missing is None else missing
    if any(a in gaps for a in KLEIN_REQUIRED):
        return False
    return not klein_blocking_invalid(invalid)


# --- Custom-node preflight -------------------------------------------------
# The class_types the shipped 'improve skin.json' historically pulled from custom
# packs, mapped to the pack that ships each + its GitHub page. Setup installs
# neither, and no doc mentioned them, so a fresh install hit a raw ComfyUI 400
# 'missing_node_type' on the first generation. The workflow no longer references
# these (strategy A), so this map is the safety net for a reverted/edited
# workflow — and the generic fallback (pack/url = None) covers any OTHER custom
# node a future change might introduce.
KLEIN_NODE_PACKS = {
    'TextBox1': ('RES4LYF', 'https://github.com/ClownsharkBatwing/RES4LYF'),
    'Any Switch (rgthree)': ('rgthree-comfy', 'https://github.com/rgthree/rgthree-comfy'),
}


def _workflow_class_types(workflow):
    return {n.get('class_type') for n in (workflow or {}).values()
            if isinstance(n, dict) and n.get('class_type')}


# Success-only TTL cache for the node preflight on the SHIPPED workflow:
# /object_info is a multi-MB payload, so don't re-fetch it for every tile
# regenerate click. Only an "all nodes present" verdict is cached (node packs
# don't uninstall mid-session); a miss or an unreachable probe is NEVER cached,
# so the "install the pack, restart ComfyUI, retry" flow re-probes immediately.
_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def klein_missing_nodes(workflow=None):
    """[{class_type, pack, url}] for every node class the Klein edit workflow needs
    that the target ComfyUI does NOT expose (i.e. absent from its /object_info
    keys). Loads the shipped 'improve skin.json' when no `workflow` is given.

    FAIL-OPEN: returns [] when /object_info can't be fetched (fetch returns None)
    — a transient probe failure must never block generation (mirrors the Test
    Studio node preflight in lora_test_studio.preflight_family)."""
    global _nodes_ok_until
    shipped = workflow is None
    if shipped:
        if time.time() < _nodes_ok_until:
            return []
        workflow = load_workflow_local(str(WORKFLOW_IMPROVE_SKIN_PATH)) or {}
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = []
    for ct in sorted(_workflow_class_types(workflow) - available):
        pack, url = KLEIN_NODE_PACKS.get(ct, (None, None))
        out.append({'class_type': ct, 'pack': pack, 'url': url})
    if shipped and not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


_enums_ok_until = 0.0


def klein_unsupported_enums(workflow=None):
    """[{node_id, class_type, input, value, pack, url}] for every widget value the
    Klein edit workflow pins that the target ComfyUI does NOT accept. Loads the
    shipped 'improve skin.json' when no `workflow` is given.

    Sibling of `klein_missing_nodes`, and the gap it leaves: that one compares
    class_types, so a graph built entirely from CORE nodes passes it — and then
    dies on a core KSampler because one of its VALUES (the `beta57` scheduler,
    which the RES4LYF pack registers into core) isn't there. Same probe, same
    /object_info payload, one level deeper into it.

    Same success-only TTL as the node verdict, and for the same reason: the
    capabilities probe calls this every 30 s, /object_info is a multi-MB payload,
    and its own cache is only 60 s — without this the app would re-download it
    every minute, forever, on a machine that is perfectly fine. Only an
    "everything supported" verdict is cached; a gap or an unreachable probe is
    never cached, so installing the missing pack and restarting ComfyUI clears the
    warning immediately instead of after a delay.

    FAIL-OPEN: [] when /object_info can't be fetched."""
    global _enums_ok_until
    from ..utils.comfyui import unsupported_enum_values
    shipped = workflow is None
    if shipped:
        if time.time() < _enums_ok_until:
            return []
        workflow = load_workflow_local(str(WORKFLOW_IMPROVE_SKIN_PATH)) or {}
    found = unsupported_enum_values(workflow)
    if shipped and not found:
        _enums_ok_until = time.time() + _NODES_OK_TTL_S
    return found


def format_missing_nodes_message(missing_nodes):
    """Human sentence for a Klein node-missing 409: each missing class_type with the
    pack that provides it + its GitHub link, then the fix instruction. Reused by
    the datasets route's existing Klein-missing error path."""
    bits = []
    for n in missing_nodes:
        ct, pack, url = n.get('class_type'), n.get('pack'), n.get('url')
        if pack and url:
            bits.append(f"{ct} (from {pack}: {url})")
        elif pack:
            bits.append(f"{ct} (from {pack})")
        else:
            bits.append(str(ct))
    return ("Your ComfyUI is missing custom node(s) this Klein workflow needs: "
            + '; '.join(bits) + ". Install via ComfyUI-Manager, then restart ComfyUI.")


def _bypass_node(workflow, node_id, passthrough_input):
    """Delete node_id and reconnect every consumer of its output slot 0 to the
    node's own `passthrough_input` upstream. Used to drop a LoRA loader whose file
    is absent (its consumers wire straight to its `model` input) so ComfyUI never
    fails validation on a missing LoRA."""
    node = workflow.get(node_id)
    if not node:
        return
    upstream = node.get('inputs', {}).get(passthrough_input)
    if upstream is None:
        return
    for other in workflow.values():
        for k, v in list(other.get('inputs', {}).items()):
            if isinstance(v, list) and len(v) == 2 and v[0] == node_id:
                other['inputs'][k] = upstream
    workflow.pop(node_id, None)


def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


def enqueue_klein_edit(user_id, source_filename, edit_prompt, klein_model=None,
                       extra_metadata=None, lora_strength=None, source_path=None,
                       extra_ref_paths=None, sampler_steps=None,
                       base_lora_strength=None, generation_loras=None,
                       output_megapixels=None):
    """Copy the source into ComfyUI input, configure the single Klein edit
    workflow, and enqueue it. Returns the app job_id. Raises ValueError on a
    missing source / unloadable workflow / missing required node, RuntimeError
    if ComfyUI isn't configured.
    `lora_strength` overrides `klein.consistency_strength` (clamped [0.0, 1.5]);
    0 disables the consistency LoRA entirely (it anchors composition, and even
    mid strengths can suppress big restagings).
    `generation_loras`: ordered [{file, strength}] of OPTIONAL user-pointed
    generation LoRAs (Idea by @waltm) to chain after the consistency LoRA —
    the FINAL list for THIS job, i.e. the rows of the preset the run picked,
    resolved from config via resolve_generation_lora_preset() (a request can
    only NAME a preset, never define files/order). None/[] = no preset (the
    default). Capped at MAX_GENERATION_LORAS.
    `source_path` overrides the default ComfyUI output dir/<source_filename> lookup
    so callers with per-dataset storage can pass the full path directly.
    `extra_ref_paths`: additional identity reference images (the dataset's extra
    refs) chained as native ReferenceLatent nodes — Klein consumes several
    references natively, and extra angles of the same face lock identity better
    than the single primary ref."""
    if source_path is None:
        out_dir = _comfy_output_dir()
        if out_dir is None:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(out_dir, source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f"source image not found: {source_filename}")
    workflow = load_workflow_local(str(WORKFLOW_IMPROVE_SKIN_PATH))
    if not workflow:
        raise ValueError("failed to load Klein edit workflow")
    for node in _REQUIRED_NODES:
        if node not in workflow:
            raise ValueError(f"workflow node {node} missing — improve skin.json has changed")

    # Resolve every loader node against what is ACTUALLY installed — the lifted
    # workflow hardcodes the developer's own ComfyUI filenames (see module
    # docstring), none of which match a fresh install. Block BEFORE copying the
    # source / enqueuing when a graph-critical asset is absent, so the caller can
    # auto-download it instead of firing a job every tile of which would fail.
    # A NAMED model is a promise, not a hint: when the caller (the dataset's stored
    # pick, or the workspace picker) says which UNET to run, a file that has left
    # the disk must stop the job by name instead of quietly resolving to the
    # canonical download — the result of that swap is indistinguishable from a
    # correct one. With no name, the historical auto-resolution is untouched.
    unet_ref = unet_for_job(klein_model)
    vae_ref = resolve_klein_vae()
    te_ref = resolve_klein_text_encoder()
    missing = klein_missing_assets()
    if any(a in missing for a in KLEIN_REQUIRED):
        raise KleinModelsMissing(missing)

    # The source image reaches ComfyUI over the FILESYSTEM, not the API — see
    # utils/comfy_fs.py. Staged through the guard so a folder that isn't shared
    # with ComfyUI's container/host says so (409) instead of raising a bare OSError
    # the routes can only turn into a detail-free 500 (reported by nofaceman).
    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    source_stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    # The input directory may belong to a remote/containerised ComfyUI. Stage a
    # fresh upright PNG there rather than copying camera EXIF/XMP/GPS bytes.
    staged_source = comfy_fs.stage_input_image(
        source_path, f"edit_source_{uid}_{source_stem}.png", comfy_input_dir)
    comfy_input = os.path.basename(staged_source)
    # Every name staged for THIS job, so its completion can delete them again.
    # Without this list each improved image left a full-resolution duplicate in
    # ComfyUI's input folder forever (measured: 3 896 orphans on one install).
    staged_inputs = [comfy_input]

    workflow["52"]["inputs"]["image"] = comfy_input
    # Prompt into the CLIPTextEncode widget directly (node 6). The old RES4LYF
    # TextBox1 (node 145) that used to carry it was dropped to de-depend the graph
    # from custom-node packs; node 6's `text` is a plain STRING input.
    workflow["6"]["inputs"]["text"] = edit_prompt
    workflow["77"]["inputs"]["seed"] = random.randint(0, 2 ** 64 - 1)
    if sampler_steps is not None:
        workflow["77"]["inputs"]["steps"] = max(1, int(sampler_steps))
    if base_lora_strength is not None and "139" in workflow:
        workflow["139"]["inputs"]["strength_model"] = float(base_lora_strength)
    # Output size: node 174 rescales the source to a total pixel budget before the
    # sampler, so it IS the resolution of the result. Hardcoded at 2 MP until now,
    # which made "Upscale" a fixed 2 MP pass whatever the source was worth.
    if output_megapixels is not None and "174" in workflow:
        workflow["174"]["inputs"]["megapixels"] = float(output_megapixels)
    # UNIQUE prefix per job: SaveImage numbers files from what's currently in
    # ComfyUI's output folder, and the app MOVES each result out right after
    # completion — with a shared prefix the counter kept re-issuing the same
    # name (live repro: 4 different seeds/prompts all saved as
    # local_DatasetFace_00002_.png), so every tile displayed the same file,
    # each new generation overwriting the previous one in the dataset dir.
    workflow["9"]["inputs"]["filename_prefix"] = f"{user_id}_DatasetFace_{uid}"
    workflow["114"]["inputs"]["unet_name"] = unet_ref
    workflow["10"]["inputs"]["vae_name"] = vae_ref
    workflow["90"]["inputs"]["clip_name"] = te_ref

    # Multi-reference (native): chain each extra identity ref into the POSITIVE
    # conditioning — Klein reads several ReferenceLatent nodes natively. The
    # primary ref stays first and strongest (2 MP, node 92); extras add identity
    # signal from other angles at 1 MP. cfg=1 → the negative chain (110) is
    # ignored by the sampler, so it is left untouched. A missing extra file is
    # skipped silently (same tolerance as the Nano Banana multi-ref path).
    prev = "92"
    for i, ref_path in enumerate(extra_ref_paths or [], start=1):
        if not ref_path or not os.path.exists(ref_path):
            logger.warning(f"klein multi-ref: extra ref missing on disk: {ref_path}")
            continue
        ref_stem = os.path.splitext(os.path.basename(str(ref_path)))[0] or f'ref{i}'
        staged_ref = comfy_fs.stage_input_image(
            ref_path, f"edit_ref{i}_{uid}_{ref_stem}.png", comfy_input_dir)
        ref_input = os.path.basename(staged_ref)
        staged_inputs.append(ref_input)
        load_id, scale_id = f"ds_ref{i}_load", f"ds_ref{i}_scale"
        enc_id, lat_id = f"ds_ref{i}_encode", f"ds_ref{i}_latent"
        workflow[load_id] = {"class_type": "LoadImage", "inputs": {"image": ref_input},
                             "_meta": {"title": f"Extra identity ref {i}"}}
        workflow[scale_id] = {"class_type": "ImageScaleToTotalPixels",
                              "inputs": {"upscale_method": "lanczos", "megapixels": 1,
                                         "resolution_steps": 1, "image": [load_id, 0]},
                              "_meta": {"title": f"Scale extra ref {i}"}}
        workflow[enc_id] = {"class_type": "VAEEncode",
                            "inputs": {"pixels": [scale_id, 0], "vae": ["10", 0]},
                            "_meta": {"title": f"Encode extra ref {i}"}}
        workflow[lat_id] = {"class_type": "ReferenceLatent",
                            "inputs": {"conditioning": [prev, 0], "latent": [enc_id, 0]},
                            "_meta": {"title": f"Reference latent extra {i}"}}
        prev = lat_id
    if prev != "92":
        workflow["77"]["inputs"]["positive"] = [prev, 0]

    # Inject the consistency LoRA between the UNET (114) and the base LoRA node
    # (139) → chain 114 -> consistency -> 139. NOTE: this LoRA anchors STRUCTURE
    # (composition/background) — its own guide recommends ~0.5 and warns 0.8-1.0
    # can stop edits from applying. Strength 0 (slider fully left) skips the
    # node entirely so the base edit behaviour is reachable. Also skipped
    # (degraded but functional) if the LoRA file or node 139 is missing.
    consistency_lora, lora_path = _consistency_lora()
    strength = cfg.get('klein.consistency_strength')
    if lora_strength is not None:
        strength = max(0.0, min(1.5, float(lora_strength)))
    # model_ref = the model output feeding node 139: starts at the UNET (114) and
    # advances as LoRA loaders are chained onto it, so every injected node hangs
    # off the previous one and 139 is rewired once, to the LAST link.
    model_ref = ["114", 0]
    # Injected loader nodes get NUMERIC ids (like every hand-authored ComfyUI
    # node), allocated just above the workflow's existing numeric ids. The
    # executed graph is identical either way — ComfyUI's server keys the prompt
    # dict by string — but its "rebuild the graph from a dropped image"
    # reconstructs the links reliably only for numeric ids: the old non-numeric
    # 'ds_consistency_lora' / 'ds_gen_lora_1' keys made the dropped-in canvas show
    # the chain collapsed to the LAST LoRA past the consistency node, even though
    # every LoRA WAS applied at generation (bug report by @waltm). The counter
    # advances only when a node is actually added, so a degraded row burns no id.
    _next_node_id = max((int(k) for k in workflow if k.isdigit()), default=0)
    if "139" not in workflow:
        logger.warning("workflow node 139 missing — consistency LoRA injection skipped")
    elif not lora_path or not os.path.exists(lora_path):
        # A strength the CALLER deliberately passed must never be silently ignored:
        # the job would run, look wrong, and give no clue why. Reported as a missing
        # ASSET so the caller's existing auto-download handles it, not a bare failure.
        # Gated on an EXPLICIT lora_strength: generation leaves it None and takes
        # klein.consistency_strength from config, where the documented contract is to
        # degrade and fetch the LoRA in the background — erroring there would stop
        # people generating at all over an optional quality LoRA.
        if lora_strength is not None and float(lora_strength) > 0:
            raise KleinModelsMissing(['klein_lora'])
        logger.warning(f"consistency LoRA not found at {lora_path} — injection skipped")
    elif not strength or float(strength) <= 0:
        logger.info("consistency LoRA strength 0 — injection skipped (LoRA off)")
    else:
        _next_node_id += 1
        cons_id = str(_next_node_id)
        workflow[cons_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": consistency_lora,
                       "strength_model": strength, "model": model_ref},
            "_meta": {"title": "Dataset consistency LoRA"},
        }
        model_ref = [cons_id, 0]

    # Optional generation LoRAs (Idea by @waltm — Discord feature request):
    # the rows of the PRESET the run picked, chained AFTER the consistency
    # LoRA in preset order, i.e. 114 -> consistency -> gen_lora_1 -> ... ->
    # gen_lora_N -> 139. The filenames come from the config presets ONLY
    # (loras-relative, pointed by the user — never hardcoded, cf. the
    # base-LoRA note below), and degradation is PER ROW, mirroring the
    # consistency block exactly: missing node 139 / missing file /
    # strength <= 0 -> skip that row with a log line (the rest of the chain
    # still links up), never a doomed ComfyUI validation error.
    #
    # A row naming the SAME file as `consistency_lora` above is an extra guard,
    # not just a redundant one: it would chain the identical LoRA a second time,
    # summing both strengths into one delta well past what the file was trained
    # for — measured as visible macro-blocking on the Krea sibling of this
    # feature (a preset row duplicating krea.identity_lora), reported by waltm on
    # Discord. Path comparison is normcase+normpath so a '/' vs '\\' or case
    # difference can't dodge the guard.
    consistency_key = (os.path.normcase(os.path.normpath(consistency_lora))
                       if consistency_lora else None)
    for i, entry in enumerate((generation_loras or [])[:MAX_GENERATION_LORAS], start=1):
        slot_lora = (entry.get('file') or '').replace('/', os.sep)
        slot_strength = entry.get('strength')
        slot_strength = max(0.0, min(1.5, float(slot_strength))) \
            if isinstance(slot_strength, (int, float)) else 0.0
        slot_path = _lora_abs(slot_lora)
        if "139" not in workflow:
            logger.warning("workflow node 139 missing — generation LoRA %r skipped", slot_lora)
        elif not slot_lora or not slot_path:
            logger.warning("generation LoRA %r not found under any loras root — skipped", slot_lora)
        elif slot_strength <= 0:
            logger.info("generation LoRA %r strength 0 — skipped (row off)", slot_lora)
        elif consistency_key and os.path.normcase(os.path.normpath(slot_lora)) == consistency_key:
            logger.warning("generation LoRA %r is the consistency LoRA — skipped "
                           "(already applied at klein.consistency_strength, a second "
                           "copy would double-stack it)", slot_lora)
        else:
            _next_node_id += 1
            node_id = str(_next_node_id)
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": slot_lora,
                           "strength_model": slot_strength, "model": model_ref},
                "_meta": {"title": f"Generation LoRA {i}: {os.path.basename(slot_lora)}"},
            }
            model_ref = [node_id, 0]

    if "139" in workflow and model_ref != ["114", 0]:
        workflow["139"]["inputs"]["model"] = model_ref

    # The base style LoRA in node 139 (klein\realistic.safetensors) belongs to the
    # source app's ComfyUI and is NOT part of the Klein install — bypass it when
    # its file is absent so ComfyUI doesn't fail validation on a missing LoRA. The
    # consistency LoRA injected above (if any) stays in the chain.
    base_lora = (workflow.get("139", {}).get("inputs", {}).get("lora_name") or '').replace('/', os.sep)
    base_lora_path = _lora_abs(base_lora)   # base + extra_model_paths loras roots
    if "139" in workflow and not base_lora_path:
        # Same rule as the consistency LoRA: bypassing is fine at strength 0 (the node
        # would contribute nothing anyway), but silently dropping it while the user
        # asked for a real strength is what made this setting look broken — it moved
        # no pixel and said nothing. Surface it as a missing asset so it gets fetched.
        if base_lora_strength is not None and float(base_lora_strength) > 0:
            raise KleinModelsMissing(['klein_enhancement_lora'])
        logger.info("base LoRA %r absent — bypassing node 139", base_lora)
        _bypass_node(workflow, "139", "model")

    job_id = str(uuid.uuid4())
    meta = {"model_name": "klein_edit_dataset"}
    if extra_metadata:
        meta.update(extra_metadata)
    meta["staged_inputs"] = staged_inputs
    queue_manager.add_job(job_type="image", user_id=str(user_id), workflow_data=workflow,
                          prompt=edit_prompt, job_id=job_id, metadata=meta)
    return job_id
