# backend/app/utils/comfyui.py
"""ComfyUI communication, workflow helpers, and model/LoRA discovery.

Lifted from the parent project's app/utils/comfyui.py (1878 lines) for LoRA
Dataset Studio, config-driven and slimmed:

  - The module-level COMFYUI_LORA_DIR / COMFYUI_OUTPUT_DIR / COMFYUI_API_ADDRESS
    constants become live accessors (`_lora_dir()`, `_out_dir()`, `api_address()`)
    that re-read `app.config` on every call, so editing config.json takes effect
    without a restart. Every lister degrades to `[]` (never raises) when ComfyUI
    isn't configured yet.
  - hidden_models/hidden_loras visibility filtering dropped (single-user app —
    nothing to hide).
  - Video/other-app listers dropped: `get_subtle_loras`, `get_klein_loras`
    (unused by klein_edit_helper, the only prospective caller — it only needs
    `load_workflow_local` + `get_flux2_klein_models`), `get_ltx_camera_loras`,
    `get_wan_video_loras`, `get_biglove_models` (duplicate of
    `get_checkpoint_models`, folded away), `get_krea_style_loras`.
  - Also dropped (no caller in this app): `configure_http_notify_node` /
    `check_comfyui_dependencies` (the video webhook-notify flow this app's
    polling-based job_queue doesn't use), `get_comfyui_queue_status`,
    `invalidate_model_caches`, `get_model_folder_paths`, `unload_ollama_model`.
"""
from __future__ import annotations

import errno
import glob
import logging
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode, urljoin

import requests
from flask import current_app

from .. import config as cfg
from . import comfy_names
from .comfy_names import local_model_path


class ComfyVramFreeVerdict(Enum):
    """The only outcomes a vision request is allowed to act on."""

    FREED = 'freed'
    COMFYUI_OFFLINE = 'comfyui_offline'
    UNKNOWN = 'unknown'

    @property
    def permits_ollama(self):
        return self in (self.FREED, self.COMFYUI_OFFLINE)


class ComfyHistoryHealth(Enum):
    """Whether one /history response can safely drive the queue."""

    READY = 'ready'
    NOT_READY = 'not_ready'
    UNHEALTHY = 'unhealthy'


@dataclass(frozen=True)
class ComfyHistoryProbe:
    health: ComfyHistoryHealth
    history: dict | None = None
    detail: str | None = None


class ComfyPromptState(Enum):
    DELETED = 'deleted'
    PENDING = 'pending'
    RUNNING = 'running'
    ABSENT = 'absent'
    UNKNOWN = 'unknown'


def _exception_chain(exc):
    """Walk nested urllib3/requests transport errors without trusting text."""
    seen, pending = set(), [exc]
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for name in ('__cause__', '__context__', 'reason'):
            nested = getattr(current, name, None)
            if isinstance(nested, BaseException):
                pending.append(nested)
        pending.extend(arg for arg in getattr(current, 'args', ())
                       if isinstance(arg, BaseException))


def _is_explicit_connection_refused(exc):
    refused = {errno.ECONNREFUSED, 10061}  # POSIX / Winsock WSAECONNREFUSED
    return any(
        isinstance(item, ConnectionRefusedError)
        or (isinstance(item, OSError) and item.errno in refused)
        for item in _exception_chain(exc)
    )


logger = logging.getLogger(__name__)


# --- Live config accessors (replace SRC's module-level COMFYUI_* constants) --

def _lora_dir() -> str | None:
    p = cfg.comfyui_dir('loras')
    return str(p) if p else None


def _out_dir() -> str | None:
    p = cfg.comfyui_dir('output')
    return str(p) if p else None


def api_address() -> str:
    """COMFYUI_API_ADDRESS-equivalent accessor. Always resolves (config.py's
    DEFAULTS ship 'http://127.0.0.1:8188') — unlike the directory accessors
    above, there's no "unconfigured" state to guard against here."""
    return cfg.get('comfyui.api_url')




# --- Per-model optimal sampler/scheduler parameters (SDXL dropdown only) ---
#
# Scope: the SDXL checkpoints listed in the /generate dropdown
# (`checkpoints/Biglove/`). Z-Image, Flux 2 Klein and Qwen models are
# explicitly out of scope — they each have their own dedicated workflow
# with hand-tuned samplers (see ZTurbo / Z-Mode / Improve Skin / Klein-KV
# routes). Mixing those into a generic map would cause workflows tuned for
# one family to silently swap to incompatible params.
#
# Within the SDXL dropdown, two sub-families coexist and need *opposite*
# settings: full SDXL (20-30 steps, CFG 3-5) vs DMD-distilled (6-9 steps,
# CFG ~1). The HQ workflow defaults to DMD settings, so picking a full
# SDXL model previously produced under-cooked images.
#
# Sources:
#  - Big Love photo5 (full SDXL):       civitai.com/models/897413
#  - Lustify GGWP V7 (full SDXL NSFW):  civitai.com/models/573152
#  - MoP DMD v10 (DMD-distilled):       civitai.com/models/1854124
MODEL_OPTIMAL_PARAMS = {
    # -- Full SDXL - original workflow defaults + DMD2 LoRA activated ------
    # bigLove_photo5 and Lustify GGWP v7 are full SDXL (Civitai specs call
    # for ~30 steps + CFG 3-5 + DPM++) but produce usable output with the
    # original workflow's LCM + ddim_uniform + CFG 1 at 8 steps. Operator
    # validated this behavior empirically. We keep those exact defaults
    # (NOT karras, which is a different sigma schedule) and additionally
    # enable the DMD2 4-step LoRA to tighten convergence — the LoRA is
    # specifically trained for 4-step SDXL acceleration, so layering it
    # on a non-distilled checkpoint is the canonical use case.
    "bigLove_photo5.safetensors": {
        "sampler_name": "lcm",
        "scheduler": "ddim_uniform",
        "steps": 8,
        "cfg": 1.0,
        "dmd2_lora_strength": 1.0,
    },
    "lustifySDXLNSFW_ggwpV7.safetensors": {
        "sampler_name": "lcm",
        "scheduler": "ddim_uniform",
        "steps": 8,
        "cfg": 1.0,
        "dmd2_lora_strength": 1.0,
    },

    # -- SDXL DMD-distilled - CFG ~= 1, 6-9 steps, DMD2 LoRA ON ------------
    "dmdmopPro_v10.safetensors": {
        "sampler_name": "lcm",
        "scheduler": "karras",
        "steps": 8,
        "cfg": 0.8,
        "dmd2_lora_strength": 1.0,
    },
    # mopMix tends to look flat with default karras+CFG 1.0. The MoP DMD
    # family officially tolerates CFG up to 1.3; bumping to 1.2 with
    # sgm_uniform recovers contrast without breaking distillation.
    "mopMix_asapnsfw.safetensors": {
        "sampler_name": "lcm",
        "scheduler": "sgm_uniform",
        "steps": 8,
        "cfg": 1.2,
        "dmd2_lora_strength": 1.0,
    },
    "mopMixtureOfPervertsDMD_v40.safetensors": {
        "sampler_name": "lcm",
        "scheduler": "karras",
        "steps": 8,
        "cfg": 1.0,
        "dmd2_lora_strength": 1.0,
    },
}

# Substring fallback for unknown SDXL filenames. Because the workflow is
# fixed at 8 steps, *every* SDXL model — distilled or not — must run in
# DMD-style inference to converge. The single fallback below mirrors that
# constraint; if you ever unlock step counts and want true full-SDXL
# behavior for non-DMD names, add a second entry with `dmd2_lora_strength: 0`
# + DPM++ samplers + higher CFG.
_FAMILY_DEFAULTS = (
    ("sdxl", {"sampler_name": "lcm", "scheduler": "karras", "steps": 8, "cfg": 1.0, "dmd2_lora_strength": 1.0}),
)


# Path to the JSON file that holds admin-edited overrides. Lives at
# backend/workflows/ so it ships with the app (SRC kept it at repo root next
# to config.json; that file didn't exist in SRC either — it's optional, and
# an absent/empty file means "code defaults only").
_SAMPLER_PARAMS_JSON_PATH = str(cfg.BACKEND_DIR / "workflows" / "sampler_params.json")


def _load_sampler_params_overrides() -> dict:
    """Read the admin override file. Returns {} if absent or invalid.

    The overrides MERGE with `MODEL_OPTIMAL_PARAMS` at lookup time — admin
    edits take precedence per-key, but a missing key falls back to the
    in-code default. This lets the admin tweak `cfg` for one model without
    having to re-specify the entire row, and it means an empty file (or
    deleted file) safely reverts to code behavior.
    """
    if not os.path.exists(_SAMPLER_PARAMS_JSON_PATH):
        return {}
    try:
        import json
        with open(_SAMPLER_PARAMS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError) as e:
        logger.warning(f"sampler_params.json invalid or unreadable: {e}; using code defaults")
        return {}


def get_effective_sampler_params() -> dict:
    """Return the merged view of code defaults + admin overrides.

    For each model present in either source, returns a single dict with
    overrides applied on top of the code default. Used both by the admin
    endpoint (to seed the editor) and by `_resolve_optimal_params`.
    """
    overrides = _load_sampler_params_overrides()
    merged = {}
    for name in set(MODEL_OPTIMAL_PARAMS) | set(overrides):
        base = dict(MODEL_OPTIMAL_PARAMS.get(name, {}))
        base.update(overrides.get(name, {}))
        merged[name] = base
    return merged


def _resolve_optimal_params(model_filename: str | None) -> dict | None:
    """Return the optimal sampler params for a model filename, or None.

    Lookup order:
      1. Exact filename match in `get_effective_sampler_params()`
         (code defaults + admin overrides from `sampler_params.json`)
      2. Substring match against `_FAMILY_DEFAULTS` patterns
      3. None (caller keeps workflow defaults)
    """
    if not model_filename:
        return None
    bare = os.path.basename(model_filename)
    effective = get_effective_sampler_params()
    if bare in effective:
        return effective[bare]
    lowered = bare.lower()
    for needle, defaults in _FAMILY_DEFAULTS:
        if needle in lowered:
            return defaults
    return None


# Sampler-related ComfyUI node types whose inputs we override. KSampler
# carries all four params; KSamplerSelect only carries sampler_name;
# BasicScheduler carries scheduler + steps. SamplerCustomAdvanced reads
# from its wired neighbours so leaving it alone is correct.
_SAMPLER_NODE_TYPES = {"KSampler", "KSamplerAdvanced"}
_SAMPLER_SELECT_TYPES = {"KSamplerSelect"}
_SCHEDULER_NODE_TYPES = {"BasicScheduler"}


# Fields the helper actually writes. `steps` is intentionally excluded for
# now: the table documents the recommended step count, but operator wants
# to validate sampler/scheduler/cfg changes first before touching step
# counts (which affect generation time and have downstream impact on the
# DetailDaemon second pass). Add `"steps"` here to enable.
_OVERRIDABLE_FIELDS = ("sampler_name", "scheduler", "cfg")


def apply_optimal_sampler_params(workflow: dict, model_filename: str | None) -> dict:
    """Override sampler/scheduler/cfg on a workflow to match the model.

    Mutates and returns `workflow`. No-op when no entry resolves for the
    given model (workflow defaults preserved). Logs every node it touches
    so post-mortem of "why did my settings change?" is straightforward.

    Only updates a field if the node already has it (avoids accidentally
    adding `cfg` to a KSamplerSelect, which would be a schema violation)
    AND only updates fields listed in `_OVERRIDABLE_FIELDS` (currently
    excludes `steps` — see comment on that constant).
    """
    params = _resolve_optimal_params(model_filename)
    if not params:
        return workflow

    log = current_app.logger if current_app else logger
    log.info(f"[apply_optimal_sampler_params] Resolved {model_filename!r} -> {params} (applied fields: {_OVERRIDABLE_FIELDS})")

    dmd2_strength = params.get("dmd2_lora_strength")

    for node_id, node in workflow.items():
        ct = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})

        if ct in _SAMPLER_NODE_TYPES:
            for key in _OVERRIDABLE_FIELDS:
                if key in inputs and key in params:
                    inputs[key] = params[key]
            log.info(f"  node {node_id} ({ct}): sampler={inputs.get('sampler_name')}, scheduler={inputs.get('scheduler')}, cfg={inputs.get('cfg')} (steps={inputs.get('steps')} left as-is)")
        elif ct in _SAMPLER_SELECT_TYPES:
            if "sampler_name" in inputs and "sampler_name" in _OVERRIDABLE_FIELDS:
                inputs["sampler_name"] = params["sampler_name"]
                log.info(f"  node {node_id} ({ct}): sampler={inputs['sampler_name']}")
        elif ct in _SCHEDULER_NODE_TYPES:
            if "scheduler" in inputs and "scheduler" in _OVERRIDABLE_FIELDS:
                inputs["scheduler"] = params["scheduler"]
            log.info(f"  node {node_id} ({ct}): scheduler={inputs.get('scheduler')} (steps={inputs.get('steps')} left as-is)")
        elif ct == "LoraLoader" and dmd2_strength is not None:
            # The HQ workflow wires a DMD2 4-step LoRA (node 10 by default,
            # but we match by lora_name in case the node id changes). It
            # must be ACTIVE for DMD-distilled checkpoints (the workflow
            # expects DMD-style inference at 8 steps + CFG ~= 1) and OFF for
            # full SDXL checkpoints (where the LoRA would conflict with
            # higher CFG / non-LCM samplers).
            lora_name = (inputs.get("lora_name") or "").lower()
            if "dmd2" in lora_name:
                inputs["strength_model"] = dmd2_strength
                inputs["strength_clip"] = dmd2_strength
                log.info(f"  node {node_id} ({ct}, DMD2): strength={dmd2_strength}")

    return workflow


# --- Workflow Loading ---

# Workflow templates, cached by (path, mtime_ns, size) — a grid re-reads the SAME
# template once per cell (a 50-cell Studio run = 50 disk reads + 50 INFO log lines).
# We cache the raw TEXT, not the parsed dict: every caller MUTATES the graph it gets
# back, so each call must still receive its own fresh object (json.loads = the copy).
# The mtime/size key means editing a workflow file on disk is picked up immediately.
_workflow_text_cache = {}


def load_workflow_local(file_path):
    """Charge un fichier JSON de workflow ComfyUI et retourne les données parsées, ou None en cas d'erreur."""
    import json
    try:
        st = os.stat(file_path)
        key = os.path.abspath(file_path)
        stamp = (st.st_mtime_ns, st.st_size)
        cached = _workflow_text_cache.get(key)
        if cached and cached[0] == stamp:
            text = cached[1]
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            _workflow_text_cache[key] = (stamp, text)
            current_app.logger.info(f"Successfully loaded workflow from {file_path}")
        return json.loads(text)
    except FileNotFoundError:
        current_app.logger.error(f"ERROR: Workflow JSON file not found at {file_path}")
        return None
    except ValueError as e:
        current_app.logger.error(f"ERROR: Failed to decode workflow JSON from {file_path}: {e}")
        return None


# --- ComfyUI API Communication ---

def _ensure_comfyui_before_generation():
    """Lazy import of the (Task 14) comfyui_service.ensure_comfyui_before_generation.

    Returns None if the service module isn't available yet — the caller
    should treat that as "skip the restart attempt" and fall through to the
    normal error path, exactly like a caught exception during the check
    (never abort generation just because the optional restart-helper is
    missing). Otherwise returns the (success, message) tuple it produced.
    """
    try:
        from ..services.comfyui_service import ensure_comfyui_before_generation
    except ImportError as e:
        logger.warning(f"Could not import ComfyUI service: {e}")
        return None
    try:
        return ensure_comfyui_before_generation()
    except Exception as e:
        logger.error(f"Error checking ComfyUI service: {e}")
        return None


def queue_prompt_to_comfyui(prompt_workflow, client_id, worker_url=None):
    """Envoie un workflow à ComfyUI pour exécution.

    Args:
        prompt_workflow: Workflow JSON ComfyUI (format API).
        client_id: Identifiant du client (user_id).
        worker_url: URL optionnelle du worker distant. Si None, utilise api_address() (local).
    """
    if not prompt_workflow:
        return None, "Workflow data is missing"

    # URL cible : worker distant ou local
    local_api = api_address()
    api_addr = worker_url or local_api
    is_local = not worker_url or api_addr.rstrip('/') == local_api.rstrip('/')

    # Vérifier que ComfyUI est démarré (seulement pour le worker local)
    if is_local:
        result = _ensure_comfyui_before_generation()
        if result is not None:
            success, message = result
            if not success:
                logger.error(f"ComfyUI is not available: {message}")
                return None, f"ComfyUI service unavailable: {message}"
            logger.info(f"ComfyUI service check: {message}")

    # Check for Ollama usage in the workflow (local only)
    if not worker_url:
        try:
            uses_ollama = any(
                isinstance(node_data, dict) and "ollama" in node_data.get("class_type", "").lower()
                for node_data in prompt_workflow.values()
            )
            if uses_ollama:
                logger.info("Ollama node detected in workflow. Ensuring Ollama is running...")
                if not ensure_ollama_running():
                    logger.warning("Failed to ensure Ollama is running. Workflow might fail.")
        except Exception as e:
            logger.error(f"Error checking for Ollama dependency: {e}")

    # Model-file widgets: respell them the way THIS ComfyUI spells them, because
    # its validator does an exact string match and nothing else (execution.py:
    # `val not in combo_options`). The separator is a property of ComfyUI's HOST,
    # never of ours: measured, a Windows ComfyUI publishes
    # 'Krea\krea2_turbo_fp8.safetensors' and a Linux one 'Krea/krea2…' — so both
    # a hardcoded backslash (which is what shipped, and made Linux generate
    # NOTHING: GitHub #21, 1Tomber) and a hardcoded forward slash (which would
    # take out every Windows install) are wrong. `/object_info` already tells us,
    # and it is the same cached payload the two preflights below read, so this
    # costs no extra request.
    #
    # Local only for the LIST — /object_info is fetched from api_address(), so a
    # remote worker must not be judged against the local install's models. It
    # still gets the os.sep fallback, which is what it had before.
    try:
        listed = fetch_object_info_model_files() if is_local else None
        prompt_workflow, respelled = comfy_names.canonical_model_widgets(
            prompt_workflow, listed)
        if respelled:
            logger.info('Model widget names respelled for the target ComfyUI: %d',
                        respelled)
    except Exception as e:      # a naming helper must never be what blocks a job
        logger.warning(f"Model-name canonicalisation skipped: {e}")

    # Capability preflight on the graph itself, LOCAL only. Our workflows pin widget
    # values (a scheduler, a sampler, a dtype) that a given install only accepts if
    # it is recent enough or loaded the node pack that registers them. Left alone,
    # ComfyUI answers a bare 400 whose explanation lives in ComfyUI's console, not
    # in the app — the report that prompted this (IndependentProcess0, Reddit) is
    # exactly that. One CACHED /object_info, so no extra request per generation, and
    # fail-OPEN when the probe is unreachable. Local only: /object_info is fetched
    # from api_address(), so a remote worker's graph must not be judged against the
    # local install's capabilities.
    if is_local:
        try:
            unsupported = unsupported_enum_values(prompt_workflow)
        except Exception as e:                    # a broken probe must never block
            logger.warning(f"Enum capability preflight skipped: {e}")
            unsupported = []
        if unsupported:
            message = format_unsupported_enums_message(unsupported)
            logger.error(f"Workflow refused before queuing — {message}")
            # Same deterministic tag as a ComfyUI 400: retrying or restarting
            # changes nothing, so the queue must fail the job now, not requalify it
            # as an outage.
            return None, f"WORKFLOW_INVALIDE (ComfyUI capability): {message}"

        # Same preflight, one field over: a model FILE name ComfyUI does not list
        # fails with the identical 400. Reported by naniii2352 (Discord) — a .gguf
        # no folder could make work, on an install whose API address and models
        # override pointed at two different ComfyUI trees. Rides the same cached
        # /object_info, so still zero extra requests per generation.
        try:
            unavailable = unavailable_model_files(prompt_workflow)
        except Exception as e:                    # a broken probe must never block
            logger.warning(f"Model-file preflight skipped: {e}")
            unavailable = []
        # A DETERMINISTIC kill may not rest on a cached observation. The list above
        # can be up to _OBJECT_INFO_TTL old and the deploy path does not invalidate
        # it, so a model deployed seconds ago used to die here for good. Re-ask once
        # — bounded and gguf-exempt, see confirm_unavailable_model_files.
        rechecked = False
        if unavailable:
            try:
                unavailable, prompt_workflow, rechecked = (
                    confirm_unavailable_model_files(prompt_workflow, unavailable))
            except Exception as e:                # a broken probe must never block
                logger.warning(f"Model-file re-check skipped: {e}")
        if unavailable:
            message = format_unavailable_models_message(unavailable,
                                                        rechecked=rechecked)
            logger.error(f"Workflow refused before queuing — {message}")
            return None, f"WORKFLOW_INVALIDE (ComfyUI capability): {message}"

    try:
        payload = {"prompt": prompt_workflow, "client_id": client_id}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            urljoin(api_addr, "/prompt"), json=payload, headers=headers, timeout=10,
            allow_redirects=False)
        response.raise_for_status()
        status = getattr(response, 'status_code', None)
        if type(status) is not int or not 200 <= status < 300:
            return None, f'ComfyUI /prompt returned unsafe HTTP status {status!r}'
        return response.json(), None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error queuing prompt to {api_addr}: {e}")

        # ComfyUI place le détail de validation (node_errors) dans le CORPS de la
        # réponse 400 — sans le logger, on ne voit jamais POURQUOI le workflow est
        # rejeté (modèle introuvable, nœud custom non chargé, input invalide...).
        err_body = ''
        _resp = getattr(e, 'response', None)
        if _resp is not None:
            try:
                err_body = (_resp.text or '')[:2000]
            except Exception:
                err_body = ''
            if err_body:
                logger.error(f"ComfyUI /prompt {getattr(_resp, 'status_code', '?')} body: {err_body}")

        # 400 = REJET DE VALIDATION (modèle absent du disque, node inconnu, input
        # invalide…) : déterministe — retenter ou redémarrer ComfyUI n'y changera
        # RIEN. Tag distinct pour que la queue échoue le job immédiatement au lieu
        # de le requalifier en « panne ».
        if getattr(_resp, 'status_code', None) == 400:
            return None, f"WORKFLOW_INVALIDE (validation ComfyUI 400): {err_body[:600]}"

        # Never blindly retry a POST here. A transport failure can occur after
        # ComfyUI accepted the first request; retrying would create an untracked
        # second prompt. The queue stores this outcome as a client-id recovery
        # barrier; recovery requires an externally verified ComfyUI restart.

        detail = f": {e}" + (f" | ComfyUI: {err_body}" if err_body else '')
        return None, f"Failed to connect or communicate with ComfyUI API ({api_addr}){detail}"
    except Exception as e:
        logger.error(f"Unexpected error queuing prompt to {api_addr}: {e}")
        return None, f"An unexpected error occurred: {e}"


def get_comfyui_history_probe(prompt_id, worker_url=None) -> ComfyHistoryProbe:
    """Classify /history without confusing a worker outage with no output yet."""
    api_addr = worker_url or api_address()
    try:
        response = requests.get(
            urljoin(api_addr, f'/history/{prompt_id}'), timeout=5, allow_redirects=False)
        status = getattr(response, 'status_code', None)
        if type(status) is not int:
            return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                     detail='malformed response')
        if status == 404:
            return ComfyHistoryProbe(ComfyHistoryHealth.NOT_READY)
        if not 200 <= status < 300:
            return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                     detail=f'HTTP {status}')
        try:
            history = response.json()
        except Exception:
            return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                     detail='malformed JSON')
        if not isinstance(history, dict):
            return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                     detail='malformed history')
        if not history:
            return ComfyHistoryProbe(ComfyHistoryHealth.NOT_READY)
        # The exact requested key is the ownership proof. A direct entry could
        # be a proxy/cache response for another prompt, so accepting it would
        # falsely complete this queue row while its real GPU work still runs.
        entry = history.get(prompt_id)
        if entry is None:
            return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                     detail='history does not contain requested prompt')
        if not isinstance(entry, dict) or not (
                'outputs' in entry or 'status' in entry):
            return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                     detail='malformed history entry')
        if ('outputs' in entry and not isinstance(entry.get('outputs'), dict)) or \
                ('status' in entry and not isinstance(entry.get('status'), dict)):
            return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                     detail='malformed history entry fields')
        return ComfyHistoryProbe(ComfyHistoryHealth.READY, history=history)
    except requests.RequestException as exc:
        response = getattr(exc, 'response', None)
        if type(getattr(response, 'status_code', None)) is int and response.status_code == 404:
            return ComfyHistoryProbe(ComfyHistoryHealth.NOT_READY)
        return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                 detail=str(exc)[:200])
    except Exception as exc:
        return ComfyHistoryProbe(ComfyHistoryHealth.UNHEALTHY,
                                 detail=str(exc)[:200])


def _queue_entry_identity(entry):
    """Return ``(prompt_id, client_id)`` from a ComfyUI ``/queue`` entry.

    Current ComfyUI versions expose queue items as
    ``[number, prompt_id, workflow, extra_data, outputs]`` while a few forks
    return dictionaries. Supporting both shapes keeps cancellation best-effort
    without coupling LDS to one ComfyUI build.
    """
    if isinstance(entry, (list, tuple)):
        prompt_id = entry[1] if len(entry) > 1 else None
        extra = entry[3] if len(entry) > 3 and isinstance(entry[3], dict) else {}
        return prompt_id, extra.get('client_id')
    if isinstance(entry, dict):
        extra = entry.get('extra_data') if isinstance(entry.get('extra_data'), dict) else {}
        return (entry.get('prompt_id') or entry.get('id'),
                entry.get('client_id') or extra.get('client_id'))
    return None, None


def comfyui_prompt_is_absent(prompt_id, worker_url=None):
    """Return True only after a healthy queue response proves this id absent.

    None means unknown: malformed /queue, transport trouble, or an unparseable
    entry must never grant permission to resume/requeue an old GPU prompt.
    """
    if not prompt_id:
        return None
    try:
        api_addr = worker_url or api_address()
        response = requests.get(
            urljoin(api_addr, '/queue'), timeout=3, allow_redirects=False)
        status = getattr(response, 'status_code', None)
        if type(status) is not int or not 200 <= status < 300:
            return None
        queue = response.json()
        if not isinstance(queue, dict):
            return None
        pending = queue.get('queue_pending')
        running = queue.get('queue_running')
        if not isinstance(pending, (list, tuple)) or not isinstance(running, (list, tuple)):
            return None
        identities = [_queue_entry_identity(entry) for entry in [*pending, *running]]
        if any(queued_prompt_id is None for queued_prompt_id, _client_id in identities):
            return None
        return not any(str(queued_prompt_id) == str(prompt_id)
                       for queued_prompt_id, _client_id in identities)
    except requests.RequestException as exc:
        logger.warning('Could not inspect ComfyUI prompt %s: %s', prompt_id, exc)
    except Exception as exc:
        logger.warning('Unexpected queue inspection failure for %s: %s', prompt_id, exc)
    return None



def cancel_comfyui_prompt_state(prompt_id, client_id, worker_url=None) -> ComfyPromptState:
    """Delete only LDS's exact queued prompt; never use global /interrupt."""
    if not prompt_id or not client_id:
        return ComfyPromptState.UNKNOWN
    api_addr = worker_url or api_address()

    def exact(entry):
        queued_prompt_id, queued_client_id = _queue_entry_identity(entry)
        return (str(queued_prompt_id or '') == str(prompt_id)
                and str(queued_client_id or '') == str(client_id))

    try:
        response = requests.get(
            urljoin(api_addr, '/queue'), timeout=3, allow_redirects=False)
        status = getattr(response, 'status_code', None)
        if type(status) is not int or not 200 <= status < 300:
            return ComfyPromptState.UNKNOWN
        queue = response.json()
        if not isinstance(queue, dict):
            return ComfyPromptState.UNKNOWN
        pending = queue.get('queue_pending')
        running = queue.get('queue_running')
        if not isinstance(pending, (list, tuple)) or not isinstance(running, (list, tuple)):
            return ComfyPromptState.UNKNOWN
        entries = [*pending, *running]
        identities = [_queue_entry_identity(entry) for entry in entries]
        if any(queued_prompt_id is None for queued_prompt_id, _client_id in identities):
            return ComfyPromptState.UNKNOWN
        if any(exact(entry) for entry in pending):
            response = requests.post(
                urljoin(api_addr, '/queue'), json={'delete': [prompt_id]}, timeout=3,
                allow_redirects=False)
            status = getattr(response, 'status_code', None)
            return (ComfyPromptState.DELETED if type(status) is int and 200 <= status < 300
                    else ComfyPromptState.UNKNOWN)
        if any(exact(entry) for entry in running):
            return ComfyPromptState.RUNNING
        # A visible target prompt under another/missing client id is uncertain:
        # it is not proof that our GPU work disappeared.
        if any(str(queued_prompt_id) == str(prompt_id)
               for queued_prompt_id, _client_id in identities):
            return ComfyPromptState.UNKNOWN
        return ComfyPromptState.ABSENT
    except requests.RequestException as exc:
        logger.warning('Could not cancel ComfyUI prompt %s: %s', prompt_id, exc)
    except Exception as exc:
        logger.warning('Unexpected error cancelling ComfyUI prompt %s: %s', prompt_id, exc)
    return ComfyPromptState.UNKNOWN


def cancel_comfyui_prompt(prompt_id, client_id=None, worker_url=None) -> bool:
    """Compatibility bool: only an exact pending-delete reports success."""
    return (cancel_comfyui_prompt_state(prompt_id, client_id, worker_url)
            is ComfyPromptState.DELETED)


def fetch_output_image_bytes(filename, subfolder='', timeout=30):
    """Fetch a finished image's bytes from the LOCAL ComfyUI over its HTTP API
    (`GET /view`), rather than reading the file off disk.

    This is path-INDEPENDENT: ComfyUI serves the image straight from its own
    output directory, wherever that happens to be. The disk reader
    (`_comfy_output_dir`) breaks the moment a user points ComfyUI at a custom
    output path (`--output-directory`, `extra_model_paths.yaml`, the desktop
    app's output setting…) because the app can't know that path — but the API
    can. This mirrors how other ComfyUI front-ends (e.g. SillyTavern) receive
    generated images.

    Returns the raw bytes, or None on any failure so the caller can fall back to
    its existing disk / not-found handling."""
    try:
        qs = urlencode({'filename': filename, 'subfolder': subfolder or '', 'type': 'output'})
        url = urljoin(api_address(), f"/view?{qs}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.warning(f"fetch_output_image_bytes failed for {filename!r}: {e}")
        return None


# /object_info is the heaviest probe in the app (megabytes of node schemas). Short
# TTL cache so one user action never pays for it twice. See fetch_object_info_classes.
_OBJECT_INFO_TTL = 60
_object_info_cache = {"data": None, "timestamp": 0, "key": None, "enums": None,
                      "files": None}

# --- /object_info timeout budget --------------------------------------------
# WHY ONLY THIS ONE IS A SETTING (the other ComfyUI timeouts in this file were
# audited at the same time, and deliberately left alone):
#
#   /object_info   ~MB, and GROWS with the install — every node class and every
#                  model file the user has. THE defect: any constant is wrong for
#                  somebody, and wrong for exactly the people with the richest
#                  ComfyUI. Fixed below.
#   /prompt (10s)  request is the graph, response is one id. Size is bounded by
#                  OUR workflow, not by the user's install. Also a WRITE: a longer
#                  budget on a retry-capable path buys duplicate submissions, not
#                  reliability.
#   /history/<id>  one prompt's outputs. Bounded.
#   /queue, /interrupt, /free   constant-size control calls.
#   /view (60s, streamed)       an image download; already generous.
#   capabilities._http_ok on /history (3s)  the reachability VERDICT. It has to
#                  stay snappy — it runs on every capability poll and gates the
#                  whole UI. Its problem was never the number, it was that a
#                  3 s miss was reported as "ComfyUI isn't running"; it now asks
#                  this probe (the one that waited long enough to know) instead.
#
# So: one honest knob, not five guessed ones.
# Two DIFFERENT budgets, because "ComfyUI is off" and "ComfyUI is slow" are two
# different failures and one number cannot serve both:
#
#   * CONNECT — how long we wait for the TCP handshake. A ComfyUI that isn't
#     running refuses the connection instantly on loopback, and a wrong host /
#     firewalled port fails here. This is the ONLY budget an absent ComfyUI ever
#     pays, which is what makes the read budget below safe to make generous: the
#     old single 8 s number had to be small *because* it was also the price of a
#     stopped ComfyUI. Splitting them removes that trade-off entirely.
#   * READ — how long ComfyUI may spend BUILDING the answer once it has accepted
#     the connection. This is the number that has to scale with the install: the
#     /object_info payload lists every node class and every model file, so it
#     grows with the custom-node packs and the weights the user has installed.
#     Configurable (`comfyui.object_info_timeout_s`) because no constant can be
#     right for every install — that is the whole lesson of this bug.
_OBJECT_INFO_CONNECT_TIMEOUT = 3
_OBJECT_INFO_TIMEOUT_MIN = 5
_OBJECT_INFO_TIMEOUT_MAX = 300

# A FAILED probe is cached too, for a much shorter window than a successful one.
# It used to be cached not at all ("fail-open, retried at once"), which is a fine
# intention and a bad mechanism: nothing retried the SAME call, but every other
# caller re-fired the full payload. On an install where /object_info takes 15 s,
# the capability poll, the Studio preflight and each generate therefore each paid
# it in full, back to back — and because ComfyUI builds that answer on its own
# event loop, our own storm of probes is what kept the cheap `/history`
# reachability check timing out. THAT is how a slow ComfyUI came to be reported
# as a stopped one. One failure now silences the storm for a few seconds; every
# consumer still fails OPEN, so this can only ever make the app decide FASTER,
# never differently, and `clear_model_caches()` drops it on demand.
_OBJECT_INFO_FAIL_TTL = 20
# Outcome of the last real ATTEMPT (a cache hit is not an attempt and never
# rewrites it). Doubles as the negative cache: `status != 'ok'` within
# _OBJECT_INFO_FAIL_TTL of `timestamp`, for the same api address, is served
# without a request.
_object_info_last = {"timestamp": 0, "key": None, "status": "unknown", "waited": 0}


def object_info_timeout() -> int:
    """Seconds ComfyUI may spend answering /object_info, from
    `comfyui.object_info_timeout_s`, clamped to 5-300.

    Total by construction (never raises, never returns None): an unreadable or
    absurd value falls back to the shipped default rather than disabling a probe
    the whole app leans on."""
    default = 45
    try:
        default = int((cfg.DEFAULTS.get('comfyui') or {}).get('object_info_timeout_s', 45))
    except (TypeError, ValueError):      # pragma: no cover - DEFAULTS is ours
        pass
    raw = cfg.get('comfyui.object_info_timeout_s')
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        logger.warning('ignoring unusable comfyui.object_info_timeout_s %r', raw)
        return default
    return max(_OBJECT_INFO_TIMEOUT_MIN, min(_OBJECT_INFO_TIMEOUT_MAX, value))


def object_info_health() -> dict:
    """What the LAST /object_info attempt did, so a caller can tell the two causes
    apart instead of collapsing them into "ComfyUI isn't running":

      status 'ok'          — answered.
      status 'timeout'     — the connection was ACCEPTED and ComfyUI then took
                             longer than `waited` seconds to produce the payload.
                             It is running; it is slow at enumerating itself.
      status 'unreachable' — nothing accepted the connection (or it died mid-read).
      status 'unknown'     — never probed since startup / a cache clear.

    `waited` is the read budget that was in force, so a message can quote the
    number the user would raise."""
    return {'status': _object_info_last['status'],
            'waited': _object_info_last['waited'] or object_info_timeout()}

# Widget inputs whose accepted values describe what a ComfyUI install CAN DO — a
# capability that depends on its version and on the node packs it loaded — as
# opposed to what it HAPPENS TO HAVE on disk (`ckpt_name`, `unet_name`,
# `lora_name`, `clip_name`, `vae_name`, `image`…). Only these are distilled from
# /object_info and checked against our graphs.
#
# The distinction matters and is deliberate: a missing FILE already has its own
# named error paths (the Setup asset gaps, the auto-download offers) and no
# amount of updating ComfyUI produces it, while a missing enum VALUE is only ever
# fixed by updating ComfyUI or installing the pack that registers it. Keeping the
# file-valued combos out also keeps this list small — the accepted-value arrays
# for model names are the bulk of an 8.8 MB /object_info payload.
_VERSION_SENSITIVE_INPUTS = frozenset({
    'scheduler', 'sampler_name', 'weight_dtype', 'dtype', 'type', 'device',
    'precision', 'upscale_method', 'downscale_method', 'crop',
})

# Values our shipped graphs require that a STOCK ComfyUI does not provide, mapped
# to what actually provides them. Without this the failure message can only say
# "your ComfyUI doesn't accept this", which sends people to update ComfyUI when
# updating ComfyUI will never help.
#
# `beta57` is the case reported by IndependentProcess0 (Reddit): it is NOT a
# recent core-ComfyUI scheduler — it is absent from core at every tag up to and
# including the current one. It is registered by the RES4LYF node pack, which
# appends it to comfy.samplers.SCHEDULER_HANDLERS/SCHEDULER_NAMES at import time,
# so once that pack is installed the CORE KSampler accepts it too. That is why it
# works on an install that has RES4LYF (for unrelated nodes) and fails everywhere
# else with a plain "Value not in list" on KSampler.
ENUM_VALUE_SOURCES = {
    ('scheduler', 'beta57'): ('RES4LYF', 'https://github.com/ClownsharkBatwing/RES4LYF'),
    ('scheduler', 'bong_tangent'): ('RES4LYF', 'https://github.com/ClownsharkBatwing/RES4LYF'),
}

# --- Model-FILE inputs: the same check, one field over ----------------------
# `_VERSION_SENSITIVE_INPUTS` above covers capability VALUES (a scheduler, a
# dtype). A model FILE NAME that ComfyUI does not list fails identically — same
# 400, same `value_not_in_list` — but it is deliberately kept in a SECOND view
# rather than added to the set above, because those file arrays are the bulk of
# the /object_info payload and the enum cache exists only by dropping them
# (see test_only_capability_inputs_are_distilled).
#
# What makes caching them affordable here is the class allowlist: we distill file
# lists ONLY for the loader classes our own graphs actually emit, not for every
# pack installed. `UnetLoaderGGUF` is listed so its presence is observable — NOT
# because our graphs use it (they do not; see `_GGUF_PACK`).
#
# Defined in utils.comfy_names, which also owns the emit-time canonicaliser: the
# preflight ("does this install list the name?") and the rewrite ("spell it the
# way this install does") must read the same two sets or they drift apart.
_MODEL_FILE_CLASSES = comfy_names.MODEL_FILE_CLASSES
_MODEL_FILE_INPUTS = comfy_names.MODEL_FILE_INPUTS

# Reported by naniii2352 (Discord, displayed name Dexter): a Krea 2 model
# quantised to GGUF (`krea2_turbo-Q4_K_M.gguf`) that no folder would make work.
# Core ComfyUI's `folder_paths.supported_pt_extensions` is
# {.ckpt,.pt,.pt2,.bin,.pth,.safetensors,.pkl,.sft} — `.gguf` is absent, so core
# never SCANS the file and it can never appear in any core loader's list, in any
# of the model roots. That is why copying it into all three did nothing.
#
# Loading one needs the third-party ComfyUI-GGUF pack, which registers its own
# `UnetLoaderGGUF` node reading its own `unet_gguf` folder key. Our graphs use
# core `UNETLoader`, so having the pack installed does NOT help them — the answer
# is the same either way, which is why the reason is decided from the EXTENSION
# and never from whether the pack is present.
_GGUF_PACK = ('ComfyUI-GGUF', 'https://github.com/city96/ComfyUI-GGUF')


def _distill_object_info(data):
    """{class_type: {input_name: frozenset(accepted values)}} for the COMBO inputs
    named in `_VERSION_SENSITIVE_INPUTS`.

    ComfyUI declares a combo input as `[[<choice>, <choice>, …], {options}]` — the
    type slot is a literal list of the accepted values. That list is the exact one
    ComfyUI prints in its "Value not in list" validation error, so comparing our
    graph against it reproduces ComfyUI's own verdict without a round trip.

    A class with no checkable input maps to an empty dict (it must still appear so
    the class-presence view keeps every key)."""
    out = {}
    for cls, spec in (data or {}).items():
        combos = {}
        sections = (spec.get('input') or {}) if isinstance(spec, dict) else {}
        if isinstance(sections, dict):
            for section in ('required', 'optional'):
                decls = sections.get(section)
                if not isinstance(decls, dict):
                    continue
                for name, decl in decls.items():
                    if name not in _VERSION_SENSITIVE_INPUTS:
                        continue
                    choices = decl[0] if isinstance(decl, (list, tuple)) and decl else None
                    if isinstance(choices, list) and all(isinstance(v, str) for v in choices):
                        combos[name] = frozenset(choices)
        out[cls] = combos
    return out


# Comparison key for a ComfyUI model file name — see comfy_names for the why.
# ComfyUI joins a subfolder with its OWN host's os.sep (measured: backslash on a
# live Windows 0.27.0, forward slash on the Linux install of GitHub #21), and the
# two ends of the wire need not be the same host, so raw comparison is never
# right in either direction.
_normalise_model_name = comfy_names.normalise_model_name


def _distill_model_files(data):
    """{class_type: {input_name: {normalised_name: PUBLISHED name}}} for the FILE
    inputs of the loader classes we ship (`_MODEL_FILE_CLASSES` /
    `_MODEL_FILE_INPUTS`).

    A mapping and not a set of keys: since GitHub #21 we do not only ask "does
    this install list the model?", we also need to WRITE BACK the exact string it
    published, because that is the only spelling its validator accepts. `in`
    still reads the keys, so every existing caller is unchanged.

    Restricted to those classes on purpose: this is the half of /object_info the
    enum view drops wholesale for size, and a node-rich install repeats the same
    model arrays across hundreds of classes."""
    out = {}
    for cls, spec in (data or {}).items():
        if cls not in _MODEL_FILE_CLASSES:
            continue
        combos = {}
        sections = (spec.get('input') or {}) if isinstance(spec, dict) else {}
        if isinstance(sections, dict):
            for section in ('required', 'optional'):
                decls = sections.get(section)
                if not isinstance(decls, dict):
                    continue
                for name, decl in decls.items():
                    if name not in _MODEL_FILE_INPUTS:
                        continue
                    choices = decl[0] if isinstance(decl, (list, tuple)) and decl else None
                    if isinstance(choices, list) and all(isinstance(v, str) for v in choices):
                        combos[name] = {_normalise_model_name(v): v for v in choices}
        if combos:
            out[cls] = combos
    return out


def _fetch_object_info(timeout=None, force=False):
    """(classes, enums, model_files) from ONE `GET /object_info`, all three served
    by the same short TTL cache — the payload is the heaviest probe in the app, so
    the checks below must never cost a second request. (None, None, None) on any
    failure.

    `timeout` is the READ budget in seconds; None (the normal case) reads
    `object_info_timeout()`. The connect budget is separate and fixed — see
    _OBJECT_INFO_CONNECT_TIMEOUT.

    `force` skips BOTH caches for this one call and refreshes them with what comes
    back. It exists for exactly one caller — the model-file refusal in
    `queue_prompt`, which must not kill a job on a snapshot that predates the model
    (see `confirm_unavailable_model_files`). Nothing on a hot path may pass it."""
    addr = api_address()
    now = time.time()
    if (not force and _object_info_cache["data"] is not None
            and _object_info_cache["key"] == addr
            and now - _object_info_cache["timestamp"] < _OBJECT_INFO_TTL):
        return (_object_info_cache["data"], _object_info_cache["enums"],
                _object_info_cache["files"])
    if (not force and _object_info_last["status"] not in ('ok', 'unknown')
            and _object_info_last["key"] == addr
            and now - _object_info_last["timestamp"] < _OBJECT_INFO_FAIL_TTL):
        # Negative cache: one failure answers the burst behind it. See
        # _OBJECT_INFO_FAIL_TTL — every consumer of this already fails OPEN.
        return None, None, None
    read_budget = int(timeout) if timeout else object_info_timeout()
    try:
        resp = requests.get(urljoin(addr, '/object_info'),
                            timeout=(_OBJECT_INFO_CONNECT_TIMEOUT, read_budget))
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # 'timeout' = the connection was accepted and ComfyUI then took too long to
        # build the payload — it IS running, it is slow at enumerating itself. Any
        # other failure means nothing answered. Collapsing the two is the bug this
        # split exists to kill: a user was sent to check whether ComfyUI was started
        # while it was, in fact, started and busy (j_o_e_l., Discord).
        status = 'timeout' if isinstance(e, requests.exceptions.ReadTimeout) else 'unreachable'
        _object_info_last.update(timestamp=now, key=addr, status=status,
                                 waited=read_budget)
        logger.warning('fetch_object_info failed (%s, %ss budget): %s',
                       status, read_budget, e)
        return None, None, None
    if not isinstance(data, dict):
        _object_info_last.update(timestamp=now, key=addr, status='unreachable',
                                 waited=read_budget)
        return None, None, None
    classes, enums = set(data.keys()), _distill_object_info(data)
    files = _distill_model_files(data)
    _object_info_cache.update(data=classes, timestamp=now, key=addr, enums=enums,
                              files=files)
    _object_info_last.update(timestamp=now, key=addr, status='ok', waited=read_budget)
    return classes, enums, files


def fetch_object_info_classes(timeout=None):
    """Set of node `class_type` names the target ComfyUI exposes = the KEYS of
    `GET /object_info`. Used by the Studio preflight to tell a required CUSTOM
    node (e.g. the Krea rebalance / detail-daemon nodes a workflow
    hardcodes) apart from a missing one BEFORE firing a grid every tile of which
    would fail ComfyUI validation.

    Returns None (not an empty set) on any failure so the caller can distinguish
    'ComfyUI didn't answer, can't verify nodes' (fail-open) from 'the graph uses
    a node ComfyUI doesn't have'.

    Cached for `_OBJECT_INFO_TTL` seconds per API address: /object_info is the single
    heaviest probe in the app (measured 8.8 MB / ~5 s on a node-rich install) and ONE
    Studio run asks for it twice (grid preflight + per-run class resolution). The node
    set only changes when ComfyUI restarts or a pack is installed; the refresh-models
    button (`clear_model_caches`) drops the cache, so a freshly installed node is
    visible on demand rather than after the TTL.

    HOW LONG it may take is a setting, not a constant: 8.8 MB / ~5 s is one
    install's number, and the payload grows with every node pack and every model
    file the user adds. See `object_info_timeout()`."""
    return _fetch_object_info(timeout)[0]


def fetch_object_info_model_files(timeout=None):
    """{class_type: {input_name: frozenset(normalised names)}} for the model-FILE
    inputs of the loader classes we ship, from the SAME cached /object_info as the
    class and enum views — never a second request.

    None (not an empty dict) when the probe failed, so callers can fail OPEN."""
    return _fetch_object_info(timeout)[2]


def fetch_object_info_enums(timeout=None):
    """{class_type: {input_name: frozenset(accepted values)}} for the capability
    inputs (see `_VERSION_SENSITIVE_INPUTS`), from the SAME cached /object_info as
    `fetch_object_info_classes` — never a second request.

    None (not an empty dict) when the probe failed, so callers can fail OPEN."""
    return _fetch_object_info(timeout)[1]


def unavailable_model_files(workflow, files=None):
    """Every model file name in `workflow` that the target ComfyUI does NOT list:
    [{node_id, class_type, input, value, reason}], sorted stably.
    `reason` is 'gguf' or 'not_listed'.

    This is the file half of the "designed for THIS machine" bug class that
    `unsupported_enum_values` covers for capability values. Both surface as the
    SAME ComfyUI 400 (`value_not_in_list`) that the user can only decode by
    reading ComfyUI's own console. Two independent causes reach it:

      * 'gguf'  — a `.gguf` model handed to a CORE loader. Core ComfyUI does not
        have `.gguf` in `supported_pt_extensions`, so it never scans the file and
        no folder will ever help. The app itself lists `.gguf` in its pickers
        (`_MODEL_SUFFIXES`), so it is the app that offers this dead end.
      * 'not_listed' — the name exists on the disk the APP scanned but not in the
        list the ComfyUI answering on the port serves. The app lists models with
        os.listdir; ComfyUI serves them from wherever its own process was
        configured. With more than one ComfyUI install — ComfyUI Desktop alone
        declares a shared root AND its install-directory root — those two lists
        disagree by construction, and no amount of copying reconciles them.

    Extension first: a `.gguf` is reported as such even when some pack DOES list
    it, because our graphs emit core loaders and would fail regardless.

    FAIL-OPEN in both directions, exactly like the enum check:
      * probe unreachable (`files` is None) -> [] — never block a working install;
      * class_type absent from /object_info -> skipped, because that is a MISSING
        NODE, which the node preflight already reports with the right fix.

    Matching is normalised (`_normalise_model_name`) so a separator or case
    difference is never mistaken for a missing model."""
    if files is None:
        files = fetch_object_info_model_files()
    if not files:
        return []
    out = []
    for node_id, node in (workflow or {}).items():
        if not isinstance(node, dict):
            continue
        listed_by_input = files.get(node.get('class_type'))
        if not listed_by_input:
            continue
        inputs = node.get('inputs')
        if not isinstance(inputs, dict):
            continue
        for name, value in inputs.items():
            listed = listed_by_input.get(name)
            # Non-str = a link ([node, slot]) or a number, never a file name.
            if listed is None or not isinstance(value, str) or not value:
                continue
            is_gguf = value.casefold().endswith('.gguf')
            if not is_gguf and _normalise_model_name(value) in listed:
                continue
            out.append({'node_id': str(node_id), 'class_type': node.get('class_type'),
                        'input': name, 'value': value,
                        'reason': 'gguf' if is_gguf else 'not_listed'})
    out.sort(key=lambda i: (i['input'], i['value'], i['node_id']))
    return out


# How old the /object_info snapshot behind a DETERMINISTIC refusal is allowed to be.
# Below this, re-asking cannot change the answer, so we don't; above it, we re-ask
# once. This single number is also what bounds the cost — see
# `confirm_unavailable_model_files`.
_OBJECT_INFO_RECHECK_MAX_AGE = 5


def confirm_unavailable_model_files(workflow, items):
    """`(items, workflow)` — `items` re-checked against a FRESH `/object_info`
    before anyone is allowed to kill a job over them.

    WHY THIS EXISTS (reported in #help, 2026-08-08): "there is a delay after
    deploying a new checkpoint from Canvas before LDS can use it — even though I
    can clearly select the deployed LoRA inside Comfy". Both halves were true.
    `_fetch_object_info` serves a snapshot for `_OBJECT_INFO_TTL` (60 s) and the
    deploy path does not invalidate it, so a model deployed at t+0 was judged
    against the list as it stood at t-59. The refusal that followed is
    DETERMINISTIC (`WORKFLOW_INVALIDE`, never retried — job_queue), so the job was
    killed for good, and the message sent the user to check an API address that was
    correct all along. A deterministic verdict may not rest on a stale observation:
    that is the whole content of this function.

    Re-asking WORKS, which is the fact that makes this the right fix rather than a
    hopeful one. ComfyUI answers `/object_info` from `folder_paths.get_filename_list`,
    whose cache is invalidated by the mtime of every scanned directory INCLUDING
    subdirectories (`recursive_search` records one entry per subdir), so a file
    dropped into `loras/<sub>/` invalidates it. Its second, mtime-blind cache
    (`CacheHelper`) is activated only for the duration of one `/object_info` request
    and cleared on exit — a within-request dedup, not a cross-request cache.

    COST — one extra `/object_info` in the worst case, and it cannot run away:
      * only reached when a graph is about to be REFUSED, never on a healthy queue;
      * never for `gguf`, which is definitive whatever any list says;
      * skipped when the snapshot is already younger than
        `_OBJECT_INFO_RECHECK_MAX_AGE`, which is what bounds a batch: the forced
        probe refreshes the cache, so a grid of N tiles against a genuinely missing
        model pays one probe per 5 s, not one per tile — and since the probe itself
        takes seconds on the installs where it is expensive, it throttles itself.

    Fails OPEN when the fresh probe fails: with no current evidence the honest move
    is to hand the graph to ComfyUI, whose own 400 is ground truth and is already
    handled. `gguf` items survive that, because no probe was ever their basis.

    Returns the workflow too: a name confirmed present is re-spelled against the
    FRESH list (`canonical_model_widgets`), because the spelling the graph carries
    was resolved against the stale one and ComfyUI validates by exact string.

    The third value is `fresh`: True when the surviving verdict rests on an
    observation no older than `_OBJECT_INFO_RECHECK_MAX_AGE` — either one we just
    forced, or one already that young. It is what earns the message the right to
    blame a second install; False keeps the message hedged. Nothing else may set
    it, which is the point: the sentence and the evidence travel together."""
    if not items:
        return items, workflow, False
    gguf = [i for i in items if i.get('reason') == 'gguf']
    if len(gguf) == len(items):
        # Extension, not availability: no list from anyone can make a core loader
        # read a .gguf, so there is nothing to re-ask and nothing to hedge.
        return items, workflow, True
    now = time.time()
    if (_object_info_cache["data"] is not None
            and _object_info_cache["key"] == api_address()
            and now - _object_info_cache["timestamp"] <= _OBJECT_INFO_RECHECK_MAX_AGE):
        return items, workflow, True    # already young enough to be trusted
    fresh = _fetch_object_info(force=True)[2]
    if not fresh:
        # No current evidence. Hand the graph to ComfyUI and let its own 400 judge.
        return gguf, workflow, bool(gguf)
    workflow, _ = comfy_names.canonical_model_widgets(workflow, fresh)
    return unavailable_model_files(workflow, files=fresh), workflow, True


def format_unavailable_models_message(items, rechecked=False):
    """One paste-safe English sentence for a model-file gap: which file, why this
    ComfyUI cannot use it, and the action that fixes it.

    Paste-safe = no filesystem path and no personal data — meant to be copied into
    a Discord/Reddit thread verbatim. Files are de-duplicated: the same model
    pinned on three nodes is ONE thing to fix.

    Deliberately never says "copy it into the model folder": that is the advice the
    reporter followed for an hour (into three folders) while neither cause could be
    fixed that way.

    `rechecked` says whether the caller established the gap against a FRESH
    /object_info (`confirm_unavailable_model_files`) rather than a cached one. It
    gates the "different install" sentence, which used to be asserted — "it is most
    likely a different ComfyUI install" — on evidence that did not support it: with
    a snapshot up to _OBJECT_INFO_TTL old, a model deployed a moment ago produced
    that exact sentence, and it sent its reader to check an API address that was
    right (#help, 2026-08-08). Unrechecked, the message now names the stale list as
    the first thing to rule out; rechecked, the install hypothesis is earned and the
    reader is told, in so many words, that a fresh deploy is NOT what they are
    looking at."""
    seen, bits, has_gguf, has_missing = set(), [], False, False
    for i in items or []:
        key = (i.get('input'), i.get('value'))
        if key in seen:
            continue
        seen.add(key)
        bits.append(f'{i.get("input")} = "{i.get("value")}" (on {i.get("class_type")})')
        if i.get('reason') == 'gguf':
            has_gguf = True
        else:
            has_missing = True
    fixes = []
    if has_gguf:
        pack, url = _GGUF_PACK
        fixes.append(
            "A .gguf model cannot be loaded by ComfyUI on its own: .gguf is not one "
            "of the file types ComfyUI reads, so it stays invisible in every model "
            f"folder — moving it will not help. Loading one needs the {pack} node "
            f"pack ({url}), and this app's workflows use ComfyUI's standard model "
            "loader, which cannot read .gguf even once that pack is installed. Use a "
            ".safetensors build of the model instead.")
    if has_missing and rechecked:
        fixes.append(
            "This file is on disk where the app looked, but the ComfyUI answering on "
            "the configured API address does not list it. That list was re-read "
            "seconds ago, after any model you just deployed, and the file is still "
            "not on it — so a fresh deploy is NOT what you are looking at, and "
            "waiting or deploying it again will not change it. That leaves a second "
            "ComfyUI: check that the ComfyUI API address and the models folder in "
            "Settings point at the SAME install (ComfyUI Desktop keeps a shared "
            "models folder AND one inside its install directory), then restart "
            "ComfyUI.")
    elif has_missing:
        fixes.append(
            "This file is on disk where the app looked, but the ComfyUI answering on "
            "the configured API address does not list it. If you deployed this model "
            "in the last minute, the list this check read may simply predate it — "
            "try once more. Otherwise the two ends are looking at different ComfyUI "
            "installs: check that the ComfyUI API address and the models folder in "
            "Settings point at the SAME install (ComfyUI Desktop keeps a shared "
            "models folder AND one inside its install directory), then restart "
            "ComfyUI.")
    return ("Your ComfyUI does not offer a model file this workflow requires: "
            + '; '.join(bits) + '. ' + ' '.join(fixes))


def unsupported_enum_values(workflow, enums=None):
    """Every hardcoded widget value in `workflow` that the target ComfyUI does NOT
    accept: [{node_id, class_type, input, value, pack, url}], sorted stably.

    This is the detection half of the "designed for THIS machine" bug class: our
    shipped graphs pin a scheduler / sampler / dtype, and an install that doesn't
    offer that exact value answers a raw ComfyUI 400 the user only ever sees by
    reading ComfyUI's own console.

    It deliberately does NOT substitute an equivalent value. A scheduler changes
    the render, so a silent swap would split users into two populations producing
    different images from the same app and the same settings — divergence nobody
    can see until they compare screenshots weeks later. One render path for
    everyone; when it isn't available, say so and stop.

    FAIL-OPEN in both directions:
      * probe unreachable (`enums` is None) -> [] — never block a working install;
      * class_type absent from /object_info -> skipped, because that is a MISSING
        NODE, which the node preflight already reports with the right fix."""
    if enums is None:
        enums = fetch_object_info_enums()
    if not enums:
        return []
    out = []
    for node_id, node in (workflow or {}).items():
        if not isinstance(node, dict):
            continue
        accepted_by_input = enums.get(node.get('class_type'))
        if not accepted_by_input:
            continue
        inputs = node.get('inputs')
        if not isinstance(inputs, dict):
            continue
        for name, value in inputs.items():
            accepted = accepted_by_input.get(name)
            # Non-str = a link ([node, slot]) or a number, never a combo choice.
            if accepted is None or not isinstance(value, str) or value in accepted:
                continue
            pack, url = ENUM_VALUE_SOURCES.get((name, value), (None, None))
            out.append({'node_id': str(node_id), 'class_type': node.get('class_type'),
                        'input': name, 'value': value, 'pack': pack, 'url': url})
    out.sort(key=lambda i: (i['input'], i['value'], i['node_id']))
    return out


def format_unsupported_enums_message(items):
    """One paste-safe English sentence for an enum gap: which value is missing,
    where it comes from when we know, and the action that fixes it.

    Paste-safe = no filesystem path and no personal data — this text is meant to
    be copied into a Discord/Reddit thread verbatim. Values are de-duplicated:
    the same scheduler pinned on three nodes is ONE thing to fix."""
    seen, bits, known_pack = set(), [], False
    for i in items or []:
        key = (i.get('input'), i.get('value'))
        if key in seen:
            continue
        seen.add(key)
        where = f'{i.get("input")} = "{i.get("value")}" (on {i.get("class_type")})'
        pack, url = i.get('pack'), i.get('url')
        if pack:
            known_pack = True
            where += f' — provided by the {pack} node pack'
            if url:
                where += f': {url}'
        bits.append(where)
    fix = ("Install that node pack in ComfyUI (ComfyUI-Manager ▸ Install via Git URL), "
           "then restart ComfyUI." if known_pack else
           "Update ComfyUI and its node packs (ComfyUI-Manager ▸ Update All), then "
           "restart ComfyUI.")
    return ("Your ComfyUI does not offer a value this workflow requires: "
            + '; '.join(bits) + '. '
            + "The app will not quietly swap in a different value — that would change "
              "how your images look compared to everyone else's. " + fix)


def free_comfyui_vram(worker_url=None, timeout=10) -> ComfyVramFreeVerdict:
    """Ask ComfyUI to unload only when it can positively acknowledge the request."""
    try:
        api_addr = (worker_url or api_address()).rstrip('/')
        if not api_addr:
            raise ValueError('empty ComfyUI API address')
        response = requests.post(
            f'{api_addr}/free',
            json={'unload_models': True, 'free_memory': True},
            timeout=timeout,
            allow_redirects=False,
        )
    except (requests.RequestException, OSError) as exc:
        # A message containing "connection refused" is not proof. Only an actual
        # nested OS refusal proves that no local ComfyUI process owns the card.
        verdict = (ComfyVramFreeVerdict.COMFYUI_OFFLINE
                   if _is_explicit_connection_refused(exc)
                   else ComfyVramFreeVerdict.UNKNOWN)
        logger.warning('ComfyUI /free did not complete: %s (%s)', exc, verdict.value)
        return verdict
    except Exception as exc:
        logger.warning('ComfyUI /free failed unexpectedly: %s', exc)
        return ComfyVramFreeVerdict.UNKNOWN

    status = getattr(response, 'status_code', None)
    if type(status) is int and 200 <= status < 300:
        logger.info('ComfyUI VRAM freed successfully')
        return ComfyVramFreeVerdict.FREED
    logger.warning('ComfyUI /free returned malformed/non-2xx status %r', status)
    return ComfyVramFreeVerdict.UNKNOWN


# --- Trained-LoRA parser (SINGLE source shared by labels + grouping) -------

# Steps d'entraînement ai-toolkit : zero-paddés à 9 chiffres (000004000). Un token
# tout-chiffres de 4+ caractères = un compteur de steps (pas une version 'v13').
_TRAINED_STEP_RE = re.compile(r'^\d{4,}$')
# Source-run tag token: `rc<id>` (cloud CloudTrainingRun id) / `rl<id>` (local
# TrainingRunRecord id) appended by lora_training.import_checkpoint. Kept out of
# the base/merge token list (so it is not mistaken for a recipe tag) and
# re-appended at the END of the display label — without it, two runs of the same
# dataset version collapse to one identical Test Studio name.
_RUN_TAG_TOKEN_RE = re.compile(r'^r[cl]\d+$')

# Familles d'entraînement (= pipeline). La clé interne ('zimage'/'sdxl'/'krea') et
# son libellé d'affichage : source UNIQUE, réutilisée par le studio (sélecteur de
# famille) et par le label de LoRA ci-dessous.
FAMILY_LABELS = {'zimage': 'Z-Image', 'sdxl': 'SDXL', 'krea': 'Krea 2', 'flux': 'FLUX.1',
                 'flux2klein': 'FLUX.2 Klein', 'anima': 'Anima'}

# Tags de base OFFICIELS qu'apposent lora_training._dest_base_tag aux LoRA déployés
# sur une base de famille (pas de merge). Chacun est UN token (tirets, pas
# d'underscore) → un checkpoint FINAL sans compteur de steps
# (`lora_<trigger>_<tag>`) reste parsable : le trigger est tout ce qui PRÉCÈDE ce
# tag, même s'il contient lui-même des underscores. Miroir des constantes de
# lora_training (KREA_BASE_LABEL / FLUX_BASE_LABEL / FLUX2KLEIN_BASE_LABELS +
# suffixes recette Z-Image) — dupliqué ici pour éviter un import circulaire ;
# à garder synchronisé si une famille/variante est ajoutée là-bas.
_FAMILY_BASE_TAGS = frozenset({
    'Z-Image-Turbo', 'Z-Image-Base', 'Z-Image-De-Turbo',
    'Krea-2-Turbo', 'Krea-2-Raw',
    'FLUX-1-dev', 'FLUX2-Klein-4B', 'FLUX2-Klein-9B',
    'Anima-Base',
})


def _safe_trigger_token(trigger: str) -> str:
    """Forme du trigger telle qu'elle est ENCODÉE dans le nom de fichier déployé —
    miroir EXACT de lora_training._safe_trigger (tout caractère hors alphanumérique
    et hors '_'/'-' devient '_'). Dupliqué ici (pas d'import du service, circulaire)
    pour retrouver la frontière du trigger dans le stem quand le caller connaît le
    trigger réel du dataset."""
    return ''.join(c if (c.isalnum() or c in '_-') else '_' for c in (trigger or ''))


def family_of_lora(filename: str) -> str | None:
    """Déduit la famille (pipeline) d'un LoRA de son DOSSIER ComfyUI : les LoRA
    entraînés atterrissent dans ``loras/sdxl``, ``loras/krea`` ou ``loras/z image``
    (cf. lora_training._lora_dest_dir). La famille est donc une fonction du chemin —
    pas besoin de la stocker en base. Renvoie None si pas de préfixe de dossier connu."""
    # Comparaison, pas un chemin : le nom arrive dans l'une OU l'autre convention
    # (Windows, Linux, valeur relue d'une config écrite sur l'autre OS), donc on
    # aplatit d'abord sur '/' — le même pivot que comfy_names.normalise_model_name,
    # pour qu'il n'y ait qu'UNE forme normalisée dans toute l'app.
    low = (filename or '').replace('\\', '/').lower()
    if low.startswith('sdxl/'):
        return 'sdxl'
    if low.startswith('krea/'):
        return 'krea'
    # flux2klein AVANT flux par lisibilité seulement : « flux/ » exige le séparateur
    # juste après « flux », donc « flux2klein/x » ne le matche pas — pas d'ambiguïté.
    if low.startswith('flux2klein/'):
        return 'flux2klein'
    if low.startswith('flux/'):
        return 'flux'
    if low.startswith('anima/'):
        return 'anima'
    if low.startswith(('z image/', 'zimage/', 'z-image/')):
        return 'zimage'
    return None


def _finish_parse(trigger: str, rest_tokens):
    """Partage final du parse : depuis un trigger déjà isolé et les tokens qui le
    SUIVENT, extrait le step (1er token tout-chiffres 4+), le reste (base/merge)
    et le tag de run `rc<id>`/`rl<id>` (identité du run — rendu en fin de label
    pour que deux runs de la même version dataset restent distincts)."""
    step, rest, run_tag = None, [], None
    for t in rest_tokens:
        if step is None and _TRAINED_STEP_RE.match(t):
            step = int(t)
        elif _RUN_TAG_TOKEN_RE.match(t):
            run_tag = t
        else:
            rest.append(t)
    return trigger, step, rest, run_tag


def _parse_trained_stem(filename: str, trigger: str | None = None):
    """Décompose un nom de LoRA entraîné ai-toolkit ``lora_<trigger>_<step?>_<base?>``
    en (trigger, step|None, [tokens_de_base]). Renvoie None si le nom ne suit PAS la
    convention (le caller retombe alors sur un label générique). Source UNIQUE du
    parse, partagée par le libellé lisible ET la clé de regroupement des checkpoints.

    ⚠️ Le trigger peut LUI-MÊME contenir des underscores (ex. ``leg_behind``) : il
    s'étale alors sur plusieurs tokens. Prendre bêtement ``tokens[0]`` le tronquait
    (« leg ») et poussait le reste (« behind ») dans la base — d'où un label
    « leg · behind » et un chip d'auto-injection erroné (bug rapporté 2026-07-17).
    On reconstitue donc le trigger COMPLET :

      1. `trigger` fourni (caller qui connaît le dataset) → on retire ce préfixe EXACT
         (via `_safe_trigger_token`, la forme encodée dans le fichier) et on parse le
         reste. C'est la seule voie 100 % fidèle : ``_safe_trigger`` est lossy (un
         trigger à espaces ET un trigger à underscores donnent le même nom de fichier).
      2. Sinon, ancre sur le STEP (token 6-10 chiffres, frontière non ambiguë) : le
         trigger = tout ce qui le précède. Couvre tous les checkpoints intermédiaires.
      3. Sinon (checkpoint FINAL sans step), si le dernier token est un tag de base de
         famille connu (`_FAMILY_BASE_TAGS`), le trigger = tout ce qui le précède.
      4. Sinon, repli legacy : ``tokens[0]`` (triggers mono-token + noms de merge
         tiers ``lora_Lola2_mopMix_pornmaster`` où la frontière est indevinable)."""
    stem = os.path.basename(filename).rsplit('.', 1)[0]
    if not stem.lower().startswith('lora_'):
        return None
    body = stem[len('lora_'):]
    tokens = [t for t in body.split('_') if t]
    if not tokens:
        return None

    # 1) Trigger connu du caller : frontière EXACTE, affichage fidèle (verbatim).
    if trigger:
        safe = _safe_trigger_token(trigger).strip('_')
        if safe and body.lower().startswith(safe.lower()):
            after = body[len(safe):]
            if after == '' or after[0] == '_':
                return _finish_parse(trigger, [t for t in after.split('_') if t])
        # Le hint ne colle pas (nom legacy) → on retombe sur les heuristiques.

    # 2) Ancre sur le step : le trigger = tokens AVANT le compteur (multi-token OK).
    step_idx = next((i for i, t in enumerate(tokens) if _TRAINED_STEP_RE.match(t)), None)
    if step_idx is not None and step_idx > 0:
        return _finish_parse('_'.join(tokens[:step_idx]), tokens[step_idx:])

    # 3) Pas de step : tag de base de famille, éventuellement suivi du tag de
    #    run (`rcN`/`rlN`) et du suffixe de version (`vN`) — le trigger = tout
    #    ce qui précède le tag de base. Sans peler ces suffixes, un final
    #    `…_Krea-2-Raw_rc27_v2` tombait dans le repli legacy et tronquait les
    #    triggers multi-token (`leg_behind` → `leg`).
    core = list(tokens)
    peeled = []
    while core and (re.fullmatch(r'v\d+', core[-1])
                    or _RUN_TAG_TOKEN_RE.match(core[-1])):
        peeled.insert(0, core.pop())
    if len(core) > 1 and core[-1] in _FAMILY_BASE_TAGS:
        return _finish_parse('_'.join(core[:-1]), [core[-1], *peeled])

    # 4) Repli legacy : premier token = trigger.
    return _finish_parse(tokens[0], tokens[1:])


def trained_lora_group(filename: str, family: str | None = None,
                       trigger: str | None = None):
    """Clé de REGROUPEMENT (trigger + base, SANS le step) + le step, pour empiler les
    checkpoints d'un même dataset sous une entrée dépliable dans le picker. Deux
    checkpoints frères (ex. ``lora_lola3869_000002000_Krea-2-Turbo`` et
    ``lora_lola3869_000002500_Krea-2-Turbo``) partagent la MÊME clé et ne diffèrent
    que par le step. Renvoie (None, None) si le nom n'est pas un LoRA entraîné.

    La clé = le displayName AMPUTÉ du segment « N steps » → cohérente avec le label
    affiché (cf. format_trained_lora_label) : le checkpoint final (sans step) a un
    displayName EXACTEMENT égal à la clé de son groupe. Le tag de run (`rc27` /
    `rl12`) fait partie de la clé : deux runs de la même version dataset ne
    doivent pas s'empiler sous une seule entrée. `trigger` (optionnel) = le
    trigger réel du dataset pour un parse EXACT (cf. _parse_trained_stem)."""
    parsed = _parse_trained_stem(filename, trigger)
    if not parsed:
        return None, None
    trigger, step, rest, run_tag = parsed
    if rest:
        base = ' '.join(rest)
    else:
        fam = family or family_of_lora(filename)
        base = FAMILY_LABELS.get(fam, fam) if fam else ''
    group = f'{trigger} · {base}' if base else trigger
    if run_tag:
        group = f'{group} · {run_tag}'
    return group, step


def format_trained_lora_label(filename: str, family: str | None = None,
                              trigger: str | None = None) -> str:
    """Libellé lisible pour un LoRA de personnage ai-toolkit nommé
    ``lora_<trigger>_<step?>_<mergebase?>.safetensors``.

    Le step est zero-paddé à 9 chiffres (``000004000``) ; affiché brut il se lit
    comme du bruit et rend deux checkpoints frères indiscernables. On expose les
    axes que l'utilisateur compare : le trigger, le step dé-paddé (``4000``), la
    base d'entraînement, et le tag de run (`rc27` / `rl12`) quand il est présent —
    sans lui, deux runs déployés de la même version dataset portent le même nom
    dans le Test Studio. Cette base apparaît dans le nom sous forme de tag de merge
    (``bigLove_zt3``) ; QUAND ce tag est absent (LoRA entraîné sur la base officielle
    de la famille, ex. ``lora_Lola2_000002000`` en Krea), on affiche au moins la
    PIPELINE (Krea 2 / SDXL / Z-Image) — sinon on ne sait pas avec quoi il a été fait.
    `family` est passée par les getters (le nom seul n'a pas le dossier) ; sinon
    déduite du chemin. `trigger` (optionnel) = le trigger réel du dataset : les
    callers qui l'ont (Test Studio, liste des checkpoints déployés) le passent pour
    un label EXACT même quand le trigger contient des underscores. Renvoie '' si le
    nom ne suit PAS la convention ai-toolkit (le caller retombe alors sur
    ``_clean_klein_lora_label``).

    Ex. 'lora_Lola2_000004000_bigLove_zt3'         -> 'Lola2 · 4000 steps · bigLove zt3'
        'krea/lora_Lola2_000002000' (family krea)  -> 'Lola2 · 2000 steps · Krea 2'
        'sdxl/lora_Lola2_mopMix_pornmaster'        -> 'Lola2 · mopMix pornmaster'
        'lora_leg_behind_000002000_Krea-2-Turbo'   -> 'leg_behind · 2000 steps · Krea 2'
        '…_Krea-2-Raw_rc27_v2'                    -> '… · Krea-2-Raw v2 · rc27'
    """
    parsed = _parse_trained_stem(filename, trigger)
    if not parsed:
        return ''
    trigger, step, rest, run_tag = parsed
    parts = [trigger]
    if step is not None:
        parts.append(f'{step} steps')
    if rest:
        parts.append(' '.join(rest))                 # tag de merge = la base d'entraînement
    else:
        fam = family or family_of_lora(filename)     # pas de tag -> au moins la pipeline
        if fam:
            parts.append(FAMILY_LABELS.get(fam, fam))
    if run_tag:
        parts.append(run_tag)
    return ' · '.join(parts)


def _clean_klein_lora_label(filename: str) -> str:
    """Strip noisy tokens out of Flux 2 Klein LoRA filenames for display.

    e.g. 'FLUX.2-klein-base-9B_LoRa_by-AI_Characters_STYLE_SmartphoneSnapshotPhotoReality_v13_TRIGGER$casual snapshot$.safetensors'
        -> 'Smartphone Snapshot Photo Reality v13'
    """
    name = filename.rsplit('.', 1)[0]
    # Remove the TRIGGER$xxx$ suffix if present (it's parsed separately).
    name = re.sub(r'_TRIGGER\$[^$]+\$', '', name)
    # Drop common Flux 2 boilerplate tokens (case-insensitive).
    # Both 'flux.2' (dotted) and 'flux2' (run-together) appear in the wild —
    # e.g. 'Flux2-Klein-9B-consistency-V2' should display as 'Consistency V2'.
    drop_tokens = {
        'flux.2', 'flux2', 'klein', 'klein9b', 'base', '9b',
        'lora', 'lor', 'by', 'ai', 'characters', 'style',
    }
    parts = re.split(r'[_\-]+', name)
    parts = [p for p in parts if p and p.lower() not in drop_tokens]
    # Insert spaces before capital letters in CamelCase tokens.
    parts = [re.sub(r'(?<!^)(?=[A-Z][a-z])', ' ', p) for p in parts]
    label = ' '.join(parts).strip()
    if label:
        # Capitalize the first character without flattening the rest (preserves "SEXGOD", "FK", etc).
        label = label[0].upper() + label[1:]
    return label or filename.rsplit('.', 1)[0]


# Curated trigger words for Klein LoRAs that don't carry a TRIGGER$xxx$ marker
# in their filename. Filename match is exact (basename only), case-sensitive.
#
# Each value is a list of trigger entries. An entry is either a plain string
# (label and inserted prompt are identical) or a (label, prompt) tuple where
# the chip shows `label` but clicking inserts `prompt` into the prompt field.
# Use tuples for long descriptive prompts that wouldn't fit visually as a chip.
KNOWN_KLEIN_TRIGGERS = {
    "realistic.safetensors": ["realistic"],
    "details.safetensors": ["realistic"],
    "FK_bukkakenew2.safetensors": [
        "semen, cum",
        ("wet stains", "wet semen stains on clothes/shirt/outfit."),
        ("face/body/hair", "bukkake, she has excessive cum and semen in her face. there is lots of semen on her body and breasts. she has cum in her hair."),
        ("mouth filled", "she has huge amounts of cum in her mouth. mouth is filled with cum. cum in mouth. mouth overflowing with cum. significant amount of a white, viscous substance is visible on her tongue and dripping from her mouth."),
        ("drooling", "cum is drooling out of her mouth. Semen strings and dripping semen."),
    ],
    "SEXGOD_ImprovedNudity_Klein9b_v4.safetensors": ["nude"],
}


def _normalize_trigger_entry(entry):
    """Normalize a curated entry into {'label', 'prompt'}."""
    match entry:
        case (label, prompt):
            return {"label": str(label), "prompt": str(prompt)}
        case _:
            return {"label": str(entry), "prompt": str(entry)}


def _trained_lora_trigger(filename: str, trigger: str | None = None) -> str | None:
    """Trigger word of an ai-toolkit TRAINED LoRA named ``lora_<trigger>_<step?>_<base?>``.

    The trigger is the token(s) the user baked into the captions (e.g.
    ``lora_Lola2_000002000`` -> ``Lola2``). User LoRAs carry no ``TRIGGER$..$`` marker
    and aren't in the curated map, so without this their keyword was lost (no
    auto-inject, no chip). Shares the SINGLE parse of format_trained_lora_label() so a
    multi-token trigger (``lora_leg_behind_000002000`` -> ``leg_behind``) auto-injects
    whole instead of truncating to ``leg``. Returns None if the name isn't
    ai-toolkit-shaped or the trigger resolves to a step counter (``lora_000002000``)."""
    parsed = _parse_trained_stem(filename, trigger)
    if not parsed:
        return None
    trig = (parsed[0] or '').strip()
    if not trig or _TRAINED_STEP_RE.match(trig):
        return None
    return trig


def _extract_klein_triggers(filename: str) -> list[dict] | None:
    """Resolve trigger word entries for a LoRA filename (used by ALL trained-LoRA
    getters: Klein / SDXL / Z-Image / Krea).

    Precedence:
      1) explicit `TRIGGER$xxx$` marker in the filename (Flux 2 convention),
      2) curated KNOWN_KLEIN_TRIGGERS lookup,
      3) ai-toolkit trained trigger token (`lora_<trigger>_…`, e.g. Lola2),
      4) None — let the UI hide the trigger chip.

    Returns a list of {label, prompt} dicts (one entry for single-trigger
    LoRAs, multiple entries for LoRAs with curated variants).
    """
    m = re.search(r'TRIGGER\$([^$]+)\$', filename)
    if m:
        return [_normalize_trigger_entry(m.group(1))]
    entries = KNOWN_KLEIN_TRIGGERS.get(filename)
    if entries is not None:
        return [_normalize_trigger_entry(e) for e in entries]
    # Trained character/style LoRA (ai-toolkit): the filename encodes the trigger.
    # Surfaces user LoRAs like `lora_Lola2_…` so their keyword auto-injects AND
    # shows as a locked chip — same as the official style LoRAs.
    trained = _trained_lora_trigger(filename)
    if trained:
        return [_normalize_trigger_entry(trained)]
    return None


# --- Model/LoRA Discovery ---

# Mapping of known SDXL checkpoint filenames to their Civitai model page, shown
# alongside the checkpoint in pickers. Static data — no filesystem/DB
# dependency; an unknown checkpoint just gets civitai_url=None.
CIVITAI_LINKS = {
    "bigLove_photo1.safetensors": "https://civitai.com/models/897413/big-love",
    "bigLove_xl4.safetensors": "https://civitai.com/models/897413/big-love",
    "bigLove_xl25.safetensors": "https://civitai.com/models/897413/big-love",
    "bigLust_v16.safetensors": "https://civitai.com/models/575395/big-lust",
    "gonzalomo_v20UnityDMD.safetensors": "https://civitai.com/models/1513492/gonzalomo-xlfluxpony",
    "gonzalomoXLFluxPony_v01Littleasp.safetensors": "https://civitai.com/models/1513492/gonzalomo-xlfluxpony",
    "gonzalomoXLFluxPony_v40UnityXLDMD.safetensors": "https://civitai.com/models/1513492/gonzalomo-xlfluxpony",
    "intorealismUltra_v20.safetensors": "https://civitai.com/models/1950841/intorealism-ultra",
    "lustifySDXLNSFW_endgameDMD2.safetensors": "https://civitai.com/models/573152/lustify-sdxl-nsfw-checkpoint",
    "mopMixtureOfPerverts_v10DMD.safetensors": "https://civitai.com/models/1854124/mop-mixture-of-perverts-dmd",
    "mopMixtureOfPerverts_v20DMD.safetensors": "https://civitai.com/models/1854124/mop-mixture-of-perverts-dmd",
    "mopMixtureOfPerverts_v31DMD.safetensors": "https://civitai.com/models/1854124/mop-mixture-of-perverts-dmd",
    "mopMixtureOfPervertsDMD_v40.safetensors": "https://civitai.com/models/1854124/mop-mixture-of-perverts-dmd",
    "plantMilkModelSuite_walnut.safetensors": "https://civitai.com/models/1162518/plant-milk-model-suite",
}

_checkpoint_models_cache = {"data": None, "timestamp": 0, "key": None}
_MODEL_CACHE_TTL = 300  # 5 minutes


def get_checkpoint_models(include_hidden=False):
    """List SDXL checkpoint files under models/checkpoints (+ its Biglove/
    subfolder and a few known variant subdirs).

    `include_hidden=True` returns bare basenames (used by callers that just
    need a filename whitelist, e.g. lora_training's base-model guard);
    `include_hidden=False` (default) returns [{name, civitai_url}] for picker
    UIs. Single-user app: no hidden-model filtering — both shapes cover the
    same set of files, `include_hidden` only changes the wire format.

    Cached with the shared 5-minute TTL. Returns [] when ComfyUI's output dir
    isn't configured yet."""
    current_time = time.time()
    cache_key = str(include_hidden)
    if (_checkpoint_models_cache["data"] is not None
            and _checkpoint_models_cache["key"] == cache_key
            and (current_time - _checkpoint_models_cache["timestamp"] < _MODEL_CACHE_TTL)):
        return _checkpoint_models_cache["data"]

    out_dir = _out_dir()
    if not out_dir:
        return []

    try:
        checkpoints_dir = os.path.normpath(os.path.join(out_dir, "..", "models", "checkpoints"))
        biglove_dir = os.path.join(checkpoints_dir, "Biglove")

        search_dirs = [d for d in (biglove_dir, checkpoints_dir) if os.path.exists(d)]
        for subdir in ("diffusers", "unet", "stable-diffusion", "xl", "sdxl"):
            subdir_path = os.path.join(checkpoints_dir, subdir)
            if os.path.exists(subdir_path):
                search_dirs.append(subdir_path)
        # Any checkpoints root declared in extra_model_paths.yaml. Portable /
        # Stability-Matrix / A1111-shared installs keep every checkpoint OUTSIDE
        # <base>/models, so this picker showed them an empty SDXL base list while
        # ComfyUI loaded those very files. Additive: no yaml -> nothing appended and
        # this list is byte-for-byte the historical one. (The scan below is
        # recursive, so subfolders under an extra root are covered like Biglove/.)
        try:
            from ..services import comfy_model_paths
            search_dirs += [d for d in comfy_model_paths.extra_roots("checkpoints")
                            if os.path.exists(d)]
        except Exception:
            pass

        if not search_dirs:
            logger.warning(f"Checkpoint model directories not found: {biglove_dir} (and parent {checkpoints_dir})")
            return []

        all_model_files = set()
        for s_dir in search_dirs:
            found = glob.glob(os.path.join(s_dir, "*.safetensors"))
            found += glob.glob(os.path.join(s_dir, "**", "*.safetensors"), recursive=True)
            all_model_files.update(os.path.basename(f) for f in found)

        sorted_models = sorted(all_model_files)
        logger.info(f"Total checkpoint models found: {len(sorted_models)}")

        if include_hidden:
            result = sorted_models
        else:
            result = [{"name": m, "civitai_url": CIVITAI_LINKS.get(m)} for m in sorted_models]

        _checkpoint_models_cache["data"] = result
        _checkpoint_models_cache["timestamp"] = time.time()
        _checkpoint_models_cache["key"] = cache_key
        return result
    except Exception as e:
        logger.error(f"Error listing checkpoint models: {e}", exc_info=True)
        return []


def resolve_checkpoint_ckpt_name(name):
    """Map a checkpoint BASENAME (as returned by get_checkpoint_models, which strips
    the folder via os.path.basename) to the path RELATIVE to models/checkpoints that
    ComfyUI's CheckpointLoaderSimple expects, e.g. 'bigLove_photo5.safetensors' ->
    'Biglove/bigLove_photo5.safetensors' on Linux, 'Biglove\\bigLove_photo5.safetensors'
    on Windows, but 'sam3.1_…' stays at the root.

    Without this the loader rejects the prompt (400 'value_not_in_list'). Names that
    already contain a separator (already a relative path) keep their segments;
    unknown names — or an unconfigured ComfyUI output dir — fall back to themselves.

    The separator is the one of the tree we WALKED (os.sep), never a hardcoded
    backslash: it used to be, and on Linux that made every subfoldered checkpoint
    unloadable (GitHub #21, 1Tomber). `queue_prompt_to_comfyui` has the last word
    and respells this against the target install's published list.

    The walk covers every checkpoints root the PICKER lists from — the default
    tree plus each root declared in extra_model_paths.yaml — because the picker
    offers all of them as bare basenames. Walking only the default tree meant a
    checkpoint in a SUBFOLDER of an extra root fell through to the bare-name
    fallback, the preflight then looked for models/checkpoints/<basename> and
    raised a 409 naming a path the file never lived at (GitHub #36, KingyWolf).
    Matching is case-insensitive with the ON-DISK spelling returned — same
    contract as _resolve_lora_rel_by_basename, and disk casing is what ComfyUI
    publishes. Additive: no yaml -> the historical single-tree walk, unchanged."""
    if not name:
        return name
    if "\\" in name or "/" in name:
        return local_model_path(name)
    out_dir = _out_dir()
    if not out_dir:
        return name
    roots = []
    try:
        roots.append(os.path.normpath(os.path.join(out_dir, "..", "models", "checkpoints")))
    except OSError:
        pass
    try:
        from ..services import comfy_model_paths
        roots += [r for r in comfy_model_paths.extra_roots("checkpoints")
                  if r not in roots]
    except Exception:
        pass
    target = name.lower()
    for ck_dir in roots:
        try:
            for root, _dirs, files in os.walk(ck_dir):
                hit = next((f for f in files if f.lower() == target), None)
                if hit:
                    return os.path.relpath(os.path.join(root, hit), ck_dir)
        except OSError:
            continue
    return name


def _model_scan_roots(out_dir):
    """The diffusion-model folders the Studio listers walk: `<ComfyUI>/models/unet`
    and `.../diffusion_models`, plus any diffusion_models root declared in
    extra_model_paths.yaml (the `unet` key folds into the same canonical type).

    Derived from the OUTPUT dir rather than from `comfy_model_paths.search_roots`
    — deliberately, for now: they are two config routes to the same folders, and
    swapping one for the other here would be a behaviour change on any install
    where they disagree, not a refactor. Extracted so the two listers cannot drift
    on the question of WHERE to look, which is half of how four scanners diverge.
    Additive: no yaml -> nothing appended, list unchanged."""
    models_root = os.path.normpath(os.path.join(out_dir, "..", "models"))
    roots = [os.path.join(models_root, b) for b in ("unet", "diffusion_models")]
    try:
        from ..services import comfy_model_paths
        roots += comfy_model_paths.extra_roots("diffusion_models")
    except Exception:                       # noqa: BLE001 — an absent yaml is normal
        pass
    return roots


_zimage_models_cache = {"data": None, "timestamp": 0}


def get_zimage_models():
    """List Z-Image UNET checkpoints: .safetensors files under a 'z image'
    subfolder of models/unet or models/diffusion_models. Returns names in the
    UNETLoader form — relative to the base dir, joined with the separator of the
    tree we walked (os.sep), e.g. 'z image\\bigLove_zt3.safetensors' on Windows
    and 'z image/bigLove_zt3.safetensors' on Linux; the queue respells it against
    the target ComfyUI's own list. Cached with the shared TTL. Returns []
    when ComfyUI's output dir isn't configured yet."""
    current_time = time.time()
    if (_zimage_models_cache["data"] is not None
            and current_time - _zimage_models_cache["timestamp"] < _MODEL_CACHE_TTL):
        return _zimage_models_cache["data"]
    out = []
    out_dir = _out_dir()
    if out_dir:
        try:
            from ..services import comfy_model_paths
            # No `root_file_accept`: this family has no root-filename rule. A
            # `diffusion_models` root also holds Krea, FLUX and Klein weights, and
            # nothing in a Z-Image filename separates them reliably — the folder
            # IS the claim here. (Krea does have such a rule; see get_krea_models.)
            # 'z-image' too: the retired capabilities scanner accepted the
            # hyphen spelling (z[ -]?image), so a user's Z-Image/ folder showed
            # ✓ in Setup while this lister — the one the Studio picker actually
            # reads — could not see it. Folding the fifth scanner onto this one
            # (probe == picker == resolver) surfaced the gap; the token list is
            # where the tolerance belongs.
            out = comfy_model_paths.scan_family_tree(
                _model_scan_roots(out_dir), ("z image", "zimage", "z-image"))
        except Exception as e:
            logger.error(f"get_zimage_models error: {e}")
    _zimage_models_cache["data"] = out
    _zimage_models_cache["timestamp"] = current_time
    return out


_krea_models_cache = {"data": None, "timestamp": 0}


def _krea_root_candidate(name) -> bool:
    """Is this ROOT-level file CLAIMED by the Krea family? A `diffusion_models`
    root also holds Z-Image, FLUX and Klein weights, so at a root the filename is
    the only claim there is.

    Only the claim. Whether a listed build is one the identity-edit LoRA behaves
    on is no longer asked here at all: a file on the user's disk is offered, and
    only `elect_krea_base` declines to PREFER a flagged one when the app is the
    one choosing.

    The wired workflow default used to be named explicitly on this line. It was
    dead code (the name carries 'krea' and matches no exclusion) and it was the
    last hardcoded filename in the lister, so it is gone.
    """
    return 'krea' in str(name or '').lower()


def get_krea_models():
    """List Krea 2 UNET checkpoints: le défaut du workflow (krea2_turbo_fp8.safetensors
    à la racine de models/unet ou models/diffusion_models) + tout .safetensors/.gguf
    sous un sous-dossier 'krea' (ex. 'Krea\\monKrea.safetensors') + tout fichier de
    RACINE dont le NOM porte 'krea'. Noms en forme
    UNETLoader (relatifs au dossier de base, séparateur de l'arbre parcouru =
    os.sep ; la file d'attente les réécrit selon la liste publiée par le ComfyUI
    ciblé). Cache TTL partagé. Vide si
    ComfyUI n'est pas encore configuré.

    THE ROOT-FILENAME RULE, AND WHY IT WAS MISSING
    ----------------------------------------------
    The directory-only rule made the app's OWN full-model deliveries invisible
    here. `fp8_local_delivery` writes the fp8 twin of a dense run to the ROOT of
    `diffusion_models` — deliberately, because that is a folder ComfyUI reads —
    so the one file the whole dense lane exists to produce could not be picked as
    a Test Studio base. The only way to try a model you had paid hours of GPU for
    was to open ComfyUI by hand.

    That it was an oversight and not a rule is settled by the Generate surface:
    `krea_edit_helper._krea_unet_folders` has always matched 'krea' in the folder
    OR in the filename, root included. Aligning on it retro-fits every twin
    already on disk without moving a byte, and it borrows the same exclusion
    list — BigLove* carries 'krea' and renders pure noise under this pipeline.

    Still NOT "every root file": a `diffusion_models` root also holds Z-Image,
    FLUX and Klein weights, and listing those as Krea bases would trade one
    silent wrong result for another."""
    current_time = time.time()
    if (_krea_models_cache["data"] is not None
            and current_time - _krea_models_cache["timestamp"] < _MODEL_CACHE_TTL):
        return _krea_models_cache["data"]
    out = []
    out_dir = _out_dir()
    if out_dir:
        try:
            from ..services import comfy_model_paths
            # No `accept=`: a Krea file on the user's disk is listed, full stop.
            # Dropping the ones KREA_INCOMPATIBLE_TOKENS flags meant a build in
            # their own Krea/ folder was absent with nothing saying why. The
            # measured fact lives on as a warning next to the name, and in
            # elect_krea_base, which still will not PREFER a flagged build when
            # the app is the one choosing.
            out = comfy_model_paths.scan_family_tree(
                _model_scan_roots(out_dir), ("krea",),
                root_file_accept=_krea_root_candidate)
        except Exception as e:
            logger.error(f"get_krea_models error: {e}")
    _krea_models_cache["data"] = out
    _krea_models_cache["timestamp"] = current_time
    return out


def clear_model_caches() -> None:
    """Drop the 5-min TTL caches of every base/model lister.

    Call this whenever the ComfyUI location changes (settings save) — otherwise a
    freshly-configured `comfyui.base_dir` wouldn't surface in the training-base
    dropdowns until the TTL expired (up to 5 min of a stale empty list, read by the
    user as "models still not found" right after they pointed the app at ComfyUI).
    SRC exposed `invalidate_model_caches`; this app dropped that helper, so the
    caches were never invalidated on config change until now."""
    for c in (_checkpoint_models_cache, _zimage_models_cache, _krea_models_cache,
              _object_info_cache):   # a newly installed node pack must show up NOW
        c["data"] = None
        c["timestamp"] = 0
        if "key" in c:
            c["key"] = None
        if "enums" in c:      # the enum view rides the same /object_info payload
            c["enums"] = None
    # The NEGATIVE /object_info cache goes with them: "I just started ComfyUI /
    # just changed the URL, refresh" must re-probe now, not in 20 s.
    _object_info_last.update(timestamp=0, key=None, status='unknown', waited=0)


def get_zimage_loras():
    """List Z-Image LoRAs: .safetensors under a 'z image' / 'zimage' subfolder of
    models/loras. Returns [{filename, displayName, triggerWord, triggerWords, group,
    step}] with filename in LoraLoader form ('z image\\zchar_emma.safetensors').
    Trigger words use the same $...$ filename convention as Klein LoRAs. Empty list
    when ComfyUI's loras dir isn't configured yet."""
    out = []
    lora_dir = _lora_dir()
    try:
        if lora_dir and os.path.isdir(lora_dir):
            for root, _dirs, files in os.walk(lora_dir):
                rel_dir = os.path.relpath(root, lora_dir)
                low = rel_dir.lower()
                if "z image" not in low and "zimage" not in low and "z-image" not in low:
                    continue
                for f in sorted(files):
                    if not f.lower().endswith(".safetensors"):
                        continue
                    rel = f if rel_dir == "." else os.path.join(rel_dir, f)
                    triggers = _extract_klein_triggers(f)
                    grp, stp = trained_lora_group(f, 'zimage')
                    out.append({
                        "filename": rel,
                        "displayName": format_trained_lora_label(f, 'zimage') or _clean_klein_lora_label(f),
                        "triggerWord": triggers[0]["prompt"] if triggers else None,
                        "triggerWords": triggers,
                        # group/step : regroupement des checkpoints d'un même dataset dans le picker.
                        "group": grp,
                        "step": stp,
                    })
    except Exception as e:
        logger.error(f"get_zimage_loras error: {e}")
    return out


def get_sdxl_loras():
    """List SDXL LoRAs: .safetensors under the 'sdxl' subfolder of models/loras.
    Ce sont les LoRA de PERSONNAGE/concept ENTRAÎNÉS pour SDXL (déployés par
    import_checkpoint), à NE PAS confondre avec les LoRA système 'subtle'
    (enhancement, hors périmètre). Returns [{filename, displayName, triggerWord,
    triggerWords, group, step}] avec filename en forme LoraLoader
    ('sdxl\\lora_Lola_000001000.safetensors'). Vide si non configuré."""
    out = []
    lora_dir = _lora_dir()
    try:
        if lora_dir and os.path.isdir(lora_dir):
            for root, _dirs, files in os.walk(lora_dir):
                rel_dir = os.path.relpath(root, lora_dir)
                low = rel_dir.lower()
                # UNIQUEMENT le dossier 'sdxl' (pas subtle/z image/klein/wan).
                if low != 'sdxl' and not low.startswith('sdxl' + os.sep):
                    continue
                for f in sorted(files):
                    if not f.lower().endswith(".safetensors"):
                        continue
                    rel = f if rel_dir == "." else os.path.join(rel_dir, f)
                    triggers = _extract_klein_triggers(f)
                    grp, stp = trained_lora_group(f, 'sdxl')
                    out.append({
                        "filename": rel,
                        "displayName": format_trained_lora_label(f, 'sdxl') or _clean_klein_lora_label(f),
                        "triggerWord": triggers[0]["prompt"] if triggers else None,
                        "triggerWords": triggers,
                        "group": grp,
                        "step": stp,
                    })
    except Exception as e:
        logger.error(f"get_sdxl_loras error: {e}")
    return out


def get_krea_loras():
    """List Krea 2 LoRAs: .safetensors under the 'krea' subfolder of models/loras.
    Ce sont les LoRA entraînés POUR Krea 2 (ex. realism_engine_krea2). Returns
    [{filename, displayName, triggerWord, triggerWords, group, step}] avec filename
    en forme LoraLoader ('krea\\realism_engine_krea2_v1.safetensors'). Vide si non
    configuré."""
    out = []
    lora_dir = _lora_dir()
    try:
        if lora_dir and os.path.isdir(lora_dir):
            for root, _dirs, files in os.walk(lora_dir):
                rel_dir = os.path.relpath(root, lora_dir)
                low = rel_dir.lower()
                # NE JAMAIS confondre avec un dossier FRÈRE 'krea_styles' (LoRA de
                # style officiels — hors périmètre, cf. get_krea_style_loras dropped).
                if low == 'krea_styles' or low.startswith('krea_styles' + os.sep):
                    continue
                if low != 'krea' and not low.startswith('krea' + os.sep):
                    continue
                for f in sorted(files):
                    if not f.lower().endswith(".safetensors"):
                        continue
                    rel = f if rel_dir == "." else os.path.join(rel_dir, f)
                    triggers = _extract_klein_triggers(f)
                    grp, stp = trained_lora_group(f, 'krea')
                    out.append({
                        "filename": rel,
                        "displayName": format_trained_lora_label(f, 'krea') or _clean_klein_lora_label(f),
                        "triggerWord": triggers[0]["prompt"] if triggers else None,
                        "triggerWords": triggers,
                        "group": grp,
                        "step": stp,
                    })
    except Exception as e:
        logger.error(f"get_krea_loras error: {e}")
    return out


def _clean_flux2_klein_model_label(filename: str) -> str:
    """Make a Flux 2 Klein model filename human-readable for a picker dropdown.

    e.g. 'flux-2-klein-9b-fp8.safetensors' -> 'Flux 2 Klein 9B fp8'.
    Falls back to the bare basename if the cleanup leaves nothing useful.
    """
    name = filename.rsplit('.', 1)[0]
    name = re.sub(r'[-_]+', ' ', name).strip()
    # Title-case word-by-word, preserving common quantization suffixes (fp8, q4, etc.)
    parts = []
    for word in name.split(' '):
        if re.fullmatch(r'(?i)(fp\d+|q\d+|bf16|nf4|gguf)', word):
            parts.append(word.lower())
        elif re.fullmatch(r'(?i)(kv|vae|clip|t5|cn)', word):
            parts.append(word.upper())
        elif re.fullmatch(r'\d+[bB]', word):
            parts.append(word.upper())
        else:
            parts.append(word.capitalize())
    cleaned = ' '.join(parts)
    return cleaned or filename.rsplit('.', 1)[0]


def get_flux2_klein_models():
    """Scan the Flux 2 Klein diffusion models directory.

    Returns a list of {filename, displayName} dicts. `filename` is the path
    relative to the ComfyUI diffusion_models root (with the 'Flux2 klein\\'
    subfolder prefix) so it can be injected directly into a workflow's
    UNETLoader `unet_name` field.

    Back-compat: if the 'Flux2 klein/' subfolder is missing or empty, falls
    back to any root-level Flux 2 Klein files so the picker still works
    during the file-move transition. Empty list when unconfigured."""
    lora_dir = _lora_dir()
    if not lora_dir:
        return []
    try:
        diffusion_dir = os.path.join(os.path.dirname(lora_dir), "diffusion_models")
        subfolder = os.path.join(diffusion_dir, "Flux2 klein")

        models = []
        if os.path.isdir(subfolder):
            for path in sorted(glob.glob(os.path.join(subfolder, "*.safetensors"))):
                filename = os.path.basename(path)
                models.append({
                    "filename": f"Flux2 klein\\{filename}",
                    "displayName": _clean_flux2_klein_model_label(filename),
                })

        # Fallback: pick up any Flux 2 Klein file still at the diffusion_models root
        # so the picker keeps working while the user is mid-migration.
        if not models and os.path.isdir(diffusion_dir):
            for path in sorted(glob.glob(os.path.join(diffusion_dir, "*klein*.safetensors"))):
                filename = os.path.basename(path)
                models.append({
                    "filename": filename,
                    "displayName": _clean_flux2_klein_model_label(filename),
                })

        logger.info(f"Found {len(models)} Flux 2 Klein model(s)")
        return models

    except Exception as e:
        logger.error(f"Error scanning Flux 2 Klein models: {e}", exc_info=True)
        return []


# --- LoRA-chain injectors ---------------------------------------------------

# Samplers / schedulers / précision exposés pour le mode Krea 2 Turbo. SOURCE UNIQUE
# partagée : whitelist côté route generate ET côté studio de test (anti-injection —
# une valeur hors liste est ignorée), + peuplent les dropdowns du front via /config.
# Krea 2 = flow-matching (DiT) : seuls les sampler/scheduler connus pour converger
# proprement (défaut workflow = er_sde / simple, en tête). weight_dtype = options
# RÉELLES du UNETLoader node 20 ('default' = bf16 sur le fichier fp8 -> pas d'overflow
# matmul fp8 avec le Krea2T-Enhancer ; 'fp8_e4m3fn' = défaut rapide).
KREA_ALLOWED_SAMPLERS = [
    'er_sde', 'euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_2m_sde',
    'dpmpp_sde', 'res_multistep', 'deis', 'ddim', 'uni_pc',
]
KREA_ALLOWED_SCHEDULERS = [
    'simple', 'sgm_uniform', 'beta', 'normal', 'ddim_uniform',
    'kl_optimal', 'linear_quadratic',
]
KREA_ALLOWED_WEIGHT_DTYPES = frozenset({
    'default', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2',
})


def inject_krea_loras(workflow, requested, allowed, unet_node="20", consumers=("26",)):
    """Chain LoraLoaderModelOnly nodes after the Krea 2 UNETLoader (node 20) and
    repoint its model consumers (KSampler node 26) to the end of the chain.

    `requested` = [{filename, strength}], `allowed` = whitelist of filenames
    (path-injection guard). Strength clamped to [-2.0, 20.0] — garde anti-absurde
    seulement : la plage UX (6 en général, 20 pour les LoRA utility type
    filter-bypass qui n'agissent qu'à strength >10) est portée par le slider front.
    Négatif autorisé (tire un slider LoRA vers son pôle négatif — même plancher
    que Z-Image/SDXL) ; les LoRA always-on restent clampés ≥0 EN AMONT par leurs
    appelants (lora_test_studio), donc ce plancher ne les élargit pas.
    An effective strength of exactly 0.0 is a true no-op: no loader node is
    created. Returns the number of LoRAs injected; 0 leaves the workflow untouched.
    Independent of the conditioning rebalance (node 30), on the prompt path."""
    if unet_node not in workflow or not isinstance(requested, list):
        return 0
    prev = unet_node
    injected = 0
    for idx, item in enumerate(requested):
        if not isinstance(item, dict):
            continue
        fn = str(item.get("filename") or "")
        if fn not in allowed:
            continue
        try:
            strength = max(-2.0, min(20.0, float(item.get("strength", 1.0))))
        except (TypeError, ValueError):
            strength = 1.0
        # Keep the graph canonical for a semantic no-op.  Current ComfyUI also
        # short-circuits a zero-strength loader, while omitting the node avoids
        # depending on that implementation detail in older or third-party nodes.
        if strength == 0.0:
            continue
        node_id = f"krea_lora_{idx}"
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": fn, "strength_model": strength, "model": [prev, 0]},
            "_meta": {"title": f"Krea 2 LoRA {idx}"},
        }
        prev = node_id
        injected += 1
    if injected:
        for cons in consumers:
            node = workflow.get(cons)
            if node and isinstance(node.get("inputs", {}).get("model"), list):
                node["inputs"]["model"] = [prev, 0]
    return injected


# Krea2T-Enhancer (MODEL-side patcher). NODE_CLASS_MAPPINGS key confirmed in SRC.
KREA2T_ENHANCER_CLASS = "ComfyUI-Krea2T-Enhancer"
KREA2T_ENHANCER_NODE_ID = "krea2t_enhancer"


def inject_krea2t_enhancer(workflow, enabled, strength):
    """Insert the Krea2T-Enhancer as the LAST MODEL patcher before KSampler(26).

    Wire-aware: consumes whatever currently feeds KSampler.model (node 20, or the
    last LoRA node when a LoRA stack is present) and repoints KSampler.model to the
    enhancer. Order-independent w.r.t. LoRA injection (call this AFTER it).

    enabled falsy  -> returns 0, workflow untouched.
    enabled truthy -> adds one node, returns 1. strength clamped to [0.0, 2.0].
    Missing KSampler / model input -> returns 0 (fail-safe; never breaks dispatch).
    """
    if not enabled:
        return 0
    ks = workflow.get("26")
    if not ks or "model" not in ks.get("inputs", {}):
        return 0
    try:
        s = max(0.0, min(2.0, float(strength)))
    except (TypeError, ValueError):
        s = 1.0
    src = ks["inputs"]["model"]
    workflow[KREA2T_ENHANCER_NODE_ID] = {
        "class_type": KREA2T_ENHANCER_CLASS,
        "inputs": {"model": src, "enabled": True, "strength": s, "debug": False},
    }
    ks["inputs"]["model"] = [KREA2T_ENHANCER_NODE_ID, 0]
    return 1


def inject_zimage_loras(workflow, requested, allowed,
                        unet_node="1", consumers=("7", "9")):
    """Chain LoraLoaderModelOnly nodes after the Z-Image UNETLoader and repoint
    its model consumers to the end of the chain.

    `requested` = [{filename, strength}], `allowed` = whitelist of filenames
    (path-injection guard). Strength clamped to [-2.0, 6.0]. Returns the number
    of LoRAs injected; 0 leaves the workflow untouched."""
    if unet_node not in workflow or not isinstance(requested, list):
        return 0
    prev = unet_node
    injected = 0
    for idx, item in enumerate(requested):
        if not isinstance(item, dict):
            continue
        fn = str(item.get("filename") or "")
        if fn not in allowed:
            continue
        try:
            # Négatif autorisé (inverse le concept, plage UI -2..2) ; max 6 conservé
            # pour rétro-compat avec les anciennes valeurs persistées.
            strength = max(-2.0, min(6.0, float(item.get("strength", 1.0))))
        except (TypeError, ValueError):
            strength = 1.0
        node_id = f"z_lora_{idx}"
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": fn, "strength_model": strength, "model": [prev, 0]},
            "_meta": {"title": f"Z-Image LoRA {idx}"},
        }
        prev = node_id
        injected += 1
    if injected:
        for cons in consumers:
            node = workflow.get(cons)
            if node and isinstance(node.get("inputs", {}).get("model"), list):
                node["inputs"]["model"] = [prev, 0]
    return injected


def inject_sdxl_loras(workflow, requested, allowed, anchor="25"):
    """Chaîne des LoraLoader (model+clip) APRÈS le LoraLoader d'ancrage (node 25 = Style
    LoRA) dans le workflow SDXL/HQ, et repointe les consommateurs de l'ancre vers le dernier
    maillon. Permet une PILE de LoRA SDXL perso (en plus du Style LoRA du node 25).

    `requested` = [{filename, strength}] (filename en forme LoraLoader 'sdxl\\…') ;
    `allowed` = whitelist de filenames (garde anti path-injection + owner). Strength clampé
    [-2.0, 6.0]. Retourne le nombre de LoRA injectés ; 0 laisse le workflow intact."""
    if anchor not in workflow or not isinstance(requested, list):
        return 0
    # Consommateurs ACTUELS de l'ancre (AVANT insertion) -> à repointer en fin de chaîne
    # (sinon le 1er maillon inséré, qui lit aussi l'ancre, serait repointé sur lui-même).
    consumers = [nid for nid, node in workflow.items()
                 if isinstance(node, dict)
                 and (node.get("inputs", {}).get("model") == [anchor, 0]
                      or node.get("inputs", {}).get("clip") == [anchor, 1])]
    prev = anchor
    injected = 0
    for idx, item in enumerate(requested):
        if not isinstance(item, dict):
            continue
        fn = str(item.get("filename") or "")
        if fn not in allowed:
            continue
        try:
            # Négatif autorisé (plage UI -2..2) ; max 6 conservé (rétro-compat).
            strength = max(-2.0, min(6.0, float(item.get("strength", 1.0))))
        except (TypeError, ValueError):
            strength = 1.0
        node_id = f"sdxl_lora_{idx}"
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {"lora_name": fn, "strength_model": strength, "strength_clip": strength,
                       "model": [prev, 0], "clip": [prev, 1]},
            "_meta": {"title": f"SDXL LoRA {idx}"},
        }
        prev = node_id
        injected += 1
    if injected:
        for cons in consumers:
            node = workflow.get(cons)
            inp = node.get("inputs", {}) if node else {}
            if inp.get("model") == [anchor, 0]:
                inp["model"] = [prev, 0]
            if inp.get("clip") == [anchor, 1]:
                inp["clip"] = [prev, 1]
    return injected


# --- Ollama Management ---
# Kept only because queue_prompt_to_comfyui's Ollama-node-detection branch
# calls ensure_ollama_running() — this app's vision captioning
# (app.services.vision_ollama) also runs through the same local Ollama.

def check_ollama_running(host="127.0.0.1", port=11434):
    """Checks if Ollama is running by connecting to its port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, port)) == 0
    except Exception as e:
        logger.error(f"Error checking Ollama status: {e}")
        return False


def start_ollama():
    """Starts Ollama in the background."""
    try:
        creation_flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        subprocess.Popen(["ollama", "serve"], creationflags=creation_flags)
        logger.info("Ollama service start command issued.")
        return True
    except FileNotFoundError:
        logger.error("Ollama executable not found in PATH.")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False


def ensure_ollama_running():
    """Checks if Ollama is running, and starts it if not."""
    if not check_ollama_running():
        logger.info("Ollama is not running. Attempting to start it...")
        if start_ollama():
            # Wait a bit for it to start
            for i in range(10):
                time.sleep(1)
                if check_ollama_running():
                    logger.info(f"Ollama started successfully after {i+1} seconds.")
                    return True
            logger.warning("Ollama start command issued but port is still closed after 10 seconds.")
            return False
        else:
            return False
    return True
