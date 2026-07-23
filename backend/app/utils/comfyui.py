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

import glob
import logging
import os
import re
import socket
import subprocess
import time
from urllib.parse import urlencode, urljoin

import requests
from flask import current_app

from .. import config as cfg

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


# --- Circuit breaker for ComfyUI history polling ---
# Exponential backoff: each time 3 consecutive failures trip the circuit,
# the open window doubles (30s -> 60s -> 120s -> 240s cap). A successful call
# (counter reset) also resets the backoff window to 30s.
_COMFYUI_CIRCUIT_INITIAL_S = 30
_COMFYUI_CIRCUIT_MAX_S = 240
_COMFYUI_CIRCUIT_FAIL_THRESHOLD = 3

_comfyui_consecutive_failures = 0
_comfyui_circuit_open_until = 0
_comfyui_circuit_next_window_s = _COMFYUI_CIRCUIT_INITIAL_S


def _check_comfyui_circuit():
    """Check if the ComfyUI circuit breaker allows a request."""
    global _comfyui_circuit_open_until
    if time.time() < _comfyui_circuit_open_until:
        return False  # Circuit is open, skip call
    return True


def _record_comfyui_failure():
    """Record a ComfyUI call failure; open the circuit with exponential backoff if threshold reached."""
    global _comfyui_consecutive_failures, _comfyui_circuit_open_until, _comfyui_circuit_next_window_s
    _comfyui_consecutive_failures += 1
    if _comfyui_consecutive_failures >= _COMFYUI_CIRCUIT_FAIL_THRESHOLD:
        window = _comfyui_circuit_next_window_s
        _comfyui_circuit_open_until = time.time() + window
        logger.warning(
            f"ComfyUI circuit breaker opened for {window}s "
            f"after {_comfyui_consecutive_failures} consecutive failures"
        )
        # Double the next window for the *next* open, capped.
        _comfyui_circuit_next_window_s = min(window * 2, _COMFYUI_CIRCUIT_MAX_S)


def _record_comfyui_success():
    """Reset failure counter and backoff window after a successful call."""
    global _comfyui_consecutive_failures, _comfyui_circuit_next_window_s
    _comfyui_consecutive_failures = 0
    _comfyui_circuit_next_window_s = _COMFYUI_CIRCUIT_INITIAL_S


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


def save_sampler_params_overrides(overrides: dict) -> None:
    """Persist admin overrides to `sampler_params.json` (atomic write).

    Raises OSError on disk failure. The admin endpoint should let those
    propagate as a 500 so the operator sees the real cause.
    """
    import json
    tmp_path = _SAMPLER_PARAMS_JSON_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, _SAMPLER_PARAMS_JSON_PATH)


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

def load_workflow_local(file_path):
    """Charge un fichier JSON de workflow ComfyUI et retourne les données parsées, ou None en cas d'erreur."""
    import json
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        current_app.logger.info(f"Successfully loaded workflow from {file_path}")
        return workflow_data
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

    try:
        payload = {"prompt": prompt_workflow, "client_id": client_id}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(urljoin(api_addr, "/prompt"), json=payload, headers=headers, timeout=10)
        response.raise_for_status()
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

        # Si c'est une erreur de connexion sur le worker local, essayer de redémarrer
        if is_local and ("Failed to establish a new connection" in str(e) or "Connection refused" in str(e)):
            logger.info("Connection error on local worker, attempting to restart ComfyUI...")
            result = _ensure_comfyui_before_generation()
            if result is not None and result[0]:
                logger.info("ComfyUI restarted successfully, retrying prompt...")
                try:
                    response = requests.post(urljoin(api_addr, "/prompt"), json=payload, headers=headers, timeout=10)
                    response.raise_for_status()
                    return response.json(), None
                except Exception as retry_error:
                    logger.error(f"Retry after ComfyUI restart failed: {retry_error}")
            elif result is not None:
                logger.error(f"Failed to restart ComfyUI: {result[1]}")

        detail = f": {e}" + (f" | ComfyUI: {err_body}" if err_body else '')
        return None, f"Failed to connect or communicate with ComfyUI API ({api_addr}){detail}"
    except Exception as e:
        logger.error(f"Unexpected error queuing prompt to {api_addr}: {e}")
        return None, f"An unexpected error occurred: {e}"


def get_comfyui_history(prompt_id, worker_url=None):
    """Récupère l'historique d'un prompt ComfyUI par son ID.

    Args:
        prompt_id: ID du prompt retourné par ComfyUI.
        worker_url: URL optionnelle du worker distant. Si None, utilise api_address().
    """
    api_addr = worker_url or api_address()
    try:
        response = requests.get(urljoin(api_addr, f"/history/{prompt_id}"), timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            if e.response.status_code == 404:
                logger.debug(f"History for {prompt_id} not found yet (404).")
                return None
            else:
                logger.error(f"Error getting history for {prompt_id} from {api_addr}: HTTP {e.response.status_code}")
                return None
        else:
            logger.error(f"Error getting history for {prompt_id} from {api_addr}: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error getting history from {api_addr}: {e}")
        return None


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


def cancel_comfyui_prompt(prompt_id, client_id=None, worker_url=None) -> bool:
    """Cancel one known ComfyUI prompt without blindly interrupting the GPU.

    ComfyUI's ``/interrupt`` endpoint is global: calling it without first
    checking ``/queue`` can stop an unrelated workflow submitted by another
    app. We therefore interrupt only when the exact prompt is currently in
    ``queue_running``. If it has not started yet, the targeted ``/queue``
    delete operation is used instead. The optional LDS job id (sent as
    ComfyUI's ``client_id``) is checked when the queue exposes it.

    Returns whether a cancellation request was sent. Network errors and an
    already-finished prompt are normal best-effort failures and return False.
    """
    if not prompt_id:
        return False
    api_addr = worker_url or api_address()

    def _matches(entry):
        queued_prompt_id, queued_client_id = _queue_entry_identity(entry)
        if str(queued_prompt_id or '') != str(prompt_id):
            return False
        # Some ComfyUI builds omit client_id from /queue. A prompt_id is unique,
        # but when both sides provide a client id require it to match as an
        # additional guard against interrupting somebody else's work.
        return not (client_id and queued_client_id
                    and str(queued_client_id) != str(client_id))

    try:
        response = requests.get(urljoin(api_addr, "/queue"), timeout=3)
        response.raise_for_status()
        queue = response.json() or {}

        if any(_matches(entry) for entry in queue.get('queue_pending') or []):
            response = requests.post(urljoin(api_addr, "/queue"),
                                     json={'delete': [prompt_id]}, timeout=3)
            response.raise_for_status()
            return True

        if any(_matches(entry) for entry in queue.get('queue_running') or []):
            response = requests.post(urljoin(api_addr, "/interrupt"), timeout=3)
            response.raise_for_status()
            return True
    except requests.exceptions.RequestException as exc:
        logger.warning("Could not cancel ComfyUI prompt %s: %s", prompt_id, exc)
    except Exception as exc:
        logger.warning("Unexpected error cancelling ComfyUI prompt %s: %s", prompt_id, exc)
    return False


def download_image_from_worker(filename, worker_url, output_dir):
    """Télécharge une image générée depuis un worker distant via l'API ComfyUI.

    Args:
        filename: Nom du fichier image (ex: "123_GeneratedImage_00001_.png").
        worker_url: URL de l'API du worker (ex: "http://192.168.1.100:8188/").
        output_dir: Répertoire local où sauvegarder l'image.

    Returns:
        True si succès, False sinon.
    """
    try:
        url = urljoin(worker_url, f"/view?filename={filename}&type=output")
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Image téléchargée depuis worker : {filename} -> {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Erreur téléchargement image {filename} depuis {worker_url} : {e}")
        return False


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


def fetch_object_info_classes(timeout=8):
    """Set of node `class_type` names the target ComfyUI exposes = the KEYS of
    `GET /object_info`. Used by the Studio preflight to tell a required CUSTOM
    node (e.g. the Krea rebalance / detail-daemon nodes a workflow
    hardcodes) apart from a missing one BEFORE firing a grid every tile of which
    would fail ComfyUI validation.

    Returns None (not an empty set) on any failure so the caller can distinguish
    'ComfyUI didn't answer, can't verify nodes' (fail-open) from 'the graph uses
    a node ComfyUI doesn't have'."""
    try:
        resp = requests.get(urljoin(api_address(), '/object_info'), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return set(data.keys()) if isinstance(data, dict) else None
    except Exception as e:
        logger.warning(f"fetch_object_info_classes failed: {e}")
        return None


def free_comfyui_vram(worker_url=None):
    """
    Free ComfyUI VRAM by calling /free endpoint.
    Should be called before video generation to avoid OOM.
    """
    try:
        api_addr = worker_url or api_address()
        # POST /free with unload_models=true and free_memory=true
        response = requests.post(
            f"{api_addr}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10
        )
        if response.status_code == 200:
            logger.info("ComfyUI VRAM freed successfully")
            return True
        else:
            logger.warning(f"ComfyUI /free returned status {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Failed to free ComfyUI VRAM: {e}")
        return False


# --- Trained-LoRA parser (SINGLE source shared by labels + grouping) -------

# Steps d'entraînement ai-toolkit : zero-paddés à 9 chiffres (000004000). Un token
# tout-chiffres de 4+ caractères = un compteur de steps (pas une version 'v13').
_TRAINED_STEP_RE = re.compile(r'^\d{4,}$')
# Source-run tag token: `rc<id>` (cloud CloudTrainingRun id) / `rl<id>` (local
# TrainingRunRecord id) appended by lora_training.import_checkpoint. It carries
# run identity for the ☁/💻 #N chips, so it is stripped from display labels.
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
    low = (filename or '').replace('/', '\\').lower()
    if low.startswith('sdxl\\'):
        return 'sdxl'
    if low.startswith('krea\\'):
        return 'krea'
    # flux2klein AVANT flux par lisibilité seulement : « flux\\ » exige le backslash
    # juste après « flux », donc « flux2klein\\x » ne le matche pas — pas d'ambiguïté.
    if low.startswith('flux2klein\\'):
        return 'flux2klein'
    if low.startswith('flux\\'):
        return 'flux'
    if low.startswith('anima\\'):
        return 'anima'
    if low.startswith(('z image\\', 'zimage\\', 'z-image\\')):
        return 'zimage'
    return None


def _finish_parse(trigger: str, rest_tokens):
    """Partage final du parse : depuis un trigger déjà isolé et les tokens qui le
    SUIVENT, extrait le step (1er token tout-chiffres 4+) et le reste (base/merge),
    en jetant le tag de run `rc<id>`/`rl<id>` (surfacé en chip ☁/💻 #N, pas du bruit
    de label)."""
    step, rest = None, []
    for t in rest_tokens:
        if step is None and _TRAINED_STEP_RE.match(t):
            step = int(t)
        elif _RUN_TAG_TOKEN_RE.match(t):
            continue
        else:
            rest.append(t)
    return trigger, step, rest


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

    # 3) Pas de step : tag de base de famille en fin → trigger = tout ce qui précède.
    if len(tokens) > 1 and tokens[-1] in _FAMILY_BASE_TAGS:
        return _finish_parse('_'.join(tokens[:-1]), [tokens[-1]])

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
    displayName EXACTEMENT égal à la clé de son groupe. `trigger` (optionnel) = le
    trigger réel du dataset pour un parse EXACT (cf. _parse_trained_stem)."""
    parsed = _parse_trained_stem(filename, trigger)
    if not parsed:
        return None, None
    trigger, step, rest = parsed
    if rest:
        base = ' '.join(rest)
    else:
        fam = family or family_of_lora(filename)
        base = FAMILY_LABELS.get(fam, fam) if fam else ''
    group = f'{trigger} · {base}' if base else trigger
    return group, step


def format_trained_lora_label(filename: str, family: str | None = None,
                              trigger: str | None = None) -> str:
    """Libellé lisible pour un LoRA de personnage ai-toolkit nommé
    ``lora_<trigger>_<step?>_<mergebase?>.safetensors``.

    Le step est zero-paddé à 9 chiffres (``000004000``) ; affiché brut il se lit
    comme du bruit et rend deux checkpoints frères indiscernables. On expose les
    axes que l'utilisateur compare : le trigger, le step dé-paddé (``4000``) et la
    base d'entraînement. Cette base apparaît dans le nom sous forme de tag de merge
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
    """
    parsed = _parse_trained_stem(filename, trigger)
    if not parsed:
        return ''
    trigger, step, rest = parsed
    parts = [trigger]
    if step is not None:
        parts.append(f'{step} steps')
    if rest:
        parts.append(' '.join(rest))                 # tag de merge = la base d'entraînement
    else:
        fam = family or family_of_lora(filename)     # pas de tag -> au moins la pipeline
        if fam:
            parts.append(FAMILY_LABELS.get(fam, fam))
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
    'Biglove\\bigLove_photo5.safetensors', but 'sam3.1_…' stays at the root.

    Without this the loader rejects the prompt (400 'value_not_in_list'). Names that
    already contain a separator (already a relative path) are returned unchanged;
    unknown names — or an unconfigured ComfyUI output dir — fall back to themselves."""
    if not name:
        return name
    if "\\" in name or "/" in name:
        return name.replace("/", "\\")
    out_dir = _out_dir()
    if not out_dir:
        return name
    try:
        ck_dir = os.path.normpath(os.path.join(out_dir, "..", "models", "checkpoints"))
        for root, _dirs, files in os.walk(ck_dir):
            if name in files:
                rel = os.path.relpath(os.path.join(root, name), ck_dir)
                return rel.replace("/", "\\")
    except OSError:
        pass
    return name


_zimage_models_cache = {"data": None, "timestamp": 0}


def get_zimage_models():
    """List Z-Image UNET checkpoints: .safetensors files under a 'z image'
    subfolder of models/unet or models/diffusion_models. Returns names in the
    UNETLoader form (relative to the base dir, backslash-joined), e.g.
    'z image\\bigLove_zt3.safetensors'. Cached with the shared TTL. Returns []
    when ComfyUI's output dir isn't configured yet."""
    current_time = time.time()
    if (_zimage_models_cache["data"] is not None
            and current_time - _zimage_models_cache["timestamp"] < _MODEL_CACHE_TTL):
        return _zimage_models_cache["data"]
    out = []
    out_dir = _out_dir()
    if out_dir:
        try:
            models_root = os.path.normpath(os.path.join(out_dir, "..", "models"))
            for base in ("unet", "diffusion_models"):
                base_dir = os.path.join(models_root, base)
                if not os.path.isdir(base_dir):
                    continue
                for root, _dirs, files in os.walk(base_dir):
                    rel_dir = os.path.relpath(root, base_dir)
                    low = rel_dir.lower()
                    if "z image" not in low and "zimage" not in low:
                        continue
                    for f in files:
                        if f.lower().endswith((".safetensors", ".gguf", ".sft")):
                            rel = f if rel_dir == "." else os.path.join(rel_dir, f)
                            out.append(rel.replace("/", "\\"))
            out = sorted(set(out))
        except Exception as e:
            logger.error(f"get_zimage_models error: {e}")
    _zimage_models_cache["data"] = out
    _zimage_models_cache["timestamp"] = current_time
    return out


_krea_models_cache = {"data": None, "timestamp": 0}


def get_krea_models():
    """List Krea 2 UNET checkpoints: le défaut du workflow (krea2_turbo_fp8.safetensors
    à la racine de models/unet ou models/diffusion_models) + tout .safetensors/.gguf
    sous un sous-dossier 'krea' (ex. 'Krea\\monKrea.safetensors'). Noms en forme
    UNETLoader (relatifs au dossier de base, backslash). Cache TTL partagé. Vide si
    ComfyUI n'est pas encore configuré."""
    current_time = time.time()
    if (_krea_models_cache["data"] is not None
            and current_time - _krea_models_cache["timestamp"] < _MODEL_CACHE_TTL):
        return _krea_models_cache["data"]
    out = []
    out_dir = _out_dir()
    if out_dir:
        try:
            models_root = os.path.normpath(os.path.join(out_dir, "..", "models"))
            for base in ("unet", "diffusion_models"):
                base_dir = os.path.join(models_root, base)
                if not os.path.isdir(base_dir):
                    continue
                # Le défaut câblé dans krea2_turbo.json (racine) reste choisissable.
                if os.path.isfile(os.path.join(base_dir, "krea2_turbo_fp8.safetensors")):
                    out.append("krea2_turbo_fp8.safetensors")
                for root, _dirs, files in os.walk(base_dir):
                    rel_dir = os.path.relpath(root, base_dir)
                    if rel_dir == "." or "krea" not in rel_dir.lower():
                        continue
                    for f in files:
                        if f.lower().endswith((".safetensors", ".gguf", ".sft")):
                            out.append(os.path.join(rel_dir, f).replace("/", "\\"))
            out = sorted(set(out))
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
    for c in (_checkpoint_models_cache, _zimage_models_cache, _krea_models_cache):
        c["data"] = None
        c["timestamp"] = 0
        if "key" in c:
            c["key"] = None


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
                    rel = (f if rel_dir == "." else os.path.join(rel_dir, f)).replace("/", "\\")
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
                    rel = (f if rel_dir == "." else os.path.join(rel_dir, f)).replace("/", "\\")
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
                    rel = (f if rel_dir == "." else os.path.join(rel_dir, f)).replace("/", "\\")
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
    Returns the number of LoRAs injected; 0 leaves the workflow untouched.
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
