"""🧹 Free memory — hand back what the machine's AI processes keep cached.

The 📊 machine-load readout answers "how full is this machine"; this is the
button beside it that answers "then give it back". Two things hold memory on
a machine running LDS, and neither returns it on its own:

* **ComfyUI** keeps every model it loaded in the session cached — offloaded to
  system RAM when it leaves the card — until it is asked to let go. Measured
  on the maintainer's machine: 34 GB of commit charge on an IDLE ComfyUI,
  the whole day's models. Its own `/free` endpoint (`unload_models` +
  `free_memory`) is the lever, the same one LDS already pulls before a
  training (`utils.comfyui.free_comfyui_vram`).
* **The vision model** LDS loaded into Ollama / LM Studio for captioning,
  kept warm by the lease so a batch does not reload it per image.
  `vision_llm.unload_vision_model` releases the models LDS itself loaded and
  never a model another tool put there — that refusal is the fence's rule and
  it does not move here.

The gesture is refused while something is rendering or training: unloading
under a job would only make that job reload everything, slower. It reports
what it measured, before and after, rather than what it hoped: ComfyUI
acknowledges the request and unloads on its own loop a moment later, so the
reading is taken after a short settle and the numbers are the OS's.
"""
from __future__ import annotations

import gc
import logging
import time

import requests

logger = logging.getLogger(__name__)

# ComfyUI unloads from its prompt-worker loop, not inside the /free request.
# Two seconds is measured slack for a 30 GB cache to be handed back.
SETTLE_SECONDS = 2.0


class MemoryReleaseBusy(RuntimeError):
    """Refused: something is using the memory the button would take away."""


def comfyui_queue_busy():
    """True when ComfyUI is rendering or has jobs waiting, False when its queue
    is empty, None when it cannot be asked (offline = nothing to free there)."""
    from ..utils.comfyui import api_address
    try:
        api_addr = (api_address() or '').rstrip('/')
        if not api_addr:
            return None
        resp = requests.get(f'{api_addr}/queue', timeout=(2, 4), allow_redirects=False)
        if resp.status_code != 200:
            return None
        queue = resp.json()
        running = queue.get('queue_running') if isinstance(queue, dict) else None
        pending = queue.get('queue_pending') if isinstance(queue, dict) else None
        if not isinstance(running, list) or not isinstance(pending, list):
            return None
        return bool(running) + len(pending) > 0
    except (requests.RequestException, ValueError, OSError):
        return None


def _busy_reason():
    from . import cloud_training
    try:
        if cloud_training.training_in_progress():
            return 'a LoRA training is running - it holds the memory it needs; free it once it ends.'
    except Exception:
        logger.debug('training check failed (free memory continues)', exc_info=True)
    if comfyui_queue_busy():
        return 'ComfyUI is rendering (its queue is not empty) - unloading now would only make ' \
               'that job reload everything; try again when it finishes.'
    return None


def _round(v):
    return round(float(v), 1) if isinstance(v, (int, float)) else None


def free_memory(*, settle_seconds=SETTLE_SECONDS) -> dict:
    """The whole gesture, synchronous: guard → ComfyUI /free → release the
    vision model LDS loaded → a fresh machine reading. Raises
    MemoryReleaseBusy when refused; every other failure is REPORTED in the
    answer (an offline ComfyUI holds nothing, a vision server that did not
    answer is said as such), never raised."""
    from . import system_stats
    from ..utils.comfyui import ComfyVramFreeVerdict, free_comfyui_vram
    reason = _busy_reason()
    if reason:
        raise MemoryReleaseBusy(reason)
    before = system_stats.machine_stats(force=True)
    verdict = free_comfyui_vram()
    vision = None
    try:
        from . import vision_llm
        vision = bool(vision_llm.unload_vision_model())
    except Exception:
        logger.debug('vision model release failed (free memory continues)', exc_info=True)
        vision = False
    gc.collect()
    if settle_seconds:
        time.sleep(settle_seconds)
    after = system_stats.machine_stats(force=True)
    ram_before, ram_after = _round(before.get('ram_used_gb')), _round(after.get('ram_used_gb'))
    vram_before, vram_after = _round(before.get('vram_used_gb')), _round(after.get('vram_used_gb'))
    freed = (round(ram_before - ram_after, 1)
             if ram_before is not None and ram_after is not None else None)
    # Keyed by VALUE, not by member: a rebuilt enum (the suite once reloaded
    # utils.comfyui) keeps its values, and a member of the old class is not a
    # key of the new one — the release runner read every verdict as 'unknown'.
    comfy = {ComfyVramFreeVerdict.FREED.value: 'freed',
             ComfyVramFreeVerdict.COMFYUI_OFFLINE.value: 'offline'}.get(getattr(verdict, 'value', None), 'unknown')
    return {
        'ok': True,
        'comfyui': comfy,
        'vision_released': vision,
        'ram_before_gb': ram_before, 'ram_after_gb': ram_after, 'ram_total_gb': _round(after.get('ram_total_gb')),
        'vram_before_gb': vram_before, 'vram_after_gb': vram_after,
        'freed_gb': freed,
    }
