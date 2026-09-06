"""🧹 Free memory — hand back what the machine's AI processes keep cached.

The 📊 machine-load readout answers "how full is this machine"; this is the
button beside it that answers "then give it back". Two things hold memory on
a machine running LDS, and neither returns it on its own:

* **ComfyUI** keeps the models it last used cached — offloaded to system RAM
  once a later load of its own needs the card — and, idle, never lets go. Measured
  on the maintainer's machine: 34 GB of commit charge on an IDLE ComfyUI,
  the whole day's models. Its own `/free` endpoint (`unload_models` +
  `free_memory`) is the lever, the same one LDS pulls before every vision
  pass and before every local training (`utils.comfyui.free_comfyui_vram`;
  the training half only since 2026-09-05 -- this line said it earlier than
  the code did).
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


# How long the forced gesture waits for ComfyUI to let go of an interrupted
# render before saying so: a render that pages notices the interrupt at its
# next step, which can be a minute away.
INTERRUPT_WAIT_SECONDS = 45


class MemoryReleaseBusy(RuntimeError):
    """Refused: something is using the memory the button would take away.
    `can_interrupt` says the obstacle is a render of LDS's own that a second
    press may interrupt (2026-09-06: a refusal must not be a wall)."""
    def __init__(self, message, *, can_interrupt=False, job_id=None):
        super().__init__(message)
        self.can_interrupt = bool(can_interrupt)
        self.job_id = job_id


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
    reason, _kind = _busy()
    return reason


def _busy():
    """(reason, kind): the sentence a refusal shows and what stands in the way
    ('training' or 'comfy'), or (None, None)."""
    from . import cloud_training
    try:
        if cloud_training.training_in_progress():
            return ('a LoRA training is running - it holds the memory it needs; free it once it ends.',
                    'training')
    except Exception:
        logger.debug('training check failed (free memory continues)', exc_info=True)
    if comfyui_queue_busy():
        return ('ComfyUI is rendering (its queue is not empty) - unloading now would only make '
                'that job reload everything; try again when it finishes.', 'comfy')
    return (None, None)


def _own_running():
    """('idle'|'own'|'foreign'|'unknown', row) — what ComfyUI runs, seen from LDS."""
    from ..job_queue import queue_manager
    try:
        return queue_manager.own_running_job()
    except Exception:
        logger.debug('own-running check failed (free memory refuses)', exc_info=True)
        return ('unknown', None)


_OFFER = (' Press 🧹 Free memory again within a minute to interrupt that render and free the '
          'memory; the render is dropped.')
_UNMAPPED = ("ComfyUI is rendering a job of LDS's own whose id was lost; it cannot be interrupted "
             'from here. Resolve it from the recovery banner, or wait for it to finish.')


def _offer(job_id, lead="ComfyUI is rendering a clip of LDS's own."):
    return MemoryReleaseBusy(lead + _OFFER, can_interrupt=True, job_id=job_id)


def _automatic_work_reason():
    """The local Live channel: a second press must not drop its clip (found
    in review). Its sentence, said for this gesture."""
    from . import live_studio
    live = live_studio.current()
    if (live is not None and live.state in ('starting', 'running', 'stopping')
            and live.params.get('gpu', 'local') != 'rented'):
        return 'Stop the local Live channel and let its current clips finish before freeing the memory.'
    return None


def _interrupt_and_wait(job, *, wait_seconds=INTERRUPT_WAIT_SECONDS):
    """The second press: the render ComfyUI runs for LDS is cancelled and
    interrupted, and the gesture waits for ComfyUI to let go. The job id
    (None when the render ended by itself meanwhile), or MemoryReleaseBusy —
    with the offer again when the obstacle is still a render of LDS's own:
    another one than the person was told about, or the same one that has not
    let go yet (a render that pages notices the interrupt at its next step)."""
    from ..job_queue import queue_manager
    verdict, running_id = queue_manager.interrupt_own_running(job.job_id)
    if verdict == 'idle':
        return None
    if verdict == 'other':
        raise _offer(running_id, "Another clip of LDS's own is rendering now.")
    if verdict == 'foreign':
        raise MemoryReleaseBusy("ComfyUI is now running a job that is not LDS's; wait for it to finish.")
    if verdict == 'unmapped':
        raise MemoryReleaseBusy(_UNMAPPED)
    if verdict != 'interrupted':
        raise MemoryReleaseBusy('ComfyUI did not answer when asked to stop its render; try again in a moment.')
    from ..utils.comfyui import running_prompt_identity
    ours = str(job.comfyui_prompt_id or '')
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        if comfyui_queue_busy() is False:
            return str(job.job_id)
        identity = running_prompt_identity()
        if identity is not None and identity[0] and identity[0] != ours:
            # Ours left the card and the next job in line (the worker's, or
            # not LDS's) started right after: nothing is unloaded under it.
            raise MemoryReleaseBusy('The render was interrupted as asked, but ComfyUI started the next job '
                                    'right after; nothing was unloaded.')
        if time.monotonic() >= deadline:
            raise MemoryReleaseBusy('ComfyUI was asked to stop the render but has not let go yet - '
                                    'a render that pages can take a minute or two to notice. '
                                    'Press 🧹 Free memory again in a moment to ask once more.',
                                    can_interrupt=True, job_id=job.job_id)
        time.sleep(1)


def _round(v):
    return round(float(v), 1) if isinstance(v, (int, float)) else None


def free_memory(*, interrupt=False, job_id=None, interrupt_wait_seconds=INTERRUPT_WAIT_SECONDS,
                settle_seconds=SETTLE_SECONDS) -> dict:
    """The whole gesture, synchronous: guard → ComfyUI /free → release the
    vision model LDS loaded → a fresh machine reading. Raises
    MemoryReleaseBusy when refused; every other failure is REPORTED in the
    answer (an offline ComfyUI holds nothing, a vision server that did not
    answer is said as such), never raised.

    `interrupt` is the second press: when the obstacle is a render of LDS's
    own (the running prompt is one of its queue rows), that render is
    cancelled and interrupted first — the first press refused with
    `can_interrupt` and the job's id, so the person chose it for THAT job
    (`job_id`; another render on the card by then is offered afresh, never
    dropped unasked). A training, the local Live channel, a prompt that is
    not LDS's, and an LDS render whose prompt id was lost are refused either
    way."""
    from . import system_stats
    from ..job_queue import GPU_ARBITER_LOCK
    from ..utils.comfyui import ComfyVramFreeVerdict, free_comfyui_vram
    reason, kind = _busy()
    interrupted = None
    held = False
    if reason and kind == 'comfy':
        state, job = _own_running()
        if state == 'unmapped':
            raise MemoryReleaseBusy(_UNMAPPED)
        if state == 'own':
            automatic = _automatic_work_reason()
            if automatic:
                raise MemoryReleaseBusy(automatic)
            if not interrupt or (job_id and str(job_id) != str(job.job_id)):
                raise _offer(job.job_id, "Another clip of LDS's own is rendering now."
                             if job_id else "ComfyUI is rendering a clip of LDS's own.")
            # The wait runs WITHOUT the arbiter lock (the dock and the Stop
            # buttons keep answering); the unload then takes it and re-checks
            # the queue, so the next job the worker sent in between is never
            # unloaded under (found in review, both ways).
            interrupted = _interrupt_and_wait(job, wait_seconds=interrupt_wait_seconds)
            if not GPU_ARBITER_LOCK.acquire(timeout=5):
                raise MemoryReleaseBusy('The render was interrupted as asked, but something else is claiming '
                                        'the GPU right now; nothing was unloaded. Try again in a moment.')
            held = True
            try:
                still, _kind = _busy()
                if still:
                    raise MemoryReleaseBusy(('The render was interrupted as asked, but ComfyUI started the '
                                             'next job right after; nothing was unloaded.') if interrupted else still)
            except BaseException:
                GPU_ARBITER_LOCK.release()
                held = False
                raise
            reason = None
    if reason:
        raise MemoryReleaseBusy(reason)
    try:
        before = system_stats.machine_stats(force=True)
        verdict = free_comfyui_vram()
    finally:
        if held:
            GPU_ARBITER_LOCK.release()
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
        'interrupted': interrupted,
        'vision_released': vision,
        'ram_before_gb': ram_before, 'ram_after_gb': ram_after, 'ram_total_gb': _round(after.get('ram_total_gb')),
        'vram_before_gb': vram_before, 'vram_after_gb': vram_after,
        'freed_gb': freed,
    }
