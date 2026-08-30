"""Slim image job queue: a FIFO worker over `ImageGenerationQueue` plus a tiny
JSON key/value store (`SystemState`) used by lifted services for cross-request
flags (e.g. training locks, test-studio run state).

Replaces the source app's ~181 KB queue_manager. `_submit`/`_poll_outputs` are
the only two functions that talk to ComfyUI; both lazy-import
`app.utils.comfyui` (lifted in Task 13) so a missing/broken module fails the
job cleanly instead of crashing the worker thread. `_dispatch_completion`
lazy-imports the owning service (routing on job metadata) so this module never
imports the services that create jobs (avoids import cycles).
"""
from __future__ import annotations
from .utils.timestamps import naive_utcnow
import json
import logging
import threading
import time
import uuid
from typing import NamedTuple

from .extensions import db
from .models import ImageGenerationQueue, SystemState

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 15 * 60
STUCK_TIMEOUT_MINUTES = 10
IDLE_SLEEP_SECONDS = 1

# A ComfyUI history outage is neither a successful empty history nor a normal
# generation failure.  Keep a durable, no-TTL barrier until the user explicitly
# reconciles the exact prompt after recovering ComfyUI.
COMFYUI_UNHEALTHY_GRACE_SECONDS = 60
COMFYUI_STALLED_BARRIER_KEY = 'comfyui_stalled_barrier'
COMFYUI_STALLED_MESSAGE = (
    'ComfyUI stopped answering. Recover or restart ComfyUI, then cancel and resume '
    'this Test Studio batch.'
)
POLL_STALLED = object()
COMFYUI_UNKNOWN_SUBMIT_MESSAGE = (
    'ComfyUI /prompt result was lost. Restart ComfyUI before cancelling or resuming '
    'this batch; LDS cannot safely identify the remote prompt.'
)
COMFYUI_RECOVERY_REQUIRED_MESSAGE = (
    'A previous ComfyUI job has an unresolved remote state. Recover or restart '
    'ComfyUI, then resolve the paused job (confirm the restart when prompted, or '
    'cancel/resume it) before starting another local generation.'
)


# Why the worker will claim nothing at all. Stable KEYS — the wording lives at
# each surface (see `QueueManager.gpu_hold`). Deliberately NOT the fifth
# condition `process_one` checks ("a job is already running"): that one is the
# serialization working as designed, not a hold.
HOLD_TRAINING = 'training'
HOLD_VISION = 'vision'
HOLD_COMFYUI_RECOVERY = 'comfyui_recovery'
HOLD_OLLAMA_FENCE = 'ollama_fence'
# Noun phrases, for a caller writing "waiting for {x}".
HOLD_LABELS = {
    HOLD_TRAINING: 'LoRA training',
    HOLD_VISION: 'a vision pass',
    HOLD_COMFYUI_RECOVERY: 'a paused ComfyUI job',
    HOLD_OLLAMA_FENCE: 'a busy local Ollama endpoint',
}


class ComfyUIRecoveryRequired(RuntimeError):
    """Raised when new ComfyUI work must wait for explicit recovery."""


class _ComfySubmitRejected(RuntimeError):
    """A deterministic pre-submit refusal (safe to surface as terminal)."""


class _ComfySubmitUnknown(RuntimeError):
    """A /prompt outcome that may already own remote GPU work."""


# One LDS process owns the SQLite-backed queue. The lock closes the small local
# race between a vision window claim and a ComfyUI /prompt submission; the
# persisted flags and queue rows remain the recovery record after a restart.
GPU_ARBITER_LOCK = threading.RLock()


def require_comfyui_enqueue_ready() -> None:
    """Refuse new ComfyUI work while an unresolved remote owner is recorded.

    Deliberately checks only the durable recovery barrier. Training and Vision
    are temporary scheduling fences: jobs may queue behind them. A stalled
    ComfyUI owner is different because accepting more work would leave rows
    looking active even though the worker is intentionally fail-closed.
    """
    with GPU_ARBITER_LOCK:
        if queue_manager.has_comfyui_stalled_barrier():
            raise ComfyUIRecoveryRequired(COMFYUI_RECOVERY_REQUIRED_MESSAGE)


# --- automatic recovery ------------------------------------------------------
# One recovery case, and only one, is provable without a human: a *known prompt*
# barrier whose id ComfyUI no longer knows. If ComfyUI answers and neither its
# /queue nor its /history contains that prompt, the remote work is gone (the
# ordinary "ComfyUI was restarted / the machine died" case) and keeping the
# barrier only blocks every dataset in the app for a job nobody can act on.
# The three refusals are deliberate, not conservatism for its own sake:
#   * ComfyUI unreachable        -> no evidence at all, so no decision.
#   * the prompt is still there  -> the job is ALIVE; clearing it would strand
#                                   real GPU work whose outputs still arrive.
#   * an unknown_submit barrier  -> no prompt id exists to ask about, and a
#                                   timed-out POST can still land later; only a
#                                   person who restarted ComfyUI can rule it out.
COMFYUI_AUTO_RESOLVED_MESSAGE = (
    'A stalled generation from a previous session was cleared automatically '
    '(ComfyUI was restarted).'
)
AUTO_RECOVERY_NOTICE_TTL_SECONDS = 600
_auto_recovery_notice = None
_auto_recovery_notice_lock = threading.Lock()


def _record_auto_recovery_notice(owner) -> dict:
    """Remember one automatic clear so the UI can say it out loud.

    In-process on purpose: this is a transient courtesy message, not recovery
    state. The durable record of what happened is the queue row plus the log
    line — this only decides whether a toast appears.
    """
    global _auto_recovery_notice
    notice = {'id': uuid.uuid4().hex,
              'message': COMFYUI_AUTO_RESOLVED_MESSAGE,
              'job_id': owner.get('job_id'),
              'dataset_id': owner.get('dataset_id'),
              'run_id': owner.get('run_id')}
    with _auto_recovery_notice_lock:
        _auto_recovery_notice = (notice, time.monotonic() + AUTO_RECOVERY_NOTICE_TTL_SECONDS)
    return notice


def peek_auto_recovery_notice() -> dict | None:
    """The pending automatic-clear notice, or None once it has aged out."""
    global _auto_recovery_notice
    with _auto_recovery_notice_lock:
        entry = _auto_recovery_notice
        if entry is None:
            return None
        notice, expires_at = entry
        if time.monotonic() >= expires_at:
            _auto_recovery_notice = None
            return None
        return dict(notice)


def clear_auto_recovery_notice() -> None:
    """Drop the pending notice (tests, and an explicit user dismissal)."""
    global _auto_recovery_notice
    with _auto_recovery_notice_lock:
        _auto_recovery_notice = None


def _dispatch_auto_resolved_cancellation(job_id) -> None:
    """Settle the linked card of a job the machine just cancelled.

    Best effort by design: the barrier is already gone and the app already
    unblocked, so a service callback that fails must not turn into a failed
    recovery. `_dispatch_completion` refuses a row still marked `stalled`, which
    is the guard that keeps this from touching anything unresolved.
    """
    try:
        job = ImageGenerationQueue.query.filter_by(
            job_id=str(job_id), status='cancelled').first()
        if job is not None:
            _dispatch_completion(job, None, True)
    except Exception:
        logger.exception(
            'job_queue: could not settle the linked card of auto-resolved job %s', job_id)


# What the SAME probe learned about the link to ComfyUI, instead of throwing it
# away. `comfyui_prompt_is_absent` answers three things — True (a healthy /queue
# proved the id gone), False (ComfyUI answered and still lists it), None (no
# readable answer at all) — and only the first was ever acted on. The other two
# were flattened into one "no proof, keep the barrier", which is how a fresh
# install ended up being told a paused job was blocking it while its ComfyUI had
# never logged a single incoming connection: LDS was not talking to that ComfyUI
# at all (jerkyjunky, Discord). The barrier is right to stay; the sentence was
# wrong. So the verdict now travels out with the resolution.
COMFYUI_LINK_REACHABLE = 'reachable'
COMFYUI_LINK_UNREACHABLE = 'unreachable'


class ComfyBarrierProbe(NamedTuple):
    """(what the probe resolved, what it learned about the link).

    `link` is None when nothing was asked — no barrier at all, or a barrier with
    no prompt id to ask about. "Not asked" must never be read as "reachable".
    """
    resolved: dict | None = None
    link: str | None = None


def probe_comfyui_barrier() -> ComfyBarrierProbe:
    """Clear the recovery barrier when the remote prompt is PROVABLY gone, and
    report what ComfyUI answered while being asked.

    Never raises: a failed probe is simply an absence of proof.
    """
    owner = queue_manager.get_comfyui_stalled_barrier()
    if owner is None:
        # No barrier, or a corrupt one. A corrupt record is exactly the case
        # where a machine must not guess: it still blocks, and it needs eyes.
        return ComfyBarrierProbe()
    if owner.get('kind') != 'prompt' or not owner.get('prompt_id'):
        # An unconfirmed submission has no id to ask about, so this function
        # learns nothing about the link either. Whoever needs that verdict has
        # to get it elsewhere rather than infer it from this silence.
        return ComfyBarrierProbe()
    prompt_id = owner['prompt_id']
    link = None
    try:
        from .utils.comfyui import comfyui_prompt_is_absent
        # True *only* after a healthy /queue answer proved the id absent.
        # False = still pending/running; None = unreachable or unparseable.
        absent = comfyui_prompt_is_absent(prompt_id)
        # False means ComfyUI answered — that IS the reachability proof, free of
        # any extra request; None means LDS got nothing it could read.
        link = COMFYUI_LINK_UNREACHABLE if absent is None else COMFYUI_LINK_REACHABLE
        if absent is not True:
            return ComfyBarrierProbe(None, link)
        # reconcile_stalled_comfy_job re-verifies queue absence on both sides of
        # a /history read and refuses an unhealthy history. This is its only
        # caller that runs without anyone asking, so it borrows those checks
        # rather than repeating a weaker version of them.
        if not queue_manager.reconcile_stalled_comfy_job(owner['job_id']):
            # It can refuse for two very different reasons. Either the remote
            # state is not settled after all — leave it alone — or there is no
            # stalled row left to cancel, in which case the barrier is an orphan
            # that nothing will ever clear. The prompt was just proven absent
            # from ComfyUI, so the second case is safe to drop.
            if not queue_manager.discard_orphan_comfyui_barrier():
                return ComfyBarrierProbe(None, link)
        else:
            # Unblocking the app is not the whole job: the dataset card that
            # owns this generation is still drawn as "in progress". Route the
            # cancellation through the normal completion seam so the tile
            # settles too, instead of waiting for a Stop nobody knows to press.
            _dispatch_auto_resolved_cancellation(owner['job_id'])
    except Exception:
        logger.exception('job_queue: automatic ComfyUI recovery probe failed')
        return ComfyBarrierProbe(None, link)
    logger.warning(
        'job_queue: automatically cleared the ComfyUI recovery barrier for job %s — '
        'prompt %s is absent from both /queue and /history (ComfyUI was restarted)',
        owner.get('job_id'), prompt_id)
    _record_auto_recovery_notice(owner)
    return ComfyBarrierProbe(owner, link)


def auto_resolve_comfyui_barrier() -> dict | None:
    """The resolution half of `probe_comfyui_barrier`, for the callers that only
    ever needed "was anything cleared?" — the route guards and the Stop paths."""
    return probe_comfyui_barrier().resolved


# A DB status check is the source of truth, but this in-process event also wakes
# the worker immediately when it is sleeping between two history requests.
_poll_cancel_events: dict[str, threading.Event] = {}
_poll_cancel_events_lock = threading.Lock()


def _cancel_event(prompt_id) -> threading.Event:
    with _poll_cancel_events_lock:
        return _poll_cancel_events.setdefault(str(prompt_id), threading.Event())


def _signal_poll_cancel(prompt_id) -> None:
    if not prompt_id:
        return
    # Do not create an event for a prompt that has never entered _poll_outputs
    # (notably cancel-during-submit). Its cancelled DB status is sufficient, and
    # leaving a pre-signalled orphan behind can poison a later reused test id.
    with _poll_cancel_events_lock:
        event = _poll_cancel_events.get(str(prompt_id))
        if event is not None:
            event.set()


def _discard_cancel_event(prompt_id) -> None:
    with _poll_cancel_events_lock:
        _poll_cancel_events.pop(str(prompt_id), None)


def _vision_window_blocks_gpu() -> bool:
    """Fail closed while an in-process Vision/CUDA window is still active."""
    try:
        # Lazy import avoids gpu_window -> job_queue import recursion at startup.
        from .gpu_window import vision_gpu_window_blocks_gpu
        return vision_gpu_window_blocks_gpu()
    except Exception:
        logger.exception('job_queue: could not read the in-process Vision GPU fence')
        return True

def _claim(job_id) -> bool:
    """Atomically claim a pending job for processing. Returns False if the job
    was cancelled/claimed since the SELECT, preventing cancel-race loss."""
    with GPU_ARBITER_LOCK:
        # Recheck both durable and in-process GPU ownership inside the
        # select -> claim window.
        if _vision_window_blocks_gpu() or queue_manager.has_comfyui_stalled_barrier():
            return False
        claimed = (ImageGenerationQueue.query
                   .filter_by(job_id=job_id, status='pending')
                   .update({'status': 'processing',
                            'started_at': naive_utcnow(),
                            'last_heartbeat': naive_utcnow()}))
        db.session.commit()
        return bool(claimed)


def _submit(workflow, client_id):
    """Queue a workflow on ComfyUI, returning the ComfyUI prompt_id string.

    queue_prompt_to_comfyui never raises: it returns (response.json(), None) on
    success or (None, error) on failure. Unpack it here -- binding the raw tuple
    into the comfyui_prompt_id String column is a ProgrammingError that fails
    every real job. Deterministic local validation is terminal; every ambiguous
    `/prompt` outcome is handed to process_one() as a recovery barrier."""
    if not isinstance(workflow, dict) or not workflow:
        raise _ComfySubmitRejected('WORKFLOW_INVALIDE: workflow data must be a non-empty object')
    from .utils.comfyui import queue_prompt_to_comfyui
    result, error = queue_prompt_to_comfyui(workflow, client_id)
    if error:
        message = str(error)
        # COMFYUI_UNREACHABLE means the request provably never left this
        # machine (pre-POST refusal, connect timeout, connection never
        # established) — ComfyUI cannot own the prompt, so this is the
        # "unavailable local submit seam" _ComfySubmitRejected always
        # documented: a clean terminal fail the user simply retries once
        # ComfyUI is up, never the human-confirm recovery barrier (GitHub #51).
        if message.startswith(('WORKFLOW_INVALIDE', 'COMFYUI_UNREACHABLE')):
            raise _ComfySubmitRejected(message)
        # A READ timeout, a reset, malformed JSON, or any non-validation HTTP
        # response can happen after ComfyUI accepted the POST. Do not let a
        # caller collapse that unknown remote ownership into an ordinary fail.
        raise _ComfySubmitUnknown(message)
    prompt_id = (result or {}).get('prompt_id')
    if not prompt_id:
        raise _ComfySubmitUnknown(f'ComfyUI returned no prompt_id: {result}')
    return prompt_id


def _stall_comfyui_prompt(prompt_id, detail=None) -> bool:
    """Durably pause exactly the still-sent prompt; false means CAS lost."""
    with GPU_ARBITER_LOCK:
        job = (ImageGenerationQueue.query
               .filter_by(comfyui_prompt_id=str(prompt_id), status='sent_to_comfy').first())
        if job is None:
            return False
        return queue_manager._stall_comfy_job(
            job.job_id, str(prompt_id), allowed_statuses=('sent_to_comfy',), detail=detail)


def _pause_unconfirmed_comfyui_prompt(prompt_id, detail=None):
    """Never terminalize a prompt whose remote state is still unconfirmed.

    A false stall is not evidence of failure: it can be a transient SQLite
    commit error. The still-active queue row then remains the fail-closed
    owner until recovery can write its durable barrier. Only a committed local
    cancellation is proof that completion cleanup/callbacks are safe.
    """
    try:
        if _stall_comfyui_prompt(prompt_id, detail):
            return POLL_STALLED
        fresh_status = (ImageGenerationQueue.query
                        .with_entities(ImageGenerationQueue.status)
                        .filter_by(comfyui_prompt_id=prompt_id).scalar())
        if fresh_status == 'cancelled':
            return True
        if fresh_status in ('stalled', 'cancel_requested') or \
                queue_manager.has_comfyui_stalled_barrier():
            return POLL_STALLED
        logger.critical(
            'job_queue: could not durably pause unconfirmed ComfyUI prompt %s; '
            'leaving its active queue row fail-closed', prompt_id)
    except Exception:
        logger.exception('job_queue: could not inspect unconfirmed ComfyUI prompt %s', prompt_id)
    return POLL_STALLED


def _poll_outputs(prompt_id, timeout=POLL_TIMEOUT_SECONDS):
    """Poll one ComfyUI prompt without mistaking an outage for an empty history.

    Returns (filename, failed) for normal terminal outcomes, or
    (None, POLL_STALLED) when history is unhealthy long enough or no terminal
    state exists at timeout. The latter deliberately leaves the queue/cell
    non-terminal so the user can recover ComfyUI, cancel the exact old prompt,
    then resume.
    """
    from .utils.comfyui import ComfyHistoryHealth, get_comfyui_history_probe

    deadline = time.monotonic() + timeout
    unhealthy_since = None
    cancel_event = _cancel_event(prompt_id)
    try:
        while True:
            job_status = (ImageGenerationQueue.query
                          .with_entities(ImageGenerationQueue.status)
                          .filter_by(comfyui_prompt_id=prompt_id).scalar())
            if cancel_event.is_set() or job_status == 'cancelled':
                return None, True
            if job_status in ('stalled', 'cancel_requested'):
                return None, POLL_STALLED

            probe = get_comfyui_history_probe(prompt_id)
            if probe.health is ComfyHistoryHealth.UNHEALTHY:
                now = time.monotonic()
                if unhealthy_since is None:
                    unhealthy_since = now
                # A shorter test/override timeout does not make an unhealthy
                # history trustworthy. Either threshold means the remote state
                # is unconfirmed and must be durably paused, never failed.
                if (now - unhealthy_since >= COMFYUI_UNHEALTHY_GRACE_SECONDS
                        or now >= deadline):
                    return None, _pause_unconfirmed_comfyui_prompt(
                        prompt_id, probe.detail or 'ComfyUI history unhealthy')
                cancel_event.wait(min(POLL_INTERVAL_SECONDS, max(0, deadline - now)))
                continue

            # A 404 / empty object is a healthy worker that has not recorded this
            # prompt yet. It resets the consecutive outage timer.
            unhealthy_since = None
            history = probe.history or {}
            entry = history.get(prompt_id, history) if isinstance(history, dict) else {}
            outputs = (entry or {}).get('outputs') or {}
            for node_output in outputs.values():
                for img in (node_output or {}).get('images') or []:
                    if isinstance(img, dict) and img.get('filename') and img.get('type', 'output') != 'temp':
                        return img['filename'], False
            status = (entry or {}).get('status') or {}
            if status.get('status_str') == 'error' or (status.get('completed') and not outputs):
                detail = _execution_error_detail(status)
                if detail:
                    job = ImageGenerationQueue.query.filter_by(comfyui_prompt_id=prompt_id).first()
                    if job:
                        job.error_message = detail
                        db.session.commit()
                return None, True

            job = ImageGenerationQueue.query.filter_by(comfyui_prompt_id=prompt_id).first()
            if job:
                if job.status == 'cancelled' or cancel_event.is_set():
                    return None, True
                if job.status in ('stalled', 'cancel_requested'):
                    return None, POLL_STALLED
                job.last_heartbeat = naive_utcnow()
                db.session.commit()

            if time.monotonic() >= deadline:
                return None, _pause_unconfirmed_comfyui_prompt(
                    prompt_id, 'ComfyUI did not reach a terminal history state before timeout')
            cancel_event.wait(POLL_INTERVAL_SECONDS)
    finally:
        _discard_cancel_event(prompt_id)


def _execution_error_detail(status) -> str | None:
    """One human-readable line out of a ComfyUI history `status` block: the
    execution_error message (+ failing node) that explains WHY a run died.
    Truncated — some tracebacks embed whole tensors."""
    try:
        for m in status.get('messages') or []:
            if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == 'execution_error':
                info = m[1] or {}
                node = info.get('node_type') or info.get('node_id') or '?'
                exc = ' '.join(str(info.get('exception_message') or '').split())
                if exc:
                    return f'ComfyUI {node}: {exc}'[:400]
    except Exception:  # malformed history must never break the poll loop
        pass
    return None


# Every LOCAL engine that renders a dataset variation links its result back
# through the same callback, so this is a SET rather than one hardcoded name.
# It is a set because of how it broke: Krea 2 Edit shipped stamping its own
# `krea_identity_edit_dataset`, the dispatch below still tested only Klein's
# name, and twelve images were generated, paid for in GPU time, marked done in
# the queue — and never attached to their rows. The tile stayed at 0/12 forever
# with nothing in the logs, because nothing had failed. A new engine must be
# added HERE, and the contract test that walks this set is what says so.
DATASET_IMAGE_JOB_NAMES = frozenset({
    'klein_edit_dataset',           # Klein (FLUX.2)
    'krea_identity_edit_dataset',   # Krea 2 Identity Edit
    'seedvr2_upscale',              # SeedVR2 (fidelity upscale)
    'qwen_camera_dataset',          # 📷 Camera angles, dataset lane
})
# It happened a SECOND time, with SeedVR2, for the same reason and with the same
# clean logs: rendered, `execution_success`, 2.2 MB PNG on disk, candidate row
# still pending with a NULL filename. The guard written after Krea promised to
# catch exactly that and could not — it walked a hardcoded tuple of two helper
# modules, so a third helper was invisible to it. `test_dataset_job_harvest.py`
# now DISCOVERS the engines (AST over app/services) and forces every stamped job
# name to be classified, and `_harvest_unlinked_completed_jobs` below repairs the
# rows a future miss would strand instead of requiring hand-written SQL.


def _drop_staged_inputs(md) -> None:
    """Delete the ComfyUI input copies this job staged, now that it is over.

    Called from `_dispatch_completion`, i.e. on EVERY terminal outcome — success,
    failure, cancel, boot recovery of a stale row — because the copy is dead in
    all four cases and only the success path was ever going to be tempting to
    special-case. The staging side (`klein_edit_helper` / `krea_edit_helper`)
    records the basenames under `staged_inputs`; a job without them (an older row
    queued before this shipped, a Studio grid cell) is simply skipped.

    Guarded end to end: leaving a stale copy behind is a wasted gigabyte, but
    letting a filesystem hiccup escape here would strand the row that this same
    function is on its way to complete.
    """
    names = md.get('staged_inputs') if isinstance(md, dict) else None
    if not names:
        return
    try:
        from . import config as cfg
        from .utils import comfy_fs
        comfy_fs.drop_staged_inputs(names, cfg.comfyui_dir('input'))
    except Exception:
        logger.exception('job_queue: staged input cleanup failed')


def _dispatch_completion(job, filename, failed):
    """Route a finished job to whichever service created it, per its metadata.
    A callback crash must never take down the worker thread."""
    if job.status == 'stalled':
        # It may still own the GPU or yield a late output: preserve all linked state.
        logger.warning('job_queue: suppressing completion callback for stalled job %s', job.job_id)
        return
    try:
        md = json.loads(job.job_metadata or '{}')
    except (TypeError, ValueError):
        md = {}
    _drop_staged_inputs(md)
    try:
        if md.get('is_lora_test'):
            from .services import lora_test_studio
            # Pass the real failure reason (ComfyUI 400 body / node error / timeout)
            # so the failed grid tile can say WHY. The generic 'generation failed'
            # is LESS useful than the tile's own default → only forward real detail.
            reason = job.error_message if job.error_message != 'generation failed' else None
            lora_test_studio.link_completed_test_image(job.job_id, filename,
                                                       failed=failed, reason=reason)
        elif md.get('is_reference_edit'):
            # A LOCAL ✦ Edit-reference render (Klein / Krea 2 Edit). Checked
            # BEFORE the model_name branch below: it rides the very same
            # enqueue_*_edit helpers, so it carries their model_name — but it has
            # no FaceDatasetImage row, and link_completed_dataset_image would find
            # nothing and log a bogus "no row for job".
            from .services import face_dataset_service
            reason = job.error_message if job.error_message != 'generation failed' else None
            face_dataset_service.link_completed_reference_edit(
                job.job_id, filename, failed=failed, reason=reason)
        elif md.get('is_video_test'):
            # A Video Test Studio clip. Its own branch rather than a model_name
            # match: the mp4 arrives under the history's `images` key like any
            # image does, so nothing upstream distinguishes it — only the
            # metadata this lane wrote can.
            from .services import video_test_studio
            reason = job.error_message if job.error_message != 'generation failed' else None
            video_test_studio.link_completed_clip(job.job_id, filename,
                                                  failed=failed, reason=reason)
        elif md.get('is_bank_improve'):
            # A Bank ✨ Upscale & improve. It rides the very same enqueue helpers
            # as the dataset lane, so it necessarily carries their model_name —
            # and the branch below would look for a FaceDatasetImage that does
            # not exist. There is deliberately nothing to link: the bank pass
            # POLLS its own queue row and writes the blob itself (same contract
            # as watermark_klein), because the progress and the Stop button for
            # the whole pass live in bank_jobs, not in one row per image.
            logger.debug('job_queue: bank improve %s finished (failed=%s) — the '
                         'bank pass owns its own result', job.job_id, failed)
        elif md.get('model_name') in DATASET_IMAGE_JOB_NAMES:
            from .services import face_dataset_service
            # The bare fallback 'generation failed' is LESS useful than the tile's
            # own default (which points at the server log) — only pass real detail.
            reason = job.error_message if job.error_message != 'generation failed' else None
            face_dataset_service.link_completed_dataset_image(
                job.job_id, filename, failed=failed, reason=reason)
    except Exception:
        logger.exception('job_queue: completion dispatch failed for job %s', job.job_id)
        # The link callback crashed before flipping its row out of 'pending' -
        # without this it strands the row looking like it's still generating.
        try:
            from .models import FaceDatasetImage, LoraTestImage, VideoTestClip
            for model in (FaceDatasetImage, LoraTestImage, VideoTestClip):
                row = model.query.filter_by(job_id=job.job_id).first()
                if row is not None:
                    row.status = 'failed'
                    db.session.commit()
        except Exception:
            logger.exception('job_queue: could not mark linked row failed for job %s', job.job_id)


class JobQueueManager:
    """Singleton worker: one background thread picks the oldest pending image
    job, submits it to ComfyUI, polls for its output, and dispatches
    completion — all synchronously per job via `process_one()`."""

    def __init__(self):
        self._app = None
        self._thread = None
        self._running = False

    def init_app(self, app):
        self._app = app

    # -- durable ComfyUI recovery barrier ---------------------------------
    @staticmethod
    def _barrier_owner(job, prompt_id=None, detail=None, *, unknown_submit=False) -> dict | None:
        """Encode a known prompt or an explicitly unknown submit outcome.

        `unknown_submit` is intentionally distinguishable from a corrupt
        barrier: the POST may have succeeded before this process died. Its
        durable client_id is audit evidence, but no local queue observation can
        prove an in-flight POST will not arrive later; restart is required.
        """
        if not job.job_id or (unknown_submit and prompt_id is not None) or \
                (not unknown_submit and not prompt_id):
            return None
        try:
            metadata = json.loads(job.job_metadata or '{}')
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        owner = {
            'job_id': str(job.job_id),
            'prompt_id': None if unknown_submit else str(prompt_id),
            # _submit always sends the LDS queue id as ComfyUI client_id.
            'client_id': str(job.job_id),
            'kind': 'unknown_submit' if unknown_submit else 'prompt',
            'reason': (COMFYUI_UNKNOWN_SUBMIT_MESSAGE if unknown_submit
                       else COMFYUI_STALLED_MESSAGE),
        }
        for key in ('dataset_id', 'cell_id', 'run_id'):
            value = metadata.get(key)
            if value is not None and str(value):
                owner[key] = str(value)
        if detail:
            owner['detail'] = str(detail)[:200]
        return owner

    @staticmethod
    def _same_barrier_owner(left, right) -> bool:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        left_kind = left.get('kind', 'prompt')
        right_kind = right.get('kind', 'prompt')
        if left_kind not in ('prompt', 'unknown_submit') or left_kind != right_kind:
            return False
        if any(left.get(key) != right.get(key) for key in ('job_id', 'client_id')):
            return False
        if left_kind == 'prompt':
            if left.get('prompt_id') != right.get('prompt_id'):
                return False
        elif left.get('prompt_id') is not None or right.get('prompt_id') is not None:
            return False
        return all(left.get(key) == right.get(key)
                   for key in ('dataset_id', 'cell_id', 'run_id')
                   if key in left or key in right)

    @staticmethod
    def _encode_comfyui_stalled_barrier(owner) -> str:
        return json.dumps({'v': owner, 'exp': None},
                          sort_keys=True, separators=(',', ':'))

    def _read_comfyui_stalled_barrier(self):
        """Return (row, raw_value, owner, valid); raw presence always blocks."""
        row = db.session.get(SystemState, COMFYUI_STALLED_BARRIER_KEY)
        if row is None:
            return None, None, None, True
        raw = row.value
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return row, raw, None, False
        owner = payload.get('v') if isinstance(payload, dict) else None
        if (not isinstance(payload, dict) or payload.get('exp') is not None
                or not isinstance(owner, dict)):
            return row, raw, None, False
        for key in ('job_id', 'client_id'):
            if not isinstance(owner.get(key), str) or not owner[key]:
                return row, raw, None, False
        kind = owner.get('kind', 'prompt')
        if kind == 'prompt':
            if not isinstance(owner.get('prompt_id'), str) or not owner['prompt_id']:
                return row, raw, None, False
        elif kind == 'unknown_submit':
            if 'prompt_id' not in owner or owner.get('prompt_id') is not None:
                return row, raw, None, False
        else:
            return row, raw, None, False
        for key in ('dataset_id', 'cell_id', 'run_id'):
            if key in owner and (not isinstance(owner[key], str) or not owner[key]):
                return row, raw, None, False
        return row, raw, owner, True

    def has_comfyui_stalled_barrier(self) -> bool:
        # Do not route this through _get_system_state: a corrupt/TTL row blocks.
        with GPU_ARBITER_LOCK:
            return db.session.get(SystemState, COMFYUI_STALLED_BARRIER_KEY) is not None

    def get_comfyui_stalled_barrier(self) -> dict | None:
        with GPU_ARBITER_LOCK:
            _, _, owner, valid = self._read_comfyui_stalled_barrier()
            return dict(owner) if valid and owner is not None else None

    def _stall_comfy_job(self, job_id, prompt_id, *, allowed_statuses, detail=None) -> bool:
        """Persist exact `status -> stalled` and its no-TTL barrier atomically."""
        if not job_id or not prompt_id:
            return False
        with GPU_ARBITER_LOCK:
            job = (ImageGenerationQueue.query.filter_by(job_id=str(job_id))
                   .filter(ImageGenerationQueue.status.in_(tuple(allowed_statuses))).first())
            if job is None or job.comfyui_prompt_id not in (None, str(prompt_id)):
                return False
            owner = self._barrier_owner(job, prompt_id, detail)
            if owner is None:
                return False
            row, _, _, _ = self._read_comfyui_stalled_barrier()
            if row is not None:  # valid, corrupt, expired, or another owner: all block
                return False

            cas = (ImageGenerationQueue.query.filter_by(job_id=str(job_id))
                   .filter(ImageGenerationQueue.status.in_(tuple(allowed_statuses))))
            # SQL `IN (NULL, prompt_id)` never matches NULL: keep both CAS paths explicit.
            if job.comfyui_prompt_id is None:
                cas = cas.filter(ImageGenerationQueue.comfyui_prompt_id.is_(None))
            else:
                cas = cas.filter_by(comfyui_prompt_id=str(prompt_id))
            changed = cas.update({
                'status': 'stalled',
                'comfyui_prompt_id': str(prompt_id),
                'completed_at': None,
                'error_message': COMFYUI_STALLED_MESSAGE,
                'last_heartbeat': naive_utcnow(),
            }, synchronize_session=False)
            if changed != 1:
                return False
            db.session.add(SystemState(
                key=COMFYUI_STALLED_BARRIER_KEY,
                value=self._encode_comfyui_stalled_barrier(owner),
            ))
            try:
                db.session.commit()  # queue row + ownership record: one transaction
            except Exception:
                db.session.rollback()
                logger.exception('job_queue: could not persist ComfyUI stalled barrier')
                return False
            return True

    def _stall_unknown_comfy_job(self, job_id, *, allowed_statuses, detail=None) -> bool:
        """Persist a recoverable barrier when `/prompt` may have succeeded
        but no prompt id was durably observed.

        The client_id is retained as audit identity. Because a timed-out POST
        can still arrive after any later queue read, this state requires an
        externally verified ComfyUI restart; it is never auto-reconciled.
        """
        if not job_id:
            return False
        with GPU_ARBITER_LOCK:
            job = (ImageGenerationQueue.query.filter_by(job_id=str(job_id))
                   .filter(ImageGenerationQueue.status.in_(tuple(allowed_statuses))).first())
            if job is None or job.comfyui_prompt_id is not None:
                return False
            owner = self._barrier_owner(job, detail=detail, unknown_submit=True)
            if owner is None:
                return False
            row, _, _, _ = self._read_comfyui_stalled_barrier()
            if row is not None:
                return False
            changed = (ImageGenerationQueue.query.filter_by(job_id=str(job_id))
                       .filter(ImageGenerationQueue.status.in_(tuple(allowed_statuses)))
                       .filter(ImageGenerationQueue.comfyui_prompt_id.is_(None))
                       .update({'status': 'stalled', 'completed_at': None,
                                'error_message': COMFYUI_UNKNOWN_SUBMIT_MESSAGE,
                                'last_heartbeat': naive_utcnow()},
                               synchronize_session=False))
            if changed != 1:
                return False
            db.session.add(SystemState(
                key=COMFYUI_STALLED_BARRIER_KEY,
                value=self._encode_comfyui_stalled_barrier(owner),
            ))
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                logger.exception('job_queue: could not persist unknown ComfyUI submit barrier')
                return False
            return True

    def reconcile_stalled_comfy_job(self, job_id) -> bool:
        """Release only the exact stalled prompt after fresh remote proof."""
        from .utils.comfyui import (ComfyHistoryHealth, ComfyPromptState,
                                    cancel_comfyui_prompt_state,
                                    comfyui_prompt_is_absent,
                                    get_comfyui_history_probe)
        with GPU_ARBITER_LOCK:
            _, raw, owner, valid = self._read_comfyui_stalled_barrier()
            if not valid or owner is None or str(owner.get('job_id')) != str(job_id):
                return False
            job = ImageGenerationQueue.query.filter_by(job_id=str(job_id), status='stalled').first()
            if job is None or str(job.comfyui_prompt_id or '') != owner['prompt_id']:
                return False
            if not self._same_barrier_owner(
                    owner, self._barrier_owner(job, owner['prompt_id'])):
                return False

            try:
                state = cancel_comfyui_prompt_state(owner['prompt_id'], owner['client_id'])
                if state is ComfyPromptState.DELETED:
                    safe = comfyui_prompt_is_absent(owner['prompt_id']) is True
                elif state is ComfyPromptState.ABSENT:
                    # Empty/404 history is not enough. Verify queue absence on both
                    # sides of the history read, then accept only a terminal history.
                    if comfyui_prompt_is_absent(owner['prompt_id']) is not True:
                        return False
                    probe = get_comfyui_history_probe(owner['prompt_id'])
                    if probe.health is ComfyHistoryHealth.UNHEALTHY:
                        return False
                    if probe.health is ComfyHistoryHealth.READY:
                        entry = (probe.history.get(owner['prompt_id'])
                                 if isinstance(probe.history, dict) else None)
                        status = entry.get('status') if isinstance(entry, dict) else None
                        if not isinstance(status, dict) or not (
                                status.get('completed')
                                or status.get('status_str') in ('success', 'error')):
                            return False
                    safe = comfyui_prompt_is_absent(owner['prompt_id']) is True
                else:
                    return False
            except Exception:
                logger.exception('job_queue: stalled prompt %s reconciliation failed', owner['prompt_id'])
                return False
            if not safe:
                return False

            _, current_raw, current_owner, current_valid = self._read_comfyui_stalled_barrier()
            if (not current_valid or current_raw != raw
                    or not self._same_barrier_owner(current_owner, owner)):
                return False
            now = naive_utcnow()
            changed = (ImageGenerationQueue.query
                       .filter_by(job_id=str(job_id), status='stalled',
                                  comfyui_prompt_id=owner['prompt_id'])
                       .update({'status': 'cancelled', 'completed_at': now,
                                'last_heartbeat': now, 'error_message': None},
                               synchronize_session=False))
            if changed != 1:
                return False
            deleted = (SystemState.query
                       .filter_by(key=COMFYUI_STALLED_BARRIER_KEY, value=raw)
                       .delete(synchronize_session=False))
            if deleted != 1:
                db.session.rollback()
                return False
            try:
                db.session.commit()  # old job cancellation + exact raw barrier delete
            except Exception:
                db.session.rollback()
                logger.exception('job_queue: could not finish stalled reconciliation')
                return False
            _signal_poll_cancel(owner['prompt_id'])
            return True

    def confirm_unknown_comfyui_restart(self, job_id, user_id=None, *,
                                        restart_confirmed=False, commit=True) -> bool:
        """Cancel one unknown ``/prompt`` outcome after explicit restart authority.

        An empty queue/history cannot prove that a timed-out POST never reaches the
        old ComfyUI process. The caller must therefore require a human's explicit
        confirmation that ComfyUI was restarted *and* verify that the replacement
        process is reachable before calling this method. This method deliberately
        performs no remote queue/history inference and never touches a known-prompt
        barrier.

        ``commit=False`` lets a linked Test Studio cell join the exact queue-row +
        raw-barrier delete in one database transaction.
        """
        if not restart_confirmed or not job_id:
            return False
        with GPU_ARBITER_LOCK:
            _, raw, owner, valid = self._read_comfyui_stalled_barrier()
            if (not valid or owner is None or owner.get('kind') != 'unknown_submit'
                    or str(owner.get('job_id')) != str(job_id)
                    or owner.get('prompt_id') is not None):
                return False

            query = (ImageGenerationQueue.query.filter_by(job_id=str(job_id), status='stalled')
                     .filter(ImageGenerationQueue.comfyui_prompt_id.is_(None)))
            if user_id is not None:
                query = query.filter_by(user_id=str(user_id))
            job = query.first()
            if job is None or not self._same_barrier_owner(
                    owner, self._barrier_owner(job, unknown_submit=True)):
                return False

            # Keep the compare-and-delete exact even though the in-process arbiter
            # holds local callers out: another process or a crash recovery must not
            # let this confirmation clear a replacement/corrupt barrier.
            _, current_raw, current_owner, current_valid = self._read_comfyui_stalled_barrier()
            if (not current_valid or current_raw != raw
                    or not self._same_barrier_owner(current_owner, owner)):
                return False

            now = naive_utcnow()
            changed = (ImageGenerationQueue.query
                       .filter_by(job_id=str(job_id), status='stalled')
                       .filter(ImageGenerationQueue.comfyui_prompt_id.is_(None)))
            if user_id is not None:
                changed = changed.filter_by(user_id=str(user_id))
            changed = changed.update({
                'status': 'cancelled', 'completed_at': now,
                'last_heartbeat': now, 'error_message': None,
            }, synchronize_session=False)
            if changed != 1:
                return False
            deleted = (SystemState.query
                       .filter_by(key=COMFYUI_STALLED_BARRIER_KEY, value=raw)
                       .delete(synchronize_session=False))
            if deleted != 1:
                db.session.rollback()
                return False
            if not commit:
                return True
            try:
                db.session.commit()  # exact queue job + exact raw barrier
            except Exception:
                db.session.rollback()
                logger.exception('job_queue: could not confirm unknown ComfyUI restart')
                return False
            return True

    def discard_orphan_comfyui_barrier(self) -> bool:
        """Delete a barrier that no longer guards anything.

        The barrier's whole job is to keep new work out while a queue row still
        claims uncertain remote ownership. If that row is gone — deleted by a
        cascade, or already finalized — the barrier guards nothing: no code path
        can ever reconcile it (every resolver matches on a `stalled` row), so it
        blocks every generation in the app FOREVER and only a hand-written SQL
        DELETE can lift it. That is not a safety property, it is a dead end.

        Deliberately not a general escape hatch: callers must first establish
        that the remote side is settled (proof for a known prompt, a confirmed
        restart for an unknown submit). This only handles "there is nothing left
        to cancel locally".
        """
        with GPU_ARBITER_LOCK:
            _, raw, owner, valid = self._read_comfyui_stalled_barrier()
            if not valid or owner is None:
                return False
            still_owned = (ImageGenerationQueue.query
                           .filter_by(job_id=str(owner['job_id']))
                           .filter(ImageGenerationQueue.status.in_(
                               ('pending', 'processing', 'sent_to_comfy',
                                'cancel_requested', 'stalled')))
                           .first())
            if still_owned is not None:
                return False
            deleted = (SystemState.query
                       .filter_by(key=COMFYUI_STALLED_BARRIER_KEY, value=raw)
                       .delete(synchronize_session=False))
            if deleted != 1:
                db.session.rollback()
                return False
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                logger.exception('job_queue: could not discard the orphan ComfyUI barrier')
                return False
            logger.warning(
                'job_queue: discarded the ComfyUI recovery barrier for job %s — its '
                'queue row no longer exists, so nothing was left to reconcile',
                owner.get('job_id'))
            return True

    def has_comfyui_work(self) -> bool:
        """True while LDS has work queued or an unresolved ComfyUI identity."""
        return (ImageGenerationQueue.query
                .filter(ImageGenerationQueue.status.in_(
                    ('pending', 'processing', 'sent_to_comfy', 'cancel_requested', 'stalled')))
                .first() is not None)

    # -- lifecycle ------------------------------------------------------
    def start(self):
        """Idempotent: a no-op if the worker thread is already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        with self._app.app_context():
            self._recover_stuck_jobs()
            # AFTER recovery on purpose: that pass moves every uncertain job to
            # `stalled`, and _dispatch_completion refuses to route a stalled job.
            # So the harvest can only ever see rows whose outcome is settled.
            self._harvest_unlinked_completed_jobs()
            self._prune_staged_inputs()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name='job-queue-worker', daemon=True)
        self._thread.start()

    def stop(self, timeout=5):
        """For tests: stop the loop and wait for the thread to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run_loop(self):
        while self._running:
            try:
                with self._app.app_context():
                    worked = self.process_one()
            except Exception:
                logger.exception('job_queue: worker loop error')
                worked = False
            time.sleep(0 if worked else IDLE_SLEEP_SECONDS)

    def _recover_stuck_jobs(self):
        """Recover only with a durable ownership record; never guess a crash
        left remote ComfyUI work dead.

        A crash after `/prompt` accepts but before its id commits leaves a
        `processing` row with no prompt id. It remains a recoverable
        `unknown_submit` barrier. A mapped row becomes an exact prompt barrier.
        In both cases callbacks and staged-input deletion stay suppressed.
        """
        with GPU_ARBITER_LOCK:
            active = (ImageGenerationQueue.query
                      .filter(ImageGenerationQueue.status.in_(
                          ('processing', 'sent_to_comfy', 'cancel_requested'))).all())
            for job in active:
                # A new worker after process restart has no trustworthy in-memory
                # poller or submit state. Even a just-claimed row may be past an
                # accepted `/prompt`; persist its ownership before doing anything
                # else rather than relying on the stale-age heuristic.
                if job.comfyui_prompt_id:
                    paused = self._stall_comfy_job(
                        job.job_id, job.comfyui_prompt_id,
                        allowed_statuses=(job.status,),
                        detail='startup recovery: remote ComfyUI state is unconfirmed')
                else:
                    paused = self._stall_unknown_comfy_job(
                        job.job_id, allowed_statuses=(job.status,),
                        detail='startup recovery: /prompt outcome was not durably mapped')
                if not paused:
                    # A pre-existing/corrupt barrier or failed commit is itself
                    # fail-closed. Keep this active row intact for explicit repair.
                    logger.critical(
                        'job_queue: startup could not durably pause uncertain job %s; '
                        'leaving it active and blocking new GPU work', job.job_id)

    def _harvest_unlinked_completed_jobs(self):
        """Attach results that FINISHED but were never linked to their row.

        The net under the routing table. Twice now an engine has shipped
        stamping a job name `_dispatch_completion` did not know: the image
        rendered, the queue row went `completed` with its filename, and the
        dataset row stayed `pending` with a NULL filename — no error, no log,
        nothing on screen. Fixing the table repairs the NEXT run; it does not
        give anyone back the images they already paid GPU time for, and the only
        remedy was hand-written SQL. This is that remedy, run automatically at
        boot, so an "Update & restart" is the whole fix.

        Driven from the ROWS, not from the queue: candidates are dataset images
        still `pending` with no file and a job id — naturally a handful, even on
        an install with tens of thousands of finished jobs. A row is only
        re-dispatched when its queue row is genuinely TERMINAL; anything still in
        flight (or paused by the recovery above, which runs first) is left alone.

        Guarded end to end: this runs before the worker starts, and a boot that
        cannot repair must still boot.
        """
        try:
            from .models import FaceDatasetImage
            stranded = (FaceDatasetImage.query
                        .filter(FaceDatasetImage.status == 'pending',
                                FaceDatasetImage.filename.is_(None),
                                FaceDatasetImage.job_id.isnot(None))
                        .all())
            if not stranded:
                return
            repaired = 0
            for row in stranded:
                job = ImageGenerationQueue.query.filter_by(job_id=row.job_id).first()
                if job is None or job.status not in ('completed', 'failed'):
                    continue          # never finished, or still owed a real dispatch
                try:
                    _dispatch_completion(job, job.result_filename,
                                         job.status == 'failed')
                    repaired += 1
                except Exception:
                    logger.exception('job_queue: harvest failed for job %s', job.job_id)
            if repaired:
                logger.info('job_queue: harvested %d finished job(s) whose result had '
                            'never been linked to its row', repaired)
        except Exception:
            logger.exception('job_queue: unlinked-result harvest failed')

    def _prune_staged_inputs(self):
        """Boot sweep for staged input copies no live job can still need.

        The per-job deletion above keeps the steady state clean, but it only
        started existing now: installs carry whatever they accumulated before
        (0.67 GB over three months on the install this was measured on), and a
        process killed mid-job leaves its copy behind whatever we do. Boot is the
        one moment when nothing is in flight, so an age-bounded sweep here is
        both safe and enough — no thread, no timer, no hot-path cost.

        The folder belongs to ComfyUI, so the sweep is fenced by NAME and by AGE
        (see comfy_fs) and, here, by the queue itself: every input a job that has
        not finished still points at is collected first and handed over as
        untouchable. Boot recovery has already preserved uncertain rows by the time
        this runs, so what is left really is work that will still be done."""
        try:
            from . import config as cfg
            from .utils import comfy_fs
            keep = set()
            for row in (ImageGenerationQueue.query
                        .filter(ImageGenerationQueue.status.in_(
                            ('pending', 'processing', 'sent_to_comfy', 'cancel_requested', 'stalled'))).all()):
                try:
                    md = json.loads(row.job_metadata or '{}')
                except (TypeError, ValueError):
                    continue
                keep.update(md.get('staged_inputs') or ())
            comfy_fs.prune_staged_inputs(cfg.comfyui_dir('input'), keep=keep)
        except Exception:
            logger.exception('job_queue: staged input prune failed')

    # -- worker -----------------------------------------------------------
    def gpu_hold(self):
        """WHICH hold is keeping the worker off the GPU, as a stable key, or None.

        A key rather than a sentence because the two surfaces that ask word it
        differently and both are right: the improve drain says "waiting for X"
        inside a progress line, the queue dock writes a full sentence with a
        remedy. Sharing one string would have forced one of them to read badly —
        and picking a sentence written for a THIRD screen is how the dock ended
        up telling someone in the dataset workspace that "the studio is
        unavailable".

        The four conditions `process_one` checks before it looks at a single row,
        minus the fifth — 'a job is already running' — which is not a hold but the
        serialization working as designed. The one that matters most, the
        in-process vision window, is not visible in the DB flags at all: a caller
        that re-derived this from `_get_system_state` alone would miss exactly the
        case where the heartbeat lost ownership and the barrier outlived the flag.

        It exists because a caller outside this module cannot tell a paused queue
        from a slow one, and one that guessed got it wrong: the bulk improve drain
        counted every poll of a queue frozen by a training run against its own
        15-minute stall timeout, then declared itself stalled and silently dropped
        the rest of the batch. Waiting on a queue that is provably held is not a
        stall, it is waiting.
        """
        if _vision_window_blocks_gpu():
            return HOLD_VISION
        if self._get_system_state('training_in_progress', False):
            return HOLD_TRAINING
        if self._get_system_state('vision_in_progress', False):
            return HOLD_VISION
        if self.has_comfyui_stalled_barrier():
            return HOLD_COMFYUI_RECOVERY
        # The Ollama release fence refuses AFTER a row is claimed, so it is not
        # one of the pre-claim gates above — but to the user it is the same
        # frozen queue. The fence publishes its last refusal (freshness-bounded,
        # re-asserted every worker cycle while jobs wait) exactly so this reader
        # can name it; before this, a fence refusal was the ONE hold with no
        # explanation anywhere in the app (found via a KoboldCPP endpoint that
        # answers /api/ps but can never unload).
        try:
            from .services.ollama_gpu_fence import last_block
            if last_block() is not None:
                return HOLD_OLLAMA_FENCE
        except Exception:
            logger.exception('job_queue: could not read the Ollama fence refusal state')
        return None

    def held_off_gpu_reason(self):
        """The same answer as a NOUN PHRASE, for a caller writing "waiting for {x}"."""
        return HOLD_LABELS.get(self.gpu_hold())

    def process_one(self) -> bool:
        """Run one queued image while closing the local ComfyUI/vision race."""
        job = None
        prompt_id = None
        submit_error = None
        dispatch_cancelled = False

        with GPU_ARBITER_LOCK:
            # Any unresolved active row is itself a fail-closed GPU owner after a
            # crash/mapping error; do not silently start a second prompt.
            if (_vision_window_blocks_gpu()
                    or self._get_system_state('training_in_progress')
                    or self._get_system_state('vision_in_progress')
                    or self.has_comfyui_stalled_barrier()
                    or ImageGenerationQueue.query.filter(
                        ImageGenerationQueue.status.in_(
                            ('processing', 'sent_to_comfy', 'cancel_requested', 'stalled'))
                    ).first() is not None):
                return False

            job = (ImageGenerationQueue.query.filter_by(status='pending')
                   .order_by(ImageGenerationQueue.priority.desc(),
                             ImageGenerationQueue.created_at.asc()).first())
            if job is None:
                return False
            if not _claim(job.job_id):
                return not self.has_comfyui_stalled_barrier()
            db.session.refresh(job)

            try:
                from .services.vision_keepalive import ensure_released_for_comfy
                released_for_comfy = ensure_released_for_comfy()
            except Exception:
                # An unreadable local lease/fence is not permission to overlap it.
                logger.exception('job_queue: could not verify vision lease release')
                released_for_comfy = False
            if not released_for_comfy:
                # Give back only our fresh claim. It remains eligible once the
                # strict Ollama unload fence confirms release; never terminalize it.
                (ImageGenerationQueue.query
                 .filter_by(job_id=job.job_id, status='processing')
                 .update({'status': 'pending', 'started_at': None,
                          'last_heartbeat': None}))
                try:
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                    logger.exception('job_queue: could not defer %s after vision lease fence',
                                     job.job_id)
                return False

            try:
                workflow = json.loads(job.workflow_data or '{}')
            except (TypeError, ValueError) as exc:
                # Parsing happens before `_submit`, so this is a local,
                # deterministic refusal with no possible remote prompt.
                submit_error = str(exc)[:400]
                logger.warning('job_queue: invalid workflow for %s: %s', job.job_id, exc)
            else:
                try:
                    prompt_id = _submit(workflow, job.job_id)
                    mapped = (ImageGenerationQueue.query
                              .filter_by(job_id=job.job_id, status='processing')
                              .update({'status': 'sent_to_comfy',
                                       'comfyui_prompt_id': prompt_id}))
                    # `mapped` becomes true only after this commit succeeds. A
                    # response from /prompt without this durable mapping is uncertain.
                    db.session.commit()
                except Exception as exc:
                    db.session.rollback()
                    if prompt_id:
                        paused = self._stall_comfy_job(
                            job.job_id, prompt_id,
                            allowed_statuses=('processing', 'cancel_requested',
                                              'cancelled', 'sent_to_comfy'),
                            detail=f'prompt mapping failed: {exc}')
                        if not paused:
                            db.session.refresh(job)
                            if job.status == 'cancelled':
                                dispatch_cancelled = True
                            else:
                                logger.critical(
                                    'job_queue: returned prompt %s has no durable mapping; '
                                    'leaving job %s active and fail-closed', prompt_id, job.job_id)
                        if not dispatch_cancelled:
                            return True  # never terminal-fail/callback an uncertain prompt
                    elif isinstance(exc, _ComfySubmitRejected):
                        # These failures happen before an HTTP accept (explicit
                        # validation refusal or an unavailable local submit seam).
                        submit_error = str(exc)[:400]
                        logger.warning('job_queue: job %s was deterministically refused: %s',
                                       job.job_id, exc)
                    else:
                        # Any other /prompt exception can mean ComfyUI accepted
                        # the job before the response was lost. Persist a client-id
                        # recovery record instead of permitting a second GPU job.
                        paused = self._stall_unknown_comfy_job(
                            job.job_id,
                            allowed_statuses=('processing', 'cancel_requested',
                                              'cancelled', 'sent_to_comfy'),
                            detail=f'unknown /prompt outcome: {exc}')
                        if not paused:
                            db.session.refresh(job)
                            if job.status == 'cancelled':
                                dispatch_cancelled = True
                            else:
                                logger.critical(
                                    'job_queue: unknown /prompt outcome for %s has no durable '
                                    'barrier; leaving job %s active and fail-closed',
                                    job.job_id, job.job_id)
                        if not dispatch_cancelled:
                            return True
                else:
                    if not mapped:
                        # A returned id with a lost mapping CAS is still exact
                        # remote ownership. Persist the prompt barrier for every
                        # active local state (not only the instrumented-cancel
                        # race) before allowing this worker to return.
                        db.session.refresh(job)
                        if job.status in ('processing', 'sent_to_comfy',
                                          'cancel_requested', 'cancelled'):
                            paused = self._stall_comfy_job(
                                job.job_id, prompt_id,
                                allowed_statuses=(job.status,),
                                detail='ComfyUI /prompt id could not be durably mapped')
                            if paused and job.status in ('cancel_requested', 'cancelled'):
                                self.reconcile_stalled_comfy_job(job.job_id)
                            db.session.refresh(job)
                            if job.status == 'stalled':
                                return True
                            if job.status == 'cancelled':
                                dispatch_cancelled = True
                        # A lost CAS/barrier write is never proof that the remote
                        # prompt failed. Keep the active row as a fail-closed owner.
                        if not dispatch_cancelled:
                            return True

        if dispatch_cancelled:
            _dispatch_completion(job, None, True)
            return True

        if submit_error is not None:
            filename, failed, error_detail = None, True, submit_error
        else:
            try:
                filename, failed = _poll_outputs(prompt_id, POLL_TIMEOUT_SECONDS)
            except Exception as exc:
                logger.exception('job_queue: poll for job %s failed', job.job_id)
                # A thrown poll has no trustworthy remote terminal observation.
                self._stall_comfy_job(job.job_id, prompt_id,
                                     allowed_statuses=('sent_to_comfy',), detail=str(exc))
                return True
            if failed is POLL_STALLED:
                # `_poll_outputs` only returns this after a durable pause or an
                # already-present global recovery block. Never route a callback.
                return True
            error_detail = None

        with GPU_ARBITER_LOCK:
            db.session.refresh(job)
            if job.status == 'stalled':
                return True
            if job.status == 'cancel_requested':
                if submit_error is None:
                    return True
                # A deterministic pre-submit refusal owns no remote prompt.
                # Its cancellation can therefore finish locally instead of
                # leaving an unresumable cancel_requested queue barrier.
                job.update_status('cancelled')
                db.session.commit()
                _dispatch_completion(job, None, True)
                return True
            if job.status == 'cancelled':
                # Keep this under the same lock as a batch cancel so its cell
                # update cannot race a late failed callback.
                _dispatch_completion(job, filename, True)
                return True
            job.update_status('failed' if failed else 'completed',
                              result_filename=filename,
                              error_message=None if not failed else
                              (error_detail or job.error_message or 'generation failed'))
            db.session.commit()

        _dispatch_completion(job, filename, failed)
        return True

    # -- public API (verbatim surface; lifted services call these) --------
    def add_job(self, job_type='image', user_id='local', workflow_data=None, prompt='',
               job_id=None, metadata=None, priority=10, *, commit=True) -> str:
        """``commit=False`` leaves the queue row PENDING in the caller's session so a
        fan-out (a Studio grid) can insert its own row and the job in ONE transaction
        — one write lock per cell instead of three. Such a caller MUST already hold
        ``GPU_ARBITER_LOCK`` before beginning its DB transaction and retain it until
        its own commit/rollback; otherwise a recovery barrier could be installed
        between this readiness check and that commit.  The worker only ever sees
        committed rows either way."""
        if job_type != 'image':
            raise ValueError(f'unsupported job_type: {job_type!r}')
        if not workflow_data:
            raise ValueError('workflow_data is required')
        job_id = job_id or str(uuid.uuid4())
        job = ImageGenerationQueue(
            job_id=job_id,
            user_id=str(user_id),
            status='pending',
            workflow_data=json.dumps(workflow_data),
            prompt=prompt,
            priority=priority,
            job_metadata=json.dumps(metadata) if metadata else None,
        )
        # Route preflights provide a fast, actionable response, but this is the
        # authoritative enqueue seam: the barrier may appear between a route
        # check and a future service call.  For commit=False this nested RLock
        # deliberately relies on the documented outer transaction guard.
        with GPU_ARBITER_LOCK:
            require_comfyui_enqueue_ready()
            db.session.add(job)
            if commit:
                db.session.commit()
        return job_id

    def cancel_job_outcome(self, job_id, user_id=None, job_type='image', *,
                           commit=True) -> str:
        """Cancel one image job and name the exact recovery outcome.

        ``cancel_job`` historically collapsed every non-success case to ``False``:
        a missing/terminal row, a known prompt that merely needs another
        reconciliation pass, and an unknown ``/prompt`` submission that can only be
        cleared after a confirmed ComfyUI restart.  Callers consequently either
        orphaned the durable barrier or kept harmless cards alive forever.

        Returns one of ``cancelled``, ``terminal``, ``missing``, ``retry``,
        ``restart_required`` or ``barrier_corrupt``. ``terminal``/``missing`` are
        safe for a caller to discard only because this method first proves that
        no corrupt barrier exists and the exact job does not own a valid one.
        """
        if job_type != 'image':
            return 'retry'
        with GPU_ARBITER_LOCK:
            query = ImageGenerationQueue.query.filter_by(job_id=str(job_id))
            if user_id is not None:
                query = query.filter_by(user_id=str(user_id))
            job = query.first()
            barrier_row, _, barrier_owner, barrier_valid = \
                self._read_comfyui_stalled_barrier()
            if barrier_row is not None and not barrier_valid:
                # Raw presence blocks every GPU action. With no trustworthy owner
                # identity, deleting any UI handle would make recovery strictly
                # worse and could orphan the global lock permanently.
                return 'barrier_corrupt'
            owns_barrier = bool(
                barrier_valid and barrier_owner
                and str(barrier_owner.get('job_id')) == str(job_id))
            if job is None:
                if owns_barrier:
                    return ('restart_required'
                            if barrier_owner.get('kind') == 'unknown_submit'
                            else 'retry')
                return 'missing'
            if job.status in ('completed', 'failed', 'cancelled'):
                if owns_barrier:
                    return ('restart_required'
                            if barrier_owner.get('kind') == 'unknown_submit'
                            else 'retry')
                return 'terminal'

            if not commit:
                # A caller holding a larger DB transaction cannot perform an
                # external proof. Refuse active work instead of silently orphaning
                # a possibly-running ComfyUI prompt.
                if job.status != 'pending':
                    return 'retry'
                job.update_status('cancelled')
                return 'cancelled'

            if job.status == 'pending':
                job.update_status('cancelled')
                db.session.commit()
                return 'cancelled'

            if job.status == 'processing':
                # Only a re-entrant cancellation from the /prompt seam can see
                # this while the shared lock is held. Persist intent without
                # claiming a remote cancellation; process_one pins its returned id.
                job.status = 'cancel_requested'
                job.last_heartbeat = naive_utcnow()
                db.session.commit()
                return 'retry'
            prompt_id = job.comfyui_prompt_id
            if job.status in ('sent_to_comfy', 'cancel_requested'):
                if prompt_id:
                    paused = self._stall_comfy_job(
                        job.job_id, prompt_id, allowed_statuses=(job.status,),
                        detail='cancellation requested')
                else:
                    paused = self._stall_unknown_comfy_job(
                        job.job_id, allowed_statuses=(job.status,),
                        detail='cancellation requested before /prompt was durably mapped')
                if not paused:
                    return 'retry'
                if not prompt_id:
                    logger.warning(
                        'job_queue: %s needs an external ComfyUI restart before '
                        'its unknown /prompt outcome can be resolved', job.job_id)
                    return 'restart_required'
            elif job.status != 'stalled':
                return 'retry'

            if not job.comfyui_prompt_id:
                logger.warning(
                    'job_queue: %s has an unknown /prompt barrier; do not clear it '
                    'without an externally verified ComfyUI restart', job.job_id)
                return 'restart_required'
            # Targeted delete / fresh absence checks happen while the exact raw
            # ownership barrier remains present.
            return ('cancelled' if self.reconcile_stalled_comfy_job(job.job_id)
                    else 'retry')

    def cancel_job(self, job_id, user_id=None, job_type='image', *, commit=True) -> bool:
        """Compatibility boolean: only a proven cancellation is ``True``."""
        return self.cancel_job_outcome(
            job_id, user_id, job_type, commit=commit) == 'cancelled'

    # -- system-state KV (underscore names required verbatim) -------------
    def _set_system_state(self, key, value, ttl_seconds=None):
        if value is None:
            SystemState.query.filter_by(key=key).delete()
            db.session.commit()
            return
        exp = time.time() + ttl_seconds if ttl_seconds is not None else None
        encoded = json.dumps({'v': value, 'exp': exp})
        row = db.session.get(SystemState, key)
        if row is None:
            db.session.add(SystemState(key=key, value=encoded))
        else:
            row.value = encoded
        db.session.commit()

    def _get_system_state(self, key, default=None):
        row = db.session.get(SystemState, key)
        if row is None or row.value is None:
            return default
        try:
            payload = json.loads(row.value)
        except (TypeError, ValueError):
            return default
        exp = payload.get('exp')
        if exp is not None and time.time() >= exp:
            try:
                db.session.delete(row)
                db.session.commit()
            except Exception:
                db.session.rollback()
            return default
        return payload.get('v', default)


queue_manager = JobQueueManager()
