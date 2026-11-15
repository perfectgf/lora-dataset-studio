from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from flask import has_app_context

from .job_queue import (GPU_ARBITER_LOCK, HOLD_COMFYUI_RECOVERY, HOLD_TRAINING, HOLD_VISION,
                        queue_manager)

logger = logging.getLogger(__name__)

# Work queued or running in ComfyUI. Not one of job_queue's HOLD_* keys: those
# say why the ComfyUI worker itself stands still, and ComfyUI is not held by
# its own queue.
HOLD_GENERATION = 'generation'

# The GPU owners that end on their own. A refusal because one of them holds the
# card is a reason to wait for it, not a failure of the request itself.
TRANSIENT_HOLDS = frozenset({HOLD_TRAINING, HOLD_VISION, HOLD_GENERATION})

# Who is in the way, as a noun phrase for "waiting for {x}".
_HOLD_WAIT_LABELS = {
    HOLD_TRAINING: 'a LoRA training',
    HOLD_VISION: 'a vision pass',
    HOLD_GENERATION: 'an image generation in ComfyUI',
    HOLD_COMFYUI_RECOVERY: 'a paused ComfyUI job to be recovered',
}


def wait_label(hold) -> str:
    return _HOLD_WAIT_LABELS.get(hold, 'another task')


class GpuBusyError(RuntimeError):
    usage_error_code = 'busy'
    usage_error_reason = 'gpu_busy'

    def __init__(self, *args, hold=None):
        super().__init__(*args)
        # WHICH owner refused, when the GPU is merely in use (TRANSIENT_HOLDS).
        # None for the refusals that are not a wait: a ComfyUI recovery, a
        # state that could not be read, a release ComfyUI did not confirm.
        self.hold = hold


# The token is intentionally context-local. A Vision batch can propagate it to
# its worker threads to renew the same persisted ownership record, but an
# unrelated request never inherits permission simply because a DB flag exists.
_vision_window_context: ContextVar[tuple[str, int] | None] = ContextVar(
    'vision_window_context', default=None)
# Process-local complement to the persisted TTL. It remains true for a long
# Vision/CUDA operation even if a background renewal fails, and is cleared only
# after its heartbeat has stopped and the outer window has exited.
_active_vision_window_tokens: set[str] = set()
# Renewals that failed in a row, to log their traceback once per streak. Read and
# written under GPU_ARBITER_LOCK only.
_renewal_failures = [0]
_MIN_FLAG_TTL_SECONDS = 30
_MAX_FLAG_TTL_SECONDS = 3600


def _normalise_flag_ttl(value) -> int:
    try:
        ttl = int(value)
    except (TypeError, ValueError):
        ttl = 300
    return max(_MIN_FLAG_TTL_SECONDS, min(_MAX_FLAG_TTL_SECONDS, ttl))


def _in_app_context(callback):
    """Run a tiny state update from a propagated worker context when needed."""
    if has_app_context():
        return callback()
    app = queue_manager._app
    if app is None:
        return False
    with app.app_context():
        return callback()

def vision_window_is_owned() -> bool:
    """Whether this execution context belongs to an active Vision GPU window."""
    return _vision_window_context.get() is not None


def vision_gpu_window_blocks_gpu() -> bool:
    """Whether an in-process Vision/CUDA window still owns the local GPU.

    This does not consult the TTL-backed database flag: it is the fail-closed
    fallback for a long operation whose renewal temporarily fails.
    """
    with GPU_ARBITER_LOCK:
        return bool(_active_vision_window_tokens)


def bind_vision_window_context(callback):
    """Bind only the Vision token to a worker; never copy Flask contextvars."""
    owned = _vision_window_context.get()
    if owned is None:
        return callback

    def _bound(*args, **kwargs):
        context_token = _vision_window_context.set(owned)
        try:
            return callback(*args, **kwargs)
        finally:
            _vision_window_context.reset(context_token)

    return _bound


def _renew_owned_vision_token(token: str, ttl: int) -> bool:
    """Refresh exactly one claimed token, including from a heartbeat thread.

    Two situations look like a lost reservation and are not one:

    * The persisted record EXPIRED while its window was still running: the
      machine slept for longer than the TTL, or every renewal was starved. No
      other owner could take the GPU meanwhile, because the in-process fence
      (``_active_vision_window_tokens``) stayed up, so the record is written
      again. Before this, a pass that woke up refused every remaining image.
    * The WRITE is refused for a moment (a full disk, a file another program
      holds) while the stored record is still this token and has not expired:
      the reservation stands and the next renewal tries again.

    Still ``False``, fail-closed: another owner holds the record, the window of
    this token has exited (a worker that outlived it cannot bring it back), or
    ownership cannot be confirmed at all.
    """
    def _renew():
        from .extensions import db
        with GPU_ARBITER_LOCK:
            try:
                current = queue_manager._get_system_state('vision_in_progress')
                if current != token:
                    if current is not None or token not in _active_vision_window_tokens:
                        return False
                    logger.warning(
                        'vision GPU window: its reservation had expired while the pass was '
                        'still running (the machine slept, or the renewals were starved); '
                        'written again')
                queue_manager._set_system_state('vision_in_progress', token,
                                                ttl_seconds=ttl)
                _renewal_failures[0] = 0
                return True
            except Exception as exc:
                # One traceback per streak. A pass now goes on through a refused
                # write, and every call renews: a disk that stays full for half
                # an hour would otherwise log the same forty lines hundreds of
                # times, onto that same disk.
                _renewal_failures[0] += 1
                if _renewal_failures[0] == 1:
                    logger.exception('vision GPU window renewal failed')
                else:
                    logger.warning('vision GPU window renewal failed again (%d in a row): %s',
                                   _renewal_failures[0], exc)
                # A commit that failed leaves its session unusable until it is
                # rolled back, and on the pass's own thread that session is the
                # pass's: without this, its next write fails with "This session
                # is in 'prepared' state" instead of the real cause.
                try:
                    db.session.rollback()
                    return queue_manager._get_system_state('vision_in_progress') == token
                except Exception:
                    logger.exception('vision GPU window: ownership could not be read back')
                    try:
                        db.session.rollback()
                    except Exception:   # noqa: BLE001 — nothing left to undo
                        pass
                    return False

    return bool(_in_app_context(_renew))


def _heartbeat_interval_seconds(ttl: int) -> float:
    """Refresh well before expiry without creating a busy polling loop."""
    return max(1.0, min(60.0, float(ttl) / 3.0))


def renew_gpu_exclusive_vision_window(flag_ttl=None) -> bool:
    """Refresh this context's persisted Vision ownership before an Ollama call.

    A worker that outlives the outer Vision window cannot revive it: the record
    is only written again while that window is still running in this process
    (see :func:`_renew_owned_vision_token`). ``False`` is fail-closed and
    callers must not start another local inference after it.
    """
    owned = _vision_window_context.get()
    if owned is None:
        return False
    token, previous_ttl = owned
    ttl = _normalise_flag_ttl(previous_ttl if flag_ttl is None else flag_ttl)
    return _renew_owned_vision_token(token, ttl)


def recover_stale_vision_window():
    """Clear only a persisted Vision lock during server startup.

    A stalled ComfyUI barrier is a different ownership record and must remain
    intact across startup. Vision work itself cannot survive this Python process,
    so its token is safe to clear.
    """
    def _recover():
        with GPU_ARBITER_LOCK:
            previous = queue_manager._get_system_state('vision_in_progress')
            if not previous:
                return False
            queue_manager._set_system_state('vision_in_progress', None)
            logger.warning('startup recovery: cleared stale vision/GPU lock from the previous process')
            return True

    return bool(_in_app_context(_recover))


@contextmanager
def gpu_exclusive_vision_window(flag_ttl=300):
    """Give one Vision operation exclusive ownership of the local GPU.

    This is a handoff, not a per-image cleanup: a batch enters once, asks
    ComfyUI to release its models once, then keeps Ollama hot for all of its
    calls. A ComfyUI batch does the inverse at its first prompt, never between
    its cells.
    """
    if _vision_window_context.get() is not None:
        # Re-entering from the same request would obscure ownership and can make
        # a stale worker look valid. Batches propagate the token only to renew.
        raise GpuBusyError('a vision task is already running')

    ttl = _normalise_flag_ttl(flag_ttl)
    token = uuid.uuid4().hex
    context_token = _vision_window_context.set((token, ttl))
    claimed = False
    active_registered = False
    body_failed = False
    heartbeat_stop = threading.Event()
    heartbeat = None
    try:
        with GPU_ARBITER_LOCK:
            try:
                if vision_gpu_window_blocks_gpu():
                    raise GpuBusyError('a vision task is already running')
                if queue_manager._get_system_state('vision_in_progress'):
                    raise GpuBusyError('a vision task is already running')
                if queue_manager._get_system_state('training_in_progress'):
                    raise GpuBusyError('training is running')
                if queue_manager.has_comfyui_stalled_barrier():
                    raise GpuBusyError(
                        'ComfyUI recovery is required before a vision task can use the GPU.')
                if queue_manager.has_comfyui_work():
                    raise GpuBusyError(
                        'ComfyUI has queued or active work; wait for it or cancel it before running vision.')
            except GpuBusyError:
                raise
            except Exception as exc:
                raise GpuBusyError(
                    'Could not confirm GPU ownership safely; try again after checking ComfyUI.') from exc

            queue_manager._set_system_state('vision_in_progress', token, ttl_seconds=ttl)
            claimed = True
            _active_vision_window_tokens.add(token)
            active_registered = True
            try:
                from .utils.comfyui import free_comfyui_vram
                verdict = free_comfyui_vram()
            except Exception:
                logger.exception('vision GPU window: ComfyUI /free raised unexpectedly')
                verdict = None

            # Asked of the member, not compared to a class imported here: the
            # enum's own property answers for whichever incarnation of the
            # class the member belongs to. The suite once reloaded
            # utils.comfyui, and a member of the old class was never `in` a
            # tuple of the new one. `is not True` keeps the gate fail-closed:
            # a stand-in whose attribute is merely truthy does not open it.
            if getattr(verdict, 'permits_ollama', False) is not True:
                if queue_manager._get_system_state('vision_in_progress') == token:
                    queue_manager._set_system_state('vision_in_progress', None)
                _active_vision_window_tokens.discard(token)
                active_registered = False
                claimed = False
                raise GpuBusyError(
                    'ComfyUI did not confirm that its GPU models were released. '
                    'Wait for it to recover, then try the vision task again.')

        # Some CUDA subprocesses and local image passes legitimately run for
        # longer than their initial TTL. The heartbeat owns only this exact token;
        # cleanup stops and joins it before clearing, so a stale thread cannot
        # revive a released or replacement window.
        def _heartbeat():
            lost = False
            while not heartbeat_stop.wait(_heartbeat_interval_seconds(ttl)):
                if _renew_owned_vision_token(token, ttl):
                    if lost:
                        lost = False
                        logger.warning('vision GPU window heartbeat renewed its reservation again')
                elif not lost:
                    # The process-local active-token fence remains set until the
                    # outer window exits, so Queue/Training stay blocked even if
                    # this persisted TTL has expired. The heartbeat keeps asking:
                    # it used to stop here for good, so one refused write (a full
                    # disk for a minute) left a pass of several hours without any
                    # renewal, and the record expired under it.
                    lost = True
                    logger.error(
                        'vision GPU window heartbeat lost ownership; in-process GPU fence retained')

        heartbeat = threading.Thread(
            target=_heartbeat, name='lds-vision-gpu-heartbeat', daemon=True)
        try:
            heartbeat.start()
        except Exception as exc:
            with GPU_ARBITER_LOCK:
                if queue_manager._get_system_state('vision_in_progress') == token:
                    queue_manager._set_system_state('vision_in_progress', None)
                _active_vision_window_tokens.discard(token)
            active_registered = False
            claimed = False
            raise GpuBusyError('Could not keep the Vision GPU reservation alive safely.') from exc

        try:
            yield
        except BaseException:
            body_failed = True
            raise
    finally:
        heartbeat_stop.set()
        if heartbeat is not None and heartbeat is not threading.current_thread():
            heartbeat.join(timeout=2)
        try:
            if claimed:
                def _clear_owned():
                    with GPU_ARBITER_LOCK:
                        if queue_manager._get_system_state('vision_in_progress') == token:
                            queue_manager._set_system_state('vision_in_progress', None)
                try:
                    _in_app_context(_clear_owned)
                except Exception:
                    # The failure of the pass is the one to report. A cleanup
                    # that fails for the same reason (a database that refuses
                    # every write) used to replace it with its own error, and a
                    # full disk was reported as a session in 'prepared' state.
                    if not body_failed:
                        raise
                    logger.exception('vision GPU window: its reservation could not be cleared '
                                     'after the pass failed; it expires on its own')
        finally:
            # Keep this after stop/join and database cleanup. Even if the app
            # context is unavailable during teardown, the in-process fence must
            # not outlive the actual Vision work.
            if active_registered:
                with GPU_ARBITER_LOCK:
                    _active_vision_window_tokens.discard(token)
            _vision_window_context.reset(context_token)
