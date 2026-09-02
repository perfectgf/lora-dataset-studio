from __future__ import annotations

import logging
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from flask import has_app_context

from .job_queue import GPU_ARBITER_LOCK, queue_manager

logger = logging.getLogger(__name__)


class GpuBusyError(RuntimeError):
    pass


# The token is intentionally context-local. A Vision batch can propagate it to
# its worker threads to renew the same persisted ownership record, but an
# unrelated request never inherits permission simply because a DB flag exists.
_vision_window_context: ContextVar[tuple[str, int] | None] = ContextVar(
    'vision_window_context', default=None)
# Process-local complement to the persisted TTL. It remains true for a long
# Vision/CUDA operation even if a background renewal fails, and is cleared only
# after its heartbeat has stopped and the outer window has exited.
_active_vision_window_tokens: set[str] = set()
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
    """Refresh exactly one claimed token, including from a heartbeat thread."""
    def _renew():
        try:
            with GPU_ARBITER_LOCK:
                if queue_manager._get_system_state('vision_in_progress') != token:
                    return False
                queue_manager._set_system_state('vision_in_progress', token,
                                                ttl_seconds=ttl)
                return True
        except Exception:
            logger.exception('vision GPU window renewal failed')
            return False

    return bool(_in_app_context(_renew))


def _heartbeat_interval_seconds(ttl: int) -> float:
    """Refresh well before expiry without creating a busy polling loop."""
    return max(1.0, min(60.0, float(ttl) / 3.0))


def renew_gpu_exclusive_vision_window(flag_ttl=None) -> bool:
    """Refresh this context's persisted Vision ownership before an Ollama call.

    A worker that outlives the outer Vision window cannot revive it: it must
    still own the exact token in ``SystemState``. ``False`` is fail-closed and
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
            while not heartbeat_stop.wait(_heartbeat_interval_seconds(ttl)):
                if not _renew_owned_vision_token(token, ttl):
                    # The process-local active-token fence remains set until the
                    # outer window exits, so Queue/Training stay blocked even if
                    # this persisted TTL has expired.
                    logger.error(
                        'vision GPU window heartbeat lost ownership; in-process GPU fence retained')
                    return

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

        yield
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
                _in_app_context(_clear_owned)
        finally:
            # Keep this after stop/join and database cleanup. Even if the app
            # context is unavailable during teardown, the in-process fence must
            # not outlive the actual Vision work.
            if active_registered:
                with GPU_ARBITER_LOCK:
                    _active_vision_window_tokens.discard(token)
            _vision_window_context.reset(context_token)
