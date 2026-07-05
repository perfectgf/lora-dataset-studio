import uuid
from contextlib import contextmanager
from .job_queue import queue_manager

class GpuBusyError(RuntimeError):
    pass

@contextmanager
def gpu_exclusive_vision_window(flag_ttl=300):
    if queue_manager._get_system_state('vision_in_progress'):
        raise GpuBusyError('a vision task is already running')
    if queue_manager._get_system_state('training_in_progress'):
        raise GpuBusyError('training is running')
    token = uuid.uuid4().hex
    queue_manager._set_system_state('vision_in_progress', token, ttl_seconds=flag_ttl)
    try:
        try:
            from .utils.comfyui import free_comfyui_vram
            free_comfyui_vram()
        except Exception:
            pass
        yield
    finally:
        # only clear the flag if we still own it (it may have expired and been re-acquired)
        if queue_manager._get_system_state('vision_in_progress') == token:
            queue_manager._set_system_state('vision_in_progress', None)
