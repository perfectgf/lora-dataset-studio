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
    queue_manager._set_system_state('vision_in_progress', True, ttl_seconds=flag_ttl)
    try:
        try:
            from .utils.comfyui import free_comfyui_vram
            free_comfyui_vram()
        except Exception:
            pass
        yield
    finally:
        queue_manager._set_system_state('vision_in_progress', None)
