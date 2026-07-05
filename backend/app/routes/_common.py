"""Helpers shared by more than one route blueprint."""
from flask import jsonify

from .. import capabilities
from ..gpu_window import GpuBusyError


def _map_error(e: Exception):
    """Map a service/vision exception to a Flask (body, status) tuple.
    Unrecognized exceptions are re-raised (-> 500, a real bug)."""
    if isinstance(e, GpuBusyError):
        return jsonify({'error': 'GPU busy', 'detail': str(e)}), 503
    if isinstance(e, ValueError):
        return jsonify({'error': str(e)}), 400
    if isinstance(e, RuntimeError):
        return jsonify({'error': str(e)}), 409
    raise e


def _require_comfyui():
    """None if ComfyUI is reachable, else the (body, status) 409 to return.
    Shared by studio.py and datasets.py's lora-test routes that actually enqueue
    a ComfyUI job (run/resume) — read-only/history/DB-only routes stay ungated."""
    if not capabilities.probe()['comfyui']['reachable']:
        return jsonify({'error': 'ComfyUI is not reachable',
                        'hint': 'Check the URL in Settings'}), 409
    return None
