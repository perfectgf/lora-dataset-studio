"""Helpers shared by more than one route blueprint."""
from flask import jsonify

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
