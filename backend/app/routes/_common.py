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


_STUDIO_FAMILY_LABELS = {'zimage': 'Z-Image', 'sdxl': 'SDXL', 'krea': 'Krea 2 Turbo'}


def _studio_missing_response(e):
    """Turn a StudioAssetsMissing into a structured 409 (same spirit as Klein's
    missing-models 409): a human message + the itemized file/node lists the front
    lists in a banner, so the user knows WHY the grid can't run instead of watching
    every tile fail silently.

    No auto-download: unlike Klein's public assets, Studio bases / VAEs / text
    encoders are large and often license-gated, and the missing custom nodes aren't
    files at all — a clear 'place X here / install node Y' is the P0 contract.
    Shared by the per-dataset run and the comparison run."""
    fam = _STUDIO_FAMILY_LABELS.get(e.family, e.family)
    bits = []
    if e.missing_files:
        bits.append(f"{len(e.missing_files)} required model file(s)")
    if e.missing_nodes:
        bits.append(f"{len(e.missing_nodes)} custom node(s)")
    msg = f"The {fam} test pipeline can't run — your ComfyUI is missing " + " and ".join(bits) + ". "
    if e.missing_files:
        msg += "Place the file(s) at the shown path(s) inside your ComfyUI folder. "
    if e.missing_nodes:
        msg += "Install the missing custom node(s) into ComfyUI. "
    msg += "Then relaunch the test."
    return jsonify({'ok': False, 'error': msg,
                    'studio_missing': {'family': e.family,
                                       'files': e.missing_files,
                                       'nodes': e.missing_nodes}}), 409
