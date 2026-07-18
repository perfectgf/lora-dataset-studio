"""Ollama control API.

Install DETECTION is passive and lives in /api/capabilities (`ollama.installed`
+ `ollama.reachable`). STARTING the server is the one explicit, user-triggered
action — POST /api/ollama/start — and it lives here, never fired from a probe.
"""
from flask import Blueprint, jsonify, request

from ..services import ollama_control

bp = Blueprint('ollama', __name__, url_prefix='/api/ollama')


@bp.post('/start')
def start_ollama():
    """Start the local Ollama server (idempotent: a running server → no-op ok).
    Always HTTP 200 — 'not installed' / 'did not start' are handled OUTCOMES,
    not server faults, and a 5xx would make apiFetch throw AND auto-toast a
    generic error on top of the specific one. The body carries
    {ok, reachable, error?, stderr?} either way; clients read `ok`."""
    result = ollama_control.start_ollama()
    return jsonify(result), 200


@bp.get('/models')
def list_models():
    """Installed Ollama models for the Captions ⚙️ Options model picker.
    Always 200 — {ok, reachable, models:[...]}; an unreachable server is a handled
    outcome (reachable:False, empty list), not a server fault."""
    return jsonify(ollama_control.list_models()), 200


@bp.post('/pull')
def pull_model():
    """Pull an Ollama model the user named (background, streamed). Always 200 — a bad
    name / unreachable server rides in the body as {ok:False, error}; clients poll GET
    /api/ollama/pull for {state, model, progress, log, error}."""
    data = request.get_json(silent=True) or {}
    return jsonify(ollama_control.start_pull(data.get('model'))), 200


@bp.get('/pull')
def pull_status():
    """Poll the current/last model pull: {state, model, progress, log, error}."""
    return jsonify(ollama_control.pull_status()), 200
