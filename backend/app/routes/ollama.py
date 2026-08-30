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
    """Installed Ollama models — kept as an ALIAS of /api/local-llm/models.

    The four model pickers (dataset options, bank options, Caption Lab, Enhance)
    moved to the provider-routed path; this one stays because the URL is public
    surface and an older cached bundle may still ask for it. Both answer the same
    shape, so neither caller can be surprised: {ok, reachable, models:[...]},
    always 200 — an unreachable server is a handled outcome, not a server fault.
    """
    from ..services import vision_llm
    return jsonify(vision_llm.list_models()), 200


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
