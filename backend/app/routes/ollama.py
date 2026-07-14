"""Ollama control API.

Install DETECTION is passive and lives in /api/capabilities (`ollama.installed`
+ `ollama.reachable`). STARTING the server is the one explicit, user-triggered
action — POST /api/ollama/start — and it lives here, never fired from a probe.
"""
from flask import Blueprint, jsonify

from ..services import ollama_control

bp = Blueprint('ollama', __name__, url_prefix='/api/ollama')


@bp.post('/start')
def start_ollama():
    """Start the local Ollama server (idempotent: a running server → no-op ok).
    502 on failure so the client can distinguish a genuine start from a no-op —
    the body carries {ok, reachable, error?, stderr?} either way."""
    result = ollama_control.start_ollama()
    return jsonify(result), (200 if result.get('ok') else 502)
