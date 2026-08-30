"""Provider-routed local LLM API.

One path both surfaces call, so a Bank picker and a Dataset picker can never end
up listing different providers' models — the divergence the repo's Bank/Dataset
parity rule exists to prevent. `/api/ollama/models` survives as an alias of the
same function for older cached bundles.
"""
from flask import Blueprint, jsonify

from ..services import vision_llm

bp = Blueprint('local_llm', __name__, url_prefix='/api/local-llm')


@bp.post('/start')
def start_server():
    """Start the configured provider's local server (idempotent).

    Always HTTP 200 — "not installed" / "did not start" are handled OUTCOMES, not
    server faults, and a 5xx would make apiFetch throw AND auto-toast a generic
    error on top of the specific one. The body carries {ok, reachable, error?,
    stderr?} either way; clients read `reachable`.
    """
    return jsonify(vision_llm.start_server()), 200


@bp.post('/load')
def load_model():
    """Load the provider's vision model now (explicit click; probes never do this).

    Always 200 -- "nothing downloaded" is a handled outcome with its remedy in the
    body, and a 5xx would stack a generic toast on top of the specific sentence.
    """
    return jsonify(vision_llm.load_model()), 200


@bp.get('/models')
def list_models():
    """Models the CONFIGURED provider can caption with.

    Always 200 — {ok, reachable, provider, models:[...]}. An unreachable server
    is a handled outcome (empty list), never a server fault: every picker that
    reads this degrades to "no models" rather than showing an error nobody can act
    on from a dropdown.
    """
    return jsonify(vision_llm.list_models()), 200
