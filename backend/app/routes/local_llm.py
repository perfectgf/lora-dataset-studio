"""Provider-routed local LLM API.

One path both surfaces call, so a Bank picker and a Dataset picker can never end
up listing different providers' models — the divergence the repo's Bank/Dataset
parity rule exists to prevent. `/api/ollama/models` survives as an alias of the
same function for older cached bundles.
"""
from flask import Blueprint, jsonify

from ..services import vision_llm

bp = Blueprint('local_llm', __name__, url_prefix='/api/local-llm')


@bp.get('/models')
def list_models():
    """Models the CONFIGURED provider can caption with.

    Always 200 — {ok, reachable, provider, models:[...]}. An unreachable server
    is a handled outcome (empty list), never a server fault: every picker that
    reads this degrades to "no models" rather than showing an error nobody can act
    on from a dropdown.
    """
    return jsonify(vision_llm.list_models()), 200
