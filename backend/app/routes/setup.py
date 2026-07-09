"""Setup API: auto-detect installed tools + run the whitelisted one-click installs."""
from flask import Blueprint, jsonify

from .. import capabilities
from .. import setup_installer

bp = Blueprint('setup', __name__, url_prefix='/api/setup')


@bp.get('/autodetect')
def setup_autodetect():
    """Discover already-installed tools (Ollama/ComfyUI/ai-toolkit) so the wizard
    can fill config itself. Reachable-port hits are safe to apply; disk paths are
    suggestions the UI confirms."""
    return jsonify(capabilities.autodetect())


@bp.post('/install/<action>')
def start_install(action):
    if action not in setup_installer.INSTALL_ACTIONS:
        return jsonify({'error': f'unknown action: {action}'}), 404
    try:
        state = setup_installer.start(action)
    except setup_installer.AlreadyRunning:
        return jsonify({'error': 'install already running'}), 409
    except setup_installer.Precondition as e:
        return jsonify({'error': str(e)}), 400
    return jsonify(state)


@bp.get('/install/<action>/status')
def install_status(action):
    if action not in setup_installer.INSTALL_ACTIONS:
        return jsonify({'error': f'unknown action: {action}'}), 404
    return jsonify(setup_installer.status(action))
