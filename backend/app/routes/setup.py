"""Setup API: auto-detect installed tools + run the whitelisted one-click installs."""
from flask import Blueprint, jsonify, request

from .. import capabilities
from .. import config as cfg
from .. import setup_installer
from ..services import comfyui_control

bp = Blueprint('setup', __name__, url_prefix='/api/setup')


def _plugin_action_admin_gate(action):
    """The historical Setup URL must preserve a plugin action's admin boundary."""
    spec = setup_installer.plugin_action_spec(action)
    download = setup_installer.model_download_spec(action)
    if spec is not None or (download is not None and download.get('plugin')):
        from ..plugins.admin import require_admin
        return require_admin()
    return None


@bp.get('/autodetect')
def setup_autodetect():
    """Discover already-installed tools (Ollama/ComfyUI/ai-toolkit) so the wizard
    can fill config itself. Reachable-port hits are safe to apply; disk paths are
    suggestions the UI confirms."""
    return jsonify(capabilities.autodetect())


@bp.get('/runtime-readiness')
def setup_runtime_readiness():
    """Cheap boot-state poll for services managed by Docker deployments.

    Unlike a forced capability refresh this performs no disk/model/import
    scan, and its response exposes no configured URL or filesystem path.
    """
    response = jsonify(capabilities.setup_runtime_readiness())
    response.headers['Cache-Control'] = 'no-store'
    return response


@bp.put('/ollama-deployment')
def save_ollama_deployment():
    """Persist the Docker Ollama choice made inside LDS Setup.

    The client selects only an enum; endpoints are fixed here so a stale/native
    URL cannot disagree with the deployment shown by Setup. This never starts a
    service and never pulls a model -- the launcher and explicit install action
    retain those separate responsibilities.
    """
    if not capabilities.setup_is_docker_runtime():
        return jsonify({'error': 'Ollama deployment choices apply only to Docker.'}), 409
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict) or set(body) != {'mode'}:
        return jsonify({'error': "body must contain only 'mode'"}), 400
    mode = body.get('mode')
    if not isinstance(mode, str) or mode not in ('none', 'host', 'docker'):
        return jsonify({'error': "mode must be 'none', 'host' or 'docker'"}), 400
    saved = cfg.save_config({'ollama': {
        'deployment_mode': mode,
        'url': capabilities.setup_ollama_deployment_url(mode),
    }})
    readiness = capabilities.setup_runtime_readiness()['ollama']
    return jsonify({'config': saved, 'readiness': readiness})


@bp.get('/comfyui-dir')
def setup_validate_comfyui_dir():
    """Classify a candidate ComfyUI folder WITHOUT saving it, so the wizard can give
    immediate, actionable feedback as the field is edited — a wrong path, an empty
    folder, or the launcher/parent folder (with the child to adopt) instead of a
    blanket "invalid" that only shows up after a save. Read-only, cheap (a couple of
    stat calls), never raises. `?path=` is the raw folder string the user typed."""
    return jsonify(capabilities.classify_comfyui_dir(request.args.get('path', '')))


@bp.post('/comfyui/start')
def start_comfyui():
    """Explicitly start LDS's narrowly validated portable ComfyUI instance.

    This endpoint intentionally accepts no path, flags, query values, or JSON. The
    service reads only the already-persisted local configuration and owns no process
    started by the user's normal ComfyUI launcher. LAN/Tailscale clients already
    allowed by the application-wide access policy may use it; the CSRF guard still
    applies.
    """
    # No client input may influence the executable, cwd, or arguments. Do not parse a
    # body at all; reject it instead. The CSRF header remains the only required POST
    # metadata and is enforced by Flask-WTF before this handler in normal operation.
    if request.query_string or request.content_length not in (None, 0) or request.content_type:
        return jsonify({'error': 'This action does not accept options.'}), 400
    comfy_mode = capabilities.setup_comfyui_mode()
    if comfy_mode in ('integrated', 'external-host'):
        if comfy_mode == 'integrated':
            detail = 'ComfyUI is managed by this Docker deployment.'
        else:
            detail = ('This Docker deployment uses ComfyUI on the host. Start it '
                      'on the host and make its API reachable from Docker.')
        return jsonify({'error': detail}), 409
    return jsonify(comfyui_control.start_comfyui()), 200


@bp.get('/comfyui-folders')
def setup_resolve_comfyui_folders():
    """Resolve the four ComfyUI working folders (output/input/models/loras) for the
    values currently TYPED in Settings, without saving anything, so each override
    field can show the effective path it falls back to and flag one that is not on
    disk. `?base_dir=` plus optional `?output_dir=&input_dir=&models_dir=&loras_dir=`.

    `?detect=1` additionally asks the running ComfyUI which custom folders it was
    launched with (its own argv, via /system_stats) so the UI can offer them in one
    click. That is one short network call, hence opt-in; `detected` is {} whenever
    ComfyUI is unreachable, too old to report its argv, or was started with no
    custom folder flags — never a guessed path."""
    base_dir = request.args.get('base_dir', '')
    overrides = {k: request.args.get(k, '') for k in
                 ('output_dir', 'input_dir', 'models_dir', 'loras_dir')}
    payload = {'folders': capabilities.classify_comfyui_folders(base_dir, overrides),
               'detected': {}}
    if request.args.get('detect'):
        payload['detected'] = capabilities.detect_comfyui_folders()
    return jsonify(payload)


@bp.get('/actions')
def plugin_actions():
    """The Setup actions the enabled plugins added, with their labels — the
    core's own labels are the frontend's table; a plugin's come from here."""
    return jsonify({'actions': setup_installer.plugin_actions_catalog()})


@bp.get('/aitoolkit-install/plan')
def aitoolkit_install_plan():
    """The dry run of Install AI Toolkit for `?dest=` (or the default folder):
    the GPU and driver verdict, the CUDA build, the free space, the source
    (git clone or archive), the Python base and the exact commands, with the
    reason when it cannot run here. Read-only: nothing is downloaded or written."""
    from ..services import aitoolkit_install
    response = jsonify(aitoolkit_install.plan(request.args.get('dest', '')))
    response.headers['Cache-Control'] = 'no-store'
    return response


@bp.post('/install/<action>')
def start_install(action):
    if not setup_installer.known_action(action):
        return jsonify({'error': f'unknown action: {action}'}), 404
    if (denied := _plugin_action_admin_gate(action)) is not None:
        return denied
    # Only the AI Toolkit install takes options (its destination folder).
    options = request.get_json(silent=True) if action == 'aitoolkit_install' else None
    try:
        state = setup_installer.start(action, options)
    except setup_installer.AlreadyRunning:
        return jsonify({'error': 'install already running'}), 409
    except (setup_installer.Precondition, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    return jsonify(state)


@bp.get('/install/<action>/status')
def install_status(action):
    if not setup_installer.known_action(action):
        return jsonify({'error': f'unknown action: {action}'}), 404
    return jsonify(setup_installer.status(action))


@bp.post('/install/<action>/cancel')
def cancel_install(action):
    if not setup_installer.known_action(action):
        return jsonify({'error': f'unknown action: {action}'}), 404
    if (denied := _plugin_action_admin_gate(action)) is not None:
        return denied
    try:
        state = setup_installer.cancel(action)
    except setup_installer.Precondition as exc:
        return jsonify({'error': str(exc)}), 409
    return jsonify(state)


@bp.get('/install-all/plan')
def install_all_plan():
    """What 'Install everything' WOULD queue for the current machine — the missing
    components the app can install itself (ML extras, the vision model when Ollama is
    up, the Klein weights when a valid ComfyUI is set). Read-only, so the button can
    show the plan (and an accurate 'X items') before the user commits."""
    caps = capabilities.probe()
    return jsonify({'plan': setup_installer.install_all_plan(caps)})


@bp.post('/install-all')
def start_install_all():
    """One click that queues every install in the plan above. Reuses the per-action
    serialization (pip FIFO) and preconditions, so it's a safe fan-out — nothing new to
    race. Returns the plan + each action's status for the global progress bar."""
    # A selected plugin batch must not silently become the core's global plan.
    # Existing global clients send either an empty object or no body.
    if request.get_data() and request.get_json(silent=True) != {}:
        return jsonify(error='Use the plugin’s preparation action to install selected components.'), 400
    caps = capabilities.probe()
    return jsonify(setup_installer.start_all(caps))


@bp.get('/install-group/<group>/plan')
def install_group_plan(group):
    """What the one-click install for a NAMED group (today: the Krea 2 Edit
    engine — node pack + four weights) would queue right now. Read-only, so the
    button can show its own count and stay honest about what is already there."""
    if group not in setup_installer.install_groups():
        return jsonify({'error': f'unknown group: {group}'}), 404
    # force=True: this plan is read right after the ComfyUI folder is saved, and a
    # 30 s-stale probe would answer "nothing to install" for a machine that has
    # nothing installed — the exact opposite of the truth.
    return jsonify({'plan': setup_installer.install_group_plan(
        group, capabilities.probe(force=True))})


@bp.post('/install-group/<group>')
def start_install_group(group):
    """Install a whole engine in one click without dragging it into the
    unattended 'Install everything' plan — a second local engine is ~20 GB, so it
    downloads when it is ASKED for, not by default."""
    if group not in setup_installer.install_groups():
        return jsonify({'error': f'unknown group: {group}'}), 404
    registry = setup_installer._plugin_registry()
    if registry and group in registry.install_groups:
        from ..plugins.admin import require_admin
        if (denied := require_admin()) is not None:
            return denied
    for action in setup_installer.install_groups()[group]:
        if (denied := _plugin_action_admin_gate(action)) is not None:
            return denied
    from ..plugins.preparation import PreparationError
    try:
        return jsonify(setup_installer.start_group(group, capabilities.probe(force=True)))
    except PreparationError as exc:
        return jsonify(error=str(exc)), exc.status
    except (setup_installer.Precondition, ValueError) as exc:
        return jsonify(error=str(exc)), 400


@bp.get('/install-all/status')
def install_all_status():
    """Batched status for the actions the caller is tracking (?actions=a,b,c) — one poll
    for the whole 'Install everything' run instead of one request per action."""
    actions = [a for a in (request.args.get('actions', '') or '').split(',') if a]
    return jsonify({'statuses': setup_installer.status_many(actions)})
