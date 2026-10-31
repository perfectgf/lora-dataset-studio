"""``/api/plugins`` — what is installed, what is on, how to change it.

Enabling or disabling writes ``plugins.enabled.<id>`` and answers how THIS
install restarts (registration happens at boot): the supervisor relaunches by
itself on start.bat and the GPU Docker image; the CPU image, Pinokio and the
portable bundle get a sentence that says what to do, never a promise the code
does not keep.

The static route that serves an external plugin's UI checks containment on
``realpath`` and refuses links: the design review served a ``.env`` through a
directory junction and a hard link placed inside a plugin folder.
"""
from __future__ import annotations

import os
import tempfile
import zipfile

from flask import Blueprint, current_app, jsonify, request, send_file

from .. import config as cfg
from .hooks import run_filter
from ..services import updater
from .api import LDS_PLUGIN_API_MAJOR, LDS_PLUGIN_API_MINOR
from .compatibility import IncompatiblePackage, installation_issues
from .install import ArchiveError, file_is_safe_to_serve, inspect_plugin_zip
from .loader import external_dir, registry_of
from .lifecycle import state_change_lock
from .manifest import load_manifest
from .registry import PluginRecord
from .. import setup_installer
from . import environment, storage, official

bp = Blueprint('plugins', __name__, url_prefix='/api/plugins')


@bp.before_request
def admin_mutation_gate():
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        from .admin import require_admin
        return require_admin()
    return None


def restart_payload() -> dict:
    if updater.is_pinokio_runtime():
        mode, how = 'pinokio', 'Stop the app in Pinokio, then Start it again, for the change to apply.'
    elif os.environ.get('LDS_RESTART_MODE', '').strip().lower() == 'supervisor':
        mode, how = 'self', 'Apply changes and restart LDS to use the requested plugin state.'
    elif updater.is_docker_runtime():
        mode, how = 'container', 'Restart the container (docker compose restart) for the change to apply.'
    else:
        mode, how = 'manual', 'Close the app and start it again for the change to apply.'
    return {'required': True, 'mode': mode, 'how': how, 'can_apply': mode == 'self'}


def _registry():
    registry = registry_of(current_app)
    if registry is None:
        return None
    return registry


def lifecycle_payload(registry):
    """The current boot and durable requested state are separate axes."""
    payload = registry.payload() if registry else {'plugins': [], 'misplaced': [], 'stale': [], 'dirs': {}}
    operations, errors = storage.pending(external_dir())
    enabled = cfg.get('plugins.enabled') or {}
    records = dict(registry.records) if registry else {}
    # A freshly prepared package is visible without importing any of its code.
    for plugin_id, operation in operations.items():
        if plugin_id in records or operation['action'] == 'remove':
            continue
        try:
            folder = storage._operation_dir(external_dir(), operation) / 'package'
            manifest = load_manifest(folder, official=official.valid_provenance(
                operation.get('provenance'), plugin_id, folder))
            records[plugin_id] = PluginRecord(plugin_id, manifest, str(folder), False, state='pending')
        except (OSError, ValueError):
            errors.append({'id': plugin_id, 'error': 'The pending plugin package is unreadable; its files are retained.'})
    plugins = []
    from .membership import payload as membership_payload
    membership = membership_payload(registry)
    for plugin_id, record in records.items():
        plugin = record.payload()
        if plugin_id in membership['plugin_ids']:
            plugin['membership'] = {key: value for key, value in membership.items() if key != 'plugin_ids'}
        operation = operations.get(plugin_id)
        desired = enabled.get(plugin_id)
        desired = True if desired is None else bool(desired)
        if operation:
            desired = operation['desired_enabled']
        runtime = registry is not None and plugin_id in registry.records
        active = runtime and record.state == 'loaded'
        action = operation['action'] if operation else (
            ('enable' if desired else 'disable') if desired != record.enabled else None)
        plugin.update(enabled=desired, desired_enabled=desired, active=active,
                      installed_version=record.manifest.version if runtime and record.manifest else None,
                      loaded_version=record.manifest.version if active else None,
                      pending_action=action, pending_restart=action is not None,
                      pending_version=(operation.get('manifest') or {}).get('version') if operation else None)
        record.pending_restart = operation is not None
        plugin['environment'] = environment.summary(record, registry) if runtime else None
        plugins.append(plugin)
    payload['plugins'] = plugins
    payload['membership'] = membership
    payload['boot_id'] = registry.boot_id if registry else None
    payload['pending_restart'] = bool(operations or errors or any(p['pending_action'] for p in plugins))
    payload['lifecycle_errors'] = list(registry.lifecycle_errors if registry else []) + errors
    try:
        payload['transactions'] = storage.transaction_history(external_dir())
    except (OSError, ValueError):
        payload['transactions'] = []
        payload['lifecycle_errors'].append({'error': 'The plugin installation history could not be read.'})
    return payload


@bp.get('/')
def list_plugins():
    payload = lifecycle_payload(_registry())
    payload['api'] = {'major': LDS_PLUGIN_API_MAJOR, 'minor': LDS_PLUGIN_API_MINOR}
    payload['restart'] = restart_payload()
    return jsonify(payload)


@bp.get('/membership')
def plugin_membership():
    from .membership import payload
    response = jsonify(payload(_registry(), refresh=True))
    response.headers['Cache-Control'] = 'no-store'
    return response


@bp.post('/apply')
def apply_changes():
    from .restart import apply_changes as apply
    return apply()


@bp.post('/<plugin_id>/preparation')
def prepare_plugin(plugin_id):
    from . import preparation
    try:
        return jsonify(preparation.start(plugin_id, request.get_json(silent=True),
                                         registry=_registry(), root=external_dir()))
    except preparation.PreparationError as exc:
        return jsonify(error=str(exc)), exc.status


def _change_blocker(plugin_id, *, disabling=True):
    if setup_installer.plugin_install_busy(plugin_id) or environment.running(plugin_id):
        return jsonify({'error': 'Wait for this plugin’s installation or running work to finish.'}), 409
    if not disabling:
        return None
    try:
        blockers = [str(r) for r in (run_filter(
            'plugin.disable_blockers', [], plugin_id, strict=True) or ()) if r]
    except Exception:
        current_app.logger.exception('could not check plugin disable blockers')
        return jsonify({'error': 'Could not verify that this plugin can be changed. '
                                 'Retry after its active work has stopped.', 'id': plugin_id}), 409
    if blockers:
        return jsonify({'error': blockers[0], 'blockers': blockers, 'id': plugin_id}), 409
    return None


def _set_enabled(plugin_id: str, value):
    with state_change_lock:
        try:
            return _set_enabled_locked(plugin_id, value)
        except (OSError, storage.StorageError):
            return jsonify({'error': 'Could not save the plugin change. Its files are retained.'}), 409


def _set_enabled_locked(plugin_id: str, value):
    registry = _registry()
    operations, _ = storage.pending(external_dir())
    if (registry is None or plugin_id not in registry.records) and plugin_id not in operations:
        return jsonify({'error': f'no plugin {plugin_id!r} in this install'}), 404
    if not value:
        blocked = _change_blocker(plugin_id)
        if blocked is not None:
            return blocked
    if plugin_id in operations:
        try:
            storage.set_pending_enabled(external_dir(), operations[plugin_id], value)
        except storage.StorageError as exc:
            return jsonify({'error': str(exc)}), 409
    else:
        cfg.save_config({'plugins': {'enabled': {plugin_id: value}}})
    return jsonify({'ok': True, 'id': plugin_id, 'enabled': bool(value), 'restart': restart_payload()})


@bp.post('/<plugin_id>/enable')
def enable_plugin(plugin_id):
    return _set_enabled(plugin_id, True)


@bp.post('/<plugin_id>/disable')
def disable_plugin(plugin_id):
    return _set_enabled(plugin_id, False)


@bp.get('/<plugin_id>/ui/<path:filename>')
@bp.get('/<plugin_id>/ui/boot/<boot_id>/<path:filename>')
def plugin_ui(plugin_id, filename, boot_id=None):
    registry = _registry()
    if boot_id is not None and (registry is None or boot_id != registry.boot_id):
        return jsonify({'error': 'this plugin URL belongs to an earlier app boot'}), 404
    record = registry.records.get(plugin_id) if registry else None
    if record is None or record.bundled or record.state != 'loaded' or not record.manifest.frontend:
        return jsonify({'error': 'no such plugin UI'}), 404
    ui_root = os.path.join(record.dir, os.path.dirname(record.manifest.frontend.replace('/', os.sep)))
    target = os.path.join(ui_root, filename.replace('/', os.sep))
    if not os.path.isfile(target) or not file_is_safe_to_serve(target, ui_root):
        return jsonify({'error': 'no such file'}), 404
    mimetype = 'text/javascript' if target.endswith(('.js', '.mjs')) else None
    response = send_file(os.path.realpath(target), mimetype=mimetype, conditional=True)
    response.cache_control.no_cache = True
    return response


@bp.post('/install')
def install_plugin():
    with state_change_lock:
        try:
            return _install_plugin_locked()
        except (OSError, storage.StorageError, zipfile.BadZipFile):
            return jsonify({'error': 'Could not prepare the plugin package. Existing files and data are retained.'}), 409


def _install_plugin_locked():
    """Two-step on purpose: without ``confirm`` the archive is inspected and
    the consent payload returned; with it, the plugin is extracted. The same
    file is uploaded twice — plugins are small, and a consent screen that
    describes a different archive than the one extracted would be worthless."""
    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({'error': 'send the plugin ZIP as the "file" field'}), 400
    confirm = str(request.form.get('confirm', '')).lower() in ('1', 'true', 'yes')
    replace = str(request.form.get('replace', '')).lower() in ('1', 'true', 'yes')
    fd, tmp = tempfile.mkstemp(prefix='lds-plugin-', suffix='.zip')
    os.close(fd)
    try:
        upload.save(tmp)
        try:
            manifest, prefix = inspect_plugin_zip(tmp)
        except ArchiveError as exc:
            return jsonify({'error': str(exc)}), 400
        registry = _registry()
        if registry is not None and manifest.id in registry.records and registry.records[manifest.id].bundled:
            return jsonify({'error': f'{manifest.id!r} is a bundled plugin id'}), 409
        def check_compatibility():
            # Re-read preferences again under admission, including another
            # process's requested OFF state since the consent preview.
            cfg.load_config(force=True)
            return installation_issues(manifest, registry.records if registry else {},
                                       cfg.get('plugins.enabled') or {})

        issues = check_compatibility()
        if not confirm:
            return jsonify({'needs_confirm': True, 'manifest': manifest.summary(),
                            'install_dir': str(external_dir() / manifest.id),
                            'compatibility_issues': issues, 'can_install': not issues})
        blocked = _change_blocker(manifest.id)
        if blocked is not None:
            return blocked
        dest = storage.managed_path(external_dir(), manifest.id)
        operations, _ = storage.pending(external_dir())
        if dest.exists() or manifest.id in operations:
            if not replace:
                return jsonify({'error': f'{manifest.id!r} is already installed; send replace=1 to overwrite it',
                                'code': 'already_installed'}), 409
        try:
            def preflight():
                found = check_compatibility()
                current_desired = (cfg.get('plugins.enabled') or {}).get(manifest.id)
                if bool(True if current_desired is None else current_desired) != bool(True if desired is None else desired):
                    found.append({'code': 'configuration_changed', 'field': f'plugins.enabled.{manifest.id}',
                                  'message': 'The requested plugin state changed. Review its installation again.',
                                  'expected': True if desired is None else bool(desired),
                                  'actual': True if current_desired is None else bool(current_desired)})
                if found:
                    raise IncompatiblePackage(found)

            if issues:
                raise IncompatiblePackage(issues)
            desired = (cfg.get('plugins.enabled') or {}).get(manifest.id)
            storage.prepare_install(external_dir(), tmp, manifest, prefix, replacing=dest.exists(),
                                    desired_enabled=True if desired is None else bool(desired), preflight=preflight)
        except IncompatiblePackage as exc:
            return jsonify({'error': str(exc), 'code': 'incompatible_plugin',
                            'compatibility_issues': exc.issues}), 409
        except ArchiveError as exc:
            return jsonify({'error': str(exc)}), 400
        except storage.StorageError as exc:
            return jsonify({'error': str(exc)}), 409
        if registry and manifest.id in registry.records:
            registry.records[manifest.id].pending_restart = True
        return jsonify({'ok': True, 'installed': manifest.summary(), 'install_dir': str(dest),
                        'restart': restart_payload()})
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


@bp.delete('/<plugin_id>')
def remove_plugin(plugin_id):
    with state_change_lock:
        try:
            return _remove_plugin_locked(plugin_id)
        except (OSError, storage.StorageError):
            return jsonify({'error': 'Could not prepare the plugin removal. Its files and data are retained.'}), 409


def _remove_plugin_locked(plugin_id):
    registry = _registry()
    record = registry.records.get(plugin_id) if registry else None
    if record is None:
        return jsonify({'error': f'no plugin {plugin_id!r} in this install'}), 404
    if record.bundled:
        return jsonify({'error': 'a bundled plugin ships with the app; disable it instead'}), 409
    if record.legacy is not None:
        return jsonify({'error': 'Disable this legacy extension, then remove its original folder after restarting LDS.'}), 409
    blocked = _change_blocker(plugin_id)
    if blocked is not None:
        return blocked
    keep_data = str(request.args.get('keep_data', '1')).lower() not in ('0', 'false', 'no')
    try:
        storage.prepare_remove(external_dir(), plugin_id, keep_data=keep_data)
    except storage.StorageError as exc:
        return jsonify({'error': str(exc)}), 409
    saved = storage.data_dir(plugin_id) if keep_data else None
    record.pending_restart = True
    return jsonify({'ok': True, 'id': plugin_id, 'data_kept_at': str(saved) if saved else None,
                    'restart': restart_payload()})


# Nest under the existing administration/CSRF boundary and keep the public
# blueprint classification unchanged. Helpers above are ready before import.
from .store.routes import bp as store_bp  # noqa: E402
bp.register_blueprint(store_bp)
