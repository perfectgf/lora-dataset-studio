"""Store browsing, dependency consent and authenticated package preparation."""
from flask import Blueprint, Response, jsonify, request
from packaging.version import InvalidVersion, Version

from ...version import APP_VERSION
from ..admin import is_admin, require_admin
from ..lifecycle import state_change_lock
from ..routes import _change_blocker, _registry, restart_payload
from . import service
from .catalog import members_releases
from .client import StoreError, StoreNotConfigured

bp = Blueprint('plugin_store', __name__, url_prefix='/store')

from .commerce import register_commerce_routes  # noqa: E402
from .patreon import register_patreon_routes  # noqa: E402
register_commerce_routes(bp)
register_patreon_routes(bp)


def _input(*, confirmation=False):
    data = request.get_json(silent=True)
    if isinstance(data, dict) and not isinstance(data.get('update_only', False), bool):
        raise StoreError('Choose a valid update operation.')
    if isinstance(data, dict) and 'ids' in data:
        ids = data['ids']
        if ('id' in data or data.get('version') is not None
                or not isinstance(ids, list) or not 1 <= len(ids) <= 50
                or any(not isinstance(pid, str) or not 1 <= len(pid) <= 128 for pid in ids)
                or len(set(ids)) != len(ids)
                or (confirmation and (not isinstance(data.get('plan_id'), str)
                                      or len(data['plan_id']) != 64))):
            raise StoreError('Select valid plugins and review their shared installation plan.')
        return {**data, 'id': sorted(ids)}
    if (not isinstance(data, dict) or not isinstance(data.get('id'), str)
            or not 1 <= len(data['id']) <= 128
            or (data.get('version') is not None and (
                not isinstance(data['version'], str) or not 1 <= len(data['version']) <= 128))
            or (confirmation and (not isinstance(data.get('plan_id'), str)
                                 or len(data['plan_id']) != 64))):
        raise StoreError('Select a valid plugin release and review its installation plan.')
    return data


@bp.get('/catalog')
def catalog():
    try:
        payload = service.browse()
    except StoreNotConfigured as exc:
        payload = {'status': 'not_configured', 'message': str(exc), 'products': []}
    except (StoreError, OSError) as exc:
        payload = {'status': 'unavailable', 'message': str(exc) if isinstance(exc, StoreError)
                   else 'The store cache is unavailable.', 'products': []}
    payload['can_manage'] = is_admin()
    registry = _registry()
    from ...services import patreon_core
    members = patreon_core.members_versions() if payload['products'] else ()
    for product in payload['products']:
        product['members_releases'] = members_releases(product['releases'], members, current=APP_VERSION)
        record = registry.records.get(product['id']) if registry else None
        release = next((r for r in product['releases'] if not r['compatibility_issues']), None)
        product['recommended'] = release
        product['installed_version'] = record.manifest.version if record else None
        try:
            product['update_available'] = bool(record and release and Version(release['manifest']['version']) > Version(record.manifest.version))
        except InvalidVersion:
            product['update_available'] = False
    return jsonify(payload)


@bp.get('/media/<plugin_id>/<version>/<image_id>/<filename>')
def presentation_image(plugin_id, version, image_id, filename):
    from .media import read_image

    try:
        data, content_type = read_image(plugin_id, version, image_id, filename)
        response = Response(data, content_type=content_type)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'none'; sandbox"
        response.headers['Cache-Control'] = 'private, max-age=3600, immutable'
        response.set_etag(filename.split('.')[0])
        return response.make_conditional(request)
    except (StoreError, OSError):
        return jsonify({'error': 'This presentation image is unavailable.'}), 404


@bp.post('/plan')
def plan():
    if (denied := require_admin()) is not None:
        return denied
    try:
        data = _input()
        with state_change_lock:
            return jsonify(service.preview_plan(_registry(), data.get('id'), data.get('version'),
                                                update_only=data.get('update_only', False)))
    except (StoreError, OSError, ValueError) as exc:
        return jsonify({'error': str(exc) if isinstance(exc, StoreError) else 'The store plan could not be prepared.'}), 409


@bp.post('/install')
def install():
    if (denied := require_admin()) is not None:
        return denied
    try:
        data = _input(confirmation=True)
        with state_change_lock:
            registry = _registry()
            def check_change(plugin_id):
                # Preparation retains running code; restart checks active work.
                # Updating an enabled plugin does not abandon its resources.
                blocked = _change_blocker(plugin_id, disabling=False)
                if blocked is not None:
                    raise StoreError(blocked[0].get_json()['error'])
            result = service.prepare(registry, data.get('id'), data.get('version'), data.get('plan_id'),
                                     check_change=check_change, update_only=data.get('update_only', False))
            result['restart'] = restart_payload()
            return jsonify(result)
    except (StoreError, OSError, ValueError) as exc:
        return jsonify({'error': str(exc) if isinstance(exc, StoreError) else 'No package set could be prepared; existing files are retained.'}), 409
