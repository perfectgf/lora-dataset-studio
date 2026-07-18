"""🖥️ System pickers — server-side folder selection for the Browse… field.

The folders the app works on live on the machine RUNNING the server (not the
browser's), so these two endpoints do the selecting there:

  POST /api/system/pick-folder   pops the OS-native dialog on the server desktop.
  GET  /api/system/list-folders  read-only in-app browser (drives + subfolders).

pick-folder never 500s on the expected "no desktop here" case — it answers 200
with {available:false} so the UI silently falls back to the in-app browser
(LAN/tablet/Linux) instead of flashing a scary error toast.
"""
import logging

from flask import Blueprint, jsonify, request

from ..services import folder_picker

logger = logging.getLogger(__name__)

bp = Blueprint('system', __name__, url_prefix='/api/system')


@bp.post('/pick-folder')
def pick_folder():
    data = request.get_json(silent=True) or {}
    initial = (data.get('initial') or '').strip() or None
    try:
        path = folder_picker.open_native_folder_dialog(initial)
    except folder_picker.NativePickerUnavailable as e:
        # Expected on a headless / Linux / service-session server: 200 so the
        # front falls back to the in-app browser without an error toast.
        return jsonify({'available': False, 'reason': str(e)})
    if path is None:
        return jsonify({'available': True, 'cancelled': True})
    return jsonify({'available': True, 'path': path})


@bp.get('/list-folders')
def list_folders():
    path = request.args.get('path') or None
    try:
        return jsonify(folder_picker.list_subfolders(path))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except PermissionError:
        return jsonify({'error': 'Permission denied for this folder.'}), 403
