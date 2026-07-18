"""Back up everything — a single master archive of every dataset + the app
config (secrets excluded), produced/restored as a background job so a
multi-gigabyte library never rides on a blocking request.

Endpoints (all under /api/backup):
    POST /full/start            start a background backup
    GET  /full/status           poll it {state, done, total, current, result|error}
    GET  /full/download?name=   download a produced archive (validated basename)
    POST /full/open-folder      reveal the backups folder in the file explorer
    POST /full/restore          upload a backup zip; a MASTER archive starts a
                                background restore, a single-dataset backup is
                                imported inline (one button restores both)
    GET  /full/restore/status   poll the restore job

The restore upload reuses the dataset-archive request-size allowance (see
``_DATASET_ARCHIVE_UPLOAD_ENDPOINTS`` in app/__init__.py — ``backup.full_restore``
is listed there) so a large master archive isn't rejected by the ordinary ceiling.
"""
import os
import shutil

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from ..config import LOCAL_USER
from .. import config as cfg
from ..services import full_backup as fb
from ..services import face_dataset_service as svc

bp = Blueprint('backup', __name__, url_prefix='/api/backup')


def _uploaded_stream(file_storage):
    """A rewound, seekable upload stream, enforcing the archive cap."""
    stream = file_storage.stream
    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(0)
    except (AttributeError, OSError, ValueError) as exc:
        raise ValueError('uploaded archive is not seekable') from exc
    if size > int(current_app.config['DATASET_ARCHIVE_MAX_UPLOAD_BYTES']):
        raise RequestEntityTooLarge()
    return stream


@bp.post('/full/start')
def full_start():
    """Kick off a background 'back up everything'. 409 if one is already running,
    or if the disk can't hold it (checked up front so the user isn't told to wait
    for a run that will fail)."""
    try:
        fb.check_disk_space()
    except fb.DiskSpaceError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 409
    try:
        fb.start_backup(current_app._get_current_object(), LOCAL_USER)
    except fb.AlreadyRunning as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 409
    return jsonify({'ok': True, 'running': True})


@bp.get('/full/status')
def full_status():
    return jsonify(fb.status('backup'))


@bp.get('/full/download')
def full_download():
    path = fb.resolve_backup_file(request.args.get('name') or '')
    if not path:
        return jsonify({'ok': False, 'error': 'backup not found'}), 404
    return send_file(path, mimetype='application/zip', as_attachment=True,
                     download_name=os.path.basename(path))


@bp.post('/full/open-folder')
def full_open_folder():
    try:
        fb.open_backups_folder()
    except Exception:
        current_app.logger.exception('could not open backups folder')
        return jsonify({'ok': False, 'error': 'could not open backups folder'}), 500
    return jsonify({'ok': True})


@bp.post('/full/restore')
def full_restore():
    """Restore from an uploaded backup zip. Auto-routes by manifest format so the
    library's single 'Import backup' button accepts both a master archive (started
    as a background job) and a plain single-dataset backup (imported inline)."""
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'ok': False, 'error': 'no file'}), 400
    try:
        stream = _uploaded_stream(f)
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400

    fmt = fb.detect_backup_format(stream)
    if fmt == fb.FULL_BACKUP_FORMAT:
        # Persist the upload for the worker (the request ends before it finishes).
        tmp = str(cfg.backups_dir() / f'.restore-{os.urandom(8).hex()}.zip')
        with open(tmp, 'wb') as dst:
            shutil.copyfileobj(stream, dst, 1024 * 1024)
        try:
            fb.start_restore(current_app._get_current_object(), LOCAL_USER, tmp)
        except fb.AlreadyRunning as exc:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return jsonify({'ok': False, 'error': str(exc)}), 409
        return jsonify({'ok': True, 'kind': 'full', 'running': True})

    if fmt == svc.BACKUP_FORMAT:
        try:
            ds = svc.import_backup_zip(LOCAL_USER, stream)
        except ValueError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400
        return jsonify({'ok': True, 'kind': 'single', 'id': ds.id, 'name': ds.name})

    return jsonify({'ok': False, 'error': 'not a LoRA Dataset Studio backup'}), 400


@bp.get('/full/restore/status')
def full_restore_status():
    return jsonify(fb.status('restore'))
