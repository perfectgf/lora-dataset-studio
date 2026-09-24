"""Imported originals and durable camera views owned by this plugin."""
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import uuid

from flask import current_app
from PIL import Image, ImageOps, UnidentifiedImageError
from lds_sdk import comfy, config as cfg
from lds_sdk.lifecycle import state_change_lock
from lds_sdk.media import redact_tokens, redact_user_paths

from . import camera_angles as ca, qwen_camera_helper as qch

MAX_UPLOAD = 32 * 1024 * 1024
MAX_PIXELS = 40_000_000
LOCK = threading.RLock()
ACTIVE = {'queued', 'running', 'stalled', 'cancel_requested'}


def root(user_id):
    base = current_app.extensions['camera_angles.context'].data_dir.resolve()
    folder = base / 'studio' / hashlib.sha256(str(user_id).encode()).hexdigest()
    if not folder.resolve().is_relative_to(base):
        raise ValueError('The camera image folder is unavailable.')
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def folder(user_id, ident):
    if not isinstance(ident, str) or len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
        raise LookupError('Camera image not found.')
    base = root(user_id)
    path = base / ident
    if path.is_symlink() or path.resolve().parent != base:
        raise LookupError('Camera image not found.')
    return path


def read(user_id, ident):
    path = folder(user_id, ident) / 'record.json'
    if path.is_symlink() or not path.is_file():
        raise LookupError('Camera image not found.')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('id') != ident:
        raise ValueError('The camera image record is invalid.')
    return record


def save(user_id, record):
    base = folder(user_id, record['id'])
    part, final = base / 'record.part', base / 'record.json'
    if part.is_symlink() or final.is_symlink():
        raise ValueError('The camera image record is unavailable.')
    part.write_text(json.dumps(record), encoding='utf-8')
    os.replace(part, final)


def import_image(user_id, upload):
    data = upload.stream.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise ValueError('Choose an image smaller than 32 MB.')
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ValueError('Choose a PNG, JPEG or WebP image.')
            if source.width * source.height > MAX_PIXELS:
                raise ValueError('Choose an image with at most 40 megapixels.')
            oriented = ImageOps.exif_transpose(source).convert('RGB')
            neutral = Image.new('RGB', oriented.size)
            neutral.paste(oriented)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError('This image could not be opened. Choose a PNG, JPEG or WebP image.') from exc
    ident = uuid.uuid4().hex
    target = folder(user_id, ident)
    target.mkdir()
    record = {'id': ident, 'name': Path((upload.filename or 'Image').replace('\\', '/')).name[:200],
              'width': neutral.width, 'height': neutral.height, 'views': []}
    try:
        neutral.save(target / 'original.png', 'PNG')
        with LOCK:
            save(user_id, record)
    except Exception:
        for name in ('original.png', 'record.part', 'record.json'):
            (target / name).unlink(missing_ok=True)
        target.rmdir()
        raise
    return record


def media_path(user_id, ident, view_id=None):
    record = read(user_id, ident)
    name = 'original.png'
    if view_id is not None:
        view = next((v for v in record['views'] if v['id'] == view_id), None)
        if view is None or view['status'] != 'done':
            raise LookupError('Camera view not available yet.')
        name = view['id'] + '.png'
    base = folder(user_id, ident)
    path = base / name
    if path.is_symlink() or path.resolve().parent != base or not path.is_file():
        raise LookupError('Camera image not found.')
    return path


def _finish(user_id, record, view, filename, failed, reason):
    if view['status'] == 'done':
        return
    if not failed and not filename:
        failed, reason = True, 'The render finished without an output image. Shoot this view again.'
    if failed:
        view.update(status='failed', error=redact_user_paths(redact_tokens(
            str(reason or 'The camera view could not be rendered.')))[:500])
    elif filename:
        output = cfg.comfyui_dir('output')
        if not output:
            raise ValueError('ComfyUI output is not configured.')
        base = Path(output).resolve()
        source = base / filename
        if source.is_symlink() or source.resolve().parent != base or not source.is_file():
            raise ValueError('The rendered camera image is unavailable.')
        destination = folder(user_id, record['id']) / (view['id'] + '.png')
        if destination.is_symlink():
            raise ValueError('The camera result file is unavailable.')
        # Keep the ComfyUI output for recovery; the plugin owns its durable copy.
        with Image.open(source) as image:
            image.convert('RGB').save(destination, 'PNG')
        view.update(status='done', error=None)
    save(user_id, record)


def completed(job_id, filename, failed=False, reason=None, metadata=None):
    metadata = metadata or {}
    user_id, ident = metadata.get('camera_user'), metadata.get('camera_source')
    with LOCK:
        record = read(user_id, ident)
        view = next((v for v in record['views'] if v['id'] == job_id), None)
        if view is not None:
            _finish(user_id, record, view, filename, failed, reason)


def detail(user_id, ident):
    with LOCK:
        record = read(user_id, ident)
        for view in record['views']:
            if view['status'] not in ACTIVE:
                continue
            job = comfy.plugin_job('camera_angles', view['id'], user_id)
            if job is None:
                _finish(user_id, record, view, None, True, 'Queueing was interrupted. Shoot this view again.')
            elif job['status'] in {'completed', 'failed', 'cancelled'}:
                try:
                    _finish(user_id, record, view, job['result_filename'],
                            job['status'] != 'completed', job.get('error'))
                except (ValueError, OSError):
                    _finish(user_id, record, view, None, True, 'The rendered camera image is unavailable.')
            else:
                view['status'] = {'pending': 'queued', 'processing': 'running',
                                  'sent_to_comfy': 'running'}.get(job['status'], job['status'])
        return record


def listing(user_id):
    with LOCK:
        entries = sorted(root(user_id).glob('*/record.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        return [read(user_id, p.parent.name) for p in entries]


def shoot(user_id, ident, poses):
    wanted = ca.normalize_requested(poses)
    missing = qch.camera_missing_assets()
    if not qch.camera_ready(missing):
        raise qch.CameraModelsMissing(missing)
    with state_change_lock, LOCK:
        record = detail(user_id, ident)
        source = media_path(user_id, ident)
        if any(view['status'] in ACTIVE for view in record['views']):
            raise ValueError('Wait for the current camera views to finish before shooting again.')
        queued = 0
        for azimuth, elevation, distance in wanted:
            view = {'id': uuid.uuid4().hex, 'pose': ca.pose_id(azimuth, elevation, distance),
                    'label': ca.pose_label(azimuth, elevation, distance), 'status': 'queued', 'error': None}
            record['views'].append(view)
            save(user_id, record)  # Intent survives a crash between admission and completion.
            try:
                qch.enqueue_camera_view(
                    str(user_id), 'original.png', str(source), ca.pose_prompt(azimuth, elevation, distance),
                    standalone_job_id=view['id'],
                    extra_metadata={'camera_source': ident, 'camera_user': str(user_id)})
                queued += 1
            except Exception:
                # Admission may have committed before failing to return. Reconcile
                # its fixed ID before declaring it failed, never submit it twice.
                if comfy.plugin_job('camera_angles', view['id'], user_id) is not None:
                    queued += 1
                else:
                    _finish(user_id, record, view, None, True, 'This camera view could not be queued.')
                if not queued:
                    raise
                break
        return {'image': record, 'queued': queued, 'requested': len(wanted)}
