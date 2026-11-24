"""The pictures of Remove Background: originals, masks, and the cut-out running on one.

Everything lives in the plugin's own folder. A source picture is copied once and
never written to again, so nothing here can change a dataset, a bank or the
gallery. A cut-out is one to three short GPU jobs: find the object, cut its
edge, look closer. Each job records its intent before it is queued, and picking
a picture again reconciles a step the app was closed on.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from flask import current_app
from PIL import Image, ImageOps, UnidentifiedImageError

from lds_sdk import comfy
from lds_sdk.images import DatasetImages, GalleryImages
from lds_sdk.media import redact_tokens, redact_user_paths
from lds_sdk.storage import send_to_trash

from . import assets, engine, graphs, masks

logger = logging.getLogger(__name__)

MAX_UPLOAD = 48 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 12_000
VIEW_SIDE = 2048  # what the editor shows and paints at; results keep the picture's own size
ACTIVE = ('queued', 'running')
# Never held while waiting for another lock: the queue's own thread enters here
# with its own lock already taken.
LOCK = threading.RLock()


class Gone(LookupError):
    """The picture is not (or no longer) there."""


# --- folders and records -------------------------------------------------------

def _root(user_id):
    base = current_app.extensions['remove_background.context'].data_dir.resolve()
    folder = base / 'pictures' / hashlib.sha256(str(user_id).encode()).hexdigest()[:32]
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink() or not folder.resolve().is_relative_to(base):
        raise ValueError('The Remove Background folder is unavailable.')
    return folder.resolve()


def _folder(user_id, ident):
    if not isinstance(ident, str) or len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
        raise Gone('Picture not found.')
    base = _root(user_id)
    path = base / ident
    if path.is_symlink() or path.resolve().parent != base:
        raise Gone('Picture not found.')
    return path


def _file(folder, name):
    path = folder / name
    if path.is_symlink():
        raise Gone('Picture not found.')
    return path


def _read(user_id, ident):
    path = _file(_folder(user_id, ident), 'record.json')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise Gone('Picture not found.') from exc
    if not isinstance(record, dict) or record.get('id') != ident:
        raise Gone('Picture not found.')
    return record


def _save(user_id, record):
    folder = _folder(user_id, record['id'])
    record['updated'] = time.time()
    part = _file(folder, 'record.part')
    part.write_text(json.dumps(record), encoding='utf-8')
    os.replace(part, _file(folder, 'record.json'))


def _write_image(folder, name, image, **options):
    part = _file(folder, name + '.part')
    image.save(part, 'PNG', **options)
    os.replace(part, _file(folder, name))


def _open(folder, name, mode):
    path = _file(folder, name)
    if not path.is_file():
        raise Gone('Picture not found.')
    with Image.open(path) as image:
        return image.convert(mode)


def _say(reason, default):
    return redact_user_paths(redact_tokens(str(reason or default)))[:400]


# --- pictures ------------------------------------------------------------------

def _decode(data):
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {'PNG', 'JPEG', 'WEBP', 'BMP', 'AVIF'}:
                raise ValueError('Choose a PNG, JPEG, WebP, BMP or AVIF picture.')
            if source.width * source.height > MAX_PIXELS or max(source.size) > MAX_SIDE:
                raise ValueError('Choose a picture of at most 40 megapixels and 12000 pixels a side.')
            source.load()
            oriented = ImageOps.exif_transpose(source)
            neutral = Image.new('RGB', oriented.size, (255, 255, 255))
            if 'A' in oriented.getbands() or 'transparency' in oriented.info:
                layer = oriented.convert('RGBA')
                neutral.paste(layer, mask=layer.getchannel('A'))
            else:
                neutral.paste(oriented.convert('RGB'))
            return neutral
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError('This picture could not be opened.') from exc


def _import(user_id, data, name, origin):
    if len(data) > MAX_UPLOAD:
        raise ValueError('Choose a picture smaller than 48 MB.')
    digest = hashlib.sha256(data).hexdigest()
    # Opening the same picture again finds the work already done on it.
    for record in listing(user_id):
        if record.get('digest') == digest and record.get('origin') == origin:
            return record
    picture = _decode(data)
    ident = uuid.uuid4().hex
    folder = _folder(user_id, ident)
    folder.mkdir()
    record = {'id': ident, 'name': Path(str(name or 'Picture').replace('\\', '/')).name[:200] or 'Picture',
              'width': picture.width, 'height': picture.height, 'created': time.time(),
              'digest': digest, 'origin': origin, 'mask': None, 'run': None}
    try:
        _write_image(folder, 'original.png', picture, compress_level=3)
        for name, side, quality in (('view.webp', VIEW_SIDE, 92), ('thumb.webp', 320, 82)):
            small = picture.resize(graphs.fitted(*picture.size, side), Image.LANCZOS)
            small.save(_file(folder, name), 'WEBP', quality=quality)
        with LOCK:
            _save(user_id, record)
    except Exception:
        for path in folder.iterdir():
            path.unlink(missing_ok=True)
        folder.rmdir()
        raise
    return record


def import_upload(user_id, upload, origin=None):
    data = upload.stream.read(MAX_UPLOAD + 1)
    return _import(user_id, data, upload.filename, origin or {'kind': 'upload'})


def import_library(user_id, surface, image_id):
    """A dataset or Gallery picture, read from its own file; the file is never written."""
    if surface == 'dataset':
        library = DatasetImages(user_id)
    elif surface == 'gallery':
        library = GalleryImages(user_id)
    else:
        raise ValueError('Choose a dataset or Gallery picture.')
    if type(image_id) is not int:
        raise ValueError('Choose a dataset or Gallery picture.')
    row = library.get(image_id)
    path = library.path(image_id) if row is not None else None
    if not path or not os.path.isfile(path):
        raise Gone('This picture is no longer on disk.')
    if os.path.getsize(path) > MAX_UPLOAD:
        raise ValueError('This picture is larger than 48 MB.')
    with open(path, 'rb') as source:
        data = source.read()
    return _import(user_id, data, row.filename,
                   {'kind': surface, 'image_id': row.id, 'dataset_id': row.dataset_id})


def listing(user_id):
    """Every picture, the most recently touched first."""
    found = []
    for path in _root(user_id).glob('*/record.json'):
        try:
            found.append(_read(user_id, path.parent.name))
        except Gone:
            continue
    return sorted(found, key=lambda record: record.get('updated') or 0, reverse=True)


def remove(user_id, ident):
    with LOCK:
        record = _read(user_id, ident)
        if (record.get('run') or {}).get('status') in ACTIVE:
            raise ValueError('Cancel the running cut-out first.')
        send_to_trash(str(_folder(user_id, ident)), context='remove_background')


def media(user_id, ident, name):
    """A stored file of one picture, by its fixed name."""
    _read(user_id, ident)
    path = _file(_folder(user_id, ident), name)
    if not path.is_file():
        raise Gone('Picture not found.')
    return path


# --- the mask and what comes out of it -----------------------------------------

def _png(image):
    stream = io.BytesIO()
    image.save(stream, 'PNG', compress_level=3)
    stream.seek(0)
    return stream


def mask_png(user_id, ident, *, alpha=False, longest=None, name='mask.png'):
    """A stored mask, as gray or as transparency."""
    _read(user_id, ident)
    mask = _open(_folder(user_id, ident), name, 'L')
    if longest:
        mask = mask.resize(graphs.fitted(*mask.size, longest), Image.LANCZOS)
    return _png(masks.as_alpha(mask) if alpha else mask)


def cutout_png(user_id, ident, *, background, clean, longest=None):
    if background not in masks.BACKGROUNDS:
        raise ValueError('Choose one of the offered backgrounds.')
    _read(user_id, ident)
    folder = _folder(user_id, ident)
    picture, mask = _open(folder, 'original.png', 'RGB'), _open(folder, 'mask.png', 'L')
    if longest:
        size = graphs.fitted(*picture.size, longest)
        picture, mask = picture.resize(size, Image.LANCZOS), mask.resize(size, Image.LANCZOS)
    return _png(masks.cut_out(picture, mask, background=background, clean=clean))


def _rebuild_mask(folder):
    """mask.png = the models' answer with the user's brush over it."""
    mask = _open(folder, 'base.png', 'L')
    strokes = _file(folder, 'paint.png')
    if strokes.is_file():
        with Image.open(strokes) as painted:
            mask = masks.painted(mask, painted.convert('LA'))
    _write_image(folder, 'mask.png', mask, compress_level=3)


def paint(user_id, ident, upload):
    """Replace the brush layer (gray with alpha); None clears it."""
    strokes = None
    if upload is not None:
        data = upload.stream.read(MAX_UPLOAD + 1)
        try:
            with Image.open(io.BytesIO(data)) as source:
                if len(data) > MAX_UPLOAD or source.format != 'PNG' or max(source.size) > VIEW_SIDE:
                    raise ValueError('This brush layer could not be read.')
                strokes = source.convert('LA')
        except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError('This brush layer could not be read.') from exc
    with LOCK:
        record = _read(user_id, ident)
        if not record.get('mask'):
            raise ValueError('Remove the background first, then touch up the mask.')
        folder = _folder(user_id, ident)
        if strokes is None:
            _file(folder, 'paint.png').unlink(missing_ok=True)
        else:
            _write_image(folder, 'paint.png', strokes)
        _rebuild_mask(folder)
        record['mask'].update(version=record['mask']['version'] + 1, painted=strokes is not None)
        _save(user_id, record)
        return record


def add_candidate(user_id, ident, *, background, clean):
    """File the cut-out as a pending candidate of the dataset its picture came from."""
    record = _read(user_id, ident)
    origin = record.get('origin') or {}
    if origin.get('kind') != 'dataset':
        raise ValueError('This picture does not come from a dataset.')
    if not record.get('mask'):
        raise ValueError('Remove the background first.')
    if masks.BACKGROUNDS.get(background) is None:
        raise ValueError('A training picture cannot be transparent. Choose a plain background.')
    engine.require_queue()
    dataset = DatasetImages(user_id)
    source = dataset.get(origin['image_id'])
    if source is None:
        raise Gone('The picture this was cut from is no longer in its dataset.')
    folder = _folder(user_id, ident)
    picture = masks.cut_out(_open(folder, 'original.png', 'RGB'), _open(folder, 'mask.png', 'L'),
                            background=background, clean=clean)
    row = dataset.create_derivative(source.id, derivation_kind='background_removed')
    if row is None:
        raise Gone('The picture this was cut from is no longer in its dataset.')
    try:
        job_id = engine.submit_candidate(user_id, picture, folder, dataset_id=row.dataset_id)
        dataset.attach_job(row.id, job_id)
    except Exception:
        dataset.fail_unqueued(row.id, 'The cut-out could not be queued.')
        raise
    finally:
        dataset.sync_activity(row.dataset_id)
    return {'dataset_id': row.dataset_id, 'image_id': row.id}


# --- cutting out ---------------------------------------------------------------

def _points(points, size):
    if not isinstance(points, list) or not 1 <= len(points) <= graphs.MAX_POINTS:
        raise ValueError('Click on what stays, up to 32 points.')
    result = []
    for point in points:
        if (not isinstance(point, dict) or any(isinstance(point.get(axis), bool)
                                               or not isinstance(point.get(axis), (int, float)) for axis in 'xy')):
            raise ValueError('Click on what stays, up to 32 points.')
        x, y = float(point['x']), float(point['y'])
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            raise ValueError('Every point must be on the picture.')
        result.append({'x': x, 'y': y, 'keep': point.get('keep') is not False})
    if not any(point['keep'] for point in result):
        raise ValueError('Click at least once on what stays.')
    return result


def _selection(upload, size):
    if upload is None:
        raise ValueError('Draw around what stays first.')
    data = upload.stream.read(MAX_UPLOAD + 1)
    try:
        with Image.open(io.BytesIO(data)) as source:
            if len(data) > MAX_UPLOAD or source.format != 'PNG' or max(source.size) > VIEW_SIDE:
                raise ValueError('This selection could not be read.')
            drawn = source.convert('RGBA').getchannel('A')
    except (OSError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError('This selection could not be read.') from exc
    selection = masks.solid(drawn if drawn.size == size else drawn.resize(size, Image.BILINEAR))
    if selection.getbbox() is None:
        raise ValueError('Draw around what stays first.')
    return selection


def start(user_id, ident, *, keep, kind, phrase='', points=None, selection=None):
    """Queue the first step of a cut-out; the next ones follow from its answer."""
    if keep not in assets.KEEPS or kind not in assets.KINDS:
        raise ValueError('Choose what stays and the kind of picture.')
    record = detail(user_id, ident)
    size = (record['width'], record['height'])
    words = graphs.phrases(phrase) if keep == 'describe' else []
    if keep == 'describe' and not words:
        raise ValueError('Say in a few words what stays.')
    clicks = _points(points, size) if keep == 'points' else []
    drawn = _selection(selection, size) if keep == 'selection' else None
    files = engine.require(keep, kind)
    folder = _folder(user_id, ident)
    run = {'id': uuid.uuid4().hex, 'status': 'queued', 'error': None, 'keep': keep, 'kind': kind,
           'words': words, 'points': clicks, 'files': files,
           'stage': 'find' if keep in assets.TARGETED else 'cut', 'box': None}
    if drawn is not None:
        run['box'] = list(masks.padded(masks.bounds(drawn), size))
    with LOCK:
        record = _read(user_id, ident)
        if (record.get('run') or {}).get('status') in ACTIVE:
            raise ValueError('Wait for the running cut-out to finish, or cancel it.')
        if drawn is not None:
            _write_image(folder, 'selection.png', drawn)
        record['run'] = run
        _save(user_id, record)  # the intent survives a crash between here and the queue
    _queue_step(user_id, record, run)
    return detail(user_id, ident)


def _queue_step(user_id, record, run):
    """Queue the step ``run['stage']`` names. Called without LOCK."""
    folder = _folder(user_id, record['id'])
    picture = _open(folder, 'original.png', 'RGB')
    if run['box']:
        picture = picture.crop(tuple(run['box']))
    if run['stage'] == 'find':
        size = graphs.fitted(*picture.size, graphs.FIND_SIDE)
        scale = (size[0] / picture.width, size[1] / picture.height)
        clicks = [{'x': min(size[0] - 1, int(p['x'] * scale[0])), 'y': min(size[1] - 1, int(p['y'] * scale[1])),
                   'keep': p['keep']} for p in run['points']]
        label = 'Find what stays'

        def build(image, prefix):
            return graphs.find(image, run['files']['sam'], prefix, words=run['words'], points=clicks)
    elif run['kind'] == 'fine':
        size = graphs.fine_size(*picture.size)
        label = 'Cut the edge (RMBG 3.0)'

        def build(image, prefix):
            return graphs.fine_matte(image, run['files'], size, prefix)
    else:
        size = graphs.fitted(*picture.size, graphs.MATTE_SIDE)
        model = run['files'][assets.KINDS[run['kind']][0]]
        label = 'Cut the edge (' + assets.MODELS[assets.KINDS[run['kind']][0]]['name'] + ')'

        def build(image, prefix):
            return graphs.matte(image, model, prefix)
    if size != picture.size:
        picture = picture.resize(size, Image.LANCZOS)
    job_id = uuid.uuid4().hex
    with LOCK:
        current = _read(user_id, record['id'])
        if (current.get('run') or {}).get('id') != run['id'] or current['run']['status'] not in ACTIVE:
            return  # cancelled while this step was being prepared
        current['run'].update(stage=run['stage'], box=run['box'], job=job_id, status='queued')
        _save(user_id, current)
    try:
        engine.submit(user_id, job_id, picture, folder, build, label=label,
                      metadata={'cutout_user': str(user_id), 'cutout_picture': record['id'],
                                'cutout_run': run['id']})
    except Exception as exc:
        # Admission may have committed before failing to return: ask before declaring it lost.
        if comfy.plugin_job(engine.PLUGIN, job_id, user_id) is None:
            _fail(user_id, record['id'], run['id'], _say(exc, 'The cut-out could not be queued.'))
        raise


def _fail(user_id, ident, run_id, reason, status='failed'):
    with LOCK:
        record = _read(user_id, ident)
        run = record.get('run') or {}
        if run.get('id') == run_id and run.get('status') in ACTIVE:
            run.update(status=status, error=reason)
            _save(user_id, record)


def _finish(user_id, ident, run_id, alpha, matted):
    """The models' answer becomes the picture's mask; an earlier brush layer does not apply to it."""
    with LOCK:
        record = _read(user_id, ident)
        run = record.get('run') or {}
        if run.get('id') != run_id or run.get('status') not in ACTIVE:
            return
        folder = _folder(user_id, ident)
        _write_image(folder, 'base.png', alpha, compress_level=3)
        _file(folder, 'paint.png').unlink(missing_ok=True)
        _rebuild_mask(folder)
        version = (record.get('mask') or {}).get('version', 0) + 1
        record['mask'] = {'version': version, 'run': run['id'], 'keep': run['keep'], 'kind': run['kind'],
                          'words': run['words'], 'matted': matted, 'painted': False}
        run.update(status='done', error=None)
        _save(user_id, record)


def _answered(user_id, record, run, answer):
    """One step's answer: queue the next step, or finish. Called without LOCK."""
    folder = _folder(user_id, record['id'])
    size = (record['width'], record['height'])
    if run['stage'] == 'find':
        target = masks.without_specks(masks.solid(answer.resize(size, Image.BICUBIC)))
        box = masks.bounds(target)
        if box is None:
            what = ', '.join(run['words'])
            _fail(user_id, record['id'], run['id'],
                  f'Nothing that looks like “{what}” was found. Try other words.' if what
                  else 'Nothing was found at these points. Click on the middle of what stays.')
            return
        _write_image(folder, 'target.png', target)
        run.update(stage='cut', box=list(masks.padded(box, size)))
        _queue_step(user_id, record, run)
        return
    box = tuple(run['box'] or (0, 0, *size))
    matte = masks.gray(answer).resize((box[2] - box[0], box[3] - box[1]), Image.BICUBIC)
    matted = True
    if run['keep'] in assets.TARGETED:
        target = _open(folder, 'target.png', 'L').crop(box)
        alpha, matted = masks.follow_target(target, matte)
    elif run['keep'] == 'selection':
        alpha = masks.within(_open(folder, 'selection.png', 'L').crop(box), matte)
    else:
        alpha = matte
        again = masks.close_up(masks.level(matte)) if run['stage'] == 'cut' else None
        if again is not None:
            run.update(stage='closer', box=list(again))
            _queue_step(user_id, record, run)
            return
    alpha = masks.on_canvas(masks.level(alpha), box, size)
    if masks.bounds(alpha) is None:
        _fail(user_id, record['id'], run['id'],
              'No subject was found. Describe it, click on it, or draw around it.')
        return
    _finish(user_id, record['id'], run['id'], alpha, matted)


def _claim(user_id, ident, job_id):
    """``(record, run, wanted)``: the run this job answers, handed to one reader only.

    The queue's thread and a page that looks again can both notice the same end.
    ``wanted`` is False when another reader has the answer, so its file stays.
    """
    with LOCK:
        record = _read(user_id, ident)
        run = record.get('run') or {}
        if run.get('job') != job_id or run.get('status') not in ACTIVE:
            return record, None, True
        if run.get('claimed') == job_id:
            return record, None, False
        run['claimed'] = job_id
        _save(user_id, record)
        return record, run, True


def _discard(path):
    """A working file under a name only this step uses."""
    if path is None:
        return
    try:
        os.remove(path)
    except OSError:
        logger.debug('remove_background: a working file stays in the ComfyUI output folder')


def _step_done(user_id, ident, job_id, filename, failed, reason):
    path = engine.output_path(filename)
    record, run, wanted = _claim(user_id, ident, job_id)
    if run is None:
        if wanted:
            _discard(path)  # the run was cancelled or replaced: nobody reads this answer
        return
    try:
        if failed or path is None:
            _fail(user_id, ident, run['id'], _say(reason, 'ComfyUI finished without an answer. Try again.'))
            return
        with Image.open(path) as source:
            answer = source.convert('L')
        _answered(user_id, record, run, answer)
    except Exception as exc:
        logger.exception('remove_background: step %s failed', job_id)
        _fail(user_id, ident, run['id'], _say(exc, 'The cut-out could not be finished.'))
    finally:
        _discard(path)


def completed(job_id, filename, failed=False, reason=None, metadata=None):
    """The queue finished one of our steps."""
    metadata = metadata or {}
    try:
        _step_done(metadata.get('cutout_user'), metadata.get('cutout_picture'), job_id, filename, failed, reason)
    except Gone:
        _discard(engine.output_path(filename))


def detail(user_id, ident):
    """The record, after catching up with a step that ended while nobody was listening."""
    with LOCK:
        record = _read(user_id, ident)
    run = record.get('run') or {}
    if run.get('status') not in ACTIVE:
        return record
    quiet = time.time() - (record.get('updated') or 0)
    job = comfy.plugin_job(engine.PLUGIN, run['job'], user_id) if run.get('job') else None
    if job is None or run.get('claimed') == run.get('job'):
        # Between the record and the queue, or while an answer is being read, there is
        # nothing to see for a moment. A long silence means the app closed on it.
        if quiet > 120:
            _fail(user_id, ident, run['id'], 'The cut-out was interrupted. Start it again.')
    elif job['status'] == 'cancelled':
        _fail(user_id, ident, run['id'], 'Cancelled.', status='cancelled')
    elif job['status'] in ('completed', 'failed'):
        _step_done(user_id, ident, run['job'], job['result_filename'],
                   job['status'] == 'failed', job.get('error'))
    elif job['status'] in ('processing', 'sent_to_comfy') and run['status'] != 'running':
        with LOCK:
            current = _read(user_id, ident)
            if (current.get('run') or {}).get('job') == run['job'] and current['run']['status'] == 'queued':
                current['run']['status'] = 'running'
                _save(user_id, current)
    return _read(user_id, ident)


def cancel(user_id, ident):
    with LOCK:
        record = _read(user_id, ident)
    run = record.get('run') or {}
    if run.get('status') not in ACTIVE:
        return record
    outcome = comfy.cancel_plugin_job(engine.PLUGIN, run['job'], user_id) if run.get('job') else 'missing'
    if outcome in ('cancelled', 'terminal', 'missing'):
        _fail(user_id, ident, run['id'], 'Cancelled.', status='cancelled')
    return _read(user_id, ident)
