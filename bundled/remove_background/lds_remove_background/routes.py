"""The Remove Background workspace, under ``/api/remove-background``."""
from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file

from lds_sdk import config, training_data
from lds_sdk.http import map_error, require_idle_comfy

from . import engine, masks, studio

bp = Blueprint('remove_background', __name__)

_PUBLIC = ('id', 'name', 'width', 'height', 'created', 'updated', 'origin', 'mask')
_RUN = ('id', 'status', 'stage', 'error', 'keep', 'kind', 'words')


def _picture(record):
    """What the browser may see of a record: no digest, no loader names."""
    run = record.get('run')
    return {**{key: record.get(key) for key in _PUBLIC},
            'run': {key: run.get(key) for key in _RUN} if run else None}


def _refused(error):
    if isinstance(error, engine.NotReady):
        return jsonify(error=str(error), missing=error.missing), 409
    if isinstance(error, LookupError):
        return jsonify(error=str(error) or 'Picture not found.'), 404
    return map_error(error)


def _body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _side():
    """The editor asks for a smaller picture than the one it downloads."""
    value = request.args.get('max', type=int)
    return min(max(value, 64), studio.VIEW_SIDE) if value else None


def _image(stream, name, mimetype='image/png'):
    response = send_file(stream, mimetype=mimetype, download_name=name,
                         as_attachment=request.args.get('download') == '1')
    # The address carries the mask's version, so a stored answer is never out of date.
    response.headers['Cache-Control'] = ('private, max-age=86400' if request.args.get('v') else 'private, no-cache')
    return response


def _stem(record):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in record['name'].rsplit('.', 1)[0])[:80] or 'picture'


@bp.get('/status')
def status():
    return jsonify(engine.status())


@bp.get('/datasets')
def datasets():
    """Where a cut-out can be added, for the picker."""
    return jsonify(datasets=[{'id': item.id, 'name': item.name}
                             for item in training_data.list_datasets(config.local_user())])


@bp.get('/pictures')
def pictures():
    return jsonify(pictures=[_picture(item) for item in studio.listing(config.local_user())])


@bp.post('/pictures')
def add_picture():
    user = config.local_user()
    try:
        if request.files.get('image'):
            origin = {'kind': 'bank'} if request.form.get('surface') == 'bank' else None
            record = studio.import_upload(user, request.files['image'], origin)
        else:
            data = _body()
            if not data.get('surface'):
                return jsonify(error='Choose a picture to import.'), 400
            record = studio.import_library(user, data['surface'], data.get('image_id'))
        return jsonify(picture=_picture(record)), 201
    except Exception as exc:
        return _refused(exc)


@bp.get('/pictures/<ident>')
def picture(ident):
    try:
        return jsonify(picture=_picture(studio.detail(config.local_user(), ident)))
    except Exception as exc:
        return _refused(exc)


@bp.delete('/pictures/<ident>')
def remove_picture(ident):
    try:
        studio.remove(config.local_user(), ident)
        return jsonify(ok=True)
    except Exception as exc:
        return _refused(exc)


@bp.get('/pictures/<ident>/view')
@bp.get('/pictures/<ident>/thumb')
def view(ident):
    name = 'thumb.webp' if request.path.endswith('/thumb') else 'view.webp'
    try:
        return _image(studio.media(config.local_user(), ident, name), name, 'image/webp')
    except Exception as exc:
        return _refused(exc)


@bp.post('/pictures/<ident>/cut')
def cut(ident):
    gate = require_idle_comfy()
    if gate:
        return gate
    # A drawn selection travels as a picture, so its other fields come as a form.
    data = request.form if request.files else _body()
    points = data.get('points')
    try:
        record = studio.start(config.local_user(), ident, keep=data.get('keep'), kind=data.get('kind'),
                              phrase=data.get('phrase') or '', points=points if isinstance(points, list) else None,
                              selection=request.files.get('selection'))
        return jsonify(picture=_picture(record)), 202
    except Exception as exc:
        return _refused(exc)


@bp.post('/pictures/<ident>/cancel')
def cancel(ident):
    try:
        return jsonify(picture=_picture(studio.cancel(config.local_user(), ident)))
    except Exception as exc:
        return _refused(exc)


@bp.post('/pictures/<ident>/paint')
def paint(ident):
    try:
        record = studio.paint(config.local_user(), ident, request.files.get('strokes'))
        return jsonify(picture=_picture(record))
    except Exception as exc:
        return _refused(exc)


@bp.get('/pictures/<ident>/paint')
def paint_layer(ident):
    try:
        return _image(studio.media(config.local_user(), ident, 'paint.png'), 'paint.png')
    except Exception as exc:
        return _refused(exc)


@bp.get('/pictures/<ident>/base')
def base_mask(ident):
    """The models' own mask as transparency, without the brush: what painting starts from."""
    try:
        return _image(studio.mask_png(config.local_user(), ident, alpha=True, longest=_side(), name='base.png'),
                      'base.png')
    except Exception as exc:
        return _refused(exc)


@bp.get('/pictures/<ident>/mask')
def mask(ident):
    user = config.local_user()
    try:
        record = studio.detail(user, ident)
        return _image(studio.mask_png(user, ident, longest=_side()), _stem(record) + '-mask.png')
    except Exception as exc:
        return _refused(exc)


@bp.get('/pictures/<ident>/cutout')
def cutout(ident):
    user = config.local_user()
    try:
        record = studio.detail(user, ident)
        background = request.args.get('background') or 'transparent'
        if background not in masks.BACKGROUNDS:
            return jsonify(error='Choose one of the offered backgrounds.'), 400
        return _image(studio.cutout_png(user, ident, background=background,
                                        clean=request.args.get('clean') == '1', longest=_side()),
                      _stem(record) + '-cutout.png')
    except Exception as exc:
        return _refused(exc)


@bp.post('/pictures/<ident>/dataset')
def add_to_its_dataset(ident):
    """A pending candidate of the dataset this picture came from."""
    gate = require_idle_comfy()
    if gate:
        return gate
    data = _body()
    try:
        return jsonify(ok=True, **studio.add_candidate(
            config.local_user(), ident, background=data.get('background') or 'white',
            clean=data.get('clean') is not False)), 202
    except Exception as exc:
        return _refused(exc)
