"""📤 Publish to Civitai — the link store, the two publish jobs and their poll.

Every write here is a state-changing POST behind the app's CSRF gate. The
slow parts (a multi-hundred-MB checkpoint, a handful of re-encoded PNGs over a
home uplink) run as background jobs the modal polls through `/jobs/<id>`; the
synchronous seams live in services.civitai_publish and are what the tests
drive. No route ever accepts a file path from the client: a save is addressed
as `(record_id, step, filename)` and resolved server-side, an image by its
library row id, a link by its own id.
"""
from flask import Blueprint, current_app, jsonify, request

from ..config import LOCAL_USER
from ..services import civitai_publish as cp

bp = Blueprint('civitai', __name__, url_prefix='/api/civitai')

_NO_KEY = cp.CivitaiPublishError(
    'no_key', 'No Civitai API key configured - paste one in Settings > Scraping & sources.')


def _fail(e: cp.CivitaiPublishError, status=400):
    return jsonify({'ok': False, 'error': e.message, 'error_code': e.code}), status


def _status_for(code):
    return {'no_key': 400, 'invalid': 400, 'bad_ref': 400, 'not_found': 404,
            'run_missing': 404, 'dataset_missing': 404, 'image_missing': 404,
            'auth': 409, 'forbidden': 409, 'network': 409, 'civitai': 409,
            'link_missing': 409, 'checkpoint_missing': 409, 'ambiguous': 409,
            'not_a_lora': 409, 'no_version': 409, 'file_metadata_leak': 409,
            'not_safetensors': 409, 'invalid_image': 409}.get(code, 409)


@bp.get('/status')
def civitai_status():
    """What the modal says in its header: whether a key is configured, whose
    it is (best-effort, cached) and which domain links open on."""
    key = cp.api_key()
    return jsonify({'ok': True, 'has_key': bool(key),
                    'username': cp.whoami(key) if key else None,
                    'link_host': cp.link_host()})


# ---- the link store ----------------------------------------------------------

@bp.get('/links')
def civitai_links():
    """Every linked save of one dataset — the picker the image gesture falls
    back on when its row carries no checkpoint stamp (every picture made with a
    run's final save, and the pictures of a removed run)."""
    dataset_id = request.args.get('dataset_id', type=int)
    if dataset_id is None:
        return jsonify({'error': 'dataset_id is required'}), 400
    return jsonify({'ok': True, 'links': cp.links_for_dataset(dataset_id)})


@bp.get('/links/<int:record_id>/<int:step>')
def civitai_link(record_id, step):
    """The link of one save: `?filename=` names it exactly (a pill always
    can); without it the step's preferred link answers — the numbered save's
    when two files share the step."""
    link = cp.link_for(record_id, step, request.args.get('filename') or None)
    return jsonify({'ok': True, 'link': cp.link_payload(link)})


@bp.post('/links')
def civitai_link_create():
    """"Mark the page": {record_id, step, filename, url|model_id, version_id?}
    → that save now IS that Civitai version. Resolved against Civitai so a
    typo or someone else's checkpoint page is refused before it is remembered."""
    key = cp.api_key()
    if not key:
        return _fail(_NO_KEY)
    d = request.get_json(silent=True) or {}
    try:
        record_id, step = int(d.get('record_id')), int(d.get('step'))
    except (TypeError, ValueError):
        return jsonify({'ok': False, 'error': 'record_id and step are required'}), 400
    ref = d.get('url') or d.get('model_id')
    try:
        link, page = cp.link_checkpoint_to_page(
            record_id, step, ref, key, filename=d.get('filename'),
            version_id=d.get('version_id'))
    except cp.CivitaiPublishError as e:
        return _fail(e, _status_for(e.code))
    return jsonify({'ok': True, 'link': cp.link_payload(link), 'page': page})


@bp.post('/links/<int:link_id>/delete')
def civitai_link_delete(link_id):
    """Forget the link (nothing on Civitai is touched — the page stays)."""
    if not cp.delete_link(link_id):
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True})


# ---- the checkpoint → a model page ---------------------------------------------

@bp.get('/checkpoint/<int:record_id>/<int:step>/draft-defaults')
def civitai_draft_defaults(record_id, step):
    """Prefill for the "create a model page" form, derived from the run and
    the dataset; `file_error` names why the upload cannot happen, if it cannot."""
    try:
        out = cp.draft_defaults(record_id, step, request.args.get('filename') or None)
    except cp.CivitaiPublishError as e:
        return _fail(e, _status_for(e.code))
    return jsonify({'ok': True, **out})


@bp.post('/checkpoint/<int:record_id>/<int:step>/publish-model')
def civitai_publish_model(record_id, step):
    """Start the model-page job. The form is validated up front (a 400 here is
    cheaper than one three minutes into an upload); the upload itself runs in
    the background and the modal polls /jobs/<id>."""
    key = cp.api_key()
    if not key:
        return _fail(_NO_KEY)
    form = request.get_json(silent=True) or {}
    filename = form.get('filename') or None
    try:
        cp._validate_model_form(form)
        path, _rec = cp.checkpoint_file_for(record_id, step, filename)
        info = cp.inspect_checkpoint(path)
        if info['leaks']:
            raise cp.CivitaiPublishError(
                'file_metadata_leak',
                'The checkpoint\'s own metadata names this machine ('
                + ', '.join(info['leaks']) + ') - it was not uploaded.')
    except cp.CivitaiPublishError as e:
        return _fail(e, _status_for(e.code))
    app = current_app._get_current_object()
    job_id = cp.start_job(app, 'model', lambda progress: cp.publish_model(
        record_id, step, form, key, filename=filename, progress=progress, user_id=LOCAL_USER))
    return jsonify({'ok': True, 'job_id': job_id})


# ---- images → a post -----------------------------------------------------------

@bp.post('/images/publish')
def civitai_publish_images():
    """Start the post job: {image_ids:[…], link_id | (record_id, step), title?,
    publish?}. The target is a LINK — the modal names the one it showed; the
    (record_id, step) form resolves the step's preferred link. Without one the
    answer says what to do (mark the page) instead of failing later."""
    key = cp.api_key()
    if not key:
        return _fail(_NO_KEY)
    d = request.get_json(silent=True) or {}
    ids = d.get('image_ids')
    if not isinstance(ids, list) or not ids:
        return jsonify({'ok': False, 'error': 'image_ids must be a non-empty list'}), 400
    link = None
    if d.get('link_id') is not None:
        from ..models import CivitaiLink
        from ..extensions import db
        try:
            link = db.session.get(CivitaiLink, int(d['link_id']))
        except (TypeError, ValueError):
            link = None
    else:
        try:
            link = cp.link_for(int(d.get('record_id')), int(d.get('step')))
        except (TypeError, ValueError):
            return jsonify({'ok': False,
                            'error': 'link_id, or record_id and step, name the page'}), 400
    if link is None:
        return _fail(cp.CivitaiPublishError(
            'link_missing', 'This checkpoint is not linked to a Civitai model page yet - mark '
                            'the page first (paste its address) or create it.'), 409)
    title = d.get('title') if isinstance(d.get('title'), str) else None
    publish = d.get('publish', True) is not False
    link_id = link.id
    app = current_app._get_current_object()

    def work(progress):
        from ..models import CivitaiLink
        from ..extensions import db
        return cp.post_images(ids, db.session.get(CivitaiLink, link_id), key, title=title,
                              publish=publish, progress=progress, user_id=LOCAL_USER)

    return jsonify({'ok': True, 'job_id': cp.start_job(app, 'post', work),
                    'link': cp.link_payload(link)})


@bp.get('/jobs/<job_id>')
def civitai_job(job_id):
    """Poll: {state: running|done|error, phase, progress, result, error,
    error_code}. An unknown id (a restart forgot it) is said as such."""
    j = cp.job_status(job_id)
    if j is None:
        return jsonify({'ok': False, 'state': 'unknown',
                        'error': 'This job is no longer tracked (the server restarted?).'}), 404
    return jsonify({'ok': True, **j})
