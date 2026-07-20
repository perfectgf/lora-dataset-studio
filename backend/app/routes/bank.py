"""🗃️ Image bank API — triage a big unsorted folder before it becomes datasets.

All heavy passes (quality scan, face clustering, promotion) return 202 and run
in ONE background thread per bank; the UI polls GET /bank/<id> whose payload
embeds the live job. 409 = a job is already running on this bank.
"""
import logging
import os

from flask import Blueprint, current_app, jsonify, request, send_file

from ..config import LOCAL_USER
from ..models import BankImage
from ..services import bank_jobs
from ..services import image_bank_service as banks

logger = logging.getLogger(__name__)

bp = Blueprint('bank', __name__, url_prefix='/api')


def _app():
    return current_app._get_current_object()


@bp.get('/banks')
def banks_list():
    return jsonify({'banks': banks.list_banks(LOCAL_USER)})


@bp.post('/bank/create')
def bank_create():
    data = request.get_json(silent=True) or {}
    try:
        bank, added = banks.create_bank(LOCAL_USER, data.get('name'),
                                        data.get('folder'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'id': bank.id, 'added': added})


@bp.get('/bank/<int:bank_id>')
def bank_get(bank_id):
    payload = banks.bank_payload(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.delete('/bank/<int:bank_id>')
def bank_delete(bank_id):
    if not banks.delete_bank(LOCAL_USER, bank_id):
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True})


@bp.get('/bank/<int:bank_id>/images')
def bank_images(bank_id):
    args = request.args

    def _int(name):
        v = args.get(name)
        try:
            return int(v) if v not in (None, '') else None
        except ValueError:
            return None

    # subfolder is a STRING facet ('' = bank root), distinct from the int filters.
    subfolder = args.get('subfolder')
    # ids: the "show selected" VIEW — a comma-separated ordered id list that
    # overrides the facets. Present-but-empty means "an empty selection" (0 rows),
    # so distinguish None (no id view) from '' (empty view).
    ids_arg = args.get('ids')
    ids = None
    if ids_arg is not None:
        ids = []
        for tok in ids_arg.split(','):
            tok = tok.strip()
            if tok:
                try:
                    ids.append(int(tok))
                except ValueError:
                    pass
    payload = banks.list_images(
        LOCAL_USER, bank_id,
        status=args.get('status') or None,
        flag=args.get('flag') or None,
        cluster=_int('cluster'), group=_int('group'), style=_int('style'),
        semantic_group=_int('semantic_group'),
        subfolder=subfolder if subfolder is not None else None,
        search=args.get('search') or None,
        sort=args.get('sort') or None,
        res_bucket=args.get('res_bucket') or None,
        framing=args.get('framing') or None,
        ids=ids,
        offset=_int('offset') or 0, limit=_int('limit') or 200)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.get('/bank/<int:bank_id>/subfolders')
def bank_subfolders(bank_id):
    payload = banks.subfolders_payload(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


def _start(fn, *args, **kwargs):
    """Shared start-a-job envelope: 202 on launch, 409 when busy, 400/503 on
    validation errors."""
    try:
        fn(*args, **kwargs)
    except bank_jobs.BankJobBusy as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    return jsonify({'ok': True}), 202


@bp.post('/bank/<int:bank_id>/scan')
def bank_scan(bank_id):
    data = request.get_json(silent=True) or {}
    return _start(banks.start_scan, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.post('/bank/<int:bank_id>/faces')
def bank_faces(bank_id):
    return _start(banks.start_faces, _app(), LOCAL_USER, bank_id)


@bp.post('/bank/<int:bank_id>/score')
def bank_score(bank_id):
    """Aesthetic + NSFW + style scoring pass (bank-scoring extra). 202/409/503."""
    return _start(banks.start_score, _app(), LOCAL_USER, bank_id)


@bp.post('/bank/<int:bank_id>/semantic-dedup')
def bank_semantic_dedup(bank_id):
    """Stage-2 semantic near-duplicate pass (crops/variants) over the ✨ Score
    embeddings — CPU, no GPU. {threshold: 0.95} overrides the config for an ad-hoc
    re-tri without a re-scan. 202/409; 400 with a "run Score first" hint when no
    embeddings exist yet."""
    data = request.get_json(silent=True) or {}
    threshold = data.get('threshold')
    try:
        threshold = float(threshold) if threshold not in (None, '') else None
    except (TypeError, ValueError):
        threshold = None
    return _start(banks.start_semantic_dedup, _app(), LOCAL_USER, bank_id,
                  threshold=threshold)


@bp.post('/bank/<int:bank_id>/watermark')
def bank_watermark(bank_id):
    """Overlaid-watermark scan (Qwen3-VL). {rescan:true} re-checks scanned rows."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_watermark, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.post('/bank/<int:bank_id>/framing')
def bank_framing(bank_id):
    """Classify every non-rejected image by shot type (face/bust/body/back),
    reusing the dataset Qwen3-VL classifier. {rescan:true} re-classifies scanned
    rows. 202/409/503."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_framing, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.get('/bank/<int:bank_id>/coverage')
def bank_coverage(bank_id):
    """Read-only coverage advice (idea by @antonp): what the kept set leans on and
    what's thin for a good LoRA, from data the passes already computed. 404 when
    the bank is gone."""
    payload = banks.coverage(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/caption')
def bank_caption(bank_id):
    """Caption a selection (image_ids) or every non-rejected image, reusing the
    dataset caption engines. {force:true} re-captions already-captioned rows.
    202/409/503/400."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_caption, _app(), LOCAL_USER, bank_id,
                  ids=data.get('image_ids') or None, force=bool(data.get('force')))


@bp.post('/bank/<int:bank_id>/pipeline')
def bank_pipeline(bank_id):
    """Launch the chained "Launch all" triage pipeline. Body: {steps:[...],
    reject_flags:[...], resolve_dups:bool}. 202/409/400 — every step's own
    prerequisite is checked INSIDE the job (a missing extra skips that step,
    it never fails the launch)."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_pipeline, _app(), LOCAL_USER, bank_id,
                  steps=data.get('steps') or None,
                  reject_flags=data.get('reject_flags') or None,
                  resolve_dups=bool(data.get('resolve_dups')))


@bp.post('/bank/<int:bank_id>/promote')
def bank_promote(bank_id):
    data = request.get_json(silent=True) or {}
    try:
        dataset_id = int(data.get('dataset_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'dataset_id is required'}), 400
    return _start(banks.start_promote, _app(), LOCAL_USER, bank_id,
                  data.get('image_ids') or [], dataset_id)


@bp.post('/bank/<int:bank_id>/cancel')
def bank_cancel(bank_id):
    return jsonify({'ok': bank_jobs.cancel(bank_id)})


@bp.get('/bank/<int:bank_id>/dup-groups')
def bank_dup_groups(bank_id):
    try:
        offset = int(request.args.get('offset') or 0)
        limit = int(request.args.get('limit') or 50)
    except ValueError:
        offset, limit = 0, 50
    payload = banks.dup_groups_payload(LOCAL_USER, bank_id,
                                       offset=offset, limit=limit)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/dups/resolve')
def bank_dups_resolve(bank_id):
    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy') or 'best'
    if strategy not in ('best', 'first'):
        return jsonify({'error': 'strategy must be best or first'}), 400
    try:
        out = banks.resolve_dups(LOCAL_USER, bank_id, strategy=strategy,
                                 group=data.get('group'),
                                 keep_ids=data.get('keep_ids'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.get('/bank/<int:bank_id>/semantic-dup-groups')
def bank_semantic_dup_groups(bank_id):
    try:
        offset = int(request.args.get('offset') or 0)
        limit = int(request.args.get('limit') or 50)
    except ValueError:
        offset, limit = 0, 50
    payload = banks.semantic_dup_groups_payload(LOCAL_USER, bank_id,
                                                offset=offset, limit=limit)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/semantic-dups/resolve')
def bank_semantic_dups_resolve(bank_id):
    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy') or 'best'
    if strategy not in ('best', 'first'):
        return jsonify({'error': 'strategy must be best or first'}), 400
    try:
        out = banks.resolve_semantic_dups(LOCAL_USER, bank_id, strategy=strategy,
                                          group=data.get('group'),
                                          keep_ids=data.get('keep_ids'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/images/status')
def bank_images_status(bank_id):
    data = request.get_json(silent=True) or {}
    try:
        n = banks.set_status(LOCAL_USER, bank_id, data.get('ids') or [],
                             data.get('status'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'changed': n})


@bp.post('/bank/<int:bank_id>/apply-flags')
def bank_apply_flags(bank_id):
    data = request.get_json(silent=True) or {}
    try:
        out = banks.apply_flags(LOCAL_USER, bank_id, data.get('flags') or [])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'rejected': out})


def _curation_filters(data):
    """The shared candidate-pool filters for the curation selectors — the same
    facets as the grid (status ∩ flag ∩ cluster ∩ style ∩ subfolder ∩ search),
    read out of a JSON body. Unknown keys (e.g. the grid's ``sort``) are ignored."""
    def _int(name):
        v = data.get(name)
        try:
            return int(v) if v not in (None, '') else None
        except (TypeError, ValueError):
            return None

    subfolder = data.get('subfolder')
    return {
        'status': data.get('status') or None,
        'flag': data.get('flag') or None,
        'cluster': _int('cluster'),
        'style': _int('style'),
        # '' is a meaningful subfolder (bank root); '__all__'/None mean "no scope".
        'subfolder': subfolder if subfolder not in (None, '__all__') else None,
        'search': data.get('search') or None,
    }


@bp.post('/bank/<int:bank_id>/select-diverse')
def bank_select_diverse(bank_id):
    """Farthest-point selection of the N most VARIED images in the current filter,
    reusing the ✨ Score embeddings (no GPU). Returns the chosen ids for the UI to
    check — never mutates. 400 with a "run Score first" hint when unscored."""
    data = request.get_json(silent=True) or {}
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    try:
        out = banks.select_diverse(LOCAL_USER, bank_id, n=n,
                                   filters=_curation_filters(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/select-similar')
def bank_select_similar(bank_id):
    """Rank the current filter by CLIP similarity to a reference bank image
    ({ref_id}); returns the top-N ids (or everything ≥ {min_score}) for the UI to
    check. Reuses the ✨ Score embeddings (no GPU). 400 when unscored / bad ref."""
    data = request.get_json(silent=True) or {}
    try:
        ref_id = int(data.get('ref_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'ref_id is required'}), 400
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    min_score = data.get('min_score')
    try:
        min_score = float(min_score) if min_score not in (None, '') else None
    except (TypeError, ValueError):
        min_score = None
    try:
        out = banks.select_similar(LOCAL_USER, bank_id, ref_id, n=n,
                                   min_score=min_score,
                                   filters=_curation_filters(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/delete-rejected')
def bank_delete_rejected(bank_id):
    """Destructive: delete the SOURCE files of every rejected image from disk
    (OS trash when send2trash is present, hard delete otherwise) and drop their
    rows. The ONLY bank action that writes to the source folder — the front-end
    gates it behind a type-DELETE confirmation."""
    try:
        out = banks.delete_rejected(LOCAL_USER, bank_id)
    except ValueError:
        return jsonify({'error': 'not found'}), 404
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 409
    return jsonify({'ok': True, **out})


def _row_or_404(bank_id, image_id):
    bank = banks.get_bank(LOCAL_USER, bank_id)
    if not bank:
        return None, None
    row = BankImage.query.filter_by(id=image_id, bank_id=bank_id).first()
    return bank, row


@bp.get('/bank/<int:bank_id>/thumb/<int:image_id>')
def bank_thumb(bank_id, image_id):
    bank, row = _row_or_404(bank_id, image_id)
    if not bank or not row:
        return jsonify({'error': 'not found'}), 404
    tpath = banks.ensure_thumb(bank, row)
    if not tpath:
        return jsonify({'error': 'unreadable'}), 404
    return send_file(tpath, mimetype='image/webp', max_age=3600)


@bp.get('/bank/<int:bank_id>/file/<int:image_id>')
def bank_file(bank_id, image_id):
    bank, row = _row_or_404(bank_id, image_id)
    if not bank or not row:
        return jsonify({'error': 'not found'}), 404
    path = banks.abs_image_path(bank, row)
    if not path or not os.path.isfile(path):
        return jsonify({'error': 'file missing'}), 404
    return send_file(path, max_age=0)
