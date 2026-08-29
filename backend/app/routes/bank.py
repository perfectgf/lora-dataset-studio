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
from ..services import bank_jobs, dataset_activity
from ._common import _map_error, _require_no_stalled_comfyui
# The ✨ improve preflight answers the SAME three engine exceptions here as in the
# dataset lane, and its 409 body is what itemizes the missing assets and starts
# their download. Imported rather than re-derived: a second copy of that mapping
# is a second thing to keep in step with the installer.
from .datasets import _improve_engine_error
from ..services import bank_filter_translator as translator
from ..services import image_bank_service as banks

logger = logging.getLogger(__name__)

bp = Blueprint('bank', __name__, url_prefix='/api')


def _app():
    return current_app._get_current_object()


def _busy(e):
    """The ONE shape of a "this bank is occupied" refusal.

    `error` stays what it always was (an English sentence, for anything that
    only knows how to print a message), but `busy_kind` is the machine-readable
    half the UI actually needs: it names the pass that holds the bank, so the
    click can be refused in the user's vocabulary — "✨ Score pass is running —
    137/412, press Stop above" — instead of echoing our sentence back at them.
    It matters that this rides on the 409 itself: the refusal often arrives
    BEFORE the first 2 s progress poll, so at that instant the response body is
    the only thing that knows which pass is in the way."""
    return jsonify({'error': str(e), 'busy_kind': e.kind}), 409


@bp.errorhandler(bank_jobs.BankJobBusy)
def _busy_after_service_fence(e):
    """Map service-level lease races to the same 409 as the advisory guard."""
    return _busy(e)


_BUSY_WRITE_EXEMPT_ENDPOINTS = frozenset({
    # Read-only preview despite using POST (it carries unsaved thresholds).
    'bank.bank_flag_preview',
    # Read-only curation queries. They may warm/read the Score cache, but never
    # mutate Bank rows or source files.
    'bank.bank_select_diverse',
    'bank.bank_select_balanced',
    'bank.bank_select_similar',
    'bank.bank_search_text',
    # Read-only host integration: opening Explorer/Finder never mutates the Bank.
    'bank.bank_open_source_folder',
    # The one write that must stay reachable precisely while a job is live.
    'bank.bank_cancel',
})


@bp.before_request
def _guard_reserved_bank_writes():
    """Keep a building destination readable/cancellable but immutable.

    Bank jobs reserve every Bank whose rows or files they own.  Most heavy-pass
    services already check their own source slot, but lightweight synchronous
    routes (status flips in particular) historically did not.  Enforce the
    reservation once at the HTTP boundary so a destination cannot be edited or
    deleted while its copy is still being assembled.
    """
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return None
    if request.endpoint in _BUSY_WRITE_EXEMPT_ENDPOINTS:
        return None
    if request.endpoint == 'bank.bank_relocate':
        # The first relocation POST is a read-only filesystem preview. Only the
        # explicit confirmation changes source_path and must wait for the job.
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not bool(body.get('confirm')):
            return None
    bank_id = (request.view_args or {}).get('bank_id')
    if bank_id is None or not bank_jobs.running(bank_id):
        return None
    snap = bank_jobs.get(bank_id) or {}
    return _busy(bank_jobs.BankJobBusy(snap.get('kind') or 'background'))


@bp.get('/banks')
def banks_list():
    """Every bank + its card previews. ?dataset_id=<id> additionally embeds each
    bank's promotable count for that dataset, so the dataset-side bank chooser
    opens on ONE request instead of one per bank. An unknown/junk dataset_id
    simply omits the field (never a 400: the list itself is still valid).

    A bank points at a LIVE folder, and this route used to re-walk EVERY bank's
    folder before answering — so merely navigating to the page cost a full
    inventory of every image in the library (measured on a real one of 8 banks /
    86 493 images: 690-1 190 ms per load, 1 341-1 777 ms on the reporter's
    instance). A GET that expensive punishes passing through.

    The walk now happens where it is worth its price: automatically when a bank
    is OPENED (``GET /bank/<id>``, cooldown-limited, forced on open), and on
    demand here with ``?rescan=1`` behind the list's 🔄 button. Without it the
    list still probes each folder's EXISTENCE (one syscall per bank) and reports
    how old the last walk is, so the page can say its counts may lag instead of
    showing stale ones silently. ``folder_sync`` carries both shapes."""
    rescan = request.args.get('rescan') == '1'
    sync = (banks.refresh_banks(LOCAL_USER, force=True) if rescan
            else banks.folder_sync_state(LOCAL_USER))
    rows = banks.list_banks(
        LOCAL_USER, dataset_id=request.args.get('dataset_id') or None)
    for row in rows:
        row['folder_sync'] = sync.get(row['id'])
    return jsonify({'banks': rows, 'rescanned': rescan})


@bp.post('/bank/create')
def bank_create():
    data = request.get_json(silent=True) or {}
    try:
        bank, added = banks.create_bank(LOCAL_USER, data.get('name'),
                                        data.get('folder'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    # Nested folders mean two banks over the same files: harmless while triaging
    # (statuses are per bank), but 🗑 Delete rejected in one amputates the other.
    # Say it now, once, rather than at the destructive click only.
    return jsonify({'ok': True, 'id': bank.id, 'added': added,
                    'overlaps': banks.overlapping_banks(LOCAL_USER, bank.id)})


@bp.post('/bank/from-dataset')
def bank_from_dataset():
    """Reverse of promote: build a NEW bank from a dataset's kept images, under a
    name the user chooses. Copies the files so the two never share (curating one
    would otherwise mutate the other). 202 + background job, like the other passes."""
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({'error': 'request body must be an object'}), 400
    preserve_analysis = data.get('preserve_analysis', True)
    if not isinstance(preserve_analysis, bool):
        return jsonify({'error': 'preserve_analysis must be a boolean'}), 400
    try:
        dataset_id = dataset_activity.normalize_dataset_id(data.get('dataset_id'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    try:
        bank_id = banks.start_dataset_import(_app(), LOCAL_USER,
                                             dataset_id, data.get('name'),
                                             preserve_analysis=preserve_analysis)
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except dataset_activity.DatasetActivityBusy as e:
        return jsonify({'error': str(e), 'busy_kind': 'bank_export'}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    # the id rides back so the UI can jump straight to the bank being filled
    return jsonify({'ok': True, 'id': bank_id}), 202


@bp.post('/bank/scrape-import')
def bank_scrape_import():
    """🕸 Scrape → BANK — the scraper's second destination, next to the dataset one.

    Body: {items:[{url,title}], bank_id?} to APPEND to an existing bank (resume),
    or {items, name} to create one. Synchronous like the dataset outlet (the same
    per-request cap bounds it), and it stores what it downloaded: the resolution /
    ratio / near-duplicate verdicts belong to the bank's own passes, not to the
    download. 409 when a pass already owns the target bank."""
    data = request.get_json(silent=True) or {}
    raw_bank_id = data.get('bank_id')
    bank_id = None
    if raw_bank_id is not None:
        try:
            bank_id = int(raw_bank_id)
        except (TypeError, ValueError):
            return jsonify({'error': 'bank_id must be a number'}), 400
    try:
        res = banks.scrape_import_to_bank(LOCAL_USER, data.get('items'),
                                          bank_id=bank_id, name=data.get('name'))
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **res})


@bp.get('/bank/<int:bank_id>')
def bank_get(bank_id):
    """The workspace payload. The source folder is re-walked first so images
    added to it show up while the bank is open — cooldown-limited, because this
    route is ALSO the 2 s job poll. ?refresh=1 forces the walk (the workspace
    sends it when it opens the bank). The outcome rides in ``folder_sync``."""
    sync = banks.refresh_bank(LOCAL_USER, bank_id,
                              force=request.args.get('refresh') == '1')
    payload = banks.bank_payload(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    payload['folder_sync'] = sync
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/open-source-folder')
def bank_open_source_folder(bank_id):
    """Open the owned Bank's recorded source; the client supplies no path."""
    try:
        path = banks.open_bank_source_folder(LOCAL_USER, bank_id)
    except banks.BankSourceFolderUnavailable as exc:
        return jsonify({'error': str(exc)}), 409
    except Exception:
        current_app.logger.exception('could not open Bank source folder')
        return jsonify({'error': 'could not open the bank source folder'}), 500
    if path is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True})


@bp.post('/bank/<int:bank_id>/flag-preview')
def bank_flag_preview(bank_id):
    """How many images each flag WOULD hold at the thresholds in the body —
    the live effect readout under the Bank's 🎚 threshold controls.

    Read-only and cheap: verdicts are recomputed from persisted raw scores, so
    this is one COUNT per flag, the same queries the workspace payload already
    runs. Nothing is saved — the user still has to press Save.

    POST (not GET) because it carries the UNSAVED candidate thresholds, exactly
    like /settings/prompt-preview carries unsaved prompt text. A junk body
    degrades to "the saved thresholds" instead of 400: a preview that answers
    'error' while you are mid-keystroke is worth less than one that answers with
    what is currently in effect."""
    body = request.get_json(silent=True) or {}
    overrides = body.get('thresholds')
    out = banks.flag_preview(LOCAL_USER, bank_id,
                             overrides if isinstance(overrides, dict) else None)
    if out is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(out)


@bp.delete('/bank/<int:bank_id>')
def bank_delete(bank_id):
    try:
        deleted = banks.delete_bank(LOCAL_USER, bank_id)
    except bank_jobs.BankJobBusy as e:
        # The blueprint guard covers an already-live reservation; this catches
        # the narrower race where a job acquires the Bank after that check.
        return _busy(e)
    if not deleted:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True})


@bp.post('/bank/<int:bank_id>/relocate')
def bank_relocate(bank_id):
    """Point a bank at a new folder after the user moved it (another disk, a
    rename). Two-step ON PURPOSE: {folder} alone only REPORTS how many of the
    bank's files are in there, {folder, confirm: true} applies it. 400 when the
    folder holds none of them (that is a different folder, not a moved one),
    The preview stays readable while a pass runs; only a confirmed write returns
    409. Nothing is ever deleted — a partial match keeps every row and its
    analysis."""
    data = request.get_json(silent=True) or {}
    confirm = bool(data.get('confirm'))
    try:
        if confirm:
            out = banks.relocate_bank(
                LOCAL_USER, bank_id, data.get('folder'), confirm=True)
        else:
            # relocate_bank also owns the write-side busy guard. Calling its
            # read-only primitive directly keeps the preview usable while a job
            # owns the Bank without weakening confirmation.
            out = banks.relocate_preview(LOCAL_USER, bank_id, data.get('folder'))
            out['needs_confirm'] = out['missing'] > 0
            out['applied'] = False
            if out['total'] and not out['found']:
                raise banks.BankRelocateMismatch(
                    'none of this bank\'s '
                    f"{out['total']} image(s) are in that folder — it does not look "
                    'like this bank. Pick the folder that CONTAINS the images '
                    '(the one you moved), not its parent.', out)
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except banks.BankRelocateMismatch as e:
        return jsonify({'error': str(e), **e.preview}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/forget-missing')
def bank_forget_missing(bank_id):
    """Drop the rows whose source file is no longer in the bank's folder — the
    folder-sync warning's OTHER remedy: not a folder that moved (that is 📦
    Move folder…), but files really deleted from it (a downloader that cleans
    up its own intermediates, a sync client, a by-hand tidy). Two-step ON
    PURPOSE: {} only REPORTS a freshly-walked count, {confirm: true} deletes
    those rows. Files on disk are never touched, and an unavailable or
    unreadable folder is refused outright — an unplugged drive must never be
    able to erase a triage. The preview stays readable while a pass runs; only
    a confirmed write returns 409."""
    data = request.get_json(silent=True) or {}
    try:
        if data.get('confirm'):
            out = {**banks.forget_missing(LOCAL_USER, bank_id), 'applied': True}
        else:
            out = {**banks.forget_missing_preview(LOCAL_USER, bank_id),
                   'applied': False}
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except banks.BankFolderUnavailable as e:
        return jsonify({'error': str(e)}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    return jsonify({'ok': True, **out})


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
        exclude=args.get('exclude') or None,
        # 🏷️ chips ticked off one image's caption. Its OWN key, never folded into
        # `search` or `exclude`: two features sharing one payload key is how a
        # filter silently ate a sibling's field once already.
        tags=args.get('tags') or None,
        sort=args.get('sort') or None,
        res_bucket=args.get('res_bucket') or None,
        framing=args.get('framing') or None,
        origin=args.get('origin') or None,
        # DEDICATED keys. Neither reuses `q`/`exclude`/`push_down` (the text
        # lane) nor `flag` (the quality lane): a facet that shares a query key
        # with another facet silently narrows the other one.
        medium=args.get('medium') or None,
        angle=args.get('angle') or None,
        ids=ids,
        # ids_only=1 answers {'ids': [...]} for the WHOLE filter in one request —
        # what ▶ Review and "Select all in filter" actually need. Same filters,
        # same sort, same route, so the two answers can never disagree about what
        # the current filter contains.
        ids_only=args.get('ids_only') == '1',
        offset=_int('offset') or 0, limit=_int('limit') or 200)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.get('/bank/<int:bank_id>/facets')
def bank_facets(bank_id):
    """The chip counters for the CURRENT filter — same query keys as /images.

    Its own route rather than a slice of the workspace payload: that payload is
    also the 2 s job poll, and hanging the filter state off it would re-measure
    every facet twice a second for a question only a filter change can change
    the answer to. Read-only, and each facet is counted with its OWN value
    lifted (see facet_counts)."""
    args = request.args

    def _int(name):
        v = args.get(name)
        try:
            return int(v) if v not in (None, '') else None
        except ValueError:
            return None

    subfolder = args.get('subfolder')
    out = banks.facet_counts(
        LOCAL_USER, bank_id,
        status=args.get('status') or None,
        flag=args.get('flag') or None,
        cluster=_int('cluster'), group=_int('group'), style=_int('style'),
        semantic_group=_int('semantic_group'),
        subfolder=subfolder if subfolder is not None else None,
        search=args.get('search') or None,
        exclude=args.get('exclude') or None,
        tags=args.get('tags') or None,
        res_bucket=args.get('res_bucket') or None,
        framing=args.get('framing') or None,
        origin=args.get('origin') or None,
        medium=args.get('medium') or None,
        angle=args.get('angle') or None)
    if out is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(out)


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
        return _busy(e)
    except dataset_activity.DatasetActivityBusy as e:
        return jsonify({'error': str(e), 'busy_kind': 'bank_import'}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    return jsonify({'ok': True}), 202


def _scope(data):
    """The two keys every pass dialog can send: WHERE to run.

    {statuses:["keep"|"pending"|"reject"]} and {image_ids:[...]}. Both optional;
    a body without them is the request that shipped before the dialogs existed,
    byte for byte. Validation lives in the service (one definition, 400 on a bad
    value) — this only reads the body."""
    return {'statuses': data.get('statuses') or None,
            'ids': data.get('image_ids') or None}


@bp.post('/bank/<int:bank_id>/scan')
def bank_scan(bank_id):
    """Quality pass. {rescan: true} walks every image again; {regroup: true} asks
    for the duplicate grouping whatever the scan finds — that is the 🎚 panel's
    "↻ Re-group duplicates", whose pool is empty on an already-scanned bank."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_scan, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')),
                  regroup=bool(data.get('regroup')), **_scope(data))


@bp.post('/bank/<int:bank_id>/faces')
def bank_faces(bank_id):
    """Face embeddings + person clustering. Takes NO scope on purpose: the
    clusters are one numbering of the whole bank (400 if one is sent anyway)."""
    return _start(banks.start_faces, _app(), LOCAL_USER, bank_id)


@bp.post('/bank/<int:bank_id>/score')
def bank_score(bank_id):
    """Aesthetic + NSFW + style scoring pass (bank-scoring extra). 202/409/503.

    {rescore: true} throws the cached embeddings away and recomputes everything —
    the ✨ Score button itself always ran, and still runs, the complete pass (it
    just skips what is already computed)."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_score, _app(), LOCAL_USER, bank_id,
                  rescore=bool(data.get('rescore')))


@bp.get('/bank/<int:bank_id>/semantic-engine')
def bank_semantic_engine_get(bank_id):
    """Selected image/text space plus the usable cache coverage for this Bank."""
    payload = banks.semantic_engine_info(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.patch('/bank/<int:bank_id>/semantic-engine')
def bank_semantic_engine_set(bank_id):
    """Persist ``clip`` or ``siglip2`` for this Bank without deleting either cache."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'request body must be an object'}), 400
    try:
        payload = banks.set_semantic_engine(LOCAL_USER, bank_id, data.get('engine'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **payload})


@bp.post('/bank/<int:bank_id>/semantic-index')
def bank_semantic_index(bank_id):
    """Build/resume the selected semantic image cache.

    This is deliberately a whole-Bank partition like ✨ Score: semantic
    near-duplicate grouping cannot mix ids produced from two partial spaces.
    ``rescan`` only discards the selected engine's cache; the other engine and
    the aesthetic CLIP cache are preserved.
    """
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({'error': 'request body must be an object'}), 400
    return _start(banks.start_semantic_index, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.post('/bank/<int:bank_id>/semantic-dedup')
def bank_semantic_dedup(bank_id):
    """Stage-2 near-duplicates over the Bank's selected semantic engine.

    CPU-only after the engine cache exists. ``threshold`` remains an explicit
    per-run override; default calibration is engine-specific.
    """
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({'error': 'request body must be an object'}), 400
    # Keep the raw value: the service owns the single finite [0, 1] validation
    # used by HTTP, pipeline and direct callers. Invalid input must never fall
    # back silently to a permissive default.
    threshold = data.get('threshold')
    # ``force`` skips the "nothing has changed since the last run" shortcut. The
    # shortcut reads state, not pixels, so a file swapped under an unchanged
    # cache is exactly what it cannot see — this is the way out.
    return _start(banks.start_semantic_dedup, _app(), LOCAL_USER, bank_id,
                  threshold=threshold, force=bool(data.get('force')))


@bp.post('/bank/<int:bank_id>/watermark')
def bank_watermark(bank_id):
    """Overlaid-watermark scan (detector or vision). {rescan:true} re-checks
    scanned rows; {limit:N} is the launch window's "try on a sample first" —
    the first N of the scope by id, deterministic (the 🔤 scan's dial)."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_watermark, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')), limit=data.get('limit'),
                  **_scope(data))


@bp.get('/bank/<int:bank_id>/watermark/preview')
def bank_watermark_preview(bank_id):
    """The 🚩 launch window's result gallery: watermark-family flagged pages
    (not 🔤 text-flagged) with their zones, oldest-id first."""
    try:
        limit = int(request.args.get('limit') or 24)
    except (TypeError, ValueError):
        limit = 24
    payload = banks.watermark_preview(LOCAL_USER, bank_id, limit)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.get('/bank/<int:bank_id>/text/preview')
def bank_text_preview(bank_id):
    """The 🔤 launch window's result gallery: text-flagged pages with their
    zones, oldest-id first (the sample's own deterministic order)."""
    try:
        limit = int(request.args.get('limit') or 24)
    except (TypeError, ValueError):
        limit = 24
    payload = banks.text_preview(LOCAL_USER, bank_id, limit)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/text')
def bank_text_scan(bank_id):
    """🔤 Burned-in text scan (RapidOCR, CPU). Folds the text zones it finds
    into the watermark mask channel so 🧽 Inpaint can repaint them.
    {rescan:true} re-reads scanned rows; dismissed rows are never re-examined.
    {limit:N} is the launch window's "try on a sample first" — only the first
    N rows of the scope, deterministic, so redo re-reads the same sample.
    Same {statuses}/{image_ids} scope dials as every other pass. 202/409/400/503."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_text_scan, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')), limit=data.get('limit'),
                  **_scope(data))


@bp.get('/bank/<int:bank_id>/watermark/levels')
def bank_watermark_levels(bank_id):
    """Where each cleaning level stands: flagged / croppable / left to inpaint /
    already cropped / already inpainted / dismissed / needing a re-scan."""
    payload = banks.watermark_levels(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/watermark/crop')
def bank_watermark_crop(bank_id):
    """Level 1 — crop away the border-band watermarks (CPU/PIL, invents no pixel).
    The source folder is never written to: the crop lands in the bank's own
    working copy. 202/409/400.

    Takes the same {statuses}/{image_ids} scope as the other passes — this one
    produces new image files, so being able to aim it is worth more here than
    anywhere else. A body without them is the request that shipped before, byte
    for byte."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_watermark_crop, _app(), LOCAL_USER, bank_id,
                  **_scope(data))


@bp.post('/bank/<int:bank_id>/watermark/inpaint')
def bank_watermark_inpaint(bank_id):
    """Level 2 — repaint what is still flagged. {method:'auto'|'lama'|'klein'}.
    202/409/400/503 (503 carries the actionable reason: engine missing, GPU busy).

    Takes the same {statuses}/{image_ids} scope as the other passes. A body
    without them repaints exactly what it repainted before."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_watermark_inpaint, _app(), LOCAL_USER, bank_id,
                  method=data.get('method') or 'auto',
                  target=data.get('target') or 'all', **_scope(data))


@bp.post('/bank/<int:bank_id>/watermark/undo')
def bank_watermark_undo(bank_id):
    """Drop the cleaned versions of {image_ids} (empty = all) and re-flag them.
    Synchronous — it only deletes blobs we made."""
    data = request.get_json(silent=True) or {}
    try:
        n = banks.undo_watermark_clean(LOCAL_USER, bank_id,
                                       data.get('image_ids') or None)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'restored': n})


@bp.post('/bank/<int:bank_id>/image/<int:image_id>/crop')
def bank_image_crop(bank_id, image_id):
    """✂ Cut ONE bank image to {x, y, w, h}, in the pixels of the image as the
    editor displayed it — the Bank's half of the dataset route of the same name,
    with the same body and the same meaning.

    Synchronous: it only writes one blob of ours. Nothing is resampled (a bank is
    upstream of the training resolution — see the service docstring).
    400 = unusable box · 404 = unknown bank/image · 409 = a pass holds the bank."""
    data = request.get_json(silent=True) or {}
    try:
        result = banks.crop_image(LOCAL_USER, bank_id, image_id,
                                  data['x'], data['y'], data['w'], data['h'])
    except KeyError:
        return jsonify({'error': 'x, y, w and h are required'}), 400
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


@bp.post('/bank/<int:bank_id>/edits/revert')
def bank_edits_revert(bank_id):
    """Drop the bank-side ✂ crop / ✨ improve of {image_ids} (empty = all) and go
    back to the image the bank started from. Synchronous — like the watermark
    undo, it only deletes blobs we made ourselves."""
    data = request.get_json(silent=True) or {}
    try:
        result = banks.revert_edits(LOCAL_USER, bank_id,
                                    data.get('image_ids') or None)
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **result})


@bp.post('/bank/<int:bank_id>/improve')
def bank_improve(bank_id):
    """✨ Upscale & improve the images in scope, in the Bank itself.

    {engine: 'klein'|'seedvr2'} — absent = the improve.engine setting, exactly
    like the dataset lane. Takes the same {statuses}/{image_ids} scope as every
    other pass; this one spends GPU-minutes per image, so aiming it matters more
    here than anywhere else.

    202 on launch · 409 when the bank is busy OR an engine is missing (the body
    then carries the itemized assets and the auto-download that was started, the
    same structured refusal the dataset routes return) · 400 on an empty pool ·
    503 when the GPU is already taken by a training or a vision pass."""
    data = request.get_json(silent=True) or {}
    gate = _require_no_stalled_comfyui()
    if gate:
        return gate
    engine = (data.get('engine') or '').strip() or None
    try:
        banks.start_improve(_app(), LOCAL_USER, bank_id, engine=engine,
                            **_scope(data))
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        # The engine preflight misses (Klein nodes/models, SeedVR2 models) keep
        # their own structured 409 — that body is what carries the itemized list
        # and starts the download, so flattening it here would cost the user the
        # one thing that makes the refusal actionable.
        engine_error = _improve_engine_error(e)
        if engine_error:
            return engine_error
        raise
    return jsonify({'ok': True}), 202


@bp.put('/bank/<int:bank_id>/image/<int:image_id>/watermark-regions')
def bank_image_watermark_regions(bank_id, image_id):
    """Replace one flagged image's hand-drawn watermark mask — the Bank's half of
    the dataset route of the same name (same payload, same validator, same
    meaning). {regions: null} drops the override and goes back to the detected
    box; {regions: []} is an explicit empty mask (nothing gets repainted).
    400 = illegal mask · 404 = unknown bank/image · 409 = no longer flagged."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or 'regions' not in data:
        return jsonify({'error': 'regions is required'}), 400
    try:
        result = banks.set_watermark_regions(LOCAL_USER, bank_id, image_id,
                                             data['regions'])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        # A 409 that is NOT the "bank is occupied" refusal — it is a state
        # conflict on one row (already cleaned/dismissed elsewhere, or by another
        # tab). Labelled like the busy one so a caller can tell them apart
        # instead of matching on our sentence.
        return jsonify({'error': str(e), 'conflict': 'not_flagged'}), 409
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


@bp.post('/bank/<int:bank_id>/watermark/dismiss')
def bank_watermark_dismiss(bank_id):
    """Rule {image_ids} NOT watermarked — they leave both cleaning levels and are
    never re-flagged by a later scan."""
    data = request.get_json(silent=True) or {}
    try:
        n = banks.dismiss_watermarks(LOCAL_USER, bank_id, data.get('image_ids'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'dismissed': n})


@bp.post('/bank/<int:bank_id>/framing')
def bank_framing(bank_id):
    """Classify every non-rejected image by shot type (face/bust/body/back),
    reusing the dataset Qwen3-VL classifier. {rescan:true} re-classifies scanned
    rows. 202/409/503."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_framing, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')), **_scope(data))


@bp.post('/bank/<int:bank_id>/medium')
def bank_medium(bank_id):
    """Classify images by medium from the CLIP vectors cached by ✨ Score.

    The calibrated prompt set remains CLIP-specific even when this Bank selects
    SigLIP2 for its general semantic tools. ``rescan`` recomputes verdicts.
    """
    data = request.get_json(silent=True) or {}
    return _start(banks.start_medium, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')), **_scope(data))


@bp.post('/bank/<int:bank_id>/angles')
def bank_angles(bank_id):
    """Measure the head ANGLE of the images that were face-scanned before the
    pass persisted it — an opt-in backfill, never automatic. Re-runs the face
    detector on those rows ONLY, writes nothing but the yaw, and leaves the
    person clusters untouched. 202/409/503; 400 when there is nothing to
    backfill."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_faces, _app(), LOCAL_USER, bank_id,
                  angles_only=True, **_scope(data))


@bp.get('/bank/<int:bank_id>/activity')
def bank_activity(bank_id):
    """The live job alone — what the progress banner and its Stop button need.

    Deliberately NOT the workspace payload: that one runs ~60 bank-wide
    aggregates and is polled every 2 s while a pass runs, which is exactly when
    it is slowest. This route reads one indexed row and an in-memory job
    snapshot, so the banner keeps arriving on a 50 000-image bank. No source
    folder re-walk either — this is the poll, not the open."""
    payload = banks.bank_activity(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


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
    {vocabulary} picks the register ('explicit'|'clinical'|'safe') and {length} the
    size preset ('concise'|'detailed', absent = standard) — same lane as the dataset
    caption; invalid → 400.

    {backend} ('auto'|'joycaption'|'ollama') and {ollama_model} override the engine
    and the vision model for THIS run only, without touching the global settings —
    same keys and same validation as the dataset caption options. {statuses} picks
    the scope: any combination of "keep" / "pending" / "reject". "reject" aims the
    pass at the bin — reachable only by asking for it by name, never part of the
    default. Every one of these keys is optional, and a body without them runs
    exactly the pass that ran before they existed.

    {force:true} SPARES the captions a human wrote or corrected (caption_origin =
    'asserted'); {include_asserted:true} is the explicit opt-out that rewrites those
    too. Its default is False and it is meaningless without force — the destructive
    reading of this endpoint is never the one you get by omitting a key.
    202/409/503/400."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_caption, _app(), LOCAL_USER, bank_id,
                  ids=data.get('image_ids') or None, force=bool(data.get('force')),
                  vocabulary=data.get('vocabulary') or None,
                  length=data.get('length') or None,
                  backend=data.get('backend') or None,
                  ollama_model=data.get('ollama_model') or None,
                  statuses=data.get('statuses') or None,
                  include_asserted=bool(data.get('include_asserted')))


@bp.post('/bank/<int:bank_id>/image/<int:image_id>/caption/preview')
def bank_image_caption_preview(bank_id, image_id):
    """🧪 Caption Lab: run ONE candidate config on ONE bank image and return the
    caption WITHOUT writing it. The Bank's half of the dataset route of the same name,
    with the same body ({backend, ollama_model, vocabulary, length, instructions}) and
    the same meaning — both call one shared bench (face_dataset_service.
    preview_caption_path), so a candidate cannot mean two things.

    Synchronous, and it holds the bank for its duration under the 'caption' kind: the
    bench owns the GPU while it runs, and ▶ Stop / POST /bank/<id>/cancel aborts it at
    the image boundary like any pass.
    400 = bad config · 404 = unknown bank/image · 409 = a pass holds the bank · 503 = GPU
    held by training or another vision task."""
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    elif not isinstance(data, dict):
        return jsonify({'error': 'JSON body must be an object'}), 400
    # 404 for an unknown BANK, like the dataset twin (datasets.py checks the container
    # first). Without this the service's ValueError('bank not found') surfaced as a 400,
    # so the same mistake answered 404 on one surface and 400 on the other.
    if banks.get_bank(LOCAL_USER, bank_id) is None:
        return jsonify({'error': 'not found'}), 404
    try:
        result = banks.preview_caption(
            LOCAL_USER, bank_id, image_id,
            backend=data.get('backend'), ollama_model=data.get('ollama_model', ''),
            vocabulary=data.get('vocabulary'), length=data.get('length'),
            instructions=data.get('instructions'))
    except bank_jobs.BankJobBusy as e:
        # BEFORE _map_error, and not for the reason one would guess: BankJobBusy derives
        # from Exception, NOT RuntimeError, so _map_error does not recognise it at all and
        # would RE-RAISE it. Caught here so the refusal is a 409 carrying `busy_kind` —
        # the machine-readable half the UI refuses the click in the user's words with.
        return _busy(e)
    except Exception as e:  # noqa: BLE001 — _map_error re-raises what it cannot map
        return _map_error(e)
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


@bp.put('/bank/<int:bank_id>/image/<int:image_id>/caption')
def bank_image_caption(bank_id, image_id):
    """Write ONE bank image's caption by hand. {caption: "..."} — an empty string
    clears it. The text lands stamped 'asserted', so a forced 🔄 Re-caption spares it
    unless the user ticks the include-asserted opt-out.

    Synchronous: it writes one row of ours, like the crop and the watermark undo.
    400 = bad body · 404 = unknown bank/image · 409 = a pass holds the bank."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'JSON body must be an object'}), 400
    if banks.get_bank(LOCAL_USER, bank_id) is None:
        return jsonify({'error': 'not found'}), 404
    try:
        result = banks.set_image_caption(LOCAL_USER, bank_id, image_id,
                                         data.get('caption'))
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


@bp.get('/bank/<int:bank_id>/scenes')
def bank_scenes(bank_id):
    """The bank's captions as ORDERED scene cards — served raw so a generation
    panel can offer them as a prompt batch without going through a dataset.
    Read-only: no GPU, no writes, answers while a pass is running.
    ?statuses=keep,pending scopes like the caption pass. 404 on a missing bank."""
    raw = (request.args.get('statuses') or '').strip()
    statuses = [s for s in (p.strip() for p in raw.split(',')) if s] or None
    try:
        payload = banks.export_scene_captions(LOCAL_USER, bank_id, statuses=statuses)
    except ValueError as e:
        msg = str(e)
        return jsonify({'error': msg}), 404 if msg == 'bank not found' else 400
    return jsonify(payload)


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
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'JSON body must be an object'}), 400
    try:
        dataset_id = dataset_activity.normalize_dataset_id(data.get('dataset_id'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return _start(banks.start_promote, _app(), LOCAL_USER, bank_id,
                  data.get('image_ids'), dataset_id)


@bp.post('/bank/<int:bank_id>/promote-to-bank')
def bank_promote_to_bank(bank_id):
    """⬆ Promote's SECOND destination: copy the selection into a brand-new bank
    instead of a dataset — isolating candidates out of a big dump without
    committing them to a training container. Same shape as /bank/from-dataset:
    202 + background job, and the new bank's id back so the UI can jump to the
    bank being filled. Empty image_ids = every kept image.

    The files are COPIED: banks never share theirs, and the app rewrites images
    in place, so anything cheaper would make the two banks one at the first
    re-crop. 409 while another pass runs on the SOURCE bank."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'JSON body must be an object'}), 400
    try:
        new_id = banks.start_bank_promote(_app(), LOCAL_USER, bank_id,
                                          data.get('image_ids'),
                                          data.get('name'))
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'id': new_id}), 202


@bp.get('/bank/<int:bank_id>/selection-size')
def bank_selection_size(bank_id):
    """How many images a promotion would copy and what they WEIGH — the number
    the confirmation shows before the click. ?ids=1,2,3 for a selection, absent
    for every kept image. Images are ~300 KB apiece so this is usually a
    footnote; video is three orders of magnitude above, which is exactly why the
    dialog states the measured figure instead of assuming one."""
    raw = request.args.get('ids')
    try:
        tokens = [part.strip() for part in raw.split(',') if part.strip()] \
            if raw else []
        if any(not token.isascii() or not token.isdigit() for token in tokens):
            raise ValueError('ids must be positive integers')
        ids = banks._normalize_promotion_ids([int(token) for token in tokens])
        out = banks.selection_size(LOCAL_USER, bank_id, ids)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if out is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(out)


@bp.get('/bank/<int:bank_id>/promotable')
def bank_promotable(bank_id):
    """How many kept images 'promote all' would copy into ?dataset_id right now
    — the honest count for the promote modal (per-target: images already on
    OTHER datasets still count)."""
    try:
        dataset_id = dataset_activity.normalize_dataset_id(
            request.args.get('dataset_id'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    n = banks.promotable_count(LOCAL_USER, bank_id, dataset_id)
    if n is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'count': n})


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
                                 keep_ids=data.get('keep_ids'),
                                 respect_existing_keep=False)
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
                                          keep_ids=data.get('keep_ids'),
                                          respect_existing_keep=False)
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


@bp.post('/bank/<int:bank_id>/undo')
def bank_undo_last(bank_id):
    """↩ Put the last BULK decision back — the net under the bank's biggest
    gesture. Synchronous: it rewrites two columns on ids we already hold.

    The reply is deliberately an honest ledger, not an "ok": {restored, missing,
    conflicts, conflict_names} so a partial restore can SAY it restored 340 of
    400 and name what it left alone. 400 = there is nothing to undo (no offer,
    or it expired); 409 = a pass is running on this bank.

    Only the status-flipping bulk actions are ever offered here. 🗑 Delete
    rejected and ⬆ Promote are not undoable cleanly, so they publish no offer —
    see ``services.bank_undo``."""
    try:
        out = banks.undo_last(LOCAL_USER, bank_id)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        # Same occupied-bank refusal as everywhere else, so the UI rephrases it
        # through the same path instead of echoing our sentence. Raised as a
        # RuntimeError rather than BankJobBusy, so the kind comes from the
        # registry — same shape as delete-rejected below.
        snap = bank_jobs.get(bank_id)
        return jsonify({'error': str(e),
                        'busy_kind': (snap or {}).get('kind')}), 409
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/rotate')
def bank_rotate(bank_id):
    """Turn {ids} by {degrees} CLOCKWISE (90/180/270, negative = left).

    Idea by 1Tomber (GitHub #17). Synchronous and cheap: it writes ONE integer
    per row — the user's own files are never touched, the turned copy is built
    lazily by the resolver on the next read."""
    data = request.get_json(silent=True) or {}
    try:
        result = banks.rotate_images(LOCAL_USER, bank_id, data.get('ids'),
                                     data.get('degrees'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **result})


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
    facets as the grid (status ∩ flag ∩ cluster ∩ style ∩ subfolder ∩ search ∩
    NOT exclude), read out of a JSON body. Unknown keys (e.g. the grid's ``sort``,
    which only orders) are ignored."""
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
        # Hiding images in the grid must hide them from a curation pick too —
        # otherwise "select 60 diverse" would hand back the very images the user
        # just declared done.
        'exclude': data.get('exclude') or None,
        # Same reason as exclude: a curation pick must not hand back images the
        # grid is currently hiding.
        'tags': data.get('tags') or None,
        # 🎨 medium and ⤢ angle scope a curation pick exactly as they scope the
        # grid, under their OWN keys.
        'medium': data.get('medium') or None,
        'angle': data.get('angle') or None,
    }


@bp.post('/bank/<int:bank_id>/select-diverse')
def bank_select_diverse(bank_id):
    """Farthest-point selection of the N most VARIED images in the current filter,
    reusing the ✨ Score embeddings (no GPU). Returns the chosen ids for the UI to
    check — never mutates. 400 with a "run Score first" hint when unscored.

    {typicality} (0–1) tempers the sampling so isolated aberrations stop winning
    on isolation alone; omitted ⇒ the service default, an explicit 0 ⇒ the
    historical pure farthest-point behaviour."""
    data = request.get_json(silent=True) or {}
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    typ = data.get('typicality')
    try:
        typ = banks._TYPICALITY_DEFAULT if typ in (None, '') else float(typ)
    except (TypeError, ValueError):
        typ = banks._TYPICALITY_DEFAULT
    try:
        out = banks.select_diverse(LOCAL_USER, bank_id, n=n, typicality=typ,
                                   filters=_curation_filters(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/select-balanced')
def bank_select_balanced(bank_id):
    """Select N images SPREAD OVER the framing labels (optionally × person)
    instead of the top of one ranking — the answer to "does my set cover what I
    want to generate?". Same embeddings and same typicality guard as
    select-diverse, applied inside each bucket. Never mutates.

    {axis} 'framing' (default) | 'framing+person'. 400 with the exact missing
    pass when Score hasn't run or nothing in the filter carries the label."""
    data = request.get_json(silent=True) or {}
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    typ = data.get('typicality')
    try:
        typ = banks._TYPICALITY_DEFAULT if typ in (None, '') else float(typ)
    except (TypeError, ValueError):
        typ = banks._TYPICALITY_DEFAULT
    try:
        out = banks.select_balanced(LOCAL_USER, bank_id, n=n,
                                    axis=data.get('axis') or 'framing',
                                    typicality=typ,
                                    filters=_curation_filters(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/select-similar')
def bank_select_similar(bank_id):
    """Rank the current filter in the Bank's selected semantic space
    ({ref_id}); returns the top-N ids (or everything ≥ {min_score}) for the UI to
    check. Reuses the active CLIP or SigLIP2 cache (no GPU inference). 400 when
    the selected index is unavailable or the reference is invalid."""
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


@bp.post('/bank/<int:bank_id>/search-text')
def bank_search_text(bank_id):
    """Rank the current filter by the selected engine's text/image similarity.

    Image vectors come from the active CLIP or SigLIP2 cache; only the phrase is
    encoded in the matching ML interpreter.

    Top-N only, and deliberately NO min_score — unlike select-similar. On a real
    bank the correct-hit and unrelated-pair score distributions overlap (correct
    0.177-0.233, unrelated up to 0.197), so no threshold separates them and a
    knob here would be a control over a boundary that does not exist. See
    ``banks.search_by_text``.

    {push_down} (and `-term` inside the query) names what to push DOWN the ranking,
    with {push_down_weight} for how hard. That IS a defensible knob and the
    threshold is not: a weight scales a subtraction inside one ranking, where a
    threshold would claim a relevance boundary the measurements say is absent.

    400 = the request cannot be answered (no query or active semantic index).
    503 = the FEATURE is unavailable here (matching encoder failed) —
    a different thing, and the UI says so differently: one is "do this first",
    the other is "this install cannot do this at all"."""
    from ..services.clip_text_encoder import TextEncodeError
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({'error': 'request body must be an object'}), 400
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    try:
        out = banks.search_by_text(LOCAL_USER, bank_id, data.get('query'), n=n,
                                   push_down=data.get('push_down'),
                                   push_down_weight=data.get('push_down_weight'),
                                   filters=_curation_filters(data))
    except TextEncodeError as e:
        return jsonify({'error': str(e), 'reason': 'encoder_unavailable'}), 503
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.get('/bank/text-search/status')
def bank_text_search_status():
    """Read warm/cache state for exactly one semantic text space."""
    from ..services import clip_text_encoder
    engine = request.args.get('engine') or 'clip'
    try:
        status = clip_text_encoder.status(engine=engine)
    except (ValueError, clip_text_encoder.TextEncodeError) as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **status})


@bp.post('/bank/text-search/release')
def bank_text_search_release():
    """Reap only the selected engine's warm text worker."""
    from ..services import clip_text_encoder
    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({'error': 'request body must be an object'}), 400
    engine = data.get('engine') or 'clip'
    try:
        released = clip_text_encoder.release(engine=engine)
    except (ValueError, clip_text_encoder.TextEncodeError) as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'released': released})


@bp.get('/bank/<int:bank_id>/delete-rejected/preview')
def bank_delete_rejected_preview(bank_id):
    """What 🗑 Delete rejected would really do, for the confirmation dialog: how
    many files, where they would go, and which OTHER banks share them (nested
    source folders make one bank's cleanup another bank's amputation)."""
    out = banks.rejected_delete_preview(LOCAL_USER, bank_id)
    if out is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(out)


@bp.post('/bank/<int:bank_id>/delete-rejected')
def bank_delete_rejected(bank_id):
    """Destructive: delete the SOURCE files of every rejected image from disk
    (OS trash, else the app's own trash, else a permanent delete) and drop their
    rows. The ONLY bank action that writes to the source folder — the front-end
    gates it behind a type-DELETE confirmation fed by the preview above.

    202 + a background bank job: handing thousands of files to the Recycle Bin
    one by one takes minutes, and it used to do that inside this request with no
    count and no Stop. The refusals (404 / 409 / dataset conflict) still happen
    HERE, before a single file moves, so the dialog gets them synchronously."""
    try:
        res = banks.start_delete_rejected(_app(), LOCAL_USER, bank_id)
    except banks.BankSharesDataset as e:
        # Not "not found": the bank exists, and the refusal is the whole point —
        # its folder is a dataset's, so this delete would amputate the dataset.
        return jsonify({'error': str(e)}), 400
    except ValueError:
        return jsonify({'error': 'not found'}), 404
    except RuntimeError as e:
        # Same refusal as every other occupied-bank 409, so the UI rephrases it
        # through the same path. This one is raised as a RuntimeError rather than
        # BankJobBusy, so the kind is read back from the registry here.
        snap = bank_jobs.get(bank_id)
        return jsonify({'error': str(e),
                        'busy_kind': (snap or {}).get('kind')}), 409
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    # Under TESTING bank_jobs runs the pass inline, so the full outcome is
    # already there and rides back; in production the client watches the bank's
    # progress bar like it does for every other pass.
    return jsonify({'ok': True, 'total': res['total'],
                    **(res['job'].get('result') or {})}), 202


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
    """The full-size image. Serves the watermark-cleaned version when one exists;
    ?original=1 serves the untouched source instead — that pair IS the before/after
    comparison (no third lightbox needed)."""
    bank, row = _row_or_404(bank_id, image_id)
    if not bank or not row:
        return jsonify({'error': 'not found'}), 404
    path = (banks.abs_image_path(bank, row)
            if request.args.get('original') in ('1', 'true')
            else banks.resolved_image_path(bank, row))
    if not path or not os.path.isfile(path):
        return jsonify({'error': 'file missing'}), 404
    return send_file(path, max_age=0)


# --- "Single person here" — folder-level person assertions -------------------
# The user knows their sources. A scraped folder is very often ONE person, and
# making the embeddings pass rediscover that costs thousands of inferences for an
# answer the folder name already gave. These endpoints let them say it, undo it,
# and — separately — pay ~15 inferences to have the claim sampled.
@bp.get('/bank/<int:bank_id>/folder-persons')
def bank_folder_persons(bank_id):
    from ..services import folder_person
    data = folder_person.payload(LOCAL_USER, bank_id)
    if data is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(data)


def _asserted_subfolder_arg():
    """The one place the request's subfolder is read. '' is a MEANINGFUL value
    (the bank root), so absence is tested against None, never falsiness."""
    data = request.get_json(silent=True) or {}
    sub = data.get('subfolder')
    if sub is None:
        sub = request.args.get('subfolder')
    return None if sub is None else str(sub)


def _folder_person_error(e):
    return jsonify({'error': str(e)}), 404 if 'bank not found' in str(e) else 400


@bp.post('/bank/<int:bank_id>/folder-person')
def bank_assert_folder_person(bank_id):
    """Declare a subfolder to hold a single person. Synchronous and instant —
    there is nothing to infer, which is the entire point."""
    from ..services import folder_person
    sub = _asserted_subfolder_arg()
    if sub is None:
        return jsonify({'error': 'subfolder is required'}), 400
    try:
        out = folder_person.assert_single_person(LOCAL_USER, bank_id, sub)
    except ValueError as e:
        return _folder_person_error(e)
    return jsonify({'ok': True, **out})


@bp.delete('/bank/<int:bank_id>/folder-person')
def bank_revoke_folder_person(bank_id):
    """Undo the assertion — the group dissolves and the folder goes back to
    normal clustering. Only the ids the assertion wrote are cleared."""
    from ..services import folder_person
    sub = _asserted_subfolder_arg()
    if sub is None:
        return jsonify({'error': 'subfolder is required'}), 400
    try:
        out = folder_person.revoke(LOCAL_USER, bank_id, sub)
    except ValueError as e:
        return _folder_person_error(e)
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/folder-person/check')
def bank_check_folder_person(bank_id):
    """Sample check: ~15 images spread across the folder, embedded on their own
    and compared at the clustering threshold. Informative only — whatever it
    finds, the assertion stands until the user revokes it."""
    from ..services import folder_person
    sub = _asserted_subfolder_arg()
    if sub is None:
        return jsonify({'error': 'subfolder is required'}), 400
    try:
        folder_person.start_sample_check(_app(), LOCAL_USER, bank_id, sub)
    except ValueError as e:
        return _folder_person_error(e)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    return jsonify({'ok': True, 'sample_size': folder_person.SAMPLE_SIZE}), 202


@bp.post('/bank/<int:bank_id>/folder-scan')
def bank_scan_folder_persons(bank_id):
    """Sample the unprobed subfolders and SUGGEST the ones that look like a
    single person. It suggests only — nothing here groups an image; confirming
    stays the one-click gesture it already was.

    The same scan runs by itself at the end of 👤 Group by person, where it is
    free (the embeddings are cached); this button is for asking before ever
    running that pass, and it says what it costs."""
    from ..services import folder_person
    try:
        folder_person.start_folder_scan(_app(), LOCAL_USER, bank_id)
    except ValueError as e:
        return _folder_person_error(e)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    return jsonify({'ok': True, 'sample_size': folder_person.SAMPLE_SIZE,
                    'limit': folder_person.MAX_SCAN_FOLDERS}), 202


# --- the preflight of 👤 Group by person -------------------------------------
# The sampling above used to be reachable only from the Subfolder panel, which
# the user who presses 🚀 Launch all never opens. These three endpoints put it in
# front of the pass instead: what it would cost (GET), run it (POST), and confirm
# the folders it found in one gesture (accept).
@bp.get('/bank/<int:bank_id>/person-preflight')
def bank_person_preflight_plan(bank_id):
    """What the preflight would sample, what it would cost, and the verdicts it
    already holds. Read-only — nothing is probed and nothing is grouped."""
    from ..services import folder_person
    data = folder_person.preflight_payload(LOCAL_USER, bank_id)
    if data is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(data)


@bp.post('/bank/<int:bank_id>/person-preflight')
def bank_person_preflight(bank_id):
    """Run the sampling as the preamble of the person pass. Same probe as
    🔎 Scan folders with a far higher ceiling, because the alternative the user
    just chose is one embedding per image over the whole bank."""
    from ..services import folder_person
    try:
        folder_person.start_preflight(_app(), LOCAL_USER, bank_id)
    except ValueError as e:
        return _folder_person_error(e)
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    return jsonify({'ok': True, 'sample_size': folder_person.SAMPLE_SIZE,
                    'limit': folder_person.MAX_PREFLIGHT_FOLDERS}), 202


@bp.post('/bank/<int:bank_id>/folder-persons/accept')
def bank_accept_folder_persons(bank_id):
    """Confirm several folders at once — the preflight's "yes to all". Every one
    of them becomes an ordinary assertion; folders that could not be asserted come
    back in `failed` rather than being dropped."""
    from ..services import folder_person
    data = request.get_json(silent=True) or {}
    subs = data.get('subfolders')
    if not isinstance(subs, list):
        return jsonify({'error': 'subfolders must be a list'}), 400
    try:
        out = folder_person.accept_suggestions(LOCAL_USER, bank_id, subs)
    except ValueError as e:
        return _folder_person_error(e)
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/describe-filter')
def bank_describe_filter(bank_id):
    """Turn a sentence into this bank's own filter — and apply NOTHING.

    The response moves the visible controls; the grid's existing counters then say
    how many images that lands on, measured, without the model. A wrong reading
    therefore costs one glance at chips the user can edit, not a silent selection.

    Errors go through the shared mapper so the local-fence refusal keeps its
    `ollama_fence_blocked` code: this surface has no degraded mode, and a model
    held by something outside the app is the one failure that carries its own
    remedy."""
    data = request.get_json(silent=True) or {}
    try:
        if banks.get_bank(LOCAL_USER, bank_id) is None:
            # Checked before prompting: a model round-trip for a bank that is not
            # there costs seconds and answers a question nobody asked.
            return jsonify({'error': 'bank not found'}), 404
        out = translator.translate(bank_id, data.get('request'))
    except Exception as e:  # noqa: BLE001 — _map_error re-raises what it cannot map
        return _map_error(e)
    return jsonify({'ok': True, **out})
