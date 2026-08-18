"""🎬 Video bank API — triage a folder of rushes before it becomes a training set.

Deliberately the same surface as the 🗃️ image bank: heavy passes return 202 and
run in ONE background thread per bank, the UI polls GET /video-bank/<id> whose
payload embeds the live job, and 409 means a pass already owns this bank. A user
does not know there are two services behind the app and should not be able to
tell — so the status codes, the `busy_kind` field and the payload shape are
copied rather than reinvented.

The one place the two lanes differ is promotion, and it differs because of the
architecture rather than the API: promoting is where the video lane finally
ENCODES something, so it takes a target profile and a clip length, and everything
that could be refused is refused synchronously before a dataset exists.
"""
import logging
import os

from flask import Blueprint, current_app, jsonify, request, send_file

from ..config import LOCAL_USER
from ..services import bank_jobs
from ..services import video_bank_service as svc
from ..services import video_metrics

logger = logging.getLogger(__name__)

bp = Blueprint('video_bank', __name__, url_prefix='/api')


def _app():
    return current_app._get_current_object()


def _busy(e):
    """The ONE shape of a "this bank is occupied" refusal, identical to the image
    lane's: `error` stays an English sentence for anything that only knows how to
    print a message, and `busy_kind` names the pass so the UI can refuse the click
    in the user's own vocabulary."""
    return jsonify({'error': str(e), 'busy_kind': e.kind}), 409


def _missing(bank_id):
    """404, not 400. "Bank not found" is not something the user can fix by editing
    the body — it means the bank was deleted in another tab."""
    return jsonify({'error': f'video bank {bank_id} not found'}), 404


def _start(bank_id, fn, *args, **kwargs):
    """Start-a-pass envelope: 404 unknown, 409 busy, 400 bad input, 503 missing
    tool, 202 on launch."""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    try:
        fn(*args, **kwargs)
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    return jsonify({'ok': True}), 202


# --- banks ---------------------------------------------------------------------

@bp.get('/video-banks')
def video_banks_list():
    """Every video bank with its counters. GET {'banks': [...]}"""
    return jsonify({'banks': svc.list_banks(LOCAL_USER)})


@bp.post('/video-bank/create')
def video_bank_create():
    """Body {name, folder}. Instant — no decode, no detection: those are passes.
    200 {'ok', 'id', 'added'}; 400 on a folder that is missing, unreadable, or
    that would make a bank share bytes with a dataset."""
    data = request.get_json(silent=True) or {}
    try:
        bank, added = svc.create_bank(LOCAL_USER, data.get('name'),
                                      data.get('folder'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'id': bank.id, 'added': added})


@bp.get('/video-bank/<int:bank_id>')
def video_bank_get(bank_id):
    """The workspace payload AND the 2 s job poll. ?refresh=1 re-walks the source
    folder first (the workspace sends it when the bank is opened), because a bank
    points at a LIVE folder people keep dropping files into."""
    sync = None
    if request.args.get('refresh') == '1':
        sync = svc.refresh_bank(LOCAL_USER, bank_id, force=True)
    payload = svc.bank_payload(LOCAL_USER, bank_id)
    if payload is None:
        return _missing(bank_id)
    payload['folder_sync'] = sync
    return jsonify(payload)


@bp.delete('/video-bank/<int:bank_id>')
def video_bank_delete(bank_id):
    """Drops the bank, its sources, its clips and its thumbnails. Never the
    datasets built out of it — that provenance is deliberately not a foreign key."""
    if not svc.delete_bank(LOCAL_USER, bank_id):
        return _missing(bank_id)
    return jsonify({'ok': True})


@bp.post('/video-bank/<int:bank_id>/refresh')
def video_bank_refresh(bank_id):
    """Re-inventory the folder on demand. Strictly additive; vanished files are
    counted, never removed. 200 {'ok', 'added', 'missing', 'unavailable', 'error'}."""
    sync = svc.refresh_bank(LOCAL_USER, bank_id, force=True)
    if sync is None:
        return _missing(bank_id)
    return jsonify({'ok': True, **sync})


@bp.get('/video-bank/<int:bank_id>/sources')
def video_bank_sources(bank_id):
    """The per-FILE view: duration, native rate, geometry, probe and detect state.
    GET {'sources': [...]}"""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    return jsonify({'sources': svc.sources_payload(LOCAL_USER, bank_id)})


# --- clips ---------------------------------------------------------------------

@bp.get('/video-bank/<int:bank_id>/clips')
def video_bank_clips(bank_id):
    """One page of the gallery. ?status= ?source_id= ?offset= ?limit=, and
    ?ids_only=1 which answers {'ids', 'total'} for the WHOLE filter in one request
    — what "select all in filter" needs, sharing this function so the two answers
    can never disagree about what the filter holds."""
    args = request.args

    def _int(name):
        try:
            return int(args.get(name)) if args.get(name) else None
        except ValueError:
            return None

    payload = svc.list_clips(
        LOCAL_USER, bank_id,
        status=args.get('status') or None,
        source_id=_int('source_id'),
        ids_only=args.get('ids_only') == '1',
        offset=_int('offset') or 0, limit=_int('limit') or 200)
    if payload is None:
        return _missing(bank_id)
    return jsonify(payload)


@bp.post('/video-bank/<int:bank_id>/triage')
def video_bank_triage(bank_id):
    """Body {ids: [], status: 'keep'|'reject'|'pending', reason?}.

    An EMPTY/absent ids list means every clip of the bank — the same "no selection
    = all of it" convention promotion uses. The fresh counters ride back on the
    response so the gallery updates without a second round trip; a triage click is
    the most repeated gesture in this lane."""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    data = request.get_json(silent=True) or {}
    try:
        out = svc.set_clip_status(LOCAL_USER, bank_id, data.get('ids'),
                                  data.get('status'), reason=data.get('reason'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


# --- retouching the cuts --------------------------------------------------------
#
# Three routes, one envelope. They are the only writes in this lane that change
# BOUNDS, and bounds are what every later pass and the encoder read — so they are
# refused while a pass owns the bank, exactly like a second pass would be.
#
# That 409 is not symmetry for its own sake. The thumbnails pass reads a clip's
# bounds, shells out to ffmpeg, and only then stamps `thumb_state='ok'`. An edit
# landing between those two writes produces a thumbnail of the OLD span marked as
# current — the precise failure this feature exists to remove, made invisible.

def _edit(bank_id, fn, *args):
    """404 unknown bank/clip/source · 409 a pass owns the bank · 400 bad range ·
    200 with the retouched row and the fresh counters, so the gallery can swap the
    tile in place instead of reloading a page of hundreds."""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    job = bank_jobs.get(svc.job_key(bank_id))
    if job and not job['finished']:
        return _busy(bank_jobs.BankJobBusy(job['kind']))
    try:
        out = fn(*args)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    if out is None:
        return jsonify({'error': 'shot not found in this bank'}), 404
    return jsonify({'ok': True, **out})


@bp.patch('/video-bank/<int:bank_id>/clip/<int:clip_id>/bounds')
def video_bank_clip_bounds(bank_id, clip_id):
    """Body {start_s, end_s}. PATCH because it edits one field pair of an existing
    shot and is idempotent.

    Moving the START is more than a trim for image-to-video targets: ai-toolkit
    conditions an i2v sample on the clip's FIRST frame, so choosing where a shot
    begins IS choosing the conditioning image. The lightbox says so where the
    gesture happens; the route only has to make the gesture possible.

    Answers {'ok', 'clip', 'counts'}. The thumbnail and the measurements of the old
    span are dropped — see `video_bank_service.set_clip_bounds`."""
    data = request.get_json(silent=True) or {}
    return _edit(bank_id, svc.set_clip_bounds, LOCAL_USER, bank_id, clip_id,
                 data.get('start_s'), data.get('end_s'))


@bp.post('/video-bank/<int:bank_id>/clip/<int:clip_id>/split')
def video_bank_clip_split(bank_id, clip_id):
    """Body {at_s}, strictly inside the shot with room for a real span either side.
    Answers {'ok', 'clip', 'new_clip', 'counts'} — the shot now ends at `at_s` and
    the new one carries the rest, both marked manual, both with their thumbnail
    forgotten. The new half inherits the parent's triage status on purpose."""
    data = request.get_json(silent=True) or {}
    return _edit(bank_id, svc.split_clip, LOCAL_USER, bank_id, clip_id,
                 data.get('at_s'))


@bp.post('/video-bank/<int:bank_id>/source/<int:source_id>/clips')
def video_bank_source_clip_create(bank_id, source_id):
    """Body {start_s, end_s} — the cut the detector missed entirely. Answers
    {'ok', 'clip', 'counts'}; the new shot is `pending`, because nobody has judged
    it yet."""
    data = request.get_json(silent=True) or {}
    return _edit(bank_id, svc.create_clip, LOCAL_USER, bank_id, source_id,
                 data.get('start_s'), data.get('end_s'))


# --- Find shots: the threshold, the preview, and the re-cut ---------------------
#
# Every route below rides the same `_edit` envelope as a trim, and for the same
# reason: they change BOUNDS, which is what every later pass and the encoder
# read. They are refused while a pass owns the bank.
#
# None of them is a 202. A re-cut reads a cached vector and does arithmetic —
# there is no decode anywhere in it — so putting it behind the job queue would
# make an instant operation look slow AND lock the bank against the next click.

@bp.post('/video-bank/<int:bank_id>/shot-threshold')
def video_bank_shot_threshold(bank_id):
    """Body {threshold: 0..1 | null}. null CLEARS the override — the bank falls
    back to the global default, which is not the same as setting 0.

    Saves the number and cuts nothing. Re-cutting is a separate click on
    purpose: changing a value must not silently rewrite hundreds of rows."""
    data = request.get_json(silent=True) or {}
    return _edit(bank_id, svc.set_bank_shot_threshold, LOCAL_USER, bank_id,
                 data.get('threshold'))


@bp.post('/video-bank/<int:bank_id>/source/<int:source_id>/shot-threshold')
def video_bank_source_shot_threshold(bank_id, source_id):
    """Body {threshold: 0..1 | null} for ONE file. A mixed corpus is mixed
    inside one folder, and no bank-level number is right for every file in it."""
    data = request.get_json(silent=True) or {}
    return _edit(bank_id, svc.set_source_shot_threshold, LOCAL_USER, bank_id,
                 source_id, data.get('threshold'))


@bp.post('/video-bank/<int:bank_id>/shot-dry-run')
def video_bank_shot_dry_run(bank_id):
    """Body {source_id?, thresholds?} → "at X you would get N shots", per
    threshold, from the cache. Writes nothing — the point of a preview is being
    able to change your mind after seeing it."""
    data = request.get_json(silent=True) or {}
    return _edit(bank_id, svc.shot_dry_run, LOCAL_USER, bank_id,
                 data.get('source_id'), data.get('thresholds'))


@bp.post('/video-bank/<int:bank_id>/recut')
def video_bank_recut(bank_id):
    """Body {threshold?} — re-cut every file that has a cached vector.

    Spares hand-made cuts, promoted clips, and any file declared a single take.
    Answers with what it did, INCLUDING what it left alone: a bank where half
    the sources predate the cache must say so rather than report a clean run."""
    data = request.get_json(silent=True) or {}
    return _edit(bank_id, svc.recut_bank, LOCAL_USER, bank_id,
                 data.get('threshold'))


@bp.post('/video-bank/<int:bank_id>/source/<int:source_id>/recut')
def video_bank_source_recut(bank_id, source_id):
    """Body {threshold?} — re-cut ONE file, replacing hand-made cuts on it too.

    That is the difference from the bank-wide route, and it is deliberate: this
    is a gesture on a file the user picked by name, and it is the only way back
    from "this file is a single take". `replaced_manual` in the answer says how
    many hand-made cuts went, so the UI can warn BEFORE rather than apologise
    after.

    503, not 400, when the file has no cached vector: it is a fact about when
    the file was detected, and the fix is running Find shots on it once."""
    data = request.get_json(silent=True) or {}
    try:
        return _edit(bank_id, svc.recut_source, LOCAL_USER, bank_id, source_id,
                     data.get('threshold'))
    except svc.ShotProbsMissing as e:
        return jsonify({'error': str(e)}), 503


@bp.post('/video-bank/<int:bank_id>/source/<int:source_id>/single-shot')
def video_bank_source_single_shot(bank_id, source_id):
    """This file is ONE take: replace its clips with a single full-length one.

    Nothing else in this space offers it, and a single-take corpus needs it more
    than any slider — there the failure is not a missed cut, it is a file
    quietly chopped into six fragments that each train on a third of a gesture."""
    return _edit(bank_id, svc.mark_single_shot, LOCAL_USER, bank_id, source_id)


#: Container → MIME. `mimetypes.guess_type` is registry-driven on Windows and
#: answers None for .mkv/.webm on plenty of installs; send_file RAISES on an
#: unguessable name, which would turn "this player cannot decode Matroska" into
#: a 500. Guessing wrong is recoverable, refusing to answer is not.
_VIDEO_MIMETYPES = {
    '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.mkv': 'video/x-matroska',
    '.webm': 'video/webm', '.avi': 'video/x-msvideo',
}


def _video_mimetype(path):
    return _VIDEO_MIMETYPES.get(os.path.splitext(path)[1].lower(), 'video/mp4')


@bp.get('/video-bank/<int:bank_id>/source/<int:source_id>/media')
def video_bank_source_media(bank_id, source_id):
    """The SOURCE file's bytes — what the lightbox plays.

    RANGE REQUESTS ARE THE POINT, hence ``conditional=True``. The player asks for
    one shot with a media fragment (`#t=41,46`); the browser turns that into a
    Range request and pulls only those bytes ONLY IF the response is 206-capable.
    On a 200 it downloads the whole rush to show five seconds — on a two-hour
    file that is worse than having no preview at all.

    ``max_age=0`` mirrors the image lane (routes/bank.py): a bank points at a
    LIVE folder and the file under a relpath can be replaced on disk.

    404 covers every refusal — unknown bank, unknown source, a relpath that
    escapes the bank's folder, a file that has since moved. See
    ``video_bank_service.source_media_path`` for why they are not distinguished.
    """
    path = svc.source_media_path(LOCAL_USER, bank_id, source_id)
    if path is None:
        return jsonify({'error': 'source file not available'}), 404
    return send_file(path, mimetype=_video_mimetype(path),
                     conditional=True, max_age=0)


@bp.get('/video-bank/<int:bank_id>/clip/<int:clip_id>/thumb')
def video_bank_clip_thumb(bank_id, clip_id):
    """The one image a bank serves. 404 when the thumbnail pass has not run — the
    gallery renders a placeholder on that, whereas a 500 would fill the console
    with errors for a perfectly ordinary state."""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    path = svc.thumb_path(bank_id, clip_id)
    if not path.is_file():
        return jsonify({'error': 'no thumbnail for this clip yet'}), 404
    return send_file(str(path), mimetype='image/jpeg')


# --- passes --------------------------------------------------------------------

@bp.post('/video-bank/<int:bank_id>/probe')
def video_bank_probe(bank_id):
    """Read what each source file IS. Body {reprobe?: bool}. 202/404/409."""
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_probe, _app(), LOCAL_USER, bank_id,
                  reprobe=bool(data.get('reprobe')))


@bp.post('/video-bank/<int:bank_id>/detect')
def video_bank_detect(bank_id):
    """Find the shot boundaries — the expensive pass. Body {redetect?: bool},
    which re-cuts only the clips nobody has promoted. 202/404/409."""
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_detect, _app(), LOCAL_USER, bank_id,
                  redetect=bool(data.get('redetect')))


@bp.post('/video-bank/<int:bank_id>/thumbs')
def video_bank_thumbs(bank_id):
    """One frame per shot, taken from its MIDDLE. Body {rethumb?: bool}."""
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_thumbs, _app(), LOCAL_USER, bank_id,
                  rethumb=bool(data.get('rethumb')))


@bp.post('/video-bank/<int:bank_id>/measure')
def video_bank_measure(bank_id):
    """Wave 2: motion, exposure, sharpness and freezes, one decode per clip.
    Body {remeasure?: bool}. Refused with the missing piece named when the
    decode extra is absent — measuring is the one pass that cannot degrade."""
    from .. import capabilities
    cap = capabilities.probe_video()
    if not cap['decode']:
        return jsonify({'error': cap['detail']}), 503
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_measure, _app(), LOCAL_USER, bank_id,
                  remeasure=bool(data.get('remeasure')))


@bp.post('/video-bank/<int:bank_id>/embed')
def video_bank_embed(bank_id):
    """🔎 Wave 3: CLIP vectors for a few frames of every shot, so a typed word can
    find the scenes it describes. Body {reembed?: bool}.

    Two 503s, and they are not the same sentence: the decode extra is missing
    (no frames to embed at all), or no interpreter here can run CLIP (the ✨ Score
    environment). Collapsing them into one "unavailable" is how someone installs
    the wrong thing twice."""
    from .. import capabilities
    cap = capabilities.probe_video()
    if not cap['decode']:
        return jsonify({'error': cap['detail']}), 503
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_embed, _app(), LOCAL_USER, bank_id,
                  reembed=bool(data.get('reembed')))


@bp.post('/video-bank/<int:bank_id>/caption')
def video_bank_caption(bank_id):
    """🗣 Wave 5: describe what happens in each shot. Body {recaption?,
    include_edited?}.

    Two 503s that are NOT the same sentence: the decode extra is missing (no
    frames to show a model), or no interpreter here can run the caption model.
    `include_edited` is the explicit opt-in to overwrite captions a human wrote —
    a bulk re-run never does that on its own."""
    from .. import capabilities
    cap = capabilities.probe_video()
    if not cap['decode']:
        return jsonify({'error': cap['detail']}), 503
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_caption, _app(), LOCAL_USER, bank_id,
                  recaption=bool(data.get('recaption')),
                  include_edited=bool(data.get('include_edited')),
                  # Per-run prompt style; absent = the configured default. An
                  # unknown value falls back rather than failing — and falls back
                  # to `standard`, never to the permissive one.
                  style=data.get('style'))


@bp.post('/video-bank/<int:bank_id>/dedup')
def video_bank_dedup(bank_id):
    """✂ Group near-identical shots. Body {threshold?}. 202/400/404/409.

    NO capability check, deliberately, and it is the only heavy-sounding pass
    here without one: this reads the vectors 🔎 Find scenes already cached and
    does dot products. A bank with no vectors is a 400 with the sentence naming
    the pass to run first — the service raises it, `_start` turns it into the
    status code."""
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_dedup, _app(), LOCAL_USER, bank_id,
                  threshold=data.get('threshold'))


@bp.post('/video-bank/<int:bank_id>/watermark')
def video_bank_watermark(bank_id):
    """🔖 Look for a watermark on each shot's ambassador frame. Body {rescan?}.

    Two 503s that are NOT the same sentence, the same split as the embed pass:
    the decode extra is missing (no frame to extract at all), or the watermark
    detector's own environment and weights are not there. Collapsing them is how
    someone installs the wrong thing twice."""
    from .. import capabilities
    cap = capabilities.probe_video()
    if not cap['decode']:
        return jsonify({'error': cap['detail']}), 503
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_watermark, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.post('/video-bank/<int:bank_id>/safezone')
def video_bank_safe_zone(bank_id):
    """🔳 Measure bands and burned-in text on each shot. Body {rescan?}.

    ONE 503, not two, and that is the difference from the watermark route above:
    without the decode extra there is no frame to look at and the pass is
    impossible, but without the OCR engine HALF of it still works. Refusing here
    on a missing RapidOCR would withhold a letterbox measurement that costs
    nothing and needs nothing — so the missing extra travels as a state on every
    shot and a sentence in the job detail instead of as a status code.
    """
    from .. import capabilities
    cap = capabilities.probe_video()
    if not cap['decode']:
        return jsonify({'error': cap['detail']}), 503
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_safe_zone, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.post('/video-bank/<int:bank_id>/defects')
def video_bank_defects(bank_id):
    """🩻 Sweep each source file for duplicated frames, blocks and soft edges.
    Body {rescan?}.

    ONE 503 and it names ffmpeg, not the decode extra — the opposite split from
    every pass above. Those decode frames with PyAV and check `cap['decode']`;
    this one hands the whole file to the ffmpeg binary and never opens it here,
    so an install with `av` and no ffmpeg must be refused for the binary it is
    actually missing. The service owns that sentence (it is the same
    ffmpeg_ready() verdict the Setup row shows), so the two can never drift into
    telling the user to install different things.
    """
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_defects, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.post('/video-bank/<int:bank_id>/aicheck')
def video_bank_ai_check(bank_id):
    """🤖 Measure how erratically each shot moves. Body {recheck?}.

    Two 503s that are NOT the same sentence, the same split as 🔖 Watermarks and
    🗣 Describe: the decode extra is missing (no frames to encode at all), or no
    interpreter here can run the model. Collapsing them is how somebody installs
    the wrong thing twice. The second one is raised by the service, which owns
    the ✨ Score sentence 🎨 Look already uses.
    """
    from .. import capabilities
    cap = capabilities.probe_video()
    if not cap['decode']:
        return jsonify({'error': cap['detail']}), 503
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_ai_check, _app(), LOCAL_USER, bank_id,
                  recheck=bool(data.get('recheck')))


@bp.patch('/video-bank/<int:bank_id>/clip/<int:clip_id>/caption')
def video_bank_clip_caption(bank_id, clip_id):
    """Store the caption a HUMAN wrote for one shot. Body {caption}.

    Marked 'edited' so a bulk re-run leaves it alone; an empty string clears it
    and puts the shot back in the pass's queue, which is the only way back from
    a caption someone regrets."""
    from ..services import video_caption
    data = request.get_json(silent=True) or {}
    row = video_caption.set_caption(LOCAL_USER, bank_id, clip_id,
                                    data.get('caption'))
    if row is None:
        return _missing(bank_id)
    return jsonify({'ok': True, 'clip': row})


@bp.get('/video-bank/<int:bank_id>/search')
def video_bank_search(bank_id):
    """Rank a bank's shots by CLIP similarity to `q`, best first.

    Query: q (the phrase), n (how many, default 60), push_down (a trait to sink
    — `-term` inside q means the same thing), push_down_weight, status (search
    inside one triage bucket).

    A GET because it is a READ and nothing else: it changes nothing, it is
    re-runnable, and a user can put a search in a bookmark. Answers the ranked
    rows WITH the ranking (see video_clip_search.search) so the grid never has to
    re-fetch them and lose the order.

    400 = this bank cannot answer yet (no phrase, never embedded). 503 = this
    INSTALL cannot answer at all (no torch/open_clip). One is "do this first",
    the other is "this machine cannot do this" — and a UI that says the first
    when it means the second sends people round a loop."""
    from ..services.clip_text_encoder import TextEncodeError
    from ..services import video_clip_search
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    args = request.args
    try:
        n = int(args.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    weight = args.get('push_down_weight')
    try:
        out = video_clip_search.search(
            LOCAL_USER, bank_id, args.get('q'), n=n,
            push_down=args.get('push_down'),
            push_down_weight=float(weight) if weight not in (None, '') else None,
            status=args.get('status') or None)
    except TextEncodeError as e:
        return jsonify({'error': str(e), 'reason': 'encoder_unavailable'}), 503
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/video-bank/<int:bank_id>/metrics-dry-run')
def video_bank_metrics_dry_run(bank_id):
    """How many clips EACH cut would flag, before anything is committed. Body =
    the thresholds to try ({motion_floor?, motion_ceiling?, luma_floor?,
    freeze_max?, sharpness_floor?}); answers per-rule counts plus total_flagged,
    which counts CLIPS, so one clip caught by two rules is not counted twice."""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    data = request.get_json(silent=True) or {}
    # Same single source as the config reader: an allow-list written out by hand
    # here is how a cut the backend honours gets silently dropped from the very
    # preview meant to make it visible.
    allowed = video_metrics.THRESHOLD_KEYS
    try:
        thresholds = {k: float(data[k]) for k in allowed if data.get(k) is not None}
    except (TypeError, ValueError):
        return jsonify({'error': 'thresholds must be numbers'}), 400
    return jsonify(svc.metrics_dry_run(LOCAL_USER, bank_id, thresholds))


@bp.post('/video-bank/<int:bank_id>/pipeline')
def video_bank_pipeline(bank_id):
    """Probe → detect → thumbnails, chained. Body {steps?: [...]} (canonical order
    is enforced whatever order they arrive in). What a user actually wants on a
    fresh bank, because each pass's input is the previous one's output."""
    data = request.get_json(silent=True) or {}
    return _start(bank_id, svc.start_pipeline, _app(), LOCAL_USER, bank_id,
                  steps=data.get('steps'))


@bp.post('/video-bank/<int:bank_id>/cancel')
def video_bank_cancel(bank_id):
    """Stop the live pass. 200 {'ok', 'cancelled'} — false simply means there was
    nothing running, which is not an error worth a red toast."""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    return jsonify({'ok': True, 'cancelled': svc.cancel(bank_id)})


@bp.post('/video-bank/<int:bank_id>/promote')
def video_bank_promote(bank_id):
    """Encode the KEPT clips into a new video dataset.

    Body {name, target_profile, frames?, width?, height?, ids?, edge_inset_s?}.
    `frames` defaults to the profile's own default length; width+height are
    optional and mean "cut at this size" (omitted = keep the source's). `ids`
    empty/absent = every kept clip. `max_per_source` caps how many clips ONE
    source may contribute (absent/null = no cap), which is what keeps a 50-clip
    dataset from quietly being three videos over-represented. `edge_inset_s`
    trims that many seconds off
    BOTH bounds of every clip (default 0 — a shot boundary is where a cut just
    happened, but turning that on by default would silently change what every
    existing recipe exports).

    202 {'ok', 'id', 'name', 'output_dir', 'clips'} — the id rides back so the UI
    can navigate straight to the dataset being filled. 400 names a legal frame
    count or a valid size; 503 means ffmpeg is missing; 409 means a pass is running.

    This is the ONLY route in the lane that writes media, by design: a bank stores
    bounds, and encoding 340 clips to keep 128 is what that design avoids."""
    if svc.get_bank(LOCAL_USER, bank_id) is None:
        return _missing(bank_id)
    data = request.get_json(silent=True) or {}
    width, height = data.get('width'), data.get('height')
    size = (width, height) if width and height else None
    try:
        out = svc.start_promote(_app(), LOCAL_USER, bank_id,
                                ids=data.get('ids'), name=data.get('name'),
                                target_profile=data.get('target_profile'),
                                frames=data.get('frames'), size=size,
                                max_per_source=data.get('max_per_source'),
                                edge_inset_s=data.get('edge_inset_s'))
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    return jsonify({'ok': True, **out}), 202


# --- 🕸 scrape → video bank ------------------------------------------------------

@bp.post('/video-bank/scrape-import')
def video_bank_scrape_import():
    """Download the picked scan items into a video bank.

    Body: {items:[{url,title,…}], bank_id?} to APPEND to any existing bank, or
    {items, name} to create one. The SAME contract as the image lane's
    `/api/bank/scrape-import`, on purpose — the two destinations answer
    {'ok','bank_id','name','created','saved','already_there','added','skipped'}
    so one client helper drives both.

    Synchronous, like the image outlet: the per-request cap
    (`SCRAPE_VIDEO_IMPORT_MAX`) is what bounds it, and a big selection arrives as
    successive batches. 400 on bad input — including the one destination that is
    refused however explicitly it was picked: a bank sitting on a dataset's own
    folder, where the clips would land inside training material. 409 when a pass
    already owns the bank."""
    data = request.get_json(silent=True) or {}
    raw_bank_id = data.get('bank_id')
    bank_id = None
    if raw_bank_id is not None:
        try:
            bank_id = int(raw_bank_id)
        except (TypeError, ValueError):
            return jsonify({'error': 'bank_id must be a number'}), 400
    try:
        res = svc.scrape_import_to_video_bank(LOCAL_USER, data.get('items'),
                                              bank_id=bank_id,
                                              name=data.get('name'))
    except bank_jobs.BankJobBusy as e:
        return _busy(e)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **res})
