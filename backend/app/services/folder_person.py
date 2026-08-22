""""Single person here" — folder-level person assertions for the 🗃️ image bank.

WHY
---
"Group by person" is the bank's most expensive pass: one InsightFace embedding
per image, thousands of images, minutes of GPU (or a long CPU crawl). On scraped
material that cost usually buys nothing — the sources are already ONE FOLDER PER
PERSON, and the pass spends its time rediscovering by inference what the folder
name said for free.

This module is the user saying it instead. One click on a subfolder:
  * every image of that folder gets a person id IMMEDIATELY — zero inference;
  * the embeddings pass then SKIPS those images entirely (that skip IS the
    saving, not a nicety on top of it);
  * the rule is PERSISTED, so a re-scan keeps it and an image that lands in the
    folder tomorrow joins the group on insert;
  * it is REVOCABLE — the user was wrong, one click puts the folder back in the
    way of normal clustering and dissolves the asserted group.

WHAT IT DOES NOT DO
-------------------
It does not verify anything. A declaration is not evidence, so a SAMPLE CHECK is
offered next to it: ~15 images spread across the folder, embedded on their own,
compared at the SAME cosine threshold the clustering uses (bank.face_threshold —
there is one truth about "same person" in this app, not two). Its verdict is
INFORMATIVE: "sample consistent (14/15 same person)" or "2 different faces in
the sample — check this folder". The assertion stands either way; the user's
folder, the user's call.

HOW IT COEXISTS WITH THE EMBEDDING CLUSTERS
-------------------------------------------
Same table, same column, same id space: `bank_image.face_cluster`. An asserted
group IS a person cluster — every reader (the person chips, the coverage advice,
the cluster filter, promote) keeps working with no knowledge of this module.
What tells them apart is `bank_image.face_cluster_origin` ('asserted' vs NULL),
and it exists for exactly two reasons:
  1. the embeddings pass must not silently overwrite the user's word, and must
     not renumber its own clusters onto an asserted id — so it skips asserted
     rows and OFFSETS its ids above every asserted id in the bank;
  2. revoking must know which ids it may clear.
Because the ids share one space, a later CROSS-FOLDER MERGE (linking two folders
that turn out to be the same person, or joining an asserted folder to a computed
cluster) is a plain id remap over `face_cluster` — nothing here forbids it, and
an asserted folder is not a wall around its images.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from functools import wraps

from sqlalchemy import func, or_

from ..extensions import db
from ..models import BankFolderPerson, BankFolderProbe, BankImage
from . import bank_jobs

logger = logging.getLogger(__name__)


def _serialized_bank_mutation(kind):
    """Use the same Bank reservation fence as every other synchronous write."""
    def decorate(fn):
        @wraps(fn)
        def guarded(user_id, bank_id, *args, _bank_lease=None, **kwargs):
            with bank_jobs.mutation_lease(
                    bank_id, kind, capability=_bank_lease) as lease:
                return fn(user_id, bank_id, *args, _bank_lease=lease, **kwargs)
        return guarded
    return decorate

# How many images the sample check embeds. Small enough to stay a few seconds of
# GPU (vs thousands for a full pass), big enough that a second person occupying a
# decent share of the folder is very likely to be drawn at least once.
SAMPLE_SIZE = 15
ASSERTED = 'asserted'
# A folder too small to sample meaningfully is not probed: below this, "15 of 3
# images agree" is arithmetic, not evidence, and a suggestion built on it would
# be noise on every stray folder a scrape leaves behind.
MIN_PROBE_IMAGES = 5
# Ceiling for the MANUAL scan, which pays ~15 embeddings per folder. A bank with
# 200 subfolders would otherwise cost 3 000 inferences behind one click. The
# biggest unprobed folders go first (they are where a suggestion is worth most)
# and the job says out loud how many it left. The automatic pass after 👤 Group
# by person has NO ceiling: it reuses embeddings that pass already cached, so it
# costs nothing to cover every folder.
MAX_SCAN_FOLDERS = 20
# Ceiling for the PREFLIGHT — the same probe, run as the PREAMBLE of 👤 Group by
# person instead of hiding behind a button nobody presses. It is generous where
# the manual scan is cautious, because the comparison is the opposite one: the
# user is about to pay one embedding per image over the whole bank, so fifteen
# per folder is a rounding error next to it (200 folders = 3 000 embeddings,
# against 200 × 300 = 60 000 for the pass they just asked for). It stays a
# ceiling and not "no limit" — a bank of ten thousand folders would turn a
# preamble into a pass of its own — and whatever it does not reach is stated out
# loud, never assumed away.
MAX_PREFLIGHT_FOLDERS = 200

# --- the re-draw budget -----------------------------------------------------
# THE HOLE THIS FILLS. The sample was a single draw: fifteen images, whatever
# came back. On scraped folders full of crops, backs and blur that draw can land
# on fifteen images with no readable face, and the folder ends with NO verdict —
# fifteen embeddings spent for nothing, and then the full pass over its three
# thousand images anyway. Exactly the cost the preflight exists to avoid, paid
# twice. Measured on a real bank: four folders of six, 3 546 / 1 866 / 488 / 466
# images, each reported "only 0 of 15 sampled images had a usable face".
#
# So a draw that cannot be read is REPLACED, not accepted. The target is still
# fifteen images with a usable face; what changes is that the probe keeps drawing
# (new images, never one it has already tried, still spread across the folder)
# until it reaches that target or runs out of budget.
#
# THE BUDGET IS THE POINT. Without a ceiling "keep drawing" IS the full pass,
# reached through the back door. Three numbers bound it, and the smallest wins:
PROBE_TARGET_FACES = SAMPLE_SIZE
# 1. a hard ceiling of draws per folder. 60 is not a round number: reaching 15
#    usable faces needs 15/p draws at a hit rate p, and 60 is exactly the budget
#    for p = 1/4 — one readable face in four images. Below that the folder is
#    genuinely face-poor, and MORE draws would not change the answer, only its
#    price; the verdict says so instead.
PROBE_MAX_DRAWS = 4 * SAMPLE_SIZE
# 2. never more than a quarter of the folder. This is what stops the probe from
#    becoming the pass on SMALL folders: sampling 60 of an 80-image folder would
#    cost most of what analysing it costs, for a suggestion. In practice this cap
#    binds below ~240 images and the ceiling above it, so folders of 60 or fewer
#    keep the exact single draw they have today.
PROBE_MAX_FOLDER_SHARE = 0.25
# 3. at most this many child calls per scan. Each round is one subprocess and one
#    model load shared by EVERY folder still short of its target, so rounds are
#    cheap in aggregate — but they are not free, and an unbounded loop is not a
#    budget. Two rounds cover the realistic cases (see _topup_draws).
PROBE_MAX_ROUNDS = 3


def draw_budget(images: int) -> int:
    """The most images a probe may ever draw from a folder of ``images``.

    Read it as the sentence it is: at most PROBE_MAX_DRAWS draws, or a quarter of
    the folder, whichever is SMALLER — and never fewer than the single sample the
    probe has always taken, so no folder is sampled less than it is today."""
    if images <= 0:
        return 0
    share = math.ceil(int(images) * PROBE_MAX_FOLDER_SHARE)
    return min(int(images),
               max(SAMPLE_SIZE, min(PROBE_MAX_DRAWS, share)))


def _topup_draws(usable, tried, budget_left) -> int:
    """How many MORE images to draw for a folder still short of its target.

    Sized from what the folder has actually shown: at a measured hit rate of
    ``usable/tried`` it takes about ``need/rate`` more draws to reach the target,
    so that is what is asked for — bounded by what is left of the budget.

    A folder that produced NOTHING has no rate to extrapolate from, and the
    honest reading of "0 of 15" is that faces are rare here, not that one more
    handful will change it. It gets the whole remaining budget in one go rather
    than three rounds of hope: same ceiling, one model load instead of three, and
    the strongest statement the budget can buy ("none in 60" beats "none in 15")."""
    need = PROBE_TARGET_FACES - usable
    if need <= 0 or budget_left <= 0:
        return 0
    if usable <= 0 or tried <= 0:
        return budget_left
    return min(budget_left, -(-need * tried // usable))     # ceil(need / rate)


def _svc():
    """The bank service, imported late — it imports this module too."""
    from . import image_bank_service as banks
    return banks


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --- reading ----------------------------------------------------------------
def asserted_subfolders(bank_id) -> set:
    """{subfolder} the user has declared to hold a single person."""
    return {row.subfolder for row in
            BankFolderPerson.query.filter_by(bank_id=bank_id).all()}


def assertion_for(bank_id, subfolder):
    return (BankFolderPerson.query
            .filter_by(bank_id=bank_id, subfolder=subfolder or '').first())


def _report_of(row) -> dict | None:
    if not row.sample_report:
        return None
    try:
        report = json.loads(row.sample_report)
        if not isinstance(report, dict):
            return None
        # Keep the old report as history but name when it no longer describes
        # the asserted folder's effective image generation.
        report['stale'] = (
            report.get('content_sig') != _folder_signature(
                row.bank_id, row.subfolder))
        return report
    except (TypeError, ValueError):
        return None


def _folder_rows_q(bank_id, subfolder):
    """Every image row of one TOP-LEVEL subfolder — the same set the Subfolder
    filter scopes to, expressed the same way (prefix match, '' = bank root)."""
    q = BankImage.query.filter_by(bank_id=bank_id)
    if (subfolder or '') == '':
        return q.filter(~BankImage.relpath.contains(os.sep))
    return q.filter(BankImage.relpath.startswith(subfolder + os.sep))


def _to_check(bank_id, subfolder) -> list:
    """Images of the folder the face machinery already looked at and could NOT
    read as one clean face: no face at all, a face too small/too turned to
    identify, an unreadable file. The assertion covers them anyway — they are
    listed, never excluded, because "I could not see a face here" is not "this
    is someone else". Only rows a pass (or the sample check) actually measured
    appear: a NULL face_state means "not looked at", a different thing."""
    rows = (_folder_rows_q(bank_id, subfolder)
            .filter(BankImage.face_state.isnot(None),
                    BankImage.face_state != 'scorable')
            .order_by(BankImage.id.asc()).limit(200).all())
    # 'name' is the BASENAME, not the relpath — the key is named for what it
    # holds. The grid already knows how to open an image from its id.
    return [{'id': r.id, 'state': r.face_state,
             'name': os.path.basename(r.relpath)} for r in rows]


def payload(user_id, bank_id) -> dict | None:
    """Every assertion of a bank, for the Subfolder panel."""
    banks = _svc()
    if not banks.get_bank(user_id, bank_id):
        return None
    out = []
    for row in (BankFolderPerson.query.filter_by(bank_id=bank_id)
                .order_by(BankFolderPerson.subfolder.asc()).all()):
        covered = (_folder_rows_q(bank_id, row.subfolder)
                   .filter(BankImage.face_cluster_origin == ASSERTED).count())
        out.append({
            'subfolder': row.subfolder,
            'cluster_id': row.cluster_id,
            'images': covered,
            'sample': _report_of(row),
            'to_check': _to_check(bank_id, row.subfolder),
        })
    return {'assertions': out, 'sample_size': SAMPLE_SIZE,
            # The per-folder CEILING of the re-drawing probe. The scan offer
            # quotes both, because "~15 images each" stopped being the whole
            # truth the day a draw could be replaced.
            'sample_max': PROBE_MAX_DRAWS,
            # Folders the app probed by itself. They are OFFERS, not decisions:
            # nothing here has grouped a single image.
            'suggestions': suggestions(bank_id),
            'scannable': len(scan_candidates(bank_id)),
            'scan_limit': MAX_SCAN_FOLDERS}


# --- writing ----------------------------------------------------------------
def _next_cluster_id(bank_id) -> int:
    """One above every person id currently in use in this bank — computed and
    asserted alike, so an assertion can never land on an id the last embeddings
    pass already handed out."""
    used = (db.session.query(func.max(BankImage.face_cluster))
            .filter(BankImage.bank_id == bank_id).scalar() or 0)
    reserved = (db.session.query(func.max(BankFolderPerson.cluster_id))
                .filter(BankFolderPerson.bank_id == bank_id).scalar() or 0)
    return int(max(used, reserved)) + 1


def asserted_offset(bank_id) -> int:
    """How far the embeddings pass must push its own 1-based cluster ids so they
    never collide with an asserted group's.

    Transfers preserve asserted image rows even though the source folder rule
    itself is intentionally Bank-local.  Such an orphaned-but-authoritative id
    still occupies the shared cluster namespace and therefore participates in
    the offset exactly like a live ``BankFolderPerson`` rule.
    """
    rule_max = (db.session.query(func.max(BankFolderPerson.cluster_id))
                .filter(BankFolderPerson.bank_id == bank_id).scalar() or 0)
    row_max = (db.session.query(func.max(BankImage.face_cluster))
               .filter(BankImage.bank_id == bank_id,
                       BankImage.face_cluster_origin == ASSERTED)
               .scalar() or 0)
    return int(max(rule_max, row_max))


@_serialized_bank_mutation('folder_person')
def assert_single_person(user_id, bank_id, subfolder, *,
                         _bank_lease=None) -> dict:
    """Declare a subfolder to hold one person. Immediate, no inference at all.

    Idempotent: asserting an already-asserted folder just re-stamps it (useful
    after images arrived while a pass was running)."""
    banks = _svc()
    if not banks.get_bank(user_id, bank_id):
        raise ValueError('bank not found')
    sub = subfolder or ''
    q = _folder_rows_q(bank_id, sub)
    total = q.count()
    if not total:
        raise ValueError('this subfolder has no images')
    row = assertion_for(bank_id, sub)
    if row is None:
        row = BankFolderPerson(bank_id=bank_id, subfolder=sub,
                               cluster_id=_next_cluster_id(bank_id))
        db.session.add(row)
        db.session.flush()
    q.update({BankImage.face_cluster: row.cluster_id,
              BankImage.face_cluster_origin: ASSERTED},
             synchronize_session=False)
    db.session.commit()
    logger.info('bank %s: subfolder asserted as one person, %s image(s), '
                'person #%s', bank_id, total, row.cluster_id)
    return {'subfolder': sub, 'cluster_id': row.cluster_id, 'images': total}


@_serialized_bank_mutation('folder_person_revoke')
def revoke(user_id, bank_id, subfolder, *, _bank_lease=None) -> dict:
    """Undo the assertion: the group dissolves and the folder goes back to normal
    clustering. Only ids this module wrote are cleared — a row whose cluster the
    embeddings pass computed (before the assertion, or in a folder that partly
    overlaps) keeps it."""
    banks = _svc()
    if not banks.get_bank(user_id, bank_id):
        raise ValueError('bank not found')
    sub = subfolder or ''
    row = assertion_for(bank_id, sub)
    if row is None:
        raise ValueError('this subfolder is not asserted')
    cleared = (_folder_rows_q(bank_id, sub)
               .filter(BankImage.face_cluster_origin == ASSERTED)
               .update({BankImage.face_cluster: None,
                        BankImage.face_cluster_origin: None},
                       synchronize_session=False))
    db.session.delete(row)
    db.session.commit()
    logger.info('bank %s: assertion revoked, %s image(s) back to clustering',
                bank_id, cleared)
    return {'subfolder': sub, 'cleared': int(cleared)}


def drop_for_bank(bank_id) -> int:
    """Delete every assertion of a bank. Called BEFORE the bank row itself (the
    delete-500 lesson: children first, no relationship to flush them for us)."""
    return (BankFolderPerson.query.filter_by(bank_id=bank_id)
            .delete(synchronize_session=False))


def stamp_new_rows(bank_id, rows) -> int:
    """Apply the standing assertions to freshly inventoried rows, IN PLACE, before
    they are inserted. This is what makes an assertion a rule and not a one-off
    stamp: an image dropped into an asserted folder tomorrow joins its group the
    moment the folder sync sees it, with no pass and no click.

    ``rows`` are the plain dicts _insert_bank_images is about to core-insert."""
    if not rows:
        return 0
    by_sub = {r.subfolder: r.cluster_id for r in
              BankFolderPerson.query.filter_by(bank_id=bank_id).all()}
    if not by_sub:
        return 0
    banks = _svc()
    stamped = 0
    for row in rows:
        cid = by_sub.get(banks._subfolder_of(row.get('relpath') or ''))
        # Both keys are written on EVERY row, not only the matching ones: these
        # dicts go to one executemany, which takes its column list from the first
        # of them — a half-stamped batch would drop the ids of all the others.
        row['face_cluster'] = cid
        row['face_cluster_origin'] = ASSERTED if cid is not None else None
        if cid is not None:
            stamped += 1
    return stamped


# --- sample check -----------------------------------------------------------
_SAMPLE_PROGRESS_RE = re.compile(r'\[embed\] (\d+)/(\d+)')


def _draw_order(n, cap, step=SAMPLE_SIZE) -> list:
    """Indices 0..n-1 in the ORDER a probe should draw them, at most ``cap``.

    The first ``step`` are the evenly-spaced pick this module has always made:
    scraped folders are ordered by name, which is usually order of arrival, so
    the first 15 files are one shoot, one day, often one outfit, and a second
    person appearing halfway through would never be drawn. Evenly spaced picks
    cover the whole folder, and being deterministic the same folder always gets
    the same verdict.

    Every FURTHER slice refines that same spread (the 30-point stratification,
    then the 45-point one…) instead of appending a clump at one end. That is what
    lets a re-draw replace unreadable images without giving up the property the
    first draw was built for: no index is ever returned twice, and any prefix of
    this list is still spread across the whole folder."""
    limit = min(int(cap), n)
    seen, order = set(), []
    k = max(1, int(step))
    while len(order) < limit:
        added = 0
        for i in range(k):
            idx = (i * n) // k
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
            added += 1
            if len(order) >= limit:
                return order
        if k >= n:            # k = n enumerates every index; nothing left to add
            break
        k = n if k + step > n else k + step
    return order


def _stratified(rows, k=SAMPLE_SIZE) -> list:
    """``k`` rows spread evenly across the folder — the first layer of
    ``_draw_order``, kept as its own name because that is what the single-folder
    sample check asks for (one draw, no re-draw)."""
    return [rows[i] for i in _draw_order(len(rows), k, step=k)]


def start_sample_check(app, user_id, bank_id, subfolder):
    """Embed ~15 images of the folder and report whether they look like ONE
    person, at the clustering threshold. Runs as a normal bank job (one per
    bank) because it loads the same model the full pass does."""
    banks = _svc()
    from .face_similarity import is_available
    from . import bank_jobs
    bank = banks.get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if assertion_for(bank_id, subfolder or '') is None:
        raise ValueError('this subfolder is not asserted')
    if not is_available():
        raise RuntimeError(
            'face scoring is not installed (Quality tools step in Setup)')
    return bank_jobs.start(app, bank_id, 'folder-check',
                           _sample_job(bank_id, subfolder or ''),
                           total=SAMPLE_SIZE)


def _verdict(largest, scorable, faces, tried=None, exhausted=False) -> tuple:
    """(verdict, sentence) — plain English, and never more certain than the images
    it saw allow. It says what the SAMPLE showed; it never says the folder is clean.

    ``tried`` is how many images were actually handed to the detector (a re-drawing
    probe tries more than a single sample); ``exhausted`` says the probe stopped
    on its BUDGET while still short of its target, which is the difference between
    a verdict and a weak verdict.

    Four outcomes, and the last two are the ones this file gained:
      consistent   — enough usable faces, and they agree;
      mixed        — several faces, whatever the sample size;
      partial      — they agree, but on fewer usable faces than asked for. A weak
                     verdict that says how weak beats no verdict at all, which is
                     what this used to produce;
      inconclusive — almost nothing readable. That is INFORMATION ABOUT THE
                     FOLDER, not a failure of the check, and the sentence says so:
                     the full pass reads these images through the SAME detector at
                     the same gates (backend/infer/face_embed_infer.py is the one
                     script all three callers drive), and the probe writes its
                     answers into the pass's OWN embedding cache — so it will get
                     the identical states back without re-detecting. It is not
                     "we could not tell, ask the expensive one"."""
    n = SAMPLE_SIZE if tried is None else int(tried)
    if scorable < 2:
        head = ('no readable face' if scorable <= 0
                else f'only {scorable} readable face')
        return 'inconclusive', (
            f'{head} in {n} images tried across the folder — crops, backs or '
            'blur. The full pass reads them the same way, so much of this '
            'folder will not group by face')
    if faces > 1:
        return 'mixed', (
            f'{faces} different faces in the sample — check this folder')
    if exhausted:
        return 'partial', (
            f'looks like one person, on thin evidence — only {scorable} usable '
            f'{"face" if scorable == 1 else "faces"} in {n} images tried')
    return 'consistent', (
        f'sample consistent ({largest}/{scorable} same person)')


def _sample_job(bank_id, subfolder):
    def run(job):
        from contextlib import nullcontext
        from . import bank_jobs
        from ..gpu_window import gpu_exclusive_vision_window
        from ..models import ImageBank
        banks = _svc()
        bank = db.session.get(ImageBank, bank_id)
        row = assertion_for(bank_id, subfolder)
        if not bank or row is None:
            return
        pool = (_folder_rows_q(bank_id, subfolder)
                .order_by(BankImage.relpath.asc()).all())
        picked = _stratified(pool)
        by_path = {}
        for r in picked:
            p = banks.analysis_image_path(bank, r, refresh_rotation=True)
            if banks._is_safe_bank_source(p, label='folder sample check'):
                by_path[p] = r.id
        paths = list(by_path)
        sample_generation = _folder_signature(
            bank_id, subfolder, include_analysis=False)
        bank_jobs.progress(job, done=0, total=len(paths), detail='sample check')
        if not paths:
            bank_jobs.progress(job, detail='no readable image to sample')
            return
        banks._bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        th = banks.thresholds()
        device, use_gpu = banks._resolve_face_device()
        # Its OWN cache, never the bank-wide face cache: a 15-row .npz written
        # here must not race (or truncate) the full pass's thousands of rows.
        cache_path = banks._bank_dir(bank_id) / 'folder_sample.npz'
        req = json.dumps({
            'images': paths,
            'models_root': banks.cfg.get('face_scoring.models_root') or None,
            'cache': str(cache_path),
            'cancel_file': str(cache_path) + '.cancel',
            'threshold': th['face_threshold'],
            'device': device,
        })
        import sys
        python = banks.cfg.get('face_scoring.python') or sys.executable
        window = (gpu_exclusive_vision_window(flag_ttl=600) if use_gpu
                  else nullcontext())
        banks._release_db_before_inference()
        data, stderr_tail, returncode = banks._drive_infer_subprocess(
            job, python, banks._EMBED_SCRIPT, req, cache_path,
            _SAMPLE_PROGRESS_RE, window)
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            bank_jobs.progress(job, detail='sample check stopped — '
                                           'the assertion is unchanged')
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'sample check produced no output '
                                       f'(rc={returncode})')
        results = data.get('results') or {}
        clusters = data.get('clusters') or {}
        # The states are REAL measurements on real images of this folder: write
        # them back (never the cluster id — that belongs to the assertion), so
        # the "to check" list has substance even on a bank whose face pass never
        # ran. This is also why the sample is not wasted work.
        sample_valid = (
            _folder_signature(bank_id, subfolder, include_analysis=False)
            == sample_generation)
        for p, image_id in by_path.items():
            live = banks._live_image(image_id)
            res = results.get(p) or {}
            if (live is None or live.status == 'reject'
                    or banks._subfolder_of(live.relpath) != subfolder
                    or not banks._prepare_analysis_write(
                        live, p, res.get('fingerprint'))):
                sample_valid = False
                continue
            if live.face_state is None:
                live.face_state = res.get('state')
                live.face_det = res.get('det')
        db.session.commit()
        if (not sample_valid
                or _folder_signature(
                    bank_id, subfolder, include_analysis=False)
                != sample_generation):
            bank_jobs.progress(
                job, detail='sample changed while it was checked — no report saved')
            return
        sizes = {}
        for cid in clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        scorable = sum(sizes.values())
        faces = len(sizes)
        largest = max(sizes.values()) if sizes else 0
        # One draw, no re-draw: this check answers "was I right about the folder
        # I already declared?", and the user asked for exactly these fifteen
        # embeddings. So it never reports 'partial' — nothing was cut short.
        verdict, sentence = _verdict(largest, scorable, faces, tried=len(paths))
        fresh = assertion_for(bank_id, subfolder)
        if fresh is not None:      # revoked while the check ran — say nothing
            fresh.sample_report = json.dumps({
                'checked_at': _now_iso(), 'sample': len(paths),
                'scorable': scorable, 'largest': largest, 'faces': faces,
                'threshold': th['face_threshold'],
                'verdict': verdict, 'note': sentence,
                'content_sig': _folder_signature(bank_id, subfolder)})
            db.session.commit()
        bank_jobs.progress(job, detail=sentence)
    return run


# --- automatic suggestion (the same probe, over every folder) ---------------
# The single-folder check above answers "was I right?" AFTER the user declared.
# The same fifteen embeddings answer "would you like to declare?" BEFORE they
# did — which is the question that actually saves them work, because on a
# scraped bank most folders are one person and the user has no reason to guess
# which. So the probe runs over the folders on its own and the app SUGGESTS.
#
# It suggests. It never asserts. A wrong assertion made silently would corrupt
# the person grouping with something the user never said, and they would have no
# reason to look for it — so confirming stays one deliberate click.
def _folder_signature(bank_id, subfolder, *, include_analysis=True) -> str:
    """DB-only identity of the exact non-rejected effective pool a probe saw.

    Count/max-id missed rotations, cleans, status changes and threshold changes,
    so a verdict could remain labelled fresh while every sampled pixel changed.
    A truncated SHA-256 fits the existing 40-character column; its input contains
    no file bytes and therefore keeps this UI freshness check cheap.
    """
    rows = (_folder_rows_q(bank_id, subfolder)
            .filter(BankImage.status != 'reject')
            .with_entities(BankImage.id, BankImage.relpath,
                           BankImage.analysis_fingerprint, BankImage.rotation,
                           BankImage.watermark_clean_method)
            .order_by(BankImage.id.asc()).all())
    try:
        threshold = _svc().thresholds().get('face_threshold')
    except Exception:  # noqa: BLE001 — freshness degrades to a stable sentinel
        threshold = None
    digest = hashlib.sha256()
    digest.update(f'face-threshold:{threshold!r}\n'.encode('utf-8'))
    for image_id, relpath, fingerprint, rotation, clean_method in rows:
        digest.update(
            f'{image_id}\0{relpath}\0'
            f'{fingerprint if include_analysis else ""}\0'
            f'{rotation or 0}\0{clean_method or ""}\n'.encode(
                'utf-8', errors='surrogatepass'))
    # Existing schema stores 40 characters.  This is a change token, while the
    # security authority for a sampled image remains its full SHA-256 below.
    return digest.hexdigest()[:40]


def _sample_pool(bank_id, subfolder):
    """The rows a probe may sample: the folder's NON-REJECTED images, minus the
    ones already MEASURED AS UNUSABLE.

    Rejected ones are excluded on purpose — they are outside what the face pass
    looks at, so sampling them would both cost embeddings nothing else has cached
    and let images the user already threw away drive a suggestion.

    The second exclusion is what makes a re-draw worth anything on a folder the
    machinery has already looked at: a row whose face_state is set and is not
    'scorable' has been through the detector and produced no usable face. Drawing
    it again cannot give a different answer (same script, same gates, and its
    embedding is cached), so it would only burn a draw. A NULL face_state means
    "never looked at" — those stay in, they are the whole pool on a fresh bank.

    It gives WAY when it would leave too little to sample. A folder the machinery
    has already read as unusable end to end would otherwise have an empty pool,
    draw nothing, get no verdict — and stay a candidate the preflight offers to
    sample again on every single launch. Falling back to the full pool costs
    nothing there (every one of those embeddings is cached) and the folder gets
    the honest "almost no readable face" it earned."""
    banks = _svc()
    from ..models import ImageBank
    bank = db.session.get(ImageBank, bank_id)
    base = _folder_rows_q(bank_id, subfolder).filter(BankImage.status != 'reject')
    all_rows = base.order_by(BankImage.relpath.asc()).all()
    rows = []
    for row in all_rows:
        if row.face_state is None or row.face_state == 'scorable':
            rows.append(row)
            continue
        # A measured-unusable verdict only saves a re-draw while it is proven to
        # describe the row's current effective bytes. Legacy/unbound and stale
        # states stay eligible so a clean/rotation cannot exclude them forever.
        path = banks.analysis_image_path(bank, row) if bank is not None else None
        if (not row.analysis_fingerprint
                or banks.bank_transfer_metadata.content_fingerprint_path(path)
                != row.analysis_fingerprint):
            rows.append(row)
    if len(rows) >= MIN_PROBE_IMAGES:
        return rows
    return all_rows


def probe_for(bank_id, subfolder):
    return (BankFolderProbe.query
            .filter_by(bank_id=bank_id, subfolder=subfolder or '').first())


def _probe_dict(row, fresh: bool, images: int = 0) -> dict:
    return {'subfolder': row.subfolder, 'verdict': row.verdict,
            'sample': row.sample, 'scorable': row.scorable,
            'largest': row.largest, 'faces': row.faces, 'note': row.note,
            # How many images accepting THIS folder would spare the pass. The
            # preflight totals them, so its offer can be answered against the
            # only number that matters: what it saves.
            'images': int(images),
            'checked_at': row.checked_at.isoformat() if row.checked_at else None,
            'stale': not fresh}


def _folder_counts(bank_id) -> dict:
    """{subfolder: non-rejected image count} — one query, the same pool the probe
    and the face pass work on."""
    from collections import Counter
    counts: Counter = Counter()
    for (rel,) in (db.session.query(BankImage.relpath)
                   .filter(BankImage.bank_id == bank_id,
                           BankImage.status != 'reject').all()):
        counts[_svc()._subfolder_of(rel)] += 1
    return dict(counts)


def suggestions(bank_id) -> list:
    """Every folder the app has probed and NOT been told about, with the stale
    ones marked rather than dropped silently."""
    asserted = asserted_subfolders(bank_id)
    counts = _folder_counts(bank_id)
    out = []
    for row in (BankFolderProbe.query.filter_by(bank_id=bank_id)
                .order_by(BankFolderProbe.subfolder.asc()).all()):
        if row.subfolder in asserted:
            continue          # already declared — a suggestion would be noise
        fresh = row.content_sig == _folder_signature(bank_id, row.subfolder)
        out.append(_probe_dict(row, fresh, counts.get(row.subfolder, 0)))
    return out


def scan_candidates(bank_id, limit=None) -> list:
    """(subfolder, image_count) for the folders worth probing, biggest first.

    Skipped: folders the user already declared (nothing to suggest), folders
    below MIN_PROBE_IMAGES (too small for a sample to mean anything), and
    folders whose probe still matches their content (already answered)."""
    counts = _folder_counts(bank_id)
    asserted = asserted_subfolders(bank_id)
    out = []
    for name, n in counts.items():
        if name in asserted or n < MIN_PROBE_IMAGES:
            continue
        got = probe_for(bank_id, name)
        if got is not None and got.content_sig == _folder_signature(bank_id, name):
            continue
        out.append((name, n))
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out if limit is None else out[:limit]


def _write_probe(bank_id, subfolder, sizes, sample_n, exhausted=False) -> str:
    """Persist one folder's verdict and return it.

    ``sample`` keeps meaning what it always meant — how many images were actually
    handed to the detector — which with a re-drawing probe is the number of images
    TRIED, not the size of one draw. No column is added: 'partial' is a new value
    in a column that already holds free text, and every database in the wild reads
    it without a migration."""
    scorable = sum(sizes.values())
    faces = len(sizes)
    largest = max(sizes.values()) if sizes else 0
    verdict, sentence = _verdict(largest, scorable, faces, tried=sample_n,
                                 exhausted=exhausted)
    row = probe_for(bank_id, subfolder)
    if row is None:
        row = BankFolderProbe(bank_id=bank_id, subfolder=subfolder)
        db.session.add(row)
    row.verdict, row.note = verdict, sentence
    row.sample, row.scorable = sample_n, scorable
    row.largest, row.faces = largest, faces
    row.content_sig = _folder_signature(bank_id, subfolder)
    row.checked_at = datetime.now(timezone.utc)
    return verdict


def drop_probes_for_bank(bank_id) -> int:
    return (BankFolderProbe.query.filter_by(bank_id=bank_id)
            .delete(synchronize_session=False))


def _probe_states(bank_id, bank, candidates) -> dict:
    """{folder: draw state} with the FIRST draw already taken.

    One state per folder, carrying everything a re-draw needs: the pool it draws
    from, the order it draws in, how far into that order it has got (which is also
    how much budget it has spent — the order is built no longer than the budget),
    the paths gathered so far and the clustering the last round returned."""
    state = {}
    for name, images in candidates:
        pool = _sample_pool(bank_id, name)
        budget = draw_budget(images)
        s = {'name': name, 'bank': bank, 'pool': pool, 'next': 0,
             'order': _draw_order(len(pool), budget), 'paths': {}, 'clusters': {},
             'pending': False,
             'generation_sig': _folder_signature(
                 bank_id, name, include_analysis=False)}
        _draw_more(s, SAMPLE_SIZE)
        state[name] = s
    return state


def _draw_more(s, want) -> int:
    """Take up to ``want`` more images from the folder, never one already tried.

    A draw the file guard refuses still SPENDS its place in the order: it was an
    attempt, it is bounded like every other, and letting the unsafe ones be free
    would turn a folder of unreadable files into an unbounded walk."""
    banks = _svc()
    added = 0
    while added < want and s['next'] < len(s['order']):
        row = s['pool'][s['order'][s['next']]]
        s['next'] += 1
        p = banks.analysis_image_path(
            s['bank'], row, refresh_rotation=True)
        if p in s['paths']:
            continue
        if banks._is_safe_bank_source(p, label='folder person probe'):
            s['paths'][p] = row.id
            added += 1
    if added:
        s['pending'] = True
    return added


def _usable(s) -> int:
    """Images of this folder the detector could actually read a face in — the
    clustering only ever contains those, which is why it is the count that
    matters and len(paths) is not."""
    return len(s['clusters'] or {})


def _short_of_target(s) -> bool:
    """Did the probe stop with fewer usable faces than it wanted, while the folder
    still had images it never drew? That — and only that — is a PARTIAL verdict:
    a folder drawn to the last image has been answered completely, however few
    faces it turned out to hold."""
    return _usable(s) < PROBE_TARGET_FACES and s['next'] < len(s['pool'])


def _apply_probe_results(bank_id, state, results) -> dict:
    """Write the states back and persist one probe per folder. Returns
    {verdict: count} for the job's report."""
    banks = _svc()
    tally = {}
    for s in state.values():
        if len(s['paths']) < 2:      # nothing to compare below two faces
            continue
        folder_valid = (_folder_signature(
            bank_id, s['name'], include_analysis=False)
            == s['generation_sig'])
        for p, image_id in s['paths'].items():
            live = banks._live_image(image_id)
            res = results.get(p) or {}
            if (live is None or live.status == 'reject'
                    or banks._subfolder_of(live.relpath) != s['name']
                    or not banks._prepare_analysis_write(
                        live, p, res.get('fingerprint'))):
                folder_valid = False
                continue
            if live.face_state is None:
                live.face_state = res.get('state')
                live.face_det = res.get('det')
        # Per-folder all-or-none: one changed sample member invalidates the
        # clustering evidence for this folder, while other folders in the same
        # bounded child call remain publishable.
        if (not folder_valid
                or _folder_signature(
                    bank_id, s['name'], include_analysis=False)
                != s['generation_sig']):
            continue
        sizes = {}
        for cid in (s['clusters'] or {}).values():
            sizes[cid] = sizes.get(cid, 0) + 1
        verdict = _write_probe(bank_id, s['name'], sizes, len(s['paths']),
                               exhausted=_short_of_target(s))
        tally[verdict] = tally.get(verdict, 0) + 1
    db.session.commit()
    return tally


def _probe_detail(tally, scanned, left) -> str:
    """What the scan found, in the user's terms — and what it did NOT reach."""
    if not scanned:
        return 'no folder left to look at'
    likely = tally.get('consistent', 0) + tally.get('partial', 0)
    bits = [f'{likely} folder(s) look like one person' if likely
            else 'no folder looked like a single person']
    if tally.get('partial'):
        bits.append(f'{tally["partial"]} of them on thin evidence')
    if tally.get('mixed'):
        n = tally['mixed']
        bits.append(f'{n} {"holds" if n == 1 else "hold"} several')
    if tally.get('inconclusive'):
        # Not "the check failed" — the folder has almost nothing a face detector
        # can read, and saying "too few faces to tell" invited a full pass that
        # reads the very same images through the very same detector.
        n = tally['inconclusive']
        bits.append(f'{n} {"has" if n == 1 else "have"} almost no readable face')
    out = f'{scanned} folder(s) sampled — ' + ', '.join(bits)
    if likely:
        out += ' — confirm the ones you recognise'
    if left:
        # Never mute a ceiling: a scan that covered 20 of 200 folders and said
        # nothing would read as "the other 180 are not one person".
        out += f' · {left} folder(s) not reached (biggest first — run it again)'
    return out


def _embed_round(job, bank_id, groups, *, allow_inference: bool):
    """One child call over ``groups``, or None if the user stopped it.

    Every round sends a folder's WHOLE accumulated path list, not just the images
    it drew this time: the clustering has to be over everything the folder has
    shown, and re-sending an image the child already embedded costs nothing —
    the .npz cache is exactly what makes a second round cheap."""
    from contextlib import nullcontext
    from ..gpu_window import gpu_exclusive_vision_window
    banks = _svc()
    banks._bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
    th = banks.thresholds()
    device, use_gpu = banks._resolve_face_device()
    # The BANK's own face cache, deliberately: after 👤 Group by person every one
    # of these images is already in it, so the probe is free. One job runs per
    # bank, so nothing else can be writing this file underneath us. It also means
    # a re-draw is never wasted work — every extra embedding it pays for is one
    # the full pass will now skip.
    cache_path = banks._face_cache_path(bank_id)
    req = json.dumps({
        'images': [p for g in groups for p in g['images']],
        'groups': groups,
        'models_root': banks.cfg.get('face_scoring.models_root') or None,
        'cache': str(cache_path),
        'cancel_file': str(cache_path) + '.cancel',
        'threshold': th['face_threshold'],
        'device': device,
    })
    import sys
    python = banks.cfg.get('face_scoring.python') or sys.executable
    window = (gpu_exclusive_vision_window(flag_ttl=900)
              if (use_gpu and allow_inference) else nullcontext())
    banks._release_db_before_inference()
    data, stderr_tail, returncode = banks._drive_infer_subprocess(
        job, python, banks._EMBED_SCRIPT, req, cache_path,
        _SAMPLE_PROGRESS_RE, window)
    if data.get('cancelled'):
        return None
    if not data.get('ok'):
        tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
        raise RuntimeError(tail or f'folder scan produced no output '
                                   f'(rc={returncode})')
    return data


def _run_probe(job, bank_id, candidates, *, allow_inference: bool):
    """Sample ``candidates`` and persist their verdicts, RE-DRAWING the folders
    whose draw came back unreadable until they reach ~15 usable faces or run out
    of budget (draw_budget / PROBE_MAX_ROUNDS).

    Folders that already have their answer drop out after the first round, so the
    second one carries only the hard folders — the ones that used to end with no
    verdict at all and the full pass behind them.

    ``allow_inference=False`` is the automatic path: it runs straight after the
    face pass, whose cache already holds every embedding it needs, so the child
    loads no model and touches no GPU. If an image is somehow missing from that
    cache the child would embed it — cheap at this size, and still bounded by the
    budget, but the flag is what lets the caller say honestly which it was."""
    from ..models import ImageBank
    bank = db.session.get(ImageBank, bank_id)
    if not bank or not candidates:
        return None
    state = _probe_states(bank_id, bank, candidates)
    results = {}
    for rnd in range(PROBE_MAX_ROUNDS):
        groups = [{'name': s['name'], 'images': list(s['paths'])}
                  for s in state.values() if s['pending'] and len(s['paths']) >= 2]
        if not groups:
            break
        data = _embed_round(job, bank_id, groups, allow_inference=allow_inference)
        if data is None:
            return None
        results.update(data.get('results') or {})
        measured = data.get('group_clusters') or {}
        for s in state.values():
            if not s['pending']:
                continue
            s['pending'] = False
            if s['name'] in measured:
                s['clusters'] = measured[s['name']]
        if rnd + 1 >= PROBE_MAX_ROUNDS:
            # No round left to measure them in, so draw nothing: an image drawn
            # and never handed to the detector would still be counted among the
            # "images tried" the verdict quotes, which is a lie about its own
            # evidence.
            break
        # Replace what could not be read, for the folders still short. A folder
        # whose budget is spent, or that has shown enough faces, asks for nothing
        # and is simply not in the next round.
        for s in state.values():
            if _short_of_target(s):
                _draw_more(s, _topup_draws(_usable(s), len(s['paths']),
                                           len(s['order']) - s['next']))
    if not any(len(s['paths']) >= 2 for s in state.values()):
        return None
    return _apply_probe_results(bank_id, state, results)


def start_folder_scan(app, user_id, bank_id, limit=None, kind='folder-scan'):
    """Sample every unprobed folder and suggest the ones that look like a single
    person. Costs ~15 embeddings per folder — more where the first images have no
    readable face and have to be replaced, never past ``draw_budget`` — over at
    most ``limit`` folders.

    ``limit``/``kind`` exist for the PREFLIGHT, which is the same work with a
    different ceiling and a different name in the progress bar — not a second
    implementation. Both write ordinary probes, and a probe is an offer whoever
    produced it."""
    from .face_similarity import is_available
    from . import bank_jobs
    banks = _svc()
    if not banks.get_bank(user_id, bank_id):
        raise ValueError('bank not found')
    if not is_available():
        raise RuntimeError(
            'face scoring is not installed (Quality tools step in Setup)')
    cap = limit or MAX_SCAN_FOLDERS
    pending = scan_candidates(bank_id)
    if not pending:
        raise ValueError('every folder here has already been looked at '
                         '(or is asserted, or too small to sample)')
    return bank_jobs.start(app, bank_id, kind,
                           _folder_scan_job(bank_id, limit=cap),
                           total=min(len(pending), cap))


def _folder_scan_job(bank_id, limit=None):
    def run(job):
        from . import bank_jobs
        # Read at CALL time, never captured at import: the ceiling is a module
        # constant tests move around, and a default frozen into the signature
        # would quietly ignore them.
        cap = limit or MAX_SCAN_FOLDERS
        pending = scan_candidates(bank_id)
        picked = pending[:cap]
        left = len(pending) - len(picked)
        bank_jobs.progress(job, done=0, total=len(picked),
                           detail=f'sampling {len(picked)} folder(s)')
        tally = _run_probe(job, bank_id, picked, allow_inference=True)
        if tally is None:
            bank_jobs.progress(job, detail='folder scan stopped — '
                                           'nothing was changed')
            return
        bank_jobs.progress(job, detail=_probe_detail(tally, len(picked), left))
    return run


def probe_after_faces(job, bank_id) -> str:
    """Run the probe over EVERY unprobed folder right after the face pass, in
    that pass's own job. This is where the suggestion belongs: the embeddings it
    needs were just computed and cached, so covering two hundred folders costs no
    inference at all — the child loads no model when nothing is left to embed.

    It never raises: the face pass has already succeeded by the time this runs,
    and a failed suggestion must not turn a finished pass into a red one."""
    try:
        pending = scan_candidates(bank_id)
        if not pending:
            return ''
        tally = _run_probe(job, bank_id, pending, allow_inference=False)
        if not tally:
            return ''
        # A partial verdict is offered exactly like a confident one, so it counts
        # here too — announcing 3 when the dialog will pre-tick 5 would send the
        # user looking for the two that "went missing".
        likely = tally.get('consistent', 0) + tally.get('partial', 0)
        if not likely:
            return ''
        return (f' · {likely} folder(s) look like a single person — '
                f'confirm to skip them next time')
    except Exception as e:      # noqa: BLE001 — a suggestion is never worth a failure
        logger.warning('bank %s: folder probe after the face pass failed: %s',
                       bank_id, e, exc_info=True)
        return ''


# --- the preflight (the probe, moved IN FRONT of the pass) ------------------
# WHY THIS EXISTS. Everything above was reachable only from the Subfolder panel
# or a 🔎 Scan folders button. The first thing a new user does is press 🚀 Launch
# all — so they never saw any of it, and paid the full pass on forty folders that
# each held one person. A saving nobody walks past is not a saving.
#
# So the sampling now runs where the decision is made: as the preamble of 👤
# Group by person, standalone or inside Launch all. Same probe, same verdicts,
# same ASSERTION at the end of it (nothing here invents a second kind of "this
# folder is one person" — the acceptance goes through assert_single_person, which
# is why ↩ Not one person after all still works on it).
#
# The safety rule of the module is unchanged and non-negotiable: it still never
# asserts by itself. What changed is that the confirmation is now ON THE ROAD the
# user is already travelling, pre-ticked, one click for all of it — instead of
# down a side street they had no reason to take.
def face_pass_cost(bank_id) -> int:
    """How many images 👤 Group by person would embed if it started right now:
    every non-rejected image no assertion already covers. This is the number the
    preflight is measured against, and it is computed exactly the way the pass
    computes its own total — so the comparison the UI prints is not a rhetorical
    one."""
    return (BankImage.query.filter_by(bank_id=bank_id)
            .filter(BankImage.status != 'reject',
                    or_(BankImage.face_cluster_origin.is_(None),
                        BankImage.face_cluster_origin != ASSERTED)).count())


def preflight_payload(user_id, bank_id) -> dict | None:
    """What the preflight would cost and what it already knows, BEFORE anything
    runs. The caller uses it to decide whether there is a question worth asking
    at all: on a bank with no subfolder (or one already fully declared) there is
    nothing to show, and showing a dialog anyway would be the detour this feature
    exists to remove."""
    banks = _svc()
    if not banks.get_bank(user_id, bank_id):
        return None
    from .face_similarity import is_available
    pending = scan_candidates(bank_id)
    covered = min(len(pending), MAX_PREFLIGHT_FOLDERS)
    picked = pending[:covered]
    return {
        # Without the extra there is no probe and no pass either; the caller
        # skips straight through and lets the pass report the missing install.
        'available': bool(is_available()),
        'sample_size': SAMPLE_SIZE,
        'sample_max': PROBE_MAX_DRAWS,
        'min_images': MIN_PROBE_IMAGES,
        'candidates': len(pending),
        'covered': covered,
        # Folders the ceiling will NOT reach. Reported as a number the UI has to
        # print, not as silence that would read as "the rest are not one person".
        'left': len(pending) - covered,
        'sample_cost': sum(min(SAMPLE_SIZE, n) for _name, n in picked),
        # The CEILING of the re-drawing probe, announced before it is paid. The
        # preflight's whole justification is "a few seconds against a full pass",
        # so a mechanism that can multiply its cost may not hide behind the
        # typical case: the dialog prints both numbers and the pass it is being
        # compared against. Reached only on folders where faces are genuinely
        # hard to find — which is exactly where the ceiling has to be visible.
        'sample_cost_max': sum(draw_budget(n) for _name, n in picked),
        'full_cost': face_pass_cost(bank_id),
        # Fresh verdicts already on file (from an earlier preflight, a 🔎 Scan,
        # or the free probe at the end of a previous pass). They cost nothing to
        # show and they are the whole answer when no folder is left to sample.
        'known': [s for s in suggestions(bank_id) if not s['stale']],
        'asserted': sorted(asserted_subfolders(bank_id)),
    }


def start_preflight(app, user_id, bank_id):
    """Sample every unprobed folder as the preamble of the person pass."""
    return start_folder_scan(app, user_id, bank_id,
                             limit=MAX_PREFLIGHT_FOLDERS, kind='folder-preflight')


@_serialized_bank_mutation('folder_person_accept')
def accept_suggestions(user_id, bank_id, subfolders, *,
                       _bank_lease=None) -> dict:
    """Confirm the preflight's offers in ONE gesture.

    Each folder goes through ``assert_single_person`` — the same persisted,
    revocable, origin='asserted' rule a manual click writes. There is deliberately
    no second state: afterwards nothing distinguishes a folder accepted here from
    one declared by hand, which is what keeps revoke, re-scan adoption and the
    pass's skip working on both.

    A folder that cannot be asserted (emptied since the probe, say) is REPORTED,
    not swallowed: the caller has to be able to say "11 of the 12 you ticked"."""
    banks = _svc()
    if not banks.get_bank(user_id, bank_id):
        raise ValueError('bank not found')
    if not isinstance(subfolders, (list, tuple)):
        raise ValueError('subfolders must be a list')
    accepted, images, failed = [], 0, []
    for sub in subfolders:
        name = '' if sub is None else str(sub)
        try:
            out = assert_single_person(
                user_id, bank_id, name, _bank_lease=_bank_lease)
        except ValueError as e:
            failed.append({'subfolder': name, 'error': str(e)})
            continue
        accepted.append(out['subfolder'])
        images += out['images']
    return {'accepted': accepted, 'images': images, 'failed': failed}
