"""🗃️ Image bank service — triage a big unsorted folder into dataset-ready
selections.

The founding use case: "I exported 9 000 images from Telegram — now what?".
A bank references that folder IN PLACE (no copy, the source files are never
written to) and layers a funnel on top:

  1. inventory      — instant: walk the folder, register every image file;
  2. quality pass   — background thread, CPU/pure-PIL: sharpness, noise,
                      uniformity, dimensions, dHash + duplicate groups;
                      raw scores persist, FLAGS are computed at read time
                      against the config thresholds ('bank' section) so
                      recalibrating never needs a rescan;
  3. duplicates     — near-duplicate groups (same 64-bit dHash family as the
                      dataset import dedup) with keep-best / keep-first /
                      manual resolution — losers are REJECTED (a status),
                      never deleted from disk;
  4. subject pass   — optional, needs the face-scoring extra: InsightFace
                      embeddings (cached in an .npz next to the thumbs) +
                      person clustering, to sort a mixed dump by WHO is in
                      the frame without any reference photo;
  5. promotion      — the kept selection is COPIED into a dataset through the
                      normal import path (normalize to webp + perceptual
                      dedup vs the dataset), inheriting every downstream tool
                      (captions, watermarks, face scoring, training).

Long passes run through bank_jobs (one background thread per bank, polled via
the bank payload) — a 9 000-image folder must scan in minutes without ever
holding an HTTP request open or freezing the UI.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from sqlalchemy import and_, func, or_

from .. import config as cfg
from ..extensions import db
from ..models import BankImage, FaceDataset, ImageBank
from . import bank_jobs
from .face_dataset_service import _dhash, _hamming, import_images
from .image_quality import ANALYSIS_MAX_SIDE, quality_metrics

logger = logging.getLogger(__name__)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
# Sanity cap — a bank is a triage layer, not a filesystem indexer. Way above
# the founding 9 000-image case, low enough to catch "I pointed it at C:\".
BANK_MAX_FILES = 50000
THUMB_MAX_SIDE = 320
_COMMIT_EVERY = 25          # scan DB flush cadence
_PROMOTE_CHUNK = 20         # files per import_images call (bounded memory)
_SQL_IN_CHUNK = 500         # SQLite bound-variable ceiling is 999


# --- thresholds -------------------------------------------------------------
def thresholds() -> dict:
    """The 'bank' config section, sanitized (a corrupt config.json value falls
    back to the default instead of poisoning every flag computation)."""
    out = {}
    for key, default in cfg.DEFAULTS['bank'].items():
        try:
            out[key] = float(cfg.get(f'bank.{key}', default))
        except (TypeError, ValueError):
            out[key] = float(default)
    out['dup_distance'] = int(out['dup_distance'])
    out['min_side'] = int(out['min_side'])
    return out


# --- storage helpers --------------------------------------------------------
def _bank_dir(bank_id) -> Path:
    return cfg.banks_root() / str(bank_id)


def _thumbs_dir(bank_id) -> Path:
    return _bank_dir(bank_id) / 'thumbs'


def _face_cache_path(bank_id) -> Path:
    return _bank_dir(bank_id) / 'face_cache.npz'


def _score_cache_path(bank_id) -> Path:
    return _bank_dir(bank_id) / 'score_cache.npz'


def abs_image_path(bank: ImageBank, row: BankImage) -> str | None:
    """Absolute source path of a bank image, or None when it escapes the
    bank's folder (belt & braces — relpaths only ever come from our own walk)."""
    base = os.path.realpath(bank.source_path)
    full = os.path.realpath(os.path.join(base, row.relpath))
    if os.path.normcase(full).startswith(os.path.normcase(base + os.sep)):
        return full
    return None


# --- CRUD -------------------------------------------------------------------
def get_bank(user_id, bank_id) -> ImageBank | None:
    return ImageBank.query.filter_by(id=bank_id, user_id=user_id).first()


def create_bank(user_id, name, folder):
    """Register a folder as a bank: walk it recursively and create one row per
    image file. Instant (no decode) — scoring is the separate scan pass.
    Returns (bank, added). ValueError on a missing folder / too many files."""
    name = (name or '').strip()
    # Windows «Copier en tant que chemin» pastes the path quoted — unquote so
    # the direct paste works first try (same nicety as the dataset folder import).
    folder = (folder or '').strip().strip('"\'')
    if not name:
        raise ValueError('name is required')
    if not folder or not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder or "(empty)"}')
    folder = os.path.realpath(folder)
    rels = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                rels.append(os.path.relpath(os.path.join(root, f), folder))
                if len(rels) > BANK_MAX_FILES:
                    raise ValueError(
                        f'too many images in the folder (max {BANK_MAX_FILES})')
    bank = ImageBank(user_id=user_id, name=name, source_path=folder)
    db.session.add(bank)
    db.session.flush()          # need bank.id for the child rows
    for i, rel in enumerate(rels, 1):
        try:
            size = os.path.getsize(os.path.join(folder, rel))
        except OSError:
            size = None
        db.session.add(BankImage(bank_id=bank.id, relpath=rel, file_size=size))
        if i % 500 == 0:
            db.session.flush()
    db.session.commit()
    return bank, len(rels)


def delete_bank(user_id, bank_id) -> bool:
    """Drop the bank's ROWS and working data (thumbs + face cache). The source
    folder and its images are never touched."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return False
    if bank_jobs.running(bank_id):
        bank_jobs.cancel(bank_id)
    BankImage.query.filter_by(bank_id=bank_id).delete(synchronize_session=False)
    db.session.delete(bank)
    db.session.commit()
    shutil.rmtree(_bank_dir(bank_id), ignore_errors=True)
    return True


# --- flags & payloads -------------------------------------------------------
def image_flags(row: BankImage, th: dict) -> list:
    """Threshold verdicts for one image, recomputed from the raw scores."""
    if row.quality_state == 'unreadable':
        return ['unreadable']
    flags = []
    if row.quality_state == 'ok':
        if row.blur_score is not None and row.blur_score < th['sharpness_min']:
            flags.append('blur')
        if row.noise_score is not None and row.noise_score > th['noise_max']:
            flags.append('noise')
        if row.uniformity_score is not None and row.uniformity_score < th['uniformity_min']:
            flags.append('uniform')
        if row.width and row.height and min(row.width, row.height) < th['min_side']:
            flags.append('small')
    # V2 scoring flags — derived from the persisted scores against the live
    # thresholds too, but NOT gated on the quality state (a watermarked or NSFW
    # image can be perfectly sharp). Only present once the relevant pass has run.
    if row.aesthetic_score is not None and row.aesthetic_score < th['aesthetic_min']:
        flags.append('low_aesthetic')
    if row.nsfw_score is not None and row.nsfw_score > th['nsfw_max']:
        flags.append('nsfw')
    if row.watermark_state == 'detected':
        flags.append('watermark')
    return flags


def _image_dict(row: BankImage, th: dict) -> dict:
    return {
        'id': row.id,
        'name': os.path.basename(row.relpath),
        'relpath': row.relpath,
        'width': row.width, 'height': row.height, 'file_size': row.file_size,
        'quality_state': row.quality_state,
        'blur_score': row.blur_score, 'noise_score': row.noise_score,
        'uniformity_score': row.uniformity_score,
        'aesthetic_score': row.aesthetic_score, 'nsfw_score': row.nsfw_score,
        'style_cluster': row.style_cluster, 'watermark_state': row.watermark_state,
        'subfolder': _subfolder_of(row.relpath),
        'flags': image_flags(row, th),
        'dup_group': row.dup_group,
        'face_state': row.face_state, 'face_cluster': row.face_cluster,
        'status': row.status, 'reject_reason': row.reject_reason,
        'promoted_dataset_id': row.promoted_dataset_id,
        'caption': row.caption,
    }


def _flag_filter(flag: str, th: dict):
    """SQLAlchemy criterion for one flag name (mirrors image_flags)."""
    if flag == 'unreadable':
        return BankImage.quality_state == 'unreadable'
    # V2 scoring flags — not gated on quality_state (see image_flags) and only
    # true where the score actually exists (a NULL score is "not scored", never
    # "below threshold"). watermark is a discrete state, no threshold column.
    if flag == 'low_aesthetic':
        return and_(BankImage.aesthetic_score.isnot(None),
                    BankImage.aesthetic_score < th['aesthetic_min'])
    if flag == 'nsfw':
        return and_(BankImage.nsfw_score.isnot(None),
                    BankImage.nsfw_score > th['nsfw_max'])
    if flag == 'watermark':
        return BankImage.watermark_state == 'detected'
    ok = BankImage.quality_state == 'ok'
    crit = {
        'blur': BankImage.blur_score < th['sharpness_min'],
        'noise': BankImage.noise_score > th['noise_max'],
        'uniform': BankImage.uniformity_score < th['uniformity_min'],
        'small': or_(BankImage.width < th['min_side'],
                     BankImage.height < th['min_side']),
    }.get(flag)
    return (ok & crit) if crit is not None else None


_QUALITY_FLAGS = ('blur', 'noise', 'uniform', 'small', 'unreadable')
# V2 score-derived flags. Kept separate from _QUALITY_FLAGS so the "flagged" /
# "clean" quality aggregate stays about the CPU quality pass, while these count
# and filter independently (each only meaningful once its pass has run).
_SCORE_FLAGS = ('low_aesthetic', 'nsfw', 'watermark')


def _subfolder_of(relpath: str) -> str:
    """Top-level subfolder of a bank-relative path ('' for a root-level file) —
    the natural scoping axis for a Telegram export (one folder per chat/date)."""
    parts = (relpath or '').replace('\\', '/').split('/', 1)
    return parts[0] if len(parts) > 1 else ''


def _unresolved_dup_groups_q(bank_id):
    """Groups still holding ≥2 NON-rejected members — i.e. still to resolve."""
    return (db.session.query(BankImage.dup_group)
            .filter(BankImage.bank_id == bank_id,
                    BankImage.dup_group.isnot(None),
                    BankImage.status != 'reject')
            .group_by(BankImage.dup_group)
            .having(func.count(BankImage.id) >= 2))


def bank_payload(user_id, bank_id) -> dict | None:
    """Everything the bank workspace needs on one poll: counts, flag totals,
    duplicate/cluster summaries, live job, thresholds."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    base = BankImage.query.filter_by(bank_id=bank_id)
    total = base.count()
    counts = {
        'total': total,
        'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
        'pending': base.filter_by(status='pending').count(),
        'keep': base.filter_by(status='keep').count(),
        'reject': base.filter_by(status='reject').count(),
        'promoted': base.filter(BankImage.promoted_dataset_id.isnot(None)).count(),
        # V2 pass progress — how many images the scoring / watermark passes reached
        # (so the UI can show "scored 0/9000" and enable the threshold facets).
        'scored': base.filter(or_(BankImage.aesthetic_score.isnot(None),
                                  BankImage.nsfw_score.isnot(None))).count(),
        'watermark_scanned': base.filter(BankImage.watermark_state.isnot(None)).count(),
    }
    flags = {}
    for flag in _QUALITY_FLAGS + _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        flags[flag] = base.filter(crit).count() if crit is not None else 0
    dup_rows = (db.session.query(BankImage.dup_group, func.count(BankImage.id))
                .filter(BankImage.bank_id == bank_id,
                        BankImage.dup_group.isnot(None))
                .group_by(BankImage.dup_group).all())
    dup = {'groups': len(dup_rows),
           'images': sum(n for _g, n in dup_rows),
           'unresolved': _unresolved_dup_groups_q(bank_id).count()}
    # Person clusters, biggest first; cover = the member with the surest face.
    cl_rows = (db.session.query(BankImage.face_cluster, func.count(BankImage.id))
               .filter(BankImage.bank_id == bank_id,
                       BankImage.face_cluster.isnot(None))
               .group_by(BankImage.face_cluster)
               .order_by(func.count(BankImage.id).desc(),
                         BankImage.face_cluster.asc())
               .limit(40).all())
    clusters = []
    for cid, size in cl_rows:
        cover = (BankImage.query
                 .filter_by(bank_id=bank_id, face_cluster=cid)
                 .order_by(BankImage.face_det.desc().nullslast(),
                           BankImage.id.asc())
                 .first())
        clusters.append({'id': cid, 'size': size,
                         'cover_image_id': cover.id if cover else None})
    faces_scanned = base.filter(BankImage.face_state.isnot(None)).count()
    # Style clusters (group by visual style), biggest first — the "group by
    # style" counterpart to the person clusters above. Cover = the lowest id of
    # the cluster (stable, no per-image quality signal to rank by here).
    st_rows = (db.session.query(BankImage.style_cluster, func.count(BankImage.id))
               .filter(BankImage.bank_id == bank_id,
                       BankImage.style_cluster.isnot(None))
               .group_by(BankImage.style_cluster)
               .order_by(func.count(BankImage.id).desc(),
                         BankImage.style_cluster.asc())
               .limit(40).all())
    style_clusters = []
    for cid, size in st_rows:
        cover = (BankImage.query
                 .filter_by(bank_id=bank_id, style_cluster=cid)
                 .order_by(BankImage.id.asc()).first())
        style_clusters.append({'id': cid, 'size': size,
                               'cover_image_id': cover.id if cover else None})
    return {
        'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
        'created_at': bank.created_at.isoformat() if bank.created_at else None,
        'counts': counts, 'flags': flags, 'dup': dup,
        'clusters': clusters, 'faces_scanned': faces_scanned,
        'style_clusters': style_clusters,
        'activity': bank_jobs.get(bank_id),
        'pipeline_report': _load_pipeline_report(bank),
        'thresholds': th,
    }


def _load_pipeline_report(bank: ImageBank):
    """The persisted 'Launch all' summary (parsed), or None. A corrupt blob is
    swallowed — a broken report must never 500 the whole bank payload."""
    import json as _json
    raw = getattr(bank, 'pipeline_report', None)
    if not raw:
        return None
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        return None


def list_banks(user_id) -> list:
    out = []
    for bank in (ImageBank.query.filter_by(user_id=user_id)
                 .order_by(ImageBank.created_at.desc()).all()):
        base = BankImage.query.filter_by(bank_id=bank.id)
        out.append({
            'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
            'created_at': bank.created_at.isoformat() if bank.created_at else None,
            'total': base.count(),
            'keep': base.filter_by(status='keep').count(),
            'reject': base.filter_by(status='reject').count(),
            'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
            'activity': bank_jobs.get(bank.id),
        })
    return out


def list_images(user_id, bank_id, status=None, flag=None, cluster=None,
                group=None, style=None, subfolder=None, search=None,
                offset=0, limit=200) -> dict | None:
    """One PAGE of the bank grid (a 9 000-image bank must never ship whole).
    Filters compose: status ∩ flag ∩ cluster ∩ dup-group ∩ style ∩ subfolder ∩ search.
    ``search`` is a plain full-text term matched (case-insensitive LIKE) against the
    caption AND the relpath — so captions double as searchable tags for a big dump
    ("red dress"), combinable with every other filter. Flag filters sort by the
    relevant score (worst first) so the review reads top-down."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    q = BankImage.query.filter_by(bank_id=bank_id)
    if status in ('pending', 'keep', 'reject'):
        q = q.filter(BankImage.status == status)
    order = BankImage.id.asc()
    if flag == 'flagged':
        crits = [c for c in (_flag_filter(f, th) for f in _QUALITY_FLAGS)
                 if c is not None]
        q = q.filter(or_(*crits))
    elif flag == 'clean':
        q = q.filter(BankImage.quality_state == 'ok')
        for f in ('blur', 'noise', 'uniform', 'small'):
            q = q.filter(~_flag_filter(f, th))
    elif flag == 'dups':
        q = q.filter(BankImage.dup_group.isnot(None))
        order = (BankImage.dup_group.asc(), BankImage.id.asc())
    elif flag == 'no_face':
        # Literally "no face was found" — ONLY face_state == 'no_face'. The other
        # non-scorable states (low_det / too_small / extreme_pose) DID detect a
        # face; lumping them in here surfaced photos with visible faces under a
        # "No face" chip. 'unreadable'/'error' are read failures, not "no face".
        q = q.filter(BankImage.face_state == 'no_face')
    elif flag in _QUALITY_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
        order = {'blur': BankImage.blur_score.asc(),
                 'noise': BankImage.noise_score.desc(),
                 'uniform': BankImage.uniformity_score.asc(),
                 'small': BankImage.width.asc(),
                 'unreadable': BankImage.id.asc()}[flag]
    elif flag in _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
        # Worst first: least aesthetic / most-confident NSFW at the top.
        order = {'low_aesthetic': BankImage.aesthetic_score.asc(),
                 'nsfw': BankImage.nsfw_score.desc(),
                 'watermark': BankImage.id.asc()}[flag]
    if cluster is not None:
        q = q.filter(BankImage.face_cluster == int(cluster))
    if group is not None:
        q = q.filter(BankImage.dup_group == int(group))
    if style is not None:
        q = q.filter(BankImage.style_cluster == int(style))
    if subfolder is not None:
        # '' scopes to root-level files; any other value to that top-level folder
        # and everything nested under it. startswith() escapes LIKE metachars.
        if subfolder == '':
            q = q.filter(~BankImage.relpath.contains(os.sep))
        else:
            q = q.filter(BankImage.relpath.startswith(subfolder + os.sep))
    term = (search or '').strip()
    if term:
        # Full-text over caption + relpath. Escape LIKE metacharacters so a literal
        # '%'/'_' in the query matches itself, then wrap in wildcards.
        esc = term.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        like = f'%{esc}%'
        q = q.filter(or_(BankImage.caption.ilike(like, escape='\\'),
                         BankImage.relpath.ilike(like, escape='\\')))
    total = q.count()
    order_by = order if isinstance(order, tuple) else (order,)
    rows = q.order_by(*order_by).offset(max(0, int(offset))) \
            .limit(max(1, min(500, int(limit)))).all()
    return {'images': [_image_dict(r, th) for r in rows], 'total': total,
            'offset': max(0, int(offset))}


# --- thumbnails -------------------------------------------------------------
def ensure_thumb(bank: ImageBank, row: BankImage) -> Path | None:
    """The image's grid thumbnail, generated lazily when the scan hasn't made
    it yet (so the grid is browsable straight after inventory)."""
    tpath = _thumbs_dir(bank.id) / f'{row.id}.webp'
    if tpath.is_file():
        return tpath
    src = abs_image_path(bank, row)
    if not src or not os.path.isfile(src):
        return None
    try:
        tpath.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.draft(None, (THUMB_MAX_SIDE * 2, THUMB_MAX_SIDE * 2))
            im = im.convert('RGB')
            im.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
            im.save(tpath, 'WEBP', quality=72)
        return tpath
    except (OSError, ValueError):
        return None


# --- quality scan (background) ----------------------------------------------
def _scan_one(src_root: str, thumbs: Path, item: tuple) -> dict:
    """Worker: decode ONE file, compute metrics + dHash + thumbnail. Pure
    filesystem/PIL — no DB access (the job thread owns the session)."""
    image_id, relpath = item
    path = os.path.join(src_root, relpath)
    out = {'id': image_id, 'quality_state': 'unreadable', 'width': None,
           'height': None, 'file_size': None, 'dhash': None, 'metrics': None}
    try:
        out['file_size'] = os.path.getsize(path)
    except OSError:
        pass
    try:
        with Image.open(path) as im:
            out['width'], out['height'] = im.size
            # JPEG fast path: decode at reduced scale — the metrics run on a
            # ≤1024 working copy anyway, and dHash (9×8) is resize-invariant.
            im.draft(None, (ANALYSIS_MAX_SIDE * 2, ANALYSIS_MAX_SIDE * 2))
            im.load()
            out['metrics'] = quality_metrics(im)
            out['dhash'] = f'{_dhash(im):016x}'
            tpath = thumbs / f'{image_id}.webp'
            if not tpath.is_file():
                t = im.convert('RGB')
                t.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), Image.LANCZOS)
                t.save(tpath, 'WEBP', quality=72)
        out['quality_state'] = 'ok'
    except (OSError, ValueError, SyntaxError):
        pass  # stays 'unreadable' — surfaced as a flag, never fatal
    return out


def start_scan(app, user_id, bank_id, rescan=False):
    """Launch the quality pass. Raises BankJobBusy when a job is already live,
    ValueError when the bank is unknown."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    q = BankImage.query.filter_by(bank_id=bank_id)
    if not rescan:
        q = q.filter(BankImage.quality_state.is_(None))
    total = q.count()
    return bank_jobs.start(app, bank_id, 'scan',
                           _scan_job(bank_id, rescan), total=total)


def _scan_job(bank_id, rescan):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        q = BankImage.query.filter_by(bank_id=bank_id)
        if not rescan:
            q = q.filter(BankImage.quality_state.is_(None))
        items = [(r.id, r.relpath) for r in q.order_by(BankImage.id.asc()).all()]
        bank_jobs.progress(job, done=0, total=len(items), detail='quality scan')
        thumbs = _thumbs_dir(bank_id)
        thumbs.mkdir(parents=True, exist_ok=True)
        src_root = bank.source_path
        workers = min(8, os.cpu_count() or 4)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            it = iter(items)
            futures = deque()

            def submit_next():
                nxt = next(it, None)
                if nxt is not None:
                    futures.append(ex.submit(_scan_one, src_root, thumbs, nxt))

            for _ in range(workers * 2):
                submit_next()
            while futures:
                res = futures.popleft().result()
                row = db.session.get(BankImage, res['id'])
                if row is not None:
                    row.quality_state = res['quality_state']
                    row.width, row.height = res['width'], res['height']
                    if res['file_size'] is not None:
                        row.file_size = res['file_size']
                    row.dhash = res['dhash']
                    if res['metrics']:
                        row.blur_score = res['metrics']['blur_score']
                        row.noise_score = res['metrics']['noise_score']
                        row.uniformity_score = res['metrics']['uniformity_score']
                    # An unreadable file can never be promoted — auto-reject it
                    # (only over 'pending': a manual decision is never flipped).
                    if res['quality_state'] == 'unreadable' and row.status == 'pending':
                        row.status, row.reject_reason = 'reject', 'unreadable'
                done += 1
                if done % _COMMIT_EVERY == 0:
                    db.session.commit()
                bank_jobs.bump(job)
                if not bank_jobs.cancelled(job):
                    submit_next()
        db.session.commit()
        if not bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail='grouping duplicates')
            groups = rebuild_dup_groups(bank_id)
            bank_jobs.progress(job, detail=f'done — {groups} duplicate group(s)')
    return run


# --- duplicate groups -------------------------------------------------------
def rebuild_dup_groups(bank_id, max_distance=None) -> int:
    """Recompute near-duplicate groups over every hashed image of the bank.
    Banded prefilter (pigeonhole: two hashes within Hamming d share at least
    one of d+1 equal bands) keeps this out of the full O(n²) — then candidate
    pairs are verified exactly and grouped by union-find. Groups of ≥2 get a
    1-based id ordered by size (biggest first). Returns the group count."""
    th = thresholds()
    d = int(th['dup_distance'] if max_distance is None else max_distance)
    rows = (db.session.query(BankImage.id, BankImage.dhash)
            .filter(BankImage.bank_id == bank_id, BankImage.dhash.isnot(None))
            .order_by(BankImage.id.asc()).all())
    ids = [r[0] for r in rows]
    hashes = [int(r[1], 16) for r in rows]
    n = len(ids)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bands = max(1, min(16, d + 1))
    band_bits = 64 // bands
    buckets: dict = {}
    for i, h in enumerate(hashes):
        for b in range(bands):
            key = (b, (h >> (b * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(i)
    seen_pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if find(a) == find(b) or (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                if _hamming(hashes[a], hashes[b]) <= d:
                    union(a, b)
    comps: dict = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    groups = sorted((m for m in comps.values() if len(m) >= 2),
                    key=lambda m: (-len(m), m[0]))
    BankImage.query.filter_by(bank_id=bank_id).update(
        {'dup_group': None}, synchronize_session=False)
    for gid, members in enumerate(groups, start=1):
        member_ids = [ids[i] for i in members]
        for i0 in range(0, len(member_ids), _SQL_IN_CHUNK):
            BankImage.query.filter(
                BankImage.id.in_(member_ids[i0:i0 + _SQL_IN_CHUNK])).update(
                {'dup_group': gid}, synchronize_session=False)
    db.session.commit()
    return len(groups)


def dup_groups_payload(user_id, bank_id, offset=0, limit=50) -> dict | None:
    """Unresolved groups (≥2 non-rejected members) with their full membership,
    for the resolution panel."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    gids = [g for (g,) in _unresolved_dup_groups_q(bank_id)
            .order_by(BankImage.dup_group.asc()).all()]
    total = len(gids)
    page = gids[max(0, int(offset)):max(0, int(offset)) + max(1, min(200, int(limit)))]
    groups = []
    for gid in page:
        rows = (BankImage.query.filter_by(bank_id=bank_id, dup_group=gid)
                .order_by(BankImage.id.asc()).all())
        groups.append({'group': gid,
                       'best_id': _best_of(rows).id if rows else None,
                       'images': [_image_dict(r, th) for r in rows]})
    return {'groups': groups, 'total': total, 'offset': max(0, int(offset))}


def _best_of(rows):
    """'Keep best' heuristic for a duplicate group. When the aesthetic pass has
    run it leads (Jeremy's ask: keep the NICE copy, not merely the biggest); a
    scored image always outranks an unscored one (sentinel -1 < the ~1..10 range).
    Then most pixels, sharpest, heaviest file — a Telegram dump's duplicates are
    mostly re-compressed or downscaled copies, so surface area is the honest
    fallback key."""
    def key(r):
        return (r.aesthetic_score if r.aesthetic_score is not None else -1.0,
                (r.width or 0) * (r.height or 0), r.blur_score or 0.0,
                r.file_size or 0, -r.id)
    return max(rows, key=key)


def resolve_dups(user_id, bank_id, strategy='best', group=None, keep_ids=None):
    """Resolve duplicate groups: keep one member, REJECT the others (reason
    'duplicate' — a status, never a file deletion, so it's reversible).
    strategy 'best'|'first' applies to one group or, when ``group`` is None,
    to every unresolved group at once; explicit ``keep_ids`` (manual pick)
    applies to their own groups. Only non-rejected members are touched; a
    member the user already KEPT stays kept (never flipped by a bulk resolve).
    Returns {'resolved': groups, 'rejected': images}."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    keep_by_group = {}
    if keep_ids:
        rows = BankImage.query.filter(BankImage.bank_id == bank_id,
                                      BankImage.id.in_(list(keep_ids)[:_SQL_IN_CHUNK])).all()
        for r in rows:
            if r.dup_group:
                keep_by_group.setdefault(r.dup_group, set()).add(r.id)
        gids = list(keep_by_group)
    elif group is not None:
        gids = [int(group)]
    else:
        gids = [g for (g,) in _unresolved_dup_groups_q(bank_id).all()]
    resolved = rejected = 0
    for gid in gids:
        rows = (BankImage.query.filter_by(bank_id=bank_id, dup_group=gid)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        if len(rows) < 2 and gid not in keep_by_group:
            continue
        if gid in keep_by_group:
            keep = keep_by_group[gid]
        elif strategy == 'first':
            keep = {rows[0].id}
        else:
            keep = {_best_of(rows).id}
        changed = False
        for r in rows:
            if r.id in keep or r.status == 'keep':
                continue
            r.status, r.reject_reason = 'reject', 'duplicate'
            rejected += 1
            changed = True
        if changed or len(rows) >= 2:
            resolved += 1
    db.session.commit()
    return {'resolved': resolved, 'rejected': rejected}


# --- statuses & flag application --------------------------------------------
def set_status(user_id, bank_id, ids, status) -> int:
    """Manual keep/reject/pending on a selection. Returns rows changed."""
    if status not in ('pending', 'keep', 'reject'):
        raise ValueError('bad status')
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ids = [int(i) for i in (ids or [])]
    n = 0
    for i0 in range(0, len(ids), _SQL_IN_CHUNK):
        rows = BankImage.query.filter(
            BankImage.bank_id == bank_id,
            BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all()
        for r in rows:
            r.status = status
            r.reject_reason = 'manual' if status == 'reject' else None
            n += 1
    db.session.commit()
    return n


def apply_flags(user_id, bank_id, flags) -> dict:
    """Bulk-reject the PENDING images carrying the given flags. Manual ✓/✕
    decisions are never flipped (only status='pending' is touched) — same
    contract as the dataset auto-triage. Returns per-flag reject counts."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    th = thresholds()
    out = {}
    for flag in flags or []:
        if flag not in _QUALITY_FLAGS + _SCORE_FLAGS:
            continue
        crit = _flag_filter(flag, th)
        if crit is None:
            continue
        rows = (BankImage.query.filter_by(bank_id=bank_id, status='pending')
                .filter(crit).all())
        for r in rows:
            r.status, r.reject_reason = 'reject', flag
        out[flag] = len(rows)
    db.session.commit()
    return out


# --- subject (face) pass ----------------------------------------------------
_EMBED_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'face_embed_infer.py')
_PROGRESS_RE = re.compile(r'\[embed\] (\d+)/(\d+)')


def _resolve_face_device():
    """(device, use_gpu) for the face pass. 'cpu' is the safe default and never
    touches the GPU; GPU is used ONLY when the face interpreter truly exposes
    CUDA (onnxruntime-gpu installed) and the config allows it. Config
    face_scoring.device: 'auto' (default — GPU if available) | 'cpu' | 'cuda'.
    A 'cuda' request without CUDA available still degrades to CPU here, so the
    parent never opens the GPU-exclusive window for a pass that will run on CPU."""
    from .. import capabilities
    pref = str(cfg.get('face_scoring.device') or 'auto').lower()
    use_gpu = pref in ('auto', 'cuda') and capabilities.face_gpu_available()
    return ('cuda' if use_gpu else 'cpu'), use_gpu


def start_faces(app, user_id, bank_id):
    """Launch the face embedding + person clustering pass over the bank's
    non-rejected images. Needs the face-scoring extra (Setup ▸ Quality tools)."""
    from .face_similarity import is_available
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not is_available():
        raise RuntimeError(
            'face scoring is not installed (Quality tools step in Setup)')
    total = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.status != 'reject').count())
    return bank_jobs.start(app, bank_id, 'faces', _faces_job(bank_id), total=total)


def _faces_job(bank_id):
    def run(job):
        import json as _json
        import sys
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        by_path = {}
        for r in rows:
            p = abs_image_path(bank, r)
            if p and os.path.isfile(p):
                by_path[p] = r.id
        paths = list(by_path)
        bank_jobs.progress(job, done=0, total=len(paths), detail='face pass')
        if not paths:
            return
        _bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        th = thresholds()
        # Device: 'cpu' (default, never touches the GPU/ComfyUI) or 'cuda'.
        # 'auto' = GPU when the face interpreter actually exposes CUDA
        # (onnxruntime-gpu installed), else CPU. The GPU path is used ONLY when
        # CUDA is truly available AND must run inside the GPU-exclusive window so
        # it never competes with a training / scoring pass; a CPU pass stays out
        # of the window (it can run alongside GPU work).
        from ..gpu_window import gpu_exclusive_vision_window
        from contextlib import nullcontext
        device, use_gpu = _resolve_face_device()
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('face_scoring.models_root') or None,
            'cache': str(_face_cache_path(bank_id)),
            'threshold': th['face_threshold'],
            'device': device,
        })
        python = cfg.get('face_scoring.python') or sys.executable
        window = gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu else nullcontext()
        with window:
            proc = subprocess.Popen(
                [python, _EMBED_SCRIPT], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            bank_jobs.set_cancel_hook(job, proc.kill)
            stderr_tail = deque(maxlen=5)

            def _drain_stderr():
                for line in proc.stderr:
                    line = line.strip()
                    if line:
                        stderr_tail.append(line)
                    m = _PROGRESS_RE.search(line)
                    if m:
                        bank_jobs.progress(job, done=int(m.group(1)),
                                           total=int(m.group(2)))

            import threading
            t = threading.Thread(target=_drain_stderr, daemon=True)
            t.start()
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
            except OSError:
                pass  # process died early — surfaced through the exit path below
            stdout = proc.stdout.read()
            proc.wait()
            t.join(timeout=5)
        if bank_jobs.cancelled(job):
            return
        line = next((ln for ln in reversed(stdout.splitlines())
                     if ln.strip().startswith('{')), '')
        try:
            data = _json.loads(line) if line else {}
        except _json.JSONDecodeError:
            data = {}
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'face pass produced no output '
                                       f'(rc={proc.returncode})')
        results = data.get('results') or {}
        clusters = data.get('clusters') or {}
        done = 0
        for p, image_id in by_path.items():
            row = db.session.get(BankImage, image_id)
            if row is None:
                continue
            res = results.get(p) or {}
            row.face_state = res.get('state')
            row.face_det = res.get('det')
            row.face_cluster = clusters.get(p)
            done += 1
            if done % 200 == 0:
                db.session.commit()
        db.session.commit()
        sizes = {}
        for cid in clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        multi = sum(1 for n in sizes.values() if n >= 2)
        bank_jobs.progress(job, detail=f'done — {multi} person cluster(s) '
                                       f'of 2+ images')
    return run


# --- scoring pass (aesthetic · NSFW · style) --------------------------------
_SCORE_SCRIPT = str(cfg.BACKEND_DIR / 'infer' / 'bank_score_infer.py')
_SCORE_PROGRESS_RE = re.compile(r'\[score\] (\d+)/(\d+)')


def _gpu_busy_reason() -> str | None:
    """A human reason the GPU is unavailable right now, or None. Same system
    flags training and the vision window use, so a bank GPU pass never races a
    training run or a captioning pass (the 'never concurrent with a training'
    guarantee). Checked up-front for an immediate 503 rather than a doomed 202."""
    from ..job_queue import queue_manager
    if queue_manager._get_system_state('training_in_progress'):
        return 'training is running on the GPU — try again once it finishes'
    if queue_manager._get_system_state('vision_in_progress'):
        return 'a vision/GPU pass is already running — try again in a moment'
    return None


def start_score(app, user_id, bank_id):
    """Launch the scoring pass (LAION aesthetic + NSFW + style clustering) over
    the bank's non-rejected images. Needs the bank-scoring extra (Setup ▸ Quality
    tools). Serialized against training/vision, so it refuses (503) when the GPU
    is held."""
    from ..capabilities import probe_bank_scoring
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not probe_bank_scoring().get('ok'):
        raise RuntimeError('bank scoring is not installed '
                           '(Quality tools step in Setup)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    total = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.status != 'reject').count())
    return bank_jobs.start(app, bank_id, 'score', _score_job(bank_id), total=total)


def _score_job(bank_id):
    def run(job):
        import json as _json
        import sys
        import threading
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .filter(BankImage.status != 'reject')
                .order_by(BankImage.id.asc()).all())
        by_path = {}
        for r in rows:
            p = abs_image_path(bank, r)
            if p and os.path.isfile(p):
                by_path[p] = r.id
        paths = list(by_path)
        bank_jobs.progress(job, done=0, total=len(paths), detail='scoring pass')
        if not paths:
            return
        _bank_dir(bank_id).mkdir(parents=True, exist_ok=True)
        th = thresholds()
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('bank_scoring.models_root') or None,
            'cache': str(_score_cache_path(bank_id)),
            'style_threshold': th['style_threshold'],
        })
        python = cfg.get('bank_scoring.python') or sys.executable
        # GPU-exclusive: frees ComfyUI VRAM and blocks a training start for the
        # duration, exactly like the dataset vision passes.
        with gpu_exclusive_vision_window(flag_ttl=1800):
            proc = subprocess.Popen(
                [python, _SCORE_SCRIPT], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            bank_jobs.set_cancel_hook(job, proc.kill)
            stderr_tail = deque(maxlen=5)

            def _drain_stderr():
                for line in proc.stderr:
                    line = line.strip()
                    if line:
                        stderr_tail.append(line)
                    m = _SCORE_PROGRESS_RE.search(line)
                    if m:
                        bank_jobs.progress(job, done=int(m.group(1)),
                                           total=int(m.group(2)))

            t = threading.Thread(target=_drain_stderr, daemon=True)
            t.start()
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
            except OSError:
                pass
            stdout = proc.stdout.read()
            proc.wait()
            t.join(timeout=5)
        if bank_jobs.cancelled(job):
            return
        line = next((ln for ln in reversed(stdout.splitlines())
                     if ln.strip().startswith('{')), '')
        try:
            data = _json.loads(line) if line else {}
        except _json.JSONDecodeError:
            data = {}
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'scoring pass produced no output '
                                       f'(rc={proc.returncode})')
        results = data.get('results') or {}
        clusters = data.get('clusters') or {}
        done = 0
        for p, image_id in by_path.items():
            row = db.session.get(BankImage, image_id)
            if row is None:
                continue
            res = results.get(p) or {}
            row.aesthetic_score = res.get('aesthetic')
            row.nsfw_score = res.get('nsfw')
            row.style_cluster = clusters.get(p)
            done += 1
            if done % 200 == 0:
                db.session.commit()
        db.session.commit()
        sizes = {}
        for cid in clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        multi = sum(1 for n in sizes.values() if n >= 2)
        ok = [r for r in results.values() if r.get('state') == 'ok']
        # Name any head that produced nothing, so a degraded pass says so out loud
        # (graceful degradation must be visible, never a silent gap).
        missing = []
        if ok and not any('aesthetic' in r for r in ok):
            missing.append('aesthetic')
        if ok and not any('nsfw' in r for r in ok):
            missing.append('NSFW')
        detail = (f'done — scored {len(ok)} image(s), '
                  f'{multi} style group(s) of 2+')
        if missing:
            detail += f' ({" + ".join(missing)} head unavailable)'
        bank_jobs.progress(job, detail=detail)
    return run


# --- watermark pass (reuses the dataset Qwen3-VL overlaid-mark detector) -----
def start_watermark(app, user_id, bank_id, rescan=False):
    """Launch the overlaid-watermark scan over the bank's non-rejected images,
    reusing the SAME Qwen3-VL detector the datasets use. Needs the vision model
    pulled; serialized against training/vision (503 when the GPU is held)."""
    from ..capabilities import probe_ollama_model
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not probe_ollama_model().get('ok'):
        raise RuntimeError('the vision model is not available '
                           '(Settings ▸ Captioning & quality)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    q = BankImage.query.filter_by(bank_id=bank_id).filter(BankImage.status != 'reject')
    if not rescan:
        q = q.filter(BankImage.watermark_state.is_(None))
    return bank_jobs.start(app, bank_id, 'watermark',
                           _watermark_job(bank_id, rescan), total=q.count())


def _watermark_job(bank_id, rescan):
    def run(job):
        from .face_dataset_service import WATERMARK_BBOX_PROMPT, _parse_watermark_bbox
        from .vision_ollama import describe_image_ollama, unload_vision_model
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        q = (BankImage.query.filter_by(bank_id=bank_id)
             .filter(BankImage.status != 'reject'))
        if not rescan:
            q = q.filter(BankImage.watermark_state.is_(None))
        rows = q.order_by(BankImage.id.asc()).all()
        bank_jobs.progress(job, done=0, total=len(rows), detail='watermark scan')
        if not rows:
            return
        detected = clean = errors = checked = 0
        with gpu_exclusive_vision_window(flag_ttl=1800):
            try:
                for i, row in enumerate(rows, 1):
                    if bank_jobs.cancelled(job):
                        break
                    path = abs_image_path(bank, row)
                    if not path or not os.path.isfile(path):
                        bank_jobs.bump(job)
                        continue
                    try:
                        with open(path, 'rb') as fh:
                            raw = describe_image_ollama(
                                fh.read(), WATERMARK_BBOX_PROMPT, num_predict=400,
                                prefer_json=True, fmt='json', keep_alive='5m')
                    except Exception:  # noqa: BLE001 — one bad file never sinks the pass
                        row.watermark_state = 'error'
                        errors += 1
                        bank_jobs.bump(job)
                        continue
                    # Empty output = Ollama unreachable, NOT "clean": leave the
                    # state untouched so a retry can finish it (same reasoning as
                    # the dataset detector), never falsely mark everything clean.
                    if not (raw or '').strip():
                        bank_jobs.bump(job)
                        continue
                    if _parse_watermark_bbox(raw):
                        row.watermark_state = 'detected'
                        detected += 1
                    else:
                        row.watermark_state = 'none'
                        clean += 1
                    checked += 1
                    if checked % 25 == 0:
                        db.session.commit()
                    bank_jobs.bump(job)
            finally:
                db.session.commit()
                unload_vision_model()  # hand the VRAM back to ComfyUI
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {detected} with a watermark '
                                           f'so far')
            return
        detail = f'done — {detected} with a watermark, {clean} clean'
        if errors:
            detail += f', {errors} unreadable'
        bank_jobs.progress(job, detail=detail)
    return run


# --- caption pass (reuses the dataset caption engines) ----------------------
def start_caption(app, user_id, bank_id, ids=None, force=False):
    """Launch the caption pass over a selection (``ids``) or, when empty, every
    non-rejected readable image. Reuses the dataset caption engines (JoyCaption /
    Ollama per Settings) through a dataset-free descriptive brick; the captions
    double as the bank's search text and ride along on promotion. Serialized
    against training/vision like the score/watermark passes (503 when the GPU is
    held). BankJobBusy when a job is already live, ValueError on a bad bank/config."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    backend = (cfg.get('captioning.backend') or 'auto').lower()
    if backend == 'none':
        raise ValueError('no captioning backend configured (Settings ▸ Captioning & quality)')
    reason = _gpu_busy_reason()
    if reason:
        raise RuntimeError(reason)
    ids = [int(i) for i in ids] if ids else None
    q = BankImage.query.filter_by(bank_id=bank_id).filter(BankImage.status != 'reject')
    if ids is not None:
        q = q.filter(BankImage.id.in_(ids[:_SQL_IN_CHUNK]))
    if not force:
        q = q.filter(or_(BankImage.caption.is_(None), BankImage.caption == ''))
    total = q.count()
    return bank_jobs.start(app, bank_id, 'caption',
                           _caption_job(bank_id, ids, force), total=total)


def _caption_job(bank_id, ids, force):
    def run(job):
        from .face_dataset_service import caption_paths
        from ..gpu_window import gpu_exclusive_vision_window
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        q = BankImage.query.filter_by(bank_id=bank_id).filter(BankImage.status != 'reject')
        if ids is not None:
            rows = []
            for i0 in range(0, len(ids), _SQL_IN_CHUNK):
                rows.extend(q.filter(BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all())
            rows.sort(key=lambda r: r.id)
        else:
            rows = q.order_by(BankImage.id.asc()).all()
        if not force:
            rows = [r for r in rows if not (r.caption or '').strip()]
        by_path = {}
        for r in rows:
            p = abs_image_path(bank, r)
            if p and os.path.isfile(p):
                by_path[p] = r.id
        paths = list(by_path)
        bank_jobs.progress(job, done=0, total=len(paths), detail='captioning')
        if not paths:
            return
        captioned = 0

        def _on_caption(path, caption):
            nonlocal captioned
            row = db.session.get(BankImage, by_path.get(path))
            if row is not None:
                row.caption = caption
                db.session.commit()
                captioned += 1

        # GPU-exclusive for the whole pass, exactly like the score/watermark passes:
        # frees ComfyUI VRAM and blocks a training start for the duration.
        with gpu_exclusive_vision_window(flag_ttl=1800):
            caption_paths(
                paths,
                should_cancel=lambda: bank_jobs.cancelled(job),
                on_caption=_on_caption,
                progress=lambda d, t: bank_jobs.progress(job, done=d, total=t))
        if bank_jobs.cancelled(job):
            bank_jobs.progress(job, detail=f'cancelled — {captioned} captioned so far')
            return
        bank_jobs.progress(job, detail=f'done — {captioned} captioned')
    return run


# --- "Launch all" pipeline --------------------------------------------------
# The overnight funnel: the user configures it once, hits Launch all, and comes
# back to a triaged, optionally pre-captioned bank. It chains the EXISTING passes
# in the order Jeremy validated. Each pass already filters status != 'reject', so
# running auto-reject BEFORE the heavy passes means score/watermark/person only
# ever touch the SURVIVORS — the costly work never pays for images we just
# dropped (the deliberate cost/quality trade-off: duplicate "keep best" therefore
# ranks on sharpness/size, not the aesthetic score that isn't computed yet).
PIPELINE_STEPS = ('scan', 'auto_reject', 'score', 'watermark', 'faces', 'caption')
# Auto-reject inside the pipeline runs right after the quality scan, so it can
# only act on the CPU-scan flags (and duplicates). The score-derived flags
# (low_aesthetic/nsfw/watermark) have no data yet at that point.
PIPELINE_REJECT_FLAGS = _QUALITY_FLAGS


def _sanitize_pipeline_steps(steps) -> list:
    """Keep only known steps, in the canonical pipeline order (the client can't
    reorder or invent a pass)."""
    want = set(steps or [])
    return [s for s in PIPELINE_STEPS if s in want]


def _score_prereq() -> str | None:
    from ..capabilities import probe_bank_scoring
    if not probe_bank_scoring().get('ok'):
        return 'bank scoring extra not installed (Setup ▸ Quality tools)'
    return None


def _watermark_prereq() -> str | None:
    from ..capabilities import probe_ollama_model
    if not probe_ollama_model().get('ok'):
        return 'vision model not available (Settings ▸ Captioning & quality)'
    return None


def _faces_prereq() -> str | None:
    from .face_similarity import is_available
    if not is_available():
        return 'face scoring extra not installed (Setup ▸ Quality tools)'
    return None


def _caption_prereq() -> str | None:
    if (cfg.get('captioning.backend') or 'auto').lower() == 'none':
        return 'no captioning backend configured (Settings ▸ Captioning & quality)'
    return None


def start_pipeline(app, user_id, bank_id, steps=None, reject_flags=None,
                   resolve_dups=False):
    """Launch the chained triage pipeline. ``steps`` selects which passes run
    (canonical order enforced); ``reject_flags`` + ``resolve_dups`` configure the
    auto-reject step. One background job like every other pass — BankJobBusy when
    one is already live, ValueError on a bad bank / empty step list."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    steps = _sanitize_pipeline_steps(steps)
    if not steps:
        raise ValueError('no pipeline steps selected')
    reject_flags = [f for f in (reject_flags or []) if f in PIPELINE_REJECT_FLAGS]
    return bank_jobs.start(
        app, bank_id, 'pipeline',
        _pipeline_job(user_id, bank_id, steps, reject_flags, bool(resolve_dups)),
        total=0)


def _bank_counts(bank_id) -> dict:
    """Live headline counts used for the per-step tallies and the final report."""
    base = BankImage.query.filter_by(bank_id=bank_id)
    dup_groups = (db.session.query(BankImage.dup_group)
                  .filter(BankImage.bank_id == bank_id,
                          BankImage.dup_group.isnot(None))
                  .distinct().count())
    style_groups = (db.session.query(BankImage.style_cluster)
                    .filter(BankImage.bank_id == bank_id,
                            BankImage.style_cluster.isnot(None))
                    .distinct().count())
    person_groups = (db.session.query(BankImage.face_cluster)
                     .filter(BankImage.bank_id == bank_id,
                             BankImage.face_cluster.isnot(None))
                     .distinct().count())
    return {
        'total': base.count(),
        'scanned': base.filter(BankImage.quality_state.isnot(None)).count(),
        'reject': base.filter_by(status='reject').count(),
        'scored': base.filter(or_(BankImage.aesthetic_score.isnot(None),
                                  BankImage.nsfw_score.isnot(None))).count(),
        'watermark_detected': base.filter(BankImage.watermark_state == 'detected').count(),
        'captioned': base.filter(and_(BankImage.caption.isnot(None),
                                       BankImage.caption != '')).count(),
        'dup_groups': dup_groups,
        'style_groups': style_groups,
        'person_groups': person_groups,
    }


def _pipeline_job(user_id, bank_id, steps, reject_flags, resolve_dups):
    def run(job):
        import json as _json
        import time as _time
        from ..gpu_window import GpuBusyError
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        results = []
        pipe = {'steps': list(steps), 'total_steps': len(steps),
                'index': 0, 'current': steps[0], 'results': results}

        def _sync(current=None, index=None):
            if index is not None:
                pipe['index'] = index
            if current is not None:
                pipe['current'] = current
            pipe['results'] = list(results)
            bank_jobs.set_pipeline(job, pipe)

        _sync()
        # Each entry closes with a status: 'done' | 'skipped' (reason) | 'error'.
        for i, step in enumerate(steps):
            if bank_jobs.cancelled(job):
                break
            _sync(current=step, index=i)
            bank_jobs.progress(job, done=0, total=0,
                               detail=f'step {i + 1}/{len(steps)}: {step}')
            entry = {'step': step, 'status': 'done', 'reason': None,
                     'detail': None, 'counts': {}}
            try:
                _run_pipeline_step(job, user_id, bank_id, step,
                                   reject_flags, resolve_dups, entry)
            except GpuBusyError as e:
                # A vision/training job grabbed the GPU mid-pipeline — skip this
                # pass and keep going (never wake the user for a transient clash).
                entry['status'] = 'skipped'
                entry['reason'] = f'GPU busy — {e}'
            except Exception as e:  # noqa: BLE001 — one bad pass never sinks the rest
                entry['status'] = 'error'
                entry['reason'] = f'{type(e).__name__}: {e}'
                db.session.rollback()
            # A step that executed stays 'done' even if a cancel landed at its
            # tail (its inner run already returned early); only steps we never
            # reach are recorded as cancelled, below.
            results.append(entry)
            _sync()

        cancelled = bank_jobs.cancelled(job)
        # Any step never reached (cancel, or a hard earlier break) is recorded so
        # the morning-after report has a row for every requested pass.
        reached = {e['step'] for e in results}
        for step in steps:
            if step not in reached:
                results.append({'step': step, 'status': 'cancelled' if cancelled
                                else 'skipped',
                                'reason': 'cancelled before it ran' if cancelled
                                else 'not reached', 'detail': None, 'counts': {}})
        _sync()

        report = {
            'started_at': job.get('started_at'),
            'finished_at': _time.time(),
            'cancelled': cancelled,
            'requested_steps': list(steps),
            'reject_flags': list(reject_flags),
            'resolve_dups': resolve_dups,
            'steps': results,
            'counts': _bank_counts(bank_id),
        }
        bank = db.session.get(ImageBank, bank_id)
        if bank is not None:
            bank.pipeline_report = _json.dumps(report)
            db.session.commit()
        done_n = sum(1 for e in results if e['status'] == 'done')
        skipped_n = sum(1 for e in results if e['status'] in ('skipped', 'cancelled'))
        err_n = sum(1 for e in results if e['status'] == 'error')
        tail = f'done — {done_n}/{len(steps)} steps ran'
        if skipped_n:
            tail += f', {skipped_n} skipped'
        if err_n:
            tail += f', {err_n} errored'
        if cancelled:
            tail = f'cancelled — {done_n}/{len(steps)} steps ran'
        bank_jobs.progress(job, detail=tail)
    return run


def _run_pipeline_step(job, user_id, bank_id, step, reject_flags, resolve_dups, entry):
    """Run ONE pipeline pass into ``entry``, reusing the standalone pass work.
    Prerequisite missing → entry marked 'skipped' with a reason, pipeline
    continues. Reuses each pass's inner ``run(job)`` so progress, cancellation
    and the GPU-exclusive window behave exactly as the standalone buttons."""
    if step == 'scan':
        _scan_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'scanned': c['scanned'], 'dup_groups': c['dup_groups']}
        entry['detail'] = (job.get('detail')
                           or f"scanned {c['scanned']}, {c['dup_groups']} duplicate group(s)")
        return
    if step == 'auto_reject':
        rejected = apply_flags(user_id, bank_id, reject_flags) if reject_flags else {}
        dup_rejected = 0
        if resolve_dups:
            dup_rejected = resolve_dups_keep_best(user_id, bank_id)
        n = sum(rejected.values()) + dup_rejected
        entry['counts'] = {'rejected': n, 'by_flag': rejected,
                           'duplicates': dup_rejected}
        parts = [f'{v} {k}' for k, v in rejected.items() if v]
        if dup_rejected:
            parts.append(f'{dup_rejected} duplicate')
        entry['detail'] = (f"rejected {n} image(s)"
                           + (f" ({', '.join(parts)})" if parts else '')
                           + ' — manual ✓/✕ untouched')
        return
    if step == 'score':
        reason = _score_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _score_job(bank_id)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'scored': c['scored'], 'style_groups': c['style_groups']}
        entry['detail'] = job.get('detail') or f"scored {c['scored']} image(s)"
        return
    if step == 'watermark':
        reason = _watermark_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _watermark_job(bank_id, rescan=False)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'watermarks': c['watermark_detected']}
        entry['detail'] = job.get('detail') or f"{c['watermark_detected']} with a watermark"
        return
    if step == 'faces':
        reason = _faces_prereq()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        _faces_job(bank_id)(job)
        c = _bank_counts(bank_id)
        entry['counts'] = {'person_groups': c['person_groups']}
        entry['detail'] = job.get('detail') or f"{c['person_groups']} person cluster(s)"
        return
    if step == 'caption':
        reason = _caption_prereq() or _gpu_busy_reason()
        if reason:
            entry['status'], entry['reason'] = 'skipped', reason
            return
        before = _bank_counts(bank_id)['captioned']
        _caption_job(bank_id, None, False)(job)
        after = _bank_counts(bank_id)['captioned']
        entry['counts'] = {'captioned': max(0, after - before), 'total_captioned': after}
        entry['detail'] = job.get('detail') or f"{after} captioned"
        return
    entry['status'], entry['reason'] = 'skipped', 'unknown step'


def resolve_dups_keep_best(user_id, bank_id) -> int:
    """Auto-resolve every unresolved duplicate group keeping the best member,
    for the pipeline's auto-reject step. Returns the number REJECTED."""
    out = resolve_dups(user_id, bank_id, strategy='best')
    return out.get('rejected', 0)


# --- subfolders (scoping facet) ---------------------------------------------
def subfolders_payload(user_id, bank_id) -> dict | None:
    """Top-level subfolders of the bank's source folder with image counts, for
    the scoping picker. Computed once on open (not polled) — a Telegram export
    nests one folder per chat/date. '' = files at the bank root."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    from collections import Counter
    counts: Counter = Counter()
    for (rel,) in (db.session.query(BankImage.relpath)
                   .filter(BankImage.bank_id == bank_id).all()):
        counts[_subfolder_of(rel)] += 1
    items = [{'name': name, 'count': n}
             for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {'subfolders': items, 'total': sum(counts.values())}


# --- promotion --------------------------------------------------------------
def start_promote(app, user_id, bank_id, ids, dataset_id):
    """Copy a selection into a dataset through the normal import path
    (normalize + perceptual dedup vs the dataset). ``ids`` empty = every KEPT,
    not-yet-promoted image. Background job (a big promotion decodes hundreds
    of files)."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    ds = FaceDataset.query.filter_by(id=dataset_id, user_id=user_id).first()
    if not ds:
        raise ValueError('dataset not found')
    if ids:
        ids = [int(i) for i in ids]
    else:
        ids = [r.id for r in
               BankImage.query.filter_by(bank_id=bank_id, status='keep')
               .filter(BankImage.promoted_dataset_id.is_(None))
               .order_by(BankImage.id.asc()).all()]
    if not ids:
        raise ValueError('nothing to promote — keep some images first')
    return bank_jobs.start(app, bank_id, 'promote',
                           _promote_job(user_id, bank_id, ids, dataset_id),
                           total=len(ids))


def _promote_job(user_id, bank_id, ids, dataset_id):
    def run(job):
        bank = db.session.get(ImageBank, bank_id)
        if not bank:
            return
        rows = []
        for i0 in range(0, len(ids), _SQL_IN_CHUNK):
            rows.extend(BankImage.query.filter(
                BankImage.bank_id == bank_id,
                BankImage.id.in_(ids[i0:i0 + _SQL_IN_CHUNK])).all())
        rows.sort(key=lambda r: r.id)
        bank_jobs.progress(job, done=0, total=len(rows), detail='promoting')
        stats: dict = {}
        imported = failed = 0
        for c0 in range(0, len(rows), _PROMOTE_CHUNK):
            if bank_jobs.cancelled(job):
                break
            chunk = rows[c0:c0 + _PROMOTE_CHUNK]
            blobs, chunk_rows, caps = [], [], []
            for r in chunk:
                p = abs_image_path(bank, r)
                try:
                    with open(p, 'rb') as fh:
                        blobs.append(fh.read())
                    chunk_rows.append(r)
                    # Carry the bank caption onto the dataset image (parallel to blobs),
                    # so a captioned selection lands already captioned.
                    caps.append(r.caption)
                except (OSError, TypeError):
                    failed += 1
            if blobs:
                new_ids, bad = import_images(user_id, dataset_id, blobs,
                                             dedupe=True, stats=stats, captions=caps)
                imported += len(new_ids)
                failed += bad
                # 'Promoted' = handed to the dataset — a dedupe skip means the
                # dataset already holds an equivalent, which counts as handled.
                for r in chunk_rows:
                    r.promoted_dataset_id = dataset_id
                db.session.commit()
            bank_jobs.bump(job, len(chunk))
        dups = stats.get('duplicates', 0)
        small = stats.get('small', 0)
        detail = f'done — {imported} imported'
        if dups:
            detail += f', {dups} already in the dataset'
        if failed:
            detail += f', {failed} failed'
        if small:
            detail += f', {small} under the recommended size'
        bank_jobs.progress(job, detail=detail)
    return run
