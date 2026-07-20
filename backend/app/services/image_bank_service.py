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
from sqlalchemy import and_, case, func, or_

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
        'semantic_dup_group': row.semantic_dup_group,
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

# Resolution tiers for the Bank grid — bucketed on MEGAPIXELS (width×height, the
# same rank as the resolution sort) so a mixed dump can be skimmed and mass-acted
# one tier at a time. Each entry is (stable_id, lo, hi) in raw pixels, a HALF-OPEN
# [lo, hi) range (lower-inclusive, upper-exclusive); hi=None means "no ceiling".
# So a 1000×1000 (1.00 MP) and a 1024×1024 (1.05 MP) both land in 'res_1_2', and a
# 2000×2000 (4.00 MP) lands in 'res_gt_4'. The 0.25 MP floor (not 0.30) is chosen
# so a 512×512 (0.26 MP) — a legit small training crop — sits in '0.25–1 MP', while
# only true junk (Telegram thumbnails ~0.1–0.2 MP, ≤448²) falls in '< 0.25 MP'.
# Ids are user-facing filter keys — never rename without an alias.
_RES_BUCKETS = (
    ('res_lt_025', 0, 250_000),
    ('res_025_1', 250_000, 1_000_000),
    ('res_1_2', 1_000_000, 2_000_000),
    ('res_2_4', 2_000_000, 4_000_000),
    ('res_gt_4', 4_000_000, None),
)
_RES_BOUNDS = {bid: (lo, hi) for bid, lo, hi in _RES_BUCKETS}


def _subfolder_of(relpath: str) -> str:
    """Top-level subfolder of a bank-relative path ('' for a root-level file) —
    the natural scoping axis for a Telegram export (one folder per chat/date)."""
    parts = (relpath or '').replace('\\', '/').split('/', 1)
    return parts[0] if len(parts) > 1 else ''


def _unresolved_dup_groups_q(bank_id, col=BankImage.dup_group):
    """Groups still holding ≥2 NON-rejected members — i.e. still to resolve. ``col``
    selects the stage: dup_group (exact/resized) or semantic_dup_group (crops)."""
    return (db.session.query(col)
            .filter(BankImage.bank_id == bank_id,
                    col.isnot(None),
                    BankImage.status != 'reject')
            .group_by(col)
            .having(func.count(BankImage.id) >= 2))


def _res_bucket_case():
    """A single SQL CASE mapping each scanned row to its resolution-tier id, used
    both to COUNT per tier (one GROUP BY) and — via _RES_BOUNDS — to FILTER a page
    to one tier. Rows with a NULL dimension never reach this (callers pre-filter
    width/height NOT NULL), so no NULL-misfile into the top tier."""
    area = BankImage.width * BankImage.height
    whens = [(area < hi, bid) for bid, _lo, hi in _RES_BUCKETS if hi is not None]
    return case(*whens, else_=_RES_BUCKETS[-1][0])


def _res_bucket_counts(bank_id) -> dict:
    """Per-tier image counts for the resolution chips (bank-wide, like the flag
    totals) in ONE GROUP BY. Unscanned rows (width/height NULL) are excluded, so a
    tier that no image falls into simply reports 0. Every tier id is present."""
    bucket = _res_bucket_case()
    rows = (db.session.query(bucket, func.count(BankImage.id))
            .filter(BankImage.bank_id == bank_id,
                    BankImage.width.isnot(None),
                    BankImage.height.isnot(None))
            .group_by(bucket).all())
    got = {bid: n for bid, n in rows}
    return {bid: int(got.get(bid, 0)) for bid, _lo, _hi in _RES_BUCKETS}


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
    res_buckets = _res_bucket_counts(bank_id)
    dup_rows = (db.session.query(BankImage.dup_group, func.count(BankImage.id))
                .filter(BankImage.bank_id == bank_id,
                        BankImage.dup_group.isnot(None))
                .group_by(BankImage.dup_group).all())
    dup = {'groups': len(dup_rows),
           'images': sum(n for _g, n in dup_rows),
           'unresolved': _unresolved_dup_groups_q(bank_id).count()}
    # Stage-2 semantic near-duplicate groups (crops/variants), same summary shape.
    sem_rows = (db.session.query(BankImage.semantic_dup_group, func.count(BankImage.id))
                .filter(BankImage.bank_id == bank_id,
                        BankImage.semantic_dup_group.isnot(None))
                .group_by(BankImage.semantic_dup_group).all())
    semantic_dup = {
        'groups': len(sem_rows),
        'images': sum(n for _g, n in sem_rows),
        'unresolved': _unresolved_dup_groups_q(
            bank_id, BankImage.semantic_dup_group).count()}
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
        'counts': counts, 'flags': flags, 'res_buckets': res_buckets, 'dup': dup,
        'semantic_dup': semantic_dup,
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
                semantic_group=None, sort=None, res_bucket=None,
                offset=0, limit=200) -> dict | None:
    """One PAGE of the bank grid (a 9 000-image bank must never ship whole).
    Filters compose: status ∩ flag ∩ cluster ∩ dup-group ∩ style ∩ subfolder ∩ search.
    ``search`` is a plain full-text term matched (case-insensitive LIKE) against the
    caption AND the relpath — so captions double as searchable tags for a big dump
    ("red dress"), combinable with every other filter. Flag filters sort by the
    relevant score (worst first) so the review reads top-down.
    ``sort`` ('res_desc'/'res_asc') overrides the order by image resolution
    (megapixels = width×height, so 900×900 outranks 1200×300); unscanned rows
    (width/height NULL) always sink to the end. It composes with every filter.
    ``res_bucket`` (a _RES_BUCKETS id) narrows to one resolution tier — a
    half-open [lo, hi) megapixel band — and composes with every filter AND the
    sort (the tier + Resolution↑/↓ combo is the mixed-dump cleanup flow)."""
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
    elif flag == 'semantic_dups':
        q = q.filter(BankImage.semantic_dup_group.isnot(None))
        order = (BankImage.semantic_dup_group.asc(), BankImage.id.asc())
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
    if semantic_group is not None:
        q = q.filter(BankImage.semantic_dup_group == int(semantic_group))
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
    if res_bucket in _RES_BOUNDS:
        # One resolution tier: [lo, hi) on megapixels (width×height). The NOT-NULL
        # guards drop unscanned rows (a NULL product would satisfy neither bound
        # cleanly), so a tier never leaks unscanned images. Composes with the sort.
        lo, hi = _RES_BOUNDS[res_bucket]
        area = BankImage.width * BankImage.height
        q = q.filter(BankImage.width.isnot(None), BankImage.height.isnot(None))
        if lo:
            q = q.filter(area >= lo)
        if hi is not None:
            q = q.filter(area < hi)
    if sort in ('res_desc', 'res_asc'):
        # Explicit resolution sort wins over the flag worst-first order. Rank by
        # megapixels (width×height), tie-break on id for a stable page boundary.
        # Unscanned rows (either dimension NULL → NULL product) sink to the end
        # in BOTH directions: order by "is NULL" first (0 before 1 in SQLite).
        area = BankImage.width * BankImage.height
        area_dir = area.desc() if sort == 'res_desc' else area.asc()
        order = (area.is_(None).asc(), area_dir, BankImage.id.asc())
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


# --- semantic near-duplicate groups (stage 2 — crops / re-compressed variants) --
def _load_score_embeddings(bank: ImageBank) -> dict:
    """{abs_path: emb (np.float32, L2-normed)} from the ✨ Score pass cache, for the
    scored 'ok' images whose file still matches what was scored. Empty when the pass
    never ran (no cache) — the caller then surfaces the "run Score first" hint. A
    STALE entry (a same-path edit since scoring, detected via the cached size+mtime
    signature) is dropped, so a semantic group is never built on an outdated
    embedding. Reads the .npz directly (numpy is in the Flask venv); torch/open_clip
    are NOT needed here — stage 2 costs no new GPU work, it reuses Score's output."""
    import numpy as np
    path = _score_cache_path(bank.id)
    if not path.is_file():
        return {}
    try:
        with np.load(str(path), allow_pickle=False) as z:
            paths = [str(p) for p in z['paths']]
            states = [str(s) for s in z['states']]
            embs = z['embs']
            sigs = ([str(s) for s in z['sigs']] if 'sigs' in z.files
                    else [''] * len(paths))
    except Exception as e:  # noqa: BLE001 — a corrupt cache = "no embeddings", never fatal
        logger.warning('bank %s score cache unreadable: %s', bank.id, e)
        return {}
    out = {}
    for i, p in enumerate(paths):
        if states[i] != 'ok':
            continue
        emb = embs[i]
        if float(np.abs(emb).sum()) <= 0:       # zero sentinel = errored image
            continue
        sig = sigs[i]
        if sig:                                 # drop a since-edited file
            try:
                st = os.stat(p)
                if f'{st.st_size}:{st.st_mtime_ns}' != sig:
                    continue
            except OSError:
                continue
        out[p] = np.asarray(emb, dtype='float32')
    return out


def rebuild_semantic_dup_groups(bank_id, threshold=None) -> int | None:
    """Stage-2 near-duplicate grouping over the CLIP embeddings the ✨ Score pass
    cached — catches crops and re-compressed variants of the SAME shot that the
    dHash (stage 1) misses. Returns the group count (groups of ≥2), or None when NO
    embeddings are available (Score hasn't run) so the caller shows the "run Score
    first" hint instead of a silent empty result.

    Cost: a semantic near-dup (cosine ≥ threshold) is necessarily inside one style
    union-find component (that clustering uses style_threshold ≤ threshold), so we
    BLOCK by style_cluster and only compare within a block — Σ block² dot-products,
    not the full n². A config with style_threshold > threshold would break that
    guarantee, so we fall back to a single global block then. Re-running at another
    threshold is CPU-only and near-instant: it re-reads the cached embeddings — no
    GPU, no re-scan."""
    import numpy as np
    bank = db.session.get(ImageBank, bank_id)
    if not bank:
        return None
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        return None
    th = thresholds()
    t = float(threshold if threshold is not None else th['semantic_dup_threshold'])
    block_by_style = th['style_threshold'] <= t
    rows = (BankImage.query.filter_by(bank_id=bank_id)
            .order_by(BankImage.id.asc()).all())
    items = []      # (image_id, block_key, emb)
    for r in rows:
        p = abs_image_path(bank, r)
        emb = emb_by_path.get(p) if p else None
        if emb is None:
            continue
        block = (r.style_cluster if r.style_cluster is not None else -1) \
            if block_by_style else 0
        items.append((r.id, block, emb))
    # A re-run fully recomputes — clear every semantic group first.
    BankImage.query.filter_by(bank_id=bank_id).update(
        {'semantic_dup_group': None}, synchronize_session=False)
    if not items:
        db.session.commit()
        return 0
    blocks: dict = {}
    for idx, (_id, block, _emb) in enumerate(items):
        blocks.setdefault(block, []).append(idx)
    parent = list(range(len(items)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    chunk = 512
    for members in blocks.values():
        if len(members) < 2:
            continue
        E = np.stack([items[i][2] for i in members]).astype('float32')
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        m = len(members)
        for i0 in range(0, m, chunk):
            sims = E[i0:i0 + chunk] @ E.T
            for a, b in np.argwhere(sims >= t):
                a += i0
                if a < b:                       # skip the diagonal + mirror pairs
                    union(members[int(a)], members[int(b)])
    comps: dict = {}
    for i in range(len(items)):
        comps.setdefault(find(i), []).append(i)
    groups = sorted((m for m in comps.values() if len(m) >= 2),
                    key=lambda m: (-len(m), items[m[0]][0]))
    for gid, members in enumerate(groups, start=1):
        member_ids = [items[i][0] for i in members]
        for i0 in range(0, len(member_ids), _SQL_IN_CHUNK):
            BankImage.query.filter(
                BankImage.id.in_(member_ids[i0:i0 + _SQL_IN_CHUNK])).update(
                {'semantic_dup_group': gid}, synchronize_session=False)
    db.session.commit()
    return len(groups)


def start_semantic_dedup(app, user_id, bank_id, threshold=None):
    """Launch the stage-2 semantic near-duplicate pass (CPU, reuses the ✨ Score
    embeddings — no GPU). ValueError (→400) when Score hasn't produced any usable
    embedding yet, so the UI shows the clear "run Score first" hint rather than a
    job that quietly does nothing."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if not _load_score_embeddings(bank):
        raise ValueError('run ✨ Score first — semantic near-duplicates reuse its '
                         'embeddings')
    return bank_jobs.start(app, bank_id, 'semantic_dedup',
                           _semantic_dedup_job(bank_id, threshold), total=0)


def _semantic_dedup_job(bank_id, threshold):
    def run(job):
        bank_jobs.progress(job, done=0, total=0, detail='finding crops & variants')
        n = rebuild_semantic_dup_groups(bank_id, threshold)
        if n is None:
            bank_jobs.progress(job, detail='no embeddings — run ✨ Score first')
            return
        bank_jobs.progress(job, detail=f'done — {n} semantic near-duplicate group(s)')
    return run


def dup_groups_payload(user_id, bank_id, offset=0, limit=50,
                       col=BankImage.dup_group) -> dict | None:
    """Unresolved groups (≥2 non-rejected members) with their full membership,
    for the resolution panel. ``col`` picks the stage: dup_group (exact/resized) or
    semantic_dup_group (crops/variants)."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        return None
    th = thresholds()
    gids = [g for (g,) in _unresolved_dup_groups_q(bank_id, col)
            .order_by(col.asc()).all()]
    total = len(gids)
    page = gids[max(0, int(offset)):max(0, int(offset)) + max(1, min(200, int(limit)))]
    groups = []
    for gid in page:
        rows = (BankImage.query.filter(BankImage.bank_id == bank_id, col == gid)
                .order_by(BankImage.id.asc()).all())
        groups.append({'group': gid,
                       'best_id': _best_of(rows).id if rows else None,
                       'images': [_image_dict(r, th) for r in rows]})
    return {'groups': groups, 'total': total, 'offset': max(0, int(offset))}


def semantic_dup_groups_payload(user_id, bank_id, offset=0, limit=50) -> dict | None:
    """dup_groups_payload for stage 2 (semantic_dup_group)."""
    return dup_groups_payload(user_id, bank_id, offset=offset, limit=limit,
                              col=BankImage.semantic_dup_group)


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


def resolve_dups(user_id, bank_id, strategy='best', group=None, keep_ids=None,
                 col=BankImage.dup_group, attr='dup_group', reason='duplicate'):
    """Resolve duplicate groups: keep one member, REJECT the others (a status,
    never a file deletion, so it's reversible). strategy 'best'|'first' applies to
    one group or, when ``group`` is None, to every unresolved group at once;
    explicit ``keep_ids`` (manual pick) applies to their own groups. Only
    non-rejected members are touched; a member the user already KEPT stays kept
    (never flipped by a bulk resolve). ``col``/``attr``/``reason`` pick the stage:
    dup_group (exact/'duplicate') or semantic_dup_group (crops/'semantic_dup').
    Returns {'resolved': groups, 'rejected': images}."""
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    keep_by_group = {}
    if keep_ids:
        rows = BankImage.query.filter(BankImage.bank_id == bank_id,
                                      BankImage.id.in_(list(keep_ids)[:_SQL_IN_CHUNK])).all()
        for r in rows:
            g = getattr(r, attr)
            if g:
                keep_by_group.setdefault(g, set()).add(r.id)
        gids = list(keep_by_group)
    elif group is not None:
        gids = [int(group)]
    else:
        gids = [g for (g,) in _unresolved_dup_groups_q(bank_id, col).all()]
    resolved = rejected = 0
    for gid in gids:
        rows = (BankImage.query.filter(BankImage.bank_id == bank_id, col == gid)
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
            r.status, r.reject_reason = 'reject', reason
            rejected += 1
            changed = True
        if changed or len(rows) >= 2:
            resolved += 1
    db.session.commit()
    return {'resolved': resolved, 'rejected': rejected}


def resolve_semantic_dups(user_id, bank_id, strategy='best', group=None,
                          keep_ids=None):
    """resolve_dups for stage 2 (semantic_dup_group, reject reason
    'semantic_dup')."""
    return resolve_dups(user_id, bank_id, strategy=strategy, group=group,
                        keep_ids=keep_ids, col=BankImage.semantic_dup_group,
                        attr='semantic_dup_group', reason='semantic_dup')


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


# --- curation selectors (diversity · reference similarity) ------------------
# Both reuse the CLIP embeddings the ✨ Score pass already cached — no GPU, no
# re-scan (same contract as the semantic-dedup stage). They only ever build a
# SELECTION (a set of image ids the UI checks); the user reviews it before any
# Keep / Reject / Promote — nothing is mutated or deleted here.
_CURATION_MAX_N = 2000       # a curated LoRA set is 20–200 images; this is generous


def _pool_query(bank_id, th, *, status=None, flag=None, cluster=None,
                style=None, subfolder=None, search=None):
    """The candidate-pool query for the curation selectors — the SAME filter
    composition as list_images (status ∩ flag ∩ cluster ∩ style ∩ subfolder ∩
    search), minus the ordering/pagination, so "give me 60 diverse images" is
    composable with whatever the grid is currently showing.

    Kept as its own function (a small, deliberate mirror of the list_images WHERE
    clauses) rather than a shared refactor: three curation-related branches touch
    this file in parallel, so an additive helper rebases clean where an edit to
    the list_images hot path would collide. When NO status is chosen the reject
    pile is excluded — you curate from what you might keep, never from the bin."""
    q = BankImage.query.filter_by(bank_id=bank_id)
    if status in ('pending', 'keep', 'reject'):
        q = q.filter(BankImage.status == status)
    else:
        q = q.filter(BankImage.status != 'reject')
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
    elif flag == 'semantic_dups':
        q = q.filter(BankImage.semantic_dup_group.isnot(None))
    elif flag == 'no_face':
        q = q.filter(BankImage.face_state == 'no_face')
    elif flag in _QUALITY_FLAGS + _SCORE_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
    if cluster is not None:
        q = q.filter(BankImage.face_cluster == int(cluster))
    if style is not None:
        q = q.filter(BankImage.style_cluster == int(style))
    if subfolder is not None:
        if subfolder == '':
            q = q.filter(~BankImage.relpath.contains(os.sep))
        else:
            q = q.filter(BankImage.relpath.startswith(subfolder + os.sep))
    term = (search or '').strip()
    if term:
        esc = term.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        like = f'%{esc}%'
        q = q.filter(or_(BankImage.caption.ilike(like, escape='\\'),
                         BankImage.relpath.ilike(like, escape='\\')))
    return q


def _pool_embeddings(bank, emb_by_path, filters):
    """(ids, E) for the filtered pool rows that HAVE a cached embedding: ids is a
    list ordered by image id (so every tie-break below is deterministic), E is the
    matching (m×d) float32 matrix, L2-normalised. Empty ids ⇒ E is None."""
    import numpy as np
    rows = (_pool_query(bank.id, thresholds(), **filters)
            .order_by(BankImage.id.asc()).all())
    ids, vecs = [], []
    for r in rows:
        p = abs_image_path(bank, r)
        emb = emb_by_path.get(p) if p else None
        if emb is not None:
            ids.append(r.id)
            vecs.append(emb)
    if not ids:
        return ids, None
    E = np.stack(vecs).astype('float32')
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return ids, E


def select_diverse(user_id, bank_id, n=60, *, filters=None):
    """Farthest-point sampling over the ✨ Score CLIP embeddings: the ``n`` images
    of the (filtered) pool that best COVER the visual space — the antidote to a
    dump of 4 000 near-identical shots. Greedy FPS: seed with the lowest-id row
    (deterministic), then repeatedly add the point whose nearest already-chosen
    neighbour is FARTHEST (max-min cosine distance). O(n·m·d) — one (m×d)·(d,)
    product per pick, ~sub-second even at m=24 000 / n=2 000.

    Returns {'image_ids': [...] (sorted), 'pool': m, 'requested': n}. Raises
    ValueError (→400, "run ✨ Score first") when no embedding exists yet, so the UI
    shows the clear hint instead of an empty, unexplained selection."""
    import numpy as np
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        raise ValueError('run ✨ Score first — diversity sampling reuses its '
                         'embeddings')
    n = max(1, min(int(n), _CURATION_MAX_N))
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    m = len(ids)
    if m <= n:                                   # whole pool already fits
        return {'image_ids': sorted(ids), 'pool': m, 'requested': n}
    # min_dist[i] = cosine distance from row i to the NEAREST chosen row so far.
    chosen = [0]                                 # seed = lowest id (E[0])
    min_dist = 1.0 - E @ E[0]
    min_dist[0] = -1.0                           # never re-pick a chosen row
    for _ in range(n - 1):
        nxt = int(np.argmax(min_dist))           # ties → lowest index = lowest id
        if min_dist[nxt] <= -1.0:                # pool exhausted (all chosen)
            break
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, 1.0 - E @ E[nxt])
        min_dist[nxt] = -1.0
    return {'image_ids': sorted(ids[i] for i in chosen),
            'pool': m, 'requested': n}


def select_similar(user_id, bank_id, ref_id, n=60, min_score=None, *, filters=None):
    """Rank the (filtered) pool by CLIP cosine similarity to a REFERENCE bank image
    (its own cached ✨ Score embedding) — "keep what looks like THIS", to pull one
    person / look out of a mixed dump. Returns the top-``n`` most similar ids, OR
    everything with cosine ≥ ``min_score`` when that is given; the reference itself
    (cosine 1.0) is always included. Reuses the cached embeddings — no GPU.

    Returns {'results': [{id, score}], 'image_ids': [...], 'pool': m, 'ref_id'}.
    Raises ValueError (→400) when Score hasn't run or the reference has no cached
    embedding (e.g. it was rejected before Score, or edited since)."""
    import numpy as np
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    emb_by_path = _load_score_embeddings(bank)
    if not emb_by_path:
        raise ValueError('run ✨ Score first — reference similarity reuses its '
                         'embeddings')
    ref = db.session.get(BankImage, int(ref_id))
    if ref is None or ref.bank_id != bank_id:
        raise ValueError('reference image not found in this bank')
    ref_path = abs_image_path(bank, ref)
    ref_emb = emb_by_path.get(ref_path) if ref_path else None
    if ref_emb is None:
        raise ValueError('the reference image has no ✨ Score embedding — score '
                         'it first (it may have been rejected before Score ran)')
    ids, E = _pool_embeddings(bank, emb_by_path, filters or {})
    if not ids:
        return {'results': [], 'image_ids': [], 'pool': 0, 'ref_id': int(ref_id)}
    rv = np.asarray(ref_emb, dtype='float32')
    rv /= (np.linalg.norm(rv) + 1e-8)
    sims = E @ rv                                 # cosine similarity, (m,)
    order = np.argsort(-sims, kind='stable')     # desc; stable ⇒ id tie-break
    if min_score is not None:
        keep = [int(k) for k in order if sims[k] >= float(min_score)]
    else:
        n = max(1, min(int(n), _CURATION_MAX_N))
        keep = [int(k) for k in order[:n]]
    results = [{'id': ids[k], 'score': round(float(sims[k]), 4)} for k in keep]
    return {'results': results, 'image_ids': [ids[k] for k in keep],
            'pool': len(ids), 'ref_id': int(ref_id)}
def _trash_or_remove(path: str) -> str:
    """Send a source file to the OS trash when send2trash is installed, else
    hard-delete it. Returns the mode actually used ('trash' | 'delete'). The
    import is optional so an install that predates the dependency still works
    (degraded to a permanent delete) — the caller reports which happened."""
    try:
        from send2trash import send2trash   # optional dependency
    except Exception:
        os.remove(path)
        return 'delete'
    send2trash(path)
    return 'trash'


def delete_rejected(user_id, bank_id) -> dict:
    """Delete the SOURCE files of every status='reject' image from disk, then
    drop their bank_image rows.

    This is the ONLY bank action that writes to the user's source folder. It is
    destructive: with send2trash installed the files go to the OS trash (real,
    OS-level recovery); without it they are permanently removed. Either way the
    app's own trash cannot bring them back — these are files outside the app.

    Non-rejected images are never touched. Per-file failures (permission, a path
    that escapes the bank folder) are collected and reported; they never abort
    the batch. A row is dropped only when its file is gone afterwards (deleted,
    trashed, or already absent) — a file we failed to remove keeps its row so the
    user can see and retry it. Returns
    {'mode', 'deleted', 'trashed', 'already_absent', 'rows_removed', 'skipped'}.
    """
    bank = get_bank(user_id, bank_id)
    if not bank:
        raise ValueError('bank not found')
    if bank_jobs.running(bank_id):
        raise RuntimeError('a job is running on this bank — stop it first')

    rows = BankImage.query.filter_by(bank_id=bank_id, status='reject').all()
    out = {'mode': 'trash', 'deleted': 0, 'trashed': 0, 'already_absent': 0,
           'rows_removed': 0, 'skipped': []}
    remove_ids = []
    saw_hard_delete = False
    for row in rows:
        path = abs_image_path(bank, row)
        if path is None:
            # relpath escapes the bank folder — refuse to touch it, keep the row.
            out['skipped'].append({'relpath': row.relpath, 'reason': 'unsafe_path'})
            continue
        if not os.path.exists(path):
            out['already_absent'] += 1
            remove_ids.append(row.id)
            continue
        try:
            mode = _trash_or_remove(path)
        except OSError as e:
            out['skipped'].append({'relpath': row.relpath, 'reason': str(e)})
            continue
        if mode == 'trash':
            out['trashed'] += 1
        else:
            out['deleted'] += 1
            saw_hard_delete = True
        remove_ids.append(row.id)

    for i0 in range(0, len(remove_ids), _SQL_IN_CHUNK):
        BankImage.query.filter(
            BankImage.id.in_(remove_ids[i0:i0 + _SQL_IN_CHUNK])
        ).delete(synchronize_session=False)
    out['rows_removed'] = len(remove_ids)
    db.session.commit()
    # 'delete' means at least one file was permanently removed (send2trash absent
    # or it refused a path); the UI wording follows this.
    out['mode'] = 'delete' if saw_hard_delete else 'trash'
    return out


# --- cooperative-cancel subprocess driver (shared by the face + score passes) --
# A bank inference pass runs for minutes over thousands of images. Its embeddings
# are cached incrementally, but a brutal proc.kill on Stop still throws away the
# in-flight slice AND leaves the UI mute. Instead we ask the child to stop CLEANLY:
# drop a sentinel file it polls between images so it flushes its cache and reports
# how much it kept; a watchdog timer hard-kills it only if it doesn't stop within
# the grace period. The child also writes a plain-text ``<cache>.count`` sidecar we
# read back here (the Flask venv has no numpy to open the .npz), so a stopped pass
# always shows an honest count — even in the rare hard-kill case.
_INFER_CANCEL_GRACE = 15.0   # seconds a cleanly-cancelled child gets before a kill
_CACHED_RE = re.compile(r'(\d+) image\(s\), (\d+) cached')


def _safe_kill(proc):
    try:
        proc.kill()
    except Exception:  # noqa: BLE001 — already gone is fine
        pass


def _read_cache_count(cache_path):
    """The count the child last flushed to ``<cache>.count``, or None."""
    try:
        with open(str(cache_path) + '.count', encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _stopped_detail(noun, data, cache_path, total):
    """The honest end-of-pass line when the user Stopped it. Prefers the child's
    own cancel counts, falls back to the flushed sidecar count, and never invents
    a number it can't back up."""
    n = data.get('cached')
    if n is None:
        n = _read_cache_count(cache_path)
    if n is None:
        return 'Stopped — progress saved to cache; relaunch to finish and cluster'
    n = int(n)
    remaining = data.get('remaining')
    remaining = int(remaining) if remaining is not None else max(0, int(total) - n)
    return (f'Stopped — {n} {noun} ({remaining} remaining); '
            'relaunch to finish and cluster')


def _drive_infer_subprocess(job, python, script, payload, cache_path,
                            progress_re, window):
    """Run an infer subprocess, streaming its stderr progress into ``job`` and
    honouring Stop cooperatively. Returns (data, stderr_tail, returncode) where
    ``data`` is the child's last JSON line (``cancelled: true`` when it stopped
    cleanly). On the first "N cached" line it sets a "resuming" hint, so relaunching
    over a partly-cached bank doesn't look like a full recompute."""
    import json
    import threading
    cancel_file = str(cache_path) + '.cancel'
    try:
        os.remove(cancel_file)   # never inherit a stale sentinel from a past run
    except OSError:
        pass
    hint = {'shown': False}
    with window:
        proc = subprocess.Popen(
            [python, script], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        killer = {'timer': None}

        def _cancel():
            # Ask for a clean stop (the child flushes + exits), and arm a watchdog
            # that hard-kills ONLY if it doesn't stop within the grace period.
            try:
                with open(cancel_file, 'w', encoding='utf-8') as f:
                    f.write('1')
            except OSError:
                pass
            t = threading.Timer(_INFER_CANCEL_GRACE, _safe_kill, args=(proc,))
            t.daemon = True
            t.start()
            killer['timer'] = t

        bank_jobs.set_cancel_hook(job, _cancel)
        stderr_tail = deque(maxlen=5)

        def _drain_stderr():
            for line in proc.stderr:
                line = line.strip()
                if line:
                    stderr_tail.append(line)
                m = progress_re.search(line)
                if m:
                    bank_jobs.progress(job, done=int(m.group(1)), total=int(m.group(2)))
                if not hint['shown']:
                    mc = _CACHED_RE.search(line)
                    if mc:
                        hint['shown'] = True
                        total, cached = int(mc.group(1)), int(mc.group(2))
                        if 0 < cached < total:
                            bank_jobs.progress(
                                job,
                                detail=f'resuming — {cached} of {total} already cached')

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
        if killer['timer']:
            killer['timer'].cancel()
    try:
        os.remove(cancel_file)
    except OSError:
        pass
    line = next((ln for ln in reversed(stdout.splitlines())
                 if ln.strip().startswith('{')), '')
    try:
        data = json.loads(line) if line else {}
    except json.JSONDecodeError:
        data = {}
    return data, stderr_tail, proc.returncode


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
        cache_path = _face_cache_path(bank_id)
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('face_scoring.models_root') or None,
            'cache': str(cache_path),
            'cancel_file': str(cache_path) + '.cancel',
            'threshold': th['face_threshold'],
            'device': device,
        })
        python = cfg.get('face_scoring.python') or sys.executable
        window = gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu else nullcontext()
        data, stderr_tail, returncode = _drive_infer_subprocess(
            job, python, _EMBED_SCRIPT, payload, cache_path, _PROGRESS_RE, window)
        # Stopped by the user — say exactly what's kept, never a mute ✗ (the cached
        # embeddings are safe; relaunching skips them and only finishes the rest).
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            bank_jobs.progress(job, detail=_stopped_detail(
                'face embeddings cached', data, cache_path, len(paths)))
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'face pass produced no output '
                                       f'(rc={returncode})')
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
        cache_path = _score_cache_path(bank_id)
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('bank_scoring.models_root') or None,
            'cache': str(cache_path),
            'cancel_file': str(cache_path) + '.cancel',
            'style_threshold': th['style_threshold'],
        })
        python = cfg.get('bank_scoring.python') or sys.executable
        # GPU-exclusive: frees ComfyUI VRAM and blocks a training start for the
        # duration, exactly like the dataset vision passes.
        data, stderr_tail, returncode = _drive_infer_subprocess(
            job, python, _SCORE_SCRIPT, payload, cache_path, _SCORE_PROGRESS_RE,
            gpu_exclusive_vision_window(flag_ttl=1800))
        # Stopped by the user — say exactly what's kept, never a mute ✗ (the cached
        # scores/embeddings are safe; relaunching skips them and finishes the rest).
        if data.get('cancelled') or (bank_jobs.cancelled(job) and not data.get('ok')):
            bank_jobs.progress(job, detail=_stopped_detail(
                'images scored', data, cache_path, len(paths)))
            return
        if not data.get('ok'):
            tail = data.get('error') or (stderr_tail[-1] if stderr_tail else '')
            raise RuntimeError(tail or f'scoring pass produced no output '
                                       f'(rc={returncode})')
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
PIPELINE_STEPS = ('scan', 'auto_reject', 'score', 'semantic_dedup', 'watermark',
                  'faces', 'caption')
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
    semantic_groups = (db.session.query(BankImage.semantic_dup_group)
                       .filter(BankImage.bank_id == bank_id,
                               BankImage.semantic_dup_group.isnot(None))
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
        'semantic_groups': semantic_groups,
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
    if step == 'semantic_dedup':
        # Runs right after Score, reusing its cached embeddings (no GPU). Groups
        # crops/variants for review; resolution stays a UI action (keep best/first
        # /manual) — near-dups are fuzzier than exact dHash, so the overnight run
        # surfaces them rather than auto-rejecting. Skipped-with-reason (never a
        # mute ✗) when Score produced no embeddings.
        n = rebuild_semantic_dup_groups(bank_id)
        if n is None:
            entry['status'], entry['reason'] = 'skipped', 'run ✨ Score first — no embeddings'
            return
        entry['counts'] = {'semantic_groups': n}
        entry['detail'] = f'{n} semantic near-duplicate group(s) to review'
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
