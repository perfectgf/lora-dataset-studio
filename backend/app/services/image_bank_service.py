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
from sqlalchemy import func, or_

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
        'flags': image_flags(row, th),
        'dup_group': row.dup_group,
        'face_state': row.face_state, 'face_cluster': row.face_cluster,
        'status': row.status, 'reject_reason': row.reject_reason,
        'promoted_dataset_id': row.promoted_dataset_id,
    }


def _flag_filter(flag: str, th: dict):
    """SQLAlchemy criterion for one flag name (mirrors image_flags)."""
    if flag == 'unreadable':
        return BankImage.quality_state == 'unreadable'
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
    }
    flags = {}
    for flag in _QUALITY_FLAGS:
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
    return {
        'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
        'created_at': bank.created_at.isoformat() if bank.created_at else None,
        'counts': counts, 'flags': flags, 'dup': dup,
        'clusters': clusters, 'faces_scanned': faces_scanned,
        'activity': bank_jobs.get(bank_id),
        'thresholds': th,
    }


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
                group=None, offset=0, limit=200) -> dict | None:
    """One PAGE of the bank grid (a 9 000-image bank must never ship whole).
    Filters compose: status ∩ flag ∩ cluster ∩ dup-group. Flag filters sort by
    the relevant score (worst first) so the review reads top-down."""
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
        q = q.filter(BankImage.face_state.isnot(None),
                     BankImage.face_state != 'scorable')
    elif flag in _QUALITY_FLAGS:
        crit = _flag_filter(flag, th)
        if crit is not None:
            q = q.filter(crit)
        order = {'blur': BankImage.blur_score.asc(),
                 'noise': BankImage.noise_score.desc(),
                 'uniform': BankImage.uniformity_score.asc(),
                 'small': BankImage.width.asc(),
                 'unreadable': BankImage.id.asc()}[flag]
    if cluster is not None:
        q = q.filter(BankImage.face_cluster == int(cluster))
    if group is not None:
        q = q.filter(BankImage.dup_group == int(group))
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
    """'Keep best' heuristic for a duplicate group: most pixels, then sharpest,
    then heaviest file — a Telegram dump's duplicates are mostly re-compressed
    or downscaled copies, so surface area is the honest first key."""
    def key(r):
        return ((r.width or 0) * (r.height or 0), r.blur_score or 0.0,
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
        if flag not in _QUALITY_FLAGS:
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
        payload = _json.dumps({
            'images': paths,
            'models_root': cfg.get('face_scoring.models_root') or None,
            'cache': str(_face_cache_path(bank_id)),
            'threshold': th['face_threshold'],
        })
        python = cfg.get('face_scoring.python') or sys.executable
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
            blobs, chunk_rows = [], []
            for r in chunk:
                p = abs_image_path(bank, r)
                try:
                    with open(p, 'rb') as fh:
                        blobs.append(fh.read())
                    chunk_rows.append(r)
                except (OSError, TypeError):
                    failed += 1
            if blobs:
                new_ids, bad = import_images(user_id, dataset_id, blobs,
                                             dedupe=True, stats=stats)
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
