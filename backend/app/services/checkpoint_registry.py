"""Provenance registry for training runs — "which VERSION of the dataset
produced this checkpoint?"

Every training LAUNCH (local or cloud) records a TrainingRunRecord carrying a
FINGERPRINT of the dataset's training-relevant state (kept images + captions +
per-file size/mtime + trigger + kind). A fingerprint never seen for this
(dataset, family) allocates the next human version (v1, v2, ...); re-running an
unchanged dataset keeps its version. The stored MANIFEST lets the UI say WHAT
changed since a version ("+2 images, 3 captions edited"), not just that it did.

Registration is best-effort by design: a registry failure must never block a
training launch (the feature is provenance, not a gate)."""
from __future__ import annotations

import hashlib
import json
import logging
import os

from sqlalchemy import and_, or_

from ..extensions import db
from ..models import FaceDatasetImage, TrainingRunRecord
from ..utils.timestamps import naive_utcfromtimestamp
from . import face_dataset_service as fds
from . import run_archive, run_snapshot

logger = logging.getLogger(__name__)


def _caption_hash(text) -> str:
    return hashlib.sha1((text or '').encode('utf-8')).hexdigest()[:8]


def _file_hash(dataset_id, filename) -> str:
    """Cheap content proxy (size:mtime) so an image EDIT (crop, upscale)
    changes the fingerprint even though id and caption stay the same.
    Missing file -> stable sentinel, never an exception."""
    if not filename:
        return '-'
    try:
        from .. import config as cfg
        p = cfg.dataset_images_root() / str(dataset_id) / filename
        st = os.stat(p)
        return hashlib.sha1(f'{st.st_size}:{int(st.st_mtime)}'.encode()).hexdigest()[:8]
    except OSError:
        return '-'


def kept_images(dataset_id) -> list:
    """The KEPT image rows of a dataset, id-sorted — the exact set a launch
    trains on. Returned so the manifest and the full snapshot are built from ONE
    read of the dataset and can never disagree about what was in it."""
    return (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())


def dataset_manifest(dataset_id, rows=None) -> list:
    """Training manifest of the KEPT images, id-sorted.

    Entries retain the historical ``[id, long_hash, file_hash]`` shape when no
    short caption exists.  A fourth short-caption hash is appended when present,
    so editing only the caption variant that ai-toolkit can sample is no longer
    invisible to run comparison.  Existing single-caption fingerprints remain
    byte-for-byte stable.

    `file_hash` stays the cheap `size:mtime` proxy on purpose: this list feeds
    `fingerprint_of()`, and every dataset version already stored in every
    existing database is keyed on it. The TRUE content hash lives in the run
    snapshot (`run_snapshot`), where it can improve the diff without renumbering
    everybody's versions."""
    rows = kept_images(dataset_id) if rows is None else rows
    manifest = []
    for row in rows:
        entry = [row.id, _caption_hash(row.caption),
                 _file_hash(dataset_id, row.filename)]
        short = (getattr(row, 'caption_short', None) or '').strip()
        if short:
            entry.append(_caption_hash(short))
        manifest.append(entry)
    return manifest


def fingerprint_of(manifest, trigger='', kind='') -> str:
    blob = json.dumps([trigger or '', kind or '', manifest],
                      separators=(',', ':'))
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]


def manifest_diff(old, new) -> dict:
    """What changed between two manifests: image ids added/removed, captions
    edited, image files edited (same id, different content proxy)."""
    old_by_id = {e[0]: e for e in (old or [])}
    new_by_id = {e[0]: e for e in (new or [])}
    added = sorted(set(new_by_id) - set(old_by_id))
    removed = sorted(set(old_by_id) - set(new_by_id))
    captions = sum(
        1 for i in set(old_by_id) & set(new_by_id)
        if (old_by_id[i][1] != new_by_id[i][1]
            or (old_by_id[i][3] if len(old_by_id[i]) > 3 else None)
            != (new_by_id[i][3] if len(new_by_id[i]) > 3 else None)))
    edited = sum(1 for i in set(old_by_id) & set(new_by_id)
                 if len(old_by_id[i]) > 2 and len(new_by_id[i]) > 2
                 and old_by_id[i][2] != new_by_id[i][2])
    return {'images_added': len(added), 'images_removed': len(removed),
            'captions_changed': captions, 'images_edited': edited}


def prepare_launch(user_id, dataset_id, base_model=None):
    """All the READING a launch registration needs, done BEFORE any lock or
    transaction: the kept-image rows, the manifest, and the full freeze
    (captions, per-image content hashes, environment).

    Split out of `register_launch` for one reason: hashing images and probing the
    environment is file I/O measured in tenths of a second, and the local launch
    path calls the registry while holding `_queue_lock` — the same launch path
    that lost cloud runs to `database is locked`. Everything slow happens here,
    outside; the write is one short commit.

    Returns an opaque dict for `register_launch(prepared=...)`, or None when the
    dataset is gone. Never raises."""
    try:
        ds = fds.get_dataset(user_id, dataset_id)
        if ds is None:
            return None
        rows = kept_images(dataset_id)
        manifest = dataset_manifest(dataset_id, rows)
        try:
            from .. import config as cfg
            images_dir = str(cfg.dataset_images_root() / str(dataset_id))
        except Exception:
            images_dir = ''
        snapshot, sig_updates = run_snapshot.build(
            ds, rows, images_dir, base_model=base_model)
        # The archive plan is materialised HERE, as plain tuples, while the ORM
        # rows are still live: the background archiver runs after the commit,
        # where touching an expired row would fire a query per image from a
        # thread that owns no session.
        sigs = (snapshot or {}).get('images') or {}
        plan = []
        for r in rows:
            sig = (sigs.get(str(r.id)) or {}).get('c')
            if sig and r.filename:
                plan.append((os.path.join(images_dir, r.filename), sig, r.filename))
        return {'ds': ds, 'manifest': manifest, 'snapshot': snapshot,
                'sig_updates': sig_updates, 'archive': plan}
    except Exception:
        logger.exception('launch snapshot preparation failed (launch continues)')
        return None


def prepared_generation_identity(prepared):
    """Serializable exact Dataset identity captured by ``prepare_launch``.

    Environment probes are deliberately excluded: they describe where a run is
    launched, not the image/caption generation exported to ai-toolkit.  Every
    kept row must carry a content signature; otherwise callers cannot prove two
    generations equal and must fail closed.
    """
    if not isinstance(prepared, dict):
        return None
    ds = prepared.get('ds')
    manifest = prepared.get('manifest')
    snapshot = prepared.get('snapshot')
    if ds is None or not isinstance(manifest, list) or not isinstance(snapshot, dict):
        return None
    images = snapshot.get('images') or {}
    if not isinstance(images, dict):
        return None
    for entry in manifest:
        if (not isinstance(entry, list) or not entry
                or not isinstance(images.get(str(entry[0])), dict)
                or not images[str(entry[0])].get('c')):
            return None
    stable_snapshot = {
        key: value for key, value in snapshot.items() if key != 'env'}
    return {
        'fingerprint': fingerprint_of(
            manifest, ds.trigger_word, getattr(ds, 'kind', '')),
        'manifest': manifest,
        'snapshot': stable_snapshot,
    }


def record_generation_identity(record):
    """The persisted counterpart of :func:`prepared_generation_identity`."""
    if record is None:
        return None
    try:
        manifest = json.loads(record.manifest or 'null')
        snapshot = json.loads(record.snapshot or 'null')
    except (TypeError, ValueError):
        return None
    if not isinstance(manifest, list) or not isinstance(snapshot, dict):
        return None
    images = snapshot.get('images') or {}
    if not isinstance(images, dict):
        return None
    for entry in manifest:
        if (not isinstance(entry, list) or not entry
                or not isinstance(images.get(str(entry[0])), dict)
                or not images[str(entry[0])].get('c')):
            return None
    return {
        'fingerprint': record.fingerprint,
        'manifest': manifest,
        'snapshot': {
            key: value for key, value in snapshot.items() if key != 'env'},
    }


def register_launch(user_id, dataset_id, family, source, base_model='',
                    variant=None, masked=True, steps=None, cloud_run_id=None,
                    settings=None, parent_record_id=None, resumed_from=None,
                    prepared=None):
    """Record a training launch and return its TrainingRunRecord (or None on
    failure — provenance must never block a launch).

    ``parent_record_id``/``resumed_from`` are set only by a CONTINUATION (the
    record this launch resumed from, and the step it resumed at) — the lineage
    edge the Runs-hub genealogy tree draws. Both NULL on a fresh launch.

    ``prepared`` is the result of `prepare_launch` when the caller did the
    reading up front (both real launch paths do). Without it the reading happens
    here, which is correct but holds whatever lock the caller is under."""
    try:
        if prepared is None:
            prepared = prepare_launch(user_id, dataset_id, base_model=base_model)
        if prepared is None:
            return None
        ds = prepared['ds']
        manifest = prepared['manifest']
        snapshot = prepared.get('snapshot') or {}
        fp = fingerprint_of(manifest, ds.trigger_word, getattr(ds, 'kind', ''))
        same = (TrainingRunRecord.query
                .filter_by(dataset_id=dataset_id, family=family, fingerprint=fp)
                .first())
        if same is not None:
            version = same.version
        else:
            newest = (TrainingRunRecord.query
                      .filter_by(dataset_id=dataset_id, family=family)
                      .order_by(TrainingRunRecord.version.desc()).first())
            version = (newest.version + 1) if newest else 1
        rec = TrainingRunRecord(
            dataset_id=dataset_id, family=family, source=source,
            cloud_run_id=cloud_run_id, base_model=base_model or '',
            variant=variant, masked=bool(masked), steps=steps,
            settings=json.dumps(settings) if settings else None,
            fingerprint=fp, manifest=json.dumps(manifest), version=version,
            snapshot=json.dumps(snapshot, separators=(',', ':')) if snapshot else None,
            parent_record_id=parent_record_id, resumed_from=resumed_from)
        db.session.add(rec)
        # The freshly computed image content hashes ride along in the SAME
        # transaction as the record — one short write, never a second one racing
        # the launch for the database lock.
        for row, sig, stat in (prepared.get('sig_updates') or ()):
            row.content_sig = sig
            row.content_sig_stat = stat
        db.session.commit()
        # Archiving copies files; it happens AFTER the commit, off the launch
        # path, and its failure is invisible to the run.
        run_archive.archive_async(prepared.get('archive'))
        return rec
    except Exception:
        logger.exception('training run registration failed (launch continues)')
        db.session.rollback()
        return None


def latest_record(dataset_id, family):
    return (TrainingRunRecord.query
            .filter_by(dataset_id=dataset_id, family=family)
            .order_by(TrainingRunRecord.id.desc()).first())


def newest_record_for(dataset_id, family, base_model='', variant=None):
    """The most recent record of a SPECIFIC training lane (dataset+family+base
    +variant) — the run a continuation of that lane resumes from. base_model is
    normalized to '' (official base) to match how register_launch stores it."""
    q = (TrainingRunRecord.query
         .filter_by(dataset_id=dataset_id, family=family, base_model=base_model or ''))
    if variant is not None:
        q = q.filter_by(variant=variant)
    return q.order_by(TrainingRunRecord.id.desc()).first()


def record_by_id(record_id):
    """One record by primary key, or None. The counterpart of the `record_id`
    stamp list_checkpoints puts on every save: a continuation resolves its
    lineage parent from the file it actually resumes, not from whichever record
    happens to be the newest."""
    if record_id is None:
        return None
    try:
        return db.session.get(TrainingRunRecord, int(record_id))
    except (TypeError, ValueError):
        return None


def network_geometry(rec):
    """The adapter topology a record explicitly says it trained.

    Rank/alpha, adapter kind, and LoKr's factor/full-rank mode all affect which
    tensors a checkpoint can load.  Only persisted, valid facts are returned:
    old records lack the newer topology keys and must stay *unknown*, never be
    retroactively interpreted as today's defaults.
    """
    out = {}
    if rec is None or not getattr(rec, 'settings', None):
        return out
    try:
        cfg = json.loads(rec.settings)
    except (ValueError, TypeError):
        return out
    if not isinstance(cfg, dict):
        return out
    for key in ('rank', 'alpha'):
        val = cfg.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, int) and val > 0:
            out[key] = val
    network_type = cfg.get('network_type')
    if network_type not in ('lora', 'lokr'):
        return out
    out['network_type'] = network_type
    if network_type == 'lora':
        # Conv LoRA adds a second set of adapter tensors. New snapshots record
        # explicit nulls for linear-only; missing keys on legacy records remain
        # unknown and are intentionally not inferred.
        for key in ('conv', 'conv_alpha'):
            if key not in cfg:
                continue
            val = cfg[key]
            if val is None:
                out[key] = None
            elif isinstance(val, int) and not isinstance(val, bool) and val > 0:
                out[key] = val
    # These keys only describe a LoKr checkpoint.  Their absence on an older
    # snapshot is meaningful: ai-toolkit's defaults changed, so guessing False
    # or auto would turn an unknown legacy fact into an unsafe assertion.
    if network_type == 'lokr':
        full_rank = cfg.get('lokr_full_rank')
        if isinstance(full_rank, bool):
            out['lokr_full_rank'] = full_rank
        factor = cfg.get('lokr_factor')
        if factor == 'auto' or (isinstance(factor, int)
                               and not isinstance(factor, bool)
                               and factor in (4, 8, 16, 32)):
            out['lokr_factor'] = factor
    return out


def resolve_lineage(record_id):
    """The whole lineage COMPONENT containing ``record_id``, as a root-first BFS
    list of TrainingRunRecord. Edges are the persisted parent_record_id links:
    climb to the single root (each node has at most one parent), then breadth-
    first over children so start's siblings and their subtrees are all included.
    Defensive against a malformed cycle (a record can't legitimately be its own
    ancestor). A record with no parent and no children is its own single-node
    lineage — the caller decides whether that's worth surfacing."""
    start = db.session.get(TrainingRunRecord, int(record_id))
    if start is None:
        return []
    root, climbed = start, {start.id}
    while root.parent_record_id and root.parent_record_id not in climbed:
        parent = db.session.get(TrainingRunRecord, root.parent_record_id)
        if parent is None:
            break
        climbed.add(parent.id)
        root = parent
    order, visited, frontier = [], set(), [root]
    while frontier:
        node = frontier.pop(0)
        if node.id in visited:
            continue
        visited.add(node.id)
        order.append(node)
        frontier.extend(TrainingRunRecord.query
                        .filter_by(parent_record_id=node.id)
                        .order_by(TrainingRunRecord.id.asc()).all())
    return order


def records_with_children(record_ids):
    """Subset of ``record_ids`` that are the parent of at least one record — so
    the Runs list can flag which rows open into a lineage without resolving each
    tree. One query; empty input short-circuits."""
    ids = [int(i) for i in record_ids if i is not None]
    if not ids:
        return set()
    rows = (db.session.query(TrainingRunRecord.parent_record_id)
            .filter(TrainingRunRecord.parent_record_id.in_(ids)).distinct().all())
    return {r[0] for r in rows}


def ensure_baseline(user_id, dataset_id, family, had_training) -> None:
    """Retrofit for PRE-FEATURE datasets: a dataset that was ALREADY trained
    before the registry existed has checkpoints but no records — without this,
    versioning would only ever apply to future work (deployed-project rule:
    always catch the past up). When training evidence exists and nothing is
    registered, record the CURRENT state as the v1 baseline (source 'legacy'):
    existing checkpoints display as v1 and the next dataset change bumps v2.
    The true historical state is unknowable — 'now' is the honest baseline.
    Best-effort and idempotent."""
    try:
        if not had_training or latest_record(dataset_id, family) is not None:
            return
        register_launch(user_id, dataset_id, family, source='legacy')
    except Exception:
        logger.exception('baseline backfill failed (non-fatal)')


_ANY = object()


def record_for_mtime(dataset_id, family, mtime_ts, *, base_model=_ANY,
                     variant=_ANY, source=_ANY):
    """The run record a FILE most plausibly belongs to: the newest record
    created BEFORE the file was written (records are created at launch, files
    after). A file older than EVERY record predates the registry — its most
    plausible owner is the OLDEST record (the legacy baseline), not the
    newest (live sighting: yesterday's local checkpoints wore a ☁ chip
    because a cloud launch happened to be the latest record). None when
    nothing is registered."""
    query = TrainingRunRecord.query.filter_by(
        dataset_id=dataset_id, family=family)
    # A local lane is scoped by base + variant + source. Without these filters,
    # an unrelated cloud launch recorded between the local launch and checkpoint
    # write can steal the checkpoint's version and run lineage solely by mtime.
    scope = []
    if base_model is not _ANY:
        scope.append(TrainingRunRecord.base_model == base_model)
    if variant is not _ANY:
        scope.append(TrainingRunRecord.variant == variant)
    if source is not _ANY:
        sources = (
            tuple(source)
            if isinstance(source, (tuple, list, set, frozenset))
            else (source,))
        # Legacy baselines intentionally have no trustworthy base/variant.
        # They are the historical fallback for a local lane, while every real
        # local record must match the precise lane. Cloud records never qualify.
        if 'legacy' in sources and scope:
            concrete = tuple(item for item in sources if item != 'legacy')
            concrete_scope = list(scope)
            if concrete:
                concrete_scope.append(TrainingRunRecord.source.in_(concrete))
                query = query.filter(or_(
                    and_(*concrete_scope),
                    TrainingRunRecord.source == 'legacy'))
            else:
                query = query.filter(TrainingRunRecord.source == 'legacy')
        else:
            query = query.filter(*scope)
            query = query.filter(TrainingRunRecord.source.in_(sources))
    else:
        query = query.filter(*scope)
    recs = query.order_by(TrainingRunRecord.created_at.desc()).all()
    if not recs:
        return None
    try:
        ts = naive_utcfromtimestamp(float(mtime_ts))
        for r in recs:
            if r.created_at and r.created_at <= ts:
                return r
    except (OverflowError, OSError, ValueError):
        pass
    return recs[-1]


def dataset_state(user_id, dataset_id, family) -> dict:
    """Current-vs-latest-version comparison for the UI: {registered, version,
    fingerprint, changed, diff} — `changed` is True when the CURRENT dataset
    differs from the latest registered version's manifest."""
    ds = fds.get_dataset(user_id, dataset_id)
    if ds is None:
        return {'registered': False}
    manifest = dataset_manifest(dataset_id)
    fp = fingerprint_of(manifest, ds.trigger_word, getattr(ds, 'kind', ''))
    latest = latest_record(dataset_id, family)
    if latest is None:
        return {'registered': False, 'fingerprint': fp}
    try:
        old = json.loads(latest.manifest or '[]')
    except ValueError:
        old = []
    changed = latest.fingerprint != fp
    return {'registered': True, 'version': latest.version,
            'fingerprint': fp, 'changed': changed,
            'diff': manifest_diff(old, manifest) if changed else None}
