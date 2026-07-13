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

from ..extensions import db
from ..models import FaceDatasetImage, TrainingRunRecord
from . import face_dataset_service as fds

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


def dataset_manifest(dataset_id) -> list:
    """[[image_id, caption_hash, file_hash], ...] of the KEPT images, id-sorted."""
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    return [[r.id, _caption_hash(r.caption), _file_hash(dataset_id, r.filename)]
            for r in rows]


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
    captions = sum(1 for i in set(old_by_id) & set(new_by_id)
                   if old_by_id[i][1] != new_by_id[i][1])
    edited = sum(1 for i in set(old_by_id) & set(new_by_id)
                 if len(old_by_id[i]) > 2 and len(new_by_id[i]) > 2
                 and old_by_id[i][2] != new_by_id[i][2])
    return {'images_added': len(added), 'images_removed': len(removed),
            'captions_changed': captions, 'images_edited': edited}


def register_launch(user_id, dataset_id, family, source, base_model='',
                    variant=None, masked=True, steps=None, cloud_run_id=None):
    """Record a training launch and return its TrainingRunRecord (or None on
    failure — provenance must never block a launch)."""
    try:
        ds = fds.get_dataset(user_id, dataset_id)
        if ds is None:
            return None
        manifest = dataset_manifest(dataset_id)
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
            fingerprint=fp, manifest=json.dumps(manifest), version=version)
        db.session.add(rec)
        db.session.commit()
        return rec
    except Exception:
        logger.exception('training run registration failed (launch continues)')
        db.session.rollback()
        return None


def latest_record(dataset_id, family):
    return (TrainingRunRecord.query
            .filter_by(dataset_id=dataset_id, family=family)
            .order_by(TrainingRunRecord.id.desc()).first())


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


def record_for_mtime(dataset_id, family, mtime_ts):
    """The run record a FILE most plausibly belongs to: the newest record
    created BEFORE the file was written (records are created at launch, files
    after). Fallback: the latest record. None when nothing is registered."""
    from datetime import datetime
    recs = (TrainingRunRecord.query
            .filter_by(dataset_id=dataset_id, family=family)
            .order_by(TrainingRunRecord.created_at.desc()).all())
    if not recs:
        return None
    try:
        ts = datetime.utcfromtimestamp(float(mtime_ts))
        for r in recs:
            if r.created_at and r.created_at <= ts:
                return r
    except (OverflowError, OSError, ValueError):
        pass
    return recs[0]


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
