"""What ACTUALLY differs between two training runs.

The Lab's compare panel could already put two settings tables side by side. What
it could not do is answer the question the settings never contain: *the dataset
changed — how?* The manifest knew that three captions were edited and refused to
say which; it knew an image was added and could only name its id; and it said
nothing at all about the machine, so a gap caused by a `git pull` in ai-toolkit
was silently charged to the dataset.

This module produces the whole comparison in one payload:

  * **settings** — the recorded recipe of each run, MERGED with the facts that
    live on the record row rather than in the snapshot (steps, base model,
    masked training, dataset version). Those were missing from the panel and are
    among the biggest differences two runs can have.
  * **images** — added / removed / caption-edited / pixel-edited, each as a LIST
    with the caption text on both sides and a picture to look at, including for
    images that have since been deleted (`run_archive`).
  * **environment** — ai-toolkit revision, torch/CUDA, GPU, and the identity of
    the base-model FILE.

Honesty rules, enforced here rather than in the UI:

  * A run recorded before snapshots existed says so (`predates_snapshot`) instead
    of rendering an empty comparison that reads like "nothing changed".
  * When only one side has the true content hash, the pixel comparison falls back
    to the legacy `size:mtime` proxy and is FLAGGED as approximate.
  * Lists are capped; the payload reports how many entries it withheld rather
    than silently truncating.

Older run is always A. Comparing "v2 with v3" and "v3 with v2" must not produce
opposite added/removed lists depending on which card was clicked first.
"""
from __future__ import annotations
from ..extensions import db

import json
import logging

from ..models import FaceDataset, FaceDatasetImage, TrainingRunRecord
from . import run_archive, run_snapshot

logger = logging.getLogger(__name__)

# A dataset can hold hundreds of images. A caption pair is a few hundred bytes,
# so a fully-rewritten 460-image dataset would be a ~300 KB response — bounded
# here, with the withheld count reported so the panel never lies by omission.
LIST_CAP = 150


def _manifest(rec):
    try:
        data = json.loads(getattr(rec, 'manifest', None) or '[]')
    except (ValueError, TypeError):
        return {}
    out = {}
    for entry in data if isinstance(data, list) else ():
        if isinstance(entry, list) and entry:
            out[entry[0]] = entry
    return out


def _record_config(rec) -> dict:
    """The run's recipe as the panel should compare it: the settings snapshot
    plus the defining facts that live on the RECORD, not in the snapshot.

    `masked`, `steps`, the base model and the dataset version were never in the
    settings blob, so the compare panel could not show that one run trained
    masked and the other did not — a difference far larger than any of the knobs
    it did show."""
    cfg = {}
    raw = getattr(rec, 'settings', None)
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                cfg = dict(parsed)
        except (ValueError, TypeError):
            cfg = {}
    cfg['steps'] = rec.steps
    cfg['base_model'] = rec.base_model or 'official base'
    cfg['dataset_version'] = f'v{rec.version}' if rec.version else None
    cfg['masked'] = 'yes' if rec.masked else 'no'
    cfg['family'] = rec.family
    if rec.variant:
        cfg['variant'] = rec.variant
    cfg['source'] = rec.source
    return cfg


def _thumb(dataset_id, image_id, sig, live_names):
    """Where the panel can fetch the image AS THIS RUN SAW IT.

    The ARCHIVED blob wins over the live file, and deliberately so: an image
    re-cropped after the run is still in the dataset, so the live file would load
    happily and show pixels that run never trained on — a confident wrong answer,
    which is the failure mode this whole feature exists to remove. The live file
    is the fallback (nothing archived: an old run, or the ceiling was reached),
    and None means the panel says the picture is unavailable rather than framing
    a broken image."""
    if sig and run_archive.path_for(sig):
        return f'/api/dataset/runs/archive/{sig}'
    name = live_names.get(image_id)
    if name:
        return f'/api/dataset/{dataset_id}/img/{name}'
    return None


def _entry(dataset_id, image_id, snap, live_names, caption_from=None):
    long_c, short_c = run_snapshot.caption_of(caption_from or snap, image_id)
    sig = run_snapshot.content_of(snap, image_id)
    meta = ((snap or {}).get('images') or {}).get(str(image_id)) or {}
    out = {'id': image_id, 'thumb': _thumb(dataset_id, image_id, sig, live_names)}
    if long_c:
        out['caption'] = long_c
    if short_c:
        out['caption_short'] = short_c
    if meta.get('e'):
        out['engine'] = meta['e']
    if meta.get('s'):
        out['origin'] = meta['s']
    return out


def _cap(items):
    return items[:LIST_CAP], max(0, len(items) - LIST_CAP)


def compare(a_id, b_id) -> dict:
    """The full two-run comparison, or `{'error': ...}` when a record is gone."""
    a = db.session.get(TrainingRunRecord, a_id)
    b = db.session.get(TrainingRunRecord, b_id)
    if a is None or b is None:
        return {'error': 'run not found'}
    # Chronological, always: A is the older run, so "removed" means "gone by B".
    if (b.created_at, b.id) < (a.created_at, a.id):
        a, b = b, a

    snap_a, snap_b = run_snapshot.loads(a), run_snapshot.loads(b)
    man_a, man_b = _manifest(a), _manifest(b)
    ids_a, ids_b = set(man_a), set(man_b)

    # Filenames of the images that are STILL in the dataset, so an image that
    # never left is shown from the live file (always current, costs no archive).
    live_names = {}
    for ds_id in {a.dataset_id, b.dataset_id}:
        for row in (FaceDatasetImage.query
                    .filter_by(dataset_id=ds_id)
                    .with_entities(FaceDatasetImage.id, FaceDatasetImage.filename)):
            if row.filename:
                live_names[row.id] = row.filename

    added = [_entry(b.dataset_id, i, snap_b, live_names) for i in sorted(ids_b - ids_a)]
    removed = [_entry(a.dataset_id, i, snap_a, live_names) for i in sorted(ids_a - ids_b)]

    caption_changed, content_changed, proxy_only = [], [], False
    for i in sorted(ids_a & ids_b):
        ea, eb = man_a[i], man_b[i]
        if len(ea) > 1 and len(eb) > 1 and ea[1] != eb[1]:
            before_l, before_s = run_snapshot.caption_of(snap_a, i)
            after_l, after_s = run_snapshot.caption_of(snap_b, i)
            caption_changed.append({
                'id': i,
                'thumb': _thumb(b.dataset_id, i, run_snapshot.content_of(snap_b, i),
                                live_names),
                'before': before_l, 'after': after_l,
                'before_short': before_s, 'after_short': after_s,
                # Neither snapshot has the text: the manifest proves the caption
                # changed, and that is all this pair can honestly claim.
                'text_recorded': bool(before_l or after_l),
            })
        sig_a, sig_b = run_snapshot.content_of(snap_a, i), run_snapshot.content_of(snap_b, i)
        if sig_a and sig_b:
            changed = sig_a != sig_b
        else:
            # Fall back to the legacy size:mtime proxy. It cannot distinguish a
            # restored backup from a re-crop, so the result is marked approximate.
            changed = len(ea) > 2 and len(eb) > 2 and ea[2] != eb[2]
            proxy_only = proxy_only or changed
        if changed:
            content_changed.append(_entry(b.dataset_id, i, snap_b, live_names))

    added, added_more = _cap(added)
    removed, removed_more = _cap(removed)
    caption_changed, captions_more = _cap(caption_changed)
    content_changed, content_more = _cap(content_changed)

    notes = []
    for rec, snap, label in ((a, snap_a, 'A'), (b, snap_b, 'B')):
        # Two different runs land here: one launched before snapshots existed, and
        # one restored from a backup (a restore keeps the machine but drops the
        # per-image half, whose ids belonged to another install). The wording
        # covers both rather than asserting a history it cannot know.
        if snap is None:
            notes.append(f'Run {label} (#{rec.id}) has no full snapshot — its '
                         'caption text, image content hashes and machine details '
                         'are not available. It predates full snapshots, or was '
                         'restored from a backup.')
        elif snap.get('restored') and not snap.get('captions'):
            notes.append(f'Run {label} (#{rec.id}) was restored from a backup: its '
                         'machine is known, but caption text and image content '
                         'hashes could not travel (a restore renumbers images).')
    if proxy_only:
        notes.append('Edited-image detection for this pair falls back to file '
                     'size and timestamp, which can flag an image that was only '
                     'restored from a backup.')
    if a.dataset_id != b.dataset_id:
        notes.append('These runs trained on DIFFERENT datasets, so every image '
                     'reads as added or removed.')

    return {
        'a': _side(a, snap_a),
        'b': _side(b, snap_b),
        'config': {'a': _record_config(a), 'b': _record_config(b)},
        'images': {
            'added': added, 'added_withheld': added_more,
            'removed': removed, 'removed_withheld': removed_more,
            'caption_changed': caption_changed, 'caption_withheld': captions_more,
            'content_changed': content_changed, 'content_withheld': content_more,
            'kept': len(ids_a & ids_b),
            'total_a': len(ids_a), 'total_b': len(ids_b),
        },
        'env': _env_rows(snap_a, snap_b),
        'notes': notes,
    }


def _side(rec, snap) -> dict:
    ds = db.session.get(FaceDataset, rec.dataset_id)
    out = {
        'record_id': rec.id,
        'version': rec.version,
        'dataset_id': rec.dataset_id,
        'dataset_name': (ds.name if ds else None),
        'created_at': rec.created_at.isoformat() if rec.created_at else None,
        'predates_snapshot': snap is None,
    }
    facts = (snap or {}).get('dataset') or {}
    if facts:
        out['dataset_facts'] = facts
    return out


# (path into env, label). Nested one level deep at most — kept as a table so a
# future probe shows up by adding a line, never by touching the panel.
_ENV_ROWS = (
    (('aitoolkit', 'commit'), 'ai-toolkit'),
    (('torch',), 'PyTorch'),
    (('cuda',), 'CUDA'),
    (('gpu',), 'GPU'),
    (('gpu_driver',), 'GPU driver'),
    (('base_file', 'name'), 'Base file'),
    (('base_file', 'sig'), 'Base file identity'),
    (('app',), 'App version'),
)


def _dig(env, path):
    node = env or {}
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if node not in ('', None) else None


def _env_rows(snap_a, snap_b) -> list:
    """One row per environment fact EITHER run recorded. A row where one side is
    null is a real difference (that run didn't record it), shown as such — the
    panel must never quietly hide half a comparison."""
    env_a = (snap_a or {}).get('env') or {}
    env_b = (snap_b or {}).get('env') or {}
    rows = []
    for path, label in _ENV_ROWS:
        va, vb = _dig(env_a, path), _dig(env_b, path)
        if va is None and vb is None:
            continue
        rows.append({'key': '.'.join(path), 'label': label,
                     'a': None if va is None else str(va),
                     'b': None if vb is None else str(vb),
                     'changed': va != vb})
    return rows
