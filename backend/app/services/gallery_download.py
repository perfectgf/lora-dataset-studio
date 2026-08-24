"""⬇ Getting generated images OFF the board — one at a time, or a gallery as ZIP.

WHY THE FILE NAME IS THE FEATURE
--------------------------------
The ◉ LoRA Canvas is the only screen in the app that knows an image's whole
ancestry: which dataset, which run, which checkpoint, which seed.  Until now
that knowledge could only be *looked at*.  The moment a picture leaves for the
Downloads folder it keeps whatever ComfyUI happened to call it —
``out_00042_.png`` — and a week later nobody can say which checkpoint produced
it.  Two files from two checkpoints of the same run are then indistinguishable,
which defeats the only reason the board exists.

A settings sidecar (a ``.txt``/``.json`` beside each image) was deliberately
ruled out.  So **the file name is the sole carrier of the lineage**, and it has
to earn that:

    ``<dataset>_run<id>_step<NNNNNN>_seed<seed>_<imageId>.<ext>``
    e.g. ``Nova-Style_run42_step002500_seed208607443_1187.png``

  * **zero-padded step** — the whole point.  Sorted as text, ``2500`` comes
    before ``500``; ``002500`` does not.  Six digits covers every realistic
    schedule, and a step above that simply stops being padded rather than being
    truncated into a lie.
  * **``stepunknown`` for a step-less save** — ``u`` sorts after the digits, so
    the group the run gallery puts last lands last here too, for free.
  * **the image id last** — the collision breaker.  Two renders of the same
    checkpoint with the same seed are a normal thing (different sampler, another
    day); the row id is unique by construction, so the name is too.
  * **the seed segment disappears when there is no seed** rather than becoming
    ``seed None``.
  * portable: NTFS's forbidden set ``<>:"/\\|?*`` never survives the slug,
    neither do control bytes, trailing dots/spaces or the DOS device names, and
    the whole thing is bounded well under the 255-byte component limit that
    Windows, ext4 and APFS all share.

WHY NOT DEFLATE
---------------
Every one of these files is a PNG or a JPEG — already compressed.  Deflating
them again buys a percent or two for a lot of CPU on a machine that is usually
also training something, so the archive is STORED.  A gallery ZIP is then
roughly a disk copy.

CAPS ARE ANNOUNCED, NEVER SILENT
--------------------------------
``gallery_download_plan`` is the cheap half — rows and ``os.path.exists`` only,
no bytes moved — so the UI can say *before* the click how many images it is
about to get, how many files have vanished, and whether the cap cut the list.
A ZIP that quietly holds 500 of 812 images is exactly the kind of half-truth
this project removes everywhere else.
"""
from __future__ import annotations

import os
import re
import zipfile

from .dataset_storage import dataset_path

# How many images one archive may hold.  The gallery panel already caps its
# PAYLOAD at 240; this is the harder, byte-moving limit and it sits at the same
# 500 the checkpoint gallery uses for its own list.  Above it the request would
# hold a worker for minutes on a machine that is usually also busy.
ZIP_IMAGE_CAP = 500

# Bounded so the whole name stays far under the 255-byte component limit shared
# by NTFS, ext4 and APFS — a dataset called "…" 400 characters long must not
# produce a file nobody can save.
MAX_SLUG_LEN = 40
MAX_NAME_LEN = 120

# The names Windows still refuses whatever the extension, because they are
# devices.  A generated name always continues with `_run…`, so this can only
# bite through the slug — it is checked on the finished stem anyway.
RESERVED_DEVICE_NAMES = frozenset(
    ['CON', 'PRN', 'AUX', 'NUL']
    + [f'COM{i}' for i in range(1, 10)]
    + [f'LPT{i}' for i in range(1, 10)])

_SLUG_KEEP = re.compile(r'[^A-Za-z0-9]+')
_EXT_OK = re.compile(r'^[a-z0-9]{1,5}$')


def slugify(text, fallback='dataset'):
    """A file-name-safe fragment of `text`, or `fallback` if nothing survives.

    Deliberately ASCII-only: a name that round-trips through a ZIP, a shell, a
    Discord upload and someone else's file manager cannot depend on the
    encoding any of them chose.  A fully non-Latin dataset name therefore keeps
    its identity through the run id rather than through its letters.
    """
    slug = _SLUG_KEEP.sub('-', str(text or '')).strip('-')
    slug = slug[:MAX_SLUG_LEN].strip('-')
    return slug or fallback


def _extension(filename):
    """The stored file's extension, lowercased and vetted. Anything odd becomes
    `png`: the extension decides what opens the file, so it is the one segment
    that must never carry user-influenced text."""
    ext = os.path.splitext(str(filename or ''))[1].lstrip('.').lower()
    return ext if _EXT_OK.match(ext) else 'png'


def _step_segment(step):
    if step is None:
        return 'stepunknown'
    try:
        n = int(step)
    except (TypeError, ValueError):
        return 'stepunknown'
    return f'step{n:06d}' if 0 <= n < 1000000 else f'step{n}'


def image_download_name(row, dataset_name=None):
    """The lineage-carrying name for ONE generated image row.

    `row` is a ``LoraTestImage``.  `dataset_name` is passed in rather than
    fetched so a ZIP of 500 images does not run 500 dataset lookups.
    """
    slug = slugify(dataset_name, f'dataset{row.dataset_id}')
    parts = [slug, f'run{row.record_id}' if row.record_id is not None else 'rununknown',
             _step_segment(row.step)]
    if row.seed is not None:
        parts.append(f'seed{row.seed}')
    parts.append(str(row.id))
    stem = '_'.join(parts)
    ext = _extension(row.filename)
    # Trim from the SLUG, never from the tail: run / step / id are the lineage
    # and dropping them would leave a long pretty name that says nothing.
    over = len(stem) + 1 + len(ext) - MAX_NAME_LEN
    if over > 0:
        slug = slug[:max(1, len(slug) - over)].strip('-') or 'dataset'
        stem = '_'.join([slug] + parts[1:])
    if stem.upper() in RESERVED_DEVICE_NAMES:
        stem = f'_{stem}'
    return f'{stem}.{ext}'


def _rows_for(record_id, step, image_ids=None):
    """The finished, file-backed rows of this scope, newest first.

    Scoped exactly like ``delete_checkpoint_images``: ``step=None`` means the
    whole run, and an id belonging to another run is simply not found.  Reusing
    the delete's shape is on purpose — a download that could reach rows the
    delete cannot would be a second, quieter definition of "this gallery".
    """
    from ..models import LoraTestImage
    q = LoraTestImage.query.filter(LoraTestImage.record_id == record_id,
                                   LoraTestImage.status == 'done',
                                   LoraTestImage.filename.isnot(None))
    if step is not None:
        q = q.filter(LoraTestImage.step == step)
    if image_ids is not None:
        wanted = []
        for i in image_ids:
            try:
                wanted.append(int(i))
            except (TypeError, ValueError):
                continue
        if not wanted:
            return []
        q = q.filter(LoraTestImage.id.in_(wanted))
    return q.order_by(LoraTestImage.id.desc()).all()


def _dataset_names(rows):
    from ..models import FaceDataset
    ids = {r.dataset_id for r in rows if r.dataset_id is not None}
    if not ids:
        return {}
    return {d.id: d.name
            for d in FaceDataset.query.filter(FaceDataset.id.in_(ids)).all()}


def zip_download_name(record_id, step, dataset_name, partial=False):
    """What the archive itself is called — the same grammar as its contents."""
    slug = slugify(dataset_name, f'dataset{record_id}')
    tail = 'selection' if partial else 'gallery'
    scope = '' if step is None else f'_{_step_segment(step)}'
    return f'{slug}_run{record_id}{scope}_{tail}.zip'


def _note(total, included, missing, truncated, cap):
    """The sentence the screen shows. Every cut is named here, once, so the
    button, the toast and the tests cannot drift into three different stories."""
    if total == 0:
        return 'There is nothing to download here yet.'
    if included == 0:
        return (f'None of these {total} images can be downloaded — '
                f'{missing} file(s) are no longer on disk.')
    bits = []
    if truncated:
        bits.append(f'Downloading the newest {included} of {total} '
                    f'(one archive holds at most {cap}).')
    else:
        bits.append(f'Downloading {included} image(s).')
    if missing:
        bits.append(f'{missing} file(s) are no longer on disk and were left out.')
    return ' '.join(bits)


def gallery_download_plan(record_id, step, image_ids=None, cap=ZIP_IMAGE_CAP):
    """What a ZIP of this scope WOULD hold — computed without moving a byte.

    Returns ``{ok, total, included, missing, truncated, cap, note, filename,
    entries}`` where each entry is ``{id, path, name}``.  `ok` is False when
    there is nothing left to put in the archive, which is the case the button
    has to refuse out loud rather than hand over an empty ZIP.
    """
    cap = max(1, int(cap or ZIP_IMAGE_CAP))
    rows = _rows_for(record_id, step, image_ids)
    total = len(rows)
    kept = rows[:cap]                      # newest first — the panel's own rule
    truncated = total > len(kept)
    names = _dataset_names(kept)
    entries, missing, seen = [], 0, set()
    for row in kept:
        path = os.path.join(dataset_path(row.dataset_id), row.filename)
        if not os.path.exists(path):
            missing += 1
            continue
        name = image_download_name(row, names.get(row.dataset_id))
        # Belt and braces: the row id is already in the name, so this can only
        # fire if the scheme ever changes underneath us.
        if name in seen:
            name = f'{row.id}_{name}'
        seen.add(name)
        entries.append({'id': row.id, 'path': path, 'name': name})
    included = len(entries)
    first = kept[0] if kept else None
    ds_name = names.get(first.dataset_id) if first is not None else None
    return {
        'ok': included > 0,
        'total': total,
        'included': included,
        'missing': missing,
        'truncated': truncated,
        'cap': cap,
        'note': _note(total, included, missing, truncated, cap),
        'filename': zip_download_name(record_id, step, ds_name,
                                      partial=image_ids is not None),
        'entries': entries,
    }


def app_zip_download_name(rows, names):
    """What a Gallery-page archive is called. One dataset → its slug keeps the
    identity (`<slug>_gallery_selection.zip`); a cross-dataset pick has no one
    name to carry, so the scope word does the job alone."""
    ds_ids = {r.dataset_id for r in rows}
    if len(ds_ids) == 1:
        only = next(iter(ds_ids))
        return f'{slugify(names.get(only), f"dataset{only}")}_gallery_selection.zip'
    return 'gallery_selection.zip'


def app_gallery_download_plan(image_ids, cap=ZIP_IMAGE_CAP):
    """The Gallery page's selection as a ZIP plan — same answer shape as
    ``gallery_download_plan``, with the (record_id, step) scope replaced by an
    explicit id list. Always a SELECTION: the feed can span thousands of images
    across every run, so "the whole scope" is not a thing this plan offers —
    a caller that wants everything must say which everything.

    Rows are matched by id across the whole table (status done, file-backed),
    exactly the rows the feed lists; an id that no longer exists simply is not
    found, which the counts then state."""
    from ..models import LoraTestImage
    cap = max(1, int(cap or ZIP_IMAGE_CAP))
    wanted = []
    for i in (image_ids or []):
        try:
            wanted.append(int(i))
        except (TypeError, ValueError):
            continue
    rows = []
    if wanted:
        rows = (LoraTestImage.query
                .filter(LoraTestImage.id.in_(wanted),
                        LoraTestImage.status == 'done',
                        LoraTestImage.filename.isnot(None))
                .order_by(LoraTestImage.id.desc()).all())
    total = len(rows)
    kept = rows[:cap]
    truncated = total > len(kept)
    names = _dataset_names(kept)
    entries, missing, seen = [], 0, set()
    for row in kept:
        path = os.path.join(dataset_path(row.dataset_id), row.filename)
        if not os.path.exists(path):
            missing += 1
            continue
        name = image_download_name(row, names.get(row.dataset_id))
        if name in seen:
            name = f'{row.id}_{name}'
        seen.add(name)
        entries.append({'id': row.id, 'path': path, 'name': name})
    included = len(entries)
    return {
        'ok': included > 0,
        'total': total,
        'included': included,
        'missing': missing,
        'truncated': truncated,
        'cap': cap,
        'note': _note(total, included, missing, truncated, cap),
        'filename': app_zip_download_name(kept, names),
        'entries': entries,
    }


def write_gallery_zip(entries, output):
    """Write the planned entries into `output`, sorted by their names.

    STORED, not deflated: these are PNG/JPEG bytes and re-compressing them costs
    CPU for nothing.  Sorted, so the archive opens in training order — which is
    exactly what the zero-padded step in the name buys.
    """
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as zf:
        for e in sorted(entries, key=lambda x: x['name']):
            try:
                zf.write(e['path'], arcname=e['name'])
            except OSError:
                # A file that vanished between the plan and the write. Skipping
                # is right — the plan already reported what it found, and one
                # unreadable file must not lose the other 499.
                continue


def single_image_download(image_id):
    """``(path, name)`` for one generated image, or ``(None, reason)``.

    A plain ``<a download>`` on the existing image URL would have been shorter
    and is exactly what this avoids: when the file is gone the browser saves the
    404 page under the .png name and the user finds out by opening it. Resolving
    here means the caller gets a refusal it can say out loud.
    """
    from ..extensions import db
    from ..models import FaceDataset, LoraTestImage
    row = db.session.get(LoraTestImage, image_id)
    if row is None or not row.filename:
        return None, 'That image is no longer in the library.'
    path = os.path.join(dataset_path(row.dataset_id), row.filename)
    if not os.path.exists(path):
        return None, 'That image file is no longer on disk.'
    ds = db.session.get(FaceDataset, row.dataset_id) if row.dataset_id else None
    return path, image_download_name(row, ds.name if ds else None)
