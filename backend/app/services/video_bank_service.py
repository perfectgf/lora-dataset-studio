"""🎬 Video bank — triage a folder of rushes into a video training set.

The image lane rests on "one row = one file". This one cannot: a two-hour rush is
one file and four hundred training clips. Everything below follows from that.

WHAT A BANK STORES: BOUNDS, NOT MEDIA. A clip is a pair of PTS timestamps until
the moment it is promoted, and the only bytes this module ever writes into the
bank are thumbnails. Encoding at detection time is the obvious design and the
wrong one — cutting 340 shots to keep 128 pays 212 encodes for files nobody asked
for, and it would put media in a container whose contract says it holds decisions.
So `ffmpeg` runs exactly once per KEPT clip, at promotion, and never before.

NO PASS EVER WRITES TO THE SOURCE FOLDER. Probing, detection, thumbnailing and
promotion never open a file in the user's rushes folder for writing: thumbnails go
to ``video_banks_root()``, clips go to ``video_datasets_root()``. That is what
makes it safe to point a bank at an archive of originals and press every button.

THE ONE THING THAT DOES ADD FILES THERE IS 🕸 SCRAPE, and only because you sent it
somewhere: picking a destination bank is the consent, and what it downloads is
added to the folder that bank follows. It is an errand, not a pass. Everything
else in this file still reads.

WHY FOUR SEAMS. Probing, shot detection, thumbnailing and encoding each need
something the app cannot assume is installed (PyAV, torch, ffmpeg). They are the
only four places this module touches media, each is one function, and each is
monkeypatched by the tests — so the whole service is testable on an install with
none of the video extras, which is also what CI is.

WHY THE JOB KEY IS NAMESPACED. ``bank_jobs`` keys its registry by bank id, and the
two lanes number their banks independently: image bank 1 and video bank 1 both
exist and are different things. Sharing the raw key makes a video detection pass
refuse a click on an unrelated image bank, with a message naming a pass the user
cannot see. Hence ``job_key()`` — the registry itself is happily reused.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config as cfg
from ..extensions import db
from ..models import (VideoBank, VideoClip, VideoDataset, VideoDatasetClip,
                      VideoSource)
from . import (bank_jobs, video_metrics, video_camera_motion, ffmpeg_tools,
               path_guard, video_clip_export, video_targets, video_training)

logger = logging.getLogger(__name__)

# Lowercase, and every comparison folds the filename's case before testing it.
# `DSC_0001.MOV` is what a camera writes and what much of scraped material carries;
# a case-sensitive match creates the bank, reports zero files and says nothing, so
# the folder simply looks empty. That is a support ticket, not a naming detail.
VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.webm', '.avi')

# A ceiling on the WALK, not on the bank's history — a bank pointed at a whole
# drive must be refused without being counted to the end. Far lower than the image
# lane's: these are files that each become hundreds of rows.
BANK_MAX_FILES = 5000

# Canonical order. Detection needs the probe's fps and duration; thumbnails need
# the bounds detection produced. Running them out of order is not a preference,
# it is a pass that finds nothing to do.
PIPELINE_STEPS = ('probe', 'detect', 'thumbs')

TRIAGE_STATUSES = ('pending', 'keep', 'reject')

# A third value for VideoSource.detect_state, next to 'ok' and 'error': the user
# declared this file a single take, and its clips were decided by hand rather
# than found. Every BULK pass — the detection pass and the bank-wide re-cut —
# skips a source in this state, which is the whole point: without it, a slider
# moved once would quietly lay three detected clips back on top of the single
# one and the declaration would mean nothing. Stored as a state rather than a
# new column because that is exactly what it is, and because the source list
# already reads this field.
SINGLE_SHOT_STATE = 'single'

# The shortest span a hand-cut shot may hold. Not a UI nicety: the shortest target
# in the catalogue still asks for a fraction of a second of real footage, and a
# 0.1 s shot promotes to a file that is one frame padded out to the profile's
# length — which trains on a still. Half a second is deliberately generous: this
# is a floor against a mis-click, not an opinion about how long a shot should be.
MIN_CLIP_S = 0.5

_INSERT_CHUNK = 2000


# --- the job slot --------------------------------------------------------------

def job_key(bank_id):
    """The video lane's key into the shared ``bank_jobs`` registry.

    A STRING, so it can never collide with the image lane's integer keys whatever
    the ids happen to be. See the module docstring for the failure this avoids."""
    return f'video:{int(bank_id)}'


def cancel(bank_id) -> bool:
    return bank_jobs.cancel(job_key(bank_id))


def activity(bank_id):
    return bank_jobs.get(job_key(bank_id))


# --- storage -------------------------------------------------------------------

def _bank_dir(bank_id) -> Path:
    return cfg.video_banks_root() / str(int(bank_id))


def _thumbs_dir(bank_id) -> Path:
    return _bank_dir(bank_id) / 'thumbs'


def thumb_path(bank_id, clip_id) -> Path:
    """One .jpg per detected shot. The ONLY media a bank writes."""
    return _thumbs_dir(bank_id) / f'clip_{int(clip_id)}.jpg'


def dataset_dir(dataset_id) -> Path:
    """A video dataset's folder. FLAT — see ``_promote_job`` for why a subfolder
    here is a defect rather than a matter of taste."""
    return cfg.video_datasets_root() / str(int(dataset_id))


def _contained_path(base_dir: str, relpath: str) -> str | None:
    """`base_dir/relpath` resolved, or None when it escapes `base_dir`.

    Both sides are realpath'd (so a symlink cannot step out) and the prefix test
    carries the SEPARATOR: without it `/srv/rushes-secret` passes the check for a
    bank rooted at `/srv/rushes`."""
    base = os.path.realpath(base_dir)
    full = os.path.realpath(os.path.join(base, relpath))
    if os.path.normcase(full).startswith(os.path.normcase(base + os.sep)):
        return full
    return None


def _abs_source_path(bank: VideoBank, relpath: str) -> str | None:
    """The containment-checked absolute path of one source file.

    A relpath is data from a database that a user can edit; resolving it without
    checking it still lands under the bank's folder is how `..` reads a file the
    bank was never pointed at."""
    return _contained_path(bank.source_path, relpath)


def source_media_path(user_id, bank_id, source_id) -> str | None:
    """The readable bytes of ONE source file, for the player. None on anything
    that is not a file this bank legitimately holds.

    One return value for four different refusals (unknown bank, unknown source,
    a relpath that escapes the bank's folder, a file that has since vanished) on
    purpose: the caller answers 404 to all of them. Distinguishing "escaped the
    folder" from "not found" tells whoever tried which paths exist."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    row = VideoSource.query.filter_by(id=source_id, bank_id=bank_id).first()
    if row is None:
        return None
    path = _abs_source_path(bank, row.relpath)
    return path if path and os.path.isfile(path) else None


def dataset_clip_media_path(user_id, dataset_id, clip_id) -> str | None:
    """The bytes of one PROMOTED clip. Same contract as source_media_path.

    The filename was written by the export job rather than typed by anyone, and
    it is still checked for containment: it is a column in a database the user
    can reach, and "we wrote it ourselves" is the assumption every path-traversal
    write-up starts with."""
    ds = get_video_dataset(user_id, dataset_id)
    if ds is None or not ds.output_dir:
        return None
    row = VideoDatasetClip.query.filter_by(dataset_id=ds.id, id=clip_id).first()
    if row is None:
        return None
    path = _contained_path(ds.output_dir, row.filename or '')
    return path if path and os.path.isfile(path) else None


# --- the four media seams ------------------------------------------------------
# Each is the ONE place this module touches something the app cannot assume is
# installed. Tests replace them; nothing else in this file imports av, torch or
# subprocesses ffmpeg.

def _probe_file(path):
    """What this file is: duration, native rate, geometry, codec. Never raises —
    see services/video_probe.probe."""
    from . import video_probe
    return video_probe.probe(path)


def _detect_source(path, fps_native=None, **options):
    """One file through the detector: its clips AND the vector that produced them.

    Imported lazily and by name so an install with no detection extra fails HERE,
    per file, into detect_state='error' — rather than at import time, which would
    take the whole app down for a capability it may never use.

    Returns the dict services/shot_detect.detect_source documents. The
    probabilities in it are what the pass persists, and they are the reason a
    threshold change afterwards costs no GPU at all."""
    from . import shot_detect
    return shot_detect.detect_source(path, fps_native=fps_native, **options)


def _is_detector_unavailable(exc) -> bool:
    """Is this "the extra is not installed" rather than "this file failed"?

    services/shot_detect raises two RuntimeErrors: ShotDetectUnavailable (the
    install lacks the detector) and ShotDetectFileError (this one file defeated
    it). They must not be handled the same way — see _detect_job.

    Matched by CLASS NAME, deliberately. The condition being identified is that
    the module may not be importable at all, so importing it here to isinstance()
    against an error raised by its absence is circular. The name is part of the
    contract agreed with that module, and a rename would surface as this branch
    going quiet rather than as a crash — which is why the behaviour is pinned by
    a test rather than left to review."""
    return type(exc).__name__ == 'ShotDetectUnavailable'


def _write_thumbnail(src_path, timestamp_s, dst_path) -> bool:
    """Grab one frame at `timestamp_s` and write it as a .jpg. True on success.

    Lazy `av` import for the same reason as detection, and every failure is a
    False rather than an exception: a bank whose thumbnails failed is still a
    perfectly workable bank, it just shows placeholders."""
    try:
        import av
        from PIL import Image
    except ImportError:
        return False
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with av.open(src_path) as container:
            stream = container.streams.video[0]
            # Seek in the stream's own time base, then decode forward to the first
            # frame at or after the target: seeking lands on the preceding keyframe.
            container.seek(int(timestamp_s / stream.time_base), stream=stream)
            for frame in container.decode(stream):
                img = frame.to_image()
                img.thumbnail((480, 480), Image.LANCZOS)
                img.convert('RGB').save(dst_path, 'JPEG', quality=82)
                return True
    except Exception:                       # noqa: BLE001 — any decode error
        return False
    return False


def _run_ffmpeg(args):
    """Execute ONE clip encode. Returns (returncode, stderr tail).

    The single subprocess of this module. stderr is truncated because ffmpeg is
    verbose and the tail is where the reason lives."""
    proc = subprocess.run(args, capture_output=True, text=True,
                          encoding='utf-8', errors='replace',
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return proc.returncode, (proc.stderr or '')[-800:]


def _ffmpeg_or_raise():
    """The encoder, or a RuntimeError naming what is missing.

    Checked BEFORE a dataset row is created, so a user with no ffmpeg gets a 503
    instead of an empty dataset folder they then have to clean up."""
    path = ffmpeg_tools.ffmpeg_path()
    if not path:
        raise RuntimeError(
            'ffmpeg is required to cut clips and was not found — install the '
            'video extra from Setup, or put ffmpeg on your PATH')
    return path


# --- banks ---------------------------------------------------------------------

def get_bank(user_id, bank_id) -> VideoBank | None:
    return VideoBank.query.filter_by(id=bank_id, user_id=user_id).first()


def _scan_folder(folder) -> list:
    """Every video file under `folder`, as relpaths. Recursive, case-insensitive."""
    rels = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                rels.append(os.path.relpath(os.path.join(root, f), folder))
                if len(rels) > BANK_MAX_FILES:
                    raise ValueError(
                        f'this folder holds more than {BANK_MAX_FILES:,} videos '
                        '— point the bank at a subfolder, or split it in two')
    return rels


def create_bank(user_id, name, folder):
    """Register a folder of rushes as a bank: one row per video file.

    Instant — no decode, no detection. Those are the separate passes, because a
    two-hour file costs minutes and an HTTP request must not.
    Returns (bank, added)."""
    name = (name or '').strip()
    # Windows «Copy as path» pastes quoted; unquote so a direct paste works first
    # try, the same nicety the image bank and the dataset import already have.
    folder = (folder or '').strip().strip('"\'')
    if not name:
        raise ValueError('name is required')
    if not folder or not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder or "(empty)"}')
    # A bank and a dataset must never share bytes. Both roots are checked: the
    # image lane's (a video bank over it would be harmless today but the rule is
    # the rule) and the video lane's own, which is the real trap — promoting into
    # a folder a bank points at would make the bank list its own output as source
    # material, and re-promote it on the next pass.
    for root in (None, cfg.video_datasets_root()):
        conflict = path_guard.dataset_folder_conflict(folder, datasets_root=root)
        if conflict:
            raise ValueError(conflict['message'])
    folder = os.path.realpath(folder)
    rels = _scan_folder(folder)
    bank = VideoBank(user_id=user_id, name=name, source_path=folder)
    db.session.add(bank)
    db.session.flush()                  # need bank.id for the child rows
    _insert_sources(bank.id, folder, rels)
    db.session.commit()
    return bank, len(rels)


def _insert_sources(bank_id, folder, rels, source_metadata_by_relpath=None) -> int:
    """INSERT one row per relpath. ``source_metadata_by_relpath`` carries the
    ALREADY-VALIDATED provenance JSON of files the app itself just downloaded
    (see ``scrape_import_to_video_bank``), so a scraped rush is born WITH its
    origin instead of the walk having no way to attach one to a bare file it
    found on disk. Every other caller omits it and the column stays NULL."""
    provenance = source_metadata_by_relpath or {}
    rows = []
    for rel in rels:
        try:
            size = os.path.getsize(os.path.join(folder, rel))
        except OSError:
            size = None
        rows.append({'bank_id': bank_id, 'relpath': rel, 'file_size': size,
                     'source_metadata': provenance.get(rel)})
    for i0 in range(0, len(rows), _INSERT_CHUNK):
        db.session.execute(VideoSource.__table__.insert(), rows[i0:i0 + _INSERT_CHUNK])
    return len(rows)


def refresh_bank(user_id, bank_id, force=False, *,
                 source_metadata_by_relpath=None, _bank_lease=None) -> dict | None:
    """Re-inventory the source folder.

    STRICTLY ADDITIVE, exactly like the image lane: the only write is an INSERT of
    relpaths we do not know yet. Files that VANISHED are counted, never removed —
    an unplugged drive or a renamed folder would otherwise wipe a triage worked
    over days in one silent pass, and the user would have no way to know why.

    SERIALISED like the image lane's walk, and the race it closes is concrete:
    the scrape intake holds the bank's lease for up to several minutes of
    downloads, and opening the bank's workspace fires a forced refresh. Without
    a lease here that concurrent walk inventoried the freshly-moved files FIRST,
    without their provenance (`_insert_sources` never updates a known row), and
    the scrape's own walk then found nothing left to add — provenance silently
    lost and `added` reporting zero. When the bank is busy the walk is simply
    skipped ({'busy': True}); the owner of the lease will run it.

    Returns {'added', 'missing', 'unavailable', 'error'}, or None when the bank is
    unknown. ``force`` is accepted for symmetry with the image lane's cooldown.
    ``source_metadata_by_relpath`` — see ``_insert_sources`` — is threaded through
    only by the scrape intake; it is what keeps THIS walk the single inventory
    path instead of the scrape growing one of its own."""
    key = job_key(bank_id)
    if _bank_lease is None:
        if bank_jobs.running(key):
            return {'added': 0, 'missing': 0, 'unavailable': False,
                    'error': None, 'busy': True}
        try:
            with bank_jobs.mutation_lease(key, 'folder_sync') as lease:
                return refresh_bank(
                    user_id, bank_id, force,
                    source_metadata_by_relpath=source_metadata_by_relpath,
                    _bank_lease=lease)
        except bank_jobs.BankJobBusy:
            return {'added': 0, 'missing': 0, 'unavailable': False,
                    'error': None, 'busy': True}
    bank_jobs.require_reservation(_bank_lease, key)
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    out = {'added': 0, 'missing': 0, 'unavailable': False, 'error': None}
    if not os.path.isdir(bank.source_path):
        out['unavailable'] = True
        out['error'] = 'the source folder is not reachable right now'
        return out
    try:
        rels = _scan_folder(bank.source_path)
    except (OSError, ValueError) as e:
        out['error'] = str(e)
        return out
    known = {s.relpath for s in
             db.session.query(VideoSource.relpath).filter_by(bank_id=bank.id)}
    on_disk = set(rels)
    new = [r for r in rels if r not in known]
    out['missing'] = len(known - on_disk)
    if new:
        _insert_sources(bank.id, bank.source_path, new, source_metadata_by_relpath)
        db.session.commit()
        out['added'] = len(new)
    return out


def delete_bank(user_id, bank_id) -> bool:
    """Throw the bank away, with its sources, clips and thumbnails.

    Children are deleted EXPLICITLY, deepest first, rather than trusted to the
    ondelete=CASCADE in the schema: SQLite only enforces foreign keys when the
    PRAGMA is on, and these models deliberately carry no ORM relationship() to
    cascade through either. What survives is any dataset built out of this bank —
    that is why VideoDatasetClip's provenance is a plain integer."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return False
    VideoClip.query.filter_by(bank_id=bank.id).delete(synchronize_session=False)
    VideoSource.query.filter_by(bank_id=bank.id).delete(synchronize_session=False)
    db.session.flush()
    db.session.delete(bank)
    db.session.commit()
    try:
        from . import trash
        if _bank_dir(bank_id).is_dir():
            trash.dispose(str(_bank_dir(bank_id)), context='video bank thumbnails')
    except Exception as e:                  # noqa: BLE001 — the rows are gone already
        logger.warning('video bank %s: could not dispose thumbnails: %s', bank_id, e)
    return True


def _counts(bank_id) -> dict:
    src = VideoSource.query.filter_by(bank_id=bank_id)
    clips = VideoClip.query.filter_by(bank_id=bank_id)
    return {
        'sources': src.count(),
        'probed': src.filter(VideoSource.probe_state.isnot(None)).count(),
        'unreadable': src.filter_by(probe_state='unreadable').count(),
        # A file the user declared a single take counts as DONE. It is not
        # waiting for anything, and leaving it out would make the workspace's
        # next-step line keep offering a detection pass over a decision.
        'detected': src.filter(VideoSource.detect_state
                               .in_(('ok', SINGLE_SHOT_STATE))).count(),
        'detect_errors': src.filter_by(detect_state='error').count(),
        'clips': clips.count(),
        'pending': clips.filter_by(status='pending').count(),
        'keep': clips.filter_by(status='keep').count(),
        'reject': clips.filter_by(status='reject').count(),
        'promoted': clips.filter(VideoClip.promoted_dataset_id.isnot(None)).count(),
        'thumbs': clips.filter_by(thumb_state='ok').count(),
        # How many shots 🔎 Search can actually find. Counted here rather than
        # derived in the UI because "0 results" and "0 searchable shots" are two
        # different answers and only this number tells them apart.
        'embedded': clips.filter_by(embed_state='ok').count(),
        # Shots that carry a caption at all — generated or hand-written. What the
        # search's readiness line and the promotion's pre-flight both read.
        'captioned': clips.filter(VideoClip.caption.isnot(None),
                                  VideoClip.caption != '').count(),
    }


def _load_pipeline_report(bank: VideoBank):
    """The persisted pass summary, parsed. A corrupt blob is swallowed — a broken
    report must never 500 the whole bank payload."""
    raw = getattr(bank, 'pipeline_report', None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _bank_row(bank: VideoBank) -> dict:
    return {
        'id': bank.id, 'name': bank.name, 'source_path': bank.source_path,
        'created_at': bank.created_at.isoformat() if bank.created_at else None,
        'counts': _counts(bank.id),
        # Whether 🕸 scrape may DOWNLOAD into this bank's folder. Now true for
        # nearly every bank — picking a bank IS the consent — and false only for
        # the one case no consent can fix: a folder that belongs to a dataset,
        # where new files would land inside training material. Surfaced so the
        # picker can offer what would actually be accepted instead of letting the
        # user choose and be refused after the click.
        'scrapable': not _dataset_folder_refusal(bank.source_path),
        # Whether the APP created this folder (a bank born of a scrape) or the
        # user pointed the bank at a folder of their own. Not a permission — it
        # decides whether the picker has something to warn about, because adding
        # files to a folder you assembled yourself is the surprising half.
        'app_folder': is_app_owned_scrape_folder(bank.source_path),
    }


def list_banks(user_id) -> list:
    banks = (VideoBank.query.filter_by(user_id=user_id)
             .order_by(VideoBank.id.desc()).all())
    return [_bank_row(b) for b in banks]


def bank_payload(user_id, bank_id) -> dict | None:
    """The workspace payload: the bank, its counters, its per-file state and the
    live job. One request per poll, like the image lane."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    payload = _bank_row(bank)
    payload['sources'] = sources_payload(user_id, bank_id)
    payload['activity'] = activity(bank_id)
    payload['pipeline_report'] = _load_pipeline_report(bank)
    payload['capability'] = _capability()
    # The saved cuts ride along so the thresholds panel opens on what is actually
    # in force — a panel that opens blank while cuts are active would invite the
    # user to "apply" an accidental clear.
    payload['thresholds'] = metric_thresholds()
    # Which checkpoint writes the captions, so nobody has to wonder what wrote
    # theirs — and so a machine that does not have it yet learns that before
    # pressing the button rather than twenty minutes into a download.
    payload['caption_model'] = caption_model_info()
    # The cut settings in force, so the Find shots panel opens on the truth
    # rather than on a blank field that would read as "no threshold set".
    payload['shot_detect'] = shot_detect_info(bank)
    return payload


def shot_detect_info(bank: VideoBank) -> dict:
    """What the Find shots panel needs: this bank's threshold, the global one it
    would fall back to, and how many of its files could be re-cut instantly.

    `threshold` is the bank's own override or None (inherit) — never the
    resolved value, so the field can show empty instead of a number nobody
    typed. `default` is what None means today."""
    shot = _shot_config()
    cached = (VideoSource.query.filter_by(bank_id=bank.id, probs_state='ok')
              .count())
    return {
        'threshold': bank.shot_threshold,
        'default': shot.threshold_default(),
        'min_shot_seconds': shot.min_shot_seconds_default(),
        'short_shot_policy': shot.short_shot_policy_default(),
        'trim_dissolves': shot.trim_dissolves_default(),
        'cached_sources': cached,
    }


def caption_model_info() -> dict:
    """Which checkpoint will caption, and whether this machine already has it.

    On the bank payload rather than behind its own route: the workspace already
    polls this every two seconds, and a second request for one string would be a
    request per bank per poll for an answer that changes when a config file is
    edited by hand."""
    from . import video_caption
    model = video_caption.configured_model()
    return {'model': model, 'cached': video_caption.model_is_cached(model),
            'is_default': model == video_caption.DEFAULT_MODEL,
            # The prompt style matters MORE than the checkpoint on real footage
            # (see video_caption's measurement), so the picker rides here too.
            'style': video_caption.configured_style(),
            'styles': video_caption.style_choices(),
            # And the vetted checkpoints, `cached` included per entry — the
            # launch window must be able to say "this one downloads first".
            'models': video_caption.model_choices()}


def _capability() -> dict:
    """Decode / detect / encode reported SEPARATELY — they fail independently and
    are fixed differently, and a single "video unavailable" is how a user
    reinstalls the wrong thing.

    `video_text` rides along as a FOURTH field rather than joining the three,
    because it is not one of them: those three decide whether a pass can run at
    all, while this one decides whether 🔳 Safe zone measures burned-in text on
    top of the bands it measures regardless. The workspace uses it for a tooltip,
    never to disable a button — see PASS_REQUIREMENTS in videoCapability.js.
    """
    try:
        from .. import capabilities
        out = dict(capabilities.probe_video())
        text = capabilities.probe_video_text()
        out['video_text'] = bool(text.get('ok'))
        out['video_text_detail'] = text.get('detail')
        return out
    except Exception as e:                  # noqa: BLE001 — never 500 the payload
        logger.warning('video capability probe failed: %s', e)
        return {'ok': False, 'detail': 'could not probe the video extra',
                'decode': False, 'detect': False, 'encode': False,
                'video_text': False, 'video_text_detail': None}


def sources_payload(user_id, bank_id) -> list:
    # Local, like every other reference to this module here: it is the decode
    # seam's neighbour and the convention keeps `av` out of import time.
    from . import video_probe
    rows = (VideoSource.query.filter_by(bank_id=bank_id)
            .order_by(VideoSource.relpath.asc()).all())
    clip_counts = dict(
        db.session.query(VideoClip.source_id, db.func.count(VideoClip.id))
        .filter(VideoClip.bank_id == bank_id).group_by(VideoClip.source_id).all())
    return [{
        'id': s.id, 'relpath': s.relpath, 'file_size': s.file_size,
        'duration_s': s.duration_s, 'fps_native': s.fps_native,
        'width': s.width, 'height': s.height, 'codec': s.codec,
        # How hard this file was squeezed. `bits_per_pixel` is DERIVED here
        # rather than stored — it is a pure function of the four values above it
        # and a stored copy would go stale the day a re-probe corrects the frame
        # rate. All three are shown and none is cut on: the 🩻 defect sweep
        # measures the damage these predict.
        'bit_rate': s.bit_rate, 'profile': s.profile,
        'bits_per_pixel': video_probe.bits_per_pixel(
            s.bit_rate, s.width, s.height, s.fps_native),
        'probe_state': s.probe_state, 'detect_state': s.detect_state,
        'clips': clip_counts.get(s.id, 0),
        # Whether this file can be re-cut INSTANTLY. Read from the column and
        # not from the disk: this runs for every source on every poll, and a
        # stat() per file per two seconds over a bank of hundreds is a cost the
        # answer does not justify.
        'has_probs': s.probs_state == 'ok',
        # This file's own override, or None when it inherits. The UI needs the
        # raw value, not the resolved one, to show an empty field rather than a
        # number the user never typed.
        'shot_threshold': s.shot_threshold,
    } for s in rows]


# --- clips ---------------------------------------------------------------------

def metric_thresholds() -> dict:
    """The cuts currently in force, read from config on every call so a Settings
    save re-sorts the bank on the next poll. All default to None — a cut that has
    not been chosen filters NOTHING, because the published defaults measurably do
    not transfer between corpora (the public motion floor lands at the 7th
    percentile of this machine's own test bank)."""
    section = cfg.get('video_bank') or {}
    # The key list is NOT repeated here — see video_metrics.THRESHOLD_KEYS for
    # why a second copy of it is how a supported cut becomes unreachable.
    return {k: section.get(k) for k in video_metrics.THRESHOLD_KEYS}


def _clip_row(clip: VideoClip, relpaths: dict, thresholds=None) -> dict:
    metrics = json.loads(clip.metrics_json) if clip.metrics_json else None
    duration_s = round(clip.end_s - clip.start_s, 3)
    # Flags are DERIVED here, at read time, from raw scores + the thresholds in
    # force — never stored. Sorted so the payload is deterministic.
    #
    # The call happens even with nothing measured, which it did not use to: every
    # cut needed the metrics pass, so skipping the unmeasured clips was free. The
    # duration cut does not — its input is the bounds — and a bank straight out of
    # detection is exactly where the flash-cut clutter is worst.
    #
    # The WHOLE blob goes to `verdicts`, not the 'ok'-only `measured` view. Two
    # of the verdicts in it are written by other passes — the near-duplicate
    # pass reads the search VECTORS and the watermark pass reads one frame, so
    # both can legitimately have judged a clip the metrics pass never measured.
    # Passing `measured` dropped every one of those flags with no error to see.
    # Nothing else changes: an 'unreadable' summary carries every score as None,
    # and a None score has never produced a flag.
    flags = (sorted(video_metrics.verdicts(metrics, thresholds,
                                           duration_s=duration_s))
             if thresholds is not None else [])
    return {
        'id': clip.id, 'source_id': clip.source_id,
        'relpath': relpaths.get(clip.source_id),
        'start_s': clip.start_s, 'end_s': clip.end_s,
        'duration_s': duration_s,
        'start_frame': clip.start_frame, 'end_frame': clip.end_frame,
        'detector': clip.detector, 'thumb_state': clip.thumb_state,
        'status': clip.status, 'reject_reason': clip.reject_reason,
        # The caption rides on every clip row: the lightbox edits it, the grid
        # shows why a hybrid search moved a shot up, and the promotion dialog
        # counts what has none. One field, three readers, no second request.
        'caption': clip.caption, 'caption_state': clip.caption_state,
        'promoted_dataset_id': clip.promoted_dataset_id,
        'metrics': metrics if metrics and metrics.get('metrics_state') == 'ok' else None,
        'flags': flags,
        # 🎥 The camera labels, derived here beside the flags and for the same
        # reason — from the RAW rates, at read time, never stored — but kept in
        # their own field rather than folded into `flags`. A flag is something
        # the user chose to cut on; a pan is a description, and putting the two
        # in one list would make the grid's amber ⚑ badge announce "your shot
        # pans right" as if it were a defect. Off the WHOLE blob, like the
        # flags, because this pass writes its state whether or not the metrics
        # pass ever ran.
        'camera': video_camera_motion.labels(metrics or {}),
        # Cut or dissolve at each end, when the detector's second head measured
        # it. None throughout for a hand-made cut and for anything detected
        # before that head was kept — no label rather than a guessed one.
        'transition': (json.loads(clip.transition_json)
                       if clip.transition_json else None),
    }


def list_clips(user_id, bank_id, *, status=None, source_id=None, ids=None,
               ids_only=False, offset=0, limit=200) -> dict | None:
    """One page of the clip gallery. ``ids_only`` answers the WHOLE filter as a
    list of ids in one request — what "select all in filter" needs, and it shares
    this function so the two answers can never disagree about what the filter
    holds."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    q = VideoClip.query.filter_by(bank_id=bank_id)
    if status in TRIAGE_STATUSES:
        q = q.filter_by(status=status)
    if source_id:
        q = q.filter_by(source_id=int(source_id))
    if ids is not None:
        q = q.filter(VideoClip.id.in_(ids)) if ids else q.filter(db.false())
    q = q.order_by(VideoClip.source_id.asc(), VideoClip.start_s.asc())
    total = q.count()
    if ids_only:
        return {'ids': [r.id for r in q.all()], 'total': total}
    rows = q.offset(max(0, int(offset))).limit(max(1, int(limit))).all()
    relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                    .filter_by(bank_id=bank_id).all())
    thresholds = metric_thresholds()
    return {'clips': [_clip_row(c, relpaths, thresholds) for c in rows],
            'total': total, 'offset': int(offset), 'limit': int(limit)}


def metrics_dry_run(user_id, bank_id, thresholds) -> dict:
    """Per-rule counts over the bank's stored raw scores — the preview that keeps
    a mis-set threshold from quietly gutting a bank. Pure read; flags nothing."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return {'total_flagged': 0}
    # EVERY clip, not only the measured ones. The population used to be "clips
    # carrying scores", which was right while every cut needed the metrics pass
    # and is wrong for the duration cut: the short shots a user wants counted are
    # the ones they have not bothered measuring. A clip with no usable scores
    # still contributes its bounds, and contributes nothing to the other rules —
    # `verdicts` never flags an absent reading.
    rows = VideoClip.query.filter_by(bank_id=bank_id).all()
    clips = []
    for row in rows:
        # The WHOLE blob, for the same reason `_clip_row` passes the whole blob:
        # two of the cuts read verdicts written by other passes, which can have
        # judged a clip the metrics decode never measured. Filtering on
        # metrics_state here made the preview answer "0 would be flagged" over a
        # bank the grid then flags — the two must describe the same bank.
        # Nothing else moves: an 'unreadable' summary carries every score as
        # None, and `verdicts` never flags an absent reading.
        scores = json.loads(row.metrics_json) if row.metrics_json else None
        clips.append((scores, round(row.end_s - row.start_s, 3)))
    return video_metrics.dry_run(clips, thresholds)


def set_clip_status(user_id, bank_id, ids, status, reason=None) -> dict:
    """Triage. ``ids`` empty or None means EVERY clip of the bank — the same
    "no selection = all of it" convention promotion uses, so the two cannot drift.

    ``reject_reason`` is cleared on anything that is not a reject: a clip flipped
    back to keep must not keep carrying why it was once refused."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        raise ValueError('bank not found')
    if status not in TRIAGE_STATUSES:
        raise ValueError(f'status must be one of {", ".join(TRIAGE_STATUSES)}')
    q = VideoClip.query.filter_by(bank_id=bank_id)
    if ids:
        q = q.filter(VideoClip.id.in_([int(i) for i in ids]))
    rows = q.all()
    for row in rows:
        row.status = status
        row.reject_reason = (str(reason)[:16] if (reason and status == 'reject')
                             else None)
    db.session.commit()
    return {'updated': len(rows), 'counts': _counts(bank_id)}


# --- retouching the cuts -------------------------------------------------------
#
# The detector is good and it is not right. It misses a boundary on a slow
# dissolve, and it happily hands back a shot whose last second is a frozen frame.
# Before these three functions the only gesture available on either was ✕ Reject,
# which throws away the eight good seconds with the bad one. `detector='manual'`
# was in the schema from the first day of this lane precisely to make room for
# them.
#
# THE INVARIANT THEY ALL SHARE: a thumbnail and a set of metrics are measurements
# OF A SPAN. Move the span and they stop describing anything — the thumbnail shows
# a frame the shot no longer contains and the freeze ratio was integrated over
# footage that is now in another clip. So they are FORGOTTEN, not clamped. The
# thumbs pass already clamps a stale sharpest-frame timestamp back inside the
# bounds, and that is a crash guard; it would happily produce a plausible,
# perfectly wrong thumbnail. Forgetting also makes `counts.thumbs` fall, which is
# what makes the workspace's next-step line offer the thumbnails pass again with
# no extra plumbing.

def _clip_of_bank(bank_id, clip_id) -> VideoClip | None:
    """The clip AND the pairing check. The bank id in the URL is not decoration:
    without this, a clip id from bank A would be editable through bank B's path."""
    return VideoClip.query.filter_by(id=int(clip_id), bank_id=bank_id).first()


def _validate_span(src: VideoSource, start_s, end_s) -> tuple:
    """Bounds a cut can actually be made at, or ValueError naming the limit.

    The upper bound is checked against the PROBED duration when there is one.
    ``ffmpeg -ss`` past the end of a file does not fail — it writes a zero-length
    or single-frame clip — so a start beyond the source surfaces at promotion,
    hours after the mistake, as a dataset with silent holes in it."""
    try:
        start = float(start_s)
        end = float(end_s)
    except (TypeError, ValueError):
        raise ValueError('start and end must be numbers') from None
    if start != start or end != end:            # NaN, which every comparison passes
        raise ValueError('start and end must be numbers')
    if start < 0:
        raise ValueError('a shot cannot start before the file does')
    if end - start < MIN_CLIP_S:
        raise ValueError(
            f'a shot must last at least {MIN_CLIP_S}s — this one would last '
            f'{max(0.0, end - start):.2f}s')
    duration = getattr(src, 'duration_s', None)
    if duration and end > duration + 1e-6:
        raise ValueError(f'this file is {duration:.2f}s long — the end must sit '
                         f'inside it')
    return round(start, 3), round(end, 3)


def _forget_measurements(bank_id, clip: VideoClip):
    """Drop everything that described the OLD span, file included.

    The JPEG has to go and not merely be un-stamped: the grid points an <img> at
    the thumb URL, which serves whatever is on disk, so a leftover file would keep
    showing the old frame until the pass ran again — the exact lie this feature
    exists to remove."""
    clip.thumb_state = None
    clip.metrics_json = None
    # The search vectors describe three INSTANTS of the old span. Kept, they would
    # make this shot answer a phrase for something that now belongs to its
    # neighbour, and hand the player a second outside the shot's own bounds — a
    # wrong answer that looks exactly like a right one. Same reasoning as the
    # thumbnail: a measurement of a span nobody has any more is not stale, it is
    # false.
    #
    # The COLUMN is cleared and the vectors are left where they are, deliberately.
    # `video_clip_search.search` requires embed_state == 'ok' before it will read
    # a shot's vectors, so this one line is what makes the old ones unreachable —
    # and rewriting the store here would rewrite the WHOLE bank's .npz (tens of MB
    # on a real bank) inside a trim, which is an interactive gesture. The orphans
    # are pruned by the next embedding pass, where a rewrite is happening anyway.
    clip.embed_state = None
    # The detector's frame indices described the old boundary and nothing cuts from
    # them; keeping them would make a later disagreement unreadable.
    clip.start_frame = None
    clip.end_frame = None
    clip.detector = 'manual'
    try:
        thumb_path(bank_id, clip.id).unlink(missing_ok=True)
    except OSError:         # noqa: BLE001 — a locked thumbnail is not worth a 500
        logger.info('video bank %s: could not remove the thumbnail of clip %s',
                    bank_id, clip.id)


def set_clip_bounds(user_id, bank_id, clip_id, start_s, end_s) -> dict | None:
    """Move one shot's boundaries. None when the bank or the clip is unknown.

    A PROMOTED clip stays editable on purpose. ``VideoDatasetClip`` copies the
    relpath and the bounds at encode time — the dataset's provenance is a SNAPSHOT,
    not a pointer — so re-cutting here cannot retro-edit what is already on disk.
    Refusing the edit would protect nothing and would forbid the second, better cut
    exactly when someone wants it: having just watched the built dataset.
    ``promoted_dataset_id`` is left alone for the same reason it exists — it also
    shields the clip from a re-detect, and a hand-corrected cut is the last thing
    that should be destroyed by one."""
    if get_bank(user_id, bank_id) is None:
        return None
    clip = _clip_of_bank(bank_id, clip_id)
    if clip is None:
        return None
    src = db.session.get(VideoSource, clip.source_id)
    start, end = _validate_span(src, start_s, end_s)
    clip.start_s, clip.end_s = start, end
    _forget_measurements(bank_id, clip)
    db.session.commit()
    return {'clip': _clip_row_for(bank_id, clip), 'counts': _counts(bank_id)}


def split_clip(user_id, bank_id, clip_id, at_s) -> dict | None:
    """Cut one shot in two at ``at_s``. None when the bank or the clip is unknown.

    The NEW half inherits the parent's triage status. Falling back to 'pending' on
    both sides would undo the decision being made: you split a kept shot because
    its tail is bad, and the half you are keeping must not silently leave the keep
    pile — you would have to find it again among hundreds.

    The new half does NOT inherit ``promoted_dataset_id``: those bounds have never
    been encoded anywhere, and claiming otherwise would put a "already in a
    dataset" badge on a span no dataset contains."""
    if get_bank(user_id, bank_id) is None:
        return None
    clip = _clip_of_bank(bank_id, clip_id)
    if clip is None:
        return None
    try:
        at = float(at_s)
    except (TypeError, ValueError):
        raise ValueError('the split point must be a number') from None
    # Both halves are validated as spans in their own right, which is what makes a
    # split flush against a bound (an empty shot) and a split 0.2 s in (a shot no
    # target can ingest) the same refusal, with the same sentence.
    src = db.session.get(VideoSource, clip.source_id)
    _validate_span(src, clip.start_s, at)
    _validate_span(src, at, clip.end_s)
    at = round(at, 3)
    tail = VideoClip(bank_id=bank_id, source_id=clip.source_id,
                     start_s=at, end_s=clip.end_s, detector='manual',
                     status=clip.status, reject_reason=clip.reject_reason)
    db.session.add(tail)
    clip.end_s = at
    _forget_measurements(bank_id, clip)
    db.session.flush()
    _forget_measurements(bank_id, tail)
    db.session.commit()
    return {'clip': _clip_row_for(bank_id, clip),
            'new_clip': _clip_row_for(bank_id, tail),
            'counts': _counts(bank_id)}


def create_clip(user_id, bank_id, source_id, start_s, end_s) -> dict | None:
    """A shot the detector never drew. None when the bank or the source is unknown.

    'pending', because it has never been judged — and it sorts into the gallery by
    its start like any other, so it appears where the user cut it rather than at
    the end of the list."""
    if get_bank(user_id, bank_id) is None:
        return None
    src = VideoSource.query.filter_by(id=int(source_id), bank_id=bank_id).first()
    if src is None:
        return None
    start, end = _validate_span(src, start_s, end_s)
    clip = VideoClip(bank_id=bank_id, source_id=src.id, start_s=start, end_s=end,
                     detector='manual', status='pending')
    db.session.add(clip)
    db.session.commit()
    return {'clip': _clip_row_for(bank_id, clip), 'counts': _counts(bank_id)}


def _clip_row_for(bank_id, clip: VideoClip) -> dict:
    """One clip in the SAME shape the gallery already reads, so a retouched shot
    can be swapped into the list in place instead of forcing a page reload."""
    relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                    .filter_by(bank_id=bank_id).all())
    return _clip_row(clip, relpaths, metric_thresholds())


# --- passes --------------------------------------------------------------------

def _require_free_bank(user_id, bank_id) -> VideoBank:
    bank = get_bank(user_id, bank_id)
    if bank is None:
        raise ValueError('bank not found')
    return bank


def start_probe(app, user_id, bank_id, reprobe=False):
    """Read what each source file IS. Cheap per file, but a bank holds hundreds."""
    _require_free_bank(user_id, bank_id)
    return bank_jobs.start(app, job_key(bank_id), 'probe',
                           _probe_job(bank_id, bool(reprobe)))


def _probe_job(bank_id, reprobe):
    def run(job):
        q = VideoSource.query.filter_by(bank_id=bank_id)
        if not reprobe:
            q = q.filter(VideoSource.probe_state.is_(None))
        rows = q.order_by(VideoSource.id.asc()).all()
        bank = db.session.get(VideoBank, bank_id)
        bank_jobs.progress(job, done=0, total=len(rows), detail='probing')
        ok = bad = 0
        for src in rows:
            if bank_jobs.cancelled(job):
                break
            path = _abs_source_path(bank, src.relpath) if bank else None
            info = (_probe_file(path) if path
                    else {'probe_state': 'unreadable'})
            src.probe_state = info.get('probe_state') or 'unreadable'
            if src.probe_state == 'ok':
                src.duration_s = info.get('duration_s')
                src.fps_native = info.get('fps_native')
                src.width = info.get('width')
                src.height = info.get('height')
                codec = info.get('codec')
                src.codec = str(codec)[:24] if codec else None
                # Both may legitimately be None — a container that does not
                # carry a per-stream bit rate is not a failed probe — so they
                # are written unconditionally rather than guarded, which is what
                # lets a RE-probe clear a value that is no longer true.
                src.bit_rate = info.get('bit_rate')
                profile = info.get('profile')
                src.profile = str(profile)[:32] if profile else None
                ok += 1
            else:
                bad += 1
            if info.get('file_size') is not None:
                src.file_size = info['file_size']
            db.session.commit()
            bank_jobs.bump(job)
        detail = f'done — {ok} readable'
        if bad:
            detail += f', {bad} unreadable'
        bank_jobs.progress(job, detail=detail)
        return {'ok': ok, 'unreadable': bad}
    return run


def start_detect(app, user_id, bank_id, redetect=False):
    """Find the shot boundaries. The expensive pass — minutes per hour of source."""
    _require_free_bank(user_id, bank_id)
    return bank_jobs.start(app, job_key(bank_id), 'detect',
                           _detect_job(bank_id, bool(redetect)))


def _detect_job(bank_id, redetect):
    def run(job):
        q = (VideoSource.query.filter_by(bank_id=bank_id, probe_state='ok')
             # Never a file the user declared a single take, in EITHER mode —
             # see SINGLE_SHOT_STATE. "Re-detect everything" is a bulk gesture
             # and a declaration is not something a bulk gesture may overrule.
             .filter(db.or_(VideoSource.detect_state.is_(None),
                            VideoSource.detect_state != SINGLE_SHOT_STATE)))
        if not redetect:
            q = q.filter(VideoSource.detect_state.is_(None))
        rows = q.order_by(VideoSource.id.asc()).all()
        bank = db.session.get(VideoBank, bank_id)
        bank_jobs.progress(job, done=0, total=len(rows), detail='detecting shots')
        made = failed = cached = 0
        for src in rows:
            if bank_jobs.cancelled(job):
                break
            path = _abs_source_path(bank, src.relpath) if bank else None
            reuse = _reusable_probs(bank_id, src, path) if redetect else None
            if reuse is not None:
                # The expensive half of this pass is DECODING, and it has
                # already been paid for this file. Re-cutting from the stored
                # vector gives byte-identical bounds at the same threshold and
                # takes milliseconds, so "Find shots again" over a bank that has
                # already been through it once is now instant.
                _drop_clips_of(bank_id, src.id, replace_manual=False)
                made += _insert_clips(bank_id, src, reuse)
                cached += 1
                src.detect_state = 'ok'
                db.session.commit()
                bank_jobs.bump(job)
                continue
            try:
                if path is None:
                    raise OSError('source file is outside the bank folder')
                result = _detect_source(path, src.fps_native,
                                        threshold=shot_threshold_for(bank_id, src))
                shots = result.get('clips') or []
            except Exception as e:      # noqa: BLE001 — one bad file, not the pass
                if _is_detector_unavailable(e):
                    # A fact about the INSTALL, not about these files. Stamping
                    # detect_state='error' on all of them would be wrong twice:
                    # it blames the material, and because the pass skips anything
                    # already marked, installing the extra afterwards would fix
                    # nothing until the user found the re-detect checkbox.
                    bank_jobs.fail(job, str(e))
                    db.session.rollback()
                    return {'clips': made, 'failed': failed, 'unavailable': True}
                logger.info('video bank %s: detection failed on a source: %s',
                            bank_id, type(e).__name__)
                src.detect_state = 'error'
                failed += 1
                db.session.commit()
                bank_jobs.bump(job)
                continue
            if redetect:
                _drop_clips_of(bank_id, src.id, replace_manual=False)
            made += _insert_clips(bank_id, src, shots)
            _remember_probs(bank_id, src, result.get('probs'))
            src.detect_state = 'ok'
            db.session.commit()
            bank_jobs.bump(job)
        detail = f'done — {made} clips found'
        if cached:
            detail += f', {cached} re-cut from cache'
        if failed:
            detail += f', {failed} files failed detection'
        bank_jobs.progress(job, detail=detail)
        return {'clips': made, 'failed': failed, 'from_cache': cached}
    return run


def _reusable_probs(bank_id, src: VideoSource, path):
    """The clips a cached vector would give for this file — or None to decode it.

    WHY THIS IS GUARDED BY THE FILE SIZE. A bank points at a LIVE folder: people
    keep dropping files into it, and they also re-export and overwrite them. A
    cache keyed on nothing but the source id would then re-cut the NEW file at
    the OLD file's boundaries and report a clean run — bounds that describe
    footage nobody has any more, which is the one failure this whole lane is
    built to avoid. The size recorded by the probe is a cheap, honest tripwire:
    it does not catch a same-size re-encode, and a genuinely changed file
    virtually never keeps its byte count.

    Anything unresolvable — no cache, no probed rate, no size on record, a size
    that moved — falls through to a real pass, which re-decodes and refills the
    cache. Being wrong in that direction costs time; being wrong in the other
    costs correctness.
    """
    from . import shot_probs
    if not src.fps_native or not src.file_size or not path:
        return None
    try:
        if os.path.getsize(path) != src.file_size:
            return None
    except OSError:
        return None
    probs = shot_probs.load_probs(bank_id, src.id)
    if not probs or not probs.get('single'):
        return None
    return _shot_config().clips_from_probs(
        probs, fps_native=src.fps_native,
        threshold=shot_threshold_for(bank_id, src))


def _insert_clips(bank_id, src: VideoSource, shots) -> int:
    """Persist the detector's bounds AS GIVEN.

    start_s/end_s are copied verbatim because they are canonical; the frame
    indices are stored because they are what the detector actually said, and they
    make a later disagreement debuggable. Nothing ever cuts from them."""
    rows = []
    for shot in shots or []:
        try:
            start_s = float(shot['start_s'])
            end_s = float(shot['end_s'])
        except (KeyError, TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        transition = shot.get('transition')
        rows.append({
            'bank_id': bank_id, 'source_id': src.id,
            'start_s': start_s, 'end_s': end_s,
            'start_frame': shot.get('start_frame'),
            'end_frame': shot.get('end_frame'),
            'detector': (shot.get('detector') or 'transnetv2')[:16],
            'transition_json': (json.dumps(transition)
                                if transition and any(transition.values())
                                else None),
            'status': 'pending',
        })
    for i0 in range(0, len(rows), _INSERT_CHUNK):
        db.session.execute(VideoClip.__table__.insert(), rows[i0:i0 + _INSERT_CHUNK])
    return len(rows)


# --- re-cutting a file without re-running the detector ---------------------------
#
# THE ONE IDEA UNDER ALL OF THIS: the detector's per-frame probabilities are now
# kept (services/shot_probs), so the threshold stopped being a decision baked
# into a GPU pass and became a value that can be argued with. A slider over a
# cached vector is the same gesture DaVinci Resolve offers over its own
# confidence graph, and it is the reason 0.5 could stay the default honestly —
# nobody ever measured it, and now nobody has to live with it either.
#
# WHY A PER-FILE OVERRIDE AND NOT JUST A PER-BANK ONE. The corpus is mixed
# INSIDE one folder: an untouched single take and a tightly edited scene sit
# next to each other, and no bank-level number is right for both. The ladder is
# file, then bank, then global — and NULL at any level means "inherit", never
# zero, because 0.0 is a real threshold that cuts on every frame.

class ShotProbsMissing(RuntimeError):
    """This source has no cached probabilities, so it cannot be re-cut instantly.

    Not an error about the file — it is a fact about when it was detected. The
    caller's answer is to offer a real detection pass, never to report the file
    as broken."""


def _shot_config():
    from . import shot_detect
    return shot_detect


def shot_threshold_for(bank_id, source):
    """The threshold that applies to ONE file: its own, else its bank's, else
    the global default. Clamped on the way out — read on a hot path, so a
    nonsense stored value degrades rather than aborting a pass."""
    from . import shot_boundaries
    bank = db.session.get(VideoBank, int(bank_id))
    src = (source if isinstance(source, VideoSource)
           else db.session.get(VideoSource, int(source)))
    return shot_boundaries.resolve_threshold(
        getattr(src, 'shot_threshold', None),
        getattr(bank, 'shot_threshold', None),
        _shot_config().threshold_default())


def _validated_threshold(value):
    """None (inherit) or a number inside [0, 1]. REFUSED rather than clamped:
    the read path clamps because it must never abort a pass already running,
    but a write has somebody there to be told they typed something wrong."""
    if value is None or value == '':
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError('the threshold must be a number between 0 and 1') from None
    if number != number or not (0.0 <= number <= 1.0):
        raise ValueError('the threshold must be a number between 0 and 1')
    return round(number, 3)


def set_bank_shot_threshold(user_id, bank_id, value) -> dict | None:
    """Set (or clear, with None) the whole bank's threshold. Cuts nothing by
    itself — re-cutting is a separate, explicit gesture, because changing a
    number must not silently rewrite hundreds of rows."""
    bank = get_bank(user_id, bank_id)
    if bank is None:
        return None
    bank.shot_threshold = _validated_threshold(value)
    db.session.commit()
    return {'threshold': bank.shot_threshold}


def set_source_shot_threshold(user_id, bank_id, source_id, value) -> dict | None:
    if get_bank(user_id, bank_id) is None:
        return None
    src = VideoSource.query.filter_by(id=int(source_id), bank_id=bank_id).first()
    if src is None:
        return None
    src.shot_threshold = _validated_threshold(value)
    db.session.commit()
    return {'threshold': src.shot_threshold,
            'effective': shot_threshold_for(bank_id, src)}


def _remember_probs(bank_id, src: VideoSource, probs):
    """Persist one source's probability vectors, or record that it has none.

    Never fatal: a bank on a full disk must still finish its detection pass and
    keep its clips. The only thing lost is the instant re-cut, and the source
    row says so rather than pretending."""
    if not probs or not probs.get('single'):
        src.probs_state = None
        return
    try:
        from . import shot_probs
        shot_probs.save_probs(bank_id, src.id, probs.get('single'),
                              probs.get('all'))
        src.probs_state = 'ok'
    except Exception as error:      # noqa: BLE001 — a cache write never sinks a pass
        logger.warning('video bank %s: could not cache the shot probabilities '
                       'of source %s: %s', bank_id, src.id, error)
        src.probs_state = None


def _load_probs_or_raise(bank_id, source_id):
    from . import shot_probs
    probs = shot_probs.load_probs(bank_id, source_id)
    if not probs or not probs.get('single'):
        raise ShotProbsMissing(
            'this file has no cached shot probabilities — run Find shots on it '
            'once and every later threshold change is instant')
    return probs


def _drop_clips_of(bank_id, source_id, *, replace_manual) -> dict:
    """Delete the clips a re-cut is about to replace, and everything that
    described them.

    PROMOTED CLIPS ARE NEVER TOUCHED, at either level: a dataset already built
    keeps its provenance, and revoking it behind a slider would make the badge
    on those clips a lie.

    HAND-MADE CUTS depend on the gesture, and the asymmetry is the design.
    A bank-wide pass spares them — an afternoon of retouching must not vanish
    behind a checkbox. A per-FILE re-cut replaces them, because it is a
    deliberate action on a file the user picked out by name, and it is the only
    way back from "this file is a single take". The count is reported so the UI
    can say what it is about to do before doing it.

    THE THUMBNAILS GO WITH THE ROWS. A thumbnail is a measurement OF A SPAN; the
    grid points an <img> at a URL that serves whatever is on disk, so a leftover
    file keeps showing a frame no clip contains. The search vectors are left in
    the bank's .npz on purpose (rewriting tens of megabytes inside a click), and
    are unreachable the moment their row is gone — the same trade
    `_forget_measurements` documents for a trim.
    """
    query = (VideoClip.query.filter_by(bank_id=bank_id, source_id=int(source_id))
             .filter(VideoClip.promoted_dataset_id.is_(None)))
    if not replace_manual:
        query = query.filter(db.or_(VideoClip.detector.is_(None),
                                    VideoClip.detector != 'manual'))
    doomed = query.all()
    manual = sum(1 for clip in doomed if clip.detector == 'manual')
    for clip in doomed:
        try:
            thumb_path(bank_id, clip.id).unlink(missing_ok=True)
        except OSError:     # noqa: BLE001 — a locked thumbnail is not worth a 500
            logger.info('video bank %s: could not remove the thumbnail of clip '
                        '%s', bank_id, clip.id)
        db.session.delete(clip)
    db.session.flush()
    return {'removed': len(doomed), 'replaced_manual': manual}


def recut_source(user_id, bank_id, source_id, threshold=None) -> dict | None:
    """Re-cut ONE file from its cached probabilities. No decode, no GPU.

    This is also the way back from "this file is a single take": it replaces
    hand-made cuts on this one file, which a bank-wide re-cut never does. See
    `_drop_clips_of` for why the two levels differ."""
    if get_bank(user_id, bank_id) is None:
        return None
    src = VideoSource.query.filter_by(id=int(source_id), bank_id=bank_id).first()
    if src is None:
        return None
    probs = _load_probs_or_raise(bank_id, src.id)
    thr = (_validated_threshold(threshold) if threshold is not None
           else shot_threshold_for(bank_id, src))
    fps = src.fps_native
    if not fps:
        raise ValueError('this file has not been probed, so its frame rate is '
                         'unknown — run Probe first')
    clips = _shot_config().clips_from_probs(probs, fps_native=fps, threshold=thr)
    dropped = _drop_clips_of(bank_id, src.id, replace_manual=True)
    made = _insert_clips(bank_id, src, clips)
    src.detect_state = 'ok'
    db.session.commit()
    return {'clips': made, 'threshold': thr, 'counts': _counts(bank_id),
            **dropped}


def recut_bank(user_id, bank_id, threshold=None) -> dict | None:
    """Re-cut every source that HAS a cached vector, sparing hand-made cuts.

    Synchronous on purpose, unlike every heavy pass in this lane: there is no
    decode here, only arithmetic over a few hundred KB per file, and putting it
    behind the job queue would make an instant operation look like a slow one
    and would lock the bank against the very next click."""
    if get_bank(user_id, bank_id) is None:
        return None
    rows = (VideoSource.query.filter_by(bank_id=bank_id)
            .order_by(VideoSource.id.asc()).all())
    from . import shot_probs
    done = made = skipped = single = 0
    for src in rows:
        if src.detect_state == SINGLE_SHOT_STATE:
            single += 1
            continue
        probs = shot_probs.load_probs(bank_id, src.id)
        if not probs or not probs.get('single') or not src.fps_native:
            # Detected before the cache existed, or never probed. Counted and
            # reported — never left with its old cuts as if it had been re-cut.
            skipped += 1
            continue
        thr = (_validated_threshold(threshold) if threshold is not None
               else shot_threshold_for(bank_id, src))
        clips = _shot_config().clips_from_probs(probs, fps_native=src.fps_native,
                                                threshold=thr)
        _drop_clips_of(bank_id, src.id, replace_manual=False)
        made += _insert_clips(bank_id, src, clips)
        done += 1
    db.session.commit()
    # `single_shot` is counted apart from `skipped`: one is "this file has no
    # cache and needs a real pass", the other is "you told me not to". Merging
    # them would offer the user a fix for something that is not broken.
    return {'sources': done, 'clips': made, 'skipped': skipped,
            'single_shot': single, 'counts': _counts(bank_id)}


def shot_dry_run(user_id, bank_id, source_id=None, thresholds=None) -> dict | None:
    """"At threshold X you would get N shots" — for one file or the whole bank.

    Reads the cache and writes nothing, exactly like the metrics dry run: the
    point of a preview is to be able to change your mind after seeing it."""
    if get_bank(user_id, bank_id) is None:
        return None
    from . import shot_boundaries, shot_probs
    query = VideoSource.query.filter_by(bank_id=bank_id)
    if source_id is not None:
        query = query.filter_by(id=int(source_id))
    rows = query.order_by(VideoSource.id.asc()).all()
    if not rows:
        return None
    # Which value the ladder marks as "in force". For ONE file that is the
    # file's own resolved threshold; for the whole bank it is the BANK's, with
    # per-file overrides deliberately ignored — reading the first source's
    # would mark a row nothing bank-wide actually uses, and that row is the one
    # every other row's "8 fewer than now" is measured against.
    if source_id is not None:
        current = shot_threshold_for(bank_id, rows[0])
    else:
        bank = db.session.get(VideoBank, int(bank_id))
        current = shot_boundaries.resolve_threshold(
            None, getattr(bank, 'shot_threshold', None),
            _shot_config().threshold_default())
    ladder = ([_validated_threshold(t) for t in thresholds] if thresholds
              else shot_boundaries.suggested_thresholds(current))
    ladder = [t for t in ladder if t is not None]
    totals = {t: 0 for t in ladder}
    answered = skipped = single = 0
    for src in rows:
        if source_id is None and src.detect_state == SINGLE_SHOT_STATE:
            # A bank-wide preview must count what the bank-wide RE-CUT would
            # produce, and that pass walks past a declared single take. Counting
            # it here promised two shots for a file that would keep its one —
            # a preview that does not match the action it previews is worse than
            # no preview. Asked about that file BY NAME the answer is different:
            # the per-file re-cut does apply to it, and is the way back from the
            # declaration.
            single += 1
            continue
        probs = shot_probs.load_probs(bank_id, src.id)
        if not probs or not probs.get('single') or not src.fps_native:
            skipped += 1
            continue
        answered += 1
        for row in _shot_config().sweep_probs(probs, fps_native=src.fps_native,
                                              thresholds=ladder):
            totals[row['threshold']] = totals.get(row['threshold'], 0) + row['shots']
    return {'rows': [{'threshold': t, 'shots': totals.get(t, 0)} for t in ladder],
            'sources': answered, 'skipped': skipped, 'single_shot': single,
            'current': current}


def mark_single_shot(user_id, bank_id, source_id) -> dict | None:
    """"This file is one single take" — replace its clips with ONE, full length.

    NO TOOL STUDIED OFFERS THIS, and a single-take corpus needs it more than any
    slider: the failure mode there is not a missed cut, it is a file quietly
    chopped into six fragments that each train on a third of a gesture. The
    result is stamped `detector='manual'`, so a bank-wide re-cut leaves it
    alone; the dedicated per-file re-cut is the way back.

    Needs a probed duration, and says so rather than guessing: `ffmpeg -ss` past
    the end of a file does not fail, it writes a one-frame clip, and the mistake
    would only surface at promotion."""
    if get_bank(user_id, bank_id) is None:
        return None
    src = VideoSource.query.filter_by(id=int(source_id), bank_id=bank_id).first()
    if src is None:
        return None
    duration = src.duration_s
    if not duration or duration <= 0:
        raise ValueError('this file has not been probed, so its length is '
                         'unknown — run Probe first')
    if duration < MIN_CLIP_S:
        raise ValueError(f'this file lasts {duration:.2f}s, less than the '
                         f'{MIN_CLIP_S}s a clip needs')
    dropped = _drop_clips_of(bank_id, src.id, replace_manual=True)
    clip = VideoClip(bank_id=bank_id, source_id=src.id, start_s=0.0,
                     end_s=round(float(duration), 3), detector='manual',
                     status='pending')
    db.session.add(clip)
    src.detect_state = SINGLE_SHOT_STATE
    db.session.commit()
    return {'clips': 1, 'counts': _counts(bank_id), **dropped}


def start_measure(app, user_id, bank_id, remeasure=False):
    """Wave 2's pass: one decode per clip, every metric out of it. The heavy
    per-clip work lives in video_metrics_scan; this wrapper only gives it the
    same job envelope (busy refusal, progress, cancel) as every other pass."""
    _require_free_bank(user_id, bank_id)
    return bank_jobs.start(app, job_key(bank_id), 'measure',
                           _measure_job(bank_id, bool(remeasure)))


def _measure_job(bank_id, remeasure):
    def run(job):
        from . import video_metrics_scan
        q = (VideoClip.query.filter_by(bank_id=bank_id)
             .join(VideoSource, VideoSource.id == VideoClip.source_id)
             .filter(VideoSource.probe_state == 'ok'))
        if not remeasure:
            q = q.filter(VideoClip.metrics_json.is_(None))
        total = q.count()
        bank_jobs.progress(job, done=0, total=total, detail='measuring clips')
        # Delegate per-clip work but keep cancel/progress here: the scan commits
        # per clip (its resume contract), so cancelling between clips loses
        # nothing and the next run picks up exactly where this one stopped.
        measured = unreadable = 0
        bank = db.session.get(VideoBank, bank_id)
        for clip in q.order_by(VideoClip.id.asc()).all():
            if bank_jobs.cancelled(job):
                break
            r = video_metrics_scan.measure_one(bank, clip)
            if r == 'ok':
                measured += 1
            else:
                unreadable += 1
            bank_jobs.bump(job)
        detail = f'done — {measured} measured'
        if unreadable:
            detail += f', {unreadable} unreadable'
        bank_jobs.progress(job, detail=detail)
        return {'measured': measured, 'unreadable': unreadable}
    return run


def _embed_available():
    """None when this install can embed frames, else the sentence saying why not.

    A named seam rather than an inline import so the refusal can be exercised in
    both directions without a torch install — and so the ONE place that decides
    "can this machine do CLIP" stays visible to a reader of this file."""
    from .clip_image_encoder import unavailable_reason
    return unavailable_reason()


def start_embed(app, user_id, bank_id, reembed=False):
    """🔎 Wave 3's pass: CLIP vectors for a few frames of every shot, so a typed
    word can find the scenes it describes.

    Refused up front — with the Setup sentence, not a generic error — when no
    interpreter here can run CLIP: a 202 followed by a job that dies on an import
    is the same information delivered ten minutes later and harder to read.

    Serialised against training and the vision passes ONLY when it will really
    use the card, exactly like the image lane's ✨ Score. On CPU this is hours of
    work that never wanted the GPU, and holding the exclusive window through it
    would leave the card idle AND unusable."""
    from ..capabilities import bank_scoring_gpu_available
    _require_free_bank(user_id, bank_id)
    reason = _embed_available()
    if reason:
        raise RuntimeError(reason)
    use_gpu = bank_scoring_gpu_available()
    if use_gpu:
        busy = _gpu_busy_reason()
        if busy:
            raise RuntimeError(busy)
    return bank_jobs.start(app, job_key(bank_id), 'embed',
                           _embed_job(bank_id, bool(reembed), use_gpu))


def _gpu_busy_reason():
    """A human reason the card is unavailable right now, or None. The same system
    flags training and the vision window raise, so an embedding pass never races
    a training run — the guarantee the whole app is built on."""
    from ..job_queue import queue_manager
    if queue_manager._get_system_state('training_in_progress'):
        return 'training is running on the GPU — try again once it finishes'
    if queue_manager._get_system_state('vision_in_progress'):
        return 'a vision/GPU pass is already running — try again in a moment'
    return None


def _embed_job(bank_id, reembed, use_gpu):
    def run(job):
        from contextlib import nullcontext

        from ..gpu_window import gpu_exclusive_vision_window
        from . import video_clip_search
        total = video_clip_search.pending_clips(bank_id, reembed).count()
        # Say WHICH device every time: on the CPU this is ~336 ms per frame and
        # three frames per shot, so a progress bar crawling for an hour with no
        # explanation reads as a hang rather than as the price of not having a
        # card configured for ✨ Score.
        bank_jobs.progress(job, done=0, total=total,
                           detail=f'embedding shots ({"GPU" if use_gpu else "CPU"})')
        window = (gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu
                  else nullcontext())
        with window:
            out = video_clip_search.run_embed(
                bank_id, reembed, use_gpu=use_gpu,
                on_clip=lambda: bank_jobs.bump(job),
                should_stop=lambda: bank_jobs.cancelled(job))
        if out.get('aborted'):
            # The pass gave up because the MACHINE was failing, not the footage.
            # Nothing after this runs: the look score and the coherence check
            # both read the vectors this pass did not produce, and a run of
            # "0 shot(s) rated" underneath would bury the one line that matters.
            # It is a detail rather than a raise for the reason the watermark
            # pass gives: everything embedded before the failure is committed and
            # keeps its verdicts.
            bank_jobs.progress(job, detail=f'stopped — {out["aborted"]}')
            return out
        out.update(_rate_the_look(job, bank_id, reembed))
        out.update(_check_coherence(job, bank_id, reembed))
        detail = f'done — {out["embedded"]} shot(s) searchable'
        if out['unreadable']:
            detail += f', {out["unreadable"]} could not be read'
        if out.get('rated'):
            detail += f' — {out["rated"]} rated for look'
        if out.get('unrated'):
            detail += (f' — {out["unrated"]} shot(s) not rated '
                       '(vectors missing from the store)')
        if out.get('aesthetic_error'):
            # Said out loud and NOT as a failure: the embedding run succeeded,
            # and a head that could not be fetched is a different problem with a
            # different fix. Silence here would leave a bank whose look cut flags
            # nothing, with nothing to explain why.
            detail += f' — look score unavailable ({out["aesthetic_error"]})'
        if out.get('coherence_unmeasured'):
            # Same silence `unrated` exists to break, and it is the SAME store
            # that caused it — reported separately anyway, because the two passes
            # can disagree (the look score needs a subprocess this one does not)
            # and one number for both would hide which of them fell over.
            detail += (f' — {out["coherence_unmeasured"]} shot(s) not checked '
                       'for scene changes (vectors missing from the store)')
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def _rate_the_look(job, bank_id, reembed):
    """🎨 The look score, riding the pass that produced the vectors it reads.

    AFTER the GPU window closes, deliberately: this is CPU arithmetic over an
    .npz and holding the card through it would leave it idle AND unavailable.

    Over the WHOLE bank rather than over what this run embedded: a bank embedded
    before the score existed must not need hours of re-decoding to gain a number
    it can get from vectors already on disk, so re-clicking 🔎 Find scenes IS the
    retrofit. ``reembed`` carries through as ``rescore`` — rewritten vectors are
    different vectors, and a stale rating beside them would be a verdict about
    footage that has moved.

    Skipped on cancel: a stopped pass has already kept everything it earned, and
    charging it a torch import on the way out is not what Stop means.
    """
    from . import video_aesthetic
    if bank_jobs.cancelled(job):
        return {}
    if not video_aesthetic.pending_clips(bank_id, bool(reembed)):
        return {}
    bank_jobs.progress(job, detail='rating how each shot looks')
    out = video_aesthetic.run_aesthetic(
        bank_id, rescore=bool(reembed),
        should_stop=lambda: bank_jobs.cancelled(job))
    # `unrated` rides along — run_aesthetic's docstring promises it is "reported
    # rather than folded into a total", and dropping it here was exactly the
    # silence it warns about: a store missing half its vectors read as a clean
    # run, and those shots re-queued (and re-paid a torch import) on every
    # subsequent pass with nothing anywhere saying why.
    return {'rated': out['rated'], 'unrated': out['unrated'],
            'aesthetic_error': out['error']}


def _check_coherence(job, bank_id, reembed):
    """🔗 Does one shot hold ONE scene, riding the pass that produced the vectors.

    LAST, after the look score, and the order is deliberate rather than
    incidental: 🎨 pays a torch import and a possible 13 MB download, this pays a
    few dot products in this very process. Putting the cheap certainty behind the
    expensive uncertainty means a machine with no egress still gets its coherence
    reading, because ``_rate_the_look`` returns its failure as a RESULT rather
    than raising.

    Over the WHOLE bank rather than over what this run embedded, for the same
    reason the look score is: a bank embedded before this shipped must not need
    hours of re-decoding to gain a number it can get from vectors already on
    disk, so re-clicking 🔎 Find scenes IS the retrofit — and here that retrofit
    costs nothing at all, not even an interpreter start. ``reembed`` carries
    through as ``recheck``: rewritten vectors are different vectors, and a stale
    reading beside them would be a verdict about footage that has moved.

    Skipped on cancel, like the look score — a stopped pass has already kept
    everything it earned.
    """
    from . import video_temporal_coherence
    if bank_jobs.cancelled(job):
        return {}
    if not video_temporal_coherence.pending_clips(bank_id, bool(reembed)):
        return {}
    bank_jobs.progress(job, detail='checking each shot holds one scene')
    out = video_temporal_coherence.run_coherence(
        bank_id, recheck=bool(reembed),
        should_stop=lambda: bank_jobs.cancelled(job))
    # Namespaced rather than merged into the embed run's own keys: `measured`
    # already means something else in half this module's results, and a caller
    # reading one dictionary must not have to know which pass wrote which word.
    return {'coherence_measured': out['measured'],
            'coherence_unmeasured': out['unmeasured']}


def start_dedup(app, user_id, bank_id, threshold=None):
    """✂ Group near-identical shots. Refused up front — with the sentence that
    says what to do — when the bank has no vectors: this pass reads what 🔎 Search
    cached and produces nothing without it, and a 202 followed by "0 groups"
    would read as "no duplicates in this bank".

    No GPU window and no capability check: it re-reads an .npz and does dot
    products. That is the whole point of building it on the embeddings that
    already exist."""
    from .video_clip_search import load_embeddings
    _require_free_bank(user_id, bank_id)
    if not load_embeddings(bank_id):
        raise ValueError('run 🔎 Find scenes first — near-duplicates reuse the '
                         'frame vectors it caches')
    return bank_jobs.start(app, job_key(bank_id), 'dedup',
                           _dedup_job(bank_id, threshold))


def _dedup_job(bank_id, threshold):
    def run(job):
        from . import video_clip_dedup
        bank_jobs.progress(job, done=0, total=0, detail='comparing shots')
        out = video_clip_dedup.run_dedup(
            bank_id, threshold, should_stop=lambda: bank_jobs.cancelled(job))
        detail = f'done — {out["groups"]} near-duplicate group(s)'
        if out['flagged']:
            detail += f', {out["flagged"]} shot(s) flagged'
        # Said out loud rather than folded into the total: "no duplicates" and
        # "most of the bank was never embedded" are the same sentence otherwise.
        if out['unevaluated']:
            detail += f' — {out["unevaluated"]} shot(s) had no vectors to compare'
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def _watermark_available():
    """None when this install can look for watermarks, else the sentence saying
    why not."""
    from .video_watermark import unavailable_reason
    return unavailable_reason()


def start_watermark(app, user_id, bank_id, rescan=False):
    """🔖 Look for a watermark on each shot's ambassador frame.

    Refused up front when the detector is not installed, for the same reason the
    embed pass refuses on a missing CLIP environment: a 202 followed by a job
    that dies on an import is the same information delivered ten minutes later
    and harder to read.

    Serialised against training and the vision passes when it will really use the
    card — this is a torch model, and the guarantee the whole app is built on is
    that no pass races a training run."""
    from ..capabilities import watermark_detect_gpu_available
    _require_free_bank(user_id, bank_id)
    reason = _watermark_available()
    if reason:
        raise RuntimeError(reason)
    use_gpu = watermark_detect_gpu_available()
    if use_gpu:
        busy = _gpu_busy_reason()
        if busy:
            raise RuntimeError(busy)
    return bank_jobs.start(app, job_key(bank_id), 'watermark',
                           _watermark_job(bank_id, bool(rescan), use_gpu))


def _watermark_job(bank_id, rescan, use_gpu):
    def run(job):
        from contextlib import nullcontext

        from ..gpu_window import gpu_exclusive_vision_window
        from . import video_watermark
        total = len(video_watermark.pending_clips(bank_id, rescan))
        bank_jobs.progress(job, done=0, total=total,
                           detail=f'looking for watermarks ({"GPU" if use_gpu else "CPU"})')
        window = (gpu_exclusive_vision_window(flag_ttl=1800) if use_gpu
                  else nullcontext())
        with window:
            out = video_watermark.run_watermark(
                bank_id, rescan,
                on_clip=lambda: bank_jobs.bump(job),
                should_stop=lambda: bank_jobs.cancelled(job))
        detail = f'done — {out["detected"]} shot(s) carry a mark'
        if out['unreadable']:
            detail += f', {out["unreadable"]} could not be judged'
        # The detector dying is a RESULT, and the sentence has to survive to the
        # UI: everything it judged before that is kept, and "0 detected" on its
        # own would read as a clean bank.
        if out['error']:
            detail = f'stopped — {out["error"]} ({out["scanned"]} shot(s) judged)'
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def start_safe_zone(app, user_id, bank_id, rescan=False):
    """🔳 Measure the container and the burned-in text of each shot.

    NEVER refused for a missing OCR engine, and that is the one thing that makes
    it different from every other capability-bearing pass here. 🔖 Watermarks and
    🗣 Describe have nothing to offer without their model, so they refuse up
    front rather than deliver a 202 and a job that dies on an import. This pass
    has two halves and only one of them needs an install: with no RapidOCR it
    still measures letterbox and pillarbox bands on every shot, records
    `safe_zone_state: 'bars_only'` so nothing can mistake that for "no text
    found", and reports the missing extra in the job's own detail line.

    No GPU window either. Decoding is PyAV and the OCR is CPU onnxruntime by
    construction, so this can run while a training run owns the card — which is
    most of the point of building the text half on onnxruntime rather than on
    the torch detector that was already here.
    """
    _require_free_bank(user_id, bank_id)
    return bank_jobs.start(app, job_key(bank_id), 'safezone',
                           _safe_zone_job(bank_id, bool(rescan)))


def _safe_zone_job(bank_id, rescan):
    def run(job):
        from . import video_safe_zone
        total = len(video_safe_zone.pending_clips(bank_id, rescan))
        bank_jobs.progress(job, done=0, total=total,
                           detail='measuring the safe zone')

        def on_text_progress(done, frames):
            # Called from the reader thread of the OCR child — bank_jobs is
            # lock-guarded, which is why this is allowed to touch it. A chunk is
            # 40 shots and over a minute of OCR, so without this the bar stands
            # still long enough to read as a hang.
            bank_jobs.progress(job, detail=f'reading burned-in text '
                                           f'({done}/{frames} frames)')

        out = video_safe_zone.run_safe_zone(
            bank_id, rescan,
            on_clip=lambda: bank_jobs.bump(job),
            should_stop=lambda: bank_jobs.cancelled(job),
            on_text_progress=on_text_progress)
        detail = f'done — {out["measured"]} shot(s) measured'
        if out['letterboxed']:
            detail += f', {out["letterboxed"]} with bands'
        if out['unreadable']:
            detail += f', {out["unreadable"]} could not be read'
        if out['error']:
            # Said out loud and NOT as a failure: the bands ARE measured, on
            # every shot this run touched. Silence here would leave a bank whose
            # text cut flags nothing and nothing anywhere saying why.
            detail += f' — {out["error"]}'
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def start_defects(app, user_id, bank_id, rescan=False):
    """🩻 Sweep each source file for duplicated frames, blocks and soft edges.

    ITS OWN BUTTON, and not a phase of Measure, which is the obvious-looking
    place for it. Three reasons, in order of how much they cost to get wrong:

      * Different UNIT. Measure decodes one clip at a time through PyAV; this
        decodes one FILE at a time through ffmpeg, because all three defects are
        properties of the encode and the macroblock grid does not change at a
        cut. Folded in, it would either decode each file once per shot — losing
        the entire argument — or turn one progress bar into a count of two
        different things.
      * Different DEPENDENCY. Measure needs `av` and runs perfectly on an
        install with no ffmpeg, which the lane explicitly supports ("with no
        encoder you can scan, detect and triage"). Folding this in would either
        make the quality pass refuse without a binary it has never needed, or
        add a silent half that skips — and a pass that quietly does less than
        its name is the failure this codebase keeps a whole vocabulary of states
        to avoid.
      * Different COST, and it is not small. Measured on 1080p25: ~9 s per
        minute of source, which on a four-hour bank is a little over half an
        hour of CPU on top of what Measure already costs. Bundling it would
        multiply a button people press often, with no way to decline.

    🔳 Safe zone's docstring already wrote the test this passes: a pass earns its
    own button when it CONSUMES nothing, and this one consumes nothing — a file
    is sweepable the moment it has probed and been cut. 🎨 Look and ✂ Duplicates
    ride other passes because they read what those passes cached; there is no
    such order to protect here.

    Refused up front, with the Setup sentence, when ffmpeg is not usable — a 202
    followed by a job that dies on a missing binary is the same news delivered
    later and harder to read. NO GPU window: ffmpeg filters run on the CPU, so
    this can sweep while a training run owns the card.
    """
    from .video_defect_sweep import unavailable_reason
    _require_free_bank(user_id, bank_id)
    reason = unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    return bank_jobs.start(app, job_key(bank_id), 'defects',
                           _defects_job(bank_id, bool(rescan)))


def _defects_job(bank_id, rescan):
    def run(job):
        from . import video_defect_sweep
        pending = video_defect_sweep.pending_sources(bank_id, rescan)
        # Progress counts CLIPS while the work advances one FILE at a time, so
        # the bar can stand still for a whole file. The detail line names the
        # file being swept for exactly that reason — a minute of silence on a
        # long rush reads as a hang otherwise.
        total = sum(len(clips) for _src, clips in pending)
        bank_jobs.progress(job, done=0, total=total, detail='sweeping for defects')
        out = video_defect_sweep.run_defects(
            bank_id, rescan,
            on_clip=lambda: bank_jobs.bump(job),
            on_file=lambda relpath: bank_jobs.progress(
                job, detail=f'sweeping {os.path.basename(relpath)}'),
            should_stop=lambda: bank_jobs.cancelled(job))
        detail = (f'done — {out["measured"]} shot(s) swept '
                  f'across {out["files"]} file(s)')
        if out['unreadable']:
            detail += f', {out["unreadable"]} with no frames in range'
        if out['error']:
            # Said out loud and NOT as a failure: every file swept before it is
            # real and kept. Silence here leaves a bank whose defect cuts flag
            # nothing and nothing anywhere saying which file was skipped.
            detail += f' — last problem: {out["error"]}'
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def start_ai_check(app, user_id, bank_id, recheck=False):
    """🤖 Measure how erratically each shot moves, and flag the too-smooth ones.

    Refused up front with the Setup sentence when no interpreter here can run
    the encoder — a 202 followed by a job that dies on an import is the same
    news, twenty minutes later and harder to read. Same refusal as 🔖 Watermarks
    and 🗣 Describe, and the same environment as 🎨 Look: the ✨ Score
    interpreter, which already carries torch and transformers.

    ITS OWN BUTTON, and the two reasons are the two this lane already uses.
    Different SAMPLING, which is the deciding one: this needs sixteen CONTIGUOUS
    frames at 8 fps in colour at 224 px, and not one of the four decodes in this
    lane produces anything like it — 🔎 embeds three frames spread across the
    whole shot, 🗣 samples eight across the span, 🔳 takes three at 768 px, and
    Measure reads the clip in greyscale at 160 px wide. A temporal statistic
    over a spread sample would measure the gaps between moments instead of the
    movement inside one, so there is nothing here to ride. And it CONSUMES
    nothing — a shot is checkable the moment it has been cut — which is 🔳 Safe
    zone's own test for a pass that earns a button rather than a queue position.

    NO GPU WINDOW, unlike 🔖 Watermarks which takes one. That pass is a few
    seconds; this one is tens of minutes over a bank, and holding the card that
    long — unloading ComfyUI, blocking a training start — for an advisory flag
    is the wrong trade. Running on the CPU is what lets it check a bank while a
    training owns the card.
    """
    from .video_ai_check import unavailable_reason
    _require_free_bank(user_id, bank_id)
    reason = unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    return bank_jobs.start(app, job_key(bank_id), 'aicheck',
                           _ai_check_job(bank_id, bool(recheck)))


def _ai_check_job(bank_id, recheck):
    def run(job):
        from . import video_ai_check
        total = len(video_ai_check.pending_clips(bank_id, recheck))
        # The download warning rides in the detail the user is already watching,
        # before the first shot: the encoder is a first-run download of several
        # hundred megabytes, and a bar sitting at 0/900 while it arrives is
        # indistinguishable from a hang.
        notice = video_ai_check.model_download_notice()
        bank_jobs.progress(job, done=0, total=total,
                           detail=notice or 'checking how each shot moves')
        out = video_ai_check.run_ai_check(
            bank_id, recheck,
            on_clip=lambda: bank_jobs.bump(job),
            should_stop=lambda: bank_jobs.cancelled(job))
        detail = f'done — {out["measured"]} shot(s) checked'
        if out['too_short']:
            # Named rather than folded into a total: these shots are not a
            # failure and re-running will never fix them — the window needs
            # about two and a half seconds and their cut is shorter than that.
            detail += (f', {out["too_short"]} too short for the window '
                       f'(under {video_ai_check.min_duration_s():.2f} s)')
        if out['unreadable']:
            detail += f', {out["unreadable"]} could not be read'
        if out['error']:
            # Said out loud and NOT as a failure: every shot checked before it
            # is real and kept. Silence here leaves a bank whose cut flags
            # nothing and nothing anywhere saying why.
            detail = f'stopped — {out["error"]} ({out["measured"]} shot(s) checked)'
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def start_camera(app, user_id, bank_id, rescan=False):
    """🎥 Read how the camera moved in every shot — pan, zoom, roll, handheld.

    Refused up front with the Setup sentence when the decode extra is missing,
    like 🔍 Measure and 🤖 AI check: this pass cannot degrade, every number it
    produces comes out of frames it decoded itself.

    ITS OWN BUTTON, and this one had a real candidate to ride so the argument is
    worth stating. 🩻 Defects already runs ONE ffmpeg pass per source file and
    `vidstabdetect` would bolt onto its filter chain for a fifth of a second —
    tempting, and wrong twice over. The unit is wrong: defects are properties of
    the FILE (a macroblock grid does not change at a cut) while camera motion is
    a property of the SHOT, so the readings would have to be cut back apart
    afterwards. And the tool is wrong: vidstab writes local motion vectors at
    whole-pixel precision with no rotation, no scale and no inlier ratio, which
    is three of the four things this pass reports plus the guard that keeps it
    honest — video_camera_motion's docstring has the measurements.

    Nothing else in the lane decodes what this needs either: it wants EVERY
    frame of a shot at 384 px, and the four existing decodes sample three, eight,
    three and 160-px-wide-everything respectively. A trajectory with gaps is not
    a smaller sample, it is a series whose steps mean different things.

    NO GPU WINDOW. It is OpenCV on the CPU at 0.07 s per second of source, so a
    bank can be read while a training owns the card — the same property 🤖 AI
    check chose deliberately, here for free.
    """
    from .video_camera_motion import unavailable_reason
    _require_free_bank(user_id, bank_id)
    reason = unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    return bank_jobs.start(app, job_key(bank_id), 'camera',
                           _camera_job(bank_id, bool(rescan)))


def _camera_job(bank_id, rescan):
    def run(job):
        from . import video_camera_motion
        total = len(video_camera_motion.pending_clips(bank_id, rescan))
        bank_jobs.progress(job, done=0, total=total,
                           detail='reading how the camera moved')
        out = video_camera_motion.run_camera_motion(
            bank_id, rescan,
            on_clip=lambda: bank_jobs.bump(job),
            should_stop=lambda: bank_jobs.cancelled(job))
        detail = f'done — {out["measured"]} shot(s) read'
        if out['too_short']:
            # Named rather than folded into a total: re-running will never fix
            # these. A trajectory needs a few frames to have a rate at all, and
            # their cut is shorter than that.
            detail += f', {out["too_short"]} too short to have a trajectory'
        if out['unreadable']:
            detail += f', {out["unreadable"]} could not be read'
        if out['error']:
            # The LAST per-clip failure, said out loud and not as a failure of
            # the run: every shot read before it is real and kept.
            detail += f' (last error: {out["error"]})'
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def _caption_available():
    """None when this install can caption, else the sentence saying why not."""
    from .video_caption_worker import unavailable_reason
    return unavailable_reason()


def start_caption(app, user_id, bank_id, recaption=False, include_edited=False,
                  style=None, model=None):
    """🗣 Wave 5's pass: what HAPPENS in each shot, in prose.

    The caption is the text the hybrid search matches AND the training prompt the
    promotion writes into each `.txt`. Refused up front with the Setup sentence
    when no interpreter here can run the model — a 202 followed by a job that
    dies on an import is the same news, ten minutes later and harder to read.

    Unlike the embedding pass this one is worth the GPU on any machine that has
    one (a 4B VLM on a CPU is minutes per shot), so it takes the exclusive window
    whenever the card is usable — and is refused outright while a training run
    owns it, rather than competing with it."""
    from ..capabilities import bank_scoring_gpu_available
    _require_free_bank(user_id, bank_id)
    reason = _caption_available()
    if reason:
        raise RuntimeError(reason)
    use_gpu = bank_scoring_gpu_available()
    if use_gpu:
        busy = _gpu_busy_reason()
        if busy:
            raise RuntimeError(busy)
    return bank_jobs.start(app, job_key(bank_id), 'caption',
                           _caption_job(bank_id, bool(recaption),
                                        bool(include_edited), use_gpu, style,
                                        model))


def _caption_job(bank_id, recaption, include_edited, use_gpu, style=None,
                 model_choice=None):
    def run(job):
        from contextlib import nullcontext

        from ..gpu_window import gpu_exclusive_vision_window
        from . import video_caption
        total = video_caption.pending_clips(bank_id, recaption,
                                            include_edited).count()
        # Both per-run choices resolve the same way: an explicit legal pick, else
        # the config default — captioning ONE bank plainly (or on the 8B) must
        # not silently re-point every other bank.
        model = video_caption.resolve_model(model_choice)
        chosen_style = (style if style in video_caption.CAPTION_STYLES
                        else video_caption.configured_style())
        # WHICH model, in the line the user is already watching: two checkpoints
        # do not write comparable captions, so "captioning shots" alone leaves a
        # bank nobody can reason about after the setting changes.
        detail = (f'captioning shots with {model} / {chosen_style} prompt '
                  f'({"GPU" if use_gpu else "CPU"})')
        # And whether it is even here yet. The download is allowed — blocking it
        # would ship a model setting that cannot point anywhere new — but never
        # in silence: a pass sitting at 0/470 while gigabytes cross someone's
        # connection is indistinguishable from a hang.
        notice = video_caption.download_notice(model)
        if notice:
            detail = f'{detail} — {notice}'
        bank_jobs.progress(job, done=0, total=total, detail=detail)
        window = (gpu_exclusive_vision_window(flag_ttl=3600) if use_gpu
                  else nullcontext())
        with window:
            out = video_caption.run_captions(
                bank_id, recaption, include_edited=include_edited,
                use_gpu=use_gpu, model=model, style=chosen_style,
                on_clip=lambda: bank_jobs.bump(job),
                should_stop=lambda: bank_jobs.cancelled(job))
        detail = (f'done — {out["captioned"]} shot(s) captioned by '
                  f'{out["model"]} ({out["style"]} prompt)')
        if out['failed']:
            detail += f', {out["failed"]} failed'
        bank_jobs.progress(job, detail=detail)
        return out
    return run


def start_thumbs(app, user_id, bank_id, rethumb=False):
    """One frame per shot, taken from the shot's MIDDLE — a boundary is where a cut
    just happened, so the opening frames are disproportionately dissolves and black."""
    _require_free_bank(user_id, bank_id)
    return bank_jobs.start(app, job_key(bank_id), 'thumbs',
                           _thumbs_job(bank_id, bool(rethumb)))


def _thumbs_job(bank_id, rethumb):
    def run(job):
        from . import video_probe
        q = VideoClip.query.filter_by(bank_id=bank_id)
        if not rethumb:
            q = q.filter(VideoClip.thumb_state.is_(None))
        rows = q.order_by(VideoClip.id.asc()).all()
        bank = db.session.get(VideoBank, bank_id)
        relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                        .filter_by(bank_id=bank_id).all())
        bank_jobs.progress(job, done=0, total=len(rows), detail='making thumbnails')
        ok = 0
        for clip in rows:
            if bank_jobs.cancelled(job):
                break
            path = _abs_source_path(bank, relpaths.get(clip.source_id) or '') \
                if bank else None
            done = False
            if path:
                # The middle of the shot is a GUESS made before any frame was
                # measured; the metrics scan's sharpest frame is a MEASUREMENT,
                # free once every frame had been read anyway — and it wins.
                # Clips the scan has not reached keep the guess. Clamped inside
                # the clip in case bounds were re-cut after measuring.
                ts = video_probe.thumbnail_timestamp(clip.start_s, clip.end_s)
                if clip.metrics_json:
                    m = json.loads(clip.metrics_json)
                    sharpest = m.get('sharpest_frame_s')
                    if (m.get('metrics_state') == 'ok' and sharpest is not None
                            and clip.start_s <= sharpest <= clip.end_s):
                        ts = sharpest
                done = _write_thumbnail(path, ts, str(thumb_path(bank_id, clip.id)))
            clip.thumb_state = 'ok' if done else 'error'
            ok += 1 if done else 0
            bank_jobs.bump(job)
        db.session.commit()
        bank_jobs.progress(job, detail=f'done — {ok}/{len(rows)} thumbnails')
        return {'thumbs': ok, 'total': len(rows)}
    return run


def _sanitize_steps(steps):
    if not steps:
        return list(PIPELINE_STEPS)
    wanted = {s for s in steps if s in PIPELINE_STEPS}
    return [s for s in PIPELINE_STEPS if s in wanted]      # canonical order


def start_pipeline(app, user_id, bank_id, steps=None):
    """Probe → detect → thumbnails, chained, with a report that survives the night.

    The passes are also individually reachable, but chaining them is what a user
    actually wants on a fresh bank: each one's input is the previous one's output,
    and running them by hand in the wrong order finds nothing to do and says so in
    a way that reads like a bug."""
    _require_free_bank(user_id, bank_id)
    wanted = _sanitize_steps(steps)
    if not wanted:
        raise ValueError('no pipeline steps selected')
    return bank_jobs.start(app, job_key(bank_id), 'pipeline',
                           _pipeline_job(user_id, bank_id, wanted))


_STEP_RUNNERS = {
    'probe': lambda bank_id: _probe_job(bank_id, False),
    'detect': lambda bank_id: _detect_job(bank_id, False),
    'thumbs': lambda bank_id: _thumbs_job(bank_id, False),
}


def _pipeline_job(user_id, bank_id, steps):
    def run(job):
        import time as _time
        results = []
        pipe = {'steps': list(steps), 'total_steps': len(steps), 'index': 0,
                'current': steps[0], 'results': results}

        def _sync(current=None, index=None):
            if index is not None:
                pipe['index'] = index
            if current is not None:
                pipe['current'] = current
            pipe['results'] = list(results)
            bank_jobs.set_pipeline(job, pipe)

        _sync()
        for i, step in enumerate(steps):
            if bank_jobs.cancelled(job):
                break
            _sync(current=step, index=i)
            entry = {'step': step, 'status': 'done', 'reason': None, 'counts': {}}
            try:
                out = _STEP_RUNNERS[step](bank_id)(job)
                entry['counts'] = out or {}
            except Exception as e:      # noqa: BLE001 — one bad pass never sinks the rest
                entry['status'] = 'error'
                entry['reason'] = f'{type(e).__name__}: {e}'
                db.session.rollback()
            results.append(entry)
            _sync()

        cancelled = bank_jobs.cancelled(job)
        reached = {e['step'] for e in results}
        for step in steps:
            if step not in reached:
                results.append({
                    'step': step,
                    'status': 'cancelled' if cancelled else 'skipped',
                    'reason': 'cancelled before it ran' if cancelled
                    else 'not reached', 'counts': {}})
        _sync()

        report = {'started_at': job.get('started_at'), 'finished_at': _time.time(),
                  'cancelled': cancelled, 'requested_steps': list(steps),
                  'steps': results, 'counts': _counts(bank_id)}
        bank = db.session.get(VideoBank, bank_id)
        if bank is not None:
            bank.pipeline_report = json.dumps(report)
            db.session.commit()
        done_n = sum(1 for e in results if e['status'] == 'done')
        tail = f'done — {done_n}/{len(steps)} steps ran'
        if cancelled:
            tail = f'cancelled — {done_n}/{len(steps)} steps ran'
        bank_jobs.progress(job, detail=tail)
    return run


# --- promotion: the ONE place media is written --------------------------------

def resolve_frames(profile_key, frames):
    """The frame count this export will use, or a ValueError naming a legal one.

    We REFUSE an illegal count rather than snapping to the nearest silently. The
    catalogue can snap, and a UI offering `frame_choices` never produces an illegal
    value, so a request that carries one came from somewhere that believed it —
    and every trainer downstream would accept it and quietly floor it in latent
    space. A refusal that names the nearest legal count is actionable; a silent
    correction produces a dataset that is not the one that was asked for."""
    profile = video_targets.get(profile_key)
    if profile is None:
        raise ValueError(f'unknown target profile: {profile_key}')
    if frames in (None, ''):
        frames = profile['frame_default']
        if not frames:
            raise ValueError(
                f'{profile["label"]} declares no default clip length — pass an '
                'explicit frame count')
    try:
        frames = int(frames)
    except (TypeError, ValueError):
        raise ValueError('frames must be a whole number of frames') from None
    if not video_targets.is_legal_frames(profile_key, frames):
        near = video_targets.snap_frames(profile_key, frames)
        raise ValueError(
            f'{profile["label"]} cannot ingest a {frames}-frame clip — the nearest '
            f'length it accepts is {near}')
    return frames


def resolve_size(profile_key, size):
    """(width, height) or None for "keep the source's size". ValueError off-grid.

    A STEP, not a whitelist: the official size lists are inference-CLI asserts and
    enforcing them would refuse perfectly trainable data. What is real is the
    divisibility the VAE and the patch size impose together."""
    if not size:
        return None
    try:
        width, height = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        raise ValueError('size must be a width and a height') from None
    if not video_targets.validate_resolution(profile_key, width, height):
        profile = video_targets.get(profile_key) or {}
        step = profile.get('size_multiple')
        raise ValueError(
            f'{width}x{height} is not a size {profile.get("label", profile_key)} '
            f'can train at — both sides must be multiples of {step}')
    return (width, height)


# The largest edge trim a promotion will accept. Not a judgement about how much
# of a shot is transition — 0.25 s is the researched figure and this is twenty
# times it — but a guard against a typo (2.5 for 0.25) silently emptying a
# dataset, since every clip it removes is removed for a reason the user set.
MAX_EDGE_INSET_S = 5.0


def _resolve_max_per_source(value):
    """The per-source cap, validated. None means no cap.

    Named refusals rather than an `int()` that escapes as a traceback: the route
    turns any ValueError into the 400 the user reads, and "invalid literal for
    int() with base 10: 'lots'" is not a sentence about clips per source.

    2.5 is refused rather than rounded. "Two and a half clips per source" is not
    a precision question, it is a request nobody can mean — and silently taking
    2 would apply a cap the user never chose to a dataset they cannot re-derive.
    """
    if value is None or value == '':
        return None
    if isinstance(value, bool):              # True is 1 to int(), and means nothing here
        raise ValueError('max clips per source must be a whole number of clips')
    try:
        cap = float(value)
    except (TypeError, ValueError):
        raise ValueError('max clips per source must be a whole number of clips')             from None
    if cap != cap or cap != int(cap):        # NaN, or 2.5
        raise ValueError('max clips per source must be a whole number of clips')
    cap = int(cap)
    if cap < 1:
        raise ValueError('max clips per source must be at least 1 — leave it '
                         'empty for no cap')
    return cap


def _resolve_edge_inset(value):
    """The trim to apply at each bound, validated. Refuses rather than clamps:
    a negative inset would EXTEND every clip past its own bounds into the
    neighbouring shot — the exact frames the detector decided did not belong to
    it — and silently honouring "-0.1" as 0 would hide the mistake."""
    if value is None or value == '':
        return 0.0
    try:
        inset = float(value)
    except (TypeError, ValueError):
        raise ValueError('edge inset must be a number of seconds') from None
    if inset != inset or inset < 0:                  # NaN or negative
        raise ValueError('edge inset cannot be negative — it would extend every '
                         'clip into the shot next to it')
    if inset > MAX_EDGE_INSET_S:
        raise ValueError(f'edge inset is capped at {MAX_EDGE_INSET_S:g}s — '
                         f'{inset:g}s would remove more than any transition is')
    return inset


def start_promote(app, user_id, bank_id, *, ids=None, name, target_profile,
                  frames=None, size=None, max_per_source=None,
                  edge_inset_s=None, trigger_word=None):
    """Encode the KEPT clips into a new video dataset.

    Everything that can be refused is refused HERE, synchronously, before a single
    row or folder is created: an unknown profile, an illegal frame count, an
    off-grid size, an empty selection, a missing ffmpeg. A background job that
    fails on its first item leaves a dataset the user then has to clean up.

    ``ids`` empty/None = every KEPT clip of the bank. Returns the dataset's
    identity so the caller can navigate straight to it."""
    bank = _require_free_bank(user_id, bank_id)
    name = (name or '').strip()
    if not name:
        raise ValueError('name is required')
    frames = resolve_frames(target_profile, frames)
    size = resolve_size(target_profile, size)
    inset = _resolve_edge_inset(edge_inset_s)
    profile = video_targets.get(target_profile)

    q = VideoClip.query.filter_by(bank_id=bank_id, status='keep')
    if ids:
        q = q.filter(VideoClip.id.in_([int(i) for i in ids]))
    rows = q.order_by(VideoClip.source_id.asc(), VideoClip.start_s.asc()).all()
    per_cap = _resolve_max_per_source(max_per_source)
    if per_cap is not None:
        # The cap trims DOMINANCE, it never punishes scarcity: each source keeps
        # its EARLIEST clips (detector order — stable and explainable, unlike a
        # random sample that changes on every promotion of the same bank).
        taken = {}
        kept = []
        for clip in rows:
            n = taken.get(clip.source_id, 0)
            if n < per_cap:
                taken[clip.source_id] = n + 1
                kept.append(clip)
        rows = kept
    clip_ids = [c.id for c in rows]
    # Composition, reported rather than judged: 60% of a dataset coming from one
    # source is invisible on disk — the folder looks exactly like a diverse one —
    # and it is the kind of imbalance that quietly overfits a source. Found by
    # our own first end-to-end test, which picked "the first 50 that pass" and
    # got three videos over-represented.
    per_source = {}
    for clip in rows:
        per_source[clip.source_id] = per_source.get(clip.source_id, 0) + 1
    # What the inset will COST, counted before a single file is written. The
    # arithmetic is free (bounds we already hold, plus the profile's own rule)
    # and it is the difference between choosing an inset and discovering it —
    # every limit stays visible.
    would_drop = 0
    if inset and profile.get('fps'):
        for clip in rows:
            span = clip.end_s - clip.start_s
            if (video_clip_export.fits_frames(span, frames, profile['fps'])
                    and not video_clip_export.fits_frames(span - 2 * inset, frames,
                                                          profile['fps'])):
                would_drop += 1
    # An empty sidecar trains as an EMPTY PROMPT and ai-toolkit says nothing about
    # it, so how many clips are about to ship without one is a limit that has to
    # be visible BEFORE the encode rather than discovered in a training run.
    captioned = sum(1 for c in rows if (c.caption or '').strip())
    # Shot-rate is a FACT about the source file, and 48+ fps footage is very
    # often slow motion once conformed — which teaches a model dreamy, floaty
    # movement (fal audited two thirds of their people clips as slo-mo). Stated,
    # never judged: no detector is pretended here, only the number the probe
    # already measured, so the user can weigh footage they know better than we do.
    high_fps_ids = {vs.id for vs in VideoSource.query.filter(
        VideoSource.id.in_({c.source_id for c in rows}),
        VideoSource.fps_native >= 48).all()} if rows else set()
    composition = {
        'high_fps_clips': sum(1 for c in rows if c.source_id in high_fps_ids),
        'sources': len(per_source),
        'captioned': captioned,
        'uncaptioned': len(rows) - captioned,
        'top_source_share': (max(per_source.values()) / len(rows)) if rows else 0.0,
        'edge_inset_s': inset,
        # Clips the INSET removes — not clips that were never long enough. Those
        # keep their own count, because only the first is fixed by lowering it.
        'inset_would_drop': would_drop,
    }
    if not clip_ids:
        raise ValueError('nothing to promote — keep some clips first')
    # "Keep the source's size" quietly bypasses the explicit-size validation, and
    # for a target that caps the canvas AREA that is exactly the gap: 1920x1088
    # is a clean multiple of 32 and still out of spec for MiniMax H3. Only bites
    # when the source size would SURVIVE — an explicit size rescales everything,
    # so big sources stop mattering. Refused here, before any folder exists.
    cap = profile.get('max_pixels')
    if cap and not size:
        oversized = (db.session.query(VideoSource)
                     .join(VideoClip, VideoClip.source_id == VideoSource.id)
                     .filter(VideoClip.id.in_(clip_ids),
                             VideoSource.width.isnot(None),
                             (VideoSource.width * VideoSource.height) > cap)
                     .count())
        if oversized:
            raise ValueError(
                f'{oversized} selected clip(s) come from sources larger than '
                f'{profile["label"]}\'s canvas cap ({cap:,} px). Pick a size '
                f'(e.g. {profile["recommended_sizes"][0][0]}x'
                f'{profile["recommended_sizes"][0][1]}) so they are rescaled, '
                f'or deselect those clips.')
    _ffmpeg_or_raise()

    trigger = (trigger_word or '').strip() or None
    dataset = VideoDataset(user_id=user_id, name=name,
                           target_profile=target_profile, fps=profile['fps'],
                           frames=frames, trigger_word=trigger,
                           width=size[0] if size else None,
                           height=size[1] if size else None,
                           output_dir='')
    db.session.add(dataset)
    db.session.flush()                      # need the id to name its folder
    out_dir = dataset_dir(dataset.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.output_dir = str(out_dir)
    db.session.commit()

    bank_jobs.start(app, job_key(bank_id), 'promote',
                    _promote_job(bank.id, dataset.id, clip_ids, target_profile,
                                 frames, size, inset, trigger),
                    total=len(clip_ids))
    return {'id': dataset.id, 'name': dataset.name,
            'output_dir': dataset.output_dir, 'clips': len(clip_ids),
            'composition': composition}


def _promote_job(bank_id, dataset_id, clip_ids, profile_key, frames, size,
                 edge_inset_s=0.0, trigger_word=None):
    """One ffmpeg per kept clip, straight into a FLAT folder.

    NOT ONE SUBFOLDER, EVER. ai-toolkit's dataset scan is os.walk — recursive —
    and excludes only dotfiles and a directory literally named `_controls`. A
    `preview/` or `rejects/` folder written here for our own convenience would be
    picked up and trained on with no message anywhere. That makes a subfolder a
    defect rather than a matter of taste, and it is why rejected clips are simply
    never encoded instead of being encoded somewhere out of the way.

    The .txt sidecar is written for EVERY clip that lands, even with no caption:
    musubi-tuner raises FileNotFoundError out of a worker future with no handler
    on the path, and diffusion-pipe drops the clip instead because its
    skip_empty_caption defaults to true. Wave 1 has no captioning at all, so
    without this every clip would take one of those two paths."""
    def run(job):
        ffmpeg = _ffmpeg_or_raise()
        bank = db.session.get(VideoBank, bank_id)
        dataset = db.session.get(VideoDataset, dataset_id)
        if bank is None or dataset is None:
            return {}
        out_dir = Path(dataset.output_dir)
        relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                        .filter_by(bank_id=bank_id).all())
        rows = {c.id: c for c in VideoClip.query.filter(
            VideoClip.id.in_(clip_ids)).all()}
        bank_jobs.progress(job, done=0, total=len(clip_ids), detail='encoding clips')

        # The profile's own fps, for telling "never long enough" apart from
        # "too short once the inset was taken off". None only for a profile that
        # declares no fps, which start_promote already refused.
        profile_fps = (video_targets.get(profile_key) or {}).get('fps')

        index = 0
        encoded = too_short = failed = dropped_by_inset = 0
        for clip_id in clip_ids:
            if bank_jobs.cancelled(job):
                break
            clip = rows.get(clip_id)
            if clip is None:
                continue
            relpath = relpaths.get(clip.source_id) or ''
            src = _abs_source_path(bank, relpath)
            if not src or not os.path.isfile(src):
                failed += 1
                bank_jobs.bump(job)
                continue
            # The filename index advances only on a clip that LANDED, so the folder
            # is contiguous: trainers walk it in filename order and a gap reads as
            # a dataset someone edited by hand.
            candidate = index + 1
            dst = out_dir / video_clip_export.clip_filename(candidate)
            # Both bounds pulled inwards: a shot boundary is where a cut just
            # happened, so the frames around BOTH ends are disproportionately
            # dissolves and fades. Zero by default — see _resolve_edge_inset.
            start_s = clip.start_s + edge_inset_s
            end_s = clip.end_s - edge_inset_s
            try:
                args = video_clip_export.command_for_profile(
                    ffmpeg=ffmpeg, src=src, dst=str(dst),
                    start_s=start_s, end_s=end_s,
                    profile_key=profile_key, frames=frames, size=size)
            except video_clip_export.ClipTooShort:
                # Loud, and it leaves NOTHING behind: a short clip encoded anyway
                # is a file ai-toolkit trains as repeated stills without a word.
                #
                # WHICH refusal it is decides what the user should do about it. A
                # clip that could never supply the frames is not fixable by
                # lowering a knob; one the inset just cost them is. Reporting
                # them as one number is how a setting quietly halves a dataset
                # and looks like the material was at fault.
                if (edge_inset_s and profile_fps
                        and video_clip_export.fits_frames(
                            clip.end_s - clip.start_s, frames, profile_fps)):
                    dropped_by_inset += 1
                else:
                    too_short += 1
                bank_jobs.bump(job)
                continue
            except ValueError as e:
                failed += 1
                logger.warning('video promote: %s', e)
                bank_jobs.bump(job)
                continue
            code, err = _run_ffmpeg(args)
            if code != 0 or not dst.exists():
                failed += 1
                logger.warning('video promote: ffmpeg exited %s: %s', code, err)
                try:
                    dst.unlink()            # never leave a half file in a dataset
                except OSError:
                    pass
                bank_jobs.bump(job)
                continue
            # THE SIDECAR IS THE PROMPT. Written for every clip that lands, with
            # the caption when there is one — and still written, empty, when
            # there is not: musubi-tuner raises FileNotFoundError out of a worker
            # future on a missing one, and diffusion-pipe drops the clip. An
            # EMPTY sidecar is not neutral either, it trains as an empty prompt
            # in silence, which is what the pre-flight count exists to surface.
            video_clip_export.write_sidecar(
                str(dst), compose_sidecar_text(trigger_word, clip.caption,
                                               clip.metrics_json))
            index = candidate
            encoded += 1
            db.session.add(VideoDatasetClip(
                dataset_id=dataset_id, filename=dst.name, caption=clip.caption,
                source_bank_id=bank_id, source_clip_id=clip.id,
                src_relpath=relpath, start_s=clip.start_s, end_s=clip.end_s))
            clip.promoted_dataset_id = dataset_id
            db.session.commit()
            bank_jobs.bump(job)

        detail = f'done — {encoded} clips encoded'
        if too_short:
            detail += f', {too_short} too short for {frames} frames'
        if dropped_by_inset:
            # Named separately in the sentence for the same reason it is counted
            # separately: this one is the user's own setting, and it is the only
            # one they can undo.
            detail += (f', {dropped_by_inset} dropped by the '
                       f'{edge_inset_s:g}s edge trim')
        if failed:
            detail += f', {failed} failed'
        bank_jobs.progress(job, detail=detail)
        return {'encoded': encoded, 'too_short': too_short,
                'dropped_by_inset': dropped_by_inset, 'failed': failed}
    return run


# --- video datasets ------------------------------------------------------------

def get_video_dataset(user_id, dataset_id) -> VideoDataset | None:
    return VideoDataset.query.filter_by(id=dataset_id, user_id=user_id).first()


def _with_trigger(trigger, caption) -> str:
    """The sidecar text: trigger first, caption after, exactly once.

    Prepended at WRITE time rather than stored inside the caption, so editing a
    caption in the UI never shows (or loses) the trigger, and changing nothing
    else about the flow: no trigger, and the sidecar is the caption verbatim.
    Idempotent on captions that already start with the trigger - a set promoted
    from a bank whose captions carry it does not get it twice, which is the
    duplication fal measured degrading prompt adherence."""
    trigger = (trigger or '').strip()
    caption = (caption or '').strip()
    if not trigger:
        return caption
    if caption.lower().startswith(trigger.lower()):
        return caption
    return f'{trigger}, {caption}' if caption else trigger


def compose_sidecar_text(trigger, caption, metrics_json) -> str:
    """The full training prompt for one exported clip: trigger, caption, and
    the MEASURED camera line.

    The camera comes from our homography classifier, never from the VLM — that
    was tried and refuted twice (the caption prompt now explicitly forbids it),
    so this is the one place the camera can enter the prompt, in words the
    classifier can prove. The `Camera:` label is the H3 idiom (the model's own
    prompts carry labeled blocks — `Audio:` — so a labeled camera line is what
    its encoder expects to read). No phrase when the classifier had nothing
    honest to say (unmeasured, or subject motion drowning the signal): a
    caption silent about the camera teaches nothing false.
    """
    from . import video_camera_motion
    base = _with_trigger(trigger, caption)
    try:
        scores = json.loads(metrics_json) if metrics_json else {}
    except (TypeError, ValueError):
        scores = {}
    phrase = video_camera_motion.camera_phrase(scores)
    if not phrase:
        return base
    line = f'Camera: {phrase}.'
    return f'{base} {line}' if base else line


def _dataset_row(ds: VideoDataset) -> dict:
    profile = video_targets.get(ds.target_profile) or {}
    seconds = video_targets.clip_seconds(ds.target_profile, ds.frames) \
        if ds.frames else None
    clip_count = VideoDatasetClip.query.filter_by(dataset_id=ds.id).count()
    return {
        'id': ds.id, 'name': ds.name, 'target_profile': ds.target_profile,
        'target_label': profile.get('label', ds.target_profile),
        'fps': ds.fps, 'frames': ds.frames,
        'clip_seconds': round(seconds, 3) if seconds else None,
        'width': ds.width, 'height': ds.height, 'output_dir': ds.output_dir,
        'clips': clip_count,
        # Computed here, where the count already is, so the two can never
        # disagree on screen. The launch routes do not read it — it prefills an
        # editable field, and what the user sends is what trains.
        'suggested_steps': video_training.suggested_steps(clip_count),
        'training_verified': profile.get('training_verified', False),
        # Surfaced on the dataset, not only in the picker: a user who built a set
        # for MiniMax H3 needs the territory restriction in front of them when
        # they come back to it, not once at creation.
        'licence_note': profile.get('licence_note'),
        'trigger_word': ds.trigger_word,
        'references': len(reference_dirs(ds)),
        'requires_references': bool(profile.get('requires_references')),
        'created_at': ds.created_at.isoformat() if ds.created_at else None,
    }


_REF_DIRNAME = '_controls'          # the ONE directory name ai-toolkit's scan skips
_REF_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp')
_MAX_REFERENCES = 4                 # upstream's own ref_1..ref_4 shape


def reference_dirs(ds) -> list:
    """The dataset's reference dirs (ref_1..ref_N), existing ones only, sorted."""
    root = Path(str(ds.output_dir or '')) / _REF_DIRNAME
    try:
        return sorted(d for d in root.iterdir()
                      if d.is_dir() and d.name.startswith('ref_'))
    except OSError:
        return []


def set_dataset_references(user_id, dataset_id, images) -> dict:
    """Attach 1-4 identity reference images to a ref2va dataset.

    `images` is [(filename, bytes)]. The trainer matches a control to a clip by
    FILE STEM inside each control dir, and a clip whose stem finds nothing is
    silently trained without its reference - so each reference is written once
    per clip stem, and the launch-side refusal can then trust that "dirs exist"
    means "every clip is covered". Replaces any previous set whole: references
    are one identity, not an album to append to."""
    ds = get_video_dataset(user_id, dataset_id)
    if ds is None:
        return None
    profile = video_targets.get(ds.target_profile) or {}
    if not profile.get('requires_references'):
        raise ValueError(f'{profile.get("label", ds.target_profile)} does not '
                         'train against reference images')
    imgs = [(n, b) for (n, b) in (images or [])
            if os.path.splitext(str(n))[1].lower() in _REF_IMAGE_EXTS and b]
    if not imgs:
        raise ValueError('attach 1-4 reference images (png/jpg/webp)')
    if len(imgs) > _MAX_REFERENCES:
        raise ValueError(f'at most {_MAX_REFERENCES} references — more of the '
                         'same identity stops adding information')
    clips = VideoDatasetClip.query.filter_by(dataset_id=ds.id).all()
    if not clips:
        raise ValueError('this dataset has no clips yet — promote first, then '
                         'attach references')
    root = Path(str(ds.output_dir)) / _REF_DIRNAME
    if root.exists():
        shutil.rmtree(root)
    stems = {os.path.splitext(c.filename)[0] for c in clips}
    for k, (name, data) in enumerate(imgs, start=1):
        ext = os.path.splitext(str(name))[1].lower()
        ref_dir = root / f'ref_{k}'
        ref_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            (ref_dir / f'{stem}{ext}').write_bytes(data)
    return {'references': len(imgs), 'clips_covered': len(stems)}


def create_stills_dataset_from_face_dataset(user_id, face_dataset_id,
                                            name=None) -> dict:
    """A VIDEO dataset of still images, built from an existing image dataset.

    "Image datasets (num_frames 1) train as single frames" is ai-toolkit's own
    road for H3, and the image side of the app already owns everything it needs
    - curated images, edited captions, a trigger. So this REUSES the image
    lane's own exporter (`export_dataset_to_aitoolkit`, masks off), which writes
    the exact flat image+.txt layout the trainer reads, trigger included; the
    rows are then registered so the dataset page shows what a promotion shows.

    The face dataset's trigger is copied onto the stills set so a later caption
    edit re-writes its sidecar with the same trigger, exactly once - the
    idempotent prepend already guards against doubling it."""
    from . import face_dataset_service as fds
    from . import lora_training as lt
    face = fds.get_dataset(user_id, face_dataset_id)
    if face is None:
        raise ValueError('image dataset not found')
    profile_key = 'minimax_h3'
    dataset = VideoDataset(
        user_id=user_id,
        name=(name or '').strip() or f'{face.name} — stills',
        target_profile=profile_key, fps=video_targets.get(profile_key)['fps'],
        frames=1, width=768, height=768,
        trigger_word=(getattr(face, 'trigger_word', None) or '').strip() or None,
        output_dir='')
    db.session.add(dataset)
    db.session.flush()
    out_dir = dataset_dir(dataset.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.output_dir = str(out_dir)
    try:
        lt.export_dataset_to_aitoolkit(user_id, face_dataset_id, masked=False,
                                       dest_dir=str(out_dir))
    except Exception:
        db.session.delete(dataset)
        db.session.commit()
        raise
    copied = 0
    for entry in sorted(os.listdir(out_dir)):
        stem, ext = os.path.splitext(entry)
        if ext.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            continue
        caption = None
        sidecar = out_dir / f'{stem}.txt'
        try:
            caption = sidecar.read_text(encoding='utf-8').strip() or None
        except OSError:
            pass
        db.session.add(VideoDatasetClip(dataset_id=dataset.id, filename=entry,
                                        caption=caption))
        copied += 1
    db.session.commit()
    if not copied:
        db.session.delete(dataset)
        db.session.commit()
        raise ValueError('that image dataset exported no images — nothing to train on')
    return {'id': dataset.id, 'name': dataset.name, 'clips': copied,
            'output_dir': dataset.output_dir}


def list_video_datasets(user_id) -> list:
    rows = (VideoDataset.query.filter_by(user_id=user_id)
            .order_by(VideoDataset.id.desc()).all())
    return [_dataset_row(d) for d in rows]


def video_dataset_payload(user_id, dataset_id) -> dict | None:
    ds = get_video_dataset(user_id, dataset_id)
    if ds is None:
        return None
    clips = (VideoDatasetClip.query.filter_by(dataset_id=ds.id)
             .order_by(VideoDatasetClip.filename.asc()).all())
    payload = _dataset_row(ds)
    payload['items'] = [{
        'id': c.id, 'filename': c.filename, 'caption': c.caption,
        'source_bank_id': c.source_bank_id, 'source_clip_id': c.source_clip_id,
        'src_relpath': c.src_relpath, 'start_s': c.start_s, 'end_s': c.end_s,
    } for c in clips]
    return payload


def set_dataset_clip_caption(user_id, dataset_id, clip_id, caption) -> dict | None:
    """Write a caption, and REWRITE THE SIDECAR IN THE SAME BREATH.

    Storing the caption in the database alone is the quiet failure of this
    feature: the trainer never reads our database, it reads the .txt next to the
    .mp4. The two must move together or the dataset trains on what it had before,
    with the UI showing what it has now."""
    ds = get_video_dataset(user_id, dataset_id)
    if ds is None:
        return None
    row = VideoDatasetClip.query.filter_by(dataset_id=ds.id, id=clip_id).first()
    if row is None:
        return None
    row.caption = (caption or '').strip() or None
    db.session.commit()
    clip_path = os.path.join(ds.output_dir, row.filename)
    written = True
    try:
        video_clip_export.write_sidecar(
            clip_path, _with_trigger(ds.trigger_word, row.caption))
    except OSError as e:
        written = False
        logger.warning('video dataset %s: could not write sidecar: %s', ds.id, e)
    return {'ok': True, 'caption': row.caption, 'sidecar_written': written}


def delete_video_dataset(user_id, dataset_id) -> bool:
    """Throw away a badly cut dataset — the ENCODE, never the triage.

    The bank's clips stay exactly as they were; they only stop claiming to have
    been promoted. That is the whole point of promoted_dataset_id being a real FK
    with SET NULL, and it is applied here by hand because SQLite enforces neither
    the cascade nor the SET NULL unless the PRAGMA is on."""
    ds = get_video_dataset(user_id, dataset_id)
    if ds is None:
        return False
    (VideoClip.query.filter_by(promoted_dataset_id=ds.id)
     .update({'promoted_dataset_id': None}, synchronize_session=False))
    VideoDatasetClip.query.filter_by(dataset_id=ds.id).delete(
        synchronize_session=False)
    # Divorce this id's cloud-run history from whoever inherits the rowid:
    # `owns()` refuses stamped runs, so a future dataset reusing the id never
    # shows — or serves — a stranger's checkpoints.
    from ..models import CloudTrainingRun
    from . import cloud_run_dataset as crd
    for run in CloudTrainingRun.query.filter_by(dataset_id=ds.id).all():
        if crd.table_of(run) != crd.VIDEO:
            continue
        try:
            params = json.loads(run.train_params or '{}')
        except (TypeError, ValueError):
            params = {}
        params['dataset_deleted'] = True
        run.train_params = json.dumps(params)
    db.session.flush()
    out_dir = ds.output_dir
    db.session.delete(ds)
    db.session.commit()
    try:
        from . import trash
        if out_dir and os.path.isdir(out_dir):
            trash.dispose(out_dir, context='video dataset')
    except Exception as e:                  # noqa: BLE001 — the rows are gone already
        logger.warning('video dataset %s: could not dispose its folder: %s',
                       dataset_id, e)
    return True


# --- 🕸 scrape → VIDEO BANK -----------------------------------------------------
#
# The scraper already listed videos: `/api/scrape/scan` returns items carrying
# `type: 'video'` from RedGifs, Erome, Picazor, TikTok, X, Civitai and every
# gallery-dl backed source. Nothing consumed them — the picker dropped them on
# the floor and the only way to get a scraped clip into a bank was to download it
# by hand, into a folder, and point a bank at it.
#
# This is the image lane's `image_bank_service.scrape_import_to_bank` adapted to
# video, and it keeps ALL of its invariants deliberately:
#
#   1. ONE inventory path. Files are downloaded into the bank's folder and the
#      ORDINARY walk (`refresh_bank`) registers them. No second insert.
#   2. No quality judgement at intake. Short, still, tiny or duplicated is what
#      the metrics pass and the triage exist to rule on; a clip refused at
#      download time is one nobody can review.
#   3. Content-hash naming, so re-importing the same bytes overwrites nothing and
#      reports `already_there` — file identity, never a "duplicate" verdict.
#   4. The lease is re-checked AFTER the downloads (which are slow) and before the
#      first write, so a stale reservation can never publish under a newer owner.
#
# WHERE THE SOURCE FOLDER'S READ-ONLY RULE STANDS, AND WHERE IT DOES NOT. Every
# PASS in this module still refuses to touch the folder a bank points at: probing,
# detection, thumbnails and promotion write to `video_banks_root()` and
# `video_datasets_root()`, never to your footage. A scrape is not a pass — it is
# an errand you sent, at a destination you named — and it adds the files it
# brings back to the folder that bank follows, your own included.
#
# This shipped the other way round for one wave: only a folder the app had itself
# created could receive a scrape. It was defensible and it was wrong, because it
# answered a question nobody had asked ("may the app write here?") instead of the
# one they had ("put these clips in THAT bank"). CHOOSING THE BANK IS THE
# CONSENT — there is no second opt-in, and the UI says plainly, at the moment of
# choosing, when the chosen folder is one of yours. The image bank has always
# worked exactly like this.
#
# What is still refused is the one thing consent cannot fix: a folder that belongs
# to a DATASET. Files landing there would sit inside training material, get
# trained on, and be attributed to a dataset nobody added them to.

# Per REQUEST cap. Far below the image outlet's 60 for a reason that is arithmetic
# rather than taste: one image is capped at 12 MB and 20 s, one video at 200 MB
# and 180 s (netfetch.MAX_DRIVER_BYTES / DOWNLOAD_TIMEOUT). Six items over two
# workers bounds a request at three download rounds — the same order of magnitude
# as the image outlet's worst case, instead of an order beyond it. Bigger
# selections are not refused: the client sends them as successive batches, the way
# it already does for images.
SCRAPE_VIDEO_IMPORT_MAX = 6

# Two, not the image lane's six. A video download saturates the link on its own;
# more of them in flight does not make the pipe wider, it only multiplies the peak
# disk of half-finished files and the number of sources one request can annoy.
_SCRAPE_VIDEO_DL_WORKERS = 2

_SCRAPE_VIDEO_TIMEOUT = 180        # wall clock of ONE direct-file fetch, = netfetch's
_SCRAPE_VIDEO_CHUNK = 256 * 1024

_SCRAPE_FOLDER_SAFE = re.compile(r'[^A-Za-z0-9 _-]')

# URL extensions that mean "this IS the media" — fetch the bytes directly rather
# than paying a yt-dlp subprocess to rediscover what the link already says. Wider
# than VIDEO_EXTS on purpose: what is STORED is decided by the file's own magic
# (see `_video_extension_from_magic`), so recognising `.m4v` here costs nothing
# and simply routes it away from the resolver.
_DIRECT_VIDEO_URL_EXTS = ('.mp4', '.m4v', '.webm', '.mov', '.mkv', '.avi', '.ts')

# ISO-BMFF brands that are NOT video, listed rather than guessed: the container
# is shared by MP4, the whole HEIF picture family and M4A audio, so `ftyp` alone
# proves nothing. Everything else with an `ftyp` is treated as MP4 — unknown
# brands there are overwhelmingly MP4 profiles, and the probe pass is the honest
# place for a file that turns out to hold no video stream.
_NON_VIDEO_BMFF_BRANDS = frozenset((
    b'avif', b'avis',                                    # AVIF stills / sequences
    b'heic', b'heix', b'heim', b'heis',                  # HEIF pictures
    b'hevc', b'hevx', b'hevm', b'hevs',                  # HEIF sequences
    b'mif1', b'msf1',                                    # HEIF generic
    b'M4A ', b'M4B ', b'M4P ',                           # audio-only
))


def is_app_owned_scrape_folder(path) -> bool:
    """Whether the APP created ``path`` — i.e. it sits under
    ``video_bank_sources_root()`` because a scrape made a bank of its own.

    NOT a permission. It used to be one, and gating on it was the mistake this
    module now documents at length; a bank you pointed at your own footage takes
    a scrape like any other. What it is still good for, both of them load-bearing:

    * it is the containment test behind the only ``rmtree`` in this lane
      (``_discard_unlaunched_scrape_bank``) — that one MUST stay incapable of
      deleting a folder the user assembled;
    * it tells the UI whether the destination is a folder of the user's own, which
      is the only case worth a sentence before they click.

    Same containment test as ``_contained_path``: both sides realpath'd, and the
    separator carried so `/x/rushes-2` cannot pass for a folder under
    `/x/rushes`."""
    if not path:
        return False
    try:
        root = os.path.realpath(str(cfg.video_bank_sources_root()))
        full = os.path.realpath(str(path))
    except OSError:
        return False
    return os.path.normcase(full).startswith(os.path.normcase(root + os.sep))


def _dataset_folder_refusal(folder) -> str | None:
    """The sentence refusing a folder that belongs to a dataset, or None.

    BOTH roots, exactly like ``create_bank``: the image lane's (harmless today,
    but the rule is the rule) and the video lane's own, which is the real trap —
    downloading into a folder a bank points at, when that folder is also a video
    dataset's output, would make the bank list its own training clips as source
    material and re-promote them on the next pass.

    This is the ONE refusal that survived the move to "picking the bank is the
    consent": consent cannot make files landing inside training material safe."""
    if not folder:
        return None
    for root in (None, cfg.video_datasets_root()):
        try:
            conflict = path_guard.dataset_folder_conflict(folder, datasets_root=root)
        except OSError:      # an unreachable root is not a conflict
            continue
        if conflict:
            return ('This bank points at a dataset\'s own folder, so scraping into '
                    f'it would drop files inside the dataset. {conflict["message"]}')
    return None


def _scrape_folder_for(name: str) -> str:
    """A fresh, unused folder for a scraped bank. Suffixes -2, -3… rather than
    reusing one: two scrapes of the same name must never silently merge into a
    single pile. Creation IS the reservation (``os.mkdir``, not exists-then-make),
    so two concurrent imports cannot end up sharing one folder."""
    stem = _SCRAPE_FOLDER_SAFE.sub('_', name or '').strip() or 'bank'
    root = cfg.video_bank_sources_root()
    candidate = root / stem
    i = 2
    while True:
        try:
            os.mkdir(candidate)
            return os.path.realpath(str(candidate))
        except FileExistsError:
            candidate = root / f'{stem}-{i}'
            i += 1


def _stage_scrape_video_bank(user_id, name) -> VideoBank:
    """Reserve the private folder and FLUSH the still-uncommitted bank row, so its
    id can be reserved in ``bank_jobs`` before the row is visible elsewhere."""
    folder = _scrape_folder_for(name)
    try:
        bank = VideoBank(user_id=user_id, name=name, source_path=folder)
        db.session.add(bank)
        db.session.flush()
        return bank
    except Exception:
        db.session.rollback()
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _discard_unlaunched_scrape_bank(user_id, bank_id, folder):
    """Remove a staged/committed destination whose import never got anywhere.

    "Never got anywhere" is checked, not assumed: the caller's guard
    (`bank_jobs.launched`) is structurally always False on this synchronous
    path — nothing ever calls `bank_jobs.start` here — so ANY exception used to
    reach this cleanup, including one raised AFTER the downloads had landed
    (a `refresh_bank` commit failing under a transient SQLite lock). Deleting
    the folder then destroys up to six freshly-downloaded videos to clean up a
    DB hiccup. A folder that already holds files is an import that DID get
    somewhere: keep the bank and the files, let the next walk inventory them,
    and let the error surface on its own."""
    try:
        db.session.rollback()
    except Exception:  # noqa: BLE001 — cleanup must not mask the original failure
        logger.warning('video bank scrape: rollback failed', exc_info=True)
    try:
        landed = bool(folder) and os.path.isdir(folder) and bool(os.listdir(folder))
    except OSError:
        landed = True   # cannot PROVE it is empty — never delete on a guess
    if landed:
        logger.warning('video bank scrape: bank %s kept — its folder already '
                       'holds downloaded files', bank_id)
        return
    if bank_id is not None and get_bank(user_id, bank_id) is not None:
        try:
            delete_bank(user_id, bank_id)
        except Exception:  # noqa: BLE001
            logger.warning('video bank scrape: could not discard bank %s', bank_id,
                           exc_info=True)
    # `folder` came from _scrape_folder_for; re-check containment anyway so a
    # future caller passing another path cannot turn this into a delete tool.
    # THIS CHECK IS NOT DECORATION any more: a scrape can now be aimed at a folder
    # of the user's own, and the only thing standing between a failed import and
    # an `rmtree` of someone's rushes is that this cleanup only ever removes a
    # folder the app itself created.
    if folder and is_app_owned_scrape_folder(folder):
        shutil.rmtree(folder, ignore_errors=True)


def _video_extension_from_magic(head: bytes) -> str | None:
    """The extension a downloaded blob gets, read from its own first bytes.

    THE EXTENSION IS NOT COPIED FROM THE URL, for a reason that would otherwise
    bite silently: the walk only inventories ``VIDEO_EXTS``, so a `.m4v` (or a
    query-string URL with no extension at all) would land in the folder, be
    ignored by the walk, and the import would report files nobody can see. Reading
    the container settles both questions at once — is this really a video, and
    under which of the five names the walk knows.

    None when the bytes are not a container we store, and that verdict is the
    intake's OWN — it is deliberately stricter than what brought the file here.
    `netfetch.download_via_ytdlp` keeps anything with a broad video signature,
    GIF included, because its own caller (a driver video) can use one. This bank
    cannot: a `.gif` in the folder is a file the walk never lists, so it would sit
    there for ever, counted by nobody. Refusing here — and letting the staging
    folder take the file away with it — is what keeps "downloaded" and
    "inventoried" the same set."""
    if not isinstance(head, (bytes, bytearray)) or len(head) < 12:
        return None
    head = bytes(head)
    if head[4:8] == b'ftyp':
        brand = head[8:12]
        # AVIF/HEIF are ISO-BMFF too, and so is an M4A. They are a picture and a
        # sound file — refuse rather than store one under a video name (the same
        # ordering trap netfetch documents) and leave the probe pass to discover
        # it has no video stream.
        if brand in _NON_VIDEO_BMFF_BRANDS:
            return None
        return '.mov' if brand == b'qt  ' else '.mp4'
    if head[:4] == b'\x1a\x45\xdf\xa3':          # EBML — Matroska family
        # WebM is a Matroska profile; the DocType string sits in the header and is
        # the only honest way to tell the two apart.
        return '.webm' if b'webm' in head[:64] else '.mkv'
    if head[:4] == b'RIFF' and head[8:12] == b'AVI ':
        return '.avi'
    return None


def _video_blob_name(path) -> str | None:
    """The name a downloaded video takes in the bank folder: its own content hash.

    Same two consequences as the image lane's `_scrape_blob_name` — a re-download
    of the SAME bytes writes the same name (idempotent, with nobody having to
    decide what a duplicate is), while a re-encode or another resolution keeps a
    name of its own and reaches the bank, where the shot metrics and the triage
    are the one place that rules on them.

    None when the file is not a container this lane stores. OSError propagates:
    "the bytes are not a video" and "the file could not be read back" must not
    collapse into one word, because the first is about the FILE and the second is
    about the machine (an antivirus lock right after writing is a measured
    reality here) — the caller counts them under different reasons."""
    with open(path, 'rb') as fh:
        head = fh.read(64)
        ext = _video_extension_from_magic(head)
        if ext is None:
            return None
        digest = hashlib.sha256()
        digest.update(head)
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f'{digest.hexdigest()[:24]}{ext}'


def _scrape_video_route(url) -> str:
    """'direct' when the URL IS the file, 'resolve' when a page has to be read.

    gallery-dl backed sources hand back a CDN media URL (`.../clip.mp4`) — asking
    yt-dlp to open that is a subprocess and a second request for bytes we can
    simply stream. RedGifs, TikTok, X and friends hand back a WATCH PAGE, where
    yt-dlp is exactly the right tool. The extension is the discriminator because
    it is the only thing both shapes agree to expose."""
    try:
        from urllib.parse import urlparse
        path = (urlparse(str(url or '')).path or '').lower()
    except Exception:  # noqa: BLE001 — an unparseable URL is simply not direct
        return 'resolve'
    return 'direct' if path.endswith(_DIRECT_VIDEO_URL_EXTS) else 'resolve'


def _stream_video_to_disk(url, dest_path) -> str:
    """Fetch a direct media URL STRAIGHT TO DISK. Returns the skip reason
    ('ok' | 'not_video' | 'too_large' | 'no_curl' | 'errors').

    Deliberately not `netfetch.fetch_hardened_bytes`: that one builds the whole
    body in memory, which is right for a 12 MB photo and wrong for a 200 MB clip.
    Every one of its guards is kept: anti-SSRF validation of the URL,
    `allow_redirects=False` (a 3xx toward an internal IP would walk around the
    validation that just ran), a content-type allow-list, and the byte cap
    enforced DURING the stream rather than after it."""
    from ..scrape.netfetch import MAX_DRIVER_BYTES, _validate_public_http_url
    ok_url, err = _validate_public_http_url(url)
    if not ok_url:
        logger.warning('video bank scrape: URL refused by the SSRF guard: %s', err)
        return 'errors'
    try:
        from curl_cffi import requests as cf_requests
    except ImportError:
        # Every direct-file item fails on an install without the scrape extras —
        # a dedicated reason (same word as `fetch_hardened_bytes`) so the client
        # can say "install the scrape extras" instead of blaming the network.
        logger.warning('video bank scrape: curl_cffi is not installed — '
                       'direct-file downloads need the scrape extras')
        return 'no_curl'
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ''
    try:
        r = cf_requests.get(url, impersonate='chrome', timeout=_SCRAPE_VIDEO_TIMEOUT,
                            stream=True, allow_redirects=False,
                            headers={'Referer': f'https://{host}/',
                                     'Accept': 'video/*,*/*'})
    except Exception:  # noqa: BLE001 — a network failure is a skipped item
        logger.warning('video bank scrape: fetch failed for host %s', host,
                       exc_info=True)
        return 'errors'
    try:
        if r.status_code != 200:
            logger.warning('video bank scrape: host %s answered %s', host,
                           r.status_code)
            return 'errors'
        ctype = (r.headers.get('content-type') or '').split(';')[0].strip().lower()
        if not ctype.startswith('video/'):
            # A content-type that is not video/* is html, a login wall or a
            # mislabelled blob. The magic check would catch it later anyway; not
            # spending 200 MB of bandwidth to find out is the point.
            return 'not_video'
        # The library's `timeout` does NOT bound a streaming body — with
        # stream=True it degrades to a connect timeout plus a low-speed guard, so
        # a server trickling one byte per second holds this thread (and the
        # bank's lease) open indefinitely. The wall clock below is the actual
        # 180 s promise the constant's comment makes.
        deadline = time.monotonic() + _SCRAPE_VIDEO_TIMEOUT
        written = 0
        with open(dest_path, 'wb') as fh:
            for chunk in r.iter_content(_SCRAPE_VIDEO_CHUNK):
                if time.monotonic() > deadline:
                    logger.warning('video bank scrape: host %s exceeded the '
                                   '%ss wall clock', host, _SCRAPE_VIDEO_TIMEOUT)
                    return 'errors'
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_DRIVER_BYTES:
                    return 'too_large'
                fh.write(chunk)
        if written == 0:
            # A 200 with an empty body is a broken download, not a verdict about
            # the file — 'not_video' here would send someone to inspect a clip
            # that never arrived.
            logger.warning('video bank scrape: host %s sent an empty body', host)
            return 'errors'
    except OSError:
        logger.warning('video bank scrape: could not write the download',
                       exc_info=True)
        return 'errors'
    finally:
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    return 'ok'


def _download_scrape_video(item, staging_dir) -> tuple:
    """Bring ONE scan item down into ``staging_dir``. Returns (reason, path|None).

    Staging, not the bank folder: the stored name is the content hash, which is
    only known once the bytes are in hand — and a half-written .mp4 sitting in the
    bank folder is a file the walk would happily inventory as a rush."""
    url = (item or {}).get('url')
    if not url:
        return ('errors', None)
    uid = uuid.uuid4().hex
    if _scrape_video_route(url) == 'direct':
        dest = os.path.join(staging_dir, uid)
        reason = _stream_video_to_disk(url, dest)
        return (reason, dest if reason == 'ok' else None)
    from ..scrape.netfetch import (MAX_DRIVER_BYTES, download_via_ytdlp,
                                   _validate_public_http_url)
    ok_url, err = _validate_public_http_url(url)
    if not ok_url:
        logger.warning('video bank scrape: URL refused by the SSRF guard: %s', err)
        return ('errors', None)
    try:
        ok, filename, error = download_via_ytdlp(url, os.path.join(staging_dir, uid))
    except Exception:  # noqa: BLE001 — the resolver is a subprocess plus optional
        # imports; whatever it fails on, this item is skipped and the rest of the
        # batch continues.
        logger.warning('video bank scrape: resolver failed', exc_info=True)
        return ('errors', None)
    if not ok or not filename:
        logger.warning('video bank scrape: resolver kept nothing: %s',
                       error or 'no file produced')
        return ('errors', None)
    path = os.path.join(staging_dir, filename)
    # The 200 MB promise is only enforced by yt-dlp's plain-HTTP downloader; the
    # fragmented ones (HLS/DASH — what a watch page usually serves) ignore
    # `--max-filesize` entirely. The direct branch counts its bytes during the
    # stream; this is the resolver branch's equivalent, after the fact.
    try:
        if os.path.getsize(path) > MAX_DRIVER_BYTES:
            os.remove(path)
            logger.warning('video bank scrape: resolver output exceeded the '
                           '%s-byte cap', MAX_DRIVER_BYTES)
            return ('too_large', None)
    except OSError:
        logger.warning('video bank scrape: could not size the resolver output',
                       exc_info=True)
        return ('errors', None)
    return ('ok', path)


def scrape_import_to_video_bank(user_id, items, bank_id=None, name=None, *,
                                _bank_lease=None, _created=False) -> dict:
    """🕸 Scrape → VIDEO BANK: the scraper's third destination.

    Downloads the SELECTED scanned videos ({'url','title',…}) into a bank's source
    folder, then lets the ordinary folder walk inventory them — the same single
    inventory path every other video bank uses. Two modes: ``bank_id`` appends to
    an EXISTING bank — any of them, including one you pointed at your own rushes
    (a bank follows a live folder, so a second scrape resumes the pile) — and
    ``name`` creates one under ``video_bank_sources_root()``.

    PICKING THE BANK IS THE CONSENT. There is no opt-in beside it: naming a
    destination is not something a user does by accident, and a second
    confirmation would only train them to click through it. The clips are added
    to the folder that bank follows, and the picker says so, with the path, when
    that folder is one of theirs. The only destination still refused is a folder
    that belongs to a DATASET — see ``_dataset_folder_refusal``.

    Nothing is judged at intake — length, motion, sharpness and duplicates are
    verdicts the metrics pass produces, with thresholds the user moves. Provenance
    goes through the SAME validation gate as the image lane
    (`normalize_source_metadata`), so what is stored is never the client's raw
    claim; a platform that gate does not recognise stores nothing rather than a
    guess.

    Returns {'bank_id', 'name', 'created', 'saved', 'already_there', 'added',
    'skipped': {...}} — ``added`` is what the walk actually inventoried. Raises
    ValueError (bad input) or BankJobBusy (a pass owns the bank)."""
    items = [it for it in (items or []) if isinstance(it, dict) and it.get('url')]
    if not items:
        raise ValueError('no items')
    if len(items) > SCRAPE_VIDEO_IMPORT_MAX:
        raise ValueError(f'max {SCRAPE_VIDEO_IMPORT_MAX} videos per import')

    if bank_id is not None:
        key = job_key(bank_id)
        if _bank_lease is None:
            # The quick advisory 409 first (it is what gives the UI a `busy_kind`
            # to name), then the atomic lease, which is the authority if this read
            # raced a new owner.
            if bank_jobs.running(key):
                snap = bank_jobs.get(key) or {}
                raise bank_jobs.BankJobBusy(snap.get('kind') or 'background')
            with bank_jobs.mutation_lease(key, 'scrape_import') as lease:
                return scrape_import_to_video_bank(
                    user_id, items, bank_id=bank_id, name=name,
                    _bank_lease=lease, _created=_created)
        bank_jobs.require_reservation(_bank_lease, key)
        bank = get_bank(user_id, bank_id)
        if bank is None:
            raise ValueError('video bank not found')
        folder = bank.source_path
        if not folder or not os.path.isdir(folder):
            raise ValueError("this bank's folder is unavailable right now")
        # A folder of the user's own is a legitimate destination — they named it.
        # A DATASET's folder is not, and no amount of naming makes it one.
        refusal = _dataset_folder_refusal(folder)
        if refusal:
            raise ValueError(refusal)
    else:
        name = (name or '').strip()
        if not name:
            raise ValueError('name is required')
        bank = reservation = None
        folder = None
        try:
            # The row becomes visible only once its reservation is installed,
            # exactly like the image lane's background import paths.
            bank = _stage_scrape_video_bank(user_id, name)
            folder = bank.source_path
            reservation = bank_jobs.reserve(job_key(bank.id), 'scrape_import')
            db.session.commit()
            return scrape_import_to_video_bank(
                user_id, items, bank_id=bank.id, name=name,
                _bank_lease=reservation, _created=True)
        except Exception:
            if bank is not None and not bank_jobs.launched(reservation):
                _discard_unlaunched_scrape_bank(user_id, bank.id, folder)
            raise
        finally:
            bank_jobs.abort(reservation)

    from flask import current_app, has_app_context
    app = current_app._get_current_object() if has_app_context() else None
    staging = tempfile.mkdtemp(prefix='vbank_scrape_')

    def _fetch(item):
        # The resolver logs through `current_app`; a worker thread has no context
        # of its own, and without this a yt-dlp failure would surface as an
        # unrelated RuntimeError instead of the reason it actually failed.
        if app is None:
            return _download_scrape_video(item, staging)
        with app.app_context():
            return _download_scrape_video(item, staging)

    try:
        with ThreadPoolExecutor(max_workers=_SCRAPE_VIDEO_DL_WORKERS) as pool:
            # Kept paired with its item: the blob name is only known once the
            # bytes are in hand, and the pairing is also the only way back to the
            # provenance a given file owns.
            downloaded = list(zip(items, pool.map(_fetch, items)))
        # Downloads are slow. Re-assert the capability before the first write so a
        # stale or purged lease can never publish beside a newer bank owner.
        bank_jobs.require_reservation(_bank_lease, job_key(bank.id))

        from .face_dataset_service import _source_metadata_storage

        skipped: dict[str, int] = {}
        saved = already_there = 0
        # blob name (== relpath: scraped files land FLAT in the bank folder) ->
        # validated provenance JSON, handed to the walk so a freshly inventoried
        # row is born WITH its source instead of the walk having no way to attach
        # one to a bare file it just found.
        source_metadata_by_relpath: dict[str, str] = {}
        for item, (reason, path) in downloaded:
            if reason != 'ok' or not path:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            try:
                blob = _video_blob_name(path)
            except OSError:
                # The file arrived but cannot be read back (antivirus lock, lost
                # handle). That is a machine problem, not a verdict about the
                # clip — 'not_video' here would be a lie about a valid file.
                logger.warning('video bank scrape: downloaded file unreadable',
                               exc_info=True)
                skipped['errors'] = skipped.get('errors', 0) + 1
                continue
            if blob is None:
                skipped['not_video'] = skipped.get('not_video', 0) + 1
                continue
            stored = _source_metadata_storage(item, image_url=item.get('url'))
            if stored:
                source_metadata_by_relpath[blob] = stored
            dest = os.path.join(folder, blob)
            if os.path.exists(dest):
                already_there += 1
                continue
            # Publish in two steps. `shutil.move` across volumes (staging lives
            # in %TEMP%, the bank's folder often does not) degrades to a
            # progressive copy INTO the destination name — a crash mid-copy
            # would leave a truncated file under its final content-hash name,
            # which the walk then inventories and which no re-import of the same
            # bytes ever repairs (the hash "already exists"). The `.part` name is
            # invisible to the walk (not in VIDEO_EXTS); `os.replace` on the same
            # volume is atomic.
            part = dest + '.part'
            try:
                shutil.move(path, part)
                os.replace(part, dest)
            except OSError:
                logger.warning('video bank scrape: could not store %s', blob,
                               exc_info=True)
                try:
                    if os.path.exists(part):
                        os.remove(part)
                except OSError:
                    pass
                skipped['errors'] = skipped.get('errors', 0) + 1
                continue
            saved += 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # ONE inventory path, the same walk that picks up files dropped in the folder
    # by hand. No third insert.
    sync = refresh_bank(user_id, bank.id, force=True, _bank_lease=_bank_lease,
                        source_metadata_by_relpath=source_metadata_by_relpath) or {}
    out = {'bank_id': bank.id, 'name': bank.name, 'created': _created,
           'saved': saved, 'already_there': already_there,
           'added': sync.get('added', 0), 'skipped': skipped}
    if sync.get('error'):
        # Files downloaded but the walk could not inventory them is NOT the
        # success it used to read as: without this field the client shows the
        # perfect-run sentence while `added` is silently zero.
        out['sync_error'] = sync['error']
    return out
