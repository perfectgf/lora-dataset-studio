import os
import logging
import mimetypes
import sqlite3
import json
from pathlib import Path
from flask import (
    Flask, Request, current_app, jsonify, request, send_from_directory)
from sqlalchemy import event
from sqlalchemy.engine import Engine
from .extensions import db, csrf
from . import config as cfg

FRONTEND_DIST = cfg.REPO_ROOT / 'frontend' / 'dist'
logger = logging.getLogger(__name__)

# --- Content types for our own static files ---------------------------------
# Flask/Werkzeug label every file they send with `mimetypes.guess_type()`. On
# Windows that module seeds itself from the registry
# (HKEY_CLASSES_ROOT\<ext>\Content Type), which ANY installed program is free to
# overwrite — so the type a browser is told depends on what else happens to be on
# the machine. When `.js` has been rewritten to `text/plain`, the browser refuses
# to execute the bundle and the app opens on a blank page with nothing at all in
# the server log. Reported, diagnosed AND fixed by gessyoo (GitHub #12); the same
# lottery has been observed hitting `.mjs` on a different machine, so this is not
# one broken PC.
#
# The cure is to never ask the registry what our own files are. Every value below
# is the standard one — the very value Python's built-in table carries when no
# registry is involved — so on a healthy machine (and on Linux/macOS) this pins
# what was already being served and changes nothing observable. It only ever
# repairs a downgrade; it never invents a type for an extension we don't ship.
_STATIC_MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',      # RFC 9239; browsers execute this and
    '.mjs': 'text/javascript',     # application/javascript identically
    '.css': 'text/css',
    '.json': 'application/json',
    '.map': 'application/json',     # source maps: devtools-only, never executed
    '.svg': 'image/svg+xml',
    '.webp': 'image/webp',
    '.png': 'image/png',
    '.bmp': 'image/bmp',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.avif': 'image/avif',
    '.ico': 'image/vnd.microsoft.icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
    '.otf': 'font/otf',
    '.wasm': 'application/wasm',
    '.txt': 'text/plain',
}


def pin_static_mime_types():
    """Force the standard content type for every extension the frontend ships.

    `mimetypes.add_type(..., strict=True)` writes into the same table
    `guess_type()` reads, and it is applied AFTER the registry has been loaded
    (`add_type` initialises the database first, then overwrites the entry), so
    the pinned value wins over whatever the registry said. Verified rather than
    assumed — see backend/tests/test_static_mime_types.py.

    Idempotent and called twice on purpose: at import time (so ANY entry point —
    backend/run.py, the packaged launcher, a WSGI server, a test importing
    create_app — is covered without having to remember) and again from
    create_app(), because a `mimetypes.init()` executed later by any other
    library rebuilds that table from the registry and would silently drop the
    pins.
    """
    for ext, ctype in _STATIC_MIME_TYPES.items():
        mimetypes.add_type(ctype, ext, strict=True)


pin_static_mime_types()

_DEFAULT_DATASET_ARCHIVE_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_DATASET_ARCHIVE_UPLOAD_ENDPOINTS = frozenset({
    'datasets.dataset_backup_import',
    'datasets.dataset_import_zip',
    # 'Back up everything' restore accepts a master archive that can be as large
    # as the whole library — it needs the same raised request ceiling.
    'backup.full_restore',
})


class ArchiveAwareRequest(Request):
    """Give the archive-upload endpoints — and only them — a raised ceiling.

    The limit has to be in place before anything reads the body, and
    Flask-WTF reads ``request.form`` in its own ``before_request`` hook, so a
    hook of ours would have to win a registration race to be of any use.
    Assigning ``request.max_content_length`` from that hook also depends on the
    property being settable, which it only became in Flask 3.1: on any older
    Flask the assignment raises ``AttributeError`` and every archive upload
    answers 500.

    Declaring the ceiling on the request class instead removes both problems.
    It is read lazily, whenever the body is first touched and however early
    that is, and it never writes to a framework property. The geometry is
    unchanged: the ordinary ``MAX_CONTENT_LENGTH`` still governs every other
    endpoint.
    """

    #: Honours an explicit per-request assignment first, so this class stays a
    #: drop-in for the Flask >= 3.1 behaviour it replaces.
    _forced_max_content_length = None

    @property
    def max_content_length(self):
        if self._forced_max_content_length is not None:
            return self._forced_max_content_length
        if self.endpoint in _DATASET_ARCHIVE_UPLOAD_ENDPOINTS and current_app:
            archive_max = int(
                current_app.config['DATASET_ARCHIVE_MAX_UPLOAD_BYTES'])
            overhead = max(0, int(
                current_app.config['DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES']))
            return archive_max + overhead
        return super().max_content_length

    @max_content_length.setter
    def max_content_length(self, value):
        self._forced_max_content_length = value

    def _raise_if_declared_body_is_too_large(self):
        """Apply the selected ceiling on Flask/Werkzeug versions before 2.3.

        Older Werkzeug guarded multipart parsing but its raw ``get_data``
        stream did not consult Flask's ``MAX_CONTENT_LENGTH`` at all.  Keep the
        same endpoint-aware limit for both entry paths so an ordinary upload
        cannot bypass its 64 MiB ceiling merely by using a non-form body.
        """
        limit = self.max_content_length
        if (limit is not None and self.content_length is not None
                and self.content_length > limit):
            from werkzeug.exceptions import RequestEntityTooLarge
            raise RequestEntityTooLarge()

    def _load_form_data(self):
        self._raise_if_declared_body_is_too_large()
        return super()._load_form_data()

    def get_data(self, *args, **kwargs):
        self._raise_if_declared_body_is_too_large()
        return super().get_data(*args, **kwargs)


def _positive_env_int(name, default):
    """Read a positive integer without making a bad optional env var fatal."""
    try:
        value = int((os.environ.get(name) or '').strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _configure_sqlite_connection(dbapi_con, _connection_record):
    """Apply the app's SQLite guarantees to every newly-opened connection.

    This listener is registered once, at module import, instead of once per
    ``create_app`` call.  App-factory tests and embedded launches can therefore
    create several Flask apps without stacking duplicate engine listeners.
    """
    if not isinstance(dbapi_con, sqlite3.Connection):
        return
    cur = dbapi_con.cursor()
    try:
        cur.execute('PRAGMA foreign_keys=ON')
        cur.execute('PRAGMA journal_mode=WAL')
        cur.execute('PRAGMA busy_timeout=5000')
        cur.execute('PRAGMA synchronous=NORMAL')
    finally:
        cur.close()


event.listen(Engine, 'connect', _configure_sqlite_connection)

# Additive schema migrations. `db.create_all()` creates missing TABLES but never
# ALTERs an existing one, so a column added to a model after the DB was first
# created stays invisible. Each entry is applied idempotently (skipped when the
# column already exists) and is additive only — never a drop. Names/types are
# hardcoded constants (no user input) → safe to interpolate into the ALTER.
_SCHEMA_ADDITIONS = (
    ('face_dataset', 'kind', 'VARCHAR(16)'),
    ('face_dataset', 'subject_type', 'VARCHAR(16)'),
    ('face_dataset', 'concept_desc', 'TEXT'),
    ('face_dataset', 'concept_terms', 'TEXT'),
    ('face_dataset', 'ref_original_filename', 'VARCHAR(255)'),
    ('face_dataset', 'fidelity', 'VARCHAR(8)'),
    ('face_dataset', 'train_settings', 'TEXT'),
    ('face_dataset', 'train_slider', 'TEXT'),
    # Historical rows are LoRA runs. A non-null server default makes the ALTER
    # safe on populated SQLite databases and keeps raw SQL readers truthful.
    ('face_dataset', 'training_mode', "VARCHAR(32) NOT NULL DEFAULT 'lora'"),
    ('face_dataset', 'train_vae_path', 'TEXT'),
    ('face_dataset', 'train_te_path', 'TEXT'),
    # Per-family memory of (base, variant) — see models.FaceDataset. Additive and
    # nullable: a dataset that predates it simply has nothing remembered yet, and
    # keeps the base/variant it already had on the family it already had.
    ('face_dataset', 'train_family_bases', 'TEXT'),
    # Per-family memory of the family-SCOPED train_settings keys (timestep_type).
    # Additive and nullable, same contract: a dataset that predates it has
    # nothing remembered and keeps exactly the settings it already had.
    ('face_dataset', 'train_family_settings', 'TEXT'),
    ('face_dataset', 'prompt_suffix', 'TEXT'),
    ('face_dataset', 'prompt_suffixes', 'TEXT'),
    ('face_dataset', 'caption_options', 'TEXT'),
    ('face_dataset', 'klein_model', 'VARCHAR(255)'),
    # NOTE — `face_dataset.internal` was added here for a feature that was
    # removed before it ever shipped in a release. Databases created while it
    # existed still carry the column; it is nullable, nothing reads or writes it,
    # and it is deliberately NOT dropped: SQLite's DROP COLUMN is unavailable on
    # older engines and would rewrite the table for no gain. Do not reuse the
    # name for anything else — an old database would hand you stale values.
    ('face_dataset_image', 'caption_short', 'TEXT'),
    ('face_dataset_image', 'fail_reason', 'TEXT'),
    # Nature de l'échec ('refused' | 'empty' | 'error') pour compter les refus
    # fournisseur séparément des pannes. Les lignes existantes restent NULL :
    # elles gardent leur phrase, et les compteurs ne les rangent nulle part.
    ('face_dataset_image', 'fail_kind', 'VARCHAR(16)'),
    ('face_dataset_image', 'parent_image_id', 'INTEGER'),
    ('face_dataset_image', 'derivation_kind', 'VARCHAR(32)'),
    ('face_dataset_image', 'upscale_ratio', 'REAL'),
    ('face_dataset_image', 'watermark_state', 'VARCHAR(16)'),
    ('face_dataset_image', 'watermark_bbox', 'TEXT'),
    ('face_dataset_image', 'watermark_regions', 'TEXT'),
    # Who ruled ('detector' | 'vision') and with what score. Existing rows stay
    # NULL — read as "before the source was recorded", never attributed to a
    # route at random (same rule the bank's identical pair already follows).
    ('face_dataset_image', 'watermark_source', 'VARCHAR(16)'),
    ('face_dataset_image', 'watermark_score', 'REAL'),
    ('face_dataset_image', 'source_metadata', 'TEXT'),
    # Back-link to the bank_image a promotion copied here. Existing rows keep
    # NULL: a bank that was promoted before this column existed still relies on
    # its own promoted_dataset_id pointer (see _promotable_query).
    ('face_dataset_image', 'bank_image_id', 'INTEGER'),
    # Cached image content hash (run snapshots). Existing rows stay NULL and are
    # hashed on the next launch that includes them — nothing to backfill, and a
    # database that never gains these columns simply keeps the old behaviour.
    ('face_dataset_image', 'content_sig', 'VARCHAR(24)'),
    ('face_dataset_image', 'content_sig_stat', 'VARCHAR(40)'),
    # Versioned, byte-fingerprinted Bank analysis used by the durable Bank <-> Dataset
    # transfer path. Legacy Dataset rows simply have no snapshot to restore.
    ('face_dataset_image', 'bank_analysis_snapshot', 'TEXT'),
    ('face_dataset_image', 'transfer_metadata', 'TEXT'),
    ('training_run_record', 'settings', 'TEXT'),
    # Full launch freeze: caption text, per-image content hashes, environment.
    # NULL on every run recorded before it existed — the compare panel says so.
    ('training_run_record', 'snapshot', 'TEXT'),
    ('training_run_record', 'parent_record_id', 'INTEGER'),
    ('training_run_record', 'resumed_from', 'INTEGER'),
    ('training_run_record', 'lineage_origin', 'VARCHAR(16)'),
    ('training_run_record', 'note', 'TEXT'),
    ('training_preset', 'dataset_kind', 'VARCHAR(16)'),
    ('training_preset', 'variants', 'TEXT'),
    # Which table `cloud_training_run.dataset_id` points into. Deliberately
    # NULLABLE with no server default: every historical run predates the column
    # and must be READ as 'face_dataset' rather than rewritten by a migration —
    # see services/cloud_run_dataset.table_of, the only reader of this value.
    ('cloud_training_run', 'dataset_table', 'VARCHAR(32)'),
    ('lora_test_image', 'error', 'TEXT'),
    ('lora_test_image', 'resolution_multiplier', 'REAL'),
    # WHICH checkpoint produced this image, written at generation time instead of
    # re-parsed from the filename on every render. Existing rows stay NULL until
    # services.checkpoint_link_backfill attributes the ones it can prove.
    ('lora_test_image', 'record_id', 'INTEGER'),
    ('lora_test_image', 'step', 'INTEGER'),
    # ✨ Upscale & improve run from the ◉ Canvas lightbox: the result is a row of
    # this table (so the board can pin it) that is NOT a Test Studio cell.
    # `derivation_kind` is what every studio query excludes on — see
    # models.LoraTestImage and lora_test_studio._cells(). Existing rows read NULL,
    # which means "an ordinary cell", so a database that predates this keeps
    # behaving exactly as it did.
    ('lora_test_image', 'parent_image_id', 'INTEGER'),
    ('lora_test_image', 'derivation_kind', 'VARCHAR(32)'),
    # Bank V2 scoring pass — the image_bank/bank_image tables shipped in the Beta,
    # so these columns need the additive path (db.create_all never ALTERs an
    # existing table).
    ('bank_image', 'aesthetic_score', 'REAL'),
    ('bank_image', 'nsfw_score', 'REAL'),
    ('bank_image', 'style_cluster', 'INTEGER'),
    ('bank_image', 'watermark_state', 'VARCHAR(16)'),
    ('bank_image', 'caption', 'TEXT'),
    # Structured source provenance survives Bank <-> Dataset transfers. Existing
    # rows remain NULL, just like they were before source attribution was added.
    ('bank_image', 'source_metadata', 'TEXT'),
    ('bank_image', 'semantic_dup_group', 'INTEGER'),
    # The visible semantic_dup_group is a projection of the selected engine.
    # These nullable lanes retain both partitions across engine switches.
    ('bank_image', 'clip_semantic_dup_group', 'INTEGER'),
    ('bank_image', 'siglip2_semantic_dup_group', 'INTEGER'),
    ('bank_image', 'framing', 'VARCHAR(8)'),
    # Bank watermark CLEANING (two manual levels) — the detected bbox is now kept
    # (the scan used to parse it and throw it away) and the cleaned blob's method
    # is recorded. Additive: existing banks keep their rows, they just carry NULLs.
    ('bank_image', 'watermark_bbox', 'TEXT'),
    ('bank_image', 'watermark_clean_method', 'VARCHAR(16)'),
    # Hand-edited watermark mask (JSON list of normalized boxes), the bank's half
    # of the dataset's watermark_regions. Additive: a database that never gains it
    # simply has no hand-edited mask and both levels keep routing on the bbox.
    ('bank_image', 'watermark_regions', 'TEXT'),
    # Which detector produced the watermark verdict, and its raw score. Both stay
    # NULL on every row scanned before the dedicated detector existed — those rows
    # are vision-model verdicts we cannot retro-label, and the panel says
    # "unknown" for them rather than inventing a source.
    ('bank_image', 'watermark_source', 'VARCHAR(16)'),
    ('bank_image', 'watermark_score', 'REAL'),
    # Bank provenance pass — effective resolution, letterbox, JPEG quality and the
    # ai/camera/unknown origin. Same additive path: existing banks keep every row
    # and simply carry NULLs until the next quality scan fills them in.
    ('bank_image', 'detail_ratio', 'REAL'),
    ('bank_image', 'bars_ratio', 'REAL'),
    ('bank_image', 'jpeg_quality', 'REAL'),
    ('bank_image', 'origin', 'VARCHAR(8)'),
    ('bank_image', 'origin_evidence', 'VARCHAR(24)'),
    # 🎨 Medium (what the picture is MADE of) and the confidence gap behind it,
    # plus the face pass's yaw. Additive: a database that never gains them keeps
    # every row and simply reports "not classified" / "not measured".
    ('bank_image', 'medium', 'VARCHAR(16)'),
    ('bank_image', 'medium_margin', 'REAL'),
    ('bank_image', 'face_yaw', 'REAL'),
    # Exact-byte authority shared by every Bank analysis lane.  Existing rows
    # stay NULL and enter the explicit legacy compatibility path until a pass
    # re-attests them; inventing a backfill hash would falsely bless stale data.
    ('bank_image', 'analysis_fingerprint', 'VARCHAR(64)'),
    ('bank_image', 'watermark_fingerprint', 'VARCHAR(64)'),
    # ⬆ Promote's second destination: the bank a selection was copied into.
    # Additive and independent of promoted_dataset_id — a database that never
    # gains it simply never shows the "promoted to a bank" badge.
    ('bank_image', 'promoted_bank_id', 'INTEGER'),
    ('bank_image', 'transfer_metadata', 'TEXT'),
    # Manual quarter-turn of a bank image (degrees clockwise, NULL = untouched).
    # Additive: a database that never gains it simply has no rotated images.
    ('bank_image', 'rotation', 'INTEGER'),
    # ✂ Crop / ✨ Upscale & improve made in the Bank itself: which edit produced
    # the current derived blob, which generation of it is on disk, and the turn
    # that was baked into it. Additive — a database that never gains them simply
    # has no bank-side edits and resolves images exactly as before.
    ('bank_image', 'edit_method', 'VARCHAR(16)'),
    ('bank_image', 'edit_generation', 'INTEGER'),
    ('bank_image', 'edit_baked_rotation', 'INTEGER'),
    # Where a face_cluster id came from: NULL = the embeddings pass computed it
    # (what every existing row means), 'asserted' = a "this subfolder is one
    # person" declaration wrote it with no inference. Additive: a database that
    # never gains it simply has no assertions and clusters exactly as before.
    ('bank_image', 'face_cluster_origin', 'VARCHAR(10)'),
    # WHO wrote a caption: NULL = never recorded | 'asserted' (a human) |
    # 'joycaption'/'ollama' (the engine). Deliberately NO server default: NULL is
    # the value that carries meaning here, and back-filling every existing row
    # with 'joycaption' or with 'asserted' would BOTH be a claim nobody measured
    # — the first would make Re-caption destroy hand-written work it just
    # promised to spare, the second would freeze every bank that exists today.
    # A row that predates the column keeps NULL, is re-captioned as it always
    # was, and is counted on screen as "origin never recorded". See
    # services/caption_origin.py.
    ('bank_image', 'caption_origin', 'VARCHAR(16)'),
    ('face_dataset_image', 'caption_origin', 'VARCHAR(16)'),
    ('face_dataset_image', 'caption_short_origin', 'VARCHAR(16)'),
    ('image_bank', 'pipeline_report', 'TEXT'),
    # When each pass last completed. A bank that predates it keeps NULL and
    # simply reports "never recorded" until its next pass writes a row.
    ('image_bank', 'last_passes', 'TEXT'),
    # Per-Bank semantic engine. The non-null default makes every historical row
    # byte-for-byte compatible with the CLIP behaviour it already had.
    ('image_bank', 'semantic_engine', "VARCHAR(16) NOT NULL DEFAULT 'clip'"),
    # Cloud stop that cannot lie: the moment the user asked for a stop, kept in
    # the database so the supervisor can terminate a pod whose monitor thread
    # never honoured it. Additive — existing runs simply carry NULL.
    ('cloud_training_run', 'stop_requested_at', 'DATETIME'),
    # 🖼🖼 Canvas: which side-by-side group a pinned image is fused into, and
    # where in it. Additive and nullable — a board that predates them simply has
    # no groups on it, and every pinned picture keeps the geometry it had.
    ('canvas_image_node', 'group_id', 'VARCHAR(40)'),
    ('canvas_image_node', 'group_pos', 'INTEGER'),
    # 🎬 Video wave 2: the metrics scan's raw per-clip summary. Additive so a
    # video bank cut by wave 1 keeps every clip and simply reads "not measured".
    ('video_clip', 'metrics_json', 'TEXT'),
    # 🎬 Video wave 3: whether this shot's frames were embedded for 🔎 Search.
    # Additive so a bank cut and measured by the earlier waves keeps every clip
    # and simply reads "not searchable yet" until the pass runs.
    ('video_clip', 'embed_state', 'VARCHAR(12)'),
    # 🎬 Video wave 5: the shot's caption and whether a human has touched it.
    # Additive, so every bank cut by the earlier waves keeps its clips and simply
    # reads "not captioned yet".
    ('video_clip', 'caption', 'TEXT'),
    ('video_clip', 'caption_state', 'VARCHAR(12)'),
    # Which checkpoint wrote the caption. Additive: rows captioned before the
    # model became configurable read NULL, which is honest — nobody recorded it.
    ('video_clip', 'caption_model', 'VARCHAR(120)'),
    ('video_clip', 'caption_style', 'VARCHAR(16)'),
    # 🕸 Scrape → video bank: where a downloaded rush came from. Additive, and
    # NULL is the honest value for every file a user pointed a bank at by hand —
    # their origin was never recorded and must not be invented.
    ('video_source', 'source_metadata', 'TEXT'),
    # 🎬 Find shots v2: the detector's threshold, overridable per bank and per
    # file. NULL means "inherit", which is deliberately NOT the same as 0.0 —
    # every bank that predates this simply keeps the global default it was cut
    # with, and nothing is re-cut behind anyone's back.
    ('video_bank', 'shot_threshold', 'REAL'),
    ('video_source', 'shot_threshold', 'REAL'),
    # Whether this file's per-frame probabilities are cached on disk. A source
    # detected before the cache existed reads NULL, which is exactly right: it
    # cannot be re-cut instantly and the UI offers it a real pass instead.
    ('video_source', 'probs_state', 'VARCHAR(12)'),
    # 🩻 What the container says about the squeeze — the video stream's bit rate
    # and codec profile, read by the same probe that reads the geometry. Both
    # additive and nullable: a bank probed before these existed keeps every
    # number it had and shows a dash for these two until it is re-probed, which
    # is the honest state. NULL is also the CORRECT permanent answer for a
    # container that carries no per-stream bit rate (MKV and WebM routinely do
    # not), so nothing may ever read a NULL here as "not probed yet".
    ('video_source', 'bit_rate', 'INTEGER'),
    ('video_source', 'profile', 'VARCHAR(32)'),
    # Cut or dissolve, and how wide, at each end of a shot. NULL on every clip
    # cut before the detector's second head was kept — no label rather than a
    # guessed one.
    ('video_clip', 'transition_json', 'TEXT'),
)

# Indexes that only a FRESH database ever got. `index=True` on a model column is
# honoured by db.create_all() when it creates the table; on a database that
# already existed, the additive path above adds the COLUMN and nothing else — so
# every install that predates one of these columns has been scanning without its
# index ever since. These are exactly the _SCHEMA_ADDITIONS columns declared
# index=True in models.py, under SQLAlchemy's own default name (ix_<table>_<col>),
# so a fresh database finds them already there and does nothing.
_INDEX_ADDITIONS = (
    ('face_dataset_image', 'bank_image_id'),
    ('bank_image', 'semantic_dup_group'),
    ('bank_image', 'style_cluster'),
    ('bank_image', 'framing'),
    ('bank_image', 'origin'),
    ('bank_image', 'medium'),
    ('lora_test_image', 'record_id'),
    ('lora_test_image', 'parent_image_id'),
)


def _apply_additive_migrations():
    from sqlalchemy import text
    for table, col, col_type in _SCHEMA_ADDITIONS:
        try:
            existing = {row[1] for row in db.session.execute(text(f'PRAGMA table_info({table})'))}
            if col not in existing:
                db.session.execute(text(f'ALTER TABLE {table} ADD COLUMN {col} {col_type}'))
                db.session.commit()
        except Exception:
            db.session.rollback()  # a failed ALTER must never block boot
    # Before the engine selector there was only ``semantic_dup_group`` and its
    # space was necessarily CLIP.  Copy that visible partition into the durable
    # lane (or into SigLIP2 for Banks already switched by an intermediate build)
    # so the first switch after this migration cannot erase existing work.  The
    # NULL guard makes this safe and idempotent on every boot.
    try:
        db.session.execute(text("""
            UPDATE bank_image
               SET clip_semantic_dup_group = semantic_dup_group
             WHERE clip_semantic_dup_group IS NULL
               AND semantic_dup_group IS NOT NULL
               AND bank_id IN (
                   SELECT id FROM image_bank
                    WHERE LOWER(TRIM(COALESCE(semantic_engine, 'clip')))
                          != 'siglip2'
               )
        """))
        db.session.execute(text("""
            UPDATE bank_image
               SET siglip2_semantic_dup_group = semantic_dup_group
             WHERE siglip2_semantic_dup_group IS NULL
               AND semantic_dup_group IS NOT NULL
               AND bank_id IN (
                   SELECT id FROM image_bank
                    WHERE LOWER(TRIM(COALESCE(semantic_engine, 'clip')))
                          = 'siglip2'
               )
        """))
        db.session.commit()
    except Exception:
        db.session.rollback()  # legacy backfill is best-effort, boot stays open
    # Same loop, same discipline: idempotent (IF NOT EXISTS), additive only, and
    # fail-open — a database that cannot take an index still boots, just slower.
    for table, col in _INDEX_ADDITIONS:
        try:
            db.session.execute(text(
                f'CREATE INDEX IF NOT EXISTS ix_{table}_{col} ON {table} ({col})'))
            db.session.commit()
        except Exception:
            db.session.rollback()


def _cleanup_orphaned_lora_test_images():
    """Remove Studio rows left by legacy databases without enforced FKs.

    New databases cascade these rows and the service explicitly removes them,
    but old releases could leave them behind after deleting their dataset.
    ``NOT EXISTS`` only targets rows whose parent is provably absent.
    """
    from sqlalchemy import text
    # A legacy Studio row may still own a live queue job. Cancel only when the
    # linkage is unambiguous: exact job_id plus the Studio metadata and matching
    # dataset_id. Unknown/mismatched jobs are deliberately left untouched; a
    # bare legacy job_id is not enough authority to cancel unrelated work.
    columns = {
        row[1] for row in db.session.execute(text(
            'PRAGMA table_info(lora_test_image)'))
    }
    cancelled_jobs = 0
    if 'job_id' in columns:
        orphan_links = db.session.execute(text(
            'SELECT job_id, dataset_id FROM lora_test_image '
            'WHERE job_id IS NOT NULL AND NOT EXISTS ('
            'SELECT 1 FROM face_dataset '
            'WHERE face_dataset.id = lora_test_image.dataset_id)'
        )).all()
        if orphan_links:
            from .models import ImageGenerationQueue
            for job_id, dataset_id in orphan_links:
                job = ImageGenerationQueue.query.filter_by(job_id=job_id).first()
                if not job or job.status in ('completed', 'failed', 'cancelled'):
                    continue
                try:
                    metadata = json.loads(job.job_metadata or '{}')
                except (TypeError, ValueError):
                    metadata = {}
                if not (metadata.get('is_lora_test') is True
                        and metadata.get('model_name') == 'zimage_lora_test'
                        and str(metadata.get('dataset_id')) == str(dataset_id)):
                    continue
                job.update_status('cancelled')
                cancelled_jobs += 1
    result = db.session.execute(text(
        'DELETE FROM lora_test_image '
        'WHERE NOT EXISTS ('
        'SELECT 1 FROM face_dataset '
        'WHERE face_dataset.id = lora_test_image.dataset_id)'
    ))
    db.session.commit()
    if result.rowcount is not None and result.rowcount > 0:
        logger.warning('startup cleanup removed %d orphaned LoRA Studio row(s)',
                       result.rowcount)
    if cancelled_jobs:
        logger.warning('startup cleanup cancelled %d safely-linked Studio job(s)',
                       cancelled_jobs)

def create_app(config_object=None):
    # Re-assert the static content types: cheap, idempotent, and it survives a
    # `mimetypes.init()` run by any library imported since this module loaded.
    pin_static_mime_types()
    app = Flask(__name__, static_folder=None)
    app.request_class = ArchiveAwareRequest
    data_dir = Path(os.environ.get('LDS_DATA_DIR', str(cfg.REPO_ROOT / 'data')))
    data_dir.mkdir(parents=True, exist_ok=True)
    app.config.update(
        SECRET_KEY=cfg.secret_key(),
        # The resolved data folder, for anything that reasons about the folder
        # itself rather than the database in it — run.py's one-server-per-data-
        # folder lock is the first customer.
        LDS_DATA_DIR=str(data_dir),
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{data_dir / 'studio.db'}",
        SQLALCHEMY_ENGINE_OPTIONS={'connect_args': {'check_same_thread': False}},
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,
        # ZIP imports legitimately exceed the ordinary upload ceiling.  The exact
        # archive-file cap stays separate from multipart framing overhead so the
        # route can enforce the former after Werkzeug has spooled the upload.
        DATASET_ARCHIVE_MAX_UPLOAD_BYTES=_positive_env_int(
            'LDS_DATASET_ARCHIVE_MAX_UPLOAD_BYTES',
            _DEFAULT_DATASET_ARCHIVE_MAX_UPLOAD_BYTES),
        DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES=(
            _DEFAULT_DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES),
        DATASET_ARCHIVE_SPOOL_MEMORY_BYTES=8 * 1024 * 1024,
    )
    app.config.update(config_object or {})

    # File logging (skipped under TESTING): every module logger flows into
    # data/app.log (rotating, 2 MB x 2) so the in-app log viewer — and a novice
    # reporting a bug — always has something to read, launcher or not (the
    # portable launcher additionally captures raw stdout into data/server.log).
    if not app.config.get('TESTING'):
        import logging
        from logging.handlers import RotatingFileHandler
        root = logging.getLogger()
        log_path = str(data_dir / 'app.log')
        if not any(isinstance(h, RotatingFileHandler)
                   and getattr(h, 'baseFilename', '') == os.path.abspath(log_path)
                   for h in root.handlers):
            fh = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024,
                                     backupCount=2, encoding='utf-8')
            fh.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s %(name)s: %(message)s'))
            fh.setLevel(logging.INFO)
            root.addHandler(fh)
            if root.level > logging.INFO or root.level == logging.NOTSET:
                root.setLevel(logging.INFO)

    db.init_app(app)
    csrf.init_app(app)

    from werkzeug.exceptions import RequestEntityTooLarge

    @app.errorhandler(RequestEntityTooLarge)
    def _request_entity_too_large(error):
        if not request.path.startswith('/api/'):
            return error
        if request.endpoint in _DATASET_ARCHIVE_UPLOAD_ENDPOINTS:
            limit = int(app.config['DATASET_ARCHIVE_MAX_UPLOAD_BYTES'])
            return jsonify({
                'ok': False,
                'error': f'archive too large (maximum {limit // (1024 * 1024)} MiB)',
            }), 413
        return jsonify({'ok': False, 'error': 'upload too large'}), 413

    with app.app_context():
        from . import models  # noqa: F401
        db.create_all()
        _apply_additive_migrations()
        _cleanup_orphaned_lora_test_images()
        # Reconnect continuations that ran before their lineage edge was persisted
        # — once, best-effort, never blocks boot (see services.lineage_backfill).
        from .services.lineage_backfill import run_if_needed as _lineage_backfill
        _lineage_backfill()
        # Give back their framing to the images promoted from a bank before the
        # promotion carried it, so they count in Composition instead of sitting
        # invisible — once, best-effort (see services.framing_backfill).
        from .services.framing_backfill import run_if_needed as _framing_backfill
        _framing_backfill()
        # Let a checkpoint KEEP every preview it is given instead of one (the
        # canvas shows a gallery under each node). Recreating the table is the
        # only step of that feature that can lose rows silently, so it is guarded
        # by a before/after row count and leaves the original in place on any
        # doubt (see services.checkpoint_preview_migration).
        from .services.checkpoint_preview_migration import (
            run_if_needed as _lift_preview_constraint)
        _lift_preview_constraint()
        # Attach the test images already on disk to the checkpoint that produced
        # them, from the evidence only — the rest stays honestly unlinked
        # (see services.checkpoint_link_backfill).
        from .services.checkpoint_link_backfill import (
            run_if_needed as _checkpoint_link_backfill)
        _checkpoint_link_backfill()
        # Vision requests are process-local, while their mutual-exclusion flag is
        # persisted in SQLite. A killed captioning request therefore cannot still
        # be running after boot; clear its stale flag immediately instead of
        # leaving the restarted app stuck on "GPU busy" until the TTL expires.
        from .gpu_window import recover_stale_vision_window
        recover_stale_vision_window()
        # Move cloud checkpoints out of the disposable staging dirs and into the
        # durable store. Until this has run, an install trained before the store
        # existed still keeps its ONLY copy of a never-deployed .safetensors in a
        # directory the cleanup is allowed to trash (see services.cloud_training).
        # Once, guarded by a persisted flag, and swallows its own failures.
        try:
            from .services.cloud_training import migrate_checkpoints_into_store
            migrate_checkpoints_into_store()
        except Exception:
            app.logger.exception('checkpoint store retrofit skipped')

    from .routes import register_blueprints
    register_blueprints(app, csrf)

    # Non-loopback clients must present the access token (run.py generates one
    # when the bind is opened) — without this, `server.host: 0.0.0.0` would hand
    # the whole LAN the API keys, the GPU and the datasets. Loopback = untouched.
    from .netguard import install_network_guard
    install_network_guard(app)

    # Registered last of the write-path guards, so a caller still has to clear CSRF
    # and the access token before we tell them anything about their own body.
    from .routes._common import reject_unparsable_json_body
    app.before_request(reject_unparsable_json_body)

    @app.get('/api/health')
    def health():
        return {'ok': True}

    @app.get('/api/csrf-token')
    def csrf_token():
        from flask_wtf.csrf import generate_csrf
        return jsonify({'csrf_token': generate_csrf()})

    @app.get('/')
    def index():
        if not FRONTEND_DIST.exists():
            return jsonify({'error': 'frontend not built — run npm run build in frontend/'}), 503
        # The csrf_token cookie is (re)planted by the after_request hook below —
        # which covers '/' AND every /api response — so a SPA session can no longer
        # outlive its token (see _refresh_csrf_cookie for the full rationale).
        return send_from_directory(FRONTEND_DIST, 'index.html')

    @app.get('/assets/<path:filename>')
    def assets(filename):
        return send_from_directory(FRONTEND_DIST / 'assets', filename)

    @app.after_request
    def _refresh_csrf_cookie(resp):
        # Flask-WTF's CSRF token is time-limited (WTF_CSRF_TIME_LIMIT, default 1 h).
        # Historically the cookie was planted ONLY on GET / — so a SPA tab left open
        # past that limit kept echoing a now-expired token, and every Save/Test POST
        # came back as a cryptic HTML 400 that only a hard refresh cleared. Re-plant a
        # freshly-timestamped token on the app shell and on every /api response (static
        # assets are skipped — pure noise): any request the SPA makes keeps the cookie
        # alive, and even the CSRF-rejection 400 itself carries a fresh cookie so the
        # client's one-shot retry lands on a valid token with no reload. This also
        # covers the Vite dev server, which proxies only /api (Flask never sees GET /,
        # so the cookie was never planted there at all).
        #
        # httponly stays False (the default) so the SPA can read the cookie and echo
        # it back in the X-CSRFToken header; samesite='Lax' mirrors the original
        # GET / cookie; no `secure` flag (the app is reached over plain http on
        # loopback/LAN). after_request runs BEFORE save_session, so a first-ever
        # session gets its csrf secret persisted alongside this cookie.
        if request.path == '/' or request.path.startswith('/api'):
            from flask_wtf.csrf import generate_csrf
            resp.set_cookie('csrf_token', generate_csrf(), samesite='Lax')
        return resp

    if not app.config.get('TESTING'):
        _start_workers(app)
    return app

def _start_workers(app):
    """Boot background machinery. Idempotent; nothing GPU-ish is required."""
    from .job_queue import queue_manager
    queue_manager.init_app(app)
    queue_manager.start()
    try:
        from .services.lora_training import start_training_scheduler
        start_training_scheduler(app)
    except ImportError:
        pass  # phase(<3): training service not lifted yet

    import threading
    from .services import cloud_training
    threading.Thread(target=cloud_training.boot_recover, args=(app,),
                     daemon=True, name='cloud-boot-recover').start()
    # Started separately from boot_recover on purpose: the watchdog that
    # enforces the runtime cap, the stop deadline and the freeze detection must
    # not share a fate with the recovery it supervises.
    cloud_training.start_supervisor(app)
