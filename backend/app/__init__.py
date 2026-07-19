import os
import logging
import sqlite3
import json
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request
from sqlalchemy import event
from sqlalchemy.engine import Engine
from .extensions import db, csrf
from . import config as cfg

FRONTEND_DIST = cfg.REPO_ROOT / 'frontend' / 'dist'
logger = logging.getLogger(__name__)

_DEFAULT_DATASET_ARCHIVE_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_DATASET_ARCHIVE_UPLOAD_ENDPOINTS = frozenset({
    'datasets.dataset_backup_import',
    'datasets.dataset_import_zip',
    # 'Back up everything' restore accepts a master archive that can be as large
    # as the whole library — it needs the same raised request ceiling.
    'backup.full_restore',
})


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
    ('face_dataset', 'concept_desc', 'TEXT'),
    ('face_dataset', 'concept_terms', 'TEXT'),
    ('face_dataset', 'ref_original_filename', 'VARCHAR(255)'),
    ('face_dataset', 'fidelity', 'VARCHAR(8)'),
    ('face_dataset', 'train_settings', 'TEXT'),
    ('face_dataset', 'train_slider', 'TEXT'),
    ('face_dataset', 'train_vae_path', 'TEXT'),
    ('face_dataset', 'train_te_path', 'TEXT'),
    ('face_dataset', 'prompt_suffix', 'TEXT'),
    ('face_dataset', 'prompt_suffixes', 'TEXT'),
    ('face_dataset', 'caption_options', 'TEXT'),
    ('face_dataset_image', 'caption_short', 'TEXT'),
    ('face_dataset_image', 'fail_reason', 'TEXT'),
    ('face_dataset_image', 'parent_image_id', 'INTEGER'),
    ('face_dataset_image', 'derivation_kind', 'VARCHAR(32)'),
    ('face_dataset_image', 'upscale_ratio', 'REAL'),
    ('face_dataset_image', 'watermark_state', 'VARCHAR(16)'),
    ('face_dataset_image', 'watermark_bbox', 'TEXT'),
    ('face_dataset_image', 'watermark_regions', 'TEXT'),
    ('face_dataset_image', 'source_metadata', 'TEXT'),
    ('training_run_record', 'settings', 'TEXT'),
    ('training_preset', 'dataset_kind', 'VARCHAR(16)'),
    ('training_preset', 'variants', 'TEXT'),
    ('lora_test_image', 'error', 'TEXT'),
    # Bank V2 scoring pass — the image_bank/bank_image tables shipped in the Beta,
    # so these columns need the additive path (db.create_all never ALTERs an
    # existing table).
    ('bank_image', 'aesthetic_score', 'REAL'),
    ('bank_image', 'nsfw_score', 'REAL'),
    ('bank_image', 'style_cluster', 'INTEGER'),
    ('bank_image', 'watermark_state', 'VARCHAR(16)'),
    ('bank_image', 'caption', 'TEXT'),
    ('image_bank', 'pipeline_report', 'TEXT'),
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
    app = Flask(__name__, static_folder=None)
    data_dir = Path(os.environ.get('LDS_DATA_DIR', str(cfg.REPO_ROOT / 'data')))
    data_dir.mkdir(parents=True, exist_ok=True)
    app.config.update(
        SECRET_KEY=cfg.secret_key(),
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

    # Flask-WTF looks in request.form before it checks the CSRF header.  For a
    # multipart upload that parses the body in its before_request hook, before the
    # view itself can raise the limit.  Register this override first so ONLY the
    # two archive endpoints may exceed the ordinary 64 MiB request ceiling.
    @app.before_request
    def _set_dataset_archive_request_limit():
        if request.endpoint in _DATASET_ARCHIVE_UPLOAD_ENDPOINTS:
            archive_max = int(app.config['DATASET_ARCHIVE_MAX_UPLOAD_BYTES'])
            overhead = max(0, int(
                app.config['DATASET_ARCHIVE_MULTIPART_OVERHEAD_BYTES']))
            request.max_content_length = archive_max + overhead

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
        # Vision requests are process-local, while their mutual-exclusion flag is
        # persisted in SQLite. A killed captioning request therefore cannot still
        # be running after boot; clear its stale flag immediately instead of
        # leaving the restarted app stuck on "GPU busy" until the TTL expires.
        from .gpu_window import recover_stale_vision_window
        recover_stale_vision_window()

    from .routes import register_blueprints
    register_blueprints(app, csrf)

    # Non-loopback clients must present the access token (run.py generates one
    # when the bind is opened) — without this, `server.host: 0.0.0.0` would hand
    # the whole LAN the API keys, the GPU and the datasets. Loopback = untouched.
    from .netguard import install_network_guard
    install_network_guard(app)

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
