import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, request
from sqlalchemy import event
from .extensions import db, csrf
from . import config as cfg

FRONTEND_DIST = cfg.REPO_ROOT / 'frontend' / 'dist'

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
    ('face_dataset', 'train_vae_path', 'TEXT'),
    ('face_dataset', 'train_te_path', 'TEXT'),
    ('face_dataset_image', 'fail_reason', 'TEXT'),
    ('face_dataset_image', 'upscale_ratio', 'REAL'),
    ('face_dataset_image', 'watermark_state', 'VARCHAR(16)'),
    ('face_dataset_image', 'watermark_bbox', 'TEXT'),
    ('face_dataset_image', 'watermark_regions', 'TEXT'),
    ('training_run_record', 'settings', 'TEXT'),
    ('lora_test_image', 'error', 'TEXT'),
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

def create_app(config_object=None):
    app = Flask(__name__, static_folder=None)
    data_dir = Path(os.environ.get('LDS_DATA_DIR', str(cfg.REPO_ROOT / 'data')))
    data_dir.mkdir(parents=True, exist_ok=True)
    app.config.update(
        SECRET_KEY=cfg.secret_key(),
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{data_dir / 'studio.db'}",
        SQLALCHEMY_ENGINE_OPTIONS={'connect_args': {'check_same_thread': False}},
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,
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

    with app.app_context():
        @event.listens_for(db.engine, 'connect')
        def _sqlite_pragmas(dbapi_con, _):
            cur = dbapi_con.cursor()
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA busy_timeout=5000')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.close()
        from . import models  # noqa: F401
        db.create_all()
        _apply_additive_migrations()
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
