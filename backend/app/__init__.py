import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify
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
    ('face_dataset_image', 'fail_reason', 'TEXT'),
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
        from flask_wtf.csrf import generate_csrf
        if not FRONTEND_DIST.exists():
            return jsonify({'error': 'frontend not built — run npm run build in frontend/'}), 503
        resp = send_from_directory(FRONTEND_DIST, 'index.html')
        resp.set_cookie('csrf_token', generate_csrf(), samesite='Lax')
        return resp

    @app.get('/assets/<path:filename>')
    def assets(filename):
        return send_from_directory(FRONTEND_DIST / 'assets', filename)

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
