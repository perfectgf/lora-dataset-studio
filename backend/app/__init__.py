import os
from pathlib import Path
from flask import Flask, send_from_directory, jsonify
from sqlalchemy import event
from .extensions import db, csrf
from . import config as cfg

FRONTEND_DIST = cfg.REPO_ROOT / 'frontend' / 'dist'

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

    from .routes import register_blueprints
    register_blueprints(app, csrf)

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
    try:
        from .job_queue import queue_manager
        queue_manager.init_app(app)
        queue_manager.start()
    except ImportError:
        pass  # phase(<12): job queue not lifted yet
    try:
        from .services.lora_training import start_training_scheduler
        start_training_scheduler(app)
    except ImportError:
        pass  # phase(<3): training service not lifted yet
