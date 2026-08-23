"""The generic extension loader: optional packages dropped into
backend/extensions/ (or LDS_EXTENSIONS_DIR) register themselves at boot.
The dir is gitignored and never shipped; with it absent the loader is a no-op.
"""
import textwrap

import pytest


def _write_ext(base, name, body):
    pkg = base / name
    pkg.mkdir(parents=True)
    (pkg / '__init__.py').write_text(textwrap.dedent(body), encoding='utf-8')


def _make_app(tmp_path, monkeypatch, ext_dir):
    # Same minimal env isolation as conftest's `app` fixture, plus the
    # extension dir under test. Built here because the env has to be in place
    # BEFORE create_app runs (the fixture's app is created too early for that).
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    monkeypatch.setenv('LDS_EXTENSIONS_DIR', str(ext_dir))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(_cfg, '_cache', None)
    from app import create_app
    return create_app({'TESTING': True, 'WTF_CSRF_ENABLED': False,
                       'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})


GOOD_EXT = '''
    from flask import Blueprint, jsonify

    __version__ = '0.0.1'
    FRONTEND_ENTRY = '/api/demo-ext/ui.js'

    bp = Blueprint('demo_ext', __name__, url_prefix='/api/demo-ext')

    @bp.get('/ping')
    def ping():
        return jsonify({'ok': True})

    def register(app, csrf):
        app.register_blueprint(bp)
'''


def test_a_dropped_in_package_registers_its_routes(tmp_path, monkeypatch):
    ext_dir = tmp_path / 'exts'
    _write_ext(ext_dir, 'demo_ext_a', GOOD_EXT.replace('demo_ext', 'demo_ext_a').replace('demo-ext', 'demo-ext-a'))
    application = _make_app(tmp_path, monkeypatch, ext_dir)
    resp = application.test_client().get('/api/demo-ext-a/ping')
    assert resp.status_code == 200
    assert resp.get_json() == {'ok': True}


def test_the_manifest_records_name_version_and_frontend_entry(tmp_path, monkeypatch):
    ext_dir = tmp_path / 'exts'
    _write_ext(ext_dir, 'demo_ext_b', GOOD_EXT.replace('demo_ext', 'demo_ext_b').replace('demo-ext', 'demo-ext-b'))
    application = _make_app(tmp_path, monkeypatch, ext_dir)
    assert application.config['EXTENSIONS_MANIFEST'] == [{
        'name': 'demo_ext_b',
        'version': '0.0.1',
        'frontend_entry': '/api/demo-ext-b/ui.js',
    }]


def test_a_broken_extension_is_skipped_and_the_app_still_boots(tmp_path, monkeypatch):
    ext_dir = tmp_path / 'exts'
    _write_ext(ext_dir, 'demo_ext_broken', '''
        def register(app, csrf):
            raise RuntimeError('boom')
    ''')
    application = _make_app(tmp_path, monkeypatch, ext_dir)
    assert application.config['EXTENSIONS_MANIFEST'] == []
    # and the app answers on a known route
    assert application.test_client().get('/api/extensions/').status_code in (200, 404)


def test_the_kill_switch_disables_loading(tmp_path, monkeypatch):
    ext_dir = tmp_path / 'exts'
    _write_ext(ext_dir, 'demo_ext_c', GOOD_EXT.replace('demo_ext', 'demo_ext_c').replace('demo-ext', 'demo-ext-c'))
    monkeypatch.setenv('LDS_EXTENSIONS', '0')
    application = _make_app(tmp_path, monkeypatch, ext_dir)
    assert application.config['EXTENSIONS_MANIFEST'] == []
    assert application.test_client().get('/api/demo-ext-c/ping').status_code == 404


def test_a_missing_dir_is_a_silent_no_op(tmp_path, monkeypatch):
    application = _make_app(tmp_path, monkeypatch, tmp_path / 'does-not-exist')
    assert application.config['EXTENSIONS_MANIFEST'] == []
