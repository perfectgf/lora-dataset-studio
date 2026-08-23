"""The generic extension loader: optional packages dropped into
backend/extensions/ (or LDS_EXTENSIONS_DIR) register themselves at boot.
The dir is gitignored and never shipped; with it absent the loader is a no-op.
"""
import textwrap


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


def test_the_network_guard_outranks_extension_hooks(tmp_path, monkeypatch):
    """Extensions load AFTER the network guard installs, so a hook an extension
    registers can never answer a request the token gate would have refused —
    before_request hooks run in registration order. Extensions are trusted
    local code either way; this keeps a public bind's front door in front."""
    ext_dir = tmp_path / 'exts'
    _write_ext(ext_dir, 'demo_ext_door', '''
        from flask import jsonify, request

        def register(app, csrf):
            @app.before_request
            def wave_health_through():
                if request.path == '/api/health':
                    return jsonify({'open': True})
    ''')
    application = _make_app(tmp_path, monkeypatch, ext_dir)
    monkeypatch.setenv('LDS_ACCESS_TOKEN', 'sekret')
    c = application.test_client()
    # Loopback turns the gate on; the extension's hook happily answers loopback.
    c.put('/api/settings', json={'config': {'server': {'require_token': True}}})
    remote = {'REMOTE_ADDR': '192.168.1.50'}
    refused = c.get('/api/health', environ_base=remote)
    assert refused.status_code == 403          # the gate spoke first
    allowed = c.get('/api/health', environ_base=remote,
                    headers={'Authorization': 'Bearer sekret'})
    assert allowed.status_code == 200
    assert allowed.get_json() == {'open': True}  # then the extension answers
