"""Public framework foundations, exercised without loading any shipped product."""
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import uuid
import venv
import zipfile

from flask import Flask
from flask_wtf.csrf import CSRFProtect
import pytest

from app import config as cfg
from app.engines import registry as engines
from app.extensions import db
from app.plugins import environment, registry, storage, transactions
from app.plugins.api import ConfigView, LDS_PLUGIN_API_MINOR, PluginContext
from app.plugins.compatibility import host_issues
from app.plugins.discovery import development_bundles
from app.plugins.install import ArchiveError, inspect_plugin_zip
from app.plugins.loader import load_plugins
from app.plugins.manifest import ManifestError, parse_manifest


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    monkeypatch.setenv('LDS_EXTENSIONS', '0')
    monkeypatch.setenv('LDS_PLUGIN_DISTRIBUTION', 'store')
    monkeypatch.delenv('LDS_BUNDLED_DIR', raising=False)
    monkeypatch.delenv('LDS_PLUGINS_DIR', raising=False)
    monkeypatch.delenv('LDS_PLUGINS', raising=False)
    monkeypatch.setattr(cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(cfg, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(cfg, 'DEFAULTS', deepcopy(cfg.DEFAULTS))
    monkeypatch.setattr(cfg, '_cache', None)
    monkeypatch.setattr(registry, '_ACTIVE', None)
    monkeypatch.setattr(engines, '_specs', {})
    # This suite has no reason to contact a service, model server or index.
    monkeypatch.setattr('requests.sessions.Session.request',
                        lambda *_a, **_k: pytest.fail('Unexpected network request'))
    original_path = list(sys.path)
    app = Flask(__name__)
    app.config.update(TESTING=True, SQLALCHEMY_DATABASE_URI='sqlite:///:memory:',
                      WTF_CSRF_ENABLED=False, SECRET_KEY='test-only')
    db.init_app(app)
    csrf = CSRFProtect(app)
    yield app, csrf, tmp_path
    with app.app_context():
        db.session.remove()
    sys.path[:] = original_path
    for name in list(sys.modules):
        if name.startswith('lds_foundation_'):
            sys.modules.pop(name, None)


def manifest(pid='sample.feature', **changes):
    return {'id': pid, 'name': 'Sample feature', 'version': '1.0.0', 'api': 1,
            'requires': [], 'owns': {}, **changes}


def write(host, pid='sample.feature', *, code='def register(ctx):\n    pass\n', **changes):
    _, _, root = host
    directory = root / 'data' / 'plugins' / pid
    directory.mkdir(parents=True)
    package = 'lds_foundation_' + pid.replace('.', '_')
    (directory / 'plugin.json').write_text(json.dumps(manifest(pid, python_package=package, **changes)))
    (directory / package).mkdir()
    (directory / package / '__init__.py').write_text(textwrap.dedent(code), encoding='utf-8')
    return directory


PING = '''
    from flask import Blueprint
    bp = Blueprint('sample', __name__)
    @bp.get('/ping')
    def ping():
        return {'ready': True}
    def register(ctx):
        ctx.register_blueprint(bp)
        ctx.register_config_defaults({'message': 'hello'})
        ctx.register_probe('sample_ready', lambda: True)
'''


def test_manual_loader_registers_neutral_route_defaults_and_probe(host):
    app, csrf, _ = host
    write(host, code=PING, owns={'probes': ['sample_ready']})
    assert 'lds_plugins' not in app.extensions
    loaded = load_plugins(app, csrf)
    assert loaded.records['sample.feature'].state == 'loaded'
    assert app.test_client().get('/api/plugins/sample.feature/ping').get_json() == {'ready': True}
    assert ConfigView('sample.feature').own('message') == 'hello'
    assert loaded.probes['sample_ready'][1]() is True
    assert load_plugins(app, csrf) is loaded
    assert len([r for r in app.url_map.iter_rules() if r.rule.endswith('/ping')]) == 1


@pytest.mark.parametrize('reason', ['disabled', 'missing', 'incompatible', 'kill'])
def test_refused_plugin_is_never_imported(host, reason, monkeypatch):
    app, csrf, root = host
    extra = {'requires': ['sample.absent']} if reason == 'missing' else {}
    if reason == 'incompatible':
        extra['api'] = 2
    write(host, code="raise AssertionError('Package must not be imported')", **extra)
    if reason == 'disabled':
        (root / 'config.json').write_text(json.dumps({'plugins': {'enabled': {'sample.feature': False}}}))
    if reason == 'kill':
        monkeypatch.setenv('LDS_PLUGINS', '0')
    loaded = load_plugins(app, csrf)
    assert 'lds_foundation_sample_feature' not in sys.modules
    if reason == 'kill':
        assert not loaded.records
    else:
        assert loaded.records['sample.feature'].state == ('incompatible' if reason == 'incompatible' else 'disabled')


def test_bad_internal_import_is_visible_and_dependant_not_imported(host):
    app, csrf, _ = host
    write(host, 'sample.base', code='import foundation_missing_internal_module')
    write(host, requires=['sample.base'], code="raise AssertionError('Dependent must not import')")
    loaded = load_plugins(app, csrf)
    assert loaded.records['sample.base'].state == 'error'
    assert 'foundation_missing_internal_module' in loaded.records['sample.base'].error
    assert loaded.records['sample.feature'].disabled_by == ['sample.base']
    assert 'lds_foundation_sample_feature' not in sys.modules


def test_conflicting_ownership_is_rejected_before_second_package_import(host):
    app, csrf, _ = host
    write(host, 'sample.alpha', owns={'probes': ['shared_ready']})
    write(host, 'sample.bravo', owns={'probes': ['shared_ready']},
          code="raise AssertionError('Conflicting plugin must not import')")
    loaded = load_plugins(app, csrf)
    assert loaded.records['sample.alpha'].state == 'loaded'
    assert loaded.records['sample.bravo'].state == 'error'
    assert 'lds_foundation_sample_bravo' not in sys.modules


def test_failed_registration_rolls_back_route_defaults_probe_and_engine(host):
    app, csrf, _ = host
    write(host, code=textwrap.dedent(PING) + textwrap.dedent('''
        original_register = register
        def register(ctx):
            original_register(ctx)
            ctx.register_engine(id='sample_engine', label='Sample', kind='local', order=1)
            raise RuntimeError('Registration failed after contributions')
    '''), owns={'probes': ['sample_ready'], 'engines': ['sample_engine']})
    loaded = load_plugins(app, csrf)
    assert loaded.records['sample.feature'].state == 'error'
    assert app.test_client().get('/api/plugins/sample.feature/ping').status_code == 404
    assert 'sample_ready' not in loaded.probes
    assert 'sample.feature' not in cfg.DEFAULTS['plugins']
    assert engines.get('sample_engine') is None


def test_config_updates_stay_under_namespaced_owner(host):
    view = ConfigView('sample.feature')
    view.update_own({'server': {'port': 7}, 'enabled': False})
    assert view.own('server.port') == 7
    assert cfg.get('server.port') != 7
    assert cfg.get('plugins.enabled') == {}


def test_managed_environment_stays_unprepared_until_explicit_installation(host):
    app, csrf, root = host
    directory = write(host, requirements='requirements.txt')
    (directory / 'requirements.txt').write_text('sample==1.0')
    loaded = load_plugins(app, csrf)
    record = loaded.records['sample.feature']
    context = PluginContext(app, loaded, record.manifest, root / 'data/plugin-data/sample.feature')
    state = context.plugin_environment()
    assert state['ready'] is False and state['can_install'] is True
    assert state['reason']
    from app import setup_installer
    assert environment.installer() is setup_installer
    assert setup_installer.known_action('plugin_environment:sample.feature')
    assert not environment.environment_dir(record).exists()


def test_environment_file_check_finds_deleted_and_zeroed_package_files(tmp_path):
    # pip keeps a same-version package whose files are gone, so Repair relies
    # on this RECORD comparison, run by the plugin interpreter itself.
    from app.setup_installer import _venv_python
    venv.create(tmp_path / 'env', with_pip=False)
    python = _venv_python(tmp_path / 'env')
    site = Path(subprocess.run([python, '-I', '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                               capture_output=True, text=True, check=True).stdout.strip())
    files = {'lds_probe_demo/__init__.py': b'VALUE = 1\n', 'lds_probe_demo/weights.bin': b'\x01' * 4096}
    for name, data in files.items():
        (site / name).parent.mkdir(exist_ok=True)
        (site / name).write_bytes(data)
    info = site / 'lds_probe_demo-1.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: lds-probe-demo\nVersion: 1.0\n')
    rows = [f'{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()},{len(data)}'
            for name, data in files.items()]
    # Unhashed rows (bytecode, RECORD itself) may legitimately be absent.
    rows += ['lds_probe_demo/__pycache__/__init__.cpython-312.pyc,,', 'lds_probe_demo-1.0.dist-info/RECORD,,']
    (info / 'RECORD').write_text('\n'.join(rows) + '\n')

    def states():
        return environment._package_states(python, ['lds-probe-demo', 'not-installed'])

    assert states() == {'lds-probe-demo': 'ok', 'not-installed': 'absent'}
    (site / 'lds_probe_demo/weights.bin').write_bytes(b'\0' * 4096)
    assert states()['lds-probe-demo'] == 'damaged'
    (site / 'lds_probe_demo/weights.bin').write_bytes(files['lds_probe_demo/weights.bin'])
    (site / 'lds_probe_demo/__init__.py').unlink()
    assert states()['lds-probe-demo'] == 'damaged'


def test_netfetch_uses_existing_public_host_primitive(host):
    app, csrf, root = host
    write(host)
    loaded = load_plugins(app, csrf)
    context = PluginContext(app, loaded, loaded.records['sample.feature'].manifest, root / 'data')
    from app.scrape import netfetch
    assert context.netfetch is netfetch


def test_store_discovery_does_not_load_bundled_sources(host, monkeypatch):
    app, csrf, root = host
    bundled = root / 'bundled' / 'demo'
    bundled.mkdir(parents=True)
    (bundled / 'plugin.json').write_text(json.dumps(manifest('demo', bundled=True)))
    assert development_bundles() is False
    assert not load_plugins(app, csrf).records
    monkeypatch.setenv('LDS_PLUGIN_DISTRIBUTION', 'development')
    assert development_bundles() is True


def test_invalid_managed_nodes_rejected_without_importing_worker(host):
    value = manifest(node_packs=[{'pack': 'Sample nodes', 'classes': ['Sample Node'],
                                 'url': 'https://example.test/nodes',
                                 'installation': {}}])
    before = set(sys.modules)
    with pytest.raises(ManifestError, match='node_packs'):
        parse_manifest(value, Path('/fixture'))
    assert 'app.services.comfyui_node_install' not in set(sys.modules) - before
    assert LDS_PLUGIN_API_MINOR >= 20
    assert callable(PluginContext.register_node_pack)


def test_v2_package_api_floor_is_checked_before_import(host):
    value = manifest(schema_version=2, publisher={'id': 'sample', 'name': 'Sample'},
                     compatibility={'lds': '>=1', 'api': f'>=1.{LDS_PLUGIN_API_MINOR + 1}', 'python': '>=3.10',
                                    'os': ['windows', 'linux', 'darwin'], 'arch': ['x86_64', 'arm64']})
    parsed = parse_manifest(value, Path('/fixture'))
    assert any(item['field'] == 'compatibility.api' for item in host_issues(parsed))


def test_packages_recorded_against_the_previous_pillow_and_filelock_keep_loading(host, monkeypatch):
    """Every published package recorded host_dependencies pillow ==12.3.0 and
    filelock ==3.32.5. The core moving to 13.0.0 / 4.2.10 must not disable them,
    yet only that exact predecessor carries over, and a package recorded against
    the new pin still refuses an older core."""
    from app.plugins import compatibility

    def parsed(**host_dependencies):
        return parse_manifest(manifest(
            schema_version=2, publisher={'id': 'sample', 'name': 'Sample'},
            compatibility={'lds': '>=1', 'api': '>=1.0', 'python': '>=3.10',
                           'os': ['windows', 'linux', 'darwin'], 'arch': ['x86_64', 'arm64']},
            host_dependencies=host_dependencies), Path('/fixture'))

    installed = {'pillow': '13.0.0', 'filelock': '4.2.10'}
    monkeypatch.setattr(compatibility, 'version', lambda name: installed[name])
    assert host_issues(parsed(Pillow='==12.3.0', filelock='==3.32.5')) == []
    assert host_issues(parsed(Pillow='==13.0.0', filelock='==4.2.10')) == []
    assert [item['field'] for item in host_issues(parsed(Pillow='==12.2.0'))] == ['host_dependencies.pillow']
    installed.update(pillow='12.3.0', filelock='3.32.5')
    assert [item['field'] for item in host_issues(parsed(Pillow='==13.0.0', filelock='==4.2.10'))] == [
        'host_dependencies.pillow', 'host_dependencies.filelock']


@pytest.mark.parametrize('name', ['../escape.py', '/absolute.py', 'C:/escape.py', 'normal/../../escape.py'])
def test_archive_paths_rejected_before_staging(host, name):
    _, _, root = host
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('plugin.json', json.dumps(manifest()))
        archive.writestr(name, 'unsafe')
    archive_path = root / 'bad.zip'
    archive_path.write_bytes(stream.getvalue())
    with pytest.raises(ArchiveError):
        inspect_plugin_zip(archive_path)
    assert not (root / 'escape.py').exists()


@pytest.mark.parametrize('requirement', ['-r other.txt', '--target /elsewhere',
                                       'sample @ https://example.test/sample.whl', '../local'])
def test_plugin_requirements_cannot_redirect_installer(requirement):
    with pytest.raises(environment.EnvironmentError):
        environment._specs([requirement])


def test_transaction_snapshot_restores_previous_package_and_data(host):
    directory = write(host)
    external = directory.parent
    data = storage.data_dir('sample.feature')
    data.mkdir(parents=True)
    (data / 'keep.txt').write_text('original')
    transactions.prepare_batch(external, [{'id': 'sample.feature', 'action': 'remove'}])
    tx = transactions.active(external)
    transactions.snapshot(external, tx)
    transactions.publish(external, tx)
    assert not directory.exists()
    (data / 'keep.txt').write_text('changed')
    transactions.rollback(external, tx)
    assert directory.exists()
    assert (data / 'keep.txt').read_text() == 'original'


def test_a_committed_change_drops_its_recovery_copies_and_keeps_its_history(host):
    app, csrf, _ = host
    directory = write(host)
    external = directory.parent
    data = storage.data_dir('sample.feature')
    data.mkdir(parents=True)
    (data / 'keep.txt').write_text('kept')
    txid = transactions.prepare_batch(external, [{'id': 'sample.feature', 'action': 'remove'}])['id']
    load_plugins(app, csrf)   # The boot transaction applies, confirms and commits the change.
    assert [(tx['id'], tx['phase']) for tx in transactions.journals(external)] == [(txid, 'committed')]
    assert not (external / '.transactions' / txid / 'items').exists()   # old package and data snapshot
    assert (external / '.transactions' / txid / 'transaction.json').is_file()
    assert transactions.history(external)[0]['id'] == txid
    assert (data / 'keep.txt').read_text() == 'kept'


def test_purge_keeps_unfinished_and_rolled_back_recovery_material(host):
    directory = write(host)
    external = directory.parent
    def journal(phase, pid):
        txid = uuid.uuid4().hex
        folder = external / '.transactions' / txid
        (folder / 'items' / pid / 'data-before').mkdir(parents=True)
        (folder / 'items' / pid / 'data-before' / 'copy.txt').write_text('snapshot')
        (folder / 'transaction.json').write_text(json.dumps({
            'id': txid, 'format': 1, 'phase': phase, 'created_at': 1.0,
            'items': [{'id': pid, 'action': 'update', 'desired_enabled': True}]}))
        return folder
    committed, rolled_back, unfinished = (journal('committed', 'one.plugin'), journal('rolled_back', 'two.plugin'),
                                          journal('prepared', 'three.plugin'))
    assert transactions.purge_committed(external) == 1
    assert not (committed / 'items').exists() and (committed / 'transaction.json').is_file()
    for kept in (rolled_back, unfinished):
        assert (kept / 'items' / next((kept / 'items').iterdir()).name / 'data-before' / 'copy.txt').is_file()
    assert transactions.purge_committed(external) == 0
