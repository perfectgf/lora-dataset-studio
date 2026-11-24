"""Setup owns compatible ML runtimes; host/borrowed Python stays read-only."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from app import config, setup_installer as si
from app.services import managed_python as mp


def _archive(path, files):
    with tarfile.open(path, 'w:gz') as archive:
        for name, content in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))


def test_python_314_automatically_provisions_when_no_compatible_base(app, monkeypatch):
    calls = []
    monkeypatch.setattr(si, '_base_python_candidates', lambda: ['/host/python'])
    monkeypatch.setattr(si, '_python_minor', lambda _: (3, 14))
    monkeypatch.setattr(mp, 'ensure_python', lambda log: calls.append(log) or '/managed/python')
    assert si._find_base_python('masks') == '/managed/python'
    assert len(calls) == 1


def test_compatible_base_is_read_only_and_needs_no_download(app, monkeypatch):
    monkeypatch.setattr(si, '_base_python_candidates', lambda: ['/existing/python'])
    monkeypatch.setattr(si, '_python_minor', lambda _: (3, 12))
    monkeypatch.setattr(mp, 'ensure_python', lambda _: pytest.fail('unexpected download'))
    assert si._find_base_python('masks') == '/existing/python'


def test_failed_bootstrap_is_logged_without_pip_fallback(app, monkeypatch):
    monkeypatch.setattr(si, '_base_python_candidates', lambda: [])
    monkeypatch.setattr(mp, 'ensure_python', lambda _: (_ for _ in ()).throw(ValueError('SHA-256 mismatch')))
    monkeypatch.setattr(si, '_run_pip', lambda *a: pytest.fail('must not run pip'))
    si._runs['masks'] = si._new_run()
    with app.app_context():
        assert si._run_ml_capability('masks') == 1
        assert not config.get('masks.python')
    assert 'SHA-256' in '\n'.join(si._runs['masks']['log'])


def test_runtime_status_never_scans_or_downloads(monkeypatch):
    monkeypatch.setattr(mp, '_asset', lambda _: pytest.fail('network from status'))
    monkeypatch.setattr(mp, '_valid_python', lambda _: pytest.fail('scan from status'))
    assert mp.status()['installation'] == 'automatic'
    assert mp.status()['isolated'] is True


@pytest.mark.parametrize('system,arch,libc,expected', [
    ('Windows', 'AMD64', '', 'x86_64-pc-windows-msvc'),
    ('Windows', 'ARM64', '', ''),
    ('Linux', 'aarch64', 'glibc', 'aarch64-unknown-linux-gnu'),
    ('Linux', 'x86_64', 'musl', ''),
    ('Darwin', 'arm64', '', 'aarch64-apple-darwin'),
])
def test_automatic_platform_contract(monkeypatch, system, arch, libc, expected):
    monkeypatch.setattr(mp.platform, 'system', lambda: system)
    monkeypatch.setattr(mp.platform, 'machine', lambda: arch)
    monkeypatch.setattr(mp.platform, 'libc_ver', lambda: (libc, ''))
    assert mp.target_triple() == expected


class Response:
    status_code = 200
    def __init__(self, data=None, body=b''):
        self.data, self.body = data, body
        self.url = 'https://release-assets.githubusercontent.com/asset'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def json(self):
        return self.data

    def iter_content(self, chunk_size):
        yield self.body


def test_publisher_checksums_and_lds_only_download():
    for triple in mp._RUNTIME_FILES:
        asset = mp._asset(triple)
        assert len(bytes.fromhex(asset['digest'][7:])) == 32
        assert asset['browser_download_url'].startswith(mp._DOWNLOAD_PREFIX + asset['digest'][7:] + '/')
        assert 0 < asset['size'] <= mp._MAX_ARCHIVE
    with pytest.raises(ValueError):
        mp._asset('unknown-platform')


def test_download_hash_is_checked_before_extraction(tmp_path, monkeypatch):
    body = b'bad archive bytes'
    monkeypatch.setattr(mp.requests, 'get', lambda *a, **k: Response(body=body))
    asset = {'size': len(body), 'digest': 'sha256:' + '0' * 64,
             'browser_download_url': mp._DOWNLOAD_PREFIX + 'runtime'}
    with pytest.raises(ValueError, match='SHA-256'):
        mp._download(asset, tmp_path / 'archive')


def test_download_and_size_match(tmp_path, monkeypatch):
    body = b'verified archive bytes'
    monkeypatch.setattr(mp.requests, 'get', lambda *a, **k: Response(body=body))
    asset = {'size': len(body), 'digest': 'sha256:' + hashlib.sha256(body).hexdigest(),
             'browser_download_url': mp._DOWNLOAD_PREFIX + 'runtime'}
    target = tmp_path / 'archive'
    mp._download(asset, target)
    assert target.read_bytes() == body


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'python/../escape',
                                   'python/C:/escape', 'python/dir\\escape'])
def test_archive_cannot_escape_staging(tmp_path, name):
    archive = tmp_path / 'archive.tar.gz'
    _archive(archive, {name: b'payload'})
    with pytest.raises(ValueError, match='Unsafe'):
        mp._extract(archive, tmp_path / 'staging')
    assert not (tmp_path / 'escape').exists()


def test_archive_cannot_link_outside_staging(tmp_path):
    archive = tmp_path / 'archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        item = tarfile.TarInfo('python/bin/escape')
        item.type, item.linkname = tarfile.SYMTYPE, '../../../escape'
        bundle.addfile(item)
    with pytest.raises(ValueError, match='link'):
        mp._extract(archive, tmp_path / 'staging')


def test_owned_destination_refuses_outside_path(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        mp.assert_owned_directory(tmp_path.parent / 'outside', tmp_path)


def test_pip_and_python_redirects_are_removed(monkeypatch):
    for name in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_CONFIG_FILE', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        monkeypatch.setenv(name, 'external')
    env = mp.subprocess_env()
    assert env['PIP_CONFIG_FILE'] == os.devnull
    assert env['PYTHONNOUSERSITE'] == '1'
    assert not set(env) & {'PIP_TARGET', 'PIP_PREFIX', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}


def test_failed_creation_restores_broken_environment(app, monkeypatch):
    with app.app_context():
        env = si._quality_env_dir()
        env.mkdir(parents=True)
        (env / 'kept.txt').write_text('old')
        monkeypatch.setattr(si, '_managed_env_valid', lambda _: False)
        monkeypatch.setattr(si, '_find_base_python', lambda _: '/base/python')
        monkeypatch.setattr(si, '_run_pip', lambda *a: 1)
        assert si._ensure_managed_ml_env('masks', env) == ''
        assert (env / 'kept.txt').read_text() == 'old'
        assert list(env.parent.glob('quality.previous-*')) == []


def test_good_managed_environment_is_reused_without_bootstrap(app, monkeypatch):
    with app.app_context():
        env = si._quality_env_dir()
        python = Path(si._venv_python(env))
        python.parent.mkdir(parents=True)
        python.touch()
        monkeypatch.setattr(si, '_managed_env_valid', lambda _: True)
        monkeypatch.setattr(si, '_ensure_modern_pip', lambda *a: None)
        monkeypatch.setattr(si, '_find_base_python', lambda _: pytest.fail('unexpected bootstrap'))
        assert si._ensure_managed_ml_env('masks', env) == str(python)


@pytest.mark.parametrize('version,prefix,base,expected', [
    ([3, 12], 'managed', 'base', True),
    ([3, 14], 'managed', 'base', False),
    ([3, 12], 'external', 'base', False),
    ([3, 12], 'managed', 'managed', False),
])
def test_managed_venv_identity_is_executed(tmp_path, monkeypatch, version, prefix, base, expected):
    env = tmp_path / 'managed'
    python = si._venv_python(env)
    result = subprocess.CompletedProcess([], 0, json.dumps([version, str(tmp_path / prefix), str(tmp_path / base)]))
    monkeypatch.setattr(si.subprocess, 'run', lambda *a, **k: result)
    assert si._managed_env_valid(python) is expected


@pytest.mark.parametrize('feature', ['face_scoring', 'masks'])
@pytest.mark.parametrize('import_ok', [True, False])
def test_quality_pip_is_managed_and_selection_waits_for_import(app, monkeypatch, feature, import_ok):
    calls = []
    with app.app_context():
        managed = si._venv_python(si._quality_env_dir())
        monkeypatch.setattr(si, '_ensure_managed_ml_env', lambda *a: managed)
        monkeypatch.setattr(si, '_drop_provided_onnxruntime', lambda a, p, specs: specs)
        monkeypatch.setattr(si, '_run_pip', lambda a, cmd: calls.append(cmd) or 0)
        monkeypatch.setattr(si, '_verify_capability_import', lambda *a, **k: import_ok)
        assert si._run_ml_capability(feature) == (0 if import_ok else 1)
        assert (config.get(feature + '.python') == managed) is import_ok
    assert all(cmd[0] == managed for cmd in calls)
    assert '--only-binary=:all:' in calls[0]


def test_borrowed_quality_runtime_is_never_changed(app, monkeypatch):
    with app.app_context():
        config.save_config({'masks': {'python': '/external/python'}})
        managed = si._venv_python(si._quality_env_dir())
        monkeypatch.setattr(si, '_ensure_managed_ml_env', lambda *a: managed)
        calls = []
        monkeypatch.setattr(si, '_run_pip', lambda a, cmd: calls.append(cmd) or 0)
        monkeypatch.setattr(si, '_verify_capability_import', lambda *a, **k: True)
        assert si._run_ml_capability('masks') == 0
        assert config.get('masks.python') == '/external/python'
        assert all(cmd[0] == managed for cmd in calls)


@pytest.mark.parametrize('failure', ['missing', 'timeout', 'oserror'])
def test_unverified_import_cannot_report_success(tmp_path, monkeypatch, failure):
    python = tmp_path / 'python'
    if failure != 'missing':
        python.touch()
    def fail(*a, **k):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired('python', 1)
        raise OSError('cannot execute')
    monkeypatch.setattr(si.subprocess, 'run', fail)
    assert si._verify_capability_import('masks', str(python)) is False


def test_failed_runtime_download_then_retry_preserves_previous_bytes(app, tmp_path, monkeypatch):
    root = config.data_dir() / 'runtimes' / 'python-3.12'
    python = mp._python(root)
    python.parent.mkdir(parents=True)
    python.write_bytes(b'previous broken runtime')
    relative = 'python/python.exe' if os.name == 'nt' else 'python/bin/python3'
    archive = tmp_path / 'publisher.tar.gz'
    _archive(archive, {relative: b'validated runtime'})
    body = archive.read_bytes()
    asset = {'name': 'fixed-python.tar.gz', 'size': len(body),
             'digest': 'sha256:' + hashlib.sha256(body).hexdigest(),
             'browser_download_url': mp._DOWNLOAD_PREFIX + 'runtime'}
    monkeypatch.setattr(mp, 'target_triple', lambda: 'fixture')
    monkeypatch.setattr(mp, '_asset', lambda _: asset)
    monkeypatch.setattr(mp, '_valid_python',
                        lambda p: Path(p).is_file() and Path(p).read_bytes() == b'validated runtime')
    monkeypatch.setattr(mp.requests, 'get', lambda *a, **k: Response(body=b'corrupt'))
    with pytest.raises(ValueError, match='SHA-256'):
        mp.ensure_python(lambda _: None)
    assert python.read_bytes() == b'previous broken runtime'
    assert not list(root.parent.glob('.python-setup-*'))
    monkeypatch.setattr(mp.requests, 'get', lambda *a, **k: Response(body=body))
    assert mp.ensure_python(lambda _: None) == str(python)
    assert python.read_bytes() == b'validated runtime'
    assert len(list(root.parent.glob('python-3.12.previous-*'))) == 1
    assert not list(root.parent.glob('.python-setup-*'))
    monkeypatch.setattr(mp, '_asset', lambda _: pytest.fail('valid runtime must be reused'))
    assert mp.ensure_python(lambda _: None) == str(python)


def test_quality_verification_uses_exact_worker_probe_import(tmp_path, monkeypatch):
    python = tmp_path / 'python'
    python.touch()
    from app import capabilities
    seen = []
    monkeypatch.setattr(si.subprocess, 'run', lambda cmd, **kw:
                        seen.append((cmd, kw)) or subprocess.CompletedProcess(cmd, 0, '', ''))
    assert si._verify_capability_import('masks', str(python))
    assert seen[0][0] == [str(python), '-s', '-c', capabilities.CAPABILITY_IMPORTS['masks']]
    assert seen[0][1]['env']['PYTHONNOUSERSITE'] == '1'
