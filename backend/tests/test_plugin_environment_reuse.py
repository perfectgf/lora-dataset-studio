"""Code updates retain compatible CPU engines, including existing receipts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.plugins import environment
from app.setup_installer import _venv_python


@pytest.fixture
def record(tmp_path, monkeypatch):
    monkeypatch.setattr(environment.cfg, 'data_dir', lambda: tmp_path / 'data')
    source = tmp_path / 'plugin'
    source.mkdir()
    (source / 'requirements.txt').write_text('torch>=2,<3\nnumpy>=1.26,<3\n')
    return SimpleNamespace(id='sample.engine', dir=source, manifest=SimpleNamespace(
        version='1.0.0', contract={'schema_version': 2},
        requirements='requirements.txt', in_process_requirements=False))


def ready(record, path, receipt):
    path.mkdir(parents=True)
    python = Path(_venv_python(path))
    python.parent.mkdir(exist_ok=True)
    python.write_text('interpreter fixture; never executed')
    (path / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    (path / environment.MARKER).write_text(json.dumps({'plugin': record.id}))
    (path / environment.RECEIPT).write_text(json.dumps(receipt))
    return str(python)


def legacy(record, **changes):
    receipt = {**environment._signature(record), 'version': '0.9.0', **changes}
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    path = environment.cfg.data_dir() / 'plugin-environments' / record.id / digest
    return path, ready(record, path, receipt)


def test_code_only_updates_and_rollback_keep_the_same_ready_engine(record):
    directory = environment.environment_dir(record)
    python = ready(record, directory, environment._signature(record))
    for version in ('1.1.0', '2.0.0', '1.0.0'):
        record.manifest.version = version
        assert environment.environment_dir(record) == directory
        assert environment.interpreter(record) == python


def test_existing_versioned_engine_is_reused_without_writes(record):
    path, python = legacy(record)
    before = (path / environment.RECEIPT).read_bytes()
    assert environment.interpreter(record) == python
    record.manifest.version = '1.2.0'
    assert environment.interpreter(record) == python
    assert (path / environment.RECEIPT).read_bytes() == before
    # An explicit repair can update the receipt without moving the old folder.
    (path / environment.RECEIPT).write_text(json.dumps(environment._signature(record)))
    assert environment.interpreter(record) == python


def test_dependency_change_uses_another_engine_and_rollback_recovers_the_old_one(record):
    path, python = legacy(record)
    requirements = record.dir / 'requirements.txt'
    before = requirements.read_text()
    requirements.write_text('torch>=3,<4\n')
    assert environment.environment_dir(record) != path
    with pytest.raises(environment.EnvironmentError, match='No environment yet'):
        environment.interpreter(record)
    requirements.write_text(before)
    assert environment.interpreter(record) == python


@pytest.mark.parametrize('changes', [
    {'runtime_policy': 'different-policy'}, {'python_abi': 'different-abi'},
    {'platform': 'different-platform'}, {'plugin': 'another.engine'},
    {'requirements_sha256': 'different-dependencies'}, {'unexpected': 'field'},
])
def test_incompatible_legacy_receipt_is_not_reused(record, changes):
    path, _ = legacy(record, **changes)
    assert environment.environment_dir(record) != path


@pytest.mark.parametrize('missing', [environment.MARKER, environment.RECEIPT, 'pyvenv.cfg', 'python'])
def test_incomplete_legacy_engine_is_not_reused(record, missing):
    path, python = legacy(record)
    (Path(python) if missing == 'python' else path / missing).unlink()
    assert environment.environment_dir(record) != path


def test_foreign_owner_and_linked_receipts_are_not_reused(record, tmp_path):
    path, _ = legacy(record)
    marker = path / environment.MARKER
    marker.write_text(json.dumps({'plugin': 'another.engine'}))
    assert environment.environment_dir(record) != path
    marker.write_text(json.dumps({'plugin': record.id}))
    (tmp_path / 'linked-receipt').hardlink_to(path / environment.RECEIPT)
    assert environment.environment_dir(record) != path


def test_legacy_schema_still_requires_its_exact_version(record):
    record.manifest.contract['schema_version'] = 1
    path = environment.environment_dir(record)
    ready(record, path, environment._signature(record))
    record.manifest.version = '1.1.0'
    with pytest.raises(environment.EnvironmentError, match='needs installation or repair'):
        environment.interpreter(record)
