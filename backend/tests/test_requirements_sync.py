"""An in-app update restarted by the start.bat supervisor reuses .venv as it is:
run.py must install the pins that update moved, and touch nothing else."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requirements_sync  # noqa: E402

RUN_PY = Path(__file__).resolve().parents[1] / 'run.py'


def _install(tmp_path, text):
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'backend' / 'requirements.txt').write_text(text, encoding='utf-8')
    (tmp_path / '.venv').mkdir()
    return tmp_path


def test_reports_only_what_this_environment_does_not_meet(tmp_path, monkeypatch):
    root = _install(tmp_path, '# comment\n-r other.txt\nPillow==13.0.0\nfilelock==4.2.10  # pinned\n'
                              'absent-dist==1.0\nlinux-only==1; sys_platform == "nonexistent-os"\n'
                              'packaging>=1\n')
    versions = {'pillow': '12.3.0', 'filelock': '4.2.10', 'packaging': '26.0'}

    def version(name):
        from importlib.metadata import PackageNotFoundError
        if name.lower() not in versions:
            raise PackageNotFoundError(name)
        return versions[name.lower()]

    monkeypatch.setattr('importlib.metadata.version', version)
    assert requirements_sync.unsatisfied(root / 'backend' / 'requirements.txt') == [
        'Pillow 12.3.0 -> ==13.0.0', 'absent-dist (missing) -> ==1.0']


def test_installs_once_into_start_bats_venv_when_a_pin_moved(tmp_path, monkeypatch):
    root = _install(tmp_path, 'Pillow==13.0.0\n')
    monkeypatch.setattr(requirements_sync, 'unsatisfied', lambda path: ['Pillow 12.3.0 -> ==13.0.0'])
    calls, said = [], []
    result = requirements_sync.ensure(
        root, prefix=str(root / '.venv'), out=lambda *a, **k: said.append(a[0]),
        run=lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    assert result == ['Pillow 12.3.0 -> ==13.0.0']
    assert len(calls) == 1 and calls[0][0][1:] == ['-m', 'pip', 'install', '-q', '-r',
                                                   str(root / 'backend' / 'requirements.txt')]
    assert 'installing the requirements this version changed' in said[0]


def test_any_other_interpreter_is_left_alone(tmp_path, monkeypatch):
    """A system Python, a hand-made venv, Pinokio's env/: never pip-installed into."""
    root = _install(tmp_path, 'Pillow==13.0.0\n')
    monkeypatch.setattr(requirements_sync, 'unsatisfied', lambda path: ['Pillow 12.3.0 -> ==13.0.0'])
    calls = []
    for prefix in (sys.base_prefix, str(root / 'env'), str(tmp_path / 'elsewhere' / '.venv')):
        assert requirements_sync.ensure(root, prefix=prefix, run=lambda *a, **k: calls.append(a)) is None
    assert calls == []


def test_a_failed_install_or_check_never_stops_the_boot(tmp_path, monkeypatch):
    root = _install(tmp_path, 'Pillow==13.0.0\n')
    said = []
    out = lambda *a, **k: said.append(a[0])  # noqa: E731
    monkeypatch.setattr(requirements_sync, 'unsatisfied', lambda path: ['Pillow 12.3.0 -> ==13.0.0'])
    requirements_sync.ensure(root, prefix=str(root / '.venv'), out=out,
                             run=lambda *a, **k: SimpleNamespace(returncode=1))

    def broken(*a, **k):
        raise OSError('no pip')
    requirements_sync.ensure(root, prefix=str(root / '.venv'), out=out, run=broken)
    monkeypatch.setattr(requirements_sync, 'unsatisfied', lambda path: (_ for _ in ()).throw(ValueError('bad')))
    assert requirements_sync.ensure(root, prefix=str(root / '.venv'), out=out) is None
    assert sum('installing them failed' in line for line in said) == 2
    assert 'starting as is' in said[-1]


def test_run_py_syncs_after_the_python_floor_and_before_anything_imports_pillow():
    src = RUN_PY.read_text(encoding='utf-8')
    assert (src.index('_enforce_python_floor(') < src.index('_sync_requirements(')
            < src.index('ensure_pillow_consistent()') < src.index('from app import create_app'))
