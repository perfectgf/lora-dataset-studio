"""An install still on Python 3.10 reaches the new version through an update that
cannot change its interpreter. run.py must stop it before anything can touch the
environment or the data folder, and say what to do for that kind of install."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import python_floor  # noqa: E402

RUN_PY = Path(__file__).resolve().parents[1] / 'run.py'


def test_the_floor_is_the_one_the_requirements_need():
    assert python_floor.MINIMUM == (3, 11)
    assert not python_floor.supported((3, 10, 11))
    assert python_floor.supported((3, 11, 0)) and python_floor.supported((3, 14, 2))


def test_run_py_checks_before_pip_or_the_app_can_run():
    """ensure_pillow_consistent may run pip and create_app opens the database:
    both come after the floor."""
    src = RUN_PY.read_text(encoding='utf-8')
    floor = src.index('_enforce_python_floor(')
    assert floor < src.index('ensure_pillow_consistent()') < src.index('from app import create_app')


@pytest.mark.parametrize('runtime,start_bat,expected', [
    (None, True, 'double-click start.bat'),
    ('pinokio', True, 'click Update'),
    (None, False, 'Recreate the virtual environment'),
])
def test_the_message_names_the_way_out_for_this_install(tmp_path, monkeypatch, runtime, start_bat, expected):
    monkeypatch.setattr(python_floor.os, 'name', 'nt')
    if start_bat:
        (tmp_path / 'start.bat').write_text('@echo off\n')
    text = python_floor.refusal(repo_root=str(tmp_path), executable='C:/x/.venv/Scripts/python.exe',
                                version='3.10.11', runtime=runtime)
    assert expected in text and '3.10.11' in text and 'Nothing was changed' in text


class _Stop(Exception):
    pass


OLD = (3, 10, 11, 'final', 0)


def test_a_supported_python_starts_untouched(capsys, tmp_path):
    python_floor.enforce(str(tmp_path), version_info=(3, 12, 9, 'final', 0))
    assert capsys.readouterr().out == ''


def test_a_direct_start_says_why_and_exits(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv('LDS_RESTART_MODE', raising=False)
    monkeypatch.delenv('LDS_RUNTIME', raising=False)

    class NoTty:
        def isatty(self):
            return False

    with pytest.raises(SystemExit) as stop:
        python_floor.enforce(str(tmp_path), stdin=NoTty(), version_info=OLD)
    assert stop.value.code == 1
    assert 'needs Python 3.11 or newer' in capsys.readouterr().out


def test_under_the_start_bat_supervisor_it_waits_instead_of_exiting(monkeypatch, capsys, tmp_path):
    """An exit would be relaunched, then return to the start.bat that the update
    replaced on disk; waiting keeps the message on screen until the window closes."""
    monkeypatch.setenv('LDS_RESTART_MODE', 'supervisor')
    naps = []

    def sleep(seconds):
        naps.append(seconds)
        raise _Stop

    with pytest.raises(_Stop):
        python_floor.enforce(str(tmp_path), sleep=sleep, version_info=OLD)
    assert naps and 'needs Python 3.11 or newer' in capsys.readouterr().out
