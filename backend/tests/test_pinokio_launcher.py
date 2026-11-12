"""The Pinokio launcher must keep pointing at the app it launches.

The launcher scripts at the repo root (pinokio.js / install.js / start.js /
update.js / reset.js) hardcode three facts about this project: the requirements
file, the server entry point, and the venv folder. None of them is imported by
Python, so a rename would move the code and leave the launcher silently
launching nothing — the failure would only surface on a stranger's machine,
inside Pinokio, which nobody runs in CI. These assertions are the cheap link
between the two.

They check the *contract*, not the prose: the referenced paths must exist on
disk, and the behaviours a Pinokio launcher needs (daemon mode, an Open Web UI
url set from the real bound address, a fast-forward-only update) must still be
declared.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = ('pinokio.js', 'install.js', 'start.js', 'update.js', 'reset.js')


def read(name):
    return (ROOT / name).read_text(encoding='utf-8')


@pytest.mark.parametrize('name', SCRIPTS + ('icon.png',))
def test_launcher_files_exist(name):
    assert (ROOT / name).is_file(), f'{name} is missing — the Pinokio launcher is incomplete'


def test_install_targets_the_real_requirements_file():
    src = read('install.js')
    assert 'backend/requirements.txt' in src
    assert (ROOT / 'backend' / 'requirements.txt').is_file()
    assert 'venv: "env"' in src


def test_start_runs_the_real_entry_point_as_a_daemon():
    src = read('start.js')
    assert 'backend/run.py' in src
    assert (ROOT / 'backend' / 'run.py').is_file()
    assert 'daemon: true' in src
    # The frontend is served by the backend from the tracked build; a Pinokio
    # install never runs npm, so the built entry point has to be committed.
    assert (ROOT / 'frontend' / 'dist' / 'index.html').is_file()


def test_start_publishes_the_url_it_actually_bound():
    """Port 5050 is a default, not a promise: run.py advances when it is taken.
    The Open Web UI tab must therefore come from the terminal match, never from
    a literal URL — and the line it matches must be one run.py really prints.
    Werkzeug's own banner is NOT one of them (create_app sends the root logger to
    data/app.log), which is why run.py prints its own."""
    src = read('start.js')
    assert 'local.set' in src
    assert '{{input.event[1]}}' in src
    assert 'Ready on' in src
    assert '[LDS] Ready on ' in (ROOT / 'backend' / 'run.py').read_text(encoding='utf-8')
    assert 'LDS_PORT: "5050"' in src
    # run.py re-execs into a sibling .venv when it finds one; under Pinokio the
    # active interpreter is env/, so that hijack must stay disabled.
    assert 'LDS_NO_REEXEC: "1"' in src


def test_update_stays_compatible_with_the_in_app_updater():
    assert 'git pull --ff-only' in read('update.js')


@pytest.mark.parametrize('name', ('install.js', 'start.js', 'update.js'))
def test_every_env_is_built_on_a_python_the_core_supports(name):
    """The core needs Python 3.11+ (Pillow 13, filelock 4.2), so env/ can no
    longer come from Pinokio's bundled 3.10: each step that may create it names
    the interpreter. Pinokio honours that only when it CREATES env/, so install
    and update first retire an env/ built on an older Python, before the step
    that installs the requirements into it."""
    src = read(name)
    assert src.count('venv: "env"') == src.count('venv_python: "3.12"') >= 1
    if name != 'start.js':
        floor = src.index('node scripts/pinokio_env_floor.js')
        assert floor < src.index('uv pip install -r backend/requirements.txt')
        assert (ROOT / 'scripts' / 'pinokio_env_floor.js').is_file()


def test_start_declares_the_runtime_the_updater_keys_off():
    """Without this the app cannot tell it was launched by Pinokio, and its
    Updates card offers an in-app restart that would detach the server from the
    launcher (see updater.is_pinokio_runtime)."""
    assert 'LDS_RUNTIME: "pinokio"' in read('start.js')
    updater = (ROOT / 'backend' / 'app' / 'services' / 'updater.py').read_text(encoding='utf-8')
    assert "PINOKIO_RUNTIME = 'pinokio'" in updater


@pytest.mark.parametrize('name', ('install.js', 'start.js'))
def test_scripts_name_the_pinokio_toolkit_they_need(name):
    """`requires.bundle` is how Pinokio 8 routes a machine with a missing
    conda/git/uv to its own setup page instead of failing mid-command. "python"
    (conda, git, uv, node) — not "ai", which would also pull CUDA and the
    Visual Studio Build Tools this app never uses."""
    src = read(name)
    assert 'requires' in src and 'bundle: "python"' in src


def test_reset_does_not_delete_user_data():
    """The stock launcher reset removes the cloned app folder. Here that folder
    holds data/ and config.json, so reset is scoped to the venv."""
    src = read('reset.js')
    assert 'path: "env"' in src
    for user_state in ('"data"', "'data'", 'config.json'):
        assert f'path: {user_state}' not in src


def test_menu_exposes_every_script():
    src = read('pinokio.js')
    for name in SCRIPTS[1:]:
        assert name in src, f'{name} is unreachable from the Pinokio menu'
    assert 'icon.png' in src


def test_venv_folder_is_ignored():
    """Pinokio creates env/ inside the checkout; committing it would be a
    multi-hundred-MB accident and would break the in-app git updater."""
    assert '/env/' in (ROOT / '.gitignore').read_text(encoding='utf-8')
