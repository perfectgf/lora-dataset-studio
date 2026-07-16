"""Static contracts shared by the app and its development entrypoints."""
import json
import re
from pathlib import Path

from app.config import DEFAULTS


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding='utf-8')


def _docker_env(dockerfile):
    """Return ENV assignments regardless of single or continued-line layout."""
    logical_lines = dockerfile.replace('\\\n', ' ')
    assignments = {}
    for line in logical_lines.splitlines():
        if not line.startswith('ENV '):
            continue
        for key, value in re.findall(r'([A-Z][A-Z0-9_]*)=([^\s]+)', line[4:]):
            assignments[key] = value
    return assignments


def test_container_runtime_tracks_server_defaults():
    """The image must work both directly and through Docker Compose."""
    server = DEFAULTS['server']
    port = server['port']
    dockerfile = (REPO_ROOT / 'Dockerfile').read_text(encoding='utf-8')
    compose = _read('docker-compose.yml')
    image_env = _docker_env(dockerfile)

    assert image_env['LDS_DATA_DIR'] == '/data'
    assert image_env['LDS_CONFIG'] == '/data/config.json'
    assert image_env['LDS_HOST'] == '0.0.0.0'
    assert image_env['LDS_PORT'] == str(port)
    assert f'EXPOSE {port}' in dockerfile
    assert f'http://127.0.0.1:{port}/api/health' in dockerfile
    assert f'ports: ["{port}:{port}"]' in compose
    assert f'LDS_PORT={port}' in compose
    assert 'LDS_HOST=0.0.0.0' in compose
    assert 'LDS_CONFIG=/data/config.json' in compose


def test_developer_entrypoints_track_server_default():
    """Examples and the Vite proxy must follow the backend's real default port."""
    port = DEFAULTS['server']['port']
    example = json.loads(_read('config.example.json'))
    vite = _read('frontend/vite.config.js')

    assert example['server']['port'] == port
    proxy = re.search(r"['\"]\/api['\"]\s*:\s*['\"]http:\/\/127\.0\.0\.1:(\d+)", vite)
    assert proxy, 'Vite must declare a loopback /api proxy'
    assert int(proxy.group(1)) == port


def test_docker_context_excludes_generated_artifacts():
    ignored = {
        line.strip().rstrip('/')
        for line in _read('.dockerignore').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    }

    assert {'.worktrees', '.pytest_cache', 'packaging/build', 'packaging/dist'} <= ignored
