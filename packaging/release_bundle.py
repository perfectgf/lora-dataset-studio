"""Build the Windows core ZIP from one committed Git tree.

Working files and installed plugins never enter this archive. Plugin packages
are distributed separately through the Store.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT_FILES = {'start.bat', 'README.md', 'LICENSE', 'config.example.json', '.env.example'}
BACKEND_FILES = {'bootstrap_dependencies.py', 'port_utils.py', 'python_floor.py', 'requirements.txt',
                 'requirements-ml.txt', 'requirements-scrape.txt', 'run.py',
                 'single_instance.py', 'supervise.py'}
# start.bat runs these: one fetches a Python, the other sets an old .venv aside.
LAUNCHER_SCRIPTS = {'scripts/bootstrap_python.ps1', 'scripts/retire_venv.ps1'}
BACKEND_DIRS = {'app', 'comfy_nodes', 'infer', 'lds_sdk', 'workflows'}
LOCAL_DIRS = {'extensions', '__pycache__', 'node_modules', 'tests', 'data',
              'bundled', 'venv', '.venv', '.pytest_cache'}
ARCHIVE_SUFFIXES = ('.zip', '.whl', '.tar', '.gz', '.bz2', '.xz', '.tgz',
                    '.tbz2', '.txz', '.7z', '.rar', '.pyz')
PUBLIC_STORE_FILES = {'store/bootstrap.json', 'store/public-root.json', 'store/patreon.json'}
REQUIRED = ROOT_FILES | {'backend/run.py', 'backend/python_floor.py', 'backend/requirements.txt',
                        'backend/app/version.py', 'backend/lds_sdk/__init__.py',
                        'frontend/dist/index.html'} | LAUNCHER_SCRIPTS | PUBLIC_STORE_FILES


def runtime_member(name: str) -> bool:
    """Select runtime paths, retaining the host's app/plugins implementation."""
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith('/') or any(part in {'', '.', '..'} for part in parts)
            or '\\' in name or ':' in name):
        raise ValueError('Invalid Git member path')
    if name in ROOT_FILES or name in LAUNCHER_SCRIPTS or name in PUBLIC_STORE_FILES:
        return True
    if any(part.casefold() in LOCAL_DIRS or part.startswith('.') for part in parts):
        return False
    if name.lower().endswith(('.exe', '.pyc', '.pyo', '.ldsplugin', '.sqlite', '.sqlite3', '.db') + ARCHIVE_SUFFIXES):
        return False
    if len(parts) >= 3 and parts[:2] == ('frontend', 'dist'):
        return True
    return (parts[0] == 'backend' and len(parts) >= 2
            and (len(parts) == 2 and parts[1] in BACKEND_FILES
                 or len(parts) >= 3 and parts[1] in BACKEND_DIRS))


def build(repo: Path, output_dir: Path, *, ref: str = 'HEAD',
          name: str = 'LoRA-Dataset-Studio-windows', tag: str = '') -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}', name) or name.endswith('.'):
        raise ValueError('Use a simple archive name, without a path')
    repo = repo.resolve(strict=True)
    output_dir = output_dir.resolve()
    destination = output_dir / (name + '.zip')
    if destination.exists():
        raise ValueError('Archive already exists; choose a new output name')
    git = ['git', '--no-replace-objects', '-C', str(repo)]
    git_env = {key: value for key, value in os.environ.items() if not key.upper().startswith('GIT_')}
    commit = subprocess.check_output([*git, 'rev-parse', '--verify', '--end-of-options',
                                      ref + '^{commit}'], text=True, env=git_env).strip()
    if not re.fullmatch(r'[0-9a-f]{40,64}', commit):
        raise ValueError('Expected a complete Git commit')
    entries = []
    tree = subprocess.check_output([*git, 'ls-tree', '-rz', '--full-tree', commit], env=git_env)
    for raw in filter(None, tree.split(b'\0')):
        metadata, encoded_path = raw.split(b'\t', 1)
        path = encoded_path.decode('utf-8')
        if not runtime_member(path):
            continue
        mode, kind, oid = metadata.split()
        if mode not in {b'100644', b'100755'} or kind != b'blob':
            raise ValueError('Runtime archive requires regular files')
        entries.append((path, oid))
    # Read exact objects: export attributes, dirty files and symlink targets
    # cannot alter what goes into the release.
    result = subprocess.run([*git, 'cat-file', '--batch'],
                            input=b''.join(oid + b'\n' for _, oid in entries),
                            stdout=subprocess.PIPE, check=True, env=git_env)
    members, offset = {}, 0
    for path, oid in entries:
        header_end = result.stdout.index(b'\n', offset)
        actual_oid, kind, size = result.stdout[offset:header_end].split()
        if actual_oid != oid or kind != b'blob':
            raise ValueError('Git returned an unexpected object')
        start = header_end + 1
        offset = start + int(size) + 1
        members[path] = result.stdout[start:offset - 1]
    if REQUIRED - members.keys():
        raise ValueError('The committed core build is incomplete')
    match = re.search(r"APP_VERSION = '([^']+)'", members['backend/app/version.py'].decode('utf-8'))
    if not match:
        raise ValueError('The committed app version is missing')
    info = {'version': match[1], 'tag': tag or 'v' + match[1], 'commit': commit,
            'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    members['build_info.json'] = (json.dumps(info, indent=2) + '\n').encode('utf-8')
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(members.items()):
            archive.writestr(name + '/' + path, content)
    return {'archive': str(destination), 'commit': commit, 'files': len(members),
            'bytes': destination.stat().st_size, 'plugins_included': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--name', default='LoRA-Dataset-Studio-windows')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.output_dir or args.repo / 'packaging/dist',
                           ref=args.ref, name=args.name, tag=args.tag)))


if __name__ == '__main__':
    main()
