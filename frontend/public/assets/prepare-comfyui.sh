#!/bin/sh
""":"
set -eu
umask 077
if command -v python3.12 >/dev/null 2>&1; then
    exec python3.12 "$0" "$@"
fi
case "$(uname -s)/$(uname -m)" in
    Linux/x86_64) triple=x86_64-unknown-linux-gnu; sha=c85811d35971d06612585a057db3b71806378df6ea8476275fb16aa5c9fe56a0 ;;
    Darwin/arm64) triple=aarch64-apple-darwin; sha=0de87cfc2649849da10a54084f87fd01fa1e653478e32eee92198c213550e15c ;;
    *) echo 'Use Python 3.12 with --source-only on this platform.' >&2; exit 1 ;;
esac
base="${XDG_DATA_HOME:-$HOME/.local/share}/lds/comfyui-bootstrap"
runtime="$base/$sha"
if [ ! -f "$runtime/.verified" ]; then
    mkdir -p "$base"
    if [ -e "$runtime" ] || [ -L "$runtime" ]; then
        echo "The existing Python location at $runtime is incomplete. Its files were preserved." >&2
        exit 1
    fi
    stage=$(mktemp -d "$base/.python-XXXXXXXX")
    cleanup() {
        if [ -n "$stage" ]; then
            case "$stage" in "$base"/.python-*) rm -rf -- "$stage" ;; esac
        fi
    }
    trap cleanup EXIT
    trap 'exit 1' HUP INT TERM
    archive="$stage/python.tar.gz"
    url="https://lds.perfectgf.fr/models/tools/$sha/cpython-3.12.16%2B20261123-$triple-install_only_stripped.tar.gz"
    curl --fail --show-error --proto '=https' --output "$archive" "$url"
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum "$archive")
    else
        actual=$(shasum -a 256 "$archive")
    fi
    [ "${actual%% *}" = "$sha" ] || { echo 'Managed Python checksum mismatch.' >&2; exit 1; }
    tar -xzf "$archive" -C "$stage"
    "$stage/python/bin/python3.12" -V
    rm -f -- "$archive"
    touch "$stage/.verified"
    if ln -sn "$stage" "$runtime" 2>/dev/null; then
        stage=''
    elif [ ! -L "$runtime" ] || [ ! -f "$runtime/.verified" ]; then
        echo "Could not publish the verified runtime at $runtime." >&2
        exit 1
    fi
    cleanup
    trap - EXIT HUP INT TERM
fi
exec "$runtime/python/bin/python3.12" "$0" "$@"
":"""


"""Prepare the pinned ComfyUI source and its environment on Linux or macOS."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.request import HTTPRedirectHandler, build_opener
import zipfile

ROOT = 'https://lds.perfectgf.fr/models/tools'
SOURCE_HASH = '618bda67bfefcc3920973055f75fe2ec392d3abf1aaaa972532999047cc764f8'
SOURCE_FOLDER = 'ComfyUI-a277b5bf213e8a44fa34817b7770b54f86ef4cea'
REQUIREMENTS = {
    'linux': ('c470124bd6027dab79e461622eb006341438ebf48ff76f4a111f058d21fee55e', 'comfyui-0.46.0-python312-linux.txt'),
    'darwin': ('47d2c350dabd860814164b0095098cf79b63abecf640dac457accf1030f32ece', 'comfyui-0.46.0-python312-mac.txt'),
}
MARKER = '.lds-comfyui-source'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download(name, sha, destination, maximum):
    digest = hashlib.sha256()
    size = 0
    with build_opener(NoRedirect).open(f'{ROOT}/{sha}/{name}', timeout=120) as response, destination.open('xb') as output:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > maximum:
                raise ValueError(f'{name} is larger than the verified release.')
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != sha:
        raise ValueError(f'The checksum of {name} does not match the verified release.')


def extract(archive, staging):
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            p = PurePosixPath(entry.filename)
            if not p.parts or p.parts[0] != SOURCE_FOLDER or '..' in p.parts or '\\' in entry.filename:
                raise ValueError('Unexpected path in the ComfyUI source archive.')
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Unexpected link in the ComfyUI source archive.')
        source.extractall(staging)
    result = staging / SOURCE_FOLDER
    if not (result/'main.py').is_file():
        raise ValueError('The extracted archive has no ComfyUI launcher.')
    return result


def main():
    parser = argparse.ArgumentParser(description='Prepare the pinned ComfyUI source and its Python environment.')
    parser.add_argument('--directory', type=Path, default=Path('ComfyUI-0.46.0'))
    parser.add_argument('--source-only', action='store_true', help='Only extract the verified source for a separately managed environment.')
    args = parser.parse_args()
    destination = args.directory.absolute()
    marker = destination / MARKER
    if destination.is_symlink() or (destination.exists() and
            (marker.is_symlink() or not marker.is_file() or marker.read_text() != SOURCE_HASH)):
        raise SystemExit('Choose a new installation folder. The selected directory already exists.')
    if not args.source_only:
        if sys.version_info[:2] != (3, 12):
            raise SystemExit('Run this script with Python 3.12.')
        if sys.platform == 'darwin':
            supported = platform.machine() == 'arm64' and int(platform.mac_ver()[0].split('.')[0]) >= 14
        else:
            libc, version = platform.libc_ver()
            supported = (sys.platform == 'linux' and platform.machine() == 'x86_64' and libc == 'glibc'
                         and tuple(map(int, version.split('.')[:2])) >= (2, 35) and shutil.which('nvidia-smi'))
        if not supported:
            raise SystemExit('Use a supported Linux NVIDIA or Apple Silicon host, or use --source-only for your own environment.')
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)
        if not destination.exists():
            archive = staging / 'source.zip'
            download('ComfyUI-0.46.0-source.zip', SOURCE_HASH, archive, 100 * 1024**2)
            result = extract(archive, staging)
            (result / MARKER).write_text(SOURCE_HASH)
            result.rename(destination)
        if args.source_only:
            print(f'Verified ComfyUI source: {destination}')
            return
        sha, name = REQUIREMENTS[sys.platform]
        lock = staging / 'requirements.txt'
        download(name, sha, lock, 1024 * 1024)
        env = destination / '.venv'
        subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)
        python = env / 'bin/python'
        install_env = {k: v for k, v in os.environ.items() if not k.upper().startswith('PIP_')}
        install_env['PIP_CONFIG_FILE'] = os.devnull
        subprocess.run([str(python), '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
                        '--only-binary=:all:', '--index-url', ROOT + ('/pytorch/cu128/' if sys.platform == 'linux' else '/pypi/simple/'), '-r', str(lock)],
                       env=install_env, check=True)
        print(f'Prepared {destination}. In that folder run .venv/bin/python main.py, then select the folder and http://127.0.0.1:8188 in LDS Setup.')


if __name__ == '__main__':
    main()
