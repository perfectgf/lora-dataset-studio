"""Install the packaged synchronous MP4 writer without changing pip packages."""
import json
import os
import tempfile
import hashlib
import urllib.request
from pathlib import Path

TOOLS_ROOT = 'https://lds.perfectgf.fr/models/tools'
WRITER_FILES = {'__init__.py': {'bytes': 9371, 'sha256': '01852093fe4493786c87abaf212e2ae907055317d778d824ef617038827f3278'}, 'LICENSE': {'bytes': 10757, 'sha256': '5c7f173199fd7fb3cc83d86d24f3541e8ae0cb8c16e912ca519ed6a1435bd8f3'}, 'NOTICE': {'bytes': 432, 'sha256': '4f1442c7a5833f99ae6302862697a0a2c06f16c7f0ae39f8785c4fc0fc2961bf'}}


def install_writer(log=print, fetch=None):
    from lds_sdk.video_host import config as cfg
    configured = cfg.comfyui_dir()
    if not configured:
        log('Configure the ComfyUI folder first.')
        return 1
    custom = (Path(configured) / 'custom_nodes').resolve()
    target = custom / 'lds_h3_fast_writer'
    if not custom.is_dir() or target.is_symlink() or target.resolve() != target:
        log('The ComfyUI custom_nodes folder is unavailable or redirected.')
        return 1
    marker = target / '.lds-video-writer.json'
    if marker.is_symlink():
        log('Refusing a redirected ownership marker.')
        return 1
    if target.exists():
        try:
            if json.loads(marker.read_text()) != {'owner': 'video', 'pack': 'lds_h3_fast_writer'}:
                raise ValueError('Unrecognized pack')
        except (OSError, ValueError):
            log('The writer folder is not managed by Video; preserve it and choose another ComfyUI folder.')
            return 1
    # This code executes inside ComfyUI, not the LDS Python interpreter. Fetch
    # the pinned public sources and verify every byte before writing any file.
    files = {}
    try:
        for name, spec in WRITER_FILES.items():
            url = f"{TOOLS_ROOT}/{spec['sha256']}/{name}"
            if fetch is None:
                with urllib.request.urlopen(url, timeout=30) as response:
                    content = response.read(spec['bytes'] + 1)
            else:
                content = fetch(url)
            if len(content) != spec['bytes'] or hashlib.sha256(content).hexdigest() != spec['sha256']:
                log(f'Refused unexpected bytes for {name}; no writer files were changed.')
                return 1
            files[name] = content
    except OSError as exc:
        log(f'Could not download the writer source: {exc}')
        return 1
    target.mkdir(exist_ok=True)
    # Record ownership before copying, so an interrupted first installation
    # remains repairable without treating its own partial files as foreign.
    marker.write_text(json.dumps({'owner': 'video', 'pack': 'lds_h3_fast_writer'}))
    for name, content in files.items():
        dest = target / name
        if dest.is_symlink() or dest.resolve().parent != target.resolve():
            log('Refusing a redirected writer file.')
            return 1
        fd, temp_name = tempfile.mkstemp(prefix='.lds-writer-', dir=target)
        tmp = Path(temp_name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
    log('Fast MP4 writer installed. Restart ComfyUI when your current renders are finished, then refresh availability.')
    return 0
