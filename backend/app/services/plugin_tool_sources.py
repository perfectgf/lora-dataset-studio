"""LDS tool distribution endpoints for plugin worker and ComfyUI installations."""
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = 'https://lds.perfectgf.fr/models/tools'
PYPI = ROOT + '/pypi/simple/'
CPU = ROOT + '/pytorch/cpu/'
MODEL_CONSTRAINTS = str(Path(__file__).with_name('author_model_constraints.txt'))


# Full identity of the public Video h3-refmods recipe after the archive was stripped of examples.
H3MOD_RECIPE_DIGESTS = frozenset({'f43a31548b6f16accecc8027454ae454fdb7d57bcfd1f612916bbd0c662939bc'})
_H3MOD_ORIGINAL = '7e1f5d3f47fd3639de03c23ee4340af5cd9b6f0e721cf7f60af60e99eceac1b2'
_H3MOD_INTERIM = 'deae752b66bba77d81bbbbca5692e967174bc78699a3ba878be5bbecf246730b'
_H3MOD_FINAL = '40fa27ee131a635bf3eea1758f6450dade42b15772975a75254ac418e8763f54'
# Fingerprints of the full {relative path: SHA-256} inventories after extraction.
# A migrated receipt retains any already installed example and nested release.
_H3MOD_ORIGINAL_FILES = '0f6c923ba136015d7fe14574416973cc4406f0505b1a287a9677217dc8db210a'
_H3MOD_INTERIM_FILES = '9f349735194da76813e17df557cc868ceb4d7a06adef223d5ccd45ec391b8459'
_H3MOD_FINAL_FILES = '9e88d27509a11becd9450adbad5b7161814f481f056b18265e80f1ddeb748a58'
_H3MOD_ARCHIVES = {
    _H3MOD_ORIGINAL: (
        'https://codeload.github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/zip/22ca77e247375dd47fc7340286b5f391887d3410',
        {_H3MOD_ORIGINAL_FILES}),
    _H3MOD_INTERIM: (f'{ROOT}/{_H3MOD_INTERIM}/ComfyUI-MiniMaxH3Mod.zip',
                     {_H3MOD_INTERIM_FILES}),
    _H3MOD_FINAL: (f'{ROOT}/{_H3MOD_FINAL}/ComfyUI-MiniMaxH3Mod.zip',
                   {_H3MOD_ORIGINAL_FILES, _H3MOD_INTERIM_FILES, _H3MOD_FINAL_FILES}),
}


def matches_h3mod_receipt(target, receipt, digest):
    """Verify the recorded owner recipe and inventory against reviewed releases."""
    recorded = target
    if receipt['recipe_id'] != target['id'] or digest(recorded) not in H3MOD_RECIPE_DIGESTS:
        return False
    files = digest(receipt['files'])
    for sha256, (url, inventories) in _H3MOD_ARCHIVES.items():
        if files in inventories:
            fields = {**recorded, 'sha256': sha256, 'url': url}
            # Receipts created before the Python constraint field used the
            # default interpreter range, which both reviewed recipes keep.
            older_fields = {k: v for k, v in fields.items() if k != 'python'}
            if receipt['recipe_sha256'] in (digest(fields), digest(older_fields)):
                return True
    return False


LEGACY_NODE_ORIGINS = {
    '04c61842bc00fd8673e6bc9a3b1b1935955461f363791070ed14d67d2a2e77fb':
        'https://codeload.github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/zip/4490bd1f482e026674543386bb2a4d176da245b9',
    '5db98923f741bafd1beb56649df23a0f7981055ab482b3c34da9778083e995df':
        'https://codeload.github.com/nicolab28/ComfyUI-ClipProj/zip/c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c',
    '7a7f71461e8196665863802a37dbfbf6bd78ce7fd0f9d33e7ffa4130c282de1c':
        'https://codeload.github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/zip/5161f0457bc8c52535212d6783eee73f439e1537',
    'ff1102a1d17f597d8b100afd17bd489a7910325729483398f1385c541f18c171':
        'https://codeload.github.com/lbouaraba/comfyui-krea2edit/zip/86f886dac23013d88996e3a2e99093ba44d322fb',
}


def is_mirrored_wheel(url):
    parsed = urlsplit(url)
    return (parsed.scheme == 'https' and parsed.netloc == 'lds.perfectgf.fr'
            and parsed.path.startswith('/models/tools/packages/')
            and parsed.path.lower().endswith('.whl') and not parsed.query and not parsed.fragment
            and not any(char.isspace() for char in url))


def is_installer_wheel(name, url, digest):
    # The author distributes the shot detector and its weights in one wheel.
    # The only download outside LDS must match that exact file and SHA-256.
    if name == 'transnetv2-pytorch':
        pinned = Path(MODEL_CONSTRAINTS).read_text(encoding='ascii').strip()
        return pinned == f'{name} @ {url}#sha256={digest}'
    return is_mirrored_wheel(url)


def pip_install(command, environment):
    """Keep local wheel installs offline and replace upstream indexes for network installs."""
    if not any(command[i:i + 2] == ['-m', 'pip'] for i in range(len(command) - 1)) or 'install' not in command:
        return command, environment
    env = dict(environment if environment is not None else os.environ)
    for key in list(env):
        if key.upper() in {'PIP_EXTRA_INDEX_URL', 'PIP_FIND_LINKS', 'PIP_NO_INDEX', 'PIP_TRUSTED_HOST'}:
            del env[key]
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL=PYPI, PIP_DISABLE_PIP_VERSION_CHECK='1')
    args = [PYPI if arg == 'https://pypi.org/simple' else
            CPU if arg == 'https://download.pytorch.org/whl/cpu' else arg for arg in command]
    if '--no-index' not in args:
        if '--index-url' not in args:
            args += ['--index-url', PYPI]
        if MODEL_CONSTRAINTS not in args:
            args += ['--constraint', MODEL_CONSTRAINTS]
    return args, env
