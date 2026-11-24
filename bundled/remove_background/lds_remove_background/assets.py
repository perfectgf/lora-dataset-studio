"""The model files of Remove Background: what each is for, where it is, how to fetch it.

Two kinds of model share the work. SAM 3.1 finds what to keep from words or
clicks and answers with a hard shape. A matting model cuts the soft edge and
cannot be told which object is meant: BiRefNet for photos, Lucida for graphics
and glass, RMBG 3.0 for single strands of hair and fur. All of them run on
nodes that ship with ComfyUI, so only weights are installed.

Files already present under any ComfyUI model root, subfolders included, are
found and reused; preparation downloads only what is missing.
"""
from __future__ import annotations

import math
from pathlib import Path

from lds_sdk import models

_BIREFNET = 'https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/'
_RMBG = 'https://huggingface.co/briaai/RMBG-3.0-Comfy/resolve/main/'

MODELS = {
    'birefnet': {
        'action': 'remove_bg_birefnet', 'name': 'BiRefNet', 'folder': 'background_removal',
        'filename': 'birefnet.safetensors', 'alternatives': (),
        'url': _BIREFNET + 'birefnet.safetensors',
        'bytes': 444_473_596,
        'sha256': '9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154',
        'license': 'MIT', 'license_url': 'https://huggingface.co/Comfy-Org/BiRefNet',
    },
    'sam': {
        'action': 'remove_bg_sam', 'name': 'SAM 3.1', 'folder': 'checkpoints',
        'filename': 'sam3.1_multiplex_fp16.safetensors', 'alternatives': (),
        'url': 'https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors',
        'bytes': 1_745_546_848,
        'sha256': '9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03',
        'license': 'SAM License', 'license_url': 'https://huggingface.co/Comfy-Org/sam3.1/blob/main/LICENSE',
    },
    'lucida': {
        'action': 'remove_bg_lucida', 'name': 'Lucida', 'folder': 'background_removal',
        'filename': 'lucida.safetensors', 'alternatives': (),
        'url': _BIREFNET + 'lucida.safetensors',
        'bytes': 884_878_856,
        'sha256': 'a43e3dd84cbe84db76bd2015ef8f7034c9503c50d9b56fa2155803b69bd2f408',
        'license': 'MIT', 'license_url': 'https://github.com/egeorcun/lucida#license',
    },
    'rmbg': {
        'action': 'remove_bg_rmbg', 'name': 'RMBG 3.0', 'folder': 'diffusion_models',
        'filename': 'rmbg3_int8_convrot.safetensors', 'alternatives': ('rmbg3_bf16.safetensors',),
        'url': _RMBG + 'diffusion_models/rmbg3_int8_convrot.safetensors',
        'bytes': 8_408_562_474,
        'sha256': 'bc29b9a87ed378596c5ca5bc0c5bdfaa8aabc8415d33f3c767f9a3cbd9fd566d',
        'license': 'CC BY-NC 4.0, non-commercial',
        'license_url': 'https://huggingface.co/briaai/RMBG-3.0-Comfy',
    },
    'rmbg_text_encoder': {
        # The Qwen3-VL 4B encoder Krea 2 loads too: any official build reads the same fixed instruction.
        'action': 'remove_bg_rmbg_text_encoder', 'name': 'RMBG 3.0 text encoder', 'folder': 'text_encoders',
        'filename': 'qwen3vl_4b_int8_convrot.safetensors',
        'alternatives': ('qwen3vl_4b_fp8_scaled.safetensors', 'qwen3vl_4b_bf16.safetensors'),
        'url': _RMBG + 'text_encoders/qwen3vl_4b_int8_convrot.safetensors',
        'bytes': 4_858_559_240,
        'sha256': '32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c',
        'license': 'Distributed with RMBG 3.0', 'license_url': 'https://huggingface.co/briaai/RMBG-3.0-Comfy',
    },
    'rmbg_vae': {
        # FLUX.2's VAE, which the Klein engine installs under the same name.
        'action': 'remove_bg_rmbg_vae', 'name': 'RMBG 3.0 VAE', 'folder': 'vae',
        'filename': 'flux2-vae.safetensors', 'alternatives': (),
        'url': _RMBG + 'vae/flux2-vae.safetensors',
        'bytes': 336_211_292,
        'sha256': '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3',
        'license': 'Distributed with RMBG 3.0', 'license_url': 'https://huggingface.co/briaai/RMBG-3.0-Comfy',
    },
}
ACTIONS = tuple(spec['action'] for spec in MODELS.values())

# What kind of picture it is decides which matting model cuts the edge.
KINDS = {
    'photo': ('birefnet',),
    'fine': ('rmbg', 'rmbg_text_encoder', 'rmbg_vae'),
    'graphic': ('lucida',),
}
# What says which part stays. SAM is needed only when the user points at something.
KEEPS = ('subject', 'selection', 'describe', 'points')
TARGETED = ('describe', 'points')

# The one-click sets: the everyday tool, and the heavy optional matting model.
BASE_GROUP = ('birefnet', 'sam', 'lucida')
FINE_GROUP = KINDS['fine']


def required(keep, kind):
    """The model slots one cut-out needs."""
    return KINDS[kind] + (('sam',) if keep in TARGETED else ())


def _basename(name):
    return Path(str(name).replace('\\', '/')).name.casefold()


def _depth(name):
    return len(Path(str(name).replace('\\', '/')).parts)


def _matches(slot):
    """(loader name, path) of every official file of ``slot`` on disk, the preferred one first."""
    spec = MODELS[slot]
    candidates = models.list_models(spec['folder'])
    found = []
    for filename in (spec['filename'], *spec['alternatives']):
        same = [item for item in candidates if _basename(item[0]) == filename.casefold()]
        found.extend(sorted(same, key=lambda item: (_depth(item[0]), item[0])))
    return found


def inspect_models():
    """``{slot: {'file', 'state'}}``; state is ready, missing or damaged."""
    report = {}
    for slot in MODELS:
        found = _matches(slot)
        if not found:
            report[slot] = {'file': None, 'state': 'missing'}
            continue
        name, path = found[0]
        verdict = models.validate_model_file(path)
        damaged = verdict.get('blocking') or verdict.get('verdict') == 'missing'
        report[slot] = {'file': name, 'state': 'damaged' if damaged else 'ready'}
    return report


def unready(report, slots):
    """The install actions that would make ``slots`` usable."""
    return [MODELS[slot]['action'] for slot in slots if report[slot]['state'] != 'ready']


def missing(slots):
    """Setup's view: the install actions whose file is not there."""
    report = inspect_models()
    return [MODELS[slot]['action'] for slot in slots if report[slot]['state'] == 'missing']


def damaged(slots):
    """Setup's view: files that are there and cannot be used."""
    report = inspect_models()
    return [{'asset': MODELS[slot]['action'], 'blocking': True,
             'reason': 'The model file is incomplete or unreadable.'}
            for slot in slots if report[slot]['state'] == 'damaged']


def _download_roots(slot):
    """Where the host installer may find an official file that is already there."""
    roots = list(models.search_roots(MODELS[slot]['folder']))
    for _name, path in _matches(slot):
        parent = str(Path(path).parent)
        if parent not in roots:
            roots.append(parent)
    return roots


def register(ctx):
    for slot, spec in MODELS.items():
        ctx.register_model_download(
            spec['action'], url=spec['url'], dest=(spec['folder'], spec['filename']),
            min_free_gb=math.ceil(spec['bytes'] / 1024 ** 3) + 1,
            min_bytes=spec['bytes'] // 2, expected_bytes=spec['bytes'], sha256=spec['sha256'],
            legacy_names=spec['alternatives'], license_url=spec['license_url'],
            extra_roots=lambda slot=slot: _download_roots(slot))
    for group, slots in (('remove_background', BASE_GROUP), ('remove_background_fine', FINE_GROUP)):
        ctx.register_install_group(group, tuple(MODELS[slot]['action'] for slot in slots),
                                   missing_key=group + '_missing', invalid_key=group + '_invalid')
