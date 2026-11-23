"""Shared model discovery, preparation and readiness for trained image families.

Only ComfyUI loader-relative names leave this module. Discovery, Settings and
the installer use the same candidates so an existing asset is not downloaded
again just because the user stores it under an extra model root.
"""
from __future__ import annotations

import json
import logging
import math
import ntpath
import os
import re

from .. import config as cfg
from ..utils.trained_image_workflows import ASSET_SPECS, REQUIRED_NODES
from . import comfy_model_paths, model_integrity

logger = logging.getLogger(__name__)

FAMILY_LABELS = {'flux': 'FLUX.1', 'anima': 'Anima', 'qwenimage21': 'Qwen-Image 2.1'}

# Z-Image Turbo, the Studio's first and default family. Its files are found by its
# own resolvers (utils.comfyui.get_zimage_models for the base model,
# services.zimage_model_resolver for the text encoder and the VAE), so only its
# official downloads and its readiness join this module: the Studio banner,
# Settings and the installer then offer them like the other families' files.
#
# Source: Comfy-Org/Z-Image-Turbo (apache-2.0, not gated), the three files
# ComfyUI's own Z-Image Turbo template loads. Every URL is pinned to one revision
# and declares the published size and SHA256; the installer checks both before a
# download takes its place, so a moved or replaced file is never swapped in.
#
# Destinations: the base model goes under `diffusion_models/z image/` because
# get_zimage_models lists only files inside a Z-Image folder. The text encoder
# keeps ComfyUI's own flat name. The VAE gets a `z image/` folder: FLUX.1's
# action already writes `vae/ae.safetensors` (the same bytes, same SHA256 on the
# hub), and two actions on one path would carry different expectations. Each
# resolver still finds the other family's copy (FLUX.1 by its basename, Z-Image
# through any `ae.safetensors`), so an existing VAE is reused, not fetched twice.
ZIMAGE = 'zimage'
ZIMAGE_LABEL = 'Z-Image'
ZIMAGE_REPO = 'Comfy-Org/Z-Image-Turbo'
ZIMAGE_REVISION = '6fc90a3b1b653e935a0d175e260736de25b84df5'


def _zimage_asset(kind, folder, filename, size_bytes, sha256, label):
    repo_file = f'split_files/{kind}/{filename}'
    return {
        'kind': kind, 'folder': folder, 'filename': filename, 'repo_id': ZIMAGE_REPO,
        'url': f'https://huggingface.co/{ZIMAGE_REPO}/resolve/{ZIMAGE_REVISION}/{repo_file}',
        'size_bytes': size_bytes, 'sha256': sha256, 'label': label,
        'min_free_gb': math.ceil(size_bytes / 1024**3) + 1,
    }


ZIMAGE_ASSET_SPECS = {
    'diffusion_model': _zimage_asset(
        'diffusion_models', 'z image', 'z_image_turbo_bf16.safetensors', 12309866400,
        '2407613050b809ffdff18a4ac99af83ea6b95443ecebdf80e064a79c825574a6',
        'Turbo diffusion model'),
    'text_encoder': _zimage_asset(
        'text_encoders', '', 'qwen_3_4b.safetensors', 8044982048,
        '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a',
        'Qwen3 4B text encoder'),
    'vae': _zimage_asset(
        'vae', 'z image', 'ae.safetensors', 335304388,
        'afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38', 'VAE'),
}
# The config keys that pin a Z-Image file (zimage_model_resolver.CFG_KEY_*). The
# base model has none: the Studio picks it per run among get_zimage_models().
_ZIMAGE_PIN_KEYS = {'text_encoder': 'zimage.text_encoder', 'vae': 'zimage.vae'}


def _specs(family):
    if family not in ASSET_SPECS:
        raise ValueError('Unknown image generation family')
    return ASSET_SPECS[family]


def _relative_name(value):
    name = str(value or '').replace('\\', '/')
    if (not name or ntpath.isabs(name) or ':' in name or any(ord(c) < 32 for c in name)
            or ntpath.splitdrive(name)[0]
            or any(p in ('', '.', '..') for p in name.split('/'))):
        return None
    return name


def _key(name):
    return str(name).replace('\\', '/').casefold()


def _candidates(kind):
    return [(rel, ab) for rel, ab in comfy_model_paths.list_models(kind)
            if _relative_name(rel) and rel.lower().endswith(('.safetensors', '.sft'))]


def _looks_like_family(family, name):
    basename = str(name).replace('\\', '/').rsplit('/', 1)[-1]
    norm = re.sub(r'[^a-z0-9]+', '', basename.lower())
    if family == 'flux':
        return ('flux' in norm and not any(s in norm for s in ('flux2', 'klein', 'krea')))
    if family == 'anima':
        return 'anima' in norm
    # Qwen Image, Qwen Image Edit and Qwen 2.1 have different architectures.
    return 'qwenimage21' in norm and not any(s in norm for s in ('edit', 'rgba'))


def _pin(family, slot):
    return cfg.get(f'studio_models.{family}.{slot}', '') or ''


def list_family_models(family):
    """Installed base models, including a valid explicit pin for a renamed build."""
    _specs(family)
    pinned = _relative_name(_pin(family, 'diffusion_model'))
    names = [rel for rel, _ in _candidates('diffusion_models')
             if _looks_like_family(family, rel) or (pinned and _key(rel) == _key(pinned))]
    preferred = [_key(n) for n in _specs(family)['diffusion_model']['alternatives']]
    def priority(name):
        basename = _key(os.path.basename(name.replace('\\', '/')))
        return (0 if pinned and _key(name) == _key(pinned) else 1,
                preferred.index(basename) if basename in preferred else len(preferred),
                name.casefold())
    return [{'filename': name, 'displayName': os.path.basename(name.replace('\\', '/'))}
            for name in sorted(set(names), key=priority)]


def _resolve_slot(family, slot, chosen=None):
    spec = _specs(family)[slot]
    candidates = _candidates(spec['kind'])
    explicit = chosen if chosen is not None else _pin(family, slot)
    if explicit:
        normalized = _relative_name(explicit)
        if not normalized:
            raise ValueError(f'{FAMILY_LABELS[family]} {spec["label"]}: choose a model name inside ComfyUI model folders')
        exact = next((rel for rel, _ in candidates if _key(rel) == _key(normalized)), None)
        # Do not silently replace an absent explicit pin with another build.
        return exact or normalized
    alternatives = spec.get('alternatives') or (spec['filename'],)
    for preferred in alternatives:
        match = next((rel for rel, _ in candidates
                      if _key(os.path.basename(rel.replace('\\', '/'))) == _key(preferred)), None)
        if match:
            return match
    if slot == 'diffusion_model':
        models = list_family_models(family)
        if models:
            return models[0]['filename']
    # A diagnostic graph still needs a concrete expected filename. Readiness
    # and Studio's existing preflight block submission when it is absent.
    return spec['filename']


def resolve_family_assets(family, base_model=None):
    return {slot: _resolve_slot(family, slot, base_model if slot == 'diffusion_model' else None)
            for slot in _specs(family)}


generation_assets = resolve_family_assets


def install_action(family, slot):
    return f'studio_{family}_{slot}'


def _zimage_dest(spec):
    return (spec['kind'], *([spec['folder']] if spec['folder'] else []), spec['filename'])


def model_downloads():
    """Installer catalog, derived from the graph's official model specifications."""
    result = {}
    for slot, spec in ZIMAGE_ASSET_SPECS.items():
        result[install_action(ZIMAGE, slot)] = {
            'url': spec['url'], 'dest': _zimage_dest(spec), 'legacy_names': (),
            'label': f'{ZIMAGE_LABEL} {spec["label"]}',
            'min_free_gb': spec['min_free_gb'],
            'min_bytes': 1024 ** 3 if slot == 'diffusion_model' else 1024 ** 2,
            'expected_bytes': spec['size_bytes'], 'sha256': spec['sha256'],
            'gated': False, 'license_url': f'https://huggingface.co/{spec["repo_id"]}',
        }
    for family, specs in ASSET_SPECS.items():
        for slot, spec in specs.items():
            is_base = slot == 'diffusion_model'
            result[install_action(family, slot)] = {
                'url': spec['url'],
                'dest': (spec['kind'], family, spec['filename']) if is_base
                        else (spec['kind'], spec['filename']),
                'legacy_names': tuple(n for n in spec.get('alternatives', ()) if n != spec['filename']),
                'label': f'{FAMILY_LABELS[family]} {spec["label"]}',
                'min_free_gb': spec['min_free_gb'],
                'min_bytes': 1024 ** 3 if is_base else 1024 ** 2,
                'gated': bool(spec.get('gated', False)),
                'license_url': f'https://huggingface.co/{spec["repo_id"]}',
            }
    return result


def _asset_status(family, assets):
    missing, invalid = [], []
    for slot, spec in _specs(family).items():
        name = assets[slot]
        # Match full loader names, not the resolver's basename fallback: two
        # subfolders may hold different models with the same basename.
        path = next((ab for rel, ab in _candidates(spec['kind']) if _key(rel) == _key(name)), None)
        action = install_action(family, slot)
        if not path:
            missing.append({'slot': slot, 'name': name, 'kind': spec['kind'],
                            'label': spec['label'], 'action': action})
            continue
        check = model_integrity.validate_model_file(path)
        if check.get('blocking'):
            invalid.append({'asset': action, 'slot': slot, 'name': name,
                            'reason': check['reason'], 'blocking': True})
    return missing, invalid


def _zimage_dest_path(slot):
    """Absolute download destination, or None while no valid ComfyUI is configured."""
    from .. import setup_installer
    try:
        path = setup_installer._download_dest_path(install_action(ZIMAGE, slot))
    except (setup_installer.Precondition, OSError, ValueError):
        return None
    return os.path.normcase(os.path.abspath(path))


def _zimage_status():
    """One entry per slot, read through the resolvers the Z-Image Studio launches with.

    `name` is what a launch loads without a choice (the base model is chosen per
    run, and get_zimage_models()[0] is its default), `present` whether that file is
    on disk, `pinned` whether a setting names it and `broken` a blocking integrity
    verdict. Only a pinned file and the official download are checked: those are the
    files someone asked for, the second being the one a download can replace."""
    from ..utils.comfyui import get_zimage_models
    from . import zimage_model_resolver as resolver
    models = get_zimage_models()
    out = []
    for slot, spec in ZIMAGE_ASSET_SPECS.items():
        dest = _zimage_dest_path(slot)
        pin_key = _ZIMAGE_PIN_KEYS.get(slot)
        pinned = bool(pin_key and str(cfg.get(pin_key) or '').strip())
        if slot == 'diffusion_model':
            name = models[0] if models else None
            present = bool(models)
            official = _key(os.path.join(spec['folder'], spec['filename']))
            path = dest if dest and any(_key(m) == official for m in models) else None
        else:
            resolve = (resolver.resolve_zimage_text_encoder if slot == 'text_encoder'
                       else resolver.resolve_zimage_vae)
            name = resolve()
            listing = {_key(rel): ab for rel, ab in comfy_model_paths.list_models(spec['kind'])}
            path = listing.get(_key(name)) if name else None
            present = path is not None
        at_dest = bool(path and dest and os.path.normcase(os.path.abspath(path)) == dest)
        check = model_integrity.validate_model_file(path) if path and (pinned or at_dest) else {}
        out.append({'slot': slot, 'spec': spec, 'action': install_action(ZIMAGE, slot),
                    'name': name, 'present': present, 'pinned': pinned, 'pin_key': pin_key,
                    'at_dest': at_dest, 'broken': bool(check.get('blocking')),
                    'reason': check.get('reason')})
    return out


def _zimage_required_nodes():
    """Node classes of the shipped Z-Image graph, plus the loader each tested LoRA adds."""
    from .lora_test_studio import WORKFLOW_ZTURBO_PATH
    with open(WORKFLOW_ZTURBO_PATH, encoding='utf-8') as fh:
        graph = json.load(fh)
    return {node['class_type'] for node in graph.values()
            if isinstance(node, dict) and node.get('class_type')} | {'LoraLoaderModelOnly'}


def _zimage_readiness(*, check_nodes=True):
    """Z-Image's readiness, in generation_readiness()'s shape.

    It lists what LDS can provide: an absent file, or a broken official download,
    each with its one-click download. A pinned file that is absent or unloadable is
    not replaced by a download; it becomes a pin warning, and the launch preflight
    keeps naming that exact file, as it always has."""
    status = _zimage_status()
    missing = [{'slot': s['slot'], 'name': '/'.join(_zimage_dest(s['spec'])[1:]),
                'kind': s['spec']['kind'], 'label': s['spec']['label'], 'action': s['action']}
               for s in status if not s['pinned'] and not s['present']]
    invalid = [{'asset': s['action'], 'slot': s['slot'], 'name': s['name'],
                'reason': s['reason'], 'blocking': True}
               for s in status if not s['pinned'] and s['present'] and s['at_dest'] and s['broken']]
    pin_warnings = [
        {'slot': s['slot'], 'message': f'The file set in {s["pin_key"]} is unavailable or invalid. '
         'Fix that setting, or clear it to use automatic detection and the recommended download.'}
        for s in status if s['pinned'] and (not s['present'] or s['broken'])]
    classes = None
    if check_nodes:
        from ..utils.comfyui import fetch_object_info_classes
        classes = fetch_object_info_classes()
    missing_nodes = []
    if classes is not None:
        from .lora_test_studio import _node_class_present
        missing_nodes = sorted(c for c in _zimage_required_nodes()
                               if not _node_class_present(c, classes))
    actions = [m['action'] for m in missing] + [i['asset'] for i in invalid]
    repairs = {i['asset'] for i in invalid}
    downloads = [
        {'action': s['action'], 'slot': s['slot'], 'label': s['spec']['label'],
         'filename': s['spec']['filename'], 'size_bytes': s['spec']['size_bytes'],
         'source_url': f'https://huggingface.co/{s["spec"]["repo_id"]}',
         'repair': s['action'] in repairs}
        for s in status if s['action'] in actions]
    from .lora_test_studio import studio_missing_node_hints
    models_ready = not missing and not invalid
    return {'family': ZIMAGE, 'label': ZIMAGE_LABEL,
            'assets': {s['slot']: s['name'] or '' for s in status},
            'ready': (models_ready and not pin_warnings and not missing_nodes
                      and classes is not None),
            'models_ready': models_ready, 'nodes_checked': classes is not None,
            'missing_assets': missing, 'invalid_assets': invalid,
            'missing_nodes': missing_nodes, 'node_packs': studio_missing_node_hints(missing_nodes),
            'install_actions': actions, 'downloads': downloads, 'pin_warnings': pin_warnings}


def _zimage_readiness_or_none(*, check_nodes):
    """The Z-Image Studio is the default page: a disk or parse error in this check
    must never take it down. None leaves the page as it was before this check,
    with the launch preflight still naming any missing file."""
    try:
        return _zimage_readiness(check_nodes=check_nodes)
    except Exception:  # noqa: BLE001 - logged; the page keeps working without it
        logger.warning('Z-Image model check failed', exc_info=True)
        return None


def zimage_studio_readiness():
    """Z-Image's missing files and their downloads, for the Studio's model banner.

    No node probe: the Z-Image graph uses built-in ComfyUI nodes only and the
    launch preflight keeps checking them, as before. No `default_model` either:
    the Studio keeps defaulting to the first listed Z-Image base model."""
    return _zimage_readiness_or_none(check_nodes=False)


def missing_assets(family):
    if family == ZIMAGE:
        # The installer's question: does the engine resolve a file at all?
        return [s['action'] for s in _zimage_status() if not s['present']]
    missing, _ = _asset_status(family, resolve_family_assets(family))
    return [item['action'] for item in missing]


def invalid_assets(family):
    if family == ZIMAGE:
        # A broken file the engine would load vetoes the installer's "already installed".
        return [{'asset': s['action'], 'slot': s['slot'], 'name': s['name'],
                 'reason': s['reason'], 'blocking': True}
                for s in _zimage_status() if s['present'] and s['broken']]
    _, invalid = _asset_status(family, resolve_family_assets(family))
    return invalid


def generation_readiness(family, base_model=None, *, check_nodes=True):
    if family == ZIMAGE:
        return _zimage_readiness(check_nodes=check_nodes)
    _specs(family)
    try:
        assets = resolve_family_assets(family, base_model)
    except ValueError as exc:
        return {'family': family, 'label': FAMILY_LABELS[family], 'assets': {},
                'ready': False, 'models_ready': False, 'nodes_checked': False,
                'missing_assets': [], 'invalid_assets': [], 'missing_nodes': [],
                'install_actions': [], 'downloads': [], 'config_error': str(exc)}
    missing, invalid = _asset_status(family, assets)
    classes = None
    if check_nodes:
        from ..utils.comfyui import fetch_object_info_classes
        classes = fetch_object_info_classes()
    missing_nodes = sorted(set(REQUIRED_NODES[family]) - classes) if classes is not None else []
    broken_slots = {item['slot'] for item in (*missing, *invalid)}
    pin_warnings = [
        {'slot': slot, 'message': 'The selected file is unavailable or invalid. Choose another file, '
         'or clear this selection and save to install the recommended model.'}
        for slot in broken_slots if _pin(family, slot)]
    pinned_slots = {item['slot'] for item in pin_warnings}
    # A canonical download cannot repair a custom pin pointing somewhere else.
    actions = ([m['action'] for m in missing if m['slot'] not in pinned_slots]
               + [m['asset'] for m in invalid if m['slot'] not in pinned_slots])
    actions = list(dict.fromkeys(actions))
    repairs = {item['asset'] for item in invalid}
    downloads = [
        {'action': action, 'slot': slot, 'label': spec['label'],
         'filename': spec['filename'], 'size_bytes': spec['size_bytes'],
         'source_url': f'https://huggingface.co/{spec["repo_id"]}',
         'repair': action in repairs}
        for slot, spec in _specs(family).items()
        if (action := install_action(family, slot)) in actions]
    return {'family': family, 'label': FAMILY_LABELS[family], 'assets': assets,
            'ready': not missing and not invalid and not missing_nodes and classes is not None,
            'models_ready': not missing and not invalid,
            'nodes_checked': classes is not None,
            'missing_assets': missing, 'invalid_assets': invalid,
            'missing_nodes': missing_nodes, 'install_actions': actions, 'downloads': downloads,
            'pin_warnings': pin_warnings}


def settings_catalog():
    """Small on-demand Settings payload; never starts a download or a render.

    Z-Image slots are not `configurable`: its files are detected (the
    zimage.text_encoder and zimage.vae settings pin them) and the Studio picks
    its base model per run, so Settings shows what is found and the downloads."""
    zimage = _zimage_readiness_or_none(check_nodes=True)
    out = [{**zimage, 'slots': [
        {'key': slot, 'label': spec['label'], 'filename': spec['filename'],
         'kind': spec['kind'], 'action': install_action(ZIMAGE, slot),
         'source_url': f'https://huggingface.co/{spec["repo_id"]}',
         'files': [], 'configurable': False}
        for slot, spec in ZIMAGE_ASSET_SPECS.items()]}] if zimage else []
    for family, specs in ASSET_SPECS.items():
        status = generation_readiness(family)
        out.append({**status, 'slots': [
            {'key': slot, 'label': spec['label'], 'filename': spec['filename'],
             'kind': spec['kind'], 'action': install_action(family, slot),
             'source_url': f'https://huggingface.co/{spec["repo_id"]}',
             'files': ([m['filename'] for m in list_family_models(family)] if slot == 'diffusion_model'
                       else [rel for rel, _ in _candidates(spec['kind'])])}
            for slot, spec in specs.items()]})
    return out
