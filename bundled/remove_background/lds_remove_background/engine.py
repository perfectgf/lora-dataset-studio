"""Readiness of the local models, and admission of one step to the host's GPU queue."""
from __future__ import annotations

import os
import uuid

from lds_sdk import comfy, config, local_render

from . import assets, graphs

PLUGIN = 'remove_background'
JOB_KIND = 'is_remove_background'
SETTINGS = 'Plugins → Remove Background → Settings'


class NotReady(RuntimeError):
    """ComfyUI cannot run this cut-out yet; ``missing`` names the install actions that help."""

    def __init__(self, message, missing=()):
        super().__init__(message)
        self.missing = list(missing)


def _missing_nodes(classes, keep, kind):
    return sorted(graphs.nodes_for(keep, kind) - classes)


def status():
    """Everything the editor and the settings need to say what can run and what is missing."""
    report = assets.inspect_models()
    classes = local_render.fetch_object_info_classes()
    configured = bool(config.comfyui_dir('input') and config.comfyui_dir('output'))
    reachable = configured and classes is not None
    result = {
        'configured': configured, 'reachable': reachable,
        'models': {slot: {**report[slot], 'action': spec['action'], 'name': spec['name'],
                          'bytes': spec['bytes'], 'license': spec['license'],
                          'license_url': spec['license_url']}
                   for slot, spec in assets.MODELS.items()},
        'kinds': {kind: {'models': list(slots), 'ready': not assets.unready(report, slots),
                         'missing_nodes': _missing_nodes(classes, 'subject', kind) if reachable else []}
                  for kind, slots in assets.KINDS.items()},
        'find': {'models': ['sam'], 'ready': not assets.unready(report, ('sam',)),
                 'missing_nodes': sorted(graphs.FIND_NODES - classes) if reachable else []},
    }
    result['groups'] = {'remove_background': assets.unready(report, assets.BASE_GROUP),
                        'remove_background_fine': assets.unready(report, assets.FINE_GROUP)}
    return result


def probe():
    """The Setup row: the everyday cut-out (BiRefNet) can run."""
    current = status()
    photo = current['kinds']['photo']
    ok = current['reachable'] and photo['ready'] and not photo['missing_nodes']
    if not current['configured'] or not current['reachable']:
        detail = 'Configure and start ComfyUI, then check Remove Background again.'
    elif photo['missing_nodes']:
        detail = 'Update ComfyUI and restart it when idle: it has no native background removal yet.'
    elif not photo['ready']:
        detail = f'Install its models in {SETTINGS}.'
    else:
        detail = 'Ready to remove backgrounds.'
    return {'ok': ok, 'detail': detail, 'missing': current['groups']['remove_background'],
            'missing_nodes': photo['missing_nodes'],
            'models': {slot: item['state'] == 'ready' for slot, item in current['models'].items()}}


def require(keep, kind):
    """The loader names for one cut-out, or ``NotReady`` with what to do about it."""
    report = assets.inspect_models()
    slots = assets.required(keep, kind)
    require_queue()
    nodes = _missing_nodes(local_render.fetch_object_info_classes() or set(), keep, kind)
    if nodes:
        raise NotReady('Update ComfyUI and restart it when idle. Missing native nodes: ' + ', '.join(nodes) + '.')
    missing = assets.unready(report, slots)
    if missing:
        names = [assets.MODELS[slot]['name'] for slot in slots if assets.MODELS[slot]['action'] in missing]
        raise NotReady('Install ' + ', '.join(names) + ' first.', missing)
    return {slot: report[slot]['file'] for slot in slots}


def require_queue():
    """ComfyUI can take a job at all, whatever its models."""
    if not (config.comfyui_dir('input') and config.comfyui_dir('output')):
        raise NotReady('Connect your ComfyUI folder in Settings first.')
    if local_render.fetch_object_info_classes() is None:
        raise NotReady('ComfyUI does not answer. Start it, then try again.')


def _stage(picture, folder, name):
    """Put a picture in ComfyUI's input folder under a name only this step uses."""
    input_dir = comfy.ensure_input_usable(str(config.comfyui_dir('input')))
    local = folder / name
    picture.save(local, 'PNG', compress_level=1)
    try:
        return comfy.stage_input_image(str(local), name, input_dir)
    finally:
        local.unlink(missing_ok=True)


def submit(user_id, job_id, picture, folder, build, *, label, metadata):
    """Queue one step. ``build(image, prefix)`` returns its graph."""
    name = f'lds_remove_background_{job_id}.png'
    workflow = build(name, f'LDS_RemoveBackground_{job_id}')
    staged = _stage(picture, folder, name)
    try:
        comfy.add_plugin_job(PLUGIN, JOB_KIND, user_id=str(user_id), workflow_data=workflow, prompt=label,
                             job_id=job_id, metadata={**metadata, 'staged_inputs': [name]})
    except Exception:
        # Only this step's own staged file is ours; admitted work is cleaned up by the host.
        try:
            os.remove(staged)
        except OSError:
            pass
        raise


def submit_candidate(user_id, picture, folder, *, dataset_id):
    """Queue the finished picture as a candidate of its dataset; the host files the result."""
    job_id = uuid.uuid4().hex
    name = f'lds_remove_background_{job_id}.png'
    staged = _stage(picture, folder, name)
    try:
        comfy.queue.add_job(
            job_type='image', user_id=str(user_id), prompt='Background removed', job_id=job_id,
            workflow_data=graphs.copy(name, f'LDS_RemoveBackground_{job_id}'),
            metadata={'model_name': PLUGIN, 'is_dataset': True, 'dataset_id': dataset_id,
                      'dataset_engine_plugin': PLUGIN, 'engine_label': 'Remove Background',
                      'variation_label': 'Background removed', 'staged_inputs': [name]})
    except Exception:
        try:
            os.remove(staged)
        except OSError:
            pass
        raise
    return job_id


def output_path(filename):
    """The saved answer of a finished step, or None."""
    folder = config.comfyui_dir('output')
    if not folder or not filename:
        return None
    path = os.path.join(str(folder), os.path.basename(str(filename)))
    return path if os.path.isfile(path) else None
