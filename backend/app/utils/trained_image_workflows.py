"""Native ComfyUI text-to-image graphs for the remaining image training families.

Asset discovery and installation belong to ``services.trained_image_models``.
This module accepts only server-resolved relative filenames and explicit model /
LoRA allowlists. The calling service checks LoRA tensor architecture before enqueue.

Graph and asset references:
https://github.com/Comfy-Org/workflow_templates/blob/main/templates/flux_dev_full_text_to_image.json
https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_anima_base_v1.json
https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json
"""
from __future__ import annotations

import math


def _asset(kind, filename, repo_id, repo_file=None, *, alternatives=(), label=None):
    repo_file = repo_file or filename
    return {
        'kind': kind, 'filename': filename, 'repo_id': repo_id,
        'repo_file': repo_file,
        'url': f'https://huggingface.co/{repo_id}/resolve/main/{repo_file}',
        'alternatives': (filename, *alternatives), 'label': label or filename,
    }


ASSET_SPECS = {
    'moss': {
        'diffusion_model': _asset('diffusion_models', 'moss_image_sft_bf16.safetensors',
                                 'Comfy-Org/MOSS-Image', 'diffusion_models/moss_image_sft_bf16.safetensors',
                                 alternatives=('moss_image_base_bf16.safetensors',), label='MOSS-Image SFT BF16'),
        'text_encoder': _asset('text_encoders', 'qwen3.5_9b_int8_convrot.safetensors',
                              'Comfy-Org/MOSS-Image', 'text_encoders/qwen3.5_9b_int8_convrot.safetensors',
                              label='Qwen3.5 9B INT8'),
        'vae': _asset('vae', 'flux2-vae.safetensors', 'Comfy-Org/MOSS-Image',
                      'vae/flux2-vae.safetensors', label='FLUX.2 VAE'),
    },
    'flux': {
        'diffusion_model': _asset('diffusion_models', 'flux1-dev.safetensors',
                                 'Comfy-Org/flux1-dev', label='FLUX.1 dev diffusion model'),
        'text_encoder': _asset('text_encoders', 't5xxl_fp16.safetensors',
                              'comfyanonymous/flux_text_encoders',
                              alternatives=('t5xxl_fp8_e4m3fn_scaled.safetensors',
                                            't5xxl_fp8_e4m3fn.safetensors'), label='T5 XXL'),
        'text_encoder_2': _asset('text_encoders', 'clip_l.safetensors',
                                'comfyanonymous/flux_text_encoders', label='CLIP-L'),
        'vae': _asset('vae', 'ae.safetensors', 'Comfy-Org/Lumina_Image_2.0_Repackaged',
                      'split_files/vae/ae.safetensors', label='FLUX.1 VAE'),
    },
    'anima': {
        'diffusion_model': _asset('diffusion_models', 'anima-base-v1.0.safetensors',
                                 'circlestone-labs/Anima',
                                 'split_files/diffusion_models/anima-base-v1.0.safetensors',
                                 label='Anima Base v1.0'),
        'text_encoder': _asset('text_encoders', 'qwen_3_06b_base.safetensors',
                              'circlestone-labs/Anima',
                              'split_files/text_encoders/qwen_3_06b_base.safetensors',
                              label='Anima Qwen3 0.6B'),
        'vae': _asset('vae', 'qwen_image_vae.safetensors', 'circlestone-labs/Anima',
                      'split_files/vae/qwen_image_vae.safetensors', label='Anima VAE'),
    },
    'qwenimage21': {
        'diffusion_model': _asset('diffusion_models', 'qwen_image_2.1_int8_convrot.safetensors',
                                 'Comfy-Org/Qwen-Image-2.1',
                                 'diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
                                 alternatives=('qwen_image_2.1_bf16.safetensors',),
                                 label='Qwen-Image 2.1 diffusion model'),
        'text_encoder': _asset('text_encoders', 'qwen3vl_8b_int8_convrot.safetensors',
                              'Comfy-Org/Qwen-Image-2.1',
                              'text_encoders/qwen3vl_8b_int8_convrot.safetensors',
                              alternatives=('qwen3vl_8b_bf16.safetensors',),
                              label='Qwen-Image 2.1 Qwen3-VL 8B'),
        'vae': _asset('vae', 'qwen_image_2.1_vae_bf16.safetensors',
                      'Comfy-Org/Qwen-Image-2.1',
                      'vae/qwen_image_2.1_vae_bf16.safetensors', label='Qwen-Image 2.1 RGBA VAE'),
    },
}

# Remote sizes from the publishers' Hugging Face tree API. These prevent the
# installer from requiring a full diffusion model's disk budget for tiny VAEs.
_ASSET_BYTES = {
    'moss': {'diffusion_model': 10164655808, 'text_encoder': 9748121532, 'vae': 336211292},
    'flux': {'diffusion_model': 23802932552, 'text_encoder': 9787841024,
             'text_encoder_2': 246144152, 'vae': 335304388},
    'anima': {'diffusion_model': 4182218328, 'text_encoder': 1192135096,
              'vae': 253806246},
    'qwenimage21': {'diffusion_model': 7256783064, 'text_encoder': 9350798360,
                    'vae': 675509688},
}
for _fam, _slots in _ASSET_BYTES.items():
    for _slot, _size in _slots.items():
        ASSET_SPECS[_fam][_slot]['size_bytes'] = _size
        ASSET_SPECS[_fam][_slot]['min_free_gb'] = math.ceil(_size / 1024**3) + 1

DEFAULTS = {
    'moss': {'steps': 50, 'cfg': 3.0, 'sampler': 'euler', 'scheduler': 'sgm_uniform'},
    'flux': {'steps': 20, 'cfg': 3.5, 'sampler': 'euler', 'scheduler': 'simple'},
    'anima': {'steps': 30, 'cfg': 4.0, 'sampler': 'euler', 'scheduler': 'simple'},
    'qwenimage21': {'steps': 25, 'cfg': 1.0, 'sampler': 'euler', 'scheduler': 'simple'},
}

# MOSS-Image extrapolates "from 0.25M (512×512) to 4M (2K)" pixels (OpenMOSS user guide):
# no Studio size goes past 2048², tier multiplier included.
MOSS_MAX_PIXELS = 2048 * 2048

_COMMON_NODES = ('UNETLoader', 'VAELoader', 'KSampler', 'VAEDecode', 'SaveImage',
                 'LoraLoaderModelOnly')
REQUIRED_NODES = {
    'moss': ('LDSMossUNETLoader', 'LDSMossCLIPLoader', 'VAELoader', 'CLIPTextEncode',
             'EmptyFlux2LatentImage', 'ModelSamplingSD3', 'SkipLayerGuidanceDiTSimple',
             'KSampler', 'VAEDecode', 'SaveImage'),
    'flux': (*_COMMON_NODES, 'DualCLIPLoader', 'CLIPTextEncode', 'FluxGuidance',
             'ModelSamplingFlux', 'EmptySD3LatentImage'),
    'anima': (*_COMMON_NODES, 'CLIPLoader', 'CLIPTextEncode', 'EmptyLatentImage'),
    'qwenimage21': (*_COMMON_NODES, 'CLIPLoader', 'TextEncodeQwenImage21', 'EmptyLatentImage'),
}


def _family(family):
    family = str(family or '').lower()
    if family not in DEFAULTS:
        raise ValueError('Unsupported trained image family')
    return family


def family_defaults(family):
    """Return sampling defaults; FLUX cfg controls distilled guidance."""
    return dict(DEFAULTS[_family(family)])


def required_node_classes(family, *, with_loras=True):
    nodes = set(REQUIRED_NODES[_family(family)])
    if not with_loras:
        nodes.discard('LoraLoaderModelOnly')
    return nodes


def _safe_name(value, label):
    """Reject paths escaping ComfyUI roots even if a bad allowlist contains one."""
    if not isinstance(value, str) or not value or value.startswith(('/', '\\')):
        raise ValueError(f'{label} must be a relative model filename')
    parts = value.replace('\\', '/').split('/')
    if ':' in value or any(ord(c) < 32 for c in value) or any(p in ('', '.', '..') for p in parts):
        raise ValueError(f'{label} must be a relative model filename')
    if not value.lower().endswith(('.safetensors', '.sft')):
        raise ValueError(f'{label} requires a safetensors model supported by native ComfyUI')
    return value


def _number(value, default, low, high, label, *, integer=False):
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid {label}') from exc
    if not math.isfinite(number) or number < low or number > high:
        raise ValueError(f'{label} must be between {low} and {high}')
    if integer and not number.is_integer():
        raise ValueError(f'{label} must be an integer')
    return int(number) if integer else number


def moss_capped_dims(width, height):
    """(width, height) scaled down, ratio kept, to at most MOSS_MAX_PIXELS, on multiples of 16."""
    if width * height <= MOSS_MAX_PIXELS:
        return width, height
    scale = math.sqrt(MOSS_MAX_PIXELS / (width * height))
    return max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)


def build_trained_image_workflow(
    family, *, assets, allowed_bases, allowed_loras, base_model=None, loras=(),
    prompt='', negative='', seed=0, width=1024, height=1024, cfg=None, steps=None,
    batch_size=1, filename_prefix='LDS', sampler=None, scheduler=None,
    weight_dtype=None, available_classes=None,
):
    """Build a native prompt graph, refusing unknown selections and architectures.

    ``assets`` maps ASSET_SPECS keys to resolved filenames. ``allowed_loras``
    must be the caller's family/ownership-filtered pool, checked against tensor
    metadata by its normal preflight. All trained adapters affect the diffusion
    model: Anima's trained text conditioner is inside that model (llm_adapter),
    and Qwen 2.1's toolkit exports ComfyUI-prefixed transformer keys. Loading them
    through an older Qwen encoder or a Z-Image graph would silently drop weights.
    A MOSS picture never exceeds MOSS_MAX_PIXELS.
    """
    family = _family(family)
    defaults = family_defaults(family)
    if family == 'moss':
        width, height = moss_capped_dims(width, height)
    if not isinstance(assets, dict):
        raise ValueError('Generation assets must be resolved before building a workflow')
    resolved = {}
    for key in ASSET_SPECS[family]:
        value = base_model if key == 'diffusion_model' and base_model else assets.get(key)
        resolved[key] = _safe_name(value, ASSET_SPECS[family][key]['label'])
    if allowed_bases is None or resolved['diffusion_model'] not in allowed_bases:
        raise ValueError('Unknown base model for this image family')
    if not isinstance(loras, (list, tuple)):
        raise ValueError('LoRAs must be a list')
    if loras and allowed_loras is None:
        raise ValueError('LoRA selections require an explicit family allowlist')
    if family == 'moss' and loras:
        raise ValueError('MOSS generation currently uses the base model only')

    width = _number(width, 1024, 64, 4096, 'width', integer=True)
    height = _number(height, 1024, 64, 4096, 'height', integer=True)
    multiple = 32 if family == 'qwenimage21' else 16
    if width % multiple or height % multiple:
        raise ValueError(f'{family} image dimensions must be multiples of {multiple}')
    batch_size = _number(batch_size, 1, 1, 64, 'batch size', integer=True)
    # Do not convert seeds through float: ComfyUI supports integers above 2**53.
    try:
        parsed_seed = int(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid seed') from exc
    if parsed_seed < 0 or parsed_seed > 2**64 - 1 or str(parsed_seed) != str(seed):
        raise ValueError('Seed must be an unsigned 64-bit integer')
    steps_defaulted = steps is None
    steps = _number(steps, defaults['steps'], 1, 100, 'steps', integer=True)
    cfg = _number(cfg, defaults['cfg'], 0, 30, 'guidance')
    dtype = weight_dtype or 'default'
    if dtype not in ('default', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'):
        raise ValueError('Unsupported diffusion model weight dtype')
    # Prequantized Qwen convrot tensors carry their own quantization metadata.
    if family == 'qwenimage21' and 'int8' in resolved['diffusion_model'].lower():
        dtype = 'default'
    workflow = {
        '1': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': resolved['diffusion_model'], 'weight_dtype': dtype}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': resolved['vae']}},
        '6': {'class_type': 'EmptySD3LatentImage' if family == 'flux' else 'EmptyLatentImage',
              'inputs': {'width': width, 'height': height, 'batch_size': batch_size}},
    }
    if family == 'moss':
        workflow['1'] = {'class_type': 'LDSMossUNETLoader', 'inputs': {
            'unet_name': resolved['diffusion_model']}}
        workflow['6']['class_type'] = 'EmptyFlux2LatentImage'
    previous = '1'
    for index, item in enumerate(loras):
        if not isinstance(item, dict):
            raise ValueError('Invalid LoRA selection')
        name = _safe_name(item.get('filename'), 'LoRA')
        if name not in allowed_loras:
            raise ValueError('Unknown LoRA for this image family')
        if item.get('family') and item['family'] != family:
            raise ValueError('LoRA architecture does not match the image family')
        strength = _number(item.get('strength'), 1.0, -2.0, 6.0, 'LoRA strength')
        node_id = f'lora_{index}'
        workflow[node_id] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': [previous, 0], 'lora_name': name, 'strength_model': strength}}
        previous = node_id

    if family == 'moss':
        workflow['2'] = {'class_type': 'LDSMossCLIPLoader', 'inputs': {
            'clip_name': resolved['text_encoder']}}
    elif family == 'flux':
        workflow['2'] = {'class_type': 'DualCLIPLoader', 'inputs': {
            'clip_name1': resolved['text_encoder_2'], 'clip_name2': resolved['text_encoder'],
            'type': 'flux', 'device': 'default'}}
    else:
        workflow['2'] = {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': resolved['text_encoder'],
            'type': 'qwen_image' if family == 'qwenimage21' else 'stable_diffusion',
            'device': 'default'}}
    if family == 'qwenimage21':
        workflow['4'] = {'class_type': 'TextEncodeQwenImage21', 'inputs': {
            'clip': ['2', 0], 'prompt': str(prompt or ''),
            'negative_prompt': str(negative or ''), 'resolution': 1024}}
        positive, negative_link = ['4', 0], ['4', 1]
    else:
        for node_id, text in (('4', prompt), ('5', negative)):
            workflow[node_id] = {'class_type': 'CLIPTextEncode', 'inputs': {
                'clip': ['2', 0], 'text': str(text or '')}}
        positive, negative_link = ['4', 0], ['5', 0]
    sampler_cfg = cfg
    if family == 'moss':
        workflow['sampling'] = {'class_type': 'ModelSamplingSD3', 'inputs': {
            'model': [previous, 0], 'shift': 5.0}}
        workflow['slg'] = {'class_type': 'SkipLayerGuidanceDiTSimple', 'inputs': {
            'model': ['sampling', 0], 'double_layers': '', 'single_layers': '22',
            'start_percent': 0.0, 'end_percent': 1.0}}
        previous = 'slg'
    if family == 'flux':
        sampler_cfg = 1.0
        workflow['guidance'] = {'class_type': 'FluxGuidance', 'inputs': {
            'conditioning': positive, 'guidance': cfg}}
        positive = ['guidance', 0]
        # Schnell has its own native sampling shift and ignores dev guidance.
        if 'schnell' not in resolved['diffusion_model'].lower():
            workflow['sampling'] = {'class_type': 'ModelSamplingFlux', 'inputs': {
                'model': [previous, 0], 'max_shift': 1.15, 'base_shift': 0.5,
                'width': width, 'height': height}}
            previous = 'sampling'
        elif steps_defaulted:
            steps = 4
    workflow['7'] = {'class_type': 'KSampler', 'inputs': {
        'model': [previous, 0], 'positive': positive, 'negative': negative_link,
        'latent_image': ['6', 0], 'seed': parsed_seed, 'steps': steps,
        'cfg': sampler_cfg, 'sampler_name': sampler or defaults['sampler'],
        'scheduler': scheduler or defaults['scheduler'], 'denoise': 1.0}}
    workflow['8'] = {'class_type': 'VAEDecode', 'inputs': {
        'samples': ['7', 0], 'vae': ['3', 0]}}
    workflow['9'] = {'class_type': 'SaveImage', 'inputs': {
        'images': ['8', 0], 'filename_prefix': filename_prefix}}
    if available_classes is not None:
        missing = sorted({n['class_type'] for n in workflow.values()} - set(available_classes))
        if missing:
            raise ValueError('Update ComfyUI: missing generation nodes: ' + ', '.join(missing))
    return workflow
