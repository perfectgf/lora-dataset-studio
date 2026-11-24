"""Apply Ref4/Ref8 beside quantized H3 weights through native ComfyUI bypass.

The published ComfyUI LoRAs have fused projections, no AdaLN adapters, and
alpha/rank scaling. Both their prefixed keys and bare H3 keys are accepted.
No dependency on a locally patched third-party loader or rewritten weight file.
"""

import logging
import math


def reference_key_map(lora):
    """Reject incomplete adapters rather than silently running a dense model."""
    suffix = '.lora_A.weight'
    modules = {key[:-len(suffix)] for key in lora if key.endswith(suffix)}
    if not modules:
        raise ValueError('This file has no supported H3 reference LoRA weights.')
    result = {}
    targets = set()
    for module in sorted(modules):
        bare = module.removeprefix('diffusion_model.')
        if 'adaln' in bare or not bare.startswith(('blocks.', 'token_refiner.blocks.')):
            raise ValueError('Use a dedicated Ref4 or Ref8 LoRA without AdaLN adapters.')
        up_key = module + '.lora_B.weight'
        if up_key not in lora:
            raise ValueError('The reference LoRA is missing a paired projection.')
        down, up = lora[module + suffix], lora[up_key]
        if len(down.shape) != 2 or len(up.shape) != 2 or up.shape[1] != down.shape[0]:
            raise ValueError('The reference LoRA has incompatible projection dimensions.')
        target = 'diffusion_model.' + bare + '.weight'
        if target in targets:
            raise ValueError('The reference LoRA contains duplicate prefixed and bare weights.')
        targets.add(target)
        result[module] = target
    paired_up = {module + '.lora_B.weight' for module in modules}
    if {key for key in lora if key.endswith('.lora_B.weight')} != paired_up:
        raise ValueError('The reference LoRA has an unpaired output projection.')
    return result


def _adapter_type():
    import comfy.weight_adapter
    import torch.nn.functional as functional

    class ReferenceLoRA(comfy.weight_adapter.LoRAAdapter):
        def bypass_forward(self, original, x, *args, **kwargs):
            if getattr(self, 'is_conv', False):
                return super().bypass_forward(original, x, *args, **kwargs)
            base = original(x, *args, **kwargs)
            up, down, alpha = self.weights[:3]
            scale = (float(alpha) / down.shape[0] if alpha is not None else 1.0)
            scale *= getattr(self, 'multiplier', 1.0)
            delta = functional.linear(functional.linear(x, down.to(dtype=x.dtype)),
                                      up.to(dtype=x.dtype))
            return base.add_(delta, alpha=scale)

    return ReferenceLoRA


def apply_reference_lora(model, lora, strength):
    import comfy.lora
    import comfy.patcher_extension
    import comfy.utils
    import comfy.weight_adapter

    strength = float(strength)
    if not math.isfinite(strength) or not -10 <= strength <= 10:
        raise ValueError('Reference LoRA strength must be between -10 and 10.')
    key_map = reference_key_map(lora)
    adapters = comfy.lora.load_lora(lora, key_map, log_missing=False)
    if set(adapters) != set(key_map.values()):
        raise ValueError('The reference LoRA could not be mapped completely; update ComfyUI.')
    clone = model.clone()
    manager = comfy.weight_adapter.BypassInjectionManager()
    adapter_cls = _adapter_type()
    for key, adapter in adapters.items():
        if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            raise ValueError('This reference loader requires ordinary LoRA projections.')
        if any(value is not None for value in adapter.weights[3:6]):
            raise ValueError('Reference LoRAs with mid, DoRA, or reshape weights are not supported.')
        try:
            module = comfy.utils.get_attr(clone.model, key[:-len('.weight')])
        except (AttributeError, KeyError) as exc:
            raise ValueError('The selected LoRA does not match this H3 model.') from exc
        up, down = adapter.weights[:2]
        # Quantized weight storage may be packed. Linear dimensions describe the
        # actual operation and stay valid for INT8 and W4A8 alike.
        if (getattr(module, 'in_features', None) != down.shape[1]
                or getattr(module, 'out_features', None) != up.shape[0]):
            raise ValueError('The selected LoRA projection does not match the H3 model.')
        manager.add_adapter(key, adapter_cls(adapter.loaded_keys, adapter.weights),
                            strength=strength)
    injections = manager.create_injections(clone.model)
    if manager.get_hook_count() != len(key_map):
        raise ValueError('Not all reference LoRA projections could be attached.')
    # ComfyUI ejects namespaces in insertion order. Hooks on the same projection
    # must unwind in reverse order or an already-ejected forward is restored.
    namespace = 'lds_h3_reference'
    chain = (*(clone.get_injections(namespace) or []), *injections)

    def inject(patcher):
        for injection in chain:
            injection.inject(patcher)

    def eject(patcher):
        for injection in reversed(chain):
            injection.eject(patcher)

    clone.set_injections(namespace, [comfy.patcher_extension.PatcherInjection(inject, eject)])
    logging.info('LDS H3 reference LoRA: %d projections attached in bypass mode', len(key_map))
    return clone


class LDSMiniMaxH3ReferenceLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required': {
            'model': ('MODEL',),
            'lora_name': (folder_paths.get_filename_list('loras'),),
            'strength': ('FLOAT', {'default': 1.0, 'min': -10.0, 'max': 10.0, 'step': 0.01}),
        }}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply_lora'
    CATEGORY = 'LoRA Dataset Studio/MiniMax H3'

    def apply_lora(self, model, lora_name, strength=1.0):
        import comfy.utils
        import folder_paths
        path = folder_paths.get_full_path('loras', lora_name)
        if not path:
            raise ValueError('The selected reference LoRA is no longer available.')
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        return (apply_reference_lora(model, lora, strength),)
