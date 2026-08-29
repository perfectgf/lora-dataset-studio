"""ComfyUI entry point for the Krea 2 preset sampler shipped by LoRA Dataset Studio.

ComfyUI imports this folder as a package and reads the two mappings below; the
work is in `sampler.py`.

The class key is namespaced (`LDSKrea2PresetSampler`) because `custom_nodes/` is
a flat namespace with no ownership: two packs registering the same key shadow one
another silently, and the app's own graphs name this class explicitly. The
DISPLAY name is the one a human reads in ComfyUI's node menu, and says which app
put it there — anyone auditing their `custom_nodes/` folder deserves to know
where a file came from without opening it.

Nothing is imported eagerly beyond the sampler module itself: an exception raised
here would take out ComfyUI's whole custom-node load pass, and the app's own
preflight is the thing meant to notice this node is missing.
"""

from .sampler import LDSKrea2PresetSampler

NODE_CLASS_MAPPINGS = {
    'LDSKrea2PresetSampler': LDSKrea2PresetSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'LDSKrea2PresetSampler': 'Krea 2 Preset Sampler (LoRA Dataset Studio)',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
