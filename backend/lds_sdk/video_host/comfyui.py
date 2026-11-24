"""Named comfyui operations shared with the host; no module handle escapes."""

def fetch_node_info(*args, **kwargs):
    from app.utils.comfyui import fetch_node_info
    return fetch_node_info(*args, **kwargs)


def fetch_object_info_classes(*args, **kwargs):
    from app.utils.comfyui import fetch_object_info_classes
    return fetch_object_info_classes(*args, **kwargs)


def fetch_output_image_bytes(*args, **kwargs):
    from app.utils.comfyui import fetch_output_image_bytes
    return fetch_output_image_bytes(*args, **kwargs)


def sageattention_importable(*args, **kwargs):
    from app.utils.comfyui import sageattention_importable
    return sageattention_importable(*args, **kwargs)


def fit_node_inputs(*args, **kwargs):
    from app.utils.comfyui import fit_node_inputs
    return fit_node_inputs(*args, **kwargs)


__all__ = ['fetch_node_info', 'fetch_object_info_classes', 'fetch_output_image_bytes',
           'fit_node_inputs', 'sageattention_importable']


# Compatibility with the original public SDK exports.
_LEGACY_EXPORTS = {'fetch_output_image_bytes': ('app.utils.comfyui', 'fetch_output_image_bytes')}

def __getattr__(name):
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module
    module, symbol = _LEGACY_EXPORTS[name]
    return getattr(import_module(module), symbol)

__all__ = list(dict.fromkeys([*__all__, *_LEGACY_EXPORTS]))
