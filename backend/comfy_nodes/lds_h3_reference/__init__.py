"""Task-specific reference LoRA loading, installed from LDS Setup."""

from .loader import LDSMiniMaxH3ReferenceLoRA

NODE_CLASS_MAPPINGS = {'LDSMiniMaxH3ReferenceLoRA': LDSMiniMaxH3ReferenceLoRA}
NODE_DISPLAY_NAME_MAPPINGS = {
    'LDSMiniMaxH3ReferenceLoRA': 'MiniMax H3 Reference LoRA (LoRA Dataset Studio)',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
