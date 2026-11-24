"""ComfyUI entry point for the MiniMax H3 block-attention switch shipped by
LoRA Dataset Studio.

ComfyUI imports this folder as a package and reads the two mappings below; the
work is in `attention.py`, which is jacokon's `h3_block_attention` node
(FastH3 Live, Apache-2.0) carried here so the app can put it on a machine it
does not own — a rented pod, or the user's own ComfyUI at one click in Setup.

The class key is namespaced (`LDSMiniMaxH3BlockAttentionSplit`), as every node
in this folder is: `custom_nodes/` is a flat namespace with no ownership, and a
user who also installs the original pack must end up with two nodes, never with
one shadowing the other. The DISPLAY name says where the file came from.

Nothing is imported eagerly beyond the node module itself: an exception raised
here would take out ComfyUI's whole custom-node load pass.
"""

from .attention import LDSMiniMaxH3BlockAttentionSplit

NODE_CLASS_MAPPINGS = {
    'LDSMiniMaxH3BlockAttentionSplit': LDSMiniMaxH3BlockAttentionSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'LDSMiniMaxH3BlockAttentionSplit': 'MiniMax H3 Block Attention Split (LoRA Dataset Studio, after jacokon)',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
