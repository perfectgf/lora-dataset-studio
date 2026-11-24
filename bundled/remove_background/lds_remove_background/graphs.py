"""ComfyUI graphs, native nodes only: find an object, or cut the edge of a picture.

Each graph reads one staged picture and saves one grayscale picture of the same
framing. Deciding what to do with the answers belongs to ``masks``.
"""
from __future__ import annotations

import json
import math

FIND_NODES = frozenset({'CheckpointLoaderSimple', 'CLIPTextEncode', 'LoadImage', 'SAM3_Detect',
                        'MaskToImage', 'SaveImage'})
MATTE_NODES = frozenset({'LoadBackgroundRemovalModel', 'RemoveBackground', 'LoadImage',
                         'MaskToImage', 'SaveImage'})
FINE_NODES = frozenset({'UNETLoader', 'CLIPLoader', 'VAELoader', 'TextEncodeFibo2', 'KSampler',
                        'VAEDecode', 'LoadImage', 'SaveImage'})

# The longest side a matting model or SAM is sent. They work at about 1024 px;
# a little more keeps the answer from being enlarged further on the way back.
MATTE_SIDE = 2048
FIND_SIDE = 1536
MAX_POINTS = 32
MAX_PHRASE = 120
# The node keeps only the best match unless told otherwise: "two dogs" means both.
MAX_MATCHES = 16

# The one instruction RMBG 3.0 was distilled with, the same for every picture.
_FINE_INSTRUCTION = json.dumps({'edit_instruction': (
    'Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the background '
    'to black. Produce soft, anti-aliased grayscale gradients at the edges of the subject to represent '
    'fine details and transparency.')})
_FINE_PIXELS = 1_048_576
_FINE_STEP = 16
_FINE_MAX_SIDE = 2048


def nodes_for(keep, kind):
    """The node classes one cut-out needs ComfyUI to have."""
    nodes = set(FINE_NODES if kind == 'fine' else MATTE_NODES)
    return nodes | FIND_NODES if keep in ('describe', 'points') else nodes


def fitted(width, height, longest):
    """The size that fits ``longest``, never enlarged."""
    scale = min(1.0, longest / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def fine_size(width, height):
    """RMBG 3.0's own size for this shape: about a megapixel, sides multiples of 16.

    The node crops a picture whose shape differs from the size it is asked for,
    and the matte would no longer line up, so the picture is resized to exactly this.
    """
    scale = math.sqrt(_FINE_PIXELS / (max(1, width) * max(1, height)))

    def side(value):
        stepped = round(value * scale / _FINE_STEP) * _FINE_STEP
        return int(min(_FINE_MAX_SIDE, max(_FINE_STEP, stepped)))
    return side(width), side(height)


def phrases(text):
    """What SAM is asked to find: short noun phrases, separated by commas.

    SAM reads "woman in red", not an order, so leading verbs and articles are
    dropped and whatever follows "remove" is ignored.
    """
    words = ' '.join(str(text or '').split())
    lowered = words.lower()
    cut = min((lowered.find(mark) for mark in (', remove ', ' and remove ', ' remove ', ' without the background')
               if lowered.find(mark) > 0), default=len(words))
    result = []
    for part in words[:cut].replace(';', ',').split(','):
        kept = [word for word in part.replace('(', ' ').replace(')', ' ').replace(':', ' ').split() if word]
        while kept and kept[0].lower() in {'keep', 'only', 'just', 'select', 'the', 'a', 'an', 'and'}:
            kept.pop(0)
        phrase = ' '.join(kept).strip(' .!?"\'')
        if phrase and phrase.lower() not in (item.lower() for item in result):
            result.append(phrase[:MAX_PHRASE])
    return result[:8]


def find(image, checkpoint, prefix, *, words=None, points=None):
    """SAM 3.1 finds what the words name, or what the clicks point at.

    The node adds together what words and clicks each find, so a click could not
    take anything away from the words' answer: one way or the other, never both.
    """
    graph = {
        '1': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': checkpoint}},
        '2': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '4': {'class_type': 'SAM3_Detect', 'inputs': {
            'model': ['1', 0], 'image': ['2', 0], 'threshold': 0.5,
            'refine_iterations': 2, 'individual_masks': False}},
        '5': {'class_type': 'MaskToImage', 'inputs': {'mask': ['4', 0]}},
        '6': {'class_type': 'SaveImage', 'inputs': {'images': ['5', 0], 'filename_prefix': prefix}},
    }
    detect = graph['4']['inputs']
    if words:
        graph['3'] = {'class_type': 'CLIPTextEncode', 'inputs': {
            'clip': ['1', 1], 'text': ', '.join(f'{word}: {MAX_MATCHES}' for word in words)}}
        detect['conditioning'] = ['3', 0]
    else:
        keep = [{'x': p['x'], 'y': p['y']} for p in points if p['keep']]
        drop = [{'x': p['x'], 'y': p['y']} for p in points if not p['keep']]
        detect['positive_coords'] = json.dumps(keep)
        if drop:
            detect['negative_coords'] = json.dumps(drop)
    return graph


def matte(image, model, prefix):
    """BiRefNet or Lucida cut the edge of the main subject of the picture."""
    return {
        '1': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '2': {'class_type': 'LoadBackgroundRemovalModel', 'inputs': {'bg_removal_name': model}},
        '3': {'class_type': 'RemoveBackground', 'inputs': {'bg_removal_model': ['2', 0], 'image': ['1', 0]}},
        '4': {'class_type': 'MaskToImage', 'inputs': {'mask': ['3', 0]}},
        '5': {'class_type': 'SaveImage', 'inputs': {'images': ['4', 0], 'filename_prefix': prefix}},
    }


def fine_matte(image, files, size, prefix):
    """RMBG 3.0 paints the matte: four steps, no guidance, as ComfyUI's own template runs it."""
    return {
        '1': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '2': {'class_type': 'UNETLoader', 'inputs': {'unet_name': files['rmbg'], 'weight_dtype': 'default'}},
        '3': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': files['rmbg_text_encoder'], 'type': 'fibo2', 'device': 'default'}},
        '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': files['rmbg_vae']}},
        '5': {'class_type': 'TextEncodeFibo2', 'inputs': {
            'clip': ['3', 0], 'prompt': _FINE_INSTRUCTION, 'negative_prompt': '',
            'images.image_1': ['1', 0], 'width': size[0], 'height': size[1], 'batch_size': 1, 'vae': ['4', 0]}},
        '6': {'class_type': 'KSampler', 'inputs': {
            'model': ['2', 0], 'positive': ['5', 0], 'negative': ['5', 1], 'latent_image': ['5', 2],
            'seed': 42, 'steps': 4, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '7': {'class_type': 'VAEDecode', 'inputs': {'samples': ['6', 0], 'vae': ['4', 0]}},
        '8': {'class_type': 'SaveImage', 'inputs': {'images': ['7', 0], 'filename_prefix': prefix}},
    }


def copy(image, prefix):
    """Hand a finished picture to the host's queue, which files it with its owner."""
    return {
        '1': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '2': {'class_type': 'SaveImage', 'inputs': {'images': ['1', 0], 'filename_prefix': prefix}},
    }
