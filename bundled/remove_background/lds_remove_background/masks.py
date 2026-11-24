"""Pixels: turn what the models answer into one alpha mask, and a mask into a cut-out.

Pillow only: the host guarantees it to plugin code, and nothing else that
touches arrays. Growing and shrinking a shape is a box blur and a threshold,
whose cost does not depend on the distance.
"""
from __future__ import annotations

from PIL import Image, ImageChops, ImageFilter

# Below this share of SAM's object, the matting model was looking at something else.
_MATTE_AGREES = 0.25
# Where a second, closer look is worth it: the subject leaves at least this much around it.
_CLOSE_UP = 0.85
_WORK_SIDE = 1536

BACKGROUNDS = {'transparent': None, 'white': (255, 255, 255), 'grey': (128, 128, 128), 'black': (0, 0, 0)}


def gray(image):
    """A model's answer as one 8-bit channel."""
    return image if image.mode == 'L' else image.convert('L')


def solid(mask, level=128):
    return mask.point([255 if value >= level else 0 for value in range(256)])


def coverage(mask):
    """How much of the picture the mask keeps, from 0 to 1."""
    counts = mask.histogram()
    total = mask.width * mask.height
    return sum(value * count for value, count in enumerate(counts)) / (255 * total) if total else 0.0


def bounds(mask, level=64):
    return solid(mask, level).getbbox()


def spread(mask, distance):
    """A hard shape grown by about ``distance`` pixels."""
    return solid(mask.filter(ImageFilter.BoxBlur(distance)), 2) if distance >= 1 else solid(mask)


def shrink(mask, distance):
    """A hard shape shrunk by about ``distance`` pixels."""
    return solid(mask.filter(ImageFilter.BoxBlur(distance)), 254) if distance >= 1 else solid(mask)


def thick(mask, distance):
    """Only the parts of a hard shape wider than about twice ``distance``."""
    return spread(shrink(mask, distance), distance)


def without_specks(mask):
    """A hard shape without the stray dots a click can leave around it; its own outline is untouched."""
    grain = max(2, round(max(mask.size) * 0.004))
    main = thick(mask, grain)
    return ImageChops.multiply(mask, spread(main, grain * 2)) if main.getbbox() else mask


def padded(box, size, share=0.12, least=24):
    """``box`` with some of its surroundings, inside a picture of ``size``."""
    left, top, right, bottom = box
    margin = max(least, round(max(right - left, bottom - top) * share))
    return (max(0, left - margin), max(0, top - margin),
            min(size[0], right + margin), min(size[1], bottom + margin))


def close_up(mask):
    """The box to look at again, or None when the subject already fills the picture."""
    box = bounds(mask)
    if box is None:
        return None
    box = padded(box, mask.size)
    if max((box[2] - box[0]) / mask.width, (box[3] - box[1]) / mask.height) > _CLOSE_UP:
        return None
    return box


def level(mask, low=8, high=246):
    """Pull the nearly empty and nearly full values to exactly that."""
    span = high - low
    return mask.point([0 if value <= low else 255 if value >= high
                       else round((value - low) * 255 / span) for value in range(256)])


def on_canvas(mask, box, size):
    """``mask`` answers for ``box`` of a picture of ``size``; everything else is left out."""
    width, height = box[2] - box[0], box[3] - box[1]
    canvas = Image.new('L', size, 0)
    piece = mask if mask.size == (width, height) else mask.resize((width, height), Image.BICUBIC)
    canvas.paste(piece, box[:2])
    return canvas


def within(selection, matte):
    """The matte, inside a rough selection. Hair may pass a little beyond its outline."""
    box = bounds(selection)
    if box is None:
        return Image.new('L', matte.size, 0)
    reach = max(6, round(max(box[2] - box[0], box[3] - box[1]) * 0.03))
    gate = spread(selection, reach).filter(ImageFilter.GaussianBlur(reach / 3))
    return ImageChops.multiply(matte, gate)


def follow_target(target, matte):
    """``(alpha, matted)``: SAM decides the shape, the matte its edge.

    Near the outline the matte's softer edge is used wherever both run together.
    Where they part by more than strands of hair (the matte also took what the
    object stands on, or lost a dark sleeve against a dark wall), SAM's own edge
    is kept. When the matte answers about something else altogether, SAM's shape
    is the whole answer and ``matted`` is False.
    """
    size = matte.size
    work = size if max(size) <= _WORK_SIDE else tuple(
        max(1, round(side * _WORK_SIDE / max(size))) for side in size)
    hard = solid(target.resize(work, Image.BILINEAR))
    soft = matte if work == size else matte.resize(work, Image.BILINEAR)
    box = hard.getbbox()
    if box is None:
        return Image.new('L', size, 0), False
    own = hard.filter(ImageFilter.GaussianBlur(1.2))
    agreed = coverage(ImageChops.multiply(hard, solid(soft)))
    if agreed < _MATTE_AGREES * coverage(hard):
        return own.resize(size, Image.BICUBIC), False

    reach = max(6, round(max(box[2] - box[0], box[3] - box[1]) * 0.035))
    slack = max(3, reach // 4)
    beyond = ImageChops.subtract(solid(soft), spread(hard, slack))
    lost = ImageChops.subtract(hard, spread(solid(soft, 40), slack))
    dispute = spread(ImageChops.lighter(thick(beyond, slack), thick(lost, slack)), reach)
    dispute = dispute.filter(ImageFilter.GaussianBlur(reach / 3))
    gate = spread(hard, reach).filter(ImageFilter.GaussianBlur(reach / 3))
    if work != size:
        own, dispute, gate = (item.resize(size, Image.BICUBIC) for item in (own, dispute, gate))
    return Image.composite(own, ImageChops.multiply(matte, gate), dispute), True


def painted(mask, strokes):
    """``mask`` with the user's brush over it.

    ``strokes`` is gray with alpha: white keeps, black removes, and where nothing
    was painted the mask is untouched, whatever the size it was painted at.
    """
    if strokes.size != mask.size:
        strokes = strokes.resize(mask.size, Image.BICUBIC)
    value, alpha = strokes.split()
    return Image.composite(value, mask, alpha)


def clean_edges(image, alpha):
    """Take the old background's colour out of the soft edge.

    A pixel that is half hair, half blue sky keeps its transparency but takes
    the colour of the solid subject next to it, spread outwards from the solid
    part, nearest first. Where the picture is see-through over a wide area, as
    glass is, its own colour is all there is to show and stays as it was.
    """
    picture = image.convert('RGB')
    core = alpha.point([255 if value >= 250 else 0 for value in range(256)])
    if core.getbbox() is None:
        return picture
    source = picture.convert('RGBA')
    source.putalpha(core)
    source = source.convert('RGBa')  # premultiplied: the blur below is a weighted average of solid pixels
    unit = max(1.0, max(image.size) / 1600)
    colour = picture
    for radius, trust in ((36, 10), (14, 8), (5, 6), (2, 6)):
        near = source.filter(ImageFilter.GaussianBlur(radius * unit)).convert('RGBA')
        # Too few solid pixels in reach make a colour out of rounding errors.
        reliable = near.getchannel('A').point([min(255, max(0, value - 8) * trust) for value in range(256)])
        colour = Image.composite(near.convert('RGB'), colour, reliable)
    weight = alpha.point([0 if value >= 250 else min(255, round((255 - value) * 1.25)) for value in range(256)])
    wide = max(8, round(max(image.size) * 0.01))
    glass = thick(alpha.point([255 if 12 <= value <= 235 else 0 for value in range(256)]), wide)
    if glass.getbbox():
        weight = ImageChops.subtract(weight, spread(glass, wide).filter(ImageFilter.GaussianBlur(wide / 2)))
    return Image.composite(colour, picture, weight)


def cut_out(image, alpha, *, background='transparent', clean=True):
    """The finished picture. Nothing of the removed background is left under transparent pixels."""
    pixels = clean_edges(image, alpha) if clean else image.convert('RGB')
    fill = BACKGROUNDS[background]
    if fill is not None:
        canvas = Image.new('RGB', image.size, fill)
        canvas.paste(pixels, mask=alpha)
        return canvas
    result = pixels.convert('RGBA')
    result.putalpha(alpha)
    return Image.composite(result, Image.new('RGBA', image.size, (0, 0, 0, 0)),
                           alpha.point([255 if value else 0 for value in range(256)]))


def as_alpha(mask):
    """The mask as a white picture whose transparency is the mask, for the browser's canvas."""
    picture = Image.new('LA', mask.size, (255, 0))
    picture.putalpha(mask)
    return picture
