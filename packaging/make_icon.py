"""Generate packaging/icon.ico for the launcher exe + window.

A rounded indigo->violet tile with a little DNA double-helix (dots on two offset sine
strands) — matches the app's 🧬 theme. Committed alongside the .ico it produces so the
build is reproducible; re-run `python packaging/make_icon.py` to regenerate.
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
OUT = Path(__file__).resolve().parent / "icon.ico"


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render(size: int = SIZE) -> Image.Image:
    top, bottom = (99, 102, 241), (139, 92, 246)   # indigo-500 -> violet-500
    base = Image.new("RGB", (size, size))
    px = base.load()
    for y in range(size):
        row = _lerp(top, bottom, y / (size - 1))
        for x in range(size):
            px[x, y] = row

    # Rounded-rect alpha mask so the tile has soft corners on any background.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1],
                                           radius=int(size * 0.22), fill=255)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(base, (0, 0), mask)

    # Two DNA strands: dots along offset sine waves, radius growing toward the centre.
    d = ImageDraw.Draw(img)
    mid, amp, turns = size / 2, size * 0.16, 2.2
    for phase, color in ((0.0, (255, 255, 255, 235)), (math.pi, (224, 231, 255, 235))):
        for i in range(13):
            t = i / 12
            y = size * 0.16 + t * size * 0.68
            x = mid + amp * math.sin(t * turns * math.pi * 2 + phase)
            r = size * (0.018 + 0.012 * math.sin(t * math.pi))   # fat in the middle
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img


def main() -> None:
    icon = render()
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon.save(OUT, format="ICO", sizes=sizes)
    print(f"wrote {OUT} ({', '.join(f'{w}x{h}' for w, h in sizes)})")


if __name__ == "__main__":
    main()
