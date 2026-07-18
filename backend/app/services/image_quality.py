"""Pure-PIL technical quality metrics for the 🗃️ image bank (CPU, no GPU, no
numpy/cv2 — the Flask venv is deliberately light, and the quality pass must
work out of the box on every install, extras or not).

The three metrics follow the classic recipes (the same ones Crucible-style
curation tools use), reformulated for PIL:

* **Sharpness** — variance of the 4-neighbour Laplacian. PIL's ``Kernel``
  filter clamps its uint8 output, so the signed Laplacian is recovered from
  TWO clamped convolutions (kernel and its negation, ``scale=4`` so the
  extreme |L|=4·255 maps exactly to 255 — no value ever clamps) and the
  variance is computed from their histograms: at every pixel exactly one of
  the two passes is non-zero, so E[L²] = 16·(E[pos²]+E[neg²]) and
  E[L] = 4·(E[pos]−E[neg]). Low variance = blurry (the ~100 rule of thumb).
* **Noise** — RMS of the residual against a Gaussian blur (σ≈1.1, the OpenCV
  5×5 default): high-frequency energy the blur removed. High = noisy/grainy.
  (Fine texture counts too — that's inherent to the method; it's a FLAG
  threshold, not a verdict.)
* **Uniformity** — plain grayscale standard deviation. Near zero = a flat or
  almost-empty frame (solid color, black frame, blank screenshot).

All metrics run on a grayscale working copy downscaled to a fixed
``ANALYSIS_MAX_SIDE`` long side, so scores are comparable across resolutions
and the per-image cost is bounded (a phone photo and a 4K export cost the
same). Raw scores are returned — thresholds live in config ('bank' section)
and are applied at read time.
"""
from PIL import Image, ImageChops, ImageFilter, ImageStat

ANALYSIS_MAX_SIDE = 1024

_LAPLACIAN = (0, 1, 0,
              1, -4, 1,
              0, 1, 0)
_LAPLACIAN_NEG = tuple(-c for c in _LAPLACIAN)
# |conv| ≤ 4·255 for this kernel → dividing by 4 fits uint8 exactly: the two
# half-convolutions never clamp and the variance below is exact (mod rounding).
_LAP_SCALE = 4

# σ of the reference Gaussian for the noise residual — matches OpenCV's implied
# sigma for its 5×5 default kernel (0.3·((5−1)·0.5−1)+0.8).
_NOISE_SIGMA = 1.1


def _moments(hist):
    """(mean, mean-of-squares) of a 256-bin PIL histogram."""
    n = sum(hist)
    if not n:
        return 0.0, 0.0
    m1 = sum(v * c for v, c in enumerate(hist)) / n
    m2 = sum(v * v * c for v, c in enumerate(hist)) / n
    return m1, m2


def analysis_copy(im: Image.Image) -> Image.Image:
    """Grayscale working copy, long side capped to ANALYSIS_MAX_SIDE.
    LANCZOS: a box/bilinear downscale would smear detail and systematically
    deflate the sharpness of large images."""
    g = im.convert('L')
    w, h = g.size
    m = max(w, h)
    if m > ANALYSIS_MAX_SIDE:
        r = ANALYSIS_MAX_SIDE / m
        g = g.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    return g


def quality_metrics(im: Image.Image) -> dict:
    """Raw technical scores for one PIL image:
    {'blur_score', 'noise_score', 'uniformity_score'} (see module docstring)."""
    g = analysis_copy(im)
    # Sharpness: Laplacian variance via the two clamp-free half-convolutions.
    # PIL leaves a 1-px border UNFILTERED (raw grayscale values, huge once
    # squared) — crop to the interior before reading the histograms.
    w, h = g.size
    interior = (1, 1, max(2, w - 1), max(2, h - 1))
    pos = g.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN, scale=_LAP_SCALE))
    neg = g.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN_NEG, scale=_LAP_SCALE))
    p1, p2 = _moments(pos.crop(interior).histogram())
    n1, n2 = _moments(neg.crop(interior).histogram())
    mean = _LAP_SCALE * (p1 - n1)
    blur = max(0.0, _LAP_SCALE * _LAP_SCALE * (p2 + n2) - mean * mean)
    # Noise: RMS of the Gaussian residual.
    blurred = g.filter(ImageFilter.GaussianBlur(radius=_NOISE_SIGMA))
    _d1, d2 = _moments(ImageChops.difference(g, blurred).histogram())
    noise = d2 ** 0.5
    # Uniformity: grayscale std.
    uniformity = ImageStat.Stat(g).stddev[0]
    return {'blur_score': round(blur, 3),
            'noise_score': round(noise, 3),
            'uniformity_score': round(uniformity, 3)}
