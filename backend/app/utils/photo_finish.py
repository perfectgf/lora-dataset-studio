"""Finishing passes applied to a rendered image, in the app rather than in the graph.

WHY NOT COMFYUI NODES
---------------------
The reference workflow these are ported from does all three with third-party
packs — `ColorMatch_UTK` (universaltoolkit), `FastUnsharpSharpen` and
`FastFilmGrain` (comfyui-vrgamedevgirl). Shipping that dependency would break the
rule the installer is built on: it downloads MODELS, never node packs, and the
graphs this app builds have to run on a bare ComfyUI. None of these three needs a
model or a GPU — they are arithmetic on pixels — so they belong on this side of
the wire, where they also become deterministic and testable without a render.

THE ORDER IS NOT ARBITRARY
--------------------------
`match_colours` -> `unsharp_mask` -> `film_grain`, and each step is where it is
for a reason that shows up in the picture:

* Colour matching FIRST, because it estimates a transform from image statistics.
  Run it after sharpening and it measures the halos sharpening just added, not
  the render; run it after grain and it measures the noise.
* Grain LAST, because sharpening amplifies high frequencies — which is exactly
  what grain is. Sharpen after graining and the grain turns into speckle, a
  texture that reads as a compression artefact rather than as film.

WHAT THE GRAIN IS ACTUALLY FOR
------------------------------
Diffusion output is smooth in a way photographs are not: the model resolves
low-frequency structure confidently and leaves the finest octave nearly empty.
A very small amount of noise refills that octave, and the eye reads the result as
a photograph rather than as a render. The useful amounts are tiny — 0.01 is a
grain a viewer never consciously sees — which is why the default is one hundredth
and not one tenth.

Everything here takes and returns float32 in [0, 1], shaped (H, W, 3). The array
functions are pure: no I/O, no config, no globals — `apply_to_file` is the only
thing that touches disk.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile

import numpy as np

logger = logging.getLogger(__name__)

# Sharpening radius in pixels. 1.0 targets the finest octave — the one diffusion
# leaves empty — instead of building the wide bright halo that reads as "over-
# sharpened" long before the number looks large.
UNSHARP_RADIUS_PX = 1.0
# Ridge added to the covariance eigenvalues before inversion. A flat image (a
# single colour, a fully clipped frame) has a singular covariance, and the
# unregularised transform then contains infinities that silently blank the image.
_COV_EPS = 1e-6


def _as_float(image) -> np.ndarray:
    """(H, W, 3) float32 in [0, 1], from anything array-like."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f'expected an (H, W, 3) image, got {arr.shape}')
    arr = arr[:, :, :3]
    # An 8-bit array arrives in [0, 255]; a float one is already normalised. The
    # max is the only honest discriminator — dtype is not, because a float32
    # array holding 0..255 is a perfectly ordinary thing to be handed.
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric positive-semidefinite 3x3."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def _inv_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    """Inverse matrix square root, ridged so a degenerate image cannot produce
    infinities that would come out as a blank frame."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, _COV_EPS, None)
    return (vectors / np.sqrt(values)) @ vectors.T


def match_colours(target, reference, strength: float = 0.8) -> np.ndarray:
    """Grade `target` onto `reference`'s colour statistics (Monge-Kantorovich linear).

    WHAT IT FIXES. An edit pass that re-renders an image at denoise 1.0 keeps the
    content and loses the grade: skin warms or cools, contrast creeps, and a
    dataset ends up holding two colour worlds depending on which images went
    through the pass. The reference is the image as it was BEFORE that pass, so
    what comes back is the same picture with its own colours.

    WHY MKL AND NOT MEAN/STD PER CHANNEL. The cheap version matches each channel
    independently, which cannot represent a cast that lives in the CORRELATION
    between channels — the usual failure being a warm shift that the per-channel
    fix turns into a magenta one. MKL matches the full 3x3 covariance, so a
    rotation of the colour cloud is expressible (Pitié & Kokaram, 2007).

    `strength` blends the result back over the input: 1.0 is the full transform,
    0.0 returns the target untouched. 0.8 leaves the pass a little of its own
    look, which is usually wanted — the point is to stop the drift, not to
    forbid the edit from changing anything.

    Shapes need not match: only per-channel statistics are read from `reference`,
    never its pixels, so a downscaled or differently-cropped reference is fine.
    """
    tgt = _as_float(target)
    ref = _as_float(reference)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return tgt

    flat_t = tgt.reshape(-1, 3).astype(np.float64)
    flat_r = ref.reshape(-1, 3).astype(np.float64)
    mu_t, mu_r = flat_t.mean(axis=0), flat_r.mean(axis=0)
    cov_t = np.cov(flat_t, rowvar=False) + np.eye(3) * _COV_EPS
    cov_r = np.cov(flat_r, rowvar=False) + np.eye(3) * _COV_EPS

    root_t = _sqrt_psd(cov_t)
    inv_root_t = _inv_sqrt_psd(cov_t)
    transform = inv_root_t @ _sqrt_psd(root_t @ cov_r @ root_t) @ inv_root_t

    graded = (flat_t - mu_t) @ transform.T + mu_r
    graded = graded.reshape(tgt.shape).astype(np.float32)
    out = tgt * (1.0 - strength) + graded * strength
    return np.clip(out, 0.0, 1.0)


def unsharp_mask(image, strength: float = 0.55,
                 radius_px: float = UNSHARP_RADIUS_PX) -> np.ndarray:
    """Local-contrast sharpening: `image + strength * (image - blur(image))`.

    `strength` 0 returns the input. There is no upper clamp beyond the caller's
    own: past roughly 1.0 the halo stops reading as detail and starts reading as
    an outline, but that is a judgement, not a rule, and a caller asking for it
    should get it.

    Blurred with PIL rather than scipy — the dependency is already there for
    every other image path in this app, and a Gaussian is a Gaussian.
    """
    arr = _as_float(image)
    strength = float(strength or 0.0)
    if strength <= 0.0 or radius_px <= 0:
        return arr
    from PIL import Image, ImageFilter
    as_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    blurred = np.asarray(
        Image.fromarray(as_u8, mode='RGB').filter(
            ImageFilter.GaussianBlur(radius=float(radius_px))),
        dtype=np.float32) / 255.0
    return np.clip(arr + strength * (arr - blurred), 0.0, 1.0)


def film_grain(image, intensity: float = 0.01, saturation_mix: float = 0.2,
               seed: int | None = None) -> np.ndarray:
    """Add fine noise: `saturation_mix` of it coloured, the rest luminance-only.

    Real film grain is mostly achromatic — the dye clouds of the three layers do
    not line up, but not by much — so pure per-channel noise looks like sensor
    noise instead. `saturation_mix` is that dial: 0 gives identical noise on the
    three channels (luminance grain), 1 gives independent noise per channel.

    `intensity` is the standard deviation in [0, 1] units, so 0.01 is ±2.5 levels
    of an 8-bit image: visible as texture, invisible as noise. `seed` makes a run
    reproducible, which is what lets a test assert on it at all.
    """
    arr = _as_float(image)
    intensity = float(intensity or 0.0)
    if intensity <= 0.0:
        return arr
    saturation_mix = float(np.clip(saturation_mix, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    h, w = arr.shape[:2]
    luma = rng.standard_normal((h, w, 1)).astype(np.float32)
    chroma = rng.standard_normal((h, w, 3)).astype(np.float32)
    # Broadcasting the luma plane across the three channels is what makes the
    # achromatic half achromatic. The two sources are INDEPENDENT normals, so a
    # plain mix has variance (1-m)² + m² — 0.5 at m=0.5, i.e. the "how coloured"
    # dial would also turn the grain DOWN by up to 29%. Dividing by hypot puts the
    # deviation back at exactly `intensity` whatever the mix (measured: 0.0500
    # at m=0, 0.2, 0.5 and 1 after this line; 0.0412 at the shipped 0.2 before).
    norm = math.hypot(1.0 - saturation_mix, saturation_mix)
    noise = (luma * (1.0 - saturation_mix) + chroma * saturation_mix) / norm
    return np.clip(arr + noise * intensity, 0.0, 1.0)


def finish(image, *, reference=None, colour_strength: float = 0.0,
           sharpen: float = 0.0, grain: float = 0.0,
           grain_saturation: float = 0.2, seed: int | None = None) -> np.ndarray:
    """The three passes in the one order that makes sense (see the module docstring).

    Every stage is skipped at 0, and all three at 0 returns the input converted
    but otherwise untouched — so "the feature is off" costs a dtype conversion,
    not a rewrite of the picture. Colour matching additionally needs a
    `reference`; without one it is skipped no matter what strength says, because
    there is nothing to match TO and inventing a reference would be a grade.
    """
    arr = _as_float(image)
    if reference is not None and colour_strength > 0:
        arr = match_colours(arr, reference, colour_strength)
    if sharpen > 0:
        arr = unsharp_mask(arr, sharpen)
    if grain > 0:
        arr = film_grain(arr, grain, grain_saturation, seed=seed)
    return arr


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """[0, 1] float -> 8-bit, rounded rather than truncated (truncation is a
    half-level darkening applied to every pixel of every finished image)."""
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def apply_to_file(path, *, reference_path=None, colour_strength: float = 0.0,
                  sharpen: float = 0.0, grain: float = 0.0,
                  grain_saturation: float = 0.2, seed: int | None = None) -> bool:
    """Run `finish` over the image at `path`, in place. True when it rewrote it.

    Returns False — having touched nothing — when every stage is off, when the
    file cannot be read, or when a requested colour match has no readable
    reference. A finishing pass is a nicety: it may never be the reason a render
    the user waited for is lost, so every failure here degrades to "the image
    stays as it came out of ComfyUI".

    The alpha channel, if any, is dropped: all three passes are defined on RGB,
    and the lanes that call this write PNGs of opaque renders.
    """
    if not (colour_strength > 0 and reference_path) and not (sharpen > 0) and not (grain > 0):
        return False
    from PIL import Image
    path = str(path)
    try:
        with Image.open(path) as handle:
            source = np.asarray(handle.convert('RGB'), dtype=np.float32) / 255.0
            # ComfyUI's SaveImage embeds the `prompt` and `workflow` graphs as
            # PNG text chunks — the reason a finished render can be dropped back
            # onto ComfyUI to recover the exact graph. A save from a pixel array
            # would strip them; they are carried across on purpose.
            text_chunks = dict(getattr(handle, 'text', None) or {})
            fmt = (handle.format or '').upper()
    except Exception as exc:
        logger.warning('finishing pass: cannot read %s — left as is (%s)', path, exc)
        return False

    reference = None
    if colour_strength > 0 and reference_path:
        try:
            with Image.open(reference_path) as handle:
                reference = np.asarray(handle.convert('RGB'), dtype=np.float32) / 255.0
        except Exception as exc:
            logger.warning('finishing pass: colour reference %s unreadable, colour '
                           'stage skipped (%s)', reference_path, exc)
            reference = None

    # Re-decide AFTER the reference is resolved, not only from the arguments. A
    # colour match whose reference turned out to be unreadable is a stage that
    # will not run, and if it was the only one asked for there is nothing left to
    # do — saving anyway would re-encode the file and report True, which is the
    # difference between "unchanged" and "changed to something identical". PNG is
    # lossless, so nothing would be visibly lost; the lie is in the return value,
    # and a caller that logs "finished" for a pass that did not happen is exactly
    # how this kind of thing stays broken.
    effective_colour = colour_strength if reference is not None else 0.0
    if effective_colour <= 0 and sharpen <= 0 and grain <= 0:
        return False

    out = finish(source, reference=reference, colour_strength=effective_colour,
                 sharpen=sharpen, grain=grain,
                 grain_saturation=grain_saturation, seed=seed)

    # Written BESIDE the render and swapped in atomically. `Image.save(path)`
    # opens the destination with truncation first, so an encoder that fails
    # half-way (disk full, a scanner holding the file) leaves the render the user
    # waited for as a stump — and Pillow only cleans up files it CREATED, which
    # this one is not. Measured on 12 KB PNG → 0-byte file before this. The
    # docstring's promise that a failure "leaves the image as it came out of
    # ComfyUI" is only true with the temp-and-replace.
    ext = os.path.splitext(path)[1].lower()
    save_kwargs = {}
    if fmt == 'PNG' or ext == '.png':
        if text_chunks:
            from PIL import PngImagePlugin
            info = PngImagePlugin.PngInfo()
            for key, value in text_chunks.items():
                info.add_text(key, value)
            save_kwargs['pnginfo'] = info
    elif ext == '.webp':
        # The bank's edited lane is lossless WebP; a default (lossy) save would
        # quietly degrade every finished blob there.
        save_kwargs['lossless'] = True
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or '.', suffix=ext or '.png',
                               prefix='.finish-')
    os.close(fd)
    try:
        Image.fromarray(to_uint8(out), mode='RGB').save(tmp, **save_kwargs)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning('finishing pass: could not write %s — render left untouched (%s)',
                       path, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    return True
