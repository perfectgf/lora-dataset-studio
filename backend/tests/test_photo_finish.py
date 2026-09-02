"""The finishing passes: colour match, unsharp, grain.

These are pure arithmetic, so they are tested as arithmetic — against their
DEFINITIONS and against properties that must hold, never against a "looks
sharper" proxy. High-frequency energy in particular is a metric that has lied
here before: it rises for sharpening, for grain, for ringing and for JPEG
artefacts alike, so it cannot tell which of them happened. Every test that
asserts an effect therefore also carries a control that the treatment happened
at all — a pass that silently returned its input would otherwise satisfy most
"nothing broke" assertions.

The colour-match test is the one worth reading: it does not check that the
output "looks like" the reference, it checks that the output's MEAN and
COVARIANCE have moved onto the reference's, which is the whole claim MKL makes
and the only one it makes.
"""
import numpy as np
import pytest

from app.utils import photo_finish as pf


def _image(seed, mean=(0.5, 0.5, 0.5), scale=0.15, size=64):
    rng = np.random.default_rng(seed)
    arr = rng.normal(np.array(mean, dtype=np.float32), scale, (size, size, 3))
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# --- Colour match ------------------------------------------------------------

def test_matching_onto_a_reference_moves_the_statistics_onto_it():
    """THE claim of the transform. A warm-cast target and a cool reference: after
    a full-strength match, the target's mean and covariance must be the
    reference's, not merely 'closer'."""
    target = _image(1, mean=(0.62, 0.48, 0.36))     # warm
    reference = _image(2, mean=(0.38, 0.46, 0.60))  # cool

    out = pf.match_colours(target, reference, strength=1.0)

    got = out.reshape(-1, 3).astype(np.float64)
    want = reference.reshape(-1, 3).astype(np.float64)
    np.testing.assert_allclose(got.mean(axis=0), want.mean(axis=0), atol=0.01)
    np.testing.assert_allclose(np.cov(got, rowvar=False), np.cov(want, rowvar=False),
                               atol=0.002)
    # Control: the pass actually did something to this target.
    assert not np.allclose(out, target, atol=0.02)


def test_a_cross_channel_cast_is_corrected_which_per_channel_matching_cannot_do():
    """Why MKL rather than mean/std per channel. The cast here lives in the
    CORRELATION between channels — a per-channel fix leaves it in place, because
    every individual channel's mean and variance are already right."""
    rng = np.random.default_rng(7)
    base = rng.normal(0.5, 0.12, (64, 64, 3)).astype(np.float32)
    # A rotation in colour space: channel means and variances survive it almost
    # unchanged, the covariance does not.
    mix = np.array([[0.8, 0.2, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    skewed = np.clip(base.reshape(-1, 3) @ mix.T, 0, 1).reshape(base.shape).astype(np.float32)

    per_channel_gap = np.abs(skewed.reshape(-1, 3).std(axis=0)
                             - base.reshape(-1, 3).std(axis=0)).max()
    assert per_channel_gap < 0.05, 'precondition: per-channel stats barely moved'

    out = pf.match_colours(skewed, base, strength=1.0)

    before = np.abs(np.cov(skewed.reshape(-1, 3), rowvar=False)
                    - np.cov(base.reshape(-1, 3), rowvar=False)).max()
    after = np.abs(np.cov(out.reshape(-1, 3), rowvar=False)
                   - np.cov(base.reshape(-1, 3), rowvar=False)).max()
    assert after < before / 10, f'covariance gap {before} -> {after}'


def test_matching_an_image_onto_itself_is_a_near_identity():
    image = _image(3)
    np.testing.assert_allclose(pf.match_colours(image, image, 1.0), image, atol=0.01)


def test_strength_zero_returns_the_target_untouched():
    target, reference = _image(4), _image(5, mean=(0.2, 0.7, 0.3))
    np.testing.assert_array_equal(pf.match_colours(target, reference, 0.0),
                                  pf.match_colours(target, reference, 0.0))
    np.testing.assert_allclose(pf.match_colours(target, reference, 0.0), target)


def test_strength_interpolates_between_the_two_ends():
    """Half strength lands between the target and the full match, per pixel.

    Deliberately NOT asserted as `half == (target + full) / 2`: the transform can
    overshoot [0, 1] and the result is clipped ONCE, at the end. Blending before
    that clip (what the code does, and what keeps the colours consistent) is not
    the same as averaging two already-clipped images, and the difference is real
    on the handful of pixels that overshoot. The betweenness is the property that
    actually holds — and the one a user would recognise as "half as much"."""
    target, reference = _image(6), _image(7, mean=(0.25, 0.55, 0.7))
    full = pf.match_colours(target, reference, 1.0)
    half = pf.match_colours(target, reference, 0.5)

    low, high = np.minimum(target, full), np.maximum(target, full)
    assert (half >= low - 1e-5).all() and (half <= high + 1e-5).all()
    # ...and on every pixel the clip did not touch, it is exactly half way — the
    # linear claim, asserted exactly, on the ~99% of pixels where it applies.
    unclipped = (full > 1e-4) & (full < 1 - 1e-4)
    assert unclipped.mean() > 0.9, 'precondition: most pixels do not clip'
    np.testing.assert_allclose(half[unclipped], ((target + full) / 2)[unclipped], atol=1e-5)
    assert not np.allclose(half, target, atol=1e-3), 'control: it moved'


def test_a_flat_image_does_not_produce_nan_or_a_blank_frame():
    """A single-colour frame has a singular covariance. Unridged, the inverse
    square root contains infinities and the output comes back blank — the kind
    of failure that only appears on a real image nobody thought to try."""
    flat = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = pf.match_colours(flat, _image(8), strength=1.0)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_the_reference_may_be_a_different_size():
    """Only statistics are read from it, so a downscaled reference is legitimate —
    and is what a caller comparing against a thumbnail would pass."""
    out = pf.match_colours(_image(9, size=64), _image(10, size=17), strength=1.0)
    assert out.shape == (64, 64, 3)


# --- Unsharp -----------------------------------------------------------------

def test_unsharp_matches_its_definition():
    """Asserted against `image + strength * (image - blur(image))` recomputed here,
    rather than against a sharpness metric — those rise for grain and ringing
    just as happily and cannot say which happened."""
    from PIL import Image, ImageFilter
    image = _image(11)
    strength = 0.55

    out = pf.unsharp_mask(image, strength)

    as_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    blurred = np.asarray(Image.fromarray(as_u8, 'RGB').filter(
        ImageFilter.GaussianBlur(radius=pf.UNSHARP_RADIUS_PX)), dtype=np.float32) / 255.0
    expected = np.clip(image + strength * (image - blurred), 0, 1)
    np.testing.assert_allclose(out, expected, atol=1e-6)
    assert not np.allclose(out, image, atol=1e-3), 'control: it changed the image'


@pytest.mark.parametrize('strength', [0, 0.0, None])
def test_unsharp_at_zero_returns_the_input(strength):
    image = _image(12)
    np.testing.assert_allclose(pf.unsharp_mask(image, strength), image)


def test_unsharp_stays_in_range_on_an_already_clipped_image():
    """Sharpening overshoots by construction; the clip is what keeps a bright
    edge from wrapping around into a dark one."""
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[:, 16:] = 1.0
    out = pf.unsharp_mask(image, 3.0)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --- Grain -------------------------------------------------------------------

def test_grain_is_reproducible_for_a_seed_and_different_without_one():
    image = _image(13)
    a = pf.film_grain(image, 0.01, 0.2, seed=42)
    b = pf.film_grain(image, 0.01, 0.2, seed=42)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, pf.film_grain(image, 0.01, 0.2, seed=43))


def test_grain_intensity_is_the_standard_deviation_it_adds():
    """0.01 has to mean 0.01 — the whole reason the useful values look absurdly
    small is that the number is in [0, 1] units, and a caller reading 0.01 as
    'one percent of something else' would dial it up until it was visible noise."""
    image = np.full((256, 256, 3), 0.5, dtype=np.float32)
    out = pf.film_grain(image, 0.01, saturation_mix=1.0, seed=1)
    assert 0.008 < float((out - image).std()) < 0.012


def test_fully_desaturated_grain_is_identical_on_the_three_channels():
    """saturation_mix 0 = luminance grain: the channel DIFFERENCES must survive
    untouched, which is what stops it reading as sensor noise.

    The image is kept well away from 0 and 1 on purpose. The claim is about the
    noise, and the output clip is per channel — on a pixel already at white, the
    channel that clips loses its share of the grain and the three deltas stop
    matching for a reason that has nothing to do with the mix."""
    image = _image(14, mean=(0.5, 0.5, 0.5), scale=0.05)
    out = pf.film_grain(image, 0.05, saturation_mix=0.0, seed=2)
    assert 0.05 < image.min() and image.max() < 0.95, 'precondition: nothing clips'
    delta = out - image
    np.testing.assert_allclose(delta[:, :, 0], delta[:, :, 1], atol=1e-6)
    np.testing.assert_allclose(delta[:, :, 1], delta[:, :, 2], atol=1e-6)
    assert float(np.abs(delta).max()) > 0.0, 'control: grain was actually added'


def test_fully_saturated_grain_is_independent_per_channel():
    out = pf.film_grain(_image(15), 0.05, saturation_mix=1.0, seed=3)
    delta = out - _image(15)
    assert not np.allclose(delta[:, :, 0], delta[:, :, 1], atol=1e-3)


def test_grain_at_zero_returns_the_input():
    image = _image(16)
    np.testing.assert_allclose(pf.film_grain(image, 0.0), image)


# --- The composition ---------------------------------------------------------

def test_finish_runs_the_three_passes_in_the_documented_order():
    """Colour first, grain last. Asserted by recomposing the same chain by hand:
    any other order gives a different array, because none of the three commutes
    with the others."""
    image, reference = _image(17), _image(18, mean=(0.3, 0.5, 0.7))
    out = pf.finish(image, reference=reference, colour_strength=0.8,
                    sharpen=0.55, grain=0.01, seed=5)
    expected = pf.film_grain(
        pf.unsharp_mask(pf.match_colours(image, reference, 0.8), 0.55),
        0.01, 0.2, seed=5)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_finish_with_everything_off_returns_the_image():
    image = _image(19)
    np.testing.assert_allclose(pf.finish(image), image)


def test_colour_matching_is_skipped_without_a_reference():
    """No reference means nothing to match TO. Inventing one would be a grade,
    which is precisely what this pass exists to avoid."""
    image = _image(20)
    np.testing.assert_allclose(pf.finish(image, colour_strength=1.0), image)


# --- The file entry point ----------------------------------------------------

def _write(path, image):
    from PIL import Image
    Image.fromarray(pf.to_uint8(image), 'RGB').save(path)


def test_apply_to_file_rewrites_the_image_and_reports_it(tmp_path):
    path = tmp_path / 'render.png'
    _write(path, _image(21))
    before = path.read_bytes()

    assert pf.apply_to_file(path, sharpen=0.55, grain=0.01, seed=1) is True
    assert path.read_bytes() != before


def test_apply_to_file_with_everything_off_does_not_touch_the_file(tmp_path):
    """OFF has to mean the bytes on disk are the ones ComfyUI wrote — not a
    re-encode that happens to look the same."""
    path = tmp_path / 'render.png'
    _write(path, _image(22))
    before = path.read_bytes()

    assert pf.apply_to_file(path) is False
    assert path.read_bytes() == before


def test_apply_to_file_degrades_rather_than_raising(tmp_path):
    """A finishing pass may never be the reason a render the user waited for is
    lost. An unreadable file, and a colour match whose reference is missing,
    both come back False with the image left alone."""
    missing = tmp_path / 'nope.png'
    assert pf.apply_to_file(missing, sharpen=0.5) is False

    path = tmp_path / 'render.png'
    _write(path, _image(23))
    before = path.read_bytes()
    # Colour match requested, reference unreadable -> that stage is skipped; with
    # no other stage on, nothing is rewritten.
    assert pf.apply_to_file(path, reference_path=missing, colour_strength=0.8) is False
    assert path.read_bytes() == before


def test_apply_to_file_still_finishes_when_only_the_reference_is_missing(tmp_path):
    """The colour stage drops out; sharpening and grain still run. Losing one
    stage must not cancel the other two."""
    path = tmp_path / 'render.png'
    _write(path, _image(24))
    before = path.read_bytes()

    assert pf.apply_to_file(path, reference_path=tmp_path / 'gone.png',
                            colour_strength=0.8, sharpen=0.55, seed=1) is True
    assert path.read_bytes() != before


def test_an_rgba_source_is_handled(tmp_path):
    from PIL import Image
    path = tmp_path / 'render.png'
    Image.fromarray(np.dstack([pf.to_uint8(_image(25)),
                               np.full((64, 64), 255, np.uint8)]), 'RGBA').save(path)
    assert pf.apply_to_file(path, sharpen=0.55) is True
    with Image.open(path) as handle:
        assert handle.mode == 'RGB'


def test_eight_bit_input_is_normalised_not_treated_as_out_of_range():
    """A uint8 array in 0..255 and the same image as floats in 0..1 must give the
    same answer — the alternative is a pass that clips everything to white."""
    as_float = _image(26)
    as_u8 = pf.to_uint8(as_float)
    np.testing.assert_allclose(pf.unsharp_mask(as_u8, 0.55),
                               pf.unsharp_mask(as_float, 0.55), atol=1.5 / 255)


# --- What the adversarial pass found -------------------------------------------

def test_a_save_that_fails_half_way_leaves_the_original_untouched(tmp_path, monkeypatch):
    """`Image.save(path)` truncates the destination before encoding, and Pillow
    only cleans up files it CREATED — so an encoder dying half-way (disk full, a
    scanner holding the file) used to leave the render as a stump while the
    function returned False, "having touched nothing". Written beside and swapped
    in atomically now; the original bytes must survive any failure."""
    from PIL import Image
    path = tmp_path / 'render.png'
    _write(path, _image(30))
    before = path.read_bytes()

    real_save = Image.Image.save

    def _boom(self, fp, *a, **k):
        # Simulate an encoder that writes a header then dies.
        if isinstance(fp, str) and '.finish-' in fp:
            with open(fp, 'wb') as fh:
                fh.write(b'\x89PNG\r\n\x1a\n')
            raise OSError(28, 'No space left on device')
        return real_save(self, fp, *a, **k)

    monkeypatch.setattr(Image.Image, 'save', _boom)
    assert pf.apply_to_file(path, sharpen=0.55) is False
    assert path.read_bytes() == before
    # ...and no temp file is left behind next to it.
    assert [p.name for p in tmp_path.iterdir()] == ['render.png']


def test_the_png_text_chunks_comfyui_writes_survive_the_pass(tmp_path):
    """SaveImage embeds `prompt` and `workflow` as PNG text chunks — the reason a
    finished render can be dropped back onto ComfyUI to recover its graph. A
    save from a pixel array strips them; the pass has to carry them across."""
    from PIL import Image, PngImagePlugin
    path = tmp_path / 'render.png'
    info = PngImagePlugin.PngInfo()
    info.add_text('prompt', '{"1": {"class_type": "KSampler"}}')
    info.add_text('workflow', '{"nodes": []}')
    Image.fromarray(pf.to_uint8(_image(31)), 'RGB').save(path, pnginfo=info)

    assert pf.apply_to_file(path, sharpen=0.55, grain=0.01, seed=1) is True
    with Image.open(path) as handle:
        assert handle.text.get('prompt') == '{"1": {"class_type": "KSampler"}}'
        assert handle.text.get('workflow') == '{"nodes": []}'


def test_a_webp_blob_is_written_back_lossless(tmp_path):
    """The bank's edited lane is lossless WebP; a default (lossy) save would
    quietly degrade every finished blob there."""
    from PIL import Image
    path = tmp_path / 'blob.webp'
    Image.fromarray(pf.to_uint8(_image(32)), 'RGB').save(path, lossless=True)
    before = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

    assert pf.apply_to_file(path, sharpen=0.55) is True
    after = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    # Lossless round trip of the sharpened result: identical to sharpening the
    # decoded input directly, to 8-bit rounding.
    np.testing.assert_allclose(after, pf.unsharp_mask(before, 0.55), atol=1.5 / 255)


@pytest.mark.parametrize('mix', [0.0, 0.2, 0.5, 1.0])
def test_grain_intensity_is_the_deviation_it_adds_whatever_the_mix(mix):
    """Two independent normals mixed as (1-m)·L + m·C have variance (1-m)²+m² —
    0.5 at m=0.5 — so the "how coloured" dial used to turn the grain DOWN by up
    to 29% as a side effect (17.6% at the shipped 0.2). Normalised now: the
    deviation is `intensity` at every mix, which is what the docstring claims."""
    image = np.full((256, 256, 3), 0.5, dtype=np.float32)
    out = pf.film_grain(image, 0.02, saturation_mix=mix, seed=3)
    assert abs(float((out - image).std()) - 0.02) < 0.0015, mix
