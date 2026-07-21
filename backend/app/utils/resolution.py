"""Z-Image / SD3-family output resolution tiers.

The format selector fixes the aspect RATIO; this maps (aspect_ratio, tier) -> a
concrete W×H. The ratio is preserved, dimensions are rounded to a multiple of 16
(SD3/Z-Image latent granularity). The tier's megapixels are capped at 1536 px
per side (Z-Image's safe band) — that capped size IS the preset. An optional
`resolution_multiplier` (1.0–1.9) then enlarges the preset LINEARLY (W and H both
× multiplier) for users who want to push past the training resolution; the result
is bounded by a higher absolute safety cap (3072/side) so ×1.9 reaches its full
size without being clipped by the tier cap while still guarding against absurd
requests. Beyond ~1.5× the base model may duplicate/artefact or OOM — the UI warns
but does not block; true high-res should still prefer the upscaler.

Mirrored EXACTLY by the frontend (react-frontend/src/components/shared/
ResolutionSelector.jsx): same ratios, tiers, caps, multiplier clamp, ÷16 snap and
round-half-up rounding, so the px shown there match what's generated.
"""
import math

# Canonical w:h ratios, matching the labels shown in AspectRatioSelector.jsx.
_RATIOS = {
    "square": (1, 1),
    "landscape": (4, 3),
    "portrait": (3, 4),
    "widescreen": (16, 9),
    "tall": (9, 16),
    "photo": (3, 2),
    "phototall": (2, 3),
    "ultrawide": (21, 9),
}

# Tier -> target megapixels (the "size" knob; ratio comes from the format).
_TIERS = {
    "fast": 0.7,
    "standard": 1.0,
    "hq": 1.3,
    "max": 1.6,
}
DEFAULT_TIER = "standard"

_CAP = 1536      # tier cap: max px per side of the PRESET (Z-Image safe band)
_ABS_CAP = 3072  # absolute safety cap applied AFTER the multiplier (long side)
_FLOOR = 512     # min px per side
_MULT = 16       # latent granularity (SD3 / Z-Image)
MULT_MIN = 1.0   # resolution multiplier range (1.0 = preset unchanged)
MULT_MAX = 1.9


def _snap(v):
    # Round HALF-UP (not Python's banker's rounding) so a value landing exactly on
    # x.5 latent steps matches the frontend's Math.round — otherwise e.g. a 1000 px
    # side (62.5 steps) snapped to 992 here but 1008 there, and front/back diverged.
    return max(_FLOOR, int(math.floor(v / _MULT + 0.5)) * _MULT)


def clamp_multiplier(multiplier):
    """Clamp the resolution multiplier to [MULT_MIN, MULT_MAX]; None/garbage → 1.0.
    Never shrinks below the preset (floor 1.0)."""
    try:
        m = float(multiplier)
    except (TypeError, ValueError):
        return MULT_MIN
    if math.isnan(m):
        return MULT_MIN
    return max(MULT_MIN, min(MULT_MAX, m))


def compute_tier_dims(aspect_ratio, resolution_tier=DEFAULT_TIER, resolution_multiplier=1.0):
    """Return (width, height) for a format + resolution tier + optional multiplier.

    Ratio-preserving, divisible by 16. Pipeline: tier megapixels → cap the PRESET at
    1536/side → enlarge linearly by `resolution_multiplier` (clamped 1.0–1.9, W and H
    both) → cap at the absolute 3072/side safety ceiling → snap ÷16 (round-half-up),
    floor 512. At multiplier 1.0 this is byte-identical to the old preset-only result.
    Unknown aspect_ratio falls back to square; unknown tier to the default tier.
    """
    rw, rh = _RATIOS.get(aspect_ratio, _RATIOS["square"])
    mp = _TIERS.get(resolution_tier, _TIERS[DEFAULT_TIER])
    r = rw / rh
    px = mp * 1_000_000
    h = math.sqrt(px / r)
    w = r * h
    # 1) Tier cap — this capped size is the PRESET the user picks.
    longest = max(w, h)
    if longest > _CAP:
        scale = _CAP / longest
        w *= scale
        h *= scale
    # 2) Linear multiplier on the preset (clamped, floor 1.0 = no change).
    m = clamp_multiplier(resolution_multiplier)
    w *= m
    h *= m
    # 3) Absolute safety cap AFTER the multiplier so 1536 never clips the enlarged
    #    result, but a runaway request still can't exceed 3072/side.
    longest = max(w, h)
    if longest > _ABS_CAP:
        scale = _ABS_CAP / longest
        w *= scale
        h *= scale
    return _snap(w), _snap(h)
