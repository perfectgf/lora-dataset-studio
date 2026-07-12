"""Rough GPU speed model for cloud-training time/cost estimates.

Deliberately approximate — every number it feeds the UI is labelled "≈". The
point is to ORDER GPU classes by speed and give a ballpark run time/cost so the
user can trade money for wall-clock at launch, not to benchmark precisely.

Relative training throughput is normalized to the RTX 3090 (= 1.0), the class we
have actually measured a real Krea run on. An unknown card name falls back to
1.0 (never crash on a GPU we haven't tabulated)."""
from __future__ import annotations

# Relative training throughput vs an RTX 3090 (higher = faster). Matched on a
# lowercase substring of vast's free-text gpu_name; the LONGEST matching key
# wins so 'rtx 6000 ada' beats a bare 'rtx 6000', and 'rtx 4090' isn't shadowed
# by a hypothetical '4090' partial.
_SPEED = {
    'rtx 3060': 0.45, 'rtx 3070': 0.6, 'rtx 3080': 0.8, 'rtx 3090': 1.0,
    'rtx 4070': 0.85, 'rtx 4080': 1.35, 'rtx 4090': 1.85,
    'rtx 5070': 1.1, 'rtx 5080': 1.9, 'rtx 5090': 2.8,
    'a4000': 0.55, 'a5000': 0.9, 'a6000': 1.2,
    'rtx 6000 ada': 1.9, 'l40s': 1.85, 'l40': 1.7,
    'a100': 2.1, 'h100': 3.6, 'h200': 4.1, 'b200': 5.5,
}

# Baseline seconds/step on the RTX 3090, by family. Krea (12B) is heavier than
# Z-Image; sdxl is here for completeness only (cloud training refuses it).
_SEC_PER_STEP = {'zimage': 0.9, 'krea': 1.1, 'sdxl': 0.75}
_DEFAULT_SEC_PER_STEP = 0.95


def speed_factor(gpu_name: str) -> float:
    """Relative throughput vs an RTX 3090 (1.0). Unknown card -> 1.0."""
    n = (gpu_name or '').lower()
    best_len, best_val = -1, 1.0
    for key, val in _SPEED.items():
        if key in n and len(key) > best_len:
            best_len, best_val = len(key), val
    return best_val


def sec_per_step(family: str) -> float:
    return _SEC_PER_STEP.get(family, _DEFAULT_SEC_PER_STEP)


def estimate_minutes(gpu_name: str, family: str, steps: int) -> float:
    """Approximate TRAINING time (excludes pod boot/download overhead — the
    caller adds that only to the cost, so the shown duration is training)."""
    train_sec = max(0, int(steps or 0)) * sec_per_step(family) / speed_factor(gpu_name)
    return train_sec / 60.0
