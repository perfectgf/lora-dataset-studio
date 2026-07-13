"""Rough GPU speed model for cloud-training time/cost estimates.

Deliberately approximate — every number it feeds the UI is labelled "≈". The
point is to ORDER GPU classes by speed and give a ballpark run time/cost so the
user can trade money for wall-clock at launch, not to benchmark precisely.

Relative training throughput is normalized to the RTX 3090 (= 1.0). The
seconds-per-step baselines are MEASURED on real vast.ai pods (2026-07-13,
runs #6/#7/#9): zimage 4.49 s/it and krea 8.84 s/it on RTX 3090, zimage
2.88 s/it on RTX 4080S (=> 1.56x). An unknown card name falls back to 1.0
(never crash on a GPU we haven't tabulated)."""
from __future__ import annotations

# Relative training throughput vs an RTX 3090 (higher = faster). Matched on a
# lowercase SPACELESS substring of vast's free-text gpu_name (vast writes both
# 'RTX 6000Ada' and 'RTX 6000 Ada'); the LONGEST matching key wins so
# 'rtx4080s' beats 'rtx4080' and 'a4000' beats 'a40'.
_SPEED = {
    'rtx3060': 0.45, 'rtx3070': 0.6, 'rtx3080': 0.8, 'rtx3090': 1.0,
    'rtx4070': 0.85, 'rtx4080s': 1.56, 'rtx4080': 1.35, 'rtx4090': 1.85,
    'rtx5070': 1.1, 'rtx5080': 1.9, 'rtx5090': 2.8,
    'a4000': 0.55, 'a5000': 0.9, 'a6000': 1.2, 'a40': 0.85, 'a30': 0.85,
    'a10': 0.6, 'v100': 0.55, 'titanrtx': 0.6,
    'quadrortx6000': 0.6, 'quadrortx8000': 0.65,
    'rtx6000ada': 1.9, 'rtx5000ada': 1.35, 'rtx4500ada': 1.0,
    # Blackwell workstation/server cards (PRO 6000 WS/S/Max-Q share the die)
    'rtxpro5000': 1.9, 'rtxpro6000': 3.0,
    'l40s': 1.85, 'l40': 1.7, 'l4': 0.5,
    'a100': 2.1, 'h100': 3.6, 'h200': 4.1, 'b200': 5.5,
}

# Baseline seconds/step on the RTX 3090, by family — measured live (see
# module docstring), not guessed. sdxl is here for completeness only (cloud
# training refuses it).
_SEC_PER_STEP = {'zimage': 4.5, 'krea': 8.8, 'sdxl': 3.0}
_DEFAULT_SEC_PER_STEP = 5.0


def speed_factor(gpu_name: str) -> float:
    """Relative throughput vs an RTX 3090 (1.0). Unknown card -> 1.0."""
    n = ''.join((gpu_name or '').lower().split())
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
