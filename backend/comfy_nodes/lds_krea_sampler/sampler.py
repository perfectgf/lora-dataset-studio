"""A multistep sampler for Krea 2 Turbo, exposed to ComfyUI as a SAMPLER.

WHAT IT IS
----------
Euler on the probability-flow ODE, with two optional departures from it:

  * **history** — blend each Euler step toward a second-order Adams-Bashforth
    (AB2) step, which uses the PREVIOUS step's derivative as well as the current
    one. AB2 is the standard two-step explicit linear multistep method; the
    variable-step form is used here because diffusion sigma schedules are not
    uniform.
  * **terminal extrapolation** — on the last step, where sigma lands on 0, take
    the straight line through the last two `denoised` estimates and evaluate it
    at sigma = 0, instead of simply returning the final `denoised`.

WHY THIS HELPS A TURBO MODEL SPECIFICALLY
-----------------------------------------
Krea 2 Turbo is run at ~8 steps, cfg 1.0. At that step count the two things this
file touches are exactly the two that dominate the result:

  * With few, large steps, plain Euler's local truncation error is no longer
    small. AB2 costs NOTHING extra — no additional model call, just the
    derivative we already computed last step — and cancels the leading error
    term. It is the cheapest accuracy available at this step count.
  * The final hop to sigma = 0 is, in Euler, "return `denoised` and hope". It is
    a single jump across the largest remaining distance in the schedule. Fitting
    a line through the last two estimates uses the model's own trend across that
    gap rather than freezing the last sample.

At 25+ steps both effects shrink toward nothing: the steps are small, Euler is
already accurate, and the terminal gap is tiny. **This node is not expected to
change a 25-step render, and that is not a defect.** Judge it at 8.

WHY IT IS A BLEND AND NOT A SWITCH
----------------------------------
AB2 is an EXTRAPOLATION: it assumes the derivative keeps trending the way it
just did. Early in sampling, when the latent is still mostly noise, that
assumption is worth very little and the extrapolation mostly amplifies noise.
So `history` is both a weight (how far toward AB2) and a schedule (`history_from`
/ `history_to`): 0 while the trajectory is meaningless, ramping in as it settles.
The same reasoning applies to the terminal step, which is why its strength is a
separate dial rather than a boolean.

RELATION TO THE STOCK SAMPLERS
------------------------------
ComfyUI already ships multistep samplers (`deis`, `res_multistep`), and they are
in the app's Krea whitelist. What is NOT reachable through them is the
combination this file exists for: a CONTINUOUS, SCHEDULED blend between Euler
and a multistep correction, plus explicit control of the sigma-to-zero landing.
`neutral` below is bit-exact Euler on purpose — it is the witness column for an
A/B, so that "the rewiring changed something" and "the sampler changed something"
can never be confused for one another.

NOTE ON `torch`/`comfy` IMPORTS
-------------------------------
This module is imported by the USER's ComfyUI interpreter, never by the app's.
`torch` and `comfy.*` are the only imports allowed here (see
`backend/comfy_nodes/README.md`); ComfyUI is by definition running when this
loads, so both are present.

But `comfy` is imported LAZILY, inside the node class, and the one k-diffusion
helper this needs (`to_d`) is written out below instead of imported. Two reasons,
both load-bearing:

  * **The numerics become testable.** With torch alone — no ComfyUI, no GPU —
    the app's own suite can import `sample_lds_krea_multistep` and assert that
    `neutral` reproduces Euler exactly. A sampler nobody can run in CI is a
    sampler whose regressions are found in somebody's render.
  * **`comfy.k_diffusion` is an internal.** Its module path has moved before.
    Reaching into it at import time turns a rename in someone else's repository
    into "ComfyUI fails to start" for every user who has this folder — the
    loudest possible failure for the smallest possible cause.
"""

from __future__ import annotations

import torch


# --- Presets -----------------------------------------------------------------
# Five points on one two-axis surface (how much history, how hard the landing).
# They exist so the choice offered to a user is a look, not five floats; the
# floats stay reachable through `custom` for the person tuning a specific model.
#
# `neutral` is not a "weak" preset. It is the reference: history 0 and no
# terminal extrapolation reduce the loop below to plain Euler, step for step, so
# selecting it answers "is the custom-sampling rewiring itself neutral?" without
# touching the app's graph.
PRESETS = {
    'neutral':  {'history': 0.00, 'terminal': False, 'terminal_strength': 0.00},
    'soft':     {'history': 0.25, 'terminal': True,  'terminal_strength': 0.25},
    'balanced': {'history': 0.25, 'terminal': True,  'terminal_strength': 0.50},
    'detailed': {'history': 0.50, 'terminal': True,  'terminal_strength': 1.00},
    'max':      {'history': 1.00, 'terminal': True,  'terminal_strength': 1.00},
}
PRESET_NAMES = list(PRESETS) + ['custom']

# The window defaults. Shared by every preset: they differ in HOW MUCH history,
# never in WHEN — one axis at a time is what makes an A/B between them readable.
DEFAULT_HISTORY_FROM = 0.25
DEFAULT_HISTORY_TO = 1.00


def _clamp01(value, fallback=0.0):
    """`value` as a float in [0, 1]; `fallback` when it is not a usable number.

    ComfyUI widget values arrive already typed, but this sampler is also driven
    by a JSON graph the app builds, where a key can be absent or a string.
    Clamping here rather than trusting the caller keeps the loop below free of
    defensive branches."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    if f != f:                      # NaN: comparisons below would all be False
        return fallback
    return min(1.0, max(0.0, f))


def _to_d(x, sigma, denoised):
    """The probability-flow ODE derivative, `(x - denoised) / sigma`.

    k-diffusion's `to_d` broadcasts sigma up to x's rank first, because there it
    is usually a per-batch-item vector. In this loop sigma is a 0-dim slice of
    the schedule, which already broadcasts against anything — but the rank-lifting
    is kept so a caller handing in a per-item sigma gets the right answer instead
    of a silently wrong one."""
    if sigma.ndim > 0:
        sigma = sigma.reshape(sigma.shape + (1,) * (x.ndim - sigma.ndim))
    return (x - denoised) / sigma


def _history_weight(progress, start, end):
    """The history blend weight at `progress` (0..1 through the step count).

    Zero before `start`, ramping linearly to one across the window, and HELD at
    one after `end`.

    That last clause is a deliberate choice, not an oversight. Reading the window
    as "outside it, weight 0" produces a discontinuity: the weight climbs to full
    at `end` and then slams back to zero for the remaining steps, so the tail of
    the trajectory is sampled by a different method than its middle — with the
    jump landing wherever the user happened to drag a slider. Reading it as "the
    blend phases IN over this window" is monotone, and is what the dials say they
    do. With the shipped default (`end` = 1.0) `progress` never exceeds the
    window, so no preset can tell the two readings apart; only a hand-tuned
    `custom` reaches this, and it should reach the sane one."""
    if end < start:
        start, end = end, start
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    span = end - start
    if span <= 1e-8:                # a zero-width window is a step function
        return 1.0
    return (progress - start) / span


@torch.no_grad()
def sample_lds_krea_multistep(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    history=0.5,
    history_from=DEFAULT_HISTORY_FROM,
    history_to=DEFAULT_HISTORY_TO,
    terminal=True,
    terminal_strength=1.0,
):
    """One denoising trajectory. Signature is ComfyUI's sampler protocol: the
    keyword arguments after `disable` are the `extra_options` handed to KSAMPLER.

    Returns the final latent. Never raises on degenerate schedules — a
    single-entry `sigmas`, a repeated sigma, a zero-length window — because a
    sampler that throws mid-grid loses every tile of a run, and every one of
    those cases has a defensible plain-Euler answer."""
    extra_args = {} if extra_args is None else extra_args

    history = _clamp01(history)
    history_from = _clamp01(history_from, DEFAULT_HISTORY_FROM)
    history_to = _clamp01(history_to, DEFAULT_HISTORY_TO)
    terminal_strength = _clamp01(terminal_strength, 1.0)
    terminal = bool(terminal)

    # `model` wants sigma as a per-batch-item tensor, not a scalar.
    sigma_batch = x.new_ones([x.shape[0]])

    steps = len(sigmas) - 1
    if steps < 1:
        return x

    # The previous step's derivative (AB2's second term) and the previous
    # `denoised` (the terminal line's second point). Both start empty: the first
    # step has no history, by definition, and falls back to Euler.
    prev_d = None
    prev_sigma = None
    prev_denoised = None
    prev_denoised_sigma = None

    for i in range(steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * sigma_batch, **extra_args)

        if callback is not None:
            # `sigma_hat` is in the contract because the previewers read it; for
            # a non-stochastic sampler it is just sigma (no churn step).
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma,
                      'denoised': denoised})

        if float(sigma_next) == 0.0:
            # --- The landing ---------------------------------------------
            # Euler's answer here is `denoised` itself. The alternative is the
            # line through (prev_sigma, prev_denoised) and (sigma, denoised),
            # read at sigma = 0 — i.e. where the model's own estimate was
            # HEADING, rather than where it last stood.
            #
            # Guarded three ways because each guard corresponds to a real
            # schedule: no previous estimate (a one-step schedule), two
            # estimates at the same sigma (a duplicated entry), and the user
            # simply turning it off.
            landed = denoised
            if (terminal and terminal_strength > 0.0
                    and prev_denoised is not None):
                gap = sigma - prev_denoised_sigma
                if abs(float(gap)) > 1e-12:
                    slope = (denoised - prev_denoised) / gap
                    extrapolated = denoised + slope * (0.0 - sigma)
                    landed = torch.lerp(landed, extrapolated, terminal_strength)
            x = landed
            break

        h = sigma_next - sigma
        d = _to_d(x, sigma, denoised)

        # `i + 1` so the schedule is expressed in steps COMPLETED: at the first
        # step nothing has been integrated yet, and a window starting at 0.0
        # should still mean "from the very first move".
        weight = history * _history_weight((i + 1) / steps, history_from, history_to)

        if prev_d is None or weight <= 0.0:
            x = x + d * h
        else:
            h_prev = sigma - prev_sigma
            if abs(float(h_prev)) < 1e-12:
                # Two identical sigmas: AB2's ratio would divide by ~0 and the
                # "correction" would be numerical garbage. Euler is exact enough
                # for a step of length ~0 anyway.
                x = x + d * h
            else:
                # Variable-step AB2. With h == h_prev this is the textbook
                # (3/2)d_n - (1/2)d_{n-1}; the ratio generalises it to the uneven
                # spacing every diffusion schedule actually has.
                ratio = h / (2.0 * h_prev)
                ab2 = h * ((1.0 + ratio) * d - ratio * prev_d)
                euler = h * d
                x = x + torch.lerp(euler, ab2, weight)

        prev_d = d
        prev_sigma = sigma
        prev_denoised = denoised
        prev_denoised_sigma = sigma

    return x


class LDSKrea2PresetSampler:
    """ComfyUI node: pick a preset, get a SAMPLER for SamplerCustom(Advanced).

    It deliberately does NOT own steps, cfg, scheduler or seed. Those stay on the
    scheduler/guider/noise nodes beside it, which is what makes this node a drop-in
    for a KSampler's sampler choice rather than a second place where the same
    numbers live."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'preset': (PRESET_NAMES, {'default': 'balanced'}),
            },
            'optional': {
                'history': ('FLOAT', {
                    'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                    'tooltip': 'preset=custom only. How far each step leans from '
                               'Euler toward the second-order correction.'}),
                'history_from': ('FLOAT', {
                    'default': DEFAULT_HISTORY_FROM, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                    'tooltip': 'preset=custom only. Fraction of the run before the '
                               'correction starts phasing in.'}),
                'history_to': ('FLOAT', {
                    'default': DEFAULT_HISTORY_TO, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                    'tooltip': 'preset=custom only. Fraction of the run by which it '
                               'is fully phased in.'}),
                'terminal': ('BOOLEAN', {
                    'default': True,
                    'tooltip': 'preset=custom only. Extrapolate the final step to '
                               'sigma 0 instead of returning the last estimate.'}),
                'terminal_strength': ('FLOAT', {
                    'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05,
                    'tooltip': 'preset=custom only. 0 = the plain estimate, '
                               '1 = the full extrapolation.'}),
            },
        }

    RETURN_TYPES = ('SAMPLER', 'STRING')
    RETURN_NAMES = ('sampler', 'settings')
    FUNCTION = 'build'
    CATEGORY = 'LoRA Dataset Studio'
    DESCRIPTION = ('Multistep sampler tuned for Krea 2 Turbo at low step counts. '
                   'neutral = plain Euler (the A/B reference); balanced is the '
                   'general-purpose setting; detailed and max push texture harder.')

    def build(self, preset, history=0.5, history_from=DEFAULT_HISTORY_FROM,
              history_to=DEFAULT_HISTORY_TO, terminal=True, terminal_strength=1.0):
        if preset == 'custom':
            options = {
                'history': _clamp01(history),
                'history_from': _clamp01(history_from, DEFAULT_HISTORY_FROM),
                'history_to': _clamp01(history_to, DEFAULT_HISTORY_TO),
                'terminal': bool(terminal),
                'terminal_strength': _clamp01(terminal_strength, 1.0),
            }
        else:
            # An unknown preset resolves to `balanced` rather than raising: this
            # node is fed by a graph the app generates, and a name the app and
            # the node disagree about (an install mid-update) must degrade to a
            # render, not to a dead tile.
            base = PRESETS.get(preset) or PRESETS['balanced']
            options = {
                'history': base['history'],
                'history_from': DEFAULT_HISTORY_FROM,
                'history_to': DEFAULT_HISTORY_TO,
                'terminal': base['terminal'],
                'terminal_strength': base['terminal_strength'],
            }

        # Imported here, not at module scope: see the note in the module
        # docstring. By the time a graph executes this node, ComfyUI is running.
        import comfy.samplers

        sampler = comfy.samplers.KSAMPLER(sample_lds_krea_multistep,
                                          extra_options=options)

        # Echoed as a STRING so a saved image's metadata can carry WHAT RAN, not
        # just which preset was asked for — the two differ whenever a preset is
        # renamed or a custom value is clamped.
        settings = (
            f"{preset} | history {options['history']:.2f} "
            f"({options['history_from']:.2f}->{options['history_to']:.2f}) | "
            f"terminal {'on' if options['terminal'] else 'off'} "
            f"{options['terminal_strength']:.2f}"
        )
        return sampler, settings
