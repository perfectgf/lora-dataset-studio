"""The Krea hi-res fix: the second latent pass, and the order it has to run in.

Four things are pinned here, and they are the ones that cannot be caught by
reading the diff — every one of them fails as a render that merely looks wrong,
never as an error:

1. **OFF adds nothing.** The setting ships at 1.0 and the whole feature has to be
   invisible until somebody raises it. Not "renders about the same" — the built
   graph must be the SAME OBJECT it was before the feature existed, because that
   is the only claim strong enough to be worth making about a default.
2. **Pass 2 carries the LoRA stack.** Pass 2 is cloned from pass 1, `model`
   included. Cloned before `inject_krea_loras`, it clones a model input that is
   about to be rewritten: pass 1 renders with the LoRAs, pass 2 renders without
   them, and the picture is merely disappointing. Same failure shape as the
   preset sampler's own ordering bug, so the same discipline — follow the wire.
3. **It composes with the preset sampler, in one direction only.** Hi-res first
   works because the preset swap repoints every consumer of the node it deletes,
   which by then is our LatentUpscaleBy. The other way round, the KSampler this
   reads is already gone; that has to be a clean no-op, not half a graph with a
   dangling upscale wired to a node id that no longer exists.
4. **`KSamplerAdvanced` is refused.** It has no `denoise` input — it truncates
   with `start_at_step` — so a clone carrying `denoise` would be a node whose one
   meaningful dial does nothing at all.

Nothing here renders: the tests build graphs and follow wires.
"""
import json

import pytest

from app import config as cfg
from app.utils.comfyui import (KREA_HIRES_DENOISE, KREA_HIRES_MAX_SCALE,
                               inject_krea_hires_fix,
                               inject_krea_loras, inject_krea_preset_sampler)

UPSCALE = 'krea_hires_upscale'
SAMPLER = 'krea_hires_sampler'


def _krea_graph():
    """The shipped Krea template, loaded from disk — not a hand-written stub, which
    would freeze the shape the graph had the day this was written and keep passing
    after the template moved underneath it."""
    with open(cfg.BACKEND_DIR / 'workflows' / 'krea2_turbo.json', encoding='utf-8') as fh:
        return json.load(fh)


# --- 1. OFF is genuinely off -------------------------------------------------

@pytest.mark.parametrize('scale', [None, 0, 1.0, 0.5, '', 'abc', object(),
                                   float('nan'), float('inf')])
def test_off_leaves_the_graph_byte_identical(scale):
    """Every shape of "no" — unset, neutral, shrinking, unparseable — has to leave
    the template untouched. Compared against a freshly loaded copy rather than
    against a node count: a mutation that swapped a wire without adding a node
    would pass a count and change every render.

    NaN is in the list because it is the one that does NOT behave: every
    comparison against it is False, so it sails past `scale <= 1.0`, and
    `min(MAX, nan)` then returns MAX. Without an explicit check a corrupt value
    does not disable this pass, it enables it at full strength — measured here,
    not theorised."""
    wf = _krea_graph()
    assert inject_krea_hires_fix(wf, scale) == 0
    assert wf == _krea_graph()


def test_a_scale_that_enlarges_adds_exactly_two_nodes():
    wf = _krea_graph()
    before = set(wf)
    assert inject_krea_hires_fix(wf, 1.5) == 1
    assert set(wf) - before == {UPSCALE, SAMPLER}
    assert wf[UPSCALE]['class_type'] == 'LatentUpscaleBy'
    assert wf[SAMPLER]['class_type'] == 'KSampler'
    # Core ComfyUI classes only: the feature has to work on a bare install, so it
    # may not reach for a pack the installer would then have to fetch.
    assert {wf[UPSCALE]['class_type'], wf[SAMPLER]['class_type']} <= {
        'LatentUpscaleBy', 'KSampler'}


# --- 2. The wiring -----------------------------------------------------------

def test_the_pass_is_inserted_between_pass_one_and_its_consumer():
    """Pass 1 -> upscale -> pass 2 -> VAEDecode. The upscale is the ONE consumer
    that must keep reading pass 1; everything else moves to pass 2."""
    wf = _krea_graph()
    assert wf['27']['inputs']['samples'] == ['26', 0], 'precondition'

    inject_krea_hires_fix(wf, 1.5)

    assert wf[UPSCALE]['inputs']['samples'] == ['26', 0]
    assert wf[SAMPLER]['inputs']['latent_image'] == [UPSCALE, 0]
    assert wf['27']['inputs']['samples'] == [SAMPLER, 0]


def test_pass_two_carries_the_lora_stack():
    """THE regression this file exists for. Follow the model wire, do not trust
    that the two nodes happen to hold equal-looking values."""
    wf = _krea_graph()
    requested = [{'filename': 'krea/a.safetensors', 'strength': 1.0},
                 {'filename': 'krea/b.safetensors', 'strength': 0.5}]
    assert inject_krea_loras(wf, requested, allowed={r['filename'] for r in requested}) == 2
    model_src = wf['26']['inputs']['model']
    assert model_src[0] != '20', 'precondition: the model no longer comes from the loader'

    inject_krea_hires_fix(wf, 1.5)

    assert wf[SAMPLER]['inputs']['model'] == model_src


def test_pass_two_inherits_every_dial_but_the_three_it_owns():
    wf = _krea_graph()
    wf['26']['inputs'].update({'seed': 424242, 'steps': 9, 'cfg': 1.5,
                               'sampler_name': 'er_sde', 'scheduler': 'beta',
                               'denoise': 1.0})
    positive, negative = wf['26']['inputs']['positive'], wf['26']['inputs']['negative']

    inject_krea_hires_fix(wf, 1.5)

    got = wf[SAMPLER]['inputs']
    assert (got['seed'], got['cfg'], got['sampler_name'], got['scheduler']) == (
        424242, 1.5, 'er_sde', 'beta')
    assert (got['positive'], got['negative']) == (positive, negative)
    # The three it owns.
    assert got['latent_image'] == [UPSCALE, 0]
    assert got['steps'] == 9, 'steps=None inherits pass 1'
    assert got['denoise'] == KREA_HIRES_DENOISE


def test_the_two_passes_do_not_share_link_objects():
    """Both nodes hold `[node, slot]` lists. Sharing the SAME list means a later
    in-place repoint of one silently moves the other — a coupling nobody would
    think to look for, and one the preset-sampler swap walks straight into."""
    wf = _krea_graph()
    inject_krea_hires_fix(wf, 1.5)
    for key in ('model', 'positive', 'negative'):
        assert wf[SAMPLER]['inputs'][key] == wf['26']['inputs'][key]
        assert wf[SAMPLER]['inputs'][key] is not wf['26']['inputs'][key]


# --- 3. Clamps and refusals --------------------------------------------------

def test_scale_and_denoise_are_clamped():
    wf = _krea_graph()
    inject_krea_hires_fix(wf, 99, denoise=5)
    assert wf[UPSCALE]['inputs']['scale_by'] == KREA_HIRES_MAX_SCALE
    assert wf[SAMPLER]['inputs']['denoise'] == 1.0

    wf = _krea_graph()
    inject_krea_hires_fix(wf, 1.5, denoise=-3, steps=999)
    assert wf[SAMPLER]['inputs']['denoise'] == 0.05
    assert wf[SAMPLER]['inputs']['steps'] == 50


def test_a_ksampler_advanced_is_refused():
    """It has no `denoise`; a clone carrying one would look configured and behave
    as if the dial were not there."""
    wf = _krea_graph()
    wf['26']['class_type'] = 'KSamplerAdvanced'
    assert inject_krea_hires_fix(wf, 1.5) == 0
    assert UPSCALE not in wf and SAMPLER not in wf


def test_a_missing_sampler_node_is_refused():
    wf = _krea_graph()
    del wf['26']
    assert inject_krea_hires_fix(wf, 1.5) == 0
    assert UPSCALE not in wf and SAMPLER not in wf


# --- 4. Composition with the preset sampler ----------------------------------

def test_hires_then_preset_gives_a_three_stage_graph():
    """The preset swap repoints every consumer of the node it deletes. By then
    that consumer is our upscale, so pass 1 becomes the preset chain and the rest
    follows — neither injector knowing about the other."""
    wf = _krea_graph()
    assert inject_krea_hires_fix(wf, 1.5) == 1
    assert inject_krea_preset_sampler(wf, 'balanced') == 1

    assert '26' not in wf
    assert wf[UPSCALE]['inputs']['samples'] == ['krea_ps_run', 0]
    assert wf[SAMPLER]['inputs']['latent_image'] == [UPSCALE, 0]
    assert wf['27']['inputs']['samples'] == [SAMPLER, 0]


def test_preset_then_hires_is_a_clean_no_op():
    """The wrong order must not build half a pass. `inject_krea_hires_fix` reads
    node 26, which the preset swap has already deleted — the answer is 0 and an
    untouched graph, never an upscale wired to a node id that is gone."""
    wf = _krea_graph()
    inject_krea_preset_sampler(wf, 'balanced')
    after_preset = json.loads(json.dumps(wf))

    assert inject_krea_hires_fix(wf, 1.5) == 0
    assert wf == after_preset


# --- 5. The setting ----------------------------------------------------------

def _settings():
    from app.services.lora_test_studio import krea_hires_settings
    return krea_hires_settings()


def test_the_shipped_default_is_off():
    """Read through the real defaults, so a future edit that ships this enabled
    fails here rather than surprising every existing install."""
    assert cfg.DEFAULTS['krea_hires']['scale'] == 1.0
    assert _settings()['scale'] is None


@pytest.mark.parametrize('raw', [None, '', 'abc', {}, 1.0, 0.5,
                                 float('nan'), float('inf')])
def test_a_bad_or_neutral_setting_degrades_to_off(monkeypatch, raw):
    """Read once per cell: a malformed value that fell back to "some upscale"
    would quietly quadruple the cost of a whole grid. OFF lives in `scale` alone
    — the other two dials are still read, so a run that arms the pass from the
    panel finds the configured rewrite and step count waiting."""
    monkeypatch.setattr(cfg, 'get',
                        lambda dotted, default=None:
                        raw if dotted == 'krea_hires.scale' else default)
    got = _settings()
    assert got['scale'] is None
    assert got['steps'] is None
    assert got['denoise'] == KREA_HIRES_DENOISE


def test_the_setting_is_clamped_and_zero_steps_means_inherit(monkeypatch):
    values = {'krea_hires.scale': 99, 'krea_hires.denoise': 0.4, 'krea_hires.steps': 0}
    monkeypatch.setattr(cfg, 'get',
                        lambda dotted, default=None: values.get(dotted, default))
    assert _settings() == {'scale': KREA_HIRES_MAX_SCALE, 'steps': None, 'denoise': 0.4}


# --- What the adversarial pass found -------------------------------------------

def test_a_run_that_arms_the_pass_while_the_switch_is_off_keeps_the_configured_dials(monkeypatch):
    """The shipped default is off. A run that arms the pass from the Studio panel
    still owes the rewrite and the step count set here — the OFF state used to
    return before reading them, so `krea_hires.steps = 12` was silently lost."""
    from app.services.lora_test_studio import _krea_hires_for_cell
    values = {'krea_hires.scale': 1.0, 'krea_hires.denoise': 0.7, 'krea_hires.steps': 12}
    monkeypatch.setattr(cfg, 'get',
                        lambda dotted, default=None: values.get(dotted, default))
    assert _settings() == {'scale': None, 'steps': 12, 'denoise': 0.7}
    got = _krea_hires_for_cell(1.5, None)
    assert (got['hires_scale'], got['hires_steps'], got['hires_denoise']) == (1.5, 12, 0.7)


@pytest.mark.parametrize('raw', [float('inf'), float('-inf'), float('nan'), 'inf'])
def test_a_non_finite_step_count_degrades_instead_of_raising(monkeypatch, raw):
    """`int(float('inf'))` raises OverflowError — neither TypeError nor
    ValueError — and it used to escape as a 500 on every Krea run. Both readers
    of a step count have to survive it."""
    values = {'krea_hires.scale': 1.5, 'krea_hires.denoise': 0.5, 'krea_hires.steps': raw}
    monkeypatch.setattr(cfg, 'get',
                        lambda dotted, default=None: values.get(dotted, default))
    assert _settings()['steps'] is None
    wf = _krea_graph()
    assert inject_krea_hires_fix(wf, 1.5, steps=raw) == 1
    assert wf[SAMPLER]['inputs']['steps'] == wf['26']['inputs']['steps']


def test_the_setting_reaches_the_built_graph(monkeypatch):
    """The seam the two halves meet at: a value in settings has to come out as
    nodes in the graph the cell builder hands to ComfyUI."""
    from app.services import lora_test_studio as lts

    values = {'krea_hires.scale': 1.5, 'krea_hires.denoise': 0.45,
              'krea_hires.steps': 6}
    monkeypatch.setattr(cfg, 'get',
                        lambda dotted, default=None: values.get(dotted, default))
    monkeypatch.setattr(lts, 'krea_default_base', lambda: None)

    wf = _krea_graph()
    lts.apply_krea_lora_test_settings(
        wf, lora_name='krea/a.safetensors', strength=1.0, prompt='a photo',
        seed=7, width=1024, height=1024,
        allowed_loras={'krea/a.safetensors'},
        **{f'hires_{k}': v for k, v in lts.krea_hires_settings().items()})

    assert wf[UPSCALE]['inputs']['scale_by'] == 1.5
    assert wf[SAMPLER]['inputs']['denoise'] == 0.45
    assert wf[SAMPLER]['inputs']['steps'] == 6
    # And the LoRA it was asked for is on pass 2 as well, through the shared wire.
    assert wf[SAMPLER]['inputs']['model'] == wf['26']['inputs']['model']
    assert wf[SAMPLER]['inputs']['model'][0].startswith('krea_lora_')
