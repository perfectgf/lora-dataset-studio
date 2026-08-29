"""The Krea 2 preset sampler: its numerics, its graph swap, and its installer.

Three things are pinned here, and they are the three that cannot be verified by
reading:

1. **`neutral` IS Euler.** The preset exists to be the reference column of an
   A/B, which is only worth anything if it reproduces the stock trajectory
   exactly. Asserted against an Euler loop written out in the test, so the two
   cannot drift by sharing an implementation.
2. **The swap keeps the wiring.** The injector reads `KSampler.model`, which the
   LoRA chain and the enhancer rewrite before it runs. Called in the wrong order
   it silently drops the whole stack and still renders. The test builds the real
   graph, in the real order, and follows the model wire to its source.
3. **The app and the node it ships agree.** The preset names live in two files —
   the node's PRESETS and the app's KREA_SAMPLER_PRESETS — and a name the app
   offers that the node does not know would be a dead dropdown entry.

Nothing here renders anything: no GPU second, no paid call. The numeric test runs
on CPU tensors of shape (1, 4, 8, 8).
"""
import os
import sys

import pytest

# The shipped node is not part of the `app` package — it is code we install into
# somebody else's ComfyUI — so it is imported by path, the way that ComfyUI will.
_NODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'comfy_nodes')


def _load_node_module():
    """The sampler module, imported without ComfyUI. Skips when torch is absent.

    This import working AT ALL is half the point of the test: `sampler.py` keeps
    `comfy` out of its module scope precisely so the numerics can be exercised
    here, and an import that starts needing ComfyUI would fail this call rather
    than quietly stop being covered."""
    pytest.importorskip('torch')
    if _NODE_DIR not in sys.path:
        sys.path.insert(0, _NODE_DIR)
    from lds_krea_sampler import sampler
    return sampler


# --- 1. The numerics ---------------------------------------------------------

def _linear_denoiser():
    """A stand-in for the diffusion model: `denoised = x * 0.5`.

    Deliberately NOT a constant. A constant denoiser makes every derivative
    identical, and AB2's correction term vanishes when d == prev_d — so a
    constant would let a broken history blend pass as neutral."""
    return lambda x, sigma, **kwargs: x * 0.5


def test_neutral_preset_reproduces_euler_exactly():
    sampler = _load_node_module()
    import torch

    sigmas = torch.tensor([14.0, 9.0, 5.0, 2.0, 0.7, 0.0])
    x0 = torch.linspace(-1.0, 1.0, 1 * 4 * 8 * 8).reshape(1, 4, 8, 8)
    model = _linear_denoiser()

    # Plain Euler, written out here rather than imported, so a change to the
    # module cannot move both sides of the comparison at once.
    x = x0.clone()
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i])
        if float(sigmas[i + 1]) == 0.0:
            x = denoised
            break
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    euler = x

    got = sampler.sample_lds_krea_multistep(
        model, x0.clone(), sigmas, **sampler.PRESETS['neutral'],
        history_from=sampler.DEFAULT_HISTORY_FROM,
        history_to=sampler.DEFAULT_HISTORY_TO)

    assert torch.equal(got, euler), 'neutral must be bit-exact Euler'


def test_history_and_terminal_each_change_the_result():
    """Both dials must actually do something — separately.

    A preset table is easy to get wrong in a way that renders fine: pass the
    wrong key and every preset silently collapses onto the default. Asserting
    that each axis moves the output ALONE is what catches that."""
    sampler = _load_node_module()
    import torch

    sigmas = torch.tensor([14.0, 9.0, 5.0, 2.0, 0.7, 0.0])
    x0 = torch.linspace(-1.0, 1.0, 1 * 4 * 8 * 8).reshape(1, 4, 8, 8)
    model = _linear_denoiser()

    def run(**kw):
        opts = {'history': 0.0, 'history_from': 0.0, 'history_to': 1.0,
                'terminal': False, 'terminal_strength': 0.0}
        opts.update(kw)
        return sampler.sample_lds_krea_multistep(model, x0.clone(), sigmas, **opts)

    base = run()
    assert not torch.equal(run(history=1.0), base), 'history had no effect'
    assert not torch.equal(run(terminal=True, terminal_strength=1.0), base), \
        'terminal extrapolation had no effect'


def test_presets_are_ordered_by_strength():
    """`neutral` .. `max` is a scale, and the UI presents it as one. A preset that
    moves the image LESS than the one before it would make the dropdown lie."""
    sampler = _load_node_module()
    import torch

    sigmas = torch.tensor([14.0, 9.0, 5.0, 2.0, 0.7, 0.0])
    x0 = torch.linspace(-1.0, 1.0, 1 * 4 * 8 * 8).reshape(1, 4, 8, 8)
    model = _linear_denoiser()

    def distance(preset):
        out = sampler.sample_lds_krea_multistep(
            model, x0.clone(), sigmas, **sampler.PRESETS[preset],
            history_from=sampler.DEFAULT_HISTORY_FROM,
            history_to=sampler.DEFAULT_HISTORY_TO)
        neutral = sampler.sample_lds_krea_multistep(
            model, x0.clone(), sigmas, **sampler.PRESETS['neutral'],
            history_from=sampler.DEFAULT_HISTORY_FROM,
            history_to=sampler.DEFAULT_HISTORY_TO)
        return float((out - neutral).abs().mean())

    walked = [distance(p) for p in ('neutral', 'soft', 'balanced', 'detailed', 'max')]
    assert walked[0] == 0.0
    assert walked == sorted(walked), f'presets are not monotone: {walked}'


def test_degenerate_schedules_do_not_raise():
    """A sampler that throws mid-grid loses every remaining tile of a run, so each
    of these has to land on a defensible answer instead of an exception."""
    sampler = _load_node_module()
    import torch

    x0 = torch.zeros(1, 4, 8, 8)
    model = _linear_denoiser()
    for sigmas in (torch.tensor([1.0]),                 # no step at all
                   torch.tensor([1.0, 0.0]),            # straight to the landing
                   torch.tensor([2.0, 2.0, 0.0]),       # a repeated sigma
                   torch.tensor([2.0, 1.0, 1.0, 0.0])):
        out = sampler.sample_lds_krea_multistep(
            model, x0.clone(), sigmas, **sampler.PRESETS['max'],
            history_from=0.0, history_to=1.0)
        assert torch.isfinite(out).all(), f'non-finite output for {sigmas.tolist()}'


def test_history_window_holds_after_its_end():
    """Past `history_to` the blend stays FULL, it does not snap back to zero.

    The alternative reading (weight 0 outside the window) puts a discontinuity in
    the middle of a trajectory, at whatever step a slider happens to land on. This
    pins the monotone reading the docstring promises."""
    sampler = _load_node_module()
    weights = [sampler._history_weight(p / 10, 0.2, 0.6) for p in range(11)]
    assert weights[0] == 0.0
    assert weights[-1] == 1.0
    assert weights == sorted(weights), f'window is not monotone: {weights}'


# --- 2. The graph swap -------------------------------------------------------

def _krea_graph():
    """The shipped Krea template, loaded from disk — not a hand-written stub.

    A stub would freeze whatever shape the graph had the day this was written and
    keep passing after the template changed underneath it."""
    import json
    from app import config as cfg
    with open(cfg.BACKEND_DIR / 'workflows' / 'krea2_turbo.json', encoding='utf-8') as fh:
        return json.load(fh)


def test_swap_replaces_the_ksampler_and_repoints_its_consumer():
    from app.utils.comfyui import inject_krea_preset_sampler

    wf = _krea_graph()
    assert wf['26']['class_type'] == 'KSampler'
    assert wf['27']['inputs']['samples'] == ['26', 0]

    assert inject_krea_preset_sampler(wf, 'balanced') == 1

    assert '26' not in wf, 'the replaced KSampler must not linger in the graph'
    assert wf['27']['inputs']['samples'][0] == 'krea_ps_run'
    # Output 0 is `output`; output 1 is `denoised_output`, a different tensor.
    assert wf['27']['inputs']['samples'][1] == 0
    assert wf['krea_ps_sampler']['inputs']['preset'] == 'balanced'
    assert wf['krea_ps_run']['class_type'] == 'SamplerCustomAdvanced'


def test_swap_carries_every_dial_off_the_ksampler():
    """seed, steps, cfg, scheduler, denoise and both conditionings must survive the
    move. Each one lands on a DIFFERENT node now, which is exactly how one gets
    dropped without anybody noticing."""
    from app.utils.comfyui import inject_krea_preset_sampler

    wf = _krea_graph()
    wf['26']['inputs'].update({'seed': 424242, 'steps': 9, 'cfg': 1.5,
                               'scheduler': 'beta', 'denoise': 0.6})
    positive, negative, latent = (wf['26']['inputs']['positive'],
                                  wf['26']['inputs']['negative'],
                                  wf['26']['inputs']['latent_image'])

    inject_krea_preset_sampler(wf, 'max')

    assert wf['krea_ps_noise']['inputs']['noise_seed'] == 424242
    assert wf['krea_ps_sigmas']['inputs']['steps'] == 9
    assert wf['krea_ps_sigmas']['inputs']['scheduler'] == 'beta'
    assert wf['krea_ps_sigmas']['inputs']['denoise'] == 0.6
    assert wf['krea_ps_guider']['inputs']['cfg'] == 1.5
    # CFGGuider, not BasicGuider: the negative prompt has to survive, or cfg is
    # inert and the negative CLIPTextEncode renders nothing.
    assert wf['krea_ps_guider']['class_type'] == 'CFGGuider'
    assert wf['krea_ps_guider']['inputs']['positive'] == positive
    assert wf['krea_ps_guider']['inputs']['negative'] == negative
    assert wf['krea_ps_run']['inputs']['latent_image'] == latent


def test_swap_after_a_lora_stack_keeps_the_whole_chain():
    """THE regression this file exists for.

    `inject_krea_preset_sampler` reads KSampler.model. Run before the LoRA
    injection it would wire the guider straight to the UNETLoader — dropping every
    LoRA, with no error and a render that merely looks wrong. Both the guider and
    the scheduler have to end up on the LAST link of the chain."""
    from app.utils.comfyui import (inject_krea_loras, inject_krea2t_enhancer,
                                   inject_krea_preset_sampler)

    wf = _krea_graph()
    requested = [{'filename': 'krea/a.safetensors', 'strength': 1.0},
                 {'filename': 'krea/b.safetensors', 'strength': 0.5}]
    assert inject_krea_loras(wf, requested, allowed={r['filename'] for r in requested}) == 2
    assert inject_krea2t_enhancer(wf, True, 1.0) == 1
    model_src = wf['26']['inputs']['model']
    assert model_src[0] != '20', 'precondition: the model no longer comes from the loader'

    inject_krea_preset_sampler(wf, 'detailed')

    assert wf['krea_ps_guider']['inputs']['model'] == model_src
    assert wf['krea_ps_sigmas']['inputs']['model'] == model_src


def test_swap_is_a_no_op_for_an_unknown_or_empty_preset():
    """Fail-safe: an unrecognised preset must not be able to half-rewire a graph.
    A value the app and the node disagree about (an install mid-update) has to
    degrade to the stock sampler, not to a broken tile."""
    from app.utils.comfyui import inject_krea_preset_sampler

    for preset in (None, '', 'Enhanced', 'nonsense', 'custom'):
        wf = _krea_graph()
        assert inject_krea_preset_sampler(wf, preset) == 0
        assert wf['26']['class_type'] == 'KSampler'
        assert not any(k.startswith('krea_ps_') for k in wf)


def test_swap_works_on_the_edit_graph_node_id():
    """The edit lane numbers its KSampler '11' and its VAEDecode '12'. The consumer
    is found by scanning for the LINK, so nothing here may be hardcoded to 26/27."""
    from app.utils.comfyui import inject_krea_preset_sampler

    wf = {
        '11': {'class_type': 'KSampler',
               'inputs': {'seed': 7, 'steps': 8, 'cfg': 1.0, 'sampler_name': 'euler',
                          'scheduler': 'simple', 'denoise': 1.0, 'model': ['7', 0],
                          'positive': ['8', 0], 'negative': ['9', 0],
                          'latent_image': ['10', 0]}},
        '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['3', 0]}},
    }
    assert inject_krea_preset_sampler(wf, 'soft', ksampler_node='11') == 1
    assert '11' not in wf
    assert wf['12']['inputs']['samples'] == ['krea_ps_run', 0]


# --- 3. The two files that must agree ----------------------------------------

def test_app_offers_exactly_the_presets_the_node_implements():
    sampler = _load_node_module()
    from app.utils.comfyui import KREA_SAMPLER_PRESETS, KREA_PRESET_SAMPLER_CLASS

    assert set(KREA_SAMPLER_PRESETS) == set(sampler.PRESETS), (
        'the app offers presets the shipped node does not implement (or vice versa)')
    # 'custom' is the node's manual mode, reachable in ComfyUI itself; the app
    # never sends it, because the graph carries no widgets for its five floats.
    assert 'custom' not in KREA_SAMPLER_PRESETS
    assert 'custom' in sampler.PRESET_NAMES

    from lds_krea_sampler import NODE_CLASS_MAPPINGS
    assert KREA_PRESET_SAMPLER_CLASS in NODE_CLASS_MAPPINGS, (
        'the class the app writes into the graph is not the one the node registers')


def test_the_helper_and_the_installer_name_the_same_thing():
    from app.services import krea_sampler_helper as ksh
    from app import setup_installer as si
    from app.utils.comfyui import KREA_PRESET_SAMPLER_CLASS

    assert ksh.KREA_SAMPLER_INSTALL_ACTION in si._BUNDLED_NODE_PACKS
    assert ksh.KREA_SAMPLER_INSTALL_ACTION in si.INSTALL_ACTIONS
    assert ksh.KREA_SAMPLER_NODE_CLASSES == (KREA_PRESET_SAMPLER_CLASS,)


def test_the_shipped_folder_is_where_the_installer_looks():
    """The one failure a user could never diagnose: the installer pointing at a
    folder the release does not carry."""
    from app import setup_installer as si

    src = si._bundled_pack_source('krea_sampler_nodes')
    assert os.path.isdir(src), src
    assert os.path.isfile(os.path.join(src, '__init__.py'))
    # backend/ is robocopied verbatim into the release ZIP; anywhere else and the
    # source would exist in a git checkout and nowhere else.
    from app import config as cfg
    assert os.path.commonpath([src, str(cfg.BACKEND_DIR)]) == str(cfg.BACKEND_DIR)


# --- 4. The installer's copy contract ----------------------------------------

@pytest.fixture
def fake_comfyui(tmp_path, monkeypatch):
    """A ComfyUI folder the installer will accept, without a real ComfyUI."""
    from app import setup_installer as si
    root = tmp_path / 'ComfyUI'
    (root / 'custom_nodes').mkdir(parents=True)
    monkeypatch.setattr(si, '_comfyui_root', lambda: str(root))
    return root


def test_deploy_copies_the_node_and_stamps_it(fake_comfyui):
    from app import setup_installer as si
    from app.version import APP_VERSION

    ok, message = si._deploy_bundled_pack('krea_sampler_nodes')
    assert ok, message
    dest = fake_comfyui / 'custom_nodes' / 'lds_krea_sampler'
    assert (dest / '__init__.py').is_file()
    assert (dest / si._BUNDLED_STAMP).read_text(encoding='utf-8') == APP_VERSION
    assert si._bundled_pack_state('krea_sampler_nodes') == 'current'
    # Idempotent: a second click must not rewrite anything.
    ok, message = si._deploy_bundled_pack('krea_sampler_nodes')
    assert ok and 'up to date' in message


def test_a_stale_copy_is_replaced_and_its_leftovers_removed(fake_comfyui):
    """An old version's file must not survive into the new copy: a stray module in
    a ComfyUI package folder is imported all the same."""
    from app import setup_installer as si

    si._deploy_bundled_pack('krea_sampler_nodes')
    dest = fake_comfyui / 'custom_nodes' / 'lds_krea_sampler'
    (dest / 'removed_in_a_later_version.py').write_text('raise SystemExit', encoding='utf-8')
    (dest / si._BUNDLED_STAMP).write_text('2000.01.01', encoding='utf-8')
    assert si._bundled_pack_state('krea_sampler_nodes') == 'stale'

    ok, _ = si._deploy_bundled_pack('krea_sampler_nodes')
    assert ok
    assert not (dest / 'removed_in_a_later_version.py').exists()
    assert si._bundled_pack_state('krea_sampler_nodes') == 'current'


def test_a_folder_we_did_not_write_is_never_deleted(fake_comfyui):
    """The installer only removes a directory it can prove it wrote. Someone's
    hand-installed copy under the same name is reported, not destroyed."""
    from app import setup_installer as si

    dest = fake_comfyui / 'custom_nodes' / 'lds_krea_sampler'
    dest.mkdir(parents=True)
    (dest / 'theirs.py').write_text('# not ours', encoding='utf-8')
    assert si._bundled_pack_state('krea_sampler_nodes') == 'foreign'

    ok, message = si._deploy_bundled_pack('krea_sampler_nodes')
    assert not ok
    assert (dest / 'theirs.py').is_file(), 'the installer deleted a folder it did not write'
    assert 'not put there by this app' in message


def test_boot_refresh_updates_a_stale_copy_but_never_installs_one(fake_comfyui):
    """The retrofit path. An absent copy stays absent — installing is the user's
    decision — while one they already chose is brought up to date on its own."""
    from app import setup_installer as si

    assert si.refresh_bundled_node_packs() == {}, 'refresh must not install unprompted'
    assert si._bundled_pack_state('krea_sampler_nodes') == 'absent'

    si._deploy_bundled_pack('krea_sampler_nodes')
    dest = fake_comfyui / 'custom_nodes' / 'lds_krea_sampler'
    (dest / si._BUNDLED_STAMP).write_text('2000.01.01', encoding='utf-8')

    assert 'krea_sampler_nodes' in si.refresh_bundled_node_packs()
    assert si._bundled_pack_state('krea_sampler_nodes') == 'current'


def test_a_truncated_install_says_so_instead_of_half_copying(fake_comfyui, monkeypatch):
    from app import setup_installer as si

    monkeypatch.setattr(si, '_bundled_pack_source',
                        lambda action: str(fake_comfyui / 'nope' / 'lds_krea_sampler'))
    ok, message = si._deploy_bundled_pack('krea_sampler_nodes')
    assert not ok
    assert 'reinstall the app files' in message
    assert not (fake_comfyui / 'custom_nodes' / 'lds_krea_sampler').exists()


# --- 5. The whole path, walked ------------------------------------------------
# Everything above tests a piece. These two walk the value from the wire name a
# browser sends to the class_type ComfyUI receives, because a threading bug — the
# knob dropped at any one of the six hand-offs between them — passes every test in
# this file up to here.

def test_the_wire_name_survives_the_settings_object(app):
    from app.services.lora_test_studio import StudioGenSettings, _sanitize_gen_knobs

    settings = StudioGenSettings.from_payload({'sampler_preset': 'detailed'})
    assert settings.sampler_preset == 'detailed'

    knobs = _sanitize_gen_knobs('krea', sampler_preset=settings.sampler_preset)
    assert knobs['sampler_preset'] == 'detailed'


def test_a_preset_from_the_payload_reaches_the_built_graph(app, monkeypatch):
    """The end-to-end assertion: `sampler_preset` in, our node class out."""
    from app.services import lora_test_studio as lts

    # The LoRA under test has to be in the family pool, or the builder rejects it
    # before it ever reaches the sampler (the path-injection guard).
    monkeypatch.setattr(lts, 'get_krea_loras',
                        lambda: [{'filename': 'krea/test.safetensors'}])
    monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
    monkeypatch.setattr(lts, 'krea_default_base', lambda: None)

    with app.app_context():
        wf = lts._build_cell_workflow(
            'local', 'krea/test.safetensors', 1.0, 'a prompt', 1, None, set(),
            train_type='krea', sampler_preset='balanced')

    classes = {n.get('class_type') for n in wf.values() if isinstance(n, dict)}
    assert 'LDSKrea2PresetSampler' in classes, (
        'the preset never reached the graph — a hand-off dropped it')
    assert 'KSampler' not in classes
    # And the preflight scans the BUILT graph, so this is also what makes a
    # missing node a 409 instead of a grid of failed tiles.
    assert 'SamplerCustomAdvanced' in classes


def test_no_preset_leaves_the_stock_graph_untouched(app, monkeypatch):
    """The default. This is what keeps the shipped node OPTIONAL: without a preset
    the graph names none of our classes, so an install that never copied the folder
    has nothing to install and nothing to fail."""
    from app.services import lora_test_studio as lts

    # The LoRA under test has to be in the family pool, or the builder rejects it
    # before it ever reaches the sampler (the path-injection guard).
    monkeypatch.setattr(lts, 'get_krea_loras',
                        lambda: [{'filename': 'krea/test.safetensors'}])
    monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
    monkeypatch.setattr(lts, 'krea_default_base', lambda: None)

    with app.app_context():
        wf = lts._build_cell_workflow(
            'local', 'krea/test.safetensors', 1.0, 'a prompt', 1, None, set(),
            train_type='krea')

    classes = {n.get('class_type') for n in wf.values() if isinstance(n, dict)}
    assert 'KSampler' in classes
    assert not any(c.startswith('LDSKrea2') for c in classes)


# --- 6. The ComfyUI-facing contract ------------------------------------------
# Everything above calls the sampling FUNCTION directly. ComfyUI never does: it
# imports the package, reads NODE_CLASS_MAPPINGS, and calls the node's `build`.
# That method was the one part of the file no test had ever executed.

def _with_stub_comfy(monkeypatch):
    """Install a minimal fake `comfy.samplers` so `build()` can run without ComfyUI.

    The stub records what KSAMPLER was handed, which is the actual contract: the
    preset table has to arrive as `extra_options`, since that is how ComfyUI feeds
    them back to the sampler function as keyword arguments. A preset dict that
    never reaches this call renders as the DEFAULTS, silently — every preset
    producing the same image, with nothing failing."""
    import types
    calls = []

    class _KSampler:
        def __init__(self, fn, extra_options=None, **kw):
            self.sampler_function = fn
            self.extra_options = extra_options or {}
            calls.append(self)

    samplers = types.ModuleType('comfy.samplers')
    samplers.KSAMPLER = _KSampler
    comfy = types.ModuleType('comfy')
    comfy.samplers = samplers
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.samplers', samplers)
    return calls


def test_the_node_registers_under_the_class_the_app_writes(monkeypatch):
    _load_node_module()
    from lds_krea_sampler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert list(NODE_CLASS_MAPPINGS) == ['LDSKrea2PresetSampler']
    # The display name is what a human sees in ComfyUI's menu, and it has to say
    # which app put the folder there — someone auditing their custom_nodes should
    # not have to open a file to find out.
    assert 'LoRA Dataset Studio' in NODE_DISPLAY_NAME_MAPPINGS['LDSKrea2PresetSampler']


def test_build_hands_each_preset_its_own_values(monkeypatch):
    sampler = _load_node_module()
    calls = _with_stub_comfy(monkeypatch)
    from lds_krea_sampler import NODE_CLASS_MAPPINGS

    node = NODE_CLASS_MAPPINGS['LDSKrea2PresetSampler']()
    seen = []
    for preset in sampler.PRESETS:
        built, settings = node.build(preset)
        assert built is calls[-1]
        assert built.sampler_function is sampler.sample_lds_krea_multistep
        assert built.extra_options['history'] == sampler.PRESETS[preset]['history']
        assert built.extra_options['terminal'] == sampler.PRESETS[preset]['terminal']
        # The echoed STRING carries what RAN, not what was asked for — the two
        # differ whenever a value is clamped.
        assert preset in settings
        seen.append(tuple(sorted(built.extra_options.items())))
    assert len(set(seen)) == len(seen), 'two presets built the same sampler'


def test_build_accepts_custom_and_clamps_what_it_is_given(monkeypatch):
    _load_node_module()
    calls = _with_stub_comfy(monkeypatch)
    from lds_krea_sampler import NODE_CLASS_MAPPINGS

    node = NODE_CLASS_MAPPINGS['LDSKrea2PresetSampler']()
    node.build('custom', history=5.0, history_from=-2.0, terminal_strength='nonsense')
    opts = calls[-1].extra_options
    assert opts['history'] == 1.0            # clamped down
    assert opts['history_from'] == 0.0       # clamped up
    assert opts['terminal_strength'] == 1.0  # unusable -> the documented fallback


def test_an_unknown_preset_still_builds_a_sampler(monkeypatch):
    """An install mid-update can send a name this node does not know. It must
    degrade to a render, not to a dead tile — the same fail-safe the app-side
    injector applies from the other direction."""
    _load_node_module()
    calls = _with_stub_comfy(monkeypatch)
    from lds_krea_sampler import NODE_CLASS_MAPPINGS

    NODE_CLASS_MAPPINGS['LDSKrea2PresetSampler']().build('a_preset_from_the_future')
    assert calls[-1].extra_options['history'] is not None


def test_the_declared_inputs_match_what_build_accepts(monkeypatch):
    """INPUT_TYPES is what ComfyUI renders AND validates against. A widget it
    declares that `build` has no parameter for is a TypeError at execution time,
    on somebody's graph."""
    import inspect
    _load_node_module()
    from lds_krea_sampler import NODE_CLASS_MAPPINGS

    node_cls = NODE_CLASS_MAPPINGS['LDSKrea2PresetSampler']
    spec = node_cls.INPUT_TYPES()
    declared = set(spec.get('required', {})) | set(spec.get('optional', {}))
    accepted = set(inspect.signature(node_cls.build).parameters) - {'self'}
    assert declared == accepted, f'declared {declared} vs accepted {accepted}'
    assert node_cls.FUNCTION == 'build'
    assert node_cls.RETURN_TYPES[0] == 'SAMPLER'
