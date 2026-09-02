"""The Video Test Studio's graph builder, and the traps it was written around.

Every assertion here corresponds to a failure that this pipeline actually had —
in the video generation stack this lane was ported from — and that produces NO
error when it comes back. That is the whole reason they are worth a test file:
a sparse graft that silently falls back to dense, a LoRA whose keys match
nothing, a scheduler left reading an unpatched model — none of them raise, none
of them turn a job red. They render a clip that answers a different question.

The builder is pure, so the entire option matrix is covered without a GPU, a
weight, or ComfyUI.
"""
import json
import os

import pytest

from app.services import video_targets
from app.services import video_test_studio as vts


def build(**kw):
    kw.setdefault('prompt', 'she turns her head')
    kw.setdefault('image', 'start.png')
    return vts.build_workflow(**kw)


def readers_of(wf, node_id):
    """Every node whose `model` input reads `node_id`'s output."""
    return sorted(nid for nid, node in wf.items()
                  if (node.get('inputs') or {}).get('model') == [node_id, 0])


def model_chain(wf, ref):
    """The node ids a model reference passes through, head first.

    A graft is correct when the model that REACHES a consumer carries every
    patch, not when the consumer names one particular node — Sage sits on top
    of the LoRA, so `['600', 0]` and `['610', 0]` can both be right. Walking the
    chain is what tells the two apart from a hard-coded id that skips a patch.
    """
    seen = []
    while isinstance(ref, list) and ref and ref[0] in wf:
        nid = ref[0]
        if nid in seen:                     # a cycle would hang the walk
            break
        seen.append(nid)
        ref = (wf[nid].get('inputs') or {}).get('model')
    return seen


# --- the base graph ---------------------------------------------------------

def test_the_embedded_workflow_is_the_one_the_builder_names():
    assert os.path.isfile(vts.workflow_path())
    wf = vts.load_base_workflow()
    assert wf[vts.N_UNET]['inputs']['unet_name'] == vts.BASE_OFFICIAL
    assert wf[vts.N_COND]['class_type'] == 'MiniMaxH3ImageToVideo'


def test_each_build_gets_a_fresh_graph():
    """A cached base dict would carry one run's turbo nodes into the next run
    that asked for none — an option nobody ticked, applied to their clip."""
    first = build(turbo=True)['workflow']
    second = build(turbo=False)['workflow']
    assert vts.N_TURBO_LORA in first
    assert vts.N_TURBO_LORA not in second


# --- lengths, resolution, seed ---------------------------------------------

@pytest.mark.parametrize('requested', [1, 20, 40, 56, 100, 1000, 'nonsense', None])
def test_every_snapped_length_is_one_the_vae_accepts(requested):
    """Nothing downstream objects to an illegal frame count: the VAE floors it
    in latent space and no exception is ever raised. This is the only gate."""
    frames = vts.snap_frames(requested)
    assert video_targets.is_legal_frames(vts.TARGET_KEY, frames), frames
    assert vts.FRAMES_MIN <= frames <= vts.FRAMES_MAX


def test_generation_reaches_past_the_training_catalogue():
    """The catalogue stops at 209 because that is where TRAINING lengths stop
    being useful. Capping generation there would cost 6 seconds of clip for a
    reason that has nothing to do with generating."""
    assert vts.snap_frames(300) > max(video_targets.frame_choices(vts.TARGET_KEY))


def test_the_generation_default_is_NOT_the_training_one():
    """The catalogue's `frame_default` answers "how long should a TRAINING clip
    be" (39 frames, 1.6 s at 24 fps). The same ai-toolkit preset carries 107 on
    its preview line, and reading the wrong one of those two numbers has cost
    this project a wrong default before. A studio that opened on 1.6 s would
    show barely a gesture."""
    training_default = video_targets.get(vts.TARGET_KEY)['frame_default']
    assert vts.FRAMES_DEFAULT != training_default
    assert vts.build_workflow(prompt='p', image='a.png')['frames'] == vts.FRAMES_DEFAULT
    assert video_targets.is_legal_frames(vts.TARGET_KEY, vts.FRAMES_DEFAULT)


def test_megapixels_are_clamped_to_the_models_range():
    assert vts.clamp_megapixels(99) == vts.MP_MAX
    assert vts.clamp_megapixels(-1) == vts.MP_MIN
    assert vts.clamp_megapixels('nonsense') == vts.MP_DEFAULT


def test_an_absent_seed_is_random_not_the_templates_42():
    """The template ships `noise_seed: 42` hard-coded. A graph that never writes
    node 15 renders the SAME clip from the same prompt every time, which reads
    as a stuck model rather than as a seed that never moved."""
    seeds = {build()['workflow'][vts.N_NOISE]['inputs']['noise_seed'] for _ in range(5)}
    assert len(seeds) == 5
    assert build(seed=1234)['workflow'][vts.N_NOISE]['inputs']['noise_seed'] == 1234


def test_a_negative_seed_means_surprise_me():
    assert build(seed=-1)['seed'] >= 0


def test_fps_comes_from_the_shared_catalogue():
    """A clip generated here and a clip cut for training must not disagree about
    what 24 fps means — so neither restates it."""
    wf = build()['workflow']
    assert wf[vts.N_CREATE_VIDEO]['inputs']['fps'] == video_targets.get(vts.TARGET_KEY)['fps']


# --- t2v --------------------------------------------------------------------

def test_t2v_unplugs_the_image_branch_and_fixes_the_canvas():
    wf = build(mode='t2v', aspect='portrait', megapixels=0.3)['workflow']
    for nid in (vts.N_LOAD_IMAGE, vts.N_SCALE, vts.N_SIZE):
        assert nid not in wf, f'{nid} still in a t2v graph'
    assert 'first_frame' not in wf[vts.N_COND]['inputs']
    w, h = wf[vts.N_COND]['inputs']['width'], wf[vts.N_COND]['inputs']['height']
    assert w % 32 == 0 and h % 32 == 0
    assert h > w, 'portrait must be taller than it is wide'


def test_i2v_keeps_the_image_branch():
    wf = build(image='shot.png')['workflow']
    assert wf[vts.N_LOAD_IMAGE]['inputs']['image'] == 'shot.png'
    assert wf[vts.N_COND]['inputs']['first_frame'] == [vts.N_SCALE, 0]


# --- turbo ------------------------------------------------------------------

def test_turbo_moves_BOTH_model_readers_onto_the_patched_model():
    """The guider path and the scheduler both read the model. Moving one leaves
    the sigmas computed from an unpatched model while sampling runs on the
    patched one — a graph that runs, and renders mush."""
    wf = build(turbo=True)['workflow']
    assert wf[vts.N_TURBO_LORA]['class_type'] == 'MiniMaxH3TurboLoRA'
    assert wf[vts.N_SAGE]['inputs']['model'] == [vts.N_TURBO_LORA, 0]
    assert wf[vts.N_SCHEDULER]['inputs']['model'] == [vts.N_TURBO_LORA, 0]


def test_turbo_brings_its_own_double_clock_sampler():
    """Video and audio denoise on different schedules. A single-calendar sampler
    over-samples the audio at four steps and audibly breaks it."""
    wf = build(turbo=True)['workflow']
    assert wf[vts.N_TURBO_SAMPLER]['class_type'] == 'MiniMaxH3TurboSampler'
    assert wf[vts.N_SAMPLER]['inputs']['sampler'] == [vts.N_TURBO_SAMPLER, 0]


def test_turbo_sets_six_steps_and_an_explicit_count_still_wins():
    assert build(turbo=True)['steps'] == vts.TURBO_STEPS
    assert build(turbo=True, steps=4)['steps'] == 4
    assert build(turbo=True, steps=999)['steps'] == 40      # clamped, not obeyed


# --- the LoRA under test ----------------------------------------------------

def test_the_tested_lora_uses_the_STANDARD_loader():
    """ai-toolkit LoRAs ship `diffusion_model.`-prefixed keys. The turbo node
    re-prefixes them, producing `diffusion_model.diffusion_model.blocks…` — a
    name that matches nothing and is dropped key by key WITHOUT a word."""
    wf = build(lora='h3/lds/x.safetensors')['workflow']
    assert wf[vts.N_TEST_LORA]['class_type'] == 'LoraLoaderModelOnly'
    assert wf[vts.N_TEST_LORA]['inputs']['lora_name'] == 'h3/lds/x.safetensors'


def test_the_lora_chains_after_the_turbo_and_takes_over_both_readers():
    wf = build(turbo=True, lora='h3/lds/x.safetensors', lora_strength=1.3)['workflow']
    assert wf[vts.N_TEST_LORA]['inputs']['model'] == [vts.N_TURBO_LORA, 0]
    assert wf[vts.N_TEST_LORA]['inputs']['strength_model'] == 1.3
    assert wf[vts.N_SAGE]['inputs']['model'] == [vts.N_TEST_LORA, 0]
    assert wf[vts.N_SCHEDULER]['inputs']['model'] == [vts.N_TEST_LORA, 0]


def test_without_turbo_the_lora_chains_straight_off_the_unet():
    wf = build(lora='h3/lds/x.safetensors')['workflow']
    assert wf[vts.N_TEST_LORA]['inputs']['model'] == [vts.N_UNET, 0]


def test_lora_strength_is_bounded():
    """Past ±2 a rank 8-16 LoRA destroys the shot before it expresses anything."""
    wf = build(lora='x.safetensors', lora_strength=50)['workflow']
    assert wf[vts.N_TEST_LORA]['inputs']['strength_model'] == 2.0
    wf = build(lora='x.safetensors', lora_strength='nonsense')['workflow']
    assert wf[vts.N_TEST_LORA]['inputs']['strength_model'] == 1.0


# --- the 10Eros base --------------------------------------------------------

def test_eros_is_never_elected_in_silence():
    assert build()['base'] == vts.BASE_OFFICIAL
    assert build(eros=False, eros_on_disk=True)['base'] == vts.BASE_OFFICIAL


def test_eros_swaps_the_base_when_the_weight_is_there():
    built = build(eros=True, eros_on_disk=True)
    assert built['base'] == vts.BASE_EROS
    assert built['workflow'][vts.N_UNET]['inputs']['unet_name'] == vts.BASE_EROS


def test_eros_fails_OPEN_when_the_weight_is_not_on_disk():
    """The box can be ticked while 21.7 GB are still downloading. A graph
    ComfyUI refuses at validation is a worse answer than a clip on the official
    base with a line saying so."""
    built = build(eros=True, eros_on_disk=False)
    assert built['base'] == vts.BASE_OFFICIAL
    assert any('10Eros' in n and 'absent' in n for n in built['notes'])


# --- sparse attention -------------------------------------------------------

def test_an_unknown_sparse_level_is_off_not_a_guess():
    assert vts.normalise_sparse('agressive') == ''
    assert vts.normalise_sparse(None) == ''
    assert vts.N_SPARSE not in build(sparse='agressive')['workflow']


def test_sparse_excises_sage_from_the_chain():
    """H3-Optimizations >= 0.2.16 refuses to compose with an attention override
    it does not own: reaching the sparse node through Sage makes the pack ABANDON
    sparse, keep Sage, and log a warning. No error, no red job — the mode simply
    renders dense. Naming an explicit backend does not protect."""
    wf = build(turbo=True, sparse='default')['workflow']
    assert readers_of(wf, vts.N_SAGE) == [], 'nobody may still read Sage'
    assert vts.N_SAGE not in wf, 'Sage leaves the graph, not just the chain'
    assert wf[vts.N_SPARSE]['inputs']['backend'] == vts.SPARSE_BACKEND


def test_sparse_settings_are_the_authors_own_levels():
    for mode, expected in vts.SPARSE_PRESETS.items():
        wf = build(sparse=mode)['workflow']
        for key, value in expected.items():
            assert wf[vts.N_SPARSE]['inputs'][key] == value


def test_without_the_upscale_sparse_applies_to_the_base():
    """It is the only pass there is, and the adherence cost is the price."""
    wf = build(sparse='default')['workflow']
    assert wf[vts.N_GUIDER]['inputs']['model'] == [vts.N_SPARSE, 0]


def test_with_the_upscale_the_base_stays_dense():
    """All of the time lives in the upscale, and the base pass is where the
    prompt decides the composition — so the guider does NOT read the sparse
    node, and the sparse graft is there for the upscale to pick up."""
    wf = build(turbo=True, sparse='default', latent_upscale=True)['workflow']
    assert wf[vts.N_GUIDER]['inputs']['model'] != [vts.N_SPARSE, 0]
    assert wf[vts.N_UPSCALE]['inputs']['model'] == [vts.N_SPARSE, 0]


def test_max_is_the_one_level_that_accelerates_the_base_too():
    wf = build(turbo=True, sparse='max', latent_upscale=True)['workflow']
    assert wf[vts.N_GUIDER]['inputs']['model'] == [vts.N_SPARSE, 0]


def test_the_base_graph_runs_on_a_ComfyUI_with_no_pack_at_all():
    """The point of `sage=False`: SageAttention comes from a pack that also pulls
    pip dependencies, and the installer never pip-installs a third-party
    requirements file. A plain clip therefore has to be possible without it —
    otherwise a new user with 40 GB of correct weights still cannot render one
    frame, and nothing on screen would say why."""
    wf = build(sage=False)['workflow']
    assert vts.N_SAGE not in wf
    third_party = {'PathchSageAttentionKJ', 'MiniMaxH3TurboLoRA',
                   'MiniMaxH3TurboSampler', 'H3SparseAttentionAdvanced',
                   'MMH3UltimateUpscale', 'MMH3LatentUpscaleWithModelParams',
                   'MMH3TemporalSplitParams'}
    used = {n['class_type'] for n in wf.values()}
    assert used & third_party == set(), used & third_party
    # And it is still a complete graph: the sampler reads a model that comes
    # from the UNET, and both decoders still read the sampler.
    assert model_chain(wf, wf[vts.N_GUIDER]['inputs']['model'])[-1] == vts.N_UNET
    assert wf[vts.N_DECODE_VIDEO]['inputs']['samples'] == [vts.N_SAMPLER, 0]


def test_without_sage_every_option_still_chains_correctly():
    """Dropping a node from the middle of a chain is where an off-by-one lives."""
    for turbo in (False, True):
        for sparse in ('', 'default'):
            wf = build(sage=False, turbo=turbo, sparse=sparse,
                       lora='h3/lds/x.safetensors')['workflow']
            chain = model_chain(wf, wf[vts.N_GUIDER]['inputs']['model'])
            assert vts.N_TEST_LORA in chain, (turbo, sparse, chain)
            assert chain[-1] == vts.N_UNET, chain
            # the scheduler must be on the same patched model as the guider path
            sched = model_chain(wf, wf[vts.N_SCHEDULER]['inputs']['model'])
            assert vts.N_TEST_LORA in sched, (turbo, sparse, sched)


# --- the latent upscale -----------------------------------------------------

def test_the_upscale_gets_the_same_patched_model_the_sampler_does():
    """The chain head is read off the guider, never written as a literal: by
    this point it is Sage, a LoRA or the sparse node depending on what was
    armed. The property that matters is not WHICH id appears — Sage legitimately
    sits on top of the LoRA — but that the model reaching the upscale went
    through every patch the sampler's did."""
    wf = build(lora='h3/lds/x.safetensors', latent_upscale=True)['workflow']
    assert wf[vts.N_UPSCALE]['inputs']['model'] == wf[vts.N_GUIDER]['inputs']['model']
    assert vts.N_TEST_LORA in model_chain(wf, wf[vts.N_UPSCALE]['inputs']['model'])


def test_the_upscale_carries_the_lora_through_every_option_combination():
    """The one that would go unnoticed: an upscaled clip rendered WITHOUT the
    LoRA looks like a LoRA that learned nothing, not like a wiring mistake."""
    for turbo in (False, True):
        for sparse in ('', 'default', 'max'):
            wf = build(turbo=turbo, sparse=sparse, latent_upscale=True,
                       lora='h3/lds/x.safetensors')['workflow']
            chain = model_chain(wf, wf[vts.N_UPSCALE]['inputs']['model'])
            assert vts.N_TEST_LORA in chain, (turbo, sparse, chain)
            if turbo:
                assert vts.N_TURBO_LORA in chain, (turbo, sparse, chain)


def test_both_decoders_move_onto_the_enlarged_latent():
    """Rewiring one would leave the picture and its own audio coming from two
    different latents."""
    wf = build(latent_upscale=True)['workflow']
    assert wf[vts.N_DECODE_VIDEO]['inputs']['samples'] == [vts.N_UPSCALE, 0]
    assert wf[vts.N_DECODE_AUDIO]['inputs']['samples'] == [vts.N_UPSCALE, 0]


def test_the_upscale_never_wires_spatial_tiling():
    """Wired once, and the output was a MOSAIC — each tile had resampled the
    whole scene instead of its own portion. Tiling exists to fit in VRAM one
    does not have; the TEMPORAL split stays, it carries clip length."""
    wf = build(latent_upscale=True)['workflow']
    assert 'spatial_split_param' not in wf[vts.N_UPSCALE]['inputs']
    assert wf[vts.N_UPSCALE]['inputs']['temporal_split_param'] == [vts.N_UPSCALE_SPLIT, 0]


def test_the_upscale_target_follows_the_source_aspect():
    wf = build(latent_upscale=True, megapixels=0.3, source_ratio=16 / 9)['workflow']
    w = wf[vts.N_UPSCALE_PARAMS]['inputs']['width']
    h = wf[vts.N_UPSCALE_PARAMS]['inputs']['height']
    assert w % 32 == 0 and h % 32 == 0
    assert w > h
    # No ratio to measure (t2v, or an unreadable file) keeps the node's defaults
    # rather than inventing a shape.
    plain = build(latent_upscale=True)['workflow'][vts.N_UPSCALE_PARAMS]['inputs']
    assert (plain['width'], plain['height']) == (1280, 704)


# --- every option at once ---------------------------------------------------

def test_the_whole_matrix_produces_one_coherent_chain():
    built = build(turbo=True, eros=True, eros_on_disk=True, lora='h3/lds/x.safetensors',
                  lora_strength=1.3, sparse='max', latent_upscale=True,
                  source_ratio=1.0, frames=56, megapixels=0.6, seed=7)
    wf = built['workflow']
    # UNET(eros) -> turbo -> lora -> sparse -> guider, and the upscale on top.
    assert wf[vts.N_UNET]['inputs']['unet_name'] == vts.BASE_EROS
    assert wf[vts.N_TURBO_LORA]['inputs']['model'] == [vts.N_UNET, 0]
    assert wf[vts.N_TEST_LORA]['inputs']['model'] == [vts.N_TURBO_LORA, 0]
    assert wf[vts.N_SPARSE]['inputs']['model'] == [vts.N_TEST_LORA, 0]
    assert wf[vts.N_GUIDER]['inputs']['model'] == [vts.N_SPARSE, 0]
    assert wf[vts.N_UPSCALE]['inputs']['model'] == [vts.N_SPARSE, 0]
    assert readers_of(wf, vts.N_SAGE) == []
    # And it is still serialisable — the whole graph goes over HTTP as JSON.
    assert json.loads(json.dumps(wf))


def test_no_grafted_id_collides_with_the_base_graph():
    """A collision would overwrite a node with no message at all."""
    base = set(vts.load_base_workflow())
    grafted = {vts.N_TURBO_LORA, vts.N_TURBO_SAMPLER, vts.N_TEST_LORA,
               vts.N_UPSCALE, vts.N_UPSCALE_PARAMS, vts.N_UPSCALE_SPLIT,
               vts.N_SPARSE}
    assert base & grafted == set()


def test_the_filename_prefix_is_unique_per_clip():
    """ComfyUI's own counter restarts from zero when it restarts, so a prefix
    that only carried a user id produced repeat filenames across sessions — and
    a repeat filename is a stale clip served under a new run's name."""
    assert vts.new_prefix(1) != vts.new_prefix(1)


# ⏱ The launch advice: pure, and silent whenever it cannot know.

_ARGV = ['main.py', '--windows-standalone-build', '--listen', '127.0.0.1']
_RAM = 47.7            # what a 48 GB machine reports (psutil total, in GiB)
_VERSION = '0.30.1'


def test_advice_names_fast_disk_only_when_the_flag_is_missing_and_ram_is_short():
    out = vts.launch_advice(_ARGV, _RAM, _VERSION)
    assert out == {'flag': '--fast-disk', 'add': True, 'remove': None,
                   'ram_total_gb': 47.7, 'weights_gb': vts.H3_HOST_RAM_GB}
    # The flag present — bare or with a value, padded or not — ends the matter.
    assert vts.launch_advice(_ARGV + ['--fast-disk'], _RAM, _VERSION) is None
    assert vts.launch_advice(_ARGV + ['--fast-disk=1'], _RAM, _VERSION) is None
    assert vts.launch_advice(_ARGV + [' --fast-disk '], _RAM, _VERSION) is None
    # `--high-ram` is the opposite choice, made on purpose: silence.
    assert vts.launch_advice(_ARGV + ['--high-ram'], _RAM, _VERSION) is None
    # The figure shown is the raw GiB, rounded once, here — the card prints it as is.
    assert vts.launch_advice(_ARGV, 46.999637603759766, _VERSION)['ram_total_gb'] == 47.0


def test_the_floor_spares_a_64_gb_machine_that_reports_less_than_64():
    # psutil reports usable RAM: a 64 GB machine reads ~63.7. The floor must
    # sit under that, or the class the comment exempts would get the card.
    assert vts.launch_advice(_ARGV, 63.7, _VERSION) is None
    assert vts.launch_advice(_ARGV, vts.FAST_DISK_RAM_FLOOR_GB, _VERSION) is None
    assert vts.launch_advice(_ARGV, 128, _VERSION) is None
    assert vts.launch_advice(_ARGV, vts.FAST_DISK_RAM_FLOOR_GB - 0.1, _VERSION) is not None
    # And it still leaves the OS and the desktop room beside the weight set.
    assert vts.FAST_DISK_RAM_FLOOR_GB > vts.H3_HOST_RAM_GB + 12


@pytest.mark.parametrize('version', [None, '', 'garbage', '0.22.0', '0.3.60', 'v0.22.9+3'])
def test_a_comfyui_that_predates_the_flag_gets_no_advice(version):
    # argparse exits on an unknown flag before the server exists: telling such
    # an install to add `--fast-disk` would stop it from starting at all.
    assert vts.launch_advice(_ARGV, _RAM, version) is None
    assert vts.knows_fast_disk(version) is False


@pytest.mark.parametrize('version', ['0.23.0', 'v0.23.0', '0.30.1', '0.30.1+16', '1.0'])
def test_a_comfyui_that_knows_the_flag_is_advised(version):
    assert vts.knows_fast_disk(version) is True
    assert vts.launch_advice(_ARGV, _RAM, version)['add'] is True


def test_a_launcher_that_switches_the_dynamic_loader_off_is_told_which_flag_to_drop():
    # `--fast-disk` steers the dynamic loader only; with `--disable-dynamic-vram`
    # on the line it is inert, so the advice must name the switch — and say
    # whether the flag itself still needs adding, or is already there.
    tuned = ['main.py', '--disable-async-offload', '--disable-dynamic-vram', '--cache-classic']
    out = vts.launch_advice(tuned, _RAM, _VERSION)
    assert out == {'flag': '--fast-disk', 'add': True, 'remove': '--disable-dynamic-vram',
                   'ram_total_gb': 47.7, 'weights_gb': vts.H3_HOST_RAM_GB}
    already = vts.launch_advice(tuned + ['--fast-disk'], _RAM, _VERSION)
    assert already['remove'] == '--disable-dynamic-vram' and already['add'] is False
    # `--enable-dynamic-vram` overrides the switch (ComfyUI's own rule): the
    # loader runs, so only the flag itself can be missing.
    forced = vts.launch_advice(tuned + ['--enable-dynamic-vram'], _RAM, _VERSION)
    assert forced == {**out, 'remove': None}
    assert vts.launch_advice(tuned + ['--enable-dynamic-vram', '--fast-disk'], _RAM, _VERSION) is None
    # The deliberate choices still win over the switch.
    assert vts.launch_advice(tuned + ['--high-ram'], _RAM, _VERSION) is None
    # And the RAM floor applies either way.
    assert vts.launch_advice(tuned, 128, _VERSION) is None


@pytest.mark.parametrize('mode', ['--novram', '--highvram', '--gpu-only', '--cpu'])
def test_a_memory_mode_that_has_no_dynamic_loader_gets_no_advice(mode):
    # These turn the loader off by design; arguing with them would name a flag
    # that does nothing there — unless `--enable-dynamic-vram` forces it back on.
    assert vts.launch_advice(['main.py', mode], _RAM, _VERSION) is None
    assert vts.launch_advice(['main.py', mode, '--disable-dynamic-vram'], _RAM, _VERSION) is None
    assert vts.launch_advice(['main.py', mode, '--enable-dynamic-vram'], _RAM, _VERSION)['add'] is True


def test_advice_stays_silent_when_it_cannot_tell():
    assert vts.launch_advice(None, _RAM, _VERSION) is None          # no argv echoed
    assert vts.launch_advice([], _RAM, _VERSION) is None
    assert vts.launch_advice('--fast-disk', _RAM, _VERSION) is None  # a string is not argv
    assert vts.launch_advice(_ARGV, None, _VERSION) is None          # no RAM figure
    assert vts.launch_advice(_ARGV, 0, _VERSION) is None
    assert vts.launch_advice(_ARGV, True, _VERSION) is None          # a bool is not a size
    # Odd argv ELEMENTS never raise: a server that echoes numbers is answered, not crashed on.
    assert vts.launch_advice(['main.py', 1, None], _RAM, _VERSION)['add'] is True
