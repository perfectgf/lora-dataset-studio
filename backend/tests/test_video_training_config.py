"""The video branch of the ai-toolkit config generator, and the checkpoint
harvest that has to survive what a Wan 2.2 MoE writes to disk.

WHY THIS FILE EXISTS AT ALL, AND WHAT IT IS ANCHORED TO
-------------------------------------------------------
An end-to-end smoke run has been through this: a dataset built by the video Bank
was handed to a local ai-toolkit and trained to decreasing loss. Every non-obvious
value asserted below is a value that run carried, not a value that looked sensible
in review. Where a number has no such provenance, the code refuses instead of
guessing, and there is a test for the refusal.

These tests are PURE — no ffmpeg, no PyAV, no torch, no GPU. The config generator
is arithmetic over a catalogue row and a database row, and the harvest is string
work over filenames, so both stay green on an install with none of the video
extras. That is a deliberate constraint on the code, not a convenience here.

THE TWO CLAIMS UNDER TEST
-------------------------
1. The emitted config says what the dataset actually contains. The clips on disk
   were encoded at one frame count and one size; a config that disagrees does not
   fail — ai-toolkit's sampler evenly spreads whatever it is asked for across the
   whole clip and reports nothing, so a wrong `num_frames` is a silent time-warp.
2. A Wan 2.2 14B checkpoint is TWO files. `save_lora` splits the state dict on
   `.transformer_1.` / `.transformer_2.` and writes `<name>_high_noise.safetensors`
   and `<name>_low_noise.safetensors`, because `split_multistage_loras` defaults to
   True. Every step number in this app was parsed by a regex anchored on
   `_<digits>.safetensors$` — an anchor those names break.
"""
import os

import pytest

from app.services import video_targets as vt
from app.services import video_training as vtrain


class _VideoDS:
    """The columns of `video_dataset` the config generator reads, and nothing
    else. A stand-in rather than a real row on purpose: the generator must not
    need a database, so that a cloud launcher can build a config from a snapshot."""

    def __init__(self, name='holiday clips', target_profile='wan22_14b',
                 frames=81, fps=16, width=384, height=384,
                 output_dir='/data/video_datasets/1', id=1):
        self.id = id
        self.name = name
        self.target_profile = target_profile
        self.frames = frames
        self.fps = fps
        self.width = width
        self.height = height
        self.output_dir = output_dir


def _proc(cfg):
    return cfg['config']['process'][0]


# --- the config says what the dataset contains --------------------------------

def test_num_frames_is_the_datasets_own_frame_count():
    """THE claim the smoke run was built to test. ai-toolkit's
    `shrink_video_to_frames` defaults to True and spreads `num_frames` evenly over
    the WHOLE clip, consulting neither the source fps nor the dataset's. Asking for
    a count the clips were not cut to therefore yields slow motion or a hyperlapse,
    silently. Emitting the dataset's own count makes that resampling a no-op."""
    cfg = vtrain.build_job_config(_VideoDS(frames=81), '/pod/ds', 1000)
    assert _proc(cfg)['datasets'][0]['num_frames'] == 81


def test_the_arch_string_comes_from_the_catalogue():
    """`aitk_arch` is the catalogue's whole reason to carry that field: our profile
    key is not ai-toolkit's arch (our `wan22_ti2v5b` is its `wan22_5b`). A literal
    here would be right for one profile and wrong for the next."""
    for key in ('wan21', 'wan22_14b'):
        cfg = vtrain.build_job_config(
            _VideoDS(target_profile=key, frames=81), '/pod/ds', 100)
        assert _proc(cfg)['model']['arch'] == vt.get(key)['aitk_arch']


def test_the_dataset_folder_is_used_verbatim():
    """The video dataset is ALREADY a flat folder of .mp4 files with homonym .txt
    captions — the shape ai-toolkit wants. There is no export step to re-run, so
    the caller's folder (staging locally, the pod path after cloudification) is
    passed straight through."""
    cfg = vtrain.build_job_config(_VideoDS(), '/pod/datasets/job7', 100)
    assert _proc(cfg)['datasets'][0]['folder_path'] == '/pod/datasets/job7'
    assert _proc(cfg)['datasets'][0]['caption_ext'] == 'txt'


def test_training_folder_is_the_cloud_seam():
    """Same seam as every image family: the caller supplies the folder so a cloud
    launch can name the pod's TRAINING_FOLDER without a local ai-toolkit install
    existing at all. Omitted, it must not invent a path silently."""
    cfg = vtrain.build_job_config(_VideoDS(), '/pod/ds', 100,
                                  training_folder='/workspace/out')
    assert _proc(cfg)['training_folder'] == '/workspace/out'
    # Omitted, it falls back to ai-toolkit's own relative default — never to the
    # caller's None, which would put a literal null in the config and blow up
    # inside the trainer instead of here.
    assert _proc(vtrain.build_job_config(
        _VideoDS(), '/pod/ds', 100))['training_folder'] == 'output'


# --- the MoE-only knobs -------------------------------------------------------

def test_moe_targets_carry_switch_boundary_every():
    """Wan 2.2 14B is a two-expert MoE and `switch_boundary_every` is what makes
    the trainer alternate between them. The smoke run carried 10, which is also
    what ai-toolkit's own shipped 24 GB example carries."""
    for key in ('wan22_14b', 'wan22_14b_i2v'):
        cfg = vtrain.build_job_config(
            _VideoDS(target_profile=key, frames=81), '/pod/ds', 100,
            base_model='org/repo')
        assert _proc(cfg)['train']['switch_boundary_every'] == 10
        assert _proc(cfg)['model']['model_kwargs'] == {
            'train_high_noise': True, 'train_low_noise': True}


def test_non_moe_targets_carry_no_switch_boundary_at_all():
    """Not "carries 0" — absent. Wan 2.1 has one transformer, and a boundary key on
    a single-stage arch is at best ignored and at worst read as a real setting."""
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='wan21', frames=81), '/pod/ds', 100)
    assert 'switch_boundary_every' not in _proc(cfg)['train']
    assert 'model_kwargs' not in _proc(cfg)['model']


# --- the two ways to fit, and the one that eats the captions ------------------

def test_captions_are_cached_and_the_text_encoder_is_never_unloaded():
    """ai-toolkit offers two routes to 24 GB and they are NOT interchangeable.
    `unload_text_encoder` encodes the TRIGGER WORD ONLY and applies that one
    embedding to every clip — every caption in the dataset is discarded, in
    silence. `cache_text_embeddings` pre-encodes the real captions instead. A
    dataset lane whose entire point is captioned clips can only use the second,
    so its absence is a bug and the presence of the first is a worse one."""
    train = _proc(vtrain.build_job_config(_VideoDS(), '/pod/ds', 100))['train']
    assert train['cache_text_embeddings'] is True
    assert 'unload_text_encoder' not in train


def test_video_sampling_is_disabled_by_default():
    """A video preview is minutes of GPU per prompt, on a rented pod, and proves
    nothing about the dataset. Default off; a caller that wants previews asks."""
    assert _proc(vtrain.build_job_config(_VideoDS(), '/pod/ds', 100))[
        'train']['disable_sampling'] is True


def test_sampling_can_be_asked_for_and_then_matches_the_dataset():
    """When previews ARE wanted they must animate at the dataset's own rate and
    length — a preview rendered at another fps is not a preview of this LoRA."""
    cfg = vtrain.build_job_config(_VideoDS(frames=81, fps=16), '/pod/ds', 100,
                                  sample_prompts=['a person walking'])
    assert _proc(cfg)['train']['disable_sampling'] is False
    assert _proc(cfg)['sample']['num_frames'] == 81
    assert _proc(cfg)['sample']['fps'] == 16
    assert _proc(cfg)['sample']['prompts'] == ['a person walking']


# --- resolution: one scalar for a rectangle -----------------------------------

def test_resolution_preserves_the_clips_pixel_budget():
    """ai-toolkit's `resolution` is not a width. `get_bucket_for_image_size` reads
    it as a pixel CAP (`max_pixels = resolution ** 2`) and rescales the clip's own
    aspect ratio under it. So the faithful scalar for a w x h clip is the geometric
    mean sqrt(w*h): it makes the cap equal the clip's own pixel count, and the
    scaler comes out at exactly 1.0 — no downscale, and no upscale of data that
    does not exist. A 832x480 clip is 399 360 pixels, sqrt = 631.9, floored to the
    profile's multiple of 16 = 624. Emitting max(w, h) = 832 would ask for
    692 224 pixels and quietly upscale every clip."""
    cfg = vtrain.build_job_config(
        _VideoDS(width=832, height=480), '/pod/ds', 100)
    assert _proc(cfg)['datasets'][0]['resolution'] == [624]


def test_square_clips_resolve_to_their_own_side():
    """The smoke run's own case: 384x384 clips, resolution [384]. Any formula that
    does not return the side for a square is wrong before the rectangles matter."""
    cfg = vtrain.build_job_config(_VideoDS(width=384, height=384), '/pod/ds', 100)
    assert _proc(cfg)['datasets'][0]['resolution'] == [384]


def test_a_dataset_with_no_recorded_size_is_refused():
    """`width`/`height` are nullable on `video_dataset`. Defaulting to 512 would
    silently retrain a 1280x704 dataset at a quarter of its pixels."""
    with pytest.raises(vtrain.VideoTrainingUnsupported):
        vtrain.build_job_config(_VideoDS(width=None, height=None), '/pod/ds', 100)


# --- refusals -----------------------------------------------------------------

def test_the_generic_profile_is_refused_because_it_has_no_arch():
    """`generic` is the catalogue's escape hatch for an uncatalogued target and its
    `aitk_arch` is None ON PURPOSE. There is no config to write for it, and the
    failure has to name that — not emit `arch: None` for the pod to choke on after
    the GPU has been rented."""
    assert vt.get('generic')['aitk_arch'] is None
    with pytest.raises(vtrain.VideoTrainingUnsupported) as e:
        vtrain.build_job_config(_VideoDS(target_profile='generic'), '/pod/ds', 100)
    assert 'generic' in str(e.value)


def test_an_unknown_profile_is_refused():
    """A profile key from a database this build has never heard of (a downgrade, a
    hand-edited row). Refuse; never fall back to Wan's rules."""
    with pytest.raises(vtrain.VideoTrainingUnsupported):
        vtrain.build_job_config(_VideoDS(target_profile='nope'), '/pod/ds', 100)


def test_frames_illegal_for_the_target_are_refused():
    """The stored `frames` is what the clips were CUT to. If it violates the
    target's VAE rule the trainer will not complain: its encoder loops
    `1 + (F-1)//4` and slices, and a Python slice never raises — the trailing
    frames just vanish. This is the only place it can be caught."""
    with pytest.raises(vtrain.VideoTrainingUnsupported):
        vtrain.build_job_config(_VideoDS(frames=64), '/pod/ds', 100)


def test_an_arch_with_no_verified_base_repo_demands_one():
    """Only two video bases are stated by an ai-toolkit config shipped ON THIS
    machine: Wan 2.1's and Wan 2.2 14B's. Inventing a repository id for LTX or
    MiniMax would dress a guess as a measurement and fail after the pod is paid
    for. Refuse — unless the caller names the base itself."""
    with pytest.raises(vtrain.VideoTrainingUnsupported) as e:
        vtrain.build_job_config(
            _VideoDS(target_profile='ltx23', frames=81), '/pod/ds', 100)
    assert 'base' in str(e.value).lower()
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='ltx23', frames=81), '/pod/ds', 100,
        base_model='Lightricks/LTX-Video-2.3')
    assert _proc(cfg)['model']['name_or_path'] == 'Lightricks/LTX-Video-2.3'


def test_the_wan22_base_and_recovery_adapter_are_the_ones_that_trained():
    """Verbatim from the config that reached decreasing loss. The 4-bit quantised
    path is only usable because of the accuracy-recovery adapter pinned in `qtype`;
    dropping the adapter and keeping uint4 is a different, worse run."""
    model = _proc(vtrain.build_job_config(_VideoDS(), '/pod/ds', 100))['model']
    assert model['name_or_path'] == 'ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16'
    assert model['qtype'] == ('uint4|ostris/accuracy_recovery_adapters/'
                              'wan22_14b_t2i_torchao_uint4.safetensors')
    assert model['quantize'] is True and model['quantize_te'] is True


def test_low_vram_can_be_turned_off_for_a_big_cloud_gpu():
    """The measured reason to run this in the cloud at all: `low_vram` filled 24 GB
    and cost 170-185 s per step at 81 frames, because every unused expert shuttles
    over PCIe. An 80 GB pod does not need it, and the config must be able to say so
    — with the CPU-offload flag actually gone, not merely False-ish."""
    cfg = vtrain.build_job_config(_VideoDS(), '/pod/ds', 100, low_vram=False)
    assert _proc(cfg)['model']['low_vram'] is False


# --- MiniMax H3: a second architecture, not a second set of Wan values ---------
#
# Everything asserted below was read in the ai-toolkit INSTALLED on this machine
# (HEAD 18f5810). It ships no example YAML for H3; its own job UI carries the
# preset, at ui/src/app/jobs/new/options.tsx:701-773. That file is the source for
# the repo id, the two qtypes, the timestep schedule, the audio flags and the
# guidance scale, and each is cross-checked against the trainer code where one
# exists. The point of this section is that NONE of these values is Wan's.

def test_minimax_h3_loads_the_repack_ai_toolkit_actually_reads():
    """`Comfy-Org/MiniMax-H3`, not `MiniMaxAI/MiniMax-H3`. The trainer resolves
    every weight file from the Comfy-Org repack (`COMFY_REPO`, minimax_h3.py:93)
    and uses the original repo only for the tokenizer's tiny config files. Naming
    the original as the base is the exact shape of a guess that builds fine and
    fails after the download — of tens of gigabytes."""
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24), '/pod/ds', 100)
    assert _proc(cfg)['model']['name_or_path'] == 'Comfy-Org/MiniMax-H3'


def test_minimax_h3_keeps_the_quantisation_its_checkpoints_already_carry():
    """The repack's files are ALREADY quantised: an int8-ConvRot DiT and an nvfp4
    AWQ text encoder. Asking for `qfloat8` — the value every other family here
    carries — does not fail and does not warn. It re-quantises a 21 GB transformer
    and a 16 GB encoder layer by layer, into a format the files were not saved in,
    on every single load. The qtypes must match the checkpoints exactly."""
    model = _proc(vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24),
        '/pod/ds', 100))['model']
    assert model['qtype'] == 'convrot8'
    assert model['qtype_te'] == 'nvfp4'


def test_wan_keeps_the_text_encoder_quantisation_that_trained():
    """The regression guard for the test above: making the qtypes per-arch must not
    move Wan off the values the smoke run carried."""
    model = _proc(vtrain.build_job_config(_VideoDS(), '/pod/ds', 100))['model']
    assert model['qtype_te'] == 'qfloat8'
    assert model['qtype'] == ('uint4|ostris/accuracy_recovery_adapters/'
                              'wan22_14b_t2i_torchao_uint4.safetensors')


def test_the_timestep_schedule_is_a_property_of_the_architecture():
    """Three arches, three different schedules in ai-toolkit's own presets: Wan 2.2
    `linear` (options.tsx:291 — which is where our proven value comes from), LTX
    `weighted` (:792, :818), MiniMax H3 `shift` (:729). A single literal is right
    for one of them and silently wrong for the other two — wrong here does not
    crash, it just trains against a noise schedule the model was not distilled on."""
    def sched(key, **kw):
        return _proc(vtrain.build_job_config(
            _VideoDS(target_profile=key, **kw), '/pod/ds', 100,
            base_model='org/repo'))['train']['timestep_type']
    assert sched('wan22_14b', frames=81, fps=16) == 'linear'
    assert sched('minimax_h3', frames=107, fps=24) == 'shift'
    assert sched('ltx23', frames=121, fps=24) == 'weighted'


def test_a_target_that_trains_on_audio_says_so_where_the_loader_looks():
    """H3 is a JOINT video+audio model and the audio is opt-in per dataset:
    `do_audio` defaults to False (config_modules.py:1072), so omitting it trains
    the audio branch on nothing while the config still looks complete. The loader
    then reads the track off the video file itself (dataloader_mixins.py:722) —
    which is why the exporter muxes rather than writing a sidecar."""
    proc = _proc(vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24), '/pod/ds', 100))
    assert proc['datasets'][0]['do_audio'] is True
    assert proc['train']['audio_loss_multiplier'] == 1.0
    assert vt.wants_audio('minimax_h3') is True


def test_a_silent_target_carries_no_audio_keys_at_all():
    """Not "do_audio: False" — absent. Wan has no audio branch, and an audio loss
    multiplier on a model with no audio stream is a setting a reader has to go and
    disprove."""
    proc = _proc(vtrain.build_job_config(_VideoDS(), '/pod/ds', 100))
    assert 'do_audio' not in proc['datasets'][0]
    assert 'audio_loss_multiplier' not in proc['train']


def test_the_dataset_block_states_the_rate_the_clips_were_cut_to():
    """Every video preset ai-toolkit ships sets `datasets[].fps` (Wan 16, LTX 24,
    H3 24). It is not decoration on the audio lane: the clip's waveform is stretched
    to `num_frames / dataset_config.fps` (dataloader_mixins.py:709-712), so with no
    fps the audio is fitted to the SOURCE duration instead of the training duration
    and drifts out of sync with the frames it is supposed to accompany."""
    for key, frames, fps in (('wan22_14b', 81, 16), ('minimax_h3', 107, 24)):
        proc = _proc(vtrain.build_job_config(
            _VideoDS(target_profile=key, frames=frames, fps=fps), '/pod/ds', 100))
        assert proc['datasets'][0]['fps'] == fps


def test_the_frame_count_is_never_re_read_off_the_clips():
    """`auto_frame_count` pulls as many frames as the video happens to have — and
    ai-toolkit's own H3 preset turns it ON (options.tsx:735). This lane exists to
    cut clips to an exact legal count and to state it; letting the loader recount
    would put a length nobody validated back in charge, and it also forbids any
    batch size above 1 (config_modules.py:1497)."""
    for key, frames, fps in (('wan22_14b', 81, 16), ('minimax_h3', 107, 24)):
        proc = _proc(vtrain.build_job_config(
            _VideoDS(target_profile=key, frames=frames, fps=fps), '/pod/ds', 100))
        assert proc['datasets'][0]['auto_frame_count'] is False


def test_h3_caches_its_latents_to_disk():
    """The 33B DiT leaves about two gigabytes of a 24 GB card unused. Encoding
    clips on the fly means the video VAE is resident alongside it for the whole
    run; ai-toolkit's own H3 preset caches instead (options.tsx:732), and so does
    LTX-2.3. Wan's proven run did not, and is left alone."""
    h3 = _proc(vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24), '/pod/ds', 100))
    assert h3['datasets'][0]['cache_latents_to_disk'] is True
    wan = _proc(vtrain.build_job_config(_VideoDS(), '/pod/ds', 100))
    assert 'cache_latents_to_disk' not in wan['datasets'][0]


def test_h3_previews_are_rendered_without_guidance():
    """H3 is guidance-distilled — its released sampler applies no CFG at all
    (pipeline.py, and the preset pins guidance_scale 1 / 28 steps at
    options.tsx:726-727). Wan's 3.5 is not a neutral default here: on a distilled
    model a guidance scale above 1 degrades the preview, so a caller comparing
    previews would be reading an artefact of the wrong sampler setting."""
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24), '/pod/ds', 100,
        sample_prompts=['a person walking'])
    assert _proc(cfg)['sample']['guidance_scale'] == 1
    assert _proc(cfg)['sample']['sample_steps'] == 28
    wan = vtrain.build_job_config(_VideoDS(), '/pod/ds', 100,
                                  sample_prompts=['a person walking'])
    assert _proc(wan)['sample']['guidance_scale'] == 3.5
    assert _proc(wan)['sample']['sample_steps'] == 4


def test_frames_off_the_17n_plus_5_grid_are_refused_for_h3():
    """Wan's own default length, 81, is illegal for H3 — and this is the one arch
    where the refusal saves a crash rather than a silent misread: H3's packing
    RAISES on a count that is not 17n+5 (packing.py:87), where Wan's encoder would
    have sliced the tail off in silence. 107 is on the grid and goes through."""
    with pytest.raises(vtrain.VideoTrainingUnsupported):
        vtrain.build_job_config(
            _VideoDS(target_profile='minimax_h3', frames=81, fps=24),
            '/pod/ds', 100)
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24), '/pod/ds', 100)
    assert _proc(cfg)['datasets'][0]['num_frames'] == 107


def test_h3_resolution_is_held_under_the_canvas_the_packer_accepts():
    """H3 caps the canvas AREA at 768*1344 (packing.py:34), which no multiple-of-32
    step can express: 1920x1088 is a clean multiple and still out of spec. The
    emitted `resolution` scalar IS a pixel cap (it is squared), so it has to be
    floored under the model's own — a legacy dataset promoted before the canvas
    guard existed would otherwise ask the packer for a canvas it refuses.
    sqrt(768*1344) = 1015.9, floored to a multiple of 32, is 992."""
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24,
                 width=1920, height=1088), '/pod/ds', 100)
    assert _proc(cfg)['datasets'][0]['resolution'] == [992]
    # A dataset already under the cap is untouched: 768x1344 is exactly the cap and
    # must not be clamped down a step by an off-by-one in the comparison.
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='minimax_h3', frames=107, fps=24,
                 width=768, height=1344), '/pod/ds', 100)
    assert _proc(cfg)['datasets'][0]['resolution'] == [992]


def test_an_uncapped_target_is_not_clamped_by_the_h3_cap():
    """`max_pixels` is None for every Wan profile. The clamp must read the profile,
    not a constant — a 1280x704 Wan dataset is legal at its own size."""
    cfg = vtrain.build_job_config(
        _VideoDS(target_profile='wan22_ti2v5b', frames=121, fps=24,
                 width=1280, height=704), '/pod/ds', 100, base_model='org/repo')
    assert _proc(cfg)['datasets'][0]['resolution'] == [928]


# --- the checkpoint pair ------------------------------------------------------

def test_a_multistage_save_yields_its_step_and_its_stage():
    """ai-toolkit's own comment above the split: saves are
    `LORA_MODEL_NAME_000005000.safetensors` or `LORA_MODEL_NAME.safetensors`, and
    the split rewrites `.safetensors` into `_high_noise.safetensors`. So the step
    is no longer at the end of the stem, and every consumer that anchored on
    `_<digits>.safetensors$` reads no step at all."""
    assert vtrain.split_checkpoint_name(
        'lora_holiday_000000050_high_noise.safetensors') == (50, 'high_noise')
    assert vtrain.split_checkpoint_name(
        'lora_holiday_000000050_low_noise.safetensors') == (50, 'low_noise')


def test_the_unsuffixed_final_save_of_a_pair_has_a_stage_but_no_step():
    """The FINAL save carries no step in either world. Reporting a step of None is
    what lets the caller flag it final, exactly as it does for a single-file arch."""
    assert vtrain.split_checkpoint_name(
        'lora_holiday_high_noise.safetensors') == (None, 'high_noise')


def test_single_file_saves_are_unchanged():
    """The regression guard that matters more than the feature: every image family
    writes one file, and this parser replaces the regex that served them."""
    assert vtrain.split_checkpoint_name(
        'lora_x_000001500.safetensors') == (1500, None)
    assert vtrain.split_checkpoint_name('lora_x.safetensors') == (None, None)
    # A short run of digits is not a step: the padding is 6+ everywhere, and a
    # trailing '_v3' or '_rc74' must not be mistaken for one.
    assert vtrain.split_checkpoint_name('lora_x_v3.safetensors') == (None, None)


def test_noise_stage_is_only_read_at_the_very_end():
    """A trigger word can contain anything. 'low_noise_study' as a name must not
    make every save of that dataset look like a low-noise half."""
    assert vtrain.split_checkpoint_name(
        'lora_low_noise_study_000000500.safetensors') == (500, None)


# --- the harvest, on a real run row -------------------------------------------

def _wan22_cloud_run(dataset_id, staging, steps=100):
    """A finished Wan 2.2 cloud run whose staging holds what `save_lora` actually
    writes: a PAIR per checkpoint, plus the unsuffixed final pair."""
    import json as _json
    from app.models import CloudTrainingRun
    from app.extensions import db
    staging.mkdir(parents=True, exist_ok=True)
    for stage in ('high_noise', 'low_noise'):
        for s in (50, 100):
            (staging / f'lora_v_{s:09d}_{stage}.safetensors').write_bytes(b'W')
        (staging / f'lora_v_{stage}.safetensors').write_bytes(b'F')
    run = CloudTrainingRun(dataset_id=dataset_id, status='done', job_name='j',
                           vast_label='lds-1', staging_dir=str(staging),
                           train_params=_json.dumps({'steps': steps}))
    db.session.add(run)
    db.session.commit()
    return run


def test_the_harvest_reads_the_step_of_both_halves(app, tmp_path):
    """Before the parser existed this listed SIX saves all claiming the run's total
    step count (the `or target` fallback) and all flagged `final` (the flag was
    `not m`). "Continue from step 50" would have resumed from step 100, and the
    hub showed six identical final pills."""
    from app.services import cloud_training as ct
    with app.app_context():
        run = _wan22_cloud_run(1, tmp_path / 'stg', steps=100)
        cks = ct._run_staging_checkpoints(run)
        assert sorted(c['step'] for c in cks) == [50, 50, 100, 100, 100, 100]
        finals = sorted(c['filename'] for c in cks if c['step'] == 100
                        and '_00000' not in c['filename'])
        assert finals == ['lora_v_high_noise.safetensors',
                          'lora_v_low_noise.safetensors']


def test_both_halves_of_a_pair_survive_the_mirror(app, tmp_path):
    """The mirror rebuilds the destination name from the parsed step. Dropping the
    stage would give BOTH halves the same name, and the second copy would be
    refused as a collision — losing one expert of every checkpoint, quietly."""
    from app.services import cloud_training as ct
    run_dir = tmp_path / 'local'
    run_dir.mkdir()
    src = tmp_path / 'stg'
    src.mkdir()
    for stage in ('high_noise', 'low_noise'):
        (src / f'lora_v_000000050_{stage}.safetensors').write_bytes(b'W')
    with app.app_context():
        run = _wan22_cloud_run(1, tmp_path / 'other')
        for stage in ('high_noise', 'low_noise'):
            ct._mirror_one(run, str(run_dir), 'lora_v_rc1',
                           str(src / f'lora_v_000000050_{stage}.safetensors'))
    assert sorted(os.listdir(run_dir)) == [
        'lora_v_rc1_000000050_high_noise.safetensors',
        'lora_v_rc1_000000050_low_noise.safetensors']


# --- the cloud seam -----------------------------------------------------------

def test_a_video_config_survives_cloudification_unchanged_in_substance(app):
    """The video branch must need NO special case in `_cloudify_job_config`. That
    function swaps the staging path for the pod path on the JSON TEXT (so every
    field naming the folder is rewritten at once), retypes the legacy `sd_trainer`
    uid to the `diffusion_trainer` the pod actually runs, and repoints
    training_folder. Emitting the same shape as the image families is what buys
    that for free — and this test is what notices if the shape drifts."""
    from app.services import cloud_training as ct
    cfg = vtrain.build_job_config(_VideoDS(), 'C:/staging/vid7', 500)
    out = ct._cloudify_job_config(
        cfg, 'video_job7', 'C:/staging/vid7',
        {'DATASETS_FOLDER': '/workspace/datasets',
         'TRAINING_FOLDER': '/workspace/out'})
    proc = _proc(out)
    assert proc['type'] == 'diffusion_trainer'
    assert proc['datasets'][0]['folder_path'] == '/workspace/datasets/video_job7'
    assert proc['training_folder'] == '/workspace/out'
    # and nothing the video branch cares about was dropped on the way
    assert proc['datasets'][0]['num_frames'] == 81
    assert proc['train']['switch_boundary_every'] == 10
    assert proc['train']['cache_text_embeddings'] is True
    assert proc['model']['arch'] == 'wan22_14b'


# --- the upload ---------------------------------------------------------------

def test_the_pod_upload_ships_mp4_files():
    """The seam that would have made the whole lane a no-op: `upload_dataset`
    filters on an extension tuple that listed images and .txt only. A video
    dataset would have uploaded its captions, zero clips, and trained on an empty
    folder — after renting the GPU."""
    from app.services import aitoolkit_remote as ar
    assert '.mp4' in ar._DATA_EXTS
