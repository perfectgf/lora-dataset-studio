"""The video training-target catalogue: what every profile must declare, and the
arithmetic that turns a user's "how long should a clip be?" into a frame count the
trainer will actually accept.

These tests are PURE — no ffmpeg, no GPU, no database. That is deliberate: the
catalogue is what every other piece of the video lane derives from, so it has to
stay green on an install that has none of the video extras.

WHERE THE NUMBERS COME FROM, AND WHY IT MATTERS. Every value here was read in the
source of the ai-toolkit INSTALLED on the machine, not from a model card and not
from the upstream repository on the web. Those three disagree, in both directions:
a first pass researched this on GitHub and concluded LTX could not be trained,
while the local install shipped an LTX trainer; then the same web source said
MiniMax H3 was supported while the local install had no trace of it. Only the
installed version decides what a machine can do.

The app drives ai-toolkit and nothing else, so "trainable" here means "this
ai-toolkit has an architecture for it" — never "some trainer somewhere does".
"""
import pytest

from app.services import video_targets as vt


# --- the catalogue's contract -------------------------------------------------

def test_profile_keys_are_stable():
    """These keys land in user databases (VideoDataset.target_profile), so renaming
    one silently orphans every dataset that carries it. Adding is safe; renaming
    needs an alias path, and this test is what makes someone notice."""
    assert vt.PROFILE_KEYS == (
        'wan21', 'wan21_i2v', 'wan22_14b', 'wan22_14b_i2v', 'wan22_ti2v5b',
        'ltx2', 'ltx23', 'minimax_h3', 'minimax_h3_ref2va', 'generic')


def test_every_profile_declares_the_full_contract():
    """A half-declared profile is worse than a missing one: the cutter would read a
    None fps and produce clips at the source's rate without saying so."""
    required = {'label', 'aitk_arch', 'fps', 'frame_rule', 'frame_choices',
                'frame_default', 'size_multiple', 'max_pixels',
                'recommended_sizes', 'audio', 'caption_style', 'dataset_layout',
                'training_verified', 'licence_note'}
    for key in vt.PROFILE_KEYS:
        assert required <= set(vt.get(key)), f'{key} is missing part of the contract'


def test_every_trainable_profile_names_the_architecture_string():
    """The catalogue key is OURS; the architecture string is ai-toolkit's, and they
    are not the same word — our `wan22_ti2v5b` is its `wan22_5b`. Without this
    field the training config simply cannot be written."""
    for key in vt.PROFILE_KEYS:
        profile = vt.get(key)
        if profile['training_verified']:
            assert profile['aitk_arch'], f'{key} claims to be trainable but names no arch'


def test_the_generic_escape_hatch_names_no_architecture():
    assert vt.get('generic')['aitk_arch'] is None
    assert vt.get('generic')['training_verified'] is False


def test_wan_2_1_and_2_2_are_different_targets_not_one():
    """They look like one family and are two architectures, resolved by an exact
    string match. Wan 2.2's 14B is a two-expert MoE whose LoRA saves as TWO files
    and which refuses to load an existing LoRA; Wan 2.1 does neither. One profile
    cannot describe both, and the old label 'Wan 2.1 / 2.2 14B' claimed it did."""
    assert vt.get('wan21')['aitk_arch'] == 'wan21'
    assert vt.get('wan22_14b')['aitk_arch'] == 'wan22_14b'
    assert vt.get('wan21')['fps'] == 16 and vt.get('wan22_14b')['fps'] == 16
    assert 'Wan 2.1' not in vt.get('wan22_14b')['label']


def test_image_to_video_is_its_own_target():
    """i2v is a distinct architecture with its own model file, not a mode of the
    t2v one. A dataset built for it is not interchangeable."""
    for key in ('wan21_i2v', 'wan22_14b_i2v'):
        assert vt.get(key)['aitk_arch'].endswith('i2v')
        assert vt.get(key)['training_verified'] is True


def test_ltx_two_and_two_three_are_separate_entries():
    """Two architectures, two UI entries, two weight paths — identical geometry,
    which is exactly why merging them is tempting and wrong: a dataset would name
    a target that cannot be resolved."""
    assert vt.get('ltx2')['aitk_arch'] == 'ltx2'
    assert vt.get('ltx23')['aitk_arch'] == 'ltx2.3'


# --- what this install can actually train -------------------------------------

def test_every_catalogued_architecture_is_trainable_here():
    """The correction that started all of this: an earlier catalogue marked LTX and
    the Wan 5B as untrainable on the strength of web research about OTHER trainers.
    Both ship an architecture in the installed ai-toolkit."""
    for key in ('wan21', 'wan21_i2v', 'wan22_14b', 'wan22_14b_i2v',
                'wan22_ti2v5b', 'ltx2', 'ltx23', 'minimax_h3'):
        assert vt.get(key)['training_verified'] is True, key


def test_training_verified_is_an_explicit_boolean_everywhere():
    for key in vt.PROFILE_KEYS:
        assert isinstance(vt.get(key)['training_verified'], bool)


def test_unknown_profile_key_returns_none():
    assert vt.get('wan27_ultra') is None


# --- frame-count rules --------------------------------------------------------

def test_every_frame_choice_obeys_its_own_declared_rule():
    """The consistency check that survives being wrong about a model: whatever rule
    a profile declares, its offered frame counts must satisfy it."""
    for key in vt.PROFILE_KEYS:
        profile = vt.get(key)
        for frames in profile['frame_choices']:
            assert vt.is_legal_frames(key, frames), (
                f'{key} offers {frames} frames, which breaks its own '
                f'{profile["frame_rule"]} rule')


def test_every_default_length_is_one_of_the_offered_lengths():
    for key in vt.PROFILE_KEYS:
        profile = vt.get(key)
        if profile['frame_choices']:
            assert profile['frame_default'] in profile['frame_choices']


def test_the_whole_wan_family_shares_the_four_n_plus_one_rule():
    """The 4 is a hard literal in the VAE's encoder loop, not a config value — the
    rule cannot drift between variants."""
    for key in ('wan21', 'wan21_i2v', 'wan22_14b', 'wan22_14b_i2v', 'wan22_ti2v5b'):
        assert vt.is_legal_frames(key, 49)
        assert not vt.is_legal_frames(key, 50)


def test_ltx_uses_a_temporal_stride_of_eight_not_four():
    """29 frames is legal for Wan and illegal for LTX. The counter-example needs
    care, and that is the point: every length Wan actually OFFERS satisfies 8n+1
    too, so a shared rule looks right on every value in the menu and only breaks on
    a snapped or hand-entered one."""
    assert vt.is_legal_frames('wan22_14b', 29)
    for key in ('ltx2', 'ltx23'):
        assert not vt.is_legal_frames(key, 29)
        assert vt.is_legal_frames(key, 49)


def test_minimax_h3_frame_counts_are_five_modulo_seventeen():
    """17 pixel frames per VAE chunk -> 5 latent frames, 3 trailing latents dropped.
    107 satisfies the rule and appears in ai-toolkit's H3 preset - as the length a
    PREVIEW renders at. The preset's training length is a different field; see
    test_the_h3_default_clip_length_is_the_one_ai_toolkit_trains_at."""
    assert vt.is_legal_frames('minimax_h3', 107)
    assert not vt.is_legal_frames('minimax_h3', 108)


def test_the_h3_default_clip_length_is_the_one_ai_toolkit_trains_at():
    """ai-toolkit's H3 preset carries two frame counts, and they mean different
    things: `sample.num_frames: 107` is how long a preview renders,
    `datasets[x].num_frames: 39` is how long a training clip is. This default was
    once read off the first and applied to the second.

    The cost of that confusion is arithmetic, not taste: the VAE packs 17n+5
    pixel frames into 5n+2 latent frames, so 107 carries 32 latent frames against
    39's 12 - 2.7x the rows in every step. Both lengths stay on the menu, because
    a longer clip teaches longer motion and that is a real trade; only the
    default moved to the one the trainer itself ships."""
    profile = vt.get('minimax_h3')
    assert profile['frame_default'] == 39
    assert 107 in profile['frame_choices']       # still offered, deliberately
    # The 2.7x, spelled out rather than asserted against a helper that does not
    # exist: 17n+5 pixel frames -> 5n+2 latent frames.
    latents = lambda f: (f - 5) // 17 * 5 + 2
    assert latents(39) == 12 and latents(107) == 32
    assert latents(107) / latents(39) > 2.5


def test_the_h3_menu_offers_the_shortest_length_people_actually_train_at():
    """22 frames is ~0.92 s, the shortest count practitioners use - a clip shorter
    than the dataset's num_frames is skipped by the loader, so the floor is a fact
    about the DATA, not the model. It is a CHOICE, never a rule: ai-toolkit's own
    snapper documents a minimum of 5, and encoding 22 as a constraint would refuse
    counts the model accepts."""
    assert 22 in vt.get('minimax_h3')['frame_choices']
    assert vt.is_legal_frames('minimax_h3', 5)   # legal, just not on the menu


def test_generic_profile_accepts_any_positive_frame_count():
    assert vt.is_legal_frames('generic', 57)
    assert not vt.is_legal_frames('generic', 0)


# --- snapping a requested length ----------------------------------------------

def test_snap_frames_returns_the_nearest_legal_count():
    """64 frames looks round and no trainer objects to it. ai-toolkit's own shipped
    example configs use num_frames: 40 — which is NOT 4n+1, so they silently drop
    three frames off every sample. Nothing validates; the app is the only place the
    rule can be enforced."""
    assert vt.snap_frames('wan22_14b', 64) == 65


def test_snap_frames_breaks_ties_downward():
    assert vt.snap_frames('wan22_14b', 57) == 49


def test_snap_frames_never_returns_an_illegal_count():
    for key in vt.PROFILE_KEYS:
        if not vt.frame_choices(key):
            continue
        for requested in range(1, 400):
            assert vt.is_legal_frames(key, vt.snap_frames(key, requested))


def test_snap_frames_on_the_generic_profile_returns_the_request_untouched():
    assert vt.snap_frames('generic', 57) == 57


# --- frames <-> seconds -------------------------------------------------------

def test_a_clip_lasts_one_frame_interval_less_than_its_frame_count_suggests():
    """81 frames is 80 INTERVALS: 5.00 s at 16 fps, which is what Wan documents.

    This is the load-bearing function of the whole lane, and not for arithmetic
    reasons: ai-toolkit's shrink_video_to_frames defaults to TRUE and spreads
    num_frames evenly across the WHOLE clip, so a 2 s clip and a 90 s clip both
    become num_frames frames — one in slow motion, one a hyperlapse. Emitting
    clips of exactly this duration is what makes that resampling a no-op."""
    assert vt.clip_seconds('wan22_14b', 81) == pytest.approx(5.0)


def test_the_five_second_design_point_holds_across_the_family():
    """121 frames at 24 fps is 5.00 s too. Different rate, same duration — a
    cross-check that the formula is (frames - 1) / fps."""
    assert vt.clip_seconds('wan22_ti2v5b', 121) == pytest.approx(5.0)
    assert vt.clip_seconds('ltx23', 121) == pytest.approx(5.0)


def test_clip_seconds_is_none_when_the_profile_has_no_fixed_fps():
    assert vt.clip_seconds('generic', 57) is None


# --- resolution ---------------------------------------------------------------

def test_the_wan_14b_grid_is_sixteen_and_the_5b_grid_is_thirty_two():
    """The 5B's VAE compresses space by 16 and then patches 2x2. It is why its
    official 720P size is 1280x704 and not 1280x720 — 720 is not divisible by 32,
    and a shared 'Wan 2.2' profile walks straight into it."""
    assert vt.validate_resolution('wan22_14b', 832, 480)
    assert vt.validate_resolution('wan22_ti2v5b', 1280, 704)
    assert not vt.validate_resolution('wan22_ti2v5b', 1280, 720)


def test_a_size_multiple_is_a_step_not_a_whitelist():
    """The bucket code floors to the multiple under a pixel cap; there is no size
    whitelist anywhere in the trainer. Encoding the official inference lists as
    limits would refuse perfectly trainable data."""
    assert vt.validate_resolution('wan22_14b', 1024, 1024)


def test_minimax_h3_also_caps_the_total_pixel_area():
    """A step alone is not enough for H3: its packing code caps the canvas at
    768*1344 px. 1920x1088 satisfies the multiple of 32 and is still out of spec."""
    assert vt.validate_resolution('minimax_h3', 1344, 768)
    assert not vt.validate_resolution('minimax_h3', 1920, 1088)


def test_a_profile_with_no_area_cap_accepts_a_large_size():
    assert vt.get('wan22_14b')['max_pixels'] is None
    assert vt.validate_resolution('wan22_14b', 1920, 1088)


def test_no_profile_accepts_a_zero_or_negative_size():
    for key in vt.PROFILE_KEYS:
        assert not vt.validate_resolution(key, 0, 480)
        assert not vt.validate_resolution(key, 848, -1)


# --- audio --------------------------------------------------------------------

def test_the_joint_audio_video_models_declare_a_track():
    """LTX-2 and MiniMax H3 train sound and picture together. Stripping the track
    teaches the model to be silent — a degradation with no error message anywhere."""
    for key in ('ltx2', 'ltx23', 'minimax_h3'):
        assert vt.get(key)['audio'] is not None


def test_minimax_h3_pins_the_sample_rate_and_the_channel_count():
    """32 kHz stereo, from its audio VAE's own constants. A blanket "keep the
    audio" is not enough: a 44.1 kHz mono source would ride through untouched and
    be resampled by the trainer, or not."""
    audio = vt.get('minimax_h3')['audio']
    assert audio['sample_rate'] == 32000
    assert audio['channels'] == 2


def test_ltx_takes_the_track_muxed_into_the_clip_at_the_source_rate():
    """The loader reads the audio FROM THE VIDEO FILE, so a sidecar .wav is
    invisible to it. No rate is imposed, so None means "keep the source's"."""
    audio = vt.get('ltx23')['audio']
    assert audio['muxed'] is True
    assert audio['sample_rate'] is None


def test_the_wan_family_declares_no_audio_at_all():
    """No Wan model overrides the audio encoder; forcing it reaches a raise."""
    for key in ('wan21', 'wan21_i2v', 'wan22_14b', 'wan22_14b_i2v', 'wan22_ti2v5b'):
        assert vt.get(key)['audio'] is None


def test_wants_audio_is_the_one_question_the_exporter_asks():
    assert vt.wants_audio('minimax_h3') is True
    assert vt.wants_audio('wan22_14b') is False
    assert vt.wants_audio('nope') is False


# --- licence ------------------------------------------------------------------

def test_a_licence_restricted_target_says_so_in_the_catalogue():
    assert vt.get('minimax_h3')['licence_note']


def test_targets_without_a_licence_restriction_carry_none():
    for key in ('wan21', 'wan22_14b', 'wan22_ti2v5b', 'generic'):
        assert vt.get(key)['licence_note'] is None
