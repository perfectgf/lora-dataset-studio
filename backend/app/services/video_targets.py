"""The video training-target catalogue — what each target model demands of a clip.

Every other piece of the video lane derives from this table: the cutter reads the
fps and the frame count, the length selector offers only counts that appear here,
the exporter reads the audio policy. It is pure data plus arithmetic on purpose —
no ffmpeg, no torch, no database — so it stays importable and testable on an
install that has none of the video extras.

WHERE THESE NUMBERS COME FROM, AND WHY THAT IS THE WHOLE POINT
--------------------------------------------------------------
Every value was read in the source of the ai-toolkit INSTALLED on this machine.
Not a model card, not the upstream repository on the web. Those three disagree,
and they disagree in BOTH directions:

  * A first pass researched this question on GitHub and concluded ai-toolkit could
    not train LTX. The installed version shipped an LTX trainer all along.
  * The same web sources then reported MiniMax H3 as supported, while the local
    install had no trace of it — that arch only arrived with an update.

So "trainable" here means one thing and nothing else: **this ai-toolkit has an
architecture for it**. The app drives ai-toolkit and no other trainer, so what
some trainer elsewhere supports is not a fact about this app.

`aitk_arch` carries that architecture string, and it is NOT our key: our
`wan22_ti2v5b` is ai-toolkit's `wan22_5b`. Without the field the training config
cannot be written at all.

WHY A CATALOGUE AND NOT CONSTANTS
---------------------------------
The obvious shortcut is to hard-code Wan's 16 fps and its 4n+1 rule. That is
wrong for most of this table: the rule is a property of each model's VAE. LTX
compresses time by 8, so 29 frames is legal for Wan and illegal for LTX. MiniMax
H3 wants frames congruent to 5 modulo 17.

That counter-example is worth choosing carefully, and 29 is not arbitrary: EVERY
length Wan actually offers (17, 25, 33 … 121) happens to satisfy 8n+1 as well. A
rule shared between the two looks correct on every value in the menu and only
breaks on a snapped or hand-entered one — the shape of a bug that survives review.

AND NOTHING DOWNSTREAM WILL CATCH A MISTAKE
-------------------------------------------
The trainer does not validate the frame count on the dataset path. Its VAE encoder
loops `1 + (F-1)//4` and slices; a Python slice never raises, so an off-rule count
silently loses its trailing frames. The proof is in ai-toolkit's own shipped
example configs, which set `num_frames: 40` — not a legal 4n+1 — and therefore
drop three frames from every sample. This module is the only place the rule can
be enforced.
"""

# Frame-count rules. Each maps to "is this count ingestible?", and every profile's
# offered lengths are checked against its own rule by the tests — so correcting a
# rule later forces its lengths to be corrected with it.
#   n4plus1     temporal VAE stride of 4. The 4 is a hard literal in the Wan
#               encoder loop, not a config value, so it cannot drift.
#   n8plus1     temporal VAE stride of 8 (LTX). NOT interchangeable with n4plus1:
#               every 8n+1 count is also 4n+1, so the looser rule would pass
#               illegal lengths through in silence.
#   mod17plus5  MiniMax H3. Not a plain stride — 17 pixel frames per VAE chunk
#               become 5 latents, with 3 trailing latents dropped overall.
#   any         no known constraint — accept anything positive and say nothing.
_RULES = {
    'n4plus1': lambda f: f > 0 and (f - 1) % 4 == 0,
    'n8plus1': lambda f: f > 0 and (f - 1) % 8 == 0,
    'mod17plus5': lambda f: f > 0 and f % 17 == 5,
    'any': lambda f: f > 0,
}

# The Wan family's legal lengths. Shared verbatim because they share one VAE.
_WAN_CHOICES = (17, 25, 33, 41, 49, 65, 81, 97, 121)

_TARGETS = {
    'wan21': {
        'label': 'Wan 2.1 T2V (1.3B / 14B)',
        'aitk_arch': 'wan21',
        'fps': 16,
        'frame_rule': 'n4plus1',
        'frame_choices': _WAN_CHOICES,
        'frame_default': 81,              # 80 intervals at 16 fps = exactly 5.00 s
        'size_multiple': 16,
        'max_pixels': None,
        # Deliberately empty: no size list has any local source. Inventing
        # "recommended" sizes would dress a guess as a measurement.
        'recommended_sizes': (),
        'audio': None,
        'caption_style': 'freeform',
        'dataset_layout': 'flat',
        'training_verified': True,
        'licence_note': None,
    },
    'wan21_i2v': {
        'label': 'Wan 2.1 I2V 14B',
        'aitk_arch': 'wan21_i2v',
        'fps': 16,
        'frame_rule': 'n4plus1',
        'frame_choices': _WAN_CHOICES,
        'frame_default': 81,
        'size_multiple': 16,
        'max_pixels': None,
        'recommended_sizes': (),
        'audio': None,
        'caption_style': 'freeform',
        'dataset_layout': 'flat',
        'training_verified': True,
        'licence_note': None,
    },
    'wan22_14b': {
        # NOT "Wan 2.1 / 2.2". They are two architectures resolved by an exact
        # string match, and this one is a two-expert MoE: its LoRA saves as TWO
        # files (high-noise / low-noise) and it refuses to load an existing LoRA.
        # Wan 2.1 does neither. One profile cannot describe both.
        'label': 'Wan 2.2 T2V A14B',
        'aitk_arch': 'wan22_14b',
        # 16, not 24. The "Wan 2.2 is 24 fps" error is everywhere because
        # Alibaba's own A14B card carries 720P@24fps boilerplate whose subject is
        # the 5B; the A14B card never states an fps for the A14B at all.
        'fps': 16,
        'frame_rule': 'n4plus1',
        'frame_choices': _WAN_CHOICES,
        'frame_default': 81,
        'size_multiple': 16,
        'max_pixels': None,
        'recommended_sizes': (),
        'audio': None,
        'caption_style': 'freeform',
        'dataset_layout': 'flat',
        'training_verified': True,
        'licence_note': None,
    },
    'wan22_14b_i2v': {
        'label': 'Wan 2.2 I2V A14B',
        'aitk_arch': 'wan22_14b_i2v',
        'fps': 16,
        'frame_rule': 'n4plus1',
        'frame_choices': _WAN_CHOICES,
        'frame_default': 81,
        'size_multiple': 16,
        'max_pixels': None,
        'recommended_sizes': (),
        'audio': None,
        'caption_style': 'freeform',
        'dataset_layout': 'flat',
        'training_verified': True,
        'licence_note': None,
    },
    'wan22_ti2v5b': {
        'label': 'Wan 2.2 TI2V-5B',
        'aitk_arch': 'wan22_5b',          # our key is not its arch — hence the field
        'fps': 24,
        # The SAME rule as the 14B: the temporal stride is still 4. Treating this
        # variant as "exactly 121 frames" hid every other legal length it has.
        'frame_rule': 'n4plus1',
        'frame_choices': (25, 49, 81, 97, 121),
        'frame_default': 121,             # 120 intervals at 24 fps = 5.00 s again
        # 32, not 16: this VAE compresses space by 16 and then patches 2x2. It is
        # exactly why the official 720P size here is 1280x704 — 720 is not
        # divisible by 32 — and the trap a shared "Wan 2.2" profile walks into.
        'size_multiple': 32,
        'max_pixels': None,
        'recommended_sizes': ((1280, 704), (704, 1280)),
        'audio': None,
        'caption_style': 'freeform',
        'dataset_layout': 'flat',
        'training_verified': True,
        'licence_note': None,
    },
    'ltx2': {
        'label': 'LTX-2',
        'aitk_arch': 'ltx2',
        'fps': 24,
        'frame_rule': 'n8plus1',
        'frame_choices': (25, 49, 81, 89, 121),
        'frame_default': 121,
        'size_multiple': 32,
        'max_pixels': None,
        'recommended_sizes': ((768, 768),),
        # Joint audio-video. The loader reads the track FROM THE VIDEO FILE, so a
        # sidecar .wav is invisible to it — `muxed` is not a preference, it is the
        # only shape that works. No rate is imposed, so None means "keep source".
        'audio': {'muxed': True, 'sample_rate': None, 'channels': None},
        'caption_style': 'paragraph_with_audio',
        'dataset_layout': 'flat',
        'training_verified': True,
        'licence_note': 'LTX-2 Community License — not Apache-2.0; read the terms '
                        'before publishing anything trained on it.',
    },
    'ltx23': {
        'label': 'LTX-2.3',
        'aitk_arch': 'ltx2.3',
        'fps': 24,
        'frame_rule': 'n8plus1',
        'frame_choices': (25, 49, 81, 89, 121),
        'frame_default': 121,
        'size_multiple': 32,
        'max_pixels': None,
        'recommended_sizes': ((768, 768),),
        'audio': {'muxed': True, 'sample_rate': None, 'channels': None},
        'caption_style': 'paragraph_with_audio',
        'dataset_layout': 'flat',
        'training_verified': True,
        'licence_note': 'LTX-2 Community License — not Apache-2.0; read the terms '
                        'before publishing anything trained on it.',
    },
    'minimax_h3': {
        'label': 'MiniMax H3',
        'aitk_arch': 'minimax_h3',
        'fps': 24,
        'frame_rule': 'mod17plus5',
        # 22 is on the menu because it is the shortest count anyone trains at in
        # practice — a clip shorter than the dataset's `num_frames` is simply
        # skipped, so the floor is a question about the DATA, not the model.
        # (The model's own floor is lower: ai-toolkit's snapper documents
        # "minimum 5". We do not encode 22 as a rule, only as a choice.)
        'frame_choices': (22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 209),
        # 39, and the number has a history worth stating. ai-toolkit's H3 preset
        # carries 107 AND 39: `sample.num_frames: 107` is how long a PREVIEW
        # renders, `datasets[x].num_frames: 39` is how long a training clip is.
        # This default was read off the first line and belonged to the second.
        # The difference is not cosmetic — the VAE packs 17n+5 pixel frames into
        # 5n+2 latent frames, so 107 is 32 latent frames against 39's 12, which
        # is 2.7x the rows in every single step (measured on a rented A100:
        # 21 s/step at 107). Longer clips cost TIME, not VRAM, and teach longer
        # motion; that is a trade the user can make on the promote screen, but it
        # should not be the one they get by accident.
        'frame_default': 39,
        'size_multiple': 32,
        # A step alone is not enough here: the packing code caps the canvas area.
        # 1920x1088 satisfies the multiple of 32 and is still out of spec.
        'max_pixels': 768 * 1344,
        # What the MODEL STATES, and only that — `resolution_note` reads this list
        # as "the sizes this target claims for itself" and says so to the user, so
        # a size we worked out ourselves does not belong in it however good it is.
        # (Learned by putting one here: 1024x576 dropped the note's stated shortest
        # edge from 768 to 576, and two tests caught it.)
        #
        # Worth knowing when choosing among them, though: ai-toolkit reads our
        # `resolution` scalar as a pixel CAP and re-buckets the clip under it, so a
        # size whose geometric mean is not itself a multiple of 32 is shrunk after
        # we cut it. Of the three below only 768x768 survives unchanged; the
        # maximum canvas comes back as 1312x736, 6.4% smaller. The size that would
        # round-trip at 16:9 is 1024x576 — offering it is a product decision, not a
        # catalogue fact, and it is filed as one.
        'recommended_sizes': ((1344, 768), (768, 1344), (768, 768)),
        # 32 kHz stereo, from the audio VAE's own constants. "Keep the audio" is
        # not enough: a 44.1 kHz mono source would ride through untouched.
        'audio': {'muxed': True, 'sample_rate': 32000, 'channels': 2},
        'caption_style': 'paragraph_with_audio',
        'dataset_layout': 'flat',
        'training_verified': True,
        # NOT a footnote. The MiniMax H3 Community Licence grants rights SOLELY
        # within its "Applicable Territory" and names the EU, the UK, South Korea
        # and the USA as Excluded Territories. It reaches the OUTPUTS too.
        # The note carries the exit as well as the wall: MiniMax grants the
        # excluded territories through a (free) authorization form, confirmed by
        # their staff on the model's licence thread. A warning that names no way
        # out reads as "never", and "never" here is simply wrong.
        'licence_note': 'MiniMax H3 Community License grants NO rights in the EU, '
                        'UK, South Korea or USA — and the restriction covers the '
                        'outputs, not just the model. Check your territory first; '
                        'MiniMax grants authorization on request '
                        '(platform.minimax.io/h3-license).',
    },
    'generic': {
        'label': 'Generic / other',
        # The escape hatch for a target we have not catalogued. It must impose
        # nothing — no arch, no fps, no rule, no lengths — rather than quietly
        # apply Wan's.
        'aitk_arch': None,
        'fps': None,
        'frame_rule': 'any',
        'frame_choices': (),
        'frame_default': None,
        'size_multiple': None,
        'max_pixels': None,
        'recommended_sizes': (),
        'audio': None,
        'caption_style': 'freeform',
        'dataset_layout': 'flat',
        'training_verified': False,
        'licence_note': None,
    },
}

# Stable, ordered, and STORED IN USER DATABASES (VideoDataset.target_profile).
# Renaming a key orphans every dataset that carries it, so a rename needs an alias
# path. Adding one is safe.
PROFILE_KEYS = tuple(_TARGETS)


def get(key):
    """The profile dict for `key`, or None if the key is unknown.

    Returns a copy: the catalogue is module-level state and a caller that mutates
    what it reads would change the rules for every later reader in the process.
    """
    profile = _TARGETS.get(key)
    return dict(profile) if profile is not None else None


def frame_choices(key):
    """The clip lengths, in frames, this profile can offer. Empty when we have no
    verified lengths — which the caller must render as "no presets", never as
    "any length is fine"."""
    profile = _TARGETS.get(key)
    return profile['frame_choices'] if profile else ()


def is_legal_frames(key, frames):
    """Can this target's VAE ingest a clip of exactly `frames` frames?

    False for an unknown profile: refusing is the safe answer when we cannot say.
    """
    profile = _TARGETS.get(key)
    if profile is None:
        return False
    return _RULES[profile['frame_rule']](frames)


def snap_frames(key, requested):
    """Move a requested clip length to the nearest length this target accepts.

    This is what keeps "about four seconds" from becoming 64 frames. Nothing
    downstream objects to 64: the VAE floors it in latent space and a Python slice
    never raises. ai-toolkit's own shipped examples use `num_frames: 40`, which is
    not 4n+1 and quietly drops three frames from every sample — so the app is the
    only place the rule is enforced at all.

    With no offered lengths (an uncatalogued target) the request passes through
    untouched — we have no grounds to move it.

    Ties break DOWNWARD. A shorter clip is a smaller latent cache and a faster
    step, and a request landing exactly between two lengths expressed a preference
    for neither.
    """
    choices = frame_choices(key)
    if not choices:
        return requested
    return min(choices, key=lambda c: (abs(c - requested), c))


def clip_seconds(key, frames):
    """How long a clip of `frames` frames lasts at the TARGET's fps. None when the
    profile declares no fps.

    (frames - 1) / fps, because N frames span N-1 intervals. The cross-check that
    this is right: BOTH Wan variants land on exactly 5.00 s at their own rate —
    81 at 16, and 121 at 24 — and so does LTX at 121/24.

    THIS IS THE LOAD-BEARING FUNCTION OF THE LANE, and not for arithmetic reasons.
    ai-toolkit's `shrink_video_to_frames` defaults to TRUE and spreads num_frames
    evenly across the WHOLE clip, consulting neither the source's fps nor the
    dataset's. A 2-second clip and a 90-second clip both become num_frames frames:
    one in slow motion, one a hyperlapse, neither reported. Emitting clips of
    exactly this duration is what turns that resampling into a no-op.
    """
    profile = _TARGETS.get(key)
    if profile is None or not profile['fps']:
        return None
    return (frames - 1) / profile['fps']


def validate_resolution(key, width, height):
    """Is width x height acceptable for this target?

    A STEP, not a whitelist: the bucket code floors to the multiple under a pixel
    cap, and there is no size whitelist anywhere in the trainer. Encoding the
    official inference lists as limits would refuse perfectly trainable data.

    Some targets also cap the total AREA, which a step alone cannot express —
    1920x1088 is a clean multiple of 32 and still out of spec for MiniMax H3.
    Unknown profile → False.
    """
    profile = _TARGETS.get(key)
    if profile is None or width <= 0 or height <= 0:
        return False
    step = profile['size_multiple']
    if step and (width % step or height % step):
        return False
    cap = profile['max_pixels']
    return not (cap and width * height > cap)


def wants_audio(key):
    """Does this target train on the clip's audio track? The one question the
    exporter needs; the rate and channel count come from `get(key)['audio']`.
    False for an unknown profile."""
    profile = _TARGETS.get(key)
    return bool(profile and profile['audio'])
