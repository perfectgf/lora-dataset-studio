"""Preview steps & CFG are editable, per dataset, on every family (GitHub #46).

Two properties matter and they pull against each other:

  1. NOTHING STORED -> the emitted job config is byte-identical to the literals
     the seven builders used to carry. The defaults are calibrated per base
     (8 steps at CFG 1 on a distilled Krea 2 Turbo, 25/4 on the undistilled Raw,
     20/4 FLUX, 25/4 Klein and Anima, 28/6 SDXL) and moving one silently would
     change every user's previews.
  2. STORED -> the override reaches the sampler on EVERY family, not just the
     one that was tested by hand.

The second is what the feature is; the first is what makes it safe to ship. A
test that only checked the override would let a refactor quietly re-baseline the
defaults, which is the failure mode this file exists to prevent.
"""
import json

import pytest

from app.services import lora_training as lt


class _DS:
    """Minimal dataset stand-in: the sample block reads a family, a variant and
    the train_settings blob, nothing else."""

    def __init__(self, train_type='krea', variant=None, settings=None):
        self.id = 1
        self.train_type = train_type
        self.train_variant = variant
        self.train_base_model = None
        # A JSON STRING, like the column: _train_settings parses it. Handing it a
        # dict silently yields {} and every override test would pass on the
        # default it was meant to replace.
        self.train_settings = json.dumps(settings or {})
        self.kind = 'character'
        self.trigger_word = 'sks person'
        self.name = 'ds'


# (family, variant, expected steps, expected guidance) — the shipped literals.
FAMILY_DEFAULTS = [
    ('krea', 'base', 25, 4),        # Raw: undistilled
    ('krea', 'turbo', 8, 1),        # Turbo: distilled
    ('flux', None, 20, 4),
    ('flux2klein', None, 25, 4),
    ('anima', None, 25, 4),
    ('sdxl', None, 28, 6),
]


@pytest.mark.parametrize('family,variant,steps,guidance', FAMILY_DEFAULTS)
def test_no_override_keeps_the_calibrated_default(family, variant, steps, guidance):
    ds = _DS(family, variant)
    assert lt._sample_recipe_defaults(ds) == (steps, guidance)
    assert lt._sample_steps(ds, family) == steps
    assert lt._sample_guidance(ds, family) == guidance


@pytest.mark.parametrize('family,variant,_s,_g', FAMILY_DEFAULTS)
def test_an_override_reaches_every_family(family, variant, _s, _g):
    ds = _DS(family, variant, {'sample_steps': 12, 'sample_guidance': 2.5})
    assert lt._sample_steps(ds, family) == 12
    assert lt._sample_guidance(ds, family) == 2.5


def test_zimage_default_follows_the_variant_recipe():
    """Z-Image resolves through zimage_training_recipe, so its default moves with
    the variant exactly like the recipe the launch will use."""
    for variant in ('turbo', 'base', 'deturbo'):
        ds = _DS('zimage', variant)
        recipe = lt.zimage_training_recipe(variant, None)
        assert lt._sample_recipe_defaults(ds) == (recipe['sample_steps'],
                                                  recipe['guidance_scale'])


def test_an_impossible_zimage_pair_still_renders_a_payload(monkeypatch):
    """A settings PAYLOAD must not explode on a variant/base combination the
    launch path would refuse — the panel has to render so the user can FIX it."""
    def _boom(*a, **k):
        raise ValueError('unusable combination')
    monkeypatch.setattr(lt, 'zimage_training_recipe', _boom)
    steps, guidance = lt._sample_recipe_defaults(_DS('zimage', 'turbo'))
    assert isinstance(steps, int) and steps > 0
    assert guidance


@pytest.mark.parametrize('bad', [0, 61, -1, 'ten', 2.5, True, None])
def test_out_of_range_steps_are_ignored_not_applied(bad):
    """An unusable stored value falls back to the family default instead of
    reaching ai-toolkit: a config is built long after the box was typed in, and
    a run that dies at the first preview costs a rented GPU hour."""
    ds = _DS('krea', 'base', {'sample_steps': bad})
    assert lt._sample_steps(ds, 'krea') == 25


@pytest.mark.parametrize('bad', [0, 0.5, 21, -3, 'four', True, None])
def test_out_of_range_guidance_is_ignored_not_applied(bad):
    ds = _DS('krea', 'base', {'sample_guidance': bad})
    assert lt._sample_guidance(ds, 'krea') == 4


def test_an_integral_override_stays_an_int():
    """4.0 and 4 must produce the same config text. ai-toolkit does not care, but
    a diff of two configs, a stored provenance snapshot and every byte-identity
    test in this suite do."""
    ds = _DS('krea', 'base', {'sample_guidance': 4.0})
    value = lt._sample_guidance(ds, 'krea')
    assert value == 4 and isinstance(value, int)


def test_both_keys_are_accepted_by_the_settings_validator():
    """They must be in TRAIN_SETTING_KEYS or a preset carrying them is silently
    dropped — presets are schema-tolerant by design, so an unknown key is not an
    error, it is a disappearance."""
    assert 'sample_steps' in lt.TRAIN_SETTING_KEYS
    assert 'sample_guidance' in lt.TRAIN_SETTING_KEYS


def test_a_resume_may_change_them_including_full_state():
    """They render a picture; they do not shape a trajectory. The full-state
    refusal list names the cadence and the learning rate, and must NOT grow these
    two — that would make a resume the one place an unreadable preview cannot be
    fixed."""
    assert 'sample_steps' in lt.RESUME_SAFE_SETTING_KEYS
    assert 'sample_guidance' in lt.RESUME_SAFE_SETTING_KEYS
    patch = lt.validate_resume_overrides({'sample_steps': 30, 'sample_guidance': 3.5})
    assert patch == {'sample_steps': 30, 'sample_guidance': 3.5}


def test_a_resume_refuses_an_out_of_range_value_loudly():
    with pytest.raises(ValueError, match='sample_steps'):
        lt.validate_resume_overrides({'sample_steps': 999})
    with pytest.raises(ValueError, match='sample_guidance'):
        lt.validate_resume_overrides({'sample_guidance': 99})


def test_auto_clears_the_override_on_a_resume():
    """'auto' has to travel the same path as a blank box in the panel, or the two
    surfaces disagree about how you go back to the default."""
    assert lt.validate_resume_overrides({'sample_steps': 'auto'}) == {'sample_steps': None}
    assert lt.validate_resume_overrides({'sample_guidance': None}) == {'sample_guidance': None}


# --- End to end: the override has to reach the EMITTED config ----------------
# The resolvers above can be perfectly right while a builder still carries its
# own literal — that is the bug this whole change removes, so it gets a test
# that reads the real job config rather than the helper feeding it.

_FAMILY_JOB_CASES = [
    ('zimage', None), ('krea', 'base'), ('krea', 'turbo'),
    ('flux', None), ('flux2klein', None), ('anima', None), ('sdxl', None),
]


def _sample_block(app, tmp_path, family, variant, settings):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app import config as cfg
    import json as _json
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, f'D-{family}-{variant}', 'zsub')
        for _ in range(20):
            svc.db.session.add(
                FaceDatasetImage(dataset_id=ds.id, status='keep', filename='x.webp'))
        ds.train_type = family
        ds.train_variant = variant
        if family == 'sdxl':
            # SDXL has no bundled base: the builder refuses without one. An
            # absolute path is the opt-in custom-weights form (the launch
            # preflight owns the file check, not the config builder).
            ds.train_base_model = str(tmp_path / 'sdxl-base.safetensors')
        ds.train_settings = _json.dumps(settings)
        svc.db.session.commit()
        job = lt.build_job_config(ds, str(tmp_path), steps=1000)
        return job['config']['process'][0]['sample']


@pytest.mark.parametrize('family,variant', _FAMILY_JOB_CASES)
def test_the_override_reaches_the_emitted_sample_block(app, tmp_path, family, variant):
    block = _sample_block(app, tmp_path, family, variant,
                          {'sample_steps': 17, 'sample_guidance': 2.5})
    assert block['sample_steps'] == 17
    assert block['guidance_scale'] == 2.5


@pytest.mark.parametrize('family,variant', _FAMILY_JOB_CASES)
def test_without_an_override_the_emitted_block_is_the_shipped_default(
        app, tmp_path, family, variant):
    """The reproducibility half: same family, nothing stored, same two numbers as
    before this feature existed."""
    block = _sample_block(app, tmp_path, family, variant, {})
    ds = _DS(family, variant)
    steps, guidance = lt._sample_recipe_defaults(ds)
    assert block['sample_steps'] == steps
    assert block['guidance_scale'] == guidance
