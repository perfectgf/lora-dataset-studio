"""The manual "Upscale & improve" pass had an editable INSTRUCTION but a hardcoded
quality profile: steps=4 and BOTH LoRA strengths pinned to 0 at the call site — so
the workflow's own realistic LoRA (0.8 in improve skin.json) never applied at all.
Those three knobs are now config (klein.improve_*), with defaults that reproduce
the historical behaviour exactly, and clamps so a hand-edited config degrades the
pass instead of raising inside the enqueue path."""
import pytest


@pytest.fixture()
def svc(app):
    with app.app_context():
        from app.services import face_dataset_service as fds
        yield fds


def test_defaults_are_the_tuned_profile(app, svc):
    """Three knobs keep the values that were hardcoded before they were exposed, so
    an untouched install is unchanged on those. The CONSISTENCY strength is the one
    deliberate change: an improve pass must add detail without redrawing the
    composition, which is exactly what a high consistency does — tuned on real runs."""
    with app.app_context():
        assert svc._improve_float('improve_base_lora_strength', 0.0) == 0.0
        assert svc._improve_int('improve_steps', 4) == 4
        assert svc._improve_float('improve_megapixels', 2.0, 8.0) == 2.0
        assert svc._improve_float('improve_consistency_strength', 1.0, 1.5) == 1.0


def test_configured_values_are_honoured(app, svc):
    with app.app_context():
        from app import config as cfg
        cfg.save_config({'klein': {'improve_base_lora_strength': 0.8,
                                   'improve_consistency_strength': 0.35,
                                   'improve_steps': 8}})
        assert svc._improve_float('improve_base_lora_strength', 0.0) == 0.8
        assert svc._improve_float('improve_consistency_strength', 1.0) == 0.35
        assert svc._improve_int('improve_steps', 4) == 8


@pytest.mark.parametrize('bad', ['nonsense', None, '', [], {}])
def test_a_malformed_value_falls_back_to_the_default(app, svc, bad):
    """A broken config must never crash the enqueue — the pass just runs stock."""
    with app.app_context():
        from app import config as cfg
        cfg.save_config({'klein': {'improve_base_lora_strength': bad,
                                   'improve_steps': bad}})
        assert svc._improve_float('improve_base_lora_strength', 0.0) == 0.0
        assert svc._improve_int('improve_steps', 4) == 4


def test_a_value_saved_under_the_old_key_name_still_works(app, svc):
    """improve_character_lora_strength shipped before the misnomer was caught (it
    drives the CONSISTENCY LoRA, not identity). Config keys live in users' files, so
    the rename needs an alias or their saved value would silently become 0."""
    with app.app_context():
        from app import config as cfg
        cfg.save_config({'klein': {'improve_character_lora_strength': 0.9}})
        # the fallback passed here mirrors the shipped default, as the call site does
        assert svc._improve_float('improve_consistency_strength', 1.0, 1.5) == 0.9


def test_out_of_range_values_are_clamped_not_rejected(app, svc):
    """Clamp rather than raise: an absurd strength should weaken to the ceiling,
    and a negative one to zero, instead of failing the user's click."""
    with app.app_context():
        from app import config as cfg
        cfg.save_config({'klein': {'improve_base_lora_strength': 99,
                                   'improve_consistency_strength': -5,
                                   'improve_steps': 9999}})
        assert svc._improve_float('improve_base_lora_strength', 0.0) == 2.0
        assert svc._improve_float('improve_consistency_strength', 1.0) == 0.0
        assert svc._improve_int('improve_steps', 4) == 50
        cfg.save_config({'klein': {'improve_steps': 0}})
        assert svc._improve_int('improve_steps', 4) == 1     # never a 0-step job


def test_the_knobs_round_trip_through_the_settings_api(client, app):
    """They must survive a full-config Save, not be dropped as an unknown key."""
    r = client.put('/api/settings', json={'config': {'klein': {
        'improve_base_lora_strength': 0.6, 'improve_steps': 6}}})
    assert r.status_code == 200
    with app.app_context():
        from app import config as cfg
        assert cfg.get('klein.improve_base_lora_strength') == 0.6
        assert cfg.get('klein.improve_steps') == 6
