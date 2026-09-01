"""✨ The Motion field, written or enriched by the local LLM (2026-09-01).

Both gestures are the image generator's own, on this app's existing waist: the
provider and the model are the ones already configured for the image passes.
What this file pins is the honesty of each — a proposal that never invents a
subject, an enhancement that can never cost the user their own words, and a
refusal that is a sentence rather than silence.
"""
import pytest

from app.services import video_motion_prompt as vmp


def test_the_answer_is_cleaned_into_something_a_sampler_can_read():
    """Models lead in ('Sure! Here is the prompt:'), quote themselves, and add
    a line of commentary. All of it would go straight to the sampler."""
    assert vmp._clean('Sure! Here is the prompt: she turns and walks away') \
        == 'she turns and walks away'
    assert vmp._clean('"she lifts her hand slowly"') == 'she lifts her hand slowly'
    assert vmp._clean('- she leans forward\nThis keeps your intent.') \
        == 'she leans forward'
    assert vmp._clean('') == ''


def test_suggest_refuses_in_words_rather_than_writing_nothing(app, monkeypatch):
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    with app.app_context():
        with pytest.raises(ValueError, match='start frame'):
            vmp.suggest_from_frame('')
        with pytest.raises(ValueError, match='not on this machine'):
            vmp.suggest_from_frame('never_staged.png')


def test_suggest_asks_about_MOVEMENT_and_forbids_replacing_the_subject():
    p = vmp._AUTO_PROMPT.lower()
    assert 'movement' in p and 'first frame' in p
    # The two failures this wording exists to prevent: a description of the
    # still, and a subject the frame does not show.
    assert 'never replace' in p
    assert 'no camera' in p          # the camera is the classifier's job
    assert 'preamble' in p


def test_an_unusable_answer_never_costs_the_user_their_own_prompt(app, monkeypatch):
    """The enhancer's one destructive failure mode, refused by construction:
    a model that answers nothing usable gives the original back."""
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text', lambda *a, **kw: '   ')
    with app.app_context():
        assert vmp.enhance('she turns her head') == 'she turns her head'


def test_enhance_returns_the_richer_line_and_refuses_an_empty_field(app, monkeypatch):
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda *a, **kw: 'She turns her head slowly to the left, '
                                         'hair swinging, then smiles.')
    with app.app_context():
        out = vmp.enhance('she turns')
        assert out.startswith('She turns her head slowly')
        with pytest.raises(ValueError, match='nothing to enrich'):
            vmp.enhance('  ')
        # A SHORT motion is still a motion: 'she blinks' must be enrichable.
        # The answer floor and the ask floor are two different numbers, and
        # sharing them refused perfectly good prompts (caught by this test).
        assert vmp.enhance('she blinks').startswith('She turns her head')


def test_without_a_local_model_both_say_which_one_is_missing(app, monkeypatch):
    monkeypatch.setattr(vmp, 'available', lambda: (False, 'Ollama: not running'))
    with app.app_context():
        with pytest.raises(ValueError, match='Ollama: not running'):
            vmp.enhance('she turns her head')
        with pytest.raises(ValueError, match='Ollama: not running'):
            vmp.suggest_from_frame('a.png')
