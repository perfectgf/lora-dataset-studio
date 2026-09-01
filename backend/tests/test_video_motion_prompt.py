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


# --- instructed, and on the model the user chose (2026-09-01) -------------------

def test_the_enhancer_has_the_two_modes_the_image_generator_has():
    """The ported design: the MODEL picks between obeying an instruction about
    the motion and enriching the motion itself. Without the instruction mode a
    click on 'make her jump instead' would decorate a sentence that says
    something else."""
    p = vmp._ENHANCE_PROMPT.lower()
    assert 'instruction mode' in p and 'enrich mode' in p
    assert 'pick automatically' in p
    # And what it must never do, in the wording rather than in a checker.
    assert 'no camera' in p
    assert 'only the resulting prompt' in p


def test_auto_is_STEERED_by_what_is_already_in_the_field(app, tmp_path, monkeypatch):
    """The frame says what is THERE; the field says what should HAPPEN in it.
    Steering rather than replacing is the difference between a suggestion and a
    tool — and the people in the answer must stay the frame's."""
    from app.services import vision_llm
    seen = {}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image',
                        lambda data, prompt, **kw: seen.update(prompt=prompt, kw=kw)
                        or 'She jumps twice, landing softly.')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        out = vmp.suggest_from_frame('f.png', instruction='make her jump twice')
    assert out.startswith('She jumps twice')
    assert 'make her jump twice' in seen['prompt']
    assert 'the frame actually shows' in seen['prompt']
    # Free proposal when the field is empty — no instruction is appended.
    with app.app_context():
        vmp.suggest_from_frame('f.png')
    assert 'user asks for this movement' not in seen['prompt'].lower()


def test_the_chosen_model_travels_and_empty_means_the_providers_own(app, monkeypatch):
    from app.services import vision_llm
    seen = {}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: seen.update(kw=kw)
                        or 'She turns her head slowly and smiles.')
    with app.app_context():
        vmp.enhance('she turns', model='qwen3-vl:8b')
        assert seen['kw']['model'] == 'qwen3-vl:8b'
        vmp.enhance('she turns', model='')
        assert seen['kw']['model'] is None      # the provider's own, not ''


def test_the_motion_model_is_its_own_setting(app):
    """Not the image passes' vision_model: the two answer different questions
    on the same machine, and tuning one must not re-point the other."""
    with app.app_context():
        assert vmp.configured_model() == ''
        assert vmp.set_model('qwen3-vl:8b') == 'qwen3-vl:8b'
        assert vmp.configured_model() == 'qwen3-vl:8b'
        from app import config as cfg
        assert (cfg.get('ollama.vision_model') or '') != 'qwen3-vl:8b'
        assert vmp.set_model('') == ''          # back to the provider's own
