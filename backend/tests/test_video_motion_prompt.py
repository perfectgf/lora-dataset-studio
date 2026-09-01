"""✨ The Motion field, written or enriched by the local LLM.

Both gestures are the image generator's own, on this app's existing waist: the
provider and the model are the ones already configured for the image passes.

What this file pins is what live use found missing (2026-09-02): the answers
came back shapeless. So the craft rules must REACH the model, the two gestures
must share them, the vision half must not be asked to compose, and the answer
must arrive whole — an earlier scrubber kept only the first line of a
multi-line reply, which cut two thirds off every answer it touched.
"""
import pytest

from app.services import video_motion_prompt as vmp


# --- the answer that reaches the sampler ---------------------------------------

def test_the_whole_answer_survives_the_scrub_not_just_its_first_line():
    """The regression that made the button look like it ignored every rule: a
    model that answers in three lines had two of them thrown away."""
    out = vmp._scrub('She turns slowly toward the window.\n'
                     'The camera pushes in and settles on her face.\n'
                     'Soft rain hisses against the glass.')
    assert out == ('She turns slowly toward the window. The camera pushes in '
                   'and settles on her face. Soft rain hisses against the glass.')
    assert '\n' not in out


def test_the_scrub_keeps_the_prompt_and_drops_everything_said_about_it():
    assert vmp._scrub('Sure! Here is the prompt: she turns and walks away') \
        == 'she turns and walks away'
    assert vmp._scrub('"she lifts her hand slowly"') == 'she lifts her hand slowly'
    # A bulleted or numbered answer is still the prompt: the marker goes, the
    # sentence stays — dropping the line would lose the motion itself.
    assert vmp._scrub('- she leans forward\nThis prompt keeps your intent.') \
        == 'she leans forward'
    assert vmp._scrub('1. she leans forward\n2. her hair falls') \
        == 'she leans forward her hair falls'
    assert vmp._scrub('```\nshe blinks\n```') == 'she blinks'
    assert vmp._scrub('Motion prompt: she blinks') == 'she blinks'
    assert vmp._scrub('') == ''


# --- the rules themselves, and that they arrive ---------------------------------

def test_both_gestures_answer_to_the_SAME_craft_rules():
    """One block, two gestures. Two divergent rule sets is how ✨ Auto and
    ✨ Enrich came back looking like different products."""
    assert vmp._H3_CRAFT in vmp._AUTO_SYSTEM
    assert vmp._H3_CRAFT in vmp._ENHANCE_SYSTEM


def test_the_craft_rules_are_the_ones_H3_actually_answers_to():
    """Sourced from the published H3 guides and from this app's own graph —
    each of these was a defect in the shapeless version."""
    # Whitespace-collapsed: a rule is what it SAYS, and a line break landing
    # between two of its words is not a change to the rule.
    c = ' '.join(vmp._H3_CRAFT.lower().split())
    assert 'exactly one' in c and 'camera' in c          # one path, never stacked
    assert 'static framing' in c                         # ... including no move
    assert '[' not in c.replace('[the', '')              # H3 has no bracket commands
    assert 'sound' in c and 'audio' in c                 # the graph decodes audio
    assert 'ending' in c                                 # resolve on a final state
    assert 'never re-describe' in c                      # the frame already says it
    assert 'speed and direction' in c                    # unquantified motion fails
    assert 'words, one paragraph' in c                   # a length, and one shape
    assert 'no bullet points' in c and 'no headings' in c
    assert 'output only the prompt' in c
    assert 'uncensored' in c                             # or the model waters it down


def test_the_enhancer_has_the_two_modes_the_image_generator_has():
    """The ported design: the MODEL picks between obeying an instruction about
    the motion and enriching the motion itself. Without the instruction mode a
    click on 'make her jump instead' would decorate a sentence that says
    something else."""
    p = vmp._ENHANCE_SYSTEM.lower()
    assert 'instruction mode' in p and 'enrich mode' in p
    assert 'pick automatically' in p


def test_the_rules_reach_the_model_and_so_does_the_sampling(app, monkeypatch):
    """Not a claim about a constant: the call is captured, and what the driver
    receives is what is checked."""
    from app.services import vision_llm
    seen = {}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: seen.update(prompt=prompt, kw=kw)
                        or 'She turns slowly, the camera pushing in.')
    with app.app_context():
        vmp.enhance('she turns')
    assert vmp._H3_CRAFT in seen['prompt'], 'the craft rules never left the module'
    assert 'she turns' in seen['prompt']
    assert seen['kw']['stop'] == vmp._STOP
    assert seen['kw']['top_p'] == vmp.TOP_P
    assert seen['kw']['temperature'] == vmp.TEMP_ENHANCE
    # The window has to hold the rules: Ollama's default in this app is 4096,
    # and a truncated system prompt is exactly a model that ignores the rules.
    assert seen['kw']['num_ctx'] == vmp.NUM_CTX >= 8192


def test_auto_runs_warmer_than_the_enhancer(app, tmp_path, monkeypatch):
    """✨ Auto is pressed AGAIN when its idea was not the one wanted, so it must
    not answer the same thing twice; ✨ Enrich is applied to a sentence somebody
    chose, so it stays near it."""
    assert vmp.TEMP_AUTO > vmp.TEMP_ENHANCE
    from app.services import vision_llm
    seen = {}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image',
                        lambda *a, **kw: 'A woman sits on a bed, hands on her knees.')
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: seen.update(kw=kw)
                        or 'She leans back slowly as the camera pushes in.')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        vmp.suggest_from_frame('f.png')
    assert seen['kw']['temperature'] == vmp.TEMP_AUTO


# --- AUTO is two steps, on purpose ---------------------------------------------

def test_auto_looks_first_and_composes_second(app, tmp_path, monkeypatch):
    """The split that fixes the shapeless answers: the vision model is asked
    for a FROZEN still and is never handed the craft rules, then the writer
    composes from that description alone. One call doing both is what produced
    a re-description of the picture instead of a movement."""
    from app.services import vision_llm
    calls = {'vision': [], 'text': []}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image',
                        lambda data, prompt, **kw: calls['vision'].append(prompt)
                        or 'A woman kneels on a bed, gaze down, hands on her thighs.')
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: calls['text'].append(prompt)
                        or 'She lifts her gaze slowly toward the lens.')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        out = vmp.suggest_from_frame('f.png')

    assert out == 'She lifts her gaze slowly toward the lens.'
    assert len(calls['vision']) == 1 and len(calls['text']) == 1
    look = calls['vision'][0].lower()
    assert 'frozen still' in look
    assert 'do not invent or imply any motion' in look
    assert vmp._H3_CRAFT not in calls['vision'][0], 'the eye was asked to compose'
    # And the writer works from what the eye said, not from the file.
    assert 'A woman kneels on a bed' in calls['text'][0]
    assert vmp._H3_CRAFT in calls['text'][0]


def test_a_free_press_carries_a_spark_and_a_steered_one_carries_the_order(
        app, tmp_path, monkeypatch):
    """Two presses of ✨ Auto must be able to land somewhere else — but not
    when the user said what should happen: then the field is an order, not a
    lottery ticket."""
    from app.services import vision_llm
    asks = []
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image', lambda *a, **kw: 'A woman stands.')
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: asks.append(prompt)
                        or 'She steps forward slowly.')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        vmp.suggest_from_frame('f.png')
        vmp.suggest_from_frame('f.png', instruction='make her jump twice')

    free, steered = asks[0], asks[1]
    assert any(e in free for e in vmp._SPARK_ENERGY)
    assert any(f in free for f in vmp._SPARK_FOCUS)
    assert any(c in free for c in vmp._SPARK_CAMERA)
    assert 'make her jump twice' in steered
    assert 'the frame actually shows' in steered
    # A steered press is not a lottery: no spark competes with the order.
    assert not any(e in steered for e in vmp._SPARK_ENERGY)


# --- the enhancer, anchored ------------------------------------------------------

def test_the_enhancer_is_anchored_on_the_frame_when_there_is_one(
        app, tmp_path, monkeypatch):
    from app.services import vision_llm
    seen = {}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image',
                        lambda *a, **kw: 'A woman on a red sofa, no window in sight.')
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: seen.update(prompt=prompt)
                        or 'She turns her head slowly to the left.')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        vmp.enhance('she turns', image='f.png')
    assert 'red sofa' in seen['prompt']
    assert 'never' in seen['prompt'] and 're-describe it' in seen['prompt']


def test_a_frame_that_cannot_be_read_costs_the_anchor_never_the_enrichment(
        app, monkeypatch):
    """Degrading, not failing: the text is still worth enriching without a
    picture, and an error here would read as "the button is broken"."""
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda *a, **kw: 'She turns her head slowly to the left.')
    with app.app_context():
        assert vmp.enhance('she turns', image='gone.png') \
            == 'She turns her head slowly to the left.'


def test_an_unusable_answer_never_costs_the_user_their_own_prompt(app, monkeypatch):
    """The enhancer's one destructive failure mode, refused by construction:
    a model that answers nothing usable gives the original back."""
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text', lambda *a, **kw: '   ')
    with app.app_context():
        assert vmp.enhance('she turns her head') == 'she turns her head'


def test_enhance_refuses_an_empty_field_but_not_a_short_motion(app, monkeypatch):
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda *a, **kw: 'She blinks slowly, lashes catching the light.')
    with app.app_context():
        with pytest.raises(ValueError, match='nothing to enrich'):
            vmp.enhance('  ')
        # A SHORT motion is still a motion. The answer floor and the ask floor
        # are two different numbers; sharing them refused 'she blinks'.
        assert vmp.enhance('she blinks').startswith('She blinks slowly')


# --- refusals, and the model that does the work -----------------------------------

def test_suggest_refuses_in_words_rather_than_writing_nothing(app, monkeypatch):
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    with app.app_context():
        with pytest.raises(ValueError, match='start frame'):
            vmp.suggest_from_frame('')
        with pytest.raises(ValueError, match='not on this machine'):
            vmp.suggest_from_frame('never_staged.png')


def test_a_frame_the_model_cannot_describe_stops_auto_with_a_sentence(
        app, tmp_path, monkeypatch):
    """Rather than writing a movement for a picture nobody looked at."""
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image', lambda *a, **kw: '')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        with pytest.raises(ValueError, match='describe that start frame'):
            vmp.suggest_from_frame('f.png')


def test_without_a_local_model_both_say_which_one_is_missing(app, monkeypatch):
    monkeypatch.setattr(vmp, 'available', lambda: (False, 'Ollama: not running'))
    with app.app_context():
        with pytest.raises(ValueError, match='Ollama: not running'):
            vmp.enhance('she turns her head')
        with pytest.raises(ValueError, match='Ollama: not running'):
            vmp.suggest_from_frame('a.png')


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
