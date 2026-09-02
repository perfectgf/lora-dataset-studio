"""✨ The Motion field, written or enriched by the local LLM.

Both gestures are the image generator's own, on this app's existing waist: the
provider and the model are the ones already configured for the image passes.

What this file pins is what live use found missing (2026-09-02): the answers
came back shapeless, and then ✨ Auto ignored the clip length. So the craft
rules must REACH the model, the two gestures must share them, the vision half
must not be asked to compose, the answer must arrive whole — an earlier
scrubber kept only the first line of a multi-line reply — and it must arrive in
H3's OFFICIAL three-field format, paced to the seconds the dials are set to,
with what the format requires (the fields on their lines, the "[Shot 1]"
opener, the <Picture 1> tag, the I2V header) guaranteed in code.
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
    """Sourced from MiniMax's own writing guide for the open weights — the one
    the ComfyUI text encoder was built against — and from this app's graph.
    An earlier block, written from the hosted platform's guides, forbade the
    bracket markers the official format is made of."""
    # Whitespace-collapsed: a rule is what it SAYS, and a line break landing
    # between two of its words is not a change to the rule.
    c = ' '.join(vmp._H3_CRAFT.lower().split())
    # The three fields, by name and in order, and the dialect the hosted
    # platform taught every model is named as the thing NOT to write.
    i = c.index('integrated_multimodal_description:')
    j = c.index('overall_soundscape:')
    k = c.index('non_diegetic_music:')
    assert i < j < k
    assert 'never write an "audio:" line' in c
    assert '[shot 1]' in c and 'the camera cuts to' in c
    assert 'shot 1 never takes a timestamp' in c
    # The camera grammar is the official vocabulary, one move per shot.
    for move in ('push in/pull out', 'pan left/right', 'truck left/right',
                 'tilt up/down', 'arc shot', 'tracking shot', 'static shot'):
        assert move in c, move
    assert 'one camera move per shot' in c
    # And the rules that made the shapeless version move, kept.
    assert 'never re-describe' in c                      # the frame already says it
    assert 'speed and a direction' in c                  # unquantified motion fails
    assert 'ending on' in c                              # resolve on a final state
    assert 'secondary motion' in c
    assert 'no bullet points' in c and 'no headings' in c
    assert 'output only the three fields' in c
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
    # The rest of the recommended non-thinking sampling travels too, and
    # thinking is OFF: a hybrid model that reasons about the format spends
    # the token budget on the reasoning and truncates the prompt.
    assert seen['kw']['top_k'] == vmp.TOP_K == 20
    assert seen['kw']['min_p'] == vmp.MIN_P == 0.0
    assert seen['kw']['presence_penalty'] == vmp.PRESENCE_PENALTY == 1.0
    assert seen['kw']['think'] is False
    # The window has to hold the rules: Ollama's default in this app is 4096,
    # and a truncated system prompt is exactly a model that ignores the rules.
    assert seen['kw']['num_ctx'] == vmp.NUM_CTX >= 8192
    # Strict, like the image generator's enhancer: without it the driver
    # dissolves every failure — the fence included — into '' and a warning,
    # and the user is told "nothing usable" about a model that was never
    # asked. With it the route gets the exception and answers 409.
    assert seen['kw']['strict'] is True


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

    # The prose answer is finished into the official format — the field, the
    # opener, the identity tag and the frame header are code's job, not the
    # model's — and the movement itself arrives whole.
    assert out.endswith(
        "integrated_multimodal_description: [Shot 1] The subject's identity, "
        'face and wardrobe are locked to <Picture 1>. She lifts her gaze slowly '
        'toward the lens.\noverall_soundscape: N/A\nnon_diegetic_music: N/A')
    assert out.startswith('For the target video, at 0.00 seconds')
    assert len(calls['vision']) == 1 and len(calls['text']) == 1
    look = calls['vision'][0].lower()
    assert 'frozen still' in look
    assert 'do not invent or imply any motion' in look
    # And the eye is told to describe THIS scene: asked for a still, a vision
    # model has been measured answering with an invented one.
    assert 'never replace it with a different, invented scene' in look
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
        out = vmp.enhance('she turns', image='gone.png')
    assert 'integrated_multimodal_description: [Shot 1] ' in out
    assert out.endswith('She turns her head slowly to the left.\n'
                        'overall_soundscape: N/A\nnon_diegetic_music: N/A')


def test_an_unusable_answer_is_refused_in_words_not_handed_back_as_a_success(
        app, monkeypatch):
    """The enhancer used to return the original when the model answered
    nothing usable — on the wire a success with nothing to show, and the
    panel said "nothing to add" about a model that had failed. It raises, as
    the image studio's writer does; the field is only written on success, so
    the caller's text is safe either way."""
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text', lambda *a, **kw: '   ')
    with app.app_context():
        with pytest.raises(RuntimeError, match='nothing usable'):
            vmp.enhance('she turns her head')


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
        assert 'She blinks slowly' in vmp.enhance('she blinks')


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


def test_the_saved_choice_writes_when_the_caller_sends_no_model(app, tmp_path, monkeypatch):
    """The launch's "Enrich at launch" sends no model, and a panel that just
    reloaded has not re-read the ⚙ window: both must still write with the
    model that was chosen there — for the still AND for the text, so the
    two steps of ✨ Auto never load two different models."""
    from app.services import vision_llm
    seen = {'text': [], 'still': []}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: seen['text'].append(kw['model'])
                        or 'integrated_multimodal_description: [Shot 1] She turns slowly toward the lens.')
    monkeypatch.setattr(vision_llm, 'describe_image',
                        lambda data, prompt, **kw: seen['still'].append(kw['model'])
                        or 'A woman in a red coat stands by a window.')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    (tmp_path / 'lds_vts_a.png').write_bytes(b'x')
    with app.app_context():
        vmp.set_model('qwen3-vl:8b')
        vmp.enhance('she turns')
        vmp.suggest_from_frame('lds_vts_a.png')
        vmp.enhance('she turns', model='llava:13b')       # the caller's wins
        vmp.set_model('')
        vmp.enhance('she turns')
    assert seen['text'] == ['qwen3-vl:8b', 'qwen3-vl:8b', 'llava:13b', None]
    assert seen['still'] == ['qwen3-vl:8b']


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


# --- the clip length reaches the writer ------------------------------------------

def test_the_clip_length_is_in_the_ask_and_paces_one_shot(app, tmp_path, monkeypatch):
    """The defect that opened the port: ✨ Auto wrote the same beat for a 1 s
    clip and a 15 s one. The seconds the dials are set to now travel into the
    ask as a shot plan, for BOTH gestures."""
    from app.services import vision_llm
    asks = []
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image', lambda *a, **kw: 'A woman stands by a window.')
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: asks.append(prompt)
                        or 'integrated_multimodal_description: [Shot 1] She turns slowly.')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        vmp.suggest_from_frame('f.png', seconds=15.04)
        vmp.enhance('she turns', image='f.png', seconds=2.29)
        vmp.enhance('she turns', seconds=0.88)
    assert 'The clip is 15 seconds long.' in asks[0]
    assert 'The clip is 2 seconds long.' in asks[1]
    assert 'The clip is 1 second long.' in asks[2]     # 22 frames is a 1 s clip
    for ask in asks:
        assert 'write ONE single continuous shot' in ask
        assert 'never write "the camera cuts to"' in ask
    # And what the seconds MEAN: measured with the length alone, a 2 s clip
    # and a 15 s one got the same four-beat sequence.
    assert '15s is a long take' in asks[0] and 'successive beats' in asks[0]
    assert '2s holds ONE movement' in asks[1] and 'not a sequence of beats' in asks[1]
    assert '1s holds ONE movement' in asks[2]
    assert 'holds ONE movement' not in vmp.shot_directive(5)
    assert 'long take' not in vmp.shot_directive(7)
    assert 'long take' in vmp.shot_directive(8)


def test_without_a_known_length_the_plan_paces_nothing_rather_than_guessing():
    d = vmp.shot_directive(None)
    assert 'The clip is' not in d
    assert 'ONE single continuous shot' in d
    assert vmp.clip_seconds(None) == 0
    assert vmp.clip_seconds('') == 0
    assert vmp.clip_seconds(-3) == 0
    assert vmp.clip_seconds('nan') == 0
    assert vmp.clip_seconds(0.88) == 1                # rounded, floored at one
    assert vmp.clip_seconds(15.04) == 15


def test_a_multi_shot_plan_carries_the_official_timecodes_the_model_must_copy():
    """The shot selector is a later chantier, but the plan it will drive is
    pinned now: evenly spaced cut marks in the official MM:SS.mmm form, the
    exact words "the camera cuts to", and never a timestamp on Shot 1."""
    assert vmp.shot_cut_marks(10, 3) == '00:03.300, 00:06.700'
    assert vmp.shot_cut_marks(15, 6) == '00:02.500, 00:05.000, 00:07.500, 00:10.000, 00:12.500'
    d = vmp.shot_directive(10, 3)
    assert 'The clip is 10 seconds long.' in d
    assert 'EXACTLY 3 shots' in d
    assert 'Use EXACTLY these cut timecodes, in order: 00:03.300, 00:06.700.' in d
    assert 'the camera cuts to' in d
    assert '"[Shot 1]" (no timestamp)' in d
    # A short clip caps the count — a cut every 0.7 s is a flicker.
    assert vmp.shot_count(4, 2) == 2
    assert vmp.shot_count(3, 1) == 1
    assert vmp.shot_count(99, 10) == vmp.MAX_SHOTS
    assert vmp.shot_count('junk', 10) == 1
    # Four seconds hold four shots at most: six on four is one every 0.67 s,
    # under the line the rule names — the cap stopped one second short of it.
    # Five seconds carry six.
    assert vmp.shot_count(6, 4) == 4
    assert vmp.shot_count(6, 5) == 6
    # Every length the graph can take, every count the plan allows: no shot
    # shorter than 0.7 s once the clip has a length to pace.
    for secs in range(1, 16):
        for shots in range(1, vmp.MAX_SHOTS + 1):
            assert secs / vmp.shot_count(shots, secs) >= 0.7, (secs, shots)
    assert 'EXACTLY 2 shots' in vmp.shot_directive(2, 4)
    # An unknown length with several shots still asks for increasing codes.
    d = vmp.shot_directive(None, 2)
    assert 'EXACTLY 2 shots' in d and 'strictly increasing timecodes' in d


# --- the official format, guaranteed in code -------------------------------------

def test_the_three_fields_are_rebuilt_from_their_labels_whatever_the_line_breaks():
    """The scrub flattens the answer to one line (its job is the model's
    commentary); the labels are found again and each field put back on its
    own line — so a model that wrapped, merged or chattered still yields the
    exact shape the encoder was trained on."""
    raw = ('Here is your prompt:\n'
           'integrated_multimodal_description: Live-action, cinematic. The woman from '
           '<Picture 1> turns slowly toward the camera.\n'
           'The camera pushes in with small amplitude at slow speed. '
           'overall_soundscape: Soft rain on glass, fabric rustle.\n'
           'non_diegetic_music: Sparse piano at a slow tempo.\n'
           'Note: hope this helps')
    out = vmp.finish(raw, with_image=True)
    assert out.split('\n\n', 1)[1] == (
        'integrated_multimodal_description: [Shot 1] Live-action, cinematic. The woman '
        'from <Picture 1> turns slowly toward the camera. The camera pushes in with '
        'small amplitude at slow speed.\n'
        'overall_soundscape: Soft rain on glass, fabric rustle.\n'
        'non_diegetic_music: Sparse piano at a slow tempo.')
    assert 'hope this helps' not in out and 'Here is' not in out


def test_the_soundscape_label_survives_the_chatter_filter():
    """`overall_soundscape:` starts with the word the chatter filter drops
    ("Overall, the scene..."); an unguarded filter swallowed the field."""
    assert vmp._scrub('overall_soundscape: rain on glass') == 'overall_soundscape: rain on glass'
    assert vmp._scrub('Overall, this prompt keeps your intent.') == ''


def test_a_prose_answer_or_an_audio_line_becomes_the_official_fields():
    """The hosted platform's dialect — prose with an "Audio:" tail — is what
    every model learned first. It is mapped, never passed through."""
    out = vmp.finish('She turns her head slowly toward the window. Audio: rain on glass.',
                     with_image=False)
    assert out == ('integrated_multimodal_description: [Shot 1] She turns her head slowly '
                   'toward the window.\n'
                   'overall_soundscape: rain on glass.\n'
                   'non_diegetic_music: N/A')


def test_a_field_that_stops_on_a_word_no_clause_ends_on_loses_that_tail():
    long = ('integrated_multimodal_description: [Shot 1] She turns slowly toward the '
            'window and the light climbs her face as the curtain lifts. Her hand rises '
            'toward the glass and then the')
    out = vmp.finish(long, with_image=False)
    assert out.startswith('integrated_multimodal_description: [Shot 1] She turns slowly')
    assert 'and then the' not in out
    assert out.split('\n')[0].endswith('curtain lifts.')
    # The same tail on a pronoun is a clause that may only have lost its full
    # stop: under the budget, nothing proves a cut, and it stays.
    out = vmp.finish(long[:-len('the')] + 'she', with_image=False)
    assert out.split('\n')[0].endswith('toward the glass and then she')
    # A short field is kept whole rather than cut to a stub; N/A is N/A.
    assert vmp._trim_dangling('N/A') == 'N/A'
    assert vmp._trim_dangling('rain on glass, then wind') == 'rain on glass, then wind'
    # Measured: the model copies the placeholder after a real soundscape.
    # The placeholder goes, the soundscape stays — at any length.
    assert vmp._trim_dangling('Soft breaths, fabric rustle. N/A') == 'Soft breaths, fabric rustle.'
    assert vmp._trim_dangling('rain N/A') == 'rain'
    assert vmp._trim_dangling('the N/A count is high.') == 'the N/A count is high.'
    # And through the whole pipeline, on the shape the model actually wrote:
    # the placeholder sits on the soundscape line, trailing spaces and all,
    # and the scrub flattens the answer before the fields are found again.
    raw = ('integrated_multimodal_description: [Shot 1] She turns her head slowly '
           'toward the window, ending on a close-up of her face.  \n'
           'overall_soundscape: Soft breaths, the faint rustle of cotton. N/A  \n'
           'non_diegetic_music: N/A')
    assert vmp.finish(raw, with_image=False) == (
        'integrated_multimodal_description: [Shot 1] She turns her head slowly '
        'toward the window, ending on a close-up of her face.\n'
        'overall_soundscape: Soft breaths, the faint rustle of cotton.\n'
        'non_diegetic_music: N/A')


def test_the_hybrid_dialects_are_mapped_back_to_the_official_grammar():
    raw = ('integrated_multimodal_description: [Shot 1] At 00:00.000, she turns. '
           'Timeline: [5s-10s] [Shot 2 At 00:05.000, the camera cuts to] her hands. '
           'overall_soundscape: N/A non_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=False)
    assert out.split('\n')[0] == (
        'integrated_multimodal_description: [Shot 1] she turns. '
        '[Shot 2] At 00:05.000, the camera cuts to her hands.')


def test_an_image_to_video_prompt_names_the_frame_and_wears_the_header_once():
    """Two things the encoder pairs with the picture block it prepends: the
    <Picture 1> tag in the description (measured missing in half the answers
    on the image generator's side) and the official alignment header. Both
    are added by code, and neither twice."""
    body = ('integrated_multimodal_description: [Shot 1] She turns slowly.\n'
            'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    out = vmp.finish(body, with_image=True)
    assert out.startswith(vmp._ALIGNMENT_HEADER + '\n\n')
    assert out.count('is fully referenced') == 1
    assert ('integrated_multimodal_description: [Shot 1] ' + vmp._IDENTITY_SENTENCE
            + ' She turns slowly.') in out
    # Idempotent: an already-finished prompt enriched again gains nothing.
    again = vmp.finish(out, with_image=True)
    assert again.count('is fully referenced') == 1
    assert again.count('<Picture 1>') == out.count('<Picture 1>')
    # A model that referenced the frame itself is not tagged a second time.
    tagged = vmp.finish('integrated_multimodal_description: [Shot 1] The woman from '
                        '<Picture 1> turns slowly.', with_image=True)
    assert vmp._IDENTITY_SENTENCE not in tagged


def test_a_text_to_video_prompt_names_no_picture(app, monkeypatch):
    """No frame is given to the encoder, so no <Picture 1> and no header —
    and the writer is TOLD there is no picture, rather than left to copy the
    identity rule from the format block."""
    from app.services import vision_llm
    seen = {}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: seen.update(prompt=prompt)
                        or 'integrated_multimodal_description: [Shot 1] She turns slowly.')
    with app.app_context():
        out = vmp.enhance('she turns', seconds=5)
    assert 'Picture 1' not in out
    assert 'is fully referenced' not in out
    assert out.startswith('integrated_multimodal_description: [Shot 1] She turns slowly.')
    assert vmp._NO_PICTURE_RULE in seen['prompt']
    assert vmp._IDENTITY_RULE not in seen['prompt']


def test_the_enhancer_with_a_frame_is_told_to_reference_it(app, tmp_path, monkeypatch):
    from app.services import vision_llm
    seen = {}
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'describe_image', lambda *a, **kw: 'A woman on a sofa.')
    monkeypatch.setattr(vision_llm, 'generate_text',
                        lambda prompt, **kw: seen.update(prompt=prompt)
                        or 'integrated_multimodal_description: [Shot 1] She turns slowly.')
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'\x89PNG\r\n')
    monkeypatch.setattr('app.config.comfyui_dir', lambda *a, **k: str(tmp_path))
    with app.app_context():
        out = vmp.enhance('she turns', image='f.png', seconds=5)
    assert vmp._IDENTITY_RULE in seen['prompt']
    assert vmp._NO_PICTURE_RULE not in seen['prompt']
    assert '<Picture 1>' in out and out.startswith(vmp._ALIGNMENT_HEADER)


def test_a_stub_answer_is_never_dressed_up_as_a_prompt(app, monkeypatch):
    """The floor is measured on the model's own words: a one-letter answer
    wrapped in the header and the tag would pass any length check and reach
    the sampler as a prompt that says nothing."""
    assert vmp.finish('I', with_image=True) == 'I'
    assert vmp.finish('', with_image=True) == ''
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text', lambda *a, **kw: 'Ok.')
    with app.app_context():
        with pytest.raises(RuntimeError, match='nothing usable'):
            vmp.enhance('she turns her head', image=None)


# --- what /verif found in the finishing, pinned ----------------------------------

def test_an_infinite_length_paces_nothing():
    """`float('inf')` passes `> 0` and `round()` raises OverflowError — one
    request body away from a 500 on a JSON number the panel never sends."""
    assert vmp.clip_seconds(float('inf')) == 0
    assert vmp.clip_seconds('inf') == 0
    assert vmp.clip_seconds('-inf') == 0
    assert vmp.clip_seconds(4.4) == 4


def test_markdown_emphasis_and_fences_around_the_labels_do_not_become_words():
    """A model that bolds the labels or wraps the answer in a fence is still
    answering in the format — before the fix, "**" was the description's
    first word and "---" a sentence of its own."""
    raw = ('```text\n**integrated_multimodal_description:** She turns toward the window slowly.\n'
           '---\n**overall_soundscape:** rain on glass\n'
           '**non_diegetic_music:** N/A\n```')
    assert vmp.finish(raw, with_image=False) == (
        'integrated_multimodal_description: [Shot 1] She turns toward the window slowly.\n'
        'overall_soundscape: rain on glass\n'
        'non_diegetic_music: N/A')
    # A markdown heading over the answer is chatter, not a shot.
    assert vmp._scrub('## Motion prompt\nShe turns slowly.') == 'She turns slowly.'


def test_a_hybrid_model_s_reasoning_never_reaches_the_sampler():
    """`think:false` travels to Ollama and not to LM Studio's chat endpoint,
    so a hybrid model chosen there may hand its <think> block back inline.
    The block goes — closed or cut open by the token budget."""
    raw = ('<think>\nThe user wants motion. The camera should push in.\n</think>\n'
           'integrated_multimodal_description: She turns toward the window slowly.\n'
           'overall_soundscape: rain\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=False)
    assert out.startswith('integrated_multimodal_description: [Shot 1] She turns toward')
    assert 'think' not in out and 'push in' not in out
    # Cut open by the budget: nothing after the tag is an answer.
    assert vmp.finish('<think>\nLet me plan the shot carefully and', with_image=True) == ''


def test_the_chatter_around_an_answer_is_dropped_in_all_its_dialects():
    raw = ('Below is the enhanced prompt:\n'
           'integrated_multimodal_description: She turns toward the window slowly.\n'
           'overall_soundscape: rain\nnon_diegetic_music: N/A\n'
           'Let me know if you want changes!\nHope this helps.\nFeel free to adjust.')
    assert vmp.finish(raw, with_image=False) == (
        'integrated_multimodal_description: [Shot 1] She turns toward the window slowly.\n'
        'overall_soundscape: rain\n'
        'non_diegetic_music: N/A')
    # The lead-in that carries the prompt on its own line is still salvaged.
    assert vmp._scrub('Below is your prompt: she turns slowly') == 'she turns slowly'


def test_the_orphan_tail_cut_knows_a_timecode_from_a_full_stop():
    """`rfind('.')` took the dot inside "00:05.000" for a sentence end and
    the field lost its second shot; and a final clause that merely lost its
    full stop is kept, not cut — only the budget, joining punctuation or a
    word no clause ends on (a determiner, a coordinator) make a tail a cut."""
    two_shots = ('[Shot 1] She turns toward the window slowly, the light on her cheek. '
                 '[Shot 2] At 00:05.000, the camera cuts to a close-up of her hands folding the letter')
    assert vmp._trim_dangling(two_shots) == two_shots
    kept = ('She turns her head slowly toward the window. The camera follows her hand '
            'as it rises to the glass and rests there, ending on a close-up of her face')
    assert vmp._trim_dangling(kept) == kept
    # The same tail with the budget hit IS the cut.
    assert vmp._trim_dangling(kept, truncated=True) == \
        'She turns her head slowly toward the window.'
    # A tail on a determiner or a coordinator is a fragment whatever the
    # budget — each word of the list, or a word could leave it unnoticed.
    for w in ('a', 'an', 'the', 'and', 'or', 'nor'):
        assert vmp._trim_dangling('She turns her head slowly toward the window. The camera '
                                  f'follows her hand as it rises to the glass and rests on {w}') == \
            'She turns her head slowly toward the window.', w
    assert vmp._trim_dangling('She turns her head slowly toward the window. Then the camera,') == \
        'She turns her head slowly toward the window.'
    # The placeholder with its own full stop still goes.
    assert vmp._trim_dangling('Soft breaths, fabric rustle. N/A.') == 'Soft breaths, fabric rustle.'
    # The budget is judged on the scrubbed answer, once, in finish() — and
    # at the budget a tail on a pronoun is the cut too.
    long = 'integrated_multimodal_description: ' + ('She turns slowly. ' * 100) + 'and then she'
    assert len(long.split()) >= vmp._BUDGET_WORDS
    out = vmp.finish(long, with_image=False)
    assert out.splitlines()[0].endswith('She turns slowly.')


def test_a_final_clause_that_lost_its_full_stop_stays_whatever_its_last_word():
    """Refuted on the previous word list: four of these five legitimate
    closing clauses were amputated because they end on a pronoun, a
    preposition or "is" ("behind her", "looking at", "everything that is")
    — the vocabulary the craft rules ask for ("toward the camera", "to her
    left"). A missing full stop is not proof of a cut; the budget is, or a
    word no clause can end on."""
    clauses = [
        'She grips the rail and arches back slowly. The rim light settles behind her',
        'Fabric slips from her shoulder and the light shifts. He steps in close behind her',
        'She turns toward the window, breathing steadily. The lens holds on what she is looking at',
        'She grips the rail and arches back slowly. The light stays there',
        'Fabric slips from her shoulder and the light shifts. The frame holds everything that is',
    ]
    for c in clauses:
        assert vmp._trim_dangling(c) == c, c
        # Through the pipeline as well: the description keeps its last clause.
        out = vmp.finish('integrated_multimodal_description: ' + c, with_image=False)
        assert out.splitlines()[0] == 'integrated_multimodal_description: [Shot 1] ' + c, c
        # The budget hit is the cut, on these same tails.
        assert vmp._trim_dangling(c, truncated=True) == c.split('. ')[0] + '.', c


def test_a_shot_marker_in_mid_field_moves_to_the_front_instead_of_doubling():
    raw = ('integrated_multimodal_description: A slow push-in. [Shot 1] She turns toward the window.\n'
           'overall_soundscape: rain\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=False)
    assert out.splitlines()[0] == \
        'integrated_multimodal_description: [Shot 1] A slow push-in. She turns toward the window.'
    assert out.count('[Shot 1]') == 1


def test_a_description_written_after_the_header_without_its_label_is_kept():
    """The model copies the header from the rules and then starts the
    description under it, label-less — the header swallowed both."""
    raw = (vmp._ALIGNMENT_HEADER
           + 'She turns toward the window slowly, the light catching her cheek.\n'
           'integrated_multimodal_description: The camera pushes in.\n'
           'overall_soundscape: rain\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=True)
    assert out.startswith(vmp._ALIGNMENT_HEADER)
    assert out.count('is fully referenced') == 1
    assert 'She turns toward the window slowly, the light catching her cheek. The camera pushes in.' in out


def test_the_identity_tag_is_looked_for_in_the_description_not_in_the_header():
    """The header names <Picture 1> too. A model that copied the header and
    wrote a description naming nobody was passing the check on the header's
    tag — the one place the encoder does not read it from."""
    raw = (vmp._ALIGNMENT_HEADER
           + 'integrated_multimodal_description: [Shot 1] She turns toward the window slowly.\n'
           'overall_soundscape: rain\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=True)
    assert vmp._IDENTITY_SENTENCE in out
    assert out.count('is fully referenced') == 1


def test_an_audio_line_becomes_soundscape_even_when_music_is_named():
    """The rescue used to run only when BOTH audio fields were empty: with a
    music line present, "Audio: rain" stayed inside the description."""
    raw = ('integrated_multimodal_description: She turns toward the window slowly. Audio: rain on glass\n'
           'overall_soundscape: N/A\nnon_diegetic_music: soft piano')
    assert vmp.finish(raw, with_image=False) == (
        'integrated_multimodal_description: [Shot 1] She turns toward the window slowly.\n'
        'overall_soundscape: rain on glass\n'
        'non_diegetic_music: soft piano')
    # Both written: joined, neither lost.
    raw = ('integrated_multimodal_description: She turns toward the window slowly. Audio: rain on glass\n'
           'overall_soundscape: wind in the curtains\nnon_diegetic_music: N/A')
    assert 'overall_soundscape: wind in the curtains, rain on glass' in vmp.finish(raw, with_image=False)


def test_a_refusal_or_an_empty_format_is_never_dressed_up(app, monkeypatch):
    """Labels, a "[Shot 1]" and a header around nothing pass every length
    check — the floor has to look at the description itself."""
    assert vmp.finish("I'm sorry, but I can't help with that request.", with_image=True) == ''
    assert vmp.finish('As an AI I cannot describe this image.', with_image=True) == ''
    empty = 'integrated_multimodal_description: N/A\noverall_soundscape: N/A\nnon_diegetic_music: N/A'
    assert len(vmp.finish(empty, with_image=True)) < vmp.MIN_CHARS
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text', lambda *a, **kw: empty)
    with app.app_context():
        with pytest.raises(RuntimeError, match='nothing usable'):
            vmp.enhance('she turns her head')


def test_a_text_to_video_finish_strips_what_an_image_to_video_pass_added(app, monkeypatch):
    """The real case: a prompt enriched with a frame, then the panel switched
    to text-only and enriched again — the model copies the header and the
    tag from the text it was given, and the encoder has no picture to pair
    them with. The pass without a picture removes them, and only them."""
    i2v = vmp.finish('integrated_multimodal_description: [Shot 1] <Picture 1> turns toward '
                     'the window slowly (from <Picture 1>).\n'
                     'overall_soundscape: rain\nnon_diegetic_music: N/A', with_image=True)
    assert vmp.has_alignment_header(i2v) and '<Picture 1>' in i2v
    t2v = vmp.finish(i2v, with_image=False)
    assert t2v == ('integrated_multimodal_description: [Shot 1] The subject turns toward '
                   'the window slowly.\n'
                   'overall_soundscape: rain\n'
                   'non_diegetic_music: N/A')
    # Nothing to strip: the text is returned as it is.
    plain = 'integrated_multimodal_description: [Shot 1] She turns.'
    assert vmp.strip_picture_references(plain) == plain
    # And through the enhancer itself, with the model echoing the tagged text.
    from app.services import vision_llm
    monkeypatch.setattr(vmp, 'available', lambda: (True, ''))
    monkeypatch.setattr(vision_llm, 'generate_text', lambda prompt, **kw: i2v)
    with app.app_context():
        out = vmp.enhance(i2v, seconds=5)
    assert 'Picture 1' not in out and 'is fully referenced' not in out


# --- what the /verif replay found in the reconstruction ---------------------------

def test_an_audio_line_inside_a_shot_leaves_the_shots_after_it_in_the_picture():
    """The rescue took everything after "Audio:" for soundscape — with a
    two-shot plan whose first shot carried an audio line, the second shot
    moved into the soundscape and the picture lost its cut."""
    raw = ('integrated_multimodal_description: [Shot 1] She lifts the cup and sips. '
           'Audio: the cup clinks on the saucer. [Shot 2] At 00:05.000, the camera '
           'cuts to a wide view of the room as she stands.\n'
           'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=False).splitlines()
    assert out[0] == ('integrated_multimodal_description: [Shot 1] She lifts the cup and sips. '
                      '[Shot 2] At 00:05.000, the camera cuts to a wide view of the room as she stands.')
    assert out[1] == 'overall_soundscape: the cup clinks on the saucer.'
    # One line per shot: both reach the soundscape, in order, the picture whole.
    raw = ('integrated_multimodal_description: [Shot 1] She sips. Audio: a clink. '
           '[Shot 2] At 00:05.000, the camera cuts to the door as it opens. Audio: hinges creak.\n'
           'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=False).splitlines()
    assert out[0].endswith('[Shot 2] At 00:05.000, the camera cuts to the door as it opens.')
    assert out[1] == 'overall_soundscape: a clink, hinges creak.'


def test_an_audio_placeholder_is_dropped_rather_than_joined():
    """"Audio: N/A" after a real soundscape was joined onto it, and the
    placeholder strip then left "soft rain," — a field ending on a comma."""
    raw = ('integrated_multimodal_description: [Shot 1] She turns toward the window slowly. '
           'Audio: N/A\noverall_soundscape: soft rain\nnon_diegetic_music: N/A')
    assert 'overall_soundscape: soft rain\n' in vmp.finish(raw, with_image=False)
    assert vmp._trim_dangling('soft rain, N/A') == 'soft rain'


def test_the_budget_is_judged_on_the_answer_not_on_the_thinking():
    """A reasoning model's <think> block counted toward the word budget, and a
    complete answer behind a long one read as cut: its last clause — one that
    had merely lost its full stop — went as the budget's tail."""
    answer = ('integrated_multimodal_description: She turns her head slowly toward the '
              'window. The camera follows her hand as it rises to the glass and rests '
              'there, ending on a close-up of her face\n'
              'overall_soundscape: rain\nnon_diegetic_music: N/A')
    thinking = '<think>\n' + ' '.join(['plan'] * 300) + '\n</think>\n'
    assert len((thinking + answer).split()) >= vmp._BUDGET_WORDS
    out = vmp.finish(thinking + answer, with_image=False)
    assert out == vmp.finish(answer, with_image=False)
    assert 'ending on a close-up of her face' in out


def test_reasoning_closed_by_a_bare_tag_goes_with_the_tag():
    """The R1 dialect: the template opens <think> in the prompt, so the output
    is the reasoning and a bare </think> before the answer. The block scrub
    wanted the opening tag; the reasoning became the description."""
    answer = ('integrated_multimodal_description: She turns toward the window slowly.\n'
              'overall_soundscape: rain\nnon_diegetic_music: N/A')
    raw = 'The user wants a slow turn. I should keep it to one beat.\n</think>\n\n' + answer
    out = vmp.finish(raw, with_image=False)
    assert out == vmp.finish(answer, with_image=False)
    assert 'one beat' not in out and 'think' not in out


def test_a_header_written_inside_the_description_is_lifted_out_and_strippable():
    """The split before the first label cannot see a header the model wrote
    AFTER the label. Left there, the field opened on the header — the marker
    hoist then tore its "(from [Shot 1])" — and the text-only strip, which
    looked for the header at the top only, left it in as prose about a
    picture the encoder is never given."""
    raw = ('integrated_multimodal_description: ' + vmp._ALIGNMENT_HEADER
           + ' [Shot 1] <Picture 1> turns her head toward the window.\n'
           'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=True)
    assert out.startswith(vmp._ALIGNMENT_HEADER + '\n\n')
    assert out.count('is fully referenced') == 1
    assert 'integrated_multimodal_description: [Shot 1] <Picture 1> turns her head' in out
    bare = vmp.finish(raw, with_image=False)
    assert 'target video' not in bare and 'Picture 1' not in bare
    assert bare.startswith('integrated_multimodal_description: [Shot 1] The subject turns her head')
    # Written twice — above the label and again inside — is still once.
    twice = vmp._ALIGNMENT_HEADER + '\n' + raw
    assert vmp.finish(twice, with_image=True).count('is fully referenced') == 1


def test_a_sentence_written_before_the_header_is_description_not_header():
    """The header split cut at the END of the header's phrase and filed
    everything before it as header — a lead sentence the model wrote first
    went out above the header instead of into the field."""
    raw = ('She is seated by the window. ' + vmp._ALIGNMENT_HEADER
           + '\nintegrated_multimodal_description: [Shot 1] <Picture 1> turns her head.\n'
           'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    out = vmp.finish(raw, with_image=True)
    assert out.startswith(vmp._ALIGNMENT_HEADER + '\n\n')
    assert ('integrated_multimodal_description: [Shot 1] She is seated by the window. '
            '<Picture 1> turns her head.') in out


def test_a_shot_named_inside_a_sentence_is_prose_not_the_marker():
    """"(as set up in [Shot 1])" is a sentence naming the shot. The hoist took
    it for the marker and moved it to the front, leaving "(as set up in )"."""
    raw = ('integrated_multimodal_description: A slow push-in (as set up in [Shot 1]) '
           'follows her across the room until she reaches the door.\n'
           'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    assert vmp.finish(raw, with_image=False).splitlines()[0] == (
        'integrated_multimodal_description: [Shot 1] A slow push-in (as set up in '
        '[Shot 1]) follows her across the room until she reaches the door.')


# --- the refutation of the header decision --------------------------------------

def test_the_header_is_known_by_its_shape_not_by_its_english():
    """Refuted on a phrase: "is fully referenced" and "align with the target
    video" are ordinary prompt English, and a prompt typed by hand that
    carried one lost the sentence around it — 76 of 128 characters, or the
    whole prompt. The header is a line with the official opening that names
    a picture; nothing else is one — not a line typed with the opening's
    words and no picture in it, which the opening alone took for the header
    (a 400 on its own, a line silently gone from a longer prompt). And the
    header is one SENTENCE: the picture is named in it, not anywhere on its
    line — a typed line opening with the official words, with a picture in
    its NEXT sentence, lost its first sentence to the header — and it is
    found wherever it starts, glued after a sentence or behind a no-break
    space, where an image-to-video launch headed it twice."""
    typed = [
        'Wide shot of a studio where every prop is fully referenced',
        'A woman walks along a beach at sunset, the golden light is fully referenced '
        'to that hour, and she turns slowly toward the camera',
        'A woman walks along a beach at sunset. The mural behind her is fully '
        'referenced from a 1970s tourism poster. She turns toward the camera.',
        'Grade the shot so the tones align with the target video, then hold.',
        "The colours align with the target video's palette throughout the clip.",
        'A slow push in. Everything must align with the target video reference.',
        'For the target video, at 3 seconds she turns toward the camera and the light settles.',
        'A slow push in.\nFor the target video, at 2 seconds in, the camera settles on her face.\n'
        'She turns away.',
        'For the target video, at 3 seconds she turns toward the camera. The picture '
        'on the wall behind her falls.',
    ]
    for p in typed:
        assert not vmp.has_alignment_header(p), p
        assert vmp.strip_picture_references(p) == p, p
        assert vmp.inject_alignment_header(p) == vmp._ALIGNMENT_HEADER + '\n\n' + p, p
    # The official line, alone or glued to the description on one line, is
    # the header, and only it goes.
    assert vmp.has_alignment_header(vmp._ALIGNMENT_HEADER)
    assert vmp.strip_picture_references(vmp._ALIGNMENT_HEADER) == ''
    assert vmp.strip_picture_references(
        vmp._ALIGNMENT_HEADER + ' She turns toward the camera.') == 'She turns toward the camera.'
    # The reference writer's end-frame line, pasted from there, is a header
    # too: it goes from a text-only launch, and an image-to-video launch
    # replaces it — here the picture is the FIRST frame, not the last.
    l2va = ('How the reference pictures align with the target video — <Picture 1> '
            '(from [Shot 1]) aligns with the 5.00-second mark of the target video.\n\n'
            'She turns.')
    assert vmp.has_alignment_header(l2va)
    assert vmp.strip_picture_references(l2va) == 'She turns.'
    assert vmp.inject_alignment_header(l2va) == vmp._ALIGNMENT_HEADER + '\n\nShe turns.'
    # A header wherever its line sits, and at any timecode: the official line
    # pasted UNDER the motion, with the picture at 1.50 s, is one header —
    # gone from a text-only launch, and replaced by this launch's line above
    # the motion, where the picture is the first frame.
    late = ('She turns toward the camera.\n'
            'For the target video, at 1.50 seconds into the target video, <Picture 1> '
            '(from [Shot 1]) is fully referenced.')
    assert vmp.has_alignment_header(late)
    assert vmp.strip_picture_references(late) == 'She turns toward the camera.'
    assert vmp.inject_alignment_header(late) == vmp._ALIGNMENT_HEADER + '\n\nShe turns toward the camera.'
    # The official line glued between two sentences of one line, or behind a
    # no-break space, is the header where it starts: one header out, the
    # sentences around it kept — headed once, not twice.
    glued = 'She turns slowly. ' + vmp._ALIGNMENT_HEADER + ' Then she smiles.'
    nbsp = '\u00a0' + vmp._ALIGNMENT_HEADER + '\n\nShe turns.'
    assert vmp.has_alignment_header(glued) and vmp.has_alignment_header(nbsp)
    assert vmp.strip_picture_references(glued) == 'She turns slowly. Then she smiles.'
    assert vmp.strip_picture_references(nbsp) == 'She turns.'
    assert vmp.inject_alignment_header(glued) == (
        vmp._ALIGNMENT_HEADER + '\n\nShe turns slowly. Then she smiles.')
    assert vmp.inject_alignment_header(nbsp) == vmp._ALIGNMENT_HEADER + '\n\nShe turns.'
    # Its own fixed point, on every shape.
    for p in typed + [l2va, late, glued, nbsp, vmp._ALIGNMENT_HEADER + '\n\nShe turns.']:
        once = vmp.inject_alignment_header(p)
        assert vmp.inject_alignment_header(once) == once, p
        assert once.count(vmp._ALIGNMENT_HEADER) == 1, p


def test_the_header_is_known_reflowed_a_blank_run_between_its_words():
    """Refuted on the exact spaces: the official line pasted from a mail, a
    terminal or a chat arrives reflowed — a no-break space for a space, two
    spaces, a line break after the timecode — and read for its exact spaces
    it was not the header: kept, and an image-to-video launch headed the
    text twice. Any blank run between the opening's words is the opening.
    The body still ends at a line break, on purpose: a typed line in the
    opening's English whose NEXT line names a picture is two sentences, not
    one header — the case the shape rule exists for."""
    official = vmp._ALIGNMENT_HEADER
    reflowed = [
        official.replace('For the', 'For\u00a0the'),
        official.replace('0.00 seconds', '0.00\u00a0seconds'),
        official.replace('the target video, at', 'the target  video, at'),
        official.replace('at 0.00 seconds', 'at 0.00\nseconds'),
        official.replace('For the target video,', 'For the target\nvideo,'),
    ]
    for h in reflowed:
        assert h != official
        text = h + '\n\nShe turns.'
        assert vmp.has_alignment_header(text), text
        assert vmp.strip_picture_references(text) == 'She turns.', text
        once = vmp.inject_alignment_header(text)
        assert once == official + '\n\nShe turns.', text
        assert once.count('is fully referenced') == 1, text
    end_frame = ('How the\u00a0reference pictures align  with the target\nvideo — <Picture 1> '
                 '(from [Shot 1]) aligns with the 5.00-second mark of the target video.\n\nShe turns.')
    assert vmp.has_alignment_header(end_frame)
    assert vmp.strip_picture_references(end_frame) == 'She turns.'
    assert vmp.inject_alignment_header(end_frame) == official + '\n\nShe turns.'
    # The opening's words with a blank run, and no picture in the sentence:
    # prompt English, kept whole — the line break after the timecode ends
    # the sentence before the picture named on the next line.
    typed = 'For the target video, at 3\nseconds she turns toward the camera.\nThe picture on the wall falls.'
    assert not vmp.has_alignment_header(typed)
    assert vmp.strip_picture_references(typed) == typed
    assert vmp.inject_alignment_header(typed) == official + '\n\n' + typed


def test_a_model_s_header_is_lifted_by_phrase_and_picture_and_written_official():
    """The writers copy the header from the text they enrich, and a copy may
    be paraphrased and land anywhere — so a MODEL's answer is still read by
    the phrase, but only in a sentence that names a picture: the phrase
    alone is prompt English, and read alone it took a description sentence
    for the header. What is lifted goes out as the official line, so the
    launch, which knows a header by its shape, never heads it twice."""
    para = ('At 0.00 seconds <Picture 1> is fully referenced.\n'
            'integrated_multimodal_description: [Shot 1] <Picture 1> turns toward the window.\n'
            'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    out = vmp.finish(para, with_image=True)
    assert out.startswith(vmp._ALIGNMENT_HEADER + '\n\n')
    assert out.count('fully referenced') == 1 and 'At 0.00 seconds <Picture 1>' not in out
    assert vmp.inject_alignment_header(out) == out
    # The same paraphrase inside the description is lifted the same way.
    inside = ('integrated_multimodal_description: [Shot 1] <Picture 1> turns toward the window. '
              'At 0.00 seconds <Picture 1> is fully referenced.\n'
              'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    out = vmp.finish(inside, with_image=True)
    assert out.startswith(vmp._ALIGNMENT_HEADER + '\n\n')
    assert out.count('fully referenced') == 1 and 'At 0.00 seconds <Picture 1>' not in out
    assert 'integrated_multimodal_description: [Shot 1] <Picture 1> turns toward the window.' in out
    # A sentence in the phrase's English that names no picture is description,
    # with a frame and without one.
    prose = ('integrated_multimodal_description: [Shot 1] She grades the shot so the tones '
             'align with the target video, then holds.\n'
             'overall_soundscape: N/A\nnon_diegetic_music: N/A')
    for with_image in (False, True):
        out = vmp.finish(prose, with_image=with_image)
        assert 'the tones align with the target video, then holds.' in out, with_image
        assert out.count(vmp._ALIGNMENT_HEADER) == (1 if with_image else 0), with_image


def test_the_identity_sentence_goes_however_the_model_reflowed_it():
    """Replaced as a literal, a copy the model broke across a line stayed —
    and its "<Picture 1>" then became "the subject", a sentence locking the
    subject to itself."""
    reflowed = ("integrated_multimodal_description: [Shot 1] The subject's identity, face and\n"
                "wardrobe are locked to <Picture 1>. She turns toward the window.\n"
                'overall_soundscape: N/A')
    assert vmp.strip_picture_references(reflowed) == (
        'integrated_multimodal_description: [Shot 1] She turns toward the window.\n'
        'overall_soundscape: N/A')


def test_a_prompt_has_motion_when_something_is_left_once_the_picture_talk_is_set_aside():
    """The launch's second look: the header, the identity sentence and the
    labels are not motion."""
    assert vmp.has_motion('she turns')
    assert vmp.has_motion(vmp._ALIGNMENT_HEADER
                          + '\n\nintegrated_multimodal_description: [Shot 1] <Picture 1> turns.')
    assert not vmp.has_motion(vmp._ALIGNMENT_HEADER)
    assert not vmp.has_motion(vmp._ALIGNMENT_HEADER + '\n\nintegrated_multimodal_description: '
                              '[Shot 1] ' + vmp._IDENTITY_SENTENCE)
    assert not vmp.has_motion('integrated_multimodal_description: [Shot 1]\noverall_soundscape: rain')
    assert not vmp.has_motion('')
