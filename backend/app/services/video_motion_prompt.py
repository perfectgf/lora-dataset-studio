"""✨ The Motion field, written or enriched by the local LLM.

Two gestures, both asked for from live use (2026-09-01) and both modelled on the
image generator's own: an AUTO that proposes the movement by looking at the
start frame, and an ENHANCER that takes what the user wrote and returns a
fuller version of the same intent.

They share one engine — `vision_llm`, the waist the image passes already speak
through, so the provider (Ollama or LM Studio) and the model are the ones
already configured, and there is no second place to set them.

WHAT THEY MAY NOT DO, and the reason each rule exists:

* AUTO looks at a STILL. It can see who is there, how they are posed and what
  the light is doing; it cannot see what happens next, because nothing has
  happened yet. So it is asked for a movement that would START from this frame,
  and the UI calls it a proposal — never a description of the clip.
* Neither may invent a subject. The prompt says to keep the person and the
  setting the frame actually shows: a suggestion that renames the subject would
  send the render somewhere the user did not ask for.
* The ENHANCER returns ONE line, and never a preamble, a list, or a commentary
  about the prompt. Everything it returns is sent to the sampler.
* A refusal is a sentence, never silence: an empty return would leave the field
  as it was and look like a button that does nothing.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# Enough for a rich motion line, short enough that a chatty model cannot turn
# the field into an essay — the sampler reads a prompt, not a paragraph.
MAX_TOKENS = 220
# What a usable ANSWER looks like. Below this the model refused in its own way
# (an empty string, "I cannot", a single word) and saying so beats writing it
# into the field.
MIN_CHARS = 12
# What a usable ASK looks like — deliberately far lower, and not the same
# number: "she blinks" is a perfectly good motion prompt, and a floor set on
# the answer's length would have refused to enrich it.
MIN_ASK_CHARS = 4

_AUTO_PROMPT = (
    'This is the FIRST FRAME of a short video clip. Write one sentence, in '
    'English, describing a movement that could start from exactly this frame: '
    'who or what moves, which way, and how fast. Keep the person, the clothing '
    'and the setting the frame actually shows — never replace them. Describe '
    'MOVEMENT, not the picture: what is already visible needs no describing. '
    'No camera directions, no preamble, no quotation marks. Under 40 words.'
)

# TWO MODES, picked by the model itself — the shape the image generator's own
# enhancer uses (ENHANCE_SYSTEM_PROMPT there), ported because it is what makes
# the button obey rather than only embellish: typing "make her jump instead" and
# pressing ✨ should DO that, not decorate the sentence that says something else.
_ENHANCE_PROMPT = (
    'You improve the MOTION prompt of a video clip. You have TWO modes — pick '
    'automatically:\n'
    '1) INSTRUCTION mode — when the text is a request ABOUT the motion ("make '
    'her jump instead", "slower", "have her walk toward the camera", '
    '"translate to English"): FOLLOW it and output the resulting motion '
    'prompt, keeping every part of the movement the instruction does not '
    'mention.\n'
    '2) ENRICH mode (default, when the text is just a motion) — keep the same '
    'subject, action and intent, and say it with more of the detail a sampler '
    'can use: which limb or object moves, in which direction, how fast, what '
    'it touches.\n'
    'One sentence, English, under 50 words. No camera directions. Output ONLY '
    'the resulting prompt — no preamble, no quotes, no commentary.\n\n'
    'Text: '
)

# What ✨ Auto does when the field is NOT empty: the frame says what is there,
# the user's line says what should happen in it. Steering rather than replacing
# is the whole difference between a suggestion and a tool.
_AUTO_INSTRUCTED = (
    '\nThe user asks for this movement in particular — follow it, using the '
    'people and the setting the frame actually shows: '
)


def _clean(text) -> str:
    """One line, no quotes, no lead-in — what the sampler will actually read."""
    out = str(text or '').strip()
    # A model that answers "Sure! Here is the prompt: …" gives the useful half
    # after the colon of its own first clause.
    out = re.sub(r'^(sure|here(\'s| is)|prompt)\b[^:\n]{0,40}:\s*', '', out,
                 flags=re.I)
    out = out.strip().strip('"').strip("'").strip()
    # Some models answer in several lines; the first non-empty one is the
    # prompt, the rest is commentary about it.
    for line in out.split('\n'):
        line = line.strip().lstrip('-•*').strip()
        if len(line) >= MIN_CHARS:
            return line
    return out


def available() -> tuple[bool, str]:
    """(usable, why-not) for the local LLM behind both gestures.

    The same probes the caption backend gate uses, so an install where the
    image passes work has these too, and one where they do not says the same
    sentence in both places rather than two different ones.
    """
    from .. import capabilities
    from . import vision_llm
    provider = vision_llm.provider()
    probe = (capabilities.probe_lmstudio_model() if provider == 'lmstudio'
             else capabilities.probe_ollama_model())
    if probe.get('ok'):
        return True, ''
    return False, f"{vision_llm.label(provider)}: {probe.get('detail') or 'not ready'}"


def suggest_from_frame(image_name, instruction=None, model=None) -> str:
    """✨ A motion line proposed from the staged start frame.

    Reads the picture back out of ComfyUI's input folder — where the picker
    staged it — because that file IS what the render will animate; describing
    anything else would propose a movement for a different image.

    `instruction` is whatever is already in the Motion field: the frame says
    what is THERE, the instruction says what should HAPPEN in it, and the model
    is asked to obey it with the people the frame actually shows. Without one
    the proposal is free.
    """
    from .. import config as cfg
    from . import vision_llm
    ok, why = available()
    if not ok:
        raise ValueError(f'no local model to write it with — {why}')
    safe = os.path.basename(str(image_name or ''))
    if not safe:
        raise ValueError('pick a start frame first')
    folder = cfg.comfyui_dir('input')
    path = os.path.join(str(folder), safe) if folder else None
    if not path or not os.path.isfile(path):
        raise ValueError('that start frame is not on this machine any more')
    with open(path, 'rb') as fh:
        data = fh.read()
    ask = _AUTO_PROMPT
    steer = str(instruction or '').strip()
    if steer:
        ask = ask + _AUTO_INSTRUCTED + steer
    text = _clean(vision_llm.describe_image(data, ask, num_predict=MAX_TOKENS,
                                            model=(model or None)))
    if len(text) < MIN_CHARS:
        raise ValueError('the model returned nothing usable — try again, or '
                         'write the motion yourself')
    return text


def enhance(prompt, model=None) -> str:
    """✨ Obey an instruction about the motion, or enrich the motion itself.

    Which of the two happens is the MODEL's call, from the text alone — the
    image generator's own design, and the reason a single button can both
    embellish "she turns" and act on "make her jump instead"."""
    from . import vision_llm
    base = str(prompt or '').strip()
    if len(base) < MIN_ASK_CHARS:
        raise ValueError('write a motion first — there is nothing to enrich')
    ok, why = available()
    if not ok:
        raise ValueError(f'no local model to enrich it with — {why}')
    text = _clean(vision_llm.generate_text(_ENHANCE_PROMPT + base,
                                           num_predict=MAX_TOKENS,
                                           model=(model or None)))
    if len(text) < MIN_CHARS:
        # The original is returned rather than an error: an enhancement that
        # could not be made must never cost the user the prompt they wrote.
        logger.warning('motion enhance: unusable answer, keeping the original')
        return base
    return text

# ── Which model writes it ────────────────────────────────────────────────────
# Its own setting, and not the image passes' `vision_model`: those two answer
# different questions on the same machine (one describes a photo for a caption,
# this one writes a movement for a sampler), and a user who tunes one must not
# silently re-point the other. Empty = whatever the provider's vision model
# already is, so an install that sets nothing behaves exactly as before.
_MODEL_KEY = 'video_caption.motion_model'


def configured_model() -> str:
    from .. import config as cfg
    return (cfg.get(_MODEL_KEY) or '').strip()


def model_choices() -> dict:
    """{provider, label, current, models, reachable} — what the ⚙ window shows.

    The list is the provider's own (vision_llm.list_models, the same one every
    other picker in this app reads), so a model pulled in Ollama appears here
    without a second registry to keep in step. An unreachable server answers
    `reachable: False` with an empty list rather than an error: the window then
    says so and keeps the current choice visible.
    """
    from . import vision_llm
    listed = vision_llm.list_models() or {}
    return {'provider': listed.get('provider') or vision_llm.provider(),
            'label': vision_llm.label(listed.get('provider')),
            'reachable': bool(listed.get('reachable')),
            'current': configured_model(),
            'models': list(listed.get('models') or [])}


def set_model(name) -> str:
    """Remember which model writes the motion. '' returns to the provider's own.

    Never validated against the list: a model can be pulled between the moment
    the window was opened and the moment it is saved, and refusing a name this
    app simply had not heard of yet would be a lie about what the server holds.
    """
    from .. import config as cfg
    value = str(name or '').strip()
    cfg.save_config({'video_caption': {'motion_model': value}})
    return value
