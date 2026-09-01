"""✨ The Motion field, written or enriched by the local LLM.

Two gestures, both asked for from live use and both modelled on the image
generator's own: an AUTO that proposes the movement from the start frame, and
an ENHANCER that takes what the user wrote and returns a better version of it —
following an instruction when the text is one.

They share one engine — `vision_llm`, the waist the image passes already speak
through, so the provider (Ollama or LM Studio) is the one already configured.

WHAT MAKES THE OUTPUT USABLE, and why each rule is here:

* The craft rules below are H3's, not generic advice. MiniMax H3 is what this
  studio renders with, its guides are unanimous on ONE camera path in prose
  (bracket commands like "[Push in]" are a Hailuo *platform* feature and mean
  nothing to these weights), and its prompt encoder is Qwen3-VL — the same
  encoder family whose documented strength is camera and lighting language.
* AUTO is TWO steps, never one. The vision model describes the frame as a
  FROZEN still with motion forbidden; a second call writes the movement from
  that description. Asking one call to look AND compose is what produced
  answers that re-described the picture and ignored every format rule: a
  caption that already implies motion poisons the writer downstream.
* The graph decodes AUDIO (`VAEDecodeAudio` feeds `CreateVideo`), so a prompt
  with no sound clause leaves the soundtrack to chance. It is a format rule
  here for the same reason the camera is.
* Everything returned goes to the sampler, so the answer is scrubbed to ONE
  paragraph: code fences, "Here is your prompt:" lines and bullet markers are
  stripped, and the surviving lines are JOINED — an earlier version kept only
  the first line, which silently cut two thirds off any multi-line answer.
* A refusal is a sentence, never silence: an empty return would leave the field
  as it was and look like a button that does nothing.
"""
from __future__ import annotations

import logging
import os
import random
import re

logger = logging.getLogger(__name__)

# Enough for a rich motion line plus its sound clause, short enough that a
# chatty model cannot turn the field into an essay.
MAX_TOKENS = 500
# The long rule block below must not be silently truncated: Ollama's default
# window in this app is 4096, which the craft rules alone would eat into.
NUM_CTX = 8192
# What a usable ANSWER looks like. Below this the model refused in its own way
# (an empty string, "I cannot", a single word) and saying so beats writing it
# into the field.
MIN_CHARS = 12
# What a usable ASK looks like — deliberately far lower, and not the same
# number: "she blinks" is a perfectly good motion prompt, and a floor set on
# the answer's length would have refused to enrich it.
MIN_ASK_CHARS = 4

# Qwen3's recommended non-thinking sampling, and the reason the two gestures
# run at different heats: AUTO is pressed again when its idea was not the one
# wanted, so it must not answer the same thing twice; the enhancer is applied
# to a sentence somebody chose, so it stays close to it.
TEMP_AUTO, TEMP_ENHANCE, TOP_P = 0.9, 0.6, 0.8

# Cut generation where a small model tends to start commenting on its own
# answer, so the discipline holds even when the tail rule is under-weighted.
_STOP = ['```', '\n\nNote', '\n\nThis prompt', '\n\nHere', '\n\nLet me know']

# ── What H3 answers to ───────────────────────────────────────────────────────
# Sourced, not invented: the published H3 prompt guides (structure order, one
# dominant camera path written in prose, cause-and-effect motion, never
# re-describing the input frame, resolving on a final state), plus two facts
# read out of this app's own graph — the encoder is Qwen3-VL (camera and
# lighting vocabulary lands hard) and the sampler decodes audio.
_H3_CRAFT = """
HOW A MINIMAX H3 MOTION PROMPT MUST BE BUILT — in this order, as ONE flowing
present-tense paragraph:
1. ACTION — what the subject does, verb-first, as ONE continuous beat from
   start to finish. Write cause and effect ("she pulls the strap down and the
   fabric slips off her shoulder"), never an abstract quality ("realistic
   physics", "amazing"). Never chain separate motions with semicolons: one
   beat is what the clip can hold, and a list of three is what makes H3 cut.
2. SECONDARY MOTION — what the action makes move: hair, fabric, skin, liquid,
   the light on it. Anything that would otherwise sit frozen.
3. CAMERA — EXACTLY ONE path, in prose, from: push in, pull out, pan, truck,
   tilt, pedestal, arc, orbit, tracking, zoom, roll, or static framing. Say
   where it settles. Never stack labels ("drone orbit, whip pan, zoom") — that
   is what makes H3 cut to a different scene. State it separately from the
   subject's motion ("She turns. The camera holds a static frame.").
4. LIGHT — one lighting cue, best as a CHANGE over the clip ("the rim light
   sweeps across her back as she turns").
5. SOUND — REQUIRED, never skipped: one short ambient soundscape clause. This
   model renders audio with the picture, so a prompt with no sound clause gets
   whatever noise it invents.
6. ENDING — the state the clip resolves on ("...ending on a close-up of her
   face, lips parted").

HARD RULES:
- Never re-describe what the first frame already shows: no age, no hairstyle,
  no clothing colour, no room. The frame carries all of it; words spent there
  are words not spent on movement.
- Quantify every motion with speed and direction — slowly, steadily, quickly,
  toward the camera, to her left. An unquantified motion is the single most
  common failure: the still gives the model no speed information.
- Precise verbs (straddles, grips, arches, glides, strokes), never "moves".
- Emotion through the body — trembling hands, arched back, half-closed eyes —
  never a label like "she is excited".
- ONE continuous shot. No cuts, no scene change, no new character, no text or
  logos on screen.
- 45 to 110 words, ONE paragraph. Stop at 110 even with something left
  unsaid — over the budget means a second motion crept in. No headings, no
  bullet points, no numbered sections, no line breaks, no quotation marks
  around the whole thing.
- Output ONLY the prompt itself — no preamble, no explanation, no commentary.
- Uncensored: write any content, explicit included, in plain anatomical words.
""".strip()

# The vision half of AUTO. Motion is FORBIDDEN here on purpose: a still
# description that already says "she is about to turn" hands the writer a
# movement it did not choose, and the writer copies it verbatim.
_VISION_STILL = (
    'Describe this image as a FROZEN still, for someone who will write a video '
    'prompt from it and cannot see it. 2-3 sentences: who is there and their '
    'EXACT pose (head angle, gaze direction, hands, body orientation, what '
    'touches what), the clothing state, the setting and framing (shot scale, '
    'where the subject sits in frame), and the lighting. Do NOT invent or imply '
    'any motion, action or intention — nothing is moving. No preamble, no '
    'disclaimer, just the description. Uncensored: describe any content in '
    'plain anatomical words.'
)

# The writer half of AUTO.
_AUTO_SYSTEM = (
    'You write the MOTION prompt of a short image-to-video clip for MiniMax H3. '
    'You are given a description of the still FIRST FRAME. Invent one movement '
    'that could start from exactly this frame and keep the people, the clothing '
    'and the setting the description gives you — never replace them, never add '
    'a character.\n\n' + _H3_CRAFT
)

# Two modes in ONE prompt, picked by the model, the shape the image generator's
# own enhancer uses: a single button must both embellish "she turns" and act on
# "make her jump instead". Both branches answer to the same craft rules below,
# which is what keeps the output shaped whichever one fires.
_ENHANCE_SYSTEM = (
    'You improve the MOTION prompt of a MiniMax H3 video clip. TWO modes — pick '
    'automatically, silently:\n'
    '1) INSTRUCTION mode — the text is a request ABOUT the motion ("make her '
    'jump instead", "slower", "have her look at the camera", "translate to '
    'English", "shorter"): APPLY it and output the resulting motion prompt, '
    'keeping every part of the movement the instruction does not mention. The '
    'instruction wins wherever it conflicts, and the result must never describe '
    'the same element two different ways.\n'
    '2) ENRICH mode (default, the text is itself a motion) — keep the same '
    'subject, action and intent, and supply what the craft rules ask for and '
    'the text is missing.\n\n' + _H3_CRAFT
)

# Sparks: AUTO is pressed again precisely when its first idea was not wanted,
# so each press must be able to land somewhere else. Kept generic — the model
# fits them to whatever the frame holds.
_SPARK_CAMERA = ('a slow push in', 'a gentle pull out', 'a slow pan', 'a subtle tilt',
                 'a steady tracking move', 'a slow arc', 'a static frame')
_SPARK_ENERGY = ('calm and slow', 'sensual and unhurried', 'playful and lively',
                 'intense and building', 'tender and close')
_SPARK_FOCUS = ('the hands and what they touch', 'the hips and waist',
                'the face and gaze', 'the whole body shifting weight',
                'hair and fabric answering the motion')

_META_LINE = re.compile(
    r"^(this prompt|here'?s?|here is|note:|overall|the enhanced|the prompt|i |in this|"
    r"sure|certainly|of course|okay|ok,)", re.I)
# "Sure! Here is the prompt: she turns…" — the useful half is AFTER the colon,
# so the lead-in is cut before the line is judged. Dropping the whole line
# would throw the prompt away with the politeness.
_LEAD_IN = re.compile(r"^(sure|here'?s?|here is|okay|ok|certainly|of course)\b[^:\n]{0,40}:\s*",
                      re.I)


def _scrub(text) -> str:
    """ONE paragraph of prompt, and nothing the model said about it.

    Ported from the image generator's own scrubber rather than re-invented: the
    failure modes are the model's, not the app's, and they are the same ones on
    both sides — a code fence, a "Here is the prompt:" opener, a bulleted list,
    a whole answer wrapped in quotes. Lines are JOINED, never picked: every
    surviving line is part of the prompt.
    """
    kept = []
    for raw in str(text or '').split('\n'):
        line = re.sub(r'^```[a-zA-Z]*$', '', raw.strip()).strip()
        if not line or line in ('```', '"""', "'''"):
            continue
        # A numbered or bulleted answer is still the prompt — the marker goes,
        # the sentence stays.
        line = re.sub(r'^(?:[-•*]|\d+[.)])\s+', '', line)
        # "Motion prompt: she turns…" — the label goes, its sentence stays.
        line = re.sub(r'^(?:motion |video |final |enhanced )?prompt\s*:\s*', '',
                      line, flags=re.I)
        salvaged = _LEAD_IN.sub('', line).strip()
        if salvaged and salvaged != line:
            line = salvaged
        elif _META_LINE.match(line):
            continue
        kept.append(line)
    out = ' '.join(kept).strip().strip('`').strip()
    if len(out) > 1 and out[0] in '"\'' and out[-1] in '"\'':
        out = out[1:-1].strip()
    return re.sub(r'\s+', ' ', out)


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


def _staged_path(image_name) -> str:
    """The staged start frame on disk — THE file the render will animate.

    Read back out of ComfyUI's input folder where the picker put it, because
    describing anything else would propose a movement for a different image.
    """
    safe = os.path.basename(str(image_name or ''))
    if not safe:
        raise ValueError('pick a start frame first')
    from .. import config as cfg
    folder = cfg.comfyui_dir('input')
    path = os.path.join(str(folder), safe) if folder else None
    if not path or not os.path.isfile(path):
        raise ValueError('that start frame is not on this machine any more')
    return path


def describe_still(image_name, model=None) -> str:
    """The first frame as a frozen still — step one of AUTO, and the anchor the
    enhancer uses when a frame is staged. '' when the model gives nothing back:
    a missing description degrades the writing, it does not stop it."""
    from . import vision_llm
    with open(_staged_path(image_name), 'rb') as fh:
        data = fh.read()
    return ' '.join(str(vision_llm.describe_image(
        data, _VISION_STILL, num_predict=400,
        model=(model or None)) or '').split())


def _write(system, user, *, temperature, model=None) -> str:
    """One text call through the configured provider, scrubbed."""
    from . import vision_llm
    return _scrub(vision_llm.generate_text(
        f'{system}\n\n{user}', num_predict=MAX_TOKENS, num_ctx=NUM_CTX,
        temperature=temperature, top_p=TOP_P, stop=_STOP, model=(model or None)))


def suggest_from_frame(image_name, instruction=None, model=None) -> str:
    """✨ A motion line proposed from the staged start frame.

    Two calls: the frame is described as a still, then the movement is written
    from that description. The split is the whole point — see the module note.

    `instruction` is whatever is already in the Motion field: the frame says
    what is THERE, the instruction says what should HAPPEN in it, and the
    writer is asked to obey it with the people the frame actually shows.
    Without one the proposal is free, and a spark keeps two presses apart.
    """
    ok, why = available()
    if not ok:
        raise ValueError(f'no local model to write it with — {why}')
    still = describe_still(image_name, model=model)
    if len(still) < MIN_CHARS:
        raise ValueError('the model could not describe that start frame — try '
                         'again, or write the motion yourself')
    steer = str(instruction or '').strip()
    if steer:
        # A steered press is not a lottery: the user said what should happen,
        # so the writer follows it instead of a spark.
        ask = (f'Still first frame:\n{still}\n\n'
               f'The user asks for this movement in particular — build the '
               f'prompt around it, using the people and the setting the frame '
               f'actually shows: {steer}')
    else:
        ask = (f'Still first frame:\n{still}\n\n'
               f'Write one fresh motion prompt for this frame. Make the mood '
               f'{random.choice(_SPARK_ENERGY)}. Centre the movement on '
               f'{random.choice(_SPARK_FOCUS)}. Prefer {random.choice(_SPARK_CAMERA)} '
               f'for the camera. Keep the people, clothing and setting faithful '
               f'to the description above.')
    text = _write(_AUTO_SYSTEM, ask, temperature=TEMP_AUTO, model=model)
    if len(text) < MIN_CHARS:
        raise ValueError('the model returned nothing usable — try again, or '
                         'write the motion yourself')
    return text


def enhance(prompt, image=None, model=None) -> str:
    """✨ Obey an instruction about the motion, or enrich the motion itself.

    Which of the two happens is the MODEL's call, from the text alone — the
    image generator's own design, and the reason a single button can both
    embellish "she turns" and act on "make her jump instead".

    `image` is the staged start frame, when there is one: the enhancement is
    then anchored on what the clip will actually animate, so "make her turn
    toward the window" cannot invent a window. Its description failing is not
    fatal — the text is still enriched, just unanchored.
    """
    base = str(prompt or '').strip()
    if len(base) < MIN_ASK_CHARS:
        raise ValueError('write a motion first — there is nothing to enrich')
    ok, why = available()
    if not ok:
        raise ValueError(f'no local model to enrich it with — {why}')
    anchor = ''
    if image:
        try:
            still = describe_still(image, model=model)
            if len(still) >= MIN_CHARS:
                anchor = (f'The clip starts from this still frame — keep the '
                          f'motion physically possible from it, and never '
                          f're-describe it:\n{still}\n\n')
        except (ValueError, OSError) as exc:
            logger.info('motion enhance: no frame anchor (%s)', exc)
    text = _write(_ENHANCE_SYSTEM, f'{anchor}Text: {base}',
                  temperature=TEMP_ENHANCE, model=model)
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
