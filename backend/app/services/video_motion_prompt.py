"""✨ The Motion field, written or enriched by the local LLM.

Three callers, all modelled on the image generator's: an AUTO that proposes
the clip from the start frame, an ENHANCER that takes what the user wrote and
returns a better version of it — following an instruction when the text is
one — and the launch itself ("Enrich at launch"), which is the enhancer run on
the prompt about to be rendered.

They share one engine — `vision_llm`, the waist the image passes already speak
through, so the provider (Ollama or LM Studio) is the one already configured.

WHAT MAKES THE OUTPUT USABLE, and why each rule is here:

* The prompt is written in H3's OFFICIAL format — MiniMax's own writing guide,
  the one the ComfyUI text encoder was built against: three labelled fields,
  `integrated_multimodal_description:`, `overall_soundscape:` and
  `non_diegetic_music:`, the shots marked "[Shot K] At MM:SS.mmm, the camera
  cuts to" inside the first. An earlier version wrote loose prose from the
  hosted platform's guides, which even claimed bracket markers meant nothing
  to these weights; the format the open weights were trained on is this one.
* The clip's LENGTH reaches the writer. Every caller sends the seconds the
  dials are set to, and a shot directive paces the action to fill exactly that.
  A 1 s clip and a 15 s clip are not the same clip, and a writer that does not
  know which one it is writing writes the same beat for both — the defect that
  opened this port ("✨ Auto ignores the length").
* AUTO is TWO steps, never one. The vision model describes the frame as a
  FROZEN still, motion forbidden — and told to describe the scene actually in
  front of it, because a VLM asked for a still has been measured inventing a
  different one; a second call writes the clip from that description. One call
  asked to look AND compose re-described the picture and ignored every rule.
* What the format REQUIRES is guaranteed in code, never hoped from the model:
  the fields on their own lines (the scrub flattens the answer to one line on
  purpose — meta-commentary is what it removes — and the fields are rebuilt
  from their labels), the "[Shot 1]" opener, the `<Picture 1>` identity tag
  (measured missing in half the answers on the image generator's side), the
  official I2V alignment header, and a field cut mid-sentence by the token
  budget loses its orphan tail.
* A text-to-video clip has NO picture: the writer is told so, and neither the
  tag nor the header is added — a `<Picture 1>` that names nothing lies to the
  encoder, which only prepends the picture block when a frame is given.
* The graph decodes AUDIO (`VAEDecodeAudio` feeds `CreateVideo`), which is why
  the two sound fields are mandatory rather than optional.
* A refusal is a sentence, never silence: an empty return would leave the field
  as it was and look like a button that does nothing.
"""
from __future__ import annotations

import logging
import math
import os
import random
import re

logger = logging.getLogger(__name__)

MAX_TOKENS = 500
NUM_CTX = 8192
MIN_CHARS = 12
MIN_ASK_CHARS = 4

TEMP_AUTO, TEMP_ENHANCE, TOP_P = 0.9, 0.6, 0.8
# The rest of the writer's recommended non-thinking sampling, sent explicitly
# because the driver's defaults are not these: top_k 20 and a presence penalty
# keep a small model from looping on the field labels, and `think` off stops a
# hybrid model from spending the token budget reasoning about the format
# instead of writing it. An instruct model ignores the flag at no cost.
TOP_K, MIN_P, PRESENCE_PENALTY = 20, 0.0, 1.0
THINK = False

_STOP = ['```', '\n\nNote', '\n\nThis prompt', '\n\nHere', '\n\nLet me know', '\n\nHope']

# The most shots a clip is cut into. The studio renders up to ~15 s, which is
# 2.5 s a shot at six — already short for a cut to bring genuinely new framing.
MAX_SHOTS = 6

# ── the craft ───────────────────────────────────────────────────────────────
# The official format first (the model was trained on it), then the rules that
# make the description field MOVE. Both gestures share it so a "better" prompt
# and a "fresh" prompt obey the same physics.
_H3_CRAFT = """OUTPUT FORMAT — the OFFICIAL MiniMax H3 prompt is EXACTLY three labelled fields, each starting on its own line, in this order (never merge them, never add other fields, never write an "Audio:" line):
integrated_multimodal_description: [Shot 1] <short style anchor, e.g. "Live-action, cinematic."> <everything visual that happens: subjects, action, camera, lighting> [Shot 2] At 00:05.000, the camera cuts to <the next shot> ...
overall_soundscape: <1-4 sentences: ambient atmosphere, action sounds, non-verbal human sounds (breathing, footsteps, fabric rustle). NEVER dialogue, singing or music here. Write "N/A" ONLY if explicit silence is requested>
non_diegetic_music: <the background score as instrumentation + tempo + dynamics, e.g. "Sparse piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume". NEVER abstract emotion words like "moody" or "tense". Write "N/A" if no music fits>

DESCRIPTION FIELD RULES:
- Open with "[Shot 1]" — Shot 1 NEVER takes a timestamp. Every later shot starts "[Shot K] At MM:SS.mmm, the camera cuts to ..." with strictly increasing timecodes — and there are later shots ONLY when the shot plan asks for them.
- Present tense, concrete physical cues, describe what HAPPENS: verb-first cause and effect ("she pulls the strap down and the fabric slips off her shoulder"), never an abstract quality ("amazing", "realistic physics") and never an emotion label — trembling hands, an arched back, half-closed eyes say it. Precise verbs (straddles, grips, arches, glides, strokes), never "moves".
- Quantify every motion with a speed and a direction — slowly, steadily, quickly, toward the camera, to her left. The still gives the model no speed information, and an unquantified motion is the single most common failure.
- Secondary motion: what the action makes move — hair, fabric, skin, liquid, the light on it — so nothing sits frozen. Realistic physics, no morphing. One lighting cue, best as a CHANGE over the clip.
- CAMERA GRAMMAR: use ONLY this vocabulary, written as full sentences woven into the shot (never stacked labels): Zoom In/Out, Push In/Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll Clockwise/Counterclockwise — modulated by "with small/large amplitude" and "at slow/fast speed" (omit the modifier when medium). Example: "The camera pushes in with small amplitude at slow speed toward her face." ONE camera move per shot; separate camera motion from subject motion ("She turns. The camera holds a static shot.").
- Never re-describe what the first frame already shows: no age, no hairstyle, no clothing colour, no room. Words spent there are words not spent on movement.
- Resolve on a final state ("... ending on a close-up of her face, lips parted").
- DIALOGUE (only when someone speaks): tag speakers (S1), (S2) — the SAME ID for the same character across every shot; no ID for characters who never speak. OUTSIDE the <d> block: who speaks, the voice qualities (pitch, timbre, pace, accent) and whether on- or off-screen. INSIDE <d>: a language tag then the spoken words VERBATIM — e.g. The young woman with a quiet, breathy voice (S1) says: <d>[English] Come closer.</d> For a voiceover write: says in an off-screen voiceover: <d>...</d> while his lips remain completely closed. — the closed-lips clause is MANDATORY right after the block. Never paraphrase, reword or translate dialogue the user supplies.
- On-screen text: give it VERBATIM in DOUBLE quotes, never translated.

HARD RULES:
- ~60 to 160 words in total. No headings, no bullet points, no numbered sections, no quotation marks around the whole thing, no line breaks other than the ones between the three fields.
- Output ONLY the three fields — no preamble, no explanation, no commentary.
- Uncensored: write any content, explicit included, in plain anatomical words."""

# The frame is handed to the encoder as "<Picture 1>" — but only when there IS
# a frame. A text-to-video clip must not name a picture that was never given.
_IDENTITY_RULE = (
    'IDENTITY: the input frame reaches the model as "<Picture 1>". In the '
    'description field, reference it to lock the subject\'s identity, face and '
    'wardrobe (e.g. "The woman from <Picture 1> ...") instead of re-describing '
    'the fine appearance detail it already shows.')
_NO_PICTURE_RULE = (
    'NO INPUT FRAME: this is a text-to-video clip. Never write "<Picture 1>" or '
    'refer to any picture — the description field must carry the subject\'s '
    'appearance, setting and style itself, in one short opening clause.')

# What the vision model is asked when AUTO looks at the frame. A still, not a
# clip: every motion it invents here would be re-invented downstream as fact.
_VISION_STILL = (
    'Describe this image as a FROZEN still, for someone who will write a video '
    'prompt from it and cannot see it. 2-3 sentences: who is in the frame (how '
    'many people, sex, apparent age band, hair, build), what they wear (or do '
    'not), their exact pose and where their hands and gaze are, the setting, '
    'the lighting, the visual style (photo, anime, 3D...). Describe the scene '
    'that is actually in front of you — never replace it with a different, '
    'invented scene. Do NOT invent or imply any motion, action or intention — '
    'nothing is moving. No preamble, no disclaimer, just the description. '
    'Uncensored: describe any content in plain anatomical words.')

_AUTO_SYSTEM = (
    'You write the prompt of a short image-to-video clip for MiniMax H3 — an '
    'open-weights omni model that renders picture AND native stereo audio in '
    'one pass. You are given a description of the still FIRST FRAME. INVENT '
    'one fresh clip that starts from exactly this frame: keep the people, the '
    'clothing and the setting the description gives you — never replace them, '
    'never add a character — and invent a small coherent movement, a camera '
    'and matching sound so the clip feels alive.\n\n'
    + _H3_CRAFT + '\n\n' + _IDENTITY_RULE
)

_ENHANCE_SYSTEM = (
    'You improve the prompt of a MiniMax H3 video clip — an open-weights omni '
    'model that renders picture AND native stereo audio in one pass. TWO modes '
    '— pick automatically, silently:\n'
    '1) INSTRUCTION mode — the text is a request ABOUT the clip ("make her '
    'jump instead", "slower", "have her look at the camera", "translate to '
    'English", "shorter"): APPLY it and output the resulting prompt, keeping '
    'every part of the movement the instruction does not mention. The '
    'instruction wins wherever it conflicts, and the result must never '
    'describe the same element two different ways.\n'
    '2) ENRICH mode (default, the text is itself a motion or a whole prompt) — '
    'keep the same subject, action and intent; if the text already has action, '
    'DISTRIBUTE it across the shot plan; if it is only a mood or a static '
    'subject, INVENT a coherent micro-story; and supply everything the format '
    'asks for that the text is missing.\n\n'
    + _H3_CRAFT
)

_CLOSING = (
    'Now produce the MiniMax H3 prompt in the OFFICIAL three-field format — '
    'integrated_multimodal_description:, overall_soundscape:, '
    'non_diegetic_music: — following the shot plan below.')

# Sparks — one of each is drawn per free press, so six presses give six clips
# instead of the model's single favourite. The steered ask carries none: the
# user's words are the spark.
_SPARK_CAMERA = (
    'a slow push in', 'a gentle pull out', 'a slow pan', 'a subtle tilt',
    'a steady tracking move', 'a slow arc', 'a static frame',
)
_SPARK_ENERGY = (
    'calm and slow', 'sensual and unhurried', 'playful and lively',
    'intense and building', 'tender and close',
)
_SPARK_FOCUS = (
    'the hands and what they touch', 'the hips and waist', 'the face and gaze',
    'the whole body shifting weight', 'hair and fabric answering the motion',
)


# ── the shot plan ───────────────────────────────────────────────────────────

def clip_seconds(seconds) -> int:
    """The clip length the way the directive states it: whole seconds, at least
    one when the caller knows the length, zero when it does not (the directive
    then paces nothing). Rounded, not floored — 0.88 s (22 frames at 24 fps) is
    a one-second clip and 15.04 s is fifteen."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(s) or s <= 0:  # NaN and the infinities included
        return 0
    return max(1, int(round(s)))


def shot_count(shots, seconds: int) -> int:
    """How many shots the plan asks for: clamped to [1, MAX_SHOTS], and never
    more than one per second on a clip of four seconds or less — a cut every
    0.7 s is a flicker, not a montage (six shots on four seconds is one every
    0.67 s; five seconds carry six)."""
    try:
        n = int(shots)
    except (TypeError, ValueError):
        n = 1
    n = max(1, min(n, MAX_SHOTS))
    if 1 <= seconds <= 4:
        n = min(n, seconds)
    return n


def shot_cut_marks(seconds: int, count: int) -> str:
    """The official cut timecodes for `count` shots over `seconds`, evenly
    spaced and strictly increasing: 10 s in 3 shots → "00:03.300, 00:06.700".
    Written for the model, so it copies them instead of inventing a timeline."""
    marks = []
    for i in range(1, max(2, int(count))):
        t = round(i * seconds / count, 1)
        if marks and t <= marks[-1]:
            t = marks[-1] + 0.5
        marks.append(t)
    return ', '.join(f'{int(t // 60):02d}:{t % 60:06.3f}' for t in marks)


def _pacing_hint(d: int) -> str:
    """How much can HAPPEN in `d` seconds. Measured without it: a 2 s clip
    and a 15 s clip got the same four-beat sequence — the length reached the
    writer and changed nothing, because "fill the full 2s" does not say that
    two seconds hold one gesture. Nothing for the middle range: three beats in
    six seconds is what the craft rules already produce."""
    if d <= 3:
        return (f' {d}s holds ONE movement: a single gesture or a single camera '
                'move carried from its start to its end state — not a sequence '
                'of beats.')
    if d >= 8:
        return (f' {d}s is a long take: write a sequence of successive beats, '
                'each flowing into the next, with enough distinct action to '
                f'fill {d}s without repeating a movement.')
    return ''


def shot_directive(seconds=None, shots=1) -> str:
    """The paragraph that tells the writer HOW LONG the clip is and how many
    shots to cut it into. This is the whole reason the length is plumbed from
    the panel: without it the model paces every clip the same way."""
    d = clip_seconds(seconds)
    n = shot_count(shots, d)
    unit = 'second' if d == 1 else 'seconds'
    if n <= 1:
        if d:
            return (
                f'The clip is {d} {unit} long. Inside the '
                'integrated_multimodal_description: field, write ONE single '
                f'continuous shot pacing the action to fill the full {d}s: open '
                'with "[Shot 1]" (no timestamp) and never write "the camera cuts '
                f'to" — no cuts.{_pacing_hint(d)}')
        return (
            'Inside the integrated_multimodal_description: field, write ONE '
            'single continuous shot: open with "[Shot 1]" (no timestamp) and '
            'never write "the camera cuts to" — no cuts.')
    tail = (
        ' Use the exact words "the camera cuts to" — the model only cuts when '
        'the text says so. A cut must bring genuinely NEW framing, viewpoint or '
        'subject state; never describe the camera as locked or static for the '
        'whole clip. Keep the SAME character identity, wardrobe and location '
        'across every shot; the overall_soundscape: and non_diegetic_music: '
        'fields describe the WHOLE clip and carry across the cuts.')
    if d:
        return (
            f'The clip is {d} {unit} long. Structure the '
            f'integrated_multimodal_description: field as EXACTLY {n} shots in '
            'the OFFICIAL multi-shot format: open with "[Shot 1]" (no timestamp) '
            'describing the first framing; then start each following shot with '
            '"[Shot K] At <timecode>, the camera cuts to" a NEW framing/angle. '
            f'Use EXACTLY these cut timecodes, in order: {shot_cut_marks(d, n)}.'
            + tail)
    return (
        f'Structure the integrated_multimodal_description: field as EXACTLY {n} '
        'shots in the OFFICIAL multi-shot format: open with "[Shot 1]" (no '
        'timestamp) describing the first framing; start every following shot '
        'with "[Shot K] At 00:0X.XXX, the camera cuts to" a NEW framing/angle, '
        'with strictly increasing timecodes inside the clip.' + tail)


# ── output hygiene ──────────────────────────────────────────────────────────
# Lines that are the model talking ABOUT the prompt rather than writing it.
# `overall(?!_)`: "Overall, the scene..." is chatter, "overall_soundscape:" is
# a field — the unguarded word swallowed the field in an earlier version.
_META_LINE = re.compile(
    r"^(this prompt|here'?s?|here is|note:|overall(?!_)|the enhanced|the prompt|i |in this"
    r"|sure|certainly|of course|okay|ok,|below (?:is|are)|output:|let me know"
    r"|hope (?:this|that|it|you)|feel free|if you'?d like|as an ai|sorry|i'?m sorry"
    r"|unfortunately)", re.I)
# A lead-in that carries the prompt on the same line ("Sure, here it is: She…")
# is salvaged, not dropped: the text after the colon is the answer.
_LEAD_IN = re.compile(
    r"^(sure|here'?s?|here is|okay|ok|certainly|of course|below)\b[^:\n]{0,40}:\s*", re.I)
# A line that is only a delimiter: a code fence, a docstring quote, a rule.
_DELIMITER_LINE = re.compile(r'```[\w+-]*|"""|\'\'\'|-{3,}|={3,}')
# A hybrid model's reasoning, when the provider hands it back inline — the
# `think` switch travels to Ollama, not to LM Studio's chat endpoint. Two
# dialects: the block with both tags (cut open by the budget it runs to the
# end — the answer never came), and the one whose template opens the tag in
# the prompt, so the output is the reasoning and a bare `</think>` before the
# answer.
_THINK_BLOCK = re.compile(r'(?is)<think>.*?(?:</think>|\Z)')
_BARE_THINK_CLOSE = re.compile(r'(?is)^.*</think>\s*')
_LABEL_RE = r'(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music)'
# A label written with markdown emphasis ("**overall_soundscape:**") is still
# the label; left alone, the stars became the field's first word.
_EMPHASISED_LABEL = re.compile(rf'(?i)[*_]{{1,3}}\s*({_LABEL_RE})\s*:\s*[*_]{{0,3}}\s*')


def _drop_reasoning(text: str) -> str:
    if '<think>' in text.lower():
        return _THINK_BLOCK.sub(' ', text)
    return _BARE_THINK_CLOSE.sub('', text)


def _scrub(text: str) -> str:
    """One line of prose from whatever the model wrapped it in: fences, list
    markers, a "Prompt:" label, a chatty lead-in, a trailing note. The fields
    are rebuilt from their labels afterwards, which is why flattening is safe."""
    lines = []
    for raw in _drop_reasoning(text or '').splitlines():
        if _DELIMITER_LINE.fullmatch(raw.strip()):
            continue
        line = _EMPHASISED_LABEL.sub(r'\1: ', raw.replace('**', ''))
        line = line.strip().strip('`').strip()
        if not line:
            continue
        if re.match(r'^#{1,6}\s', line):
            # A markdown heading is a title over the answer, not a shot —
            # unless the model put a label in it.
            if not re.search(_LABEL_RE, line, flags=re.I):
                continue
            line = re.sub(r'^#{1,6}\s+', '', line)
        line = re.sub(r'^(?:[-•*]|\d+[.)])\s+', '', line)
        line = re.sub(r'^(?:motion |video |final |enhanced )?prompt\s*:\s*', '', line, flags=re.I)
        m = _LEAD_IN.match(line)
        if m:
            line = line[m.end():].strip()
            if not line:
                continue
        elif _META_LINE.match(line):
            continue
        lines.append(line)
    out = ' '.join(lines).strip().strip('`').strip()
    if len(out) > 1 and out[0] == out[-1] and out[0] in '"\'':
        out = out[1:-1].strip()
    return re.sub(r'\s+', ' ', out)


def _purge_hybrid(text: str) -> str:
    """A model that half-remembers another dialect writes "[Shot 1] At
    00:00.000", "Timeline:", "[0s-5s] [Shot 2]" or folds the timecode inside
    the bracket. Each is mapped back to the official grammar; a text without
    shot markers is left alone."""
    if '[Shot' not in text:
        return text
    out = re.sub(r'(\[Shot\s*1\]\s*)At\s+00[:.]00[:.]000\s*,?\s*', r'\1', text, flags=re.I)
    out = re.sub(r'\bTimeline\s*:\s*', '', out)
    out = re.sub(r'\[\d+(?:\.\d+)?s(?:\s*-\s*\d+(?:\.\d+)?s)?\]\s*(?=\[Shot)', '', out, flags=re.I)
    out = re.sub(r'\[Shot\s*(\d+)\s+At\s+([0-9:.]+)\s*,\s*the camera cuts to\s*\]',
                 r'[Shot \1] At \2, the camera cuts to', out, flags=re.I)
    return out


_ALIGNMENT_HEADER = (
    'For the target video, at 0.00 seconds into the target video, '
    '<Picture 1> (from [Shot 1]) is fully referenced.')
_IDENTITY_SENTENCE = "The subject's identity, face and wardrobe are locked to <Picture 1>."


# A character that does not end a sentence: the dot in "0.00 seconds" is
# followed by a digit, not by a space or the end.
_NOT_END = r'(?:[^.!?\n]|[.!?](?!["\')\]]*(?:\s|$)))'
# The header is known by its SHAPE — the official opener at the start of a
# line, to its full stop — never by a phrase: "is fully referenced" and
# "align with the target video" are ordinary prompt English ("grade the shot
# so the tones align with the target video, then hold"), and read by the
# phrase a hand-typed prompt carrying one reached the sampler amputated
# around it, or empty. Two openers: the image-to-video line this app writes,
# and the end-frame line of the reference writer, which a prompt pasted from
# there carries. And the SENTENCE names a picture — the opening alone is
# prompt English too ("For the target video, at 3 seconds she turns"), and
# read on the opening alone a typed line went: a 400 for a prompt that was
# only it, a line silently gone from a longer one. In the sentence, not on
# the line: read for a picture anywhere on the line, "For the target video,
# at 3 seconds she turns toward the camera. The picture on the wall falls."
# lost its first sentence. The same test a MODEL's sentence passes
# (`_header_sentence`): the wording, and a picture in it. And the sentence
# starts where a sentence can — a line start, or the boundary after one —
# behind any blank but a newline (a no-break space included): the official
# line glued after a sentence, or pasted behind U+00A0, was not seen, and an
# image-to-video launch headed it twice. What follows the full stop is taken
# only up to and with a line break: removed from the middle of a line, the
# sentence leaves the space between its neighbours. Between the words of
# the opening, any blank run: a header pasted from a mail or a terminal
# arrives reflowed — a no-break space for a space, two spaces, a line break
# after the timecode — and read for its exact spaces it was not seen, and an
# image-to-video launch headed it twice. The body after the opening still
# ends at a line break: a typed line in the opening's English, with a
# picture named on its NEXT line, is not one sentence.
_HEADER_LINE = re.compile(
    rf'(?im)(?:^|(?<=[.!?:\]\n]))[^\S\n]*'
    rf'(?:For\s+the\s+target\s+video,\s+at\s+[0-9.]+\s+seconds?\b'
    rf'|How\s+the\s+reference\s+pictures\s+align\s+with\s+the\s+target\s+video\b)'
    rf'{_NOT_END}*?picture{_NOT_END}*[.!?]?["\')\]]*(?:[^\S\n]*\n+)?')


def has_alignment_header(text: str) -> bool:
    """Whether the text carries the official header — a sentence of its
    shape, wherever it starts, see `_HEADER_LINE`."""
    return bool(_HEADER_LINE.search(text or ''))


_HEADER_PHRASE = r'\bis fully referenced\b|\balign with the target video\b'
# The sentence carrying the header's phrase, wherever the model put it — its
# own line, or inside the description after the label — from the sentence
# end, label or shot marker before it to the phrase and its full stop. The
# phrase closes the official wording, so nothing after it is taken: a
# description glued to the header without a space stays description.
_HEADER_SENTENCE = re.compile(
    rf'(?is)(?:^|(?<=[.!?:\]\n]))\s*{_NOT_END}*?(?:{_HEADER_PHRASE})[.!?]?["\')\]]*\s*')


def _header_sentence(text: str):
    """The sentence a MODEL wrote as the header, wherever it put it — the
    official line copied from the text it enriched, or its own paraphrase of
    it — known by the header's phrase AND a picture named in the same
    sentence. The phrase alone is prompt English, and read on the phrase
    alone the lift took a description sentence for the header. What the
    lift takes goes out as the official line: the launch knows a header by
    that shape, and never heads it twice. (An end frame, when it comes,
    will want the numbers of the end-frame line kept.)"""
    for m in _HEADER_SENTENCE.finditer(text or ''):
        if 'picture' in m.group(0).lower():
            return m
    return None


# Where a sentence ends: a full stop followed by space or the end — so the
# dot inside a timecode ("00:05.000") or a decimal is not one.
_SENTENCE_END = re.compile(r'[.!?](?=["\')\]]*(?:\s|$))')
# A tail that stops on a word no English clause can end on — a determiner
# waiting for its noun, a coordinator waiting for its second half — is a cut
# whatever the budget said. Nothing else is: a final clause that merely lost
# its full stop ends on a preposition or a pronoun as often as on a noun
# ("settles behind her", "what she is looking at"), and the craft rules ask
# for exactly that vocabulary ("toward the camera", "to her left"). Measured:
# a wider list — prepositions, pronouns, "is/are" — amputated four of five
# legitimate closing clauses (two on "her", one on "at", one on "is").
_FRAGMENT_TAIL = re.compile(r'(?i)(?:^|\s)(?:a|an|the|and|or|nor)$')


def _trim_dangling(txt: str, *, truncated: bool = False) -> str:
    """A field the token budget cut mid-sentence ends on a fragment the model
    would render as a half-thought. Cut back to the last sentence end — but
    only when what remains is a real field, never down to a stub, and only
    when the tail IS a fragment: it hangs on joining punctuation or on a word
    no clause ends on (a determiner, a coordinator). A final clause that
    merely lost its full stop stays whatever its last word ("... ending on a
    close-up of her face", "... settles behind her"): a missing full stop is
    not proof of a cut — unless the answer hit the budget, where every
    unfinished tail is the cut.

    A field that carries content AND a trailing "N/A" (measured: the model
    copies the placeholder from the format block after a real soundscape)
    loses the placeholder, whatever its length."""
    txt = (txt or '').strip()
    if txt.upper() != 'N/A':
        # The comma that joined the placeholder goes with it — only that one:
        # a field's own trailing comma is the fragment signal read below.
        txt = re.sub(r'[\s,;]*\bN/A\b[.\s]*$', '', txt).strip()
    if not txt or txt[-1] in '.!?"\'>' or txt.upper() == 'N/A':
        return txt
    ends = [m.end() for m in _SENTENCE_END.finditer(txt)]
    if not ends:
        return txt
    tail = txt[ends[-1]:].strip()
    fragment = (truncated or tail.endswith((',', ';', ':', '-', '—', '–'))
                or bool(_FRAGMENT_TAIL.search(tail)))
    if not fragment:
        return txt
    kept = txt[:ends[-1]].strip()
    return kept if len(kept) > 40 else txt


def _split_header(pre: str) -> tuple[str, str]:
    """(header, rest) for the text before the first label: the official
    header when the model wrote one — in its own words too, `_header_sentence`
    — and whatever surrounds it: a description that lost its label, which an
    earlier version swallowed with the header, and a sentence written BEFORE
    the header, which a later one filed as header (it went out above the
    header instead of into the field)."""
    pre = (pre or '').strip()
    m = _header_sentence(pre) if pre else None
    if not m:
        return '', pre
    rest = f'{pre[:m.start()]} {pre[m.end():]}'
    return _ALIGNMENT_HEADER, re.sub(r'[ \t]{2,}', ' ', rest).strip()


def _lift_header(desc: str, header: str) -> tuple[str, str]:
    """A header the model wrote INSIDE the description — after the label,
    where the split before the first label cannot see it — moves to the
    header slot, or goes when one is there already. Left in the field it
    opened the description (the marker hoist then tore its "(from [Shot
    1])"), and the text-only strip, which looks for the header where the
    writer puts it, left it in as prose about a picture the encoder never
    gets."""
    for _ in range(4):
        m = _header_sentence(desc)
        if not m:
            break
        header = header or _ALIGNMENT_HEADER
        desc = re.sub(r'[ \t]{2,}', ' ', f'{desc[:m.start()]} {desc[m.end():]}').strip()
    return desc, header


def _lift_audio(desc: str) -> tuple[str, list[str]]:
    """The "Audio:" tails out of the description — the hosted dialect —
    each cut at the next shot marker: written inside a shot, the line is
    THAT shot's sound, and the shots after it stay picture (taken to the
    end, a two-shot plan lost its second shot to the soundscape). A
    placeholder tail ("Audio: N/A") is dropped, not carried."""
    tails = []
    while True:
        m = re.search(r'\bAudio\s*:\s*', desc)
        if not m:
            return desc, tails
        nxt = re.search(r'\[Shot\s*\d+\]', desc[m.end():], flags=re.I)
        end = m.end() + nxt.start() if nxt else len(desc)
        tail = desc[m.end():end].strip()
        desc = re.sub(r'[ \t]{2,}', ' ', f'{desc[:m.start()]} {desc[end:]}').strip()
        if tail and tail.upper() != 'N/A':
            tails.append(tail)


def _join_sound(sound: str, tails: list[str]) -> str:
    """The soundscape field plus the audio tails, joined — the last one
    keeps its own punctuation, the ones before it lose theirs."""
    parts = ([] if sound.strip().upper() in ('', 'N/A') else [sound]) + tails
    return ', '.join([p.rstrip(' .,;') for p in parts[:-1]] + [parts[-1]])


def restructure_fields(text: str, *, truncated: bool = False) -> str:
    """The three fields on their own lines, whatever the model's line breaks
    were: the scrub flattened the answer, this finds the labels again. Text
    before the first label is the alignment header when it is one, otherwise
    it is description that lost its label. An "Audio:" line — the dialect of
    the hosted platform — becomes the soundscape it was meant to be."""
    t = (text or '').strip()
    if not t:
        return t
    hits = list(re.finditer(rf'(?i)({_LABEL_RE})\s*:', t))
    if not hits:
        desc, sound, music, header = t, '', '', ''
    else:
        header, lead = _split_header(t[:hits[0].start()])
        fields = {}
        for i, h in enumerate(hits):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(t)
            key = h.group(1).lower()
            fields[key] = (fields.get(key, '') + ' ' + t[h.end():end].strip()).strip()
        desc = fields.get('integrated_multimodal_description', '')
        if lead:
            desc = f'{lead} {desc}'.strip()
        sound = fields.get('overall_soundscape', '')
        music = fields.get('non_diegetic_music', '')
    desc, header = _lift_header(desc, header)
    desc, tails = _lift_audio(desc)
    if tails:
        # The hosted dialect's "Audio:" tail is soundscape wherever it sits —
        # joined to the field when the model wrote that one as well.
        sound = _join_sound(sound, tails)
    desc, sound, music = (_trim_dangling(desc, truncated=truncated),
                          _trim_dangling(sound, truncated=truncated),
                          _trim_dangling(music, truncated=truncated))
    if not desc:
        return t
    m = re.search(r'(?i)(?:^|(?<=[.!?:\]]))\s*\[Shot\s*1\]\s*', desc)
    if not m:
        # No marker opening a sentence: the field gets one. A "[Shot 1]"
        # inside a sentence ("as set up in [Shot 1]") is prose that names
        # the shot, not the marker — hoisted, it left "(as set up in )".
        desc = f'[Shot 1] {desc}'
    elif m.start() > 0:
        # The marker is there with something in front of it: the marker moves
        # to the front and the text stays, rather than a second "[Shot 1]".
        desc = f'[Shot 1] {desc[:m.start()].strip()} {desc[m.end():].strip()}'.strip()
    lines = [f'integrated_multimodal_description: {desc}',
             f'overall_soundscape: {sound or "N/A"}',
             f'non_diegetic_music: {music or "N/A"}']
    body = '\n'.join(lines)
    return f'{header}\n\n{body}' if header else body


def _prefix_description(text: str, sentence: str) -> str:
    """Insert `sentence` at the head of the description field — after the
    label and after "[Shot 1]" when it is there — so the field still opens
    with its marker."""
    m = re.search(r'(?i)(integrated_multimodal_description\s*:\s*(?:\[Shot 1\]\s*)?)', text)
    if m:
        return f'{text[:m.end()]}{sentence} {text[m.end():]}'
    return f'{sentence} {text}'


def _description_field(text: str) -> str:
    """What the description field holds — the whole text when the labels are
    not there — without its label or its "[Shot 1]" opener."""
    m = re.search(r'(?is)integrated_multimodal_description\s*:\s*(.*?)'
                  r'(?=\n\s*(?:overall_soundscape|non_diegetic_music)\s*:|\Z)', text or '')
    body = m.group(1) if m else (text or '')
    return re.sub(r'(?i)^\s*\[Shot\s*1\]\s*', '', body).strip()


def ensure_identity_tag(text: str) -> str:
    """The prompt names the frame as <Picture 1> or the model has no anchor for
    who is in the clip — the tag is the ONE thing the encoder pairs with the
    picture block it prepends. Looked for in the description itself: the
    header names the picture too, and it is not the anchor."""
    if not text or 'Picture 1' in _description_field(text):
        return text
    return _prefix_description(text, _IDENTITY_SENTENCE)


# The identity sentence however the model reflowed it: replaced as a literal,
# a copy broken across a line stayed, and its "<Picture 1>" then became "the
# subject" — a sentence locking the subject to itself.
_IDENTITY_RE = re.compile(
    r'(?i)[ \t]*' + r'\s+'.join(map(re.escape, _IDENTITY_SENTENCE.split())) + r'[ \t]*')


def strip_picture_references(text: str) -> str:
    """A text-to-video prompt names no picture: the I2V header and the
    identity sentence go, and a stray "<Picture 1>" becomes the subject it
    stood for. The encoder prepends a picture block only when a frame is
    given, so a tag without one names nothing — and the case is real: a
    prompt enriched as image-to-video, then the panel switched to text-only.
    The header is the official sentence, by shape — its opening, naming a
    picture, wherever it starts: a prompt typed in the header's English, its
    opening included, keeps every sentence it has."""
    t = text or ''
    if not t or ('Picture 1' not in t and not has_alignment_header(t)):
        return t
    t = _HEADER_LINE.sub('', t)
    t = _IDENTITY_RE.sub(' ', t)
    t = re.sub(r'(?i)\s*\(?\bfrom <Picture 1>\)?', '', t)
    t = re.sub(r'(?i)<Picture 1>', 'the subject', t)
    t = re.sub(r'(^|[.!?]\s+|\]\s+)the subject', r'\1The subject', t)
    t = re.sub(r'[ \t]+\n', '\n', t)
    return re.sub(r'[ \t]{2,}', ' ', t).strip()


def inject_alignment_header(text: str) -> str:
    """The official I2V header, once: it tells the model the picture IS the
    first frame at 0.00 s, rather than a reference to resemble. A header
    already there — wherever its sentence starts — is replaced, not kept:
    the reference writer's end-frame line, pasted from there, says the
    picture is the LAST frame, the reverse of what this launch does — so the
    text carries one header, this one, and the call is its own fixed point.
    (When a last frame is wired into the workflow, its end-frame line must
    survive here instead of being replaced.)"""
    if not (text or '').strip():
        return text
    body = _HEADER_LINE.sub('', text).strip()
    return f'{_ALIGNMENT_HEADER}\n\n{body}' if body else _ALIGNMENT_HEADER


def has_motion(text: str) -> bool:
    """Whether a prompt still says anything once the header, the identity
    sentence and the labels are set aside — what the launch asks AFTER its
    rewrite. Asked before it, a clip's prompt pasted back with its motion
    deleted passed as text and reached the sampler empty."""
    return bool(_description_field(strip_picture_references(text)))


# Past this many words the answer is at the token budget (~500 tokens of
# prose with labels and timecodes), so an unfinished tail is the cut, not a
# missing full stop. The rules ask for 60-160 words; nothing legitimate is
# near it. Counted on the scrubbed answer: a reasoning model's <think> block
# is prose that tokenizes lighter than labels and timecodes, and counted raw
# it made complete answers look cut (a block that DID eat the budget leaves
# a tail the fragment test sees).
_BUDGET_WORDS = 280


def finish(text: str, *, with_image: bool) -> str:
    """The raw answer to a prompt the graph can take verbatim. Returns the bare
    scrubbed core when it is too short to be a prompt — or the bare description
    when THAT is (a refusal, a stub: labels and a header around nothing would
    pass any length check) — so the caller's floor sees the model's failure
    rather than a decorated one."""
    core = _purge_hybrid(_scrub(text))
    truncated = len(core.split()) >= _BUDGET_WORDS
    if len(core) < MIN_CHARS:
        return core
    out = restructure_fields(core, truncated=truncated)
    desc = _description_field(out)
    if len(desc) < MIN_CHARS:
        return desc
    if with_image:
        return inject_alignment_header(ensure_identity_tag(out))
    return strip_picture_references(out)


# ── the calls ───────────────────────────────────────────────────────────────

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


def _writer_model(model) -> str | None:
    """The model both steps use: the caller's, else the ⚙ choice, else the
    provider's own (None). Resolved HERE so the launch — which sends no model
    — and a panel that reloaded write with the model that was chosen, rather
    than the ⚙ window being the only place that ever read the setting."""
    return str(model or configured_model() or '').strip() or None


def describe_still(image_name, model=None) -> str:
    """The first frame as a frozen still — step one of AUTO, and the anchor the
    enhancer uses when a frame is staged. '' when the model gives nothing back:
    a missing description degrades the writing, it does not stop it."""
    from . import vision_llm
    with open(_staged_path(image_name), 'rb') as fh:
        data = fh.read()
    return ' '.join(str(vision_llm.describe_image(
        data, _VISION_STILL, num_predict=400,
        model=_writer_model(model)) or '').split())


def _write(system, user, *, temperature, model=None, with_image) -> str:
    """One text call through the configured provider, finished into the
    official format. `with_image` decides whether the frame is named: the
    identity tag and the alignment header go on an image-to-video prompt and
    on nothing else. Strict, like the image generator's enhancer: a call that
    fails raises its sentence (the fence keeps its type, so the route can
    answer 409 with the unload offer) instead of dissolving into ''."""
    from . import vision_llm
    raw = vision_llm.generate_text(
        f'{system}\n\n{user}', num_predict=MAX_TOKENS, num_ctx=NUM_CTX,
        temperature=temperature, top_p=TOP_P, top_k=TOP_K, min_p=MIN_P,
        presence_penalty=PRESENCE_PENALTY, think=THINK, stop=_STOP,
        model=_writer_model(model), strict=True)
    return finish(raw or '', with_image=with_image)


def suggest_from_frame(image_name, instruction=None, model=None,
                       seconds=None, shots=1) -> str:
    """✨ A clip proposed from the staged start frame, in the official format.

    Two calls: the frame is described as a still, then the clip is written
    from that description. The split is the whole point — see the module note.

    `instruction` is whatever is already in the Motion field: the frame says
    what is THERE, the instruction says what should HAPPEN in it, and the
    writer is asked to obey it with the people the frame actually shows.
    Without one the proposal is free, and a spark keeps two presses apart.

    `seconds` is the clip length the dials are set to; `shots` how many shots
    to cut it into (one, until the panel offers more). Both reach the writer
    as the shot plan, so a 15 s clip is paced as one.
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
    ask = f'{ask}\n\n{_CLOSING}\n\n{shot_directive(seconds, shots)}'
    text = _write(_AUTO_SYSTEM, ask, temperature=TEMP_AUTO, model=model,
                  with_image=True)
    if len(text) < MIN_CHARS:
        raise ValueError('the model returned nothing usable — try again, or '
                         'write the motion yourself')
    return text


def enhance(prompt, image=None, model=None, seconds=None, shots=1) -> str:
    """✨ Obey an instruction about the motion, or enrich the motion itself —
    in the official format, paced to the clip length.

    Which of the two happens is the MODEL's call, from the text alone — the
    image generator's own design, and the reason a single button can both
    embellish "she turns" and act on "make her jump instead".

    `image` is the staged start frame, when there is one: the enhancement is
    then anchored on what the clip will actually animate, so "make her turn
    toward the window" cannot invent a window, and the frame is referenced as
    <Picture 1>. Its description failing is not fatal — the text is still
    enriched, just unanchored. No image means text-to-video: the writer is
    told there is no picture, and none is named or headed.
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
    system = f'{_ENHANCE_SYSTEM}\n\n{_IDENTITY_RULE if image else _NO_PICTURE_RULE}'
    ask = f'{anchor}Text: {base}\n\n{_CLOSING}\n\n{shot_directive(seconds, shots)}'
    text = _write(system, ask, temperature=TEMP_ENHANCE, model=model,
                  with_image=bool(image))
    if len(text) < MIN_CHARS:
        # An error, not the original handed back: the caller's field is only
        # written on success, so the prompt is safe either way — and a click
        # that "worked" with nothing to show for it hid the model's failure
        # behind "nothing to add". Same answer as the image studio's writer.
        logger.warning('motion enhance: unusable answer (%d chars)', len(text))
        raise RuntimeError('The model answered nothing usable as a prompt — your text '
                           'is unchanged; try again, or pick another model under ⚙.')
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
