"""🗣 What HAPPENS in a shot — the caption pass, and the text two features read.

A caption here does two jobs, and the second is the one that makes it worth the
GPU time: it is the text the 🔎 hybrid search matches literally, AND it is what
the promotion writes into a clip's `.txt` sidecar — which IS the training prompt.
The CLIP pass (video_clip_search.py) already finds what a moment LOOKS like; no
frame carries "turns and walks away", because that is a fact about time.

WHICH ENGINE: what this machine HAS (resolve_backend). LDS's own transformers
worker is the default — native frame timestamps, bf16, the umT5 token count —
and when no interpreter here can run it, the pass runs through the local LLM
the user already operates (Ollama or LM Studio, the same vision_llm waist and
settings the image passes obey). History, dated: on 2026-08-04 Ollama returned
EMPTY on multi-frame requests on this machine, which is why this lane was born
transformers-only; remeasured 2026-09-01 (Ollama 0.32, qwen3-vl pulled since):
16 frames in one call, full answers — the claim expired. The structural guard
is what actually protects the bank and it applies to EVERY engine: an empty
answer is stored as an error, never as a caption.

FRAMES DECODED HERE, MODEL RUN THERE, exactly like the embedding pass: PyAV is
in the Flask venv and torch is not. The frame extraction reuses that pass's own
seam (`video_clip_search._write_frames`) at a larger size — a captioner reads
faces, text and gesture where an embedder needs a thumbnail — and the frames are
deleted with the caption that replaced them.

SAMPLED ACROSS THE WHOLE SHOT, not around its middle. Eight frames spanning the
span is what makes an ACTION visible; three frames from the centre describe a
moment and would lose the very thing this pass exists for.

A GENERATED CAPTION IS A DRAFT. `caption_state` tells 'ok' (generated) from
'edited' (a human corrected it), and a bulk re-run skips 'edited' rows. Losing a
correction to the next pass is the one thing that would stop anyone from ever
making one — so overwriting a human's words requires asking for it by name.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile

from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource

from . import caption_fields

logger = logging.getLogger(__name__)

# How many frames the captioner sees. Sixteen, per the 2026-08 captioning
# survey: the payoff of a richer caption is a payoff of MOTION (MiraData
# Table 4 — caption density doubles the dynamism and tracking scores while
# image quality stays flat), and motion is read BETWEEN frames, so this count
# is the ceiling on what any prompt can extract. Sixteen 448px frames keep the
# 4B model around 13 GB with its KV cache — still fits a 24 GB card. The 0.2 s
# spacing floor below means SHORT shots still get fewer, never duplicates.
CAPTION_FRAMES = 16

# Long side of each frame handed to the model. Larger than the embedding pass's
# 256 px on purpose: CLIP sees 224 and needs a thumbnail, while a captioner is
# being asked to read faces, gestures and signage.
CAPTION_LONG_SIDE = 448

# Kept off the cut for the same reason the embedding pass keeps off it: a shot
# boundary is where a dissolve lives, and a caption about a dissolve is a caption
# about the edit rather than about the scene.
EDGE_MARGIN_S = 0.25

# Clips per flush. The unit of lost work if the machine dies mid-pass; the
# database is committed per clip, so it is not a unit of lost consistency.
COMMIT_EVERY = 1



# How many shots in a row may fail before the pass says so ON SCREEN instead of
# at the end — the image lane's _FENCE_STREAK_WARN doctrine, same number, same
# reasoning: five appears within a minute of a real outage (a dead local LLM, a
# blocked GPU fence, a vanished worker) and a couple of unlucky shots never
# trip it. The pass keeps running — a re-run finishes the failed rows — but the
# person watching the bar can stop it and fix the engine first.
FAIL_STREAK_WARN = 5
# The checkpoint that shipped with this pass, and the value the setting falls
# back to. Named here rather than only in the worker because the parent has to
# answer "which model will run" WITHOUT starting one — for the job line, for the
# download warning, and for the row it writes next to every caption.
DEFAULT_MODEL = 'Qwen/Qwen3-VL-4B-Instruct'


def configured_model():
    """The checkpoint to caption with. Blank setting = the shipped default, so an
    install that sets nothing captions exactly as it did before — an empty string
    reaching the worker would fail on a model id of nothing."""
    from .. import config as cfg
    return (cfg.get('video_caption.model') or '').strip() or DEFAULT_MODEL


def _hf_cache_dirs():
    """Every `hub` directory a Hugging Face download could have landed in, in
    resolution order. Read from the environment rather than by importing
    huggingface_hub, which is not in the Flask venv — this has to answer before
    any model process exists."""
    import os
    roots = []
    for var in ('HUGGINGFACE_HUB_CACHE', 'HF_HUB_CACHE'):
        if os.environ.get(var):
            roots.append(os.environ[var])
    if os.environ.get('HF_HOME'):
        roots.append(os.path.join(os.environ['HF_HOME'], 'hub'))
    from .. import config as cfg
    root = (cfg.get('bank_scoring.models_root') or '').strip()
    if root:
        roots.append(os.path.join(root, 'hub'))
    roots.append(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub'))
    return roots


def model_is_cached(model_id):
    """Is this checkpoint already on this machine?

    The hub layout is `models--<org>--<name>/snapshots/<rev>/`, and the SNAPSHOT
    is what makes it usable: an interrupted download leaves the directory behind,
    and calling that "present" would skip the warning for precisely the case that
    needs it.

    Returns True when it cannot tell — a cache we could not inspect is not
    evidence of absence, and crying wolf about a layout we could not read trains
    people to ignore the one warning that matters (the same rule
    clip_text_encoder.weights_warning() follows)."""
    import os
    folder = 'models--' + str(model_id or '').replace('/', '--')
    try:
        for root in _hf_cache_dirs():
            snap = os.path.join(root, folder, 'snapshots')
            if os.path.isdir(snap) and any(os.scandir(snap)):
                return True
    except OSError:
        return True
    except Exception:  # noqa: BLE001 — an unreadable cache is not an absent model
        return True
    return False


# How long a backend resolution stays trusted. The bank payload carries the
# runtime line on a 2 s poll; probing Ollama's HTTP port on every poll would
# turn a status line into load.
_BACKEND_TTL_S = 15.0
_backend_cache = {'at': 0.0, 'value': None}


def _local_llm_ready(provider) -> tuple[bool, str, str]:
    """(ok, model, why-not) for the configured local LLM provider.

    The app's STANDARD gates, not a home-grown ping: probe_ollama_model checks
    the server answers AND the vision model is pulled; probe_lmstudio_model
    checks the server answers AND the model is loadable. The first cut of this
    only pinged /api/version — which declared "available" a server whose model
    was never pulled, and (worse, LM Studio) a server that was OFF whenever a
    model id was configured, because resolve_model returns the setting without
    a network call. Every clip of such a pass fails; the gate exists to say so
    BEFORE the click (review finding, 2026-09-01)."""
    from .. import capabilities
    if provider == 'lmstudio':
        from . import vision_lmstudio
        model = vision_lmstudio.get_vision_model()
        probe = capabilities.probe_lmstudio_model()
        return bool(probe.get('ok')), model, probe.get('detail') or ''
    from . import vision_ollama
    model = vision_ollama.get_vision_model()
    probe = capabilities.probe_ollama_model(model=model)
    return bool(probe.get('ok')), model, probe.get('detail') or ''


def resolve_backend(fresh=False) -> dict:
    """Which engine will write the captions, decided by what this machine HAS.

    {'backend': 'transformers'|'local_llm', 'engine', 'label', 'model',
     'record_id', 'available', 'reason'} — `record_id` is what caption_model
    records, `reason` the user-facing sentence when nothing can run.

    `video_caption.backend` forces a side ('transformers' | 'local_llm');
    blank = auto: LDS's own worker when the ✨ Score interpreter can run it —
    native timestamps and the real umT5 token count — else the local LLM the
    user already operates. WHICH local server and WHICH model are not new
    settings: local_llm.provider and the provider's vision_model already say
    it for the image passes, and two dials for one fact is how they drift."""
    import time as _time
    if not fresh and _backend_cache['value'] is not None \
            and _time.monotonic() - _backend_cache['at'] < _BACKEND_TTL_S:
        return _backend_cache['value']

    from .. import config as cfg

    def _transformers():
        from .video_caption_worker import unavailable_reason
        reason = unavailable_reason()
        model = configured_model()
        return {'backend': 'transformers', 'engine': 'transformers-local',
                'label': 'Transformers', 'model': model, 'record_id': model,
                'available': reason is None, 'reason': reason}

    def _local():
        from . import vision_llm
        prov = vision_llm.provider()
        label = vision_llm.label(prov)
        ok, model, why = _local_llm_ready(prov)
        return {'backend': 'local_llm', 'engine': prov, 'label': label,
                'model': model or '',
                'record_id': f'{prov}:{model}' if model else prov,
                'available': ok,
                'reason': None if ok else f'{label}: {why}'}

    forced = (cfg.get('video_caption.backend') or '').strip().lower()
    if forced == 'transformers':
        chosen = _transformers()
    elif forced == 'local_llm':
        chosen = _local()
    else:
        chosen = _transformers()
        if not chosen['available']:
            alt = _local()
            if alt['available']:
                chosen = alt
            else:
                # Both absent: ONE sentence naming both, because "install a
                # torch Python" is the wrong advice for someone who runs Ollama.
                chosen = dict(chosen)
                chosen['reason'] = (f"{chosen['reason']} No local LLM could "
                                    f"step in either: {alt['reason']}")
    # One update call, both keys: two separate assignments let a concurrent
    # reader pair a value with the other write's timestamp (review finding 6).
    _backend_cache.update({'value': chosen, 'at': _time.monotonic()})
    return chosen


def caption_unavailable_reason():
    """None when SOMETHING here can caption, else the sentence saying why not —
    the start gate's question, so it probes fresh rather than trusting a poll's
    cache."""
    resolved = resolve_backend(fresh=True)
    return None if resolved['available'] else resolved['reason']


def umt5_tokenizer_dir():
    """The folder holding umT5's `spiece.model`, or None.

    umT5 is the text encoder behind every Wan 2.x model, and it truncates past
    512 tokens IN SILENCE — while a word count is only a guess about tokens
    (1.36 per word, measured over 48 captions from the shipped prompt; that
    guess is what the export preflight is left with when this returns None).

    `video_caption.tokenizer_dir` names a folder outright. Blank = look, without
    downloading anything, in the Hugging Face caches this machine already
    declares — the same roots model_is_cached reads — for any umT5 snapshot:
    ai-toolkit's `umt5_xxl_encoder` mirror keeps it under `tokenizer/`, Wan's own
    repos under `google/umt5-xxl/`. A preflight must never cost a fetch."""
    import glob
    from .. import config as cfg
    explicit = (cfg.get('video_caption.tokenizer_dir') or '').strip()
    if explicit:
        if os.path.isfile(os.path.join(explicit, 'spiece.model')):
            return explicit
        logger.warning('video_caption.tokenizer_dir has no spiece.model: %s', explicit)
        return None
    for root in _hf_cache_dirs():
        for pattern in ('models--*umt5*/snapshots/*/spiece.model',
                        'models--*umt5*/snapshots/*/tokenizer/spiece.model',
                        'models--*/snapshots/*/google/umt5-xxl/spiece.model'):
            try:
                hits = sorted(glob.glob(os.path.join(root, pattern)))
            except OSError:
                hits = []
            if hits:
                return os.path.dirname(hits[0])
    return None


def download_notice(model_id):
    """A sentence to put in the JOB LINE when this checkpoint is not here yet,
    else ''.

    THE DOWNLOAD ITSELF IS ALLOWED — transformers fetches on first use and
    blocking that would mean shipping a model picker that cannot pick anything
    new. What is not allowed is doing it in SILENCE: a pass sitting at 0/470 for
    twenty minutes while gigabytes cross someone's connection is
    indistinguishable from a hang, and they are the one paying for it. So the
    warning rides in the detail the user is already watching, before the first
    clip.

    No size is quoted. We cannot know how big an arbitrary checkpoint is without
    asking the network, and an invented figure is one somebody would plan
    around."""
    if model_is_cached(model_id):
        return ''
    return (f'{model_id} is not on this machine yet — the first run downloads it '
            f'before captioning anything. Leave it running, or stop and set '
            f'video_caption.model back.')


def caption_frame_times(start_s, end_s):
    """The timestamps the captioner is shown, ascending, inside the shot.

    Evenly spaced across the WHOLE span — an action is a fact about time, and
    sampling the middle would describe a moment. A shot too short to hold
    CAPTION_FRAMES distinct instants gets fewer rather than duplicates: handing
    the model eight copies of one picture pays for all eight."""
    start = float(start_s)
    end = max(float(end_s), start)
    span = end - start
    margin = min(EDGE_MARGIN_S, span / 4.0)
    lo, hi = start + margin, end - margin
    if hi <= lo:
        return [round((start + end) / 2.0, 3)]
    # One frame per ~0.2 s at most: below that, consecutive frames of any real
    # footage are the same picture and the model pays for the duplicate.
    n = max(1, min(CAPTION_FRAMES, int((hi - lo) / 0.2) + 1))
    if n == 1:
        return [round((lo + hi) / 2.0, 3)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 3) for i in range(n)]


# --- the prompt --------------------------------------------------------------------
# WHAT THE TRAINER READS. ai-toolkit reads the `.txt` next to the clip as the
# sample's prompt, verbatim — `caption_ext: "txt"` in its own shipped video
# configs (C:\ai-toolkit\config\examples\train_lora_wan22_14b_24gb.yaml:30), and
# `clean_caption` in C:\ai-toolkit\toolkit\dataloader_mixins.py:98-109 does
# nothing at all (every normalising line in it is commented out). So whatever is
# written here IS the prompt, punctuation and preamble included.
#
# Hence two demands in the prompt below that look like style and are not:
#   * ACTION FIRST. A video model learns what happens between the first frame and
#     the last. "A woman in a red coat. A street. A car." is a caption for a
#     photograph and teaches the motion nothing.
#   * NO PREAMBLE. Every caption beginning "This video shows" teaches the model
#     that phrase. Stripping it later is a find-and-replace nobody remembers to
#     run, so it is refused twice — asked for here, and cleaned in clean_caption.
# WHY A FULL PARAGRAPH AND NOT "one or two sentences" (the shape until 2026-08-30):
# the 2026-08 captioning survey converged from three independent directions —
# MiraData's ablation (dense structured captions double motion tracking while
# image quality stays flat), the VC4VG instruction study (the instruction beats
# the checkpoint), and the vendors' own caption specs (LTX-2: 150-220 words,
# WAN: 60-200) — on 150-220 words of flowing prose. The precision clause is
# Moving Alphabet's measured cliff: an invented detail costs more than a missing
# one and cannot be trained away afterwards. And the camera is deliberately NOT
# asked for: VLMs were tested on it twice and failed twice — our homography
# classifier writes the camera line at export instead, in words it can prove.
_PROMPT = (
    'Describe what happens in this video clip as one flowing paragraph of '
    'roughly 150 to 200 words. Lead with the ACTION and follow it in order — '
    'what moves first, what happens next, how it ends. Write the motion '
    'precisely: which limb or object moves, in which direction, how quickly, '
    'what it touches or passes. As the paragraph unfolds, weave in who or what '
    'the subject is (appearance, clothing, distinguishing details), the '
    'setting and what surrounds the action, and the look and mood of the '
    'footage (light, palette, texture). Describe only what is clearly '
    'visible: an invented detail is far more damaging than a missing one, so '
    'leave out anything you cannot actually see, and describe the scene that '
    'is actually shown, never a substitute for it. Do not describe the camera '
    'work and do not mention sound — both are recorded separately. Do not '
    'begin with "This video shows" or any similar preamble, do not list '
    'objects as an inventory, and do not mention the frames or the video '
    'itself. Write it as one training caption in plain prose.'
)

# ── The second prompt, and the measurement that produced it ──────────────────
# The model was made configurable first, on the theory that the checkpoint
# decided how plainly a caption described its footage. An A/B on real material
# said otherwise:
#
#   base model    + this standard prompt           -> evasive, circled the subject
#   uncensored 8B + this standard prompt           -> barely better, hid behind
#                                                     the camera
#   uncensored 8B + the permission prompt below    -> named things precisely
#   BASE model    + the permission prompt below    -> named things precisely too,
#                                                     with the best action writing
#
# The dominant lock is the PROMPT. A captioner describes what it has been given
# permission to describe, and the prompt above never granted it — so the base
# model, asked properly, outperformed an uncensored one asked the old way.
#
# TWO CLAUSES CARRY THE DIFFERENCE and both are pinned by tests: name what is
# visible and what occurs, and do NOT substitute the two words the measurement
# caught it hiding behind. A THIRD came from the 2026-08-07 image bench (vault):
# asked politely, an abliterated checkpoint does not euphemize — it REPLACES the
# whole scene with an invented harmless one, stably across calls. Forbidding the
# euphemism does not forbid that, so the permission prompt says both; the
# standard prompt carries the content-neutral half (describe the scene shown,
# never a substitute), which is also what the 2026-09-01 arbitration caught the
# base model doing on ordinary footage (a woman invented in place of a man). A later tidy-up of this wording would quietly restore
# the euphemism, which is a dataset defect rather than a style regression: a
# caption that talks around its footage teaches the trained model to look away,
# and the output reads perfectly well either way.
_PROMPT_PLAIN = (
    'Describe what happens in this video clip as one flowing paragraph of '
    'roughly 150 to 200 words. Lead with the ACTION and follow it in order — '
    'what moves first, what happens next, how it ends. Write the motion '
    'precisely: which limb or object moves, in which direction, how quickly, '
    'what it touches or passes. As the paragraph unfolds, weave in who or what '
    'the subject is (appearance, clothing, distinguishing details), the '
    'setting and what surrounds the action, and the look and mood of the '
    'footage (light, palette, texture). This may be ADULT (NSFW) footage and '
    'describing it is authorized. When nudity or sexual content is '
    'present, name it plainly and specifically — state what body parts are '
    'visible and what acts occur; never euphemize, never write "intimate" or '
    '"sensual" in place of what is actually shown, and never sanitize, '
    'soften or replace the scene with a different one. Describe only what is '
    'clearly visible: an invented detail is far more damaging than a missing '
    'one, so leave out anything you cannot actually see. Do not describe the '
    'camera work and do not mention sound — both are recorded separately. Do '
    'not begin with any preamble, do not list objects as an inventory, and do '
    'not mention the frames or the video itself. Write it as one training '
    'caption in plain prose.'
)

# C12-C: the structured tail. The paragraph stays the caption that trains; the
# five labelled lines are what a target with a PUBLISHED budget can be served
# instead of a paragraph nobody should cut mid-sentence, and what a later UI
# can show as facets. Parsed by caption_fields.split_caption_fields — forgiving,
# never trusted: a model that skips the block costs the fields, not the caption.
_FIELDS_TAIL = (
    ' When the paragraph is done, write a line containing only --- and then five '
    'short labelled lines, each on its own line and each under 20 words: '
    'Subject: who or what is on screen. Motion: what moves and how. Setting: '
    'where it happens. Style: light, palette and mood. Short: the whole shot in '
    '12 to 20 words.'
)

# The styles a caption run can be asked for. `standard` is first and is the
# default: an install that sets nothing captions exactly as it did before.
# Labels are what the UI shows, so they live here rather than being invented
# twice on two sides of the API.
CAPTION_STYLES = {
    'standard': {
        'label': 'Standard',
        'hint': 'A full-paragraph caption: the action as it unfolds, the '
                'subject, the setting and the mood. The camera line is added '
                'from our own motion classifier at export.',
        'prompt': _PROMPT + _FIELDS_TAIL,
    },
    'plain': {
        'label': 'Plain',
        'hint': 'Also names explicit content instead of describing around it. '
                'Measurably better on adult footage, where the standard prompt '
                'produces captions that are about something other than the shot.',
        'prompt': _PROMPT_PLAIN + _FIELDS_TAIL,
    },
}

DEFAULT_STYLE = 'standard'


def configured_style():
    """The caption style to use unless a run asks for another. Anything unknown
    or blank lands on the default — a typo in a config file must not take the
    pass down, and must certainly not grant a permission nobody asked for."""
    from .. import config as cfg
    style = (cfg.get('video_caption.style') or '').strip().lower()
    return style if style in CAPTION_STYLES else DEFAULT_STYLE


def style_choices():
    """[{key, label, hint}] for the picker, in declared order — default first."""
    return [{'key': k, 'label': v['label'], 'hint': v['hint']}
            for k, v in CAPTION_STYLES.items()]


def caption_prompt(style=None):
    """The instruction handed to the model, for `style` (default when unknown).

    A function rather than a constant, as it always was — the per-variant future
    it was written for is this one."""
    key = (style or DEFAULT_STYLE)
    return CAPTION_STYLES.get(key, CAPTION_STYLES[DEFAULT_STYLE])['prompt']


# The checkpoints a run can be pointed at FROM THE LAUNCH WINDOW. A short vetted
# list, not a free field: the config key (`video_caption.model`) already accepts
# any Hugging Face id for whoever knows what they are doing — the per-run picker
# exists so switching between the known-good options does not require editing
# config, and a typo'd id that would launch a download of nothing stays
# impossible from the UI. Hints carry the two facts that decide the choice
# (measured/benched in the 2026-08-18 captioning survey): the 4B is the proven
# default, the 8B writes better MOTION but almost fills a 24 GB card.
MODEL_CHOICES = {
    DEFAULT_MODEL: {
        'label': 'Qwen3-VL 4B (default)',
        'hint': 'The shipped captioner — fits alongside other GPU work.',
    },
    'Qwen/Qwen3-VL-8B-Instruct': {
        'label': 'Qwen3-VL 8B',
        'hint': 'Describes motion better, at twice the size — it wants the '
                '24 GB card almost to itself, so close other GPU apps first.',
    },
}


def model_choices():
    """[{key, label, hint, cached}] for the launch window — the configured model
    first (it is the default of the picker, whatever it is), then the vetted
    list. A custom-configured checkpoint appears as its own entry rather than
    being hidden by the curation: the picker must be able to SAY what the pass
    will otherwise silently use."""
    configured = configured_model()
    out = []
    seen = set()
    for key in [configured, *MODEL_CHOICES]:
        if key in seen:
            continue
        seen.add(key)
        meta = MODEL_CHOICES.get(key) or {
            'label': key,
            'hint': 'Set in video_caption.model — not one of the vetted picks.',
        }
        out.append({'key': key, 'label': meta['label'], 'hint': meta['hint'],
                    'cached': model_is_cached(key)})
    return out


def resolve_model(model=None):
    """The checkpoint a run should use: an explicit pick from the vetted list
    (or the configured one, which is always a legal pick), else the configured
    default. Unknown values fall back rather than failing — same contract as
    styles, for the same reason — and never fall back to something LARGER than
    what was configured."""
    allowed = {c['key'] for c in model_choices()}
    chosen = (model or '').strip()
    return chosen if chosen in allowed else configured_model()


# Preambles a VLM reaches for even when told not to. Anchored at the start and
# case-insensitive; the following word is re-capitalised so the caption still
# reads as a sentence rather than as a decapitated one.
_PREAMBLE = re.compile(
    r'^\s*(?:the\s+)?(?:this\s+|the\s+)?(?:video|clip|scene|footage|image)\s+'
    r'(?:shows|depicts|features|captures|presents|begins\s+with)\s+',
    re.IGNORECASE)


def clean_caption(text):
    """The caption as it will be written to the sidecar: trimmed, de-preambled.

    Returns '' for anything empty, which the caller treats as a FAILURE and never
    as a caption — that is the Ollama failure mode defended against structurally
    rather than by trusting a model to always speak."""
    s = str(text or '').strip()
    if not s:
        return ''
    s = _PREAMBLE.sub('', s).strip()
    if not s:
        return ''
    return s[0].upper() + s[1:]


# --- the two heavy seams -------------------------------------------------------------
def _write_caption_frames(src_path, times, dest_dir, stem):
    """[jpeg path] for one shot's sampled frames. Monkeypatched in tests.

    Delegates to the embedding pass's PyAV seam rather than opening a second
    decode loop — same file, same seek-and-decode-forward contract, only a larger
    output size (see video_clip_search._write_frames)."""
    from .video_clip_search import _write_frames
    labelled = [(f'cap{i}', t) for i, t in enumerate(times)]
    written = _write_frames(src_path, labelled, dest_dir, stem,
                            long_side=CAPTION_LONG_SIDE)
    return [path for _label, _t, path in written]


def _caption_frames(paths, prompt, *, worker=None, span_s=None):
    """The caption for one shot's frames, through the warm worker. The second
    seam, monkeypatched in tests so nothing here ever loads a real model.
    ``span_s`` — the seconds the frames actually span — rides along so the
    model's frame timestamps tell the truth about time."""
    if worker is None:
        raise RuntimeError('no caption worker')
    return worker.caption(paths, prompt, span_s=span_s)


# --- the pass ---------------------------------------------------------------------
def pending_clips(bank_id, recaption=False, include_edited=False):
    """The shots this pass would work on, oldest first.

    Only shots of a source that PROBED — an unreadable file has no frames to
    show a model. A clip whose caption a human EDITED is skipped even by a
    re-run, unless asked for by name."""
    q = (VideoClip.query.filter_by(bank_id=bank_id)
         .join(VideoSource, VideoSource.id == VideoClip.source_id)
         .filter(VideoSource.probe_state == 'ok'))
    if not recaption:
        # NULL caption OR a state that never produced one. A cleared caption puts
        # the clip back in the queue, which is what "clear it and run again"
        # has to mean.
        q = q.filter(db.or_(VideoClip.caption.is_(None), VideoClip.caption == ''))
    elif not include_edited:
        q = q.filter(db.or_(VideoClip.caption_state.is_(None),
                            VideoClip.caption_state != 'edited'))
    return q.order_by(VideoClip.id.asc())


def caption_one(bank, clip, *, worker, scratch, relpaths, model=None,
                style=None):
    """Caption ONE shot and commit it. Returns 'ok' or 'error'.

    Never raises: a bank is captioned in bulk and a model that refuses clip 200
    must not throw away 199 captions. Commits per clip — the resume contract that
    makes Stop safe."""
    from .video_bank_service import _abs_source_path
    path = _abs_source_path(bank, relpaths.get(clip.source_id) or '')
    caption = ''
    fields = None
    tokens = None
    frames = []
    try:
        if path:
            times = caption_frame_times(clip.start_s, clip.end_s)
            frames = _write_caption_frames(path, times, scratch, f'clip_{clip.id}')
            # The span the sampled frames actually cover, so the model is told
            # the truth about time. Without it, transformers stamps the frames
            # at a default 24 fps: eight frames of a five-second shot read as
            # <0.0s>…<0.3s>, and every description of speed and duration is
            # wrong at the source.
            span_s = times[-1] - times[0] if len(times) > 1 else 0.0
            raw = _caption_frames(frames, caption_prompt(style), worker=worker,
                                  span_s=span_s)
            # C12-C: the paragraph is the caption; the labelled tail, when the
            # model wrote one, becomes the fields. A missing tail costs nothing.
            prose, fields = caption_fields.split_caption_fields(raw)
            caption = clean_caption(prose)
            # Measured by the worker in umT5 tokens when it has the tokenizer
            # (None otherwise — the preflight then estimates, and says so).
            # Counted on the prose BEFORE clean_caption strips a preamble, so
            # the stored count can only OVERSTATE the served text — a false
            # "over the window" at worst, never a missed overrun.
            tokens = getattr(worker, 'last_tokens', None)
    except Exception as e:  # noqa: BLE001 — one shot never sinks the pass
        logger.warning('caption: clip %s failed: %s', clip.id, e)
        caption = ''
    finally:
        for f in frames:
            try:
                os.unlink(f)
            except OSError:
                pass
    if caption:
        clip.caption = caption
        clip.caption_state = 'ok'
        # Stamped with the caption it belongs to, in the same transaction. A
        # bank captioned across a setting change stays readable row by row.
        clip.caption_model = model or None
        clip.caption_style = style or None
        # The labelled tail, when the model wrote one (C12-C). None is honest:
        # a caption without fields is served whole, never a guessed split.
        clip.caption_fields = json.dumps(fields, ensure_ascii=False) if fields else None
        clip.caption_tokens = tokens if isinstance(tokens, int) else None
    else:
        # Nothing wrote it, so nothing is claimed — neither a checkpoint nor a
        # style produced that emptiness.
        clip.caption_state = 'error'
        clip.caption_model = None
        clip.caption_style = None
        clip.caption_fields = None
        clip.caption_tokens = None
    db.session.commit()
    return 'ok' if caption else 'error'


def run_captions(bank_id, recaption=False, *, include_edited=False, on_clip=None,
                 on_detail=None, should_stop=None, use_gpu=False, model=None,
                 style=None, backend=None):
    """Caption every shot of a bank that has none yet.

    `should_stop` is polled at each clip BOUNDARY — a graceful cancel, the same
    contract as the image lane's caption batch: what is done is kept, and the
    next run starts where this one stopped."""
    from .video_caption_worker import CaptionWorker, LocalLlmCaptionWorker
    bank = db.session.get(VideoBank, bank_id)
    if bank is None:
        return {'captioned': 0, 'failed': 0, 'model': model or configured_model(),
                'style': style or configured_style()}
    model = model or configured_model()
    style = style if style in CAPTION_STYLES else configured_style()
    rows = pending_clips(bank_id, recaption, include_edited).all()
    if not rows:
        return {'captioned': 0, 'failed': 0, 'model': model, 'style': style}
    relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                    .filter_by(bank_id=bank_id).all())
    scratch = tempfile.mkdtemp(prefix=f'lds-vcap-{bank_id}-')
    captioned = failed = 0
    try:
        # The job resolves ONCE and hands the dict down: between the job line
        # and this point sits a count query that can outlive the cache TTL, and
        # a re-resolve here could build a different engine than the line just
        # announced (review finding 6).
        backend = backend or resolve_backend()
        if backend['backend'] == 'local_llm':
            # What the user installed writes the captions. caption_model gets
            # the engine-prefixed id so a bank captioned across engines stays
            # readable; no umT5 count arrives from an HTTP server, so the
            # preflight estimates and labels it — exactly the C12-C fallback.
            model = backend['record_id']
            worker_cm = LocalLlmCaptionWorker(provider=backend['engine'],
                                              model=backend['model'])
        else:
            worker_cm = CaptionWorker(use_gpu=use_gpu, model=model,
                                      tokenizer_dir=umt5_tokenizer_dir())
        fail_streak = 0
        with worker_cm as worker:
            for clip in rows:
                if should_stop is not None and should_stop():
                    break
                outcome = caption_one(bank, clip, worker=worker, scratch=scratch,
                                      relpaths=relpaths, model=model,
                                      style=style)
                if outcome == 'ok':
                    fail_streak = 0
                else:
                    fail_streak += 1
                    if fail_streak >= FAIL_STREAK_WARN and on_detail is not None:
                        on_detail(f'captioning — {fail_streak} shots in a row '
                                  f'failed ({model}). They stay uncaptioned and '
                                  'a re-run finishes them — stop the pass if you '
                                  'want to look at the engine first.')
                if outcome == 'ok':
                    captioned += 1
                else:
                    failed += 1
                if on_clip is not None:
                    on_clip()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    # The model rides back with the counts: two checkpoints do not write
    # comparable captions, so "470 captioned" is only half an answer.
    return {'captioned': captioned, 'failed': failed, 'model': model,
            'style': style}


def set_caption(user_id, bank_id, clip_id, text):
    """Store a caption a HUMAN wrote. Returns the clip row, or None when unknown.

    Marked 'edited' so a bulk re-run leaves it alone. Clearing it puts the clip
    back in the queue — which is what "clear it and run the pass again" has to
    mean, and the only way back from a caption someone regrets."""
    from .video_bank_service import _clip_of_bank, _clip_row_for, get_bank
    if get_bank(user_id, bank_id) is None:
        return None
    clip = _clip_of_bank(bank_id, clip_id)
    if clip is None:
        return None
    caption = str(text or '').strip()
    clip.caption = caption or None
    clip.caption_state = 'edited' if caption else None
    # A human wrote it, so neither a checkpoint nor a prompt style is credited.
    clip.caption_model = None
    clip.caption_style = None
    # And nothing the MACHINE derived survives either: stale fields would serve
    # the OLD caption's facets (and, past the budget, its short form INSTEAD of
    # the human's words at export), and a stale measured count would make the
    # preflight's tokens_measured a lie about a deleted text. The estimate path
    # takes over, and says so (review finding, 2026-09-01).
    clip.caption_fields = None
    clip.caption_tokens = None
    db.session.commit()
    return _clip_row_for(bank_id, clip)
