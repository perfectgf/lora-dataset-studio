"""🗣 What HAPPENS in a shot — the caption pass, and the text two features read.

A caption here does two jobs, and the second is the one that makes it worth the
GPU time: it is the text the 🔎 hybrid search matches literally, AND it is what
the promotion writes into a clip's `.txt` sidecar — which IS the training prompt.
The CLIP pass (video_clip_search.py) already finds what a moment LOOKS like; no
frame carries "turns and walks away", because that is a fact about time.

⛔ NOT OLLAMA. It fails silently on video on this machine — an empty response,
no error. See infer/video_caption_infer.py for the whole reasoning and for what
was verified locally before any of this was written.

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

import logging
import os
import re
import shutil
import tempfile

from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource

logger = logging.getLogger(__name__)

# How many frames the captioner sees. Eight spans a shot densely enough for a
# gesture to be visible while staying one modest forward pass — a video VLM's
# cost grows with frames, and this pass runs over a whole bank.
CAPTION_FRAMES = 8

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
_PROMPT = (
    'Describe what happens in this video clip, in one or two plain sentences. '
    'Lead with the ACTION and how it unfolds — who or what moves, and how — then '
    'the setting and the look. Include camera movement when there is any (pans, '
    'tilts, push in, handheld). Do not begin with "This video shows" or any '
    'similar preamble, do not list objects as an inventory, and do not mention '
    'the frames or the video itself. Write it as a caption, not as a report.'
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
# caught it hiding behind. A later tidy-up of this wording would quietly restore
# the euphemism, which is a dataset defect rather than a style regression: a
# caption that talks around its footage teaches the trained model to look away,
# and the output reads perfectly well either way.
_PROMPT_PLAIN = (
    'Describe what happens in this video clip, in one or two plain sentences. '
    'Lead with the ACTION and how it unfolds — who or what moves, and how — then '
    'the setting and the look. Include camera movement when there is any (pans, '
    'tilts, push in, handheld). When nudity or sexual content is present, name it '
    'plainly and specifically — state what body parts are visible and what acts '
    'occur; never euphemize, never write "intimate" or "sensual" in place of what '
    'is actually shown. Do not begin with any preamble, do not list objects as an '
    'inventory, and do not mention the frames or the video itself.'
)

# The styles a caption run can be asked for. `standard` is first and is the
# default: an install that sets nothing captions exactly as it did before.
# Labels are what the UI shows, so they live here rather than being invented
# twice on two sides of the API.
CAPTION_STYLES = {
    'standard': {
        'label': 'Standard',
        'hint': 'Describes the action, the setting and the camera.',
        'prompt': _PROMPT,
    },
    'plain': {
        'label': 'Plain',
        'hint': 'Also names explicit content instead of describing around it. '
                'Measurably better on adult footage, where the standard prompt '
                'produces captions that are about something other than the shot.',
        'prompt': _PROMPT_PLAIN,
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
            caption = clean_caption(_caption_frames(frames, caption_prompt(style),
                                                    worker=worker, span_s=span_s))
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
    else:
        # Nothing wrote it, so nothing is claimed — neither a checkpoint nor a
        # style produced that emptiness.
        clip.caption_state = 'error'
        clip.caption_model = None
        clip.caption_style = None
    db.session.commit()
    return 'ok' if caption else 'error'


def run_captions(bank_id, recaption=False, *, include_edited=False, on_clip=None,
                 should_stop=None, use_gpu=False, model=None, style=None):
    """Caption every shot of a bank that has none yet.

    `should_stop` is polled at each clip BOUNDARY — a graceful cancel, the same
    contract as the image lane's caption batch: what is done is kept, and the
    next run starts where this one stopped."""
    from .video_caption_worker import CaptionWorker
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
        with CaptionWorker(use_gpu=use_gpu, model=model) as worker:
            for clip in rows:
                if should_stop is not None and should_stop():
                    break
                if caption_one(bank, clip, worker=worker, scratch=scratch,
                               relpaths=relpaths, model=model,
                               style=style) == 'ok':
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
    db.session.commit()
    return _clip_row_for(bank_id, clip)
