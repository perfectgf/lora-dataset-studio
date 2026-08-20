"""🔎 Type a word, get the scenes — CLIP embeddings per shot, and the search over them.

A bank of rushes is a haystack whose needles have no names. Metrics tell you which
shots are sharp and which move; they cannot tell you which one has the red car in
it. This module is the answer to that question, and it is the one the user asked
for out loud: type "a woman walking on a beach" and get the shots back, ranked.

WHY SEVERAL FRAMES PER SHOT, AND WHY THE MAX
A shot is a SPAN, and a thumbnail is one instant of it. Embed only the ambassador
frame and a car that drives into view in the last second is invisible — not
mis-ranked, invisible, with no error to see and no way for the user to tell the
scene from one that genuinely is not there. So each shot contributes three frames
(one after its start, its ambassador — the sharpest frame the metrics scan
measured, ``sharpest_frame_s`` — and one before its end), and a shot's score for
a query is the MAX over them. The max, not the mean: a mean asks "is this shot
ABOUT the query", and what the user is asking is "does the query APPEAR in this
shot". Averaging in the two seconds where it does not appear is exactly how the
right answer loses to a blander one.

WHY THE FRAMES ARE DECODED HERE AND EMBEDDED THERE
PyAV is in the Flask venv; torch/open_clip are not (see scoring_python.py). So the
decode happens in-process — the same seam shape as ``video_metrics_scan`` — the
frames land as JPEGs in a scratch folder, and a warm CLIP worker in the ✨ Score
interpreter turns them into vectors (``clip_image_encoder``). The scratch JPEGs are
deleted as soon as their vectors exist: they are an IPC detail, not an asset, and
keeping three per clip would quietly cost hundreds of MB on a real bank.

The frames are extracted at EMBED_LONG_SIDE, NOT at the metrics scan's 160 px.
That pass reduces hard on purpose (a Laplacian at 1080p costs more than the
decode), but CLIP ViT-L/14 sees 224×224 and feeding it an upscaled 160 px frame
would degrade every vector in the bank to save nothing — the decode, not the
resize, is the cost. Two different passes, two different sizes, on purpose.

WHY THE VECTORS ARE A FILE AND THE STATE IS A COLUMN
Three 768-float vectors per clip is a matrix, not a set of rows: every reader
wants all of it at once (a search is one matrix multiply) and no reader wants one
clip's. So the vectors are an .npz next to the bank's thumbnails, exactly like the
image lane's ✨ Score cache, and ``VideoClip.embed_state`` carries only what the
resume contract and the counters need — both answerable without numpy.

RESUME. The flush order is: vectors to disk FIRST, states to the database second.
Killed in between, a clip has a vector nobody claims: the next run re-embeds it
and overwrites it with an identical one. The reverse order would leave a clip
marked embedded with no vector — findable by nothing, forever, with no way to
notice. One of those two failures is free and the other is permanent.

WHAT THIS DELIBERATELY DOES NOT DO
No similarity threshold, and no default one to tune. Measured on this exact
checkpoint, correct top-1 hits and guaranteed-unrelated pairs OVERLAP (0.177-0.233
against a 0.197 unrelated ceiling): no cut separates them, so a threshold control
would be a knob over a boundary that does not exist. The ranking is the product,
and ``pool_median`` — what a typical shot of THIS bank scores for THIS phrase —
is the only honest yardstick, because it is measured rather than assumed.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile

from ..extensions import db
from ..models import VideoBank, VideoClip, VideoSource
from . import atomic_npz
from . import clip_text_encoder

logger = logging.getLogger(__name__)

# What CLIP ViT-L/14 actually sees is 224×224. Extracting at exactly that would
# leave the preprocessor nothing to crop from on a non-square frame, so the long
# side is a little larger and the aspect ratio is kept — the centre crop then
# behaves the same way it did for every bank image already embedded.
EMBED_LONG_SIDE = 256

# How far inside its own bounds a shot's first and last embedded frames sit. A
# boundary is where a cut just happened: the frames immediately around it are
# disproportionately dissolves, black and half-faded. Same reasoning as the
# thumbnail pass never grabbing frame 0, and small enough that it cannot walk
# past a neighbouring shot.
EDGE_MARGIN_S = 0.25

# Below this, a shot holds one frame's worth of information and paying CLIP three
# times for near-identical pictures buys nothing.
MIN_SPAN_FOR_THREE_S = 1.0

# Clips per flush. The unit of lost work if the machine dies mid-pass — not of
# lost consistency, which the flush ORDER owns (see the module docstring).
FLUSH_EVERY = 25

# The most results one search can return. A ranking longer than this is not a
# ranking, it is the bank in a different order.
MAX_RESULTS = 300

_memo = None            # (cache key, {clip_id: [frame dicts]}) — see load_embeddings


# --- which frames of a shot get embedded ---------------------------------------
def frame_times(start_s, end_s, sharpest_s=None):
    """[(label, seconds)] — the frames to embed for ONE shot, in time order.

    Three of them ('start', 'key', 'end') for anything longer than
    MIN_SPAN_FOR_THREE_S, one otherwise. ``sharpest_s`` is the metrics scan's
    ambassador frame and is used when it falls inside the shot; a timestamp
    outside it belongs to bounds that have since been re-cut, and the middle is
    the honest fallback rather than a clamp that would invent a measurement."""
    start = float(start_s)
    end = max(float(end_s), start)
    span = end - start
    mid = start + span / 2.0
    key = mid
    if sharpest_s is not None:
        try:
            s = float(sharpest_s)
        except (TypeError, ValueError):
            s = None
        if s is not None and start <= s <= end:
            key = s
    if span < MIN_SPAN_FOR_THREE_S:
        return [('key', round(key, 3))]
    margin = min(EDGE_MARGIN_S, span / 4.0)
    return [('start', round(start + margin, 3)),
            ('key', round(key, 3)),
            ('end', round(end - margin, 3))]


def _clip_frame_times(clip):
    """``frame_times`` for a VideoClip row, reading the ambassador frame out of
    the metrics summary the scan stored (absent = the scan has not run here)."""
    import json
    sharpest = None
    if clip.metrics_json:
        try:
            m = json.loads(clip.metrics_json)
            if m.get('metrics_state') == 'ok':
                sharpest = m.get('sharpest_frame_s')
        except (ValueError, TypeError):
            sharpest = None
    return frame_times(clip.start_s, clip.end_s, sharpest)


# --- the two heavy seams --------------------------------------------------------
def _write_frames(src_path, times, dest_dir, stem, long_side=None):
    """[(label, seconds, jpeg path)] — decode ONE shot's frames to disk.

    The single seam that touches PyAV, monkeypatched in tests so the suite runs
    with no video extra. Raises on a segment that cannot be decoded — the caller
    turns that into 'unreadable' for THIS shot and moves on.

    `long_side` overrides EMBED_LONG_SIDE for a caller that needs a bigger frame
    — the caption pass reads faces and signage where an embedder needs a
    thumbnail. One decode loop for both passes rather than two copies of the
    seek-and-decode-forward contract, which is the part that is easy to get
    subtly wrong.

    One `av.open` for all three timestamps rather than three: opening a
    multi-gigabyte rush is not free, and the frames wanted are seconds apart in
    the same file. Seeking lands on the preceding keyframe, so each target is
    reached by decoding forward from its own seek."""
    import av
    from PIL import Image

    os.makedirs(dest_dir, exist_ok=True)
    out = []
    with av.open(str(src_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for label, t in times:
            try:
                container.seek(int(t / (stream.time_base or 1)), stream=stream)
            except Exception:            # noqa: BLE001 — some streams refuse to
                pass                     # seek; decoding from 0 still works
            picked = None
            for frame in container.decode(stream):
                picked = frame
                ts = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
                if ts >= t:
                    break
            if picked is None:
                continue
            path = os.path.join(dest_dir, f'{stem}_{label}.jpg')
            img = picked.to_image()
            side = int(long_side or EMBED_LONG_SIDE)
            img.thumbnail((side, side), Image.LANCZOS)
            img.convert('RGB').save(path, 'JPEG', quality=88)
            out.append((label, t, path))
    if not out:
        raise RuntimeError('no frame could be decoded from this shot')
    return out


def _encode_frame_files(paths, *, encoder=None):
    """[vector | None] for JPEG paths, through the warm CLIP worker. The second
    seam, monkeypatched in tests so nothing here ever loads a real model."""
    if encoder is None:
        raise RuntimeError('no frame encoder')
    return encoder.encode(paths)


# --- the vector store ------------------------------------------------------------
def embed_cache_path(bank_id):
    """Next to the bank's thumbnails: it is derived data about THIS bank, and
    deleting the bank must take it along (delete_bank disposes of the folder)."""
    from .video_bank_service import _bank_dir
    return _bank_dir(bank_id) / 'clip_embeddings.npz'


def forget_memory_cache():
    """Drop the in-memory copy. Tests use it to simulate a restart; the file is
    untouched, which is the point."""
    global _memo
    _memo = None


def save_embeddings(bank_id, store):
    """Write {clip_id: [{label, time_s, vec}]} as parallel arrays.

    Parallel arrays rather than one row per clip because a clip has a VARIABLE
    number of frames (a half-second shot has one) and the reader wants a single
    matrix anyway. Written to a temp file and renamed, so a crash mid-write
    cannot leave a bank with a half a vector store."""
    import numpy as np
    path = embed_cache_path(bank_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids, labels, times, vecs = [], [], [], []
    for cid, frames in store.items():
        for f in frames:
            ids.append(int(cid))
            labels.append(str(f['label']))
            times.append(float(f['time_s']))
            vecs.append(np.asarray(f['vec'], dtype='float32'))
    if not ids:
        # Nothing to store is a legitimate state (a bank whose every shot was
        # unreadable). Writing an empty .npz keeps the reader on one code path.
        arrays = dict(clip_ids=np.zeros(0, dtype='int64'),
                      labels=np.array([], dtype='<U8'),
                      times=np.zeros(0, dtype='float32'),
                      vecs=np.zeros((0, 1), dtype='float32'))
    else:
        arrays = dict(clip_ids=np.asarray(ids, dtype='int64'),
                      labels=np.asarray(labels),
                      times=np.asarray(times, dtype='float32'),
                      vecs=np.stack(vecs).astype('float32'))
    atomic_npz.save_npz_atomic(path, arrays)
    forget_memory_cache()


def _store_entry_count(candidate):
    """How many frame vectors an embedding store holds — 0 when it is unusable.

    This is the validator salvage runs before promoting anything, so it is
    deliberately strict AND it touches every array: an .npz whose writer was
    killed mid-compression opens fine (the header is written first) and only
    fails when a member is actually decompressed. Reading the shapes is what
    turns "the file exists" into "the file is whole".
    """
    import numpy as np
    with np.load(str(candidate), allow_pickle=False) as z:
        if set(z.files) != {'clip_ids', 'labels', 'times', 'vecs'}:
            raise ValueError('embedding store keys do not match')
        ids, labels, times, vecs = z['clip_ids'], z['labels'], z['times'], z['vecs']
        if not (len(ids) == len(labels) == len(times) == len(vecs)):
            raise ValueError('embedding store arrays are not row-aligned')
        return int(len(ids))


def _salvage_orphan_store(bank_id, path):
    """Claim the vectors of an embed run that finished a write it could not rename.

    A frame vector is CLIP over a decoded video frame — the most expensive thing
    this module produces — so a stranded temporary is worth opening. It is
    promoted only if it reads back whole and holds at least as many rows as the
    store already in place; anything else is deleted and logged.
    """
    try:
        atomic_npz.salvage_orphan_tmp(
            path, _store_entry_count,
            lambda message: logger.info('video bank %s: %s', bank_id, message))
    except Exception as error:  # noqa: BLE001 — salvage never breaks a search
        logger.warning('video bank %s: could not salvage embeddings: %s',
                       bank_id, error)


def load_embeddings(bank_id):
    """{clip_id: [{label, time_s, vec}]} for a bank, memoised on the file's own
    size+mtime. A missing or corrupt file is an EMPTY store, never an error: the
    caller's answer to both is the same honest "run 🔎 Search first"."""
    global _memo
    import numpy as np
    path = embed_cache_path(bank_id)
    _salvage_orphan_store(bank_id, path)
    try:
        st = path.stat()
        key = (int(bank_id), str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return {}
    if _memo is not None and _memo[0] == key:
        return _memo[1]
    try:
        with np.load(str(path), allow_pickle=False) as z:
            ids = z['clip_ids']
            labels = z['labels']
            times = z['times']
            vecs = z['vecs']
    except Exception as e:  # noqa: BLE001 — a corrupt store = "not embedded yet"
        logger.warning('video bank %s: embedding store unreadable: %s', bank_id, e)
        return {}
    out = {}
    for i in range(len(ids)):
        out.setdefault(int(ids[i]), []).append(
            {'label': str(labels[i]), 'time_s': float(times[i]),
             'vec': np.asarray(vecs[i], dtype='float32')})
    _memo = (key, out)
    return out


# --- the pass --------------------------------------------------------------------
# How many shots may fail back to back before the pass stops believing the shots
# are the problem.
#
# THE TWO MISTAKES ARE NOT THE SAME SIZE. A shot left unmarked comes back in the
# next run: the cost is one decode. A shot marked 'unreadable' is retired, and
# `pending_clips` never offers it again — the cost is the shot, permanently, with
# no error anywhere to explain it. So the guard is allowed to be wrong in the
# cheap direction and must not be wrong in the expensive one.
#
# 10 rather than 3, because a genuinely damaged CAMERA CARD produces runs of
# broken files (one interrupted transfer, one bad sector, one directory), and
# stopping on three of those would refuse to make progress on a bank that has a
# rough patch in it. 10 rather than 100, because every failure past the first is
# already free information: 855 identical failures is not 855 facts.
#
# The counter resets on every success, so scattered bad files anywhere in a bank
# never trip it — only an UNBROKEN run does, which is the shape an environment
# failure has and the shape a bank of real footage does not.
SYSTEMIC_STREAK = 10


def embed_clips(bank_id, clips, *, encoder=None, scratch=None, on_clip=None,
                should_stop=None):
    """Embed `clips` and persist their vectors.

    Returns {'embedded', 'unreadable', 'aborted'} — `aborted` is None on a normal
    run, else the sentence saying why the pass stopped believing its own
    failures.

    Split out of ``run_embed`` so the job wrapper can own cancellation and
    progress while this owns the flush contract — the same division the metrics
    pass makes between ``video_bank_service._measure_job`` and
    ``video_metrics_scan.measure_one``.

    A SHOT IS ONLY BLAMED ONCE SOMETHING ELSE HAS WORKED. 'unreadable' is a
    verdict about a FILE, and this loop is the only place that can tell that
    verdict apart from "this machine cannot embed anything right now". It gets
    that wrong in exactly one way — by writing the verdict before it knows —
    which is how an unrelated package in a user site-packages retired a bank of
    861 shots one at a time, each one logged as if the footage were at fault.
    So a failure is HELD, not written, until a later shot succeeds and proves
    the environment was fine; a run of them aborts the pass instead."""
    bank = db.session.get(VideoBank, bank_id)
    if bank is None:
        return {'embedded': 0, 'unreadable': 0, 'aborted': None}
    from .video_bank_service import _abs_source_path
    relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                    .filter_by(bank_id=bank_id).all())
    store = _pruned_store(bank_id)
    pending_states = {}
    # Shots that failed with nothing having succeeded since. Held out of
    # `pending_states` on purpose: what is not in there cannot be flushed, so a
    # systemic failure cannot reach the database however long the run was.
    held = []
    embedded = unreadable = 0
    aborted = None
    last_error = None

    def flush():
        # Vectors FIRST, states second — see the module docstring on resume.
        if not pending_states:
            return
        save_embeddings(bank_id, store)
        for cid, state in pending_states.items():
            row = db.session.get(VideoClip, cid)
            if row is not None:
                row.embed_state = state
        db.session.commit()
        pending_states.clear()

    def retire_held():
        """Write the held failures — a success (or a clean end of run) has proved
        they were about those files and not about this machine."""
        nonlocal unreadable
        for cid in held:
            store.pop(cid, None)
            pending_states[cid] = 'unreadable'
            unreadable += 1
        held.clear()

    for clip in clips:
        if should_stop is not None and should_stop():
            break
        path = _abs_source_path(bank, relpaths.get(clip.source_id) or '')
        times = _clip_frame_times(clip)
        frames = []
        error = None
        if not path:
            error = 'the source file is not where the bank says it is'
        else:
            try:
                written = _write_frames(path, times, scratch, f'clip_{clip.id}')
                vecs = _encode_frame_files([p for _, _, p in written],
                                           encoder=encoder)
                for (label, t, fpath), vec in zip(written, vecs):
                    if vec is not None:
                        frames.append({'label': label, 'time_s': float(t),
                                       'vec': vec})
                    _discard(fpath)
            except clip_text_encoder.TextEncodeError as e:
                # The WORKER, not the shot. This one is raised only when the
                # encoder could not start or stopped answering — a statement
                # about this interpreter, which the next shot would repeat
                # identically. Blaming the footage for it is the whole defect.
                aborted = (f'the frame encoder could not run: {e} — no shot was '
                           'marked; fix the ✨ Score interpreter and click '
                           '🔎 Find scenes again')
                logger.warning('video bank %s: embedding pass abandoned: %s',
                               bank_id, e)
                held.clear()
                break
            except Exception as e:  # noqa: BLE001 — one shot never sinks the pass
                logger.warning('video bank %s: clip %s not embedded: %s',
                               bank_id, clip.id, e)
                error = f'{type(e).__name__}: {e}'
                frames = []
        if frames:
            retire_held()
            store[clip.id] = frames
            pending_states[clip.id] = 'ok'
            embedded += 1
        else:
            last_error = error or last_error
            held.append(clip.id)
            if len(held) >= SYSTEMIC_STREAK:
                # `last_error` is None when the failures were silent — every
                # frame refused by the worker, no exception. Saying so beats
                # printing "None" at a user who is trying to find out what broke.
                why = last_error or 'the encoder returned no vector for any frame'
                aborted = (f'{len(held)} shots in a row could not be embedded — '
                           f'stopping rather than retiring them one by one '
                           f'({why}). Nothing was marked; click '
                           f'🔎 Find scenes again once it is fixed')
                logger.warning('video bank %s: embedding pass abandoned after %s '
                               'consecutive failures: %s',
                               bank_id, len(held), why)
                held.clear()
                break
        if on_clip is not None:
            on_clip()
        if len(pending_states) >= FLUSH_EVERY:
            flush()
    # A run that ended normally — or on Stop — keeps the verdicts it earned: a
    # tail of failures shorter than the streak really is about those files.
    retire_held()
    flush()
    return {'embedded': embedded, 'unreadable': unreadable, 'aborted': aborted}


def _pruned_store(bank_id):
    """The bank's vectors, minus the orphans.

    An orphan is a shot that was embedded and has since been re-cut or deleted:
    ``_forget_measurements`` clears its ``embed_state`` and deliberately does NOT
    rewrite this file, because a trim is an interactive gesture and the store is
    tens of MB on a real bank. Unreachable already (the search reads the column
    first), they are dropped HERE — inside a pass that is going to rewrite the
    file anyway, so the cleanup costs nothing extra and the store cannot grow
    orphans forever."""
    store = dict(load_embeddings(bank_id))
    if not store:
        return store
    alive = {cid for (cid,) in db.session.query(VideoClip.id)
             .filter(VideoClip.bank_id == bank_id,
                     VideoClip.embed_state == 'ok').all()}
    return {cid: frames for cid, frames in store.items() if cid in alive}


def _discard(path):
    """A scratch frame has done its job the moment its vector exists. Failing to
    remove it is not worth an exception — the whole folder goes at the end."""
    try:
        os.unlink(path)
    except OSError:
        pass


def pending_clips(bank_id, reembed=False):
    """The shots this pass would work on, oldest first. Only shots of a source
    that PROBED — an unreadable file has no frames to decode, and counting it as
    'unreadable' every run would make the pass look permanently broken.

    NEW SHOTS FIRST, AND THE FAILED ONES WHEN THERE ARE NONE. 'unreadable' is a
    hypothesis about a file formed by a pass that may have been wrong about its
    whole environment — and a hypothesis nothing ever re-tests is a fact by
    accident. A bank whose 861 shots were all retired by a broken interpreter
    answered "0 shots to embed" to 🔎 Find scenes and "run Find scenes first" to
    ✂ Duplicates, at the same time, with no way out of the app.

    So the retry rides the button that is already there rather than a new one:
    it costs nothing on the normal path (a bank with new shots never reaches the
    second tier), it needs no explaining, and after any bad pass the answer to
    "what do I do now" is the thing the user was going to click anyway. The
    price of being wrong is a decode the user asked for; the price of never
    asking again is the bank."""
    q = (VideoClip.query.filter_by(bank_id=bank_id)
         .join(VideoSource, VideoSource.id == VideoClip.source_id)
         .filter(VideoSource.probe_state == 'ok'))
    if reembed:
        return q.order_by(VideoClip.id.asc())
    fresh = q.filter(VideoClip.embed_state.is_(None))
    if fresh.first() is not None:
        return fresh.order_by(VideoClip.id.asc())
    return (q.filter(VideoClip.embed_state == 'unreadable')
            .order_by(VideoClip.id.asc()))


def run_embed(bank_id, reembed=False, *, on_clip=None, should_stop=None,
              use_gpu=False):
    """Embed every shot of a bank that has not been embedded yet.

    The scratch folder is a temp directory removed on the way out, whatever
    happened: the JPEGs are an IPC detail between two processes, and a pass that
    was killed must not leave hundreds of MB of frames behind."""
    from .clip_image_encoder import ImageEncoder
    rows = pending_clips(bank_id, reembed).all()
    if not rows:
        return {'embedded': 0, 'unreadable': 0, 'aborted': None}
    scratch = tempfile.mkdtemp(prefix=f'lds-vembed-{bank_id}-')
    try:
        with ImageEncoder(use_gpu=use_gpu) as encoder:
            return embed_clips(bank_id, rows, encoder=encoder, scratch=scratch,
                               on_clip=on_clip, should_stop=should_stop)
    except clip_text_encoder.TextEncodeError as e:
        # A BACKSTOP, not the live path: the encoder starts lazily on the first
        # frame, so today this error always surfaces inside the loop above, which
        # is where the "do not blame the shots" rule is written and tested. It is
        # caught here anyway because the day the worker is started eagerly — by a
        # warm-up, a device check, a preflight — this is the line between "the
        # job says why" and "the whole pass raises past its own contract".
        logger.warning('video bank %s: embedding pass never started: %s', bank_id, e)
        return {'embedded': 0, 'unreadable': 0,
                'aborted': f'the frame encoder could not run: {e} — no shot was '
                           'marked; fix the ✨ Score interpreter and click '
                           '🔎 Find scenes again'}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# --- the search -------------------------------------------------------------------
# Same calibration as the image lane's push-down, and the same reason it exists at
# all: CLIP does not negate. Measured on this checkpoint, "an astronaut without a
# helmet" scored HIGHER on a helmeted astronaut than "with a helmet" did — the
# word is not weakly handled, it is ignored, and the results come back full and
# confident carrying exactly what was asked to be gone. Subtracting the unwanted
# phrase's similarity is the only mechanism that works. See
# image_bank_service.PUSH_DOWN_WEIGHT_DEFAULT for the 7,316-image calibration this
# number comes from; it is imported rather than re-derived so the two lanes cannot
# drift into two different meanings of the same control.
def _push_down_weight(value):
    from .image_bank_service import _push_down_weight as image_weight
    return image_weight(value)


# ── 🗣 The caption half of the search ─────────────────────────────────────────
# CLIP ranks what a moment LOOKS like. It cannot find "turns and walks away",
# because that is a fact about time and no single frame carries it. A caption
# carries exactly that, and nothing the writer did not name — so the two are
# complements, not alternatives, and the search uses both when both exist.
#
# THE WEIGHT IS INHERITED BY ANALOGY, NOT MEASURED. The 0.6 this project HAS
# measured (image_bank_service.PUSH_DOWN_WEIGHT_DEFAULT, over 7 316 images) is
# the weight of a SUBTRACTED excluded phrase — a different experiment with a
# different question. No calibration exists for blending a literal caption match
# with a visual one, so this constant is a starting point, it is named, it is
# returned to the caller, and the UI must never present it as a measured figure.
HYBRID_CAPTION_WEIGHT = 0.6

# Words that appear in nearly every caption. Counting them would hand a free
# half-match to any query written as a sentence rather than as keywords.
_STOPWORDS = frozenset((
    'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'is', 'are',
    'as', 'by', 'with', 'from', 'it', 'its', 'this', 'that', 'their', 'his',
    'her', 'they', 'he', 'she'))

_WORD = re.compile(r"[a-z0-9']+")


def _terms(text):
    return [w for w in _WORD.findall(str(text or '').lower())
            if w not in _STOPWORDS]


def caption_hit(caption, query):
    """0..1 — the share of the query's meaningful words present in `caption`.

    A SHARE rather than a boolean: "a red car" fully present is a stronger claim
    than one word of three, and collapsing them would rank a caption containing
    only "car" level with an exact match. A clip with no caption scores 0 and is
    NOT excluded — it is still findable by CLIP, and treating "no caption" as "no
    match" would silently delete every un-captioned shot from every ranking."""
    wanted = _terms(query)
    if not wanted:
        return 0.0
    have = set(_terms(caption))
    if not have:
        return 0.0
    return sum(1 for w in wanted if w in have) / len(wanted)


def search(user_id, bank_id, query, n=60, *, push_down=None,
           push_down_weight=None, status=None):
    """Rank a bank's shots by CLIP similarity to a written phrase.

    Returns {'results': [{clip_id, score, frame_s, frame_label, match?,
    excluded_match?}], 'clips': [ranked clip rows], 'clip_ids', 'query', 'cached',
    'pool', 'embedded', 'unembedded', 'score_range', 'pool_median', 'push_down',
    'push_down_weight'}.

      * ``frame_s`` is the second the match was found at — the number the player
        seeks to. Without it the user is handed a 30-second shot and told the
        answer is somewhere inside.
      * ``unembedded`` is load-bearing: a shot with no vector cannot be found by
        ANY phrase, and answering "3 results" without saying so lets the user
        conclude the scene is not in the bank.
      * ``clips`` carries the ranked rows because the ranking IS an order, and the
        clip list endpoint sorts by file and start time — fetching them in a
        second request would silently discard what was just computed.

    Raises ValueError (→400) for an empty query, an unknown bank or a bank with
    no embeddings, and clip_text_encoder.TextEncodeError (→503) when no
    interpreter here can run CLIP at all. Those are two different sentences to a
    user: one is "do this first", the other is "this install cannot do this"."""
    import numpy as np
    from .video_bank_service import _clip_row, get_bank, metric_thresholds
    bank = get_bank(user_id, bank_id)
    if bank is None:
        raise ValueError('video bank not found')
    # `-term` is the same grammar the image lane uses, and it is the ONLY way to
    # express "not this" to a model that ignores the word.
    positive, from_query = clip_text_encoder.split_query(query)
    text = clip_text_encoder.normalize_query(positive)
    excl = clip_text_encoder.normalize_query(
        ', '.join(t for t in (push_down or '', from_query) if t))
    if not text:
        # A bare "-hat" is not a search: ranking by "least like a hat" returns
        # whatever is least like anything, which is noise wearing the costume of
        # an answer.
        raise ValueError('a search query is required — a pushed-down term alone '
                         'cannot rank anything')
    store = load_embeddings(bank_id)
    q = VideoClip.query.filter_by(bank_id=bank_id)
    if status:
        q = q.filter_by(status=status)
    rows = {c.id: c for c in q.all()}
    # BOTH conditions, and the state comes first: a retouch clears `embed_state`
    # and only then tries to rewrite the store, so a store it could not rewrite
    # must not put a shot back into a ranking that no longer describes it. The
    # column is the authority; the store is the data.
    pool = [cid for cid, row in rows.items()
            if row.embed_state == 'ok' and store.get(cid)]
    if not pool:
        raise ValueError('run 🔎 Search first — it embeds the frames this '
                         'ranking reads')
    weight = _push_down_weight(push_down_weight if push_down_weight is not None
                               else None)
    # Encoded AFTER the cheap refusals: never make someone wait on a CLIP load to
    # then be told their bank was never embedded.
    qv, cached = clip_text_encoder.encode_query(text)
    qv = np.asarray(qv, dtype='float32')
    qv /= (float(np.linalg.norm(qv)) + 1e-8)
    nv = None
    if excl:
        nv, ncached = clip_text_encoder.encode_query(excl)
        nv = np.asarray(nv, dtype='float32')
        nv /= (float(np.linalg.norm(nv)) + 1e-8)
        # `cached` promises INSTANT, and half a cache hit is not instant.
        cached = bool(cached) and bool(ncached)

    # The caption half. Computed only over the pool, and only when a caption
    # exists anywhere in it — a bank that never ran the caption pass must rank
    # EXACTLY as it did before this feature.
    hits = {cid: caption_hit(rows[cid].caption, text) for cid in pool}
    captioned = sum(1 for cid in pool if (rows[cid].caption or '').strip())
    hybrid = captioned > 0

    scored = []
    for cid in pool:
        best = None
        for frame in store[cid]:
            v = frame['vec']
            match = float(v @ qv)
            score = match
            excluded = None
            if nv is not None:
                excluded = float(v @ nv)
                score -= weight * excluded
            # MAX over the shot's frames — see the module docstring. The frame
            # that WINS is the one reported, so the timestamp the player seeks to
            # is the one that actually matched.
            if best is None or score > best[0]:
                best = (score, match, excluded, frame)
        scored.append((cid, best))
    # SCALED BY THE RANKING'S OWN SPREAD, not added raw. CLIP cosines live in a
    # narrow per-bank band (0.09-0.23 measured on this checkpoint) while a
    # caption hit is 0..1, so adding them directly would make any literal match
    # outrank every visual one by a factor of five — a caption filter wearing a
    # blend's clothes. The pool's own gap between its best score and its median
    # is the only conversion factor that is measured rather than invented, and it
    # is the same reasoning that made pool_median the yardstick for the CLIP-only
    # ranking.
    if hybrid:
        raw = [b[0] for _, b in scored]
        spread = max(raw) - float(np.median(raw)) if len(raw) > 1 else 0.0
        if spread <= 0:
            # Every shot scores the same visually — a real case on a
            # single-subject bank, and the one where the caption is the ONLY
            # thing that can separate them. Falling back to the score's own
            # magnitude keeps the caption a tie-break instead of a no-op.
            spread = abs(max(raw)) or 1.0
        if spread > 0:
            scored = [(cid, (b[0] + HYBRID_CAPTION_WEIGHT * hits[cid] * spread,
                             b[1], b[2], b[3]))
                      for cid, b in scored]
    # Descending score, clip id as the tie-break, so the same query on the same
    # bank always returns the same order.
    scored.sort(key=lambda p: (-p[1][0], p[0]))
    n = max(1, min(int(n or 60), MAX_RESULTS))
    keep = scored[:n]

    results = []
    for cid, (score, match, excluded, frame) in keep:
        row = {'clip_id': cid, 'score': round(score, 4),
               'frame_s': round(float(frame['time_s']), 3),
               'frame_label': frame['label']}
        if excluded is not None:
            row['match'] = round(match, 4)
            row['excluded_match'] = round(excluded, 4)
        if hybrid:
            # Reported per result so the grid can say WHY a shot moved up. A
            # ranking that reordered itself for a reason the user cannot see is
            # a ranking they cannot check.
            row['caption_hit'] = round(hits[cid], 3)
        results.append(row)
    thresholds = metric_thresholds()
    relpaths = dict(db.session.query(VideoSource.id, VideoSource.relpath)
                    .filter_by(bank_id=bank_id).all())
    # The range is in POSITIVE-match units even when a term was pushed down: it is
    # read against pool_median to judge whether the ranking discriminates, and a
    # composite score (which can go negative) is not on that scale.
    matches = [b[1] for _, b in keep]
    all_matches = [b[1] for _, b in scored]
    return {
        'query': text, 'cached': bool(cached),
        'pool': len(pool), 'embedded': len(pool),
        'unembedded': max(0, len(rows) - len(pool)),
        'results': results,
        'clip_ids': [r['clip_id'] for r in results],
        'clips': [_clip_row(rows[r['clip_id']], relpaths, thresholds)
                  for r in results],
        'score_range': ({'top': round(max(matches), 4),
                         'bottom': round(min(matches), 4)} if matches else None),
        'pool_median': (round(float(np.median(all_matches)), 4)
                        if all_matches else None),
        'push_down': excl or None,
        'push_down_weight': round(weight, 3) if excl else None,
        # What the ranking LEANED ON. The readiness line says "CLIP only" or
        # "CLIP + captions" from this, and the weight rides along because it is
        # NOT a measured constant (see HYBRID_CAPTION_WEIGHT) — presenting it as
        # one would be the dishonest part of an otherwise honest feature.
        'hybrid': hybrid,
        'captioned': captioned,
        'caption_weight': HYBRID_CAPTION_WEIGHT if hybrid else None,
    }
