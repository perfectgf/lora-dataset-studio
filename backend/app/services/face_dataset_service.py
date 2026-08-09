"""Face-dataset orchestration: CRUD, fan-out, import, classify, caption, export.

The vision passes (classify/caption) call describe_image_ollama; the CALLER (the
route) is responsible for wrapping them in the GPU-exclusive window. The ComfyUI
output dir is resolved via `cfg.comfyui_dir('output')` so tests can monkeypatch cfg.
"""
from __future__ import annotations
from decimal import Decimal
import io
import json
import lzma
import logging
import math
import ntpath
import os
import posixpath
import random
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import warnings
import zipfile
import zlib
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from ..extensions import db
from ..models import (CanvasImageNode, CanvasNodePosition, FaceDataset,
                      FaceDatasetImage, LoraTestImage)
from .. import config as cfg
from . import (bank_transfer_metadata, caption_origin, dataset_activity,
               image_encoding, input_budget, reference_edit_jobs, trash)
from .dataset_storage import dataset_path, ensure_dataset_dir
from .image_provenance import provenance_metrics
from .image_quality import ANALYSIS_MAX_SIDE, quality_metrics
from .ollama_control import normalize_ollama_model_ref

# Garde le modèle vision chaud entre les images d'un même batch caption/classify
# (sinon Ollama le recharge - cold start ~10s - à CHAQUE image). Déchargé en fin
# de batch pour rendre la VRAM à ComfyUI. ComfyUI est déjà en pause pendant la passe.
_VISION_BATCH_KEEPALIVE = '5m'
from .face_variations import (CAPTION_PROMPT, CAPTION_PROMPT_BOORU,
                              DESCRIPTIVE_CAPTION_PROMPT,
                              CAPTION_REFINE_CONCEPT_PROMPT, CAPTION_LEAK_FIX_PROMPT,
                              EXPAND_CONCEPT_TERMS_PROMPT,
                              CLASSIFY_PROMPT, HEAD_BBOX_PROMPT, WATERMARK_BBOX_PROMPT,
                              JOYCAPTION_PROMPT, aspect_for_label, caption_prompt_for,
                              caption_prompt_for_style, caption_prompt_for_concept,
                              caption_has_identity_leak, caption_has_concept_leak,
                              compose_prompt_suffix, concept_lexical_field,
                              drop_identity_sentences, drop_identity_tags,
                              drop_style_lead_in,
                              is_nsfw_label, prompt_by_label, wrap_variation,
                              wrap_variation_klein, wrap_variation_krea,
                              get_identity_prompt,
                              normalize_subject_type,
                              KLEIN_IMAGE_IMPROVE_PROMPT)

logger = logging.getLogger(__name__)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


# Garde-fou (PAS une limite produit) sur une caption STOCKÉE : la colonne est un TEXT
# sans contrainte DB, mais on borne quand même pour qu'une sortie vision emballée
# (boucle, collage pathologique) ne gonfle pas la base sans fin. Le vrai budget de
# longueur est l'encodeur de texte du trainer (T5 de FLUX/Klein, ~512 tokens ≈ bien
# au-delà d'une caption descriptive normale) et JoyCaption/Qwen bornent déjà leur propre
# sortie (max_new_tokens). Le plafond est donc volontairement TRÈS large et, quand il
# mord, _cap_caption coupe à une FIN DE PHRASE — jamais en plein mot. Historique : à 800
# il tranchait les captions descriptives en pleine phrase (« …a pale, neutral tone, and a »).
CAPTION_MAX_CHARS = 10000

# Exact Unicode whitespace set that Python's ``str.strip()`` recognizes.  SQLite's
# default ``trim`` only removes U+0020, so it would otherwise sample a caption made
# solely of (for example) U+2003 and let the final Python cleanup turn it into an
# apparent empty result.  Supplying this set to SQLite keeps the SQL eligibility
# predicate and the API's ``.strip()`` contract aligned without loading all rows.
_PYTHON_STRIP_CHARS = (
    '\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680'
    '\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a'
    '\u2028\u2029\u202f\u205f\u3000'
)


def _cap_caption(text):
    """Borne une caption à CAPTION_MAX_CHARS sans jamais couper en plein mot ni au
    milieu d'une phrase. Sous le plafond, le texte (strippé) est rendu tel quel ; au
    dessus, on garde les phrases entières jusqu'au plafond, sinon on retombe sur le
    dernier mot entier. Rend toujours une chaîne (l'entrée vide reste vide)."""
    text = (text or '').strip()
    if len(text) <= CAPTION_MAX_CHARS:
        return text
    head = text[:CAPTION_MAX_CHARS]
    last_end = 0
    for m in re.finditer(r'[.!?]["\'”’)\]]?(?=\s|$)', head):
        last_end = m.end()
    if last_end:
        return head[:last_end].strip()
    return head.rsplit(' ', 1)[0].strip() or head.strip()

# Padding du head-crop AUTO de la référence (côté du carré = grand côté de la bbox
# tête × pad). Volontairement plus large que l'ancien 1.7 (jugé « trop serré ») pour
# garder épaules + contexte par défaut ; le recadrage manuel depuis l'original permet
# d'ajuster ensuite dans les deux sens. Ne concerne QUE la référence (les imports
# gardent le défaut 1.7 de face_crop_to_square_webp).
REF_CROP_PAD = 2.0

# Un crop dont le côté source fait moins de size/1.5 se retrouve agrandi ≥50% par le
# LANCZOS du resize final — au-delà, la texture visible est majoritairement inventée
# par l'upscale plutôt que capturée du sujet. Seuil d'avertissement composition_upscaled
# (dataset_payload), pas un blocage : un unique gros plan upscalé n'est pas un problème,
# un dataset qui n'en a QUE des upscalés l'est (biais loss vers ce patch, cf. issue GitHub).
UPSCALE_WARN_THRESHOLD = 1.5


# Backward-compatible aliases for existing service consumers. New cross-module
# callers use the public names from dataset_storage so read paths cannot
# accidentally create directories.
_dataset_path = dataset_path
_dataset_dir = ensure_dataset_dir


def _bank_analysis_cache_dir(dataset_id):
    """Private portable cache vault owned by one Dataset."""
    return os.path.join(_dataset_dir(dataset_id), '.bank-analysis-cache')


def _bank_analysis_cache_is_referenced(dataset_id, cache_ref) -> bool:
    """Prove a content-addressed sidecar has a committed owning snapshot."""
    if not bank_transfer_metadata.is_content_addressed_cache_ref(cache_ref):
        return False
    try:
        stored_snapshots = (db.session.query(
            FaceDatasetImage.bank_analysis_snapshot)
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.bank_analysis_snapshot.isnot(None)).all())
        referenced = any(
            (snapshot := bank_transfer_metadata.parse_snapshot(stored)) is not None
            and snapshot.get('cache_ref') == cache_ref
            for stored, in stored_snapshots
        )
        return bool(referenced and bank_transfer_metadata.read_cache_sidecar(
            _bank_analysis_cache_dir(dataset_id), cache_ref) is not None)
    except Exception:  # noqa: BLE001 - cleanup must never hide the primary failure
        logger.exception('could not verify Bank analysis sidecar ownership')
        return False


def _remove_unreferenced_bank_analysis_cache(dataset_id, cache_ref) -> None:
    if (cache_ref
            and not _bank_analysis_cache_is_referenced(dataset_id, cache_ref)):
        bank_transfer_metadata.remove_cache_sidecar(
            _bank_analysis_cache_dir(dataset_id), cache_ref)


def _restore_from_trash(trashed_path, original_path) -> None:
    """Best-effort filesystem compensation when a matching DB commit fails."""
    if not trashed_path or not original_path or not os.path.exists(trashed_path):
        return
    try:
        if os.path.exists(original_path):
            logger.error('cannot restore trashed path because destination exists: %s',
                         original_path)
            return
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        shutil.move(trashed_path, original_path)
    except OSError:
        # The bytes are still recoverable in Trash; never mask the DB exception.
        logger.exception('failed to restore %s from Trash after DB rollback',
                         original_path)


def _img_path(img) -> str:
    return os.path.join(_dataset_dir(img.dataset_id), img.filename)


def _invalidate_image_content_analysis(img):
    """Drop derived content and face analysis after a pixel-level mutation.

    The content cache doubles as the optimistic-concurrency token for a running
    one-image face score.  Clearing both fields is intentionally cheap: the
    next training snapshot simply re-hashes this one file.
    """
    img.content_sig = None
    img.content_sig_stat = None
    img.face_state = None
    img.face_score = None


def _ref_path(ds) -> str:
    return os.path.join(_dataset_dir(ds.id), ds.ref_filename)


# (path, mtime_ns, size) -> (w, h) | None. dataset_payload is POLLED, and it
# measured the reference on every single call: sub-millisecond, but a fresh disk
# open on a hot path, forever. Keyed on the file's identity rather than its name,
# so re-cropping the reference (which rewrites the same filename) invalidates the
# entry by itself — a stale shape here would silence the "your square reference
# will squeeze the body shots" warning, or raise a false one. Small and bounded:
# a handful of reference files per install, cleared wholesale when it grows.
_PIXEL_SIZE_CACHE: dict = {}
_PIXEL_SIZE_CACHE_MAX = 512


def image_pixel_size(path):
    """(w, h) of an image file, or None when it cannot be measured.

    PIL reads the header only — no decode — and the answer is cached per
    (path, mtime, size). The dataset payload exposes the reference dimensions
    for clients that need to describe or crop the source; Krea dataset cards now
    choose their own target aspect through the Fit v1.2 path. Degrades to None
    on ANY failure (missing file, exotic format, Pillow absent): an unmeasurable
    image must never turn a payload read into a 500. A file that cannot be
    stat'ed is measured without caching — never guessed."""
    try:
        st = os.stat(path)
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if key in _PIXEL_SIZE_CACHE:
        return _PIXEL_SIZE_CACHE[key]
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        size = (int(w), int(h)) if w and h and w > 0 and h > 0 else None
    except Exception:
        size = None
    if len(_PIXEL_SIZE_CACHE) >= _PIXEL_SIZE_CACHE_MAX:
        _PIXEL_SIZE_CACHE.clear()
    _PIXEL_SIZE_CACHE[key] = size
    return size


_VALID_STATUS = ('pending', 'keep', 'reject', 'failed')
MAX_FANOUT = 60


def fanout_in_flight(dataset_id) -> int:
    """Generations already queued on this dataset (pending row, no file yet)."""
    return (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).count())


def check_fanout_budget(dataset_id, total):
    """Refuse a WHOLE multi-engine batch up front when it would blow MAX_FANOUT.

    generate_variations / generate_variations_nanobanana each enforce the cap on
    their own call, which is enough for a single engine but NOT for a run split
    across several: three 25-image calls each pass individually while the run
    totals 75, and the third one would be refused only after the first two had
    already created rows — a half-dispatched batch. The multi-engine route calls
    this with the aggregate BEFORE dispatching anything, so the run is all-or-
    nothing. The per-call checks stay as defense in depth."""
    total = int(total)
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    in_flight = fanout_in_flight(dataset_id)
    if in_flight + total > MAX_FANOUT:
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
# Shown when a delete can't move a file to Trash because it's still open in
# another process (typically an antivirus scan of a just-cleaned image, or an
# open preview). Raised as a RuntimeError so the route maps it to a clean 409
# toast instead of a bare 500. The dataset is left fully intact (DB + disk).
_TRASH_LOCK_MESSAGE = (
    "Couldn't delete this because one of its files is still open in another "
    "program — most often an antivirus scan of a just-cleaned image, or an open "
    "preview. Close it or wait a few seconds, then try again.")
# Shown when a delete is refused because a training run (local or cloud) is still
# running on the dataset. Deleting under it would orphan the run's provenance row
# and — for a cloud run — leave a paid vast pod training against images we just
# trashed. RuntimeError -> 409 (routes._common._map_error); dataset untouched.
_ACTIVE_RUN_TEMPLATE = (
    'A training run is active on this dataset — stop it (or let it finish) '
    'before {action}.')
_ACTIVE_RUN_MESSAGE = _ACTIVE_RUN_TEMPLATE.format(action='deleting')
SMALL_IMAGE_SOURCE = 'small_image_source'
KLEIN_SMALL_IMAGE = 'klein_small_image'
KLEIN_IMAGE_IMPROVE = 'klein_image_improve'

# The three "Upscale & improve" knobs live in config (klein.improve_*). Read
# through clamps: a hand-edited config with a string, a negative or a wild value
# must degrade the pass to something sane, never raise inside the enqueue path.
_IMPROVE_MAX_STRENGTH = 2.0
_IMPROVE_MAX_STEPS = 50


# Config keys renamed after they shipped. improve_character_lora_strength was a
# MISNOMER: the value drives klein.consistency_strength (composition anchoring),
# never an identity LoRA. Renamed rather than left lying, but a value already saved
# under the old name must keep working — config keys live in users' config.json.
_IMPROVE_KEY_ALIASES = {
    'improve_consistency_strength': ('improve_character_lora_strength',),
}


def _improve_float(key, default, ceiling=_IMPROVE_MAX_STRENGTH) -> float:
    """Per-key ceiling: the consistency LoRA is itself clamped to 1.5 downstream, and
    the megapixel budget is a resolution, not a strength — one shared ceiling would
    either lie to the user or silently cap a value the UI had offered."""
    raw = cfg.get(f'klein.{key}')
    # cfg.get merges the shipped defaults, so the new key NEVER reads as absent —
    # "still at its default" is what actually means "the user has not set this one",
    # and only then may a value saved under the old name speak for it.
    if raw is None or raw == default:
        for legacy in _IMPROVE_KEY_ALIASES.get(key, ()):
            legacy_value = cfg.get(f'klein.{legacy}')
            if legacy_value is not None:
                raw = legacy_value
                break
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(ceiling, v))


def _improve_int(key, default) -> int:
    try:
        v = int(cfg.get(f'klein.{key}'))
    except (TypeError, ValueError):
        return default
    return max(1, min(_IMPROVE_MAX_STEPS, v))


def _generation_steps() -> int:
    """Sampler steps for a Klein GENERATION job (variations, regenerate, small-image
    rescue). The shipped workflow hardcodes 5 at node 77 and nothing ever passed
    `sampler_steps` on these paths, so the knob existed but was unreachable
    (reported by ashish.sinha, Discord). Default 5 = that exact behaviour; a bad
    config value degrades to it rather than crashing the enqueue."""
    return _improve_int('generation_steps', 5)


def _generation_base_lora_strength() -> float:
    """Enhancement-LoRA strength (node 139, klein/realistic.safetensors) for every
    Klein lane that is NOT "Upscale & improve": the reference edit, variations,
    regenerate, and the small-image rescue.

    The shipped workflow pins that node at 0.8 and none of those lanes ever passed
    a value, so the LoRA applied at full 0.8 with nothing to turn it down. That was
    invisible while the file existed on no install (enqueue_klein_edit bypasses a
    missing LoRA), and became real the day Setup started downloading it
    (klein_enhancement_lora, 031766f): a detail/style LoRA at 0.8 on top of every
    edit steers the render toward its own look instead of the instruction — the
    "Klein edits are not conformant" report.

    Default 0.0 = what every install rendered before that download existed, and the
    same default the improve pass already carries. Raising it is now a choice, on a
    setting, instead of a hardcoded workflow widget."""
    return _improve_float('edit_base_lora_strength', 0.0)
# KLEIN_IMAGE_IMPROVE_PROMPT is the shipped DEFAULT of the editable klein_improve
# prompt (imported from face_variations, which owns the identity/quality prompt
# registry). Re-exported here so `svc.KLEIN_IMAGE_IMPROVE_PROMPT` keeps resolving.
_SMALL_IMAGE_DERIVATIONS = (SMALL_IMAGE_SOURCE, KLEIN_SMALL_IMAGE)
# A striped in-process lock is sufficient for LDS's single local server process
# and makes the active-candidate check + row creation + enqueue one critical
# section.  In particular, a second simultaneous lightbox click waits until the
# first row has its job_id, then takes the idempotent return path below.
_IMAGE_IMPROVE_LOCKS = tuple(threading.Lock() for _ in range(64))
# An in-place pixel edit is a fold on the CURRENT file: two requests for the same
# image must run in order (two mirror clicks restore the original orientation,
# four rotate-right clicks come back round), not both read the same source pixels
# and race to promote a result computed from the same "before".  Mirror and
# rotation deliberately share ONE stripe set so they serialize against each other
# too.  Stripes avoid an unbounded lock map.
_IMAGE_PIXEL_EDIT_LOCKS = tuple(threading.Lock() for _ in range(64))
# Face scoring starts a heavyweight CPU subprocess.  Stripes keep one dataset's
# requests serial without retaining an unbounded lock map; a collision only
# makes an unrelated request retry, never permits concurrent scorers.
_FACE_SCORING_LOCKS = tuple(threading.Lock() for _ in range(64))
# LDS runs one threaded Flask process (backend/run.py and Dockerfile). Striped
# locks therefore serialize dataset dedupe snapshots without an unbounded map.
# RLock permits a promotion to hold the stripe across all chunks while nested
# import_images calls retain the same protection.
_DATASET_INGEST_LOCKS = tuple(threading.RLock() for _ in range(64))
_FACE_SCORING_BUSY_DETAIL = 'face scoring is already running; try again shortly'


def _face_scoring_lock(dataset_id):
    return _FACE_SCORING_LOCKS[hash(str(dataset_id)) % len(_FACE_SCORING_LOCKS)]


def _dataset_ingest_lock(user_id, dataset_id):
    return _DATASET_INGEST_LOCKS[
        hash((str(user_id), str(dataset_id))) % len(_DATASET_INGEST_LOCKS)]


def _guard_not_bank_export(dataset_id, activity_token=None):
    exclusive = dataset_activity.exclusive_kind(dataset_id)
    if (exclusive and dataset_activity.owns_exclusive(
            dataset_id, activity_token, exclusive)):
        return
    if exclusive == 'bank_export':
        raise RuntimeError(
            'This dataset is being copied to a Bank. Wait for that copy to '
            'finish before editing or deleting it.')
    if exclusive == 'bank_import':
        raise RuntimeError(
            'This dataset is receiving images from a Bank. Wait for that copy '
            'to finish before editing or deleting it.')
    if exclusive == 'training_export':
        raise RuntimeError(
            'This dataset is being frozen for training. Wait for that export '
            'to finish before editing or deleting it.')
    if exclusive == 'backup':
        raise RuntimeError(
            'This dataset is being backed up. Wait for that backup to finish '
            'before editing or deleting it.')
    if exclusive:
        raise RuntimeError(
            f'This dataset has an exclusive {exclusive} operation in progress. '
            'Wait for it to finish before editing or deleting it.')


def _serialize_dataset_ingest(fn):
    @wraps(fn)
    def wrapped(user_id, dataset_id, *args, **kwargs):
        activity_token = kwargs.pop('_dataset_activity_token', None)
        with _dataset_ingest_lock(user_id, dataset_id):
            _guard_not_bank_export(dataset_id, activity_token)
            return fn(user_id, dataset_id, *args, **kwargs)
    return wrapped


def _serialize_dataset_image_ingest(fn):
    @wraps(fn)
    def wrapped(user_id, image_id, *args, **kwargs):
        image = db.session.get(FaceDatasetImage, image_id)
        if image is None:
            return fn(user_id, image_id, *args, **kwargs)
        with _dataset_ingest_lock(user_id, image.dataset_id):
            _guard_not_bank_export(image.dataset_id)
            return fn(user_id, image_id, *args, **kwargs)
    return wrapped


def _face_scoring_busy_error():
    return {'kind': 'busy', 'detail': _FACE_SCORING_BUSY_DETAIL}



class KleinNodesMissing(Exception):
    """Klein graph preflight failure carried from the service to the HTTP mapper."""

    def __init__(self, missing, missing_nodes):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('Klein custom nodes are missing')


# Références ADDITIONNELLES par dataset (au-delà de la principale) : servent
# UNIQUEMENT Nano Banana (multi-images d'entrée) - Klein/crop/scoring restent
# sur la principale. Cap bas pour garder des payloads API légers.
MAX_EXTRA_REFS = 3

# A reference handed to an external image API is a disposable transport copy,
# never a reason to disclose a camera master byte-for-byte.  Keep the same 2048px
# API-transport convention as ``normalize_to_webp`` and a modest raw-input cap:
# three dataset extras plus transient anchors must not turn a base64 request into
# an unbounded allocation.  The cap applies again to persistent legacy refs read
# from disk, because a restored backup can contain a preserved JPEG/PNG/BMP.
EXTERNAL_REFERENCE_MAX_BYTES = 25 * 1024 * 1024
EXTERNAL_REFERENCE_MAX_SIDE = 2048
# The modal exposes three request-scoped anchors. Enforce the same bound before
# route reads so a hand-written multipart request cannot create an unbounded
# in-memory snapshot.
MAX_EDIT_REFERENCE_UPLOADS = 3


def extra_ref_filenames(ds) -> list:
    """Références additionnelles du dataset (JSON en base, parse tolérant)."""
    try:
        v = json.loads(ds.ref_extra_filenames or '[]')
    except (ValueError, TypeError):
        return []
    return [f for f in v if isinstance(f, str)] if isinstance(v, list) else []


def sanitize_external_reference(image_bytes: bytes, *, label: str = 'reference image') -> bytes:
    """Return a bounded, upright, metadata-free WebP for an external API.

    This is an egress boundary, deliberately separate from dataset storage:
    local masters stay untouched, while every API engine receives fresh pixels
    without EXIF/XMP/GPS/ICC payloads.  It also rejects unsupported/animated
    containers and unsafe headers before asking Pillow to decode pixels.
    """
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise ValueError(f'{label} must be a non-empty image')
    raw = bytes(image_bytes)
    if len(raw) > EXTERNAL_REFERENCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large (max {EXTERNAL_REFERENCE_MAX_BYTES // (1024 * 1024)} MiB)')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                # This shared header validator admits only static JPEG/PNG/WebP/BMP
                # and applies the global side/pixel budget before ``load``.
                _preserved_import_header_extension(source, label=label)
                source.load()
                oriented = ImageOps.exif_transpose(source)
                has_alpha = ('A' in oriented.getbands()
                             or 'transparency' in getattr(oriented, 'info', {}))
                if has_alpha:
                    rgba = oriented.convert('RGBA')
                    clean = Image.new('RGB', rgba.size, (255, 255, 255))
                    clean.paste(rgba, mask=rgba.getchannel('A'))
                else:
                    clean = Image.new('RGB', oriented.size)
                    clean.paste(oriented.convert('RGB'))
        clean.thumbnail((EXTERNAL_REFERENCE_MAX_SIDE, EXTERNAL_REFERENCE_MAX_SIDE),
                        Image.LANCZOS)
        out = io.BytesIO()
        # ``clean`` is a fresh canvas, so no source EXIF/XMP/GPS/ICC metadata can
        # reach an external provider even on Pillow format-specific save paths.
        clean.save(out, 'WEBP', quality=92)
        payload = out.getvalue()
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f'{label} rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, SyntaxError, MemoryError) as exc:
        raise ValueError(f'{label} is unreadable') from exc
    if len(payload) > EXTERNAL_REFERENCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large after preparation '
            f'(max {EXTERNAL_REFERENCE_MAX_BYTES // (1024 * 1024)} MiB)')
    return payload


def _read_external_reference(path, *, label: str) -> bytes:
    """Read at most one external-reference budget, then sanitize exact bytes."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read(EXTERNAL_REFERENCE_MAX_BYTES + 1)
    except (OSError, TypeError, MemoryError) as exc:
        raise ValueError(f'{label} is unavailable') from exc
    if len(raw) > EXTERNAL_REFERENCE_MAX_BYTES:
        raise ValueError(
            f'{label} is too large (max {EXTERNAL_REFERENCE_MAX_BYTES // (1024 * 1024)} MiB)')
    return sanitize_external_reference(raw, label=label)


def _all_ref_bytes(ds) -> list:
    """Safe external derivatives of primary then persistent extra references.

    The primary remains mandatory.  A missing/corrupt legacy extra is skipped
    like it was before, but it can never leak raw EXIF/GPS bytes to an API.
    """
    out = [_read_external_reference(_ref_path(ds), label='primary reference')]
    for fn in extra_ref_filenames(ds):
        p = os.path.join(_dataset_dir(ds.id), fn)
        try:
            out.append(_read_external_reference(p, label='extra reference'))
        except ValueError:
            # An old/hand-edited extra is optional identity context. Do not turn
            # it into an opaque provider error, but do not send raw fallback bytes.
            logger.warning('dataset %s: skipping unavailable/unsafe extra reference', ds.id)
    return out


_EXTRA_REF_MARKER = '_datasetrefx_'
_EXTRA_REF_ORIG_MARKER = '_datasetrefxorig_'


def extra_ref_original_name(filename):
    """Name of the full-frame ORIGINAL kept beside an extra reference
    (`..._datasetrefx_<id>.webp` -> `..._datasetrefxorig_<id>.webp`), or None when
    the name doesn't follow the convention. A NAMING convention rather than a new
    column: extras live in `ref_extra_filenames`, a JSON list of names inside a
    schema that user databases froze long ago — deriving the companion needs no
    migration and restores from a backup as-is."""
    if not isinstance(filename, str) or _EXTRA_REF_ORIG_MARKER in filename:
        return None
    if _EXTRA_REF_MARKER not in filename:
        return None
    return filename.replace(_EXTRA_REF_MARKER, _EXTRA_REF_ORIG_MARKER, 1)


def extra_ref_crop_source(ds, filename) -> str:
    """The file the ✂ editor must display for an extra reference: the kept
    full-frame ORIGINAL when there is one, else the extra itself (still fully
    croppable — see crop_extra_ref, which snapshots it on the first crop)."""
    orig = extra_ref_original_name(filename)
    if orig and os.path.isfile(os.path.join(_dataset_dir(ds.id), orig)):
        return orig
    return filename


def add_extra_ref(user_id, dataset_id, image_bytes) -> str:
    """Ajoute une référence additionnelle. Normalisée WEBP ratio conservé, SANS
    head-crop GPU : un plan buste/corps est une bonne réf d'identité pour Nano
    Banana, et l'upload ne doit pas dépendre de la fenêtre GPU. Retourne le nom
    de fichier ; ValueError si dataset absent, réf principale manquante ou cap."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('set the primary reference first')
    extras = extra_ref_filenames(ds)
    if len(extras) >= MAX_EXTRA_REFS:
        raise ValueError(f'{MAX_EXTRA_REFS} extra references max')
    fn = f"{user_id}_datasetrefx_{uuid.uuid4().hex[:8]}.webp"
    dsdir = _dataset_dir(dataset_id)
    # Keep the full-frame ORIGINAL beside it (same deal as the primary reference):
    # ✂ Crop reads the original, so a re-crop can widen back out instead of only
    # eating further into the previous crop.
    orig_fn = extra_ref_original_name(fn)
    write_image_atomic(os.path.join(dsdir, orig_fn),
                       normalize_to_webp(image_bytes, size=2048))
    write_image_atomic(os.path.join(dsdir, fn), normalize_to_webp(image_bytes))
    ds.ref_extra_filenames = json.dumps(extras + [fn])
    db.session.commit()
    return fn


def crop_extra_ref(user_id, dataset_id, filename, x, y, w, h) -> bool:
    """Manually crop ONE extra reference to (x,y,w,h), long side capped at 1024
    (never enlarged - a smaller box keeps its own pixels).
    The box is in the crop SOURCE's pixel space (what extra_ref_crop_source names,
    i.e. what the editor displayed) and the result overwrites the extra only — the
    original stays untouched, so re-crops widen as freely as they tighten.

    `filename` is client-supplied: membership in the dataset's stored extras is the
    path guard (identical to remove_extra_ref) — nothing derived from it is opened
    before that check passes."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    extras = extra_ref_filenames(ds)
    if filename not in extras:
        return False
    dsdir = _dataset_dir(dataset_id)
    dst = os.path.join(dsdir, filename)
    if not os.path.isfile(dst):
        return False
    orig = extra_ref_original_name(filename)
    src = os.path.join(dsdir, orig) if orig else None
    if src and not os.path.isfile(src):
        # Retrofit for extras imported before originals were kept: what's on disk
        # IS still the uncropped full frame (cropping is the only thing that ever
        # rewrites an extra), so snapshotting it now costs one copy and gives those
        # datasets the same widen-back-out behaviour as a fresh import — instead of
        # "works for future imports only".
        shutil.copyfile(dst, src)
    ok, _scale = _crop_resize_file(src or dst, x, y, w, h, dst=dst)
    return ok


def remove_extra_ref(user_id, dataset_id, filename) -> bool:
    """Retire une référence additionnelle, en plaçant son fichier en corbeille."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    extras = extra_ref_filenames(ds)
    if filename not in extras:
        return False
    original_path = os.path.join(_dataset_path(dataset_id), filename)
    trashed_path = None
    if os.path.exists(original_path):
        trashed_path = trash.send_to_trash(
            original_path, context=f'dataset-{dataset_id}-extra-ref')
    try:
        ds.ref_extra_filenames = json.dumps([f for f in extras if f != filename])
        db.session.commit()
    except Exception:
        db.session.rollback()
        _restore_from_trash(trashed_path, original_path)
        raise
    # The kept original follows its extra to the trash — never leave it orphaned in
    # the dataset folder. Best effort: losing the extra itself is what matters.
    orig = extra_ref_original_name(filename)
    orig_path = os.path.join(_dataset_path(dataset_id), orig) if orig else None
    if orig_path and os.path.exists(orig_path):
        try:
            trash.send_to_trash(orig_path, context=f'dataset-{dataset_id}-extra-ref')
        except OSError:
            logger.warning(f'dataset {dataset_id}: could not trash extra-ref original {orig}')
    return True


# --- CRUD ------------------------------------------------------------------
# Natures de dataset. 'concept' inverse la logique personnage (cf import_images /
# caption_images). 'style' = esthétique globale : captions de CONTENU pur (le style
# n'est jamais décrit → il est absorbé par le LoRA), pas de trigger dans les captions
# ni dans la config. Tout le reste (dont NULL) = 'character' (défaut historique).
DATASET_KINDS = ('character', 'concept', 'style')


def normalize_kind(kind) -> str | None:
    """'concept'/'style' -> tels quels ; tout le reste -> None (character, stocké NULL)."""
    k = (kind or '').strip().lower()
    return k if k in ('concept', 'style') else None


def _safe_json(text):
    """None-safe json.loads for TEXT columns holding JSON (never raises)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


_PEXELS_PAGE_HOSTS = frozenset({'pexels.com', 'www.pexels.com'})
_PEXELS_IMAGE_HOSTS = frozenset({'images.pexels.com'})
_SOURCE_URL_MAX_CHARS = 2048
_PHOTOGRAPHER_MAX_CHARS = 160


def _safe_public_https_url(value):
    """URL https sans credentials, longueur bornée — hôte LIBRE. Les résultats de
    recherche web viennent de sites arbitraires ; c'est la FORME qu'on contrôle
    ici, le contenu ayant déjà été validé par le fetch durci de l'import."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if (not trimmed or len(trimmed) > _SOURCE_URL_MAX_CHARS
            or any(ord(ch) < 32 for ch in trimmed)):
        return None
    try:
        parsed = urlsplit(trimmed)
    except ValueError:
        return None
    if (parsed.scheme != 'https' or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        return None
    return trimmed


def _safe_source_https_url(value, allowed_hosts):
    """Return a stripped HTTPS URL on an exact allowlisted host, else None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if (not value or len(value) > _SOURCE_URL_MAX_CHARS
            or any(ord(ch) < 32 for ch in value)):
        return None
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or '').lower()
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (parsed.scheme != 'https' or host not in allowed_hosts
            or parsed.username is not None or parsed.password is not None
            or port not in (None, 443)):
        return None
    return value


def normalize_source_metadata(value, *, image_url=None):
    """Validate the generic provenance object currently supported by LDS.

    Unknown platforms are deliberately dropped for backwards compatibility.
    Pexels provenance is accepted only when both attribution links are exact
    Pexels HTTPS hosts; at scrape-import time the downloaded image must also be
    hosted by the official Pexels image CDN. Extra keys never reach storage or
    the dataset payload.

    Web-search provenance keeps only the page the image was found on: there is
    no photographer to credit and the image can be hosted anywhere.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    platform = value.get('platform')
    if platform == 'websearch':
        # Recherche web : aucun photographe à créditer, seulement la page où
        # l'image a été trouvée. `image_url` n'est pas restreint à un CDN ici —
        # une image du web ouvert est hébergée n'importe où.
        source_url = _safe_public_https_url(value.get('source_url'))
        return {'platform': 'websearch', 'source_url': source_url} if source_url else None
    if platform != 'pexels':
        return None
    if image_url is not None and not _safe_source_https_url(
            image_url, _PEXELS_IMAGE_HOSTS):
        return None
    photographer = value.get('photographer')
    if not isinstance(photographer, str):
        return None
    photographer = photographer.strip()
    if not photographer or len(photographer) > _PHOTOGRAPHER_MAX_CHARS:
        return None
    photographer = ' '.join(photographer.split())
    source_url = _safe_source_https_url(value.get('source_url'), _PEXELS_PAGE_HOSTS)
    photographer_url = _safe_source_https_url(
        value.get('photographer_url'), _PEXELS_PAGE_HOSTS)
    if not source_url or not photographer_url:
        return None
    return {
        'platform': 'pexels',
        'source_url': source_url,
        'photographer': photographer,
        'photographer_url': photographer_url,
    }


def _source_metadata_storage(value, *, image_url=None):
    metadata = normalize_source_metadata(value, image_url=image_url)
    return (json.dumps(metadata, ensure_ascii=False, separators=(',', ':'))
            if metadata else None)


def _source_metadata_from_scrape_item(item):
    if not isinstance(item, dict) or item.get('platform') not in ('pexels', 'websearch'):
        return None
    return normalize_source_metadata(item, image_url=item.get('url'))


def _watermark_regions_payload(img) -> dict:
    """Return the nullable stored override and the editor's always-list value."""
    stored = _safe_json(img.watermark_regions)
    if not isinstance(stored, list):
        stored = None
    if stored is not None:
        effective = stored
    else:
        bbox = _safe_json(img.watermark_bbox)
        effective = ([bbox] if img.watermark_state == 'detected'
                     and isinstance(bbox, list) and len(bbox) == 4 else [])
    return {
        'watermark_regions': stored,
        'effective_watermark_regions': effective,
    }


def is_concept(ds) -> bool:
    return bool(ds) and (getattr(ds, 'kind', None) or '').lower() == 'concept'


def is_style(ds) -> bool:
    return bool(ds) and (getattr(ds, 'kind', None) or '').lower() == 'style'


def is_conceptual(ds) -> bool:
    """Concept OU style : les kinds où l'invariant du set n'est PAS une identité.
    Regroupe les comportements communs : heuristiques personnage (équilibre de
    composition, fuite d'identité) sans objet, masques personne interdits (ils
    effaceraient ce qu'on apprend), barème de steps sous-linéaire (√n)."""
    return is_concept(ds) or is_style(ds)


def face_masking_enabled(ds) -> bool:
    """True when a CONCEPT dataset opted into face masking (Advanced training
    options). Reported by shivdbz2010 (GitHub issue #15): a concept LoRA also
    learns the faces of its dataset and then fights a character LoRA over the
    identity; masking the faces teaches the act without the identity.

    OPT-IN, and deliberately stored in the train_settings JSON blob rather than
    on the request (like dual_captions): `masked` already threads through seven
    call sites in routes/training.py plus the cloud lane, and a parallel flag
    would double that. One read, at export time — which also means the local
    queue, the scheduler, a cloud run and a re-run of an OLD dataset all inherit
    it without a single extra line, and no existing dataset changes behaviour.

    Concept only. A Character wants its identity learned, and a Style must learn
    how it renders a face — masking there would amputate the thing being taught."""
    if not ds or not is_concept(ds):
        return False
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get('mask_faces'))
    except (ValueError, TypeError):
        return False


# Person masking (background down-weighted to 10 %) has always defaulted to ON.
# Keep the constant next to the reader: the default is what makes the migration
# to a stored setting a no-op for every dataset that never touched it.
PERSON_MASK_DEFAULT = True


def person_masking_enabled(ds) -> bool:
    """Whether this dataset trains with PERSON masks (subject isolated, background
    at 10 % loss weight). Default ON.

    Used to be a `masked` query parameter carried by the browser's localStorage,
    which the server only ever saw at launch. Three consequences, all real: the
    readiness badge could not warn that a dataset set to masked would train
    unmasked for want of rembg; opening the app from a phone silently reverted to
    the default; and no run snapshot recorded it, so two runs differing only by
    masking looked identical. Stored in the train_settings JSON blob now, exactly
    like `mask_faces` — one read, at export time, so the local queue, the
    scheduler, a cloud run and a re-run of an OLD dataset all inherit it.

    ABSENT key = the historical default (True): no existing dataset changes
    behaviour by upgrading. An explicit False is a VALUE, not a falsy no-op —
    the opposite of `mask_faces`, whose default is OFF.

    Concept/Style are forced OFF here, mirroring the export guard: a person mask
    erases a concept, and an always-on style must learn the whole frame."""
    if not ds:
        return PERSON_MASK_DEFAULT
    if is_conceptual(ds):
        return False
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return PERSON_MASK_DEFAULT
    try:
        stored = json.loads(raw).get('masked')
    except (ValueError, TypeError):
        return PERSON_MASK_DEFAULT
    return PERSON_MASK_DEFAULT if stored is None else bool(stored)


def person_masking_stored(ds):
    """The RAW stored opt-in — True / False / None when the dataset never answered.
    The panel needs the tri-state (not the resolved boolean) to know whether the
    one-time localStorage carry-over notice still has anything to disclose."""
    raw = getattr(ds, 'train_settings', None) if ds else None
    if not raw:
        return None
    try:
        stored = json.loads(raw).get('masked')
    except (ValueError, TypeError):
        return None
    return None if stored is None else bool(stored)


# Concept descriptions whose ACT lives on the face. Masking the head then erases
# the very thing being taught -- the community workflow this feature follows hit
# exactly this and had to subtract the mouth back out of its face masks. We WARN
# and let the user decide (they know their dataset); we never block.
_FACE_ANCHORED = frozenset({
    'face', 'faces', 'facial', 'head', 'mouth', 'lips', 'lip', 'tongue', 'teeth',
    'throat', 'chin', 'jaw', 'cheek', 'cheeks', 'eye', 'eyes', 'gaze', 'stare',
    'staring', 'expression', 'smile', 'smiling', 'grimace', 'ahegao', 'blowjob',
    'kiss', 'kissing', 'licking', 'lick', 'sucking', 'suck', 'oral', 'deepthroat',
    'facesitting', 'cum', 'cumshot', 'facial_expression', 'nose', 'ear', 'ears',
})


def concept_face_conflict(ds) -> bool:
    """True when this concept's own description names the face/mouth/gaze — i.e.
    when face masking would likely mask away the concept itself. Derived from the
    dataset's concept_desc, never a global list of 'risky' concepts."""
    if not ds or not is_concept(ds):
        return False
    toks = set(re.split(r'[^a-z]+', (getattr(ds, 'concept_desc', '') or '').lower()))
    return bool(toks & _FACE_ANCHORED)


def dual_captions_enabled(ds) -> bool:
    """True when the dataset opted into ai-toolkit dual long+short captioning (Advanced
    training options). The flag lives in the train_settings JSON blob (like the other
    expert levers); default OFF = the historical single-caption behaviour, byte-for-byte.
    Self-contained JSON read so lora_training can reuse it without a circular import."""
    if not ds:
        return False
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return False
    try:
        return bool(json.loads(raw).get('dual_captions'))
    except (ValueError, TypeError):
        return False


# --- Per-dataset caption method options --------------------------------------
# The Captions ⚙️ Options popover writes these to the caption_options JSON column.
# All three are OVERRIDES of the global captioning defaults: an empty value means
# "follow the global default" (captioning.backend / ollama.vision_model), so a
# dataset that never touched the popover behaves byte-for-byte as before.
_CAPTION_BACKENDS = ('auto', 'joycaption', 'ollama', 'none')
# Extra instructions are APPENDED to the built caption prompt (never replace it),
# so the kind rules (identity/concept/style omission) and the post-filter cleaners
# stay in force — a bounded length keeps a runaway paste out of every prompt.
_CAPTION_INSTRUCTIONS_MAX = 2000

# Vocabulary preset: a one-click register for how the model names nude/sexual content.
# '' = untouched (the model's own default — byte-identical to the pre-preset behaviour).
# This is the NSFW lane: paired with an abliterated Ollama model, 'explicit' makes the
# captioner spell the acts out. It rides in as an appended instruction just like the free
# text, so the identity/concept omission rules and the leak cleaners still post-filter —
# the preset changes wording, never what is allowed to bind to the trigger. NONE of the
# cleaners (identity/body/concept ban-lists) touch crude anatomical vocabulary, so an
# 'explicit' caption keeps its terms; only identity/concept leaks are ever scrubbed.
_CAPTION_VOCABULARIES = ('explicit', 'clinical', 'safe')
_VOCABULARY_INSTRUCTION = {
    'explicit': (
        'Describe any sexual, nude or intimate content directly and explicitly, using '
        'plain, crude anatomical terms. Do not censor, euphemise, hedge or soften — name '
        'the body parts and the acts as they are.'),
    'clinical': (
        'Describe any nude or sexual content in neutral, clinical, anatomical terms. '
        'Avoid crude slang and euphemism alike — be precise and matter-of-fact.'),
    'safe': (
        'Keep the description strictly non-explicit. Do not use sexual or crude terms; '
        'refer to any nudity only in general, non-graphic language.'),
}

# Length preset (idea by djpraxis on Reddit): how LONG the caption should be, on an axis
# ORTHOGONAL to the vocabulary register. '' = standard — nothing appended, byte-identical
# to the pre-preset prompt, which is why the default stays the empty string forever.
#
# Two things these texts must never stop doing:
#  1. STAY PROSE. 'concise' explicitly forbids a comma-separated tag list, because
#     face_variations.caption_style() votes 'booru' on >=3 short comma segments with no
#     sentence punctuation — a "short caption" phrased as key phrases would trip the
#     MISMATCH_CAPTION guard at training launch on every prose family. Asking for one
#     full sentence keeps a Concise dataset trainable without a force.
#  2. STAY A LENGTH, NOT A CONTENT RULE. The kind omission rules (identity / concept /
#     style) are built into the base prompt and post-filtered by the cleaners; a preset
#     that told the model WHAT to leave out would fight them. These only say how much.
#
# 'concise' is NOT the short half of dual long+short captioning: that one is DERIVED from
# the stored long caption by a separate text pass (_SHORTEN_BASE) and lives in its own
# column. Concise changes the long caption itself. The two axes compose.
_CAPTION_LENGTHS = ('concise', 'detailed')
_LENGTH_INSTRUCTION = {
    'concise': (
        'Keep the caption SHORT: ONE single sentence of roughly 20 to 30 words, naming '
        'only the subject, the pose or action, the clothing and the setting. Write it as '
        'a complete sentence in plain prose — never a comma-separated list of tags or key '
        'phrases. Do not add a second sentence, a list, or any commentary.'),
    'detailed': (
        'Write a DETAILED caption: several complete sentences in plain prose, covering the '
        'subject and pose, the expression, the clothing and accessories, the setting and '
        'background, the lighting, and the camera framing and angle. Describe only what is '
        'clearly visible — do not speculate, and do not add commentary about the image.'),
}


def caption_options(ds) -> dict:
    """Normalized per-dataset caption overrides: {backend, ollama_model, instructions}.
    Empty strings = "use the global default". Never raises ({} defaults on a missing or
    corrupt blob) so every caption path can read it unconditionally."""
    out = {'backend': '', 'ollama_model': '', 'instructions': '', 'vocabulary': '',
           'length': ''}
    raw = getattr(ds, 'caption_options', None) if ds else None
    if not raw:
        return out
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return out
    if not isinstance(data, dict):
        return out
    backend = str(data.get('backend') or '').strip().lower()
    if backend in _CAPTION_BACKENDS:
        out['backend'] = backend
    try:
        out['ollama_model'] = normalize_ollama_model_ref(
            data.get('ollama_model', ''), allow_empty=True)
    except ValueError:
        # Legacy/manual DB blobs are untrusted input too. Keep every other valid
        # option but fall back to the global model instead of propagating a bad ref.
        out['ollama_model'] = ''
    out['instructions'] = str(data.get('instructions') or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    vocab = str(data.get('vocabulary') or '').strip().lower()
    if vocab in _CAPTION_VOCABULARIES:
        out['vocabulary'] = vocab
    length = str(data.get('length') or '').strip().lower()
    if length in _CAPTION_LENGTHS:
        out['length'] = length
    return out


def set_caption_options(user_id, dataset_id, patch) -> dict:
    """Persist a caption-options patch (only the provided keys change). An invalid engine
    raises ValueError (mapped 400 by the route). Empty keys are dropped so a fully-default
    dataset stores NULL — identical to one that never opened the popover. Returns the
    resulting normalized options."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur = caption_options(ds)
    if 'backend' in patch:
        b = str(patch.get('backend') or '').strip().lower()
        if b and b not in _CAPTION_BACKENDS:
            raise ValueError(f'invalid captioning backend: {b}')
        cur['backend'] = b
    if 'ollama_model' in patch:
        cur['ollama_model'] = normalize_ollama_model_ref(
            patch.get('ollama_model'), allow_empty=True)
    if 'instructions' in patch:
        cur['instructions'] = str(patch.get('instructions') or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    if 'vocabulary' in patch:
        v = str(patch.get('vocabulary') or '').strip().lower()
        if v and v not in _CAPTION_VOCABULARIES:
            raise ValueError(f'invalid caption vocabulary: {v}')
        cur['vocabulary'] = v
    if 'length' in patch:
        ln = str(patch.get('length') or '').strip().lower()
        if ln and ln not in _CAPTION_LENGTHS:
            raise ValueError(f'invalid caption length: {ln}')
        cur['length'] = ln
    stored = {k: v for k, v in cur.items() if v}
    ds.caption_options = json.dumps(stored) if stored else None
    db.session.commit()
    return cur


# --- Which Klein model this dataset runs on ----------------------------------
# Stored on the DATASET, not in localStorage: it describes what the dataset is
# made of, so it must survive a browser change and be the same from a phone. The
# generation picker had a per-browser value (editPage_flux2KleinModel_v1) that
# improve never even read — hence "no option anywhere to choose the model used
# for improve". NULL = auto (resolve_klein_unet decides), which is exactly what
# every improve did before this setting existed.
def dataset_klein_model(ds):
    """The bare Klein model file name in use, or None for auto.

    ONE VALUE, NOT ONE PER DATASET — and the `ds` argument is now ignored on
    purpose. The setting used to live in a per-dataset column, which meant the
    same question ("which Klein model runs?") had two different answers depending
    on which screen asked, and Settings could disagree with the dataset you were
    looking at. Every model setting is now the same setting wherever it is
    edited; `klein.unet` is that value, and it is the one the resolver has always
    honoured.

    ⚠️ The `face_dataset.klein_model` COLUMN is deliberately left in place and
    simply unread. Dropping it would destroy picks made under the old behaviour,
    SQLite cannot drop a column on the versions this app supports, and an
    unread column costs nothing — while a deleted one cannot be reconsidered.
    """
    return (cfg.get('klein.unet') or '').strip() or None


def set_dataset_klein_model(user_id, dataset_id, name):
    """Persist the Klein model pick. '' / None clears it back to auto —
    un-choosing has to be a real gesture, not a value you can never take back.

    It writes the GLOBAL `klein.unet`, the same key Settings edits: choosing from
    a dataset screen and choosing from Settings are the same act on the same
    value. `dataset_id` is still required and still validated — a write from a
    screen that is not looking at a real dataset is a bug worth answering with a
    404, and dropping the argument would change every caller for nothing.

    Only a BARE file name is accepted: the picker lists bare names (the loader
    prefix is resolve_klein_unet's job), so a value carrying a path separator is
    never something the UI produced. Existence is deliberately NOT checked here —
    a model can be moved away long after it was chosen, and the honest place to
    say so is the run (KleinModelGone names the file), not a settings write that
    would silently drop the user's answer."""
    if not get_dataset(user_id, dataset_id):
        raise ValueError('dataset not found')
    value = (name or '').strip()
    # BOTH separators, on every OS: os.path.basename alone reads a backslash as
    # an ordinary character on Linux, so `sub\model.safetensors` walked straight
    # through this guard there and only Windows was actually protected.
    if value and (ntpath.basename(value) != value
                  or posixpath.basename(value) != value
                  or value in ('.', '..')):
        raise ValueError('a Klein model is named by its file name, without a folder')
    cfg.save_config({'klein': {'unet': value}})
    return dataset_klein_model(None)


# --- Who actually WROTE the captions of a pass ---------------------------------
# The 'auto' backend is the default, and it is a CHAIN, not a coin toss: JoyCaption
# drafts what it can (a whole batch in one model load), Ollama covers what it didn't
# — and on a Concept dataset Ollama rewrites Joy's drafts as well. Three different
# writing styles can therefore come out of one pass, and until now nothing told the
# user which one they were reading: the pass returned a count and nothing else, so
# "these captions don't look like last time" had no answer anywhere in the app.
# These keys are the answer, counted where the text is actually STORED.
CAPTION_WRITER_JOYCAPTION = 'joycaption'          # JoyCaption's own words
CAPTION_WRITER_OLLAMA = 'ollama'                  # written by the Ollama vision model
CAPTION_WRITER_REFINED = 'joycaption_refined'     # a JoyCaption draft rewritten by Ollama
# Stored nowhere and read by the UI as keys — treat them like catalog labels: adding
# one is free, renaming one breaks the caller that reads it.


def _writer(report, key, n=1):
    """Count one stored caption against the engine that wrote it. ``report`` is the
    caller's optional out-dict — None means nobody asked, so this costs nothing.

    Deliberately counts the DRAFTING engine, i.e. whose prose the user is reading.
    The concept pipeline's omission guard can rewrite a sentence afterwards to remove
    a banned term; that is a filter over someone else's words, not a second author,
    and counting it as one would make the numbers stop matching what people see."""
    if report is not None:
        report[key] = report.get(key, 0) + n


# How much of an engine's own refusal is worth quoting back. Long enough for the
# guard sentence that names the limit AND the offending size, short enough that a
# toast stays a toast.
_SKIP_REASON_CHARS = 200


def _first_caption_skip_reason(pending, errors) -> str:
    """One human reason for a batch that left images uncaptioned, quoted VERBATIM
    from the engine.

    Verbatim on purpose: classifying the message here ("too large", "unreadable")
    would mean this module deciding what another process meant, and the day the
    engine adds a reason nobody thought of, the user would be told a confident
    wrong category instead of the true sentence. '' when the engine said nothing —
    an empty string is how the caller knows to add no clause at all.
    """
    for entry in (pending or []):
        path = entry[1] if isinstance(entry, tuple) else entry
        reason = (errors or {}).get(path)
        if reason:
            return str(reason).strip()[:_SKIP_REASON_CHARS]
    return ''


def _record_caption_skips(outcome, pending, errors) -> None:
    """Report the images a caption pass HANDLED without captioning. ``outcome`` is
    the caller's optional out-dict — None means nobody asked."""
    if outcome is None or not pending:
        return
    outcome['skipped'] = outcome.get('skipped', 0) + len(pending)
    reason = _first_caption_skip_reason(pending, errors)
    if reason and not outcome.get('skipped_reason'):
        outcome['skipped_reason'] = reason


def _resolve_caption_backend(ds) -> str:
    """The engine a caption run uses: the dataset override when set, else the global
    captioning.backend (default 'auto')."""
    return (caption_options(ds).get('backend')
            or cfg.get('captioning.backend') or 'auto').lower()


def _with_caption_instructions(prompt: str, instructions: str) -> str:
    """Append the user's extra instructions to a built caption prompt. The base prompt
    (with its kind omission rules) stays first so the model still reads them; the extras
    ride at the end under a clear header. The output cleaners run regardless, so this can
    never reintroduce a banned identity/concept term."""
    extra = (instructions or '').strip()
    if not extra:
        return prompt
    return f'{prompt}\n\nAdditional instructions from the user:\n{extra}'


def _caption_preset_parts(vocabulary=None, length=None) -> list:
    """The preset instructions for a run, in their fixed order: vocabulary register first
    (how to name things), then length (how much to write). One list so the dataset pass,
    the Caption Lab preview and the image bank never drift on that order."""
    parts = []
    register = _VOCABULARY_INSTRUCTION.get((vocabulary or '').strip().lower())
    if register:
        parts.append(register)
    size = _LENGTH_INSTRUCTION.get((length or '').strip().lower())
    if size:
        parts.append(size)
    return parts


def _combined_caption_instructions(opts) -> str:
    """The text appended to a caption prompt for a run: the presets (vocabulary register,
    then length), then the user's free-text instructions LAST — closest to the model, so a
    hand-written steer overrides a preset it contradicts. Empty when none is set, so a
    dataset that never touched the popover produces byte-identical prompts. All of it rides
    at the END of the prompt, after the kind omission rules, and the output cleaners still
    post-filter."""
    parts = _caption_preset_parts(opts.get('vocabulary'), opts.get('length'))
    extra = (opts.get('instructions') or '').strip()
    if extra:
        parts.append(extra)
    return '\n\n'.join(parts)


# Cibles de fidélité (datasets personnage). 'body' = le LoRA reproduit AUSSI la
# morphologie : captions bannissent en plus les marques corporelles permanentes
# (elles se lient au trigger), composition recommandée plus corps/buste, import
# plein cadre par défaut.
FIDELITIES = ('face', 'body')


def normalize_fidelity(f) -> str:
    f = (f or '').strip().lower()
    return f if f in FIDELITIES else 'face'


def is_body_fidelity(ds) -> bool:
    return bool(ds) and (getattr(ds, 'fidelity', None) or 'face').lower() == 'body'


def set_fidelity(user_id, dataset_id, fidelity) -> bool:
    """Switch face-only <-> full-body fidelity later. Affects FUTURE captions
    (re-caption to apply) + the composition target + the import crop default."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    ds.fidelity = normalize_fidelity(fidelity)
    db.session.commit()
    return True


# Familles de modèle entraînables (= pipeline ai-toolkit). Source de vérité côté UI
# ET validation : choisie à la création, drive le format de caption (sdxl→booru, sinon
# prose) et le regroupement du menu. Reste modifiable ensuite (TrainingPanel).
# NB : 'flux2klein' (FLUX.2 Klein) — PAS 'klein' : ce namespace est déjà pris par
# le moteur de GÉNÉRATION (engines.klein, unet/klein/) ; un train_type 'klein'
# télescoperait les résolveurs de modèles et les chemins loras du Studio.
TRAIN_TYPES = ('zimage', 'sdxl', 'krea', 'flux', 'flux2klein', 'anima')


def normalize_train_type(t) -> str:
    """Famille valide en minuscules, défaut 'zimage' (toute valeur inconnue/None)."""
    t = (t or '').strip().lower()
    return t if t in TRAIN_TYPES else 'zimage'


# --- Prompt suffixes (creative direction, community feature request) ----------
# Free user text that rides on every generated variation: a GLOBAL suffix plus an
# optional per-framing map (same buckets as the composition). Persisted on the
# dataset row, applied at WRAP time only (never baked into variation_prompt — a
# regenerate would double-apply it). Composition: per-framing first, then global
# (see face_variations.compose_prompt_suffix).
SUFFIX_FRAMINGS = ('face', 'bust', 'body', 'back')
MAX_SUFFIX_LEN = 300


def _normalize_prompt_suffix(value):
    """Provided global-suffix string -> stripped/capped text or None (cleared)."""
    if not isinstance(value, str):
        raise ValueError('prompt_suffix must be a string')
    return value.strip()[:MAX_SUFFIX_LEN] or None


def _normalize_prompt_suffixes(value):
    """Provided per-framing map -> JSON text keeping only non-empty known keys,
    or None when nothing remains ({} therefore CLEARS the map). The whole map is
    replaced on each write — simple, predictable modal semantics."""
    if not isinstance(value, dict):
        raise ValueError('prompt_suffixes must be an object {face,bust,body,back}')
    out = {}
    for k in SUFFIX_FRAMINGS:
        v = value.get(k)
        if v is None:
            continue
        if not isinstance(v, str):
            raise ValueError(f'prompt_suffixes.{k} must be a string')
        v = v.strip()[:MAX_SUFFIX_LEN]
        if v:
            out[k] = v
    return json.dumps(out, ensure_ascii=False) if out else None


def prompt_suffixes_dict(ds) -> dict:
    """The stored per-framing suffix map as a clean dict (defensive JSON parse;
    unknown keys / non-string values dropped). {} when unset."""
    raw = getattr(ds, 'prompt_suffixes', None) if ds else None
    if not raw:
        return {}
    try:
        m = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(m, dict):
        return {}
    return {k: v.strip() for k, v in m.items()
            if k in SUFFIX_FRAMINGS and isinstance(v, str) and v.strip()}


def dataset_prompt_suffix(ds, framing=None) -> str:
    """The dataset's EFFECTIVE creative-direction suffix for one shot (per-framing
    then global). Every wrap call site funnels through here so the suffix is
    applied exactly once, at generation time — the stored variation_prompt stays
    raw and regeneration can never double-apply it."""
    if not ds:
        return ''
    return compose_prompt_suffix(getattr(ds, 'prompt_suffix', None),
                                 getattr(ds, 'prompt_suffixes', None), framing)


def subject_type_of(ds) -> str:
    """The dataset's subject type, normalised — NULL/legacy -> 'human'. The single
    reader every wrap call site uses so a legacy dataset (column NULL) generates
    exactly as before."""
    return normalize_subject_type(getattr(ds, 'subject_type', None) if ds else None)


# InsightFace/antelopev2 is a detector+embedder trained on PHOTOGRAPHED faces. On a
# drawn character it detects nothing most of the time, and the rare "detection" is a
# meaningless cosine — the pass used to fail OPEN: grey tiles or a plausible number,
# with nothing saying the tool simply cannot read this kind of image.
# The message states the way out on purpose: there is NO extra setting to force the
# pass. A knob whose only correct value is "off" is a knob nobody can set right; the
# subject type IS the switch, it is one click away, and it says what it means. A
# genuinely photographic dataset mislabelled anime is fixed where the mistake is.
FACE_SCORING_DRAWN_REASON = (
    'Face similarity needs a photographic face; it cannot read a drawn one. '
    'Set the subject type to Human if this dataset is photographic.')


def face_scoring_block_reason(ds):
    """Why InsightFace scoring must NOT run on this dataset, or None to go ahead.

    The SINGLE place the rule lives: the dataset pass, the Studio cell scoring and
    best-epoch selection all consult this one function, and the dataset payload
    republishes its result so the UI never re-derives the rule either. A gate
    posted at four sites would drift; this one cannot.

    Scoped to face SIMILARITY. Head-cropping (`face_crop_to_square_webp` ->
    `detect_head_bbox`) goes through Qwen3-VL, a general vision model that reads a
    drawn head perfectly well — it is deliberately NOT gated here."""
    if subject_type_of(ds) == 'anime':
        return FACE_SCORING_DRAWN_REASON
    return None


def create_dataset(user_id, name, trigger_word, kind=None, concept_desc=None, train_type=None,
                   fidelity=None, prompt_suffix=None, prompt_suffixes=None, subject_type=None,
                   *, commit=True):
    """Create a dataset and return its row.

    ``commit=False`` is reserved for callers that need to coordinate the row with
    another resource (for example a restored filesystem tree).  The row is still
    flushed so its id is available, but ownership of commit/rollback stays with
    the caller.  Ordinary callers keep the historical commit-on-return contract.
    """
    k = normalize_kind(kind)
    desc = (concept_desc or '').strip()
    if k == 'concept' and not desc:
        # The concept description is what the captioner OMITS; without it the
        # inverted-caption logic has nothing to bind the trigger to. Required.
        raise ValueError('concept_desc required for a concept dataset')
    ds = FaceDataset(user_id=str(user_id), name=(name or '').strip()[:100],
                     trigger_word=(trigger_word or '').strip()[:60] or 'zchar',
                     # concept_desc n'a de sens que pour un concept ; un STYLE n'a rien
                     # à omettre nommément (les captions décrivent le contenu, jamais le
                     # rendu — c'est le prompt de caption qui porte cette règle).
                     kind=k, concept_desc=(desc[:500] if k == 'concept' else None),
                     # subject_type steers the generation catalog + identity lock;
                     # None left as NULL (== 'human') so a plain create is unchanged.
                     subject_type=(normalize_subject_type(subject_type)
                                   if subject_type is not None else None),
                     train_type=normalize_train_type(train_type),
                     # fidelity ne concerne que les personnages (concept : l'acte est
                     # omis ; style : les sujets varient, aucune identité à protéger).
                     fidelity=(normalize_fidelity(fidelity) if k is None else None),
                     # Direction créative optionnelle (globale + par cadrage) appliquée
                     # au wrap de chaque variation générée — cf. dataset_prompt_suffix.
                     prompt_suffix=(_normalize_prompt_suffix(prompt_suffix)
                                    if prompt_suffix is not None else None),
                     prompt_suffixes=(_normalize_prompt_suffixes(prompt_suffixes)
                                      if prompt_suffixes is not None else None))
    db.session.add(ds)
    db.session.flush()
    if k == 'style' and not (trigger_word or '').strip():
        # Le token d'un style est un identifiant INTERNE, jamais un mot d'activation :
        # `_run_name`/`lora_{trigger}` nomment le run d'entraînement avec. Deux styles
        # créés sans trigger retomberaient tous deux sur 'zchar' → le garde anti-
        # collision bloquerait le 2e entraînement. On sale le défaut avec l'id.
        ds.trigger_word = f'zsty_{ds.id}'
    if commit:
        db.session.commit()
    return ds


def family_base_memory(ds) -> dict:
    """Parsed `train_family_bases` — {family: {'base': str, 'variant': str|None}}.

    Anything unparsable/foreign reads as {} (same discipline as _train_settings):
    a corrupted blob must degrade to "nothing remembered", never to a crash."""
    raw = getattr(ds, 'train_family_bases', None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for fam, entry in data.items():
        if fam in TRAIN_TYPES and isinstance(entry, dict):
            out[fam] = {'base': entry.get('base') or '',
                        'variant': entry.get('variant') or None}
    return out


def remembered_family_base(ds, family):
    """(base, variant) this dataset last used on `family`, or (None, None) when
    that family has never been configured here. `None` is deliberately distinct
    from `''` (= "officially chose the official base")."""
    entry = family_base_memory(ds).get(normalize_train_type(family))
    if entry is None:
        return None, None
    return entry['base'], entry['variant']


def family_settings_memory(ds) -> dict:
    """Parsed `train_family_settings` — {family: {setting: value}}, restricted to
    the family-scoped keys. Same degrade-to-{} discipline as family_base_memory:
    a corrupted blob means "nothing remembered", never a crash."""
    from . import lora_training as _lt
    raw = getattr(ds, 'train_family_settings', None)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for fam, entry in data.items():
        if fam in TRAIN_TYPES and isinstance(entry, dict):
            out[fam] = {k: v for k, v in entry.items()
                        if k in _lt._FAMILY_SCOPED_SETTING_KEYS}
    return out


def remembered_family_settings(ds, family):
    """The family-scoped settings this dataset last used on `family`, or None
    when that family was never configured here. `{}` (configured, everything on
    Auto) is deliberately distinct from None (never configured)."""
    return family_settings_memory(ds).get(normalize_train_type(family))


def set_train_type(user_id, dataset_id, train_type, *, commit=True,
                   target_training_mode=None) -> bool:
    """Change the target model family later (kept in sync with the TrainingPanel
    selector so the menu re-groups). Normalizes; unknown -> zimage. False if absent.

    The base and the variant are FAMILY-SCOPED even though `train_base_model` /
    `train_variant` are single columns: a Z-Image merge is not a thing a Krea run
    can load, and 'turbo' means a different checkpoint on each family. So the
    outgoing family's pair is stashed in `train_family_bases` and the incoming
    family's remembered pair takes its place — a family never yet configured
    starts from the official base, and coming back to Z-Image finds the merge
    exactly where it was left. Nothing is destroyed and nothing is asked.

    The SAME treatment is given to the handful of `train_settings` keys whose
    meaning is bound to the family (lora_training._FAMILY_SCOPED_SETTING_KEYS —
    `timestep_type`, whose canonical value differs per family): stashed in
    `train_family_settings`, restored on the way back, and CLEARED (back to the
    incoming family's own default) when that family has nothing remembered. The
    other advanced settings stay global on purpose — see the comment on
    _FAMILY_SCOPED_SETTING_KEYS for why quantisation and resolution are not
    here. ``commit=False`` lets a caller join this family transition to a wider
    validated settings transaction without an intermediate database state.
    ``target_training_mode`` is reserved for that wider transaction: the legacy
    family-only endpoint must validate against the currently persisted mode."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    new_fam = normalize_train_type(train_type)
    old_fam = normalize_train_type(getattr(ds, 'train_type', None))
    from . import lora_training as _lt
    intended_mode = (_lt.training_mode(ds) if target_training_mode is None
                     else _lt.normalize_training_mode(target_training_mode))
    if intended_mode == 'full_transformer' and new_fam != 'krea':
        raise ValueError(
            'full_transformer training requires the Krea 2 model family. '
            'Switch the training mode to LoRA in Training settings before '
            'changing the model family.')
    if new_fam == old_fam:
        if commit:
            db.session.commit()
        return True
    memory = family_base_memory(ds)
    # Never remember a base the OUTGOING family provably cannot load. Datasets
    # created before this column exist in exactly that state (a Z-Image merge
    # left attached to a Krea 2 dataset); stashing it under 'krea' would freeze
    # the bug into the memory and hand it back on the way home.
    outgoing = ds.train_base_model or ''
    if _lt.foreign_base_reason(old_fam, outgoing):
        outgoing = ''
    memory[old_fam] = {'base': outgoing,
                       'variant': ds.train_variant or None}
    remembered = memory.get(new_fam)
    ds.train_base_model = (remembered or {}).get('base') or None
    ds.train_variant = (remembered or {}).get('variant') or None
    ds.train_family_bases = json.dumps(memory)

    # --- family-scoped train_settings keys, same stash/restore contract --------
    scoped = _lt._FAMILY_SCOPED_SETTING_KEYS
    settings = _lt._train_settings(ds)
    # An applied preset is itself family-scoped. Invalidate that replacement
    # before stashing/restoring individual family values, otherwise its hidden
    # topology/optimizer/step fields could survive under the new family's UI.
    _lt.clear_active_preset_settings(settings)
    smemory = family_settings_memory(ds)
    smemory[old_fam] = {k: settings[k] for k in scoped if k in settings}
    incoming = smemory.get(new_fam)
    for k in scoped:
        if incoming is not None and k in incoming:
            settings[k] = incoming[k]
        else:
            # Never configured on the incoming family (or explicitly left on
            # Auto there) → drop the key so the family's own canonical default
            # applies. Dropping is what makes it byte-identical to a dataset
            # that never touched the setting, exactly like update_train_settings.
            settings.pop(k, None)
    ds.train_settings = json.dumps(settings) if settings else None
    ds.train_family_settings = json.dumps(smemory)

    ds.train_type = new_fam
    if commit:
        db.session.commit()
    return True


def _guard_kind_switch(dataset_id):
    """Raise RuntimeError (-> 409) when live work on the dataset still assumes the
    CURRENT kind: an active training run, a server-side batch (caption / re-caption
    / watermark / face / classify) or an in-flight generation. Switching the kind
    mid-flight would mix caption strategies, or land generated variations into a set
    that no longer generates. ``dataset_activity`` covers the batch AND generation
    cases (the Klein/API fan-out is tracked as a 'generate' activity)."""
    _guard_no_active_training(dataset_id)
    if dataset_activity.get(dataset_id) is not None:
        raise RuntimeError(
            'This dataset has work in progress (generation, captioning or a quality '
            'pass). Wait for it to finish before changing the kind.')


def update_dataset_settings(user_id, dataset_id, *, name=None, trigger_word=None,
                            concept_desc=None, kind=None, prompt_suffix=None,
                            prompt_suffixes=None, subject_type=None):
    """Edit a dataset's identity AFTER creation. Returns {'ok', 'concept_desc_changed'}
    (plus {'kind_changed', 'kind', 'previous_kind'} when the kind actually changed),
    or None if the dataset is absent; raises ValueError on invalid input and
    RuntimeError (-> 409) when a kind switch is asked while work is in progress.

    Changing the **trigger word** needs NO re-caption: captions are stored without it
    (it's prepended at export). It is, however, the ON-DISK naming key, so everything
    the dataset already produced is renamed to follow — see _propagate_trigger_rename,
    reported back as `trigger_rename`. Refused (409) while a run is live, because the
    run folder is what ai-toolkit auto-resumes from. Changing a concept dataset's **description**
    (what the captions must omit) invalidates the cached LLM avoid-list (concept_terms)
    so it regenerates — but images already captioned keep the OLD omission until
    re-captioned (same 'future captions' contract as set_fidelity).

    Changing the **kind** (character / concept / style) is the disruptive one: it flips
    the caption strategy and which workspace panels show. It is honest, not magic —
    NOTHING is deleted (images, captions, scores, watermark work and training history
    stay), but existing captions keep the OLD strategy until re-captioned (the route's
    caller nudges it). Invariants mirror create_dataset: fidelity is character-only
    (cleared for concept/style); the concept avoid-list cache is dropped so it rebuilds
    for the new kind; a concept target requires an omit-description (passed here or
    already stored); a style keeps its stored trigger token but never uses it as an
    activation word. Past run identifiers are unaffected — a run is named by the model
    family + trigger, never the kind (see lora_training._run_name).

    **prompt_suffix** (global text) / **prompt_suffixes** (map {face,bust,body,back}):
    None = untouched; '' / {} = cleared. Applied at generation time only, so editing
    them changes FUTURE generations/regenerations — existing images are untouched."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return None
    # The on-disk naming key, measured ONCE before any mutation and once after them
    # all. Two different edits can move it (the trigger word, or a style's name — see
    # below), and a dataset can be edited by both in a single save, so comparing the
    # start and end states is the only reading that can't disagree with itself.
    _lt = _lora_training()
    naming_before = _lt._safe_trigger(ds) if _lt else None
    prev_label = (ds.kind or '').lower() or 'character'
    kind_changed = False
    if kind is not None:
        new_kind = normalize_kind(kind)          # None | 'concept' | 'style'
        new_label = new_kind or 'character'
        if new_label != prev_label:
            _guard_kind_switch(dataset_id)
            if new_label == 'concept':
                # A concept needs the omit-description: take the one passed in this
                # same save, else any value already stored (a switch back to concept).
                desc_src = concept_desc if concept_desc is not None else ds.concept_desc
                if not (desc_src or '').strip():
                    raise ValueError('concept_desc required for a concept dataset')
            ds.kind = new_kind
            if new_label != 'character':
                # Fidelity is a character-only target (mirrors create_dataset). The
                # value is remembered by nothing else, so a switch back defaults to face.
                ds.fidelity = None
            # The cached concept avoid-list is concept-specific; drop it so the
            # detector/captioner rebuild it for the new kind. concept_desc itself is
            # left in place (harmless for other kinds, restored on a switch back).
            ds.concept_terms = None
            kind_changed = True
    if name is not None:
        n = (name or '').strip()
        if n:
            new_name = n[:100]
            # A STYLE has no visible trigger — it is always-on, so the field is hidden
            # and the token that names its files is retained internally, out of reach.
            # Its NAME is therefore the only identity it can edit, so for a style (and
            # only a style) the name drives the naming token too; without this, a style
            # dataset could never rename the LoRAs it had already produced. The token is
            # pure file naming for a style (never an activation word), so moving it
            # changes nothing about captions or generation.
            if is_style(ds) and new_name != (ds.name or '') and _lt:
                token = _lt._safe_trigger(SimpleNamespace(
                    trigger_word=new_name, id=ds.id))[:60]
                if token != (ds.trigger_word or ''):
                    _guard_no_active_training(dataset_id, action='renaming a style dataset')
                    ds.trigger_word = token
            ds.name = new_name
    trigger_rename = None        # (old_safe, new_safe) when the on-disk naming key moved
    # A STYLE has no trigger FIELD — the settings modal sends back the stored token
    # verbatim (`trigger_word: style ? d.trigger_word : ...`). Honouring that echo
    # here overwrote the token the name block had just derived from the new name, so
    # renaming a style changed its label and nothing else: the reported bug. For a
    # style the name is the only lever, so an incoming trigger is never an edit.
    if is_style(ds):
        trigger_word = None
    if trigger_word is not None:
        t = (trigger_word or '').strip()
        if t:
            if t[:60] != (ds.trigger_word or ''):
                # The trigger is the ON-DISK naming key (u{user}_{trigger} run folders,
                # lora_{trigger} deployed files), so changing it renames everything this
                # dataset already produced. Refuse mid-flight: the run folder IS what
                # ai-toolkit auto-resumes from, and moving it under a live job would
                # strand the run. The rename itself is decided from naming_before /
                # naming_after around the whole edit, not here.
                _guard_no_active_training(dataset_id, action='changing the trigger word')
            ds.trigger_word = t[:60]
        elif not is_style(ds):
            # A character/concept trigger is the summon token — it cannot be blank.
            # A style has no activation trigger, so an empty value just keeps the
            # retained internal token as-is.
            raise ValueError('trigger_word cannot be empty')
    concept_changed = False
    if concept_desc is not None and is_concept(ds):
        d = (concept_desc or '').strip()
        if not d:
            raise ValueError('concept_desc required for a concept dataset')
        if d[:500] != (ds.concept_desc or ''):
            ds.concept_desc = d[:500]
            ds.concept_terms = None   # invalidate the cached LLM avoid-list → regenerated next caption
            concept_changed = True
    if prompt_suffix is not None:
        ds.prompt_suffix = _normalize_prompt_suffix(prompt_suffix)
    if prompt_suffixes is not None:
        ds.prompt_suffixes = _normalize_prompt_suffixes(prompt_suffixes)
    # subject_type: None = untouched. Only steers FUTURE wraps (existing images keep
    # their stored variation_prompt), so no in-flight guard is needed.
    if subject_type is not None:
        ds.subject_type = normalize_subject_type(subject_type)
    naming_after = _lt._safe_trigger(ds) if _lt else None
    if naming_before and naming_after and naming_before != naming_after:
        trigger_rename = (naming_before, naming_after)
    db.session.commit()
    res = {'ok': True, 'concept_desc_changed': concept_changed}
    if kind_changed:
        res.update(kind_changed=True, kind=(ds.kind or 'character'),
                   previous_kind=prev_label)
    if trigger_rename:
        moved = _propagate_trigger_rename(ds, *trigger_rename)
        # Only reported when it actually did something: a dataset that never trained
        # has no artefacts to move, and a silent 0-file rename is indistinguishable
        # from no rename at all — so the response stays exactly as it was before.
        if moved['files'] or not moved['ok']:
            res['trigger_rename'] = moved
    return res


def _lora_training():
    """lora_training, or None in a phase-1 install where it isn't present yet.
    Lazy: face_dataset_service <-> lora_training is a circular import at module level."""
    try:
        from . import lora_training as lt
        return lt
    except ImportError:
        return None


def _propagate_trigger_rename(ds, old_safe, new_safe) -> dict:
    """Carry a trigger rename through to disk AND to the rows that point at the
    renamed files. Returns {'ok', 'files', 'rows', 'conflicts'} for the caller to
    report; never raises — a failed rename leaves a working dataset whose old
    artefacts simply keep the old name (exactly today's behaviour).

    The database rewrite is DERIVED FROM THE FILES ACTUALLY RENAMED rather than
    rebuilt from the trigger: stored checkpoint values carry a ComfyUI subfolder
    ('z image\\...') and a family/step suffix, so reconstructing them here would
    duplicate — and eventually contradict — the naming rules in lora_training.
    Matching on basename keeps this correct whatever those rules become."""
    lt = _lora_training()
    if lt is None:
        return {'ok': False, 'files': 0, 'rows': 0, 'conflicts': []}
    out = lt.rename_training_artifacts(ds.user_id, old_safe, new_safe)
    if not out['ok']:
        # A destination already existed (a dataset already using the new trigger).
        # Nothing was moved, so nothing in the DB may be rewritten either.
        return {'ok': False, 'files': 0, 'rows': 0, 'conflicts': out['conflicts']}

    renames = out['renamed']
    by_basename = {os.path.basename(src): os.path.basename(dest)
                   for src, dest in renames if src.endswith('.safetensors')}
    dir_moves = [(src, dest) for src, dest in renames if not os.path.splitext(src)[1]]
    rows = 0

    def _remap(value):
        """The new name for a stored LoRA reference, or None when it isn't one of
        the files we just moved. Compares on basename so a stored subfolder prefix
        ('z image\\lora_X.safetensors') survives untouched."""
        if not value:
            return None
        base = os.path.basename(str(value).replace('\\', '/'))
        new_base = by_basename.get(base)
        return str(value)[:-len(base)] + new_base if new_base else None

    if by_basename:
        for row in LoraTestImage.query.filter_by(dataset_id=ds.id).all():
            new_ck = _remap(row.checkpoint)
            if new_ck:
                row.checkpoint = new_ck
                rows += 1
        # The dataset's winning Test-Studio settings pin a LoRA filename too.
        settings = _safe_json(ds.best_settings)
        if isinstance(settings, dict):
            new_ck = _remap(settings.get('lora_filename'))
            if new_ck:
                settings['lora_filename'] = new_ck
                ds.best_settings = json.dumps(settings)
                rows += 1

    # Cloud runs store the local run identity (u{user}_{trigger}{tag}) and cache
    # absolute paths under the renamed run folders — both carry the old trigger.
    from ..models import CloudTrainingRun
    old_run, new_run = f'u{ds.user_id}_{old_safe}', f'u{ds.user_id}_{new_safe}'
    for run in CloudTrainingRun.query.filter_by(dataset_id=ds.id).all():
        if run.run_name and lt._trigger_boundary(run.run_name, old_run):
            run.run_name = new_run + run.run_name[len(old_run):]
            rows += 1
        for attr in ('staging_dir', 'checkpoint_local_path'):
            cur = getattr(run, attr, None)
            for src, dest in dir_moves:
                if cur and os.path.normcase(str(cur)).startswith(os.path.normcase(src)):
                    setattr(run, attr, dest + str(cur)[len(src):])
                    rows += 1
                    break
    db.session.commit()
    return {'ok': True, 'files': len(renames), 'rows': rows, 'conflicts': []}


def get_dataset(user_id, dataset_id):
    ds = db.session.get(FaceDataset, dataset_id)
    return ds if ds and str(ds.user_id) == str(user_id) else None


def random_kept_caption(user_id, dataset_id) -> str | None:
    """Return one cleaned caption from an owned dataset's kept images.

    The candidate count and random offset stay in SQL so a large dataset never
    has all of its captions materialized just to choose one.  ``None`` means
    the dataset exists but has no non-blank kept caption; an inaccessible
    dataset raises ``LookupError`` so the route can return a 404 without
    leaking ownership details.
    """
    if not get_dataset(user_id, dataset_id):
        raise LookupError('dataset not found')

    from sqlalchemy import func

    cleaned = func.trim(FaceDatasetImage.caption, _PYTHON_STRIP_CHARS)
    eligible = (db.session.query(FaceDatasetImage.caption)
                .join(FaceDataset, FaceDatasetImage.dataset_id == FaceDataset.id)
                .filter(FaceDatasetImage.dataset_id == dataset_id,
                        FaceDataset.user_id == str(user_id),
                        FaceDatasetImage.status == 'keep',
                        FaceDatasetImage.caption.isnot(None),
                        cleaned != ''))
    count = eligible.count()
    if not count:
        return None

    caption = (eligible.order_by(FaceDatasetImage.id.asc())
               .offset(random.randrange(count)).limit(1).scalar())
    # Keep the API contract robust if an unusual Unicode whitespace-only value
    # slipped through SQLite's trim character set.
    return (caption or '').strip() or None


def list_datasets(user_id):
    """Every dataset of this user, newest touched first.

    The choke-point for "the datasets that exist": it feeds the library page,
    `full_backup.build_full_backup` (its `datasets_total`, its per-dataset zips
    AND its manifest), the name de-duplication of
    `full_backup.restore_full_backup`, the canvas dataset index, the canvas
    positions, the HF base-model index and `lora_training.find_run_collision`.
    Eight surfaces, one query — so a rule about which datasets count belongs
    HERE, never copied into a caller.
    """
    return (FaceDataset.query.filter_by(user_id=str(user_id))
            .order_by(FaceDataset.updated_at.desc()).all())


def dataset_list_stats(user_id):
    """Per-dataset aggregates for the library page — image counts and the
    families ever trained — in two grouped queries (never one per dataset).
    Returns {dataset_id: {'images_total', 'images_kept', 'images_captioned',
    'trained_families': [str]}}; datasets absent from a map just have zeros."""
    from sqlalchemy import case, func
    from ..models import TrainingRunRecord
    # Same scope as list_datasets: the counts shown next to the library must add
    # up to the library. One subquery, reused by BOTH grouped queries.
    owned = (db.session.query(FaceDataset.id)
             .filter_by(user_id=str(user_id))).subquery()
    stats = {}
    img_rows = (db.session.query(
        FaceDatasetImage.dataset_id,
        func.count(FaceDatasetImage.id),
        func.sum(case((FaceDatasetImage.status == 'keep', 1), else_=0)),
        func.sum(case(((FaceDatasetImage.status == 'keep')
                       & (func.coalesce(FaceDatasetImage.caption, '') != ''), 1), else_=0)))
        .filter(FaceDatasetImage.dataset_id.in_(db.session.query(owned.c.id)))
        .group_by(FaceDatasetImage.dataset_id).all())
    for ds_id, total, kept, captioned in img_rows:
        stats[ds_id] = {'images_total': int(total or 0), 'images_kept': int(kept or 0),
                        'images_captioned': int(captioned or 0), 'trained_families': []}
    fam_rows = (db.session.query(TrainingRunRecord.dataset_id, TrainingRunRecord.family)
                .filter(TrainingRunRecord.dataset_id.in_(db.session.query(owned.c.id)))
                .distinct().all())
    for ds_id, fam in fam_rows:
        entry = stats.setdefault(ds_id, {'images_total': 0, 'images_kept': 0,
                                         'images_captioned': 0, 'trained_families': []})
        if fam and fam not in entry['trained_families']:
            entry['trained_families'].append(fam)
    for entry in stats.values():
        entry['trained_families'].sort()
    return stats


def _clear_watermark_metadata(img):
    img.watermark_state = None
    img.watermark_bbox = None
    img.watermark_regions = None
    img.watermark_source = None
    img.watermark_score = None


def _unkeep_parent_for_kept_improvement(img):
    """Make a kept Klein improvement the dataset's active choice.

    An improve result is a separate image row, so this deliberately changes only
    the original row's review state: no files, captions or lineage are removed.
    The parent lookup is scoped to the candidate's dataset because legacy
    ``parent_image_id`` has no foreign key and may be stale or point elsewhere.
    A queued result can be marked Keep before its bytes arrive; it cannot replace
    the source until a regular file has actually landed in the dataset folder.
    """
    filename = img.filename
    if (img.derivation_kind != KLEIN_IMAGE_IMPROVE
            or not img.parent_image_id
            or not isinstance(filename, str)
            or not filename
            or '/' in filename
            or '\\' in filename
            or os.path.basename(filename) != filename
            or ntpath.basename(filename) != filename
            or posixpath.basename(filename) != filename
            or img.parent_image_id == img.id):
        return False
    try:
        candidate_path = _img_path(img)
    except (TypeError, ValueError):
        return False
    if not os.path.isfile(candidate_path):
        return False
    # Keep can race with an unkeep/reject click while completion is linking its
    # file.  Flush our local completion/status work, then let one SQL statement
    # consult the CURRENT candidate row and update only a still-kept parent.
    # Reading ``img.status`` here would use a stale SQLAlchemy object and could
    # evict the parent after the user already changed their mind.
    from sqlalchemy import exists, update
    from sqlalchemy.orm import aliased

    db.session.flush()
    candidate = aliased(FaceDatasetImage)
    candidate_is_kept = exists().where(
        candidate.id == img.id,
        candidate.dataset_id == img.dataset_id,
        candidate.parent_image_id == img.parent_image_id,
        candidate.derivation_kind == KLEIN_IMAGE_IMPROVE,
        candidate.status == 'keep',
        candidate.filename == filename,
    )
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(FaceDatasetImage.id == img.parent_image_id,
               FaceDatasetImage.dataset_id == img.dataset_id,
               FaceDatasetImage.status == 'keep',
               candidate_is_kept)
        .values(status='pending')
        .execution_options(synchronize_session=False))
    return bool(result.rowcount)


def _rekeep_pending_parent_for_reimprove(img):
    """CAS the source back to Keep while a currently kept result is re-run."""
    if (img.derivation_kind != KLEIN_IMAGE_IMPROVE
            or not img.parent_image_id
            or img.parent_image_id == img.id):
        return False
    from sqlalchemy import exists, update
    from sqlalchemy.orm import aliased

    candidate = aliased(FaceDatasetImage)
    candidate_is_kept = exists().where(
        candidate.id == img.id,
        candidate.dataset_id == img.dataset_id,
        candidate.parent_image_id == img.parent_image_id,
        candidate.derivation_kind == KLEIN_IMAGE_IMPROVE,
        candidate.status == 'keep',
    )
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(FaceDatasetImage.id == img.parent_image_id,
               FaceDatasetImage.dataset_id == img.dataset_id,
               FaceDatasetImage.status == 'pending',
               candidate_is_kept)
        .values(status='keep')
        .execution_options(synchronize_session=False))
    return bool(result.rowcount)


def _nullable_equals(column, value):
    return column.is_(None) if value is None else column == value


def _matches_reimprove_state(row, img, state):
    """SQL predicates for the snapshot that the re-run is allowed to replace."""
    return (
        row.id == img.id,
        row.dataset_id == img.dataset_id,
        row.parent_image_id == img.parent_image_id,
        row.derivation_kind == KLEIN_IMAGE_IMPROVE,
        row.status == state['status'],
        _nullable_equals(row.filename, state['filename']),
        _nullable_equals(row.job_id, state['job_id']),
    )


def _transition_reimprove_candidate(img, old_state, parent, label, prompt, job_id,
                                    expected_transition_caption,
                                    expected_transition_caption_origin=None):
    """CAS one improvement into its in-flight replacement state.

    The job has already been queued, but a status click can land while enqueue is
    in progress.  Do not overwrite that newer decision; the caller cancels the
    unlinked job when this snapshot no longer matches.
    """
    from sqlalchemy import case, update

    values = {
        'filename': None,
        'status': 'pending',
        'job_id': job_id,
        'variation_label': label,
        'variation_prompt': prompt[:500],
        'framing': parent.framing,
        'fail_reason': None,
        'fail_kind': None,
        'watermark_state': None,
        'watermark_bbox': None,
        'watermark_regions': None,
    }
    if not old_state['caption']:
        # A blank caption inherits the parent on a normal re-run, but an editor
        # can save text while enqueue_klein_edit is waiting.  Fill only if it is
        # STILL blank in the database; otherwise preserve that newer work.
        still_blank = ((FaceDatasetImage.caption.is_(None))
                       | (FaceDatasetImage.caption == ''))
        values['caption'] = case(
            (still_blank, expected_transition_caption), else_=FaceDatasetImage.caption)
        # The stamp moves in the SAME case(), on the same condition, inside the
        # same statement — a second UPDATE could land between the two and leave a
        # caption whose recorded author is another caption's.
        values['caption_origin'] = case(
            (still_blank, expected_transition_caption_origin),
            else_=FaceDatasetImage.caption_origin)
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(*_matches_reimprove_state(FaceDatasetImage, img, old_state))
        .values(**values)
        .execution_options(synchronize_session=False))
    if result.rowcount:
        db.session.expire(img)
    return bool(result.rowcount)


def _restore_reimprove_candidate_after_trash_failure(
        img, old_state, job_id, expected_transition_caption):
    """Restore only the exact transient state written by this re-run."""
    from sqlalchemy import case, update

    transient = dict(old_state, status='pending', filename=None, job_id=job_id)
    # Restore the exact old caption only while it is still what this transition
    # would have written.  A caption changed during Trash I/O wins instead.
    restore_values = {field: value for field, value in old_state.items()
                      if field not in ('caption', 'caption_origin')}
    still_ours = _nullable_equals(FaceDatasetImage.caption,
                                  expected_transition_caption)
    restore_values['caption'] = case(
        (still_ours, old_state['caption']), else_=FaceDatasetImage.caption)
    # Same condition, same statement: a Trash failure that restored the old
    # sentence but not its stamp would quietly demote a hand-written caption to
    # "origin never recorded", i.e. re-writable — a protection lost to an error
    # path nobody watches.
    restore_values['caption_origin'] = case(
        (still_ours, old_state.get('caption_origin')),
        else_=FaceDatasetImage.caption_origin)
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(*_matches_reimprove_state(FaceDatasetImage, img, transient))
        .values(**restore_values)
        .execution_options(synchronize_session=False))
    if result.rowcount:
        db.session.expire(img)
    return bool(result.rowcount)


def _undo_rekeep_parent_after_reimprove_trash_failure(img, old_state):
    """Undo only a fallback whose candidate was successfully restored."""
    if (img.derivation_kind != KLEIN_IMAGE_IMPROVE
            or not img.parent_image_id
            or img.parent_image_id == img.id):
        return False
    from sqlalchemy import exists, update
    from sqlalchemy.orm import aliased

    candidate = aliased(FaceDatasetImage)
    candidate_is_restored = exists().where(
        *_matches_reimprove_state(candidate, img, old_state))
    result = db.session.execute(
        update(FaceDatasetImage)
        .where(FaceDatasetImage.id == img.parent_image_id,
               FaceDatasetImage.dataset_id == img.dataset_id,
               FaceDatasetImage.status == 'keep',
               candidate_is_restored)
        .values(status='pending')
        .execution_options(synchronize_session=False))
    return bool(result.rowcount)


@_serialize_dataset_image_ingest
def set_image_status(user_id, image_id, status):
    if status not in _VALID_STATUS:
        raise ValueError('invalid status')
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return False
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds or str(ds.user_id) != str(user_id):
        return False
    if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
        raise ValueError('resolve small-image rescue pairs with the dedicated review action')
    if status == 'reject':
        _clear_watermark_metadata(img)
    img.status = status
    if status == 'keep':
        _unkeep_parent_for_kept_improvement(img)
    db.session.commit()
    return True


def _owned_image(user_id, image_id):
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return None
    ds = db.session.get(FaceDataset, img.dataset_id)
    return img if ds and str(ds.user_id) == str(user_id) else None


# --- rows that can vanish under a long pass ---------------------------------
# A pass loads its rows, then walks them for minutes to hours with one model
# call per image, committing as it goes. `expire_on_commit` is on, so every row
# it has not reached yet is a lazy re-SELECT — and the grid stays fully
# interactive while the pass runs, so deleting a bad tile mid-caption is
# ordinary use rather than a race anyone has to engineer.
#
# SQLAlchemy's default answer is fatal: an attribute read on an expired row
# whose database row is gone raises ObjectDeletedError (measured — including for
# the PRIMARY KEY, which is why ids captured BEFORE any commit are the immune
# shape), and a commit carrying a write staged on such a row raises too, leaving
# the session poisoned for the `finally`. Either killed the WHOLE pass and threw
# away the work already done on every other image.
#
# `analyze_faces` is the reference shape in this file and needs none of this: it
# carries plain tuples and writes through `update(...).where(...)` + rowcount.
# The passes below keep their ORM writes and get the same immunity by holding
# ids and re-reading through the helper immediately before each touch.
def _live_image_row(image_id):
    """The dataset row as the database has it RIGHT NOW, or None when it is gone.

    Always re-reads (``populate_existing``): a row the session still holds
    unexpired would otherwise come back from the identity map, and the whole
    question here is whether the image still exists. Measured on a 36 000-row
    bank: on an already-expired row this is ~31 us/image CHEAPER than the lazy
    attribute refresh it replaces, because that refresh emitted a SELECT anyway.
    """
    if image_id is None:
        return None
    return db.session.get(FaceDatasetImage, image_id, populate_existing=True)


def resolve_small_image_rescue(user_id, dataset_id, candidate_id, choice):
    """Resolve an original/Klein rescue pair in one DB commit.

    The pair is deliberately not mutable through the generic single/batch status
    paths: exactly one of these three decisions is the source of truth.
    Returns None when the owned dataset/candidate does not exist.
    """
    if choice not in ('original', 'klein', 'reject'):
        raise ValueError('choice must be original, klein, or reject')

    def _load_pair():
        ds = get_dataset(user_id, dataset_id)
        if not ds:
            return None, None
        candidate = (FaceDatasetImage.query
                     .filter_by(id=candidate_id, dataset_id=dataset_id).first())
        if not candidate:
            return None, None
        if candidate.derivation_kind != KLEIN_SMALL_IMAGE or not candidate.parent_image_id:
            raise ValueError('image is not a Klein small-image rescue candidate')
        source = (FaceDatasetImage.query
                  .filter_by(id=candidate.parent_image_id, dataset_id=dataset_id,
                             derivation_kind=SMALL_IMAGE_SOURCE).first())
        if not source:
            raise ValueError('small-image rescue source is missing or invalid')
        return source, candidate

    def _resolved_as(source, candidate):
        states = (source.status, candidate.status)
        return {('keep', 'reject'): 'original',
                ('reject', 'keep'): 'klein',
                ('reject', 'reject'): 'reject'}.get(states)

    def _payload(source, candidate):
        return {'choice': choice,
                'source': {'id': source.id, 'status': source.status},
                'candidate': {'id': candidate.id, 'status': candidate.status}}

    # Cancel before touching pair statuses: queue_manager uses the same scoped DB
    # session and commits its job row, so calling it after mutations would split
    # the supposedly atomic source/candidate decision.
    source, candidate = _load_pair()
    if source is None:
        return None
    already = _resolved_as(source, candidate)
    if already:
        result = _payload(source, candidate)
        db.session.rollback()
        if already != choice:
            raise RuntimeError(f'small-image rescue was already resolved as {already}')
        return result  # idempotent retry
    job_id = (candidate.job_id if choice != 'klein' and not candidate.filename else None)
    db.session.rollback()  # close the preflight read transaction before queue cancellation
    if job_id:
        try:
            from ..job_queue import queue_manager
            queue_manager.cancel_job(job_id, str(user_id), 'image')
        except Exception:
            logger.exception('small-image rescue: failed to cancel job %s', job_id)
    db.session.rollback()

    # SQLite's BEGIN IMMEDIATE serializes competing resolutions before either one
    # reads the transition state. The second caller therefore observes the first
    # committed choice and follows the idempotent/conflict branch.
    from sqlalchemy import text
    try:
        db.session.execute(text('BEGIN IMMEDIATE'))
        source, candidate = _load_pair()
        if source is None:
            db.session.rollback()
            return None
        already = _resolved_as(source, candidate)
        if already:
            if already != choice:
                raise RuntimeError(f'small-image rescue was already resolved as {already}')
            result = _payload(source, candidate)
            db.session.rollback()
            return result
        if source.status != 'pending' or candidate.status not in ('pending', 'failed'):
            raise RuntimeError('small-image rescue is not in a resolvable state')
        if choice == 'klein':
            if candidate.status == 'failed' or not candidate.filename:
                raise ValueError('Klein rescue result is not ready')
            source.status, candidate.status = 'reject', 'keep'
            _clear_watermark_metadata(source)
        elif choice == 'original':
            source.status, candidate.status = 'keep', 'reject'
            _clear_watermark_metadata(candidate)
        else:
            source.status = candidate.status = 'reject'
            _clear_watermark_metadata(source)
            _clear_watermark_metadata(candidate)
        db.session.commit()
        result = _payload(source, candidate)
    except Exception:
        db.session.rollback()
        raise
    _sync_generate_activity(dataset_id)
    return result


_UNSET = object()


@_serialize_dataset_image_ingest
def set_image_caption(user_id, image_id, caption, short=_UNSET):
    """Save one image's long caption; optionally its short variant. `short` defaults to a
    sentinel so a caller that only edits the long caption (the inline grid textarea) never
    wipes an existing short — only the expanded editor passes `short` to touch it.

    THIS IS THE APP'S ONLY CAPTION EDITOR, so it is where the 'asserted' stamp is
    born: what is saved here is a human's words, and a forced caption pass must
    skip it rather than overwrite it (services/caption_origin.py). Clearing the
    box clears the stamp with the text — protection follows the sentence, never a
    marker left behind on an empty field. The two captions are stamped
    independently: typing a long one does not claim authorship of a short one the
    dual-caption pass derived."""
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    caption_origin.stamp(img, _cap_caption(caption) or None, caption_origin.ASSERTED)
    if short is not _UNSET:
        caption_origin.stamp(img, _cap_caption(short) or None, caption_origin.ASSERTED,
                             field='caption_short')
    db.session.commit()
    return True


def _crop_resize_file(path, x, y, w, h, size=1024, dst=None):
    """Crop the file at `path` to (x,y,w,h) and normalise the crop's LONG side DOWN
    to at most `size`, PRESERVING the box's aspect ratio: a 2000x1500 box yields
    1024x768, a 2:3 box yields 683x1024 — no padding, no distortion (ai-toolkit
    buckets handle non-square training images). Writes to `dst` (default: overwrite
    `path`). Passing a distinct `dst` lets the reference crop read the untouched
    full-frame ORIGINAL and write the derived crop — so a re-crop can widen back
    out instead of only tightening the previous crop.

    A box SMALLER than `size` is left at its own size. The resize used to be
    unconditional, so a 240x180 crop was blown up to 1024x768 — and that upscale
    carried essentially nothing: shrinking the result back to 240 recovers the
    original at 48.96 dB (max channel error 10), for 2.3x the bytes. Since the
    encoder went lossless that is close to a megabyte of interpolated pixels per
    small crop, and it hands the trainer a tile whose apparent resolution is a
    fiction. Cropping in cannot create detail; it should not pretend to.

    Returns (ok, upscale_ratio), or (False, None) on failure. The ratio is
    unchanged in value and meaning — `size / long_side_of_box`, i.e. how far the
    box sits under the training resolution (>1 = under it) — because it is a
    STORED column (`FaceDatasetImage.upscale_ratio`) feeding the composition
    warning, and capping it along with the pixels would silently retire that
    warning. Only the pixels stopped pretending; the measurement did not move.

    ENCODING: the source format is preserved and written under
    `image_encoding.LOSSLESS`. This used to be an unconditional lossy WEBP q92, so
    cropping a PNG degraded it AND left PNG-named files holding WEBP bytes.

    Crop is the one operation for which lossless was a real trade rather than an
    obvious win — it RESAMPLES, so it destroys information whatever the encoder does,
    and lossless costs 4.59x the bytes. It was chosen on measurement, not principle:
    lossy WEBP has an error floor (chroma subsampled to 4:2:0 at every quality, so
    q100 still leaves max channel error 16 for 1.74x the size), and that error
    COMPOUNDS — five successive crops land at PSNR 45 dB whether they are q92 or
    q100, while lossless stays byte-identical to the first crop. See the measurement
    table in `image_encoding`'s module docstring.

    ⚠️ What this does NOT claim: only the ENCODING is lossless. A box longer than
    `size` is still resampled down, which destroys information whatever the encoder
    does. A box at or under `size` is now a pure cut, so it IS lossless end to end —
    as is the watermark crop (`_apply_watermark_crop`), which never resizes."""
    if not os.path.exists(path):
        return False, None
    with Image.open(path) as opened:
        # The DESTINATION name decides (it may differ from the source: the reference
        # editor reads the kept full frame and writes the derived crop), so the file
        # written always contains what its extension promises.
        fmt = image_encoding.format_for_path(dst or path, opened)
        opened.load()
        icc = _valid_icc_profile(opened.info.get('icc_profile'))
        # Browser crop coordinates describe the EXIF-oriented visual frame, not
        # the raw camera raster. Bake the orientation before interpreting x/y.
        oriented = ImageOps.exif_transpose(opened)
        # Narrow the mode BEFORE resampling: Pillow silently drops to nearest-neighbour
        # on paletted images, which would undo the point of removing the lossy encoder.
        src = oriented.convert(image_encoding.resample_mode(oriented))
    box = (max(0, int(x)), max(0, int(y)), min(src.width, int(x + w)), min(src.height, int(y + h)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return False, None
    bw, bh = box[2] - box[0], box[3] - box[1]
    # Normalise DOWN only: `long` is what we actually render, `size` stays the
    # reference the reported ratio is measured against (see the docstring).
    long = min(size, max(bw, bh))
    if bw >= bh:
        out_w, out_h = long, max(1, round(long * bh / bw))
    else:
        out_w, out_h = max(1, round(long * bw / bh)), long
    scale = size / max(bw, bh)
    out = io.BytesIO()
    image_encoding.save_edit(src.crop(box).resize((out_w, out_h), Image.LANCZOS),
                             out, fmt, image_encoding.LOSSLESS, icc_profile=icc)
    with open(dst or path, 'wb') as fh:
        fh.write(out.getvalue())
    return True, scale


@_serialize_dataset_image_ingest
def crop_image(user_id, image_id, x, y, w, h):
    """Crop a dataset image to (x,y,w,h), long side capped at 1024, no pad (a box
    smaller than that keeps its own size). Returns bool."""
    img = _owned_image(user_id, image_id)
    if not img or not img.filename:
        return False
    ok, scale = _crop_resize_file(_img_path(img), x, y, w, h)
    if ok:
        _clear_watermark_metadata(img)
        img.upscale_ratio = scale
        _invalidate_image_content_analysis(img)
        db.session.commit()
    return ok


def _valid_icc_profile(raw):
    """Return an ICC payload only when LittleCMS can parse it.

    Pillow will otherwise copy arbitrary bytes into the rewritten image, and some
    encoders fail late on malformed profiles.  ICC is the one embedded metadata
    item worth retaining here (colour rendering); EXIF orientation is deliberately
    baked into the pixels by ``ImageOps.exif_transpose`` and must not be reattached.
    """
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return None
    try:
        from PIL import ImageCms
        ImageCms.getOpenProfile(io.BytesIO(raw))
    except Exception:
        return None
    return bytes(raw)


def transformed_image_bytes(path, transform, *, max_source_bytes: int | None = None):
    """Apply ``transform`` (a PIL image -> PIL image callable) fully in memory,
    without touching ``path``, and return the re-encoded bytes. ``max_source_bytes``
    is an optional exact-read fence for a live external folder (Bank); ordinary
    Dataset edits retain their historical path-backed behaviour.

    THE shared encoder of every in-place pixel edit that REORDERS pixels without
    rebuilding any (mirror, rotation). Dataset rows may point at JPEG, PNG, WebP
    or BMP files (and restored legacy rows can still carry a misleading extension).
    Preserve the format Pillow actually detects and encode
    it under `image_encoding.LOSSLESS` — the policy this operation REQUIRES, passed
    explicitly so that tuning another operation's encoder can never silently
    degrade this one.

    ⚠️ Only JPEG loses anything here, and it loses it on EVERY edit — Pillow has
    no DCT-domain (jpegtran-style) path, so a 90° turn of a JPEG is a re-encode,
    not a lossless block transform. PNG, WebP and BMP preserve their decoded RGB
    pixels; BMP has no useful alpha path, so edits intentionally flatten it to RGB.

    The format is read from the CONTENT, not the file name: a mirror/rotation has
    no business converting a legacy extension mismatch it did not create. Crop,
    which rewrites the file wholesale and may write to a DIFFERENT destination,
    uses `image_encoding.format_for_path` instead.
    """
    source = path
    if max_source_bytes is not None:
        try:
            max_source_bytes = int(max_source_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid image source byte limit') from exc
        if max_source_bytes < 1:
            raise ValueError('invalid image source byte limit')
        try:
            with open(path, 'rb') as raw_source:
                raw = raw_source.read(max_source_bytes + 1)
        except (OSError, MemoryError) as exc:
            raise ValueError('invalid image file') from exc
        if len(raw) > max_source_bytes:
            raise ValueError(
                f'image source is too large (max {max_source_bytes // (1024 * 1024)} MiB)')
        # Decode the exact bounded bytes just read, not a path that a live Bank
        # folder could replace between a header check and the actual edit.
        source = io.BytesIO(raw)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(source) as src:
                image_encoding.validate_input_header_dimensions(src, label='image edit')
                fmt = (src.format or '').upper()
                if fmt not in image_encoding.EDITABLE_FORMATS:
                    raise ValueError(f'unsupported image format: {fmt or "unknown"}')
                if getattr(src, 'n_frames', 1) != 1:
                    raise ValueError('animated images are not supported')
                src.load()
                icc = _valid_icc_profile(src.info.get('icc_profile'))
                # EXIF orientation is baked into the pixels FIRST, so the edit the
                # user asked for is applied to the image they were shown — and the
                # tag is dropped (never reattached), so nothing rotates it twice.
                oriented = ImageOps.exif_transpose(src)
                edited = transform(oriented)

                out = io.BytesIO()
                edited, save_kwargs = image_encoding.save_params(
                    edited, fmt, image_encoding.LOSSLESS, icc_profile=icc)
                edited.save(out, fmt, **save_kwargs)
                payload = out.getvalue()
                # Read AFTER save_params: it may have converted the mode, and the
                # self-check below compares the decoded size against this.
                expected_size = edited.size
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, MemoryError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise ValueError('invalid image file') from e

    # Decode the exact encoded result before it is allowed near the live path.
    try:
        with Image.open(io.BytesIO(payload)) as check:
            check.load()
            if (check.format or '').upper() != fmt or check.size != expected_size:
                raise OSError('encoded edit validation failed')
    except (UnidentifiedImageError, OSError, SyntaxError, MemoryError) as e:
        raise ValueError('could not encode the edited image') from e
    return payload


def _mirrored_image_bytes(path):
    """Horizontal mirror — kept as a named wrapper for the mirror lane."""
    return transformed_image_bytes(path, ImageOps.mirror)


#: The only turns we offer, in degrees CLOCKWISE. Anything else is refused: a
#: free-angle rotation would need padding or cropping (it invents or drops
#: pixels), which is a different feature from "this photo is on its side".
ROTATION_DEGREES = (90, 180, 270)

#: Clockwise degrees -> Pillow transpose op. Pillow's ROTATE_* names are
#: COUNTER-clockwise, so 90 clockwise is ROTATE_270. These are exact pixel
#: permutations: no resampling, no interpolation, no pixel invented.
_ROTATE_OPS = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def normalize_rotation(degrees):
    """Fold any int to 0/90/180/270 clockwise, or raise ValueError.

    Accepts negatives (-90 == 270) and multiples of 360 so callers can pass a
    delta without doing the modulo themselves.
    """
    try:
        value = int(degrees)
    except (TypeError, ValueError):
        raise ValueError('rotation must be 90, 180 or 270 degrees') from None
    value %= 360
    if value % 90:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    return value


def rotate_transform(degrees):
    """The PIL transform for a normalised clockwise angle (0 => identity)."""
    op = _ROTATE_OPS.get(normalize_rotation(degrees))
    if op is None:
        return lambda image: image
    return lambda image: image.transpose(op)


def _rotated_image_bytes(path, degrees):
    """Rotate ``path`` clockwise by ``degrees`` in memory, format preserved."""
    if normalize_rotation(degrees) == 0:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    return transformed_image_bytes(path, rotate_transform(degrees))


@_serialize_dataset_image_ingest
def _edit_image_in_place(user_id, image_id, make_payload, *, tag):
    """Promote a re-encoded copy of one owned dataset image over its own file.

    ``make_payload(path) -> bytes`` prepares the new bytes; this owns everything
    that makes the swap safe — the per-image lock, the "did something else touch
    the file while we worked" check, the atomic replace and the watermark
    metadata rollback. Mirror and rotation share it verbatim so a fix to one is
    a fix to both.
    """
    lock = _IMAGE_PIXEL_EDIT_LOCKS[
        hash((str(user_id), image_id)) % len(_IMAGE_PIXEL_EDIT_LOCKS)]
    with lock:
        img = _owned_image(user_id, image_id)
        if not img:
            return None
        if not img.filename:
            raise ValueError('image file required')
        path = _img_path(img)
        if not os.path.isfile(path):
            raise RuntimeError('image file missing')

        try:
            before = os.stat(path)
            payload = make_payload(path)
        except ValueError:
            raise
        except OSError as e:
            raise RuntimeError('could not read image file') from e

        tmp_path = None
        try:
            try:
                fd, tmp_path = tempfile.mkstemp(
                    prefix=f'.{os.path.basename(path)}.{tag}-', suffix='.tmp',
                    dir=os.path.dirname(path),
                )
                with os.fdopen(fd, 'wb') as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                # Validate the on-disk temp as well as the in-memory encoding.
                with Image.open(tmp_path) as check:
                    check.verify()
            except (UnidentifiedImageError, OSError, SyntaxError) as e:
                raise RuntimeError(f'could not prepare the {tag} result') from e

            # Do not overwrite a crop/clean that raced this preparation outside
            # the edit lock.  (Mirror and rotation share the SAME stripe, so two
            # pixel edits of one image can never read the same source twice.)
            try:
                current = os.stat(path)
            except OSError as e:
                raise RuntimeError('image file missing') from e
            if (current.st_mtime_ns, current.st_size) != (before.st_mtime_ns, before.st_size):
                raise RuntimeError('image changed while editing; retry')

            watermark_snapshot = (
                img.watermark_state, img.watermark_bbox, img.watermark_regions,
                img.watermark_source, img.watermark_score)
            watermark_changed = any(value is not None for value in watermark_snapshot)
            if watermark_changed:
                _clear_watermark_metadata(img)
                try:
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                    raise

            try:
                # Same-directory replacement is atomic; the original remains live
                # until this single operation succeeds.
                os.replace(tmp_path, path)
                tmp_path = None
            except OSError as e:
                if watermark_changed:
                    (img.watermark_state, img.watermark_bbox,
                     img.watermark_regions, img.watermark_source,
                     img.watermark_score) = watermark_snapshot
                    try:
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                        logger.exception(
                            'failed to restore watermark metadata after %s promotion failure', tag)
                raise RuntimeError('could not update image file') from e


            _invalidate_image_content_analysis(img)
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                raise
            return {
                'image_id': img.id,
                # A request token is intentionally independent of filename and
                # HTTP Last-Modified granularity; the frontend appends it to ?v=.
                'cache_bust': time.time_ns(),
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning('could not remove %s temp file %s', tag, tmp_path)


def mirror_image(user_id, image_id):
    """Permanently mirror one owned dataset image horizontally.

    Returns ``None`` for an unknown/foreign row, otherwise a cache-bust payload.
    The filename and all semantic/provenance metadata remain stable.  Only
    watermark metadata is cleared because its pixel coordinates are no longer
    valid after a horizontal flip.
    """
    return _edit_image_in_place(
        user_id, image_id, _mirrored_image_bytes, tag='mirror')


def rotate_image(user_id, image_id, degrees):
    """Permanently rotate one owned dataset image by 90/180/270° CLOCKWISE.

    Same contract as :func:`mirror_image` — ``None`` for an unknown/foreign row,
    otherwise a cache-bust payload; the filename and every semantic/provenance
    field stay put, and only the watermark metadata is cleared (its normalised
    bbox is expressed in the OLD frame and a quarter turn invalidates it).

    A quarter turn is an exact pixel permutation, so nothing is resampled; what
    it costs is the re-encode of the container (see ``transformed_image_bytes``),
    which is pixel-exact for PNG/WEBP and lossy for JPEG.
    """
    turn = normalize_rotation(degrees)
    if turn == 0:
        raise ValueError('rotation must be 90, 180 or 270 degrees')
    return _edit_image_in_place(
        user_id, image_id, lambda path: _rotated_image_bytes(path, turn),
        tag='rotate')


@_serialize_dataset_image_ingest
def delete_image(user_id, image_id):
    """Delete a dataset image row and move its file to the app trash.

    If the image is still a pending generation, its queue job is cancelled
    first. Returns bool.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
        raise ValueError('resolve the small-image rescue pair before cleanup')
    original_path = (os.path.join(_dataset_path(img.dataset_id), img.filename)
                     if img.filename else None)
    snapshot = bank_transfer_metadata.parse_snapshot(img.bank_analysis_snapshot)
    cache_ref = snapshot.get('cache_ref') if snapshot else None
    cache_dir = _bank_analysis_cache_dir(img.dataset_id)
    cache_ref_shared = False
    if cache_ref:
        for other in (FaceDatasetImage.query
                      .filter(FaceDatasetImage.dataset_id == img.dataset_id,
                              FaceDatasetImage.id != img.id,
                              FaceDatasetImage.bank_analysis_snapshot.isnot(None))):
            other_snapshot = bank_transfer_metadata.parse_snapshot(
                other.bank_analysis_snapshot)
            if other_snapshot and other_snapshot.get('cache_ref') == cache_ref:
                cache_ref_shared = True
                break
    trashed_path = None
    try:
        if img.status == 'pending' and not img.filename and img.job_id:
            from ..job_queue import queue_manager
            if not queue_manager.cancel_job(
                    img.job_id, str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'This generation still has unconfirmed ComfyUI work; cancel it safely before deleting.')
        if original_path and os.path.exists(original_path):
            trashed_path = trash.send_to_trash(
                original_path, context=f'dataset-{img.dataset_id}-image-{img.id}')
        db.session.delete(img)
        db.session.commit()
    except trash.TrashLockError as e:
        db.session.rollback()
        _restore_from_trash(trashed_path, original_path)
        raise RuntimeError(_TRASH_LOCK_MESSAGE) from e
    except Exception:
        db.session.rollback()
        _restore_from_trash(trashed_path, original_path)
        raise
    # Row deletion is the sole point where its historical opaque vault ceases
    # to have an owner. Best-effort after commit avoids losing it on DB rollback.
    if cache_ref and not cache_ref_shared:
        bank_transfer_metadata.remove_cache_sidecar(cache_dir, cache_ref)
    return True


def _guard_no_active_training(dataset_id, *, action='deleting'):
    """Raise RuntimeError (-> 409) when a LOCAL or CLOUD training run is mid-flight
    on this dataset, so delete_dataset refuses instead of silently orphaning the
    run. Lazy imports dodge the cloud_training/lora_training <-> face_dataset_service
    import cycle; a module absent in a phase-1 install just means 'no such run'.

    TERMINAL runs (done/stopped/error/error_pod_kept) don't block: their provenance
    rows stay behind with an orphaned dataset_id (the existing no-FK pattern), which
    preserves run history and importable-checkpoint records after the dataset is gone."""
    try:
        from . import cloud_training as ct
    except ImportError:
        ct = None
    if ct is not None and ct.active_runs_for(dataset_id):
        raise RuntimeError(_ACTIVE_RUN_TEMPLATE.format(action=action))
    try:
        from . import lora_training as lt
    except ImportError:
        lt = None
    if lt is not None and lt.is_local_run_active(dataset_id):
        raise RuntimeError(_ACTIVE_RUN_TEMPLATE.format(action=action))


@_serialize_dataset_ingest
def delete_dataset(user_id, dataset_id):
    """Delete an owned dataset and move its complete folder to app trash.

    Refuses (RuntimeError -> 409) while a local or cloud training run is active on
    the dataset — deleting under a running run orphans its record and abandons a
    paid vast pod. Child image and Studio rows are explicitly removed for legacy
    databases whose foreign key had neither enforcement nor ``ON DELETE CASCADE``;
    terminal training-run records are intentionally left behind (orphaned
    dataset_id) to keep run history. Cancels any in-flight generations first.
    Returns False if not owned.
    """
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    _guard_no_active_training(dataset_id)
    # Capture le trigger AVANT de supprimer la ligne : sert à purger les artefacts
    # d'entraînement orphelins (LoRA déployés dans ComfyUI, run/export ai-toolkit,
    # job config) qui survivaient à la suppression du dataset et restaient
    # sélectionnables en génération. Import paresseux = pas d'import circulaire ;
    # lora_training n'existe pas encore en phase 1 -> purge silencieusement sautée.
    lt = None
    purge_user, purge_trigger = ds.user_id, None
    try:
        from . import lora_training as lt
        purge_trigger = lt._safe_trigger(ds)
    except ImportError:
        pass
    imgs = FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all()
    studio_rows = LoraTestImage.query.filter_by(dataset_id=dataset_id).all()
    # ◉ LoRA Canvas card positions. The model declares a relationship() to
    # face_dataset so the unit of work orders the DELETEs, but a mapper-level
    # dependency only covers rows that are IN the session — so they are loaded
    # and deleted explicitly here like every other child, and flushed before the
    # parent below. A dataset must never fail to delete over a display
    # preference: that exact bug already answered HTTP 500 once in this project.
    canvas_rows = CanvasNodePosition.query.filter_by(dataset_id=dataset_id).all()
    # 🖼 Pinned-image nodes: same story, same trap. They reference
    # lora_test_image rows that are being deleted in this very transaction.
    canvas_imgs = CanvasImageNode.query.filter_by(dataset_id=dataset_id).all()
    dataset_path = _dataset_path(dataset_id)
    trashed_path = None
    try:
        # Keep Studio queue cancellation atomic with deleting its owning rows.
        # Exact job_id + owned dataset scope prevents cross-dataset cancellation.
        from ..job_queue import queue_manager
        for img in imgs:
            if img.status == 'pending' and not img.filename and img.job_id:
                if not queue_manager.cancel_job(
                        img.job_id, str(user_id), 'image', commit=False):
                    raise RuntimeError(
                        'A dataset generation still has unconfirmed ComfyUI work; cancel it safely first.')
        for cell in studio_rows:
            if (cell.job_id
                    and cell.status not in ('done', 'failed', 'cancelled')):
                if not queue_manager.cancel_job(
                        cell.job_id, str(user_id), 'image', commit=False):
                    raise RuntimeError(
                        'A Test Studio cell still has unconfirmed ComfyUI work; cancel it safely first.')
        if os.path.exists(dataset_path):
            trashed_path = trash.send_to_trash(
                dataset_path, context=f'dataset-{dataset_id}')
        for img in imgs:
            db.session.delete(img)
        # Explicit for old databases whose FK definition cannot be altered by
        # db.create_all(). New databases also have ON DELETE CASCADE as a guard.
        for cell in studio_rows:
            db.session.delete(cell)
        for pos in canvas_rows:
            db.session.delete(pos)
        for pin in canvas_imgs:
            db.session.delete(pin)
        # Force the child DELETEs to reach the DB BEFORE the parent's. The child
        # models declare only a table-level ForeignKey (no relationship()), so the
        # unit of work has no ordering dependency between them and would otherwise
        # emit `DELETE FROM face_dataset` first. On a legacy DB whose FK lacks
        # ON DELETE CASCADE that parent-first order raises IntegrityError (the
        # children still physically exist); on a cascade DB it works but leaves a
        # SAWarning. Flushing the children here makes the order deterministic on
        # every DB vintage — the belt no longer depends on the DB doing the cascade.
        db.session.flush()
        db.session.delete(ds)
        db.session.commit()
    except trash.TrashLockError as e:
        db.session.rollback()
        _restore_from_trash(trashed_path, dataset_path)
        raise RuntimeError(_TRASH_LOCK_MESSAGE) from e
    except Exception:
        db.session.rollback()
        _restore_from_trash(trashed_path, dataset_path)
        raise
    # Purge les artefacts d'entraînement (LoRA ComfyUI + ai-toolkit + config). Best
    # effort : un échec ici ne doit pas faire échouer la suppression du dataset.
    if lt is not None:
        try:
            removed = lt.purge_training_artifacts(purge_user, purge_trigger)
            if removed:
                logger.info('delete_dataset %s : %d artefact(s) LoRA purgé(s)', dataset_id, len(removed))
        except Exception as e:
            logger.warning('delete_dataset %s : purge artefacts LoRA échouée : %s', dataset_id, e)
    return True


def _finish_cancelled_generation_row(img):
    """Remove one safely terminal generation while preserving rescue originals."""
    if img.derivation_kind == KLEIN_SMALL_IMAGE:
        img.status = 'failed'
        img.fail_reason = 'Klein small-image rescue was cancelled.'
    else:
        db.session.delete(img)


def cancel_pending(user_id, dataset_id):
    """Cancel all in-flight (pending) generations of a dataset.

    A local queue row is removed only after ``cancel_job`` proves that its exact
    ComfyUI prompt is gone.  If that proof cannot be obtained yet, keep the image
    row and its ``job_id``: it is the only UI handle from which the user can press
    Stop again once ComfyUI answers.  Dropping that row used to leave the durable
    global recovery barrier orphaned, making every GPU action report ``GPU busy``
    with no recoverable card left.

    Returns explicit recovery counts. ``retry_pending`` means LDS can retry the
    exact known prompt; ``restart_required`` means ComfyUI must be restarted and
    that restart explicitly confirmed before LDS may clear an unknown submission.

    ⏹ Stop generation also stops the server-side ✨ improve BATCH: cancelling the
    rows alone used to be pointless, because whatever was feeding the queue simply
    queued the next wave. The flag is armed FIRST so the worker can't slip another
    image in between the arming and the row deletion."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return {'cancelled': 0, 'recovery_pending': 0,
                'retry_pending': 0, 'restart_required': 0,
                'recovery_error': 0}
    dataset_activity.request_cancel(dataset_id, dataset_activity.IMPROVE_KINDS)
    # Only in-flight generations (pending AND no result file yet) - leave
    # completed-but-uncurated images alone.
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).all())
    targeted_ids = [img.id for img in rows]
    retry_pending = 0
    restart_required = 0
    recovery_error = 0
    for img in rows:
        if img.job_id:  # Klein rows only - API rows never carry a job_id
            try:
                from ..job_queue import queue_manager
                outcome = queue_manager.cancel_job_outcome(
                    img.job_id, str(user_id), 'image')
            except Exception:
                logging.getLogger(__name__).exception(
                    'could not safely cancel generation job %s', img.job_id)
                outcome = 'retry'
            if outcome == 'restart_required':
                restart_required += 1
                continue
            if outcome == 'barrier_corrupt':
                recovery_error += 1
                continue
            if outcome == 'retry':
                retry_pending += 1
                continue
            # cancelled / terminal / missing are all safe: cancel_job_outcome
            # proved that this exact job owns no durable recovery barrier.
        _finish_cancelled_generation_row(img)
    db.session.commit()
    # Counted from the DATABASE, never from the loop. `cancel_job_outcome` can
    # roll back internally, and a rollback discards the `db.session.delete(img)`
    # staged by EARLIER iterations — while a counter incremented in the loop has
    # already counted them. Stop then reported cancellations that never happened
    # and the tiles were still there on refresh. Asking which rows actually went
    # cannot drift from what the user sees.
    still_there = set()
    for i0 in range(0, len(targeted_ids), 500):
        chunk = targeted_ids[i0:i0 + 500]
        still_there.update(
            row_id for (row_id,) in db.session.query(FaceDatasetImage.id)
            .filter(FaceDatasetImage.id.in_(chunk)).all())
    n = len(targeted_ids) - len(still_there)
    # Stop deleted the in-flight rows: clear the Klein 'generate' indicator now
    # (its completion callbacks won't fire for cancelled jobs). An API batch's own
    # begin/end entry is untouched — its worker unwinds and end()s on its own.
    _sync_generate_activity(dataset_id)
    return {
        'cancelled': n,
        'retry_pending': retry_pending,
        'restart_required': restart_required,
        'recovery_error': recovery_error,
        'recovery_pending': retry_pending + restart_required + recovery_error,
    }


def confirm_unknown_generation_restart(user_id, dataset_id, *,
                                       restart_confirmed=False) -> int:
    """Clear this dataset's unknown-submit barrier after a human-confirmed restart.

    The reachability check belongs to the route. This service owns the atomic
    identity decision: only pending cards whose exact ``job_id`` matches the
    durable unknown-submit barrier are finalized.
    """
    if not restart_confirmed:
        raise ValueError('Confirm that ComfyUI was restarted before recovery.')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    from ..job_queue import queue_manager
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None))
            .filter(FaceDatasetImage.job_id.isnot(None)).all())
    targeted_ids = [img.id for img in rows]
    for img in rows:
        if not queue_manager.confirm_unknown_comfyui_restart(
                img.job_id, str(user_id), restart_confirmed=True):
            continue
        _finish_cancelled_generation_row(img)
    db.session.commit()
    # Counted from the database, not from the loop — same reason as
    # cancel_pending: confirm_unknown_comfyui_restart can roll back internally,
    # which discards deletes staged by earlier iterations that a loop counter has
    # already counted.
    still_there = set()
    for i0 in range(0, len(targeted_ids), 500):
        chunk = targeted_ids[i0:i0 + 500]
        still_there.update(
            row_id for (row_id,) in db.session.query(FaceDatasetImage.id)
            .filter(FaceDatasetImage.id.in_(chunk)).all())
    n = len(targeted_ids) - len(still_there)
    _sync_generate_activity(dataset_id)
    return n


def purge_unused(user_id, dataset_id):
    """Permanently delete all REJECTED and FAILED images of a dataset (rows +
    files). Returns the number purged."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id)
            .filter(FaceDatasetImage.status.in_(('reject', 'failed')))
            .filter(FaceDatasetImage.derivation_kind.notin_(_SMALL_IMAGE_DERIVATIONS)
                    | FaceDatasetImage.derivation_kind.is_(None)).all())
    n = 0
    for img in rows:
        if delete_image(user_id, img.id):
            n += 1
    return n


# --- Sauvegarde / restauration complète d'un dataset ---------------------------
# ZIP portable (≠ export d'entraînement) : manifest + réglages + TOUTES les images
# avec statuts/captions/scores — pour archiver ou déplacer un dataset entre machines.
BACKUP_FORMAT = 'lds-dataset-backup'
BACKUP_VERSION = 2
_BACKUP_MAX_FILES = 1400
_BACKUP_MAX_ROWS = 600
_BACKUP_MAX_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB uncompressed (zip-bomb guard)
_BACKUP_MAX_METADATA_BYTES = 192 * 1024 * 1024
_BACKUP_MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
# The import raster budget permits at most 16 Mi pixels.  96 MiB leaves room for
# a valid 16 MP RGBA/BMP source plus container overhead, while making a single
# archive entry incapable of filling a disk or RAM by itself.
_BACKUP_MAX_IMAGE_BYTES = 96 * 1024 * 1024
_BACKUP_NAME_RE = re.compile(r'^[\w.-]+\.(webp|jpg|jpeg|png|bmp)$', re.IGNORECASE)
_BACKUP_WINDOWS_DEVICE_RE = re.compile(
    r'^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])$', re.IGNORECASE)
_BACKUP_ANALYSIS_CACHE_RE = re.compile(
    r'^analysis-cache/(?P<ref>(?:[0-9a-f]{32}|[0-9a-f]{64}))\.npz$')
_BACKUP_EXTENSION_CANONICAL = {
    '.jpg': '.jpg', '.jpeg': '.jpg', '.png': '.png', '.webp': '.webp', '.bmp': '.bmp',
}

# Champs snapshotés tels quels par ligne image. ``job_id`` reste exclu car il est
# lié à la machine source; ``klein_model`` est portable et doit survivre au trajet.
# ⚠️ 'caption_origin'/'caption_short_origin' are in this tuple ON PURPOSE and must
# stay: this is the ONLY place the backup knows about them (the column names exist
# nowhere else in the export/restore path), and dropping them would not lose a
# caption — it would lose the PROTECTION on every hand-written caption, silently,
# at the first backup round-trip, leaving a restored dataset that a forced pass
# happily overwrites.
_BACKUP_IMG_FIELDS = ('filename', 'source', 'framing', 'variation_label', 'status',
                      'caption', 'caption_short',
                      'caption_origin', 'caption_short_origin',
                      'variation_prompt', 'klein_model', 'face_score', 'face_state',
                      'upscale_ratio', 'watermark_state', 'watermark_bbox',
                      'watermark_source', 'watermark_score',
                      'watermark_regions', 'parent_image_id', 'derivation_kind',
                      'fail_reason', 'fail_kind', 'source_metadata',
                      'bank_analysis_snapshot', 'transfer_metadata')


def _backup_basename(value):
    """Return a portable image basename, or None for paths/invalid values."""
    if not isinstance(value, str) or not value:
        return None
    if '/' in value or '\\' in value or not _BACKUP_NAME_RE.fullmatch(value):
        return None
    # Windows treats these stems as devices even when an extension is present
    # (``NUL.jpg`` writes to NUL). Reject them on every platform so an archive
    # restored on Windows cannot commit a row whose blob was never created.
    stem = value.split('.', 1)[0].rstrip(' .')
    if _BACKUP_WINDOWS_DEVICE_RE.fullmatch(stem):
        return None
    return value


def _backup_file_generation(path):
    """Identity/generation of one regular backup source, without symlinks."""
    try:
        info = os.stat(path, follow_symlinks=False)
    except (OSError, TypeError, ValueError):
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _backup_handle_generation(stream):
    try:
        info = os.fstat(stream.fileno())
    except (OSError, AttributeError, ValueError):
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _write_backup_file_member(archive, path, archive_name, expected_generation):
    """Stream one already-identified regular file through the same open handle."""
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('dataset changed while the backup was created') from exc
    with os.fdopen(descriptor, 'rb') as source:
        if _backup_handle_generation(source) != expected_generation:
            raise ValueError('dataset changed while the backup was created')
        with archive.open(archive_name, 'w') as destination:
            remaining = expected_generation[2]
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError('dataset changed while the backup was created')
                destination.write(chunk)
                remaining -= len(chunk)
            # The generation's exact length is part of the archive contract.  A
            # concurrent append must not escape the writer's aggregate preflight.
            if source.read(1):
                raise ValueError('dataset changed while the backup was created')
        if _backup_handle_generation(source) != expected_generation:
            raise ValueError('dataset changed while the backup was created')


def _validate_backup_limits(member_sizes=(), *, row_count=None) -> None:
    """Apply the portable-backup budgets to reader and writer inventories."""
    members = tuple(member_sizes)
    if len(members) > _BACKUP_MAX_FILES + 2:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    total = 0
    for name, size in members:
        total += size
        if total > _BACKUP_MAX_BYTES:
            raise ValueError('backup too large (max 2 GB uncompressed)')
        if (name.startswith(('images/', 'ref/'))
                and size > _BACKUP_MAX_IMAGE_BYTES):
            basename = name.rsplit('/', 1)[-1]
            raise ValueError(
                f'backup image {basename} is too large '
                f'(max {_BACKUP_MAX_IMAGE_BYTES // (1024 * 1024)} MiB per image)')
    if row_count is not None and row_count > _BACKUP_MAX_ROWS:
        raise ValueError(
            f'too many image rows in backup (max {_BACKUP_MAX_ROWS})')


def _normalized_backup_image_meta(meta, *, version=BACKUP_VERSION):
    """Validate every active FaceDatasetImage value before restore staging."""
    if not isinstance(meta, dict):
        raise ValueError('invalid backup image metadata')
    out = dict(meta)

    def optional_text(field, limit, *, allowed=None):
        value = meta.get(field)
        if value is None:
            return None
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f'invalid backup image {field}')
        if allowed is not None and value not in allowed:
            raise ValueError(f'invalid backup image {field}')
        return value

    def optional_number(field, low=None, high=None):
        value = meta.get(field)
        if value is None:
            return None
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f'invalid backup image {field}')
        number = float(value)
        if ((low is not None and number < low)
                or (high is not None and number > high)):
            raise ValueError(f'invalid backup image {field}')
        return number

    filename = meta.get('filename')
    if filename is not None and not isinstance(filename, str):
        raise ValueError('invalid backup image filename')
    if (version >= 2 and filename is not None
            and _backup_basename(filename) != filename):
        raise ValueError('invalid backup image filename')
    source = optional_text('source', 12, allowed=('generated', 'import'))
    status = optional_text(
        'status', 10, allowed=('pending', 'keep', 'reject', 'failed'))
    # v1 and small hand-built integration archives predate ``source``; use the
    # model's historical defaults without weakening validation of supplied data.
    source = source or 'generated'
    status = status or 'pending'
    out['source'], out['status'] = source, status
    out['framing'] = optional_text(
        'framing', 12, allowed=('face', 'bust', 'body', 'back', 'unknown'))
    out['variation_label'] = optional_text('variation_label', 120)
    out['caption'] = optional_text('caption', CAPTION_MAX_CHARS)
    out['caption_short'] = optional_text('caption_short', CAPTION_MAX_CHARS)
    for field, caption_field in (
            ('caption_origin', 'caption'),
            ('caption_short_origin', 'caption_short')):
        origin = optional_text(field, 16, allowed=caption_origin.VALUES)
        out[field] = origin if (out.get(caption_field) or '').strip() else None
    out['variation_prompt'] = optional_text('variation_prompt', 500)
    out['klein_model'] = optional_text('klein_model', 255)
    out['face_score'] = optional_number('face_score', -1.0, 1.0)
    out['face_state'] = optional_text(
        'face_state', 16,
        allowed=('scorable', 'no_face', 'low_det', 'too_small',
                 'extreme_pose', 'unreadable', 'error'))
    out['fail_reason'] = optional_text('fail_reason', 32768)
    out['fail_kind'] = optional_text(
        'fail_kind', 16, allowed=('refused', 'empty', 'error'))
    out['upscale_ratio'] = optional_number('upscale_ratio', 0.0, 1_000_000.0)
    out['watermark_state'] = optional_text(
        'watermark_state', 16,
        allowed=('none', 'detected', 'dismissed', 'cleaned', 'failed', 'error'))
    out['watermark_source'] = optional_text(
        'watermark_source', 16, allowed=('detector', 'vision'))
    out['watermark_score'] = optional_number('watermark_score', 0.0, 1.0)

    def box_storage(field, *, many):
        raw = meta.get(field)
        if raw is None:
            return None
        if not isinstance(raw, str) or len(raw) > 32768:
            raise ValueError(f'invalid backup image {field}')
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, RecursionError, MemoryError):
            raise ValueError(f'invalid backup image {field}')
        boxes = parsed if many else [parsed]
        if not isinstance(boxes, list) or len(boxes) > WATERMARK_REGION_LIMIT:
            raise ValueError(f'invalid backup image {field}')
        normalized = []
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f'invalid backup image {field}')
            if any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in box):
                raise ValueError(f'invalid backup image {field}')
            x1, y1, x2, y2 = map(float, box)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f'invalid backup image {field}')
            normalized.append([x1, y1, x2, y2])
        # Validation is semantic, transport remains byte-for-byte at the TEXT
        # level (including harmless whitespace) for a faithful backup roundtrip.
        return raw

    out['watermark_bbox'] = box_storage('watermark_bbox', many=False)
    out['watermark_regions'] = box_storage('watermark_regions', many=True)

    parent_id = meta.get('parent_image_id')
    if (parent_id is not None and (isinstance(parent_id, bool)
                                   or not isinstance(parent_id, int)
                                   or parent_id <= 0)):
        raise ValueError('invalid backup parent image id')
    out['parent_image_id'] = parent_id
    return out


def _normalized_backup_manifest(manifest):
    """Typed/bounded portable Dataset settings from an untrusted archive."""
    if not isinstance(manifest, dict):
        raise ValueError('invalid backup manifest')
    out = dict(manifest)

    def text_field(field, limit, *, allowed=None, required=False):
        value = manifest.get(field)
        if value is None:
            if required:
                raise ValueError(f'invalid backup {field}')
            return None
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f'invalid backup {field}')
        if allowed is not None and value not in allowed:
            raise ValueError(f'invalid backup {field}')
        return value

    out['name'] = text_field('name', 100)
    out['trigger_word'] = text_field('trigger_word', 60)
    out['kind'] = text_field(
        'kind', 16, allowed=('character', 'concept', 'style'))
    out['fidelity'] = text_field('fidelity', 8, allowed=FIDELITIES)
    out['concept_desc'] = text_field('concept_desc', 500)
    out['train_type'] = text_field('train_type', 16, allowed=TRAIN_TYPES)
    out['training_mode'] = text_field(
        'training_mode', 32, allowed=('lora', 'full_transformer')) or 'lora'
    out['train_base_model'] = text_field('train_base_model', 4096)
    variant = text_field('train_variant', 20)
    if variant is not None and not re.fullmatch(r'[A-Za-z0-9_.-]+', variant):
        raise ValueError('invalid backup train_variant')
    out['train_variant'] = variant

    def parsed_json_field(field, expected_type, max_bytes=1024 * 1024):
        raw = manifest.get(field)
        if raw is None:
            return None
        if not isinstance(raw, str) or len(raw.encode('utf-8')) > max_bytes:
            raise ValueError(f'invalid backup {field}')
        try:
            value = json.loads(
                raw,
                parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
        except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
            raise ValueError(f'invalid backup {field}')
        if not isinstance(value, expected_type):
            raise ValueError(f'invalid backup {field}')
        stack = [(value, 0)]
        seen = 0
        while stack:
            current, depth = stack.pop()
            seen += 1
            if seen > 100_000 or depth > 32:
                raise ValueError(f'invalid backup {field}')
            if isinstance(current, float) and not math.isfinite(current):
                raise ValueError(f'invalid backup {field}')
            if isinstance(current, dict):
                stack.extend((item, depth + 1) for item in current.values())
            elif isinstance(current, list):
                stack.extend((item, depth + 1) for item in current)
        return raw

    out['train_settings'] = parsed_json_field('train_settings', dict)
    out['best_settings'] = parsed_json_field('best_settings', dict)
    out['concept_terms'] = parsed_json_field(
        'concept_terms', list, max_bytes=256 * 1024)
    if out['concept_terms'] is not None:
        terms = json.loads(out['concept_terms'])
        if (len(terms) > 2048
                or any(not isinstance(term, str) or len(term) > 500
                       for term in terms)):
            raise ValueError('invalid backup concept_terms')
    return out


def _read_validated_backup_image(z: zipfile.ZipFile, info: zipfile.ZipInfo,
                                 basename: str) -> bytes:
    """Return one bounded, fully-decoded static backup image.

    The outer archive's central-directory total protects the aggregate, but a
    single entry still needs its own raw cap before it is inflated.  Crucially,
    validation happens while the bytes are only in memory: a malformed image
    must never be copied into the restore staging directory and later promoted.
    """
    max_bytes = _BACKUP_MAX_IMAGE_BYTES
    if info.file_size > max_bytes:
        raise ValueError(
            f'backup image {basename} is too large '
            f'(max {max_bytes // (1024 * 1024)} MiB per image)')
    try:
        with z.open(info) as source:
            # Do not trust a crafted central-directory ``file_size`` alone.
            # Reading one extra byte limits actual decompression even if that
            # metadata is inconsistent with the entry payload.
            raw = source.read(max_bytes + 1)
    except (OSError, EOFError, RuntimeError, MemoryError, zipfile.BadZipFile,
            zlib.error, lzma.LZMAError) as exc:
        raise ValueError(f'backup image {basename} could not be read') from exc
    if len(raw) > max_bytes:
        raise ValueError(
            f'backup image {basename} is too large '
            f'(max {max_bytes // (1024 * 1024)} MiB per image)')
    try:
        content_ext = _preserved_import_extension(raw, label=f'backup image {basename}')
    except (ValueError, MemoryError) as exc:
        raise ValueError(f'backup image {basename} is invalid: {exc}') from exc
    named_ext = _BACKUP_EXTENSION_CANONICAL.get(os.path.splitext(basename)[1].lower())
    if named_ext != content_ext:
        raise ValueError(
            f'backup image {basename} extension does not match its decoded content')
    return raw


def _backup_extra_ref_names(raw, *, limit=MAX_EXTRA_REFS):
    """Parse the stored JSON list into unique portable basenames."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    seen = set()
    for value in raw:
        name = _backup_basename(value)
        key = name.casefold() if name else None
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if limit is not None and len(out) >= limit:
            break
    return out


def _portable_train_base_model(value):
    """Keep model ids/relative paths, never machine-local absolute paths."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    drive, _ = ntpath.splitdrive(value)
    if (not value or drive or ntpath.isabs(value) or posixpath.isabs(value)):
        return None
    return value


def _write_backup_zip_locked(user_id: int, dataset_id: int,
                             output: BinaryIO, *, activity_token=None) -> None:
    """Self-contained backup of one dataset: manifest.json (settings) +
    images.json (rows) + ref/ + images/ files. Ordinary rows without a file are
    skipped, but small-image rescue metadata rows are retained so their pair can
    never become orphaned after restore."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    dsdir = _dataset_dir(dataset_id)
    from sqlalchemy import or_
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .filter(or_(FaceDatasetImage.filename.isnot(None),
                        FaceDatasetImage.derivation_kind.in_(_SMALL_IMAGE_DERIVATIONS)))
            .all())
    # Fail before touching the destination stream: an archive emitted with more
    # rows than the importer accepts is neither portable nor useful.
    _validate_backup_limits(row_count=len(rows))
    primary_ref_names = []
    ref_name_keys = set()
    for raw_name in (ds.ref_filename, ds.ref_original_filename):
        name = _backup_basename(raw_name)
        key = name.casefold() if name else None
        if (not name or key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, name))):
            continue
        ref_name_keys.add(key)
        primary_ref_names.append(name)
    portable_extras = []
    for name in _backup_extra_ref_names(ds.ref_extra_filenames, limit=None):
        key = name.casefold()
        if (key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, name))):
            continue
        ref_name_keys.add(key)
        portable_extras.append(name)
        if len(portable_extras) >= MAX_EXTRA_REFS:
            break
    # Extra-ref ORIGINALS travel too (as plain ref/ files, not in the manifest list):
    # restore keeps basenames, so the naming convention still ties them to their
    # extra — a restored dataset can widen a crop back out just like the source one.
    extra_originals = []
    for name in portable_extras:
        orig = extra_ref_original_name(name)
        key = orig.casefold() if orig else None
        if (not orig or key in ref_name_keys
                or not os.path.isfile(os.path.join(dsdir, orig))):
            continue
        ref_name_keys.add(key)
        extra_originals.append(orig)
    ref_names = primary_ref_names + portable_extras + extra_originals
    image_file_names = {
        name.casefold(): name for img in rows
        if (name := _backup_basename(img.filename))
        and os.path.isfile(os.path.join(dsdir, name))
    }
    file_backed_rows = [img for img in rows if _backup_basename(img.filename)]
    if len(image_file_names) != len(file_backed_rows):
        raise ValueError('dataset has duplicate or missing image files')
    collisions = ref_name_keys.intersection(image_file_names)
    if collisions:
        collision = image_file_names[next(iter(collisions))]
        raise ValueError(f'ref/image filename collision in dataset: {collision}')

    manifest = {
        'format': BACKUP_FORMAT, 'version': BACKUP_VERSION,
        'name': ds.name, 'trigger_word': ds.trigger_word,
        'kind': ds.kind, 'fidelity': ds.fidelity,
        'concept_desc': ds.concept_desc, 'concept_terms': ds.concept_terms,
        'train_type': ds.train_type,
        # Optional in backup v1: old archives omit it and restore as LoRA.
        'training_mode': (ds.training_mode
                          if ds.training_mode in ('lora', 'full_transformer')
                          else 'lora'),
        'train_base_model': _portable_train_base_model(ds.train_base_model),
        'train_variant': ds.train_variant, 'train_settings': ds.train_settings,
        'best_settings': ds.best_settings,
        'ref_filename': (_backup_basename(ds.ref_filename)
                         if _backup_basename(ds.ref_filename) in primary_ref_names else None),
        'ref_original_filename': (
            _backup_basename(ds.ref_original_filename)
            if _backup_basename(ds.ref_original_filename) in primary_ref_names else None),
        'ref_extra_filenames': json.dumps(portable_extras),
    }
    manifest_generation = tuple(
        getattr(ds, field) for field in (
            'name', 'trigger_word', 'kind', 'fidelity', 'concept_desc',
            'concept_terms', 'train_type', 'training_mode', 'train_base_model',
            'train_variant', 'train_settings', 'best_settings', 'ref_filename',
            'ref_original_filename', 'ref_extra_filenames'))
    row_generations = {
        img.id: tuple(getattr(img, field) for field in _BACKUP_IMG_FIELDS)
        for img in rows
    }
    backup_files = [
        (os.path.join(dsdir, name), f'ref/{name}') for name in ref_names
    ] + [
        (os.path.join(dsdir, name), f'images/{name}')
        for name in image_file_names.values()
    ]
    file_generations = {}
    for path, _archive_name in backup_files:
        generation = _backup_file_generation(path)
        if generation is None:
            raise ValueError('dataset file disappeared before backup')
        file_generations[path] = generation
    # backup_image_id is archive-local only. It lets restore remap parent_image_id
    # to the newly allocated row ids instead of retaining ids from the source DB.
    images_meta = []
    analysis_cache_payloads = {}
    analysis_cache_dir = _bank_analysis_cache_dir(dataset_id)
    for row_index, img in enumerate(rows, 1):
        if row_index == 1 or row_index % 25 == 0:
            dataset_activity.progress(
                activity_token, detail='sealing backup metadata')
        row = dict({'backup_image_id': img.id},
                   **{f: getattr(img, f) for f in _BACKUP_IMG_FIELDS})
        # Archive a structured, revalidated object rather than the raw TEXT
        # column. A malformed legacy/local row can never export arbitrary links.
        row['source_metadata'] = normalize_source_metadata(img.source_metadata)
        # A snapshot is durable only when it has the expected version, fingerprint
        # and bounded analysis shape.  Invalid legacy/local text is deliberately
        # omitted rather than becoming an opaque payload in a portable backup.
        snapshot = bank_transfer_metadata.parse_snapshot(img.bank_analysis_snapshot)
        if snapshot is None and img.bank_analysis_snapshot:
            try:
                declared = (img.bank_analysis_snapshot
                            if isinstance(img.bank_analysis_snapshot, dict)
                            else json.loads(img.bank_analysis_snapshot))
            except (TypeError, ValueError, UnicodeError, RecursionError, MemoryError):
                declared = None
            if isinstance(declared, dict) and declared.get('cache_ref') is not None:
                raise ValueError('invalid Bank analysis cache snapshot')
        if snapshot and snapshot.get('cache_ref'):
            cache_ref = snapshot['cache_ref']
            if not bank_transfer_metadata.is_content_addressed_cache_ref(cache_ref):
                raise ValueError(
                    'Bank analysis cache is not content-addressed; re-promote '
                    'the image before creating a backup')
            if cache_ref not in analysis_cache_payloads:
                cache_path = os.path.join(
                    analysis_cache_dir, f'{cache_ref}.npz')
                try:
                    size = os.path.getsize(cache_path)
                    if size <= 0 or size > bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES:
                        raise ValueError('Bank analysis cache sidecar is too large')
                    with open(cache_path, 'rb') as source:
                        cache_payload = source.read(
                            bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES + 1)
                except OSError as exc:
                    raise ValueError(
                        'Bank analysis cache sidecar is missing or unreadable') from exc
                if (len(cache_payload) != size
                        or bank_transfer_metadata.read_cache_sidecar_bytes(
                            cache_payload, expected_ref=cache_ref) is None):
                    raise ValueError(
                        'Bank analysis cache sidecar is malformed or has changed')
                analysis_cache_payloads[cache_ref] = cache_payload
        row['bank_analysis_snapshot'] = (
            bank_transfer_metadata.normalized_snapshot_storage(snapshot)
            if snapshot is not None else None)
        transfer_metadata = (
            bank_transfer_metadata.normalized_transfer_metadata_storage(
                img.transfer_metadata))
        if img.transfer_metadata is not None and transfer_metadata is None:
            raise ValueError('invalid Bank/Dataset transfer metadata')
        row['transfer_metadata'] = transfer_metadata
        images_meta.append(row)
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=1).encode('utf-8')
    images_payload = json.dumps(
        images_meta, ensure_ascii=False, indent=1).encode('utf-8')
    if (len(manifest_payload) > _BACKUP_MAX_METADATA_BYTES
            or len(images_payload) > _BACKUP_MAX_METADATA_BYTES):
        raise ValueError('dataset backup metadata is too large')
    backup_member_sizes = [
        ('manifest.json', len(manifest_payload)),
        ('images.json', len(images_payload)),
        *((archive_name, file_generations[path][2])
          for path, archive_name in backup_files),
        *((f'analysis-cache/{cache_ref}.npz', len(payload))
          for cache_ref, payload in analysis_cache_payloads.items()),
    ]
    _validate_backup_limits(backup_member_sizes, row_count=len(images_meta))
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        total_members = 2 + len(backup_files) + len(analysis_cache_payloads)
        dataset_activity.progress(
            activity_token, done=0, total=total_members,
            detail='writing portable backup')
        z.writestr('manifest.json', manifest_payload)
        dataset_activity.bump(activity_token)
        z.writestr('images.json', images_payload)
        dataset_activity.bump(activity_token)
        for path, archive_name in backup_files:
            if _backup_file_generation(path) != file_generations[path]:
                raise ValueError('dataset changed while the backup was created')
            _write_backup_file_member(
                z, path, archive_name, file_generations[path])
            if _backup_file_generation(path) != file_generations[path]:
                raise ValueError('dataset changed while the backup was created')
            dataset_activity.bump(activity_token)
        for cache_ref, payload in sorted(analysis_cache_payloads.items()):
            # Archive exactly the immutable bytes validated above.  A live
            # sidecar replacement between validation and ZipFile.write cannot
            # smuggle different cache contents into the backup.
            z.writestr(f'analysis-cache/{cache_ref}.npz', payload)
            dataset_activity.bump(activity_token)
        # Final generation fence: activity/ingest reservations block normal app
        # writes, while this catches an already-running request or an external
        # same-folder change that crossed the reservation boundary.
        current_ds = (FaceDataset.query.filter_by(id=dataset_id, user_id=user_id)
                      .populate_existing().one_or_none())
        current_manifest_generation = (tuple(
            getattr(current_ds, field) for field in (
                'name', 'trigger_word', 'kind', 'fidelity', 'concept_desc',
                'concept_terms', 'train_type', 'training_mode',
                'train_base_model', 'train_variant', 'train_settings',
                'best_settings', 'ref_filename', 'ref_original_filename',
                'ref_extra_filenames')) if current_ds is not None else None)
        current_rows = (FaceDatasetImage.query
                        .filter(FaceDatasetImage.id.in_(tuple(row_generations)))
                        .populate_existing().all()) if row_generations else []
        current_row_generations = {
            img.id: tuple(getattr(img, field) for field in _BACKUP_IMG_FIELDS)
            for img in current_rows if img.dataset_id == dataset_id
        }
        current_row_ids = {
            row_id for row_id, in db.session.query(FaceDatasetImage.id)
            .filter_by(dataset_id=dataset_id)
            .filter(or_(FaceDatasetImage.filename.isnot(None),
                        FaceDatasetImage.derivation_kind.in_(
                            _SMALL_IMAGE_DERIVATIONS))).all()
        }
        if (current_manifest_generation != manifest_generation
                or current_row_generations != row_generations
                or current_row_ids != set(row_generations)
                or any(_backup_file_generation(path) != generation
                       for path, generation in file_generations.items())):
            raise ValueError('dataset changed while the backup was created')


def write_backup_zip(user_id: int, dataset_id: int, output: BinaryIO) -> None:
    """Create one coherent backup generation under Dataset-wide exclusion."""
    dataset_id = dataset_activity.normalize_dataset_id(dataset_id)
    with _dataset_ingest_lock(user_id, dataset_id):
        token = dataset_activity.begin_exclusive(
            dataset_id, 'backup', detail='creating portable backup')
        if token is None:
            raise ValueError(
                'dataset has work in progress; wait before creating a backup')
        try:
            return _write_backup_zip_locked(
                user_id, dataset_id, output, activity_token=token)
        finally:
            dataset_activity.end(token)


def build_backup_zip(user_id: int, dataset_id: int) -> bytes:
    """Compatibility wrapper for callers that still need an in-memory archive."""
    output = io.BytesIO()
    write_backup_zip(user_id, dataset_id, output)
    return output.getvalue()


def _coerce_archive_stream(archive):
    """Return (seekable stream, owned stream or None) without copying file uploads."""
    if isinstance(archive, (bytes, bytearray, memoryview)):
        owned = io.BytesIO(bytes(archive))
        return owned, owned
    if not hasattr(archive, 'read') or not hasattr(archive, 'seek'):
        raise ValueError('not a zip file')
    try:
        archive.seek(0)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('zip archive is not seekable') from exc
    return archive, None


def _preflight_zip_central_directory(
        stream, *, max_entries, max_central_bytes, label) -> None:
    """Bound EOCD-declared allocation before ``ZipFile`` parses the directory.

    ``_EndRecData`` is the bounded EOCD reader used by CPython 3.12 itself.  Keep
    a compatibility fallback for runtimes that do not expose it, and always
    rewind because ``ZipFile`` must receive the original stream from offset zero.
    """
    end_reader = getattr(zipfile, '_EndRecData', None)
    if not callable(end_reader):
        stream.seek(0)
        return
    try:
        try:
            end_record = end_reader(stream)
        except (OSError, EOFError, AttributeError, OverflowError, TypeError, ValueError,
                zipfile.BadZipFile) as exc:
            raise ValueError('not a zip file') from exc
    finally:
        try:
            stream.seek(0)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError('zip archive is not seekable') from exc
    if end_record is None:
        raise ValueError('not a zip file')
    try:
        entries = end_record[getattr(zipfile, '_ECD_ENTRIES_TOTAL', 4)]
        central_size = end_record[getattr(zipfile, '_ECD_SIZE', 5)]
    except (IndexError, TypeError) as exc:
        raise ValueError('not a zip file') from exc
    if (isinstance(entries, bool) or not isinstance(entries, int)
            or entries < 0
            or isinstance(central_size, bool) or not isinstance(central_size, int)
            or central_size < 0):
        raise ValueError('not a zip file')
    if entries > max_entries:
        raise ValueError(f'too many files in {label} (max {max_entries})')
    if central_size > max_central_bytes:
        raise ValueError(f'{label} central directory is too large')


def _preflight_backup_central_directory(stream) -> None:
    return _preflight_zip_central_directory(
        stream, max_entries=_BACKUP_MAX_FILES + 2,
        max_central_bytes=_BACKUP_MAX_CENTRAL_DIRECTORY_BYTES,
        label='backup')


def import_backup_zip(user_id: int, archive: bytes | BinaryIO):
    """Restore a backup as a NEW dataset (never merges into an existing one).
    Hardened: manifest format/version check, per-entry filename whitelist (no
    separators/traversal), file-count and uncompressed-size caps. Returns the
    created FaceDataset."""
    stream, owned = _coerce_archive_stream(archive)
    try:
        _preflight_backup_central_directory(stream)
        try:
            z = zipfile.ZipFile(stream)
        except zipfile.BadZipFile as exc:
            raise ValueError('not a zip file') from exc
        try:
            return _import_backup_zipfile(user_id, z)
        finally:
            z.close()
    finally:
        if owned is not None:
            owned.close()


def _import_backup_zipfile(user_id: int, z: zipfile.ZipFile):
    # Validate the central directory BEFORE inflating JSON.  Previously a tiny
    # compressed manifest/images.json could bypass the image-only size total and
    # consume unbounded RAM during z.read/json.loads.
    all_infos = z.infolist()
    _validate_backup_limits(
        (info.filename, info.file_size) for info in all_infos)
    metadata = {}
    for info in all_infos:
        if info.filename not in ('manifest.json', 'images.json'):
            continue
        if info.filename in metadata:
            raise ValueError(f'duplicate {info.filename} in backup')
        if info.file_size > _BACKUP_MAX_METADATA_BYTES:
            raise ValueError(f'{info.filename} is too large')
        metadata[info.filename] = info
    if set(metadata) != {'manifest.json', 'images.json'}:
        raise ValueError('not a dataset backup (manifest.json/images.json missing or invalid)')
    try:
        manifest = json.loads(z.read(metadata['manifest.json']).decode('utf-8'))
        images_meta = json.loads(z.read(metadata['images.json']).decode('utf-8'))
    except (ValueError, UnicodeError, RecursionError, MemoryError,
            RuntimeError, NotImplementedError, EOFError, zipfile.BadZipFile,
            zlib.error, lzma.LZMAError):
        raise ValueError('not a dataset backup (manifest.json/images.json missing or invalid)')
    if not isinstance(manifest, dict):
        raise ValueError('invalid backup manifest')
    if manifest.get('format') != BACKUP_FORMAT:
        raise ValueError('not a dataset backup')
    version = manifest.get('version')
    if (isinstance(version, bool) or not isinstance(version, int)
            or version < 1):
        raise ValueError('invalid backup version')
    if version > BACKUP_VERSION:
        raise ValueError('backup made by a newer version of the app - update first')
    manifest = _normalized_backup_manifest(manifest)
    restored_training_mode = manifest['training_mode']
    if not isinstance(images_meta, list):
        raise ValueError('invalid backup image metadata')
    _validate_backup_limits(row_count=len(images_meta))
    images_meta = [
        _normalized_backup_image_meta(meta, version=version)
        for meta in images_meta
    ]
    seen_backup_ids = set()
    metadata_image_names = {}
    rescue_sources = set()
    rescue_parent_counts = {}
    for meta in images_meta:
        if not isinstance(meta, dict):
            raise ValueError('invalid backup image metadata')
        filename = meta.get('filename')
        if filename is not None and not isinstance(filename, str):
            raise ValueError('invalid backup image filename')
        if filename is not None:
            portable_name = _backup_basename(filename)
            if portable_name != filename and version >= 2:
                raise ValueError('invalid backup image filename')
            if portable_name == filename:
                filename_key = filename.casefold()
                if filename_key in metadata_image_names and version >= 2:
                    raise ValueError('duplicate backup image filename')
                metadata_image_names[filename_key] = filename
        backup_id = meta.get('backup_image_id')
        if backup_id is not None:
            if isinstance(backup_id, bool) or not isinstance(backup_id, int) or backup_id <= 0:
                raise ValueError('invalid backup image id')
            if backup_id in seen_backup_ids:
                raise ValueError('duplicate backup image id')
            seen_backup_ids.add(backup_id)
        derivation = meta.get('derivation_kind')
        if derivation not in (None, SMALL_IMAGE_SOURCE, KLEIN_SMALL_IMAGE,
                              KLEIN_IMAGE_IMPROVE):
            raise ValueError('invalid image derivation in backup')
        if (version >= 2 and filename is None
                and derivation != KLEIN_SMALL_IMAGE):
            raise ValueError('invalid metadata-only image row in backup')
        if derivation == SMALL_IMAGE_SOURCE:
            if backup_id is None or meta.get('parent_image_id') is not None:
                raise ValueError('invalid small-image source provenance')
            rescue_sources.add(backup_id)
        elif derivation == KLEIN_SMALL_IMAGE:
            parent_id = meta.get('parent_image_id')
            if backup_id is None or isinstance(parent_id, bool) or not isinstance(parent_id, int):
                raise ValueError('invalid Klein rescue provenance')
            rescue_parent_counts[parent_id] = rescue_parent_counts.get(parent_id, 0) + 1
            if rescue_parent_counts[parent_id] > 1:
                raise ValueError('multiple Klein rescue candidates for one source')
    if any(parent_id not in rescue_sources for parent_id in rescue_parent_counts):
        raise ValueError('Klein rescue candidate has no valid source')
    infos = []
    for info in all_infos:
        if info.is_dir() or info.filename in ('manifest.json', 'images.json'):
            continue
        if info.filename.startswith('analysis-cache/'):
            continue
        if info.filename.startswith(('ref/', 'images/')):
            prefix, candidate = info.filename.split('/', 1)
            base = _backup_basename(candidate)
            if base is None or info.filename != f'{prefix}/{base}':
                if version >= 2:
                    raise ValueError('invalid image/ref filename in backup')
                continue
            infos.append(info)
            continue
        if version >= 2:
            raise ValueError(f'unexpected file in backup: {info.filename}')
    if len(infos) > _BACKUP_MAX_FILES:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    archive_names = {'ref': {}, 'images': {}}
    for info in infos:
        prefix, candidate = info.filename.split('/', 1)
        name = _backup_basename(candidate)
        if name:
            key = name.casefold()
            if key in archive_names[prefix]:
                raise ValueError(
                    f'backup has duplicate {prefix} filename: {name}')
            archive_names[prefix][key] = name
    collisions = set(archive_names['ref']).intersection(archive_names['images'])
    if collisions:
        collision = archive_names['images'][next(iter(collisions))]
        raise ValueError(f'backup has colliding ref/image filename: {collision}')
    if version >= 2:
        archive_image_keys = set(archive_names['images'])
        metadata_image_keys = set(metadata_image_names)
        missing_images = metadata_image_keys.difference(archive_image_keys)
        orphan_images = archive_image_keys.difference(metadata_image_keys)
        if missing_images:
            raise ValueError(
                f'backup image is missing: {metadata_image_names[next(iter(missing_images))]}')
        if orphan_images:
            raise ValueError(
                f'unreferenced image in backup: '
                f'{archive_names["images"][next(iter(orphan_images))]}')
        for key in archive_image_keys:
            if archive_names['images'][key] != metadata_image_names[key]:
                raise ValueError('backup image filename case does not match metadata')

        required_ref_names = []
        for field in ('ref_filename', 'ref_original_filename'):
            raw_name = manifest.get(field)
            if raw_name is None:
                continue
            name = _backup_basename(raw_name)
            if name != raw_name:
                raise ValueError(f'invalid backup {field}')
            required_ref_names.append(name)
        extra_ref_names = _backup_extra_ref_names(
            manifest.get('ref_extra_filenames'))
        required_ref_names.extend(extra_ref_names)
        required_ref_keys = {name.casefold() for name in required_ref_names}
        allowed_ref_keys = set(required_ref_keys)
        for name in extra_ref_names:
            original = extra_ref_original_name(name)
            if original:
                allowed_ref_keys.add(original.casefold())
        archive_ref_keys = set(archive_names['ref'])
        missing_refs = required_ref_keys.difference(archive_ref_keys)
        orphan_refs = archive_ref_keys.difference(allowed_ref_keys)
        if missing_refs:
            raise ValueError('backup reference image is missing')
        if orphan_refs:
            raise ValueError(
                f'unreferenced reference image in backup: '
                f'{archive_names["ref"][next(iter(orphan_refs))]}')

    # Derive cache ownership only after the exact set of restorable rows/files
    # has been validated. A crafted skipped row cannot smuggle an otherwise
    # unowned sidecar into the restored Dataset folder.
    requested_cache_refs = set()
    if version >= 2:
        for meta in images_meta:
            raw_snapshot = meta.get('bank_analysis_snapshot')
            snapshot = bank_transfer_metadata.parse_snapshot(raw_snapshot)
            if raw_snapshot is not None and snapshot is None:
                raise ValueError('invalid Bank analysis snapshot in backup')
            if snapshot and snapshot.get('cache_ref'):
                cache_ref = snapshot['cache_ref']
                if not bank_transfer_metadata.is_content_addressed_cache_ref(
                        cache_ref):
                    raise ValueError(
                        'backup analysis cache reference is not content-addressed')
                requested_cache_refs.add(cache_ref)
    cache_infos = {}
    seen_cache_refs = set()
    for info in all_infos:
        if info.is_dir():
            continue
        if not info.filename.startswith('analysis-cache/'):
            continue
        match = _BACKUP_ANALYSIS_CACHE_RE.fullmatch(info.filename)
        if not match:
            if version >= 2:
                raise ValueError('invalid analysis cache filename in backup')
            continue
        cache_ref = match.group('ref')
        if cache_ref in seen_cache_refs:
            raise ValueError(f'duplicate analysis cache in backup: {cache_ref}')
        seen_cache_refs.add(cache_ref)
        if version >= 2 and not bank_transfer_metadata.is_content_addressed_cache_ref(
                cache_ref):
            raise ValueError('backup analysis cache reference is not content-addressed')
        if cache_ref not in requested_cache_refs:
            if version >= 2:
                raise ValueError(f'unreferenced analysis cache in backup: {cache_ref}')
            continue
        if (info.file_size <= 0
                or info.file_size > bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES):
            raise ValueError('analysis cache sidecar is too large')
        cache_infos[cache_ref] = info
    missing_cache_refs = requested_cache_refs.difference(cache_infos)
    if missing_cache_refs:
        raise ValueError(
            f'analysis cache missing from backup: {sorted(missing_cache_refs)[0]}')

    # Validate and bind every requested cache before creating a staging folder or
    # opening a database transaction.  A v2 restore is all-or-nothing: a bad CRC,
    # truncated NPZ, wrong SHA or invalid shape aborts the whole archive instead
    # of silently deleting cache_ref and restoring weaker metadata.
    validated_cache_payloads = {}
    for cache_ref, info in cache_infos.items():
        try:
            with z.open(info) as source:
                raw = source.read(
                    bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES + 1)
        except (OSError, EOFError, RuntimeError, MemoryError,
                zipfile.BadZipFile, zlib.error, lzma.LZMAError) as exc:
            raise ValueError('analysis cache sidecar is unreadable') from exc
        if (len(raw) != info.file_size
                or len(raw) > bank_transfer_metadata.CACHE_SIDECAR_MAX_BYTES
                or bank_transfer_metadata.read_cache_sidecar_bytes(
                    raw, expected_ref=cache_ref) is None):
            raise ValueError(
                'analysis cache sidecar is malformed or has a digest mismatch')
        validated_cache_payloads[cache_ref] = raw
    name = (manifest.get('name') or 'Restored dataset')[:100]
    trigger = (manifest.get('trigger_word') or 'restored')[:60]
    # Extract first into a sibling directory: it is on the same volume as the final
    # dataset folder, so promotion is a single rename.  The database transaction is
    # only opened after extraction succeeds; no empty dataset can become visible.
    root = str(cfg.dataset_images_root())
    staging_dir = os.path.join(root, f'.restore-{uuid.uuid4().hex}.tmp')
    os.mkdir(staging_dir)
    final_dir = None
    promoted = False
    db_started = False
    try:
        extracted_images = set()
        extracted_refs = {}
        for info in infos:
            prefix, candidate = info.filename.split('/', 1)
            base = _backup_basename(candidate)
            if not base:
                continue   # nested path or weird name -> skip, never traverse
            # Decode/verify all archive image bytes before they ever reach the
            # restore staging directory. That keeps the eventual rename/promotion
            # atomic even for compact pixel bombs or content/extension lies.
            raw = _read_validated_backup_image(z, info, base)
            with open(os.path.join(staging_dir, base), 'wb') as dst:
                # Keep this copy seam (rather than ``dst.write(raw)``): it is
                # deliberately fault-injectable by the atomic restore regression.
                shutil.copyfileobj(io.BytesIO(raw), dst, 1024 * 1024)
            if prefix == 'ref':
                extracted_refs.setdefault(base.casefold(), base)
            else:
                extracted_images.add(base)

        extracted_cache_refs = set()
        if validated_cache_payloads:
            cache_dir = os.path.join(staging_dir, '.bank-analysis-cache')
            os.mkdir(cache_dir)
            for cache_ref, raw in validated_cache_payloads.items():
                cache_path = os.path.join(cache_dir, f'{cache_ref}.npz')
                try:
                    with open(cache_path, 'wb') as dst:
                        dst.write(raw)
                        dst.flush()
                        os.fsync(dst.fileno())
                except OSError as exc:
                    raise ValueError(
                        'could not restore analysis cache sidecar') from exc
                extracted_cache_refs.add(cache_ref)

        db_started = True
        ds = create_dataset(user_id, name, trigger, kind=manifest.get('kind'),
                            concept_desc=manifest.get('concept_desc'),
                            train_type=manifest.get('train_type'), commit=False)
        for field in ('concept_terms', 'train_variant', 'train_settings',
                      'best_settings', 'fidelity'):
            setattr(ds, field, manifest.get(field))
        ds.training_mode = restored_training_mode
        ds.train_base_model = _portable_train_base_model(manifest.get('train_base_model'))
        ds.ref_filename = _backup_basename(manifest.get('ref_filename'))
        ds.ref_original_filename = _backup_basename(
            manifest.get('ref_original_filename'))
        final_dir = os.path.join(root, str(ds.id))
        if os.path.exists(final_dir):
            # Never merge with or delete a pre-existing orphan directory.
            raise RuntimeError(f'dataset folder already exists for id {ds.id}')

        n_rows = 0
        restored_rows = []
        valid_source_ids = {
            meta.get('backup_image_id') for meta in images_meta
            if isinstance(meta, dict)
            and meta.get('derivation_kind') == SMALL_IMAGE_SOURCE
            and meta.get('filename') in extracted_images
        }
        for meta in images_meta:
            if not isinstance(meta, dict):
                continue
            fn = meta.get('filename')
            derivation = meta.get('derivation_kind')
            is_candidate = derivation == KLEIN_SMALL_IMAGE
            if fn and fn not in extracted_images:
                continue
            if not fn and not is_candidate:
                continue   # only rescue candidates have meaningful metadata-only rows
            if is_candidate and meta.get('parent_image_id') not in valid_source_ids:
                continue   # never restore an orphaned candidate
            values = {f: meta.get(f) for f in _BACKUP_IMG_FIELDS
                      if f not in ('filename', 'parent_image_id')}
            # Backup input is untrusted. Unknown/invalid provenance is dropped,
            # while valid Pexels or web-search metadata is canonicalized back to JSON TEXT.
            values['source_metadata'] = _source_metadata_storage(
                values.get('source_metadata'))
            snapshot = bank_transfer_metadata.parse_snapshot(
                values.get('bank_analysis_snapshot'))
            if (snapshot and snapshot.get('cache_ref')
                    and snapshot['cache_ref'] not in extracted_cache_refs):
                raise ValueError('analysis cache sidecar was not restored')
            values['bank_analysis_snapshot'] = (
                bank_transfer_metadata.normalized_snapshot_storage(snapshot)
                if snapshot is not None else None)
            raw_transfer_metadata = values.get('transfer_metadata')
            values['transfer_metadata'] = (
                bank_transfer_metadata.normalized_transfer_metadata_storage(
                    raw_transfer_metadata))
            if (raw_transfer_metadata is not None
                    and values['transfer_metadata'] is None):
                raise ValueError('invalid Bank/Dataset transfer metadata in backup')
            if is_candidate and not fn and values.get('status') in ('pending', 'keep'):
                values['status'] = 'failed'
                values['fail_reason'] = (
                    'Klein rescue was in flight when this backup was created; '
                    'the original image is preserved, but the job must be started again.'
                )
            img = FaceDatasetImage(dataset_id=ds.id,
                                   **values,
                                   filename=fn)
            db.session.add(img)
            restored_rows.append((img, meta))
            n_rows += 1
        # Allocate new ids first, then restore the graph strictly within this backup.
        # A missing/skipped parent clears the relationship rather than pointing at an
        # unrelated row that happens to reuse the old numeric id.
        db.session.flush()
        id_map = {meta.get('backup_image_id'): img.id for img, meta in restored_rows
                  if meta.get('backup_image_id') is not None}
        for img, meta in restored_rows:
            img.parent_image_id = id_map.get(meta.get('parent_image_id'))
        # Reference fields are rebuilt exclusively from actual ref/ archive files.
        # Never retain paths, missing names, image-only files, or case variants.
        ds.ref_filename = (extracted_refs.get(ds.ref_filename.casefold())
                           if ds.ref_filename else None)
        ds.ref_original_filename = (
            extracted_refs.get(ds.ref_original_filename.casefold())
            if ds.ref_original_filename else None)
        used_ref_keys = {
            ref.casefold() for ref in (ds.ref_filename, ds.ref_original_filename) if ref
        }
        restored_extras = []
        for requested in _backup_extra_ref_names(
                manifest.get('ref_extra_filenames'), limit=None):
            key = requested.casefold()
            actual = extracted_refs.get(key)
            if not actual or key in used_ref_keys:
                continue
            used_ref_keys.add(key)
            restored_extras.append(actual)
            if len(restored_extras) >= MAX_EXTRA_REFS:
                break
        ds.ref_extra_filenames = json.dumps(restored_extras)

        os.replace(staging_dir, final_dir)
        promoted = True
        db.session.commit()
    except Exception:
        try:
            if db_started:
                db.session.rollback()
        finally:
            if promoted and final_dir:
                shutil.rmtree(final_dir, ignore_errors=True)
        raise
    finally:
        # Exists on extraction/build/promotion failure; after promotion the old path
        # is already gone.  Never leave hidden partial restores behind.
        shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info(f"dataset backup restored: '{name}' -> #{ds.id} ({n_rows} image rows)")
    return ds


@_serialize_dataset_ingest
def replace_in_captions(user_id, dataset_id, find, replace, mode='text'):
    """Bulk-edit the captions of KEPT images (the ones that train). Two modes:

    - 'text': whole-word replace, CASE-INSENSITIVE — the same match rule as the
      grid filter ("smile" hits "a warm smile" but not "smiling") and the most-
      frequent-words counter, both case-insensitive. So clicking a "bulldog ×41"
      chip and stripping it removes all 41 whatever their casing (the captions
      hold "Bulldog"); a case-sensitive substring replace matched 0 and looked
      broken. Whole-word so "red" never eats the "red" inside "colored". When
      `replace` is empty the gaps a stripped word leaves in prose are tidied.
    - 'tag':  the caption is treated as a comma-separated tag list (booru); `find`
      must match a WHOLE tag (trimmed, case-insensitive) and is replaced by
      `replace` — or dropped when `replace` is empty. Avoids the ', ,' artifacts a
      substring removal would leave in tag captions. Result is deduped
      case-insensitively (keeping first occurrence / original casing).

    Returns the number of captions actually changed."""
    if mode not in ('text', 'tag'):
        raise ValueError('invalid mode')
    find = (find or '').strip() if mode == 'tag' else (find or '')
    if not find:
        raise ValueError('find is required')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.caption.isnot(None)).all())
    changed = 0
    for img in rows:
        old = img.caption or ''
        if mode == 'text':
            pattern = re.compile(rf'\b{re.escape(find)}\b', re.IGNORECASE)
            new = pattern.sub(replace or '', old)
            if not (replace or '').strip():          # stripping: tidy prose gaps
                new = re.sub(r'\s+([,.;:])', r'\1', new)   # space before punctuation
                new = re.sub(r'(,\s*){2,}', ', ', new)     # collapsed repeated commas
                new = re.sub(r'\s{2,}', ' ', new)          # collapsed double spaces
                new = new.strip(' ,;')
        else:
            tags = [t.strip() for t in old.split(',')]
            out, seen = [], set()
            for t in tags:
                if not t:
                    continue
                nt = (replace or '').strip() if t.lower() == find.lower() else t
                if not nt or nt.lower() in seen:
                    continue
                seen.add(nt.lower())
                out.append(nt)
            new = ', '.join(out)
        new = _cap_caption(new) or None
        if new != img.caption:
            # A find/replace is a CORRECTION, so the row becomes the user's even
            # when a model wrote the sentence it started from: cleaning a term out
            # of 200 captions is exactly the work a later forced pass must not undo.
            caption_origin.stamp(img, new, caption_origin.ASSERTED)
            changed += 1
    if changed:
        db.session.commit()
    return changed


# Batch curation (multi-select in the grid). 'pending' = reset the triage state.
BATCH_ACTIONS = ('keep', 'reject', 'pending', 'delete', 'clear_caption')
BATCH_MAX_IMAGES = 10_000
_SQLITE_ID_MAX = (1 << 63) - 1


def normalize_batch_image_ids(image_ids) -> list[int]:
    if not isinstance(image_ids, list) or not image_ids:
        raise ValueError("'ids' must be a non-empty list")
    if len(image_ids) > BATCH_MAX_IMAGES:
        raise ValueError(f'too many image ids (max {BATCH_MAX_IMAGES})')
    out = []
    seen = set()
    for value in image_ids:
        if (isinstance(value, bool) or not isinstance(value, int)
                or value <= 0 or value > _SQLITE_ID_MAX):
            raise ValueError("'ids' must contain positive integer image ids")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


@_serialize_dataset_ingest
def batch_image_action(user_id, dataset_id, image_ids, action):
    """Apply one whitelisted action to a set of this dataset's images in one call
    (the grid's multi-select). Ownership is checked once on the dataset; ids that
    don't belong to it (or don't exist) are silently skipped, so a stale selection
    after a poll refresh can't touch another dataset's rows. Returns the number of
    images actually affected."""
    if action not in BATCH_ACTIONS:
        raise ValueError('invalid action')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    ids = normalize_batch_image_ids(image_ids)
    rows = []
    for i0 in range(0, len(ids), 500):
        rows.extend(FaceDatasetImage.query
                    .filter_by(dataset_id=dataset_id)
                    .filter(FaceDatasetImage.id.in_(ids[i0:i0 + 500])).all())
    n = 0
    if action != 'clear_caption' and any(
            img.derivation_kind in _SMALL_IMAGE_DERIVATIONS for img in rows):
        raise ValueError('resolve small-image rescue pairs with the dedicated review action')
    if action == 'delete':
        # Per-image path: reuses delete_image (file removal + pending-job cancel).
        for img in rows:
            if delete_image(user_id, img.id):
                n += 1
        return n
    for img in rows:
        if action == 'clear_caption':
            # The stamp goes with the text. An 'asserted' marker left on a row with
            # no caption would be a row every future pass skips — permanently blank,
            # for a reason nothing on screen could explain.
            caption_origin.stamp(img, None, None)
        else:
            # Never resurrect a failed generation into keep/reject — the tile has
            # no file; regenerate is the only way out of 'failed'.
            if img.status == 'failed':
                continue
            if action == 'reject':
                _clear_watermark_metadata(img)
            img.status = action
        n += 1
    if action == 'keep':
        # This is deliberately a second phase.  If both a source and its
        # improvement are selected, every explicit choice is applied first,
        # then the kept candidate wins regardless of database/query order.
        for img in rows:
            if img.status == 'keep':
                _unkeep_parent_for_kept_improvement(img)
    db.session.commit()
    return n


def _ref_crop_source_path(ds) -> str:
    """The image a manual/auto re-crop reads from: the full-frame ORIGINAL when we
    kept one, else the cropped ref (legacy datasets uploaded before we stored the
    original — they can still be re-cropped, only not wider than the existing crop)."""
    name = ds.ref_original_filename or ds.ref_filename
    return os.path.join(_dataset_dir(ds.id), name)


def crop_reference(user_id, dataset_id, x, y, w, h):
    """Manually crop the dataset reference to (x,y,w,h), long side capped at 1024
    (never enlarged). The box is
    in the ORIGINAL's pixel space (the editor shows the original), and we write the
    derived square to ref_filename WITHOUT touching the original — so the user can
    re-crop wider or tighter any number of times."""
    with reference_mutation(dataset_id):
        ds = get_dataset(user_id, dataset_id)
        if not ds or not ds.ref_filename:
            return False
        ok, _scale = _crop_resize_file(
            _ref_crop_source_path(ds), x, y, w, h, dst=_ref_path(ds))
        if ok:
            invalidate_reference_edit(dataset_id)   # a pending Before/After is now stale
        return ok


def recrop_reference_auto(user_id, dataset_id):
    """Re-run the automatic head-crop on the ORIGINAL, overwriting ref_filename.
    Returns (ok, head_detected). CALLER holds the GPU vision window. Lets the user
    reset to the auto framing after manual edits, without re-uploading the photo."""
    with reference_mutation(dataset_id):
        ds = get_dataset(user_id, dataset_id)
        if not ds or not ds.ref_filename:
            return False, False
        try:
            with open(_ref_crop_source_path(ds), 'rb') as fh:
                raw = fh.read()
        except OSError:
            return False, False
        webp, detected = face_crop_to_square_webp(
            raw, pad=REF_CROP_PAD, return_detected=True)
        with open(_ref_path(ds), 'wb') as fh:
            fh.write(webp)
        invalidate_reference_edit(dataset_id)    # a pending Before/After is now stale
        return True, detected


def _edit_engine_call(engine, refs, prompt):
    """Dispatch an edit to an API engine: reference list + prompt -> edited bytes
    (or None). PURE dispatch — no file reading (the caller snapshots the refs in
    the REQUEST thread, never the worker). The reference is a face crop, so the
    edit keeps its framing (1:1) — an in-place change, not a re-composition. The
    ChatGPT auth lane is pinned once, like the batch, so it never silently
    reroutes onto the paid API key mid-call."""
    if engine not in API_ENGINES:
        raise ValueError(f'unknown edit engine: {engine}')
    generate = _api_generate_fn(engine)
    gen_kwargs = {'aspect_ratio': '1:1'}
    if engine == 'chatgpt':
        from .chatgpt_image import _use_subscription
        gen_kwargs['force_lane'] = 'subscription' if _use_subscription() else 'api'
    return generate(refs, prompt, **gen_kwargs)


def normalize_edit_engines(engines):
    """Canonical ordered engine selection for both legacy and batch requests."""
    raw_values = [engines] if isinstance(engines, str) else list(engines or ())
    selected = []
    for raw in raw_values:
        engine = str(raw or '').strip().lower()
        if not engine:
            raise ValueError('select at least one engine for the reference edit')
        if engine not in selected:
            selected.append(engine)
    allowed = editable_engines()
    if not selected:
        raise ValueError('select at least one engine for the reference edit')
    # Apply the finite bound after normalization/deduplication. Membership below
    # makes it structurally unreachable today, but keeps the contract explicit.
    if len(selected) > len(allowed):
        raise ValueError(f'select at most {len(allowed)} edit engines')
    if any(engine not in allowed for engine in selected):
        raise ValueError(edit_engine_choice_message())
    return tuple(selected)


def _preflight_local_reference_edit(ds, engine):
    """Run local preflights available without enqueuing a render."""
    if engine == KREA_ENGINE:
        from . import krea_edit_helper as helper
        helper.preflight()
    # Klein's complete admission check intentionally lives in enqueue_klein_edit;
    # all local enqueues finish before any paid API thread is started below.


def start_reference_edit(app, user_id, dataset_id, engine, prompt,
                         extra_edit_ref_bytes=None, retry_batch_id=None):
    """Start a background reference-edit job and RETURN AT ONCE (the request no
    longer blocks 1-3 min, so a backgrounded mobile tab can't kill it). Snapshots
    the reference + extras + modal images HERE, in the request thread (never
    re-read in the worker), registers the candidate job and a 'edit_reference'
    activity so the existing payload poll tracks it, then spawns the worker.

    Ref list sent to the engine: the primary reference FIRST (ChatGPT's
    /images/edits treats image[0] as the edit base), then the dataset's extra refs
    and any transient edit-reference images the user added in the modal — all ride
    along as identity anchors so the edit keeps the same face. Every API engine
    forwards the WHOLE list (OpenRouter as `input_references`); a model that takes
    fewer says so in its own error rather than having some silently dropped here.

    LOCAL engines (Klein, Krea 2 Edit) take a different route entirely — see
    _start_local_reference_edit: no blocking call, a ComfyUI queue job instead.
    They also take their second reference from DIFFERENT places (Klein: the
    dataset's extra angles, by path; Krea: one image uploaded in this dialog),
    which is a fact of their graphs — see LOCAL_EDIT_REF_SUPPORT — and is stated
    in the UI at pick time rather than discovered as a silent drop here.

    Raises ValueError for a bad engine / empty prompt / missing reference (the
    route maps it to 400/404)."""
    _guard_not_bank_export(dataset_id)
    engines = normalize_edit_engines(engine)
    prompt = (prompt or '').strip()
    if not prompt:
        raise ValueError('describe the edit first')
    # Capture the mutation epoch before *any* dataset/reference-state read. If a
    # mutation commits after this point, start_batch's CAS rejects the snapshot;
    # if it committed before this point, the reads below see the new state.
    reference_revision = reference_edit_jobs.reference_revision(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        raise ValueError('reference image required')
    if not os.path.exists(_ref_path(ds)):
        raise ValueError('reference image file missing')
    api_engines = tuple(item for item in engines if item in API_ENGINES)
    local_engines = tuple(item for item in engines if item in LOCAL_ENGINES)
    transient_refs = tuple(extra_edit_ref_bytes or ())
    if len(transient_refs) > MAX_EDIT_REFERENCE_UPLOADS:
        raise ValueError(
            f'add at most {MAX_EDIT_REFERENCE_UPLOADS} extra edit references')

    # One immutable primary+persistent-reference snapshot feeds every lane. API
    # siblings consume the bytes directly; local siblings below consume temporary
    # files written once from these exact bytes, never a later read of the master.
    dataset_ref_bytes = tuple(_all_ref_bytes(ds))
    # Which selected local engines can actually receive the dialog's uploads —
    # computed BEFORE the refusal below, because it is what the refusal turns on.
    modal_local = local_engines_taking_modal_refs(local_engines)
    # Sanitize the dialog's uploads HERE — once, and before anything
    # destructive. start_batch below SUPERSEDES the batch on screen: it unlinks
    # the previous candidate and cancels a render still in flight. A rejected
    # image (HEIC, animated GIF, truncated PNG — all of which pass the browser's
    # image/* filter) must therefore fail before it, or dropping the wrong file
    # destroys a candidate the user had not kept yet. The API lane always did
    # this; the local lane used to be refused outright, so no ordering existed.
    # Both lanes now read the SAME validated bytes.
    modal_bytes = tuple(
        sanitize_external_reference(raw, label=f'extra edit reference {index}')
        for index, raw in enumerate(transient_refs, 1) if raw)
    refs = None
    if api_engines:
        refs = tuple(list(dataset_ref_bytes) + list(modal_bytes))
    elif transient_refs and not modal_local:
        # Refuse ONLY when nothing selected can read these bytes. Krea now can,
        # so this is no longer "local engines cannot take uploads" — it is the
        # narrower, still-true "the engine you picked has nowhere to put them".
        local = local_engines[0]
        raise ValueError(
            f'{engine_labels().get(local, local)} has no slot for the extra reference images '
            'added here — remove them, or pick an engine that takes one')

    # Validate every selected local lane before replacing the current results.
    # Full enqueue happens before API threads below, closing remaining admission
    # gaps without ever billing a paid sibling first.
    for local in local_engines:
        _preflight_local_reference_edit(ds, local)
    dsdir = _dataset_dir(dataset_id)
    started = reference_edit_jobs.start_batch(
        dataset_id, dsdir, engines, prompt,
        expected_revision=reference_revision,
        expected_batch_id=retry_batch_id)
    batch_token, tokens = started['batch_token'], started['tokens']
    act_token = dataset_activity.begin(
        dataset_id, 'edit_reference', total=len(engines),
        engine=engines[0] if len(engines) == 1 else None)
    if not reference_edit_jobs.attach_activity(dataset_id, batch_token, act_token):
        dataset_activity.end(act_token)
        raise RuntimeError('reference edit was superseded while it was starting')

    # Prove admission for every local sibling before starting a paid API thread.
    # If the second local enqueue fails, clear cancels the first queue job and
    # closes the shared activity exactly once.
    local_snapshot_paths = []
    local_modal_paths = []
    try:
        if local_engines:
            snapshot_tag = uuid.uuid4().hex[:8]
            # The primary is always needed. The dataset's EXTRAS only when a
            # selected engine reads them — Krea reads the dialog instead, so a
            # Krea-only edit used to write files nobody would ever open. Derived
            # from the support table, never from engine names.
            wants_dataset_extras = bool(local_engines_taking_dataset_refs(local_engines))
            for index, raw in enumerate(dataset_ref_bytes):
                if index and not wants_dataset_extras:
                    break
                filename = (
                    f'{user_id}{reference_edit_jobs.CANDIDATE_MARKER}'
                    f'snapshot_{snapshot_tag}_{index}.webp')
                path = os.path.join(dsdir, filename)
                local_snapshot_paths.append(path)
                write_image_atomic(path, raw)
            # The dialog's own uploads, given the SAME treatment as the primary:
            # already-validated bytes, written once, handed over as paths. Only
            # staged when an engine will read them — an upload for an API-only
            # batch has no business touching the dataset folder.
            for index, raw in enumerate(modal_bytes if modal_local else ()):
                filename = (
                    f'{user_id}{reference_edit_jobs.CANDIDATE_MARKER}'
                    f'modalref_{snapshot_tag}_{index}.webp')
                path = os.path.join(dsdir, filename)
                local_modal_paths.append(path)
                write_image_atomic(path, raw)
        for local in local_engines:
            _enqueue_local_reference_edit(
                user_id, dataset_id, ds, local, prompt, tokens[local],
                local_snapshot_paths[0], local_snapshot_paths[1:], local_modal_paths)
    except Exception:
        reference_edit_jobs.clear_batch(dataset_id, batch_token, dsdir)
        raise
    finally:
        for path in local_snapshot_paths + local_modal_paths:
            reference_edit_jobs._unlink(path)

    for api_engine in api_engines:
        token = tokens[api_engine]
        try:
            threading.Thread(
                target=_run_reference_edit,
                args=(app, user_id, dataset_id, token, act_token,
                      api_engine, refs, prompt, True),
                daemon=True).start()
        except Exception as exc:
            logger.exception(
                'reference edit worker could not start (dataset %s, engine %s)',
                dataset_id, api_engine)
            reference_edit_jobs.set_failed(
                dataset_id, token, f'{api_engine}: failed to start edit: {exc}')
            _finish_reference_edit_activity(
                dataset_id, token, act_token, shared_activity=True)

    # The route returns this exact opaque id to the browser. Reading the registry
    # after this function returns would race a second tab starting another batch.
    return started['batch_id']


#: Which second reference each LOCAL engine takes, and — the part that matters —
#: WHERE it comes from. The two local engines want opposite things, so one pool
#: cannot serve both:
#:
#:   * 'dataset_only' (Klein) — the dataset's extra refs, chained as native
#:     ReferenceLatent nodes, no ceiling of its own. Those are ANGLES OF THE SAME
#:     FACE and they lock identity across every generation, not just this edit.
#:     Persistent input, so the dataset's reference card is their home.
#:   * 'modal_one' (Krea) — ONE image uploaded in the edit dialog, and none of
#:     the dataset's. Its node pack trained the `_b` slot for a DIFFERENT subject
#:     ("scene first, subject second"), which makes the dataset pool precisely
#:     the wrong source: everything in it is another angle of the same person,
#:     the one photo that slot mis-handles (documented failure: the subject comes
#:     back duplicated). It is a per-edit compositional input — "put her in this
#:     room", "next to him" — so it belongs to the edit, not to the dataset.
#:
#: That split IS the design. The first version of this feature fed Krea from the
#: dataset pool and therefore guaranteed the wrong photo on every run.
#: LOAD-BEARING, not documentation: the enqueue below reads it, so a third local
#: engine cannot be added without deciding where its references come from. The
#: values are mirrored in frontend EDIT_REF_SUPPORT (contract-tested), because
#: the UI has to say this at pick time, not discover it as a silent drop.
LOCAL_EDIT_REF_SUPPORT = {'klein': 'dataset_only', 'krea': 'modal_one'}

#: How many DATASET extras each support value forwards. None = no ceiling beyond
#: the dataset's own MAX_EXTRA_REFS. A support value absent from this map takes
#: none — which is what a newly added engine should do until someone decides.
LOCAL_EDIT_REF_LIMITS = {'dataset_only': None}

#: How many of the MODAL's own uploads each support value forwards. They reach a
#: local engine as temporary FILES written from the request bytes — the same
#: hand-off the primary reference already used, which is why "local engines
#: cannot take the images added here" was always a routing decision rather than
#: a limitation of the graphs.
MODAL_EDIT_REF_LIMITS = {'modal_one': 1}


def local_edit_extra_refs(engine, extra_ref_paths):
    """The DATASET extras THIS engine consumes, in order (Klein's angles).

    One place decides, so the enqueue below and what the modal claims can never
    disagree — the failure this prevents is a UI promising angles to an engine
    whose graph was never going to read them."""
    limit = LOCAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(engine), 0)
    paths = list(extra_ref_paths or [])
    return paths if limit is None else paths[:limit]


def local_edit_modal_refs(engine, modal_ref_paths):
    """The MODAL's uploads THIS engine consumes, in order (Krea's second subject)."""
    limit = MODAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(engine), 0)
    return list(modal_ref_paths or [])[:limit]


def local_engines_taking_dataset_refs(engines):
    """The selected local engines that read the dataset's extra angles. Empty
    means nothing will open them, which is what lets the caller skip writing
    temporary copies no consumer exists for."""
    return [e for e in (engines or [])
            if LOCAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(e), 0) != 0]


def local_engines_taking_modal_refs(engines):
    """Selected local engines that read the dialog's own uploads. An empty result
    with uploads present is what turns them into a loud refusal instead of a
    silent drop."""
    return [e for e in (engines or [])
            if MODAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(e), 0)]


def _enqueue_local_reference_edit(user_id, dataset_id, ds, engine, prompt, token,
                                  ref_path, extra_ref_paths, modal_ref_paths=()):
    """Reference edit on the user's OWN GPU: free, private, no key, no bill — and
    therefore the lane that makes "try five prompts until it's right" reasonable.

    It does NOT get its own waiting machinery. The edit is enqueued on the app's
    existing ComfyUI image queue exactly like a generated variation, and the queue
    worker's completion dispatch calls link_completed_reference_edit when it
    lands. The registry entry is what the modal polls either way, so the client
    sees one contract for both lanes (running -> ready|failed).

    Every local enqueue completes before a selected API thread starts. A missing
    weight or node pack therefore surfaces as the same actionable 409 as generate,
    and any already-enqueued local sibling is cancelled without billing an API."""
    meta = {'is_reference_edit': True, 'dataset_id': dataset_id}
    try:
        if engine == KREA_ENGINE:
            from . import krea_edit_helper as helper
            job_id = helper.enqueue_krea_edit(
                user_id=str(user_id), source_filename=os.path.basename(ref_path),
                source_path=ref_path, edit_prompt=prompt, extra_metadata=meta,
                # From the DIALOG, never from the dataset's angles: the `_b` slot
                # wants a different subject, and the dataset pool holds only more
                # views of the same one. One image — the slot has room for one.
                extra_ref_paths=local_edit_modal_refs(engine, modal_ref_paths))
        else:
            from .klein_edit_helper import enqueue_klein_edit
            # The dataset's extra refs DO reach Klein (native ReferenceLatent
            # chaining) — the same anchors the API lane sends as bytes. Gated on
            # the table above rather than on the engine name, so the two can't
            # disagree.
            extras = local_edit_extra_refs(engine, extra_ref_paths)
            job_id = enqueue_klein_edit(
                user_id=str(user_id), source_filename=os.path.basename(ref_path),
                source_path=ref_path, edit_prompt=prompt, extra_ref_paths=extras,
                # The dataset's model, like every other Klein lane: this edit
                # produces the REFERENCE the whole dataset is built from, so it is
                # the last place that should run on a different model than the
                # images it anchors. None (never chose) = the historical auto pick.
                klein_model=dataset_klein_model(ds),
                sampler_steps=_generation_steps(),
                # An EDIT must obey the instruction, not a style LoRA nobody
                # picked: node 139 is pinned at 0.8 in the workflow file and the
                # setting (default 0) is what decides it here.
                base_lora_strength=_generation_base_lora_strength(),
                extra_metadata=meta)
    except Exception as exc:
        from .klein_edit_helper import KleinModelsMissing
        from .krea_edit_helper import KreaModelsMissing
        if isinstance(exc, (KleinModelsMissing, KreaModelsMissing)):
            # Typed on purpose: the route turns these into the SAME auto-download
            # 409 the generate path returns. Flattening them to a ValueError would
            # downgrade "I've started fetching the weight" to a bare 400.
            raise
        logger.exception('local reference edit could not be queued (dataset %s)', dataset_id)
        raise ValueError(f'{engine_labels().get(engine, engine)}: {exc}') from exc
    if not reference_edit_jobs.attach_job(
            dataset_id, token, job_id, user_id=str(user_id)):
        # Superseded between the enqueue and here: cancel the render nobody awaits.
        _cancel_local_edit_job(job_id)
        raise RuntimeError('reference edit was superseded while it was starting')
    return job_id


def _cancel_local_edit_job(job_id):
    """Best-effort cancel of one just-enqueued job not owned by a live batch."""
    if not job_id:
        return
    try:
        from ..job_queue import queue_manager
        queue_manager.cancel_job(job_id)
    except Exception:
        logger.warning('reference edit: could not cancel queue job %s', job_id, exc_info=True)


def _finish_reference_edit_activity(dataset_id, token, fallback_act_token=None,
                                    shared_activity=False):
    """Advance one candidate and close the batch activity exactly once."""
    update = reference_edit_jobs.activity_update(dataset_id, token)
    if update is not None:
        progress_token = update.get('activity_token')
        if progress_token is not None:
            dataset_activity.progress(
                progress_token, done=update['done'], total=update['total'])
        if update.get('end_token') is not None:
            dataset_activity.end(update['end_token'])
        elif not update.get('managed') and not shared_activity:
            dataset_activity.end(fallback_act_token)
        return
    # Legacy direct worker calls register no shared token. A managed batch that
    # disappeared was already ended by its supersede/clear/TTL cleanup.
    if not shared_activity:
        dataset_activity.end(fallback_act_token)


def link_completed_reference_edit(job_id, filename, failed=False, reason=None):
    """Queue-worker callback for a LOCAL reference edit: turn the finished ComfyUI
    output into the candidate the modal is waiting on.

    Symmetric with link_completed_dataset_image, minus the DB row — a reference
    edit has no FaceDatasetImage; its whole state is the in-memory registry entry.
    No entry = the user discarded or superseded meanwhile: the output is deleted
    rather than left in ComfyUI's folder."""
    entry = reference_edit_jobs.find_by_job(job_id)
    dataset_id = entry['dataset_id'] if entry else None
    try:
        if entry is None:
            _drop_comfy_output(filename)
            return
        if failed:
            reference_edit_jobs.set_failed(
                dataset_id, entry['token'],
                f"{entry['engine']}: {reason or 'the render failed — see 🪵 Server log in Settings'}")
            return
        data = _read_comfy_output(filename)
        if not data:
            reference_edit_jobs.set_failed(
                dataset_id, entry['token'],
                f"{entry['engine']}: the finished image could not be retrieved from ComfyUI "
                '(not on disk, and the /view API fetch failed)')
            return
        # Same naming as the API worker's candidate — the marker is what keeps it
        # out of the grid and the backups; the owner prefix keeps it recognisable.
        cand_fn = (f"{entry.get('user_id') or 'local'}"
                   f'{reference_edit_jobs.CANDIDATE_MARKER}{uuid.uuid4().hex[:8]}.webp')
        cand_path = os.path.join(entry['dir'], cand_fn)
        write_image_atomic(cand_path, normalize_to_webp(data))
        _drop_comfy_output(filename)
        if not reference_edit_jobs.set_ready(dataset_id, entry['token'], cand_fn):
            reference_edit_jobs._unlink(cand_path)     # superseded: drop our orphan
    except Exception as exc:
        logger.exception('reference edit link failed (job %s)', job_id)
        if entry is not None:
            reference_edit_jobs.set_failed(dataset_id, entry['token'],
                                           f"{entry['engine']}: {exc}")
    finally:
        if entry is not None:
            _finish_reference_edit_activity(
                dataset_id, entry['token'], entry.get('act_token'),
                shared_activity=True)


def _validated_comfy_output_name(filename):
    """Return an opaque Comfy output basename, or ``None`` when untrusted.

    Comfy completion payloads cross a trust boundary.  In particular, a stale
    completion is still cleaned up, so accepting a path here would turn that
    cleanup into an arbitrary-file delete.  Keep the accepted shape deliberately
    narrower than a generic relative path: Comfy reference-edit outputs are
    always direct children of its output directory.
    """
    if not isinstance(filename, str) or not filename or '\x00' in filename:
        return None
    if filename in {'.', '..'} or filename != filename.strip():
        return None
    if '/' in filename or '\\' in filename or ':' in filename:
        return None
    if (os.path.isabs(filename) or ntpath.isabs(filename)
            or posixpath.isabs(filename) or ntpath.splitdrive(filename)[0]):
        return None
    if (os.path.basename(filename) != filename
            or ntpath.basename(filename) != filename
            or posixpath.basename(filename) != filename):
        return None
    return filename


def _resolve_comfy_output(filename):
    """Resolve a trusted direct-child output without traversing a reparse path.

    The boolean is distinct from ``path is not None``: a valid basename with no
    configured local output directory may still be fetched through Comfy's
    ``/view`` endpoint, while any rejected path must fail closed everywhere.
    """
    name = _validated_comfy_output_name(filename)
    d = _comfy_output_dir()
    if name is None:
        return None, None, False
    if not d:
        return name, None, True
    try:
        root = os.path.abspath(os.fspath(d))
        candidate = os.path.abspath(os.path.join(root, name))
        if os.path.normcase(os.path.commonpath((root, candidate))) != os.path.normcase(root):
            return name, None, False
    except (OSError, TypeError, ValueError):
        return name, None, False

    # Check the candidate and every existing ancestor using lstat, before a
    # content read/delete can follow a symlink or Windows junction/reparse point.
    current = candidate
    while True:
        _st, blocked = _safe_lstat(current)
        if blocked:
            return name, None, False
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    try:
        canonical_root = os.path.realpath(root)
        canonical_candidate = os.path.realpath(candidate)
        contained = os.path.commonpath((canonical_root, canonical_candidate))
        if os.path.normcase(contained) != os.path.normcase(canonical_root):
            return name, None, False
    except (OSError, ValueError):
        return name, None, False
    return name, candidate, True


def _comfy_output_path(filename):
    _name, candidate, allowed = _resolve_comfy_output(filename)
    return candidate if allowed else None


def _is_reparse_stat(st):
    attrs = getattr(st, 'st_file_attributes', 0)
    reparse_flag = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    return stat.S_ISLNK(st.st_mode) or bool(attrs & reparse_flag)


def _safe_lstat(path):
    """Return (stat, blocked): metadata errors and reparse points fail closed."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, False
    except OSError:
        return None, True
    return st, _is_reparse_stat(st)


def _read_comfy_output(filename):
    """Bytes of a finished ComfyUI output, from disk when we can see its folder,
    else over the /view API (a custom or unconfigured output path). None when
    neither works."""
    name, p, allowed = _resolve_comfy_output(filename)
    if not allowed:
        return None
    if p:
        root_st, root_blocked = _safe_lstat(os.path.dirname(p))
        if root_blocked:
            return None
        file_st, file_blocked = _safe_lstat(p)
        if file_blocked or (file_st is not None and not stat.S_ISREG(file_st.st_mode)):
            return None
        if root_st is not None and not stat.S_ISDIR(root_st.st_mode):
            return None
        if file_st is not None:
            fd = None
            try:
                flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
                flags |= getattr(os, 'O_NOFOLLOW', 0)
                fd = os.open(p, flags)
                opened_st = os.fstat(fd)
                if (not stat.S_ISREG(opened_st.st_mode)
                        or not os.path.samestat(file_st, opened_st)):
                    return None
                with os.fdopen(fd, 'rb') as fh:
                    fd = None
                    return fh.read()
            except OSError:
                # A file that changed after lstat is treated as hostile rather
                # than retried through Comfy's /view endpoint.
                return None
            finally:
                if fd is not None:
                    os.close(fd)
    from ..utils.comfyui import fetch_output_image_bytes
    return fetch_output_image_bytes(name)


def _drop_comfy_output(filename):
    """Remove an unlinked ComfyUI output. Only ever called on a file this app just
    produced for a transient candidate — never on user data."""
    _name, p, allowed = _resolve_comfy_output(filename)
    if not allowed:
        return
    if not p:
        return
    root_st, root_blocked = _safe_lstat(os.path.dirname(p))
    file_st, file_blocked = _safe_lstat(p)
    if (root_blocked or file_blocked or root_st is None or file_st is None
            or not stat.S_ISDIR(root_st.st_mode)
            or not stat.S_ISREG(file_st.st_mode)):
        return
    try:
        os.remove(p)
    except OSError:
        pass


def _run_reference_edit(app, user_id, dataset_id, token, act_token, engine, refs,
                        prompt, shared_activity=False):
    """Worker body: call the engine, write the candidate, mark the job ready — all
    in a background thread (factored out so tests can call it synchronously).

    ORDERING (load-bearing): set_ready() runs BEFORE dataset_activity.end(). The
    payload poll stops the moment activity clears, with ONE final refresh — which
    must already see the ready candidate, or the modal stays on the spinner. A
    superseded worker (a newer edit started meanwhile) gets False from
    set_ready/set_failed and deletes its own orphan candidate."""
    with app.app_context():
        try:
            # Last reversible boundary. A delayed worker whose batch was already
            # discarded/superseded must not send (and bill) a provider request.
            if not reference_edit_jobs.claim_api_dispatch(dataset_id, token):
                return
            try:
                out = _edit_engine_call(engine, refs, prompt)
            except SubscriptionQuotaExceeded as e:
                reference_edit_jobs.set_failed(dataset_id, token, str(e))
                return
            except SubscriptionUnavailable as e:
                reference_edit_jobs.set_failed(dataset_id, token, f'chatgpt: {e}')
                return
            except EngineError as e:
                # A NAMED engine failure (no key, rejected key, no credits, unknown
                # model, a model that won't take reference images). The message is
                # already user-facing and actionable, so it is surfaced verbatim —
                # loudly, with the engine named — instead of a stack trace. Matters
                # most for OpenRouter, whose model is free text in Settings: a slug
                # that can't edit must say so, not look like a refused prompt.
                reference_edit_jobs.set_failed(dataset_id, token, f'{engine}: {e}')
                return
            if out is None:
                # Un refus NOMMÉ est déjà parti par `except EngineError` au-dessus
                # (EngineRefused en hérite) : ici il ne reste que le vide muet des
                # moteurs qui ne savent pas distinguer refus et hoquet.
                reference_edit_jobs.set_failed(dataset_id, token,
                                               f'{engine}: {_EMPTY_MSG}')
                return
            cand_fn = f'{user_id}{reference_edit_jobs.CANDIDATE_MARKER}{uuid.uuid4().hex[:8]}.webp'
            cand_path = os.path.join(_dataset_dir(dataset_id), cand_fn)
            write_image_atomic(cand_path, normalize_to_webp(out))
            # set_ready BEFORE the finally end() — see ORDERING above.
            if not reference_edit_jobs.set_ready(dataset_id, token, cand_fn):
                reference_edit_jobs._unlink(cand_path)      # superseded: drop our orphan
        except Exception as e:
            logger.exception('reference edit worker failed (dataset %s)', dataset_id)
            reference_edit_jobs.set_failed(dataset_id, token, f'{engine}: {e}')
        finally:
            _finish_reference_edit_activity(
                dataset_id, token, act_token,
                shared_activity=shared_activity)


def keep_reference_edit(user_id, dataset_id, engine=None, batch_id=None):
    """Promote the READY candidate to be the reference (reuses the atomic,
    fail-safe commit_edited_reference), then delete the candidate file + clear the
    job. Returns the new ref_filename, or None when there is no ready candidate
    (route -> 409) — including a candidate file that vanished under us."""
    # The mutation lock closes the claim -> file/DB promotion TOCTOU. Without it,
    # an upload/crop could commit and invalidate after claim_ready(), only for this
    # stale candidate to overwrite and delete that newer reference before the
    # post-commit revision check noticed.
    with reference_mutation(dataset_id):
        claim = reference_edit_jobs.claim_ready(
            dataset_id, engine, batch_id=batch_id)
        if not claim:
            return None
        dsdir = _dataset_dir(dataset_id)
        try:
            with open(os.path.join(dsdir, claim['candidate_filename']), 'rb') as fh:
                data = fh.read()
        except OSError:
            reference_edit_jobs.clear_claimed(
                dataset_id, claim['batch_token'], claim['claim_token'], dsdir)
            return None
        try:
            new_ref = commit_edited_reference(user_id, dataset_id, data)
        except Exception:
            # A failed write leaves both the old master and every candidate intact,
            # so the user can retry Keep after fixing the storage problem.
            reference_edit_jobs.release_claim(
                dataset_id, claim['batch_token'], claim['claim_token'])
            raise
        cleared = reference_edit_jobs.clear_claimed(
            dataset_id, claim['batch_token'], claim['claim_token'], dsdir,
            reference_mutated=True)
        if cleared is None:
            # Defensive for TTL/process-lifecycle anomalies. Real reference
            # mutations cannot interleave here because they use this same lock.
            reference_edit_jobs.invalidate(dataset_id, dsdir)
        return new_ref


def _clear_reference_edit(dataset_id):
    """Drop the pending edit, delete its candidate, and — for a LOCAL edit still
    rendering — cancel the ComfyUI job and close its activity. Without the cancel,
    abandoning a local edit left the GPU busy on a result nobody would ever see
    and the ✦ activity badge lit until the TTL."""
    reference_edit_jobs.clear(dataset_id, _dataset_dir(dataset_id))


def discard_reference_edit(dataset_id):
    """Drop a pending edit (running=abandon OR ready) and delete its candidate
    file. An API call already sent is still billed — honesty preserved, no
    'refund' implied; a local render is cancelled, because it can be."""
    _guard_not_bank_export(dataset_id)
    _clear_reference_edit(dataset_id)


def reference_mutation(dataset_id):
    """Context manager shared by every primary-reference mutation path."""
    _guard_not_bank_export(dataset_id)
    return reference_edit_jobs.reference_mutation(dataset_id)


def invalidate_reference_edit(dataset_id):
    """Drop any pending edit candidate when the reference itself changes
    (crop/recrop/change/keep): a Before/After computed from the OLD reference would
    be a visual lie. Idempotent — a no-op when nothing is pending."""
    with reference_mutation(dataset_id):
        reference_edit_jobs.invalidate(dataset_id, _dataset_dir(dataset_id))


def commit_edited_reference(user_id, dataset_id, image_bytes):
    """Serialize and promote edited bytes as the dataset's reference."""
    with reference_mutation(dataset_id):
        return _commit_edited_reference_locked(user_id, dataset_id, image_bytes)


def _commit_edited_reference_locked(user_id, dataset_id, image_bytes):
    """Promote an edited candidate (bytes) to BE the dataset reference. The edited
    image is the new source of truth, so it becomes BOTH ref_filename (working
    crop) and ref_original_filename (the full frame ✂ Crop re-reads) — a later
    crop widens back out INSIDE the edited frame; re-cropping the pre-edit
    original would drop the edit (e.g. the glasses just added).

    ATOMIC, fail-safe order: write the two NEW files and confirm they are on disk
    BEFORE unlinking the old ones, and only repoint the DB after. A failed write
    (unusable candidate bytes, full disk) leaves the dataset on its PREVIOUS
    reference — a Keep must never strand it with no reference. Deleting the old
    files is safe because every in-flight batch snapshotted the reference at
    launch (API reads the bytes before the thread starts; Klein copies the file
    into ComfyUI's input at enqueue), so nothing running depends on them.

    Returns the new ref_filename. Raises ValueError if the dataset/reference is
    gone; propagates the write error (old reference intact) on failure."""
    ds = get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        raise ValueError('reference image required')
    dsdir = _dataset_dir(dataset_id)
    old_ref, old_orig = ds.ref_filename, ds.ref_original_filename
    new_ref = f"{user_id}_datasetref_{uuid.uuid4().hex[:8]}.webp"
    new_orig = f"{user_id}_datasetreforig_{uuid.uuid4().hex[:8]}.webp"
    ref_path = os.path.join(dsdir, new_ref)
    orig_path = os.path.join(dsdir, new_orig)
    # 1) WRITE the new files (working ref ≤1024, full-frame original ≤2048).
    #    normalize_to_webp raises on unusable bytes BEFORE any file is created, so
    #    a corrupt candidate never touches the existing reference.
    try:
        webp = normalize_to_webp(image_bytes, size=1024)
        orig_webp = normalize_to_webp(image_bytes, size=2048)
        with open(ref_path, 'wb') as fh:
            fh.write(webp)
        with open(orig_path, 'wb') as fh:
            fh.write(orig_webp)
    except Exception:
        # Roll back any partial write; the old reference is untouched.
        for p in (ref_path, orig_path):
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    # 2) VERIFY both landed before touching anything the dataset still points at.
    if not (os.path.exists(ref_path) and os.path.exists(orig_path)):
        for p in (ref_path, orig_path):
            try:
                os.remove(p)
            except OSError:
                pass
        raise RuntimeError('failed to write edited reference')
    # 3) REPOINT the dataset, then commit.
    ds.ref_filename = new_ref
    ds.ref_original_filename = new_orig
    db.session.commit()
    # 4) Only now delete the superseded files (nothing in flight depends on them).
    for fn in (old_ref, old_orig):
        if fn and fn not in (new_ref, new_orig):
            try:
                os.remove(os.path.join(dsdir, fn))
            except OSError:
                pass
    return new_ref


def _watermark_route_payload(img):
    """The routes Clean WOULD take for a 'detected' image, as a dict spread into the
    image payload:
      - 'watermark_route'        : the DEFAULT route ('crop' | 'lama' | 'review'), used
                                   by the 🚩 tooltip and the batch/lightbox planned line;
      - 'watermark_route_nocrop' : the SAME routing with auto-crop disabled ('lama' |
                                   'review') -- only ever differs when the default is
                                   'crop'. It lets the review lightbox offer a per-image
                                   crop-vs-inpaint choice (and name the inpaint fallback)
                                   without duplicating _route_watermark in JS.
    Both are None for a non-'detected' row. It needs the pixel dims (the grid doesn't
    carry them), so it opens the file ONCE -- but only for 'detected' rows (a bounded
    subset), so the single-dataset payload never reads every image header. Defensive: any
    read/parse error yields None routes and the UI falls back to the generic hint."""
    none = {'watermark_route': None, 'watermark_route_nocrop': None}
    if img.watermark_state != 'detected':
        return none
    bbox = _safe_json(img.watermark_bbox)
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return none
    try:
        with Image.open(_img_path(img)) as im:
            # The bbox comes from the browser/VLM visual frame, so route against
            # the same EXIF-oriented dimensions the user can see. This is a
            # payload/polling read: use the header-only helper, never decode the
            # whole master merely to draw a route badge.
            W, H = image_encoding.visual_size_from_header(im)
    except (OSError, ValueError):
        return none
    box = tuple(bbox)
    route, _ = _route_watermark(box, W, H)
    # Only recompute the crop-disabled route when crop is what the default picked --
    # otherwise the two are identical, so skip the redundant pure-function call.
    route_nc = route if route != 'crop' else _route_watermark(box, W, H, allow_crop=False)[0]
    return {'watermark_route': route, 'watermark_route_nocrop': route_nc}


def _image_engine(img):
    """Which engine produced this image — 'klein' | 'nanobanana' | 'chatgpt' — or
    None when it CANNOT be told.

    `klein_model` carries two different kinds of value: an engine id for the API
    rows (set by generate_variations_nanobanana) and a local .safetensors file
    name for the Klein rows. That is enough to answer honestly for both, but not
    for every legacy row: images generated before the column was populated, and
    imported photos, hold nothing. Those get None → the UI shows NO badge, which
    is the right answer. Guessing 'klein' for an empty value would label old
    Nano Banana images as local, and a wrong badge is worse than none."""
    value = (img.klein_model or '').strip()
    if not value:
        return None
    if value in API_ENGINES:
        return value
    # Krea 2 Edit rows store the engine id here, like the API ones: the engine
    # resolves its base model deterministically at enqueue AND at regenerate
    # (krea_edit_helper.resolve_krea_unet), so there is no per-row model to keep.
    if value == KREA_ENGINE:
        return KREA_ENGINE
    return 'klein'   # a local model file name — the row was rendered on the GPU


def dataset_payload(user_id, dataset_id):
    from . import lora_test_studio as studio
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return None
    imgs = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .order_by(FaceDatasetImage.id.desc()).all())
    ref_size = image_pixel_size(_ref_path(ds)) if ds.ref_filename else None
    comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    # Combien, PAR bucket, viennent d'une box bien plus petite que la résolution
    # d'entraînement (upscale_ratio >= UPSCALE_WARN_THRESHOLD) plutôt que d'une prise
    # native : le compte `comp` seul traite un gros plan natif et un gros plan
    # recadré x3 comme équivalents vis-à-vis de la cible — ce sous-compte permet à
    # l'UI de signaler un dataset qui « remplit » sa cible face/bust surtout en
    # recadrant (texture agrandie à l'import, ou tuile sous-résolue en manuel).
    comp_upscaled = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    for i in imgs:
        # Composition counts only usable images: rejected and failed ones don't
        # contribute to the training-target tally the UI tracks deficits against.
        if i.framing in comp and i.status not in ('reject', 'failed'):
            comp[i.framing] += 1
            if (i.upscale_ratio or 0) >= UPSCALE_WARN_THRESHOLD:
                comp_upscaled[i.framing] += 1
    # concept OU style : le champ `fidelity`/`concept_desc` du payload est gouverné par
    # is_conceptual (character-only). La DÉTECTION de fuite, elle, est spécifique au KIND :
    #   - character : fuite d'IDENTITÉ (hair/skin/eyes)  → caption_has_identity_leak
    #   - concept   : fuite de CONCEPT (le set nomme le concept au lieu du trigger) →
    #                 caption_has_concept_leak — on ne force PLUS 0 (le badge « 0 leak »
    #                 faussement rassurant de l'incident leg_behind)
    #   - style     : rien (la description des sujets EST le contenu contrôlable) → 0 honnête
    concept = is_conceptual(ds)
    kind_concept = is_concept(ds)
    kind_style = is_style(ds)
    body = is_body_fidelity(ds)
    # Cached concept ban-list (JSON on the row) → the concept-leak detector unions it with
    # concept_desc + the derived body/pose field, so the badge and the caption-time
    # enforcement agree on what "leaking" means. Ignored for non-concept kinds.
    _concept_terms = ds.concept_terms if kind_concept else None

    def _img_leaks(i):
        if i.status != 'keep' or not i.caption:
            return False
        if kind_concept:
            return caption_has_concept_leak(i.caption, ds.concept_desc, _concept_terms)
        if kind_style:
            return False
        return caption_has_identity_leak(i.caption, body=body)

    return {
        'id': ds.id, 'name': ds.name, 'trigger_word': ds.trigger_word,
        'train_type': (ds.train_type or 'zimage'),
        'kind': (ds.kind or 'character'),
        # WHAT the subject is (NULL/legacy -> 'human'); drives the generation
        # catalog + identity lock. Orthogonal to `kind`.
        'subject_type': subject_type_of(ds),
        # Why face-similarity scoring is refused for this dataset (string), or null
        # to go ahead. Published so the UI disables the button and states the reason
        # from the SAME rule the server enforces, instead of re-implementing
        # "subject_type === 'anime'" in JSX and drifting from it later.
        'face_scoring_blocked': face_scoring_block_reason(ds),
        # How much work 🎭 Analyze faces actually has: {total, unscored} over the
        # kept set PLUS the undecided triage pile (FACE_SCORING_STATUSES). Lets the
        # button name its scope instead of running a mystery pass.
        'face_scoring_scope': face_scoring_counts(imgs),
        # Dual long+short captioning toggle (Advanced options) → the caption editor shows
        # the short field only when this is on.
        'dual_captions': dual_captions_enabled(ds),
        # Concept face masking (Advanced options) + whether this concept's own
        # description names the face/mouth/gaze. The second one drives a WARNING,
        # not a block: only the user knows whether the face carries their concept.
        'mask_faces': face_masking_enabled(ds),
        'concept_face_conflict': concept_face_conflict(ds),
        'fidelity': (ds.fidelity or 'face') if not concept else 'face',
        'concept_desc': (ds.concept_desc or '') if concept else '',
        # Creative-direction suffixes (global + per-framing) → settings modal
        # prefill. Applied at wrap time; never part of the stored per-image prompt.
        'prompt_suffix': ds.prompt_suffix or '',
        'prompt_suffixes': prompt_suffixes_dict(ds),
        # Where this dataset's images actually live. It was displayed NOWHERE,
        # which is how people ended up hunting for it in the file manager and
        # pasting it into "create a bank" — a bank over a dataset's live files,
        # whose 🗑 Delete rejected then deleted images out of the dataset. Showing
        # the path (with the sentence that it belongs to the dataset) removes the
        # reason to go looking; `services.path_guard` refuses the paste anyway.
        'storage_path': _dataset_path(ds.id),
        'ref_filename': ds.ref_filename,
        # Pixel size of the ACTIVE reference (the cropped one — that is the file
        # every engine is handed). Kept for crop-aware clients; Krea dataset cards
        # now use the selected card's target frame. None when unmeasurable.
        'ref_width': (ref_size or (None, None))[0],
        'ref_height': (ref_size or (None, None))[1],
        'ref_original_filename': ds.ref_original_filename or '',
        'ref_extra_filenames': extra_ref_filenames(ds),
        # Per extra ref, the file its ✂ editor must open (full-frame original when
        # kept, else the extra itself) — aligned index-by-index with the list above.
        'ref_extra_crop_sources': [extra_ref_crop_source(ds, fn)
                                   for fn in extra_ref_filenames(ds)],
        'composition': comp,
        'composition_upscaled': comp_upscaled,
        # Réglages gagnants du Studio (JSON → objet). Manquait du payload : le badge
        # ★ du workspace ne s'affichait jamais, et le garde-fou « suppression d'un
        # checkpoint référencé » en a besoin.
        'best_settings': _safe_json(ds.best_settings),
        # The pinned LoRA filenames, FLATTENED out of the per-family map above.
        # The delete guard-rail used to read `best_settings.lora_filename`, a key
        # that only exists in the legacy flat shape — so on any dataset pinned
        # since best settings went per-family the ⚠ warning was silently dead.
        'best_settings_loras': studio.best_settings_lora_filenames(ds),
        'face_thresholds': {'green': cfg.get('face_scoring.green'), 'orange': cfg.get('face_scoring.orange')},
        'images': [{'id': i.id, 'filename': i.filename, 'source': i.source,
                    # Which engine made this image, when it can be told HONESTLY
                    # (see _image_engine) — the tile badge that makes a
                    # multi-engine run comparable. None = no badge.
                    'engine': _image_engine(i),
                    'framing': i.framing, 'variation_label': i.variation_label,
                    'status': i.status, 'caption': i.caption,
                    'caption_short': i.caption_short,
                    # WHO wrote each of the two texts ('asserted' | 'joycaption' |
                    # 'ollama' | NULL = never recorded — services/caption_origin.py).
                    # It travels WITH the sentence, per image and per field: the
                    # default 'auto' backend chains JoyCaption then Ollama inside a
                    # single run, and the short caption has its own writers, so
                    # neither the settings value nor the long caption's stamp can
                    # answer "who wrote the words I am reading".
                    'caption_origin': i.caption_origin,
                    'caption_short_origin': i.caption_short_origin,
                    'fail_reason': i.fail_reason,
                    # 'refused' | 'empty' | 'error' | None — de quelle NATURE est
                    # l'échec, pour que l'UI puisse compter les refus fournisseur
                    # sans relire la phrase (cf. models.FaceDatasetImage).
                    'fail_kind': i.fail_kind,
                    'parent_image_id': i.parent_image_id,
                    'derivation_kind': i.derivation_kind,
                    'source_metadata': normalize_source_metadata(i.source_metadata),
                    'upscale_ratio': i.upscale_ratio,
                    # Core creative prompt (generated tiles) → seeds the ✏️ edit
                    # bubble so the user edits the real prompt, not a blank box.
                    'variation_prompt': i.variation_prompt,
                    # Per-image leak flag (identity for character, concept for concept,
                    # never for style): lets the UI LIST the offending captions for quick
                    # manual treatment (the aggregate badge alone forced a grid hunt).
                    'leak': _img_leaks(i),
                    'face_score': i.face_score, 'face_state': i.face_state,
                    # Watermark V1: state drives the tile badge (🚩 detected / ⊘ dismissed
                    # / ✨ cleaned / ⚠ failed) and the "Clean (N)" count; bbox lets the UI
                    # draw the detected box (review lightbox); watermark_route(_nocrop)
                    # name the planned action ('crop'|'lama'|'review') with auto-crop on
                    # and off, so the lightbox can offer a per-image crop-vs-inpaint choice.
                    'watermark_state': i.watermark_state,
                    'watermark_bbox': _safe_json(i.watermark_bbox),
                    # WHICH detector ruled ('detector' | 'vision' | None = before
                    # the column existed). The two routes disagree at the margins
                    # and only one of them can flag an image WITHOUT a position,
                    # so the screen has to be able to say who judged what instead
                    # of presenting one pile as if a single instrument made it.
                    'watermark_source': i.watermark_source,
                    'watermark_score': i.watermark_score,
                    **_watermark_regions_payload(i),
                    **_watermark_route_payload(i)} for i in imgs],
        # Kind-specific leak count (see _img_leaks): character = identity, concept = the
        # caption naming the concept (NEVER forced 0 any more), style = 0 (not applicable).
        # `captioned` bounds the badge ("N leaking / M checked") so a 0 reads as a real
        # result on M captions, not a check that never ran.
        'caption_leak': {
            'leaking': sum(1 for i in imgs if _img_leaks(i)),
            'captioned': sum(1 for i in imgs if i.status == 'keep' and i.caption),
        },
        # Live server-side batch on this dataset (watermark detect/clean, caption/
        # re-caption, face analysis, framing classify) as {kind, done, total,
        # started_at} — or None. The front-end RESTORES the in-progress button state
        # from this on reload and polls the payload until it clears (the indicator was
        # React-local before, so a refresh mid-batch dropped it). In-memory registry:
        # empty after a server restart, so a batch killed with the process leaves no
        # phantom indicator.
        'activity': dataset_activity.get(dataset_id),
        # Pending reference EDIT (server background job) as {status, engine, prompt,
        # candidate_filename, error, started_at} — or None. The modal RESTORES its
        # Before/After from this after a tab sleep or reload; the 'edit_reference'
        # activity above keeps this polled while it runs. get() lazily purges an
        # abandoned candidate past its TTL, so this can't strand a stale file.
        # The dataset dir is passed so a candidate that LANDED before a restart
        # is found again and re-offered: it is a paid provider result, and losing
        # the registry entry used to leave it unreachable until the sweep deleted
        # it. Recovery reads sidecars, never filenames, and only ever runs when
        # the registry has nothing live for this dataset.
        'reference_edit': reference_edit_jobs.get(dataset_id,
                                                  _dataset_dir(dataset_id)),
    }


# --- Image normalization ---------------------------------------------------
def write_image_atomic(path, data: bytes) -> None:
    """Publish an image file in one step: it is either absent or COMPLETE.

    `open(path, 'wb')` truncates immediately, and the bytes usually arrive a
    second or two later (a WEBP re-encode of a 1024px generation is not free).
    Under its FINAL name that leaves an empty file on disk for the whole
    encode, and the grid polls the dataset while a batch runs: the browser
    asks for it, the server answers 200 with zero bytes, and the tile renders
    black. Reported after an OpenRouter generation, but nothing about it was
    engine-specific — every generated image had the same window.

    Writing beside the target and renaming closes it: os.replace is atomic on
    the same filesystem, so a reader sees the old state or the new one, never
    a half-written one. A missing file is already handled everywhere (the tile
    shows its pending state), which is the honest answer while it is encoding.
    """
    tmp = f'{path}.part'
    try:
        with open(tmp, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)          # never leave a .part behind on failure
        except OSError:
            pass
        raise


def normalize_to_webp(image_bytes: bytes, size: int = 1024,
                      quality: int = 92, lossless: bool = False) -> bytes:
    """Resize so the longest side ≤ `size`, KEEP the aspect ratio (no square pad),
    return WEBP. Pour les variations Nano Banana : un plan corps reste en portrait
    (pas de bandes noires que le LoRA apprendrait). ai-toolkit gère le bucketing.

    `size=0` means "do not resample at all" — the ceiling below still applies
    because it is a FORMAT limit, not a taste: Pillow refuses to write a WebP
    past 16383 px ("Image size exceeds WebP limit of 16383 pixels"), so an
    uncapped call would turn a big panorama into a failed import.

    DERIVATIVE ON PURPOSE — this is INGEST/TRANSPORT (the ≤2048 px copy handed to a
    generation API, and the normalisation of freshly generated bytes), not an edit of
    an image the user already curated. It must NOT be routed through `image_encoding`:
    inflating an upload 4x to protect pixels the remote engine will re-encode anyway
    buys nothing. See the module docstring of `image_encoding` for the split."""
    # A normalised WebP has no reason to retain a camera orientation tag. The
    # shared loader validates header geometry before any full decode, then bakes
    # the visible orientation into its temporary pixels.
    im = _load_import_derivative_image(image_bytes).convert('RGB')
    limit = min(size, IMPORT_MAX_SIDE_CEILING) if size else IMPORT_MAX_SIDE_CEILING
    im.thumbnail((limit, limit), Image.LANCZOS)   # only ever shrinks
    out = io.BytesIO()
    if lossless:
        im.save(out, 'WEBP', lossless=True)
    else:
        im.save(out, 'WEBP', quality=quality)
    return out.getvalue()


# --- Import resolution & encoding (Settings ▸ Captioning & quality) ----------
# Hard ceiling on a *normalised WebP* dataset image, in px. MEASURED, not chosen:
# Pillow's WebP encoder raises "Image size exceeds WebP limit of 16383 pixels" past
# that side. It applies only to the opt-in normalisation modes; preserving a source
# file never re-encodes it just to meet a WebP implementation limit.
# NOT the input budget: this one bounds what a WebP normalisation mode WRITES,
# and it stays put when the user raises the ingress budget. Conflating the two
# would let "accept a 24 000 px panorama" silently become "write a 24 000 px
# WebP derivative", which the encoder cannot do anyway.
IMPORT_MAX_SIDE_CEILING = 8192
_IMPORT_ENCODINGS = {                       # label -> storage policy
    'preserve': {'preserve': True, 'quality': None, 'lossless': False},
    'standard': {'preserve': False, 'quality': 92, 'lossless': False},
    'high': {'preserve': False, 'quality': 100, 'lossless': False},
    'lossless': {'preserve': False, 'quality': 100, 'lossless': True},
}

# Only static formats both the dataset tools and the trainer's disposable PNG
# staging pass can read. The extension comes from decoded CONTENT, never from the
# upload name: an `image.jpg` carrying PNG bytes must not leave a lying filename.
_PRESERVED_IMPORT_EXTENSIONS = {
    'JPEG': '.jpg',
    'PNG': '.png',
    'WEBP': '.webp',
    'BMP': '.bmp',
}

# A raw master is intentionally NOT resized on import, but importing it must not
# turn the process into an unbounded decompressor. These limits apply uniformly to
# every image ingress path (preserve, crop, explicit normalisation, ZIP and scrape)
# and are checked from Pillow's header before ``load()``. They are a SETTING now
# (`image_input.*`, default 64 Mi-pixels / 16384 px per side, 0 = no limit), so
# they are read through a function: a module-level snapshot taken at import would
# freeze the first value the process ever saw. The budget is a memory guard, not
# an encoder limit; an accepted preserved image remains byte-for-byte untouched.


def preserved_import_limits() -> tuple[int, int]:
    """The effective (max_side, max_pixels) ingress budget; 0 = no limit."""
    return image_encoding.input_budget()


def import_encode_policy() -> dict:
    """What an imported image will ACTUALLY be stored as, resolved once so the
    UI, the toast and the encoder all quote the same policy.

    Total by construction: an unusable configured value logs and degrades to the
    shipped default rather than breaking every import. `capped` is True when a
    WebP-normalisation mode asked for more than that format allows. In `preserve`
    mode the value is retained for a future explicit normalisation choice, but has
    no effect on the stored source bytes."""
    defaults = cfg.DEFAULTS['dataset_import']
    raw_side = cfg.get('dataset_import.max_side', defaults['max_side'])
    try:
        max_side = int(raw_side)
        if max_side < 0:
            raise ValueError(raw_side)
    except (TypeError, ValueError):
        logger.warning('ignoring unusable dataset_import.max_side %r', raw_side)
        max_side = int(defaults['max_side'])
    capped = max_side > IMPORT_MAX_SIDE_CEILING
    if capped:
        max_side = IMPORT_MAX_SIDE_CEILING
    encoding = str(cfg.get('dataset_import.encoding', defaults['encoding']) or '')
    if encoding not in _IMPORT_ENCODINGS:
        if encoding:
            logger.warning('ignoring unusable dataset_import.encoding %r', encoding)
        encoding = defaults['encoding']
    policy = _IMPORT_ENCODINGS[encoding]
    input_max_side, input_max_pixels = preserved_import_limits()
    # A preserved image is never sent through a WebP encoder, so a WebP ceiling
    # cannot cap it. Keep the resolved max_side in the payload so switching
    # back to a normalising mode remains predictable.
    return {'max_side': max_side, 'encoding': encoding,
            'capped': capped and not policy['preserve'],
            'ceiling': IMPORT_MAX_SIDE_CEILING,
            # Explicit names for the ingress safety budget. The older
            # `preserve_*` aliases stay below for clients released while this
            # policy only described raw-preserve imports.
            'input_max_side': input_max_side,
            'input_max_pixels': input_max_pixels,
            'preserve_max_side': input_max_side,
            'preserve_max_pixels': input_max_pixels,
            **policy}


def _validate_import_header_dimensions(im: Image.Image, *, label: str) -> None:
    """Reject an unsafe raster header before any caller asks Pillow to decode it."""
    image_encoding.validate_input_header_dimensions(im, label=label)


def _import_header_dimensions(image_bytes: bytes, *, label: str = 'import') -> tuple[int, int]:
    """Read bounded image dimensions without ever asking Pillow for pixel data.

    This is the common ingress check for the small-image warning, scrape sorting,
    crop and explicit normalisation paths.  It is deliberately separate from the
    preserve validator: those paths additionally require a static, supported
    container, while this helper is only about stopping an unsafe raster header
    before any caller decides to call ``load()``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as im:
                _validate_import_header_dimensions(im, label=label)
                return im.size
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f'{label} rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, MemoryError) as exc:
        raise ValueError(f'{label} received an unreadable image') from exc


def _preserved_import_header_extension(im: Image.Image, *, label: str = 'preserve mode') -> str:
    """Validate a raw static source header and return its content extension.

    This is deliberately called before ``im.load()``.  A file can advertise a
    huge raster in a very small compressed payload, so dimensions must be
    rejected before the decoder allocates its full pixel buffer.
    """
    fmt = (getattr(im, 'format', None) or '').upper()
    ext = _PRESERVED_IMPORT_EXTENSIONS.get(fmt)
    if ext is None:
        raise ValueError(
            f'{label} supports only static JPEG, PNG, WebP, or BMP images '
            f'(got {fmt or "unknown"})')
    if getattr(im, 'n_frames', 1) != 1:
        raise ValueError(
            f'{label} supports only static JPEG, PNG, WebP, or BMP images '
            '(animated images are not supported)')
    _validate_import_header_dimensions(im, label=label)
    return ext


def _preserved_import_extension(image_bytes: bytes, *, label: str = 'preserve mode') -> str:
    """Validate a raw static source and return its canonical content extension.

    Preserving bytes must not mean accepting arbitrary browser/media formats the
    rest of the dataset pipeline cannot safely edit, serve or train from. GIF,
    TIFF, AVIF and animated WebP are deliberately refused here instead of being
    silently flattened or saved under a made-up extension.  Header dimensions
    are checked before ``load()`` under a local Pillow bomb-warning policy; no
    process-global warning filter is ever changed.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as im:
                ext = _preserved_import_header_extension(im, label=label)
                # Fully decode only after the explicit header budget passed. PIL can
                # identify a truncated file from the header alone; `load` enforces the
                # same readability guarantee the old normalisation path provided.
                im.load()
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f'{label} rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, MemoryError) as exc:
        raise ValueError(f'{label} received an unreadable image') from exc
    return ext


def _load_import_derivative_image(image_bytes: bytes) -> Image.Image:
    """Decode a bounded, visually upright temporary image for derived imports.

    Preserve mode calls its stricter static-format validator above.  Crop and
    explicit WebP-normalisation use this shared geometry guard so they cannot
    decode a compressed bomb merely because they are allowed to derive pixels.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as opened:
                _validate_import_header_dimensions(opened, label='import')
                opened.load()
                return ImageOps.exif_transpose(opened).copy()
    except ValueError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('import rejected an unsafe image header') from exc
    except (OSError, UnidentifiedImageError, MemoryError) as exc:
        raise ValueError('import received an unreadable image') from exc


def import_store_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Return exactly what a non-cropped import should store and its extension.

    `preserve` keeps approved source bytes byte-for-byte. The three legacy
    encoding modes intentionally still create WebP derivatives, preserving their
    historical resizing and quality controls for users who explicitly choose them.
    """
    p = import_encode_policy()
    if p['preserve']:
        return image_bytes, _preserved_import_extension(image_bytes)
    return (normalize_to_webp(image_bytes, size=p['max_side'],
                              quality=p['quality'], lossless=p['lossless']),
            '.webp')


def import_encode(image_bytes: bytes) -> bytes:
    """Backward-compatible bytes-only view of :func:`import_store_image`.

    New ingest lanes need the true extension as well and use
    :func:`import_store_image` directly. Generated images and API transport
    copies deliberately keep their own fixed sizes: this policy is about what the
    user hands in, not about what the app produces.
    """
    return import_store_image(image_bytes)[0]


def detect_head_bbox(image_bytes):
    """Return normalized (x1, y1, x2, y2) of the main head via Qwen3-VL, or None.

    None also covers Ollama being unreachable/misconfigured (describe_image_ollama
    never raises) -- the caller (face_crop_to_square_webp) already treats "no
    detection" as a normal case and falls back to a centered crop, so uploads
    keep working (degraded but functional)."""
    try:
        from .vision_ollama import describe_image_ollama
    except ImportError:
        return None
    # fmt='json' forces Ollama's grammar mode: the model must emit a JSON object from
    # the first token, so reasoning-prone (abliterated) checkpoints can't ramble a
    # <think> trace past num_predict and never reach the coords (a silent-None cause).
    #
    # keep_alive is decided by CONTENTION, not by this call site (see
    # services/vision_keepalive.py). This is the burst case the policy exists for:
    # cropping five references in a row used to pay the 12.8 s cold load five times
    # because each upload is its own isolated call. When the card is contended — or
    # when the signal can't be read — the policy returns 0 and nothing changes.
    from .vision_keepalive import keep_alive_for_isolated_call
    raw = describe_image_ollama(image_bytes, HEAD_BBOX_PROMPT, num_predict=400,
                                prefer_json=True, fmt='json',
                                keep_alive=keep_alive_for_isolated_call())
    try:
        s = raw.index('{')
        obj = json.loads(raw[s:raw.index('}', s) + 1])
        y1, x1, y2, x2 = (float(obj[k]) for k in ('y1', 'x1', 'y2', 'x2'))
    except (ValueError, KeyError, AttributeError, TypeError):
        return None
    # Qwen3-VL frequently SWAPS corners (returns y1>y2 or x1>x2). Normalize to
    # min/max instead of rejecting — rejecting was a silent-None cause that fell back
    # to a body-centered crop even when the head was correctly located.
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)


# Marge d'elargissement de la bbox watermark (fraction du cote). Les bbox VLM sont
# GROSSIERES et souvent trop serrees : sans marge, le crop/inpaint laisse un lisere du
# watermark. 2.5% de chaque cote = filet de securite sans engloutir le sujet.
_WATERMARK_BBOX_MARGIN = 0.025


def _parse_watermark_bbox(raw):
    """PURE parser for a WATERMARK_BBOX_PROMPT answer. Returns a MARGIN-EXPANDED
    normalized (x1,y1,x2,y2) in [0,1], or None (no watermark / unparseable). Split out
    from the vision call so the batch can tell an EMPTY vision output (Ollama down ->
    leave the state untouched) apart from a clean 'present:false' answer (-> 'none').

    Same bbox handling as detect_head_bbox: 0-1000 grid, swapped corners normalized to
    min/max. A `present:false` (or a missing/invalid box) -> None. VLM boxes run tight,
    so we pad by _WATERMARK_BBOX_MARGIN and clamp -- the router needs the whole mark."""
    try:
        s = raw.index('{')
        obj = json.loads(raw[s:raw.index('}', s) + 1])
    except (ValueError, AttributeError, TypeError):
        return None
    if 'present' in obj and not obj.get('present'):
        return None
    try:
        y1, x1, y2, x2 = (float(obj[k]) for k in ('y1', 'x1', 'y2', 'x2'))
    except (KeyError, TypeError, ValueError):
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    m = _WATERMARK_BBOX_MARGIN
    return (max(0.0, x1 / 1000.0 - m), max(0.0, y1 / 1000.0 - m),
            min(1.0, x2 / 1000.0 + m), min(1.0, y2 / 1000.0 + m))


def detect_watermark_bbox(image_bytes, *, keep_alive=0):
    """Return normalized (x1, y1, x2, y2) of an OVERLAID watermark via Qwen3-VL, or
    None (no overlaid watermark, or the model is unreachable / the JSON won't parse).
    fmt='json' forces Ollama's grammar mode, same as detect_head_bbox.

    The prompt targets watermark/logo/URL/username text ADDED ON TOP of the photo, NOT
    scene text (signs, clothing prints) -- see WATERMARK_BBOX_PROMPT. Box is margin-
    expanded (see _parse_watermark_bbox). `keep_alive` mirrors describe_image_ollama:
    0 unloads after this call; a batch passes a duration and unloads at the end."""
    try:
        from .vision_ollama import describe_image_ollama
    except ImportError:
        return None
    raw = describe_image_ollama(image_bytes, WATERMARK_BBOX_PROMPT, num_predict=400,
                                prefer_json=True, fmt='json', keep_alive=keep_alive)
    return _parse_watermark_bbox(raw)


def face_crop_to_square_webp(image_bytes: bytes, size: int = 1024, pad: float = 1.7,
                             *, return_detected: bool = False, use_vision: bool = True,
                             return_scale: bool = False):
    """Head-crop (Qwen3-VL bbox, generous padding for hair + shoulders) into a
    SQUARE that FILLS `size` - no black padding, no distortion (the square is
    shrunk to fit inside the image so it never needs letterboxing). Falls back to
    a centered-square crop if no head is detected. CALLER holds the GPU window.

    `return_detected=True` -> (webp_bytes, head_detected) so the caller can WARN the
    user when it silently fell back to a centered crop (e.g. vision model not pulled)
    instead of leaving them puzzled by a body-centered reference.

    `return_scale=True` -> also returns the upscale ratio applied to reach `size`
    (>1 means the detected/fallback box was smaller than `size` and got LANCZOS-
    enlarged — see UPSCALE_WARN_THRESHOLD). Additive and independent from
    `return_detected` so existing 2-tuple callers (the /ref route) are unaffected.

    `use_vision=False` -> skip the bbox detection entirely (fast pure-PIL centered
    square, no GPU window needed) — the manual-first reference flow.

    INGEST, not an edit: this runs once on the bytes being IMPORTED, and its name is
    part of its contract (callers write the result to a `.webp`). Re-cropping that
    reference afterwards goes through `_crop_resize_file`, which does preserve the
    format losslessly."""
    # The VLM sees an upright transport derivative. Work in that same visual
    # coordinate space so its normalized head box lands on the visible subject.
    im = _load_import_derivative_image(image_bytes).convert('RGB')
    W, H = im.size
    norm = detect_head_bbox(image_bytes) if use_vision else None
    half = 0
    if norm:
        x1, y1, x2, y2 = norm[0] * W, norm[1] * H, norm[2] * W, norm[3] * H
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2 - (y2 - y1) * 0.10  # shift up to keep the hair
        half = max(x2 - x1, y2 - y1) * pad / 2
        half = min(half, cx, W - cx, cy, H - cy)  # keep the square inside the image
    head_detected = half >= 8
    if head_detected:
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    else:  # no/failed detection → centered largest square
        side = min(W, H)
        left, top = (W - side) // 2, (H - side) // 2
        box = (left, top, left + side, top + side)
    box_side = max(1, box[2] - box[0])
    scale = size / box_side
    out = io.BytesIO()
    im.crop(box).resize((size, size), Image.LANCZOS).save(out, 'WEBP', quality=92)
    webp = out.getvalue()
    if return_detected and return_scale:
        return webp, head_detected, scale
    if return_detected:
        return webp, head_detected
    if return_scale:
        return webp, scale
    return webp


# --- Import + classify (Qwen3-VL) ------------------------------------------
@_serialize_dataset_ingest
def import_images(user_id, dataset_id, files_bytes, crop=False, dedupe=False, stats=None,
                  source_metadata=None, captions=None, caption_origins=None,
                  bank_image_ids=None,
                  framings=None, bank_analysis_snapshots=None,
                  watermark_states=None, watermark_bboxes=None,
                  watermark_regions=None, watermark_sources=None,
                  watermark_scores=None, statuses=None,
                  transfer_metadatas=None, dedupe_seen=None,
                  preserve_exact_bytes=False, created_ids_sink=None,
                  provenance_changes_sink=None):
    """Store original static bytes (or head-crop) + create import rows (status=keep).
    When crop=True, each image is auto head-cropped via Qwen3-VL - the CALLER
    must then hold the GPU-exclusive window - and is by construction a face,
    so framing='face' is set directly (no classify pass needed).

    dedupe=True (the /import route) drops perceptual duplicates by dHash — both
    within the batch and vs the dataset's existing files. The hash is computed on
    the final stored image, so a re-import of the same photo matches its earlier
    crop instead of comparing a full frame to a head crop. Skips are counted in
    stats['duplicates'] when a stats dict is passed.
    Default stays False: service-level callers (scrape flow dedupes upstream on
    the ORIGINALS, before paying the crop) keep the historical behavior.

    ``source_metadata`` is an optional list parallel to ``files_bytes``. Only
    validated Pexels or web-search provenance is stored; existing callers can omit it.

    ``captions`` is an optional list parallel to ``files_bytes`` — a pre-existing
    caption to carry onto the new row (the image-bank promotion path passes the bank
    captions here, so a promoted selection starts already captioned). Empty/None entries
    leave the row uncaptioned. A skipped duplicate simply drops its caption with it.

    ``framings`` is an optional list parallel to ``files_bytes`` — a framing
    ALREADY known for the blob (the image-bank promotion path passes the framing
    its own classify pass wrote, so a promoted selection lands counted in the
    composition instead of sitting at 0 until something re-classifies it). Only
    the catalog buckets are accepted; anything else lands as None so the dataset
    classifier can still fill it. Ignored when crop=True (a head crop IS a face).

    ``bank_image_ids`` is an optional list parallel to ``files_bytes`` — the
    bank_image each blob came from, recorded on the new row. A blob dropped as a
    perceptual DUPLICATE hands its bank id to the row it matched (when that row
    carries none yet): the dataset does hold that bank image, just under another
    row, and the bank's "already promoted here" answer must say so. That link is
    what lets the bank re-offer an image once the user deletes it here. Bank ids
    that could NOT be linked (the matched row already belongs to another bank —
    a scalar column can only credit one) are listed in ``stats['bank_unlinked']``.

    ``bank_analysis_snapshots`` is an internal Bank-promotion marker parallel to
    ``files_bytes``.  When present, this importer recalculates deterministic
    quality/provenance from the final Dataset bytes and seals a v3 snapshot with
    their SHA-256.  A byte-identical Bank capture also retains its complete row
    analysis plus path-free Score/Face embeddings in a bounded sidecar; a
    transformed image gets deterministic analysis only.  The regular current
    Dataset fields stay separate and remain user-owned.

    ``dedupe_seen`` is an optional internal mutable cache of ``(dhash, row_id)``
    pairs for chunked imports. When omitted, the importer loads the dataset's
    existing hashes itself, preserving the standalone-call behavior.

    Returns (ids, failed_count)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return [], 0
    # Sans head-crop, on préserve le ratio ET les octets source autorisés : l'ancien
    # chemin « carré padé » ajoutait des bandes noires que le LoRA apprendrait, et
    # forçait tous les imports personnage en carré — un plan buste/corps importé
    # doit rester tel quel (ai-toolkit gère le bucketing multi-ratios).
    seen = (dedupe_seen if dedupe_seen is not None
            else _existing_dhash_rows(dataset_id)) if dedupe else None
    metadata_by_index = list(source_metadata) if source_metadata is not None else []
    captions_by_index = list(captions) if captions is not None else []
    # Parallel to ``captions`` and travelling WITH it. Without this list a bank
    # caption a human wrote or corrected arrives in the dataset stamped as
    # nothing, i.e. re-writable — the protection would survive exactly one hop.
    caption_origins_by_index = (list(caption_origins)
                                if caption_origins is not None else [])
    bank_ids_by_index = list(bank_image_ids) if bank_image_ids is not None else []
    framings_by_index = list(framings) if framings is not None else []
    snapshots_by_index = (list(bank_analysis_snapshots)
                          if bank_analysis_snapshots is not None else [])
    watermark_states_by_index = list(watermark_states) if watermark_states is not None else []
    watermark_bboxes_by_index = list(watermark_bboxes) if watermark_bboxes is not None else []
    watermark_regions_by_index = list(watermark_regions) if watermark_regions is not None else []
    watermark_sources_by_index = list(watermark_sources) if watermark_sources is not None else []
    watermark_scores_by_index = list(watermark_scores) if watermark_scores is not None else []
    statuses_by_index = list(statuses) if statuses is not None else []
    transfer_metadata_by_index = (list(transfer_metadatas)
                                  if transfer_metadatas is not None else [])

    def bank_id_at(i):
        return bank_ids_by_index[i] if i < len(bank_ids_by_index) else None

    def caption_origin_at(i, cap):
        """The stamp that rides with this caption — validated, never trusted raw.

        An unknown token would be stored and then compared against 'asserted'
        forever without ever matching, which is a protection that silently is
        not one. A caption with no stamp stays NULL: "never recorded".
        """
        if not (cap or '').strip():
            return None
        value = (caption_origins_by_index[i]
                 if i < len(caption_origins_by_index) else None)
        return value if value in caption_origin.VALUES else None

    def framing_at(i):
        # A head crop IS a face by construction; otherwise take the caller's value
        # when it is one of the composition buckets (an 'unknown'/None verdict must
        # stay NULL so the dataset classifier can still pick the row up).
        if crop:
            return 'face'
        fr = framings_by_index[i] if i < len(framings_by_index) else None
        return fr if fr in ('face', 'bust', 'body', 'back') else None

    def snapshot_at(i):
        return snapshots_by_index[i] if i < len(snapshots_by_index) else None

    def watermark_state_at(i):
        state = (watermark_states_by_index[i]
                 if i < len(watermark_states_by_index) else None)
        return state if state in ('none', 'detected', 'dismissed', 'cleaned', 'failed', 'error') else None

    def watermark_bbox_at(i):
        value = (watermark_bboxes_by_index[i]
                 if i < len(watermark_bboxes_by_index) else None)
        return value if isinstance(value, str) else None

    def watermark_regions_at(i):
        value = (watermark_regions_by_index[i]
                 if i < len(watermark_regions_by_index) else None)
        return value if isinstance(value, str) else None

    def watermark_source_at(i):
        value = (watermark_sources_by_index[i]
                 if i < len(watermark_sources_by_index) else None)
        return value if value in ('detector', 'vision') else None

    def watermark_score_at(i):
        value = (watermark_scores_by_index[i]
                 if i < len(watermark_scores_by_index) else None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None

    def status_at(i):
        value = statuses_by_index[i] if i < len(statuses_by_index) else None
        return value if value in ('pending', 'keep', 'reject') else 'keep'

    def transfer_metadata_at(i):
        value = (transfer_metadata_by_index[i]
                 if i < len(transfer_metadata_by_index) else None)
        if value is None:
            return None
        normalized = (
            bank_transfer_metadata.normalized_transfer_metadata_storage(value))
        if normalized is None:
            raise RuntimeError('invalid Bank/Dataset transfer metadata')
        return normalized

    ids = []
    failed = 0
    for index, raw in enumerate(files_bytes):
        # Garde-fou qualité : ai-toolkit ne fait que RÉDUIRE — une image sous
        # 768 px de petit côté reste floue à l'entraînement. Comptée (toast),
        # jamais bloquée : c'est parfois la seule photo disponible.
        if stats is not None:
            try:
                if min(_import_header_dimensions(raw)) < SCRAPE_IMPORT_MIN_SIDE:
                    stats['small'] = stats.get('small', 0) + 1
            except Exception:
                pass
        try:
            if preserve_exact_bytes:
                if crop:
                    raise ValueError('exact-byte Bank promotion cannot crop')
                # This path is a preservation operation, independent of the
                # user's normal Dataset import encoding preference.  Validate
                # the static supported format, then store the identical bytes.
                extension = _preserved_import_extension(
                    raw, label='bank dataset promotion')
                stored, scale = bytes(raw), None
            elif crop:
                stored, scale = face_crop_to_square_webp(raw, return_scale=True)
                extension = '.webp'
            else:
                stored, extension = import_store_image(raw)
                scale = None
        except Exception as e:
            if preserve_exact_bytes:
                raise RuntimeError(
                    f'could not preserve Bank image bytes: {e}') from e
            failed += 1
            logger.warning(f"dataset import: image skipped (dataset {dataset_id}): {e}")
            continue
        final_analysis = None
        captured_analysis = None
        captured_cache_bundle = None
        if snapshot_at(index) is not None:
            final_analysis = bank_deterministic_analysis(stored)
            if final_analysis is None:
                raise RuntimeError('could not seal Bank analysis for Dataset image')
            captured_analysis = (
                snapshot_at(index) if isinstance(snapshot_at(index), dict) else None)
            captured_matches = (
                captured_analysis is not None
                and captured_analysis.get('fingerprint')
                == bank_transfer_metadata.content_fingerprint_bytes(stored))
            if captured_matches and captured_analysis.get('caches'):
                captured_cache_bundle = captured_analysis['caches']

        def seal_analysis_snapshot():
            """Persist the sidecar only when this candidate is about to commit."""
            if final_analysis is None:
                return None, None
            cache_ref = None
            try:
                if captured_cache_bundle:
                    cache_ref = bank_transfer_metadata.write_cache_sidecar(
                        _bank_analysis_cache_dir(dataset_id), captured_cache_bundle)
                    if cache_ref is None:
                        raise RuntimeError(
                            'could not preserve Bank Score/Face cache in Dataset')
                snapshot = bank_transfer_metadata.snapshot_storage(
                    final_analysis, stored, captured=captured_analysis,
                    cache_ref=cache_ref)
                if snapshot is None:
                    raise RuntimeError(
                        'could not seal Bank analysis for Dataset image')
                return snapshot, cache_ref
            except Exception:
                _remove_unreferenced_bank_analysis_cache(dataset_id, cache_ref)
                raise
        fp = None
        if dedupe:
            try:
                with Image.open(io.BytesIO(stored)) as im:
                    fp = _dhash(im)
            except (OSError, ValueError):
                fp = None   # unreadable output would have failed above; belt & braces
            if fp is not None:
                match = None
                stale_ids = set()
                for cached_hash, mid in tuple(seen):
                    if _hamming(fp, cached_hash) > SCRAPE_DHASH_MAX_DISTANCE:
                        continue
                    live = (FaceDatasetImage.query
                            .filter(
                                FaceDatasetImage.id == mid,
                                FaceDatasetImage.dataset_id == dataset_id,
                                FaceDatasetImage.status.in_(('keep', 'pending')))
                            .first())
                    if live is None or not live.filename:
                        stale_ids.add(mid)
                        continue
                    try:
                        live_path = os.path.join(
                            _dataset_dir(dataset_id), live.filename)
                        if (preserve_exact_bytes
                                and Path(live_path).read_bytes() != stored):
                            # Perceptually similar is not byte-identical and
                            # cannot carry this image's exact analysis vault.
                            continue
                        with Image.open(live_path) as im:
                            live_hash = _dhash(im)
                    except (OSError, ValueError):
                        stale_ids.add(mid)
                        continue
                    if live_hash != cached_hash:
                        for cache_index, (_old_hash, cached_id) in enumerate(seen):
                            if cached_id == mid:
                                seen[cache_index] = (live_hash, mid)
                                break
                    if _hamming(fp, live_hash) <= SCRAPE_DHASH_MAX_DISTANCE:
                        match = mid
                        break
                if stale_ids:
                    seen[:] = [
                        (h, mid) for h, mid in seen if mid not in stale_ids
                    ]
                if match is not None:
                    if stats is not None:
                        stats['duplicates'] = stats.get('duplicates', 0) + 1
                    # The dataset already holds this image — hand the provenance to
                    # the row that holds it, so the source can tell it landed. When
                    # that row is already claimed (another bank supplied the same
                    # photo first), report the id back: the caller has no verifiable
                    # trace here and needs to fall back on its own bookkeeping.
                    bid = bank_id_at(index)
                    analysis_snapshot = None
                    analysis_cache_ref = None
                    try:
                        if bid:
                            analysis_snapshot, analysis_cache_ref = (
                                seal_analysis_snapshot())
                        linked_before = db.session.get(FaceDatasetImage, match)
                        previous = ((linked_before.bank_image_id,
                                     linked_before.bank_analysis_snapshot)
                                    if linked_before is not None else None)
                        linked = (bool(bid) and _attach_bank_provenance(
                            match, bid, bank_analysis_snapshot=analysis_snapshot,
                            bank_analysis_cache_dir=_bank_analysis_cache_dir(
                                dataset_id)))
                        linked_after = db.session.get(FaceDatasetImage, match)
                        if (provenance_changes_sink is not None
                                and previous is not None
                                and linked_after is not None
                                and previous != (linked_after.bank_image_id,
                                                linked_after.bank_analysis_snapshot)):
                            provenance_changes_sink.append({
                                'image_id': match,
                                'old_bank_image_id': previous[0],
                                'old_snapshot': previous[1],
                                'new_bank_image_id': linked_after.bank_image_id,
                                'new_snapshot': linked_after.bank_analysis_snapshot,
                            })
                        if bid and not linked and stats is not None:
                            stats.setdefault('bank_unlinked', []).append(bid)
                    except Exception:
                        # `_attach_bank_provenance` commits on success.  A fault
                        # before that commit leaves the session unusable until
                        # rollback; a fault after it is resolved by the durable
                        # ownership proof in `finally` below.
                        db.session.rollback()
                        raise
                    finally:
                        _remove_unreferenced_bank_analysis_cache(
                            dataset_id, analysis_cache_ref)
                    logger.info(f"dataset import: perceptual duplicate skipped (dataset {dataset_id})")
                    continue
        analysis_snapshot, analysis_cache_ref = seal_analysis_snapshot()
        transfer_metadata = transfer_metadata_at(index)
        restored = bank_transfer_metadata.dataset_restore_values(
            transfer_metadata,
            bank_transfer_metadata.content_fingerprint_bytes(stored))
        fn = f"{user_id}_dataset_{uuid.uuid4().hex[:8]}{extension}"
        stored_path = os.path.join(_dataset_dir(dataset_id), fn)
        try:
            write_image_atomic(stored_path, stored)
            cap = (captions_by_index[index] if index < len(captions_by_index) else None)
            cap = _cap_caption(cap) if (cap or '').strip() else None
            restored_short = restored.get('caption_short')
            restored_short = (_cap_caption(restored_short)
                              if isinstance(restored_short, str)
                              and restored_short.strip() else None)
            restored_short_origin = (restored.get('caption_short_origin')
                                     if restored_short else None)
            img = FaceDatasetImage(
                                   dataset_id=dataset_id,
                                   source=restored.get('source') or 'import',
                                   status=status_at(index),
                                   filename=fn, framing=framing_at(index),
                                   variation_label=restored.get('variation_label'),
                                   variation_prompt=restored.get('variation_prompt'),
                                   klein_model=restored.get('klein_model'),
                                   face_score=restored.get('face_score'),
                                   face_state=restored.get('face_state'),
                                   fail_reason=restored.get('fail_reason'),
                                   fail_kind=restored.get('fail_kind'),
                                   upscale_ratio=(restored.get('upscale_ratio')
                                                  if restored.get('upscale_ratio')
                                                  is not None else scale),
                                   caption=cap, caption_short=restored_short,
                                   caption_origin=caption_origin_at(index, cap),
                                   caption_short_origin=restored_short_origin,
                                   bank_image_id=bank_id_at(index),
                                   bank_analysis_snapshot=analysis_snapshot,
                                   transfer_metadata=transfer_metadata,
                                   watermark_state=watermark_state_at(index),
                                   watermark_bbox=watermark_bbox_at(index),
                                   watermark_regions=watermark_regions_at(index),
                                   watermark_source=watermark_source_at(index),
                                   watermark_score=watermark_score_at(index),
                                   source_metadata=_source_metadata_storage(
                                       metadata_by_index[index]
                                       if index < len(metadata_by_index) else None))
            db.session.add(img)
            db.session.commit()
        except Exception:
            db.session.rollback()
            try:
                os.unlink(stored_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning('dataset import: could not remove uncommitted image %s',
                               stored_path, exc_info=True)
            _remove_unreferenced_bank_analysis_cache(
                dataset_id, analysis_cache_ref)
            raise
        if dedupe and fp is not None:
            seen.append((fp, img.id))
        ids.append(img.id)
        if created_ids_sink is not None:
            created_ids_sink.append(img.id)
    return ids, failed


def rollback_imported_images(user_id, dataset_id, image_ids,
                             provenance_changes=None) -> bool:
    """Remove only rows created by a failed multi-chunk Bank promotion.

    The caller holds the Dataset ingest stripe for the whole promotion.  Files
    are moved to the recoverable app trash before one DB commit; any failure
    restores every moved file and leaves the rows intact.
    """
    wanted = {int(value) for value in image_ids}
    provenance_changes = list(provenance_changes or ())
    if not wanted and not provenance_changes:
        return True
    ds = get_dataset(user_id, dataset_id)
    if ds is None:
        return False
    rows = (FaceDatasetImage.query
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.id.in_(wanted)).all())
    if {row.id for row in rows} != wanted:
        return False
    moved = []
    cache_refs = set()
    try:
        for change in reversed(provenance_changes):
            row = (FaceDatasetImage.query
                   .filter_by(id=change.get('image_id'), dataset_id=dataset_id)
                   .one_or_none())
            if (row is None
                    or row.bank_image_id != change.get('new_bank_image_id')
                    or row.bank_analysis_snapshot != change.get('new_snapshot')):
                raise RuntimeError('dedupe provenance changed during rollback')
            new_snapshot = bank_transfer_metadata.parse_snapshot(
                row.bank_analysis_snapshot)
            if new_snapshot and new_snapshot.get('cache_ref'):
                cache_refs.add(new_snapshot['cache_ref'])
            row.bank_image_id = change.get('old_bank_image_id')
            row.bank_analysis_snapshot = change.get('old_snapshot')
        for row in rows:
            snapshot = bank_transfer_metadata.parse_snapshot(
                row.bank_analysis_snapshot)
            if snapshot and snapshot.get('cache_ref'):
                cache_refs.add(snapshot['cache_ref'])
            if row.filename:
                original = os.path.join(_dataset_path(dataset_id), row.filename)
                if os.path.exists(original):
                    trashed = trash.send_to_trash(
                        original,
                        context=f'dataset-{dataset_id}-failed-bank-promotion-{row.id}')
                    moved.append((trashed, original))
            db.session.delete(row)
        db.session.commit()
    except Exception:
        db.session.rollback()
        for trashed, original in reversed(moved):
            _restore_from_trash(trashed, original)
        return False
    cache_dir = _bank_analysis_cache_dir(dataset_id)
    for cache_ref in cache_refs:
        shared = False
        for other in (FaceDatasetImage.query
                      .filter(FaceDatasetImage.dataset_id == dataset_id,
                              FaceDatasetImage.bank_analysis_snapshot.isnot(None))):
            snapshot = bank_transfer_metadata.parse_snapshot(
                other.bank_analysis_snapshot)
            if snapshot and snapshot.get('cache_ref') == cache_ref:
                shared = True
                break
        if not shared:
            bank_transfer_metadata.remove_cache_sidecar(cache_dir, cache_ref)
    return True


# --- Import d'un dataset d'entraînement existant (ZIP kohya-style / dossier) --
# Des images + sidecars .txt de même nom (la convention kohya/ai-toolkit), soit
# dans un ZIP uploadé, soit dans un dossier du disque du serveur (app locale
# mono-user : le chemin est SON disque). Les images gardent leur ratio
# (source préservée par défaut, sans crop), les captions atterrissent sur les rows,
# dédup perceptuelle vs le lot ET le dataset. Les fichiers sont réécrits sous
# des noms générés (jamais celui de la source → aucune traversée possible),
# profondeur de dossiers libre (le ZIP accepte toute arborescence ; le dossier
# est parcouru récursivement pour rester aligné).
DATASET_ZIP_MAX_FILES = 400
DATASET_ZIP_MAX_BYTES = 2 * 1024 * 1024 * 1024
DATASET_ZIP_MAX_IMAGE_BYTES = 128 * 1024 * 1024
_DATASET_ZIP_MAX_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
_DATASET_ZIP_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


@_serialize_dataset_ingest
def _merge_training_images(user_id, dataset_id, entries, captions, stats=None):
    """Cœur commun ZIP/dossier : `entries` = liste de (stem, display_name, getter)
    où `getter()` rend les bytes de l'image, `captions` = {stem: texte}. Chaque
    image lisible devient une row 'import' (status=keep, ratio préservé), la
    caption de même stem est attachée (tronquée à CAPTION_MAX_CHARS), les
    doublons perceptuels (dHash) vs le lot ET le dataset sont sautés — mais leur
    caption, elle, atterrit sur la row déjà présente si celle-ci n'en a pas
    (aller-retour « je légende ailleurs »). Returns (ids, failed)."""
    seen = _existing_dhash_rows(dataset_id)   # [(dhash, image_id)]
    ids, failed = [], 0
    for stem, display, getter in entries:
        try:
            raw = getter()
        except (OSError, ValueError, MemoryError, zipfile.BadZipFile,
                zipfile.LargeZipFile, zlib.error, lzma.LZMAError,
                NotImplementedError, RuntimeError):
            failed += 1
            continue
        if stats is not None:   # même garde qualité que l'import de photos
            try:
                if min(_import_header_dimensions(raw)) < SCRAPE_IMPORT_MIN_SIDE:
                    stats['small'] = stats.get('small', 0) + 1
            except Exception:
                pass
        try:
            stored, extension = import_store_image(raw)
        except Exception as e:
            failed += 1
            logger.warning(f"dataset import: image skipped ({display}): {e}")
            continue
        try:
            with Image.open(io.BytesIO(stored)) as im:
                fp = _dhash(im)
        except (OSError, ValueError):
            fp = None
        incoming = (captions.get(stem) or '').strip() or None
        if fp is not None:
            match = next((mid for h, mid in seen
                          if _hamming(fp, h) <= SCRAPE_DHASH_MAX_DISTANCE), None)
            if match is not None:
                # THE round trip: export the images, caption them in another
                # tool, bring the .txt files back. Those images are duplicates
                # BY DESIGN — dropping the row silently dropped the caption with
                # it ("0 imported · N duplicates skipped"), which made the whole
                # trip a dead end (reported by Qeeyana on Reddit). The pixels are
                # already here; what is new is the text, so the text lands on the
                # row that holds them. A caption written HERE is never
                # overwritten — an import cannot silently rewrite curated work.
                if stats is not None:
                    stats['duplicates'] = stats.get('duplicates', 0) + 1
                row = FaceDatasetImage.query.get(match) if incoming else None
                if row is not None:
                    if (row.caption or '').strip():
                        if stats is not None:
                            stats['captions_kept'] = stats.get('captions_kept', 0) + 1
                    else:
                        # A .txt sidecar is work done by a human in another tool —
                        # the whole point of the round-trip. It lands 'asserted',
                        # which is the same rule the branch above already applies by
                        # hand ("a caption written HERE is never overwritten"),
                        # generalised so a LATER forced pass honours it too.
                        caption_origin.stamp(row, _cap_caption(incoming),
                                             caption_origin.ASSERTED)
                        db.session.commit()
                        if stats is not None:
                            stats['captions_applied'] = \
                                stats.get('captions_applied', 0) + 1
                continue
        fn = f"{user_id}_dsimport_{uuid.uuid4().hex[:8]}{extension}"
        write_image_atomic(os.path.join(_dataset_dir(dataset_id), fn), stored)
        cap = _cap_caption(incoming) if incoming else None
        if cap and stats is not None:
            stats['captions'] = stats.get('captions', 0) + 1
        img = FaceDatasetImage(
            dataset_id=dataset_id, source='import', status='keep', filename=fn,
            caption=cap,
            caption_origin=caption_origin.ASSERTED if cap else None)
        db.session.add(img)
        db.session.commit()
        if fp is not None:
            seen.append((fp, img.id))     # so the rest of the batch dedupes too
        ids.append(img.id)
    return ids, failed


def import_dataset_zip(user_id: int, dataset_id: int,
                       archive: bytes | BinaryIO, stats=None):
    """Import an existing training dataset into THIS dataset (merge, not create):
    every image in the zip becomes an 'import' row (status=keep), a same-stem
    .txt sidecar becomes its caption (truncated to CAPTION_MAX_CHARS). Returns
    (ids, failed). ValueError on a non-zip / oversized archive."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    stream, owned = _coerce_archive_stream(archive)
    try:
        _preflight_zip_central_directory(
            stream, max_entries=DATASET_ZIP_MAX_FILES,
            max_central_bytes=_DATASET_ZIP_MAX_CENTRAL_DIRECTORY_BYTES,
            label='zip')
        try:
            z = zipfile.ZipFile(stream)
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile,
                zlib.error, lzma.LZMAError, NotImplementedError,
                RuntimeError) as exc:
            raise ValueError('not a zip file') from exc
        try:
            infos = [i for i in z.infolist() if not i.is_dir()]
            if len(infos) > DATASET_ZIP_MAX_FILES:
                raise ValueError(
                    f'too many files in the zip (max {DATASET_ZIP_MAX_FILES})')
            if sum(i.file_size for i in infos) > DATASET_ZIP_MAX_BYTES:
                raise ValueError('zip too large (max 2 GB uncompressed)')
            oversized = next((
                i for i in infos
                if i.filename.lower().endswith(_DATASET_ZIP_IMG_EXTS)
                and i.file_size > DATASET_ZIP_MAX_IMAGE_BYTES
            ), None)
            if oversized is not None:
                raise ValueError('image too large in zip (max 128 MiB per image)')
            captions = {}
            for i in infos:
                if i.filename.lower().endswith('.txt') and i.file_size <= 64 * 1024:
                    try:
                        captions[os.path.splitext(i.filename)[0]] = \
                            z.read(i).decode('utf-8', 'replace').strip()
                    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile,
                            zlib.error, lzma.LZMAError, NotImplementedError,
                            RuntimeError):
                        pass
            entries = [
                (os.path.splitext(i.filename)[0], i.filename,
                 lambda i=i: z.read(i))
                for i in infos if i.filename.lower().endswith(_DATASET_ZIP_IMG_EXTS)
            ]
            return _merge_training_images(
                user_id, dataset_id, entries, captions, stats=stats)
        finally:
            z.close()
    finally:
        if owned is not None:
            owned.close()


def import_dataset_folder(user_id, dataset_id, folder, stats=None):
    """Same merge as import_dataset_zip but straight from a folder on the
    server's disk — no need to zip an existing kohya dataset first. Recursive
    (the zip accepts any folder depth, the folder walk mirrors that); non-image
    files are ignored, same-stem .txt sidecars become captions. Returns
    (ids, failed). ValueError on a missing folder / oversized content."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Windows «Copier en tant que chemin» colle le chemin entre guillemets —
    # on les retire pour que le coller-direct marche du premier coup.
    folder = (folder or '').strip().strip('"\'')
    if not folder or not os.path.isdir(folder):
        raise ValueError(f'folder not found or not readable: {folder or "(empty)"}')
    paths = []
    for root, _dirs, files in os.walk(folder):
        paths.extend(os.path.join(root, f) for f in files)
    if len(paths) > DATASET_ZIP_MAX_FILES:
        raise ValueError(f'too many files in the folder (max {DATASET_ZIP_MAX_FILES})')
    sizes, regular_paths = {}, set()
    for p in paths:
        try:
            # Do not follow a symlink into an arbitrary file/pipe outside the
            # folder the user selected. A named pipe must never reach ``open``:
            # it can block the request forever before there are image bytes to
            # validate.
            source_stat = os.lstat(p)
        except OSError:
            sizes[p] = 0
            continue
        if stat.S_ISREG(source_stat.st_mode):
            regular_paths.add(p)
            sizes[p] = source_stat.st_size
        else:
            sizes[p] = 0
    if sum(sizes.values()) > DATASET_ZIP_MAX_BYTES:
        raise ValueError('folder too large (max 2 GB)')
    oversized = next((
        p for p in paths
        if p.lower().endswith(_DATASET_ZIP_IMG_EXTS)
        and p in regular_paths
        and sizes.get(p, 0) > DATASET_ZIP_MAX_IMAGE_BYTES
    ), None)
    if oversized is not None:
        # Match ZIP import's per-image rule before a regular/sparse file is ever
        # opened. The bounded reader below repeats it to cover a live-folder race.
        raise ValueError('image too large in folder (max 128 MiB per image)')
    captions = {}
    for p in paths:
        if (p in regular_paths and p.lower().endswith('.txt')
                and sizes.get(p, 0) <= 64 * 1024):
            try:
                with open(p, 'rb') as fh:
                    captions[os.path.splitext(p)[0]] = \
                        fh.read().decode('utf-8', 'replace').strip()
            except OSError:
                pass

    def _read(p):
        source_stat = os.lstat(p)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError('folder image is not a regular file')
        if source_stat.st_size > DATASET_ZIP_MAX_IMAGE_BYTES:
            raise ValueError('image too large in folder (max 128 MiB per image)')
        with open(p, 'rb') as fh:
            opened_stat = os.fstat(fh.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValueError('folder image is not a regular file')
            raw = fh.read(DATASET_ZIP_MAX_IMAGE_BYTES + 1)
        if len(raw) > DATASET_ZIP_MAX_IMAGE_BYTES:
            raise ValueError('image too large in folder (max 128 MiB per image)')
        return raw

    def _non_regular_image():
        raise ValueError('folder image is not a regular file')

    entries = [
        (os.path.splitext(p)[0], p,
         (lambda p=p: _read(p)) if p in regular_paths else _non_regular_image)
        for p in paths if p.lower().endswith(_DATASET_ZIP_IMG_EXTS)
    ]
    return _merge_training_images(user_id, dataset_id, entries, captions, stats=stats)


# --- Scrape direct → dataset concept ----------------------------------------
# Construction de dataset AUTONOME : on scanne une URL de galerie (routes scrape
# READ-ONLY, /api/scrape/scan + /thumb) et on télécharge les images choisies
# DIRECTEMENT dans le dataset — le pool scrape partagé de l'app source n'est PAS
# porté (cette app ne scrape que pour construire des datasets concept). Filtres :
# dedup perceptuel + résolution + ratio = les 3 filtres « toujours rentables » ;
# flou/watermark restent une décision HUMAINE (la sélection dans la grille de scan).
SCRAPE_IMPORT_MAX = 60             # cap par import (download synchrone parallélisé)
SCRAPE_IMPORT_MIN_SIDE = 768       # ai-toolkit ne fait que downscaler : 768 reste exploitable
SCRAPE_IMPORT_MAX_RATIO = 3.0      # au-delà de 3:1, aucun bucket trainer ne gère proprement
SCRAPE_DHASH_MAX_DISTANCE = 8      # Hamming ≤ 8 sur 64 bits = doublon perceptuel
_SCRAPE_DL_TYPES = ('image/jpeg', 'image/jpg', 'image/png', 'image/webp',
                    'image/bmp')  # pas de gif/svg
_SCRAPE_DL_MAX_BYTES = 25 * 1024 * 1024
_SCRAPE_DL_WORKERS = 6


def _dhash(im: Image.Image) -> int:
    """dHash 64 bits (gradient horizontal sur grayscale 9×8) — PIL pur, insensible
    au resize/re-encodage, donc stable entre un scrape original et sa version
    normalisée webp déjà importée."""
    g = im.convert('L').resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


# Bank analysis needs to decode before its pure-Pillow metrics can downscale, so
# it keeps a header guard at this call.  A Bank may still copy a larger file; it
# simply starts unanalysed and can be reviewed separately.
# It follows the SHARED input budget rather than a private copy of it:
# an image the user was allowed to import must not come back unanalysable because
# a second, stricter number lives here. Kept as module attributes for the tests
# and callers that read them, but resolved live by the check below.
BANK_ANALYSIS_MAX_SIDE = image_encoding.DEFAULT_INPUT_MAX_SIDE
BANK_ANALYSIS_MAX_PIXELS = image_encoding.DEFAULT_INPUT_MAX_PIXELS


def _bank_analysis_dimensions_allowed(im: Image.Image) -> bool:
    """Reject headers whose full decode would exceed the local analysis budget."""
    max_side, max_pixels = preserved_import_limits()
    try:
        width, height = im.size
        return (isinstance(width, int) and isinstance(height, int)
                and width > 0 and height > 0
                and (not max_side or (width <= max_side and height <= max_side))
                and (not max_pixels or width * height <= max_pixels))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False


def _loaded_bank_deterministic_analysis(im: Image.Image) -> dict | None:
    """Apply the header guard, then decode only an image safe for this analysis."""
    if not _bank_analysis_dimensions_allowed(im):
        logger.warning('bank analysis skipped image beyond the %s input budget',
                       image_encoding.input_budget_sentence())
        return None
    # Match the Bank scan's JPEG fast path. It bounds decode work before the
    # quality metric performs its own <=1024px analysis copy; other formats
    # keep their native decoder behavior after the local header guard.
    im.draft(None, (ANALYSIS_MAX_SIDE * 2, ANALYSIS_MAX_SIDE * 2))
    im.load()
    return _bank_deterministic_values(im)


def bank_deterministic_analysis(image_source) -> dict | None:
    """Measure the deterministic Bank fields from one final image.

    Bank -> Dataset always invokes this on its emitted final file, and every Bank
    -> Bank copy invokes it on the destination file. Keeping it here next to the
    Dataset dHash makes both transfer directions use exactly the same pure-Pillow
    formulas as the Bank quality scan, without carrying stale source ML outputs.
    """
    try:
        # Pillow may warn at open time, but the explicit header guard below runs
        # before ``load()``. If an installation promotes that warning to an
        # exception, the dedicated catch remains safe without changing global
        # warning filters shared by concurrent Bank scans.
        if isinstance(image_source, (bytes, bytearray)):
            handle = io.BytesIO(image_source)
            with Image.open(handle) as im:
                return _loaded_bank_deterministic_analysis(im)
        with Image.open(image_source) as im:
            return _loaded_bank_deterministic_analysis(im)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        logger.warning('bank analysis skipped Pillow decompression bomb')
        return None
    except (OSError, TypeError, ValueError, SyntaxError, UnidentifiedImageError):
        return None


def _bank_deterministic_values(im: Image.Image) -> dict:
    """The strict v2 snapshot schema, computed from an already decoded image."""
    metrics = quality_metrics(im)
    provenance = provenance_metrics(im)
    return {
        'quality_state': 'ok',
        'blur_score': metrics['blur_score'],
        'noise_score': metrics['noise_score'],
        'uniformity_score': metrics['uniformity_score'],
        'dhash': f'{_dhash(im):016x}',
        'detail_ratio': provenance['detail_ratio'],
        'bars_ratio': provenance['bars_ratio'],
        'jpeg_quality': provenance['jpeg_quality'],
        'origin': provenance['origin'],
        'origin_evidence': provenance['origin_evidence'],
    }


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def _existing_dhash_rows(dataset_id) -> list:
    """[(dHash, image_id)] des images déjà dans le dataset (keep/pending),
    recalculés à la volée : resize 9×8 ≈ qq ms/image et un dataset plafonne à
    ~200 images — pas de colonne/migration pour si peu. L'id accompagne le hash
    pour que l'appelant sache QUELLE image un doublon a rencontrée (l'import
    depuis une bank y raccroche sa provenance)."""
    out = []
    rows = FaceDatasetImage.query.filter(
        FaceDatasetImage.dataset_id == dataset_id,
        FaceDatasetImage.status.in_(('keep', 'pending'))).all()
    for r in rows:
        if not r.filename:
            continue
        try:
            with Image.open(os.path.join(_dataset_dir(dataset_id), r.filename)) as im:
                out.append((_dhash(im), r.id))
        except (OSError, ValueError):
            continue
    return out


def _existing_dhashes(dataset_id) -> list:
    """Les seuls dHashes (sans les ids) — voir _existing_dhash_rows."""
    return [h for h, _id in _existing_dhash_rows(dataset_id)]


def _attach_bank_provenance(image_id, bank_image_id, *, bank_analysis_snapshot=None,
                            bank_analysis_cache_dir=None) -> bool:
    """Raccroche une image de dataset DÉJÀ présente à la bank_image dont elle est
    le doublon perceptuel, et dit si le lien a été pris. N'écrase jamais une
    provenance existante : la première bank qui a fourni l'image la garde (sinon
    deux banks se voleraient le lien à chaque promotion croisée) — l'appelant
    apprend alors que CETTE bank n'a pas de trace vérifiable ici."""
    if not image_id:
        return False
    row = db.session.get(FaceDatasetImage, image_id)
    if row is None:
        return False
    # A dHash duplicate can be only visually similar, not byte-identical.  Its
    # source Bank scores are useful only when the Dataset file proves it is the
    # exact normalized transfer output; never attach a stale-looking snapshot.
    changed = False
    # First Bank wins for the snapshot exactly as it does for bank_image_id.  A
    # later Bank may hold a perceptual duplicate with different scores, but it
    # must never rewrite the analysis attributed to the original provenance.
    # The sole exception fills a legacy/mid-upgrade row that is already linked to
    # this SAME Bank but has no snapshot yet.
    owns_snapshot = (row.bank_image_id is None
                     or (row.bank_image_id == bank_image_id
                         and row.bank_analysis_snapshot is None))
    if owns_snapshot and bank_analysis_snapshot and row.filename:
        path = os.path.join(_dataset_dir(row.dataset_id), row.filename)
        snapshot = bank_transfer_metadata.compatible_snapshot(
            bank_analysis_snapshot, path)
        cache_ok = (snapshot is not None and (
            not snapshot.get('cache_ref')
            or (bank_analysis_cache_dir is not None
                and bank_transfer_metadata.read_cache_sidecar(
                    bank_analysis_cache_dir, snapshot['cache_ref']) is not None)))
        if cache_ok:
            row.bank_analysis_snapshot = bank_analysis_snapshot
            changed = True
    linked = bool(bank_image_id and row.bank_image_id == bank_image_id)
    if bank_image_id and row.bank_image_id is None:
        row.bank_image_id = bank_image_id
        linked = True
        changed = True
    if changed:
        db.session.commit()
    return linked


def _accept_scrape_bytes(raw, seen_hashes, skipped, rescue_small=False):
    """Filtre une image téléchargée : résolution / ratio / dedup perceptuel.
    Retourne les bytes si acceptée (et enregistre son dHash dans seen_hashes),
    sinon None en incrémentant le compteur skipped adéquat. Quand rescue_small
    est vrai, une petite image continue vers ratio+dedup au lieu d'être rejetée;
    elle ne sera jamais importée directement dans l'entraînement."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                # Scrape quality/dHash must not become the first full decode of a
                # crafted response: run the same header budget as every import lane.
                _preserved_import_header_extension(im)
                im.load()
                w, h = im.size
                if min(w, h) < SCRAPE_IMPORT_MIN_SIDE and not rescue_small:
                    skipped['low_res'] += 1
                    return None
                if max(w, h) > SCRAPE_IMPORT_MAX_RATIO * min(w, h):
                    skipped['extreme_ratio'] += 1
                    return None
                fp = _dhash(im)
    except (OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        skipped['errors'] += 1
        return None
    if any(_hamming(fp, s) <= SCRAPE_DHASH_MAX_DISTANCE for s in seen_hashes):
        skipped['duplicates'] += 1
        return None
    seen_hashes.append(fp)
    return raw


def _scrape_resolution_key(downloaded):
    """Sort key for rescue batches: the best-resolution duplicate must win."""
    reason, raw = downloaded
    if reason != 'ok' or not raw:
        return (0, 0)
    try:
        width, height = _import_header_dimensions(raw, label='scrape import')
        return (min(width, height), width * height)
    except ValueError:
        return (0, 0)


def _save_small_scrape_pair(user_id, dataset_id, raw, prompt, source_metadata=None):
    """Persist the untouched scrape source and enqueue one Klein candidate.

    Returns True when queued, False when enqueue failed. The original and result
    rows are committed before enqueue so a failed queue operation never loses the
    source file or leaves an untracked job.
    """
    from .klein_edit_helper import enqueue_klein_edit

    # This helper is usually called only after `_accept_scrape_bytes`, but it is
    # also a service seam. Validate again rather than letting a direct caller make
    # this its first unbounded decoder.
    ext = _preserved_import_extension(raw)
    filename = f"{user_id}_scrape_small_{uuid.uuid4().hex[:8]}{ext}"
    source_path = os.path.join(_dataset_dir(dataset_id), filename)
    with open(source_path, 'wb') as fh:
        fh.write(raw)

    stored_metadata = _source_metadata_storage(source_metadata)
    source = FaceDatasetImage(
        dataset_id=dataset_id, source='import', status='pending', filename=filename,
        derivation_kind=SMALL_IMAGE_SOURCE,
        variation_label='Small scraped image · original',
        source_metadata=stored_metadata,
    )
    db.session.add(source)
    db.session.flush()
    label = 'Klein rescue · small scraped image'
    candidate = FaceDatasetImage(
        dataset_id=dataset_id, source='generated', status='pending',
        parent_image_id=source.id, derivation_kind=KLEIN_SMALL_IMAGE,
        variation_label=label, variation_prompt=prompt,
        source_metadata=stored_metadata,
    )
    db.session.add(candidate)
    db.session.commit()

    try:
        job_id = enqueue_klein_edit(
            user_id=str(user_id), source_filename=filename, source_path=source_path,
            # Same model as everything else this dataset makes — a rescued 512 px
            # scrape ends up in the SAME training set as the improved images, so
            # running it on another model is exactly the drift the setting exists
            # to stop. None (never chose) = the historical auto pick.
            klein_model=dataset_klein_model(get_dataset(user_id, dataset_id)),
            edit_prompt=prompt, sampler_steps=_generation_steps(),
            base_lora_strength=_generation_base_lora_strength(),
            extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                            'variation_label': label,
                            'derivation_kind': KLEIN_SMALL_IMAGE,
                            'parent_image_id': source.id},
        )
    except Exception as exc:
        candidate.status = 'failed'
        candidate.fail_reason = f'Klein small-image rescue could not be queued: {exc}'
        db.session.commit()
        logger.exception('small-image rescue enqueue failed for dataset %s source %s',
                         dataset_id, source.id)
        return False
    candidate.job_id = job_id
    db.session.commit()
    return True


def _download_scrape_item(item):
    """Télécharge UNE image d'un item de scan ({url,title}) en mémoire, durci
    anti-SSRF (mêmes garanties que /thumb). Retourne (reason, data|None) où
    reason ∈ {'ok','not_image','errors'}. Sûr hors app-context (thread pool)."""
    from ..scrape.netfetch import fetch_hardened_bytes, _validate_public_http_url
    url = (item or {}).get('url')
    if not url:
        return ('errors', None)
    ok_url, _err = _validate_public_http_url(url)
    if not ok_url:
        return ('errors', None)
    ok, data, _ctype, reason = fetch_hardened_bytes(
        url, allowed_types=_SCRAPE_DL_TYPES, max_bytes=_SCRAPE_DL_MAX_BYTES,
        require_image_magic=True)
    if not ok:
        # 'type'/'noimage' = pas une vraie image raster ; le reste = erreur réseau.
        return ('not_image' if reason in ('type', 'noimage') else 'errors', None)
    return ('ok', data)


def scrape_import_urls(user_id, dataset_id, items, rescue_small=False):
    """Télécharge les images scannées SÉLECTIONNÉES directement dans le dataset
    concept — flux AUTONOME. `items` = [{'url','title'}]. Download parallélisé
    (borné), puis filtre + dedup séquentiels (état partagé), puis import brut
    aspect-kept via import_images(crop=False). Renvoie
    {'imported': n, 'rescue_queued': n, 'rescue_failed': n,
     'skipped': {duplicates, low_res, extreme_ratio, not_image, errors}}."""
    from concurrent.futures import ThreadPoolExecutor
    skipped = {'duplicates': 0, 'low_res': 0, 'extreme_ratio': 0,
               'not_image': 0, 'errors': 0}
    items = [it for it in (items or []) if isinstance(it, dict) and it.get('url')]
    if not items:
        return {'imported': 0, 'rescue_queued': 0, 'rescue_failed': 0,
                'skipped': skipped}
    with ThreadPoolExecutor(max_workers=_SCRAPE_DL_WORKERS) as pool:
        # Keep each response tied to its scan item. Rescue sorting changes order,
        # so a separate byte list would otherwise attach the wrong photographer.
        downloaded = list(zip(items, pool.map(_download_scrape_item, items)))

    # In rescue mode a low-resolution duplicate must never claim the dHash first
    # and make the usable HD source look like the duplicate. The legacy path keeps
    # request order exactly as before.
    if rescue_small:
        downloaded.sort(key=lambda pair: _scrape_resolution_key(pair[1]), reverse=True)

    seen_hashes = _existing_dhashes(dataset_id)
    accepted, rescue_candidates = [], []
    for item, (reason, data) in downloaded:
        if reason != 'ok':
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        ok_bytes = _accept_scrape_bytes(data, seen_hashes, skipped,
                                        rescue_small=rescue_small)
        if ok_bytes is not None:
            if rescue_small:
                try:
                    is_small = (min(_import_header_dimensions(
                        ok_bytes, label='scrape import')) < SCRAPE_IMPORT_MIN_SIDE)
                except ValueError:
                    skipped['errors'] += 1
                    continue
                target = rescue_candidates if is_small else accepted
                target.append((ok_bytes, _source_metadata_from_scrape_item(item)))
            else:
                accepted.append((ok_bytes, _source_metadata_from_scrape_item(item)))

    # Capacity and model preflight happen once, after every quality/dedup filter,
    # but before creating a source/result pair. No small candidate => no Klein scan.
    if rescue_candidates:
        in_flight = (FaceDatasetImage.query
                     .filter_by(dataset_id=dataset_id, status='pending')
                     .filter(FaceDatasetImage.filename.is_(None)).count())
        if in_flight + len(rescue_candidates) > MAX_FANOUT:
            raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
        from .klein_edit_helper import (KLEIN_REQUIRED, KleinModelsMissing,
                                        klein_missing_assets)
        missing = klein_missing_assets()
        if any(asset in missing for asset in KLEIN_REQUIRED):
            raise KleinModelsMissing(missing)

    ids, failed = import_images(
        user_id, dataset_id, [raw for raw, _metadata in accepted], crop=False,
        source_metadata=[metadata for _raw, metadata in accepted])
    skipped['errors'] += failed
    raw_prompt = cfg.get('klein.small_image_prompt', '')
    prompt = '' if raw_prompt is None else str(raw_prompt)
    rescue_queued = rescue_failed = 0
    for raw, source_metadata in rescue_candidates:
        try:
            queued = _save_small_scrape_pair(
                user_id, dataset_id, raw, prompt, source_metadata=source_metadata)
        except Exception:
            rescue_failed += 1
            logger.exception('small-image rescue save failed for dataset %s', dataset_id)
            continue
        if queued:
            rescue_queued += 1
        else:
            rescue_failed += 1
    if rescue_candidates:
        _sync_generate_activity(dataset_id)
    return {'imported': len(ids), 'rescue_queued': rescue_queued,
            'rescue_failed': rescue_failed, 'skipped': skipped}


def _parse_classify(raw):
    try:
        start = raw.index('{')
        obj = json.loads(raw[start:raw.index('}', start) + 1])
    except (ValueError, AttributeError):
        return 'unknown', None
    fr = obj.get('framing')
    fr = fr if fr in ('face', 'bust', 'body', 'back') else 'unknown'
    label = ', '.join(str(obj.get(k)) for k in ('angle', 'expression') if obj.get(k))
    return fr, (label or None)


def classify_images(user_id, dataset_id):
    """Classify imported images lacking a framing via Qwen3-VL. Returns count."""
    _guard_not_bank_export(dataset_id)
    try:
        from .vision_ollama import describe_image_ollama, unload_vision_model
    except ImportError:
        raise RuntimeError('vision (Ollama) service not configured/available yet')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = FaceDatasetImage.query.filter_by(
        dataset_id=dataset_id, source='import', framing=None).all()
    # Ids, not ORM objects: see _live_image_row. The commit at the bottom of this
    # loop expires every row it has not reached, and a tile deleted from the grid
    # meanwhile used to kill the whole classification.
    row_ids = [img.id for img in rows]
    n = 0
    vanished = 0
    # Persistent progress indicator (survives a page reload): try/finally guarantees
    # end() runs even if the batch raises → no phantom "Classifying…" spinner.
    token = dataset_activity.begin(dataset_id, 'classify', total=len(row_ids))
    try:
        for i, image_id in enumerate(row_ids):
            dataset_activity.progress(token, done=i + 1)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            path = _img_path(img) if img.filename else ''
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as fh:
                raw = describe_image_ollama(fh.read(), CLASSIFY_PROMPT, num_predict=1200,
                                            prefer_json=True, keep_alive=_VISION_BATCH_KEEPALIVE)
            if not (raw or '').strip():
                # Échec vision (Ollama indisponible) ≠ « framing indéterminé » :
                # on laisse framing=None (retry possible) au lieu d'écrire 'unknown'
                # définitivement, qui bloquerait toute reclassification.
                continue
            framing, label = _parse_classify(raw)
            img.framing = framing
            img.variation_label = label
            db.session.commit()
            n += 1
    finally:
        unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
        dataset_activity.end(token)
    if vanished:
        logger.info('classify: %s image(s) were deleted while the pass ran, skipped',
                    vanished)
    return n


# --- Captioning (JoyCaption / Qwen3-VL, backend picked in Settings) --------
# --- Concept-omission guarantee (ban-list + verify + corrective rewrite) -----
# Negative prompting ALONE leaks (~35% measured e2e on 3 unseen concepts): the
# robustness comes from a deterministic OUTPUT check + targeted correction. Pipeline
# per caption: regex detection (ban-list) -> if leak, Qwen rewrite naming the leaked
# words (<=2 tries) -> mechanical safety net (drop the offending clause). The Qwen
# calls are threaded in via `describe` (our vision seam is a local import inside the
# caption batch); `describe=None` degrades to mechanical scrub only (backend 'joycaption').

# The abliterated Qwen3-VL SOMETIMES emits its reasoning trace ("the task says... we
# need to remove...") or an infinite loop instead of the refined caption - seen ~1/4
# of images. We detect these unusable outputs to fall back on a DIRECT Qwen caption.
# Matches the reasoning/meta phrasings the abliterated Qwen leaks INSTEAD of a caption.
# Widened after real leaks slipped through ("Yes, this describes…", "The original caption
# says…", "Now, check for…", "I think this works"): allow words between "the task/caption"
# and its verb, and add the yes/now/check/i-think markers. Descriptive prose essentially
# never contains these, so a false reject just falls back to a direct caption - cheap.
_REFINE_REASONING_RE = re.compile(
    r'(?:'
    r'\bthe (?:problem|instruction|task|draft|original|caption)(?:\s+\w+){0,4}\s+'
    r'(?:says?|said|mentions?|has|reads?|describes?|is)\b'
    r'|\bwe (?:need|can|should) to (?:remove|rephrase|avoid|describe|keep)'
    r'|\bso we (?:need|can|should)\b'
    r'|\blet me\b|\brephrase\b|\bwait,|\bnow,\s|\bcheck for\b'
    r'|\bi think\b|\bi need to\b|\byes,\s+(?:this|that|the|we|it|but)'
    r')', re.I)

# A concept caption is scene-exhaustive prose; anything this short is a degenerate
# output (e.g. "taking a picture") that just names the concept - never a real caption.
_MIN_CONCEPT_CAPTION_CHARS = 40


def _refine_output_ok(text, prior) -> bool:
    """True if `text` looks like a CLEAN caption - not the Qwen reasoning trace, not a
    degenerate one-liner, not a loop/rambling (bounded to ~2x the source caption `prior`)."""
    t = (text or '').strip()
    if len(t) < _MIN_CONCEPT_CAPTION_CHARS or _REFINE_REASONING_RE.search(t):
        return False
    return len(t) <= 2 * len(prior or '') + 400


def _usable_caption(text) -> bool:
    """A committable concept caption: non-empty prose that is NOT a reasoning trace.
    Length is deliberately NOT gated here - a legitimately terse caption left after the
    clause-scrub must still commit; only the refine-vs-fallback choice (_refine_output_ok)
    weighs length. A degenerate "taking a picture" is handled upstream: the ban-list
    scrubs the concept out, leaving an empty string this rejects."""
    t = (text or '').strip()
    return bool(t) and not _REFINE_REASONING_RE.search(t)


# Words from concept_desc that are never discriminating (articles + generic adjectives
# a legit caption uses elsewhere: "bare shoulders", "full-body"...).
_TERMS_STOP = frozenset((
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'with', 'to', 'from',
    'that', 'this', 'as', 'is', 'are', 'his', 'her', 'their', 'its', 'it', 'one',
    'act', 'shown', 'worn', 'being', 'person', 'subject', 'focal', 'point', 'visible',
    'bare', 'exposed', 'full', 'close', 'closeup', 'close-up', 'wearing', 'showing'))


# A concept training caption must describe the SUBJECT, never the act of image capture.
# The abliterated Qwen reliably leaks capture-language ("holding a phone to frame the
# shot", "point-of-view mirror", "capturing her reflection") that the LLM ban-list
# expansion never fully enumerates - for "a candid mirror selfie" it returned only
# mirror/self-* variants, so phone/smartphone/camera/reflection leaked into ~45/54
# captions. This DETERMINISTIC lexicon is unioned into the ban-list whenever the concept
# is photographic (selfie/mirror/photo/portrait/pov/camera/phone), so those words are
# ALWAYS scrubbed regardless of the LLM. Reproducible from a fresh clone - no reliance on
# the flaky expansion for words we already know.
_CAPTURE_TRIGGERS = ('selfie', 'mirror', 'photo', 'picture', 'portrait', 'camera',
                     'phone', 'pov', 'point of view', 'snapshot', 'webcam', 'pic ')
_CAPTURE_LEXICON = frozenset((
    'selfie', 'self-portrait', 'self-portraiture', 'self-photograph', 'self-shot',
    'mirror', 'reflection', 'reflected', 'reflective surface',
    'phone', 'smartphone', 'cellphone', 'cell phone', 'mobile phone', 'iphone',
    'camera', 'webcam', 'front-facing', 'pov', 'point of view', 'point-of-view'))


def _fallback_concept_terms(desc) -> list:
    """Minimal ban-list WITHOUT the LLM: the meaningful words of concept_desc itself
    (always included, even when the LLM expansion succeeds - the user's words are the
    ground truth), PLUS the capture lexicon when the concept is photographic, PLUS the
    derived body/pose lexical field (so a POSE concept's periphrases - "knees lifted",
    "feet raised", "thighs" for "leg behind head position" - are scrubbed even though the
    description never spells them, and the LLM expansion is FORBIDDEN from listing pose
    words). Deterministic, reproducible from a fresh clone - the leg_behind fix."""
    d = (desc or '').lower()
    words = re.split(r'[^a-zA-Z-]+', d)
    terms = {w.strip('-') for w in words
             if len(w.strip('-')) >= 3 and w.strip('-') not in _TERMS_STOP}
    if any(k in d for k in _CAPTURE_TRIGGERS):
        terms |= _CAPTURE_LEXICON
    terms |= set(concept_lexical_field(desc))
    return sorted(terms)


def _concept_terms_re(terms):
    """Leak-detection regex: word boundaries, space/hyphen interchangeable ("two-piece"
    <-> "two piece"), plurals/-s/-es/-ing/-ed tolerated. None if the list is empty."""
    pats = []
    for t in terms or []:
        t = (t or '').strip().lower()
        if len(t) < 3:
            continue
        p = re.escape(t).replace(r'\ ', r'[\s-]+').replace(r'\-', r'[\s-]+')
        pats.append(p)
    if not pats:
        return None
    return re.compile(r'\b(?:' + '|'.join(pats) + r')(?:e?s|ing|ed)?\b', re.I)


def _scrub_concept_clauses(caption, leak_re):
    """MECHANICAL net: drop the clauses (segments between , ; .) containing a forbidden
    term - the whole clause, not just the word, to keep grammatical prose. If it destroys
    too much (<30 chars), remove only the words."""
    parts = re.split(r'([.;,])', caption or '')
    kept = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        punc = parts[i + 1] if i + 1 < len(parts) else ''
        if seg.strip() and leak_re.search(seg):
            continue
        kept.append(seg + punc)
    out = re.sub(r'\s{2,}', ' ', ''.join(kept)).strip(' ,;')
    if len(out) >= 30:
        return out
    out = re.sub(r'\s{2,}', ' ', leak_re.sub('', caption or '')).strip(' ,;')
    return out


def _parse_terms_json(raw) -> list:
    """Extract the term list from an LLM blocklist reply. Tolerates noise around the
    object AND — critically for the abliterated Qwen, which frequently LOOPS and never
    closes the JSON array (so json.loads fails) — salvages the quoted strings directly,
    KEEPING their order: the model emits the good, concept-specific terms first, then
    combinatorial padding ("mirror selfie shot", "self-portrait photograph"…). Ordered
    de-dup (the loop repeats), stopwords dropped, capped so the padding can't dominate."""
    raw = raw or ''
    terms = None
    start, end = raw.find('{'), raw.rfind('}')
    if 0 <= start < end:
        try:
            data = json.loads(raw[start:end + 1])
            if isinstance(data, dict) and isinstance(data.get('terms'), list):
                terms = data['terms']
        except ValueError:
            terms = None
    if terms is None:
        # Unclosed/looping array → pull the quoted strings after "terms" in order.
        m = re.search(r'"terms"\s*:\s*\[(.*)', raw, re.S)
        terms = re.findall(r'"([^"\\]{1,60})"', m.group(1) if m else raw)
    out, seen = [], set()
    for t in terms:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if 3 <= len(t) <= 40 and t not in _TERMS_STOP and t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) >= 25:
                break
    return out


def _get_concept_terms(ds, image_path=None, describe=None) -> list:
    """Dataset ban-list: union of (LLM expansion cached in ds.concept_terms) and (words
    of concept_desc). The expansion runs ONCE (vision model already warm in the GPU
    window, the image is just a vehicle - the prompt ignores it) and is cached ONLY if it
    succeeds (a failure retries next batch). `describe` is our describe_image_ollama seam;
    None -> fallback words only (no LLM call)."""
    base = _fallback_concept_terms(ds.concept_desc)
    stored = []
    if getattr(ds, 'concept_terms', None):
        try:
            stored = [t for t in json.loads(ds.concept_terms) if isinstance(t, str)]
        except ValueError:
            stored = []
    if stored:
        return sorted(set(stored) | set(base))
    if image_path and describe is not None:
        try:
            with open(image_path, 'rb') as fh:
                raw = describe(
                    fh.read(),
                    EXPAND_CONCEPT_TERMS_PROMPT.format(concept=(ds.concept_desc or '').strip()),
                    # 1200 is ample for a 6-15 term list; keeping it tight bounds the
                    # abliterated model's combinatorial loop so the salvage in
                    # _parse_terms_json keeps the good leading terms.
                    num_predict=1200, prefer_json=True, fmt='json',
                    keep_alive=_VISION_BATCH_KEEPALIVE)
        except OSError:
            raw = ''
        expanded = _parse_terms_json(raw)
        if expanded:
            ds.concept_terms = json.dumps(expanded)
            db.session.commit()
            logger.info('concept terms: %d terms generated for ds%s', len(expanded), ds.id)
            return sorted(set(expanded) | set(base))
        logger.info('concept terms: empty LLM expansion for ds%s -> desc fallback', ds.id)
    return base


def _enforce_concept_omission(caption, leak_re, image_bytes, concept_desc, describe=None):
    """Guarantee omission: detect forbidden terms in `caption`, ask Qwen for a rewrite
    that NAMES the offending words (<=2 tries, kept by _refine_output_ok), then a
    mechanical net (clause drop). Returns the caption (unchanged if no leak). `describe`
    is the vision seam; None -> skip the LLM fix, go straight to the mechanical scrub."""
    if not leak_re or not (caption or '').strip():
        return caption
    if describe is not None:
        for _ in range(2):
            leaked = sorted({m.group(0).lower() for m in leak_re.finditer(caption)})
            if not leaked:
                return caption
            fixed = ''
            try:
                fixed = describe(
                    image_bytes,
                    CAPTION_LEAK_FIX_PROMPT.format(existing=caption, concept=concept_desc,
                                                   leaked=', '.join(leaked)),
                    num_predict=5000, keep_alive=_VISION_BATCH_KEEPALIVE)
            except Exception:  # noqa: BLE001 - best-effort correction
                fixed = ''
            fixed = (fixed or '').strip().strip('"').strip()
            if _refine_output_ok(fixed, caption):
                caption = fixed
    if leak_re.search(caption):
        caption = _scrub_concept_clauses(caption, leak_re)
    return caption


def _caption_concept(ds, force, backend, token=None, image_ids=None,
                     ollama_model=None, extra_instructions='', report=None):
    """Concept caption pipeline (INVERTED logic): describe everything INCLUDING identity
    but OMIT the recurring act so it binds to the trigger. JoyCaption is literal (it NAMES
    the act/fluids/watermark) -> its drafts are REFINED by Qwen, then every caption passes
    the ban-list omission guarantee. Backend gating is honored:
      - 'joycaption' -> Joy drafts only + mechanical scrub (no Qwen calls);
      - 'ollama'     -> Joy skipped, every image direct-Qwen + enforcement;
      - 'auto'       -> Joy drafts refined by Qwen, no-Joy images direct-Qwen, all enforced."""
    concept_desc = (ds.concept_desc or '').strip()
    # Dynamic omission clause: for a POSE concept the generic "describe their pose and
    # body position" line would instruct the VLM to describe the very concept - the
    # builder folds in a concept-specific negative ("do NOT describe the position of the
    # legs/knees/feet…") that overrides it. Byte-identical to the old prompt for non-body
    # concepts. This is the generation-side half of the leg_behind fix.
    cap_prompt = caption_prompt_for_concept(concept_desc)
    # Extra user instructions apply to the DIRECT-caption prompt (the Qwen refine of a Joy
    # draft is a structured transform left untouched). The concept omission still fronts
    # the prompt and the ban-list enforcement still post-filters every caption.
    cap_prompt = _with_caption_instructions(cap_prompt, extra_instructions)
    q = FaceDatasetImage.query.filter_by(dataset_id=ds.id, status='keep')
    if image_ids is not None:
        q = q.filter(FaceDatasetImage.id.in_(image_ids))
    if not force:
        q = q.filter((FaceDatasetImage.caption.is_(None)) | (FaceDatasetImage.caption == ''))
    # (image_id, path), not (row, path): the loops below commit per image and
    # this pass runs for a long time over a live grid. See _live_image_row.
    todo = [(img.id, _img_path(img)) for img in q.all() if img.filename]
    todo = [(image_id, p) for image_id, p in todo if p and os.path.exists(p)]
    if not todo:
        return 0
    # Total for the persistent progress indicator (token owned by the caller).
    dataset_activity.progress(token, total=len(todo),
                              detail=f'Preparing {len(todo)} concept caption(s)…')
    n = 0
    vanished = 0
    remaining = list(todo)
    refine_targets = []  # (image_id, p, joycap) -> Joy draft refined by Qwen
    # 1) JoyCaption batch (draft) when the backend allows it.
    if backend in ('auto', 'joycaption'):
        jc = {}
        try:
            from .joycaption import caption_images_joycaption, is_available
            if is_available():
                dataset_activity.progress(
                    token, detail=f'Loading JoyCaption model and captioning {len(todo)} images…')
                jc = caption_images_joycaption(
                    [p for _, p in todo], prompt=cap_prompt, activity_token=token,
                    should_cancel=lambda: dataset_activity.cancel_requested(ds.id))
            elif backend == 'joycaption':
                raise RuntimeError('JoyCaption backend is not available - check the ai-toolkit folder in Settings')
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning('caption concept: JoyCaption indisponible (%s)', e)
        still = []
        for image_id, p in remaining:
            cap = (jc.get(p) or '').strip().strip('"').strip()
            if cap:
                refine_targets.append((image_id, p, cap))
            else:
                still.append((image_id, p))
        remaining = still
    # 2a) Backend 'joycaption' forced: no Qwen. Store Joy drafts scrubbed mechanically
    #     (leak_re from the desc words only) - respects "no Ollama fallback".
    if backend == 'joycaption':
        leak_re = _concept_terms_re(_fallback_concept_terms(concept_desc))
        for image_id, p, joycap in refine_targets:
            if dataset_activity.cancel_requested(ds.id):
                break   # graceful stop at an image boundary (see caption_images)
            dataset_activity.bump(token)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            try:
                with open(p, 'rb') as fh:
                    data = fh.read()
            except OSError:
                data = b''
            final = _enforce_concept_omission(joycap, leak_re, data, concept_desc) or joycap
            caption_origin.stamp(img, _cap_caption(final), caption_origin.JOYCAPTION)
            db.session.commit()
            n += 1
            _writer(report, CAPTION_WRITER_JOYCAPTION)
        return n
    # 2b) Qwen passes ('auto'/'ollama'): refine Joy drafts, direct-caption the rest, all
    #     enforced. One model load -> unload once at the end.
    if refine_targets or remaining:
        try:
            from .vision_ollama import describe_image_ollama, unload_vision_model
        except ImportError:
            raise RuntimeError('vision (Ollama) service not configured/available yet')
        # Bind the per-dataset model once for EVERY Concept inference pass. Without
        # this, only the main caption/refine used the override while blocklist
        # expansion and omission rewrites silently loaded the global model.
        def describe(image_bytes, prompt, **kwargs):
            if ollama_model:
                kwargs['model'] = ollama_model
            return describe_image_ollama(image_bytes, prompt, **kwargs)
        # Ban-list (LLM expansion cached + desc words) -> leak regex, compiled ONCE per
        # batch, AFTER the Joy subprocess finished (never two models in VRAM at once).
        sample = refine_targets[0][1] if refine_targets else remaining[0][1]
        leak_re = _concept_terms_re(_get_concept_terms(ds, image_path=sample,
                                                       describe=describe))
        try:
            for image_id, p, joycap in refine_targets:
                if dataset_activity.cancel_requested(ds.id):
                    break   # graceful stop at an image boundary (see caption_images)
                dataset_activity.bump(token)
                with open(p, 'rb') as fh:
                    data = fh.read()
                refined = ''
                # The refine prompt is where the concept-omitting caption is actually
                # PRODUCED when JoyCaption is available (the dominant path), so the
                # per-dataset extra instructions — including the NSFW vocabulary preset —
                # must ride here too. Applied ONLY to cap_prompt before, they never reached
                # the refine, so an 'explicit' preset silently produced a neutral caption:
                # the (abliterated) refiner rewrote the crude Joy draft "as a clean caption"
                # with no register directive. Empty extras keep the prompt byte-identical.
                refine_prompt = _with_caption_instructions(
                    CAPTION_REFINE_CONCEPT_PROMPT.format(existing=joycap,
                                                         concept=concept_desc),
                    extra_instructions)
                try:
                    refined = describe(
                        data, refine_prompt,
                        num_predict=5000,
                        keep_alive=_VISION_BATCH_KEEPALIVE,
                        timeout=(10, 300))
                except Exception as e:  # noqa: BLE001 - refine best-effort
                    logger.warning('caption concept: Qwen refine failed (%s)', e)
                refined = (refined or '').strip().strip('"').strip()
                # Which engine gets the credit follows the text through the three
                # outcomes below, rather than being decided by the branch we are in:
                # a Joy draft kept because the refine was unusable is JoyCaption's
                # sentence, not Qwen's.
                writer = CAPTION_WRITER_REFINED
                if _refine_output_ok(refined, joycap):
                    final = refined
                    origin = caption_origin.OLLAMA
                else:
                    # Unusable refine (reasoning trace / loop) -> direct Qwen caption
                    # (natively omits the concept), else keep the Joy draft.
                    logger.info('caption concept: refine rejected -> direct Qwen (image %s)',
                                image_id)
                    alt = ''
                    try:
                        alt = describe(data, cap_prompt, num_predict=2000,
                                       keep_alive=_VISION_BATCH_KEEPALIVE,
                                       timeout=(10, 300))
                    except Exception:  # noqa: BLE001
                        alt = ''
                    alt = (alt or '').strip().strip('"').strip()
                    final = alt or joycap
                    writer = CAPTION_WRITER_OLLAMA if alt else CAPTION_WRITER_JOYCAPTION
                    origin = caption_origin.OLLAMA if alt else caption_origin.JOYCAPTION
                final = _enforce_concept_omission(final, leak_re, data, concept_desc,
                                                  describe=describe) or final
                # Re-read only now: everything above is model work measured in
                # seconds per image, and the tile can be deleted during it.
                img = _live_image_row(image_id)
                if img is None:
                    vanished += 1
                    continue
                if not _usable_caption(final):
                    # Refine AND direct both unusable → fall back to the Joy draft (clean
                    # prose), scrubbed of any leak; leave blank if even that fails.
                    final = _enforce_concept_omission(joycap, leak_re, data, concept_desc,
                                                      describe=describe) or joycap
                    writer = CAPTION_WRITER_JOYCAPTION
                    origin = caption_origin.JOYCAPTION
                    if not _usable_caption(final):
                        # force=re-do-all: overwrite any stale pre-fix caption with blank
                        # (trigger-only is valid for a concept LoRA) rather than retain it.
                        # The stamp is cleared WITH the text: a blanked row must not keep
                        # an origin describing a sentence that no longer exists.
                        if force and (img.caption or ''):
                            caption_origin.stamp(img, '', None)
                            db.session.commit()
                        logger.info('caption concept: no usable caption for image %s '
                                    '-> left blank', image_id)
                        continue
                caption_origin.stamp(img, _cap_caption(final), origin)
                db.session.commit()
                n += 1
                _writer(report, writer)
            for image_id, p in remaining:
                if dataset_activity.cancel_requested(ds.id):
                    break   # graceful stop at an image boundary (see caption_images)
                dataset_activity.bump(token)
                with open(p, 'rb') as fh:
                    data = fh.read()
                cap = describe(
                    data, cap_prompt, num_predict=2000,
                    keep_alive=_VISION_BATCH_KEEPALIVE,
                    auto_start_local=True, timeout=(10, 300))
                cap = (cap or '').strip().strip('"').strip()
                if cap:
                    cap = _enforce_concept_omission(cap, leak_re, data, concept_desc,
                                                    describe=describe) or cap
                # Re-read after the call, for the same reason as the refine loop.
                img = _live_image_row(image_id)
                if img is None:
                    vanished += 1
                    continue
                if _usable_caption(cap):
                    caption_origin.stamp(img, _cap_caption(cap), caption_origin.OLLAMA)
                    db.session.commit()
                    n += 1
                    _writer(report, CAPTION_WRITER_OLLAMA)
                else:
                    if force and (img.caption or ''):
                        caption_origin.stamp(img, '', None)
                        db.session.commit()
                    logger.info('caption concept: no usable direct caption for image '
                                '%s -> left blank', image_id)
        finally:
            if ollama_model:
                unload_vision_model(model=ollama_model)
            else:
                unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
    if vanished:
        logger.info('caption concept: %s image(s) were deleted while the pass ran, '
                    'skipped', vanished)
    return n


def caption_images(user_id, dataset_id, force=False, mode=None, image_ids=None, report=None,
                   outcome=None):
    """Caption les images gardees. Defaut: seulement celles SANS caption ; force=True
    re-capte TOUTES les gardees (ecrase) - pour rejouer apres un changement de prompt.
    Chaque caption passe par drop_identity_sentences (retire une eventuelle phrase
    d'identite isolee).

    `image_ids` (optionnel) restreint la passe a ce sous-ensemble d'images gardees —
    utilise par le bouton 🔄 Re-caption cible du panneau Identity-leak (une seule image
    ou « toutes les fuyantes ») ; None -> tout le dataset (comportement batch). Meme
    moteur, meme mode, meme contexte kind et memes regles de nettoyage que le lot complet.

    `captioning.backend` (réglages) pilote qui capte quoi :
      - 'none'       -> désactivé, RuntimeError (mappée 409 par la route).
      - 'joycaption' -> JoyCaption seul, PAS de repli Ollama.
      - 'ollama'     -> Ollama (Qwen3-VL) seul, JoyCaption jamais tenté.
      - 'auto'       -> comportement historique : JoyCaption en priorité,
                        fallback Ollama pour les images qu'il n'a pas captées.

    `report` (optionnel) : dict rempli avec le nombre de captions écrites PAR MOTEUR
    ({'joycaption': n, 'ollama': n, 'joycaption_refined': n} — clés absentes quand le
    moteur n'a rien écrit). C'est la seule façon, après coup, de savoir QUI a rédigé :
    'auto' enchaîne les deux moteurs sans le dire, et leurs styles diffèrent. La
    valeur de retour reste le NOMBRE total de captions (contrat inchangé).

    `outcome` (optionnel) : dict rempli avec ce que la passe a TRAITÉ SANS écrire —
    {'skipped': n, 'skipped_reason': "…"} quand le moteur a refusé des images. Un
    refus par image n'interrompt pas le lot, donc sans ce canal la seule trace était
    l'écart entre le nombre d'images et le nombre de captions ; la raison, elle,
    restait dans le log du serveur."""
    _guard_not_bank_export(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    # Per-dataset method overrides (Captions ⚙️ Options): the chosen engine, an extra
    # instruction appended to the prompt, and the Ollama vision model to run. Each falls
    # back to the global default when the dataset never set it.
    opts = caption_options(ds)
    backend = (opts.get('backend') or cfg.get('captioning.backend') or 'auto').lower()
    if backend == 'none':
        raise RuntimeError('No captioning backend configured')
    # Vocabulary preset (NSFW register) + free-text steer, combined into the one block that
    # rides at the end of every prompt this run builds.
    extra_instructions = _combined_caption_instructions(opts)
    ollama_model = (opts.get('ollama_model') or '').strip() or None
    # A targeted subset (Identity-leak panel): normalize to ints once, drop non-numeric.
    # `None` = whole dataset; an EMPTY subset (nothing to re-caption) short-circuits to 0
    # rather than silently captioning everything.
    ids = None
    if image_ids is not None:
        ids = [int(i) for i in image_ids
               if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
        if not ids:
            return 0
    # Dataset CONCEPT : logique INVERSÉE (décrire tout SAUF l'acte récurrent → il se lie
    # au trigger). Pipeline dédié Joy→Qwen + garantie d'omission (ban-list) : entièrement
    # à part du chemin character ci-dessous. Respecte le backend gating.
    # The persistent indicator is owned HERE (begin/finally) so the concept body stays
    # unindented; it only feeds progress via the passed token.
    if is_concept(ds):
        token = dataset_activity.begin(
            dataset_id, 'recaption' if force else 'caption',
            detail='Preparing concept captioning…')
        started = time.monotonic()
        logger.info('captioning started: dataset=%s backend=%s force=%s kind=concept',
                    dataset_id, backend, force)
        try:
            n = _caption_concept(ds, force, backend, token=token, image_ids=ids,
                                 ollama_model=ollama_model,
                                 extra_instructions=extra_instructions, report=report)
            logger.info('captioning finished: dataset=%s backend=%s captioned=%s elapsed=%.1fs',
                        dataset_id, backend, n, time.monotonic() - started)
            return n
        except Exception:
            logger.exception('captioning failed: dataset=%s backend=%s kind=concept elapsed=%.1fs',
                             dataset_id, backend, time.monotonic() - started)
            raise
        finally:
            dataset_activity.end(token)
    # Style de caption : prose (Z-Image) vs tags booru (SDXL booru-native type bigLove).
    # Défaut AUTO selon le type entraîné ; un mode explicite (UI) l'emporte — c'est ce
    # qui rend le captioning « model-matched » réglable sans 2e mécanisme.
    # Anima est HYBRIDE (booru ET langage naturel sont natifs) : son défaut reste la
    # prose, mais mode='booru' est un choix légitime, pas un contournement — le garde
    # MISMATCH_CAPTION du lancement ne dit rien sur anima (lora_training.assert_trainable).
    ttype = (getattr(ds, 'train_type', None) or 'zimage').lower()
    mode = (mode or ('booru' if ttype == 'sdxl' else 'prose')).lower()
    style = is_style(ds)
    if style:
        # Dataset STYLE : captions de CONTENU pur — le rendu n'est jamais décrit pour
        # qu'il soit absorbé par le LoRA. AUCUN nettoyage d'identité : les sujets varient,
        # leur description EST le contenu contrôlable. Le prompt porte la règle, mais elle
        # est NÉGATIVE et JoyCaption ne la suit pas — d'où le post-filtre d'amorce, exact
        # pendant de drop_identity_sentences sur la voie character.
        cap_prompt = caption_prompt_for_style(mode)
        def cleaner(text):
            return drop_style_lead_in(text)
    else:
        # Fidélité corps : le prompt bannit EN PLUS les marques corporelles permanentes
        # (tatouages/cicatrices/piercings…) et le post-filtre les retire — elles doivent
        # se lier au trigger, pas aux mots (même principe que le visage).
        body = is_body_fidelity(ds)
        cap_prompt = caption_prompt_for(mode, body=body)
        base_cleaner = drop_identity_tags if mode == 'booru' else drop_identity_sentences
        def cleaner(text):
            return base_cleaner(text, body=body)
    # Extra user instructions ride at the END of the prompt (both engines) — the kind
    # omission rules stay first, and the cleaner above still post-filters the output.
    cap_prompt = _with_caption_instructions(cap_prompt, extra_instructions)
    q = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
    if ids is not None:
        q = q.filter(FaceDatasetImage.id.in_(ids))
    if not force:
        q = q.filter((FaceDatasetImage.caption.is_(None)) | (FaceDatasetImage.caption == ''))
    rows = q.all()
    # (image_id, path), not (row, path): both loops below commit per image, which
    # expires every row still to come, and this pass runs for minutes-to-hours
    # over a grid the user keeps working in. See _live_image_row.
    todo = [(img.id, _img_path(img)) for img in rows if img.filename]
    todo = [(image_id, p) for image_id, p in todo if p and os.path.exists(p)]
    if not todo:
        return 0
    # Persistent progress indicator (survives a page reload): 'recaption' when force
    # overwrites existing captions, else 'caption'. try/finally guarantees end() runs
    # even if the vision pass raises → no phantom "Captioning…" spinner after a crash.
    token = dataset_activity.begin(
        dataset_id, 'recaption' if force else 'caption', total=len(todo),
        detail=f'Preparing to caption {len(todo)} image(s)…')
    started = time.monotonic()
    logger.info('captioning started: dataset=%s backend=%s mode=%s force=%s images=%s',
                dataset_id, backend, mode, force, len(todo))
    try:
        n = 0
        vanished = 0
        remaining = todo
        # In 'auto', why JoyCaption didn't contribute (deps missing / crash). Kept so a
        # LATER Ollama failure reports BOTH reasons instead of only the Ollama one —
        # otherwise a user whose JoyCaption is silently unavailable debugs blind (issue #6).
        joycaption_note = ''
        # 1) JoyCaption en BATCH (un seul chargement du 8B NF4, via le venv ai-toolkit) -
        # sauté entièrement quand le backend force 'ollama'.
        if backend in ('auto', 'joycaption'):
            jc = {}
            # Why an image came back WITHOUT a caption, keyed by path. In 'auto' the
            # Ollama phase covers those images and the reason is noise; with
            # 'joycaption' forced there IS no second phase, so this is the only
            # material available to tell the user what happened to them.
            jc_errors: dict = {}
            try:
                from .joycaption import availability, caption_images_joycaption, is_available
                if is_available():
                    dataset_activity.progress(
                        token,
                        detail=f'Loading JoyCaption model and captioning {len(todo)} images…')
                    # Consigne « ne décris pas le visage » → les traits se lient au trigger,
                    # pas aux mots de la caption (deep-research 2026-06-14).
                    jc = caption_images_joycaption(
                        [p for _, p in todo], prompt=cap_prompt, activity_token=token,
                        should_cancel=lambda: dataset_activity.cancel_requested(dataset_id),
                        errors_out=jc_errors)
                elif backend == 'joycaption':
                    # Explicit choice, explicit failure: a user who forced 'joycaption' in
                    # Settings must be told WHY (the exact missing deps + pip command),
                    # not get a silent 0 (only 'auto' is allowed to fall back to Ollama).
                    raise RuntimeError(
                        'JoyCaption backend is not available — '
                        + (availability().get('detail') or 'check the ai-toolkit folder in Settings'))
                else:  # auto: JoyCaption unavailable -> remember the reason, fall back to Ollama
                    joycaption_note = availability().get('detail') or 'JoyCaption unavailable'
            except RuntimeError:
                raise
            except Exception as e:
                joycaption_note = str(e)
                logger.warning('caption_images: JoyCaption indisponible (%s)', e)
            still = []
            for image_id, p in remaining:
                cap = (jc.get(p) or '').strip().strip('"').strip()
                if cap:
                    img = _live_image_row(image_id)
                    if img is None:      # deleted while the batch ran
                        vanished += 1
                        dataset_activity.bump(token)
                        continue
                    cleaned = cleaner(cap) or cap
                    caption_origin.stamp(img, _cap_caption(cleaned),
                                         caption_origin.JOYCAPTION)
                    db.session.commit()
                    n += 1
                    _writer(report, CAPTION_WRITER_JOYCAPTION)
                    dataset_activity.bump(token)   # this image is captioned (done)
                else:
                    still.append((image_id, p))
            remaining = still
            dataset_activity.progress(
                token, detail=f'JoyCaption finished; {len(remaining)} image(s) remaining…')
            if backend == 'joycaption':  # backend forcé JoyCaption -> pas de repli Ollama
                # These images are HANDLED — refused, but handled: nothing else will
                # look at them in this run. Leaving them uncounted is what made a
                # finished pass look STUCK: the indicator froze at "37/89" forever
                # because 52 refusals never advanced it, and the toast said "37
                # captioned" with no hint that 52 images had been turned away.
                if remaining:
                    dataset_activity.bump(token, len(remaining))
                    _record_caption_skips(outcome, remaining, jc_errors)
                    logger.info('captioning: %d image(s) refused by JoyCaption, first '
                                'reason: %s', len(remaining),
                                _first_caption_skip_reason(remaining, jc_errors))
                logger.info('captioning finished: dataset=%s backend=%s captioned=%s '
                            'skipped=%s deleted_mid_pass=%s elapsed=%.1fs',
                            dataset_id, backend, n, len(remaining), vanished,
                            time.monotonic() - started)
                return n
        # 2) Ollama (Qwen3-VL) pour les images non couvertes par JoyCaption ('auto'),
        # ou pour TOUT le lot si le backend force 'ollama'.
        if remaining:
            try:
                from .vision_ollama import describe_image_ollama, unload_vision_model
            except ImportError:
                raise RuntimeError('vision (Ollama) service not configured/available yet')
            try:
                for index, (image_id, p) in enumerate(remaining, 1):
                    # Graceful stop: the user asked to stop and we're at an image
                    # boundary (nothing decoding) — leave the rest uncaptioned and let
                    # the finally below free the model, exactly like a normal finish.
                    if dataset_activity.cancel_requested(dataset_id):
                        break
                    dataset_activity.progress(
                        token,
                        detail=f'Captioning with Ollama — image {index}/{len(remaining)}…')
                    with open(p, 'rb') as fh:
                        cap = describe_image_ollama(
                            fh.read(), cap_prompt, num_predict=2000, model=ollama_model,
                            keep_alive=_VISION_BATCH_KEEPALIVE,
                            auto_start_local=(index == 1), timeout=(10, 300))
                    cap = (cap or '').strip().strip('"').strip()
                    if cap:
                        # Re-read AFTER the call: the answer we are about to store
                        # took a full VLM inference to arrive, and the tile can
                        # have been deleted from the grid in that time.
                        img = _live_image_row(image_id)
                        if img is None:
                            vanished += 1
                            dataset_activity.bump(token)
                            continue
                        cleaned = cleaner(cap) or cap
                        # Which engine wrote THIS row, not which backend was asked
                        # for: in 'auto' the two branches both write inside one run.
                        caption_origin.stamp(img, _cap_caption(cleaned),
                                             caption_origin.OLLAMA)
                        db.session.commit()
                        n += 1
                        _writer(report, CAPTION_WRITER_OLLAMA)
                    dataset_activity.bump(token)   # image handled (captioned or not)
            except RuntimeError as e:
                # 'auto' tried JoyCaption first and it was unavailable, then Ollama
                # failed too — report BOTH so the user isn't repairing blind (they'd
                # otherwise see only the Ollama error and never learn JoyCaption's deps
                # are missing, issue #6). backend='ollama' has no note -> re-raise as-is.
                if joycaption_note:
                    raise RuntimeError(f'JoyCaption unavailable: {joycaption_note} · Ollama: {e}') from e
                raise
            finally:
                unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
        logger.info('captioning finished: dataset=%s backend=%s captioned=%s '
                    'deleted_mid_pass=%s elapsed=%.1fs',
                    dataset_id, backend, n, vanished, time.monotonic() - started)
        return n
    except Exception:
        logger.exception('captioning failed: dataset=%s backend=%s elapsed=%.1fs',
                         dataset_id, backend, time.monotonic() - started)
        raise
    finally:
        dataset_activity.end(token)


def caption_paths(paths, *, prompt=None, backend=None, ollama_model=None,
                  extra_instructions=None, should_cancel=None, on_caption=None,
                  progress=None) -> dict:
    """Caption a list of image FILE PATHS with the app's configured engines, returning
    {path: caption}. Dataset-free, purely DESCRIPTIVE captioning (no trigger word, no
    identity/concept/style omission) for the image bank and the future launch-all
    pipeline — a bank caption is a plain description that doubles as search text.

    Reuses the SAME inference bricks as the dataset caption pass (`caption_images`):
    JoyCaption in one batch load, then Ollama (Qwen3-VL) per image for whatever it
    didn't cover, gated by `captioning.backend`. What it deliberately SKIPS is all the
    per-dataset kind logic (prompt building, leak cleaners, dual shorts).

    prompt          : override the default neutral descriptive prompt.
    backend         : override captioning.backend ('auto'|'joycaption'|'ollama'|'none').
    ollama_model    : override the Ollama vision model (None = global default).
    extra_instructions : appended to the prompt (both engines), like the dataset options.
    should_cancel() : polled at each image boundary in the Ollama phase for a graceful
                      stop (JoyCaption runs as one batch and isn't interruptible mid-load,
                      same as the dataset pass). The Ollama phase overlaps several calls
                      (see vision_pool), so a stop drains what is in flight — a couple of
                      seconds — and every drained answer is still handed to on_caption.
    on_caption(path, caption, engine) : fired as each caption lands, for incremental
                      persistence. ALWAYS called on the caller's own thread, never on a
                      worker, so it is free to use the database session. `engine` is the
                      one that ACTUALLY wrote this caption ('joycaption' | 'ollama'),
                      which under the 'auto' backend differs from image to image inside
                      one run — the caller records it as the caption's origin.
    progress(done, total)     : progress callback (every handled image, captioned or not).

    Best-effort: a totally unavailable engine raises RuntimeError (so the caller can
    surface WHY); an individual empty caption is simply skipped. Unloads the Ollama model
    at the end (VRAM back to ComfyUI). Holding the GPU-exclusive vision window is the
    CALLER's job, so launch-all can keep ONE window across several steps."""
    paths = [p for p in (paths or []) if p and os.path.isfile(p)]
    total = len(paths)
    out: dict = {}
    if progress:
        progress(0, total)
    if not paths:
        return out
    backend = (backend or cfg.get('captioning.backend') or 'auto').lower()
    if backend == 'none':
        raise RuntimeError('No captioning backend configured')
    cap_prompt = prompt or DESCRIPTIVE_CAPTION_PROMPT
    if extra_instructions:
        cap_prompt = _with_caption_instructions(cap_prompt, (extra_instructions or '').strip())
    ollama_model = (ollama_model or '').strip() or None
    done = 0

    def _emit(p, cap, engine=None):
        nonlocal done
        out[p] = cap
        if on_caption:
            # The ENGINE rides with the caption so the caller can record who wrote
            # it. Third argument, defaulted on the callback side, so a handler
            # written before this seam existed still works unchanged.
            on_caption(p, cap, engine)
        done += 1
        if progress:
            progress(done, total)

    remaining = list(paths)
    # 1) JoyCaption batch (single 8B NF4 load via the ai-toolkit venv) — skipped when
    # the backend forces 'ollama'.
    joycaption_note = ''
    if backend in ('auto', 'joycaption'):
        jc = {}
        try:
            from .joycaption import availability, caption_images_joycaption, is_available
            if is_available():
                jc = caption_images_joycaption(remaining, prompt=cap_prompt,
                                               should_cancel=should_cancel)
            elif backend == 'joycaption':
                raise RuntimeError(
                    'JoyCaption backend is not available — '
                    + (availability().get('detail') or 'check the ai-toolkit folder in Settings'))
            else:  # auto: unavailable → remember why, fall back to Ollama
                joycaption_note = availability().get('detail') or 'JoyCaption unavailable'
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 — any JoyCaption crash falls back to Ollama in auto
            joycaption_note = str(e)
            logger.warning('caption_paths: JoyCaption unavailable (%s)', e)
        still = []
        for p in remaining:
            cap = (jc.get(p) or '').strip().strip('"').strip()
            if cap:
                _emit(p, _cap_caption(cap), caption_origin.JOYCAPTION)
            else:
                still.append(p)
        remaining = still
        if backend == 'joycaption':
            return out
    # 2) Ollama (Qwen3-VL) for whatever JoyCaption didn't cover, or the whole set when
    # the backend forces 'ollama'.
    if remaining:
        try:
            from .vision_ollama import describe_image_ollama, unload_vision_model
        except ImportError:
            raise RuntimeError('vision (Ollama) service not configured/available yet')
        from .vision_pool import map_vision

        def _describe(path, *, auto_start=False):
            """One caption call. Runs on a WORKER thread under map_vision, so it
            touches nothing but the file and the network."""
            with open(path, 'rb') as fh:
                return describe_image_ollama(
                    fh.read(), cap_prompt, num_predict=2000, model=ollama_model,
                    keep_alive=_VISION_BATCH_KEEPALIVE,
                    auto_start_local=auto_start, timeout=(10, 300))

        def _land(path, cap):
            """Persist one answer. Always on the CALLING thread — `on_caption` is
            what writes to the database, and that session isn't thread-safe."""
            nonlocal done
            cap = (cap or '').strip().strip('"').strip()
            if cap:
                _emit(path, _cap_caption(cap), caption_origin.OLLAMA)
            else:
                done += 1  # handled-but-empty still advances the bar
                if progress:
                    progress(done, total)

        try:
            # The first image runs ALONE, and is the only one allowed to start a
            # stopped local Ollama: a cold server must be woken (and diagnosed)
            # once, not by several callers racing into the same restart. It also
            # warms the model, so the calls that follow overlap real inference
            # instead of queueing behind a model load.
            first, rest = remaining[0], remaining[1:]
            if not (should_cancel and should_cancel()):
                _land(first, _describe(first, auto_start=True))
                # The rest overlap: most of a caption call is round-trip waiting,
                # not GPU work (services/vision_pool.py has the measurements).
                # should_cancel is still polled per image, so the graceful stop
                # keeps its meaning — it just drains the calls in flight first.
                for path, cap, error in map_vision(rest, _describe,
                                                   should_cancel=should_cancel):
                    if error is not None:
                        # A file that vanished mid-pass, a permission error: one
                        # image is skipped and counted, the batch goes on.
                        logger.warning('caption_paths: %s skipped: %s',
                                       os.path.basename(path), error)
                        done += 1
                        if progress:
                            progress(done, total)
                        continue
                    _land(path, cap)
        except RuntimeError as e:
            # 'auto' tried JoyCaption first and it was unavailable, then Ollama failed too
            # — report BOTH so the caller isn't debugging blind (issue #6 reasoning).
            if joycaption_note:
                raise RuntimeError(f'JoyCaption unavailable: {joycaption_note} · Ollama: {e}') from e
            raise
        finally:
            unload_vision_model()  # hand the VRAM back to ComfyUI
    return out


# --- Caption Lab: per-candidate preview (no persistence) ---------------------
# The 🧪 Caption Lab lets the user try a caption CONFIG (engine × Ollama model ×
# vocabulary register) on ONE image and read the result WITHOUT writing anything to
# the row. It rides on caption_paths() — the dataset-free by-path brick — so it runs
# purely DESCRIPTIVE captioning (no kind omission, no dual short): the point is to
# compare raw model output side by side and pick the config, not to produce the final
# stored caption (that still goes through the normal caption pass with its kind rules).

def _compose_preview_instructions(vocabulary, instructions, length=None) -> str | None:
    """Combine the presets (the SAME appended register and length text the dataset pass
    uses) with the user's free extra instructions into the single ``extra_instructions``
    string caption_paths appends to the prompt. Same order as the dataset pass — presets
    first, free text last. None when nothing is set (byte-identical to a plain descriptive
    pass)."""
    parts = _caption_preset_parts(vocabulary, length)
    extra = (instructions or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    if extra:
        parts.append(extra)
    return '\n'.join(parts) if parts else None


# Public so the image bank's caption lane validates against — and appends — the SAME
# vocabulary registers and length presets as the dataset pass, rather than duplicating the
# tuples or the texts.
CAPTION_VOCABULARIES = _CAPTION_VOCABULARIES
CAPTION_LENGTHS = _CAPTION_LENGTHS
# ...and the same closed list of ENGINES, so a per-run engine override validates against
# one tuple instead of a copy that drifts the day a fifth engine lands.
CAPTION_BACKENDS = _CAPTION_BACKENDS


def vocabulary_instruction(vocabulary) -> str | None:
    """The caption instruction appended for a vocabulary register (one of
    CAPTION_VOCABULARIES: 'explicit' | 'clinical' | 'safe'), or None for '' / an unknown
    value. Shared with the image bank so its NSFW lane reuses the dataset's exact register
    text — 'explicit' only spells acts out when paired with an abliterated vision model,
    and the output cleaners still run, so it changes wording, never what binds."""
    return _VOCABULARY_INSTRUCTION.get((vocabulary or '').strip().lower())


def caption_preset_instructions(vocabulary=None, length=None) -> str | None:
    """The combined preset block (vocabulary register, then length) for a run that has no
    per-dataset options to read — the image bank's per-run lane. None when neither is set,
    so a call without presets appends nothing at all."""
    parts = _caption_preset_parts(vocabulary, length)
    return '\n\n'.join(parts) if parts else None


def preview_caption(user_id, dataset_id, image_id, *, backend=None, ollama_model='',
                    vocabulary=None, length=None, instructions=None,
                    should_cancel=None) -> dict:
    """Caption ONE dataset image with a candidate config and return the text WITHOUT
    persisting it — the Caption Lab's ephemeral A/B probe. Reuses caption_paths(), so the
    engine/model/GPU serialization contract is identical to the batch pass.

    backend      : '' / None → global default; else one of _CAPTION_BACKENDS ('none' is
                   rejected here — a preview with captioning disabled makes no sense).
    vocabulary   : '' / None → the model's own wording; else an _CAPTION_VOCABULARIES
                   preset, appended as an instruction exactly like the dataset options.
    length       : '' / None → standard (nothing appended); else a _CAPTION_LENGTHS
                   preset ('concise' | 'detailed'), on an axis orthogonal to vocabulary.
    instructions : free extra instructions, appended after both presets.
    should_cancel: polled by caption_paths at the image boundary (Ollama phase) so the
                   existing Stop path can abort a preview cleanly.

    Returns {caption, chars, duration_ms, cancelled}. Raises ValueError (bad image/config)
    → 400, RuntimeError (engine unavailable) → 409, GpuBusyError → 503 (via the route's
    vision window). Never writes to the DB or the filesystem."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    img = db.session.get(FaceDatasetImage, image_id)
    if not img or img.dataset_id != ds.id or not img.filename:
        raise ValueError('image not found')
    path = _img_path(img)
    if not os.path.isfile(path):
        raise ValueError('image file missing on disk')
    backend = (backend or '').strip().lower() or None
    if backend and backend not in _CAPTION_BACKENDS:
        raise ValueError(f'invalid captioning backend: {backend}')
    if backend == 'none':
        raise ValueError('captioning is disabled for this candidate')
    vocab = (vocabulary or '').strip().lower() or None
    if vocab and vocab not in _CAPTION_VOCABULARIES:
        raise ValueError(f'invalid caption vocabulary: {vocab}')
    size = (length or '').strip().lower() or None
    if size and size not in _CAPTION_LENGTHS:
        raise ValueError(f'invalid caption length: {size}')
    extra = _compose_preview_instructions(vocab, instructions, size)
    ollama_model = normalize_ollama_model_ref(
        ollama_model, allow_empty=True) or None
    started = time.perf_counter()
    out = caption_paths([path], backend=backend, ollama_model=ollama_model,
                        extra_instructions=extra, should_cancel=should_cancel)
    duration_ms = int((time.perf_counter() - started) * 1000)
    caption = (out.get(path) or '').strip()
    # A stop consumed before the (single) image ran leaves no caption — surface it so the
    # Lab card reads "cancelled" rather than a misleading empty result.
    cancelled = bool(not caption and should_cancel and should_cancel())
    return {'caption': caption, 'chars': len(caption),
            'duration_ms': duration_ms, 'cancelled': cancelled}


# --- Short-caption derivation (ai-toolkit dual long+short captioning) --------
# When a dataset opts into dual captions, ai-toolkit trains each image with BOTH the long
# and the short caption in the same step (short_and_long_captions doubles the batch — see
# BaseSDTrainProcess.process_general_training_batch in the installed toolkit). The short is
# DERIVED from the already-stored long via a text-only Ollama pass (no vision decode, no
# second model, no GPU-heavy image work), then run through the SAME kind omission the long
# went through so shortening can never reintroduce a banned identity/concept/aesthetic term.

_SHORTEN_BASE = (
    'Rewrite the following image caption as a much SHORTER caption: one concise sentence, '
    'or a few key comma-separated phrases, naming only the most salient clearly-visible '
    'elements. Do NOT add any detail that is not already present. Do NOT explain yourself '
    'or add commentary. Reply with ONLY the short caption.\n')


def _shorten_prompt(ds, long_caption) -> str:
    """Text-only shortening prompt whose kind rule MIRRORS the long-caption omission:
    character omits identity, concept omits the recurring element, style omits the look."""
    if is_style(ds):
        rule = ('Describe visible CONTENT only (subject, action, setting). Never name any '
                'aesthetic, medium, art style, or artist.\n')
    elif is_concept(ds):
        rule = (f'Never mention or describe this recurring element: '
                f'{(ds.concept_desc or "").strip()}. Keep it fully omitted.\n')
    else:
        rule = ("Never mention or describe the person's identity, face, or facial "
                'features.\n')
    return f'{_SHORTEN_BASE}{rule}\nCAPTION:\n{(long_caption or "").strip()}\n'


def _scrub_short_like_long(ds, text, mode) -> str:
    """Apply the SAME deterministic kind omission a long caption gets — reusing the
    existing scrubbers, none of which touch the GPU: style content-only strip, concept
    ban-list clause-scrub (describe=None → mechanical net only), character identity drop."""
    t = (text or '').strip().strip('"').strip()
    if not t:
        return ''
    if is_style(ds):
        # Same net as the long pass: the shortening LLM is told not to name the medium,
        # and like the captioner it sometimes opens with one anyway.
        return style_content_caption(ds, drop_style_lead_in(t))
    if is_concept(ds):
        leak_re = _concept_terms_re(_get_concept_terms(ds, describe=None))
        return _enforce_concept_omission(t, leak_re, b'', (ds.concept_desc or '').strip(),
                                         describe=None) or ''
    cleaner = drop_identity_tags if mode == 'booru' else drop_identity_sentences
    return cleaner(t, body=is_body_fidelity(ds)) or ''


def derive_short_captions(user_id, dataset_id, image_ids=None, force=False, mode=None,
                          token=None, generate=None) -> int:
    """Derive caption_short from each kept image's stored long caption (text-only Ollama,
    kind omission preserved). No-op unless the dataset has dual captions enabled.

    `force=False` fills only images that still lack a short; `force=True` overwrites (the
    re-caption path — a fresh long implies a fresh short). `mode` matches the long pass
    (booru for SDXL, else prose). `generate` is the text seam (injected in tests); None →
    the real generate_text_ollama with the batch keep-alive + one unload at the end.

    Best-effort per image: an empty/failed generation (or one scrubbed down to nothing)
    leaves the short as-is — a still-missing short degrades to the long caption at export.
    Returns the number of shorts written."""
    ds = get_dataset(user_id, dataset_id)
    if not ds or not dual_captions_enabled(ds):
        return 0
    ttype = (getattr(ds, 'train_type', None) or 'zimage').lower()
    mode = (mode or ('booru' if ttype == 'sdxl' else 'prose')).lower()
    q = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
    if image_ids is not None:
        ids = [int(i) for i in image_ids
               if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
        if not ids:
            return 0
        q = q.filter(FaceDatasetImage.id.in_(ids))
    rows = [i for i in q.all() if (i.caption or '').strip()]
    if not force:
        rows = [i for i in rows if not (i.caption_short or '').strip()]
    if not rows:
        return 0
    if generate is None:
        from .vision_ollama import generate_text_ollama, unload_vision_model
        # Same model override as the long-caption pass so the short is derived by (and the
        # VRAM freed for) the model the dataset actually captions with.
        omodel = caption_options(ds).get('ollama_model') or None
        def gen(p):
            return generate_text_ollama(p, num_predict=400, model=omodel,
                                        keep_alive=_VISION_BATCH_KEEPALIVE)
        def _unload():
            return unload_vision_model(model=omodel)
    else:
        gen = generate
        def _unload():
            return None
    # When no caller owns an indicator (the /caption route runs shorts as a follow-up
    # pass), own one here so this loop is visible AND Stop-able like the long pass: the
    # kind matches (caption/recaption) so request_cancel finds it and the amber banner
    # names it. A caller-supplied token means the long pass still owns the indicator.
    own_token = None
    if token is None:
        own_token = dataset_activity.begin(dataset_id, 'recaption' if force else 'caption',
                                           total=len(rows),
                                           detail=f'Deriving {len(rows)} short caption(s)…')
        token = own_token
    n = 0
    vanished = 0
    # Ids, not ORM objects: see _live_image_row. The commit below expires every
    # row still to come, and this loop pays a text generation per image — long
    # enough for the user to delete a tile from the grid meanwhile, which used to
    # kill the pass on the next `img.caption` read.
    row_ids = [img.id for img in rows]
    try:
        for image_id in row_ids:
            if dataset_activity.cancel_requested(dataset_id):
                break   # graceful stop at an image boundary (see caption_images)
            dataset_activity.bump(token)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            short = _scrub_short_like_long(ds, gen(_shorten_prompt(ds, img.caption)), mode)
            if not short:
                continue
            img = _live_image_row(image_id)
            if img is None:      # deleted DURING its own generation
                vanished += 1
                continue
            # The SHORT gets its own stamp, on its own column: this pass derives it
            # with a text model while the long caption above it may well have been
            # typed by hand. One origin for the two would mislabel one of them.
            caption_origin.stamp(img, _cap_caption(short) or None,
                                 caption_origin.OLLAMA, field='caption_short')
            db.session.commit()
            n += 1
    finally:
        _unload()
        if own_token is not None:
            dataset_activity.end(own_token)
    if vanished:
        logger.info('short captions: %s image(s) were deleted while the pass ran, '
                    'skipped', vanished)
    return n


# --- Face similarity scoring (InsightFace antelopev2, CPU subprocess) -------
# WHICH ROWS A FACE PASS SCORES.
#   'keep'    = the curated set. The original (and, until now, the only) scope.
#   'pending' = the TRIAGE PILE: images that have landed but carry no ✓/✕ yet —
#               i.e. exactly the freshly GENERATED variations. Those are the ones
#               whose identity nobody can judge by eye ("is this still her?" on a
#               grainy party photo is not an eyeball question), and 🎯 Auto-triage
#               (DatasetGrid.jsx) has ALWAYS selected on `status === 'pending' &&
#               scorable` — a set this pass could never produce while it filtered
#               on 'keep' alone. The bar was built against a scope that did not
#               exist; widening it here is the whole wiring.
# 'reject'/'failed' stay out: scoring an image the user already threw away, or one
# with no file, is GPU-free but not free — and it would re-arm auto-triage on rows
# it must never touch.
FACE_SCORING_STATUSES = ('keep', 'pending')


def _face_score_content_revision(path):
    """Return the current (content signature, stat) pair, or None on a race.

    The signature makes edits that happen to preserve byte length detectable;
    the second stat read rejects a file changed while it was being fingerprinted.
    """
    from . import run_snapshot

    stat_key = run_snapshot._stat_key(path)
    if stat_key is None:
        return None
    signature = run_snapshot._content_sig(path)
    if not signature or run_snapshot._stat_key(path) != stat_key:
        return None
    return signature, stat_key


def face_scoring_counts(imgs):
    """{'total', 'unscored'} over an ALREADY-LOADED image list — pure, no query,
    so `dataset_payload` pays nothing for it. `unscored` counts rows the pass has
    never written a verdict for (face_state is NULL), which is what the button
    label needs to promise honest work ("Analyze 42 faces") instead of a silent
    no-op on a dataset that is already fully scored."""
    rows = [i for i in (imgs or [])
            if i.filename and i.status in FACE_SCORING_STATUSES]
    return {'total': len(rows),
            'unscored': sum(1 for i in rows if i.face_state is None)}


def face_scoring_rows(dataset_id):
    """The rows a face pass would score, straight from the DB."""
    return (FaceDatasetImage.query
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                    FaceDatasetImage.filename.isnot(None))
            .all())


def analyze_faces(user_id, dataset_id) -> dict:
    """Score les images GARDEES **et la pile de triage** vs la reference
    (InsightFace antelopev2, CPU subprocess) — cf. FACE_SCORING_STATUSES.
    Persiste face_score (cosinus brut, None si non note) + face_state. AUCUNE
    suppression, aucune decision : la passe ecrit un chiffre, c'est 🎯 Auto-triage
    qui agit dessus. Tourne sur CPU -> pas de fenetre GPU. Retourne {state: count}."""
    _guard_not_bank_export(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Checked BEFORE the reference guard on purpose: an anime dataset with no
    # reference must hear the useful thing ("this tool can't read a drawn face"),
    # not "set a reference photo first" — which would send the user off to fix
    # something that would not have helped. Returned as a scoring_error rather
    # than raised so the existing toast path states the reason instead of the pass
    # disappearing silently — a refusal that does not explain itself is the very
    # failure mode this gate exists to remove.
    blocked = face_scoring_block_reason(ds)
    if blocked:
        return {}, {'kind': 'subject_not_photographic', 'detail': blocked}
    if not ds.ref_filename:
        raise ValueError('reference photo missing')
    ref_path = _ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference photo missing')
    rows = face_scoring_rows(dataset_id)
    by_path = {}
    for img in rows:
        p = _img_path(img)
        if os.path.exists(p):
            by_path[p] = img
    try:
        from .face_similarity import score_dataset_faces
    except ImportError:
        raise RuntimeError('face scoring service not configured/available yet')
    # scoring_error ({kind, detail} | None) remonte jusqu'au toast : un scorer
    # cassé doit dire POURQUOI, pas « 0 analyzed » en vert.
    # Persistent indicator (survives reload). The scoring is a single CPU
    # subprocess, but NOT an opaque one: it prints "[face] i/N" for every image it
    # finishes, and the service now streams those into this counter — the bar used
    # to sit at 0 for the whole (multi-minute) pass and then fill in one jump,
    # which is indistinguishable from a hung pass. try/finally clears the
    # indicator even if scoring raises.
    score_lock = _face_scoring_lock(ds.id)
    if not score_lock.acquire(blocking=False):
        return {}, _face_scoring_busy_error()

    # Stamp every eligible file before inference.  A crop/mirror/rotate clears
    # this pair, making the final per-row write below fail closed if pixels move.
    reserved_by_path = {}
    try:
        from sqlalchemy import update
        for p, img in by_path.items():
            revision = _face_score_content_revision(p)
            if revision is None:
                continue
            content_sig, content_sig_stat = revision
            reservation = db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == img.id,
                       FaceDatasetImage.dataset_id == ds.id,
                       FaceDatasetImage.filename == img.filename,
                       FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                       _nullable_equals(FaceDatasetImage.content_sig, img.content_sig),
                       _nullable_equals(FaceDatasetImage.content_sig_stat,
                                        img.content_sig_stat))
                .values(content_sig=content_sig, content_sig_stat=content_sig_stat)
                .execution_options(synchronize_session=False))
            if reservation.rowcount == 1:
                reserved_by_path[p] = (img.id, img.filename,
                                       content_sig, content_sig_stat)
        db.session.commit()
    except Exception:
        db.session.rollback()
        score_lock.release()
        raise
    if not reserved_by_path:
        score_lock.release()
        return {}, None

    try:
        token = dataset_activity.begin(dataset_id, 'analyze_faces', total=len(reserved_by_path))
    except Exception:
        score_lock.release()
        raise

    try:
        results, scoring_error = score_dataset_faces(
            ref_path, list(reserved_by_path.keys()),
            on_progress=lambda done, total: dataset_activity.progress(
                token, done=done, total=total))
        counts = {}
        # The counter is already at N: the persist loop below is a fraction of the
        # pass (no model load, no inference), so it does NOT bump — doing so would
        # count every image twice and take the bar past its own total.
        for p, (image_id, filename, content_sig, content_sig_stat) in reserved_by_path.items():
            r = results.get(p)
            if not r:
                continue
            if _face_score_content_revision(p) != (content_sig, content_sig_stat):
                continue
            write = db.session.execute(
                update(FaceDatasetImage)
                .where(FaceDatasetImage.id == image_id,
                       FaceDatasetImage.dataset_id == ds.id,
                       FaceDatasetImage.filename == filename,
                       FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                       FaceDatasetImage.content_sig == content_sig,
                       FaceDatasetImage.content_sig_stat == content_sig_stat)
                .values(face_state=r.get('state'), face_score=r.get('sim'))
                .execution_options(synchronize_session=False))
            if write.rowcount != 1:
                # Another request won the row after inference.  It is newer than
                # this pass, so leave it exactly as it is.
                db.session.rollback()
                continue
            db.session.commit()
            state = r.get('state')
            counts[state] = counts.get(state, 0) + 1
        return counts, scoring_error
    finally:
        dataset_activity.end(token)
        score_lock.release()


def analyze_image_face(user_id, image_id):
    """Score one owned dataset image against its dataset reference on CPU only.

    The single-image action deliberately uses the same scorer contract as the
    batch pass. Operational scorer failures are returned with the untouched
    current fields, while invalid image/dataset input remains a validation
    error for the route to map.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    ds = get_dataset(user_id, img.dataset_id)
    if not ds:
        return None

    def _result(scoring_error=None, stale=False, row=None):
        row = img if row is None else row
        result = {'image_id': row.id, 'face_state': row.face_state,
                  'face_score': row.face_score, 'scoring_error': scoring_error}
        if stale:
            result['stale'] = True
        return result

    def _stale_result():
        db.session.expire_all()
        fresh = _owned_image(user_id, image_id)
        if not fresh:
            return None
        return _result(stale=True, row=fresh)


    # Match the batch behaviour: explain the photographic-subject gate before
    # asking for a reference that could never make this kind of dataset scorable.
    blocked = face_scoring_block_reason(ds)
    if blocked:
        return _result({'kind': 'subject_not_photographic', 'detail': blocked})
    if not ds.ref_filename or not os.path.isfile(_ref_path(ds)):
        raise ValueError('reference photo missing')
    if img.status not in FACE_SCORING_STATUSES:
        raise ValueError('image is not eligible for face scoring')
    filename_snapshot = img.filename
    if not img.filename or not os.path.isfile(_img_path(img)):
        raise ValueError('image file missing')

    ref_path = _ref_path(ds)
    image_path = _img_path(img)
    score_lock = _face_scoring_lock(ds.id)
    if not score_lock.acquire(blocking=False):
        return _result(_face_scoring_busy_error())
    try:
        try:
            from . import face_similarity
        except ImportError:
            return _result({'kind': 'unavailable',
                            'detail': 'face scoring service not configured/available yet'})

        # Reserve the content identity before launching the subprocess.  A pixel
        # edit clears this cache pair, so the final write below can never promote
        # a score calculated for an earlier version of the same filename.
        revision = _face_score_content_revision(image_path)
        if revision is None:
            return _stale_result()
        content_sig, content_sig_stat = revision
        previous_sig = img.content_sig
        previous_stat = img.content_sig_stat
        from sqlalchemy import update
        reservation = db.session.execute(
            update(FaceDatasetImage)
            .where(FaceDatasetImage.id == img.id,
                   FaceDatasetImage.dataset_id == ds.id,
                   FaceDatasetImage.filename == filename_snapshot,
                   FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                   _nullable_equals(FaceDatasetImage.content_sig, previous_sig),
                   _nullable_equals(FaceDatasetImage.content_sig_stat, previous_stat))
            .values(content_sig=content_sig, content_sig_stat=content_sig_stat)
            .execution_options(synchronize_session=False))
        if reservation.rowcount != 1:
            db.session.rollback()
            return _stale_result()
        db.session.commit()
        db.session.expire(img)

        # Recheck after the reservation: do not start the expensive process for
        # a file that changed while its identity was being recorded.
        from . import run_snapshot
        if run_snapshot._stat_key(image_path) != content_sig_stat:
            return _stale_result()

        try:
            results, scoring_error = face_similarity.score_dataset_faces(
                ref_path, [image_path])
        except Exception as e:
            logger.warning('single face scoring failed for image %s: %s', image_id, e)
            return _result({'kind': 'failed', 'detail': str(e) or 'face scoring failed'})
        if scoring_error:
            return _result(scoring_error)
        scored = results.get(image_path) if isinstance(results, dict) else None
        if not isinstance(scored, dict) or not scored.get('state'):
            return _result({'kind': 'failed',
                            'detail': 'face scorer returned no result for this image'})

        # A stat check is cheap but coarse on some filesystems; re-reading the
        # content signature here also catches a same-size edit in the same second.
        if _face_score_content_revision(image_path) != (content_sig, content_sig_stat):
            return _stale_result()

        write = db.session.execute(
            update(FaceDatasetImage)
            .where(FaceDatasetImage.id == img.id,
                   FaceDatasetImage.dataset_id == ds.id,
                   FaceDatasetImage.filename == filename_snapshot,
                   FaceDatasetImage.status.in_(FACE_SCORING_STATUSES),
                   FaceDatasetImage.content_sig == content_sig,
                   FaceDatasetImage.content_sig_stat == content_sig_stat)
            .values(face_state=scored['state'], face_score=scored.get('sim'))
            .execution_options(synchronize_session=False))
        if write.rowcount != 1:
            db.session.rollback()
            return _stale_result()
        db.session.commit()
        db.session.expire(img)
        return _result()
    finally:
        score_lock.release()


# --- Watermark auto-correction (V1) ----------------------------------------
# Scraped images often carry an OVERLAID watermark (site logo, URL, @username, studio
# text) that the LoRA would learn. V1 = detect (Qwen3-VL bbox) then route removal by
# cost/risk: CROP a border-band mark (PIL pur, invents no pixel), LaMa-inpaint a small
# off-center mark (non-generative, only masked pixels change), else leave it for manual
# review. NO YOLO, NO generative inpaint -- those are V2.
WATERMARK_BORDER_BAND = 0.20       # a mark within this outer strip is croppable
WATERMARK_MAX_INPAINT_AREA = 0.10  # bbox area above this fraction -> manual review
WATERMARK_MIN_SIDE = 768           # never crop a side below this (ai-toolkit only downscales)
WATERMARK_REGION_LIMIT = 32
WATERMARK_REGION_MIN_SIDE = 0.005


def normalize_watermark_regions(value, *, allow_null=True) -> list[list[float]] | None:
    if value is None:
        if allow_null:
            return None
        raise ValueError('regions must be a list')
    if not isinstance(value, list) or len(value) > WATERMARK_REGION_LIMIT:
        raise ValueError('regions must contain at most 32 boxes')
    out = []
    for box in value:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('each region must be [x1,y1,x2,y2]')
        try:
            invalid_number = any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) for v in box
            )
        except OverflowError:
            invalid_number = True
        if invalid_number:
            raise ValueError('region coordinates must be finite numbers')
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('region coordinates must be ordered within [0,1]')
        min_side = Decimal(str(WATERMARK_REGION_MIN_SIDE))
        if (Decimal(str(x2)) - Decimal(str(x1)) < min_side
                or Decimal(str(y2)) - Decimal(str(y1)) < min_side):
            raise ValueError('region is too small')
        out.append([round(v, 4) for v in (x1, y1, x2, y2)])
    return out


def set_watermark_regions(user_id, dataset_id, image_id, regions) -> dict | None:
    """Atomically replace a detected image's manual watermark-region override."""
    _guard_not_bank_export(dataset_id)
    owned_query = (FaceDatasetImage.query
                   .join(FaceDataset, FaceDatasetImage.dataset_id == FaceDataset.id)
                   .filter(FaceDatasetImage.id == image_id,
                           FaceDatasetImage.dataset_id == dataset_id,
                           FaceDataset.user_id == str(user_id)))
    img = owned_query.one_or_none()
    if not img:
        return None
    if img.watermark_state != 'detected':
        raise RuntimeError('image is no longer detected')
    normalized = normalize_watermark_regions(regions)
    stored = json.dumps(normalized) if normalized is not None else None
    updated = (FaceDatasetImage.query
               .filter_by(id=img.id, watermark_state='detected')
               .update({'watermark_regions': stored}, synchronize_session=False))
    if updated != 1:
        db.session.rollback()
        if owned_query.one_or_none() is None:
            return None
        raise RuntimeError('image is no longer detected')
    db.session.commit()
    return _watermark_regions_payload(img)


def _route_watermark(bbox, W, H, *, min_side=WATERMARK_MIN_SIDE, allow_crop=True):
    """Decide how to remove the watermark at normalized `bbox` (x1,y1,x2,y2) on a
    W x H image. Returns ('crop', (left, top, right, bottom)) | ('lama', None) |
    ('review', None). PURE function (no I/O) so the routing is unit-testable.

    CROP (default, invents no pixel) when the mark sits ENTIRELY inside one outer
    border band (<= WATERMARK_BORDER_BAND of the side) AND the resulting crop keeps
    BOTH sides >= min_side -- we cut the band up to the mark's INNER edge. LaMa when
    the mark is small (area <= WATERMARK_MAX_INPAINT_AREA) and does not straddle the
    image center. Otherwise (large, or on the central subject with no safe crop) ->
    manual review, never a risky auto-edit.

    allow_crop=False (the "Allow auto-crop" preference turned off, or a per-image
    "force inpaint" from the review lightbox) SKIPS the crop branches entirely: a
    border mark then falls through to the inpaint/review logic below and is repainted
    (LaMa/Klein per the chosen engine) instead of cropped. Nothing else changes -- the
    min_side guard still governs whether crop is ever offered when it IS allowed."""
    x1, y1, x2, y2 = bbox
    px1, py1, px2, py2 = x1 * W, y1 * H, x2 * W, y2 * H
    band = WATERMARK_BORDER_BAND
    # Border-band crops, tried top/bottom/left/right. The kept box is (left,top,right,bottom).
    if allow_crop:
        if y2 <= band and (H - py2) >= min_side and W >= min_side:        # top band
            return 'crop', (0, int(round(py2)), W, H)
        if y1 >= 1 - band and py1 >= min_side and W >= min_side:          # bottom band
            return 'crop', (0, 0, W, int(round(py1)))
        if x2 <= band and (W - px2) >= min_side and H >= min_side:        # left band
            return 'crop', (int(round(px2)), 0, W, H)
        if x1 >= 1 - band and px1 >= min_side and H >= min_side:          # right band
            return 'crop', (0, 0, int(round(px1)), H)
    # Not a safe border crop (off-band, or the crop would fall below min_side).
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    overlaps_center = (x1 < 0.5 < x2) and (y1 < 0.5 < y2)
    if area <= WATERMARK_MAX_INPAINT_AREA and not overlaps_center:
        return 'lama', None
    return 'review', None


def _preserve_original(path) -> bool:
    """Durably preserve ``path`` before any destructive watermark edit.

    Returns ``True`` only when a usable sibling ``.orig`` exists.  A direct
    ``copy2(path, backup)`` can leave a truncated backup if the disk fills up;
    treating that as success would allow a subsequent crop/inpaint to destroy
    the only intact master.  Copy through a sibling temporary file and promote
    it atomically instead.  Existing backups are deliberately never overwritten
    (they are the older, recoverable master from a prior clean pass).
    """
    stem, ext = os.path.splitext(path)
    backup = f'{stem}.orig{ext or ".webp"}'
    if os.path.exists(backup):
        # A prior interrupted *old* implementation could have left a partial
        # .orig behind. Do not assume its mere existence makes an edit safe.
        try:
            if not os.path.isfile(backup) or os.path.getsize(backup) <= 0:
                raise OSError('existing backup is empty or not a regular file')
            with Image.open(backup) as check:
                check.verify()
            return True
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            logger.error('watermark: refusing edit; existing backup is unusable for %s: %s',
                         path, exc)
            return False

    staged_backup = None
    try:
        fd, staged_backup = tempfile.mkstemp(
            prefix=f'.{os.path.basename(backup)}.preserve-', suffix='.part',
            dir=os.path.dirname(path),
        )
        os.close(fd)
        shutil.copy2(path, staged_backup)
        # ``copy2`` returning does not guarantee that the data was flushed to
        # disk. Fsync the staged bytes before making the backup visible.
        # Windows requires a writable descriptor for fsync; the staged copy is
        # complete at this point, so ``rb+`` does not alter its bytes.
        with open(staged_backup, 'rb+') as handle:
            os.fsync(handle.fileno())
        with Image.open(staged_backup) as check:
            check.verify()
        os.replace(staged_backup, backup)
        staged_backup = None
        return True
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logger.error('watermark: could not preserve original %s; refusing edit: %s', path, exc)
        return False
    finally:
        if staged_backup:
            try:
                os.unlink(staged_backup)
            except OSError:
                pass


def _stage_oriented_watermark_edit(path) -> str | None:
    """Create an upright, metadata-free sibling for a destructive watermark pass.

    Browser/VLM boxes live in visual orientation, whereas the master may still
    carry camera EXIF.  LaMa and Klein edit a path in place, so they must receive
    a disposable, EXIF-transposed file.  The caller promotes this sibling only
    after the engine reports success; a failure must leave the master byte-for-
    byte untouched.
    """
    staged = None
    try:
        with Image.open(path) as opened:
            fmt = image_encoding.format_for_path(path, opened)
            opened.load()
            icc = _valid_icc_profile(opened.info.get('icc_profile'))
            oriented = ImageOps.exif_transpose(opened)
            payload = io.BytesIO()
            image_encoding.save_edit(oriented, payload, fmt, image_encoding.LOSSLESS,
                                     icc_profile=icc)
        suffix = os.path.splitext(path)[1] or '.webp'
        fd, staged = tempfile.mkstemp(
            prefix=f'.{os.path.basename(path)}.wm-orient-', suffix=suffix,
            dir=os.path.dirname(path),
        )
        with os.fdopen(fd, 'wb') as fh:
            fh.write(payload.getvalue())
            fh.flush()
            os.fsync(fh.fileno())
        return staged
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logger.warning('watermark: could not stage EXIF-oriented edit for %s: %s', path, exc)
        if staged:
            try:
                os.unlink(staged)
            except OSError:
                pass
        return None


def _promote_staged_watermark_edit(staged_path, live_path) -> bool:
    """Atomically replace a master only after a staged engine result verifies."""
    try:
        expected = image_encoding.format_for_path(live_path)
        with Image.open(staged_path) as check:
            if (check.format or '').upper() != expected:
                raise OSError('staged result format does not match its extension')
            check.verify()
        os.replace(staged_path, live_path)
        return True
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        logger.warning('watermark: could not promote staged edit for %s: %s', live_path, exc)
        return False


def _discard_staged_watermark_edit(staged_path) -> None:
    try:
        os.unlink(staged_path)
    except OSError:
        pass


def _apply_watermark_crop(path, box) -> bool:
    """Crop `path` to `box` (left,top,right,bottom px) WITHOUT resizing -- the whole
    point of the crop route is that it invents no pixel (the aspect-ratio change is
    absorbed by ai-toolkit's bucketing). Returns bool.

    Because nothing is resampled here, PNG/WebP/BMP retain the surviving pixels
    losslessly under ``image_encoding``. JPEG has no lossless write path and is
    deliberately re-encoded at the documented high-quality 4:4:4 setting. It used
    to re-save every format as WebP q92, which quietly re-compressed the ENTIRE
    image to remove a band at its edge."""
    try:
        with Image.open(path) as opened:
            fmt = image_encoding.format_for_path(path, opened)
            opened.load()
            icc = _valid_icc_profile(opened.info.get('icc_profile'))
            # `box` is visual/browser (and VLM) space. If called directly rather
            # than through clean_watermarks, keep the same orientation contract.
            im = ImageOps.exif_transpose(opened)
    except (OSError, ValueError):
        return False
    box = (max(0, int(box[0])), max(0, int(box[1])),
           min(im.width, int(box[2])), min(im.height, int(box[3])))
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return False
    out = io.BytesIO()
    image_encoding.save_edit(im.crop(box), out, fmt, image_encoding.LOSSLESS,
                             icc_profile=icc)
    write_image_atomic(path, out.getvalue())
    return True


def detect_watermarks(user_id, dataset_id, *, include_dismissed=False, backend=None,
                      should_cancel=None, report=None):
    """Scan the KEPT images for an overlaid watermark and persist watermark_state
    ('detected'|'none') + watermark_bbox (JSON normalized box). Returns
    {'detected': n, 'none': n, 'checked': n} — that dict is the route's response
    shape and four tests pin it EXACTLY, so anything else the caller needs travels
    through ``report`` (a dict this fills in) or the route's own keys.

    TWO ROUTES, and which one runs is decided by ``watermark_detect.backend`` —
    the same setting the bank reads, resolved by the same function, because two
    screens obeying two rules is the defect this replaced:

    * the dedicated DETECTOR extra (SigLIP2 ranks, Grounding DINO locates) — no
      Ollama needed, ~0.14 s/image, and it writes a SCORE;
    * the vision model (Qwen3-VL), exactly as before — one chat question per
      image, ~1.7 s, no score. This is what 'auto' picks when the extra is not
      installed, so an untouched install behaves identically to yesterday.

    ``should_cancel`` is polled BETWEEN images: what was already judged is
    committed and kept, the rest simply stays unscanned and a later run finishes
    it (detect looks at every kept row on every pass).

    Images the user already judged NOT a watermark ('dismissed', a false positive
    ruled out in the review lightbox) are SKIPPED so a re-run never re-flags them --
    that's the anti-frustration point. Pass include_dismissed=True to re-examine them
    (a deliberate "check everything again"). CALLER decides on the GPU-exclusive
    vision window: the vision route always needs it, the detector only when it
    actually runs on CUDA."""
    _guard_not_bank_export(dataset_id)
    from . import watermark_detector
    resolution = (backend if isinstance(backend, dict)
                  else watermark_detector.resolve_backend(backend))
    if report is not None:
        report.update(resolution)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return {'detected': 0, 'none': 0, 'checked': 0}
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None)).all())
    # Ids, not ORM objects: see _live_image_row. Both loops commit per image, so
    # every row they have not reached is expired, and deleting a tile from the
    # grid mid-scan used to kill the scan.
    row_ids = [img.id for img in rows]
    if resolution['backend'] == 'detector':
        return _detect_watermarks_detector(
            dataset_id, row_ids, include_dismissed=include_dismissed,
            should_cancel=should_cancel, report=report)
    return _detect_watermarks_vision(
        dataset_id, row_ids, include_dismissed=include_dismissed,
        should_cancel=should_cancel, report=report)


def _detect_watermarks_vision(dataset_id, row_ids, *, include_dismissed,
                              should_cancel, report):
    """The original Qwen3-VL pass, unchanged except for the cancel poll."""
    try:
        from .vision_ollama import describe_image_ollama, unload_vision_model
    except ImportError:
        raise RuntimeError('vision (Ollama) service not configured/available yet')
    counts = {'detected': 0, 'none': 0, 'checked': 0}
    # Deliberately NOT a key in `counts`: that dict is this route's response
    # shape and four tests pin it exactly. A counter that is zero on every
    # ordinary run does not justify changing an API contract — and surfacing it
    # usefully would mean a UI decision, not a silent extra field. Logged below.
    vanished = 0
    stopped = False
    # Persistent progress indicator (survives a page reload); try/finally clears it
    # even if the vision pass raises → no phantom "Scanning…" spinner.
    token = dataset_activity.begin(dataset_id, 'watermark_detect', total=len(row_ids))
    try:
        for i, image_id in enumerate(row_ids):
            # Between images, never inside one: the current inference finishes,
            # everything already committed stays, and the pass unwinds through
            # the SAME cleanup as a normal end (model unload, indicator end).
            if should_cancel and should_cancel():
                stopped = True
                break
            dataset_activity.progress(token, done=i + 1)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            # Dismissed = a confirmed false positive; don't waste a vision call re-asking
            # (and never silently re-flag it) unless the caller opts back in.
            if not include_dismissed and img.watermark_state == 'dismissed':
                continue
            path = _img_path(img)
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as fh:
                raw = describe_image_ollama(fh.read(), WATERMARK_BBOX_PROMPT, num_predict=400,
                                            prefer_json=True, fmt='json',
                                            keep_alive=_VISION_BATCH_KEEPALIVE)
            if not (raw or '').strip():
                # Vision unreachable/empty != "no watermark" (same reasoning as
                # classify_images): leave the state UNTOUCHED (retry possible) instead
                # of falsely marking every image clean when Ollama is just down.
                continue
            img.watermark_regions = None
            # Stamp WHICH route ruled, on every row this pass touches — a dataset
            # can hold verdicts from both (promotion carries a bank's across).
            img.watermark_source = 'vision'
            img.watermark_score = None          # this route has no score
            bbox = _parse_watermark_bbox(raw)
            if bbox:
                img.watermark_state = 'detected'
                img.watermark_bbox = json.dumps([round(v, 4) for v in bbox])
                counts['detected'] += 1
            else:
                img.watermark_state = 'none'
                img.watermark_bbox = None
                counts['none'] += 1
            counts['checked'] += 1
            db.session.commit()
    finally:
        unload_vision_model()  # rend la VRAM a ComfyUI en fin de batch
        dataset_activity.end(token)
    if vanished:
        logger.info('watermark detect: %s image(s) were deleted while the pass ran, '
                    'skipped', vanished)
    if report is not None:
        # located == detected here: the vision route never flags without a box
        # (no box parsed IS the "clean" answer).
        report.update({'stopped': stopped, 'located': counts['detected'],
                       'unlocated': 0, 'errors': 0})
    return counts


def _detect_watermarks_detector(dataset_id, row_ids, *, include_dismissed,
                                should_cancel, report):
    """The same pass run by the dedicated detector extra. Deliberately the same
    SHAPE as the vision pass — same skip rules, same per-image commit, same
    survives-a-deletion discipline — because the two must be interchangeable.

    Two structural differences the caller has to know about:

    * a single child process holds both models (loading them costs ~10 s), so a
      stop travels as a sentinel FILE the child polls between images rather than
      a kill — killing a process mid-forward is how a stop becomes a half-write;
    * this route can legitimately answer "detected, position unknown" when the
      locator finds nothing. The vision route cannot. That row is flagged with a
      NULL bbox, counted apart in ``report['unlocated']``, and the screen says so
      — 🧽 Clean has no box to route on and would otherwise stamp it 'failed'.
    """
    from . import watermark_detector
    counts = {'detected': 0, 'none': 0, 'checked': 0}
    planned = []
    for image_id in row_ids:
        img = _live_image_row(image_id)
        if img is None:
            continue
        if not include_dismissed and img.watermark_state == 'dismissed':
            continue
        path = _img_path(img)
        if not path or not os.path.exists(path):
            continue
        planned.append((image_id, path))
    located = unlocated = errors = vanished = 0
    stopped = False
    if not planned:
        if report is not None:
            report.update({'stopped': False, 'located': 0, 'unlocated': 0, 'errors': 0})
        return counts
    by_path = {}
    for image_id, path in planned:
        # A dataset can hold the same file twice; pop one waiting row per verdict
        # so both do not land on the first row.
        by_path.setdefault(path, []).append(image_id)

    cancel_dir = tempfile.mkdtemp(prefix='lds-wmdet-ds-')
    cancel_file = os.path.join(cancel_dir, 'cancel')

    def _cancelled():
        if not (should_cancel and should_cancel()):
            return False
        try:                                # the child polls for this file
            open(cancel_file, 'wb').close()
        except OSError:
            pass
        return True

    token = dataset_activity.begin(dataset_id, 'watermark_detect', total=len(planned))
    try:
        for path, state, score, regions, _error in watermark_detector.scan(
                [p for _i, p in planned], should_cancel=_cancelled,
                cancel_file=cancel_file):
            dataset_activity.bump(token)
            waiting = by_path.get(path) or []
            image_id = waiting.pop(0) if waiting else None
            img = _live_image_row(image_id) if image_id is not None else None
            if img is None:                 # deleted while it was being analysed
                vanished += 1
                continue
            img.watermark_source = 'detector'
            img.watermark_score = (round(float(score), 4) if score is not None else None)
            if state == 'error':
                # One unreadable file never sinks the pass, and it is NOT "clean":
                # the row keeps whatever state it had so a retry can finish it.
                errors += 1
                db.session.commit()
                continue
            img.watermark_regions = None
            if state == 'detected':
                img.watermark_state = 'detected'
                if regions:
                    # ONE box, the child's first — it orders them
                    # most-peripheral-first precisely because this line takes one
                    # (see the bank's identical write for the full reasoning).
                    img.watermark_bbox = json.dumps(
                        [round(float(v), 4) for v in regions[0][:4]])
                    located += 1
                else:
                    img.watermark_bbox = None
                    unlocated += 1
                counts['detected'] += 1
            else:
                img.watermark_state = 'none'
                img.watermark_bbox = None
                counts['none'] += 1
            counts['checked'] += 1
            db.session.commit()
        stopped = bool(should_cancel and should_cancel())
    except watermark_detector.DetectorUnavailable as e:
        # The extra probed OK but could not actually run (weights half downloaded,
        # a torch that no longer imports there). Everything already judged is
        # committed; say what happened and name the way out instead of failing
        # silently or, worse, marking unscanned rows clean.
        db.session.commit()
        logger.warning('dataset watermark detect: detector unavailable (%s)', e)
        raise RuntimeError(
            f'the watermark detector could not run ({e}). Nothing was mis-flagged — '
            'the images it had not reached are still unscanned. Set Settings ▸ '
            'Captioning & quality ▸ Watermark detection to "Vision model" to finish '
            'the pass without it.') from e
    finally:
        db.session.commit()
        dataset_activity.end(token)
        shutil.rmtree(cancel_dir, ignore_errors=True)
    if vanished:
        logger.info('watermark detect: %s image(s) were deleted while the pass ran, '
                    'skipped', vanished)
    if report is not None:
        report.update({'stopped': stopped, 'located': located,
                       'unlocated': unlocated, 'errors': errors})
    return counts


def dismiss_watermarks(user_id, dataset_id, image_ids):
    """Mark 'detected' images as 'dismissed' -- the user ruled, in the review lightbox,
    that the flag is a FALSE positive. Dismissed images drop the 🚩 badge, leave the
    Clean batch, and are skipped by future detect passes (see detect_watermarks) so
    they're never re-flagged. Only 'detected' rows of THIS dataset transition (ids that
    don't belong / aren't detected are silently ignored, like batch_image_action).
    Returns the number of rows dismissed. The bbox is kept (harmless, and a later
    include_dismissed re-scan overwrites it)."""
    _guard_not_bank_export(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    ids = [int(i) for i in (image_ids or [])
           if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
    if not ids:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, watermark_state='detected')
            .filter(FaceDatasetImage.id.in_(ids)).all())
    for img in rows:
        img.watermark_state = 'dismissed'
        img.watermark_regions = None
    if rows:
        db.session.commit()
    return len(rows)


def _clean_inpaint_engine(route, method):
    """Which inpaint engine a NON-crop image gets, given the batch `method`
    ('auto'|'lama'|'klein'). Crop-routed images always crop (invents no pixel) — this
    only decides how a mark is *repainted*:
      - method 'klein' → Klein for both the small-off-center ('lama') route AND the
        on-subject ('review') route, so review becomes actionable (the whole V2 point);
      - otherwise → LaMa for 'lama', and 'review' stays manual review (unchanged V1)."""
    if method == 'klein':
        return 'klein'
    return 'lama' if route == 'lama' else 'review'


@_serialize_dataset_ingest
def clean_watermarks(user_id, dataset_id, image_ids=None, device='cpu', method='auto',
                     allow_crop=None):
    """Apply the crop/inpaint/review routing to every image marked 'detected'. Returns
    ({'cropped', 'inpainted', 'inpainted_klein', 'needs_review', 'failed', 'skipped'},
    error|None) -- same tuple contract as score_dataset_faces: `error` is None unless an
    inpaint that was ATTEMPTED failed (never a silent swallow). Crop stays in PIL.

    `allow_crop` gates the border-crop route (see _route_watermark). None (the default)
    resolves the persisted `watermark.allow_crop` preference, so a plain call and the
    batch Clean button both honour Settings; the review lightbox passes an explicit
    True/False to force crop or inpaint for ONE image. When False, a border mark is
    repainted (LaMa/Klein per `method`) instead of cropped -- nothing else changes.

    `method` selects the inpaint engine (the batch UI's LaMa|Klein toggle):
      - 'auto'/'lama' → LaMa (fast, non-generative) for small off-center marks; on-subject
        marks stay 'review'. Uses the resolved CPU/GPU `device`; GPU mode is protected by
        the route's exclusive window.
      - 'klein' → masked Flux.2 Klein inpaint + pixel-space composite for the off-center
        AND the on-subject marks (making 'review' actionable). Each image is one serialized
        ComfyUI round-trip; `device` is irrelevant (ComfyUI owns the GPU).

    LaMa absent (probe False) is NOT an error: LaMa-routed images are counted as
    `skipped` (crop still runs) so the UI can nudge "install the ML extras". Klein absent
    is likewise `skipped`.

    image_ids (optional): restrict the pass to this subset -- the review lightbox cleans
    ONE image at a time. The filter still requires watermark_state='detected' AND
    dataset ownership, so a stale/foreign id is a no-op (never touches another dataset,
    never re-edits an already-cleaned image). None = every detected image (bulk button)."""
    from . import watermark_lama, watermark_klein
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # None = "no explicit choice" -> fall back to the persisted preference (default
    # True), so the batch button follows Settings; the lightbox passes a real bool.
    if allow_crop is None:
        allow_crop = bool(cfg.get('watermark.allow_crop'))
    q = (FaceDatasetImage.query
         .filter_by(dataset_id=dataset_id, watermark_state='detected')
         .filter(FaceDatasetImage.filename.isnot(None)))
    if image_ids is not None:
        ids = [int(i) for i in (image_ids or [])
               if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
        q = q.filter(FaceDatasetImage.id.in_(ids or [-1]))   # empty subset -> match nothing
    rows = q.all()
    row_ids = [img.id for img in rows]
    out = {'cropped': 0, 'inpainted': 0, 'inpainted_klein': 0, 'needs_review': 0,
           'failed': 0, 'skipped': 0}
    # NOT a key in `out`: that dict is the route's response shape and existing
    # tests pin it. 'skipped' already means "engine unavailable" and must not be
    # overloaded with "the image no longer exists". Logged at the end instead.
    vanished = 0
    error = None
    lama_ok = watermark_lama.is_available()
    klein_ok = method == 'klein' and watermark_klein.is_available()
    # The Klein model this DATASET runs on — the same pick ✨ improve and Klein
    # generation use. A watermark clean overwrites the image in place, so running
    # it on a model the dataset did not choose is the one lane where the swap
    # cannot be spotted afterwards by comparing to a source.
    klein_model = dataset_klein_model(ds)
    if klein_ok and klein_model:
        # Refuse the WHOLE pass by name, before a single file is touched: every
        # image would fail identically, and a half-cleaned dataset is worse than
        # an untouched one. None (never chose) skips this — nothing was promised.
        from . import klein_edit_helper as keh
        if not keh.klein_model_on_disk(klein_model):
            raise keh.KleinModelGone(klein_model)
    # (image_id, live_path, staged_path, bboxes, manual_regions). An ID, not an
    # ORM row: this list is carried across the whole per-image loop AND across
    # the LaMa batch, which runs for minutes -- by the time the tail loop writes,
    # those rows have been expired by a dozen commits and any one of them may
    # have been deleted from the grid. See _live_image_row.
    lama_pending = []

    def _backup_failed(img, staged_path=None):
        """Record a fail-closed preservation error and discard any edit staging."""
        nonlocal error
        if staged_path:
            _discard_staged_watermark_edit(staged_path)
        # Retain manual regions for a future retry, but never present this row
        # as clean/detected when no recoverable pre-edit master exists.
        img.watermark_state = 'failed'
        out['failed'] += 1
        error = {
            'kind': 'failed',
            'detail': 'could not preserve original; master was left unchanged',
        }

    def _run_klein(img, path, boxes, manual):
        """One serialized Klein inpaint through an upright disposable sibling."""
        nonlocal error
        if not klein_ok:
            out['skipped'] += 1               # leave 'detected' (Klein not ready)
            return
        staged = _stage_oriented_watermark_edit(path)
        if not staged:
            if not manual:                    # retain manual regions for a retry
                img.watermark_state = 'failed'
            out['failed'] += 1
            error = {'kind': 'failed', 'detail': 'could not stage image EXIF orientation'}
            return
        if not _preserve_original(path):
            _backup_failed(img, staged)
            return
        try:
            ok, err = watermark_klein.inpaint_watermark_klein(user_id, staged, boxes,
                                                              klein_model=klein_model)
            if ok and _promote_staged_watermark_edit(staged, path):
                img.watermark_state = 'cleaned'
                if manual:
                    img.watermark_regions = None
                out['inpainted_klein'] += 1
            elif ok:
                if not manual:
                    img.watermark_state = 'failed'
                out['failed'] += 1
                error = {'kind': 'failed', 'detail': 'could not promote staged watermark edit'}
            elif err and err.get('kind') == 'unavailable':
                out['skipped'] += 1
            else:
                if not manual:                # keep manual retry metadata (like LaMa)
                    img.watermark_state = 'failed'
                out['failed'] += 1
                if err:
                    error = err
        finally:
            _discard_staged_watermark_edit(staged)
    # Persistent progress indicator (survives a page reload). The device is included
    # so the UI can honestly state whether ComfyUI is paused for the GPU pass.
    device_label = 'GPU' if device == 'cuda' else 'CPU'
    token = dataset_activity.begin(
        dataset_id, 'watermark_clean', total=len(rows),
        detail=f'Cleaning watermarks on {device_label}…')
    try:
        for i, image_id in enumerate(row_ids):
            dataset_activity.progress(token, done=i + 1)
            img = _live_image_row(image_id)
            if img is None:      # deleted while the pass ran
                vanished += 1
                continue
            path = _img_path(img)
            if img.watermark_regions is not None:
                try:
                    regions = normalize_watermark_regions(
                        _safe_json(img.watermark_regions), allow_null=False,
                    )
                except ValueError as e:
                    out['failed'] += 1
                    error = {'kind': 'failed',
                             'detail': f'invalid watermark regions: {e}'}
                    db.session.commit()
                    continue
                if not regions:
                    out['needs_review'] += 1
                    db.session.commit()
                    continue
                if not os.path.exists(path):
                    out['failed'] += 1
                    db.session.commit()
                    continue
                if method == 'klein':
                    _run_klein(img, path, regions, True)
                    db.session.commit()
                    continue
                if not lama_ok:
                    out['skipped'] += 1
                    db.session.commit()
                    continue
                staged = _stage_oriented_watermark_edit(path)
                if not staged:
                    out['failed'] += 1
                    error = {'kind': 'failed',
                             'detail': 'could not stage image EXIF orientation'}
                    db.session.commit()
                    continue
                if not _preserve_original(path):
                    _backup_failed(img, staged)
                    db.session.commit()
                    continue
                lama_pending.append((img.id, path, staged, regions, True))
                continue
            bbox = _safe_json(img.watermark_bbox)
            if not (isinstance(bbox, list) and len(bbox) == 4):
                # Flagged, position unknown. The detector cascade produces this
                # legitimately (its locator found nothing) and promotion carries
                # it in from a bank; stamping 'failed' would DESTROY a correct
                # flag over a missing coordinate. It goes to manual review, where
                # a zone can be drawn — the same answer the bank gives.
                out['needs_review'] += 1
                db.session.commit()
                continue
            if not os.path.exists(path):
                img.watermark_state = 'failed'
                out['failed'] += 1
                db.session.commit()
                continue
            try:
                with Image.open(path) as im:
                    # Stored detection boxes are in the browser/VLM's upright
                    # coordinate space, never the raw camera raster. This branch
                    # may route to review/no-op, so keep it header-only until an
                    # actual crop/staging edit needs the pixels.
                    W, H = image_encoding.visual_size_from_header(im)
            except (OSError, ValueError):
                img.watermark_state = 'failed'
                out['failed'] += 1
                db.session.commit()
                continue
            route, box = _route_watermark(tuple(bbox), W, H, allow_crop=allow_crop)
            if route == 'crop':
                if not _preserve_original(path):
                    _backup_failed(img)
                elif _apply_watermark_crop(path, box):
                    # NOTE dHash: the perceptual hash used for import-dedupe is recomputed
                    # ON THE FLY from the file (_existing_dhashes / _dhash), NOT stored in a
                    # column -- there is no stored dHash to leave untouched. So after a crop
                    # the dedupe compares against the CLEANED pixels; re-importing the same
                    # watermarked visual is NOT guaranteed to dedupe against it (a border
                    # crop shifts the whole hash). Preserving the original-dHash behaviour the
                    # spec asks for would need a new stored column -> deferred (out of V1 scope).
                    img.watermark_state = 'cleaned'
                    out['cropped'] += 1
                else:
                    img.watermark_state = 'failed'
                    out['failed'] += 1
            else:
                engine = _clean_inpaint_engine(route, method)
                if engine == 'klein':
                    _run_klein(img, path, [bbox], False)
                elif engine == 'lama':
                    if not lama_ok:
                        out['skipped'] += 1      # leave state='detected' (crop-only mode)
                    else:
                        staged = _stage_oriented_watermark_edit(path)
                        if staged:
                            if _preserve_original(path):
                                lama_pending.append((img.id, path, staged, [bbox], False))
                            else:
                                _backup_failed(img, staged)
                        else:
                            img.watermark_state = 'failed'
                            out['failed'] += 1
                            error = {'kind': 'failed',
                                     'detail': 'could not stage image EXIF orientation'}
                else:  # 'review' -> stays 'detected' so the badge/count keep flagging it
                    out['needs_review'] += 1
            db.session.commit()
        if lama_pending:
            try:
                if len(lama_pending) == 1:
                    _pid, live_path, staged_path, boxes, manual = lama_pending[0]
                    if manual:
                        ok, err = watermark_lama.inpaint_watermarks(
                            staged_path, boxes,
                            **({'device': device} if device != 'cpu' else {}))
                    else:
                        ok, err = watermark_lama.inpaint_watermark(
                            staged_path, boxes[0],
                            **({'device': device} if device != 'cpu' else {}))
                    results = {staged_path: (ok, err)}
                else:
                    results = watermark_lama.inpaint_batch(
                        [{'image_path': staged_path, 'bboxes': boxes}
                         for _pid, _live_path, staged_path, boxes, _manual in lama_pending],
                        device=device,
                    )
                for pending_id, live_path, staged_path, _boxes, manual in lama_pending:
                    img = _live_image_row(pending_id)
                    if img is None:
                        # Deleted while the batch ran: there is no row left to
                        # point at the repainted file, so drop the staged edit
                        # rather than promote it over a master nobody owns.
                        _discard_staged_watermark_edit(staged_path)
                        vanished += 1
                        continue
                    ok, err = results.get(
                        staged_path,
                        (False, {'kind': 'failed', 'detail': 'missing inpaint result'}),
                    )
                    if ok and _promote_staged_watermark_edit(staged_path, live_path):
                        img.watermark_state = 'cleaned'
                        if manual:
                            img.watermark_regions = None
                        out['inpainted'] += 1
                    elif ok:
                        if not manual:
                            img.watermark_state = 'failed'
                        out['failed'] += 1
                        error = {'kind': 'failed',
                                 'detail': 'could not promote staged watermark edit'}
                    elif err and err.get('kind') == 'unavailable':
                        out['skipped'] += 1
                    else:
                        # Manual correction regions are user-authored retry metadata. Keep
                        # the image detected when LaMa fails so Clean can be retried.
                        if not manual:
                            img.watermark_state = 'failed'
                        out['failed'] += 1
                        if err:
                            error = err
                    db.session.commit()
            except Exception as exc:  # engine/process faults must not leak a staged edit
                logger.exception('watermark: LaMa execution failed for dataset %s', dataset_id)
                error = {'kind': 'failed', 'detail': f'watermark inpaint failed: {exc}'}
                for pending_id, _live_path, _staged_path, _boxes, manual in lama_pending:
                    img = _live_image_row(pending_id)
                    if img is None:
                        vanished += 1
                        continue
                    if not manual:
                        img.watermark_state = 'failed'
                    out['failed'] += 1
                    db.session.commit()
            finally:
                # The engine can crash before returning a result; in that case its
                # disposable EXIF-oriented copy still has to disappear, while the
                # master remains exactly where it was.
                for _pid, _live_path, staged_path, _boxes, _manual in lama_pending:
                    _discard_staged_watermark_edit(staged_path)
        if vanished:
            logger.info('watermark clean: %s image(s) were deleted while the pass '
                        'ran, skipped', vanished)
        return out, error
    finally:
        dataset_activity.end(token)


@_serialize_dataset_ingest
def restore_watermark_original(user_id, dataset_id, image_id) -> dict | None:
    """Undo a watermark Clean on ONE image: copy the preserved `<stem>.orig<ext>` back
    over the current file and flip the row from 'cleaned' (or 'failed') back to
    'detected', so it re-enters the Clean set and the user can re-clean it -- e.g. retry
    with the OTHER engine, or re-edit the zones. Returns a payload dict (state + planned
    route + regions) on success, None when the image isn't found/owned, and raises
    FileNotFoundError when no original was preserved (the image was never cleaned, or the
    sibling was removed) -> the route maps that to a 404.

    Design: the `.orig` is KEPT after a restore. It stays the single source of truth for
    the original pixels, so any number of clean -> restore -> clean cycles never loses it:
    _preserve_original is write-once (guarded by os.path.exists), so a later re-clean sees
    the existing sibling and won't overwrite it with an already-edited image. bbox/regions
    are preserved as-is (a crop/inpaint doesn't move the normalized box, and the user may
    want to re-clean the same zones). The crop route shrinks the image; restoring the
    .orig also restores the ORIGINAL dimensions -- nothing stored depends on them (the
    planned-route recompute reads the file live in _payload_watermark_route)."""
    owned_query = (FaceDatasetImage.query
                   .join(FaceDataset, FaceDatasetImage.dataset_id == FaceDataset.id)
                   .filter(FaceDatasetImage.id == image_id,
                           FaceDatasetImage.dataset_id == dataset_id,
                           FaceDataset.user_id == str(user_id)))
    img = owned_query.one_or_none()
    if not img or not img.filename:
        return None
    path = _img_path(img)
    stem, ext = os.path.splitext(path)
    backup = f'{stem}.orig{ext or ".webp"}'
    if not os.path.exists(backup):
        raise FileNotFoundError('no original to restore')
    shutil.copy2(backup, path)   # bring the watermarked original back in place
    # Re-flag as 'detected' so the badge/Clean count pick it up again; bbox and manual
    # regions are left exactly as stored (re-cleanable, possibly with the other engine).
    img.watermark_state = 'detected'
    db.session.commit()
    return {'watermark_state': img.watermark_state,
            **_watermark_route_payload(img),
            **_watermark_regions_payload(img)}


# --- Fan-out generation (Klein edit) ---------------------------------------
def _sync_generate_activity(dataset_id):
    """Reconcile the Klein 'generate' indicator with the dataset's live count of
    in-flight Klein jobs (pending rows that still carry a job_id and have no file
    yet). Klein completions arrive one-by-one on the job-queue monitor thread with
    only a job_id — no batch handle — so we track the honest pending COUNT rather
    than a per-batch job set (duplicated/cancelled completions would corrupt one).
    Called on enqueue, on each completion, and on cancel; the registry TTL is the
    last-resort net. API rows (job_id is NULL) are excluded — those batches own a
    separate begin()/end() 'generate' entry from _run_nanobanana_batch."""
    local = (FaceDatasetImage.query
             .filter_by(dataset_id=dataset_id, status='pending')
             .filter(FaceDatasetImage.filename.is_(None))
             .filter(FaceDatasetImage.job_id.isnot(None)))
    pending = local.count()
    # There are TWO local engines now, and the indicator names one. Both queue on
    # the same single GPU and complete the same way, so the COUNT is shared; the
    # label just tells the truth about what is on it. Klein wins a mixed run only
    # because it is the historical default — a wrong badge is worse than a vague
    # one, so 'krea' is only claimed when every in-flight local row really is Krea.
    # NB: `klein_model != 'krea'` alone would DROP the NULL rows (SQL three-valued
    # logic), i.e. count a legacy Klein row as "not non-Krea" and mislabel the run.
    engine = 'klein'
    if pending and not local.filter(db.or_(
            FaceDatasetImage.klein_model.is_(None),
            FaceDatasetImage.klein_model != KREA_ENGINE)).count():
        engine = KREA_ENGINE
    dataset_activity.sync_pending(dataset_id, 'generate', pending, engine=engine)


def generate_variations(user_id, dataset_id, variations, multiplier, klein_model=None,
                        lora_strength=None, generation_lora_preset=None):
    """For each (variation x multiplier), enqueue a Klein edit of the reference
    and create a pending FaceDatasetImage. Returns the created image ids.

    The row is committed BEFORE enqueuing (so an enqueue/commit failure can never
    leave an untracked orphan job); on enqueue failure the row is marked 'failed'
    and the error re-raised (already-enqueued variations keep their rows).

    `generation_lora_preset`: NAME of the generation-LoRA preset picked for
    this run (optional generation LoRAs, Idea by @waltm) — resolved from the
    CONFIG only (fail-closed: the request can't define files/strengths/order;
    an unknown name degrades to no extra LoRAs with a log). The preset's chain
    applies to EVERY variation of the run — picking the preset IS the intent,
    there is no automatic per-variation gating."""
    _guard_not_bank_export(dataset_id)
    try:
        from .klein_edit_helper import enqueue_klein_edit
    except ImportError:
        raise RuntimeError('ComfyUI is not configured')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
    # No model named by the caller → the DATASET's pick (None = auto, i.e. exactly
    # what generation resolved before the setting existed). An explicit request
    # value still wins: it is what the workspace picker sends, and a legacy browser
    # that still holds editPage_flux2KleinModel_v1 must keep generating with it.
    if not klein_model:
        klein_model = dataset_klein_model(ds)
    # Preflight the Klein model files BEFORE creating any rows: a missing model
    # then surfaces as one actionable "downloading, retry" 409 (route handler) —
    # not a dataset full of failed tiles, each doomed by a ComfyUI validation
    # error on a file that isn't there.
    from .klein_edit_helper import klein_missing_assets, KLEIN_REQUIRED, KleinModelsMissing
    _missing = klein_missing_assets()
    if any(a in _missing for a in KLEIN_REQUIRED):
        raise KleinModelsMissing(_missing)
    mult = max(1, int(multiplier))
    total = len(variations) * mult
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    # Anti-DoS: the fan-out is free (never debited) → cap pending in-flight
    # generations per dataset so one user can't monopolize the single GPU.
    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + total > MAX_FANOUT:
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
    # Extra identity refs (multi-references) : chaînées en ReferenceLatent natifs
    # côté Klein — mêmes fichiers que le chemin Nano Banana multi-réfs.
    extra_paths = [os.path.join(_dataset_dir(ds.id), fn) for fn in extra_ref_filenames(ds)]
    # Optional generation LoRAs: resolve the picked preset from the config ONCE
    # (fail-closed — unknown name -> [] with a log). Same chain for every job.
    from .klein_edit_helper import resolve_generation_lora_preset
    run_loras = resolve_generation_lora_preset(generation_lora_preset)
    ids = []
    # try/finally: advertise the live 'generate' indicator even if an enqueue
    # fails partway (the already-queued rows are still in flight). Each Klein job
    # completes asynchronously; _sync_generate_activity keeps the count honest and
    # link_completed_dataset_image clears it when the last one lands.
    try:
        for v in variations:
            for _ in range(mult):
                img = FaceDatasetImage(dataset_id=dataset_id, source='generated', status='pending',
                                       variation_label=v.get('label'), framing=v.get('framing'),
                                       variation_prompt=v['prompt'], klein_model=klein_model)
                db.session.add(img)
                db.session.commit()
                # Captured NOW, while the row certainly exists: ⏹ Stop deletes
                # exactly this shape (status='pending' AND filename IS NULL), and
                # it can land while the enqueue below is in flight.
                image_id = img.id
                # NSFW (flag explicite OU label du catalogue NSFW) : wrapper sans le
                # clamp SFW — chemin Klein local uniquement, les moteurs API sont
                # refusés en amont (route + generate_variations_nanobanana).
                nsfw = bool(v.get('nsfw')) or is_nsfw_label(v.get('label'))
                try:
                    job_id = enqueue_klein_edit(
                        user_id=str(user_id), source_filename=ds.ref_filename,
                        source_path=_ref_path(ds),
                        # Dataset suffix applied AT WRAP — the row above keeps the
                        # raw catalog prompt, so regenerate re-applies the CURRENT
                        # suffix exactly once (never a double application).
                        edit_prompt=wrap_variation_klein(
                            v['prompt'], nsfw=nsfw, framing=v.get('framing'),
                            suffix=dataset_prompt_suffix(ds, v.get('framing')),
                            subject_type=subject_type_of(ds),
                            # Picks this shot's concrete garment, like the Krea
                            # path — deterministic, so a regenerate reproduces it.
                            label=v.get('label') or ''),
                        klein_model=klein_model,
                        lora_strength=lora_strength, extra_ref_paths=extra_paths,
                        generation_loras=run_loras, sampler_steps=_generation_steps(),
                        base_lora_strength=_generation_base_lora_strength(),
                        extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                                        'variation_label': v.get('label')})
                except Exception:
                    row = _live_image_row(image_id)
                    if row is not None:
                        row.status = 'failed'
                        db.session.commit()
                    raise
                row = _live_image_row(image_id)
                if row is None:
                    continue     # Stop removed it mid-enqueue; nothing to report
                row.job_id = job_id
                db.session.commit()
                ids.append(image_id)
    finally:
        _sync_generate_activity(dataset_id)
    return ids


def generate_variations_krea(user_id, dataset_id, variations, multiplier,
                            generation_lora_preset=None):
    """Krea 2 Identity Edit fan-out — the second LOCAL engine, same contract as
    `generate_variations` (Klein): one pending row committed BEFORE its job is
    enqueued, the whole batch preflighted up front, the created ids returned.

    Fewer knobs than the Klein path, but no longer none: Krea has no consistency
    LoRA, and its identity LoRA IS the pipeline. Stacking untested LoRAs on an
    edit model still degrades it — that caution was this lane's reason for having
    no LoRA input at all, and it is now the USER's call per run instead of ours:
    Krea 2 cannot render some registers a dataset needs, and a LoRA is the only
    lever that reaches them. `generation_lora_preset` NAMES a preset from config
    (absent = none, which is still the default and still the byte-identical
    graph). The other dial, `grounding_px`, remains a SETTING rather than a
    per-run argument, because it changes the meaning of every shot in the batch
    identically.

    The row stores the ENGINE ID in `klein_model`, like the API rows do, so the
    grid badge can say "Krea 2 Edit"; the base model itself is re-resolved
    deterministically at enqueue and at regenerate."""
    _guard_not_bank_export(dataset_id)
    from . import krea_edit_helper as keh
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
    # Assets AND custom nodes, before any row exists: a missing piece then
    # surfaces as one actionable 409 instead of a grid of silently-failing tiles.
    keh.preflight()
    # Resolved ONCE per run, not per variation: every cell of a run gets the same
    # always-on stack (that is what "always-on" means), and one config read is
    # enough. Unknown/blank name -> [] (fail-closed, see the resolver).
    run_loras = keh.resolve_generation_lora_preset(generation_lora_preset)
    mult = max(1, int(multiplier))
    total = len(variations) * mult
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + total > MAX_FANOUT:
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
    ref_path = _ref_path(ds)
    ids = []
    try:
        for v in variations:
            for _ in range(mult):
                img = FaceDatasetImage(dataset_id=dataset_id, source='generated',
                                       status='pending', variation_label=v.get('label'),
                                       framing=v.get('framing'),
                                       variation_prompt=v['prompt'],
                                       klein_model=KREA_ENGINE)
                db.session.add(img)
                db.session.commit()
                # Same reason as the Klein path: ⏹ Stop deletes exactly this
                # pending/filename-less shape, and the enqueue below is the window.
                image_id = img.id
                nsfw = bool(v.get('nsfw')) or is_nsfw_label(v.get('label'))
                try:
                    job_id = keh.enqueue_krea_edit(
                        user_id=str(user_id), source_filename=ds.ref_filename,
                        source_path=ref_path,
                        # Suffix applied AT WRAP, like Klein: the row keeps the raw
                        # catalog prompt so a regenerate re-applies the CURRENT
                        # suffix exactly once. The label rides along because it
                        # picks this shot's outfit deterministically.
                        edit_prompt=wrap_variation_krea(
                            v['prompt'], nsfw=nsfw, framing=v.get('framing'),
                            suffix=dataset_prompt_suffix(ds, v.get('framing')),
                            subject_type=subject_type_of(ds),
                            label=v.get('label') or ''),
                        # Krea v1.2 fit geometry accepts the catalog canvas even
                        # when it differs from the dataset reference.
                        aspect_ratio=aspect_for_label(v.get('label'), v.get('framing')),
                        generation_loras=run_loras,
                        extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                                        'variation_label': v.get('label')})
                except Exception:
                    row = _live_image_row(image_id)
                    if row is not None:
                        row.status = 'failed'
                        db.session.commit()
                    raise
                row = _live_image_row(image_id)
                if row is None:
                    continue     # Stop removed it mid-enqueue; nothing to report
                row.job_id = job_id
                db.session.commit()
                ids.append(image_id)
    finally:
        _sync_generate_activity(dataset_id)
    return ids


# --- The ✨ Upscale & improve profile, read ONCE per pass ----------------------
# Every knob below lives in config and is user-editable, so it must be read at
# ENQUEUE time, never frozen into the candidate row: re-running the pass (🔄 on
# an improved tile) has to pick up whatever the user has since changed. These
# four helpers are the single source of truth shared by the first pass
# (improve_existing_image) and the re-run (reimprove_image) — two copies would
# drift, and a re-run that used yesterday's settings is exactly the bug.
def _improve_prompt() -> str:
    """The improvement instruction, editable in Settings ▸ identity_prompts.
    klein_improve, and switchable OFF entirely — disabled applies NO prompt
    (pure upscale)."""
    if cfg.get('identity_prompts.klein_improve_enabled', True):
        return get_identity_prompt('klein_improve')
    return ''


# The tile label of an improve candidate, per engine. STORED in
# FaceDatasetImage.variation_label, so the rule about stored strings applies —
# and it is the reason Klein's wording is byte-identical to what it always was:
# renaming it would strand every candidate already in every user's database
# behind a label nothing produces any more, and would need an alias table to
# read them back. SeedVR2 gets its OWN new string instead, so nothing is
# renamed and no alias path is needed.
#
# Checked before writing this (2026-08-02): the literal 'Klein upscale &
# improve' is matched by NO runtime code, front or back — it is a display label
# only (grep hits are this builder, test fixtures, and prose in Settings/help).
# Had anything keyed off it, adding a second value would have needed the alias
# table, not just a new branch.
_IMPROVE_LABELS = {
    'klein': 'Klein upscale & improve',   # NEVER change: stored in user databases
    'seedvr2': 'SeedVR2 upscale',
}


def _improve_candidate_label(source, engine='klein') -> str:
    """Label of the candidate produced from ``source`` (its parent image).

    Names the engine that ACTUALLY ran: a SeedVR2 result labelled "Klein upscale
    & improve" tells the user the one thing they chose this pass to avoid."""
    base_label = _IMPROVE_LABELS.get(engine, _IMPROVE_LABELS['klein'])
    source_label = (source.variation_label or '').strip()
    return (f'{base_label} · {source_label}' if source_label else base_label)[:120]


def _improve_enqueue_profile(ds=None) -> dict:
    """Profile reproduced from the user-provided ComfyUI PNG metadata: the
    dataset's Klein model plus the sampling/LoRA/resolution overrides.

    `ds` is the dataset the improved image belongs to. Reading its model HERE is
    what makes the choice reach all three improve lanes at once: the single pass,
    the 🔄 re-run, and the batch (which drains through improve_existing_image).
    None / a dataset that never chose yields klein_model=None — the exact value
    every improve sent before this setting existed.

    The defaults (1.0 / 4 / 0.0 / 2.0) are the values that were once hardcoded,
    so an untouched install behaves exactly as before. Clamped, because a bad
    config value must degrade the pass, never crash the enqueue. Each fallback
    MUST equal the shipped config default: _improve_float treats "still at its
    default" as "the user has not set this", which is what lets a value saved
    under the old key name speak for it."""
    return {
        'klein_model': dataset_klein_model(ds),
        'lora_strength': _improve_float('improve_consistency_strength', 1.0, 1.5),
        'sampler_steps': _improve_int('improve_steps', 4),
        'base_lora_strength': _improve_float('improve_base_lora_strength', 0.0),
        'output_megapixels': _improve_float('improve_megapixels', 2.0, 8.0),
    }


def _improve_extra_metadata(source, label, engine='klein') -> dict:
    return {
        'is_dataset': True,
        'dataset_id': source.dataset_id,
        'variation_label': label,
        'derivation_kind': KLEIN_IMAGE_IMPROVE,
        'parent_image_id': source.id,
        'source_image_id': source.id,
        'action': 'upscale_improve',
        # WHICH engine produced this candidate. Additive: `derivation_kind` and
        # `action` keep the values every existing row and every reader already
        # carries (they are stored, and stored ids are never renamed here), so a
        # SeedVR2 result curates, undoes and re-improves exactly like a Klein one.
        'improve_engine': engine,
    }


# --- Which engine runs the ✨ improve pass ------------------------------------
# Two passes, one lane. Klein REWRITES (a diffusion edit that re-renders skin
# and micro-detail from a prompt: it fixes a soft photo and it changes it);
# SeedVR2 RESTORES (one-step diffusion super-resolution that leaves the content
# where it was). Which one you want depends on whether the frame's exact look is
# the thing you are training on — so it is a choice, not a default we can pick
# for everyone. Requested in issue #32 by SurpassHR, whose complaint was exactly
# that Klein "tends to change the detail and color of the original image".
IMPROVE_ENGINES = ('klein', 'seedvr2')


def resolve_improve_engine(requested=None):
    """The engine an improve request will run on: the explicit pick when it names
    a known engine, else the `improve.engine` setting, else Klein.

    Fail-SAFE rather than fail-closed: an unknown name falls back instead of
    raising, because a stale tab must degrade to the historical behaviour rather
    than refuse a batch. Klein is the fallback because it is what every improve
    did before this setting existed."""
    for candidate in (requested, cfg.get('improve.engine')):
        name = str(candidate or '').strip().lower()
        if name in IMPROVE_ENGINES:
            return name
        if name:
            logger.warning('unknown improve engine %r — falling back to klein', candidate)
    return 'klein'


def _improve_preflight(engine):
    """Raise the engine's structured missing-assets exception when it cannot run.

    Called BEFORE any candidate row is created, so a missing model surfaces once
    per batch instead of once per image. Each engine keeps its own exception type
    (KleinModelsMissing / KleinNodesMissing / SeedVR2ModelsMissing) — the routes
    already turn each into its own actionable 409 body, and collapsing them into
    one would lose the "install the node pack" vs "place the weights" distinction
    that makes those bodies useful."""
    if engine == 'seedvr2':
        from . import seedvr2_helper
        seedvr2_helper.preflight()
        return
    from . import klein_edit_helper as keh
    missing = keh.klein_missing_assets()
    missing_nodes = keh.klein_missing_nodes()
    if missing_nodes:
        raise KleinNodesMissing(missing, missing_nodes)
    if any(asset in missing for asset in keh.KLEIN_REQUIRED):
        raise keh.KleinModelsMissing(missing)


def _enqueue_improve(engine, *, user_id, source, source_path, prompt, label,
                     dataset, extra_metadata=None):
    """Hand ONE improve off to the chosen engine and return its job id.

    The two engines take deliberately different arguments — Klein needs a prompt,
    a consistency-LoRA strength and a step count; SeedVR2 needs none of them
    (there is no prompt in a restoration) — so this is where that difference
    stops, and every caller above it stays engine-agnostic.

    `extra_metadata` overrides what the finished job is linked back TO. Default
    (None) keeps the dataset-image contract every existing caller relies on. The
    ◉ Canvas improve passes its own because its source is a `LoraTestImage`, not
    a `FaceDatasetImage`: the two live in different tables with independent id
    spaces, and the completion callback is chosen by this metadata. The engine
    dispatch below stays the single place that knows Klein from SeedVR2 — that is
    the whole point of routing the second lane through here rather than growing a
    parallel copy of it."""
    meta = (dict(extra_metadata) if extra_metadata is not None
            else _improve_extra_metadata(source, label, engine=engine))
    if engine == 'seedvr2':
        from . import seedvr2_helper
        return seedvr2_helper.enqueue_seedvr2_upscale(
            user_id=str(user_id), source_filename=source.filename,
            source_path=source_path, extra_metadata=meta)
    from . import klein_edit_helper as keh
    return keh.enqueue_klein_edit(
        user_id=str(user_id), source_filename=source.filename,
        source_path=source_path, edit_prompt=prompt,
        **_improve_enqueue_profile(dataset), extra_metadata=meta)


def improve_existing_image(user_id, image_id, engine=None):
    """Serialize one source's improve request, including the queue hand-off."""
    image = _owned_image(user_id, image_id)
    if image is None:
        return None
    _guard_not_bank_export(image.dataset_id)
    lock = _IMAGE_IMPROVE_LOCKS[hash((str(user_id), image_id))
                                % len(_IMAGE_IMPROVE_LOCKS)]
    with lock:
        return _improve_existing_image_locked(user_id, image_id, engine=engine)


def _improve_existing_image_locked(user_id, image_id, engine=None):
    """Queue one non-destructive upscale/improvement of an existing image.

    The source row and file are deliberately never modified.  The result is a
    regular generated dataset image linked back to the source only for
    provenance; unlike the small-scrape review pair it remains compatible with
    the ordinary keep/reject/delete actions.

    Returns ``{'candidate_id', 'job_id'}``, ``None`` for an image not owned by
    ``user_id``, and returns the already-active candidate idempotently when the
    same source is clicked twice.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    _guard_not_bank_export(img.dataset_id)
    if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
        raise ValueError(
            'resolve the small-image rescue pair before improving either image')
    if img.derivation_kind == KLEIN_IMAGE_IMPROVE:
        raise ValueError('an upscale & improve candidate cannot be improved again')
    if not img.filename:
        raise ValueError('image file required')
    source_path = _img_path(img)
    if not os.path.isfile(source_path):
        raise ValueError('image file missing')

    # A completed Klein job remains status=pending until the user curates it, so
    # both an in-flight candidate (no filename yet) and an unreviewed result are
    # active.  Repeated clicks return that same job instead of consuming the GPU
    # or producing visually indistinguishable duplicates.
    active = (FaceDatasetImage.query
              .filter_by(dataset_id=img.dataset_id, parent_image_id=img.id,
                         derivation_kind=KLEIN_IMAGE_IMPROVE, status='pending')
              .order_by(FaceDatasetImage.id.desc()).first())
    if active:
        if active.job_id:
            return {'candidate_id': active.id, 'job_id': active.job_id}
        # This tiny state exists only between the row commit and queue enqueue.
        # Refuse a concurrent click rather than creating a second candidate.
        raise RuntimeError('this image improvement is already being queued')

    engine = resolve_improve_engine(engine)
    _improve_preflight(engine)

    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=img.dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + 1 > MAX_FANOUT:
        raise ValueError(
            f'too many generations in flight ({in_flight}), wait or cancel')

    prompt = _improve_prompt()
    # What the tile SHOWS as the prompt behind this candidate. A SeedVR2 run has
    # no prompt at all — it is a restoration — so storing the Klein improve
    # prompt on one would put a sentence on screen that had no effect on the
    # image. The honest value is the pass that ran.
    stored_prompt = (prompt[:500] if engine == 'klein'
                     else 'SeedVR2 upscale (no prompt — restoration pass)')
    label = _improve_candidate_label(img, engine)
    candidate = FaceDatasetImage(
        dataset_id=img.dataset_id, source='generated', status='pending',
        parent_image_id=img.id, derivation_kind=KLEIN_IMAGE_IMPROVE,
        # The stamp travels with the sentence it describes: a candidate that
        # inherits a hand-written caption inherits the protection on it, or the
        # first forced pass would rewrite the words on the copy while sparing them
        # on the original.
        framing=img.framing, caption=img.caption,
        caption_origin=img.caption_origin,
        variation_label=label, variation_prompt=stored_prompt,
        # The generated candidate remains derived from the credited source.
        # Revalidate before copying so a malformed legacy row cannot surface.
        source_metadata=_source_metadata_storage(img.source_metadata),
    )
    db.session.add(candidate)
    db.session.commit()
    # Captured while both rows certainly exist: the commit above expires them,
    # and ⏹ Stop deletes exactly this candidate's shape (pending, no filename)
    # — the enqueue below is the window, same as the variation paths.
    candidate_id = candidate.id
    dataset_id_of_source = img.dataset_id

    try:
        job_id = _enqueue_improve(
            engine, user_id=user_id, source=img, source_path=source_path,
            prompt=prompt, label=label,
            dataset=get_dataset(user_id, dataset_id_of_source))
    except Exception:
        # No broken tile: the original is still untouched and the user can retry
        # as soon as the queue/ComfyUI issue is fixed. Nothing to remove if Stop
        # already removed it — and trying would raise from inside this `except`,
        # replacing the real enqueue error with a database one.
        row = _live_image_row(candidate_id)
        if row is not None:
            db.session.delete(row)
            db.session.commit()
        raise

    row = _live_image_row(candidate_id)
    if row is None:
        # Stop removed the candidate mid-enqueue. Reporting its id would have the
        # tile poll for a generation that can never arrive.
        _sync_generate_activity(dataset_id_of_source)
        return None
    row.job_id = job_id
    db.session.commit()
    _sync_generate_activity(dataset_id_of_source)
    return {'candidate_id': candidate_id, 'job_id': job_id}


# The three ways a re-run can be impossible, worded as the user reads them. The
# tile mirrors them (frontend/src/components/dataset/improveRerun.js) so the
# button explains itself BEFORE the click rather than through a 400 after it.
REIMPROVE_PARENT_GONE = ('the source image this improvement came from was deleted '
                         '— nothing left to re-improve from')
REIMPROVE_SOURCE_FILE_GONE = ('the source image file is missing on disk '
                              '— nothing left to re-improve from')
REIMPROVE_IN_FLIGHT = 'this improvement is still generating'
REIMPROVE_STATE_CHANGED = ('this improvement changed while it was being re-queued '
                           '— review it and try again')


def reimprove_image(user_id, image_id):
    """Re-run the ✨ Upscale & improve pass that produced ``image_id``.

    The generic regenerate route is deliberately CLOSED to these rows: it starts
    from the dataset's reference photo and the catalog prompt, so on an improved
    tile it would quietly produce an unrelated variation. The right gesture is
    this one — run the improve pass again, from the SAME parent image, with the
    settings as they are TODAY (klein.improve_* + the klein_improve instruction
    are user-editable, and tuning them is the whole reason to re-run).

    Replaces IN PLACE, exactly like regenerate_image: same row id, same
    parent/derivation links, the previous result goes to the Trash once the new
    job is safely queued. A second candidate next to the first would break the
    one-live-improvement-per-source invariant that improve_existing_image and
    bulk_improve_eligible_ids already enforce.

    Returns ``{'candidate_id', 'job_id'}``, or None when the image is not owned
    by ``user_id``. Raises ValueError (-> 400) when the row is not an improvement
    or its parent is gone, RuntimeError (-> 409) while the pass is still running.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    _guard_not_bank_export(img.dataset_id)
    if img.derivation_kind != KLEIN_IMAGE_IMPROVE:
        raise ValueError('only an upscale & improve result can be re-improved')
    # Take the same stripe as a first-pass improve OF THE PARENT: the two paths
    # compete for the same "one live candidate per source" slot.
    lock = _IMAGE_IMPROVE_LOCKS[hash((str(user_id), img.parent_image_id))
                                % len(_IMAGE_IMPROVE_LOCKS)]
    with lock:
        return _reimprove_image_locked(user_id, image_id)


def _reimprove_image_locked(user_id, image_id):
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    _guard_not_bank_export(img.dataset_id)
    if img.derivation_kind != KLEIN_IMAGE_IMPROVE:
        raise ValueError('only an upscale & improve result can be re-improved')
    if img.status == 'pending' and not img.filename:
        raise RuntimeError(REIMPROVE_IN_FLIGHT)

    # The parent is what this pass runs on. It carries no ForeignKey (legacy
    # databases), so a deleted source leaves a dangling id — check the row, not
    # just the column.
    parent = (FaceDatasetImage.query
              .filter_by(id=img.parent_image_id, dataset_id=img.dataset_id).first()
              if img.parent_image_id else None)
    if not parent or not parent.filename:
        raise ValueError(REIMPROVE_PARENT_GONE)
    source_path = _img_path(parent)
    if not os.path.isfile(source_path):
        raise ValueError(REIMPROVE_SOURCE_FILE_GONE)

    # A re-run uses the CURRENTLY selected engine, not the one that produced the
    # row: "re-improve" means "try again with what I have set now", and someone
    # who switched to SeedVR2 precisely because the Klein result changed too much
    # would otherwise get the same Klein result back.
    engine = resolve_improve_engine()
    _improve_preflight(engine)

    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=img.dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + 1 > MAX_FANOUT:
        raise ValueError(
            f'too many generations in flight ({in_flight}), wait or cancel')

    prompt = _improve_prompt()
    label = _improve_candidate_label(parent, engine)

    # Enqueue BEFORE touching the row (regenerate_image's ordering): a ComfyUI
    # refusal must leave the current result on screen, not a broken tile.
    from ..job_queue import queue_manager
    old_state = {field: getattr(img, field) for field in (
        'filename', 'caption', 'caption_origin', 'status', 'fail_reason',
        'fail_kind', 'job_id',
        'variation_label', 'variation_prompt', 'framing',
        'watermark_state', 'watermark_bbox', 'watermark_regions')}
    # The caption the transition will write, AND who wrote it — the two are read
    # from the same row so an inherited parent caption arrives with the parent's
    # authorship rather than as an anonymous string.
    expected_transition_caption = (old_state['caption']
                                   if old_state['caption'] else parent.caption)
    expected_transition_caption_origin = (old_state['caption_origin']
                                          if old_state['caption']
                                          else parent.caption_origin)
    old_path = _img_path(img) if img.filename else None
    job_id = _enqueue_improve(
        engine, user_id=user_id, source=parent, source_path=source_path,
        prompt=prompt, label=label,
        dataset=get_dataset(user_id, img.dataset_id))

    try:
        # Do this while the candidate is still Keep.  The CAS observes both
        # rows in the database, so an intervening parent reject/failed decision
        # is never overwritten by this re-run's temporary fallback.
        parent_rekept = _rekeep_pending_parent_for_reimprove(img)
        if not _transition_reimprove_candidate(
                img, old_state, parent, label, prompt, job_id,
                expected_transition_caption,
                expected_transition_caption_origin):
            # The status/file/job snapshot changed after enqueue. Rolling back
            # also undoes a just-applied parent fallback; the except path below
            # cancels this unlinked job and maps the race to a 409.
            raise RuntimeError(REIMPROVE_STATE_CHANGED)
        db.session.commit()
    except Exception:
        db.session.rollback()
        try:
            queue_manager.cancel_job(job_id, str(user_id), 'image')
        except Exception:
            logger.exception('reimprove: failed to cancel unlinked job %s', job_id)
        raise

    # The DB no longer references the old file. If Trash itself fails, restore
    # the exact previous row state and cancel the job we just queued.
    try:
        if old_path and os.path.exists(old_path):
            trash.send_to_trash(
                old_path, context=f'dataset-{img.dataset_id}-reimprove-{img.id}')
    except Exception:
        try:
            # Do not restore the old row over an unresolved new prompt: its
            # eventual callback still owns `job_id` and would otherwise write
            # into the restored candidate.
            if not queue_manager.cancel_job(job_id, str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'The replacement generation still has unconfirmed ComfyUI work.')
            restored_candidate = _restore_reimprove_candidate_after_trash_failure(
                img, old_state, job_id, expected_transition_caption)
            if parent_rekept and restored_candidate:
                # Candidate first: if a user changed it during the Trash call,
                # its CAS fails and the fallback parent remains Keep instead of
                # overwriting that newer decision.
                _undo_rekeep_parent_after_reimprove_trash_failure(img, old_state)
            db.session.commit()
        except Exception:
            db.session.rollback()
            logger.exception('reimprove: failed to restore row %s after Trash error',
                             image_id)
        raise

    _sync_generate_activity(img.dataset_id)
    return {'candidate_id': img.id, 'job_id': job_id}


# --- Bulk Klein upscale & improve: a SERVER job --------------------------------
# The ✨ Improve button used to loop in the BROWSER, one request per image. On a
# 250-image selection that produced two bugs with a single root cause — the batch
# only existed in the tab:
#   * everything past MAX_FANOUT was REFUSED. That cap is a CONCURRENCY limit
#     ("how many generations may be in flight at once"), and a client loop that
#     keeps pushing simply walks into it: 60 queued, 190 counted as failures.
#   * ⏹ Stop was powerless. cancel_pending did its job (rows cancelled, ComfyUI
#     prompts interrupted) and the tab immediately re-queued the next 60. Closing
#     the tab killed whatever was left.
# So the batch runs server-side now: one background thread per dataset, advertised
# through dataset_activity (kind 'improve') so the progress SURVIVES a reload, and
# draining the selection in WAVES — it waits for a slot to free instead of hitting
# the wall — with a cooperative stop checked at every image boundary.
IMPROVE_SLOT_POLL_SECONDS = 2.0
# Give up (and say so) if no slot frees for this long. A ComfyUI that died mid-batch
# would otherwise leave the thread polling a count that never drops, and the dataset
# stuck behind an "in progress" indicator until the registry TTL expires.
IMPROVE_SLOT_TIMEOUT_SECONDS = 15 * 60
# Chunk the id lookup: a selection is user-sized and SQLite caps bound parameters.
_IMPROVE_ID_CHUNK = 400


def _improve_in_flight(dataset_id):
    """Live count of generations in flight on ``dataset_id`` — the very number
    improve_existing_image checks against MAX_FANOUT. Ends the worker thread's read
    transaction first (a rollback on a clean session is a no-op) so each poll sees
    the rows COMMITTED by the job-queue monitor thread rather than a stale snapshot;
    without it the count would never drop and the batch would stall forever."""
    db.session.rollback()
    return (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).count())


def bulk_improve_eligible_ids(user_id, dataset_id, image_ids):
    """The subset of ``image_ids`` this dataset can actually improve, in selection
    order and de-duplicated. Mirrors the client-side partition
    (frontend/src/utils/kleinBulkImprove.js) so the total the job advertises is the
    number it will really work on — a batch that announced 250 and refused 40 of
    them one by one is exactly the dishonesty this rewrite removes."""
    wanted, seen = [], set()
    for raw in image_ids or []:
        try:
            image_id = int(raw)
        except (TypeError, ValueError):
            continue
        if image_id not in seen:
            seen.add(image_id)
            wanted.append(image_id)
    if not wanted:
        return []
    rows = {}
    for start in range(0, len(wanted), _IMPROVE_ID_CHUNK):
        chunk = wanted[start:start + _IMPROVE_ID_CHUNK]
        for row in (FaceDatasetImage.query
                    .filter(FaceDatasetImage.dataset_id == dataset_id,
                            FaceDatasetImage.id.in_(chunk)).all()):
            rows[row.id] = row
    # Sources whose improvement is already pending review (or still generating):
    # re-improving them would just make an indistinguishable duplicate.
    busy_parents = {row.parent_image_id for row in (
        FaceDatasetImage.query
        .filter_by(dataset_id=dataset_id, derivation_kind=KLEIN_IMAGE_IMPROVE,
                   status='pending').all())}
    eligible = []
    for image_id in wanted:
        img = rows.get(image_id)
        if not img or not img.filename:
            continue
        if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
            continue
        if img.derivation_kind == KLEIN_IMAGE_IMPROVE:
            continue
        if image_id in busy_parents:
            continue
        eligible.append(image_id)
    return eligible


def start_bulk_improve(app, user_id, dataset_id, image_ids, engine=None):
    """Start the server-side ✨ Upscale & improve batch over ``image_ids``.

    Returns ``{'queued', 'skipped', 'engine'}`` — how many images the job will
    process, how many of the selection were not eligible, and which engine ran
    (the caller echoes it so the toast can name it). Raises ValueError (-> 400) on
    an unknown dataset / an empty eligible set, RuntimeError (-> 409) when a batch
    is already running, and the engine's missing-assets exceptions (-> structured
    409) so a missing model surfaces ONCE instead of once per image."""
    _guard_not_bank_export(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if dataset_activity.running(dataset_id, dataset_activity.IMPROVE_KINDS):
        raise RuntimeError('an improvement batch is already running on this dataset')
    engine = resolve_improve_engine(engine)
    _improve_preflight(engine)
    eligible = bulk_improve_eligible_ids(user_id, dataset_id, image_ids)
    if not eligible:
        raise ValueError('no selected image is eligible for improvement')
    skipped = max(0, len(set(image_ids or [])) - len(eligible))
    total = len(eligible)
    token = dataset_activity.begin(dataset_id, 'improve', total=total,
                                   detail=f'Queuing improvements… 0/{total}',
                                   engine=engine)

    def _run():
        try:
            with app.app_context():
                _drain_improve_queue(user_id, dataset_id, eligible, token,
                                     engine=engine)
        except Exception:   # noqa: BLE001 — a background crash must not strand the indicator
            logger.exception('bulk improve batch failed on dataset %s', dataset_id)
        finally:
            dataset_activity.end(token)
            dataset_activity.clear_cancel(dataset_id, dataset_activity.IMPROVE_KINDS)

    # Under TESTING the job runs INLINE (same rule as bank_jobs): the suite uses a
    # per-connection in-memory sqlite, so a real worker thread would open a fresh,
    # EMPTY database.
    if app.config.get('TESTING'):
        _run()
    else:
        threading.Thread(target=_run, daemon=True,
                         name=f'ds-{dataset_id}-improve').start()
    return {'queued': total, 'skipped': skipped, 'engine': engine}


def _drain_improve_queue(user_id, dataset_id, image_ids, token, sleep=time.sleep,
                         engine=None):
    """Queue one improvement per id, in WAVES that respect the MAX_FANOUT
    concurrency cap: when the dataset already has that many generations in flight
    the worker WAITS for a slot (the count drops as ComfyUI writes the files) rather
    than firing a request doomed to be refused. Stops at the next image boundary
    when ⏹ Stop arms the flag. Returns a summary dict (also used by the tests)."""
    total = len(image_ids)
    queued = failed = 0
    waited = 0.0
    stopped = stalled = False

    def _stop_requested():
        return dataset_activity.cancel_requested(dataset_id,
                                                 dataset_activity.IMPROVE_KINDS)

    for index, image_id in enumerate(image_ids):
        while not stopped and not stalled and _improve_in_flight(dataset_id) + 1 > MAX_FANOUT:
            if _stop_requested():
                stopped = True
            elif waited >= IMPROVE_SLOT_TIMEOUT_SECONDS:
                stalled = True
            else:
                dataset_activity.progress(
                    token,
                    detail=f'Queuing improvements… {queued}/{total} — waiting for a '
                           f'free generation slot ({total - index} left)')
                sleep(IMPROVE_SLOT_POLL_SECONDS)
                waited += IMPROVE_SLOT_POLL_SECONDS
        if stalled:
            break
        if stopped or _stop_requested():
            stopped = True
            break
        waited = 0.0
        try:
            improve_existing_image(user_id, image_id, engine=engine)
            queued += 1
        except Exception as exc:   # noqa: BLE001 — one refusal never sinks the batch
            failed += 1
            logger.warning('bulk improve: image %s could not be queued (%s)',
                           image_id, exc)
        dataset_activity.bump(token)
        dataset_activity.progress(
            token, detail=f'Queuing improvements… {queued}/{total}')
    return {'total': total, 'queued': queued, 'failed': failed,
            'stopped': stopped, 'stalled': stalled,
            'remaining': total - queued - failed}


def regenerate_image(user_id, image_id, lora_strength=None, prompt=None, app=None,
                     engine=None, klein_model=None, generation_lora_preset=None):
    """Re-enqueue a single generated variation IN PLACE (same row id): cancel any
    in-flight job, drop the old file, reset the row to pending with the new
    job_id. Returns the new job_id, or None if the image is not owned / not a
    generated variation. Raises ValueError if the dataset has no reference or
    the variation prompt can't be recovered.

    `prompt` (optional) is the user-EDITED core creative prompt from the tile's
    ✏️ bubble. When given it REPLACES and is PERSISTED into `variation_prompt`
    (so a later plain regenerate / reject-regenerate reuses the edit), then feeds
    the identity-guard wrapper like any catalog prompt — the face lock is still
    applied on top, the user only steers the creative half. Empty/None = the
    current behaviour (recover the prompt from the row or the label).

    `engine` (optional, one of ``KNOWN_ENGINES``) is an EXPLICIT caller
    override. The ordinary workspace Retry omits it, so it reuses the engine
    recorded on the row; callers that deliberately pass one can still move a
    tile to another lane. Exception:
    an NSFW-labelled tile always stays on the local Klein path (fail-closed —
    NSFW never goes to third-party APIs, mirroring the batch generate rule).
    `klein_model` (optional) is the workspace's Klein model pick, used when a
    row born on an API engine switches to Klein (its klein_model column holds
    an engine TAG, not a real model file).
    `generation_lora_preset` (optional): NAME of the generation-LoRA preset
    picked in the workspace (Idea by @waltm). Both local engines resolve it —
    Klein and Krea each from their OWN config list (`klein.generation_lora_presets`
    / `krea.generation_lora_presets`), so the same name can mean two different
    chains depending on which engine `target` resolves to below — resolved from
    the CONFIG only (fail-closed; unknown name degrades to no extra LoRAs)."""
    img = _owned_image(user_id, image_id)
    if not img or img.source != 'generated':
        return None
    _guard_not_bank_export(img.dataset_id)
    if img.derivation_kind == KLEIN_SMALL_IMAGE:
        raise ValueError('small-image rescue candidates cannot be regenerated; re-import the source')
    if img.derivation_kind == KLEIN_IMAGE_IMPROVE:
        raise ValueError('upscale & improve candidates cannot be regenerated from the dataset reference')
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds.ref_filename:
        raise ValueError('reference image required')
    edited = (prompt or '').strip()
    stored_prompt = edited[:500] if edited else img.variation_prompt
    prompt = stored_prompt or prompt_by_label(img.variation_label or '')
    if prompt is None:
        raise ValueError('variation prompt unknown')
    requested = (engine or '').strip() or None
    if requested is not None and requested not in KNOWN_ENGINES:
        raise ValueError(f'unknown engine: {requested}')
    # A row remembers its origin through `klein_model`: an engine TAG for the API
    # rows and for Krea, a real model FILE for Klein. Anything that isn't a known
    # tag is therefore a Klein row.
    origin = img.klein_model if img.klein_model in KNOWN_ENGINES else 'klein'
    target = requested or origin
    if is_nsfw_label(img.variation_label) and target in API_ENGINES:
        # Fail-closed: NSFW never reaches a third-party API. It stays on whatever
        # LOCAL engine the row came from (a Krea row keeps Krea) — forcing Klein
        # here would silently change engine behind the user's back.
        target = origin if origin in LOCAL_ENGINES else 'klein'
    else:
        # Engines disabled in Settings must not be used even when the row (or a
        # stale workspace selection) points at them: fall back to the default
        # engine, then to the first enabled one. An empty list means "all
        # enabled" (legacy configs).
        enabled = [e for e in (cfg.get('engines.enabled') or [])
                   if e in KNOWN_ENGINES]
        if enabled and target not in enabled:
            default = cfg.get('engines.default')
            target = default if default in enabled else enabled[0]
        # ...and the NSFW clamp must survive that fallback.
        if is_nsfw_label(img.variation_label) and target in API_ENGINES:
            target = origin if origin in LOCAL_ENGINES else 'klein'
    # Complete every fallible target-specific preflight before changing either
    # the row or its current file. Klein enqueue is itself part of preparation:
    # if the later DB transition fails, that exact new job is cancelled below.
    from ..job_queue import queue_manager
    old_state = {
        field: getattr(img, field) for field in (
            'filename', 'caption', 'status', 'fail_reason', 'fail_kind', 'job_id',
            'klein_model', 'variation_prompt', 'watermark_state',
            'watermark_bbox', 'watermark_regions', 'face_score', 'face_state',
            'content_sig', 'content_sig_stat')
    }
    old_path = (os.path.join(_dataset_path(img.dataset_id), img.filename)
                if img.filename else None)
    new_job_id = None
    api_generate = None
    aspect = None
    ref_bytes = None
    model = None
    if target in API_ENGINES:
        engine = target
        api_generate = _api_generate_fn(engine)
        ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
        if not os.path.exists(ref_path):
            raise ValueError('reference image file missing')
        aspect = aspect_for_label(img.variation_label, img.framing)
        ref_bytes = _all_ref_bytes(ds)  # principale + extras (multi-références)
    elif target == KREA_ENGINE:
        # Krea 2 Identity Edit: same shape as the Klein branch below, minus the
        # knobs it doesn't have. Its preflight raises KreaModelsMissing HERE,
        # before the row transition — so the tile keeps its current image.
        engine = KREA_ENGINE
        from . import krea_edit_helper as _keh
        ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
        new_job_id = _keh.enqueue_krea_edit(
            user_id=str(user_id), source_filename=ds.ref_filename,
            source_path=ref_path,
            edit_prompt=wrap_variation_krea(
                prompt, nsfw=is_nsfw_label(img.variation_label),
                framing=img.framing,
                suffix=dataset_prompt_suffix(ds, img.framing),
                subject_type=subject_type_of(ds),
                label=img.variation_label or ''),
            aspect_ratio=aspect_for_label(img.variation_label, img.framing),
            generation_loras=_keh.resolve_generation_lora_preset(generation_lora_preset),
            extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                            'variation_label': img.variation_label})
    else:
        try:
            from .klein_edit_helper import enqueue_klein_edit, resolve_generation_lora_preset
        except ImportError:
            raise RuntimeError('ComfyUI is not configured')
        # The route cannot know the effective engine when this is an ordinary
        # retry: it may be a Krea/API row, or a Klein row with a model filename.
        # Check Klein-only nodes only after the row's target has been resolved,
        # before modifying its current file/state.
        from . import klein_edit_helper as _kleh
        missing_nodes = _kleh.klein_missing_nodes()
        if missing_nodes:
            raise KleinNodesMissing(_kleh.klein_missing_assets(), missing_nodes)
        # Klein target: keep the row's real model file when it has one; a row born
        # on an API engine holds an engine TAG here, not a model — use the
        # workspace's Klein pick instead (None = enqueue's default model).
        # …and when the workspace sent none either, the DATASET's pick (None = auto).
        model = (img.klein_model if img.klein_model not in API_ENGINES
                 else ((klein_model or '').strip() or dataset_klein_model(ds)))
        ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
        extra_paths = [os.path.join(_dataset_path(ds.id), fn)
                       for fn in extra_ref_filenames(ds)]
        new_job_id = enqueue_klein_edit(
            user_id=str(user_id), source_filename=ds.ref_filename,
            source_path=ref_path,
            edit_prompt=wrap_variation_klein(
                prompt, nsfw=is_nsfw_label(img.variation_label),
                framing=img.framing,
                # CURRENT dataset suffix, applied at wrap: `prompt` is the raw
                # stored/edited creative prompt, so this is the ONLY application.
                suffix=dataset_prompt_suffix(ds, img.framing),
                subject_type=subject_type_of(ds),
                label=img.variation_label or ''),
            klein_model=model,
            lora_strength=lora_strength, extra_ref_paths=extra_paths,
            generation_loras=resolve_generation_lora_preset(generation_lora_preset),
            sampler_steps=_generation_steps(),
            base_lora_strength=_generation_base_lora_strength(),
            extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                            'variation_label': img.variation_label})

    # Persist the replacement state first. The old file remains in place until
    # this commit succeeds, eliminating rows that reference an already-moved file.
    try:
        if old_state['status'] == 'pending' and not old_state['filename'] \
                and old_state['job_id']:
            if not queue_manager.cancel_job(
                    old_state['job_id'], str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'The previous generation still has unconfirmed ComfyUI work; cancel it safely first.')
        if edited:
            img.variation_prompt = stored_prompt
        _clear_watermark_metadata(img)
        img.face_score = None
        img.content_sig = None
        img.content_sig_stat = None
        img.face_state = None
        # Engine TAG for the API engines and for Krea (each resolves its own
        # model); the real model FILE for Klein.
        img.klein_model = (engine if target in API_ENGINES or target == KREA_ENGINE
                           else model)
        img.filename = None
        # The row loses its file AND its words; the stamp goes with them, or the
        # regenerated image would be born already protected against captioning.
        caption_origin.stamp(img, None, None)
        img.status = 'pending'
        img.job_id = new_job_id
        img.fail_reason = None
        img.fail_kind = None
        db.session.commit()
    except Exception:
        db.session.rollback()
        if new_job_id:
            try:
                queue_manager.cancel_job(new_job_id, str(user_id), 'image')
            except Exception:
                logger.exception('regenerate: failed to cancel unlinked job %s',
                                 new_job_id)
        raise

    # The DB no longer references the old filename. If Trash itself fails, put
    # the exact previous row state back and cancel the prepared Klein job.
    try:
        if old_path and os.path.exists(old_path):
            trash.send_to_trash(
                old_path, context=f'dataset-{img.dataset_id}-regenerate-{img.id}')
    except Exception:
        try:
            # The row must keep the replacement job identity unless its exact
            # cancellation is committed in the same restoration transaction.
            if new_job_id and not queue_manager.cancel_job(
                    new_job_id, str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'The replacement generation still has unconfirmed ComfyUI work.')
            for field, value in old_state.items():
                setattr(img, field, value)
            db.session.commit()
        except Exception:
            db.session.rollback()
            logger.exception('regenerate: failed to restore row %s after Trash error',
                             image_id)
        raise

    # API target ('nanobanana'/'chatgpt' — requested, or the row's origin when
    # no engine was given): the row's klein_model column carries the engine tag.
    # With an `app` handle the call runs in a background thread (the row flips
    # to in-flight IMMEDIATELY so the tile shows "…" and the polling/banner UI
    # reacts at once); without it the call is synchronous (test path / legacy
    # callers).
    if target in API_ENGINES:
        if app is not None:
            # Threaded path: _run_nanobanana_batch owns the 'generate' indicator
            # (begin/bump/end) so a single API regenerate takes the same lock as a
            # batch — every concurrent action stays disabled until it finishes.
            try:
                threading.Thread(target=_run_nanobanana_batch,
                                 args=(app, [(img.id, prompt, aspect,
                                              dataset_prompt_suffix(ds, img.framing))],
                                       ref_bytes, engine, img.dataset_id),
                                 daemon=True).start()
            except Exception as e:
                img.status = 'failed'
                img.fail_reason = f'{engine}: failed to start generation: {e}'[:500]
                db.session.commit()
                raise
            return engine
        # Synchronous path (legacy / no-app callers): guard the same 'generate'
        # indicator directly so the payload advertises the regenerate too, and a
        # raise never leaks the entry (finally end()).
        token = None
        try:
            token = dataset_activity.begin(
                img.dataset_id, 'generate', total=1, engine=engine)
            gen_kwargs = {'aspect_ratio': aspect}
            if engine == 'chatgpt':
                from .chatgpt_image import _use_subscription
                gen_kwargs['force_lane'] = 'subscription' if _use_subscription() else 'api'
            try:
                out = api_generate(
                    ref_bytes,
                    wrap_variation(prompt, ref_count=len(ref_bytes),
                                   suffix=dataset_prompt_suffix(ds, img.framing),
                                   subject_type=subject_type_of(ds)),
                    **gen_kwargs)
            except EngineRefused as e:
                # Même règle que dans le lot : un refus se nomme, il ne se
                # déguise pas en panne (et il n'invente pas de contournement).
                out = None
                img.status = 'failed'
                img.fail_reason = f'{engine}: {str(e)[:400]}'
                img.fail_kind = 'refused'
                db.session.commit()
                return engine
            except SubscriptionQuotaExceeded:
                out = None
                img.status = 'failed'
                img.fail_reason = _QUOTA_MSG
                img.fail_kind = 'error'
                db.session.commit()
                return engine
            except SubscriptionUnavailable as e:
                out = None
                img.status = 'failed'
                img.fail_reason = f'chatgpt: {e}'
                img.fail_kind = 'error'
                db.session.commit()
                return engine
            if out:
                fn = f"{user_id}_{_ENGINE_FILE_TAG[engine]}_{uuid.uuid4().hex[:8]}.webp"
                write_image_atomic(os.path.join(_dataset_dir(img.dataset_id), fn),
                                   normalize_to_webp(out))
                img.filename = fn
            else:
                img.status = 'failed'
                img.fail_reason = f'{engine}: {_EMPTY_MSG}'
                img.fail_kind = 'empty'
            db.session.commit()
            return engine
        except Exception as e:
            db.session.rollback()
            current = db.session.get(FaceDatasetImage, image_id)
            if current and current.filename is None:
                current.status = 'failed'
                current.fail_reason = f'{engine}: {e}'[:500]
                current.fail_kind = 'error'
                db.session.commit()
            raise
        finally:
            if token is not None:
                dataset_activity.end(token)

    # Advertise the in-flight Klein job so a single regenerate takes the same lock
    # as a batch; link_completed_dataset_image clears it on completion.
    _sync_generate_activity(img.dataset_id)
    return new_job_id


# --- Fan-out generation (API engines: Nano Banana / ChatGPT) ---------------
# Both engines share the exact generate_variation contract (refs + prompt +
# aspect -> bytes|None), so the whole fan-out below is engine-parametric. The
# filename tag keeps the provenance readable in the dataset folder.
# APPEND-ONLY: both the engine ids and the file tags are persisted (dataset rows
# record the engine, and the tag is baked into the filename on disk), so a value
# here is never renamed or reordered.
API_ENGINES = ('nanobanana', 'chatgpt', 'openrouter')
_ENGINE_FILE_TAG = {'nanobanana': 'NBFace', 'chatgpt': 'GPTFace', 'openrouter': 'ORFace'}

# The LOCAL engines — they render on the user's own GPU through ComfyUI, cost
# nothing, and are the only ones allowed to receive NSFW shots. Klein is the
# historical one; Krea 2 Identity Edit is the second (krea_edit_helper).
# APPEND-ONLY for the same reason as API_ENGINES: 'krea' is persisted in
# FaceDatasetImage.klein_model as this row's engine tag.
LOCAL_ENGINES = ('klein', 'krea')
KREA_ENGINE = 'krea'
LOCAL_ENGINE_LABELS = {'klein': 'Klein', 'krea': 'Krea 2 Edit'}
# Every engine a generate/regenerate request may name.
KNOWN_ENGINES = LOCAL_ENGINES + API_ENGINES

# Human names for the engines, in the SAME wording as the frontend's ENGINE_LABELS
# (frontend/src/components/dataset/engineSelection.js). Only used to word messages
# — the ids above are the persisted values. A contract test pins ids AND labels
# across the two languages, so neither side can grow an engine on its own.
API_ENGINE_LABELS = {'nanobanana': 'Nano Banana Pro', 'chatgpt': 'ChatGPT',
                     'openrouter': 'OpenRouter'}


def engine_labels():
    """Every engine id -> its human label, both lanes. Merged rather than kept as a
    third dict: the two halves are already the source of truth for their side."""
    return dict(LOCAL_ENGINE_LABELS, **API_ENGINE_LABELS)


def editable_engines():
    """Engines /ref/edit accepts, LOCAL ones first (free, on the user's own GPU),
    then the paid API ones — the canonical order the workspace cards use too.

    A FUNCTION, not a constant: it is derived at call time from the two tuples
    above, so a lane growing an engine reaches the edit path with no second edit
    here. Klein and Krea used to be excluded because the edit ran as a blocking
    provider call and they have no blocking call to make; they now ride the same
    ComfyUI queue as every other local render, so the exclusion had outlived its
    reason — and it was the reason the app's only FREE edit lane was invisible."""
    return tuple(LOCAL_ENGINES) + tuple(API_ENGINES)


def edit_engine_choice_message():
    """The refusal for a non-editable engine, DERIVED from editable_engines():
    "pick Klein, Krea 2 Edit, Nano Banana Pro, ChatGPT or OpenRouter". Hardcoding
    the sentence is how the previous one ("pick ChatGPT or Nano Banana") kept
    naming two engines after a third became editable — a message that lies is
    worse than no message."""
    labels = engine_labels()
    names = [labels.get(e, e) for e in editable_engines()]
    if not names:
        return 'no image engine can edit the reference'
    head, last = names[:-1], names[-1]
    return 'pick ' + (f"{', '.join(head)} or {last}" if head else last)

from .chatgpt_image import SubscriptionQuotaExceeded, SubscriptionUnavailable
from .engine_errors import EngineError, EngineFatal, EngineRefused

# La phrase des moteurs qui ne SAVENT pas pourquoi ils rentrent bredouilles
# (ChatGPT, OpenRouter : 200 sans image, aucune métadonnée de refus lisible).
# Nano Banana ne passe plus par là — il lève NanoBananaRefused avec la vraie
# cause. Ce qui reste ici doit donc dire l'ambiguïté au lieu de la trancher :
# l'ancienne version promettait « retry usually works », ce qui était faux la
# moitié du temps et envoyait réessayer un refus définitif. On nomme l'ignorance.
_EMPTY_MSG = ('empty response — no image and no reason given (a content-policy '
              'refusal and a transient API error look identical here)')

_QUOTA_MSG = ('chatgpt: subscription image quota reached — remaining rows were '
              'stopped; rerun in API-key mode or wait for your plan quota to reset')
_LOST_MSG = ('chatgpt: subscription connection lost — remaining rows stopped; '
             'reconnect in Settings, then regenerate')


def _api_generate_fn(engine):
    if engine == 'chatgpt':
        from .chatgpt_image import generate_variation
    elif engine == 'openrouter':
        from .openrouter import generate_variation
    else:
        from .nanobanana import generate_variation
    return generate_variation


def _run_nanobanana_batch(app, items, ref_bytes, engine='nanobanana', dataset_id=None):
    """Worker body: generate each (image_id, prompt) via the selected API engine
    and link the result. Runs in a background thread (factored out so tests can
    call it synchronously). Each row commits independently; an API failure marks
    that row 'failed' (visible + regenerable) without stopping the batch.

    ``dataset_id`` (when known) drives the 'generate' activity indicator: one
    begin() with total=len(items), a bump() per item handled (success OR fail),
    and end() in a finally — so the ⚡ Generate button (and every concurrent
    action) stays disabled for the WHOLE batch, and the indicator can never leak
    even if a row raises. Also used for single-image API regenerate (items=1),
    which therefore takes the same lock. ``None`` = no indicator (legacy callers)."""
    api_generate = _api_generate_fn(engine)
    from concurrent.futures import ThreadPoolExecutor
    # Guard d'identité adapté au nombre de références (multi = « use EVERY ref »).
    n_refs = len(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else 1
    tag = _ENGINE_FILE_TAG.get(engine, 'NBFace')
    # Pin the ChatGPT auth lane ONCE for the whole batch. Without this, a
    # mid-batch token refresh failure (auth.openai.com non-200 -> logout())
    # would make every later row's OWN _use_subscription() call see
    # connected=False and silently reroute onto the paid API key — breaking
    # the feature's headline invariant. Pinning + stopping the batch instead
    # (via SubscriptionUnavailable below) closes that hole.
    force_lane = None
    if engine == 'chatgpt':
        from .chatgpt_image import _use_subscription
        force_lane = 'subscription' if _use_subscription() else 'api'
    # Set the moment ANY row hits the plan quota (or the pinned subscription
    # lane loses its token) — every later row would fail too, so the rest of
    # the batch fails fast instead of burning one call each.
    quota_exhausted = threading.Event()
    stop_msg = {'text': _QUOTA_MSG}   # set to the actual stop reason when it fires
    # Refus fournisseur du lot. Une liste (append est atomique sous le GIL) pour
    # que le log de fin rende un DÉCOMPTE au lieu de laisser 12 trous sur 40 se
    # découvrir tuile par tuile. L'UI, elle, recompte depuis fail_kind.
    refused = []
    token = dataset_activity.begin(dataset_id, 'generate', total=len(items), engine=engine) \
        if dataset_id is not None else None

    def _run_one(item):
        # item = (image_id, prompt, aspect, suffix) ; aspect optionnel (rétro-compat
        # → '1:1'), suffix optionnel (direction créative du dataset, déjà composée
        # par cadrage au call-site — rétro-compat → '').
        image_id, prompt = item[0], item[1]
        aspect = item[2] if len(item) > 2 else '1:1'
        suffix = item[3] if len(item) > 3 else ''
        # subject_type optionnel (retro-compat -> 'human' = lock historique).
        subject_type = item[4] if len(item) > 4 else 'human'
        # Stop AVANT l'appel API : cancel_pending supprime les lignes en vol — si
        # celle-ci a disparu, ne pas payer une génération qui sera jetée (le bouton
        # Stop doit économiser le RESTE du batch, pas seulement masquer les tuiles).
        with app.app_context():
            row = db.session.get(FaceDatasetImage, image_id)
            if row is None or row.status != 'pending':
                logger.info(f"{engine} batch: row {image_id} cancelled - API call skipped")
                return
        if quota_exhausted.is_set():
            # A previous row hit the plan quota: skip the API for every row not
            # yet started (later calls would 429 too). Up to max_workers rows may
            # already be in flight past this check when the event trips — each is
            # still failed via the dedicated except below, so the batch wastes at
            # most ~max_workers calls, not all.
            with app.app_context():
                img = db.session.get(FaceDatasetImage, image_id)
                if img is not None:
                    img.status = 'failed'
                    img.fail_reason = stop_msg['text']
                    img.fail_kind = 'error'
                    db.session.commit()
            return
        out = None
        fail_reason = None
        fail_kind = 'error'
        gen_kwargs = {'aspect_ratio': aspect}
        if engine == 'chatgpt':
            gen_kwargs['force_lane'] = force_lane
        try:
            out = api_generate(ref_bytes,
                               wrap_variation(prompt, ref_count=n_refs, suffix=suffix,
                                              subject_type=subject_type),
                               **gen_kwargs)
            if not out:
                # api_generate signale certains refus/vides par un retour falsy
                # sans lever — sans raison, la tuile "failed" resterait muette.
                fail_reason = f'{engine}: {_EMPTY_MSG}'
                fail_kind = 'empty'
        except EngineRefused as e:
            # Le fournisseur a répondu, et a refusé CETTE image. Ce n'est pas une
            # panne : on ne coupe pas le lot (les lignes suivantes ont leurs
            # chances — le filtre n'est pas déterministe), on nomme la cause et
            # on la compte à part pour pouvoir en rendre un décompte honnête.
            refused.append(image_id)
            logger.info(f"{engine} batch: refused at row {image_id}: {e}")
            fail_reason = f'{engine}: {str(e)[:400]}'
            fail_kind = 'refused'
        except SubscriptionQuotaExceeded as e:
            quota_exhausted.set(); stop_msg['text'] = _QUOTA_MSG
            logger.warning(f"{engine} batch: quota exhausted at row {image_id}: {e}")
            fail_reason = _QUOTA_MSG
        except SubscriptionUnavailable as e:
            quota_exhausted.set(); stop_msg['text'] = _LOST_MSG
            logger.warning(f"{engine} batch: subscription lost at row {image_id}: {e}")
            fail_reason = _LOST_MSG
        except EngineFatal as e:
            # No key, key rejected, no credits, unknown or unusable model: every
            # remaining row would fail on the exact same cause, so stop the batch
            # instead of asking the provider the same refused question once per
            # image. Reuses the ChatGPT quota machinery — same need, same shape.
            # Raised by all three API engines (see services/engine_errors.py):
            # the model of each is user-editable, and a wrong slug must fail once,
            # loudly, not N times in a row.
            msg = f'{engine}: {str(e)[:400]} — remaining rows were stopped'
            quota_exhausted.set(); stop_msg['text'] = msg
            logger.warning(f"{engine} batch: fatal at row {image_id}: {e}")
            fail_reason = msg
        except Exception as e:
            logger.warning(f"{engine} batch: generation error for row {image_id}: {e}")
            fail_reason = f'{engine}: {str(e)[:400]}'
        with app.app_context():
            img = db.session.get(FaceDatasetImage, image_id)
            if img is None:
                return
            if out:
                ds = db.session.get(FaceDataset, img.dataset_id)
                if ds is None:
                    # The whole DATASET was deleted while this batch ran. The row
                    # above survived only because its cascade has not landed yet;
                    # reading ds.user_id here raised AttributeError, which escaped
                    # _run_one and abandoned every REMAINING item of the batch.
                    # Nothing to write this result to, so drop it and let the rest
                    # of the batch finish.
                    logger.info('%s batch: dataset gone for row %s, dropping the '
                                'result', engine, image_id)
                    return
                fn = f"{ds.user_id}_{tag}_{uuid.uuid4().hex[:8]}.webp"
                try:
                    # Conserve le ratio demandé (pas de letterbox carré sur les corps).
                    write_image_atomic(os.path.join(_dataset_dir(img.dataset_id), fn),
                                       normalize_to_webp(out))
                    img.filename = fn
                except Exception as e:
                    logger.warning(f"{engine} batch: save failed for row {image_id}: {e}")
                    img.status = 'failed'
                    img.fail_reason = f'saving the image failed: {str(e)[:400]}'
                    img.fail_kind = 'error'
            else:
                img.status = 'failed'
                img.fail_reason = fail_reason
                img.fail_kind = fail_kind
            db.session.commit()

    def _one(item):
        # Progress-tracking wrapper: bump the indicator once per item handled,
        # whatever the outcome (a raised _run_one still counts as one handled and
        # never strands the counter). No-op when token is None (bump(None)).
        try:
            return _run_one(item)
        finally:
            dataset_activity.bump(token)

    logger.info(f"{engine} batch: start ({len(items)} variation(s))")
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(_one, items))
    finally:
        dataset_activity.end(token)   # idempotent; end(None) is a no-op
    tally = f" — {len(refused)} refused by the provider" if refused else ''
    logger.info(f"{engine} batch: done ({len(items)} variation(s)){tally}")


def generate_variations_nanobanana(app, user_id, dataset_id, variations, multiplier,
                                   engine='nanobanana'):
    """API fan-out (Nano Banana or ChatGPT, per `engine`): pre-create pending
    rows (job_id stays None - that is the marker for API-generated rows), then
    fill them from a background thread. The existing polling/banner/cancel UI
    works unchanged (pending + no file = in flight). Returns the created ids."""
    _guard_not_bank_export(dataset_id)
    if engine not in API_ENGINES:
        raise ValueError(f'unknown API engine: {engine}')
    # Fail-closed : les variations NSFW ne partent JAMAIS vers un moteur API
    # (comptes/API tiers) — elles n'existent que sur le chemin Klein local.
    if any(v.get('nsfw') or is_nsfw_label(v.get('label')) for v in variations):
        raise ValueError('NSFW variations run on the local Klein engine only')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
    ref_path = _ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference image file missing')
    mult = max(1, int(multiplier))
    total = len(variations) * mult
    if total == 0:
        raise ValueError('no variations selected')
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    # Principale + refs additionnelles : Nano Banana s'appuie sur toutes les
    # images pour la cohérence d'identité (une seule = comportement historique).
    ref_bytes = _all_ref_bytes(ds)
    subject_type = subject_type_of(ds)   # steers the identity lock at wrap time

    ids, items = [], []
    for v in variations:
        for _ in range(mult):
            # klein_model=<engine> marks API-generated rows (the regenerate
            # path dispatches on it; never collides with real .safetensors names).
            img = FaceDatasetImage(dataset_id=dataset_id, source='generated', status='pending',
                                   variation_label=v.get('label'), framing=v.get('framing'),
                                   variation_prompt=v['prompt'], klein_model=engine, job_id=None)
            db.session.add(img)
            db.session.commit()
            ids.append(img.id)
            # Suffix composed HERE (per-framing) and carried by the work item: the
            # row keeps the raw prompt, the batch worker applies it at wrap time.
            items.append((img.id, v['prompt'],
                          aspect_for_label(v.get('label'), v.get('framing')),
                          dataset_prompt_suffix(ds, v.get('framing')),
                          subject_type))

    threading.Thread(target=_run_nanobanana_batch,
                     args=(app, items, ref_bytes, engine, dataset_id),
                     daemon=True).start()
    return ids


# --- Completion linking (called from the job queue) -------------------------
def link_completed_dataset_image(job_id, filename, failed=False, reason=None):
    """Attach a finished fan-out job to its FaceDatasetImage row.

    Called from the job-queue completion/failure/cancel paths, which may run in
    a long-lived monitor thread whose SQLAlchemy session holds a STALE read
    snapshot (rows committed by other threads are invisible). If the first
    lookup misses, end the transaction (rollback) and retry on a fresh snapshot
    before concluding the row really doesn't exist.
    `reason` (the job row's error_message, e.g. a ComfyUI execution error) shows
    on the failed tile so the user sees WHY, not a generic 'see the log'."""
    img = FaceDatasetImage.query.filter_by(job_id=job_id).first()
    if img is None:
        db.session.rollback()  # drop the stale read snapshot, then re-read
        img = FaceDatasetImage.query.filter_by(job_id=job_id).first()
    if img is None:
        logger.warning(f"dataset link: no FaceDatasetImage row for job {job_id}")
        return
    if img.derivation_kind == KLEIN_SMALL_IMAGE and img.status in ('keep', 'reject'):
        # The user already resolved the pair while this job/callback was racing.
        # The terminal review decision wins: do not attach
        # a late file and do not turn reject into failed. This is a temporary,
        # unlinked Comfy output (never user data), so direct removal is intentional.
        output_dir = _comfy_output_dir()
        late_output = os.path.join(output_dir, filename) if output_dir and filename else None
        if late_output and os.path.isfile(late_output):
            try:
                os.remove(late_output)
            except OSError:
                pass
        try:
            _sync_generate_activity(img.dataset_id)
        except Exception:
            logger.exception(
                'dataset link: terminal rescue activity sync failed for job %s', job_id)
        return
    if failed:
        # A cancel racing with the worker dispatches a failure callback. Never let
        # that callback overwrite an already-resolved rescue choice (keep/reject).
        if not (img.derivation_kind == KLEIN_SMALL_IMAGE
                and img.status in ('keep', 'reject')):
            img.status = 'failed'
            img.fail_reason = (img.fail_reason or reason
                               or 'Klein generation failed (see 🪵 Server log in Settings for the ComfyUI error)')
    else:
        output_dir = _comfy_output_dir()
        src = os.path.join(output_dir, filename) if output_dir else None
        dst = os.path.join(_dataset_dir(img.dataset_id), filename)
        if src and os.path.exists(src) and os.path.exists(dst):
            # Collision guard: NEVER overwrite another tile's file. ComfyUI's
            # SaveImage counter re-issued the same name when earlier results
            # were moved out of its output folder — every tile then displayed
            # the same (last) image. The prefix is unique per job now, but a
            # residual collision must degrade to a rename, not a silent loss.
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
            dst = os.path.join(_dataset_dir(img.dataset_id), filename)
            logger.warning(f"dataset link: name collision, storing as {filename}")
        img.filename = filename
        if src and os.path.exists(src):
            shutil.move(src, dst)          # file where we expected it on disk
        elif os.path.exists(dst):
            pass                           # already brought in (retry / dup completion)
        else:
            # The file isn't on disk where we look — ComfyUI was pointed at a
            # custom output path, or none is configured. Fetch it over the /view
            # API instead (path-independent, like other ComfyUI front-ends). #2
            from ..utils.comfyui import fetch_output_image_bytes
            data = fetch_output_image_bytes(filename)
            if data:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'wb') as f:
                    f.write(data)
            else:
                img.status = 'failed'
                img.fail_reason = ('The finished image could not be retrieved from ComfyUI '
                                   '(not on disk, and the /view API fetch failed).')
                logger.warning(f"dataset link: file not on disk and /view API fetch failed (job {job_id})")
        # A user may have marked this in-flight improvement Keep while waiting.
        # Only the freshly linked, on-disk result may now replace its parent;
        # the helper also preserves a later explicit return to Pending.
        _unkeep_parent_for_kept_improvement(img)
    db.session.commit()
    # This job just left the in-flight set: reconcile the Klein 'generate'
    # indicator (clears it when this was the last job of the batch). Guarded — a
    # bookkeeping hiccup must never break completion linking; the TTL is the net.
    try:
        _sync_generate_activity(img.dataset_id)
    except Exception:
        logger.exception(f"dataset link: generate-activity sync failed for job {job_id}")


# --- Migration helper (run once manually after deploy) ---------------------
def migrate_existing_images_to_per_dataset():
    """Migration helper - run once manually after deploy. Not called automatically."""
    counts = {'moved': 0, 'skipped': 0, 'missing': 0}
    output_dir = _comfy_output_dir()
    if output_dir is None:
        return counts
    datasets = FaceDataset.query.all()
    for ds in datasets:
        if ds.ref_filename:
            src = os.path.join(output_dir, ds.ref_filename)
            dst = os.path.join(_dataset_dir(ds.id), ds.ref_filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                counts['moved'] += 1
            elif os.path.exists(dst):
                counts['skipped'] += 1
            else:
                counts['missing'] += 1
        for img in FaceDatasetImage.query.filter_by(dataset_id=ds.id).all():
            if not img.filename:  # pending/failed rows without a file
                continue
            src = os.path.join(output_dir, img.filename)
            dst = os.path.join(_dataset_dir(img.dataset_id), img.filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                counts['moved'] += 1
            elif os.path.exists(dst):
                counts['skipped'] += 1
            else:
                counts['missing'] += 1
    return counts


# --- Export ----------------------------------------------------------------
_TRAIN_FAMILY_LABELS = {
    'zimage': 'Z-Image',
    'krea': 'Krea 2',
    'flux2klein': 'FLUX.2 Klein',
    'flux': 'FLUX.1',
    'sdxl': 'SDXL',
    'anima': 'Anima',
}


def _dataset_info(ds, n, composition) -> str:
    """Factual, family/kind-aware README without stale tuning advice."""
    family = (getattr(ds, 'train_type', None) or 'zimage').lower()
    kind = (getattr(ds, 'kind', None) or 'character').lower()
    lines = [
        '# LoRA Dataset Studio export',
        '',
        f'Dataset kind: {kind}',
        f'Training family: {_TRAIN_FAMILY_LABELS.get(family, family)}',
        f'Images: {n}',
        f'Composition: {composition}',
        '',
    ]
    if kind == 'style':
        lines.extend([
            'Activation: always-on Style (no trigger token).',
            'Captions describe visible content only; the aesthetic is omitted.',
        ])
    else:
        lines.extend([
            f'Activation token: {ds.trigger_word}',
            'Caption sidecars already include this token.',
        ])
    return '\n'.join(lines) + '\n'


def style_content_caption(ds, caption) -> str:
    """Return a Style caption without a legacy leading internal identifier.

    New captions are already content-only. This final seam also repairs sidecars
    generated by older LDS releases (``trigger, content``) without deleting an
    ordinary content word that merely happens to equal the id: only an exact id or
    an id followed by explicit caption punctuation is stripped.
    """
    cap = (caption or '').strip()
    if not is_style(ds):
        return cap
    trigger = (getattr(ds, 'trigger_word', None) or '').strip()
    if not trigger:
        return cap
    if cap.strip(' .!?:;,').strip().casefold() == trigger.casefold():
        return ''
    return re.sub(
        rf'^{re.escape(trigger)}\s*[,;:.!?]\s*', '', cap,
        count=1, flags=re.IGNORECASE).strip()


def _export_caption(ds, caption) -> str:
    """The exact text a trainer reads for one image: the dataset trigger prepended
    to the stored caption for character/concept datasets. A style LoRA is always-on:
    its sidecars contain CONTENT ONLY, with no hidden activation token. Single source
    of truth shared by the ZIP export and write_caption_files, so on-disk .txt
    sidecars always match what the ZIP would contain."""
    cap = style_content_caption(ds, caption)
    if is_style(ds):
        return cap
    return f"{ds.trigger_word}, {cap}" if cap else ds.trigger_word


def write_export_zip(user_id: int, dataset_id: int, output: BinaryIO) -> None:
    """Training-ready ZIP in the PUBLIC-TOOL layout, not an app-internal format:
    one `10_<trigger>/` folder of `image.png` + same-stem `image.txt` caption
    pairs (captions carry the resolved trigger inline, except always-on Style
    datasets whose sidecars are content-only). That single shape feeds
    every mainstream trainer as-is: ai-toolkit (point the dataset at the folder;
    the folder name is ignored), kohya_ss / sd-scripts (drop under img/ — the
    `10_` prefix IS kohya's repeats convention), OneTrainer & friends (image+txt
    pairs). The info file is .md so no caption-scanner ever picks it up."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    kept = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    if not kept:
        raise ValueError('no kept images to export')
    safe = ''.join(c for c in ds.name if c.isalnum() or c in ('-', '_')) or 'dataset'
    safe_trigger = ''.join(c for c in ds.trigger_word if c.isalnum() or c in ('-', '_')) or 'lora'
    folder = f"10_{safe_trigger}"
    comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Garder la PHOTO RÉELLE de référence dans le set : les datasets 100 %
        # synthétiques dérivent de la distribution réelle (deep-research 2026-06-14).
        # On l'inclut comme ancre réelle (_000), caption = trigger seul.
        ref_path = _ref_path(ds) if ds.ref_filename else ''
        # The reference row has no content caption. Exporting it for a Style set
        # would force either a blank sidecar or the internal run identifier into
        # training, both of which violate the always-on/content-only contract.
        if ref_path and os.path.exists(ref_path) and not is_style(ds):
            try:
                rpng = io.BytesIO()
                with Image.open(ref_path) as source:
                    ImageOps.exif_transpose(source).convert('RGB').save(rpng, 'PNG')
                zf.writestr(f"{folder}/{safe}_000_ref.png", rpng.getvalue())
                zf.writestr(f"{folder}/{safe}_000_ref.txt", ds.trigger_word)
            except OSError:
                pass
        for n, img in enumerate(kept, 1):
            path = _img_path(img) if img.filename else ''
            if not img.filename or not os.path.exists(path):
                continue
            png = io.BytesIO()
            with Image.open(path) as source:
                ImageOps.exif_transpose(source).convert('RGB').save(png, 'PNG')
            base = f"{folder}/{safe}_{n:03d}"
            zf.writestr(f"{base}.png", png.getvalue())
            zf.writestr(f"{base}.txt", _export_caption(ds, img.caption))
            if img.framing in comp:
                comp[img.framing] += 1
        zf.writestr(f"{folder}/_dataset_info.md",
                    _dataset_info(ds, len(kept), comp))


def build_export_zip(user_id: int, dataset_id: int) -> bytes:
    """Compatibility wrapper for callers that still need an in-memory archive."""
    output = io.BytesIO()
    write_export_zip(user_id, dataset_id, output)
    return output.getvalue()


def write_caption_files(user_id, dataset_id) -> dict:
    """Write a kohya/ai-toolkit-style `<image>.txt` sidecar NEXT TO each kept
    captioned image in the dataset folder (data/datasets/<id>/) — same caption
    text as the ZIP export (trigger prepended except for content-only Style), for
    tools that read the folder directly instead of downloading the ZIP. Overwrites
    existing .txt files (it's a resync after re-captioning/edits); kept images
    without a caption are counted, not written — they'd get only a bare trigger
    (character/concept) or an empty Style sidecar, so caption them first. Returns
    {'ok', 'written', 'skipped_uncaptioned'}."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    kept = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    written = skipped_uncaptioned = removed_stale = 0
    for img in kept:
        if not img.filename or not os.path.exists(_img_path(img)):
            continue                       # nothing on disk to sit next to
        stem = os.path.splitext(os.path.basename(img.filename))[0]
        sidecar = os.path.join(_dataset_dir(dataset_id), f'{stem}.txt')
        if not (img.caption or '').strip():
            if os.path.isfile(sidecar):
                os.remove(sidecar)
                removed_stale += 1
            skipped_uncaptioned += 1
            continue
        body = _export_caption(ds, img.caption)
        if not body:                      # legacy Style caption = internal id only
            if os.path.isfile(sidecar):
                os.remove(sidecar)
                removed_stale += 1
            skipped_uncaptioned += 1
            continue
        with open(sidecar, 'w', encoding='utf-8') as fh:
            fh.write(body)
        written += 1
    return {'ok': True, 'written': written,
            'skipped_uncaptioned': skipped_uncaptioned,
            'removed_stale': removed_stale}
