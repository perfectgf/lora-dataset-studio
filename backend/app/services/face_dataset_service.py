"""Face-dataset orchestration: CRUD, fan-out, import, classify, caption, export.

The vision passes (classify/caption) call describe_image_ollama; the CALLER (the
route) is responsible for wrapping them in the GPU-exclusive window. The ComfyUI
output dir is resolved via `cfg.comfyui_dir('output')` so tests can monkeypatch cfg.
"""
from __future__ import annotations
from decimal import Decimal
import io
import json
import logging
import math
import ntpath
import os
import posixpath
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from types import SimpleNamespace
from typing import BinaryIO
from urllib.parse import urlsplit

from PIL import Image, ImageOps, UnidentifiedImageError

from ..extensions import db
from ..models import FaceDataset, FaceDatasetImage, LoraTestImage
from .. import config as cfg
from . import dataset_activity, trash
from .dataset_storage import dataset_path, ensure_dataset_dir

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
                              is_nsfw_label, prompt_by_label, wrap_variation,
                              wrap_variation_klein, get_identity_prompt,
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


def _ref_path(ds) -> str:
    return os.path.join(_dataset_dir(ds.id), ds.ref_filename)

_VALID_STATUS = ('pending', 'keep', 'reject', 'failed')
MAX_FANOUT = 60
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
# KLEIN_IMAGE_IMPROVE_PROMPT is the shipped DEFAULT of the editable klein_improve
# prompt (imported from face_variations, which owns the identity/quality prompt
# registry). Re-exported here so `svc.KLEIN_IMAGE_IMPROVE_PROMPT` keeps resolving.
_SMALL_IMAGE_DERIVATIONS = (SMALL_IMAGE_SOURCE, KLEIN_SMALL_IMAGE)
# A striped in-process lock is sufficient for LDS's single local server process
# and makes the active-candidate check + row creation + enqueue one critical
# section.  In particular, a second simultaneous lightbox click waits until the
# first row has its job_id, then takes the idempotent return path below.
_IMAGE_IMPROVE_LOCKS = tuple(threading.Lock() for _ in range(64))
# A mirror is a toggle: two requests for the same image must run in order (two
# clicks restore the original orientation), not both read the same source pixels
# and race to promote an identical result.  Stripes avoid an unbounded lock map.
_IMAGE_MIRROR_LOCKS = tuple(threading.Lock() for _ in range(64))


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


def extra_ref_filenames(ds) -> list:
    """Références additionnelles du dataset (JSON en base, parse tolérant)."""
    try:
        v = json.loads(ds.ref_extra_filenames or '[]')
    except (ValueError, TypeError):
        return []
    return [f for f in v if isinstance(f, str)] if isinstance(v, list) else []


def _all_ref_bytes(ds) -> list:
    """Bytes de la référence principale puis des extras présents sur disque
    (ordre stable, principale d'abord - c'est elle que Gemini doit prioriser).
    Un extra au fichier manquant est ignoré silencieusement (jamais bloquant)."""
    with open(_ref_path(ds), 'rb') as fh:
        out = [fh.read()]
    for fn in extra_ref_filenames(ds):
        p = os.path.join(_dataset_dir(ds.id), fn)
        try:
            with open(p, 'rb') as fh:
                out.append(fh.read())
        except OSError:
            logger.warning(f"dataset {ds.id}: extra ref missing on disk: {fn}")
    return out


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
    with open(os.path.join(_dataset_dir(dataset_id), fn), 'wb') as fh:
        fh.write(normalize_to_webp(image_bytes))
    ds.ref_extra_filenames = json.dumps(extras + [fn])
    db.session.commit()
    return fn


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
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict) or value.get('platform') != 'pexels':
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
    if not isinstance(item, dict) or item.get('platform') != 'pexels':
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


def caption_options(ds) -> dict:
    """Normalized per-dataset caption overrides: {backend, ollama_model, instructions}.
    Empty strings = "use the global default". Never raises ({} defaults on a missing or
    corrupt blob) so every caption path can read it unconditionally."""
    out = {'backend': '', 'ollama_model': '', 'instructions': '', 'vocabulary': ''}
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
    out['ollama_model'] = str(data.get('ollama_model') or '').strip()
    out['instructions'] = str(data.get('instructions') or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    vocab = str(data.get('vocabulary') or '').strip().lower()
    if vocab in _CAPTION_VOCABULARIES:
        out['vocabulary'] = vocab
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
        cur['ollama_model'] = str(patch.get('ollama_model') or '').strip()
    if 'instructions' in patch:
        cur['instructions'] = str(patch.get('instructions') or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    if 'vocabulary' in patch:
        v = str(patch.get('vocabulary') or '').strip().lower()
        if v and v not in _CAPTION_VOCABULARIES:
            raise ValueError(f'invalid caption vocabulary: {v}')
        cur['vocabulary'] = v
    stored = {k: v for k, v in cur.items() if v}
    ds.caption_options = json.dumps(stored) if stored else None
    db.session.commit()
    return cur


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


def _combined_caption_instructions(opts) -> str:
    """The text appended to a caption prompt for a run: the vocabulary preset (if any),
    then the user's free-text instructions. Empty when neither is set — so a dataset that
    never touched the popover produces byte-identical prompts. Both ride at the END of the
    prompt, after the kind omission rules, and the output cleaners still post-filter."""
    parts = []
    preset = _VOCABULARY_INSTRUCTION.get(opts.get('vocabulary'))
    if preset:
        parts.append(preset)
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
TRAIN_TYPES = ('zimage', 'sdxl', 'krea', 'flux', 'flux2klein')


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


def create_dataset(user_id, name, trigger_word, kind=None, concept_desc=None, train_type=None,
                   fidelity=None, prompt_suffix=None, prompt_suffixes=None, *, commit=True):
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


def set_train_type(user_id, dataset_id, train_type) -> bool:
    """Change the target model family later (kept in sync with the TrainingPanel
    selector so the menu re-groups). Normalizes; unknown -> zimage. False if absent."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    ds.train_type = normalize_train_type(train_type)
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
                            prompt_suffixes=None):
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


def list_datasets(user_id):
    return (FaceDataset.query.filter_by(user_id=str(user_id))
            .order_by(FaceDataset.updated_at.desc()).all())


def dataset_list_stats(user_id):
    """Per-dataset aggregates for the library page — image counts and the
    families ever trained — in two grouped queries (never one per dataset).
    Returns {dataset_id: {'images_total', 'images_kept', 'images_captioned',
    'trained_families': [str]}}; datasets absent from a map just have zeros."""
    from sqlalchemy import case, func
    from ..models import TrainingRunRecord
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
    db.session.commit()
    return True


def _owned_image(user_id, image_id):
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return None
    ds = db.session.get(FaceDataset, img.dataset_id)
    return img if ds and str(ds.user_id) == str(user_id) else None


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


def set_image_caption(user_id, image_id, caption, short=_UNSET):
    """Save one image's long caption; optionally its short variant. `short` defaults to a
    sentinel so a caller that only edits the long caption (the inline grid textarea) never
    wipes an existing short — only the expanded editor passes `short` to touch it."""
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    img.caption = _cap_caption(caption) or None
    if short is not _UNSET:
        img.caption_short = _cap_caption(short) or None
    db.session.commit()
    return True


def _crop_resize_file(path, x, y, w, h, size=1024, dst=None):
    """Crop the file at `path` to (x,y,w,h) and resize the crop so its LONG side
    equals `size`, PRESERVING the box's aspect ratio: a square box keeps the
    historical size x size output, a 2:3 box yields 683x1024 — no padding, no
    distortion (ai-toolkit buckets handle non-square training images). Writes to
    `dst` (default: overwrite `path`). Passing a distinct `dst` lets the reference
    crop read the untouched full-frame ORIGINAL and write the derived crop — so a
    re-crop can widen back out instead of only tightening the previous crop.

    Returns (ok, upscale_ratio) — ratio is size / long_side_of_box (>1 means the
    box was smaller than `size` and got enlarged), or None on failure."""
    if not os.path.exists(path):
        return False, None
    src = Image.open(path).convert('RGB')
    box = (max(0, int(x)), max(0, int(y)), min(src.width, int(x + w)), min(src.height, int(y + h)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return False, None
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw >= bh:
        out_w, out_h = size, max(1, round(size * bh / bw))
    else:
        out_w, out_h = max(1, round(size * bw / bh)), size
    scale = size / max(bw, bh)
    out = io.BytesIO()
    src.crop(box).resize((out_w, out_h), Image.LANCZOS).save(out, 'WEBP', quality=92)
    with open(dst or path, 'wb') as fh:
        fh.write(out.getvalue())
    return True, scale


def crop_image(user_id, image_id, x, y, w, h):
    """Crop a dataset image to (x,y,w,h), resized to 1024 (no pad). Returns bool."""
    img = _owned_image(user_id, image_id)
    if not img or not img.filename:
        return False
    ok, scale = _crop_resize_file(_img_path(img), x, y, w, h)
    if ok:
        _clear_watermark_metadata(img)
        img.upscale_ratio = scale
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


def _mirrored_image_bytes(path):
    """Prepare a horizontal mirror fully in memory without touching ``path``.

    Dataset rows normally point at WEBP files, but restored/legacy datasets may
    contain PNG or JPEG bytes (even under a misleading extension).  Preserve the
    format Pillow actually detects: PNG stays lossless, WEBP is rewritten lossless
    so repeated toggles do not accumulate damage, and JPEG uses high-quality 4:4:4.
    """
    try:
        with Image.open(path) as src:
            fmt = (src.format or '').upper()
            if fmt not in {'PNG', 'WEBP', 'JPEG'}:
                raise ValueError(f'unsupported image format: {fmt or "unknown"}')
            if getattr(src, 'n_frames', 1) != 1:
                raise ValueError('animated images are not supported')
            src.load()
            icc = _valid_icc_profile(src.info.get('icc_profile'))
            oriented = ImageOps.exif_transpose(src)
            mirrored = ImageOps.mirror(oriented)

            save_kwargs = {}
            if icc:
                save_kwargs['icc_profile'] = icc
            if fmt == 'PNG':
                # Keep the native PNG mode/bit depth and use only lossless DEFLATE.
                save_kwargs.update(compress_level=6)
            elif fmt == 'WEBP':
                # WEBP input can carry alpha; RGB(A) preserves it while avoiding
                # encoder-dependent conversions for unusual legacy modes.
                has_alpha = 'A' in mirrored.getbands()
                mirrored = mirrored.convert('RGBA' if has_alpha else 'RGB')
                save_kwargs.update(lossless=True, quality=100, method=6)
            else:  # JPEG
                mirrored = mirrored.convert('RGB')
                save_kwargs.update(quality=95, subsampling=0, optimize=True)

            out = io.BytesIO()
            mirrored.save(out, fmt, **save_kwargs)
            payload = out.getvalue()
            expected_size = mirrored.size
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError('invalid image file') from e

    # Decode the exact encoded result before it is allowed near the live path.
    try:
        with Image.open(io.BytesIO(payload)) as check:
            check.load()
            if (check.format or '').upper() != fmt or check.size != expected_size:
                raise OSError('encoded mirror validation failed')
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ValueError('could not encode mirrored image') from e
    return payload


def mirror_image(user_id, image_id):
    """Permanently mirror one owned dataset image horizontally.

    Returns ``None`` for an unknown/foreign row, otherwise a cache-bust payload.
    The filename and all semantic/provenance metadata remain stable.  Only
    watermark metadata is cleared because its pixel coordinates are no longer
    valid after a horizontal flip.
    """
    lock = _IMAGE_MIRROR_LOCKS[
        hash((str(user_id), image_id)) % len(_IMAGE_MIRROR_LOCKS)]
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
            payload = _mirrored_image_bytes(path)
        except ValueError:
            raise
        except OSError as e:
            raise RuntimeError('could not read image file') from e

        tmp_path = None
        try:
            try:
                fd, tmp_path = tempfile.mkstemp(
                    prefix=f'.{os.path.basename(path)}.mirror-', suffix='.tmp',
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
                raise RuntimeError('could not prepare mirrored image') from e

            # Do not overwrite a crop/clean that raced this preparation outside
            # the mirror lock.  (All mirror requests themselves are serialized.)
            try:
                current = os.stat(path)
            except OSError as e:
                raise RuntimeError('image file missing') from e
            if (current.st_mtime_ns, current.st_size) != (before.st_mtime_ns, before.st_size):
                raise RuntimeError('image changed while mirroring; retry')

            watermark_snapshot = (
                img.watermark_state, img.watermark_bbox, img.watermark_regions)
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
                     img.watermark_regions) = watermark_snapshot
                    try:
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                        logger.exception(
                            'failed to restore watermark metadata after mirror promotion failure')
                raise RuntimeError('could not update image file') from e

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
                    logger.warning('could not remove mirror temp file %s', tmp_path)


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
    trashed_path = None
    try:
        if img.status == 'pending' and not img.filename and img.job_id:
            from ..job_queue import queue_manager
            queue_manager.cancel_job(
                img.job_id, str(user_id), 'image', commit=False)
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
    dataset_path = _dataset_path(dataset_id)
    trashed_path = None
    try:
        # Keep Studio queue cancellation atomic with deleting its owning rows.
        # Exact job_id + owned dataset scope prevents cross-dataset cancellation.
        from ..job_queue import queue_manager
        for img in imgs:
            if img.status == 'pending' and not img.filename and img.job_id:
                queue_manager.cancel_job(
                    img.job_id, str(user_id), 'image', commit=False)
        for cell in studio_rows:
            if (cell.job_id
                    and cell.status not in ('done', 'failed', 'cancelled')):
                queue_manager.cancel_job(
                    cell.job_id, str(user_id), 'image', commit=False)
        if os.path.exists(dataset_path):
            trashed_path = trash.send_to_trash(
                dataset_path, context=f'dataset-{dataset_id}')
        for img in imgs:
            db.session.delete(img)
        # Explicit for old databases whose FK definition cannot be altered by
        # db.create_all(). New databases also have ON DELETE CASCADE as a guard.
        for cell in studio_rows:
            db.session.delete(cell)
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


def cancel_pending(user_id, dataset_id):
    """Cancel all in-flight (pending) generations of a dataset and drop their
    rows. Returns the number cancelled."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    # Only in-flight generations (pending AND no result file yet) - leave
    # completed-but-uncurated images alone.
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).all())
    n = 0
    for img in rows:
        if img.job_id:  # Klein rows only - API rows never carry a job_id
            try:
                from ..job_queue import queue_manager
                queue_manager.cancel_job(img.job_id, str(user_id), 'image')
            except Exception:
                pass
        if img.derivation_kind == KLEIN_SMALL_IMAGE:
            # Preserve the review pair and its original file. A cancelled rescue
            # is equivalent to an engine failure: the original can still be kept.
            img.status = 'failed'
            img.fail_reason = 'Klein small-image rescue was cancelled.'
        else:
            db.session.delete(img)
        n += 1
    db.session.commit()
    # Stop deleted the in-flight rows: clear the Klein 'generate' indicator now
    # (its completion callbacks won't fire for cancelled jobs). An API batch's own
    # begin/end entry is untouched — its worker unwinds and end()s on its own.
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
BACKUP_VERSION = 1
_BACKUP_MAX_FILES = 600
_BACKUP_MAX_ROWS = 600
_BACKUP_MAX_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB uncompressed (zip-bomb guard)
_BACKUP_MAX_METADATA_BYTES = 4 * 1024 * 1024
_BACKUP_NAME_RE = re.compile(r'^[\w.-]+\.(webp|jpg|jpeg|png)$', re.IGNORECASE)

# Champs snapshotés tels quels par ligne image (job_id/klein_model exclus : liés
# à la machine source — un backup restauré ne peut pas « regénérer »).
_BACKUP_IMG_FIELDS = ('filename', 'source', 'framing', 'variation_label', 'status',
                      'caption', 'caption_short', 'variation_prompt', 'face_score', 'face_state',
                      'upscale_ratio', 'watermark_state', 'watermark_bbox',
                      'watermark_regions', 'parent_image_id', 'derivation_kind',
                      'fail_reason', 'source_metadata')


def _backup_basename(value):
    """Return a portable image basename, or None for paths/invalid values."""
    if not isinstance(value, str) or not value:
        return None
    if '/' in value or '\\' in value or not _BACKUP_NAME_RE.fullmatch(value):
        return None
    return value


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


def write_backup_zip(user_id: int, dataset_id: int, output: BinaryIO) -> None:
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
    ref_names = primary_ref_names + portable_extras
    image_file_names = {
        name.casefold(): name for img in rows
        if (name := _backup_basename(img.filename))
        and os.path.isfile(os.path.join(dsdir, name))
    }
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
    # backup_image_id is archive-local only. It lets restore remap parent_image_id
    # to the newly allocated row ids instead of retaining ids from the source DB.
    images_meta = []
    for img in rows:
        row = dict({'backup_image_id': img.id},
                   **{f: getattr(img, f) for f in _BACKUP_IMG_FIELDS})
        # Archive a structured, revalidated object rather than the raw TEXT
        # column. A malformed legacy/local row can never export arbitrary links.
        row['source_metadata'] = normalize_source_metadata(img.source_metadata)
        images_meta.append(row)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=1))
        z.writestr('images.json', json.dumps(images_meta, ensure_ascii=False, indent=1))
        for n in ref_names:
            p = os.path.join(dsdir, n)
            z.write(p, f'ref/{n}')
        for img in rows:
            name = _backup_basename(img.filename)
            if not name:
                continue   # metadata-only small-rescue candidate
            p = os.path.join(dsdir, name)
            if os.path.isfile(p):
                z.write(p, f'images/{name}')


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
    except (OSError, ValueError) as exc:
        raise ValueError('zip archive is not seekable') from exc
    return archive, None


def import_backup_zip(user_id: int, archive: bytes | BinaryIO):
    """Restore a backup as a NEW dataset (never merges into an existing one).
    Hardened: manifest format/version check, per-entry filename whitelist (no
    separators/traversal), file-count and uncompressed-size caps. Returns the
    created FaceDataset."""
    stream, owned = _coerce_archive_stream(archive)
    try:
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
    if len(all_infos) > _BACKUP_MAX_FILES + 2:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    if sum(info.file_size for info in all_infos) > _BACKUP_MAX_BYTES:
        raise ValueError('backup too large (max 2 GB uncompressed)')
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
    except (ValueError, UnicodeError, zipfile.BadZipFile):
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
    for field in ('name', 'trigger_word'):
        value = manifest.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'invalid backup {field}')
    if not isinstance(images_meta, list):
        raise ValueError('invalid backup image metadata')
    if len(images_meta) > _BACKUP_MAX_ROWS:
        raise ValueError(f'too many image rows in backup (max {_BACKUP_MAX_ROWS})')
    seen_backup_ids = set()
    rescue_sources = set()
    rescue_parent_counts = {}
    for meta in images_meta:
        if not isinstance(meta, dict):
            raise ValueError('invalid backup image metadata')
        filename = meta.get('filename')
        if filename is not None and not isinstance(filename, str):
            raise ValueError('invalid backup image filename')
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
    infos = [i for i in all_infos
             if not i.is_dir() and i.filename.startswith(('ref/', 'images/'))]
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
            with z.open(info) as src, open(os.path.join(staging_dir, base), 'wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            if prefix == 'ref':
                extracted_refs.setdefault(base.casefold(), base)
            else:
                extracted_images.add(base)

        db_started = True
        ds = create_dataset(user_id, name, trigger, kind=manifest.get('kind'),
                            concept_desc=manifest.get('concept_desc'),
                            train_type=manifest.get('train_type'), commit=False)
        for field in ('concept_terms', 'train_variant', 'train_settings',
                      'best_settings', 'fidelity'):
            setattr(ds, field, manifest.get(field))
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
            # while valid Pexels metadata is canonicalized back to JSON TEXT.
            values['source_metadata'] = _source_metadata_storage(
                values.get('source_metadata'))
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


def replace_in_captions(user_id, dataset_id, find, replace, mode='text'):
    """Bulk-edit the captions of KEPT images (the ones that train). Two modes:

    - 'text': plain substring replace, case-sensitive.
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
            new = old.replace(find, replace or '')
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
            img.caption = new
            changed += 1
    if changed:
        db.session.commit()
    return changed


# Batch curation (multi-select in the grid). 'pending' = reset the triage state.
BATCH_ACTIONS = ('keep', 'reject', 'pending', 'delete', 'clear_caption')


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
    ids = [int(i) for i in (image_ids or []) if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
    if not ids:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id)
            .filter(FaceDatasetImage.id.in_(ids)).all())
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
            img.caption = None
        else:
            # Never resurrect a failed generation into keep/reject — the tile has
            # no file; regenerate is the only way out of 'failed'.
            if img.status == 'failed':
                continue
            if action == 'reject':
                _clear_watermark_metadata(img)
            img.status = action
        n += 1
    db.session.commit()
    return n


def _ref_crop_source_path(ds) -> str:
    """The image a manual/auto re-crop reads from: the full-frame ORIGINAL when we
    kept one, else the cropped ref (legacy datasets uploaded before we stored the
    original — they can still be re-cropped, only not wider than the existing crop)."""
    name = ds.ref_original_filename or ds.ref_filename
    return os.path.join(_dataset_dir(ds.id), name)


def crop_reference(user_id, dataset_id, x, y, w, h):
    """Manually crop the dataset reference to (x,y,w,h), resized to 1024. The box is
    in the ORIGINAL's pixel space (the editor shows the original), and we write the
    derived square to ref_filename WITHOUT touching the original — so the user can
    re-crop wider or tighter any number of times."""
    ds = get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        return False
    ok, _scale = _crop_resize_file(_ref_crop_source_path(ds), x, y, w, h, dst=_ref_path(ds))
    return ok


def recrop_reference_auto(user_id, dataset_id):
    """Re-run the automatic head-crop on the ORIGINAL, overwriting ref_filename.
    Returns (ok, head_detected). CALLER holds the GPU vision window. Lets the user
    reset to the auto framing after manual edits, without re-uploading the photo."""
    ds = get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        return False, False
    try:
        with open(_ref_crop_source_path(ds), 'rb') as fh:
            raw = fh.read()
    except OSError:
        return False, False
    webp, detected = face_crop_to_square_webp(raw, pad=REF_CROP_PAD, return_detected=True)
    with open(_ref_path(ds), 'wb') as fh:
        fh.write(webp)
    return True, detected


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
            W, H = im.size
    except (OSError, ValueError):
        return none
    box = tuple(bbox)
    route, _ = _route_watermark(box, W, H)
    # Only recompute the crop-disabled route when crop is what the default picked --
    # otherwise the two are identical, so skip the redundant pure-function call.
    route_nc = route if route != 'crop' else _route_watermark(box, W, H, allow_crop=False)[0]
    return {'watermark_route': route, 'watermark_route_nocrop': route_nc}


def dataset_payload(user_id, dataset_id):
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return None
    imgs = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .order_by(FaceDatasetImage.id.desc()).all())
    comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    # Combien, PAR bucket, sont des crops fortement agrandis (upscale_ratio >=
    # UPSCALE_WARN_THRESHOLD) plutôt que du natif : le compte `comp` seul traite un
    # gros plan natif et un gros plan upscalé x3 comme équivalents vis-à-vis de la
    # cible — ce sous-compte permet à l'UI de signaler un dataset qui « remplit »
    # sa cible face/bust surtout avec de la texture fabriquée par le resize.
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
        # Dual long+short captioning toggle (Advanced options) → the caption editor shows
        # the short field only when this is on.
        'dual_captions': dual_captions_enabled(ds),
        'fidelity': (ds.fidelity or 'face') if not concept else 'face',
        'concept_desc': (ds.concept_desc or '') if concept else '',
        # Creative-direction suffixes (global + per-framing) → settings modal
        # prefill. Applied at wrap time; never part of the stored per-image prompt.
        'prompt_suffix': ds.prompt_suffix or '',
        'prompt_suffixes': prompt_suffixes_dict(ds),
        'ref_filename': ds.ref_filename,
        'ref_original_filename': ds.ref_original_filename or '',
        'ref_extra_filenames': extra_ref_filenames(ds), 'composition': comp,
        'composition_upscaled': comp_upscaled,
        # Réglages gagnants du Studio (JSON → objet). Manquait du payload : le badge
        # ★ du workspace ne s'affichait jamais, et le garde-fou « suppression d'un
        # checkpoint référencé » en a besoin.
        'best_settings': _safe_json(ds.best_settings),
        'face_thresholds': {'green': cfg.get('face_scoring.green'), 'orange': cfg.get('face_scoring.orange')},
        'images': [{'id': i.id, 'filename': i.filename, 'source': i.source,
                    'framing': i.framing, 'variation_label': i.variation_label,
                    'status': i.status, 'caption': i.caption,
                    'caption_short': i.caption_short,
                    'fail_reason': i.fail_reason,
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
    }


# --- Image normalization ---------------------------------------------------
def normalize_to_webp(image_bytes: bytes, size: int = 1024) -> bytes:
    """Resize so the longest side ≤ `size`, KEEP the aspect ratio (no square pad),
    return WEBP. Pour les variations Nano Banana : un plan corps reste en portrait
    (pas de bandes noires que le LoRA apprendrait). ai-toolkit gère le bucketing."""
    im = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    im.thumbnail((size, size), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, 'WEBP', quality=92)
    return out.getvalue()


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
    raw = describe_image_ollama(image_bytes, HEAD_BBOX_PROMPT, num_predict=400,
                                prefer_json=True, fmt='json')
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
    square, no GPU window needed) — the manual-first reference flow."""
    im = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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
def import_images(user_id, dataset_id, files_bytes, crop=False, dedupe=False, stats=None,
                  source_metadata=None, captions=None):
    """Normalize (or head-crop) + persist + create import rows (status=keep).
    When crop=True, each image is auto head-cropped via Qwen3-VL - the CALLER
    must then hold the GPU-exclusive window - and is by construction a face,
    so framing='face' is set directly (no classify pass needed).

    dedupe=True (the /import route) drops perceptual duplicates by dHash — both
    within the batch and vs the dataset's existing files. The hash is computed on
    the NORMALIZED result (what's actually stored), so a re-import of the same
    photo matches its earlier crop instead of comparing a full frame to a head
    crop. Skips are counted in stats['duplicates'] when a stats dict is passed.
    Default stays False: service-level callers (scrape flow dedupes upstream on
    the ORIGINALS, before paying the crop) keep the historical behavior.

    ``source_metadata`` is an optional list parallel to ``files_bytes``. Only
    validated Pexels provenance is stored; existing callers can omit it.

    ``captions`` is an optional list parallel to ``files_bytes`` — a pre-existing
    caption to carry onto the new row (the image-bank promotion path passes the bank
    captions here, so a promoted selection starts already captioned). Empty/None entries
    leave the row uncaptioned. A skipped duplicate simply drops its caption with it.

    Returns (ids, failed_count)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return [], 0
    # Sans head-crop, on préserve TOUJOURS le ratio (normalize_to_webp) : l'ancien
    # chemin « carré padé » ajoutait des bandes noires que le LoRA apprendrait, et
    # forçait tous les imports personnage en carré — un plan buste/corps importé
    # doit rester tel quel (ai-toolkit gère le bucketing multi-ratios).
    seen = _existing_dhashes(dataset_id) if dedupe else None
    metadata_by_index = list(source_metadata) if source_metadata is not None else []
    captions_by_index = list(captions) if captions is not None else []
    ids = []
    failed = 0
    for index, raw in enumerate(files_bytes):
        # Garde-fou qualité : ai-toolkit ne fait que RÉDUIRE — une image sous
        # 768 px de petit côté reste floue à l'entraînement. Comptée (toast),
        # jamais bloquée : c'est parfois la seule photo disponible.
        if stats is not None:
            try:
                with Image.open(io.BytesIO(raw)) as im0:
                    if min(im0.size) < SCRAPE_IMPORT_MIN_SIDE:
                        stats['small'] = stats.get('small', 0) + 1
            except Exception:
                pass
        try:
            if crop:
                webp, scale = face_crop_to_square_webp(raw, return_scale=True)
            else:
                webp, scale = normalize_to_webp(raw), None
        except Exception as e:
            failed += 1
            logger.warning(f"dataset import: image skipped (dataset {dataset_id}): {e}")
            continue
        if dedupe:
            try:
                with Image.open(io.BytesIO(webp)) as im:
                    fp = _dhash(im)
            except (OSError, ValueError):
                fp = None   # unreadable output would have failed above; belt & braces
            if fp is not None:
                if any(_hamming(fp, s) <= SCRAPE_DHASH_MAX_DISTANCE for s in seen):
                    if stats is not None:
                        stats['duplicates'] = stats.get('duplicates', 0) + 1
                    logger.info(f"dataset import: perceptual duplicate skipped (dataset {dataset_id})")
                    continue
                seen.append(fp)
        fn = f"{user_id}_dataset_{uuid.uuid4().hex[:8]}.webp"
        with open(os.path.join(_dataset_dir(dataset_id), fn), 'wb') as fh:
            fh.write(webp)
        cap = (captions_by_index[index] if index < len(captions_by_index) else None)
        cap = _cap_caption(cap) if (cap or '').strip() else None
        img = FaceDatasetImage(dataset_id=dataset_id, source='import', status='keep',
                               filename=fn, framing='face' if crop else None,
                               upscale_ratio=scale, caption=cap,
                               source_metadata=_source_metadata_storage(
                                   metadata_by_index[index]
                                   if index < len(metadata_by_index) else None))
        db.session.add(img)
        db.session.commit()
        ids.append(img.id)
    return ids, failed


# --- Import d'un dataset d'entraînement existant (ZIP kohya-style / dossier) --
# Des images + sidecars .txt de même nom (la convention kohya/ai-toolkit), soit
# dans un ZIP uploadé, soit dans un dossier du disque du serveur (app locale
# mono-user : le chemin est SON disque). Les images gardent leur ratio
# (normalize_to_webp, pas de crop), les captions atterrissent sur les rows,
# dédup perceptuelle vs le lot ET le dataset. Les fichiers sont réécrits sous
# des noms générés (jamais celui de la source → aucune traversée possible),
# profondeur de dossiers libre (le ZIP accepte toute arborescence ; le dossier
# est parcouru récursivement pour rester aligné).
DATASET_ZIP_MAX_FILES = 400
DATASET_ZIP_MAX_BYTES = 2 * 1024 * 1024 * 1024
DATASET_ZIP_MAX_IMAGE_BYTES = 128 * 1024 * 1024
_DATASET_ZIP_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def _merge_training_images(user_id, dataset_id, entries, captions, stats=None):
    """Cœur commun ZIP/dossier : `entries` = liste de (stem, display_name, getter)
    où `getter()` rend les bytes de l'image, `captions` = {stem: texte}. Chaque
    image lisible devient une row 'import' (status=keep, ratio préservé), la
    caption de même stem est attachée (tronquée à CAPTION_MAX_CHARS), les
    doublons perceptuels (dHash) vs le lot ET le dataset sont sautés.
    Returns (ids, failed)."""
    seen = _existing_dhashes(dataset_id)
    ids, failed = [], 0
    for stem, display, getter in entries:
        try:
            raw = getter()
        except (OSError, zipfile.BadZipFile):
            failed += 1
            continue
        if stats is not None:   # même garde qualité que l'import de photos
            try:
                with Image.open(io.BytesIO(raw)) as im0:
                    if min(im0.size) < SCRAPE_IMPORT_MIN_SIDE:
                        stats['small'] = stats.get('small', 0) + 1
            except Exception:
                pass
        try:
            webp = normalize_to_webp(raw)
        except Exception as e:
            failed += 1
            logger.warning(f"dataset import: image skipped ({display}): {e}")
            continue
        try:
            with Image.open(io.BytesIO(webp)) as im:
                fp = _dhash(im)
        except (OSError, ValueError):
            fp = None
        if fp is not None:
            if any(_hamming(fp, s) <= SCRAPE_DHASH_MAX_DISTANCE for s in seen):
                if stats is not None:
                    stats['duplicates'] = stats.get('duplicates', 0) + 1
                continue
            seen.append(fp)
        fn = f"{user_id}_dsimport_{uuid.uuid4().hex[:8]}.webp"
        with open(os.path.join(_dataset_dir(dataset_id), fn), 'wb') as fh:
            fh.write(webp)
        cap = (captions.get(stem) or '').strip() or None
        if cap:
            cap = _cap_caption(cap)
            if stats is not None:
                stats['captions'] = stats.get('captions', 0) + 1
        img = FaceDatasetImage(dataset_id=dataset_id, source='import', status='keep',
                               filename=fn, caption=cap)
        db.session.add(img)
        db.session.commit()
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
        try:
            z = zipfile.ZipFile(stream)
        except zipfile.BadZipFile as exc:
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
                    except (OSError, zipfile.BadZipFile):
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
    sizes = {}
    for p in paths:
        try:
            sizes[p] = os.path.getsize(p)
        except OSError:
            sizes[p] = 0
    if sum(sizes.values()) > DATASET_ZIP_MAX_BYTES:
        raise ValueError('folder too large (max 2 GB)')
    captions = {}
    for p in paths:
        if p.lower().endswith('.txt') and sizes.get(p, 0) <= 64 * 1024:
            try:
                with open(p, 'rb') as fh:
                    captions[os.path.splitext(p)[0]] = \
                        fh.read().decode('utf-8', 'replace').strip()
            except OSError:
                pass

    def _read(p):
        with open(p, 'rb') as fh:
            return fh.read()

    entries = [(os.path.splitext(p)[0], p, lambda p=p: _read(p))
               for p in paths if p.lower().endswith(_DATASET_ZIP_IMG_EXTS)]
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
_SCRAPE_DL_TYPES = ('image/jpeg', 'image/jpg', 'image/png', 'image/webp')  # pas de gif/svg
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


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def _existing_dhashes(dataset_id) -> list:
    """dHashes des images déjà dans le dataset (keep/pending), recalculés à la
    volée : resize 9×8 ≈ qq ms/image et un dataset plafonne à ~200 images —
    pas de colonne/migration pour si peu."""
    out = []
    rows = FaceDatasetImage.query.filter(
        FaceDatasetImage.dataset_id == dataset_id,
        FaceDatasetImage.status.in_(('keep', 'pending'))).all()
    for r in rows:
        if not r.filename:
            continue
        try:
            with Image.open(os.path.join(_dataset_dir(dataset_id), r.filename)) as im:
                out.append(_dhash(im))
        except (OSError, ValueError):
            continue
    return out


def _accept_scrape_bytes(raw, seen_hashes, skipped, rescue_small=False):
    """Filtre une image téléchargée : résolution / ratio / dedup perceptuel.
    Retourne les bytes si acceptée (et enregistre son dHash dans seen_hashes),
    sinon None en incrémentant le compteur skipped adéquat. Quand rescue_small
    est vrai, une petite image continue vers ratio+dedup au lieu d'être rejetée;
    elle ne sera jamais importée directement dans l'entraînement."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            w, h = im.size
            if min(w, h) < SCRAPE_IMPORT_MIN_SIDE and not rescue_small:
                skipped['low_res'] += 1
                return None
            if max(w, h) > SCRAPE_IMPORT_MAX_RATIO * min(w, h):
                skipped['extreme_ratio'] += 1
                return None
            fp = _dhash(im)
    except (OSError, ValueError):
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
        with Image.open(io.BytesIO(raw)) as im:
            return (min(im.size), im.width * im.height)
    except (OSError, ValueError):
        return (0, 0)


def _save_small_scrape_pair(user_id, dataset_id, raw, prompt, source_metadata=None):
    """Persist the untouched scrape source and enqueue one Klein candidate.

    Returns True when queued, False when enqueue failed. The original and result
    rows are committed before enqueue so a failed queue operation never loses the
    source file or leaves an untracked job.
    """
    from .klein_edit_helper import enqueue_klein_edit

    with Image.open(io.BytesIO(raw)) as im:
        ext = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(im.format)
    if not ext:
        raise ValueError('unsupported scrape image format')
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
            edit_prompt=prompt,
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
                    with Image.open(io.BytesIO(ok_bytes)) as im:
                        is_small = min(im.size) < SCRAPE_IMPORT_MIN_SIDE
                except (OSError, ValueError):
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
    try:
        from .vision_ollama import describe_image_ollama, unload_vision_model
    except ImportError:
        raise RuntimeError('vision (Ollama) service not configured/available yet')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = FaceDatasetImage.query.filter_by(
        dataset_id=dataset_id, source='import', framing=None).all()
    n = 0
    # Persistent progress indicator (survives a page reload): try/finally guarantees
    # end() runs even if the batch raises → no phantom "Classifying…" spinner.
    token = dataset_activity.begin(dataset_id, 'classify', total=len(rows))
    try:
        for i, img in enumerate(rows):
            dataset_activity.progress(token, done=i + 1)
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
                     ollama_model=None, extra_instructions=''):
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
    todo = [(img, _img_path(img)) for img in q.all() if img.filename]
    todo = [(img, p) for img, p in todo if p and os.path.exists(p)]
    if not todo:
        return 0
    # Total for the persistent progress indicator (token owned by the caller).
    dataset_activity.progress(token, total=len(todo),
                              detail=f'Preparing {len(todo)} concept caption(s)…')
    n = 0
    remaining = list(todo)
    refine_targets = []  # (img, p, joycap) -> Joy draft refined by Qwen
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
        for img, p in remaining:
            cap = (jc.get(p) or '').strip().strip('"').strip()
            if cap:
                refine_targets.append((img, p, cap))
            else:
                still.append((img, p))
        remaining = still
    # 2a) Backend 'joycaption' forced: no Qwen. Store Joy drafts scrubbed mechanically
    #     (leak_re from the desc words only) - respects "no Ollama fallback".
    if backend == 'joycaption':
        leak_re = _concept_terms_re(_fallback_concept_terms(concept_desc))
        for img, p, joycap in refine_targets:
            if dataset_activity.cancel_requested(ds.id):
                break   # graceful stop at an image boundary (see caption_images)
            dataset_activity.bump(token)
            try:
                with open(p, 'rb') as fh:
                    data = fh.read()
            except OSError:
                data = b''
            final = _enforce_concept_omission(joycap, leak_re, data, concept_desc) or joycap
            img.caption = _cap_caption(final)
            db.session.commit()
            n += 1
        return n
    # 2b) Qwen passes ('auto'/'ollama'): refine Joy drafts, direct-caption the rest, all
    #     enforced. One model load -> unload once at the end.
    if refine_targets or remaining:
        try:
            from .vision_ollama import describe_image_ollama, unload_vision_model
        except ImportError:
            raise RuntimeError('vision (Ollama) service not configured/available yet')
        # Ban-list (LLM expansion cached + desc words) -> leak regex, compiled ONCE per
        # batch, AFTER the Joy subprocess finished (never two models in VRAM at once).
        sample = refine_targets[0][1] if refine_targets else remaining[0][1]
        leak_re = _concept_terms_re(_get_concept_terms(ds, image_path=sample,
                                                       describe=describe_image_ollama))
        try:
            for img, p, joycap in refine_targets:
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
                    refined = describe_image_ollama(
                        data, refine_prompt,
                        num_predict=5000, model=ollama_model,
                        keep_alive=_VISION_BATCH_KEEPALIVE,
                        timeout=(10, 300))
                except Exception as e:  # noqa: BLE001 - refine best-effort
                    logger.warning('caption concept: Qwen refine failed (%s)', e)
                refined = (refined or '').strip().strip('"').strip()
                if _refine_output_ok(refined, joycap):
                    final = refined
                else:
                    # Unusable refine (reasoning trace / loop) -> direct Qwen caption
                    # (natively omits the concept), else keep the Joy draft.
                    logger.info('caption concept: refine rejected -> direct Qwen (image %s)', img.id)
                    alt = ''
                    try:
                        alt = describe_image_ollama(data, cap_prompt, num_predict=2000,
                                                    model=ollama_model,
                                                    keep_alive=_VISION_BATCH_KEEPALIVE,
                                                    timeout=(10, 300))
                    except Exception:  # noqa: BLE001
                        alt = ''
                    alt = (alt or '').strip().strip('"').strip()
                    final = alt or joycap
                final = _enforce_concept_omission(final, leak_re, data, concept_desc,
                                                  describe=describe_image_ollama) or final
                if not _usable_caption(final):
                    # Refine AND direct both unusable → fall back to the Joy draft (clean
                    # prose), scrubbed of any leak; leave blank if even that fails.
                    final = _enforce_concept_omission(joycap, leak_re, data, concept_desc,
                                                      describe=describe_image_ollama) or joycap
                    if not _usable_caption(final):
                        # force=re-do-all: overwrite any stale pre-fix caption with blank
                        # (trigger-only is valid for a concept LoRA) rather than retain it.
                        if force and (img.caption or ''):
                            img.caption = ''
                            db.session.commit()
                        logger.info('caption concept: no usable caption for image %s -> left blank', img.id)
                        continue
                img.caption = _cap_caption(final)
                db.session.commit()
                n += 1
            for img, p in remaining:
                if dataset_activity.cancel_requested(ds.id):
                    break   # graceful stop at an image boundary (see caption_images)
                dataset_activity.bump(token)
                with open(p, 'rb') as fh:
                    data = fh.read()
                cap = describe_image_ollama(
                    data, cap_prompt, num_predict=2000, model=ollama_model,
                    keep_alive=_VISION_BATCH_KEEPALIVE,
                    auto_start_local=True, timeout=(10, 300))
                cap = (cap or '').strip().strip('"').strip()
                if cap:
                    cap = _enforce_concept_omission(cap, leak_re, data, concept_desc,
                                                    describe=describe_image_ollama) or cap
                if _usable_caption(cap):
                    img.caption = _cap_caption(cap)
                    db.session.commit()
                    n += 1
                else:
                    if force and (img.caption or ''):
                        img.caption = ''
                        db.session.commit()
                    logger.info('caption concept: no usable direct caption for image %s -> left blank', img.id)
        finally:
            unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
    return n


def caption_images(user_id, dataset_id, force=False, mode=None, image_ids=None):
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
                        fallback Ollama pour les images qu'il n'a pas captées."""
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
                                 extra_instructions=extra_instructions)
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
    # Défaut AUTO selon le type entraîné ; un mode explicite (UI) l'emporte.
    ttype = (getattr(ds, 'train_type', None) or 'zimage').lower()
    mode = (mode or ('booru' if ttype == 'sdxl' else 'prose')).lower()
    style = is_style(ds)
    if style:
        # Dataset STYLE : captions de CONTENU pur — le rendu n'est jamais décrit (le
        # prompt porte la règle) pour qu'il soit absorbé par le LoRA. AUCUN nettoyage
        # d'identité : les sujets varient, leur description EST le contenu contrôlable.
        cap_prompt = caption_prompt_for_style(mode)
        def cleaner(text):
            return text
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
    todo = [(img, _img_path(img)) for img in rows if img.filename]
    todo = [(img, p) for img, p in todo if p and os.path.exists(p)]
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
        remaining = todo
        # In 'auto', why JoyCaption didn't contribute (deps missing / crash). Kept so a
        # LATER Ollama failure reports BOTH reasons instead of only the Ollama one —
        # otherwise a user whose JoyCaption is silently unavailable debugs blind (issue #6).
        joycaption_note = ''
        # 1) JoyCaption en BATCH (un seul chargement du 8B NF4, via le venv ai-toolkit) -
        # sauté entièrement quand le backend force 'ollama'.
        if backend in ('auto', 'joycaption'):
            jc = {}
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
                        should_cancel=lambda: dataset_activity.cancel_requested(dataset_id))
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
            for img, p in remaining:
                cap = (jc.get(p) or '').strip().strip('"').strip()
                if cap:
                    cleaned = cleaner(cap) or cap
                    img.caption = _cap_caption(cleaned)
                    db.session.commit()
                    n += 1
                    dataset_activity.bump(token)   # this image is captioned (done)
                else:
                    still.append((img, p))
            remaining = still
            dataset_activity.progress(
                token, detail=f'JoyCaption finished; {len(remaining)} image(s) remaining…')
            if backend == 'joycaption':  # backend forcé JoyCaption -> pas de repli Ollama
                logger.info('captioning finished: dataset=%s backend=%s captioned=%s elapsed=%.1fs',
                            dataset_id, backend, n, time.monotonic() - started)
                return n
        # 2) Ollama (Qwen3-VL) pour les images non couvertes par JoyCaption ('auto'),
        # ou pour TOUT le lot si le backend force 'ollama'.
        if remaining:
            try:
                from .vision_ollama import describe_image_ollama, unload_vision_model
            except ImportError:
                raise RuntimeError('vision (Ollama) service not configured/available yet')
            try:
                for index, (img, p) in enumerate(remaining, 1):
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
                        cleaned = cleaner(cap) or cap
                        img.caption = _cap_caption(cleaned)
                        db.session.commit()
                        n += 1
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
        logger.info('captioning finished: dataset=%s backend=%s captioned=%s elapsed=%.1fs',
                    dataset_id, backend, n, time.monotonic() - started)
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
                      same as the dataset pass).
    on_caption(path, caption) : fired as each caption lands, for incremental persistence.
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

    def _emit(p, cap):
        nonlocal done
        out[p] = cap
        if on_caption:
            on_caption(p, cap)
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
                _emit(p, _cap_caption(cap))
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
        try:
            for index, p in enumerate(remaining, 1):
                if should_cancel and should_cancel():
                    break  # graceful stop at an image boundary (see caption_images)
                with open(p, 'rb') as fh:
                    cap = describe_image_ollama(
                        fh.read(), cap_prompt, num_predict=2000, model=ollama_model,
                        keep_alive=_VISION_BATCH_KEEPALIVE,
                        auto_start_local=(index == 1), timeout=(10, 300))
                cap = (cap or '').strip().strip('"').strip()
                if cap:
                    _emit(p, _cap_caption(cap))
                else:
                    done += 1  # handled-but-empty still advances the bar
                    if progress:
                        progress(done, total)
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

def _compose_preview_instructions(vocabulary, instructions) -> str | None:
    """Combine a vocabulary preset (the SAME appended register the dataset pass uses,
    from _VOCABULARY_INSTRUCTION) with the user's free extra instructions into the single
    ``extra_instructions`` string caption_paths appends to the prompt. None when neither
    is set (byte-identical to a plain descriptive pass)."""
    parts = []
    if vocabulary:
        parts.append(_VOCABULARY_INSTRUCTION[vocabulary])
    extra = (instructions or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    if extra:
        parts.append(extra)
    return '\n'.join(parts) if parts else None


# Public so the image bank's caption lane validates against — and appends — the SAME
# vocabulary registers as the dataset pass, rather than duplicating the tuple or the text.
CAPTION_VOCABULARIES = _CAPTION_VOCABULARIES


def vocabulary_instruction(vocabulary) -> str | None:
    """The caption instruction appended for a vocabulary register (one of
    CAPTION_VOCABULARIES: 'explicit' | 'clinical' | 'safe'), or None for '' / an unknown
    value. Shared with the image bank so its NSFW lane reuses the dataset's exact register
    text — 'explicit' only spells acts out when paired with an abliterated vision model,
    and the output cleaners still run, so it changes wording, never what binds."""
    return _VOCABULARY_INSTRUCTION.get((vocabulary or '').strip().lower())


def preview_caption(user_id, dataset_id, image_id, *, backend=None, ollama_model=None,
                    vocabulary=None, instructions=None, should_cancel=None) -> dict:
    """Caption ONE dataset image with a candidate config and return the text WITHOUT
    persisting it — the Caption Lab's ephemeral A/B probe. Reuses caption_paths(), so the
    engine/model/GPU serialization contract is identical to the batch pass.

    backend      : '' / None → global default; else one of _CAPTION_BACKENDS ('none' is
                   rejected here — a preview with captioning disabled makes no sense).
    vocabulary   : '' / None → the model's own wording; else an _CAPTION_VOCABULARIES
                   preset, appended as an instruction exactly like the dataset options.
    instructions : free extra instructions, appended after the vocabulary preset.
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
    extra = _compose_preview_instructions(vocab, instructions)
    ollama_model = (ollama_model or '').strip() or None
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
        return style_content_caption(ds, t)
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
    try:
        for img in rows:
            if dataset_activity.cancel_requested(dataset_id):
                break   # graceful stop at an image boundary (see caption_images)
            dataset_activity.bump(token)
            short = _scrub_short_like_long(ds, gen(_shorten_prompt(ds, img.caption)), mode)
            if not short:
                continue
            img.caption_short = _cap_caption(short) or None
            db.session.commit()
            n += 1
    finally:
        _unload()
        if own_token is not None:
            dataset_activity.end(own_token)
    return n


# --- Face similarity scoring (InsightFace antelopev2, CPU subprocess) -------
def analyze_faces(user_id, dataset_id) -> dict:
    """Score les images GARDEES vs la reference (InsightFace antelopev2, CPU subprocess).
    Persiste face_score (cosinus brut, None si non note) + face_state. Lot A : AUCUNE
    suppression. Tourne sur CPU -> pas de fenetre GPU. Retourne {state: count}."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference photo missing')
    ref_path = _ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference photo missing')
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None)).all())
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
    # Persistent indicator (survives reload). The scoring is a single CPU subprocess
    # (opaque — done stays 0 during it, then fills as results are committed); try/
    # finally clears the indicator even if scoring raises.
    token = dataset_activity.begin(dataset_id, 'analyze_faces', total=len(by_path))
    try:
        results, scoring_error = score_dataset_faces(ref_path, list(by_path.keys()))
        counts = {}
        for p, img in by_path.items():
            dataset_activity.bump(token)
            r = results.get(p)
            if not r:
                continue
            img.face_state = r.get('state')
            img.face_score = r.get('sim')   # None si non-scorable
            db.session.commit()
            counts[img.face_state] = counts.get(img.face_state, 0) + 1
        return counts, scoring_error
    finally:
        dataset_activity.end(token)


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


def _preserve_original(path) -> None:
    """Copy `path` to a sibling `<stem>.orig<suffix>` before a destructive edit, so the
    watermarked original stays recoverable. The app trash util (send_to_trash) MOVES a
    file -- unusable here since the cleaned image must keep serving from the SAME path
    (and LaMa overwrites it in place) -- so we keep a sibling copy instead. Only written
    ONCE (a re-clean must not clobber the true original with an already-modified one).
    These .orig files carry no DB row, so export/backup (which iterate rows) ignore them."""
    stem, ext = os.path.splitext(path)
    backup = f'{stem}.orig{ext or ".webp"}'
    if not os.path.exists(backup):
        try:
            shutil.copy2(path, backup)
        except OSError as e:
            logger.warning('watermark: could not preserve original %s: %s', path, e)


def _apply_watermark_crop(path, box) -> bool:
    """Crop `path` to `box` (left,top,right,bottom px) and re-save WEBP q92 WITHOUT
    resizing -- the whole point of the crop route is that it invents no pixel (the
    aspect-ratio change is absorbed by ai-toolkit's bucketing). Returns bool."""
    try:
        im = Image.open(path).convert('RGB')
    except (OSError, ValueError):
        return False
    box = (max(0, int(box[0])), max(0, int(box[1])),
           min(im.width, int(box[2])), min(im.height, int(box[3])))
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return False
    out = io.BytesIO()
    im.crop(box).save(out, 'WEBP', quality=92)
    with open(path, 'wb') as fh:
        fh.write(out.getvalue())
    return True


def detect_watermarks(user_id, dataset_id, *, include_dismissed=False):
    """Scan the KEPT images for an overlaid watermark via Qwen3-VL and persist
    watermark_state ('detected'|'none') + watermark_bbox (JSON normalized box).
    CALLER holds the GPU-exclusive vision window (same as classify/caption). Returns
    {'detected': n, 'none': n, 'checked': n}.

    Images the user already judged NOT a watermark ('dismissed', a false positive
    ruled out in the review lightbox) are SKIPPED so a re-run never re-flags them --
    that's the anti-frustration point. Pass include_dismissed=True to re-examine them
    (a deliberate "check everything again")."""
    try:
        from .vision_ollama import describe_image_ollama, unload_vision_model
    except ImportError:
        raise RuntimeError('vision (Ollama) service not configured/available yet')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return {'detected': 0, 'none': 0, 'checked': 0}
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None)).all())
    counts = {'detected': 0, 'none': 0, 'checked': 0}
    # Persistent progress indicator (survives a page reload); try/finally clears it
    # even if the vision pass raises → no phantom "Scanning…" spinner.
    token = dataset_activity.begin(dataset_id, 'watermark_detect', total=len(rows))
    try:
        for i, img in enumerate(rows):
            dataset_activity.progress(token, done=i + 1)
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
    return counts


def dismiss_watermarks(user_id, dataset_id, image_ids):
    """Mark 'detected' images as 'dismissed' -- the user ruled, in the review lightbox,
    that the flag is a FALSE positive. Dismissed images drop the 🚩 badge, leave the
    Clean batch, and are skipped by future detect passes (see detect_watermarks) so
    they're never re-flagged. Only 'detected' rows of THIS dataset transition (ids that
    don't belong / aren't detected are silently ignored, like batch_image_action).
    Returns the number of rows dismissed. The bbox is kept (harmless, and a later
    include_dismissed re-scan overwrites it)."""
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
    out = {'cropped': 0, 'inpainted': 0, 'inpainted_klein': 0, 'needs_review': 0,
           'failed': 0, 'skipped': 0}
    error = None
    lama_ok = watermark_lama.is_available()
    klein_ok = method == 'klein' and watermark_klein.is_available()
    lama_pending = []  # (img, path, bboxes, manual_regions)

    def _run_klein(img, path, boxes, manual):
        """One serialized Klein inpaint of `img` in place. Preserves the .orig first;
        on success flips to 'cleaned' (+ clears manual regions like the LaMa path)."""
        nonlocal error
        if not klein_ok:
            out['skipped'] += 1               # leave 'detected' (Klein not ready)
            return
        _preserve_original(path)
        ok, err = watermark_klein.inpaint_watermark_klein(user_id, path, boxes)
        if ok:
            img.watermark_state = 'cleaned'
            if manual:
                img.watermark_regions = None
            out['inpainted_klein'] += 1
        elif err and err.get('kind') == 'unavailable':
            out['skipped'] += 1
        else:
            if not manual:                    # keep manual retry metadata (like LaMa)
                img.watermark_state = 'failed'
            out['failed'] += 1
            if err:
                error = err
    # Persistent progress indicator (survives a page reload). The device is included
    # so the UI can honestly state whether ComfyUI is paused for the GPU pass.
    device_label = 'GPU' if device == 'cuda' else 'CPU'
    token = dataset_activity.begin(
        dataset_id, 'watermark_clean', total=len(rows),
        detail=f'Cleaning watermarks on {device_label}…')
    try:
        for i, img in enumerate(rows):
            dataset_activity.progress(token, done=i + 1)
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
                _preserve_original(path)
                lama_pending.append((img, path, regions, True))
                continue
            bbox = _safe_json(img.watermark_bbox)
            if not os.path.exists(path) or not (isinstance(bbox, list) and len(bbox) == 4):
                img.watermark_state = 'failed'
                out['failed'] += 1
                db.session.commit()
                continue
            try:
                with Image.open(path) as im:
                    W, H = im.size
            except (OSError, ValueError):
                img.watermark_state = 'failed'
                out['failed'] += 1
                db.session.commit()
                continue
            route, box = _route_watermark(tuple(bbox), W, H, allow_crop=allow_crop)
            if route == 'crop':
                _preserve_original(path)
                if _apply_watermark_crop(path, box):
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
                        _preserve_original(path)
                        lama_pending.append((img, path, [bbox], False))
                else:  # 'review' -> stays 'detected' so the badge/count keep flagging it
                    out['needs_review'] += 1
            db.session.commit()
        if lama_pending:
            if len(lama_pending) == 1:
                img, path, boxes, manual = lama_pending[0]
                if manual:
                    ok, err = watermark_lama.inpaint_watermarks(
                        path, boxes, **({'device': device} if device != 'cpu' else {}))
                else:
                    ok, err = watermark_lama.inpaint_watermark(
                        path, boxes[0], **({'device': device} if device != 'cpu' else {}))
                results = {path: (ok, err)}
            else:
                results = watermark_lama.inpaint_batch(
                    [{'image_path': path, 'bboxes': boxes}
                     for _img, path, boxes, _manual in lama_pending],
                    device=device,
                )
            for img, path, _boxes, manual in lama_pending:
                ok, err = results.get(path, (False, {'kind': 'failed', 'detail': 'missing inpaint result'}))
                if ok:
                    img.watermark_state = 'cleaned'
                    if manual:
                        img.watermark_regions = None
                    out['inpainted'] += 1
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
        return out, error
    finally:
        dataset_activity.end(token)


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
    pending = (FaceDatasetImage.query
               .filter_by(dataset_id=dataset_id, status='pending')
               .filter(FaceDatasetImage.filename.is_(None))
               .filter(FaceDatasetImage.job_id.isnot(None)).count())
    dataset_activity.sync_pending(dataset_id, 'generate', pending, engine='klein')


def generate_variations(user_id, dataset_id, variations, multiplier, klein_model,
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
    try:
        from .klein_edit_helper import enqueue_klein_edit
    except ImportError:
        raise RuntimeError('ComfyUI is not configured')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
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
                            suffix=dataset_prompt_suffix(ds, v.get('framing'))),
                        klein_model=klein_model,
                        lora_strength=lora_strength, extra_ref_paths=extra_paths,
                        generation_loras=run_loras,
                        extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                                        'variation_label': v.get('label')})
                except Exception:
                    img.status = 'failed'
                    db.session.commit()
                    raise
                img.job_id = job_id
                db.session.commit()
                ids.append(img.id)
    finally:
        _sync_generate_activity(dataset_id)
    return ids


def improve_existing_image(user_id, image_id):
    """Serialize one source's improve request, including the queue hand-off."""
    lock = _IMAGE_IMPROVE_LOCKS[hash((str(user_id), image_id))
                                % len(_IMAGE_IMPROVE_LOCKS)]
    with lock:
        return _improve_existing_image_locked(user_id, image_id)


def _improve_existing_image_locked(user_id, image_id):
    """Queue one non-destructive Klein upscale/improvement of an existing image.

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

    from . import klein_edit_helper as keh
    missing = keh.klein_missing_assets()
    missing_nodes = keh.klein_missing_nodes()
    if missing_nodes:
        raise KleinNodesMissing(missing, missing_nodes)
    if any(asset in missing for asset in keh.KLEIN_REQUIRED):
        raise keh.KleinModelsMissing(missing)

    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=img.dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + 1 > MAX_FANOUT:
        raise ValueError(
            f'too many generations in flight ({in_flight}), wait or cancel')

    # Profile reproduced from the user-provided ComfyUI PNG metadata.
    # Keep the selected/default Klein model; override only prompt/sampling/LoRA.
    # The improvement instruction is editable (Settings ▸ identity_prompts.klein_improve)
    # and can be turned OFF entirely — disabled applies NO prompt (pure upscale).
    if cfg.get('identity_prompts.klein_improve_enabled', True):
        prompt = get_identity_prompt('klein_improve')
    else:
        prompt = ''
    stored_prompt = prompt[:500]
    base_label = 'Klein upscale & improve'
    source_label = (img.variation_label or '').strip()
    label = (f'{base_label} · {source_label}' if source_label else base_label)[:120]
    candidate = FaceDatasetImage(
        dataset_id=img.dataset_id, source='generated', status='pending',
        parent_image_id=img.id, derivation_kind=KLEIN_IMAGE_IMPROVE,
        framing=img.framing, caption=img.caption,
        variation_label=label, variation_prompt=stored_prompt,
        # The generated candidate remains derived from the credited source.
        # Revalidate before copying so a malformed legacy row cannot surface.
        source_metadata=_source_metadata_storage(img.source_metadata),
    )
    db.session.add(candidate)
    db.session.commit()

    try:
        job_id = keh.enqueue_klein_edit(
            user_id=str(user_id), source_filename=img.filename,
            source_path=source_path, edit_prompt=prompt,
            # The "Upscale & improve" quality profile, now user-tunable. Defaults
            # (0 / 4 / 0) are the values that were hardcoded here, so an untouched
            # install behaves exactly as before. Clamped, because a bad config
            # value must degrade the pass, never crash the enqueue.
            # The fallback MUST equal the shipped config default: _improve_float treats
            # "still at its default" as "the user has not set this", which is what lets
            # a value saved under the old key name speak for it.
            lora_strength=_improve_float('improve_consistency_strength', 1.0, 1.5),
            sampler_steps=_improve_int('improve_steps', 4),
            base_lora_strength=_improve_float('improve_base_lora_strength', 0.0),
            output_megapixels=_improve_float('improve_megapixels', 2.0, 8.0),
            extra_metadata={
                'is_dataset': True,
                'dataset_id': img.dataset_id,
                'variation_label': label,
                'derivation_kind': KLEIN_IMAGE_IMPROVE,
                'parent_image_id': img.id,
                'source_image_id': img.id,
                'action': 'upscale_improve',
            },
        )
    except Exception:
        # No broken tile: the original is still untouched and the user can retry
        # as soon as the queue/ComfyUI issue is fixed.
        db.session.delete(candidate)
        db.session.commit()
        raise

    candidate.job_id = job_id
    db.session.commit()
    _sync_generate_activity(img.dataset_id)
    return {'candidate_id': candidate.id, 'job_id': job_id}


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

    `engine` (optional, 'nanobanana'/'chatgpt'/'klein') is the generator
    CURRENTLY selected in the workspace — it wins over the engine that
    originally produced the row, so a tile born on Klein doesn't pin every
    regenerate to Klein after the user switched to Nano Banana (and vice
    versa). None = legacy behaviour (reuse the row's origin). Exception:
    an NSFW-labelled tile always stays on the local Klein path (fail-closed —
    NSFW never goes to third-party APIs, mirroring the batch generate rule).
    `klein_model` (optional) is the workspace's Klein model pick, used when a
    row born on an API engine switches to Klein (its klein_model column holds
    an engine TAG, not a real model file).
    `generation_lora_preset` (optional): NAME of the generation-LoRA preset
    picked in the workspace (Idea by @waltm), Klein path only — resolved from
    the CONFIG only (fail-closed; unknown name degrades to no extra LoRAs)."""
    img = _owned_image(user_id, image_id)
    if not img or img.source != 'generated':
        return None
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
    if requested is not None and requested != 'klein' and requested not in API_ENGINES:
        raise ValueError(f'unknown engine: {requested}')
    target = requested or (img.klein_model if img.klein_model in API_ENGINES else 'klein')
    if is_nsfw_label(img.variation_label):
        target = 'klein'              # fail-closed: NSFW never reaches an API engine
    else:
        # Engines disabled in Settings must not be used even when the row (or a
        # stale workspace selection) points at them: fall back to the default
        # engine, then to the first enabled one. An empty list means "all
        # enabled" (legacy configs); NSFW above already forced local Klein.
        enabled = [e for e in (cfg.get('engines.enabled') or [])
                   if e == 'klein' or e in API_ENGINES]
        if enabled and target not in enabled:
            default = cfg.get('engines.default')
            target = default if default in enabled else enabled[0]
    # Complete every fallible target-specific preflight before changing either
    # the row or its current file. Klein enqueue is itself part of preparation:
    # if the later DB transition fails, that exact new job is cancelled below.
    from ..job_queue import queue_manager
    old_state = {
        field: getattr(img, field) for field in (
            'filename', 'caption', 'status', 'fail_reason', 'job_id',
            'klein_model', 'variation_prompt', 'watermark_state',
            'watermark_bbox', 'watermark_regions')
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
    else:
        try:
            from .klein_edit_helper import enqueue_klein_edit, resolve_generation_lora_preset
        except ImportError:
            raise RuntimeError('ComfyUI is not configured')
        # Klein target: keep the row's real model file when it has one; a row born
        # on an API engine holds an engine TAG here, not a model — use the
        # workspace's Klein pick instead (None = enqueue's default model).
        model = (img.klein_model if img.klein_model not in API_ENGINES
                 else ((klein_model or '').strip() or None))
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
                suffix=dataset_prompt_suffix(ds, img.framing)),
            klein_model=model,
            lora_strength=lora_strength, extra_ref_paths=extra_paths,
            generation_loras=resolve_generation_lora_preset(generation_lora_preset),
            extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                            'variation_label': img.variation_label})

    # Persist the replacement state first. The old file remains in place until
    # this commit succeeds, eliminating rows that reference an already-moved file.
    try:
        if old_state['status'] == 'pending' and not old_state['filename'] \
                and old_state['job_id']:
            queue_manager.cancel_job(
                old_state['job_id'], str(user_id), 'image', commit=False)
        if edited:
            img.variation_prompt = stored_prompt
        _clear_watermark_metadata(img)
        img.klein_model = engine if target in API_ENGINES else model
        img.filename = None
        img.caption = None
        img.status = 'pending'
        img.job_id = new_job_id
        img.fail_reason = None
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
            for field, value in old_state.items():
                setattr(img, field, value)
            if new_job_id:
                queue_manager.cancel_job(
                    new_job_id, str(user_id), 'image', commit=False)
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
                                   suffix=dataset_prompt_suffix(ds, img.framing)),
                    **gen_kwargs)
            except SubscriptionQuotaExceeded:
                out = None
                img.status = 'failed'
                img.fail_reason = _QUOTA_MSG
                db.session.commit()
                return engine
            except SubscriptionUnavailable as e:
                out = None
                img.status = 'failed'
                img.fail_reason = f'chatgpt: {e}'
                db.session.commit()
                return engine
            if out:
                fn = f"{user_id}_{_ENGINE_FILE_TAG[engine]}_{uuid.uuid4().hex[:8]}.webp"
                with open(os.path.join(_dataset_dir(img.dataset_id), fn), 'wb') as fh:
                    fh.write(normalize_to_webp(out))
                img.filename = fn
            else:
                img.status = 'failed'
                img.fail_reason = f'{engine}: empty response (often a content-policy refusal or a transient API error - retry usually works)'
            db.session.commit()
            return engine
        except Exception as e:
            db.session.rollback()
            current = db.session.get(FaceDatasetImage, image_id)
            if current and current.filename is None:
                current.status = 'failed'
                current.fail_reason = f'{engine}: {e}'[:500]
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
API_ENGINES = ('nanobanana', 'chatgpt')
_ENGINE_FILE_TAG = {'nanobanana': 'NBFace', 'chatgpt': 'GPTFace'}

from .chatgpt_image import SubscriptionQuotaExceeded, SubscriptionUnavailable

_QUOTA_MSG = ('chatgpt: subscription image quota reached — remaining rows were '
              'stopped; rerun in API-key mode or wait for your plan quota to reset')
_LOST_MSG = ('chatgpt: subscription connection lost — remaining rows stopped; '
             'reconnect in Settings, then regenerate')


def _api_generate_fn(engine):
    if engine == 'chatgpt':
        from .chatgpt_image import generate_variation
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
    token = dataset_activity.begin(dataset_id, 'generate', total=len(items), engine=engine) \
        if dataset_id is not None else None

    def _run_one(item):
        # item = (image_id, prompt, aspect, suffix) ; aspect optionnel (rétro-compat
        # → '1:1'), suffix optionnel (direction créative du dataset, déjà composée
        # par cadrage au call-site — rétro-compat → '').
        image_id, prompt = item[0], item[1]
        aspect = item[2] if len(item) > 2 else '1:1'
        suffix = item[3] if len(item) > 3 else ''
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
                    db.session.commit()
            return
        out = None
        fail_reason = None
        gen_kwargs = {'aspect_ratio': aspect}
        if engine == 'chatgpt':
            gen_kwargs['force_lane'] = force_lane
        try:
            out = api_generate(ref_bytes,
                               wrap_variation(prompt, ref_count=n_refs, suffix=suffix),
                               **gen_kwargs)
            if not out:
                # api_generate signale certains refus/vides par un retour falsy
                # sans lever — sans raison, la tuile "failed" resterait muette.
                fail_reason = f'{engine}: empty response (often a content-policy refusal or a transient API error - retry usually works)'
        except SubscriptionQuotaExceeded as e:
            quota_exhausted.set(); stop_msg['text'] = _QUOTA_MSG
            logger.warning(f"{engine} batch: quota exhausted at row {image_id}: {e}")
            fail_reason = _QUOTA_MSG
        except SubscriptionUnavailable as e:
            quota_exhausted.set(); stop_msg['text'] = _LOST_MSG
            logger.warning(f"{engine} batch: subscription lost at row {image_id}: {e}")
            fail_reason = _LOST_MSG
        except Exception as e:
            logger.warning(f"{engine} batch: generation error for row {image_id}: {e}")
            fail_reason = f'{engine}: {str(e)[:400]}'
        with app.app_context():
            img = db.session.get(FaceDatasetImage, image_id)
            if img is None:
                return
            if out:
                ds = db.session.get(FaceDataset, img.dataset_id)
                fn = f"{ds.user_id}_{tag}_{uuid.uuid4().hex[:8]}.webp"
                try:
                    # Conserve le ratio demandé (pas de letterbox carré sur les corps).
                    with open(os.path.join(_dataset_dir(img.dataset_id), fn), 'wb') as fh:
                        fh.write(normalize_to_webp(out))
                    img.filename = fn
                except Exception as e:
                    logger.warning(f"{engine} batch: save failed for row {image_id}: {e}")
                    img.status = 'failed'
                    img.fail_reason = f'saving the image failed: {str(e)[:400]}'
            else:
                img.status = 'failed'
                img.fail_reason = fail_reason
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
    logger.info(f"{engine} batch: done ({len(items)} variation(s))")


def generate_variations_nanobanana(app, user_id, dataset_id, variations, multiplier,
                                   engine='nanobanana'):
    """API fan-out (Nano Banana or ChatGPT, per `engine`): pre-create pending
    rows (job_id stays None - that is the marker for API-generated rows), then
    fill them from a background thread. The existing polling/banner/cancel UI
    works unchanged (pending + no file = in flight). Returns the created ids."""
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
                          dataset_prompt_suffix(ds, v.get('framing'))))

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
                Image.open(ref_path).convert('RGB').save(rpng, 'PNG')
                zf.writestr(f"{folder}/{safe}_000_ref.png", rpng.getvalue())
                zf.writestr(f"{folder}/{safe}_000_ref.txt", ds.trigger_word)
            except OSError:
                pass
        for n, img in enumerate(kept, 1):
            path = _img_path(img) if img.filename else ''
            if not img.filename or not os.path.exists(path):
                continue
            png = io.BytesIO()
            Image.open(path).convert('RGB').save(png, 'PNG')
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
