"""Face-dataset orchestration: CRUD, fan-out, import, classify, caption, export.

The vision passes (classify/caption) call describe_image_ollama; the CALLER (the
route) is responsible for wrapping them in the GPU-exclusive window. The ComfyUI
output dir is resolved via `cfg.comfyui_dir('output')` so tests can monkeypatch cfg.
"""
from __future__ import annotations
import io
import json
import logging
import os
import shutil
import threading
import uuid
import zipfile

from PIL import Image

from ..extensions import db
from ..models import FaceDataset, FaceDatasetImage
from .. import config as cfg

# Garde le modèle vision chaud entre les images d'un même batch caption/classify
# (sinon Ollama le recharge — cold start ~10s — à CHAQUE image). Déchargé en fin
# de batch pour rendre la VRAM à ComfyUI. ComfyUI est déjà en pause pendant la passe.
_VISION_BATCH_KEEPALIVE = '5m'
from .face_variations import (CAPTION_PROMPT, CAPTION_PROMPT_BOORU, CLASSIFY_PROMPT, HEAD_BBOX_PROMPT,
                              JOYCAPTION_PROMPT, aspect_for_label,
                              caption_has_identity_leak, drop_identity_sentences, drop_identity_tags,
                              prompt_by_label, wrap_variation)

logger = logging.getLogger(__name__)


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


# Longueur max d'une caption stockée (colonne TEXT, pas de contrainte DB). 600 coupait
# les captions buste/environnement en plein mot ; 800 laisse passer la phrase d'ambiance
# finale tout en restant sous la fenêtre tokenizer (~512 tokens) à l'export trigger inclus.
CAPTION_MAX_CHARS = 800


def _dataset_dir(dataset_id) -> str:
    d = str(cfg.dataset_images_root() / str(dataset_id))
    os.makedirs(d, exist_ok=True)
    return d


def _img_path(img) -> str:
    return os.path.join(_dataset_dir(img.dataset_id), img.filename)


def _ref_path(ds) -> str:
    return os.path.join(_dataset_dir(ds.id), ds.ref_filename)

_VALID_STATUS = ('pending', 'keep', 'reject', 'failed')
MAX_FANOUT = 60
# Références ADDITIONNELLES par dataset (au-delà de la principale) : servent
# UNIQUEMENT Nano Banana (multi-images d'entrée) — Klein/crop/scoring restent
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
    (ordre stable, principale d'abord — c'est elle que Gemini doit prioriser).
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
    """Retire une référence additionnelle (entrée JSON + fichier). False si inconnue."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
    extras = extra_ref_filenames(ds)
    if filename not in extras:
        return False
    try:
        os.remove(os.path.join(_dataset_dir(dataset_id), filename))
    except OSError:
        pass
    ds.ref_extra_filenames = json.dumps([f for f in extras if f != filename])
    db.session.commit()
    return True


# --- CRUD ------------------------------------------------------------------
def create_dataset(user_id, name, trigger_word):
    ds = FaceDataset(user_id=str(user_id), name=(name or '').strip()[:100],
                     trigger_word=(trigger_word or '').strip()[:60] or 'zchar')
    db.session.add(ds)
    db.session.commit()
    return ds


def get_dataset(user_id, dataset_id):
    ds = db.session.get(FaceDataset, dataset_id)
    return ds if ds and str(ds.user_id) == str(user_id) else None


def list_datasets(user_id):
    return (FaceDataset.query.filter_by(user_id=str(user_id))
            .order_by(FaceDataset.updated_at.desc()).all())


def set_image_status(user_id, image_id, status):
    if status not in _VALID_STATUS:
        raise ValueError('invalid status')
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return False
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds or str(ds.user_id) != str(user_id):
        return False
    img.status = status
    db.session.commit()
    return True


def _owned_image(user_id, image_id):
    img = db.session.get(FaceDatasetImage, image_id)
    if not img:
        return None
    ds = db.session.get(FaceDataset, img.dataset_id)
    return img if ds and str(ds.user_id) == str(user_id) else None


def set_image_caption(user_id, image_id, caption):
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    img.caption = (caption or '').strip()[:CAPTION_MAX_CHARS] or None
    db.session.commit()
    return True


def _crop_resize_file(path, x, y, w, h, size=1024):
    """Crop the file at `path` to (x,y,w,h) and RESIZE the crop to size x size
    (no black padding — the manual crop is square, aspect=1). Overwrites in place."""
    if not os.path.exists(path):
        return False
    src = Image.open(path).convert('RGB')
    box = (max(0, int(x)), max(0, int(y)), min(src.width, int(x + w)), min(src.height, int(y + h)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    out = io.BytesIO()
    src.crop(box).resize((size, size), Image.LANCZOS).save(out, 'WEBP', quality=92)
    with open(path, 'wb') as fh:
        fh.write(out.getvalue())
    return True


def crop_image(user_id, image_id, x, y, w, h):
    """Crop a dataset image to (x,y,w,h), resized to 1024 (no pad). Returns bool."""
    img = _owned_image(user_id, image_id)
    if not img or not img.filename:
        return False
    return _crop_resize_file(_img_path(img), x, y, w, h)


def delete_image(user_id, image_id):
    """Permanently delete a dataset image (DB row + file). If the image is
    still a pending generation, its queue job is cancelled first. Returns bool."""
    img = _owned_image(user_id, image_id)
    if not img:
        return False
    if img.status == 'pending' and not img.filename and img.job_id:  # still generating
        try:
            from ..job_queue import queue_manager
            queue_manager.cancel_job(img.job_id, str(user_id), 'image')
        except Exception:
            pass
    if img.filename:
        try:
            os.remove(_img_path(img))
        except OSError:
            pass
    db.session.delete(img)
    db.session.commit()
    return True


def delete_dataset(user_id, dataset_id):
    """Permanently delete a whole dataset: all image rows + files, the dataset
    row, and its per-dataset image folder. Cancels any in-flight generations
    first. Returns bool (False if not owned)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return False
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
    for img in imgs:
        if img.status == 'pending' and not img.filename and img.job_id:  # still generating
            try:
                from ..job_queue import queue_manager
                queue_manager.cancel_job(img.job_id, str(user_id), 'image')
            except Exception:
                pass
        db.session.delete(img)
    db.session.delete(ds)
    db.session.commit()
    # Drop the per-dataset image folder (files + ref). rmtree, not unlink.
    shutil.rmtree(_dataset_dir(dataset_id), ignore_errors=True)
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
    # Only in-flight generations (pending AND no result file yet) — leave
    # completed-but-uncurated images alone.
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).all())
    n = 0
    for img in rows:
        if img.job_id:  # Klein rows only — API rows never carry a job_id
            try:
                from ..job_queue import queue_manager
                queue_manager.cancel_job(img.job_id, str(user_id), 'image')
            except Exception:
                pass
        db.session.delete(img)
        n += 1
    db.session.commit()
    return n


def purge_unused(user_id, dataset_id):
    """Permanently delete all REJECTED and FAILED images of a dataset (rows +
    files). Returns the number purged."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    rows = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id)
            .filter(FaceDatasetImage.status.in_(('reject', 'failed'))).all())
    n = 0
    for img in rows:
        if delete_image(user_id, img.id):
            n += 1
    return n


def crop_reference(user_id, dataset_id, x, y, w, h):
    """Manually crop the dataset reference image to (x,y,w,h), resized to 1024."""
    ds = get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        return False
    return _crop_resize_file(_ref_path(ds), x, y, w, h)


def dataset_payload(user_id, dataset_id):
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return None
    imgs = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .order_by(FaceDatasetImage.id.desc()).all())
    comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    for i in imgs:
        # Composition counts only usable images: rejected and failed ones don't
        # contribute to the training-target tally the UI tracks deficits against.
        if i.framing in comp and i.status not in ('reject', 'failed'):
            comp[i.framing] += 1
    return {
        'id': ds.id, 'name': ds.name, 'trigger_word': ds.trigger_word,
        'train_type': (ds.train_type or 'zimage'),
        'ref_filename': ds.ref_filename,
        'ref_extra_filenames': extra_ref_filenames(ds), 'composition': comp,
        'face_thresholds': {'green': cfg.get('face_scoring.green'), 'orange': cfg.get('face_scoring.orange')},
        'images': [{'id': i.id, 'filename': i.filename, 'source': i.source,
                    'framing': i.framing, 'variation_label': i.variation_label,
                    'status': i.status, 'caption': i.caption,
                    'face_score': i.face_score, 'face_state': i.face_state} for i in imgs],
        'caption_leak': {
            'leaking': sum(1 for i in imgs
                           if i.status == 'keep' and caption_has_identity_leak(i.caption)),
            'captioned': sum(1 for i in imgs if i.status == 'keep' and i.caption),
        },
    }


# --- Image normalization ---------------------------------------------------
def normalize_to_square_webp(image_bytes: bytes, size: int = 1024) -> bytes:
    """Resize to fit size x size, pad to square (black), return WEBP bytes."""
    im = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    out = io.BytesIO()
    canvas.save(out, 'WEBP', quality=92)
    return out.getvalue()


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
    raw = describe_image_ollama(image_bytes, HEAD_BBOX_PROMPT, num_predict=400, prefer_json=True)
    try:
        s = raw.index('{')
        obj = json.loads(raw[s:raw.index('}', s) + 1])
        y1, x1, y2, x2 = (float(obj[k]) for k in ('y1', 'x1', 'y2', 'x2'))
    except (ValueError, KeyError, AttributeError, TypeError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)


def face_crop_to_square_webp(image_bytes: bytes, size: int = 1024, pad: float = 1.7) -> bytes:
    """Head-crop (Qwen3-VL bbox, generous padding for hair + shoulders) into a
    SQUARE that FILLS `size` — no black padding, no distortion (the square is
    shrunk to fit inside the image so it never needs letterboxing). Falls back to
    a centered-square crop if no head is detected. CALLER holds the GPU window."""
    im = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    W, H = im.size
    norm = detect_head_bbox(image_bytes)
    half = 0
    if norm:
        x1, y1, x2, y2 = norm[0] * W, norm[1] * H, norm[2] * W, norm[3] * H
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2 - (y2 - y1) * 0.10  # shift up to keep the hair
        half = max(x2 - x1, y2 - y1) * pad / 2
        half = min(half, cx, W - cx, cy, H - cy)  # keep the square inside the image
    if half >= 8:
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    else:  # no/failed detection → centered largest square
        side = min(W, H)
        left, top = (W - side) // 2, (H - side) // 2
        box = (left, top, left + side, top + side)
    out = io.BytesIO()
    im.crop(box).resize((size, size), Image.LANCZOS).save(out, 'WEBP', quality=92)
    return out.getvalue()


# --- Import + classify (Qwen3-VL) ------------------------------------------
def import_images(user_id, dataset_id, files_bytes, crop=False):
    """Normalize (or head-crop) + persist + create import rows (status=keep).
    When crop=True, each image is auto head-cropped via Qwen3-VL — the CALLER
    must then hold the GPU-exclusive window — and is by construction a face,
    so framing='face' is set directly (no classify pass needed).
    Returns (ids, failed_count)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return [], 0
    ids = []
    failed = 0
    for raw in files_bytes:
        try:
            webp = face_crop_to_square_webp(raw) if crop else normalize_to_square_webp(raw)
        except Exception as e:
            failed += 1
            logger.warning(f"dataset import: image skipped (dataset {dataset_id}): {e}")
            continue
        fn = f"{user_id}_dataset_{uuid.uuid4().hex[:8]}.webp"
        with open(os.path.join(_dataset_dir(dataset_id), fn), 'wb') as fh:
            fh.write(webp)
        img = FaceDatasetImage(dataset_id=dataset_id, source='import', status='keep',
                               filename=fn, framing='face' if crop else None)
        db.session.add(img)
        db.session.commit()
        ids.append(img.id)
    return ids, failed


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
    try:
        for img in rows:
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
    return n


# --- Captioning (JoyCaption / Qwen3-VL, backend picked in Settings) --------
def caption_images(user_id, dataset_id, force=False, mode=None):
    """Caption les images gardees. Defaut: seulement celles SANS caption ; force=True
    re-capte TOUTES les gardees (ecrase) — pour rejouer apres un changement de prompt.
    Chaque caption passe par drop_identity_sentences (retire une eventuelle phrase
    d'identite isolee).

    `captioning.backend` (réglages) pilote qui capte quoi :
      - 'none'       -> désactivé, RuntimeError (mappée 409 par la route).
      - 'joycaption' -> JoyCaption seul, PAS de repli Ollama.
      - 'ollama'     -> Ollama (Qwen3-VL) seul, JoyCaption jamais tenté.
      - 'auto'       -> comportement historique : JoyCaption en priorité,
                        fallback Ollama pour les images qu'il n'a pas captées."""
    backend = (cfg.get('captioning.backend') or 'auto').lower()
    if backend == 'none':
        raise RuntimeError('No captioning backend configured')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    # Style de caption : prose (Z-Image) vs tags booru (SDXL booru-native type bigLove).
    # Défaut AUTO selon le type entraîné ; un mode explicite (UI) l'emporte.
    ttype = (getattr(ds, 'train_type', None) or 'zimage').lower()
    mode = (mode or ('booru' if ttype == 'sdxl' else 'prose')).lower()
    cap_prompt = CAPTION_PROMPT_BOORU if mode == 'booru' else JOYCAPTION_PROMPT
    cleaner = drop_identity_tags if mode == 'booru' else drop_identity_sentences
    q = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
    if not force:
        q = q.filter((FaceDatasetImage.caption.is_(None)) | (FaceDatasetImage.caption == ''))
    rows = q.all()
    todo = [(img, _img_path(img)) for img in rows if img.filename]
    todo = [(img, p) for img, p in todo if p and os.path.exists(p)]
    if not todo:
        return 0
    n = 0
    remaining = todo
    # 1) JoyCaption en BATCH (un seul chargement du 8B NF4, via le venv ai-toolkit) —
    # sauté entièrement quand le backend force 'ollama'.
    if backend in ('auto', 'joycaption'):
        jc = {}
        try:
            from .joycaption import caption_images_joycaption, is_available
            if is_available():
                # Consigne « ne décris pas le visage » → les traits se lient au trigger,
                # pas aux mots de la caption (deep-research 2026-06-14).
                jc = caption_images_joycaption([p for _, p in todo], prompt=cap_prompt)
            elif backend == 'joycaption':
                # Explicit choice, explicit failure: a user who forced 'joycaption' in
                # Settings must be told it's unavailable, not get a silent 0 (only
                # 'auto' is allowed to fall back to Ollama quietly).
                raise RuntimeError('JoyCaption backend is not available — check the ai-toolkit folder in Settings')
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning('caption_images: JoyCaption indisponible (%s)', e)
        still = []
        for img, p in remaining:
            cap = (jc.get(p) or '').strip().strip('"').strip()
            if cap:
                cleaned = cleaner(cap) or cap
                img.caption = cleaned[:CAPTION_MAX_CHARS]
                db.session.commit()
                n += 1
            else:
                still.append((img, p))
        remaining = still
        if backend == 'joycaption':  # backend forcé JoyCaption -> pas de repli Ollama
            return n
    # 2) Ollama (Qwen3-VL) pour les images non couvertes par JoyCaption ('auto'),
    # ou pour TOUT le lot si le backend force 'ollama'.
    if remaining:
        try:
            from .vision_ollama import describe_image_ollama, unload_vision_model
        except ImportError:
            raise RuntimeError('vision (Ollama) service not configured/available yet')
        try:
            for img, p in remaining:
                with open(p, 'rb') as fh:
                    cap = describe_image_ollama(fh.read(), cap_prompt, num_predict=2000,
                                                keep_alive=_VISION_BATCH_KEEPALIVE)
                cap = (cap or '').strip().strip('"').strip()
                if cap:
                    cleaned = cleaner(cap) or cap
                    img.caption = cleaned[:CAPTION_MAX_CHARS]
                    db.session.commit()
                    n += 1
        finally:
            unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
    return n


# --- Face similarity scoring (InsightFace antelopev2, CPU subprocess) -------
def analyze_faces(user_id, dataset_id) -> dict:
    """Score les images GARDEES vs la reference (InsightFace antelopev2, CPU subprocess).
    Persiste face_score (cosinus brut, None si non note) + face_state. Lot A : AUCUNE
    suppression. Tourne sur CPU -> pas de fenetre GPU. Retourne {state: count}."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset introuvable')
    if not ds.ref_filename:
        raise ValueError('photo de référence manquante')
    ref_path = _ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('photo de référence manquante')
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
    results = score_dataset_faces(ref_path, list(by_path.keys()))
    counts = {}
    for p, img in by_path.items():
        r = results.get(p)
        if not r:
            continue
        img.face_state = r.get('state')
        img.face_score = r.get('sim')   # None si non-scorable
        db.session.commit()
        counts[img.face_state] = counts.get(img.face_state, 0) + 1
    return counts


# --- Fan-out generation (Klein edit) ---------------------------------------
def generate_variations(user_id, dataset_id, variations, multiplier, klein_model,
                        lora_strength=None):
    """For each (variation x multiplier), enqueue a Klein edit of the reference
    and create a pending FaceDatasetImage. Returns the created image ids.

    The row is committed BEFORE enqueuing (so an enqueue/commit failure can never
    leave an untracked orphan job); on enqueue failure the row is marked 'failed'
    and the error re-raised (already-enqueued variations keep their rows)."""
    try:
        from .klein_edit_helper import enqueue_klein_edit
    except ImportError:
        raise RuntimeError('ComfyUI is not configured')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
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
        raise ValueError(f'trop de générations en cours ({in_flight}), attends ou annule')
    ids = []
    for v in variations:
        for _ in range(mult):
            img = FaceDatasetImage(dataset_id=dataset_id, source='generated', status='pending',
                                   variation_label=v.get('label'), framing=v.get('framing'),
                                   variation_prompt=v['prompt'], klein_model=klein_model)
            db.session.add(img)
            db.session.commit()
            try:
                job_id = enqueue_klein_edit(
                    user_id=str(user_id), source_filename=ds.ref_filename,
                    source_path=_ref_path(ds),
                    edit_prompt=wrap_variation(v['prompt']), klein_model=klein_model,
                    lora_strength=lora_strength,
                    extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                                    'variation_label': v.get('label')})
            except Exception:
                img.status = 'failed'
                db.session.commit()
                raise
            img.job_id = job_id
            db.session.commit()
            ids.append(img.id)
    return ids


def regenerate_image(user_id, image_id, lora_strength=None, app=None):
    """Re-enqueue a single generated variation IN PLACE (same row id): cancel any
    in-flight job, drop the old file, reset the row to pending with the new
    job_id. Returns the new job_id, or None if the image is not owned / not a
    generated variation. Raises ValueError if the dataset has no reference or
    the variation prompt can't be recovered."""
    img = _owned_image(user_id, image_id)
    if not img or img.source != 'generated':
        return None
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds.ref_filename:
        raise ValueError('reference image required')
    prompt = img.variation_prompt or prompt_by_label(img.variation_label or '')
    if prompt is None:
        raise ValueError('variation prompt unknown')
    if img.status == 'pending' and not img.filename and img.job_id:  # still generating
        try:
            from ..job_queue import queue_manager
            queue_manager.cancel_job(img.job_id, str(user_id), 'image')
        except Exception:
            pass
    if img.filename:
        try:
            os.remove(_img_path(img))
        except OSError:
            pass

    # API rows (marked klein_model='nanobanana'/'chatgpt', no queue job):
    # regenerate via the SAME engine that produced the row. With an `app`
    # handle the call runs in a background thread (the row flips to in-flight
    # IMMEDIATELY so the tile shows "…" and the polling/banner UI reacts at
    # once); without it the call is synchronous (test path / legacy callers).
    if img.klein_model in API_ENGINES:
        engine = img.klein_model
        api_generate = _api_generate_fn(engine)
        ref_path = _ref_path(ds)
        if not os.path.exists(ref_path):
            raise ValueError('reference image file missing')
        img.filename = None
        img.caption = None
        img.status = 'pending'
        db.session.commit()
        aspect = aspect_for_label(img.variation_label, img.framing)
        ref_bytes = _all_ref_bytes(ds)  # principale + extras (multi-références)
        if app is not None:
            threading.Thread(target=_run_nanobanana_batch,
                             args=(app, [(img.id, prompt, aspect)], ref_bytes, engine),
                             daemon=True).start()
            return engine
        out = api_generate(ref_bytes, wrap_variation(prompt, ref_count=len(ref_bytes)),
                           aspect_ratio=aspect)
        if out:
            fn = f"{user_id}_{_ENGINE_FILE_TAG[engine]}_{uuid.uuid4().hex[:8]}.webp"
            with open(os.path.join(_dataset_dir(img.dataset_id), fn), 'wb') as fh:
                fh.write(normalize_to_webp(out))
            img.filename = fn
        else:
            img.status = 'failed'
        db.session.commit()
        return engine

    try:
        from .klein_edit_helper import enqueue_klein_edit
    except ImportError:
        raise RuntimeError('ComfyUI is not configured')
    job_id = enqueue_klein_edit(
        user_id=str(user_id), source_filename=ds.ref_filename,
        source_path=_ref_path(ds),
        edit_prompt=wrap_variation(prompt), klein_model=img.klein_model,
        lora_strength=lora_strength,
        extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                        'variation_label': img.variation_label})
    img.status = 'pending'
    img.filename = None
    img.caption = None
    img.job_id = job_id
    db.session.commit()
    return job_id


# --- Fan-out generation (API engines: Nano Banana / ChatGPT) ---------------
# Both engines share the exact generate_variation contract (refs + prompt +
# aspect -> bytes|None), so the whole fan-out below is engine-parametric. The
# filename tag keeps the provenance readable in the dataset folder.
API_ENGINES = ('nanobanana', 'chatgpt')
_ENGINE_FILE_TAG = {'nanobanana': 'NBFace', 'chatgpt': 'GPTFace'}


def _api_generate_fn(engine):
    if engine == 'chatgpt':
        from .chatgpt_image import generate_variation
    else:
        from .nanobanana import generate_variation
    return generate_variation


def _run_nanobanana_batch(app, items, ref_bytes, engine='nanobanana'):
    """Worker body: generate each (image_id, prompt) via the selected API engine
    and link the result. Runs in a background thread (factored out so tests can
    call it synchronously). Each row commits independently; an API failure marks
    that row 'failed' (visible + regenerable) without stopping the batch."""
    api_generate = _api_generate_fn(engine)
    from concurrent.futures import ThreadPoolExecutor
    # Guard d'identité adapté au nombre de références (multi = « use EVERY ref »).
    n_refs = len(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else 1
    tag = _ENGINE_FILE_TAG.get(engine, 'NBFace')

    def _one(item):
        # item = (image_id, prompt, aspect) ; aspect optionnel (rétro-compat → '1:1').
        image_id, prompt = item[0], item[1]
        aspect = item[2] if len(item) > 2 else '1:1'
        out = None
        try:
            out = api_generate(ref_bytes, wrap_variation(prompt, ref_count=n_refs),
                               aspect_ratio=aspect)
        except Exception as e:
            logger.warning(f"{engine} batch: generation error for row {image_id}: {e}")
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
            else:
                img.status = 'failed'
            db.session.commit()

    logger.info(f"{engine} batch: start ({len(items)} variation(s))")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_one, items))
    logger.info(f"{engine} batch: done ({len(items)} variation(s))")


def generate_variations_nanobanana(app, user_id, dataset_id, variations, multiplier,
                                   engine='nanobanana'):
    """API fan-out (Nano Banana or ChatGPT, per `engine`): pre-create pending
    rows (job_id stays None — that is the marker for API-generated rows), then
    fill them from a background thread. The existing polling/banner/cancel UI
    works unchanged (pending + no file = in flight). Returns the created ids."""
    if engine not in API_ENGINES:
        raise ValueError(f'unknown API engine: {engine}')
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
            items.append((img.id, v['prompt'], aspect_for_label(v.get('label'), v.get('framing'))))

    threading.Thread(target=_run_nanobanana_batch, args=(app, items, ref_bytes, engine),
                     daemon=True).start()
    return ids


# --- Completion linking (called from the job queue) -------------------------
def link_completed_dataset_image(job_id, filename, failed=False):
    """Attach a finished fan-out job to its FaceDatasetImage row.

    Called from the job-queue completion/failure/cancel paths, which may run in
    a long-lived monitor thread whose SQLAlchemy session holds a STALE read
    snapshot (rows committed by other threads are invisible). If the first
    lookup misses, end the transaction (rollback) and retry on a fresh snapshot
    before concluding the row really doesn't exist."""
    img = FaceDatasetImage.query.filter_by(job_id=job_id).first()
    if img is None:
        db.session.rollback()  # drop the stale read snapshot, then re-read
        img = FaceDatasetImage.query.filter_by(job_id=job_id).first()
    if img is None:
        logger.warning(f"dataset link: no FaceDatasetImage row for job {job_id}")
        return
    if failed:
        img.status = 'failed'
    else:
        output_dir = _comfy_output_dir()
        if output_dir is None:  # ComfyUI not configured -> can't locate the file, treat as failed
            img.status = 'failed'
            logger.warning(f"dataset link: ComfyUI output dir not configured (job {job_id})")
        else:
            img.filename = filename
            # Move the completed file from the shared output dir to the per-dataset dir.
            src = os.path.join(output_dir, filename)
            dst = os.path.join(_dataset_dir(img.dataset_id), filename)
            if os.path.exists(src):
                shutil.move(src, dst)
            elif not os.path.exists(dst):
                logger.warning(f"dataset link: file not found at src={src} or dst={dst}")
    db.session.commit()


# --- Migration helper (run once manually after deploy) ---------------------
def migrate_existing_images_to_per_dataset():
    """Migration helper — run once manually after deploy. Not called automatically."""
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
_INFO = ("Trigger: {trigger}\nImages: {n}\nComposition: {comp}\n\n"
         "ai-toolkit Z-Image suggested: de-distill adapter ON, rank 12-16, ~2000 steps, "
         "batch 1-2, save checkpoint every 500, caption dropout 0.05.\n")


def build_export_zip(user_id, dataset_id) -> bytes:
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    kept = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
            .order_by(FaceDatasetImage.id.asc()).all())
    if not kept:
        raise ValueError('no kept images to export')
    safe = ''.join(c for c in ds.name if c.isalnum() or c in ('-', '_')) or 'dataset'
    comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Garder la PHOTO RÉELLE de référence dans le set : les datasets 100 %
        # synthétiques dérivent de la distribution réelle (deep-research 2026-06-14).
        # On l'inclut comme ancre réelle (_000), caption = trigger seul.
        ref_path = _ref_path(ds) if ds.ref_filename else ''
        if ref_path and os.path.exists(ref_path):
            try:
                rpng = io.BytesIO()
                Image.open(ref_path).convert('RGB').save(rpng, 'PNG')
                zf.writestr(f"{safe}_dataset/{safe}_000_ref.png", rpng.getvalue())
                zf.writestr(f"{safe}_dataset/{safe}_000_ref.txt", ds.trigger_word)
            except OSError:
                pass
        for n, img in enumerate(kept, 1):
            path = _img_path(img) if img.filename else ''
            if not img.filename or not os.path.exists(path):
                continue
            png = io.BytesIO()
            Image.open(path).convert('RGB').save(png, 'PNG')
            base = f"{safe}_dataset/{safe}_{n:03d}"
            zf.writestr(f"{base}.png", png.getvalue())
            cap = (img.caption or '').strip()
            zf.writestr(f"{base}.txt", f"{ds.trigger_word}, {cap}" if cap else ds.trigger_word)
            if img.framing in comp:
                comp[img.framing] += 1
        zf.writestr(f"{safe}_dataset/_dataset_info.txt",
                    _INFO.format(trigger=ds.trigger_word, n=len(kept), comp=comp))
    return buf.getvalue()
