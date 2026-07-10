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
import re
import shutil
import threading
import uuid
import zipfile

from PIL import Image

from ..extensions import db
from ..models import FaceDataset, FaceDatasetImage
from .. import config as cfg

# Garde le modèle vision chaud entre les images d'un même batch caption/classify
# (sinon Ollama le recharge - cold start ~10s - à CHAQUE image). Déchargé en fin
# de batch pour rendre la VRAM à ComfyUI. ComfyUI est déjà en pause pendant la passe.
_VISION_BATCH_KEEPALIVE = '5m'
from .face_variations import (CAPTION_PROMPT, CAPTION_PROMPT_BOORU, CAPTION_PROMPT_CONCEPT,
                              CAPTION_REFINE_CONCEPT_PROMPT, CAPTION_LEAK_FIX_PROMPT,
                              EXPAND_CONCEPT_TERMS_PROMPT,
                              CLASSIFY_PROMPT, HEAD_BBOX_PROMPT,
                              JOYCAPTION_PROMPT, aspect_for_label, caption_prompt_for,
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

# Padding du head-crop AUTO de la référence (côté du carré = grand côté de la bbox
# tête × pad). Volontairement plus large que l'ancien 1.7 (jugé « trop serré ») pour
# garder épaules + contexte par défaut ; le recadrage manuel depuis l'original permet
# d'ajuster ensuite dans les deux sens. Ne concerne QUE la référence (les imports
# gardent le défaut 1.7 de face_crop_to_square_webp).
REF_CROP_PAD = 2.0


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
# Natures de dataset. 'concept' inverse la logique personnage (cf import_images /
# caption_images). Tout le reste (dont NULL) = 'character' (défaut historique).
DATASET_KINDS = ('character', 'concept')


def normalize_kind(kind) -> str | None:
    """'concept' -> 'concept' ; tout le reste -> None (character, stocké NULL)."""
    return 'concept' if (kind or '').strip().lower() == 'concept' else None


def is_concept(ds) -> bool:
    return bool(ds) and (getattr(ds, 'kind', None) or '').lower() == 'concept'


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
TRAIN_TYPES = ('zimage', 'sdxl', 'krea')


def normalize_train_type(t) -> str:
    """Famille valide en minuscules, défaut 'zimage' (toute valeur inconnue/None)."""
    t = (t or '').strip().lower()
    return t if t in TRAIN_TYPES else 'zimage'


def create_dataset(user_id, name, trigger_word, kind=None, concept_desc=None, train_type=None,
                   fidelity=None):
    k = normalize_kind(kind)
    desc = (concept_desc or '').strip()
    if k == 'concept' and not desc:
        # The concept description is what the captioner OMITS; without it the
        # inverted-caption logic has nothing to bind the trigger to. Required.
        raise ValueError('concept_desc required for a concept dataset')
    ds = FaceDataset(user_id=str(user_id), name=(name or '').strip()[:100],
                     trigger_word=(trigger_word or '').strip()[:60] or 'zchar',
                     kind=k, concept_desc=(desc[:500] if k == 'concept' else None),
                     train_type=normalize_train_type(train_type),
                     # fidelity ne concerne que les personnages (un concept décrit
                     # librement les corps — c'est l'acte qui est omis).
                     fidelity=(normalize_fidelity(fidelity) if k != 'concept' else None))
    db.session.add(ds)
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


def _crop_resize_file(path, x, y, w, h, size=1024, dst=None):
    """Crop the file at `path` to (x,y,w,h) and RESIZE the crop to size x size
    (no black padding - the manual crop is square, aspect=1). Writes to `dst`
    (default: overwrite `path`). Passing a distinct `dst` lets the reference crop
    read the untouched full-frame ORIGINAL and write the derived square — so a
    re-crop can widen back out instead of only tightening the previous crop."""
    if not os.path.exists(path):
        return False
    src = Image.open(path).convert('RGB')
    box = (max(0, int(x)), max(0, int(y)), min(src.width, int(x + w)), min(src.height, int(y + h)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    out = io.BytesIO()
    src.crop(box).resize((size, size), Image.LANCZOS).save(out, 'WEBP', quality=92)
    with open(dst or path, 'wb') as fh:
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


# --- Sauvegarde / restauration complète d'un dataset ---------------------------
# ZIP portable (≠ export d'entraînement) : manifest + réglages + TOUTES les images
# avec statuts/captions/scores — pour archiver ou déplacer un dataset entre machines.
BACKUP_FORMAT = 'lds-dataset-backup'
BACKUP_VERSION = 1
_BACKUP_MAX_FILES = 600
_BACKUP_MAX_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB uncompressed (zip-bomb guard)
_BACKUP_NAME_RE = re.compile(r'^[\w.-]+\.(webp|jpg|jpeg|png)$', re.IGNORECASE)

# Champs snapshotés tels quels par ligne image (job_id/klein_model exclus : liés
# à la machine source — un backup restauré ne peut pas « regénérer »).
_BACKUP_IMG_FIELDS = ('filename', 'source', 'framing', 'variation_label', 'status',
                      'caption', 'variation_prompt', 'face_score', 'face_state')


def build_backup_zip(user_id, dataset_id) -> bytes:
    """Self-contained backup of one dataset: manifest.json (settings) +
    images.json (rows) + ref/ + images/ files. Skips rows without a file
    (in-flight/failed generations are not restorable)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    dsdir = _dataset_dir(dataset_id)
    rows = (FaceDatasetImage.query.filter_by(dataset_id=dataset_id)
            .filter(FaceDatasetImage.filename.isnot(None)).all())
    manifest = {
        'format': BACKUP_FORMAT, 'version': BACKUP_VERSION,
        'name': ds.name, 'trigger_word': ds.trigger_word,
        'kind': ds.kind, 'fidelity': ds.fidelity,
        'concept_desc': ds.concept_desc, 'concept_terms': ds.concept_terms,
        'train_type': ds.train_type, 'train_base_model': ds.train_base_model,
        'train_variant': ds.train_variant, 'best_settings': ds.best_settings,
        'ref_filename': ds.ref_filename, 'ref_original_filename': ds.ref_original_filename,
        'ref_extra_filenames': ds.ref_extra_filenames,
    }
    images_meta = [{f: getattr(img, f) for f in _BACKUP_IMG_FIELDS} for img in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=1))
        z.writestr('images.json', json.dumps(images_meta, ensure_ascii=False, indent=1))
        ref_names = [n for n in (ds.ref_filename, ds.ref_original_filename) if n]
        try:
            ref_names += list(json.loads(ds.ref_extra_filenames or '[]'))
        except ValueError:
            pass
        for n in ref_names:
            p = os.path.join(dsdir, n)
            if os.path.isfile(p):
                z.write(p, f'ref/{n}')
        for img in rows:
            p = os.path.join(dsdir, img.filename)
            if os.path.isfile(p):
                z.write(p, f'images/{img.filename}')
    return buf.getvalue()


def import_backup_zip(user_id, zip_bytes):
    """Restore a backup as a NEW dataset (never merges into an existing one).
    Hardened: manifest format/version check, per-entry filename whitelist (no
    separators/traversal), file-count and uncompressed-size caps. Returns the
    created FaceDataset."""
    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise ValueError('not a zip file')
    try:
        manifest = json.loads(z.read('manifest.json').decode('utf-8'))
        images_meta = json.loads(z.read('images.json').decode('utf-8'))
    except (KeyError, ValueError):
        raise ValueError('not a dataset backup (manifest.json/images.json missing or invalid)')
    if manifest.get('format') != BACKUP_FORMAT:
        raise ValueError('not a dataset backup')
    if int(manifest.get('version') or 0) > BACKUP_VERSION:
        raise ValueError('backup made by a newer version of the app - update first')
    infos = [i for i in z.infolist() if i.filename.startswith(('ref/', 'images/'))]
    if len(infos) > _BACKUP_MAX_FILES:
        raise ValueError(f'too many files in backup (max {_BACKUP_MAX_FILES})')
    if sum(i.file_size for i in infos) > _BACKUP_MAX_BYTES:
        raise ValueError('backup too large (max 2 GB uncompressed)')
    name = (manifest.get('name') or 'Restored dataset')[:100]
    trigger = (manifest.get('trigger_word') or 'restored')[:60]
    ds = create_dataset(user_id, name, trigger, kind=manifest.get('kind'),
                        concept_desc=manifest.get('concept_desc'),
                        train_type=manifest.get('train_type'))
    for field in ('concept_terms', 'train_base_model', 'train_variant', 'best_settings',
                  'ref_filename', 'ref_original_filename', 'ref_extra_filenames', 'fidelity'):
        setattr(ds, field, manifest.get(field))
    dsdir = _dataset_dir(ds.id)
    os.makedirs(dsdir, exist_ok=True)
    extracted = set()
    for info in infos:
        base = os.path.basename(info.filename)
        if not _BACKUP_NAME_RE.match(base) or base != info.filename.split('/', 1)[1]:
            continue   # nested path or weird name -> skip, never traverse
        with z.open(info) as src, open(os.path.join(dsdir, base), 'wb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        extracted.add(base)
    n_rows = 0
    for meta in images_meta:
        fn = meta.get('filename')
        if not fn or fn not in extracted:
            continue   # metadata without its file -> drop the row, not the import
        img = FaceDatasetImage(dataset_id=ds.id,
                               **{f: meta.get(f) for f in _BACKUP_IMG_FIELDS if f != 'filename'},
                               filename=fn)
        db.session.add(img)
        n_rows += 1
    # Refs referenced by the manifest but absent from the zip -> clear (no dangling).
    if ds.ref_filename and ds.ref_filename not in extracted:
        ds.ref_filename = None
    if ds.ref_original_filename and ds.ref_original_filename not in extracted:
        ds.ref_original_filename = None
    db.session.commit()
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
        new = new.strip()[:CAPTION_MAX_CHARS] or None
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
    return _crop_resize_file(_ref_crop_source_path(ds), x, y, w, h, dst=_ref_path(ds))


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
    concept = is_concept(ds)
    body = is_body_fidelity(ds)
    return {
        'id': ds.id, 'name': ds.name, 'trigger_word': ds.trigger_word,
        'train_type': (ds.train_type or 'zimage'),
        'kind': 'concept' if concept else 'character',
        'fidelity': (ds.fidelity or 'face') if not concept else 'face',
        'concept_desc': (ds.concept_desc or '') if concept else '',
        'ref_filename': ds.ref_filename,
        'ref_original_filename': ds.ref_original_filename or '',
        'ref_extra_filenames': extra_ref_filenames(ds), 'composition': comp,
        'face_thresholds': {'green': cfg.get('face_scoring.green'), 'orange': cfg.get('face_scoring.orange')},
        'images': [{'id': i.id, 'filename': i.filename, 'source': i.source,
                    'framing': i.framing, 'variation_label': i.variation_label,
                    'status': i.status, 'caption': i.caption,
                    'face_score': i.face_score, 'face_state': i.face_state} for i in imgs],
        # Dataset CONCEPT : décrire l'identité est VOULU (le concept, pas le visage,
        # se lie au trigger) → le badge « fuite d'identité » n'a aucun sens, on le zéro.
        # Fidélité corps : les marques corporelles comptent aussi comme fuite.
        'caption_leak': {
            'leaking': 0 if concept else sum(
                1 for i in imgs if i.status == 'keep' and caption_has_identity_leak(i.caption, body=body)),
            'captioned': sum(1 for i in imgs if i.status == 'keep' and i.caption),
        },
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


def face_crop_to_square_webp(image_bytes: bytes, size: int = 1024, pad: float = 1.7,
                             *, return_detected: bool = False):
    """Head-crop (Qwen3-VL bbox, generous padding for hair + shoulders) into a
    SQUARE that FILLS `size` - no black padding, no distortion (the square is
    shrunk to fit inside the image so it never needs letterboxing). Falls back to
    a centered-square crop if no head is detected. CALLER holds the GPU window.

    `return_detected=True` -> (webp_bytes, head_detected) so the caller can WARN the
    user when it silently fell back to a centered crop (e.g. vision model not pulled)
    instead of leaving them puzzled by a body-centered reference."""
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
    head_detected = half >= 8
    if head_detected:
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    else:  # no/failed detection → centered largest square
        side = min(W, H)
        left, top = (W - side) // 2, (H - side) // 2
        box = (left, top, left + side, top + side)
    out = io.BytesIO()
    im.crop(box).resize((size, size), Image.LANCZOS).save(out, 'WEBP', quality=92)
    return (out.getvalue(), head_detected) if return_detected else out.getvalue()


# --- Import + classify (Qwen3-VL) ------------------------------------------
def import_images(user_id, dataset_id, files_bytes, crop=False, dedupe=False, stats=None):
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

    Returns (ids, failed_count)."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return [], 0
    # Sans head-crop, on préserve TOUJOURS le ratio (normalize_to_webp) : l'ancien
    # chemin « carré padé » ajoutait des bandes noires que le LoRA apprendrait, et
    # forçait tous les imports personnage en carré — un plan buste/corps importé
    # doit rester tel quel (ai-toolkit gère le bucketing multi-ratios).
    seen = _existing_dhashes(dataset_id) if dedupe else None
    ids = []
    failed = 0
    for raw in files_bytes:
        try:
            webp = face_crop_to_square_webp(raw) if crop else normalize_to_webp(raw)
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
        img = FaceDatasetImage(dataset_id=dataset_id, source='import', status='keep',
                               filename=fn, framing='face' if crop else None)
        db.session.add(img)
        db.session.commit()
        ids.append(img.id)
    return ids, failed


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


def _accept_scrape_bytes(raw, seen_hashes, skipped):
    """Filtre une image téléchargée : résolution / ratio / dedup perceptuel.
    Retourne les bytes si acceptée (et enregistre son dHash dans seen_hashes),
    sinon None en incrémentant le compteur skipped adéquat."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            w, h = im.size
            if min(w, h) < SCRAPE_IMPORT_MIN_SIDE:
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


def scrape_import_urls(user_id, dataset_id, items):
    """Télécharge les images scannées SÉLECTIONNÉES directement dans le dataset
    concept — flux AUTONOME. `items` = [{'url','title'}]. Download parallélisé
    (borné), puis filtre + dedup séquentiels (état partagé), puis import brut
    aspect-kept via import_images(crop=False). Renvoie
    {'imported': n, 'skipped': {duplicates, low_res, extreme_ratio, not_image, errors}}."""
    from concurrent.futures import ThreadPoolExecutor
    skipped = {'duplicates': 0, 'low_res': 0, 'extreme_ratio': 0,
               'not_image': 0, 'errors': 0}
    items = [it for it in (items or []) if isinstance(it, dict) and it.get('url')]
    if not items:
        return {'imported': 0, 'skipped': skipped}
    with ThreadPoolExecutor(max_workers=_SCRAPE_DL_WORKERS) as pool:
        downloaded = list(pool.map(_download_scrape_item, items))

    seen_hashes = _existing_dhashes(dataset_id)
    accepted = []
    for reason, data in downloaded:
        if reason != 'ok':
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        ok_bytes = _accept_scrape_bytes(data, seen_hashes, skipped)
        if ok_bytes is not None:
            accepted.append(ok_bytes)
    ids, failed = import_images(user_id, dataset_id, accepted, crop=False)
    skipped['errors'] += failed
    return {'imported': len(ids), 'skipped': skipped}


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
_REFINE_REASONING_RE = re.compile(
    r'\b(the (?:problem|instruction|task|draft|original) (?:says|mentions|has)'
    r'|we (?:need|can|should) to (?:remove|rephrase|avoid)'
    r'|so we (?:need|can|should)|let me |\brephrase\b|\bwait,'
    r'|now, (?:remove|the|rephrase|let|we))', re.I)


def _refine_output_ok(text, prior) -> bool:
    """True if `text` looks like a CLEAN caption - neither the Qwen reasoning trace
    nor a loop/rambling (bounded to ~2x the source caption `prior`)."""
    t = (text or '').strip()
    if not t or _REFINE_REASONING_RE.search(t):
        return False
    return len(t) <= 2 * len(prior or '') + 400


# Words from concept_desc that are never discriminating (articles + generic adjectives
# a legit caption uses elsewhere: "bare shoulders", "full-body"...).
_TERMS_STOP = frozenset((
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'with', 'to', 'from',
    'that', 'this', 'as', 'is', 'are', 'his', 'her', 'their', 'its', 'it', 'one',
    'act', 'shown', 'worn', 'being', 'person', 'subject', 'focal', 'point', 'visible',
    'bare', 'exposed', 'full', 'close', 'closeup', 'close-up', 'wearing', 'showing'))


def _fallback_concept_terms(desc) -> list:
    """Minimal ban-list WITHOUT the LLM: the meaningful words of concept_desc itself
    (always included, even when the LLM expansion succeeds - the user's words are the
    ground truth)."""
    words = re.split(r'[^a-zA-Z-]+', (desc or '').lower())
    return sorted({w.strip('-') for w in words
                   if len(w.strip('-')) >= 3 and w.strip('-') not in _TERMS_STOP})


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
    """Extract {"terms": [...]} from an LLM output (tolerates noise around the object)."""
    raw = raw or ''
    start, end = raw.find('{'), raw.rfind('}')
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except ValueError:
        return []
    terms = data.get('terms') if isinstance(data, dict) else None
    if not isinstance(terms, list):
        return []
    out = []
    for t in terms:
        if isinstance(t, str):
            t = t.strip().lower()
            if 3 <= len(t) <= 40 and t not in _TERMS_STOP:
                out.append(t)
    return sorted(set(out))[:40]


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
                    num_predict=5000, prefer_json=True, fmt='json',
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


def _caption_concept(ds, force, backend):
    """Concept caption pipeline (INVERTED logic): describe everything INCLUDING identity
    but OMIT the recurring act so it binds to the trigger. JoyCaption is literal (it NAMES
    the act/fluids/watermark) -> its drafts are REFINED by Qwen, then every caption passes
    the ban-list omission guarantee. Backend gating is honored:
      - 'joycaption' -> Joy drafts only + mechanical scrub (no Qwen calls);
      - 'ollama'     -> Joy skipped, every image direct-Qwen + enforcement;
      - 'auto'       -> Joy drafts refined by Qwen, no-Joy images direct-Qwen, all enforced."""
    concept_desc = (ds.concept_desc or '').strip()
    cap_prompt = CAPTION_PROMPT_CONCEPT.format(concept=concept_desc)
    q = FaceDatasetImage.query.filter_by(dataset_id=ds.id, status='keep')
    if not force:
        q = q.filter((FaceDatasetImage.caption.is_(None)) | (FaceDatasetImage.caption == ''))
    todo = [(img, _img_path(img)) for img in q.all() if img.filename]
    todo = [(img, p) for img, p in todo if p and os.path.exists(p)]
    if not todo:
        return 0
    n = 0
    remaining = list(todo)
    refine_targets = []  # (img, p, joycap) -> Joy draft refined by Qwen
    # 1) JoyCaption batch (draft) when the backend allows it.
    if backend in ('auto', 'joycaption'):
        jc = {}
        try:
            from .joycaption import caption_images_joycaption, is_available
            if is_available():
                jc = caption_images_joycaption([p for _, p in todo], prompt=cap_prompt)
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
            try:
                with open(p, 'rb') as fh:
                    data = fh.read()
            except OSError:
                data = b''
            final = _enforce_concept_omission(joycap, leak_re, data, concept_desc) or joycap
            img.caption = final[:CAPTION_MAX_CHARS]
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
                with open(p, 'rb') as fh:
                    data = fh.read()
                refined = ''
                try:
                    refined = describe_image_ollama(
                        data, CAPTION_REFINE_CONCEPT_PROMPT.format(existing=joycap,
                                                                   concept=concept_desc),
                        num_predict=5000, keep_alive=_VISION_BATCH_KEEPALIVE)
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
                                                    keep_alive=_VISION_BATCH_KEEPALIVE)
                    except Exception:  # noqa: BLE001
                        alt = ''
                    alt = (alt or '').strip().strip('"').strip()
                    final = alt or joycap
                final = _enforce_concept_omission(final, leak_re, data, concept_desc,
                                                  describe=describe_image_ollama) or final
                img.caption = final[:CAPTION_MAX_CHARS]
                db.session.commit()
                n += 1
            for img, p in remaining:
                with open(p, 'rb') as fh:
                    data = fh.read()
                cap = describe_image_ollama(data, cap_prompt, num_predict=2000,
                                            keep_alive=_VISION_BATCH_KEEPALIVE)
                cap = (cap or '').strip().strip('"').strip()
                if cap:
                    cap = _enforce_concept_omission(cap, leak_re, data, concept_desc,
                                                    describe=describe_image_ollama) or cap
                    img.caption = cap[:CAPTION_MAX_CHARS]
                    db.session.commit()
                    n += 1
        finally:
            unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
    return n


def caption_images(user_id, dataset_id, force=False, mode=None):
    """Caption les images gardees. Defaut: seulement celles SANS caption ; force=True
    re-capte TOUTES les gardees (ecrase) - pour rejouer apres un changement de prompt.
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
    # Dataset CONCEPT : logique INVERSÉE (décrire tout SAUF l'acte récurrent → il se lie
    # au trigger). Pipeline dédié Joy→Qwen + garantie d'omission (ban-list) : entièrement
    # à part du chemin character ci-dessous. Respecte le backend gating.
    if is_concept(ds):
        return _caption_concept(ds, force, backend)
    # Style de caption : prose (Z-Image) vs tags booru (SDXL booru-native type bigLove).
    # Défaut AUTO selon le type entraîné ; un mode explicite (UI) l'emporte.
    ttype = (getattr(ds, 'train_type', None) or 'zimage').lower()
    mode = (mode or ('booru' if ttype == 'sdxl' else 'prose')).lower()
    # Fidélité corps : le prompt bannit EN PLUS les marques corporelles permanentes
    # (tatouages/cicatrices/piercings…) et le post-filtre les retire — elles doivent
    # se lier au trigger, pas aux mots (même principe que le visage).
    body = is_body_fidelity(ds)
    cap_prompt = caption_prompt_for(mode, body=body)
    base_cleaner = drop_identity_tags if mode == 'booru' else drop_identity_sentences
    def cleaner(text):
        return base_cleaner(text, body=body)
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
    # 1) JoyCaption en BATCH (un seul chargement du 8B NF4, via le venv ai-toolkit) -
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
                raise RuntimeError('JoyCaption backend is not available - check the ai-toolkit folder in Settings')
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
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
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
        # Stop AVANT l'appel API : cancel_pending supprime les lignes en vol — si
        # celle-ci a disparu, ne pas payer une génération qui sera jetée (le bouton
        # Stop doit économiser le RESTE du batch, pas seulement masquer les tuiles).
        with app.app_context():
            row = db.session.get(FaceDatasetImage, image_id)
            if row is None or row.status != 'pending':
                logger.info(f"{engine} batch: row {image_id} cancelled - API call skipped")
                return
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
    rows (job_id stays None - that is the marker for API-generated rows), then
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
