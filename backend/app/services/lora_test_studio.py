"""LoRA Test Studio - checkpoint x strength sweep over the Z-Image pipeline.

MVP of the « Studio de test de LoRA » (design 2026-06-12) : pour un dataset
entraîné, balaye une grille checkpoint x strength en générations Z-Image
(seed fixe, 1 prompt identité), note 👍/👎 chaque cellule et persiste les
réglages gagnants sur le FaceDataset.

Clones the dataset fan-out mechanics exactly:
  - row committed BEFORE enqueue (no orphan jobs),
  - queue jobs tagged with metadata ``is_lora_test`` and linked back on
    completion/failure/cancel by ``link_completed_test_image`` (called from
    job_queue, same anchor point as ``is_dataset``),
  - completed files moved to the per-dataset folder,
  - free (never debited) but hard-capped (MAX_TEST_IMAGES per run, one active
    run per dataset, refused while training/vision holds the GPU).

Lifted from the parent project's app/services/lora_test_studio.py (1981
lines) for LoRA Dataset Studio: SRC's module-level WORKFLOW_ZTURBO_PATH /
WORKFLOW_HQ_PATH / WORKFLOW_KREA_TURBO_PATH constants become
``cfg.BACKEND_DIR / 'workflows' / '<name>.json'`` accessors below;
COMFYUI_OUTPUT_DIR becomes the live `_comfy_output_dir()` accessor (same
pattern as klein_edit_helper). Single-user app: the ownership subsystem
(`lora_ownership.filenames_owned_by_others`, cross-user `_run_owned` /
`_owned_test_image` checks) is dropped - everything on disk that matches a
dataset's trigger boundary IS that dataset's checkpoint, and every test-image
row IS the local user's. `save_test_image_to_gallery` /
`_studio_image_to_generation_settings` and the `GenerationLog`
history-hiding stanza are dropped too - this app has no gallery/generator
log to save into or hide from (`saved_to_gallery` isn't a column on our
`LoraTestImage`).
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import shutil
import uuid
from datetime import datetime

from .. import config as cfg
from ..extensions import db
from ..models import FaceDataset, LoraTestImage
from . import face_dataset_service as fds
from ..job_queue import queue_manager
from ..utils.comfyui import (FAMILY_LABELS, KREA_ALLOWED_SAMPLERS, KREA_ALLOWED_SCHEDULERS,
                             KREA_ALLOWED_WEIGHT_DTYPES, apply_optimal_sampler_params,
                             family_of_lora, format_trained_lora_label, get_krea_loras,
                             get_sdxl_loras, get_zimage_loras, get_zimage_models,
                             inject_krea2t_enhancer, load_workflow_local,
                             resolve_checkpoint_ckpt_name)
from ..utils.zimage_helper import apply_zimage_settings

logger = logging.getLogger(__name__)

# Plafond dur d'images par run (~4-6 min de GPU max en Z-Image Turbo).
MAX_TEST_IMAGES = 24

# Prompt preset d'identité (le trigger word du dataset est substitué).
IDENTITY_PROMPT_TEMPLATE = "{trigger}, close-up portrait, neutral expression, looking at camera"

# Résolution du workflow ZTurbo (constante implicite du design).
TEST_WIDTH, TEST_HEIGHT = 832, 1216

# Chemins des workflows (copies verbatim de SRC/workflows/image-generation/).
WORKFLOW_ZTURBO_PATH = cfg.BACKEND_DIR / 'workflows' / 'ZImage_bigLove_ZT3_optimal.json'
WORKFLOW_HQ_PATH = cfg.BACKEND_DIR / 'workflows' / 'image_real_HQ.json'
WORKFLOW_KREA_TURBO_PATH = cfg.BACKEND_DIR / 'workflows' / 'krea2_turbo.json'
WORKFLOW_KREA_IMG2IMG_PATH = cfg.BACKEND_DIR / 'workflows' / 'krea2_turbo_img2img.json'


def _comfy_output_dir():
    d = cfg.comfyui_dir('output')
    return str(d) if d else None


# Formats testables (≈1 MP, multiples de 64 - sûrs pour Z-Image). Le cadrage peut
# influencer le rendu du LoRA (« la balance »), d'où le choix laissé à l'utilisateur.
TEST_ASPECTS = {
    '9:16': (832, 1216),
    '3:4':  (896, 1152),
    '1:1':  (1024, 1024),
    '4:3':  (1152, 896),
    '16:9': (1216, 832),
}
# SDXL : MÊMES formats, mais côté long plafonné à 1024 = la base SDXL qui ne duplique
# pas (les buckets ≈1 MP de Z-Image, côté long 1216, déforment les merges/DMD SDXL type
# bigLove/mopMix). Multiples de 64. Choix utilisateur 2026-06-24 (« SDXL-safe ≤1024 »).
TEST_ASPECTS_SDXL = {
    '9:16': (576, 1024),
    '3:4':  (768, 1024),
    '1:1':  (1024, 1024),
    '4:3':  (1024, 768),
    '16:9': (1024, 576),
}
# Formats Studio → valeurs d'aspectRatio de Generate (miroir de
# react-frontend/src/components/dataset/studio/constants.js:ASPECT_TO_GENERATE).
_STUDIO_ASPECT_TO_GENERATE = {
    '9:16': 'portrait', '3:4': 'portrait', '1:1': 'square',
    '4:3': 'landscape', '16:9': 'landscape',
}
_MODE_LABEL_BY_FAMILY = {'zimage': 'Z-Image', 'krea': 'Krea 2 Turbo', 'sdxl': 'SDXL'}
DEFAULT_ASPECT = '9:16'
# Paliers de résolution (parité Generate) - mêmes clés que resolution.py/_TIERS. NULL =
# table de formats fixe historique (comportement inchangé si le front n'envoie rien).
RESOLUTION_TIERS = ('fast', 'standard', 'hq', 'max')
# Table de correspondance format studio ('9:16'…) → vocabulaire nommé de compute_tier_dims
# ('square','landscape'…). Le studio n'expose que ces 5 ratios.
_ASPECT_TO_TIER_RATIO = {
    '1:1': 'square', '4:3': 'landscape', '3:4': 'portrait',
    '16:9': 'widescreen', '9:16': 'tall',
}


def _aspect_dims(aspect, train_type=None, resolution_tier=None):
    """(width, height) d'un format. Si `resolution_tier` (fast|standard|hq|max) est fourni,
    délègue à `compute_tier_dims` (ratio nommé + mégapixels du palier, comme Generate) ;
    sinon table fixe par famille (SDXL côté long ≤1024, sinon table Z-Image historique).
    Format inconnu → défaut. SDXL + palier : on re-borne le côté long à 1024 (bande
    SDXL-safe, multiples de 64) car compute_tier_dims monte jusqu'à 1536 (safe Z-Image,
    déforme les merges/DMD SDXL)."""
    if resolution_tier in RESOLUTION_TIERS:
        named = _ASPECT_TO_TIER_RATIO.get(aspect)
        if named:
            from ..utils.resolution import compute_tier_dims
            w, h = compute_tier_dims(named, resolution_tier)
            if (train_type or '').lower() == 'sdxl':
                longest = max(w, h)
                if longest > 1024:
                    sc = 1024.0 / longest
                    w = max(64, int(round(w * sc / 64)) * 64)
                    h = max(64, int(round(h * sc / 64)) * 64)
            return w, h
    table = TEST_ASPECTS_SDXL if (train_type or '').lower() == 'sdxl' else TEST_ASPECTS
    return table.get(aspect, table[DEFAULT_ASPECT])

# Axes optionnels CFG / steps (Z-Image Turbo : défaut cfg=1.0, 8 steps). Tester
# plusieurs valeurs aide à trouver le réglage qui tient le mieux l'identité.
DEFAULT_CFG = 1.0
DEFAULT_STEPS = 8
CFG_CHOICES = [1.0, 1.5, 2.0, 2.5, 3.0]
STEPS_CHOICES = [6, 8, 10, 12, 16, 20, 24, 32, 40]


def _basename(path: str) -> str:
    """Basename tolerant to ComfyUI's backslash-relative LoRA paths."""
    return (path or '').replace('\\', '/').rsplit('/', 1)[-1]


def _wilson_lower_bound(likes: int, voted: int, z: float = 1.96) -> float:
    """Borne basse de l'intervalle de Wilson (95%) sur le taux de 👍.

    C'est la métrique de tri correcte pour « meilleure config d'après les votes » :
    un compte brut (likes − dislikes) favorise les configs simplement TESTÉES plus
    souvent ; le taux brut (likes/voted) favorise les configs à 1 seul vote. Wilson
    combine taux ÉLEVÉ *et* confiance (nb de votes) : 2👍/2 (0.34) bat 6👍4👎 (0.31),
    et 5👍/5 (0.57) bat 2👍/2 (0.34). 0.0 si aucun vote."""
    if voted <= 0:
        return 0.0
    p = likes / voted
    z2 = z * z
    denom = 1.0 + z2 / voted
    centre = p + z2 / (2 * voted)
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * voted)) / voted)
    return (centre - margin) / denom


def identity_prompt(ds) -> str:
    return IDENTITY_PROMPT_TEMPLATE.format(trigger=(ds.trigger_word or '').strip())


def _prompt_with_trigger(prompt, trigger_word):
    """Préfixe le trigger word du dataset au prompt (même ordre que
    IDENTITY_PROMPT_TEMPLATE), SAUF si prompt/trigger vide ou si le trigger est déjà
    présent comme TOKEN entier (insensible à la casse) → dédup, pas de doublon.

    Utilisé UNIQUEMENT au montage du workflow (`_build_cell_workflow`) : le prompt
    stocké sur la cellule reste BRUT (menu « prompts récents » propre)."""
    p = (prompt or '').strip()
    t = (trigger_word or '').strip()
    if not p or not t:
        return p
    if re.search(r'(?:^|[^0-9A-Za-z])' + re.escape(t) + r'(?:[^0-9A-Za-z]|$)', p, re.IGNORECASE):
        return p
    return f'{t}, {p}'


# --- Discovery ---------------------------------------------------------------
# Familles testables, dans l'ordre d'affichage du sélecteur. Libellés = source
# unique partagée avec le label de LoRA (app.utils.comfyui.FAMILY_LABELS).
FAMILIES = ('zimage', 'sdxl', 'krea')


def _pool_for_family(family: str) -> list[dict]:
    """Pool de LoRA d'une famille : SDXL → loras/sdxl, Krea → loras/krea, sinon
    loras/z image. Source unique du branchement par pipeline."""
    f = (family or 'zimage').lower()
    if f == 'sdxl':
        return get_sdxl_loras()
    if f == 'krea':
        return get_krea_loras()
    return get_zimage_loras()


def _trigger_token_match(norm: str, trigger: str) -> bool:
    """True si `norm` commence par `trigger` SUIVI d'un séparateur (`_`/`-`) ou de la
    fin de chaîne - le trigger doit être un TOKEN entier, pas juste un préfixe.

    ⚠️ Régression corrigée (bug user 2026-07-01) : un simple `startswith` faisait
    qu'un trigger COURT s'offrait les LoRA d'un trigger plus LONG qui le préfixe
    ('lola' ⊂ 'lola3869' ⊂ 'lola2') - ex. le dataset 'Lola' affichait les checkpoints
    'lola3869'. Le nom est toujours '<trigger>-<step>' ou '<trigger>_<step>' (ou le
    trigger nu), donc le caractère juste APRÈS le trigger doit être un séparateur."""
    if not norm.startswith(trigger):
        return False
    rest = norm[len(trigger):]
    return rest == '' or rest[0] in ('_', '-')


def _trigger_match_checkpoints(ds, family=None) -> list[dict]:
    """Checkpoints dont le nom matche le trigger word du dataset. Base commune à
    `list_test_checkpoints`. Deux conventions (insensible à la casse), car
    import_checkpoint copie le nom brut ai-toolkit alors que d'anciens imports étaient
    renommés :
      - '<Trigger>-<step>'        (nom propre,  ex. Lola-500)
      - 'lora_<Trigger>_<step>'   (nom brut ai-toolkit, ex. lora_EVA6938_000001000)
    Le POOL scanné dépend de `family` (sélecteur de famille du studio) ; à défaut on
    retombe sur `ds.train_type`. Un même dataset entraîné sous PLUSIEURS pipelines a
    des LoRA dans plusieurs dossiers (loras/sdxl, loras/krea, loras/z image) → c'est
    `family` qui choisit lequel exposer. Le match est délimité par un séparateur
    (cf. `_trigger_token_match`) : un trigger préfixe d'un autre ('lola' ⊂ 'lola3869')
    ne s'offre PAS les LoRA du voisin. Returns [{filename, label}] (forme LoraLoader)."""
    trigger = (ds.trigger_word or '').strip().lower()
    if not trigger:
        return []
    fam = (family or getattr(ds, 'train_type', None) or 'zimage').lower()
    pool = _pool_for_family(fam)
    out = []
    for lora in pool:
        base = _basename(lora['filename'])
        stem = base.rsplit('.', 1)[0]
        norm = stem.lower()
        if norm.startswith('lora_'):  # tolère le préfixe brut ai-toolkit
            norm = norm[len('lora_'):]
        if _trigger_token_match(norm, trigger):
            out.append({'filename': lora['filename'],
                        'label': format_trained_lora_label(lora['filename'], fam) or stem})
    return out


def list_test_checkpoints(ds, family=None) -> list[dict]:
    """Checkpoints testables pour ce dataset = trigger match (dans la famille donnée).
    `ds` est déjà restreint au user appelant en amont (single-user app : pas de
    filtre d'ownership cross-user). Returns [{filename, label}], filename en forme
    LoraLoader."""
    return _trigger_match_checkpoints(ds, family)


def available_families(ds) -> list[dict]:
    """Familles (pipelines) sous lesquelles CE dataset a effectivement été entraîné =
    celles dont le pool contient ≥1 checkpoint testable (trigger match).
    Le même dataset peut apparaître sous plusieurs familles (ex. lola2 en ZIT+SDXL+Krea).
    Returns [{family, label, count}], ordre FAMILIES. Vide si aucun LoRA déployé."""
    out = []
    for fam in FAMILIES:
        n = len(list_test_checkpoints(ds, fam))
        if n:
            out.append({'family': fam, 'label': FAMILY_LABELS.get(fam, fam), 'count': n})
    return out


def permanent_lora_candidates(family) -> list[dict]:
    """LoRA « always-on » (style/utilitaire) proposables en mode PERMANENT dans le studio :
    les entrées du pool de la famille dont le nom NE commence PAS par `lora_` (= pas un
    checkpoint de personnage ai-toolkit, mais un LoRA de style/effet - ex. Krea
    realism_engine_krea2, krea2filterbypass3, PornMaster_Detail_Slider…). Ce sont des LoRA
    partagés (pas de scoping owner). Returns [{filename, label}] (label = displayName du pool).
    Concrètement surtout pour Krea (les dossiers sdxl/z-image ne contiennent que des `lora_*`)."""
    out = []
    for lora in _pool_for_family(family):
        base = _basename(lora['filename'])
        if base.lower().startswith('lora_'):
            continue  # personnage entraîné → c'est un AXE de test, pas un always-on
        out.append({'filename': lora['filename'],
                    'label': lora.get('displayName') or base.rsplit('.', 1)[0]})
    return out


def _resolve_family(ds, requested, families=None) -> str:
    """Famille effective du studio : la `requested` si elle est réellement présente ;
    sinon le `train_type` du dataset s'il est présent ; sinon la 1ʳᵉ famille présente ;
    sinon le `train_type` brut (fallback ultime, pool potentiellement vide). Garantit
    qu'on n'affiche jamais une famille sans aucun LoRA quand d'autres en ont."""
    fams = available_families(ds) if families is None else families
    keys = [f['family'] for f in fams]
    req = (requested or '').lower()
    if req in keys:
        return req
    default = (getattr(ds, 'train_type', None) or 'zimage').lower()
    if default in keys:
        return default
    return keys[0] if keys else default


def list_sdxl_base_models() -> list[dict]:
    """Checkpoints SDXL utilisables comme BASE de test = ceux de Generate.
    Returns [{filename, label}]."""
    from ..utils.comfyui import get_checkpoint_models
    out = []
    for m in get_checkpoint_models():
        name = m.get('name')
        if name:
            out.append({'filename': name, 'label': name.split('\\')[-1]})
    return out


def list_all_testable_checkpoints(user_id) -> list[dict]:
    """Pour le sélecteur autonome : agrège les checkpoints testables de TOUS les
    datasets du user, UNE ENTRÉE PAR (dataset × famille).

    ⚠️ Un dataset est MULTI-FAMILLE : le même trigger peut être déployé sous
    loras/{z image, sdxl, krea}. On itère donc `available_families(ds)` (qui dérive la
    famille du DOSSIER via family_of_lora, pas du scalaire `ds.train_type`) et on émet
    une entrée par famille présente.

    [{dataset_id, dataset_name, lora_label, trigger_word, family, family_label,
      train_type (= family, pour le badge front), checkpoints:[{filename,label}]}]."""
    out = []
    datasets = (FaceDataset.query.filter_by(user_id=str(user_id))
                .order_by(FaceDataset.id.asc()).all())
    for ds in datasets:
        for fam in available_families(ds):   # {'family','label','count'} par famille présente
            cks = list_test_checkpoints(ds, fam['family'])
            if not cks:
                continue
            out.append({'dataset_id': ds.id, 'dataset_name': ds.name,
                        'lora_label': ds.trigger_word or ds.name,
                        'trigger_word': ds.trigger_word,
                        'family': fam['family'],
                        'family_label': fam['label'],
                        'train_type': fam['family'],   # badge/verrou front = famille de CETTE entrée
                        'checkpoints': cks})
    return out


# --- Guards ------------------------------------------------------------------
def gpu_busy_reason() -> str | None:
    """Return a human error when the GPU is held by a long-running exclusive
    task (LoRA training / vision pass), else None. The queue itself serializes
    normal generations, so no further locking is needed."""
    if queue_manager._get_system_state('training_in_progress', False):
        return "LoRA training in progress - the studio is unavailable (GPU busy)."
    if queue_manager._get_system_state('vision_in_progress', False):
        return "Vision pass in progress (GPU busy) - try again in a moment."
    return None


def _active_run_count(dataset_id=None) -> int:
    """In-flight cells (pending, no file yet). dataset_id=None → garde GLOBALE
    (tous datasets confondus, ce qu'exige une comparaison multi-LoRA) ; fourni →
    une seule run active par dataset (comportement historique)."""
    q = (LoraTestImage.query
         .filter_by(status='pending')
         .filter(LoraTestImage.filename.is_(None)))
    if dataset_id is not None:
        q = q.filter_by(dataset_id=dataset_id)
    return q.count()


def build_matrix(checkpoints, strengths, aspects=None, cfgs=None, steps_list=None, steps2_list=None) -> list[tuple]:
    """Materialize the (checkpoint, strength, aspect) grid cells, validated:
    non-empty checkpoint/strength axes, strengths in [0.05, 2.0] (deduped, order
    kept), aspects within the whitelist (deduped, défaut 9:16). PAS de plafond sur
    le nombre de cellules : la file est sérielle et l'utilisateur voit le compte +
    l'estimation de durée avant de lancer (choix assumé sur sa propre machine)."""
    cps = [c for c in (checkpoints or []) if isinstance(c, str) and c.strip()]
    sts = []
    for s in (strengths or []):
        try:
            v = round(float(s), 2)
        except (TypeError, ValueError):
            raise ValueError(f'invalid strength: {s!r}')
        if not 0.05 <= v <= 2.0:
            raise ValueError(f'strength out of range [0.05, 2.0]: {v}')
        if v not in sts:
            sts.append(v)
    asp = []
    for a in (aspects or []):
        if a in TEST_ASPECTS and a not in asp:
            asp.append(a)
    if not asp:
        asp = [DEFAULT_ASPECT]
    cfs = []
    for v in (cfgs or []):
        try:
            fv = round(float(v), 2)
        except (TypeError, ValueError):
            continue
        if 1.0 <= fv <= 15.0 and fv not in cfs:
            cfs.append(fv)
    if not cfs:
        cfs = [DEFAULT_CFG]
    sps = []
    for v in (steps_list or []):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= iv <= 50 and iv not in sps:
            sps.append(iv)
    if not sps:
        sps = [DEFAULT_STEPS]
    # Axe steps2 (SDXL : 2e passe / detail daemon, node 57). Optionnel : sans valeurs
    # → [None] (la 2e passe retombe sur les steps de la 1re ; Z-Image n'a pas de 2e passe).
    sps2 = []
    for v in (steps2_list or []):
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= iv <= 50 and iv not in sps2:
            sps2.append(iv)
    if not sps2:
        sps2 = [None]
    if not cps or not sts:
        raise ValueError('at least one checkpoint and one strength are required')
    # Pas de plafond : la file est sérielle et l'utilisateur a déjà l'estimation
    # du nombre de cellules / de la durée dans l'UI avant de lancer.
    return [(c, s, a, cf, sp, sp2)
            for c in cps for s in sts for a in asp for cf in cfs for sp in sps for sp2 in sps2]


# --- Workflow build + enqueue -------------------------------------------------
def apply_sdxl_lora_test_settings(workflow, *, base_ckpt, lora_name, strength,
                                  prompt, seed, width, height, cfg=None, steps=None,
                                  steps2=None, batch_size=1, filename_prefix=None,
                                  allowed_bases=None, allowed_loras=None,
                                  detail_amount=None):
    """Configure une cellule de test sur le workflow HQ (SDXL) : checkpoint de base
    (node 1) + LoRA testé via le LoraLoader subtle (node 25) + prompt/seed/dims/steps.
    Le workflow HQ a DEUX passes : `steps` = passe 1 (KSampler node 5) ; `steps2` =
    passe 2 (detail daemon, BasicScheduler node 57). `steps2=None` → la passe 2 retombe
    sur `steps`. Node IDs = ceux d'app/main/routes.py. Mutate en place. Lève ValueError
    si le checkpoint/LoRA n'est pas dans sa whitelist (anti path-injection)."""
    if allowed_bases is not None and base_ckpt not in allowed_bases:
        raise ValueError(f"unknown SDXL checkpoint: {base_ckpt}")
    if allowed_loras is not None and lora_name not in allowed_loras:
        raise ValueError(f"unknown SDXL LoRA: {lora_name}")

    def _set(node_id, key, value):
        n = workflow.get(node_id)
        if isinstance(n, dict) and key in n.get("inputs", {}):
            n["inputs"][key] = value

    # base_ckpt est un BASENAME (get_checkpoint_models dépouille le dossier) ; le loader
    # ComfyUI veut le chemin relatif (ex. 'Biglove\\…') → résoudre, sinon 400.
    _set("1", "ckpt_name", resolve_checkpoint_ckpt_name(base_ckpt))
    _set("25", "lora_name", lora_name)
    _set("25", "strength_model", float(strength))
    _set("25", "strength_clip", float(strength))
    _set("3", "text", prompt)
    _set("5", "seed", int(seed))
    if steps is not None:
        _set("5", "steps", int(steps))          # passe 1 (KSampler)
    # passe 2 (detail daemon, node 57) : steps2 si fourni, sinon retombe sur steps.
    _pass2 = steps2 if steps2 is not None else steps
    if _pass2 is not None:
        _set("57", "steps", int(_pass2))
    if cfg is not None:
        _set("5", "cfg", float(cfg))
    _set("6", "width", int(width))
    _set("6", "height", int(height))
    _set("6", "batch_size", int(batch_size))
    # DetailDaemon (classe DetailDaemonSamplerNode, node scanné par type comme la route
    # generate) : la valeur du slider EST le détail effectif (fade=0). Clamp défensif
    # [0,1] ; None → défaut du workflow conservé. Bande SDXL-safe ≈ 0-0.25.
    if detail_amount is not None:
        try:
            _da = max(0.0, min(1.0, float(detail_amount)))
        except (TypeError, ValueError):
            _da = None
        if _da is not None:
            for _n in workflow.values():
                if (isinstance(_n, dict) and _n.get("class_type") == "DetailDaemonSamplerNode"
                        and "detail_amount" in _n.get("inputs", {})):
                    _n["inputs"]["detail_amount"] = _da
    if filename_prefix is not None:
        _set("9", "filename_prefix", filename_prefix)


def apply_krea_lora_test_settings(workflow, *, lora_name, strength, prompt, seed,
                                  width, height, cfg=None, steps=None, batch_size=1,
                                  filename_prefix=None, allowed_loras=None, extra_loras=None,
                                  rebalance=None, sampler=None, scheduler=None,
                                  weight_dtype=None, enhancer_strength=None):
    """Configure une cellule de test sur le workflow Krea 2 Turbo : le LoRA testé est
    injecté après le UNETLoader (node 20 → KSampler node 26), + prompt/seed/dims/steps/cfg.
    `extra_loras` = LoRA « always-on » (style/utilitaire) chaînés EN PLUS dans le même
    maillon (appliqués tels quels à cette cellule, hors batch). Krea est MONO-passe (pas
    de steps2).

    `rebalance` (node 30, NSFW/texture rebalance) - même sémantique que la génération
    (routes.py) : None = on NE touche PAS le node, défaut ON du workflow ; ≤1.0 = OFF
    (multiplier=1.0 + per_layer_weights neutres → passthrough SFW) ; >1.0 = ON à cette
    force (clampé 1..8). Mutate en place. Lève ValueError si le LoRA testé n'est pas dans
    sa whitelist (anti path-injection)."""
    if allowed_loras is not None and lora_name not in allowed_loras:
        raise ValueError(f"unknown Krea LoRA: {lora_name}")

    def _set(node_id, key, value):
        n = workflow.get(node_id)
        if isinstance(n, dict) and key in n.get("inputs", {}):
            n["inputs"][key] = value

    _set("23", "text", prompt)                    # prompt (CLIPTextEncode Krea)
    _set("25", "width", int(width))
    _set("25", "height", int(height))
    _set("25", "batch_size", int(batch_size))
    _set("26", "seed", int(seed))
    if steps is not None:
        _set("26", "steps", max(1, min(50, int(steps))))
    if cfg is not None:
        _set("26", "cfg", max(1.0, min(10.0, float(cfg))))
    # Sampler / scheduler (node 26) + précision UNET (node 20) - validés contre les
    # MÊMES whitelists que la génération (anti-injection ; hors liste = ignoré).
    if sampler in KREA_ALLOWED_SAMPLERS:
        _set("26", "sampler_name", sampler)
    if scheduler in KREA_ALLOWED_SCHEDULERS:
        _set("26", "scheduler", scheduler)
    if weight_dtype in KREA_ALLOWED_WEIGHT_DTYPES:
        _set("20", "weight_dtype", weight_dtype)
    if filename_prefix is not None:
        _set("28", "filename_prefix", filename_prefix)
    # Node 30 (Krea2RebalanceConditioning) : reweight des taps de conditioning Qwen3-VL.
    # ON (>1) relève les taps filtrés-sécurité → sortie non censurée + peau moins « plastique » ;
    # OFF (≤1) = passthrough identité (SFW). None = laisser le défaut du workflow (ON 4.0).
    if rebalance is not None and isinstance(workflow.get("30"), dict):
        m = max(1.0, min(8.0, float(rebalance)))
        if m <= 1.0:
            _set("30", "multiplier", 1.0)
            _set("30", "per_layer_weights", "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0")
        else:
            _set("30", "multiplier", m)
    # LoRA testé + always-on : une seule chaîne node 20 → 26 (même mécanique que la
    # génération Krea). `allowed` contient TOUT le pool Krea (les always-on y sont).
    from ..utils.comfyui import inject_krea_loras
    requested = [{"filename": lora_name, "strength": float(strength)}]
    for e in (extra_loras or []):
        fn = str((e or {}).get("filename") or "")
        if not fn:
            continue
        try:
            st = float(e.get("strength", 1.0))
        except (TypeError, ValueError):
            st = 1.0
        requested.append({"filename": fn, "strength": st})
    allowed = set(allowed_loras) if allowed_loras is not None else {r["filename"] for r in requested}
    inject_krea_loras(workflow, requested, allowed=allowed)
    # Krea2T-Enhancer (patcher texte-adhérence) injecté APRÈS les LoRA (wire-aware :
    # se branche sur ce qui alimente KSampler.model). enhancer_strength None = OFF ;
    # sinon ON à cette force (clampée 0..2 dans inject_krea2t_enhancer).
    if enhancer_strength is not None:
        inject_krea2t_enhancer(workflow, True, enhancer_strength)


def _build_cell_workflow(user_id, checkpoint, strength, prompt, seed, z_model,
                         allowed_loras, width=TEST_WIDTH, height=TEST_HEIGHT,
                         cfg=None, steps=None, steps2=None, dataset_id=None, train_type='zimage',
                         extra_loras=None, rebalance=None, negative=None, sampler=None,
                         scheduler=None, weight_dtype=None, enhancer_strength=None,
                         detail_amount=None, trigger_word=None):
    """Load the ZTurbo (Z-Image) / HQ (SDXL) / Krea workflow and configure one grid cell.
    `extra_loras` = LoRA « always-on » (style/utilitaire) appliqués à CETTE cellule en plus
    du checkpoint testé (hors batch). `rebalance` = node 30 NSFW/texture (Krea uniquement,
    None ailleurs). Raises ValueError if the workflow file is unloadable.

    Le filename_prefix inclut le dataset_id ET un uuid court par cellule : sans
    ça, le compteur ComfyUI (qui repart de 0 à chaque restart) produisait des
    noms identiques entre datasets (`{uid}_LoraTest_00022_`) → collisions de
    cache navigateur et confusion visuelle entre LoRA (ex. images eva6938 vues
    dans le studio d'un autre LoRA). L'uuid garantit l'unicité même au sein d'un
    dataset (re-runs après restart ComfyUI)."""
    # Trigger word auto-injecté ICI (montage seul) - le prompt reste brut en base.
    prompt = _prompt_with_trigger(prompt, trigger_word)
    ds_tag = f"d{dataset_id}_" if dataset_id is not None else ""
    fname = f"{user_id}_{ds_tag}LoraTest_{uuid.uuid4().hex[:8]}"
    extra_loras = extra_loras or []
    if (train_type or 'zimage').lower() == 'sdxl':
        workflow = load_workflow_local(str(WORKFLOW_HQ_PATH))
        if not workflow:
            raise ValueError('HQ workflow not found/unreadable')
        from ..utils.comfyui import get_checkpoint_models, inject_sdxl_loras
        allowed_bases = {m.get('name') for m in get_checkpoint_models() if m.get('name')}
        allowed_sdxl_loras = {l['filename'] for l in get_sdxl_loras()}
        # Comme la génération SDXL normale : régler sampler/scheduler/cfg ET surtout
        # toggler le LoRA DMD2 (ON pour checkpoints DMD-distillés type bigLove/mop, OFF
        # pour SDXL full) selon le modèle de base. Sans ça, sortie cassée. Appliqué AVANT
        # l'injection de test pour que la cfg/les steps du studio (axes) gagnent ensuite.
        apply_optimal_sampler_params(workflow, z_model)
        apply_sdxl_lora_test_settings(
            workflow, base_ckpt=z_model, lora_name=checkpoint, strength=strength,
            prompt=prompt, seed=seed, width=width, height=height, cfg=cfg, steps=steps,
            steps2=steps2, batch_size=1, filename_prefix=fname,
            allowed_bases=allowed_bases, allowed_loras=allowed_sdxl_loras,
            detail_amount=detail_amount,
        )
        if extra_loras:  # always-on chaînés après le Style LoRA (node 25)
            inject_sdxl_loras(workflow, extra_loras, {e['filename'] for e in extra_loras})
        return workflow
    if (train_type or 'zimage').lower() == 'krea':
        workflow = load_workflow_local(str(WORKFLOW_KREA_TURBO_PATH))
        if not workflow:
            raise ValueError('Krea workflow not found/unreadable')
        allowed_krea = {l['filename'] for l in get_krea_loras()}
        apply_krea_lora_test_settings(
            workflow, lora_name=checkpoint, strength=strength, prompt=prompt,
            seed=seed, width=width, height=height, cfg=cfg, steps=steps,
            batch_size=1, filename_prefix=fname, allowed_loras=allowed_krea,
            extra_loras=extra_loras, rebalance=rebalance,
            sampler=sampler, scheduler=scheduler, weight_dtype=weight_dtype,
            enhancer_strength=enhancer_strength,
        )
        return workflow
    workflow = load_workflow_local(str(WORKFLOW_ZTURBO_PATH))
    if not workflow:
        raise ValueError('ZTurbo workflow not found/unreadable')
    apply_zimage_settings(
        workflow,
        z_model=z_model,
        z_loras=[{'filename': checkpoint, 'strength': strength}] + list(extra_loras),
        prompt=prompt,
        negative=negative,
        seed=seed,
        width=width, height=height, batch_size=1,
        z_cfg=cfg, z_steps=steps,
        filename_prefix=fname,
        # always-on inclus dans la whitelist (sinon inject_zimage_loras les filtrerait).
        allowed_loras=(set(allowed_loras) | {e['filename'] for e in extra_loras}) if extra_loras else allowed_loras,
    )
    return workflow


def _enqueue_cell(user_id, dataset_id, workflow, prompt) -> str:
    """Enqueue one cell as a normal (serialized) image job. Free: never
    debited - the failure path in job_queue skips the refund for
    is_lora_test jobs exactly like is_dataset (no credit minting)."""
    job_id = str(uuid.uuid4())
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=prompt, job_id=job_id,
                          metadata={'model_name': 'zimage_lora_test',
                                    'is_lora_test': True,
                                    'dataset_id': dataset_id})
    return job_id


def _sanitize_gen_knobs(run_family, *, negative=None, sampler=None, scheduler=None,
                        weight_dtype=None, enhancer=None, enhancer_strength=None,
                        detail_amount=None, resolution_tier=None, init_image=None,
                        denoise=None) -> dict:
    """Normalise + valide les réglages de génération GLOBAUX d'un run (parité Generate),
    filtrés PAR FAMILLE (un sampler Krea n'a aucun sens en Z-Image). Renvoie un dict prêt
    à la fois à persister sur LoraTestImage ET à passer à `_build_cell_workflow`. Chaque
    valeur hors périmètre/whitelist retombe à None (le workflow garde alors son défaut).

    Encodages : `enhancer_strength` NULL = Krea2T OFF (sinon force ON, clampée 0..2, défaut
    1.0 quand `enhancer` truthy sans force) ; `negative` vide → None ; `denoise` clampé
    0.05..1.0 ; `resolution_tier` doit être dans RESOLUTION_TIERS."""
    fam = (run_family or 'zimage').lower()
    neg = ((negative or '').strip() or None) if fam == 'zimage' else None
    smp = sampler if (fam == 'krea' and sampler in KREA_ALLOWED_SAMPLERS) else None
    sch = scheduler if (fam == 'krea' and scheduler in KREA_ALLOWED_SCHEDULERS) else None
    wdt = weight_dtype if (fam == 'krea' and weight_dtype in KREA_ALLOWED_WEIGHT_DTYPES) else None
    enh = None
    if fam == 'krea' and enhancer:
        try:
            enh = max(0.0, min(2.0, float(enhancer_strength if enhancer_strength is not None else 1.0)))
        except (TypeError, ValueError):
            enh = 1.0
    dta = None
    if fam == 'sdxl' and detail_amount is not None:
        try:
            dta = max(0.0, min(1.0, float(detail_amount)))
        except (TypeError, ValueError):
            dta = None
    tier = resolution_tier if resolution_tier in RESOLUTION_TIERS else None
    den = None
    if fam == 'krea' and denoise is not None:
        try:
            den = max(0.05, min(1.0, float(denoise)))
        except (TypeError, ValueError):
            den = None
    ini = ((init_image or '').strip() or None) if fam == 'krea' else None
    return {'negative': neg, 'sampler': smp, 'scheduler': sch, 'weight_dtype': wdt,
            'enhancer_strength': enh, 'detail_amount': dta, 'resolution_tier': tier,
            'init_image': ini, 'denoise': den}


# --- Run lifecycle -----------------------------------------------------------
def _batch_lora_axis(batch_loras, run_family) -> list:
    """Valide la liste « ⚖ batch axis » (mêmes règles anti path-injection que les
    always-on) et renvoie l'axe de test [None, {filename,strength}, …] - None =
    la cellule de RÉFÉRENCE sans le LoRA. Dédupé, borné à 4 LoRA (coût GPU)."""
    perm_allowed = {c['filename'] for c in permanent_lora_candidates(run_family)}
    entries = []
    for e in (batch_loras or []):
        fn = str((e or {}).get('filename') or '')
        if fn not in perm_allowed or any(x['filename'] == fn for x in entries):
            continue
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        entries.append({'filename': fn, 'strength': st})
    return [None] + entries[:4] if entries else [None]


def _batch_lora_label(row):
    """Nom lisible du LoRA « batch » d'une cellule (entrée batch:true de son JSON
    extra_loras), ou None - badge de la grille/lightbox."""
    try:
        for e in json.loads(row.extra_loras or '[]'):
            if isinstance(e, dict) and e.get('batch'):
                return _basename(e.get('filename', '')).rsplit('.', 1)[0]
    except (ValueError, TypeError):
        pass
    return None


def create_run(user_id, dataset_id, checkpoints, strengths, seed=None, prompt=None, z_model=None, z_models=None, aspects=None, cfgs=None, steps_list=None, steps2_list=None, count=1, family=None, permanent_loras=None, batch_loras=None, rebalance=None, rebalance_strength=None, negative=None, sampler=None, scheduler=None, weight_dtype=None, enhancer=None, enhancer_strength=None, detail_amount=None, resolution_tier=None, init_image=None, denoise=None) -> dict:
    """Validate + materialize the grid and enqueue every cell.

    Each row is committed BEFORE its enqueue (anti-orphan rule of the dataset
    fan-out); an enqueue failure marks that row 'failed' and re-raises -
    already-enqueued cells keep their jobs. Returns {'created', 'seed', 'count', 'ids'}."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not (ds.trigger_word or '').strip():
        raise ValueError('trigger word is required')

    reason = gpu_busy_reason()
    if reason:
        raise ValueError(reason)
    if _active_run_count(dataset_id):
        raise ValueError('a test run is already in progress on this dataset - '
                         'wait for it to finish or cancel')

    # La FAMILLE (pipeline) du run est dérivée des checkpoints sélectionnés : ils
    # vivent tous dans le même dossier loras/<famille> (le frontend ne propose qu'une
    # famille à la fois via le sélecteur). On ne peut pas mélanger ZIT/SDXL/Krea dans
    # un run (bases + workflow différents). `family` sert de repli si les checkpoints
    # n'ont pas de préfixe de dossier (anciens noms renommés).
    cps_in = [c for c in (checkpoints or []) if isinstance(c, str) and c.strip()]
    if not cps_in:
        raise ValueError('at least one checkpoint is required')
    fams = {family_of_lora(c) for c in cps_in}
    fams.discard(None)
    if len(fams) > 1:
        raise ValueError('a test run cannot mix multiple families (ZIT/SDXL/Krea)')
    run_family = (next(iter(fams), None) or family or getattr(ds, 'train_type', None) or 'zimage').lower()

    allowed = {c['filename'] for c in list_test_checkpoints(ds, run_family)}
    unknown = [c for c in cps_in if c not in allowed]
    if unknown:
        raise ValueError(f'unknown checkpoint(s) for this dataset: {unknown}')

    # LoRA « always-on » (style/utilitaire) appliqués à CHAQUE cellule (hors batch).
    # Validés contre les candidats de la famille (anti path-injection) + strength clamp.
    perm_allowed = {c['filename'] for c in permanent_lora_candidates(run_family)}
    extra_loras = []
    for e in (permanent_loras or []):
        fn = str((e or {}).get('filename') or '')
        if fn not in perm_allowed:
            continue
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        extra_loras.append({'filename': fn, 'strength': st})
    # Axe « ⚖ batch » : chaque config tourne une fois SANS puis une fois AVEC
    # chaque LoRA coché batch (les always-on ci-dessus s'appliquent partout).
    batch_axis = _batch_lora_axis(batch_loras, run_family)

    # NSFW / texture rebalance (node 30) - Krea UNIQUEMENT (les autres familles n'ont pas
    # ce node). Encodage en UN seul FLOAT, persisté → resume fidèle :
    #   rebalance=False        → 1.0 (OFF, passthrough SFW)
    #   rebalance=True         → rebalance_strength clampé 1..8 (ON, défaut 4.0)
    #   None / non-Krea        → None (on laisse le défaut ON du workflow, node intact)
    cell_rebalance = None
    if run_family == 'krea' and rebalance is not None:
        if rebalance:
            try:
                cell_rebalance = max(1.0, min(8.0, float(rebalance_strength if rebalance_strength is not None else 4.0)))
            except (TypeError, ValueError):
                cell_rebalance = 4.0
        else:
            cell_rebalance = 1.0

    # Réglages de génération GLOBAUX du run (parité Generate), validés + gatés par famille.
    knobs = _sanitize_gen_knobs(
        run_family, negative=negative, sampler=sampler, scheduler=scheduler,
        weight_dtype=weight_dtype, enhancer=enhancer, enhancer_strength=enhancer_strength,
        detail_amount=detail_amount, resolution_tier=resolution_tier,
        init_image=init_image, denoise=denoise)

    cells = build_matrix(checkpoints, strengths, aspects, cfgs, steps_list, steps2_list)

    # Pool de bases selon la FAMILLE : SDXL → checkpoints SDXL (de Generate), Krea →
    # base fixe (UNET du workflow, pas d'axe de base), Z-Image → modèles Z-Image.
    if run_family == 'sdxl':
        models = [m['filename'] for m in list_sdxl_base_models()]
        if not models:
            raise ValueError('no SDXL checkpoint available')
    elif run_family == 'krea':
        models = [None]  # Krea : UNET fixe → un seul « modèle » (z_model=None)
    else:
        models = get_zimage_models()
        if not models:
            raise ValueError('no Z-Image model available')
    # Modèle(s) de base - AXE de balayage optionnel (validés contre la whitelist).
    # z_models (liste) prioritaire ; sinon z_model unique (rétrocompat) ; sinon le 1er.
    _req_models = list(z_models) if z_models else ([z_model] if z_model else [])
    valid_models = [m for m in _req_models if m in models] or [models[0]]

    try:
        seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
    except (TypeError, ValueError):
        raise ValueError(f'invalid seed: {seed!r}')

    # Nombre de générations par config (batch) : N seeds DISTINCTS, PARTAGÉS entre
    # toutes les configs (comparaison équitable à seeds identiques). Borné 1..4.
    try:
        count = max(1, min(int(count or 1), 4))
    except (TypeError, ValueError):
        count = 1
    _MAX = 2**31 - 1
    seeds = [1 + ((seed + i - 1) % _MAX) for i in range(count)]  # distincts, dans [1, 2^31-1]

    # Prompt custom optionnel ; sinon prompt d'identité par défaut (trigger).
    prompt = (prompt or '').strip() or identity_prompt(ds)
    ids = []
    for zm in valid_models:                       # AXE modèle de base (multi-sélection)
        for checkpoint, strength, cell_aspect, cell_cfg, cell_steps, cell_steps2 in cells:
            # Format/CFG/steps (1 et 2) testés comme axes à part entière (multi-sélection).
            width, height = _aspect_dims(cell_aspect, run_family, knobs['resolution_tier'])
            for batch_lora in batch_axis:  # AXE ⚖ batch : sans, puis avec chaque LoRA coché
              row_extra = extra_loras + ([{**batch_lora, 'batch': True}] if batch_lora else [])
              wf_extra = extra_loras + ([batch_lora] if batch_lora else [])
              cell_extra_json = json.dumps(row_extra) if row_extra else None
              for cell_seed in seeds:  # N images par config (seeds différents), bande dans la cellule
                img = LoraTestImage(dataset_id=dataset_id, checkpoint=checkpoint,
                                    strength=strength, seed=cell_seed, run_seed=seed,
                                    status='pending', z_model=zm, aspect=cell_aspect,
                                    prompt=prompt, cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                    extra_loras=cell_extra_json, krea_rebalance=cell_rebalance,
                                    negative=knobs['negative'], sampler=knobs['sampler'],
                                    scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                    enhancer_strength=knobs['enhancer_strength'],
                                    detail_amount=knobs['detail_amount'],
                                    resolution_tier=knobs['resolution_tier'],
                                    init_image=knobs['init_image'], denoise=knobs['denoise'])
                db.session.add(img)
                db.session.commit()
                try:
                    workflow = _build_cell_workflow(user_id, checkpoint, strength,
                                                    prompt, cell_seed, zm, allowed,
                                                    width=width, height=height,
                                                    cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                                    dataset_id=dataset_id,
                                                    train_type=run_family, extra_loras=wf_extra,
                                                    rebalance=cell_rebalance,
                                                    negative=knobs['negative'], sampler=knobs['sampler'],
                                                    scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                                    enhancer_strength=knobs['enhancer_strength'],
                                                    detail_amount=knobs['detail_amount'],
                                                    trigger_word=ds.trigger_word)
                    job_id = _enqueue_cell(user_id, dataset_id, workflow, prompt)
                except Exception:
                    img.status = 'failed'
                    db.session.commit()
                    raise
                img.job_id = job_id
                db.session.commit()
                ids.append(img.id)
    logger.info(f"lora-test: run dataset {dataset_id} -> {len(ids)} cellule(s) "
                f"({len(valid_models)} modèle(s)), base seed {seed} ×{count}")
    return {'created': len(ids), 'seed': seed, 'count': count, 'ids': ids}


def create_comparison_run(user_id, selections, strengths, seed=None, prompt=None,
                          z_model=None, aspects=None, cfgs=None, steps_list=None, steps2_list=None,
                          count=1, permanent_loras=None, batch_loras=None, rebalance=None, rebalance_strength=None,
                          negative=None, sampler=None, scheduler=None, weight_dtype=None,
                          enhancer=None, enhancer_strength=None, detail_amount=None,
                          resolution_tier=None, init_image=None, denoise=None) -> dict:
    """Lance UN run de comparaison sur plusieurs LoRA. `selections` =
    [{dataset_id, checkpoint}]. Toutes les cellules partagent un run_id + le seed
    (équité). Le prompt : `prompt` commun si fourni, sinon l'identity_prompt du
    dataset de CHAQUE cellule (chaque LoRA a son trigger). 1 selection => run mono-LoRA.

    Parité Generate (2026-07-01) : always-on LoRA, rebalance Krea, steps2 SDXL et les
    réglages globaux (négatif/sampler/scheduler/precision/enhancer/detail/tier) sont
    partagés par TOUTES les cellules du run (gatés + validés par famille via _sanitize_gen_knobs)."""
    if not selections:
        raise ValueError('no LoRA selected')
    reason = gpu_busy_reason()
    if reason:
        raise ValueError(reason)
    if _active_run_count():
        raise ValueError('a test run is already in progress - wait for it to finish or cancel')
    # La FAMILLE du run est dérivée du DOSSIER des checkpoints (family_of_lora), PAS du
    # scalaire `ds.train_type` (un dataset est multi-famille). Un run = une seule famille
    # (bases + workflow différents). On résout la base AVANT la boucle, selon la famille.
    fams = {family_of_lora(str(sel.get('checkpoint') or '')) for sel in (selections or [])}
    fams.discard(None)
    if len(fams) > 1:
        raise ValueError('a test run cannot mix multiple families (ZIT/SDXL/Krea)')
    run_type = (next(iter(fams), None) or 'zimage').lower()
    if run_type == 'sdxl':
        models = [m['filename'] for m in list_sdxl_base_models()]
        if not models:
            raise ValueError('no SDXL checkpoint available')
    elif run_type == 'krea':
        models = [None]  # Krea : UNET fixe (node 20) → pas d'axe de base (z_model=None)
    else:
        models = get_zimage_models()
        if not models:
            raise ValueError('no Z-Image model available')
    z_model = z_model if (z_model and z_model in models) else models[0]
    try:
        seed = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
    except (TypeError, ValueError):
        raise ValueError(f'invalid seed: {seed!r}')
    try:
        count = max(1, min(int(count or 1), 4))
    except (TypeError, ValueError):
        count = 1
    _MAX = 2**31 - 1
    seeds = [1 + ((seed + i - 1) % _MAX) for i in range(count)]
    common_prompt = (prompt or '').strip() or None
    # LoRA « always-on » (style/utilitaire) validés contre la famille (anti path-injection),
    # appliqués à CHAQUE cellule - même mécanique que create_run.
    perm_allowed = {c['filename'] for c in permanent_lora_candidates(run_type)}
    extra_loras = []
    for e in (permanent_loras or []):
        fn = str((e or {}).get('filename') or '')
        if fn not in perm_allowed:
            continue
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        extra_loras.append({'filename': fn, 'strength': st})
    # Axe « ⚖ batch » : chaque config tourne une fois SANS puis une fois AVEC
    # chaque LoRA coché batch (même mécanique que create_run).
    batch_axis = _batch_lora_axis(batch_loras, run_type)
    # Rebalance Krea (node 30) - même encodage float que create_run (None=défaut, ≤1=OFF, >1=ON@force).
    cell_rebalance = None
    if run_type == 'krea' and rebalance is not None:
        if rebalance:
            try:
                cell_rebalance = max(1.0, min(8.0, float(rebalance_strength if rebalance_strength is not None else 4.0)))
            except (TypeError, ValueError):
                cell_rebalance = 4.0
        else:
            cell_rebalance = 1.0
    # Réglages de génération GLOBAUX (parité Generate), validés + gatés par famille.
    knobs = _sanitize_gen_knobs(
        run_type, negative=negative, sampler=sampler, scheduler=scheduler,
        weight_dtype=weight_dtype, enhancer=enhancer, enhancer_strength=enhancer_strength,
        detail_amount=detail_amount, resolution_tier=resolution_tier,
        init_image=init_image, denoise=denoise)

    run_id = uuid.uuid4().hex
    ids = []
    for sel in selections:
        ds = fds.get_dataset(user_id, sel.get('dataset_id'))
        if not ds:
            raise ValueError(f"dataset {sel.get('dataset_id')} not found")
        allowed = {c['filename'] for c in list_test_checkpoints(ds, run_type)}
        checkpoint = sel.get('checkpoint')
        if checkpoint not in allowed:
            raise ValueError(f'unknown checkpoint for {ds.name}: {checkpoint}')
        cell_prompt = common_prompt or identity_prompt(ds)
        cells = build_matrix([checkpoint], strengths, aspects, cfgs, steps_list, steps2_list)
        for cp, strength, cell_aspect, cell_cfg, cell_steps, cell_steps2 in cells:
            width, height = _aspect_dims(cell_aspect, run_type, knobs['resolution_tier'])
            for batch_lora in batch_axis:  # AXE ⚖ batch : sans, puis avec chaque LoRA coché
              row_extra = extra_loras + ([{**batch_lora, 'batch': True}] if batch_lora else [])
              wf_extra = extra_loras + ([batch_lora] if batch_lora else [])
              cell_extra_json = json.dumps(row_extra) if row_extra else None
              for cell_seed in seeds:
                img = LoraTestImage(dataset_id=ds.id, checkpoint=cp, strength=strength,
                                    seed=cell_seed, run_seed=seed, run_id=run_id,
                                    status='pending', z_model=z_model, aspect=cell_aspect,
                                    prompt=cell_prompt, cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                    extra_loras=cell_extra_json, krea_rebalance=cell_rebalance,
                                    negative=knobs['negative'], sampler=knobs['sampler'],
                                    scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                    enhancer_strength=knobs['enhancer_strength'],
                                    detail_amount=knobs['detail_amount'],
                                    resolution_tier=knobs['resolution_tier'],
                                    init_image=knobs['init_image'], denoise=knobs['denoise'])
                db.session.add(img); db.session.commit()
                try:
                    workflow = _build_cell_workflow(user_id, cp, strength, cell_prompt,
                                                    cell_seed, z_model, allowed, width=width,
                                                    height=height, cfg=cell_cfg, steps=cell_steps,
                                                    steps2=cell_steps2, dataset_id=ds.id,
                                                    train_type=run_type, extra_loras=wf_extra,
                                                    rebalance=cell_rebalance,
                                                    negative=knobs['negative'], sampler=knobs['sampler'],
                                                    scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                                    enhancer_strength=knobs['enhancer_strength'],
                                                    detail_amount=knobs['detail_amount'],
                                                    trigger_word=ds.trigger_word)
                    job_id = _enqueue_cell(user_id, ds.id, workflow, cell_prompt)
                except Exception:
                    img.status = 'failed'; db.session.commit(); raise
                img.job_id = job_id; db.session.commit(); ids.append(img.id)
    logger.info(f"lora-test: comparison run {run_id} -> {len(ids)} cellule(s), {len(selections)} LoRA, seed {seed}")
    return {'created': len(ids), 'seed': seed, 'count': count, 'run_id': run_id, 'ids': ids}


def _run_owned(user_id, run_id) -> bool:
    """Single-user app: every run belongs to the local user - no cross-user
    ownership DB to consult (SRC checked every cell's dataset against
    `user_id`)."""
    return True


def cancel_run(user_id, dataset_id=None, run_id=None) -> int:
    """Stoppe les cellules en vol : annule les jobs de queue et marque les
    cellules 'cancelled' (au lieu de les supprimer) pour pouvoir REPRENDRE le
    run plus tard avec leurs réglages exacts (prompt/seed/modèle/format).
    Retourne le nombre stoppé.

    Cible : si `run_id` est fourni, opère sur ce run ; sinon, comportement
    historique par `dataset_id`."""
    if run_id is not None:
        if not _run_owned(user_id, run_id):
            return 0
        rows = (LoraTestImage.query
                .filter_by(run_id=run_id, status='pending')
                .filter(LoraTestImage.filename.is_(None)).all())
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            return 0
        rows = (LoraTestImage.query
                .filter_by(dataset_id=dataset_id, status='pending')
                .filter(LoraTestImage.filename.is_(None)).all())
    n = 0
    for img in rows:
        if img.job_id:
            try:
                queue_manager.cancel_job(img.job_id, str(user_id), 'image')
            except Exception:
                pass
        img.status = 'cancelled'
        img.job_id = None
        n += 1
    db.session.commit()
    return n


def resume_run(user_id, dataset_id=None, run_id=None) -> dict:
    """Reprend un run stoppé : ré-enfile toutes les cellules 'cancelled'/'failed'
    avec LEURS réglages stockés (même prompt/seed/modèle/format/strength). C'est
    le « relancer l'ancien run avec le même prompt » demandé.

    Cible : si `run_id` est fourni, ré-enfile ce run ; sinon, comportement
    historique par `dataset_id`."""
    if run_id is not None:
        if not _run_owned(user_id, run_id):
            raise ValueError('run not found')
        reason = gpu_busy_reason()
        if reason:
            raise ValueError(reason)
        if _active_run_count():
            raise ValueError('a test run is already in progress')
        rows = (LoraTestImage.query.filter_by(run_id=run_id)
                .filter(LoraTestImage.status.in_(['cancelled', 'failed'])).all())
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            raise ValueError('dataset not found')
        reason = gpu_busy_reason()
        if reason:
            raise ValueError(reason)
        if _active_run_count(dataset_id):
            raise ValueError('a test run is already in progress')
        rows = (LoraTestImage.query.filter_by(dataset_id=dataset_id)
                .filter(LoraTestImage.status.in_(['cancelled', 'failed'])).all())
    if not rows:
        raise ValueError('no cell to resume')
    # Le run_id peut couvrir plusieurs datasets (run multi-LoRA) → on résout le
    # dataset PAR cellule, avec un cache. La FAMILLE de chaque cellule est déduite du
    # dossier de son checkpoint (sdxl/krea/z image) - pas du train_type du dataset, qui
    # peut différer quand le même dataset a été entraîné sous plusieurs pipelines. La
    # whitelist est donc cachée par (dataset, famille).
    ds_cache, allowed_cache = {}, {}
    _sdxl_bases = None  # liste des bases SDXL, calculée à la demande (cache)

    def _ds(did):
        if did not in ds_cache:
            ds_cache[did] = fds.get_dataset(user_id, did)
        return ds_cache[did]

    def _allowed(did, fam):
        key = (did, fam)
        if key not in allowed_cache:
            d = _ds(did)
            allowed_cache[key] = {c['filename'] for c in list_test_checkpoints(d, fam)} if d else set()
        return allowed_cache[key]
    n = 0
    for img in rows:
        cell_ds = _ds(img.dataset_id)
        # Famille = dossier du checkpoint (repli train_type) → whitelist + base + dims + workflow.
        cell_family = (family_of_lora(img.checkpoint)
                       or getattr(cell_ds, 'train_type', None) or 'zimage').lower()
        allowed = _allowed(img.dataset_id, cell_family)
        if not cell_ds or img.checkpoint not in allowed:
            continue  # dataset/checkpoint disparu → on saute
        # Pool de bases selon la famille de CETTE cellule (SDXL → bases SDXL ; Krea →
        # base fixe ; sinon Z-Image), sinon un resume SDXL retomberait sur une base Z-Image.
        if cell_family == 'sdxl':
            if _sdxl_bases is None:
                _sdxl_bases = [m['filename'] for m in list_sdxl_base_models()]
            cell_models = _sdxl_bases
        elif cell_family == 'krea':
            cell_models = [None]
        else:
            cell_models = get_zimage_models()
        z_model = (img.z_model if (img.z_model and img.z_model in cell_models)
                   else (cell_models[0] if cell_models else None))
        aspect = img.aspect if img.aspect in TEST_ASPECTS else DEFAULT_ASPECT
        # Palier de résolution persisté → mêmes dims qu'au 1er run (sinon table fixe).
        width, height = _aspect_dims(aspect, cell_family, getattr(img, 'resolution_tier', None))
        prompt = (img.prompt or '').strip() or identity_prompt(cell_ds)
        seed = img.seed or random.randint(1, 2**31 - 1)
        # LoRA always-on stockés sur la cellule → réappliqués à l'identique au resume.
        try:
            cell_extra = json.loads(img.extra_loras) if img.extra_loras else None
        except (json.JSONDecodeError, TypeError):
            cell_extra = None
        try:
            # Tous les réglages globaux (parité Generate) relus depuis la cellule → resume fidèle.
            workflow = _build_cell_workflow(user_id, img.checkpoint, img.strength,
                                            prompt, seed, z_model, allowed,
                                            width=width, height=height,
                                            cfg=img.cfg, steps=img.steps, steps2=img.steps2,
                                            dataset_id=img.dataset_id,
                                            train_type=cell_family, extra_loras=cell_extra,
                                            rebalance=img.krea_rebalance,
                                            negative=getattr(img, 'negative', None),
                                            sampler=getattr(img, 'sampler', None),
                                            scheduler=getattr(img, 'scheduler', None),
                                            weight_dtype=getattr(img, 'weight_dtype', None),
                                            enhancer_strength=getattr(img, 'enhancer_strength', None),
                                            detail_amount=getattr(img, 'detail_amount', None),
                                            trigger_word=getattr(cell_ds, 'trigger_word', None))
            job_id = _enqueue_cell(user_id, img.dataset_id, workflow, prompt)
            img.status = 'pending'
            img.filename = None
            img.job_id = job_id
            img.seed = seed
            db.session.commit()
            n += 1
        except Exception:
            img.status = 'failed'
            db.session.commit()
    return {'resumed': n}


# --- Completion linking (called from job_queue) --------------------------------
def _cleanup_output_file(filename, failed):
    """Supprime de OUTPUT_DIR un fichier de sortie orphelin (complétion d'un job
    dont la ligne n'est plus valable) - best-effort."""
    if failed or not filename:
        return
    out_dir = _comfy_output_dir()
    if not out_dir:
        return
    try:
        p = os.path.join(out_dir, filename)
        if os.path.isfile(p):
            os.remove(p)
    except OSError:
        pass


def link_completed_test_image(job_id, filename, failed=False):
    """Attach a finished studio job to its LoraTestImage row.

    Mirror of link_completed_dataset_image: runs in the queue monitor thread
    whose SQLAlchemy session may hold a STALE read snapshot - if the first
    lookup misses, rollback (end the transaction) and re-read on a fresh
    snapshot before concluding the row doesn't exist."""
    img = LoraTestImage.query.filter_by(job_id=job_id).first()
    if img is None:
        db.session.rollback()  # drop the stale read snapshot, then re-read
        img = LoraTestImage.query.filter_by(job_id=job_id).first()
    if img is None:
        logger.warning(f"lora-test link: no LoraTestImage row for job {job_id}")
        _cleanup_output_file(filename, failed)  # job sans ligne (annulé/repris) → orphelin
        return
    # Ne finaliser que les cellules ENCORE en attente : une complétion tardive d'un
    # job dont la ligne a été annulée/reprise (nouveau job_id, statut ≠ pending) ne
    # doit pas écraser le bon run - on jette son fichier au lieu de le déplacer.
    if not failed and img.status != 'pending':
        logger.info(f"lora-test link: ligne {img.id} déjà {img.status} pour job {job_id} - ignoré")
        _cleanup_output_file(filename, failed)
        return
    if failed:
        img.status = 'failed'
    else:
        img.filename = filename
        img.status = 'done'
        # Move the completed file from shared OUTPUT_DIR to the per-dataset dir
        # (served by /api/dataset/<id>/img/<filename>, cleaned with the dataset).
        out_dir = _comfy_output_dir()
        if out_dir:
            src = os.path.join(out_dir, filename)
            dst = os.path.join(fds._dataset_dir(img.dataset_id), filename)
            if os.path.exists(src):
                shutil.move(src, dst)
            elif not os.path.exists(dst):
                logger.warning(f"lora-test link: file not found at src={src} or dst={dst}")
        else:
            logger.warning(f"lora-test link: ComfyUI output dir not configured - cannot move {filename}")
    db.session.commit()


# --- Rating + best settings ---------------------------------------------------
def _owned_test_image(user_id, image_id):
    """Single-user app: no cross-user ownership check (SRC compared the
    image's dataset.user_id against `user_id`) - just the row lookup."""
    return db.session.get(LoraTestImage, image_id)


def rate_image(user_id, image_id, rating) -> bool:
    if rating not in (1, -1, 0):
        return False
    img = _owned_test_image(user_id, image_id)
    if not img:
        return False
    img.rating = rating
    db.session.commit()
    return True


def _model_label(z_model):
    return _basename(z_model).rsplit('.', 1)[0] if z_model else None


# En deçà de ce nombre de votes, un score est statistiquement fragile → drapeau
# « échantillon faible » dans l'UI (le tri reste Wilson, qui pénalise déjà les
# petits échantillons ; ce flag ne sert qu'à AVERTIR l'œil).
LOW_CONFIDENCE_MIN = 3


def cell_scores(dataset_id, family=None) -> list[dict]:
    """Score par CONFIG = (checkpoint, strength, format, modèle, cfg, steps),
    agrégé sur toutes les images de cette config (cross-runs). Le modèle fait
    partie de la clé : deux modèles sur la même case ne fusionnent plus.

    `family` (optionnel) restreint aux cellules de cette pipeline - déduite du
    dossier du checkpoint - pour que scores/best ne mélangent pas ZIT/SDXL/Krea d'un
    même dataset entraîné sous plusieurs familles. Un checkpoint sans préfixe de
    dossier (ancien nom) compte comme 'zimage'.

    `score` (👍−👎) reste exposé pour l'affichage, mais le TRI se fait sur `rank`
    = borne basse de Wilson sur le taux de 👍 (taux × confiance) - pas sur le
    compte brut, qui biaisait vers les configs simplement plus testées. Tri
    best-first : rank ↓, nb de votes ↓ (confiance), strength ↑ (anti-overfit)."""
    rows = LoraTestImage.query.filter_by(dataset_id=dataset_id).all()
    if family:
        fam = family.lower()
        rows = [r for r in rows if (family_of_lora(r.checkpoint) or 'zimage') == fam]
    agg = {}
    for r in rows:
        key = (r.checkpoint, r.strength, r.aspect, r.z_model, r.cfg, r.steps, r.steps2)
        e = agg.setdefault(key, {'checkpoint': r.checkpoint, 'strength': r.strength,
                                 'aspect': r.aspect, 'z_model': r.z_model,
                                 'z_model_label': _model_label(r.z_model),
                                 'cfg': r.cfg, 'steps': r.steps, 'steps2': r.steps2,
                                 'score': 0, 'likes': 0, 'dislikes': 0,
                                 'images': 0, 'voted': 0, 'rank': 0.0})
        e['images'] += 1
        if r.rating == 1:
            e['likes'] += 1
            e['voted'] += 1
        elif r.rating == -1:
            e['dislikes'] += 1
            e['voted'] += 1
    for e in agg.values():
        e['score'] = e['likes'] - e['dislikes']
        e['rank'] = round(_wilson_lower_bound(e['likes'], e['voted']), 4)
        # Taux d'approbation (likes/votés) - None si rien voté (pas de 0/0 trompeur).
        e['like_rate'] = round(e['likes'] / e['voted'], 4) if e['voted'] else None
        # Confiance : drapeau quand l'échantillon de votes est trop mince.
        e['low_confidence'] = e['voted'] < LOW_CONFIDENCE_MIN
    return sorted(agg.values(),
                  key=lambda e: (-e['rank'], -e['voted'], e['strength']))


def model_net_scores(dataset_id) -> dict:
    """Sentiment net par modèle (👍−👎 sur toutes ses images) - exposé pour
    l'affichage. Le gate de best_cell, lui, utilise le TAUX (voir _model_like_rates)."""
    rows = LoraTestImage.query.filter_by(dataset_id=dataset_id).all()
    net = {}
    for r in rows:
        if r.rating == 1:
            net[r.z_model] = net.get(r.z_model, 0) + 1
        elif r.rating == -1:
            net[r.z_model] = net.get(r.z_model, 0) - 1
    return net


def _model_like_rates(scores) -> dict:
    """Taux de 👍 par modèle (likes/voted) agrégé sur ses configs - sert à
    écarter un modèle globalement mal noté. {model: rate|None} (None = 0 vote)."""
    acc = {}
    for e in scores:
        likes, voted = acc.get(e['z_model'], (0, 0))
        acc[e['z_model']] = (likes + e['likes'], voted + e['voted'])
    return {m: (likes / voted if voted else None) for m, (likes, voted) in acc.items()}


def model_comparison(dataset_id, scores=None) -> list[dict]:
    """Agrégat de votes PAR modèle de base (z_model), pour comparer les bases
    ÉQUITABLEMENT. Classé par taux (Wilson lower bound), PAS par compte brut - qui
    favorise mécaniquement le modèle le plus testé (biais de volume). Chaque entrée
    porte images/voted pour rendre l'échantillon visible + low_confidence.

    `scores` partageable (cf. best_cell) pour éviter de re-scanner la table."""
    scores = cell_scores(dataset_id) if scores is None else scores
    acc = {}
    for e in scores:
        a = acc.setdefault(e['z_model'], {
            'z_model': e['z_model'], 'z_model_label': e['z_model_label'],
            'likes': 0, 'dislikes': 0, 'images': 0, 'voted': 0, 'checkpoints': set()})
        a['likes'] += e['likes']
        a['dislikes'] += e['dislikes']
        a['images'] += e['images']
        a['voted'] += e['voted']
        a['checkpoints'].add(e['checkpoint'])
    out = []
    for a in acc.values():
        out.append({
            'z_model': a['z_model'], 'z_model_label': a['z_model_label'],
            'likes': a['likes'], 'dislikes': a['dislikes'],
            'net': a['likes'] - a['dislikes'],
            'images': a['images'], 'voted': a['voted'],
            'like_rate': round(a['likes'] / a['voted'], 4) if a['voted'] else None,
            'wilson': round(_wilson_lower_bound(a['likes'], a['voted']), 4),
            'low_confidence': a['voted'] < LOW_CONFIDENCE_MIN,
            'n_checkpoints': len(a['checkpoints']),
        })
    out.sort(key=lambda m: (-m['wilson'], -m['voted']))
    return out


def checkpoint_model_breakdown(dataset_id, scores=None) -> list[dict]:
    """Par (checkpoint, z_model) : nb d'images générées / votées + taux de 👍.
    C'est le « nombre de générées par modèle, par LoRA » - le dénominateur qui
    montre où l'échantillon est mince (ex. Lola testé 12× sur bigLove vs 3× sur
    l'officiel). Trié par label de checkpoint puis taux décroissant.

    `scores` partageable (cf. best_cell)."""
    scores = cell_scores(dataset_id) if scores is None else scores
    acc = {}
    for e in scores:
        key = (e['checkpoint'], e['z_model'])
        a = acc.setdefault(key, {
            'checkpoint': e['checkpoint'],
            'label': format_trained_lora_label(e['checkpoint']) or _basename(e['checkpoint']).rsplit('.', 1)[0],
            'z_model': e['z_model'], 'z_model_label': e['z_model_label'],
            'likes': 0, 'dislikes': 0, 'images': 0, 'voted': 0})
        a['likes'] += e['likes']
        a['dislikes'] += e['dislikes']
        a['images'] += e['images']
        a['voted'] += e['voted']
    out = []
    for a in acc.values():
        a['net'] = a['likes'] - a['dislikes']
        a['like_rate'] = round(a['likes'] / a['voted'], 4) if a['voted'] else None
        a['low_confidence'] = a['voted'] < LOW_CONFIDENCE_MIN
        out.append(a)
    out.sort(key=lambda a: (a['label'], -(a['like_rate'] or 0), -a['voted']))
    return out


def best_cell(dataset_id, scores=None) -> dict | None:
    """Config recommandée d'après les votes :
      1. candidats = configs nettes positives (👍 > 👎) ;
      2. tri par `rank` Wilson ↓ (taux × confiance) - le MÉRITE de la config prime ;
      3. départages : nb de votes ↓ (confiance), puis taux de 👍 GLOBAL du modèle ↓
         (à config équivalente, on préfère le modèle mieux noté), puis strength ↑.
    Le sentiment du modèle est un DÉPARTAGE, pas un filtre : une config nettement
    mieux notée n'est jamais écartée parce que son modèle est moyen ailleurs (sinon
    le sweep par-case n'aurait aucun sens). Retourne None tant que rien n'est aimé.

    `scores` peut être passé (déjà calculé) pour éviter de re-scanner la table -
    studio_payload partage un seul cell_scores entre best_cell/best_preset/best_per_checkpoint."""
    scores = cell_scores(dataset_id) if scores is None else scores
    candidates = [e for e in scores if e['likes'] > e['dislikes']]
    if not candidates:
        return None
    rates = _model_like_rates(scores)

    def model_pref(m):
        r = rates.get(m)
        return r if r is not None else 0.5  # modèle sans vote = neutre
    candidates.sort(key=lambda e: (-e['rank'], -e['voted'],
                                   -model_pref(e['z_model']), e['strength']))
    return candidates[0]


def best_preset(dataset_id, scores=None) -> dict | None:
    """La config recommandée (best_cell, modèle inclus) enrichie d'une image
    représentative (prompt/seed/filename) de CETTE config exacte."""
    bc = best_cell(dataset_id, scores=scores)
    if not bc:
        return None
    img = (LoraTestImage.query
           .filter_by(dataset_id=dataset_id, checkpoint=bc['checkpoint'],
                      strength=bc['strength'], aspect=bc.get('aspect'),
                      z_model=bc.get('z_model'), cfg=bc.get('cfg'),
                      steps=bc.get('steps'), steps2=bc.get('steps2'), status='done')
           .order_by(LoraTestImage.id.desc()).first())
    return {
        **bc,
        'label': format_trained_lora_label(bc['checkpoint']) or _basename(bc['checkpoint']).rsplit('.', 1)[0],
        'prompt': getattr(img, 'prompt', None) if img else None,
        'seed': img.seed if img else None,
        'filename': img.filename if img else None,
    }


def best_per_checkpoint(dataset_id, scores=None) -> list[dict]:
    """Meilleur réglage PAR checkpoint (les votes varient beaucoup d'un modèle à
    l'autre - un best global ne suffit pas). Pour chaque checkpoint ayant ≥1 config
    nette positive (👍>👎), retourne sa config la mieux notée (MÊME tri Wilson que
    best_cell), enrichie d'une image représentative. Trié par rank décroissant.

    `scores` partageable (cf. best_cell) pour éviter de re-scanner la table."""
    scores = cell_scores(dataset_id) if scores is None else scores
    candidates = [e for e in scores if e['likes'] > e['dislikes']]
    if not candidates:
        return []
    rates = _model_like_rates(scores)

    def model_pref(m):
        r = rates.get(m)
        return r if r is not None else 0.5
    candidates.sort(key=lambda e: (-e['rank'], -e['voted'],
                                   -model_pref(e['z_model']), e['strength']))
    best_by_cp = {}
    for e in candidates:  # déjà triés → le 1er vu par checkpoint = le meilleur
        best_by_cp.setdefault(e['checkpoint'], e)
    out = []
    for bc in best_by_cp.values():
        img = (LoraTestImage.query
               .filter_by(dataset_id=dataset_id, checkpoint=bc['checkpoint'],
                          strength=bc['strength'], aspect=bc.get('aspect'),
                          z_model=bc.get('z_model'), cfg=bc.get('cfg'),
                          steps=bc.get('steps'), steps2=bc.get('steps2'), status='done')
               .order_by(LoraTestImage.id.desc()).first())
        out.append({**bc,
                    'label': format_trained_lora_label(bc['checkpoint']) or _basename(bc['checkpoint']).rsplit('.', 1)[0],
                    'prompt': getattr(img, 'prompt', None) if img else None,
                    'seed': img.seed if img else None,
                    'filename': img.filename if img else None})
    out.sort(key=lambda e: -e['rank'])
    return out


def _best_map(ds) -> dict:
    """best_settings persistés en map {famille: réglage}. RÉTRO-COMPAT : un ancien
    format PLAT (un seul réglage, repérable à sa clé top-level `lora_filename`) est
    rattaché au train_type du dataset. Retourne {} si vide/illisible."""
    if not ds.best_settings:
        return {}
    try:
        data = json.loads(ds.best_settings)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    if 'lora_filename' in data:  # ancien format plat (mono-famille)
        return {(getattr(ds, 'train_type', None) or 'zimage').lower(): data}
    return data


def _best_for_family(ds, family) -> dict | None:
    """Réglage mémorisé pour CETTE famille (None si aucun)."""
    return _best_map(ds).get((family or 'zimage').lower())


def set_best_settings(user_id, dataset_id, checkpoint, strength,
                      z_model=None, cfg=None, steps=None, steps2=None, aspect=None) -> dict:
    """Persiste la config gagnante COMPLÈTE - checkpoint, strength, modèle/cfg/steps(1+2)/
    format. Mémorisé PAR FAMILLE (un même dataset a un meilleur réglage distinct en ZIT,
    SDXL, Krea) : la famille est déduite du dossier du checkpoint. Le checkpoint doit
    appartenir à la whitelist de SA famille ; le modèle, s'il est fourni, est validé
    contre les bases du bon type (Krea = base fixe → modèle ignoré). Retourne le réglage."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    family = (family_of_lora(checkpoint) or getattr(ds, 'train_type', None) or 'zimage').lower()
    allowed = {c['filename'] for c in list_test_checkpoints(ds, family)}
    if checkpoint not in allowed:
        raise ValueError('unknown checkpoint for this dataset')
    try:
        strength = round(float(strength), 2)
    except (TypeError, ValueError):
        raise ValueError(f'invalid strength: {strength!r}')
    if not 0.05 <= strength <= 2.0:
        raise ValueError(f'strength out of range: {strength}')
    # Whitelist de bases selon la FAMILLE (SDXL → bases SDXL ; Krea → aucune base
    # sélectionnable ; sinon Z-Image), sinon une base d'une autre famille était jetée.
    if family == 'sdxl':
        allowed_bases = {m['filename'] for m in list_sdxl_base_models()}
    elif family == 'krea':
        allowed_bases = set()
    else:
        allowed_bases = set(get_zimage_models())
    if z_model and z_model not in allowed_bases:
        z_model = None  # modèle inconnu → on ne l'enregistre pas (au lieu de mentir)
    try:
        cfg = round(float(cfg), 2) if cfg is not None else None
    except (TypeError, ValueError):
        cfg = None
    try:
        steps = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        steps = None
    try:
        steps2 = int(steps2) if steps2 is not None else None
    except (TypeError, ValueError):
        steps2 = None
    aspect = aspect if aspect in TEST_ASPECTS else None
    best = {
        'lora_filename': checkpoint,
        'strength': strength,
        'z_model': z_model,
        'cfg': cfg,
        'steps': steps,
        'steps2': steps2,
        'aspect': aspect,
        'family': family,
        'decided_at': datetime.utcnow().isoformat(),
    }
    best_map = _best_map(ds)
    best_map[family] = best
    ds.best_settings = json.dumps(best_map)
    db.session.commit()
    return best


def clear_best_settings(user_id, dataset_id, family=None) -> bool:
    """Efface le réglage mémorisé. `family` → n'efface que cette famille (les autres
    survivent) ; absent → efface tout. Idempotent (pas d'erreur s'il n'y a rien)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if family:
        m = _best_map(ds)
        m.pop((family or '').lower(), None)
        ds.best_settings = json.dumps(m) if m else None
    else:
        ds.best_settings = None
    db.session.commit()
    return True


# --- Scoring facial objectif (« best epoch » auto - méthode jandordoe) --------
def score_faces(user_id, dataset_id, family=None) -> dict:
    """Score InsightFace (antelopev2, subprocess CPU - ne touche PAS le GPU) de
    chaque cellule TERMINÉE de la famille vs la RÉFÉRENCE du dataset. Persiste
    face_score/face_state par cellule, puis renvoie le classement par checkpoint.

    C'est l'automatisation de la méthode jandordoe : générer les checkpoints à
    seed fixe (le Studio le fait déjà), puis choisir l'epoch au MEILLEUR score
    facial mesuré au lieu du dernier. Idempotent : rescorer écrase les scores."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference photo missing')
    ref_path = fds._ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference photo missing')
    eff = _resolve_family(ds, family, available_families(ds))
    rows = (LoraTestImage.query.filter_by(dataset_id=dataset_id, status='done')
            .filter(LoraTestImage.filename.isnot(None)).all())
    rows = [r for r in rows if (family_of_lora(r.checkpoint) or 'zimage') == eff]
    ds_dir = fds._dataset_dir(dataset_id)
    by_path = {}
    for r in rows:
        p = os.path.join(ds_dir, r.filename)
        if os.path.exists(p):
            by_path[p] = r
    if not by_path:
        return {'scored': 0, 'total': 0, 'ranking': []}
    from .face_similarity import score_dataset_faces
    results = score_dataset_faces(ref_path, list(by_path.keys()))
    scored = 0
    for p, r in by_path.items():
        res = results.get(p)
        if not res:
            continue
        r.face_state = res.get('state')
        r.face_score = res.get('sim')
        scored += 1
    db.session.commit()
    logger.info(f"lora-test: score-faces dataset {dataset_id} ({eff}) -> "
                f"{scored}/{len(by_path)} cellule(s) scorée(s)")
    return {'scored': scored, 'total': len(by_path), 'ranking': face_ranking(dataset_id, eff)}


def face_ranking(dataset_id, family) -> list:
    """Classement des checkpoints par similarité faciale MOYENNE (cellules déjà
    scorées, famille donnée). [{checkpoint, label, avg, n}] trié meilleur d'abord -
    le front marque le 1er comme « 🏆 best epoch »."""
    rows = (LoraTestImage.query.filter_by(dataset_id=dataset_id)
            .filter(LoraTestImage.face_score.isnot(None)).all())
    rows = [r for r in rows if (family_of_lora(r.checkpoint) or 'zimage') == family]
    agg = {}
    for r in rows:
        a = agg.setdefault(r.checkpoint, [0.0, 0])
        a[0] += float(r.face_score)
        a[1] += 1
    out = [{'checkpoint': cp,
            'label': format_trained_lora_label(cp) or _basename(cp).rsplit('.', 1)[0],
            'avg': round(s / n, 4), 'n': n}
           for cp, (s, n) in agg.items()]
    out.sort(key=lambda e: (-e['avg'], -e['n']))
    return out


def delete_prompt(user_id, dataset_id, prompt) -> int:
    """Supprime toutes les cellules de test d'un PROMPT donné (+ leurs fichiers) :
    retire ce prompt du menu « prompts récents » et nettoie ses images de test.
    Annule les jobs encore en vol. Ownership scoped (anti-IDOR). Retourne le nombre
    de cellules supprimées."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    p = (prompt or '').strip()
    if not p:
        return 0
    rows = LoraTestImage.query.filter_by(dataset_id=dataset_id, prompt=p).all()
    if not rows:
        return 0
    dataset_dir = fds._dataset_dir(dataset_id)
    n = 0
    for r in rows:
        # Cellule encore en file → annuler le job avant de la supprimer.
        if r.status == 'pending' and r.job_id and not r.filename:
            try:
                queue_manager.cancel_job(r.job_id, str(user_id), 'image')
            except Exception:
                pass
        if r.filename:
            try:
                fp = os.path.join(dataset_dir, r.filename)
                if os.path.exists(fp):
                    os.remove(fp)
            except OSError:
                pass
        db.session.delete(r)
        n += 1
    db.session.commit()
    logger.info(f"lora-test: prompt supprimé sur dataset {dataset_id} -> {n} cellule(s)")
    return n


# --- Payload (poll) ------------------------------------------------------------
def studio_payload(user_id, dataset_id, family=None) -> dict | None:
    """Everything the studio panel needs in one poll, SCOPÉ à une FAMILLE (pipeline).

    `family` = ZIT/SDXL/Krea sélectionnée par l'utilisateur ; résolue à la famille
    effective (parmi celles réellement présentes pour ce dataset). Checkpoints, grille,
    scores, best et bases sont tous restreints à cette famille - un même dataset
    entraîné sous plusieurs pipelines n'en mélange plus les résultats. `available_families`
    liste les familles présentes (pour le sélecteur) ; `family` renvoie l'effective."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return None
    fams = available_families(ds)
    eff = _resolve_family(ds, family, fams)
    rows_all = (LoraTestImage.query.filter_by(dataset_id=dataset_id)
                .order_by(LoraTestImage.id.asc()).all())
    # Grille = cellules de la famille effective (famille déduite du checkpoint).
    rows = [r for r in rows_all if (family_of_lora(r.checkpoint) or 'zimage') == eff]
    best = _best_for_family(ds, eff)
    # Pool de bases selon la FAMILLE effective : SDXL → checkpoints SDXL (forme
    # {value,label}) ; Krea → base fixe (UNET du workflow, aucun sélecteur) ; sinon
    # modèles Z-Image. `train_type` = famille effective (le front adapte picker + handoff).
    if eff == 'sdxl':
        z_models = [{'value': m['filename'], 'label': m['label']}
                    for m in list_sdxl_base_models()]
    elif eff == 'krea':
        z_models = []
    else:
        z_models = [{'value': m, 'label': _basename(m).rsplit('.', 1)[0]}
                    for m in get_zimage_models()]
    return {
        'checkpoints': list_test_checkpoints(ds, eff),
        'trigger_word': ds.trigger_word,
        'train_type': eff,
        'family': eff,
        # Familles entraînées de ce dataset (sélecteur) : [{family,label,count}].
        'available_families': fams,
        # LoRA « always-on » disponibles pour cette famille (style/utilitaire, hors batch).
        'permanent_loras': permanent_lora_candidates(eff),
        'prompt': identity_prompt(ds),
        'z_models': z_models,
        'aspects': list(TEST_ASPECTS.keys()),
        'default_aspect': DEFAULT_ASPECT,
        'cfg_choices': CFG_CHOICES, 'default_cfg': DEFAULT_CFG,
        'steps_choices': STEPS_CHOICES, 'default_steps': DEFAULT_STEPS,
        # 2e passe (detail daemon) : exposée UNIQUEMENT pour SDXL (le workflow HQ a deux
        # passes). NULL sinon → le frontend ne montre pas le 2e picker de steps.
        'steps2_choices': (STEPS_CHOICES if eff == 'sdxl' else None),
        'default_steps2': (DEFAULT_STEPS if eff == 'sdxl' else None),
        'max_images': MAX_TEST_IMAGES,
        'cells': [{'id': r.id, 'checkpoint': r.checkpoint,
                   'label': format_trained_lora_label(r.checkpoint) or _basename(r.checkpoint).rsplit('.', 1)[0],
                   'strength': r.strength, 'aspect': r.aspect, 'filename': r.filename,
                   'rating': r.rating, 'seed': r.seed, 'run_seed': r.run_seed, 'status': r.status,
                   'prompt': r.prompt, 'z_model': r.z_model,
                   'z_model_label': (_basename(r.z_model).rsplit('.', 1)[0] if r.z_model else None),
                   'cfg': r.cfg, 'steps': r.steps, 'steps2': r.steps2,
                   'batch_lora': _batch_lora_label(r),
                   'face_score': r.face_score, 'face_state': r.face_state}
                  for r in rows],
        # cell_scores scanne la table une fois (filtré famille) → partagé entre
        # best_cell/best_preset/best_per_checkpoint (sinon 4 scans identiques).
        'scores': (_scores := cell_scores(dataset_id, family=eff)),
        'best_cell': best_cell(dataset_id, scores=_scores),
        'best_preset': best_preset(dataset_id, scores=_scores),
        'best_per_model': best_per_checkpoint(dataset_id, scores=_scores),
        # Comparaison équitable des bases (par z_model) + détail par (checkpoint, base).
        'model_comparison': model_comparison(dataset_id, scores=_scores),
        'checkpoint_breakdown': checkpoint_model_breakdown(dataset_id, scores=_scores),
        # Classement facial objectif des checkpoints (« best epoch », cellules scorées).
        'face_ranking': face_ranking(dataset_id, eff),
        'pending': _active_run_count(dataset_id),
        # Cellules stoppées/échouées reprenables - global (resume opère sur tout le dataset).
        'resumable': sum(1 for r in rows_all if r.status in ('cancelled', 'failed')),
        # Prompts récents distincts (family-agnostiques) pour recharger/relancer un
        # run - GLOBAUX à l'utilisateur (tous datasets), plus cloisonnés par dataset.
        'recent_prompts': user_recent_prompts(ds.user_id),
        'gpu_busy': gpu_busy_reason(),
        'best_settings': best,
    }


def lora_net_scores(run_id) -> list[dict]:
    """Classement PAR-LoRA d'un run : agrège les votes des cellules par dataset_id
    (= un LoRA). Trié par score net (likes - dislikes) puis likes, décroissant."""
    rows = LoraTestImage.query.filter_by(run_id=run_id).filter(
        LoraTestImage.filename.isnot(None)).all()
    agg = {}
    for r in rows:
        a = agg.setdefault(r.dataset_id, {'dataset_id': r.dataset_id, 'likes': 0,
                                          'dislikes': 0, 'voted': 0, 'total': 0,
                                          'lora_label': format_trained_lora_label(r.checkpoint)
                                          or _basename(r.checkpoint).rsplit('.', 1)[0]})
        a['total'] += 1
        if r.rating == 1: a['likes'] += 1; a['voted'] += 1
        elif r.rating == -1: a['dislikes'] += 1; a['voted'] += 1
    for a in agg.values():
        a['net'] = a['likes'] - a['dislikes']
        a['wilson'] = _wilson_lower_bound(a['likes'], a['voted'])
        ds = FaceDataset.query.get(a['dataset_id'])
        a['dataset_name'] = ds.name if ds else f"#{a['dataset_id']}"
    return sorted(agg.values(), key=lambda a: (a['net'], a['likes']), reverse=True)


def studio_payload_run(user_id, run_id) -> dict | None:
    """Payload d'un run (mono ou multi-LoRA). Requêté par run_id + ajoute le
    classement par-LoRA et la liste des LoRA présents."""
    rows = (LoraTestImage.query.filter_by(run_id=run_id)
            .order_by(LoraTestImage.id.asc()).all())
    if not rows:
        return None
    ds_ids = {r.dataset_id for r in rows}
    owned = {d.id for d in FaceDataset.query.filter(FaceDataset.user_id == str(user_id),
             FaceDataset.id.in_(ds_ids)).all()}
    if ds_ids - owned:
        return None
    def _lbl(d):
        return next((_basename(r.checkpoint).rsplit('.', 1)[0] for r in rows if r.dataset_id == d), str(d))
    def _name(d):
        ds = FaceDataset.query.get(d); return ds.name if ds else str(d)
    return {
        'run_id': run_id,
        'loras': [{'dataset_id': d, 'lora_label': _lbl(d), 'dataset_name': _name(d)}
                  for d in sorted(ds_ids)],
        'cells': [{'id': r.id, 'dataset_id': r.dataset_id, 'checkpoint': r.checkpoint,
                   'label': _basename(r.checkpoint).rsplit('.', 1)[0], 'strength': r.strength,
                   'aspect': r.aspect, 'filename': r.filename, 'rating': r.rating, 'seed': r.seed,
                   'run_seed': r.run_seed, 'status': r.status, 'prompt': r.prompt,
                   'z_model': r.z_model, 'cfg': r.cfg, 'steps': r.steps, 'steps2': r.steps2,
                   'batch_lora': _batch_lora_label(r)} for r in rows],
        'lora_ranking': lora_net_scores(run_id),
        'pending': sum(1 for r in rows if r.status == 'pending' and not r.filename),
        'resumable': sum(1 for r in rows if r.status in ('cancelled', 'failed')),
        'gpu_busy': gpu_busy_reason(),
    }


def _recent_prompts(rows, limit=6) -> list[dict]:
    """Prompts distincts utilisés (récent→ancien) AVEC une vignette : une image 👍
    générée avec ce prompt (à défaut, la plus récente terminée), + le nombre d'images.
    Permet de voir ce que fait chaque prompt dans le menu. `thumb_dataset_id` porte
    le dataset de la vignette (nécessaire quand les rows couvrent PLUSIEURS datasets).
    Retour: [{prompt, thumbnail(filename|None), thumb_dataset_id, thumb_rating, count}]."""
    seen = {}  # prompt -> dict (ordre d'insertion = récent→ancien)
    for r in sorted(rows, key=lambda x: -x.id):  # plus récent d'abord
        p = (r.prompt or '').strip()
        if not p:
            continue
        if p not in seen:
            if len(seen) >= limit:
                continue
            seen[p] = {'prompt': p, 'thumbnail': None, 'thumb_dataset_id': None,
                       'thumb_rating': 0, 'count': 0}
        e = seen[p]
        if r.filename:
            e['count'] += 1
            if r.rating == 1 and e['thumb_rating'] != 1:      # préférer un 👍 (le + récent)
                e['thumbnail'], e['thumb_rating'] = r.filename, 1
                e['thumb_dataset_id'] = r.dataset_id
            elif e['thumbnail'] is None:                       # sinon la 1re terminée vue (= + récente)
                e['thumbnail'], e['thumb_rating'] = r.filename, (r.rating or 0)
                e['thumb_dataset_id'] = r.dataset_id
    return list(seen.values())


def user_recent_prompts(user_id, limit=10) -> list[dict]:
    """Prompts de test récents de l'UTILISATEUR, TOUS datasets confondus (demande
    2026-07-03 : la mémoire des prompts/presets ne doit plus être cloisonnée par
    dataset - un prompt réglé sur Emma doit se recharger sur Adele). Scan borné aux
    1500 dernières cellules (perf) ; chaque entrée porte `thumb_dataset_id` pour que
    le front construise l'URL de vignette du BON dataset."""
    ds_ids = [d.id for d in FaceDataset.query.filter_by(user_id=str(user_id)).all()]
    if not ds_ids:
        return []
    rows = (LoraTestImage.query.filter(LoraTestImage.dataset_id.in_(ds_ids))
            .order_by(LoraTestImage.id.desc()).limit(1500).all())
    return _recent_prompts(rows, limit=limit)


def delete_prompt_everywhere(user_id, prompt) -> int:
    """Supprime un prompt récent (et ses cellules/images de test) sur TOUS les
    datasets de l'utilisateur - pendant « suppression » de la liste globale."""
    p = (prompt or '').strip()
    if not p:
        return 0
    n = 0
    for d in FaceDataset.query.filter_by(user_id=str(user_id)).all():
        try:
            n += delete_prompt(user_id, d.id, p)
        except ValueError:
            continue
    return n
