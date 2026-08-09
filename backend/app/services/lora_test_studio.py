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
  - free (never debited), one active run per dataset, refused while
    training/vision holds the GPU.

    ⚠️ Il n'y a PAS de plafond sur le nombre de cellules d'un run, et c'est
    délibéré : cf. `build_matrix`, « la file est sérielle et l'utilisateur voit
    le compte + l'estimation de durée avant de lancer ». Cette ligne a longtemps
    annoncé un « hard-capped (MAX_TEST_IMAGES per run) » que rien n'appliquait —
    `MAX_TEST_IMAGES` n'est lu QUE pour être renvoyé au frontend (`max_images`),
    où il sert de seuil d'AVERTISSEMENT. Un commentaire qui promet une garantie
    que le code ne tient pas est pire qu'un commentaire absent.

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

import itertools
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
from ..gpu_window import GpuBusyError
from ..models import FaceDataset, ImageGenerationQueue, LoraTestImage
from . import face_dataset_service as fds, trash
from . import lora_training as lt
from ..job_queue import GPU_ARBITER_LOCK, queue_manager
from ..utils.comfyui import (FAMILY_LABELS, KREA_ALLOWED_SAMPLERS, KREA_ALLOWED_SCHEDULERS,
                             KREA_ALLOWED_WEIGHT_DTYPES, apply_optimal_sampler_params,
                             family_of_lora, format_trained_lora_label, get_krea_loras,
                             get_krea_models, get_sdxl_loras, get_zimage_loras,
                             get_zimage_models, inject_krea2t_enhancer,
                             load_workflow_local, resolve_checkpoint_ckpt_name)
from ..utils.zimage_helper import apply_zimage_settings

logger = logging.getLogger(__name__)

# ✨ The derivation kind of a row that lives in lora_test_image WITHOUT being a
# Test Studio cell: the Upscale & improve result produced from the ◉ Canvas
# lightbox. It has to be in this table because `canvas_image_node.image_id` is a
# `lora_test_image.id` and the board must be able to pin it — but no checkpoint×
# strength sweep ever made it.
CANVAS_IMAGE_IMPROVE = 'canvas_image_improve'


def _is_cell():
    """The predicate `_cells()` applies, for the queries that cannot START from it.

    A join that selects columns from another table (see
    `measured_seconds_per_image`) builds its own query and cannot be handed a
    `LoraTestImage.query`. It still needs the SAME rule, so the rule is written
    once here and both spellings share it — two hand-written copies of
    `derivation_kind IS NULL` is how the two would eventually disagree.
    """
    return LoraTestImage.derivation_kind.is_(None)


def _cells():
    """EVERY query in this module that means "the Test Studio cells" starts here.

    A derived row (see CANVAS_IMAGE_IMPROVE) is in the same table and must not be
    read as a cell. Auditing the 21 read sites found TEN that break otherwise,
    and none of them is theoretical:

      * `_active_run_count` counts pending rows with no file, so an improve still
        rendering answered "a test run is already in progress" and BLOCKED every
        new Test Studio launch;
      * the resume path picks up cancelled/failed rows by dataset and would have
        re-queued a failed improve as a Z-Image cell — wrong workflow, wrong
        engine;
      * `cell_scores` / `model_net_scores` aggregate by (checkpoint, strength, …),
        so a 👍 given to an UPSCALE in the gallery became a vote for the
        checkpoint that did not produce it;
      * `best_cell` and its per-checkpoint sibling take `order_by(id.desc())`, so
        the improvement — newest by construction — became the representative
        image of the winning config;
      * the face-scoring pass would spend GPU on it and `face_ranking` would
        average its score into its checkpoint's.

    So the filter is not a detail of one query, it is the meaning of "cell", and
    it lives in ONE place. `test_canvas_image_improve.py` forbids a bare
    `LoraTestImage.query` anywhere else in this module, so a future reader cannot
    reintroduce the leak by writing the obvious thing. A query that legitimately
    needs every row carries a `lds-allow-bare-lora-test-query:` comment saying
    why — the four that exist today all resolve ONE row by `job_id`, and the
    completion callback is among them: filtering derived rows out of it would
    leave every improvement pending forever.

    ⚠️ WHERE THIS HELPER MUST **NOT** BE USED — this boundary is the feature.
    `services/cloud_training.py` (checkpoint_gallery, run_gallery,
    canvas_image_nodes) and `services/gallery_download.py` read the SAME table and
    deliberately do NOT filter these rows out: showing the improvement in the
    gallery it was made from, next to its source, and letting it be pinned onto
    the board, IS what the ✨ button was asked for. "Harmonising" by applying this
    helper there would silently delete the feature, so a test pins both
    directions: absent from the studio's cells, present in the checkpoint gallery
    and pinnable.
    """
    return LoraTestImage.query.filter(_is_cell())


# Plafond dur d'images par run (~4-6 min de GPU max en Z-Image Turbo).
MAX_TEST_IMAGES = 24

# 🔆 LA plage de force d'un LoRA dans cette app — UN seul couple de bornes.
#
# Elle borne DEUX choses qui ne se ressemblent pas mais atterrissent dans la
# même colonne (`LoraTestImage.strength`) : l'axe de balayage « Strengths » du
# Test Studio, et le poids de TÊTE d'une pile 🧬 Blend (create_comparison_run
# fait passer `combo[0]` par build_matrix). Deux bornes séparées ont donc un
# mode de panne précis et silencieux : un blend réglé au-dessus de la borne de
# l'axe n'est pas rendu plus faible, il est REFUSÉ, run entier compris.
#
# Le plafond est passé de 4.0 à 5.0 le 08/08/2026. C'était un plafond de
# confort : rien côté ComfyUI n'interdit d'aller plus haut, et pousser un LoRA
# sous-entraîné ou un style qu'on veut écrasant sont des usages réels qui
# obligeaient à sortir de l'app. Ça reste un plafond — au-delà ce n'est plus
# « fort », c'est du bruit — et le plancher négatif ne bouge pas : -2.0 est le
# pôle inverse d'un slider LoRA, pas une force.
#
# ⚠️ Miroirs côté navigateur, à bouger dans le MÊME commit :
#   frontend/src/components/dataset/studio/loraStack.js  (COMBINE_MAX_WEIGHT)
#   frontend/src/components/dataset/studio/constants.js  (STRENGTH_CHOICES_EXTENDED)
MIN_LORA_STRENGTH = -2.0
MAX_LORA_STRENGTH = 5.0

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


def _aspect_dims(aspect, train_type=None, resolution_tier=None, resolution_multiplier=1.0):
    """(width, height) d'un format. Si `resolution_tier` (fast|standard|hq|max) est fourni,
    délègue à `compute_tier_dims` (ratio nommé + mégapixels du palier, comme Generate),
    avec le multiplicateur de résolution (1.0–1.9, clampé, défaut 1.0 = palier inchangé) ;
    sinon table fixe par famille (SDXL côté long ≤1024, sinon table Z-Image historique -
    le multiplicateur n'agit QUE sur le chemin par palier, pas sur les tables legacy).
    Format inconnu → défaut. SDXL + palier : on re-borne le côté long à 1024×multiplicateur
    (la bande SDXL-safe monte aussi avec le multiplicateur, multiples de 64) car
    compute_tier_dims monte jusqu'à 1536 (safe Z-Image, déforme les merges/DMD SDXL)."""
    if resolution_tier in RESOLUTION_TIERS:
        named = _ASPECT_TO_TIER_RATIO.get(aspect)
        if named:
            from ..utils.resolution import clamp_multiplier, compute_tier_dims
            w, h = compute_tier_dims(named, resolution_tier, resolution_multiplier)
            if (train_type or '').lower() == 'sdxl':
                # Plafond SDXL mis à l'échelle du multiplicateur, sinon celui-ci serait
                # silencieusement écrasé (le front affiche déjà 1024×mult pour SDXL).
                ceiling = 1024.0 * clamp_multiplier(resolution_multiplier)
                longest = max(w, h)
                if longest > ceiling:
                    sc = ceiling / longest
                    w = max(64, int(round(w * sc / 64)) * 64)
                    h = max(64, int(round(h * sc / 64)) * 64)
            return w, h
    table = TEST_ASPECTS_SDXL if (train_type or '').lower() == 'sdxl' else TEST_ASPECTS
    return table.get(aspect, table[DEFAULT_ASPECT])

# Axes optionnels CFG / steps. Le défaut de la FAMILLE reste le réglage distillé
# (cfg=1.0, 8 steps) : c'est ce que valent Z-Image Turbo, Krea 2 Turbo et les
# checkpoints SDXL DMD-distillés que le studio teste. Tester plusieurs valeurs aide
# à trouver le réglage qui tient le mieux l'identité.
DEFAULT_CFG = 1.0
DEFAULT_STEPS = 8
# Additive only — these lists are echoed into the Studio pickers and a value that
# disappears would strand a persisted selection. 3.5/4.0/5.0 and 30/50 exist so the
# NON-distilled Z-Image Base defaults below are reachable from the picker at all.
CFG_CHOICES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
# 25 is the sample-step count a dense Krea 2 run previews with — see
# KREA_RAW_DEFAULTS below; without it in the picker the recommended setting for a
# full-model artifact would not be selectable at all.
STEPS_CHOICES = [6, 8, 10, 12, 16, 20, 24, 25, 30, 32, 40, 50]

# --- Per-BASE-MODEL sampler defaults (bobba84, GitHub #18) --------------------
# Z-Image ships in two flavours that need opposite sampler settings, and the app
# used to hand both the same one: picking "Z-Image Base" in the Test Studio landed
# on cfg 1 / 8 steps, which are Turbo's numbers. Turbo is guidance-DISTILLED — cfg 1
# is correct there and ruinous on Base, where it means "no guidance at all". A user
# trying Base with those settings concludes the model is bad.
#
# PROVENANCE OF THE BASE NUMBERS: ComfyUI's own Z-Image day-0 announcement states
# Z-Image-Base "requires 30-50 steps with cfg 3~5 for optimal quality". We take the
# CONSERVATIVE end of the step range (30, the cheapest of the recommended window)
# and the middle of the cfg range (4.0). These are documented starting points, NOT
# values this project measured — they are defaults for an axis the user can and
# should sweep, which is the entire point of the Studio grid.
ZIMAGE_TURBO_DEFAULTS = {'cfg': DEFAULT_CFG, 'steps': DEFAULT_STEPS}
ZIMAGE_BASE_DEFAULTS = {'cfg': 4.0, 'steps': 30}

# Whole-word phrases (see zimage_model_resolver._phrase: separators and case are
# normalised away) that identify a build. Distilled markers are checked FIRST, so a
# name carrying both stays on today's behaviour rather than flipping to slow+guided.
_ZIMAGE_DISTILLED_PHRASES = ('turbo', 'zt1', 'zt2', 'zt3', 'distill', 'distilled',
                             'lightning', 'lightx2v', 'step')
_ZIMAGE_BASE_PHRASES = ('base', 'deturbo', 'de turbo', 'raw')


def zimage_build_of(model_name) -> str:
    """'turbo' | 'base' | 'unknown' for a Z-Image UNET filename, read from its NAME —
    the only signal available, since these are loose files a user downloaded. 'unknown'
    deliberately keeps the historical Turbo defaults: the overwhelming majority of
    Z-Image checkpoints in the wild are Turbo finetunes, and changing the defaults for
    an unrecognised name would be a regression for everyone who is fine today."""
    from .zimage_model_resolver import _phrase
    key = _phrase(_basename(model_name))
    if any(f' {p} ' in key for p in _ZIMAGE_DISTILLED_PHRASES):
        return 'turbo'
    if any(f' {p} ' in key for p in _ZIMAGE_BASE_PHRASES):
        return 'base'
    return 'unknown'


def zimage_model_defaults(model_name) -> dict:
    """{'cfg', 'steps'} for ONE Z-Image base model. Turbo/unknown -> today's values."""
    return dict(ZIMAGE_BASE_DEFAULTS if zimage_build_of(model_name) == 'base'
                else ZIMAGE_TURBO_DEFAULTS)


# --- Krea 2: the same trap, one family over ------------------------------------
# The full-model (dense) lane delivers a RAW Krea 2 checkpoint — undistilled, and
# it needs a real CFG and a real step count. The Studio's family defaults are
# Turbo's (cfg 1 / 8 steps), and applied to a Raw model they render a blurry
# sketch that reads as "the fine-tune failed". Reported after the first dense run
# was tested that way.
#
# The numbers are not invented here: they are the sample settings the run's OWN
# preview sheet was rendered with, imported from the training recipe so a change
# there moves the test lane with it.
_KREA_DISTILLED_PHRASES = ('turbo', 'distill', 'distilled', 'lightning',
                           'lightx2v', 'step', 'schnell')
# 'full' and 'fp8' cover this app's own dense deliveries (Krea_full_<trigger>…
# and its _fp8 twin); 'raw'/'base'/'undistilled' cover Krea-2-Raw derivatives.
_KREA_RAW_PHRASES = ('raw', 'base', 'full', 'undistilled', 'fp8')


def krea_build_of(model_name) -> str:
    """'turbo' | 'raw' | 'unknown' for a Krea 2 checkpoint, read from its NAME.

    Same contract as ``zimage_build_of``: distilled markers win, and 'unknown'
    keeps today's Turbo defaults — most Krea checkpoints in the wild are Turbo
    finetunes and changing their defaults would be a regression.
    """
    from .zimage_model_resolver import _phrase
    key = _phrase(_basename(model_name))
    if any(f' {p} ' in key for p in _KREA_DISTILLED_PHRASES):
        return 'turbo'
    if any(f' {p} ' in key for p in _KREA_RAW_PHRASES):
        return 'raw'
    return 'unknown'


def krea_model_defaults(model_name) -> dict:
    """{'cfg', 'steps'} for ONE Krea 2 checkpoint. Turbo/unknown -> today's values."""
    if krea_build_of(model_name) != 'raw':
        return {'cfg': DEFAULT_CFG, 'steps': DEFAULT_STEPS}
    from . import lora_training as lt
    return {'cfg': float(lt.FULL_TRANSFORMER_SAMPLE_GUIDANCE),
            'steps': int(lt.FULL_TRANSFORMER_SAMPLE_STEPS)}


def studio_model_defaults(family, models) -> dict:
    """{model_value: {'cfg', 'steps'}} for the bases the Studio offers, so the front
    can seed its axes from the SELECTED base instead of one family-wide constant.
    Z-Image and Krea 2 both ship a distilled and an undistilled build that need
    opposite sampler settings; SDXL returns nothing and keeps
    `default_cfg`/`default_steps`. The shape is per-family on purpose so the next
    family that needs it has nowhere else to put it."""
    fam = (family or '').lower()
    resolver = {'zimage': zimage_model_defaults, 'krea': krea_model_defaults}.get(fam)
    if resolver is None:
        return {}
    out = {}
    for m in models or []:
        value = m.get('value') if isinstance(m, dict) else m
        if value:
            out[value] = resolver(value)
    return out


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


def _prompt_with_triggers(prompt, trigger_words):
    """Same as `_prompt_with_trigger` but for a STACK of LoRA loaded together
    (combine mode): every trigger is prefixed, in selection order, so
    `["a", "b"], "portrait"` gives `"a, b, portrait"`.

    Accepts a bare string (the historical single-LoRA call) or any iterable.
    Folding runs right-to-left because each step prepends; the dedup of
    `_prompt_with_trigger` also collapses two LoRA that share a trigger, so a
    stack never emits the same token twice."""
    if trigger_words is None or isinstance(trigger_words, str):
        return _prompt_with_trigger(prompt, trigger_words)
    p = (prompt or '').strip()
    for t in reversed(list(trigger_words)):
        p = _prompt_with_trigger(p, t)
    return p


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

    ⚠️ Régression corrigée (bug found 2026-07-01) : un simple `startswith` faisait
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
    ne s'offre PAS les LoRA du voisin. Returns [{filename, label}] (forme LoraLoader).

    ⚠️ Le trigger est CANONICALISÉ via `lt._safe_trigger` (la MÊME fonction qui nomme
    le fichier côté entraînement/déploiement) avant le match : un trigger multi-mots
    ('raw test upscale') se déploie en 'lora_raw_test_upscale_…' (espaces → '_'), donc
    matcher le trigger brut avec espaces ne préfixait JAMAIS le nom sous-scoré et le
    dataset disparaissait silencieusement du sélecteur du Studio (bug 2026-07-17).
    Aucun consommateur de noms ne doit re-slugifier à la main : tous passent par
    `_safe_trigger`."""
    trigger = lt._safe_trigger(ds).lower()
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
            # Pass the dataset's REAL trigger (not the safe/lowercased match form) so
            # a multi-token trigger like `leg_behind` labels faithfully instead of
            # splitting into `leg · behind` (the deployed filename can't disambiguate
            # the trigger's own underscores from the field separators on its own).
            entry = {'filename': lora['filename'],
                     'label': format_trained_lora_label(
                         lora['filename'], fam,
                         trigger=getattr(ds, 'trigger_word', None)) or stem}
            # Discreet retrofit badge for a mislabelled deploy: read the file's
            # REAL arch and flag it when it contradicts the folder's family, so a
            # wrong-family checkpoint is visible in the picker (not silently no-op).
            _p = _resolve_lora_abs_path(lora['filename'])
            _detected = lt.detect_lora_arch(_p) if _p else None
            if lt.lora_arch_conflicts(_detected, fam):
                entry['arch_mismatch'] = _detected
                entry['arch_label'] = lt._LORA_ARCH_LABEL.get(_detected, _detected)
            out.append(entry)
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
def _comfyui_recovery_target() -> dict | None:
    """Return the exact Studio view owning the durable ComfyUI barrier.

    The barrier is global, so the dataset currently displayed by the UI may not
    be the one that needs recovery.  Only expose a navigation target when the
    durable owner, queue row and unfinished Studio cell still agree.
    """
    owner = queue_manager.get_comfyui_stalled_barrier()
    if not isinstance(owner, dict) or not isinstance(owner.get('job_id'), str):
        return None
    job_id = owner['job_id']
    queue_row = ImageGenerationQueue.query.filter_by(job_id=job_id, status='stalled').first()
    # lds-allow-bare-lora-test-query: resolved by job_id — a stalled ComfyUI job
    # must be matched to whatever row owns it, derived rows included.
    cells = (LoraTestImage.query
             .filter_by(job_id=job_id, status='pending')
             .filter(LoraTestImage.filename.is_(None)).limit(2).all())
    if queue_row is None or len(cells) != 1:
        return None
    cell = cells[0]
    if (('dataset_id' in owner and owner.get('dataset_id') != str(cell.dataset_id))
            or ('run_id' in owner and owner.get('run_id') != cell.run_id)
            or ('cell_id' in owner and owner.get('cell_id') != str(cell.id))):
        return None
    kind = owner.get('kind', 'prompt')
    if kind == 'unknown_submit':
        if queue_row.comfyui_prompt_id is not None or owner.get('prompt_id') is not None:
            return None
    elif (kind != 'prompt'
          or str(queue_row.comfyui_prompt_id or '') != str(owner.get('prompt_id') or '')):
        return None
    return {
        'dataset_id': cell.dataset_id,
        'run_id': cell.run_id,
        'family': family_of_lora(cell.checkpoint) or 'zimage',
        'kind': kind,
    }


def gpu_busy_reason() -> str | None:
    """Return a human error when the GPU is held by a long-running exclusive
    task (LoRA training / vision pass), else None. The queue itself serializes
    normal generations, so no further locking is needed."""
    if queue_manager._get_system_state('training_in_progress', False):
        return "LoRA training in progress - the studio is unavailable (GPU busy)."
    if queue_manager._get_system_state('vision_in_progress', False):
        return "Vision pass in progress (GPU busy) - try again in a moment."
    if queue_manager.has_comfyui_stalled_barrier():
        target = _comfyui_recovery_target()
        if target and target['kind'] == 'unknown_submit':
            return ('A Test Studio image is paused because its ComfyUI submission outcome is unknown. '
                    'Restart ComfyUI, open the paused test, confirm the restart, then resume it.')
        return ('A Test Studio image is paused because ComfyUI stopped answering. '
                'Open the paused test, click Stop to recover it, then Resume.')
    return None


def _active_run_count(dataset_id=None) -> int:
    """In-flight cells (pending, no file yet). dataset_id=None → garde GLOBALE
    (tous datasets confondus, ce qu'exige une comparaison multi-LoRA) ; fourni →
    une seule run active par dataset (comportement historique)."""
    q = (_cells()
         .filter_by(status='pending')
         .filter(LoraTestImage.filename.is_(None)))
    if dataset_id is not None:
        q = q.filter_by(dataset_id=dataset_id)
    return q.count()


def _queue_activity(rows) -> dict:
    """Real queue state for the in-flight Test Studio cells in ``rows``.

    ``LoraTestImage.status`` intentionally stays ``pending`` until the output
    callback links a file, so it cannot distinguish waiting from GPU work. The
    linked ``ImageGenerationQueue`` row can. ``pending`` remains the historical
    total of unfinished cells for API compatibility; ``queued`` and
    ``generating`` split that total during the normal queue lifecycle.
    """
    live = [r for r in rows if r.status == 'pending' and not r.filename]
    job_ids = {r.job_id for r in live if r.job_id}
    queue_rows = (ImageGenerationQueue.query
                  .filter(ImageGenerationQueue.job_id.in_(job_ids)).all()
                  if job_ids else [])
    queue_by_job = {q.job_id: q for q in queue_rows}
    raw_by_job = {job_id: q.status for job_id, q in queue_by_job.items()}

    def _display_status(raw):
        if raw == 'pending':
            return 'queued'
        if raw in ('processing', 'sent_to_comfy'):
            return 'generating'
        if raw in ('cancel_requested', 'stalled'):
            return 'stalled'
        return raw

    queue_status = {job_id: _display_status(status)
                    for job_id, status in raw_by_job.items()}
    queue_error = {
        job_id: q.error_message for job_id, q in queue_by_job.items()
        if q.status == 'stalled' and isinstance(q.error_message, str) and q.error_message.strip()
    }
    queued = sum(1 for r in live if raw_by_job.get(r.job_id) == 'pending')
    generating = sum(1 for r in live
                     if raw_by_job.get(r.job_id) in ('processing', 'sent_to_comfy'))
    return {
        'pending': len(live),
        'queued': queued,
        'generating': generating,
        # Alias for consumers that use queue terminology rather than UI copy.
        'running': generating,
        'queue_status': queue_status,
        'queue_error': queue_error,
    }


def _unknown_submit_recovery(rows, activity):
    """UI-safe recovery metadata for the one exact paused Studio cell.

    A generic stalled tile may still have a known ComfyUI prompt and must use
    remote reconciliation instead. Expose this action only when the durable raw
    barrier, queue state, and linked cell all agree on an unknown submission.
    """
    owner = queue_manager.get_comfyui_stalled_barrier()
    if (owner is None or owner.get('kind') != 'unknown_submit'
            or owner.get('prompt_id') is not None
            or not isinstance(owner.get('job_id'), str)):
        return None
    job_id = owner['job_id']
    matching = [row for row in rows
                if row.status == 'pending' and not row.filename and row.job_id == job_id]
    if len(matching) != 1 or activity['queue_status'].get(job_id) != 'stalled':
        return None
    cell = matching[0]
    try:
        cell_id = int(owner.get('cell_id'))
    except (TypeError, ValueError):
        return None
    if str(cell_id) != owner.get('cell_id') or cell.id != cell_id:
        return None
    if (('dataset_id' in owner and owner.get('dataset_id') != str(cell.dataset_id))
            or ('run_id' in owner and owner.get('run_id') != cell.run_id)):
        return None
    return {
        'required': True,
        'kind': 'unknown_submit',
        'job_id': job_id,
        'cell_id': cell.id,
        'requires_comfyui_restart_confirmation': True,
    }


def build_matrix(checkpoints, strengths, aspects=None, cfgs=None, steps_list=None, steps2_list=None) -> list[tuple]:
    """Materialize the (checkpoint, strength, aspect) grid cells, validated:
    non-empty checkpoint/strength axes, strengths in [MIN_LORA_STRENGTH,
    MAX_LORA_STRENGTH] (0 = base model /
    LoRA off, a valid control column; above 2.0 = over-cook / breaking-point range,
    behind the « + » disclosure in the UI; NEGATIVE = the LoRA pulled the other
    way — the whole point of a slider LoRA, and a legit probe for any LoRA —
    behind a symmetric « − » disclosure) (deduped, order
    kept), aspects within the whitelist (deduped, défaut 9:16). PAS de plafond sur
    le nombre de cellules : la file est sérielle et l'utilisateur voit le compte +
    l'estimation de durée avant de lancer (choix assumé sur sa propre machine).

    ⚠️ Ce plafond n'est PAS seulement celui de l'axe de balayage. En mode 🧬 Blend,
    le poids de TÊTE de chaque combinaison passe par ici (cf. create_comparison_run :
    `combo_strengths = [combo[0]]`) et atterrit dans la même colonne
    `LoraTestImage.strength`. Un plafond de blend plus haut que celui-ci ne
    donnerait donc pas un rendu clampé mais un run REFUSÉ — c'est pour ça qu'il
    n'y a qu'un nombre, ici, et que COMBINE_MAX_WEIGHT le réutilise."""
    cps = [c for c in (checkpoints or []) if isinstance(c, str) and c.strip()]
    sts = []
    for s in (strengths or []):
        try:
            v = round(float(s), 2)
        except (TypeError, ValueError):
            raise ValueError(f'invalid strength: {s!r}')
        if not MIN_LORA_STRENGTH <= v <= MAX_LORA_STRENGTH:
            raise ValueError(
                f'strength out of range [{MIN_LORA_STRENGTH}, {MAX_LORA_STRENGTH}]: {v}')
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


def _krea_zero_strength_first(items, run_family, strength_of) -> list:
    """Stable-partition Krea around controls with the tested LoRA switched off.

    Moving every exact-zero tested-LoRA cell before its non-zero counterparts
    avoids returning to a tested-LoRA-free graph after Krea has begun loading
    the tested LoRAs. Both partitions keep their original order
    (base-model/checkpoint/aspect axes included), negative strengths remain in
    the non-zero partition, and other model families are deliberately left
    byte-for-byte ordered as before. Permanent/batch LoRAs remain untouched and
    may still patch a zero-strength control.
    """
    planned = list(items)
    if run_family != 'krea':
        return planned
    zero = []
    nonzero = []
    for item in planned:
        (zero if strength_of(item) == 0.0 else nonzero).append(item)
    return zero + nonzero


# --- « 🔎 Describe » : image → TEST PROMPT via le modèle vision Ollama ---------
# Upload guard (l'endpoint plafonne aussi, ceci est la borne de service).
STUDIO_DESCRIBE_MAX_BYTES = 20 * 1024 * 1024

# Variante « prompt de génération » du captioning prose (CAPTION_PROMPT) : décrit la
# scène/pose/cadrage/tenue directement, SANS identité (le LoRA la porte) et SANS trigger
# word (le Studio l'injecte séparément dans le workflow — le prompt stocké reste brut).
STUDIO_DESCRIBE_PROMPT = (
    "You are writing a TEXT-TO-IMAGE GENERATION PROMPT that would recreate this image.\n\n"
    "ABSOLUTE RULE - never describe WHO the person is. Do not mention identity or any "
    "identity-fixing trait: hair (its length, colour, style, texture), face shape, facial "
    "features, eye colour, eyebrows, nose, lips, jawline, skin tone or texture, freckles, "
    "age, gender, or ethnicity. Refer to a person only as \"the subject\". Do not invent a "
    "name and do not add any trigger word or token.\n\n"
    "DO describe, the way a prompt would: the shot type and framing (close-up, "
    "three-quarter, full-body, wide), the pose and body position, the expression and gaze "
    "as a state (smiling, looking at the viewer, eyes closed), the clothing and accessories "
    "with their colours, the setting or location, and the lighting and mood.\n\n"
    "Output ONE compact paragraph of plain natural-language prose, ready to paste as a "
    "generation prompt, beginning with the shot type and framing. Output only the prompt "
    "itself - no preamble, no \"Here is\", no quotation marks, no commentary.")


def describe_test_prompt(image_bytes: bytes) -> str:
    """Describe an uploaded image into a ready-to-paste Studio TEST PROMPT via the
    Ollama vision model (the same abliterated Qwen3-VL the app captions with, so NSFW
    passes). Resizes to <=1024 long side (like captioning) before the call, force-starts
    a stopped LOCAL Ollama. Whether the model stays resident afterwards is decided by
    CONTENTION (services/vision_keepalive.py): with a generation queued or a training
    running, ComfyUI gets its VRAM back immediately, exactly as before; on an otherwise
    idle card the model is leased warm so describing several images in a row doesn't pay
    the 12.8 s cold load every time. The lease is revoked the moment the queue picks up
    a job.

    Raises ValueError on a missing / oversized / unreadable (non-image) upload, and
    RuntimeError when Ollama is unavailable or rejects the request (its own reason is
    carried straight through via describe_image_ollama's auto_start_local path)."""
    if not image_bytes:
        raise ValueError('no image provided')
    if len(image_bytes) > STUDIO_DESCRIBE_MAX_BYTES:
        raise ValueError(f'image too large (max {STUDIO_DESCRIBE_MAX_BYTES // (1024 * 1024)} MB)')
    try:
        webp = fds.normalize_to_webp(image_bytes, size=1024)
    except Exception as e:
        raise ValueError('unreadable image — expected a webp, png or jpg file') from e
    # The /describe-image route owns the one GPU-exclusive Vision window. Keep
    # this service callable without recursively claiming it a second time.
    from .vision_ollama import describe_image_ollama
    from .vision_keepalive import keep_alive_for_isolated_call
    text = describe_image_ollama(
        webp, STUDIO_DESCRIBE_PROMPT, num_predict=500, auto_start_local=True,
        keep_alive=keep_alive_for_isolated_call())
    text = (text or '').strip().strip('"').strip()
    if not text:
        raise RuntimeError(
            'The vision model returned an empty description — check the configured '
            'vision model in Settings and the application log.')
    return text


STUDIO_ENHANCE_MAX_CHARS = 4000
# Enrichissement d'un prompt de test (texte→texte, PAS de vision). Contraintes dures :
# ne rien inventer sur l'IDENTITÉ (le LoRA la porte) et ne PAS toucher au trigger word
# (le Studio l'injecte au montage — un trigger recopié/déformé par le LLM créerait un
# doublon ou un token mort).
STUDIO_ENHANCE_PROMPT = (
    "You are rewriting a TEXT-TO-IMAGE GENERATION PROMPT so it renders better.\n\n"
    "Keep every subject, action, clothing item, setting and camera choice the author "
    "already wrote — you enrich, you do not replace. Add what a good prompt states and "
    "this one leaves out: shot type and framing, pose, lighting, background, mood, "
    "lens/photographic quality.\n\n"
    "ABSOLUTE RULES - do not describe WHO the person is: no hair, face, eyes, skin, age, "
    "gender or ethnicity (a LoRA supplies the identity). Do not add, repeat, translate or "
    "invent any trigger word, token or name. Do not add negatives, weights, "
    "parentheses-emphasis or LoRA tags.\n\n"
    "Output ONE compact paragraph of plain natural-language prose, ready to paste. Output "
    "only the prompt itself - no preamble, no \"Here is\", no quotation marks, no "
    "commentary.\n\n"
    "PROMPT TO ENHANCE:\n{prompt}")


def enhance_test_prompt(prompt: str) -> str:
    """Enrich a Studio test prompt with the LOCAL Ollama text model — the same
    abliterated model the app captions with (a vanilla model refuses the NSFW prompts
    this app produces), through the SAME client as captioning (`vision_ollama`); no
    second Ollama seam exists.

    A stopped LOCAL Ollama is started on demand, exactly like Describe. Whether the
    model stays resident afterwards is decided by contention (vision_keepalive), so
    enhancing three prompts in a row doesn't pay the cold load three times.

    Raises ValueError on an empty/oversized prompt, and RuntimeError when Ollama is
    unreachable, has no usable model, or answers nothing — the caller maps those to
    400 / 409 so the button never fails silently on an install without Ollama."""
    p = (prompt or '').strip()
    if not p:
        raise ValueError('write a prompt first — there is nothing to enhance')
    if len(p) > STUDIO_ENHANCE_MAX_CHARS:
        raise ValueError(f'prompt too long to enhance (max {STUDIO_ENHANCE_MAX_CHARS} characters)')
    from .ollama_control import ensure_captioning_ready
    from .vision_keepalive import keep_alive_for_isolated_call
    from .vision_ollama import generate_text_ollama
    ready = ensure_captioning_ready()
    if not ready.get('ok'):
        raise RuntimeError(
            (ready.get('error') or 'Ollama is unavailable')
            + ' — Enhance needs the local Ollama model configured in Settings › Local tools.')
    text = generate_text_ollama(STUDIO_ENHANCE_PROMPT.format(prompt=p), num_predict=500,
                                keep_alive=keep_alive_for_isolated_call(), strict=True)
    text = (text or '').strip().strip('"').strip()
    if not text:
        raise RuntimeError(
            'The model returned an empty prompt — check the configured Ollama model in '
            'Settings and the application log.')
    return text


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


def _resolve_lora_rel_by_basename(basename):
    """Loras-root-relative name (the exact string a LoraLoader wants) of the FIRST
    file whose basename matches `basename` across every loras search root, or None.
    Lets a workflow-wired accelerator LoRA be found WHEREVER the user keeps it —
    root, a differently-named subfolder, an extra_model_paths loras root — instead of
    depending on the developer's own subfolder. Reuses the same disk view as the
    picker/probe (comfy_model_paths.list_models), so anything the app lists is
    resolvable here; [] with no ComfyUI configured → None."""
    from . import comfy_model_paths
    target = (basename or '').lower()
    if not target:
        return None
    for rel, _ab in comfy_model_paths.list_models('loras'):
        if os.path.basename(rel).lower() == target:
            return rel
    return None


def _bypass_lora_loader(workflow, node_id):
    """Delete a two-output LoraLoader (model + clip) and reconnect each consumer of
    its model output (slot 0) / clip output (slot 1) to that node's OWN upstream model
    / clip inputs, so ComfyUI never fails validation on the missing LoRA. The two-slot
    form of klein_edit_helper._bypass_node (which handles a single model output)."""
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        return
    upstream = {0: node.get('inputs', {}).get('model'),
                1: node.get('inputs', {}).get('clip')}
    for other in workflow.values():
        if not isinstance(other, dict):
            continue
        for k, v in list(other.get('inputs', {}).items()):
            if (isinstance(v, list) and len(v) == 2 and v[0] == node_id
                    and upstream.get(v[1]) is not None):
                other['inputs'][k] = upstream[v[1]]
    workflow.pop(node_id, None)


def _apply_sdxl_accelerator(workflow):
    """Make the SDXL HQ workflow's DMD2 accelerator LoRA independent of the dev's own
    ComfyUI layout. The template wires 'DMD2\\dmd2_sdxl_4step_lora_fp16.safetensors' (a
    personal subfolder) at strength 1.0 — a SPEED/quality accelerator, NOT a
    graph-critical asset like the base checkpoint / VAE / text encoder. So:
      * resolve it by canonical basename across every loras root (a user who keeps the
        public DMD2 LoRA under any other folder still gets it wired — mission's #1
        preference), and
      * BYPASS the loader when it is absent EVERYWHERE, so a fresh SDXL Studio degrades
        to a plain render instead of hard-blocking the whole family on a file that only
        exists on the dev's disk (mirrors the Klein node-139 bypass).
    Distilled base checkpoints (the workflow's design point) render unchanged without
    it; a full SDXL checkpoint renders softer — the honest trade-off vs. a blocked grid.
    Idempotent and shape-agnostic: matches the DMD2 loader by `lora_name`, not node id."""
    for nid, node in list(workflow.items()):
        if not isinstance(node, dict) or node.get('class_type') != 'LoraLoader':
            continue
        ref = str(node.get('inputs', {}).get('lora_name') or '')
        if 'dmd2' not in ref.lower():
            continue
        # Fast path: the wired path is already on disk (dev, or anyone who placed it
        # there) → leave it, skip the loras walk that a grid would repeat per cell.
        if _resolve_lora_abs_path(ref):
            break
        found = _resolve_lora_rel_by_basename(os.path.basename(ref.replace('\\', '/')))
        if found:
            node['inputs']['lora_name'] = found
        else:
            _bypass_lora_loader(workflow, nid)
        break
    return workflow


# WHAT THE « Official » ENTRY LOADS, AND WHY IT IS NO LONGER A FILENAME
# ---------------------------------------------------------------------
# It used to be one: `krea2_turbo_fp8.safetensors`, the basename frozen into
# krea2_turbo.json's node 20 and repeated here as a constant. Two defects.
#
#   * It is NOT the file Setup installs. `setup_installer` fetches Comfy-Org's own
#     `krea2_turbo_fp8_scaled.safetensors`; ComfyUI validates a loader widget by
#     exact string match against the list it publishes, and those two names are
#     not the same string. So on an install that simply followed Setup, the
#     Studio's default base named a file that is not there and the whole prompt
#     was refused ("Value not in list: unet_name") before a step ran. It worked
#     only on machines that happened to have that community repack.
#   * That repack carries tensors this family does not declare. Measured on the
#     real header: 432 tensors against the family's 430, the two extras being
#     `last.down.weight` / `last.up.weight` `[6144, 6144]`, which its own
#     `__metadata__` (`egg_format`, `egg_w`, `egg_h`, `egg_c`) describes as an
#     embedded image — ~75 MB of picture shipped inside a base model.
#
# So the default is ELECTED from what is on disk, by the ranking documented on
# `krea_edit_helper.resolve_krea_unet` and shared with it — the Generate resolver
# and this picker must never elect different files out of the same folder.

def krea_default_base():
    """ComfyUI-relative name of the base the « Official » entry loads, or None
    when nothing on disk qualifies — then node 20 keeps the workflow's own value,
    which is the historical behaviour and the case the missing-asset preflight
    already owns.

    Elected out of `get_krea_models()`, the very list this screen offers: ranking
    a name the picker does not list would elect a base its own whitelist refuses."""
    try:
        from .krea_edit_helper import elect_krea_base
        return elect_krea_base(get_krea_models())
    except Exception:                       # noqa: BLE001 — never fatal to a render
        logger.exception('Krea default base election failed')
        return None


def krea_default_base_entry() -> dict:
    """The « Official » row of the Krea base pickers: ``{value, label, note, source}``.

    ``value`` stays ``''`` forever — it is persisted on run rows and read back, so
    it is an id, not a label. What CHANGES with the disk is what the row says: the
    name stays "Official" only while the elected base IS the file Setup installs.
    Anything else is named for what it is, so a base nobody chose is never
    presented as the official one.

    ``source`` is the file that row will actually load (None when node 20 keeps the
    workflow's own value). Callers need it because the sampler defaults belong to
    THAT file: a Raw build elected as the default must not be offered with the
    Turbo numbers — cfg 1 / 8 steps on an undistilled base renders a blurry sketch
    people read as a failed training (GitHub #18, bobba84, on the Z-Image side)."""
    from .krea_edit_helper import KREA_ASSETS, KREA_CANONICAL_UNET
    entry = {'value': '', 'label': 'Official – Krea 2 Turbo', 'note': None,
             'source': None}
    elected = krea_default_base()
    if not elected:
        return entry
    entry['source'] = elected
    bare = _basename(elected)
    if bare.lower() == KREA_CANONICAL_UNET.lower():
        return entry
    entry['label'] = f'Default – {bare.rsplit(".", 1)[0]}'
    notes = [f'The Krea 2 Turbo base Setup installs ({KREA_CANONICAL_UNET}) is not '
             f'on this machine, so the Studio renders on the best Krea 2 build it '
             f'found here: {bare}.']
    health = _krea_base_health(elected)
    if health and health.get('note'):
        notes.append(health['note'])
    # A note that only DIAGNOSES leaves the reader with a fact and no gesture, so
    # it ends on the action and on where the file lands — the same folder every
    # other Krea message names.
    notes.append(f'To render on the official base instead: Setup ▸ Install ▸ Krea 2 '
                 f'downloads {KREA_CANONICAL_UNET} into '
                 f'{KREA_ASSETS["krea_model"]["path"]}, and this entry goes back to '
                 f'it on its own — nothing else to change.')
    entry['note'] = ' '.join(notes)
    return entry


def _krea_base_health(rel_name):
    """`model_integrity.base_health` for an elected base, but ONLY when the verdict
    is worth a sentence — a file that announces it carries something other than
    weights. A plain fp8 cast is a normal thing to render on and says nothing here
    (the TRAINING picker is where precision earns a warning). None when the file
    cannot be resolved or read."""
    try:
        from . import comfy_model_paths, model_integrity
        path = comfy_model_paths.resolve_model_file('diffusion_models', rel_name)
        if not path:
            return None
        health = model_integrity.base_health(path)
        return health if health['rank'] == model_integrity.HEALTH_FOREIGN_PAYLOAD else None
    except Exception:                       # noqa: BLE001 — advisory only
        return None


def krea_alt_base_models() -> list:
    """Bases Krea locales ALTERNATIVES à celle de l'entrée « Official » : les
    checkpoints trouvés par get_krea_models() moins le défaut élu. Vide → aucun
    choix à offrir (les sélecteurs restent cachés, comportement historique).

    L'exclusion se fait sur le BASENAME : la même base recopiée à la racine ET
    dans un sous-dossier ne doit pas apparaître deux fois sous deux libellés."""
    default = krea_default_base()
    bare = _basename(default).lower() if default else None
    return [m for m in get_krea_models() if not bare or _basename(m).lower() != bare]


def apply_krea_lora_test_settings(workflow, *, lora_name, strength, prompt, seed,
                                  width, height, cfg=None, steps=None, batch_size=1,
                                  filename_prefix=None, allowed_loras=None, extra_loras=None,
                                  rebalance=None, sampler=None, scheduler=None,
                                  weight_dtype=None, enhancer_strength=None,
                                  base_model=None, allowed_bases=None):
    """Configure une cellule de test sur le workflow Krea 2 Turbo : le LoRA testé est
    injecté après le UNETLoader (node 20 → KSampler node 26), + prompt/seed/dims/steps/cfg.
    `extra_loras` = LoRA « always-on » (style/utilitaire) chaînés EN PLUS dans le même
    maillon (appliqués tels quels à cette cellule, hors batch). Krea est MONO-passe (pas
    de steps2).

    `rebalance` (node 30, NSFW/texture rebalance) - même sémantique que la génération
    (routes.py) : None = on NE touche PAS le node, défaut ON du workflow ; ≤1.0 = OFF
    (multiplier=1.0 + per_layer_weights neutres → passthrough SFW) ; >1.0 = ON à cette
    force (clampé 1..8). Mutate en place. Lève ValueError si le LoRA testé n'est pas dans
    sa whitelist (anti path-injection).

    `base_model` : UNET Krea local à charger dans le node 20 à la place du défaut
    câblé du workflow — même mécanique de base que SDXL (`base_ckpt`) / Z-Image
    (`z_model`). None = on ne touche pas au node (défaut). Validé contre
    `allowed_bases` (anti path-injection, comme le LoRA)."""
    if allowed_loras is not None and lora_name not in allowed_loras:
        raise ValueError(f"unknown Krea LoRA: {lora_name}")
    if base_model and allowed_bases is not None and base_model not in allowed_bases:
        raise ValueError(f"unknown Krea base model: {base_model}")

    def _set(node_id, key, value):
        n = workflow.get(node_id)
        if isinstance(n, dict) and key in n.get("inputs", {}):
            n["inputs"][key] = value

    if base_model:
        _set("20", "unet_name", base_model)
    else:
        # « Official » : elect the base rather than trust the filename frozen into
        # the workflow JSON — see krea_default_base for what that literal actually
        # named. Server-elected, so it does not go through `allowed_bases` (that
        # whitelist guards a USER-supplied value against path injection). None →
        # node untouched, exactly as before.
        elected = krea_default_base()
        if elected:
            _set("20", "unet_name", elected)

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
    # Node 30 (ConditioningKrea2Rebalance) : reweight des taps de conditioning Qwen3-VL.
    # ON (>1) relève les taps filtrés-sécurité → sortie non censurée + peau moins « plastique » ;
    # OFF (≤1) = passthrough identité (SFW). None = laisser le défaut du workflow (ON 4.0).
    # Le workflow épingle preset="custom" + renormalize=false sur le node : la classe
    # ConditioningKrea2Rebalance est publiée par DEUX packs (l'original nova452 à 3 inputs
    # ET le fork huwhitememes qui ajoute preset/renormalize). Sur le fork, preset défaut
    # "balanced" IGNORE per_layer_weights et renormalize=true annule l'effet de multiplier ;
    # "custom" + renormalize=false garantissent NOS poids/force quelle que soit l'install
    # (inputs inconnus = silencieusement ignorés par ComfyUI sur l'original).
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
    if allowed_loras is not None:
        # `allowed_loras` is a FAMILY-POOL scan (krea/ subfolder only): always-on
        # AND external (Canvas plugin node) entries in `extra_loras` were already
        # validated (path-injection / fail-closed) before reaching here, so this
        # whitelist must not re-filter them out — the same silent-drop bug the
        # Z-Image path guards against at `allowed_loras=(set(...) | {...})` above.
        # Without this union, `inject_krea_loras` below drops every extra whose
        # filename lives outside krea/ with NO error: persisted on the cell's
        # JSON, never mounted in the graph.
        allowed |= {r["filename"] for r in requested[1:]}
    inject_krea_loras(workflow, requested, allowed=allowed)
    # Krea2T-Enhancer (patcher texte-adhérence) injecté APRÈS les LoRA (wire-aware :
    # se branche sur ce qui alimente KSampler.model). enhancer_strength None = OFF ;
    # sinon ON à cette force (clampée 0..2 dans inject_krea2t_enhancer).
    if enhancer_strength is not None:
        inject_krea2t_enhancer(workflow, True, enhancer_strength)


# --- Node-class resolution (variant-tolerant custom nodes) --------------------
# Some ComfyUI custom nodes register under DIFFERENT class names across installs
# (a pack rename, a fork, a locally-edited copy). Our workflow JSON can only carry
# ONE class string, so a target ComfyUI that has the node under another name would
# fail the preflight (409 "install pack X") AND fail every tile if enqueued — even
# though the capability is right there. NODE_CLASS_ALIASES maps the CANONICAL class
# (the exact string our templates carry) to the alternative name(s) the SAME node
# is known to register as. Consumed by BOTH the preflight (a required class counts
# as present when the canonical OR any alias is in /object_info) and the cell builder
# (rewrites a node's class_type to whichever variant the target actually exposes, so
# the enqueued graph validates). Add an entry only for a node we have SEEN register
# under two names — never a speculative alias.
NODE_CLASS_ALIASES = {
    # Krea 2 "conditioning rebalance" (node 30 of krea2_turbo*.json). The published
    # pack (nova452/ComfyUI-Conditioning-Rebalance) registers it as
    # ConditioningKrea2Rebalance — the name our templates carry — but some installs,
    # incl. the dev's own (the origin of the permuted name we first shipped), register
    # the very same node as Krea2RebalanceConditioning.
    'ConditioningKrea2Rebalance': ('Krea2RebalanceConditioning',),
}


def _node_class_present(class_type, available):
    """True when `available` (a set of /object_info class names) exposes `class_type`
    OR any of its known aliases (NODE_CLASS_ALIASES). Presence only — the caller
    decides what a miss means; pass a real set (never None) so 'probe failed' stays a
    separate, fail-open decision at the call site."""
    if class_type in available:
        return True
    return any(alt in available for alt in NODE_CLASS_ALIASES.get(class_type, ()))


def _resolve_workflow_node_classes(workflow, available):
    """Rewrite each node whose CANONICAL class_type is absent from `available` but a
    known ALIAS is present, to that alias — so the ENQUEUED graph names the class the
    target ComfyUI actually registers (the resolver philosophy, applied to node
    classes). No-op when `available` is falsy (probe failed / not threaded → fail
    open: keep the canonical name, the preflight/per-tile path still reports a true
    miss) or when the canonical class is already present. Mutates in place; returns
    `workflow`."""
    if not available:
        return workflow
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type')
        aliases = NODE_CLASS_ALIASES.get(ct)
        if not aliases or ct in available:
            continue
        for alt in aliases:
            if alt in available:
                node['class_type'] = alt
                break
    return workflow


def _target_node_classes():
    """The target ComfyUI's /object_info class set, fetched ONCE per run so the grid's
    per-cell class resolution doesn't re-pull the (large) payload per tile. None when
    ComfyUI can't be reached — callers fail open (keep canonical names)."""
    from ..utils.comfyui import fetch_object_info_classes
    return fetch_object_info_classes()


def _build_cell_workflow(user_id, checkpoint, strength, prompt, seed, z_model,
                         allowed_loras, width=TEST_WIDTH, height=TEST_HEIGHT,
                         cfg=None, steps=None, steps2=None, dataset_id=None, train_type='zimage',
                         extra_loras=None, rebalance=None, negative=None, sampler=None,
                         scheduler=None, weight_dtype=None, enhancer_strength=None,
                         detail_amount=None, trigger_word=None, available_classes=None):
    """Load the ZTurbo (Z-Image) / HQ (SDXL) / Krea workflow and configure one grid cell.
    `extra_loras` = LoRA « always-on » (style/utilitaire) appliqués à CETTE cellule en plus
    du checkpoint testé (hors batch). `rebalance` = node 30 NSFW/texture (Krea uniquement,
    None ailleurs). Raises ValueError if the workflow file is unloadable.

    `available_classes` = the target ComfyUI's /object_info class set (from
    `_target_node_classes()`, fetched once per run). When provided, the built graph's
    variant custom-node classes are rewritten to whatever the install actually
    registers (NODE_CLASS_ALIASES) so the enqueued workflow validates on installs that
    carry a node under an alternative name; None = keep the canonical names (fail open).

    Le filename_prefix inclut le dataset_id ET un uuid court par cellule : sans
    ça, le compteur ComfyUI (qui repart de 0 à chaque restart) produisait des
    noms identiques entre datasets (`{uid}_LoraTest_00022_`) → collisions de
    cache navigateur et confusion visuelle entre LoRA (ex. images eva6938 vues
    dans le studio d'un autre LoRA). L'uuid garantit l'unicité même au sein d'un
    dataset (re-runs après restart ComfyUI)."""
    # Trigger word auto-injecté ICI (montage seul) - le prompt reste brut en base.
    # `trigger_word` peut être une LISTE (combine : un trigger par LoRA de la pile).
    prompt = _prompt_with_triggers(prompt, trigger_word)
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
        # DMD2 accelerator (node 10): resolve across loras roots or bypass when absent,
        # so the SDXL Studio never depends on the dev's personal 'DMD2\' subfolder nor
        # hard-blocks on a quality-only accelerator. Runs LAST so it reads node 10's
        # real upstream after any always-on chaining.
        _apply_sdxl_accelerator(workflow)
        return _resolve_workflow_node_classes(workflow, available_classes)
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
            # Base Krea locale optionnelle (z_model, même canal que SDXL/Z-Image) ;
            # None = UNET câblé du workflow. Whitelist = scan disque (anti-injection).
            base_model=z_model, allowed_bases=set(get_krea_models()),
        )
        # Résolveur de classes (node 30 = ConditioningKrea2Rebalance) : si le ComfyUI
        # cible n'expose ce node QUE sous un nom permuté (ex. l'install du dev :
        # Krea2RebalanceConditioning), réécrire le class_type vers le nom réel pour que
        # le graphe enqueué valide. available_classes None = on garde le canonique.
        return _resolve_workflow_node_classes(workflow, available_classes)
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
    return _resolve_workflow_node_classes(workflow, available_classes)


def _enqueue_cell(user_id, dataset_id, workflow, prompt, job_id=None, commit=True,
                  *, cell_id=None, run_id=None) -> str:
    """Enqueue one serialized Test Studio cell with durable cell identity.

    ``job_id`` is minted before the cell insert. ``cell_id`` / ``run_id`` are
    deliberately copied into queue metadata once the cell has an id, so a paused
    ComfyUI prompt can be shown and recovered without guessing which grid tile it
    belongs to. ``commit=False`` retains the one-transaction cell + queue insert.
    """
    job_id = job_id or str(uuid.uuid4())
    metadata = {
        'model_name': 'zimage_lora_test',
        'is_lora_test': True,
        'dataset_id': dataset_id,
    }
    if cell_id is not None:
        metadata['cell_id'] = int(cell_id)
    if run_id:
        metadata['run_id'] = str(run_id)
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt=prompt, job_id=job_id,
                          metadata=metadata, commit=commit)
    return job_id


def _persist_and_enqueue_cell(img, user_id, dataset_id, prompt, build_workflow) -> str:
    """Insert ONE grid cell and its queue job in a SINGLE transaction, and return
    its job_id.

    Why one commit and not zero (a single commit for the whole grid): a grid is
    enqueued cell by cell and an enqueue failure at cell 20/50 must LEAVE the 19
    already-queued cells in the database — a batch commit would roll their rows back
    while their jobs stay in the queue (orphan jobs, ghost tiles). Why not three
    (the historical shape: insert row, enqueue, re-write row with its job_id): each
    commit takes SQLite's write lock, and a 50-cell grid firing 150 of them back to
    back is exactly the profile that starves a concurrent writer into
    'database is locked'.

    The workflow is built before taking the GPU arbiter: resolving files/nodes
    must not hold local GPU scheduling.  The arbiter is then acquired *before*
    the first cell write and retained through the cell + queue commit.  This
    keeps the only safe lock order (GPU_ARBITER_LOCK -> SQLite transaction) and
    prevents a recovery barrier from appearing after ``add_job(commit=False)``
    checked readiness but before this transaction becomes durable.

    On failure the half-built transaction is rolled back (dropping the queue row that
    may already have been staged) and the cell is re-inserted as 'failed' with the
    reason, so the caller's `raise` still surfaces a visible, explained tile."""
    job_id = str(uuid.uuid4())
    img.job_id = job_id

    try:
        workflow = build_workflow()
    except Exception as e:
        # Workflow construction has not touched the queue transaction.  Roll
        # back any read/autoflush side effect from a builder before recording
        # the explained failure marker.
        db.session.rollback()
        img.job_id = None
        img.status = 'failed'
        img.error = str(e)[:400] or 'enqueue failed'   # say WHY, not a mute red tile
        db.session.add(img)
        db.session.commit()
        raise

    with GPU_ARBITER_LOCK:
        db.session.add(img)
        try:
            # A flush is not a commit: it gives the queue metadata the exact cell id
            # while preserving the one-transaction insert invariant below.
            db.session.flush()
            _enqueue_cell(user_id, dataset_id, workflow, prompt, job_id=job_id,
                          commit=False, cell_id=img.id, run_id=img.run_id)
            db.session.commit()
        except Exception as e:
            # rollback expunges the pending cell + job rows; the cell object goes back
            # to transient and can be re-added as the failed marker.  Keep the outer
            # arbiter through this replacement commit too: never acquire it after a
            # SQLite write transaction has begun.
            db.session.rollback()
            img.job_id = None
            img.status = 'failed'
            img.error = str(e)[:400] or 'enqueue failed'
            db.session.add(img)
            db.session.commit()
            raise
    return job_id


def _sanitize_gen_knobs(run_family, *, negative=None, sampler=None, scheduler=None,
                        weight_dtype=None, enhancer=None, enhancer_strength=None,
                        detail_amount=None, resolution_tier=None, resolution_multiplier=None,
                        init_image=None, denoise=None) -> dict:
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
    # Multiplicateur de résolution clampé [1.0, 1.9] (défaut 1.0). Ne s'applique qu'au
    # chemin par palier ; sans palier (table fixe) il reste 1.0 et n'a aucun effet.
    from ..utils.resolution import clamp_multiplier
    mult = clamp_multiplier(resolution_multiplier if resolution_multiplier is not None else 1.0)
    den = None
    if fam == 'krea' and denoise is not None:
        try:
            den = max(0.05, min(1.0, float(denoise)))
        except (TypeError, ValueError):
            den = None
    ini = ((init_image or '').strip() or None) if fam == 'krea' else None
    return {'negative': neg, 'sampler': smp, 'scheduler': sch, 'weight_dtype': wdt,
            'enhancer_strength': enh, 'detail_amount': dta, 'resolution_tier': tier,
            'resolution_multiplier': mult, 'init_image': ini, 'denoise': den}


# --- Studio preflight (model files on disk + custom nodes in ComfyUI) ---------
# Klein already preflights its assets (KleinModelsMissing → 409 + auto-download);
# Krea/SDXL/Z-Image did NOT — the studio workflows hardcode the developer's own
# VAE / text-encoder / accelerator-LoRA names (none of which exist on
# a fresh install), so a fresh user launched a grid and every tile failed ComfyUI
# validation SILENTLY (empty grid, no reason). This block gives each family the
# same up-front check: verify (a) every model file the BUILT workflow references
# is on disk (via the exact filenames the workflow will send — zero divergence),
# and (b) every custom node the workflow uses exists in the target ComfyUI
# (/object_info), and raises StudioAssetsMissing so the route answers ONE
# actionable 409 instead.

class StudioAssetsMissing(Exception):
    """A Studio family's workflow references model files not on disk, or custom
    nodes the target ComfyUI doesn't expose, so every grid tile would fail ComfyUI
    validation and land as a silently-empty cell. Raised BEFORE any row/job is
    created so the caller can answer one actionable 409 (same spirit as Klein's
    KleinModelsMissing).

    `.family` = pipeline key ('zimage'/'sdxl'/'krea'); `.missing_files` =
    [{path, kind}] with `path` a display path like 'models/vae/…'; `.missing_nodes`
    = [class_type]; `.invalid_files` = [{path, kind, reason}] for a referenced model
    that IS on disk but is NOT real, loadable weights (an HTML gate page saved as
    .safetensors, a truncated download) — the same silent-empty-tile failure as a
    missing file, but the fix is 'delete + re-download', not 'place the file'."""
    def __init__(self, family, missing_files, missing_nodes, invalid_files=None):
        self.family = family
        self.missing_files = list(missing_files)
        self.missing_nodes = list(missing_nodes)
        self.invalid_files = list(invalid_files or [])
        n_f, n_n, n_i = len(self.missing_files), len(self.missing_nodes), len(self.invalid_files)
        super().__init__(f'{family} studio assets missing: {n_f} file(s), '
                         f'{n_n} node(s), {n_i} invalid file(s)')


class StudioArchMismatch(Exception):
    """A selected checkpoint's REAL architecture (read from its safetensors header)
    contradicts the family whose pipeline the Studio would run it under. ComfyUI
    silently drops every incompatible LoRA key, so the entire grid renders as if
    the LoRA were off (strength 0) with no error anywhere — the 2026-07-13
    incident (a Z-Image LoRA mislabelled Krea produced 117 no-op tiles). Raised
    BEFORE any row/job is created so the caller answers one actionable 409 (same
    spirit as StudioAssetsMissing).

    `.family` = the Studio's pipeline key; `.detected` = the checkpoint's real
    family; `.checkpoint` = the LoraLoader-form path that mismatched."""
    def __init__(self, family, detected, checkpoint):
        self.family = family
        self.detected = detected
        self.checkpoint = checkpoint
        super().__init__(f'{checkpoint} is a {detected} LoRA, not {family}')


def _is_unsafe_external_lora_name(fn) -> bool:
    """True if `fn` could resolve OUTSIDE a loras root once handed to
    `_ci_resolve` (path traversal / drive-letter / rooted path). `os.path.isabs`
    alone is not enough: on Windows it is False for a POSIX-style rooted path
    like '/abs/x.safetensors' (no drive letter), yet `_ci_resolve` still walks
    it as a normal — if odd — first component, and `_resolve_lora_abs_path`
    would otherwise `lstrip(os.sep)` it into something that LOOKS validated.
    So every rooted form is rejected explicitly, not inferred from `isabs`."""
    s = str(fn or '')
    if not s or os.path.isabs(s) or s.startswith(('/', '\\')) or ':' in s:
        return True
    return any(part == '..' for part in s.replace('\\', '/').split('/'))


def _resolve_lora_abs_path(checkpoint) -> str | None:
    """Absolute path of a LoraLoader-form checkpoint ('<subfolder>\\name.safetensors',
    relative to models/loras), resolved case-INSENSITIVELY (the workflow paths
    carry mixed casing — 'z image', 'Krea' — and a case-sensitive cloud FS must
    still find the file). None when ComfyUI's loras dir isn't configured or the
    file can't be located.

    Searched across EVERY loras root in ComfyUI's own priority order (the yaml's
    included), like the loader node this path is handed to — a LoRA deployed into
    an extra_model_paths root, or into the old default one before GitHub #25, must
    resolve either way."""
    rel = str(checkpoint or '').replace('\\', os.sep).replace('/', os.sep).lstrip(os.sep)
    if not rel:
        return None
    from . import comfy_model_paths
    try:
        roots = comfy_model_paths.search_roots('loras')
    except Exception:
        roots = []
    for loras in roots:
        found = _ci_resolve(str(loras), rel)
        if found and os.path.isfile(found):
            return found
    return None


def _preflight_checkpoint_arch(run_family, checkpoints):
    """Raise StudioArchMismatch if any selected checkpoint's REAL arch (safetensors
    header) contradicts `run_family`. family_of_lora keys off the FOLDER, which is
    exactly the blind spot a mislabelled deploy exploits — so we read the header
    here. Undetectable/foreign headers pass (no false block); only a POSITIVE
    cross-namespace contradiction stops the run."""
    for cp in checkpoints:
        p = _resolve_lora_abs_path(cp)
        if not p:
            continue
        detected = lt.detect_lora_arch(p)
        if lt.lora_arch_conflicts(detected, run_family):
            raise StudioArchMismatch(run_family, detected, cp)


# ComfyUI loader class_type -> (input keys carrying a model FILENAME, the models/
# subfolders that loader lists files from, human kind). A loader lists files
# relative to ONE of these subfolders; the file counts as present if it resolves
# under any (a UNET lives in unet/ OR diffusion_models/ on shared installs). Only
# loaders the studio workflows actually use are mapped.
_STUDIO_MODEL_LOADERS = {
    'UNETLoader': (('unet_name',), ('unet', 'diffusion_models'), 'diffusion model'),
    'CheckpointLoaderSimple': (('ckpt_name',), ('checkpoints',), 'checkpoint'),
    'VAELoader': (('vae_name',), ('vae',), 'VAE'),
    'CLIPLoader': (('clip_name',), ('text_encoders', 'clip'), 'text encoder'),
    'DualCLIPLoader': (('clip_name1', 'clip_name2'), ('text_encoders', 'clip'), 'text encoder'),
    'LoraLoader': (('lora_name',), ('loras',), 'LoRA'),
    'LoraLoaderModelOnly': (('lora_name',), ('loras',), 'LoRA'),
}


def _models_root():
    try:
        d = cfg.comfyui_dir('models')
    except Exception:
        return None
    return str(d) if d else None


def _ci_resolve(root, rel):
    """The real absolute path of root/rel with each component matched
    case-INSENSITIVELY below `root`, or None when no such entry exists. ComfyUI on
    Windows is case-insensitive and the workflow templates carry mixed folder casing
    (node refs 'Z image\\…' / 'Krea\\…' vs the on-disk 'z image' / 'krea') — a
    case-sensitive filesystem (cloud) must NOT read those as missing. `root` is
    assumed to exist."""
    cur = root
    for part in rel.split(os.sep):
        if not part or part == '.':
            continue
        nxt = os.path.join(cur, part)
        if os.path.exists(nxt):
            cur = nxt
            continue
        try:
            match = next((e for e in os.listdir(cur) if e.lower() == part.lower()), None)
        except OSError:
            return None
        if match is None:
            return None
        cur = os.path.join(cur, match)
    return cur if os.path.exists(cur) else None


def _ci_join_exists(root, rel):
    """os.path.exists(root/rel) with each component matched case-INSENSITIVELY below
    `root` — the boolean form of _ci_resolve (see it for why casing is tolerated)."""
    return _ci_resolve(root, rel) is not None


def _resolve_model_abs(models_root, subfolders, ref):
    """Absolute path of a PRESENT loader ref (mirrors _model_file_present's search:
    models_root/<subfolder>/ then extra_model_paths roots, case-insensitive), or None
    when it isn't on disk. Lets the preflight read the exact file ComfyUI would open
    and check it is real weights, not just present."""
    rel_ref = (ref or '').replace('\\', os.sep).replace('/', os.sep).lstrip(os.sep)
    if not rel_ref:
        return None
    for sub in subfolders:
        p = _ci_resolve(models_root, os.path.join(sub, rel_ref))
        if p:
            return p
    try:
        from . import comfy_model_paths
        for sub in subfolders:
            for root in comfy_model_paths.extra_roots(sub):
                p = _ci_resolve(root, rel_ref)
                if p:
                    return p
    except Exception:
        pass
    return None


def _model_file_present(models_root, subfolders, ref):
    """True if `ref` (a loader value, possibly with its own subfolder prefix)
    resolves to a real file under models_root/<subfolder>/ for any candidate
    subfolder, OR under an extra_model_paths.yaml root for those types (where the
    extra dir IS the type root, so `ref` resolves directly beneath it). With no yaml
    only the base models_root is checked, so this is unchanged. This keeps the Studio
    preflight from raising a false 'missing file' 409 for a model that lives in an
    extra path — ComfyUI resolves it natively at run time from the same yaml."""
    rel_ref = (ref or '').replace('\\', os.sep).replace('/', os.sep).lstrip(os.sep)
    if not rel_ref:
        return True  # empty ref = loader left at a wired default upstream — not our miss
    if any(_ci_join_exists(models_root, os.path.join(sub, rel_ref)) for sub in subfolders):
        return True
    try:
        from . import comfy_model_paths
        for sub in subfolders:
            for root in comfy_model_paths.extra_roots(sub):
                if _ci_join_exists(root, rel_ref):
                    return True
    except Exception:
        pass
    return False


def _scan_workflow_assets(workflow, models_root):
    """(missing_files, invalid_files, class_types) for a BUILT cell workflow.
    missing_files = [{path, kind}] for every model-loader reference NOT on disk;
    invalid_files = [{path, kind, reason}] for a reference that IS on disk but is not
    real, loadable weights (an HTML gate page saved as .safetensors, a truncated
    download) — this would fail ComfyUI validation and leave every tile silently
    EMPTY, the same silent-failure class as a missing file, so the preflight owns it
    too. Both are skipped entirely when models_root is unknown (the base-pool guards
    already caught that case). class_types = every node class in the graph."""
    missing, invalid, classes = [], [], set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get('class_type')
        if not ct:
            continue
        classes.add(ct)
        spec = _STUDIO_MODEL_LOADERS.get(ct)
        if not (spec and models_root):
            continue
        keys, subfolders, kind = spec
        inputs = node.get('inputs', {}) if isinstance(node.get('inputs'), dict) else {}
        for k in keys:
            ref = inputs.get(k)
            if not isinstance(ref, str) or not ref.strip():
                continue
            display = f'models/{subfolders[0]}/{ref}'.replace('\\', '/')
            abs_path = _resolve_model_abs(models_root, subfolders, ref)
            if abs_path is None:
                entry = {'path': display, 'kind': kind}
                # A resolver that came up empty leaves `_meta.lds_missing_hint` on the
                # node saying WHAT it accepted and WHERE it looked (see
                # utils/zimage_helper._resolve_zimage_assets). Carrying it into the 409
                # is what keeps the preflight honest now that resolution is automatic:
                # "this file is missing" alone would suggest the exact name is required,
                # when a dozen spellings would have done.
                meta = node.get('_meta')
                hint = meta.get('lds_missing_hint') if isinstance(meta, dict) else None
                if hint:
                    entry['hint'] = str(hint)
                if entry not in missing:
                    missing.append(entry)
                continue
            # Present — but is it real weights? Only a BLOCKING verdict (an HTML/text
            # file or a truncated header) counts: those can't load at all. No size
            # floor here — a legitimately small Studio VAE/LoRA must not be flagged.
            from . import model_integrity
            verdict = model_integrity.validate_model_file(abs_path)
            if verdict['blocking']:
                entry = {'path': display, 'kind': kind, 'reason': verdict['reason']}
                if entry not in invalid:
                    invalid.append(entry)
    return missing, invalid, classes


def preflight_family(family, workflows):
    """Raise StudioAssetsMissing if the target ComfyUI is missing any model file or
    custom node the family's BUILT workflow(s) need, OR if a referenced model file is
    present but not real weights (an HTML gate page saved as .safetensors, a truncated
    download — it would fail ComfyUI validation and leave every tile silently empty).
    `workflows` = representative built cell workflow(s) (one per base) — checking the
    ACTUAL built graph means zero divergence from what will be enqueued. Best-effort:
    only raises on a CONCRETE absence/invalidity; a build that couldn't be produced or
    an unreachable /object_info fails OPEN (the per-tile error capture still surfaces
    the reason).
    """
    models_root = _models_root()
    missing_files, invalid_files, all_classes = [], [], set()
    for wf in workflows:
        if not wf:
            continue
        mf, inv, classes = _scan_workflow_assets(wf, models_root)
        for e in mf:
            if e not in missing_files:
                missing_files.append(e)
        for e in inv:
            if e not in invalid_files:
                invalid_files.append(e)
        all_classes |= classes
    # Custom nodes: compare the graph's class_types to /object_info. Fail-OPEN when
    # it can't be fetched (None) — never block on a transient probe failure. A class
    # counts as present when the target exposes it OR a known alias (a node registered
    # under a permuted/forked name is the SAME capability — cf. NODE_CLASS_ALIASES),
    # so we never 409 an install that has the node under an alternative class name.
    missing_nodes = []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is not None and all_classes:
        missing_nodes = sorted(c for c in all_classes if not _node_class_present(c, available))
    if missing_files or invalid_files or missing_nodes:
        raise StudioAssetsMissing(family, missing_files, missing_nodes, invalid_files)


# Custom-node class_types the Studio family workflows pull from community packs,
# mapped to the pack that ships each + how to find it in ComfyUI-Manager. Turns a
# bare "node X is missing" 409 into an actionable "install pack Y (search: Z), then
# restart ComfyUI". Same spirit as klein_edit_helper.KLEIN_NODE_PACKS; an unknown
# node simply gets no hint (the generic message still shows), never an error.
STUDIO_NODE_PACKS = {
    # Krea 2 Turbo rebalance (node 30 of krea2_turbo*.json). The class is published by
    # BOTH the original nova452/ComfyUI-Conditioning-Rebalance (3 inputs) and the
    # huwhitememes fork (adds preset/renormalize) under the same key, so preflight-by-
    # class accepts either; the workflow pins preset=custom + renormalize=false so our
    # per-layer weights win on the fork too.
    'ConditioningKrea2Rebalance': {
        'pack': 'ComfyUI-Conditioning-Rebalance',
        'url': 'https://github.com/nova452/ComfyUI-Conditioning-Rebalance',
        'search': 'Krea 2 Conditioning',
    },
}


def studio_missing_node_hints(nodes):
    """[{class_type, pack, url, search}] for each missing node class that maps to a
    known community pack (unknown classes omitted). Lets a studio_missing 409 name
    WHAT to install instead of only the bare class_type — same spirit as Klein's
    format_missing_nodes_message."""
    out = []
    for ct in nodes or []:
        pk = STUDIO_NODE_PACKS.get(ct)
        if pk:
            out.append({'class_type': ct, **pk})
    return out


def _preflight_run(user_id, run_family, checkpoint, bases, allowed, prompt, seed,
                   dataset_id, trigger_word):
    """Build a representative cell workflow for `run_family` (one per distinct base
    in `bases`) and run `preflight_family` on it. Raises StudioAssetsMissing when
    the target ComfyUI can't run the grid. A representative build that itself fails
    is skipped (the enqueue loop would surface that path's own error)."""
    wfs = []
    seen = set()
    for base in (bases or [None]):
        key = base or ''
        if key in seen:
            continue
        seen.add(key)
        try:
            wfs.append(_build_cell_workflow(
                user_id, checkpoint, 1.0, prompt or '', seed or 1, base, allowed,
                dataset_id=dataset_id, train_type=run_family, trigger_word=trigger_word))
        except Exception as e:  # noqa: BLE001 — a bad representative build ≠ a missing asset
            logger.warning('studio preflight: representative build failed (base=%r): %s', base, e)
    preflight_family(run_family, wfs)


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


# 📝 Un lot de prompts reste UN run. Deux lancements de suite sont refusés par le
# garde « a test run is already in progress » (et le GPU est sérialisé de toute
# façon) : le lot est donc un AXE, comme les formats ou les cfg — une cellule par
# prompt, mêmes checkpoints, mêmes réglages, même seed.
#
# ⚠️ AUCUN PLAFOND, et c'est la règle de CE module (cf. l'en-tête et build_matrix :
# « PAS de plafond sur le nombre de cellules : la file est sérielle et
# l'utilisateur voit le compte + l'estimation de durée avant de lancer »). Une
# première version de ce lot refusait au-delà de 24 prompts. Ce 24 était un
# jugement, pas une mesure : rien ne casse à 33 — le corps de requête pèse
# quelques kilo-octets contre 64 Mo autorisés, `prompt` est un TEXT sans
# longueur, la file n'a pas de profondeur maximale et aucune vue de résultats ne
# tronque. Le seul coût réel est le TEMPS GPU, et il se gouverne par un
# avertissement chiffré avant le clic — pas par un refus sur un axe pris au
# hasard parmi les six que le run multiplie.


def _prompt_axis(prompts, fallback) -> list:
    """L'axe 📝 prompt d'un run : la liste cochée, nettoyée et dédupliquée dans
    l'ordre d'arrivée ; vide → `[fallback]`, c'est-à-dire EXACTEMENT le
    comportement d'avant (un seul prompt, celui du champ). `fallback` peut être
    None quand l'appelant laisse chaque cellule retomber sur le prompt d'identité
    de son dataset (comparaison multi-datasets)."""
    seen, out = set(), []
    for p in (prompts or []):
        if not isinstance(p, str):
            continue
        s = p.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out or [fallback]


# --- Rythme MESURÉ de la machine ---------------------------------------------
# L'UI annonçait « ~12 s/image » en dur. C'est vrai sur une 4090 en Z-Image Turbo
# et faux partout ailleurs : sur une carte lente, un balayage annoncé « 20 min »
# en prend deux heures, et l'utilisateur ne l'apprend qu'en le vivant. Or la file
# enregistre `started_at`/`completed_at` de chaque job depuis toujours — le vrai
# chiffre était déjà là, personne ne le lisait.
_PACE_SCAN_ROWS = 200      # lignes de file relues au plus (bornées, index sur completed_at)
_PACE_SAMPLE_SIZE = 30     # échantillons retenus : assez pour une médiane stable
_PACE_MIN_SAMPLES = 3      # en-dessous, on ne prétend rien et l'UI garde son défaut
_PACE_MIN_SECONDS = 0.5    # garde-fou bas : une cellule « faite » en 0,1 s n'a rien rendu
_PACE_MAX_SECONDS = 900.0  # garde-fou haut : une machine mise en veille pendant un job
                           # signerait « 8 h par image » et ruinerait l'estimation
DEFAULT_SECONDS_PER_IMAGE = 12.0   # le repli historique, quand il n'y a pas d'historique


def measured_seconds_per_image(family=None) -> float | None:
    """La durée MÉDIANE d'une génération de test réellement observée ici.

    Médiane et non moyenne : un job resté coincé derrière un téléchargement de
    modèle tirerait une moyenne vers le haut pour les cent runs suivants.
    `family` restreint aux cellules de cette pipeline (une image Krea et une
    Z-Image Turbo ne coûtent pas la même chose) ; sans assez d'échantillons on
    renvoie None — l'appelant dit alors « ~ » avec son défaut plutôt que
    d'inventer un chiffre précis à partir de deux mesures."""
    try:
        rows = (db.session.query(ImageGenerationQueue.started_at,
                                 ImageGenerationQueue.completed_at,
                                 LoraTestImage.checkpoint)
                .join(LoraTestImage, LoraTestImage.job_id == ImageGenerationQueue.job_id)
                .filter(ImageGenerationQueue.status == 'completed',
                        ImageGenerationQueue.started_at.isnot(None),
                        ImageGenerationQueue.completed_at.isnot(None),
                        # ✨ An Upscale & improve job is NOT a test generation and
                        # its duration says nothing about the pace of a sweep: it
                        # is a 2 MP Klein edit or a SeedVR2 restoration, minutes
                        # where a Turbo cell takes seconds. It passes the family
                        # filter below (a derived row copies its source's
                        # `checkpoint`) and sits well inside the 0.5-900 s window,
                        # so with a 30-sample median a handful of them visibly
                        # inflate the duration this estimate promises before a
                        # launch. Same rule as _cells(), same predicate.
                        _is_cell())
                .order_by(ImageGenerationQueue.completed_at.desc())
                .limit(_PACE_SCAN_ROWS).all())
    except Exception:                      # base legacy sans l'une des colonnes
        logger.debug('pace: queue timings unreadable', exc_info=True)
        return None
    secs = []
    for started, completed, checkpoint in rows:
        if family and family_of_lora(checkpoint or '') not in (None, family):
            continue
        try:
            d = (completed - started).total_seconds()
        except (TypeError, AttributeError):
            continue
        if _PACE_MIN_SECONDS <= d <= _PACE_MAX_SECONDS:
            secs.append(d)
        if len(secs) >= _PACE_SAMPLE_SIZE:
            break
    if len(secs) < _PACE_MIN_SAMPLES:
        return None
    secs.sort()
    mid = len(secs) // 2
    median = secs[mid] if len(secs) % 2 else (secs[mid - 1] + secs[mid]) / 2
    return round(median, 1)


def checkpoint_origins(checkpoints, explicit=None) -> dict:
    """{deployed filename: (record_id, step)} — WHICH training checkpoint each
    selected LoRA came from, so every cell can record it on its row instead of
    the app re-deriving it from the filename on every render (the heuristic that
    already shipped a bug, see LoraTestImage.record_id).

    `explicit` is the mapping a caller that ALREADY knows the answer provides —
    the LoRA Canvas, where the user picked a lineage pill, so the run and the
    step are the identity of what was clicked. It always wins.

    Without it the origin is read back from the run tag the DEPLOY stamped into
    the name (`_rl<record>` / `_rc<cloud run>` + the zero-padded step): the Test
    Studio picks a filename out of a folder and has no other handle. That tag was
    written by the app, not inferred from a trigger word — and a name that
    carries none resolves to (None, None), i.e. an honestly unlinked cell.

    Resolved ONCE per distinct filename: a 40-cell grid over 6 checkpoints costs
    6 lookups."""
    out = {}
    for cp in checkpoints or []:
        if cp in out:
            continue
        hint = (explicit or {}).get(cp)
        if hint:
            try:
                out[cp] = (int(hint['record_id']), int(hint['step']))
                continue
            except (KeyError, TypeError, ValueError):
                pass                     # malformed hint → fall through to the tag
        try:
            from .checkpoint_link_backfill import resolve_checkpoint_name
            hit = resolve_checkpoint_name(cp)
        except Exception:                # a registry read must never fail a launch
            hit = None
        out[cp] = (hit[0], hit[1]) if hit else (None, None)
    return out


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


def _combined_lora_labels(row) -> list:
    """Noms lisibles des LoRA EMPILÉS avec celui de la cellule (entrées
    `combined:true` de son JSON extra_loras) — badge « + X » de la grille et de
    la lightbox. Liste vide quand la cellule n'est pas une pile.

    `filename`/`dataset_id`/`trigger` sont écrits par les runs lancés DEPUIS la vue
    pile ; le JSON d'une cellule est figé à sa création, donc les runs plus anciens
    n'ont que `label`/`weight` et ces clés valent None — la composition s'affiche
    alors sans trigger au lieu de disparaître.

    `record_id`/`step` — la PROVENANCE de génération du membre, c'est-à-dire la
    pastille du board dont il sort — suivent la même règle et la même raison :
    écrits depuis le run qui les connaissait, absents (None) sur tout ce qui a été
    lancé avant. Un membre sans origine n'est pas une erreur, c'est une pile plus
    ancienne, et le lecteur DOIT pouvoir faire la différence entre « pas de
    parent » et « parent inconnu » plutôt que d'en inventer un."""
    out = []
    try:
        for e in json.loads(row.extra_loras or '[]'):
            if isinstance(e, dict) and e.get('combined'):
                name = _basename(e.get('filename', '')).rsplit('.', 1)[0]
                out.append({'label': format_trained_lora_label(e.get('filename', '')) or name,
                            'weight': e.get('strength'),
                            'filename': e.get('filename') or None,
                            'dataset_id': e.get('dataset_id'),
                            'record_id': e.get('record_id'),
                            'step': e.get('step'),
                            'trigger': e.get('trigger') or None})
    except (ValueError, TypeError):
        pass
    return out


def stack_of_row(row) -> list | None:
    """Composition ORDONNÉE de la pile d'une cellule, ou None si ce n'en est pas une.

    Le LoRA de TÊTE est la cellule elle-même (son `checkpoint`, son poids = `strength` :
    create_comparison_run réduit l'axe strengths au poids de tête en mode combine) ; les
    suivants sont les entrées `combined:true`. Son trigger est relu du dataset — il n'est
    pas figé dans le JSON, contrairement à ceux des LoRA empilés."""
    combined = _combined_lora_labels(row)
    if not combined:
        return None
    ds = FaceDataset.query.get(row.dataset_id)
    head = {'label': (format_trained_lora_label(row.checkpoint)
                      or _basename(row.checkpoint or '').rsplit('.', 1)[0]),
            'weight': row.strength, 'filename': row.checkpoint,
            'dataset_id': row.dataset_id,
            # La tête porte SON origine depuis toujours, en colonnes : la cellule
            # est déjà rattachée à une pastille. Reprise ici pour que les membres
            # d'une pile se lisent tous de la même façon, tête comprise.
            'record_id': row.record_id, 'step': row.step,
            'trigger': (getattr(ds, 'trigger_word', None) or None) if ds else None,
            'head': True}
    return [head] + [{**c, 'head': False} for c in combined]


def _stack_signature(members) -> str:
    """Identité d'une pile INDÉPENDANTE de ses poids : ses fichiers, triés. Deux runs
    de même signature sont deux variantes de poids de la MÊME pile — c'est ce qui
    permet de les afficher côte à côte."""
    return '|'.join(sorted(str((m or {}).get('filename') or '') for m in (members or [])))


# Fenêtre de scan des variantes : on ne remonte pas tout l'historique du dataset pour
# retrouver les relances d'une pile. Un run de pile fait peu de cellules (1 × count ×
# batch), donc quelques centaines de lignes couvrent largement une session de réglage.
_STACK_SCAN_ROWS = 600


def stack_variants(run_id, rows, limit=8) -> list:
    """Les runs de la MÊME pile (mêmes LoRA, poids éventuellement différents), du plus
    récent au plus ancien, run courant compris et marqué `active`.

    Sert la comparaison « et si je mettais 0.6 au deuxième ? » : chaque variante porte
    son vecteur de poids, ses cellules (votables telles quelles : le vote est par id de
    cellule) et son bilan de votes. Limité à `limit` variantes et à `_STACK_SCAN_ROWS`
    lignes scannées — une pile relancée des dizaines de fois ne montre que les plus
    récentes, et une variante dont les cellules débordent la fenêtre s'affiche tronquée."""
    members = stack_of_row(rows[0]) if rows else None
    if not members:
        return []
    sig = _stack_signature(members)
    head_ds = members[0].get('dataset_id')
    scanned = (_cells()
               .filter(LoraTestImage.dataset_id == head_ds,
                       LoraTestImage.extra_loras.isnot(None))
               .order_by(LoraTestImage.id.desc()).limit(_STACK_SCAN_ROWS).all())
    # Regroupement par (run, VECTEUR DE POIDS) et non par run seul. Un run était
    # forcément une combinaison unique jusqu'au balayage 🧬 ; depuis, UN run porte
    # N combinaisons, et grouper par run seul les écraserait en une variante
    # unique étiquetée avec les poids de sa première cellule — un mensonge sur
    # l'image qu'on regarde. Avec un poids par LoRA le vecteur est constant sur
    # tout le run, donc le regroupement est exactement celui d'avant.
    def _weight_vector(row):
        comp = stack_of_row(row)
        return tuple((m.get('filename'), m.get('weight')) for m in (comp or []))

    groups = {}
    for r in scanned:
        if not r.run_id:
            continue
        groups.setdefault((r.run_id, _weight_vector(r)), []).append(r)
    # Le run courant ne dépend pas de la fenêtre de scan : ses combinaisons sont
    # réinjectées telles quelles, chacune sous sa propre clé.
    for r in rows:
        groups.setdefault((run_id, _weight_vector(r)), [])
        if r not in groups[(run_id, _weight_vector(r))]:
            groups[(run_id, _weight_vector(r))].append(r)

    out = []
    for (rid, _vector), grp in groups.items():
        # `limit` ne doit JAMAIS évincer le run affiché : ses colonnes sont celles
        # que l'utilisateur regarde — et un balayage en a plusieurs. Les autres
        # s'arrêtent au plafond.
        if len(out) >= limit and rid != run_id:
            continue
        # Les cellules sans run_id (colonne ajoutée après coup sur des bases legacy)
        # ne forment pas UN run : les agréger fabriquerait une variante fantôme dont
        # les images viennent de générations sans rapport.
        if not rid or not grp:
            continue
        cells = sorted(grp, key=lambda x: x.id)
        comp = stack_of_row(cells[0])
        if not comp or _stack_signature(comp) != sig:
            continue
        out.append({
            'run_id': rid,
            'active': rid == run_id,
            'weights': [{'label': m['label'], 'weight': m['weight'],
                         'filename': m['filename']} for m in comp],
            'likes': sum(1 for c in cells if c.rating == 1),
            'dislikes': sum(1 for c in cells if c.rating == -1),
            'done': sum(1 for c in cells if c.status == 'done' and c.filename),
            'cells': [{'id': c.id, 'dataset_id': c.dataset_id, 'checkpoint': c.checkpoint,
                       'label': _basename(c.checkpoint or '').rsplit('.', 1)[0],
                       'filename': c.filename, 'rating': c.rating, 'status': c.status,
                       'seed': c.seed, 'aspect': c.aspect, 'strength': c.strength,
                       'error': c.error if c.status == 'failed' else None} for c in cells],
        })
    # Le run courant d'abord, le reste dans l'ordre de scan (récent → ancien).
    out.sort(key=lambda v: not v['active'])
    return out


def create_run(user_id, dataset_id, checkpoints, strengths, seed=None, prompt=None, z_model=None, z_models=None, aspects=None, cfgs=None, steps_list=None, steps2_list=None, count=1, family=None, permanent_loras=None, batch_loras=None, rebalance=None, rebalance_strength=None, negative=None, sampler=None, scheduler=None, weight_dtype=None, enhancer=None, enhancer_strength=None, detail_amount=None, resolution_tier=None, resolution_multiplier=None, init_image=None, denoise=None, origins=None, prompts=None) -> dict:
    """Validate + materialize the grid and enqueue every cell.

    `prompts` (📝 lot) est un AXE : chaque configuration est rendue une fois par
    prompt coché dans l'historique. Absent/vide → un seul prompt, `prompt`, comme
    avant.

    Each cell's row and its queue job land in ONE commit (`_persist_and_enqueue_cell`);
    an enqueue failure marks that row 'failed' and re-raises - already-enqueued cells
    keep their rows AND their jobs. Returns
    {'created', 'seed', 'count', 'run_id', 'ids'}."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not (ds.trigger_word or '').strip():
        raise ValueError('trigger word is required')

    reason = gpu_busy_reason()
    if reason:
        raise GpuBusyError(reason)
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
        resolution_multiplier=resolution_multiplier,
        init_image=init_image, denoise=denoise)

    cells = build_matrix(checkpoints, strengths, aspects, cfgs, steps_list, steps2_list)

    # Pool de bases selon la FAMILLE : SDXL → checkpoints SDXL (de Generate), Krea →
    # base fixe (UNET du workflow, pas d'axe de base), Z-Image → modèles Z-Image.
    if run_family == 'sdxl':
        models = [m['filename'] for m in list_sdxl_base_models()]
        if not models:
            raise ValueError('no SDXL checkpoint available')
    elif run_family == 'krea':
        # None en tête = UNET câblé du workflow (défaut historique et repli) ; les
        # checkpoints Krea locaux deviennent un axe de base optionnel comme ailleurs.
        models = [None] + get_krea_models()
    else:
        models = get_zimage_models()
        if not models:
            raise ValueError('no Z-Image model available')
    # Modèle(s) de base - AXE de balayage optionnel (validés contre la whitelist).
    # z_models (liste) prioritaire ; sinon z_model unique (rétrocompat) ; sinon le 1er.
    # '' (entrée « Official » du picker Krea) ≡ None = défaut de la famille — mappé
    # AVANT validation pour que « Official + alternative » reste un axe à 2 valeurs.
    _req_models = list(z_models) if z_models else ([z_model] if z_model else [])
    _req_models = [None if m in ('', None) else m for m in _req_models]
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
    # 📝 Lot de prompts : l'axe vaut [prompt] quand rien n'est coché → chemin
    # strictement identique à avant. Le preflight et les journaux parlent du 1er.
    prompt_axis = _prompt_axis(prompts, prompt)
    prompt = prompt_axis[0]

    # Arch guard : la famille est dérivée du DOSSIER (family_of_lora) — un LoRA
    # mal classé (ex. un Z-Image déployé dans loras/krea) passerait ce filtre et
    # tournerait comme un no-op silencieux. On lit l'arch RÉELLE de chaque
    # checkpoint sélectionné dans son en-tête AVANT toute ligne → 409 actionnable.
    _preflight_checkpoint_arch(run_family, cps_in)
    # Preflight : le ComfyUI cible a-t-il RÉELLEMENT chaque modèle + custom node
    # dont le workflow de la famille a besoin ? On construit le graphe représentatif
    # (par base) et on le vérifie AVANT de créer la moindre ligne → un utilisateur
    # frais reçoit un seul 409 actionnable au lieu d'une grille de tuiles muettes.
    # (Krea/SDXL n'avaient AUCUN preflight ; seul Klein en avait un.)
    _preflight_run(user_id, run_family, cells[0][0], valid_models, allowed,
                   prompt, seeds[0], dataset_id, ds.trigger_word)

    # Classes du ComfyUI cible, lues UNE fois pour toute la grille : le builder s'en
    # sert pour réécrire les nodes à variantes (node 30 Krea) vers le nom réellement
    # enregistré. None (probe échouée) = on garde les noms canoniques.
    available_classes = _target_node_classes()
    # WHICH lineage checkpoint each selected LoRA is, stamped on every cell it
    # produces (see checkpoint_origins) — the canvas gallery reads these columns,
    # it never re-parses a filename.
    origin_of = checkpoint_origins(cps_in, origins)
    # One opaque id per invocation is the strict boundary of a render-equivalent
    # timeline.  A later launch with the same prompt/seed/settings must never be
    # spliced into this one; resume_run reuses the ids already stored on rows.
    run_id = uuid.uuid4().hex
    # Partition the complete base-model × cell plan, not each base separately:
    # after Krea starts applying a tested LoRA, it must not return to a
    # tested-LoRA-off control merely because the optional base axis advanced.
    cell_plan = _krea_zero_strength_first(
        ((zm, cell) for zm in valid_models for cell in cells),
        run_family,
        lambda planned: planned[1][1],
    )
    ids = []
    for zm, cell in cell_plan:
        checkpoint, strength, cell_aspect, cell_cfg, cell_steps, cell_steps2 = cell
        # Format/CFG/steps (1 et 2) testés comme axes à part entière (multi-sélection).
        width, height = _aspect_dims(cell_aspect, run_family, knobs['resolution_tier'],
                                     knobs['resolution_multiplier'])
        for batch_lora in batch_axis:  # AXE ⚖ batch : sans, puis avec chaque LoRA coché
          row_extra = extra_loras + ([{**batch_lora, 'batch': True}] if batch_lora else [])
          wf_extra = extra_loras + ([batch_lora] if batch_lora else [])
          cell_extra_json = json.dumps(row_extra) if row_extra else None
          for cell_prompt in prompt_axis:  # AXE 📝 lot : une passe par prompt coché
           for cell_seed in seeds:  # N images par config (seeds différents), bande dans la cellule
            img = LoraTestImage(dataset_id=dataset_id, checkpoint=checkpoint,
                                strength=strength, seed=cell_seed, run_seed=seed,
                                run_id=run_id,
                                status='pending', z_model=zm, aspect=cell_aspect,
                                prompt=cell_prompt, cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                extra_loras=cell_extra_json, krea_rebalance=cell_rebalance,
                                negative=knobs['negative'], sampler=knobs['sampler'],
                                scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                enhancer_strength=knobs['enhancer_strength'],
                                detail_amount=knobs['detail_amount'],
                                resolution_tier=knobs['resolution_tier'],
                                resolution_multiplier=knobs['resolution_multiplier'],
                                init_image=knobs['init_image'], denoise=knobs['denoise'],
                                record_id=origin_of.get(checkpoint, (None, None))[0],
                                step=origin_of.get(checkpoint, (None, None))[1])
            _persist_and_enqueue_cell(
                img, user_id, dataset_id, cell_prompt,
                lambda: _build_cell_workflow(user_id, checkpoint, strength,
                                             cell_prompt, cell_seed, zm, allowed,
                                             width=width, height=height,
                                             cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                             dataset_id=dataset_id,
                                             train_type=run_family, extra_loras=wf_extra,
                                             rebalance=cell_rebalance,
                                             negative=knobs['negative'], sampler=knobs['sampler'],
                                             scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                             enhancer_strength=knobs['enhancer_strength'],
                                             detail_amount=knobs['detail_amount'],
                                             trigger_word=ds.trigger_word,
                                             available_classes=available_classes))
            ids.append(img.id)
    logger.info(f"lora-test: run {run_id} dataset {dataset_id} -> {len(ids)} cellule(s) "
                f"({len(valid_models)} modèle(s), {len(prompt_axis)} prompt(s)), "
                f"base seed {seed} ×{count}")
    return {'created': len(ids), 'seed': seed, 'count': count,
            'run_id': run_id, 'ids': ids}


# 🧬 Plafond d'un poids de blend (pile combinée) — la borne haute de la plage
# commune ci-dessus, pas un second nombre. Le poids de tête d'une combinaison
# traverse build_matrix : un plafond de blend au-dessus du sien ferait échouer
# le run au lieu de le clamper. Miroir navigateur : `COMBINE_MAX_WEIGHT` dans
# frontend/src/components/dataset/studio/loraStack.js.
COMBINE_MAX_WEIGHT = MAX_LORA_STRENGTH


def _combine_weight(sel) -> float:
    """Poids d'un LoRA dans une pile combinée : 0..COMBINE_MAX_WEIGHT, arrondi au
    centième, 1.0 par défaut/valeur illisible."""
    try:
        return max(0.0, min(COMBINE_MAX_WEIGHT,
                            round(float((sel or {}).get('weight', 1.0)), 2)))
    except (TypeError, ValueError):
        return 1.0


def _combine_weights(sel) -> list:
    """Les poids que CE LoRA balaye dans la pile : la liste `weights` si elle est
    fournie (cases de poids du panneau 🧬 Blend), sinon le scalaire `weight`.

    Toujours non vide, clampée 0..COMBINE_MAX_WEIGHT, arrondie au centième, dédupliquée en gardant
    l'ordre reçu. Une sélection qui ne parle que de `weight` (client d'avant le
    balayage, ou repli d'un frontend neuf sur un backend ancien) rend donc
    exactement une valeur — le balayage est ADDITIF, il ne réinterprète rien."""
    raw = (sel or {}).get('weights')
    if not isinstance(raw, (list, tuple)) or not raw:
        return [_combine_weight(sel)]
    out = []
    for v in raw:
        try:
            w = max(0.0, min(COMBINE_MAX_WEIGHT, round(float(v), 2)))
        except (TypeError, ValueError):
            continue
        if w not in out:
            out.append(w)
    return out or [_combine_weight(sel)]


def create_comparison_run(user_id, selections, strengths, seed=None, prompt=None,
                          z_model=None, z_models=None, aspects=None, cfgs=None,
                          steps_list=None, steps2_list=None,
                          count=1, permanent_loras=None, batch_loras=None, rebalance=None, rebalance_strength=None,
                          negative=None, sampler=None, scheduler=None, weight_dtype=None,
                          enhancer=None, enhancer_strength=None, detail_amount=None,
                          resolution_tier=None, resolution_multiplier=None,
                          init_image=None, denoise=None, combine=None,
                          prompts=None, external_loras=None) -> dict:
    """Lance UN run de comparaison sur plusieurs LoRA. `selections` =
    [{dataset_id, checkpoint}] — chaque entrée peut aussi porter `record_id`/`step`
    (le LoRA Canvas les connaît : ce sont l'identité de la pastille cliquée), ce qui
    est alors stampé tel quel sur les cellules ; sinon l'origine est relue du tag de
    déploiement (cf. checkpoint_origins). Toutes les cellules partagent un run_id + le seed
    (équité). Le prompt : `prompt` commun si fourni, sinon l'identity_prompt du
    dataset de CHAQUE cellule (chaque LoRA a son trigger). 1 selection => run mono-LoRA.

    Parité Generate (2026-07-01) : always-on LoRA, rebalance Krea, steps2 SDXL et les
    réglages globaux (négatif/sampler/scheduler/precision/enhancer/detail/tier) sont
    partagés par TOUTES les cellules du run (gatés + validés par famille via _sanitize_gen_knobs).

    `combine=True` (≥2 sélections) bascule du mode COMPARAISON (1 cellule par LoRA,
    chacun seul) au mode PILE : les LoRA sélectionnés sont chargés ENSEMBLE dans la
    MÊME génération, chacun au `weight` porté par sa sélection, et les triggers des
    datasets correspondants sont TOUS injectés dans le prompt. L'axe `strengths`
    n'a alors plus de sens (chaque LoRA a son poids) : il est remplacé par le poids
    du 1er LoRA de la pile. La règle « un run = une seule famille » vaut aussi ici —
    combiner un LoRA Krea et un LoRA SDXL est impossible (bases et workflows
    différents), et c'est refusé avec un message nommant les familles.

    `external_loras` (Canvas plugin nodes) : `[{filename, strength}]` de N'IMPORTE
    QUEL fichier models/loras, stacké sur CHAQUE cellule via le même canal
    `extra_loras` que les always-on — mais sans restriction au pool de la famille :
    un nom introuvable est une erreur dure (jamais un skip silencieux), et l'arch
    preflight le couvre comme un checkpoint normal."""
    if not selections:
        raise ValueError('no LoRA selected')
    reason = gpu_busy_reason()
    if reason:
        raise GpuBusyError(reason)
    if _active_run_count():
        raise ValueError('a test run is already in progress - wait for it to finish or cancel')
    # La FAMILLE du run est dérivée du DOSSIER des checkpoints (family_of_lora), PAS du
    # scalaire `ds.train_type` (un dataset est multi-famille). Un run = une seule famille
    # (bases + workflow différents). On résout la base AVANT la boucle, selon la famille.
    fams = {family_of_lora(str(sel.get('checkpoint') or '')) for sel in (selections or [])}
    fams.discard(None)
    if len(fams) > 1:
        # Nommer les familles en cause : « ZIT/SDXL/Krea » ne disait pas LESQUELLES
        # étaient cochées, et en mode combine c'est l'erreur la plus probable.
        named = ' + '.join(_MODE_LABEL_BY_FAMILY.get(f, f) for f in sorted(fams))
        raise ValueError(
            f'a test run cannot mix LoRA families ({named}) — they need different '
            'base models and workflows. Keep one family per run.')
    run_type = (next(iter(fams), None) or 'zimage').lower()
    if run_type == 'sdxl':
        models = [m['filename'] for m in list_sdxl_base_models()]
        if not models:
            raise ValueError('no SDXL checkpoint available')
    elif run_type == 'krea':
        # None en tête = UNET câblé (node 20), repli des runs sans base explicite ;
        # les checkpoints Krea locaux sont désormais sélectionnables.
        models = [None] + get_krea_models()
    else:
        models = get_zimage_models()
        if not models:
            raise ValueError('no Z-Image model available')
    # Modèle(s) de base — AXE de balayage, exactement comme dans `create_run` :
    # le Canvas offre « BASE MODEL (MULTI) » et n'en lançait qu'UN, en silence.
    # z_models (liste) prioritaire ; sinon z_model unique (rétrocompat) ; sinon le 1er.
    _req_models = list(z_models) if z_models else ([z_model] if z_model else [])
    _req_models = [None if m in ('', None) else m for m in _req_models]
    valid_models = [m for m in _req_models if m in models] or [models[0]]
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
    # 📝 Lot de prompts (une passe par prompt coché). Rien de coché → [common_prompt],
    # donc [None] quand aucun prompt commun n'est fourni : chaque cellule retombe
    # sur le prompt d'identité de SON dataset, exactement comme avant.
    prompt_axis = _prompt_axis(prompts, common_prompt)
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
    # 🔌 External LoRAs (Canvas plugin nodes): ANY models/loras file, stacked on
    # top of every cell via the same extra_loras channel. Unlike always-on LoRA
    # they are NOT restricted to the family pool, so validation is fail-closed:
    # a name that does not resolve under a loras root is a hard error, never a
    # silent skip.
    externals = []
    for e in (external_loras or []):
        fn = str((e or {}).get('filename') or '').strip()
        if not fn or any(x['filename'] == fn for x in externals):
            continue
        # Path-traversal guard: `external_loras` is the FIRST free-text channel
        # to reach `_resolve_lora_abs_path` → `_ci_resolve` (every other caller —
        # permanent/batch/checkpoint — is gated by a disk-scan allowlist first).
        # `_ci_resolve` walks each component checking `os.path.exists` and treats
        # '..' as an ordinary component, so it happily climbs OUT of the loras
        # root. Checked BEFORE the resolve call, not after: a name that escapes
        # must never even get a "not found" vs "found" answer.
        if _is_unsafe_external_lora_name(fn):
            raise ValueError(f'invalid external LoRA name: {fn}')
        if not _resolve_lora_abs_path(fn):
            raise ValueError(f'external LoRA not found: {fn}')
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0
        externals.append({'filename': fn, 'strength': st, 'external': True})
        if len(externals) >= 16:   # same cap as the PUT route + the board's UI
            break
    extra_loras.extend(externals)
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
        resolution_multiplier=resolution_multiplier,
        init_image=init_image, denoise=denoise)

    # Arch guard (même contrat que create_run) : l'arch RÉELLE de chaque
    # checkpoint sélectionné, lue dans son en-tête, doit correspondre à la famille
    # du run — sinon ComfyUI le droppe en silence (grille no-op). Vérifié AVANT
    # toute ligne → 409 actionnable.
    _preflight_checkpoint_arch(
        run_type,
        [s.get('checkpoint') for s in selections if s.get('checkpoint')]
        + [x['filename'] for x in externals])
    # Un dataset = UN scan de LoRA. `list_test_checkpoints` walks the family's whole
    # LoRA folder (and stats every match): its result only depends on (dataset, family),
    # so a 24-cell grid over 8 checkpoints of the same dataset re-scanned that folder 9
    # times for one identical answer. Memoised for the duration of THIS call only — the
    # deployed set can change between two runs.
    _ckpt_memo = {}

    def _dataset_and_checkpoints(ds_id):
        """(dataset, allowed checkpoint filenames) for this run's family, scanned once."""
        if ds_id not in _ckpt_memo:
            _ds = fds.get_dataset(user_id, ds_id)
            _allowed = {c['filename'] for c in list_test_checkpoints(_ds, run_type)} if _ds else set()
            _ckpt_memo[ds_id] = (_ds, _allowed)
        return _ckpt_memo[ds_id]

    # Preflight (même contrat que create_run) : le ComfyUI cible peut-il vraiment
    # exécuter le workflow de cette famille ? On vérifie sur la 1re sélection valable
    # (le run est mono-famille) AVANT de créer les lignes → un seul 409 actionnable.
    for _sel in selections:
        _pf_ds, _pf_allowed = _dataset_and_checkpoints(_sel.get('dataset_id'))
        if not _pf_ds:
            continue
        _pf_cp = _sel.get('checkpoint')
        if _pf_cp in _pf_allowed:
            _preflight_run(user_id, run_type, _pf_cp, valid_models, _pf_allowed,
                           prompt_axis[0] or identity_prompt(_pf_ds), seeds[0],
                           _sel.get('dataset_id'), getattr(_pf_ds, 'trigger_word', None))
            break

    # Classes du ComfyUI cible, lues UNE fois pour tout le run (cf. create_run) →
    # réécriture des nodes à variantes (node 30 Krea) vers le nom réellement enregistré.
    available_classes = _target_node_classes()
    # Origine (run + step) de chaque LoRA sélectionné : explicite quand l'appelant
    # la connaît (canvas), sinon relue du tag de déploiement. Une seule résolution
    # par nom de fichier distinct.
    origin_of = checkpoint_origins(
        [s.get('checkpoint') for s in selections if s.get('checkpoint')],
        {s['checkpoint']: s for s in selections
         if s.get('checkpoint') and s.get('record_id') is not None
         and s.get('step') is not None})

    # --- Mode PILE (combine) ---------------------------------------------------
    # En comparaison, chaque sélection produit ses PROPRES cellules (un LoRA seul par
    # image). En combine, la sélection décrit UNE pile : le 1er LoRA reste le
    # « testé » (il porte la colonne de la grille et le dataset dont le prompt par
    # défaut est tiré), les suivants sont chaînés dans le MÊME graphe via le canal
    # `extra_loras` — celui des always-on, déjà câblé pour les trois familles
    # (inject_zimage_loras / inject_krea_loras / inject_sdxl_loras). Chaque
    # secondaire est revalidé contre les checkpoints réellement déployés de SON
    # dataset : la whitelist des extras est permissive côté montage, l'anti
    # path-injection se joue donc ici.
    #
    # 🧬 BALAYAGE : chaque sélection peut porter PLUSIEURS poids (`weights`), et le
    # run rend alors le PRODUIT CARTÉSIEN des combinaisons — une configuration
    # chacune, dans le MÊME run. Un seul poids par LoRA (le cas d'avant, et celui
    # d'un client qui n'envoie que `weight`) donne un produit d'un élément : le
    # chemin est donc strictement le même qu'avant pour tout ce qui existait.
    # Le poids du LoRA de TÊTE reste porté par `LoraTestImage.strength` et ceux des
    # membres par le JSON `extra_loras` — donc chaque cellule sait déjà dire de
    # quelle combinaison elle est, sans une colonne de plus.
    combine = bool(combine) and len(selections) > 1
    # [(stack_extra, stack_row)] par combinaison, alignés sur `combos`.
    stack_triggers = []
    combos = [None]
    members = []
    if combine:
        for sel in selections[1:]:
            _ds_i, _allowed_i = _dataset_and_checkpoints(sel.get('dataset_id'))
            if not _ds_i:
                raise ValueError(f"dataset {sel.get('dataset_id')} not found")
            fn = sel.get('checkpoint')
            if fn not in _allowed_i:
                raise ValueError(f'unknown checkpoint for {_ds_i.name}: {fn}')
            # 🧬 PROVENANCE DE GÉNÉRATION : d'où vient CE membre sur le board.
            # `origin_of` a déjà résolu l'origine de tous les checkpoints
            # sélectionnés, membres compris, juste au-dessus — c'est le moment
            # où l'information est la plus sûre (l'appelant vient de cliquer la
            # pastille, ou le tag de déploiement est encore celui d'aujourd'hui).
            # Sans elle, une pile ne sait dire de quelle pastille elle descend
            # que pour son LoRA de TÊTE, et un blend est par nature multi-parents.
            _origin_i = origin_of.get(fn, (None, None))
            members.append({'filename': fn, 'weights': _combine_weights(sel),
                            'dataset_id': _ds_i.id,
                            'record_id': _origin_i[0], 'step': _origin_i[1],
                            'trigger': getattr(_ds_i, 'trigger_word', None) or None})
            if getattr(_ds_i, 'trigger_word', None):
                stack_triggers.append(_ds_i.trigger_word)
        # Une combinaison = (poids de tête, poids du membre 1, …). Le dernier LoRA
        # varie le plus vite, comme dans le panneau qui l'annonce.
        head_weights = _combine_weights(selections[0])
        combos = [tuple(c) for c in itertools.product(
            head_weights, *[m['weights'] for m in members])]
        selections = selections[:1]

    run_id = uuid.uuid4().hex
    # Materialize the original selection-major plan, then stable-partition it
    # once for Krea. Zero tested-LoRA-off controls across *all* selected
    # checkpoints therefore finish before the first non-zero tested-LoRA cell,
    # while each group's checkpoint-major and strength order stays unchanged.
    cell_plan = []
    # Base-major, comme `create_run` : un balayage à une seule base produit
    # EXACTEMENT le plan d'avant, et à plusieurs les bases se lisent l'une
    # après l'autre au lieu de s'entrelacer.
    for zm in valid_models:
        for sel in selections:
            checkpoint = sel.get('checkpoint')
            for combo in combos:
                # En pile, l'axe strengths n'a plus de sens (chaque LoRA porte son
                # poids) : il vaut le poids de TÊTE de la combinaison courante.
                combo_strengths = [combo[0]] if combo is not None else strengths
                for cell in build_matrix([checkpoint], combo_strengths, aspects, cfgs,
                                         steps_list, steps2_list):
                    cell_plan.append((sel, cell, combo, zm))
    cell_plan = _krea_zero_strength_first(
        cell_plan, run_type, lambda planned: planned[1][1])

    ids = []
    for sel, cell, combo, zm in cell_plan:
        # Les poids des MEMBRES de cette combinaison. `stack_extra` (monté dans le
        # graphe) garde le format des always-on ; seule la copie PERSISTÉE porte
        # l'identité du membre, pour que la vue pile puisse redonner son dataset et
        # son trigger sans re-deviner.
        stack_extra, stack_row = [], []
        for i, m in enumerate(members):
            entry = {'filename': m['filename'], 'strength': combo[i + 1]}
            stack_extra.append(entry)
            # Seule la copie PERSISTÉE porte l'origine : `stack_extra` garde le
            # format des always-on, que le constructeur de workflow attend.
            stack_row.append({**entry, 'combined': True,
                              'dataset_id': m['dataset_id'], 'trigger': m['trigger'],
                              'record_id': m['record_id'], 'step': m['step']})
        ds, allowed = _dataset_and_checkpoints(sel.get('dataset_id'))
        if not ds:
            raise ValueError(f"dataset {sel.get('dataset_id')} not found")
        checkpoint = sel.get('checkpoint')
        if checkpoint not in allowed:
            raise ValueError(f'unknown checkpoint for {ds.name}: {checkpoint}')
        cp, strength, cell_aspect, cell_cfg, cell_steps, cell_steps2 = cell
        width, height = _aspect_dims(cell_aspect, run_type, knobs['resolution_tier'],
                                     knobs['resolution_multiplier'])
        for batch_lora in batch_axis:  # AXE ⚖ batch : sans, puis avec chaque LoRA coché
          row_extra = extra_loras + stack_row + ([{**batch_lora, 'batch': True}] if batch_lora else [])
          wf_extra = extra_loras + stack_extra + ([batch_lora] if batch_lora else [])
          cell_extra_json = json.dumps(row_extra) if row_extra else None
          for axis_prompt in prompt_axis:  # AXE 📝 lot : une passe par prompt coché
           cell_prompt = axis_prompt or identity_prompt(ds)
           for cell_seed in seeds:
            img = LoraTestImage(dataset_id=ds.id, checkpoint=cp, strength=strength,
                                seed=cell_seed, run_seed=seed, run_id=run_id,
                                status='pending', z_model=zm, aspect=cell_aspect,
                                prompt=cell_prompt, cfg=cell_cfg, steps=cell_steps, steps2=cell_steps2,
                                extra_loras=cell_extra_json, krea_rebalance=cell_rebalance,
                                negative=knobs['negative'], sampler=knobs['sampler'],
                                scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                enhancer_strength=knobs['enhancer_strength'],
                                detail_amount=knobs['detail_amount'],
                                resolution_tier=knobs['resolution_tier'],
                                resolution_multiplier=knobs['resolution_multiplier'],
                                init_image=knobs['init_image'], denoise=knobs['denoise'],
                                record_id=origin_of.get(cp, (None, None))[0],
                                step=origin_of.get(cp, (None, None))[1])
            _persist_and_enqueue_cell(
                img, user_id, ds.id, cell_prompt,
                # `zm` est lié PAR DÉFAUT, pas capturé : le graphe est construit
                # après coup, et une capture tardive donnerait à toutes les
                # cellules la base de la DERNIÈRE — un balayage qui ment.
                lambda _zm=zm: _build_cell_workflow(user_id, cp, strength, cell_prompt,
                                     cell_seed, _zm, allowed, width=width,
                                     height=height, cfg=cell_cfg, steps=cell_steps,
                                     steps2=cell_steps2, dataset_id=ds.id,
                                     train_type=run_type, extra_loras=wf_extra,
                                     rebalance=cell_rebalance,
                                     negative=knobs['negative'], sampler=knobs['sampler'],
                                     scheduler=knobs['scheduler'], weight_dtype=knobs['weight_dtype'],
                                     enhancer_strength=knobs['enhancer_strength'],
                                     detail_amount=knobs['detail_amount'],
                                     # Pile combinée : TOUS les triggers de la pile,
                                     # celui du LoRA de tête en premier.
                                     trigger_word=([ds.trigger_word] + stack_triggers
                                                   if combine else ds.trigger_word),
                                     available_classes=available_classes))
            ids.append(img.id)
    # `len(members)`, pas `len(stack_extra)` : celui-ci vit maintenant DANS la boucle
    # et vaudrait la dernière combinaison — ou n'existerait pas du tout sur un plan
    # vide. Le nombre de combinaisons est journalisé : c'est le premier chiffre
    # qu'on cherche quand un balayage rend plus d'images que prévu.
    logger.info(f"lora-test: {'combined' if combine else 'comparison'} run {run_id} -> "
                f"{len(ids)} cellule(s), {len(selections) + len(members)} LoRA, "
                f"{len(combos) if combine else 1} combinaison(s), "
                f"{len(prompt_axis)} prompt(s), seed {seed}")
    return {'created': len(ids), 'seed': seed, 'count': count, 'run_id': run_id, 'ids': ids}


def _run_owned(user_id, run_id) -> bool:
    """Single-user app: every run belongs to the local user - no cross-user
    ownership DB to consult (SRC checked every cell's dataset against
    `user_id`)."""
    return True


def cancel_run(user_id, dataset_id=None, run_id=None) -> int:
    """Cancel only cells whose exact ComfyUI work is safely gone.

    The entire selection and cancellation sweep holds the same GPU arbiter as
    ``process_one``. A worker therefore cannot claim the next grid cell between
    two safe cancellations. An uncertain prompt stays attached to its pending
    cell and is rendered as paused until a later Cancel can reconcile it.
    """
    if run_id is not None:
        if not _run_owned(user_id, run_id):
            return 0
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            return 0

    with GPU_ARBITER_LOCK:
        if run_id is not None:
            rows = (_cells()
                    .filter_by(run_id=run_id, status='pending')
                    .filter(LoraTestImage.filename.is_(None)).all())
        else:
            rows = (_cells()
                    .filter_by(dataset_id=dataset_id, status='pending')
                    .filter(LoraTestImage.filename.is_(None)).all())

        cancelled = 0
        for img in rows:
            if not img.job_id:
                img.status = 'cancelled'
                cancelled += 1
                continue
            try:
                safe = queue_manager.cancel_job(img.job_id, str(user_id), 'image')
            except Exception:
                logger.exception('lora-test: could not safely cancel queue job %s', img.job_id)
                safe = False
            if not safe:
                # Recover a request interrupted after the queue row committed but
                # before this cell could be committed. A terminal cancelled queue
                # row is durable proof that clearing this cell is safe.
                queue_row = ImageGenerationQueue.query.filter_by(job_id=img.job_id).first()
                safe = queue_row is not None and queue_row.status == 'cancelled'
            if not safe:
                continue
            img.status = 'cancelled'
            img.job_id = None
            cancelled += 1

        if cancelled:
            db.session.commit()
        return cancelled


def confirm_unknown_comfyui_restart(user_id, *, dataset_id=None, run_id=None,
                                    restart_confirmed=False) -> int:
    """Make exactly one unknown-submit Test Studio cell resumable again.

    A user must explicitly confirm an external ComfyUI restart at the route
    boundary. We then cancel only the stalled queue job identified by the raw
    barrier and its one linked pending cell in the same commit. Known prompt
    barriers keep their stricter remote reconciliation path.
    """
    if restart_confirmed is not True:
        raise ValueError('Confirm that you restarted ComfyUI before clearing this paused job.')
    if (dataset_id is None) == (run_id is None):
        raise ValueError('choose exactly one Test Studio run or dataset')
    if run_id is not None:
        if not _run_owned(user_id, run_id):
            raise ValueError('run not found')
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            raise ValueError('dataset not found')

    with GPU_ARBITER_LOCK:
        owner = queue_manager.get_comfyui_stalled_barrier()
        if (owner is None or owner.get('kind') != 'unknown_submit'
                or not isinstance(owner.get('job_id'), str)
                or owner.get('prompt_id') is not None):
            raise RuntimeError('There is no unknown ComfyUI submission awaiting restart confirmation.')
        job_id = owner['job_id']

        # The queue job is the authoritative remote identity. The cell match is
        # deliberately exact too: metadata protects current rows, while job_id
        # protects legacy rows created before cell_id/run_id were persisted.
        queue_job = (ImageGenerationQueue.query
                     .filter_by(job_id=job_id, user_id=str(user_id), status='stalled')
                     .filter(ImageGenerationQueue.comfyui_prompt_id.is_(None)).first())
        if queue_job is None:
            raise RuntimeError('The paused ComfyUI job changed; refresh its status before retrying.')

        try:
            cell_id = int(owner.get('cell_id'))
        except (TypeError, ValueError):
            raise RuntimeError('This paused job has no exact Test Studio cell to recover.')
        if str(cell_id) != owner.get('cell_id'):
            raise RuntimeError('This paused job has an invalid Test Studio cell identity.')

        # lds-allow-bare-lora-test-query: resolved by job_id (and an explicit id)
        # — recovery identifies ONE known row, whatever kind it is.
        cell_query = (LoraTestImage.query.filter_by(id=cell_id, job_id=job_id, status='pending')
                      .filter(LoraTestImage.filename.is_(None)))
        if run_id is not None:
            cell_query = cell_query.filter_by(run_id=str(run_id))
        else:
            cell_query = cell_query.filter_by(dataset_id=dataset_id)
        cell = cell_query.first()
        if cell is None:
            raise RuntimeError('The paused ComfyUI job does not belong to this Test Studio view.')
        for key, value in (('dataset_id', str(cell.dataset_id)), ('run_id', cell.run_id)):
            if key in owner and owner.get(key) != value:
                raise RuntimeError('The paused ComfyUI job identity does not match its Test Studio cell.')

        if not queue_manager.confirm_unknown_comfyui_restart(
                job_id, str(user_id), restart_confirmed=True, commit=False):
            db.session.rollback()
            raise RuntimeError('The paused ComfyUI job changed; refresh its status before retrying.')
        cell.status = 'cancelled'
        cell.job_id = None
        cell.error = None
        try:
            db.session.commit()  # Test Studio cell + exact queue job + raw barrier
        except Exception as exc:
            db.session.rollback()
            logger.exception('lora-test: could not confirm unknown ComfyUI restart for %s', job_id)
            raise RuntimeError('Could not record the ComfyUI recovery; nothing was resumed.') from exc
        return 1


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
            raise GpuBusyError(reason)
        if _active_run_count():
            raise ValueError('a test run is already in progress')
        rows = (_cells().filter_by(run_id=run_id)
                .filter(LoraTestImage.status.in_(['cancelled', 'failed'])).all())
    else:
        ds = fds.get_dataset(user_id, dataset_id)
        if not ds:
            raise ValueError('dataset not found')
        reason = gpu_busy_reason()
        if reason:
            raise GpuBusyError(reason)
        if _active_run_count(dataset_id):
            raise ValueError('a test run is already in progress')
        rows = (_cells().filter_by(dataset_id=dataset_id)
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
    # Classes du ComfyUI cible, lues UNE fois pour tout le resume → réécriture des nodes
    # à variantes (node 30 Krea) vers le nom réellement enregistré (cf. create_run).
    available_classes = _target_node_classes()
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
            # None en tête : les cellules legacy (z_model NULL) et celles dont la
            # base locale a disparu du disque retombent sur le UNET câblé, jamais
            # sur un modèle arbitraire.
            cell_models = [None] + get_krea_models()
        else:
            cell_models = get_zimage_models()
        z_model = (img.z_model if (img.z_model and img.z_model in cell_models)
                   else (cell_models[0] if cell_models else None))
        aspect = img.aspect if img.aspect in TEST_ASPECTS else DEFAULT_ASPECT
        # Palier + multiplicateur de résolution persistés → mêmes dims qu'au 1er run
        # (sinon table fixe / multiplicateur 1.0 sur les cellules legacy sans la colonne).
        width, height = _aspect_dims(aspect, cell_family, getattr(img, 'resolution_tier', None),
                                     getattr(img, 'resolution_multiplier', None) or 1.0)
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
                                            trigger_word=getattr(cell_ds, 'trigger_word', None),
                                            available_classes=available_classes)
            job_id = _enqueue_cell(user_id, img.dataset_id, workflow, prompt,
                                   cell_id=img.id, run_id=img.run_id)
            img.status = 'pending'
            img.filename = None
            img.job_id = job_id
            img.seed = seed
            img.error = None  # clean slate on a successful re-enqueue
            db.session.commit()
            n += 1
        except Exception as e:
            img.status = 'failed'
            img.error = str(e)[:400] or 'resume failed'
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


def link_completed_test_image(job_id, filename, failed=False, reason=None):
    """Attach a finished studio job to its LoraTestImage row.

    Mirror of link_completed_dataset_image: runs in the queue monitor thread
    whose SQLAlchemy session may hold a STALE read snapshot - if the first
    lookup misses, rollback (end the transaction) and re-read on a fresh
    snapshot before concluding the row doesn't exist.
    `reason` (the job row's error_message: a ComfyUI 400 validation body / node
    execution error / timeout) is persisted on the failed cell so the tile can
    say WHY it's empty instead of a mute red square (P0-b)."""
    # lds-allow-bare-lora-test-query: resolved by job_id. This is the completion
    # callback and it MUST see derived rows — an ✨ improve started from the canvas
    # lands through exactly this path, so filtering it out here would leave every
    # improvement pending forever.
    img = LoraTestImage.query.filter_by(job_id=job_id).first()
    if img is None:
        db.session.rollback()  # drop the stale read snapshot, then re-read
        # lds-allow-bare-lora-test-query: same lookup, fresh snapshot.
        img = LoraTestImage.query.filter_by(job_id=job_id).first()
    if img is None:
        logger.warning(f"lora-test link: no LoraTestImage row for job {job_id}")
        _cleanup_output_file(filename, failed)  # job sans ligne (annulé/repris) → orphelin
        return
    # Ne finaliser que les cellules ENCORE en attente : une complétion tardive d'un
    # job dont la ligne a été annulée/reprise (nouveau job_id, statut ≠ pending) ne
    # doit pas écraser le bon run - on jette son fichier au lieu de le déplacer.
    if img.status != 'pending':
        logger.info(f"lora-test link: ligne {img.id} déjà {img.status} pour job {job_id} - ignoré")
        _cleanup_output_file(filename, failed)
        return
    if failed:
        img.status = 'failed'
        img.error = (reason
                     or 'Generation failed (see 🪵 Server log in Settings for the ComfyUI error).')
    else:
        img.filename = filename
        img.status = 'done'
        # Bring the completed file into the per-dataset dir (served by
        # /api/dataset/<id>/img/<filename>, cleaned with the dataset). Prefer a
        # local disk move from ComfyUI's output dir; if the file isn't there —
        # ComfyUI was pointed at a custom output path, or none is configured —
        # fetch it over the /view API instead (path-independent). See GH #2.
        dst = os.path.join(fds._dataset_dir(img.dataset_id), filename)
        out_dir = _comfy_output_dir()
        src = os.path.join(out_dir, filename) if out_dir else None
        if src and os.path.exists(src):
            shutil.move(src, dst)
        elif not os.path.exists(dst):
            from ..utils.comfyui import fetch_output_image_bytes
            data = fetch_output_image_bytes(filename)
            if data:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'wb') as f:
                    f.write(data)
            else:
                # The result vanished (not on disk, /view fetch failed) — mark the
                # cell failed WITH a reason rather than leaving a 'done' row whose
                # <img> would 404 into a mute broken tile (P0-b, mirrors the dataset
                # fan-out's fail path).
                img.filename = None
                img.status = 'failed'
                img.error = ('The finished image could not be retrieved from ComfyUI '
                             '(not on disk, and the /view API fetch failed).')
                logger.warning(f"lora-test link: file not on disk and /view API fetch failed for {filename}")
    db.session.commit()


# --- ✨ Upscale & improve, from the ◉ Canvas lightbox --------------------------
# The board's pictures are `LoraTestImage` rows, so the dataset improve route
# (`/api/dataset/image/<id>/improve`, which resolves a `FaceDatasetImage`) cannot
# serve them: the two tables have INDEPENDENT id spaces, and passing a board id to
# it does not 404 — it finds a real, unrelated dataset image and improves that
# one. A silent wrong answer is the worst failure available here, which is why
# this lane exists instead of a one-line reuse of the other route.
#
# What it does NOT re-implement: the engine choice, the readiness preflight and
# the Klein/SeedVR2 hand-off are `face_dataset_service`'s, called as-is. So the
# 409 bodies that offer to install SeedVR2 on demand are the same ones, and a
# change to either engine reaches both surfaces at once.

# Worded as the user reads them, so the button can explain itself BEFORE the
# click instead of surfacing an error after it.
IMPROVE_SOURCE_GONE = 'that image is no longer in the library'
IMPROVE_FILE_GONE = 'that image file is no longer on disk'
IMPROVE_NOT_DONE = 'this image is still rendering'
IMPROVE_ALREADY_DERIVED = 'an upscale & improve result cannot be improved again'


def improve_canvas_image(user_id, image_id, engine=None):
    """Queue one non-destructive ✨ Upscale & improve of a board image.

    The source row and its file are never modified: the result is a SEPARATE row
    that copies `record_id`/`step`, so it appears in the same checkpoint gallery,
    right next to the picture it was made from, and can be pinned onto the board —
    which is the whole request.

    Returns ``{'candidate_id', 'job_id', 'engine'}``, or ``None`` when the image
    is not the caller's. Clicking twice while one is in flight returns the SAME
    candidate rather than spending the GPU on a duplicate. A candidate that
    FAILED does not block a new attempt — pressing ✨ again is how you retry,
    because the Test Studio's own resume path deliberately no longer picks these
    rows up (it would re-queue them as Z-Image cells, which is the wrong engine
    and the wrong workflow).
    """
    row = db.session.get(LoraTestImage, image_id)
    if row is None:
        return None
    ds = fds.get_dataset(user_id, row.dataset_id)
    if not ds:
        return None
    if row.derivation_kind:
        raise ValueError(IMPROVE_ALREADY_DERIVED)
    if row.status != 'done' or not row.filename:
        raise ValueError(IMPROVE_NOT_DONE)
    source_path = os.path.join(fds._dataset_dir(row.dataset_id), row.filename)
    if not os.path.isfile(source_path):
        raise ValueError(IMPROVE_FILE_GONE)

    # Idempotent while one is ACTUALLY in flight (pending). A finished candidate
    # does not block a second one: unlike the dataset lane there is no keep/reject
    # review to force here — the result is just another picture in the gallery.
    # lds-allow-bare-lora-test-query: this one wants the EXACT OPPOSITE of
    # _cells() — it looks for derived rows on purpose.
    active = (LoraTestImage.query
              .filter_by(parent_image_id=row.id, derivation_kind=CANVAS_IMAGE_IMPROVE,
                         status='pending')
              .order_by(LoraTestImage.id.desc()).first())
    if active is not None and active.job_id:
        return {'candidate_id': active.id, 'job_id': active.job_id,
                'engine': engine or ''}

    engine = fds.resolve_improve_engine(engine)
    fds._improve_preflight(engine)          # same refusals, same actionable 409s
    prompt = fds._improve_prompt() if engine == 'klein' else ''

    candidate = LoraTestImage(
        dataset_id=row.dataset_id,
        # NOT NULL columns, and honest: this IS an upscale of that checkpoint's
        # render at that strength.
        checkpoint=row.checkpoint, strength=row.strength,
        status='pending', filename=None,
        # WHERE it shows up: same checkpoint, same step → same gallery, next to
        # its source. `run_id` stays NULL, which is what keeps it out of the
        # checkpoint timeline (it filters `run_id IS NOT NULL`) — a 2 MP upscale
        # spliced into an epoch-by-epoch morph would be a lie.
        record_id=row.record_id, step=row.step, run_id=None,
        seed=row.seed,           # the download name keeps the lineage
        # The pass that ACTUALLY ran. A SeedVR2 restoration sends no prompt at
        # all, so storing Klein's instruction on one would put a sentence on
        # screen that had no effect on the picture (same rule as the dataset lane).
        prompt=(prompt[:500] if engine == 'klein'
                else 'SeedVR2 upscale (no prompt — restoration pass)'),
        # Deliberately NOT copied: z_model, aspect, cfg, steps, sampler, scheduler,
        # negative, extra_loras. None of them decided this image — the improve
        # profile did — and the lightbox renders them as "Made with", where a
        # copied value would be a lie about how the picture was produced.
        parent_image_id=row.id,
        derivation_kind=CANVAS_IMAGE_IMPROVE,
    )
    db.session.add(candidate)
    db.session.commit()                      # row BEFORE enqueue: no orphan job
    candidate_id = candidate.id

    try:
        job_id = fds._enqueue_improve(
            engine, user_id=user_id, source=row, source_path=source_path,
            prompt=prompt, label='', dataset=ds,
            # `is_lora_test` is what routes the finished job back to
            # link_completed_test_image (job_queue._dispatch_completion checks it
            # FIRST, before the model_name branch), so the result lands in THIS
            # table instead of being looked up as a dataset image that does not
            # exist.
            extra_metadata={
                'is_lora_test': True,
                'dataset_id': str(row.dataset_id),
                'cell_id': candidate_id,
                'derivation_kind': CANVAS_IMAGE_IMPROVE,
                'parent_image_id': row.id,
                'action': 'upscale_improve',
                'improve_engine': engine,
            })
    except Exception:
        # No ghost row. A candidate left `pending` with no file is exactly the
        # shape `_active_run_count` counts, so a failed enqueue that kept its row
        # would have been a permanent "a test run is already in progress" if the
        # derivation filter ever slipped. Belt and braces on the worst case.
        stale = db.session.get(LoraTestImage, candidate_id)
        if stale is not None:
            db.session.delete(stale)
            db.session.commit()
        raise

    saved = db.session.get(LoraTestImage, candidate_id)
    if saved is None:                        # cancelled mid-enqueue
        return None
    saved.job_id = job_id
    db.session.commit()
    return {'candidate_id': candidate_id, 'job_id': job_id, 'engine': engine}


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
    rows = _cells().filter_by(dataset_id=dataset_id).all()
    # Failed cells produced no image and can't be judged — exclude them so a broken
    # config doesn't inflate the 'images' denominator or otherwise pollute the
    # ranking / best-config pick (P0-b).
    rows = [r for r in rows if r.status != 'failed']
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
    rows = _cells().filter_by(dataset_id=dataset_id).all()
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
    img = (_cells()
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
        img = (_cells()
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


def best_settings_lora_filenames(ds) -> list[str]:
    """Every LoRA filename this dataset pins as a ★ best setting — a LIST, one
    entry per family, because the pin is stored per family (a dataset can have a
    winning ZIT combo and a winning SDXL one at the same time).

    This is what the "you are about to delete the pinned LoRA" guard-rail needs.
    Readers used to reach for `best_settings.lora_filename` straight off the
    payload, which only ever matched the LEGACY flat format: since the pin became
    a {family: setting} map that key does not exist any more, so the ⚠ line
    silently stopped appearing for every modern pin. Going through _best_map
    covers both shapes at once. Order is deterministic (family order as stored),
    duplicates collapsed."""
    out: list[str] = []
    for setting in _best_map(ds).values():
        if not isinstance(setting, dict):
            continue
        fn = setting.get('lora_filename')
        if fn and fn not in out:
            out.append(str(fn))
        # Une pile épinglée épingle TOUS ses LoRA : supprimer le second membre casse
        # le réglage gagnant aussi sûrement que supprimer celui de tête, le garde-fou
        # de suppression doit donc les voir tous.
        for member in setting.get('stack') or []:
            mfn = member.get('lora_filename') if isinstance(member, dict) else None
            if mfn and mfn not in out:
                out.append(str(mfn))
    return out


def set_best_settings(user_id, dataset_id, checkpoint, strength,
                      z_model=None, cfg=None, steps=None, steps2=None, aspect=None,
                      stack=None) -> dict:
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
    if not 0.05 <= strength <= 4.0:
        raise ValueError(f'strength out of range: {strength}')
    # Whitelist de bases selon la FAMILLE (SDXL → bases SDXL ; Krea → UNET locaux
    # scannés ; sinon Z-Image), sinon une base d'une autre famille était jetée.
    if family == 'sdxl':
        allowed_bases = {m['filename'] for m in list_sdxl_base_models()}
    elif family == 'krea':
        allowed_bases = set(get_krea_models())
    else:
        allowed_bases = set(get_zimage_models())
    z_model = z_model or None  # '' (entrée « Official » Krea) ≡ défaut → NULL
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
    # PILE (🧬 combine) : le réglage gagnant d'une pile, ce sont SES poids — pas un
    # checkpoint isolé. Le LoRA de tête reste dans `lora_filename`/`strength`, donc
    # tous les lecteurs existants (pin ★ du Canvas, « ★ Appliquer », garde-fou de
    # suppression, badge du workspace) continuent de fonctionner sans rien savoir des
    # piles ; les membres empilés s'ajoutent à côté, dans `stack`. Chaque membre est
    # revalidé contre les checkpoints déployés de SON dataset (anti path-injection :
    # le corps de la requête est de la donnée, pas une source de chemins).
    stack_out = []
    for member in (stack or []):
        if not isinstance(member, dict):
            raise ValueError('invalid stack member')
        member_ds = fds.get_dataset(user_id, member.get('dataset_id'))
        if not member_ds:
            raise ValueError('unknown dataset in stack')
        member_fn = member.get('lora_filename') or member.get('filename')
        member_family = (family_of_lora(member_fn)
                         or getattr(member_ds, 'train_type', None) or 'zimage').lower()
        if member_fn not in {c['filename'] for c in list_test_checkpoints(member_ds, member_family)}:
            raise ValueError('unknown checkpoint in stack')
        stack_out.append({'lora_filename': member_fn, 'dataset_id': member_ds.id,
                          'weight': _combine_weight({'weight': member.get('weight')}),
                          'trigger': getattr(member_ds, 'trigger_word', None) or None})
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
        # Absent (et non `[]`) quand ce n'est pas une pile : un réglage mono-LoRA
        # d'avant cette vue et un réglage mono-LoRA d'aujourd'hui restent identiques.
        **({'stack': stack_out} if stack_out else {}),
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
    # Third InsightFace lane, same single rule (fds.face_scoring_block_reason).
    # Returned in the shape the panel already renders (scoring_error) so the button
    # explains itself instead of scoring 0 cells in green.
    blocked = fds.face_scoring_block_reason(ds)
    if blocked:
        return {'scored': 0, 'total': 0, 'ranking': [],
                'scoring_error': {'kind': 'subject_not_photographic', 'detail': blocked}}
    if not ds.ref_filename:
        raise ValueError('reference photo missing')
    ref_path = fds._ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference photo missing')
    eff = _resolve_family(ds, family, available_families(ds))
    rows = (_cells().filter_by(dataset_id=dataset_id, status='done')
            .filter(LoraTestImage.filename.isnot(None)).all())
    rows = [r for r in rows if (family_of_lora(r.checkpoint) or 'zimage') == eff]
    ds_dir = fds._dataset_dir(dataset_id)
    by_path = {}
    for r in rows:
        p = os.path.join(ds_dir, r.filename)
        if os.path.exists(p):
            by_path[p] = r
    if not by_path:
        return {'scored': 0, 'total': 0, 'scoring_error': None, 'ranking': []}
    from .face_similarity import score_dataset_faces
    # scoring_error ({kind, detail} | None) remonte jusqu'au toast : un scorer
    # cassé doit dire POURQUOI, pas « done — 0/14 » en vert (user-reported).
    results, scoring_error = score_dataset_faces(ref_path, list(by_path.keys()))
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
    return {'scored': scored, 'total': len(by_path), 'scoring_error': scoring_error,
            'ranking': face_ranking(dataset_id, eff)}


def face_ranking(dataset_id, family) -> list:
    """Classement des checkpoints par similarité faciale MOYENNE (cellules déjà
    scorées, famille donnée). [{checkpoint, label, avg, n}] trié meilleur d'abord -
    le front marque le 1er comme « 🏆 best epoch »."""
    rows = (_cells().filter_by(dataset_id=dataset_id)
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
    """Delete one prompt’s Test Studio cells only after safe queue cancellation.

    A pending cell can be cancelled locally. A sent/running prompt must first be
    proven absent by ``queue_manager.cancel_job``; otherwise its row and files
    remain intact. This avoids turning an ambiguous ComfyUI request into an
    orphaned GPU job while the user is trying to delete the prompt.
    """
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    p = (prompt or '').strip()
    if not p:
        return 0
    dataset_dir = fds._dataset_path(dataset_id)

    with GPU_ARBITER_LOCK:
        rows = _cells().filter_by(dataset_id=dataset_id, prompt=p).all()
        if not rows:
            return 0
        # Do every safe cancellation before touching a file. A refusal retains
        # the prompt identity so the normal ComfyUI recovery flow can reconcile
        # it later; earlier safe cancellations are harmless and visible.
        for row in rows:
            if row.job_id and row.status not in ('done', 'failed', 'cancelled'):
                if not queue_manager.cancel_job(row.job_id, str(user_id), 'image'):
                    raise ValueError(
                        'ComfyUI work for this prompt is still running or needs recovery; '
                        'recover ComfyUI, cancel the paused cell, then try again.')

        moved = []
        seen_paths = set()
        try:
            for row in rows:
                if row.filename:
                    fp = os.path.join(dataset_dir, row.filename)
                    path_key = os.path.normcase(os.path.abspath(fp))
                    if path_key not in seen_paths and os.path.exists(fp):
                        destination = trash.send_to_trash(
                            fp, context=f'dataset-{dataset_id}-studio-prompt')
                        moved.append((destination, fp))
                        seen_paths.add(path_key)
                db.session.delete(row)
            db.session.commit()
        except Exception:
            db.session.rollback()
            for destination, original in reversed(moved):
                fds._restore_from_trash(destination, original)
            raise
    n = len(rows)
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
    rows_all = (_cells().filter_by(dataset_id=dataset_id)
                .order_by(LoraTestImage.id.asc()).all())
    # Grille = cellules de la famille effective (famille déduite du checkpoint).
    rows = [r for r in rows_all if (family_of_lora(r.checkpoint) or 'zimage') == eff]
    activity = _queue_activity(rows_all)
    best = _best_for_family(ds, eff)
    # Pool de bases selon la FAMILLE effective : SDXL → checkpoints SDXL (forme
    # {value,label}) ; Krea → base fixe (UNET du workflow, aucun sélecteur) ; sinon
    # modèles Z-Image. `train_type` = famille effective (le front adapte picker + handoff).
    base_note = None
    # Repli CFG/steps de la famille. Krea l'ajuste sur la base RÉELLEMENT élue :
    # une base non distillée élue par défaut avec les chiffres Turbo (cfg 1 /
    # 8 steps) rend une esquisse floue lue comme « l'entraînement a raté ».
    default_cfg, default_steps = DEFAULT_CFG, DEFAULT_STEPS
    if eff == 'sdxl':
        z_models = [{'value': m['filename'], 'label': m['label']}
                    for m in list_sdxl_base_models()]
    elif eff == 'krea':
        # Bases Krea locales ALTERNATIVES au défaut élu. L'entrée de tête (value
        # vide) reste le défaut ; son libellé dit QUEL fichier c'est quand ce n'est
        # pas celui que Setup installe. Aucune alternative sur disque → liste vide,
        # le front cache le sélecteur (comportement historique) — mais `base_note`,
        # lui, sort quand même : c'est justement cette install-là qui doit le lire.
        _krea_entry = krea_default_base_entry()
        base_note = _krea_entry['note']
        if _krea_entry['source']:
            _d = krea_model_defaults(_krea_entry['source'])
            default_cfg, default_steps = _d['cfg'], _d['steps']
        _alts = krea_alt_base_models()
        z_models = ([{'value': '', 'label': _krea_entry['label']}]
                    + [{'value': m, 'label': _basename(m).rsplit('.', 1)[0]} for m in _alts]
                    if _alts else [])
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
        # Ce que le défaut de base a d'anormal, quand il en a (Krea : le fichier
        # que Setup installe n'est pas là / celui élu porte autre chose que des
        # poids). None le reste du temps — le front n'affiche rien.
        'base_note': base_note,
        'aspects': list(TEST_ASPECTS.keys()),
        'default_aspect': DEFAULT_ASPECT,
        'cfg_choices': CFG_CHOICES, 'default_cfg': default_cfg,
        'steps_choices': STEPS_CHOICES, 'default_steps': default_steps,
        # Per-BASE-MODEL cfg/steps, keyed by the same `value` as `z_models` (bobba84,
        # GitHub #18): Z-Image Base is not guidance-distilled and must not inherit
        # Turbo's cfg 1 / 8 steps. `default_cfg`/`default_steps` stay the fallback for
        # every base not listed here, so an older frontend behaves exactly as before.
        'model_defaults': studio_model_defaults(eff, z_models),
        # 2e passe (detail daemon) : exposée UNIQUEMENT pour SDXL (le workflow HQ a deux
        # passes). NULL sinon → le frontend ne montre pas le 2e picker de steps.
        'steps2_choices': (STEPS_CHOICES if eff == 'sdxl' else None),
        'default_steps2': (DEFAULT_STEPS if eff == 'sdxl' else None),
        'max_images': MAX_TEST_IMAGES,
        # Le rythme RÉEL de cette machine sur cette pipeline (médiane observée),
        # ou null quand l'historique est trop court : l'estimation de durée du
        # panneau cesse d'être « ~12 s/image » sur toutes les cartes du monde.
        'seconds_per_image': measured_seconds_per_image(eff),
        'cells': [{'id': r.id, 'checkpoint': r.checkpoint,
                   'label': format_trained_lora_label(r.checkpoint) or _basename(r.checkpoint).rsplit('.', 1)[0],
                   'strength': r.strength, 'aspect': r.aspect, 'filename': r.filename,
                   'rating': r.rating, 'seed': r.seed, 'run_seed': r.run_seed,
                   # WHICH launch this cell belongs to. The column has always been
                   # written (`create_run`), but it was never served, so the grid
                   # had to guess a run from `run_seed` + prompt — and a batch of N
                   # prompts then looked like N separate runs, of which it showed
                   # one. Null on rows predating the column; the frontend keeps the
                   # old grouping for those.
                   'run_id': r.run_id, 'status': r.status,
                   'queue_status': activity['queue_status'].get(r.job_id),
                   'queue_error': activity['queue_error'].get(r.job_id),
                   'prompt': r.prompt, 'z_model': r.z_model,
                   'z_model_label': (_basename(r.z_model).rsplit('.', 1)[0] if r.z_model else None),
                   'cfg': r.cfg, 'steps': r.steps, 'steps2': r.steps2,
                   'batch_lora': _batch_lora_label(r),
                   'combined_loras': _combined_lora_labels(r),
                   # Why the tile is empty (failed cells only) → shown on hover (P0-b).
                   'error': r.error if r.status == 'failed' else None,
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
        'pending': activity['pending'],
        'queued': activity['queued'],
        'generating': activity['generating'],
        'running': activity['running'],
        # Cellules stoppées/échouées reprenables - global (resume opère sur tout le dataset).
        'resumable': sum(1 for r in rows_all if r.status in ('cancelled', 'failed')),
        # Prompts récents distincts (family-agnostiques) pour recharger/relancer un
        # run - GLOBAUX à l'utilisateur (tous datasets), plus cloisonnés par dataset.
        'recent_prompts': user_recent_prompts(ds.user_id),
        'gpu_busy': gpu_busy_reason(),
        'comfyui_recovery': _unknown_submit_recovery(rows, activity),
        'comfyui_recovery_target': _comfyui_recovery_target(),
        'best_settings': best,
    }


def lora_net_scores(run_id) -> list[dict]:
    """Classement PAR-LoRA d'un run : agrège les votes des cellules par dataset_id
    (= un LoRA). Trié par score net (likes - dislikes) puis likes, décroissant."""
    rows = _cells().filter_by(run_id=run_id).filter(
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
    rows = (_cells().filter_by(run_id=run_id)
            .order_by(LoraTestImage.id.asc()).all())
    if not rows:
        return None
    ds_ids = {r.dataset_id for r in rows}
    owned = {d.id for d in FaceDataset.query.filter(FaceDataset.user_id == str(user_id),
             FaceDataset.id.in_(ds_ids)).all()}
    if ds_ids - owned:
        return None
    activity = _queue_activity(rows)
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
                   'run_seed': r.run_seed, 'status': r.status,
                   'queue_status': activity['queue_status'].get(r.job_id),
                   'queue_error': activity['queue_error'].get(r.job_id), 'prompt': r.prompt,
                   'z_model': r.z_model, 'cfg': r.cfg, 'steps': r.steps, 'steps2': r.steps2,
                   'batch_lora': _batch_lora_label(r),
                   'combined_loras': _combined_lora_labels(r),
                   'error': r.error if r.status == 'failed' else None} for r in rows],
        'lora_ranking': lora_net_scores(run_id),
        # Run PILE (🧬 combine) : sa composition (chaque LoRA, son poids, son trigger)
        # et les autres relances de la même pile. Le classement par-LoRA ci-dessus est
        # alors trompeur (une pile n'a qu'un LoRA « testé ») : le front montre la
        # composition à sa place. `stack` vaut None sur un run de comparaison.
        'stack': (_stack := stack_of_row(rows[0])),
        'stack_variants': stack_variants(run_id, rows) if _stack else [],
        'pending': activity['pending'],
        'queued': activity['queued'],
        'generating': activity['generating'],
        'running': activity['running'],
        'resumable': sum(1 for r in rows if r.status in ('cancelled', 'failed')),
        'gpu_busy': gpu_busy_reason(),
        'comfyui_recovery': _unknown_submit_recovery(rows, activity),
        'comfyui_recovery_target': _comfyui_recovery_target(),
    }


def _recent_prompts(rows, limit=None) -> list[dict]:
    """Prompts distincts utilisés (récent→ancien) AVEC une vignette : une image 👍
    générée avec ce prompt (à défaut, la plus récente terminée), + le nombre d'images.
    Permet de voir ce que fait chaque prompt dans le menu. `thumb_dataset_id` porte
    le dataset de la vignette (nécessaire quand les rows couvrent PLUSIEURS datasets).
    `limit=None` (défaut) = tous les prompts distincts trouvés dans `rows` — le
    plafond arbitraire (10) a été retiré à la demande de l'utilisateur ; la seule
    borne restante est le scan des 1500 dernières cellules dans user_recent_prompts.
    Retour: [{prompt, thumbnail(filename|None), thumb_dataset_id, thumb_rating, count}]."""
    seen = {}  # prompt -> dict (ordre d'insertion = récent→ancien)
    for r in sorted(rows, key=lambda x: -x.id):  # plus récent d'abord
        p = (r.prompt or '').strip()
        if not p:
            continue
        if p not in seen:
            if limit is not None and len(seen) >= limit:
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


def user_recent_prompts(user_id, limit=None) -> list[dict]:
    """Prompts de test récents de l'UTILISATEUR, TOUS datasets confondus (demande
    2026-07-03 : la mémoire des prompts/presets ne doit plus être cloisonnée par
    dataset - un prompt réglé sur Emma doit se recharger sur Adele). `limit=None`
    (défaut) = tous les prompts distincts trouvés (le plafond de 10 a été retiré à
    la demande de l'utilisateur). La seule borne restante est le scan des 1500
    dernières cellules (perf) ; chaque entrée porte `thumb_dataset_id` pour que
    le front construise l'URL de vignette du BON dataset."""
    ds_ids = [d.id for d in FaceDataset.query.filter_by(user_id=str(user_id)).all()]
    if not ds_ids:
        return []
    rows = (_cells().filter(LoraTestImage.dataset_id.in_(ds_ids))
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
