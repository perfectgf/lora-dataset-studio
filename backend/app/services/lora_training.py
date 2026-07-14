"""Automatisation de l'entraînement LoRA Z-Image via ai-toolkit.

L'app prépare (export dataset + job-config) et lance l'UI ai-toolkit ; elle ne
réimplémente pas l'entraîneur. Pause GPU via le flag system_state
`training_in_progress` honoré par le superviseur ComfyUI.

Lifted from the parent project's app/services/lora_training.py (1288 lines)
for LoRA Dataset Studio: SRC's module-level AITOOLKIT_DIR/HF_HOME/DATASETS_DIR/
OUTPUT_DIR/LORA_DEST_DIR* constants become live `cfg.aitoolkit_path(...)` /
`cfg.comfyui_dir(...)` accessors below, each raising a clean RuntimeError when
its backend isn't configured yet (so config.json edits apply without a
restart, and routes can map the RuntimeError to a 409). `UI_URL` (ai-toolkit's
web UI, unused - this app drives the CLI) and the whole ownership subsystem
(`record_lora_ownership`, the ownership-filtered checkpoint listing) are
dropped - single local user, cf. plan's Global Constraints.
"""
from __future__ import annotations
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime

from PIL import Image

from .. import config as cfg
from ..models import FaceDataset, FaceDatasetImage
from ..job_queue import queue_manager
from . import face_dataset_service as fds
from .person_mask import generate_person_masks

logger = logging.getLogger(__name__)

# Résolution + VRAM Krea 2 (modèle 12B). MESURÉ 2026-06-26 : à 1024 SANS unload TE la VRAM
# sature (24,0/24,5 Go) → ~180 s/it (ETA ~7 j, inexploitable) ; à 768 → 3,5 s/it (~50× plus
# rapide → goulot = ACTIVATIONS, pas le streaming des poids). Stratégie qualité : on GARDE 1024
# mais on libère le Qwen3-VL via cache_text_embeddings + unload_text_encoder (~4-8 Go) pour
# tenir sans offload. Si 1024 sature encore → baisser ce SEUL curseur à 896 (mesurer), puis 768
# (cadence prouvée). Curseur de tuning #1, un seul endroit.
KREA_TRAIN_RESOLUTION = 1024

# TTL des flags system_state d'un run (training_in_progress / _pid / _dataset_id /
# _target_step). L'anti-concurrence repose sur le PID VIVANT, mais le GARDE lit
# d'abord le flag `training_in_progress` (cf. is-training checks) : si son TTL
# expire AU MILIEU d'un run, le flag retombe à False, le garde rouvre la porte et
# la file lance un 2e entraînement par-dessus le 1er (collision mémoire → « page
# file too small »). Un run Krea-2-Raw (non distillé, CFG 4 / 25 steps de preview)
# dépasse 4 h → l'ancien TTL 4 h expirait avant la fin ET privait le snapshot du
# checkpoint final de son target_step. 12 h couvre le plus long run réaliste ;
# `process_training_queue` re-arme de toute façon les flags à chaque poll tant que
# le PID vit, donc c'est une ceinture, pas la bretelle.
_TRAIN_STATE_TTL = 12 * 3600


# --- Path accessors (replace SRC's module-level AITOOLKIT_DIR/HF_HOME/... constants) --

def _aitoolkit_dir():
    d = cfg.aitoolkit_path('dir')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _hf_home():
    d = cfg.aitoolkit_path('hf_home')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _datasets_dir():
    d = cfg.aitoolkit_path('datasets')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _output_dir():
    d = cfg.aitoolkit_path('output')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    return d


def _venv_python():
    p = cfg.aitoolkit_path('venv_python')
    if not p:
        raise RuntimeError('ai-toolkit is not configured')
    return p


def _jobs_dir():
    d = cfg.aitoolkit_path('jobs')
    if not d:
        raise RuntimeError('ai-toolkit is not configured')
    d.mkdir(parents=True, exist_ok=True)
    return d


# ComfyUI-side destinations (deploy target for a trained LoRA, and the SDXL base
# checkpoint pool). Distinct error message from the aitoolkit accessors above:
# a dataset can be trainable (aitoolkit OK) while ComfyUI itself is unconfigured,
# and the two are gated independently by the Settings/capabilities probe.
def _lora_dest_dir_zimage():
    d = cfg.comfyui_dir('loras')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d / 'z image'


def _lora_dest_dir_sdxl():
    d = cfg.comfyui_dir('loras')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d / 'sdxl'


def _lora_dest_dir_krea():
    d = cfg.comfyui_dir('loras')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d / 'krea'


def _lora_dest_dir_flux():
    d = cfg.comfyui_dir('loras')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d / 'flux'


def _lora_dest_dir_flux2klein():
    d = cfg.comfyui_dir('loras')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d / 'flux2klein'


def _sdxl_checkpoints_dir():
    d = cfg.comfyui_dir('models')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return d / 'checkpoints'


def is_installed() -> bool:
    """ai-toolkit est-il installé (venv python présent) ?"""
    p = cfg.aitoolkit_path('venv_python')
    return bool(p) and p.is_file()


def _aitoolkit_supports_krea() -> bool:
    """L'ai-toolkit installé connaît-il l'arch Krea 2 ? C'est CRITIQUE : ai-toolkit
    fait `if ModelClass.arch == config.arch` puis, sans match, retombe
    SILENCIEUSEMENT sur le loader SD legacy (get_model.py:get_model_class) - aucune
    erreur levée. Une config `arch:'krea2'` sur un ai-toolkit pas à jour chargerait
    donc Krea-2-Turbo comme un checkpoint SD et planterait de façon confuse. On
    scanne les sources d'archs (extensions_built_in) ; lecture fraîche → dès que
    le mainteneur fait `git pull`, la détection passe à True sans redémarrage.

    On exige l'arch EXACTE `arch = "krea2"` (la chaîne émise par _build_job_config_krea),
    pas la simple sous-chaîne « krea » : sinon une mention incidente (commentaire,
    variable) ferait un FAUX POSITIF, et surtout si l'arch upstream diffère (ex.
    « krea2_turbo ») la garde donnerait un feu vert alors que get_model_class ne
    matcherait pas → fallback SD silencieux, précisément ce qu'on veut empêcher."""
    root = cfg.aitoolkit_path('dir')
    if not root:
        return False
    ext_root = root / 'extensions_built_in'
    if not ext_root.is_dir():
        return False
    pat = re.compile(r'arch\s*=\s*[\'"]krea2[\'"]')
    for dp, _dn, files in os.walk(str(ext_root)):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            try:
                with open(os.path.join(dp, fn), encoding='utf-8', errors='ignore') as fh:
                    if pat.search(fh.read()):
                        return True
            except OSError:
                continue
    return False


def _aitoolkit_supports_flux2klein() -> bool:
    """L'ai-toolkit installé connaît-il FLUX.2 Klein ? Même enjeu CRITIQUE que
    _aitoolkit_supports_krea (lire son commentaire) : les archs flux2_klein_4b/9b
    sont des EXTENSIONS (extensions_built_in/diffusion_models/flux2), pas des archs
    cœur comme 'flux' — un ai-toolkit pas à jour ne les connaît pas et
    get_model_class retomberait SILENCIEUSEMENT sur le loader SD legacy → LoRA
    corrompu. On exige l'arch EXACTE `arch = "flux2_klein_4b"` ou `"..._9b"` (les
    chaînes émises par _build_job_config_flux2klein), jamais la sous-chaîne
    « klein » seule — une mention incidente ferait un faux positif. Lecture
    fraîche : un `git pull` du mainteneur passe la détection à True sans restart."""
    root = cfg.aitoolkit_path('dir')
    if not root:
        return False
    ext_root = root / 'extensions_built_in'
    if not ext_root.is_dir():
        return False
    pat = re.compile(r'arch\s*=\s*[\'"]flux2_klein_(?:4b|9b)[\'"]')
    for dp, _dn, files in os.walk(str(ext_root)):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            try:
                with open(os.path.join(dp, fn), encoding='utf-8', errors='ignore') as fh:
                    if pat.search(fh.read()):
                        return True
            except OSError:
                continue
    return False


def _safe_trigger(ds) -> str:
    t = (ds.trigger_word or f'dataset{ds.id}').strip()
    return ''.join(c if (c.isalnum() or c in '_-') else '_' for c in t) or f'dataset{ds.id}'


def _train_type(ds, family=None) -> str:
    """Famille de modèle entraînée : 'zimage' (défaut/None), 'sdxl', 'krea',
    'flux' ou 'flux2klein'.
    `family` (override) prime sur le train_type persisté quand fourni (non vide) -
    c'est ce qui permet au sélecteur de famille de l'UI de piloter la lecture des
    runs/checkpoints/déploiements SANS écraser le train_type persisté du dataset."""
    return ((family or None) or getattr(ds, 'train_type', None) or 'zimage').lower()


def _lora_dest_dir(ds, family=None) -> str:
    """Dossier loras ComfyUI où DÉPLOYER le LoRA entraîné, routé par famille :
    krea → loras/krea/ (pour qu'il apparaisse dans le menu de génération Krea via
    get_krea_loras), sdxl → loras/sdxl/, zimage (défaut) → « z image/ ». Garde les
    familles séparées (un LoRA Krea ne doit pas polluer le Test Studio Z-Image)."""
    fam = _train_type(ds, family)
    if fam == 'sdxl':
        return str(_lora_dest_dir_sdxl())
    if fam == 'krea':
        return str(_lora_dest_dir_krea())
    if fam == 'flux':
        return str(_lora_dest_dir_flux())
    if fam == 'flux2klein':
        return str(_lora_dest_dir_flux2klein())
    return str(_lora_dest_dir_zimage())


def _sdxl_base_choices() -> set:
    """Whitelist serveur des bases SDXL = basenames des checkpoints ComfyUI.
    include_hidden=True pour ne pas exclure un checkpoint masqué légitime, et
    pour récupérer une forme stable quelle que soit la variante de retour."""
    from ..utils.comfyui import get_checkpoint_models
    out = set()
    for c in (get_checkpoint_models(include_hidden=True) or []):
        out.add(c['name'] if isinstance(c, dict) else c)
    return out


def _sdxl_base_path(base_model: str) -> str:
    """Résout le .safetensors SDXL sous models/checkpoints. get_checkpoint_models
    APLATIT en basename (l'info de sous-dossier - ex. Biglove/ - est perdue) → on
    cherche récursivement le basename. Refuse chemin absolu / '..' (anti-traversal ;
    la whitelist amont _sdxl_base_choices garantit déjà un basename connu)."""
    name = str(base_model or '')
    parts = name.replace('\\', '/').split('/')
    if os.path.isabs(name) or '..' in parts:
        raise ValueError('invalid SDXL base path')
    checkpoints_dir = str(_sdxl_checkpoints_dir())
    cand = os.path.join(checkpoints_dir, name)
    if os.path.exists(cand):
        return cand
    base = os.path.basename(name.replace('\\', '/'))
    for root, _dirs, files in os.walk(checkpoints_dir):
        if base in files:
            return os.path.join(root, base)
    return name  # fallback (ne devrait pas arriver : base whitelistée + existante)


# Sentinelle « base non fournie » : distingue l'absence d'argument (→ base
# PERSISTÉE du dataset) de la valeur '' (= base officielle, un choix explicite).
_PERSISTED = object()


def _base_tag_for(base_model) -> str:
    """Suffixe de run pour une base EXPLICITE ('' / None = officiel → '')."""
    if not base_model:
        return ''
    base = os.path.basename(str(base_model).replace('\\', '/')).rsplit('.', 1)[0]
    safe = ''.join(c if (c.isalnum() or c in '_-') else '_' for c in base)
    return f'_{safe}' if safe else ''


def _base_tag(ds) -> str:
    """Suffixe de run dérivé de la base d'entraînement PERSISTÉE (vide = officiel).
    Isole les checkpoints d'un run sur merge de ceux du run officiel du même
    dataset (sinon ai-toolkit auto-resume depuis le mauvais base → mélange)."""
    return _base_tag_for(getattr(ds, 'train_base_model', None))


KREA_BASE_LABEL = 'Krea-2-Turbo'   # mirrors name_or_path 'krea/Krea-2-Turbo'
# Flux a une seule base officielle (FLUX.1-dev). Sans point dans le label (sinon
# _base_tag_for le prendrait pour une extension et tronquerait à « FLUX ») → tag
# stable '_FLUX-1-dev' qui isole les runs/LoRA Flux des runs Z-Image officiels
# (tag vide) au même trigger — même garde anti-collision que Krea (cf. _dest_base_tag).
FLUX_BASE_LABEL = 'FLUX-1-dev'
# FLUX.2 Klein a DEUX bases officielles (4B et 9B) → tags DISTINCTS obligatoires :
# les poids 4B et 9B sont incompatibles, et un même trigger entraîné sur les deux
# variantes partagerait sinon le même dossier de run (auto-resume croisé → LoRA
# corrompu) et le même nom de LoRA déployé. Sans point dans les labels (même piège
# d'extension que FLUX_BASE_LABEL : _base_tag_for tronque après un '.').
FLUX2KLEIN_BASE_LABELS = {'4b': 'FLUX2-Klein-4B', '9b': 'FLUX2-Klein-9B'}


def _krea_is_raw(ds) -> bool:
    """Krea 2 training base. `train_variant` 'base'/'raw' → Krea-2-Raw (non-distilled,
    the official recommendation « train on Raw, validate on Turbo » — best quality,
    the LoRA transfers to Turbo at inference); 'turbo' → Krea-2-Turbo + Ostris adapter
    (VRAM-friendly). Default RAW when unset — that's the chosen product default, so the
    tag and the job-config never disagree even if train_variant was never persisted."""
    return (getattr(ds, 'train_variant', None) or 'base').lower() in ('base', 'raw')


def _flux2klein_is_9b(ds) -> bool:
    """FLUX.2 Klein model size. `train_variant` '9b' → the 9B base (32-48 GB VRAM,
    the cloud-first lane); anything else → the 4B base (16-24 GB, the local lane).
    Default 4B when unset — the chosen product default (mirrors _default_variant_for),
    so the run tag and the job-config never disagree even if train_variant was
    never persisted."""
    return (getattr(ds, 'train_variant', None) or '4b').lower() == '9b'


def _default_variant_for(family) -> str:
    """Variante par défaut d'une famille quand aucune n'est fournie NI persistée :
    Krea → 'base' (Raw, reco officielle), FLUX.2 Klein → '4b' (la voie locale
    16-24 Go ; le 9B est la voie cloud), sinon 'turbo'. Utilisé par tous les
    chemins de lancement (direct / file / reprise / cloud) pour que le défaut
    tienne de bout en bout, pas seulement quand l'UI envoie explicitement la variante."""
    fam = family or 'zimage'
    if fam == 'krea':
        return 'base'
    if fam == 'flux2klein':
        return '4b'
    return 'turbo'


def _valid_variants_for(family) -> tuple:
    """Variantes acceptées au lancement, PAR FAMILLE : flux2klein n'a que ses deux
    tailles de modèle ('4b'/'9b') ; les familles historiques gardent l'enum
    turbo/base/deturbo (comportement inchangé). Une variante hors liste retombe
    sur le défaut de la famille (jamais d'erreur) : c'est ce qui neutralise une
    variante PERSISTÉE d'une autre famille quand l'utilisateur change de type
    (ex. un dataset ex-Krea avec train_variant='base' lancé en flux2klein)."""
    return ('4b', '9b') if (family or 'zimage') == 'flux2klein' \
        else ('turbo', 'base', 'deturbo')


# --- Réglages ai-toolkit avancés, éditables par dataset (persistés en JSON dans
#     `train_settings`). Absent/NULL → défaut family-aware issu de la recherche
#     (cf. Research vault 2026-07-10). Toute valeur hors des listes autorisées
#     retombe sur le défaut : on ne pousse JAMAIS une config invalide à ai-toolkit. ---
_DEFAULT_RANK = {'zimage': 16, 'krea': 32, 'sdxl': 32, 'flux': 16, 'flux2klein': 16}   # Z-Image reste 16 (choix user) ; Krea/SDXL 32 ; Flux/FLUX.2 Klein 16 (défaut des exemples officiels)
_RANK_CHOICES = (8, 16, 24, 32, 48, 64)
# multi-échelle par défaut ; '768' seul = LE levier basse-VRAM (Krea 12B : 1024
# sature un 24 GB à ~180 s/it, 768 mesuré ~3,5 s/it — cf. commentaire de tête).
_RES_CHOICES = {'768,1024': [768, 1024], '1024': [1024], '768': [768]}
_SAVE_CHOICES = (250, 500, 1000)
# --- Expert levers (train_settings, ALL default to current behaviour when absent,
#     so a newcomer who never touches them gets the exact same config as before) ---
_DROPOUT_CHOICES = (0.05, 0.1, 0.15, 0.2, 0.3)          # LoRA network dropout ; absent = off
_ALPHA_CHOICES = (1, 2, 4, 8, 16, 24, 32, 48, 64)       # alpha découplé du rank ; absent = dérivé
_TIMESTEP_TYPE_CHOICES = ('sigmoid', 'linear', 'weighted', 'shift')  # pondération flowmatch ; SDXL le désactive
_DEFAULT_TIMESTEP = {'zimage': 'sigmoid', 'krea': 'linear', 'flux': 'sigmoid',
                     'flux2klein': 'weighted'}   # ce que « Auto » résout (sdxl : aucun) ; flux subject → sigmoid (reco ai-toolkit) ; flux2klein → weighted (défaut canonique options.ts, PAS sigmoid)
# Batch 2 — optimiseur / planning du LR / batch effectif (valeurs VÉRIFIÉES dans
# ai-toolkit : get_optimizer + toolkit/scheduler.py). CAME n'est PAS supporté.
_OPTIMIZER_CHOICES = ('adamw8bit', 'adafactor', 'automagic', 'prodigy')
_LR_SCHEDULER_CHOICES = ('constant', 'linear', 'cosine', 'cosine_with_restarts', 'constant_with_warmup')
_WARMUP_CHOICES = (50, 100, 200, 500)          # num_warmup_steps ; UNIQUEMENT avec constant_with_warmup
_GRAD_ACCUM_CHOICES = (1, 2, 4)


def _train_settings(ds) -> dict:
    """Parse le blob JSON `train_settings` en dict (jamais lève ; {} si absent/cassé)."""
    raw = getattr(ds, 'train_settings', None)
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def _lora_rank(ds, family) -> int:
    r = _train_settings(ds).get('rank')
    return r if r in _RANK_CHOICES else _DEFAULT_RANK.get(family, 32)


def _lora_alpha(rank, family) -> int:
    """ai-toolkit : alpha = rank (échelle 1.0) pour zimage/krea. SDXL garde son
    choix délibéré alpha = rank/2 (« demi-force », validé par la recherche)."""
    return max(1, rank // 2) if family == 'sdxl' else rank


def _lora_alpha_eff(ds, rank, family) -> int:
    """Alpha EFFECTIF : un `alpha` explicite dans train_settings prime sur le dérivé.
    Découpler alpha du rank = levier de LR « doux » (échelle effective = alpha/rank)."""
    a = _train_settings(ds).get('alpha')
    return a if a in _ALPHA_CHOICES else _lora_alpha(rank, family)


def _network_block(ds, rank, family) -> dict:
    """Bloc `network` LoRA partagé par les 3 job-configs : rank + alpha (override-aware)
    + dropout optionnel (régularisateur anti-overfit, clé omise quand off)."""
    net = {'type': 'lora', 'linear': rank, 'linear_alpha': _lora_alpha_eff(ds, rank, family)}
    d = _train_settings(ds).get('dropout')
    if isinstance(d, (int, float)) and d in _DROPOUT_CHOICES:
        net['dropout'] = d
    return net


def _timestep_type_eff(ds, default: str) -> str:
    """Pondération des timesteps : override la valeur family-default si l'utilisateur en
    a choisi une valide (gardé à l'enum ai-toolkit ; inconnu → le défaut)."""
    t = _train_settings(ds).get('timestep_type')
    return t if t in _TIMESTEP_TYPE_CHOICES else default


def _optimizer_eff(ds) -> str:
    o = _train_settings(ds).get('optimizer')
    return o if o in _OPTIMIZER_CHOICES else 'adamw8bit'


def _lr_eff(ds) -> float:
    """Prodigy pilote le LR lui-même → convention lr≈1.0 ; les autres gardent 1e-4."""
    return 1.0 if _optimizer_eff(ds).startswith('prodigy') else 1e-4


def _grad_accum(ds) -> int:
    g = _train_settings(ds).get('grad_accum')
    return g if g in _GRAD_ACCUM_CHOICES else 1


def _lr_sched_fields(ds) -> dict:
    """{} par défaut (= 'constant' d'ai-toolkit). Sinon {lr_scheduler [+ lr_scheduler_params
    {num_warmup_steps} pour constant_with_warmup]} à fusionner dans le bloc train. Le warmup
    n'est câblé QUE pour constant_with_warmup : les schedulers torch (cosine/linear/constant)
    n'acceptent pas num_warmup_steps → le passer les ferait planter (cf. toolkit/scheduler.py)."""
    s = _train_settings(ds).get('lr_scheduler')
    if s not in _LR_SCHEDULER_CHOICES or s == 'constant':
        return {}
    out = {'lr_scheduler': s}
    if s == 'constant_with_warmup':
        w = _train_settings(ds).get('warmup')
        out['lr_scheduler_params'] = {'num_warmup_steps': w if w in _WARMUP_CHOICES else 100}
    return out


def _train_res(ds) -> list:
    return _RES_CHOICES.get(_train_settings(ds).get('resolution'), [768, 1024])


def _save_every(ds) -> int:
    v = _train_settings(ds).get('save_every')
    return v if v in _SAVE_CHOICES else 250


# Combien de saves intermédiaires ai-toolkit CONSERVE pendant le run (local et
# cloud) : au-delà, il supprime les plus anciens lui-même. L'historique (10)
# laissait s'accumuler ~10 Go de checkpoints par run Krea.
_MAX_SAVES_CHOICES = (2, 3, 4, 6, 10)


def _max_step_saves(ds) -> int:
    v = _train_settings(ds).get('max_step_saves')
    return v if v in _MAX_SAVES_CHOICES else 4


# --- Prompts de preview (sample) -----------------------------------------------
# ai-toolkit génère une image par prompt tous les `sample_every` steps pendant le
# run (dossier .../samples), pour voir le LoRA converger. Les défauts historiques
# décrivaient un VISAGE (« close-up portrait, headshot… ») — hors sujet pour un
# dataset « concept ». D'où un défaut distinct selon le kind, et un override total
# par l'utilisateur (Advanced options → Preview prompts).
_SAMPLE_EVERY_CHOICES = (100, 250, 500, 1000)
_MAX_SAMPLE_PROMPTS = 8   # 1 image générée / prompt / palier → borne le coût des previews

_DEFAULT_SAMPLE_PROMPTS_CHARACTER = [
    '{trigger}, close-up portrait, neutral expression',
    '{trigger}, headshot, soft studio light',
    '{trigger}, full body, walking outdoors, smiling',
    '{trigger}, sitting in a cafe, casual outfit',
]
# Un concept n'est pas un visage : on l'exerce seul sous quelques cadrages neutres
# (le vocabulaire « portrait / headshot » tirerait un LoRA non-visage hors sujet).
_DEFAULT_SAMPLE_PROMPTS_CONCEPT = [
    '{trigger}',
    '{trigger}, high detail, sharp focus',
    '{trigger}, wide shot',
    '{trigger}, cinematic lighting',
]


# Un style n'a PAS de trigger : le LoRA teinte toute image dès qu'il est chargé.
# Les previews sont donc des scènes génériques variées — si le style s'y voit,
# l'entraînement prend ; le vocabulaire portrait/headshot tirerait hors sujet.
_DEFAULT_SAMPLE_PROMPTS_STYLE = [
    'a woman reading in a sunlit cafe',
    'a city street at night, rain',
    'a mountain landscape, wide shot',
    'a still life of fruit on a wooden table',
]


def _default_sample_prompts(ds) -> list:
    if fds.is_style(ds):
        return list(_DEFAULT_SAMPLE_PROMPTS_STYLE)
    return list(_DEFAULT_SAMPLE_PROMPTS_CONCEPT if fds.is_concept(ds)
                else _DEFAULT_SAMPLE_PROMPTS_CHARACTER)


def _inject_trigger(prompt: str, trigger: str) -> str:
    """Une preview DOIT solliciter le LoRA : si la ligne ne mentionne pas déjà le
    trigger (insensible à la casse), on le préfixe — sinon l'image teste le modèle
    de base, pas l'entraînement en cours."""
    p = (prompt or '').strip()
    if not trigger:
        return p
    if not p:
        return trigger
    return p if trigger.lower() in p.lower() else f'{trigger}, {p}'


def _resolved_default_sample_prompts(ds, trigger) -> list:
    """Défauts (selon le kind) avec `{trigger}` substitué — pour l'aperçu UI."""
    if fds.is_style(ds):   # style : pas de trigger, jamais injecté
        return list(_default_sample_prompts(ds))
    return [_inject_trigger(l.replace('{trigger}', trigger), trigger)
            for l in _default_sample_prompts(ds)]


def _sample_prompts(ds, trigger) -> list:
    """Prompts de preview effectifs : liste custom de train_settings si présente,
    sinon défaut selon le kind. `{trigger}` (placeholder explicite) ET le trigger en
    clair sont gérés ; le trigger est auto-préfixé s'il manque. Toujours ≥1 prompt,
    ≤_MAX_SAMPLE_PROMPTS (borne le nombre d'images générées par palier)."""
    raw = _train_settings(ds).get('sample_prompts')
    tmpl = raw if (isinstance(raw, list)
                   and any(isinstance(x, str) and x.strip() for x in raw)) \
        else _default_sample_prompts(ds)
    # STYLE : aucun trigger — le LoRA teinte tout, une preview générique le
    # sollicite déjà. Injecter le trigger polluerait le prompt d'un token inconnu.
    style = fds.is_style(ds)
    out = []
    for line in tmpl:
        if not isinstance(line, str) or not line.strip():
            continue
        resolved = line.replace('{trigger}', '' if style else trigger).strip(', ') or line
        out.append(resolved if style else _inject_trigger(resolved, trigger))
        if len(out) >= _MAX_SAMPLE_PROMPTS:
            break
    if out:
        return out
    return [_default_sample_prompts(ds)[0]] if style else [_inject_trigger('', trigger)]


def _sample_every(ds) -> int:
    v = _train_settings(ds).get('sample_every')
    return v if v in _SAMPLE_EVERY_CHOICES else 250


def launch_settings_snapshot(ds, family=None) -> dict:
    """Les réglages EFFECTIFS envoyés à ai-toolkit pour CE lancement — défauts
    résolus, pas les choix stockés. Stampé dans le registre de provenance
    (TrainingRunRecord.settings) par chaque launch local et cloud ; la page
    Runs l'affiche par run (« quels réglages sont partis ? »). Compact : les
    leviers experts n'apparaissent que s'ils dévient du défaut."""
    fam = family or _train_type(ds)
    rank = _lora_rank(ds, fam)
    snap = {
        # trigger_word is part of the reproducible RECIPE (someone re-running
        # the LoRA needs it) and is not a secret — it already appears in the
        # run name. The Share-config file surfaces it; settingsLine ignores it.
        'trigger': _safe_trigger(ds),
        'rank': rank,
        'alpha': _lora_alpha_eff(ds, rank, fam),
        'resolution': _train_res(ds),
        'save_every': _save_every(ds),
        'max_step_saves': _max_step_saves(ds),
        'optimizer': _optimizer_eff(ds),
        'lr': _lr_eff(ds),
    }
    if fam != 'sdxl':
        snap['timestep_type'] = _timestep_type_eff(ds, _DEFAULT_TIMESTEP.get(fam, 'sigmoid'))
    s = _train_settings(ds)
    for k in ('dropout', 'lr_scheduler', 'warmup', 'grad_accum', 'sample_every'):
        if s.get(k):
            snap[k] = s[k]
    return snap


def effective_train_settings(ds, family=None) -> dict:
    """Réglages pour la famille courante — ce que « Advanced options » affiche et
    ce que build_job_config enverra. `rank` = choix STOCKÉ (None = auto/défaut) pour
    que le select re-coche « Auto » ; `effective_rank`/`alpha`/`default_rank` = ce
    qui sera réellement utilisé (pour le libellé explicatif)."""
    fam = family or _train_type(ds)
    s = _train_settings(ds)
    stored_rank = s.get('rank') if s.get('rank') in _RANK_CHOICES else None
    eff_rank = stored_rank if stored_rank else _DEFAULT_RANK.get(fam, 32)
    res = s.get('resolution')
    trig = _safe_trigger(ds)
    stored_prompts = s.get('sample_prompts')
    return {'rank': stored_rank,                       # None → Auto (défaut family-aware)
            'effective_rank': eff_rank,                # ce qui part à ai-toolkit
            'alpha': _lora_alpha_eff(ds, eff_rank, fam),   # alpha EFFECTIF (override-aware) — libellé
            'default_rank': _DEFAULT_RANK.get(fam, 32),
            # --- Expert levers (None/off = comportement actuel ; le select recoche « Auto ») ---
            'alpha_setting': s.get('alpha') if s.get('alpha') in _ALPHA_CHOICES else None,
            'default_alpha': _lora_alpha(eff_rank, fam),
            'alpha_choices': list(_ALPHA_CHOICES),
            'dropout': s.get('dropout') if s.get('dropout') in _DROPOUT_CHOICES else None,
            'dropout_choices': list(_DROPOUT_CHOICES),
            'timestep_type': s.get('timestep_type') if s.get('timestep_type') in _TIMESTEP_TYPE_CHOICES else None,
            'timestep_type_choices': list(_TIMESTEP_TYPE_CHOICES),
            'default_timestep_type': _DEFAULT_TIMESTEP.get(fam),   # None pour sdxl → contrôle masqué
            'timestep_type_supported': fam != 'sdxl',
            'optimizer': s.get('optimizer') if s.get('optimizer') in _OPTIMIZER_CHOICES else None,   # None → adamw8bit
            'optimizer_choices': list(_OPTIMIZER_CHOICES),
            'lr_scheduler': s.get('lr_scheduler') if s.get('lr_scheduler') in _LR_SCHEDULER_CHOICES else None,  # None → constant
            'lr_scheduler_choices': list(_LR_SCHEDULER_CHOICES),
            'warmup': s.get('warmup') if s.get('warmup') in _WARMUP_CHOICES else None,
            'warmup_choices': list(_WARMUP_CHOICES),
            'grad_accum': s.get('grad_accum') if s.get('grad_accum') in _GRAD_ACCUM_CHOICES else None,   # None → 1
            'grad_accum_choices': list(_GRAD_ACCUM_CHOICES),
            'resolution': res if res in _RES_CHOICES else '768,1024',
            'save_every': _save_every(ds),
            'max_step_saves': _max_step_saves(ds),
            'max_step_saves_choices': list(_MAX_SAVES_CHOICES),
            'sample_every': _sample_every(ds),
            # liste STOCKÉE brute (telle que tapée) ou [] → textarea vide = « défauts ».
            'sample_prompts': stored_prompts if isinstance(stored_prompts, list) else [],
            # défaut résolu (kind + trigger courant) : placeholder/aperçu quand vide.
            'sample_prompts_default': _resolved_default_sample_prompts(ds, trig),
            'sample_every_choices': list(_SAMPLE_EVERY_CHOICES),
            'max_sample_prompts': _MAX_SAMPLE_PROMPTS}


def update_train_settings(user_id, dataset_id, patch: dict) -> dict:
    """Valide + fusionne un patch {rank?, resolution?, save_every?, sample_every?,
    sample_prompts?} dans train_settings. Une clé à None/'auto'/vide est RETIRÉE
    (retour au défaut). Retourne les réglages effectifs pour la famille courante."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur = _train_settings(ds)
    if 'rank' in patch:
        r = patch['rank']
        if r in (None, 'auto'):
            cur.pop('rank', None)
        elif r in _RANK_CHOICES:
            cur['rank'] = r
        else:
            raise ValueError(f'rank must be one of {_RANK_CHOICES} (or auto)')
    if 'resolution' in patch:
        v = patch['resolution']
        if v in _RES_CHOICES:
            cur['resolution'] = v
        else:
            raise ValueError(f'resolution must be one of {list(_RES_CHOICES)}')
    if 'save_every' in patch:
        v = patch['save_every']
        if v in _SAVE_CHOICES:
            cur['save_every'] = v
        else:
            raise ValueError(f'save_every must be one of {_SAVE_CHOICES}')
    if 'max_step_saves' in patch:
        v = patch['max_step_saves']
        if v in (None, 'auto'):
            cur.pop('max_step_saves', None)
        elif v in _MAX_SAVES_CHOICES:
            cur['max_step_saves'] = v
        else:
            raise ValueError(f'max_step_saves must be one of {_MAX_SAVES_CHOICES}')
    if 'sample_every' in patch:
        v = patch['sample_every']
        if v in _SAMPLE_EVERY_CHOICES:
            cur['sample_every'] = v
        else:
            raise ValueError(f'sample_every must be one of {_SAMPLE_EVERY_CHOICES}')
    if 'sample_prompts' in patch:
        v = patch['sample_prompts']
        # Accepte aussi une string multi-lignes (une par prompt) pour le confort UI.
        if isinstance(v, str):
            v = v.splitlines()
        if v in (None, ''):
            cur.pop('sample_prompts', None)               # vide → retour aux défauts kind-aware
        elif isinstance(v, list):
            cleaned = [str(x).strip() for x in v if str(x).strip()][:_MAX_SAMPLE_PROMPTS]
            if cleaned:
                cur['sample_prompts'] = cleaned
            else:
                cur.pop('sample_prompts', None)
        else:
            raise ValueError('sample_prompts must be a list of strings (or empty to reset)')
    if 'dropout' in patch:
        v = patch['dropout']
        if v in (None, 0, 0.0, 'off', ''):
            cur.pop('dropout', None)                       # off → clé retirée
        elif v in _DROPOUT_CHOICES:
            cur['dropout'] = v
        else:
            raise ValueError(f'dropout must be one of {_DROPOUT_CHOICES} (or off)')
    if 'alpha' in patch:
        v = patch['alpha']
        if v in (None, 'auto'):
            cur.pop('alpha', None)                         # auto → alpha dérivé du rank
        elif v in _ALPHA_CHOICES:
            cur['alpha'] = v
        else:
            raise ValueError(f'alpha must be one of {_ALPHA_CHOICES} (or auto)')
    if 'timestep_type' in patch:
        v = patch['timestep_type']
        if v in (None, 'auto', ''):
            cur.pop('timestep_type', None)                 # auto → défaut family-aware
        elif v in _TIMESTEP_TYPE_CHOICES:
            cur['timestep_type'] = v
        else:
            raise ValueError(f'timestep_type must be one of {_TIMESTEP_TYPE_CHOICES} (or auto)')
    if 'optimizer' in patch:
        v = patch['optimizer']
        if v in (None, 'auto', '', 'adamw8bit'):
            cur.pop('optimizer', None)                     # défaut → clé retirée
        elif v in _OPTIMIZER_CHOICES:
            cur['optimizer'] = v
        else:
            raise ValueError(f'optimizer must be one of {_OPTIMIZER_CHOICES} (or auto)')
    if 'lr_scheduler' in patch:
        v = patch['lr_scheduler']
        if v in (None, 'auto', '', 'constant'):
            cur.pop('lr_scheduler', None)                  # constant = défaut → clé retirée
        elif v in _LR_SCHEDULER_CHOICES:
            cur['lr_scheduler'] = v
        else:
            raise ValueError(f'lr_scheduler must be one of {_LR_SCHEDULER_CHOICES} (or auto)')
    if 'warmup' in patch:
        v = patch['warmup']
        if v in (None, 0, 'off', ''):
            cur.pop('warmup', None)
        elif v in _WARMUP_CHOICES:
            cur['warmup'] = v
        else:
            raise ValueError(f'warmup must be one of {_WARMUP_CHOICES} (or off)')
    if 'grad_accum' in patch:
        v = patch['grad_accum']
        if v in (None, 1, 'auto'):
            cur.pop('grad_accum', None)                    # 1 = défaut → clé retirée
        elif v in _GRAD_ACCUM_CHOICES:
            cur['grad_accum'] = v
        else:
            raise ValueError(f'grad_accum must be one of {_GRAD_ACCUM_CHOICES} (or auto)')
    ds.train_settings = json.dumps(cur) if cur else None
    fds.db.session.commit()
    return effective_train_settings(ds)


# Every key update_train_settings knows how to validate — KEEP IN SYNC when a
# new expert lever is added above. This is what makes presets schema-tolerant:
# a preset key outside this list is IGNORED (and reported), never fatal.
TRAIN_SETTING_KEYS = ('rank', 'resolution', 'save_every', 'max_step_saves',
                      'sample_every', 'sample_prompts', 'dropout', 'alpha',
                      'timestep_type', 'optimizer', 'lr_scheduler', 'warmup',
                      'grad_accum')

# Built-in presets: shipped with the app (every install sees them), read-only,
# versioned with the code. The recommended character recipe: the researched
# family defaults pinned explicitly, plus the checkpoint-SELECTION machinery —
# a save + a probe preview at every 250 steps and no snapshot cap, because on
# character sets the quality comes from picking the earliest checkpoint that
# holds the identity, not from exotic hyper-parameters. Steps stay adaptive
# (~120 × kept images). A test asserts every builtin applies with zero
# ignored/rejected keys, so a drifting choice-list can't silently break them.
BUILTIN_TRAIN_PRESETS = [
    {
        'id': 'builtin-krea-character',
        'name': 'Krea character — recommended',
        'train_type': 'krea',
        'builtin': True,
        'settings': {
            'rank': 32,                    # Krea researched default (alpha derives = 32)
            'resolution': '768,1024',      # multi-scale: close-up → full body
            'save_every': 250,
            'max_step_saves': 10,          # keep every snapshot — all sweet-spot candidates
            'sample_every': 250,           # one probe sheet per checkpoint
            'sample_prompts': [            # identity AND flexibility probes — overfit
                                           # (waxy skin, frozen pose) shows here first
                '{trigger}, close-up portrait, neutral expression, soft studio light',
                '{trigger}, headshot, golden hour sunlight, slight smile',
                '{trigger}, bust shot, profile view, window light',
                '{trigger}, full body, walking outdoors in a park, casual jeans and t-shirt',
                '{trigger}, full body, elegant evening dress, dim moody lighting',
                '{trigger}, sitting at a cafe table, laughing, candid photo',
                '{trigger}, sportswear, stretching in a gym, harsh fluorescent light',
                '{trigger}, wide shot, standing on a beach at dusk, wind in hair',
            ],
        },
    },
    # Concept/style runs scale SUB-linearly (recommended_steps: 475·√n clamped
    # [2000, 12000] — code anchors: ~30-40 images → ~3000 steps, ~400 → ~9500).
    # save/sample every 500 (vs 250 for characters) is the coverage compromise:
    # max_step_saves keeps the N most RECENT saves (ai-toolkit deletes the
    # oldest), so 10×500 spans the last 5000 steps — the whole run at the small
    # anchor, the second half at the large one — while halving the preview GPU
    # cost of long runs (1 image per prompt per interval). Probes exercise the
    # concept across framings, contexts and lighting: a concept LoRA that only
    # reproduces its training context has overfit.
    {
        'id': 'builtin-concept',
        'name': 'Concept — recommended',
        'train_type': 'zimage',
        'builtin': True,
        'settings': {
            'resolution': '768,1024',
            'save_every': 500,
            'max_step_saves': 10,
            'sample_every': 500,
            'sample_prompts': [
                '{trigger}',
                '{trigger}, close-up, high detail, sharp focus',
                '{trigger}, wide shot showing the full scene',
                '{trigger}, in an unusual setting, outdoors',
                '{trigger}, soft natural window light',
                '{trigger}, night scene, artificial light',
                '{trigger}, seen from a high angle',
                '{trigger}, cinematic composition, shallow depth of field',
            ],
        },
    },
    # Same steps scale and save/preview cadence as concept (same 475·√n
    # recipe drives both). Style previews carry NO trigger — a style LoRA has
    # none and the export strips `{trigger}` on style datasets — so varied
    # CONTENT is the probe: if the aesthetic shows on a portrait AND a night
    # street AND a still life, the style generalized instead of memorizing
    # its training scenes.
    {
        'id': 'builtin-style',
        'name': 'Style — recommended',
        'train_type': 'zimage',
        'builtin': True,
        'settings': {
            'resolution': '768,1024',
            'save_every': 500,
            'max_step_saves': 10,
            'sample_every': 500,
            'sample_prompts': [
                'a woman reading in a sunlit cafe',
                'a city street at night, rain, neon reflections',
                'a mountain landscape, wide shot, morning mist',
                'a still life of fruit on a wooden table',
                'a cozy interior, warm lamp light',
                'a runner mid-stride on a bridge, motion',
                'a cat sleeping on a windowsill',
                'a modern building facade, strong shadows',
            ],
        },
    },
]


def snapshot_train_settings(user_id, dataset_id) -> dict:
    """The dataset's RAW explicit settings (what a preset captures) — only the
    keys the user actually changed, not the effective/derived view."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    return _train_settings(ds)


def apply_train_settings_dict(user_id, dataset_id, settings: dict):
    """REPLACE the dataset's explicit settings with a preset's dict, running
    every key through the validated update_train_settings path. Content is
    never fatal: unknown keys (newer/older app versions) are ignored, invalid
    values collected — both reported so the UI can say what didn't land.
    Returns (effective_settings, ignored_keys, rejected)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    ignored = sorted(k for k in settings if k not in TRAIN_SETTING_KEYS)
    rejected = []
    ds.train_settings = None          # a preset REPLACES, it doesn't overlay
    fds.db.session.commit()
    for k in TRAIN_SETTING_KEYS:
        if k not in settings:
            continue
        try:
            update_train_settings(user_id, dataset_id, {k: settings[k]})
        except ValueError as e:
            rejected.append({'key': k, 'reason': str(e)})
    return (effective_train_settings(fds.get_dataset(user_id, dataset_id)),
            ignored, rejected)


def _dest_base_tag(ds, base_model=_PERSISTED, family=None) -> str:
    """Deployment-name suffix, family-aware. Like _base_tag, but for Krea
    (which has no base column - always Krea-2-Turbo) falls back to a constant tag
    so the LoRA carries the model name like SDXL does. `family` override permet au
    sélecteur UI de router vers Krea même si le train_type persisté diffère."""
    tag = _base_tag(ds) if base_model is _PERSISTED else _base_tag_for(base_model)
    if not tag and _train_type(ds, family) == 'krea':
        # Raw and Turbo are DIFFERENT base checkpoints → distinct tags so their
        # run folders / deployed LoRA names never collide (same trigger, same
        # family, but incompatible weights would otherwise share a folder).
        tag = _base_tag_for('Krea-2-Raw' if _krea_is_raw(ds) else KREA_BASE_LABEL)
    # Même garde pour Flux : sa base officielle donne un tag vide, qui télescoperait
    # un run Z-Image officiel du même trigger (même dossier `u{user}_{trigger}` →
    # ai-toolkit auto-resume le mauvais run, poids mélangés). Le tag `_FLUX-1-dev`
    # isole le run et le LoRA déployé de la famille Z-Image.
    if not tag and _train_type(ds, family) == 'flux':
        tag = _base_tag_for(FLUX_BASE_LABEL)
    # FLUX.2 Klein : même garde, mais le tag encode AUSSI la variante (4B vs 9B
    # sont deux checkpoints incompatibles) — sans ça, deux runs du même trigger
    # sur les deux tailles partageraient dossier de run et nom déployé.
    if not tag and _train_type(ds, family) == 'flux2klein':
        tag = _base_tag_for(
            FLUX2KLEIN_BASE_LABELS['9b' if _flux2klein_is_9b(ds) else '4b'])
    return tag


def _run_name(ds, base_model=_PERSISTED, family=None) -> str:
    """Nom de dossier de run unique par (user, trigger, base, FAMILLE) - évite qu'un
    même trigger_word chez deux datasets partage/écrase les dossiers, isole un run
    sur base custom du run officiel, ET isole les familles entre elles. `base_model`
    absent → base persistée ; fourni (même '') → cette base précise.

    Fix B (2026-07-01) : le tag vient de `_dest_base_tag` (et non `_base_tag`), donc
    un run **Krea** porte le suffixe `_Krea-2-Turbo` dans le NOM DE DOSSIER. Sans ça,
    Z-Image base-officielle (tag vide) et Krea (base vide) au même trigger tombaient
    dans le même dossier `u{user}_{trigger}` → ai-toolkit mélangeait les deux runs et
    l'import récupérait le mauvais checkpoint. zimage/sdxl restent nommés à l'identique."""
    tag = _dest_base_tag(ds, base_model, family)
    return f'u{ds.user_id}_{_safe_trigger(ds)}{tag}'


def find_run_collision(user_id, dataset_id, base_model=_PERSISTED):
    """Autre dataset du MÊME user qui produirait le même dossier de run
    (`u{user}_{trigger}{base_tag}`) que (dataset_id, base_model). C'est la source
    de collision : ai-toolkit auto-resume depuis ce dossier → LoRA mélangés, et
    deux lancements simultanés corrompent l'`optimizer.pt` partagé (incident
    Test/Test 2, 2026-06-16). Retourne le FaceDataset en conflit, ou None.

    La clé de collision est le dossier (trigger + base) ; la variante
    (turbo/deturbo) n'y entre PAS → deux datasets « même trigger + même base » se
    télescopent quoi qu'il arrive. On compare le run-name CIBLE (base en cours de
    sélection) aux run-names PERSISTÉS des autres datasets du user."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return None
    target = _run_name(ds) if base_model is _PERSISTED else _run_name(ds, base_model)
    others = (FaceDataset.query
              .filter(FaceDataset.user_id == str(ds.user_id),
                      FaceDataset.id != int(ds.id))
              .all())
    for o in others:
        if _run_name(o) == target:
            return o
    return None


def _masks_dir(dataset_folder: str) -> str:
    """Dossier des masques d'un export (convention mask_path ai-toolkit : dossier
    frère, mêmes noms de fichiers)."""
    return f'{dataset_folder}_masks'


def _mask_fields(dataset_folder: str) -> dict:
    """Champs `mask_path`/`mask_min_value` à fusionner dans l'entrée datasets de la
    job-config SI des masques ont été exportés (masked training, méthode jandordoe :
    fond pondéré à 10 % de la loss → l'identité se lie au sujet, pas au décor).
    Dossier absent/vide → {} (l'entraînement reste strictement l'historique)."""
    md = _masks_dir(dataset_folder)
    try:
        if os.path.isdir(md) and any(f.lower().endswith('.png') for f in os.listdir(md)):
            return {'mask_path': md, 'mask_min_value': 0.1}
    except OSError:
        pass
    return {}


def export_dataset_to_aitoolkit(user_id, dataset_id, masked: bool = True, dest_dir=None) -> str:
    """Écrit les images `keep` en paires .png/.txt dans
    DATASETS_DIR/<trigger>. Le caption = caption éditée + trigger (le trigger
    est toujours présent même si la caption est vide). Retourne le dossier.

    `masked` (défaut ON) : génère aussi un masque « personne » par image (rembg
    u2net, subprocess CPU - cf app/services/person_mask) dans `<dossier>_masks` →
    la job-config passe en MASKED TRAINING (fond à 10 %). Échec des masques =
    jamais bloquant : l'entraînement part simplement sans masques (loggé).

    `dest_dir` (cloud seam) : exporte LÀ au lieu de DATASETS_DIR/<run_name> - ne
    requiert PAS ai-toolkit configuré localement (pas d'appel à _datasets_dir()).
    Défaut (None) = comportement historique inchangé."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if masked and fds.is_conceptual(ds):
        # A person-mask would erase the very thing we want the LoRA to learn (the
        # recurring act for a concept; the whole-image rendering for a style - which
        # lives as much in backgrounds as in people). Force masked training OFF for
        # concept AND style datasets even if the caller/UI asked for it -- server guard.
        logger.info('dataset %s %s -> masked training forced OFF (server guard)',
                    dataset_id, ds.kind)
        masked = False
    trigger = _safe_trigger(ds)
    out = str(dest_dir) if dest_dir else str(_datasets_dir() / _run_name(ds))
    if os.path.isdir(out):
        shutil.rmtree(out)  # ré-export propre
    masks_out = _masks_dir(out)
    if os.path.isdir(masks_out):
        shutil.rmtree(masks_out)  # jamais de masques périmés (ré-export ou toggle OFF)
    os.makedirs(out, exist_ok=True)
    kept = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None)).all())
    if not kept:
        raise ValueError('no kept images to export')
    n = 0
    exported = []
    for img in kept:
        src = os.path.join(fds._dataset_dir(img.dataset_id), img.filename)
        if not os.path.isfile(src):
            continue
        stem = f'{trigger}_{n:03d}'
        dst = os.path.join(out, f'{stem}.png')
        Image.open(src).convert('RGB').save(dst, 'PNG')
        exported.append(dst)
        cap = (img.caption or '').strip()
        body = f'{trigger}, {cap}' if cap else trigger
        with open(os.path.join(out, f'{stem}.txt'), 'w', encoding='utf-8') as fh:
            fh.write(body)
        n += 1
    if n == 0:
        raise ValueError('no valid image file found on disk')
    masked_ok = False
    if masked:
        # generate_person_masks returns a DICT ({"ok", "written", "results"}, or {}
        # on any failure/unavailability) -- a non-empty dict is always truthy, so a
        # verbatim `if wrote:` on the return value would never take the cleanup
        # branch. Read the actual count instead.
        res = generate_person_masks(exported, masks_out)
        wrote = int(res.get('written') or 0) if isinstance(res, dict) else 0
        if wrote:
            masked_ok = True
            logger.info(f'export dataset {dataset_id}: {wrote}/{n} masque(s) personne -> {masks_out}')
        else:
            logger.warning(f'export dataset {dataset_id}: masques indisponibles - training SANS masked loss')
            if os.path.isdir(masks_out):
                shutil.rmtree(masks_out, ignore_errors=True)
    # A REQUESTED masked run that produced no masks (rembg missing, or generation
    # crashed at runtime) silently trains UNMASKED. Record it per-run so the live
    # progress view can warn — instead of the fallback being invisible. `masked` is
    # the FINAL intent: concept/style were already forced OFF above (by design), so
    # they never set this flag.
    queue_manager._set_system_state('training_masks_skipped', bool(masked and not masked_ok),
                                    ttl_seconds=_TRAIN_STATE_TTL)
    logger.info(f'export dataset {dataset_id} -> {out} ({n} paires)')
    return out


# --- Overrides STYLE (communs aux 3 familles) -----------------------------------
# Un LoRA de style n'a PAS de trigger (il teinte toute image dès qu'il est chargé) :
# on retire trigger_word de la config pour qu'ai-toolkit n'injecte rien dans les
# captions. Et on monte le caption dropout à 30 % : le modèle voit régulièrement
# l'image SANS caption, ce qui lie le rendu au LoRA lui-même plutôt qu'aux mots —
# la reco usuelle des styles sans trigger (le 5 % character sert l'association
# trigger→identité, sans objet ici).
_STYLE_CAPTION_DROPOUT = 0.30


def _apply_style_overrides(ds, process: dict) -> dict:
    """Mute la config d'UN process ai-toolkit pour un dataset style. No-op sinon."""
    if not fds.is_style(ds):
        return process
    process.pop('trigger_word', None)
    for d in process.get('datasets', ()):
        d['caption_dropout_rate'] = _STYLE_CAPTION_DROPOUT
    # timestep_type 'sigmoid' est la reco LoRA de SUJET (cf commentaire zimage) ;
    # pour un style on retombe sur le défaut ai-toolkit de la famille.
    if process.get('train', {}).get('timestep_type') == 'sigmoid':
        process['train'].pop('timestep_type')
    return process


def build_job_config(ds, dataset_folder: str, steps: int = 3000, training_folder=None) -> dict:
    """Job-config ai-toolkit pour le preset officiel `zimage:turbo`
    (« Z-Image Turbo w/ Training Adapter »). Clés alignées sur ce que génère
    l'UI ai-toolkit (ui/src/app/jobs/new/options.ts) + structure LoRA 24 Go de
    référence - vérifiées au runtime contre la version installée (cf. spec §3).
    Points non négociables : arch='zimage', name_or_path='Tongyi-MAI/Z-Image-Turbo',
    assistant_lora_path = l'adapter de training (retiré à l'inférence),
    quantize qfloat8 + low_vram pour tenir sur 24 Go.

    SDXL (train_type='sdxl') part dans une branche dédiée (_build_job_config_sdxl) -
    le chemin zimage ci-dessous reste strictement inchangé.

    `training_folder` (cloud seam) : utilisé TEL QUEL comme process.training_folder
    dans les 3 familles - aucun appel à _output_dir() (pas d'ai-toolkit local requis).
    Défaut (None) = comportement historique inchangé (_output_dir() / _run_name(ds))."""
    if _train_type(ds) == 'sdxl':
        cfg_ = _build_job_config_sdxl(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0])
        return cfg_
    if _train_type(ds) == 'krea':
        cfg_ = _build_job_config_krea(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0])
        return cfg_
    if _train_type(ds) == 'flux':
        cfg_ = _build_job_config_flux(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0])
        return cfg_
    if _train_type(ds) == 'flux2klein':
        cfg_ = _build_job_config_flux2klein(ds, dataset_folder, steps, training_folder=training_folder)
        _apply_style_overrides(ds, cfg_['config']['process'][0])
        return cfg_
    trigger = _safe_trigger(ds)
    base_model = getattr(ds, 'train_base_model', None)
    variant = (getattr(ds, 'train_variant', None) or 'turbo').lower()

    # Base : officielle (repo HF diffusers) OU merge ComfyUI converti en diffusers.
    model = {'arch': 'zimage', 'quantize': True, 'quantize_te': True,
             'low_vram': True, 'qtype': 'qfloat8'}
    if base_model:
        from .zimage_convert import converted_dir
        model['name_or_path'] = converted_dir(base_model)       # dossier diffusers converti
        model['extras_name_or_path'] = 'Tongyi-MAI/Z-Image-Turbo'  # tokenizer/TE/VAE partagés
    else:
        model['name_or_path'] = 'Tongyi-MAI/Z-Image-Turbo'
    # Adapter de dé-distillation : UNIQUEMENT pour la variante Turbo (distillée).
    # Base / De-Turbo sont déjà non distillés → pas d'adapter (chargé à -1.0 sinon).
    if variant == 'turbo':
        model['assistant_lora_path'] = ('ostris/zimage_turbo_training_adapter/'
                                        'zimage_turbo_training_adapter_v2.safetensors')
    # Previews : Turbo = 8 steps / cfg 1 ; non-distillé = plus de steps + CFG réel.
    sample_steps, guidance = (8, 1) if variant == 'turbo' else (25, 4)
    _zrank = _lora_rank(ds, 'zimage')   # défaut 16 (choix user) ; éditable via train_settings

    cfg_ = {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_output_dir() / _run_name(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _zrank, 'zimage'),
                'save': {'dtype': 'float16', 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    # 5% de dropout caption : le modèle voit parfois le trigger seul,
                    # ce qui renforce l'association trigger→identité (reco LoRA de
                    # sujet ; l'identité doit vivre dans le trigger, pas les mots).
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    # 'sigmoid' = reco runbook pour un LoRA de sujet (l'exemple
                    # ai-toolkit confirme : "for just subject, change to sigmoid").
                    'timestep_type': _timestep_type_eff(ds, 'sigmoid'),
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_lr_sched_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',   # cohérence avec SDXL : défaut ai-toolkit = False (booléen) → fragile
                    'sample_every': _sample_every(ds),
                    'guidance_scale': guidance,
                    'sample_steps': sample_steps,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }
    _apply_style_overrides(ds, cfg_['config']['process'][0])
    return cfg_


def _build_job_config_krea(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit pour Krea 2. Deux bases selon `train_variant` (cf.
    _krea_is_raw), toutes deux arch='krea2', alignées sur l'UI ai-toolkit
    (ui/src/app/jobs/new/options.ts) :

    - RAW (défaut, reco officielle « train on Raw, validate on Turbo ») :
      name_or_path='krea/Krea-2-Raw' (non distillé), AUCUN assistant_lora_path (rien
      à dé-distiller), previews en CFG 4 / 25 steps (le Raw a besoin d'un vrai CFG).
      1er run = download des poids Raw (~24 Go) et run > 4 h → d'où _TRAIN_STATE_TTL 12 h.
    - TURBO (opt-in, VRAM-friendly) : name_or_path='krea/Krea-2-Turbo' + l'adapter de
      training Ostris (retiré à l'inférence, comme Z-Image), previews CFG 1 / 8 steps.

    Commun : quantize qfloat8 + low_vram pour tenir sur 24 Go. ⚠️ Requiert ai-toolkit
    À JOUR (commit « Add support for Krea2 », arch 'krea2') sinon l'arch est inconnue
    (garde _aitoolkit_supports_krea). Réseau = 'lora' : VÉRIFIÉ canonique 2026-06-26.
    Résolution KREA_TRAIN_RESOLUTION (1024, TE déchargé) car 768 seul tenait sinon."""
    trigger = _safe_trigger(ds)
    is_raw = _krea_is_raw(ds)
    _krank = _lora_rank(ds, 'krea')   # défaut 32/32 (recherche) ; éditable via train_settings
    model = {
        'arch': 'krea2',
        'name_or_path': 'krea/Krea-2-Raw' if is_raw else 'krea/Krea-2-Turbo',
        'quantize': True, 'quantize_te': True, 'low_vram': True, 'qtype': 'qfloat8',
    }
    # Adapter de dé-distillation : Turbo UNIQUEMENT (le Raw est déjà non distillé →
    # rien à retirer ; le charger dessus dégraderait le training).
    if not is_raw:
        model['assistant_lora_path'] = ('ostris/krea2_turbo_training_adapter/'
                                        'krea2_turbo_training_adapter_v1.safetensors')
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_output_dir() / _run_name(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _krank, 'krea'),
                'save': {'dtype': 'float16', 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    # Pré-cache les embeddings du Qwen3-VL pour pouvoir le DÉCHARGER pendant le
                    # training (cf. unload_text_encoder) → libère ~4-8 Go → 1024 tient sans offload.
                    # Valide ici car train_text_encoder=False (sorties figées → cachables sans perte).
                    'cache_text_embeddings': True,
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'unload_text_encoder': True,  # décharge le Qwen3-VL après caching → VRAM pour le DiT 12B → 1024 rapide
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    'timestep_type': _timestep_type_eff(ds, 'linear'),  # défaut canonique krea2 (options.ts)
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_lr_sched_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    # Turbo (distillé) : cfg 1 / 8 steps ; Raw (non distillé) : cfg 4 / 25 steps.
                    'guidance_scale': 4 if is_raw else 1,
                    'sample_steps': 25 if is_raw else 8,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


def _build_job_config_flux(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit pour FLUX.1-dev (arch='flux'). Valeurs VÉRIFIÉES contre
    l'ai-toolkit installé : `ui/.../options.ts` (entrée 'flux' : name_or_path
    'black-forest-labs/FLUX.1-dev', quantize + quantize_te True, sampler /
    noise_scheduler 'flowmatch') ET le notebook officiel `FLUX_1_dev_LoRA_Training`
    (linear/alpha 16, lr 1e-4, previews guidance 4 / 20 steps).

    arch='flux' est une arch CŒUR d'ai-toolkit (toolkit/config_modules.py) — supportée
    par tout ai-toolkit, donc AUCUNE garde de version (contrairement à krea2, extension).
    FLUX.1-dev est un modèle GATED sur Hugging Face : le 1er run télécharge ~24 Go et
    exige un HF_TOKEN ayant accepté la licence (même mécanique que Krea, aussi gated).

    VRAM : Flux est un DiT 12B (même classe que Krea 2). On ajoute low_vram + qfloat8
    (comme Krea, dont la mesure LDS a montré la nécessité à 24 Go) au-dessus des defaults
    options.ts — curseur basse-VRAM = la résolution 768 (cf. _train_res / KREA_TRAIN)."""
    trigger = _safe_trigger(ds)
    _frank = _lora_rank(ds, 'flux')   # défaut 16 (exemple flux officiel) ; éditable via train_settings
    model = {
        'arch': 'flux',
        'name_or_path': 'black-forest-labs/FLUX.1-dev',
        'quantize': True, 'quantize_te': True, 'low_vram': True, 'qtype': 'qfloat8',
    }
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_output_dir() / _run_name(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _frank, 'flux'),
                'save': {'dtype': 'float16', 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    # 'sigmoid' = reco LoRA de SUJET pour les modèles flowmatch (l'exemple
                    # flux d'ai-toolkit documente ce choix ; identique à Z-Image).
                    'timestep_type': _timestep_type_eff(ds, 'sigmoid'),
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_lr_sched_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    'guidance_scale': 4,   # FLUX.1-dev : guidance ~4 (notebook officiel)
                    'sample_steps': 20,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


def _build_job_config_flux2klein(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit pour FLUX.2 Klein. Deux tailles selon `train_variant`
    (cf. _flux2klein_is_9b) : arch='flux2_klein_4b' (défaut, voie locale 16-24 Go)
    ou 'flux2_klein_9b' (32-48 Go, voie cloud surtout). Valeurs VÉRIFIÉES contre
    l'ai-toolkit installé : `ui/.../options.ts` (entrées flux2_klein_4b/9b) et
    `extensions_built_in/diffusion_models/flux2/flux2_klein_model.py`.

    Divergences vs le chemin flux (options.ts fait foi) :
    - timestep_type 'weighted' — le défaut canonique des deux entrées Klein
      (PAS 'sigmoid' comme flux/zimage) ;
    - model_kwargs {'match_target_res': False} — clé propre à cette arch,
      absente du chemin flux ;
    - base NON distillée (flux2_is_guidance_distilled=False côté ai-toolkit) →
      les previews utilisent un VRAI CFG : guidance 4 / 25 steps (les défauts
      « non distillé » de l'UI ai-toolkit — même duo que Krea Raw), là où
      FLUX.1-dev (guidance-distillé) sample en guidance 4 / 20 steps.

    Les deux name_or_path sont des modèles GATED sur Hugging Face : accepter la
    licence + HF_TOKEN avant le 1er run, même mécanique que FLUX.1-dev et Krea.
    ⚠️ Contrairement à 'flux' (arch CŒUR), flux2_klein_* sont des EXTENSIONS →
    garde de version obligatoire (_aitoolkit_supports_flux2klein) sinon
    get_model_class retombe en silence sur le loader SD legacy (LoRA corrompu).
    quantize/low_vram/qfloat8 comme les autres familles ; curseur basse-VRAM =
    la résolution 768 (cf. _train_res)."""
    trigger = _safe_trigger(ds)
    is_9b = _flux2klein_is_9b(ds)
    _fkrank = _lora_rank(ds, 'flux2klein')   # défaut 16 ; éditable via train_settings
    model = {
        'arch': 'flux2_klein_9b' if is_9b else 'flux2_klein_4b',
        'name_or_path': ('black-forest-labs/FLUX.2-klein-base-9B' if is_9b
                         else 'black-forest-labs/FLUX.2-klein-base-4B'),
        'quantize': True, 'quantize_te': True, 'low_vram': True, 'qtype': 'qfloat8',
        'model_kwargs': {'match_target_res': False},
    }
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_output_dir() / _run_name(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _fkrank, 'flux2klein'),
                'save': {'dtype': 'float16', 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'flowmatch',
                    'timestep_type': _timestep_type_eff(ds, 'weighted'),
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_lr_sched_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'flowmatch',
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    # Base non distillée → vrai CFG (cf. docstring) : 4 / 25 steps.
                    'guidance_scale': 4,
                    'sample_steps': 25,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


def _build_job_config_sdxl(ds, dataset_folder: str, steps: int, training_folder=None) -> dict:
    """Job-config ai-toolkit arch='sdxl' - valeurs VÉRIFIÉES dans ai-toolkit
    ui/.../options.ts (entrée 'sdxl', 2026-06-14) : quantize/quantize_te False,
    noise_scheduler/sampler 'ddpm', timestep_type DÉSACTIVÉ, guidance 6. Base =
    checkpoint SDXL ComfyUI local (single-file, pas de conversion)."""
    trigger = _safe_trigger(ds)
    base_model = getattr(ds, 'train_base_model', None)
    if not base_model:
        raise ValueError('SDXL: a base checkpoint is required')
    model = {'arch': 'sdxl', 'name_or_path': _sdxl_base_path(base_model),
             'quantize': False, 'quantize_te': False}
    _srank = _lora_rank(ds, 'sdxl')   # défaut 32 ; alpha = rank/2 (demi-force, conservé)
    return {
        'job': 'extension',
        'config': {
            'name': f'lora_{trigger}',
            'process': [{
                'type': 'sd_trainer',
                'training_folder': (training_folder if training_folder
                                    else str(_output_dir() / _run_name(ds))),
                'device': 'cuda:0',
                'trigger_word': trigger,
                'network': _network_block(ds, _srank, 'sdxl'),
                'save': {'dtype': 'float16', 'save_every': _save_every(ds),
                         'max_step_saves_to_keep': _max_step_saves(ds)},
                'datasets': [{
                    'folder_path': dataset_folder,
                    'caption_ext': 'txt',
                    'caption_dropout_rate': 0.05,
                    'cache_latents_to_disk': True,
                    'resolution': _train_res(ds),
                    **_mask_fields(dataset_folder),
                }],
                'train': {
                    'batch_size': 1,
                    'steps': steps,
                    'gradient_accumulation': _grad_accum(ds),
                    'train_unet': True,
                    'train_text_encoder': False,
                    'gradient_checkpointing': True,
                    'noise_scheduler': 'ddpm',   # SDXL = epsilon/DDPM (≠ flowmatch Z-Image)
                    'optimizer': _optimizer_eff(ds),
                    'lr': _lr_eff(ds),
                    'dtype': 'bf16',
                    **_lr_sched_fields(ds),
                },
                'model': model,
                'sample': {
                    'sampler': 'ddpm',
                    # neg='' EXPLICITE : sans cette clé, ai-toolkit met neg=False (booléen) et le
                    # tokenizer CLIP de transformers 5.x rejette [False] → ValueError au sample
                    # baseline (« text input must be of type str »). SDXL crashait juste avant la
                    # 1re step. '' est un str valide → sample sans négatif (voulu pour un LoRA sujet).
                    'neg': '',
                    'sample_every': _sample_every(ds),
                    'guidance_scale': 6,
                    'sample_steps': 28,
                    'prompts': _sample_prompts(ds, trigger),
                },
            }],
        },
    }


_CK_RE = re.compile(r'_(\d{4,})\.safetensors$')


def _run_dir(user_id, dataset_id, base_model=_PERSISTED, family=None) -> str:
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # ai-toolkit écrit ses checkpoints/samples dans <training_folder>/<name>/
    # où name = 'lora_<trigger>' (cf. build_job_config). On pointe ce sous-dossier.
    # `base_model` cible le run d'une base PRÉCISE (sélection UI) ; `family` cible la
    # famille sélectionnée (Krea vs Z-Image) - sans quoi le panneau montre les
    # checkpoints du mauvais run quand deux familles partagent le même trigger.
    return str(_output_dir() / _run_name(ds, base_model, family) / f'lora_{_safe_trigger(ds)}')


def open_training_folder(user_id, dataset_id, target='loras', family=None,
                         base_model=_PERSISTED) -> str:
    """Ouvre dans l'explorateur de fichiers du POSTE (app locale mono-utilisateur,
    le navigateur tourne sur la même machine) le dossier demandé :
    'loras' → dossier d'import ComfyUI de la famille (loras/krea, loras/sdxl,
    loras/z image) ; 'run' → dossier de checkpoints du run courant (base+famille) ;
    'dataset' → dossier des images du dataset (data/datasets/<id>/ — où « 💾 Write
    .txt files » dépose les captions sidecar ; aucune dépendance ai-toolkit).
    Cibles FIXES résolues côté serveur — le client n'envoie jamais de chemin.
    Crée le dossier au besoin (avant un premier import il n'existe pas encore).
    Retourne le chemin ouvert."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if target == 'run':
        path = _run_dir(user_id, dataset_id, base_model, family)
    elif target == 'loras':
        path = _lora_dest_dir(ds, family)
    elif target == 'dataset':
        path = fds._dataset_dir(dataset_id)
    else:
        raise ValueError('unknown folder target')
    os.makedirs(path, exist_ok=True)
    if os.name == 'nt':
        os.startfile(path)                                   # Explorateur Windows
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])
    logger.info('open folder (%s): %s', target, path)
    return path


def list_checkpoints(user_id, dataset_id, base_model=_PERSISTED, family=None) -> list[dict]:
    """Checkpoints .safetensors du run de la base+famille données (absentes → persistées),
    triés par step croissant. Retour: [{step:int, filename:str, final?:bool}].

    Inclut le fichier FINAL `lora_<trigger>.safetensors` (écrit à la fin d'un run
    abouti, SANS numéro de step) : c'est le résultat terminé, et le regex numéroté
    l'excluait → le LoRA fini était invisible/non importable depuis le panneau."""
    run = _run_dir(user_id, dataset_id, base_model, family)
    if not os.path.isdir(run):
        return []
    out = []
    for f in os.listdir(run):
        m = _CK_RE.search(f)
        if m:
            out.append({'step': int(m.group(1)), 'filename': f})
    out.sort(key=lambda c: c['step'])
    # Fichier final (run = .../lora_<trigger> → lora_<trigger>.safetensors).
    final_name = os.path.basename(run) + '.safetensors'
    if os.path.isfile(os.path.join(run, final_name)):
        last = out[-1]['step'] if out else 0
        out.append({'step': last, 'filename': final_name, 'final': True})
    # Provenance annotation: which dataset VERSION most plausibly produced
    # each file (newest registry record older than the file). Pre-feature
    # datasets have no records -> no annotation, shape unchanged otherwise.
    from . import checkpoint_registry
    ds = fds.get_dataset(user_id, dataset_id)
    fam = _train_type(ds, family) if ds else None
    for c in out:
        try:
            rec = checkpoint_registry.record_for_mtime(
                dataset_id, fam, os.path.getmtime(os.path.join(run, c['filename'])))
        except OSError:
            rec = None
        if rec is not None:
            c['version'] = rec.version
            c['source'] = rec.source
            c['trained_at'] = rec.created_at.isoformat() if rec.created_at else None
    return out


def import_checkpoint(user_id, dataset_id, filename, base_model=_PERSISTED, family=None,
                      src_dir=None, version=None) -> str:
    """Copie le checkpoint choisi vers le dossier loras de ComfyUI : loras/z image/
    pour Z-Image, loras/sdxl/ pour SDXL, loras/krea/ pour Krea (routage par famille,
    pour ne pas polluer le Test Studio Z-Image). Anti path-traversal :
    le filename doit appartenir à la liste des checkpoints du run.

    Le nom de DESTINATION encode la base d'entraînement (_base_tag) : ai-toolkit
    écrit toujours `lora_<trigger>_<step>.safetensors` quel que soit le modèle de
    base (le `name` du job n'est pas base-aware), donc un LoRA entraîné sur un
    merge ComfyUI et un autre entraîné sur la base officielle produisent des
    fichiers IDENTIQUES qui, une fois copiés dans le dossier partagé de ComfyUI,
    sont indiscernables et s'écrasent au même step. On insère ici le tag du merge
    (`lora_<trigger>_<step>_<merge>.safetensors`) - la base officielle reste sans
    suffixe - pour les rendre reconnaissables ET éviter la collision. Le fichier
    source ai-toolkit n'est pas renommé (l'auto-resume continue de fonctionner).

    `base_model`/`family` ciblent le run d'une base+famille précises (sélection UI) ;
    absents → persistés. Run dir, whitelist, dossier ET suffixe de destination
    utilisent la MÊME base+famille → cohérent (un LoRA Krea part bien en loras/krea).

    `src_dir` (cloud seam) : le checkpoint est lu LÀ (dossier de staging où le pod a
    déposé le résultat téléchargé) au lieu du run ai-toolkit local - aucun besoin
    d'ai-toolkit configuré (ni _run_dir(), ni list_checkpoints(), qui appellent tous
    deux _output_dir()). La whitelist ici est PUREMENT anti-traversal : tout
    .safetensors réellement présent dans src_dir est autorisé (pas de filtre de
    forme _CK_RE — le checkpoint FINAL d'un run abouti, `lora_<trigger>.safetensors`,
    n'a pas de suffixe de step et doit passer). Défaut (None) = comportement
    historique inchangé."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if src_dir:
        run_dir = str(src_dir)
        try:
            allowed = {f for f in os.listdir(run_dir)
                       if f.lower().endswith('.safetensors')}
        except OSError:
            allowed = set()
    else:
        run_dir = _run_dir(user_id, dataset_id, base_model, family)
        allowed = {c['filename'] for c in list_checkpoints(user_id, dataset_id, base_model, family)}
    if filename not in allowed:
        raise ValueError('unknown checkpoint')
    # Déploiement routé par famille : sdxl → loras/sdxl, krea → loras/krea, sinon
    # « z image » (ne pollue pas le Test Studio Z-Image ; un LoRA Krea atterrit
    # directement dans le dossier lu par le menu de génération Krea).
    dest_dir = _lora_dest_dir(ds, family)
    os.makedirs(dest_dir, exist_ok=True)
    tag = _dest_base_tag(ds, base_model, family)
    # Dataset-version suffix (_v3): makes successive dataset states
    # distinguishable in the ComfyUI/Test Studio dropdowns AND prevents a
    # cloud/local re-run of a CHANGED dataset from silently overwriting the
    # deployed LoRA of the previous version. `version` is passed explicitly by
    # the cloud import (the run knows its version); local imports resolve the
    # file's run via the provenance registry (file mtime vs launch times).
    # No registry rows (pre-feature datasets) -> no suffix, names unchanged.
    if version is None and not src_dir:
        from . import checkpoint_registry
        try:
            mtime = os.path.getmtime(os.path.join(run_dir, filename))
            rec = checkpoint_registry.record_for_mtime(
                dataset_id, _train_type(ds, family), mtime)
            version = rec.version if rec else None
        except OSError:
            version = None
    stem, ext = os.path.splitext(filename)
    # Cloud jobs are named `lds<run>_u<user>_<trigger>_<base>` on the pod, so
    # their checkpoints arrive as `lds12_ulocal_tata_cv_Krea-2-Raw_000000250`.
    # Deployed as-is, that stem is invisible to every trigger-prefix matcher
    # (Test Studio's `lora_<trigger>_…` whitelist, labels) — "my cloud
    # checkpoints are unusable", user-reported — and the deploy suffix used to
    # re-append a base tag the stem already carried. Normalize to the LOCAL
    # ai-toolkit convention at deploy time: `lora_<trigger>[_<step>]`, rebuilt
    # from the dataset's own trigger (no string surgery on the tag).
    if re.match(r'^lds\d+_u[0-9A-Za-z]+_', stem):
        step = re.search(r'_(\d{6,10})$', stem)
        stem = f'lora_{_safe_trigger(ds)}' + (f'_{step.group(1)}' if step else '')
    suffix = f'{tag}' + (f'_v{int(version)}' if version else '')
    dest_name = f'{stem}{suffix}{ext}' if suffix else filename
    dest = os.path.join(dest_dir, dest_name)
    shutil.copy2(os.path.join(run_dir, filename), dest)
    logger.info(f'import checkpoint {filename} -> {dest}')
    return dest


def list_imported_checkpoints(user_id, dataset_id, family=None) -> list[dict]:
    """LoRA de CE dataset déjà déployés dans le dossier loras de la FAMILLE demandée
    (chargeables par le Test Studio / la page generate). [{filename, label}].
    `family` (sélecteur UI) prime sur le train_type persisté : sans ça, la liste
    « IN COMFYUI (loras/…) » montrait toujours la famille persistée (ex. Krea) même
    quand l'utilisateur regardait la page Z-Image ou SDXL.

    Single-user app: no ownership DB to filter against (SRC's list_test_checkpoints
    consulted lora_ownership to hide LoRA belonging to OTHER users) -- everything on
    disk that matches this dataset's trigger boundary IS this dataset's checkpoint.
    A direct filesystem scan of the family's deploy folder replaces that call.
    `filename` is returned in LoraLoader form (family-subfolder\\name.safetensors),
    matching delete_imported_checkpoint's path resolution."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        return []
    fam = _train_type(ds, family)
    prefix = f'lora_{_safe_trigger(ds)}'
    try:
        dest_dir = _lora_dest_dir(ds, family)
    except RuntimeError:
        return []
    if not os.path.isdir(dest_dir):
        return []
    from ..utils.comfyui import format_trained_lora_label
    # Cloud-trained checkpoints are auto-imported into the same folder but
    # named after the pod job (`lds<N>_<run>…`), not `lora_<trigger>…` — the
    # prefix filter alone hid them from the "IN COMFYUI" list even though the
    # files were right there (user-observed 2026-07-13). Accept any filename
    # that IS a known cloud checkpoint of THIS dataset.
    cloud_names = set()
    cloud_prefixes = set()
    try:
        from ..models import CloudTrainingRun
        for r in CloudTrainingRun.query.filter_by(dataset_id=dataset_id).all():
            if r.checkpoint_local_path:
                cloud_names.add(os.path.basename(r.checkpoint_local_path))
            # Every staging file of this run starts with its pod-job prefix
            # (`lds<id>_…`, see cloud_training job_name). Matching on the prefix
            # covers EVERY harvested epoch AND survives the `_<base_tag>` +
            # `_v<N>` suffixes import_checkpoint appends to the deployed name —
            # the exact-basename match above misses both (user-observed
            # 2026-07-13: imports succeeded but "in ComfyUI" stayed at 0).
            cloud_prefixes.add(f'lds{r.id}_')
    except Exception:
        pass
    subfolder = os.path.basename(os.path.normpath(dest_dir))
    out = []
    for fn in sorted(os.listdir(dest_dir)):
        if not fn.lower().endswith('.safetensors'):
            continue
        # deployed cloud names may carry the _v<N> dataset-version suffix —
        # strip it before matching against the staging basenames
        stem = re.sub(r'_v\d+(?=\.safetensors$)', '', fn)
        if not _trigger_boundary(fn, prefix) \
                and fn not in cloud_names and stem not in cloud_names \
                and not any(fn.startswith(p) for p in cloud_prefixes):
            continue
        out.append({'filename': os.path.join(subfolder, fn),
                    'label': format_trained_lora_label(fn, fam) or fn})
    return out


def delete_imported_checkpoint(user_id, dataset_id, filename, family=None) -> str:
    """Supprime un checkpoint déployé du dossier loras de ComfyUI. Garde-fous :
    le filename doit appartenir aux checkpoints importés du dataset (whitelist,
    famille-scopée) ET le chemin résolu doit rester dans le dossier loras de la
    FAMILLE sélectionnée (z image / sdxl / krea) - anti path-traversal, fail-closed.
    `family` (menu UI) prime sur le train_type persisté, comme la liste affichée."""
    ds = fds.get_dataset(user_id, dataset_id)
    allowed = {c['filename'] for c in list_imported_checkpoints(user_id, dataset_id, family=family)}
    if filename not in allowed:
        raise ValueError('unknown checkpoint')
    # ds is guaranteed truthy here: an unowned/missing dataset makes
    # list_imported_checkpoints return [] above, which already raised.
    root = os.path.abspath(_lora_dest_dir(ds, family))
    loras_root = os.path.dirname(root)
    rel = filename.replace('\\', os.sep).replace('/', os.sep)
    dest = os.path.abspath(os.path.join(loras_root, rel))
    if os.path.commonpath([dest, root]) != root or not os.path.isfile(dest):
        raise ValueError('file not found')
    # trash, never destroy: a wrong click on a deployed LoRA is recoverable
    # until 'Empty trash' in Settings.
    from . import trash
    trash.send_to_trash(dest, context=f'lora_ds{dataset_id}')
    logger.info(f'trashed imported checkpoint {dest}')
    return os.path.basename(dest)


def _local_training_active_for(dataset_id) -> bool:
    """True while THIS dataset trains locally — its run dir is being written
    (deleting a checkpoint ai-toolkit is about to rewrite invites corruption)."""
    try:
        if not queue_manager._get_system_state('training_in_progress'):
            return False
        active_ds = queue_manager._get_system_state('training_dataset_id')
        return active_ds is not None and int(active_ds) == int(dataset_id)
    except Exception:
        return False


def delete_checkpoint(user_id, dataset_id, filename, base_model=_PERSISTED,
                      family=None) -> str:
    """Move ONE run-dir checkpoint to the trash. Whitelisted against
    list_checkpoints (anti path-traversal), refused while this dataset trains
    locally. Returns the trashed filename."""
    if _local_training_active_for(dataset_id):
        raise ValueError('this dataset is training right now — stop the run '
                         'before deleting its checkpoints')
    allowed = {c['filename'] for c in
               list_checkpoints(user_id, dataset_id, base_model, family)}
    if filename not in allowed:
        raise ValueError('unknown checkpoint')
    run_dir = _run_dir(user_id, dataset_id, base_model, family)
    from . import trash
    trash.send_to_trash(os.path.join(run_dir, filename),
                        context=f'ckpt_ds{dataset_id}')
    return filename


def cleanup_checkpoints(user_id, dataset_id, keep, base_model=_PERSISTED,
                        family=None) -> dict:
    """'Clean up this run': trash every run-dir checkpoint NOT in `keep`
    (typically the final + the best-epoch pick). Returns {'removed', 'kept'}."""
    if _local_training_active_for(dataset_id):
        raise ValueError('this dataset is training right now — stop the run '
                         'before cleaning its checkpoints')
    keep_set = {str(k) for k in (keep or [])}
    run_dir = _run_dir(user_id, dataset_id, base_model, family)
    from . import trash
    removed = 0
    for c in list_checkpoints(user_id, dataset_id, base_model, family):
        if c['filename'] in keep_set:
            continue
        try:
            trash.send_to_trash(os.path.join(run_dir, c['filename']),
                                context=f'cleanup_ds{dataset_id}')
            removed += 1
        except OSError as e:
            logger.warning('cleanup: could not trash %s: %s', c['filename'], e)
    return {'removed': removed, 'kept': sorted(keep_set)}


def _dir_size(path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def dataset_disk_usage(user_id, dataset_id, base_model=_PERSISTED, family=None) -> dict:
    """Where this dataset's training bytes live: the selected run dir, the
    cloud staging dirs of its runs, and its deployed LoRA. Best-effort."""
    out = {'run_dir_bytes': 0, 'cloud_staging_bytes': 0, 'deployed_bytes': 0}
    try:
        rd = _run_dir(user_id, dataset_id, base_model, family)
        if os.path.isdir(rd):
            out['run_dir_bytes'] = _dir_size(rd)
    except Exception:
        pass
    try:
        from ..models import CloudTrainingRun
        for r in CloudTrainingRun.query.filter_by(dataset_id=dataset_id).all():
            if r.staging_dir and os.path.isdir(r.staging_dir):
                out['cloud_staging_bytes'] += _dir_size(r.staging_dir)
    except Exception:
        pass
    try:
        ds = fds.get_dataset(user_id, dataset_id)
        root = _lora_dest_dir(ds, family)
        for c in list_imported_checkpoints(user_id, dataset_id, family=family):
            p = os.path.join(os.path.dirname(root),
                             c['filename'].replace('\\', os.sep))
            try:
                out['deployed_bytes'] += os.path.getsize(p)
            except OSError:
                pass
    except Exception:
        pass
    out['total_bytes'] = sum(v for k, v in out.items() if k.endswith('_bytes'))
    return out


def _trigger_boundary(name: str, prefix: str) -> bool:
    """`name` commence par `prefix` ET la suite est vide ou commence par `_`/`.` -
    frontière de trigger EXACTE. Évite que « Lola » attrape « Lola2 »/« Lola69382 »
    (le caractère après le préfixe doit être un séparateur, pas un chiffre/lettre)."""
    if not name.startswith(prefix):
        return False
    rest = name[len(prefix):]
    return rest == '' or rest[0] in '_.'


def purge_training_artifacts(user_id, trigger_safe) -> list[str]:
    """Supprime TOUS les artefacts d'entraînement d'un (user, trigger), appelé à la
    suppression d'un dataset : LoRA déployés dans ComfyUI (z image + sdxl + krea), run
    ai-toolkit (output/), export (datasets/) et job config (config/generated/).

    Sécurité : matching sur la FRONTIÈRE EXACTE du trigger (jamais un sibling type
    Lola vs Lola2) ; les noms viennent d'os.listdir (bare, pas de path-traversal) ;
    trigger vide → no-op (sinon `u{user}_` balaierait tout). Retourne les chemins
    retirés (pour log/affichage). Idempotent : un 2e appel ne retire plus rien.

    Each backend (ComfyUI loras dir / ai-toolkit output+datasets dirs) is probed
    independently -- an unconfigured backend just yields no roots to sweep for
    that step instead of aborting the whole purge (this runs from
    face_dataset_service.delete_dataset as best-effort cleanup)."""
    trigger_safe = (trigger_safe or '').strip()
    if not trigger_safe or user_id in (None, ''):
        return []
    removed: list[str] = []
    run_prefix = f'u{user_id}_{trigger_safe}'    # ex. u1_Lola69382
    lora_prefix = f'lora_{trigger_safe}'         # ex. lora_Lola69382
    # 1) LoRA déployés dans ComfyUI (z image + sdxl + krea + flux + flux2klein séparés)
    lora_roots = []
    for accessor in (_lora_dest_dir_zimage, _lora_dest_dir_sdxl, _lora_dest_dir_krea,
                     _lora_dest_dir_flux, _lora_dest_dir_flux2klein):
        try:
            lora_roots.append(str(accessor()))
        except RuntimeError:
            pass
    for root in lora_roots:
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if fn.endswith('.safetensors') and _trigger_boundary(fn, lora_prefix) and os.path.isfile(p):
                try:
                    os.remove(p); removed.append(p)
                except OSError as e:
                    logger.warning('purge: remove %s échoué : %s', p, e)
    # 2) run output + 3) export datasets (dossiers entiers)
    output_datasets_roots = []
    for accessor in (_output_dir, _datasets_dir):
        try:
            output_datasets_roots.append(str(accessor()))
        except RuntimeError:
            pass
    for root in output_datasets_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if _trigger_boundary(name, run_prefix) and os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True); removed.append(p)
    # 4) job configs : nommés d'après le run name (base/famille), donc un même
    #    trigger peut en avoir plusieurs (ex. un run zimage + un run krea). On
    #    balaie tout config dont le stem est sur la frontière de ce trigger,
    #    comme les étapes 2-3 pour les dossiers.
    try:
        jobs_dir = str(_jobs_dir())
    except RuntimeError:
        jobs_dir = None
    if jobs_dir and os.path.isdir(jobs_dir):
        for fn in os.listdir(jobs_dir):
            if not fn.endswith('.json'):
                continue
            p = os.path.join(jobs_dir, fn)
            if _trigger_boundary(fn[:-len('.json')], run_prefix) and os.path.isfile(p):
                try:
                    os.remove(p); removed.append(p)
                except OSError as e:
                    logger.warning('purge: remove %s échoué : %s', p, e)
    logger.info('purge_training_artifacts u%s/%s : %d artefact(s) retiré(s)',
                user_id, trigger_safe, len(removed))
    return removed


def write_job_config(ds, dataset_folder: str, steps: int = 3000) -> str:
    job_cfg = build_job_config(ds, dataset_folder, steps=steps)
    # Name by the base/family-aware run name, NOT the trigger alone: a zimage run
    # and a krea run of the same trigger have distinct run names everywhere else
    # (training_folder, dataset_folder), so keying this file by trigger only made
    # the second launch silently clobber the first's config record.
    path = _jobs_dir() / f'{_run_name(ds)}.json'
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(job_cfg, fh, indent=2)
    return str(path)


def recommended_steps(dataset_id) -> int:
    """Steps cibles selon le *type* de dataset — la recette suit le dataset, pas l'inverse.

    Character (défaut) : ~120 steps/image, bornés [1500, 3500]. On verrouille une
    identité sur un petit set curé (~100-150 vues/image, consensus des guides
    ai-toolkit/Z-Image) ; un 3000 fixe surentraînait les petits datasets et
    sous-entraînait les gros. À 25 images (preset équilibré) ça redonne 3000.

    Concept / style : échelle SOUS-LINÉAIRE (√n), bornée [2000, 12000]. Un concept
    doit généraliser, pas mémoriser : plus le set grossit, moins chaque image doit
    être vue. Appliquer le taux « character » (120/img) à 400 images donnerait
    48 000 steps (overfit garanti) ; le clamp à 3500 donnait l'inverse (sous-
    entraîné). 475·√n colle aux deux points d'ancrage du consensus : ~30-40 images
    de style → ~3000 steps (guides Z-Image/SDXL), ~400 images → ~9500 steps
    (~24 vues/image, retours communautaires sur les gros sets concept/style).
    """
    ds = FaceDataset.query.get(dataset_id)
    n = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep').count()
    if ds is not None and (ds.kind or 'character') in ('concept', 'style'):
        target = int(round(475 * math.sqrt(max(n, 1)), -2))
        return max(2000, min(12000, target))
    target = int(round(n * 120, -2))  # ~120 steps/image, arrondi à la centaine
    return max(1500, min(3500, target))


def default_steps(ds) -> int:
    """Adaptive step count for a dataset — single source of truth shared by
    local launch_training and cloud training (parity guarantee). Thin ds-based
    wrapper over recommended_steps(dataset_id) (the calc used by launch_training
    when steps=None) so callers holding the ds object don't need the id."""
    return recommended_steps(ds.id)


def recommended_steps_info(dataset_id) -> dict:
    """Version « transparente » de recommended_steps pour l'UI : le nombre + le
    pourquoi, afin que l'app apprenne au débutant au lieu de décider en boîte
    noire. Ne mute rien."""
    ds = FaceDataset.query.get(dataset_id)
    n = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep').count()
    kind = (ds.kind or 'character') if ds is not None else 'character'
    steps = recommended_steps(dataset_id)
    if kind in ('concept', 'style'):
        views = round(steps / n, 1) if n else 0
        what = 'style' if kind == 'style' else 'concept'
        rationale = (f"{what.capitalize()} — {n} images kept. Sublinear scaling (475·√n, "
                     f"clamped 2000–12000): the bigger the set, the fewer views per "
                     f"image (~{views}/img here), so the LoRA generalizes the {what} "
                     f"instead of memorizing shots. Variety matters more than count.")
    else:
        rationale = (f"Character — {n} images kept. ~120 steps/image (clamped "
                     f"1500–3500): a small curated set seen many times locks the "
                     f"identity without drifting.")
    return {'steps': steps, 'kind': kind, 'n_images': n, 'rationale': rationale}


# --- Preflight d'entraînement (garde-fous, lecture seule) -----------------------
# Plancher DUR / recommandé par famille. Sous le plancher → blocker ; entre les
# deux → warning à confirmer. 10 images fixes pour tout le monde sous-estimait
# SDXL (booru, plus gourmand en variété) et laissait passer des runs voués au
# surapprentissage.
TRAIN_MIN_IMAGES = {'zimage': (12, 20), 'sdxl': (20, 30), 'krea': (15, 20), 'flux': (15, 20),
                    'flux2klein': (15, 20)}
_FAMILY_LABEL = {'zimage': 'Z-Image', 'sdxl': 'SDXL', 'krea': 'Krea 2', 'flux': 'FLUX.1',
                 'flux2klein': 'FLUX.2 Klein'}
# VRAM mesurée : Krea 2 (12B) sature un 24 GB à 1024 (cf. KREA_TRAIN_RESOLUTION). Flux
# est un DiT de même classe (12B) → même seuil recommandé.
_KREA_MIN_VRAM_GB = 24
# flux2klein est VOLONTAIREMENT absent : le check est variant-aveugle (la variante
# se choisit au lancement, après ce preflight) et le défaut 4B tient en 16-24 Go —
# un warning « il faut ~24 GB » serait un faux positif sur la voie locale normale.
# Le 9B (32-48 Go) est la voie cloud ; un seuil 24 le sous-estimerait de toute façon.
_VRAM24_FAMILIES = ('krea', 'flux')   # familles 12B qui recommandent ~24 GB à 1024


def training_preflight(user_id, dataset_id, train_type=None) -> dict:
    """Pre-launch sanity report: {'blockers': [...], 'warnings': [...]}. Blockers
    stop the launch (too few images for the family); warnings ask for one explicit
    confirm in the UI. Pure reads — never mutates, never raises on probe failures
    (an unknown GPU must not block a run).

    Émet AUSSI `checks` (liste structurée {id,label,status,detail,target}) +
    `verdict` ('ready'|'warnings'|'blocked') pour la pastille de préparation du
    workspace — construits DANS LA MÊME PASSE que blockers/warnings (une seule
    source de vérité, aucune règle dupliquée). `target` = id de section du
    workspace (gf-generate/gf-images) où corriger — None quand rien à cibler.
    NB : le check 'captioned' (images gardées sans caption) est un fail dans
    `checks` (assert_trainable refusera le launch) mais volontairement PAS un
    blocker ici — le flux modal existant (launch → erreur explicite) est conservé."""
    from .face_variations import caption_has_identity_leak
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    ttype = _train_type(ds, train_type)
    label = _FAMILY_LABEL.get(ttype, ttype)
    blockers, warnings = [], []
    checks = []

    def _check(cid, clabel, status, detail, target=None):
        checks.append({'id': cid, 'label': clabel, 'status': status,
                       'detail': detail, 'target': target})

    rows = FaceDatasetImage.query.filter_by(dataset_id=dataset_id).all()
    kept = [r for r in rows if r.status == 'keep' and r.filename]
    n = len(kept)
    # CONCEPT / STYLE : plusieurs dimensions ci-dessous (équilibre de composition,
    # fuite d'identité) sont des heuristiques de LoRA PERSONNAGE sans objet quand
    # l'invariant du set n'est pas une identité — on les saute pour ne pas générer
    # de faux avertissements.
    concept = fds.is_conceptual(ds)

    # 1) minimum d'images par famille
    floor, reco = TRAIN_MIN_IMAGES.get(ttype, (12, 20))
    if n < floor:
        blockers.append(f'{n} kept image(s) — the hard minimum for a {label} LoRA is {floor}. '
                        'Generate or import more before training.')
        _check('images', 'Enough images', 'fail',
               f'{n} kept — the hard minimum for {label} is {floor}', 'gf-generate')
    elif n < reco:
        warnings.append(f'{n} kept image(s) — {reco} recommended for a solid {label} LoRA.')
        _check('images', 'Enough images', 'warn',
               f'{n} kept — {reco}+ recommended for a solid {label} LoRA', 'gf-generate')
    else:
        _check('images', 'Enough images', 'ok', f'{n} kept ({reco}+ recommended)')

    # 2) équilibre de composition — heuristique PERSONNAGE (viser un mix face/bust/body/
    # back pour rendre un visage à toutes les distances). Sans objet pour un CONCEPT (il
    # s'apprend sur les cadrages tels quels), et un dataset non classé (framing=None) y
    # déclencherait un faux « tout en gros plan visage » → on saute pour les concepts.
    if n and not concept:
        comp = {'face': 0, 'bust': 0, 'body': 0, 'back': 0}
        for r in kept:
            if r.framing in comp:
                comp[r.framing] += 1
        _comp_ok = True
        if comp['bust'] + comp['body'] + comp['back'] == 0:
            warnings.append('every kept image is a face shot — the LoRA will struggle to '
                            'render busts and full-body scenes.')
            _check('composition', 'Framing balance', 'warn',
                   'all kept images are face shots — add bust/body shots', 'gf-generate')
            _comp_ok = False
        if fds.is_body_fidelity(ds) and comp['body'] == 0:
            warnings.append('body fidelity is ON but there is no full-body shot — the body '
                            "can't be learned without body images.")
            if _comp_ok:
                _check('composition', 'Framing balance', 'warn',
                       'body fidelity is ON but there is no full-body shot', 'gf-generate')
                _comp_ok = False
        if _comp_ok:
            _check('composition', 'Framing balance', 'ok',
                   f"face {comp['face']} · bust {comp['bust']} · body {comp['body']} · back {comp['back']}")

    # 3bis) toutes les gardées ont une caption — WARN, plus un mur : le launch
    # demande un confirm (« train anyway ») au lieu de refuser (UNCAPTIONED:
    # dans assert_trainable). Les captions restent fortement recommandées.
    uncaptioned = sum(1 for r in kept if not (r.caption or '').strip())
    if n:
        if uncaptioned:
            warnings.append(f'{uncaptioned}/{n} kept image(s) have no caption — '
                            'strongly recommended; launching will ask you to confirm.')
            _check('captioned', 'Every kept image captioned', 'warn',
                   f'{uncaptioned}/{n} kept image(s) have no caption — launching asks to confirm', 'gf-images')
        else:
            _check('captioned', 'Every kept image captioned', 'ok', f'{n}/{n} captioned')

    # 3) captions suspectes (trop courtes / dupliquées)
    caps = [(r.caption or '').strip() for r in kept if (r.caption or '').strip()]
    if caps:
        _cap_ok = True
        short = sum(1 for c in caps if len(c.split()) < 8)
        if short / len(caps) > 0.3:
            warnings.append(f'{short}/{len(caps)} caption(s) are very short (<8 words) — '
                            'weak captions weaken prompt control.')
            _check('caption_quality', 'Caption quality', 'warn',
                   f'{short}/{len(caps)} captions are very short (<8 words)', 'gf-images')
            _cap_ok = False
        if len(set(c.lower() for c in caps)) < len(caps) * 0.7:
            warnings.append('many captions are identical — the model learns nothing from '
                            'repeated text; re-caption for variety.')
            if _cap_ok:
                _check('caption_quality', 'Caption quality', 'warn',
                       'many captions are identical — re-caption for variety', 'gf-images')
                _cap_ok = False
        if _cap_ok:
            _check('caption_quality', 'Caption quality', 'ok',
                   'varied, ≥8 words')

    # 4) fuite d'identité — on RETIENT les images fautives (pas juste le compte) pour
    # que l'UI liste lesquelles au moment du preflight, éditables sur place.
    # CONCEPT : décrire l'identité (visage/cheveux/corps) est VOULU — c'est le concept,
    # pas le visage, qui se lie au trigger → la « fuite d'identité » n'a aucun sens ici.
    # On saute entièrement cette dimension (comme le badge caption_leak du payload), sinon
    # CHAQUE caption concept déclenche un faux avertissement au preflight.
    body = fds.is_body_fidelity(ds)
    leak_images = [] if concept else [
        {'id': r.id, 'filename': r.filename, 'caption': (r.caption or '').strip()}
        for r in kept
        if (r.caption or '').strip()
        and caption_has_identity_leak((r.caption or '').strip(), body=body)]
    if leak_images:
        warnings.append(f'{len(leak_images)} caption(s) still describe the identity (face/hair'
                        f'{"/body marks" if body else ""}) — it will bind to those words '
                        'instead of the trigger. Re-caption or edit them.')
        _check('leaks', 'No identity leaks', 'warn',
               f'{len(leak_images)} caption(s) describe hair/face/skin — identity will bind '
               'to those words, not the trigger', 'gf-images')
    elif caps and not concept:
        _check('leaks', 'No identity leaks', 'ok', '0 leaking caption')

    # 5) quasi-doublons parmi les kept (dHash pairwise, n<=~60 -> négligeable). On
    # retient les PAIRES (leurs deux images) pour que l'UI montre lesquelles rejeter.
    dup_pairs = []
    try:
        hp = []  # [(row, dhash)] pour les kept lisibles sur disque
        for r in kept:
            p = fds._img_path(r)
            if p and os.path.exists(p):
                with Image.open(p) as im:
                    hp.append((r, fds._dhash(im)))
        for i in range(len(hp)):
            for j in range(i + 1, len(hp)):
                if fds._hamming(hp[i][1], hp[j][1]) <= fds.SCRAPE_DHASH_MAX_DISTANCE:
                    ra, rb = hp[i][0], hp[j][0]
                    dup_pairs.append({'a': {'id': ra.id, 'filename': ra.filename},
                                      'b': {'id': rb.id, 'filename': rb.filename}})
        if dup_pairs:
            warnings.append(f'{len(dup_pairs)} pair(s) of kept images are near-duplicates — '
                            'the model overfits repeated content; reject one of each pair.')
            _check('duplicates', 'No near-duplicates', 'warn',
                   f'{len(dup_pairs)} near-duplicate pair(s) — reject one of each', 'gf-images')
        elif n:
            _check('duplicates', 'No near-duplicates', 'ok', '0 pair')
    except Exception:
        pass   # best-effort: an unreadable file must not block the preflight

    # 11) images encore en attente de tri (elles ne s'entraînent PAS)
    untriaged = sum(1 for r in rows if r.status == 'pending' and r.filename)
    if untriaged:
        warnings.append(f'{untriaged} image(s) still await triage (✓/✕) — they will NOT '
                        'be part of the training.')
        _check('triage', 'Everything triaged', 'warn',
               f'{untriaged} image(s) still await ✓/✕ — they will NOT train', 'gf-images')
    elif rows:
        _check('triage', 'Everything triaged', 'ok', 'no image awaiting ✓/✕')

    # 7) VRAM (Krea 2 mesuré à 24 GB ; None = inconnu, jamais bloquant)
    try:
        from .. import capabilities
        vram = capabilities.gpu_vram_gb()
        if vram is not None and ttype in _VRAM24_FAMILIES and vram < _KREA_MIN_VRAM_GB:
            warnings.append(f'{label} training needs ~{_KREA_MIN_VRAM_GB} GB of VRAM at 1024 '
                            f'— this GPU reports {vram} GB; expect OOM or extreme slowness. '
                            'Drop the resolution to 768 in Advanced options to fit.')
            _check('vram', 'GPU memory', 'warn',
                   f'{label} needs ~{_KREA_MIN_VRAM_GB} GB VRAM — this GPU reports {vram} GB')
    except Exception:
        pass

    # Verdict agrégé pour la pastille : un fail = 🔴, sinon un warn = 🟡, sinon 🟢.
    statuses = {c['status'] for c in checks}
    verdict = ('blocked' if 'fail' in statuses
               else 'warnings' if 'warn' in statuses else 'ready')

    return {'blockers': blockers, 'warnings': warnings,
            # Détail « lesquelles » pour l'UI : images dont la caption fuit, et paires
            # quasi-doublons — le message reste agrégé, mais on peut drill-down + agir.
            'leak_images': leak_images, 'dup_pairs': dup_pairs,
            'checks': checks, 'verdict': verdict,
            'kept': n, 'floor': floor, 'recommended': reco}


# --- Garde-fou espace disque ---------------------------------------------------
# Un run plein (10 checkpoints ~0,3-2 Go + latents/samples) et une conversion
# diffusers (~12 Go) qui crashent à 90 % pour cause de disque plein laissent des
# artefacts corrompus. On refuse AVANT, avec un message actionnable.
MIN_FREE_GB_TRAIN = 10
MIN_FREE_GB_CONVERT = 15


def free_disk_gb(path) -> float | None:
    """Free space (GB) on the drive holding `path` (climbs to the nearest existing
    parent — the target dir may not exist yet). None if it can't be determined
    (never blocks on a stat failure)."""
    try:
        p = os.path.abspath(str(path))
        while p and not os.path.exists(p):
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return None


def assert_free_disk(path, min_gb, what) -> None:
    """Raise ValueError when the drive holding `path` has under `min_gb` GB free."""
    free = free_disk_gb(path)
    if free is not None and free < min_gb:
        raise ValueError(
            f'not enough disk space for {what}: {free:.1f} GB free on the target drive, '
            f'~{min_gb} GB needed - free up space and retry')


def _log_tail(path: str, n: int = 30) -> str:
    """Dernières `n` lignes d'un fichier log (pour remonter une erreur ai-toolkit)."""
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            return ''.join(fh.readlines()[-n:]).strip()
    except OSError:
        return '(log illisible)'


def _watch_training(app, proc, log_path, dataset_id) -> None:
    """Thread daemon : attend la fin du process ai-toolkit puis fait avancer la
    file (libère ComfyUI / lance le suivant) DÈS la fin, sans dépendre du polling
    client. Sur un crash (rc≠0), remonte la fin du log. process_training_queue()
    reste le filet de secours si Flask redémarre (le watcher meurt, le flag est
    rattrapé au prochain poll ou à l'expiration du TTL)."""
    try:
        proc.wait()
        rc = proc.returncode
    except Exception:
        return
    try:
        with app.app_context():
            if rc not in (0, None):
                tail = _log_tail(log_path)
                logger.error("Entraînement ai-toolkit dataset %s terminé en ERREUR (rc=%s). "
                             "Fin du log :\n%s", dataset_id, rc, tail)
                # Surface l'erreur à l'UI (sinon un crash = juste « terminé » silencieux).
                queue_manager._set_system_state(
                    'training_error', {'dataset_id': dataset_id, 'rc': rc, 'log_tail': tail[-1500:]},
                    ttl_seconds=3600)
            else:
                logger.info("Entraînement ai-toolkit dataset %s terminé (rc=%s).", dataset_id, rc)
            process_training_queue()  # libère le GPU / enchaîne la file immédiatement
    except Exception as e:
        logger.warning("watcher training : post-traitement échoué : %s", e)


def archive_previous_run(ds) -> str | None:
    """Écarte le dossier du run existant (rename en `*_archived_<horodatage>`,
    jamais de suppression) pour que le prochain lancement reparte de ZÉRO au lieu
    de l'auto-resume ai-toolkit — le cas « j'ai remanié le dataset, je veux un
    LoRA neuf ». Les checkpoints archivés restent sur disque (récupérables à la
    main) et tombent avec le dataset : le nom garde le préfixe `lora_<trigger>`
    donc purge_training_artifacts les balaie aussi. Les copies déjà importées
    dans ComfyUI (loras/<famille>) ne sont pas touchées. None si aucun run."""
    run_dir = _output_dir() / _run_name(ds)
    if not run_dir.is_dir():
        return None
    dest = f'{run_dir}_archived_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    try:
        os.rename(run_dir, dest)
    except OSError as e:
        # Dossier verrouillé (ex. antivirus, explorateur ouvert) → message actionnable.
        raise ValueError(f'could not archive the previous run ({e}) - close anything '
                         f'using "{run_dir}" and retry')
    logger.info('fresh training: previous run archived -> %s', dest)
    return dest


def launch_training(user_id, dataset_id, steps: int | None = None, check_captions: bool = True,
                    base_model=None, variant: str | None = None, train_type: str | None = None,
                    allow_caption_mismatch: bool = False, masked: bool = True,
                    fresh: bool = False, allow_uncaptioned: bool = False) -> dict:
    """Export + config + pause ComfyUI (flag) + lance l'entraînement ai-toolkit
    en CLI headless (`run.py <config>`).

    ``steps`` = step cible (None → calculé par recommended_steps selon le nombre
    d'images). ai-toolkit reprend AUTOMATIQUEMENT depuis le dernier checkpoint
    présent dans le training_folder (get_latest_save_path), donc relancer avec un
    steps > dernier_step continue l'entraînement. ``fresh=True`` écarte d'abord le
    run existant (archive_previous_run) → repart de zéro sur le dataset actuel.

    Retourne {pid, config_path, log_path}. Raises RuntimeError if ai-toolkit isn't
    installed/configured (route maps this to 409, not 400 - it's a backend
    availability problem, not a bad request)."""
    if not is_installed():
        raise RuntimeError('ai-toolkit is not configured')
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Disque plein à mi-run = checkpoints corrompus ; refuser AVANT d'exporter.
    assert_free_disk(_output_dir(), MIN_FREE_GB_TRAIN, 'a training run')
    # Garde-fou anti double-lancement : un entraînement DÉJÀ vivant (flag levé +
    # pid en vie) → refuser. Deux process sur le même GPU/dossier corrompent
    # l'optimizer partagé (incident Test/Test 2). Un pid mort avec flag encore
    # levé (avance de file) passe : on ne bloque que sur un process réellement vivant.
    if (queue_manager._get_system_state('training_in_progress', False)
            and _pid_alive(queue_manager._get_system_state('training_pid', None))):
        raise ValueError('a training is already in progress - wait for it to finish or queue this dataset')
    if check_captions:
        assert_trainable(dataset_id, train_type=train_type,
                         allow_caption_mismatch=allow_caption_mismatch,
                         allow_uncaptioned=allow_uncaptioned)
    # Base d'entraînement : None/'' = officielle ; sinon un merge ComfyUI qui DOIT
    # avoir été converti en diffusers d'abord (gate). On persiste le choix sur le
    # dataset → _run_name/_run_dir/list_checkpoints deviennent base-aware (run isolé).
    base_model = (base_model or '').strip() or None
    variant = (variant or '').strip().lower()
    # La famille de CE lancement vient du param train_type s'il est donné, sinon du
    # dataset — c'est elle qui fixe l'enum de variantes valide (flux2klein : 4b/9b ;
    # les autres : turbo/base/deturbo) et le défaut (Krea → Raw, flux2klein → 4B).
    launch_fam = _train_type(ds, train_type)
    if variant not in _valid_variants_for(launch_fam):
        variant = _default_variant_for(launch_fam)
    if train_type is not None:
        ds.train_type = train_type
    # Conversion diffusers : UNIQUEMENT pour Z-Image (SDXL = single-file direct,
    # pas de conversion → on ne bloque pas sur is_converted).
    if base_model and _train_type(ds) == 'zimage':
        from .zimage_convert import is_converted
        if not is_converted(base_model):
            raise ValueError('custom base not converted - prepare it first (button "Convert base")')
    # SDXL : la base vient brute du body → whitelist serveur (anti path-traversal,
    # comme prepare-base le fait pour Z-Image). Refus immédiat si inconnue.
    if base_model and _train_type(ds) == 'sdxl' and base_model not in _sdxl_base_choices():
        raise ValueError('unknown SDXL checkpoint')
    # Krea 2 : refuser TÔT si l'ai-toolkit installé n'a pas l'arch krea2 (sinon
    # fallback silencieux vers le loader SD legacy → mauvais modèle, plantage confus).
    if _train_type(ds) == 'krea' and not _aitoolkit_supports_krea():
        raise ValueError(
            "ai-toolkit doesn't support Krea 2 yet (krea2 arch missing) - "
            "update it (git pull) before training a Krea LoRA.")
    # FLUX.2 Klein : même garde que Krea (archs d'EXTENSION, fallback SD silencieux
    # sur un ai-toolkit pas à jour → LoRA corrompu, cf. _aitoolkit_supports_flux2klein).
    if _train_type(ds) == 'flux2klein' and not _aitoolkit_supports_flux2klein():
        raise ValueError(
            "ai-toolkit doesn't support FLUX.2 Klein yet (flux2_klein arch missing) - "
            "update it (git pull) before training a FLUX.2 Klein LoRA.")
    # Garde-fou anti-collision de dossier : un AUTRE dataset du user avec le même
    # (trigger, base) écrirait dans le même run → LoRA mélangés. Refuser AVANT de
    # persister/lancer, en nommant le conflit pour que l'utilisateur change un trigger.
    clash = find_run_collision(user_id, dataset_id, base_model=base_model)
    if clash:
        raise ValueError(
            f"training collision: dataset '{clash.name}' (#{clash.id}) already uses "
            f"the same trigger '{ds.trigger_word}' on the same base - they would write "
            f"to the same folder. Change the trigger_word of one of the two before training.")
    ds.train_base_model = base_model
    ds.train_variant = variant
    fds.db.session.commit()
    # Repartir de zéro : écarter le run existant APRÈS la persistance base/variante
    # (_run_name lit les valeurs persistées → on archive bien LE run qui serait repris).
    archived = archive_previous_run(ds) if fresh else None
    # Steps adaptatifs si non imposés ; sinon override borné (jamais < 500).
    steps = default_steps(ds) if steps is None else max(500, int(steps))
    # masked (défaut ON) : masques personne exportés à côté du dataset → la
    # job-config passe en masked training (fond 10 %). OFF ou indispo = historique.
    dataset_folder = export_dataset_to_aitoolkit(user_id, dataset_id, masked=masked)
    config_path = write_job_config(ds, dataset_folder, steps=steps)
    # Provenance registry: record WHICH dataset version this launch trains on
    # (fingerprint + manifest -> human version v1/v2/...). Best-effort — a
    # registry failure must never block a training launch.
    from . import checkpoint_registry
    checkpoint_registry.register_launch(
        user_id, dataset_id, family=_train_type(ds), source='local',
        base_model=base_model or '', variant=variant, masked=bool(masked),
        steps=int(steps), settings=launch_settings_snapshot(ds))
    # Pause GPU longue durée : le superviseur stoppe ComfyUI -> comfyui_ready=False
    # -> le dispatch worker se met en pause tout seul.
    queue_manager._set_system_state('training_error', None, ttl_seconds=1)  # reset crash précédent
    queue_manager._set_system_state('training_in_progress', True, ttl_seconds=_TRAIN_STATE_TTL)
    queue_manager._set_system_state('training_dataset_id', int(dataset_id), ttl_seconds=_TRAIN_STATE_TTL)
    # Step cible : sert à snapshotter le final en nom NUMÉROTÉ à la fin (cf.
    # _snapshot_final_checkpoint) - ai-toolkit écrit le final sans numéro.
    queue_manager._set_system_state('training_target_step', int(steps), ttl_seconds=_TRAIN_STATE_TTL)
    # HF_HOME route les poids base/adapter sur le disque configuré. PYTHONIOENCODING
    # évite les crashs cp1252 sur les logs unicode. Jamais shell=True ; args en liste.
    env = dict(os.environ, HF_HOME=str(_hf_home()), PYTHONIOENCODING='utf-8')
    run_dir = _output_dir() / _run_name(ds)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(run_dir / 'training.log')
    try:
        logf = open(log_path, 'w', encoding='utf-8')
        proc = subprocess.Popen([str(_venv_python()), 'run.py', config_path],
                                cwd=str(_aitoolkit_dir()), env=env, shell=False,
                                stdout=logf, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (FileNotFoundError, OSError) as e:
        queue_manager._set_system_state('training_in_progress', False, ttl_seconds=None)
        raise ValueError(f"could not start training: {e}")
    queue_manager._set_system_state('training_pid', proc.pid, ttl_seconds=_TRAIN_STATE_TTL)
    # Watcher event-driven : libère ComfyUI / enchaîne la file dès la fin du
    # process (le poll de /train/status reste le filet de secours).
    try:
        from flask import current_app
        threading.Thread(target=_watch_training,
                         args=(current_app._get_current_object(), proc, log_path, int(dataset_id)),
                         daemon=True).start()
    except Exception as e:
        logger.warning("watcher training non démarré : %s", e)
    return {'started': True, 'pid': proc.pid, 'config_path': config_path, 'steps': steps,
            'dataset_folder': dataset_folder, 'log_path': log_path,
            'fresh': bool(fresh), 'archived_run': archived}


def continue_training(user_id, dataset_id, extra_steps: int = 1000,
                      base_model=_PERSISTED, variant=None) -> dict:
    """Reprend l'entraînement depuis le dernier checkpoint de la base ciblée et
    vise ``dernier_step + extra_steps``. ai-toolkit auto-resume depuis le
    training_folder ; il faut donc qu'au moins un checkpoint existe POUR CETTE BASE.

    `base_model` absent → base persistée du dataset (ex. file d'attente). Fourni
    (sélection UI) → on reprend le run DE CETTE base précise : sinon on proposait
    « Continuer » sur une base sans run et on relançait en fait l'ancienne base."""
    if queue_manager._get_system_state('training_in_progress', False):
        raise ValueError('a training is already in progress')
    ds = fds.get_dataset(user_id, dataset_id)
    base = (ds.train_base_model if ds else None) if base_model is _PERSISTED else base_model
    var = (variant or (ds.train_variant if ds else None) or 'turbo')
    cks = list_checkpoints(user_id, dataset_id, base_model=base)
    if not cks:
        raise ValueError("no checkpoint to resume for this base - run a training first")
    latest = max(c['step'] for c in cks)
    try:
        extra = max(100, int(extra_steps))
    except (TypeError, ValueError):
        extra = 1000
    # Reprendre AVEC la base/variante ciblée - sinon launch_training les remettrait
    # à l'officiel et ai-toolkit reprendrait depuis le mauvais run.
    res = launch_training(user_id, dataset_id, steps=latest + extra, check_captions=False,
                          base_model=base, variant=var)
    res['resumed_from'] = latest
    res['target_steps'] = latest + extra
    return res


def stop_training() -> None:
    """Tue le process d'entraînement (s'il tourne) PUIS lève le flag → le
    superviseur relance ComfyUI. L'ordre compte : si on levait le flag d'abord,
    ComfyUI reprendrait le GPU pendant que l'entraînement tourne encore."""
    pid = queue_manager._get_system_state('training_pid', None)
    if pid:
        try:
            if os.name == 'nt':
                # /T tue aussi les sous-process (dataloaders, etc.).
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(int(pid))],
                               shell=False, capture_output=True)
            else:
                os.kill(int(pid), 15)
        except (ValueError, OSError) as e:
            logger.warning(f"stop_training: kill pid {pid} échoué : {e}")
    # Stop = arrêt voulu : on VIDE la file D'ABORD (sinon le prochain poll
    # relancerait l'entraînement suivant), PUIS on lève le flag EN DERNIER (c'est
    # lui qui signale à ComfyUI de reprendre le GPU - l'ordre compte).
    _save_queue([])
    queue_manager._set_system_state('training_pid', None, ttl_seconds=None)
    queue_manager._set_system_state('training_in_progress', False, ttl_seconds=None)


def _dataset_name(dataset_id):
    if dataset_id is None:
        return None
    ds = FaceDataset.query.get(int(dataset_id))
    return ds.name if ds else f'#{dataset_id}'


def kept_uncaptioned_count(dataset_id) -> int:
    """Nombre d'images GARDÉES (status keep) sans caption - bloque l'entraînement."""
    return (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter((FaceDatasetImage.caption.is_(None)) | (FaceDatasetImage.caption == ''))
            .count())


def assert_trainable(dataset_id, train_type=None, allow_caption_mismatch=False,
                     allow_uncaptioned=False) -> None:
    """Lève ValueError si le dataset n'est pas prêt : trop peu d'images gardées,
    captions manquantes, ou STYLE de caption incohérent avec le type de modèle
    (SDXL booru-native attend des tags booru ; Z-Image attend de la prose). Le
    `train_type` effectif est passé par l'appelant car il n'est persisté qu'APRÈS
    cet appel. `allow_caption_mismatch=True` = override explicite (bouton « forcer »).
    `allow_uncaptioned=True` = confirm explicite « train anyway » : les captions
    manquantes ne sont plus un mur, juste un « êtes-vous sûr ? » (demande
    utilisateur — pouvoir expérimenter), le préfixe UNCAPTIONED: déclenche le
    confirm côté front comme MISMATCH_CAPTION:."""
    kept = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep').count()
    if kept < 10:
        raise ValueError(f"not enough kept images ({kept}/10)")
    ds_ = FaceDataset.query.get(dataset_id)
    # STYLE : les captions sont OPTIONNELLES (le rendu se lie au LoRA, pas aux mots ;
    # dropout à 30 % de toute façon) → on ne bloque PAS sur les captions manquantes.
    # Mais si des captions EXISTENT, le garde prose↔booru plus bas reste pertinent
    # (un style SDXL captionné en prose = même mismatch qu'un character).
    style = fds.is_style(ds_)
    missing = kept_uncaptioned_count(dataset_id)
    if missing and not style and not allow_uncaptioned:
        raise ValueError(
            f"UNCAPTIONED: {missing} kept image(s) have no caption. Captions are "
            "strongly recommended — whatever a caption does NOT explain binds to "
            "the trigger — but you can train without them.")
    if allow_caption_mismatch:
        return
    # Garde-fou style ↔ type : un LoRA SDXL entraîné sur des captions PROSE = mismatch
    # booru-native → « images disjointes » (recherche 2026-06-14) ; et l'inverse pour Z-Image.
    ttype = (train_type or '').strip().lower()
    if not ttype:
        ds = FaceDataset.query.get(dataset_id)
        ttype = (getattr(ds, 'train_type', None) or 'zimage').lower() if ds else 'zimage'
    expected = 'booru' if ttype == 'sdxl' else 'prose'
    from .face_variations import caption_style
    caps = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.caption.isnot(None)).all())
    sample = [c.caption for c in caps if c.caption and c.caption.strip()][:12]
    if sample:
        booru_n = sum(1 for s in sample if caption_style(s) == 'booru')
        actual = 'booru' if booru_n * 2 >= len(sample) else 'prose'   # vote majoritaire
        if actual != expected:
            if expected == 'booru':
                raise ValueError(
                    "MISMATCH_CAPTION: this SDXL dataset has PROSE captions, but a booru "
                    "model (bigLove type) is prompted with tags. Re-caption in 'Booru tags' mode "
                    "before training, or force the training.")
            raise ValueError(
                "MISMATCH_CAPTION: this Z-Image dataset has booru TAG captions, but Z-Image "
                "expects prose. Re-caption in 'Prose' mode, or force the training.")


def training_status(user_id=None) -> dict:
    cur_id = queue_manager._get_system_state('training_dataset_id', None)
    in_progress = bool(queue_manager._get_system_state('training_in_progress', False))
    return {'in_progress': in_progress,
            'installed': is_installed(),
            'pid': queue_manager._get_system_state('training_pid', None),
            'current': ({'dataset_id': cur_id, 'name': _dataset_name(cur_id)}
                        if (in_progress and cur_id is not None) else None),
            # Dernier crash d'entraînement (rc≠0) remonté par le watcher, pour l'UI.
            'error': queue_manager._get_system_state('training_error', None),
            'queue': train_queue_view(user_id) if user_id is not None else []}


# --- Suivi de progression (log tail + loss curve + samples) -------------------
# ai-toolkit redirige tqdm dans training.log : les mises à jour sont séparées par
# des \r sur une même « ligne », d'où le split sur [\r\n]. Un segment type :
#   lora_x:   2%|▏| 60/3000 [01:23<1:07:41, 1.38s/it, lr: 1.0e+00 loss: 3.412e-01]
_PROG_STEP_RE = re.compile(r'(\d+)/(\d+)')
_PROG_LOSS_RE = re.compile(r'loss[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)')
_PROG_SPEED_RE = re.compile(r'([\d.]+\s*(?:s/it|it/s))')
_PROG_ETA_RE = re.compile(r'<\s*([\d:]+)\s*,')
_SAMPLE_RE = re.compile(r'__(\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$', re.IGNORECASE)
_PROG_LOG_MAX_BYTES = 4 * 1024 * 1024   # tail cap: 3000 tqdm updates ≈ 0.5 MB
_PROG_CURVE_MAX_POINTS = 200
_PROG_SAMPLES_MAX = 24


def _parse_training_log(text: str) -> dict:
    """Extract (step, total, loss, speed, eta, loss_curve) from raw log text.
    Pure function — unit-testable without a real run."""
    out = {'step': None, 'total': None, 'loss': None, 'speed': None, 'eta': None,
           'loss_curve': []}
    curve = []
    for seg in re.split(r'[\r\n]+', text):
        lm = _PROG_LOSS_RE.search(seg)
        # Only trust real tqdm segments ('%|' bar or a loss postfix) — the log also
        # contains incidental 'X/Y' text (dataset counts, resolutions) that must not
        # be read as progress.
        if '%|' not in seg and not lm:
            continue
        sm = None
        for sm in _PROG_STEP_RE.finditer(seg):
            pass                             # last step/total occurrence of the segment
        if not sm:
            continue
        step, total = int(sm.group(1)), int(sm.group(2))
        if total <= 0 or step > total:
            continue                         # e.g. '1024x1024' image sizes, not progress
        out['step'], out['total'] = step, total
        if lm:
            try:
                loss = float(lm.group(1))
            except ValueError:
                continue
            out['loss'] = loss
            if not curve or curve[-1][0] != step:
                curve.append([step, loss])
        spm = _PROG_SPEED_RE.search(seg)
        if spm:
            out['speed'] = spm.group(1).strip()
        em = _PROG_ETA_RE.search(seg)
        if em:
            out['eta'] = em.group(1)
    # Downsample evenly so the payload stays small on long runs.
    if len(curve) > _PROG_CURVE_MAX_POINTS:
        stride = len(curve) / _PROG_CURVE_MAX_POINTS
        curve = [curve[int(i * stride)] for i in range(_PROG_CURVE_MAX_POINTS - 1)] + [curve[-1]]
    out['loss_curve'] = curve
    return out


def _samples_dir(user_id, dataset_id, base_model=_PERSISTED, family=None) -> str:
    return os.path.join(_run_dir(user_id, dataset_id, base_model, family), 'samples')


def list_training_samples(user_id, dataset_id, base_model=_PERSISTED, family=None,
                          limit=_PROG_SAMPLES_MAX) -> list[dict]:
    """Sample previews ai-toolkit writes every sample_every steps
    (<run>/samples/<ts>__<step>_<promptidx>.jpg). Newest steps first, capped
    (limit=None → all, for the best-epoch scoring pass)."""
    d = _samples_dir(user_id, dataset_id, base_model, family)
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        m = _SAMPLE_RE.search(f)
        if m:
            out.append({'filename': f, 'step': int(m.group(1)), 'prompt_idx': int(m.group(2))})
    out.sort(key=lambda s: (-s['step'], s['prompt_idx']))
    return out if limit is None else out[:limit]


def score_checkpoint_samples(user_id, dataset_id, base_model=_PERSISTED, family=None) -> dict:
    """Best-epoch selection (jandordoe method): every training sample is an output
    of the LoRA at its step — scoring their face similarity vs the dataset
    reference (insightface, CPU, one subprocess for the whole set) tells which
    step holds the identity best. The recommended checkpoint is the saved one
    closest to that step.

    Returns {'available': bool, 'reason'?: str, 'steps': [{'step','mean_sim','n'}],
    'best_step': int|None, 'checkpoint': str|None} — never raises on missing
    prerequisites, the UI shows `reason` instead."""
    from . import face_similarity
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        return {'available': False, 'reason': 'this dataset has no reference photo'}
    ref_path = os.path.join(fds._dataset_dir(ds.id), ds.ref_filename)
    if not face_similarity.is_available():
        return {'available': False,
                'reason': 'face scoring is not installed (Quality tools step in Setup)'}
    samples = list_training_samples(user_id, dataset_id, base_model, family, limit=None)
    if not samples:
        return {'available': False, 'reason': 'no training samples yet (they appear every 250 steps)'}
    sdir = _samples_dir(user_id, dataset_id, base_model, family)
    paths = [os.path.join(sdir, s['filename']) for s in samples]
    results, scoring_error = face_similarity.score_dataset_faces(ref_path, paths)
    if not results:
        detail = (scoring_error or {}).get('detail')
        return {'available': False,
                'reason': f'face scoring failed: {detail}' if detail
                else 'face scoring failed (see server log)'}
    by_step = {}
    for s, p in zip(samples, paths):
        r = results.get(p)
        if r and r.get('state') == 'scorable' and r.get('sim') is not None:
            by_step.setdefault(s['step'], []).append(float(r['sim']))
    steps = [{'step': st, 'mean_sim': round(sum(v) / len(v), 4), 'n': len(v)}
             for st, v in sorted(by_step.items())]
    if not steps:
        return {'available': False, 'reason': 'no scorable face in the samples'}
    best = max(steps, key=lambda s: s['mean_sim'])
    # Map the winning sample step to the CLOSEST saved checkpoint (samples every
    # 250 steps, checkpoints every 500 — they rarely align exactly).
    cks = list_checkpoints(user_id, dataset_id, base_model, family)
    ck = min(cks, key=lambda c: abs(c['step'] - best['step']))['filename'] if cks else None
    return {'available': True, 'steps': steps, 'best_step': best['step'], 'checkpoint': ck}


def training_progress(user_id, dataset_id, base_model=_PERSISTED, family=None) -> dict:
    """Live view of a run: parsed log progress + sample listing. Never raises on a
    missing/unreadable log (a run that hasn't started writing yet is normal) —
    only on an unknown dataset (route → 404 via get_dataset)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    cur_id = queue_manager._get_system_state('training_dataset_id', None)
    active = (bool(queue_manager._get_system_state('training_in_progress', False))
              and cur_id is not None and int(cur_id) == int(dataset_id)
              and _pid_alive(queue_manager._get_system_state('training_pid', None)))
    log_path = os.path.join(str(_output_dir() / _run_name(ds, base_model, family)), 'training.log')
    parsed = {'step': None, 'total': None, 'loss': None, 'speed': None, 'eta': None,
              'loss_curve': []}
    log_exists = os.path.isfile(log_path)
    if log_exists:
        try:
            size = os.path.getsize(log_path)
            with open(log_path, encoding='utf-8', errors='replace') as fh:
                if size > _PROG_LOG_MAX_BYTES:
                    fh.seek(size - _PROG_LOG_MAX_BYTES)
                parsed = _parse_training_log(fh.read())
        except OSError:
            log_exists = False
    return {'active': active, 'log_exists': log_exists, **parsed,
            'masks_skipped': bool(active and queue_manager._get_system_state('training_masks_skipped', False)),
            'samples': list_training_samples(user_id, dataset_id, base_model, family)}


# --- File d'attente d'entraînement -------------------------------------------
TRAIN_QUEUE_KEY = 'lora_train_queue'


def _pid_alive(pid) -> bool:
    try:
        import psutil
        return bool(pid) and psutil.pid_exists(int(pid))
    except Exception:
        return False


def get_train_queue() -> list:
    q = queue_manager._get_system_state(TRAIN_QUEUE_KEY, [])
    return q if isinstance(q, list) else []


def _save_queue(q: list) -> None:
    queue_manager._set_system_state(TRAIN_QUEUE_KEY, q, ttl_seconds=None)


def enqueue_training(user_id, dataset_id, extra_steps=None,
                     base_model=_PERSISTED, variant=None, train_type=None,
                     allow_caption_mismatch=False, not_before=None, masked=True,
                     steps=None, allow_uncaptioned=False) -> dict:
    """Ajoute un dataset à la file (lancé à la fin du training courant).

    `base_model`/`variant` permettent de CHOISIR explicitement la base du job en
    file (absent → base persistée). Sans ça, on ne pouvait pas choisir le modèle
    d'un job mis en file pendant qu'un autre entraînement tourne (le sélecteur
    était masqué et l'enqueue réutilisait silencieusement la base persistée).

    `steps` = cible ABSOLUE de steps pour un lancement neuf (None → adaptatif via
    recommended_steps). À NE PAS confondre avec `extra_steps` (mode « continuer »
    = +N steps depuis le dernier checkpoint). Snapshotté dans la file pour que le
    lancement différé respecte le même plafond (ex. « s'arrêter à 2000 »)."""
    ds = fds.get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    # Pas de mise en file si le dataset n'est pas prêt (captions manquantes, etc.).
    if extra_steps is None:
        assert_trainable(dataset_id, train_type=train_type,
                         allow_caption_mismatch=allow_caption_mismatch,
                         allow_uncaptioned=allow_uncaptioned)
    if train_type is not None:
        ds.train_type = train_type
        fds.db.session.commit()
    ttype = _train_type(ds)
    base = (ds.train_base_model if base_model is _PERSISTED else base_model) or None
    var = (variant or ds.train_variant or _default_variant_for(ttype))
    # Base custom (merge) Z-Image = doit être convertie AVANT (SDXL = single-file
    # direct, pas de conversion → on saute la vérif). Refus immédiat et lisible.
    if extra_steps is None and base and ttype == 'zimage':
        from .zimage_convert import is_converted
        if not is_converted(base):
            raise ValueError('custom base not converted - prepare it first (button "Convert base")')
    # SDXL : whitelist serveur de la base (anti path-traversal).
    if base and ttype == 'sdxl' and base not in _sdxl_base_choices():
        raise ValueError('unknown SDXL checkpoint')
    # Krea 2 : même garde qu'au lancement - pas de mise en file d'un job qui
    # tomberait dans le fallback SD legacy faute d'arch krea2 dans l'ai-toolkit.
    if ttype == 'krea' and not _aitoolkit_supports_krea():
        raise ValueError(
            "ai-toolkit doesn't support Krea 2 yet (krea2 arch missing) - "
            "update it (git pull) before queuing a Krea LoRA.")
    # FLUX.2 Klein : même garde qu'au lancement (archs d'extension, cf. launch).
    if ttype == 'flux2klein' and not _aitoolkit_supports_flux2klein():
        raise ValueError(
            "ai-toolkit doesn't support FLUX.2 Klein yet (flux2_klein arch missing) - "
            "update it (git pull) before queuing a FLUX.2 Klein LoRA.")
    # Même garde-fou de collision qu'au lancement : pas de mise en file d'un job
    # qui partagerait le dossier de run d'un autre dataset (même trigger + base).
    clash = find_run_collision(user_id, dataset_id, base_model=base)
    if clash:
        raise ValueError(f"training collision with '{clash.name}' (#{clash.id}): "
                         f"same trigger + same base. Change the trigger_word before queuing.")
    q = get_train_queue()
    if any(int(it.get('dataset_id', -1)) == int(dataset_id) for it in q):
        return {'queued': False, 'reason': 'already queued'}
    # Snapshot de la base/variante/type CHOISIE au moment de la mise en file (le
    # lancement différé doit garder CE choix, pas relancer sur l'officiel/zimage).
    # `not_before` (ISO, heure locale serveur) = entraînement PROGRAMMÉ : le job
    # reste en file jusqu'à l'échéance ; s'il devient dû pendant qu'un autre
    # entraînement tourne, il attend simplement son tour (jamais d'erreur).
    # Cible de steps ABSOLUE (plafond choisi côté UI) - coercition défensive : un
    # '' / 0 / non-numérique retombe sur None (= adaptatif), jamais de crash JSON.
    try:
        steps_target = int(steps) if steps else None
    except (TypeError, ValueError):
        steps_target = None
    q.append({'dataset_id': int(dataset_id), 'user_id': str(user_id), 'extra_steps': extra_steps,
              'base_model': base, 'variant': var, 'train_type': ttype,
              'not_before': not_before, 'masked': bool(masked), 'steps': steps_target})
    _save_queue(q)
    return {'queued': True, 'position': len(q), 'not_before': not_before}


def dequeue_training(dataset_id) -> int:
    q = get_train_queue()
    new = [it for it in q if int(it.get('dataset_id', -1)) != int(dataset_id)]
    _save_queue(new)
    return len(q) - len(new)


def train_queue_view(user_id) -> list:
    out = []
    for it in get_train_queue():
        ds = fds.get_dataset(it.get('user_id', user_id), it.get('dataset_id'))
        bm = it.get('base_model')
        base_label = (os.path.basename(str(bm).replace('\\', '/')).rsplit('.', 1)[0]
                      if bm else 'Official')
        out.append({'dataset_id': it.get('dataset_id'),
                    'name': ds.name if ds else f"#{it.get('dataset_id')}",
                    'extra_steps': it.get('extra_steps'),
                    # Cible de steps absolue choisie à la mise en file (None = adaptatif).
                    'steps': it.get('steps'),
                    'base_model': bm, 'base_label': base_label,
                    # Échéance de programmation (ISO local) - None = dès que possible.
                    'not_before': it.get('not_before')})
    return out


def _launch_queued_item(item) -> None:
    ds_id = item['dataset_id']
    uid = item.get('user_id')
    extra = item.get('extra_steps')
    if extra:
        continue_training(uid, ds_id, extra_steps=extra)  # reprend la base/type persistés
    else:
        launch_training(uid, ds_id, steps=item.get('steps'),
                        base_model=item.get('base_model'),
                        # None → launch_training applique le défaut family-aware (Krea → Raw).
                        variant=item.get('variant'),
                        train_type=item.get('train_type'),
                        masked=item.get('masked', True))


_queue_lock = threading.Lock()


def process_training_queue() -> str | None:
    """Avance la file : si le training courant est FINI (process mort mais flag
    encore levé), lance le suivant ; sinon, si rien ne tourne et la file n'est pas
    vide, lance le prochain. À appeler périodiquement (le poll de /train/status le
    fait). Retourne un libellé d'action ou None. SÉRIALISÉ par _queue_lock : sans
    ça, le watcher et un poll /train/status peuvent avancer la file en même temps
    → double-lancement du même entraînement."""
    with _queue_lock:
        return _advance_training_queue()


def _snapshot_final_checkpoint(dataset_id, step) -> str | None:
    """Copie le final bare `lora_<trigger>.safetensors` vers son nom NUMÉROTÉ
    `lora_<trigger>_<step:09d>.safetensors`. ai-toolkit écrit le résultat final SANS
    numéro de step ; sans ce snapshot :
      - continuer un entraînement écrase ce final sans aucune trace (perte) ;
      - list_checkpoints sous-estime le step de reprise (il compte le bare au DERNIER
        numéro existant, pas à son vrai step) → `continue_training` repart trop bas.
    Le snapshot rend chaque final permanent ET visible à son vrai step. Idempotent
    (ne réécrit jamais un numéroté existant). Retourne le nom créé, ou None."""
    try:
        step = int(step)
    except (TypeError, ValueError):
        return None
    if step <= 0 or dataset_id is None:
        return None
    ds = FaceDataset.query.get(int(dataset_id))
    if not ds:
        return None
    trigger = _safe_trigger(ds)
    run = str(_output_dir() / _run_name(ds) / f'lora_{trigger}')
    final = os.path.join(run, f'lora_{trigger}.safetensors')
    numbered = os.path.join(run, f'lora_{trigger}_{step:09d}.safetensors')
    if not os.path.isfile(final) or os.path.exists(numbered):
        return None
    try:
        shutil.copy2(final, numbered)
        logger.info('snapshot final → %s (step %d)', numbered, step)
        return os.path.basename(numbered)
    except OSError as e:
        logger.warning('snapshot final échoué : %s', e)
        return None


def _due_index(q) -> int | None:
    """Index du premier job DÛ de la file : sans `not_before`, ou dont l'échéance
    (ISO, heure locale serveur) est atteinte. Un job PROGRAMMÉ pour plus tard ne
    bloque pas ceux placés derrière lui. `not_before` illisible → dû (fail-open)."""
    now = datetime.now()
    for i, it in enumerate(q):
        nb = it.get('not_before')
        if not nb:
            return i
        try:
            if datetime.fromisoformat(str(nb)) <= now:
                return i
        except (TypeError, ValueError):
            return i
    return None


def _advance_training_queue() -> str | None:
    flag = bool(queue_manager._get_system_state('training_in_progress', False))
    pid = queue_manager._get_system_state('training_pid', None)
    vision_busy = bool(queue_manager._get_system_state('vision_in_progress', False))
    q = get_train_queue()

    if flag:
        if _pid_alive(pid):
            # Re-arm the 4h TTLs on every poll: without this, a training run
            # longer than 4h would see these flags silently expire mid-run,
            # and the GPU gate (job_queue / gpu_busy_reason) would think
            # nothing is running and let the queue/vision grab the GPU back.
            queue_manager._set_system_state('training_in_progress', True, ttl_seconds=_TRAIN_STATE_TTL)
            queue_manager._set_system_state('training_pid', pid, ttl_seconds=_TRAIN_STATE_TTL)
            cur_dataset_id = queue_manager._get_system_state('training_dataset_id', None)
            if cur_dataset_id is not None:
                queue_manager._set_system_state('training_dataset_id', cur_dataset_id, ttl_seconds=_TRAIN_STATE_TTL)
            cur_target_step = queue_manager._get_system_state('training_target_step', None)
            if cur_target_step is not None:
                queue_manager._set_system_state('training_target_step', cur_target_step, ttl_seconds=_TRAIN_STATE_TTL)
            return None  # toujours en cours
        # Process mort alors que le flag est levé → training terminé.
        # Snapshot du final en nom NUMÉROTÉ (immuable) AVANT d'enchaîner/libérer :
        # sinon un futur « continuer » écrase ce final sans trace. Idempotent, et ce
        # point tourne aussi via le poll /train/status (robuste à un restart Flask).
        try:
            _snapshot_final_checkpoint(
                queue_manager._get_system_state('training_dataset_id', None),
                queue_manager._get_system_state('training_target_step', None))
        except Exception as e:
            logger.warning('snapshot final (advance) échoué : %s', e)
        due = _due_index(q)
        if due is not None and not vision_busy:
            nxt = q[due]
            try:
                _launch_queued_item(nxt)  # remet le flag + un nouveau pid (pas de flap GPU)
                _save_queue(q[:due] + q[due + 1:])  # retirer SEULEMENT après lancement réussi
                logger.info(f"File training : terminé → lancement dataset {nxt['dataset_id']}")
                return f"next:{nxt['dataset_id']}"
            except Exception as e:
                # Échec → on retire l'item (évite une boucle infinie) mais on
                # SURFACE l'erreur au lieu de la perdre silencieusement.
                _save_queue(q[:due] + q[due + 1:])
                queue_manager._set_system_state(
                    'training_queue_error',
                    {'dataset_id': nxt.get('dataset_id'), 'error': str(e)}, ttl_seconds=3600)
                logger.error(f"File training : échec lancement {nxt.get('dataset_id')}: {e}")
                return None
        # File vide (ou uniquement des jobs programmés plus tard) → libérer le GPU
        # (le superviseur relance ComfyUI ; le ticker relancera le job à l'échéance).
        queue_manager._set_system_state('training_in_progress', False, ttl_seconds=1)
        queue_manager._set_system_state('training_pid', None, ttl_seconds=1)
        logger.info("File training : terminé, aucune suite due → flag libéré")
        return 'released'

    due = _due_index(q)
    if due is not None and not vision_busy:
        nxt = q[due]
        try:
            _launch_queued_item(nxt)
            _save_queue(q[:due] + q[due + 1:])  # retirer SEULEMENT après lancement réussi
            logger.info(f"File training : lancement dataset {nxt['dataset_id']}")
            return f"launched:{nxt['dataset_id']}"
        except Exception as e:
            _save_queue(q[:due] + q[due + 1:])
            queue_manager._set_system_state(
                'training_queue_error',
                {'dataset_id': nxt.get('dataset_id'), 'error': str(e)}, ttl_seconds=3600)
            logger.error(f"File training : échec lancement {nxt.get('dataset_id')}: {e}")
            return None
    return None


# --- Programmation d'entraînements (jour + heure) -----------------------------
_scheduler_started = False


def start_training_scheduler(app, interval_seconds=60):
    """Ticker de fond : avance la file toutes les `interval_seconds` MÊME sans
    navigateur ouvert. Sans lui, seuls le poll /train/status et le watcher de fin
    de process faisaient avancer la file - un entraînement programmé à 3 h du
    matin ne serait jamais parti. Idempotent (un seul thread par process)."""
    global _scheduler_started
    if _scheduler_started:
        return
    _scheduler_started = True

    def _tick():
        import time
        while True:
            time.sleep(interval_seconds)
            try:
                with app.app_context():
                    process_training_queue()
            except Exception as e:  # jamais fatal - le tick suivant réessaie
                logger.debug('training scheduler tick: %s', e)

    threading.Thread(target=_tick, daemon=True, name='train-scheduler').start()
    logger.info('Training scheduler démarré (tick %ss)', interval_seconds)
