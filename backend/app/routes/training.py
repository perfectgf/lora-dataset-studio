"""Training API: launch/continue/queue/stop a LoRA training run via ai-toolkit,
plus checkpoint listing/import/delete and Z-Image base-conversion prep.

No login - single local user (`cfg.LOCAL_USER`). Every route except
`/dataset/train/status` is gated on `capabilities.probe()['aitoolkit']['valid']`
(409 with a UI hint): `/train/status` must stay pollable even when
ai-toolkit isn't configured, so it degrades to `{'available': False}` instead.
"""
import os
import re
from datetime import datetime

from flask import Blueprint, current_app, request, jsonify

from .. import capabilities
from .. import config as cfg
from ..config import LOCAL_USER
from ..services import cloud_training as ct
from ..services import face_dataset_service as svc
from ..services import lora_training as lt
from ..services import zimage_convert as zc
from ..utils.comfyui import get_zimage_models, get_checkpoint_models
from ._common import _map_error

bp = Blueprint('training', __name__, url_prefix='/api')


def _require_aitoolkit():
    """None if ai-toolkit is usable, else the (body, status) 409 to return."""
    if not capabilities.probe()['aitoolkit']['valid']:
        return jsonify({'error': 'ai-toolkit is not configured',
                        'hint': 'Set its folder in Settings'}), 409
    return None


def _require_cloud():
    """None if cloud training is configured, else the (body, status) 409 to return."""
    if not capabilities.probe().get('cloud_training'):
        return jsonify({'error': 'Cloud training is not configured',
                        'hint': 'Add your vast.ai API key in Settings'}), 409
    return None


@bp.post('/dataset/<int:dataset_id>/train')
def dataset_train(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    try:
        # steps optionnel : None → adaptatif. base_model='' → officiel ; sinon merge
        # (doit être converti d'abord). variant règle l'adapter de de-distillation.
        # base_model peut être un chemin ABSOLU (« Custom weights… », local-only).
        # vae_path/te_path = overrides SDXL uniquement (le service refuse en 400
        # pour toute autre famille). Présence-conditionnelle : absent → le service
        # garde la valeur persistée (sentinelle _PERSISTED), jamais un reset muet.
        kw = {}
        if 'vae_path' in d:
            kw['vae_path'] = d.get('vae_path')
        if 'te_path' in d:
            kw['te_path'] = d.get('te_path')
        res = lt.launch_training(LOCAL_USER, dataset_id, steps=d.get('steps'),
                                 base_model=d.get('base_model'),
                                 variant=d.get('variant', 'turbo'),
                                 train_type=d.get('train_type'),
                                 allow_caption_mismatch=bool(d.get('allow_caption_mismatch')),
                                 allow_uncaptioned=bool(d.get('allow_uncaptioned')),
                                 allow_unverified_weights=bool(d.get('allow_unverified_weights')),
                                 masked=d.get('masked', True),
                                 # fresh=True : écarte le run existant (archivé, pas
                                 # détruit) → repart de zéro au lieu de l'auto-resume.
                                 fresh=bool(d.get('fresh')), **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/<int:dataset_id>/train/continue')
def dataset_train_continue(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    # base_model/variant = base sélectionnée (absente → base persistée du run).
    kw = {'extra_steps': d.get('extra_steps', 1000)}
    if 'base_model' in d:
        kw['base_model'] = d.get('base_model')
    if d.get('variant'):
        kw['variant'] = d.get('variant')
    try:
        res = lt.continue_training(LOCAL_USER, dataset_id, **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.get('/dataset/train/status')
def dataset_train_status():
    # Le poll doit toujours répondre 200 (jamais d'erreur) : sans ai-toolkit
    # configuré, on renvoie juste 'indisponible' au lieu d'un 409 qui casserait
    # le polling UI.
    if not capabilities.probe()['aitoolkit']['valid']:
        return jsonify({'available': False})
    # Le poll fait avancer la file : fin du training courant → lancement du suivant.
    try:
        lt.process_training_queue()
    except Exception:
        pass
    return jsonify(lt.training_status(LOCAL_USER))


@bp.post('/dataset/<int:dataset_id>/train/enqueue')
def dataset_train_enqueue(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    # base_model/variant = base CHOISIE pour le job en file (absente → persistée).
    kw = {'extra_steps': d.get('extra_steps'), 'masked': d.get('masked', True)}
    if 'base_model' in d:
        kw['base_model'] = d.get('base_model')
    if d.get('variant'):
        kw['variant'] = d.get('variant')
    if d.get('train_type'):
        kw['train_type'] = d.get('train_type')
    if d.get('allow_caption_mismatch'):
        kw['allow_caption_mismatch'] = True
    if d.get('allow_uncaptioned'):
        kw['allow_uncaptioned'] = True
    if d.get('allow_unverified_weights'):
        kw['allow_unverified_weights'] = True
    # SDXL custom overrides (service refuses them 400 for any other family).
    if 'vae_path' in d:
        kw['vae_path'] = d.get('vae_path')
    if 'te_path' in d:
        kw['te_path'] = d.get('te_path')
    # steps = cible absolue choisie côté UI (None → adaptatif). Forwarding conditionnel.
    if d.get('steps') is not None:
        kw['steps'] = d.get('steps')
    try:
        res = lt.enqueue_training(LOCAL_USER, dataset_id, **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/<int:dataset_id>/train/schedule')
def dataset_train_schedule(dataset_id):
    """Programme un entraînement (jour + heure locale). Contrairement à SRC, une
    échéance déjà PASSÉE est refusée (400) plutôt que dégradée en « dû
    immédiatement » : un `at` dans le passé est presque toujours une saisie
    erronée côté UI, pas une intention de lancer tout de suite."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    raw = str(d.get('at') or '').strip()   # datetime-local: "YYYY-MM-DDTHH:MM"
    try:
        at = datetime.fromisoformat(raw)
        # Normalize tz-aware to local naive for comparison
        if at.tzinfo is not None:
            at = at.astimezone().replace(tzinfo=None)
        if at <= datetime.now():
            return jsonify({'error': 'scheduled time is in the past'}), 400
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid schedule time'}), 400
    kw = {'extra_steps': d.get('extra_steps'), 'not_before': at.isoformat(timespec='minutes'),
          'masked': d.get('masked', True)}
    if 'base_model' in d:
        kw['base_model'] = d.get('base_model')
    if d.get('variant'):
        kw['variant'] = d.get('variant')
    if d.get('train_type'):
        kw['train_type'] = d.get('train_type')
    if d.get('allow_caption_mismatch'):
        kw['allow_caption_mismatch'] = True
    if d.get('allow_uncaptioned'):
        kw['allow_uncaptioned'] = True
    if d.get('allow_unverified_weights'):
        kw['allow_unverified_weights'] = True
    if 'vae_path' in d:
        kw['vae_path'] = d.get('vae_path')
    if 'te_path' in d:
        kw['te_path'] = d.get('te_path')
    if d.get('steps') is not None:
        kw['steps'] = d.get('steps')
    try:
        res = lt.enqueue_training(LOCAL_USER, dataset_id, **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/<int:dataset_id>/train/dequeue')
def dataset_train_dequeue(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    # Ownership : on ne retire de la file que SES propres datasets (anti-IDOR).
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    n = lt.dequeue_training(dataset_id)
    return jsonify({'ok': True, 'removed': n})


@bp.post('/dataset/train/stop')
def dataset_train_stop():
    # Single-user app : pas de vérif d'ownership sur l'entraînement en cours.
    gate = _require_aitoolkit()
    if gate:
        return gate
    lt.stop_training()
    return jsonify({'ok': True})


@bp.get('/dataset/<int:dataset_id>/train/checkpoints')
def dataset_train_checkpoints(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    # base_model = base sélectionnée dans le dropdown (param absent → base persistée).
    bm = request.args.get('base_model')
    # train_type = famille sélectionnée dans le menu LORA TYPE (param absent →
    # famille persistée).
    fam = request.args.get('train_type') or None
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    from ..models import CloudTrainingRun
    from ..services import checkpoint_registry
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    fam_resolved = lt._train_type(ds, fam)
    # Retrofit for datasets trained BEFORE the provenance registry existed:
    # training evidence without records -> record the current state as the v1
    # baseline, so versioning covers the past, not only future runs. Runs
    # BEFORE list_checkpoints so the fresh baseline annotates this response.
    had_training = (bool(lt.list_checkpoints(LOCAL_USER, dataset_id, **kw))
                    or any((ct._run_family(r) or fam_resolved) == fam_resolved
                           for r in CloudTrainingRun.query
                           .filter_by(dataset_id=dataset_id).all()))
    checkpoint_registry.ensure_baseline(LOCAL_USER, dataset_id, fam_resolved,
                                        had_training)
    return jsonify({'checkpoints': lt.list_checkpoints(LOCAL_USER, dataset_id, **kw),
                    # cloud saves synced locally (incl. an ACTIVE run's latest)
                    # — separate field: the resume-or-fresh prompt reasons on
                    # LOCAL checkpoints only
                    'cloud_checkpoints': ct.cloud_checkpoints(dataset_id, fam_resolved),
                    'recommended_steps': lt.recommended_steps(dataset_id),
                    'recommended_steps_info': lt.recommended_steps_info(dataset_id),
                    'imported': lt.list_imported_checkpoints(LOCAL_USER, dataset_id, family=fam),
                    'disk_usage': lt.dataset_disk_usage(LOCAL_USER, dataset_id, **kw),
                    # provenance: latest registered dataset version vs the
                    # dataset's CURRENT state (drift warning in the panel)
                    'dataset_state': checkpoint_registry.dataset_state(
                        LOCAL_USER, dataset_id, fam_resolved)})


@bp.get('/dataset/<int:dataset_id>/train/progress')
def dataset_train_progress(dataset_id):
    """Live run view for the TrainingPanel: parsed log progress (step/total/loss/
    speed/eta + downsampled loss curve) and the sample previews ai-toolkit writes.
    Answers 200 with log_exists=false before the log shows up — pollable early."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    bm = request.args.get('base_model')
    fam = request.args.get('train_type') or None
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    try:
        return jsonify(lt.training_progress(LOCAL_USER, dataset_id, **kw))
    except Exception as e:
        return _map_error(e)


_SAMPLE_NAME_RE = re.compile(r'^[\w.-]+\.(?:jpg|jpeg|png|webp)$', re.IGNORECASE)


@bp.get('/dataset/<int:dataset_id>/train/sample/<filename>')
def dataset_train_sample(dataset_id, filename):
    """Serve one training sample image. Filename is whitelist-validated (no
    separators/traversal) and resolved strictly inside the run's samples dir."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    if not _SAMPLE_NAME_RE.match(filename) or filename != os.path.basename(filename):
        return jsonify({'error': 'invalid filename'}), 400
    bm = request.args.get('base_model')
    fam = request.args.get('train_type') or None
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    try:
        d = lt._samples_dir(LOCAL_USER, dataset_id, **kw)
    except Exception as e:
        return _map_error(e)
    path = os.path.join(d, filename)
    if not os.path.isfile(path):
        return jsonify({'error': 'not found'}), 404
    from flask import send_file
    return send_file(path, conditional=True)


@bp.get('/dataset/<int:dataset_id>/train/preflight')
def dataset_train_preflight(dataset_id):
    """Pre-launch sanity report (blockers + warnings): image floor per family,
    composition balance, caption quality, identity leaks, near-duplicates,
    untriaged images, VRAM. The TrainingPanel calls it before Train/Queue/
    Schedule and turns warnings into ONE confirm."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        return jsonify({'ok': True, **lt.training_preflight(
            LOCAL_USER, dataset_id, train_type=request.args.get('train_type') or None)})
    except Exception as e:
        return _map_error(e)


@bp.post('/dataset/<int:dataset_id>/train/best-epoch')
def dataset_train_best_epoch(dataset_id):
    """Score every training sample vs the reference (face similarity, CPU) and
    recommend the checkpoint closest to the best-scoring step. Synchronous —
    one insightface subprocess for the whole set (~seconds to ~1 min)."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    bm = d.get('base_model')
    fam = d.get('train_type') or None
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    try:
        return jsonify({'ok': True, **lt.score_checkpoint_samples(LOCAL_USER, dataset_id, **kw)})
    except Exception as e:
        return _map_error(e)


@bp.get('/dataset/<int:dataset_id>/train/base-info')
def dataset_train_base_info(dataset_id):
    """Bases entraînables (officielle + merges Z-Image), base/variante choisies du
    dataset, et statut de conversion - pour le sélecteur du TrainingPanel."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    bases = [{'value': '', 'label': 'Official - Z-Image-Turbo (recommended)'}]
    converted = {}
    for m in get_zimage_models():
        bases.append({'value': m, 'label': m.replace('\\', '/').split('/')[-1].rsplit('.', 1)[0]})
        converted[m] = zc.is_converted(m)
    # Bases SDXL = checkpoints ComfyUI existants (single-file, pas de conversion).
    # get_checkpoint_models() renvoie des DICTS {name, civitai_url, score} (pas des
    # strings comme get_zimage_models) → extraire 'name'.
    sdxl_bases = []
    for c in (get_checkpoint_models() or []):
        name = c['name'] if isinstance(c, dict) else c
        sdxl_bases.append({'value': name,
                           'label': name.replace('\\', '/').split('/')[-1].rsplit('.', 1)[0]})
    # Krea 2 : base officielle fixe (pas de checkpoint custom, pas de conversion) ; le
    # choix Raw/Turbo se fait via le sélecteur `variant`, pas ici → label neutre.
    krea_bases = [{'value': '', 'label': 'Official - Krea 2'}]
    # Flux : base officielle fixe (FLUX.1-dev, gated HF) — pas de checkpoint custom ni
    # de conversion. Entrée explicite pour que l'UI n'aille PAS retomber sur les bases
    # Z-Image (fallback `bases_by_type[type] || bases`) quand la famille est Flux.
    flux_bases = [{'value': '', 'label': 'Official - FLUX.1-dev'}]
    # FLUX.2 Klein : bases officielles fixes (gated HF) — le choix 4B/9B se fait via
    # le sélecteur `variant` (comme Raw/Turbo pour Krea), pas ici → label neutre.
    flux2klein_bases = [{'value': '', 'label': 'Official - FLUX.2 Klein'}]
    # Les listers de bases (get_checkpoint_models / get_zimage_models) résolvent le
    # dossier des modèles depuis comfyui.base_dir → vides tant qu'il n'est pas
    # configuré. On expose ce fait pour que l'UI dise « configure ComfyUI dans Setup »
    # au lieu d'un « No checkpoint found » aveugle (le vrai motif sur un clone neuf).
    models_dir = None
    try:
        models_dir = cfg.comfyui_dir('models')
    except Exception:
        models_dir = None
    comfyui_configured = bool(models_dir) and os.path.isdir(str(models_dir))
    return jsonify({'bases': bases, 'base': ds.train_base_model or '',
                    # « Custom weights… » (local-only) : chemin custom persisté +
                    # overrides SDXL (VAE/TE). Le sélecteur les ressème ; la
                    # whitelist par famille est ré-appliquée au lancement (400).
                    'vae_path': ds.train_vae_path or '',
                    'te_path': ds.train_te_path or '',
                    'custom_weights_families': list(lt.CUSTOM_WEIGHTS_FAMILIES),
                    'vae_te_families': list(lt.VAE_TE_OVERRIDE_FAMILIES),
                    # Défaut family-aware : Krea → Raw (reco officielle), FLUX.2 Klein
                    # → 4B, sinon Turbo. Déféré au service (_default_variant_for) pour
                    # que l'UI et le lancement (_krea_is_raw/_flux2klein_is_9b) s'accordent.
                    'variant': ds.train_variant or lt._default_variant_for(ds.train_type or 'zimage'),
                    'converted': converted,
                    'convert': zc.convert_status(),
                    'train_type': ds.train_type or 'zimage',
                    'comfyui_configured': comfyui_configured,
                    'models_dir': str(models_dir) if models_dir else '',
                    # Réglages avancés effectifs (persistés ∪ défauts family-aware) pour
                    # la famille courante : rank/alpha/resolution/save_every → le panneau
                    # « Advanced options » les affiche et laisse les éditer.
                    'train_settings': lt.effective_train_settings(ds),
                    'bases_by_type': {'zimage': bases, 'sdxl': sdxl_bases,
                                      'krea': krea_bases, 'flux': flux_bases,
                                      'flux2klein': flux2klein_bases}})


@bp.post('/dataset/<int:dataset_id>/train/settings')
def dataset_train_settings(dataset_id):
    """Persiste un patch de réglages avancés {rank?, resolution?, save_every?} sur le
    dataset (validé + borné côté service). Renvoie les réglages effectifs résultants."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        eff = lt.update_train_settings(LOCAL_USER, dataset_id, d)
    except ValueError as e:
        return _map_error(e)
    return jsonify({'ok': True, 'train_settings': eff})


# --- Training presets ---------------------------------------------------------
# Named, shareable snapshots of the advanced settings. Stored AS-IS (raw keys);
# validation happens at APPLY time through the per-key path, so a preset file
# from another app version degrades gracefully (unknown keys ignored, invalid
# values reported). No ai-toolkit gate: presets are pure configuration.

def _preset_payload(p):
    import json
    try:
        settings = json.loads(p.settings or '{}')
    except ValueError:
        settings = {}
    return {'id': p.id, 'name': p.name, 'train_type': p.train_type,
            'settings': settings}


@bp.get('/train/presets')
def train_presets_list():
    """Built-ins first (shipped with the app, read-only — the frontend applies
    them by sending their settings, and hides delete), then the user's own."""
    from ..models import TrainingPreset
    rows = TrainingPreset.query.order_by(TrainingPreset.name).all()
    return jsonify({'presets': [*lt.BUILTIN_TRAIN_PRESETS,
                                *(_preset_payload(p) for p in rows)]})


@bp.post('/train/presets')
def train_presets_save():
    """Create or overwrite (by name). Two sources: `dataset_id` snapshots that
    dataset's current explicit settings (the 💾 Save-current path); `settings`
    stores an explicit dict (the ⬆ import path)."""
    import json
    from ..extensions import db
    from ..models import TrainingPreset
    d = request.get_json(silent=True) or {}
    name = (d.get('name') or '').strip()[:80]
    if not name:
        return jsonify({'error': 'name required'}), 400
    train_type = (d.get('train_type') or '').strip() or None
    if d.get('dataset_id') is not None:
        ds = svc.get_dataset(LOCAL_USER, d['dataset_id'])
        if not ds:
            return jsonify({'error': 'dataset not found'}), 404
        settings = lt.snapshot_train_settings(LOCAL_USER, ds.id)
        train_type = train_type or (ds.train_type or 'zimage')
    else:
        settings = d.get('settings')
        if not isinstance(settings, dict):
            return jsonify({'error': "'settings' must be an object"}), 400
    row = TrainingPreset.query.filter_by(name=name).first()
    created = row is None
    if created:
        row = TrainingPreset(name=name)
        db.session.add(row)
    row.train_type = train_type or 'zimage'
    row.settings = json.dumps(settings)
    db.session.commit()
    return jsonify({'ok': True, 'created': created, **_preset_payload(row)})


@bp.delete('/train/presets/<int:preset_id>')
def train_presets_delete(preset_id):
    from ..extensions import db
    from ..models import TrainingPreset
    row = db.session.get(TrainingPreset, preset_id)
    if not row:
        return jsonify({'error': 'not found'}), 404
    db.session.delete(row)
    db.session.commit()
    return jsonify({'ok': True})


@bp.post('/dataset/<int:dataset_id>/train/presets/apply')
def dataset_train_preset_apply(dataset_id):
    """Replace the dataset's advanced settings with a preset's ({preset_id})
    or with a raw dict ({settings}). Returns the effective settings plus what
    was ignored (unknown keys) and rejected (invalid values) — never fatal on
    content, so old exports keep working as the app evolves."""
    import json
    from ..extensions import db
    from ..models import TrainingPreset
    d = request.get_json(silent=True) or {}
    if d.get('preset_id') is not None:
        row = db.session.get(TrainingPreset, int(d['preset_id']))
        if not row:
            return jsonify({'error': 'unknown preset'}), 404
        try:
            settings = json.loads(row.settings or '{}')
        except ValueError:
            settings = {}
    else:
        settings = d.get('settings')
        if not isinstance(settings, dict):
            return jsonify({'error': "'settings' must be an object"}), 400
    try:
        eff, ignored, rejected = lt.apply_train_settings_dict(LOCAL_USER, dataset_id, settings)
    except ValueError as e:
        return _map_error(e)
    return jsonify({'ok': True, 'train_settings': eff,
                    'ignored': ignored, 'rejected': rejected})


@bp.post('/dataset/<int:dataset_id>/train/prepare-base')
def dataset_train_prepare_base(dataset_id):
    """Convertit un merge ComfyUI en diffusers (thread d'arrière-plan) pour
    pouvoir entraîner dessus. Statut via /train/base-info (convert)."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    bm = (request.get_json(silent=True) or {}).get('base_model', '')
    if not bm:
        return jsonify({'error': 'base model required'}), 400
    # Whitelist stricte : seul un modèle Z-Image réellement listé est convertible
    # (anti path-traversal - l'entrée transporte un chemin jusqu'à un subprocess).
    if bm not in get_zimage_models():
        return jsonify({'error': 'unknown base model'}), 400
    if zc.is_converted(bm):
        return jsonify({'ok': True, 'status': 'done'})
    try:
        zc.start_convert_async(current_app._get_current_object(), bm)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'status': 'running'})


@bp.post('/dataset/<int:dataset_id>/train/open-folder')
def dataset_train_open_folder(dataset_id):
    """Ouvre un dossier dans l'explorateur du poste (app locale) : target 'loras'
    (import ComfyUI de la famille), 'run' (checkpoints du run) ou 'dataset'
    (images + captions .txt du dataset — pas de dépendance ai-toolkit).
    Chemins résolus serveur — le body ne transporte jamais de chemin."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    kw = {'target': d.get('target') or 'loras'}
    if d.get('train_type'):
        kw['family'] = d.get('train_type')
    if 'base_model' in d:
        kw['base_model'] = d.get('base_model')
    try:
        path = lt.open_training_folder(LOCAL_USER, dataset_id, **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'path': path})


@bp.post('/dataset/<int:dataset_id>/train/checkpoint/delete')
def dataset_train_checkpoint_delete(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    body = request.get_json(silent=True) or {}
    fn = body.get('filename', '')
    fam = body.get('train_type') or None
    try:
        removed = lt.delete_imported_checkpoint(LOCAL_USER, dataset_id, fn, family=fam)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'removed': removed})


@bp.post('/dataset/<int:dataset_id>/train/run-checkpoint/delete')
def dataset_train_run_checkpoint_delete(dataset_id):
    """Move ONE RUN checkpoint to the trash — run-dir file, or a cloud run's
    synced save when cloud_run_id is given (the deployed-LoRA delete above is
    a separate route). Nothing is destroyed until 'Empty trash' in Settings."""
    gate = _require_aitoolkit()
    if gate and not capabilities.probe().get('cloud_training'):
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    body = request.get_json(silent=True) or {}
    try:
        if body.get('cloud_run_id'):
            removed = ct.delete_cloud_checkpoint(dataset_id, body['cloud_run_id'],
                                                 body.get('filename', ''))
        else:
            kw = {} if 'base_model' not in body else {'base_model': body.get('base_model')}
            if body.get('train_type'):
                kw['family'] = body.get('train_type')
            removed = lt.delete_checkpoint(LOCAL_USER, dataset_id,
                                           body.get('filename', ''), **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'removed': removed})


@bp.post('/dataset/<int:dataset_id>/train/checkpoints/cleanup')
def dataset_train_checkpoints_cleanup(dataset_id):
    """'Clean up this run': trash every run-dir checkpoint NOT in keep_filenames
    (typically final + best-epoch)."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    body = request.get_json(silent=True) or {}
    kw = {} if 'base_model' not in body else {'base_model': body.get('base_model')}
    if body.get('train_type'):
        kw['family'] = body.get('train_type')
    try:
        res = lt.cleanup_checkpoints(LOCAL_USER, dataset_id,
                                     body.get('keep_filenames') or [], **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/train/cloud/purge')
def dataset_train_cloud_purge():
    """Trash the staging dirs of finished cloud runs (dataset copies, samples,
    checkpoint duplicates already imported)."""
    return jsonify({'ok': True, **ct.purge_finished_runs()})


@bp.post('/dataset/<int:dataset_id>/train/import')
def dataset_train_import(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    body = request.get_json(silent=True) or {}
    fn = body.get('filename', '')
    # base_model = base du run d'où vient le checkpoint (absente → base persistée) ;
    # train_type = famille sélectionnée (absente → persistée) → même run + même dossier.
    kw = {} if 'base_model' not in body else {'base_model': body.get('base_model')}
    fam = body.get('train_type') or None
    if fam:
        kw['family'] = fam
    # cloud_run_id: import a CLOUD checkpoint (synced into the run's staging —
    # possibly mid-run) instead of a local ai-toolkit file. The run must belong
    # to this dataset; its dataset version rides along into the deployed name.
    if body.get('cloud_run_id'):
        from ..models import CloudTrainingRun
        crun = CloudTrainingRun.query.get(int(body['cloud_run_id']))
        if not crun or crun.dataset_id != dataset_id or not crun.staging_dir:
            return jsonify({'error': 'unknown cloud run'}), 404
        kw['src_dir'] = crun.staging_dir
        kw['version'] = ct._run_param(crun, 'version')
        kw.pop('base_model', None)          # cloud trains on the official base
    try:
        dest = lt.import_checkpoint(LOCAL_USER, dataset_id, fn, **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'dest': os.path.basename(dest)})


@bp.post('/dataset/<int:dataset_id>/train/cloud')
def dataset_train_cloud(dataset_id):
    gate = _require_cloud()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = ct.launch_cloud_training(
            LOCAL_USER, dataset_id,
            # No hardcoded 'turbo' default: an absent variant now resolves to
            # the family-aware default in the service (Krea → Raw, like local).
            steps=d.get('steps'),
            variant=d.get('variant'),
            train_type=d.get('train_type'),
            masked=d.get('masked', True),
            allow_caption_mismatch=bool(d.get('allow_caption_mismatch')),
            allow_uncaptioned=bool(d.get('allow_uncaptioned')),
            gpu_name=d.get('gpu_name'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/train/cloud/retry')
def dataset_train_cloud_retry():
    """↻ Retry d'un run en erreur (page Cloud) : relance avec les paramètres
    exacts du run raté — pod frais, mêmes garde-fous que tout launch."""
    gate = _require_cloud()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = ct.retry_cloud_run(LOCAL_USER, int(d.get('run_id') or 0))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/train/cloud/continue')
def dataset_train_cloud_continue():
    """▶ Continue d'un run cloud TERMINÉ (page Runs) : reprend depuis son dernier
    checkpoint harvesté et vise dernier_step + extra_steps — pod frais, mêmes
    garde-fous que tout launch ; le monitor dépose le checkpoint sur le pod avant
    de démarrer (auto-resume ai-toolkit)."""
    gate = _require_cloud()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = ct.continue_cloud_run(LOCAL_USER, int(d.get('run_id') or 0),
                                    extra_steps=d.get('extra_steps', 1000))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.get('/dataset/<int:dataset_id>/train/cloud/offers')
def dataset_train_cloud_offers(dataset_id):
    """Live GPU speed tiers for the launch dialog (price/h + approx time+cost).
    Read-only — rents nothing; the launch call rents the chosen class."""
    gate = _require_cloud()
    if gate:
        return gate
    try:
        data = ct.gpu_tiers(LOCAL_USER, dataset_id,
                            train_type=request.args.get('train_type'),
                            steps=request.args.get('steps', type=int))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **data})


@bp.get('/dataset/train/cloud/status')
def dataset_train_cloud_status():
    return jsonify(ct.cloud_status())


@bp.get('/dataset/train/cloud/runs')
def dataset_train_cloud_runs():
    """Active + recent cloud runs for the dedicated Cloud-runs hub page.
    Open like status (no gate): an unconfigured backend just returns empties."""
    return jsonify(ct.all_runs(limit=request.args.get('limit', default=20, type=int)))


@bp.get('/dataset/train/runs/<run_key>/share')
def dataset_train_run_share(run_key):
    """⎘ Share configuration: a paste-safe .txt of EVERYTHING this launch sent
    to ai-toolkit (family/variant/base + the full settings snapshot) plus the
    run's outcome — for sharing a recipe or asking for help on Discord/GitHub.
    `run_key` is 'cloud-<id>' (any cloud run) or 'rec-<id>' (a local run).
    Open like the other Runs-hub reads (no gate): unknown key -> 404."""
    from flask import Response
    from ..services import run_share
    out = run_share.build_run_config_text(run_key)
    if out is None:
        return jsonify({'error': 'unknown run'}), 404
    return Response(
        out['text'], mimetype='text/plain; charset=utf-8',
        headers={'Content-Disposition': f'attachment; filename="{out["filename"]}"'})


@bp.get('/dataset/<int:dataset_id>/train/cloud/progress')
def dataset_train_cloud_progress(dataset_id):
    try:
        return jsonify(ct.cloud_progress(LOCAL_USER, dataset_id,
                                         train_type=request.args.get('train_type')))
    except Exception as e:
        return _map_error(e)


@bp.post('/dataset/train/cloud/stop')
def dataset_train_cloud_stop():
    d = request.get_json(silent=True) or {}
    return jsonify({'ok': ct.request_stop(d.get('run_id'))})


@bp.get('/dataset/<int:dataset_id>/train/cloud/sample/<path:filename>')
def dataset_train_cloud_sample(dataset_id, filename):
    from flask import send_from_directory, abort
    # ?train_type= resolves THAT family's newest run (several families may
    # train the same dataset in parallel); absent -> plain newest, unchanged.
    run = ct.latest_run_for(dataset_id, request.args.get('train_type'))
    if not run or not run.staging_dir:
        abort(404)
    # send_from_directory refuses path traversal by construction
    return send_from_directory(os.path.join(run.staging_dir, 'samples'), filename)


@bp.get('/dataset/<int:dataset_id>/train/cloud/checkpoint')
def dataset_train_cloud_checkpoint(dataset_id):
    from flask import send_file, abort
    # ?run_id targets THAT run's file: with several finished runs of a family
    # in the hub history, 'newest run of the family' would serve the WRONG
    # checkpoint from an older row's button.
    rid = request.args.get('run_id', type=int)
    if rid is not None:
        from ..models import CloudTrainingRun
        run = CloudTrainingRun.query.get(rid)
        if run and run.dataset_id != dataset_id:
            run = None
    else:
        run = ct.latest_run_for(dataset_id, request.args.get('train_type'))
    if not run or not run.checkpoint_local_path \
            or not os.path.isfile(run.checkpoint_local_path):
        abort(404)
    return send_file(run.checkpoint_local_path, as_attachment=True)
