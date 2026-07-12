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
        res = lt.launch_training(LOCAL_USER, dataset_id, steps=d.get('steps'),
                                 base_model=d.get('base_model'),
                                 variant=d.get('variant', 'turbo'),
                                 train_type=d.get('train_type'),
                                 allow_caption_mismatch=bool(d.get('allow_caption_mismatch')),
                                 masked=d.get('masked', True),
                                 # fresh=True : écarte le run existant (archivé, pas
                                 # détruit) → repart de zéro au lieu de l'auto-resume.
                                 fresh=bool(d.get('fresh')))
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
    return jsonify({'checkpoints': lt.list_checkpoints(LOCAL_USER, dataset_id, **kw),
                    'recommended_steps': lt.recommended_steps(dataset_id),
                    'recommended_steps_info': lt.recommended_steps_info(dataset_id),
                    'imported': lt.list_imported_checkpoints(LOCAL_USER, dataset_id, family=fam)})


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
                    # Défaut family-aware : Krea → Raw (reco officielle), sinon Turbo.
                    # Le back-end (_krea_is_raw) applique le même défaut, ils s'accordent.
                    'variant': ds.train_variant or ('base' if (ds.train_type or 'zimage') == 'krea' else 'turbo'),
                    'converted': converted,
                    'convert': zc.convert_status(),
                    'train_type': ds.train_type or 'zimage',
                    'comfyui_configured': comfyui_configured,
                    'models_dir': str(models_dir) if models_dir else '',
                    # Réglages avancés effectifs (persistés ∪ défauts family-aware) pour
                    # la famille courante : rank/alpha/resolution/save_every → le panneau
                    # « Advanced options » les affiche et laisse les éditer.
                    'train_settings': lt.effective_train_settings(ds),
                    'bases_by_type': {'zimage': bases, 'sdxl': sdxl_bases, 'krea': krea_bases}})


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
    """Ouvre le dossier des LoRA dans l'explorateur du poste (app locale) :
    target 'loras' (import ComfyUI de la famille) ou 'run' (checkpoints du run).
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
            steps=d.get('steps'),
            variant=d.get('variant', 'turbo'),
            train_type=d.get('train_type'),
            masked=d.get('masked', True),
            allow_caption_mismatch=bool(d.get('allow_caption_mismatch')),
            gpu_name=d.get('gpu_name'))
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
    run = ct.latest_run_for(dataset_id, request.args.get('train_type'))
    if not run or not run.checkpoint_local_path \
            or not os.path.isfile(run.checkpoint_local_path):
        abort(404)
    return send_file(run.checkpoint_local_path, as_attachment=True)
