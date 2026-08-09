"""Training API: launch/continue/queue/stop a LoRA training run via ai-toolkit,
plus checkpoint listing/import/delete and Z-Image base-conversion prep.

No login - single local user (`cfg.LOCAL_USER`). Every route except
`/dataset/train/status` is gated on `capabilities.probe()['aitoolkit']['valid']`
(409 with a UI hint): `/train/status` must stay pollable even when
ai-toolkit isn't configured, so it degrades to `{'available': False}` instead.
"""
import os
import re
import time
from datetime import datetime
from threading import Lock

from flask import Blueprint, current_app, request, jsonify

from .. import capabilities
from .. import config as cfg
from ..config import LOCAL_USER
from ..services import cloud_training as ct
from ..services import face_dataset_service as svc
from ..services import face_mask
from ..services import face_mask_preview as fmp
from ..models import FaceDatasetImage
from ..services import lora_training as lt
from ..services import zimage_convert as zc
from ..utils.comfyui import get_zimage_models, get_checkpoint_models, get_krea_models
from ._common import _map_error

bp = Blueprint('training', __name__, url_prefix='/api')


class _CloseCallbackFile:
    """Delegate a file while running one callback on its first real close.

    ``send_file`` uses a direct WSGI file wrapper.  Some servers close that
    wrapper without invoking ``Response.call_on_close``; tying the lease to the
    file itself keeps response concurrency slots from leaking in that path.
    """

    def __init__(self, wrapped, callback):
        self._wrapped = wrapped
        self._callback = callback
        self._closed = False
        self._close_lock = Lock()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._wrapped.close()
        finally:
            self._callback()


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


def _full_transformer_artifact_response(run):
    """409 redirect metadata for routes that otherwise treat a file as a LoRA.

    A dense checkpoint is a FULL model, wherever it lives — on this disk since
    the local delivery landed, in a private Hugging Face repository before that.
    Neither makes it an adapter, and a stray/staging ``.safetensors`` file is
    never sufficient proof that one can be deployed to ComfyUI as one.
    """
    if not run or not ct._is_full_transformer_run(run):
        return None
    local_name = ct._run_param(run, 'local_weight_filename')
    local_dir = ct._run_param(run, 'local_artifact_dir')
    if local_name and local_dir:
        # It IS on this machine now — but re-serving 26 GB through the browser
        # would only write a second copy of a file the user already has. Say
        # where it is instead.
        error = (f'this full model is already on this computer: {local_name} in '
                 f'{local_dir}. It is a full checkpoint, not a LoRA adapter, so '
                 'it cannot be imported as one.')
    else:
        error = ('full_transformer artifacts are delivered through Hugging Face '
                 'and cannot be imported or downloaded as a LoRA checkpoint')
    return jsonify({
        'error': error,
        'local_weight_filename': local_name,
        'local_artifact_dir': local_dir,
        'training_mode': 'full_transformer',
        'artifact_kind': (ct._run_param(run, 'artifact_kind')
                          or 'full_transformer'),
        'artifact_status': ct._run_param(run, 'artifact_status'),
        'artifact_status_detail': ct._run_param(
            run, 'artifact_status_detail'),
        'hf_url': ct._run_param(run, 'hf_url'),
        'status': run.status,
    }), 409


@bp.post('/dataset/<int:dataset_id>/train')
def dataset_train(dataset_id):
    gate = _require_aitoolkit()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    has_training_mode = 'training_mode' in d
    try:
        mode = lt.training_mode(
            ds, d.get('training_mode') if has_training_mode else None)
    except Exception as e:
        return _map_error(e)
    if mode == 'full_transformer':
        return jsonify({
            'error': ('full_transformer training is cloud-only; choose Cloud training '
                      'or switch to LoRA'),
            'training_mode': mode,
        }), 400
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
        if has_training_mode:
            kw['training_mode'] = mode
        res = lt.launch_training(LOCAL_USER, dataset_id, steps=d.get('steps'),
                                 base_model=d.get('base_model'),
                                 variant=d.get('variant', 'turbo'),
                                 train_type=d.get('train_type'),
                                 allow_caption_mismatch=bool(d.get('allow_caption_mismatch')),
                                 allow_uncaptioned=bool(d.get('allow_uncaptioned')),
                                 allow_caption_quality=bool(d.get('allow_caption_quality')),
                                 allow_unverified_weights=bool(d.get('allow_unverified_weights')),
                                 # « Continue anyway » du panneau de préparation : lève le
                                 # garde-fou plancher d'images (jamais une impossibilité physique).
                                 allow_not_ready=bool(d.get('allow_not_ready')),
                                 # Absent = read the dataset's stored setting
                                 # (persisted; it used to be a browser-only value).
                                 masked=d.get('masked'),
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
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    # Every checkpoint in the local ai-toolkit lane is a LoRA.  Its source
    # artifact decides the continuation mode; today's dataset selector (or a
    # stale client body) cannot reinterpret those weights as a dense model.
    mode = 'lora'
    # base_model/variant = base sélectionnée (absente → base persistée du run).
    kw = {'extra_steps': d.get('extra_steps', 1000)}
    if 'base_model' in d:
        kw['base_model'] = d.get('base_model')
    if d.get('variant'):
        kw['variant'] = d.get('variant')
    if d.get('train_type'):
        kw['train_type'] = d.get('train_type')
    # from_step = reprise depuis un checkpoint précis (défaut = dernier). overrides =
    # réglages sûrs (le service refuse toute clé hors liste → 400).
    if d.get('from_step') is not None:
        kw['from_step'] = d.get('from_step')
    if d.get('overrides') is not None:
        kw['overrides'] = d.get('overrides')
    # Resume semantics are always explicit on the wire. Older clients safely
    # degrade to weights-only; a full-state request must carry the opaque bundle
    # id that the checkpoint listing advertised.
    kw['resume_mode'] = d.get('resume_mode', 'weights_only')
    if d.get('state_bundle_id') is not None:
        kw['state_bundle_id'] = d.get('state_bundle_id')
    kw['masked'] = d.get('masked')
    kw['allow_unverified_weights'] = bool(d.get('allow_unverified_weights'))
    kw['allow_caption_mismatch'] = bool(d.get('allow_caption_mismatch'))
    kw['allow_uncaptioned'] = bool(d.get('allow_uncaptioned'))
    kw['allow_caption_quality'] = bool(d.get('allow_caption_quality'))
    kw['allow_not_ready'] = bool(d.get('allow_not_ready'))
    kw['training_mode'] = mode
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
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    has_training_mode = 'training_mode' in d
    try:
        mode = lt.training_mode(
            ds, d.get('training_mode') if has_training_mode else None)
    except Exception as e:
        return _map_error(e)
    if mode == 'full_transformer':
        return jsonify({'error': 'full_transformer training is cloud-only and cannot be queued locally',
                        'training_mode': mode}), 400
    # base_model/variant = base CHOISIE pour le job en file (absente → persistée).
    kw = {'extra_steps': d.get('extra_steps'), 'masked': d.get('masked')}
    if has_training_mode:
        kw['training_mode'] = mode
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
    if d.get('allow_caption_quality'):
        kw['allow_caption_quality'] = True
    if d.get('allow_unverified_weights'):
        kw['allow_unverified_weights'] = True
    if d.get('allow_not_ready'):
        kw['allow_not_ready'] = True
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
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    has_training_mode = 'training_mode' in d
    try:
        mode = lt.training_mode(
            ds, d.get('training_mode') if has_training_mode else None)
    except Exception as e:
        return _map_error(e)
    if mode == 'full_transformer':
        return jsonify({'error': 'full_transformer training is cloud-only and cannot be scheduled locally',
                        'training_mode': mode}), 400
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
          'masked': d.get('masked')}
    if has_training_mode:
        kw['training_mode'] = mode
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
    if d.get('allow_caption_quality'):
        kw['allow_caption_quality'] = True
    if d.get('allow_unverified_weights'):
        kw['allow_unverified_weights'] = True
    if d.get('allow_not_ready'):
        kw['allow_not_ready'] = True
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
    d = request.get_json(silent=True) or {}
    expected_dataset_id = d.get('dataset_id')
    expected_run_token = d.get('run_token')
    if (expected_dataset_id is None) != (expected_run_token is None):
        return jsonify({
            'ok': False,
            'error': 'dataset_id and run_token must be provided together',
        }), 400
    if expected_dataset_id is not None:
        try:
            expected_dataset_id = int(expected_dataset_id)
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'error': 'invalid dataset id'}), 400
    if expected_run_token is not None:
        if not isinstance(expected_run_token, str) or not expected_run_token.strip():
            return jsonify({'ok': False, 'error': 'invalid run token'}), 400
        expected_run_token = expected_run_token.strip()
    # Preserve the original no-argument call for the dataset-manager button;
    # only the Runs hub opts into target-aware protection.
    stopped = (lt.stop_training()
               if expected_dataset_id is None and expected_run_token is None
               else lt.stop_training(
                   expected_dataset_id=expected_dataset_id,
                   expected_run_token=expected_run_token))
    if stopped is False:
        # The Runs hub polls every few seconds. Its card can therefore describe
        # run A just after A ended and queued run B started; never let that stale
        # button kill B. Older callers omit dataset_id and keep global semantics.
        return jsonify({
            'ok': False,
            'error': 'This local run is no longer active. The Runs page was refreshed.',
        }), 409
    return jsonify({'ok': True})


@bp.get('/dataset/<int:dataset_id>/train/checkpoints')
def dataset_train_checkpoints(dataset_id):
    """Every save this dataset owns, LOCAL and CLOUD, in one payload.

    NOT gated on ai-toolkit. This response carries both lanes, and a cloud-only
    install (vast.ai key, no ai-toolkit — `_require_cloud` lets it launch) has
    no ai-toolkit by definition: the historical 409 hid its OWN cloud
    checkpoints from it, silently, because `listCheckpoints` turns any non-200
    into an empty list. So the ai-toolkit capability now degrades the LOCAL
    half to empty instead of refusing the whole request — same reasoning as the
    lane-aware gate on `train/preflight` above. The cloud half, the imported
    list (a ComfyUI folder scan) and disk usage never needed ai-toolkit.
    """
    local_ok = capabilities.probe()['aitoolkit']['valid']
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    # base_model = base sélectionnée dans le dropdown (param absent → base persistée).
    bm = request.args.get('base_model')
    # train_type = famille sélectionnée dans le menu LORA TYPE (param absent →
    # famille persistée).
    fam = request.args.get('train_type') or None
    variant = request.args.get('variant') or None
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    if variant:
        kw['variant'] = variant
    from ..models import CloudTrainingRun
    from ..services import checkpoint_registry, dense_artifacts
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    fam_resolved = lt._train_type(ds, fam)
    # Retrofit for datasets trained BEFORE the provenance registry existed:
    # training evidence without records -> record the current state as the v1
    # baseline, so versioning covers the past, not only future runs. Runs
    # BEFORE list_checkpoints so the fresh baseline annotates this response.
    had_training = ((local_ok and lt.has_local_checkpoints(LOCAL_USER, dataset_id, **kw))
                    or any((ct._run_family(r) or fam_resolved) == fam_resolved
                           for r in CloudTrainingRun.query
                           .filter_by(dataset_id=dataset_id).all()))
    checkpoint_registry.ensure_baseline(LOCAL_USER, dataset_id, fam_resolved,
                                        had_training)
    # Deployment stamp (testable + deployed_filename) on EVERY listed save, from
    # the same join the ◉ Graph pills use — this is what lets the panel show
    # "✓ Deployed + ⏏ Undeploy" in place of a misleading second "Import →", and
    # aim the undeploy at the right ComfyUI file. Local rows name their own run,
    # so they are grouped by run; a cloud group IS one run.
    local_cks = ct.annotate_deployed_by_run(
        dataset_id, fam_resolved,
        lt.list_checkpoints(LOCAL_USER, dataset_id, **kw) if local_ok else [])
    cloud_groups = ct.cloud_checkpoint_groups(dataset_id, fam_resolved, variant=variant)
    for _g in cloud_groups:
        ct.annotate_deployed_checkpoints(dataset_id, fam_resolved,
                                         _g.get('checkpoints') or [],
                                         run_tag=('cloud', _g.get('run_id')))
    return jsonify({'checkpoints': local_cks,
                    # cloud saves synced locally (incl. an ACTIVE run's latest)
                    # — separate field: the resume-or-fresh prompt reasons on
                    # LOCAL checkpoints only
                    'cloud_checkpoints': ct.annotate_deployed_by_run(
                        dataset_id, fam_resolved,
                        [dict(c, run_source='cloud') for c in ct.cloud_checkpoints(
                            dataset_id, fam_resolved, variant=variant)]),
                    # same saves grouped BY SOURCE RUN (id/status/gpu/cost/time)
                    # so the panel labels which run produced which epochs and
                    # deep-links each group back to its Runs row
                    'cloud_checkpoint_groups': cloud_groups,
                    'recommended_steps': lt.recommended_steps(
                        dataset_id, train_type=fam_resolved, variant=variant),
                    'recommended_steps_info': lt.recommended_steps_info(
                        dataset_id, train_type=fam_resolved, variant=variant),
                    # FULL MODELS — a second lane in the same panel, deliberately
                    # not folded into the two above. Those deploy LoRA adapters
                    # into loras/<family>; a 26 GB transformer is not an adapter
                    # and must not inherit the adapter's verbs (see
                    # cloud_checkpoint_groups' guard, which stays). Additive: an
                    # older frontend simply ignores the key.
                    'dense_models': dense_artifacts.list_dense_models(
                        dataset_id, fam_resolved),
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
    variant = request.args.get('variant') or None
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    if variant:
        kw['variant'] = variant
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
    variant = request.args.get('variant') or None
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    if variant:
        kw['variant'] = variant
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
    Schedule and turns warnings into ONE confirm.

    `?lane=cloud` drops the rows that read THIS machine (GPU memory, torch build)
    — they describe hardware that will not run a cloud job. Absent or `local`
    returns the historical payload unchanged.

    `?masked=1|0` states whether the launch intends masked (person-mask) training,
    a client-side preference the server cannot read. Absent = not stated, and the
    person-mask readiness row is omitted."""
    # The gate follows the lane. A cloud-only install has no ai-toolkit, so the
    # historical _require_aitoolkit() would 409 exactly where these warnings matter
    # most (money is about to be spent) — and the caller treats a non-200 as "no
    # objection", which would have made the whole cloud preflight a silent no-op.
    lane = request.args.get('lane') or None
    raw_masked = request.args.get('masked')
    masked = None if raw_masked is None else raw_masked not in ('0', 'false', '')
    gate = _require_cloud() if lane == 'cloud' else _require_aitoolkit()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        preflight_kw = {}
        # Presence matters: ``?base_model=`` is the explicit official base,
        # whereas an absent parameter means "use the persisted selection".
        if 'base_model' in request.args:
            preflight_kw['base_model'] = request.args.get('base_model') or ''
        return jsonify({'ok': True, **lt.training_preflight(
            LOCAL_USER, dataset_id,
            train_type=request.args.get('train_type') or None,
            variant=request.args.get('variant') or None,
            lane=lane, masked=masked,
            training_mode=request.args.get('training_mode') or None,
            **preflight_kw)})
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
    if d.get('variant'):
        kw['variant'] = d.get('variant')
    try:
        return jsonify({'ok': True, **lt.score_checkpoint_samples(LOCAL_USER, dataset_id, **kw)})
    except Exception as e:
        return _map_error(e)


def _face_preview_kept(dataset_id):
    """The kept set as PLAIN data — {path: (image_id, filename)} — plus a
    fingerprint of it. Plain on purpose: the detection runs in a worker thread and
    ORM rows must not travel across the session that loaded them."""
    kept = (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='keep')
            .filter(FaceDatasetImage.filename.isnot(None)).all())
    by_path, stamps = {}, []
    for img in kept:
        p = os.path.join(svc._dataset_dir(dataset_id), img.filename)
        try:
            st = os.stat(p)
        except OSError:
            continue        # deleted under us — not part of what would be trained
        by_path[p] = (img.id, img.filename)
        stamps.append((img.id, img.filename, st.st_size, st.st_mtime_ns))
    return by_path, fmp.fingerprint(stamps)


def _face_preview_payload(results, by_path, limit):
    """Fold a detection result map into what the panel draws.

    The SAMPLE is deliberately failure-first: images where no face was found are
    the instructive ones (profile, cropped, too small) — a preview of successes
    only would hide exactly what the user needs to see. `coverage` is computed
    over the WHOLE kept set regardless, because a partially masked set is the bad
    case."""
    missed = [p for p, r in results.items() if (r or {}).get('state') != 'masked']
    found = [p for p, r in results.items() if (r or {}).get('state') == 'masked']
    samples = [{'image_id': by_path[p][0], 'filename': by_path[p][1],
                'state': (results[p] or {}).get('state'),
                'boxes': (results[p] or {}).get('boxes') or []}
               for p in (missed + found)[:limit] if p in by_path]
    return {'samples': samples, 'coverage': face_mask.coverage_summary(results),
            'expand': face_mask.expand_factor()}


# How often the running pass writes its detections down. Whichever comes first:
# ten images, or thirty seconds. Both bounds exist because either alone leaves a
# hole — ten images is several minutes on this CPU-only detector, and a
# time-only rule would rewrite the sidecar during a stretch where nothing new
# arrived. What they buy is the size of the loss when the process dies without
# being asked: at most ten images of re-detection, instead of the whole pass.
_FACE_BANK_EVERY = 10
_FACE_BANK_EVERY_S = 30


def _face_preview_guard(dataset_id, require_tool=True):
    """Shared gate. Returns an error response, or None."""
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    if not svc.is_concept(ds):
        return jsonify({'ok': False, 'error': 'face masking is for concept datasets'}), 400
    if require_tool and not face_mask.is_available():
        # Degrade by SAYING SO, INSTANTLY. The option itself is disabled in the UI
        # on the same capability, so this is the belt to that brace — and an install
        # without face detection must hear it now, not after a timeout.
        return jsonify({'ok': False, 'error': 'face detection unavailable',
                        'reason': 'face_scoring'}), 409
    return None


@bp.post('/dataset/<int:dataset_id>/train/face-mask-preview')
def dataset_face_mask_preview(dataset_id):
    """START (or JOIN) the face detection behind the training panel's mask preview,
    so it can draw what face masking would cover before anything is trained.

    Detection only — no generation, no training, nothing written to disk.

    Asynchronous on purpose. A blocking request could show nothing while it ran
    (the InsightFace model load alone is tens of seconds before image 1) and could
    not be rejoined: leaving the page threw the pass away and coming back offered
    to start a second one. The job lives in face_mask_preview, so a second click —
    or a return to the page — joins the pass in flight instead of duplicating it.

    RAW boxes are returned, not grown ones: the expand factor is applied in the
    browser (utils/faceMaskBox.js mirrors infer/face_mask_infer.dilate_box), so
    dragging the slider redraws instantly instead of paying for another InsightFace
    pass. One pass, then the knob is free."""
    gate = _face_preview_guard(dataset_id)
    if gate:
        return gate
    limit = max(1, min(12, int((request.get_json(silent=True) or {}).get('limit') or 6)))
    by_path, fp = _face_preview_kept(dataset_id)
    if not by_path:
        # Nothing kept is a valid answer, not a job. Publish it so a return to the
        # page shows the same thing rather than an inviting button.
        fmp.set_result(dataset_id, _face_preview_payload({}, {}, limit), fp)
        fmp.clear_partial(dataset_id)
        return jsonify({'ok': True, 'started': False,
                        **fmp.snapshot(dataset_id, fp, total=0)})

    paths = list(by_path)
    app = current_app._get_current_object()
    # What a previous, STOPPED pass already found on this exact kept set. Guarded
    # by the same fingerprint as the stored result, so a set that moved yields {}
    # and this is a full pass again.
    banked = fmp.partial(dataset_id, fp)
    todo = [p for p in paths if p not in banked]
    done_already = len(paths) - len(todo)

    def _work(job):
        if done_already:
            # The bar opens where the previous pass left off. Seeded HERE and not
            # after start() returns: under TESTING the job runs inline, so a seed
            # placed outside would land after the pass instead of before it —
            # and in production it would race the first real progress line.
            fmp.progress(job, {'done': done_already, 'total': len(paths)})
        if not todo:
            # Everything was already detected before the stop — there is nothing
            # left to run, only a result to publish. Paying an InsightFace load
            # to re-derive it would be the exact waste this path exists to avoid.
            fmp.set_result(dataset_id, _face_preview_payload(banked, by_path, limit), fp)
            fmp.clear_partial(dataset_id)
            return

        # Detections banked WHILE the pass runs, from the child's per-image
        # `item` lines. Before this, everything lived in the child's memory until
        # its final JSON line, so a pass that ended any way other than "finished"
        # or "stopped" — the watchdog budget elapsing, a crash, the server going
        # down — lost all of it and the next start began at image 1.
        live = {}
        flush = {'at': time.monotonic(), 'n': 0}

        def _bank():
            """Persist what has arrived so far, DEBOUNCED.

            The sidecar is rewritten whole (~60 KB for 138 entries), so one write
            per image would be 138 rewrites of the same file. One per ten bounds
            the loss to ten images; the time bound is there because ten images is
            minutes on this CPU-only detector and a crash inside that window must
            not cost them either.

            Every path that ENDS the pass banks `merged` itself, so this one only
            ever has to cover the case where nothing gets to run afterwards."""
            if not live:
                return
            if flush['n'] < _FACE_BANK_EVERY \
                    and time.monotonic() - flush['at'] < _FACE_BANK_EVERY_S:
                return
            flush['at'], flush['n'] = time.monotonic(), 0
            fmp.remember_partial(dataset_id, {**banked, **live}, fp)

        def _on_progress(rec):
            item = rec.get('item')
            if item:
                live[item['path']] = item['result']
                flush['n'] += 1
                _bank()
                return
            # The child counts ITS OWN images, which on a resume are only the
            # remaining ones. The panel must keep counting the whole kept set, or
            # a resumed pass would restart its bar at "image 1 of 106" and read
            # as though the stop had thrown everything away.
            rec = dict(rec)
            if rec.get('done') is not None:
                rec['done'] = rec['done'] + done_already
            if rec.get('total') is not None:
                rec['total'] = len(paths)
            fmp.progress(job, rec)

        data = face_mask.detect_faces(
            todo, on_progress=_on_progress,
            should_stop=lambda: fmp.stop_requested(job))
        # `live` first, the child's final answer last: the final line is the
        # authoritative one, `live` is what survives when there is no final line.
        merged = {**banked, **live, **(data.get('results') or {})}
        if data.get('timed_out'):
            # The budget elapsed. This is NOT a failure to throw away: the pass
            # was working, it simply needed longer than it was given, and the
            # images already analysed are exactly what makes the next attempt
            # cheap. Banking them is the difference between "start over" and
            # "carry on from 97 of 138".
            fmp.remember_partial(dataset_id, merged, fp)
            fmp.mark_interrupted(job, data.get('error')
                                 or 'face detection ran out of time')
            return
        if not data.get('ok'):
            # An operation that failed must LOOK failed. The reason travels all the
            # way to the panel instead of dying in a log line — but whatever the
            # pass had already found is still kept, so a crash on image 100 does
            # not make the retry re-detect the first 99.
            if merged:
                fmp.remember_partial(dataset_id, merged, fp)
            fmp.fail(job, data.get('error') or 'face detection failed')
            return
        if data.get('cancelled'):
            # Stopped. The boxes computed before the stop are the whole point of
            # asking the child to wind up rather than killing it, so they are
            # banked for the next start. The stored RESULT is deliberately left
            # alone: `coverage` is a safety figure over the WHOLE kept set, and
            # publishing "masked on 12 of 47" for a 153-image set would be a
            # wrong number rather than a partial one.
            fmp.remember_partial(dataset_id, merged, fp)
            fmp.mark_stopped(job)
            return
        # Zero faces is a RESULT, not a failure: a concept dataset may legitimately
        # hold no people. It publishes like any other pass.
        fmp.set_result(dataset_id, _face_preview_payload(merged, by_path, limit), fp)
        fmp.clear_partial(dataset_id)

    job, started = fmp.start(app, dataset_id, _work, total=len(paths), fp=fp)
    return jsonify({'ok': True, 'started': started,
                    **fmp.snapshot(dataset_id, fp, total=len(paths))}), 202


@bp.post('/dataset/<int:dataset_id>/train/face-mask-preview/stop')
def dataset_face_mask_preview_stop(dataset_id):
    """Ask the running detection to wind up, KEEPING what it already found.

    Not a kill. The child holds every box it detected in memory until it prints
    its final JSON line, so killing it would discard the pass — and since the
    detector load is a fixed price paid before image 1, a discarding Stop would
    make the retry cost the whole run again. It is asked instead, notices between
    two images, and hands back what it has; those detections are banked under the
    kept set's fingerprint and the next start resumes from them.

    The delay is honest and bounded by one image, EXCEPT during the model load
    (and the first-run download), which nothing can interrupt from outside: a
    stop asked there lands at the top of the loop, before image 1 — where there
    is nothing detected to lose anyway. The panel says which of the two costs
    applies at the moment it offers the button."""
    gate = _face_preview_guard(dataset_id, require_tool=False)
    if gate:
        return gate
    stopped = fmp.request_stop(dataset_id)
    by_path, fp = _face_preview_kept(dataset_id)
    return jsonify({'ok': True, 'stopping': stopped,
                    **fmp.snapshot(dataset_id, fp, total=len(by_path))})


@bp.get('/dataset/<int:dataset_id>/train/face-mask-preview')
def dataset_face_mask_preview_status(dataset_id):
    """Rebind: the running pass (phase + i/M) and the last preview computed for
    this dataset. Called on mount and while polling, so leaving the page and
    coming back picks the pass back up instead of restarting it.

    `result.stale` is the honest half: the stored preview describes the kept set
    it was computed from, and if that set moved since, showing it as fresh would
    be worse than showing nothing."""
    gate = _face_preview_guard(dataset_id, require_tool=False)
    if gate:
        return gate
    by_path, fp = _face_preview_kept(dataset_id)
    return jsonify({'ok': True, 'available': face_mask.is_available(),
                    **fmp.snapshot(dataset_id, fp, total=len(by_path))})


def _krea_installed_bases():
    """Every Krea 2 checkpoint on this machine, as TRAINING base entries.

    The Studio and the trainer address a model with two different — and both
    legitimate — vocabularies, so this is a translation, not a shortcut:

    * the Studio picker speaks ComfyUI-relative loader names (``Krea\\x.safetensors``)
      because it hands them to a ComfyUI running on this machine;
    * the trainer reads a base value through ``lt._is_custom_weights`` — an
      ABSOLUTE path means "custom weights, load this file", a RELATIVE name means
      "a catalog entry the app installed" (Z-Image merges, the SDXL whitelist,
      both of which have their own resolver), and ``''`` means the official base.
      Krea has no relative-name resolver: ``foreign_base_reason`` classifies a
      relative value on Krea as another family's base and the run silently falls
      back to the official weights. The trainer can also run on a REMOTE pod that
      must RECEIVE the file, which is why an absolute local path is the right
      currency there and a ComfyUI folder name is not.

    So: scan with the Studio's scanner (`get_krea_models` — no fifth scanner),
    then resolve each name to the concrete file through the same search roots
    ComfyUI itself uses. A name that resolves to nothing is DROPPED rather than
    emitted relative: offering a value the trainer would quietly ignore is worse
    than a shorter list.

    Each entry also carries what the picker must say about the file before it is
    chosen (`model_integrity.training_base_advisory`): a packed inference export
    is not trainable and says why, a bare fp8 cast is trainable and says what it
    costs. Header-only and cached per (path, mtime, size).
    """
    from ..services import comfy_model_paths
    from ..services import model_integrity
    out = []
    for rel in (get_krea_models() or []):
        path = comfy_model_paths.resolve_model_file('diffusion_models', rel)
        if not path:
            continue
        advisory = model_integrity.training_base_advisory(path)
        out.append({
            'value': path,
            'label': os.path.basename(str(rel).replace('\\', '/')).rsplit('.', 1)[0],
            'trainable': advisory['trainable'],
            'quantization': advisory['form'],
            'note': advisory['note'],
        })
    return out


@bp.get('/train/base-file-advisory')
def train_base_file_advisory():
    """The verdict a LISTED base already carries, for a path the user TYPED.

    « Custom weights… » is a free text field, and until now the only thing that
    ever read the file was the save and the launch. So the one base nobody can
    pick from a list — a checkpoint downloaded five minutes ago, the exact case
    the field exists for — was also the only one whose "this is a packed export,
    the trainer cannot load it" arrived after the dataset had been exported and,
    on the cloud lane, after a GPU had been rented. Same answer, same wording,
    same `model_integrity.training_base_advisory` — just at typing time.

    Read-only and header-only, and it never echoes the path back: the reply
    carries the BASENAME, because these payloads end up in pasted diagnostics.
    `status` is `ok` / `missing` / `not_a_model`; a file it cannot inspect is
    reported as such and never as a refusal — refusing a base nobody could read
    would be worse than the failure it prevents.
    """
    from ..services import model_integrity
    raw = (request.args.get('path') or '').strip().strip('"')
    if not raw:
        return jsonify({'status': 'missing', 'trainable': True, 'note': None}), 200
    name = os.path.basename(raw.replace('\\', '/'))
    if not name.lower().endswith(('.safetensors', '.sft')):
        # .gguf is loadable by ComfyUI and NOT by a trainer; anything else is not
        # a single-file checkpoint at all. Both are "not a base", said the same way.
        return jsonify({'status': 'not_a_model', 'filename': name,
                        'trainable': False, 'level': 'error', 'quantization': '',
                        'note': f'{name} is not a .safetensors checkpoint — training '
                                f'needs a single-file model in that format.'}), 200
    if not os.path.isfile(raw):
        return jsonify({'status': 'missing', 'filename': name, 'trainable': False,
                        'level': 'error', 'quantization': '',
                        'note': f'{name} is not at that path.'}), 200
    advisory = model_integrity.training_base_advisory(raw)
    return jsonify({'status': 'ok', 'filename': name,
                    'trainable': advisory['trainable'],
                    'level': advisory['level'],
                    'quantization': advisory['form'],
                    'note': advisory['note']}), 200


@bp.get('/dataset/<int:dataset_id>/train/base-info')
def dataset_train_base_info(dataset_id):
    """Bases entraînables (officielle + merges Z-Image), base/variante choisies du
    dataset, et statut de conversion - pour le sélecteur du TrainingPanel."""
    gate = _require_aitoolkit()
    if gate and not capabilities.probe().get('cloud_training'):
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
    # Krea 2 : la base officielle (le choix Raw/Turbo se fait via le sélecteur
    # `variant`, pas ici → label neutre) PUIS tout checkpoint Krea 2 installé sur
    # cette machine — un modèle que l'utilisateur vient d'entraîner, un build
    # communautaire. Même scanner que le Studio (get_krea_models), pas un
    # cinquième : leur divergence a déjà produit un bug.
    krea_bases = [{'value': '', 'label': 'Official - Krea 2'}] + _krea_installed_bases()
    # Flux : base officielle fixe (FLUX.1-dev, gated HF) — pas de checkpoint custom ni
    # de conversion. Entrée explicite pour que l'UI n'aille PAS retomber sur les bases
    # Z-Image (fallback `bases_by_type[type] || bases`) quand la famille est Flux.
    flux_bases = [{'value': '', 'label': 'Official - FLUX.1-dev'}]
    # FLUX.2 Klein : bases officielles fixes (gated HF) — le choix 4B/9B se fait via
    # le sélecteur `variant` (comme Raw/Turbo pour Krea), pas ici → label neutre.
    flux2klein_bases = [{'value': '', 'label': 'Official - FLUX.2 Klein'}]
    # Anima : une seule base officielle publique, pas de checkpoint custom. Sans
    # cette entrée le panneau retombait sur les bases Z-Image et annonçait
    # « Official - Z-Image-Turbo » sous la famille Anima — jusque dans la ligne de
    # résumé du bouton Train (le repli côté panneau est mort depuis, cf.
    # trainingFamilyScope.js ; l'entrée reste la source de vérité).
    anima_bases = [{'value': '', 'label': f'Official - {lt.ANIMA_BASE_LABEL}'}]
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
    # `base` must be what this run will ACTUALLY train on. train_base_model is a
    # single column shared by every family, so a dataset switched from Z-Image to
    # Krea 2 still carries the Z-Image merge — the builders already ignore it
    # (they gate on an ABSOLUTE path), so reporting it made the panel's summary
    # line, and the cloud dialog's "push this base", describe a run that was never
    # going to happen. Report the effective base ('') and say why, once, instead.
    _stored_base = ds.train_base_model or ''
    _base_mismatch = lt.foreign_base_message(ds.train_type or 'zimage', _stored_base)
    return jsonify({'bases': bases, 'base': '' if _base_mismatch else _stored_base,
                    # Present ONLY when the persisted base belongs to another
                    # family: the note the panel shows so the change isn't silent.
                    'base_family_mismatch': _base_mismatch,
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
                    # First-class provenance/launch selector. Old databases are
                    # migrated to lora and NULL test doubles resolve identically.
                    'training_mode': lt.training_mode(ds),
                    'comfyui_configured': comfyui_configured,
                    'models_dir': str(models_dir) if models_dir else '',
                    # Réglages avancés effectifs (persistés ∪ défauts family-aware) pour
                    # la famille courante : rank/alpha/resolution/save_every → le panneau
                    # « Advanced options » les affiche et laisse les éditer.
                    'train_settings': lt.effective_train_settings(ds),
                    # Slider LoRA mode (Beta) : état + prompts persistés + knobs résolus
                    # (colonne dédiée train_slider — jamais écrasé par un preset).
                    'slider': lt.effective_slider_settings(ds),
                    # Can this machine actually train Anima? The arch is an ai-toolkit
                    # EXTENSION, so an older checkout simply doesn't have it (the launch
                    # refuses with a 400). Exposed here rather than in /api/capabilities
                    # because the check walks the extensions tree: base-info is fetched
                    # when the training panel opens, capabilities is polled every 30s.
                    # The panel uses it to stay quiet instead of recommending Anima to
                    # someone who cannot run it.
                    'anima_supported': lt._aitoolkit_supports_anima(),
                    # Une entrée par famille de TRAIN_TYPES, sans exception : c'est
                    # ce que le panneau lit pour peupler son sélecteur de base
                    # (test_every_family_gets_its_own_base_list).
                    'bases_by_type': {'zimage': bases, 'sdxl': sdxl_bases,
                                      'krea': krea_bases, 'flux': flux_bases,
                                      'flux2klein': flux2klein_bases,
                                      'anima': anima_bases}})


@bp.post('/dataset/<int:dataset_id>/train/settings')
def dataset_train_settings(dataset_id):
    """Persiste un patch de réglages avancés {rank?, resolution?, save_every?} sur le
    dataset (validé + borné côté service). Renvoie les réglages effectifs résultants."""
    gate = _require_aitoolkit()
    if gate and not capabilities.probe().get('cloud_training'):
        return gate
    d = request.get_json(silent=True) or {}
    try:
        eff = lt.update_train_settings(LOCAL_USER, dataset_id, d)
    except ValueError as e:
        return _map_error(e)
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    return jsonify({'ok': True, 'train_settings': eff,
                    'training_mode': lt.training_mode(ds),
                    'train_type': ds.train_type,
                    'base_model': ds.train_base_model or '',
                    'variant': (ds.train_variant
                                or lt._default_variant_for(ds.train_type)),
                    # Canonical post-commit state.  The UI must only flip its
                    # Slider switch after this response confirms ``enabled=false``.
                    'slider': lt.effective_slider_settings(ds)})


@bp.post('/dataset/<int:dataset_id>/train/slider')
def dataset_train_slider(dataset_id):
    """Slider LoRA mode (Beta) : persiste un patch {enabled?, positive?, negative?,
    target_class?, anchor?, guidance?, anchor_strength?} (validé côté service,
    colonne dédiée train_slider). Renvoie l'état slider effectif."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        eff = lt.update_slider_settings(LOCAL_USER, dataset_id, d)
    except ValueError as e:
        return _map_error(e)
    return jsonify({'ok': True, 'slider': eff})


# --- Training presets ---------------------------------------------------------
# Named, shareable snapshots of the advanced settings. Stored AS-IS (raw keys);
# validation happens at APPLY time through the per-key path, so a preset file
# from another app version degrades gracefully (unknown keys ignored, invalid
# values reported). No ai-toolkit gate: presets are pure configuration.

_STYLE_SAMPLE_PROMPTS = [
    'a woman reading in a sunlit cafe',
    'a city street at night, rain, neon reflections',
    'a mountain landscape, wide shot, morning mist',
    'a still life of fruit on a wooden table',
    'a cozy interior, warm lamp light',
    'a runner mid-stride on a bridge, motion',
    'a cat sleeping on a windowsill',
    'a modern building facade, strong shadows',
]


def _style_preset_settings(rank, alpha, resolution='768,1024',
                           timestep_type=None):
    """One researched Style baseline with family-specific architecture levers.

    The prompts deliberately describe content only. Style mode is always-on, so
    the dataset's internal run identifier must never become a hidden activation
    token. Network dropout is intentionally absent: Style caption dropout is a
    separate, family-aware service policy and must not be conflated with it.
    """
    out = {
        'rank': rank,
        'alpha': alpha,
        'resolution': resolution,
        'save_every': 250,
        'max_step_saves': 10,
        'sample_every': 250,
        'sample_prompts': list(_STYLE_SAMPLE_PROMPTS),
    }
    if timestep_type:
        out['timestep_type'] = timestep_type
    return out


# Route-owned metadata wraps the validated setting dictionaries. Keeping scope
# outside train_settings means old exported presets remain schema-tolerant while
# new built-ins can be rejected before a single dataset field is mutated.
_STYLE_BUILTIN_PRESETS = [
    # 32/32 linear = the cleanest published Krea recipe (RunComfy "Train on
    # Raw, validate on Turbo": linear timesteps + flowmatch, rank/alpha 32) and
    # the vault's strongest confirmation ("Every Krea-2 source uses 32/32").
    {
        'id': 'builtin-style-krea-raw',
        'name': 'Krea 2 · Style (Raw)',
        'train_type': 'krea',
        'dataset_kind': 'style',
        # ``base`` is the canonical UI value; ``raw`` is accepted as a readable
        # compatibility spelling for callers that used the model's public name.
        'variants': ['base', 'raw'],
        'builtin': True,
        'description': "Krea's own style path — rank 32/32 with linear "
                       'timesteps, trained on Raw (validate on Turbo); '
                       'content-only probes every 250 steps.',
        'settings': _style_preset_settings(32, 32, timestep_type='linear'),
    },
    # weighted = Klein's canonical timestep (ai-toolkit options.ts); rank 32 =
    # the community's style step-up from the 16 default (RunComfy; the Herbst
    # 50-run study points even higher, 128/64, above this app's rank ceiling).
    {
        'id': 'builtin-style-klein-base',
        'name': 'FLUX.2 Klein · Style',
        'train_type': 'flux2klein',
        'dataset_kind': 'style',
        'variants': ['4b', '9b'],
        'builtin': True,
        'description': "Rank 32/32 with Klein's canonical weighted timesteps; "
                       'BFL recommends the shorter step envelope for style — '
                       'content-only probes every 250 steps.',
        'settings': _style_preset_settings(32, 32, timestep_type='weighted'),
    },
    # 32/32 confirmed by the concept research ("32/32 worked for everything,
    # style included"); weighted = the zimage arch default in options.ts and
    # the community's non-character recommendation.
    {
        # id kept ('...-base') for saved references, but the recipe is the
        # Z-Image ARCH DEFAULT (weighted timesteps), not a Base-only choice — so
        # it now applies to every Z-Image variant (Turbo / Base / De-Turbo), like
        # the FLUX.1 and SDXL style presets. Previously gated to ['base'], which
        # left a Turbo Z-Image style dataset with no built-in style preset.
        'id': 'builtin-style-zimage-base',
        'name': 'Z-Image · Style',
        'train_type': 'zimage',
        'dataset_kind': 'style',
        'variants': [],
        'builtin': True,
        'description': 'Rank 32/32 with weighted timesteps (the Z-Image arch '
                       'default, all variants); content-only probes so no hidden '
                       'trigger leaks into an always-on style.',
        'settings': _style_preset_settings(32, 32, timestep_type='weighted'),
    },
    # Corrected from 16/16 (the canonical FLUX subject dimension): the concept
    # research groups FLUX with the 32/32 prose family for style ("start
    # 32/32"), matching the practitioner consensus that style wants more
    # capacity than subject training.
    {
        'id': 'builtin-style-flux1',
        'name': 'FLUX.1 dev · Style',
        'train_type': 'flux',
        'dataset_kind': 'style',
        # FLUX.1 has one training recipe; the historical dataset variant field
        # is irrelevant for it, hence no variant restriction.
        'variants': [],
        'builtin': True,
        'description': 'Style wants more capacity than the 16/16 subject '
                       'default: rank 32/32 (research consensus) with weighted '
                       'timesteps and 250-step probes.',
        'settings': _style_preset_settings(32, 32, timestep_type='weighted'),
    },
    # Corrected from 32/16: the concept research recommends FULL alpha for
    # style ("Alpha = dim, recommandé style") — half-strength stays a
    # character-recipe trick. SDXL is ddpm: no flow-match timestep weighting.
    {
        'id': 'builtin-style-sdxl',
        'name': 'SDXL · Style',
        'train_type': 'sdxl',
        'dataset_kind': 'style',
        'variants': [],
        'builtin': True,
        'description': 'Full-strength rank 32/32 at native 1024 — research '
                       'recommends alpha = rank for style (half-strength is a '
                       'character trick).',
        'settings': _style_preset_settings(32, 32, resolution='1024'),
    },
]


def _builtin_train_presets():
    """Public built-in catalogue, with the legacy generic Style entry hidden.

    Copy every object so route metadata never mutates lora_training's module
    constants (important for tests and for long-lived Flask workers).
    """
    out = []
    seen = set()
    for source in (*lt.BUILTIN_TRAIN_PRESETS, *_STYLE_BUILTIN_PRESETS):
        preset_id = source.get('id')
        if preset_id == 'builtin-style' or preset_id in seen:
            continue
        preset = dict(source)
        preset['settings'] = dict(source.get('settings') or {})
        # Every current entry carries explicit scope metadata; the setdefaults
        # only shield hypothetical future entries from missing keys.
        preset.setdefault('dataset_kind', None)
        preset.setdefault('variants', [])
        out.append(preset)
        seen.add(preset_id)
    return out


_STYLE_PRESET_ID_BY_FAMILY = {
    'krea': 'builtin-style-krea-raw',
    'flux2klein': 'builtin-style-klein-base',
    'zimage': 'builtin-style-zimage-base',
    'flux': 'builtin-style-flux1',
    'sdxl': 'builtin-style-sdxl',
}


def _builtin_preset_by_id(preset_id, train_type):
    """Resolve a public ID, including the pre-family ``builtin-style`` alias."""
    if preset_id == 'builtin-style':
        preset_id = _STYLE_PRESET_ID_BY_FAMILY.get(train_type)
    return next((p for p in _builtin_train_presets()
                 if p.get('id') == preset_id), None)


def _dataset_kind(ds):
    return (getattr(ds, 'kind', None) or 'character').strip().lower()


def _normalise_preset_kind(value):
    kind = str(value or '').strip().lower()
    return kind if kind in svc.DATASET_KINDS else None


def _normalise_preset_variants(values, train_type):
    """Validated, de-duplicated recipe variants; ``None`` means bad input."""
    if values is None:
        return []
    if not isinstance(values, list):
        return None
    family = svc.normalize_train_type(train_type)
    if family in ('flux', 'sdxl'):
        allowed = set()
    else:
        allowed = set(lt._valid_variants_for(family))
        if family == 'krea':
            allowed.add('raw')
    out = []
    for value in values:
        variant = str(value or '').strip().lower()
        if not variant or variant not in allowed:
            return None
        if variant not in out:
            out.append(variant)
    return out


def _preset_scope_error(preset, ds, variant, reason):
    return jsonify({
        'ok': False,
        'error': reason,
        'error_code': 'PRESET_SCOPE',
        'preset_scope': {
            'preset_id': preset.get('id'),
            'preset_train_type': preset.get('train_type'),
            'preset_dataset_kind': preset.get('dataset_kind'),
            'preset_variants': preset.get('variants') or [],
            'dataset_train_type': ds.train_type or 'zimage',
            'dataset_kind': _dataset_kind(ds),
            'variant': variant,
        },
    }), 409


def _validate_preset_scope(preset, ds, requested_train_type=None,
                           requested_variant=None):
    """Return ``None`` when compatible, otherwise a structured 409 response.

    Validation happens wholly before ``apply_train_settings_dict``. This makes a
    stale preset selection harmless even though applying a preset is a replace
    operation that clears the previous explicit settings first.
    """
    dataset_family = ds.train_type or 'zimage'
    if requested_train_type:
        selected_family = str(requested_train_type).strip().lower()
        if selected_family not in svc.TRAIN_TYPES:
            return _preset_scope_error(
                preset, ds, requested_variant,
                'The requested model family is not supported.')
        if selected_family != dataset_family:
            return _preset_scope_error(
                preset, ds, requested_variant,
                'The selected model family changed before this preset was applied.')
    preset_family = preset.get('train_type')
    if preset_family and preset_family != dataset_family:
        return _preset_scope_error(
            preset, ds, requested_variant,
            f'This preset is for {preset_family}, not {dataset_family}.')
    preset_kind = preset.get('dataset_kind')
    if preset_kind and preset_kind != _dataset_kind(ds):
        return _preset_scope_error(
            preset, ds, requested_variant,
            f'This preset is for {preset_kind} datasets, not {_dataset_kind(ds)}.')
    variant = str(
        requested_variant or getattr(ds, 'train_variant', None)
        or lt._default_variant_for(dataset_family)).strip().lower()
    allowed = [str(v).strip().lower() for v in (preset.get('variants') or [])]
    if allowed and variant not in allowed:
        return _preset_scope_error(
            preset, ds, variant,
            f'This preset requires variant {" or ".join(allowed)}, not {variant}.')
    return None


def _preset_payload(p):
    import json
    try:
        settings = json.loads(p.settings or '{}')
    except ValueError:
        settings = {}
    if not isinstance(settings, dict):
        settings = {}
    try:
        variants = json.loads(p.variants or '[]')
    except (TypeError, ValueError):
        variants = []
    if not isinstance(variants, list):
        variants = []
    return {'id': p.id, 'name': p.name, 'train_type': p.train_type,
            # Existing DB rows have NULL scope metadata and remain deliberately
            # usable; their family is still validated at apply time.
            'dataset_kind': _normalise_preset_kind(p.dataset_kind),
            'variants': variants,
            'settings': settings}


@bp.get('/train/presets')
def train_presets_list():
    """Built-ins first (shipped with the app and read-only), then user presets.

    The old generic ``builtin-style`` is intentionally absent: callers should
    select the family recipe. Its ID is still accepted by the apply endpoint.
    """
    from ..models import TrainingPreset
    rows = TrainingPreset.query.order_by(TrainingPreset.name).all()
    return jsonify({'presets': [*_builtin_train_presets(),
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
    requested_family = None
    if 'train_type' in d and d.get('train_type') is not None:
        requested_family = str(d.get('train_type')).strip().lower()
        if requested_family not in svc.TRAIN_TYPES:
            return jsonify({'error': 'invalid train_type'}), 400
    dataset_kind = _normalise_preset_kind(d.get('dataset_kind'))
    if d.get('dataset_kind') is not None and dataset_kind is None:
        return jsonify({'error': 'invalid dataset_kind'}), 400
    supplied_variants = None
    if 'variants' in d:
        supplied_variants = d.get('variants')
    elif 'variant' in d:
        supplied_variants = [d.get('variant')]
    if d.get('dataset_id') is not None:
        ds = svc.get_dataset(LOCAL_USER, d['dataset_id'])
        if not ds:
            return jsonify({'error': 'dataset not found'}), 404
        train_type = ds.train_type or 'zimage'
        if (requested_family is not None
                and requested_family != train_type):
            return jsonify({
                'ok': False,
                'error': 'The selected model family changed before this preset was saved.',
                'error_code': 'PRESET_SCOPE',
            }), 409
        actual_kind = _dataset_kind(ds)
        if dataset_kind is not None and dataset_kind != actual_kind:
            return jsonify({
                'ok': False,
                'error': 'The dataset kind changed before this preset was saved.',
                'error_code': 'PRESET_SCOPE',
            }), 409
        settings = lt.snapshot_train_settings(LOCAL_USER, ds.id)
        dataset_kind = actual_kind
        variants = _normalise_preset_variants(supplied_variants, train_type)
        if variants is None:
            return jsonify({'error': 'invalid variants for train_type'}), 400
        if supplied_variants is None:
            # A snapshot is a recipe for the currently selected architecture.
            # Single-recipe families do not need a synthetic variant scope.
            if train_type in ('flux', 'sdxl'):
                variants = []
            else:
                selected = (d.get('variant') or ds.train_variant
                            or lt._default_variant_for(train_type))
                variants = _normalise_preset_variants([selected], train_type)
                if variants is None:
                    return jsonify({'error': 'invalid variant for train_type'}), 400
    else:
        train_type = requested_family or 'zimage'
        variants = _normalise_preset_variants(supplied_variants, train_type)
        if variants is None:
            return jsonify({'error': 'invalid variants for train_type'}), 400
        settings = d.get('settings')
        if not isinstance(settings, dict):
            return jsonify({'error': "'settings' must be an object"}), 400
    row = TrainingPreset.query.filter_by(name=name).first()
    created = row is None
    if created:
        row = TrainingPreset(name=name)
        db.session.add(row)
    row.train_type = train_type
    row.dataset_kind = dataset_kind
    row.variants = json.dumps(variants) if variants else None
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
    was ignored (unknown keys) and rejected (invalid values). Built-ins and DB
    presets are scope-checked before mutation; raw settings keep the legacy,
    schema-tolerant import path."""
    import json
    from ..extensions import db
    from ..models import TrainingPreset
    d = request.get_json(silent=True) or {}
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    applied_id = None
    applied_scope = None
    if d.get('preset_id') is not None:
        raw_id = d.get('preset_id')
        applied_id = raw_id
        builtin = (_builtin_preset_by_id(raw_id, ds.train_type or 'zimage')
                   if isinstance(raw_id, str) else None)
        if builtin:
            scoped = _validate_preset_scope(
                builtin, ds, requested_train_type=d.get('train_type'),
                requested_variant=d.get('variant'))
            if scoped:
                return scoped
            settings = dict(builtin.get('settings') or {})
            applied_id = builtin['id']
            applied_scope = {
                'preset_id': applied_id,
                'train_type': builtin.get('train_type'),
                'dataset_kind': builtin.get('dataset_kind'),
                'variants': builtin.get('variants') or [],
                'selected_variant': (
                    d.get('variant') or ds.train_variant
                    or lt._default_variant_for(ds.train_type or 'zimage')),
            }
        else:
            try:
                numeric_id = int(raw_id)
            except (TypeError, ValueError):
                return jsonify({'error': 'unknown preset'}), 404
            row = db.session.get(TrainingPreset, numeric_id)
            if not row:
                return jsonify({'error': 'unknown preset'}), 404
            try:
                row_variants = json.loads(row.variants or '[]')
            except (TypeError, ValueError):
                row_variants = []
            if not isinstance(row_variants, list):
                row_variants = []
            # Legacy numeric presets have always carried a family. New rows also
            # carry kind/variant scope; NULL/[] retains historical compatibility.
            db_preset = {
                'id': row.id,
                'train_type': row.train_type,
                'dataset_kind': row.dataset_kind,
                'variants': row_variants,
            }
            scoped = _validate_preset_scope(
                db_preset, ds, requested_train_type=d.get('train_type'),
                requested_variant=d.get('variant'))
            if scoped:
                return scoped
            try:
                settings = json.loads(row.settings or '{}')
            except ValueError:
                settings = {}
            if not isinstance(settings, dict):
                settings = {}
            applied_id = row.id
            applied_scope = {
                'preset_id': applied_id,
                'train_type': row.train_type,
                'dataset_kind': row.dataset_kind,
                'variants': row_variants,
                'selected_variant': (
                    d.get('variant') or ds.train_variant
                    or lt._default_variant_for(ds.train_type or 'zimage')),
            }
    else:
        settings = d.get('settings')
        if not isinstance(settings, dict):
            return jsonify({'error': "'settings' must be an object"}), 400
    try:
        eff, ignored, rejected = lt.apply_train_settings_dict(
            LOCAL_USER, dataset_id, settings, preset_scope=applied_scope)
    except ValueError as e:
        return _map_error(e)
    return jsonify({'ok': True, 'preset_id': applied_id,
                    'train_type': ds.train_type or 'zimage',
                    'variant': (d.get('variant') or ds.train_variant
                                or lt._default_variant_for(ds.train_type or 'zimage')),
                    'train_settings': eff,
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
    if d.get('variant'):
        kw['variant'] = d.get('variant')
    try:
        path = lt.open_training_folder(LOCAL_USER, dataset_id, **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'path': path})


@bp.post('/dataset/<int:dataset_id>/train/checkpoint/delete')
def dataset_train_checkpoint_delete(dataset_id):
    # Deleting a DEPLOYED LoRA is a ComfyUI-folder operation that never touches
    # ai-toolkit, and `list_imported_checkpoints` deliberately lists cloud-trained
    # files too (they are auto-imported into the same folder). Gating this on the
    # local trainer left a cloud-only install able to SEE those files and unable
    # to remove them — same cloud escape its run-checkpoint sibling already has.
    gate = _require_aitoolkit()
    if gate and not capabilities.probe().get('cloud_training'):
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
            if body.get('variant'):
                kw['variant'] = body.get('variant')
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
    if body.get('variant'):
        kw['variant'] = body.get('variant')
    try:
        res = lt.cleanup_checkpoints(LOCAL_USER, dataset_id,
                                     body.get('keep_filenames') or [], **kw)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/train/cloud/purge')
def dataset_train_cloud_purge():
    """Trash the working files of finished cloud runs — the exported dataset
    copy, the sample images and the logs. Checkpoints are moved into the durable
    store instead and never trashed. Also reports orphan run folders found on
    disk, which the caller can then purge explicitly."""
    return jsonify({'ok': True, **ct.purge_finished_runs()})


@bp.get('/dataset/train/cloud/orphans')
def dataset_train_cloud_orphans():
    """Run folders on disk that no run row claims — the tens of GB the cleanup
    used to answer 'already clean' about. Walks the disk, so it is its own
    endpoint and never rides the hub's poll."""
    orphans = ct.orphan_staging_dirs()
    return jsonify({'ok': True, 'orphans': orphans,
                    'total_bytes': sum(o['size_bytes'] for o in orphans)})


@bp.post('/dataset/train/cloud/purge-orphans')
def dataset_train_cloud_purge_orphans():
    """Trash the named orphan run folders (all of them when `names` is absent).
    Loose checkpoints inside them are rescued into the store first."""
    body = request.get_json(silent=True) or {}
    names = body.get('names')
    if names is not None and not isinstance(names, list):
        return jsonify({'error': 'names must be a list'}), 400
    try:
        res = ct.purge_orphan_staging_dirs(names)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.get('/dataset/train/cloud/staging-sizes')
def dataset_train_cloud_staging_sizes():
    """How much disk each cloud run's staging dir still holds, so the Runs hub can
    show "8.2 GB on disk" on a card and name that weight in the per-run 🧹
    confirmation. DELIBERATELY its own endpoint (and not a field of the runs
    payload): sizing means walking thousands of files, which must not ride the
    hub's 5 s poll. Optional ?run_ids=1,2,3 narrows the walk to the shown cards."""
    raw = (request.args.get('run_ids') or '').strip()
    ids = None
    if raw:
        try:
            ids = [int(x) for x in raw.split(',') if x.strip()]
        except ValueError:
            return jsonify({'error': 'run_ids must be a comma-separated list of run ids'}), 400
    sizes = ct.staging_sizes(ids)
    return jsonify({'ok': True,
                    'sizes': {str(k): v for k, v in sizes.items()},
                    'total_bytes': sum(sizes.values())})


@bp.post('/dataset/train/cloud/purge-run')
def dataset_train_cloud_purge_run():
    """Trash the staging dir of ONE finished cloud run — targeted cleanup, so a
    45-run history no longer forces an all-or-nothing purge. Spares exactly what
    the global purge spares (active runs, kept pods) via the shared rule."""
    body = request.get_json(silent=True) or {}
    if body.get('run_id') in (None, ''):
        return jsonify({'error': 'run_id is required'}), 400
    try:
        res = ct.purge_run_staging(body['run_id'])
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


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
    if body.get('variant'):
        kw['variant'] = body.get('variant')
    # cloud_run_id: import a CLOUD checkpoint (synced into the run's staging —
    # possibly mid-run) instead of a local ai-toolkit file. The run must belong
    # to this dataset; its dataset version rides along into the deployed name.
    if body.get('cloud_run_id'):
        from ..models import CloudTrainingRun
        from ..services import cloud_run_dataset as crd
        crun = CloudTrainingRun.query.get(int(body['cloud_run_id']))
        # (id, table), not id alone: this route is reached from a FACE dataset,
        # and a video run of the same id would otherwise pass the check and get
        # deployed into this dataset's ComfyUI folder.
        if not crun or not crd.owns(crun, dataset_id):
            return jsonify({'error': 'unknown cloud run'}), 404
        dense_response = _full_transformer_artifact_response(crun)
        if dense_response:
            return dense_response
        # Where THIS save actually sits: the durable checkpoint store, or the
        # legacy staging dir on an install that has not been retrofitted yet.
        saves = ct.run_checkpoint_files(crun)
        src = saves.get(os.path.basename(fn or '')) or (
            sorted(saves.values())[0] if saves else None)
        if not src:
            return jsonify({'error': 'unknown cloud run'}), 404
        kw['src_dir'] = os.path.dirname(src)
        kw['version'] = ct._run_param(crun, 'version')
        # Tag the deployed name with THIS cloud run's id (☁ #N) so importing the
        # same step from two different runs never overwrites one with the other.
        kw['run_id'] = crun.id
        kw['run_source'] = 'cloud'
        # Never route a cloud file through the dataset row: replay the exact
        # family/variant/base stamped at cloud launch. Legacy rows fall back to
        # the request's family/variant, but still use the official empty base.
        kw['base_model'] = ct._run_param(crun, 'base_model') or ''
        cloud_fam = ct._run_param(crun, 'train_type')
        cloud_variant = ct._run_param(crun, 'variant')
        if cloud_fam:
            kw['family'] = cloud_fam
        if cloud_variant:
            kw['variant'] = cloud_variant
    try:
        res = lt.import_checkpoint(LOCAL_USER, dataset_id, fn, return_meta=True, **kw)
    except Exception as e:
        return _map_error(e)
    out = {'ok': True, 'dest': os.path.basename(res['dest'])}
    if res.get('collision'):
        # A different LoRA already used that name — the import was renamed rather
        # than silently overwriting it; tell the UI so it can surface the note.
        out['note'] = (f'A different LoRA already used that name — saved as '
                       f'{res["name"]} to avoid overwriting it.')
    return jsonify(out)


@bp.post('/dataset/<int:dataset_id>/train/cloud')
def dataset_train_cloud(dataset_id):
    gate = _require_cloud()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    try:
        mode = lt.training_mode(
            ds, d.get('training_mode') if 'training_mode' in d else None)
        res = ct.launch_cloud_training(
            LOCAL_USER, dataset_id,
            # No hardcoded 'turbo' default: an absent variant now resolves to
            # the family-aware default in the service (Krea → Raw, like local).
            steps=d.get('steps'),
            base_model=d.get('base_model', ''),
            variant=d.get('variant'),
            train_type=d.get('train_type'),
            training_mode=mode,
            masked=d.get('masked'),
            allow_caption_mismatch=bool(d.get('allow_caption_mismatch')),
            allow_uncaptioned=bool(d.get('allow_uncaptioned')),
            allow_caption_quality=bool(d.get('allow_caption_quality')),
            allow_unverified_weights=bool(d.get('allow_unverified_weights')),
            allow_not_ready=bool(d.get('allow_not_ready')),
            # « Train anyway » on the HF private-storage pre-check: the ceiling
            # it compares against is an estimate, so the user always keeps the
            # last word (see hf_storage).
            allow_hf_storage=bool(d.get('allow_hf_storage')),
            # ... and the same last word about THIS machine's disk, for a full
            # model that is delivered here.
            allow_local_disk=bool(d.get('allow_local_disk')),
            allow_parallel_run=bool(d.get('allow_parallel_run')),
            gpu_name=d.get('gpu_name'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.get('/dataset/<int:dataset_id>/train/cloud/custom-base')
def dataset_train_cloud_custom_base(dataset_id):
    """Readiness of a CUSTOM base for cloud training: is it already pushed to
    the private `lds-base-<hash>` repo on the user's Hugging Face account
    (cache-hit → launch straight away), or does it need the one-time push?
    Also reports the background push job's state (poll-friendly, never 500s
    on a missing repo — that is just ready=false)."""
    gate = _require_cloud()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    from ..services import hf_base_push
    try:
        state = hf_base_push.base_push_state(
            LOCAL_USER, dataset_id,
            request.args.get('train_type'), request.args.get('variant'),
            (request.args.get('base_model') or '').strip(),
            cfg.secret('HF_TOKEN'))
    except hf_base_push.HfPublishError as e:
        return jsonify({'error': e.message, 'error_code': e.code}), 400
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **state})


@bp.post('/dataset/<int:dataset_id>/train/cloud/custom-base/push')
def dataset_train_cloud_custom_base_push(dataset_id):
    """One-time background upload of the custom base to a PRIVATE repo on the
    user's Hugging Face account (private is forced server-side — no toggle).
    Multi-GB → daemon thread; the UI polls the custom-base route above. The
    confirmable CUSTOM_WEIGHTS_UNVERIFIED arch sniff answers synchronously so
    the dialog can confirm-and-retry."""
    gate = _require_cloud()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    token = cfg.secret('HF_TOKEN')
    if not token:
        return jsonify({'error': 'no Hugging Face token configured — paste an '
                        'HF_TOKEN in Settings ▸ API keys'}), 400
    d = request.get_json(silent=True) or {}
    from ..services import hf_base_push
    try:
        out = hf_base_push.start_push(
            current_app._get_current_object(), dataset_id,
            d.get('train_type'), d.get('variant'),
            (d.get('base_model') or '').strip(), token,
            allow_unverified_weights=bool(d.get('allow_unverified_weights')))
    except hf_base_push.HfPublishError as e:
        return jsonify({'error': e.message, 'error_code': e.code}), 400
    except ValueError as e:
        # preflight_custom_paths' confirmable marker (CUSTOM_WEIGHTS_UNVERIFIED)
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


# --- Hugging Face private storage (Settings ▸ Training) -----------------------
# Every route below talks to the Hub, so none of them runs on page load: the
# Settings card fetches on an explicit click. The namespace is resolved from the
# token, never taken from the client.

def _hf_storage_namespace():
    """(namespace, token) for the account whose private storage matters, or a
    (None, reason) pair. HF_TOKEN owns the lds-base-* caches; HF_CLOUD_TOKEN is
    scoped to the dense DELIVERY namespace and may not even be able to list
    them, so the general token is preferred and the dense one is the fallback."""
    from ..services.hf_publish import HfPublishError, _make_api
    for key in ('HF_TOKEN', 'HF_CLOUD_TOKEN'):
        token = cfg.secret(key)
        if not token:
            continue
        try:
            who = _make_api(token).whoami() or {}
        except HfPublishError:
            continue
        except Exception:
            continue
        name = str((who or {}).get('name') or '').strip()
        if name:
            return name, token, who
    return None, None, {}


# --- Local fp8 quantization ---------------------------------------------------
# Same conversion as the post-training pod export, started by hand on a model
# already on this machine. No ai-toolkit and no cloud gate: it is a pure
# file-in / file-out operation on the CPU (see fp8_quantize's module note).

@bp.post('/tools/fp8-quantize/plan')
def tools_fp8_quantize_plan():
    """What quantizing this file would produce, or WHY it is refused.

    Always 200: the panel disables its button with the reason instead of showing
    an error toast after a click.
    """
    from ..services import fp8_quantize
    d = request.get_json(silent=True) or {}
    return jsonify(fp8_quantize.describe(d.get('path')))


@bp.post('/tools/fp8-quantize')
def tools_fp8_quantize_start():
    from ..services import fp8_quantize
    d = request.get_json(silent=True) or {}
    try:
        info = fp8_quantize.start_async(current_app._get_current_object(),
                                        d.get('path'),
                                        overwrite=bool(d.get('overwrite')))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **info, 'status': fp8_quantize.status()})


@bp.get('/tools/fp8-quantize/status')
def tools_fp8_quantize_status():
    from ..services import fp8_quantize
    return jsonify({'ok': True, **(fp8_quantize.status() or {})})


# --- One-click fp8 delivery ------------------------------------------------------
# The normal path: no path to paste, no machine to rent. It resolves the master
# (a Hugging Face repository, or a full-precision file the app already knows
# about), downloads it if it is not here, quantizes it and leaves the fp8 in the
# ComfyUI folder that loads it. NOT behind _require_cloud: nothing is rented, and
# the person who most needs this has no vast key at all.

@bp.post('/tools/fp8-deliver/plan')
def tools_fp8_deliver_plan():
    """Where the file will land, what it will weigh, and whether the disk can
    take it — answered BEFORE the click. Always 200: a refusal is a disabled
    button carrying its reason, not a toast after the user committed."""
    from ..services import fp8_local_delivery
    d = request.get_json(silent=True) or {}
    return jsonify(fp8_local_delivery.describe(
        repo_id=d.get('repo_id'), filename=d.get('filename'), path=d.get('path'),
        family=d.get('family'), keep_master=d.get('keep_master', True),
        destination_dir=d.get('destination_dir')))


@bp.post('/tools/fp8-deliver')
def tools_fp8_deliver_start():
    from ..services import fp8_local_delivery
    d = request.get_json(silent=True) or {}
    try:
        info = fp8_local_delivery.start(
            current_app._get_current_object(),
            repo_id=d.get('repo_id'), filename=d.get('filename'),
            path=d.get('path'), family=d.get('family'),
            keep_master=d.get('keep_master', True),
            destination_dir=d.get('destination_dir'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **info, 'status': fp8_local_delivery.status()})


@bp.get('/tools/fp8-deliver/status')
def tools_fp8_deliver_status():
    from ..services import fp8_local_delivery
    return jsonify({'ok': True, **(fp8_local_delivery.status() or {})})


@bp.post('/tools/fp8-deliver/cancel')
def tools_fp8_deliver_cancel():
    """Stop the job. The bytes already downloaded stay on disk, so starting
    again resumes rather than restarts."""
    from ..services import fp8_local_delivery
    return jsonify({'ok': True, 'cancelled': fp8_local_delivery.cancel(),
                    **(fp8_local_delivery.status() or {})})


# --- full models: the fp8 twin's last mile, and the trash -------------------------
# The master is NEVER sent to ComfyUI. It is 26 GB of a model folder to do a job
# the ~13 GB twin does better, and it is the only file that can be trained again.
# That asymmetry is the whole point of this lane, so it lives in the routes too:
# there is no endpoint here that could deploy a master, not even by mistake.

@bp.post('/dataset/<int:dataset_id>/train/dense/send-plan')
def dataset_dense_send_plan(dataset_id):
    """What "Send to ComfyUI" would do — link or copy, where, at what cost.
    Always 200: a refusal is a disabled button carrying its reason."""
    from ..services import dense_artifacts
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    return jsonify(dense_artifacts.send_plan(dataset_id, d.get('run_id')))


@bp.post('/dataset/<int:dataset_id>/train/dense/send')
def dataset_dense_send(dataset_id):
    from ..services import dense_artifacts
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    try:
        info = dense_artifacts.send_to_comfyui(
            current_app._get_current_object(), dataset_id, d.get('run_id'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **info, 'job': dense_artifacts.status()})


@bp.get('/tools/dense-send/status')
def tools_dense_send_status():
    """Global, like the fp8 job's: one send at a time, and it outlives the tab."""
    from ..services import dense_artifacts
    return jsonify({'ok': True, **(dense_artifacts.status() or {})})


@bp.post('/dataset/<int:dataset_id>/train/dense/delete')
def dataset_dense_delete(dataset_id):
    """Move ONE of a full model's files to the app trash — recoverable on
    purpose: these cost hours of GPU, and a mis-click must not be final."""
    from ..services import dense_artifacts
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    try:
        out = dense_artifacts.delete_artifact(
            dataset_id, d.get('run_id'), d.get('filename'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **out})


@bp.post('/cloud/quantize/plan')
def cloud_quantize_plan():
    """Cost, duration cap and storage impact of quantizing a delivered artifact
    in the cloud — always answered BEFORE anything is rented."""
    gate = _require_cloud()
    if gate:
        return gate
    from ..services import cloud_quantize
    d = request.get_json(silent=True) or {}
    try:
        return jsonify({'ok': True, **cloud_quantize.plan(
            d.get('repo_id'), filename=d.get('filename'),
            keep_bf16=d.get('keep_bf16', True))})
    except Exception as e:
        return _map_error(e)


@bp.post('/cloud/quantize')
def cloud_quantize_start():
    gate = _require_cloud()
    if gate:
        return gate
    from ..services import cloud_quantize
    d = request.get_json(silent=True) or {}
    try:
        planned = cloud_quantize.start(
            current_app._get_current_object(), d.get('repo_id'),
            filename=d.get('filename'), keep_bf16=d.get('keep_bf16', True),
            # The price the user read on screen: the rental re-searches offers,
            # and must not silently land on a dearer machine than the one quoted.
            quoted_price=d.get('quoted_price_per_hour'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **planned, 'status': cloud_quantize.status()})


@bp.get('/cloud/quantize/status')
def cloud_quantize_status():
    """State of the cloud quantization, and a sweep for orphaned pods.

    The sweep runs HERE on purpose: this endpoint is what the UI polls, so a
    machine left behind by an app restart is reaped by the act of looking at it.
    """
    gate = _require_cloud()
    if gate:
        return gate
    from ..services import cloud_quantize
    reaped = cloud_quantize.reconcile_orphans()
    return jsonify({'ok': True, 'reaped': reaped, **(cloud_quantize.status() or {})})


@bp.get('/cloud/hf-storage')
def cloud_hf_storage():
    """Measured private storage + the lds-base-* cache inventory.

    Answers 200 with ok=false rather than an error status when the Hub cannot be
    measured: this card must be able to SAY it does not know."""
    gate = _require_cloud()
    if gate:
        return gate
    from ..services import hf_storage
    namespace, token, who = _hf_storage_namespace()
    if not namespace:
        return jsonify({'ok': False, 'reason': 'no_token',
                        'error': 'no Hugging Face token configured — add HF_TOKEN '
                                 'in Settings ▸ Local tools'})
    usage = hf_storage.private_storage_usage(namespace, token)
    inventory = hf_storage.base_cache_inventory(namespace, token, LOCAL_USER,
                                                _usage=usage)
    # No dataset in hand here — this card describes the account, not one run.
    # It therefore quotes the SHIPPED defaults (1 checkpoint + the fp8 export);
    # the per-run figure, which follows that dataset's own "keep" choice, is
    # shown in the training panel and enforced at launch.
    forecast = hf_storage.dense_storage_forecast(
        namespace, token, keeps=lt.dense_max_step_saves_for(None), who=who,
        _usage=usage, fp8_export=True)
    forecast.pop('usage', None)
    inventory.pop('usage', None)
    return jsonify({'ok': usage['ok'], 'namespace': namespace,
                    'reason': usage.get('reason'),
                    'partial': usage.get('partial', False),
                    'used_bytes': usage.get('used_bytes'),
                    'private_repo_count': usage.get('private_repo_count'),
                    'unsized_repo_count': usage.get('unsized_repo_count'),
                    'forecast': forecast, **inventory})


@bp.delete('/cloud/hf-storage/base/<repo_name>')
def cloud_hf_storage_delete_base(repo_name):
    """Delete ONE lds-base-* cache repo (the service validates the name)."""
    gate = _require_cloud()
    if gate:
        return gate
    from ..services import hf_storage
    from ..services.hf_publish import HfPublishError
    namespace, token, _who = _hf_storage_namespace()
    if not namespace:
        return jsonify({'error': 'no Hugging Face token configured'}), 400
    try:
        out = hf_storage.delete_base_cache(namespace, repo_name, token)
    except HfPublishError as e:
        return jsonify({'error': e.message, 'error_code': e.code}), 400
    except Exception as e:
        return _map_error(e)
    return jsonify(out)


@bp.post('/cloud/hf-storage/base/delete-all')
def cloud_hf_storage_delete_all_bases():
    """Delete every lds-base-* cache of the account. Partial failures reported."""
    gate = _require_cloud()
    if gate:
        return gate
    from ..services import hf_storage
    namespace, token, _who = _hf_storage_namespace()
    if not namespace:
        return jsonify({'error': 'no Hugging Face token configured'}), 400
    try:
        out = hf_storage.delete_all_base_caches(namespace, token, LOCAL_USER)
    except Exception as e:
        return _map_error(e)
    return jsonify(out)


@bp.post('/dataset/train/retry')
def dataset_train_retry():
    """↻ Retry a FAILED LOCAL run (Runs page): relaunch training with the exact
    identity params stamped for that launch. A real launch_training — normal
    preflight, GPU-collision refusal, no bypass — replaying the live dataset
    (slider settings included), not a resurrection of the dead process.

    The confirmable refusals are answered HERE, in the payload, exactly like the
    Start handlers above — not inherited from the failed launch. Retry re-exports
    the LIVE dataset, so the guards run against today's images: a consent given
    for "1 image has no caption" must not silently wave through the twelve that
    lost their caption since. Same reason the flags default to False: a retry is
    a launch, and a launch asks. Reported by 1Tomber (GitHub #23), whose retry
    was refused with no way to confirm and no way to see why."""
    gate = _require_aitoolkit()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = lt.retry_local_run(
            LOCAL_USER, int(d.get('record_id') or 0),
            **{k: bool(d.get(k)) for k in lt.CONFIRMATION_FLAGS})
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
    """▶ Continue d'un run cloud TERMINÉ (page Runs) : reprend depuis un checkpoint
    harvesté (from_step, défaut = dernier) et vise step_de_reprise + extra_steps —
    pod frais, mêmes garde-fous que tout launch ; le monitor dépose le checkpoint
    sur le pod avant de démarrer (auto-resume ai-toolkit). overrides = réglages sûrs
    (le service refuse toute autre clé → 400)."""
    gate = _require_cloud()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = ct.continue_cloud_run(LOCAL_USER, int(d.get('run_id') or 0),
                                    extra_steps=d.get('extra_steps', 1000),
                                    from_step=d.get('from_step'),
                                    overrides=d.get('overrides'),
                                    resume_mode=d.get('resume_mode', 'weights_only'),
                                    state_bundle_id=d.get('state_bundle_id'),
                                    transport=d.get('transport'),
                                    allow_parallel_run=bool(d.get('allow_parallel_run')))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/train/cloud/resume-plan')
def dataset_train_cloud_resume_plan():
    """The two roads a full model can take back to a pod, with their duration
    and their GPU cost — answered BEFORE anything is rented, like every other
    plan endpoint here. Always 200: a road that cannot be taken comes back
    unavailable with the reason, because a disabled button that explains itself
    is the whole point of this panel."""
    gate = _require_cloud()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        return jsonify({'ok': True, **ct.dense_resume_plan(
            LOCAL_USER, int(d.get('run_id') or 0),
            from_step=d.get('from_step'))})
    except Exception as e:
        return _map_error(e)


@bp.post('/dataset/train/cloud/recheck-delivery')
def dataset_train_cloud_recheck_delivery():
    """Re-verify one dense run's Hugging Face delivery without renting a GPU."""
    body = request.get_json(silent=True) or {}
    if body.get('run_id') in (None, ''):
        return jsonify({'error': 'run_id is required'}), 400
    try:
        result = ct.recheck_full_transformer_delivery(body['run_id'])
    except Exception as exc:
        return _map_error(exc)
    return jsonify(result)


@bp.post('/dataset/train/cloud/hub-presence')
def dataset_train_cloud_hub_presence():
    """Are these runs' Hugging Face repositories still there — asked NOW.

    Deliberately its own endpoint, and deliberately not a field of the
    checkpoints listing: the panel must paint before anything touches the
    network, and a Hub outage must never delay a page that is mostly about
    local files. The panel calls this after it has rendered, and only ever
    replaces "not re-checked since delivery" with something it measured.

    Cheap by construction — one metadata request per repository, cached with a
    short TTL in ``hub_presence`` — and never destructive: it reads, and does
    not rewrite ``artifact_status`` (that record is the delivery's minute, and
    ``recheck-delivery`` above is the one operation allowed to restate it).
    """
    from ..models import CloudTrainingRun
    from ..services import hub_presence

    body = request.get_json(silent=True) or {}
    wanted = body.get('run_ids')
    if not isinstance(wanted, list) or not wanted:
        return jsonify({'error': 'run_ids is required'}), 400
    ids = []
    for value in wanted:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    repo_of = {}
    for run in CloudTrainingRun.query.filter(CloudTrainingRun.id.in_(ids)).all():
        if not ct._is_full_transformer_run(run):
            continue
        repo = str(ct._run_param(run, 'hf_repo_id') or '').strip()
        if repo:
            repo_of[run.id] = repo
    checked = hub_presence.check_many(repo_of.values(),
                                      force=body.get('force') is True)
    return jsonify({'ok': True, 'results': {
        str(run_id): {**checked[repo], 'run_id': run_id}
        for run_id, repo in repo_of.items() if repo in checked}})


@bp.post('/dataset/train/cloud/fetch-local')
def dataset_train_cloud_fetch_local():
    """Bring ONE kept dense run's files home (or stop doing it).

    Answers immediately: the transfer is tens of minutes of ~26 GB and runs in
    the background, reporting through the run's own phase line. ``cancel: true``
    stops it and keeps every byte already downloaded.
    """
    body = request.get_json(silent=True) or {}
    if body.get('run_id') in (None, ''):
        return jsonify({'error': 'run_id is required'}), 400
    try:
        result = ct.fetch_dense_locally(body['run_id'],
                                        cancel=body.get('cancel') is True)
    except Exception as exc:
        return _map_error(exc)
    return jsonify(result)


@bp.post('/dataset/<int:dataset_id>/train/cloud/continue-local')
def dataset_train_cloud_continue_local(dataset_id):
    """▶ Continue d'un checkpoint LOCAL dans le CLOUD (voie « Cloud » de la modale
    Continue, côté dataset) : le fichier du run local est semé sur un pod frais
    (resume_ckpt_path) et le job vise step_de_reprise + extra_steps. Mêmes
    garde-fous que tout launch cloud (clé vast.ai, budget, limite de runs actifs,
    unicité par famille) — c'est un launch_cloud_training normal."""
    gate = _require_cloud()
    if gate:
        return gate
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    # The seed comes from the local lane, whose checkpoints are all LoRAs.
    # Freeze the source artifact kind instead of consulting mutable UI state.
    mode = 'lora'
    kw = {'extra_steps': d.get('extra_steps', 1000),
          'training_mode': mode}
    if 'base_model' in d:
        kw['base_model'] = d.get('base_model')
    if d.get('variant'):
        kw['variant'] = d.get('variant')
    if d.get('train_type'):
        kw['train_type'] = d.get('train_type')
    if d.get('from_step') is not None:
        kw['from_step'] = d.get('from_step')
    if d.get('overrides') is not None:
        kw['overrides'] = d.get('overrides')
    kw['resume_mode'] = d.get('resume_mode', 'weights_only')
    if d.get('state_bundle_id') is not None:
        kw['state_bundle_id'] = d.get('state_bundle_id')
    if d.get('gpu_name'):
        kw['gpu_name'] = d.get('gpu_name')
    kw['masked'] = d.get('masked')
    kw['allow_unverified_weights'] = bool(d.get('allow_unverified_weights'))
    kw['allow_caption_mismatch'] = bool(d.get('allow_caption_mismatch'))
    kw['allow_uncaptioned'] = bool(d.get('allow_uncaptioned'))
    kw['allow_caption_quality'] = bool(d.get('allow_caption_quality'))
    kw['allow_not_ready'] = bool(d.get('allow_not_ready'))
    kw['allow_parallel_run'] = bool(d.get('allow_parallel_run'))
    try:
        res = ct.continue_local_run_in_cloud(LOCAL_USER, dataset_id, **kw)
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
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    try:
        mode = lt.training_mode(ds, request.args.get('training_mode') or None)
        data = ct.gpu_tiers(LOCAL_USER, dataset_id,
                            train_type=request.args.get('train_type'),
                            variant=request.args.get('variant'),
                            steps=request.args.get('steps', type=int),
                            training_mode=mode)
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


@bp.get('/dataset/train/runs/<int:record_id>/lineage')
def dataset_train_run_lineage(record_id):
    """🌳 Genealogy tree of the lineage a run belongs to: nodes (every launch
    linked by continuations, local + cloud) and parent→child edges. Addressed
    by TrainingRunRecord id — the universal run node key (cloud rows expose it
    as record_id too). Open like the other Runs-hub reads: unknown id → 404."""
    tree = ct.run_lineage(record_id)
    if not tree.get('nodes'):
        return jsonify({'error': 'unknown run'}), 404
    return jsonify(tree)


@bp.get('/dataset/train/runs/compare')
def dataset_train_runs_compare():
    """Everything that differs between two runs: recipe, dataset (added/removed
    images, edited captions WITH their text, re-edited pixels) and the machine.

    Query: `?a=<record_id>&b=<record_id>`. Deliberately its own read rather than
    a fatter lineage payload — caption text is kilobytes per run and the graph
    draws dozens of nodes. Unknown id → 404."""
    from ..services import run_compare
    try:
        a = int(request.args.get('a', ''))
        b = int(request.args.get('b', ''))
    except (TypeError, ValueError):
        return jsonify({'error': 'two run ids are required'}), 400
    out = run_compare.compare(a, b)
    if out.get('error'):
        return jsonify(out), 404
    return jsonify(out)


@bp.get('/dataset/runs/archive/<sig>')
def dataset_run_archive_blob(sig):
    """Serve an ARCHIVED training image by its content hash — the only way to
    look at an image that has since been deleted from its dataset. Unknown or
    never-archived hash → 404 (the panel then says the picture is unavailable
    instead of showing a wrong one)."""
    from flask import send_file
    from ..services import run_archive
    path = run_archive.path_for(sig)
    if not path:
        return jsonify({'error': 'not archived'}), 404
    return send_file(path, max_age=31536000)


@bp.get('/dataset/train/runs/<int:record_id>/deletion-impact')
def dataset_train_run_deletion_impact(record_id):
    """What deleting this run would take with it, counted — read by the
    confirmation dialog so a destructive action is announced BEFORE it happens.

    Returns checkpoint notes, preview links, generated images that would lose
    their provenance (they are unlinked, never deleted), canvas positions,
    children that would be detached, and archived source images this run is the
    last referrer of. Unknown id → 404."""
    impact = ct.run_deletion_impact(record_id)
    if impact is None:
        return jsonify({'error': 'unknown run'}), 404
    return jsonify(impact)


@bp.delete('/dataset/train/runs/<int:record_id>')
def dataset_train_run_delete(record_id):
    """Remove a GONE run (no checkpoints on disk) from the lineage graph with
    everything that only existed for it: the record, its checkpoint notes, its
    preview links and its canvas position. Generated images are UNLINKED, not
    deleted; archived source blobs are freed only when no other run references
    them. A run whose checkpoints are still on disk is refused with 409 (delete
    those first) — never a silent erase. Children that resumed from it are
    detached, not deleted. Unknown id → 404.

    `?cascade=1` is the OPT-IN destructive mode the run panel's "Delete run"
    asks for: the checkpoints go to the trash, the generated images with them,
    then the row. It is a query flag rather than a new default so no existing
    caller of this route starts shredding files because the semantics moved
    under it. Even then children are DETACHED, 👍 images and LoRAs already
    deployed into ComfyUI are KEPT — see services.run_cascade_delete. A dataset
    that is training right now → 409; a checkpoint that could not be moved →
    409 with the counts of what did go, and the run row left in place."""
    if (request.args.get('cascade') or '').lower() in ('1', 'true', 'yes'):
        return _dataset_train_run_delete_cascade(record_id)
    status = ct.delete_run_record(record_id)
    if status == 'not_found':
        return jsonify({'error': 'unknown run'}), 404
    if status == 'has_saves':
        return jsonify({'error': 'This run still has checkpoints on disk. Delete '
                                 'its checkpoints first, then remove the run.'}), 409
    if status == 'has_model':
        return jsonify({'error': 'This run trained a full model that is still in '
                                 'its private Hugging Face repository. Removing '
                                 'the run would discard the only record of where '
                                 'that model is. Delete the repository first if '
                                 'you really want it gone.'}), 409
    if status == 'conflict':
        return jsonify({'error': 'This run is still referenced and could not be '
                                 'removed. Refresh and try again.'}), 409
    return jsonify({'ok': True})


def _dataset_train_run_delete_cascade(record_id):
    """The `?cascade=1` branch. Split out so the conservative path above reads
    exactly as it did — the two modes share a URL, never a body of code.

    A PARTIAL result is an error (409), not a 200 with a sad number: the run row
    is still there and its remaining weights are still on disk, so telling the
    user "deleted" would be a lie he only discovers on the next refresh. The
    message is already path-redacted by the service."""
    from ..services import run_cascade_delete
    out = run_cascade_delete.delete_run_cascade(record_id)
    status = out.get('status')
    if status == 'not_found':
        return jsonify({'error': 'unknown run'}), 404
    if status == 'training':
        where = ('a cloud pod' if out.get('error') == 'cloud' else 'this dataset')
        return jsonify({**out, 'error': f'{where} is training right now — stop the '
                                        'run before deleting it.'}), 409
    if status == 'partial':
        return jsonify({**out, 'error': 'Some files could not be removed, so the run '
                                        'was kept. ' + (out.get('error') or '')}), 409
    if status == 'conflict':
        return jsonify({**out, 'error': 'This run is still referenced and could not be '
                                        'removed. Refresh and try again.'}), 409
    return jsonify({**out, 'ok': True})


@bp.put('/dataset/train/runs/<int:record_id>/note')
def dataset_train_run_note(record_id):
    """Save the Lab's free-form note on a run. Unknown run → 404."""
    text = (request.get_json(silent=True) or {}).get('note', '')
    if not ct.set_run_note(record_id, text):
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True})


@bp.put('/dataset/train/runs/<int:record_id>/checkpoints/<int:step>/note')
def dataset_train_checkpoint_note(record_id, step):
    """Save the Lab's free-form note on one checkpoint (record_id, step)."""
    text = (request.get_json(silent=True) or {}).get('note', '')
    if not ct.set_checkpoint_note(record_id, step, text):
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True})


@bp.post('/dataset/<int:dataset_id>/lineage/previews')
def dataset_lineage_generate_previews(dataset_id):
    """🎨 Lab inline generation: render a same-prompt/same-seed preview for each
    selected checkpoint by reusing the Test-Studio engine pinned to those
    checkpoints at strength 1.0. Body: {prompt, seed, family,
    checkpoints:[{record_id, step}]}.

    A training holding the GPU → 409 (a training and a generation never share the
    GPU) — never a silent no-op. No checkpoint deployed for the family → 409
    needs_setup (the same 'set it up first' honesty the rest of the app uses). A
    vision pass holding the GPU, or a ComfyUI that isn't set up, surface through
    the engine as the Studio's usual 503 / structured-409. Unknown dataset → 404."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    if ct.training_in_progress():
        return jsonify({'error': 'GPU busy',
                        'detail': 'A LoRA training is running — previews share the '
                                  'GPU with training, so try again once it finishes.'
                        }), 409
    d = request.get_json(silent=True) or {}
    cks = d.get('checkpoints') or []
    if not isinstance(cks, list) or not cks:
        return jsonify({'error': 'no checkpoints selected'}), 400
    try:
        out = ct.generate_checkpoint_previews(
            LOCAL_USER, dataset_id, cks, prompt=d.get('prompt'),
            seed=d.get('seed'), family=d.get('family'))
    except Exception as e:
        return _map_error(e)
    if out.get('needs_setup'):
        return jsonify({'error': 'No deployed checkpoint to preview',
                        'detail': 'None of the selected checkpoints has a deployed '
                                  'LoRA for this family yet — import/deploy it first, '
                                  'then generate a preview.',
                        'needs_setup': True, 'skipped': out.get('skipped', [])}), 409
    return jsonify(out)


@bp.get('/dataset/train/runs/<run_key>/preview')
def dataset_train_run_preview(run_key):
    """Newest sample image a run produced — the Runs-hub card thumbnail.
    `run_key` is 'cloud-<id>' or 'rec-<id>' (same addressing as /share). The
    file path is resolved fully server-side from the run's own record (staging
    dir for cloud, the stamped local run dir otherwise) — the client never
    sends a path. Open like the other Runs-hub reads: unknown/sampleless → 404."""
    from flask import send_file
    p = ct.run_preview_path(run_key)
    if not p:
        return jsonify({'error': 'no preview for this run'}), 404
    return send_file(p, conditional=True)


def _parse_run_id_arg():
    """?run_id= as an int, or None when absent — but a malformed value (e.g.
    '?run_id=abc') is a 404, never the silent fallback to the newest run that
    `request.args.get('run_id', type=int)` gives: it also turns "the run I
    just addressed" into a stranger."""
    raw = request.args.get('run_id')
    if raw is None:
        return None, True
    try:
        return int(raw), True
    except (TypeError, ValueError):
        return None, False


@bp.get('/dataset/<int:dataset_id>/train/cloud/progress')
def dataset_train_cloud_progress(dataset_id):
    run_id, ok = _parse_run_id_arg()
    if not ok:
        return jsonify({'error': 'invalid run_id'}), 404
    try:
        return jsonify(ct.cloud_progress(
            LOCAL_USER, dataset_id,
            train_type=request.args.get('train_type'),
            run_id=run_id))
    except LookupError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return _map_error(e)


@bp.post('/dataset/train/cloud/stop')
def dataset_train_cloud_stop():
    """Stop a cloud run and report what really happened.

    The answer is never a courtesy 'ok': when no monitor thread is in a state
    to honour the request, request_stop terminates the pod itself, and if even
    that fails the payload carries the instance id the user must destroy by
    hand (HTTP stays 200 — the request was understood, the outcome is in the
    body, which is what the UI renders)."""
    d = request.get_json(silent=True) or {}
    res = ct.request_stop(d.get('run_id'))
    if not isinstance(res, dict):       # defensive: legacy bool contract
        res = {'ok': bool(res)}
    return jsonify(res)


@bp.get('/dataset/<int:dataset_id>/train/cloud/sample/<path:filename>')
def dataset_train_cloud_sample(dataset_id, filename):
    from flask import send_from_directory, abort
    # ?run_id addresses THAT run; ?train_type= resolves THAT family's newest
    # run (several families may train the same dataset in parallel); neither
    # -> plain newest, unchanged.
    run_id, ok = _parse_run_id_arg()
    if not ok:
        abort(404)
    run = ct.run_for(dataset_id, run_id=run_id,
                     train_type=request.args.get('train_type'))
    if not run or not run.staging_dir:
        abort(404)
    # send_from_directory refuses path traversal by construction
    return send_from_directory(os.path.join(run.staging_dir, 'samples'), filename)


@bp.get('/dataset/<int:dataset_id>/train/cloud/checkpoint')
def dataset_train_cloud_checkpoint(dataset_id):
    from flask import send_file, abort
    # ?run_id targets THAT run's file: with several finished runs of a family
    # in the hub history, 'newest run of the family' would serve the WRONG
    # checkpoint from an older row's button. A malformed value is a 404, not
    # the silent fallback to the newest run — this route hands out FILES, the
    # worst place to serve a stranger's weights for a typo'd id.
    rid, ok = _parse_run_id_arg()
    if not ok:
        abort(404)
    if rid is not None:
        from ..models import CloudTrainingRun
        from ..services import cloud_run_dataset as crd
        run = CloudTrainingRun.query.get(rid)
        # Same (id, table) ownership test: this endpoint SERVES the run's
        # checkpoint file, so an id-only match would hand a face dataset's caller
        # the weights of a video run that shares its id.
        if run and not crd.owns(run, dataset_id):
            run = None
    else:
        run = ct.latest_run_for(dataset_id, request.args.get('train_type'))
    if not run:
        abort(404)
    dense_response = _full_transformer_artifact_response(run)
    if dense_response:
        return dense_response
    # ?filename targets ONE harvested epoch of this run (the ◉ Graph's
    # per-checkpoint ⬇). Absent → the run's final LoRA, the historical
    # behaviour.
    fn = request.args.get('filename')
    if fn:
        # Resolved through the run's own save list (durable store first, legacy
        # staging second) — basename-only by construction, so the client can
        # never point this at a path of its choosing.
        path = ct.run_checkpoint_path(run, fn)
        if not path or not os.path.isfile(path):
            abort(404)
        return send_file(path, as_attachment=True)
    if not run.checkpoint_local_path or not os.path.isfile(run.checkpoint_local_path):
        abort(404)
    return send_file(run.checkpoint_local_path, as_attachment=True)


@bp.get('/dataset/<int:dataset_id>/train/checkpoint/file')
def dataset_train_checkpoint_file(dataset_id):
    """Download ONE local run-dir checkpoint by filename — the ◉ Graph's
    per-checkpoint ⬇ for local runs (the cloud sibling is train/cloud/checkpoint
    ?filename). Path-guarded server-side to this run's own saves (see
    lt.checkpoint_file_path); the client never sends a path. Unknown → 404."""
    from flask import send_file, abort
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        abort(404)
    fam = request.args.get('train_type') or None
    variant = request.args.get('variant') or None
    bm = request.args.get('base_model')
    kw = {} if bm is None else {'base_model': bm}
    if fam:
        kw['family'] = fam
    if variant:
        kw['variant'] = variant
    path = lt.checkpoint_file_path(
        LOCAL_USER, dataset_id, request.args.get('filename', ''), **kw)
    if not path:
        abort(404)
    return send_file(path, as_attachment=True)


@bp.get('/train/activity')
def train_activity():
    """🏋️ Live "something is training" signal for the nav indicator, local and
    cloud. Ungated and free by design (one flag + one COUNT): every page polls
    it, so it must never probe, touch the disk or reach the network. An
    unconfigured cloud simply reports zero."""
    return jsonify(ct.training_activity())


@bp.get('/train/canvas/datasets')
def train_canvas_datasets():
    """◉ LoRA Canvas index: which datasets have runs worth drawing, how many, and
    in which families. Cheap by design (no checkpoints, no disk) — the canvas
    fetches each selected dataset's genealogy separately, so the board and its
    filter appear immediately instead of after a full-library disk scan."""
    return jsonify(ct.canvas_dataset_index(LOCAL_USER))


@bp.post('/train/canvas/generate')
def train_canvas_generate():
    """◉ Generate from the LoRA Canvas — the same Test-Studio engine, driven by
    the checkpoints ticked on the board instead of by a picker. Body:
    {selections:[{dataset_id, checkpoint, record_id, step}], external_loras,
    …every Studio setting}. Selections MAY span several datasets (that is the
    point of the canvas); they may NOT span several families — the engine
    refuses, and the reason travels back so the button can say it. Same gates
    as the other launch routes: ComfyUI not set up → 409/503, missing
    models/nodes → the actionable 409 the Studio already returns.

    🧬 `combine: true` (the board's Blend toggle) switches from one pass per
    ticked checkpoint to ONE generation loading them all, each at the `weight`
    its selection carries, with every dataset's trigger word injected. It is the
    Test Studio's own Blend mode — the same engine argument, so the two screens
    cannot drift into two answers for one word."""
    from ._common import (_require_comfyui, _studio_arch_mismatch_response,
                          _studio_missing_response)
    gate = _require_comfyui()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = ct.canvas_generate(
            LOCAL_USER, d.get('selections') or [],
            strengths=d.get('strengths') or [1.0],
            seed=d.get('seed'), prompt=d.get('prompt'),
            # 📝 Lot : une passe par prompt coché dans l'historique du panneau.
            prompts=d.get('prompts'), z_model=d.get('z_model'),
            # ◉ La base est un AXE : le panneau du board dit « BASE MODEL (MULTI) ».
            z_models=d.get('z_models'),
            aspects=d.get('aspects'), cfgs=d.get('cfgs'), steps_list=d.get('steps'),
            steps2_list=d.get('steps2'), count=d.get('count'),
            permanent_loras=d.get('permanent_loras'), batch_loras=d.get('batch_loras'),
            external_loras=d.get('external_loras'),
            rebalance=d.get('rebalance'),
            rebalance_strength=d.get('rebalance_strength'),
            negative=d.get('negative'), sampler=d.get('sampler'),
            scheduler=d.get('scheduler'), weight_dtype=d.get('weight_dtype'),
            enhancer=d.get('enhancer'), enhancer_strength=d.get('enhancer_strength'),
            detail_amount=d.get('detail_amount'),
            resolution_tier=d.get('resolution_tier'),
            resolution_multiplier=d.get('resolution_multiplier'),
            init_image=d.get('init_image'), denoise=d.get('denoise'),
            combine=d.get('combine'))
    except Exception as e:
        from ..services.lora_test_studio import StudioArchMismatch, StudioAssetsMissing
        if isinstance(e, StudioArchMismatch):
            return _studio_arch_mismatch_response(e)
        if isinstance(e, StudioAssetsMissing):
            return _studio_missing_response(e)
        return _map_error(e)
    return jsonify({'ok': True, **{k: res[k]
                                   for k in ('created', 'seed', 'count', 'run_id')}})


@bp.get('/train/checkpoint/<int:record_id>/<int:step>/images')
def train_checkpoint_images(record_id, step):
    """🖼 Everything this checkpoint ever generated, newest first — the gallery
    the ◉ Canvas opens under a node. Reads the link written at generation time,
    so it holds images made from any surface (Test Studio, canvas, comparison
    grid). Open like the other Runs-hub reads; a checkpoint with no image simply
    answers an empty list plus the `unlinked` counter."""
    return jsonify(ct.checkpoint_gallery(
        record_id, step, limit=request.args.get('limit', default=120, type=int)))


@bp.post('/train/checkpoint/<int:record_id>/<int:step>/images/delete')
def train_checkpoint_images_delete(record_id, step):
    """🗑 Delete generated images from a checkpoint's gallery. Body:
    {image_ids: [id, …]}.

    A real delete: these rows are the Test Studio's cells, so they leave both
    surfaces — the confirmation says so before arming the button. Files are
    disposed of the recoverable way (OS recycle bin, else the app trash, else a
    permanent unlink only when both refuse); the mode used rides back in the
    answer, and `checkpoint_gallery` announces it beforehand. Ids not linked to
    this checkpoint are refused rather than deleted, and per-image failures are
    reported in `skipped` without aborting the batch — an empty selection is a
    no-op, never an error."""
    ids = (request.get_json(silent=True) or {}).get('image_ids') or []
    if not isinstance(ids, list):
        return jsonify({'error': 'image_ids must be a list'}), 400
    try:
        out = ct.delete_checkpoint_images(record_id, step, ids)
    except OSError as e:
        current_app.logger.warning('checkpoint gallery delete failed: %s', e)
        return jsonify({'error': 'Could not delete these images — a file is '
                                 'locked or unreachable. Try again.'}), 500
    return jsonify({'ok': True, **out})


@bp.get('/train/run/<int:record_id>/timeline')
def train_run_timeline(record_id):
    """Render-equivalent images across checkpoints, split into safe series."""
    from ..services import checkpoint_timeline as timeline
    try:
        return jsonify(timeline.checkpoint_timeline(record_id))
    except LookupError as exc:
        return jsonify({'error': str(exc)}), 404


@bp.get('/train/run/<int:record_id>/timeline/<series_id>/gif')
def train_run_timeline_gif(record_id, series_id):
    """Download a bounded animated GIF for one server-resolved timeline."""
    from flask import send_file
    from ..services import checkpoint_timeline as timeline
    if not timeline.acquire_gif_response_slot():
        response = jsonify({'error': 'too many timeline GIF downloads are active'})
        response.status_code = 429
        response.headers['Retry-After'] = '1'
        return response
    try:
        output, filename = timeline.render_timeline_gif(
            record_id,
            series_id,
            duration_ms=(request.args.get('duration_ms')
                         if request.args.get('duration_ms') is not None
                         else request.args.get('duration')),
            fade_frames=(request.args.get('fade_frames')
                         if request.args.get('fade_frames') is not None
                         else request.args.get('fade')),
            max_edge=request.args.get('max_edge'),
        )
    except timeline.GifRenderBusyError as exc:
        timeline.release_gif_response_slot()
        response = jsonify({'error': str(exc)})
        response.status_code = 429
        response.headers['Retry-After'] = '1'
        return response
    except timeline.GifRenderTooLargeError as exc:
        timeline.release_gif_response_slot()
        return jsonify({'error': str(exc)}), 413
    except LookupError as exc:
        timeline.release_gif_response_slot()
        return jsonify({'error': str(exc)}), 404
    except Exception:
        timeline.release_gif_response_slot()
        raise

    leased_output = _CloseCallbackFile(
        output, timeline.release_gif_response_slot)

    try:
        response = send_file(leased_output, mimetype='image/gif', as_attachment=True,
                             download_name=filename, max_age=0)
    except Exception:
        leased_output.close()
        raise
    response.call_on_close(leased_output.close)
    return response


@bp.get('/train/run/<int:record_id>/timeline/<series_id>/frame/<int:image_id>')
def train_run_timeline_frame(record_id, series_id, image_id):
    """Serve a metadata-free bounded WebP for timeline playback and WebM."""
    from flask import send_file
    from ..services import checkpoint_timeline as timeline
    if not timeline.acquire_preview_response_slot():
        response = jsonify({'error': 'too many timeline previews are active'})
        response.status_code = 429
        response.headers['Retry-After'] = '1'
        return response
    try:
        output, filename = timeline.render_timeline_preview(
            record_id, series_id, image_id)
    except timeline.GifRenderBusyError as exc:
        timeline.release_preview_response_slot()
        response = jsonify({'error': str(exc)})
        response.status_code = 429
        response.headers['Retry-After'] = '1'
        return response
    except timeline.GifRenderTooLargeError as exc:
        timeline.release_preview_response_slot()
        return jsonify({'error': str(exc)}), 413
    except LookupError as exc:
        timeline.release_preview_response_slot()
        return jsonify({'error': str(exc)}), 404
    except Exception:
        timeline.release_preview_response_slot()
        raise

    leased_output = _CloseCallbackFile(
        output, timeline.release_preview_response_slot)

    try:
        response = send_file(leased_output, mimetype='image/webp', as_attachment=False,
                             download_name=filename, max_age=300, conditional=True)
    except Exception:
        leased_output.close()
        raise
    response.call_on_close(leased_output.close)
    return response


@bp.get('/train/run/<int:record_id>/images')
def train_run_images(record_id):
    """🖼 Everything ONE RUN ever generated, grouped by checkpoint — the gallery
    the ◉ Canvas opens on a run CARD. Shaped like the checkpoint one (same rows,
    same `unlinked` footnote, same announced `delete_mode`), with `groups`
    instead of a flat list: steps descending, the step-less group last.

    Capped, and it says so: `per_step` images per checkpoint, `limit` overall,
    with `truncated` per group and for the whole answer. A run with fourteen
    checkpoints must not answer with a payload nobody can scroll."""
    return jsonify(ct.run_gallery(
        record_id,
        limit=request.args.get('limit', default=None, type=int),
        per_step=request.args.get('per_step', default=None, type=int)))


@bp.post('/train/run/<int:record_id>/images/delete')
def train_run_images_delete(record_id):
    """🗑 Delete generated images from a RUN's gallery. Body: {image_ids: [id, …]}.

    THE checkpoint delete with its scope widened (``step=None``) — same recycle
    bin, same refusal of ids that belong elsewhere, same per-image `skipped`
    report. Ids outside this run are refused, so widening the scope to a run
    never widens it to the library."""
    ids = (request.get_json(silent=True) or {}).get('image_ids') or []
    if not isinstance(ids, list):
        return jsonify({'error': 'image_ids must be a list'}), 400
    try:
        out = ct.delete_checkpoint_images(record_id, None, ids)
    except OSError as e:
        current_app.logger.warning('run gallery delete failed: %s', e)
        return jsonify({'error': 'Could not delete these images — a file is '
                                 'locked or unreachable. Try again.'}), 500
    return jsonify({'ok': True, **out})


def _zip_ids_arg():
    """The optional `ids=1,2,3` selection. Absent → the whole scope; present but
    unparseable → an empty selection, which the plan then refuses out loud. A
    silently-ignored malformed argument would hand over the WHOLE gallery to a
    click that meant "these three"."""
    raw = request.args.get('ids')
    if raw is None:
        return None
    return [p for p in raw.split(',') if p.strip()]


def _gallery_zip(record_id, step):
    """Shared body of the two ZIP routes — see services.gallery_download for why
    the file NAME is the whole feature. The plan runs first so a scope with
    nothing left on disk is refused with a reason instead of answering an empty
    archive; `_zip_download` (routes.datasets) owns the spool whose lifetime has
    to outlive this function."""
    from ..services import gallery_download as gdl
    from .datasets import _zip_download
    plan = gdl.gallery_download_plan(record_id, step, image_ids=_zip_ids_arg())
    if not plan['ok']:
        return jsonify({'error': plan['note']}), 404
    response = _zip_download(lambda out: gdl.write_gallery_zip(plan['entries'], out),
                             plan['filename'])
    # Readable by a fetch() caller, so the panel can state what actually went in
    # even when the archive itself is handed straight to the browser.
    response.headers['X-Lds-Zip-Images'] = str(plan['included'])
    response.headers['X-Lds-Zip-Total'] = str(plan['total'])
    return response


@bp.get('/train/image/<int:image_id>/download')
def train_image_download(image_id):
    """⬇ ONE generated image, under a name that still says where it came from.

    Resolved here rather than left to `<a download>` on the image URL: a file
    that has been cleaned off the disk would otherwise be saved as a 404 page
    wearing a .png name, and the user would find out by opening it."""
    from flask import send_file
    from ..services import gallery_download as gdl
    path, name = gdl.single_image_download(image_id)
    if path is None:
        return jsonify({'error': name}), 404
    return send_file(path, as_attachment=True, download_name=name)


@bp.get('/train/run/<int:record_id>/images/zip')
def train_run_images_zip(record_id):
    """⬇ A whole RUN's gallery as one ZIP — optional `?ids=` for a selection."""
    return _gallery_zip(record_id, None)


@bp.get('/train/run/<int:record_id>/images/zip/plan')
def train_run_images_zip_plan(record_id):
    """What that ZIP would hold, without building it: counts, the cap, and how
    many files have gone missing. The panel asks this BEFORE it downloads so
    every cut is on screen rather than discovered inside the archive."""
    from ..services import gallery_download as gdl
    plan = gdl.gallery_download_plan(record_id, None, image_ids=_zip_ids_arg())
    return jsonify({k: v for k, v in plan.items() if k != 'entries'})


@bp.get('/train/checkpoint/<int:record_id>/<int:step>/images/zip')
def train_checkpoint_images_zip(record_id, step):
    """⬇ One CHECKPOINT's gallery as a ZIP — the run route, narrowed to a step."""
    return _gallery_zip(record_id, step)


@bp.get('/train/checkpoint/<int:record_id>/<int:step>/images/zip/plan')
def train_checkpoint_images_zip_plan(record_id, step):
    """The checkpoint-scoped preflight — same answer, narrower scope."""
    from ..services import gallery_download as gdl
    plan = gdl.gallery_download_plan(record_id, step, image_ids=_zip_ids_arg())
    return jsonify({k: v for k, v in plan.items() if k != 'entries'})


@bp.get('/train/canvas/positions')
def train_canvas_positions():
    """◉ LoRA Canvas: every remembered card position, grouped by dataset id.
    One request for the whole board — the lanes need their overrides before the
    first paint, and N round-trips for a few dozen tiny rows would cost more
    than the genealogy fetches they precede."""
    return jsonify(ct.canvas_positions(LOCAL_USER))


@bp.put('/dataset/<int:dataset_id>/canvas/positions')
def dataset_canvas_positions_save(dataset_id):
    """Remember where cards sit in ONE lane. Body: {positions:[{record_id,x,y}]}.
    Upsert, so re-sending the same coordinates is a no-op — the canvas re-pins a
    lane whenever it gains a run."""
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(ct.save_canvas_positions(
            LOCAL_USER, dataset_id, data.get('positions')))
    except LookupError:
        return jsonify({'error': 'not found'}), 404


@bp.delete('/dataset/<int:dataset_id>/canvas/positions')
def dataset_canvas_positions_clear(dataset_id):
    """✦ Tidy up one lane: forget every dragged position and fall back to the
    automatic tree."""
    try:
        return jsonify(ct.clear_canvas_positions(LOCAL_USER, dataset_id))
    except LookupError:
        return jsonify({'error': 'not found'}), 404


@bp.get('/train/canvas/external-loras')
def canvas_external_loras_get():
    """🔌 The board's external LoRA plugin nodes, as persisted."""
    return jsonify({'loras': cfg.get('canvas.external_loras', []) or []})


@bp.put('/train/canvas/external-loras')
def canvas_external_loras_put():
    """Replace the board's external LoRA nodes. Sanitizes: dedupe by filename,
    reject path-traversal/absolute/drive-letter names (dropped, not erred —
    this is a save-what-survives sanitizer, same spirit as the sibling
    `save_canvas_positions`), cap 16, strength clamped [0..2] (default 1.0),
    x/y coerced to floats."""
    from ..services.lora_test_studio import _is_unsafe_external_lora_name
    data = request.get_json(silent=True) or {}
    raw = data.get('loras')
    cleaned, seen = [], set()
    for e in (raw if isinstance(raw, list) else []):
        if not isinstance(e, dict):
            continue
        fn = str(e.get('filename') or '').strip()
        if not fn or fn in seen or _is_unsafe_external_lora_name(fn):
            continue
        seen.add(fn)
        try:
            st = max(0.0, min(2.0, round(float(e.get('strength', 1.0)), 2)))
        except (TypeError, ValueError):
            st = 1.0

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0
        cleaned.append({'filename': fn, 'strength': st,
                        'x': _f(e.get('x')), 'y': _f(e.get('y'))})
        if len(cleaned) >= 16:
            break
    cfg.save_config({'canvas': {'external_loras': cleaned}})
    return jsonify({'ok': True, 'loras': cleaned})


@bp.get('/train/canvas/images')
def train_canvas_images():
    """🖼 Every image pinned on the ◉ LoRA Canvas, grouped by dataset id, with
    the image row alongside its geometry — one request for the whole board, like
    the card positions it sits next to. Rows whose image is gone are pruned
    server-side rather than answered."""
    return jsonify(ct.canvas_image_nodes(LOCAL_USER))


@bp.put('/dataset/<int:dataset_id>/canvas/images')
def dataset_canvas_images_save(dataset_id):
    """Remember pinned images of ONE lane.
    Body: {nodes:[{image_id,x,y,w,h,visible}]}.

    Closing a pinned image is this call with ``visible: false`` — the geometry
    stays, so re-opening puts the picture back exactly where and at the size it
    was closed at."""
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(ct.save_canvas_image_nodes(
            LOCAL_USER, dataset_id, data.get('nodes')))
    except LookupError:
        return jsonify({'error': 'not found'}), 404


@bp.delete('/dataset/<int:dataset_id>/canvas/images')
def dataset_canvas_images_clear(dataset_id):
    """Forget every pinned image of one lane, geometry included. Deliberately
    NOT what ✦ Tidy up calls — see clear_canvas_image_nodes."""
    try:
        return jsonify(ct.clear_canvas_image_nodes(LOCAL_USER, dataset_id))
    except LookupError:
        return jsonify({'error': 'not found'}), 404


@bp.get('/train/canvas/layouts')
def train_canvas_layouts():
    """💾 The named board arrangements this install has kept."""
    return jsonify(ct.canvas_layout_presets(LOCAL_USER))


@bp.post('/train/canvas/layouts')
def train_canvas_layouts_save():
    """Keep the board's current arrangement under a name.
    Body: {name, positions:{ds:[{record_id,x,y}]}, images:{ds:[{image_id,...}]}}.

    Saving under an existing name overwrites it — "save" on a board you have
    just adjusted means "this is the arrangement now"."""
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(ct.save_canvas_layout_preset(
            LOCAL_USER, data.get('name'),
            positions=data.get('positions'), images=data.get('images')))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@bp.post('/train/canvas/layouts/<int:preset_id>/apply')
def train_canvas_layouts_apply(preset_id):
    """Put a remembered arrangement back. Everything travels through the live
    writers, so anything that no longer exists is simply not restored."""
    try:
        return jsonify(ct.apply_canvas_layout_preset(LOCAL_USER, preset_id))
    except LookupError:
        return jsonify({'error': 'not found'}), 404


@bp.delete('/train/canvas/layouts/<int:preset_id>')
def train_canvas_layouts_delete(preset_id):
    try:
        return jsonify(ct.delete_canvas_layout_preset(LOCAL_USER, preset_id))
    except LookupError:
        return jsonify({'error': 'not found'}), 404


@bp.get('/dataset/<int:dataset_id>/train/lineage')
def dataset_train_dataset_lineage(dataset_id):
    """🌳 Genealogy forest of ALL this dataset's runs (every launch + its
    checkpoints), for the ◉ Graph the Checkpoints & LoRAs manager opens.
    Optionally scoped to the family/variant the panel shows (train_type/variant).
    Unlike the per-run lineage there is no single current run. Empty → 200 with
    an empty tree so the caller renders nothing, never an error page."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    return jsonify(ct.dataset_lineage(
        dataset_id, train_type=request.args.get('train_type') or None,
        variant=request.args.get('variant') or None))
