"""Face Dataset Maker API — create/list/get + generate (API-engine or Klein
fan-out) + import/classify/caption (Qwen3-VL) + curation + crop + export ZIP.

No login — single local user (`cfg.LOCAL_USER`). Vision-dependent routes borrow
the GPU-exclusive window (`gpu_exclusive_vision_window`) so a vision pass never
fights ComfyUI for the single GPU.
"""
import io
import os
import uuid

from flask import Blueprint, request, jsonify, send_file, send_from_directory

from ..config import LOCAL_USER
from ..gpu_window import gpu_exclusive_vision_window
from ..services import face_dataset_service as svc
from ..services import lora_test_studio as lts
from ..services.face_variations import VARIATION_CATALOG, select_preset
from ..utils.comfyui import KREA_ALLOWED_SAMPLERS, KREA_ALLOWED_SCHEDULERS, get_krea_loras
from ._common import _map_error, _require_comfyui

bp = Blueprint('datasets', __name__, url_prefix='/api')

_PRESET_NAMES = ('balanced_25', 'zimage_12', 'balanced_multiformat',
                 'face_focused', 'fullbody_focused')


@bp.post('/dataset/create')
def dataset_create():
    data = request.get_json(silent=True) or {}
    name, trigger = (data.get('name') or '').strip(), (data.get('trigger_word') or '').strip()
    if not name or not trigger:
        return jsonify({'error': 'name and trigger_word are required'}), 400
    try:
        ds = svc.create_dataset(LOCAL_USER, name, trigger, kind=data.get('kind'),
                                concept_desc=data.get('concept_desc'),
                                train_type=data.get('train_type'))
    except ValueError as e:
        # concept dataset without a concept description -> 400 (not a 500)
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'id': ds.id})


@bp.post('/dataset/<int:dataset_id>/train-type')
def dataset_set_train_type(dataset_id):
    """Change a dataset's target model family (Z-Image/SDXL/Krea) after creation.
    Dataset metadata — NOT ai-toolkit-gated, so you can organize the menu even
    before training is configured. Keeps the TrainingPanel and the grouped menu in sync."""
    data = request.get_json(silent=True) or {}
    ok = svc.set_train_type(LOCAL_USER, dataset_id, data.get('train_type'))
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.get('/dataset/variations')
def dataset_variations():
    return jsonify({'catalog': VARIATION_CATALOG,
                    'presets': {n: [e['id'] for e in select_preset(n)] for n in _PRESET_NAMES}})


@bp.get('/dataset/list')
def dataset_list():
    dss = svc.list_datasets(LOCAL_USER)
    return jsonify({'datasets': [
        {'id': d.id, 'name': d.name, 'trigger_word': d.trigger_word, 'ref_filename': d.ref_filename,
         'kind': 'concept' if (d.kind or '').lower() == 'concept' else 'character',
         'train_type': (d.train_type or 'zimage')}
        for d in dss]})


@bp.get('/dataset/<int:dataset_id>')
def dataset_get(dataset_id):
    payload = svc.dataset_payload(LOCAL_USER, dataset_id)
    return (jsonify(payload), 200) if payload else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/ref')
def dataset_set_ref(dataset_id):
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'error': 'no file'}), 400
    raw = f.read()
    try:
        with gpu_exclusive_vision_window():
            webp, head_detected = svc.face_crop_to_square_webp(
                raw, pad=svc.REF_CROP_PAD, return_detected=True)
    except Exception as e:
        return _map_error(e)
    dsdir = svc._dataset_dir(dataset_id)
    # Keep the full-frame ORIGINAL (aspect-kept, capped ~2048) so the crop editor can
    # widen back out later — the auto head-crop is only the default framing, not a
    # one-way lossy door (the old behavior discarded it and re-crops could only tighten).
    orig_fn = f"{LOCAL_USER}_datasetreforig_{uuid.uuid4().hex[:8]}.webp"
    with open(os.path.join(dsdir, orig_fn), 'wb') as fh:
        fh.write(svc.normalize_to_webp(raw, size=2048))
    fn = f"{LOCAL_USER}_datasetref_{uuid.uuid4().hex[:8]}.webp"
    with open(os.path.join(dsdir, fn), 'wb') as fh:
        fh.write(webp)
    ds.ref_original_filename = orig_fn
    ds.ref_filename = fn
    svc.db.session.commit()
    resp = {'ok': True, 'ref_filename': fn, 'head_crop': head_detected}
    if not head_detected:
        # GUARD-RAIL: don't silently ship a body-centered crop. Tell the user WHY the
        # auto head-crop didn't run — the usual cause on a fresh install is the Ollama
        # vision model not being pulled — and how to recover (Setup + the Crop button).
        from .. import capabilities
        model_ready = capabilities.probe_ollama_model()['ok']
        resp['warning'] = (
            "Auto head-crop needs the Ollama vision model, which isn't ready yet — "
            'used a centered crop. Finish the Ollama step in Setup, then click Crop to re-center on the face.'
            if not model_ready else
            "Couldn't detect a face — used a centered crop. Use the Crop button to adjust it manually."
        )
    return jsonify(resp)


@bp.post('/dataset/<int:dataset_id>/ref/extra')
def dataset_add_extra_ref(dataset_id):
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'error': 'no file'}), 400
    try:
        fn = svc.add_extra_ref(LOCAL_USER, dataset_id, f.read())
    except ValueError as e:
        return _map_error(e)
    return jsonify({'ok': True, 'filename': fn})


@bp.post('/dataset/<int:dataset_id>/ref/extra/delete')
def dataset_remove_extra_ref(dataset_id):
    data = request.get_json(silent=True) or {}
    ok = svc.remove_extra_ref(LOCAL_USER, dataset_id, data.get('filename') or '')
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/ref/crop')
def dataset_ref_crop(dataset_id):
    data = request.get_json(silent=True) or {}
    try:
        ok = svc.crop_reference(LOCAL_USER, dataset_id,
                                int(data['x']), int(data['y']), int(data['w']), int(data['h']))
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'invalid crop box'}), 400
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/ref/recrop-auto')
def dataset_ref_recrop_auto(dataset_id):
    """Reset the reference to the automatic head-crop, re-run on the kept ORIGINAL
    (no re-upload needed). Same GPU vision window as the initial upload."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        with gpu_exclusive_vision_window():
            ok, head_detected = svc.recrop_reference_auto(LOCAL_USER, dataset_id)
    except Exception as e:
        return _map_error(e)
    if not ok:
        return jsonify({'error': 'no reference to re-crop'}), 400
    resp = {'ok': True, 'head_crop': head_detected}
    if not head_detected:
        from .. import capabilities
        model_ready = capabilities.probe_ollama_model()['ok']
        resp['warning'] = (
            "Auto head-crop needs the Ollama vision model, which isn't ready yet — "
            'used a centered crop. Finish the Ollama step in Setup, then adjust with Crop.'
            if not model_ready else
            "Couldn't detect a face — used a centered crop. Use Crop to adjust it manually."
        )
    return jsonify(resp)


@bp.post('/dataset/<int:dataset_id>/generate')
def dataset_generate(dataset_id):
    data = request.get_json(silent=True) or {}
    generator = data.get('generator') or 'klein'
    try:
        if generator in svc.API_ENGINES:
            # API path (Gemini Nano Banana Pro or OpenAI ChatGPT gpt-image-2):
            # no GPU, rows filled by a background thread — the existing polling
            # UI tracks them.
            from flask import current_app
            ids = svc.generate_variations_nanobanana(
                current_app._get_current_object(), LOCAL_USER, dataset_id,
                data.get('variations') or [], data.get('multiplier', 1),
                engine=generator)
        else:
            ids = svc.generate_variations(LOCAL_USER, dataset_id,
                                          data.get('variations') or [], data.get('multiplier', 1),
                                          data.get('klein_model'),
                                          lora_strength=data.get('lora_strength'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'created': len(ids)})


@bp.post('/dataset/<int:dataset_id>/import')
def dataset_import(dataset_id):
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    files = [f.read() for f in request.files.getlist('files') if f and f.filename]
    if not files:
        return jsonify({'error': 'no files'}), 400
    if len(files) > 20:
        return jsonify({'error': 'max 20 images per import'}), 400
    # Head-crop OPTIONNEL (form field crop='0' → OFF) : un plan buste/corps importé
    # doit pouvoir rester tel quel — le crop tête carré systématique transformait
    # tout import en gros plan. Dataset CONCEPT : jamais de head-crop.
    # Sans crop → import BRUT (ratio préservé) → aucune passe vision → PAS de
    # fenêtre GPU exclusive (on ne stoppe pas ComfyUI pour rien).
    stats = {}
    want_crop = (not svc.is_concept(ds)) and request.form.get('crop', '1') != '0'
    if not want_crop:
        ids, failed = svc.import_images(LOCAL_USER, dataset_id, files, crop=False,
                                        dedupe=True, stats=stats)
        return jsonify({'ok': True, 'imported': len(ids), 'failed': failed,
                        'duplicates': stats.get('duplicates', 0)})
    try:
        # batch (head-crop vision par image) : heartbeat de la fenêtre = ComfyUI arrêté
        # tout le batch ; le TTL n'est qu'un filet anti-crash.
        with gpu_exclusive_vision_window(flag_ttl=600):
            ids, failed = svc.import_images(LOCAL_USER, dataset_id, files, crop=True,  # auto head-crop
                                            dedupe=True, stats=stats)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'imported': len(ids), 'failed': failed,
                    'duplicates': stats.get('duplicates', 0)})


@bp.post('/dataset/<int:dataset_id>/scrape-import')
def dataset_scrape_import(dataset_id):
    """Scrape DIRECT → dataset CONCEPT : downloads the SELECTED scanned images
    ({items:[{url,title}]}) straight into the dataset. Quality filters + dedup
    live in the service. Concept-only (the character path would need per-image
    head-crop + a GPU window)."""
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    if not svc.is_concept(ds):
        return jsonify({'error': 'concept datasets only'}), 400
    data = request.get_json(silent=True) or {}
    items = data.get('items') or []
    if not isinstance(items, list) or not items:
        return jsonify({'error': 'no items'}), 400
    if len(items) > svc.SCRAPE_IMPORT_MAX:
        return jsonify({'error': f'max {svc.SCRAPE_IMPORT_MAX} images per import'}), 400
    res = svc.scrape_import_urls(LOCAL_USER, dataset_id, items)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/<int:dataset_id>/classify')
def dataset_classify(dataset_id):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        with gpu_exclusive_vision_window(flag_ttl=1800):
            n = svc.classify_images(LOCAL_USER, dataset_id)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'classified': n})


@bp.post('/dataset/<int:dataset_id>/caption')
def dataset_caption(dataset_id):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    force = bool(data.get('force'))
    mode = data.get('mode')  # 'prose' | 'booru' | None (None → auto selon train_type)
    try:
        with gpu_exclusive_vision_window(flag_ttl=1800):
            n = svc.caption_images(LOCAL_USER, dataset_id, force=force, mode=mode)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'captioned': n})


@bp.post('/dataset/<int:dataset_id>/analyze-faces')
def dataset_analyze_faces(dataset_id):
    # CPU (onnxruntime CPU-only) -> PAS de fenêtre GPU exclusive, ComfyUI non stoppé.
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        counts = svc.analyze_faces(LOCAL_USER, dataset_id)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'states': counts, 'analyzed': sum(counts.values())})


@bp.post('/dataset/image/<int:image_id>/status')
def dataset_image_status(image_id):
    data = request.get_json(silent=True) or {}
    try:
        ok = svc.set_image_status(LOCAL_USER, image_id, data.get('status'))
    except Exception as e:
        return _map_error(e)
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/delete')
def dataset_delete(dataset_id):
    ok = svc.delete_dataset(LOCAL_USER, dataset_id)
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/cancel')
def dataset_cancel(dataset_id):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    n = svc.cancel_pending(LOCAL_USER, dataset_id)
    return jsonify({'ok': True, 'cancelled': n})


@bp.post('/dataset/image/<int:image_id>/delete')
def dataset_image_delete(image_id):
    ok = svc.delete_image(LOCAL_USER, image_id)
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/image/<int:image_id>/regenerate')
def dataset_image_regenerate(image_id):
    data = request.get_json(silent=True) or {}
    try:
        from flask import current_app
        job_id = svc.regenerate_image(LOCAL_USER, image_id,
                                      lora_strength=data.get('lora_strength'),
                                      app=current_app._get_current_object())
    except Exception as e:
        return _map_error(e)
    if job_id is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, 'job_id': job_id})


@bp.post('/dataset/<int:dataset_id>/captions/replace')
def dataset_captions_replace(dataset_id):
    """Bulk find/replace across the KEPT images' captions.
    Body: {find: str, replace: str, mode: 'text'|'tag'} — 'tag' does whole-tag
    (comma-separated) replacement/removal for booru captions."""
    data = request.get_json(silent=True) or {}
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        n = svc.replace_in_captions(LOCAL_USER, dataset_id,
                                    data.get('find'), data.get('replace') or '',
                                    mode=data.get('mode') or 'text')
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'changed': n})


@bp.post('/dataset/<int:dataset_id>/images/batch')
def dataset_images_batch(dataset_id):
    """Multi-select curation: apply one action to many images in one request.
    Body: {ids: [int, ...], action: keep|reject|pending|delete|clear_caption}."""
    data = request.get_json(silent=True) or {}
    action = data.get('action')
    ids = data.get('ids')
    if action not in svc.BATCH_ACTIONS:
        return jsonify({'error': 'invalid action'}), 400
    if not isinstance(ids, list) or not ids:
        return jsonify({'error': "'ids' must be a non-empty list"}), 400
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    n = svc.batch_image_action(LOCAL_USER, dataset_id, ids, action)
    return jsonify({'ok': True, 'affected': n})


@bp.post('/dataset/<int:dataset_id>/purge')
def dataset_purge(dataset_id):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    n = svc.purge_unused(LOCAL_USER, dataset_id)
    return jsonify({'ok': True, 'purged': n})


@bp.post('/dataset/image/<int:image_id>/caption')
def dataset_image_caption(image_id):
    data = request.get_json(silent=True) or {}
    ok = svc.set_image_caption(LOCAL_USER, image_id, data.get('caption', ''))
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/image/<int:image_id>/crop')
def dataset_image_crop(image_id):
    data = request.get_json(silent=True) or {}
    try:
        ok = svc.crop_image(LOCAL_USER, image_id,
                            int(data['x']), int(data['y']), int(data['w']), int(data['h']))
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'invalid crop box'}), 400
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.get('/dataset/<int:dataset_id>/export')
def dataset_export(dataset_id):
    try:
        data = svc.build_export_zip(LOCAL_USER, dataset_id)
    except ValueError as e:
        return _map_error(e)
    return send_file(io.BytesIO(data), mimetype='application/zip', as_attachment=True,
                     download_name=f'dataset_{dataset_id}.zip')


@bp.get('/dataset/<int:dataset_id>/img/<path:filename>')
def dataset_image_file(dataset_id, filename):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(svc._dataset_dir(dataset_id), filename)


# ---------------------------------------------------------------------------
# LoRA Test Studio routes (checkpoint x strength sweep, per-dataset)
# ---------------------------------------------------------------------------

@bp.get('/dataset/<int:dataset_id>/lora-test/status')
def lora_test_status(dataset_id):
    """Poll payload: testable checkpoints, grid cells, scores, best cell,
    pending count and the persisted best_settings. `?family=` scope la pipeline
    (ZIT/SDXL/Krea) ; absent → famille effective par défaut du dataset."""
    payload = lts.studio_payload(LOCAL_USER, dataset_id, family=request.args.get('family'))
    return (jsonify(payload), 200) if payload else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/lora-test/run')
def lora_test_run(dataset_id):
    gate = _require_comfyui()
    if gate:
        return gate
    d = request.get_json(silent=True) or {}
    try:
        res = lts.create_run(LOCAL_USER, dataset_id,
                             d.get('checkpoints') or [], d.get('strengths') or [],
                             seed=d.get('seed'), prompt=d.get('prompt'),
                             z_model=d.get('z_model'), z_models=d.get('z_models'),
                             aspects=d.get('aspects'),
                             cfgs=d.get('cfgs'), steps_list=d.get('steps'),
                             steps2_list=d.get('steps2'),
                             count=d.get('count'), family=d.get('family'),
                             permanent_loras=d.get('permanent_loras'),
                             batch_loras=d.get('batch_loras'),
                             rebalance=d.get('rebalance'),
                             rebalance_strength=d.get('rebalance_strength'),
                             # Parité Generate — réglages globaux du run.
                             negative=d.get('negative'), sampler=d.get('sampler'),
                             scheduler=d.get('scheduler'), weight_dtype=d.get('weight_dtype'),
                             enhancer=d.get('enhancer'), enhancer_strength=d.get('enhancer_strength'),
                             detail_amount=d.get('detail_amount'),
                             resolution_tier=d.get('resolution_tier'),
                             init_image=d.get('init_image'), denoise=d.get('denoise'))
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'created': res['created'], 'seed': res['seed'],
                    'count': res.get('count', 1)})


@bp.post('/dataset/lora-test/image/<int:image_id>/rate')
def lora_test_rate(image_id):
    d = request.get_json(silent=True) or {}
    ok = lts.rate_image(LOCAL_USER, image_id, d.get('rating'))
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'ok': False, 'error': 'invalid'}), 400)


@bp.post('/dataset/<int:dataset_id>/lora-test/cancel')
def lora_test_cancel(dataset_id):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    n = lts.cancel_run(LOCAL_USER, dataset_id)
    return jsonify({'ok': True, 'cancelled': n})


@bp.post('/dataset/<int:dataset_id>/lora-test/resume')
def lora_test_resume(dataset_id):
    gate = _require_comfyui()
    if gate:
        return gate
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        res = lts.resume_run(LOCAL_USER, dataset_id)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/<int:dataset_id>/lora-test/best')
def lora_test_best(dataset_id):
    d = request.get_json(silent=True) or {}
    try:
        best = lts.set_best_settings(LOCAL_USER, dataset_id,
                                     d.get('checkpoint'), d.get('strength'),
                                     z_model=d.get('z_model'), cfg=d.get('cfg'),
                                     steps=d.get('steps'), steps2=d.get('steps2'),
                                     aspect=d.get('aspect'))
    except ValueError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True, 'best_settings': best})


@bp.delete('/dataset/<int:dataset_id>/lora-test/best')
def lora_test_best_clear(dataset_id):
    """Supprime le réglage mémorisé du dataset. `?family=` → n'efface que cette
    pipeline (les autres familles gardent leur meilleur réglage) ; absent → tout."""
    try:
        lts.clear_best_settings(LOCAL_USER, dataset_id, family=request.args.get('family'))
    except ValueError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True})


@bp.delete('/dataset/<int:dataset_id>/lora-test/prompt')
def lora_test_prompt_delete(dataset_id):
    """Supprime un prompt récent (et ses cellules/images de test)."""
    d = request.get_json(silent=True) or {}
    try:
        n = lts.delete_prompt(LOCAL_USER, dataset_id, d.get('prompt', ''))
    except ValueError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True, 'deleted': n})


@bp.post('/dataset/<int:dataset_id>/lora-test/score-faces')
def lora_test_score_faces(dataset_id):
    """Score facial objectif des cellules du Studio (InsightFace, subprocess CPU
    — pas de fenêtre GPU) vs la référence du dataset → « best epoch » auto."""
    d = request.get_json(silent=True) or {}
    try:
        res = lts.score_faces(LOCAL_USER, dataset_id, family=d.get('family'))
    except ValueError as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True, **res})


@bp.get('/index_config')
def index_config():
    """Static config for the Studio's generation-settings panel (Krea sampler/
    scheduler dropdowns + always-on LoRA candidates). Deterministic field list:
    only fields actually read by StudioGenerationSettings.jsx off /api/index_config
    (`config.krea_loras`, `config.krea_samplers`, `config.krea_schedulers`) — SRC's
    route also returns zimage/sdxl model lists, prompt-builder options, klein
    models etc., none of which any lifted component reads."""
    return jsonify({
        'krea_loras': get_krea_loras(),
        'krea_samplers': KREA_ALLOWED_SAMPLERS,
        'krea_schedulers': KREA_ALLOWED_SCHEDULERS,
    })
