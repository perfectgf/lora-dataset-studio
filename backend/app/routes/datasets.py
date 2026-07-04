"""Face Dataset Maker API — create/list/get + generate (API-engine or Klein
fan-out) + import/classify/caption (Qwen3-VL) + curation + crop + export ZIP.

No login — single local user (`cfg.LOCAL_USER`). Vision-dependent routes borrow
the GPU-exclusive window (`_vision_window`) so a vision pass never fights
ComfyUI for the single GPU; until Task 15 lands that window is a no-op.
"""
import io
import os
import uuid

from flask import Blueprint, request, jsonify, send_file, send_from_directory

from ..config import LOCAL_USER
from ..services import face_dataset_service as svc
from ..services.face_variations import VARIATION_CATALOG, select_preset

bp = Blueprint('datasets', __name__, url_prefix='/api')

_PRESET_NAMES = ('balanced_25', 'zimage_12', 'balanced_multiformat',
                 'face_focused', 'fullbody_focused')


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _vision_window(**kwargs):
    """GPU-exclusive window for vision passes (classify/caption/head-crop).
    Falls back to a no-op context manager until Task 15 lands `gpu_window`."""
    try:
        from ..gpu_window import gpu_exclusive_vision_window
    except ImportError:
        return _NullContext()
    return gpu_exclusive_vision_window(**kwargs)


def _map_error(e: Exception):
    """Map a service/vision exception to a Flask (body, status) tuple.
    Unrecognized exceptions are re-raised (-> 500, a real bug)."""
    try:
        from ..gpu_window import GpuBusyError
    except ImportError:
        GpuBusyError = ()  # isinstance(x, ()) is always False -> branch below is skipped
    if isinstance(e, GpuBusyError):
        return jsonify({'error': 'GPU busy, try again'}), 503
    if isinstance(e, ValueError):
        return jsonify({'error': str(e)}), 400
    if isinstance(e, RuntimeError):
        return jsonify({'error': str(e)}), 409
    raise e


@bp.post('/dataset/create')
def dataset_create():
    data = request.get_json(silent=True) or {}
    name, trigger = (data.get('name') or '').strip(), (data.get('trigger_word') or '').strip()
    if not name or not trigger:
        return jsonify({'error': 'name and trigger_word are required'}), 400
    ds = svc.create_dataset(LOCAL_USER, name, trigger)
    return jsonify(svc.dataset_payload(LOCAL_USER, ds.id))


@bp.get('/dataset/variations')
def dataset_variations():
    return jsonify({'catalog': VARIATION_CATALOG,
                    'presets': {n: [e['id'] for e in select_preset(n)] for n in _PRESET_NAMES}})


@bp.get('/dataset/list')
def dataset_list():
    dss = svc.list_datasets(LOCAL_USER)
    return jsonify({'datasets': [
        {'id': d.id, 'name': d.name, 'trigger_word': d.trigger_word, 'ref_filename': d.ref_filename}
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
        with _vision_window():
            webp = svc.face_crop_to_square_webp(raw)  # auto head-crop
    except Exception as e:
        return _map_error(e)
    fn = f"{LOCAL_USER}_datasetref_{uuid.uuid4().hex[:8]}.webp"
    with open(os.path.join(svc._dataset_dir(dataset_id), fn), 'wb') as fh:
        fh.write(webp)
    ds.ref_filename = fn
    svc.db.session.commit()
    return jsonify({'ok': True, 'ref_filename': fn})


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
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    files = [f.read() for f in request.files.getlist('files') if f and f.filename]
    if not files:
        return jsonify({'error': 'no files'}), 400
    if len(files) > 20:
        return jsonify({'error': 'max 20 images per import'}), 400
    try:
        # batch (head-crop vision par image) : heartbeat de la fenêtre = ComfyUI arrêté
        # tout le batch ; le TTL n'est qu'un filet anti-crash.
        with _vision_window(flag_ttl=300):
            ids, failed = svc.import_images(LOCAL_USER, dataset_id, files, crop=True)  # auto head-crop
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'imported': len(ids), 'failed': failed})


@bp.post('/dataset/<int:dataset_id>/classify')
def dataset_classify(dataset_id):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        with _vision_window(flag_ttl=300):
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
        with _vision_window(flag_ttl=300):
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
