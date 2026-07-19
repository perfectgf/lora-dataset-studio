"""Face Dataset Maker API — create/list/get + generate (API-engine or Klein
fan-out) + import/classify/caption (Qwen3-VL) + curation + crop + export ZIP.

No login — single local user (`cfg.LOCAL_USER`). Vision-dependent routes borrow
the GPU-exclusive window (`gpu_exclusive_vision_window`) so a vision pass never
fights ComfyUI for the single GPU.
"""
import logging
import os
import tempfile
import uuid

from flask import Blueprint, request, jsonify, send_file, send_from_directory, current_app
from werkzeug.exceptions import RequestEntityTooLarge

from ..config import LOCAL_USER
from .. import config as cfg
from ..gpu_window import gpu_exclusive_vision_window
from ..services import face_dataset_service as svc
from ..services import dataset_activity
from ..services.dataset_storage import dataset_path, ensure_dataset_dir
from ..services import lora_test_studio as lts
from ..services import studio_grid_export as sge
from ..services.face_variations import (NSFW_VARIATION_CATALOG, VARIATION_CATALOG,
                                        is_nsfw_label, select_preset)
from ..utils.comfyui import KREA_ALLOWED_SAMPLERS, KREA_ALLOWED_SCHEDULERS, get_krea_loras
from ._common import (_map_error, _require_comfyui, _studio_arch_mismatch_response,
                      _studio_missing_response)

bp = Blueprint('datasets', __name__, url_prefix='/api')

_PRESET_NAMES = ('balanced_25', 'zimage_12', 'balanced_multiformat',
                 'face_focused', 'fullbody_focused', 'body_emphasis')


def _uploaded_archive_stream(file_storage):
    """Return a rewound, seekable upload stream after enforcing the file cap.

    Werkzeug already stores multipart files in a SpooledTemporaryFile.  Reading
    the FileStorage here would needlessly copy the complete archive back into RAM.
    """
    stream = file_storage.stream
    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(0)
    except (AttributeError, OSError, ValueError) as exc:
        raise ValueError('uploaded archive is not seekable') from exc
    if size > int(current_app.config['DATASET_ARCHIVE_MAX_UPLOAD_BYTES']):
        raise RequestEntityTooLarge()
    return stream


def _zip_download(write_zip, filename):
    """Build a seekable ZIP with bounded RAM, then let WSGI stream it.

    The spool must outlive this function: send_file consumes it only after the
    view returns.  Response.close owns normal cleanup; the except path handles a
    writer/send_file failure before a response exists.
    """
    spool = tempfile.SpooledTemporaryFile(
        max_size=int(current_app.config['DATASET_ARCHIVE_SPOOL_MEMORY_BYTES']),
        mode='w+b')
    try:
        write_zip(spool)
        size = spool.tell()
        spool.seek(0)
        response = send_file(spool, mimetype='application/zip', as_attachment=True,
                             download_name=filename)
        response.content_length = size
        response.call_on_close(spool.close)
        return response
    except Exception:
        spool.close()
        raise


@bp.post('/dataset/create')
def dataset_create():
    data = request.get_json(silent=True) or {}
    name, trigger = (data.get('name') or '').strip(), (data.get('trigger_word') or '').strip()
    # A style LoRA has no trigger — it tints every image once loaded, so there's
    # nothing to type in a prompt (create_dataset() auto-generates a unique
    # zsty_<id> placeholder for it). Character/concept datasets still need one:
    # it's the token the user types to summon them. Without this kind check the
    # blanket "trigger required" below rejected an empty-trigger style create
    # even though the UI (and the service layer) advertise it as optional.
    kind = svc.normalize_kind(data.get('kind'))
    if not name or (not trigger and kind != 'style'):
        return jsonify({'error': 'name and trigger_word are required'}), 400
    try:
        ds = svc.create_dataset(LOCAL_USER, name, trigger, kind=data.get('kind'),
                                concept_desc=data.get('concept_desc'),
                                train_type=data.get('train_type'),
                                fidelity=data.get('fidelity'),
                                prompt_suffix=data.get('prompt_suffix'),
                                prompt_suffixes=data.get('prompt_suffixes'))
    except ValueError as e:
        # concept dataset without a concept description -> 400 (not a 500)
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'id': ds.id})


@bp.post('/dataset/<int:dataset_id>/fidelity')
def dataset_set_fidelity(dataset_id):
    """Toggle face-only vs full-body fidelity. Affects FUTURE captions (re-caption
    to apply), the composition target and the import crop default."""
    data = request.get_json(silent=True) or {}
    ok = svc.set_fidelity(LOCAL_USER, dataset_id, data.get('fidelity'))
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/train-type')
def dataset_set_train_type(dataset_id):
    """Change a dataset's target model family (Z-Image/SDXL/Krea) after creation.
    Dataset metadata — NOT ai-toolkit-gated, so you can organize the menu even
    before training is configured. Keeps the TrainingPanel and the grouped menu in sync."""
    data = request.get_json(silent=True) or {}
    ok = svc.set_train_type(LOCAL_USER, dataset_id, data.get('train_type'))
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/settings')
def dataset_update_settings(dataset_id):
    """Edit name / trigger word / (concept) description / KIND after creation. Changing
    the trigger is safe (it's prepended at export — no re-caption). Changing a concept
    dataset's description resets the caption avoid-list cache; re-caption to apply it
    to existing captions (response flags concept_desc_changed for the UI hint).
    Changing the **kind** (character/concept/style) flips the caption strategy and the
    visible panels without deleting anything; response flags kind_changed + previous_kind
    so the UI can nudge a re-caption. It is refused (409) while a training run, a batch
    or an in-flight generation is live on the dataset.
    Also edits the creative-direction prompt suffixes (global text +
    {face,bust,body,back} map) — applied to FUTURE generations at wrap time;
    absent = untouched, '' / {} = cleared."""
    data = request.get_json(silent=True) or {}
    try:
        res = svc.update_dataset_settings(
            LOCAL_USER, dataset_id, name=data.get('name'),
            trigger_word=data.get('trigger_word'), concept_desc=data.get('concept_desc'),
            kind=data.get('kind'),
            prompt_suffix=data.get('prompt_suffix'),
            prompt_suffixes=data.get('prompt_suffixes'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        # Kind switch refused while work is in progress -> 409 (routes._common).
        return _map_error(e)
    return (jsonify(res), 200) if res else (jsonify({'error': 'not found'}), 404)


@bp.get('/dataset/variations')
def dataset_variations():
    return jsonify({'catalog': VARIATION_CATALOG,
                    # NSFW entries ship separately: the UI only shows them behind
                    # the 🔞 toggle, and ONLY for the local Klein engine.
                    'nsfw_catalog': NSFW_VARIATION_CATALOG,
                    'presets': {n: [e['id'] for e in select_preset(n)] for n in _PRESET_NAMES}})


@bp.get('/dataset/list')
def dataset_list():
    dss = svc.list_datasets(LOCAL_USER)
    # Library-page aggregates (counts + trained families) ride along so the
    # tiles can show real status without one request per dataset.
    stats = svc.dataset_list_stats(LOCAL_USER)
    empty = {'images_total': 0, 'images_kept': 0, 'images_captioned': 0,
             'trained_families': []}
    return jsonify({'datasets': [
        {'id': d.id, 'name': d.name, 'trigger_word': d.trigger_word, 'ref_filename': d.ref_filename,
         'kind': ((d.kind or '').lower() or 'character'),
         'train_type': (d.train_type or 'zimage'),
         **(stats.get(d.id) or empty)}
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
    # Garde-fou qualité : une référence basse résolution dégrade TOUTES les
    # variations générées (l'anchor identité part de là). On avertit, sans bloquer.
    low_res_warning = None
    try:
        from PIL import Image as PILImage
        import io as _io
        with PILImage.open(_io.BytesIO(raw)) as im0:
            if min(im0.size) < 768:
                low_res_warning = (
                    f'This reference is only {im0.size[0]}x{im0.size[1]} px — under 768 px the '
                    'generated variations will inherit the softness. A sharper photo gives a better LoRA.')
    except Exception:
        pass
    # Auto head-crop OPT-IN (form field crop='1') : par défaut on fait un carré
    # centré PIL pur — instantané, pas de passe vision, pas de pause ComfyUI —
    # et l'utilisateur ajuste avec ✂ Crop (l'éditeur lit l'original plein cadre).
    # Même UX que l'import de photos ; « Reset to auto » reste le chemin vision explicite.
    want_auto = request.form.get('crop', '0') == '1'
    try:
        if want_auto:
            with gpu_exclusive_vision_window():
                webp, head_detected = svc.face_crop_to_square_webp(
                    raw, pad=svc.REF_CROP_PAD, return_detected=True)
        else:
            webp, head_detected = svc.face_crop_to_square_webp(
                raw, pad=svc.REF_CROP_PAD, return_detected=True, use_vision=False)
    except Exception as e:
        return _map_error(e)
    dsdir = ensure_dataset_dir(dataset_id)
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
    if want_auto and not head_detected:
        # GUARD-RAIL: don't silently ship a body-centered crop when auto WAS asked.
        # Tell the user WHY it didn't run — the usual cause on a fresh install is the
        # Ollama vision model not being pulled — and how to recover (Setup + Crop).
        # (Manual mode: the centered crop is the intended behavior, no warning.)
        from .. import capabilities
        model_ready = capabilities.probe_ollama_model()['ok']
        resp['warning'] = (
            "Auto head-crop needs the Ollama vision model, which isn't ready yet — "
            'used a centered crop. Finish the Ollama step in Setup, then click Crop to re-center on the face.'
            if not model_ready else
            "Couldn't detect a face — used a centered crop. Use the Crop button to adjust it manually."
        )
    if low_res_warning:
        resp['warning'] = f"{resp['warning']} {low_res_warning}" if resp.get('warning') else low_res_warning
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


_KLEIN_ASSET_LABELS = {
    'klein_model': 'Klein model', 'klein_text_encoder': 'text encoder',
    'klein_vae': 'VAE', 'klein_lora': 'consistency LoRA',
}


def _autostart_klein_downloads(missing):
    """Kick off background downloads for the missing Klein assets. Returns
    (started, needs_token). Never raises — a download that can't start (already
    running, disk precondition) is reported, not fatal. An asset flagged `gated`
    in setup_installer._KLEIN_DOWNLOADS is only fired when an HF_TOKEN exists (it
    would 401 otherwise); public assets always fire. Every Klein download is public
    today — the KV UNET included — so needs_token stays False unless a future asset
    re-gates (the flag is data-driven, not hardcoded to a filename)."""
    from .. import setup_installer, config as cfg
    has_token = bool(cfg.secret('HF_TOKEN'))
    started, needs_token = [], False
    for action in missing:
        spec = setup_installer._KLEIN_DOWNLOADS.get(action, {})
        if spec.get('gated') and not has_token:
            needs_token = True
            continue  # gated: can't succeed without a token — instruct instead of firing
        try:
            setup_installer.start(action)
            started.append(action)
        except setup_installer.AlreadyRunning:
            started.append(action)  # already in flight still counts as "downloading"
        except Exception:
            pass  # Precondition (disk) — surfaced via the message, never a crash
    return started, needs_token


def _klein_missing_response(missing, missing_nodes=None):
    """Turn a Klein preflight miss into a (body, 409): auto-start any missing model
    downloads into the validated ComfyUI tree and/or list the custom nodes the
    target ComfyUI lacks, and tell the user to retry. When ComfyUI itself isn't a
    real install there's nowhere to place model files — return the 'configure
    ComfyUI first' message instead. Shared by the batch generate and the
    single-tile regenerate paths; `missing_nodes` = [{class_type, pack, url}] from
    klein_edit_helper.klein_missing_nodes (empty for the common models-only case)."""
    from .. import capabilities, config as cfg
    from ..services import klein_edit_helper as keh
    missing = missing or []
    missing_nodes = missing_nodes or []
    # The invalid-base short-circuit only applies when model files need a home to
    # download into; a node-only miss just needs a "install pack, restart" message.
    if missing and not capabilities.resolve_comfyui_base(cfg.get('comfyui.base_dir') or '')['valid']:
        return jsonify({'ok': False,
                        'error': 'Point the app at your ComfyUI install folder in '
                                 'Setup ▸ ComfyUI first, so the Klein models can be '
                                 'downloaded into it.'}), 409
    parts, started, needs_token = [], [], False
    if missing:
        started, needs_token = _autostart_klein_downloads(missing)
        names = ', '.join(_KLEIN_ASSET_LABELS.get(m, m) for m in missing)
        it = 'them' if len(missing) > 1 else 'it'
        parts.append(f"Klein needs {names}. I've started downloading {it} into your "
                     "ComfyUI folder — watch progress in Setup ▸ ComfyUI, then retry "
                     "generation.")
        if needs_token:
            parts.append("⚠ The Klein model is license-gated: accept the licence on its "
                         "Hugging Face page and paste an HF_TOKEN in Settings ▸ API keys, "
                         "otherwise it can't download.")
    if missing_nodes:
        parts.append(keh.format_missing_nodes_message(missing_nodes))
    return jsonify({'ok': False, 'error': ' '.join(parts),
                    'klein_missing': missing, 'downloading': started,
                    'needs_token': needs_token,
                    'klein_nodes_missing': missing_nodes}), 409


def _autostart_optional_klein():
    """Fire-and-forget: fetch any still-missing OPTIONAL Klein asset (the
    consistency LoRA) after a successful generate, so it's present next time.
    Never blocks or raises — required assets are already present at this point."""
    from ..services import klein_edit_helper as keh
    optional = [m for m in keh.klein_missing_assets() if m in keh.KLEIN_RECOMMENDED]
    if optional:
        _autostart_klein_downloads(optional)


@bp.post('/dataset/<int:dataset_id>/generate')
def dataset_generate(dataset_id):
    data = request.get_json(silent=True) or {}
    generator = data.get('generator') or 'klein'
    variations = data.get('variations') or []
    # Route-level fail-closed: NSFW variations never reach an API engine — they
    # exist only on the local Klein path (the service re-checks, defense in depth).
    if generator in svc.API_ENGINES and any(
            v.get('nsfw') or is_nsfw_label(v.get('label')) for v in variations):
        return jsonify({'ok': False,
                        'error': 'NSFW variations run on the local Klein engine only — '
                                 'switch the generator to Klein.'}), 400
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
            # Klein node preflight (once per request — /object_info is large, so
            # never per-tile): if the workflow needs a custom node this ComfyUI
            # lacks, answer one actionable 409 instead of a grid of tiles each
            # failing ComfyUI validation. Fail-open when /object_info is
            # unreachable. Combined with the model scan so a fresh install gets
            # ONE 409 covering both (and the model downloads start in parallel
            # with the user's node-pack install).
            from ..services import klein_edit_helper as keh
            missing_nodes = keh.klein_missing_nodes()
            if missing_nodes:
                return _klein_missing_response(keh.klein_missing_assets(), missing_nodes)
            ids = svc.generate_variations(LOCAL_USER, dataset_id,
                                          data.get('variations') or [], data.get('multiplier', 1),
                                          data.get('klein_model'),
                                          lora_strength=data.get('lora_strength'),
                                          # Optional generation-LoRA preset
                                          # (Idea by @waltm): a NAME resolved
                                          # from config — absent/'' = none.
                                          generation_lora_preset=data.get('generation_lora_preset'))
            _autostart_optional_klein()  # bg-fetch the consistency LoRA if it's absent
    except Exception as e:
        from ..services.klein_edit_helper import KleinModelsMissing
        if isinstance(e, KleinModelsMissing):  # a required Klein model isn't installed
            return _klein_missing_response(e.missing)
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
    # tout import en gros plan. Dataset CONCEPT ou STYLE : jamais de head-crop
    # (l'invariant n'est pas un visage ; un style vit autant dans les décors).
    # Sans crop → import BRUT (ratio préservé) → aucune passe vision → PAS de
    # fenêtre GPU exclusive (on ne stoppe pas ComfyUI pour rien).
    stats = {}
    want_crop = (not svc.is_conceptual(ds)) and request.form.get('crop', '1') != '0'
    if not want_crop:
        ids, failed = svc.import_images(LOCAL_USER, dataset_id, files, crop=False,
                                        dedupe=True, stats=stats)
        return jsonify({'ok': True, 'imported': len(ids), 'failed': failed,
                        'duplicates': stats.get('duplicates', 0),
                        'small': stats.get('small', 0)})
    try:
        # batch (head-crop vision par image) : heartbeat de la fenêtre = ComfyUI arrêté
        # tout le batch ; le TTL n'est qu'un filet anti-crash.
        with gpu_exclusive_vision_window(flag_ttl=600):
            ids, failed = svc.import_images(LOCAL_USER, dataset_id, files, crop=True,  # auto head-crop
                                            dedupe=True, stats=stats)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'imported': len(ids), 'failed': failed,
                    'duplicates': stats.get('duplicates', 0),
                    'small': stats.get('small', 0)})


@bp.post('/dataset/<int:dataset_id>/scrape-import')
def dataset_scrape_import(dataset_id):
    """Scrape DIRECT → dataset: downloads the SELECTED scanned images
    ({items:[{url,title}]}) straight into the dataset. Quality filters + dedup
    live in the service. Open to ALL dataset kinds: images import full-frame
    (aspect kept, no head-crop) and the user crops each tile manually — the old
    concept-only gate dated from when character imports forced a GPU head-crop."""
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    items = data.get('items') or []
    rescue_small = data.get('rescue_small', False)
    if not isinstance(rescue_small, bool):
        return jsonify({'error': 'rescue_small must be a boolean'}), 400
    if not isinstance(items, list) or not items:
        return jsonify({'error': 'no items'}), 400
    if len(items) > svc.SCRAPE_IMPORT_MAX:
        return jsonify({'error': f'max {svc.SCRAPE_IMPORT_MAX} images per import'}), 400
    try:
        res = svc.scrape_import_urls(LOCAL_USER, dataset_id, items,
                                     rescue_small=rescue_small)
    except Exception as e:
        from ..services.klein_edit_helper import KleinModelsMissing
        if isinstance(e, KleinModelsMissing):
            return _klein_missing_response(e.missing)
        return _map_error(e)
    return jsonify({'ok': True, **res})


@bp.post('/dataset/<int:dataset_id>/small-image-rescue/<int:candidate_id>/resolve')
def dataset_small_image_rescue_resolve(dataset_id, candidate_id):
    """Atomically choose the original, the Klein result, or neither."""
    data = request.get_json(silent=True) or {}
    try:
        result = svc.resolve_small_image_rescue(
            LOCAL_USER, dataset_id, candidate_id, data.get('choice'))
    except Exception as e:
        return _map_error(e)
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


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
    """Caption the kept images. Optional {image_ids:[...]} scopes the pass to a subset
    (the Identity-leak panel re-captions one leaking image, or all of them, in place) —
    a targeted call always OVERWRITES (those captions already exist), so it implies
    force. Omitted → the whole-dataset batch, gated by {force} as before. Same engine,
    mode and kind rules for both; serialized against training by the vision window."""
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    image_ids = data.get('image_ids')
    if image_ids is not None and not isinstance(image_ids, list):
        return jsonify({'error': "'image_ids' must be a list"}), 400
    force = bool(data.get('force')) or image_ids is not None
    mode = data.get('mode')  # 'prose' | 'booru' | None (None → auto selon train_type)
    try:
        with gpu_exclusive_vision_window(flag_ttl=1800):
            n = svc.caption_images(LOCAL_USER, dataset_id, force=force, mode=mode,
                                   image_ids=image_ids)
            # Did the long pass end because the user hit Stop? If so, skip the short
            # pass entirely — the point of stopping is to run NO more inference.
            stopped = dataset_activity.cancel_requested(dataset_id)
            # Dual captions on: regenerate the short variants for the same scope in the
            # SAME vision window (serialized against training). Best-effort — the long
            # captions are already saved, so a short-pass hiccup must not fail the call.
            if not stopped and svc.dual_captions_enabled(ds):
                try:
                    svc.derive_short_captions(LOCAL_USER, dataset_id, image_ids=image_ids,
                                              force=force, mode=mode)
                except Exception:
                    logging.getLogger(__name__).warning(
                        'short-caption derivation failed for dataset %s', dataset_id,
                        exc_info=True)
                # The short pass owns its own Stop-able indicator, so re-check.
                stopped = stopped or dataset_activity.cancel_requested(dataset_id)
    except Exception as e:
        return _map_error(e)
    finally:
        # Consume the flag once the whole operation has unwound so a stop can never
        # bleed into a later run (begin() also disarms defensively).
        dataset_activity.clear_cancel(dataset_id)
    return jsonify({'ok': True, 'captioned': n, 'stopped': stopped})


@bp.post('/dataset/<int:dataset_id>/caption/cancel')
def dataset_caption_cancel(dataset_id):
    """Ask an in-progress captioning batch to stop gracefully at the next image
    boundary. What's already captioned is kept; the rest stays uncaptioned. Never
    interrupts a running inference and never kills a process. Idempotent: a second
    Stop while the same batch still runs re-arms and returns 200. 404 when the dataset
    is unknown, 409 when no captioning batch is currently running."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    if not dataset_activity.request_cancel(dataset_id):
        return jsonify({'error': 'no captioning batch in progress'}), 409
    return jsonify({'ok': True, 'stopping': True})


@bp.get('/dataset/<int:dataset_id>/caption/options')
def dataset_caption_options_get(dataset_id):
    """Per-dataset caption method overrides {backend, ollama_model, instructions} for the
    ⚙️ Options popover. Empty values mean "follow the global default"."""
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, 'options': svc.caption_options(ds)})


@bp.post('/dataset/<int:dataset_id>/caption/options')
def dataset_caption_options_set(dataset_id):
    """Persist a caption-options patch (only the provided keys change). An invalid engine
    → 400. Applies to the next caption/re-caption run (including targeted + dual short)."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    try:
        options = svc.set_caption_options(LOCAL_USER, dataset_id, data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'options': options})


@bp.post('/dataset/<int:dataset_id>/analyze-faces')
def dataset_analyze_faces(dataset_id):
    # CPU (onnxruntime CPU-only) -> PAS de fenêtre GPU exclusive, ComfyUI non stoppé.
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        counts, scoring_error = svc.analyze_faces(LOCAL_USER, dataset_id)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, 'states': counts, 'analyzed': sum(counts.values()),
                    'scoring_error': scoring_error})


@bp.post('/dataset/<int:dataset_id>/watermarks/detect')
def dataset_watermarks_detect(dataset_id):
    """Scan kept images for overlaid watermarks (Qwen3-VL) — GPU-exclusive vision
    window like classify/caption. Persists watermark_state/bbox; deletes nothing.
    Skips images already dismissed as false positives unless {include_dismissed:true}."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    include_dismissed = bool(data.get('include_dismissed'))
    try:
        with gpu_exclusive_vision_window(flag_ttl=1800):
            counts = svc.detect_watermarks(LOCAL_USER, dataset_id,
                                           include_dismissed=include_dismissed)
    except Exception as e:
        return _map_error(e)
    return jsonify({'ok': True, **counts})


def _klein_clean_preflight():
    """None if a Klein-inpaint clean can run, else the (body, status) to return:
    503 when the GPU is held by training/vision (the job would just wait), a 409 that
    lists + auto-downloads any missing Klein model or names a missing custom node (same
    shape as the generate path). Mirrors lora_test_studio.gpu_busy_reason + the Klein
    generate preflight so the batch never enqueues a doomed round-trip."""
    from ..job_queue import queue_manager
    from ..services import klein_edit_helper as keh
    if queue_manager._get_system_state('training_in_progress', False):
        return jsonify({'error': 'GPU busy', 'detail': 'LoRA training in progress'}), 503
    if queue_manager._get_system_state('vision_in_progress', False):
        return jsonify({'error': 'GPU busy', 'detail': 'a vision pass is running'}), 503
    missing = keh.klein_missing_assets()
    missing_nodes = keh.klein_missing_nodes()
    if any(a in missing for a in keh.KLEIN_REQUIRED) or missing_nodes:
        return _klein_missing_response(missing, missing_nodes)
    return None


@bp.post('/dataset/<int:dataset_id>/watermarks/clean')
def dataset_watermarks_clean(dataset_id):
    """Apply crop/inpaint/review routing to the 'detected' images. Crop uses PIL. The
    inpaint engine follows {method:'auto'|'lama'|'klein'} (default 'auto'): LaMa follows
    Settings > Captioning & quality (Auto/GPU/CPU) and pauses ComfyUI through the
    exclusive vision window on a GPU pass; Klein does masked crop-and-stitch inpaint
    through the serialized ComfyUI queue (no vision window — that would deadlock the
    worker). Returns counts + the inpaint error. Optional {image_ids:[...]} scopes the
    pass to a subset (the review lightbox cleans one image at a time); omitted → every
    detected image (the bulk 🧽 Clean button). Optional {allow_crop:bool} overrides the
    persisted crop preference: omitted → Settings' watermark.allow_crop; True/False →
    force crop / force inpaint (the lightbox's per-image crop-vs-inpaint choice)."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    image_ids = data.get('image_ids')
    if image_ids is not None and not isinstance(image_ids, list):
        return jsonify({'error': "'image_ids' must be a list"}), 400
    method = (data.get('method') or 'auto')
    if method not in ('auto', 'lama', 'klein'):
        return jsonify({'error': "'method' must be 'auto', 'lama' or 'klein'"}), 400
    # allow_crop is optional: omitted -> clean_watermarks resolves the persisted
    # watermark.allow_crop preference (so the batch button follows Settings); a bool
    # forces crop (True) or inpaint (False) — the review lightbox's per-image choice.
    # Forwarded only when present so a plain {method:...} call keeps its exact old shape.
    allow_crop = data.get('allow_crop')
    if allow_crop is not None and not isinstance(allow_crop, bool):
        return jsonify({'error': "'allow_crop' must be a boolean"}), 400
    crop_kw = {} if allow_crop is None else {'allow_crop': allow_crop}
    try:
        if method == 'klein':
            resp = _klein_clean_preflight()
            if resp is not None:
                return resp
            counts, error = svc.clean_watermarks(
                LOCAL_USER, dataset_id, image_ids=image_ids, method='klein', **crop_kw)
        else:
            from contextlib import nullcontext
            from ..services import watermark_lama
            device = watermark_lama.resolve_device()
            window = gpu_exclusive_vision_window(flag_ttl=1800) if device == 'cuda' else nullcontext()
            with window:
                counts, error = svc.clean_watermarks(
                    LOCAL_USER, dataset_id, image_ids=image_ids, device=device,
                    method=method, **crop_kw)
    except Exception as e:
        from ..services.klein_edit_helper import KleinModelsMissing
        if isinstance(e, KleinModelsMissing):
            return _klein_missing_response(e.missing)
        return _map_error(e)
    return jsonify({'ok': True, 'error': error, **counts})


@bp.post('/dataset/<int:dataset_id>/watermarks/dismiss')
def dataset_watermarks_dismiss(dataset_id):
    """Mark flagged images as NOT a watermark (a false positive ruled out in the review
    lightbox). Body: {image_ids:[...]}. Dismissed images drop the 🚩 badge and are
    skipped by future detect passes. CPU only, no GPU window."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    image_ids = data.get('image_ids')
    if not isinstance(image_ids, list) or not image_ids:
        return jsonify({'error': "'image_ids' must be a non-empty list"}), 400
    n = svc.dismiss_watermarks(LOCAL_USER, dataset_id, image_ids)
    return jsonify({'ok': True, 'dismissed': n})


@bp.put('/dataset/<int:dataset_id>/image/<int:image_id>/watermark-regions')
def dataset_image_watermark_regions(dataset_id, image_id):
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or 'regions' not in data:
        return jsonify({'error': 'regions is required'}), 400
    try:
        result = svc.set_watermark_regions(
            LOCAL_USER, dataset_id, image_id, data['regions'],
        )
    except (ValueError, RuntimeError) as e:
        return _map_error(e)
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


@bp.post('/dataset/<int:dataset_id>/image/<int:image_id>/watermark-restore')
def dataset_image_watermark_restore(dataset_id, image_id):
    """Undo a watermark Clean on ONE image: restore the preserved original in place and
    re-flag it 'detected' so it can be re-cleaned (e.g. with the other engine). 404 when
    the image isn't found/owned OR no original was preserved (it was never cleaned)."""
    try:
        result = svc.restore_watermark_original(LOCAL_USER, dataset_id, image_id)
    except FileNotFoundError:
        return jsonify({'error': 'no original to restore'}), 404
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


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
    try:
        ok = svc.delete_dataset(LOCAL_USER, dataset_id)
    except Exception as e:
        # A file still open in another process (antivirus scan of a just-cleaned
        # image) surfaces as a clear 409 toast instead of a bare 500.
        return _map_error(e)
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/<int:dataset_id>/cancel')
def dataset_cancel(dataset_id):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    n = svc.cancel_pending(LOCAL_USER, dataset_id)
    return jsonify({'ok': True, 'cancelled': n})


@bp.post('/dataset/image/<int:image_id>/delete')
def dataset_image_delete(image_id):
    try:
        ok = svc.delete_image(LOCAL_USER, image_id)
    except Exception as e:
        return _map_error(e)
    return (jsonify({'ok': True}), 200) if ok else (jsonify({'error': 'not found'}), 404)


@bp.post('/dataset/image/<int:image_id>/improve')
def dataset_image_improve(image_id):
    """Create a regular Klein-upscaled candidate without touching the source."""
    try:
        result = svc.improve_existing_image(LOCAL_USER, image_id)
    except Exception as e:
        from ..services.klein_edit_helper import KleinModelsMissing
        if isinstance(e, svc.KleinNodesMissing):
            return _klein_missing_response(e.missing, e.missing_nodes)
        if isinstance(e, KleinModelsMissing):
            return _klein_missing_response(e.missing)
        return _map_error(e)
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


@bp.post('/dataset/image/<int:image_id>/regenerate')
def dataset_image_regenerate(image_id):
    data = request.get_json(silent=True) or {}
    # Optional edited core prompt from the tile's ✏️ bubble — None keeps the
    # current behaviour (recover from the row / label); the identity guard is
    # re-applied on top either way (see regenerate_image).
    edited_prompt = (data.get('prompt') or '').strip() or None
    # Optional engine override: the generator currently selected in the
    # workspace (absent = legacy behaviour, reuse the row's origin engine).
    engine = (data.get('engine') or '').strip() or None
    klein_model = (data.get('klein_model') or '').strip() or None
    try:
        from flask import current_app
        # Klein node preflight (skip when the user explicitly picked an API engine,
        # which doesn't touch ComfyUI): surface a missing custom node as one 409
        # instead of a silent failed re-roll. Fail-open if /object_info is down;
        # combined with the model scan (same rationale as the batch generate).
        if engine not in svc.API_ENGINES:
            from ..services import klein_edit_helper as keh
            missing_nodes = keh.klein_missing_nodes()
            if missing_nodes:
                return _klein_missing_response(keh.klein_missing_assets(), missing_nodes)
        job_id = svc.regenerate_image(LOCAL_USER, image_id,
                                      lora_strength=data.get('lora_strength'),
                                      prompt=edited_prompt,
                                      engine=engine, klein_model=klein_model,
                                      generation_lora_preset=data.get('generation_lora_preset'),
                                      app=current_app._get_current_object())
    except Exception as e:
        from ..services.klein_edit_helper import KleinModelsMissing
        if isinstance(e, KleinModelsMissing):
            return _klein_missing_response(e.missing)  # auto-download, tell them to retry
        return _map_error(e)
    if job_id is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, 'job_id': job_id})


@bp.post('/dataset/<int:dataset_id>/import-zip')
def dataset_import_zip(dataset_id):
    """Merge an EXISTING training dataset (ZIP of images + kohya-style same-stem
    .txt captions) into this dataset. Aspect preserved, dHash dedupe, captions
    attached to the rows."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'error': 'no file'}), 400
    stats = {}
    try:
        ids, failed = svc.import_dataset_zip(
            LOCAL_USER, dataset_id, _uploaded_archive_stream(f), stats=stats)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'imported': len(ids), 'failed': failed,
                    'duplicates': stats.get('duplicates', 0),
                    'captions': stats.get('captions', 0),
                    'small': stats.get('small', 0)})


@bp.post('/dataset/<int:dataset_id>/import-folder')
def dataset_import_folder(dataset_id):
    """Merge an EXISTING training dataset from a FOLDER on this machine's disk
    (images + kohya-style same-stem .txt captions, any depth) — same merge as
    import-zip without zipping first. Body: {path}. Local single-user app: an
    arbitrary path is fine (it's the user's own disk); a missing folder is a
    clear 400, non-image files are ignored."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(silent=True) or {}
    path = (data.get('path') or '').strip()
    if not path:
        return jsonify({'error': 'path is required'}), 400
    stats = {}
    try:
        ids, failed = svc.import_dataset_folder(LOCAL_USER, dataset_id, path, stats=stats)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'imported': len(ids), 'failed': failed,
                    'duplicates': stats.get('duplicates', 0),
                    'captions': stats.get('captions', 0),
                    'small': stats.get('small', 0)})


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


@bp.post('/dataset/<int:dataset_id>/captions/write-files')
def dataset_captions_write_files(dataset_id):
    """Write kohya-style same-stem .txt captions NEXT TO the kept images in the
    dataset folder (same text as the export ZIP: trigger prepended) — for
    external tools that read the folder directly, no ZIP download needed.
    Overwrites existing .txt files (resync)."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    try:
        res = svc.write_caption_files(LOCAL_USER, dataset_id)
    except Exception as e:
        return _map_error(e)
    return jsonify(res)


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
    try:
        n = svc.batch_image_action(LOCAL_USER, dataset_id, ids, action)
    except Exception as e:
        return _map_error(e)
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
    # caption_short only touched when the key is present (the expanded editor sends it);
    # the inline long-caption textarea omits it so it can't wipe an existing short.
    kwargs = {'short': data['caption_short']} if 'caption_short' in data else {}
    ok = svc.set_image_caption(LOCAL_USER, image_id, data.get('caption', ''), **kwargs)
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


@bp.post('/dataset/image/<int:image_id>/mirror')
def dataset_image_mirror(image_id):
    """Permanently flip one owned dataset image horizontally in place."""
    try:
        result = svc.mirror_image(LOCAL_USER, image_id)
    except Exception as e:
        return _map_error(e)
    if result is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True, **result})


@bp.get('/dataset/<int:dataset_id>/export')
def dataset_export(dataset_id):
    try:
        return _zip_download(
            lambda output: svc.write_export_zip(LOCAL_USER, dataset_id, output),
            f'dataset_{dataset_id}.zip')
    except ValueError as e:
        return _map_error(e)


@bp.get('/dataset/<int:dataset_id>/backup')
def dataset_backup(dataset_id):
    """Full portable backup (manifest + settings + ALL images with status/captions/
    scores) — distinct from /export, the training-format ZIP."""
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in (ds.name if ds else str(dataset_id)))
    try:
        return _zip_download(
            lambda output: svc.write_backup_zip(LOCAL_USER, dataset_id, output),
            f'lds_backup_{safe}.zip')
    except ValueError as e:
        return _map_error(e)


@bp.post('/dataset/backup/import')
def dataset_backup_import():
    """Restore a backup zip as a NEW dataset."""
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'error': 'no file'}), 400
    try:
        ds = svc.import_backup_zip(LOCAL_USER, _uploaded_archive_stream(f))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'id': ds.id, 'name': ds.name})


# ---------------------------------------------------------------------------
# Publish to Hugging Face (export a dataset repo to the Hub — export only)
# ---------------------------------------------------------------------------

@bp.get('/dataset/<int:dataset_id>/publish-hf/whoami')
def dataset_publish_hf_whoami(dataset_id):
    """Prefill helper for the Publish modal: the token owner's username and the
    suggested `<username>/<slug>` repo id. Best-effort — a missing/invalid token
    just yields username=null (the modal degrades to a free-text field)."""
    from ..services import hf_publish
    ds = svc.get_dataset(LOCAL_USER, dataset_id)
    if not ds:
        return jsonify({'error': 'not found'}), 404
    username = hf_publish.hf_namespace(cfg.secret('HF_TOKEN'))
    return jsonify({'ok': True, 'username': username,
                    'default_repo_id': hf_publish.default_repo_id(username, ds),
                    'licenses': list(hf_publish.LICENSE_CHOICES)})


@bp.post('/dataset/<int:dataset_id>/publish-hf')
def dataset_publish_hf(dataset_id):
    """Kick off the background upload of this dataset to the HF Hub. Server-side
    guards: HF_TOKEN must exist, `consent` MUST be true (not merely a UI checkbox),
    dataset must exist. The slow upload runs in a daemon thread; the UI polls the
    status route. Structured preflight errors (read-only token, repo exists) also
    surface via the status poll."""
    from ..services import hf_publish
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    token = cfg.secret('HF_TOKEN')
    if not token:
        return jsonify({'error': 'no Hugging Face token configured — paste an '
                        'HF_TOKEN in Settings ▸ API keys'}), 400
    data = request.get_json(silent=True) or {}
    if data.get('consent') is not True:
        return jsonify({'error': 'you must confirm you have the right to share these '
                        'images and the consent of any identifiable person'}), 400
    repo_id = (data.get('repo_id') or '').strip()
    if not repo_id:
        return jsonify({'error': 'repo id is required'}), 400
    license = (data.get('license') or '').strip().lower()
    if license not in hf_publish.LICENSE_CHOICES:
        return jsonify({'error': f'unsupported license: {license or "(empty)"}'}), 400
    out = hf_publish.start_publish(
        current_app._get_current_object(), dataset_id, repo_id,
        private=bool(data.get('private', True)), nfaa=bool(data.get('nfaa', True)),
        license=license, include_ref=bool(data.get('include_ref', False)), token=token)
    return jsonify({'ok': True, **out})


@bp.get('/dataset/<int:dataset_id>/publish-hf/status')
def dataset_publish_hf_status(dataset_id):
    """Poll: {state: idle|running|done|error, repo_url, error, error_code, count}."""
    from ..services import hf_publish
    return jsonify(hf_publish.publish_status(dataset_id))


@bp.get('/dataset/<int:dataset_id>/img/<path:filename>')
def dataset_image_file(dataset_id, filename):
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(dataset_path(dataset_id), filename)


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
        from ..services.lora_test_studio import StudioArchMismatch, StudioAssetsMissing
        if isinstance(e, StudioArchMismatch):   # wrong-arch checkpoint → actionable 409
            return _studio_arch_mismatch_response(e)
        if isinstance(e, StudioAssetsMissing):  # models/nodes absent → actionable 409
            return _studio_missing_response(e)
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


@bp.post('/dataset/<int:dataset_id>/lora-test/export-grid')
def lora_test_export_grid(dataset_id):
    """Compose ONE run's checkpoint × strength grid into a single shareable image
    (labels + FORMAT headers baked in) and stream it as a browser download — the
    classic XY plot, clean, ready for Civitai/Reddit.

    DB-only + PIL (no ComfyUI, so ungated). Body: {family, run_seed, prompt (run
    identity — null = most recent run), aspect ('16:9'… or 'all'), include_prompt
    (default false — prompts can be personal/NSFW), cell_size (512|768), format
    ('jpeg'|'png'), footer (bool)}. 404 unknown dataset ; 409 empty/unknown run."""
    if not svc.get_dataset(LOCAL_USER, dataset_id):
        return jsonify({'error': 'not found'}), 404
    d = request.get_json(silent=True) or {}
    try:
        data, mime, meta = sge.export_grid(
            LOCAL_USER, dataset_id,
            family=d.get('family'), run_seed=d.get('run_seed'), prompt=d.get('prompt'),
            aspect=d.get('aspect'), include_prompt=bool(d.get('include_prompt')),
            cell_size=d.get('cell_size'), fmt=d.get('format'),
            footer=d.get('footer', True))
    except sge.GridExportEmpty as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    resp = current_app.response_class(data, mimetype=mime)
    resp.headers['Content-Disposition'] = f'attachment; filename="{meta["download_name"]}"'
    resp.headers['Content-Length'] = str(len(data))
    # Surface the cap decision so the UI can note « downscaled to fit » after the fact.
    resp.headers['X-Grid-Downscaled'] = '1' if meta['downscaled'] else '0'
    resp.headers['X-Grid-Size'] = f'{meta["width"]}x{meta["height"]}'
    return resp


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
