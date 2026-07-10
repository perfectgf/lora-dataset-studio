"""Tests for the Klein dataset generation path (klein_edit_helper + comfyui_service
+ face_dataset_service wiring). ComfyUI itself is never contacted: the queue's
`add_job` is patched so these tests exercise the workflow-build + job-row
bookkeeping without a real ComfyUI server."""
import io
import json
import os

from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _configure_comfy_dirs(tmp_path, cfg, install_models=True):
    """Real tmp dirs for a ComfyUI tree so the helper's file ops run against a
    real filesystem instead of unconfigured Nones. With install_models=True the
    three graph-critical Klein files are placed in their canonical folders (the
    layout the Setup downloads use) so the model preflight passes; main.py makes
    resolve_comfyui_base() treat the folder as a real ComfyUI install."""
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'models' / 'loras' / 'klein').mkdir(parents=True)
    (base / 'main.py').write_text('# fake ComfyUI', encoding='utf-8')
    if install_models:
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        (base / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors').write_bytes(b'unet')
        (base / 'models' / 'vae').mkdir(parents=True)
        (base / 'models' / 'vae' / 'flux2-vae.safetensors').write_bytes(b'vae')
        (base / 'models' / 'text_encoders').mkdir(parents=True)
        (base / 'models' / 'text_encoders' / 'qwen_3_8b_fp8mixed.safetensors').write_bytes(b'te')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _make_dataset_with_ref(svc, LOCAL_USER, name, trigger):
    ds = svc.create_dataset(LOCAL_USER, name, trigger)
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    svc.db.session.commit()
    return ds


def test_generate_variations_creates_pending_rows_with_job_id(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.services.face_variations import select_preset
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager

    with app.app_context():
        _configure_comfy_dirs(tmp_path, cfg)
        ds = _make_dataset_with_ref(svc, LOCAL_USER, 'Lola', 'lola')

        calls = []

        def fake_add_job(**kwargs):
            calls.append(kwargs)
            return kwargs['job_id']

        monkeypatch.setattr(queue_manager, 'add_job', fake_add_job)

        variations = select_preset('zimage_12')[:2]
        ids = svc.generate_variations(LOCAL_USER, ds.id, variations, 1, klein_model='k.safetensors')

        assert len(ids) == 2
        assert len(calls) == 2
        for call in calls:
            assert call['metadata']['model_name'] == 'klein_edit_dataset'
        rows = FaceDatasetImage.query.filter_by(dataset_id=ds.id).all()
        assert len(rows) == 2
        assert all(r.status == 'pending' and r.job_id for r in rows)


def test_link_completed_dataset_image_moves_file(app, tmp_path):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER

    with app.app_context():
        base = _configure_comfy_dirs(tmp_path, cfg)
        ds = svc.create_dataset(LOCAL_USER, 'Kai', 'kai')
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending',
                               job_id='job-xyz', klein_model='k.safetensors')
        svc.db.session.add(img)
        svc.db.session.commit()

        out_dir = base / 'output'
        (out_dir / 'out.png').write_bytes(_png())

        svc.link_completed_dataset_image('job-xyz', 'out.png')

        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.status == 'pending'  # unchanged; only filename/failed are set here
        assert refreshed.filename == 'out.png'
        assert not (out_dir / 'out.png').exists()
        assert os.path.exists(os.path.join(svc._dataset_dir(ds.id), 'out.png'))


def test_link_completed_dataset_image_failed_marks_row_no_move(app, tmp_path):
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER

    with app.app_context():
        base = _configure_comfy_dirs(tmp_path, cfg)
        ds = svc.create_dataset(LOCAL_USER, 'Fail', 'fail')
        img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending',
                               job_id='job-fail', klein_model='k.safetensors')
        svc.db.session.add(img)
        svc.db.session.commit()

        out_dir = base / 'output'
        (out_dir / 'never.png').write_bytes(_png())

        svc.link_completed_dataset_image('job-fail', 'never.png', failed=True)

        refreshed = svc.db.session.get(FaceDatasetImage, img.id)
        assert refreshed.status == 'failed'
        assert refreshed.filename is None
        assert (out_dir / 'never.png').exists()  # never moved


def test_generate_klein_unconfigured_comfyui_raises_models_missing(app):
    """No comfyui.base_dir configured -> the model preflight finds NONE of the
    Klein files on disk and raises KleinModelsMissing (which the route turns into
    an actionable 'configure ComfyUI / downloading' 409, not a 500 or a dataset
    full of failed tiles)."""
    import pytest
    from app.services import face_dataset_service as svc
    from app.services.klein_edit_helper import KleinModelsMissing
    from app.services.face_variations import select_preset
    from app.config import LOCAL_USER

    with app.app_context():
        ds = _make_dataset_with_ref(svc, LOCAL_USER, 'NoComfy', 'nocomfy')
        with pytest.raises(KleinModelsMissing):
            svc.generate_variations(LOCAL_USER, ds.id, select_preset('zimage_12')[:1], 1,
                                    klein_model='k.safetensors')


def test_save_prefix_is_unique_per_job(app, tmp_path, monkeypatch):
    """SaveImage numbers files from what's in ComfyUI's output folder, and the
    app moves results out right after completion — a SHARED prefix made the
    counter re-issue the same name (live repro: 4 different seeds all saved as
    local_DatasetFace_00002_.png) so every tile displayed the same image. The
    prefix must therefore be unique per job."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager
    with app.app_context():
        _configure_comfy_dirs(tmp_path, cfg)
        src = tmp_path / 'ref.png'; src.write_bytes(_png())
        prefixes = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (prefixes.append(kw['workflow_data']['9']['inputs']['filename_prefix']),
                                          kw['job_id'])[1])
        for _ in range(2):
            keh.enqueue_klein_edit(user_id='local', source_filename='ref.png',
                                   edit_prompt='p', source_path=str(src))
        assert len(prefixes) == 2
        assert prefixes[0] != prefixes[1]            # unique per job
        assert all(p.startswith('local_DatasetFace_') for p in prefixes)


def test_link_never_overwrites_existing_dataset_file(app, tmp_path):
    """Residual name collision (two jobs reporting the same output filename):
    the second link must store under a RENAMED file, not overwrite the first
    tile's image."""
    from app import config as cfg
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER

    with app.app_context():
        base = _configure_comfy_dirs(tmp_path, cfg)
        ds = svc.create_dataset(LOCAL_USER, 'Col', 'col')
        rows = []
        for jid in ('job-a', 'job-b'):
            img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending',
                                   job_id=jid, klein_model='k.safetensors')
            svc.db.session.add(img)
            rows.append(img)
        svc.db.session.commit()

        out_dir = base / 'output'
        (out_dir / 'same.png').write_bytes(b'FIRST')
        svc.link_completed_dataset_image('job-a', 'same.png')
        (out_dir / 'same.png').write_bytes(b'SECOND')      # ComfyUI re-issued the name
        svc.link_completed_dataset_image('job-b', 'same.png')

        a = svc.db.session.get(FaceDatasetImage, rows[0].id)
        b = svc.db.session.get(FaceDatasetImage, rows[1].id)
        assert a.filename == 'same.png'
        assert b.filename != 'same.png'                    # renamed, not overwritten
        da = os.path.join(svc._dataset_dir(ds.id), a.filename)
        db_ = os.path.join(svc._dataset_dir(ds.id), b.filename)
        with open(da, 'rb') as fh:
            assert fh.read() == b'FIRST'                   # first tile intact
        with open(db_, 'rb') as fh:
            assert fh.read() == b'SECOND'


def test_generate_klein_bad_dataset_id_returns_400_not_409(client):
    """Task 8 flagged: generate_variations imports klein_edit_helper BEFORE
    validating dataset_id, so pre-lift (ImportError -> RuntimeError -> 409) a
    bad dataset_id on the Klein path surfaced as 409 instead of the dataset's
    normal 'not found' status. Now that klein_edit_helper exists, the import
    always succeeds and get_dataset()'s ValueError('dataset not found') is
    what the route sees -> _map_error's ValueError branch -> 400. Verified
    self-healed."""
    resp = client.post('/api/dataset/999999/generate', json={
        'generator': 'klein',
        'variations': [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
        'multiplier': 1,
        'klein_model': 'some_model',
    })
    assert resp.status_code == 400


def test_generate_klein_without_comfyui_still_returns_409(client):
    """Route-level regression check (companion to test_dataset_routes.py's
    equivalent, now exercising the real klein_edit_helper module instead of
    an ImportError): a valid dataset + ref but no comfyui.base_dir configured
    must still surface a clean 409, not a 500."""
    ds_resp = client.post('/api/dataset/create', json={'name': 'Kai2', 'trigger_word': 'kai2'})
    ds_id = ds_resp.get_json()['id']
    data = {'file': (io.BytesIO(_png()), 'ref.png')}
    client.post(f'/api/dataset/{ds_id}/ref', data=data, content_type='multipart/form-data')
    resp = client.post(f'/api/dataset/{ds_id}/generate', json={
        'generator': 'klein',
        'variations': [{'label': 'x', 'framing': 'face', 'prompt': 'p'}],
        'multiplier': 1,
        'klein_model': 'some_model',
    })
    assert resp.status_code == 409


def test_workflow_json_loads_and_consistency_lora_from_config(app, tmp_path, monkeypatch):
    """improve skin.json must be valid JSON with the required nodes, and the
    injected consistency-LoRA settings must come from config (klein.*), not
    the old hardcoded SRC constants."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.job_queue import queue_manager

    with app.app_context():
        # Sanity: the workflow file itself is valid JSON with the nodes the
        # helper depends on.
        with open(keh.WORKFLOW_IMPROVE_SKIN_PATH, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        for node in keh._REQUIRED_NODES:
            assert node in raw
        assert '139' in raw  # existing LoRA node the consistency LoRA chains before

        base = _configure_comfy_dirs(tmp_path, cfg)
        lora_dir = base / 'models' / 'loras'
        patched_lora = 'klein/test-consistency.safetensors'
        (lora_dir / 'klein' / 'test-consistency.safetensors').write_bytes(b'fake-lora')
        cfg.save_config({'klein': {'consistency_lora': patched_lora, 'consistency_strength': 0.42}})

        source_dir = base / 'source'
        source_dir.mkdir()
        source_path = source_dir / 'ref.png'
        source_path.write_bytes(_png())

        captured = {}

        def fake_add_job(**kwargs):
            captured.update(kwargs)
            return kwargs['job_id']

        monkeypatch.setattr(queue_manager, 'add_job', fake_add_job)

        keh.enqueue_klein_edit(user_id='local', source_filename='ref.png',
                               edit_prompt='a prompt', klein_model='k.safetensors',
                               source_path=str(source_path))

        assert captured, 'add_job should have been called'
        workflow = captured['workflow_data']
        assert 'ds_consistency_lora' in workflow
        lora_node = workflow['ds_consistency_lora']
        # Verify consistency_lora path separators are normalized to the OS (backslash on Windows)
        expected_lora_name = patched_lora.replace('/', os.sep)
        assert lora_node['inputs']['lora_name'] == expected_lora_name
        assert lora_node['inputs']['strength_model'] == 0.42
        # The loaders now reference the ACTUAL installed files, not the workflow's
        # hardcoded developer filenames.
        assert workflow['114']['inputs']['unet_name'] == os.path.join('klein', 'flux-2-klein-9b-fp8.safetensors')
        assert workflow['10']['inputs']['vae_name'] == 'flux2-vae.safetensors'
        assert workflow['90']['inputs']['clip_name'] == 'qwen_3_8b_fp8mixed.safetensors'
        # The base 'realistic' LoRA (node 139) isn't installed here -> it's bypassed
        # and removed, and its consumer (102.model) is rewired to the consistency LoRA.
        assert '139' not in workflow
        assert workflow['102']['inputs']['model'] == ['ds_consistency_lora', 0]
