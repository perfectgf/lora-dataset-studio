"""Tests for the LoRA Test Studio service (checkpoint x strength sweep).

ComfyUI is never contacted: `queue_manager.add_job`/`_build_cell_workflow` are
monkeypatched for the enqueue-path tests, and the workflow-build test loads
the real copied workflow JSON but stops short of a network call."""
import pytest


def test_build_matrix_shape_and_validation(app):
    from app.services.lora_test_studio import build_matrix
    m = build_matrix(['a.safetensors', 'b.safetensors'], [0.8, 1.0], aspects=['9:16'])
    assert len(m) == 4 and all(len(t) == 6 for t in m)
    try:
        build_matrix(['a'], [99.0]); ok = False
    except Exception:
        ok = True
    assert ok


def test_wilson_ranking_prefers_confident_likes(app):
    from app.services.lora_test_studio import _wilson_lower_bound
    assert _wilson_lower_bound(9, 10) > _wilson_lower_bound(1, 1)


def test_cell_scores_and_best_cell(app):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'S', 's')
        for rating in (1, 1, -1):
            svc.db.session.add(LoraTestImage(dataset_id=ds.id, checkpoint='z image\\lora_s_000002000.safetensors',
                                             strength=1.0, status='done', rating=rating))
        svc.db.session.add(LoraTestImage(dataset_id=ds.id, checkpoint='z image\\lora_s_000002500.safetensors',
                                         strength=1.0, status='done', rating=-1))
        svc.db.session.commit()
        scores = lts.cell_scores(ds.id, family='zimage')
        assert scores[0]['checkpoint'].endswith('000002000.safetensors')
        best = lts.best_cell(ds.id, scores)
        assert best and best['strength'] == 1.0


def test_face_ranking_aggregates_by_checkpoint(app):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'F', 'f')
        for ck, s1, s2 in (('z image\\lora_f_000002000.safetensors', 0.6, 0.7),
                           ('z image\\lora_f_000002500.safetensors', 0.4, 0.5)):
            for s in (s1, s2):
                svc.db.session.add(LoraTestImage(dataset_id=ds.id, checkpoint=ck, strength=1.0,
                                                 status='done', face_score=s))
        svc.db.session.commit()
        rk = lts.face_ranking(ds.id, 'zimage')
        assert rk[0]['checkpoint'].endswith('000002000.safetensors') and rk[0]['n'] == 2


def test_create_run_commits_rows_before_enqueue(app, monkeypatch, tmp_path):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    from app import config
    with app.app_context():
        base = tmp_path / 'Comfy'
        lora_dir = base / 'models' / 'loras' / 'z image'
        lora_dir.mkdir(parents=True)
        ck = 'z image\\lora_s_000002000.safetensors'
        (lora_dir / 'lora_s_000002000.safetensors').touch()
        # create_run resolves a base Z-Image model BEFORE building any cell (verbatim
        # SRC guard: "aucun modèle Z-Image disponible") — a real unet/z image entry
        # is required for get_zimage_models() to return non-empty.
        unet_dir = base / 'models' / 'unet' / 'z image'
        unet_dir.mkdir(parents=True)
        (unet_dir / 'zmodel.safetensors').touch()
        config.save_config({'comfyui': {'base_dir': str(base)}})
        # get_zimage_models() has a 5-minute TTL cache (app.utils.comfyui); reset it so
        # this test's real directory is seen instead of another test's stale result.
        import app.utils.comfyui as comfyui_utils
        monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})
        ds = svc.create_dataset(LOCAL_USER, 'S2', 's')
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        # create_run calls queue_manager.add_job through lts._enqueue_cell, which
        # generates its own job_id and returns THAT (ignoring add_job's return value)
        # -- patch _enqueue_cell itself so the assertion below can pin the job_id.
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, **k: 'job-xyz')
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        out = lts.create_run(LOCAL_USER, ds.id, [ck], [1.0], prompt='p', count=1)
        rows = LoraTestImage.query.filter_by(dataset_id=ds.id).all()
        assert out['created'] == len(rows) >= 1
        assert all(r.job_id == 'job-xyz' and r.status == 'pending' for r in rows)


def test_create_run_with_resolution_tier_resolves_dims_via_lifted_resolution_module(app, monkeypatch, tmp_path):
    """Task 22 carry-forward: `_aspect_dims`'s lazy `from ..utils.resolution import
    compute_tier_dims` must resolve now that resolution.py is lifted — before the
    lift, any run requesting a resolution_tier raised ModuleNotFoundError."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    from app import config
    from app.utils.resolution import compute_tier_dims
    with app.app_context():
        base = tmp_path / 'Comfy'
        lora_dir = base / 'models' / 'loras' / 'z image'
        lora_dir.mkdir(parents=True)
        ck = 'z image\\lora_t_000002000.safetensors'
        (lora_dir / 'lora_t_000002000.safetensors').touch()
        unet_dir = base / 'models' / 'unet' / 'z image'
        unet_dir.mkdir(parents=True)
        (unet_dir / 'zmodel.safetensors').touch()
        config.save_config({'comfyui': {'base_dir': str(base)}})
        import app.utils.comfyui as comfyui_utils
        monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})
        ds = svc.create_dataset(LOCAL_USER, 'Tier', 't')
        captured = {}

        def fake_build(*a, **k):
            captured['width'] = k.get('width')
            captured['height'] = k.get('height')
            return {'1': {}}
        monkeypatch.setattr(lts, '_build_cell_workflow', fake_build)
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, **k: 'job-tier')
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        out = lts.create_run(LOCAL_USER, ds.id, [ck], [1.0], prompt='p', count=1,
                             resolution_tier='hq')
        rows = LoraTestImage.query.filter_by(dataset_id=ds.id).all()
        assert out['created'] == len(rows) == 1
        assert rows[0].resolution_tier == 'hq'
        # No aspect requested -> DEFAULT_ASPECT '9:16', named 'tall' in
        # _ASPECT_TO_TIER_RATIO (the only mapping create_run's _aspect_dims uses).
        expected = compute_tier_dims('tall', 'hq')
        assert (captured['width'], captured['height']) == expected


def test_create_comparison_run_commits_rows_before_enqueue(app, monkeypatch, tmp_path):
    """Same commit-before-enqueue anti-orphan guarantee as create_run, exercised
    on the multi-LoRA comparison path (its own row-commit + enqueue loop)."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    from app import config
    with app.app_context():
        base = tmp_path / 'Comfy'
        lora_dir = base / 'models' / 'loras' / 'z image'
        lora_dir.mkdir(parents=True)
        ck = 'z image\\lora_c_000002000.safetensors'
        (lora_dir / 'lora_c_000002000.safetensors').touch()
        unet_dir = base / 'models' / 'unet' / 'z image'
        unet_dir.mkdir(parents=True)
        (unet_dir / 'zmodel.safetensors').touch()
        config.save_config({'comfyui': {'base_dir': str(base)}})
        import app.utils.comfyui as comfyui_utils
        monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, **k: 'job-cmp')
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        out = lts.create_comparison_run(LOCAL_USER, [{'dataset_id': ds.id, 'checkpoint': ck}],
                                        [1.0], prompt='p', count=1)
        rows = LoraTestImage.query.filter_by(dataset_id=ds.id).all()
        assert out['created'] == len(rows) >= 1
        assert all(r.job_id == 'job-cmp' and r.status == 'pending' and r.run_id == out['run_id']
                  for r in rows)


def test_rate_image_accepts_only_valid_ratings(app):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'R', 'r')
        img = LoraTestImage(dataset_id=ds.id, checkpoint='z image\\lora_r_000001000.safetensors',
                            strength=1.0, status='done')
        svc.db.session.add(img)
        svc.db.session.commit()
        assert lts.rate_image(LOCAL_USER, img.id, 1) is True
        assert lts.rate_image(LOCAL_USER, img.id, -1) is True
        assert lts.rate_image(LOCAL_USER, img.id, 0) is True
        assert lts.rate_image(LOCAL_USER, img.id, 2) is False
        assert lts.rate_image(LOCAL_USER, img.id, 'like') is False


def test_studio_payload_on_fresh_dataset_is_well_formed_and_empty(app):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Empty', 'emptytrig')
        payload = lts.studio_payload(LOCAL_USER, ds.id)
        assert payload is not None
        assert payload['checkpoints'] == []
        assert payload['available_families'] == []
        assert payload['cells'] == []
        assert payload['scores'] == []
        assert payload['best_cell'] is None
        assert payload['pending'] == 0
        assert payload['resumable'] == 0
        assert payload['max_images'] == lts.MAX_TEST_IMAGES
        # SRC's 'saved_to_gallery'/history-hiding fields are dropped for this app.
        assert 'saved_to_gallery' not in json_dump_keys(payload)


def json_dump_keys(payload):
    """All dict keys anywhere in the payload (cells are a list of dicts)."""
    keys = set(payload.keys())
    for cell in payload.get('cells', []):
        keys |= set(cell.keys())
    return keys


def test_studio_payload_unknown_dataset_returns_none(app):
    from app.services import lora_test_studio as lts
    from app.config import LOCAL_USER
    with app.app_context():
        assert lts.studio_payload(LOCAL_USER, 999999) is None


def test_link_completed_test_image_failed_marks_cell_failed_without_move(app, tmp_path):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    from app import config
    with app.app_context():
        base = tmp_path / 'Comfy'
        (base / 'output').mkdir(parents=True)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        ds = svc.create_dataset(LOCAL_USER, 'Fail', 'failtrig')
        img = LoraTestImage(dataset_id=ds.id, checkpoint='z image\\lora_failtrig_000001000.safetensors',
                            strength=1.0, status='pending', job_id='job-fail')
        svc.db.session.add(img)
        svc.db.session.commit()

        out_file = base / 'output' / 'never.png'
        out_file.write_bytes(b'fake-png')

        lts.link_completed_test_image('job-fail', 'never.png', failed=True)

        refreshed = svc.db.session.get(LoraTestImage, img.id)
        assert refreshed.status == 'failed'
        assert refreshed.filename is None
        assert out_file.exists()  # never moved (failed path doesn't touch the file)


def test_link_completed_test_image_moves_file_into_dataset_dir(app, tmp_path):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    from app import config
    with app.app_context():
        base = tmp_path / 'Comfy'
        (base / 'output').mkdir(parents=True)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        ds = svc.create_dataset(LOCAL_USER, 'Done', 'donetrig')
        img = LoraTestImage(dataset_id=ds.id, checkpoint='z image\\lora_donetrig_000001000.safetensors',
                            strength=1.0, status='pending', job_id='job-done')
        svc.db.session.add(img)
        svc.db.session.commit()

        (base / 'output' / 'out.png').write_bytes(b'fake-png')

        lts.link_completed_test_image('job-done', 'out.png', failed=False)

        refreshed = svc.db.session.get(LoraTestImage, img.id)
        assert refreshed.status == 'done'
        assert refreshed.filename == 'out.png'
        assert not (base / 'output' / 'out.png').exists()
        import os
        assert os.path.exists(os.path.join(svc._dataset_dir(ds.id), 'out.png'))


def test_build_cell_workflow_zimage_loads_real_json_and_injects_lora(app):
    """Exercises the real copied ZImage_bigLove_ZT3_optimal.json workflow file
    (no ComfyUI contact): the checkpoint under test must show up as an injected
    LoraLoaderModelOnly node chained after the UNETLoader (node 1)."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        checkpoint = 'z image\\lora_zt_000001000.safetensors'
        workflow = lts._build_cell_workflow(
            user_id='local', checkpoint=checkpoint, strength=0.9, prompt='a prompt',
            seed=42, z_model=None, allowed_loras={checkpoint}, dataset_id=1,
            train_type='zimage', trigger_word='zt')
        assert workflow['1']['class_type'] == 'UNETLoader'
        lora_nodes = [n for n in workflow.values()
                     if isinstance(n, dict) and n.get('class_type') == 'LoraLoaderModelOnly']
        assert any(n['inputs']['lora_name'] == checkpoint for n in lora_nodes)
        # Model consumers (BasicScheduler node 7, CFGGuider node 9) were repointed
        # to the end of the injected LoRA chain, not left on the bare UNETLoader.
        assert workflow['7']['inputs']['model'] != ['1', 0]
        assert workflow['9']['inputs']['model'] != ['1', 0]


def test_run_owned_and_owned_test_image_are_single_user_no_ops(app):
    """Checklist item 2: `_run_owned` always True, `_owned_test_image` drops the
    user comparison (single-user app, no cross-user ownership DB)."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        assert lts._run_owned('some-other-user', 'nonexistent-run-id') is True
        ds = svc.create_dataset(LOCAL_USER, 'Owned', 'ownedtrig')
        img = LoraTestImage(dataset_id=ds.id, checkpoint='z image\\lora_ownedtrig_000001000.safetensors',
                            strength=1.0, status='done')
        svc.db.session.add(img)
        svc.db.session.commit()
        assert lts._owned_test_image('some-other-user', img.id) is not None
        assert lts._owned_test_image(LOCAL_USER, 999999) is None
