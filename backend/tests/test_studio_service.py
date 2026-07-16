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


def test_build_matrix_accepts_extended_strengths_up_to_4(app):
    """Progressive-disclosure « + » exposes strengths above 2.0 (over-cook range);
    the sweep validation now accepts up to 4.0 and rejects anything beyond."""
    from app.services.lora_test_studio import build_matrix
    m = build_matrix(['a.safetensors'], [2.5, 3.5, 4.0], aspects=['9:16'])
    assert sorted({t[1] for t in m}) == [2.5, 3.5, 4.0]     # carried, not clamped to 2.0
    for bad in (4.01, 4.5, 10.0):
        with pytest.raises(ValueError, match=r'out of range \[0.0, 4.0\]'):
            build_matrix(['a.safetensors'], [bad])


def test_cell_workflow_carries_extended_strength_unclamped(app):
    """A > 2.0 test strength must reach the LoraLoader as-is (no silent clamp back
    to the old 2.0 ceiling) so the exaggerated effect is actually rendered."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        checkpoint = 'z image\\lora_zt_000001000.safetensors'
        workflow = lts._build_cell_workflow(
            user_id='local', checkpoint=checkpoint, strength=3.5, prompt='a prompt',
            seed=42, z_model=None, allowed_loras={checkpoint}, dataset_id=1,
            train_type='zimage', trigger_word='zt')
        lora_nodes = [n for n in workflow.values()
                      if isinstance(n, dict) and n.get('class_type') == 'LoraLoaderModelOnly']
        tested = [n for n in lora_nodes if n['inputs']['lora_name'] == checkpoint]
        assert tested and tested[0]['inputs']['strength_model'] == 3.5


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
        assert payload['queued'] == payload['generating'] == payload['running'] == 0
        assert payload['resumable'] == 0
        assert payload['max_images'] == lts.MAX_TEST_IMAGES
        # SRC's 'saved_to_gallery'/history-hiding fields are dropped for this app.
        assert 'saved_to_gallery' not in json_dump_keys(payload)


def test_studio_payload_splits_queued_and_generating_from_real_queue(app):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Activity', 'activity')
        queued_id = queue_manager.add_job(workflow_data={'1': {}})
        running_id = queue_manager.add_job(workflow_data={'1': {}})
        running_job = ImageGenerationQueue.query.filter_by(job_id=running_id).one()
        running_job.update_status('sent_to_comfy', comfyui_prompt_id='prompt-running')
        svc.db.session.add_all([
            LoraTestImage(dataset_id=ds.id, checkpoint='z image\\activity_a.safetensors',
                          strength=1.0, status='pending', job_id=queued_id),
            LoraTestImage(dataset_id=ds.id, checkpoint='z image\\activity_b.safetensors',
                          strength=1.0, status='pending', job_id=running_id),
        ])
        svc.db.session.commit()

        payload = lts.studio_payload(LOCAL_USER, ds.id)
        assert payload['pending'] == 2
        assert payload['queued'] == 1
        assert payload['generating'] == payload['running'] == 1
        assert {cell['queue_status'] for cell in payload['cells']} == {'queued', 'generating'}


def test_studio_payload_run_queue_counts_are_scoped_to_run_id(app):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Run activity', 'runactivity')
        queued_id = queue_manager.add_job(workflow_data={'1': {}})
        running_id = queue_manager.add_job(workflow_data={'1': {}})
        other_id = queue_manager.add_job(workflow_data={'1': {}})
        running_job = ImageGenerationQueue.query.filter_by(job_id=running_id).one()
        running_job.update_status('processing')
        svc.db.session.add_all([
            LoraTestImage(dataset_id=ds.id, run_id='run-a', checkpoint='z image\\a.safetensors',
                          strength=1.0, status='pending', job_id=queued_id),
            LoraTestImage(dataset_id=ds.id, run_id='run-a', checkpoint='z image\\b.safetensors',
                          strength=1.0, status='pending', job_id=running_id),
            LoraTestImage(dataset_id=ds.id, run_id='run-b', checkpoint='z image\\c.safetensors',
                          strength=1.0, status='pending', job_id=other_id),
        ])
        svc.db.session.commit()

        payload = lts.studio_payload_run(LOCAL_USER, 'run-a')
        assert payload['pending'] == 2
        assert payload['queued'] == 1
        assert payload['generating'] == payload['running'] == 1
        assert len(payload['cells']) == 2


def test_cancel_run_commits_whole_batch_before_interrupting_comfyui(app):
    from unittest.mock import patch
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Stop batch', 'stopbatch')
        running_id = queue_manager.add_job(workflow_data={'1': {}})
        queued_id = queue_manager.add_job(workflow_data={'1': {}})
        running = ImageGenerationQueue.query.filter_by(job_id=running_id).one()
        running.update_status('sent_to_comfy', comfyui_prompt_id='prompt-stop')
        svc.db.session.add_all([
            LoraTestImage(dataset_id=ds.id, run_id='run-stop', checkpoint='z image\\a.safetensors',
                          strength=1.0, status='pending', job_id=running_id),
            LoraTestImage(dataset_id=ds.id, run_id='run-stop', checkpoint='z image\\b.safetensors',
                          strength=1.0, status='pending', job_id=queued_id),
        ])
        svc.db.session.commit()

        def assert_batch_is_already_stopped(_prompt_id, _job_id):
            assert {j.status for j in ImageGenerationQueue.query.all()} == {'cancelled'}
            assert {c.status for c in LoraTestImage.query.all()} == {'cancelled'}
            return True

        with patch.object(queue_manager, 'interrupt_comfyui_job',
                          side_effect=assert_batch_is_already_stopped) as interrupt:
            assert lts.cancel_run(LOCAL_USER, run_id='run-stop') == 2
        interrupt.assert_called_once_with('prompt-stop', running_id)


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


def test_apply_krea_base_model_sets_node20_and_validates(app):
    """Base Krea locale : `base_model` remplace le UNET câblé du node 20, None le
    laisse intact, hors-whitelist → ValueError (anti path-injection)."""
    from app.services import lora_test_studio as lts
    from app.utils.comfyui import load_workflow_local
    with app.app_context():
        lora = 'krea\\lora_k_000001000.safetensors'
        base = 'krea\\my_custom_krea.safetensors'
        common = dict(lora_name=lora, strength=0.9, prompt='p', seed=1,
                      width=832, height=1216, allowed_loras={lora})
        wf = load_workflow_local(str(lts.WORKFLOW_KREA_TURBO_PATH))
        wired = wf['20']['inputs']['unet_name']
        lts.apply_krea_lora_test_settings(wf, **common)                     # None → intact
        assert wf['20']['inputs']['unet_name'] == wired
        lts.apply_krea_lora_test_settings(wf, **common, base_model=base,
                                          allowed_bases={base})
        assert wf['20']['inputs']['unet_name'] == base
        with pytest.raises(ValueError, match='unknown Krea base'):
            lts.apply_krea_lora_test_settings(wf, **common,
                                              base_model='..\\evil.safetensors',
                                              allowed_bases={base})


def test_krea_alt_base_models_excludes_wired_default(app, monkeypatch):
    """Les listes de bases ALTERNATIVES excluent le UNET câblé du workflow (déjà
    représenté par l'entrée « Official ») — quel que soit son dossier/sa casse."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        monkeypatch.setattr(lts, 'get_krea_models', lambda: [
            'Krea\\krea2_turbo_fp8.safetensors',      # défaut câblé (sous-dossier)
            'krea2_turbo_fp8.safetensors',            # copie racine du même défaut
            'krea\\my_custom_krea.safetensors',
        ])
        assert lts.krea_alt_base_models() == ['krea\\my_custom_krea.safetensors']


def test_build_cell_workflow_krea_honors_local_base(app, monkeypatch):
    """Bout-en-bout cellule Krea : z_model (base locale) atterrit dans le node 20
    et le LoRA testé est bien injecté — même canal de base que SDXL/Z-Image."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        lora = 'krea\\lora_k_000001000.safetensors'
        base = 'krea\\my_custom_krea.safetensors'
        monkeypatch.setattr(lts, 'get_krea_loras', lambda: [{'filename': lora}])
        monkeypatch.setattr(lts, 'get_krea_models', lambda: [base])
        wf = lts._build_cell_workflow(
            user_id='local', checkpoint=lora, strength=0.9, prompt='a prompt',
            seed=42, z_model=base, allowed_loras={lora}, dataset_id=1,
            train_type='krea', trigger_word='kt')
        assert wf['20']['inputs']['unet_name'] == base
        lora_nodes = [n for n in wf.values()
                      if isinstance(n, dict) and n.get('class_type') == 'LoraLoaderModelOnly']
        assert any(n['inputs']['lora_name'] == lora for n in lora_nodes)
        # z_model=None (entrée « Official ») → UNET câblé intact.
        wf2 = lts._build_cell_workflow(
            user_id='local', checkpoint=lora, strength=0.9, prompt='a prompt',
            seed=42, z_model=None, allowed_loras={lora}, dataset_id=1,
            train_type='krea', trigger_word='kt')
        assert wf2['20']['inputs']['unet_name'] == 'Krea\\krea2_turbo_fp8.safetensors'


def _configure_comfy(tmp_path, monkeypatch):
    """A tmp ComfyUI base with an empty models/ tree; returns its path."""
    from app import config
    base = tmp_path / 'Comfy'
    (base / 'models').mkdir(parents=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    return base


# --- P0-a: Studio preflight (model files on disk + custom nodes) --------------

def test_preflight_family_flags_missing_model_file(app, tmp_path, monkeypatch):
    """A VAE the built graph references but that's absent on disk → StudioAssetsMissing
    listing it with its expected models/ path (the fresh-user Krea/SDXL silent-fail)."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        (base / 'models' / 'unet' / 'present.safetensors').touch()
        # object_info: every node available → isolate the file check.
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader', 'VAELoader'})
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}},
              '2': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'nope_vae.safetensors'}}}
        with pytest.raises(lts.StudioAssetsMissing) as ei:
            lts.preflight_family('zimage', [wf])
        e = ei.value
        assert e.family == 'zimage' and e.missing_nodes == []
        assert any(f['path'] == 'models/vae/nope_vae.safetensors' and f['kind'] == 'VAE'
                   for f in e.missing_files)
        # The present UNET is NOT reported missing.
        assert all('present.safetensors' not in f['path'] for f in e.missing_files)


def test_preflight_family_flags_missing_custom_node_via_object_info(app, tmp_path, monkeypatch):
    """A custom node the graph uses but that /object_info doesn't list → reported
    as a missing node (compare class_type ⊄ available)."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        (base / 'models' / 'unet' / 'present.safetensors').touch()
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader'})  # no Krea2RebalanceConditioning
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}},
              '30': {'class_type': 'Krea2RebalanceConditioning', 'inputs': {}}}
        with pytest.raises(lts.StudioAssetsMissing) as ei:
            lts.preflight_family('krea', [wf])
        assert ei.value.missing_nodes == ['Krea2RebalanceConditioning']
        assert ei.value.missing_files == []


def test_preflight_family_passes_when_everything_present(app, tmp_path, monkeypatch):
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        (base / 'models' / 'unet' / 'present.safetensors').touch()
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader'})
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}}}
        lts.preflight_family('zimage', [wf])  # no raise


def test_preflight_object_info_unreachable_fails_open_on_nodes(app, tmp_path, monkeypatch):
    """When /object_info can't be fetched (None), the node check is SKIPPED (fail-open)
    — never block a launch on a transient probe failure; the per-tile error capture
    (P0-b) still surfaces a genuinely-missing node at runtime."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        (base / 'models' / 'unet' / 'present.safetensors').touch()
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda *a, **k: None)
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}},
              '9': {'class_type': 'SomeMissingCustomNode', 'inputs': {}}}
        lts.preflight_family('krea', [wf])  # file present + node check skipped → no raise


def test_preflight_matches_folder_casing_insensitively(app, tmp_path, monkeypatch):
    """The workflow templates carry 'Z image\\…' / 'Krea\\…' while the folders on
    disk are 'z image' / 'krea' — the file check must resolve regardless of case."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        te_dir = base / 'models' / 'text_encoders' / 'z image'
        te_dir.mkdir(parents=True)
        (te_dir / 'qwen_3_4b.safetensors').touch()
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'CLIPLoader'})
        wf = {'2': {'class_type': 'CLIPLoader',
                    'inputs': {'clip_name': 'Z image\\qwen_3_4b.safetensors'}}}
        lts.preflight_family('zimage', [wf])  # 'Z image' ref resolves to 'z image' dir


def test_create_run_preflights_missing_zimage_vae_and_text_encoder(app, tmp_path, monkeypatch):
    """End-to-end fresh-user scenario: the LoRA + base UNET are on disk but the
    Z-Image workflow's hardcoded VAE ('z ae') and text encoder ('Z image/qwen_3_4b')
    aren't → create_run raises StudioAssetsMissing BEFORE creating a single row
    (no grid of doomed tiles). Uses the REAL _build_cell_workflow."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        lora_dir = base / 'models' / 'loras' / 'z image'
        lora_dir.mkdir(parents=True)
        ck = 'z image\\lora_pf_000002000.safetensors'
        (lora_dir / 'lora_pf_000002000.safetensors').touch()
        unet_dir = base / 'models' / 'unet' / 'z image'
        unet_dir.mkdir(parents=True)
        (unet_dir / 'zmodel.safetensors').touch()
        # Deliberately NO models/vae/z ae.safetensors and NO text_encoders/…/qwen_3_4b.
        import app.utils.comfyui as comfyui_utils
        monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader', 'CLIPLoader', 'VAELoader',
                                             'CLIPTextEncode', 'EmptySD3LatentImage',
                                             'BasicScheduler', 'KSamplerSelect', 'CFGGuider',
                                             'RandomNoise', 'SamplerCustomAdvanced', 'VAEDecode',
                                             'SaveImage', 'LoraLoaderModelOnly'})
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        ds = svc.create_dataset(LOCAL_USER, 'PF', 'pf')
        with pytest.raises(lts.StudioAssetsMissing) as ei:
            lts.create_run(LOCAL_USER, ds.id, [ck], [1.0], prompt='p', count=1)
        paths = ' '.join(f['path'] for f in ei.value.missing_files)
        assert 'z ae.safetensors' in paths and 'qwen_3_4b.safetensors' in paths
        assert LoraTestImage.query.filter_by(dataset_id=ds.id).count() == 0  # no rows created


# --- P0-b: failed cells say WHY + are excluded from ranking -------------------

def test_link_completed_test_image_failed_records_reason(app, tmp_path):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    from app import config
    with app.app_context():
        base = tmp_path / 'Comfy'
        (base / 'output').mkdir(parents=True)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        ds = svc.create_dataset(LOCAL_USER, 'Why', 'whytrig')
        img = LoraTestImage(dataset_id=ds.id, checkpoint='z image\\lora_whytrig_000001000.safetensors',
                            strength=1.0, status='pending', job_id='job-why')
        svc.db.session.add(img)
        svc.db.session.commit()
        lts.link_completed_test_image('job-why', None, failed=True,
                                      reason='WORKFLOW_INVALIDE (validation ComfyUI 400): VAE not found')
        refreshed = svc.db.session.get(LoraTestImage, img.id)
        assert refreshed.status == 'failed'
        assert refreshed.error == 'WORKFLOW_INVALIDE (validation ComfyUI 400): VAE not found'


def test_failed_cell_excluded_from_cell_scores_ranking(app):
    """A failed cell shares its config key with a real done cell — it must NOT
    inflate the 'images' denominator nor otherwise pollute the ranking."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Rank', 'ranktrig')
        ck = 'z image\\lora_ranktrig_000002000.safetensors'
        svc.db.session.add(LoraTestImage(dataset_id=ds.id, checkpoint=ck, strength=1.0,
                                         status='done', rating=1))
        svc.db.session.add(LoraTestImage(dataset_id=ds.id, checkpoint=ck, strength=1.0,
                                         status='failed', error='boom'))
        svc.db.session.commit()
        scores = lts.cell_scores(ds.id, family='zimage')
        assert len(scores) == 1
        assert scores[0]['images'] == 1  # the failed row is excluded, not counted


def test_studio_payload_exposes_error_only_on_failed_cell(app):
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Pay', 'paytrig')
        ck = 'z image\\lora_paytrig_000001000.safetensors'
        failed = LoraTestImage(dataset_id=ds.id, checkpoint=ck, strength=1.0,
                               status='failed', error='the reason')
        done = LoraTestImage(dataset_id=ds.id, checkpoint=ck, strength=1.0,
                             status='done', error='stale', filename='x.png')
        svc.db.session.add_all([failed, done])
        svc.db.session.commit()
        payload = lts.studio_payload(LOCAL_USER, ds.id)
        by_id = {c['id']: c for c in payload['cells']}
        assert by_id[failed.id]['error'] == 'the reason'
        assert by_id[done.id]['error'] is None  # non-failed cells never leak an error


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


# --- P2: no private HttpNotifyNode in embedded workflows ----------------------

def _all_workflow_files():
    from app.services import lora_test_studio as lts
    import glob, os
    wf_dir = os.path.join(str(lts.cfg.BACKEND_DIR), 'workflows')
    return sorted(glob.glob(os.path.join(wf_dir, '*.json')))


def test_no_embedded_workflow_references_httpnotifynode():
    """The private `HttpNotifyNode` (a vestige of another app that POSTs to a
    hardcoded localhost:5000 and that no fresh user owns) must not appear in ANY
    embedded workflow — otherwise the studio preflight flags it as missing and the
    SDXL grid silently produces nothing on a clean install."""
    import json
    offenders = []
    for p in _all_workflow_files():
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
        for node in data.values():
            if isinstance(node, dict) and node.get('class_type') == 'HttpNotifyNode':
                offenders.append(p)
    assert offenders == [], f'HttpNotifyNode still present in: {offenders}'


def test_sdxl_workflow_has_saveimage_wired_to_decoded_image():
    """image_real_HQ.json (SDXL) must end in a standard SaveImage fed by the final
    VAEDecode — so its result lands in ComfyUI history (type='output') and is fetched
    by the same history/`/view` path as Z-Image/Krea/Klein. Its default filename_prefix
    must be meaningful (the private node's was the unrelated 'HQ_GeneratedImage')."""
    import json
    from app.services import lora_test_studio as lts
    with open(str(lts.WORKFLOW_HQ_PATH), encoding='utf-8') as f:
        data = json.load(f)
    saves = [(nid, n) for nid, n in data.items()
             if isinstance(n, dict) and n.get('class_type') == 'SaveImage']
    assert len(saves) == 1, 'SDXL workflow must have exactly one SaveImage'
    nid, save = saves[0]
    src = save['inputs']['images']
    assert isinstance(src, list) and len(src) == 2
    src_node = data.get(src[0])
    assert src_node and src_node.get('class_type') == 'VAEDecode'
    assert save['inputs'].get('filename_prefix')  # non-empty, meaningful


def test_sdxl_builder_filename_prefix_actually_reaches_saveimage(app):
    """Regression: `apply_sdxl_lora_test_settings` set filename_prefix on node id '9',
    which used to NOT EXIST in the workflow (the sole output was HttpNotifyNode/'65')
    → the per-cell prefix was a silent no-op and every cell reused ComfyUI's counter
    names (browser-cache collisions across LoRAs). The SaveImage now lives at node '9',
    so the prefix must land on it."""
    import json
    from app.services import lora_test_studio as lts
    with app.app_context():
        with open(str(lts.WORKFLOW_HQ_PATH), encoding='utf-8') as f:
            data = json.load(f)
        lts.apply_sdxl_lora_test_settings(
            data, base_ckpt='Biglove\\base.safetensors',
            lora_name='sdxl\\lora_nova_000001000.safetensors', strength=1.0,
            prompt='p', seed=1, width=1024, height=1024,
            filename_prefix='local_d7_LoraTest_abcd1234')
        save = next(n for n in data.values()
                    if isinstance(n, dict) and n.get('class_type') == 'SaveImage')
        assert save['inputs']['filename_prefix'] == 'local_d7_LoraTest_abcd1234'


def test_sdxl_preflight_scan_drops_httpnotify_keeps_detaildaemon():
    """The preflight's class-type scan of the SDXL workflow must no longer surface
    HttpNotifyNode (so a fresh user is never told to install a node nobody ships),
    while DetailDaemonSamplerNode — a FUNCTIONAL custom node the graph really needs —
    stays required, and SaveImage (core) is present."""
    import json
    from app.services import lora_test_studio as lts
    with open(str(lts.WORKFLOW_HQ_PATH), encoding='utf-8') as f:
        data = json.load(f)
    _missing, classes = lts._scan_workflow_assets(data, None)
    assert 'HttpNotifyNode' not in classes
    assert 'SaveImage' in classes
    assert 'DetailDaemonSamplerNode' in classes
