"""Tests for the LoRA Test Studio service (checkpoint x strength sweep).

ComfyUI is never contacted: `queue_manager.add_job`/`_build_cell_workflow` are
monkeypatched for the enqueue-path tests, and the workflow-build test loads
the real copied workflow JSON but stops short of a network call."""
import struct
import threading
import pytest

# Smallest structurally-valid safetensors header (8-byte LE length + '{}'). The
# Studio preflight now rejects a present-but-unloadable model file (an HTML gate
# page, a truncated stub), so a fixture whose file must read as REAL weights writes
# these bytes instead of touch()ing a 0-byte stub.
_ST = struct.pack('<Q', 2) + b'{}'


def test_build_matrix_shape_and_validation(app):
    from app.services.lora_test_studio import build_matrix
    m = build_matrix(['a.safetensors', 'b.safetensors'], [0.8, 1.0], aspects=['9:16'])
    assert len(m) == 4 and all(len(t) == 6 for t in m)
    try:
        build_matrix(['a'], [99.0]); ok = False
    except Exception:
        ok = True
    assert ok


def test_build_matrix_accepts_extended_strengths_up_to_the_ceiling(app):
    """Progressive-disclosure « + » exposes strengths above 2.0 (over-cook range);
    the sweep validation accepts up to MAX_LORA_STRENGTH and rejects beyond.

    The ceiling moved 4.0 → 5.0 on 2026-08-08 with the 🧬 Blend weight, which
    lands in the same column through the same validation."""
    from app.services.lora_test_studio import build_matrix, MAX_LORA_STRENGTH
    assert MAX_LORA_STRENGTH == 5.0
    m = build_matrix(['a.safetensors'], [2.5, 3.5, 4.0, 5.0], aspects=['9:16'])
    assert sorted({t[1] for t in m}) == [2.5, 3.5, 4.0, 5.0]  # carried, not clamped
    for bad in (5.01, 6.0, 10.0):
        with pytest.raises(ValueError, match=r'out of range \[-2.0, 5.0\]'):
            build_matrix(['a.safetensors'], [bad])


def test_build_matrix_accepts_negative_strengths_down_to_minus_2(app):
    """Progressive-disclosure « − » exposes NEGATIVE strengths (the other pole of
    a slider LoRA — and a legit probe on any LoRA): the sweep validation accepts
    down to -2.0 and rejects anything below (mirror of the 4.0 ceiling)."""
    from app.services.lora_test_studio import build_matrix
    m = build_matrix(['a.safetensors'], [-2.0, -1.0, -0.5, 0, 1.0], aspects=['9:16'])
    assert sorted({t[1] for t in m}) == [-2.0, -1.0, -0.5, 0.0, 1.0]   # carried as-is
    for bad in (-2.01, -3.0, -10.0):
        with pytest.raises(ValueError, match=r'out of range \[-2.0, 5.0\]'):
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


def test_cell_workflow_carries_negative_strength_unclamped(app):
    """A negative test strength must reach the LoraLoader as-is — exercised on
    the Z-Image cell path here (inject_zimage_loras floor -2.0); SDXL sets node
    25 directly with no clamp, and the Krea injector trap is covered by
    test_inject_krea_loras_passes_negative_strength below."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        checkpoint = 'z image\\lora_zt_000001000.safetensors'
        workflow = lts._build_cell_workflow(
            user_id='local', checkpoint=checkpoint, strength=-1.5, prompt='a prompt',
            seed=42, z_model=None, allowed_loras={checkpoint}, dataset_id=1,
            train_type='zimage', trigger_word='zt')
        lora_nodes = [n for n in workflow.values()
                      if isinstance(n, dict) and n.get('class_type') == 'LoraLoaderModelOnly']
        tested = [n for n in lora_nodes if n['inputs']['lora_name'] == checkpoint]
        assert tested and tested[0]['inputs']['strength_model'] == -1.5


def test_inject_krea_loras_zero_strength_is_true_noop(app):
    """A zero-strength Krea LoRA must not install model hooks at all."""
    from app.utils.comfyui import inject_krea_loras
    workflow = {
        '20': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'krea2_turbo_fp8.safetensors'}},
        '26': {'class_type': 'KSampler', 'inputs': {'model': ['20', 0]}},
    }

    n = inject_krea_loras(
        workflow,
        [{'filename': 'krea\\zero.safetensors', 'strength': 0.0}],
        allowed={'krea\\zero.safetensors'},
    )

    assert n == 0
    assert not any(node.get('class_type') == 'LoraLoaderModelOnly'
                   for node in workflow.values())
    assert workflow['26']['inputs']['model'] == ['20', 0]


def test_inject_krea_loras_skips_zero_and_chains_nonzero_loras(app):
    """A zero entry between active LoRAs must not break their model chain."""
    from app.utils.comfyui import inject_krea_loras
    workflow = {
        '20': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'krea2_turbo_fp8.safetensors'}},
        '26': {'class_type': 'KSampler', 'inputs': {'model': ['20', 0]}},
    }
    requested = [
        {'filename': 'krea\\first.safetensors', 'strength': 0.75},
        {'filename': 'krea\\zero.safetensors', 'strength': 0.0},
        {'filename': 'krea\\last.safetensors', 'strength': 1.25},
    ]

    n = inject_krea_loras(
        workflow,
        requested,
        allowed={item['filename'] for item in requested},
    )

    assert n == 2
    assert 'krea_lora_1' not in workflow
    assert workflow['krea_lora_0']['inputs'] == {
        'lora_name': 'krea\\first.safetensors',
        'strength_model': 0.75,
        'model': ['20', 0],
    }
    assert workflow['krea_lora_2']['inputs'] == {
        'lora_name': 'krea\\last.safetensors',
        'strength_model': 1.25,
        'model': ['krea_lora_0', 0],
    }
    assert workflow['26']['inputs']['model'] == ['krea_lora_2', 0]


def test_inject_krea_loras_passes_negative_strength(app):
    """The Krea injector used to clamp at max(0.0, …): a negative tested strength
    silently became 0 (LoRA off) with no error anywhere. Floor is now -2.0."""
    from app.utils.comfyui import inject_krea_loras
    workflow = {
        '20': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'krea2_turbo_fp8.safetensors'}},
        '26': {'class_type': 'KSampler', 'inputs': {'model': ['20', 0]}},
    }
    n = inject_krea_loras(workflow, [{'filename': 'krea\\slider.safetensors', 'strength': -1.5}],
                          allowed={'krea\\slider.safetensors'})
    assert n == 1
    assert workflow['krea_lora_0']['inputs']['strength_model'] == -1.5
    # The anti-absurd floor still exists (mirrors the -2.0 UI/server bound).
    workflow2 = {
        '20': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'krea2_turbo_fp8.safetensors'}},
        '26': {'class_type': 'KSampler', 'inputs': {'model': ['20', 0]}},
    }
    inject_krea_loras(workflow2, [{'filename': 'krea\\slider.safetensors', 'strength': -50}],
                      allowed={'krea\\slider.safetensors'})
    assert workflow2['krea_lora_0']['inputs']['strength_model'] == -2.0


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
        (unet_dir / 'zmodel.safetensors').write_bytes(_ST)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        # get_zimage_models() has a 5-minute TTL cache (app.utils.comfyui); reset it so
        # this test's real directory is seen instead of another test's stale result.
        import app.utils.comfyui as comfyui_utils
        monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})
        ds = svc.create_dataset(LOCAL_USER, 'S2', 's')
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        # The cell row now carries the job_id it was CREATED with (the id is minted
        # before the insert, so row + queue job land in one commit) -- capture what
        # the enqueue was handed and assert the row matches it.
        seen = []
        seen_run_ids = []

        def fake_enqueue(user_id, dataset_id, workflow, prompt, job_id=None,
                         commit=True, **_metadata):
            seen.append(job_id)
            seen_run_ids.append(_metadata.get('run_id'))
            return job_id
        monkeypatch.setattr(lts, '_enqueue_cell', fake_enqueue)
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        out = lts.create_run(LOCAL_USER, ds.id, [ck], [1.0], prompt='p', count=1)
        rows = LoraTestImage.query.filter_by(dataset_id=ds.id).all()
        assert out['created'] == len(rows) >= 1
        assert seen and all(j for j in seen)
        assert sorted(r.job_id for r in rows) == sorted(seen)
        assert all(r.status == 'pending' for r in rows)
        assert len(out['run_id']) == 32
        assert {r.run_id for r in rows} == {out['run_id']}
        assert set(seen_run_ids) == {out['run_id']}


def test_cell_enqueue_holds_gpu_arbiter_until_commit_before_recovery_barrier(
        app, monkeypatch):
    """A recovery writer cannot slip between add_job(commit=False) and commit.

    The events make the contested window deterministic: the Studio thread has
    staged both rows but pauses immediately after add_job returns.  A recovery
    thread then attempts the same arbiter.  It must remain blocked until Studio
    is allowed to commit and release the outer lock.
    """
    from app.config import LOCAL_USER
    from app.job_queue import (COMFYUI_STALLED_BARRIER_KEY, GPU_ARBITER_LOCK,
                               queue_manager)
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts

    with app.app_context():
        dataset = svc.create_dataset(LOCAL_USER, 'Atomic Studio', 'atomicstudio')
        dataset_id = dataset.id

    after_add_job = threading.Event()
    release_commit = threading.Event()
    barrier_attempted = threading.Event()
    barrier_lock_acquired = threading.Event()
    barrier_written = threading.Event()
    errors = []
    build_lock_state = []
    real_enqueue = lts._enqueue_cell

    def gated_enqueue(*args, **kwargs):
        result = real_enqueue(*args, **kwargs)
        after_add_job.set()
        if not release_commit.wait(timeout=10):
            raise TimeoutError('test did not release the Studio commit')
        return result

    monkeypatch.setattr(lts, '_enqueue_cell', gated_enqueue)

    def studio_writer():
        try:
            with app.app_context():
                image = LoraTestImage(
                    dataset_id=dataset_id,
                    checkpoint='z image\\atomic.safetensors',
                    strength=1.0,
                    status='pending',
                )

                def build_workflow():
                    build_lock_state.append(GPU_ARBITER_LOCK._is_owned())
                    return {'1': {}}

                lts._persist_and_enqueue_cell(
                    image, LOCAL_USER, dataset_id, 'atomic prompt', build_workflow)
        except BaseException as exc:
            errors.append(exc)

    def recovery_writer():
        try:
            barrier_attempted.set()
            with GPU_ARBITER_LOCK:
                barrier_lock_acquired.set()
                with app.app_context():
                    queue_manager._set_system_state(
                        COMFYUI_STALLED_BARRIER_KEY,
                        {
                            'job_id': 'concurrent-recovery',
                            'client_id': 'concurrent-recovery',
                            'prompt_id': None,
                            'kind': 'unknown_submit',
                        },
                    )
                barrier_written.set()
        except BaseException as exc:
            errors.append(exc)

    studio_thread = threading.Thread(target=studio_writer)
    recovery_thread = threading.Thread(target=recovery_writer)
    studio_thread.start()
    assert after_add_job.wait(timeout=10), 'Studio never reached its pre-commit window'
    recovery_thread.start()
    assert barrier_attempted.wait(timeout=10), 'recovery writer did not start'
    assert not barrier_lock_acquired.wait(timeout=0.1), (
        'recovery acquired GPU_ARBITER_LOCK before the Studio transaction committed')

    release_commit.set()
    studio_thread.join(timeout=10)
    recovery_thread.join(timeout=10)
    assert not studio_thread.is_alive() and not recovery_thread.is_alive()
    assert errors == []
    assert build_lock_state == [False]  # workflow resolution does not reserve the GPU
    assert barrier_lock_acquired.is_set() and barrier_written.is_set()

    with app.app_context():
        cell = LoraTestImage.query.filter_by(dataset_id=dataset_id).one()
        queued = ImageGenerationQueue.query.filter_by(job_id=cell.job_id).one()
        assert cell.status == queued.status == 'pending'
        assert queue_manager.get_comfyui_stalled_barrier()['job_id'] == 'concurrent-recovery'


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
        (unet_dir / 'zmodel.safetensors').write_bytes(_ST)
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


# Literal transcription of the FRONTEND `tierDims` (react-frontend/src/components/
# shared/ResolutionSelector.jsx) for the Z-Image/Krea path (maxLongSide undefined).
# It MUST stay byte-for-byte equivalent to compute_tier_dims — this is the invariant
# the studio relies on (px shown == px generated). Keep both in sync; this mirror
# exists so a one-sided edit (a changed cap, snap rounding or multiplier clamp) fails
# the test loudly instead of drifting silently.
def _front_tier_dims(aspect, mp, multiplier=1.0):
    import math
    _R = {'square': (1, 1), 'landscape': (4, 3), 'portrait': (3, 4), 'widescreen': (16, 9),
          'tall': (9, 16), 'photo': (3, 2), 'phototall': (2, 3), 'ultrawide': (21, 9)}
    CAP, ABS_CAP, FLOOR, M = 1536, 3072, 512, 16
    snap = lambda v: max(FLOOR, int(math.floor(v / M + 0.5)) * M)  # JS Math.round (positives)
    m = max(1.0, min(1.9, multiplier))
    rw, rh = _R.get(aspect, _R['square'])
    r = rw / rh
    h = math.sqrt(mp * 1e6 / r); w = r * h
    longest = max(w, h)
    if longest > CAP:
        s = CAP / longest; w *= s; h *= s
    w *= m; h *= m
    longest = max(w, h)
    if longest > ABS_CAP:
        s = ABS_CAP / longest; w *= s; h *= s
    return snap(w), snap(h)


def test_compute_tier_dims_mirrors_frontend_over_full_grid():
    """INVARIANT: compute_tier_dims (backend source of truth) and the frontend
    tierDims produce the SAME (w, h) for every (aspect, tier, multiplier) — so the
    Test Studio's live W×H readout matches the pixels actually generated. Swept over
    all 8 ratios × 4 tiers × 10 multiplier steps (1.0…1.9)."""
    from app.utils.resolution import compute_tier_dims, _RATIOS, _TIERS
    for aspect in _RATIOS:
        for tier, mp in _TIERS.items():
            for mi in range(10, 20):
                mult = mi / 10
                assert compute_tier_dims(aspect, tier, mult) == _front_tier_dims(aspect, mp, mult), \
                    f'front/back divergence at {aspect}/{tier}/x{mult}'


def test_compute_tier_dims_multiplier_semantics():
    """Multiplier: default 1.0 = preset unchanged; linear enlarge on both sides;
    clamped to [1.0, 1.9] (None/garbage/too-small → 1.0, never shrinks)."""
    from app.utils.resolution import compute_tier_dims, clamp_multiplier
    base = compute_tier_dims('square', 'standard')            # default multiplier
    assert base == (1008, 1008)                                # matches the frontend preset
    assert compute_tier_dims('square', 'standard', 1.0) == base
    # ×1.9 enlarges both sides (pre-snap base 1000 × 1.9 = 1900 → snap 1904).
    assert compute_tier_dims('square', 'standard', 1.9) == (1904, 1904)
    # Clamp: below 1.0 floors to preset, above 1.9 caps, junk → 1.0.
    assert compute_tier_dims('square', 'standard', 0.5) == base
    assert compute_tier_dims('square', 'standard', 5.0) == compute_tier_dims('square', 'standard', 1.9)
    assert clamp_multiplier(None) == 1.0 and clamp_multiplier('x') == 1.0
    assert clamp_multiplier(2.5) == 1.9 and clamp_multiplier(0.2) == 1.0


def test_aspect_dims_applies_multiplier():
    """_aspect_dims threads the multiplier into compute_tier_dims (tier path) and,
    for SDXL, scales the 1024 safe-band ceiling by the multiplier so it isn't
    silently clobbered. Legacy fixed-table path (no tier) ignores the multiplier."""
    from app.services.lora_test_studio import _aspect_dims
    from app.utils.resolution import compute_tier_dims
    # Z-Image/Krea: exactly compute_tier_dims with the multiplier.
    assert _aspect_dims('1:1', 'zimage', 'standard', 1.9) == compute_tier_dims('square', 'standard', 1.9)
    # A bigger multiplier yields a strictly larger square (monotonic).
    w1, _ = _aspect_dims('1:1', 'zimage', 'standard', 1.0)
    w2, _ = _aspect_dims('1:1', 'zimage', 'standard', 1.9)
    assert w2 > w1
    # SDXL widescreen: base long side exceeds 1024, so it rides the SDXL safe-band
    # ceiling (scaled by the multiplier) and snaps to ÷64. At ×1.0 the historical
    # 1024 cap holds; ×1.9 raises it instead of clobbering the multiplier.
    sw0, sh0 = _aspect_dims('16:9', 'sdxl', 'standard', 1.0)
    sw, sh = _aspect_dims('16:9', 'sdxl', 'standard', 1.9)
    assert sw0 <= 1024
    assert sw % 64 == 0 and sh % 64 == 0 and sw > sw0
    # No tier → legacy fixed table, multiplier is inert.
    assert _aspect_dims('1:1', 'zimage', None, 1.9) == _aspect_dims('1:1', 'zimage', None, 1.0)


def _studio_fixture(tmp_path, monkeypatch, name, trigger, steps=(2000,)):
    """A configured ComfyUI tree + a dataset whose trigger matches `steps` checkpoints.
    Returns (dataset, [checkpoint filenames])."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config
    base = tmp_path / 'Comfy'
    lora_dir = base / 'models' / 'loras' / 'z image'
    lora_dir.mkdir(parents=True, exist_ok=True)
    cks = []
    for st in steps:
        fn = f'lora_{trigger}_{st:09d}.safetensors'
        (lora_dir / fn).write_bytes(_ST)
        cks.append('z image\\' + fn)
    unet_dir = base / 'models' / 'unet' / 'z image'
    unet_dir.mkdir(parents=True, exist_ok=True)
    (unet_dir / 'zmodel.safetensors').write_bytes(_ST)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    import app.utils.comfyui as comfyui_utils
    monkeypatch.setattr(comfyui_utils, '_zimage_models_cache', {'data': None, 'timestamp': 0})
    return svc.create_dataset(LOCAL_USER, name, trigger), cks


def _capture_krea_enqueue(monkeypatch, lts, checkpoints, base_models=()):
    """Remove external Krea/Comfy dependencies and capture materialization order."""
    seen = []
    monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
    monkeypatch.setattr(lts, '_active_run_count', lambda *args: 0)
    monkeypatch.setattr(
        lts, 'list_test_checkpoints',
        lambda _ds, _family=None: [{'filename': checkpoint}
                                    for checkpoint in checkpoints],
    )
    monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _family: [])
    monkeypatch.setattr(lts, 'get_krea_models', lambda: list(base_models))
    monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *args, **kwargs: None)
    monkeypatch.setattr(lts, '_preflight_run', lambda *args, **kwargs: None)
    monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
    monkeypatch.setattr(
        lts, 'checkpoint_origins',
        lambda selected, explicit=None: {checkpoint: (None, None)
                                         for checkpoint in selected},
    )

    def capture(img, *_args, **_kwargs):
        seen.append((img.z_model, img.checkpoint, img.strength))

    monkeypatch.setattr(lts, '_persist_and_enqueue_cell', capture)
    return seen


def test_create_run_krea_enqueues_all_zero_cells_before_stable_nonzero_cells(
        app, monkeypatch):
    """Krea's tested-LoRA-off controls lead; model/checkpoint order stays stable.

    A negative force is deliberately part of the patched/non-zero partition.
    """
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Krea order', 'korder')
        checkpoints = [
            'krea\\lora_korder_a.safetensors',
            'krea\\lora_korder_b.safetensors',
        ]
        alt_base = 'krea\\alternate_base.safetensors'
        seen = _capture_krea_enqueue(
            monkeypatch, lts, checkpoints, base_models=[alt_base])

        out = lts.create_run(
            LOCAL_USER, ds.id, checkpoints, [1.0, 0.0, -0.5, 0.75],
            prompt='p', count=1, z_models=[None, alt_base])

        expected = []
        for base in (None, alt_base):
            expected.extend((base, checkpoint, 0.0) for checkpoint in checkpoints)
        for base in (None, alt_base):
            for checkpoint in checkpoints:
                expected.extend([
                    (base, checkpoint, 1.0),
                    (base, checkpoint, -0.5),
                    (base, checkpoint, 0.75),
                ])
        assert out['created'] == len(seen) == len(expected)
        assert seen == expected


def test_create_comparison_run_krea_enqueues_all_zero_cells_before_stable_nonzero_cells(
        app, monkeypatch):
    """Comparison plans are partitioned globally, not once per selection."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Krea compare order', 'kcompare')
        checkpoints = [
            'krea\\lora_kcompare_a.safetensors',
            'krea\\lora_kcompare_b.safetensors',
        ]
        seen = _capture_krea_enqueue(monkeypatch, lts, checkpoints)

        out = lts.create_comparison_run(
            LOCAL_USER,
            [{'dataset_id': ds.id, 'checkpoint': checkpoint}
             for checkpoint in checkpoints],
            [1.0, 0.0, -0.5, 0.75],
            prompt='p',
            count=1,
        )

        expected = [(None, checkpoint, 0.0) for checkpoint in checkpoints]
        for checkpoint in checkpoints:
            expected.extend([
                (None, checkpoint, 1.0),
                (None, checkpoint, -0.5),
                (None, checkpoint, 0.75),
            ])
        assert out['created'] == len(seen) == len(expected)
        assert seen == expected


def test_zero_first_partition_does_not_reorder_other_families(app):
    """Klein/Z-Image/SDXL retain the exact historical enqueue order."""
    from app.services.lora_test_studio import _krea_zero_strength_first
    original = [('a', 1.0), ('a', 0.0), ('b', -0.5), ('b', 0.0)]
    for family in ('klein', 'zimage', 'sdxl'):
        assert _krea_zero_strength_first(
            original, family, lambda item: item[1]) == original


def test_create_comparison_run_commits_rows_before_enqueue(app, monkeypatch, tmp_path):
    """Same anti-orphan guarantee as create_run on the multi-LoRA comparison path:
    every created cell is COMMITTED, already carrying the job_id its enqueue used."""
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds, cks = _studio_fixture(tmp_path, monkeypatch, 'C', 'c')
        seen = []

        def fake_enqueue(user_id, dataset_id, workflow, prompt, job_id=None,
                         commit=True, **_metadata):
            seen.append(job_id)
            return job_id
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', fake_enqueue)
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        out = lts.create_comparison_run(LOCAL_USER, [{'dataset_id': ds.id, 'checkpoint': cks[0]}],
                                        [1.0], prompt='p', count=1)
        rows = LoraTestImage.query.filter_by(dataset_id=ds.id).all()
        assert out['created'] == len(rows) >= 1
        assert seen and all(j for j in seen)
        assert sorted(r.job_id for r in rows) == sorted(seen)
        assert all(r.status == 'pending' and r.run_id == out['run_id'] for r in rows)


def test_comparison_run_failure_keeps_previous_cells_and_marks_the_failed_one(app, monkeypatch, tmp_path):
    """THE invariant behind 'one commit per cell, not zero': an enqueue that blows up
    on cell N must leave the N-1 already-queued cells in the DB WITH their job_id
    (their ComfyUI jobs exist - rolling their rows back would orphan them), and cell N
    persisted as 'failed' with the reason. Cell N must NOT keep a job_id: its queue row
    was rolled back with it."""
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage, ImageGenerationQueue
    from app.config import LOCAL_USER
    with app.app_context():
        ds, cks = _studio_fixture(tmp_path, monkeypatch, 'F', 'f')
        calls = {'n': 0}
        real_enqueue = lts._enqueue_cell

        def flaky_enqueue(user_id, dataset_id, workflow, prompt, job_id=None,
                           commit=True, **metadata):
            calls['n'] += 1
            if calls['n'] == 3:              # blow up on the THIRD of five cells
                raise RuntimeError('comfy exploded')
            return real_enqueue(user_id, dataset_id, workflow, prompt,
                                job_id=job_id, commit=commit, **metadata)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', flaky_enqueue)
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        with pytest.raises(RuntimeError, match='comfy exploded'):
            lts.create_comparison_run(LOCAL_USER, [{'dataset_id': ds.id, 'checkpoint': cks[0]}],
                                      [0.6, 0.8, 1.0, 1.2, 1.4], prompt='p', count=1)
        rows = LoraTestImage.query.filter_by(dataset_id=ds.id).order_by(LoraTestImage.id).all()
        assert len(rows) == 3                       # the 2 survivors + the failed one
        queued = {j.job_id for j in ImageGenerationQueue.query.all()}
        for r in rows[:2]:
            assert r.status == 'pending' and r.job_id and r.job_id in queued
        assert rows[2].status == 'failed' and 'comfy exploded' in (rows[2].error or '')
        assert rows[2].job_id is None
        assert len(queued) == 2                     # no job without its cell row


def test_comparison_run_writes_one_transaction_per_cell_and_scans_loras_once(app, monkeypatch, tmp_path):
    """Perf contract of the Studio launch: a grid must cost ONE commit per cell (not
    the historical three) and ONE LoRA-folder scan per (dataset, family) - not one per
    selection. 3 selections x 2 strengths = 6 cells => 6 commits, 1 scan."""
    from app.services import lora_test_studio as lts
    from app.config import LOCAL_USER
    from sqlalchemy import event
    from sqlalchemy.orm import Session
    with app.app_context():
        ds, cks = _studio_fixture(tmp_path, monkeypatch, 'P', 'p', steps=(1000, 2000, 3000))
        scans = {'n': 0}
        real_list = lts.list_test_checkpoints

        def counting_list(_ds, family=None):
            scans['n'] += 1
            return real_list(_ds, family)
        monkeypatch.setattr(lts, 'list_test_checkpoints', counting_list)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
        commits = {'n': 0}

        def _count(_session):
            commits['n'] += 1
        event.listen(Session, 'after_commit', _count)
        try:
            out = lts.create_comparison_run(
                LOCAL_USER, [{'dataset_id': ds.id, 'checkpoint': c} for c in cks],
                [0.8, 1.0], prompt='p', count=1)
        finally:
            event.remove(Session, 'after_commit', _count)
        assert out['created'] == 6
        assert commits['n'] == 6, f'expected 1 commit per cell, got {commits["n"]}'
        assert scans['n'] == 1, f'expected 1 LoRA scan for the dataset, got {scans["n"]}'


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


def test_cancel_run_reconciles_exact_running_prompt_and_cancels_whole_batch(app):
    from unittest.mock import patch
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.job_queue import queue_manager
    from app.models import ImageGenerationQueue, LoraTestImage
    from app.utils.comfyui import ComfyPromptState
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

        # The queue owns the exact remote proof now: targeted delete followed by
        # fresh queue absence. No real ComfyUI request belongs in this service test.
        with patch('app.utils.comfyui.cancel_comfyui_prompt_state',
                   return_value=ComfyPromptState.DELETED) as cancel_prompt, \
             patch('app.utils.comfyui.comfyui_prompt_is_absent',
                   return_value=True) as prompt_absent:
            assert lts.cancel_run(LOCAL_USER, run_id='run-stop') == 2
        cancel_prompt.assert_called_once_with('prompt-stop', running_id)
        prompt_absent.assert_called_once_with('prompt-stop')
        assert {j.status for j in ImageGenerationQueue.query.all()} == {'cancelled'}
        cells = LoraTestImage.query.order_by(LoraTestImage.id).all()
        assert {cell.status for cell in cells} == {'cancelled'}
        assert all(cell.job_id is None for cell in cells)


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
        # z_model=None (entrée par défaut du picker) → base ÉLUE, pas le nom figé
        # dans krea2_turbo.json : ce littéral n'est pas le fichier que Setup
        # installe, donc sur une install qui a suivi Setup il désignait un fichier
        # absent et ComfyUI refusait le prompt (cf. test_krea_default_base_election).
        wf2 = lts._build_cell_workflow(
            user_id='local', checkpoint=lora, strength=0.9, prompt='a prompt',
            seed=42, z_model=None, allowed_loras={lora}, dataset_id=1,
            train_type='krea', trigger_word='kt')
        assert wf2['20']['inputs']['unet_name'] == base


def _build_krea_cell(lts, monkeypatch, *, available_classes):
    """Build one real Krea cell (loads krea2_turbo.json) for the node-class resolution
    tests. Node 30 ships as the canonical ConditioningKrea2Rebalance."""
    lora = 'krea\\lora_k_000001000.safetensors'
    monkeypatch.setattr(lts, 'get_krea_loras', lambda: [{'filename': lora}])
    monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
    return lts._build_cell_workflow(
        user_id='local', checkpoint=lora, strength=0.9, prompt='p', seed=1,
        z_model=None, allowed_loras={lora}, dataset_id=1, train_type='krea',
        trigger_word='kt', available_classes=available_classes)


def test_build_krea_cell_rewrites_rebalance_class_to_registered_alias(app, monkeypatch):
    """Résolveur de CLASSES : quand le ComfyUI cible n'enregistre le node de rebalance
    QUE sous le nom permuté (Krea2RebalanceConditioning — le cas de l'install du dev,
    origine du nom permuté qu'on avait d'abord livré), le builder réécrit le class_type
    du node 30 vers ce nom pour que le graphe ENQUEUÉ valide. Les inputs épinglés
    (preset='custom', renormalize=False) restent posés — la réécriture ne touche QUE le
    class_type (inoffensif sur une variante qui ignore ces inputs)."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        wf = _build_krea_cell(lts, monkeypatch,
                              available_classes={'Krea2RebalanceConditioning'})
        assert wf['30']['class_type'] == 'Krea2RebalanceConditioning'
        assert wf['30']['inputs']['preset'] == 'custom'
        assert wf['30']['inputs']['renormalize'] is False


def test_build_krea_cell_keeps_canonical_class_when_registered(app, monkeypatch):
    """Quand le ComfyUI cible expose bien la classe canonique, le node 30 la GARDE
    (pas de réécriture) et les inputs épinglés sont intacts."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        wf = _build_krea_cell(lts, monkeypatch,
                              available_classes={'ConditioningKrea2Rebalance'})
        assert wf['30']['class_type'] == 'ConditioningKrea2Rebalance'
        assert wf['30']['inputs']['preset'] == 'custom'
        assert wf['30']['inputs']['renormalize'] is False


def test_build_krea_cell_keeps_canonical_class_without_object_info(app, monkeypatch):
    """available_classes=None (probe /object_info échouée ou non fournie) → AUCUNE
    réécriture : on garde le nom canonique (fail-open ; le preflight / la capture
    d'erreur par tuile signalera un vrai manque)."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        wf = _build_krea_cell(lts, monkeypatch, available_classes=None)
        assert wf['30']['class_type'] == 'ConditioningKrea2Rebalance'


def test_multiword_trigger_style_lora_is_discoverable_in_studio(app, monkeypatch):
    """A style dataset's trigger can contain spaces ('raw test upscale'), but the
    training/deploy side slugifies it into the filename via _safe_trigger
    ('lora_raw_test_upscale_…'). The Studio's checkpoint match must canonicalize
    the trigger the SAME way, or the trained LoRA silently vanishes from the picker.

    Regression (2026-07-17): `_trigger_match_checkpoints` matched the RAW, still
    space-containing trigger against the underscored filename → no prefix match →
    the whole dataset disappeared from `/api/studio/checkpoints`."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'raw test upscale', 'raw test upscale',
                                kind='style', train_type='krea')
        # The exact on-disk deploy name, incl. eecc080's _rc<id>_v<N> import tag.
        deployed = 'krea\\lora_raw_test_upscale_Krea-2-Raw_rc52_v1.safetensors'
        monkeypatch.setattr(lts, 'get_krea_loras', lambda: [{'filename': deployed}])
        cks = lts.list_test_checkpoints(ds, 'krea')
        assert [c['filename'] for c in cks] == [deployed]
        assert 'krea' in [f['family'] for f in lts.available_families(ds)]


def test_trigger_match_still_respects_token_boundary_after_slugify(app, monkeypatch):
    """Canonicalizing the trigger must NOT loosen the token-boundary guard: the
    slugified trigger still only matches when the checkpoint continues with a
    separator (`_`/`-`) or ends — a prefix glued to more letters is rejected.
    Mirror of the historical 'lola' ⊂ 'lola3869' fix, but on a multi-word
    (post-slugify) trigger 'raw test' → 'raw_test'."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'raw test', 'raw test',
                                kind='style', train_type='krea')
        # 'raw_testupscale' glues more letters onto the slug (no separator) → reject.
        glued = 'krea\\lora_raw_testupscale_Krea-2-Raw_rc52_v1.safetensors'
        own = 'krea\\lora_raw_test_Krea-2-Raw_rc60_v1.safetensors'
        monkeypatch.setattr(lts, 'get_krea_loras', lambda: [{'filename': glued},
                                                            {'filename': own}])
        cks = [c['filename'] for c in lts.list_test_checkpoints(ds, 'krea')]
        assert cks == [own]   # NOT the glued 'raw_testupscale' non-sibling


def test_underscore_trigger_labels_faithfully_in_studio(app, monkeypatch):
    """A character/concept trigger can itself contain an underscore (`leg_behind`).
    The deployed filename embeds it verbatim (`lora_leg_behind_…`), where its own
    underscore is indistinguishable from the field separators. The Studio checkpoint
    label must stay faithful (`leg_behind · …`), never split into `leg · behind`
    (bug reported 2026-07-17): `_trigger_match_checkpoints` passes the dataset's real
    trigger to the label formatter for an exact parse."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'leg behind pose', 'leg_behind',
                                kind='concept', concept_desc='a leg-behind yoga pose',
                                train_type='krea')
        deployed = 'krea\\lora_leg_behind_000002000_Krea-2-Turbo_rc52_v1.safetensors'
        monkeypatch.setattr(lts, 'get_krea_loras', lambda: [{'filename': deployed}])
        cks = lts.list_test_checkpoints(ds, 'krea')
        assert [c['filename'] for c in cks] == [deployed]           # still discoverable
        label = cks[0]['label']
        assert label.startswith('leg_behind · ')                   # faithful trigger
        assert 'leg · behind' not in label                         # not split
        assert '2000 steps' in label


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
        (base / 'models' / 'unet' / 'present.safetensors').write_bytes(_ST)
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
        (base / 'models' / 'unet' / 'present.safetensors').write_bytes(_ST)
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader'})  # no ConditioningKrea2Rebalance
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}},
              '30': {'class_type': 'ConditioningKrea2Rebalance', 'inputs': {}}}
        with pytest.raises(lts.StudioAssetsMissing) as ei:
            lts.preflight_family('krea', [wf])
        assert ei.value.missing_nodes == ['ConditioningKrea2Rebalance']
        assert ei.value.missing_files == []


def test_preflight_family_accepts_registered_rebalance_alias(app, tmp_path, monkeypatch):
    """Le workflow porte la classe canonique ConditioningKrea2Rebalance, mais le ComfyUI
    cible ne l'enregistre QUE sous l'alias Krea2RebalanceConditioning (l'install du dev /
    une variante legacy) : le preflight NE 409 PAS — c'est le MÊME node (cf.
    NODE_CLASS_ALIASES), donc l'exiger sous son autre nom est satisfait."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        (base / 'models' / 'unet' / 'present.safetensors').write_bytes(_ST)
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader', 'Krea2RebalanceConditioning'})
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}},
              '30': {'class_type': 'ConditioningKrea2Rebalance', 'inputs': {}}}
        lts.preflight_family('krea', [wf])  # no raise — the alias satisfies the requirement


def test_shipped_krea_workflows_pin_rebalance_node(app):
    """Node 30 must use the class the published pack registers
    (ConditioningKrea2Rebalance — mots permutés vs l'ancien nom qui 409-bloquait
    tout le studio Krea) AND pin preset=custom + renormalize=false, so our per-layer
    weights + multiplier are honored even on the huwhitememes fork (whose preset
    default 'balanced' ignores per_layer_weights and renormalize=true cancels the
    multiplier). Both txt2img and img2img graphs carry it."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        for path in (lts.WORKFLOW_KREA_TURBO_PATH, lts.WORKFLOW_KREA_IMG2IMG_PATH):
            wf = lts.load_workflow_local(str(path))
            assert wf, f'{path} unreadable'
            node = wf['30']
            assert node['class_type'] == 'ConditioningKrea2Rebalance'
            assert node['inputs']['preset'] == 'custom'
            assert node['inputs']['renormalize'] is False
            # The rebalance vector the service tunes against is still the shipped default.
            assert node['inputs']['per_layer_weights'].endswith('2.5,5.0,1.1,4.0,1.0')


def test_studio_missing_node_hints_names_krea_pack():
    """studio_missing_node_hints maps the Krea rebalance class to an installable pack
    (name + ComfyUI-Manager search + URL); unknown classes get no hint (no error)."""
    from app.services import lora_test_studio as lts
    hints = lts.studio_missing_node_hints(['ConditioningKrea2Rebalance', 'SomeUnknownNode'])
    assert len(hints) == 1
    h = hints[0]
    assert h['class_type'] == 'ConditioningKrea2Rebalance'
    assert h['pack'] == 'ComfyUI-Conditioning-Rebalance'
    assert 'github.com' in h['url'] and h['search']


def test_preflight_family_passes_when_everything_present(app, tmp_path, monkeypatch):
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        (base / 'models' / 'unet' / 'present.safetensors').write_bytes(_ST)
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader'})
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}}}
        lts.preflight_family('zimage', [wf])  # no raise


def test_preflight_family_flags_present_but_invalid_model_file(app, tmp_path, monkeypatch):
    """kostas212 / #help at the Studio layer: a model the built graph references IS on
    disk but is really an HTML licence-gate page saved as .safetensors (or a truncated
    download). It would fail ComfyUI validation and leave every tile SILENTLY empty, so
    the preflight flags it INVALID — a distinct, actionable state from 'missing'
    (delete + re-download, not 'place the file here')."""
    from app.services import lora_test_studio as lts, model_integrity
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        # Present, real-looking name — but the bytes are the HTML gate page.
        (base / 'models' / 'unet' / 'present.safetensors').write_bytes(
            b'<!doctype html><html>Access to this model is gated</html>')
        model_integrity.clear_cache()
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes',
                            lambda *a, **k: {'UNETLoader'})
        wf = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'present.safetensors'}}}
        with pytest.raises(lts.StudioAssetsMissing) as ei:
            lts.preflight_family('zimage', [wf])
        e = ei.value
        assert e.missing_files == []                 # NOT missing — it is on disk
        assert len(e.invalid_files) == 1
        assert e.invalid_files[0]['path'] == 'models/unet/present.safetensors'
        assert e.invalid_files[0]['kind'] == 'diffusion model'
        assert 'HTML' in e.invalid_files[0]['reason']


def test_preflight_object_info_unreachable_fails_open_on_nodes(app, tmp_path, monkeypatch):
    """When /object_info can't be fetched (None), the node check is SKIPPED (fail-open)
    — never block a launch on a transient probe failure; the per-tile error capture
    (P0-b) still surfaces a genuinely-missing node at runtime."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'unet').mkdir(parents=True)
        (base / 'models' / 'unet' / 'present.safetensors').write_bytes(_ST)
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
        (te_dir / 'qwen_3_4b.safetensors').write_bytes(_ST)
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
        (lora_dir / 'lora_pf_000002000.safetensors').write_bytes(_ST)
        unet_dir = base / 'models' / 'unet' / 'z image'
        unet_dir.mkdir(parents=True)
        (unet_dir / 'zmodel.safetensors').write_bytes(_ST)
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
    _missing, _invalid, classes = lts._scan_workflow_assets(data, None)
    assert 'HttpNotifyNode' not in classes
    assert 'SaveImage' in classes
    assert 'DetailDaemonSamplerNode' in classes


# --- Dev-layout independence: SDXL DMD2 accelerator (resolve or bypass) --------

def test_sdxl_accelerator_resolves_dmd2_under_any_loras_subfolder(app, tmp_path, monkeypatch):
    """The SDXL HQ workflow wires the DMD2 accelerator under the DEV's own 'DMD2\\'
    subfolder. A user who keeps the public DMD2 LoRA under ANY other folder must still
    get it wired — _apply_sdxl_accelerator resolves it by canonical basename across the
    loras roots, not the dev's exact path."""
    import os
    from app.services import lora_test_studio as lts, comfy_model_paths
    from app.utils.comfyui import load_workflow_local
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        acc = base / 'models' / 'loras' / 'accel'
        acc.mkdir(parents=True)
        (acc / 'dmd2_sdxl_4step_lora_fp16.safetensors').write_bytes(_ST)
        comfy_model_paths.clear_cache()
        wf = load_workflow_local(str(lts.WORKFLOW_HQ_PATH))
        lts._apply_sdxl_accelerator(wf)
        dmd2 = [n for n in wf.values()
                if isinstance(n, dict) and n.get('class_type') == 'LoraLoader'
                and 'dmd2' in (n['inputs'].get('lora_name') or '').lower()]
        assert len(dmd2) == 1                                      # loader still present
        rel = dmd2[0]['inputs']['lora_name']
        assert os.path.basename(rel) == 'dmd2_sdxl_4step_lora_fp16.safetensors'
        assert 'accel' in rel.lower() and 'DMD2' not in rel        # found where it lives


def test_sdxl_accelerator_bypasses_dmd2_when_absent(app, tmp_path, monkeypatch):
    """No DMD2 LoRA anywhere on disk → the accelerator loader is BYPASSED (not left to
    fail ComfyUI validation on a personal file), and its model+clip consumers are
    rewired to its own upstream so the SDXL grid renders instead of the whole family
    hard-blocking (mirrors the Klein node-139 bypass)."""
    from app.services import lora_test_studio as lts, comfy_model_paths
    from app.utils.comfyui import load_workflow_local
    with app.app_context():
        _configure_comfy(tmp_path, monkeypatch)                    # empty loras tree
        comfy_model_paths.clear_cache()
        wf = load_workflow_local(str(lts.WORKFLOW_HQ_PATH))
        up_model = wf['10']['inputs']['model']                     # ['25', 0]
        up_clip = wf['10']['inputs']['clip']                       # ['25', 1]
        lts._apply_sdxl_accelerator(wf)
        assert '10' not in wf                                      # DMD2 loader removed
        assert not any(isinstance(n, dict) and n.get('class_type') == 'LoraLoader'
                       and 'dmd2' in (n['inputs'].get('lora_name') or '').lower()
                       for n in wf.values())
        # Model consumers (KSampler 5, BasicScheduler 57, BasicGuider 58) read the
        # removed node's upstream model; the clip consumer (node 3) reads its clip.
        assert wf['5']['inputs']['model'] == up_model
        assert wf['57']['inputs']['model'] == up_model
        assert wf['58']['inputs']['model'] == up_model
        assert wf['3']['inputs']['clip'] == up_clip


def test_build_cell_workflow_sdxl_missing_dmd2_preflight_passes(app, tmp_path, monkeypatch):
    """End-to-end fresh-user SDXL: base checkpoint + tested LoRA on disk but NOT the
    dev's DMD2 accelerator → the built cell bypasses the accelerator and the family
    preflight raises NOTHING for it (previously the whole SDXL Studio 409-blocked on a
    quality-only accelerator that lives only on the dev's disk)."""
    from app.services import lora_test_studio as lts, comfy_model_paths
    import app.utils.comfyui as cu
    with app.app_context():
        base = _configure_comfy(tmp_path, monkeypatch)
        (base / 'models' / 'checkpoints').mkdir(parents=True)
        (base / 'models' / 'checkpoints' / 'mybase.safetensors').write_bytes(_ST)
        lora_dir = base / 'models' / 'loras' / 'sdxl'
        lora_dir.mkdir(parents=True)
        (lora_dir / 'lora_nova_000001000.safetensors').write_bytes(_ST)   # tested LoRA present
        comfy_model_paths.clear_cache()
        tested = 'sdxl\\lora_nova_000001000.safetensors'
        monkeypatch.setattr(lts, 'get_sdxl_loras', lambda: [{'filename': tested}])
        monkeypatch.setattr(cu, 'get_checkpoint_models',
                            lambda *a, **k: [{'name': 'mybase.safetensors'}])
        monkeypatch.setattr(lts, 'resolve_checkpoint_ckpt_name', lambda n: 'mybase.safetensors')
        # Fail-open on the node probe → isolate the model-FILE check.
        monkeypatch.setattr('app.utils.comfyui.fetch_object_info_classes', lambda *a, **k: None)
        wf = lts._build_cell_workflow(
            user_id='local', checkpoint=tested, strength=1.0, prompt='p', seed=1,
            z_model='mybase.safetensors', allowed_loras={tested}, dataset_id=1,
            train_type='sdxl', trigger_word=None)
        assert '10' not in wf                       # accelerator bypassed in the real build
        lts.preflight_family('sdxl', [wf])          # no StudioAssetsMissing for the DMD2 file


def test_embedded_workflow_model_refs_are_all_layout_independent():
    """AUDIT GUARD against dev-layout dependencies. Every model-file reference in every
    embedded workflow must be neutralised by a handler that makes it independent of the
    developer's own ComfyUI layout — an OVERRIDE (a user pick replaces it), a canonical
    RESOLVER, a BYPASS-when-absent, or a preflight 409 ('place X here'). A NEW or CHANGED
    hardcoded ref that no handler covers shows up as a diff and fails this test, so
    category-d (a silent dependency on a file that only exists on the dev's disk) cannot
    creep back. When you add/change a workflow model ref, add it here WITH its handler.

    Handlers (all layout-independent):
      OVERRIDDEN           - the builder always replaces this ref with the user's pick
                             (base checkpoint / UNET, or the LoRA under test).
      RESOLVED             - resolved canonically against disk before enqueue (the Klein
                             UNET/VAE/TE resolvers; the SDXL DMD2 accelerator).
      BYPASSED             - dropped from the graph when its file is absent (the Klein
                             node-139 base LoRA; the SDXL DMD2 accelerator).
      PREFLIGHT_DOCUMENTED - graph-critical family asset, blocked by the Studio preflight
                             409 AND documented in the guide/README (Z-Image VAE + TE).
      PREFLIGHT            - graph-critical family asset, blocked by the Studio preflight
                             409 (Krea VAE + TE).
      DORMANT              - workflow ships but is not wired into any run path yet.
    """
    import json, os
    ALLOWED = {'OVERRIDDEN', 'RESOLVED', 'BYPASSED', 'PREFLIGHT_DOCUMENTED',
               'PREFLIGHT', 'DORMANT'}
    LOADER_KEYS = {
        'UNETLoader': ('unet_name',),
        'CheckpointLoaderSimple': ('ckpt_name',),
        'VAELoader': ('vae_name',),
        'CLIPLoader': ('clip_name',),
        'DualCLIPLoader': ('clip_name1', 'clip_name2'),
        'LoraLoader': ('lora_name',),
        'LoraLoaderModelOnly': ('lora_name',),
    }
    # (workflow basename, node id, input key) -> (expected ref, handler)
    EXPECTED = {
        ('ZImage_bigLove_ZT3_optimal.json', '1', 'unet_name'):
            ('z image\\bigLove_zt3.safetensors', 'OVERRIDDEN'),
        # RESOLVED since bobba84 / GitHub #18: both refs are rewritten to whatever the
        # target ComfyUI actually holds (services/zimage_model_resolver) instead of
        # demanding this exact spelling. They stay PREFLIGHT-documented when nothing
        # resolves — the workflow keeps these values and the 409 names them.
        ('ZImage_bigLove_ZT3_optimal.json', '2', 'clip_name'):
            ('Z image\\qwen_3_4b.safetensors', 'RESOLVED'),
        ('ZImage_bigLove_ZT3_optimal.json', '3', 'vae_name'):
            ('z ae.safetensors', 'RESOLVED'),
        ('image_real_HQ.json', '1', 'ckpt_name'):
            ('Biglove\\mopMixtureOfPervertsDMD_v40.safetensors', 'OVERRIDDEN'),
        ('image_real_HQ.json', '10', 'lora_name'):
            ('DMD2\\dmd2_sdxl_4step_lora_fp16.safetensors', 'RESOLVED'),   # or BYPASSED when absent
        ('image_real_HQ.json', '25', 'lora_name'):
            ('subtle\\subtle-sdxl_enhance.safetensors', 'OVERRIDDEN'),
        ('improve skin.json', '10', 'vae_name'):
            ('flux2_vae.safetensors.safetensors', 'RESOLVED'),
        ('improve skin.json', '90', 'clip_name'):
            ('qwen_3_8b_fp8mixed.safetensors', 'RESOLVED'),
        ('improve skin.json', '114', 'unet_name'):
            ('Flux2 klein\\flux-2-klein-9b-kv-fp8.safetensors', 'RESOLVED'),
        ('improve skin.json', '139', 'lora_name'):
            ('klein\\realistic.safetensors', 'BYPASSED'),
        ('klein_inpaint.json', '114', 'unet_name'):
            ('klein\\flux-2-klein-9b-fp8.safetensors', 'RESOLVED'),
        ('klein_inpaint.json', '10', 'vae_name'):
            ('flux2-vae.safetensors', 'RESOLVED'),
        ('klein_inpaint.json', '90', 'clip_name'):
            ('qwen_3_8b_fp8mixed.safetensors', 'RESOLVED'),
        ('krea2_turbo.json', '20', 'unet_name'):
            ('Krea\\krea2_turbo_fp8.safetensors', 'OVERRIDDEN'),
        ('krea2_turbo.json', '21', 'clip_name'):
            ('qwen3vl_4b_fp8_scaled.safetensors', 'PREFLIGHT'),
        ('krea2_turbo.json', '22', 'vae_name'):
            ('qwen_image_vae.safetensors', 'PREFLIGHT'),
        ('krea2_turbo_img2img.json', '20', 'unet_name'):
            ('Krea\\krea2_turbo_fp8.safetensors', 'DORMANT'),
        ('krea2_turbo_img2img.json', '21', 'clip_name'):
            ('qwen3vl_4b_fp8_scaled.safetensors', 'DORMANT'),
        ('krea2_turbo_img2img.json', '22', 'vae_name'):
            ('qwen_image_vae.safetensors', 'DORMANT'),
    }
    assert all(cat in ALLOWED for _ref, cat in EXPECTED.values())
    actual = {}
    for p in _all_workflow_files():
        name = os.path.basename(p)
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
        for nid, node in data.items():
            if not isinstance(node, dict):
                continue
            for k in LOADER_KEYS.get(node.get('class_type'), ()):
                ref = node.get('inputs', {}).get(k)
                if isinstance(ref, str) and ref.strip():
                    actual[(name, nid, k)] = ref
    assert actual == {k: ref for k, (ref, _cat) in EXPECTED.items()}


# --- Combine mode: several LoRAs loaded in the SAME generation ----------------

def test_prompt_with_triggers_prefixes_every_trigger_in_order(app):
    """A combined stack injects ALL its triggers, first LoRA first, and never twice."""
    from app.services.lora_test_studio import _prompt_with_triggers
    assert _prompt_with_triggers('portrait', ['aaa', 'bbb']) == 'aaa, bbb, portrait'
    # already present as a whole token -> not repeated (case-insensitive)
    assert _prompt_with_triggers('BBB on a roof', ['aaa', 'bbb']) == 'aaa, BBB on a roof'
    # two LoRAs sharing one trigger collapse to a single token
    assert _prompt_with_triggers('portrait', ['aaa', 'aaa']) == 'aaa, portrait'
    # empty / missing triggers are skipped, a bare string keeps the old behaviour
    assert _prompt_with_triggers('portrait', ['aaa', '', None]) == 'aaa, portrait'
    assert _prompt_with_triggers('portrait', 'aaa') == 'aaa, portrait'
    assert _prompt_with_triggers('portrait', None) == 'portrait'


def test_combine_run_stacks_every_lora_with_its_own_weight_and_all_triggers(
        app, monkeypatch, tmp_path):
    """THE contract of combine mode, read off the arguments the workflow builder gets:
    ONE cell, the tested LoRA at its own weight, every other selected LoRA chained as an
    extra at ITS weight, and every dataset's trigger word carried into the prompt."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        ds_a, cks_a = _studio_fixture(tmp_path, monkeypatch, 'Alpha', 'aaa')
        ds_b = svc.create_dataset(LOCAL_USER, 'Beta', 'bbb')
        name_b = 'lora_bbb_000002000.safetensors'
        cp_b = 'z image' + chr(92) + name_b
        (tmp_path / 'Comfy' / 'models' / 'loras' / 'z image' / name_b).write_bytes(_ST)
        by_ds = {ds_a.id: [{'filename': cks_a[0]}], ds_b.id: [{'filename': cp_b}]}
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda ds, _family=None: by_ds[ds.id])
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
        monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
        monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
        built = []
        monkeypatch.setattr(lts, '_build_cell_workflow',
                            lambda *a, **k: (built.append(k), {'1': {}})[1])
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)

        out = lts.create_comparison_run(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cks_a[0], 'weight': 0.9},
             {'dataset_id': ds_b.id, 'checkpoint': cp_b, 'weight': 0.55}],
            [0.6, 0.8, 1.0],            # sweep axis: meaningless here, must be dropped
            prompt='on a rooftop', count=1, combine=True)

        # ONE cell: the strength sweep is replaced by the per-LoRA weights.
        assert out['created'] == 1 and len(built) == 1
        kw = built[0]
        assert kw['extra_loras'] == [{'filename': cp_b, 'strength': 0.55}]
        assert kw['trigger_word'] == ['aaa', 'bbb']
        assert kw['train_type'] == 'zimage'

        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        assert row.checkpoint == cks_a[0] and row.strength == 0.9
        combined = lts._combined_lora_labels(row)
        assert len(combined) == 1 and combined[0]['weight'] == 0.55
        assert combined[0]['label']


def test_combine_run_rejects_a_checkpoint_its_dataset_does_not_own(
        app, monkeypatch, tmp_path):
    """The extras channel is permissive at build time, so the secondaries of a stack
    are re-validated against the checkpoints their OWN dataset really deployed -
    otherwise a crafted selection would be a path-injection hole into the LoRA folder."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds_a, cks_a = _studio_fixture(tmp_path, monkeypatch, 'Gamma', 'ggg')
        ds_b = svc.create_dataset(LOCAL_USER, 'Delta', 'ddd')
        by_ds = {ds_a.id: [{'filename': cks_a[0]}], ds_b.id: []}
        monkeypatch.setattr(lts, 'list_test_checkpoints',
                            lambda ds, _family=None: by_ds[ds.id])
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
        monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
        monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
        evil = 'z image' + chr(92) + '..' + chr(92) + 'evil.safetensors'
        with pytest.raises(ValueError, match='unknown checkpoint'):
            lts.create_comparison_run(
                LOCAL_USER,
                [{'dataset_id': ds_a.id, 'checkpoint': cks_a[0]},
                 {'dataset_id': ds_b.id, 'checkpoint': evil}],
                [1.0], combine=True)


def test_combine_run_refuses_to_mix_families_and_names_them(app, monkeypatch):
    """Two LoRAs from different families need different bases AND workflows: refused,
    with both family names in the message (the picker blocks it too, belt + braces)."""
    from app.services import lora_test_studio as lts
    from app.config import LOCAL_USER
    with app.app_context():
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
        with pytest.raises(ValueError) as excinfo:
            lts.create_comparison_run(
                LOCAL_USER,
                [{'dataset_id': 1, 'checkpoint': 'krea' + chr(92) + 'lora_a.safetensors'},
                 {'dataset_id': 2, 'checkpoint': 'sdxl' + chr(92) + 'lora_b.safetensors'}],
                [1.0], combine=True)
        msg = str(excinfo.value)
        assert 'Krea 2 Turbo' in msg and 'SDXL' in msg and 'one family per run' in msg


def test_combine_weight_is_clamped_and_defaults_to_one(app):
    from app.services.lora_test_studio import (
        _combine_weight, COMBINE_MAX_WEIGHT, MAX_LORA_STRENGTH)
    # ⚠️ The blend ceiling IS the sweep ceiling, not a second number: the head
    # weight of a combination is validated by build_matrix, so a blend clamp
    # above it would refuse the run instead of clamping it.
    assert COMBINE_MAX_WEIGHT == MAX_LORA_STRENGTH == 5.0
    assert _combine_weight({'weight': 0.5555}) == 0.56
    assert _combine_weight({'weight': 3.4}) == 3.4      # 2.0 was the old wall
    assert _combine_weight({'weight': 9}) == 5.0
    assert _combine_weight({'weight': -3}) == 0.0
    assert _combine_weight({'weight': 'oops'}) == 1.0
    assert _combine_weight({}) == 1.0 and _combine_weight(None) == 1.0


def test_a_blend_weight_above_two_survives_the_whole_launch_path(app, monkeypatch, tmp_path):
    """The regression this pairing exists for: raising the browser ceiling alone
    does not weaken the image, it kills the RUN — the head weight goes through
    build_matrix, which used to reject anything above 4.0."""
    from app.services import lora_test_studio as lts
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        ds_a, cks_a = _studio_fixture(tmp_path, monkeypatch, 'Ada', 'aaa')
        ds_b, cks_b = _studio_fixture(tmp_path, monkeypatch, 'Bea', 'bbb')
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)
        out = lts.create_comparison_run(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cks_a[0], 'weight': 4.5},
             {'dataset_id': ds_b.id, 'checkpoint': cks_b[0], 'weight': 5.0}],
            [1.0], prompt='p', count=1, combine=True)
        assert out['created'] == 1
        cell = LoraTestImage.query.get(out['ids'][0])
        assert cell.strength == 4.5, 'the head weight reaches the cell unclamped'


def test_combine_of_a_single_selection_stays_a_normal_run(app, monkeypatch, tmp_path):
    """`combine` on one LoRA is a no-op: the strength sweep still applies."""
    from app.services import lora_test_studio as lts
    from app.config import LOCAL_USER
    with app.app_context():
        ds, cks = _studio_fixture(tmp_path, monkeypatch, 'Solo', 'sss')
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
        monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
        monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)
        out = lts.create_comparison_run(
            LOCAL_USER, [{'dataset_id': ds.id, 'checkpoint': cks[0]}],
            [0.6, 0.8], prompt='p', count=1, combine=True)
        assert out['created'] == 2


def test_combined_stack_reaches_the_real_zimage_graph_as_chained_loaders(app):
    """End of the chain, on the REAL workflow JSON: two combined LoRAs become two
    chained LoraLoaderModelOnly nodes at their own strengths, and both triggers are
    prefixed to the prompt that reaches the text encoder."""
    from app.services import lora_test_studio as lts
    from app.config import LOCAL_USER
    with app.app_context():
        cp_a = 'z image' + chr(92) + 'lora_aaa.safetensors'
        cp_b = 'z image' + chr(92) + 'lora_bbb.safetensors'
        wf = lts._build_cell_workflow(
            LOCAL_USER, cp_a, 0.9, 'on a rooftop', 42, 'zmodel.safetensors',
            {cp_a}, train_type='zimage',
            extra_loras=[{'filename': cp_b, 'strength': 0.55}],
            trigger_word=['aaa', 'bbb'])
        loaders = {nid: n for nid, n in wf.items()
                   if n.get('class_type') == 'LoraLoaderModelOnly'}
        assert {(n['inputs']['lora_name'], n['inputs']['strength_model'])
                for n in loaders.values()} == {(cp_a, 0.9), (cp_b, 0.55)}
        # chained, not parallel: exactly one loader is fed by another loader
        fed_by_loader = [n for n in loaders.values()
                         if n['inputs']['model'][0] in loaders]
        assert len(fed_by_loader) == 1
        texts = [n['inputs'].get('text') for n in wf.values()
                 if isinstance(n.get('inputs', {}).get('text'), str)]
        assert any(t.startswith('aaa, bbb, on a rooftop') for t in texts)


# --- Stack results view: composition, weight variants, best setting ----------

def _stack_run(lts, svc, LOCAL_USER, tmp_path, monkeypatch, weights):
    """Launch ONE combined run of a fixed two-LoRA stack at `weights`, returning
    (run_id, ds_a, ds_b, checkpoints). Callers relaunch it at other weights to build
    the variant history the stack view is supposed to line up."""
    ds_a, cks_a = _studio_fixture(tmp_path, monkeypatch, 'Alpha', 'aaa')
    ds_b = svc.create_dataset(LOCAL_USER, 'Beta', 'bbb')
    name_b = 'lora_bbb_000002000.safetensors'
    cp_b = 'z image' + chr(92) + name_b
    (tmp_path / 'Comfy' / 'models' / 'loras' / 'z image' / name_b).write_bytes(_ST)
    by_ds = {ds_a.id: [{'filename': cks_a[0]}], ds_b.id: [{'filename': cp_b}]}
    monkeypatch.setattr(lts, 'list_test_checkpoints', lambda ds, _family=None: by_ds[ds.id])
    monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
    monkeypatch.setattr(lts, '_active_run_count', lambda *a: 0)
    monkeypatch.setattr(lts, '_preflight_checkpoint_arch', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_preflight_run', lambda *a, **k: None)
    monkeypatch.setattr(lts, '_target_node_classes', lambda: None)
    monkeypatch.setattr(lts, 'permanent_lora_candidates', lambda _f: [])
    monkeypatch.setattr(lts, '_build_cell_workflow', lambda *a, **k: {'1': {}})
    monkeypatch.setattr(lts, '_enqueue_cell', lambda *a, job_id=None, **k: job_id)

    def launch(w_a, w_b):
        return lts.create_comparison_run(
            LOCAL_USER,
            [{'dataset_id': ds_a.id, 'checkpoint': cks_a[0], 'weight': w_a},
             {'dataset_id': ds_b.id, 'checkpoint': cp_b, 'weight': w_b}],
            [1.0], prompt='on a rooftop', count=1, combine=True)['run_id']

    return launch, ds_a, ds_b, cks_a[0], cp_b


def test_stack_composition_names_every_lora_with_its_weight_and_trigger(
        app, monkeypatch, tmp_path):
    """The panel's whole point: a stacked run must say WHAT is in the stack. The head
    LoRA carries the cell's own strength, the stacked one its weight, and both name
    their trigger word - otherwise a stack is a single anonymous column."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    with app.app_context():
        launch, ds_a, ds_b, cp_a, cp_b = _stack_run(
            lts, svc, LOCAL_USER, tmp_path, monkeypatch, None)
        run_id = launch(0.9, 0.55)
        row = LoraTestImage.query.filter_by(run_id=run_id).one()
        members = lts.stack_of_row(row)
        assert [(m['filename'], m['weight'], m['trigger'], m['head']) for m in members] == [
            (cp_a, 0.9, 'aaa', True), (cp_b, 0.55, 'bbb', False)]
        assert [m['dataset_id'] for m in members] == [ds_a.id, ds_b.id]
        # A comparison (non-combined) cell is NOT a stack and must stay one.
        row.extra_loras = None
        assert lts.stack_of_row(row) is None


def test_stack_variants_line_up_the_relaunches_of_the_same_stack(
        app, monkeypatch, tmp_path):
    """Relaunching the same two LoRAs at other weights must produce COMPARABLE columns:
    same members = same stack, so every relaunch shows up as a weight variant with its
    own vector and its own votes, current run first. A run whose members differ is a
    different stack and must not be mixed in."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.models import LoraTestImage
    from app import db
    with app.app_context():
        launch, ds_a, _ds_b, cp_a, cp_b = _stack_run(
            lts, svc, LOCAL_USER, tmp_path, monkeypatch, None)
        first = launch(1.0, 1.0)
        second = launch(1.0, 0.6)
        # A vote on the first variant, so the summary is not all zeros.
        row_first = LoraTestImage.query.filter_by(run_id=first).one()
        row_first.rating = 1
        row_first.status, row_first.filename = 'done', 'a.png'
        db.session.commit()

        payload = lts.studio_payload_run(LOCAL_USER, second)
        assert [m['weight'] for m in payload['stack']] == [1.0, 0.6]
        variants = payload['stack_variants']
        assert [v['run_id'] for v in variants] == [second, first]
        assert variants[0]['active'] is True and variants[1]['active'] is False
        assert [w['weight'] for w in variants[1]['weights']] == [1.0, 1.0]
        assert variants[1]['likes'] == 1 and variants[1]['dislikes'] == 0
        assert variants[1]['done'] == 1
        # Cells travel with their variant so each column is votable in place.
        assert variants[0]['cells'][0]['id']

        # A run of the SAME head LoRA alone is not a variant of this stack.
        solo = lts.create_comparison_run(
            LOCAL_USER, [{'dataset_id': ds_a.id, 'checkpoint': cp_a}],
            [1.0], prompt='p', count=1)['run_id']
        assert solo not in {v['run_id'] for v in
                            lts.studio_payload_run(LOCAL_USER, second)['stack_variants']}
        # And a comparison run has no stack block at all.
        assert lts.studio_payload_run(LOCAL_USER, solo)['stack'] is None
        assert lts.studio_payload_run(LOCAL_USER, solo)['stack_variants'] == []


def test_best_setting_of_a_stack_stores_its_weights_and_pins_every_member(
        app, monkeypatch, tmp_path):
    """"Best setting" for a stack is its WEIGHTS, not one checkpoint. The head stays in
    `lora_filename`/`strength` so every existing reader keeps working, the stacked LoRAs
    ride along in `stack`, and the delete guard-rail sees them all."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        launch, ds_a, ds_b, cp_a, cp_b = _stack_run(
            lts, svc, LOCAL_USER, tmp_path, monkeypatch, None)
        best = lts.set_best_settings(
            LOCAL_USER, ds_a.id, cp_a, 0.9,
            stack=[{'dataset_id': ds_b.id, 'lora_filename': cp_b, 'weight': 0.55}])
        assert best['lora_filename'] == cp_a and best['strength'] == 0.9
        assert best['stack'] == [{'lora_filename': cp_b, 'dataset_id': ds_b.id,
                                  'weight': 0.55, 'trigger': 'bbb'}]
        ds = svc.get_dataset(LOCAL_USER, ds_a.id)
        assert lts.best_settings_lora_filenames(ds) == [cp_a, cp_b]

        # No stack -> no `stack` key: a mono-LoRA setting is byte-for-byte what it was.
        assert 'stack' not in lts.set_best_settings(LOCAL_USER, ds_a.id, cp_a, 0.9)
        # A member checkpoint its dataset never deployed is refused (path injection).
        evil = 'z image' + chr(92) + '..' + chr(92) + 'evil.safetensors'
        with pytest.raises(ValueError, match='unknown checkpoint in stack'):
            lts.set_best_settings(LOCAL_USER, ds_a.id, cp_a, 0.9,
                                  stack=[{'dataset_id': ds_b.id, 'lora_filename': evil}])


# --- Enhance: prompt enrichment through the existing Ollama client ------------

def test_enhance_test_prompt_uses_the_captioning_ollama_client(app, monkeypatch):
    from app.services import lora_test_studio as lts, vision_ollama, ollama_control
    monkeypatch.setattr(ollama_control, 'ensure_captioning_ready',
                        lambda *a, **k: {'ok': True})
    seen = {}

    def fake_generate(prompt, **kwargs):
        seen['prompt'] = prompt
        return '"  a richer prompt  "'
    monkeypatch.setattr(vision_ollama, 'generate_text_ollama', fake_generate)
    with app.app_context():
        assert lts.enhance_test_prompt('a girl') == 'a richer prompt'
    assert 'a girl' in seen['prompt']
    assert 'trigger word' in seen['prompt']       # the model is told to leave it alone


def test_enhance_test_prompt_fails_loudly_without_ollama(app, monkeypatch):
    """No Ollama = a 409-mapped RuntimeError naming what to do, never a silent no-op
    (generate_text_ollama itself returns '' best-effort - that must not reach the UI)."""
    from app.services import lora_test_studio as lts, ollama_control
    monkeypatch.setattr(ollama_control, 'ensure_captioning_ready',
                        lambda *a, **k: {'ok': False, 'error': 'Ollama could not start'})
    with app.app_context():
        with pytest.raises(RuntimeError, match='Ollama could not start'):
            lts.enhance_test_prompt('a girl')


def test_enhance_test_prompt_rejects_empty_and_oversized_prompts(app):
    from app.services import lora_test_studio as lts
    with app.app_context():
        for bad in ('', '   ', None):
            with pytest.raises(ValueError, match='nothing to enhance'):
                lts.enhance_test_prompt(bad)
        with pytest.raises(ValueError, match='too long'):
            lts.enhance_test_prompt('x' * (lts.STUDIO_ENHANCE_MAX_CHARS + 1))


def test_enhance_test_prompt_carries_the_real_refusal_not_an_empty_answer(app, monkeypatch):
    """A refusal from the local-Ollama fence must reach the user WORDED AS ITSELF.

    The fence blocks when a model is loaded outside LDS; the cure is to unload it.
    generate_text_ollama used to collapse every failure to "" best-effort, so the
    Studio could only report "the model returned an empty prompt - check the
    configured Ollama model in Settings" - pointing at a setting that was correct
    and hiding the one action that fixes it.
    """
    from app.services import lora_test_studio as lts, vision_ollama, ollama_control
    monkeypatch.setattr(ollama_control, 'ensure_captioning_ready',
                        lambda *a, **k: {'ok': True})

    def blocked(url, model, keep_alive=None):
        raise vision_ollama.LocalOllamaFenceError(
            'A local Ollama model is already in use outside LDS. LDS will not change '
            'it; unload it first or configure a dedicated Ollama endpoint for LDS.')
    monkeypatch.setattr(vision_ollama, '_admit_local_ollama', blocked)

    with app.app_context():
        # The TYPE survives too, not just the sentence: the route turns it into
        # the `ollama_fence_blocked` code that earns the unload button.
        with pytest.raises(vision_ollama.LocalOllamaFenceError,
                           match='already in use outside LDS'):
            lts.enhance_test_prompt('a girl')


def test_generate_text_ollama_stays_best_effort_for_the_batch_captioner(app, monkeypatch):
    """The strict path is opt-in: the caption shortener has a long caption to fall
    back on, so a refusal there must stay a silent "" and never raise."""
    from app.services import vision_ollama

    def blocked(url, model):
        raise vision_ollama.LocalOllamaFenceError('fence says no')
    monkeypatch.setattr(vision_ollama, '_admit_local_ollama', blocked)

    with app.app_context():
        assert vision_ollama.generate_text_ollama('shorten this') == ''


def test_enhance_test_prompt_raises_when_the_model_answers_nothing(app, monkeypatch):
    from app.services import lora_test_studio as lts, vision_ollama, ollama_control
    monkeypatch.setattr(ollama_control, 'ensure_captioning_ready',
                        lambda *a, **k: {'ok': True})
    monkeypatch.setattr(vision_ollama, 'generate_text_ollama', lambda *a, **k: '')
    with app.app_context():
        with pytest.raises(RuntimeError, match='empty prompt'):
            lts.enhance_test_prompt('a girl')
