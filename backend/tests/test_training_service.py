import json
import os
import time

import pytest


def _configure_aitoolkit(tmp_path, monkeypatch, app):
    """Fake ai-toolkit install: venv python + run.py present, dir configured."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    venv_py = root / 'venv' / 'Scripts' / 'python.exe'
    venv_py.write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid

    def wait(self):
        return None


# --- Live progress: log parsing + samples listing -----------------------------

_TQDM = ('lora_t:   2%|▏         | 60/3000 [01:23<1:07:41,  1.38s/it, lr: 1.0e+00 loss: 3.412e-01]\r'
         'lora_t:   3%|▏         | 100/3000 [02:18<1:06:12,  1.37s/it, lr: 1.0e+00 loss: 3.104e-01]\r'
         'lora_t:   5%|▎         | 150/3000 [03:26<1:05:03,  1.37s/it, lr: 1.0e+00 loss: 2.981e-01]')


def test_parse_training_log_extracts_progress():
    from app.services.lora_training import _parse_training_log
    p = _parse_training_log(_TQDM)
    assert p['step'] == 150 and p['total'] == 3000
    assert p['loss'] == pytest.approx(2.981e-01)
    assert p['speed'] == '1.38s/it' or p['speed'] == '1.37s/it'
    assert p['eta'] == '1:05:03'
    assert p['loss_curve'] == [[60, pytest.approx(0.3412)],
                               [100, pytest.approx(0.3104)],
                               [150, pytest.approx(0.2981)]]


def test_parse_training_log_ignores_incidental_ratios():
    """Non-tqdm 'X/Y' text (dataset counts, resolutions) must not be read as
    progress — only segments with a '%|' bar or a loss postfix count."""
    from app.services.lora_training import _parse_training_log
    p = _parse_training_log('Loading dataset: 25/25 images\nresolution 1024/1024\n')
    assert p['step'] is None and p['loss_curve'] == []


def test_parse_training_log_downsamples_curve():
    from app.services import lora_training as lt
    text = '\r'.join(
        f'x:  1%|▏| {i}/9000 [00:01<00:01, 1.0s/it, loss: {0.5 - i * 1e-5:.3e}]'
        for i in range(1, 1001))
    p = lt._parse_training_log(text)
    assert len(p['loss_curve']) == lt._PROG_CURVE_MAX_POINTS
    assert p['loss_curve'][-1][0] == 1000     # last point always kept
    assert p['step'] == 1000 and p['total'] == 9000


def test_training_progress_reads_log_and_samples(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Prog', 'prog')
        run_parent = lt._output_dir() / lt._run_name(ds)
        run_dir = run_parent / f'lora_{lt._safe_trigger(ds)}'
        (run_dir / 'samples').mkdir(parents=True)
        (run_parent / 'training.log').write_text(_TQDM, encoding='utf-8')
        (run_dir / 'samples' / '1738259371342__000000250_0.jpg').write_bytes(b'x')
        (run_dir / 'samples' / '1738259371342__000000500_1.jpg').write_bytes(b'x')
        p = lt.training_progress(LOCAL_USER, ds.id)
    assert p['log_exists'] is True and p['active'] is False
    assert p['step'] == 150 and p['total'] == 3000
    assert [s['step'] for s in p['samples']] == [500, 250]   # newest first


def test_training_progress_no_log_yet(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoLog', 'nolog')
        p = lt.training_progress(LOCAL_USER, ds.id)
    assert p == {'active': False, 'log_exists': False, 'step': None, 'total': None,
                 'loss': None, 'speed': None, 'eta': None, 'loss_curve': [], 'samples': [],
                 'masks_skipped': False}


# --- Disk-space guard ----------------------------------------------------------

def test_assert_free_disk_blocks_when_low(monkeypatch, tmp_path):
    from app.services import lora_training as lt
    monkeypatch.setattr(lt.shutil, 'disk_usage', lambda p: type('u', (), {'free': 2e9})())
    with pytest.raises(ValueError, match='not enough disk space'):
        lt.assert_free_disk(tmp_path, 10, 'a training run')


def test_assert_free_disk_passes_and_never_blocks_on_stat_failure(monkeypatch, tmp_path):
    from app.services import lora_training as lt
    monkeypatch.setattr(lt.shutil, 'disk_usage', lambda p: type('u', (), {'free': 500e9})())
    lt.assert_free_disk(tmp_path, 10, 'x')          # plenty of space -> no raise
    def boom(p): raise OSError('no stat')
    monkeypatch.setattr(lt.shutil, 'disk_usage', boom)
    lt.assert_free_disk(tmp_path, 10, 'x')          # undeterminable -> no raise
    # climbs to the nearest existing parent for a not-yet-created target
    monkeypatch.setattr(lt.shutil, 'disk_usage', lambda p: type('u', (), {'free': 500e9})())
    assert lt.free_disk_gb(tmp_path / 'not' / 'yet' / 'created') == 500.0


def test_launch_training_refuses_on_low_disk(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage', lambda p: type('u', (), {'free': 1e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Low', 'low')
        with pytest.raises(ValueError, match='not enough disk space'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False)


# --- Best-epoch selection (face similarity on the run's samples) --------------

def _prog_dataset_with_samples(app, tmp_path, monkeypatch, name='Best', trigger='best'):
    """Dataset with a reference + a fake run: samples at steps 250/500 and
    checkpoints at 500/1000. Returns (ds, sample_paths_by_step)."""
    import os
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    ds = svc.create_dataset(LOCAL_USER, name, trigger)
    os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
    open(os.path.join(svc._dataset_dir(ds.id), 'ref.webp'), 'wb').write(b'x')
    ds.ref_filename = 'ref.webp'; svc.db.session.commit()
    run_dir = lt._output_dir() / lt._run_name(ds) / f'lora_{lt._safe_trigger(ds)}'
    (run_dir / 'samples').mkdir(parents=True)
    by_step = {}
    for step in (250, 500):
        for idx in (0, 1):
            p = run_dir / 'samples' / f'173__{step:09d}_{idx}.jpg'
            p.write_bytes(b'x')
            by_step.setdefault(step, []).append(str(p))
    (run_dir / f'lora_{lt._safe_trigger(ds)}_000000500.safetensors').write_bytes(b'x')
    (run_dir / f'lora_{lt._safe_trigger(ds)}_000001000.safetensors').write_bytes(b'x')
    return ds, by_step


def test_score_checkpoint_samples_picks_best_step(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, by_step = _prog_dataset_with_samples(app, tmp_path, monkeypatch)
        # step 250 scores 0.40/0.44 ; step 500 scores 0.62/0.58 -> best = 500
        sims = {by_step[250][0]: 0.40, by_step[250][1]: 0.44,
                by_step[500][0]: 0.62, by_step[500][1]: 0.58}
        # score_checkpoint_samples importe face_similarity À L'APPEL → patcher le
        # module suffit (pas de référence figée à contourner).
        from app.services import face_similarity as fsim
        monkeypatch.setattr(fsim, 'is_available', lambda: True)
        monkeypatch.setattr(fsim, 'score_dataset_faces',
                            lambda ref, paths, **kw: ({p: {'state': 'scorable', 'sim': sims[p]}
                                                       for p in paths}, None))
        r = lt.score_checkpoint_samples(LOCAL_USER, ds.id)
    assert r['available'] is True
    assert r['best_step'] == 500
    assert r['steps'] == [{'step': 250, 'mean_sim': 0.42, 'n': 2},
                          {'step': 500, 'mean_sim': 0.6, 'n': 2}]
    assert r['checkpoint'].endswith('_000000500.safetensors')


def test_score_checkpoint_samples_degrades_cleanly(app, tmp_path, monkeypatch):
    """Missing prerequisites answer {'available': False, reason} — never a 500."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoRef', 'noref')
        r = lt.score_checkpoint_samples(LOCAL_USER, ds.id)
        assert r['available'] is False and 'reference' in r['reason']
        import os
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        open(os.path.join(svc._dataset_dir(ds.id), 'ref.webp'), 'wb').write(b'x')
        ds.ref_filename = 'ref.webp'; svc.db.session.commit()
        from app.services import face_similarity as fsim
        monkeypatch.setattr(fsim, 'is_available', lambda: True)
        r = lt.score_checkpoint_samples(LOCAL_USER, ds.id)   # no samples dir yet
        assert r['available'] is False and 'samples' in r['reason']


def test_recommended_steps_clamps(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'T', 't')
        for _ in range(5):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='x.webp'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 1500        # 5*120=600 -> clamp
        for _ in range(35):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='y.webp'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 3500        # 40*120=4800 -> clamp


def test_recommended_steps_concept_scales_sublinearly(app):
    """Concept datasets: √n scaling, not 120/img — a 400-image concept set must
    land near ~9500 steps (~24 views/img), not be clamped to the character 3500."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'C', 'c')
        ds.kind = 'concept'
        for _ in range(9):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='x.webp'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 2000        # 475*3=1425 -> clamp bas
        for _ in range(391):                               # -> 400 images
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='y.webp'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 9500        # 475*20=9500, dans [2000,12000]
        info = lt.recommended_steps_info(ds.id)
        assert info['kind'] == 'concept' and info['steps'] == 9500 and info['n_images'] == 400
        assert 'generalizes' in info['rationale']


def test_style_dataset_job_config_and_steps(app, tmp_path):
    """Style is always-on: no process/prompt trigger, conservative dropout and
    the family/variant timestep recipe remains intact."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'S', 'zsty', kind='style')
        assert svc.is_style(ds) and svc.is_conceptual(ds) and not svc.is_concept(ds)
        assert ds.fidelity is None            # fidelity = personnage uniquement
        for _ in range(100):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep', filename='x.webp'))
        ds.train_settings = json.dumps({'sample_prompts': [
            'zsty, a harbor at dawn', '{trigger}, a quiet forest path']})
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id) == 2000        # Z-Image Turbo cap
        assert lt.recommended_steps_info(ds.id)['kind'] == 'style'
        cfg_ = lt.build_job_config(ds, str(tmp_path), steps=2000)
        proc = cfg_['config']['process'][0]
        assert 'trigger_word' not in proc                  # style = pas de trigger
        assert proc['datasets'][0]['caption_dropout_rate'] == 0.05
        assert proc['train']['timestep_type'] == 'sigmoid'
        assert all('zsty' not in p for p in proc['sample']['prompts'])
        assert proc['sample']['prompts'] == ['a harbor at dawn', 'a quiet forest path']
        snap = lt.launch_settings_snapshot(ds)
        assert snap['style_mode'] == 'always_on'
        assert snap['effective_caption_dropout'] == 0.05
        assert 'trigger' not in snap


def test_style_dataset_captions_required_with_existing_override(app):
    """Blank/whitespace Style captions block unless the explicit override is used."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'S2', 'zsty2', kind='style')
        for i in range(12):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                filename='x.webp',
                                                caption='   ' if i == 0 else None))
        svc.db.session.commit()
        with pytest.raises(ValueError, match='UNCAPTIONED'):
            lt.assert_trainable(ds.id)
        lt.assert_trainable(ds.id, allow_uncaptioned=True)


def test_style_caption_quality_guard_and_override(app):
    """Trigger-only and normalized-identical captions are detected together."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'SQ', 'raw_lds', kind='style')
        variants = (' raw_lds. ', 'RAW_LDS!', ' raw_lds  ', 'raw_lds:')
        for i in range(12):
            svc.db.session.add(FaceDatasetImage(
                dataset_id=ds.id, status='keep', filename=f'{i}.webp',
                caption=variants[i % len(variants)]))
        svc.db.session.commit()
        quality = lt.style_caption_quality(ds.id)
        assert quality['trigger_only_count'] == 12
        assert quality['all_identical'] is True
        with pytest.raises(ValueError, match='CAPTION_QUALITY'):
            lt.assert_trainable(ds.id)
        lt.assert_trainable(ds.id, allow_caption_quality=True)


def test_style_steps_are_family_and_variant_aware(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Step style', 'style_steps', kind='style')
        for i in range(10):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                filename=f'{i}.webp'))
        svc.db.session.commit()
        assert lt.recommended_steps(ds.id, 'flux2klein', '4b') == 1200
        assert lt.recommended_steps(ds.id, 'krea', 'raw') == 2000
        assert lt.recommended_steps(ds.id, 'krea', 'turbo') == 1000
        assert lt.recommended_steps(ds.id, 'krea', 'deturbo') == 1000  # legacy -> Turbo
        assert lt.recommended_steps(ds.id, 'zimage', 'turbo') == 1000
        assert lt.recommended_steps(ds.id, 'zimage', 'base') == 1500
        info = lt.recommended_steps_info(ds.id, train_type='krea', variant='raw')
        assert info['min_steps'] == 2000 and info['max_steps'] == 3000
        assert info['train_type'] == 'krea' and info['variant'] == 'raw'


def test_style_krea_cached_text_has_zero_caption_dropout(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Krea style', 'krea_style', kind='style',
                                train_type='krea')
        ds.train_variant = 'base'
        proc = lt.build_job_config(ds, str(tmp_path), steps=2000,
                                   training_folder='__test__')['config']['process'][0]
        assert 'trigger_word' not in proc
        assert proc['datasets'][0]['cache_text_embeddings'] is True
        assert proc['datasets'][0]['caption_dropout_rate'] == 0.0
        assert proc['train']['timestep_type'] == 'linear'
        snap = lt.launch_settings_snapshot(ds)
        assert snap['effective_caption_dropout'] == 0.0


def test_style_aitoolkit_export_is_content_only(app, tmp_path):
    from PIL import Image
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Export style', 'zsty_export', kind='style')
        filename = 'source.webp'
        Image.new('RGB', (32, 32), (30, 20, 10)).save(
            os.path.join(svc._dataset_dir(ds.id), filename), 'WEBP')
        caption = 'A glass vase on a table beside a bright window.'
        svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                            filename=filename,
                                            caption=f'zsty_export, {caption}'))
        svc.db.session.commit()
        out = lt.export_dataset_to_aitoolkit(
            LOCAL_USER, ds.id, masked=True, dest_dir=tmp_path / 'export')
        sidecar = next((tmp_path / 'export').glob('*.txt'))
        assert sidecar.read_text(encoding='utf-8') == caption
        assert 'zsty_export' not in sidecar.read_text(encoding='utf-8')


def test_style_captioned_still_checks_prose_booru_mismatch(app):
    """Audit fix: the style skip covers ONLY missing captions — when captions DO
    exist, a prose-captioned style dataset trained as SDXL must still raise the
    MISMATCH_CAPTION guard (same failure mode as a character dataset)."""
    import pytest
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'S3', 'zsty3', kind='style', train_type='sdxl')
        prose = 'A woman reading a book in a sunlit cafe, sitting by the window.'
        # 20 kept = the SDXL family floor, so this isolates the caption-mismatch
        # guard from the readiness image-floor guard (which now fires below 20).
        for i in range(20):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                filename='x.webp', caption=f'{prose} Scene {i}.'))
        svc.db.session.commit()
        with pytest.raises(ValueError, match='MISMATCH_CAPTION'):
            lt.assert_trainable(ds.id, train_type='sdxl')
        lt.assert_trainable(ds.id, train_type='sdxl', allow_caption_mismatch=True)


def test_style_default_trigger_salted_no_collision(app):
    """Audit fix: two styles created WITHOUT a trigger must not both land on the
    'zchar' default (the anti-collision guard would block the 2nd training run)."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        a = svc.create_dataset(LOCAL_USER, 'SA', '', kind='style')
        b = svc.create_dataset(LOCAL_USER, 'SB', '', kind='style')
        assert a.trigger_word == f'zsty_{a.id}'
        assert b.trigger_word == f'zsty_{b.id}'
        assert a.trigger_word != b.trigger_word
        c = svc.create_dataset(LOCAL_USER, 'SC', 'zsty_ink', kind='style')
        assert c.trigger_word == 'zsty_ink'   # un trigger explicite est respecté


def test_style_preview_strips_only_legacy_trigger_prefix(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Ink', 'ink', kind='style')
        ds.train_settings = json.dumps({'sample_prompts': [
            'an ink illustration on paper',
            'ink, a portrait under window light',
            '{trigger}, a mountain landscape',
        ]})
        svc.db.session.commit()
        prompts = lt._sample_prompts(ds, 'ink')
    assert prompts == [
        'an ink illustration on paper',
        'a portrait under window light',
        'a mountain landscape',
    ]


def test_job_config_masked_fields_only_when_masks_exist(app, tmp_path):
    from app.services import lora_training as lt
    folder = tmp_path / 'ds'
    folder.mkdir()
    assert lt._mask_fields(str(folder)) == {}
    masks = tmp_path / 'ds_masks'
    masks.mkdir()
    (masks / 'a.png').touch()
    assert lt._mask_fields(str(folder)) == {'mask_path': str(masks), 'mask_min_value': 0.1}


def test_enqueue_snapshots_steps_and_not_before(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Q', 'q')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        lt.enqueue_training(LOCAL_USER, ds.id, steps=2000, not_before='2999-01-01T00:00',
                            allow_caption_mismatch=True, allow_uncaptioned=True,
                            allow_caption_quality=True)
        q = lt.get_train_queue()
        assert q[0]['steps'] == 2000 and q[0]['not_before'].startswith('2999')
        assert q[0]['allow_caption_mismatch'] is True
        assert q[0]['allow_uncaptioned'] is True
        assert q[0]['allow_caption_quality'] is True
        assert lt._due_index(q) is None                   # scheduled in the future -> not due

        captured = {}
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **kw: captured.update(kw) or {'started': True})
        lt._launch_queued_item(q[0])
        assert captured['allow_caption_mismatch'] is True
        assert captured['allow_uncaptioned'] is True
        assert captured['allow_caption_quality'] is True


def test_enqueue_snapshots_and_replays_allow_not_ready(app, monkeypatch):
    """The « Continue anyway » ack rides the queue item and is replayed to the
    launch when the job reaches the GPU (parity with the caption flags)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'QN', 'qn')
        monkeypatch.setattr(lt, 'assert_trainable', lambda *a, **k: None)
        lt.enqueue_training(LOCAL_USER, ds.id, steps=1000, allow_not_ready=True)
        q = lt.get_train_queue()
        assert q[0]['allow_not_ready'] is True

        captured = {}
        monkeypatch.setattr(lt, 'launch_training',
                            lambda *a, **kw: captured.update(kw) or {'started': True})
        lt._launch_queued_item(q[0])
        assert captured['allow_not_ready'] is True


def test_continue_revalidates_current_dataset_then_forwards_confirmations(
        app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Resume guarded', 'resume_guarded')
        calls = []
        monkeypatch.setattr(
            lt, 'assert_trainable',
            lambda dataset_id, **kw: calls.append(('preflight', dataset_id, kw)))
        monkeypatch.setattr(
            lt, 'list_checkpoints',
            lambda *a, **kw: [{'step': 750, 'filename': 'resume.safetensors'}])
        monkeypatch.setattr(
            lt, 'launch_training',
            lambda *a, **kw: calls.append(('launch', kw)) or {'started': True})

        result = lt.continue_training(
            LOCAL_USER, ds.id, extra_steps=500,
            allow_caption_mismatch=True, allow_uncaptioned=True,
            allow_caption_quality=True)

    assert result['target_steps'] == 1250
    assert calls[0][0] == 'preflight'
    assert calls[0][2]['allow_caption_mismatch'] is True
    assert calls[0][2]['allow_uncaptioned'] is True
    assert calls[0][2]['allow_caption_quality'] is True
    assert calls[1][0] == 'launch'
    assert calls[1][1]['check_captions'] is False
    assert calls[1][1]['allow_caption_mismatch'] is True
    assert calls[1][1]['allow_uncaptioned'] is True
    assert calls[1][1]['allow_caption_quality'] is True


def test_continue_and_queued_resume_block_if_dataset_degraded(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Resume degraded', 'resume_degraded')
        launched = []
        monkeypatch.setattr(
            lt, 'assert_trainable',
            lambda *a, **kw: (_ for _ in ()).throw(
                ValueError('UNCAPTIONED: dataset changed since the checkpoint')))
        monkeypatch.setattr(
            lt, 'list_checkpoints', lambda *a, **kw: [{'step': 750}])
        monkeypatch.setattr(
            lt, 'launch_training', lambda *a, **kw: launched.append(kw))

        with pytest.raises(ValueError, match='UNCAPTIONED'):
            lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
        with pytest.raises(ValueError, match='UNCAPTIONED'):
            lt.enqueue_training(LOCAL_USER, ds.id, extra_steps=500)
        assert launched == []
        assert lt.get_train_queue() == []


def test_queued_resume_replays_confirmation_flags(monkeypatch):
    from app.services import lora_training as lt
    captured = {}
    monkeypatch.setattr(
        lt, 'continue_training',
        lambda *a, **kw: captured.update(kw) or {'started': True})
    lt._launch_queued_item({
        'dataset_id': 9, 'user_id': 'local', 'extra_steps': 500,
        'base_model': '', 'variant': 'base', 'train_type': 'zimage',
        'allow_caption_mismatch': True,
        'allow_uncaptioned': True,
        'allow_caption_quality': True,
    })
    assert captured['allow_caption_mismatch'] is True
    assert captured['allow_uncaptioned'] is True
    assert captured['allow_caption_quality'] is True


def test_step_cap_floor_500(app, tmp_path, monkeypatch):
    """launch_training(steps=200) floors to 500; the written job config reflects it."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    _configure_aitoolkit(tmp_path, monkeypatch, app)

    captured = {}

    def fake_popen(args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return _FakeProc()

    monkeypatch.setattr(lt.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(lt, '_watch_training', lambda *a, **k: None)

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Floor', 'floortrig')
        for i in range(12):
            svc.db.session.add(lt.FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                   filename=f'x{i}.webp', caption='a caption here'))
        svc.db.session.commit()

        def fake_export(user_id, dataset_id, masked=True):
            folder = tmp_path / 'exported'
            folder.mkdir(exist_ok=True)
            return str(folder)

        monkeypatch.setattr(lt, 'export_dataset_to_aitoolkit', fake_export)

        result = lt.launch_training(LOCAL_USER, ds.id, steps=200, masked=False)
        assert result['steps'] == 500

        with open(result['config_path'], encoding='utf-8') as fh:
            config = json.load(fh)
        train_cfg = config['config']['process'][0]['train']
        assert train_cfg['steps'] == 500
        ds0 = config['config']['process'][0]['datasets'][0]
        assert 'mask_path' not in ds0


def test_masked_training_zero_written_disables_masks(app, tmp_path, monkeypatch):
    """generate_person_masks returns a dict with written=0 -> masks dir removed,
    masked training disabled (adaptation for the dict-shaped contract)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        from app import config as cfg
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Mask0', 'masktrig0')
        img_dir = svc._dataset_dir(ds.id)
        for i in range(3):
            fn = f'k{i}.png'
            from PIL import Image
            Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
            svc.db.session.add(lt.FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        svc.db.session.commit()

        monkeypatch.setattr(lt, 'generate_person_masks',
                            lambda paths, out_dir: {'ok': True, 'written': 0, 'results': {}})

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        masks_dir = lt._masks_dir(out)
        assert not os.path.isdir(masks_dir)


def test_masked_training_written_nonzero_keeps_masks(app, tmp_path, monkeypatch):
    """generate_person_masks returns written>0 -> masks dir survives, masked training stays on."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        from app import config as cfg
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Mask1', 'masktrig1')
        img_dir = svc._dataset_dir(ds.id)
        for i in range(3):
            fn = f'k{i}.png'
            from PIL import Image
            Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
            svc.db.session.add(lt.FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn))
        svc.db.session.commit()

        def fake_masks(paths, out_dir):
            os.makedirs(out_dir, exist_ok=True)
            for p in paths:
                base = os.path.splitext(os.path.basename(p))[0]
                open(os.path.join(out_dir, base + '.png'), 'wb').close()
            return {'ok': True, 'written': len(paths), 'results': {p: 'ok' for p in paths}}

        monkeypatch.setattr(lt, 'generate_person_masks', fake_masks)

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True)
        masks_dir = lt._masks_dir(out)
        assert os.path.isdir(masks_dir)
        assert lt._mask_fields(out) == {'mask_path': masks_dir, 'mask_min_value': 0.1}


def test_launch_training_unconfigured_aitoolkit_raises_runtime_error(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'NoAI', 'noaitrig')
        with pytest.raises(RuntimeError):
            lt.launch_training(LOCAL_USER, ds.id)


def test_build_job_config_masked_false_no_mask_path(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'NoMask', 'nomasktrig')
        folder = tmp_path / 'plain'
        folder.mkdir()
        config = lt.build_job_config(ds, str(folder), steps=1500)
        ds0 = config['config']['process'][0]['datasets'][0]
        assert 'mask_path' not in ds0


def test_continue_training_refuses_while_in_progress(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app.job_queue import queue_manager
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Busy', 'busytrig')
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=3600)
        with pytest.raises(ValueError):
            lt.continue_training(LOCAL_USER, ds.id)


def test_process_training_queue_rearms_ttl_while_pid_alive(app, monkeypatch):
    """A training run longer than the state TTL (_TRAIN_STATE_TTL, 12h — long
    enough for a Krea-2-Raw run) must not have its flags expire mid-run: each
    poll while the pid is still alive re-arms them."""
    from app.services import lora_training as lt
    from app.job_queue import queue_manager
    with app.app_context():
        queue_manager._set_system_state('training_in_progress', True, ttl_seconds=1)
        queue_manager._set_system_state('training_pid', 4242, ttl_seconds=1)
        queue_manager._set_system_state('training_dataset_id', 7, ttl_seconds=1)
        queue_manager._set_system_state('training_target_step', 1500, ttl_seconds=1)
        queue_manager._set_system_state('training_run_token', 'long-run-token', ttl_seconds=1)
        queue_manager._set_system_state('training_variant', 'deturbo', ttl_seconds=1)
        queue_manager._set_system_state('training_train_type', 'zimage', ttl_seconds=1)
        monkeypatch.setattr(lt, '_pid_alive', lambda pid: True)

        assert lt.process_training_queue() is None  # still running -> no action taken

        time.sleep(1.2)  # past the ORIGINAL (pre-rearm) TTL
        assert queue_manager._get_system_state('training_in_progress', False) is True
        assert queue_manager._get_system_state('training_pid', None) == 4242
        assert queue_manager._get_system_state('training_dataset_id', None) == 7
        assert queue_manager._get_system_state('training_target_step', None) == 1500
        assert queue_manager._get_system_state('training_run_token', None) == 'long-run-token'
        assert queue_manager._get_system_state('training_variant', None) == 'deturbo'


def test_import_list_delete_checkpoint_roundtrip_filesystem_scan(app, tmp_path):
    """import_checkpoint copies into the ComfyUI loras dir (ownership stripped:
    no lora_ownership call); list_imported_checkpoints finds it via a plain
    filesystem scan filtered by trigger boundary (not the old ownership-filtered
    list_test_checkpoints call); delete_imported_checkpoint removes it, confined
    to the family's deploy folder."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg

    with app.app_context():
        aitoolkit_dir = tmp_path / 'aitoolkit'
        comfy_dir = tmp_path / 'comfy'
        cfg.save_config({'aitoolkit': {'dir': str(aitoolkit_dir)},
                         'comfyui': {'base_dir': str(comfy_dir)}})
        ds = svc.create_dataset(LOCAL_USER, 'Import', 'ImportTrig')

        # Fake ai-toolkit run dir with one numbered checkpoint (list_checkpoints reads this).
        run_dir = (lt._output_dir() / lt._run_name(ds)
                   / 'lora_ImportTrig')
        run_dir.mkdir(parents=True)
        ck_name = 'lora_ImportTrig_000001500.safetensors'
        (run_dir / ck_name).write_bytes(b'fake-weights')

        assert [c['filename'] for c in lt.list_checkpoints(LOCAL_USER, ds.id)] == [ck_name]
        assert lt.list_imported_checkpoints(LOCAL_USER, ds.id) == []  # not deployed yet

        dest = lt.import_checkpoint(LOCAL_USER, ds.id, ck_name)
        assert os.path.isfile(dest)
        assert 'z image' in dest  # zimage is the default family

        imported = lt.list_imported_checkpoints(LOCAL_USER, ds.id)
        assert len(imported) == 1
        deployed_name = os.path.basename(dest)
        assert deployed_name.endswith('_Z-Image-Turbo.safetensors')
        assert imported[0]['filename'] == os.path.join('z image', deployed_name)
        assert imported[0]['label']  # format_trained_lora_label produced something non-empty

        removed_name = lt.delete_imported_checkpoint(LOCAL_USER, ds.id, imported[0]['filename'])
        assert removed_name == deployed_name
        assert not os.path.isfile(dest)
        assert lt.list_imported_checkpoints(LOCAL_USER, ds.id) == []


def test_imported_list_shows_cloud_named_checkpoints(app, tmp_path):
    """Cloud-trained LoRA land in the same ComfyUI folder but named after the
    pod job (lds<N>_…), not lora_<trigger>… — the trigger-boundary filter hid
    them from the "IN COMFYUI" list even though the files were there
    (user-observed 2026-07-13). A filename that IS a known cloud checkpoint of
    this dataset must be listed."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    from app.extensions import db
    from app.models import CloudTrainingRun

    with app.app_context():
        comfy_dir = tmp_path / 'comfy'
        cfg.save_config({'comfyui': {'base_dir': str(comfy_dir)}})
        ds = svc.create_dataset(LOCAL_USER, 'Cloudy', 'CloudTrig')
        dest_dir = comfy_dir / 'models' / 'loras' / 'z image'
        dest_dir.mkdir(parents=True)
        cloud_name = 'lds9_ulocal_cloudy_000003000.safetensors'
        (dest_dir / cloud_name).write_bytes(b'w')
        # a stranger file with a foreign name must STAY hidden
        (dest_dir / 'other_dataset_lora.safetensors').write_bytes(b'w')
        run = CloudTrainingRun(dataset_id=ds.id, status='done', job_name='j',
                               vast_label='lds-9',
                               checkpoint_local_path=str(tmp_path / cloud_name))
        db.session.add(run)
        db.session.commit()
        names = [c['filename'] for c in lt.list_imported_checkpoints(LOCAL_USER, ds.id)]
        assert os.path.join('z image', cloud_name) in names
        assert not any('other_dataset' in n for n in names)


def test_default_variant_is_family_aware():
    """Raw is the default ONLY for Krea (official « train on Raw » reco); every
    other family keeps Turbo. This is what makes « Raw par défaut » hold on the
    launch/queue paths even when no variant is passed."""
    from app.services import lora_training as lt
    assert lt._default_variant_for('krea') == 'base'      # -> Krea-2-Raw
    assert lt._default_variant_for('zimage') == 'turbo'
    assert lt._default_variant_for('sdxl') == 'turbo'
    assert lt._default_variant_for(None) == 'turbo'


def test_explicit_variant_controls_krea_and_flux2_run_tags(app):
    """Cloud/import/mirror replay a stamped variant and must not fall through
    to a later mutable dataset selection."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        krea = svc.create_dataset(
            LOCAL_USER, 'Krea stale', 'krea_stale', train_type='krea')
        krea.train_variant = 'turbo'
        assert lt._run_name(
            krea, base_model='', family='krea', variant='base'
        ).endswith('_Krea-2-Raw')
        assert lt._run_name(
            krea, base_model='', family='krea', variant='turbo'
        ).endswith('_Krea-2-Turbo')

        klein = svc.create_dataset(
            LOCAL_USER, 'Klein stale', 'klein_stale',
            train_type='flux2klein')
        klein.train_variant = '4b'
        assert lt._run_name(
            klein, base_model='', family='flux2klein', variant='9b'
        ).endswith('_FLUX2-Klein-9B')
        assert lt._run_name(
            klein, base_model='', family='flux2klein', variant='4b'
        ).endswith('_FLUX2-Klein-4B')


def test_zimage_official_recipe_matrix_and_isolated_run_names(app, tmp_path):
    """Each official Z-Image variant must resolve to its real checkpoint and
    adapter contract; incompatible variants must never share an auto-resume dir."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Zed', 'zed', train_type='zimage')
        folder = tmp_path / 'dataset'; folder.mkdir()

        expected = {
            'turbo': (lt.ZIMAGE_TURBO_BASE, lt.ZIMAGE_TURBO_TRAINING_ADAPTER,
                      None, 8, 1, 'sigmoid'),
            'base': (lt.ZIMAGE_BASE, None, None, 35, 4, 'weighted'),
            'deturbo': (lt.ZIMAGE_DETURBO_BASE, None, lt.ZIMAGE_TURBO_BASE,
                        25, 3, 'weighted'),
        }
        names = {}
        for variant, (base, adapter, extras, sample_steps, cfg, timestep) in expected.items():
            ds.train_variant = variant
            process = lt.build_job_config(
                ds, str(folder), steps=1500, training_folder='__test__'
            )['config']['process'][0]
            model = process['model']
            assert model['name_or_path'] == base
            assert model.get('assistant_lora_path') == adapter
            assert model.get('extras_name_or_path') == extras
            assert process['sample']['sample_steps'] == sample_steps
            assert process['sample']['guidance_scale'] == cfg
            assert process['train']['timestep_type'] == timestep
            names[variant] = lt._run_name(ds)

        assert len(set(names.values())) == 3
        assert names['turbo'].endswith('_Z-Image-Turbo')
        assert names['base'].endswith('_Z-Image-Base')
        assert names['deturbo'].endswith('_Z-Image-De-Turbo')


def test_zimage_recipe_validation_provenance_and_legacy_annotation(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with pytest.raises(ValueError, match='invalid Z-Image variant'):
        lt.zimage_training_recipe('typo')

    custom = lt.zimage_training_recipe('turbo', 'my_merge.safetensors')
    assert custom['effective_base'] == 'my_merge.safetensors'
    assert custom['extras_name_or_path'] == lt.ZIMAGE_TURBO_BASE
    assert custom['training_adapter'] == lt.ZIMAGE_TURBO_TRAINING_ADAPTER

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Safe', 'safe', train_type='zimage')
        ds.train_variant = 'deturbo'
        snap = lt.launch_settings_snapshot(ds)
        assert snap['recipe_version'] == lt.ZIMAGE_RECIPE_VERSION
        assert snap['effective_base'] == lt.ZIMAGE_DETURBO_BASE
        assert snap['training_adapter'] is None

    legacy = lt.zimage_recipe_diagnostic('zimage', 'deturbo')
    assert legacy['status'] == 'legacy_incompatible'
    assert 'not stopped or modified' in legacy['warning']
    safe = lt.zimage_recipe_diagnostic(
        'zimage', 'deturbo', snap['effective_base'], snap['training_adapter'],
        snap['recipe_version'])
    assert safe == {'status': 'safe', 'warning': None}


def test_zimage_custom_base_confirmation_and_full_identifier_hash(app):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    for variant in ('base', 'deturbo'):
        with pytest.raises(ValueError, match='^CUSTOM_WEIGHTS_UNVERIFIED:'):
            lt.assert_zimage_custom_recipe_confirmed(
                'zimage', r'merges\model.safetensors', variant)
        lt.assert_zimage_custom_recipe_confirmed(
            'zimage', r'merges\model.safetensors', variant,
            allow_unverified_weights=True)
    # Custom Turbo keeps its adapter and does not need this distillation-type
    # acknowledgement.
    lt.assert_zimage_custom_recipe_confirmed(
        'zimage', r'merges\model.safetensors', 'turbo')

    with app.app_context():
        ds = svc.create_dataset(
            LOCAL_USER, 'Hash', 'hashy', train_type='zimage')
        first = lt._run_name(
            ds, r'folder-a\same.safetensors', 'zimage', 'turbo')
        second = lt._run_name(
            ds, r'folder-b\same.safetensors', 'zimage', 'turbo')
        absolute = lt._run_name(
            ds, r'C:\weights\same.safetensors', 'zimage', 'turbo')
        assert len({first, second, absolute}) == 3
        assert all('_h' in name for name in (first, second, absolute))


def test_variant_scopes_local_checkpoint_sample_and_progress_paths(
        app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    _configure_aitoolkit(tmp_path, monkeypatch, app)
    with app.app_context():
        ds = svc.create_dataset(
            LOCAL_USER, 'Variants', 'variants', train_type='zimage')
        base_dir = lt._run_dir(
            LOCAL_USER, ds.id, base_model='', family='zimage', variant='base')
        deturbo_dir = lt._run_dir(
            LOCAL_USER, ds.id, base_model='', family='zimage', variant='deturbo')
        assert base_dir != deturbo_dir
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(deturbo_dir, exist_ok=True)
        open(os.path.join(base_dir, 'lora_variants_000000500.safetensors'),
             'wb').write(b'base')
        open(os.path.join(deturbo_dir, 'lora_variants_000001000.safetensors'),
             'wb').write(b'deturbo')
        assert [c['step'] for c in lt.list_checkpoints(
            LOCAL_USER, ds.id, '', 'zimage', 'base')] == [500]
        assert [c['step'] for c in lt.list_checkpoints(
            LOCAL_USER, ds.id, '', 'zimage', 'deturbo')] == [1000]
        assert lt._samples_dir(
            LOCAL_USER, ds.id, '', 'zimage', 'base') != lt._samples_dir(
                LOCAL_USER, ds.id, '', 'zimage', 'deturbo')


def test_training_status_exposes_active_recipe_and_unique_token(app, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds = svc.create_dataset(
            LOCAL_USER, 'Active', 'active', train_type='zimage')
        recipe = lt.zimage_training_recipe('turbo')
        state = {
            'training_in_progress': True,
            'training_dataset_id': ds.id,
            'training_pid': 123,
            'training_run_token': 'opaque-token',
            'training_train_type': 'zimage',
            'training_variant': 'turbo',
            'training_base_model': '',
            'training_effective_base': recipe['effective_base'],
            'training_training_adapter': recipe['training_adapter'],
            'training_recipe_version': recipe['recipe_version'],
        }
        monkeypatch.setattr(
            lt.queue_manager, '_get_system_state',
            lambda key, default=None: state.get(key, default))
        current = lt.training_status()['current']
        assert current['run_token'] == 'opaque-token'
        assert current['variant'] == 'turbo'
        assert current['effective_base'] == lt.ZIMAGE_TURBO_BASE
        assert current['training_adapter'] == lt.ZIMAGE_TURBO_TRAINING_ADAPTER
        assert current['recipe_status'] == 'safe'
        assert current['recipe_warning'] is None


def test_build_job_config_krea_raw_default_and_turbo_optin(app, tmp_path):
    """Krea 2 defaults to the RAW base (non-distilled krea/Krea-2-Raw, NO training
    adapter, previews at CFG 4 / 25 steps). Opting into Turbo swaps to
    krea/Krea-2-Turbo + the Ostris adapter and CFG 1 / 8 steps. The two must carry
    DIFFERENT run-name tags so their incompatible weights never share a folder."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Kira', 'zchar_kira', train_type='krea')
        folder = tmp_path / 'ds'; folder.mkdir()

        # Default (no train_variant persisted) -> RAW, the recommended base.
        assert lt._krea_is_raw(ds) is True
        rp = lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]
        assert rp['model']['name_or_path'] == 'krea/Krea-2-Raw'
        assert 'assistant_lora_path' not in rp['model']
        assert rp['sample']['guidance_scale'] == 4 and rp['sample']['sample_steps'] == 25
        assert lt._run_name(ds).endswith('_Krea-2-Raw')

        # Opt into Turbo -> Turbo base + adapter + Turbo sampling + distinct tag.
        ds.train_variant = 'turbo'
        svc.db.session.commit()
        assert lt._krea_is_raw(ds) is False
        tp = lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]
        assert tp['model']['name_or_path'] == 'krea/Krea-2-Turbo'
        assert 'krea2_turbo_training_adapter' in tp['model']['assistant_lora_path']
        assert tp['sample']['guidance_scale'] == 1 and tp['sample']['sample_steps'] == 8
        assert lt._run_name(ds).endswith('_Krea-2-Turbo')


def test_open_training_folder_fixed_targets_and_creates(app, tmp_path, monkeypatch):
    """'loras' → dossier d'import ComfyUI de la famille ; 'run' → dossier du run.
    Chemins résolus serveur, créés au besoin, ouverts via l'explorateur (seam
    os.startfile patché) ; cible inconnue → ValueError."""
    import os as _os
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    opened = []
    monkeypatch.setattr(lt.os, 'startfile', lambda p: opened.append(p), raising=False)
    monkeypatch.setattr(lt.subprocess, 'Popen', lambda a, **k: opened.append(a[-1]))
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')},
                         'comfyui': {'base_dir': str(tmp_path / 'comfy')}})
        ds = svc.create_dataset(LOCAL_USER, 'K', 'foldertrig', train_type='krea')
        p1 = lt.open_training_folder(LOCAL_USER, ds.id, target='loras')
        assert p1.replace('/', _os.sep).endswith(_os.sep.join(('models', 'loras', 'krea')))
        assert _os.path.isdir(p1)                       # créé au besoin
        p2 = lt.open_training_folder(LOCAL_USER, ds.id, target='run')
        assert 'foldertrig' in p2 and _os.path.isdir(p2)
        assert opened == [p1, p2]
        with pytest.raises(ValueError, match='unknown folder target'):
            lt.open_training_folder(LOCAL_USER, ds.id, target='../evil')


def test_archive_previous_run_renames_never_deletes(app, tmp_path):
    """fresh=True écarte le run existant par RENAME `*_archived_<ts>` (checkpoints
    conservés sur disque, purgeables via le préfixe trigger) ; sans run → None."""
    import os
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'K', 'freshtrig', train_type='krea')
        assert lt.archive_previous_run(ds) is None          # aucun run → no-op
        run_dir = lt._output_dir() / lt._run_name(ds)
        run_dir.mkdir(parents=True)
        (run_dir / 'lora_freshtrig_000002000.safetensors').write_bytes(b'ck')
        dest = lt.archive_previous_run(ds)
        assert dest and not run_dir.exists()                # écarté, pas détruit
        assert os.path.isfile(os.path.join(dest, 'lora_freshtrig_000002000.safetensors'))
        # Le nom archivé reste sur la frontière du run (u<user>_<trigger>, le
        # préfixe que purge_training_artifacts balaie) → la suppression du
        # dataset emporte aussi les archives.
        assert lt._trigger_boundary(os.path.basename(dest), 'ulocal_freshtrig')
        # Un relancement voit maintenant un dossier vierge → plus d'auto-resume.
        assert lt.archive_previous_run(ds) is None


def test_train_settings_family_defaults(app):
    """No train_settings → researched family-aware defaults: Krea/SDXL rank 32,
    Z-Image rank 16; SDXL keeps alpha = rank/2; others alpha = rank."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        z = svc.create_dataset(LOCAL_USER, 'Z', 'zt', train_type='zimage')
        k = svc.create_dataset(LOCAL_USER, 'K', 'kt', train_type='krea')
        assert lt._lora_rank(z, 'zimage') == 16 and lt._lora_alpha(16, 'zimage') == 16
        assert lt._lora_rank(k, 'krea') == 32 and lt._lora_alpha(32, 'krea') == 32
        assert lt._lora_alpha(32, 'sdxl') == 16      # SDXL half-strength preserved
        eff = lt.effective_train_settings(k)
        assert eff['rank'] is None and eff['effective_rank'] == 32   # None = Auto
        assert eff['resolution'] == '768,1024' and eff['save_every'] == 250


def test_default_job_config_uses_researched_defaults(app, tmp_path):
    """Fresh Krea/Z-Image config: multi-scale [768,1024], save_every 250, and the
    family rank (Krea 32/32, Z-Image 16/16)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        for tt, rank in (('zimage', 16), ('krea', 32)):
            ds = svc.create_dataset(LOCAL_USER, tt, f't_{tt}', train_type=tt)
            p = lt.build_job_config(ds, str(folder), 1500)['config']['process'][0]
            assert p['network']['linear'] == rank and p['network']['linear_alpha'] == rank
            assert p['datasets'][0]['resolution'] == [768, 1024]
            assert p['save']['save_every'] == 250


def test_update_train_settings_persists_validates_and_applies(app, tmp_path):
    """update_train_settings persists a valid patch, rejects out-of-range values,
    feeds build_job_config, and 'auto' clears a knob back to the family default."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        ds = svc.create_dataset(LOCAL_USER, 'K', 'kt', train_type='krea')
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'rank': 64, 'resolution': '1024', 'save_every': 500})
        assert eff['rank'] == 64 and eff['resolution'] == '1024' and eff['save_every'] == 500
        # '768' seul = levier basse-VRAM (GPU < 24 GB) → accepté et appliqué au job.
        assert lt.update_train_settings(LOCAL_USER, ds.id, {'resolution': '768'})['resolution'] == '768'
        p768 = lt.build_job_config(ds, str(folder), 1500)['config']['process'][0]
        assert p768['datasets'][0]['resolution'] == [768]
        lt.update_train_settings(LOCAL_USER, ds.id, {'resolution': '1024'})   # restore pour la suite
        p = lt.build_job_config(ds, str(folder), 1500)['config']['process'][0]
        assert p['network']['linear'] == 64 and p['network']['linear_alpha'] == 64
        assert p['datasets'][0]['resolution'] == [1024] and p['save']['save_every'] == 500
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, ds.id, {'rank': 7})        # not in choices
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, ds.id, {'save_every': 123})
        eff2 = lt.update_train_settings(LOCAL_USER, ds.id, {'rank': 'auto'})  # clears to default
        assert eff2['rank'] is None and eff2['effective_rank'] == 32


def test_sample_prompts_defaults_are_kind_aware(app):
    """Preview defaults differ by kind: a character LoRA gets the portrait battery,
    a concept LoRA gets the concept alone (portrait wording would drag it off-topic).
    Both are surfaced trigger-resolved for the UI placeholder; nothing is stored yet."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        char = svc.create_dataset(LOCAL_USER, 'P', 'ptrig', train_type='zimage')
        con = svc.create_dataset(LOCAL_USER, 'C', 'ctrig', train_type='zimage',
                                 kind='concept', concept_desc='a mirror selfie')
        ce = lt.effective_train_settings(char)
        assert ce['sample_every'] == 250 and ce['sample_prompts'] == []
        # character default = portrait battery, trigger baked in for the placeholder
        assert any('portrait' in p for p in ce['sample_prompts_default'])
        assert all(p.startswith('ptrig') for p in ce['sample_prompts_default'])
        # concept default = the trigger alone, never portrait/headshot wording
        cd = lt.effective_train_settings(con)['sample_prompts_default']
        assert 'ctrig' in cd
        assert not any('portrait' in p or 'headshot' in p for p in cd)


def test_sample_prompts_custom_persist_inject_and_build(app, tmp_path):
    """Custom preview prompts persist raw, auto-prepend the trigger when missing (so
    the preview actually exercises the LoRA) but never double it, and flow into
    build_job_config. sample_every is validated against its choice set."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        folder = tmp_path / 'ds'; folder.mkdir()
        ds = svc.create_dataset(LOCAL_USER, 'Z', 'ztrig', train_type='zimage')
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {
            'sample_prompts': ['on a beach at sunset', 'ztrig in the snow'],
            'sample_every': 500,
        })
        assert eff['sample_every'] == 500
        assert eff['sample_prompts'] == ['on a beach at sunset', 'ztrig in the snow']  # stored raw
        p = lt.build_job_config(ds, str(folder), 1500)['config']['process'][0]['sample']
        assert p['sample_every'] == 500
        # line 1 had no trigger → prepended; line 2 already had it → untouched (not doubled)
        assert p['prompts'] == ['ztrig, on a beach at sunset', 'ztrig in the snow']
        with pytest.raises(ValueError):
            lt.update_train_settings(LOCAL_USER, ds.id, {'sample_every': 300})   # not in choices


def test_sample_prompts_string_cap_and_reset(app):
    """A newline string is accepted (UI convenience), blanks dropped, list capped at
    _MAX_SAMPLE_PROMPTS, and an empty value resets to the kind defaults."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Z', 'ztrig', train_type='zimage')
        many = '\n'.join(f'scene {i}' for i in range(20))
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'sample_prompts': many})
        assert len(eff['sample_prompts']) == lt._MAX_SAMPLE_PROMPTS      # capped
        eff2 = lt.update_train_settings(LOCAL_USER, ds.id, {'sample_prompts': 'a\n\n   \nb'})
        assert eff2['sample_prompts'] == ['a', 'b']                      # blanks dropped
        eff3 = lt.update_train_settings(LOCAL_USER, ds.id, {'sample_prompts': ''})
        assert eff3['sample_prompts'] == []                             # cleared → defaults
        assert any('portrait' in p for p in lt._sample_prompts(ds, 'ztrig'))
