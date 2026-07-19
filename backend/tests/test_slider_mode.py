"""Slider LoRA mode (Beta) — per-dataset MODE (not a dataset kind) backed by
ai-toolkit's modern `concept_slider` extension (extends DiffusionTrainer).

What matters here:
- the emitted job config swaps `type: sd_trainer` for `type: concept_slider`,
  drops the trigger word, carries the exact ConceptSliderTrainerConfig kwargs
  in `slider:` and strips masks (the guided slider loss never reads them);
- the dataset stays REQUIRED (denoising substrate) but the image floor drops
  and every caption guard goes silent (captions are encoded then ignored);
- slider runs live in their OWN run folder (`_slider` tag) so ai-toolkit's
  auto-resume can never mix a subject LoRA with a slider LoRA;
- Z-Image gets the community workaround for ai-toolkit issue #554 (batch 1 +
  text-embedding cache OFF) stamped into the config;
- the cloud lane refuses slider mode honestly (local-only V1);
- settings live in the dedicated `train_slider` column, so applying a training
  preset (which REPLACES train_settings) can never wipe a slider setup.
"""
import json

import pytest

from app.config import LOCAL_USER


def _mk(app, n_keep=0, caption='a nice varied caption with many words',
        train_type='zimage', trigger='sl_trig', name='Sl'):
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    ds = svc.create_dataset(LOCAL_USER, name, trigger, train_type=train_type)
    for i in range(n_keep):
        svc.db.session.add(FaceDatasetImage(
            dataset_id=ds.id, filename=f'k{i}.webp', status='keep', framing='face',
            caption=(f'{caption} #{i}' if caption is not None else None)))
    svc.db.session.commit()
    return ds


def _enable_slider(ds, positive='very muscular body', negative='skinny, frail body',
                   target_class='person', anchor='', **extra):
    from app.services import lora_training as lt
    patch = {'enabled': True, 'positive': positive, 'negative': negative,
             'target_class': target_class, 'anchor': anchor, **extra}
    return lt.update_slider_settings(LOCAL_USER, ds.id, patch)


# --- 1) settings: dedicated column, validation, preset isolation ----------------

def test_update_slider_settings_roundtrip_and_validation(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app)
        eff = _enable_slider(ds, anchor='a photo of a person', guidance=4,
                             anchor_strength=0.5)
        assert eff['enabled'] is True
        assert eff['positive'] == 'very muscular body'
        assert eff['negative'] == 'skinny, frail body'
        assert eff['anchor'] == 'a photo of a person'
        assert eff['guidance'] == 4.0 and eff['anchor_strength'] == 0.5
        # numeric knobs are range-validated, never silently clamped
        with pytest.raises(ValueError, match='guidance'):
            lt.update_slider_settings(LOCAL_USER, ds.id, {'guidance': 42})
        with pytest.raises(ValueError, match='anchor_strength'):
            lt.update_slider_settings(LOCAL_USER, ds.id, {'anchor_strength': -1})
        # over-long prompts are refused (not truncated behind the user's back)
        with pytest.raises(ValueError, match='too long'):
            lt.update_slider_settings(LOCAL_USER, ds.id, {'positive': 'x' * 501})
        # disabling drops the flag but keeps the typed prompts (state, not wipe)
        eff = lt.update_slider_settings(LOCAL_USER, ds.id, {'enabled': False})
        assert eff['enabled'] is False and eff['positive'] == 'very muscular body'


def test_slider_settings_survive_preset_apply(app):
    """A preset REPLACES train_settings; the slider column must be untouched —
    that's the whole reason it is a dedicated column."""
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app)
        _enable_slider(ds)
        lt.apply_train_settings_dict(LOCAL_USER, ds.id, {'rank': 16, 'save_every': 500})
        assert lt.slider_mode_enabled(ds) is True
        assert lt.effective_slider_settings(ds)['positive'] == 'very muscular body'


def test_slider_default_rank_is_low_but_explicit_rank_wins(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app)
        assert lt._lora_rank(ds, 'zimage') == 16          # normal default untouched
        _enable_slider(ds)
        assert lt._lora_rank(ds, 'zimage') == 8           # public sliders: rank 4-8
        assert lt.effective_train_settings(ds)['default_rank'] == 8
        lt.update_train_settings(LOCAL_USER, ds.id, {'rank': 32})
        assert lt._lora_rank(ds, 'zimage') == 32          # user choice always wins


def test_slider_default_alpha_is_four_but_explicit_alpha_wins(app, tmp_path):
    """Ostris slider notebook ships rank 8 / alpha 4 (scale 0.5, "bigger is not
    always better, especially for sliders"). The emitted network alpha defaults to
    4, the snapshot agrees, and the panel exposes 4 as the default — while the
    existing alpha knob still lets a user put 8 back for repro of an older run."""
    from app.services import lora_training as lt
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = _mk(app, train_type='zimage')
        _enable_slider(ds)
        folder = tmp_path / 'ds_sl'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]
        assert p['network']['linear'] == 8 and p['network']['linear_alpha'] == 4
        assert lt.launch_settings_snapshot(ds)['alpha'] == 4
        assert lt.effective_train_settings(ds)['default_alpha'] == 4
        assert lt.effective_slider_settings(ds)['default_alpha'] == 4
        # explicit alpha override wins (a user reproducing a pre-change slider run)
        lt.update_train_settings(LOCAL_USER, ds.id, {'alpha': 8})
        p8 = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]
        assert p8['network']['linear_alpha'] == 8


# --- 2) job config emission (the ConceptSliderTrainerConfig contract) -----------

def _slider_process(app, tmp_path, train_type, variant=None, anchor='',
                    base_model=None, **slider_extra):
    from app.services import lora_training as lt
    from app import config as cfg
    cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    ds = _mk(app, train_type=train_type, trigger=f'sl_{train_type}',
             name=f'Sl{train_type}')
    if variant:
        ds.train_variant = variant
    if base_model is not None:
        ds.train_base_model = base_model
    from app.services import face_dataset_service as svc
    svc.db.session.commit()
    _enable_slider(ds, anchor=anchor, **slider_extra)
    folder = tmp_path / f'ds_{train_type}'
    folder.mkdir(exist_ok=True)
    return ds, lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]


def test_build_job_config_slider_common_contract_all_families(app, tmp_path, monkeypatch):
    """Every family flips to the concept_slider process with the exact slider
    block, no trigger_word, stripped masks and bipolar preview samples — while
    keeping its own model block (base/adapter/quantize) unchanged."""
    from app.services import lora_training as lt
    cases = [('zimage', None, None), ('krea', 'turbo', None), ('flux', None, None),
             ('flux2klein', None, None), ('sdxl', None, 'base.safetensors')]
    # SDXL resolves its base under ComfyUI models — bypass the path lookup.
    monkeypatch.setattr(lt, '_sdxl_base_path', lambda b: f'C:/fake/{b}')
    with app.app_context():
        for fam, variant, base in cases:
            ds, p = _slider_process(app, tmp_path, fam, variant=variant,
                                    base_model=base)
            assert p['type'] == 'concept_slider', fam
            assert 'trigger_word' not in p, fam
            assert p['slider'] == {
                'guidance_strength': 3.0,
                'anchor_strength': 1.0,
                'positive_prompt': 'very muscular body',
                'negative_prompt': 'skinny, frail body',
                'target_class': 'person',
            }, fam
            d = p['datasets'][0]
            assert 'mask_path' not in d and 'mask_min_value' not in d, fam
            # bipolar preview sheet: same prompt at ±multipliers
            assert 'prompts' not in p['sample'], fam
            assert [s['network_multiplier'] for s in p['sample']['samples']] \
                == [-2, -1, 1, 2], fam
            assert all(s['prompt'] == 'a photo of a person'
                       for s in p['sample']['samples']), fam
            # slider default rank rides the existing network block
            assert p['network']['linear'] == 8, fam


def test_build_job_config_slider_zimage_issue_554_workaround(app, tmp_path):
    """Z-Image slider: batch_size 1 (already the family default) AND the text
    embedding cache explicitly OFF — the community workaround for ai-toolkit
    issue #554 (broken embedding cache on the zimage slider path)."""
    with app.app_context():
        ds, p = _slider_process(app, tmp_path, 'zimage')
        assert p['train']['batch_size'] == 1
        assert p['datasets'][0]['cache_text_embeddings'] is False
        # the family model block is untouched (arch + quantize recipe)
        assert p['model']['arch'] == 'zimage'


def test_build_job_config_slider_krea_keeps_adapter_and_te_cache(app, tmp_path):
    """Krea Turbo slider keeps the de-distillation training adapter and its
    text-embedding cache: ConceptSliderTrainer explicitly supports cached TE
    (it encodes the slider prompts BEFORE the parent unloads the encoder)."""
    with app.app_context():
        ds, p = _slider_process(app, tmp_path, 'krea', variant='turbo')
        assert p['model']['assistant_lora_path'] == (
            'ostris/krea2_turbo_training_adapter/'
            'krea2_turbo_training_adapter_v1.safetensors')
        assert p['datasets'][0]['cache_text_embeddings'] is True
        assert p['train']['unload_text_encoder'] is True


def test_build_job_config_slider_anchor_emitted_only_when_set(app, tmp_path):
    """ConceptSliderTrainerConfig defaults anchor_class to None (anchors OFF);
    an empty string would ENABLE an anchor on the unconditional prompt — so the
    key is emitted only when the user typed one."""
    with app.app_context():
        ds, p = _slider_process(app, tmp_path, 'zimage')
        assert 'anchor_class' not in p['slider']
        ds2, p2 = _slider_process(app, tmp_path, 'krea', variant='turbo',
                                  anchor='a photo of a person')
        assert p2['slider']['anchor_class'] == 'a photo of a person'


def test_build_job_config_normal_mode_regression_guard(app, tmp_path):
    """Slider OFF -> byte-for-byte the historical sd_trainer process (no slider
    block, trigger word present)."""
    from app.services import lora_training as lt
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = _mk(app, train_type='zimage', trigger='sl_norm', name='SlNorm')
        folder = tmp_path / 'ds_norm'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]
        assert p['type'] == 'sd_trainer'
        assert 'slider' not in p
        assert p['trigger_word'] == 'sl_norm'


# --- 3) run identity: the slider tag isolates the run folder --------------------

def test_run_name_slider_tag_isolates_runs(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app)
        normal = lt._run_name(ds)
        _enable_slider(ds)
        slider = lt._run_name(ds)
        assert slider == normal + '_slider'
        assert slider != normal


# --- 4) launch guards: substrate floor, no caption walls, prompts required ------

def test_assert_trainable_slider_branch(app):
    from app.services import lora_training as lt
    with app.app_context():
        # 6 kept images, NO captions at all: a normal run would wall on
        # UNCAPTIONED (and on the 10-image floor); a slider run passes.
        ds = _mk(app, n_keep=6, caption=None)
        _enable_slider(ds)
        lt.assert_trainable(ds.id)                        # no raise
        # below the substrate floor -> actionable refusal
        ds2 = _mk(app, n_keep=3, caption=None, trigger='sl_t2', name='Sl2')
        _enable_slider(ds2)
        with pytest.raises(ValueError, match='denoising substrate'):
            lt.assert_trainable(ds2.id)
        # missing prompt pair -> actionable refusal
        ds3 = _mk(app, n_keep=6, trigger='sl_t3', name='Sl3')
        _enable_slider(ds3, negative='')
        with pytest.raises(ValueError, match='positive and a negative prompt'):
            lt.assert_trainable(ds3.id)


def test_assert_trainable_slider_skips_caption_style_mismatch(app):
    """Booru captions on a zimage dataset trip MISMATCH_CAPTION normally; in
    slider mode captions are ignored by the loss, so no mismatch wall."""
    from app.services import lora_training as lt
    with app.app_context():
        booru = '1girl, solo, cafe, sitting, window, jeans, smile, looking_at_viewer'
        ds = _mk(app, n_keep=12, caption=booru)
        with pytest.raises(ValueError, match='MISMATCH_CAPTION'):
            lt.assert_trainable(ds.id, train_type='zimage')
        _enable_slider(ds)
        lt.assert_trainable(ds.id, train_type='zimage')   # no raise


def test_preflight_slider_branch(app):
    from app.services import lora_training as lt
    with app.app_context():
        # missing prompts -> blocker + fail check targeting the training panel
        ds = _mk(app, n_keep=6, caption=None)
        lt.update_slider_settings(LOCAL_USER, ds.id, {'enabled': True})
        r = lt.training_preflight(LOCAL_USER, ds.id)
        assert r['verdict'] == 'blocked'
        assert any(c['id'] == 'slider_prompts' and c['status'] == 'fail'
                   for c in r['checks'])
        # prompts set, 6 substrate images, zero captions -> ready-ish (no caption
        # walls, no composition/leak/duplicate noise), floor is the slider floor
        _enable_slider(ds)
        r = lt.training_preflight(LOCAL_USER, ds.id)
        assert r['floor'] == 4
        assert not r['blockers']
        assert not any(c['id'] in ('caption_quality', 'composition', 'leaks',
                                   'duplicates') for c in r['checks'])
        cap = next(c for c in r['checks'] if c['id'] == 'captioned')
        assert cap['status'] == 'ok' and 'ignored' in cap['detail']
        # below substrate floor -> blocked
        ds2 = _mk(app, n_keep=3, trigger='sl_p2', name='SlP2')
        _enable_slider(ds2)
        r2 = lt.training_preflight(LOCAL_USER, ds2.id)
        assert r2['verdict'] == 'blocked' and r2['floor'] == 4


def test_recommended_steps_slider_policy_fixed(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app, n_keep=6)
        _enable_slider(ds)
        assert lt.recommended_steps(ds.id) == lt.SLIDER_DEFAULT_STEPS == 1000
        info = lt.recommended_steps_info(ds.id)
        assert info['slider'] is True and 'substrate' in info['rationale']
        # dataset size does NOT drive the target
        ds2 = _mk(app, n_keep=60, trigger='sl_s2', name='SlS2')
        _enable_slider(ds2)
        assert lt.recommended_steps(ds2.id) == 1000


# --- 5) provenance / export / support guard -------------------------------------

def test_launch_settings_snapshot_carries_prompt_pair_not_trigger(app):
    from app.services import lora_training as lt
    with app.app_context():
        ds = _mk(app)
        _enable_slider(ds, anchor='a photo of a person')
        snap = lt.launch_settings_snapshot(ds)
        assert snap['slider_mode'] is True
        assert snap['slider']['positive_prompt'] == 'very muscular body'
        assert snap['slider']['negative_prompt'] == 'skinny, frail body'
        assert snap['slider']['anchor_class'] == 'a photo of a person'
        assert 'trigger' not in snap
        assert snap['rank'] == 8


def test_export_forces_masks_off_in_slider_mode(app, tmp_path, monkeypatch):
    """masked=True on a character dataset in slider mode must NOT generate person
    masks (the slider loss never reads them) — server guard, like concept/style."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from PIL import Image
    with app.app_context():
        ds = _mk(app, n_keep=0)
        _enable_slider(ds)
        img_dir = tmp_path / 'imgs'; img_dir.mkdir()
        from app.models import FaceDatasetImage
        for i in range(4):
            Image.new('RGB', (64, 64), 'white').save(img_dir / f'k{i}.png')
            svc.db.session.add(FaceDatasetImage(
                dataset_id=ds.id, filename=f'k{i}.png', status='keep',
                caption='substrate'))
        svc.db.session.commit()
        monkeypatch.setattr(svc, '_dataset_dir', lambda did: str(img_dir))
        called = {}
        monkeypatch.setattr(lt, 'generate_person_masks',
                            lambda *a, **kw: called.setdefault('masks', True) or {})
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=True,
                                             dest_dir=str(tmp_path / 'out'))
        assert 'masks' not in called          # mask generation never invoked
        assert (tmp_path / 'out' / f'sl_trig_000.png').exists()


def test_launch_refuses_when_concept_slider_extension_missing(app, tmp_path, monkeypatch):
    """An older ai-toolkit without the concept_slider extension would crash at
    job boot on the unknown process type — refuse early, actionably."""
    from app.services import lora_training as lt
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    (root / 'extensions_built_in' / 'sd_trainer').mkdir(parents=True)
    (root / 'extensions_built_in' / 'sd_trainer' / '__init__.py').write_text(
        'uid = "sd_trainer"\n', encoding='utf-8')
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
        assert lt._aitoolkit_supports_concept_slider() is False
        ds = _mk(app, n_keep=6)
        _enable_slider(ds)
        with pytest.raises(ValueError, match=r'update it \(git pull\)'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False)
        # with the extension present, the guard opens
        ext = root / 'extensions_built_in' / 'concept_slider'
        ext.mkdir(parents=True)
        (ext / '__init__.py').write_text('class X:\n    uid = "concept_slider"\n',
                                         encoding='utf-8')
        assert lt._aitoolkit_supports_concept_slider() is True


# --- 5b) VRAM default: slider trains at 768 only unless overridden --------------

def test_slider_defaults_to_768_only_resolution(app, tmp_path):
    """The concept_slider loss makes several prediction passes per step, so its
    VRAM peak sits far above a normal run — multi-scale 768+1024 OOMs on 24 GB.
    A slider run with no explicit resolution therefore emits 768 only, and the
    stamped snapshot agrees (provenance can never disagree with the job)."""
    from app.services import lora_training as lt
    with app.app_context():
        ds, p = _slider_process(app, tmp_path, 'krea', variant='turbo')
        assert p['datasets'][0]['resolution'] == [768]
        assert lt.launch_settings_snapshot(ds)['resolution'] == [768]


def test_slider_respects_explicit_resolution(app, tmp_path):
    """A DEFAULT, not a clamp: an explicit user resolution is obeyed in slider
    mode — both 768+1024 and 1024 ride straight through to the job + snapshot."""
    from app.services import lora_training as lt
    with app.app_context():
        ds, _ = _slider_process(app, tmp_path, 'krea', variant='turbo')
        folder = str(tmp_path / 'ds_krea')
        lt.update_train_settings(LOCAL_USER, ds.id, {'resolution': '768,1024'})
        p = lt.build_job_config(ds, folder, steps=1000)['config']['process'][0]
        assert p['datasets'][0]['resolution'] == [768, 1024]
        assert lt.launch_settings_snapshot(ds)['resolution'] == [768, 1024]
        lt.update_train_settings(LOCAL_USER, ds.id, {'resolution': '1024'})
        p2 = lt.build_job_config(ds, folder, steps=1000)['config']['process'][0]
        assert p2['datasets'][0]['resolution'] == [1024]
        assert lt.launch_settings_snapshot(ds)['resolution'] == [1024]


def test_non_slider_resolution_default_unchanged(app, tmp_path):
    """Regression guard: a NORMAL run still defaults to the 768+1024 multi-scale
    family default — the 768-only default is slider-specific."""
    from app.services import lora_training as lt
    from app import config as cfg
    from app.services import face_dataset_service as svc
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = _mk(app, train_type='krea', trigger='res_norm', name='ResNorm')
        ds.train_variant = 'turbo'
        svc.db.session.commit()
        folder = tmp_path / 'ds_norm'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=1000)['config']['process'][0]
        assert p['datasets'][0]['resolution'] == [768, 1024]
        assert lt.launch_settings_snapshot(ds)['resolution'] == [768, 1024]


def test_effective_train_settings_reports_slider_768_default(app, tmp_path):
    """The training panel reads effective_train_settings: it must report the
    768-only slider default (and flip once the user picks a resolution) so the
    control + summary never claim 768+1024 for a run that emits 768."""
    from app.services import lora_training as lt
    with app.app_context():
        ds, _ = _slider_process(app, tmp_path, 'zimage')
        eff = lt.effective_train_settings(ds)
        assert eff['effective_resolution'] == [768]
        assert eff['resolution_explicit'] is False
        lt.update_train_settings(LOCAL_USER, ds.id, {'resolution': '1024'})
        eff2 = lt.effective_train_settings(ds)
        assert eff2['effective_resolution'] == [1024]
        assert eff2['resolution_explicit'] is True


# --- 6) cloud lane: slider rides the same pod, frozen per run -------------------

def test_cloud_launch_accepts_slider_and_snapshots_settings(app, monkeypatch):
    """Slider mode no longer refuses the cloud lane (the pod's ai-toolkit runs
    the concept_slider trainer). The launch proceeds and FREEZES the slider blob
    into the run params (dedicated snapshot key), so a later toggle-off can't
    retarget the in-flight run — same immutability contract as train_settings."""
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    from app.services import cloud_training as ct
    monkeypatch.setattr(ct, '_start_monitor', lambda *a, **k: None)
    monkeypatch.setattr(ct, '_reconcile_before_launch', lambda a: None)
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: [])
    with app.app_context():
        ds = _mk(app, n_keep=6)
        _enable_slider(ds)
        res = ct.launch_cloud_training(LOCAL_USER, ds.id)
        run = ct.CloudTrainingRun.query.get(res['run_id'])
        params = json.loads(run.train_params)
        snap = json.loads(params['train_slider_snapshot'])
        assert snap['enabled'] is True
        assert snap['positive'] == 'very muscular body'
        # the pre-launch offers view no longer refuses either (returns a dict)
        assert isinstance(ct.gpu_tiers(LOCAL_USER, ds.id), dict)


def test_cloudify_preserves_concept_slider_type(app):
    """_cloudify_job_config retypes the standard local 'sd_trainer' to the pod's
    'diffusion_trainer', but a 'concept_slider' job keeps its uid: the pod runs
    that built-in extension as-is, and flattening it would silently drop the
    slider loss and train an ordinary LoRA."""
    from app.services import cloud_training as ct
    pod = {'DATASETS_FOLDER': '/pod/ds', 'TRAINING_FOLDER': '/pod/out'}

    def _cfg(ptype):
        return {'config': {'name': 'x', 'process': [{
            'type': ptype, 'training_folder': '__POD__', 'device': 'cpu',
            'datasets': [{'folder_path': 'C:\\staging\\dataset'}]}]}}

    out = ct._cloudify_job_config(_cfg('concept_slider'), 'job1',
                                  'C:\\staging\\dataset', pod)
    assert out['config']['process'][0]['type'] == 'concept_slider'
    out2 = ct._cloudify_job_config(_cfg('sd_trainer'), 'job1',
                                   'C:\\staging\\dataset', pod)
    assert out2['config']['process'][0]['type'] == 'diffusion_trainer'


def test_slider_snapshot_freezes_pod_job_against_later_edits(app, tmp_path):
    """The pod job is built minutes after launch through _run_config_dataset,
    which must read the LAUNCH-TIME slider snapshot: disabling slider mode on the
    dataset in between cannot turn an in-flight slider run into a plain LoRA."""
    from app.services import cloud_training as ct
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = _mk(app, train_type='krea', trigger='sl_snap', name='SlSnap')
        ds.train_variant = 'turbo'
        svc.db.session.commit()
        _enable_slider(ds)
        # The frozen blob, exactly as launch_cloud_training stamps it.
        params = {'train_type': 'krea', 'variant': 'turbo', 'base_model': '',
                  'train_slider_snapshot': ds.train_slider}
        # User disables slider on the dataset AFTER the launch was stamped.
        lt.update_slider_settings(LOCAL_USER, ds.id, {'enabled': False})
        assert lt.slider_mode_enabled(ds) is False
        folder = tmp_path / 'ds_krea'; folder.mkdir()
        view = ct._run_config_dataset(ds, params)
        p = lt.build_job_config(view, str(folder), steps=1000)['config']['process'][0]
        assert p['type'] == 'concept_slider'
        assert p['slider']['positive_prompt'] == 'very muscular body'


def test_legacy_run_without_slider_snapshot_reads_live_column(app, tmp_path):
    """A pre-feature run row carries no train_slider snapshot: _run_config_dataset
    then falls back to the live dataset column (never crashes on the missing key,
    mirroring the train_settings legacy fallback)."""
    from app.services import cloud_training as ct
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = _mk(app, train_type='krea', trigger='sl_leg', name='SlLeg')
        ds.train_variant = 'turbo'
        svc.db.session.commit()
        _enable_slider(ds)
        params = {'train_type': 'krea', 'variant': 'turbo', 'base_model': ''}
        folder = tmp_path / 'ds_leg'; folder.mkdir()
        view = ct._run_config_dataset(ds, params)
        p = lt.build_job_config(view, str(folder), steps=1000)['config']['process'][0]
        assert p['type'] == 'concept_slider'


# --- 7) API surface --------------------------------------------------------------

def test_slider_route_and_base_info_payload(app, client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe', lambda: {
        'aitoolkit': {'valid': True}, 'cloud_training': False})
    ds_id = client.post('/api/dataset/create',
                        json={'name': 'SlApi', 'trigger_word': 'sl_api'}).get_json()['id']
    r = client.post(f'/api/dataset/{ds_id}/train/slider',
                    json={'enabled': True, 'positive': 'p', 'negative': 'n'})
    assert r.status_code == 200
    d = r.get_json()
    assert d['ok'] is True and d['slider']['enabled'] is True
    # invalid knob -> 400, never a silent clamp
    r2 = client.post(f'/api/dataset/{ds_id}/train/slider', json={'guidance': 99})
    assert r2.status_code == 400
