"""Anima ('anima') — a first-class training family (circlestone-labs Anima, a
Cosmos Predict2 DiT 2B + Qwen3 text encoder + T5 conditioner + Qwen-Image VAE).

What matters here: the family is accepted end-to-end, the extension-arch guard
refuses a launch on an ai-toolkit that lacks the 'anima' arch (silent SD-loader
fallback otherwise, PR ostris/ai-toolkit #860), the job-config mirrors ai-toolkit
options.ts (arch 'anima', PUBLIC base, quantize off, weighted timesteps, the anime
default negative, rank 32), the official base is PUBLIC (no gate), deploy routing
lands in loras/anima, and — crucially for THIS wave — the cloud path is REFUSED
(local-first until the pod image ships a recent ai-toolkit + diffusers).
"""
import pytest


def _configure_aitoolkit(tmp_path, app, supports_anima=True):
    """Fake ai-toolkit install (venv python + run.py), with or without the 'anima'
    extension arch under extensions_built_in — what the support guard scans."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    ext = root / 'extensions_built_in' / 'diffusion_models' / 'anima'
    ext.mkdir(parents=True)
    if supports_anima:
        (ext / 'anima.py').write_text(
            'class AnimaModel:\n    arch = "anima"\n', encoding='utf-8')
    else:
        # An incidental 'anima' mention in a comment must NOT count (exact arch).
        (ext / 'placeholder.py').write_text(
            '# anima support not merged yet\n'
            'class Placeholder:\n    arch = "flux"\n', encoding='utf-8')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


# --- 1) train_type accepted / normalized ---------------------------------------

def test_normalize_train_type_accepts_anima():
    from app.services import face_dataset_service as svc
    assert svc.normalize_train_type('anima') == 'anima'
    assert svc.normalize_train_type('ANIMA') == 'anima'      # case-fold
    assert svc.normalize_train_type('bogus') == 'zimage'     # unknown -> default
    assert svc.normalize_train_type(None) == 'zimage'


# --- 2) extension-arch guard + actionable launch refusal ------------------------

def test_aitoolkit_supports_anima_scans_extension_arch(app, tmp_path):
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, app, supports_anima=True)
    with app.app_context():
        assert lt._aitoolkit_supports_anima() is True


def test_aitoolkit_supports_anima_false_without_arch(app, tmp_path):
    """No 'anima' arch on disk -> False, even with 'anima' mentioned in a comment
    (exact-arch match, no substring false positive). Unconfigured -> False too."""
    from app.services import lora_training as lt
    with app.app_context():
        assert lt._aitoolkit_supports_anima() is False       # not configured
    _configure_aitoolkit(tmp_path, app, supports_anima=False)
    with app.app_context():
        assert lt._aitoolkit_supports_anima() is False


def test_launch_and_enqueue_refuse_anima_when_arch_missing(app, tmp_path, monkeypatch):
    """Without the guard, get_model_class would silently fall back to the SD legacy
    loader -> corrupted LoRA. The refusal must be actionable (git pull), on both the
    direct launch and the queue path."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app, supports_anima=False)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'AN', 'zchar_an', train_type='anima')
        with pytest.raises(ValueError, match=r'update it \(git pull\)'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False)
        monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
        with pytest.raises(ValueError, match=r'update it \(git pull\)'):
            lt.enqueue_training(LOCAL_USER, ds.id, extra_steps=100)


# --- 3) job config: mirrors ai-toolkit options.ts (PR #860) --------------------

def test_build_job_config_anima(app, tmp_path):
    """arch 'anima', PUBLIC base, quantize OFF (2B), weighted timesteps + flowmatch,
    the anime default negative, rank 32/32, and the VRAM-caching knobs."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Ani', 'zchar_ani', train_type='anima')
        folder = tmp_path / 'ds'; folder.mkdir()
        p = lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]
        m = p['model']
        assert m['arch'] == 'anima'
        assert m['name_or_path'] == 'circlestone-labs/Anima-Base-v1.0-Diffusers'
        assert m['quantize'] is False and m['quantize_te'] is False
        assert 'low_vram' not in m and 'qtype' not in m
        assert p['train']['timestep_type'] == 'weighted'
        assert p['train']['noise_scheduler'] == 'flowmatch'
        assert p['train']['unload_text_encoder'] is True
        assert p['datasets'][0]['cache_text_embeddings'] is True
        assert p['datasets'][0]['caption_ext'] == 'txt'
        assert p['sample']['sampler'] == 'flowmatch'
        assert p['sample']['neg'] == lt.ANIMA_SAMPLE_NEG
        assert 'score_1' in p['sample']['neg']              # the anime default neg
        assert p['sample']['guidance_scale'] == 4 and p['sample']['sample_steps'] == 25
        assert p['network'] == {'type': 'lora', 'linear': 32, 'linear_alpha': 32}


def test_official_base_repo_anima_is_public(app, tmp_path):
    """Anima's official base is PUBLIC (non-gated) -> resolvable for the pre-rent
    HEAD; a custom-weights file resolves to None (nothing for a pod to fetch)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'AB', 'zchar_ab', train_type='anima')
        assert lt.official_base_repo(ds, 'anima') == 'circlestone-labs/Anima-Base-v1.0-Diffusers'
        ds.train_base_model = r'C:\weights\my-anima.safetensors'
        svc.db.session.commit()
        assert lt.official_base_repo(ds, 'anima') is None


# --- 4) cloud path REFUSED (local-first this wave) -----------------------------

def test_cloud_training_refuses_anima(app, tmp_path, monkeypatch):
    """Local-first: cloud must refuse Anima BEFORE reserving anything, with a
    readable reason (the pod image predates the 'anima' arch). Both cloud entry
    points (tiers estimate + launch) enforce it."""
    from app.services import cloud_training as ct
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    # Fake VAST key so the anima refusal (which sits AFTER the key check) is reached.
    monkeypatch.setattr(ct.cfg, 'secret',
                        lambda k: 'fake-key' if k == 'VAST_API_KEY' else None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'AC', 'zchar_ac', train_type='anima')
        with pytest.raises(ValueError, match='Anima cloud training is coming'):
            ct.gpu_tiers(LOCAL_USER, ds.id)
        with pytest.raises(ValueError, match='Anima cloud training is coming'):
            ct.launch_cloud_training(LOCAL_USER, ds.id, train_type='anima')


# --- 5) run tag isolation + deploy routing -------------------------------------

def test_dest_base_tag_anima_isolated_from_zimage(app):
    """Anima's single official base yields an empty raw tag that would telescope a
    zimage official run (same trigger) — the '_Anima-Base' tag isolates it."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'AT', 'zchar_at', train_type='anima')
        assert lt._dest_base_tag(ds) == '_Anima-Base'
        assert lt._run_name(ds).endswith('_Anima-Base')
        ds.train_type = 'zimage'
        svc.db.session.commit()
        # A zimage official run carries its own recipe tag (default Turbo) — the
        # point is that it never collides with Anima's '_Anima-Base'.
        assert lt._dest_base_tag(ds) == '_Z-Image-Turbo'
        assert lt._dest_base_tag(ds) != '_Anima-Base'


def test_lora_dest_dir_routes_anima(app, tmp_path):
    import os
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(tmp_path / 'comfy')}})
        ds = svc.create_dataset(LOCAL_USER, 'AD', 'zchar_ad', train_type='anima')
        dest = lt._lora_dest_dir(ds)
        assert dest.replace('/', os.sep).endswith(
            os.sep.join(('models', 'loras', 'anima')))


def test_family_of_lora_classifies_anima_folder():
    from app.utils.comfyui import family_of_lora, FAMILY_LABELS
    assert family_of_lora(r'anima\x.safetensors') == 'anima'
    assert family_of_lora('anima/x.safetensors') == 'anima'
    assert FAMILY_LABELS['anima'] == 'Anima'


# --- 6) prose captions (not booru) ---------------------------------------------

def test_anima_expects_prose_captions(app):
    """Everything != sdxl expects prose: booru-tag captions on an anima dataset
    trip the MISMATCH_CAPTION guard (forceable, like the others)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'AP', 'zchar_ap', train_type='anima')
        booru = '1girl, solo, cafe, sitting, window, jeans, smile, looking_at_viewer'
        for _ in range(15):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                filename='x.webp', caption=booru))
        svc.db.session.commit()
        with pytest.raises(ValueError, match='MISMATCH_CAPTION'):
            lt.assert_trainable(ds.id, train_type='anima')
        lt.assert_trainable(ds.id, train_type='anima', allow_caption_mismatch=True)


# --- 7) built-in presets --------------------------------------------------------

def test_builtin_presets_include_anima():
    """One Character + one Concept preset for Anima, both extrapolated from
    ai-toolkit defaults (rank 32, weighted) and honestly flagged as such."""
    from app.services.lora_training import BUILTIN_TRAIN_PRESETS
    anima = [pr for pr in BUILTIN_TRAIN_PRESETS if pr['train_type'] == 'anima']
    kinds = {pr['dataset_kind'] for pr in anima}
    assert kinds == {'character', 'concept'}
    for pr in anima:
        assert pr['settings']['timestep_type'] == 'weighted'
        assert 'no anima-specific' in pr['description'].lower()
    char = next(pr for pr in anima if pr['dataset_kind'] == 'character')
    assert char['settings']['rank'] == 32 and char['settings']['alpha'] == 32
