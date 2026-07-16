"""FLUX.2 Klein ('flux2klein') — the 5th training family.

What matters here: the family is accepted end-to-end ('klein' alone stays the
GENERATION engine's namespace and must NOT become a train_type), the two model
sizes (4B default / 9B) drive arch + name_or_path + distinct run tags (their
weights are incompatible — shared folders would corrupt a resume), the
extension-arch guard refuses a launch on an ai-toolkit that lacks flux2_klein_*
(silent SD-loader fallback otherwise), and the cloud path is OPEN for this
family (unlike flux, which stays local-only)."""
import pytest


def _configure_aitoolkit(tmp_path, app, supports_klein=True):
    """Fake ai-toolkit install (venv python + run.py), with or without the
    flux2_klein extension archs under extensions_built_in — what the support
    guard actually scans."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    ext = root / 'extensions_built_in' / 'diffusion_models' / 'flux2'
    ext.mkdir(parents=True)
    if supports_klein:
        (ext / 'flux2_klein_model.py').write_text(
            'class Flux2Klein4BModel:\n    arch = "flux2_klein_4b"\n\n'
            'class Flux2Klein9BModel:\n    arch = "flux2_klein_9b"\n',
            encoding='utf-8')
    else:
        # flux2 base only — an incidental 'klein' mention must NOT count.
        (ext / 'flux2_model.py').write_text(
            '# klein variants not merged yet\n'
            'class Flux2Model:\n    arch = "flux2"\n', encoding='utf-8')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


# --- 1) train_type accepted / normalized ---------------------------------------

def test_normalize_train_type_accepts_flux2klein_unknown_stays_zimage():
    from app.services import face_dataset_service as svc
    assert svc.normalize_train_type('flux2klein') == 'flux2klein'
    assert svc.normalize_train_type('FLUX2KLEIN') == 'flux2klein'   # case-fold
    assert svc.normalize_train_type('bogus') == 'zimage'            # unknown -> default
    assert svc.normalize_train_type(None) == 'zimage'
    # 'klein' alone is the GENERATION engine's namespace, never a train family.
    assert svc.normalize_train_type('klein') == 'zimage'


# --- 2) extension-arch guard + actionable launch refusal ------------------------

def test_aitoolkit_supports_flux2klein_scans_extension_archs(app, tmp_path):
    from app.services import lora_training as lt
    _configure_aitoolkit(tmp_path, app, supports_klein=True)
    with app.app_context():
        assert lt._aitoolkit_supports_flux2klein() is True


def test_aitoolkit_supports_flux2klein_false_without_arch(app, tmp_path):
    """No flux2_klein_* arch on disk -> False, even with 'klein' mentioned in a
    comment (exact-arch match, no substring false positive). Unconfigured
    ai-toolkit -> False too."""
    from app.services import lora_training as lt
    with app.app_context():
        assert lt._aitoolkit_supports_flux2klein() is False   # not configured
    _configure_aitoolkit(tmp_path, app, supports_klein=False)
    with app.app_context():
        assert lt._aitoolkit_supports_flux2klein() is False


def test_launch_refuses_flux2klein_when_arch_missing(app, tmp_path, monkeypatch):
    """Without the guard, get_model_class would silently fall back to the SD
    legacy loader -> corrupted LoRA. The refusal must be actionable (git pull)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app, supports_klein=False)
    monkeypatch.setattr(lt.shutil, 'disk_usage',
                        lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FK', 'zchar_fk', train_type='flux2klein')
        with pytest.raises(ValueError, match=r'update it \(git pull\)'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False)
        # Same guard on the queue path — no deferred job doomed to the fallback.
        monkeypatch.setattr(lt, 'assert_trainable', lambda *_a, **_kw: None)
        with pytest.raises(ValueError, match=r'update it \(git pull\)'):
            lt.enqueue_training(LOCAL_USER, ds.id, extra_steps=100)


# --- 3) job config: 4B default / 9B opt-in + the family's divergences ----------

def test_build_job_config_flux2klein_4b_default_and_9b_optin(app, tmp_path):
    """4B is the default (local 16-24 GB lane); 9B swaps arch AND name_or_path.
    Family divergences vs flux: timestep 'weighted' (not sigmoid),
    model_kwargs {'match_target_res': False}, previews with real CFG (the base
    is non-distilled)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Klee', 'zchar_klee', train_type='flux2klein')
        folder = tmp_path / 'ds'; folder.mkdir()

        assert lt._flux2klein_is_9b(ds) is False          # no variant -> 4B
        p = lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]
        m = p['model']
        assert m['arch'] == 'flux2_klein_4b'
        assert m['name_or_path'] == 'black-forest-labs/FLUX.2-klein-base-4B'
        assert m['quantize'] is True and m['quantize_te'] is True
        assert m['low_vram'] is True and m['qtype'] == 'qfloat8'
        assert m['model_kwargs'] == {'match_target_res': False}
        assert p['train']['timestep_type'] == 'weighted'
        assert p['train']['noise_scheduler'] == 'flowmatch'
        assert p['sample']['sampler'] == 'flowmatch'
        # non-distilled base -> real CFG previews (like Krea Raw, unlike FLUX.1)
        assert p['sample']['guidance_scale'] == 4 and p['sample']['sample_steps'] == 25
        assert p['datasets'][0]['caption_ext'] == 'txt'
        assert p['network'] == {'type': 'lora', 'linear': 16, 'linear_alpha': 16}

        ds.train_variant = '9b'
        svc.db.session.commit()
        assert lt._flux2klein_is_9b(ds) is True
        p9 = lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]
        assert p9['model']['arch'] == 'flux2_klein_9b'
        assert p9['model']['name_or_path'] == 'black-forest-labs/FLUX.2-klein-base-9B'
        assert p9['model']['model_kwargs'] == {'match_target_res': False}


def test_flux2klein_expects_prose_captions(app):
    """Everything != sdxl expects prose: booru-tag captions on a flux2klein
    dataset trip the MISMATCH_CAPTION guard (forceable, like the others)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.models import FaceDatasetImage
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FB', 'zchar_fb', train_type='flux2klein')
        booru = '1girl, solo, cafe, sitting, window, jeans, smile, looking_at_viewer'
        for _ in range(12):
            svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                                filename='x.webp', caption=booru))
        svc.db.session.commit()
        with pytest.raises(ValueError, match='MISMATCH_CAPTION'):
            lt.assert_trainable(ds.id, train_type='flux2klein')
        lt.assert_trainable(ds.id, train_type='flux2klein', allow_caption_mismatch=True)


# --- 4) distinct run tags per size (no 4B/9B telescoping) ----------------------

def test_dest_base_tag_distinct_for_4b_and_9b(app):
    """4B and 9B are incompatible checkpoints: same trigger, two sizes -> two
    run folders / deployed names. A shared tag would make ai-toolkit auto-resume
    across sizes (corrupted LoRA)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FT', 'zchar_ft', train_type='flux2klein')
        tag4 = lt._dest_base_tag(ds)                      # default variant -> 4B
        assert tag4 == '_FLUX2-Klein-4B'
        assert lt._run_name(ds).endswith('_FLUX2-Klein-4B')
        ds.train_variant = '9b'
        svc.db.session.commit()
        tag9 = lt._dest_base_tag(ds)
        assert tag9 == '_FLUX2-Klein-9B'
        assert tag4 != tag9
        # ... and both are distinct from a zimage official run (empty tag).
        ds.train_type = 'zimage'
        svc.db.session.commit()
        assert lt._dest_base_tag(ds) == ''


def test_default_and_valid_variants_for_flux2klein():
    """'4b' is the family default everywhere no variant is given; the accepted
    enum is per-family ('4b'/'9b' only — a leftover 'turbo'/'base' from another
    family must fall back to 4B, not leak into the config)."""
    from app.services import lora_training as lt
    assert lt._default_variant_for('flux2klein') == '4b'
    assert lt._valid_variants_for('flux2klein') == ('4b', '9b')
    # historical families keep their enum untouched
    assert lt._valid_variants_for('krea') == ('turbo', 'base', 'deturbo')
    assert lt._valid_variants_for(None) == ('turbo', 'base', 'deturbo')
    assert lt._default_variant_for('krea') == 'base'
    assert lt._default_variant_for('zimage') == 'turbo'


# --- 5) deploy routing ----------------------------------------------------------

def test_lora_dest_dir_routes_flux2klein(app, tmp_path):
    import os
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(tmp_path / 'comfy')}})
        ds = svc.create_dataset(LOCAL_USER, 'FD', 'zchar_fd', train_type='flux2klein')
        dest = lt._lora_dest_dir(ds)
        assert dest.replace('/', os.sep).endswith(
            os.sep.join(('models', 'loras', 'flux2klein')))
        # family override (UI selector) wins over the persisted type
        assert lt._lora_dest_dir(ds, family='krea').endswith('krea')


# --- 7) family badge/label classification ---------------------------------------

def test_family_of_lora_classifies_flux2klein_folder():
    from app.utils.comfyui import family_of_lora, FAMILY_LABELS
    assert family_of_lora(r'flux2klein\x.safetensors') == 'flux2klein'
    assert family_of_lora('flux2klein/x.safetensors') == 'flux2klein'
    # the 'flux' prefix match must not swallow the flux2klein folder (nor the
    # reverse): both classifications stay exact.
    assert family_of_lora(r'flux\x.safetensors') == 'flux'
    assert FAMILY_LABELS['flux2klein'] == 'FLUX.2 Klein'
