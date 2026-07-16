"""Custom weights paths for TRAINING (V1, local-only).

Covers the binding guardrails of the feature:
  a) per-family whitelist — VAE/TE are SDXL-only, refused (400) elsewhere;
  b) launch preflight — missing/corrupt file refused, arch sniff drives a
     CONFIRMABLE refusal (CUSTOM_WEIGHTS_UNVERIFIED:) exactly like UNCAPTIONED;
  c) _dest_base_tag encodes the full (weights, VAE, TE) triplet — two combos
     never share a run folder, the same combo is stable, and every official/
     whitelist run keeps its exact historical folder name;
  d) cloud training REFUSES a custom base (no silent fallback to the official one);
  e) provenance — the resolved paths reach launch_settings_snapshot and are
     redacted in the ⎘ Share config.

Plus: the per-family builders emit the right ai-toolkit config field for a
custom .safetensors (name_or_path override; SDXL vae_path/te_name_or_path).
"""
import json
import struct

import pytest


# --- fake .safetensors: 8-byte LE header length + JSON metadata (no real weights) ---

def _write_safetensors(path, keys):
    meta, off = {}, 0
    for k in keys:
        meta[k] = {'dtype': 'F32', 'shape': [1], 'data_offsets': [off, off + 4]}
        off += 4
    header = json.dumps(meta).encode('utf-8')
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(header)))
        fh.write(header)
        fh.write(b'\x00' * off)
    return str(path)


_SDXL_KEYS = ['model.diffusion_model.input_blocks.0.0.weight',
              'conditioner.embedders.0.transformer.text_model.x',
              'conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight',
              'first_stage_model.encoder.conv_in.weight']
_SD15_KEYS = ['model.diffusion_model.input_blocks.0.0.weight',
              'cond_stage_model.transformer.text_model.x',
              'first_stage_model.encoder.conv_in.weight']
_FLUX_KEYS = ['double_blocks.0.img_attn.qkv.weight', 'single_blocks.0.linear1.weight',
              'img_in.weight', 'txt_in.weight', 'final_layer.linear.weight']
_KREA_KEYS = ['first.weight', 'blocks.0.attn.qkv.weight', 'txtfusion.0.attn.qkv.weight',
              'tmlp.0.weight', 'last.linear.weight']


def _mkfile(tmp_path, name, keys):
    return _write_safetensors(tmp_path / name, keys)


def _configure_aitoolkit(tmp_path, app):
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


# --- arch detection from tensor NAMES ------------------------------------------

def test_detect_safetensors_arch_per_family():
    from app.services import lora_training as lt
    assert lt._detect_safetensors_arch(set(_SDXL_KEYS)) == 'sdxl'
    assert lt._detect_safetensors_arch(set(_SD15_KEYS)) == 'sd15'
    assert lt._detect_safetensors_arch(set(_FLUX_KEYS)) == 'flux'
    assert lt._detect_safetensors_arch(set(_KREA_KEYS)) == 'krea2'
    assert lt._detect_safetensors_arch({'foo.bar', 'baz.qux'}) is None


def test_is_custom_weights_only_absolute(tmp_path):
    from app.services import lora_training as lt
    assert lt._is_custom_weights(str(tmp_path / 'x.safetensors')) is True
    assert lt._is_custom_weights('some-checkpoint.safetensors') is False   # relative → whitelist name
    assert lt._is_custom_weights('Biglove/foo.safetensors') is False
    assert lt._is_custom_weights('') is False
    assert lt._is_custom_weights(None) is False


def test_safetensors_keys_reads_header(tmp_path):
    from app.services import lora_training as lt
    f = _mkfile(tmp_path, 'flux.safetensors', _FLUX_KEYS)
    assert lt._safetensors_tensor_keys(f) == set(_FLUX_KEYS)


def test_safetensors_keys_raises_on_corrupt(tmp_path):
    from app.services import lora_training as lt
    bad = tmp_path / 'bad.safetensors'
    bad.write_bytes(b'this is not a safetensors file at all')
    with pytest.raises(ValueError, match='not a readable'):
        lt._safetensors_tensor_keys(str(bad))


# --- preflight: hard failures vs confirmable ------------------------------------

def test_preflight_missing_file_hard_refuse(tmp_path):
    from app.services import lora_training as lt
    with pytest.raises(ValueError, match='not found'):
        lt.preflight_custom_paths('sdxl', weights=str(tmp_path / 'nope.safetensors'))


def test_preflight_corrupt_file_hard_refuse(tmp_path):
    from app.services import lora_training as lt
    bad = tmp_path / 'bad.safetensors'
    bad.write_bytes(b'\x00\x01\x02not-json')
    with pytest.raises(ValueError, match='not a readable'):
        lt.preflight_custom_paths('sdxl', weights=str(bad))


def test_preflight_matching_arch_passes(tmp_path):
    from app.services import lora_training as lt
    lt.preflight_custom_paths('sdxl', weights=_mkfile(tmp_path, 's.safetensors', _SDXL_KEYS))
    lt.preflight_custom_paths('krea', weights=_mkfile(tmp_path, 'k.safetensors', _KREA_KEYS))
    lt.preflight_custom_paths('flux', weights=_mkfile(tmp_path, 'f.safetensors', _FLUX_KEYS))
    # FLUX.2 Klein shares the flux DiT naming — same signature accepted.
    lt.preflight_custom_paths('flux2klein', weights=_mkfile(tmp_path, 'f2.safetensors', _FLUX_KEYS))


def test_preflight_wrong_arch_is_confirmable(tmp_path):
    from app.services import lora_training as lt
    sdxl_file = _mkfile(tmp_path, 's.safetensors', _SDXL_KEYS)
    with pytest.raises(ValueError, match='CUSTOM_WEIGHTS_UNVERIFIED') as e:
        lt.preflight_custom_paths('krea', weights=sdxl_file)
    assert 'SDXL' in str(e.value)          # names the detected arch
    # confirm → retry with allow_unverified_weights clears it
    lt.preflight_custom_paths('krea', weights=sdxl_file, allow_unverified_weights=True)


def test_preflight_undetectable_arch_is_confirmable(tmp_path):
    from app.services import lora_training as lt
    blob = _mkfile(tmp_path, 'mystery.safetensors', ['foo.bar', 'baz.qux'])
    with pytest.raises(ValueError, match='CUSTOM_WEIGHTS_UNVERIFIED'):
        lt.preflight_custom_paths('flux', weights=blob)
    lt.preflight_custom_paths('flux', weights=blob, allow_unverified_weights=True)


def test_preflight_vae_and_te(tmp_path):
    from app.services import lora_training as lt
    # missing VAE file → hard refuse
    with pytest.raises(ValueError, match='VAE file not found'):
        lt.preflight_custom_paths('sdxl', vae_path=str(tmp_path / 'missing-vae.safetensors'))
    # present VAE .safetensors → header parsed, passes
    lt.preflight_custom_paths('sdxl', vae_path=_mkfile(tmp_path, 'vae.safetensors', _SDXL_KEYS))
    # a bare HF repo id for the TE is accepted (unverifiable locally)
    lt.preflight_custom_paths('sdxl', te_path='stabilityai/some-text-encoder')
    # a LOCAL te path that doesn't exist → hard refuse
    with pytest.raises(ValueError, match='text-encoder path not found'):
        lt.preflight_custom_paths('sdxl', te_path=str(tmp_path / 'no' / 'such' / 'te'))


# --- guardrail (a): per-family whitelist for VAE/TE -----------------------------

def test_vae_te_refused_off_sdxl_at_launch(app, tmp_path, monkeypatch):
    """VAE/TE overrides are SDXL-only; providing them for another family is a
    400 (explicit refusal, never a silent ignore). Uses flux (a core arch — no
    support-guard fires before the whitelist)."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    monkeypatch.setattr(lt.shutil, 'disk_usage', lambda p: type('u', (), {'free': 500e9})())
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FX', 'zchar_fx', train_type='flux')
        with pytest.raises(ValueError, match='SDXL-only'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, vae_path='C:\\x\\vae.safetensors')
        with pytest.raises(ValueError, match='SDXL-only'):
            lt.launch_training(LOCAL_USER, ds.id, check_captions=False, te_path='C:\\x\\te')


def test_enqueue_rejects_vae_off_sdxl(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    _configure_aitoolkit(tmp_path, app)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'FXQ', 'zchar_fxq', train_type='flux')
        # extra_steps set → skips assert_trainable; the whitelist still fires.
        with pytest.raises(ValueError, match='SDXL-only'):
            lt.enqueue_training(LOCAL_USER, ds.id, extra_steps=100, vae_path='C:\\x\\vae.safetensors')


# --- guardrail (c): the run-dir tag encodes the whole triplet -------------------

def test_dest_base_tag_encodes_triplet(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    w1 = _mkfile(tmp_path, 'sdxlA.safetensors', _SDXL_KEYS)
    w2 = _mkfile(tmp_path, 'sdxlB.safetensors', _SDXL_KEYS)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'CT', 'zchar_ct', train_type='sdxl')
        ds.train_base_model = w1
        svc.db.session.commit()
        tag_w1 = lt._dest_base_tag(ds)
        # same combo → identical tag (stable auto-resume)
        assert lt._dest_base_tag(ds) == tag_w1
        # different weights → different tag
        ds.train_base_model = w2
        svc.db.session.commit()
        assert lt._dest_base_tag(ds) != tag_w1
        # add a VAE override → tag changes again (encodes VAE)
        ds.train_base_model = w1
        ds.train_vae_path = _mkfile(tmp_path, 'vae.safetensors', _SDXL_KEYS)
        svc.db.session.commit()
        tag_w1_vae = lt._dest_base_tag(ds)
        assert tag_w1_vae != tag_w1
        # add a TE override → tag changes again (encodes TE)
        ds.train_te_path = 'org/te'
        svc.db.session.commit()
        assert lt._dest_base_tag(ds) != tag_w1_vae


def test_dest_base_tag_official_runs_unchanged(app, tmp_path):
    """Official families have stable distinct tags; Z-Image Turbo deliberately
    no longer reuses the historical suffix-less folder."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        z = svc.create_dataset(LOCAL_USER, 'Z', 'zc_z', train_type='zimage')
        assert lt._dest_base_tag(z) == '_Z-Image-Turbo'
        k = svc.create_dataset(LOCAL_USER, 'K', 'zc_k', train_type='krea')
        assert lt._dest_base_tag(k) == '_Krea-2-Raw'      # unchanged constant tag
        fk = svc.create_dataset(LOCAL_USER, 'FK', 'zc_fk', train_type='flux2klein')
        assert lt._dest_base_tag(fk) == '_FLUX2-Klein-4B'


# --- builders emit the right ai-toolkit config field ----------------------------

def _build(app, tmp_path, family, **cols):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, family, f'zc_{family}', train_type=family)
        for k, v in cols.items():
            setattr(ds, k, v)
        svc.db.session.commit()
        folder = tmp_path / f'ds_{family}'
        folder.mkdir(exist_ok=True)
        return lt.build_job_config(ds, str(folder), steps=1500)['config']['process'][0]


def test_builders_emit_custom_name_or_path(app, tmp_path):
    weights = _mkfile(tmp_path, 'w.safetensors', _FLUX_KEYS)
    for fam in ('krea', 'flux', 'flux2klein'):
        p = _build(app, tmp_path, fam, train_base_model=weights)
        assert p['model']['name_or_path'] == weights, fam
        # TE/VAE remain official for these families (no override keys emitted).
        assert 'vae_path' not in p['model'] and 'te_name_or_path' not in p['model']


def test_sdxl_builder_emits_custom_weights_and_overrides(app, tmp_path):
    weights = _mkfile(tmp_path, 'sdxl.safetensors', _SDXL_KEYS)
    vae = _mkfile(tmp_path, 'vae.safetensors', _SDXL_KEYS)
    p = _build(app, tmp_path, 'sdxl', train_base_model=weights,
               train_vae_path=vae, train_te_path='org/te-repo')
    assert p['model']['name_or_path'] == weights
    assert p['model']['vae_path'] == vae
    assert p['model']['te_name_or_path'] == 'org/te-repo'


# --- guardrail (d): cloud refuses a persisted custom base -----------------------

def test_cloud_refuses_persisted_custom_weights(app, tmp_path, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    monkeypatch.setattr(ct, '_reconcile_before_launch', lambda a: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'CC', 'zc_cc', train_type='krea')
        ds.train_base_model = _mkfile(tmp_path, 'w.safetensors', _KREA_KEYS)
        svc.db.session.commit()
        with pytest.raises(ValueError, match='local-only'):
            ct.launch_cloud_training(LOCAL_USER, ds.id)


def test_cloud_refuses_persisted_sdxl_override(app, tmp_path, monkeypatch):
    from app.services import cloud_training as ct
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    monkeypatch.setattr(ct, '_reconcile_before_launch', lambda a: None)
    with app.app_context():
        # zimage is a cloud-supported family; a stray VAE override must still block.
        ds = svc.create_dataset(LOCAL_USER, 'CV', 'zc_cv', train_type='zimage')
        ds.train_vae_path = str(tmp_path / 'vae.safetensors')
        svc.db.session.commit()
        with pytest.raises(ValueError, match='local-only'):
            ct.launch_cloud_training(LOCAL_USER, ds.id)


def test_continue_bypasses_confirmable_preflight(app, tmp_path, monkeypatch):
    """Resuming a run whose custom base was accepted once must not re-hit the
    confirmable sniff (this path can't answer the confirm) — continue_training
    passes allow_unverified_weights=True while keeping the vae/te triplet."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    captured = {}

    def fake_launch(user_id, dataset_id, **kw):
        captured.update(kw)
        return {'started': True, 'pid': 1}

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'CN', 'zc_cn', train_type='flux')
        ds.train_base_model = _mkfile(tmp_path, 'w.safetensors', _FLUX_KEYS)
        svc.db.session.commit()
        monkeypatch.setattr(lt, 'list_checkpoints', lambda *a, **k: [{'step': 800}])
        monkeypatch.setattr(lt, 'launch_training', fake_launch)
        lt.continue_training(LOCAL_USER, ds.id, extra_steps=500)
    assert captured['allow_unverified_weights'] is True
    # vae/te not forwarded → launch_training keeps its _PERSISTED default (the run's triplet).
    assert 'vae_path' not in captured and 'te_path' not in captured


# --- guardrail (e): provenance snapshot + redacted share ------------------------

def test_snapshot_carries_custom_paths(app, tmp_path):
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    weights = _mkfile(tmp_path, 'w.safetensors', _SDXL_KEYS)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'SN', 'zc_sn', train_type='sdxl')
        ds.train_base_model = weights
        ds.train_vae_path = str(tmp_path / 'vae.safetensors')
        ds.train_te_path = 'org/te'
        svc.db.session.commit()
        snap = lt.launch_settings_snapshot(ds)
        assert snap['base_weights'] == weights
        assert snap['vae_path'] == str(tmp_path / 'vae.safetensors')
        assert snap['te_name_or_path'] == 'org/te'


def test_share_config_redacts_custom_paths(client, app):
    from app.extensions import db
    from app.models import TrainingRunRecord
    ds = client.post('/api/dataset/create',
                     json={'name': 'Sh', 'trigger_word': 'sh'}).get_json()['id']
    home = 'C:\\Users\\secretuser\\models\\my-sdxl.safetensors'
    with app.app_context():
        rec = TrainingRunRecord(
            dataset_id=ds, family='sdxl', source='local', fingerprint='fp',
            version=1, steps=2000, masked=True,
            settings=json.dumps({'trigger': 'sh', 'rank': 32,
                                 'base_weights': home,
                                 'vae_path': 'C:\\Users\\secretuser\\vae.safetensors'}),
            manifest=json.dumps([[1, 'a', 'b']]), base_model=home)
        db.session.add(rec)
        db.session.commit()
        rid = rec.id
    body = client.get(f'/api/dataset/train/runs/rec-{rid}/share').get_data(as_text=True)
    # the home-dir prefix is stripped to ~; the OS account never leaks…
    assert 'secretuser' not in body
    # …but the recognizable tail of the path is kept (paste-safe, still useful).
    assert 'my-sdxl.safetensors' in body
    assert 'vae.safetensors' in body
