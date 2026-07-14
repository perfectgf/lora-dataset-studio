"""The LoRA-architecture guardrail: read a trained LoRA's REAL family from its
safetensors header (metadata first, tensor-name sniff second) and refuse to
deploy / test it under a contradicting family.

Root incident (2026-07-13): a Z-Image LoRA mislabelled 'krea' (pre-6952b11
wrong-arch bug) was deployed under loras/krea/ and tested in the Krea Studio;
ComfyUI silently dropped every incompatible key, so 117 tiles rendered with the
LoRA effectively OFF and no error anywhere. These tests pin the detector, the
deploy refusal, the Studio 409, and the cache.
"""
import json
import os
import struct

import pytest

from app.config import LOCAL_USER


# --- Synthetic safetensors headers -------------------------------------------
def _write_st(path, keys, metadata=None):
    """Write a minimal but VALID .safetensors: 8-byte LE header length + JSON
    header (each key a tiny F32 tensor) + a trivial data block. Enough for the
    header-only reader; `metadata` lands in the __metadata__ block."""
    header = {}
    if metadata:
        header['__metadata__'] = metadata
    off = 0
    for k in keys:
        header[k] = {'dtype': 'F32', 'shape': [1], 'data_offsets': [off, off + 4]}
        off += 4
    blob = json.dumps(header).encode('utf-8')
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(blob)))
        fh.write(blob)
        fh.write(b'\x00' * off)
    return str(path)


# Tensor-name signatures verified against real deployed LoRAs (see the agent's
# header dump: A:\ComfyUI\models\loras\*).
_ZIMAGE_KEYS = ['diffusion_model.layers.0.attention.to_k.lora_A.weight',
                'diffusion_model.layers.0.attention.to_k.lora_B.weight',
                'diffusion_model.layers.0.adaLN_modulation.0.lora_A.weight']
_KREA_KEYS = ['diffusion_model.blocks.0.attn.wk.lora_A.weight',
              'diffusion_model.blocks.0.attn.wk.lora_B.weight',
              'diffusion_model.txtfusion.lora_A.weight']
_SDXL_KEYS = ['lora_unet_down_blocks_1_attentions_0_proj_in.lora_down.weight',
              'lora_unet_down_blocks_1_attentions_0_proj_in.lora_up.weight',
              'lora_unet_down_blocks_1_attentions_0_proj_in.alpha']
_FLUX_KEYS = ['diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight',
              'diffusion_model.single_blocks.0.linear1.lora_A.weight']
_UNKNOWN_KEYS = ['some.random.tensor.weight', 'foo.bar.baz.qux']


# --- Detector: metadata path --------------------------------------------------
@pytest.mark.parametrize('version,expected', [
    ('zimage', 'zimage'),
    ('krea2', 'krea'),
    ('sdxl_1.0', 'sdxl'),
    ('flux', 'flux'),
    ('flux2_klein_9b', 'flux2klein'),
    ('flux2_klein_4b', 'flux2klein'),
    ('sd_1.5', None),        # a family we don't train → no verdict (no false block)
    ('sd_2.1', None),
    ('totally-unknown', None),
])
def test_detect_by_metadata(app, tmp_path, version, expected):
    from app.services import lora_training as lt
    p = _write_st(tmp_path / f'{version}.safetensors', _UNKNOWN_KEYS,
                  {'ss_base_model_version': version})
    assert lt.detect_lora_arch(p) == expected


# --- Detector: tensor-name fallback (no usable metadata) ----------------------
@pytest.mark.parametrize('keys,expected', [
    (_ZIMAGE_KEYS, 'zimage'),
    (_KREA_KEYS, 'krea'),
    (_SDXL_KEYS, 'sdxl'),
    (_FLUX_KEYS, 'flux'),            # names can't split FLUX.1 vs FLUX.2 → generic
    (_UNKNOWN_KEYS, None),
])
def test_detect_by_tensor_names_when_metadata_absent(app, tmp_path, keys, expected):
    from app.services import lora_training as lt
    p = _write_st(tmp_path / 'nometa.safetensors', keys)   # NO __metadata__
    assert lt.detect_lora_arch(p) == expected


def test_metadata_wins_over_tensor_names(app, tmp_path):
    """ai-toolkit's stamp is authoritative: a header whose metadata says zimage
    but whose tensors look krea must resolve to zimage."""
    from app.services import lora_training as lt
    p = _write_st(tmp_path / 'mixed.safetensors', _KREA_KEYS,
                  {'ss_base_model_version': 'zimage'})
    assert lt.detect_lora_arch(p) == 'zimage'


def test_undetectable_and_unreadable_return_none(app, tmp_path):
    from app.services import lora_training as lt
    assert lt.detect_lora_arch(str(tmp_path / 'missing.safetensors')) is None   # no file
    garbage = tmp_path / 'garbage.safetensors'
    garbage.write_bytes(b'not a safetensors at all')
    assert lt.detect_lora_arch(str(garbage)) is None                            # bad header
    # foreign metadata value + unrecognized tensor names → None (not a false hit)
    p = _write_st(tmp_path / 'foreign.safetensors', _UNKNOWN_KEYS,
                  {'ss_base_model_version': 'wan21'})
    assert lt.detect_lora_arch(p) is None


# --- Conflict semantics -------------------------------------------------------
def test_lora_arch_conflicts_semantics(app):
    from app.services import lora_training as lt
    # cross-namespace = SILENT drop = conflict
    assert lt.lora_arch_conflicts('zimage', 'krea') is True
    assert lt.lora_arch_conflicts('sdxl', 'zimage') is True
    assert lt.lora_arch_conflicts('krea', 'flux') is True
    # same family = fine
    assert lt.lora_arch_conflicts('krea', 'krea') is False
    # flux <-> flux2klein share the tensor namespace (a version mismatch fails
    # LOUDLY on shape, not silently) → deliberately NOT a conflict
    assert lt.lora_arch_conflicts('flux', 'flux2klein') is False
    assert lt.lora_arch_conflicts('flux2klein', 'flux') is False
    # undetectable / unknown either side → never block
    assert lt.lora_arch_conflicts(None, 'krea') is False
    assert lt.lora_arch_conflicts('krea', None) is False
    assert lt.lora_arch_conflicts('krea', 'something-weird') is False


# --- Cache: keyed on (path, mtime, size) --------------------------------------
def test_cache_hits_and_invalidates(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    lt._LORA_ARCH_CACHE.clear()
    p = _write_st(tmp_path / 'c.safetensors', _KREA_KEYS,
                  {'ss_base_model_version': 'krea2'})
    assert lt.detect_lora_arch(p) == 'krea'
    # A second call must NOT re-read the header: prove it by making the reader blow
    # up — the cached verdict for the unchanged (path,mtime,size) is returned.
    def _boom(_):
        raise AssertionError('header should not be read again (cache miss)')
    monkeypatch.setattr(lt, '_read_safetensors_header', _boom)
    assert lt.detect_lora_arch(p) == 'krea'
    # Rewriting the file (new size/mtime) invalidates the entry → re-read happens
    # and reflects the NEW content.
    monkeypatch.undo()
    _write_st(p, _ZIMAGE_KEYS, {'ss_base_model_version': 'zimage'})
    assert lt.detect_lora_arch(p) == 'zimage'


# --- Deploy guard: import_checkpoint -----------------------------------------
def test_import_refuses_arch_mismatch(app, tmp_path):
    """A Z-Image LoRA cannot be deployed under the Krea family (the exact incident)."""
    from app.services import lora_training as lt, face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Kit', 'kitty')
        run_dir = tmp_path / 'run'
        run_dir.mkdir()
        f = run_dir / 'lora_kitty_000002000.safetensors'
        _write_st(f, _ZIMAGE_KEYS, {'ss_base_model_version': 'zimage'})
        with pytest.raises(ValueError, match='Z-Image'):
            lt.import_checkpoint(LOCAL_USER, ds.id, f.name, family='krea',
                                 src_dir=str(run_dir))
        # nothing was copied out (the refusal is BEFORE the deploy)


def test_import_accepts_matching_and_undetectable(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt, face_dataset_service as svc
    from app import config
    with app.app_context():
        base = tmp_path / 'Comfy'
        (base / 'models' / 'loras').mkdir(parents=True)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        ds = svc.create_dataset(LOCAL_USER, 'Kit', 'kitty')
        run_dir = tmp_path / 'run'
        run_dir.mkdir()
        # matching Krea LoRA → deploys
        good = run_dir / 'lora_kitty_000002000.safetensors'
        _write_st(good, _KREA_KEYS, {'ss_base_model_version': 'krea2'})
        dest = lt.import_checkpoint(LOCAL_USER, ds.id, good.name, family='krea',
                                    src_dir=str(run_dir))
        assert os.path.isfile(dest)
        # undetectable header (no metadata, foreign names) → NOT blocked
        unk = run_dir / 'lora_kitty_000003000.safetensors'
        _write_st(unk, _UNKNOWN_KEYS)
        dest2 = lt.import_checkpoint(LOCAL_USER, ds.id, unk.name, family='krea',
                                     src_dir=str(run_dir))
        assert os.path.isfile(dest2)


# --- Studio guard: preflight 409-style on mismatch ---------------------------
def _configure_comfy(tmp_path):
    from app import config
    base = tmp_path / 'Comfy'
    (base / 'models').mkdir(parents=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def test_create_run_refuses_wrong_arch_checkpoint(app, tmp_path, monkeypatch):
    """A Z-Image LoRA sitting in loras/krea/ (mislabelled deploy) selected in the
    Krea Studio → StudioArchMismatch BEFORE any row is created (no grid of no-op
    tiles). family_of_lora keys off the folder; the header sniff catches it."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    from app.models import LoraTestImage
    with app.app_context():
        base = _configure_comfy(tmp_path)
        krea_dir = base / 'models' / 'loras' / 'krea'
        krea_dir.mkdir(parents=True)
        _write_st(krea_dir / 'lora_kitty_000002000.safetensors', _ZIMAGE_KEYS,
                  {'ss_base_model_version': 'zimage'})
        ck = 'krea\\lora_kitty_000002000.safetensors'
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        ds = svc.create_dataset(LOCAL_USER, 'Kit', 'kitty')
        with pytest.raises(lts.StudioArchMismatch) as ei:
            lts.create_run(LOCAL_USER, ds.id, [ck], [1.0], family='krea',
                           prompt='p', count=1)
        assert ei.value.detected == 'zimage' and ei.value.family == 'krea'
        assert LoraTestImage.query.filter_by(dataset_id=ds.id).count() == 0


def test_create_run_allows_matching_checkpoint_arch(app, tmp_path, monkeypatch):
    """A genuine Krea LoRA in loras/krea/ passes the arch guard (it then proceeds
    to the asset preflight — stubbed absent here to stop before ComfyUI)."""
    from app.services import lora_test_studio as lts, face_dataset_service as svc
    with app.app_context():
        base = _configure_comfy(tmp_path)
        krea_dir = base / 'models' / 'loras' / 'krea'
        krea_dir.mkdir(parents=True)
        _write_st(krea_dir / 'lora_kitty_000002000.safetensors', _KREA_KEYS,
                  {'ss_base_model_version': 'krea2'})
        ck = 'krea\\lora_kitty_000002000.safetensors'
        monkeypatch.setattr(lts, 'gpu_busy_reason', lambda: None)
        # The arch guard must NOT fire; a later StudioAssetsMissing (real assets
        # absent) is the acceptable stop — anything but StudioArchMismatch proves
        # the arch check passed.
        ds = svc.create_dataset(LOCAL_USER, 'Kit', 'kitty')
        with pytest.raises(Exception) as ei:
            lts.create_run(LOCAL_USER, ds.id, [ck], [1.0], family='krea',
                           prompt='p', count=1)
        assert not isinstance(ei.value, lts.StudioArchMismatch)


def test_arch_mismatch_route_maps_to_409(app, client, monkeypatch):
    """The lora-test/run route turns StudioArchMismatch into a structured 409 with
    the studio_arch_mismatch payload (same contract as studio_missing)."""
    from app.services import face_dataset_service as svc
    from app.services import lora_test_studio as lts
    from app.routes import _common
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Kit', 'kitty')
        dsid = ds.id
    monkeypatch.setattr(_common.capabilities, 'probe',
                        lambda *a, **k: {'comfyui': {'reachable': True}})

    def _raise(*a, **k):
        raise lts.StudioArchMismatch('krea', 'zimage', 'krea\\lora_kitty_000002000.safetensors')
    monkeypatch.setattr(lts, 'create_run', _raise)
    r = client.post(f'/api/dataset/{dsid}/lora-test/run',
                    json={'checkpoints': ['krea\\lora_kitty_000002000.safetensors'],
                          'strengths': [1.0], 'family': 'krea'})
    assert r.status_code == 409
    body = r.get_json()
    assert body['studio_arch_mismatch']['detected'] == 'zimage'
    assert body['studio_arch_mismatch']['family'] == 'krea'
    assert 'Z-Image' in body['error']


# --- Listing retrofit badge --------------------------------------------------
def test_imported_listing_flags_mislabelled_deploy(app, tmp_path):
    """A Z-Image file already sitting in loras/krea/ is surfaced with an
    arch_mismatch badge (retrofit signal — no file is moved)."""
    from app.services import lora_training as lt, face_dataset_service as svc
    from app import config
    with app.app_context():
        base = tmp_path / 'Comfy'
        krea_dir = base / 'models' / 'loras' / 'krea'
        krea_dir.mkdir(parents=True)
        config.save_config({'comfyui': {'base_dir': str(base)}})
        _write_st(krea_dir / 'lora_kitty_000002000.safetensors', _ZIMAGE_KEYS,
                  {'ss_base_model_version': 'zimage'})
        ds = svc.create_dataset(LOCAL_USER, 'Kit', 'kitty')
        rows = lt.list_imported_checkpoints(LOCAL_USER, ds.id, family='krea')
        assert len(rows) == 1
        assert rows[0]['arch_mismatch'] == 'zimage'
        assert rows[0]['arch_label'] == 'Z-Image'
