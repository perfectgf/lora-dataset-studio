"""The Klein generation-LoRA picker scan (klein_lora_picker + the settings route).

Pins the contract the free-text -> picker upgrade rests on:
  * the scan walks EVERY loras search root recursively (base models/loras + every
    extra_model_paths.yaml root) — LoRAs aren't sorted into a klein/ subfolder;
  * each emitted `name` is the EXACT ComfyUI-relative value the resolver
    (klein_edit_helper._lora_abs) turns back into the real file — picker == resolver;
  * each LoRA is arch-badged from its own header (Klein-compatible flux/flux2klein
    vs a silently-dropping SDXL/Krea/Z-Image vs undetectable), compatible-first;
  * the mtime cache serves repeats and `force` bypasses it;
  * the route degrades to {loras: []} (free-text fallback) when ComfyUI is
    unconfigured or the scan fails — never a blocking empty dropdown.
"""
import json
import os
import struct

import pytest


# --- Synthetic safetensors headers (same shape as test_lora_arch_detection) ---
def _write_st(path, keys, metadata=None):
    """A minimal but VALID .safetensors: 8-byte LE header length + JSON header +
    a trivial data block. Enough for detect_lora_arch's header-only read."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
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


_FLUX2KLEIN_META = {'ss_base_model_version': 'flux2_klein_9b'}
_FLUX_META = {'ss_base_model_version': 'flux'}
_SDXL_META = {'ss_base_model_version': 'sdxl_1.0'}
_FLUX_KEYS = ['diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight',
              'diffusion_model.single_blocks.0.linear1.lora_A.weight']
_UNKNOWN_KEYS = ['some.random.tensor.weight', 'foo.bar.baz.qux']


@pytest.fixture(autouse=True)
def _clear_caches():
    from app.services import comfy_model_paths as cmp
    from app.services import klein_lora_picker as klp
    from app.services import lora_training as lt
    cmp.clear_cache(); cmp._warned.clear()
    klp.clear_cache()
    lt._LORA_ARCH_CACHE.clear()
    yield
    cmp.clear_cache(); cmp._warned.clear()
    klp.clear_cache()
    lt._LORA_ARCH_CACHE.clear()


def _comfy_base(tmp_path, cfg):
    base = tmp_path / 'ComfyUI'
    (base / 'models' / 'loras').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def _loras(base):
    return base / 'models' / 'loras'


# --- Recursive scan across base + extra roots, relative loader names ----------
def test_scan_is_recursive_across_base_and_extra_roots(app, tmp_path):
    from app import config as cfg
    from app.services import klein_lora_picker as klp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        # base loras: a nested subfolder AND a root-level file (no klein/ convention).
        _write_st(_loras(base) / 'klein' / 'style_a.safetensors', _UNKNOWN_KEYS, _FLUX2KLEIN_META)
        _write_st(_loras(base) / 'root_level.safetensors', _FLUX_KEYS)
        # an extra loras root registered via extra_model_paths.yaml
        ext = tmp_path / 'external' / 'loras'
        _write_st(ext / 'deep' / 'nested' / 'style_b.safetensors', _UNKNOWN_KEYS, _SDXL_META)
        (base / 'extra_model_paths.yaml').write_text(
            f"comfyui:\n  loras: {ext}\n", encoding='utf-8')

        names = {e['name'] for e in klp.scan_generation_loras()}
        assert os.path.join('klein', 'style_a.safetensors') in names        # base subfolder
        assert 'root_level.safetensors' in names                            # base root
        assert os.path.join('deep', 'nested', 'style_b.safetensors') in names  # extra, deep


# --- picker == resolver: every emitted name resolves to the real file ---------
def test_emitted_names_resolve_via_klein_edit_helper(app, tmp_path):
    """The value the picker emits MUST be exactly what klein_edit_helper resolves
    back to an absolute path at generate time (else the LoRA silently no-ops)."""
    from app import config as cfg
    from app.services import klein_lora_picker as klp
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_st(_loras(base) / 'klein' / 'style_a.safetensors', _FLUX_KEYS)
        ext = tmp_path / 'external' / 'loras'
        _write_st(ext / 'sub' / 'style_b.safetensors', _FLUX_KEYS)
        (base / 'extra_model_paths.yaml').write_text(
            f"comfyui:\n  loras: {ext}\n", encoding='utf-8')
        for entry in klp.scan_generation_loras():
            resolved = keh._lora_abs(entry['name'])
            assert resolved and os.path.isfile(resolved), \
                f"picker name {entry['name']!r} did not resolve"


# --- Architecture badge + compatible-first ordering --------------------------
def test_arch_badge_and_grouping(app, tmp_path):
    from app import config as cfg
    from app.services import klein_lora_picker as klp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_st(_loras(base) / 'a_klein.safetensors', _UNKNOWN_KEYS, _FLUX2KLEIN_META)
        _write_st(_loras(base) / 'b_flux.safetensors', _UNKNOWN_KEYS, _FLUX_META)
        _write_st(_loras(base) / 'c_sdxl.safetensors', _UNKNOWN_KEYS, _SDXL_META)
        _write_st(_loras(base) / 'd_unknown.safetensors', _UNKNOWN_KEYS)   # no metadata, foreign names

        loras = klp.scan_generation_loras()
        by_name = {e['name']: e for e in loras}
        assert by_name['a_klein.safetensors'] == {
            'name': 'a_klein.safetensors', 'arch': 'flux2klein',
            'label': 'FLUX.2 Klein', 'compatible': 'yes'}
        assert by_name['b_flux.safetensors']['compatible'] == 'yes'       # shares Klein namespace
        assert by_name['b_flux.safetensors']['label'] == 'FLUX.1'
        assert by_name['c_sdxl.safetensors']['compatible'] == 'no'        # silent no-op in Klein graph
        assert by_name['c_sdxl.safetensors']['label'] == 'SDXL'
        assert by_name['d_unknown.safetensors'] == {
            'name': 'd_unknown.safetensors', 'arch': None,
            'label': None, 'compatible': 'unknown'}
        # compatible-first ordering: the two 'yes' come before 'unknown'/'no'.
        compat_seq = [e['compatible'] for e in loras]
        assert compat_seq == sorted(compat_seq, key={'yes': 0, 'unknown': 1, 'no': 2}.get)
        assert compat_seq[0] == 'yes' and compat_seq[-1] == 'no'


# --- Cache: hit, force bypass, mtime invalidation ----------------------------
def test_cache_hit_and_force_bypass(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import klein_lora_picker as klp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_st(_loras(base) / 'x.safetensors', _FLUX_KEYS)
        first = klp.scan_generation_loras()
        assert len(first) == 1

        # A second call must NOT re-walk: make the walker blow up and prove the
        # cached list comes back unchanged.
        def _boom(*a, **k):
            raise AssertionError('list_models should not be called on a cache hit')
        monkeypatch.setattr(klp.comfy_model_paths, 'list_models', _boom)
        assert klp.scan_generation_loras() == first
        # force=True bypasses the cache -> it DOES call the (now exploding) walker.
        with pytest.raises(AssertionError):
            klp.scan_generation_loras(force=True)


def test_cache_invalidates_on_root_mtime(app, tmp_path):
    from app import config as cfg
    from app.services import klein_lora_picker as klp
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_st(_loras(base) / 'x.safetensors', _FLUX_KEYS)
        assert len(klp.scan_generation_loras()) == 1
        # Add a second LoRA at the root and bump the root's mtime so the signature
        # changes (deterministic across filesystems) -> the scan re-runs.
        _write_st(_loras(base) / 'y.safetensors', _FLUX_KEYS)
        st = os.stat(str(_loras(base)))
        os.utime(str(_loras(base)), (st.st_atime + 5, st.st_mtime + 5))
        assert len(klp.scan_generation_loras()) == 2


def test_scan_empty_when_unconfigured(app, tmp_path):
    from app.services import klein_lora_picker as klp
    with app.app_context():
        # No comfyui.base_dir -> no loras root -> empty list, never an error.
        assert klp.scan_generation_loras() == []


# --- Route -------------------------------------------------------------------
def test_route_returns_loras(app, client, tmp_path):
    from app import config as cfg
    with app.app_context():
        base = _comfy_base(tmp_path, cfg)
        _write_st(_loras(base) / 'klein' / 'style_a.safetensors', _UNKNOWN_KEYS, _FLUX2KLEIN_META)
    r = client.get('/api/loras/list')
    assert r.status_code == 200
    loras = r.get_json()['loras']
    assert any(e['name'] == os.path.join('klein', 'style_a.safetensors')
               and e['compatible'] == 'yes' for e in loras)
    # ?force=1 is accepted and returns the same shape.
    r2 = client.get('/api/loras/list?force=1')
    assert r2.status_code == 200 and isinstance(r2.get_json()['loras'], list)


def test_route_degrades_gracefully_when_unconfigured(app, client):
    # ComfyUI unconfigured -> {loras: []}, HTTP 200 (the picker falls back to free text).
    r = client.get('/api/loras/list')
    assert r.status_code == 200
    assert r.get_json() == {'loras': []}


def test_route_never_500s_on_scan_failure(app, client, monkeypatch):
    """A scan blowing up degrades to {loras: []}, never a 500 the picker can't use."""
    from app.services import klein_lora_picker as klp
    monkeypatch.setattr(klp, 'scan_generation_loras',
                        lambda **k: (_ for _ in ()).throw(RuntimeError('boom')))
    r = client.get('/api/loras/list')
    assert r.status_code == 200 and r.get_json() == {'loras': []}
